#!/usr/bin/env python3
"""
测试实验脚本 — 一键运行全部 9 组实验（采样数据）

使用方式:
    python run_test.py                        # 运行全部 9 组
    python run_test.py --experiments E0 E1    # 只运行指定实验
    python run_test.py --seed 123             # 自定义随机种子
    python run_test.py --n_ag 3 --n_hc 3     # 自定义采样数量
    python run_test.py --resample             # 强制重新采样数据
    python run_test.py --run_id my_test       # 指定 run ID（默认自动时间戳）

每次运行自动创建独立目录，互不覆盖:
    test_experiments/
    ├── run_test.py
    ├── sampled_data/              # 采样数据（跨 run 共享，同 seed 同数据）
    │   ├── AG/
    │   └── HC/
    └── runs/
        ├── run_0327_1252/         # 第一次运行
        │   ├── results/
        │   │   ├── E0.json ... B6.json
        │   │   ├── summary.md
        │   │   └── figures/       # 自动生成的 6 张图表
        │   └── logs/
        │       ├── E0/            # hprompt input/output
        │       ├── E2/            # 含 intent + localize 日志
        │       └── B1/            # baseline 推理输出 txt
        └── run_0327_1430/         # 第二次运行（完全独立）
            ├── results/
            └── logs/
"""

import os
import sys
import json
import time
import random
import shutil
import asyncio
import argparse
import contextlib
import io
from pathlib import Path
from datetime import datetime
from tqdm import tqdm  # only used by baseline's internal Lib/utils

SCRIPT_DIR = Path(__file__).parent.resolve()
AUTOMATED_FA_DIR = SCRIPT_DIR.parent
REPO_ROOT = AUTOMATED_FA_DIR.parent
FEEDBACK_CODE = REPO_ROOT / "feedback" / "feedback"

AG_SOURCE = REPO_ROOT / "Who_and_When" / "Algorithm-Generated"
HC_SOURCE = REPO_ROOT / "Who_and_When" / "Hand-Crafted"

sys.path.insert(0, str(AUTOMATED_FA_DIR))
sys.path.insert(0, str(FEEDBACK_CODE))

ALL_EXPERIMENTS = ["E1", "E2", "B1", "B2", "B3", "B4", "B5", "B6"]

EXPERIMENT_CONFIG = {
    "E1": {"type": "feedback", "feedback_level": "task_aware",  "bypass_intent": True},
    "E2": {"type": "feedback", "feedback_level": "blind",       "bypass_intent": True},
    "B1": {"type": "baseline", "method": "all_at_once",  "no_gt": False},
    "B2": {"type": "baseline", "method": "step_by_step", "no_gt": False},
    "B3": {"type": "baseline", "method": "binary_search","no_gt": False},
    "B4": {"type": "baseline", "method": "all_at_once",  "no_gt": True},
    "B5": {"type": "baseline", "method": "step_by_step", "no_gt": True},
    "B6": {"type": "baseline", "method": "binary_search","no_gt": True},
}


def sample_data(n_ag=5, n_hc=5, seed=42, force=False, run_dir=None):
    base = Path(run_dir) if run_dir else SCRIPT_DIR / "sampled_data"
    ag_out = base / "sampled_data" / "AG" if run_dir else SCRIPT_DIR / "sampled_data" / "AG"
    hc_out = base / "sampled_data" / "HC" if run_dir else SCRIPT_DIR / "sampled_data" / "HC"

    need_resample = force
    if not need_resample:
        if not ag_out.exists() or not hc_out.exists():
            need_resample = True
        else:
            existing_ag = sorted(ag_out.glob("*.json"))
            existing_hc = sorted(hc_out.glob("*.json"))
            if len(existing_ag) != n_ag or len(existing_hc) != n_hc:
                print(f"  采样数量不匹配 (现有 AG={len(existing_ag)}/HC={len(existing_hc)}, 需要 AG={n_ag}/HC={n_hc})，重新采样")
                need_resample = True

    if not need_resample:
        existing_ag = sorted(ag_out.glob("*.json"))
        existing_hc = sorted(hc_out.glob("*.json"))
        print(f"  已有采样数据 (AG={len(existing_ag)}, HC={len(existing_hc)})，跳过采样")
        print(f"  AG: {[f.name for f in existing_ag]}")
        print(f"  HC: {[f.name for f in existing_hc]}")
        return str(ag_out), str(hc_out)

    random.seed(seed)
    ag_files = sorted(AG_SOURCE.glob("*.json"))
    hc_files = sorted(HC_SOURCE.glob("*.json"))
    selected_ag = random.sample(ag_files, min(n_ag, len(ag_files)))
    selected_hc = random.sample(hc_files, min(n_hc, len(hc_files)))

    if ag_out.exists():
        shutil.rmtree(ag_out)
    if hc_out.exists():
        shutil.rmtree(hc_out)
    ag_out.mkdir(parents=True)
    hc_out.mkdir(parents=True)

    for f in selected_ag:
        shutil.copy2(f, ag_out / f.name)
    for f in selected_hc:
        shutil.copy2(f, hc_out / f.name)

    print(f"  AG ({n_ag}条): {[f.name for f in selected_ag]}")
    print(f"  HC ({n_hc}条): {[f.name for f in selected_hc]}")
    return str(ag_out), str(hc_out)


# ============================================================
# Feedback 实验 (E0/E1/E2)
# ============================================================

async def run_feedback_experiment(exp_name, ag_dir, hc_dir, cfg, results_dir, logs_dir):
    from benchmark_adapter import adapt_benchmark_sample
    from main_feedback import FeedbackPipeline
    from benchmark_evaluator import extract_prediction, compute_all_metrics

    pipeline = FeedbackPipeline()
    exp_log_dir = os.path.join(logs_dir, exp_name)
    os.makedirs(exp_log_dir, exist_ok=True)

    feedback_level = cfg["feedback_level"]
    bypass_intent = cfg["bypass_intent"]
    predictions = []

    all_samples = []
    for is_hc, data_dir, ds_label in [(False, ag_dir, "AG"), (True, hc_dir, "HC")]:
        json_files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".json")],
            key=lambda x: int("".join(filter(str.isdigit, x)) or 0),
        )
        for jf in json_files:
            all_samples.append((is_hc, data_dir, ds_label, jf))

    total = len(all_samples)
    for idx, (is_hc, data_dir, ds_label, jf) in enumerate(all_samples, 1):
        json_path = os.path.join(data_dir, jf)
        sample_id = jf.replace(".json", "")
        log_prefix = f"{ds_label}_{sample_id}_"
        print(f"    [{idx}/{total}] {ds_label}/{jf} ...", end="", flush=True)

        try:
            sample = adapt_benchmark_sample(json_path, is_hc, feedback_level)
        except Exception as e:
            print(f" SKIP (adapter: {e})", flush=True)
            continue

        t0 = time.time()
        try:
            gt_for_llm = sample["task_ground_truth"] if feedback_level == "task_aware" else ""
            result = await pipeline.process_from_dicts(
                question=sample["question"],
                ground_truth=gt_for_llm,
                workflow_config=sample["workflow_config"],
                execution_trace=sample["execution_trace"],
                bypass_intent=bypass_intent,
                prompt_variant="benchmark",
                log_dir=exp_log_dir,
                log_prefix=log_prefix,
                feedback_level=feedback_level,
                workflow_graph_str=sample.get("workflow_graph_str"),
                execution_trace_raw=sample.get("execution_trace_raw"),
            )
        except Exception as e:
            print(f" FAIL ({e})", flush=True)
            continue
        latency = (time.time() - t0) * 1000

        if not result.get("success"):
            print(f" FAIL ({result.get('error')})", flush=True)
            continue

        loc_result = result.get("localization_result", {})
        pred = extract_prediction(loc_result)
        usage = result.get("usage", {})

        entry = {
            "file": jf, "dataset": ds_label,
            "true_agent": sample["ground_truth"]["mistake_agent"],
            "true_step": sample["ground_truth"]["mistake_step"],
            "pred_agent": pred["predicted_agent"],
            "pred_step": pred["predicted_step"],
            "confidence": pred["confidence"],
            "ranked_candidates": pred["ranked_candidates"],
            "llm_calls": 1,
            "total_input_tokens": usage.get("prompt_tokens", 0),
            "total_output_tokens": usage.get("completion_tokens", 0),
            "latency_ms": latency,
            "is_handcrafted": is_hc,
            "history_length": sample["meta"]["history_length"],
            "agent_count": sample["meta"]["agent_count"],
        }
        predictions.append(entry)

        a_ok = "✓" if entry["pred_agent"] == entry["true_agent"] else "✗"
        s_ok = "✓" if str(entry["pred_step"]) == str(entry["true_step"]) else "✗"
        tok = f"{usage.get('prompt_tokens',0)}+{usage.get('completion_tokens',0)}tok"
        print(f" Agent:{a_ok} Step:{s_ok} ({latency:.0f}ms, {tok})", flush=True)

    metrics = compute_all_metrics(predictions)
    output = {
        "experiment": exp_name,
        "config": cfg,
        "metrics": metrics,
        "per_sample": predictions,
    }
    out_path = os.path.join(results_dir, f"{exp_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    return metrics


# ============================================================
# Baseline 实验 (B1-B6)
# ============================================================

def _init_openai_client():
    import yaml
    from openai import OpenAI
    config_path = REPO_ROOT / "feedback" / "feedback" / "prompts" / "openai.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        api_config = yaml.safe_load(f)
    return OpenAI(api_key=api_config["api_key"], base_url=api_config["api_base"])


class _TokenTracker:
    """Wraps an OpenAI client to transparently track token usage across all calls."""

    def __init__(self, client):
        self._client = client
        self.reset()

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0

    @property
    def chat(self):
        return _ChatProxy(self)

    def __getattr__(self, name):
        return getattr(self._client, name)


class _ChatProxy:
    def __init__(self, tracker):
        self._tracker = tracker

    @property
    def completions(self):
        return _CompletionsProxy(self._tracker)


class _CompletionsProxy:
    def __init__(self, tracker):
        self._tracker = tracker

    def create(self, **kwargs):
        resp = self._tracker._client.chat.completions.create(**kwargs)
        self._tracker.call_count += 1
        if hasattr(resp, "usage") and resp.usage:
            self._tracker.prompt_tokens += resp.usage.prompt_tokens or 0
            self._tracker.completion_tokens += resp.usage.completion_tokens or 0
        return resp


def run_baseline_experiment(exp_name, ag_dir, hc_dir, cfg, results_dir, logs_dir, model="gpt-5.4-nano"):
    from Lib.utils import (
        all_at_once as gpt_all_at_once,
        step_by_step as gpt_step_by_step,
        binary_search as gpt_binary_search,
    )
    from benchmark_evaluator import _parse_baseline_predictions, compute_all_metrics

    import math

    method = cfg["method"]
    no_gt = cfg["no_gt"]
    method_fn = {"all_at_once": gpt_all_at_once, "step_by_step": gpt_step_by_step, "binary_search": gpt_binary_search}[method]
    raw_client = _init_openai_client()
    tracker = _TokenTracker(raw_client)

    exp_log_dir = os.path.join(logs_dir, exp_name)
    os.makedirs(exp_log_dir, exist_ok=True)

    all_predictions = []
    datasets = [
        (False, ag_dir, "AG"),
        (True,  hc_dir, "HC"),
    ]
    for is_hc, data_dir, ds_label in datasets:
        log_path = os.path.join(exp_log_dir, f"{ds_label}_{method}.txt")
        print(f"    {ds_label}/{method} (no_gt={no_gt}) ...", flush=True)

        tracker.reset()
        buf = io.StringIO()
        err_buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err_buf):
            method_fn(
                client=tracker,
                directory_path=data_dir,
                is_handcrafted=is_hc,
                model=model,
                max_tokens=1024,
                no_ground_truth=no_gt,
                test_mode=False,
                skip_files=set(),
            )
        output_text = buf.getvalue()
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(output_text)

        baseline_preds = _parse_baseline_predictions(log_path)

        json_files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".json")],
            key=lambda x: int("".join(filter(str.isdigit, x)) or 0),
        )
        n_samples = len(json_files)
        per_sample_prompt = tracker.prompt_tokens // max(n_samples, 1)
        per_sample_completion = tracker.completion_tokens // max(n_samples, 1)
        per_sample_calls = tracker.call_count / max(n_samples, 1)

        for jf in json_files:
            with open(os.path.join(data_dir, jf), "r", encoding="utf-8") as f:
                sample_data = json.load(f)
            true_agent = sample_data.get("mistake_agent", "")
            true_step = str(sample_data.get("mistake_step", ""))
            history = sample_data.get("history", [])

            all_agents, seen = [], set()
            for msg in history:
                name = msg.get("role", "Unknown") if is_hc else msg.get("name", msg.get("role", "Unknown"))
                if is_hc and name.startswith("Orchestrator"):
                    name = "Orchestrator"
                if is_hc and name == "human":
                    continue
                if name not in seen:
                    seen.add(name)
                    all_agents.append(name)

            bp = baseline_preds.get(jf, {})
            pred_agent = bp.get("predicted_agent")
            pred_step = bp.get("predicted_step")
            ranked = []
            if pred_agent:
                ranked.append({"agent": pred_agent, "step": int(pred_step) if pred_step else 0, "confidence": 1.0, "reason": "baseline top-1"})
                for ag in all_agents:
                    if ag != pred_agent:
                        ranked.append({"agent": ag, "step": 0, "confidence": 0.0, "reason": "not predicted"})

            entry = {
                "file": jf, "dataset": ds_label,
                "true_agent": true_agent, "true_step": true_step,
                "pred_agent": pred_agent, "pred_step": pred_step,
                "confidence": 1.0,
                "ranked_candidates": ranked,
                "llm_calls": round(per_sample_calls),
                "total_input_tokens": per_sample_prompt,
                "total_output_tokens": per_sample_completion,
                "latency_ms": 0,
                "is_handcrafted": is_hc,
                "history_length": len(history),
                "agent_count": len(all_agents),
            }
            all_predictions.append(entry)

            a_ok = "✓" if pred_agent == true_agent else "✗"
            s_ok = "✓" if pred_step == true_step else "✗"
            parsed = "✓" if pred_agent else "✗parse"
            print(f"      {ds_label}/{jf}: {parsed} Agent:{a_ok} Step:{s_ok}", flush=True)

    metrics = compute_all_metrics(all_predictions)
    output = {
        "experiment": exp_name,
        "config": cfg,
        "metrics": metrics,
        "per_sample": all_predictions,
    }
    out_path = os.path.join(results_dir, f"{exp_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    return metrics


# ============================================================
# 汇总
# ============================================================

def generate_summary(results_dir, experiments):
    all_metrics = {}
    for exp in experiments:
        path = os.path.join(results_dir, f"{exp}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_metrics[exp] = data.get("metrics", {})

    if not all_metrics:
        return

    header = "| 实验 | Agent Acc | Step Acc | Joint Acc | Top-3 Agent | MRR Agent | Step MAE | Avg Input Tok | Avg Output Tok |"
    sep =    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    rows = [header, sep]
    for exp, m in all_metrics.items():
        eff = m.get("efficiency", {})
        mae = m.get("step_mae")
        mae_str = f"{mae:.2f}" if mae is not None else "-"
        rows.append(
            f"| {exp} "
            f"| {m.get('agent_accuracy',0):.2%} "
            f"| {m.get('step_accuracy',0):.2%} "
            f"| {m.get('joint_accuracy',0):.2%} "
            f"| {m.get('topk_agent_accuracy',{}).get('top3',0):.2%} "
            f"| {m.get('mrr_agent',0):.4f} "
            f"| {mae_str} "
            f"| {eff.get('avg_input_tokens',0):.0f} "
            f"| {eff.get('avg_output_tokens',0):.0f} |"
        )

    md = "\n".join(rows)

    # 按 history_length 分桶的详细指标表
    strat_lines = ["\n## 按对话长度分桶（20步一区间）\n"]
    for exp in experiments:
        path = os.path.join(results_dir, f"{exp}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        by_hl = data.get("stratified_metrics", {}).get("by_history_length", {})
        if not by_hl:
            continue
        strat_lines.append(f"### {exp}\n")
        strat_lines.append("| 区间 | N | Agent Acc | Step Acc | Joint Acc | Top-3 Agent | MRR Agent | Step MAE |")
        strat_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for bucket, sm in sorted(by_hl.items(), key=lambda x: int(x[0].split("-")[0])):
            mae_v = sm.get("step_mae")
            mae_s = f"{mae_v:.2f}" if mae_v is not None else "-"
            strat_lines.append(
                f"| {bucket} | {sm.get('count',0)} "
                f"| {sm.get('agent_accuracy',0):.2%} "
                f"| {sm.get('step_accuracy',0):.2%} "
                f"| {sm.get('joint_accuracy',0):.2%} "
                f"| {sm.get('topk_agent_accuracy',{}).get('top3',0):.2%} "
                f"| {sm.get('mrr_agent',0):.4f} "
                f"| {mae_s} |"
            )
        strat_lines.append("")
    strat_md = "\n".join(strat_lines)

    summary_path = os.path.join(results_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# 实验结果汇总\n\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(md + "\n")
        f.write(strat_md + "\n")

    print("\n" + "=" * 100)
    print("实验结果汇总")
    print("=" * 100)
    print(md)
    if strat_md.strip():
        print(strat_md)
    print(f"\n详细结果: {results_dir}/")
    print(f"hprompt 日志: {os.path.join(os.path.dirname(results_dir), 'logs')}/")
    print("=" * 100)


# ============================================================
# 可视化
# ============================================================

def generate_figures(results_dir, experiments):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[可视化] matplotlib 未安装，跳过图表生成")
        return

    plt.rcParams["font.size"] = 10

    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    all_data = {}
    for exp in experiments:
        path = os.path.join(results_dir, f"{exp}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            all_data[exp] = json.load(f)

    if len(all_data) < 2:
        print("[可视化] 结果不足 2 组，跳过图表生成")
        return

    labels = list(all_data.keys())
    feedback_labels = [l for l in labels if l.startswith("E")]
    baseline_labels = [l for l in labels if l.startswith("B")]

    def _get(exp, *keys, default=0):
        d = all_data[exp].get("metrics", {})
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, default)
            else:
                return default
        return d if d is not None else default

    colors_map = {
        "E1": "#1565C0", "E2": "#2196F3",
        "B1": "#4CAF50", "B2": "#FF9800", "B3": "#F44336",
        "B4": "#81C784", "B5": "#FFB74D", "B6": "#E57373",
    }

    # ── 图1: 核心指标对比 (Agent / Step / Joint Accuracy) ──
    metric_keys = ["agent_accuracy", "step_accuracy", "joint_accuracy"]
    metric_labels_zh = ["Agent Acc", "Step Acc", "Joint Acc"]
    x = np.arange(len(metric_keys))
    width = 0.8 / max(len(labels), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 5))
    for i, lab in enumerate(labels):
        vals = [_get(lab, mk) for mk in metric_keys]
        color = colors_map.get(lab, f"C{i}")
        bars = ax.bar(x + i * width, vals, width, label=lab, color=color)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.0%}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(metric_labels_zh)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=7, ncol=min(len(labels), 5), loc="upper right")
    ax.set_title("Core Metrics Comparison (all 9 experiments)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "1_core_metrics.png"), dpi=150)
    plt.close(fig)

    # ── 图2: 排名指标 (Top-3 Agent / MRR Agent) ──
    rank_keys = [("topk_agent_accuracy", "top3"), ("mrr_agent",)]
    rank_labels = ["Top-3 Agent Acc", "MRR Agent"]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.0), 5))
    x2 = np.arange(len(rank_keys))
    for i, lab in enumerate(labels):
        vals = [_get(lab, *rk) for rk in rank_keys]
        color = colors_map.get(lab, f"C{i}")
        bars = ax.bar(x2 + i * width, vals, width, label=lab, color=color)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Score")
    ax.set_xticks(x2 + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(rank_labels)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=7, ncol=min(len(labels), 5), loc="upper right")
    ax.set_title("Ranking Metrics")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "2_ranking_metrics.png"), dpi=150)
    plt.close(fig)

    # ── 图3: 开卷 vs 闭卷对比 ──
    pairs = [("E2", "E1"), ("B4", "B1"), ("B5", "B2"), ("B6", "B3")]
    pairs = [(c, o) for c, o in pairs if c in all_data and o in all_data]
    if pairs:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax_i, mk in enumerate(metric_keys):
            ax = axes[ax_i]
            pair_labels = [f"{c}→{o}" for c, o in pairs]
            closed_vals = [_get(c, mk) for c, _ in pairs]
            open_vals = [_get(o, mk) for _, o in pairs]
            x3 = np.arange(len(pairs))
            w3 = 0.35
            ax.bar(x3 - w3 / 2, closed_vals, w3, label="Closed-book", color="#BBDEFB")
            ax.bar(x3 + w3 / 2, open_vals, w3, label="Open-book", color="#1565C0")
            ax.set_xticks(x3)
            ax.set_xticklabels(pair_labels, fontsize=8)
            ax.set_ylim(0, 1.1)
            ax.set_title(metric_labels_zh[ax_i])
            ax.legend(fontsize=7)
        fig.suptitle("Closed-book vs Open-book (same method)", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "3_open_vs_closed.png"), dpi=150)
        plt.close(fig)

    # ── 图4: 置信度校准图 (仅 feedback 方法) ──
    if feedback_labels:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        for lab in feedback_labels:
            bins = _get(lab, "calibration_bins", default=[])
            if not bins:
                continue
            confs = [b["avg_confidence"] for b in bins if b.get("count", 0) > 0]
            accs = [b["accuracy"] for b in bins if b.get("count", 0) > 0]
            if confs:
                color = colors_map.get(lab, "C0")
                ece = _get(lab, "ece")
                ax.plot(confs, accs, "o-", label=f"{lab} (ECE={ece:.3f})", color=color)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title("Reliability Diagram (feedback methods)")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "4_reliability_diagram.png"), dpi=150)
        plt.close(fig)

    # ── 图5: 效率对比 (Token 分项 / Total Tokens / LLM Calls) ──
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    x5 = np.arange(len(labels))

    input_toks = [_get(l, "efficiency", "avg_input_tokens") for l in labels]
    output_toks = [_get(l, "efficiency", "avg_output_tokens") for l in labels]
    total_toks = [i + o for i, o in zip(input_toks, output_toks)]
    w5 = 0.35
    ax1.bar(x5 - w5 / 2, input_toks, w5, label="Input", color="#42A5F5")
    ax1.bar(x5 + w5 / 2, output_toks, w5, label="Output", color="#FF7043")
    ax1.set_xticks(x5)
    ax1.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax1.set_ylabel("Avg Tokens / Sample")
    ax1.set_title("Input vs Output Tokens")
    ax1.legend(fontsize=8)
    for xi, (iv, ov) in enumerate(zip(input_toks, output_toks)):
        if iv > 0:
            ax1.text(xi - w5 / 2, iv, f"{iv:.0f}", ha="center", va="bottom", fontsize=6)
        if ov > 0:
            ax1.text(xi + w5 / 2, ov, f"{ov:.0f}", ha="center", va="bottom", fontsize=6)

    bar_colors = [colors_map.get(l, "C0") for l in labels]
    bars_t = ax2.bar(x5, total_toks, color=bar_colors)
    ax2.set_xticks(x5)
    ax2.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax2.set_ylabel("Avg Total Tokens / Sample")
    ax2.set_title("Total Token Consumption")
    for bar in bars_t:
        if bar.get_height() > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7)

    calls = [_get(l, "efficiency", "avg_llm_calls") for l in labels]
    ax3.bar(x5, calls, color=bar_colors)
    ax3.set_xticks(x5)
    ax3.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax3.set_ylabel("Avg LLM Calls / Sample")
    ax3.set_title("LLM Calls per Sample")
    for xi, c in enumerate(calls):
        if c > 0:
            ax3.text(xi, c + 0.05, f"{c:.1f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Efficiency Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "5_efficiency.png"), dpi=150)
    plt.close(fig)

    # ── 图6: 每样本详情热力图 (Agent 是否预测正确) ──
    all_files = set()
    for exp in labels:
        for s in all_data[exp].get("per_sample", []):
            all_files.add(f"{s.get('dataset','')}/{s['file']}")
    all_files = sorted(all_files)
    if all_files:
        matrix = np.full((len(labels), len(all_files)), np.nan)
        for i, exp in enumerate(labels):
            for s in all_data[exp].get("per_sample", []):
                fkey = f"{s.get('dataset','')}/{s['file']}"
                j = all_files.index(fkey)
                agent_ok = 1 if s.get("pred_agent") == s.get("true_agent") else 0
                step_ok = 1 if str(s.get("pred_step")) == str(s.get("true_step")) else 0
                matrix[i, j] = agent_ok + step_ok  # 0=both wrong, 1=agent only, 2=both correct

        fig, ax = plt.subplots(figsize=(max(8, len(all_files) * 0.8), max(4, len(labels) * 0.5)))
        cmap = plt.cm.colors.ListedColormap(["#EF5350", "#FFF176", "#66BB6A"])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(all_files)))
        ax.set_xticklabels(all_files, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(["Both ✗", "Agent ✓ only", "Both ✓"], fontsize=8)
        ax.set_title("Per-sample Prediction Heatmap (red=wrong, yellow=agent only, green=both correct)")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "6_per_sample_heatmap.png"), dpi=150)
        plt.close(fig)

    print(f"\n[可视化] 图表已保存到 {fig_dir}/")
    for f in sorted(os.listdir(fig_dir)):
        if f.endswith(".png"):
            print(f"  - {f}")


# ============================================================
# 主入口
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="一键运行 9 组实验（采样数据）")
    parser.add_argument("--experiments", nargs="+", default=ALL_EXPERIMENTS, choices=ALL_EXPERIMENTS,
                        help="要运行的实验列表 (默认全部)")
    parser.add_argument("--n_ag", type=int, default=10, help="从 AG 采样的数量")
    parser.add_argument("--n_hc", type=int, default=10, help="从 HC 采样的数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resample", action="store_true", help="强制重新采样")
    parser.add_argument("--model", type=str, default="gpt-5.4-nano", help="模型名称")
    parser.add_argument("--run_id", type=str, default=None,
                        help="本次运行的 ID（默认自动生成时间戳，如 run_0327_1430）")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("run_%m%d_%H%M")
    run_dir = SCRIPT_DIR / "runs" / run_id
    results_dir = str(run_dir / "results")
    logs_dir = str(run_dir / "logs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    from loguru import logger
    logger.remove()
    logger.add(os.path.join(logs_dir, "run.log"), level="DEBUG")

    print("=" * 60)
    print(f"  测试实验 ({len(args.experiments)} 组)")
    print(f"  实验列表: {', '.join(args.experiments)}")
    print(f"  Run ID : {run_id}")
    print(f"  模型: {args.model}  种子: {args.seed}")
    print(f"  采样: AG={args.n_ag}, HC={args.n_hc}")
    print(f"  输出: {run_dir}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n[1] 采样数据")
    ag_dir, hc_dir = sample_data(args.n_ag, args.n_hc, args.seed, args.resample, run_dir=str(run_dir))

    total = len(args.experiments)
    all_results = {}
    for exp_i, exp_name in enumerate(args.experiments, 1):
        cfg = EXPERIMENT_CONFIG[exp_name]
        print(f"\n{'─'*50}", flush=True)
        print(f"  [{exp_i}/{total}] {exp_name} ({cfg['type']})", flush=True)
        t0 = time.time()
        try:
            if cfg["type"] == "feedback":
                metrics = await run_feedback_experiment(exp_name, ag_dir, hc_dir, cfg, results_dir, logs_dir)
            else:
                metrics = run_baseline_experiment(exp_name, ag_dir, hc_dir, cfg, results_dir, logs_dir, args.model)
            elapsed = time.time() - t0
            all_results[exp_name] = metrics
            print(f"  => {exp_name} done ({elapsed:.1f}s) Agent:{metrics['agent_accuracy']:.2%} Step:{metrics['step_accuracy']:.2%} Joint:{metrics['joint_accuracy']:.2%}", flush=True)
        except Exception as e:
            print(f"  => {exp_name} FAILED: {e}", flush=True)

    generate_summary(results_dir, args.experiments)
    generate_figures(results_dir, args.experiments)

    print(f"\n本次运行完整输出: {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
