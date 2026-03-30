#!/usr/bin/env python3
"""
benchmark_evaluator.py — 统一评测脚本

一键运行 feedback 管线 + 计算全部指标 + 输出可视化图表。
"""

import os
import sys
import re
import json
import time
import asyncio
import argparse
import math
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional

# 将 feedback 子模块加入 Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
FEEDBACK_ROOT = os.path.join(REPO_ROOT, "feedback")
FEEDBACK_CODE = os.path.join(FEEDBACK_ROOT, "feedback")
if FEEDBACK_CODE not in sys.path:
    sys.path.insert(0, FEEDBACK_CODE)

from benchmark_adapter import adapt_benchmark_sample


# ============================================================
# T4.2: 预测结果提取
# ============================================================

def extract_prediction(localization_result: dict) -> dict:
    loc = localization_result.get("localization", {})
    assessment = localization_result.get("assessment", {})
    target_scope = loc.get("target_scope", [])
    return {
        "predicted_agent": target_scope[0] if target_scope else None,
        "predicted_step": str(loc.get("error_step")) if loc.get("error_step") is not None else None,
        "confidence": assessment.get("confidence", 0.0),
        "ranked_candidates": loc.get("ranked_candidates", []),
    }


# ============================================================
# T4.3: 指标计算
# ============================================================

def agent_accuracy(preds: List[dict]) -> float:
    if not preds:
        return 0.0
    correct = sum(1 for p in preds if p["pred_agent"] == p["true_agent"])
    return correct / len(preds)


def step_accuracy(preds: List[dict]) -> float:
    if not preds:
        return 0.0
    correct = sum(1 for p in preds if str(p["pred_step"]) == str(p["true_step"]))
    return correct / len(preds)


def joint_accuracy(preds: List[dict]) -> float:
    if not preds:
        return 0.0
    correct = sum(
        1 for p in preds
        if p["pred_agent"] == p["true_agent"] and str(p["pred_step"]) == str(p["true_step"])
    )
    return correct / len(preds)


def topk_agent_accuracy(preds: List[dict], k: int) -> float:
    if not preds:
        return 0.0
    correct = 0
    for p in preds:
        candidates = p.get("ranked_candidates", [])[:k]
        if any(c.get("agent") == p["true_agent"] for c in candidates):
            correct += 1
    return correct / len(preds)


def topk_step_accuracy(preds: List[dict], k: int) -> float:
    if not preds:
        return 0.0
    correct = 0
    for p in preds:
        candidates = p.get("ranked_candidates", [])[:k]
        if any(str(c.get("step")) == str(p["true_step"]) for c in candidates):
            correct += 1
    return correct / len(preds)


def mrr_agent(preds: List[dict]) -> float:
    if not preds:
        return 0.0
    total = 0.0
    for p in preds:
        candidates = p.get("ranked_candidates", [])
        for i, c in enumerate(candidates):
            if c.get("agent") == p["true_agent"]:
                total += 1.0 / (i + 1)
                break
    return total / len(preds)


def mrr_step(preds: List[dict]) -> float:
    if not preds:
        return 0.0
    total = 0.0
    for p in preds:
        candidates = p.get("ranked_candidates", [])
        for i, c in enumerate(candidates):
            if c.get("agent") == p["true_agent"] and str(c.get("step")) == str(p["true_step"]):
                total += 1.0 / (i + 1)
                break
    return total / len(preds)


def step_mae(preds: List[dict]) -> Optional[float]:
    agent_correct = [p for p in preds if p["pred_agent"] == p["true_agent"] and p["pred_step"] is not None]
    if not agent_correct:
        return None
    total = sum(abs(int(p["pred_step"]) - int(p["true_step"])) for p in agent_correct)
    return total / len(agent_correct)


def step_window_accuracy(preds: List[dict], w: int) -> float:
    if not preds:
        return 0.0
    correct = 0
    for p in preds:
        if p["pred_step"] is not None:
            try:
                if abs(int(p["pred_step"]) - int(p["true_step"])) <= w:
                    correct += 1
            except (ValueError, TypeError):
                pass
    return correct / len(preds)


def confidence_calibration(preds: List[dict], n_bins: int = 10) -> dict:
    if not preds:
        return {"ece": 0.0, "calibration_bins": []}
    bins = [[] for _ in range(n_bins)]
    for p in preds:
        conf = p.get("confidence", 0.0)
        if conf is None:
            conf = 0.0
        b = min(int(conf * n_bins), n_bins - 1)
        is_correct = 1 if p["pred_agent"] == p["true_agent"] else 0
        bins[b].append((conf, is_correct))

    ece = 0.0
    calibration_bins = []
    for i, bin_items in enumerate(bins):
        if not bin_items:
            calibration_bins.append({"avg_confidence": (i + 0.5) / n_bins, "accuracy": 0.0, "count": 0})
            continue
        avg_conf = sum(c for c, _ in bin_items) / len(bin_items)
        acc = sum(a for _, a in bin_items) / len(bin_items)
        ece += abs(avg_conf - acc) * len(bin_items) / len(preds)
        calibration_bins.append({"avg_confidence": avg_conf, "accuracy": acc, "count": len(bin_items)})

    return {"ece": ece, "calibration_bins": calibration_bins}


def efficiency_stats(preds: List[dict]) -> dict:
    n = len(preds) or 1
    return {
        "avg_llm_calls": sum(p.get("llm_calls", 1) for p in preds) / n,
        "avg_input_tokens": sum(p.get("total_input_tokens", 0) for p in preds) / n,
        "avg_output_tokens": sum(p.get("total_output_tokens", 0) for p in preds) / n,
        "avg_latency_ms": sum(p.get("latency_ms", 0) for p in preds) / n,
    }


def stratified_metrics(preds: List[dict]) -> dict:
    def _compute(subset):
        if not subset:
            return {}
        cal = confidence_calibration(subset)
        mae = step_mae(subset)
        return {
            "agent_accuracy": agent_accuracy(subset),
            "step_accuracy": step_accuracy(subset),
            "joint_accuracy": joint_accuracy(subset),
            "topk_agent_accuracy": {"top1": topk_agent_accuracy(subset, 1), "top3": topk_agent_accuracy(subset, 3)},
            "mrr_agent": mrr_agent(subset),
            "step_mae": mae,
            "ece": cal["ece"],
            "count": len(subset),
        }

    result = {"by_data_type": {}, "by_history_length": {}, "by_agent_count": {}}

    by_type = defaultdict(list)
    for p in preds:
        key = "hand_crafted" if p.get("is_handcrafted") else "algorithm_generated"
        by_type[key].append(p)
    for k, v in by_type.items():
        result["by_data_type"][k] = _compute(v)

    by_hl = defaultdict(list)
    for p in preds:
        hl = p.get("history_length", 0)
        lo = (hl // 20) * 20
        hi = lo + 19
        bucket = f"{lo}-{hi}"
        by_hl[bucket].append(p)
    for k, v in sorted(by_hl.items(), key=lambda x: int(x[0].split("-")[0])):
        result["by_history_length"][k] = _compute(v)

    by_ac = defaultdict(list)
    for p in preds:
        by_ac[str(p.get("agent_count", "?"))].append(p)
    for k, v in by_ac.items():
        result["by_agent_count"][k] = _compute(v)

    return result


def compute_all_metrics(preds: List[dict]) -> dict:
    cal = confidence_calibration(preds)
    return {
        "agent_accuracy": agent_accuracy(preds),
        "step_accuracy": step_accuracy(preds),
        "joint_accuracy": joint_accuracy(preds),
        "topk_agent_accuracy": {"top1": topk_agent_accuracy(preds, 1), "top3": topk_agent_accuracy(preds, 3)},
        "topk_step_accuracy": {"top1": topk_step_accuracy(preds, 1), "top3": topk_step_accuracy(preds, 3)},
        "mrr_agent": mrr_agent(preds),
        "mrr_step": mrr_step(preds),
        "step_mae": step_mae(preds),
        "step_window_accuracy": {
            "w0": step_window_accuracy(preds, 0),
            "w1": step_window_accuracy(preds, 1),
            "w2": step_window_accuracy(preds, 2),
        },
        "ece": cal["ece"],
        "calibration_bins": cal["calibration_bins"],
        "efficiency": efficiency_stats(preds),
    }


# ============================================================
# T4.1: 评测主流程
# ============================================================

def _progress_path(output_path: str) -> str:
    return output_path + ".progress.jsonl"


def _load_progress(output_path: str) -> tuple:
    """从 .progress.jsonl 加载已完成的条目，返回 (predictions, failed, done_files)"""
    ppath = _progress_path(output_path)
    predictions, failed, done_files = [], [], set()
    if not os.path.exists(ppath):
        return predictions, failed, done_files
    with open(ppath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("_status") == "failed":
                failed.append({"file": entry["file"], "error": entry.get("error", "")})
            else:
                predictions.append(entry)
            done_files.add(entry["file"])
    return predictions, failed, done_files


def _append_progress(output_path: str, entry: dict):
    """追加一条记录到 .progress.jsonl"""
    ppath = _progress_path(output_path)
    os.makedirs(os.path.dirname(ppath) or ".", exist_ok=True)
    with open(ppath, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


async def evaluate_all(
    data_dir: str,
    is_handcrafted: bool,
    feedback_level: str,
    bypass_intent: bool,
    prompt_variant: str,
    output_path: str,
    test_mode: bool = False,
):
    from main_feedback import FeedbackPipeline

    pipeline = FeedbackPipeline()

    json_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".json")],
        key=lambda x: int("".join(filter(str.isdigit, x)) or 0),
    )
    if test_mode:
        json_files = json_files[:1]

    predictions, failed, done_files = _load_progress(output_path)
    if done_files:
        print(f"[断点续传] 已完成 {len(done_files)} 条，从断点继续")

    total = len(json_files)

    for i, jf in enumerate(json_files):
        if jf in done_files:
            continue

        json_path = os.path.join(data_dir, jf)
        print(f"[{i+1}/{total}] Processing {jf} ...", flush=True)

        try:
            sample = adapt_benchmark_sample(json_path, is_handcrafted, feedback_level)
        except Exception as e:
            print(f"  [SKIP] Adapter error: {e}")
            fail_entry = {"file": jf, "error": f"adapter: {e}", "_status": "failed"}
            failed.append({"file": jf, "error": f"adapter: {e}"})
            _append_progress(output_path, fail_entry)
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
                prompt_variant=prompt_variant,
                feedback_level=feedback_level,
                workflow_graph_str=sample.get("workflow_graph_str"),
                execution_trace_raw=sample.get("execution_trace_raw"),
            )
        except Exception as e:
            print(f"  [FAIL] Pipeline error: {e}")
            fail_entry = {"file": jf, "error": f"pipeline: {e}", "_status": "failed"}
            failed.append({"file": jf, "error": f"pipeline: {e}"})
            _append_progress(output_path, fail_entry)
            continue
        latency = (time.time() - t0) * 1000

        if not result.get("success"):
            print(f"  [FAIL] {result.get('error')}")
            fail_entry = {"file": jf, "error": result.get("error"), "_status": "failed"}
            failed.append({"file": jf, "error": result.get("error")})
            _append_progress(output_path, fail_entry)
            continue

        loc_result = result.get("localization_result", {})
        pred = extract_prediction(loc_result)

        usage = result.get("usage", {})
        entry = {
            "file": jf,
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
            "is_handcrafted": is_handcrafted,
            "history_length": sample["meta"]["history_length"],
            "agent_count": sample["meta"]["agent_count"],
        }
        predictions.append(entry)
        _append_progress(output_path, entry)

        agent_ok = "✓" if entry["pred_agent"] == entry["true_agent"] else "✗"
        step_ok = "✓" if str(entry["pred_step"]) == str(entry["true_step"]) else "✗"
        print(f"  Agent: {entry['pred_agent']} {agent_ok}  Step: {entry['pred_step']} {step_ok}  ({latency:.0f}ms)")

    # 全部完成后汇总输出最终结果 JSON
    metrics = compute_all_metrics(predictions)
    strat = stratified_metrics(predictions)
    output = {
        "config": {
            "data_dir": data_dir,
            "is_handcrafted": is_handcrafted,
            "feedback_level": feedback_level,
            "bypass_intent": bypass_intent,
            "prompt_variant": prompt_variant,
            "total_samples": total,
            "successful_predictions": len(predictions),
            "failed_predictions": len(failed),
        },
        "metrics": metrics,
        "stratified_metrics": strat,
        "per_sample": predictions,
        "failures": failed,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print(f"Samples: {len(predictions)}/{total} succeeded, {len(failed)} failed")
    print(f"Agent Accuracy: {metrics['agent_accuracy']:.4f}")
    print(f"Step Accuracy:  {metrics['step_accuracy']:.4f}")
    print(f"Joint Accuracy: {metrics['joint_accuracy']:.4f}")
    print(f"{'='*60}")

    return output


# ============================================================
# Baseline 评测：解析原论文 inference.py 输出并计算统一指标
# ============================================================

def _parse_baseline_predictions(eval_file: str) -> dict:
    """解析原论文 inference.py 的 txt 输出文件，返回 {filename: {predicted_agent, predicted_step}}"""
    if not os.path.exists(eval_file):
        print(f"Error: eval file not found: {eval_file}")
        return {}
    with open(eval_file, 'r', encoding='utf-8') as f:
        data = f.read()

    predictions = {}
    pattern = r"Prediction for ([^:]+\.json):(.*?)(?=Prediction for|\Z)"
    for block in re.finditer(pattern, data, re.DOTALL):
        idx = block.group(1).strip()
        content = block.group(2).strip()
        agent_match = (
            re.search(r"Agent Name:\s*\**\s*([\w_]+)", content, re.IGNORECASE)
            or re.search(r"^\s*\**\s*([\w_]+)\s*\**\s*:", content, re.MULTILINE)
        )
        step_match = (
            re.search(r"Step Number:\s*\**\s*(\d+)", content, re.IGNORECASE)
            or re.search(r"Step\s*(?:Number)?\s*[:=]\s*\**\s*(\d+)", content, re.IGNORECASE)
        )
        if agent_match and step_match:
            predictions[idx] = {
                "predicted_agent": agent_match.group(1),
                "predicted_step": step_match.group(1),
            }
    return predictions


def evaluate_baseline(
    eval_file: str,
    data_dir: str,
    is_handcrafted: bool,
    method_name: str,
    output_path: str,
):
    """评测原论文方法的输出，生成统一格式的结果 JSON"""
    baseline_preds = _parse_baseline_predictions(eval_file)
    if not baseline_preds:
        print(f"No predictions parsed from {eval_file}")
        return

    json_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".json")],
        key=lambda x: int("".join(filter(str.isdigit, x)) or 0),
    )

    predictions = []
    for jf in json_files:
        json_path = os.path.join(data_dir, jf)
        with open(json_path, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)

        true_agent = sample_data.get("mistake_agent", "")
        true_step = str(sample_data.get("mistake_step", ""))

        history = sample_data.get("history", [])
        index_agent = "role" if is_handcrafted else "name"
        all_agents = []
        seen = set()
        for msg in history:
            if is_handcrafted:
                name = msg.get("role", "Unknown")
                if name.startswith("Orchestrator"):
                    name = "Orchestrator"
                if name == "human":
                    continue
            else:
                name = msg.get("name", msg.get("role", "Unknown"))
            if name not in seen:
                seen.add(name)
                all_agents.append(name)

        bp = baseline_preds.get(jf, {})
        pred_agent = bp.get("predicted_agent")
        pred_step = bp.get("predicted_step")

        ranked = []
        if pred_agent:
            ranked.append({
                "agent": pred_agent,
                "step": int(pred_step) if pred_step else 0,
                "confidence": 1.0,
                "reason": "baseline top-1 prediction",
            })
            for ag in all_agents:
                if ag != pred_agent:
                    ranked.append({
                        "agent": ag,
                        "step": 0,
                        "confidence": 0.0,
                        "reason": "not predicted by baseline",
                    })

        entry = {
            "file": jf,
            "true_agent": true_agent,
            "true_step": true_step,
            "pred_agent": pred_agent,
            "pred_step": pred_step,
            "confidence": 1.0,
            "ranked_candidates": ranked,
            "llm_calls": 1 if "all_at_once" in method_name else 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "latency_ms": 0,
            "is_handcrafted": is_handcrafted,
            "history_length": len(history),
            "agent_count": len(all_agents),
        }
        predictions.append(entry)

    metrics = compute_all_metrics(predictions)
    strat = stratified_metrics(predictions)

    parsed_count = sum(1 for p in predictions if p["pred_agent"] is not None)
    output = {
        "config": {
            "eval_file": eval_file,
            "data_dir": data_dir,
            "is_handcrafted": is_handcrafted,
            "method": method_name,
            "total_samples": len(json_files),
            "successful_predictions": parsed_count,
            "failed_predictions": len(json_files) - parsed_count,
        },
        "metrics": metrics,
        "stratified_metrics": strat,
        "per_sample": predictions,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"Baseline evaluation results saved to {output_path}")
    print(f"  Parsed: {parsed_count}/{len(json_files)}")
    print(f"  Agent Accuracy: {metrics['agent_accuracy']:.4f}")
    print(f"  Step Accuracy:  {metrics['step_accuracy']:.4f}")
    print(f"  Joint Accuracy: {metrics['joint_accuracy']:.4f}")


# ============================================================
# 指标计算正确性验证
# ============================================================

def test_metrics():
    """用手工构造的数据验证所有指标计算是否正确"""
    preds = [
        {
            "file": "1.json", "true_agent": "A", "true_step": "1",
            "pred_agent": "A", "pred_step": "1", "confidence": 0.9,
            "ranked_candidates": [
                {"agent": "A", "step": 1, "confidence": 0.9, "reason": ""},
                {"agent": "B", "step": 2, "confidence": 0.3, "reason": ""},
                {"agent": "C", "step": 3, "confidence": 0.1, "reason": ""},
            ],
            "llm_calls": 1, "total_input_tokens": 100, "total_output_tokens": 50, "latency_ms": 1000,
            "is_handcrafted": False, "history_length": 5, "agent_count": 3,
        },
        {
            "file": "2.json", "true_agent": "B", "true_step": "3",
            "pred_agent": "D", "pred_step": "5", "confidence": 0.7,
            "ranked_candidates": [
                {"agent": "D", "step": 5, "confidence": 0.7, "reason": ""},
                {"agent": "B", "step": 3, "confidence": 0.6, "reason": ""},
                {"agent": "C", "step": 2, "confidence": 0.2, "reason": ""},
            ],
            "llm_calls": 1, "total_input_tokens": 200, "total_output_tokens": 80, "latency_ms": 2000,
            "is_handcrafted": False, "history_length": 7, "agent_count": 3,
        },
        {
            "file": "3.json", "true_agent": "C", "true_step": "2",
            "pred_agent": "D", "pred_step": "4", "confidence": 0.8,
            "ranked_candidates": [
                {"agent": "D", "step": 4, "confidence": 0.8, "reason": ""},
                {"agent": "A", "step": 1, "confidence": 0.5, "reason": ""},
                {"agent": "C", "step": 2, "confidence": 0.3, "reason": ""},
            ],
            "llm_calls": 1, "total_input_tokens": 150, "total_output_tokens": 60, "latency_ms": 1500,
            "is_handcrafted": True, "history_length": 10, "agent_count": 4,
        },
        {
            "file": "4.json", "true_agent": "E", "true_step": "0",
            "pred_agent": "E", "pred_step": "0", "confidence": 0.85,
            "ranked_candidates": [
                {"agent": "E", "step": 0, "confidence": 0.85, "reason": ""},
                {"agent": "F", "step": 1, "confidence": 0.4, "reason": ""},
            ],
            "llm_calls": 1, "total_input_tokens": 80, "total_output_tokens": 40, "latency_ms": 800,
            "is_handcrafted": False, "history_length": 4, "agent_count": 2,
        },
    ]

    results = {}
    errors = []

    def check(name, got, expected, tol=1e-6):
        results[name] = {"got": got, "expected": expected}
        if expected is None:
            ok = got is None
        elif isinstance(expected, float) and math.isnan(expected):
            ok = isinstance(got, float) and math.isnan(got)
        else:
            ok = abs(got - expected) < tol
        status = "✓" if ok else "✗"
        if got is None or expected is None:
            print(f"  {status} {name}: got={got}, expected={expected}")
        else:
            print(f"  {status} {name}: got={got:.6f}, expected={expected:.6f}")
        if not ok:
            errors.append(name)

    print("\n=== 指标计算正确性验证 ===\n")

    # Agent Acc: A=A✓ D≠B✗ D≠C✗ E=E✓ → 2/4=0.5
    check("agent_accuracy", agent_accuracy(preds), 0.5)

    # Step Acc: 1=1✓ 5≠3✗ 4≠2✗ 0=0✓ → 2/4=0.5
    check("step_accuracy", step_accuracy(preds), 0.5)

    # Joint: (A,1)✓ (D,5)✗ (D,4)✗ (E,0)✓ → 2/4=0.5
    check("joint_accuracy", joint_accuracy(preds), 0.5)

    # Top-1 Agent: ranked[0]==true? A=A✓ D≠B✗ D≠C✗ E=E✓ → 2/4=0.5
    check("top1_agent", topk_agent_accuracy(preds, 1), 0.5)

    # Top-3 Agent: true in ranked[:3]? A∈[A,B,C]✓ B∈[D,B,C]✓ C∈[D,A,C]✓ E∈[E,F]✓ → 4/4=1.0
    check("top3_agent", topk_agent_accuracy(preds, 3), 1.0)

    # Top-1 Step: ranked[0].step==true? 1=1✓ 5≠3✗ 4≠2✗ 0=0✓ → 2/4=0.5
    check("top1_step", topk_step_accuracy(preds, 1), 0.5)

    # Top-3 Step: true_step in any ranked[:3].step? 1∈{1,2,3}✓ 3∈{5,3,2}✓ 2∈{4,1,2}✓ 0∈{0,1}✓ → 4/4=1.0
    check("top3_step", topk_step_accuracy(preds, 3), 1.0)

    # MRR Agent: A@1→1.0, B@2→0.5, C@3→0.333, E@1→1.0 → (1+0.5+0.333+1)/4=0.70833
    check("mrr_agent", mrr_agent(preds), (1.0 + 0.5 + 1/3 + 1.0) / 4)

    # MRR Step: (A,1)@1→1.0, (B,3)@2→0.5, (C,2)@3→0.333, (E,0)@1→1.0 → same
    check("mrr_step", mrr_step(preds), (1.0 + 0.5 + 1/3 + 1.0) / 4)

    # Step MAE: only agent-correct (samples 1,4): |1-1|+|0-0|=0 → 0/2=0.0
    check("step_mae", step_mae(preds), 0.0)

    # Step Window w0: |1-1|=0✓ |5-3|=2✗ |4-2|=2✗ |0-0|=0✓ → 2/4=0.5
    check("step_window_w0", step_window_accuracy(preds, 0), 0.5)

    # Step Window w1: 0≤1✓ 2≤1✗ 2≤1✗ 0≤1✓ → 2/4=0.5
    check("step_window_w1", step_window_accuracy(preds, 1), 0.5)

    # Step Window w2: 0≤2✓ 2≤2✓ 2≤2✓ 0≤2✓ → 4/4=1.0
    check("step_window_w2", step_window_accuracy(preds, 2), 1.0)

    # ECE: confs=[0.9, 0.7, 0.8, 0.85], correct=[1,0,0,1]
    # bin7(0.7-0.8): [0.7], correct=[0] → avg_c=0.7 acc=0 |0.7-0|=0.7 w=1/4
    # bin8(0.8-0.9): [0.8, 0.85], correct=[0,1] → avg_c=0.825 acc=0.5 |0.825-0.5|=0.325 w=2/4
    # bin9(0.9-1.0): [0.9], correct=[1] → avg_c=0.9 acc=1.0 |0.9-1|=0.1 w=1/4
    # ECE = 0.7*0.25 + 0.325*0.5 + 0.1*0.25 = 0.175 + 0.1625 + 0.025 = 0.3625
    cal = confidence_calibration(preds, n_bins=10)
    check("ece", cal["ece"], 0.3625)

    # Efficiency
    eff = efficiency_stats(preds)
    check("avg_llm_calls", eff["avg_llm_calls"], 1.0)
    check("avg_input_tokens", eff["avg_input_tokens"], (100+200+150+80)/4)
    check("avg_output_tokens", eff["avg_output_tokens"], (50+80+60+40)/4)
    check("avg_latency_ms", eff["avg_latency_ms"], (1000+2000+1500+800)/4)

    # Stratified
    strat = stratified_metrics(preds)
    ag_strat = strat["by_data_type"].get("algorithm_generated", {})
    check("strat_ag_agent_acc", ag_strat.get("agent_accuracy", -1), 2/3)
    hc_strat = strat["by_data_type"].get("hand_crafted", {})
    check("strat_hc_agent_acc", hc_strat.get("agent_accuracy", -1), 0.0)

    print(f"\n{'='*40}")
    if errors:
        print(f"FAILED: {len(errors)} metrics incorrect: {errors}")
    else:
        print(f"ALL {len(results)} METRICS PASSED ✓")
    print(f"{'='*40}\n")
    return len(errors) == 0


# ============================================================
# T4.5: 可视化
# ============================================================

def visualize_results(result_files: List[str], output_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed; skipping visualization.")
        return

    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    for rf in result_files:
        with open(rf, "r", encoding="utf-8") as f:
            data = json.load(f)
        label = os.path.splitext(os.path.basename(rf))[0]
        all_results[label] = data

    # --- 主对比表 (Markdown + LaTeX) ---
    headers = ["Method", "Agent Acc", "Step Acc", "Joint Acc", "Top-3 Agent", "MRR Agent", "Step MAE"]
    rows = []
    for label, data in all_results.items():
        m = data.get("metrics", {})
        rows.append([
            label,
            f"{m.get('agent_accuracy', 0):.4f}",
            f"{m.get('step_accuracy', 0):.4f}",
            f"{m.get('joint_accuracy', 0):.4f}",
            f"{m.get('topk_agent_accuracy', {}).get('top3', 0):.4f}",
            f"{m.get('mrr_agent', 0):.4f}",
            f"{m['step_mae']:.2f}" if m.get('step_mae') is not None else "-",
        ])

    md_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")
    with open(os.path.join(output_dir, "comparison_table.md"), "w") as f:
        f.write("\n".join(md_lines))

    tex_lines = [
        "\\begin{tabular}{" + "l" + "c" * (len(headers) - 1) + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        tex_lines.append(" & ".join(row) + " \\\\")
    tex_lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(output_dir, "comparison_table.tex"), "w") as f:
        f.write("\n".join(tex_lines))

    # --- 柱状图 ---
    labels = list(all_results.keys())
    metric_names = ["agent_accuracy", "step_accuracy", "joint_accuracy"]
    metric_labels = ["Agent Acc", "Step Acc", "Joint Acc"]

    x = np.arange(len(metric_names))
    width = 0.8 / max(len(labels), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, label in enumerate(labels):
        m = all_results[label].get("metrics", {})
        vals = [m.get(mn, 0) for mn in metric_names]
        ax.bar(x + i * width, vals, width, label=label)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title("Ablation Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ablation_bar.png"), dpi=150)
    plt.close(fig)

    # --- 置信度校准图 ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for label, data in all_results.items():
        bins = data.get("metrics", {}).get("calibration_bins", [])
        if not bins:
            continue
        confs = [b["avg_confidence"] for b in bins if b["count"] > 0]
        accs = [b["accuracy"] for b in bins if b["count"] > 0]
        if confs:
            ax.plot(confs, accs, "o-", label=label)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "reliability_diagram.png"), dpi=150)
    plt.close(fig)

    # --- 分层对比图 ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    strat_keys = ["by_data_type", "by_history_length", "by_agent_count"]
    strat_titles = ["By Data Type", "By History Length", "By Agent Count"]
    for ax_i, (sk, st) in enumerate(zip(strat_keys, strat_titles)):
        ax = axes[ax_i]
        all_cats = set()
        for data in all_results.values():
            all_cats |= set(data.get("stratified_metrics", {}).get(sk, {}).keys())
        all_cats = sorted(all_cats)
        if not all_cats:
            continue
        x_s = np.arange(len(all_cats))
        w_s = 0.8 / max(len(labels), 1)
        for i, label in enumerate(labels):
            strat_data = all_results[label].get("stratified_metrics", {}).get(sk, {})
            vals = [strat_data.get(c, {}).get("agent_accuracy", 0) for c in all_cats]
            ax.bar(x_s + i * w_s, vals, w_s, label=label)
        ax.set_xticks(x_s + w_s * (len(labels) - 1) / 2)
        ax.set_xticklabels(all_cats, fontsize=7, rotation=15, ha='right')
        ax.set_ylim(0, 1)
        ax.set_title(st)
        ax.set_ylabel("Agent Accuracy")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "stratified_bar.png"), dpi=150)
    plt.close(fig)

    # --- 效率对比图 (双子图：LLM Calls + Latency 分开) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    x_e = np.arange(len(labels))
    calls = [all_results[l].get("metrics", {}).get("efficiency", {}).get("avg_llm_calls", 0) for l in labels]
    latencies = [all_results[l].get("metrics", {}).get("efficiency", {}).get("avg_latency_ms", 0) for l in labels]

    ax1.bar(x_e, calls, color='steelblue')
    ax1.set_xticks(x_e)
    ax1.set_xticklabels(labels, fontsize=7, rotation=15, ha='right')
    ax1.set_ylabel("Avg LLM Calls")
    ax1.set_title("LLM Calls per Sample")

    ax2.bar(x_e, latencies, color='darkorange')
    ax2.set_xticks(x_e)
    ax2.set_xticklabels(labels, fontsize=7, rotation=15, ha='right')
    ax2.set_ylabel("Avg Latency (ms)")
    ax2.set_title("Latency per Sample")

    fig.suptitle("Efficiency Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "efficiency_bar.png"), dpi=150)
    plt.close(fig)

    print(f"Visualization saved to {output_dir}/")


# ============================================================
# T4.6: 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark evaluator for feedback localization pipeline.")
    parser.add_argument("--test", action="store_true", help="Test mode: only process the first sample")
    parser.add_argument("--data_dir", type=str, help="Path to benchmark JSON directory")
    parser.add_argument("--is_handcrafted", type=str, default="False", choices=["True", "False"])
    parser.add_argument("--feedback_level", type=str, default="task_aware", choices=["blind", "generic", "task_aware"])
    parser.add_argument("--bypass_intent", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--prompt_variant", type=str, default="benchmark", choices=["default", "benchmark"])
    parser.add_argument("--output", type=str, default="results/output.json", help="Output JSON path")
    parser.add_argument("--visualize", action="store_true", help="Visualization mode")
    parser.add_argument("--result_files", nargs="+", help="Result JSON files for visualization")
    parser.add_argument("--output_dir", type=str, default="results/figures/", help="Output directory for figures")
    parser.add_argument("--eval_baseline", type=str, default=None,
                        help="Evaluate a baseline method output file (txt from inference.py)")
    parser.add_argument("--method_name", type=str, default="baseline",
                        help="Method name label for baseline evaluation")
    parser.add_argument("--test_metrics", action="store_true",
                        help="Run metric calculation correctness tests")

    args = parser.parse_args()

    if args.test_metrics:
        ok = test_metrics()
        sys.exit(0 if ok else 1)

    if args.eval_baseline:
        if not args.data_dir:
            print("Error: --data_dir required for --eval_baseline")
            sys.exit(1)
        evaluate_baseline(
            eval_file=args.eval_baseline,
            data_dir=args.data_dir,
            is_handcrafted=args.is_handcrafted == "True",
            method_name=args.method_name,
            output_path=args.output,
        )
        return

    if args.visualize:
        if not args.result_files:
            print("Error: --result_files required in --visualize mode")
            sys.exit(1)
        visualize_results(args.result_files, args.output_dir)
        return

    if not args.data_dir:
        print("Error: --data_dir required")
        sys.exit(1)

    asyncio.run(evaluate_all(
        data_dir=args.data_dir,
        is_handcrafted=args.is_handcrafted == "True",
        feedback_level=args.feedback_level,
        bypass_intent=args.bypass_intent == "True",
        prompt_variant=args.prompt_variant,
        output_path=args.output,
        test_mode=args.test,
    ))


if __name__ == "__main__":
    main()
