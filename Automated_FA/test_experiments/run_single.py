#!/usr/bin/env python3
"""
单独运行指定实验的指定文件，输出简洁结果。

用法:
    python run_single.py
"""

import os
import sys
import json
import time
import shutil
import asyncio
import tempfile
import contextlib
import io
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
AUTOMATED_FA_DIR = SCRIPT_DIR.parent
REPO_ROOT = AUTOMATED_FA_DIR.parent
FEEDBACK_CODE = REPO_ROOT / "feedback" / "feedback"

AG_SOURCE = REPO_ROOT / "Who_and_When" / "Algorithm-Generated"
HC_SOURCE = REPO_ROOT / "Who_and_When" / "Hand-Crafted"

sys.path.insert(0, str(AUTOMATED_FA_DIR))
sys.path.insert(0, str(FEEDBACK_CODE))

TASKS = [
    {"exp": "E2", "dataset": "HC", "file": "34.json",  "is_hc": True,  "type": "feedback", "feedback_level": "blind",      "bypass_intent": True},
    {"exp": "E1", "dataset": "HC", "file": "40.json",  "is_hc": True,  "type": "feedback", "feedback_level": "task_aware",  "bypass_intent": True},
    {"exp": "B1", "dataset": "AG", "file": "114.json", "is_hc": False, "type": "baseline", "method": "all_at_once",         "no_gt": False},
]


async def run_feedback_single(task):
    from benchmark_adapter import adapt_benchmark_sample
    from main_feedback import FeedbackPipeline
    from benchmark_evaluator import extract_prediction

    source = HC_SOURCE if task["is_hc"] else AG_SOURCE
    json_path = str(source / task["file"])

    sample = adapt_benchmark_sample(json_path, task["is_hc"], task["feedback_level"])
    pipeline = FeedbackPipeline()

    gt_for_llm = sample["task_ground_truth"] if task["feedback_level"] == "task_aware" else ""
    result = await pipeline.process_from_dicts(
        question=sample["question"],
        ground_truth=gt_for_llm,
        workflow_config=sample["workflow_config"],
        execution_trace=sample["execution_trace"],
        bypass_intent=task["bypass_intent"],
        prompt_variant="benchmark",
        feedback_level=task["feedback_level"],
        workflow_graph_str=sample.get("workflow_graph_str"),
        execution_trace_raw=sample.get("execution_trace_raw"),
    )

    if not result.get("success"):
        return {"error": result.get("error")}

    loc_result = result.get("localization_result", {})
    pred = extract_prediction(loc_result)

    return {
        "true_agent": sample["ground_truth"]["mistake_agent"],
        "true_step":  sample["ground_truth"]["mistake_step"],
        "pred_agent": pred["predicted_agent"],
        "pred_step":  pred["predicted_step"],
    }


def run_baseline_single(task):
    from Lib.utils import all_at_once as gpt_all_at_once
    from benchmark_evaluator import _parse_baseline_predictions
    import yaml
    from openai import OpenAI

    source = AG_SOURCE if not task["is_hc"] else HC_SOURCE
    json_path = source / task["file"]

    with open(json_path, "r", encoding="utf-8") as f:
        sample_data = json.load(f)

    true_agent = sample_data.get("mistake_agent", "")
    true_step = str(sample_data.get("mistake_step", ""))

    config_path = REPO_ROOT / "feedback" / "feedback" / "prompts" / "openai.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        api_config = yaml.safe_load(f)
    client = OpenAI(api_key=api_config["api_key"], base_url=api_config["api_base"])

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy2(str(json_path), os.path.join(tmpdir, task["file"]))

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            gpt_all_at_once(
                client=client,
                directory_path=tmpdir,
                is_handcrafted=task["is_hc"],
                model="gpt-5.4-nano",
                max_tokens=1024,
                no_ground_truth=task["no_gt"],
                test_mode=False,
                skip_files=set(),
            )
        output_text = buf.getvalue()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
            tf.write(output_text)
            tmp_log = tf.name

        try:
            baseline_preds = _parse_baseline_predictions(tmp_log)
        finally:
            os.unlink(tmp_log)

    bp = baseline_preds.get(task["file"], {})
    return {
        "true_agent": true_agent,
        "true_step":  true_step,
        "pred_agent": bp.get("predicted_agent"),
        "pred_step":  bp.get("predicted_step"),
    }


def format_result(task, res):
    ds = task["dataset"]
    fname = task["file"]
    exp = task["exp"]

    if "error" in res:
        return f"{ds}/{fname}: FAIL ({res['error']})"

    a_ok = "✓" if res["pred_agent"] == res["true_agent"] else "✗"
    s_ok = "✓" if str(res["pred_step"]) == str(res["true_step"]) else "✗"
    parsed = "✓" if res["pred_agent"] else "✗parse"
    return f"{ds}/{fname}: {parsed} Agent:{a_ok} Step:{s_ok}"


async def main():
    from loguru import logger
    logger.remove()

    for task in TASKS:
        exp = task["exp"]
        ds = task["dataset"]
        fname = task["file"]
        print(f"[{exp}] 正在运行 {ds}/{fname} ...", flush=True)

        t0 = time.time()
        try:
            if task["type"] == "feedback":
                res = await run_feedback_single(task)
            else:
                res = run_baseline_single(task)
        except Exception as e:
            res = {"error": str(e)}
        elapsed = time.time() - t0

        line = format_result(task, res)
        print(f"[{exp}] {line}  ({elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
