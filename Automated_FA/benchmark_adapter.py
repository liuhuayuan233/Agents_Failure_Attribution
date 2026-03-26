#!/usr/bin/env python3
"""
benchmark_adapter.py — 数据转换核心模块

将 Who&When benchmark JSON 转换为 feedback 系统可消费的格式：
(WorkflowConfig, ExecutionTrace, feedback_text)
"""

import json
import os
import sys
import argparse
from datetime import datetime


MAGENTIC_ONE_DESCRIPTIONS = {
    "Orchestrator": "中心协调者。理解任务、制定计划、分配子任务给执行 Agent、评估进展。trace 中 'Orchestrator (thought)' 为内部推理，'Orchestrator (-> X)' 为向 X 派发任务。",
    "WebSurfer": "网页浏览 Agent。搜索、点击链接、滚动页面、提取页面信息。",
    "Assistant": "通用助手。具备语言理解、Python 编程和命令行能力。",
    "FileSurfer": "文件处理 Agent。读取、搜索和分析本地文件。",
    "ComputerTerminal": "代码执行环境。运行 Python/Shell 并返回结果。",
}


def extract_agent_name(msg: dict, is_handcrafted: bool) -> str:
    """
    统一提取 agent 名称。
    - Algorithm-Generated: 取 msg["name"]
    - Hand-Crafted: 取 msg["role"]，并做以下规范化：
      - "Orchestrator (thought)" → "Orchestrator"
      - "Orchestrator (-> WebSurfer)" → "Orchestrator"
      - "Orchestrator (termination condition)" → "Orchestrator"
      - "human" → "human"
      - 其余原样返回
    """
    if not is_handcrafted:
        return msg.get("name", msg.get("role", "Unknown"))

    role = msg.get("role", "Unknown")
    if role.startswith("Orchestrator"):
        return "Orchestrator"
    if role == "human":
        return "human"
    return role


def _get_node_type(agent_name: str) -> str:
    name_lower = agent_name.lower().replace("_", "")
    if name_lower in ("computerterminal",):
        return "ExecutionNode"
    if name_lower == "orchestrator":
        return "OrchestratorNode"
    if name_lower == "human" or name_lower == "humaninput":
        return "HumanInputNode"
    return "AgentNode"


def convert_history_to_trace(data: dict, is_handcrafted: bool) -> dict:
    """
    将 benchmark JSON 的 history 转为 feedback 系统的 execution trace 格式。
    step 保持 0-indexed，与 benchmark 的 mistake_step 对齐。
    """
    history = data.get("history", [])
    execution = []

    for i, msg in enumerate(history):
        agent_name = extract_agent_name(msg, is_handcrafted)
        prev_content = history[i - 1].get("content", "") if i > 0 else ""

        step_entry = {
            "step": i,
            "node_name": agent_name,
            "node_type": _get_node_type(agent_name),
            "status": "completed",
            "inputs": {"context": prev_content} if prev_content else {},
            "outputs": {"response": msg.get("content", "")},
            "args_used": None,
        }
        execution.append(step_entry)

    workflow_name = "magentic_one_system" if is_handcrafted else "captain_agent_system"
    task_id = data.get("question_ID", "unknown")

    trace = {
        "workflow_name": workflow_name,
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "status": "failed",
        "total_duration_ms": None,
        "execution": execution,
    }
    return trace


def build_workflow_config(data: dict, is_handcrafted: bool) -> dict:
    """
    从 benchmark 数据构造 WorkflowConfig（dict 格式）。
    本质上是 Agent 角色清单，不构造 DAG 拓扑连接。
    """
    history = data.get("history", [])
    system_prompts = data.get("system_prompt", {})

    seen = set()
    agents = []
    for msg in history:
        name = extract_agent_name(msg, is_handcrafted)
        if name not in seen and name != "human":
            seen.add(name)
            agents.append(name)

    nodes = []
    for agent_name in agents:
        node_type = _get_node_type(agent_name)

        if is_handcrafted:
            description = MAGENTIC_ONE_DESCRIPTIONS.get(agent_name, f"{agent_name} agent")
            node = {
                "name": agent_name,
                "type": node_type,
                "description": description,
                "args": {
                    "orchestration": "centralized",
                },
                "outputs": None,
            }
        else:
            sys_prompt = system_prompts.get(agent_name, "")
            node = {
                "name": agent_name,
                "type": node_type,
                "args": {
                    "system_prompt": sys_prompt,
                    "orchestration": "group_chat",
                },
                "outputs": None,
            }
        nodes.append(node)

    if is_handcrafted:
        config = {
            "name": "magentic_one_system",
            "description": "Magentic-One 架构。Orchestrator 中心调度，按需将子任务分配给执行 Agent。",
            "nodes": nodes,
            "outputs": [],
        }
    else:
        config = {
            "name": "captain_agent_system",
            "description": "CaptainAgent 编排的多专家协作系统。Agent 之间通过群组对话协作，由 GroupChatManager 动态调度发言顺序。",
            "nodes": nodes,
            "outputs": [],
        }
    return config


def build_feedback_text(data: dict, level: str) -> str:
    """
    根据不同实验条件构造 feedback_text。
    - "generic": 不含 ground_truth（闭卷模式）
    - "task_aware": 含 ground_truth（开卷模式，与原论文等量信息）
    """
    question = data.get("question", "")
    ground_truth = data.get("ground_truth", "")

    if level == "generic":
        return (
            f"任务描述：{question}\n\n"
            f"该多 Agent 系统未能正确完成此任务。"
            f"请分析对话轨迹，找出是哪个 Agent 在哪个步骤犯了导致任务失败的关键错误。"
        )
    elif level == "task_aware":
        return (
            f"任务描述：{question}\n"
            f"正确答案：{ground_truth}\n\n"
            f"该多 Agent 系统未能给出正确答案。"
            f"请分析对话轨迹，找出是哪个 Agent 在哪个步骤犯了导致任务失败的关键错误。"
        )
    else:
        raise ValueError(f"Unknown feedback level: {level}. Must be 'generic' or 'task_aware'.")


def adapt_benchmark_sample(json_path: str, is_handcrafted: bool, feedback_level: str = "generic") -> dict:
    """
    读取单个 benchmark JSON，返回转换后的数据包。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    workflow_config = build_workflow_config(data, is_handcrafted)
    execution_trace = convert_history_to_trace(data, is_handcrafted)
    feedback_text = build_feedback_text(data, feedback_level)
    task_ground_truth = data.get("ground_truth", "")

    history = data.get("history", [])
    unique_agents = set()
    for msg in history:
        name = extract_agent_name(msg, is_handcrafted)
        if name != "human":
            unique_agents.add(name)

    return {
        "workflow_config": workflow_config,
        "execution_trace": execution_trace,
        "execution_trace_raw": execution_trace,
        "feedback_text": feedback_text,
        "question": data.get("question", ""),
        "task_ground_truth": task_ground_truth,
        "ground_truth": {
            "mistake_agent": data.get("mistake_agent", ""),
            "mistake_step": data.get("mistake_step", ""),
        },
        "meta": {
            "file": os.path.basename(json_path),
            "question": data.get("question", ""),
            "is_handcrafted": is_handcrafted,
            "history_length": len(history),
            "agent_count": len(unique_agents),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark adapter: convert Who&When JSON to feedback system format."
    )
    parser.add_argument("--test", action="store_true", help="Test mode: process one sample and print summary")
    parser.add_argument("--json_path", type=str, required=True, help="Path to benchmark JSON file")
    parser.add_argument("--is_handcrafted", type=str, default="False", choices=["True", "False"])
    parser.add_argument("--feedback_level", type=str, default="generic", choices=["generic", "task_aware"])

    args = parser.parse_args()
    is_handcrafted = args.is_handcrafted == "True"

    result = adapt_benchmark_sample(args.json_path, is_handcrafted, args.feedback_level)

    if args.test:
        summary = {
            "workflow_config": {
                "name": result["workflow_config"]["name"],
                "description": result["workflow_config"]["description"],
                "nodes": [
                    {"name": n["name"], "type": n["type"]}
                    for n in result["workflow_config"]["nodes"]
                ],
            },
            "execution_trace": {
                "workflow_name": result["execution_trace"]["workflow_name"],
                "task_id": result["execution_trace"]["task_id"],
                "status": result["execution_trace"]["status"],
                "num_steps": len(result["execution_trace"]["execution"]),
                "steps_preview": [
                    {
                        "step": s["step"],
                        "node_name": s["node_name"],
                        "node_type": s["node_type"],
                        "output_preview": (s["outputs"].get("response", "")[:100] + "...")
                        if len(s["outputs"].get("response", "")) > 100
                        else s["outputs"].get("response", ""),
                    }
                    for s in result["execution_trace"]["execution"]
                ],
            },
            "feedback_text": (result["feedback_text"][:200] + "...")
            if len(result["feedback_text"]) > 200
            else result["feedback_text"],
            "ground_truth": result["ground_truth"],
            "meta": result["meta"],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
