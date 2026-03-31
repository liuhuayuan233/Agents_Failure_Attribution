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
import yaml
from datetime import datetime
from collections import defaultdict


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
        step_entry = {
            "step": i,
            "node_name": agent_name,
            "node_type": _get_node_type(agent_name),
            "status": "completed",
            "inputs": {},
            "outputs": {"response": msg.get("content", "")},
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


def _build_trace_for_display(trace: dict) -> dict:
    """构建精简版 trace 用于 prompt 展示（去掉顶层元数据和冗余字段）。"""
    return {
        "execution": [
            {
                "step": step["step"],
                "node_name": step["node_name"],
                "node_type": step["node_type"],
                "output": step["outputs"].get("response", ""),
            }
            for step in trace["execution"]
        ]
    }


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


def _compress_step_list(steps: list) -> str:
    """将步骤列表压缩为范围字符串：[1,2,3,5,6,9] → '1-3, 5-6, 9'"""
    if not steps:
        return ""
    ranges = []
    start = steps[0]
    end = steps[0]
    for s in steps[1:]:
        if s == end + 1:
            end = s
        else:
            ranges.append(f"{start}-{end}" if end > start else str(start))
            start = end = s
    ranges.append(f"{start}-{end}" if end > start else str(start))
    return ", ".join(ranges)


def _content_preview(content: str, max_len: int = 1000) -> str:
    if not content:
        return ""
    return content[:max_len] + ("..." if len(content) > max_len else "")


def build_workflow_graph(data: dict, is_handcrafted: bool) -> str:
    """
    从对话历史中推导工作流架构，以类似 workflow.yaml 的格式表示。
    step 保持 0-indexed，与 trace 和 benchmark ground truth 一致。
    outputs 包含具体的传输内容摘要。
    agents_count 仅统计真正的 Agent（排除 ExecutionNode 和 HumanInputNode）。
    """
    history = data.get("history", [])
    MAX_EXAMPLES = 3

    seen_agents = set()
    agent_order = []
    agent_steps = defaultdict(list)

    for i, msg in enumerate(history):
        name = extract_agent_name(msg, is_handcrafted)
        if name == "human":
            continue
        agent_steps[name].append(i)
        if name not in seen_agents:
            seen_agents.add(name)
            agent_order.append(name)

    transition_map = defaultdict(list)
    chain = []
    for i, msg in enumerate(history):
        name = extract_agent_name(msg, is_handcrafted)
        if name == "human":
            continue
        chain.append(name)
        if i > 0:
            prev = extract_agent_name(history[i - 1], is_handcrafted)
            if prev != "human":
                transition_map[(prev, name)].append((i - 1, i))

    

    nodes = []
    for agent in agent_order:
        node = {
            "name": agent,
            "type": _get_node_type(agent),
            "appearances": len(agent_steps[agent]),
            "active_steps": _compress_step_list(agent_steps[agent]),
        }
        outputs = {}
        for (from_a, to_a), pairs in transition_map.items():
            if from_a == agent:
                key = to_a if to_a != agent else f"{to_a}(self)"
                examples = []
                for fs, ts in pairs[:MAX_EXAMPLES]:
                    preview = _content_preview(history[fs].get("content", ""))
                    examples.append(f"step {fs}→{ts}: {preview}")
                if len(pairs) > MAX_EXAMPLES:
                    examples.append(f"... 共{len(pairs)}次转换")
                outputs[key] = examples
        if outputs:
            node["outputs"] = outputs
        nodes.append(node)

    if len(chain) > 20:
        compressed = _compress_execution_chain(chain)
    else:
        compressed = " → ".join(chain)

    graph = {
        "total_steps": len(history),
        "nodes_count": len(agent_order),
        "nodes": nodes,
        "execution_chain": compressed,
    }
    return yaml.dump(graph, allow_unicode=True, default_flow_style=False, sort_keys=False)


def _compress_execution_chain(chain: list) -> str:
    """压缩长执行链，折叠重复的循环模式。"""
    if len(chain) <= 20:
        return " → ".join(chain)

    for period in range(2, min(len(chain) // 2 + 1, 8)):
        pattern = chain[:period]
        count = 0
        i = 0
        while i + period <= len(chain):
            if chain[i:i + period] == pattern:
                count += 1
                i += period
            else:
                break
        if count >= 3:
            remainder = chain[i:]
            pat_str = " → ".join(pattern)
            result = f"[{pat_str}] ×{count}"
            if remainder:
                if len(remainder) > 10:
                    result += f" → ... → {' → '.join(remainder[-5:])}"
                else:
                    result += f" → {' → '.join(remainder)}"
            return result

    head = " → ".join(chain[:10])
    tail = " → ".join(chain[-5:])
    return f"{head} → ... ({len(chain)} steps total) ... → {tail}"


def build_feedback_text(data: dict, level: str) -> str:
    """
    根据不同实验条件构造 feedback_text。
    - "blind": 不含任何额外信息（纯轨迹分析）
    - "generic": 含任务描述，不含 ground_truth
    - "task_aware": 含任务描述 + ground_truth（开卷模式，与原论文等量信息）
    """
    question = data.get("question", "")
    ground_truth = data.get("ground_truth", "")

    if level == "blind":
        return ""
    elif level == "generic":
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
        raise ValueError(f"Unknown feedback level: {level}. Must be 'blind', 'generic', or 'task_aware'.")


def adapt_benchmark_sample(json_path: str, is_handcrafted: bool, feedback_level: str = "generic") -> dict:
    """
    读取单个 benchmark JSON，返回转换后的数据包。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    workflow_config = build_workflow_config(data, is_handcrafted)
    execution_trace = convert_history_to_trace(data, is_handcrafted)
    feedback_text = build_feedback_text(data, feedback_level)
    workflow_graph_str = build_workflow_graph(data, is_handcrafted)
    task_ground_truth = data.get("ground_truth", "")

    history = data.get("history", [])
    unique_agents = set()
    for msg in history:
        name = extract_agent_name(msg, is_handcrafted)
        if name != "human":
            unique_agents.add(name)

    trace_display = _build_trace_for_display(execution_trace)

    return {
        "workflow_config": workflow_config,
        "workflow_graph_str": workflow_graph_str,
        "execution_trace": execution_trace,
        "execution_trace_raw": trace_display,
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
    parser.add_argument("--feedback_level", type=str, default="generic", choices=["blind", "generic", "task_aware"])

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
