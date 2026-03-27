# Benchmark 适配 TODO List

> **目标**：将 Who&When 数据集作为 benchmark，接入 `feedback` 仓库的定位算法，与原论文的 AutoFA 方法在统一指标下对比。
>
> **设计原则**：
> - 不修改 benchmark 原始数据
> - feedback 库只做最小改动，保持三阶段管线核心设计
> - 任务按依赖关系排列；完成到 **🏁 里程碑 1** 即可产出基础对比结果和可视化图表，后续任务为增量优化
>
> **关键发现**：原论文的三种 AutoFA 方法（All-at-Once / Step-by-Step / Binary Search）**全部**在 prompt 中传入了 `ground_truth`（正确答案）。即原论文方法是在 **知道正确答案** 的前提下定位错误，属于"开卷"模式。

---

## 阶段零：仓库组织

### ✅ T0. 将 feedback 仓库作为 git submodule 添加到评测仓库

**操作步骤**：

1. 在 `feedback` 仓库中，基于 `lhy` 分支创建 `benchmark` 分支：
   ```bash
   cd /Users/liuhuayuan/homework/feedback
   git checkout lhy
   git checkout -b benchmark
   git push origin benchmark
   ```

2. 在 `Agents_Failure_Attribution` 仓库中添加 submodule：
   ```bash
   cd /Users/liuhuayuan/homework/Agents_Failure_Attribution
   git submodule add -b benchmark <feedback仓库的git地址> feedback
   git add .gitmodules feedback
   git commit -m "Add feedback repo as submodule (benchmark branch)"
   ```

3. 后续所有对 feedback 库的改动（T2、T3）都在 submodule 内的 `benchmark` 分支上进行，不影响 `lhy` 分支的正常使用。

**验收标准**：`Agents_Failure_Attribution/feedback/` 目录存在且内容正确，`git submodule status` 显示正常。

**后续路径约定**：
- 评测仓库根目录：`Agents_Failure_Attribution/`
- feedback 子模块：`Agents_Failure_Attribution/feedback/`
- feedback 核心代码：`Agents_Failure_Attribution/feedback/feedback/`（即 submodule 内的 feedback 目录）
- 新建的适配代码：`Agents_Failure_Attribution/Automated_FA/`

---

## 阶段一：数据适配层（Adapter）

### ✅ T1. benchmark_adapter.py — 数据转换核心模块

**新建文件**：`Agents_Failure_Attribution/Automated_FA/benchmark_adapter.py`

**功能**：读取 benchmark JSON，输出 feedback 系统可消费的三元组 `(WorkflowConfig, ExecutionTrace, feedback_text)`。

**子任务**：

#### T1.1 Agent 名称提取函数 `extract_agent_name(msg, is_handcrafted)`

```python
def extract_agent_name(msg: dict, is_handcrafted: bool) -> str:
    """
    统一提取 agent 名称。
    - Algorithm-Generated: 取 msg["name"]
    - Hand-Crafted: 取 msg["role"]，并做以下规范化：
      - "Orchestrator (thought)" → "Orchestrator"
      - "Orchestrator (-> WebSurfer)" → "Orchestrator"
      - "Orchestrator (termination condition)" → "Orchestrator"
      - "human" → "human"
      - 其余原样返回（"WebSurfer", "Assistant", "FileSurfer", "ComputerTerminal"）
    """
```

**注意**：`mistake_agent` 在 Hand-Crafted 数据中用的是简化名（如 `"WebSurfer"`），所以规范化函数必须对齐这个粒度。

#### T1.2 History → ExecutionTrace 转换函数 `convert_history_to_trace(data, is_handcrafted)`

```python
def convert_history_to_trace(data: dict, is_handcrafted: bool) -> dict:
    """
    将 benchmark JSON 的 history 转为 feedback 系统的 execution trace 格式（dict，后续可 dump 为 YAML）。

    映射规则：
    - history[i] → execution[i]
    - step: i （保持 0-indexed，与 benchmark 的 mistake_step 对齐）
    - node_name: extract_agent_name(history[i])
    - node_type:
        - "Computer_terminal" / "ComputerTerminal" → "ExecutionNode"
        - "Orchestrator" → "OrchestratorNode"
        - "human" → "HumanInputNode"
        - 其余 → "AgentNode"
    - status: "completed"
    - inputs: {"context": history[i-1]["content"]} （前一条消息作为输入上下文，i=0 时为空）
    - outputs: {"response": history[i]["content"]}
    - args_used: None

    顶层元数据：
    - workflow_name: "captain_agent_system" 或 "magentic_one_system"
    - task_id: data["question_ID"] 或文件名
    - timestamp: 当前时间 ISO 格式
    - status: "failed"（benchmark 都是失败任务）
    - total_duration_ms: None
    """
```

**关键决策**：step 保持 0-indexed。这样预测输出的 step 与 `mistake_step` 直接可比，不需要额外映射。

#### T1.3 WorkflowConfig 构造函数 `build_workflow_config(data, is_handcrafted)`

```python
def build_workflow_config(data: dict, is_handcrafted: bool) -> dict:
    """
    从 benchmark 数据构造 WorkflowConfig（dict 格式）。
    本质上是 Agent 角色清单，不构造 DAG 拓扑连接。

    Algorithm-Generated:
    - 遍历 history 收集唯一 agent 名称
    - 每个 agent → 一个 node
    - node.type: "ExecutionNode"(Computer_terminal) / "AgentNode"(其余)
    - node.args.system_prompt: 取自 data["system_prompt"][agent_name]（如有）
    - node.args.orchestration: "group_chat"
    - 不设 outputs 连接
    - name: "captain_agent_system"
    - description: "CaptainAgent 编排的多专家协作系统。Agent 之间通过群组对话协作，由 GroupChatManager 动态调度发言顺序。"

    Hand-Crafted:
    - 遍历 history 收集唯一逻辑 agent（规范化后）
    - 预设的角色描述字典 MAGENTIC_ONE_DESCRIPTIONS：
      {
        "Orchestrator": "中心协调者。理解任务、制定计划、分配子任务给执行 Agent、评估进展。trace 中 'Orchestrator (thought)' 为内部推理，'Orchestrator (-> X)' 为向 X 派发任务。",
        "WebSurfer": "网页浏览 Agent。搜索、点击链接、滚动页面、提取页面信息。",
        "Assistant": "通用助手。具备语言理解、Python 编程和命令行能力。",
        "FileSurfer": "文件处理 Agent。读取、搜索和分析本地文件。",
        "ComputerTerminal": "代码执行环境。运行 Python/Shell 并返回结果。"
      }
    - 每个 agent → 一个 node，type/description 从上述字典取
    - node.args.orchestration: "centralized"
    - 不设 outputs 连接
    - name: "magentic_one_system"
    - description: "Magentic-One 架构。Orchestrator 中心调度，按需将子任务分配给执行 Agent。"
    """
```

#### T1.4 Feedback 文本构造函数 `build_feedback_text(data, level)`

```python
def build_feedback_text(data: dict, level: str) -> str:
    """
    根据不同实验条件构造 feedback_text。

    level 取值：

    - "generic": 不含 ground_truth 的闭卷模式（比原论文更难的设定）
      → f"任务描述：{data['question']}\n\n"
        f"该多 Agent 系统未能正确完成此任务。"
        f"请分析对话轨迹，找出是哪个 Agent 在哪个步骤犯了导致任务失败的关键错误。"

    - "task_aware": 含 ground_truth（与原论文等量信息，公平对比）
      → f"任务描述：{data['question']}\n"
        f"正确答案：{data['ground_truth']}\n\n"
        f"该多 Agent 系统未能给出正确答案。"
        f"请分析对话轨迹，找出是哪个 Agent 在哪个步骤犯了导致任务失败的关键错误。"
    """
```

**注意**：原论文的三种方法全部在 prompt 中传入了 `ground_truth`，因此 `task_aware` 是与原论文的公平对比条件。`generic` 是更严格的测试。

#### T1.5 总入口函数 `adapt_benchmark_sample(json_path, is_handcrafted, feedback_level)`

```python
def adapt_benchmark_sample(json_path: str, is_handcrafted: bool, feedback_level: str = "generic") -> dict:
    """
    读取单个 benchmark JSON，返回：
    {
        "workflow_config": {...},       # WorkflowConfig dict
        "execution_trace": {...},       # ExecutionTrace dict
        "execution_trace_raw": {...},   # 原始 trace dict（传给 localize 的 raw 参数）
        "feedback_text": "...",         # 合成的反馈文本
        "ground_truth": {               # 标注答案，用于评测
            "mistake_agent": "...",
            "mistake_step": "..."
        },
        "meta": {                       # 元数据
            "file": "23.json",
            "question": "...",
            "is_handcrafted": True/False,
            "history_length": 10,
            "agent_count": 4
        }
    }
    """
```

**验收标准**：

```bash
# --test 模式：只跑一条样本，验证转换逻辑
python benchmark_adapter.py --test --json_path ../Who_and_When/Algorithm-Generated/23.json --is_handcrafted False
python benchmark_adapter.py --test --json_path ../Who_and_When/Hand-Crafted/8.json --is_handcrafted True
```

输出三元组的 JSON 摘要（截断长文本），人工检查格式正确性。`--test` 不调用任何 LLM，纯数据转换。

---

### ✅ T2. feedback 子模块改动 — 支持 benchmark 模式

**改动位置**：`Agents_Failure_Attribution/feedback/feedback/`（submodule 内 benchmark 分支）

#### T2.1 FeedbackPipeline 增加 bypass_intent 参数

**文件**：`feedback/feedback/main_feedback.py`，修改 `FeedbackPipeline.process()` 方法签名：

```python
async def process(
    self,
    feedback_text: str,
    workflow_config_path: str,
    execution_trace_path: str,
    output_dir: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    audio_data: Optional[bytes] = None,
    audio_format: Optional[str] = None,
    image_data: Optional[bytes] = None,
    # ===== 新增参数 =====
    bypass_intent: bool = False,
    default_intent_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
```

当 `bypass_intent=True` 时：
- 跳过 `self.intent_service.recognize()` 调用
- 使用 `default_intent_result`（如果提供了），否则构造一个默认的 `IntentUnderstandingResult`：
  ```python
  IntentUnderstandingResult(
      raw_feedback=feedback_text,
      cleaned_feedback=feedback_text,
      feedback_type=FeedbackType.NATURAL_LANGUAGE,
      feedback_type_confidence=1.0,
      feedback_content=FeedbackContent(sentiment=-1.0),
      intent=Intent(
          operation=OperationType.EVALUATE,
          operation_confidence=1.0,
          target=IntentTarget(scope=ScopeType.WORKFLOW, confidence=1.0),
          attributes=[],
          reasoning="Benchmark mode: bypass intent recognition"
      ),
      clarification_needed=False,
  )
  ```
- 其余流程（定位 → 跳过升级）照旧
- 跳过正面评价检测（`_is_positive_evaluation`）

**注意**：不改变原有接口的默认行为，所有原有调用不受影响。

#### T2.2 支持直接传入 dict 而非文件路径

当前 `process()` 要求 `workflow_config_path` 和 `execution_trace_path` 是文件路径。为避免为每个 benchmark 样本临时写文件，增加一种直接传 dict 的方式。

**方案**：新增一个便捷方法 `process_from_dicts()`：

```python
async def process_from_dicts(
    self,
    feedback_text: str,
    workflow_config: dict,
    execution_trace: dict,
    bypass_intent: bool = False,
    default_intent_result: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    prompt_variant: str = "default",
) -> Dict[str, Any]:
    """
    直接接收 dict 而非文件路径。内部构造 WorkflowConfig/ExecutionTrace 对象后调用定位。
    跳过升级阶段（不需要 prompt 文件）。
    """
```

内部逻辑：
1. 用 `WorkflowConfig(**workflow_config)` / `ExecutionTrace(...)` 构造 Pydantic 对象
2. 如果 `bypass_intent`，用默认 IntentUnderstandingResult
3. 否则调用 `self.intent_service.recognize(text=feedback_text, context=execution_trace)`
4. 调用 `self.localization_service.localize(..., prompt_variant=prompt_variant)`
5. **不调用** `self.upgrade_service.upgrade(...)`（跳过升级）
6. 返回 `{"intent_result": ..., "localization_result": ..., "success": True}`

#### T2.3 定位服务支持选择 prompt 变体

**文件**：`feedback/feedback/service/workflow_localization.py`

在 `WorkflowLocalizationService.__init__()` 中增加 prompt 注册机制：

```python
def __init__(self):
    ...
    prompts_dir = Path(__file__).parent.parent / "prompts"
    self.prompt_paths = {
        "default": str(prompts_dir / "workflow_localization.hprompt"),
        "benchmark": str(prompts_dir / "workflow_localization_benchmark.hprompt"),
    }
```

在 `localize()` 方法中增加 `prompt_variant` 参数：

```python
async def localize(
    self,
    intent_result: IntentUnderstandingResult,
    workflow_config: WorkflowConfig,
    execution_trace: ExecutionTrace,
    execution_trace_raw: Optional[Dict[str, Any]] = None,
    prompt_variant: str = "default",  # ← 新增
) -> LocalizationResult:
```

在内部 `_call_llm()` 中使用 `self.prompt_paths[prompt_variant]` 加载对应 prompt 文件。

---

### ✅ T3. Benchmark 定位 Prompt

**新建文件**：`Agents_Failure_Attribution/feedback/feedback/prompts/workflow_localization_benchmark.hprompt`

> ⚠️ **核心要求**：此 prompt 必须以现有的 `workflow_localization.hprompt` 为基础进行改造，保留其核心分析方法论（客观评估优先、执行轨迹认知原则、定位问题区域、揭示根因等），因为这是定位算法的核心设计理念。在此基础上适配多 Agent 失败归因场景。

**与原 prompt 的继承关系**：

| 维度 | 原 `workflow_localization.hprompt` | 新 `benchmark` 变体 |
|------|-----------------------------------|-------------------|
| 角色 | 工作流架构评估与诊断专家 | 多 Agent 系统失败分析专家 |
| 输入 | 意图理解结果 + 工作流 DAG 配置 + 节点执行 trace | 意图理解结果 + Agent 角色清单 + 对话历史 trace |
| 分析方法论 | **保留**：客观评估优先、从执行轨迹还原实际流向、不要推测应该如何流动 | 同左 |
| 执行轨迹认知原则 | **保留**：轨迹是运行时事实记录、区分"节点可用信息"与"轨迹中出现的信息" | 适配为：区分"某 Agent 当时可获得的信息"与"对话中已出现的信息" |
| 问题分类 | STRUCTURAL / PROMPT_OR_CONFIG / INPUT_DEPENDENT / EXPECTATION_MISMATCH | 固定为 `AGENT_ERROR`（benchmark 场景不涉及架构评估） |
| 定位粒度 | target_scope = 节点名列表 | target_scope = Agent 名列表 + **error_step** + **ranked_candidates** |
| 根因分析 | runtime_anomalies + architectural_bottleneck + structural_contradiction | **保留**结构，用于分析 Agent 的错误行为 |
| refactoring_boundary | 结构性问题时必填 | 固定为 `null` |

**输出格式**（在 `LocalizationResult` JSON 结构上扩展）：

```json
{
  "assessment": {
    "problem_category": "AGENT_ERROR",
    "confidence": 0.85,
    "summary": "Church_Historian_Expert 在 step 3 提交的代码未导入必要的 Python 包..."
  },
  "localization": {
    "target_scope": ["Church_Historian_Expert"],
    "error_step": 3,
    "stage_context": "专家协作的代码编写阶段",
    "ranked_candidates": [
      {"agent": "Church_Historian_Expert", "step": 3, "confidence": 0.82, "reason": "..."},
      {"agent": "DataVerification_Expert", "step": 5, "confidence": 0.45, "reason": "..."},
      {"agent": "Art_Historian_Expert", "step": 1, "confidence": 0.15, "reason": "..."}
    ]
  },
  "root_cause_analysis": {
    "runtime_anomalies": "...",
    "architectural_bottleneck": "NONE",
    "structural_contradiction": "..."
  },
  "refactoring_boundary": null
}
```

**Prompt 设计要点**：
- 从 `workflow_localization.hprompt` 的 system prompt 出发改造，保留"核心原则"章节的分析方法论
- `ranked_candidates` 必须覆盖 trace 中所有逻辑 Agent（排除 `human`），按 confidence 降序
- 每个 candidate 必须给出具体的 `step`（0-indexed）和 `reason`
- `target_scope[0]` 和 `error_step` 代表 Top-1 预测
- 强调"step 编号是对话 history 数组中的 0-indexed 位置"
- 保留原 prompt 的 `$user$` 部分模板变量结构（`%intent_info%`、`%workflow_config%`、`%execution_trace%`）
- 提供 1-2 个 few-shot 示例（可从 Algorithm-Generated 中挑选典型 case）

---

### ✅ T3.5 修复原论文代码已知 Bug（ISSUES.md）

**改动文件**：
- `Automated_FA/evaluate.py`
- `Automated_FA/Lib/utils.py`
- `Automated_FA/Lib/local_model.py`

**Bug 1（问题三）**：`evaluate.py` 第 88-91 行，`in` 改为 `==`：

```python
# 修复前
if actual_agent in pred['predicted_agent']:
if actual_step in pred['predicted_step']:

# 修复后
if actual_agent == pred['predicted_agent']:
if actual_step == pred['predicted_step']:
```

**Bug 2（问题四）**：Step-by-Step 未找到错误时无 `Prediction for` 输出。
- `utils.py` 第 181-183 行、`local_model.py` 第 208-209 行
- 改为：未找到错误时仍输出 `Prediction for {json_file}:` 格式，Agent Name 设为最后一个发言的 Agent，Step Number 设为最后一步（兜底预测），或标记为 `NONE`

**Bug 3（问题五）**：`local_model.py` 第 231 行 Binary Search 输出格式 `Prediction for {json_file} (Binary Search Result):` 与评测正则不匹配。
- 改为 `Prediction for {json_file}:`（去掉 ` (Binary Search Result)` 后缀）

**验收标准**：修复后用原论文方法跑一遍 `evaluate.py`，确认无 Warning 输出（"Could not parse Agent Name/Step Number"）。

---

### ✅ T3.6 原论文推理代码增加 `--no_ground_truth` 开关

**改动文件**：
- `Automated_FA/inference.py`
- `Automated_FA/Lib/utils.py`
- `Automated_FA/Lib/local_model.py`

**目的**：原论文三种方法全部在 prompt 中传入了 `ground_truth`。为了做公平的闭卷对比，需要支持不传 ground_truth 的模式。

**步骤 1**：`inference.py` 新增命令行参数：

```python
parser.add_argument(
    "--no_ground_truth",
    action="store_true",
    default=False,
    help="不在 prompt 中传入正确答案（闭卷模式）"
)
```

并将 `args.no_ground_truth` 传递给各方法函数。

**步骤 2**：`Lib/utils.py` 中三个方法（`all_at_once`、`step_by_step`、`binary_search`）各有一行：

```python
f"The Answer for the problem is: {ground_truth}\n"
```

改为条件判断：

```python
gt_line = f"The Answer for the problem is: {ground_truth}\n" if not no_ground_truth else ""
```

共 3 处改动（每个方法 1 处）。函数签名加 `no_ground_truth: bool = False` 参数。

**步骤 3**：`Lib/local_model.py` 中同样有 3 处（`analyze_all_at_once_local`、`analyze_step_by_step_local`、`_construct_binary_search_prompt_local`），做相同的条件判断改动。

**验收标准**：
```bash
# --test 模式：只跑第一个 JSON 文件，验证 prompt 拼接是否正确
# 原有行为不变（开卷）
python inference.py --method all_at_once --model gpt-4o --test ...
# 新增闭卷模式
python inference.py --method all_at_once --model gpt-4o --no_ground_truth --test ...
```

`--test` 参数：只处理目录下的第一个 JSON 文件，输出完整 prompt 到 stdout 后退出（不实际调用 LLM API），用于检查 prompt 中是否包含/不包含 "The Answer for the problem is" 这句话。

---

## 阶段二：评测框架

### ✅ T4. benchmark_evaluator.py — 统一评测脚本

**新建文件**：`Agents_Failure_Attribution/Automated_FA/benchmark_evaluator.py`

**功能**：一键运行 feedback 管线 + 计算全部指标 + 输出可视化图表。

#### T4.1 基础评测流程

```python
async def evaluate_all(
    data_dir: str,           # benchmark 数据目录
    is_handcrafted: bool,
    feedback_level: str,     # "generic" / "task_aware"
    bypass_intent: bool,
    prompt_variant: str,     # "benchmark"
    output_path: str,        # 结果 JSON 输出路径
):
    """
    1. 遍历 data_dir 下所有 .json 文件
    2. 对每个文件调用 benchmark_adapter.adapt_benchmark_sample()
    3. 调用 FeedbackPipeline.process_from_dicts()
    4. 从 LocalizationResult 中提取预测结果
    5. 记录每个样本的 LLM 调用次数和 token 消耗
    6. 与 ground_truth 比对，累计各指标
    7. 输出结果 JSON + 可视化图表
    """
```

#### T4.2 预测结果提取函数 `extract_prediction(localization_result)`

```python
def extract_prediction(localization_result: dict) -> dict:
    """
    从 LocalizationResult 中提取预测：
    
    返回:
    {
        "predicted_agent": localization_result["localization"]["target_scope"][0],
        "predicted_step": str(localization_result["localization"]["error_step"]),
        "confidence": localization_result["assessment"]["confidence"],
        "ranked_candidates": localization_result["localization"].get("ranked_candidates", [])
    }
    
    如果 target_scope 为空或字段缺失，返回 predicted_agent=None, predicted_step=None。
    """
```

#### T4.3 指标计算模块

实现以下指标函数，所有指标接收 `List[dict]` 格式的预测-真值对列表：

```python
predictions = [
    {
        "file": "23.json",
        "true_agent": "Church_Historian_Expert",
        "true_step": 3,
        "pred_agent": "Church_Historian_Expert",
        "pred_step": 3,
        "confidence": 0.82,
        "ranked_candidates": [...],
        "llm_calls": 1,
        "total_input_tokens": 5200,
        "total_output_tokens": 800,
        "latency_ms": 3500
    },
    ...
]
```

**基础指标**：

| 函数 | 指标 | 计算方法 |
|------|------|---------|
| `agent_accuracy(preds)` | Agent Accuracy | `pred_agent == true_agent` 的比例 |
| `step_accuracy(preds)` | Step Accuracy | `pred_step == true_step` 的比例（精确匹配，修复原 `in` bug） |
| `joint_accuracy(preds)` | Joint Accuracy | Agent 和 Step **同时**正确的比例 |

**排名指标**（需要 `ranked_candidates`）：

| 函数 | 指标 | 计算方法 |
|------|------|---------|
| `topk_agent_accuracy(preds, k)` | Top-k Agent Acc | `true_agent` 在 `ranked_candidates[:k]` 中任一 candidate 的 agent 匹配 |
| `topk_step_accuracy(preds, k)` | Top-k Step Acc | 同上但匹配 step |
| `mrr_agent(preds)` | MRR (Agent) | 对每个样本，找 `true_agent` 在 ranked_candidates 中首次出现的位置 r，计算 1/r 的平均值 |
| `mrr_step(preds)` | MRR (Step) | 同上但匹配 (agent, step) pair |

**容错指标**：

| 函数 | 指标 | 计算方法 |
|------|------|---------|
| `step_mae(preds)` | Step MAE | `mean(\|pred_step - true_step\|)`，仅计算 Agent 正确的样本 |
| `step_window_accuracy(preds, w)` | Step Acc@w | `\|pred_step - true_step\| <= w` 的比例 |

**置信度校准**：

| 函数 | 指标 | 计算方法 |
|------|------|---------|
| `confidence_calibration(preds, n_bins)` | ECE | 将样本按 confidence 分桶，计算每桶 \|avg_confidence - accuracy\| 加权平均 |

纯代码计算，输入是已有的 predictions 列表：
1. 将所有样本按 `confidence` 分为 `n_bins=10` 个桶
2. 每个桶内计算平均 confidence 和实际 accuracy（Agent 是否正确）
3. ECE = 各桶 |confidence - accuracy| × (桶内样本数 / 总样本数) 的加权平均
4. 输出 ECE 数值 + 每桶的 `(avg_confidence, accuracy, count)` 列表（用于画 Reliability Diagram）

**效率统计**：

| 函数 | 指标 | 计算方法 |
|------|------|---------|
| `efficiency_stats(preds)` | LLM 调用效率 | 汇总 llm_calls / input_tokens / output_tokens / latency 的平均值 |

在每个样本的 `process_from_dicts()` 调用过程中记录 LLM 返回的 usage 信息（`handyllm` 的响应中包含 token 消耗），以及调用次数和耗时。汇总后与原论文方法对比：
- All-at-Once: 1 次 / 样本
- Step-by-Step: O(n) 次 / 样本（n = history 长度）
- Binary Search: O(log n) 次 / 样本
- feedback (bypass): 1 次 / 样本
- feedback (full): 2 次 / 样本

**分层分析**：

```python
def stratified_metrics(preds, compute_fn):
    """
    按多个维度分组计算指标：
    - by_data_type: algorithm_generated / hand_crafted
    - by_history_length: short(1-5) / medium(6-8) / long(9+)
    - by_agent_count: 2 / 3 / 4
    """
```

#### T4.4 结果输出格式

输出 JSON：

```json
{
  "config": {
    "data_dir": "...",
    "is_handcrafted": false,
    "feedback_level": "task_aware",
    "bypass_intent": true,
    "prompt_variant": "benchmark",
    "total_samples": 126,
    "successful_predictions": 120,
    "failed_predictions": 6
  },
  "metrics": {
    "agent_accuracy": 0.65,
    "step_accuracy": 0.42,
    "joint_accuracy": 0.38,
    "topk_agent_accuracy": {"top1": 0.65, "top3": 0.82},
    "topk_step_accuracy": {"top1": 0.42, "top3": 0.61},
    "mrr_agent": 0.72,
    "mrr_step": 0.51,
    "step_mae": 1.8,
    "step_window_accuracy": {"w0": 0.42, "w1": 0.58, "w2": 0.68},
    "ece": 0.08,
    "calibration_bins": [...],
    "efficiency": {
      "avg_llm_calls": 1.0,
      "avg_input_tokens": 5200,
      "avg_output_tokens": 800,
      "avg_latency_ms": 3500
    }
  },
  "stratified_metrics": {
    "by_data_type": { ... },
    "by_history_length": { ... },
    "by_agent_count": { ... }
  },
  "per_sample": [ ... ]
}
```

#### T4.5 可视化模块

在 `benchmark_evaluator.py` 内或拆为单独的 `visualize_results.py`，读取多组实验结果 JSON，输出图表到 `results/figures/`：

| 图表 | 说明 | 用途 |
|------|------|------|
| **主对比表** | 所有方法 × 所有指标（LaTeX / Markdown 格式） | 报告核心表格 |
| **消融柱状图** | E0 → E1（→ E2）各指标的变化 | 展示消融分析 |
| **置信度校准图** | Reliability Diagram（置信度 vs 实际准确率） | 展示校准质量 |
| **分层对比图** | 按数据类型 / 对话长度 / Agent 数分组的柱状图 | 展示场景差异 |
| **效率对比图** | 各方法的 LLM 调用次数 / Token 消耗对比 | 展示效率优势 |

使用 matplotlib / seaborn。

#### T4.6 命令行入口

```bash
# --test 模式：只跑一条样本（Algorithm-Generated/23.json），验证整条管线端到端跑通
# 会实际调用 LLM API（1 次），输出单样本的预测结果 + 全部指标（基于 1 条数据）
python benchmark_evaluator.py \
  --test \
  --data_dir ../Who_and_When/Algorithm-Generated \
  --is_handcrafted False \
  --feedback_level task_aware \
  --bypass_intent True \
  --prompt_variant benchmark \
  --output results/test_single.json

# 正式运行：跑完整个数据集
python benchmark_evaluator.py \
  --data_dir ../Who_and_When/Algorithm-Generated \
  --is_handcrafted False \
  --feedback_level task_aware \
  --bypass_intent True \
  --prompt_variant benchmark \
  --output results/feedback_task_aware_ag.json

# 可视化（读取多组结果 JSON 生成对比图表）
python benchmark_evaluator.py --visualize \
  --result_files results/feedback_generic_ag.json results/feedback_task_aware_ag.json \
  --output_dir results/figures/
```

`--test` 参数：只处理数据目录中的第一个 JSON 文件。会实际调用 LLM（验证管线端到端），但只消耗 1 次 API 调用。输出完整的结果 JSON（含指标，但基于单样本所以指标值仅供验证格式）。

---

## 🏁 里程碑 1：基础对比 + 可视化

**完成 T0-T4 后**，你可以运行以下实验并得到完整对比结果和图表：

### 实验矩阵

**开卷组（传 ground_truth，原论文默认设定）**：

| 编号 | 方法 | 命令概要 |
|------|------|---------|
| B1 | 原论文 All-at-Once | `inference.py --method all_at_once` |
| B2 | 原论文 Step-by-Step | `inference.py --method step_by_step` |
| B3 | 原论文 Binary Search | `inference.py --method binary_search` |
| E1 | feedback (task_aware, bypass intent) | `benchmark_evaluator.py --feedback_level task_aware --bypass_intent True` |

**闭卷组（不传 ground_truth）**：

| 编号 | 方法 | 命令概要 |
|------|------|---------|
| B4 | 原论文 All-at-Once (no GT) | `inference.py --method all_at_once --no_ground_truth` |
| B5 | 原论文 Step-by-Step (no GT) | `inference.py --method step_by_step --no_ground_truth` |
| B6 | 原论文 Binary Search (no GT) | `inference.py --method binary_search --no_ground_truth` |
| E0 | feedback (generic, bypass intent) | `benchmark_evaluator.py --feedback_level generic --bypass_intent True` |

每个实验在 Algorithm-Generated（126 条）和 Hand-Crafted（58 条）上各跑一次，共 **8 × 2 = 16 组**。

### 对比维度

| 对比 | 实验对 | 分析意义 |
|------|--------|---------|
| 公平开卷对比 | E1 vs B1/B2/B3 | feedback 方法 vs 原论文方法（同等信息量） |
| 公平闭卷对比 | E0 vs B4/B5/B6 | feedback 方法 vs 原论文方法（同等信息量） |
| ground_truth 的增益 | E1 vs E0, B1 vs B4, B2 vs B5, B3 vs B6 | 知道正确答案对各方法的提升幅度 |

产出：
- 全部指标（Agent Acc、Step Acc、Joint Acc、Top-k、MRR、Step MAE、Step Window、ECE、效率）
- 可视化图表（对比表、柱状图、校准图、分层图、效率图）

**此时即可选择停止，已有完整的对比数据和图表可用于报告。**

---

## 阶段三：消融实验（可选，增强实验完整性）

### T5. 实验 E2 — 完整管线 + 合成反馈

在 T4 基础上运行 `--bypass_intent False --feedback_level task_aware`。

与 E1 对比差异 = **意图识别阶段的增益**。

**额外工作**：无，T2.1 已支持 bypass_intent=False。

---

## 🏁 里程碑 2：消融实验

**完成 T5 后**，在里程碑 1 的 8 组实验基础上增加 E2，可以构建 feedback 方法的内部消融表：

| 实验 | 意图识别 | 传 ground_truth | 消融意义 |
|------|---------|----------------|---------|
| E0 | 跳过 | ❌ | 闭卷 baseline |
| E1 | 跳过 | ✅ | E1-E0 = 知道正确答案的增益 |
| E2 | 完整 | ✅ | E2-E1 = 意图识别阶段的增益 |

**此时可选择停止。**

---

## 附录：任务依赖关系

```
T0 (git submodule)
  │
  ↓
T1 (Adapter)
 ├── T1.1 Agent 名称提取
 ├── T1.2 History → Trace
 ├── T1.3 WorkflowConfig 构造
 ├── T1.4 Feedback 文本构造
 └── T1.5 总入口

T2 (Pipeline 改动)          T3 (Benchmark Prompt)     T3.5 (修复原论文 Bug)
 ├── T2.1 bypass_intent      ↑ 必须参考原 prompt         │ evaluate.py in→==
 ├── T2.2 process_from_dicts │                           │ Step-by-Step 兜底输出
 └── T2.3 prompt_variant     │                           │ Binary Search 格式修复
         │                   │                           │
         │                   │                  T3.6 (原论文 --no_ground_truth)
         │                   │                           │
         └────────┬──────────┘                           │
                  │                                      │
                  └──────────────┬────────────────────────┘
                                 ↓
                  T4 (评测框架 + 可视化)
                   ├── T4.1 评测流程
                   ├── T4.2 预测提取
                   ├── T4.3 指标计算（含校准 + 效率 + 分层）
                   ├── T4.4 结果输出
                   ├── T4.5 可视化图表
                   └── T4.6 命令行
                                 │
                   实验: E0, E1, B1-B6 (开卷+闭卷 共 8 组 × 2 数据集)
                                 │
                  ═══════════════╪═══════════  🏁 里程碑 1（基础对比 + 图表）
                                 │
                  T5 (E2 消融: +意图识别)
                                 │
                  ═══════════════╪═══════════  🏁 里程碑 2（消融实验）
```
