# Benchmark 适配问题分析

## 目标

将 `Agents_Failure_Attribution` 的 Who&When 数据集作为 benchmark，使用 `feedback` 仓库的定位算法（含意图理解）替代原有的 AutoFA 方法进行评测。

---

## 问题一览

| # | 问题 | 类别 | 严重程度 |
|---|------|------|---------|
| 1 | 轨迹数据格式完全不同 | 数据格式 | 🔴 |
| 2 | feedback 系统要求 workflow_config，benchmark 无此概念 | 缺失输入 | 🔴 |
| 3 | feedback 系统依赖 feedback_text 驱动，benchmark 无用户反馈文本 | 缺失输入 | 🔴 |
| 4 | 定位输出格式与评测指标不对齐 | 输出格式 | 🔴 |
| 5 | Agent（角色对话）vs Node（节点执行）语义模型不同 | 概念模型 | 🟡 |
| 6 | 两种 benchmark 数据的 agent 标识字段不一致 | 数据格式 | 🟡 |
| 7 | Step 编号语义不同 | 数据格式 | 🟡 |
| 8 | benchmark 数据无 prompt 文件，upgrade 阶段无法执行 | 缺失输入 | 🟢 |
| 9 | 评测脚本本身存在的已知 Bug | 评测框架 | 🟡 |

---

## 问题 1：轨迹数据格式完全不同 🔴

### Benchmark 的轨迹格式（多 Agent 对话历史）

```json
{
  "history": [
    {
      "content": "I will look up the portrait...",
      "role": "assistant",
      "name": "Art_Historian_Expert"
    },
    {
      "content": "exitcode: 1 (execution failed)...",
      "name": "Computer_terminal",
      "role": "user"
    }
  ]
}
```

- 本质是**扁平化的对话消息列表**
- 每条消息只有 `content`、`role`、`name`（或仅 `role`）三个字段
- 没有结构化的输入/输出/参数信息
- 没有执行时间、节点类型、执行状态等元数据

### Feedback 系统期望的轨迹格式（工作流执行 trace）

```yaml
execution:
- step: 1
  node_name: node_requirement_parser
  node_type: QueryLLMNode
  status: completed
  duration_ms: 5204
  inputs:
    user_request: "..."
  outputs:
    parsed_requirements: "{...}"
  args_used:
    hprompt: hprompt/node_requirement_parser.hprompt
    model: gemini-2.5-flash
    temperature: 0.3
```

- 结构化的**节点执行记录**
- 每个节点有明确的 `node_name`、`node_type`、`status`、`inputs`、`outputs`、`args_used`
- 包含 `workflow_name`、`task_id`、`timestamp`、`total_duration_ms` 等顶层元数据

### 差异核心

| 维度 | Benchmark | Feedback |
|------|-----------|----------|
| 粒度 | 每条 agent 发言 | 每个工作流节点执行 |
| 内容 | 自然语言文本 | 结构化输入/输出 JSON |
| 元数据 | 无 | 节点类型/状态/耗时/参数 |
| 格式 | JSON 数组 | YAML 结构 |

**需要编写转换层**，将 benchmark 的 `history` 扁平对话列表转换为 feedback 系统能接受的 `ExecutionTrace` 格式。

---

## 问题 2：Feedback 系统要求 workflow_config，benchmark 无此概念 🔴

Feedback 的定位服务 `WorkflowLocalizationService.localize()` 需要三个输入：

```python
async def localize(
    self,
    intent_result: IntentUnderstandingResult,
    workflow_config: WorkflowConfig,           # ← 必须
    execution_trace: ExecutionTrace,
    execution_trace_raw: Optional[Dict] = None
) -> LocalizationResult:
```

其中 `WorkflowConfig` 定义了工作流的节点拓扑：

```python
class WorkflowConfig(BaseModel):
    name: str
    description: Optional[str]
    nodes: List[WorkflowNodeConfig]  # 每个节点的 name, type, args, outputs
    outputs: Optional[List[Dict]]
```

**Benchmark 数据完全没有工作流配置的概念**。

- Algorithm-Generated 数据有 `system_prompt`（各 agent 的 system prompt），但没有节点拓扑、连接关系、参数配置
- Hand-Crafted 数据甚至连 `system_prompt` 都没有

需要从 benchmark 数据中**构造或模拟**出 `WorkflowConfig`。

---

## 问题 3：Feedback 系统依赖 feedback_text 驱动，benchmark 无用户反馈文本 🔴

Feedback 管线的入口是 `FeedbackPipeline.process()`：

```python
async def process(
    self,
    feedback_text: str,              # ← 必须：用户反馈文本
    workflow_config_path: str,
    execution_trace_path: str,
    ...
) -> Dict[str, Any]:
```

**第一阶段（意图识别）完全依赖 feedback_text**：

```python
intent_result = await self.intent_service.recognize(
    text=feedback_text,  # ← 用户的自然语言反馈
    ...
)
```

然而 benchmark 的任务定义是：**给定一个失败的多 Agent 对话轨迹，直接从轨迹中找出错误的 Agent 和 Step**。这里**没有"用户反馈"**的概念。

- Benchmark 的 `question` 是原始任务描述，不是对执行结果的反馈
- Benchmark 的 `ground_truth` 是任务的正确答案，不是用户对工作流的评价
- Benchmark 的 `mistake_reason` 是标注者的错误说明，不是用户的反馈输入

需要决定：是**构造合成反馈文本**（如"任务失败了，请定位错误"），还是**改造管线使意图识别阶段可选**。

---

## 问题 4：定位输出格式与评测指标不对齐 🔴

### Benchmark 的评测指标

评测脚本 `evaluate.py` 期望的预测输出是：

```
Agent Name: Church_Historian_Expert
Step Number: 3
```

Ground truth 是：

```json
{
    "mistake_agent": "Church_Historian_Expert",
    "mistake_step": "3"
}
```

评测计算 **Agent Accuracy** 和 **Step Accuracy**（精确的 agent 名 + step 编号匹配）。

### Feedback 系统的定位输出

`LocalizationResult` 的结构是：

```python
class LocalizationResult(BaseModel):
    assessment: Dict[str, Any]       # {"problem_category": "...", "confidence": 0.8, "summary": "..."}
    localization: Dict[str, Any]     # {"target_scope": ["node_xxx", "node_yyy"], "stage_context": "..."}
    root_cause_analysis: Dict[str, Any]
    refactoring_boundary: Optional[Dict[str, Any]]
```

差异对比：

| 维度 | Benchmark 评测 | Feedback 输出 |
|------|---------------|--------------|
| Agent/节点 | 单一 agent 名称 | `target_scope` 列表（可能多个节点） |
| Step 编号 | 精确的数字编号 | 无直接 step 编号，只有节点名称 |
| 错误原因 | `mistake_reason`（标注参考） | `root_cause_analysis`（LLM 生成） |
| 问题分类 | 无 | `problem_category`（PROMPT/PARAMETER/STRUCTURAL 等） |
| 置信度 | 无 | `confidence` (0-1) |

**需要编写映射层**，将 `LocalizationResult` 的 `target_scope` 映射回 agent 名称和 step 编号。

---

## 问题 5：Agent（角色对话）vs Node（节点执行）语义模型不同 🟡

两个系统对"执行单元"的建模完全不同：

### Benchmark：多 Agent 对话系统

- 一个 **Agent** 可以发言多次（同一个 Agent 在 history 中多次出现）
- **Step** 是对话轮次的编号（第 0 次发言、第 1 次发言...）
- 一个 Agent 的不同 step 内容可以完全不同（前一次提问，后一次写代码）
- 存在特殊 agent 如 `Computer_terminal`（执行代码的环境）、`Orchestrator (thought)`（内部思考）

### Feedback：工作流节点 DAG

- 一个 **Node** 通常只执行一次
- **Step** 是节点在 DAG 中的拓扑排序编号
- 每个 Node 有固定的类型（QueryLLMNode / SwitchNode）和固定的 prompt
- 节点之间有明确的连接关系（输出端口 → 输入端口）

**核心矛盾**：Benchmark 中同一个 Agent 的多次发言分布在不同 step，但 feedback 系统的定位粒度是"节点"而非"节点的某次调用"。如果将每个 Agent 的每次发言视为一个独立节点，则无法体现同一 Agent 的角色连续性；如果将整个 Agent 视为一个节点，则丢失了 step 层面的定位精度。

---

## 问题 6：两种 benchmark 数据的 agent 标识字段不一致 🟡

- **Algorithm-Generated** 数据：agent 名称存储在 `name` 字段

```json
{"content": "...", "role": "user", "name": "Church_Historian_Expert"}
```

- **Hand-Crafted** 数据：agent 名称存储在 `role` 字段

```json
{"content": "...", "role": "WebSurfer"}
{"content": "...", "role": "Orchestrator (thought)"}
{"content": "...", "role": "Orchestrator (-> WebSurfer)"}
```

Hand-Crafted 数据的 `role` 字段还包含复合角色（如 `Orchestrator (thought)`、`Orchestrator (-> WebSurfer)`），而 `mistake_agent` 只用简单名称（如 `WebSurfer`）。

转换层需要统一处理这两种格式的 agent 名称提取逻辑。

---

## 问题 7：Step 编号语义不同 🟡

- **Benchmark**：step 编号是对话消息的 **0-indexed 位置**（history 数组中的下标）
- **Feedback trace**：step 编号是工作流节点的 **1-indexed 执行序号**

此外，benchmark 中的 step 包含所有 agent 的发言（包括 `Computer_terminal` 的执行结果输出），而 feedback 中的 step 只包含实际的处理节点。

如果将 benchmark 数据转换为 feedback 格式，需要明确 step 编号映射规则。

---

## 问题 8：Benchmark 数据无 prompt 文件，upgrade 阶段无法执行 🟢

Feedback 系统的完整管线是 **意图识别 → 定位 → 升级**。升级阶段 `WorkflowUpgradeService.upgrade()` 需要：

1. 加载目标节点关联的 `.hprompt` 提示词文件
2. 生成修改后的提示词 / 工作流配置文件

Benchmark 数据没有 hprompt 文件（Algorithm-Generated 有 `system_prompt` 字段，但格式不同）。

**但这个问题影响较小**：评测只需要运行到定位阶段即可，升级阶段可以跳过。

---

## 问题 9：评测脚本本身存在的已知 Bug 🟡

参见 `ISSUES.md`，原评测框架存在以下问题，在适配新算法时需一并修复：

1. **本地模型 Binary Search 输出格式不匹配**：正则解析失败
2. **Step/Agent 匹配用 `in` 而非 `==`**：可能导致准确率虚高（如 `"3" in "13"` 为 True）
3. **推理与评测是两个独立脚本**：无法一键运行

---

## 适配工作总结

要实现两个系统的对接，至少需要完成以下适配工作：

### 必须解决（否则完全无法运行）

| 工作项 | 说明 |
|--------|------|
| **数据转换层** | 将 benchmark 的 `history` 对话列表转换为 `ExecutionTrace` 格式 |
| **WorkflowConfig 构造** | 从 benchmark 数据中提取/构造出 `WorkflowConfig`（节点列表、类型等） |
| **反馈文本构造策略** | 决定如何为 benchmark 样本生成 `feedback_text`，或使意图识别阶段可选 |
| **输出结果映射** | 将 `LocalizationResult.localization.target_scope` 映射为 `Agent Name` + `Step Number` |

### 建议解决（提升评测质量）

| 工作项 | 说明 |
|--------|------|
| **统一 agent 标识提取** | 统一 Algorithm-Generated（`name` 字段）和 Hand-Crafted（`role` 字段）的 agent 名称提取 |
| **Step 编号对齐** | 建立 benchmark step（0-indexed 对话位置）与转换后 trace step 的双向映射 |
| **评测脚本修复** | 修复 `evaluate.py` 中 `in` vs `==` 等已知问题 |
| **Pipeline 模式切换** | 支持"仅定位"模式（跳过升级阶段），减少不必要的计算 |
