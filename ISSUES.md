# Agents_Failure_Attribution 代码问题文档


## 问题三：Step/Agent 准确率匹配使用 `in` 而非 `==`

**严重程度：🟡 中**

**文件位置：** `Automated_FA/evaluate.py` 第 88-91 行

**问题描述：**

```python
if actual_agent in pred['predicted_agent']:
    correct_agent += 1
if actual_step in pred['predicted_step']:
    correct_step += 1
```

使用 `in`（子串匹配）而非 `==`（精确匹配）进行准确率判断。

**影响示例：**

| 真实值 | 预测值 | `in` 结果 | 是否应正确 |
|--------|--------|-----------|-----------|
| `"3"` | `"13"` | ✅ True | ❌ 应为 False |
| `"3"` | `"30"` | ✅ True | ❌ 应为 False |
| `"3"` | `"23"` | ✅ True | ❌ 应为 False |

这会导致 Step Accuracy 虚高。Agent 匹配同理，但由于 Agent 名字通常较长，误匹配概率相对较小。

---

## 问题四：Step-by-Step 方法未找到错误时无预测输出

**严重程度：🟢 低**

**文件位置：**
- `Automated_FA/Lib/utils.py` 第 181-183 行
- `Automated_FA/Lib/local_model.py` 第 208-209 行

**问题描述：**

当 Step-by-Step 方法遍历完所有步骤都未检测到错误时：

```python
if not error_found:
    print(f"\nNo decisive errors found by step-by-step analysis in file {json_file}")
```

此输出不包含 `Prediction for` 前缀，评测脚本无法解析该文件的预测结果。由于评测分母为数据集总文件数，这些文件等效于 Agent 和 Step 均判错。

---

## 问题五：All-at-Once 方法依赖 LLM 严格遵循输出格式

**严重程度：🟡 中**

**文件位置：**
- `Automated_FA/Lib/utils.py` 第 92-93 行（prompt 格式要求）
- `Automated_FA/Lib/local_model.py` 第 122 行（prompt 格式要求）

**问题描述：**

All-at-Once 方法将 LLM 的原始回复直接作为预测输出：

```python
print(f"Prediction for {json_file}:")
if result:
    print(result)
```

评测脚本需要从 LLM 回复中用正则 `Agent Name:\s*([\w_]+)` 和 `Step Number:\s*(\d+)` 解析结果。但 LLM 可能不严格遵循格式，例如：

- 输出 `Agent Name: The Art_Historian_Expert`（多余的 "The"）
- 输出 `Step Number: Step 3`（多余的 "Step"）
- 输出 `**Agent Name:** Art_Historian_Expert`（Markdown 加粗）
- Agent 名称包含非 `\w` 字符时无法被 `[\w_]+` 捕获

**影响：** 解析失败时该文件的预测被跳过，降低整体准确率。
