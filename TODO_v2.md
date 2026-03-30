1. ✅ 去掉E2实验，E0重命名为E2，不含失败说明，任务描述和正确答案信息（需要适当修改传入信息）
2. ✅ 根据1要求修改相关提示词（如不需要 Agents_Failure_Attribution/feedback/feedback/prompts/intent_recognition_benchmark.hprompt 了），以及对应代码，指标生成等逻辑
3. ✅ 修改 Agents_Failure_Attribution/feedback/feedback/prompts/workflow_localization_benchmark.hprompt，我不想要Agent 角色清单这种东西。我还是希望根据对话历史轨迹解析出类似 Agents_Failure_Attribution/feedback/workflows/resume_screening_workflow/workflow.yaml 的工作流架构信息（如果无法很好的以DAG表示出来，那就表示成链状的，部分节点重复也可以。但我还是希望先探索出更有意义的工作流图出来）
