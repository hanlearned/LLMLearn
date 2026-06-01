# 项目 7：Prompt 评估与 A/B 测试平台

把「我觉得更好」变成「胜率 72%」。LLMOps 核心能力。

📖 完整方案/实现/复盘/面试：`docs/stage05/project07_prompt_ab_test.md`

```bash
python stage05_promptops/project07_prompt_ab_test/ab_test.py
```

三个工程要点：固定测试集（控制变量）、成对比较（比打分更稳）、位置去偏（交换 A/B 顺序双向评判，抵消裁判位置偏见）。换掉 `PROMPT_A/B` 和 `TEST_QUESTIONS` 即可测你自己的 Prompt。
