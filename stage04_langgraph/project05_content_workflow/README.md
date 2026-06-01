# 项目 5：多 Agent 内容创作工作流

编辑部分工（规划/写作/审校/返工/定稿）自动产出高质量内容。LangGraph 多节点 + 条件边循环综合实战。

📖 完整方案/实现/复盘/面试：`docs/stage04/project05_content_workflow.md`

```bash
pip install langgraph        # 仓库根目录，已配 .env
python stage04_langgraph/project05_content_workflow/workflow.py
```

要点：planner→writer→editor→(质量门控 gate：达标定稿 / 不达标打回 writer)→finalizer。editor 温度=0 求稳，writer 温度=0.7 求创意，MAX_REVISIONS 止损防死循环。
