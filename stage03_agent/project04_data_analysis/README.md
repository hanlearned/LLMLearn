# 项目 4：智能数据分析助手（ChatBI 雏形）

用大白话问数据，Agent 自主选工具、传参、算结果。NL2SQL/ChatBI 的最小落地。

📖 完整方案/实现/复盘/面试：`docs/stage03/project04_data_analysis_agent.md`

## 快速开始
```bash
pip install langgraph pandas        # 在仓库根目录，且已配 .env
python stage03_agent/project04_data_analysis/agent.py
```

## 要点
- 用**受控结构化工具**（get_schema / aggregate / filter_by_month）而非让模型 eval 代码 —— 安全。
- 用 LangGraph `create_react_agent` 自动跑工具循环。
- 打印工具调用轨迹，体现可观测性。
- 换 `sales_data.csv` 为你的数据即可复用。
