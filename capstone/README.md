# 🎓 Capstone：企业智能客服 Agent

课程毕业项目。把六个 Stage 的能力拧成一个端到端系统：RAG + 工具调用 + LangGraph Agent + 多轮记忆 + FastAPI 流式部署。

📖 完整方案/架构/实现/面试讲法：
- `docs/capstone/capstone_overview.md`（方案与架构）
- `docs/capstone/capstone_implementation.md`（实现详解）

```bash
pip install langgraph langchain-chroma fastapi uvicorn   # 仓库根目录，已配 .env

# 命令行体验（RAG 查政策 / 查订单 / 开工单 / 多轮记忆）
python capstone/agent.py

# 起服务（流式 + 会话隔离）
python capstone/api.py     # http://127.0.0.1:8000/docs
```

## 六大能力对照
| 能力 | 体现 |
|------|------|
| RAG | `search_policy` 工具检索政策知识库 |
| 工具调用 | `get_order_status` / `create_ticket` |
| Agent 编排 | `create_react_agent` 自主决策 |
| 记忆 | `checkpointer` + session_id 隔离 |
| 部署 | FastAPI + SSE 流式 |
| 评测/可观测 | 工具调用轨迹打印（可扩展 LLM-as-Judge） |

文件：`agent.py`（Agent + 工具）、`api.py`（部署层）、`data/`（客服政策知识库）。
