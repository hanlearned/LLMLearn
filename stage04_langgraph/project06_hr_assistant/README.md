# 项目 6：AI 招聘助手（MCP + RAG + Agent）

综合度最高的项目之一：RAG 检索岗位 JD + Agent 调度 + MCP 标准化工具。

📖 完整方案/实现/复盘/面试：`docs/stage04/project06_hr_assistant.md`

```bash
# 主流程（RAG + Agent）
pip install langgraph langchain-chroma
python stage04_langgraph/project06_hr_assistant/hr_agent.py

# 可选：MCP Server
pip install mcp
python stage04_langgraph/project06_hr_assistant/mcp_server.py
```

核心认知：**RAG 可被封装成 Agent 的一个工具**（`search_jd` 内部就是一条 RAG 链）。`score_candidate` 还能做成 MCP Server，跨客户端即插即用。

文件：`hr_agent.py`（主流程）、`mcp_server.py`（MCP 暴露工具）、`data/`（岗位 JD 知识库）。
