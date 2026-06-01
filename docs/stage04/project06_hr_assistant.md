# 项目 6：AI 招聘助手（MCP + RAG + Agent）

> 把 Stage 2 的 RAG 和 Stage 3/4 的 Agent 拧成一个系统，再引入前沿的 MCP 协议。这是本课程综合度最高的项目之一，直接对标「Agent 平台 / HR SaaS」岗位。
>
> 代码：`stage04_langgraph/project06_hr_assistant/`

---

## 一、需求与方案设计

### 业务目标
HR 用自然语言交互：既能问「这个岗位要求什么」，也能让系统「给候选人打匹配分」。系统要自己判断该查知识库还是该调用打分能力。

### 三项技术怎么组合

```
                 ┌─ 工具1 search_jd ──→ RAG 检索岗位 JD 知识库（Stage 2）
HR 提问 → Agent ─┤
 (Stage 3/4)     └─ 工具2 score_candidate ──→ LLM 给候选人打分
                                  ↑
                        可做成 MCP Server 对外提供（前沿加分项）
```

**最关键的认知**：RAG 不是和 Agent 并列的东西——**RAG 可以被封装成 Agent 的一个工具**。`search_jd` 内部就是一条 RAG 检索链，对 Agent 而言它只是「众多工具之一」。理解这一点，你就理解了复杂 Agent 系统的搭法。

---

## 二、实现详解

### 难点 1：把 RAG 封装成工具
```python
@tool
def search_jd(query: str) -> str:
    """检索招聘岗位的职责与要求。"""
    docs = retriever.invoke(query)
    return "\n---\n".join(d.page_content for d in docs)
```
Agent 通过 docstring 知道「想了解岗位要求时调它」。这样 Agent 就拥有了访问企业私有知识的能力。

### 难点 2：引导 Agent 的工具调用顺序
给候选人打分需要先知道岗位要求。System Prompt 里明确：「应先 search_jd 拿到要求，再 score_candidate」。于是 Agent 会自动编排成「先检索、后打分」的两步——这就是 Agent 的多步规划能力。

### 难点 3：MCP —— 让工具成为「标准插座」
`mcp_server.py` 用 FastMCP 把 `score_candidate` 暴露为标准 MCP 工具：

```python
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("hr-tools")

@mcp.tool()
def score_candidate(jd_requirements: str, resume: str) -> str:
    ...

mcp.run(transport="stdio")
```

**MCP 的价值**：这个打分能力一次实现，HR 系统、面试官的 Agent、甚至 Claude Desktop 都能即插即用，无需为每个客户端重写对接。LangGraph 这边用 `langchain-mcp-adapters` 的 `MultiServerMCPClient` 就能把 MCP 工具转成普通 LangChain 工具，用法和本地工具**完全一致**。

> MCP 类比 USB：以前每个工具一套私有接口，MCP 让工具变成标准插座。这是 2024-2025 最重要的 Agent 生态进展之一，面试高频。

---

## 三、运行

```bash
# 主流程（RAG + Agent，进程内工具，最简单）
pip install langgraph langchain-chroma
python stage04_langgraph/project06_hr_assistant/hr_agent.py

# 可选：体验 MCP Server
pip install mcp
python stage04_langgraph/project06_hr_assistant/mcp_server.py
```

---

## 四、面试怎么考

- **「RAG 和 Agent 是什么关系？」** → 不是并列；RAG 可作为 Agent 的一个检索工具。Agent 是大脑（决定做什么），RAG 是它访问知识的一只手。
- **「MCP 解决什么问题？和普通 Function Calling 区别？」** → Function Calling 是模型层面「决定调工具」；MCP 是工具层面的**标准化协议**，让工具跨客户端复用。二者不冲突：MCP 工具最终也是通过 Function Calling 被模型调用。
- **「这套系统怎么扩展到真实招聘平台？」** → search_jd 接真实 JD 库、score_candidate 接简历库做批量筛选、加多 Agent（初筛 Agent + 面试 Agent + 排期 Agent）用 Supervisor 编排。
- **「为什么把打分做成 MCP 而不是内嵌？」** → 复用性。多个系统都要这个能力时，MCP 让你一次实现、统一维护、处处接入。
