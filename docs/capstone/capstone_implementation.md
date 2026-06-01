# 🎓 Capstone：实现详解与运行

> 接 [方案与架构](capstone_overview.md)，本篇逐模块拆解代码、给出运行步骤和进阶扩展方向。

---

## 一、模块拆解

### 1. RAG 工具：让 Agent 拥有「查政策」的能力
```python
@tool
def search_policy(query: str) -> str:
    """检索客服政策（退换货、物流、会员…）。涉及"政策/规则/能不能"的问题用它。"""
    docs = _get_retriever().invoke(query)
    return "\n---\n".join(d.page_content for d in docs)
```
- `_get_retriever()` 用懒加载把政策知识库建成 Chroma 检索器，首次调用才建库。
- docstring 写清「什么时候用我」——Agent 据此决定是否调用。这是把整条 RAG 链「降维」成一个工具的关键。

### 2. 业务工具：查订单、开工单
`get_order_status` 查（模拟的）订单库，`create_ticket` 在自助无解时开人工工单。真实项目把内部换成调业务系统 API 即可，对 Agent 的接口不变。

### 3. Agent：一行编排
```python
create_react_agent(
    model=get_llm(temperature=0),
    tools=[search_policy, get_order_status, create_ticket],
    prompt=SYSTEM_PROMPT,
    checkpointer=MemorySaver(),   # ← 记忆
)
```
ReAct 循环（决定→调用→喂回→再决定）由 LangGraph 自动跑。System Prompt 里写明了行为原则（先查政策再答、没订单号要先问、自助无解开工单），引导 Agent 少犯错。

### 4. 记忆：按会话隔离
`chat()` 把 `session_id` 传成 `thread_id`。checkpointer 按 thread_id 存对话状态，于是「那如果一直没收到呢？」这种依赖上文的追问，Agent 能接得住。

### 5. 部署：FastAPI + 流式
`api.py` 用 `stream_mode="messages"` 让 Agent 按 token 流式输出，包成 SSE 推给前端。`session_id` 在请求体里，天然支持多用户并发会话。

---

## 二、运行

```bash
pip install langgraph langchain-chroma fastapi uvicorn   # 仓库根目录，已配 .env

# 命令行体验三轮对话（含 RAG、查单、开工单、记忆）
python capstone/agent.py

# 或起服务
python capstone/api.py        # http://127.0.0.1:8000/docs
```

命令行会看到 Agent 对三个问题分别**自主选择不同工具**：第一问走 RAG 查政策、第二问走订单查询、第三问结合记忆开工单。

---

## 三、进阶扩展（从「能用」到「能打」）

按这些方向迭代，项目含金量再上一个台阶——每一条都对应一个 Stage 的深化：

1. **多 Agent（Supervisor）**：拆成「售前 Agent / 售后 Agent / 物流 Agent」，用 [Supervisor 模式](../stage04/04-06_supervisor.md)调度。
2. **人工介入（HITL）**：退款这类敏感动作前 [interrupt 暂停](../stage04/04-05_human_in_the_loop.md)，等人工审批。
3. **持久化记忆**：MemorySaver 换成 SqliteSaver，重启不丢会话。
4. **评测闭环**：建客服评测集，用 [LLM-as-a-Judge](../stage05/05-05_llm_as_judge.md) 监控回答质量。
5. **缓存与限流**：接 [缓存](../stage06/06-04_caching.md) 降本、按 API Key 限流。
6. **接真实数据**：政策库换成公司真实 FAQ、订单工具接真实订单系统。

---

## 四、面试怎么讲这个项目（STAR + 技术深度）

- **一句话介绍**：「做了一个企业智能客服 Agent，用 LangGraph 编排，能自主调用 RAG 知识检索和订单/工单业务工具，带多轮记忆，用 FastAPI 流式部署。」
- **追问「RAG 和 Agent 怎么结合的？」**：RAG 被封装成 Agent 的一个工具，Agent 自己判断要不要查政策——不是写死的流程。
- **追问「怎么保证不乱答政策？」**：政策一律走 RAG 检索 + Prompt 强约束「依据检索结果回答、不编造」，温度 0。
- **追问「多用户会话怎么隔离？」**：session_id 映射 thread_id，checkpointer 按 thread 存记忆，互不串台。
- **追问「怎么知道它好不好/怎么迭代？」**：打印工具调用轨迹定位问题；建评测集 + LLM-as-Judge 量化回答质量，用数字驱动迭代。
- **追问「上线还缺什么？」**：限流、缓存、持久化记忆、敏感操作 HITL、可观测（LangSmith）——能说全说明你懂工程化，不只是会 demo。

---

🎉 **完成 Capstone，对照[总纲能力清单](../roadmap.md)逐条打勾。全部打勾，你就具备了 Agent 开发工程师的核心能力。**
