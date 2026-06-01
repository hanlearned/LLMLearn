# 🎓 Capstone：实现详解与运行

> 接 [方案与架构](capstone_overview.md)。本篇逐模块拆代码、给运行步骤和面试讲法。代码在 `capstone/`。

---

## 一、模块拆解

### 1. 多 Agent 编排图 `backend/graph.py`（核心）
整张图是一个 `StateGraph`，共享状态包含 `messages`(记忆)、`route`(路由)、`answer`(回复)、`refund`/`approved`(退款)。

**主管路由（Supervisor）**：`triage_node` 用结构化输出判意图：
```python
class Route(BaseModel):
    intent: Literal["policy", "order", "refund", "other"]
decision = llm.with_structured_output(Route).invoke(...)
```
再用条件边把请求分发给对应专家节点。这比让一个 Agent 啥都干更可控、更好调试。

**专家节点**：
- `policy_node`：RAG 检索政策 + 依据作答（防幻觉）。
- `order_node`：`bind_tools` 调订单查询工具，一轮工具往返。
- `other_node`：直接闲聊。

### 2. 退款 Human-in-the-Loop
退款拆成两个节点，中间用 `interrupt_before` 卡住：
```python
g.compile(checkpointer=checkpointer, interrupt_before=["execute_refund"])
```
- `prepare_refund_node`：算出退款金额，返回「待审批」消息，**图在此暂停**。
- `execute_refund_node`：只有被恢复后才执行，并根据 `approved` 决定真退还是取消。

API 侧：
```python
# /api/chat：invoke 后检查是否停在 execute_refund 前
needs_approval = "execute_refund" in graph.get_state(cfg).next
# /api/approve：写入审批结果再恢复
graph.update_state(cfg, {"approved": req.approve})
graph.invoke(None, config=cfg)
```

### 3. 持久化记忆 `get_checkpointer()`
优先 `SqliteSaver`（重启不丢），缺依赖时优雅退回 `MemorySaver`。`session_id` 当 `thread_id`，多用户互不串台、同用户多轮有上下文。

### 4. 后端 + 前端 `backend/api.py` + `frontend/index.html`
- 后端 `GET /` 用 `FileResponse` 直接托管前端页面，零额外静态服务器。
- 前端原生 JS：生成随机 `session_id`，调 `/api/chat`；当返回 `needs_approval` 时渲染「批准/拒绝」按钮调 `/api/approve`。无构建步骤，打开即用。

### 5. 评测看板 `eval/run_eval.py`
把评测集灌进图、拿答案、LLM 打正确性分，聚合出平均分/通过率并写 `report.md`。Stage 5 的评测能力用到自己项目上，形成闭环。

---

## 二、运行

```bash
pip install -r capstone/requirements.txt      # 仓库根目录，已配 .env

# 本地起服务（前后端一体）
python capstone/backend/api.py
# 打开 http://127.0.0.1:8000/

# Docker 一键
docker compose -f capstone/docker-compose.yml up --build

# 评测看板
python capstone/eval/run_eval.py
```

**重点体验路径**：在网页输入「订单 10001 我要退款」→ Agent 路由到退款 → 返回「待审批」并弹出按钮 → 点「批准」才真正退款、点「拒绝」则取消。这一条就把 多Agent路由 + 工具 + HITL + 记忆 全串起来了。

---

## 三、面试怎么讲这个项目

- **一句话**：「做了个全栈智能客服 Agent，用 LangGraph 编排多 Agent（主管路由 + 政策/订单/退款专家），RAG 查政策、工具查订单、退款走人工审批 HITL，SqliteSaver 持久记忆，FastAPI + 前端页面 + Docker 部署，还配了评测看板。」
- **「多 Agent 怎么协作？」**：triage 用结构化输出判意图分发到专家节点，是 Supervisor 模式；好处是职责单一、可单独优化、可插护栏。
- **「退款这种敏感操作怎么保证安全？」**：interrupt_before 在执行前冻结图，前端人工审批，approved 写回状态后才恢复执行。涉及钱/不可逆动作一律 HITL。
- **「多用户会话怎么不串？」**：session_id=thread_id，checkpointer 按 thread 存状态。
- **「怎么证明它好用/怎么迭代？」**：eval 看板把正确性跑成分数和通过率，改一处看分数升降。
- **「还缺什么能上生产？」**：鉴权限流、Redis 缓存、可观测(LangSmith)、更细的护栏与兜底、灰度。能说全=懂工程化不只是会 demo。

---

🎉 **完成 Capstone，对照[总纲能力清单](../roadmap.md)逐条打勾。全部打勾，你就具备了 Agent 开发工程师的核心能力。**
