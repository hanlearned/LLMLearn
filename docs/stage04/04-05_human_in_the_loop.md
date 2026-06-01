# 04-05 人在回路 Human-in-the-Loop：`interrupt` 暂停与恢复执行

> 🎯 **一句话**：在关键节点（下单、退款、发邮件）让图**主动暂停**，把控制权交还给人；人确认后用 `Command(resume=...)` 让它从断点**接着往下跑**。这就是「人工审批」能力的本质。

---

## 为什么需要它

全自动 Agent 很危险：模型可能误判而执行不可逆操作——真转账、真删库、真发邮件。生产里大量场景需要**人来把关**：

- 高风险动作执行前要人**审批**；
- 模型不确定时要人**补充信息**；
- 输出前要人**编辑/纠正**。

人在回路（HITL）的核心诉求是：流程能在某一步**停住、等人、再继续**，而且中途状态不能丢。这恰好建立在 04-04 的持久化之上——能存档，才能暂停后恢复。LangGraph 提供两种暂停方式：节点内用 `interrupt()` 函数动态暂停，或编译时声明 `interrupt_before` 在某节点前静态暂停。

---

## 核心用法

```python
from dotenv import load_dotenv
load_dotenv()

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from common.llm_provider import get_llm

llm = get_llm(temperature=0)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    draft: str


def draft_email(state: State) -> dict:
    reply = llm.invoke(f"给客户写一封致歉邮件，主题：{state['messages'][-1].content}")
    return {"draft": reply.content}


def human_approval(state: State) -> dict:
    # ★ interrupt() 一调用，图立刻暂停，把括号里的值抛回给调用方
    decision = interrupt({
        "question": "这封邮件可以发吗？回复 approve / 修改后的文本",
        "draft": state["draft"],
    })
    # —— 恢复执行后，decision 就是人给的值，函数从这里继续 ——
    if decision == "approve":
        return {"messages": [("ai", f"已发送：\n{state['draft']}")]}
    return {"messages": [("ai", f"已按修改发送：\n{decision}")]}


builder = StateGraph(State)
builder.add_node("draft", draft_email)
builder.add_node("approve", human_approval)
builder.add_edge(START, "draft")
builder.add_edge("draft", "approve")
builder.add_edge("approve", END)

graph = builder.compile(checkpointer=MemorySaver())   # ★ HITL 必须有 checkpointer
config = {"configurable": {"thread_id": "case-1"}}

# 第一次跑：到 human_approval 就停住
result = graph.invoke({"messages": [("user", "发货延迟")], "draft": ""}, config)
print(result["__interrupt__"])        # 拿到 interrupt 抛出的问题与草稿，展示给人

# —— 人看完，决定批准 ——
final = graph.invoke(Command(resume="approve"), config)   # ★ 用同一 thread_id 恢复
final["messages"][-1].pretty_print()
```

### 逐块「本质在干什么」

**① `interrupt(payload)`——在节点中间「按下暂停键」。**
调用它，图会：把当前状态存进 checkpointer → 立刻终止本次 `invoke` → 把 `payload`（你给人看的信息）通过结果里的 `__interrupt__` 抛回调用方。**注意它停在节点函数内部**，下面的代码这次不会执行。

**② 必须有 checkpointer。**
暂停意味着「把现场冻结起来，等会儿解冻」。没有存档，现场就没了。所以 HITL 图编译时**必须**挂 checkpointer，且调用必须带 `thread_id`。

**③ `Command(resume=值)`——解冻并把人的输入注入。**
人做完决定，再次 `invoke` 时传 `Command(resume="approve")`。框架从该 `thread_id` 的快照恢复现场，**让 `interrupt()` 这个调用「返回」人给的值**——于是 `decision` 拿到 `"approve"`，节点函数从 `interrupt()` 之后继续跑到结束。整个过程就像函数被「定格」又「续播」。

**④ 第二次 invoke 不传新 state，只传 `Command`。**
因为状态已在快照里，恢复时不需要重传输入，只需告诉框架「人给的答复是什么」。

---

## 关键原理

**`interrupt` vs `interrupt_before`：动态 vs 静态。**

```python
# 方式 A（推荐，本篇用的）：节点内动态暂停，能把上下文 payload 抛给人
decision = interrupt({"draft": state["draft"]})

# 方式 B：编译期静态声明「进入某节点前暂停」，不需改节点代码
graph = builder.compile(checkpointer=MemorySaver(),
                        interrupt_before=["approve"])
# 暂停后用 get_state 查看状态，必要时 update_state 改状态，再 invoke(None, config) 继续
```

- **方式 A（`interrupt`）**：暂停点在节点逻辑里，能精确决定「停在哪、给人看什么」，恢复时人的输入直接注入；表达力强，是新代码首选。
- **方式 B（`interrupt_before` / `interrupt_after`）**：在某节点前后整体暂停，适合「无脑卡一道审批」；恢复时常配合 `update_state` 手动改状态后再 `invoke(None, config)`。

**`update_state`——恢复前直接改状态。** 除了让 `interrupt` 返回值，你还能在恢复前直接覆写状态字段，相当于「人替模型改了输入」：`graph.update_state(config, {"draft": "人工改好的正文"})` 写状态，再 `graph.invoke(None, config)` 用改后的状态继续。

执行轨迹：

```
START → draft → [approve: interrupt() 暂停] ……等人……
                        │  Command(resume="approve")
                        ▼
                 approve 续播 → END
```

---

## 你来改

- [ ] 把恢复改成 `Command(resume="改好的正文")`，观察走「按修改发送」分支。
- [ ] 改用方式 B：去掉 `interrupt`，编译时加 `interrupt_before=["approve"]`，用 `get_state` 查看、`update_state` 改 `draft` 后 `invoke(None, config)` 继续。
- [ ] 加「拒绝」分支：人回 `"reject"` 时节点回到 `draft` 重写（把 `approve` 改成条件边）。
- [ ] （思考）一个流程要被人审批两次，两次 `interrupt` 之间的状态靠什么保住？

---

## 面试怎么考

**Q：LangGraph 怎么实现「人工审批」这类人在回路能力？**
A：在关键节点调用 `interrupt(payload)`，图会把当前状态存进 checkpointer 并立即暂停，把 `payload` 通过结果的 `__interrupt__` 返回给调用方展示给人。人做出决定后，用同一 `thread_id` 再次 `invoke` 并传 `Command(resume=人的输入)`，框架从快照恢复现场，让 `interrupt()` 返回人给的值，节点从断点继续执行。

**Q：为什么人在回路必须搭配 checkpointer？**
A：因为「暂停等人」可能跨秒、跨分钟甚至跨进程重启。暂停的本质是把整张图的状态冻结存档，恢复时再解冻。没有 checkpointer 就没有存档，暂停后状态丢失、无法恢复。所以 HITL 图编译时必须挂 checkpointer 并带 `thread_id`。

**Q：`interrupt`（函数）和 `interrupt_before`（编译参数）有什么区别？**
A：`interrupt` 是**节点内动态暂停**，可携带上下文 payload，恢复时人的输入直接作为返回值注入，表达力强、是首选。`interrupt_before/after` 是**编译期静态声明**，让图在进入/离开某节点前整体暂停，不改节点代码，恢复时常配合 `get_state` 查看、`update_state` 修改状态后 `invoke(None, config)` 继续，适合简单的「整体卡审批」。
