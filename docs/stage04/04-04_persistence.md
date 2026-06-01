# 04-04 持久化 `checkpointer`：thread_id、断点续聊与 time-travel

> 🎯 **一句话**：给图编译时挂一个 `checkpointer`，LangGraph 就会在**每一步之后自动存档**整个状态。于是同一个 `thread_id` 能跨多次调用记住对话、断线能续聊、甚至能回到任意历史快照「时间旅行」重跑。

---

## 为什么需要它

到目前为止，`graph.invoke(...)` 是**无记忆**的：跑完一次，状态就丢了。下次再 `invoke`，又从空白开始。可真实 Agent 必须有记忆：

- **多轮对话**：用户第二句「那它的人口呢？」里的「它」，得靠记住上一轮的「北京」；
- **长流程**：一个审批工作流可能跨几分钟甚至几天，进程重启后要能从断点继续；
- **可恢复 / 可回溯**：出错了要能回到上一步重试，调试时要能查「第 3 步时状态长啥样」。

`checkpointer`（检查点存储器）就是 LangGraph 的持久层。挂上它，框架在**每个节点执行后**把完整状态序列化存一份快照。`thread_id` 则是「会话的身份证」——同一个 `thread_id` 的多次调用共享同一条状态历史，不同 `thread_id` 互相隔离。

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
from common.llm_provider import get_llm

llm = get_llm(temperature=0)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# ★ 关键：compile 时传入 checkpointer
memory = MemorySaver()                       # 进程内存档（重启即丢，适合开发/测试）
graph = builder.compile(checkpointer=memory)

# ★ 每次调用都要带 thread_id，标识这是哪个会话
config = {"configurable": {"thread_id": "user-42"}}

graph.invoke({"messages": [("user", "我叫小明，记住我。")]}, config)
# 第二次调用：只发新消息，历史由 checkpointer 自动补齐
out = graph.invoke({"messages": [("user", "我叫什么？")]}, config)
out["messages"][-1].pretty_print()           # 模型答得出「小明」—— 因为记住了

# 换一个 thread_id，就是另一段独立对话（不记得小明）
graph.invoke({"messages": [("user", "我叫什么？")]},
             {"configurable": {"thread_id": "user-99"}})
```

### 逐块「本质在干什么」

**① `MemorySaver()`——最简单的存档器。**
它把每步快照存在进程内存里。**进程一退就全丢**，所以只适合开发、单测、Demo。生产要换持久化后端（见下文）。

**② `compile(checkpointer=memory)`——开启自动存档。**
挂上后，每个节点跑完，框架自动把当前完整状态（含 `messages` 全历史）序列化进 checkpointer。你不用写任何「保存」代码。

**③ `config={"configurable": {"thread_id": ...}}`——指定会话身份。**
`thread_id` 是会话主键。带同一个 `thread_id` 调用，框架会**先加载该线程的最新快照作为起点**，再把你这次传的新消息 `add_messages` 追加上去。所以第二次只发「我叫什么？」，模型却能看到完整历史。

**④ 不同 `thread_id` = 不同会话，天然隔离。**
`user-99` 读不到 `user-42` 的历史。这就是多用户并发的隔离基础——一个服务用 `thread_id` 区分千万个用户的对话。

---

## 关键原理

**checkpointer 存的是「每一步后的完整状态快照」，不是「增量」。** 这带来三大能力：

| 能力 | 怎么实现 |
|:---|:---|
| **断点续聊** | 带同一 `thread_id` 再 invoke，自动加载最新快照接着跑 |
| **崩溃恢复** | 用持久化后端（SQLite/Postgres），进程重启后快照还在 |
| **time-travel（时间旅行）** | 每步快照都有 `checkpoint_id`，可指定回到任意历史点重跑 |

**换持久化后端只改一行：**

```python
# 开发：内存
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

# 生产：SQLite 文件（进程重启不丢）
from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver.from_conn_string("checkpoints.sqlite")
# 更高并发用 langgraph.checkpoint.postgres.PostgresSaver
```

**查历史与 time-travel：**

```python
graph.get_state(config)                      # 看当前快照
for s in graph.get_state_history(config):    # 遍历该线程所有历史快照
    print(s.config["configurable"]["checkpoint_id"], len(s.values["messages"]))

# time-travel：带某个旧 checkpoint_id 去 invoke，就从那一步分叉重跑
old = {"configurable": {"thread_id": "user-42", "checkpoint_id": "<旧id>"}}
graph.invoke({"messages": [("user", "换个问法")]}, old)
```

> 💡 时间旅行的价值：调试时回到「模型走错的那一步」改一点输入重跑，不必从头来过——这是 04-05 人工干预的技术底座。

---

## 你来改

- [ ] 把 `MemorySaver` 换成 `SqliteSaver.from_conn_string("chk.sqlite")`，跑两次脚本（中间退出进程），验证第二次进程仍记得「小明」。
- [ ] 用 `graph.get_state_history(config)` 打印某线程的全部快照数量，理解「一步一存」。
- [ ] 起 3 个不同 `thread_id` 各聊一句，确认它们互不串台。
- [ ] （思考）`MemorySaver` 和 Stage 3 的 `ConversationBufferMemory` 都叫「记忆」，本质区别在哪？

---

## 面试怎么考

**Q：LangGraph 怎么实现多轮对话记忆？和传统 Memory 组件有何不同？**
A：靠 `checkpointer` + `thread_id`。编译时挂 checkpointer，框架在每步后自动把**完整状态快照**存档；调用时带同一 `thread_id`，框架先加载该线程最新快照再追加新输入。与传统 `ConversationBufferMemory` 不同的是，它持久化的是**整张图的全部状态**（不止聊天记录），且天生支持崩溃恢复、time-travel、多线程隔离。

**Q：`thread_id` 起什么作用？换一个会怎样？**
A：`thread_id` 是会话的唯一标识，决定加载哪条状态历史。同一 `thread_id` 的多次调用共享并续接同一段状态；换一个 `thread_id` 就是一段全新的、隔离的会话，读不到其他线程的历史。生产中用它区分不同用户/会话，实现并发隔离。

**Q：什么是 time-travel？它依赖什么？**
A：time-travel 指回到某个**历史检查点**重新执行图。它依赖 checkpointer「每步存一个带 `checkpoint_id` 的完整快照」：通过 `get_state_history` 拿到某旧快照的 config，再带这个 `checkpoint_id` 去 invoke，就能从那一步分叉重跑。常用于调试和人工纠偏。
