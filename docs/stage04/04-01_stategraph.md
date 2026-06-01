# 04-01 ⭐`StateGraph`：用「共享状态」串起整张工作流图

> 🎯 **一句话**：`StateGraph` 是 LangGraph 的地基——它让你把一个工作流声明成「一个可变的共享状态 + 若干读写这个状态的节点」，而**状态怎么被更新（覆盖还是累加）由 reducer 决定**。读懂它，后面所有节点、边、循环、持久化、多 Agent 才有根。

---

## 为什么需要它

普通函数式的链路（`prompt | llm | parser`）是**单向数据流**：输入进、输出出，中途没有「全局可读可写的记忆」。可一旦要做 Agent：

- 模型节点要往对话历史里**追加**一条消息，工具节点也要追加，最后还要让模型看到全部历史；
- 多个节点要共享同一份「当前用户、检索到的文档、已调用工具次数」；
- 某些字段要**累加**（消息列表），某些字段要**覆盖**（当前步骤名）。

如果用普通变量传来传去，很快就乱成一团。`StateGraph` 的答案是：**定义一个状态结构（State），所有节点都围绕它读写**。节点不关心彼此，只关心「我读状态的哪几个字段、我返回哪几个字段的更新」——这就是图计算的解耦本质。

---

## 核心用法

```python
from dotenv import load_dotenv
load_dotenv()

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from common.llm_provider import get_llm

llm = get_llm(temperature=0)            # 工作流务必温度=0，结果稳定可复现


# 1. 定义状态：一个 TypedDict，字段就是整张图的「共享内存」
class State(TypedDict):
    messages: Annotated[list, add_messages]   # 累加型字段：新消息追加到列表
    user_name: str                            # 普通字段：默认「覆盖」语义


# 2. 节点函数：接收当前 state，返回「局部更新」（只返回要改的字段）
def chatbot(state: State) -> dict:
    reply = llm.invoke(state["messages"])
    return {"messages": [reply]}              # 不是替换整个列表，而是「追加这一条」


# 3. 建图：注册节点、连边
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")            # 入口
builder.add_edge("chatbot", END)              # 出口

# 4. 编译成可执行对象
graph = builder.compile()

# 5. 运行：传入符合 State 结构的初始字典
result = graph.invoke({"messages": [("user", "你好，我叫小明")], "user_name": "小明"})
for m in result["messages"]:
    m.pretty_print()
```

### 逐块「本质在干什么」

**① `class State(TypedDict)`——声明共享内存的形状。**
`TypedDict` 只是个「带字段类型的字典」。它不是运行时强校验，而是告诉 LangGraph：这张图的状态有哪些字段。运行时 state 就是一个普通 `dict`。

**② `Annotated[list, add_messages]`——给字段挂一个 reducer。**
这是 `StateGraph` 最关键的设计。`Annotated[类型, reducer]` 的第二个参数是一个函数，规定「**当某节点返回这个字段的新值时，如何把新值合并进旧值**」。`add_messages` 的逻辑是「把新消息**追加**到旧列表末尾，并按 ID 去重/更新」。没有 reducer 的字段（如 `user_name`），默认语义是**直接覆盖**。

**③ 节点返回「局部更新」而非完整状态。**
`chatbot` 只返回 `{"messages": [reply]}`，没有返回 `user_name`。LangGraph 会拿这个局部 dict，对每个字段调用对应 reducer：`messages` 走 `add_messages`（追加），其余字段保持不变。**节点永远不需要也不应该返回整个 state。**

**④ `compile()`——把「图的蓝图」固化成可执行体。**
`builder` 只是搭积木；`compile()` 做拓扑校验（有没有孤儿节点、入口出口是否可达），返回一个实现了 `invoke / stream / batch` 的 `Runnable`。它和普通 LangChain Runnable 接口一致，可以无缝塞进更大的链路。

---

## 关键原理

**reducer 是 StateGraph 的灵魂。** 你可以把一次 `graph.invoke` 想成多轮「状态更新事务」：每个节点跑完产出一个局部 dict，框架按字段逐个执行 `新状态[k] = reducer(旧状态[k], 局部更新[k])`。

```python
# add_messages 的等价心智模型（实际还会处理 ID 去重、删除消息等）
def add_messages(old: list, new: list) -> list:
    return old + new      # 累加，而非覆盖
```

- **覆盖 vs 累加，决定了字段的行为**。`messages` 必须累加（否则对话历史会被每个节点冲掉）；`current_step` 这类「当前态」就该覆盖。
- **reducer 可以自定义**。比如计数器字段用 `Annotated[int, operator.add]`，节点返回 `{"count": 1}` 就会在旧值上 +1。
- **并行节点的合并也靠 reducer**。当多个节点并发写同一字段时，框架用 reducer 把它们的结果合并，避免「最后写入者覆盖一切」的竞态。这是后面 04-06 多 Agent 并发的底层保证。

> 💡 记住这句话：**“状态怎么变，不在节点里，在字段的 reducer 上。”**

---

## 你来改

- [ ] 给 `State` 加一个 `step_count: Annotated[int, operator.add]` 字段（`import operator`），让 `chatbot` 返回 `{"messages": [reply], "step_count": 1}`，多轮调用观察它累加。
- [ ] 把 `messages` 字段的 `Annotated[list, add_messages]` 改成普通 `list`，再跑一次，观察对话历史是否被「覆盖」（提示：会丢历史）。
- [ ] 加第二个节点 `greeter`，读 `state["user_name"]` 返回一句问候追加进 `messages`，把边改成 `START → greeter → chatbot → END`。
- [ ] （思考）若想要一个「只保留最近 5 条消息」的字段，reducer 该怎么写？

---

## 面试怎么考

**Q：LangGraph 的 State 和普通函数传参有什么本质区别？**
A：普通传参是单向数据流，函数间无共享记忆。LangGraph 的 State 是一份**全局共享、可被多个节点读写的字典**，节点之间通过它解耦——节点只声明「读哪些字段、返回哪些字段的更新」，不直接调用彼此。这让循环、分支、多 Agent 协作都能在同一份状态上自然展开。

**Q：`Annotated[list, add_messages]` 里的 `add_messages` 起什么作用？不写会怎样？**
A：它是该字段的 **reducer**，规定节点返回新值时如何合并进旧值——`add_messages` 是「追加」语义。不写 reducer 时字段默认是「覆盖」语义，那么每个节点返回 `messages` 都会把整段对话历史冲掉，Agent 立刻失忆。所以凡是要累积的字段（消息、检索结果、日志）都必须挂累加型 reducer。

**Q：节点函数为什么只返回局部 dict，而不是整个 state？**
A：因为状态合并是框架按字段调 reducer 完成的，节点只需声明「我改了哪些字段」。返回局部 dict 有三个好处：① 避免节点之间误覆盖彼此的字段；② 让并行节点能各写各的字段、由 reducer 安全合并；③ 代码更清晰，一眼看出每个节点的副作用边界。
