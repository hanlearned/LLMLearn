# 04-02 节点与边：`add_node` / `add_edge` / `START` / `END`

> 🎯 **一句话**：节点是「做事的函数」，边是「下一步去哪」。把一堆节点用边连起来、首尾接上 `START` 和 `END`，一张能跑的工作流图就成型了。本篇讲清节点函数的固定签名和连边规则。

---

## 为什么需要它

04-01 我们定义了共享状态，但状态只是「数据」。真正干活的是**节点**，决定执行顺序的是**边**。

为什么要把流程拆成「节点 + 边」而不是写一个大函数？因为 Agent 工作流天然是**有向图**：模型节点之后可能去工具节点、可能直接结束、可能回到自己。一旦你把每一步显式建成节点，就能：

- 单独测试、替换、复用某个节点；
- 在 LangSmith / 可视化里看到完整执行轨迹；
- 在任意节点之间插入审批、日志、重试，而不动其他节点。

`START` 和 `END` 是两个**虚拟节点**：`START` 代表「图的入口」，`END` 代表「图的终点」。它们不执行任何逻辑，只用来标记「数据从哪进、到哪停」。

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

llm = get_llm(temperature=0)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str


# 节点 A：根据 topic 生成提纲
def outline(state: State) -> dict:
    prompt = f"为主题《{state['topic']}》列一个 3 点提纲，简洁。"
    reply = llm.invoke(prompt)
    return {"messages": [reply]}              # 局部更新：追加一条消息


# 节点 B：基于上一步的提纲写一段正文
def write(state: State) -> dict:
    last = state["messages"][-1].content      # 读上一节点留下的提纲
    reply = llm.invoke(f"根据这个提纲写一段 100 字正文：\n{last}")
    return {"messages": [reply]}


builder = StateGraph(State)
builder.add_node("outline", outline)          # add_node(名字, 函数)
builder.add_node("write", write)

builder.add_edge(START, "outline")            # 入口 → A
builder.add_edge("outline", "write")          # A → B（普通固定边）
builder.add_edge("write", END)                # B → 终点

graph = builder.compile()
result = graph.invoke({"messages": [], "topic": "为什么要学 LangGraph"})
for m in result["messages"]:
    m.pretty_print()
```

### 逐块「本质在干什么」

**① `add_node("outline", outline)`——给函数登记一个「站名」。**
第一个参数是节点名（字符串，边要用它来引用），第二个是节点函数。节点名在图里必须唯一。

**② 节点函数签名是固定的：`def f(state) -> dict`。**
入参永远是**当前完整状态**（一个符合 `State` 的 dict），返回值永远是**局部更新 dict**（只含你要改的字段）。这是 LangGraph 的硬约定——记牢「**收 state，返 dict**」。函数内部读 `state["xxx"]`，框架负责把你返回的 dict 按 reducer 合并回状态。

**③ `add_edge("outline", "write")`——画一条固定的有向边。**
意思是「outline 跑完，无条件去 write」。固定边不做任何判断，适合线性流程。如果下一步要根据状态决定，就得用条件边（见 04-03）。

**④ `START` / `END` 必须接上。**
没有从 `START` 出发的边，图不知道从哪开始跑；没有到 `END` 的边，图不知道何时停。`compile()` 会校验可达性，孤立节点或断路会报错。

---

## 关键原理

**节点之间不直接传参，全靠共享状态接力。** 注意 `write` 节点没有从 `outline` 拿返回值——它是从 `state["messages"][-1]` 读到提纲的。这正是图模型的解耦点：**节点 A 把结果写进状态，节点 B 从状态里读**，二者不知道彼此的存在，只认识同一份 State。

- **一个节点可以有多条入边**：多个上游都能指向它，框架会在所有上游就绪后触发它（这是并行汇聚的基础）。
- **一个节点也可以有多条出边**：用固定边会变成「并行分叉」（同时去多个节点），用条件边则是「按状态选一条」。
- **节点函数返回空 dict `{}`** 表示「我不改状态，只产生副作用」（如打印日志、调外部 API 但不存结果）。

```
START ──→ outline ──→ write ──→ END
            │            ▲
         写 messages   读 messages[-1]
```

---

## 你来改

- [ ] 在 `outline` 和 `write` 之间插一个 `review` 节点，读最新消息、追加一句「（已审阅）」，并把边改成 `outline → review → write`。
- [ ] 让 `write` 节点返回 `{}`（不写状态），观察 `END` 后 `messages` 里是否还有正文（应当没有）。
- [ ] 给图加一个并行分叉：`START → outline`，然后 `outline` 同时连到 `write` 和一个新节点 `tag`（都用 `add_edge`），二者都连到 `END`，观察两个节点都被执行。
- [ ] （思考）若 `write` 想拿到 `outline` 之外某个更早节点的输出，应该怎么设计状态字段？

---

## 面试怎么考

**Q：LangGraph 节点函数的签名约定是什么？返回值有什么讲究？**
A：固定为 `def node(state) -> dict`——入参是当前完整状态字典，返回值是**只包含待更新字段的局部 dict**。框架拿这个局部 dict 按各字段的 reducer 合并回全局状态。绝不返回整个 state，也不直接 return 给下一个节点，下游节点通过读状态拿数据。

**Q：`START` 和 `END` 是真实节点吗？不连它们会怎样？**
A：它们是**虚拟节点**，不执行任何逻辑，仅标记图的入口和出口。`compile()` 会做可达性校验：没有从 `START` 出发的边，图无法启动；没有到 `END` 的路径，图不知何时停止——这两种情况都会编译报错或运行不终止。

**Q：固定边（`add_edge`）和条件边（`add_conditional_edges`）的区别？**
A：固定边是无条件的「A 跑完一定去 B」，用于线性流程；同一节点多条固定出边会变成**并行分叉**。条件边则在节点跑完后调一个路由函数，**根据当前状态动态选择**下一个节点，用于分支和循环。Agent 的「继续调工具还是结束」就靠条件边实现（见 04-03）。
