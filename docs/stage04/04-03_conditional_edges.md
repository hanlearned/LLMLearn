# 04-03 ⭐条件边 `add_conditional_edges`：分支与「Agent 循环」的本质

> 🎯 **一句话**：条件边让图在某节点跑完后，**根据当前状态动态决定下一步去哪**。分支靠它，循环也靠它——所谓「Agent 自主循环」，本质就是一条「条件边在『还要调工具吗』为真时回到自己」。

---

## 为什么需要它

固定边（04-02）是写死的「A 之后一定去 B」。但 Agent 的核心是**不确定性**：

- 模型这一轮**可能**要调工具，也**可能**直接给出最终答案；
- 路由器**可能**判断该走「退款流程」，也**可能**走「投诉流程」；
- 工具调用**可能**一次就够，也**可能**要循环好几轮。

这些「该走哪」必须**在运行时看状态才能定**。条件边就是把这个决策权交给一个你写的**路由函数**：节点跑完后，框架把当前状态喂给它，它返回一个「标签」，框架据此跳到对应节点。

**最关键的洞察：循环不是新机制，而是「条件边指向了一个早已执行过的节点」。** Agent 的 ReAct 循环，就是「工具节点 → 回到模型节点 → 模型判断还要不要调工具 → 要就再去工具节点」，靠条件边把控制流绕回去。

---

## 核心用法

```python
from dotenv import load_dotenv
load_dotenv()

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from common.llm_provider import get_llm


@tool
def get_weather(city: str) -> str:
    """查询某城市天气。"""
    return f"{city} 晴，26℃。"


llm = get_llm(temperature=0).bind_tools([get_weather])   # 让模型能发出 tool_calls


class State(TypedDict):
    messages: Annotated[list, add_messages]


def call_model(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


# 路由函数：看最新一条 AI 消息有没有 tool_calls，决定去哪
def route(state: State) -> str:
    last = state["messages"][-1]
    if last.tool_calls:          # 模型想调工具
        return "tools"
    return "end"                 # 模型给出了最终答案


builder = StateGraph(State)
builder.add_node("model", call_model)
builder.add_node("tools", ToolNode([get_weather]))   # 预制工具执行节点

builder.add_edge(START, "model")
builder.add_conditional_edges(          # ← 条件边
    "model",                            # 从哪个节点出发
    route,                              # 路由函数
    {"tools": "tools", "end": END},     # 标签 → 目标节点 的映射
)
builder.add_edge("tools", "model")      # ★ 工具跑完回到模型 —— 这就是「循环」

graph = builder.compile()
result = graph.invoke({"messages": [("user", "北京天气怎么样？")]})
result["messages"][-1].pretty_print()
```

### 逐块「本质在干什么」

**① 路由函数 `route(state) -> str`——返回一个「标签」。**
它接收当前状态，返回一个字符串（或 `END`）。注意它**只做判断、不改状态**，返回值不是节点名而是「分支标签」，由下面的字典翻译成真正的目标节点。这层间接让路由逻辑和图结构解耦。

**② 分支字典 `{"tools": "tools", "end": END}`——标签到目标的翻译表。**
键是路由函数可能返回的标签，值是对应跳转的节点名（或 `END`）。这里标签 `"end"` 被映射到 `END` 虚拟节点。**路由函数返回的标签必须在字典里有，否则运行时报错。**

**③ `add_edge("tools", "model")`——闭合循环的那一笔。**
工具执行完，无条件回到 `model` 节点。于是流程变成：`model →(要调工具)→ tools → model →(还要吗?)→ ...`。模型每轮重新判断，直到它不再产出 `tool_calls`，路由函数返回 `"end"`，循环退出。**这条回边，就是整个 Agent 循环的引擎。**

**④ `ToolNode`——预制的工具执行节点。**
它读最新 AI 消息里的 `tool_calls`，自动调对应工具，把结果作为 `ToolMessage` 追加进 `messages`。你也可以自己写这个节点，`ToolNode` 只是把通用逻辑封好了。

---

## 关键原理

执行轨迹（一次成功循环）：

```
START → model ──(有 tool_calls)──→ tools ──→ model ──(无 tool_calls)──→ END
          ↑__________________________________________│
                        条件边把控制流绕回来 = 循环
```

- **循环 = 条件边 + 回边**。没有任何「loop 关键字」，循环是图结构的自然产物：一条边指回上游节点。
- **退出条件藏在路由函数里**。这里是「模型不再要工具」。任何循环都必须有一个能让路由函数返回 `END` 的条件，否则就是死循环。
- **防失控有 `recursion_limit`**。`graph.invoke(..., config={"recursion_limit": 25})` 限制最大步数，超出抛 `GraphRecursionError`——这是兜底护栏，防止模型一直要工具停不下来。
- **路由函数可以返回多个分支**。不止「循环 / 结束」，监管者（04-06）就用路由函数返回 `"researcher" / "writer" / "end"` 三选一，实现多 Agent 分派。

---

## 你来改

- [ ] 把 `recursion_limit` 调成 2，问一个需要多轮工具的问题，观察 `GraphRecursionError`。
- [ ] 在 `route` 里加一个分支：若用户消息含「再见」，直接返回 `"end"` 跳过模型（需把条件边加到合适节点）。
- [ ] 不用 `ToolNode`，自己手写 `tools` 节点：遍历 `last.tool_calls`，调函数，构造 `ToolMessage` 返回。
- [ ] （思考）若路由函数返回了一个字典里没有的标签会怎样？如何用「默认分支」兜底？

---

## 面试怎么考

**Q：LangGraph 里的 Agent 循环是怎么实现的？有没有专门的循环语法？**
A：没有循环关键字。循环是**条件边 + 回边**的结构产物：模型节点用条件边判断「还要不要调工具」，要就去工具节点，工具节点用固定边**回到模型节点**，从而绕成环。模型每轮重新决策，直到不再产出 `tool_calls`，路由函数返回 `END`，循环自然退出。

**Q：路由函数（条件边的判断函数）能修改状态吗？返回的是节点名吗？**
A：路由函数**只读状态、只做判断，不应修改状态**。它返回的是一个「分支标签」字符串，再经分支字典翻译成真正的目标节点——这层间接让路由逻辑与图拓扑解耦。也可直接返回节点名（省略字典），但用标签映射可读性更好。

**Q：怎么防止 Agent 循环停不下来？**
A：两道防线。① **业务退出条件**：路由函数必须存在一个能返回 `END` 的判断（如「模型不再要工具」），这是正常出口。② **硬护栏 `recursion_limit`**：通过 `config={"recursion_limit": N}` 限制最大执行步数，超出抛 `GraphRecursionError`，防止模型反复要工具导致的死循环。
