# 04-06 ⭐Supervisor 多 Agent 模式：主管用结构化输出调度子 Agent

> 🎯 **一句话**：与其让一个「全能 Agent」啥都干，不如建一个**主管节点（Supervisor）**——它用结构化输出决定「这一步派给哪个专精子 Agent」，子 Agent 干完汇报回主管，主管再决定下一步或收尾。这是当下最主流、最可控的多 Agent 协作架构。

---

## 为什么需要它

单 Agent 塞满几十个工具时，会出现三个老问题：模型**选错工具**、上下文**被无关工具描述撑爆**、出错**难定位**。多 Agent 的思路是「分而治之」：把能力拆成几个**专精子 Agent**（研究员只会查资料、写手只会写作），每个 Agent 上下文干净、职责单一。

但拆开后谁来协调？这就是 **Supervisor（主管）模式**：一个中心节点像项目经理一样，看当前进展，**决定下一棒交给谁**。它和另一种「群聊 / handoff」模式形成对比：

| 模式 | 控制流 | 特点 |
|:---|:---|:---|
| **Supervisor（主管）** | 星型：所有路由经过主管 | 集中可控、易调试、易加审批；主管是瓶颈 |
| **群聊 / Handoff（对等交接）** | 网状：Agent 之间直接「移交」 | 灵活、去中心；但易失控、难追踪谁在主导 |

生产首选 Supervisor——因为它的路由决策**全在一个地方**，好观测、好兜底、好插人工审批（04-05）。

---

## 核心用法

```python
from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from common.llm_provider import get_llm

llm = get_llm(temperature=0)


# 1. 两个专精子 Agent（各自只配自己的工具）
@tool
def web_search(q: str) -> str:
    """联网检索资料。"""
    return f"[检索结果] 关于「{q}」的三条要点……"

researcher = create_react_agent(llm, tools=[web_search],
                                prompt="你是研究员，只负责查资料并给出要点。")
writer = create_react_agent(llm, tools=[],
                            prompt="你是写手，把已有资料整理成通顺短文。")


# 2. 主管的「结构化输出」：强制它只能在固定选项里二选一
class Route(BaseModel):
    next: Literal["researcher", "writer", "FINISH"] = Field(
        description="下一步交给谁；都做完了就 FINISH")

router_llm = llm.with_structured_output(Route)   # ★ 关键：结构化输出做路由


# State 既存对话消息，也存主管这一步的决策 next
class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: str


# 3. 主管节点：看历史 → 结构化输出决定下一棒
def supervisor(state: State) -> dict:
    sys = ("你是主管。先让 researcher 查资料，再让 writer 成文，"
           "两步都完成后输出 FINISH。")
    decision = router_llm.invoke([("system", sys)] + state["messages"])
    return {"next": decision.next}                # 把决策塞进状态供路由读


# 子 Agent 节点：调用对应 Agent，把产出追加进消息
def run_researcher(state: State) -> dict:
    out = researcher.invoke({"messages": state["messages"]})
    return {"messages": [out["messages"][-1]]}

def run_writer(state: State) -> dict:
    out = writer.invoke({"messages": state["messages"]})
    return {"messages": [out["messages"][-1]]}


# 4. 路由函数：把主管写进状态的决策翻译成跳转
def route(state: State) -> str:
    return state["next"]                # "researcher" / "writer" / "FINISH"


builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", run_researcher)
builder.add_node("writer", run_writer)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route, {
    "researcher": "researcher",
    "writer": "writer",
    "FINISH": END,
})
builder.add_edge("researcher", "supervisor")    # ★ 子 Agent 干完汇报回主管
builder.add_edge("writer", "supervisor")        # ★ 同上

graph = builder.compile()
result = graph.invoke({"messages": [("user", "写一段关于量子计算商用进展的短文")],
                       "next": ""})
result["messages"][-1].pretty_print()
```

### 逐块「本质在干什么」

**① `create_react_agent` 造专精子 Agent。**
每个子 Agent 本身就是一张完整的 ReAct 图（03-10），但**只配自己领域的工具和提示词**。上下文干净，是多 Agent「分而治之」的单元。

**② `with_structured_output(Route)`——主管的决策必须是结构化的。**
这是 Supervisor 模式的**命门**。如果让主管用自然语言说「我觉得该让研究员去」，你没法可靠地解析它。`with_structured_output` + `Literal["researcher","writer","FINISH"]` 把主管的输出**约束成枚举之一**，路由 100% 可解析、不会蹦出图里没有的节点名。

**③ 主管节点只决策、不干活。**
`supervisor` 读历史，产出一个 `next` 字段写进状态。它自己不查资料、不写作——干活的是子 Agent 节点。

**④ 子 Agent 用固定边「汇报回主管」。**
`researcher → supervisor`、`writer → supervisor` 是回边。于是控制流是星型：**主管 → 某子 Agent → 回主管 → 再决策……** 直到主管输出 `FINISH`，条件边把流程导向 `END`。这正是 04-03 的循环结构，只不过循环体里是「派活」。

---

## 关键原理

**Supervisor = 「一个会路由的中心节点」+「若干汇报回它的子 Agent」+「结构化输出保证路由可控」。**

```
                ┌──────────── supervisor ────────────┐
                │      (with_structured_output)        │
        ┌───────┴────────┬─────────────┬──────────────┘
        ▼ researcher     ▼ writer      ▼ FINISH→END
        └──汇报回──┘      └──汇报回──┘
```

- **为什么强制结构化输出**：路由的可靠性 = 系统的可靠性。枚举约束让主管永远只能选合法子 Agent，杜绝「幻觉出一个不存在的角色」。
- **状态怎么在子 Agent 间传递**：全靠共享的 `messages`。研究员把检索结果追加进去，写手就能读到——子 Agent 之间**不直接对话**，只通过共享状态接力（与 04-01/04-02 一脉相承）。
- **Supervisor vs 群聊/Handoff 的本质差异**：Supervisor 是**集中式控制**（所有跳转经过一个决策点，可观测、可审批、可加 `recursion_limit` 兜底）；Handoff 是**去中心式交接**（Agent 间用工具直接移交控制权，更灵活但更易失控、更难追踪）。生产从 Supervisor 起步。
- **现成封装**：`langgraph-supervisor` 包的 `create_supervisor(...)` 把这套手搭逻辑封成一行；理解手搭版，用封装才不慌。

---

## 你来改

- [ ] 加第三个子 Agent `reviewer`（审校）：`Route` 的 `Literal` 加 `"reviewer"`，补节点、条件边映射和回边。
- [ ] 调用时加 `config={"recursion_limit": 8}`，把主管提示改坏让它反复派活，观察护栏触发。
- [ ] 给主管节点加一个 04-05 的 `interrupt`，每次派活前让人确认，体会集中式控制便于插审批。
- [ ] （思考）改成 Handoff 模式，研究员要「直接把活交给写手」，状态里需新增什么字段表达「交接目标」？

---

## 面试怎么考

**Q：Supervisor 多 Agent 模式是怎么工作的？为什么主管要用结构化输出？**
A：建一个中心主管节点，它读当前进展、用**结构化输出**（如 `with_structured_output` 约束成 `Literal["agentA","agentB","FINISH"]`）决定下一棒派给哪个专精子 Agent；子 Agent 干完通过固定边汇报回主管，主管再决策，直到输出 `FINISH` 走向 `END`。强制结构化输出是因为路由必须 100% 可解析——枚举约束保证主管永远只能选图中存在的合法节点，杜绝幻觉出不存在的角色导致路由崩溃。

**Q：Supervisor 模式和群聊 / Handoff 模式有什么区别？各自适合什么场景？**
A：Supervisor 是**集中式星型控制**，所有路由经过一个主管，优点是可观测、易调试、易加人工审批和递归护栏，缺点是主管是单点瓶颈，适合流程相对明确、强调可控的生产系统。Handoff（群聊）是**去中心式网状交接**，Agent 之间用「移交工具」直接把控制权交给同伴，更灵活、更适合开放探索，但易失控、难追踪谁在主导。生产通常从 Supervisor 起步。

**Q：多 Agent 之间是怎么共享信息的？子 Agent 会互相直接对话吗？**
A：不直接对话。它们共享同一份图状态（通常是 `Annotated[list, add_messages]` 的 `messages`）：上一个子 Agent 把产出追加进共享消息，下一个子 Agent 从中读取，靠**共享状态接力**而非点对点通信。这与 StateGraph 的核心思想一致——节点解耦，只围绕共享状态读写，由 reducer 负责把各自的更新安全合并。
