# 03-10 ⭐推荐写法：LangGraph `create_react_agent` 一行建 Agent

> 🎯 **一句话**：`langgraph.prebuilt.create_react_agent` 是现代 Agent 的标准答案——一行代码绑定工具、跑起循环，全程状态透明、可持久化、可中断、可观测。**新项目就用它**。

> ✅ **本课立场**：03-04~03-07 的 `AgentExecutor` 系列只为「读懂老代码、应付面试」。真正写生产 Agent，从这一篇开始。

---

## 为什么需要它

`AgentExecutor`（03-04）的循环是个黑盒，状态藏在 `intermediate_steps` 里，想加「人工审批」「中途存档恢复」「条件分支」「多 Agent 协作」都很别扭。

LangGraph 把 Agent 循环**显式建成一张状态图**：

```
START → [模型节点] → 有 tool_calls? ──是──→ [工具节点] ──┐
              ↑                                          │
              └──────────────────────────────────────────┘
                       否 → END
```

`create_react_agent` 是这张图的**预制件（prebuilt）**——它已经帮你把「模型节点 ↔ 工具节点」的 ReAct 循环搭好了，你只管传 LLM 和工具。其内核仍是 03-03 那套 Tool Calling 循环，但外壳换成了可控、可持久化的状态图。

---

## 核心用法

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from common.llm_provider import get_llm


# 1. 工具：和前面完全一样，@tool 即可
@tool
def get_weather(city: str) -> str:
    """查询某城市的实时天气。"""
    return f"{city} 晴，26℃。"

@tool
def get_population(city: str) -> str:
    """查询某城市的人口。"""
    return {"北京": "2185 万", "上海": "2487 万"}.get(city, "未知")


# 2. 一行建 Agent：传模型 + 工具（可选 prompt）
llm = get_llm(temperature=0)                  # Agent 务必温度=0
agent = create_react_agent(
    llm,
    tools=[get_weather, get_population],
    prompt="你是一个生活助手，需要时调用工具。",  # 系统提示，可选
)

# 3. 跑：输入/输出都是 messages 列表
result = agent.invoke({"messages": [("user", "北京天气怎么样？人口多少？")]})

# 4. 读 messages 轨迹：完整的「问题→调用→结果→回答」都在里面
for m in result["messages"]:
    m.pretty_print()
```

**逐块「本质在干什么」：**

- **`create_react_agent(llm, tools, prompt)`**：返回一个**已编译好的状态图**（也是 Runnable）。它内部已连好「模型节点 → 条件判断 → 工具节点 → 回模型节点」的循环，不需要你写 `AgentExecutor`、不需要 `agent_scratchpad` 占位。
- **统一的 `messages` 接口**：输入是 `{"messages": [...]}`，输出也是 `{"messages": [...]}`。**所有状态都在 messages 里**——用户问题、模型的 `tool_calls`、`ToolMessage` 工具结果、最终回答，按顺序排成一条消息链。这就是 03-03 手写循环里那个 `messages`，现在由图自动维护。
- **`pretty_print()` 看轨迹**：依次打印 Human → AI(tool_calls) → Tool(结果) → AI(tool_calls) → Tool → AI(最终答案)。一眼看清 Agent 每一步做了什么，调试天然友好（详见 03-11）。

### 加记忆（持久化）：一行 checkpointer

```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_react_agent(llm, tools=[get_weather, get_population],
                           checkpointer=MemorySaver())   # 加上检查点

cfg = {"configurable": {"thread_id": "user-001"}}        # 用 thread_id 区分会话
agent.invoke({"messages": [("user", "我在北京")]}, config=cfg)
print(agent.invoke({"messages": [("user", "这里天气怎么样？")]}, config=cfg)
      ["messages"][-1].content)        # 它记得「这里」=北京
```

**本质在干什么？** `checkpointer` 把每一步的完整状态（含全部 messages）按 `thread_id` 存档。下次同一 thread 继续对话时自动恢复历史——这就是 03-08/03-09 的「记忆」，但在 LangGraph 里是**一等公民**，且天然支持中断后恢复。生产换成 `SqliteSaver` / `PostgresSaver` 即持久化落库。

### 流式观测每一步

```python
for chunk in agent.stream({"messages": [("user", "上海人口多少？")]},
                          stream_mode="values"):
    chunk["messages"][-1].pretty_print()   # 每个节点执行后都吐一次最新状态
```

`stream_mode="values"` 让你**实时看到**模型决定调工具、工具返回、模型再生成的全过程，无需等整个循环跑完。

---

## 与老 `AgentExecutor` 的对比

| 维度 | 老 `AgentExecutor`（03-04~07，legacy） | LangGraph `create_react_agent`（推荐） |
|---|---|---|
| 循环实现 | 黑盒，封在内部 | 显式状态图，每步是可控节点 |
| 中间状态 | 藏在 `intermediate_steps` | 全在 `messages`，透明可读 |
| 提示词 | 需手配 `agent_scratchpad` 占位 | 不需要，自动管理 |
| 记忆/持久化 | 外挂 Memory，割裂 | `checkpointer` 内建，一等公民 |
| 人工介入(HITL) | 很难 | 原生支持（中断/恢复） |
| 条件分支/多Agent | 别扭 | 图天然支持（见 Stage 4） |
| 流式细粒度观测 | 弱 | `stream_mode` 多档可选 |
| 中断后恢复 | 不支持 | 靠 checkpointer 支持 |
| 官方态度 | legacy，建议迁移 | 推荐 |

---

## 关键原理

1. **内核没变，外壳升级**：它仍是「模型决定调工具 → 执行 → 喂回」的 Tool Calling 循环（03-03），只是循环被建模成状态图，于是变得可控、可存、可观测。
2. **messages 即状态**：默认 State 是一个 `messages` 列表（用 `add_messages` reducer 自动累加）。理解 Agent 行为 = 读这条消息链。
3. **prebuilt = 开箱即用的图**：`create_react_agent` 适合「单 Agent + 一组工具」的标准场景。需要更复杂编排（分支、子图、多 Agent）时，用 `StateGraph` 手搓（Stage 4）——但起点都是这张预制图。
4. **它是通往 Stage 4 的桥**：本篇让你先享受「一行建 Agent」，Stage 4 再揭开 `StateGraph` 把图拆开自己搭。

---

## 你来改

- [ ] 加一个会「连环调用」的问题，如「北京和上海哪个人口多、天气如何」，用 `pretty_print` 看它分几步调了哪些工具。
- [ ] 加上 `MemorySaver` 和同一个 `thread_id` 连聊三轮，让后面的话依赖前面的上下文，验证记忆生效。
- [ ] 用 `stream_mode="updates"` 替换 `"values"`，对比两种流式模式打印内容的差异（提示：updates 只给每步增量）。

---

## 面试怎么考

**Q：为什么生产推荐用 LangGraph 的 `create_react_agent` 而非 `AgentExecutor`？**
A：AgentExecutor 循环是黑盒、状态藏在 intermediate_steps 里，难做人工审批、持久化、中断恢复、条件分支和多 Agent。LangGraph 把循环显式建成状态图，每步是可控节点，状态全在 messages 中透明可读，checkpointer 内建记忆与断点恢复，stream 支持细粒度观测。内核仍是同一套 Tool Calling 循环。

**Q：`create_react_agent` 的输入输出是什么？状态存在哪里？**
A：输入 `{"messages": [...]}`，输出也是 `{"messages": [...]}`。Agent 的全部状态（用户问题、模型 tool_calls、ToolMessage 工具结果、最终回答）都按序存在这个 messages 列表里，由图用 `add_messages` reducer 自动累加。

**Q：LangGraph Agent 怎么实现记忆和「断点续聊」？**
A：给 agent 传 `checkpointer`（开发用 `MemorySaver`，生产用 `SqliteSaver`/`PostgresSaver`），并在 config 里用 `thread_id` 标识会话。每步完整状态会按 thread 存档，同一 thread 再次调用自动恢复历史，从而实现多轮记忆和中断后恢复。
