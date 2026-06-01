# 03-06 OpenAI Functions / Tools Agent：原理与适用场景

> 🎯 **一句话**：这类 Agent 直接利用模型**原生的 function/tool calling 能力**来决定调用工具，比文本协议版 ReAct 更稳定——但它仍是 legacy，新项目请用 LangGraph（见 03-10）。

> ⚠️ **软弃用提示**：`create_openai_functions_agent` / `create_openai_tools_agent`（来自 `langchain.agents`）已 legacy。本篇讲原理与适用判断，生产请迁移 LangGraph。

---

## 为什么需要它

03-05 的文本协议 ReAct 靠正则解析模型输出，很脆。OpenAI 在 2023 年给模型加了原生 **function calling**：模型直接以结构化字段输出「要调哪个函数、参数是什么」，不再需要我们解析文本。

`create_openai_functions_agent` / `create_openai_tools_agent` 就是把这种原生能力封装成 Agent：

- **functions**（单数，旧）：模型一次只返回一个函数调用，对应早期 `function_call` 字段。
- **tools**（复数，新）：模型可一次返回多个工具调用（并行），对应现在的 `tool_calls` 字段。**优先用 tools 版**。

它解决的核心问题：**用结构化字段替代脆弱的文本解析**，让工具选择稳定可靠。这也是为什么它一度成为主流——直到 LangGraph 把整个循环也做得可控。

---

## 核心用法

```python
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from common.llm_provider import get_llm


@tool
def search_flight(city: str, date: str) -> str:
    """查询某城市某天的航班。"""
    return f"{date} 飞往 {city} 的航班：CA1234 08:00。"


prompt = ChatPromptTemplate.from_messages([
    ("system", "你是订票助手，需要时调用工具。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),   # 存放中间工具调用记录
])

llm = get_llm(temperature=0)                  # 工具调用务必温度=0
tools = [search_flight]

agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(executor.invoke({"input": "帮我查 6 月 10 号去上海的航班"})["output"])
```

**逐块「本质在干什么」：**

- **`create_openai_tools_agent`**：构造的决策单元内部用 `llm.bind_tools(tools)`，让模型用原生 `tool_calls` 输出调用决定——没有任何正则解析。这是它比文本 ReAct 稳的根本原因。
- **`agent_scratchpad`**：和 03-04 一样的草稿纸占位，但这里填充的是结构化的 `AIMessage(tool_calls=...)` 和 `ToolMessage`，不是文本。
- **`AgentExecutor`**：依旧负责循环——执行工具、把 `ToolMessage` 回填、再问模型，直到模型不再返回 `tool_calls`。
- 注意：尽管名字带 "openai"，只要厂商模型兼容 OpenAI 的 tool calling 协议（本课程的 DeepSeek/Kimi/硅基流动多数支持），就能用。

---

## 关键原理

1. **它就是 03-03 Tool Calling 的封装**：`create_openai_tools_agent` 的内核 = `bind_tools` + 解析 `tool_calls`，外面套 `AgentExecutor` 循环。理解了 03-03，这篇没有新东西，只是「谁来写循环」的差异。
2. **functions vs tools 的演进**：`function_call`（单调用）→ `tool_calls`（多调用、并行）。`create_openai_functions_agent` 对应前者、更旧；`create_openai_tools_agent` 对应后者、是它的升级。
3. **依赖模型原生能力**：模型必须支持 function/tool calling 协议。不支持的小模型只能退回文本协议 ReAct（03-05）。
4. **适用场景判断**：
   - 模型支持 tool calling、工具参数清晰 → 这类 Agent（稳定）。
   - 模型**不支持** tool calling（如某些本地小模型）→ 只能用文本 ReAct。
   - **新项目、需要人工介入/持久化/分支/多 Agent** → 一律 LangGraph（03-10），不要再用本篇的 legacy 封装。

---

## 你来改

- [ ] 加一个 `book_flight(flight_no)` 工具，问「查 6 月 10 号去上海的航班并订第一班」，观察它如何连续两轮调用（先查后订）。
- [ ] 把 `create_openai_tools_agent` 换成 `create_openai_functions_agent`，对比两者在多工具并行场景下的差异。
- [ ] 对照 03-03，确认这个 Agent 内部就是你手写过的那套 Tool Calling 循环。

---

## 面试怎么考

**Q：OpenAI functions agent 和文本协议 ReAct 的本质区别是什么？**
A：ReAct 靠 prompt 约定文本格式 + 正则解析来获取「调哪个工具」，脆弱；functions/tools agent 用模型原生的结构化 `tool_calls` 字段输出调用决定，无需解析文本，稳定得多。前者不依赖模型特殊能力，后者要求模型支持 function calling。

**Q：`create_openai_functions_agent` 和 `create_openai_tools_agent` 有什么区别？**
A：前者对应旧的单个 `function_call`，一次只能调一个函数；后者对应新的 `tool_calls`，支持一次返回多个工具调用（并行执行）。新代码优先用 tools 版。

**Q：既然它比 ReAct 稳，为什么还被标为 legacy？**
A：稳的只是「工具选择」环节，外层循环仍由黑盒 `AgentExecutor` 驱动，难以做人工审批、状态持久化、条件分支和多 Agent 编排。LangGraph 把循环显式化为可控状态图，因此官方推荐迁移。
