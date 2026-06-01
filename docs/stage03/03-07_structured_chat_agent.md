# 03-07 Structured Chat Agent：多输入工具的提示词协议

> 🎯 **一句话**：Structured Chat Agent 解决「模型不支持原生 tool calling，但工具又是多参数」时怎么办——用一套 **JSON 提示词协议**让模型把多个参数结构化地吐出来。

> ⚠️ **软弃用提示**：`create_structured_chat_agent`（来自 `langchain.agents`）已 legacy。本篇讲它的协议设计思想，新项目请用 LangGraph（见 03-10）。

---

## 为什么需要它

回顾两种前置方案的局限：
- **文本 ReAct（03-05）**：`Action Input` 只是一行纯文本，**只能优雅地传单个参数**。多参数工具（如 `下单(商品, 数量, 地址)`）就很别扭。
- **OpenAI tools agent（03-06）**：能传多参数，但**要求模型原生支持 tool calling**。

那「模型不支持原生 tool calling，工具又是多参数」的夹缝场景怎么办？`create_structured_chat_agent` 的答案是：**约定一套 JSON 格式的提示词协议**，让模型把工具名和多个参数一起输出成 JSON，我们解析这段 JSON 来执行。它是「文本协议」的升级版——把 `Action Input` 从一行文本升级成结构化 JSON。

---

## 核心用法

```python
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain import hub
from langchain_core.tools import tool
from common.llm_provider import get_llm


@tool
def book_room(hotel: str, nights: int, breakfast: bool) -> str:
    """预订酒店房间（多参数工具）。"""
    extra = "含早餐" if breakfast else "不含早餐"
    return f"已预订 {hotel}，{nights} 晚，{extra}。"


# structured chat agent 依赖一个内置 JSON 协议提示词
prompt = hub.pull("hwchase17/structured-chat-agent")

llm = get_llm(temperature=0)                     # 务必温度=0，协议对格式敏感
tools = [book_room]

agent = create_structured_chat_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
                         handle_parsing_errors=True)   # JSON 解析易错，开容错

print(executor.invoke({"input": "帮我订如家 3 晚，要含早餐"})["output"])
```

**逐块「本质在干什么」：**

- **`hub.pull("hwchase17/structured-chat-agent")`**：拉取内置的协议提示词。它的核心是教模型用如下 JSON 块回复（简化示意）：

```json
{
  "action": "book_room",
  "action_input": {"hotel": "如家", "nights": 3, "breakfast": true}
}
```

  `action` 是工具名，`action_input` 是**一个包含所有参数的 JSON 对象**——这就是它支持多参数的关键。
- **`create_structured_chat_agent`**：决策单元会解析模型回复里的这段 JSON，提取 action 和多个参数去执行。结束时模型输出 `{"action": "Final Answer", "action_input": "..."}`。
- **`handle_parsing_errors=True`**：模型有时会把 JSON 写坏（多逗号、混入解释文字）。开启后，解析失败时把错误回喂给模型让它重写，而非直接崩——这暴露了文本/JSON 协议的固有脆弱。

---

## 关键原理

1. **三代「Action 表达方式」的演进**，本质循环都一样，只是参数怎么传不同：

   | 方案 | Action 表达 | 多参数 | 依赖模型原生能力 | 稳定性 |
   |---|---|---|---|---|
   | 文本 ReAct（03-05） | 一行文本 | 弱 | 否 | 低 |
   | **Structured Chat（本篇）** | **JSON 对象** | **支持** | **否** | 中 |
   | OpenAI tools（03-06） | 原生 tool_calls | 支持 | 是 | 高 |

2. **它的生态位**：模型**不支持** tool calling、又要调多参数工具时的折中方案。本课程多数模型已支持 tool calling，所以实际很少需要它。
3. **JSON 协议仍是 prompt 解析**：和文本 ReAct 同源，只是格式更结构化。模型写坏 JSON 就解析失败，所以 `handle_parsing_errors` 几乎必开。
4. **被取代的逻辑同前两篇**：循环仍是黑盒 `AgentExecutor`。LangGraph 用结构化状态 + 原生 tool calling 同时解决了「多参数」和「循环可控」，故本篇 legacy。

---

## 你来改

- [ ] 给 `book_room` 再加一个 `room_type: str` 参数，问「订如家大床房 2 晚不要早餐」，看 JSON `action_input` 里的多参数填写。
- [ ] 把 `handle_parsing_errors` 关掉，把温度调高，多跑几次诱发 JSON 解析错误，体会其脆弱性。
- [ ] 想清楚一个判断题：如果你的模型支持 tool calling，你还会选 structured chat agent 吗？为什么？

---

## 面试怎么考

**Q：structured chat agent 和普通文本 ReAct 的核心差异？解决了什么问题？**
A：文本 ReAct 的 `Action Input` 是一行文本，难以优雅表达多参数；structured chat 把它升级为 JSON 对象（`action` + `action_input` 字典），从而支持多参数工具。两者同属「prompt 协议 + 解析」路线，都不要求模型原生 tool calling。

**Q：它和 OpenAI tools agent 各自适用什么场景？**
A：模型**不支持**原生 tool calling 但要调多参数工具 → structured chat（JSON 协议折中）；模型**支持** tool calling → OpenAI tools agent（更稳定）。如今主流模型都支持 tool calling，故 structured chat 实际使用越来越少。

**Q：这类基于 JSON 协议的 agent 有什么固有缺陷？**
A：依赖模型严格输出合法 JSON，模型一旦写坏格式（多余文字、非法 JSON）就解析失败，需要 `handle_parsing_errors` 兜底重试，稳定性和效率都不如原生 tool calling。加上循环不可控，故被 LangGraph 取代。
