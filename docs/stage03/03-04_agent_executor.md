# 03-04 `AgentExecutor` 原理：执行循环、intermediate_steps、max_iterations

> 🎯 **一句话**：`AgentExecutor` 是老版 LangChain 里驱动 Agent 跑「思考→调工具→再思考」循环的引擎。理解它是为了**读懂老代码、应付面试**——新项目请直接用 LangGraph（见 03-10）。

> ⚠️ **软弃用提示**：`AgentExecutor` 及 `langchain.agents` 下的 `create_*_agent` 系列已被官方标记为「legacy / 推荐迁移到 LangGraph」。本篇只讲原理，不建议在生产/新项目使用。

---

## 为什么需要它（以及为什么被取代）

03-03 我们手写了 Tool Calling 的循环。把它工程化——加上「解析模型输出、调度工具、记录每步、控制最大轮数、超时与异常处理」——就是 `AgentExecutor` 干的事。它曾是 LangChain Agent 的标准运行时。

但它有两个硬伤，催生了 LangGraph：
- **黑盒循环**：循环逻辑封死在内部，想在「调工具前加审批」「中途人工介入」「自定义分支」非常别扭。
- **状态不透明**：中间状态藏在 `intermediate_steps` 里，难以持久化、回放、流式细粒度观测。

LangGraph 把这个循环显式建成「状态图」，每一步都是可控节点，于是取代了它。

---

## 核心用法

```python
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from common.llm_provider import get_llm


@tool
def get_word_length(word: str) -> int:
    """返回一个单词的字符长度。"""
    return len(word)


prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手，必要时调用工具。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),   # 关键：存放中间步骤的占位
])

llm = get_llm(temperature=0)
tools = [get_word_length]

agent = create_tool_calling_agent(llm, tools, prompt)         # 1. 造「决策单元」
executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
                         max_iterations=5)                    # 2. 套上执行循环

result = executor.invoke({"input": "单词 strawberry 有几个字母？"})
print(result["output"])
```

**逐块「本质在干什么」：**

- **`create_tool_calling_agent`**：造的是「决策单元」（一个 Runnable），它只负责「看当前状态 → 输出『调哪个工具』或『最终答案』」。它**自己不循环、不执行工具**。
- **`agent_scratchpad` 占位符**：这是 Agent 的「草稿纸」。每跑完一步，`AgentExecutor` 会把「调了什么工具、得到什么结果」格式化后塞进这里，下一轮模型就看得见历史步骤。少了它，Agent 会失忆、反复调同一个工具。
- **`AgentExecutor`**：才是「循环引擎」。它反复调用决策单元，解析输出，执行工具，把结果回填 scratchpad，直到拿到最终答案或触发上限。
- **`max_iterations=5`**：循环上限，防止模型陷入「调工具→调工具→…」死循环。达到上限会强制返回。

---

## 关键原理

`AgentExecutor.invoke` 内部循环（伪代码）：

```python
intermediate_steps = []          # 累积 (AgentAction, observation) 列表
for i in range(max_iterations):
    # 1. 决策单元基于「输入 + 历史步骤」决定下一步
    output = agent.plan(input, intermediate_steps)

    # 2. 如果决策是「我有最终答案了」→ 跳出循环返回
    if isinstance(output, AgentFinish):
        return output.return_values            # {"output": "..."}

    # 3. 否则决策是「调用某工具」→ 执行它
    observation = tools_by_name[output.tool].invoke(output.tool_input)

    # 4. 把这一步记进 intermediate_steps，进入下一轮
    intermediate_steps.append((output, observation))

# 超过上限：返回兜底结果（early_stopping_method 决定怎么收尾）
```

- **`AgentAction` vs `AgentFinish`**：决策单元每轮输出二者之一。`AgentAction`=「调工具」（含 tool、tool_input），`AgentFinish`=「结束，这是答案」。循环靠它来判断是否终止。
- **`intermediate_steps`**：`(动作, 观察结果)` 的列表，是 Agent 的全部「记忆」。它会被渲染进 `agent_scratchpad` 喂回模型。返回时设 `return_intermediate_steps=True` 可拿到它，用于调试。
- **`max_iterations` + `early_stopping_method`**：前者限制轮数，后者决定到顶后如何收尾（`"force"` 直接返回提示，`"generate"` 让模型基于已有信息硬凑一个答案）。
- **异常处理**：`handle_parsing_errors=True` 可在模型输出格式错时，把错误作为 observation 回喂，让它重试而非直接崩。

---

## 你来改

- [ ] 给 `executor` 设 `return_intermediate_steps=True`，打印 `result["intermediate_steps"]`，看清每一轮调了什么。
- [ ] 把 `max_iterations` 改成 `1`，问一个需要两步的问题，观察它如何被强制中断。
- [ ] 把这个例子在脑子里和 03-03 手写循环对应起来：`AgentExecutor` 的循环对应你手写的 `while`，`intermediate_steps` 对应你维护的 `messages`。

---

## 面试怎么考

**Q：`AgentExecutor` 的执行循环是怎样的？什么时候停？**
A：循环里反复让决策单元基于「输入 + intermediate_steps」产出 `AgentAction` 或 `AgentFinish`：是 Action 就执行对应工具、把 (动作,观察) 记入 intermediate_steps 继续；是 Finish 就返回答案。终止条件有二：模型给出 AgentFinish，或达到 `max_iterations`。

**Q：`agent_scratchpad` / `intermediate_steps` 是干什么的？**
A：它们是 Agent 的中间记忆。`intermediate_steps` 累积每轮的 (动作, 观察结果)，渲染进 `agent_scratchpad` 占位后喂回模型，让模型看见已做过的步骤，从而推进而非重复。缺了它 Agent 会失忆、循环空转。

**Q：为什么生产推荐用 LangGraph 取代 AgentExecutor？**
A：AgentExecutor 的循环是封闭黑盒，难以插入人工审批、条件分支、状态持久化与细粒度流式观测。LangGraph 把循环显式建模为状态图，每步是可控节点，天然支持中断/恢复、检查点、HITL 和多 Agent 编排，因此官方将 AgentExecutor 标为 legacy。
