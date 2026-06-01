# 03-05 ReAct 范式：Thought / Action / Observation 循环的本质

> 🎯 **一句话**：ReAct 让模型把「推理」和「行动」交织起来——边想（Thought）边做（Action）边看结果（Observation），循环往复直到得出答案。它是所有「会用工具的 Agent」的思想原点。

> ⚠️ **软弃用提示**：`langchain.agents` 里基于文本解析的 `create_react_agent` 属 legacy。新项目请用 LangGraph 的同名预制件（见 03-10）。本篇重在讲清 ReAct 的**思想与循环本质**，并手写一个不依赖框架的版本。

---

## 为什么需要它

ReAct = **Rea**soning + **Act**ing（论文 *ReAct: Synergizing Reasoning and Acting in LLMs*）。它要解决的痛点是：

- 纯「思维链（CoT）」只会在脑子里推理，拿不到外部最新信息，容易一本正经地编（幻觉）。
- 纯「调工具」又缺乏推理，不知道何时该调、调完该怎么用。

ReAct 把两者缝合：**先想一步（这步我该干嘛）→ 采取行动（调工具）→ 观察结果 → 再想下一步**。推理指导行动，行动的结果又修正推理。这就是 Agent 能「多步解决复杂问题」的根本机制。

---

## 核心用法：不依赖框架，手写一个 ReAct 循环

ReAct 的经典实现是**纯靠 prompt 约定 + 文本解析**：让模型按固定格式输出 `Thought / Action / Action Input`，我们解析出来执行工具，把结果作为 `Observation` 拼回去，循环。

```python
from dotenv import load_dotenv
load_dotenv()

import re
from common.llm_provider import get_llm

# ── 1. 工具：普通 Python 函数即可 ──
def get_population(city: str) -> str:
    data = {"北京": "2185 万", "上海": "2487 万"}
    return data.get(city.strip(), "未知")

TOOLS = {"get_population": get_population}

# ── 2. ReAct 提示词协议：教模型严格按格式输出 ──
SYSTEM = """你要回答问题，可使用工具：
get_population(city): 查询某城市人口。

严格按以下格式逐步输出，每次只输出一组：
Thought: 你的推理
Action: 工具名
Action Input: 工具参数
（我会返回 Observation）
当你能回答时，输出：
Thought: 我已知道答案
Final Answer: 最终答案
"""

llm = get_llm(temperature=0)        # ReAct 对格式敏感，务必温度=0

def run_react(question: str, max_steps: int = 5) -> str:
    scratchpad = ""                  # 累积 Thought/Action/Observation 文本
    for step in range(max_steps):
        prompt = f"{SYSTEM}\n\n问题：{question}\n{scratchpad}"
        text = llm.invoke(prompt).content

        # 命中 Final Answer → 结束循环
        if "Final Answer:" in text:
            return text.split("Final Answer:")[-1].strip()

        # 解析出 Action 和 Action Input
        action = re.search(r"Action:\s*(.+)", text).group(1).strip()
        action_input = re.search(r"Action Input:\s*(.+)", text).group(1).strip()

        # 执行工具，得到 Observation
        observation = TOOLS[action](action_input)

        # 把这一轮（模型的思考 + 真实观察）拼回草稿纸，进入下一轮
        scratchpad += f"\n{text}\nObservation: {observation}\n"
        print(f"[第{step+1}步] Action={action}({action_input}) -> {observation}")

    return "超过最大步数，未得出答案。"

print(run_react("北京和上海哪个人口多？"))
```

**逐块「本质在干什么」：**

- **提示词协议是灵魂**：ReAct 没有任何「魔法」，它完全靠 prompt 约定 `Thought/Action/Action Input/Observation` 这套固定格式，再靠我们解析文本来落地。模型不按格式输出，循环就崩——这正是它脆弱、被结构化 Tool Calling 取代的原因。
- **`scratchpad`（草稿纸）**：每轮把「模型的思考 + 我们真实执行得到的 Observation」追加进去。下一轮把整张草稿纸喂回模型，它就能看见「我已经查过北京=2185万」，从而推进到查上海、再比较。
- **循环终止**：模型自己判断「信息够了」，输出 `Final Answer`，我们检测到就跳出。否则到 `max_steps` 强停。
- **推理与行动交织**：第一轮 Thought「我得先查北京」→ Action 查北京 → Observation 2185万；第二轮 Thought「还要查上海」→ … → 最后 Thought「都查到了，能比较了」→ Final Answer。这就是 ReAct。

---

## 关键原理

1. **ReAct = 把 CoT 的每一步「落地」成可执行动作**。CoT 是「纯想」，ReAct 是「想一步、做一步、看一步」，用真实 Observation 校正推理，大幅降低幻觉。
2. **两种落地方式**：
   - **文本协议版**（本篇手写 / 老 `create_react_agent`）：靠 prompt 格式 + 正则解析。模型不需要原生工具能力，但**脆弱**——格式跑偏就解析失败。
   - **Tool Calling 版**（03-03 / LangGraph）：靠模型原生 `tool_calls` 字段，结构化、稳定，是现在的主流。本质循环一模一样，只是「Action 的表达方式」从文本变成了结构化字段。
3. **Observation 是事实锚点**：每轮注入真实工具结果，让后续推理建立在事实而非臆测上。
4. **终止与上限**：靠模型输出终止标记 + `max_steps` 双保险，否则会无限循环。

---

## 你来改

- [ ] 给 `TOOLS` 再加一个 `get_area(city)` 工具，问「北京和上海哪个人口密度高」，观察模型如何分多步调用两个工具。
- [ ] 把 `temperature` 调到 `0.9`，多跑几次，观察格式是否开始跑偏、解析报错——体会文本协议的脆弱性。
- [ ] 把本篇的文本协议循环，和 03-03 的 Tool Calling 循环并排比较，找出「Action 表达方式」这一个核心差异。

---

## 面试怎么考

**Q：ReAct 是什么？它和思维链（CoT）有什么区别？**
A：ReAct = Reasoning + Acting，让模型交替进行推理（Thought）和行动（Action/调工具），用工具返回的 Observation 校正后续推理，循环至 Final Answer。CoT 只在内部推理、不与外部交互，易幻觉且拿不到实时信息；ReAct 把每步推理「落地」为可执行动作并用真实结果纠偏。

**Q：老的文本协议版 ReAct 为什么脆弱、被取代？**
A：它靠 prompt 约定 `Thought/Action/Action Input` 文本格式 + 正则解析来驱动循环，模型一旦输出不守格式（漏字段、多说话）就解析失败、循环崩溃。现代用模型原生的结构化 `tool_calls`，不依赖文本解析，稳定得多，所以文本版 `create_react_agent` 被标为 legacy。

**Q：ReAct 循环如何终止？如何防止死循环？**
A：模型判断信息充足时输出终止标记（如 Final Answer），程序检测到即返回；同时设 `max_iterations/max_steps` 上限强制收尾。两者配合避免模型反复调工具空转。
