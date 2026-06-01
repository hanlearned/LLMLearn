# 03-03 Tool Calling 机制：模型如何「决定调用 → 执行 → 喂回结果」

> 🎯 **一句话**：Tool Calling 是 Agent 的发动机——它定义了「模型决定调哪个工具、我们执行、再把结果喂回模型」这一完整往返。看懂这一轮，你就看懂了所有 Agent 框架的底层。

---

## 为什么需要它

Agent 看起来很神奇，但拆开内核，它只是在反复做同一件事：**让模型在「直接回答」和「调用工具」之间二选一，工具结果再喂回去，直到模型给出最终答案**。这个「模型 ↔ 工具」的单次往返就叫一轮 Tool Calling。

`AgentExecutor`、ReAct、LangGraph 的 `create_react_agent`，本质都是把这一轮 Tool Calling **套进一个循环**而已。所以这一篇是整个 Stage 3 的地基：手写一轮，胜过背十个框架 API。

---

## 核心用法：手写完整的一轮 Tool Calling

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from common.llm_provider import get_llm


# ── 1. 定义工具 ──
@tool
def add(a: int, b: int) -> int:
    """计算两个整数相加。"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数相乘。"""
    return a * b

tools = [add, multiply]
tools_by_name = {t.name: t for t in tools}

# ── 2. 把工具绑定给模型 ──
llm = get_llm(temperature=0)                 # 工具调用务必温度=0
llm_with_tools = llm.bind_tools(tools)

# ── 3. 第一次请求：模型决定调用哪个工具 ──
messages = [HumanMessage("3 加 5 等于多少？")]
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)                       # 把模型的「调用决定」加入历史

print(ai_msg.tool_calls)
# [{'name': 'add', 'args': {'a': 3, 'b': 5}, 'id': 'call_abc'}]

# ── 4. 我们执行模型点名的工具 ──
for call in ai_msg.tool_calls:
    selected = tools_by_name[call["name"]]    # 按名字找到工具对象
    result = selected.invoke(call["args"])    # 真正执行
    # 把结果包成 ToolMessage，注意 tool_call_id 要对上
    messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

# ── 5. 第二次请求：把工具结果喂回，模型给出自然语言答案 ──
final = llm_with_tools.invoke(messages)
print(final.content)                          # 3 加 5 等于 8。
```

**逐块「本质在干什么」：**

- **第 2 步 `bind_tools`**：把工具 Schema 附着到模型上，得到一个「知道这些工具存在」的新模型对象。原 `llm` 不变。
- **第 3 步第一次 invoke**：模型读到问题和工具说明，**不直接回答**，而是返回一个 `AIMessage`，其 `content` 为空、`tool_calls` 里写明「我要调 add，参数 a=3,b=5」。这就是「决定调用」。
- **关键：`messages.append(ai_msg)`**：必须把模型这条「调用决定」留在对话历史里。否则第二次请求时模型不知道自己说过要调工具，会报「工具结果对不上」的错。
- **第 4 步执行**：框架不会自动执行工具。我们按 `call["name"]` 找到工具，用 `call["args"]` 执行，再用 `ToolMessage` 把结果包回去。`tool_call_id` 必须等于 `call["id"]`——这是「哪个结果对应哪次调用」的回执单号。
- **第 5 步第二次 invoke**：模型看到 `[问题, 我决定调add, add的结果=8]`，这才生成最终自然语言回答。

整个往返是 **两次模型请求 + 一次工具执行**。Agent 就是把第 3~5 步放进 `while` 循环：只要模型还返回 `tool_calls` 就继续执行并喂回，直到它返回纯文本答案。

---

## 关键原理

1. **模型从不执行工具**：它只输出「想调什么、传什么参数」。执行权 100% 在你手里。这既是安全边界（你可以拦截危险调用），也是责任（你得自己写执行和喂回）。
2. **消息历史是状态载体**：一轮里产生三类消息——`HumanMessage`（问题）、带 `tool_calls` 的 `AIMessage`（决定）、`ToolMessage`（结果）。三者必须按序、配对（靠 `tool_call_id`）地留在历史里，模型才能正确续推。
3. **可能一次调多个工具**：`tool_calls` 是个列表。强模型会并行点名多个工具，你需要遍历执行、生成多条 `ToolMessage`。
4. **循环 = Agent**：把「invoke → 有 tool_calls 就执行喂回 → 再 invoke」包成循环，加一个 `max_iterations` 上限防死循环，就是一个最朴素的 Agent。后面所有框架都是它的工程化封装。

---

## 你来改

- [ ] 把问题换成「3 加 5，再乘以 2」，观察模型是否分两轮调用（先 add 再 multiply），自己补一个 `while` 循环让它跑完。
- [ ] 在循环里加 `max_iterations=5` 上限和迭代计数打印，超过就强制停止。
- [ ] 故意不 `append(ai_msg)`，看会报什么错，理解「调用决定必须留在历史」的原因。

---

## 面试怎么考

**Q：完整描述一次 Tool Calling 的往返流程。**
A：① `bind_tools` 把工具 Schema 绑给模型；② 第一次 invoke，模型返回带 `tool_calls` 的 AIMessage（决定调用，不回答）；③ 我们按 name/args 执行工具，结果包成 ToolMessage（`tool_call_id` 配对）；④ 把 AIMessage 和 ToolMessage 都加进历史，第二次 invoke，模型据此生成最终答案。是「两次模型请求夹一次工具执行」。

**Q：`tool_call_id` 有什么用？漏了会怎样？**
A：它是「调用」与「结果」的配对回执号。一次可能调多个工具，模型靠 id 把每条 ToolMessage 对应回各自的调用。漏了或对不上，模型无法关联结果，API 会报错或推理错乱。

**Q：Agent 和单轮 Tool Calling 的区别是什么？**
A：单轮是一次往返；Agent 是把这个往返放进循环，让模型可以多步、多次调用工具（前一步结果决定下一步），直到产出最终答案，并用 `max_iterations` 防死循环。本质上 Agent = 带终止条件的 Tool Calling 循环。
