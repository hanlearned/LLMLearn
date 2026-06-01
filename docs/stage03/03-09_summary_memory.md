# 03-09 摘要记忆：长对话的 token 爆炸与「滚动摘要」

> 🎯 **一句话**：当对话太长、全量历史撑爆上下文窗口时，用「滚动摘要」把旧对话压缩成一段精炼摘要，只带摘要 + 最近几轮原文，控制 token。

> ⚠️ **替代提示**：旧的 `ConversationSummaryMemory` / `ConversationSummaryBufferMemory` 已 legacy。本篇讲清**思路与取舍**，并用现代 Runnable 写法手搓一个滚动摘要，新项目可在 LangGraph 状态里实现同等逻辑。

---

## 为什么需要它

Buffer 记忆（03-08）全量保存历史，但有硬上限：

- 模型上下文窗口有限，几十轮后历史可能超长直接报错。
- token 越多，**延迟越高、费用越贵**，且早期细节会稀释模型注意力。

**滚动摘要（Summary）记忆**的思路：不保留全部原文，而是**把较早的对话用 LLM 压缩成一段摘要**，新对话不断「滚动」并入这段摘要。这样无论聊多久，喂给模型的历史长度大致恒定——是「记得久」和「省 token」之间的折中。

---

## 核心用法（手搓滚动摘要）

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from common.llm_provider import get_llm

llm = get_llm(temperature=0)

# 1. 摘要器：把「旧摘要 + 新对话」压缩成一段更新后的摘要
summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是对话摘要器。把已有摘要和新对话融合，输出一段简洁的累计摘要，"
               "保留关键事实（人名、偏好、决定），丢弃寒暄。"),
    ("human", "已有摘要：\n{summary}\n\n新对话：\n{new_lines}\n\n请输出更新后的摘要："),
])
summarizer = summarize_prompt | llm

# 2. 主对话：只带「摘要」+「最近几轮原文」
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手。以下是早前对话的摘要，供你参考：\n{summary}"),
    MessagesPlaceholder("recent"),     # 只放最近 N 轮原文
    ("human", "{input}"),
])
chat = chat_prompt | llm

# 3. 简易状态：一段摘要 + 最近几轮窗口
summary = ""
recent: list = []
WINDOW = 4          # 最近保留 4 条消息原文，更早的并入摘要

def ask(user_input: str) -> str:
    global summary, recent
    reply = chat.invoke({"summary": summary, "recent": recent, "input": user_input}).content
    recent += [HumanMessage(user_input), AIMessage(reply)]

    # 窗口超限：把溢出的旧消息滚动并入摘要
    if len(recent) > WINDOW:
        overflow, recent = recent[:-WINDOW], recent[-WINDOW:]
        new_lines = "\n".join(f"{m.type}: {m.content}" for m in overflow)
        summary = summarizer.invoke({"summary": summary, "new_lines": new_lines}).content
    return reply

print(ask("我叫小明，是个 Python 后端工程师"))
print(ask("我最喜欢的框架是 FastAPI"))
print(ask("帮我推荐一本书"))
print(ask("我叫什么？做什么工作？喜欢什么框架？"))   # 即便已滚动，摘要里仍记得
```

**逐块「本质在干什么」：**

- **`summarizer`（摘要器）**：一条专门的链，输入「旧摘要 + 新溢出的对话」，输出「融合后的新摘要」。这就是「滚动」——摘要是被不断增量更新的，不是每次从头重写全部历史。
- **`recent` 窗口**：最近 `WINDOW` 条消息保留**原文**（细节精确），更早的才压成摘要（粗粒度）。这是最常用的混合策略（对应旧 `ConversationSummaryBufferMemory`）。
- **滚动触发**：每轮结束检查窗口是否溢出，溢出的旧消息交给摘要器并入 `summary`，窗口只留最近几条。于是喂给模型的总长度大致恒定。
- **代价显现**：每次滚动都**额外调用一次 LLM**（摘要器），且摘要是有损压缩——细节可能丢失。这正是它的核心取舍。

---

## 关键原理

1. **Summary = 有损压缩换长度恒定**：用「概括」替换「原文」，token 不再随轮数线性增长，但牺牲细节保真度。
2. **三种策略的取舍**：

   | 策略 | token 增长 | 细节保真 | 额外开销 | 适用 |
   |---|---|---|---|---|
   | Buffer（全量） | 线性增长 | 完整 | 无 | 短对话 |
   | Window（只留最近N轮） | 恒定 | 仅近期完整，远期全丢 | 无 | 只关心近期 |
   | **Summary（滚动摘要）** | **大致恒定** | **近期精确+远期概括** | **每次滚动多调一次 LLM** | **长对话需记住早期事实** |

3. **混合最实用**：近期原文 + 远期摘要（本篇做法），兼顾精度与长度，是工业界常见选择。
4. **风险点**：摘要质量依赖摘要器 prompt；关键事实（金额、ID、决定）若被压丢会出错，要在 prompt 里强调「必须保留关键事实」。
5. **现代落地**：不要用旧 Memory 类。在 LangGraph 里，把 `summary` 和 `recent` 放进图的 State，用一个「摘要节点」在消息超阈值时更新摘要——逻辑与本篇一致，但状态可持久化、可观测。

---

## 你来改

- [ ] 把 `WINDOW` 改成 2，多聊几轮，打印每轮的 `summary`，观察摘要如何滚动增长。
- [ ] 在对话里给一个具体数字（如「我的预算是 8500 元」），聊很多轮后再问预算，检验摘要是否保住了这个关键事实。
- [ ] 对比同一段长对话用 Buffer vs Summary 两种方式，估算各自喂给模型的 token 量差异。

---

## 面试怎么考

**Q：长对话为什么会出问题？摘要记忆怎么解决？**
A：Buffer 全量历史随轮数线性增长，会撑爆上下文窗口、推高延迟与成本。摘要记忆把较早对话用 LLM 滚动压缩成一段摘要，只带摘要 + 最近几轮原文，使喂给模型的长度大致恒定，从而支持超长对话。

**Q：摘要记忆的代价/风险是什么？**
A：① 每次更新摘要要额外调一次 LLM，增加延迟和成本；② 摘要是有损压缩，关键细节（金额、ID、决定）可能被丢，导致后续回答出错。需在摘要 prompt 里强制保留关键事实，并常配合「近期原文窗口」保精度。

**Q：Buffer、Window、Summary 三种记忆怎么选？**
A：短对话用 Buffer（完整简单）；只关心近期、不在乎早期细节用 Window（恒定长度）；长对话又要记住早期关键信息用 Summary，且实践中多用「近期原文 + 远期摘要」的混合策略平衡精度与长度。
