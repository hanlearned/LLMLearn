# 03-08 对话记忆：用消息历史维持多轮（`RunnableWithMessageHistory` 现代写法）

> 🎯 **一句话**：LLM 本身无状态、记不住上一句话；「对话记忆」就是把历史消息存起来、每次请求时拼回去，让多轮对话有上下文。

> ⚠️ **替代提示**：老的 `ConversationBufferMemory` / `ConversationChain`（来自 `langchain.memory`）已 legacy。现代写法用 `RunnableWithMessageHistory`（或在 LangGraph 里用 checkpointer）。本篇教现代写法，并讲清旧类被替代的原因。

---

## 为什么需要它

每次 `llm.invoke()` 都是独立请求，模型不会记得你上一句说了什么。多轮对话能「记住」，唯一原因是**我们每次都把历史消息一起发过去**。所谓「记忆」，本质就是**对消息历史的存储与拼接管理**。

最朴素的策略叫 **Buffer（缓冲）记忆**：原封不动地保存全部历史消息，每轮全量拼回去。简单、忠实，缺点是历史越长 token 越多（解决方案见 03-09 摘要记忆）。

旧的 `ConversationBufferMemory` 把这件事封装成了一个「记忆对象」，但它和 LCEL/Runnable 体系格格不入、接口僵硬，已被 `RunnableWithMessageHistory` 取代——后者是「给任意 Runnable 套上历史管理」的通用方案。

---

## 核心用法（现代写法）

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from common.llm_provider import get_llm


# 1. prompt 里用 MessagesPlaceholder 给「历史」留一个坑
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手。"),
    MessagesPlaceholder("history"),     # 历史消息会被填进这里
    ("human", "{input}"),
])
chain = prompt | get_llm(temperature=0)

# 2. 用一个字典按 session_id 存各自的历史
_store = {}
def get_history(session_id: str):
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]

# 3. 给 chain 套上「自动读写历史」的外壳
chat = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",      # 用户输入对应 prompt 的哪个变量
    history_messages_key="history",  # 历史填到哪个占位符
)

# 4. 调用时用 config 指定是哪一通对话（session）
cfg = {"configurable": {"session_id": "user-001"}}
print(chat.invoke({"input": "我叫小明"}, config=cfg).content)
print(chat.invoke({"input": "我叫什么名字？"}, config=cfg).content)   # 答：小明
```

**逐块「本质在干什么」：**

- **`MessagesPlaceholder("history")`**：在 prompt 里挖一个坑，运行时往里填历史消息列表。没有它，历史无处可放。
- **`get_history(session_id)`**：记忆的真正存储。这里用内存字典按 `session_id` 隔离不同用户/会话——换成 Redis、数据库就是生产级持久化记忆。
- **`RunnableWithMessageHistory`**：核心外壳。每次调用它会**自动**：①按 session 取出历史填进 `history` 占位 → ②跑链 → ③把这轮的用户输入和模型回复**追加写回**历史。我们不用手动管理拼接。
- **`config` 里的 `session_id`**：决定这次调用读写哪一份历史。不同 session_id 互不串话，这是多用户并发的关键。

第二次问「我叫什么名字」，模型能答出「小明」，正是因为外壳把第一轮的对话历史填回了 prompt。

---

## 关键原理

1. **记忆 = 存储 + 拼接**：模型无状态，记忆全靠我们把历史塞回请求。任何记忆策略，区别只在「存什么、拼多少」。
2. **Buffer 策略**：全量保存、全量拼回。优点忠实、实现简单；缺点 token 随轮数线性增长，长对话会超上下文窗口、变慢变贵 → 引出摘要记忆（03-09）。
3. **session 隔离**：`get_history` 按 key 返回不同历史对象，是多用户系统的基本盘。`InMemoryChatMessageHistory` 仅供开发；生产用 `RedisChatMessageHistory`、`SQLChatMessageHistory` 等持久化实现。
4. **为何弃用旧 Memory 类**：`ConversationBufferMemory` 与 Runnable/LCEL 体系不兼容、状态管理隐式且难组合。`RunnableWithMessageHistory` 是通用的「历史装饰器」，能包住任意链；而在 Agent 场景，更推荐用 **LangGraph 的 checkpointer**（见 Stage 4）统一管理对话状态——它把记忆、工具调用、分支都纳入同一套持久化状态。

---

## 你来改

- [ ] 用两个不同的 `session_id` 各对话一轮，验证它们互不串记忆。
- [ ] 把 `InMemoryChatMessageHistory` 的历史在每轮后打印出来，观察消息是怎么一条条累积的。
- [ ] 思考：连续聊 50 轮后请求会发生什么？这正是 03-09 摘要记忆要解决的问题。

---

## 面试怎么考

**Q：LLM 是无状态的，多轮对话是怎么「记住」上下文的？**
A：靠每次请求把历史消息一起发给模型。所谓记忆就是「存储历史 + 每轮拼接回 prompt」。框架（如 `RunnableWithMessageHistory`）只是自动化了取历史、填占位、写回这三步。

**Q：`ConversationBufferMemory` 为什么被弃用？现在用什么？**
A：它与 LCEL/Runnable 体系不兼容、接口僵硬、状态隐式难组合。现代用 `RunnableWithMessageHistory` 给任意链套历史管理；Agent 场景更推荐 LangGraph 的 checkpointer 持久化状态，统一管理记忆与流程。

**Q：Buffer 记忆有什么缺点？怎么缓解？**
A：全量保存历史，token 随轮数线性增长，长对话会撑爆上下文窗口并增加延迟与成本。缓解手段：窗口记忆（只留最近 N 轮）、摘要记忆（把旧对话滚动压缩成摘要，见 03-09）、或两者结合。
