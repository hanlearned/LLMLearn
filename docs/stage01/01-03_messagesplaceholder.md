# 01-03 `MessagesPlaceholder`：动态消息列表占位

> 🎯 **目标**：理解 `MessagesPlaceholder` 的作用，掌握如何在 Prompt 模板中动态插入历史对话记录。

---

## 一句话功能

`MessagesPlaceholder` 是 `ChatPromptTemplate` 的**占位符组件**，用于在 Prompt 的指定位置**动态插入一条消息列表**（通常是历史对话记录），而不是单条固定模板消息。

---

## 为什么需要它？

前面的 `ChatPromptTemplate.from_messages()` 只能定义**固定结构**的模板：

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位助教。"),
    ("human", "{question}"),
])
```

但在多轮对话或 Agent 场景中，历史消息的数量是**不固定的**——可能 2 轮、5 轮、10 轮。你无法用固定模板提前写好所有位置。

`MessagesPlaceholder` 就是来解决这个问题的：**它占一个坑，运行时你把多少条消息塞进去，它就展开多少条。**

---

## 核心代码示例

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 本质：在 system 和当前 human 问题之间，插入一个动态的消息列表占位符
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位耐心的数学老师。"),
    MessagesPlaceholder(variable_name="history"),  # ← 占位符
    ("human", "{question}"),
])

# 运行时注入历史消息（数量任意）
messages = prompt.invoke({
    "history": [
        HumanMessage(content="1+1=？"),
        AIMessage(content="2"),
        HumanMessage(content="为什么不是 11？"),
        AIMessage(content="因为在十进制里，1+1 表示两个一相加。"),
    ],
    "question": "那二进制呢？",
})

print(messages)
```

输出：
```text
ChatPromptValue(
    messages=[
        SystemMessage(content='你是一位耐心的数学老师。'),
        HumanMessage(content='1+1=？'),
        AIMessage(content='2'),
        HumanMessage(content='为什么不是 11？'),
        AIMessage(content='因为在十进制里，1+1 表示两个一相加。'),
        HumanMessage(content='那二进制呢？')
    ]
)
```

---

## `MessagesPlaceholder` 的本质

### 1. 它是一个"列表展开器"

`MessagesPlaceholder` 本身不生产消息，它只是**把传入的消息列表原样展开**，插入到 Prompt 的对应位置。

```python
MessagesPlaceholder(variable_name="history")
# 运行时："history" 对应 List[BaseMessage]，直接展开为多条消息
```

### 2. 和 `HumanMessagePromptTemplate` 的区别

| 组件 | 输入 | 输出 | 场景 |
|:---|:---|:---|:---|
| `("human", "{text}")` | 字符串 | 单条 `HumanMessage` | 固定模板消息 |
| `MessagesPlaceholder` | `List[BaseMessage]` | 多条消息原样展开 | 动态历史记录 |

### 3. 变量名必须匹配

`MessagesPlaceholder(variable_name="history")` 要求运行时传入的字典里必须有 `"history"` 这个 key，且值是 `List[BaseMessage]` 类型。如果传入字符串会报错。

---

## 常见用法：配合记忆组件

在实际工程中，`MessagesPlaceholder` 几乎总是和**记忆组件**一起使用：

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
memory.save_context({"input": "你好"}, {"output": "你好！有什么可以帮你的？"})

# 从记忆中加载历史消息
history = memory.load_memory_variables({})["history"]

# 注入到 Prompt
messages = prompt.invoke({"history": history, "question": "今天天气怎么样？"})
```

**记忆组件负责存取历史，`MessagesPlaceholder` 负责在 Prompt 中留出位置。** 两者配合，才能实现有"上下文记忆"的对话。

---

## 常见误区

### Q：`MessagesPlaceholder` 可以插入单条消息吗？
A：**可以**，但语义不对。如果你只传一条消息，它也能展开；但它的设计意图是处理**数量不固定的列表**。单条消息直接用 `("human", "{text}")` 更清晰。

### Q：占位符放的位置有影响吗？
A：**有影响**。通常放在 `system` 之后、`human` 当前问题之前，这样模型能看到"系统设定 → 历史对话 → 当前问题"的完整上下文。如果放在 `system` 之前，系统指令可能被历史消息淹没。

### Q：历史消息太长怎么办？
A：这是记忆组件（`ConversationBufferMemory`、`ConversationSummaryMemory`）的职责，不是 `MessagesPlaceholder` 的职责。Placeholder 只管"插入"，不管"截断"或"摘要"。

---

## 面试考点

**Q：`MessagesPlaceholder` 和普通的 `("human", "{text}")` 模板有什么区别？**  
A：普通模板每次只生成一条固定角色的消息；`MessagesPlaceholder` 接收一个消息列表并**原样展开为多条消息**，用于处理数量不固定的动态内容（如历史对话记录）。

**Q：没有 `MessagesPlaceholder` 能实现多轮对话吗？**  
A：理论上可以手动拼接消息列表后传给 `llm.invoke(messages)`，但这样就丧失了 `ChatPromptTemplate` 的模板化管理能力。`MessagesPlaceholder` 让"历史消息"成为 Prompt 模板的一个标准化插槽，是工程化的必要组件。

**Q：`MessagesPlaceholder` 在 Agent 中起什么作用？**  
A：Agent 的 ReAct 循环会产生大量"思考-行动-观察"消息，数量完全不固定。`MessagesPlaceholder` 把这些中间步骤统一插入到 Prompt 中，让模型看到完整的决策链条。

---

## 课后任务

- [ ] 用 `MessagesPlaceholder` 定义一个支持历史对话的 Prompt 模板
- [ ] 手动构造 3 轮对话历史（`HumanMessage` + `AIMessage` 交替），注入模板并观察输出
- [ ] 尝试把 `MessagesPlaceholder` 放到 `system` 消息之前，观察模型是否能正确遵循系统指令
