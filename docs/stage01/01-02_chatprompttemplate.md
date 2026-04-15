# 01-02 `ChatPromptTemplate`：消息模板与变量注入

> 🎯 **目标**：理解 `ChatPromptTemplate` 的底层数据结构，掌握 `SystemMessage`、`HumanMessage`、`AIMessage` 的组合与变量注入。

---

## 从字符串到消息对象

大模型对话的本质不是"发一段字符串"，而是**发一个消息列表**。每条消息都有角色（role）和内容（content）：

- `system`：设定 AI 的角色和行为准则
- `human` / `user`：用户的输入
- `ai` / `assistant`：模型的回复（常用于 few-shot）

LangChain 用 `ChatPromptTemplate` 把这个过程**模板化**。

---

## 核心代码示例

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位擅长用通俗类比讲解技术概念的助教。"),
        ("human", "请用 {style} 的风格解释什么是 {concept}。"),
    ]
)

# 变量注入
messages = prompt.invoke({
    "concept": "LangChain",
    "style": "给小学生讲课"
})

print(messages)
```

输出：
```text
ChatPromptValue(
    messages=[
        SystemMessage(content='你是一位擅长用通俗类比讲解技术概念的助教。'),
        HumanMessage(content='请用 给小学生讲课 的风格解释什么是 LangChain。')
    ]
)
```

---

## 底层结构解析

`ChatPromptTemplate` 最终会被解析为一个**消息列表**（`List[BaseMessage]`），这是所有 Chat Model 的标准输入格式。它的好处是：

1. **可复用**：Prompt 结构写一次，运行时动态填充变量
2. **可管理**：可以把 Prompt 存到文件、数据库或 Prompt Hub 里
3. **可观测**：LangSmith 能直接看到每个角色的消息内容

---

## 带历史对话的模板

```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位耐心的数学老师。"),
        ("human", "{question}"),
        ("ai", "{ai_answer}"),
        ("human", "{follow_up_question}"),
    ]
)

messages = prompt.invoke({
    "question": "1+1=？",
    "ai_answer": "2",
    "follow_up_question": "为什么不是 11？"
})
```

这就是 **Few-Shot Prompting** 的基础写法。

---

## `partial`：部分变量预设

有时候某些变量是固定的，只在运行时才知道一部分：

```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位 {role}。"),
        ("human", "{question}"),
    ]
)

# 先固定 role
partial_prompt = prompt.partial(role="Python 专家")

# 运行时只传 question
messages = partial_prompt.invoke({"question": "什么是 GIL？"})
```

---

## 面试考点

**Q：为什么不用 f-string，而要用 `ChatPromptTemplate`？**  
A：f-string 无法复用、无法版本管理、无法被 LangSmith 追踪。`ChatPromptTemplate` 是 LangChain 推荐的工业化 Prompt 管理方式，支持变量注入、`partial` 预设、序列化存储。

**Q：`SystemMessage` 对模型输出的影响有多大？**  
A：System 消息是模型的"最高指令"，优先级通常高于 Human 消息。好的 System Prompt 能显著提升输出稳定性、格式一致性和安全性。

---

## 课后任务

- [ ] 用 `ChatPromptTemplate` 定义一个包含 `system` + `human` 的模板，成功注入变量
- [ ] 尝试用 `partial()` 预设一个变量，运行时只传剩余变量
- [ ] 在模板里加入一条 `ai` 消息，实现单样本 few-shot 提示
