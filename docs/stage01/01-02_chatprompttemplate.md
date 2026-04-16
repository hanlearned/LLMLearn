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

# 本质：把 (role, template_str) 的元组列表，转换成一组带占位符的消息模板对象。
# 底层创建了一个 ChatPromptTemplate 实例，内部持有 SystemMessagePromptTemplate 和
# HumanMessagePromptTemplate，它们各自记录着 "原始模板字符串" 和 "变量名列表"。
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位擅长用通俗类比讲解技术概念的助教。"),
        ("human", "请用 {style} 的风格解释什么是 {concept}。"),
    ]
)

# 本质：执行 "模板渲染"。
# 1. 遍历 prompt 内部的消息模板列表；
# 2. 用传入的字典 {"concept": "LangChain", "style": "给小学生讲课"} 替换占位符；
# 3. 生成最终可用的消息对象列表（List[BaseMessage]），封装在 ChatPromptValue 中返回。
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

### 关键方法速查

| 方法 | 本质在干什么 |
|------|-------------|
| `ChatPromptTemplate.from_messages(...)` | **工厂方法**。解析你传入的元组列表，为每一条消息生成对应的 `*MessagePromptTemplate` 对象，最终组装成一个可复用的 `ChatPromptTemplate` 实例。 |
| `prompt.invoke(variables)` | **模板渲染**。用 `variables` 字典填充模板中的 `{占位符}`，返回可直接喂给 LLM 的消息列表 `ChatPromptValue`。 |

---

## 底层结构解析

`ChatPromptTemplate` 最终会被解析为一个**消息列表**（`List[BaseMessage]`），这是所有 Chat Model 的标准输入格式。它的好处是：

1. **可复用**：Prompt 结构写一次，运行时动态填充变量
2. **可管理**：可以把 Prompt 存到文件、数据库或 Prompt Hub 里
3. **可观测**：LangSmith 能直接看到每个角色的消息内容

---

## 带历史对话的模板

```python
# 本质：和上面一样，只是消息模板列表里多了一条 ai 角色的模板。
# ai 角色的模板在渲染后生成 AIMessage，常用于 Few-Shot 示例。
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位耐心的数学老师。"),
        ("human", "{question}"),
        ("ai", "{ai_answer}"),
        ("human", "{follow_up_question}"),
    ]
)

# 本质：一次性为所有消息模板中的占位符填充数值，生成完整的多轮对话消息列表。
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

# 本质：基于当前 prompt 创建一个新的 ChatPromptTemplate 副本，
# 把 "role" 这个变量提前绑定为 "Python 专家"。
# 新模板内部只剩下 "question" 这一个待填充占位符。
partial_prompt = prompt.partial(role="Python 专家")

# 本质：此时 invoke 只需要传入剩余变量即可，role 已经被锁死。
messages = partial_prompt.invoke({"question": "什么是 GIL？"})
```

### `partial` 的本质

`partial()` 相当于 Prompt 模板的**柯里化（Currying）**。它不会修改原对象，而是返回一个"预设了部分变量"的新模板。这在工程上非常有用：

- **固定系统角色**：一个应用里有多个不同角色的 AI，可以先用 `partial(role="...")` 切分出不同角色的 Prompt 模板，避免运行时传错。
- **固定输出格式**：比如 `partial(format_instructions="...")` 提前把 JSON Schema 要求注入进去（这正是 `PydanticOutputParser` 的常用套路）。

### `partial` 关键方法速查

| 方法 | 本质在干什么 |
|------|-------------|
| `prompt.partial(**kwargs)` | **柯里化绑定**。创建一个新的 `ChatPromptTemplate` 副本，把 `kwargs` 中指定的变量预先填充好，返回的模板在后续 `invoke()` 时只需传入剩余变量。 |

---

## 面试考点

**Q：为什么不用 f-string，而要用 `ChatPromptTemplate`？**  
A：f-string 无法复用、无法版本管理、无法被 LangSmith 追踪。`ChatPromptTemplate` 是 LangChain 推荐的工业化 Prompt 管理方式，支持变量注入、`partial` 预设、序列化存储。

**Q：`SystemMessage` 对模型输出的影响有多大？**  
A：System 消息是模型的"最高指令"，优先级通常高于 Human 消息。好的 System Prompt 能显著提升输出稳定性、格式一致性和安全性。

**Q：`prompt.invoke()` 和 f-string 的区别是什么？**  
A：`invoke()` 不是简单的字符串替换，它会把模板中的 `{变量}` 解析出来，填充到对应角色的 `*MessagePromptTemplate` 中，最终生成一个结构化的 `ChatPromptValue`（包含 `SystemMessage`、`HumanMessage` 等对象列表）。这个结构才是 LLM 真正期望的输入格式。

---

## 课后任务

- [ ] 用 `ChatPromptTemplate` 定义一个包含 `system` + `human` 的模板，成功注入变量
- [ ] 尝试用 `partial()` 预设一个变量，运行时只传剩余变量
- [ ] 在模板里加入一条 `ai` 消息，实现单样本 few-shot 提示
