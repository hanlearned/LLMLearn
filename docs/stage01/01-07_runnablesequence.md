# 01-07 `RunnableSequence` 与 `|`：链式组合原理

> 🎯 **目标**：理解 `prompt | llm | parser` 的语法本质、数据流向与内部实现，掌握 LCEL（LangChain Expression Language）最核心的链式编排能力。

---

## 一段代码引出核心问题

```python
chain = prompt | llm | parser
result = chain.invoke({"concept": "LangChain"})
```

为什么 Python 里可以用 `|`（管道符）把三个对象串起来？数据是怎么流动的？`prompt` 的输出凭什么能直接喂给 `llm`？

---

## `|` 的本质：运算符重载

LangChain 给所有核心组件都定义了一个统一的基类——`Runnable`。`ChatPromptTemplate`、`ChatOpenAI`、`StrOutputParser` 都继承自它。

在 `Runnable` 中，Python 的 `|` 运算符被重载为 `__or__` 方法：

```python
# 伪代码，示意核心逻辑
class Runnable:
    def __or__(self, other: Runnable) -> RunnableSequence:
        return RunnableSequence(self, other)
```

所以：

```python
chain = prompt | llm | parser
```

等价于：

```python
chain = RunnableSequence(
    first=prompt,
    middle=[llm],
    last=parser,
)
```

也等价于逐步拼接：

```python
step1 = prompt.__or__(llm)        # RunnableSequence(prompt, llm)
chain = step1.__or__(parser)      # RunnableSequence(prompt, llm, parser)
```

---

## 执行顺序：从左到右的流水线

`chain.invoke(input)` 的内部执行顺序是严格**从左到右**的：

```
用户输入 ──► prompt.invoke() ──► llm.invoke() ──► parser.invoke() ──► 最终结果
  dict          messages            AIMessage            str
```

1. **`prompt.invoke(input)`**：把用户传入的字典变量渲染成消息列表（`ChatPromptValue` / `List[BaseMessage]`）。
2. **`llm.invoke(messages)`**：调用大模型，返回 `AIMessage` 对象。
3. **`parser.invoke(AIMessage)`**：从模型输出中提取纯文本或结构化数据，返回 `str` / `dict` / `Pydantic Model`。

前一组件的输出类型，恰好是后一组件的输入类型，整条链才能严丝合缝地运转。

---

## 内部实现简介

`RunnableSequence` 的 `invoke` 方法核心逻辑非常简单（伪代码）：

```python
def invoke(self, input, config):
    # 1. 执行第一个组件
    output = self.first.invoke(input, config)

    # 2. 依次执行中间组件
    for step in self.middle:
        output = step.invoke(output, config)

    # 3. 执行最后一个组件
    return self.last.invoke(output, config)
```

就是这么一个**顺序调用 + 状态传递**的过程。它的工程价值在于：

- **统一接口**：无论链多长，对外暴露的始终只有一个 `invoke()`。
- **可追踪**：LangSmith 可以把链里每一步的输入输出都记录下来。
- **可扩展**：随时在中间插入新组件（如 `Memory`、`Retriever`、`Router`），而不用重构调用代码。

---

## 为什么要用 `|` 而不是直接函数嵌套？

对比两种写法：

```python
# 写法 A：嵌套调用
result = parser.invoke(llm.invoke(prompt.invoke({"concept": "LangChain"})))

# 写法 B：管道组合
chain = prompt | llm | parser
result = chain.invoke({"concept": "LangChain"})
```

写法 B 的优势：

1. **可读性**：数据流向一目了然，像 Unix 管道的 `cat file | grep x | wc -l`。
2. **可复用**：`chain` 可以被到处传递、持久化、部署为 API。
3. **可观测**：LangChain 对 `RunnableSequence` 做了大量内置优化，包括异常回退、回调钩子、流式透传等。

---

## 完整示例

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="deepseek-chat", ...)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位善于用通俗类比讲解技术概念的助教。"),
    ("human", "请解释什么是 {concept}。"),
])

parser = StrOutputParser()

# 用管道符组装 Chain
chain = prompt | llm | parser

# 调用时只需给最左侧 Prompt 需要的变量
result = chain.invoke({"concept": "LangChain"})
print(result)
```

---

## 常见误区

### Q：`|` 两边对象的顺序能不能换？
**不能**。`llm | prompt` 会导致类型不匹配（`llm` 期望 `List[BaseMessage]`，但 `prompt` 的输出是字符串），运行时会报错。

### Q：链里可以插非 LangChain 对象吗？
**可以，但有限制**。只要对象实现了 `Runnable` 协议（有 `invoke` / `ainvoke` / `stream` / `batch` 方法），就能用 `|` 拼接。纯 Python 函数可以用 `RunnableLambda` 包装后接入。

---

## 面试考点

**Q：`prompt | llm | parser` 的 `|` 是怎么实现的？**  
A：`Runnable` 基类重载了 `__or__` 方法，每执行一次 `|` 都会生成一个 `RunnableSequence` 对象。调用 `invoke()` 时，内部按 `first → middle → last` 的顺序依次执行，并把前一组件的输出作为下一组件的输入。

**Q：LCEL 的 Chain 和普通函数嵌套有什么区别？**  
A：LCEL Chain 是声明式的，具备统一的 `Runnable` 接口，支持流式透传、并行扩展（`RunnableParallel`）、自动追踪和部署（LangServe）。普通函数嵌套虽然也能跑通，但丧失了框架带来的编排、观测和工程化能力。

**Q：如果 Chain 中间某一步报错了，怎么处理？**  
A：LangChain 支持在 `Runnable` 上绑定 `.with_fallbacks()` 做降级，或者在外层用 `try/except` 捕获。更复杂的容错可以结合 `RunnableLambda` 写自定义异常处理节点。

---

## 课后任务

- [ ] 用 `|` 把 `ChatPromptTemplate`、`ChatOpenAI`、`StrOutputParser` 组装成一条 Chain，并成功运行
- [ ] 尝试在中间插入一个自定义的 `RunnableLambda`，比如把输出打印到控制台
- [ ] 把 `chain = prompt | llm | parser` 改写成 `RunnableSequence` 的显式写法，体会两者的等价性
