# 01-04 `Runnable.invoke()`：同步调用与异常处理

> 🎯 **目标**：理解 `invoke()` 的调用本质、输入输出契约，以及常见异常的处理方式。

---

## 一句话功能

`invoke()` 是 LangChain `Runnable` 接口的**统一同步调用入口**。无论你的 Chain 由多少个组件拼接而成，对外暴露的始终只有一个 `invoke(input)` 方法。

---

## 核心代码示例

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from common.llm_provider import get_llm

llm = get_llm()

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位技术助教。"),
    ("human", "请解释什么是 {concept}？"),
])

parser = StrOutputParser()
chain = prompt | llm | parser

# 本质：触发整条链的同步执行，从第一个组件到最后一个组件顺序流转
result = chain.invoke({"concept": "LangChain"})
print(result)
```

---

## `invoke()` 的本质

### 1. 它是 `Runnable` 协议的统一接口

LangChain 中几乎所有核心组件都实现了 `Runnable` 接口：

| 组件 | 是否实现 Runnable | 输入类型 | 输出类型 |
|:---|:---|:---|:---|
| `ChatPromptTemplate` | ✅ | `dict` | `ChatPromptValue` |
| `ChatOpenAI` | ✅ | `ChatPromptValue` / `List[BaseMessage]` | `AIMessage` |
| `StrOutputParser` | ✅ | `AIMessage` / `str` | `str` |
| `PydanticOutputParser` | ✅ | `str` | `BaseModel` 子类 |

`invoke()` 就是这套协议的**统一调用约定**。你不需要关心组件内部怎么实现，只要知道：给它正确的输入类型，它就能返回预期的输出类型。

### 2. 内部执行顺序

```python
result = chain.invoke({"concept": "LangChain"})
```

执行流程：

```
invoke(input)
    └── RunnableSequence.invoke(input)
            ├── first.invoke(input)       → prompt.invoke({"concept": "LangChain"})
            ├── middle[0].invoke(output)  → llm.invoke(ChatPromptValue)
            ├── middle[1].invoke(output)  → ...（如果有更多中间节点）
            └── last.invoke(output)       → parser.invoke(AIMessage)
                    → return 最终结果
```

**每一步都是同步阻塞调用**，直到最后一步返回，整个 `invoke()` 才结束。

### 3. 输入输出类型由链的最后一环决定

```python
chain = prompt | llm | StrOutputParser()
# invoke 返回 str

chain = prompt | llm | PydanticOutputParser(pydantic_object=Person)
# invoke 返回 Person 对象
```

---

## 关键参数：`config`

`invoke()` 的完整签名是：

```python
invoke(input, config=None, **kwargs)
```

`config` 字典可以传入一些运行时配置，最常用的有两个：

### `callbacks`：注入回调，用于 LangSmith 追踪

```python
from langchain.callbacks.tracers import LangChainTracer

tracer = LangChainTracer(project_name="learn_llm")
result = chain.invoke(
    {"concept": "LangChain"},
    config={"callbacks": [tracer]}
)
```

### `tags` 和 `metadata`：给调用打标签，方便日志过滤

```python
result = chain.invoke(
    {"concept": "LangChain"},
    config={
        "tags": ["stage1", "demo"],
        "metadata": {"user_id": "han", "session_id": "abc123"}
    }
)
```

---

## 常见异常处理

### `OutputParserException`

当模型输出不符合预期格式时，`PydanticOutputParser` 或 `JsonOutputParser` 会抛出此异常。

```python
from langchain_core.exceptions import OutputParserException

try:
    result = chain.invoke({"concept": "LangChain"})
except OutputParserException as e:
    print(f"解析失败，原始输出：{e.llm_output}")
    # 记录日志，或 fallback 到默认回答
```

### `AuthenticationError` / `RateLimitError`

来自 OpenAI SDK 的异常，LangChain 会直接透传。

```python
from openai import AuthenticationError, RateLimitError

try:
    result = chain.invoke({"concept": "LangChain"})
except AuthenticationError:
    print("API Key 无效")
except RateLimitError:
    print("请求频率超限或余额不足")
```

---

## `invoke()` vs `batch()` vs `stream()`

| 方法 | 作用 | 返回类型 | 适用场景 |
|:---|:---|:---|:---|
| `invoke()` | 单次同步调用 | 单个结果 | 大多数场景 |
| `batch()` | 批量同步调用 | 结果列表 | 一次处理多条输入 |
| `stream()` | 流式调用 | 生成器（逐 chunk） | 需要实时显示 |

```python
# batch 示例：一次处理 3 个概念
results = chain.batch([
    {"concept": "LangChain"},
    {"concept": "RAG"},
    {"concept": "Agent"},
])
# 返回列表：[str, str, str]
```

---

## 面试考点

**Q：`invoke()` 的底层执行顺序是什么？**  
A：`invoke()` 会递归触发 `RunnableSequence` 的顺序执行：从 `first` 开始，依次遍历 `middle` 列表，最后执行 `last`。前一组件的输出作为下一组件的输入，整个过程是同步阻塞的。

**Q：`invoke()` 和 `batch()` 的区别？**  
A：`invoke()` 处理单条输入，返回单个结果；`batch()` 接收输入列表，内部可能做并行优化（如同时发起多个 LLM 请求），返回结果列表。`batch()` 的默认并发数可以通过 `config` 里的 `max_concurrency` 控制。

**Q：如何在 `invoke()` 里开启 LangSmith 追踪？**  
A：在 `config` 参数里传入 `{"callbacks": [LangChainTracer(project_name=xxx)]}`，或者在环境变量里设置 `LANGCHAIN_TRACING_V2=true` 和 `LANGCHAIN_API_KEY`，实现全局自动追踪。

---

## 课后任务

- [ ] 用 `chain.invoke()` 成功运行一条包含 Prompt + LLM + Parser 的链
- [ ] 在 `invoke()` 中传入 `config={"tags": ["test"]}`，观察 LangSmith 中是否出现该标签（如有配置 LangSmith）
- [ ] 用 `chain.batch()` 一次性处理 3 条不同的输入，观察返回的列表结构
- [ ] 捕获一次 `OutputParserException`，打印 `e.llm_output` 查看模型原始输出
