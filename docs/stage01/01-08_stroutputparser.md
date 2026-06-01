# `StrOutputParser`：把模型输出提取成纯字符串

`StrOutputParser` 是最简单的输出解析器：它把模型返回的 `AIMessage` 对象里的 `.content` 抠出来，直接给你一个 `str`。

## 为什么需要它

`llm.invoke(...)` 返回的不是字符串，而是一个 `AIMessage` 对象（带 content、metadata、tool_calls 等）。如果你只想要那段文本，每次都写 `.content` 很啰嗦，更重要的是——**在 LCEL 管道里没法接 `.content`**。

```python
chain = prompt | llm            # 输出是 AIMessage
chain = prompt | llm | StrOutputParser()   # 输出直接是 str，干净
```

放在链路末尾，下游（比如另一个 prompt、或 FastAPI 返回）就能直接用字符串，不用再手动取 content。

## 核心用法

```python
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from common.llm_provider import get_llm

prompt = ChatPromptTemplate.from_template("用一句话解释 {word}")
chain = prompt | get_llm() | StrOutputParser()

print(chain.invoke({"word": "RAG"}))   # 直接打印字符串，不是 AIMessage
```

**逐块讲解**：
- `prompt | get_llm()`：到这一步输出是 `AIMessage`。
- `| StrOutputParser()`：解析器接收 `AIMessage`，返回 `message.content`（一个 str）。
- 流式时它也兼容：`chain.stream(...)` 会逐块吐出字符串片段。

## 关键原理

`StrOutputParser` 本质就是一个 `Runnable`，它的 `parse()` 对 `AIMessage` 取 `.content`、对纯字符串原样返回。它无状态、零配置，是 LCEL 链里最常见的「收尾」组件。

## 你来改
1. 去掉 `| StrOutputParser()`，打印结果，观察 `AIMessage` 和 `str` 的区别。
2. 在解析器后再接一个 `RunnableLambda(lambda s: s.upper())`，把英文输出转大写。

## 面试怎么考
- **「`prompt | llm` 的输出能直接当字符串用吗？」** → 不能，是 `AIMessage`，要 `.content` 或接 `StrOutputParser`。
- **「StrOutputParser 和 JsonOutputParser 区别？」** → 前者只取纯文本；后者还会把文本按 JSON 解析成 dict（见下一篇）。
