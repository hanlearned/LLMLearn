# 01-05 `Runnable.stream()`：流式输出与 SSE 基础

> 🎯 **目标**：掌握 `stream()` 的调用方式、Chunk 的传输机制，以及流式输出对用户体验的工程意义。

---

## 一句话功能

`Runnable.stream()` 把模型的**完整响应拆成多个文本片段（Chunk）逐块返回**，让用户在终端或前端看到"打字机"式的实时输出效果。

---

## 核心代码示例

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from common.llm_provider import get_llm

llm = get_llm()

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位擅长写短诗的诗人。"),
    ("human", "以 {theme} 为主题，写一首 4 行的现代诗。"),
])

parser = StrOutputParser()
chain = prompt | llm | parser

# 本质：发送 stream=true 请求，模型每生成一个 token 就 push 一个 chunk
for chunk in chain.stream({"theme": "深夜的代码"}):
    # end="" 防止换行破坏拼接；flush=True 强制立即刷新到屏幕
    print(chunk, end="", flush=True)
```

---

## `stream()` 的本质

### 1. 底层是 API 的 `stream=true` 参数

LangChain 在调用模型时，会自动在 HTTP 请求体里加上 `stream: true`：

```json
{
  "model": "deepseek-chat",
  "messages": [...],
  "stream": true
}
```

模型不再等全部文本生成完才返回，而是**每生成一个 token 就通过 SSE（Server-Sent Events）推送一次**。

### 2. `stream()` 返回生成器（Generator）

```python
for chunk in chain.stream({"theme": "深夜的代码"}):
    ...
```

- `chunk` 在 `StrOutputParser` 链路中是**字符串片段**（1~4 个 token）
- 内存占用极低，因为不需要一次性缓存完整响应
- 只有进入 `for` 循环时才会真正触发网络请求

### 3. 为什么必须写 `end="", flush=True`

```python
print(chunk, end="", flush=True)
```

| 参数 | 默认值 | 作用 |
|:---|:---|:---|
| `end` | `"\n"` | print 结束后追加的字符。默认换行会破坏打字机效果，改为 `""` 让多个 chunk 拼接在同一行 |
| `flush` | `False` | 是否强制刷新输出缓冲。终端默认会攒够一定数据才显示，`flush=True` 保证每个 chunk 立即出现在屏幕上 |

---

## 完整链路中的 Chunk 形态

```
用户输入(dict)
    ↓
prompt.invoke() → ChatPromptValue（完整消息列表，一次性生成）
    ↓
llm.stream()    → 逐个 yield AIMessageChunk（token 块）
    ↓
parser.stream() → 逐个 yield str（文本片段）
    ↓
终端/前端       → 实时拼接显示
```

注意：**Prompt 的渲染不是流式的**。`prompt.invoke()` 仍然是一次性生成完整消息列表，流式只发生在 `llm → parser` 这一段。

---

## 流式 + PydanticOutputParser 的局限

如果把 `stream()` 用在 `prompt | llm | PydanticOutputParser` 上，每个 `chunk` 会是**不完整的 JSON 字符串片段**，而不是结构化的对象。

```python
# 第一个 chunk: '{"name":"'
# 第二个 chunk: '张三","age":'
# 第三个 chunk: '28}'
```

`PydanticOutputParser` 只有在流完全结束后，才能拼出完整 JSON 并校验为 Pydantic 对象。LangChain 为此提供了 `astream_events()` 和 `diff` 模式，但那是高阶内容，Stage 6 讲 FastAPI + SSE 时会深入。

**工程建议**：需要结构化输出时，优先用 `invoke()`；需要实时体验时，用 `stream()` + `StrOutputParser`。

---

## 常见误区

### Q：为什么我在 PyCharm / VS Code 的运行窗口看不到打字机效果？
A：IDE 的运行窗口有输出缓冲，会把所有 chunk 攒在一起显示。在**系统终端**（Windows Terminal、iTerm、VS Code 内置终端直接 `python xxx.py`）运行才能看到真正的流式效果。

### Q：`stream()` 比 `invoke()` 更快吗？
A：**总耗时几乎一样**，甚至可能略长（因为有多次网络传输开销）。但 `stream()` 的**首字延迟极低**（First Token Latency），用户感知上的等待时间大幅减少。

---

## 面试考点

**Q：`invoke()` 和 `stream()` 的底层区别是什么？**  
A：`invoke()` 发送同步请求，等模型生成完整响应后一次性返回；`stream()` 发送流式请求（`stream=true`），通过 SSE 或 chunked transfer 逐 token 接收，降低首字延迟，提升用户体验。

**Q：为什么流式输出对用户体验很重要？**  
A：大模型生成长文本可能需要数秒甚至数十秒。流式输出让用户在**几百毫秒内就能看到第一个字**，感知上的等待时间大幅降低。这是 ChatGPT、Claude 等产品的核心体验设计之一。

**Q：LangChain 的 `stream()` 支持异步吗？**  
A：支持。对应方法是 `chain.astream()`，返回异步生成器（`AsyncIterator`），在 FastAPI 等异步框架中必须使用 `astream()` 才能避免阻塞事件循环。

---

## 课后任务

- [ ] 用 `stream()` 运行一条 Chain，在本地终端观察打字机效果
- [ ] 在循环中加入计数器，统计一共输出了多少个 chunk 和多少个字符
- [ ] 尝试把 `stream()` 和 `PydanticOutputParser` 组合，观察 chunk 的形态（不完整的 JSON 片段）
- [ ] （进阶）了解 `chain.astream()` 的用法，为后续 FastAPI 集成做准备
