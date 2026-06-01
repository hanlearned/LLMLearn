# 06-03 流式 API：astream + SSE，让回答「逐字蹦出来」（重点篇）

> 🎯 **一句话**：用 LangChain 的 `astream` 配合 SSE（Server-Sent Events），把模型生成的 token 边产边推给前端，实现 ChatGPT 那样的打字机效果——大幅降低用户感知延迟，是 LLM 产品的体验生命线。

---

## 为什么需要它

LLM 生成一段长回答可能要 5~15 秒。若用普通接口，用户得**干等全部生成完**才看到一个字，体验极差，还容易被当成「卡死」。

流式（streaming）把生成过程实时推送：模型吐一个 token，就立刻发给前端显示。用户在**第一个字出现的瞬间**（首 token 延迟，TTFT）就知道系统在工作，感知延迟从「总时长」骤降到「首 token 时间」。SSE 是实现服务端单向流式推送的最简协议，天然契合「服务器持续往客户端发文字」的场景。

---

## 核心用法

### 1. 用 StreamingResponse 手写 SSE（零额外依赖）

```python
# main.py — uvicorn main:app --reload
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from common.llm_provider import get_llm

app = FastAPI()
chain = ChatPromptTemplate.from_template("{q}") | get_llm(temperature=0.7)

class Req(BaseModel):
    q: str

async def sse_generator(question: str):
    # astream 逐块产出 AIMessageChunk，每块拿 .content 推给前端
    async for chunk in chain.astream({"q": question}):
        if chunk.content:
            yield f"data: {chunk.content}\n\n"   # SSE 格式：data: 内容\n\n
    yield "data: [DONE]\n\n"                      # 约定的结束信号

@app.post("/stream")
async def stream(req: Req):
    return StreamingResponse(
        sse_generator(req.q),
        media_type="text/event-stream",          # SSE 的关键 MIME 类型
    )
```

**逐块讲解：**
- **`chain.astream(...)`**：异步流式调用，每 yield 一个 `AIMessageChunk`（增量 token）。这是流式的源头。
- **SSE 帧格式**：每条消息必须是 `data: 内容\n\n`（两个换行结尾）。浏览器 EventSource 据此切分事件。
- **`media_type="text/event-stream"`**：告诉客户端这是 SSE 流，连接保持打开、持续接收。
- **`[DONE]` 结束信号**：SSE 本身不标识结束，约定发一个 `[DONE]` 让前端知道可以收尾。

### 2. 用 sse-starlette（更规范，自动管理 SSE 细节）

```python
from sse_starlette.sse import EventSourceResponse

@app.post("/stream2")
async def stream2(req: Req):
    async def gen():
        async for chunk in chain.astream({"q": req.q}):
            if chunk.content:
                yield {"data": chunk.content}     # 只给 data，库帮你拼 SSE 帧
        yield {"data": "[DONE]"}
    return EventSourceResponse(gen())
```

**本质在干什么？** `sse-starlette` 的 `EventSourceResponse` 替你处理 SSE 帧格式、心跳保活、断连清理，你只管 yield `{"data": ...}`。生产更推荐它，少踩格式与保活的坑。

### 3. 前端如何消费

```javascript
// 浏览器原生 EventSource（GET）；POST 场景常用 fetch + ReadableStream 读取
const resp = await fetch("/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ q: "讲个笑话" }),
});
const reader = resp.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  const text = decoder.decode(value);
  for (const line of text.split("\n\n")) {
    if (line.startsWith("data: ")) {
      const content = line.slice(6);
      if (content === "[DONE]") break;
      document.getElementById("out").textContent += content;  // 逐字追加
    }
  }
}
```

**本质在干什么？** 前端持续 `read()` 流，解析每个 `data:` 帧，把增量内容追加到页面——就形成打字机效果。遇到 `[DONE]` 停止。

---

## 关键原理 / 实践要点

1. **首 token 延迟（TTFT）才是体验关键**：流式没有缩短总生成时间，但把用户感知延迟从「总时长」变成「首 token 时间」。优化 TTFT（如减小 prompt、用更快模型）对体验收益巨大。
2. **必须全链路 async**：`astream` + `async def` 生成器 + ASGI 服务器，任一环阻塞都会让流卡顿。
3. **SSE vs WebSocket**：SSE 是服务端→客户端单向、基于 HTTP、自动重连、实现简单，**LLM 文本流首选**；需要双向实时（如语音）才用 WebSocket。
4. **`astream` vs `astream_events`**：`astream` 给最终输出的 token；`astream_events` 给链中**每个节点**的事件（含中间步骤、工具调用），做 Agent 过程可视化时用后者。
5. **结束信号与错误处理**：约定 `[DONE]` 标识完成；生成中途报错要在流里发一个错误事件，别让前端无限等待。代理/Nginx 需关闭对该路由的缓冲（`X-Accel-Buffering: no`），否则流会被攒着一起发。

---

## 你来改

- [ ] 在 `sse_generator` 里用 `time` 记录从请求到第一个 chunk 的耗时，打印 TTFT，感受它远小于总时长。
- [ ] 把 `astream` 换成 `astream_events(version="v2")`，过滤出 `on_chat_model_stream` 事件，对比两种 API。
- [ ] 故意在生成中途抛异常，给前端发一个 `data: [ERROR]` 事件并优雅收尾。

---

## 面试怎么考

**Q：LLM 应用为什么要做流式？首 token 延迟是什么意思？**
A：长回答需等数秒至十几秒，非流式下用户要等全部生成完才看到内容，体验差。流式边生成边推送，用户在第一个 token 出现时就感知到系统在响应。首 token 延迟（TTFT）即从请求到收到第一个 token 的时间；流式不减少总生成时长，但把感知延迟降到 TTFT，所以优化 TTFT 收益最大。

**Q：为什么用 SSE 而不是 WebSocket？怎么实现？**
A：LLM 文本流是服务端单向持续推送，SSE 基于 HTTP、单向、自动重连、实现简单，正好匹配，且无需 WebSocket 的双向握手开销。实现上 FastAPI 用 StreamingResponse（media_type=text/event-stream）或 sse-starlette 的 EventSourceResponse，内部 `async for chunk in chain.astream(...)` 逐块 yield `data: 内容\n\n`，发 `[DONE]` 收尾。需要双向实时（语音）才用 WebSocket。

**Q：astream 和 astream_events 区别？流式有哪些坑？**
A：astream 产出最终输出的 token；astream_events 产出链中每个节点的事件（含工具调用、中间步骤），适合 Agent 过程可视化。常见坑：全链路必须 async 否则卡顿；要约定结束信号 [DONE]；中途错误要发错误事件；Nginx/代理要关缓冲（X-Accel-Buffering: no）否则流被攒着一起发，失去流式意义。
