"""
Stage 6 · 用 FastAPI 把 LLM 做成「流式」HTTP 服务

为什么要流式（streaming）？
- LLM 生成一段长回答要好几秒。一次性等全部生成完再返回，用户盯着空白屏幕，体验很差。
- 流式让 token 一边生成一边吐给前端（像 ChatGPT 那样逐字蹦），「首字延迟」从几秒降到几百毫秒。
- 技术上用 SSE（Server-Sent Events）：服务端持续往同一个 HTTP 连接里推数据。

本文件提供两个接口对比：
    POST /chat         普通：等全部生成完一次性返回
    POST /chat/stream  流式：SSE 逐块推送（生产推荐）

启动（文件名以数字开头，不能用 uvicorn 模块路径，直接运行本文件即可）：
    pip install fastapi uvicorn
    python stage06_deployment/01_fastapi_streaming.py
    # 浏览器开 http://127.0.0.1:8000/docs
    # 流式测试：curl -N -X POST http://127.0.0.1:8000/chat/stream -H "Content-Type: application/json" -d '{"message":"用三句话介绍 RAG"}'
"""

import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from common.llm_provider import get_llm

app = FastAPI(title="LLM 流式服务示例")
llm = get_llm(temperature=0.7)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    """普通接口：阻塞直到生成完毕，一次性返回完整答案。"""
    return {"answer": llm.invoke(req.message).content}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式接口：用 SSE 把 token 一块块推给前端。"""

    async def event_generator():
        # llm.astream 是异步生成器，每次 yield 一个内容块（chunk）
        async for chunk in llm.astream(req.message):
            if chunk.content:
                # SSE 协议格式：每条消息以 "data: " 开头、"\n\n" 结尾
                yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"  # 约定一个结束标记，前端据此停止接收

    # media_type 必须是 text/event-stream，浏览器/前端才会按 SSE 解析
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
