"""
Capstone · 部署层：把客服 Agent 做成带会话与流式的 HTTP 服务

能力6（工程化部署）的落地：
- 每个用户用 session_id 隔离对话记忆（映射到 Agent 的 thread_id）
- 提供流式接口，回答逐字返回，体验接近真实在线客服
- 同一个 Agent 实例常驻，多用户靠 thread_id 区分上下文

直接运行：
    pip install langgraph langchain-chroma fastapi uvicorn
    python capstone/api.py
    # 调试：http://127.0.0.1:8000/docs
"""

import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import agent as cs_agent  # noqa: E402  复用 agent.py 的全部业务逻辑

app = FastAPI(title="云启智能客服 API", description="Capstone · RAG + Agent + 记忆 + 部署")
_agent = None


@app.on_event("startup")
def _startup():
    global _agent
    _agent = cs_agent.build_agent()


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="会话 ID，用于隔离不同用户的多轮记忆", examples=["user-A"])
    message: str = Field(..., examples=["我的订单 10001 到哪了？"])


@app.post("/chat")
def chat(req: ChatRequest):
    """普通问答：返回完整答案。"""
    answer = cs_agent.chat(_agent, req.message, session_id=req.session_id, show_trace=False)
    return {"answer": answer}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式问答：用 SSE 把 Agent 最终回复逐 token 推送。"""
    config = {"configurable": {"thread_id": req.session_id}}

    async def gen():
        # stream_mode="messages" 让 LangGraph 按 token 粒度流式输出消息块
        async for chunk, _meta in _agent.astream(
            {"messages": [("user", req.message)]}, config=config, stream_mode="messages"
        ):
            # 只推模型生成的文本块（跳过工具消息等）
            if getattr(chunk, "content", "") and chunk.type == "AIMessageChunk":
                yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
