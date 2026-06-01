"""
Capstone 后端 · FastAPI 服务

把多 Agent 客服图暴露成 HTTP 接口，并托管前端页面。核心接口：
    POST /api/chat     {session_id, message}  → {answer, needs_approval}
    POST /api/approve  {session_id, approve}  → {answer}   （退款审批）
    GET  /api/health
    GET  /                                    → 前端聊天页面

needs_approval=true 时，前端会弹出「批准/拒绝」按钮，调用 /api/approve 决定是否真的退款——
这就是 Human-in-the-Loop 在产品层的体现。

直接运行：
    pip install langgraph langgraph-checkpoint-sqlite langchain-chroma fastapi uvicorn
    python capstone/backend/api.py
    # 浏览器打开 http://127.0.0.1:8000/
"""

from dotenv import load_dotenv

load_dotenv()

import config  # noqa: F401,E402  （把仓库根加入 sys.path）
import graph as g  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

app = FastAPI(title="云启智能客服 · Capstone", description="多 Agent + RAG + HITL + 持久记忆")
_graph = None


@app.on_event("startup")
def _startup():
    global _graph
    _graph = g.build_graph(g.get_checkpointer())


class ChatReq(BaseModel):
    session_id: str = Field(..., examples=["user-A"])
    message: str = Field(..., examples=["10001 这个订单我想退款"])


class ApproveReq(BaseModel):
    session_id: str
    approve: bool


def _cfg(session_id: str):
    return {"configurable": {"thread_id": session_id}}


@app.post("/api/chat")
def chat(req: ChatReq):
    cfg = _cfg(req.session_id)
    result = _graph.invoke({"messages": [("user", req.message)]}, config=cfg)
    # 若图停在 execute_refund 之前，说明触发了退款、正等待人工审批
    snapshot = _graph.get_state(cfg)
    needs_approval = bool(snapshot.next) and "execute_refund" in snapshot.next
    return {"answer": result.get("answer", ""), "needs_approval": needs_approval}


@app.post("/api/approve")
def approve(req: ApproveReq):
    cfg = _cfg(req.session_id)
    # 把审批结果写进状态，再恢复执行（传 None=从中断点继续）
    _graph.update_state(cfg, {"approved": req.approve})
    result = _graph.invoke(None, config=cfg)
    return {"answer": result.get("answer", "")}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(str(config.FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
