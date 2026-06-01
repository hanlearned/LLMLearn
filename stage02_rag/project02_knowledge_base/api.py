"""
项目 2 · 第三步：用 FastAPI 把知识库问答暴露成 HTTP 服务

「工程化前置」原则的第一次落地：从项目 2 开始，每个项目都是一个能被前端/其他系统
调用的服务，而不是只能在终端跑的脚本。

启动（直接运行本文件即可，无需配置 uvicorn 模块路径）：
    pip install fastapi uvicorn
    # 先确保已 ingest 建库
    python stage02_rag/project02_knowledge_base/api.py
    # 然后访问 http://127.0.0.1:8000/docs 在线调试

接口：
    POST /chat   body: {"question": "..."}   ->  {"answer": "...", "sources": [...]}
"""

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel, Field

import qa  # noqa: E402  复用 qa.py 里的业务逻辑

app = FastAPI(title="云启企业知识库 API", description="Stage 2 项目 2 · RAG 问答服务")

# 在应用启动时把 RAG 链建好并常驻内存，避免每次请求都重建（重要的性能习惯）
_chain = None


@app.on_event("startup")
def _startup():
    global _chain
    _chain = qa.build_chain()


class ChatRequest(BaseModel):
    question: str = Field(..., description="用户问题", examples=["公积金缴存比例是多少？"])


class Source(BaseModel):
    source: str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """对单个问题返回答案与出处。"""
    return qa.answer(_chain, req.question)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
