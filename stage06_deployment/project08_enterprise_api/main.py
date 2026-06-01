"""
项目 8 · 企业级 LLM API 服务平台

把前面学的一切包成一个「能上线」的服务，具备生产服务的四个基本要素：
    1. 鉴权（API Key）     —— 不是谁都能调，按 Key 区分调用方
    2. 缓存（LLM Cache）   —— 相同问题不重复花钱调模型，秒级返回
    3. 流式（SSE）         —— 长回答逐字返回，体验好、首字延迟低
    4. 健康检查 / 可观测   —— /health 供负载均衡探活

直接运行（无需配置 uvicorn 模块路径）：
    pip install fastapi uvicorn
    python stage06_deployment/project08_enterprise_api/main.py
    # 文档与调试：http://127.0.0.1:8000/docs
    # 鉴权：请求头加 X-API-Key: demo-key-123

容器化部署见同目录 Dockerfile。
"""

import pathlib
import sys
import time

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from pydantic import BaseModel

from common.llm_provider import get_llm

# 开启全局 LLM 缓存：完全相同的请求（同模型同参数同输入）直接命中缓存，不再调 API。
# 生产可换成 RedisCache 做跨实例共享。
set_llm_cache(InMemoryCache())

app = FastAPI(title="云启企业 LLM API", description="Stage 6 项目 8 · 生产级 LLM 服务")
llm = get_llm(temperature=0)

# 简化的「合法 API Key 白名单」。真实项目应存数据库并支持配额/限流。
VALID_KEYS = {"demo-key-123", "demo-key-456"}


def check_key(x_api_key: str | None):
    """鉴权：校验请求头里的 X-API-Key。"""
    if x_api_key not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="无效的 API Key，请在请求头加 X-API-Key")


class ChatRequest(BaseModel):
    message: str


@app.post("/v1/chat")
def chat(req: ChatRequest, x_api_key: str | None = Header(default=None)):
    """普通问答。演示缓存：连续问同一句，第二次会明显更快（命中缓存）。"""
    check_key(x_api_key)
    t0 = time.time()
    answer = llm.invoke(req.message).content
    elapsed = time.time() - t0
    return {"answer": answer, "elapsed_seconds": round(elapsed, 3)}


@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest, x_api_key: str | None = Header(default=None)):
    """流式问答（SSE）。"""
    check_key(x_api_key)

    async def gen():
        async for chunk in llm.astream(req.message):
            if chunk.content:
                yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/health")
def health():
    """健康检查，供负载均衡/K8s 探活，无需鉴权。"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
