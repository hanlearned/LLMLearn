# 项目 8：企业级 LLM API 服务平台

> 把前面学的一切包成一个「能上线」的服务。从「能跑的脚本」到「能上线的服务」，差的就是这一层工程化。做完它，你就能独立交付一个生产可用的 LLM 后端。
>
> 代码：`stage06_deployment/project08_enterprise_api/`

---

## 一、需求与方案设计

### 从脚本到服务，缺的四样东西
一个能上线的 LLM 服务，至少要有这四个基本要素：

| 要素 | 解决什么 | 本项目实现 |
|------|---------|-----------|
| **鉴权** | 不能谁都能调、要能区分调用方 | `X-API-Key` 请求头校验 |
| **缓存** | 相同请求别重复花钱调模型 | `set_llm_cache(InMemoryCache())` |
| **流式** | 长回答别让用户干等 | SSE `/v1/chat/stream` |
| **健康检查** | 负载均衡/K8s 要能探活 | `/health`（免鉴权） |

> 关于 LangServe：早期常用 LangServe 一键暴露 Chain，但它维护趋缓，社区已普遍回归 **FastAPI + LangChain 手写接口**（更灵活、可控）。本项目走这条主流路线，所以叫"LangServe 平台"但实现是 FastAPI。

---

## 二、实现详解

### 难点 1：LLM 缓存怎么省钱
```python
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
set_llm_cache(InMemoryCache())
```
设置一次全局缓存后，**完全相同**的请求（同模型、同参数、同输入）第二次直接命中缓存、不再调 API。接口里返回了 `elapsed_seconds`，连问两次同样的问题，第二次耗时会从几秒掉到接近 0。生产中把 `InMemoryCache` 换成 `RedisCache` 即可多实例共享缓存。

### 难点 2：流式接口的 SSE 格式
```python
async def gen():
    async for chunk in llm.astream(req.message):
        if chunk.content:
            yield f"data: {chunk.content}\n\n"   # SSE 格式：data: 开头，空行结尾
    yield "data: [DONE]\n\n"
return StreamingResponse(gen(), media_type="text/event-stream")
```
`media_type="text/event-stream"` 是关键，前端/浏览器据此按 SSE 解析。`[DONE]` 是和前端约定的结束标记。

### 难点 3：容器化的层缓存技巧
Dockerfile 里**先拷 requirements 装依赖、再拷代码**：

```dockerfile
COPY .../requirements.txt .
RUN pip install -r requirements.txt   # 这层只在依赖变化时才重建
COPY common/ ./common/
COPY stage06_deployment/project08_enterprise_api/ ./.../
```

改代码时依赖层命中缓存，构建从几分钟降到几秒。另外 **API Key 通过 `-e` 环境变量注入**，绝不写进镜像——这是安全红线。

---

## 三、运行

```bash
# 本地直接跑
pip install -r stage06_deployment/project08_enterprise_api/requirements.txt
python stage06_deployment/project08_enterprise_api/main.py
# 调试：http://127.0.0.1:8000/docs ，请求头加 X-API-Key: demo-key-123

# 容器化（在仓库根目录）
docker build -t yunqi-llm-api -f stage06_deployment/project08_enterprise_api/Dockerfile .
docker run -p 8000:8000 -e SILICONFLOW_API_KEY=sk-xxx yunqi-llm-api
```

测试流式：
```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/stream \
  -H "Content-Type: application/json" -H "X-API-Key: demo-key-123" \
  -d '{"message":"用三句话介绍 LangGraph"}'
```

---

## 四、复盘与进阶
1. **限流 & 配额**：按 API Key 做每秒/每月限流（Redis 计数）。
2. **Redis 缓存**：换成 `RedisCache`，多实例共享、重启不丢。
3. **可观测**：接入 LangSmith 或 OpenTelemetry，记录每次调用的延迟、token、成本。
4. **接 RAG/Agent**：把 `/v1/chat` 内部换成项目 2 的 RAG 链或项目 4 的 Agent，就是一个完整产品后端。

## 五、面试怎么考
- **「LLM 服务怎么降本？」** → 缓存（相同请求不重复调）、小模型路由（简单问题用便宜模型）、prompt 精简、流式提升体验侧感知。
- **「为什么要流式？怎么实现？」** → 降低首字延迟、改善体验。服务端 `astream` + SSE（text/event-stream）逐块推送。
- **「Dockerfile 怎么优化构建速度/安全？」** → 先装依赖再拷代码利用层缓存；用 slim 基础镜像；密钥用环境变量注入不进镜像；`.dockerignore` 排除 venv/.env。
- **「LangServe 还是 FastAPI？」** → 现在主流是 FastAPI 手写，灵活可控；能说出 LangServe 维护趋缓是加分。
