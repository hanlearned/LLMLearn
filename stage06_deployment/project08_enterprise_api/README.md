# 项目 8：企业级 LLM API 服务平台

把所学包成「能上线」的服务：鉴权 + 缓存 + 流式 + 健康检查 + 容器化。

📖 完整方案/实现/复盘/面试：`docs/stage06/project08_enterprise_api.md`

```bash
# 本地
pip install -r stage06_deployment/project08_enterprise_api/requirements.txt
python stage06_deployment/project08_enterprise_api/main.py
# http://127.0.0.1:8000/docs ，请求头 X-API-Key: demo-key-123

# Docker（仓库根目录）
docker build -t yunqi-llm-api -f stage06_deployment/project08_enterprise_api/Dockerfile .
docker run -p 8000:8000 -e SILICONFLOW_API_KEY=sk-xxx yunqi-llm-api
```

接口：`/v1/chat`（带缓存，返回 elapsed_seconds）、`/v1/chat/stream`（SSE 流式）、`/health`（免鉴权探活）。把 `/v1/chat` 内部换成项目 2 的 RAG 链或项目 4 的 Agent，即成完整产品后端。
