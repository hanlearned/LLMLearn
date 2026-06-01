# 项目 2：企业级智能知识库问答系统

RAG 在真实业务里的完整落地：把企业文档变成「问它就答、答得有据」的机器人。

📖 **完整方案 / 实现详解 / 复盘 / 面试问法**见文档站：`docs/stage02/project02_knowledge_base.md`

## 快速开始

```bash
pip install -r requirements.txt        # 在仓库根目录
# 在仓库根目录建好 .env（填一个 API Key）

# 1. 建库（仅文档更新时重跑）
python stage02_rag/project02_knowledge_base/ingest.py

# 2a. 命令行问答
python stage02_rag/project02_knowledge_base/qa.py

# 2b. 或起 API：http://127.0.0.1:8000/docs
python stage02_rag/project02_knowledge_base/api.py
```

## 文件说明

| 文件 | 职责 |
|------|------|
| `config.py` | 路径与超参集中管理 |
| `ingest.py` | 离线建库（加载→切分→向量化→持久化） |
| `qa.py` | 问答业务逻辑（CLI + 被 API 复用） |
| `api.py` | FastAPI HTTP 服务 |

知识源在 `../data/`，换成你自己的文档重跑 `ingest.py` 即可。
