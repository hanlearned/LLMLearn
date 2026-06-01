# 🎓 Capstone：全栈企业智能客服 Agent（v2）

课程毕业大项目。一个**前后端齐全、可一键部署**的智能客服系统，融合全部六大能力：

- **Supervisor 多 Agent**：主管节点路由意图 → 政策/订单/退款/闲聊 专家节点
- **RAG**：政策问答从知识库检索作答（有据可查）
- **工具调用**：订单查询走业务工具
- **Human-in-the-Loop**：退款是高风险动作，执行前中断、等人工在前端点「批准/拒绝」
- **持久化记忆**：SqliteSaver 按 session 存对话，重启不丢
- **全栈 + 部署**：FastAPI 后端 + 原生 JS 聊天前端 + 评测看板 + Docker / docker-compose

📖 完整方案/架构/实现/面试讲法：
- `docs/capstone/capstone_overview.md`（方案与架构）
- `docs/capstone/capstone_implementation.md`（实现详解）

## 目录结构
```
capstone/
├── backend/          后端
│   ├── config.py     路径与参数
│   ├── kb.py         政策知识库 RAG 检索器
│   ├── tools.py      订单/退款业务数据
│   ├── graph.py      ⭐多 Agent 编排图（triage 路由 + 退款 HITL + 持久化）
│   └── api.py        FastAPI（/api/chat、/api/approve、托管前端）
├── frontend/
│   └── index.html    原生 JS 聊天界面（含审批按钮）
├── eval/
│   ├── eval_set.json 评测集
│   └── run_eval.py   评测看板（跑图 + LLM 打分 + 生成报告）
├── data/cs_policy.md 政策知识库
├── Dockerfile · docker-compose.yml · requirements.txt
```

## 运行
```bash
pip install -r capstone/requirements.txt    # 仓库根目录，已配 .env

# 方式一：本地起服务（含前端）
python capstone/backend/api.py
# 浏览器打开 http://127.0.0.1:8000/  —— 试试「订单 10001 想退款」体验人工审批

# 方式二：Docker 一键
docker compose -f capstone/docker-compose.yml up --build

# 评测看板
python capstone/eval/run_eval.py
```

演示订单号：10001 / 10002 / 10003。
