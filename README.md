# learn_llm —— LangChain & LLM 应用开发系统学习路线

> 本仓库用于系统学习 LangChain 框架与 LLM 应用开发，**目标是达到 Agent 开发工程师水平**。所有学习过程中产生的代码、笔记、示例和项目都将存放于此。

> 🎯 **先读这个**：[总纲 · 从 0 到 Agent 开发工程师](docs/roadmap.md) —— 讲清为什么这样学、学完能干什么、怎么验证自己到了（含能力自检清单）。

---

## 📚 在线文档

- **GitBook 风格文档站点**：https://hanlearned.github.io/LLMLearn
- **源码仓库**：https://github.com/hanlearned/LLMLearn

文档采用技术粒度拆分，每篇文章只讲一个具体 API / 类 / 方法，并附带一句话功能描述，方便快速查阅与复习。

---

## 🗺️ 学习路线概览（6 Stage + 8 项目）

| Stage | 主题 | 核心内容 | 项目 |
| :--- | :--- | :--- | :--- |
| **Stage 1** | LangChain 核心基础 | ChatOpenAI、ChatPromptTemplate、Runnable、输出解析器、LangSmith | 项目 1：结构化简历解析器 |
| **Stage 2** | RAG 系统深度开发 | Document Loaders、Embedding、VectorStore、检索策略、Re-rank | 项目 2：企业级智能知识库问答系统<br>项目 3：GraphRAG 原型系统 |
| **Stage 3** | Agent 智能体开发 | @tool、Tool Calling、AgentExecutor、Memory | 项目 4：智能数据分析助手 |
| **Stage 4** | LangGraph 多 Agent 工作流 | StateGraph、MemorySaver、Human-in-the-loop、Supervisor | 项目 5：多 Agent 内容创作工作流<br>项目 6：AI 招聘助手（MCP + RAG + Agent） |
| **Stage 5** | Prompt 工程与 LLMOps | CoT、ToT、Prompt Hub、LLM-as-a-Judge | 项目 7：Prompt 评估与 A/B 测试平台 |
| **Stage 6** | 工程化部署与性能优化 | FastAPI、LCEL、StreamingResponse、Docker、vLLM / Ollama | 项目 8：LangServe 企业 API 服务平台 |
| **🎓 Capstone** | 毕业综合项目 | RAG + 工具 + 多 Agent + 记忆 + 部署 一体化 | 企业智能客服 Agent（对标岗位要求） |

> **本路线相比通用教程的三处刻意取舍**：① Agent 一律用 **LangGraph** 实现，`AgentExecutor` 只讲原理不依赖（它已软弃用）；② **评测/可观测** 从 Stage 2 就渗透，不是放到最后；③ **工程化（FastAPI）** 从项目 2 起每个项目都带一层，而非攒到结尾。详见 [总纲](docs/roadmap.md)。

### 🔥 进阶实战项目（贴近真实岗位，`projects_advanced/`）

在 8 个阶段项目之外，额外提供 3 个更硬核、更接近真实工作的项目：

| 项目 | 能力点 | 代码 |
| :--- | :--- | :--- |
| NL2SQL 数据问答 Agent | 接真实 SQLite 库 + 只读护栏 + SQL 错误自愈 | `projects_advanced/nl2sql_agent/` |
| Agent 自动化评测系统 | LLMOps：多维 LLM-Judge + 报告生成 | `projects_advanced/agent_eval_system/` |
| 深度研究 Agent | 规划→联网检索→带引用综述（LangGraph 显式编排） | `projects_advanced/research_agent/` |

### 🎓 全栈毕业大项目（`capstone/`）

一个前后端齐全、可一键部署的**智能客服系统**：Supervisor 多 Agent 路由 + RAG 政策问答 + 订单工具 + **退款人工审批（HITL）** + SqliteSaver 持久记忆 + 原生 JS 聊天前端 + 评测看板 + Docker/compose。运行 `python capstone/backend/api.py` 后打开 `http://127.0.0.1:8000/`。

---

## 📁 项目目录结构

```
learn_llm/
├── docs/                    # Docsify 在线文档站点源码
│   ├── index.html
│   ├── _sidebar.md
│   ├── SUMMARY.md
│   ├── README.md
│   └── stage01~stage06/     # 各阶段技术文章（Markdown）
├── stage01_basics/          # Stage 1：核心基础代码
├── stage02_rag/             # Stage 2：RAG 系统代码
├── stage03_agent/           # Stage 3：Agent 智能体代码
├── stage04_langgraph/       # Stage 4：LangGraph 工作流代码
├── stage05_promptops/       # Stage 5：Prompt 工程与 LLMOps 代码
├── stage06_deployment/      # Stage 6：工程化部署代码
├── notes/                   # 本地学习笔记备份
├── venv/                    # Python 虚拟环境（已加入 .gitignore）
├── .env                     # API Key 配置文件（已加入 .gitignore，需手动创建）
├── AGENT.md                 # 状态同步与下一步行动指引
└── README.md                # 本文件
```

---

## 🚀 快速开始

### 1. 环境准备

确保已创建虚拟环境并安装依赖：

```bash
# 进入项目目录
cd learn_llm

# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate          # Linux/macOS
# .\venv\Scripts\Activate.ps1     # Windows PowerShell

# 一键安装全阶段依赖
pip install -r requirements.txt
```

> 依赖按 Stage 分组写在 `requirements.txt` 中，前期阶段不装后期重依赖也能跑。

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入**任意一个**厂商的 Key 即可（代码会自动探测）：

```env
# 四选一即可（推荐 SiliconFlow，注册送额度且支持 Embedding/Reranker）
SILICONFLOW_API_KEY=sk-xxxxxxxx
# DEEPSEEK_API_KEY=sk-xxxxxxxx
# MOONSHOT_API_KEY=sk-xxxxxxxx
# OPENAI_API_KEY=sk-xxxxxxxx
```

> 推荐 [SiliconFlow 硅基流动](https://cloud.siliconflow.cn/)：注册送额度，同时提供对话模型与 `BGE` Embedding/Reranker，RAG 阶段不用在本地下几个 G 的模型。DeepSeek 亦可（[注册地址](https://platform.deepseek.com/)，新用户送额度），但 DeepSeek 无 Embedding 接口，RAG 阶段需另配。

### 3. 运行第一个示例

```bash
python stage01_basics/01_hello_langchain.py
```

---

## 🛠️ 技术栈

- **LangChain** / **LangChain-OpenAI** / **LangChain-Community**
- **OpenAI API** / **DeepSeek API**
- **Python 3.12+**
- **Docsify**（文档站点）
- **FastAPI**、**Docker**、**Redis**（后续部署阶段）

---

## 📝 学习模式约定

1. **代码先行**：每个知识点必须产出可运行的代码。
2. **我来讲**：提供最小可运行示例 + 逐行讲解。
3. **你来改**：基于示例完成扩展任务，遇到报错一起 Debug。
4. **在线同步**：新文章写入 `docs/` 后，`git push` 即可自动更新站点。
5. **面试导向**：每个技术点要讲清楚"为什么这样设计"和"招聘 JD 怎么考"。

---

## 🔗 相关链接

| 资源 | 地址 |
| :--- | :--- |
| 在线文档站点 | https://hanlearned.github.io/LLMLearn |
| 源码仓库 | https://github.com/hanlearned/LLMLearn |
| DeepSeek 开放平台 | https://platform.deepseek.com/ |

---

Happy Learning! 🎉
