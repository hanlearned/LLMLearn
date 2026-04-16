# learn_llm —— LangChain & LLM 应用开发系统学习路线

> 本仓库用于系统学习 LangChain 框架与 LLM 应用开发。所有学习过程中产生的代码、笔记、示例和项目都将存放于此。

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

# 激活虚拟环境（Windows PowerShell）
.\venv\Scripts\Activate.ps1

# 核心依赖列表
# langchain, langchain-openai, langchain-community, openai, python-dotenv
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=sk-xxxxxxxx
```

> 注册地址：[https://platform.deepseek.com/](https://platform.deepseek.com/)，新用户送 10 元额度，足够学完整个课程。

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
