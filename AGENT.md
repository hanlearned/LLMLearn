# learn_llm —— LangChain & LLM 应用开发系统学习路线

> 本目录用于跟随 Kimi 学习 LangChain 框架。所有学习过程中产生的代码、笔记、示例和项目都将存放于此。

---

## 📌 当前状态（最后一次更新：2026-04-16）

### 已完成
- [x] 制定基于 2025 年招聘市场（Boss/猎聘）的 **6 Stage + 8 项目** 系统学习路线
- [x] 创建 Python 虚拟环境 `venv/`，安装核心依赖：`langchain`, `langchain-openai`, `langchain-community`, `openai`, `python-dotenv`
- [x] 搭建 **GitHub Pages 在线文档站点**（Docsify + GitBook 风格主题）
  - 站点地址：https://hanlearned.github.io/LLMLearn
  - 仓库地址：https://github.com/hanlearned/LLMLearn
  - 目录采用 **技术粒度拆分**：每篇只讲一个具体 API/类/方法，并附带一句话功能描述
- [x] 写好前两篇技术文档并上线：
  - `01-01_chatopenai.md`：配置与调用大模型
  - `01-02_chatprompttemplate.md`：消息模板与变量注入
- [x] 写好第一个可运行程序：`stage01_basics/01_hello_langchain.py`
- [x] **DeepSeek API 已配置并跑通**，`01_hello_langchain.py` 成功调用模型并返回结果
- [x] 完成 **LLM 单例模式封装**：`stage01_basics/common/llm_provider.py`
  - 用 `__new__` 实现全局唯一实例，后续所有程序统一通过 `get_llm()` 获取模型
- [x] 完成 `PydanticOutputParser` 最小示例：`stage01_basics/02_pydantic_output_parser.py`
  - 强制 AI 按 JSON Schema 输出结构化数据，并自动校验字段类型
- [x] 完成 `MessagesPlaceholder` 最小示例：`stage01_basics/04_messagesplaceholder.py`
  - 动态消息列表占位，Agent/Memory 前置知识
- [x] 新增技术文档：`docs/stage01/01-03_messagesplaceholder.md`
- [x] 完成 `RunnableParallel` 最小示例：`stage01_basics/05_runnable_parallel.py`
  - 一份输入同时分发给多个 Runnable，掌握串并联组合
- [x] 新增技术文档：`docs/stage01/01-06_runnable_parallel.md`
- [x] 新增技术文档：`docs/stage01/01-07_runnablesequence.md`
  - 讲清楚 `prompt | llm | parser` 的管道语法本质、执行顺序与内部实现
- [x] 完善 `01-02_chatprompttemplate.md`
  - 为每个代码示例补充"本质在干什么"的逐行解释
- [x] 编写 `README.md`，汇总项目介绍、在线文档地址、快速开始指南
- [x] 完成 **项目 1：结构化简历解析器**：`stage01_basics/project01_resume_parser.py`
  - 用嵌套 Pydantic Schema 从非结构化简历文本中提取结构化数据
- [x] 新增 `docs/stage01/01-10_pydantic_output_parser.md`
  - 含 Markdown 代码块清洗能力溯源（`parse_json_markdown`，LangChain 0.0.x 起即存在）
- [x] 新增 `docs/stage01/01-05_runnable_stream.md` + `03_runnable_stream.py`
  - 流式输出原理、`stream=true` 底层机制、Chunk 形态
- [x] 新增 `docs/stage01/01-04_runnable_invoke.md`
  - `invoke()` 本质、`config` 参数、异常处理、`invoke` vs `batch` vs `stream`

### 当前卡点
无。API 已通，基础组件（Prompt → LLM → Parser 链式组合）已掌握，项目 1 已跑通。

---

## 🆙 路线升级（2026-06-01）

为达成「Agent 开发工程师水平」的目标，对课程做了一次结构性升级：
- [x] 新增 **总纲文档** `docs/roadmap.md`：岗位能力画像、设计取舍、**能力自检清单**
- [x] 升级基建：根级 `common/` 共享包（多厂商 LLM 自动探测 + Embedding 选型）、`requirements.txt`
- [x] **Agent 全部改用 LangGraph 实现**，`AgentExecutor` 仅讲原理（已软弃用）
- [x] **评测/可观测前置**：Stage 2 新增 `02-12_rag_evaluation.md`，贯穿全程
- [x] 新增 **Capstone 毕业项目**：企业智能客服 Agent（融合全部六大能力）
- [x] 全面铺开 Stage 2–6 文档、可运行代码与 8+1 个实战项目

## 🆙 进阶加厚（2026-06-01 第二轮）

- [x] 新增 3 个**进阶实战项目**（`projects_advanced/`）：NL2SQL 数据问答 Agent（接真实 SQLite + 只读护栏 + 错误自愈）、Agent 自动化评测系统（多维 LLM-Judge + 报告）、深度研究 Agent（规划→联网检索→带引用综述）
- [x] **Capstone 升级为 v2 全栈版**（`capstone/`）：backend(多 Agent 图 + 退款 HITL + SqliteSaver 持久化) + frontend(原生 JS 聊天界面 + 审批按钮) + eval(评测看板) + Dockerfile/docker-compose
- [x] 导航(_sidebar/SUMMARY)、README、roadmap 同步；全量 py_compile 通过、零死链

## 🚀 下一步行动（Next Step）

- 按 `docs/roadmap.md` 的能力自检清单逐条打勾
- 入门：`stage02_rag/01_minimal_rag.py`；进阶：`projects_advanced/nl2sql_agent/`
- 毕业大项目：`python capstone/backend/api.py` → 浏览器 `http://127.0.0.1:8000/`

---

## 🗺️ 完整学习路线（供快速回顾）

### Stage 1：LangChain 核心基础
- `ChatOpenAI`：配置与调用大模型
- `ChatPromptTemplate`：消息模板与变量注入
- `MessagesPlaceholder`：动态消息列表占位
- `Runnable.invoke()`：同步调用与异常处理
- `Runnable.stream()`：流式输出与 SSE 基础
- `RunnableParallel`：并行执行多个子任务
- `RunnableSequence` 与 `|`：链式组合原理
- `StrOutputParser`：提取纯文本输出
- `JsonOutputParser`：强制 JSON 格式输出
- `PydanticOutputParser`：结构化对象输出与 Schema 校验
- `LangSmithCallbackHandler`：追踪调用链路
- **项目 1：结构化简历解析器**

### Stage 2：RAG 系统深度开发
- Document Loaders、RecursiveCharacterTextSplitter、Embedding 模型
- Chroma / FAISS / Milvus、检索策略、VectorStoreRetriever
- create_retrieval_chain、MultiQueryRetriever、Re-rank
- **项目 2：企业级智能知识库问答系统**
- **项目 3：GraphRAG 原型系统**

### Stage 3：Agent 智能体开发
- `@tool`、StructuredTool、Tool Calling 机制
- AgentExecutor、create_react_agent、create_openai_functions_agent
- ConversationBufferMemory / SummaryMemory
- **项目 4：智能数据分析助手**

### Stage 4：LangGraph 多 Agent 工作流
- StateGraph、Nodes & Edges、add_conditional_edges
- MemorySaver、Human-in-the-loop、Supervisor
- **项目 5：多 Agent 内容创作工作流**
- **项目 6：AI 招聘助手（MCP + RAG + Agent）**

### Stage 5：Prompt 工程与 LLMOps
- CoT、ToT、Self-Consistency、Prompt Hub、LLM-as-a-Judge
- **项目 7：Prompt 评估与 A/B 测试平台**

### Stage 6：工程化部署与性能优化
- FastAPI 集成、LCEL 高级模式、StreamingResponse & SSE
- GPTCache / Redis、Docker、vLLM / Ollama
- **项目 8：LangServe 企业 API 服务平台**

---

## 📁 项目目录结构

```
learn_llm/
├── docs/                         # Docsify 在线文档站点
│   ├── index.html
│   ├── _sidebar.md
│   ├── SUMMARY.md
│   ├── README.md
│   └── stage01~stage06/          # Markdown 技术文章
├── stage01_basics/               # Stage 1 代码
│   ├── common/                   # 公共工具（LLM 单例封装等）
│   ├── 01_hello_langchain.py
│   └── 02_pydantic_output_parser.py
├── stage02_rag/                  # Stage 2 代码
├── stage03_agent/                # Stage 3 代码
├── stage04_langgraph/            # Stage 4 代码
├── stage05_promptops/            # Stage 5 代码
├── stage06_deployment/           # Stage 6 代码
├── notes/                        # 本地笔记备份
├── venv/                         # Python 虚拟环境（已加入 .gitignore）
├── .env                          # API Key（已加入 .gitignore，需手动创建）
├── README.md                     # 项目总览与快速开始
└── AGENT.md                      # 本文件：状态同步与下一步指引
```

---

## 📝 学习模式约定

1. **代码先行**：每个知识点必须产出可运行的代码
2. **我来讲**：我给最小可运行示例 + 逐行讲解
3. **你来改**：基于示例完成扩展任务，遇到报错一起 Debug
4. **在线同步**：新文章写入 `docs/` 后，用户负责 `git push`，站点自动更新
5. **面试导向**：每个技术点要讲清楚"为什么这样设计"和"招聘 JD 怎么考"
