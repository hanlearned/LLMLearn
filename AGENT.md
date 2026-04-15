# learn_llm —— LangChain & LLM 应用开发系统学习路线

> 本路线基于 2025 年 Boss 直聘、猎聘等招聘平台对「大模型应用工程师 / AI Agent 开发工程师」岗位 JD 的梳理，聚焦市场最核心的两大区分点：**Agent 编排能力** 与 **RAG 系统工程能力**。所有内容以 **LangChain 生态** 为主线，兼顾 LangGraph、LlamaIndex、CrewAI 等主流框架。

---

## 一、市场能力画像（Why）

| 能力维度 | 市场要求 | 出现频率 |
|---------|---------|---------|
| **Agent 框架** | LangChain、LangGraph、LlamaIndex、CrewAI、AutoGen | 70%+ |
| **RAG 全链路** | 文档解析 → Chunking → Embedding → 向量检索 → Re-rank → 生成 | 65%+ |
| **Tool / Function Calling** | 为 Agent 定义、注册、调用外部工具/API | 60%+ |
| **推理模式** | ReAct、Plan-and-Execute、CoT、ToT | 50%+ |
| **Prompt 工程** | 结构化提示词设计、版本管理、效果评估 | 40%+ |
| **工程化** | Docker、RESTful API、向量数据库、LLMOps、vLLM | 75%+ |
| **前沿加分项** | MCP 协议、GraphRAG、多模态 Agent、NL2SQL | 30%+ |

---

## 二、学习阶段总览

### Stage 0：前置准备
- Python 高阶特性（异步、类型注解、装饰器）
- 基础 LLM 原理（Transformer、预训练 vs 微调、Token 机制）
- OpenAI API / 国内大模型 API（DeepSeek、Qwen、GLM）调用方式
- 版本管理：Git、Conda/uv、Poetry

### Stage 1：LangChain 核心基础
**目标**：能用 LangChain 独立完成一次大模型调用、Prompt 管理、输出解析。
- LangChain 架构设计：Components → Chains → Agents
- Prompt Template（StringPromptTemplate、ChatPromptTemplate、FewShotPromptTemplate）
- Model I/O：LLM vs ChatModel、流式输出、回调机制
- Output Parsers：PydanticOutputParser、StructuredOutputParser、JsonOutputParser
- 文档加载器与文本拆分器（Document Loaders & Text Splitters）
- 向量存储与 Embedding（OpenAI、BGE、M3E、Chroma、FAISS）
- **实战项目 1：结构化简历解析器**
  - **解说**：上传一份 PDF 简历，通过 `PyPDFLoader` 提取文本，利用 `PydanticOutputParser` 让 LLM 按定义好的 Schema（姓名、技能、工作经历）输出结构化 JSON。学习点在于掌握 **非结构化数据 → 结构化输出** 的完整链路，这是 RAG 与 Agent 的数据预处理基础。

### Stage 2：RAG 系统深度开发
**目标**：掌握从文档到精准问答的完整 RAG Pipeline，能优化检索准确率与召回率。
- 文档解析生态：Unstructured、Marker、LlamaParse
- Chunking 策略：固定长度、递归、语义切分、Markdown/标题感知切分
- Embedding 模型选型与微调概念（BGE、GTE、M3E、Jina）
- 向量数据库对比与实操：Chroma（本地）、Milvus（企业级）、PGVector（关系型融合）
- 检索策略：相似度检索、MMR（最大边际相关性）、混合检索（BM25 + Vector）
- Re-Rank 与结果精排：Cross-Encoder、Cohere Rerank、BGE-Reranker
- 高级 RAG：Self-RAG、Corrective RAG、RAG-Fusion、Hypothetical Document Embedding
- **实战项目 2：企业级智能知识库问答系统**
  - **解说**：以某科技公司产品手册为数据源，构建一个支持多文档、多格式的知识库系统。流程包括：① 使用 `UnstructuredMarkdownLoader` 和 `PyPDFLoader` 解析文档；② 采用 `RecursiveCharacterTextSplitter` 做语义切分；③ 用 `BAAI/bge-large-zh` 生成 Embedding 并存入 `Milvus`；④ 实现 Hybrid Search（Dense + BM25）；⑤ 引入 `BGE-Reranker` 对 Top-K 结果重排序；⑥ 用 `RetrievalQA` 或 `ConversationalRetrievalChain` 生成带引用的回答。项目重点在于理解 **检索质量决定生成质量**，掌握召回率、准确率、延迟的权衡。
- **实战项目 3：GraphRAG 原型系统**
  - **解说**：在基础 RAG 之上，对技术文档抽取实体关系，构建轻量级知识图谱（使用 `NetworkX` + LLM 实体抽取），实现基于图遍历的增强检索。了解 GraphRAG 与 VectorRAG 的互补场景，这是 2025 年高端岗位的热门加分项。

### Stage 3：Agent 智能体开发
**目标**：深入理解 Agent 核心机制，能设计具备规划、记忆、工具调用能力的智能体。
- Agent 核心概念：Agent = LLM + Prompt + Tools + Memory + Planner
- Tool Calling / Function Calling 机制详解
  - `@tool` 装饰器、StructuredTool、BaseTool
  - OpenAI Function Calling vs LangChain Tools 的映射关系
- Agent 推理模式：
  - **ReAct**：推理（Reasoning）+ 行动（Acting）的循环
  - **Plan-and-Solve / Plan-and-Execute**：先规划后执行
  - **Self-Ask**：分解问题逐步求解
- Memory 设计：ConversationBufferMemory、ConversationSummaryMemory、VectorStoreRetrieverMemory
- Agent 类型：Zero-Shot ReAct、Structured Chat、OpenAI Functions Agent、Self-Ask with Search
- **实战项目 4：智能数据分析助手（Data Analysis Agent）**
  - **解说**：构建一个能回答业务数据问题的 Agent。为其注册以下 Tools：① `SQLDatabaseToolkit`（连接 SQLite/PostgreSQL，执行 NL2SQL）；② `PythonREPLTool`（执行 Pandas 数据分析代码）；③ `DuckDuckGoSearchRun`（补充外部行业数据）。Agent 采用 `create_react_agent` 模式，用户问"上个月销售额 Top5 品类是什么？"时，Agent 先推理应该用哪个 Tool，执行 SQL 查询，再用 Python 绘制柱状图并给出分析结论。学习点在于 **Tool 的边界设计**、**Observation 的传递**、**ReAct Loop 的调试与容错**。

### Stage 4：多 Agent 协作与 LangGraph
**目标**：掌握复杂工作流编排，能构建多 Agent 协同系统。
- LangGraph 核心：StateGraph、Nodes、Edges、Conditional Edges
- 循环与持久化：MemorySaver、Checkpointer、Human-in-the-loop
- 多 Agent 架构：Supervisor、Swarm、Hierarchical
- CrewAI / AutoGen 快速体验与对比
- **实战项目 5：多 Agent 内容创作工作流（LangGraph）**
  - **解说**：模拟一个内容团队，用 LangGraph 编排 3 个 Agent：
    - **Researcher**：负责搜集资料、整理大纲（Tools：Search）
    - **Writer**：负责根据大纲撰写文章（Tools：无，纯生成）
    - **Editor**：负责审校、打分、决定是否重写（Conditional Edge）
  - 状态对象 `State` 在三个节点间流转，Editor 评分低于 80 分则回退到 Writer 重写，高于 80 分则结束。通过本项目掌握 **图状态机**、**条件分支**、**循环反馈** 的编排思想，这是构建复杂企业级 Agent 的核心能力。
- **实战项目 6：AI 招聘助手（MCP + Agent + RAG）**
  - **解说**：模拟 HR 助手，整合三大模块：
    - ① **RAG 模块**：检索公司岗位知识库，回答候选人关于福利、流程的问题；
    - ② **Agent 模块**：调用简历解析 Tool、日历预约 Tool、邮件发送 Tool，完成"帮我安排下周三面试并发送邮件"的指令；
    - ③ **MCP 协议**：使用 MCP Server 暴露简历解析和邮件发送能力，让 Agent 通过 MCP Client 调用。
  - 这是全栈整合项目，覆盖招聘 JD 中最常见的"0-1 项目落地能力"要求。

### Stage 5：Prompt 工程与 LLMOps
**目标**：能进行工业化级别的提示词管理与效果评估。
- 结构化 Prompt 设计：System / User / Assistant 角色分离、XML/JSON 格式约束
- 高阶技巧：Chain of Thought (CoT)、Tree of Thoughts (ToT)、Self-Consistency、Step-Back Prompting
- Prompt 版本管理：LangSmith、PromptLayer、Weights & Biases
- 评估体系：RAGAS（RAG 评估）、LLM-as-a-Judge、人工标注对齐
- Tracing & Observability：LangSmith 追踪 Chain/Agent 执行链路
- **实战项目 7：Prompt 评估与 A/B 测试平台**
  - **解说**：针对同一业务场景（如客服回复），设计 3 版不同 Prompt，用 LangSmith 记录每次调用的输入、输出、延迟、Token 消耗。制定评估指标（相关性、友好度、是否引用知识库），通过 LLM-as-a-Judge 批量打分，最终选出最优 Prompt 版本并固化到配置中心。学习点在于 **数据驱动的 Prompt 优化**。

### Stage 6：工程化部署与性能优化
**目标**：具备将 LLM 应用部署上线的能力。
- API 框架：FastAPI、LangServe、LangChain Expression Language (LCEL)
- 异步与并发：Asyncio、Streaming Response、SSE
- 模型部署：Ollama（本地）、vLLM（高吞吐）、Xinference（统一接口）
- 容器化：Docker、Docker Compose
- 缓存与加速：GPTCache、Redis 缓存 Embedding 结果
- 安全与护栏：输入过滤、输出审核、敏感信息检测
- **实战项目 8：LangServe 企业 API 服务平台**
  - **解说**：将项目 2（知识库问答）和项目 4（数据分析 Agent）封装为 RESTful API，使用 FastAPI + LangServe 部署。实现接口鉴权（JWT）、流式输出（SSE）、请求限流、Docker 容器化打包。配合前端（React/Vue 可选）或 Apifox 进行端到端测试。这是求职时最能体现 **全栈落地能力** 的项目。

---

## 三、我们的学习模式（你 + Kimi）

既然是一起学，那就不是"我给你书单你自己看"，而是**我直接带着你写代码、讲原理、排错**。每个阶段按以下节奏推进：

### 1. 我来讲：先给"最小可运行示例"
- 不先堆概念，而是先给你一段能直接跑通的代码。
- 边跑边讲：这段代码调用了 LangChain 的哪个组件、为什么要这样设计、背后的设计哲学是什么。

### 2. 你来改：基于示例做扩展
- 我会给你一个明确的"小任务"，比如：把 Prompt 从普通字符串改成 ChatPromptTemplate、把输出从纯文本改成结构化 JSON。
- 你动手改，遇到报错直接发给我，我帮你一起 Debug。

### 3. 一起做项目：从零搭建到可演示
- 每个 Stage 末尾的实战项目，**我们一起从零开始写**：
  - 我先给出项目骨架和核心代码；
  - 你填充业务逻辑或做本地适配（比如换成本地大模型 API）；
  - 最后项目必须在你的环境里完整跑通，并形成一份 README 项目解说。

### 4. 代码与笔记管理
- 所有代码统一放在 `D:\kimi_projects\learn_llm\` 下，按 Stage 分子目录：
  ```
  learn_llm/
  ├── stage01_basics/           # Stage 1：基础与简历解析器
  ├── stage02_rag/              # Stage 2：RAG 系统
  ├── stage03_agent/            # Stage 3：Agent 开发
  ├── stage04_langgraph/        # Stage 4：多 Agent 与工作流
  ├── stage05_promptops/        # Stage 5：Prompt 工程与评估
  ├── stage06_deployment/       # Stage 6：工程化部署
  └── notes/                    # 学习笔记、问题记录、面试总结
  ```
- 每完成一个 Stage，在 `notes/` 里写一篇 `stageX_summary.md`，记录：
  - 本阶段核心知识点（3-5 条）
  - 遇到的坑和解决方法
  - 面试时可能会被问到的问题及回答口径

### 5. 答疑与 Checkpoint
- **Checkpoint 机制**：每个小任务完成后，发代码运行截图或结果给我，确认理解无误后再进入下一步。
- **随时提问**：概念不懂、报错不会修、方案想不清楚，随时抛给我。
- **市场结合**：每学一个技术点，我会同步告诉你"招聘 JD 里这个点怎么考的"，让你始终对准求职目标。

---

## 四、学习建议

1. **代码先行**：每个阶段必须产出可运行的代码仓库，拒绝只看不写。
2. **项目驱动**：以 8 个实战项目为里程碑，每个项目配套一篇技术博客或 README 解说。
3. **面试导向**：每个项目要能清晰回答"为什么选这个技术方案"、"遇到了什么坑"、"如何评估效果"。
4. **紧跟前沿**：定期浏览 arXiv、LangChain Blog、Twitter/X 技术圈，关注 MCP、GraphRAG、Multi-Agent 的最新进展。

---

## 五、当前目标

- 系统学习 LangChain 框架的核心概念与用法
- 通过实践掌握 Chain、Agent、Memory、Tools 等模块
- 逐步构建基于 LLM 的应用示例
- **最终目标**：具备独立设计、开发、部署和优化企业级 Agent + RAG 系统的能力，匹配市场上大模型应用开发工程师岗位的核心要求。
