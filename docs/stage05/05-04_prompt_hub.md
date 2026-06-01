# 05-04 Prompt 版本管理：用 LangChain Hub 复用与版本化提示词

> 🎯 **一句话**：把 Prompt 当代码一样版本化——用 LangChain Hub（`hub.pull`/`hub.push`）集中托管、版本化、团队共享提示词，避免「prompt 散落在各文件里、改一处漏一处、无法回滚」。

---

## 为什么需要它

随着应用变大，Prompt 会越来越多、越来越长，且常被复制粘贴到多个文件。问题随之而来：改了一版效果变差想回滚却没有历史；产品同学想调措辞却要找工程师改代码；同一个 RAG prompt 在三个地方各有一份、彼此漂移。

**Prompt 是模型行为的「源代码」，理应像代码一样被版本管理。** LangChain Hub 提供一个集中仓库：上传 prompt 拿到带版本号的引用，任何代码用 `hub.pull("name")` 拉取，团队共享同一份事实来源（single source of truth）。

---

## 核心用法

### 1. 从 Hub 拉取现成的优质 Prompt

```python
from dotenv import load_dotenv
load_dotenv()                          # 需在 .env 配置 LANGCHAIN_API_KEY

from langchain import hub
from common.llm_provider import get_llm

# 拉取社区维护的经典 ReAct / RAG 提示词，直接用
prompt = hub.pull("rlm/rag-prompt")    # 著名的 RAG 提示模板
print(prompt.messages)                  # 看它长什么样

chain = prompt | get_llm(temperature=0)
print(chain.invoke({
    "context": "LangChain 是一个 LLM 应用开发框架。",
    "question": "LangChain 是什么？",
}).content)
```

**本质在干什么？** `hub.pull("rlm/rag-prompt")` 从公共 Hub 下载一个**已被验证好用**的 PromptTemplate 对象，直接接进 LCEL 链。你不必从零写 RAG 提示，站在社区肩膀上。仓库名格式是 `owner/prompt-name`。

### 2. 把自己的 Prompt 推到 Hub 并版本化

```python
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

my_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是电商客服，语气友好，回答控制在 50 字内。"),
    ("human", "{question}"),
])

# push 后得到一个带 commit hash 的版本；同名再 push 即新增一版（旧版仍可访问）
url = hub.push("your-handle/ecommerce-cs", my_prompt)
print("已发布：", url)
```

**本质在干什么？** `hub.push` 把本地 Prompt 对象上传，**每次 push 生成一个不可变的版本（commit hash）**。这意味着历史永远可追溯、可回滚——和 git commit 一个道理。

### 3. 锁定特定版本拉取（生产环境必做）

```python
# 不指定版本 → 永远拉最新，可能被他人改动「悄悄影响」线上
prompt_latest = hub.pull("your-handle/ecommerce-cs")

# 指定 commit hash → 锁死某一版，线上行为可复现、不被意外变更影响
prompt_pinned = hub.pull("your-handle/ecommerce-cs:a1b2c3d")
```

**本质在干什么？** 拉取时在名字后加 `:版本hash` 可**锁定版本**。生产环境应锁版本，避免别人 push 新版后线上行为无声改变；开发/实验环境可拉最新快速迭代。这正是「版本化」的核心价值。

---

## 关键原理 / 实践要点

1. **Prompt 即代码**：纳入版本管理后，才能回滚、对比、审计「上线哪一版、谁改的、改了什么」。Hub 给每次 push 打不可变 hash，等价于 git commit。
2. **关注点分离**：产品/运营可在 Hub 网页端直接调 prompt 措辞，无需改代码、无需发版；工程代码只 `hub.pull` 引用。Prompt 与应用逻辑解耦。
3. **复用优于重写**：常见 RAG/ReAct/总结类 prompt 社区已有沉淀，先 `hub.pull` 找现成的，再按需 fork 修改。
4. **生产锁版本**：线上一定用 `name:hash` 锁定；只有开发环境才拉 latest。否则一次他人 push 就可能引发线上事故。
5. **不止 Hub**：若不想依赖外部服务，也可把 prompt 存成项目内的 `.yaml`/`.json`，用 git 管理 + `load_prompt` 加载——同样实现版本化，只是少了团队协作的网页界面。

---

## 你来改

- [ ] `hub.pull` 拉取 `hwchase17/openai-functions-agent`，打印它的结构，理解官方 Agent 提示词长什么样。
- [ ] 把你 Stage 3 写的某个工具调用系统提示 `hub.push` 到自己的 handle 下，改一版再 push，确认能拉到两个版本。
- [ ] 用 `:hash` 锁定旧版拉取，验证它不受新 push 影响。

---

## 面试怎么考

**Q：为什么要做 Prompt 版本管理？LangChain Hub 怎么用？**
A：Prompt 是决定模型行为的「源代码」，需要像代码一样可版本化、可回滚、可审计、可共享。LangChain Hub 提供集中仓库：`hub.push(name, prompt)` 上传并生成不可变版本 hash，`hub.pull(name)` 或 `hub.pull(name:hash)` 拉取，团队共享同一事实来源。

**Q：生产环境拉取 Prompt 要注意什么？**
A：必须锁定版本（`name:commit_hash`），否则别人 push 新版会让线上行为无声变化。开发环境才拉 latest 快速迭代。这正是版本化的核心价值——可复现、可控变更。

**Q：Prompt 版本化带来的工程价值是什么？**
A：关注点分离（产品/运营在网页端改措辞、无需发版）、可回滚（效果变差能退回旧版）、可复用（社区/团队沉淀的优质模板直接 pull）、可审计（每版有 hash 和作者）。不依赖外部服务时也可用项目内 YAML + git 实现。
