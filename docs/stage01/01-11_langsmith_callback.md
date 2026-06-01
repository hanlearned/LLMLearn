# LangSmith：追踪调用链路（可观测性入门）

LangSmith 是 LangChain 官方的「可观测性平台」：自动记录每一次链/Agent 的输入、输出、每一步耗时和 token 消耗，让你能**看见**模型到底干了什么。

## 为什么需要它

LLM 应用是「黑盒」：一条链跑出来结果不对，你很难知道是 prompt 拼错了、检索召回差、还是模型本身的问题。LangSmith 把每一步摊开成一条可视化的 trace，**调试 RAG 和 Agent 时几乎是刚需**——尤其 Agent 多步调用工具时，没有 trace 基本没法排查。

## 核心用法

LangSmith 最大的优点：**几乎零侵入**。配好环境变量，已有代码一行不改就自动上报。

```bash
# .env 里加上（注册 https://smith.langchain.com 获取 key）
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls-xxxxxx
LANGSMITH_PROJECT=llmlearn
```

```python
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from common.llm_provider import get_llm

chain = ChatPromptTemplate.from_template("解释 {x}") | get_llm()
chain.invoke({"x": "向量检索"})
# 跑完去 smith.langchain.com 就能看到这次调用的完整 trace
```

**本质**：LangChain 内部通过 **Callback 机制**在每个组件的「开始/结束/出错」时触发钩子，LangSmith 就是一个监听这些事件并上报的 callback handler。环境变量打开后框架自动挂上它。

## 关键原理
- Callback 是 LangChain 的事件总线：`on_llm_start`、`on_chain_end`、`on_tool_error`… 你也能自定义 handler 做日志、计费、告警。
- trace 是树状的：一条 Agent 调用下面挂着多次工具调用、每次工具里又可能挂 LLM 调用，层级一目了然。

## 你来改
1. 配好 key 后跑 Stage 3 的 Agent，去 LangSmith 看它调了几次工具、每步耗时。
2. 写一个自定义 `BaseCallbackHandler`，在 `on_llm_end` 里打印 token 用量。

## 面试怎么考
- **「Agent 答错了你怎么排查？」** → 看 trace（LangSmith / 打印 intermediate steps），定位是 prompt、检索、工具选择还是模型的问题。能讲「可观测性」是工程成熟度的体现。
- **「LangSmith 怎么接入？侵入大吗？」** → 配 4 个环境变量即可，零代码改动；底层是 Callback 机制自动上报。
