# Agent/RAG 自动化评测系统（LLMOps 工具）

一个**通用的、可复用的** LLM 应用评测框架。核心思想：把「被测系统」抽象成一个
`(question) -> answer` 的函数，配上一份评测集，就能用 **LLM-as-a-Judge** 对答案做
**多维度打分**（正确性 / 忠实度 / 相关性，各 1-5 分，且每个分数都附理由），
最后聚合出平均分与通过率，并产出 **Markdown 报告 + JSON 结果**。

> 适用场景：你改了 RAG 的切分策略 / 换了模型 / 调了 Prompt，想用「数字」而不是
> 「肉眼感觉」回答——这次改动到底变好还是变差？

## 目录结构

```
agent_eval_system/
├── eval_set.json   # 评测集：10 条 {question, reference, context?}
├── judge.py        # 裁判：Pydantic + with_structured_output 多维度评分
├── harness.py      # 主程序：跑被测系统 -> 打分 -> 聚合 -> 生成报告
├── report.md       # 运行后生成：带逐条明细表格 + 汇总
└── report.json     # 运行后生成：完整结构化结果
```

## 运行步骤

1. 在仓库根目录的 `.env` 填入任一厂商的 API Key（见根目录 `.env.example`）。
2. 安装依赖（仓库根 `requirements.txt` 已含 langchain / langchain-openai / pydantic 等）。
3. 在**仓库根目录**执行：

```bash
# 只跑裁判自检（验证结构化输出可用）
python projects_advanced/agent_eval_system/judge.py

# 跑完整评测，生成 report.md / report.json
python projects_advanced/agent_eval_system/harness.py
```

运行结束后终端会打印汇总，并在本目录写出 `report.md`、`report.json`。

## 评分维度与通过判定

| 维度 | 含义 |
| --- | --- |
| correctness 正确性 | 答案与参考答案在事实上是否一致 |
| faithfulness 忠实度 | 答案是否有据可依、未编造（幻觉） |
| relevance 相关性 | 答案是否切题、聚焦问题本身 |

- 每个维度 1-5 的整数分，并由裁判附中文理由。
- 单维度 `>= 4`（`harness.py` 中 `PASS_THRESHOLD`）算该维度通过；
  **三维全部通过**才算该样本整体通过。通过率 = 整体通过样本数 / 总样本数。

## 如何把 target_system 换成你自己的 Agent / RAG

`harness.py` 与被测系统之间的**唯一契约**就是这个函数签名：

```python
def target_system(question: str) -> str:
    ...
```

只要保持签名不变，把函数体换成你的系统调用即可，harness 其余部分无需改动：

```python
# 例：换成你的 RAG 链
from your_project.rag import rag_chain

def target_system(question: str) -> str:
    return rag_chain.invoke({"question": question})

# 例：换成你的 LangGraph Agent
from your_project.agent import agent

def target_system(question: str) -> str:
    state = agent.invoke({"messages": [("user", question)]})
    return state["messages"][-1].content
```

如果你的系统是 RAG，建议在 `eval_set.json` 的每条样本里填上 `context`
（该问题应当被召回的关键事实），后续可据此扩展「检索命中率」等检索侧指标。

## 设计要点速记

- **多维度评分**：拆成正交维度，定位问题更精细，比单一「对/错」可解释。
- **打分附理由**：让裁判「先写理由再给分」，分数更稳、可审计。
- **结构化输出稳健**：用 `with_structured_output(Pydantic)` 走 tool calling，
  不依赖正则解析裸文本 JSON。
- **可复现**：裁判与 demo 被测系统都用 `temperature=0`。
- **鲁棒**：被测系统单条异常、裁判偶发空返回都有兜底，不中断整批评测。
