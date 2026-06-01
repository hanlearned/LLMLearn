# 高级实战 · Agent/RAG 自动化评测系统（LLMOps 工具）

当你的 LLM 应用从「demo 能跑」走向「线上要交付」，第一个绕不开的工程问题就是：
**改一处，怎么知道是变好还是变坏？** 换模型、调 Prompt、改 RAG 切分、给 Agent 加一个工具——
每一次都是在「优化」，但凭什么说它优化了？靠人工抽几条「肉眼看感觉还行」，不可复现、不可对比、
覆盖不全。本项目就是把这件事工程化：一套**通用评测框架**，给定评测集 + 被测系统，
自动用 LLM-as-a-Judge 多维度打分、聚合、出报告。

对应代码：`projects_advanced/agent_eval_system/`（`eval_set.json` / `judge.py` / `harness.py` / `README.md`）。

## 一、需求与方案设计

### 要解决的问题

- **可量化**：把「答得好不好」从主观感受变成可对比的数字（平均分、通过率）。
- **可解释**：单一「对/错」太粗，定位不了问题。要拆成正交维度，并让裁判给出理由。
- **可复现**：同一份评测集、同一个系统，今天跑和明天跑结果应当一致。
- **可复用**：框架不应绑死某一个系统。无论被测的是裸 LLM、RAG 链还是 LangGraph Agent，
  都能用同一套 harness 评。

### 核心抽象

整个框架只依赖两个输入：

1. **评测集**：一批 `{question, reference, context?}`。`reference` 是参考答案/要点，
   `context` 可选（RAG 场景下填「该问题应被召回的关键事实」）。
2. **被测系统**：一个签名为 `(question: str) -> answer: str` 的函数。

这个极简契约是「通用」的关键——被测系统内部是 Chain 还是 Agent、是不是 RAG，harness 一概不关心。

### 评分维度（三个正交维度，各 1-5 分）

| 维度 | 含义 | 主要抓什么 |
| --- | --- | --- |
| correctness 正确性 | 答案与参考答案在事实上是否一致 | 错误、遗漏关键要点 |
| faithfulness 忠实度 | 答案是否有据可依、未编造 | 幻觉 |
| relevance 相关性 | 答案是否切题、聚焦问题 | 答非所问、跑题 |

判定规则：单维度 `>= 4` 算该维度通过；**三维全过**才算该样本整体通过；
通过率 = 整体通过样本数 / 总样本数。阈值在 `harness.py` 的 `PASS_THRESHOLD` 里可调。

### 数据流

```
eval_set.json
     │  每条样本
     ▼
target_system(question) ──► answer
     │
     ▼
judge_one(question, answer, reference) ──► JudgeScore(三维分数+理由)
     │  收集所有 records
     ▼
aggregate ──► 各维度均值 / 总均值 / 通过率
     │
     ▼
report.md（汇总 + 逐条表格） + report.json（完整结构化结果）
```

## 二、实现详解

### 1. 多维评分的 Schema（judge.py）

评分结果用 Pydantic 建模，每个维度都是「一个整数分 + 一段理由」成对出现：

```python
class JudgeScore(BaseModel):
    correctness: int = Field(ge=1, le=5, description="正确性(1-5)：...")
    correctness_reason: str = Field(description="对正确性评分的简要中文理由...")
    faithfulness: int = Field(ge=1, le=5, description="忠实度(1-5)：...")
    faithfulness_reason: str = Field(description="...")
    relevance: int = Field(ge=1, le=5, description="相关性(1-5)：...")
    relevance_reason: str = Field(description="...")
```

两个细节很重要：

- `ge=1, le=5` 把分数约束在合法区间，模型给出越界值会被校验拦下。
- `Field(description=...)` 不只是注释——`with_structured_output` 会把这些描述注入给模型，
  相当于把**评分细则**写进了 schema，模型据此理解每个字段该填什么。

### 2. 为什么「先理由、后给分」能提升稳定性

把 `reason` 字段和分数一起放进 schema，并在 system prompt 里要求「为每个维度给出理由」，
本质是一种**让模型先思考再下结论**的结构（类 chain-of-thought）。模型在生成理由的过程中被迫
对照参考答案逐点核对，分数因此更有依据、波动更小，也更可审计——你能看到「为什么给 3 分」。

### 3. LLM-Judge 的去偏（debiasing）

裁判模型有几个众所周知的偏置，必须在 prompt 里显式压制（见 `judge.py` 的 `_JUDGE_SYSTEM`）：

- **长度偏置**：答案越长越倾向给高分。→ 明确要求「不要因为答案更长就给高分」。
- **风格/自信偏置**：措辞华丽、语气笃定就给高分。→ 要求「只看事实与内容」。
- **措辞绑死**：参考答案是「要点」不是唯一标准答案。→ 要求「不同表述说对了要点同样得高分」。
- **维度串扰**：某一维度的好印象拉高其他维度。→ 要求「三个维度相互独立分别评分」。
- **不确定处理**：拿不准时倾向给中间分 3，并在理由里说明，避免乱给极端分。

更进一步的去偏（本项目未做，作为进阶）：交换 A/B 顺序做位置去偏、多次采样取中位数、
用多个裁判模型投票等。

### 4. 结构化输出为何稳健

裸 LLM 让它「输出 JSON」时，常见翻车是：套了 ```json 代码块、前后多了解释、字段拼错或缺失。
用正则/`json.loads` 去解析这种文本非常脆。LangChain 0.3 的：

```python
structured_llm = llm.with_structured_output(JudgeScore)
chain = prompt | structured_llm   # invoke 直接拿到 JudgeScore 对象
```

底层走的是模型的 **tool / function calling**：模型被要求「调用一个参数即 JudgeScore 的函数」，
平台保证返回结构合法，再由 LangChain 反序列化成 Pydantic 对象。无需在 prompt 里塞格式说明，
解析路径短而稳。`harness.py` 仍对偶发 `None` 返回与被测系统异常做了兜底，保证批量评测不被单条拖垮。

### 5. 可复现：温度 0

裁判 LLM 与 demo 被测系统都用 `get_llm(temperature=0)`。评测追求确定性——同样的输入应得到
同样的评分，否则「这次比上次高 0.2 分」可能只是采样噪声。这也是仓库约定中
「裁判用 `get_llm(temperature=0)`」的由来。

### 6. 报告产出

`render_markdown` 生成两部分：**汇总表**（各维度均值、总均值、通过数/通过率）+ **逐条明细表**
（问题、截断答案、三维分数、PASS/FAIL、主要理由）。表格里对长文本截断，并转义 `|` 与换行，
防止破坏 Markdown 表格。同时写一份 `report.json` 保留完整结构化结果，便于程序化对比多次运行
或接入 CI。

## 三、运行

在**仓库根目录**执行（脚本头部已 `load_dotenv()` 并注入根目录到 `sys.path`）：

```bash
# 裁判自检：验证 with_structured_output 链路通畅
python projects_advanced/agent_eval_system/judge.py

# 跑完整评测，生成 report.md / report.json
python projects_advanced/agent_eval_system/harness.py
```

终端会逐条打印进度，最后输出汇总（各维度均值、总均值、通过率），并写出 `report.md` /
`report.json`。具体分数取决于所用模型。

## 四、复盘与进阶

- **换被测系统**：唯一契约是 `target_system(question) -> str`。把函数体换成你的 RAG 链
  `rag_chain.invoke(...)` 或 LangGraph Agent 的 `invoke` 即可，harness 其余不动（详见 README）。
- **评测集才是资产**：本项目只放了 10 条 demo。生产中应持续积累几十上百条、覆盖各类边界，
  每次改动跑一遍做**回归**。
- **加检索层指标**：RAG 场景可利用样本里的 `context`，补「检索命中率 Hit Rate」等检索侧指标，
  实现「检索/生成」分层归因（参见 stage02 的 RAG 评测）。
- **更强去偏**：位置交换、多次采样取中位数、多裁判投票、用人工标注集校准裁判与人的一致性。
- **接 CI/LLMOps**：把 `report.json` 的总均值/通过率设阈值，低于阈值让流水线失败；或接 LangSmith。
- **成本与速度**：每条样本调一次被测系统 + 一次裁判，样本多时可加并发（`batch`）或缓存被测答案。

## 五、面试怎么考

- **你怎么评测一个 RAG/Agent，而不是凭感觉？**
  → 评测集 + 被测系统抽象成函数；LLM-as-a-Judge 多维度打分；聚合均分与通过率；可回归对比。
- **为什么要多维度而不是一个总分？**
  → 正交维度可定位问题：correctness 看事实、faithfulness 抓幻觉、relevance 看跑题；总分无法归因。
- **LLM 当裁判有什么坑，怎么缓解？**
  → 长度/风格/自信/位置等偏置；缓解：prompt 显式去偏、要求先理由后打分、温度 0、
  位置交换、多次采样取中位数、多裁判投票、与人工标注做一致性校准。
- **怎么保证打分能被程序稳定解析？**
  → `with_structured_output(Pydantic)` 走 tool calling，约束输出结构；不靠正则解析裸 JSON；
  再加空返回/异常兜底。
- **怎么保证评测可复现？**
  → 固定评测集 + 裁判温度 0 + 记录每次结果到 JSON，便于横向对比。
- **这套框架怎么复用到我自己的系统？**
  → 只需实现 `(question) -> answer` 的函数；RAG 还可在样本里加 `context` 扩展检索层指标。
