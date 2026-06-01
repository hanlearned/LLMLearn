# 02-12 RAG 评测：用数字回答「我的 RAG 到底好不好」

RAG 评测解决的问题：调 chunk_size、换 Embedding、加 rerank、改 prompt……每一步你都在「优化」，但凭什么说它变好了？靠人工抽几条看「感觉还行」不可复现、不可比较、改一处不知道是好是坏。评测把 RAG 质量量化成可对比的指标，让优化从「拍脑袋」变成「看数字」——这是把 RAG 从 demo 推向生产的分水岭。

## 为什么需要它

RAG 是个**多环节串联**的系统：加载 → 切分 → 向量化 → 检索 → 生成。任何一环出问题都会让最终答案变差，但**最终答案差，你不知道是哪一环的锅**。

更要命的是「改 A 修好了甲，却悄悄弄坏了乙」。没有评测集，你每次改完只能手动试几条，覆盖不全、无法回归。评测的价值就是：

- **定位瓶颈**：把指标拆到「检索」和「生成」两层，分别打分。如果检索指标高但答案差，问题在生成（prompt/LLM）；如果检索指标就低，问题在召回（切分/Embedding/检索策略）。
- **可回归对比**：固定一份评测集，每次改动跑一遍，用同一组数字横向比，确保是真进步而非错觉。
- **抓幻觉**：专门衡量「答案是否忠于检索到的内容」，这是 RAG 最致命的风险。

## 核心指标：分「检索层」和「生成层」两块

RAG 评测的精髓是**分层归因**。把指标对应到环节，才能定位问题：

**检索层（召回得好不好）**

- **命中率 Hit Rate**：top-k 召回的文档里，是否**至少有一篇**是标准答案所在的文档。最直观的「召回有没有捞到」。
- **MRR（平均倒数排名）**：命中文档排在第几位（`1/rank`）。不只看有没有，还看「排得够不够靠前」——这正是 rerank 要优化的指标。
- **Context Recall（上下文召回）**：标准答案中的每个事实点，是否都能在检索到的 context 里找到支撑。衡量「检索是否完整地把答案需要的信息都召回了」。

**生成层（拿到 context 后答得好不好）**

- **Faithfulness（忠实度 / 反幻觉）**：答案里的每个论断，是否都**能由检索到的 context 推出**，没有编造。这是 RAG 最关键的指标——宁可答不全，不能瞎编。
- **Answer Relevancy（答案相关性）**：答案是否切题地回应了用户的问题（而不是答非所问或一堆无关废话）。

记住这张归因表：

| 现象 | 大概率问题在 | 看哪个指标 |
|:---|:---|:---|
| 检索就没捞到相关文档 | 切分 / Embedding / 检索策略 | Hit Rate、Context Recall |
| 相关文档召回了但排太后 | 缺 rerank | MRR |
| context 里有答案但答错/答不全 | prompt / LLM | Faithfulness、Answer Relevancy |
| 答案里有 context 没有的内容 | 幻觉 | Faithfulness |

## 核心用法

### 第一步：构建评测集

评测集是一组 `(问题, 标准答案, 标准出处)`。可人工标注，也可以用 LLM 从文档里「反向出题」加速：

```python
from dotenv import load_dotenv; load_dotenv()

# 评测集：question=问题, ground_truth=标准答案, contexts 检索时填入
eval_set = [
    {"question": "退货需要在几天内申请？", "ground_truth": "7 天内"},
    {"question": "哪些商品不支持退货？",   "ground_truth": "定制类和食品类商品不支持退货"},
    {"question": "退货运费由谁承担？",     "ground_truth": "质量问题由商家承担，否则买家承担"},
]
```

### 第二步：手写一个最小评测（理解原理）

命中率和忠实度的「土法实现」，帮你看清指标到底在算什么：

```python
from common.embeddings_provider import get_embeddings
from common.llm_provider import get_llm
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

vs = Chroma(collection_name="my_kb", embedding_function=get_embeddings(),
            persist_directory="./chroma_db")
retriever = vs.as_retriever(search_kwargs={"k": 4})
llm = get_llm(temperature=0)

prompt = ChatPromptTemplate.from_template(
    "只依据上下文回答，找不到就说\"未提及\"。\n上下文：{context}\n问题：{input}")
rag = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

# Hit Rate：标准答案的关键词是否出现在召回的 context 里
def hit_rate(sample):
    docs = retriever.invoke(sample["question"])
    joined = "".join(d.page_content for d in docs)
    return any(kw in joined for kw in sample["ground_truth"].split())

# Faithfulness：用 LLM 当裁判，判断答案是否完全由 context 支撑（LLM-as-a-Judge）
judge = get_llm(temperature=0)
judge_prompt = ChatPromptTemplate.from_template("""\
判断「答案」是否完全由「上下文」支撑、没有编造。只回 1（忠实）或 0（有幻觉）。
上下文：{context}
答案：{answer}""")

def faithfulness(sample):
    resp = rag.invoke({"input": sample["question"]})
    ctx = "\n".join(d.page_content for d in resp["context"])
    verdict = (judge_prompt | judge).invoke(
        {"context": ctx, "answer": resp["answer"]}).content.strip()
    return 1 if verdict.startswith("1") else 0

hits = sum(hit_rate(s) for s in eval_set) / len(eval_set)
faith = sum(faithfulness(s) for s in eval_set) / len(eval_set)
print(f"Hit Rate: {hits:.2f} | Faithfulness: {faith:.2f}")
```

逐块讲「本质在干什么」：

- **`hit_rate`**：只判断「召回的 context 里有没有标准答案的内容」，不碰生成环节——纯测检索。这是最便宜的检索体检。
- **`faithfulness` 用 LLM 当裁判（LLM-as-a-Judge）**：忠实度、相关性这类「语义判断」没法用字符串匹配，于是让另一个 LLM 读「context + 答案」给出判定。裁判用 `temperature=0` 保证稳定，prompt 要求只输出 1/0 便于聚合成分数。
- **分数是「平均」**：每条样本得 0/1，全集求均值就是该指标的得分，可在不同配置间横向比。

### 第三步：用 ragas 做专业化评测

手写适合理解原理，生产化评测推荐 **ragas**——它把上述指标标准化、内置成熟的 LLM-as-a-Judge prompt，输出多维度分数：

```python
# pip install ragas datasets
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy, context_recall, context_precision,
)
from datasets import Dataset

# 先用你的 RAG 链跑出每个问题的 answer 和 contexts
rows = []
for s in eval_set:
    resp = rag.invoke({"input": s["question"]})
    rows.append({
        "question": s["question"],
        "answer": resp["answer"],
        "contexts": [d.page_content for d in resp["context"]],  # 注意是 list[str]
        "ground_truth": s["ground_truth"],
    })

# ragas 需要传入用于评判的 LLM 和 Embedding（可复用仓库的 provider）
result = evaluate(
    Dataset.from_list(rows),
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    llm=get_llm(temperature=0),
    embeddings=get_embeddings(),
)
print(result)   # {'faithfulness': 0.92, 'answer_relevancy': 0.88, 'context_recall': 0.80, ...}
```

逐块讲「本质在干什么」：

- **`contexts` 必须是 `list[str]`**：ragas 要逐条 context 去核对事实点，所以传的是每个 Document 的正文列表，不是拼好的字符串。
- **四个指标各管一摊**：`faithfulness`/`answer_relevancy` 测生成层，`context_recall`/`context_precision` 测检索层（precision 看召回里有多少是真有用的、噪声多不多）。一次跑出全景。
- **ragas 内部也是 LLM-as-a-Judge**：它把答案拆成原子论断、把 context 拆成事实点，用 LLM 逐一核对，比土法的字符串匹配精细得多。所以 ragas 评测本身也要花 LLM 调用（有成本）。

## 关键参数 / 原理

- **LLM-as-a-Judge 的可信度**：裁判也是 LLM，会有偏差（如偏好长答案、自我一致性问题）。提升手段：裁判用更强/温度=0 的模型、prompt 给明确评分标准（rubric）、必要时多次投票取多数。详见 Stage 5 的 LLM-as-a-Judge 专题。
- **检索层 vs 生成层要分开看**：这是 RAG 评测最重要的方法论。先确认检索指标（Hit Rate/Context Recall）达标，再看生成指标——否则生成层分数低，你分不清是「没召回到」还是「召回了但没答好」。
- **评测集规模与代表性**：几十条精标样本好过几百条糙样本。要覆盖真实问题分布（含边界、含「库里没有」的问题来测系统会不会硬编）。
- **指标会互相拉扯**：调高 `k` 提升 Context Recall，但可能引入噪声拉低 Context Precision 和 Faithfulness。评测的意义就是看这种 trade-off 的净效果，而不是单看一个指标。
- **离线评测 ≠ 线上效果**：离线评测集是固定的，上线后还要看真实流量的反馈（点踩、人工抽检）。评测是必要条件不是充分条件。

## 你来改

1. 手写 `hit_rate`，把 `k` 从 2 调到 8，画出 Hit Rate 随 k 的变化曲线，找到「召回率开始饱和」的 k——这就是你的召回数性价比拐点。
2. 用 ragas 跑两套配置：A=纯向量检索，B=向量+rerank（02-11）。对比 `context_recall` 和 `faithfulness`，验证 rerank 是否真的提升了端到端质量，用数字而不是感觉下结论。

## 面试怎么考

**Q：RAG 评测为什么要把指标分成检索层和生成层？**
A：RAG 是检索+生成串联系统，最终答案差无法直接定位是哪一环的问题。把指标分层（检索看 Hit Rate/Context Recall，生成看 Faithfulness/Answer Relevancy）能做归因：检索指标低说明问题在切分/Embedding/检索策略，检索高但生成差说明问题在 prompt/LLM。分层才能精准优化。

**Q：Faithfulness（忠实度）衡量什么？为什么它是 RAG 最关键的指标？**
A：衡量答案中的每个论断是否都能由检索到的 context 推出、没有编造。它直接对应 RAG 最致命的风险——幻觉。RAG 的价值就是「基于可信资料回答」，如果答案脱离 context 自由发挥，等于退化成普通 LLM 还更危险，所以忠实度优先级最高。

**Q：什么是 LLM-as-a-Judge？它的局限和应对？**
A：用一个 LLM 充当裁判，读「问题/上下文/答案」按标准给出评分，用于忠实度、相关性这类无法用规则量化的语义判断。局限是裁判本身有偏差（偏好长答案、不稳定、可能被诱导）。应对：裁判用强模型+温度 0、给明确 rubric、多次投票取多数、必要时人工抽检校准。ragas 就是基于这套思路把指标标准化的工具。
