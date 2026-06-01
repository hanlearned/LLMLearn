# 02-07 检索策略：similarity、MMR、阈值、混合检索

检索策略解决的问题：拿到查询向量后，「怎么从库里挑出最该给 LLM 的那几条」不止一种答案。只取最相似的几条简单粗暴，但会带来「结果高度重复」「相似但不够格的也混进来」「同义召回强、精确术语召回弱」等问题。不同策略针对不同毛病，是 RAG 调优的核心旋钮。

## 为什么需要它

默认的相似度检索（similarity）有几个典型痛点：

- **冗余**：top-5 可能是同一段话的 5 个高度相似的块，信息量其实只有 1 条，浪费上下文。
- **滥竽充数**：就算库里没有真正相关的内容，它也会硬返回 k 条「最不相关里相对相关」的，给 LLM 喂噪声。
- **语义≠字面**：向量检索擅长同义改写，但对「精确的型号、人名、专有名词、数字」这类字面匹配反而不如关键词。

MMR 治冗余、阈值治滥竽充数、混合检索（向量+BM25）治字面匹配弱。会用这几招，RAG 召回质量上一个台阶。

## 核心用法

```python
from dotenv import load_dotenv; load_dotenv()
from common.embeddings_provider import get_embeddings
from langchain_chroma import Chroma

vs = Chroma(collection_name="my_kb", embedding_function=get_embeddings(),
            persist_directory="./chroma_db")

# 1) similarity：纯按相似度取 top-k（默认）
r1 = vs.similarity_search("如何退货", k=4)

# 2) MMR：在「相关」和「多样」之间权衡，去重去冗余
r2 = vs.max_marginal_relevance_search("如何退货", k=4, fetch_k=20, lambda_mult=0.5)

# 3) 相似度阈值：低于阈值的直接不返回（宁缺毋滥）
retriever = vs.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5, "k": 4},
)
r3 = retriever.invoke("如何退货")     # 可能返回 0 条（库里确实没相关内容时）

# 4) 混合检索：向量(语义) + BM25(关键词) 用 EnsembleRetriever 融合
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# BM25 需要原始 chunk 列表（关键词检索不依赖向量库）
bm25 = BM25Retriever.from_documents(chunks)   # chunks 来自切分阶段
bm25.k = 4
vec_retriever = vs.as_retriever(search_kwargs={"k": 4})

hybrid = EnsembleRetriever(
    retrievers=[bm25, vec_retriever],
    weights=[0.4, 0.6],        # 关键词权重 0.4，向量权重 0.6
)
r4 = hybrid.invoke("退货 7 天 无理由")
```

逐块讲「本质在干什么」：

- **MMR（最大边际相关）**：先用相似度捞出 `fetch_k` 个候选，再逐个挑选——每次选「既和查询相关、又和已选结果不重复」的那个。`lambda_mult` 是平衡旋钮。
- **`similarity_score_threshold`**：给相似度设地板，达不到就不返回。这让 RAG 能诚实地说「我不知道」，而不是强行用噪声编答案。
- **`EnsembleRetriever`**：分别跑两个检索器，用 **RRF（倒数排名融合）** 把两个排名合并成一个，`weights` 调各路的话语权。BM25 补字面匹配的短板，向量补语义的短板。

## 关键参数 / 原理

- **MMR 的 `lambda_mult`（0~1）**：1 = 完全只看相关性（退化成 similarity）；0 = 完全只看多样性（结果发散）。0.5 是常用平衡点。`fetch_k`（默认 20）是候选池大小，越大多样性挑选空间越大但越慢。
- **阈值的「方向陷阱」**：`similarity_score_threshold` 里分数是**相似度（越大越相关）**，LangChain 内部已把距离转成 0~1 的相似度。阈值设多少没有通用值，要拿评测集校准——设太高会漏召（连相关的都被滤掉），太低等于没设。
- **BM25 的本质**：经典的关键词检索算法（TF-IDF 的改进），按词频和逆文档频率打分，对**精确术语、专有名词、数字**极敏感，但完全不懂同义。它和向量检索是互补关系。
- **混合检索为什么强**：用户 query 往往「既有语义又有关键词」（如「2023 年 iPhone 15 的保修政策」——年份/型号靠 BM25，「保修政策」靠向量）。单一策略总有盲区，融合后召回更稳。`weights` 视场景调，技术文档/法条类可调高 BM25 权重。
- **RRF（Reciprocal Rank Fusion）**：融合多路排名的标准做法，按 `1/(rank+常数)` 累加得分，不依赖各路分数量纲是否可比，鲁棒（详见 02-10）。

## 你来改

1. 对同一查询分别用 `similarity_search` 和 `max_marginal_relevance_search`，把 `lambda_mult` 从 1.0 调到 0.2，观察结果从「高度重复」到「发散多样」的变化。
2. 构造一个含精确术语的查询（如某个产品型号），对比纯向量检索和 `EnsembleRetriever`（含 BM25）的召回，体会 BM25 在字面匹配上的补强；再调 `weights` 看排序变化。

## 面试怎么考

**Q：MMR 和普通相似度检索的区别？解决什么问题？**
A：相似度检索只按与查询的相关度取 top-k，容易返回一堆几乎重复的块。MMR 在「与查询相关」和「与已选结果不重复」之间权衡（由 lambda_mult 控制），主动去冗余，让有限的 k 条覆盖更多信息面，提升上下文的有效性。

**Q：为什么要做向量+BM25 的混合检索？**
A：向量检索擅长语义/同义匹配，但对精确术语、型号、数字、人名等字面匹配弱；BM25 关键词检索正好相反。真实查询常二者皆有，融合（如 EnsembleRetriever + RRF）能覆盖彼此盲区，召回更稳健。

**Q：similarity_score_threshold 有什么用？设置时要注意什么？**
A：给相似度设下限，达不到的结果不返回，避免库里没相关内容时硬塞噪声给 LLM，让系统能「答不知道」。注意分数是相似度（越大越相关）方向别搞反，且阈值需用评测集校准——过高漏召、过低失效。
