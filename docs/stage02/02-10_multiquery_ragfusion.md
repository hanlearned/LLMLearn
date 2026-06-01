# 02-10 MultiQueryRetriever 与 RAG-Fusion：用查询改写扩大召回面

MultiQueryRetriever / RAG-Fusion 解决的问题：用户的一句提问往往只覆盖了某个角度的措辞，而相关文档可能用完全不同的词写同一件事。单条 query 的向量只能召回「和这句话相似」的内容，容易漏召。查询改写让 LLM 把原问题扩写成多个不同角度/措辞的子问题，分别检索再合并，从而把召回面撑大。

## 为什么需要它

向量检索的召回质量高度依赖「query 的措辞」。用户问「电脑开不了机怎么办」，文档里写的是「主机无法启动的排查步骤」——措辞差异让向量相似度打了折扣，相关文档可能排在十名开外，进不了 top-k。

这是 RAG 召回的「单视角盲区」。解决思路：**不要只用一句话去检索**。

- **MultiQueryRetriever**：让 LLM 把原问题改写成 N 个语义等价但措辞不同的问题，每个都去检索，把结果**去重合并**。多个视角覆盖更多相关文档。
- **RAG-Fusion**：MultiQuery 的进阶版。同样生成多个子查询，但合并时不是简单去重，而是用 **RRF（倒数排名融合）** 对各路结果**重新打分排序**——在多个子查询里都排名靠前的文档会被推到最前，更鲁棒。

## 核心用法

```python
from dotenv import load_dotenv; load_dotenv()
from common.llm_provider import get_llm
from common.embeddings_provider import get_embeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever

llm = get_llm(temperature=0)
base_retriever = Chroma(
    collection_name="my_kb", embedding_function=get_embeddings(),
    persist_directory="./chroma_db",
).as_retriever(search_kwargs={"k": 4})

# MultiQueryRetriever：LLM 自动改写出多个 query 并合并去重
mq = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

import logging                              # 打开日志能看到 LLM 改写出的子问题
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

docs = mq.invoke("电脑开不了机怎么办")
print("合并后召回:", len(docs))
```

RAG-Fusion 核心是 RRF，原理简单到可以手写：

```python
def reciprocal_rank_fusion(results: list[list], k: int = 60):
    """results: 多个子查询各自的 Document 排名列表。返回融合重排后的列表。"""
    scores = {}                              # 用文档内容做去重 key
    for ranked in results:
        for rank, doc in enumerate(ranked):  # rank 从 0 开始
            key = doc.page_content
            scores.setdefault(key, [0, doc])
            scores[key][0] += 1 / (rank + k)  # 排名越靠前(rank小)，加分越多
    fused = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in fused]

# 生成多查询 -> 各自检索 -> RRF 融合
sub_queries = ["电脑开不了机怎么办", "主机无法启动如何排查", "电脑黑屏不通电的原因"]
all_ranked = [base_retriever.invoke(q) for q in sub_queries]
fused_docs = reciprocal_rank_fusion(all_ranked)[:4]
```

逐块讲「本质在干什么」：

- **`MultiQueryRetriever.from_llm`**：内部用一个 prompt 让 LLM 生成若干改写问题，对每个调 `base_retriever`，结果按内容去重后合并。对调用方仍是普通 retriever 接口。
- **打开日志**：能看到 LLM 实际改写出了哪几个子问题，是调试 MultiQuery 的关键——改写质量直接决定召回。
- **RRF 函数**：`1/(rank+k)` 是核心。它只看**排名**不看原始分数，所以能融合「分数量纲不可比」的多路结果（向量距离 vs BM25 分）。常数 `k`（一般 60）压平头部差距，避免某一路完全主导。

## 关键参数 / 原理

- **MultiQuery 的代价**：N 个子查询 = N 次检索 + 1 次 LLM 改写调用，延迟和成本上升。改写用 `temperature=0` 更稳。子问题数量默认 3 个左右，多了边际收益递减。
- **MultiQuery vs RAG-Fusion 的区别**：MultiQuery 合并方式是「并集去重」，所有召回平等；RAG-Fusion 用 RRF **重排**，让跨子查询的「共识文档」（多个子查询都召回到的）排名更高，抗噪更强、top 结果质量更好。
- **RRF 里的 `k`（≈60）**：平滑常数，越大则不同排名间的分差越小（更看重「是否被召回」而非「排第几」），越小越看重头部排名。60 是论文经验值。
- **和混合检索的关系**：RRF 也是 02-07 `EnsembleRetriever` 融合向量+BM25 用的算法。RAG-Fusion 是「同一检索器、多个改写 query」用 RRF；EnsembleRetriever 是「同一 query、多个检索器」用 RRF。两者可叠加。
- **失效场景**：当原 query 已经很精确、库覆盖也好时，改写带来的额外召回多是噪声，反而拉低精度。要靠评测判断是否值得开。

## 你来改

1. 打开 MultiQuery 的日志，对一个口语化问题观察 LLM 改写出的 3 个子问题，对比「只用原 query」和「MultiQuery」的召回差异。
2. 用上面的 RRF 函数，把「3 个改写 query 各自的 top-5」融合成最终 top-5，打印每个文档的 RRF 得分，找出「在多个子查询里都靠前」因而被推到最前的那个文档。

## 面试怎么考

**Q：MultiQueryRetriever 解决了向量检索的什么问题？**
A：解决「单条 query 措辞局限导致的漏召」。它让 LLM 把原问题改写成多个不同角度/措辞的子问题分别检索再合并，覆盖相关文档可能使用的不同表达，从而扩大召回面、提升召回率。

**Q：RAG-Fusion 和 MultiQuery 的区别？RRF 是什么？**
A：两者都生成多个子查询，区别在合并方式：MultiQuery 是并集去重，RAG-Fusion 用 RRF 重排。RRF（倒数排名融合）按 `1/(rank+k)` 累加各路排名得分，只依赖排名、不依赖原始分数量纲，让在多路中都靠前的「共识文档」排到最前，鲁棒性更强。

**Q：查询扩展一定能提升 RAG 效果吗？**
A：不一定。当原 query 已精确、库覆盖好时，改写带来的额外召回多为噪声，可能拉低精度，同时增加 LLM 改写和多次检索的延迟与成本。是否开启应以评测集上的指标（召回率/精度/端到端答案质量）为准。
