# 02-08 VectorStoreRetriever：as_retriever() 与那几个关键参数

`as_retriever()` 解决的问题：向量库的 `similarity_search` 是个具体方法，但 LCEL 链（如 `create_retrieval_chain`）需要的是一个标准化、可组合的「检索器」接口。`as_retriever()` 把向量库包装成统一的 `Retriever`（一个 Runnable），让检索能像积木一样接进任何链路；同时把检索策略和参数收敛到 `search_type` + `search_kwargs` 两个旋钮里。

## 为什么需要它

直接调 `vs.similarity_search(query, k=4)` 有两个问题：一是它是向量库专属方法，换成 BM25、混合检索、带 rerank 的检索器，调用方式全变了，链路没法通用；二是它不是 Runnable，没法用 `|` 接进 LCEL。

`Retriever` 是 LangChain 对「给一段 query、返回一批 Document」的统一抽象——不管底层是 Chroma、FAISS、BM25 还是网络搜索，对外都是 `retriever.invoke(query) -> list[Document]`。`as_retriever()` 就是把向量库转成这个标准件，并通过参数声明「用哪种检索策略、取几条」。

## 核心用法

```python
from dotenv import load_dotenv; load_dotenv()
from common.embeddings_provider import get_embeddings
from langchain_chroma import Chroma

vs = Chroma(collection_name="my_kb", embedding_function=get_embeddings(),
            persist_directory="./chroma_db")

# 默认：similarity，k=4
retriever = vs.as_retriever()

# 指定策略 + 参数
retriever = vs.as_retriever(
    search_type="mmr",                     # similarity / mmr / similarity_score_threshold
    search_kwargs={
        "k": 4,            # 最终返回几条
        "fetch_k": 20,     # MMR 专用：先捞多少候选再做多样性挑选
        "lambda_mult": 0.5,  # MMR 专用：相关性 vs 多样性平衡（0~1）
    },
)

# 阈值策略
retriever = vs.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5, "k": 4},
)

# 元数据过滤：只在符合条件的文档里检索
retriever = vs.as_retriever(
    search_kwargs={"k": 4, "filter": {"source": "docs/policy.md"}},
)

# Retriever 是 Runnable，可直接 invoke，也可用 | 接进链
docs = retriever.invoke("如何退货")
for d in docs:
    print(d.metadata.get("source"), d.page_content[:30])
```

逐块讲「本质在干什么」：

- **`search_type`** 选检索算法：`similarity`（纯相似度）、`mmr`（去冗余）、`similarity_score_threshold`（带阈值过滤）。它决定 `as_retriever` 内部去调向量库的哪个方法。
- **`search_kwargs`** 是「透传字典」：里面的键最终传给底层检索方法。`k`/`filter` 通用，`fetch_k`/`lambda_mult` 只对 MMR 生效，`score_threshold` 只对阈值策略生效——传错不会报错但不起作用，是常见坑。
- **`filter`（元数据过滤）**：先按 metadata 筛掉不符合条件的文档，再在剩下的里做向量检索。RAG 里极有用：按用户权限、文档类型、时间范围限定检索范围。
- **返回值是 `list[Document]`**：和直接 `similarity_search` 一样，但现在它是个 Runnable，能进 LCEL。

## 关键参数 / 原理

- **`k`**：最终喂给 LLM 的块数。太小可能漏关键信息；太大稀释相关性、涨 token 成本。常用 3~6，配合 rerank 时可先取大 k（如 20）再精排到 3~5。
- **`fetch_k`（仅 MMR）**：MMR 的「候选池」。它先用相似度拉 `fetch_k` 条，再从中挑 `k` 条最多样的。`fetch_k` 必须 ≥ `k`，越大多样性挑选余地越大、越慢，默认 20。
- **`lambda_mult`（仅 MMR，0~1）**：1 偏相关、0 偏多样，0.5 平衡。见 02-07。
- **`score_threshold`（仅阈值策略，0~1）**：相似度下限，越大越严。它和 `k` 同时存在时，先按相似度排序取 k，再砍掉低于阈值的，所以实际返回数可能 < k 甚至为 0。
- **`filter` 的语法随后端而变**：Chroma 用 `{"key": value}` 或 `{"key": {"$in": [...]}}`；不同向量库过滤 DSL 不同，迁移时要注意。
- **底层映射**：`as_retriever` 返回的是 `VectorStoreRetriever`，它的 `invoke` 内部根据 `search_type` 分派到 `similarity_search` / `max_marginal_relevance_search` / `similarity_search_with_relevance_scores`。

## 你来改

1. 用同一个库建三个 retriever（similarity / mmr / 阈值），对同一查询各 `invoke` 一次，对比返回条数和内容差异；故意给 similarity 策略传 `lambda_mult`，确认它被无声忽略。
2. 给若干 chunk 的 metadata 加上 `{"category": "售后"}` 或 `{"category": "产品"}`，用 `filter` 只检索某一类，验证元数据过滤把检索范围真的限定住了。

## 面试怎么考

**Q：as_retriever() 相比直接调 similarity_search 有什么意义？**
A：它把向量库包装成统一的 `Retriever`（Runnable）接口，使检索能用 `|` 接进 LCEL 链、能在不改下游代码的前提下替换底层检索器（向量/BM25/混合/带 rerank），并把策略和参数统一到 search_type + search_kwargs，是面向接口而非实现的工程化做法。

**Q：fetch_k 和 k 有什么区别？**
A：`k` 是最终返回数；`fetch_k` 是 MMR 的候选池大小——先按相似度拉 fetch_k 条，再从中挑出 k 条最相关且最不冗余的。fetch_k 只对 MMR 有效且应 ≥ k，越大多样性挑选空间越大但越慢。

**Q：search_kwargs 里的 filter 有什么用？**
A：按文档 metadata 在检索前/检索中做过滤，把检索范围限定在符合条件的文档内（如按用户权限、文档来源、时间范围）。它能提升相关性、做数据隔离，是企业 RAG 做权限控制和多租户的关键手段。
