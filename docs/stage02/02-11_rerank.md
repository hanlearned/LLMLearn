# 02-11 Re-rank：向量检索之后，再来一轮「精排」

Re-rank 解决的问题：向量检索快，但它把 query 和文档各自独立编码成向量再算距离，是一种「粗排」——为了快牺牲了精度，相关文档可能排进了 top-20 但没进 top-3。Re-rank 用更精细但更慢的模型，对粗排召回的候选**两两重新打分排序**，把真正最相关的顶上来，再交给 LLM。这是「召回-精排」二阶段检索的经典套路。

## 为什么需要它

向量检索（双塔/Bi-Encoder）的本质局限：query 和 document **分别**编码成向量，编码时彼此看不见对方。两个语义其实匹配的句子，可能因为各自向量的细微偏差而距离偏大，排到后面。

所以最佳实践是**两阶段**：

1. **召回（粗排）**：向量检索快速从全库捞出 top-20~50 候选。重速度、要召回率高（别漏）。
2. **精排（rerank）**：用 **Cross-Encoder** 对每个「query+候选文档」**拼在一起**送进模型，输出一个精确的相关性分数，重新排序，取 top-3~5。重精度。

Cross-Encoder 因为让 query 和文档在模型内部充分交互（注意力互相能看到），相关性判断远比向量距离准；代价是慢（每个候选都要过一次模型），所以只能用在「少量候选」的精排阶段，不能用来扫全库。

## 核心用法

```python
from dotenv import load_dotenv; load_dotenv()
from common.embeddings_provider import get_embeddings
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 1) 粗排：向量库先召回较多候选（k 取大些，给精排留空间）
base = Chroma(collection_name="my_kb", embedding_function=get_embeddings(),
              persist_directory="./chroma_db").as_retriever(search_kwargs={"k": 20})

# 2) 精排：BGE-Reranker（Cross-Encoder），重新打分后取 top_n
ce = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")  # 中文强
reranker = CrossEncoderReranker(model=ce, top_n=4)

# 3) 用 ContextualCompressionRetriever 串起来：召回 -> 压缩(精排取 top_n)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base,
)

docs = compression_retriever.invoke("退货需要满足什么条件")
for d in docs:                              # 已是精排后的 top-4
    print(d.metadata.get("source"), d.page_content[:40])
```

逐块讲「本质在干什么」：

- **`base` 的 `k=20`**：粗排故意多召回，因为相关文档可能没在向量 top-3 里，给精排一个足够大的候选池才捞得回来。
- **`HuggingFaceCrossEncoder`**：加载一个 Cross-Encoder 模型（`bge-reranker-v2-m3` 中文效果好）。它接收 `(query, document)` 对，输出相关性分数。
- **`CrossEncoderReranker(top_n=4)`**：一个「文档压缩器」——对 base 召回的 20 条逐一用 Cross-Encoder 打分，按分排序，只留 `top_n` 条。「压缩」在这里指「从多到少地精选」。
- **`ContextualCompressionRetriever`**：把「base 召回」+「compressor 精排」组合成一个标准 retriever，对外仍是 `invoke(query) -> list[Document]`，能无缝接进 02-09 的 RAG 链替换原 retriever。

## 关键参数 / 原理

- **Bi-Encoder vs Cross-Encoder**：Bi-Encoder（Embedding）独立编码、可预存向量、检索 O(1) 查表，快但糙；Cross-Encoder 把 query+doc 拼起来联合编码，精但每对都要实时算、无法预存。所以「Bi 召回 + Cross 精排」各取所长。
- **`k`（召回数）与 `top_n`（精排留存）**：`k` 决定精排的天花板——相关文档若没进这 k 条，精排也救不回来，所以 k 要够大（20~50）；`top_n` 决定喂给 LLM 几条，3~5 常见。
- **精度 vs 延迟**：rerank 增加一次（批量）模型推理，候选越多越慢。本地 Cross-Encoder 在 CPU 上较慢，生产常用 GPU 或厂商的 rerank API（如 SiliconFlow/Cohere/Jina 的 rerank 接口）。
- **BGE-Reranker 系列**：`bge-reranker-base/large/v2-m3`，中文 rerank 一线效果。和 Embedding 不同，它直接输出相关性分数、不产生向量。
- **何时该上 rerank**：当「召回里明明有答案，但 LLM 答不好」时——往往是相关文档排在 top-k 之外或之后位置，被无关内容挤掉。rerank 是 RAG 提精度性价比最高的一招之一。

## 你来改

1. 对同一查询，对比「向量检索直接取 top-4」和「向量召回 top-20 + rerank 取 top-4」的结果，找出被 rerank 从靠后位置提到前面的文档，体会精排的价值。
2. 把 rerank 后的 retriever 接进 02-09 的 `create_retrieval_chain`（直接替换原 retriever），跑一遍完整 RAG，对比加 rerank 前后答案质量。

## 面试怎么考

**Q：向量检索已经能排序了，为什么还要 rerank？**
A：向量检索用 Bi-Encoder 把 query 和文档独立编码再算距离，是粗排，精度有限，相关文档可能排在 top-k 之外。rerank 用 Cross-Encoder 把 query 和候选文档拼在一起联合打分，相关性判断更准，能把真正相关的顶到前面。代价是慢，所以只用于精排少量候选。

**Q：Bi-Encoder 和 Cross-Encoder 的区别？为什么不直接全用 Cross-Encoder？**
A：Bi-Encoder 分别编码 query 和 doc，向量可预存、检索极快但精度糙；Cross-Encoder 联合编码 query+doc，精度高但每对都要实时推理、无法预存。全库用 Cross-Encoder 要对每个文档实时算分，成本不可接受，所以用「Bi 召回 + Cross 精排」两阶段。

**Q：LangChain 里怎么接 rerank？召回数 k 该怎么设？**
A：用 `ContextualCompressionRetriever`，base_retriever 负责向量召回、base_compressor 用 `CrossEncoderReranker` 精排。k 要设大些（20~50）保证相关文档进入候选池——精排无法找回没被召回的文档；top_n 设 3~5 控制最终喂给 LLM 的量。
