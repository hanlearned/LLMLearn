# 02-05 Chroma 与 FAISS：把向量存下来、快速查回来

Chroma / FAISS 解决的问题：Embedding 把文档变成了上万条向量，但你不可能每次检索都把全库向量和查询向量逐一算相似度——太慢。向量数据库负责**持久化存储**这些向量，并用索引结构实现「毫秒级找出最相似的 k 条」。它是 RAG 的「记忆仓库」。

## 为什么需要它

不用向量库，朴素做法是把所有向量放内存里，查询时和每条算余弦相似度再排序（暴力检索）。几百条没问题，但上万条、上百万条时，每次查询都全量计算，延迟爆炸，而且**进程一退向量全没了**，下次得重新 embedding（费钱费时）。

向量库解决两件事：**持久化**（存到磁盘，重启即用）和**加速检索**（用 HNSW、IVF 等近似最近邻索引，用一点点精度换几个数量级的速度）。Chroma 和 FAISS 是本地/单机场景最常用的两个，零运维、适合学习和中小项目。

## 核心用法

```python
from dotenv import load_dotenv; load_dotenv()
from common.embeddings_provider import get_embeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

emb = get_embeddings()
docs = TextLoader("docs/note.md", encoding="utf-8").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60).split_documents(docs)

# ===== Chroma：建库时直接 persist 到磁盘 =====
from langchain_chroma import Chroma

vs = Chroma.from_documents(
    documents=chunks,
    embedding=emb,
    collection_name="my_kb",
    persist_directory="./chroma_db",   # 指定目录即自动落盘
)

# 相似度查询：返回最相关的 k 个 Document
results = vs.similarity_search("如何退货", k=3)
for d in results:
    print(d.metadata.get("source"), d.page_content[:40])

# 带分数：分数越小越相似（Chroma 默认返回 L2 距离）
for d, score in vs.similarity_search_with_score("如何退货", k=3):
    print(round(score, 4), d.page_content[:30])

# 加载已有库（不重新 embedding，直接复用磁盘上的）
vs2 = Chroma(
    collection_name="my_kb",
    embedding_function=emb,
    persist_directory="./chroma_db",
)

# ===== FAISS：建库后手动 save/load 到本地 =====
from langchain_community.vectorstores import FAISS

fvs = FAISS.from_documents(chunks, emb)
fvs.save_local("./faiss_index")        # 写出 index.faiss + index.pkl 两个文件

fvs2 = FAISS.load_local(
    "./faiss_index", emb,
    allow_dangerous_deserialization=True,   # pkl 反序列化需显式确认
)
print(fvs2.similarity_search("如何退货", k=2)[0].page_content[:40])
```

逐块讲「本质在干什么」：

- **`from_documents`**：一步完成「对每个 chunk 调 `embed_documents` → 把向量+正文+metadata 存进库」。这是建库入口。
- **Chroma 的 `persist_directory`**：传了目录就自动把数据落盘（新版无需再手动调 `.persist()`）。下次用 `Chroma(persist_directory=...)` 构造即加载，**不会重新 embedding**。
- **FAISS 的 `save_local` / `load_local`**：FAISS 本身是纯内存库，必须显式存盘。`load_local` 需要 `allow_dangerous_deserialization=True`，因为它用 pickle，加载不可信文件有安全风险。
- **`similarity_search` vs `..._with_score`**：后者额外返回距离分数，可用于设阈值过滤掉「最相似但其实也不够相似」的结果。

## 关键参数 / 原理

- **Chroma vs FAISS 怎么选**：Chroma 自带持久化、支持 metadata 过滤、有 collection 概念，更像「数据库」，适合要增删改查、按元数据筛选的场景；FAISS 是 Facebook 的纯检索库，极致快、索引类型丰富（IVF/HNSW/PQ），但**默认不带元数据过滤、需手动管存盘**，适合追求检索性能、数据相对静态的场景。
- **距离度量**：Chroma 默认 `l2`（欧氏距离，越小越近），可在 `collection_metadata={"hnsw:space": "cosine"}` 改成余弦。FAISS 默认 L2，配合归一化向量时常用内积。**注意分数方向**：距离是越小越相似，相似度是越大越相似，别搞反。
- **索引原理（近似最近邻 ANN）**：精确检索是 O(N) 暴力。Chroma/FAISS 用 HNSW（分层图）或 IVF（倒排+聚类）把检索降到近似 O(log N)，代价是可能漏掉极个别真·最近邻——这就是「召回率 vs 速度」的权衡。
- **增量更新**：Chroma 支持 `vs.add_documents([...])` 持续追加；FAISS 用 `fvs.add_documents`/`merge_from`，但改完要重新 `save_local`。
- **collection_name**：Chroma 一个目录可放多个 collection（相当于多张表），加载时名字要对上。

## 你来改

1. 用同一批 chunk 分别建 Chroma 和 FAISS，对同一个查询各取 top-3，对比返回的 Document 和分数顺序是否一致；把 Chroma 的距离改成 `cosine` 再对比。
2. 建好 Chroma 库后关闭进程，重新起一个脚本**只**用 `Chroma(persist_directory=...)` 加载并检索，确认没有触发任何新的 embedding 调用（说明持久化生效）。

## 面试怎么考

**Q：Chroma 和 FAISS 的核心区别？什么时候选哪个？**
A：Chroma 是带持久化、支持 metadata 过滤和 collection 管理的轻量向量数据库，开箱即用、适合需要增删改查和元数据筛选的应用；FAISS 是高性能纯检索库，索引类型丰富、速度极致，但默认无元数据过滤、需手动存盘，适合数据较静态、追求检索性能的场景。学习和中小 RAG 用 Chroma 最省心。

**Q：向量库为什么能比暴力检索快？代价是什么？**
A：用近似最近邻索引（HNSW/IVF）把全量逐一比对（O(N)）降到近似对数级。代价是「近似」——可能极小概率漏掉真正的最近邻，即用一点召回率换大幅提速；可通过调索引参数（如 HNSW 的 ef、IVF 的 nprobe）在速度和召回间权衡。

**Q：`similarity_search_with_score` 返回的分数越大越相似吗？**
A：要看度量。Chroma 默认返回 L2 距离，**越小越相似**；若设为余弦距离也是越小越相似。不能想当然认为分数大=相似，用阈值过滤前必须确认度量方向，否则会过滤反。
