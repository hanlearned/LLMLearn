# 02-06 Milvus：什么时候你需要一个「企业级」向量数据库

Milvus 解决的问题：Chroma/FAISS 在单机、百万级以内、单人开发时很好用，但当向量上亿、要多副本高可用、多人并发写、按租户隔离、横向扩容时，它们就力不从心了。Milvus 是为「大规模、分布式、生产环境」而生的向量数据库，把检索从「单机库」升级成「集群服务」。

## 为什么需要它

先说清楚：**绝大多数学习项目和中小应用，Chroma 就够了，别过早上 Milvus**。但当你遇到下面任意一条，单机库会成为瓶颈：

- **数据量级**：向量从百万级涨到千万/亿级，单机内存装不下，FAISS 撑不住。
- **高可用**：业务要求 7×24，单机一挂全停；需要多节点副本、故障自动转移。
- **并发与隔离**：多个团队/租户共用，要并发写入、权限隔离、资源配额。
- **运维能力**：要监控、备份、在线扩缩容、滚动升级——这些「数据库该有的能力」单机库基本没有。

Milvus 把存储层、查询层、协调层分离（存算分离），可以独立扩容查询节点应对高 QPS，靠对象存储（如 MinIO/S3）放海量向量，是把 RAG 推向生产规模时的常见选择。同类还有 Qdrant、Weaviate、Pinecone（云托管）。

## 核心用法

```python
from dotenv import load_dotenv; load_dotenv()
from common.embeddings_provider import get_embeddings
from langchain_milvus import Milvus           # pip install langchain-milvus pymilvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

emb = get_embeddings()
chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60).split_documents(
    TextLoader("docs/note.md", encoding="utf-8").load()
)

# 建库 / 写入：connection_args 指向你的 Milvus 服务
vs = Milvus.from_documents(
    documents=chunks,
    embedding=emb,
    collection_name="my_kb",
    connection_args={"uri": "http://localhost:19530"},  # 服务地址
    index_params={"index_type": "HNSW", "metric_type": "COSINE"},
    drop_old=True,        # 重建时先删旧 collection（首次建库常用）
)

# 检索：接口和 Chroma 完全一致（LangChain 统一抽象的价值）
for d in vs.similarity_search("如何退货", k=3):
    print(d.metadata.get("source"), d.page_content[:40])

# 连接已有库（不传 documents，只连服务复用 collection）
vs2 = Milvus(
    embedding_function=emb,
    collection_name="my_kb",
    connection_args={"uri": "http://localhost:19530"},
)
```

起服务（Docker，概念演示，生产用 Helm/K8s 部署集群）：

```bash
# 官方提供 standalone（单机版）一键脚本，适合开发自测
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone.sh
bash standalone.sh start        # 起 Milvus + etcd + MinIO 三个容器
# 默认监听 19530（gRPC）/ 9091（健康检查），停服 bash standalone.sh stop
```

逐块讲「本质在干什么」：

- **`connection_args` 指向远程服务**：和 Chroma 最大的不同——Milvus 是「客户端连服务」，向量存在独立的服务进程/集群里，你的应用只是客户端。
- **`index_params` 显式建索引**：生产库要按数据规模选索引（HNSW 高召回低延迟、IVF_FLAT 省内存、DiskANN 超大规模落盘）和度量（COSINE/L2/IP）。
- **检索接口和 Chroma 一模一样**：得益于 LangChain 的 `VectorStore` 抽象，换库基本只改建库那几行，下游检索链路不动——这正是用 LangChain 的好处。

## 关键参数 / 原理

- **何时该上 Milvus**：以「单机库扛不住」为信号——数据量上千万、要高可用/水平扩展、多租户并发、需要专业运维监控。否则坚持用 Chroma，避免无谓的运维负担。
- **架构区别**：Chroma/FAISS 是「嵌入式库」（和应用同进程或本地文件）；Milvus 是「独立分布式服务」，组件包括协调节点、查询节点、数据节点、索引节点 + etcd（元数据）+ MinIO/S3（向量存储），存算分离，可按需扩各类节点。
- **索引与度量**：`metric_type` 要和 Embedding 匹配（bge 归一化向量用 COSINE/IP）。索引类型决定召回-延迟-内存的三角权衡，超大规模常用 DiskANN（向量落 SSD，内存放不下也能查）。
- **Partition 分区**：可按租户/时间分区，检索时只扫相关分区，兼顾隔离和性能。
- **一致性级别**：Milvus 支持可调一致性（Strong/Bounded/Eventually），写入后能否立刻查到可配置，这是分布式系统特有的考量，单机库没有。

## 你来改

1. （概念）画一张表对比 Chroma、FAISS、Milvus 在「部署形态、最大数据量级、高可用、metadata 过滤、运维成本」五个维度的差异。
2. （动手，可选）用 `standalone_embed.sh` 在本地起一个 Milvus，把 02-05 的建库代码从 Chroma 换成 `langchain_milvus.Milvus`，确认下游 `similarity_search` 代码一行不改也能跑通——体会 LangChain 抽象层的价值。

## 面试怎么考

**Q：Chroma/FAISS 和 Milvus 的本质区别是什么？**
A：Chroma/FAISS 是单机/嵌入式向量库，和应用同进程或本地文件，零运维、适合中小规模；Milvus 是分布式向量数据库服务，存算分离、可水平扩展、支持高可用和多租户，面向亿级向量和生产环境。前者是「库」，后者是「服务/集群」。

**Q：什么情况下才应该从 Chroma 迁到 Milvus？**
A：当单机库出现瓶颈：数据量超出单机内存（千万/亿级）、需要高可用与故障转移、高并发写入、多租户隔离、或需要专业运维（监控/备份/在线扩缩容）。否则不建议过早引入，会徒增复杂度。

**Q：从 Chroma 换到 Milvus，RAG 代码要大改吗？**
A：基本不用。LangChain 用统一的 `VectorStore` 抽象，主要改动只是建库时换成 `Milvus(...)` 并配 `connection_args`、`index_params`；下游 `as_retriever`、检索链等代码因为面向抽象接口，几乎不动。
