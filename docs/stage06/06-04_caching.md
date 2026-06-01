# 06-04 缓存：省钱又提速的 LLM / Embedding 缓存

> 🎯 **一句话**：给 LLM 调用和 Embedding 计算加缓存——相同输入直接返回上次结果，不再重复花钱调模型，把重复请求的延迟从「秒级」降到「毫秒级」，是 LLM 应用最立竿见影的降本提速手段。

---

## 为什么需要它

LLM 调用既慢又贵，而真实流量里有大量**重复或近似重复**的请求（同一个常见问题被无数人问、文档重建索引时同一段文本被反复 embedding）。每次都真调模型，纯属浪费钱和时间。

缓存的逻辑很简单：**输入相同 → 直接返回缓存的输出**，跳过 LLM/Embedding 调用。命中一次就省一次 API 费用，并把响应从秒级压到毫秒级。

---

## 核心用法

### 1. LLM 响应缓存：InMemoryCache（开发）/ Redis（生产）

```python
from dotenv import load_dotenv
load_dotenv()

from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from common.llm_provider import get_llm

set_llm_cache(InMemoryCache())     # 全局生效：所有 LLM 调用自动走缓存

llm = get_llm(temperature=0)
llm.invoke("什么是 LangChain？")     # 第一次：真调模型，较慢
llm.invoke("什么是 LangChain？")     # 第二次：命中缓存，瞬间返回
```

**本质在干什么？** `set_llm_cache` 设置全局缓存层。LangChain 以「(prompt, 模型参数)」为 key 查缓存：命中就跳过 API 直接返回。`InMemoryCache` 存进程内存，重启即失，仅适合开发/单机。

```python
# 生产换 Redis：跨进程/多副本共享、可持久化、可设过期
from langchain_community.cache import RedisCache
import redis

set_llm_cache(RedisCache(redis.Redis(host="localhost", port=6379)))
```

**本质在干什么？** `RedisCache` 把缓存放进 Redis：多个服务副本共享同一份缓存、重启不丢、能设 TTL 过期。这才是生产用法。

### 2. Embedding 缓存：CacheBackedEmbeddings

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from common.embedding_provider import get_embeddings   # 项目内的 embedding 封装

store = LocalFileStore("./embed_cache/")
cached_embed = CacheBackedEmbeddings.from_bytes_store(
    get_embeddings(), store, namespace="bge-v1",       # namespace 区分不同模型
)
# 同一段文本第二次 embedding 直接读缓存，不再调模型
vecs = cached_embed.embed_documents(["LangChain 是框架", "LangChain 是框架"])
```

**本质在干什么？** RAG 重建索引时大量文本会被重复 embedding。`CacheBackedEmbeddings` 以文本哈希为 key 把向量存起来（这里用本地文件，也可换 Redis），未变的文本不再重算。`namespace` 防止不同模型的向量串味。

### 3. 语义缓存（概念）

```python
# 精确缓存只命中「一模一样」的输入。语义缓存命中「意思相近」的输入：
# 「LangChain 是啥」和「什么是 LangChain」精确缓存不命中，语义缓存能命中。
# 实现思路：把问题 embedding，在向量库里找相似度超阈值的历史问题，命中则返回其答案。
# 工具：GPTCache、Redis 语义缓存等。
```

**本质在干什么？** 精确缓存要求字符完全一致，对自然语言太苛刻。**语义缓存**用 embedding + 相似度匹配「意思相近」的问题，命中率高得多。代价是引入一次 embedding 计算和相似度阈值调参（阈值太松会返回错答案）。

---

## 关键原理 / 实践要点

1. **缓存 key 包含模型参数**：相同 prompt 但不同 temperature/model 是不同 key。注意 `temperature>0` 时缓存会固定住随机结果（第一次抽到啥就一直返回啥）——需要多样性的场景慎用精确缓存。
2. **InMemory vs Redis**：内存缓存快但不跨进程、重启丢；多副本/生产必须用 Redis 等外部存储，并设合理 TTL。
3. **Embedding 缓存收益巨大**：索引重建、增量更新时同一文本反复 embedding，缓存能省下大量计算和 API 费用，`namespace` 务必按模型区分。
4. **精确 vs 语义缓存**：精确缓存零误差但命中率低；语义缓存命中率高但有「返回近似但不完全对口答案」的风险，阈值要保守，关键业务慎用。
5. **缓存失效**：底层知识/文档更新后，旧答案可能过时——要能按 key 清理或设 TTL，避免返回陈旧结果。

---

## 你来改

- [ ] 用 `time` 测量同一 prompt 第一次和第二次 `invoke` 的耗时差，直观感受缓存提速。
- [ ] 把 `temperature` 设为 0.9 后开缓存，连问同一问题，观察结果是否被「冻结」成同一个，理解副作用。
- [ ] 给 `CacheBackedEmbeddings` 换不同 `namespace` 各 embedding 一次，确认互不命中。

---

## 面试怎么考

**Q：LLM 缓存怎么做？InMemoryCache 和 Redis 怎么选？**
A：用 `set_llm_cache` 设全局缓存，LangChain 以 (prompt, 模型参数) 为 key 查缓存，命中则跳过 API 直接返回，省钱并把延迟降到毫秒级。InMemoryCache 存进程内存、快但不跨进程且重启丢失，仅开发用；生产用 RedisCache，多副本共享、可持久化、可设 TTL。

**Q：精确缓存和语义缓存区别？**
A：精确缓存要求输入字符完全一致才命中，零误差但对自然语言命中率低；语义缓存把问题 embedding 后在向量库找相似度超阈值的历史问题，能命中「意思相近」的提问，命中率高但有返回近似/不完全对口答案的风险，阈值需保守。工具如 GPTCache。

**Q：Embedding 也要缓存吗？怎么做？要注意什么？**
A：要。RAG 重建/增量更新索引时同一文本会被反复 embedding，浪费计算和费用。用 CacheBackedEmbeddings 以文本哈希为 key 缓存向量（本地文件或 Redis），未变文本不再重算。注意用 namespace 按模型区分，避免不同模型向量混用；文档更新后要清理过期缓存。
