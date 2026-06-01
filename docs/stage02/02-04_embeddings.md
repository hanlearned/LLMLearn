# 02-04 Embedding 选型：决定 RAG 检索上限的那一步

Embedding 解决的问题：检索的本质是「按语义找相似」，但计算机只会算数。Embedding 模型把每段文本编码成一个高维向量，让「语义相近」变成「向量距离近」，这样才能用数学方法（余弦相似度）从海量文本里把相关内容捞出来。它是 RAG 整条链路里**对最终效果影响最大**的一环。

## 为什么需要它

为什么不能用关键词匹配（如 BM25）就好？因为用户问「怎么退货」，文档里写的是「申请售后/退换流程」——关键词对不上，但语义是一回事。Embedding 把两者都映射到向量空间里相近的位置，于是能召回。

为什么选型如此关键？因为**检索质量是 RAG 的天花板**。LLM 再强，也只能基于你检索到的内容回答；如果 Embedding 把相关文档排在了第 50 名，它们根本进不了 LLM 的上下文。选错 Embedding（尤其中文场景用英文模型），相关文档和无关文档的向量混在一起，召回直接崩盘。

## 核心用法

```python
from dotenv import load_dotenv; load_dotenv()
from common.embeddings_provider import get_embeddings

emb = get_embeddings()                     # 仓库已按可用 Key 自动选 bge-large-zh

vec = emb.embed_query("如何申请退货")        # 单条 -> List[float]
print("维度:", len(vec))                    # bge-large-zh-v1.5 -> 1024 维

vecs = emb.embed_documents([               # 批量编码文档块（建库时用这个）
    "退货请在 7 天内提交申请",
    "今天天气不错",
])

# 直观感受「语义相似 = 向量距离近」
import numpy as np
def cos(a, b):
    a, b = np.array(a), np.array(b)
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

q = emb.embed_query("怎么退货")
print("相关:", round(cos(q, vecs[0]), 3))    # 明显更高
print("无关:", round(cos(q, vecs[1]), 3))    # 明显更低
```

逐块讲「本质在干什么」：

- **`embed_query` vs `embed_documents`**：有些模型对「查询」和「文档」用不同的编码方式（如加指令前缀），LangChain 用两个方法区分。建库用 `embed_documents`，检索时对用户问题用 `embed_query`。
- **`get_embeddings()` 的自动选型**：仓库封装优先用 SiliconFlow 的 `bge-large-zh-v1.5`，没 Key 时兜底本地 `bge-small-zh`。你不用关心底层是哪家，接口统一。
- **余弦相似度**：检索的数学内核。相关文本的向量夹角小、cos 值接近 1；无关的接近 0。向量库内部就是在算这个并排序。

## 关键参数 / 原理

- **中文为何首选 bge-large-zh**：BAAI 的 BGE 系列在中文检索基准（C-MTEB）上长期领先，是在中文语料上训练的，对中文语义把握远好于 `text-embedding-3`（偏英文/通用）。`large` 比 `small/base` 维度更高、效果更好，代价是更慢更占空间。学习/小项目用 `bge-large-zh-v1.5`，资源紧张可降到 `base`。
- **向量维度**：bge-large-zh 是 1024 维，OpenAI `text-embedding-3-small` 是 1536 维。维度不是越高越好——它决定了向量库的存储和检索成本，也必须**全库统一**：换 Embedding 模型 = 维度可能变 = 整个向量库要重建，不能新旧混用。
- **归一化（normalize）**：把向量缩放成单位长度。归一化后，余弦相似度 = 点积，计算更快；FAISS 的内积索引（`IndexFlatIP`）必须配合归一化向量才等价于余弦检索。bge 系列官方推荐归一化（仓库本地分支已设 `normalize_embeddings=True`）。
- **查询指令前缀**：bge 系列建议给查询加前缀（如 `"为这个句子生成表示以用于检索相关文章："`）来提升检索效果，v1.5 已弱化此需求；通过 API 用时一般无需手动加。
- **三类来源对比**：OpenAI（API、通用强、要海外网络）、HuggingFace 本地（免费离线、首次下模型、需 GPU 才快）、BGE via API（中文一线、无需本地资源，**学习推荐**）。

## 你来改

1. 用 `get_embeddings()` 对一组句子编码，自己写余弦相似度函数，验证「同义不同词」（如「退货」vs「退换货」）的相似度明显高于「无关句子」。
2. 打印 `len(emb.embed_query("test"))` 确认维度；然后思考：如果你已经用 bge-large（1024 维）建好了库，现在想换成 OpenAI（1536 维），能直接换吗？为什么必须重建库？

## 面试怎么考

**Q：中文 RAG 为什么不直接用 OpenAI 的 embedding？**
A：OpenAI 的 embedding 偏英文/通用语料，对中文细粒度语义区分不如在中文上训练的 BGE 系列。在中文检索基准（C-MTEB）上 bge-large-zh 等模型显著更优，且无需海外网络。所以中文场景首选 bge-large-zh-v1.5。

**Q：向量归一化有什么用？不归一化会怎样？**
A：归一化把向量变成单位长度，使余弦相似度等价于点积，计算更快，也是 FAISS 内积索引做余弦检索的前提。不归一化时，向量的「长度」会干扰相似度判断（长向量天然点积大），用内积索引会得到错误排序。

**Q：换 Embedding 模型后，原来的向量库还能用吗？**
A：不能。不同模型的向量空间不同、维度也可能不同，新查询向量和旧库向量不在同一空间，相似度无意义。必须用新模型把所有文档重新编码、重建向量库。
