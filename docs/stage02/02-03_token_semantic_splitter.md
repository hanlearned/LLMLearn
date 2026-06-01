# 02-03 TokenTextSplitter vs CharacterTextSplitter：为什么要按 token 切

TokenTextSplitter 解决的问题：LLM 和 Embedding 模型看世界的单位不是「字符」而是「token」。如果你用字符数控制块大小，切出来的块在 token 维度上可能超限或浪费——按 token 切才能精准对齐模型的真实容量。

## 为什么需要它

`CharacterTextSplitter`（及 02-02 的 Recursive 版）默认按**字符数**算长度。问题在于：**字符数和 token 数不成正比**。英文里 1 token ≈ 4 个字符；中文里 1 个汉字常常就是 1-2 个 token。也就是说同样 500 字符，中文的 token 数可能是英文的好几倍。

后果很现实：你按「500 字符」切块，以为很安全，结果中文块实际有 600+ token，超过了 Embedding 模型的输入上限（很多模型是 512 token），被悄悄截断——块的后半段根本没被编码进向量，检索时这部分内容等于不存在。反过来，按字符切英文又可能远没吃满 token 预算，浪费上下文。`TokenTextSplitter` 让块大小直接以 token 计量，和模型的真实约束一一对应。

## 核心用法

```python
from dotenv import load_dotenv; load_dotenv()
from langchain_text_splitters import (
    CharacterTextSplitter, TokenTextSplitter, RecursiveCharacterTextSplitter,
)

text = "RAG 系统的检索质量取决于 Embedding 把语义编码成向量的质量。" * 30

# A) 按字符切：chunk_size 单位是「字符」
char_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20, separator="。")

# B) 按 token 切：chunk_size 单位是「token」，用 tiktoken 的编码器计数
token_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20)

# C) 推荐：Recursive 的「保语义」+ token 的「准计量」二合一
hybrid = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=20,
)

for name, sp in [("char", char_splitter), ("token", token_splitter), ("hybrid", hybrid)]:
    chunks = sp.split_text(text)
    print(name, "块数:", len(chunks), "| 首块字符长:", len(chunks[0]))
```

逐块讲「本质在干什么」：

- **`TokenTextSplitter`** 内部用 `tiktoken`（OpenAI 的分词器）把文本编码成 token 序列，再每 `chunk_size` 个 token 切一刀。它**不看语义边界**，纯粹按 token 数硬切，所以可能切断句子。
- **`from_tiktoken_encoder` 是「最佳实践」组合**：外层仍是 Recursive 的递归分隔符（保住段落/句子边界），但**衡量块长度时换成数 token 而非数字符**。既不切碎语义，又精确卡住 token 上限。
- **三者的块数会不同**：同样 `chunk_size=100`，char 切出来块多/少取决于字符密度，token 切则严格对齐模型预算。

## 关键参数 / 原理

- **`encoding_name` / `model_name`**：`TokenTextSplitter` 默认用 `gpt2` 编码器。要和你实际用的模型对齐，可传 `encoding_name="cl100k_base"`（GPT-4/3.5 系）或 `model_name=...`。不同 tokenizer 对中文的 token 数估计差异不小。
- **为什么 token 计数对中文尤其重要**：`cl100k_base` 对常见汉字大多 1 字 = 1 token，但生僻字、标点、emoji 可能拆成 2-3 个 byte token。纯字符计数完全感知不到这种膨胀。
- **截断风险的根源**：Embedding 模型有 `max_seq_length`（bge-large-zh 是 512 token）。超过部分被截断且**不报错**，是 RAG 里很隐蔽的 bug。按 token 控制块大小（留点余量，如 480）能根除它。
- **选型结论**：日常优先用 `RecursiveCharacterTextSplitter.from_tiktoken_encoder`——拿到「保语义」和「准 token」两个好处。纯 `TokenTextSplitter` 只在「就是要严格等长 token 块」（如某些定长输入场景）时用。

## 你来改

1. 找一段中英混排文本，分别用 `CharacterTextSplitter(chunk_size=200)` 和 `TokenTextSplitter(chunk_size=200)` 切，打印每块的「字符长度」和「token 长度」（用 `len(tiktoken.get_encoding("cl100k_base").encode(chunk))` 数 token），观察字符 200 对应的 token 数波动有多大。
2. 把 `from_tiktoken_encoder` 的 `chunk_size` 设到 600（超过 bge 的 512 上限），思考为什么这会导致 Embedding 静默截断，再调回 480 留余量。

## 面试怎么考

**Q：为什么 RAG 切分推荐按 token 而不是按字符？**
A：因为 Embedding/LLM 的容量限制是以 token 计的，字符数和 token 数不成线性关系（中文尤甚）。按字符切可能让块在 token 维度上超过 Embedding 的输入上限而被静默截断，导致部分内容没进向量；按 token 切能精准对齐模型真实约束。

**Q：`TokenTextSplitter` 会切断句子吗？怎么兼顾语义和 token 精度？**
A：会，它纯按 token 数硬切、不看语义边界。最佳实践是用 `RecursiveCharacterTextSplitter.from_tiktoken_encoder`：递归分隔符负责在句子/段落边界下刀保语义，长度衡量用 tiktoken 数 token 保精度，二者兼得。

**Q：不同模型的 token 数会一样吗？这对切分有什么影响？**
A：不一样。不同 tokenizer（cl100k_base、gpt2、各家自研）对同一文本的 token 切分规则不同，中文差异更明显。切分时应尽量用与目标模型一致的 encoder 计数，否则「按 token 控制」会失准。
