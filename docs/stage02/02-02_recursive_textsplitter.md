# 02-02 RecursiveCharacterTextSplitter：把长文档切成「语义不破碎」的小块

RecursiveCharacterTextSplitter 解决的问题：文档动辄几千字，但 Embedding 模型和检索都需要「小而完整」的文本块。直接按固定字数硬切会把一句话拦腰斩断，破坏语义；这个切分器通过「优先在自然边界（段落 → 句子 → 词）下刀」，在控制块大小的同时尽量保住语义完整。

## 为什么需要它

为什么不能整篇文档丢进向量库？因为检索是按「块」召回的。块太大，一次召回带进来大量无关内容，稀释了相关信息，也撑爆 LLM 上下文；块太小，一个完整观点被切散，召回到的片段缺少上下文，答非所问。

为什么不用最简单的「每 500 字切一刀」（`CharacterTextSplitter`）？因为它只认一种分隔符，到了 500 字不管你是不是句子中间，咔嚓就断。`RecursiveCharacterTextSplitter` 的「递归」就是为解决这个：它有一组**从粗到细的分隔符**，先试着按段落切，块还太大就退一级按句子切，再不行才按字符切——尽量在自然停顿处下刀。

## 核心用法

```python
from dotenv import load_dotenv; load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = open("docs/note.md", encoding="utf-8").read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每块目标长度（按字符数）
    chunk_overlap=80,      # 相邻块重叠的字符数
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],  # 从粗到细
    length_function=len,   # 用什么衡量「长度」，默认 len（字符数）
)

chunks = splitter.split_text(text)        # 输入字符串 -> list[str]
print(len(chunks), len(chunks[0]))

# 实际项目里更常见：直接切 Document，自动继承 metadata
from langchain_community.document_loaders import TextLoader
docs = TextLoader("docs/note.md", encoding="utf-8").load()
split_docs = splitter.split_documents(docs)   # list[Document]，每块带原 source
```

逐块讲「本质在干什么」：

- **`separators` 是优先级列表**：切分器拿到一段超长文本，先用 `"\n\n"`（段落）切；若某段仍超过 `chunk_size`，对它再用下一级 `"\n"` 切；层层递归，直到块够小或退到 `""`（逐字符）兜底。
- **`chunk_overlap` 是「滑窗重叠」**：相邻两块共享一段文字。目的是防止「答案正好横跨两块边界」时两边都召回不全——重叠区相当于给边界处的语义上了双保险。
- **`split_documents` 而非 `split_text`**：前者保留并复制每个 Document 的 metadata 到所有子块，做溯源时块仍知道自己来自哪个文件。

## 关键参数 / 原理

- **`chunk_size`**：直接决定检索颗粒度。**偏大**（800-1000）适合需要长上下文的问答（如政策解读），但召回噪声多；**偏小**（200-400）精度高、适合 FAQ 式精确匹配，但可能丢上下文。中文一般 300-500 字是平衡点。
- **`chunk_overlap`**：经验值取 `chunk_size` 的 10%-20%。太小（=0）边界信息易丢；太大则块间高度冗余，向量库膨胀、召回重复内容。
- **中文 `separators` 要自定义**：默认分隔符是为英文设计的（按空格分词），中文没有空格，必须把中文标点 `。！？，` 加进去，否则递归很快退到逐字符切，切出来的块会在句子中间断裂。
- **`chunk_size` 的单位是「length_function 的返回值」**：默认 `len` 数字符。若想按 token 控制，把 `length_function` 换成 tokenizer 的计数函数（见 02-03）。
- **重叠对召回的影响**：`overlap` 越大，关键句被某个块「完整包含」的概率越高，召回率上升，但代价是存储和后续 LLM token 成本增加。

## 你来改

1. 把同一篇中文长文分别用 `chunk_size=200` 和 `chunk_size=800` 切，打印块数和前两块内容，对比「块多而碎」与「块少而全」的差异。
2. 故意把 `separators` 设成只有 `[""]`（强制逐字符切），再设成包含中文标点的版本，对比两种切法在「句子是否被切断」上的差别——你会直观看到为什么中文必须定制分隔符。

## 面试怎么考

**Q：`RecursiveCharacterTextSplitter` 和 `CharacterTextSplitter` 的区别？**
A：`CharacterTextSplitter` 只用**单一**分隔符，到长度就切，容易切断句子。`Recursive` 用一组**从粗到细的分隔符递归**切分，优先在段落、句子等自然边界下刀，只有在仍超长时才退到更细的粒度，语义完整性显著更好，是绝大多数 RAG 场景的默认选择。

**Q：chunk_overlap 的作用是什么？设成 0 有什么风险？**
A：让相邻块共享一段文字，防止答案正好落在块边界而两侧都召回不全。设 0 时若关键信息横跨两块，可能两块各拿到一半，检索时相关度都不够高而漏召，导致 LLM 缺上下文。

**Q：chunk_size 怎么定？过大或过小分别有什么问题？**
A：没有万能值，取决于文档类型和问答粒度。过大：单块噪声多、上下文成本高、相关信号被稀释；过小：语义被切散、召回片段缺上下文。实践上要结合检索评测（命中率/上下文召回）调，中文常用 300-500 字起步。
