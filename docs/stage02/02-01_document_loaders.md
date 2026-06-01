# 02-01 Document Loaders：把各种格式的原始文件读成统一的 Document

Document Loaders 解决的问题：RAG 的第一步是「喂数据」，但你的资料散落在 PDF、Markdown、CSV、网页里，格式各异。Loader 的职责就是把这些异构文件统一读成 LangChain 的标准对象 `Document`，后续切分、向量化、检索全都基于这个统一结构。

## 为什么需要它

如果不用 Loader，你得自己写：PDF 用 `pypdf` 抽文字、CSV 用 `pandas` 读行、Markdown 直接 open 读字符串……每种格式一套解析逻辑，而且抽完之后还得自己拼装「正文 + 出处信息」。

更麻烦的是**出处（metadata）会丢**。RAG 不只要正文，还要知道「这段话来自哪个文件的第几页」，否则没法做引用溯源。Loader 在读取时就帮你把 `source`、`page` 等信息塞进 `metadata`，省掉大量胶水代码，也保证了下游组件拿到的永远是同一种对象。

## 核心用法

```python
from dotenv import load_dotenv; load_dotenv()

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, DirectoryLoader,
)

# 1) PDF：按页拆分，每页一个 Document，metadata 自带 page 页码
pdf_docs = PyPDFLoader("docs/sample.pdf").load()

# 2) 纯文本 / Markdown：整文件一个 Document（encoding 防中文乱码）
md_docs = TextLoader("docs/note.md", encoding="utf-8").load()

# 3) CSV：默认「每行一个 Document」，列名:值 拼成 page_content
csv_docs = CSVLoader("data/faq.csv", encoding="utf-8").load()

# 4) 目录批量加载：用 glob 过滤后缀，loader_cls 指定用哪个 Loader
dir_docs = DirectoryLoader(
    "knowledge/", glob="**/*.md",
    loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
).load()

doc = pdf_docs[0]
print(type(doc))            # <class 'langchain_core.documents.Document'>
print(doc.page_content[:50])  # 正文文本
print(doc.metadata)         # {'source': 'docs/sample.pdf', 'page': 0}
```

逐块讲「本质在干什么」：

- **`.load()` 返回 `list[Document]`**：所有 Loader 接口统一，区别只在「一个文件切成几个 Document」。PyPDF 按页切、CSV 按行切、Text 不切。
- **`Document` 只有两个核心字段**：`page_content`（字符串正文）和 `metadata`（字典，存出处）。整个 RAG 管线流动的就是这两样东西。
- **`DirectoryLoader` 是「批量版」**：用 `glob` 模式匹配文件，对每个文件套用 `loader_cls`。它本身不解析内容，只负责遍历和分发。

## 关键参数 / 原理

- **`PyPDFLoader` 的颗粒度是「页」**：扫描件（图片型 PDF）抽不出文字，需要 OCR 类 Loader（如 `UnstructuredPDFLoader`）。`metadata["page"]` 从 0 开始。
- **`CSVLoader` 的 `source_column`**：默认每个 Document 的 `source` 都是文件路径。设 `source_column="url"` 可让某一列的值作为出处，做引用时更精确。`csv_args` 可传分隔符、列名等。
- **`encoding`**：中文文件几乎必传 `encoding="utf-8"`，否则在某些系统上默认 GBK 直接报 `UnicodeDecodeError`。
- **`DirectoryLoader` 的 `show_progress=True` 和 `use_multithreading=True`**：大目录加载慢时，开多线程能显著提速；`silent_errors=True` 可跳过个别坏文件而不中断整批。
- **load() vs lazy_load()**：`load()` 一次性读进内存，文件多/大时用 `lazy_load()` 返回迭代器，边读边处理省内存。

## 你来改

1. 准备一个 `faq.csv`（两列：`question,answer`），用 `CSVLoader` 加载，打印第一个 Document 的 `page_content`，观察它是怎么把列名和值拼成 `"question: ...\nanswer: ..."` 的；再设 `source_column="question"`，看 `metadata["source"]` 的变化。
2. 用 `DirectoryLoader` 加载一个混放 `.md` 和 `.txt` 的目录，要求只加载 `.md`。然后改 `glob="**/*"` 并去掉 `loader_cls`，观察默认 Loader 对未知格式的处理（提示：会用 `UnstructuredFileLoader`，需额外装包）。

## 面试怎么考

**Q：`Document` 对象包含哪些信息？为什么 metadata 在 RAG 里很关键？**
A：`page_content`（正文）和 `metadata`（字典）。metadata 携带 `source`、`page` 等出处信息，是 RAG「答案溯源/引用」的基础；同时可用于检索时的元数据过滤（如只在某个文件、某个时间范围内检索）。

**Q：加载一个 100 页的 PDF，PyPDFLoader 会返回几个 Document？这对后续切分有什么影响？**
A：返回 100 个（每页一个），`metadata["page"]` 标记页码。但「按页」不是合适的检索颗粒度——一页可能太长，跨页的语义又会被页边界切断。所以加载后通常还要再过一遍 `TextSplitter` 重新切块，页码 metadata 会被继承下来。

**Q：扫描版 PDF 用 PyPDFLoader 加载，结果 page_content 是空的，为什么？**
A：PyPDFLoader 只能抽「文本层」。扫描件本质是图片，没有文本层，必须先做 OCR（如 `UnstructuredPDFLoader` 配合 OCR 引擎，或 RapidOCR）才能得到文字。
