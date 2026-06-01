# 02-09 create_retrieval_chain：把「检索 + 生成」装成一条 RAG 链

`create_retrieval_chain` 解决的问题：RAG 的完整流程是「拿用户问题 → 检索相关文档 → 把文档塞进 prompt → LLM 生成答案」。手写这套胶水代码繁琐且容易出错。LangChain 用 `create_retrieval_chain` + `create_stuff_documents_chain` 两个工厂函数，把这套标准流程封装成一条可 `invoke`/`stream` 的 LCEL 链，并把检索到的出处一并返回。

## 为什么需要它

不用它，你得自己：调 retriever 拿 docs → 把多个 Document 的 `page_content` 拼成一个字符串 → 填进 prompt 的 `{context}` → 调 LLM → 还想保留出处就得自己另存一份。这些步骤每个 RAG 项目都一样，容易在「文档怎么拼」「变量名对不对」上踩坑。

这两个函数把它标准化：

- **`create_stuff_documents_chain`**：负责「stuff（塞填）」——把一批 Document 格式化后填进 prompt 的 `{context}` 占位符，再过 LLM。「stuff」是最常用的文档组合策略（一次性全塞进上下文）。
- **`create_retrieval_chain`**：在前者外面再包一层，自动先调 retriever 检索，把结果喂给上面的链，并在输出里**同时返回 `answer` 和 `context`**（出处）。

## 核心用法

```python
from dotenv import load_dotenv; load_dotenv()
from common.llm_provider import get_llm
from common.embeddings_provider import get_embeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

llm = get_llm(temperature=0)
vs = Chroma(collection_name="my_kb", embedding_function=get_embeddings(),
            persist_directory="./chroma_db")
retriever = vs.as_retriever(search_kwargs={"k": 4})

# prompt 必须包含 {context}（放检索到的文档）和 {input}（用户问题）
prompt = ChatPromptTemplate.from_template("""\
你是知识库助手。只依据下面的「上下文」回答，找不到答案就说"资料中未提及"，不要编造。

上下文：
{context}

问题：{input}
""")

# 1) 文档组合链：把 docs 填进 {context} 再过 llm
doc_chain = create_stuff_documents_chain(llm, prompt)

# 2) 检索链：先检索，再把 docs 喂给 doc_chain
rag_chain = create_retrieval_chain(retriever, doc_chain)

resp = rag_chain.invoke({"input": "退货需要在几天内申请？"})
print(resp["answer"])                       # LLM 生成的答案
for d in resp["context"]:                   # 检索到的出处 Document 列表
    print("出处:", d.metadata.get("source"), "|", d.page_content[:30])
```

逐块讲「本质在干什么」：

- **prompt 的两个固定变量**：`{context}`（由链自动填入检索到的文档）和 `{input}`（用户问题）。变量名是约定，写错链就拿不到数据。
- **`create_stuff_documents_chain(llm, prompt)`**：返回的链接收 `{"context": list[Document], "input": ...}`，内部把 Document 列表格式化拼接后填进 `{context}`，再调 LLM。你不用手动拼字符串。
- **`create_retrieval_chain(retriever, doc_chain)`**：返回的链只需 `{"input": 问题}`。它内部：用 `input` 调 retriever → 把结果作为 `context` → 调 doc_chain → 组装输出。
- **输出结构是字典**：`{"input": ..., "context": [Document...], "answer": "..."}`。`answer` 给用户看，`context` 用来做引用溯源/可信度展示——这是 RAG 产品「附来源」功能的数据来源。

## 关键参数 / 原理

- **为什么 prompt 里必须有 `{context}`**：`create_stuff_documents_chain` 强制要求 prompt 含此变量，否则报错——这是它注入文档的唯一入口。少了它链不知道把检索结果放哪。
- **`document_prologue` / 文档格式化**：可传 `document_separator`、`document_prompt` 自定义每个 Document 怎么渲染进 context（例如在每段前加 `[来源: {source}]`，让 LLM 在答案里带出处）。
- **「stuff」策略的边界**：它把所有检索文档一次性塞进上下文，简单高效，但文档总量不能超过 LLM 上下文窗口。`k` 太大或块太长会溢出。海量文档场景需 map-reduce/refine 等策略（牺牲速度换容量）。
- **拿到 context 出处的意义**：`resp["context"]` 让你能向用户展示「答案依据来自哪几段、哪个文件」，是 RAG 可信度和反幻觉的关键。结合 prompt 里「找不到就说未提及」的约束，能显著抑制编造。
- **流式**：`rag_chain.stream(...)` 可流式输出，但要注意流里既有 context 又有 answer 的分块，前端需按 key 区分处理。
- **现代 vs 旧版**：这是 LCEL 风格的现代写法，替代已废弃的 `RetrievalQA`。新项目一律用 `create_retrieval_chain`。

## 你来改

1. 给 prompt 加一条「请在答案末尾用 `[来源: 文件名]` 标注依据」，并通过自定义 `document_prompt` 把 `metadata["source"]` 渲染进 context，验证 LLM 能否带出来源。
2. 故意问一个库里完全没有的问题，确认在「找不到就说未提及」的约束下 LLM 不编造；再去掉这条约束对比，体会 prompt 约束对反幻觉的作用。

## 面试怎么考

**Q：create_retrieval_chain 和 create_stuff_documents_chain 各负责什么？**
A：`create_stuff_documents_chain` 负责「文档组合+生成」——把一批 Document 用 stuff 策略填进 prompt 的 `{context}` 再过 LLM；`create_retrieval_chain` 在其外包一层「先检索」——用 input 调 retriever，把结果作为 context 传入前者，并在输出里同时返回 answer 和 context。

**Q：RAG 链的输出里 context 字段有什么用？**
A：它是本次检索命中的 Document 列表，用于答案溯源/引用展示和可信度评估，也是做反幻觉（让用户核对依据）和后续评测（上下文召回、忠实度）的数据基础。

**Q：「stuff」文档策略有什么局限，什么时候不能用？**
A：stuff 把所有检索文档一次性塞进上下文，简单高效但受 LLM 上下文窗口限制。当 k 很大、块很长、或要对超多文档做汇总时会溢出，此时需改用 map-reduce 或 refine 等分批处理策略，以速度换容量。
