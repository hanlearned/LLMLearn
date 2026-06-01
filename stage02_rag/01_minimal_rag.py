"""
Stage 2 · 最小可运行 RAG 管道（端到端）

这是整个 Stage 2 的「脊柱」程序。一个 RAG 系统再复杂，骨架都是这五步：

    加载(Load) → 切分(Split) → 向量化入库(Embed & Store) → 检索(Retrieve) → 生成(Generate)

跑通这一个文件，你就掌握了 RAG 的主干。后面所有高级技巧（混合检索、重排、评测）
都是在这五步的某一环上做增强。

运行：
    # 在项目根目录，先 pip install -r requirements.txt，配好 .env
    python stage02_rag/01_minimal_rag.py
"""

import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()

# 把仓库根目录加入路径，这样 `from common.xxx` 才能找到根级 common 包
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from common.llm_provider import get_llm
from common.embeddings_provider import get_embeddings

DATA_DIR = pathlib.Path(__file__).parent / "data"


# ------------------------------------------------------------------
# 第 1 步：加载（Load）—— 把磁盘上的原始文件读成 Document 对象
# ------------------------------------------------------------------
def load_documents():
    """逐个加载 data/ 下的 markdown，返回 Document 列表。

    本质：每个 Document = page_content（正文）+ metadata（来源等元数据）。
    metadata 里的 source 后面会用来给答案标注「出处」，这是 RAG 防幻觉的关键。
    """
    docs = []
    for md_file in sorted(DATA_DIR.glob("*.md")):
        loaded = TextLoader(str(md_file), encoding="utf-8").load()
        docs.extend(loaded)
    print(f"[1/5] 已加载 {len(docs)} 个文档，来源：{[d.metadata['source'].split('/')[-1] for d in docs]}")
    return docs


# ------------------------------------------------------------------
# 第 2 步：切分（Split）—— 把长文档切成适合检索的小块
# ------------------------------------------------------------------
def split_documents(docs):
    """用递归字符切分器把文档切成 chunk。

    为什么要切？① 向量检索是「块级」的，块太大噪声多、太小语义不全；
    ② LLM 上下文有限，不能把整本手册塞进 prompt。
    chunk_overlap 让相邻块有重叠，避免把一句话从中间切断导致语义丢失。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[2/5] 切分为 {len(chunks)} 个 chunk（chunk_size=300, overlap=50）")
    return chunks


# ------------------------------------------------------------------
# 第 3 步：向量化入库（Embed & Store）—— 建立可检索的向量库
# ------------------------------------------------------------------
def build_vectorstore(chunks):
    """把每个 chunk 用 Embedding 编码成向量，存入 Chroma。

    本质：Embedding 把「语义」映射到高维空间里的一个点，语义相近的文本点也相近。
    检索时把「问题」也编码成向量，找空间里最近的几个 chunk —— 这就是「语义检索」。
    """
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="minimal_rag",
        # 不传 persist_directory 即用内存库；演示用内存即可，项目里再持久化
    )
    print(f"[3/5] 已构建向量库（{len(chunks)} 个向量）")
    return vectorstore


# ------------------------------------------------------------------
# 第 4+5 步：检索 + 生成（Retrieve & Generate）—— 组装 RAG 链
# ------------------------------------------------------------------
def build_rag_chain(vectorstore):
    """用现代 LCEL 方式组装「检索 + 带上下文生成」链。

    create_stuff_documents_chain：把检索到的文档「塞进」prompt 的 {context} 占位符。
    create_retrieval_chain：先用 retriever 检索，再把结果喂给上面的文档链。
    最终输入 {"input": 问题}，输出 {"answer": 答案, "context": 检索到的文档列表}。
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 关键：prompt 里强制「只依据 context 回答，并要求标注出处」——这是工程上防幻觉的核心手段
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是云启科技的内部知识助手。只能依据下面提供的【参考资料】回答用户问题。\n"
                "如果参考资料中没有相关信息，必须明确回答「未在知识库中找到相关内容」，绝不能编造。\n"
                "回答要简洁准确。\n\n"
                "【参考资料】\n{context}",
            ),
            ("human", "{input}"),
        ]
    )

    llm = get_llm(temperature=0)  # 知识问答要稳定、可复现，温度设为 0
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    print("[4/5] RAG 链已组装（retriever k=3 + stuff 文档链）")
    return rag_chain


def ask(rag_chain, question: str):
    """提问并打印答案 + 出处。"""
    print("\n" + "=" * 60)
    print(f"❓ 问题：{question}")
    result = rag_chain.invoke({"input": question})
    print(f"\n🤖 回答：{result['answer']}")
    # 打印命中的来源片段，体现「答案有据可查」
    print("\n📎 参考来源：")
    for i, doc in enumerate(result["context"], 1):
        source = doc.metadata.get("source", "?").split("/")[-1]
        snippet = doc.page_content.replace("\n", " ")[:50]
        print(f"   [{i}] {source}：{snippet}…")


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)
    rag_chain = build_rag_chain(vectorstore)
    print("[5/5] 就绪，开始问答\n")

    # 三个问题：前两个能在知识库里找到，第三个故意问不存在的，验证「不编造」
    ask(rag_chain, "员工每月最多可以远程办公几天？需要怎么申请？")
    ask(rag_chain, "云启智能助手专业版多少钱？支持上传多少个文档？")
    ask(rag_chain, "公司给员工配股票期权吗？")  # 知识库里没有 → 应回答「未找到」
