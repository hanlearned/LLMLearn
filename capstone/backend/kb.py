"""客服政策知识库：把 data/ 下的政策文档建成 RAG 检索器（懒加载、进程内单例）。"""

import config
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import common.sqlite_compat  # noqa: F401  # 老系统 sqlite 兼容，须在 chromadb 之前
from langchain_chroma import Chroma

from common.embeddings_provider import get_embeddings

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        docs = []
        for f in sorted(config.DATA_DIR.glob("*.md")):
            docs.extend(TextLoader(str(f), encoding="utf-8").load())
        chunks = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40).split_documents(docs)
        vs = Chroma.from_documents(chunks, get_embeddings(), collection_name="capstone_cs_kb")
        _retriever = vs.as_retriever(search_kwargs={"k": config.TOP_K})
    return _retriever
