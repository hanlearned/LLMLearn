"""
项目 2 · 第一步：构建知识库（ingest / 灌库）

把 data/ 下的文档加载、切分、向量化，持久化到磁盘上的 Chroma。
这一步只需在「文档有更新」时跑一次，问答服务启动时直接加载已建好的库即可——
这就是「离线建库、在线检索」的工程划分。

运行：
    python stage02_rag/project02_knowledge_base/ingest.py
"""

from dotenv import load_dotenv

load_dotenv()

import config  # noqa: E402  （config 内部已把仓库根加入 sys.path）
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import common.sqlite_compat  # noqa: F401  # 老系统 sqlite 兼容，须在 chromadb 之前
from langchain_chroma import Chroma

from common.embeddings_provider import get_embeddings


def load_and_split():
    docs = []
    for f in sorted(config.DATA_DIR.glob("*.md")):
        loaded = TextLoader(str(f), encoding="utf-8").load()
        docs.extend(loaded)
    print(f"加载 {len(docs)} 个文档")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"切分为 {len(chunks)} 个 chunk")
    return chunks


def main():
    chunks = load_and_split()
    # persist_directory 一传，Chroma 就会把向量落盘，下次可直接加载
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.PERSIST_DIR,
    )
    print(f"✅ 向量库已持久化到 {config.PERSIST_DIR}")


if __name__ == "__main__":
    main()
