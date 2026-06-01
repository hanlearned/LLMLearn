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

    # 幂等重建：Chroma.from_documents 会「追加」到已有集合，所以重复跑 ingest 会让向量翻倍。
    # 先删掉旧的持久化目录，保证每次 ingest 都是干净重建，结果可复现。
    import os
    import shutil

    if os.path.isdir(config.PERSIST_DIR):
        shutil.rmtree(config.PERSIST_DIR)
        print(f"已清理旧向量库：{config.PERSIST_DIR}")

    # persist_directory 一传，Chroma 就会把向量落盘，下次可直接加载
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.PERSIST_DIR,
    )
    print(f"✅ 向量库已持久化到 {config.PERSIST_DIR}（{len(chunks)} 个向量）")


if __name__ == "__main__":
    main()
