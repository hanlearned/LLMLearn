"""
RAG 管道测试：加载 → 切分 → 向量库 → 检索，全程用确定性假 Embedding，无需 API Key。

验证的是「管道是否跑通」（接口、数据流），而非语义召回质量
（假向量没有真实语义，所以这里只断言「能检索到 k 条」「切分非空」这类结构性属性）。
"""

import common.sqlite_compat  # noqa: F401  # 必须在 chromadb 之前
from conftest import REPO_ROOT
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = REPO_ROOT / "stage02_rag" / "data"


def _load_chunks():
    docs = []
    for f in sorted(DATA_DIR.glob("*.md")):
        docs.extend(TextLoader(str(f), encoding="utf-8").load())
    return RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(docs)


def test_data_files_exist():
    assert list(DATA_DIR.glob("*.md")), "示例知识库数据缺失"


def test_split_produces_chunks():
    chunks = _load_chunks()
    assert len(chunks) > 5
    assert all(c.page_content for c in chunks)
    # metadata 里应带 source，供答案标注出处
    assert all("source" in c.metadata for c in chunks)


def test_chroma_build_and_retrieve(fake_embeddings):
    chunks = _load_chunks()
    vs = Chroma.from_documents(chunks, fake_embeddings, collection_name="test_rag")
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke("远程办公申请")
    assert len(docs) == 3  # 检索管道跑通，返回了 k 条
    assert all(hasattr(d, "page_content") for d in docs)
