"""
Stage 2 · 进阶检索：让「检索得准」的四种武器

最小 RAG（01）用的是最朴素的向量相似度检索。真实业务里它经常召回不全或召回噪声，
本文件演示四种最常用的增强手段，建议逐个注释/解开运行，对比召回结果的差异：

    1. MMR          —— 召回结果去冗余，兼顾相关性与多样性
    2. 混合检索      —— 向量检索 + BM25 关键词检索，互补（专有名词/编号靠 BM25）
    3. MultiQuery   —— 让 LLM 把一个问题改写成多个，扩大召回面
    4. 重排 Re-rank  —— 先粗召回一批，再用更强的模型精排，把最相关的顶上来

运行：
    python stage02_rag/02_advanced_retrieval.py
"""

import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
    MultiQueryRetriever,
)
from langchain.retrievers.document_compressors import LLMChainFilter

from common.llm_provider import get_llm
from common.embeddings_provider import get_embeddings

DATA_DIR = pathlib.Path(__file__).parent / "data"


def prepare_chunks():
    docs = []
    for md_file in sorted(DATA_DIR.glob("*.md")):
        docs.extend(TextLoader(str(md_file), encoding="utf-8").load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(docs)


def show(title, docs):
    print(f"\n----- {title}：召回 {len(docs)} 条 -----")
    for i, d in enumerate(docs, 1):
        print(f"  [{i}] {d.page_content.replace(chr(10), ' ')[:60]}…")


def main():
    chunks = prepare_chunks()
    vectorstore = Chroma.from_documents(chunks, get_embeddings(), collection_name="adv_rag")
    question = "报销超过 5000 元需要谁审批？多久到账？"

    # ---------- 0. 基线：朴素相似度检索 ----------
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    show("① 基线 similarity", base_retriever.invoke(question))

    # ---------- 1. MMR：去冗余 ----------
    # 朴素检索可能召回 3 条几乎重复的块；MMR 在「相关」和「彼此不同」之间权衡。
    # lambda_mult 越大越偏相关、越小越偏多样。
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 8, "lambda_mult": 0.5},
    )
    show("② MMR 去冗余", mmr_retriever.invoke(question))

    # ---------- 2. 混合检索：向量 + BM25 ----------
    # 向量检索懂「语义」但对精确的数字/编号/专有名词不敏感；BM25 是关键词匹配，正好互补。
    # EnsembleRetriever 用 RRF（倒数排序融合）把两路结果合并。
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 3
    hybrid = EnsembleRetriever(retrievers=[vectorstore.as_retriever(search_kwargs={"k": 3}), bm25], weights=[0.5, 0.5])
    show("③ 混合检索 (向量+BM25)", hybrid.invoke(question))

    # ---------- 3. MultiQuery：查询改写扩召回 ----------
    # 用户问法千变万化。让 LLM 把原问题改写成几个等价问法，分别检索再取并集，召回更全。
    llm = get_llm(temperature=0)
    multi = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
    show("④ MultiQuery 扩召回", multi.invoke(question))

    # ---------- 4. 重排 Re-rank：先粗召回再精排 ----------
    # 先用基线召回较多（k 调大），再用一个「压缩器」逐条判断是否真的相关，过滤掉噪声。
    # 这里用 LLMChainFilter（让 LLM 判定相关性）演示；生产中常用 BGE-Reranker / CrossEncoder。
    wide_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    compressor = LLMChainFilter.from_llm(llm)
    rerank_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=wide_retriever
    )
    show("⑤ 重排后 (LLM 过滤精排)", rerank_retriever.invoke(question))

    print("\n💡 观察：BM25 对「5000」这种精确数字更敏感；重排能滤掉粗召回里的噪声块。")


if __name__ == "__main__":
    main()
