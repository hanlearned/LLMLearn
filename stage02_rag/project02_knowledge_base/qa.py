"""
项目 2 · 第二步：问答引擎（在线检索 + 生成）

加载 ingest.py 建好的持久化向量库，组装带「出处引用」的 RAG 链。
这个模块同时被 CLI（本文件 __main__）和 API（api.py）复用 —— 业务逻辑只写一遍。

直接命令行问答：
    python stage02_rag/project02_knowledge_base/qa.py
"""

from dotenv import load_dotenv

load_dotenv()

import config  # noqa: E402
import common.sqlite_compat  # noqa: F401  # 老系统 sqlite 兼容，须在 chromadb 之前
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from common.embeddings_provider import get_embeddings
from common.llm_provider import get_llm

SYSTEM_PROMPT = (
    "你是云启科技的企业知识库助手。严格遵守以下规则：\n"
    "1. 只能依据下面【参考资料】中的内容回答，不得使用任何外部知识。\n"
    "2. 若参考资料不足以回答，明确说「未在知识库中找到相关内容，建议联系人工」。\n"
    "3. 回答末尾用一行列出你引用了第几条资料，格式：（参考：资料1、资料3）。\n\n"
    "【参考资料】\n{context}"
)


def get_vectorstore():
    """加载已持久化的向量库（不重新建库）。"""
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=config.PERSIST_DIR,
    )


def build_chain():
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": config.TOP_K})
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{input}")])
    document_chain = create_stuff_documents_chain(get_llm(temperature=0), prompt)
    return create_retrieval_chain(retriever, document_chain)


def answer(chain, question: str) -> dict:
    """返回结构化结果：answer + sources（出处），供 API 直接序列化。"""
    result = chain.invoke({"input": question})
    sources = []
    for doc in result["context"]:
        sources.append(
            {
                "source": doc.metadata.get("source", "?").split("/")[-1],
                "snippet": doc.page_content.replace("\n", " ")[:80],
            }
        )
    return {"answer": result["answer"], "sources": sources}


def _cli():
    chain = build_chain()
    print("企业知识库助手已就绪（输入 q 退出）")
    while True:
        q = input("\n你的问题：").strip()
        if q.lower() in {"q", "quit", "exit"}:
            break
        if not q:
            continue
        res = answer(chain, q)
        print(f"\n🤖 {res['answer']}")
        print("📎 出处：")
        for i, s in enumerate(res["sources"], 1):
            print(f"   [{i}] {s['source']}：{s['snippet']}…")


if __name__ == "__main__":
    _cli()
