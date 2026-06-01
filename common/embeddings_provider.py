"""
Embedding 模型统一封装

RAG 的第一性原理：检索质量 = Embedding 把语义编码成向量的质量。
所以「选对 Embedding」比「选对 LLM」对 RAG 效果影响更大。

本封装的选择策略（按优先级）
---------------------------
1. SiliconFlow API：提供 BAAI/bge-large-zh-v1.5（中文检索一线效果），OpenAI 兼容协议，
   不用在本地下几个 G 的模型，最适合学习阶段。
2. OpenAI API：text-embedding-3-small，英文/通用强。
3. 本地 HuggingFace：离线、免费，但首次会下载模型（需要 sentence-transformers）。

用法
----
    from common.embeddings_provider import get_embeddings
    emb = get_embeddings()
    vec = emb.embed_query("你好")   # -> List[float]
"""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=None)
def get_embeddings(model: str | None = None):
    """返回一个 LangChain Embeddings 对象（已按可用 Key 自动选型）。"""
    sf_key = os.getenv("SILICONFLOW_API_KEY")
    oa_key = os.getenv("OPENAI_API_KEY")

    if sf_key and not sf_key.startswith("your_"):
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model or "BAAI/bge-large-zh-v1.5",
            api_key=sf_key,
            base_url="https://api.siliconflow.cn/v1",
            # bge 系列向量维度由模型决定，check_embedding_ctx_length 关掉避免 tiktoken 报错
            check_embedding_ctx_length=False,
        )

    if oa_key and not oa_key.startswith("your_"):
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model or "text-embedding-3-small", api_key=oa_key)

    # 兜底：本地模型（需要 pip install sentence-transformers）
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model or "BAAI/bge-small-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    emb = get_embeddings()
    v = emb.embed_query("LangChain 是一个 LLM 应用开发框架")
    print(f"Embedding 维度：{len(v)}，前 5 维：{v[:5]}")
