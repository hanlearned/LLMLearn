"""
Stage 2 · RAG 评测：别再「肉眼看答得对不对」

工程师和爱好者的分水岭：能不能用数字说清「我把切分从 500 改成 300 后，到底变好还是变差」。

本文件演示一套最小但完整的评测闭环，量化两个维度：
    1. 检索质量（Retrieval）：命中率 Hit Rate —— 该召回的内容有没有被召回到？
    2. 生成质量（Generation）：忠实度 Faithfulness —— 答案是不是只基于检索内容、没瞎编？

评测的核心是「评测集」：一批 (问题, 标准答案要点) 。这里手工标注几条，真实项目里
应积累几十上百条，每次改动 RAG 都跑一遍，用分数驱动迭代。

运行：
    python stage02_rag/03_rag_eval.py
"""

import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from common.llm_provider import get_llm
from common.embeddings_provider import get_embeddings

DATA_DIR = pathlib.Path(__file__).parent / "data"

# 评测集：question = 问题；expect_keywords = 正确答案/正确文档里应出现的关键词（用于判命中）
EVAL_SET = [
    {"question": "员工每月最多远程办公几天？", "expect_keywords": ["2 天", "报备"]},
    {"question": "公积金缴存比例是多少？", "expect_keywords": ["12%"]},
    {"question": "云启智能助手专业版每月多少钱？", "expect_keywords": ["299"]},
    {"question": "病假超过 3 天的部分按多少比例发工资？", "expect_keywords": ["70%"]},
    {"question": "API 返回结果包含哪几个字段？", "expect_keywords": ["answer", "sources", "request_id"]},
]


def build_rag():
    docs = []
    for md in sorted(DATA_DIR.glob("*.md")):
        docs.extend(TextLoader(str(md), encoding="utf-8").load())
    chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(docs)
    vs = Chroma.from_documents(chunks, get_embeddings(), collection_name="eval_rag")
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "只依据【参考资料】回答，没有就说未找到，不要编造。\n\n【参考资料】\n{context}"),
            ("human", "{input}"),
        ]
    )
    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(get_llm(temperature=0), prompt))
    return retriever, chain


# ---------- 指标 1：检索命中率（程序化判定，无需 LLM）----------
def hit_rate(retriever):
    """对每个问题，检查召回的 chunk 里是否包含期望关键词。命中率 = 命中数 / 总数。"""
    hits = 0
    for case in EVAL_SET:
        ctx = " ".join(d.page_content for d in retriever.invoke(case["question"]))
        # 只要任一关键词出现即算这条「检索命中」
        if any(kw in ctx for kw in case["expect_keywords"]):
            hits += 1
        else:
            print(f"  ✗ 未召回到答案所需内容：{case['question']}")
    return hits / len(EVAL_SET)


# ---------- 指标 2：答案忠实度（LLM-as-a-Judge）----------
JUDGE_PROMPT = ChatPromptTemplate.from_template(
    "你是严格的评测员。判断【答案】是否完全基于【参考资料】，没有捏造资料中不存在的信息。\n"
    "只输出一个 1-5 的整数：5=完全忠实有据，3=部分有据，1=明显编造。只输出数字。\n\n"
    "【参考资料】\n{context}\n\n【问题】{question}\n【答案】{answer}\n\n忠实度评分（1-5）："
)


def faithfulness(chain):
    """让 LLM 给每条答案的「忠实度」打分，取平均。"""
    judge = JUDGE_PROMPT | get_llm(temperature=0) | StrOutputParser()
    scores = []
    for case in EVAL_SET:
        result = chain.invoke({"input": case["question"]})
        ctx = "\n".join(d.page_content for d in result["context"])
        raw = judge.invoke({"context": ctx, "question": case["question"], "answer": result["answer"]})
        digits = [c for c in raw if c.isdigit()]
        scores.append(int(digits[0]) if digits else 3)
    return sum(scores) / len(scores)


if __name__ == "__main__":
    retriever, chain = build_rag()
    print(f"评测集共 {len(EVAL_SET)} 条\n")

    print("▶ 指标 1：检索命中率（Hit Rate）")
    hr = hit_rate(retriever)
    print(f"  命中率 = {hr:.0%}\n")

    print("▶ 指标 2：答案忠实度（Faithfulness, LLM-as-Judge, 1-5）")
    fs = faithfulness(chain)
    print(f"  平均忠实度 = {fs:.2f} / 5\n")

    print("💡 怎么用：把 chunk_size、k、是否加重排等当作变量，每次只改一个，跑这个脚本看分数升降。")
    print("   真实项目可引入 ragas 库自动算 Context Recall / Answer Relevancy 等更细的指标。")
