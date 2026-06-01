"""
项目 3 · GraphRAG 原型：当「关系型问题」难倒普通 RAG 时

普通向量 RAG 的软肋：它检索的是「语义相似的片段」，但回答不了需要"串联多跳关系"的问题。
比如「和报销审批相关的角色，分别还管哪些事？」——答案散落在多个片段里，靠相似度凑不齐。

GraphRAG 的思路：先把文档抽成一张「知识图谱」（实体 + 关系），提问时定位到相关实体，
再沿着图的边把关联信息「捞」出来，一起喂给 LLM。

本文件是一个**零额外依赖、可运行**的最小 GraphRAG 原型，分三步：
    1. 抽取（Extract）：用 LLM 从每个 chunk 里抽出 (主体, 关系, 客体) 三元组
    2. 建图（Build）  ：把三元组组装成邻接表形式的图
    3. 问答（Query）  ：从问题里找实体 → 取其邻居子图 → 喂给 LLM 生成答案

运行：
    python stage02_rag/project03_graphrag/graphrag.py
"""

import pathlib
import sys
from collections import defaultdict
from typing import List

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from common.llm_provider import get_llm

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"


# ---------- 用结构化输出约束 LLM 抽取三元组 ----------
class Triple(BaseModel):
    subject: str = Field(description="主体实体，如「员工」「专业版」")
    relation: str = Field(description="关系/谓词，如「享有」「需要审批」")
    object: str = Field(description="客体实体或值，如「5天年假」「部门负责人」")


class TripleList(BaseModel):
    triples: List[Triple] = Field(description="从文本中抽取的所有三元组")


EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是知识图谱抽取器。从给定文本中抽取关键的(主体,关系,客体)三元组。\n"
            "只抽取明确、有信息量的关系，实体名称要规范简洁。",
        ),
        ("human", "文本：\n{text}"),
    ]
)


def extract_triples(chunks) -> List[Triple]:
    """对每个 chunk 调用 LLM 抽三元组。with_structured_output 保证返回的是合法对象。"""
    extractor = EXTRACT_PROMPT | get_llm(temperature=0).with_structured_output(TripleList)
    all_triples: List[Triple] = []
    for i, chunk in enumerate(chunks, 1):
        try:
            result = extractor.invoke({"text": chunk.page_content})
            all_triples.extend(result.triples)
            print(f"  抽取 chunk {i}/{len(chunks)}：+{len(result.triples)} 三元组")
        except Exception as e:  # 抽取偶尔失败不应中断全流程
            print(f"  chunk {i} 抽取失败：{e}")
    return all_triples


# ---------- 建图：邻接表（无需 networkx）----------
def build_graph(triples: List[Triple]):
    """graph[实体] = [(关系, 邻居), ...]，无向存储以便双向检索。"""
    graph = defaultdict(list)
    for t in triples:
        graph[t.subject].append((t.relation, t.object))
        graph[t.object].append((f"被{t.relation}", t.subject))
    return graph


def find_entities(graph, question: str) -> List[str]:
    """最朴素的实体定位：图中哪些实体名出现在问题里就算命中（生产里可用 LLM/NER）。"""
    return [ent for ent in graph if ent and ent in question]


def gather_subgraph(graph, entities: List[str]) -> str:
    """把命中实体的邻居关系拼成文本（一跳邻居），作为 LLM 的上下文。"""
    lines = []
    for ent in entities:
        for relation, neighbor in graph.get(ent, []):
            lines.append(f"{ent} —[{relation}]→ {neighbor}")
    return "\n".join(lines) if lines else "（图谱中未找到相关实体）"


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "依据下面从知识图谱中检索到的关系事实回答问题，没有就说未找到。\n\n【图谱事实】\n{facts}"),
        ("human", "{question}"),
    ]
)


def main():
    # 1. 加载 + 切分
    docs = []
    for f in sorted(DATA_DIR.glob("*.md")):
        docs.extend(TextLoader(str(f), encoding="utf-8").load())
    chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40).split_documents(docs)
    print(f"[1/3] 加载并切分为 {len(chunks)} 个 chunk\n")

    # 2. 抽取 + 建图
    print("[2/3] 用 LLM 抽取三元组并建图…")
    triples = extract_triples(chunks)
    graph = build_graph(triples)
    print(f"\n图谱构建完成：{len(graph)} 个实体，{len(triples)} 条关系\n")

    # 3. 基于图的问答
    print("[3/3] 图检索问答：")
    answer_chain = ANSWER_PROMPT | get_llm(temperature=0)
    for q in ["年假和员工是什么关系？", "报销和审批之间有哪些关系？"]:
        ents = find_entities(graph, q)
        facts = gather_subgraph(graph, ents)
        resp = answer_chain.invoke({"facts": facts, "question": q})
        print(f"\n❓ {q}\n  命中实体：{ents}\n  🤖 {resp.content}")


if __name__ == "__main__":
    main()
