"""
高级实战项目 · 深度研究 Agent（Deep Research）

一句话定位
----------
给定一个研究问题，自动完成「规划子问题 → 联网检索 → 综合成报告」三段式流程，
最终产出一篇结构化、带引用来源的中文综述。

为什么用显式 StateGraph 而不是单个 create_react_agent？
------------------------------------------------------
- create_react_agent 是「一个模型 + 一堆工具自循环」，由模型自己决定下一步做什么，
  过程不可控、难复盘，也很难在「检索完成后强制综述」这种确定性流程上稳定收敛。
- 深度研究本质是一条有明确阶段的流水线（规划→检索→综述），用 LangGraph 把每个阶段
  画成一个节点，状态在节点间显式传递，流程可读、可测、可观测，也方便后续插入
  「人工审核」「质量评分」等节点。这正是「编排（Orchestration）」相对「自由 Agent」
  的工程优势。

三个节点
--------
1. plan       ：把研究问题拆成 3 个互补的子问题（LLM 结构化输出列表）。
2. research   ：对每个子问题做 DuckDuckGo 搜索，把搜索结果交给 LLM 提炼成
                带要点、带来源链接的「发现」。网络失败时自动降级为「模型已有知识」。
3. synthesize ：综合所有发现，写一篇结构化、带引用来源的中文综述报告。

连接：START → plan → research → synthesize → END

运行
----
    # 需要联网（DuckDuckGo 搜索）。无网时会自动降级，不会崩溃。
    python projects_advanced/research_agent/research_agent.py
"""

from __future__ import annotations

import pathlib
import sys
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()
# 本文件位于 projects_advanced/research_agent/research_agent.py，距仓库根 2 层。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field

from common.llm_provider import get_llm

# 研究类任务希望稳定、少幻觉，温度调低。
llm = get_llm(temperature=0.2)


# ==================================================================
# 0. 联网搜索工具：keyless 的 DuckDuckGo
#    关键：所有搜索调用都包在 try/except 里，网络/依赖出问题时降级，保证无网也不崩。
# ==================================================================
def web_search(query: str, max_results: int = 4) -> list[dict]:
    """
    用 DuckDuckGo 搜索，返回 [{title, link, snippet}, ...]。

    失败（无网络、依赖缺失、被限流等）时返回空列表，由调用方决定如何降级。
    """
    try:
        # 延迟导入：即使环境没装 duckduckgo-search，也只在真正搜索时才报错。
        from langchain_community.tools import DuckDuckGoSearchResults

        # output_format="list" 让工具直接返回结构化列表，省去自己解析字符串。
        tool = DuckDuckGoSearchResults(num_results=max_results, output_format="list")
        results = tool.invoke(query)
        # 不同版本字段名略有差异，做一次兼容归一化。
        normalized: list[dict] = []
        for item in results or []:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "title": item.get("title", "") or "",
                    "link": item.get("link", "") or item.get("href", "") or "",
                    "snippet": item.get("snippet", "") or item.get("body", "") or "",
                }
            )
        return normalized
    except Exception as exc:  # noqa: BLE001 —— 故意宽捕获，搜索失败必须降级而非中断
        print(f"  [搜索降级] DuckDuckGo 搜索失败（{type(exc).__name__}: {exc}），改用模型已有知识兜底。")
        return []


# ==================================================================
# 1. 状态定义：整张图共享的数据结构
# ==================================================================
class ResearchState(TypedDict):
    question: str            # 用户的研究问题
    sub_queries: list[str]   # plan 节点拆出的子问题列表
    findings: list[dict]     # research 节点产出的发现：[{query, content, sources}]
    report: str              # synthesize 节点产出的最终综述


# 用 Pydantic 约束 plan 节点的结构化输出，避免模型自由发挥导致解析失败。
class SubQueries(BaseModel):
    """规划阶段的结构化输出：恰好 3 个子问题。"""

    queries: list[str] = Field(
        description="3 个互补、可独立检索的子问题，覆盖原问题的不同侧面",
        min_length=3,
        max_length=3,
    )


# ==================================================================
# 2. 节点：plan —— 把研究问题拆成 3 个子问题
# ==================================================================
def plan_node(state: ResearchState) -> dict:
    """把宽泛的研究问题拆解成 3 个互补、可独立检索的子问题。"""
    print(f"\n[1/3 规划 plan] 研究问题：{state['question']}")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名严谨的研究规划助手。请把用户的研究问题拆成恰好 3 个"
                "互补、可独立联网检索的子问题，覆盖原问题的不同侧面（如：进展/方法、"
                "代表性工作/案例、挑战与趋势）。子问题应具体、便于搜索。",
            ),
            ("human", "研究问题：{question}"),
        ]
    )
    # with_structured_output 让模型按 SubQueries 模式返回，直接拿到 Python 对象。
    planner = prompt | llm.with_structured_output(SubQueries)

    try:
        result: SubQueries = planner.invoke({"question": state["question"]})
        sub_queries = result.queries
    except Exception as exc:  # noqa: BLE001 —— 结构化输出失败时退化为「原问题本身」
        print(f"  [规划降级] 结构化拆解失败（{exc}），退化为单一子问题。")
        sub_queries = [state["question"]]

    for i, q in enumerate(sub_queries, 1):
        print(f"  子问题 {i}：{q}")
    return {"sub_queries": sub_queries}


# ==================================================================
# 3. 节点：research —— 对每个子问题检索 + 提炼成「发现」
# ==================================================================
# 提炼链：把搜索到的原始材料压成带要点、带来源的发现。
_distill_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名研究分析师。下面给出针对某个子问题检索到的网络资料。"
            "请基于资料提炼出 3-5 条要点，每条简洁、客观。"
            "严格只使用资料中出现的信息，不要编造；若资料不足，请明确说明"
            "「资料有限」。要点之后另起一行列出你引用到的来源链接。",
        ),
        (
            "human",
            "子问题：{query}\n\n检索到的资料：\n{material}",
        ),
    ]
)
_distill_chain = _distill_prompt | llm | StrOutputParser()

# 无网兜底链：没有搜索结果时，用模型已有知识作答并显式标注「未经检索」。
_fallback_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "联网检索不可用。请基于你已有的知识，对下面的子问题给出 3-5 条要点。"
            "务必在开头标注「（以下内容未经实时检索，可能过时，请谨慎核对）」，"
            "不确定的地方要说明不确定。",
        ),
        ("human", "子问题：{query}"),
    ]
)
_fallback_chain = _fallback_prompt | llm | StrOutputParser()


def research_node(state: ResearchState) -> dict:
    """对每个子问题：先搜索，再让 LLM 提炼。搜不到就走模型知识兜底。"""
    print("\n[2/3 检索 research] 逐个子问题检索并提炼……")
    findings: list[dict] = []

    for i, query in enumerate(state["sub_queries"], 1):
        print(f"\n  -> 子问题 {i}：{query}")
        results = web_search(query)

        if results:
            # 把搜索结果拼成给模型的材料，同时记录来源以便综述引用。
            material_lines = []
            sources: list[str] = []
            for r in results:
                material_lines.append(
                    f"- 标题：{r['title']}\n  链接：{r['link']}\n  摘要：{r['snippet']}"
                )
                if r["link"]:
                    sources.append(r["link"])
            material = "\n".join(material_lines)
            print(f"     命中 {len(results)} 条结果，提炼中……")
            content = _distill_chain.invoke({"query": query, "material": material})
        else:
            # 降级：无搜索结果时用模型已有知识，并把来源标注为「无（模型知识）」。
            sources = []
            content = _fallback_chain.invoke({"query": query})

        findings.append({"query": query, "content": content, "sources": sources})
        print(f"     发现已生成（来源 {len(sources)} 条）。")

    return {"findings": findings}


# ==================================================================
# 4. 节点：synthesize —— 综合所有发现，写成带引用的综述
# ==================================================================
def synthesize_node(state: ResearchState) -> dict:
    """把各子问题的发现汇总成一篇结构化、带引用来源的中文综述报告。"""
    print("\n[3/3 综述 synthesize] 汇总所有发现，撰写综述……")

    # 把所有发现拼成上下文，并给每条发现编号，便于报告中引用。
    blocks = []
    all_sources: list[str] = []
    for i, f in enumerate(state["findings"], 1):
        src_text = "\n".join(f["sources"]) if f["sources"] else "（无，基于模型已有知识）"
        blocks.append(
            f"### 发现 {i}：{f['query']}\n{f['content']}\n来源：\n{src_text}"
        )
        all_sources.extend(f["sources"])
    findings_text = "\n\n".join(blocks)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名资深研究员。请基于下方各子问题的「发现」，撰写一篇结构化的"
                "中文综述报告，包含：1) 概述；2) 分主题展开（每个主题对应一个子问题）；"
                "3) 总结与展望；4) 文末「参考来源」列出所有去重后的链接。"
                "要求：只依据给定发现，不要编造事实；在正文相应处用方括号标注来源链接"
                "以便溯源；若某部分依据有限，请如实说明。",
            ),
            (
                "human",
                "原始研究问题：{question}\n\n各子问题的发现如下：\n\n{findings}",
            ),
        ]
    )
    report = (prompt | llm | StrOutputParser()).invoke(
        {"question": state["question"], "findings": findings_text}
    )
    return {"report": report}


# ==================================================================
# 5. 组装图
# ==================================================================
def build_graph():
    """构建 START → plan → research → synthesize → END 的研究流水线。"""
    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    print("研究流水线结构：")
    print(app.get_graph().draw_ascii())

    question = "2024年 RAG（检索增强生成）技术有哪些主要进展？"
    print(f"\n{'=' * 60}\n开始深度研究：{question}\n{'=' * 60}")

    result = app.invoke({"question": question})

    print(f"\n{'=' * 60}\n最终综述报告\n{'=' * 60}\n")
    print(result["report"])
