"""
项目 5 · 多 Agent 内容创作工作流

业务场景：自动化产出一篇合格的公众号短文。单个模型一把梭质量不稳；
模拟人类编辑部的「分工 + 审校 + 返工」流程，质量明显更可控。

角色（每个是一个图节点 / 子 Agent）：
    规划员 planner   → 定大纲
    写手   writer    → 按大纲写正文（可带编辑意见返工）
    编辑   editor    → 打分 + 给修改意见（质量门控）
    定稿员 finalizer → 配标题、润色成品

流程：planner → writer → editor →（不合格且未超次数 → 回 writer）→（合格 → finalizer → END）

这是 Stage 4 所有技能的综合：StateGraph + 多节点 + 条件边循环 + 质量门控。

运行：
    pip install langgraph
    python stage04_langgraph/project05_content_workflow/workflow.py
"""

import pathlib
import sys
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from langgraph.graph import StateGraph, START, END

from common.llm_provider import get_llm

writer_llm = get_llm(temperature=0.7)   # 写作要点创意 → 温度高
editor_llm = get_llm(temperature=0)     # 评审要稳定 → 温度 0

MAX_REVISIONS = 2


class State(TypedDict):
    topic: str
    outline: str
    draft: str
    notes: str
    score: int
    revisions: int
    final: str


def planner_node(state: State) -> dict:
    outline = writer_llm.invoke(f"为主题《{state['topic']}》列一个 3 点的公众号短文大纲，只返回大纲。").content
    print(f"📋 [规划员] 大纲已定")
    return {"outline": outline, "revisions": 0}


def writer_node(state: State) -> dict:
    prompt = f"根据大纲写一篇 200 字左右的公众号短文。\n主题：{state['topic']}\n大纲：{state['outline']}"
    if state.get("notes"):
        prompt += f"\n\n编辑上一轮的修改意见（务必改进）：{state['notes']}\n上一稿：{state['draft']}"
    draft = writer_llm.invoke(prompt).content
    revisions = state.get("revisions", 0) + (1 if state.get("notes") else 0)
    print(f"✍️  [写手] 第 {revisions + 1} 稿完成")
    return {"draft": draft, "revisions": revisions}


def editor_node(state: State) -> dict:
    resp = editor_llm.invoke(
        f"你是严格的编辑。给下面短文打分(1-10)并给一条具体修改意见，格式严格『分数|意见』：\n{state['draft']}"
    ).content
    try:
        s, notes = resp.split("|", 1)
        score = int("".join(c for c in s if c.isdigit()) or "6")
    except ValueError:
        score, notes = 6, resp
    print(f"🔍 [编辑] 评分 {score}/10")
    return {"score": score, "notes": notes.strip()}


def finalizer_node(state: State) -> dict:
    final = writer_llm.invoke(
        f"给下面短文起一个吸睛标题并整体润色，输出『标题 + 正文』：\n{state['draft']}"
    ).content
    print(f"✅ [定稿员] 成品完成")
    return {"final": final}


def gate(state: State) -> str:
    """质量门控：达标或用尽返工次数 → 定稿；否则打回写手。"""
    if state["score"] >= 8 or state["revisions"] >= MAX_REVISIONS:
        return "finalize"
    print(f"   ↩️ 不达标（{state['score']}分），打回重写")
    return "revise"


def build_app():
    g = StateGraph(State)
    g.add_node("planner", planner_node)
    g.add_node("writer", writer_node)
    g.add_node("editor", editor_node)
    g.add_node("finalizer", finalizer_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "writer")
    g.add_edge("writer", "editor")
    g.add_conditional_edges("editor", gate, {"revise": "writer", "finalize": "finalizer"})
    g.add_edge("finalizer", END)
    return g.compile()


if __name__ == "__main__":
    app = build_app()
    result = app.invoke({"topic": "为什么每个开发者都该学一点 RAG"})
    print("\n" + "=" * 60)
    print(result["final"])
