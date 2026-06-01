"""
Stage 4 · 条件边与「自我修正循环」

固定顺序的流水线（01）不够用，真实流程要会**判断**和**回头重来**：
    写一版 → 自评打分 → 不合格就改写再评 → 合格才放行。

这正是 LangGraph 最有价值的能力：用 `add_conditional_edges` 实现分支与循环。
Agent 的「自我修正」（self-correction）本质就是一条「回到自己」的条件边。

本例：写一句产品宣传语 → LLM 自评分（1-10）→ 不到 8 分就带着意见重写，最多 3 轮。

运行：
    python stage04_langgraph/02_conditional_routing.py
"""

import pathlib
import sys
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langgraph.graph import StateGraph, START, END

from common.llm_provider import get_llm

llm = get_llm(temperature=0.7)
judge = get_llm(temperature=0)


class State(TypedDict):
    topic: str
    draft: str
    score: int
    feedback: str
    attempts: int


def write_node(state: State) -> dict:
    """写作节点。如果有上一轮的反馈，就带着反馈改写。"""
    fb = state.get("feedback", "")
    instruction = f"为「{state['topic']}」写一句广告语。"
    if fb:
        instruction += f"\n上一版不够好，请根据意见改进：{fb}\n上一版：{state['draft']}"
    draft = llm.invoke(instruction).content
    attempts = state.get("attempts", 0) + 1
    print(f"  [write] 第{attempts}稿：{draft}")
    return {"draft": draft, "attempts": attempts}


def review_node(state: State) -> dict:
    """评审节点：给草稿打分并给出改进意见。"""
    resp = judge.invoke(
        f"给这句广告语打分(1-10)并给一句改进意见，格式严格为『分数|意见』：\n{state['draft']}"
    )
    raw = resp.content
    try:
        score_str, feedback = raw.split("|", 1)
        score = int("".join(c for c in score_str if c.isdigit()) or "5")
    except ValueError:
        score, feedback = 5, raw
    print(f"  [review] 评分：{score}，意见：{feedback.strip()}")
    return {"score": score, "feedback": feedback.strip()}


# ------------------------------------------------------------------
# 路由函数：返回的字符串决定走哪条边。这是条件边的核心。
# ------------------------------------------------------------------
def should_continue(state: State) -> str:
    """达标或用尽次数 → 结束；否则 → 回到 write 重写。"""
    if state["score"] >= 8:
        print("  → 达标，放行")
        return "accept"
    if state["attempts"] >= 3:
        print("  → 已重写 3 次，止损放行当前稿")
        return "accept"
    print("  → 不达标，打回重写")
    return "revise"


def build_graph():
    graph = StateGraph(State)
    graph.add_node("write", write_node)
    graph.add_node("review", review_node)

    graph.add_edge(START, "write")
    graph.add_edge("write", "review")
    # 条件边：从 review 出发，按 should_continue 的返回值映射到目标节点
    graph.add_conditional_edges(
        "review",
        should_continue,
        {"accept": END, "revise": "write"},  # "revise" 指回 write，形成循环
    )
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    print("图结构：")
    print(app.get_graph().draw_ascii())
    print("\n开始执行：")
    result = app.invoke({"topic": "云启智能助手企业版"})
    print(f"\n最终采用（第{result['attempts']}稿，{result['score']}分）：{result['draft']}")
