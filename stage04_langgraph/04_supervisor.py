"""
Stage 4 · Supervisor 多 Agent 协作模式

单个 Agent 什么都想干，往往什么都干不好。更好的架构是「分工」：
一个**主管（Supervisor）**负责调度，几个**专家 Agent** 各管一摊，主管决定每一步派给谁。

这是多 Agent 系统最主流的模式（对比「群聊自由 handoff」，Supervisor 更可控、更易调试）。

本例：主管协调两个专家完成「调研并写一段介绍」：
    Supervisor 看当前进度 → 决定派给 researcher（查资料）还是 writer（成文）还是 FINISH
    每个专家干完活把结果写回共享状态，控制权交回 Supervisor，循环直到 FINISH。

运行：
    python stage04_langgraph/04_supervisor.py
"""

import pathlib
import sys
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from common.llm_provider import get_llm

llm = get_llm(temperature=0)


# 共享状态：messages 用 add_messages reducer 自动累加（多 Agent 共享同一条消息流）
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next: str


WORKERS = ["researcher", "writer"]


# 主管用结构化输出做路由决策，避免解析自由文本
class Route(BaseModel):
    next: Literal["researcher", "writer", "FINISH"] = Field(description="下一步派给谁，或 FINISH 结束")


def supervisor_node(state: State) -> dict:
    """主管：根据对话进度决定下一步交给哪个专家，或结束。"""
    system = (
        "你是项目主管，协调以下专家完成用户任务：\n"
        "- researcher：负责查资料、给出要点\n"
        "- writer：负责把要点写成通顺的成品段落\n"
        "规则：通常先 researcher 后 writer；当已经有了令人满意的成品段落，就回 FINISH。"
    )
    decision = llm.with_structured_output(Route).invoke(
        [HumanMessage(system)] + state["messages"]
    )
    print(f"  [supervisor] 决定 → {decision.next}")
    return {"next": decision.next}


def researcher_node(state: State) -> dict:
    """调研专家。"""
    resp = llm.invoke([HumanMessage("作为调研员，针对以下任务列出 3 个关键要点：\n" +
                                    state["messages"][0].content)])
    print(f"  [researcher] 产出要点")
    return {"messages": [AIMessage(content="调研要点：\n" + resp.content, name="researcher")]}


def writer_node(state: State) -> dict:
    """写作专家：基于已有消息（含调研要点）成文。"""
    resp = llm.invoke([HumanMessage("作为写手，基于下面的对话写一段 100 字左右的通顺介绍：\n" +
                                    "\n".join(m.content for m in state["messages"]))])
    print(f"  [writer] 产出成品")
    return {"messages": [AIMessage(content="成品：\n" + resp.content, name="writer")]}


def route(state: State) -> str:
    """把主管的决定映射到目标节点。"""
    return state["next"]


def build_app():
    graph = StateGraph(State)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "supervisor")
    # 主管 → 各专家 / 结束（条件边）
    graph.add_conditional_edges("supervisor", route,
                                {"researcher": "researcher", "writer": "writer", "FINISH": END})
    # 专家干完 → 交回主管（这就是「控制权回归」）
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    return graph.compile()


if __name__ == "__main__":
    app = build_app()
    print("图结构：")
    print(app.get_graph().draw_ascii())
    print("\n开始执行：")
    # recursion_limit 兜底，防止主管来回踢皮球死循环
    result = app.invoke(
        {"messages": [HumanMessage("介绍一下 RAG（检索增强生成）技术")], "next": ""},
        config={"recursion_limit": 15},
    )
    print("\n最终成品：")
    print(result["messages"][-1].content)
