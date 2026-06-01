"""
Stage 4 · 持久化 + 人工介入（Human-in-the-Loop）

两个生产级 Agent 的刚需：
    1. 持久化（checkpointer）：把每一步的状态存下来，会话能断点续跑、能回溯。
    2. 人工介入（HITL）：在「发邮件 / 转账 / 删数据」这类高风险操作前**暂停**，等人点头再继续。

LangGraph 用同一套机制实现：checkpointer 存状态 + interrupt 在指定节点前中断。
中断后整张图的状态被冻结保存，人工确认后用同一个 thread_id「恢复」即可。

本例：起草一封邮件 → 在「发送」前中断 → 人工审批 → 恢复执行真正发送。

运行：
    python stage04_langgraph/03_persistence_hitl.py
"""

import pathlib
import sys
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from common.llm_provider import get_llm

llm = get_llm(temperature=0.5)


class State(TypedDict):
    instruction: str
    draft: str
    sent: bool


def draft_node(state: State) -> dict:
    """起草邮件。"""
    content = llm.invoke(f"根据要求写一封简短中文邮件，只返回正文：\n{state['instruction']}").content
    print(f"  [draft] 已起草：\n{content}\n")
    return {"draft": content}


def send_node(state: State) -> dict:
    """真正发送（高风险操作）。会在它之前被 interrupt 拦下。"""
    print(f"  [send] 📧 邮件已发送！内容：{state['draft'][:30]}…")
    return {"sent": True}


def build_app():
    graph = StateGraph(State)
    graph.add_node("draft", draft_node)
    graph.add_node("send", send_node)
    graph.add_edge(START, "draft")
    graph.add_edge("draft", "send")
    graph.add_edge("send", END)

    memory = MemorySaver()
    # interrupt_before：在执行 send 节点「之前」中断，把决定权交还给人
    return graph.compile(checkpointer=memory, interrupt_before=["send"])


def main():
    app = build_app()
    config = {"configurable": {"thread_id": "mail-001"}}  # thread_id 标识这次会话

    # 第一段：执行到 send 之前会自动停下
    print("=== 阶段一：执行到高风险节点前自动暂停 ===")
    app.invoke({"instruction": "向团队通知明天上午10点开周会"}, config=config)

    # 查看当前被冻结的状态：下一步要执行 send
    snapshot = app.get_state(config)
    print(f"⏸️ 已暂停。等待执行的下一个节点：{snapshot.next}")
    print(f"   待发送草稿：{snapshot.values['draft'][:40]}…\n")

    # 第二段：模拟人工审批
    print("=== 阶段二：人工审批 ===")
    approved = True  # 改成 False 体验「人工否决」
    if approved:
        print("✅ 人工已批准，恢复执行……")
        # 传 None 表示「不给新输入，从断点继续往下跑」
        app.invoke(None, config=config)
    else:
        print("❌ 人工否决，流程终止，邮件不会发送。")

    final = app.get_state(config)
    print(f"\n最终 sent 状态：{final.values.get('sent', False)}")


if __name__ == "__main__":
    main()
    print("\n💡 同一个 thread_id 的状态被 checkpointer 完整保存，所以能『暂停→换个时间→恢复』。")
    print("   把 MemorySaver 换成 SqliteSaver，状态就能跨进程重启持久化。")
