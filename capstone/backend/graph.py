"""
Capstone 核心 · 多 Agent 客服编排图（LangGraph）

把所有能力编织成一张有状态的图，体现「主管路由 + 专家分工 + 人工审批 + 持久记忆」：

                         ┌──→ policy_node   (RAG 查政策) ──┐
   START → triage_node ──┼──→ order_node    (查订单)      ──┼──→ END
   (主管/意图路由)        ├──→ other_node    (闲聊)        ──┘
                         └──→ prepare_refund → [⏸ 人工审批] → execute_refund → END
                                                (退款是高风险动作，HITL 拦截)

- 主管(triage)用结构化输出决定把请求派给哪个专家节点 —— 这就是 Supervisor 模式。
- 退款走「准备 → 中断等人工批 → 执行」，是 Human-in-the-Loop 的典型落地。
- checkpointer 持久化：同一 session 的多轮对话有记忆，且中断的退款能在稍后被恢复。
"""

from typing import Annotated, Literal, TypedDict

import config  # noqa: F401  （副作用：把仓库根加入 sys.path）
import kb
import tools
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from common.llm_provider import get_llm

llm = get_llm(temperature=0)


# ---------------- 共享状态 ----------------
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # 多轮记忆
    route: str            # 主管的路由决定
    answer: str           # 给用户的最终回复
    refund: dict          # 待审批的退款 {order_id, amount}
    approved: bool        # 人工审批结果（由 API 在恢复执行前写入）


def _last_user_text(state: State) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


# ---------------- 主管节点：意图路由 ----------------
class Route(BaseModel):
    intent: Literal["policy", "order", "refund", "other"] = Field(
        description="用户意图：policy=问政策规则；order=查订单物流；refund=要求退款；other=其他/闲聊"
    )


def triage_node(state: State) -> dict:
    decision = llm.with_structured_output(Route).invoke(
        [HumanMessage(
            "判断用户最新消息的意图，返回 intent。\n用户消息：" + _last_user_text(state)
        )]
    )
    print(f"  [triage] 路由 → {decision.intent}")
    return {"route": decision.intent}


# ---------------- 专家节点 1：政策问答（RAG）----------------
def policy_node(state: State) -> dict:
    q = _last_user_text(state)
    docs = kb.get_retriever().invoke(q)
    context = "\n---\n".join(d.page_content for d in docs)
    ans = llm.invoke(
        f"你是云启商城客服。只依据下面政策资料回答，没有就说未找到。简洁友好。\n\n资料：\n{context}\n\n问题：{q}"
    ).content
    return {"answer": ans, "messages": [AIMessage(ans)]}


# ---------------- 专家节点 2：订单查询（工具调用）----------------
@tool
def get_order_status(order_id: str) -> str:
    """根据订单号查询订单状态与物流。"""
    return tools.order_status_text(order_id)


def order_node(state: State) -> dict:
    bound = llm.bind_tools([get_order_status])
    msgs = [HumanMessage(
        "你是客服，用户要查订单。若消息里有订单号就调用工具查询；没有就礼貌索要订单号。\n"
        "用户消息：" + _last_user_text(state)
    )]
    ai = bound.invoke(msgs)
    if ai.tool_calls:
        msgs.append(ai)
        for c in ai.tool_calls:
            res = get_order_status.invoke(c["args"])
            msgs.append(ToolMessage(content=str(res), tool_call_id=c["id"]))
        ai = bound.invoke(msgs)
    ans = ai.content
    return {"answer": ans, "messages": [AIMessage(ans)]}


# ---------------- 退款流程：准备 →（人工审批）→ 执行 ----------------
class RefundInfo(BaseModel):
    order_id: str = Field(description="用户要退款的订单号；找不到填空字符串")


def prepare_refund_node(state: State) -> dict:
    info = llm.with_structured_output(RefundInfo).invoke(
        [HumanMessage("从用户消息中提取要退款的订单号。\n消息：" + _last_user_text(state))]
    )
    order = tools.get_order(info.order_id)
    if not order:
        ans = "请提供您要退款的订单号，我来帮您发起退款申请。"
        return {"answer": ans, "messages": [AIMessage(ans)], "route": "refund_incomplete"}
    refund = {"order_id": info.order_id, "amount": order["amount"]}
    ans = f"已为订单 {info.order_id}（{order['item']}）发起退款申请，金额 ¥{order['amount']}，正在等待主管审批。"
    print(f"  [prepare_refund] 待审批：{refund}")
    return {"refund": refund, "answer": ans, "messages": [AIMessage(ans)]}


def execute_refund_node(state: State) -> dict:
    # 只有在「图被中断、人工做出决定、再被恢复」后才会执行到这里。
    # approved 由 API 在恢复前通过 update_state 写入。
    r = state.get("refund", {})
    if state.get("approved"):
        ans = f"✅ 审批通过，已为订单 {r.get('order_id')} 退款 ¥{r.get('amount')}，预计 1-7 个工作日原路退回。"
        print(f"  [execute_refund] 已执行退款：{r}")
    else:
        ans = f"❌ 退款申请（订单 {r.get('order_id')}）未通过主管审批，已取消。如有疑问可联系人工客服。"
        print(f"  [execute_refund] 退款被拒：{r}")
    return {"answer": ans, "messages": [AIMessage(ans)]}


# ---------------- 闲聊节点 ----------------
def other_node(state: State) -> dict:
    ans = llm.invoke(
        "你是云启商城的友好客服。简短回应用户，并引导其说明需要什么帮助。\n用户：" + _last_user_text(state)
    ).content
    return {"answer": ans, "messages": [AIMessage(ans)]}


# ---------------- 路由函数 ----------------
def route_from_triage(state: State) -> str:
    return state["route"]


def after_prepare_refund(state: State) -> str:
    # 没提供有效订单号 → 直接结束让用户补充；否则进入需要审批的执行节点
    return END if state.get("route") == "refund_incomplete" else "execute_refund"


# ---------------- 组装图 ----------------
def build_graph(checkpointer):
    g = StateGraph(State)
    g.add_node("triage", triage_node)
    g.add_node("policy", policy_node)
    g.add_node("order", order_node)
    g.add_node("prepare_refund", prepare_refund_node)
    g.add_node("execute_refund", execute_refund_node)
    g.add_node("other", other_node)

    g.add_edge(START, "triage")
    g.add_conditional_edges("triage", route_from_triage, {
        "policy": "policy", "order": "order", "refund": "prepare_refund", "other": "other",
    })
    g.add_edge("policy", END)
    g.add_edge("order", END)
    g.add_edge("other", END)
    g.add_conditional_edges("prepare_refund", after_prepare_refund,
                            {"execute_refund": "execute_refund", END: END})
    g.add_edge("execute_refund", END)

    # interrupt_before：执行退款前暂停，把决定权交给人工 → Human-in-the-Loop
    return g.compile(checkpointer=checkpointer, interrupt_before=["execute_refund"])


def get_checkpointer():
    """优先用 SQLite 持久化（重启不丢记忆）；缺依赖时退回内存版。"""
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(config.CHECKPOINT_DB, check_same_thread=False)
        return SqliteSaver(conn)
    except Exception as e:  # noqa: BLE001
        from langgraph.checkpoint.memory import MemorySaver

        print(f"[graph] 未启用 SQLite 持久化（{e}），改用内存记忆。")
        return MemorySaver()
