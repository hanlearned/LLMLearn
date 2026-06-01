"""
Capstone · 企业智能客服 Agent（毕业综合项目）

这是整个课程的「集大成」项目。一个真实的智能客服，需要同时具备六大能力——
本文件把它们拧在一起：

    能力1 LCEL 编排   → 工具内部用 prompt|llm 组合
    能力2 RAG         → search_policy 工具：从客服政策知识库检索（Stage 2）
    能力3 工具调用     → get_order_status / create_ticket 业务工具（Stage 3）
    能力4 LangGraph   → create_react_agent 编排「检索/查单/开工单」的自主决策（Stage 3/4）
    能力5 记忆/可观测  → checkpointer 多轮记忆 + 打印工具调用轨迹（Stage 3/4）
    能力6 部署        → 见 api.py：FastAPI + 流式 + 会话隔离（Stage 6）

客服 Agent 的核心价值：用户一句话（"我上周买的东西还没到，能退吗？"），Agent 自己决定
先查物流、再查退货政策、必要时开工单——这正是 Agent 相对传统「关键词客服」的代差。

运行：
    pip install langgraph langchain-chroma
    python capstone/agent.py
"""

import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from common.embeddings_provider import get_embeddings
from common.llm_provider import get_llm

DATA_DIR = pathlib.Path(__file__).parent / "data"

# 模拟订单数据库（真实场景这里查业务系统）
_ORDERS = {
    "10001": {"status": "已发货", "logistics": "顺丰 SF123456，预计明天送达", "item": "蓝牙耳机"},
    "10002": {"status": "待发货", "logistics": "暂无", "item": "定制马克杯"},
}

_retriever = None


def _get_retriever():
    """懒加载：把客服政策知识库建成 RAG 检索器。"""
    global _retriever
    if _retriever is None:
        docs = []
        for f in sorted(DATA_DIR.glob("*.md")):
            docs.extend(TextLoader(str(f), encoding="utf-8").load())
        chunks = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40).split_documents(docs)
        vs = Chroma.from_documents(chunks, get_embeddings(), collection_name="cs_kb")
        _retriever = vs.as_retriever(search_kwargs={"k": 3})
    return _retriever


# ---------------- 工具集：RAG 检索 + 业务动作 ----------------
@tool
def search_policy(query: str) -> str:
    """检索客服政策（退换货、物流、会员、发票、赔付等规则）。涉及"政策/规则/能不能"的问题用它。"""
    docs = _get_retriever().invoke(query)
    return "\n---\n".join(d.page_content for d in docs)


@tool
def get_order_status(order_id: str) -> str:
    """根据订单号查询订单状态与物流。需要用户提供订单号。"""
    order = _ORDERS.get(order_id.strip())
    if not order:
        return f"未找到订单 {order_id}，请核对订单号。"
    return f"订单{order_id}（{order['item']}）：状态={order['status']}，物流={order['logistics']}"


@tool
def create_ticket(summary: str) -> str:
    """当问题无法自助解决、需要人工处理时，创建售后工单。summary 是问题简述。"""
    # 真实场景写入工单系统；这里返回一个假工单号
    return f"已为你创建售后工单 T-8800，问题：{summary}。人工客服将在 24 小时内联系你。"


SYSTEM_PROMPT = (
    "你是云启商城的智能客服，友好、专业、简洁。\n"
    "可用工具：search_policy（查政策规则）、get_order_status（查订单，需订单号）、create_ticket（开人工工单）。\n"
    "原则：\n"
    "1. 涉及规则/政策的问题，先 search_policy 再依据检索结果回答，不要凭空编造政策。\n"
    "2. 涉及具体订单的问题，若用户没给订单号，先礼貌索要。\n"
    "3. 自助无法解决或用户要求人工时，用 create_ticket 开工单。\n"
    "4. 回答末尾可以问一句『还有什么可以帮您？』。"
)


def build_agent():
    """构建带记忆的客服 Agent。checkpointer 让同一用户的多轮对话有上下文。"""
    return create_react_agent(
        model=get_llm(temperature=0),
        tools=[search_policy, get_order_status, create_ticket],
        prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )


def chat(agent, text, session_id, show_trace=True):
    """单轮对话。session_id 作为 thread_id 隔离不同用户的记忆。"""
    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke({"messages": [("user", text)]}, config=config)
    if show_trace:
        for m in result["messages"]:
            if getattr(m, "tool_calls", None):
                for c in m.tool_calls:
                    print(f"   🔧 {c['name']}({str(c['args'])[:50]})")
    print(f"🤖 {result['messages'][-1].content}")
    return result["messages"][-1].content


if __name__ == "__main__":
    agent = build_agent()
    sid = "user-A"
    for turn in [
        "你们的退货政策是什么？生鲜能退吗？",          # → 走 RAG 查政策
        "我的订单 10001 到哪了？",                       # → 走订单查询工具
        "那如果一直没收到，我想找人工处理",              # → 开工单（结合上文记忆）
    ]:
        print(f"\n{'='*60}\n👤 {turn}")
        chat(agent, turn, session_id=sid)
