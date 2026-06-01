"""
项目 6 · AI 招聘助手（RAG + Agent，含 MCP 扩展说明）

综合实战：把 Stage 2（RAG）和 Stage 3/4（Agent）拧成一个系统。
HR 用自然语言交互，Agent 自己决定「该查岗位要求」还是「该给候选人打分」。

Agent 装备两类能力：
    1. RAG 检索工具 search_jd —— 从岗位 JD 知识库检索要求（这是把 RAG 封装成 Agent 的一个工具！）
    2. 业务工具 score_candidate —— 给候选人简历相对某岗位打匹配分

🔌 关于 MCP：score_candidate 这类工具也可以做成一个独立的 MCP Server 对外提供，
   让任何支持 MCP 的客户端（Claude Desktop、其它 Agent）即插即用。
   见同目录 mcp_server.py 与 docs 里的说明。本文件用进程内工具，跑起来最简单。

运行：
    pip install langgraph langchain-chroma
    python stage04_langgraph/project06_hr_assistant/hr_agent.py
"""

import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from common.embeddings_provider import get_embeddings
from common.llm_provider import get_llm

DATA_DIR = pathlib.Path(__file__).parent / "data"


# ---------- 把 RAG 检索封装成「检索器」，供工具调用 ----------
def build_jd_retriever():
    docs = []
    for f in sorted(DATA_DIR.glob("*.md")):
        docs.extend(TextLoader(str(f), encoding="utf-8").load())
    chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(docs)
    vs = Chroma.from_documents(chunks, get_embeddings(), collection_name="hr_jd")
    return vs.as_retriever(search_kwargs={"k": 3})

_retriever = None


@tool
def search_jd(query: str) -> str:
    """检索招聘岗位的职责与要求。当需要了解某个岗位要求什么技能/经验时调用。"""
    global _retriever
    if _retriever is None:
        _retriever = build_jd_retriever()
    docs = _retriever.invoke(query)
    return "\n---\n".join(d.page_content for d in docs)


@tool
def score_candidate(jd_requirements: str, resume: str) -> str:
    """给候选人简历相对岗位要求打匹配分（0-100）并说明理由。
    jd_requirements: 岗位要求文本（可先用 search_jd 获取）。
    resume: 候选人简历文本。
    """
    llm = get_llm(temperature=0)
    prompt = (
        f"你是资深 HR。根据岗位要求给候选人打匹配分(0-100)，并列出 2 条匹配点和 1 条短板。\n\n"
        f"【岗位要求】\n{jd_requirements}\n\n【候选人简历】\n{resume}\n\n"
        f"输出格式：匹配分: X/100\n匹配点: ...\n短板: ..."
    )
    return llm.invoke(prompt).content


def build_agent():
    system = (
        "你是企业招聘助手。可用工具：search_jd（查岗位要求）、score_candidate（给候选人打分）。\n"
        "当需要给候选人打分时，应先用 search_jd 拿到对应岗位的要求，再调用 score_candidate。"
    )
    return create_react_agent(model=get_llm(temperature=0),
                              tools=[search_jd, score_candidate], prompt=system)


SAMPLE_RESUME = (
    "张三，本科计算机，3 年后端经验。精通 Python 和 FastAPI，做过企业知识库 RAG 项目，"
    "用过 Chroma 和 LangChain，熟悉 LangGraph 搭 Agent，写过技术博客。"
)


def ask(agent, q):
    print(f"\n{'='*60}\n👤 {q}")
    result = agent.invoke({"messages": [("user", q)]})
    for m in result["messages"]:
        if getattr(m, "tool_calls", None):
            for c in m.tool_calls:
                print(f"   🔧 {c['name']}({str(c['args'])[:60]}…)")
    print(f"🤖 {result['messages'][-1].content}")


if __name__ == "__main__":
    agent = build_agent()
    ask(agent, "LLM 应用开发工程师（后端）这个岗位要求什么技能？")
    ask(agent, f"帮我评估这位候选人是否适合后端 LLM 应用开发岗位：{SAMPLE_RESUME}")
