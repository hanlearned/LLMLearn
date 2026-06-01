"""
Stage 3 · ⭐推荐写法：用 LangGraph 的 create_react_agent 构建 Agent

02 手写循环是为了理解原理。真实项目里不要自己造轮子——用 LangGraph 的 `create_react_agent`，
一行就得到一个**生产级** Agent：自带状态管理、工具循环、可持久化记忆、可流式观察轨迹。

⚠️ 注意区分：
- 老的 `from langchain.agents import create_react_agent / AgentExecutor` —— 已软弃用。
- 新的 `from langgraph.prebuilt import create_react_agent` —— 本课程推荐，下面用的就是它。

本例额外演示「带记忆的多轮对话」：靠 checkpointer + thread_id 实现同一会话上下文延续。

运行：
    pip install langgraph
    python stage03_agent/03_langgraph_agent.py
"""

import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from common.llm_provider import get_llm


@tool
def get_population(city: str) -> int:
    """查询城市常住人口（万人）。"""
    return {"北京": 2184, "上海": 2487, "广州": 1873, "深圳": 1766}.get(city, 0)


@tool
def add(a: float, b: float) -> float:
    """两数相加。"""
    return a + b


def build_agent():
    """一行构建带记忆的 ReAct Agent。

    - model：绑定工具调用能力的 LLM（temperature=0 让工具选择稳定）
    - tools：工具列表，Agent 会自动完成「决定→调用→喂回→循环」
    - checkpointer：记忆。MemorySaver 把每个 thread_id 的对话状态存在内存里
    """
    memory = MemorySaver()
    return create_react_agent(
        model=get_llm(temperature=0),
        tools=[get_population, add],
        checkpointer=memory,
    )


def ask(agent, text, thread_id):
    """发一条消息。靠 thread_id 区分会话；相同 thread_id 自动带上历史。"""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [("user", text)]}, config=config)
    # result["messages"] 是完整轨迹；最后一条是最终回答
    print(f"\n👤 {text}")
    print(f"🤖 {result['messages'][-1].content}")
    return result


def print_trace(result):
    """打印完整执行轨迹——调试 Agent「为什么这么干」时全靠它。"""
    print("\n   --- 执行轨迹 ---")
    for m in result["messages"]:
        role = type(m).__name__.replace("Message", "")
        if getattr(m, "tool_calls", None):
            print(f"   [{role}] 调用工具：{[(c['name'], c['args']) for c in m.tool_calls]}")
        elif role == "Tool":
            print(f"   [Tool] 返回：{m.content}")
        elif m.content:
            print(f"   [{role}] {m.content[:60]}")


if __name__ == "__main__":
    agent = build_agent()

    # 第一轮：需要多步工具调用
    r1 = ask(agent, "北京和上海的人口加起来是多少万？", thread_id="user-001")
    print_trace(r1)

    # 第二轮：同一 thread_id，测试记忆——"再加上广州"依赖上一轮的上下文
    ask(agent, "在刚才的基础上再加上广州的人口呢？", thread_id="user-001")

    # 换一个 thread_id：全新会话，没有上面的记忆
    ask(agent, "我们刚才聊到哪个数字了？", thread_id="user-999")
    print("   （上面这个新会话应答不上来，证明记忆是按 thread_id 隔离的）")
