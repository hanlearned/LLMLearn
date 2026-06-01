"""
Stage 3 · 不依赖任何框架，手写一个 Agent 循环

01 跑通了「一轮」Tool Calling。Agent 的精髓是把这一轮**自动重复**，直到任务完成。
本文件手写这个循环，让你彻底看清：所谓 Agent，不过是一个「带工具的 while 循环」。

任务设计成必须**多步**才能完成，以体现循环的价值：
    「北京和上海的人口加起来是多少？」
    → 模型需要：查北京人口 → 查上海人口 → 把两个数相加 → 回答
    → 这要求模型连续调用工具好几轮，正是 ReAct（Reason+Act）的雏形。

运行：
    python stage03_agent/02_react_from_scratch.py
"""

import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from common.llm_provider import get_llm


@tool
def get_population(city: str) -> int:
    """查询指定城市的常住人口（单位：万人）。"""
    data = {"北京": 2184, "上海": 2487, "广州": 1873, "深圳": 1766}
    return data.get(city, 0)


@tool
def add(a: float, b: float) -> float:
    """计算两个数字相加。"""
    return a + b


def run_agent(question: str, max_steps: int = 6):
    """手写 Agent 循环：决定 → 执行 → 喂回 → 重复，直到模型不再要求调工具。"""
    tools = [get_population, add]
    tools_by_name = {t.name: t for t in tools}
    llm = get_llm(temperature=0).bind_tools(tools)

    messages = [
        SystemMessage("你是一个会使用工具的助手。需要数据时调用工具，拿到所有需要的数据后再给出最终答案。"),
        HumanMessage(question),
    ]

    print(f"❓ 任务：{question}\n")
    for step in range(1, max_steps + 1):
        ai_msg = llm.invoke(messages)
        messages.append(ai_msg)

        # 终止条件：模型不再要求调工具 → 它认为任务完成了
        if not ai_msg.tool_calls:
            print(f"\n✅ 第 {step} 步：模型给出最终答案")
            print(f"🤖 {ai_msg.content}")
            return ai_msg.content

        # 否则：执行模型这一步点名的所有工具，把结果喂回
        print(f"🔁 第 {step} 步：模型决定调用工具")
        for call in ai_msg.tool_calls:
            result = tools_by_name[call["name"]].invoke(call["args"])
            print(f"    Action: {call['name']}({call['args']})  →  Observation: {result}")
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    print("⚠️ 达到最大步数仍未完成（可能陷入循环）——这就是为什么真实 Agent 都要设 max_steps 兜底。")


if __name__ == "__main__":
    run_agent("北京和上海的人口加起来是多少万人？")
    print("\n" + "=" * 60)
    run_agent("深圳的人口是多少？")  # 这个一步就够，对比观察循环会提前结束
    print("\n💡 这就是 Agent 的内核。框架（如 LangGraph）只是把这个循环做得更健壮：")
    print("   加上状态管理、持久化、错误重试、人工介入——但 while 循环的本质不变。")
