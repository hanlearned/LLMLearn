"""
Stage 3 · 工具基础 + 一轮完整的 Tool Calling

Agent 的本质能力 = 让模型「自己决定调用哪个工具、传什么参数」。这件事的底层机制叫
Tool Calling（旧称 Function Calling）。理解这一轮往返，你就理解了所有 Agent 框架的内核。

一轮 Tool Calling 的完整往返：
    1. 把工具「说明书」绑定给模型（bind_tools）
    2. 模型读问题，决定要不要调工具、调哪个、传什么参数 → 返回 tool_calls（注意：此时还没真正执行！）
    3. 我们的代码真正执行工具，拿到结果
    4. 把工具结果作为 ToolMessage 喂回模型 → 模型生成最终自然语言答案

运行：
    python stage03_agent/01_tools_basics.py
"""

import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from common.llm_provider import get_llm


# ------------------------------------------------------------------
# 1. 用 @tool 定义工具
#    关键：函数名、类型注解、docstring 会被打包成给模型看的「工具说明书」。
#    所以 docstring 必须写清楚「这个工具是干嘛的」——模型靠它判断要不要调。
# ------------------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """查询指定城市当前的天气情况。city 是城市中文名，如「北京」。"""
    # 演示用：返回假数据。真实场景这里会调用天气 API。
    fake = {"北京": "晴，18℃", "上海": "多云，22℃", "广州": "小雨，27℃"}
    return fake.get(city, f"暂无 {city} 的天气数据")


@tool
def add(a: float, b: float) -> float:
    """计算两个数字相加的结果。"""
    return a + b


def inspect_tool_schema():
    """看看 @tool 到底把什么交给了模型——这就是模型做选择的依据。"""
    print("=== 工具说明书（模型看到的就是这些）===")
    print(f"name: {get_weather.name}")
    print(f"description: {get_weather.description}")
    print(f"args schema: {get_weather.args}\n")


def one_round_tool_calling():
    """手动跑完一轮 Tool Calling，把每一步摊开给你看。"""
    tools = [get_weather, add]
    tools_by_name = {t.name: t for t in tools}

    # 步骤 1：绑定工具。temperature=0 让工具选择更稳定
    llm = get_llm(temperature=0).bind_tools(tools)

    # 步骤 2：模型决定调用哪个工具（此刻还没执行！）
    messages = [HumanMessage("北京现在天气怎么样？")]
    ai_msg = llm.invoke(messages)
    messages.append(ai_msg)

    print("=== 步骤2：模型的决定 ===")
    print(f"模型直接回复的文本：{ai_msg.content!r}（通常为空，因为它选择了调工具）")
    print(f"模型要求调用的工具：{ai_msg.tool_calls}\n")

    if not ai_msg.tool_calls:
        print("模型认为无需调工具，直接回答了。")
        return

    # 步骤 3：我们的代码真正执行模型点名的工具
    print("=== 步骤3：执行工具 ===")
    for call in ai_msg.tool_calls:
        chosen = tools_by_name[call["name"]]
        result = chosen.invoke(call["args"])
        print(f"执行 {call['name']}({call['args']}) → {result}")
        # 步骤 4 的准备：把结果包成 ToolMessage，带上 tool_call_id 与请求对应
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    # 步骤 4：把工具结果喂回模型，得到最终自然语言答案
    final = llm.invoke(messages)
    print(f"\n=== 步骤4：模型整合工具结果后的最终回答 ===\n{final.content}")


if __name__ == "__main__":
    inspect_tool_schema()
    one_round_tool_calling()
    print("\n💡 注意：这只是『一轮』。Agent = 把这个『决定→执行→喂回』循环自动重复，直到模型不再要求调工具。")
    print("   手动写这个循环 → 见 02_react_from_scratch.py；用框架自动跑 → 见 03_langgraph_agent.py。")
