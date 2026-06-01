"""
项目 4 · 智能数据分析助手（ChatBI 雏形）

业务场景：业务同学不会写 SQL/pandas，但想问「3 月哪个区域卖得最好？」。
传统做法要数据分析师手写查询；用 Agent，模型自己挑工具、传参数、算出答案。

技术要点：给 Agent 装上一组「安全的数据分析工具」，让它把自然语言问题翻译成工具调用。
这里**不**给模型直接执行任意 Python 的能力（eval 有注入风险），而是封装好几个**受控的结构化工具**——
这是生产级 NL2Analytics 的常见安全取舍。

运行：
    pip install langgraph pandas
    python stage03_agent/project04_data_analysis/agent.py
"""

import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from common.llm_provider import get_llm

CSV_PATH = pathlib.Path(__file__).parent / "sales_data.csv"
# 全局加载一次数据；工具闭包引用它
DF = pd.read_csv(CSV_PATH, parse_dates=["date"])


# ------------------------------------------------------------------
# 受控的数据分析工具集——每个工具只做一件确定的事，参数被类型与 docstring 约束
# ------------------------------------------------------------------
@tool
def get_schema() -> str:
    """返回数据表的列名、类型和前 3 行样例。分析前应先调用它了解数据结构。"""
    info = [f"列：{list(DF.columns)}", f"行数：{len(DF)}", f"时间范围：{DF['date'].min().date()} ~ {DF['date'].max().date()}"]
    info.append("样例：\n" + DF.head(3).to_string(index=False))
    info.append(f"region 取值：{DF['region'].unique().tolist()}")
    info.append(f"category 取值：{DF['category'].unique().tolist()}")
    return "\n".join(info)


@tool
def aggregate(metric: str, operation: str, group_by: str = "") -> str:
    """对数值列做聚合统计。
    metric: 要统计的数值列，只能是 'units' 或 'revenue'。
    operation: 聚合方式，只能是 'sum' / 'mean' / 'max' / 'min' / 'count'。
    group_by: 可选，按哪个列分组（如 'region' / 'product' / 'category'）；留空则对全表聚合。
    """
    if metric not in {"units", "revenue"}:
        return f"错误：metric 只能是 units 或 revenue，收到 {metric}"
    if operation not in {"sum", "mean", "max", "min", "count"}:
        return f"错误：operation 不支持 {operation}"
    if group_by:
        if group_by not in DF.columns:
            return f"错误：没有列 {group_by}"
        result = DF.groupby(group_by)[metric].agg(operation).sort_values(ascending=False)
        return result.to_string()
    return f"{operation}({metric}) = {DF[metric].agg(operation)}"


@tool
def filter_by_month(month: int) -> str:
    """筛选某个月份（1-12）的数据，返回该月每个区域的营收汇总。"""
    sub = DF[DF["date"].dt.month == month]
    if sub.empty:
        return f"{month} 月没有数据"
    return sub.groupby("region")["revenue"].sum().sort_values(ascending=False).to_string()


def build_agent():
    system = (
        "你是数据分析助手。面对用户问题，先用 get_schema 了解数据，再选择合适的工具计算，"
        "最后用简洁中文给出结论（带具体数字）。不要编造数据，一切以工具返回为准。"
    )
    return create_react_agent(
        model=get_llm(temperature=0),
        tools=[get_schema, aggregate, filter_by_month],
        prompt=system,
    )


def ask(agent, question):
    print(f"\n{'='*60}\n👤 {question}")
    result = agent.invoke({"messages": [("user", question)]})
    # 打印 Agent 调了哪些工具（可观测性）
    for m in result["messages"]:
        if getattr(m, "tool_calls", None):
            for c in m.tool_calls:
                print(f"   🔧 {c['name']}({c['args']})")
    print(f"🤖 {result['messages'][-1].content}")


if __name__ == "__main__":
    agent = build_agent()
    ask(agent, "总营收是多少？")
    ask(agent, "哪个区域的总营收最高？")
    ask(agent, "3 月份各区域营收情况如何？")
    ask(agent, "哪个产品卖出的件数最多？")
