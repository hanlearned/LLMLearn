"""
项目 6 · MCP Server（可选进阶）：把工具做成「标准插座」

什么是 MCP（Model Context Protocol）？
- 一个开放协议，让「工具/数据源」以标准方式对外暴露，任何支持 MCP 的客户端
  （Claude Desktop、Cursor、你自己的 Agent）都能即插即用，不用为每个工具写一遍对接。
- 类比：USB 之于硬件。以前每个工具一套私有接口；MCP 让工具变成「标准插座」。

为什么招聘场景适合 MCP？
- 「给候选人打分」「查岗位库」这类能力，HR 系统、面试官的 Agent、甚至 Claude Desktop
  可能都想用。做成 MCP Server，一次实现，处处接入。

本文件用官方 `mcp` 库的 FastMCP，把 score_candidate 暴露为一个 MCP 工具。

运行（需要 pip install mcp）：
    python stage04_langgraph/project06_hr_assistant/mcp_server.py
  它会以 stdio 传输方式启动，等待 MCP 客户端连接。

如何让 LangGraph Agent 消费这个 MCP Server？
    pip install langchain-mcp-adapters
    from langchain_mcp_adapters.client import MultiServerMCPClient
    client = MultiServerMCPClient({
        "hr": {"command": "python", "args": ["mcp_server.py"], "transport": "stdio"}
    })
    tools = await client.get_tools()          # MCP 工具 → LangChain 工具
    agent = create_react_agent(get_llm(), tools)  # 和本地工具用法完全一样！
"""

import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("未安装 mcp 库。这是可选进阶示例，运行需要：pip install mcp")
    raise SystemExit(0)

from common.llm_provider import get_llm

mcp = FastMCP("hr-tools")


@mcp.tool()
def score_candidate(jd_requirements: str, resume: str) -> str:
    """给候选人简历相对岗位要求打匹配分(0-100)并说明理由。"""
    llm = get_llm(temperature=0)
    prompt = (
        f"你是资深 HR。根据岗位要求给候选人打匹配分(0-100)，列出 2 条匹配点和 1 条短板。\n\n"
        f"【岗位要求】\n{jd_requirements}\n\n【候选人简历】\n{resume}"
    )
    return llm.invoke(prompt).content


@mcp.tool()
def normalize_skill(skill: str) -> str:
    """把口语化的技能名归一化为标准技能标签，如『会用 langchain』→『LangChain』。"""
    return get_llm(temperature=0).invoke(f"把技能『{skill}』归一化为标准技术名词，只返回名词本身。").content


if __name__ == "__main__":
    # 以 stdio 传输启动 MCP Server，等待客户端连接
    mcp.run(transport="stdio")
