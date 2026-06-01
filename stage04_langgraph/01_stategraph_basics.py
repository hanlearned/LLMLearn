"""
Stage 4 · StateGraph 基础：把工作流画成一张「状态图」

为什么需要 LangGraph？
- Stage 3 的 Agent 是「一个模型 + 一堆工具自循环」，适合单一任务。
- 但真实业务常是**多步骤、多角色、有分支、要回头**的流程（写作→审稿→改稿→再审）。
  这种用一个 Agent 硬怼很别扭，用「图」来表达却天然贴切：节点是步骤，边是流转。

StateGraph 的三件套：
    1. State：一个共享的「状态字典」，所有节点读它、写它（用 TypedDict 定义结构）
    2. Node：一个函数，接收当前 state，返回要更新的字段（局部更新，框架自动合并）
    3. Edge：节点间的连线，决定执行顺序；START 是入口，END 是出口

本例：一个最简单的两节点流水线——把输入文本先「润色」再「翻译」。

运行：
    pip install langgraph
    python stage04_langgraph/01_stategraph_basics.py
"""

import pathlib
import sys
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langgraph.graph import StateGraph, START, END

from common.llm_provider import get_llm

llm = get_llm(temperature=0.3)


# ------------------------------------------------------------------
# 1. 定义状态：整张图共享的数据结构
#    每个字段就是节点之间传递的「变量」。
# ------------------------------------------------------------------
class State(TypedDict):
    raw: str         # 原始输入
    polished: str    # 润色后的中文
    translated: str  # 翻译后的英文


# ------------------------------------------------------------------
# 2. 定义节点：每个节点是一个函数，输入 state，返回「要更新哪些字段」
#    注意：返回的是局部 dict，框架会自动合并进总 state，不用手动拷贝整个 state。
# ------------------------------------------------------------------
def polish_node(state: State) -> dict:
    """节点一：把原始文本润色得更书面。"""
    resp = llm.invoke(f"把下面这句话润色得更书面、专业，只返回结果：\n{state['raw']}")
    print(f"  [polish] {resp.content}")
    return {"polished": resp.content}


def translate_node(state: State) -> dict:
    """节点二：把润色后的文本翻译成英文。读的是上一个节点写入的 polished。"""
    resp = llm.invoke(f"把下面这句中文翻译成地道英文，只返回结果：\n{state['polished']}")
    print(f"  [translate] {resp.content}")
    return {"translated": resp.content}


# ------------------------------------------------------------------
# 3. 组装图：加节点、连边、编译
# ------------------------------------------------------------------
def build_graph():
    graph = StateGraph(State)
    graph.add_node("polish", polish_node)
    graph.add_node("translate", translate_node)

    graph.add_edge(START, "polish")        # 入口 → 润色
    graph.add_edge("polish", "translate")  # 润色 → 翻译
    graph.add_edge("translate", END)       # 翻译 → 出口

    return graph.compile()  # compile 后才能 invoke/stream


if __name__ == "__main__":
    app = build_graph()

    # 可视化图结构（纯文本），帮助建立「流程就是一张图」的直觉
    print("图结构：")
    print(app.get_graph().draw_ascii())

    print("\n开始执行：")
    result = app.invoke({"raw": "这个东西还挺好用的，就是有点小贵。"})

    print("\n最终状态：")
    print(f"  原始：{result['raw']}")
    print(f"  润色：{result['polished']}")
    print(f"  翻译：{result['translated']}")
