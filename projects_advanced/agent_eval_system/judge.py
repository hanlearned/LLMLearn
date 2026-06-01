"""
高级实战 · Agent/RAG 自动化评测系统 —— LLM-as-a-Judge 评分裁判

本模块只做一件事：给定（问题、被测系统的答案、参考答案/要点），
用一个温度=0 的「裁判 LLM」从三个维度打分（1-5），并要求附打分理由。

为什么要「多维度 + 附理由」？
--------------------------------
1. 单一「对/错」过于粗糙，无法定位问题。拆成三个正交维度更可解释：
       correctness  正确性：答案与参考答案在事实上是否一致、有无错误。
       faithfulness 忠实度：答案是否「有据可依」、没有编造参考之外的内容（幻觉）。
       relevance    相关性：答案是否切题、聚焦问题本身、没有答非所问。
2. 要求模型先写 reason 再给分（让它「先思考再下结论」），分数更稳、更可审计。

为什么用 with_structured_output 而不是手写正则/JSON 解析？
--------------------------------------------------------
裸 LLM 输出 JSON 经常带 ```json 包裹、多余解释、字段缺失，正则解析很脆。
LangChain 0.3 的 `model.with_structured_output(PydanticModel)` 会借助底层模型的
tool/function calling，直接把输出约束成 Pydantic 对象，省去格式化 prompt，解析稳健。
"""

from __future__ import annotations

import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()
# 本文件位于 projects_advanced/agent_eval_system/judge.py，向上 2 层即仓库根目录。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from common.llm_provider import get_llm


# ---------------------------------------------------------------------------
# 1. 评分 Schema：用 Pydantic 定义裁判必须返回的结构。
#    每个维度都成对出现：一个 1-5 的整数分 + 一段中文理由。
#    Field 的 description 会被 with_structured_output 注入给模型，相当于评分细则。
# ---------------------------------------------------------------------------
class JudgeScore(BaseModel):
    """裁判对单条样本的多维度评分结果。"""

    correctness: int = Field(
        ge=1,
        le=5,
        description="正确性(1-5)：答案与参考答案在事实层面是否一致。5=完全正确，3=部分正确/有小瑕疵，1=明显错误。",
    )
    correctness_reason: str = Field(description="对正确性评分的简要中文理由，需指出具体一致或冲突之处。")

    faithfulness: int = Field(
        ge=1,
        le=5,
        description="忠实度(1-5)：答案是否有据可依、未编造参考答案之外的内容。5=无任何臆造，1=大量幻觉。",
    )
    faithfulness_reason: str = Field(description="对忠实度评分的简要中文理由，指出是否存在无依据的编造。")

    relevance: int = Field(
        ge=1,
        le=5,
        description="相关性(1-5)：答案是否切题、聚焦问题本身。5=完全切题，1=答非所问。",
    )
    relevance_reason: str = Field(description="对相关性评分的简要中文理由。")


# ---------------------------------------------------------------------------
# 2. 裁判 Prompt：明确角色、评分标准与去偏要求。
#    去偏要点写进 system，避免「答案长就给高分」「措辞华丽就给高分」等常见偏置。
# ---------------------------------------------------------------------------
_JUDGE_SYSTEM = """你是一名严格、客观的答案评测专家。你的任务是参照【参考答案】，对【被测答案】从三个维度打分（1-5 的整数），并为每个维度给出简短中文理由。

评分标准：
- correctness 正确性：被测答案在事实上与参考答案是否一致，有无错误或遗漏关键要点。
- faithfulness 忠实度：被测答案是否只陈述有依据的内容，有没有编造参考答案中没有、且明显不实的信息（幻觉）。
- relevance   相关性：被测答案是否紧扣问题、没有答非所问或大量跑题。

去偏要求（务必遵守）：
1. 只看事实与内容，不要因为答案更长、措辞更华丽、语气更自信就给高分。
2. 参考答案是「要点」而非唯一措辞，被测答案用不同表述说对了要点同样得高分。
3. 三个维度相互独立分别评分，不要让某一维度的印象影响其他维度。
4. 拿不准时倾向于给中间分（3），并在理由中说明不确定之处。"""

_JUDGE_HUMAN = """【问题】
{question}

【参考答案/要点】
{reference}

【被测答案】
{answer}

请依据评分标准与去偏要求，输出三个维度的分数与理由。"""

_judge_prompt = ChatPromptTemplate.from_messages(
    [("system", _JUDGE_SYSTEM), ("human", _JUDGE_HUMAN)]
)


def _build_judge_chain():
    """构造裁判链：prompt | (绑定了结构化输出的温度0模型)。"""
    # 裁判必须用温度 0，保证评分可复现。
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(JudgeScore)
    return _judge_prompt | structured_llm


# 模块级单例：避免每条样本都重建链。
_JUDGE_CHAIN = None


def judge_one(question: str, answer: str, reference: str) -> JudgeScore:
    """
    对单条样本打分。

    参数
    ----
    question:  原始问题。
    answer:    被测系统给出的答案。
    reference: 参考答案 / 要点。

    返回
    ----
    JudgeScore：含三个维度分数与各自理由的结构化对象。
    若模型偶发返回 None（结构化解析失败），兜底返回一个保守的中间分对象，
    保证批量评测不会因为单条异常而整体中断。
    """
    global _JUDGE_CHAIN
    if _JUDGE_CHAIN is None:
        _JUDGE_CHAIN = _build_judge_chain()

    result = _JUDGE_CHAIN.invoke(
        {"question": question, "answer": answer, "reference": reference}
    )
    if result is None:  # 极少数情况下结构化输出可能为空，给出可追溯的兜底。
        return JudgeScore(
            correctness=3,
            correctness_reason="裁判结构化输出解析失败，给出保守中间分。",
            faithfulness=3,
            faithfulness_reason="裁判结构化输出解析失败，给出保守中间分。",
            relevance=3,
            relevance_reason="裁判结构化输出解析失败，给出保守中间分。",
        )
    return result


if __name__ == "__main__":
    # 自检：python projects_advanced/agent_eval_system/judge.py
    demo = judge_one(
        question="LCEL 是什么？",
        answer="LCEL 是 LangChain 的表达式语言，用 | 把组件串成统一的 Runnable 链。",
        reference="LCEL 是用管道符 | 把 Prompt/Model/Parser 串成 Runnable 的声明式语言，统一了调用接口。",
    )
    print("裁判评分：", demo.model_dump())
