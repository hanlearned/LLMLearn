"""
Stage 5 · CoT 思维链 + Self-Consistency 自洽投票

两个能立竿见影提升「推理类任务」准确率的技巧：
    1. CoT（Chain-of-Thought）：让模型「一步步想」再给答案，把推理过程显式化，正确率显著提升。
    2. Self-Consistency：对同一题用较高温度采样多个 CoT 答案，取「多数票」，进一步抗随机错误。

本例用一道需要多步推理的数学题对比：直接答 vs CoT vs CoT+多数投票。

运行：
    python stage05_promptops/01_cot_self_consistency.py
"""

import pathlib
import sys
from collections import Counter

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common.llm_provider import get_llm

QUESTION = (
    "小明有 3 箱苹果，每箱 12 个。他卖掉了 8 个，又买进 2 箱（每箱还是 12 个），"
    "然后送给邻居 5 个。请问小明现在有多少个苹果？"
)


def extract_number(text: str) -> str:
    """从回答里抠出最后一个数字，作为最终答案，便于投票。"""
    nums = [s for s in "".join(c if c.isdigit() else " " for c in text).split()]
    return nums[-1] if nums else "?"


def direct_answer():
    """对照组：不引导思考，直接要答案。"""
    llm = get_llm(temperature=0)
    resp = llm.invoke(QUESTION + "\n直接给出数字答案，不要解释。").content
    print(f"① 直接回答：{resp.strip()}")


def cot_answer():
    """CoT：引导逐步推理。"""
    llm = get_llm(temperature=0)
    resp = llm.invoke(QUESTION + "\n让我们一步步计算，最后用『答案：X』给出结果。").content
    print(f"\n② CoT 逐步推理：\n{resp.strip()}")


def self_consistency(n: int = 5):
    """Self-Consistency：高温采样 n 次 CoT，对最终数字投票。"""
    llm = get_llm(temperature=0.8)  # 温度要高，才能产生多样的推理路径
    answers = []
    for i in range(n):
        resp = llm.invoke(QUESTION + "\n一步步推理后，用『答案：X』结尾。").content
        ans = extract_number(resp)
        answers.append(ans)
        print(f"   第{i+1}次采样 → 答案 {ans}")
    winner, count = Counter(answers).most_common(1)[0]
    print(f"\n③ Self-Consistency 多数投票：{answers} → 最终 {winner}（得票 {count}/{n}）")


if __name__ == "__main__":
    # 正确答案：3*12=36，卖8剩28，+2箱24 → 52，送5 → 47
    print(f"题目：{QUESTION}\n（正确答案是 47）\n")
    direct_answer()
    cot_answer()
    print()
    self_consistency()
    print("\n💡 推理类任务：CoT 几乎always值得；不确定性高时再叠加 Self-Consistency 投票。代价是多次调用、更贵。")
