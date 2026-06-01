"""
项目 7 · Prompt A/B 测试平台

痛点：调 Prompt 全靠「感觉好像变好了」。这在团队里不可信、不可复现。
工程化做法：固定一批测试问题，让两版 Prompt 各自作答，再用 LLM 当裁判**成对比较**，
统计胜率。从此「A 比 B 好」是一个有数字、可复现的结论。

本平台的三个工程要点：
    1. 测试集固定 —— 同一批问题，控制变量
    2. 成对比较（pairwise）—— 比「分别打分」更可靠，裁判更擅长「二选一」
    3. 位置去偏 —— 交换 A/B 出现顺序各评一次，消除「裁判偏爱第一个」的位置偏见

运行：
    python stage05_promptops/project07_prompt_ab_test/ab_test.py
"""

import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from langchain_core.prompts import ChatPromptTemplate

from common.llm_provider import get_llm

# ---- 被测对象：两版「系统提示词」 ----
PROMPT_A = "你是客服助手，简洁回答用户问题。"
PROMPT_B = (
    "你是专业、友好的客服助手。回答时：先用一句话直接给出结论，再用 1-2 句补充说明，"
    "语气亲切，结尾询问是否还有其他需要帮助的。"
)

# ---- 固定测试集 ----
TEST_QUESTIONS = [
    "你们的退货政策是什么？",
    "我忘记密码了怎么办？",
    "会员有什么权益？",
    "为什么我的订单还没发货？",
]


def generate(system_prompt: str, question: str) -> str:
    """用某版 system prompt 生成回答。"""
    chain = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{q}")]) | get_llm(temperature=0.3)
    return chain.invoke({"q": question}).content


JUDGE = ChatPromptTemplate.from_template(
    "你是严格的客服质量评审。针对同一个用户问题，有两个回答。\n"
    "评判哪个更好（更准确、有用、体验好）。只输出 '1' 或 '2' 或 'tie'，不要解释。\n\n"
    "【用户问题】{question}\n\n【回答1】\n{ans1}\n\n【回答2】\n{ans2}\n\n更好的是："
)


def judge_pairwise(question, ans1, ans2) -> str:
    """裁判二选一，返回 '1'/'2'/'tie'。"""
    chain = JUDGE | get_llm(temperature=0)
    out = chain.invoke({"question": question, "ans1": ans1, "ans2": ans2}).content.strip().lower()
    if "1" in out:
        return "1"
    if "2" in out:
        return "2"
    return "tie"


def run():
    wins = {"A": 0, "B": 0, "tie": 0}
    print(f"测试集 {len(TEST_QUESTIONS)} 题，每题做位置去偏的双向评判\n")

    for q in TEST_QUESTIONS:
        ans_a = generate(PROMPT_A, q)
        ans_b = generate(PROMPT_B, q)

        # 位置去偏：第一次 A 在前，第二次 B 在前，两次都赢才算真赢
        v1 = judge_pairwise(q, ans_a, ans_b)   # 1=>A, 2=>B
        v2 = judge_pairwise(q, ans_b, ans_a)   # 1=>B, 2=>A（位置交换）

        a_score = (v1 == "1") + (v2 == "2")
        b_score = (v1 == "2") + (v2 == "1")
        if a_score > b_score:
            winner = "A"
        elif b_score > a_score:
            winner = "B"
        else:
            winner = "tie"
        wins[winner] += 1
        print(f"  Q: {q[:20]}… → 本题胜者：{winner}（A得{a_score}/B得{b_score}）")

    total = len(TEST_QUESTIONS)
    print("\n===== A/B 测试结果 =====")
    print(f"  Prompt A 胜：{wins['A']}/{total}（{wins['A']/total:.0%}）")
    print(f"  Prompt B 胜：{wins['B']}/{total}（{wins['B']/total:.0%}）")
    print(f"  平局：{wins['tie']}/{total}")
    better = "B" if wins["B"] > wins["A"] else ("A" if wins["A"] > wins["B"] else "难分伯仲")
    print(f"  → 结论：Prompt {better} 更优")


if __name__ == "__main__":
    run()
    print("\n💡 扩展：测试集换成你真实业务问题、接入更多 Prompt 版本、把胜率落库做成趋势看板，就是一个 LLMOps 平台。")
