"""
Capstone 评测看板：用数字证明这个客服 Agent「答得好不好」

跑法：把评测集里的问题灌给客服图，拿到答案后用 LLM-as-a-Judge 给每条打分
（正确性 1-5），最后聚合出平均分、通过率，并生成 Markdown 报告。
这就是把 Stage 5 的评测能力用到自己 Capstone 上 —— 闭环。

运行：
    pip install langgraph langchain-chroma
    python capstone/eval/run_eval.py
"""

import json
import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()

# 让本脚本能 import 到 backend 里的 graph（以及它依赖的 config/kb/tools）
BACKEND = pathlib.Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

import graph as g  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from common.llm_provider import get_llm  # noqa: E402

EVAL_SET = json.loads((pathlib.Path(__file__).parent / "eval_set.json").read_text(encoding="utf-8"))
PASS = 4  # 正确性 >= 4 视为通过

JUDGE = ChatPromptTemplate.from_template(
    "你是严格评测员。对比【参考答案】判断【实际答案】的正确性，打 1-5 分"
    "（5=完全正确一致，3=部分正确，1=错误或答非所问）。只输出一个数字。\n\n"
    "【问题】{q}\n【参考答案】{ref}\n【实际答案】{ans}\n\n正确性评分："
)


def main():
    app = g.build_graph(MemorySaver())
    judge = JUDGE | get_llm(temperature=0) | StrOutputParser()

    rows, scores = [], []
    for i, case in enumerate(EVAL_SET):
        result = app.invoke(
            {"messages": [("user", case["question"])]},
            config={"configurable": {"thread_id": f"eval-{i}"}},
        )
        ans = result.get("answer", "")
        raw = judge.invoke({"q": case["question"], "ref": case["reference"], "ans": ans})
        score = int(next((c for c in raw if c.isdigit()), "3"))
        scores.append(score)
        rows.append((case["question"], score, ans))
        print(f"  [{score}/5] {case['question']}")

    avg = sum(scores) / len(scores)
    pass_rate = sum(s >= PASS for s in scores) / len(scores)

    # 控制台看板
    print("\n===== 评测看板 =====")
    print(f"样本数：{len(scores)}　平均正确性：{avg:.2f}/5　通过率(≥{PASS})：{pass_rate:.0%}")

    # 写 Markdown 报告
    lines = [
        "# Capstone 客服 Agent 评测报告\n",
        f"- 样本数：{len(scores)}",
        f"- 平均正确性：**{avg:.2f}/5**",
        f"- 通过率（≥{PASS}）：**{pass_rate:.0%}**\n",
        "| # | 问题 | 得分 | 答案摘要 |",
        "|---|------|------|----------|",
    ]
    for i, (q, s, ans) in enumerate(rows, 1):
        snippet = ans.replace("\n", " ")[:40]
        lines.append(f"| {i} | {q} | {s}/5 | {snippet}… |")
    report = pathlib.Path(__file__).parent / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"详细报告已写入：{report}")


if __name__ == "__main__":
    main()
