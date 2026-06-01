"""
高级实战 · Agent/RAG 自动化评测系统 —— 评测主程序（Harness）

这是一个「通用评测框架」的最小可用实现，核心抽象只有两个：
    1. 评测集 eval_set：一批 {question, reference, context?}。
    2. 被测系统 target_system：一个 (question) -> answer 的函数。

只要把 target_system 换成你自己的 Agent / RAG 链，这套 harness 就能复用：
对每条样本生成答案 -> 调裁判多维度打分 -> 聚合平均分/通过率 -> 产出 Markdown + JSON 报告。

运行：
    python projects_advanced/agent_eval_system/harness.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
# 本文件位于 projects_advanced/agent_eval_system/harness.py，向上 2 层即仓库根目录。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common.llm_provider import get_llm
from projects_advanced.agent_eval_system.judge import judge_one

# 当前目录，用于定位评测集与报告输出。
HERE = pathlib.Path(__file__).parent
EVAL_SET_PATH = HERE / "eval_set.json"
REPORT_MD_PATH = HERE / "report.md"
REPORT_JSON_PATH = HERE / "report.json"

# 评分维度（顺序固定，便于报告列对齐）。
DIMENSIONS = ["correctness", "faithfulness", "relevance"]
# 通过阈值：某维度得分 >= 该值才算「该维度通过」；一条样本三维全通过才算整体通过。
PASS_THRESHOLD = 4


# ---------------------------------------------------------------------------
# 1. 被测系统（demo）：一个 system prompt 固定的裸 LLM。
#    真实项目里请把这里换成你的 RAG 链 / Agent 的 invoke 封装，签名保持 (str) -> str 即可。
# ---------------------------------------------------------------------------
_TARGET_SYSTEM_PROMPT = (
    "你是一名 LangChain / LLM 应用开发助教，请用准确、简洁的中文回答关于 "
    "LLM、RAG、Agent、LangChain 的基础问题。只回答与问题直接相关的内容，不要展开无关话题。"
)
_target_prompt = ChatPromptTemplate.from_messages(
    [("system", _TARGET_SYSTEM_PROMPT), ("human", "{question}")]
)
# 被测系统温度也设 0，让 demo 结果稳定可复现（实际系统可保留自身温度）。
_target_chain = _target_prompt | get_llm(temperature=0) | StrOutputParser()


def target_system(question: str) -> str:
    """被测系统：输入问题，输出答案字符串。这是评测框架与系统之间的唯一契约。"""
    return _target_chain.invoke({"question": question})


# ---------------------------------------------------------------------------
# 2. 评测主流程。
# ---------------------------------------------------------------------------
def load_eval_set() -> list[dict]:
    """加载评测集 JSON。"""
    return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))


def run_eval(eval_set: list[dict]) -> list[dict]:
    """
    遍历评测集：对每条生成答案并打分，返回逐条结果列表。
    每条结果含：question / reference / answer / 三维分数+理由 / 是否整体通过。
    """
    records: list[dict] = []
    total = len(eval_set)
    for i, sample in enumerate(eval_set, start=1):
        question = sample["question"]
        reference = sample["reference"]
        print(f"[{i}/{total}] 评测中：{question[:30]}...")

        # 2.1 调被测系统拿答案。单条异常不应中断整批评测。
        try:
            answer = target_system(question)
        except Exception as exc:  # noqa: BLE001 —— 评测框架需对被测系统的任意异常鲁棒
            answer = f"[被测系统调用失败] {exc}"

        # 2.2 调裁判打分。
        score = judge_one(question=question, answer=answer, reference=reference)

        # 2.3 单条是否整体通过：三个维度都 >= 阈值。
        passed = all(getattr(score, dim) >= PASS_THRESHOLD for dim in DIMENSIONS)

        records.append(
            {
                "question": question,
                "reference": reference,
                "answer": answer,
                "correctness": score.correctness,
                "correctness_reason": score.correctness_reason,
                "faithfulness": score.faithfulness,
                "faithfulness_reason": score.faithfulness_reason,
                "relevance": score.relevance,
                "relevance_reason": score.relevance_reason,
                "passed": passed,
            }
        )
    return records


def aggregate(records: list[dict]) -> dict:
    """聚合：各维度平均分、总平均分、整体通过率。"""
    n = len(records) or 1  # 防除零
    dim_avg = {dim: round(sum(r[dim] for r in records) / n, 2) for dim in DIMENSIONS}
    overall_avg = round(sum(dim_avg.values()) / len(DIMENSIONS), 2)
    pass_count = sum(1 for r in records if r["passed"])
    pass_rate = round(pass_count / n, 4)
    return {
        "num_samples": len(records),
        "dimension_avg": dim_avg,
        "overall_avg": overall_avg,
        "pass_threshold": PASS_THRESHOLD,
        "pass_count": pass_count,
        "pass_rate": pass_rate,
    }


# ---------------------------------------------------------------------------
# 3. 报告产出。
# ---------------------------------------------------------------------------
def _truncate(text: str, limit: int = 120) -> str:
    """报告表格里压缩长文本，避免单元格过宽；并把换行/竖线转义以防破坏 Markdown 表格。"""
    text = text.replace("\n", " ").replace("|", "\\|")
    return text if len(text) <= limit else text[:limit] + "…"


def render_markdown(records: list[dict], summary: dict) -> str:
    """生成 Markdown 报告文本：汇总 + 逐条明细表格。"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# Agent/RAG 自动化评测报告")
    lines.append("")
    lines.append(f"- 生成时间：{ts}")
    lines.append(f"- 样本数：{summary['num_samples']}")
    lines.append(f"- 通过阈值：每个维度分数 >= {summary['pass_threshold']}（三维全过才算整体通过）")
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    for dim in DIMENSIONS:
        lines.append(f"| 平均 {dim} | {summary['dimension_avg'][dim]} |")
    lines.append(f"| 总平均分 | {summary['overall_avg']} |")
    lines.append(f"| 通过数 | {summary['pass_count']} / {summary['num_samples']} |")
    lines.append(f"| 通过率 | {summary['pass_rate'] * 100:.1f}% |")
    lines.append("")
    lines.append("## 逐条明细")
    lines.append("")
    header = (
        "| # | 问题 | 答案(截断) | 正确性 | 忠实度 | 相关性 | 整体 | 主要理由(截断) |"
    )
    lines.append(header)
    lines.append("| --- | --- | --- | :---: | :---: | :---: | :---: | --- |")
    for i, r in enumerate(records, start=1):
        overall = "PASS" if r["passed"] else "FAIL"
        reason = _truncate(r["correctness_reason"], 80)
        lines.append(
            f"| {i} | {_truncate(r['question'], 40)} | {_truncate(r['answer'], 60)} | "
            f"{r['correctness']} | {r['faithfulness']} | {r['relevance']} | {overall} | {reason} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_reports(records: list[dict], summary: dict) -> None:
    """把结果写入 report.md 与 report.json。"""
    REPORT_MD_PATH.write_text(render_markdown(records, summary), encoding="utf-8")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "records": records,
    }
    REPORT_JSON_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _print_summary(summary: dict) -> None:
    """终端打印汇总。"""
    print("\n" + "=" * 48)
    print("评测汇总")
    print("=" * 48)
    print(f"样本数        : {summary['num_samples']}")
    for dim in DIMENSIONS:
        print(f"平均 {dim:<12}: {summary['dimension_avg'][dim]}")
    print(f"总平均分      : {summary['overall_avg']}")
    print(
        f"通过率        : {summary['pass_rate'] * 100:.1f}% "
        f"({summary['pass_count']}/{summary['num_samples']}，阈值>= {summary['pass_threshold']})"
    )
    print("=" * 48)


def main() -> None:
    eval_set = load_eval_set()
    records = run_eval(eval_set)
    summary = aggregate(records)
    write_reports(records, summary)
    _print_summary(summary)
    print(f"\n报告已写入：\n  - {REPORT_MD_PATH}\n  - {REPORT_JSON_PATH}")


if __name__ == "__main__":
    main()
