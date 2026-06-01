"""
NL2SQL 数据问答 Agent —— 用自然语言查数据库

业务场景：运营/老板不会写 SQL，但想随口问「3 月订单总额多少？」「哪个城市客户最多？」。
本项目把数据库读能力封装成几个 @tool，交给 LangGraph 的 create_react_agent，
让模型自己「看结构 → 写 SQL → 跑 → 出错则改」，最后用中文给出带数字的结论。

两大生产级要点（文档里重点讲）：
  1) 只读护栏：run_sql 只放行 SELECT，任何写/改表语句一律拒绝——这是把「让模型直连数据库」
     从危险变可控的关键。
  2) 错误自愈：SQL 跑错时，把数据库的真实报错原样返回给模型，它能据此重写 SQL 再试一次。

前置：先建库
    python projects_advanced/nl2sql_agent/seed_db.py
运行本 Agent：
    python projects_advanced/nl2sql_agent/agent.py
"""

import pathlib
import re
import sqlite3
import sys

from dotenv import load_dotenv

load_dotenv()
# 本文件深度：projects_advanced/nl2sql_agent/agent.py —— 要回退 2 层才到仓库根
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from common.llm_provider import get_llm

# 数据库路径：与 seed_db.py 生成的位置一致（都在本目录）
DB_PATH = pathlib.Path(__file__).parent / "shop.db"

# 一次查询最多返回多少行——防止「SELECT *」把上万行塞进上下文，既费 token 又拖慢响应
MAX_ROWS = 50


def _connect(read_only: bool = True) -> sqlite3.Connection:
    """打开数据库连接。默认**只读**——这是最硬的护栏：即便有写语句漏过文本检查，
    数据库层也会直接拒绝写入。库不存在时给出清晰指引，而不是抛底层异常。"""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"找不到数据库 {DB_PATH}，请先运行：python projects_advanced/nl2sql_agent/seed_db.py"
        )
    if read_only:
        # file: URI + mode=ro 打开只读连接，任何 INSERT/UPDATE/DELETE 都会在 DB 层报错
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 让查询结果能按列名访问，便于格式化
    return conn


# ==================================================================
# 工具集：探查结构（list_tables / get_schema）+ 只读执行（run_sql）
# ==================================================================
@tool
def list_tables() -> str:
    """列出数据库里所有的表名。写 SQL 前应先调用它，搞清楚有哪些表。"""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = [r["name"] for r in rows]
        return "数据库中的表：" + ", ".join(names) if names else "数据库中暂无任何表。"
    finally:
        conn.close()


@tool
def get_schema(table: str) -> str:
    """返回指定表的列定义（列名、类型、是否非空、是否主键）。
    写涉及某表的 SQL 前，先用它确认列名，避免拼错字段。
    table: 表名，必须是 list_tables 返回的名字之一。
    """
    conn = _connect()
    try:
        # 先校验表是否存在，避免对不存在的表执行 PRAGMA 返回空、让模型一头雾水
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            tables = [
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            return f"错误：不存在名为 '{table}' 的表。可用的表有：{', '.join(tables)}"

        # PRAGMA table_info 返回每列：cid, name, type, notnull, dflt_value, pk
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        lines = [f"表 {table} 的列定义："]
        for c in cols:
            flags = []
            if c["pk"]:
                flags.append("主键")
            if c["notnull"]:
                flags.append("非空")
            suffix = f"（{', '.join(flags)}）" if flags else ""
            lines.append(f"  - {c['name']}: {c['type']}{suffix}")
        return "\n".join(lines)
    finally:
        conn.close()


@tool
def run_sql(sql: str) -> str:
    """执行一条**只读**的 SQL 查询并返回结果（最多 50 行）。
    只允许 SELECT 查询；任何写操作（INSERT/UPDATE/DELETE/DROP/ALTER/CREATE 等）都会被拒绝。
    若 SQL 写错导致报错，会把数据库的错误信息原样返回，你可以据此修正后重试。
    sql: 一条完整的 SELECT 语句。
    """
    cleaned = sql.strip().rstrip(";").strip()

    # --- 护栏 1：拒绝多语句（防 `SELECT 1; DELETE ...` 这类拖尾注入）---
    # rstrip 掉结尾分号后，正文里若还有分号，说明是多条语句拼接，直接拒绝。
    if ";" in cleaned:
        return "拒绝执行：检测到多条语句（包含分号），只允许单条只读 SELECT 查询。"

    # --- 护栏 2：必须以 SELECT 或 WITH（CTE 也是只读查询）开头 ---
    head = cleaned.lstrip("(").lstrip()  # 兼容 (SELECT ...) 这种带括号的写法
    if not re.match(r"(?i)^(SELECT|WITH)\b", head):
        return "拒绝执行：只允许 SELECT 查询，请改写为只读的 SELECT 语句。"

    # --- 护栏 3（最硬）：用只读连接执行；即便上面没拦住，写操作也会在 DB 层失败 ---
    # 注意：这样 REPLACE()/IIF() 等只读标量函数能正常用，不会被关键字黑名单误杀。
    conn = _connect(read_only=True)
    try:
        cur = conn.execute(cleaned)
        rows = cur.fetchmany(MAX_ROWS + 1)  # 多取一行用于判断是否被截断
        if not rows:
            return "查询成功，但没有匹配的数据行。"

        col_names = [d[0] for d in cur.description]
        truncated = len(rows) > MAX_ROWS
        rows = rows[:MAX_ROWS]

        # 拼成可读文本表：表头 + 每行
        out = [" | ".join(col_names)]
        for r in rows:
            out.append(" | ".join("" if v is None else str(v) for v in r))
        result = "\n".join(out)
        if truncated:
            result += f"\n...（结果超过 {MAX_ROWS} 行，已截断，请加 LIMIT 或聚合后再查）"
        return result
    except sqlite3.Error as e:
        # 关键：不要把异常吞掉，原样回传，模型才能「看懂哪里错了」并重写
        return f"SQL 执行出错：{e}。请检查表名/列名是否正确，修正后重试。"
    finally:
        conn.close()


# ------------------------------------------------------------------
# System Prompt：把「正确的工作流」写进去，引导 Agent 少走弯路
# ------------------------------------------------------------------
SYSTEM_PROMPT = """你是一个严谨的数据库数据分析助手，帮助不会写 SQL 的用户用自然语言查询数据。

工作流程（请严格遵守）：
1. 拿到问题后，先调用 list_tables 了解有哪些表。
2. 对可能用到的表，调用 get_schema 确认它的列名和类型，不要凭空假设字段名。
3. 基于真实结构编写**只读的 SELECT 语句**，调用 run_sql 执行。
4. 如果 run_sql 返回了报错信息，说明你的 SQL 写错了——仔细读错误，修正 SQL 后重试，不要放弃。
5. 拿到查询结果后，用简洁中文给出结论，并带上具体数字。

硬性约束：
- 你只能查询（SELECT），绝不能尝试修改数据或表结构。
- 一切结论必须以 run_sql 的真实返回为准，严禁编造数据。
- 涉及金额求和用 amount 列；按月筛选可用 strftime('%m', order_date) 或 order_date LIKE '2024-03%'。
"""


def build_agent():
    """一行构建 NL2SQL ReAct Agent。

    - model：temperature=0，让「写哪条 SQL」尽量确定、可复现
    - tools：探查结构 + 只读执行
    - prompt：把工作流与只读约束写死，降低出错率
    ReAct 的「决定→调用→喂回→再决定」循环由 LangGraph 自动驱动。
    """
    return create_react_agent(
        model=get_llm(temperature=0),
        tools=[list_tables, get_schema, run_sql],
        prompt=SYSTEM_PROMPT,
    )


def ask(agent, question: str) -> None:
    """问一个问题，并打印「工具调用轨迹 + 最终答案」，方便观察 Agent 怎么想的。"""
    print(f"\n{'=' * 64}\n👤 {question}")
    result = agent.invoke({"messages": [("user", question)]})

    # 打印轨迹：模型每一次工具调用（含参数）、以及工具返回了什么
    for m in result["messages"]:
        if getattr(m, "tool_calls", None):
            for c in m.tool_calls:
                print(f"   🔧 调用 {c['name']}({c['args']})")
        elif type(m).__name__ == "ToolMessage":
            # 工具返回可能很长，截断展示前 120 字符
            preview = str(m.content).replace("\n", " ⏎ ")
            print(f"   ↩️  返回 {preview[:120]}{'…' if len(preview) > 120 else ''}")

    print(f"🤖 {result['messages'][-1].content}")


DEMO_QUESTIONS = [
    "哪个城市的客户最多？",
    "销售额最高的商品是什么？卖了多少钱？",
    "2024 年第一季度（1-3 月）的订单总额是多少？",
    "哪位客户的累计消费金额最高？",
    "各品类的销售额分别是多少？从高到低排。",
]


def main():
    import sys

    agent = build_agent()
    # 带 --demo 或无参时跑演示；带 --chat 进入交互式问答
    if "--chat" in sys.argv:
        print("NL2SQL 交互模式（输入问题，q 退出）。例如：哪个城市客户最多？")
        while True:
            try:
                q = input("\n你问：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in {"q", "quit", "exit"}:
                break
            if q:
                ask(agent, q)
    else:
        # 演示：覆盖分组计数、销售额排序、季度聚合、跨表关联、分组排序
        for q in DEMO_QUESTIONS:
            ask(agent, q)
        print("\n💡 想自己提问？运行：python projects_advanced/nl2sql_agent/agent.py --chat")


if __name__ == "__main__":
    main()
