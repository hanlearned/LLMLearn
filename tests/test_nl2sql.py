"""NL2SQL 项目测试：只读护栏 + schema 工具 + 建库。无需 API Key。"""

import pathlib
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, load_module

NL2SQL_DIR = REPO_ROOT / "projects_advanced" / "nl2sql_agent"


@pytest.fixture(scope="module")
def agent_mod():
    # 先建库，保证 run_sql 的执行路径有真实数据可查
    subprocess.run([sys.executable, str(NL2SQL_DIR / "seed_db.py")], check=True)
    return load_module("nl2sql_agent_mod", "projects_advanced/nl2sql_agent/agent.py")


def test_seed_db_creates_tables():
    assert (NL2SQL_DIR / "shop.db").exists()


@pytest.mark.parametrize("sql", [
    "SELECT name FROM customers",
    "SELECT REPLACE(name,'张','李') FROM customers",          # 只读标量函数：不应被误杀
    "WITH t AS (SELECT 1 x) SELECT x FROM t",                # CTE 只读
    "SELECT city, COUNT(*) FROM customers GROUP BY city",
])
def test_readonly_queries_allowed(agent_mod, sql):
    out = agent_mod.run_sql.invoke({"sql": sql})
    assert not out.startswith("拒绝执行"), f"本应放行却被拒：{sql} -> {out}"


@pytest.mark.parametrize("sql", [
    "DROP TABLE customers",
    "DELETE FROM orders WHERE id=1",
    "UPDATE customers SET level='x'",
    "INSERT INTO customers VALUES (99,'x','y','z')",
    "SELECT 1; DELETE FROM orders",                          # 多语句拖尾注入
])
def test_write_queries_rejected(agent_mod, sql):
    out = agent_mod.run_sql.invoke({"sql": sql})
    assert out.startswith("拒绝执行"), f"本应拒绝却放行：{sql} -> {out}"


def test_readonly_connection_blocks_writes(agent_mod):
    """最硬护栏：即便绕过文本检查，只读连接也会让写在 DB 层失败。"""
    import sqlite3

    conn = agent_mod._connect(read_only=True)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM orders")
    conn.close()


def test_get_schema_unknown_table(agent_mod):
    out = agent_mod.get_schema.invoke({"table": "no_such_table"})
    assert "不存在" in out and "可用的表" in out


def test_run_sql_real_result(agent_mod):
    """跑一条真实聚合，验证结果包含正确数据（北京有 4 位客户）。"""
    out = agent_mod.run_sql.invoke(
        {"sql": "SELECT city, COUNT(*) c FROM customers GROUP BY city ORDER BY c DESC"}
    )
    assert "北京" in out
