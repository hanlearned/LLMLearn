# NL2SQL 数据问答 Agent

用**自然语言**查询数据库的高级实战项目。运营/老板不会写 SQL，直接问
「哪个城市的客户最多？」「3 月订单总额多少？」，Agent 自己看表结构、写 SQL、
执行、出错自愈，最后用中文给出带数字的结论。

技术栈：LangChain 0.3+ / LangGraph 0.2+（`create_react_agent`）+ Python 标准库 `sqlite3`。

## 核心看点

1. **只读护栏（三层防御）**：① 拒绝多语句（防 `SELECT 1; DROP...` 拖尾注入）；② 必须 `SELECT`/`WITH`
   开头；③ 用**只读连接**（`mode=ro`）执行，任何写操作在 SQLite 层直接失败。比纯关键字黑名单更可靠
   （也不会误杀 `REPLACE()` 等只读函数）。
2. **错误自愈**：SQL 写错时，把数据库的真实报错原样回传给模型，它能据此重写 SQL 再试。
3. **结果截断**：单次查询最多返回 50 行，避免 `SELECT *` 把上万行塞进上下文。
4. **有测试**：`tests/test_nl2sql.py` 参数化覆盖护栏放行/拒绝、只读连接兜底、schema、真实聚合。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `seed_db.py` | 用 sqlite3 建库并填充 `shop.db`（customers / products / orders 三表） |
| `agent.py`   | NL2SQL Agent：装备 `list_tables` / `get_schema` / `run_sql` 三个工具 |
| `shop.db`    | 运行 `seed_db.py` 后生成的 SQLite 数据库文件 |
| `README.md`  | 本文件 |

## 运行步骤

```bash
# 1. 在仓库根目录的 .env 里填好任意一个厂商的 API Key（见 .env.example）

# 2. 先建库（生成 shop.db：15 客户 / 12 商品 / 37 订单，覆盖全年 12 个月、5 个品类）
python projects_advanced/nl2sql_agent/seed_db.py

# 3a. 演示模式（自动跑 5 个自然语言问题）
python projects_advanced/nl2sql_agent/agent.py

# 3b. 交互模式（自己提问）
python projects_advanced/nl2sql_agent/agent.py --chat
```

运行 `agent.py` 后，每个问题都会打印 **工具调用轨迹**（调了哪些工具、传了什么参数、
返回了什么）和 **最终中文答案**，方便你观察 Agent 的推理过程。

## 数据模型

- `customers(id, name, city, level)`：客户，含城市与会员等级
- `products(id, name, category, price)`：商品，含品类与单价
- `orders(id, customer_id, product_id, quantity, amount, order_date)`：订单，外键关联前两表

更深入的设计与复盘见：`docs/advanced/nl2sql_agent.md`。
