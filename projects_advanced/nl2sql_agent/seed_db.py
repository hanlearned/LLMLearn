"""
NL2SQL 项目 · 建库脚本

用 Python 标准库 sqlite3 创建一个迷你电商库 shop.db，包含三张表：
  - customers：客户（id, name, city, level）
  - products ：商品（id, name, category, price）
  - orders   ：订单（id, customer_id, product_id, quantity, amount, order_date）

数据是「教学用的小样本」，但外键关联都对得上，金额 = 单价 × 数量，
方便后面让 Agent 用真实 SQL 跑出真实答案。

运行：
    python projects_advanced/nl2sql_agent/seed_db.py
运行后会在本目录生成 shop.db，并打印「建库完成」。
"""

import pathlib
import sqlite3

# 把 db 文件固定放在本脚本所在目录下（无论你在哪个工作目录运行，路径都正确）
DB_PATH = pathlib.Path(__file__).parent / "shop.db"


# ------------------------------------------------------------------
# 示例数据：故意设计得「有规律可问」——城市分布不均、品类有高低单价、
# 订单跨越多个月份，这样自然语言问题（最多城市/最高销售额/某月总额）才有意义。
# ------------------------------------------------------------------
CUSTOMERS = [
    # (id, 姓名, 城市, 会员等级)
    (1, "张伟", "北京", "黄金"),
    (2, "李娜", "上海", "白银"),
    (3, "王芳", "北京", "黄金"),
    (4, "刘洋", "深圳", "普通"),
    (5, "陈静", "北京", "白银"),
    (6, "杨过", "广州", "黄金"),
    (7, "赵敏", "上海", "普通"),
    (8, "周杰", "北京", "普通"),
    (9, "孙莉", "杭州", "白银"),
    (10, "吴磊", "成都", "普通"),
    (11, "郑爽", "上海", "黄金"),
    (12, "钱进", "深圳", "白银"),
    (13, "冯巩", "北京", "普通"),
    (14, "蒋雯", "广州", "白银"),
    (15, "韩寒", "杭州", "黄金"),
]

PRODUCTS = [
    # (id, 商品名, 品类, 单价)
    (1, "机械键盘", "数码配件", 399.0),
    (2, "无线鼠标", "数码配件", 129.0),
    (3, "降噪耳机", "数码配件", 899.0),
    (4, "保温杯", "家居生活", 89.0),
    (5, "羊毛围巾", "服饰", 199.0),
    (6, "笔记本电脑", "数码", 6499.0),
    (7, "智能手表", "数码", 1299.0),
    (8, "蓝牙音箱", "数码配件", 349.0),
    (9, "记事本", "文具", 25.0),
    (10, "运动鞋", "服饰", 459.0),
    (11, "空气炸锅", "家居生活", 399.0),
    (12, "机械表", "服饰", 2999.0),
]

# 订单：amount 统一按「商品单价 × 数量」计算，避免手填出错
_PRICE = {pid: price for pid, _name, _cat, price in PRODUCTS}
_RAW_ORDERS = [
    # (id, 客户id, 商品id, 数量, 下单日期)  —— 覆盖 2024 全年 12 个月，便于做时间维度聚合
    (1, 1, 6, 1, "2024-01-15"), (2, 1, 1, 2, "2024-01-28"),
    (3, 2, 3, 1, "2024-02-03"), (4, 9, 7, 1, "2024-02-14"), (5, 3, 2, 3, "2024-02-25"),
    (6, 3, 6, 1, "2024-03-05"), (7, 4, 5, 2, "2024-03-12"), (8, 5, 4, 4, "2024-03-22"),
    (9, 11, 12, 1, "2024-03-28"),
    (10, 6, 3, 2, "2024-04-02"), (11, 7, 1, 1, "2024-04-11"), (12, 10, 9, 5, "2024-04-19"),
    (13, 1, 3, 1, "2024-05-06"), (14, 12, 7, 1, "2024-05-15"), (15, 3, 6, 1, "2024-05-20"),
    (16, 5, 10, 1, "2024-05-25"),
    (17, 2, 11, 1, "2024-06-01"), (18, 6, 6, 1, "2024-06-08"), (19, 13, 2, 1, "2024-06-15"),
    (20, 8, 8, 2, "2024-07-03"), (21, 14, 5, 3, "2024-07-19"), (22, 15, 6, 1, "2024-07-28"),
    (23, 9, 11, 1, "2024-08-05"), (24, 11, 3, 1, "2024-08-17"), (25, 4, 8, 1, "2024-08-29"),
    (26, 1, 7, 1, "2024-09-09"), (27, 7, 9, 10, "2024-09-21"),
    (28, 15, 12, 1, "2024-10-02"), (29, 10, 10, 1, "2024-10-15"), (30, 3, 3, 1, "2024-10-25"),
    (31, 12, 1, 2, "2024-11-06"), (32, 6, 6, 1, "2024-11-11"), (33, 2, 4, 3, "2024-11-24"),
    (34, 5, 6, 1, "2024-12-01"), (35, 14, 7, 1, "2024-12-12"),
    (36, 8, 5, 1, "2024-12-20"), (37, 11, 6, 1, "2024-12-28"),
]
ORDERS = [
    (oid, cid, pid, qty, round(_PRICE[pid] * qty, 2), date)
    for (oid, cid, pid, qty, date) in _RAW_ORDERS
]


def main() -> None:
    # 若已存在旧库则删掉重建，保证每次运行结果一致（幂等）
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()

        # --- 建表（带外键约束，体现规范的关系型设计）---
        cur.executescript(
            """
            CREATE TABLE customers (
                id    INTEGER PRIMARY KEY,
                name  TEXT    NOT NULL,
                city  TEXT    NOT NULL,
                level TEXT    NOT NULL
            );

            CREATE TABLE products (
                id       INTEGER PRIMARY KEY,
                name     TEXT    NOT NULL,
                category TEXT    NOT NULL,
                price    REAL    NOT NULL
            );

            CREATE TABLE orders (
                id          INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                product_id  INTEGER NOT NULL,
                quantity    INTEGER NOT NULL,
                amount      REAL    NOT NULL,
                order_date  TEXT    NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(id),
                FOREIGN KEY (product_id)  REFERENCES products(id)
            );
            """
        )

        # --- 填数据（参数化插入，养成防 SQL 注入的好习惯）---
        cur.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", CUSTOMERS)
        cur.executemany("INSERT INTO products  VALUES (?, ?, ?, ?)", PRODUCTS)
        cur.executemany("INSERT INTO orders    VALUES (?, ?, ?, ?, ?, ?)", ORDERS)

        conn.commit()

        # 简单回显，方便确认数据量
        n_c = cur.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        n_p = cur.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        n_o = cur.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        print(f"建库完成：customers={n_c} 行，products={n_p} 行，orders={n_o} 行")
        print(f"数据库文件：{DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
