"""
SQLite 兼容垫片：解决「chromadb 要求 sqlite3 >= 3.35，但系统自带的太老」的常见报错。

很多 Linux 服务器/旧系统自带的 sqlite3 还停留在 3.31 之类的老版本，
直接 import chromadb 会抛：
    RuntimeError: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0

业界标准解法：装 pysqlite3-binary（自带新版 sqlite），在 import chromadb 之前把它顶替成 sqlite3。

⚠️ 用法：在**任何会 import chromadb / langchain_chroma 的脚本里，于那行 import 之前**先
    `import common.sqlite_compat  # noqa: F401`
本模块被 import 时立即执行顶替；新系统没装 pysqlite3 也不会报错（静默跳过）。
"""

import sys

try:
    import pysqlite3  # 来自 pysqlite3-binary

    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite3.dbapi2"] = pysqlite3.dbapi2
except ImportError:
    # 没装 pysqlite3-binary：新系统的内置 sqlite3 通常已 >= 3.35，无需顶替。
    pass
