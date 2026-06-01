"""Capstone 后端配置：路径与可调参数集中管理。"""

import pathlib
import sys

# 仓库根加入路径，便于 import 根级 common 包
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CAPSTONE_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = CAPSTONE_DIR / "data"
FRONTEND_DIR = CAPSTONE_DIR / "frontend"

# 持久化记忆的 SQLite 文件（已被 .gitignore 忽略）
CHECKPOINT_DB = str(CAPSTONE_DIR / "memory.sqlite")

TOP_K = 3
