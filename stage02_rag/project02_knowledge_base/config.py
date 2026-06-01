"""项目 2 配置：把可调参数集中管理，方便评测时做变量对照实验。"""

import pathlib
import sys

# 让本包内任意脚本都能 import 到仓库根级 common
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# 知识库源文件目录（复用 Stage 2 的示例数据；换成你自己的文档即可）
DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"

# 向量库持久化目录（已在 .gitignore 中忽略）
PERSIST_DIR = str(pathlib.Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "enterprise_kb"

# RAG 关键超参数 —— 评测时就是改这几个值做 A/B
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 4
