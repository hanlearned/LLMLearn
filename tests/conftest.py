"""
pytest 公共夹具。

设计目标：**不需要真实 API Key 也能跑大部分测试**。
- 设一个假 Key，让 ChatOpenAI / Embeddings 对象能被「构造」（构造期不联网，只有真正 .invoke 才联网）。
- 提供 load_module()：用 importlib 以唯一模块名加载项目文件，避免不同项目里同名 agent.py / config.py 撞车。
- 提供 fake_embeddings()：确定性假向量，让 RAG 管道（加载→切分→向量库→检索）无需联网即可端到端验证。
"""

import importlib.util
import os
import pathlib
import sys

# 假 Key：仅用于让对象可构造；不会真的发请求（除非测试显式 .invoke）。
os.environ.setdefault("SILICONFLOW_API_KEY", "sk-dummy-for-tests")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402


def load_module(unique_name: str, rel_path: str):
    """以唯一名加载某个 .py 文件为模块，规避同名文件互相覆盖。"""
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(unique_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_embeddings():
    """确定性假 Embedding：相同文本→相同向量。够用来验证向量库读写与检索流程是否跑通。"""
    from langchain_core.embeddings import DeterministicFakeEmbedding

    return DeterministicFakeEmbedding(size=256)
