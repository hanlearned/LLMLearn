"""
LLMLearn 全局共享基础设施。

为什么放在仓库根目录而不是每个 Stage 各放一份？
- Stage 2 之后每个阶段都要用到「拿一个 LLM」「拿一个 Embedding」「拿一个向量库」，
  逻辑完全一致，复制 5 份是维护灾难。
- 这里做成一个根级 `common` 包，所有阶段脚本通过一行 path 注入即可复用：

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # 仓库根
    from common.llm_provider import get_llm

  （Stage 1 历史代码仍保留它自己的 stage01_basics/common，互不影响。）
"""
