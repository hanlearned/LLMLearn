# 深度研究 Agent（Deep Research）

一个用 **LangGraph StateGraph** 实现的多步研究流程：给定一个研究问题，自动完成
「**规划子问题 → 联网检索 → 综合成报告**」，产出一篇结构化、带引用来源的中文综述。

与「单个自由 Agent（create_react_agent）」不同，这里把研究流程**显式编排**成一张状态图，
每个阶段是一个节点，过程可读、可测、可观测。

## 流程

```
START → plan → research → synthesize → END
```

- **plan**：把研究问题用 LLM 结构化输出拆成 3 个互补子问题。
- **research**：对每个子问题做 DuckDuckGo 搜索，再让 LLM 提炼成带要点、带来源链接的「发现」。
- **synthesize**：综合所有发现，写一篇带引用来源的中文综述。

状态字段：`question`、`sub_queries`、`findings`、`report`。

## 运行

```bash
# 1. 在仓库根目录的 .env 填入任一厂商 Key（SiliconFlow / DeepSeek / Kimi / OpenAI）
# 2. 安装依赖（已在根 requirements.txt 中）
pip install -r requirements.txt
# 3. 运行（在仓库根目录执行）
python projects_advanced/research_agent/research_agent.py
```

程序会先打印图结构（ASCII），再分阶段打印「规划 → 检索 → 综述」的过程，最后输出综述报告。

## 是否需要联网？

**需要联网。** 检索阶段使用 keyless 的 **DuckDuckGo** 搜索（`duckduckgo-search`，无需 API Key）。

**无网/搜索失败时的降级行为（不会崩溃）：**

- 单个子问题搜索失败 → 打印 `[搜索降级]` 提示，改用**模型已有知识**作答，并在该发现开头
  标注「（以下内容未经实时检索，可能过时）」。
- `plan` 结构化拆解失败 → 退化为「以原问题作为唯一子问题」继续跑。

所有外部调用都包在 `try/except` 中，保证**无网也能跑完整条流水线**（只是引用质量下降）。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `research_agent.py` | StateGraph 实现：`plan` / `research` / `synthesize` 三节点 + 搜索容错 + `__main__` 示例 |
| `README.md` | 本文件 |

配套设计文档见 `docs/advanced/research_agent.md`。
