# 深度研究 Agent（Deep Research）设计文档

> 代码：`projects_advanced/research_agent/research_agent.py`
> 技术栈：LangChain 0.3+ / LangGraph 0.2+ / DuckDuckGo（keyless）

---

## 一、需求与方案设计

### 1.1 我们要解决什么

「深度研究」是一类典型的复杂信息任务：用户抛出一个宽泛问题（如「2024 年 RAG 有哪些主要进展？」），
期望得到一份**有结构、有依据、可溯源**的综述，而不是模型一口气编出来的「看起来对」的段落。

要做好这件事，至少要拆成三步：

1. **规划**：把宽泛问题拆成几个可独立检索的子问题，避免「一锅炖」导致检索词太泛。
2. **检索 + 提炼**：对每个子问题联网搜索，把原始材料压成带要点、带来源的「发现」。
3. **综述**：把所有发现汇总成结构化报告，并标注引用来源。

### 1.2 为什么用显式 StateGraph，而不是单个 create_react_agent？

| 维度 | 单个自由 Agent（create_react_agent） | 显式 StateGraph（本项目） |
| --- | --- | --- |
| 流程控制 | 模型自己决定下一步，可能反复横跳、提前停 | 阶段顺序由图固定：规划→检索→综述，必然走完 |
| 可观测 | 中间步骤藏在 Agent 的内部循环里 | 每个节点独立打印、可单测、可断点 |
| 可扩展 | 想加「人工审核 / 质量评分」很别扭 | 加一个节点、连两条边即可 |
| 防失控 | 可能陷入工具调用死循环 | 无自由循环，token / 步数可预期 |
| 复盘 | 难定位是规划错还是检索错 | 每段输出落在状态字段上，问题定位清晰 |

核心结论：**当流程的「阶段」是确定的，就应该用编排（Orchestration）把它画成图，而不是把确定性
交给模型自由发挥。** 自由 Agent 适合「开放式探索」，显式图适合「有明确产线的复杂任务」。

### 1.3 状态与节点设计

状态（`TypedDict`）只保留四个字段，节点之间靠它传递数据：

```python
class ResearchState(TypedDict):
    question: str            # 原始研究问题
    sub_queries: list[str]   # plan 拆出的子问题
    findings: list[dict]     # research 产出：[{query, content, sources}]
    report: str              # synthesize 产出的综述
```

图结构：

```
START → plan → research → synthesize → END
```

每个节点是一个纯函数：读 state、返回「要更新哪些字段」的局部 dict，框架自动合并。

---

## 二、实现详解

### 2.1 plan：结构化拆解子问题

宽泛问题直接搜索效果差，先让 LLM 拆成 **恰好 3 个** 互补子问题（如「进展/方法」「代表性工作」
「挑战/趋势」三个侧面）。这里用 **Pydantic + `with_structured_output`** 约束输出，避免模型自由
发挥导致后续解析失败：

```python
class SubQueries(BaseModel):
    queries: list[str] = Field(..., min_length=3, max_length=3)

planner = prompt | llm.with_structured_output(SubQueries)
```

`min_length/max_length=3` 在 schema 层就把「必须 3 个」这件事交给模型遵守，比事后字符串切割稳。

### 2.2 research：检索 + 提炼，引用与防幻觉

对每个子问题：先 DuckDuckGo 搜索拿到 `[{title, link, snippet}]`，再把它们拼成「材料」交给
LLM 提炼。**防幻觉的三个手段都集中在这一段：**

1. **只给材料、不给自由**：system prompt 明确「严格只使用资料中出现的信息，不要编造；资料不足
   时明确说明『资料有限』」。
2. **来源随发现一起留存**：`findings` 里每条都带 `sources`（链接列表），综述阶段才能真正引用，
   而不是事后让模型「补一个看起来像来源的 URL」。
3. **低温度**：`get_llm(temperature=0.2)`，研究任务要稳定、复现性好。

### 2.3 synthesize：带引用的综述

把所有发现编号拼成上下文，要求模型：概述 → 分主题展开（每个主题对应一个子问题）→ 总结展望 →
文末「参考来源」去重列出所有链接，并在正文相应处用方括号标注来源链接。**报告的事实只能来自
上一阶段的发现**，模型在这里只做「组织与表达」，不引入新事实——这是把「检索」与「写作」职责
分离后的天然收益。

### 2.4 联网容错（无网不崩）

这是生产可用性的关键。所有外部调用都包 `try/except`，分两层降级：

```python
def web_search(query):
    try:
        from langchain_community.tools import DuckDuckGoSearchResults
        tool = DuckDuckGoSearchResults(num_results=4, output_format="list")
        return normalize(tool.invoke(query))
    except Exception as exc:        # 无网 / 限流 / 依赖缺失
        print(f"[搜索降级] ... {exc}")
        return []                   # 返回空，由调用方决定兜底
```

- **搜索失败** → 该子问题改用 `_fallback_chain`（模型已有知识），并强制在开头标注
  「（以下内容未经实时检索，可能过时）」，让读者知道这部分可信度较低。
- **plan 结构化失败** → 退化为「以原问题作为唯一子问题」，流程照样跑完。

效果：**有网时是带引用的实时研究；无网时退化为带免责声明的知识总结**，全程不抛异常中断。

延迟导入 `DuckDuckGoSearchResults`（放在函数内）也是有意为之：即使环境没装
`duckduckgo-search`，也只在真正搜索时才触发，不影响 import 阶段。

---

## 三、运行

```bash
# 1. 仓库根 .env 填任一厂商 Key（SiliconFlow / DeepSeek / Kimi / OpenAI）
pip install -r requirements.txt
# 2. 在仓库根目录运行
python projects_advanced/research_agent/research_agent.py
```

输出顺序：图结构 ASCII → 规划的 3 个子问题 → 逐个子问题的检索与提炼日志 → 最终综述报告。
**需要联网**；无网时会打印 `[搜索降级]` 并自动兜底（详见 §2.4）。

---

## 四、复盘与进阶

当前实现是「线性三段」的最小可用版本，工程上还能继续演进：

- **检索并行化**：`research` 节点目前串行遍历子问题，可改用 LangGraph 的 `Send` / fan-out
  把 3 个子问题并行检索，再 fan-in 汇总，显著降延迟。
- **质量回路**：在 `synthesize` 后加一个 `critique` 节点给报告打分，不达标则带反馈回到
  `research` 补检索（条件边 + 循环），这才是「Deep Research」真正的深度来源。
- **检索增强**：当前依赖搜索摘要（snippet）；可加一个「抓取正文」节点（如 WebFetch / 正文提取）
  让提炼基于全文而非摘要，发现质量更高。
- **去重与可信度排序**：对多个子问题命中的重复来源做去重，并按域名权威度排序后再引用。
- **持久化 / 中断恢复**：接入 `langgraph-checkpoint-sqlite`，长研究任务可断点续跑、可人工介入。
- **可观测**：接入 LangSmith 追踪每个节点的输入输出，便于定位「是规划差还是检索差」。

---

## 五、面试怎么考

- **Q：为什么不用一个 ReAct Agent 全包？**
  A：研究流程的阶段是确定的（规划→检索→综述），把确定性交给模型自由循环会带来不可控、难复盘、
  易死循环。用 StateGraph 显式编排，过程可读、可测、可扩展；自由 Agent 更适合开放式探索。

- **Q：怎么防止综述阶段编造来源 / 幻觉？**
  A：职责分离——检索阶段把 `link` 随发现一起存进状态，综述阶段只能引用已存在的来源；prompt 约束
  「只依据给定发现、不编造」；低温度；资料不足时要求模型如实声明。

- **Q：`with_structured_output` 解决了什么？**
  A：让规划阶段稳定产出「恰好 3 个子问题」的结构化结果，schema 层（Pydantic min/max length）兜底，
  避免用字符串切割解析自由文本的脆弱性。

- **Q：网络不可用会怎样？怎么保证生产可用？**
  A：所有外部调用 try/except；搜索失败降级为模型知识并加免责声明，规划失败退化为单子问题，
  全程不中断。延迟导入第三方搜索库，import 阶段也不受依赖缺失影响。

- **Q：如何让它「更深」？**
  A：加 critique 评分节点 + 条件边形成「检索-评估-补检索」循环；子问题并行 fan-out/fan-in；
  抓取正文而非仅摘要；接 checkpoint 支持长任务断点续跑。

- **Q：LangGraph 的 State / Node / Edge 各是什么？**
  A：State 是节点共享的状态结构（TypedDict）；Node 是读 state、返回局部更新 dict 的函数；
  Edge 决定执行顺序，START/END 为入口出口；compile 后才能 invoke/stream。
