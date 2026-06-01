# 项目 5：多 Agent 内容创作工作流

> 用「编辑部分工」的思路自动产出高质量内容：规划、写作、审校、返工、定稿各司其职。这是 LangGraph 多节点 + 条件边循环的综合实战。
>
> 代码：`stage04_langgraph/project05_content_workflow/workflow.py`

---

## 一、需求与方案设计

### 为什么不用单个模型一把梭？
直接让模型「写一篇好文章」，质量飘忽、不可控。人类编辑部的做法是**分工 + 审校 + 返工**：策划定方向、写手出稿、编辑挑刺、不行就返工。把这套流程搬给 Agent，质量稳定得多。

### 角色与流程（每个角色 = 一个图节点）

```
        ┌──────────────── 不达标且未超返工次数 ────────────────┐
        ↓                                                      │
START → planner → writer → editor ──(质量门控 gate)──→ finalizer → END
        定大纲    写正文    打分+意见         达标 ↗
```

| 角色 | 温度 | 职责 |
|------|------|------|
| planner | 0.7 | 定 3 点大纲 |
| writer | 0.7 | 按大纲写正文，带编辑意见时返工 |
| editor | 0 | 打分(1-10) + 给一条具体修改意见 |
| finalizer | 0.7 | 配标题、整体润色 |

**关键设计**：editor 温度设 0（评审要稳定一致），writer 温度设 0.7（写作要创意）。同一个工作流里不同节点用不同温度，这是工程上的精细之处。

---

## 二、实现详解

### 难点 1：质量门控与返工循环
核心是 `gate` 路由函数 + 条件边：

```python
def gate(state):
    if state["score"] >= 8 or state["revisions"] >= MAX_REVISIONS:
        return "finalize"   # 达标 或 返工够了 → 定稿
    return "revise"         # 否则打回 writer

g.add_conditional_edges("editor", gate, {"revise": "writer", "finalize": "finalizer"})
```

`"revise" → "writer"` 这条边指回上游，就形成了「写→审→改→再审」的循环。`MAX_REVISIONS` 是止损阀门——没有它，遇到模型死活达不到 8 分就会无限返工。

### 难点 2：让 writer 知道「上一轮哪里不好」
state 里存了 `notes`（编辑意见）和上一版 `draft`。writer 节点检测到有 notes 就把它和旧稿拼进 prompt，实现「带着反馈改进」而不是从头瞎写。**状态在节点间传递上下文**，正是 StateGraph 的价值。

### 难点 3：解析编辑的打分
让 editor 输出严格的 `分数|意见` 格式再 split 解析，并用 try/except 兜底（模型偶尔不守格式）。这是和 LLM 打交道的常态防御。

---

## 三、运行

```bash
pip install langgraph
python stage04_langgraph/project05_content_workflow/workflow.py
```

会看到规划→写作→评审的全过程打印，不达标时自动返工，最后输出带标题的成品。

---

## 四、复盘与进阶
1. 加一个 **SEO 节点**：检查关键词密度、生成摘要。
2. 加 **人工介入**（见 03_persistence_hitl）：定稿前让人审一眼。
3. 把 editor 换成多个不同视角的评审（事实性、流畅度、合规），并行打分取均值。

## 五、面试怎么考
- **「多 Agent 比单 Agent 好在哪？」** → 分工让每个角色专注、可单独优化、可插入门控和人工介入；流程可观测可调试。代价是延迟和成本上升。
- **「怎么防止返工死循环？」** → 设最大返工次数 / `recursion_limit` 兜底，达标或超限即放行。
- **「这个图的状态怎么在节点间流转？」** → 共享 State（TypedDict），节点返回局部更新由框架合并；下游节点读上游写入的字段。
