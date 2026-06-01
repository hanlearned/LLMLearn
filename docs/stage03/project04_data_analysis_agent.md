# 项目 4：智能数据分析助手（ChatBI 雏形）

> 让不会写 SQL/pandas 的业务同学，用大白话问数据：「3 月哪个区域卖得最好？」。这是 NL2SQL / ChatBI 赛道最核心的能力，也是 Agent + Tool Calling 最经典的落地场景。
>
> 代码：`stage03_agent/project04_data_analysis/`

---

## 一、需求与方案设计

### 业务目标
用户用自然语言提数据问题，系统自动算出答案并给出结论（带具体数字）。

### 核心难点：怎么把「自然语言」变成「数据计算」？
两条路线：

| 路线 | 做法 | 风险 |
|------|------|------|
| ❌ 给模型一个 Python 执行器 | 让模型生成 `df.groupby(...)` 并 `eval` 执行 | **代码注入**：模型可能生成 `import os; os.system(...)`，生产大忌 |
| ✅ 封装受控结构化工具 | 预先写好 `aggregate / filter` 等安全工具，模型只能选工具+传参 | 参数被类型和枚举约束，安全可控 |

本项目走**第二条**：把数据能力拆成几个职责单一、参数受控的工具，Agent 负责「读问题→选工具→传参→解释结果」。这是生产级 NL2Analytics 的主流安全取舍。

### 架构
```
用户问题 → LangGraph ReAct Agent ──┬─ get_schema()    了解数据结构
                                    ├─ aggregate()     分组聚合统计
                                    └─ filter_by_month() 按月筛选
                                          ↓
                                    工具返回真实计算结果 → Agent 总结成中文结论
```

---

## 二、实现详解

### 难点 1：工具怎么设计才「又安全又够用」
- `aggregate(metric, operation, group_by)`：metric 限定 `units/revenue`，operation 限定 `sum/mean/max/...`，非法值直接返回错误提示。模型即使瞎传也越不出边界。
- docstring 写得像「使用说明书」：明确每个参数能取什么值。**模型选工具、传参完全靠 docstring**，写不清楚模型就会传错。

### 难点 2：引导 Agent「先看数据再动手」
System Prompt 里要求「先用 get_schema 了解数据，再选工具计算」。否则模型可能凭空假设列名。这种「行为引导」是 Agent 工程里很重要的一环——好的 Prompt 让 Agent 少走弯路、少调错工具。

### 难点 3：可观测性
代码遍历 `result["messages"]`，把 Agent 实际调用了哪些工具、传了什么参数打印出来。**线上排查 Agent「为什么算错」第一步就是看它调了什么工具**，这个习惯从项目就要养成（详见 [Agent 轨迹调试](03-11_agent_trace_debug.md)）。

### 为什么用 LangGraph 的 create_react_agent
工具循环（决定→调用→喂回→再决定）由框架自动跑，我们只管定义工具和 Prompt。对比手写循环（见 `02_react_from_scratch.py`），框架还白送了状态管理和可持久化记忆。

---

## 三、运行

```bash
pip install langgraph pandas
python stage03_agent/project04_data_analysis/agent.py
```

会看到 Agent 对四个问题分别**自主选择并调用工具**，打印工具调用轨迹和中文结论。把 `sales_data.csv` 换成你自己的数据、加几个工具，就是你自己的 ChatBI。

---

## 四、复盘与进阶

1. **接真数据库**：把工具内部从 pandas 换成 SQL 查询，就是真正的 NL2SQL。
2. **加画图工具**：让 Agent 调用 matplotlib 工具生成图表返回图片路径。
3. **加澄清能力**：问题有歧义时，让 Agent 反问用户而不是瞎猜（多轮）。
4. **加护栏**：对聚合结果做合理性校验，防止把脏数据当结论。

---

## 五、面试怎么考

- **「为什么不直接让模型写 pandas 代码执行？」** → 代码注入风险 + 结果不可控。生产用受控工具，把模型的自由度限制在安全边界内。能讲出这个权衡是关键加分点。
- **「模型怎么知道该调哪个工具？」** → 靠工具的 name + docstring + 参数 schema（bind_tools 把它们打包给模型），模型据此做选择。所以 docstring 质量直接决定 Agent 准确率。
- **「Agent 调错工具/算错怎么排查？」** → 打印执行轨迹（调了哪些工具、传了什么参数、返回了什么），定位是「选错工具」还是「传错参数」还是「工具本身有 bug」。
- **「这和直接写个 BI 报表有什么区别？」** → 报表是预定义的；Agent 能回答开放式、未预设的问题，灵活性是数量级的差异。
