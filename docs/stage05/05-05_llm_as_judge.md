# 05-05 LLM-as-a-Judge：用模型给输出打分（重点篇）

> 🎯 **一句话**：LLM-as-a-Judge 用一个强模型当「裁判」，按预设标准给另一个模型的输出打分或在两份输出间择优——把开放式生成质量这种「没有标准答案、人工评测又太贵」的评估自动化、规模化。

---

## 为什么需要它

写作、摘要、客服回复、RAG 答案这类**开放式输出没有唯一正确答案**，无法像数学题那样用准确率衡量。人工评测准但慢、贵、不可规模化，每改一版 prompt 都要人重看一遍根本扛不住。

LLM-as-a-Judge 的思路：**用模型代替人来打分**。它能 24 小时跑、成本是人工的零头、标准一致，适合在 prompt 迭代、模型选型、回归测试、线上质量监控中做大规模自动评估——这是 LLMOps 的核心基础设施。

---

## 核心用法

### 1. 打分式（Pointwise）：给单条输出按维度评分

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from common.llm_provider import get_llm

judge = get_llm(temperature=0)         # 裁判务必 temperature=0，保证可复现

judge_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是严格的评测专家。请按以下标准给【回答】打分：\n"
     "- 相关性(1-5)：是否切题\n"
     "- 准确性(1-5)：是否符合【参考资料】、有无编造\n"
     "- 完整性(1-5)：是否覆盖要点\n"
     "先简述评分理由，再输出 JSON："
     '{{"relevance":int,"accuracy":int,"completeness":int,"reason":"..."}}'),
    ("human", "问题：{question}\n参考资料：{context}\n回答：{answer}"),
])

judge_chain = judge_prompt | judge | JsonOutputParser()

score = judge_chain.invoke({
    "question": "LangChain 的 LCEL 是什么？",
    "context": "LCEL 是 LangChain 表达式语言，用 | 组合 Runnable。",
    "answer": "LCEL 是 LangChain 的链式表达式语言，用管道符组合组件。",
})
print(score)   # {'relevance':5,'accuracy':5,'completeness':4,'reason':...}
```

**逐块讲解：**
- **明确的评分维度 + 量表**：不要笼统问「好不好」，而要拆成相关性/准确性/完整性等具体维度，每维给 1-5 的明确量表。维度越具体，评分越稳。
- **提供参考资料**：判断「准确性/是否编造」必须给裁判 ground truth（参考资料），否则它只能凭空猜。
- **先理由后打分**：让裁判先写 `reason` 再给分（CoT 思想），评分更可靠且可审计。
- **结构化输出**：用 JSON + `JsonOutputParser`，分数可直接进数据库做统计聚合。
- **temperature=0**：裁判必须可复现，温度为 0。

### 2. 成对比较式（Pairwise）：A 和 B 哪个更好

```python
pairwise_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "对比两个回答哪个更好。只输出 JSON："
     '{{"winner":"A"或"B"或"tie","reason":"..."}}'),
    ("human", "问题：{q}\n回答A：{a}\n回答B：{b}"),
])
pairwise = pairwise_prompt | judge | JsonOutputParser()

r = pairwise.invoke({"q": "解释什么是向量数据库",
                     "a": "存向量并支持相似度检索的数据库。",
                     "b": "一种数据库。"})
print(r)   # {'winner':'A','reason':'更具体完整'}
```

**本质在干什么？** 成对比较问「A 还是 B 更好」，比绝对打分**更容易、更稳定**——人和模型都更擅长「比较」而非「打绝对分」。A/B 测试两版 prompt、两个模型时首选 pairwise。

### 3. 缓解位置偏见：交换顺序双跑

```python
def fair_compare(q, a, b):
    # 正反各比一次，抵消「裁判偏爱靠前选项」的位置偏见
    r1 = pairwise.invoke({"q": q, "a": a, "b": b})["winner"]          # A=a, B=b
    r2 = pairwise.invoke({"q": q, "a": b, "b": a})["winner"]          # 交换
    # 把第二次结果映射回原始 a/b
    r2_mapped = {"A": "B", "B": "A", "tie": "tie"}[r2]
    if r1 == r2_mapped:
        return r1                       # 两次一致 → 可信
    return "tie"                        # 换序后结论翻转 → 判平局，存疑
```

**本质在干什么？** 裁判模型存在**位置偏见**——倾向于选靠前（或靠后）的选项。交换 A/B 顺序各跑一次：若结论一致才采信，翻转则判平局。这是 pairwise 评测的标准去偏手段。

---

## 关键原理 / 实践要点

1. **打分 vs 成对比较**：pointwise 给绝对分，便于横向聚合、设阈值告警；pairwise 只比相对优劣，更稳更准，适合 A/B 选型。能用比较就用比较。
2. **裁判的常见偏见与缓解**：
   - **位置偏见**：偏爱固定位置 → 交换顺序双跑取一致。
   - **冗长偏见**：偏爱更长的答案 → 在标准里强调「简洁也是优点」，或控制长度。
   - **自我偏好**：模型偏爱自己/同家族模型的输出 → 用**异源**强模型当裁判。
   - **风格/谄媚偏见**：偏爱措辞华丽、迎合性强的 → 评分标准里明确只看实质质量。
3. **裁判要比被评者强或至少同级**：用弱模型评强模型输出不可靠。常用更强的模型专做裁判。
4. **维度拆解 + 量表 + 理由**：把「好」拆成可操作维度，给明确分级，要求先说理由再打分——评分一致性大幅提升。
5. **要校准**：先用一小批**人工标注**样本验证「裁判打分与人类判断的相关性」，相关性够高再大规模用。裁判不是绝对真理，是人工评测的高性价比近似。
6. **工程落地**：LangSmith 内置 LLM-as-judge evaluator，可对数据集批量评测并出报表；自建时把分数落库 + 监控趋势即可做线上质量看板。

---

## 你来改

- [ ] 给打分 prompt 故意去掉「参考资料」，看裁判对「是否编造」的判断如何退化。
- [ ] 用 `fair_compare` 对两版客服回复做带去偏的成对比较，再不去偏跑一次，观察结论是否被位置偏见影响。
- [ ] 造 5 条人工标注（好/坏），用你的裁判跑一遍，算它和人工标签的一致率，判断裁判是否可信。

---

## 面试怎么考

**Q：LLM-as-a-Judge 解决什么问题？怎么设计评分 Prompt？**
A：解决开放式输出（写作、摘要、RAG 答案）没有标准答案、人工评测贵且不可规模化的问题，用强模型当裁判自动打分。评分 prompt 设计要点：拆成具体维度（相关性/准确性/完整性）、每维给明确量表、提供参考资料判断准确性、要求先写理由再打分（CoT）、输出结构化 JSON、裁判用 temperature=0 保证可复现。

**Q：打分式和成对比较式怎么选？**
A：打分式给绝对分，便于聚合统计和设阈值；成对比较只判 A/B 谁更好，对模型和人都更容易、结论更稳定，适合 prompt/模型 A/B 选型。原则上能用比较就用比较，需要绝对指标看板时才用打分。

**Q：LLM 裁判有哪些偏见？怎么缓解？**
A：位置偏见（偏爱固定位置，交换顺序双跑取一致缓解）、冗长偏见（偏爱长答案，标准里强调简洁）、自我偏好（偏爱同家族输出，用异源裁判）、谄媚/风格偏见（标准里只看实质）。此外裁判应不弱于被评模型，且上线前要用人工标注校准裁判与人类判断的相关性。

**Q：能完全用 LLM 裁判取代人工评测吗？**
A：不能完全取代，但能覆盖绝大多数日常回归与监控。它是人工评测的高性价比近似，需先用小批人工标注校准其与人类的一致性，再大规模使用；关键决策（如发布门槛）仍应保留人工抽检。
