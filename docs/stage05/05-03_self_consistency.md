# 05-03 Self-Consistency：多次采样投票，少数服从多数

> 🎯 **一句话**：Self-Consistency（自洽性）对同一问题用较高温度采样多个 CoT 推理，再对最终答案投票取多数，靠「多条独立思路殊途同归」来抹平单次推理的随机错误，显著提升答案稳定性。

---

## 为什么需要它

CoT 只走一条推理链，而较高温度下模型每次走的路可能不同——有时对、有时错，结果不稳定。但有个关键观察：**正确答案往往可以由多条不同推理路径到达，而错误答案各错各的、难以多次重合**。

Self-Consistency 据此设计：让模型对同一题独立推理多次（每次路径可能不同），然后**对最终答案做多数投票**。错误被分散，正确被汇聚——这是一种「集成学习」思想在 prompt 层的应用，无需训练、即插即用。

---

## 核心用法

```python
from dotenv import load_dotenv
load_dotenv()

from collections import Counter
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from common.llm_provider import get_llm

# 关键：温度 > 0，让多次采样产生不同推理路径
llm = get_llm(temperature=0.7)

cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "请一步步推理，最后单独用一行『答案：X』给出最终数字。"),
    ("human", "{question}"),
])
chain = cot_prompt | llm | StrOutputParser()


def extract_answer(text: str) -> str:
    """从一段 CoT 文本里抽出『答案：X』的 X。"""
    m = re.findall(r"答案：\s*([\-\d\.]+)", text)
    return m[-1] if m else "?"


def self_consistency(question: str, n: int = 5) -> str:
    # 1) 用 batch 并行采样 n 条独立推理（比 for 循环快）
    inputs = [{"question": question}] * n
    outputs = chain.batch(inputs)
    # 2) 各自抽出最终答案
    answers = [extract_answer(o) for o in outputs]
    # 3) 多数投票
    winner, count = Counter(answers).most_common(1)[0]
    print(f"采样答案分布：{Counter(answers)} → 取多数：{winner}")
    return winner


print(self_consistency("23 个学生，每人发 4 支笔，已发出 60 支，还需几支？"))
```

**逐块讲解：**
- **温度 > 0**：这是 Self-Consistency 的前提。`temperature=0` 时每次输出几乎相同，投票毫无意义。`0.5~0.8` 能制造「不同但合理」的推理路径。
- **batch 并行采样**：用 `chain.batch([...]*n)` 一次性发 n 个相同请求，比串行 for 循环快得多。
- **抽取最终答案**：每条 CoT 文本各自解析出最终数字（投票的是**结论**，不是整段推理）。
- **多数投票**：`Counter.most_common(1)` 取出现次数最多的答案。这一步把随机错误过滤掉。

---

## 关键原理 / 实践要点

1. **为什么有效**：正确答案是多条推理路径的共同终点，错误答案各不相同。多数投票让正确答案在分布中胜出——本质是采样集成（marginalize over reasoning paths）。
2. **temperature 的作用**：它控制路径多样性。太低（→0）所有采样雷同、投票无意义；太高（→1.5）推理崩坏、噪声过大。常用 `0.5~0.8`。
3. **必须有可比对的「确定答案」**：投票要求答案能被归一化比较（数字、选项、分类标签）。
4. **适用场景**：数学题、多选题、分类、抽取等**有唯一/离散正确答案**的任务，提升最明显。
5. **不适用场景**：开放式生成（写作、摘要、翻译、代码）——这类没有「多数答案」可投，强行投票反而劣化。此时应改用 LLM-as-a-Judge（见 05-05）择优。
6. **成本**：n 次采样 = n 倍 token 成本。n 通常取 5~10，性价比拐点之后收益递减。

---

## 你来改

- [ ] 把 `temperature` 改成 0，观察采样分布是否塌缩成同一答案，理解为何需要随机性。
- [ ] 对一道开放式问题（如「给产品起个名」）跑 Self-Consistency，体会为何它失效。
- [ ] 把 n 从 5 调到 11，看一道偏难的题正确率是否上升、成本上升多少。

---

## 面试怎么考

**Q：Self-Consistency 的原理是什么？为什么需要 temperature > 0？**
A：对同一问题用较高温度采样多条 CoT 推理，再对最终答案多数投票。原理是正确答案常由多条路径共同到达、错误答案各异，投票让正确胜出，本质是对推理路径的采样集成。temperature>0 才能产生多样化路径；温度为 0 时采样几乎相同，投票无意义。

**Q：Self-Consistency 适用和不适用哪些任务？**
A：适用于有唯一或离散正确答案的任务（数学、多选、分类、信息抽取），投票能过滤随机错误。不适用于开放式生成（写作、摘要、代码），因为没有可投票的「多数答案」，此时应改用 LLM-as-a-Judge 等择优方法。

**Q：它和 ToT 有何不同？**
A：ToT 是显式搜索树、每步评估剪枝、有过程控制；Self-Consistency 不评估中间过程，只是独立采样 n 条完整推理后对结论投票，更简单、成本可预测，但只适合离散答案任务。
