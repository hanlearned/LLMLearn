# 06-02 LCEL 高级：分支、降级、重试、运行时配置与可视化

> 🎯 **一句话**：掌握 `RunnableLambda`/`RunnableBranch`/`with_fallbacks`/`with_retry`/`configurable_fields` 这几个组合算子，就能把 LCEL 从「直管道」升级成「带路由、能容错、可运行时配置」的生产级链，并用 `get_graph().print_ascii()` 把它画出来调试。

---

## 为什么需要它

`prompt | llm | parser` 这种直管道只能应付最简单的场景。真实链路需要：根据输入走不同分支、主模型挂了自动切备用、瞬时报错自动重试、运行时动态换模型/温度而不改代码。

LCEL 把这些能力都做成了**可链式组合的算子**——它们本身也是 Runnable，能无缝拼进管道，保持统一的 `invoke/stream/batch/ainvoke` 接口。

---

## 核心用法

### 1. RunnableLambda：把普通函数塞进链

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_core.runnables import RunnableLambda

# 任意 Python 函数包一层就成了 Runnable，可接入管道做预处理/后处理
clean = RunnableLambda(lambda x: {"question": x["question"].strip().lower()})
print(clean.invoke({"question": "  Hello LangChain  "}))
```

**本质在干什么？** `RunnableLambda` 把自定义函数升格为 Runnable，让你在链里插入任意逻辑（清洗、格式化、调外部 API），同时享有统一接口和自动并行/流式。

### 2. RunnableBranch：条件路由（if/elif/else）

```python
from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from common.llm_provider import get_llm

llm = get_llm(temperature=0)
code_chain = ChatPromptTemplate.from_template("写代码：{q}") | llm | StrOutputParser()
chat_chain = ChatPromptTemplate.from_template("聊天回答：{q}") | llm | StrOutputParser()

router = RunnableBranch(
    (lambda x: "代码" in x["q"], code_chain),   # 条件1成立 → code_chain
    chat_chain,                                  # 都不成立 → 默认分支
)
print(router.invoke({"q": "帮我写代码：冒泡排序"}))
```

**本质在干什么？** `RunnableBranch` 是 LCEL 版 if/elif/else：按顺序匹配条件，命中就走对应链，最后一个参数是 default。常用于「先分类再路由到不同 prompt/模型」。

### 3. with_fallbacks：主链失败自动降级

```python
primary = get_llm(model="deepseek-ai/DeepSeek-V3", temperature=0)
backup = get_llm(model="deepseek-chat", temperature=0)

# 主模型抛异常（限流/宕机）时，自动切到备用模型
robust_llm = primary.with_fallbacks([backup])
print(robust_llm.invoke("一句话介绍 LangChain").content)
```

**本质在干什么？** `with_fallbacks` 提供降级链：主 Runnable 抛错时依次尝试备选。用于跨模型/跨厂商容灾——主厂商限流就切备用厂商，提升可用性。

### 4. with_retry：瞬时错误自动重试

```python
# 网络抖动、临时 429 等瞬时错误自动重试，带指数退避
resilient = llm.with_retry(stop_after_attempt=3)
print(resilient.invoke("hi").content)
```

**本质在干什么？** `with_retry` 对**同一个** Runnable 重试若干次（默认指数退避），对付瞬时故障。和 fallbacks 互补：retry 是「再试同一个」，fallback 是「换一个」。

### 5. configurable_fields：运行时动态改参数

```python
from langchain_core.runnables import ConfigurableField

configurable_llm = get_llm().configurable_fields(
    temperature=ConfigurableField(id="temp"),   # 暴露 temperature 为可配置项
)
# 同一条链，运行时按需切换温度，无需重建
print(configurable_llm.invoke("起个名", config={"configurable": {"temp": 0.0}}).content)
print(configurable_llm.invoke("起个名", config={"configurable": {"temp": 1.0}}).content)
```

**本质在干什么？** `configurable_fields` 把某些参数（温度、模型名等）暴露为运行时可配置项，调用时通过 `config={"configurable":{...}}` 动态覆盖——一条链服务多种配置，不必为每种参数建一条链。

### 6. 可视化调试：print_ascii

```python
chain = ChatPromptTemplate.from_template("{q}") | llm | StrOutputParser()
chain.get_graph().print_ascii()   # 在终端画出链的结构图
```

**本质在干什么？** 复杂链（含分支、并行）容易绕晕，`get_graph().print_ascii()` 把数据流向画成 ASCII 图，一眼看清「输入怎么流经各节点」，是排查 LCEL 结构问题的利器。

---

## 关键原理 / 实践要点

1. **一切皆 Runnable**：这些算子返回的仍是 Runnable，可继续 `|` 拼接、可嵌套，保持统一接口——这是 LCEL 可组合性的根基。
2. **retry vs fallback**：retry 重试同一个组件（治瞬时抖动）；fallback 换备选组件（治整体不可用）。生产常组合：`primary.with_retry().with_fallbacks([backup])`。
3. **configurable 用于一链多态**：A/B 测试不同温度/模型、按用户等级切配置，都靠它在运行时动态调整，避免代码里写死或建多条链。
4. **RunnableBranch 路由要兜底**：最后必须有 default 分支，否则无匹配会报错。
5. **先画图再调**：链一复杂就 `print_ascii()`，确认结构符合预期，比盲读代码快得多。

---

## 你来改

- [ ] 给一条链同时加 `with_retry` 和 `with_fallbacks`，故意把主模型名写错触发降级，观察是否切到备用。
- [ ] 用 `configurable_fields` 同时暴露 temperature 和 model，运行时切换两种模型对比输出。
- [ ] 写一个三分支 `RunnableBranch`（代码/翻译/闲聊），对每类输入路由到不同 prompt，并 `print_ascii()` 看结构。

---

## 面试怎么考

**Q：with_retry 和 with_fallbacks 区别？怎么配合？**
A：with_retry 对同一个 Runnable 重试若干次（默认指数退避），治网络抖动、临时 429 等瞬时错误；with_fallbacks 在主 Runnable 抛错时切换到备选组件，治整体不可用（如某厂商宕机/限流）。生产常组合 `primary.with_retry().with_fallbacks([backup])`，先重试同一个、彻底失败再换备用。

**Q：configurable_fields 解决什么问题？**
A：把链中参数（温度、模型名等）暴露为运行时可配置项，调用时通过 config 动态覆盖，实现「一条链服务多种配置」。用于 A/B 测试、按用户等级切模型、动态调温度，避免写死或为每种配置建多条链。

**Q：RunnableBranch 和 RunnableLambda 各是什么？**
A：RunnableBranch 是 LCEL 的 if/elif/else 条件路由，按顺序匹配条件走对应子链、末尾兜底 default；RunnableLambda 把普通 Python 函数升格为 Runnable，用于在链中插入清洗、格式化等自定义逻辑。两者都是 Runnable，可无缝拼进管道。复杂链可用 `get_graph().print_ascii()` 可视化调试。
