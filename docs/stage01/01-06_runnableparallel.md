# 01-06 `RunnableParallel`：并行执行多个子任务

> 🎯 **目标**：掌握 `RunnableParallel` 的使用方式，理解"分叉-合并"模式在 Chain 中的工程意义。

---

## 一句话功能

`RunnableParallel` 接收**一份输入**，同时分发给多个子 Runnable 并行执行，最终把各分支结果按指定 key 合并为**一个字典**返回。

---

## 为什么需要它？

前面的 `RunnableSequence`（`prompt | llm | parser`）是**串行**的：前一步做完，后一步才能开始。

但业务中很多任务之间**没有依赖关系**，完全可以同时跑：

| 场景 | 串行做法 | 并行做法 |
|:---|:---|:---|
| 情感分析 + 关键词提取 | 调两次模型，等 2× latency | 同时跑两个 prompt |
| 同时查多个向量库 | 先查 A 再查 B | 同时查 A、B、C |
| 生成 + 统计 | 生成完再统计字数 | 生成和统计同时进行 |

`RunnableParallel` 就是 LangChain 提供的"分叉路口"。

---

## 核心代码示例

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from common.llm_provider import get_llm

llm = get_llm()

# 分支 1：生成笑话
joke_chain = (
    ChatPromptTemplate.from_template("讲一个关于{topic}的笑话，30字以内。")
    | llm | StrOutputParser()
)

# 分支 2：生成短诗
poem_chain = (
    ChatPromptTemplate.from_template("写一首关于{topic}的短诗，30字以内。")
    | llm | StrOutputParser()
)

# 分支 3：本地计算（不走 LLM）
count_chain = RunnableLambda(lambda x: f"'{x['topic']}' 共 {len(x['topic'])} 个字")

# 并行：三个分支同时开工
parallel = RunnableParallel(
    joke=joke_chain,
    poem=poem_chain,
    count=count_chain,
)

result = parallel.invoke({"topic": "程序员"})
# 输出：{"joke": "...", "poem": "...", "count": "'程序员' 共 3 个字"}
```

---

## `RunnableParallel` 的本质

### 1. 它是"输入分发器 + 结果合并器"

```python
RunnableParallel(
    joke=joke_chain,    # key 名决定输出字典的 key
    poem=poem_chain,
)
```

- **输入侧**：同一份 `{"topic": "程序员"}` 被复制给 `joke_chain` 和 `poem_chain`
- **执行侧**：LangChain 内部用线程池调度，I/O 等待期间可以切换
- **输出侧**：等所有分支完成后，按 key 组装成字典返回

### 2. 和 `RunnableSequence` 的无缝混用

`|` 操作符既可以把 Runnable 串起来，也可以把并行组串进更大的链路：

```python
# 先并行生成素材，再串行整合为报告
report_prompt = ChatPromptTemplate.from_template("""\
基于以下素材写段介绍：
{joke}
{poem}
{count}
""")
report_chain = parallel | report_prompt | llm | StrOutputParser()
```

执行顺序：
```
输入 → [并行: joke + poem + count] → 合并为 dict → report_prompt → llm → parser
```

### 3. 本地任务用 `RunnableLambda` 包装

不是每个分支都必须调 LLM。对于纯 Python 函数，用 `RunnableLambda` 包装后就能放进 `RunnableParallel`：

```python
count_chain = RunnableLambda(lambda x: len(x["topic"]))
```

这样并行组里可以同时存在"重型 I/O 任务"（调模型）和"轻量本地任务"（字符串处理）。

---

## 常见误区

### Q：`RunnableParallel` 是真并行还是伪并行？

**A：** 是**线程级并行**（`ThreadPoolExecutor`）。对于 I/O 密集型任务（网络请求、数据库查询）效率很高；但受 Python GIL 限制，纯 CPU 密集型任务无法利用多核。如果需要多核并行，要改用多进程或 `asyncio`。

### Q：其中一个分支报错了会怎样？

**A：** 默认**快速失败**（fail-fast）。任何一个分支抛异常，整个 `invoke()` 会中断并抛出异常。你可以在各分支内部用 `RunnableLambda` 自己做 try-catch，或者在外层用 `try-except` 捕获。

### Q：各分支的输入必须完全一样吗？

**A：** 默认是**同一份输入原样分发**。但如果某个分支需要不同的输入格式，可以在分支前接一个 `RunnableLambda` 做数据转换。例如：

```python
RunnableParallel(
    summary=summary_chain,
    tags=RunnableLambda(lambda x: x["content"]) | tag_chain,
)
```

---

## 面试考点

**Q：`RunnableParallel` 和 `RunnableSequence` 的核心区别是什么？**

**A：** `RunnableSequence`（`|`）是**串行**：前一个组件的输出作为后一个的输入；`RunnableParallel` 是**并行**：一份输入同时分发给多个子组件，各自独立执行，最终合并为字典。

**Q：`RunnableParallel` 的底层并行机制是什么？**

**A：** LangChain 使用 `concurrent.futures.ThreadPoolExecutor` 实现。这意味着它是线程级并行，适合 I/O 密集型场景（如并发调用多个 LLM API），但不适合 CPU 密集型任务。

**Q：并行后再串行的场景里，后一个组件如何接收并行结果？**

**A：** `RunnableParallel` 的输出是一个字典，后接的 `ChatPromptTemplate` 或 `Runnable` 会自动按 key 名提取变量。例如 `{"joke": "...", "poem": "..."}` 可以直接被 `ChatPromptTemplate.from_template("笑话：{joke}\n诗歌：{poem}")` 消费。

---

## 课后任务

- [ ] 运行 `05_runnable_parallel.py`，观察并行执行与串并联组合的输出
- [ ] 修改代码，增加第 4 个分支（例如"生成一句名言"），观察输出字典的变化
- [ ] 尝试把 `joke_chain` 和 `poem_chain` 分别换成不同的模型参数（temperature），观察结果差异
- [ ] （思考题）如果两个分支都需要调用同一个外部 API（如天气查询），用 `RunnableParallel` 并行调用会不会触发两次请求？如何优化？
