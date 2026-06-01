# 03-11 调试 Agent：看懂执行轨迹，定位「为什么调错工具 / 死循环」

> 🎯 **一句话**：Agent 出问题（调错工具、不调工具、死循环、参数错）几乎都能通过「读懂它的执行轨迹」来定位——本篇教你把 Agent 的每一步打印出来、读明白、对症下药。

---

## 为什么需要它

Agent 是个会「自己做决定」的循环，黑盒跑一遍只看到最终答案，错了根本不知道错在哪一步：是模型没选对工具？参数填错了？工具报错了？还是陷入了反复调用的死循环？

**调试 Agent 的核心 = 让循环的每一步可见**。看清「第几步、想了什么、调了哪个工具、传了什么参数、工具返回了什么」，问题往往一眼就现形。LangGraph 的 `messages` 轨迹 + `stream` 让这件事变得简单。

---

## 核心用法

### 方法一：直接打印 messages 轨迹（最常用）

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from common.llm_provider import get_llm


@tool
def get_weather(city: str) -> str:
    """查询某城市天气。"""
    return f"{city} 晴，26℃。"

agent = create_react_agent(get_llm(temperature=0), tools=[get_weather])
result = agent.invoke({"messages": [("user", "北京和上海天气如何？")]})

for m in result["messages"]:
    m.pretty_print()
```

**本质在干什么？** `messages` 是 Agent 的完整「黑匣子记录」。`pretty_print()` 会按序打印：

```
================== Human ==================
北京和上海天气如何？
=================== Ai ====================        ← 模型决定调工具
Tool Calls: get_weather(city='北京')
================== Tool ===================        ← 工具返回
北京 晴，26℃。
=================== Ai ====================        ← 再次决定
Tool Calls: get_weather(city='上海')
================== Tool ===================
上海 晴，26℃。
=================== Ai ====================        ← 最终答案
北京和上海今天都是晴，26℃。
```

每个 `Ai` 块的 `Tool Calls` 告诉你**模型选了哪个工具、传了什么参数**；每个 `Tool` 块告诉你**工具实际返回了什么**。调错工具、传错参数、工具报错，在这条链上都看得一清二楚。

### 方法二：`stream` 实时观测，抓死循环

```python
for i, chunk in enumerate(agent.stream(
        {"messages": [("user", "北京和上海天气如何？")]},
        stream_mode="values")):
    print(f"--- 第 {i} 步 ---")
    chunk["messages"][-1].pretty_print()
    if i > 15:                         # 步数异常多 → 疑似死循环，及时熔断
        print("步数过多，疑似死循环，中断观察")
        break
```

**本质在干什么？** `stream_mode="values"` 让每个节点执行完就吐出当前最新状态，你能**边跑边看**。如果同一个工具被反复用同样参数调用、步数迟迟不收敛，就是死循环的现场证据。两种常用模式：
- `stream_mode="values"`：每步给出**完整**最新状态（看全局演进）。
- `stream_mode="updates"`：每步只给**增量**（看本步新增了什么，更省眼力）。

### 方法三：全局开 verbose / debug 日志

```python
from langchain.globals import set_verbose, set_debug

set_verbose(True)   # 打印每步关键信息（调了什么、返回什么）
set_debug(True)     # 更啰嗦：连原始 prompt、原始响应都打印（排查 prompt 问题用）

agent.invoke({"messages": [("user", "北京天气？")]})
```

**本质在干什么？** `set_verbose` 给出适中粒度的过程日志；`set_debug` 打印连模型收到的**完整 prompt 和原始返回**——当你怀疑「是不是工具描述/系统提示写得不对导致选错」时，用 `set_debug` 看模型到底收到了什么。用完记得关掉，否则日志淹没控制台。

---

## 关键原理：常见故障 → 诊断 → 处方

| 症状 | 在轨迹里看什么 | 根因 & 处方 |
|---|---|---|
| **调错工具** | AI 块的 `Tool Calls` 选了不该选的工具 | 工具 `description` 含糊/重叠 → 重写描述，明确「何时该用」（见 03-01） |
| **该调却不调** | 模型直接给文本答案，没有 `Tool Calls` | 描述没覆盖该场景，或系统提示没鼓励用工具 → 补描述/提示；确认 `temperature=0` |
| **参数填错** | `Tool Calls` 的 args 类型/值不对 | 缺类型注解或 Field 说明 → 用 `args_schema` 加约束（见 03-02），并加 Pydantic 校验 |
| **工具报错** | Tool 块是异常文本 | 工具内部 bug 或入参非法 → 加 `handle_tool_error`，把错误回喂让模型纠正 |
| **死循环** | 同一工具用同样参数反复出现，步数不收敛 | 工具结果没解决问题、模型反复试 → 设 `recursion_limit` 上限熔断 + 检查工具是否真的有用 |

补充要点：
1. **轨迹优先于猜测**：先把 messages 打出来，不要凭感觉改 prompt。90% 的问题在轨迹里直接可见。
2. **死循环熔断**：LangGraph 的 agent 有 `recursion_limit`（默认 25），超过会抛错而非无限跑。可在 config 里调小（如 `{"recursion_limit": 8}`）快速暴露问题。
3. **LangSmith 进阶**：设好环境变量后，LangGraph 会自动把每步上报 LangSmith，提供可视化时间线、每步 token/耗时、原始 prompt——生产环境的标准排障工具（Stage 1 的 LangSmith 篇）。
4. **temperature=0 是调试前提**：非 0 温度下行为随机、复现困难，调 Agent 务必先 `temperature=0`。

---

## 你来改

- [ ] 故意把某个工具的 docstring 写得和另一个工具高度重叠，触发「调错工具」，再从轨迹里确认现象，然后改好描述。
- [ ] 把 config 设成 `{"recursion_limit": 4}`，问一个需要 5 步的问题，观察熔断报错，理解 `recursion_limit` 的作用。
- [ ] 用 `set_debug(True)` 跑一次，找到模型实际收到的完整 system prompt 和工具描述，确认它和你写的一致。

---

## 面试怎么考

**Q：Agent 调错了工具，你怎么排查？**
A：先打印 `messages` 轨迹（或用 `stream`/LangSmith），定位到模型在哪个 AI 步选错了工具、传了什么参数。调错工具几乎都是工具 `description` 含糊或多个工具描述重叠所致——重写描述、明确各自适用场景；参数错则用 `args_schema` 加约束。务必在 `temperature=0` 下复现。

**Q：怎么发现并处理 Agent 死循环？**
A：在轨迹/流式输出里观察是否同一工具用相同参数反复出现、步数不收敛。处理上靠 `recursion_limit`（LangGraph 默认 25）熔断，调小可快速暴露；根因通常是工具没真正解决问题或模型误判，需修工具或优化提示，而非一味放大上限。

**Q：`set_verbose` 和 `set_debug` 有什么区别？什么时候用 LangSmith？**
A：`set_verbose` 打印适中粒度的过程信息（调了什么、返回什么）；`set_debug` 更底层，连发给模型的完整 prompt 和原始响应都打印，适合排查「是不是提示/工具描述写错」。LangSmith 提供可视化时间线、每步 token/耗时与原始数据，是生产环境长期可观测与排障的标准方案。
