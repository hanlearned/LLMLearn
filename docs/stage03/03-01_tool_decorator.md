# 03-01 `@tool` 装饰器：把普通函数变成模型能调用的工具

> 🎯 **一句话**：`@tool` 解决的是「如何把一个普通 Python 函数，翻译成大模型看得懂、能主动选择调用的『工具说明书』」这件事。

---

## 为什么需要它

大模型本身只会「生成文本」，它不会执行代码、查数据库、调天气 API。要让它具备「行动能力」，我们得给它一批工具，并告诉它每个工具：**叫什么名字、是干什么的、需要哪些参数、参数是什么类型**。

这套「告诉模型的工具描述」必须是结构化的 JSON Schema（OpenAI 称之为 function 定义）。手写这份 Schema 又啰嗦又容易和函数实现不同步。`@tool` 装饰器做的事，就是**自动从你的函数签名 + 类型注解 + docstring 里提取出这份 Schema**，让你只管写函数。

一句话：`@tool` = 「普通函数」→「带自描述能力的 LangChain 工具对象」的自动转换器。

---

## 核心用法

```python
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气。

    Args:
        city: 城市名称，例如「北京」「上海」。
    """
    # 真实场景这里会调用天气 API，这里写死演示
    return f"{city} 今天晴，气温 26℃。"


# 看看 LangChain 帮我们生成了什么
print(get_weather.name)         # get_weather
print(get_weather.description)  # 查询指定城市的实时天气。...
print(get_weather.args)         # {'city': {'title': 'City', 'type': 'string', ...}}

# 像普通函数一样调用（注意：用 invoke，参数用字典）
print(get_weather.invoke({"city": "北京"}))  # 北京 今天晴，气温 26℃。
```

**本质在干什么？**

- `@tool` 把 `get_weather` 函数包装成了一个 `StructuredTool` 对象（不再是普通函数）。
- `name`：默认取函数名 `get_weather`，这是模型用来「点名」要调用哪个工具的标识。
- `description`：取自 **docstring**。这是最关键的一环——模型完全靠这段文字判断「该不该用、什么时候用这个工具」。docstring 写得含糊，模型就会调错或不调。
- `args`：从**类型注解** `city: str` 自动推导出参数的 JSON Schema，类型、必填性都在里面。
- 调用工具要用 `.invoke({...})`，参数以字典传入，而不是 `get_weather("北京")`。

把工具绑给模型时，LangChain 会把上面这些信息序列化成标准 function schema：

```python
from common.llm_provider import get_llm

llm = get_llm(temperature=0)            # Agent / 工具调用务必温度=0
llm_with_tools = llm.bind_tools([get_weather])

resp = llm_with_tools.invoke("北京天气怎么样？")
print(resp.tool_calls)
# [{'name': 'get_weather', 'args': {'city': '北京'}, 'id': '...'}]
```

**本质在干什么？** 模型读到了 `get_weather` 的「说明书」，判断这次请求需要查天气，于是**没有直接回答**，而是返回了一个 `tool_calls`：告诉你「请帮我调用 get_weather，参数 city=北京」。真正的执行还得我们自己来（详见 03-03）。

---

## 关键原理

1. **docstring 即 prompt**：工具描述会被原样塞进发给模型的请求里，消耗 token 也影响决策。它本质上是一段「定向 prompt」，要写清楚「这个工具能做什么、什么场景该用」。
2. **类型注解即 Schema**：`city: str` → `{"type": "string"}`；`n: int` → `{"type": "integer"}`。没有类型注解，LangChain 无法生成合法 Schema，模型可能传错类型。
3. **自定义元数据**：可以覆盖默认行为。

```python
@tool("weather_lookup", return_direct=False)
def get_weather(city: str) -> str:
    """..."""
```

`@tool("weather_lookup")` 把工具名改成 `weather_lookup`；`return_direct=True` 则表示「工具一旦执行，结果直接作为 Agent 的最终答复返回，不再让模型加工」。

4. **工具就是 Runnable**：`StructuredTool` 也继承自 `Runnable`，所以它有 `invoke` / `ainvoke` / `batch`，能进 LCEL 链，也能被异步调用。

---

## 你来改

- [ ] 写一个 `calculator(expression: str) -> str` 工具，内部用 `eval` 计算表达式（仅供练习，生产别用 eval），绑定给模型问「123 乘以 456 等于多少」，打印 `tool_calls`。
- [ ] 故意把 docstring 删掉，重新 `bind_tools` 后再问，观察模型是否还能正确选择工具。体会 docstring 的作用。
- [ ] 给工具加第二个参数 `unit: str`（摄氏/华氏），看看 `get_weather.args` 多了什么字段。

---

## 面试怎么考

**Q：`@tool` 是怎么让模型「知道」一个函数的？模型真的执行了你的 Python 函数吗？**
A：`@tool` 从函数名、类型注解、docstring 提取出一份 function JSON Schema，`bind_tools` 时随请求发给模型。模型**并不执行**你的函数，它只是根据 Schema 决定「要不要调、传什么参数」，返回一个 `tool_calls` 结构，真正的执行由我们的代码完成。

**Q：docstring 对工具调用准确率有多大影响？**
A：决定性影响。description 是模型选工具、判断适用场景的唯一依据，等同于一段定向 prompt。描述模糊会导致漏调、错调；多个工具描述重叠会导致选错。生产中常需要专门打磨工具描述。

**Q：`@tool` 和直接写 function schema 字典有什么区别？**
A：手写 schema 灵活但易与实现脱节、维护成本高；`@tool` 自动同步、类型安全、产出的是带 `Runnable` 能力的工具对象。复杂场景可用 `args_schema` 指定 Pydantic 模型（见 03-02）兼顾自动化与精细控制。
