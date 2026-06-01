# 03-02 `StructuredTool` 与 `args_schema`：复杂多参数工具与参数校验

> 🎯 **一句话**：当工具参数变多、变复杂、需要校验和详细说明时，用 Pydantic 模型（`args_schema`）精确定义参数，让模型传参更准、出错更早暴露。

---

## 为什么需要它

`@tool` 适合简单工具，但真实业务工具往往是「多参数 + 有约束」的，比如「下单」需要 `商品ID`、`数量（>0）`、`收货地址`，还得对每个参数给模型一段解释。光靠函数签名和一句 docstring，描述能力不够，也无法表达「数量必须为正整数」这类约束。

`StructuredTool` + Pydantic `args_schema` 解决两件事：
1. **更丰富的参数说明**：用 `Field(description=...)` 给每个参数单独写说明，模型选参更准。
2. **参数校验**：调用前用 Pydantic 校验类型与约束，非法参数（如负数量）直接报错，而不是让错误数据流进业务逻辑。

---

## 核心用法

```python
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain_core.tools import tool


# 1. 用 Pydantic 定义参数 Schema（每个字段都能单独写说明 + 加约束）
class CreateOrderInput(BaseModel):
    product_id: str = Field(description="商品的唯一编号，例如 'SKU-1001'")
    quantity: int = Field(description="购买数量，必须为正整数", gt=0)
    address: str = Field(description="收货地址全称")


# 2. 把 Schema 挂到工具上
@tool(args_schema=CreateOrderInput)
def create_order(product_id: str, quantity: int, address: str) -> str:
    """创建一个商品订单。当用户明确表达要下单/购买时调用。"""
    return f"已为商品 {product_id} 创建 {quantity} 件订单，寄往 {address}。"


print(create_order.args)
# product_id / quantity / address 三个字段，各自带 description 与约束

# 校验生效：传非法数量会直接抛错
create_order.invoke({"product_id": "SKU-1001", "quantity": 2, "address": "北京朝阳"})
# create_order.invoke({..., "quantity": -1})  # 触发 Pydantic 校验错误
```

**本质在干什么？**

- `CreateOrderInput` 是这个工具的「参数契约」。`Field(description=...)` 的每段文字都会进入发给模型的 Schema，相当于逐参数地给模型写说明。
- `gt=0`（greater than 0）是约束。`.invoke` 时 LangChain 先用 Pydantic 校验入参，不合法立刻抛 `ValidationError`，挡在业务代码之前。
- `@tool(args_schema=...)` 让装饰器不再去函数签名推断参数，而是直接采用你的 Pydantic 模型——表达力更强。

不用装饰器，也可以用 `StructuredTool.from_function` 显式构造，适合「函数已存在、不想加装饰器」或「运行时动态造工具」的场景：

```python
from langchain_core.tools import StructuredTool

def _create_order(product_id: str, quantity: int, address: str) -> str:
    return f"已为 {product_id} 下单 {quantity} 件，寄往 {address}。"

create_order_tool = StructuredTool.from_function(
    func=_create_order,
    name="create_order",
    description="创建一个商品订单。当用户明确要下单/购买时调用。",
    args_schema=CreateOrderInput,
)
```

**本质在干什么？** `@tool` 其实是 `StructuredTool.from_function` 的语法糖。显式构造让你能把「函数实现」和「工具定义」解耦，并在运行时灵活组装（比如批量为多个函数生成工具）。

绑定给模型后，模型会按 Schema 把多个参数一次性填好：

```python
from common.llm_provider import get_llm

llm = get_llm(temperature=0)
resp = llm.bind_tools([create_order]).invoke("帮我买 3 件 SKU-1001，寄到上海浦东")
print(resp.tool_calls)
# [{'name': 'create_order',
#   'args': {'product_id': 'SKU-1001', 'quantity': 3, 'address': '上海浦东'}, ...}]
```

---

## 关键原理

1. **`@tool` ≈ `StructuredTool.from_function`**：单参数简单工具用前者，多参数/动态构造用后者。两者产物都是 `StructuredTool`。
2. **Schema 双重作用**：对内是校验器（Pydantic 校验入参），对外是说明书（序列化成 function schema 发给模型）。一份定义两头用。
3. **校验失败的处理**：默认抛异常。在 Agent 循环里，可配置把校验错误作为 `Observation` 反馈给模型，让它「自我纠正」重试（`handle_tool_error`）。
4. **`Field` 的约束会进 Schema**：`gt`、`enum`（用 `Literal`）等约束部分会体现在 JSON Schema 里，能引导模型少传非法值——但模型不保证 100% 遵守，校验仍是最后防线。

---

## 你来改

- [ ] 给 `quantity` 增加上限 `le=100`（最多 100 件），并用 `Literal["顺丰","圆通"]` 加一个 `express` 字段，观察 `args` 变化。
- [ ] 把 `create_order` 改成 `StructuredTool.from_function` 写法，确认行为一致。
- [ ] 让模型下一个「数量为 -5」的订单（诱导它传非法值），观察校验在什么阶段拦截。

---

## 面试怎么考

**Q：`@tool` 和 `StructuredTool` 是什么关系？什么时候必须用 `args_schema`？**
A：`@tool` 是 `StructuredTool.from_function` 的语法糖，产物都是 `StructuredTool`。当工具参数多、需要逐参数详细说明、需要类型/数值约束校验，或要在运行时动态构造工具时，就用 Pydantic `args_schema`。

**Q：参数校验失败了，在 Agent 流程里应该怎么处理？**
A：默认抛 `ValidationError`。生产里更常见的做法是捕获后把错误信息作为 Observation 回喂给模型，让它根据报错修正参数重试一次；同时记录日志。框架层可用工具的 `handle_tool_error` 配置实现自动兜底。

**Q：为什么要在工具层做参数校验，而不是只靠模型「好好填参数」？**
A：模型是概率生成的，不能保证参数合法。校验是确定性的最后防线，能在脏数据进入数据库/支付等关键操作前拦截，避免不可逆后果。Schema 约束只是「引导」模型，Pydantic 校验才是「保证」。
