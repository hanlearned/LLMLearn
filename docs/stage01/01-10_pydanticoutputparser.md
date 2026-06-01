# 01-10 `PydanticOutputParser`：结构化对象输出与 Schema 校验

> 🎯 **目标**：掌握用 `PydanticOutputParser` 强制模型输出结构化 JSON，并理解其内置的容错机制与边界。

---

## 一句话功能

`PydanticOutputParser` 是 LangChain 中最常用的**非函数调用式结构化输出解析器**。它把 Pydantic 模型的 Schema 自动转成 Prompt 指令，让模型按固定格式输出 JSON，再自动反序列化为类型安全的 Python 对象。

---

## 核心代码示例

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class Celebrity(BaseModel):
    name: str = Field(description="人物姓名")
    birth_year: int = Field(description="出生年份")
    achievements: list[str] = Field(description="主要成就列表")

parser = PydanticOutputParser(pydantic_object=Celebrity)

# 获取格式说明，自动注入 Prompt
format_instructions = parser.get_format_instructions()
```

`get_format_instructions()` 会生成一段包含完整 JSON Schema 的文本，告诉模型必须输出哪些字段、什么类型。

---

## 完整 Chain 示例

```python
from langchain_core.prompts import ChatPromptTemplate
from common.llm_provider import get_llm

llm = get_llm()

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位百科助手。{format_instructions}"),
    ("human", "请介绍人物：{person_name}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser

result = chain.invoke({"person_name": "埃隆·马斯克"})
# result 是 Celebrity 对象，可直接 .name / .birth_year
```

---

## 内置容错：Markdown 代码块自动剥离

### 问题背景
开发者常担心：如果模型不听话，在 JSON 外面包了 Markdown 代码块（```json ... ```），解析会不会失败？

### 实际行为
**不会失败。** `PydanticOutputParser` 继承自 `JsonOutputParser`，其 `parse()` 方法内部调用了 `parse_json_markdown()`。该函数的核心逻辑是：

1. 先尝试直接解析整个字符串为 JSON
2. 若失败，用正则匹配 ```` ```json ... ``` ```` 里的内容
3. 提取后再解析

### 源码验证（langchain-core）

```python
# langchain_core/utils/json.py
def parse_json_markdown(json_string: str, ...) -> Any:
    try:
        return _parse_json(json_string, parser=parser)
    except json.JSONDecodeError:
        # Try to find JSON string within triple backticks
        match = _json_markdown_re.search(json_string)
        json_str = json_string if match is None else match.group(2)
    return _parse_json(json_str, parser=parser)
```

### 版本信息

| 组件 | 版本说明 |
|------|---------|
| `PydanticOutputParser` | **Since v0.1**（langchain-core 独立后即存在） |
| `parse_json_markdown` | LangChain 0.0.x 时代已引入，langchain-core 0.1+ 保留 |
| 本文验证环境 | `langchain-core==1.2.28` |

---

## 清洗边界：它能处理什么，不能处理什么

| 模型输出形式 | 能否自动处理 | 说明 |
|:---|:---|:---|
| `{"name":"张三"}` | ✅ | 直接解析 |
| ```` ```json\n{"name":"张三"}\n``` ```` | ✅ | 正则提取代码块内容 |
| ```` ```\n{"name":"张三"}\n``` ```` | ✅ | 无语言标记的代码块也能匹配 |
| `{"name":"张三"}\n谢谢！` | ✅ | 尾部空白字符被 strip |
| `好的：\n{"name":"张三"}` | ❌ | **JSON 前面有解释性文字，正则无法匹配** |

**结论**：Markdown 代码块和首尾空白无需担心；真正需要防范的是**JSON 前面夹着废话**。

---

## 增强方案：前置清洗节点

如果生产环境对稳定性要求极高，可以在 `llm` 和 `parser` 之间加一个 `RunnableLambda` 做兜底清洗：

```python
import re
from langchain_core.runnables import RunnableLambda

def extract_json(text: str) -> str:
    # 优先匹配代码块
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # 兜底匹配第一个 { ... }
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    return brace_match.group(0).strip() if brace_match else text.strip()

# 组装增强版 Chain
robust_chain = prompt | llm | RunnableLambda(extract_json) | parser
```

---

## 面试考点

**Q：`PydanticOutputParser` 和 `JsonOutputParser` 的区别？**
A：`PydanticOutputParser` 继承自 `JsonOutputParser`，额外增加了 Pydantic 模型的类型校验能力。`JsonOutputParser` 只保证输出是合法 JSON；`PydanticOutputParser` 还会校验字段类型、必填项、枚举值等。

**Q：模型输出 Markdown 代码块时，`PydanticOutputParser` 会报错吗？**
A：**不会。** 自 LangChain 早期版本（0.0.x）起，内部就通过 `parse_json_markdown` 自动剥离 Markdown 代码块。但 JSON 前后夹杂解释性文字时仍会失败，此时需要用 `RunnableLambda` 做前置清洗，或配置 `.with_fallbacks()` 降级。

**Q：`get_format_instructions()` 生成的内容是什么？**
A：一段自然语言 + JSON Schema 的混合文本，明确告诉模型输出格式、字段名称、字段类型、必填项等。这是 Zero-Shot 场景下让模型听话的核心手段。

---

## 课后任务

- [ ] 用 `PydanticOutputParser` 定义一个包含嵌套模型的 Schema（如 `Resume` 包含 `List[WorkExperience]`）
- [ ] 验证：让模型把 JSON 包在 ```` ```json ```` 里，观察解析是否成功
- [ ] 在 Chain 中加入 `RunnableLambda(extract_json)`，验证"JSON 前有废话"的场景也能成功解析
