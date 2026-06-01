# `JsonOutputParser`：强制模型输出可解析的 JSON

`JsonOutputParser` 把模型返回的文本解析成 Python `dict`，并能在 prompt 里自动注入「请输出 JSON」的格式说明。

## 为什么需要它

业务里经常需要**结构化结果**（一个 dict）而不是一段话——比如让模型抽取 `{"姓名":..., "电话":...}`。如果只用 `StrOutputParser`，你拿到的是一坨文本，还得自己 `json.loads`，而且模型常常多嘴加上 ```json 代码块标记导致解析失败。`JsonOutputParser` 帮你：① 在 prompt 里加格式指令；② 容错解析（自动清掉代码块包裹）。

## 核心用法

```python
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from common.llm_provider import get_llm

parser = JsonOutputParser()
prompt = ChatPromptTemplate.from_template(
    "从下面文本抽取联系人，返回 JSON（字段 name、phone）。\n{format_instructions}\n文本：{text}"
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | get_llm(temperature=0) | parser
result = chain.invoke({"text": "联系人张三，电话 13800001111"})
print(result["name"], result["phone"])   # result 是 dict
```

**逐块讲解**：
- `parser.get_format_instructions()`：生成一段「请输出合法 JSON」的提示，用 `.partial` 预填进 prompt。
- `| parser`：把模型输出的文本解析成 dict，自动处理 ```json 包裹。
- `result` 是 `dict`，可直接下标取值。

## 关键原理
解析失败（模型没吐合法 JSON）时会抛 `OutputParserException`。它和 `PydanticOutputParser` 的区别：JsonOutputParser 只保证「是合法 JSON」，不校验字段类型；要强类型校验和对象化，用下一篇的 `PydanticOutputParser`。

## 你来改
1. 故意把 `temperature` 调高、format_instructions 去掉，观察解析更容易失败。
2. 让模型抽取一个列表（多个联系人），看 `result` 变成 `list[dict]`。

## 面试怎么考
- **「怎么让模型稳定输出 JSON？」** → 加格式指令 + 低温度 + 用 JsonOutputParser/结构化输出兜底解析；新模型可直接用 `with_structured_output`。
- **「JsonOutputParser 和 PydanticOutputParser 选哪个？」** → 只要 dict 用前者；要类型校验、IDE 提示、嵌套模型用后者。
