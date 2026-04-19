"""
直接测试：PydanticOutputParser 面对 Markdown 代码块时的行为
"""

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")


parser = PydanticOutputParser(pydantic_object=Person)

# 场景 1：干净的 JSON（正常情况）
clean_json = '{"name": "张三", "age": 28}'

# 场景 2：包裹在 Markdown 代码块中（模型不听话时的常见输出）
markdown_json = '''```json
{"name": "张三", "age": ,28c}
```'''

print("=== 场景 1：干净 JSON ===")
try:
    result = parser.parse(clean_json)
    print(f"成功：{result}")
except Exception as e:
    print(f"失败：{type(e).__name__}: {e}")

print("\n=== 场景 2：Markdown 代码块包裹 ===")
try:
    result = parser.parse(markdown_json)
    print(parser.get_format_instructions())
    print(f"成功：{result}")
except Exception as e:
    print(f"失败：{type(e).__name__}: {e}")
