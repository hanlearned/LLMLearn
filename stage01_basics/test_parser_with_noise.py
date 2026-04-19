"""
测试更真实的失败场景：JSON 前后夹杂了解释性文字
"""

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")


parser = PydanticOutputParser(pydantic_object=Person)

# 场景 3：前面有解释，后面有 JSON
noise_before = '好的，以下是符合您要求的 JSON 格式数据：\n{"name": "张三", "age": 28}'

# 场景 4：JSON 后面还有废话
noise_after = '{"name": "张三", "age": 28}\n希望这个回答对您有帮助！'

# 场景 5：前后都有废话
noise_both = '以下是提取结果：\n{"name": "张三", "age": 28}\n如有疑问请随时联系。'

scenarios = [
    ("前面有废话", noise_before),
    ("后面有废话", noise_after),
    ("前后都有废话", noise_both),
]

for desc, text in scenarios:
    print(f"=== {desc} ===")
    try:
        result = parser.parse(text)
        print(f"成功：{result}\n")
    except Exception as e:
        print(f"失败：{type(e).__name__}")
        print(f"信息：{e}\n")
