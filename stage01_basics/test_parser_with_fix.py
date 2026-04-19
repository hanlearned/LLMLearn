"""
解决方案：在 parser 前加一个 JSON 提取节点，增强鲁棒性
"""

import re
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda


class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")


parser = PydanticOutputParser(pydantic_object=Person)


def extract_json(text: str) -> str:
    """
    从可能包含废话的文本中，提取第一个 JSON 对象或数组。
    支持 Markdown 代码块和普通文本混排。
    """
    # 优先匹配 ```json ... ``` 或 ``` ... ``` 里的内容
    code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # 如果没有代码块，尝试匹配第一个 { ... } 结构
    # 用简单的花括号匹配（非嵌套复杂场景的极简方案）
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0).strip()

    # 兜底：原样返回，让 parser 自己尝试
    return text.strip()


# 组装增强版 Chain：llm -> extract_json -> parser
# 注意：这里用 | 把函数也接进去了！RunnableLambda 会自动包装普通函数
robust_parser = RunnableLambda(extract_json) | parser

# 测试数据
noise_both = '以下是提取结果：\n{"name": "张三", "age": 28}\n如有疑问请随时联系。'
markdown_block = '好的：\n```json\n{"name": "李四", "age": 30}\n```\n结束。'

print("=== 场景：前后都有废话 ===")
try:
    result = robust_parser.invoke(noise_both)
    print(f"成功：{result}")
except Exception as e:
    print(f"失败：{type(e).__name__}: {e}")

print("\n=== 场景：Markdown 代码块 ===")
try:
    result = robust_parser.invoke(markdown_block)
    print(f"成功：{result}")
except Exception as e:
    print(f"失败：{type(e).__name__}: {e}")
