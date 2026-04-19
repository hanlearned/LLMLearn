"""
验证：当模型输出 Markdown 代码块时，PydanticOutputParser 是否会报错
"""

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from common.llm_provider import get_llm


class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")


parser = PydanticOutputParser(pydantic_object=Person)
llm = get_llm()

# 故意诱导模型输出 Markdown 代码块
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个 JSON 生成器。请输出 JSON 格式的人物信息。"),
    ("human", "介绍一个名为 {name} 的 {age} 岁人物。"),
])

chain = prompt | llm | parser

if __name__ == "__main__":
    try:
        result = chain.invoke({"name": "张三", "age": 28})
        print("成功：", result)
    except Exception as e:
        print(f"报错类型：{type(e).__name__}")
        print(f"报错信息：{e}")
