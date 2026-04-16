"""
PydanticOutputParser 最小示例：强制模型返回结构化数据
场景：让 AI 介绍一位名人，并以固定 JSON Schema 输出
"""

from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from common.llm_provider import get_llm


# ------------------
# 1. 定义输出 Schema（Pydantic Model）
# ------------------
class Celebrity(BaseModel):
    name: str = Field(description="人物姓名")
    birth_year: int = Field(description="出生年份")
    field: str = Field(description="所属领域，如科技/艺术/体育")
    achievements: list[str] = Field(description="主要成就列表，至少 3 项")
    summary: str = Field(description="100 字以内的人物简介")


# ------------------
# 2. 初始化解析器
# ------------------
parser = PydanticOutputParser(pydantic_object=Celebrity)

# ------------------
# 3. 获取 LLM 单例
# ------------------
llm = get_llm()

# ------------------
# 4. 构建 Prompt（把 Schema 说明注入 system 提示）
# ------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一位知识渊博的百科助手。请根据用户要求介绍人物，并按指定格式输出。\n"
            "{format_instructions}",
        ),
        (
            "human",
            "请介绍人物：{person_name}",
        ),
    ]
).partial(format_instructions=parser.get_format_instructions())

# ------------------
# 5. 组装 Chain
# ------------------
chain = prompt | llm | parser

# ------------------
# 6. 运行
# ------------------
if __name__ == "__main__":
    person_name = "埃隆·马斯克"
    print(f"正在查询人物：{person_name}\n")

    result: Celebrity = chain.invoke({"person_name": person_name})

    print("解析后的结构化数据：")
    print(f"  姓名: {result.name}")
    print(f"  出生年份: {result.birth_year}")
    print(f"  领域: {result.field}")
    print(f"  成就: {result.achievements}")
    print(f"  简介: {result.summary}")
    print(f"\n原始对象: {result}")
