"""
项目 1：结构化简历解析器
目标：输入一段非结构化的简历文本，让 LLM 提取关键信息并按固定 Schema 输出
"""

from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from common.llm_provider import get_llm


# ------------------
# 1. 定义简历 Schema
# ------------------
class WorkExperience(BaseModel):
    company: str = Field(description="公司名称")
    position: str = Field(description="职位")
    duration: str = Field(description="工作时间段，如 2020.03 - 2023.08")
    responsibilities: List[str] = Field(description="主要职责或业绩，至少 2 项")


class Education(BaseModel):
    school: str = Field(description="学校名称")
    degree: str = Field(description="学历/学位，如本科、硕士")
    major: str = Field(description="专业")
    graduation_year: int = Field(description="毕业年份")


class Resume(BaseModel):
    name: str = Field(description="姓名")
    phone: str = Field(description="手机号码")
    email: str = Field(description="邮箱地址")
    summary: str = Field(description="100 字以内的个人简介")
    skills: List[str] = Field(description="技能列表，至少 5 项")
    education: List[Education] = Field(description="教育经历列表")
    work_experience: List[WorkExperience] = Field(description="工作经历列表")


# ------------------
# 2. 初始化解析器
# ------------------
parser = PydanticOutputParser(pydantic_object=Resume)

# ------------------
# 3. 获取 LLM 单例
# ------------------
llm = get_llm()

# ------------------
# 4. 构建 Prompt
# ------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一位资深的人力资源信息提取专家。你的任务是从非结构化的简历文本中，"
            "提取关键信息并严格按照要求的 JSON Schema 格式输出。不要添加任何解释性文字。\n"
            "{format_instructions}",
        ),
        (
            "human",
            "以下是候选人的简历文本，请提取结构化信息：\n\n{resume_text}",
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
    resume_text = """
    姓名：张伟
    电话：138-0013-8000
    邮箱：zhangwei@example.com

    个人简介：
    拥有 5 年后端开发经验的工程师，熟悉高并发系统设计与微服务架构，
    热衷于用技术解决业务痛点，具备良好的团队协作能力。

    技能：
    Python、Go、Docker、Kubernetes、MySQL、Redis、Kafka、Microservices、CI/CD

    教育经历：
    2013 - 2017，北京大学，计算机科学与技术，本科
    2017 - 2020，清华大学，软件工程，硕士

    工作经历：
    1. 字节跳动，高级后端开发工程师，2020.03 - 2023.08
       - 负责抖音电商订单核心链路设计，支撑日均千万级订单流量
       - 主导订单服务从单体架构迁移至微服务，系统可用性从 99.9% 提升至 99.99%
       - 设计并落地分布式缓存一致性方案，降低数据库峰值压力 40%

    2. 阿里巴巴，后端开发工程师，2018.06 - 2020.02
       - 参与淘宝商品详情页性能优化项目，页面首屏加载时间缩短 30%
       - 负责内部运维工具链开发，提升团队发布效率 20%
    """

    print("正在解析简历，请稍候...\n")
    result: Resume = chain.invoke({"resume_text": resume_text})

    print(f"姓名: {result.name}")
    print(f"电话: {result.phone}")
    print(f"邮箱: {result.email}")
    print(f"简介: {result.summary}")
    print(f"技能: {', '.join(result.skills)}")
    print("\n教育经历:")
    for edu in result.education:
        print(f"  - {edu.school} | {edu.degree} | {edu.major} | {edu.graduation_year}")
    print("\n工作经历:")
    for work in result.work_experience:
        print(f"  - {work.company} | {work.position} | {work.duration}")
        for resp in work.responsibilities:
            print(f"    · {resp}")
    print(f"\n原始对象: {result}")
