"""
第一个 LangChain 程序：Hello LangChain
目标：理解 "Chain" 的核心思想 —— 把 Prompt、模型、解析器串成一条流水线
"""

from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（仍然需要在入口处加载一次）
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from common.llm_provider import get_llm

# ------------------
# 1. 获取已封装好的 LLM 单例
# ------------------
llm = get_llm()

# ------------------
# 2. 构建 Prompt
# ------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一位善于用通俗类比讲解技术概念的助教。请用中文回答。",
        ),
        (
            "human",
            "请用 {style} 的风格解释什么是 {concept}，并用一个生活中的类比说明。",
        ),
    ]
)

# ------------------
# 3. 输出解析器
# ------------------
parser = StrOutputParser()

# ------------------
# 4. 组装 Chain（流水线）
# ------------------
chain = prompt | llm | parser

# ------------------
# 5. 运行 Chain
# ------------------
if __name__ == "__main__":
    concept = "LangChain"
    style = "给大学生讲课"
    print(f"正在向 AI 提问：请用「{style}」的风格解释什么是 {concept}\n")

    response = chain.invoke({"concept": concept, "style": style})

    print("AI 的回答：")
    print(response)
