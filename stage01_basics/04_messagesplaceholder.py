"""
MessagesPlaceholder 示例：动态插入历史对话记录
场景：让模型在多轮对话中保持上下文记忆
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from common.llm_provider import get_llm


llm = get_llm()

# 本质：在 system 和当前问题之间，预留一个"历史消息列表"的插槽
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位耐心的数学老师。回答要简洁，不超过两句话。"),
    MessagesPlaceholder(variable_name="history"),  # ← 动态占位符
    ("human", "{question}"),
])

parser = StrOutputParser()
chain = prompt | llm | parser


if __name__ == "__main__":
    # 模拟 3 轮对话历史
    history = [
        HumanMessage(content="1+1=？"),
        AIMessage(content="等于 2。"),
        HumanMessage(content="为什么不是 11？"),
        AIMessage(content="因为在十进制里，1+1 表示两个一相加，结果是 2。"),
    ]

    question = "那二进制呢？"

    print("=== 历史对话 ===")
    for msg in history:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"{role}: {msg.content}")
    print(f"\n当前问题: {question}\n")

    print("=== AI 回答 ===")
    result = chain.invoke({"history": history, "question": question})
    print(result)
