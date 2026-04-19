"""
Runnable.stream()：流式输出
目标：理解 LangChain 的流式调用机制，观察 chunk 是如何逐块传输的
"""

from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from common.llm_provider import get_llm


llm = get_llm()

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位擅长写短诗的诗人。"),
    ("human", "以 {theme} 为主题，写一首 4 行的现代诗。"),
])

parser = StrOutputParser()

chain = prompt | llm | parser


if __name__ == "__main__":
    theme = "深夜的代码"
    print(f"主题：{theme}\n")
    print("--- 流式输出（打字机效果）---\n")

    # stream() 返回一个生成器，每次 yield 一个文本片段（chunk）
    for chunk in chain.stream({"theme": theme}):
        # end="" 表示不换行，flush=True 表示立即输出到屏幕
        print(chunk, end="", flush=True)

    print("\n\n--- 输出结束 ---")
