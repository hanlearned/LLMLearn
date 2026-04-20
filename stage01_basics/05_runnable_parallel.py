"""
RunnableParallel 示例：并行执行多个子任务
目标：理解 LangChain 的并行调用机制，掌握串并联 Chain 的组合方式
"""

from dotenv import load_dotenv

load_dotenv()

from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from common.llm_provider import get_llm


llm = get_llm()

# 分支 1：生成笑话
joke_chain = (
    ChatPromptTemplate.from_template("讲一个关于{topic}的笑话，30字以内。")
    | llm
    | StrOutputParser()
)

# 分支 2：生成短诗
poem_chain = (
    ChatPromptTemplate.from_template("写一首关于{topic}的短诗，30字以内。")
    | llm
    | StrOutputParser()
)

# 分支 3：本地计算（不走 LLM，模拟"统计字数"专家）
count_chain = RunnableLambda(lambda x: f"'{x['topic']}' 共 {len(x['topic'])} 个字")

# RunnableParallel：一份输入，同时分发到三个分支
parallel = RunnableParallel(
    joke=joke_chain,
    poem=poem_chain,
    count=count_chain,
)

# 串并联混用：先并行生成素材，再串行整合为报告
report_prompt = ChatPromptTemplate.from_template("""\
请基于以下素材，写一段 50 字以内的介绍：

笑话：{joke}
诗歌：{poem}
统计：{count}

请用活泼的口吻整合：
""")
report_chain = parallel | report_prompt | llm | StrOutputParser()


if __name__ == "__main__":
    topic = "程序员"

    print("=== 1. 并行执行：三个分支同时跑 ===")
    result = parallel.invoke({"topic": topic})
    for k, v in result.items():
        print(f"{k}: {v}")

    print("\n=== 2. 串并联组合：并行素材 → 串行整合 ===")
    report = report_chain.invoke({"topic": topic})
    print(report)
