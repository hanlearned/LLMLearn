"""
LLM 单例封装器

为什么用单例？
- ChatOpenAI 实例的创建涉及读取环境变量、网络客户端初始化等开销。
- 在一个应用生命周期内，通常只需要同一个模型配置（相同的 model / base_url / temperature）。
- 单例保证全局只有一份实例，避免重复配置和潜在的连接泄漏。
"""

import os
from langchain_openai import ChatOpenAI


class LLMProvider:
    """
    基于 __new__ 的单例模式封装。
    用法：llm = LLMProvider().llm
    """

    _instance: "LLMProvider" = None

    def __new__(cls) -> "LLMProvider":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._llm = ChatOpenAI(
                model="deepseek-chat",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1",
                temperature=0.7,
            )
        return cls._instance

    @property
    def llm(self) -> ChatOpenAI:
        """获取全局唯一的 LLM 实例。"""
        return self._llm


# 也可以导出一个函数式接口，用起来更短
def get_llm() -> ChatOpenAI:
    """快捷函数：返回单例 LLM 实例。"""
    return LLMProvider().llm
