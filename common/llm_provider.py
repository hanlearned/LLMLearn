"""
多厂商 LLM 统一封装（升级版）

设计目标
--------
1. **一份 .env 通吃**：自动探测你填了哪个厂商的 Key（SiliconFlow / DeepSeek / Kimi / OpenAI），
   填哪个就用哪个，无需改代码。
2. **单例**：同一进程内同一配置只创建一次实例，省去重复初始化与连接开销。
3. **可覆盖**：需要更高/更低随机性、换模型、换厂商时，用参数临时覆盖。

为什么 Agent 阶段尤其重要？
--------------------------
Agent 要做工具选择（Tool Calling），必须用「支持 function calling 的对话模型」。
本封装默认选用各厂商支持工具调用的主力 chat 模型，避免你在 Stage 3 才发现模型不支持工具。

用法
----
    from common.llm_provider import get_llm
    llm = get_llm()                     # 自动探测厂商，默认温度
    llm = get_llm(temperature=0)        # Agent/工具调用建议温度=0，结果更稳定
    llm = get_llm(model="Qwen/Qwen2.5-72B-Instruct")  # 指定模型
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_openai import ChatOpenAI

# 厂商优先级：探测到谁的 Key 就用谁（越靠前优先级越高）。
# 每一项：环境变量名 -> (base_url, 默认 chat 模型)
_PROVIDERS = [
    ("SILICONFLOW_API_KEY", "https://api.siliconflow.cn/v1", "deepseek-ai/DeepSeek-V3"),
    ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "deepseek-chat"),
    ("MOONSHOT_API_KEY", "https://api.moonshot.cn/v1", "moonshot-v1-8k"),
    ("OPENAI_API_KEY", None, "gpt-4o-mini"),  # base_url=None 走官方默认
]


def _detect_provider() -> tuple[str, str | None, str]:
    """返回 (api_key, base_url, default_model)，找不到任何 Key 时抛出友好错误。"""
    for env_name, base_url, default_model in _PROVIDERS:
        key = os.getenv(env_name)
        if key and not key.startswith("your_"):  # 跳过 .env.example 里的占位符
            return key, base_url, default_model
    raise RuntimeError(
        "未检测到任何可用的 API Key。请在项目根目录的 .env 中填入以下之一：\n"
        "  SILICONFLOW_API_KEY（推荐，注册送额度：https://cloud.siliconflow.cn/）\n"
        "  DEEPSEEK_API_KEY / MOONSHOT_API_KEY / OPENAI_API_KEY"
    )


@lru_cache(maxsize=None)
def _build_llm(model: str | None, temperature: float, max_tokens: int | None) -> ChatOpenAI:
    """按 (model, temperature, max_tokens) 缓存实例——参数相同就复用同一对象。"""
    api_key, base_url, default_model = _detect_provider()
    return ChatOpenAI(
        model=model or default_model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=60,
        max_retries=2,
    )


def get_llm(
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """
    获取一个（缓存的）LLM 实例。

    参数
    ----
    model: 指定模型名；None 表示用所探测厂商的默认 chat 模型。
    temperature: 采样温度。**做 Agent / 工具调用 / 结构化输出时建议传 0**。
    max_tokens: 最大生成长度，None 表示不限制（用模型默认）。
    """
    return _build_llm(model, temperature, max_tokens)


if __name__ == "__main__":
    # 自检：python common/llm_provider.py
    from dotenv import load_dotenv

    load_dotenv()
    key, base, model = _detect_provider()
    print(f"已探测到厂商：base_url={base or 'OpenAI 官方'}，默认模型={model}")
    llm = get_llm(temperature=0)
    print("模型回复：", llm.invoke("用一句话证明你在线").content)
