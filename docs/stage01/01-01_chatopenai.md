# 01-01 `ChatOpenAI`：配置与调用大模型

> 🎯 **目标**：掌握 `ChatOpenAI` 的初始化参数、多厂商兼容配置，以及同步/异步调用方式。

---

## 为什么用 `ChatOpenAI`？

`ChatOpenAI` 是 LangChain 对大模型交互的核心封装。它最大的价值在于**统一接口**：无论底层是 OpenAI、DeepSeek、Kimi 还是 Azure OpenAI，上层的调用方式完全一致。

这在实际工程中的意义是：**一处配置，随处调用**。想切换模型时，只需要改 `base_url` 和 `model`，业务代码完全不用动。

---

## 核心代码示例

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# DeepSeek
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.7,
)

# 同步调用
response = llm.invoke("请用一句话介绍 LangChain")
print(response.content)
```

---

## 关键参数详解

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `str` | 模型名称，如 `deepseek-chat`、`gpt-4o`、`moonshot-v1-8k` |
| `api_key` | `str` | API 密钥 |
| `base_url` | `str` | 自定义 API 基地址，用于兼容 OpenAI 格式的第三方服务 |
| `temperature` | `float` | 创造性控制，0.0-2.0，越低越稳定 |
| `max_tokens` | `int` | 单次返回的最大 token 数 |
| `streaming` | `bool` | 是否开启流式输出 |
| `timeout` | `int` | 请求超时时间（秒） |

---

## 多厂商配置速查

### OpenAI
```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)
```

### Kimi (Moonshot)
```python
llm = ChatOpenAI(
    model="moonshot-v1-8k",
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.cn/v1",
)
```

### 阿里云百炼
```python
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

---

## 异步调用

```python
response = await llm.ainvoke("请用一句话介绍 LangChain")
print(response.content)
```

异步调用在高并发场景（如 FastAPI 服务）中至关重要，后续 Stage 6 会大量用到。

---

## 常见报错

### `AuthenticationError: 401`
- API Key 无效、过期或未正确加载
- 检查 `.env` 文件名、环境变量名、Key 是否完整

### `NotFoundError: 404`
- `base_url` 写错了
- 模型名称在该平台不存在

### `RateLimitError: 429`
- 请求频率超限或余额不足

---

## 面试考点

**Q：为什么 LangChain 用 `ChatOpenAI` 封装 DeepSeek/Kimi？**  
A：因为它们都兼容 OpenAI 的 REST API 格式（`/v1/chat/completions`）。LangChain 通过统一接口屏蔽了底层差异，使得模型切换对业务层透明。

**Q：`temperature` 和 `top_p` 的区别？**  
A：`temperature` 控制采样随机性，`top_p` 控制核采样范围。一般只调其中一个，不建议同时调整。

---

## 课后任务

- [ ] 用 `ChatOpenAI` 成功调用一次 DeepSeek/OpenAI/Kimi
- [ ] 尝试把 `temperature` 分别设为 `0.0` 和 `1.0`，观察回答差异
- [ ] 用 `ainvoke()` 做一次异步调用（在 `async def main()` 里执行）
