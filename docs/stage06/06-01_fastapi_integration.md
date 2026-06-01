# 06-01 FastAPI 集成：把 Chain / Agent 暴露成 REST 接口

> 🎯 **一句话**：用 FastAPI 把你的 LCEL Chain 或 Agent 包成 HTTP 接口，让前端、移动端、其他服务都能通过 REST 调用——这是 LLM 应用从「脚本」走向「线上服务」的第一步。

---

## 为什么需要它

`python xxx.py` 只能本地跑一次。真实产品需要前端能随时发请求拿结果、需要并发、需要参数校验、需要错误处理。

FastAPI 是 Python 生态做 LLM 服务的事实标准：**原生 async**（契合 LLM 高 I/O 等待，并发友好）、**Pydantic 请求/响应校验**（和 LangChain 的结构化输出一脉相承）、**自动生成 Swagger 文档**。把 Chain 包进 FastAPI，就得到一个可被任何客户端调用的标准服务。

---

## 核心用法

### 最小可运行 app

```python
# main.py — 运行：uvicorn main:app --reload
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from common.llm_provider import get_llm

app = FastAPI(title="LLM Service")

# 1) 请求 / 响应模型：Pydantic 负责校验与文档
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户提问")
    style: str = Field("简洁", description="回答风格")

class ChatResponse(BaseModel):
    answer: str

# 2) 依赖注入：把 chain 的构造交给 FastAPI，便于复用与测试替换
def get_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是助手，用{style}的风格回答。"),
        ("human", "{question}"),
    ])
    return prompt | get_llm(temperature=0) | StrOutputParser()

# 3) 接口：async def，内部用 ainvoke 不阻塞事件循环
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, chain=Depends(get_chain)):
    answer = await chain.ainvoke({"question": req.question, "style": req.style})
    return ChatResponse(answer=answer)

@app.get("/health")
async def health():
    return {"status": "ok"}
```

**逐块讲解：**
- **Pydantic 请求/响应模型**：`ChatRequest` 自动校验入参（缺字段、空字符串直接返回 422），`response_model` 约束出参结构。这和 LangChain 的结构化输出是同一套 Pydantic，无缝衔接。
- **依赖注入 `Depends`**：`get_chain` 作为依赖被注入，好处是逻辑解耦、测试时可轻松替换成 mock chain，重对象也能配合缓存复用。
- **`async def` + `ainvoke`**：LLM 调用是长时 I/O 等待。用异步接口 + `ainvoke`，等待期间事件循环能处理别的请求，并发能力远超同步阻塞。
- **`/health` 健康检查**：容器编排（K8s/Docker）靠它判断服务存活，生产标配。

### 启动与调用

```bash
uvicorn main:app --reload --port 8000
# 浏览器打开 http://localhost:8000/docs 就有交互式 Swagger 文档

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"什么是RAG？","style":"通俗"}'
```

**本质在干什么？** `uvicorn` 是 ASGI 服务器，把 `app` 跑起来。`/docs` 是 FastAPI 自动生成的 Swagger UI，可在浏览器直接试接口，无需写客户端。

### 暴露 Agent 同理

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(get_llm(temperature=0), tools=[...])

@app.post("/agent")
async def run_agent(req: ChatRequest):
    result = await agent.ainvoke({"messages": [("user", req.question)]})
    return {"answer": result["messages"][-1].content}
```

**本质在干什么？** Agent 也是 Runnable，同样有 `ainvoke`，包法和 Chain 完全一致——取最后一条 message 的 content 作为答案返回。

---

## 关键原理 / 实践要点

1. **一律用 async + ainvoke**：LLM 接口是 I/O 密集型，异步才能高并发。同步 `def` 接口里调阻塞 `invoke` 会拖垮吞吐。
2. **Pydantic 做契约**：请求/响应模型即 API 契约，自动校验 + 自动文档，减少前后端扯皮。
3. **依赖注入管理资源**：chain、向量库、数据库连接都通过 `Depends` 注入，利于复用、测试、生命周期管理。
4. **错误处理**：LLM 可能超时/限流，应捕获异常返回合适的 HTTP 状态码（如 503），别让 500 裸奔；可加超时与重试。
5. **不要每次请求都新建 LLM**：`get_llm` 已用 lru_cache 单例，避免重复建连接。重对象（向量库）也应进程级初始化一次。
6. **流式见下篇**：返回一大段文本体验差，生产常用 SSE 流式（06-03）。

---

## 你来改

- [ ] 给 `/chat` 加 try/except，模型超时时返回 503 和友好提示，而非 500。
- [ ] 用 `response_model` 把回答改成结构化对象（含 answer + 字数统计），体会 Pydantic 出参校验。
- [ ] 写一个 `/batch` 接口，用 `chain.abatch` 一次处理多个问题。

---

## 面试怎么考

**Q：为什么用 FastAPI 暴露 LLM 服务？接口为什么要 async？**
A：FastAPI 原生 async、Pydantic 校验、自动 Swagger 文档，契合 LLM 应用。LLM 调用是长时 I/O 等待，用 `async def` + `ainvoke` 能在等待期间处理其他请求，并发能力远超同步阻塞；同步接口里调阻塞 invoke 会拖垮吞吐。

**Q：Pydantic 模型和依赖注入在这里各起什么作用？**
A：Pydantic 请求/响应模型充当 API 契约，自动校验入参（非法返回 422）、约束出参、生成文档，且与 LangChain 结构化输出同源。依赖注入（Depends）把 chain、数据库等资源解耦注入，便于复用、测试替换和生命周期管理。

**Q：把 Chain 和 Agent 暴露成接口有区别吗？**
A：本质相同，都是 Runnable，都用 `ainvoke`。Chain 直接返回输出；Agent 返回的是 messages 列表，取最后一条的 content 即可。生产中还需加健康检查、异常处理、超时重试，长输出场景再上 SSE 流式。
