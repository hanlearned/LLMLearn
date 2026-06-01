# 06-06 本地大模型：Ollama 与 vLLM 的适用场景对比

> 🎯 **一句话**：把模型跑在自己机器上而非调云 API——Ollama 让个人/原型用一行命令在本地起一个 OpenAI 兼容服务，vLLM 则面向生产用高吞吐推理引擎扛并发；两者都能让你用熟悉的 `ChatOpenAI` 指向本地 `base_url` 直接调用。

---

## 为什么需要它

调云 API 有三个痛点：**数据隐私**（敏感数据不能出公司）、**成本**（高频调用 token 费惊人）、**可控性**（限流、涨价、模型下线受制于人）。

本地部署把模型放进自己的基础设施：数据不出门、按硬件一次性投入换无限调用、版本完全自主。代价是要自己管 GPU、显存、运维。Ollama 解决「个人怎么轻松跑」，vLLM 解决「生产怎么高吞吐扛住并发」。

---

## 核心用法

### 1. Ollama：一行拉模型，OpenAI 兼容接口

```bash
# 安装后拉取并运行模型（自动下载权重）
ollama pull qwen2.5:7b
ollama run qwen2.5:7b "你好"          # 命令行直接聊

# Ollama 默认在 11434 端口提供 OpenAI 兼容接口
```

```python
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI

# 关键：用熟悉的 ChatOpenAI，只把 base_url 指向本地 Ollama
llm = ChatOpenAI(
    model="qwen2.5:7b",
    base_url="http://localhost:11434/v1",   # Ollama 的 OpenAI 兼容端点
    api_key="ollama",                        # 本地无需真 key，占位即可
    temperature=0,
)
print(llm.invoke("用一句话介绍你自己").content)
```

**逐块讲解：**
- **`ollama pull` / `run`**：一条命令搞定下载 + 启动，对个人极友好，自动处理量化、显存。
- **OpenAI 兼容接口**：Ollama 在 `/v1` 暴露与 OpenAI 一致的接口，所以**代码几乎不用改**——`ChatOpenAI` 把 `base_url` 一指就能用。这正是本项目 `common/llm_provider.py` 用 `ChatOpenAI` + base_url 设计的价值：换本地模型零成本。
- **`api_key="ollama"`**：本地服务不校验 key，给个占位字符串满足 SDK 要求即可。

### 2. vLLM：高吞吐生产级 serving

```bash
# 用 vLLM 起一个 OpenAI 兼容的 API 服务（需要 GPU）
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000
```

```python
from langchain_openai import ChatOpenAI

# 同样是 ChatOpenAI 指 base_url，代码与调云端无差别
llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    temperature=0,
)
print(llm.invoke("解释什么是 PagedAttention").content)
```

**本质在干什么？** vLLM 是专为高并发推理优化的引擎，核心技术 PagedAttention + continuous batching 让它在多请求下显存利用率和吞吐远超朴素部署。它同样暴露 OpenAI 兼容接口，接入方式和 Ollama 一模一样——区别只在它为「同时扛大量请求」而生。

---

## 关键原理 / 实践要点

1. **统一接入方式**：Ollama 和 vLLM 都提供 OpenAI 兼容接口，LangChain 里都用 `ChatOpenAI` + `base_url` 接入，业务代码与调云端一致——**换模型只改配置不改逻辑**。
2. **Ollama 定位**：个人开发、本地原型、低并发。开箱即用、自动量化、CPU/小显存也能跑小模型；但吞吐有限，不适合扛高并发生产流量。
3. **vLLM 定位**：生产高吞吐 serving。靠 PagedAttention + continuous batching 把 GPU 吃满，同时服务大量并发请求；但部署门槛高、需要像样的 GPU、运维更重。
4. **选型口诀**：本地试玩/单人用/演示 → Ollama；线上服务、要扛并发、追求吞吐与延迟 → vLLM。
5. **本地 ≠ 免费**：省了 token 费，但 GPU 采购/电费/运维是实打实成本。低频场景云 API 往往更划算，高频或强隐私场景本地部署才回本。
6. **隐私与合规是核心驱动力**：医疗、金融、政企等数据不能出域的场景，本地部署常是硬性要求，而非成本选择。

---

## 你来改

- [ ] 用 Ollama 拉一个小模型（如 `qwen2.5:1.5b`），把本项目 `common/llm_provider.py` 的 base_url 临时指向它，跑通一个 Stage 1 示例。
- [ ] 对比同一问题在云端模型和本地 Ollama 小模型上的回答质量与延迟。
- [ ] 查阅 vLLM 的 continuous batching，用一句话解释它为什么比「逐个请求处理」吞吐高。

---

## 面试怎么考

**Q：为什么要本地部署大模型？有什么代价？**
A：三大驱动力——数据隐私（敏感数据不出域）、成本（高频调用省 token 费）、可控性（不受限流/涨价/下线影响）。代价是要自购并运维 GPU、管显存、做扩缩容；低频场景云 API 往往更划算，高频或强合规场景本地才回本。

**Q：Ollama 和 vLLM 怎么选？**
A：Ollama 面向个人/原型/低并发，一行命令拉模型、自动量化、开箱即用，但吞吐有限；vLLM 面向生产高吞吐 serving，靠 PagedAttention + continuous batching 吃满 GPU 扛大量并发，但部署门槛和运维成本高。本地试玩用 Ollama，线上扛并发用 vLLM。

**Q：本地模型怎么接进 LangChain？**
A：两者都暴露 OpenAI 兼容接口，直接用 `ChatOpenAI` 把 `base_url` 指向本地服务（Ollama 默认 11434，vLLM 自定义端口），api_key 给占位字符串即可，业务代码与调云端完全一致。这意味着换模型只改配置、不改逻辑。
