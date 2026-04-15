# Stage 1 · Lesson 01：Hello LangChain —— 你的第一条 Chain

> 🎯 **本课目标**：跑通第一个 LangChain 程序，理解 **Chain（流水线）** 的核心设计思想。

---

## 一、为什么不是直接调 OpenAI API？

直接写 `requests.post()` 调大模型当然可以，但 LangChain 的价值在于**把 Prompt、模型调用、输出解析、上下文记忆等环节组装成一条可复用、可观测、可替换的流水线**。

这在招聘面试中被称为 **"LLM 应用工程化能力"**。

---

## 二、环境准备

### 1. 创建虚拟环境
```bash
cd D:\kimi_projects\learn_llm
python -m venv venv
```

### 2. 激活虚拟环境并安装依赖
```bash
.\venv\Scripts\Activate.ps1
pip install langchain langchain-openai langchain-community openai python-dotenv
```

### 3. 准备 API Key
在 `D:\kimi_projects\learn_llm\` 目录下创建 `.env` 文件：
```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```
> 💡 推荐用 [DeepSeek 开放平台](https://platform.deepseek.com/)，新用户送 10 元额度，国内访问稳定。

---

## 三、核心代码

文件路径：`stage01_basics/01_hello_langchain.py`

```python
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.7,
)

# 2. Prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位善于用通俗类比讲解技术概念的助教。请用中文回答。"),
    ("human", "请用 {style} 的风格解释什么是 {concept}，并用一个生活中的类比说明。"),
])

# 3. 输出解析器
parser = StrOutputParser()

# 4. 组装 Chain（LCEL 语法）
chain = prompt | llm | parser

# 5. 运行
if __name__ == "__main__":
    response = chain.invoke({
        "concept": "LangChain",
        "style": "给大学生讲课"
    })
    print(response)
```

---

## 四、逐行讲解：4 个核心抽象

### 1. ChatPromptTemplate —— 结构化 Prompt
- 区分 `system`（角色设定）和 `human`（用户输入）。
- 支持变量占位符 `{concept}`、`{style}`，运行时动态填充。
- **面试考点**：为什么不用 f-string？因为 PromptTemplate 可以被复用、被版本管理、被评估，是工业化 Prompt 管理的基础。

### 2. ChatOpenAI —— 模型封装
- DeepSeek、Kimi、智谱等国内模型大多兼容 OpenAI 接口格式，所以都能用 `ChatOpenAI` 调用。
- 好处：**一处配置，随处调用**。想换模型？改一行 `base_url` 和 `model` 即可。

### 3. StrOutputParser —— 统一输出格式
- 模型原始返回的是 `AIMessage` 对象（含 metadata）。
- `StrOutputParser()` 自动提取 `.content`，让下游只处理纯文本。
- 下节课我们会把它换成 `PydanticOutputParser`，直接输出结构化 JSON。

### 4. LCEL Chain —— 流水线语法
```python
chain = prompt | llm | parser
```
这是 LangChain 推荐的现代写法，称为 **LCEL（LangChain Expression Language）**。
- `|` 表示数据从左到右流动。
- 支持 `.invoke()`、`.batch()`、`.stream()`、`.astream()`。
- 天然可接入 LangSmith 做追踪观测。

---

## 五、常见报错排查

### ❌ 401 AuthenticationError
**原因**：API Key 无效、过期、或填错了变量名。

**排查步骤**：
1. 检查 `.env` 文件名是否正确（不是 `env` 也不是 `.env.txt`）。
2. 检查 Key 是否完整复制（以 `sk-` 开头，无引号、无空格）。
3. 检查代码里读取的环境变量名是否和 `.env` 里的一致（`DEEPSEEK_API_KEY`）。
4. 注意：Kimi Code CLI 的 Key 不能用于调用 Moonshot Open API。

### ❌ ModuleNotFoundError
**原因**：没有在虚拟环境里安装依赖，或虚拟环境未激活。

**解决**：
```bash
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt  # 或重新安装缺失的包
```

---

## 六、课后任务（Checkpoint）

完成以下任务，确认本课知识点已掌握：

- [ ] 成功运行 `01_hello_langchain.py` 并看到 AI 回答。
- [ ] 修改 `concept` 变量为 `"RAG"`，观察 AI 如何解释。
- [ ] 修改 `style` 变量为 `"给小学生讲课"` 或 `"脱口秀演员"`，观察风格变化。
- [ ] （可选）尝试把 `temperature` 改为 `0.0` 和 `1.0`，观察回答的稳定性/创造性变化。

---

## 七、下节预告

**Lesson 02：结构化输出 —— PydanticOutputParser**

我们将把 AI 的纯文本回答，强制约束成精确的 JSON 格式。这是 **简历解析器（项目 1）** 的核心技术。

我们将学习：
- 如何定义 Pydantic Schema
- 如何让 LLM 乖乖按格式输出
- 输出不符合格式时如何自动重试

---

## 附录：运行命令速查

```bash
cd D:\kimi_projects\learn_llm
.\venv\Scripts\Activate.ps1
python stage01_basics\01_hello_langchain.py
```
