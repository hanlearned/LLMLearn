# 项目 1：结构化简历解析器

> Stage 1 的收官项目。把一段乱七八糟的简历文本，变成一个字段齐全、类型正确的结构化对象。这是「文档智能 / 信息抽取」最典型的落地，也是结构化输出能力的实战检验。
>
> 代码：`stage01_basics/project01_resume_parser.py`

---

## 一、需求与方案设计

### 业务目标
输入非结构化简历文本，输出结构化数据（姓名、电话、技能列表、教育/工作经历……），供 HR 系统入库、检索、筛选。

### 为什么用 Pydantic 结构化输出
人写的简历格式千差万别，但下游系统要的是**固定字段**。让模型「抽取并按 Schema 输出」最稳的方式是：用 Pydantic 定义目标结构 → `PydanticOutputParser` 生成格式指令并解析校验。好处是：① 字段类型有保证（`graduation_year` 一定是 int）；② 支持**嵌套结构**（一个人有多段教育/工作经历）；③ 拿到的是带 IDE 提示的对象，不是裸 dict。

---

## 二、实现详解

### 难点 1：嵌套 Schema 表达「一对多」
简历里教育、工作都是「列表套对象」。用嵌套 BaseModel 表达：

```python
class Education(BaseModel):
    school: str; degree: str; major: str; graduation_year: int

class Resume(BaseModel):
    name: str
    skills: List[str]                       # 字符串列表
    education: List[Education]              # 对象列表（嵌套）
    work_experience: List[WorkExperience]  # 对象列表（嵌套）
```

`PydanticOutputParser` 会把这整棵嵌套结构翻译成 JSON Schema 塞进 prompt，模型据此输出层级化 JSON。

### 难点 2：每个字段写清 description
`Field(description="工作时间段，如 2020.03 - 2023.08")` 不只是注释——它会进入给模型的格式说明，**直接影响抽取质量**。写清楚「至少 2 项」「毕业年份」这类约束，模型才抽得准。

### 难点 3：把格式指令注入 prompt
```python
prompt = ChatPromptTemplate.from_messages([...]).partial(
    format_instructions=parser.get_format_instructions()
)
chain = prompt | llm | parser
```
`.partial` 预填格式说明，链路 `prompt | llm | parser` 一气呵成：解析失败会抛 `OutputParserException`，提示该加容错或降温度。

---

## 三、运行

```bash
python stage01_basics/01_hello_langchain.py   # 先确认环境通
python stage01_basics/project01_resume_parser.py
```

输出会把张伟的简历解析成结构化对象并分块打印。仓库里还附了几个测试：`test_parser_raw.py`（原始解析）、`test_parser_with_noise.py`（带噪声输入）、`test_pydantic_parser_error.py`（解析失败场景）、`test_parser_with_fix.py`（带纠错），建议都跑一遍体会鲁棒性。

---

## 四、复盘与进阶
1. **解析失败兜底**：用 `OutputFixingParser` 包一层，模型输出不合法时自动让 LLM 修复。
2. **换 `with_structured_output`**：新模型可直接 `llm.with_structured_output(Resume)`，更简洁。
3. **批量处理**：用 `chain.batch([...])` 并发解析多份简历。

## 五、面试怎么考
- **「怎么让模型稳定输出结构化数据？」** → Pydantic 定义 Schema + 格式指令注入 + 低温度 + 解析器校验/纠错；新模型用 `with_structured_output`。
- **「嵌套结构怎么处理？」** → 嵌套 BaseModel，parser 自动生成层级 JSON Schema。
- **「抽取不准怎么优化？」** → 把约束写进字段 description、给 few-shot 示例、必要时拆成多步抽取。
