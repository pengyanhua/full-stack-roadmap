# Agent基础概念

## 目录
1. [Agent简介](#agent简介)
2. [核心架构](#核心架构)
3. [Agent类型对比](#agent类型对比)
4. [LangChain Agent实现](#langchain-agent实现)
5. [自定义工具开发](#自定义工具开发)
6. [Agent记忆系统](#agent记忆系统)
7. [完整Agent实战](#完整agent实战)

---

## Agent简介

### 什么是AI Agent？

AI Agent（智能代理）是一个能够感知环境、做出决策并执行行动的自主系统。与传统的"输入-输出"模式不同，Agent具备**自主规划**、**工具使用**和**迭代执行**的能力。

```
┌─────────────────────────────────────────────────────────────┐
│                    传统LLM vs Agent对比                       │
├──────────────────────────┬──────────────────────────────────┤
│       传统LLM调用        │          Agent系统               │
│                          │                                  │
│   用户输入               │   用户输入                       │
│     │                    │     │                            │
│     ▼                    │     ▼                            │
│   ┌──────┐               │   ┌──────────┐                   │
│   │ LLM  │               │   │ 规划模块  │◄─── 记忆系统     │
│   └──┬───┘               │   └────┬─────┘                   │
│      │                   │        │                          │
│      ▼                   │        ▼                          │
│   直接输出               │   ┌──────────┐                   │
│                          │   │ 工具调用  │──► 搜索/计算/DB  │
│   (一次性完成)           │   └────┬─────┘                   │
│                          │        │                          │
│                          │        ▼                          │
│                          │   ┌──────────┐                   │
│                          │   │ 观察结果  │                   │
│                          │   └────┬─────┘                   │
│                          │        │                          │
│                          │        ▼                          │
│                          │   ┌──────────┐                   │
│                          │   │ 继续/终止 │──► 循环执行      │
│                          │   └────┬─────┘                   │
│                          │        │                          │
│                          │        ▼                          │
│                          │   最终输出                       │
└──────────────────────────┴──────────────────────────────────┘
```

### Agent的核心特征

| 特征 | 说明 | 示例 |
|------|------|------|
| **自主性** | 无需人工干预即可完成任务 | 自动搜索→分析→总结 |
| **反应性** | 根据环境变化调整行为 | 搜索失败时切换关键词 |
| **主动性** | 主动采取行动实现目标 | 主动拆分复杂任务 |
| **社交性** | 与其他Agent或人类协作 | 多Agent辩论/协作 |

### Agent vs Chatbot vs Pipeline

```python
# 1. 传统Chatbot - 单次问答
def chatbot(question: str) -> str:
    response = llm.invoke(question)
    return response

# 2. Pipeline - 固定流程
def pipeline(question: str) -> str:
    context = retriever.search(question)      # 固定步骤1
    response = llm.invoke(question, context)   # 固定步骤2
    return response

# 3. Agent - 动态决策
def agent(question: str) -> str:
    while not is_done:
        thought = llm.think(question, history)  # 思考
        action = llm.decide(thought)            # 决策
        result = execute_tool(action)           # 执行
        history.append(result)                  # 记录
        is_done = llm.should_stop(history)      # 判断
    return llm.summarize(history)               # 总结
```

---

## 核心架构

### Agent核心架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                        Agent 核心架构                             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │                     用户输入                             │     │
│  └──────────────────────────┬──────────────────────────────┘     │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                   ① 规划模块 (Planner)                    │    │
│  │  ┌─────────┐  ┌──────────┐  ┌───────────┐               │    │
│  │  │任务分解  │  │策略选择   │  │优先级排序  │               │    │
│  │  └─────────┘  └──────────┘  └───────────┘               │    │
│  └──────────────────────────┬───────────────────────────────┘    │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                   ② 执行模块 (Executor)                   │    │
│  │                                                          │    │
│  │   ┌─────────┐    ┌─────────┐    ┌─────────┐             │    │
│  │   │ 搜索工具 │    │ 计算工具 │    │ 代码工具 │    ...      │    │
│  │   └─────────┘    └─────────┘    └─────────┘             │    │
│  └──────────────────────────┬───────────────────────────────┘    │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                   ③ 记忆模块 (Memory)                     │    │
│  │  ┌──────────┐  ┌──────────┐  ┌───────────┐              │    │
│  │  │ 短期记忆  │  │ 长期记忆  │  │ 向量记忆   │              │    │
│  │  │(对话上下文)│  │(经验知识) │  │(语义检索)  │              │    │
│  │  └──────────┘  └──────────┘  └───────────┘              │    │
│  └──────────────────────────┬───────────────────────────────┘    │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                   ④ 反思模块 (Reflection)                 │    │
│  │  ┌──────────┐  ┌──────────┐  ┌───────────┐              │    │
│  │  │结果评估   │  │错误修正   │  │策略调整    │              │    │
│  │  └──────────┘  └──────────┘  └───────────┘              │    │
│  └──────────────────────────┬───────────────────────────────┘    │
│                             │                                    │
│                    ┌────────┴────────┐                           │
│                    │   任务完成？     │                           │
│                    └────────┬────────┘                           │
│                      是 /       \ 否                             │
│                       /           \                              │
│                      ▼             ▼                             │
│              ┌──────────┐  ┌──────────────┐                     │
│              │ 输出结果  │  │ 返回规划模块  │                     │
│              └──────────┘  └──────────────┘                     │
└──────────────────────────────────────────────────────────────────┘
```

### 核心组件详解

#### 1. 规划模块（Planner）

规划模块负责将用户的高层目标分解为可执行的子任务。

```python
import json
from openai import OpenAI

client = OpenAI()

def create_plan(user_goal: str) -> list[dict]:
    """将用户目标分解为子任务"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """你是一个任务规划专家。将用户目标分解为具体的子任务列表。
                返回JSON格式:
                [
                    {"step": 1, "task": "任务描述", "tool": "需要的工具", "depends_on": []},
                    ...
                ]"""
            },
            {"role": "user", "content": user_goal}
        ],
        response_format={"type": "json_object"}
    )
    plan = json.loads(response.choices[0].message.content)
    return plan.get("tasks", [])

# 示例
plan = create_plan("帮我调研Python Web框架的市场占有率并生成报告")
for step in plan:
    print(f"步骤{step['step']}: {step['task']} (工具: {step['tool']})")

# 输出示例:
# 步骤1: 搜索Python Web框架列表 (工具: web_search)
# 步骤2: 获取各框架的GitHub星标和下载量 (工具: api_call)
# 步骤3: 分析数据并计算占比 (工具: calculator)
# 步骤4: 生成对比图表 (工具: code_executor)
# 步骤5: 撰写调研报告 (工具: text_generator)
```

#### 2. 执行模块（Executor）

执行模块负责调用具体的工具完成子任务。

```python
from typing import Any, Callable

class ToolExecutor:
    """工具执行器"""

    def __init__(self):
        self.tools: dict[str, Callable] = {}

    def register_tool(self, name: str, func: Callable, description: str):
        """注册工具"""
        self.tools[name] = {
            "function": func,
            "description": description
        }

    def execute(self, tool_name: str, **kwargs) -> Any:
        """执行工具"""
        if tool_name not in self.tools:
            raise ValueError(f"未知工具: {tool_name}")
        try:
            result = self.tools[tool_name]["function"](**kwargs)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def list_tools(self) -> list[dict]:
        """列出所有可用工具"""
        return [
            {"name": name, "description": info["description"]}
            for name, info in self.tools.items()
        ]

# 使用示例
executor = ToolExecutor()
executor.register_tool(
    "calculator",
    lambda expression: eval(expression),
    "计算数学表达式"
)
executor.register_tool(
    "search",
    lambda query: f"搜索结果: {query}",
    "搜索互联网信息"
)

result = executor.execute("calculator", expression="2 + 3 * 4")
print(result)  # {'status': 'success', 'result': 14}
```

#### 3. 记忆模块（Memory）

```python
from datetime import datetime
from collections import deque

class AgentMemory:
    """Agent记忆系统"""

    def __init__(self, short_term_limit: int = 20):
        # 短期记忆：最近的对话和操作
        self.short_term = deque(maxlen=short_term_limit)
        # 长期记忆：持久化的经验知识
        self.long_term: list[dict] = []
        # 工作记忆：当前任务的上下文
        self.working_memory: dict = {}

    def add_short_term(self, content: str, role: str = "system"):
        """添加短期记忆"""
        self.short_term.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def add_long_term(self, content: str, category: str = "general"):
        """添加长期记忆"""
        self.long_term.append({
            "content": content,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "access_count": 0
        })

    def search_memory(self, query: str, top_k: int = 5) -> list[dict]:
        """搜索相关记忆（简单关键词匹配）"""
        results = []
        keywords = query.lower().split()
        for memory in self.long_term:
            score = sum(1 for kw in keywords if kw in memory["content"].lower())
            if score > 0:
                results.append({**memory, "relevance": score})
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:top_k]

    def get_context(self) -> list[dict]:
        """获取当前上下文（短期记忆）"""
        return list(self.short_term)

    def set_working(self, key: str, value):
        """设置工作记忆"""
        self.working_memory[key] = value

    def get_working(self, key: str, default=None):
        """获取工作记忆"""
        return self.working_memory.get(key, default)
```

#### 4. 反思模块（Reflection）

```python
def reflect_on_result(
    task: str,
    result: str,
    expected: str = None
) -> dict:
    """反思执行结果"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """评估任务执行结果的质量。返回JSON:
                {
                    "quality_score": 0-10,
                    "is_complete": true/false,
                    "issues": ["问题1", ...],
                    "suggestions": ["建议1", ...],
                    "next_action": "继续/重试/调整策略/完成"
                }"""
            },
            {
                "role": "user",
                "content": f"任务: {task}\n结果: {result}\n期望: {expected or '无特定期望'}"
            }
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
```

---

## Agent类型对比

### 主流Agent架构对比

```
┌──────────────────────────────────────────────────────────────────┐
│                     Agent类型对比图                               │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────┐ │
│  │   ReAct      │  │Plan-Execute │  │  Self-Ask   │  │  MRKL  │ │
│  │             │  │             │  │             │  │        │ │
│  │ Think→Act   │  │ Plan first  │  │ Ask sub-Qs  │  │ Router │ │
│  │ →Observe    │  │ then Execute│  │ then Answer │  │→Expert │ │
│  │ →Think...   │  │ then Revise │  │             │  │        │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────┘ │
│  交替思考与行动    先规划后执行     递归分解问题    模块化路由  │
└──────────────────────────────────────────────────────────────────┘
```

### 详细对比表

| 特性 | ReAct | Plan-Execute | Self-Ask | MRKL |
|------|-------|-------------|----------|------|
| **核心思想** | 思考-行动交替 | 先规划再执行 | 递归自问自答 | 模块化专家路由 |
| **规划方式** | 逐步规划 | 预先完整规划 | 递归分解 | 无显式规划 |
| **适用场景** | 通用任务 | 复杂多步任务 | 知识密集型 | 多领域切换 |
| **灵活性** | 高 | 中 | 中 | 高 |
| **效率** | 中 | 高（并行执行） | 低（递归开销） | 高 |
| **可解释性** | 高（思考可见） | 高（计划可见） | 高（问题链可见） | 低 |
| **实现复杂度** | 低 | 中 | 低 | 高 |
| **代表框架** | LangChain | LangGraph | Self-Ask | Gorilla |
| **LLM调用次数** | 较多 | 较少 | 较多 | 较少 |

### 各类型Agent示意代码

```python
# ============================================================
# 1. ReAct Agent - 思考→行动→观察循环
# ============================================================
class ReActAgent:
    """ReAct模式：交替进行推理(Reasoning)和行动(Acting)"""

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def run(self, question: str, max_steps: int = 5) -> str:
        history = []
        for step in range(max_steps):
            # Thought: 思考下一步
            thought = self.llm.think(question, history)
            history.append(f"Thought: {thought}")

            # Action: 选择并执行工具
            action, action_input = self.llm.decide_action(thought)
            history.append(f"Action: {action}({action_input})")

            # Observation: 获取结果
            observation = self.tools[action](action_input)
            history.append(f"Observation: {observation}")

            # 判断是否完成
            if self.llm.is_final_answer(history):
                return self.llm.generate_answer(history)

        return self.llm.generate_answer(history)


# ============================================================
# 2. Plan-Execute Agent - 先规划再执行
# ============================================================
class PlanExecuteAgent:
    """先制定完整计划，再逐步执行"""

    def __init__(self, planner_llm, executor_llm, tools):
        self.planner = planner_llm
        self.executor = executor_llm
        self.tools = tools

    def run(self, question: str) -> str:
        # Phase 1: 制定计划
        plan = self.planner.create_plan(question)
        results = []

        # Phase 2: 逐步执行
        for step in plan:
            result = self.executor.execute_step(step, self.tools)
            results.append(result)

            # Phase 3: 检查是否需要修改计划
            if self.planner.should_replan(plan, results):
                plan = self.planner.revise_plan(question, plan, results)

        return self.executor.summarize(results)


# ============================================================
# 3. Self-Ask Agent - 递归自问自答
# ============================================================
class SelfAskAgent:
    """通过自问子问题来回答复杂问题"""

    def __init__(self, llm, search_tool):
        self.llm = llm
        self.search = search_tool

    def run(self, question: str) -> str:
        # 判断是否需要分解
        needs_followup = self.llm.needs_followup(question)

        if not needs_followup:
            return self.llm.direct_answer(question)

        # 生成子问题
        sub_question = self.llm.generate_followup(question)
        # 递归回答子问题
        sub_answer = self.run(sub_question)  # 递归
        # 结合子答案回答原问题
        return self.llm.answer_with_context(question, sub_question, sub_answer)


# ============================================================
# 4. MRKL Agent - 模块化路由
# ============================================================
class MRKLAgent:
    """Modular Reasoning, Knowledge and Language"""

    def __init__(self, llm, expert_modules: dict):
        self.llm = llm
        self.experts = expert_modules  # {"math": math_expert, "search": search_expert, ...}

    def run(self, question: str) -> str:
        # 路由到合适的专家模块
        expert_name = self.llm.route(question, list(self.experts.keys()))
        expert = self.experts[expert_name]
        # 专家模块处理
        result = expert.process(question)
        # 整合结果
        return self.llm.format_answer(question, result)
```

---

## LangChain Agent实现

### 环境准备

```bash
pip install langchain langchain-openai langchain-community python-dotenv
```

### OpenAI Functions Agent 完整实现

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

# ============================================================
# 1. 初始化LLM
# ============================================================
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ============================================================
# 2. 定义工具
# ============================================================
@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入应为有效的Python数学表达式，如 '2 + 3 * 4'。"""
    try:
        # 安全计算（仅允许数学运算）
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def get_current_time() -> str:
    """获取当前日期和时间"""
    from datetime import datetime
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')}"

@tool
def text_analyzer(text: str) -> str:
    """分析文本的基本统计信息，如字数、句子数等"""
    words = len(text.split())
    chars = len(text)
    sentences = text.count('。') + text.count('.') + text.count('!') + text.count('?')
    return f"字符数: {chars}, 词数: {words}, 句子数: {sentences}"

# 搜索工具
search = DuckDuckGoSearchRun()

# 工具列表
tools = [calculator, get_current_time, text_analyzer, search]

# ============================================================
# 3. 创建Prompt
# ============================================================
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """你是一个有用的AI助手，可以使用多种工具来帮助用户。

请遵循以下原则：
1. 仔细分析用户的问题，确定需要使用哪些工具
2. 如果需要多步操作，逐步执行
3. 在给出最终回答前，确保信息准确
4. 用中文回答用户的问题"""
    ),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ============================================================
# 4. 创建Agent
# ============================================================
agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,           # 显示详细执行过程
    max_iterations=10,      # 最大迭代次数
    handle_parsing_errors=True,  # 处理解析错误
    return_intermediate_steps=True  # 返回中间步骤
)

# ============================================================
# 5. 运行Agent
# ============================================================
if __name__ == "__main__":
    # 示例1：数学计算
    result = agent_executor.invoke({"input": "计算 (25 * 4 + 30) / 5 的结果"})
    print(f"回答: {result['output']}")
    print(f"中间步骤: {len(result['intermediate_steps'])} 步")

    # 示例2：多工具协作
    result = agent_executor.invoke({
        "input": "现在几点了？另外帮我计算一下今年还剩多少天"
    })
    print(f"回答: {result['output']}")

    # 示例3：搜索+分析
    result = agent_executor.invoke({
        "input": "搜索Python 3.12的新特性，并统计搜索结果的文本长度"
    })
    print(f"回答: {result['output']}")
```

### 带对话历史的Agent

```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 创建消息历史存储
message_histories: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """获取或创建会话历史"""
    if session_id not in message_histories:
        message_histories[session_id] = ChatMessageHistory()
    return message_histories[session_id]

# 包装Agent使其支持消息历史
agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# 使用带历史的Agent
config = {"configurable": {"session_id": "user_001"}}

response1 = agent_with_history.invoke(
    {"input": "我叫张三，请记住我的名字"},
    config=config
)
print(response1["output"])

response2 = agent_with_history.invoke(
    {"input": "我叫什么名字？"},
    config=config
)
print(response2["output"])  # Agent会记住"张三"
```

---

## 自定义工具开发

### 使用@tool装饰器

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional

# ============================================================
# 方式1：简单工具（自动推断参数）
# ============================================================
@tool
def word_count(text: str) -> int:
    """统计文本中的单词数量。输入为要统计的文本字符串。"""
    return len(text.split())


# ============================================================
# 方式2：使用Pydantic定义输入Schema
# ============================================================
class WeatherInput(BaseModel):
    """天气查询的输入参数"""
    city: str = Field(description="城市名称，如'北京'、'上海'")
    unit: str = Field(default="celsius", description="温度单位：celsius或fahrenheit")

@tool(args_schema=WeatherInput)
def get_weather(city: str, unit: str = "celsius") -> str:
    """查询指定城市的天气信息"""
    # 模拟天气API
    weather_data = {
        "北京": {"temp": 22, "condition": "晴", "humidity": 45},
        "上海": {"temp": 26, "condition": "多云", "humidity": 65},
        "广州": {"temp": 30, "condition": "雷阵雨", "humidity": 80},
    }
    data = weather_data.get(city)
    if not data:
        return f"未找到{city}的天气数据"

    temp = data["temp"] if unit == "celsius" else data["temp"] * 9 / 5 + 32
    unit_str = "°C" if unit == "celsius" else "°F"
    return f"{city}天气: {data['condition']}, 温度{temp}{unit_str}, 湿度{data['humidity']}%"


# ============================================================
# 方式3：使用StructuredTool
# ============================================================
from langchain_core.tools import StructuredTool

def _search_database(
    table: str,
    query: str,
    limit: int = 10
) -> str:
    """搜索数据库"""
    # 模拟数据库查询
    return f"从{table}表中查询'{query}'，返回{limit}条记录"

database_tool = StructuredTool.from_function(
    func=_search_database,
    name="database_search",
    description="在数据库中搜索信息。需要指定表名和查询条件。",
)


# ============================================================
# 方式4：异步工具
# ============================================================
@tool
async def async_web_request(url: str) -> str:
    """发送异步HTTP请求获取网页内容"""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            text = await response.text()
            return text[:500]  # 返回前500个字符


# ============================================================
# 方式5：自定义BaseTool子类（最大灵活性）
# ============================================================
from langchain_core.tools import BaseTool
from typing import Type

class FileOperationInput(BaseModel):
    """文件操作的输入参数"""
    action: str = Field(description="操作类型：read, write, list")
    path: str = Field(description="文件或目录路径")
    content: Optional[str] = Field(default=None, description="写入的内容（仅write操作需要）")

class FileOperationTool(BaseTool):
    """文件操作工具"""
    name: str = "file_operation"
    description: str = "执行文件操作：读取、写入或列出目录内容"
    args_schema: Type[BaseModel] = FileOperationInput

    def _run(self, action: str, path: str, content: str = None) -> str:
        """同步执行"""
        if action == "read":
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            except FileNotFoundError:
                return f"文件不存在: {path}"
        elif action == "write":
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content or "")
            return f"已写入文件: {path}"
        elif action == "list":
            import os
            items = os.listdir(path)
            return f"目录内容: {', '.join(items)}"
        else:
            return f"未知操作: {action}"

    async def _arun(self, action: str, path: str, content: str = None) -> str:
        """异步执行（调用同步方法）"""
        return self._run(action, path, content)

# 使用
file_tool = FileOperationTool()
print(file_tool.invoke({"action": "list", "path": "."}))
```

### 工具测试

```python
def test_tools():
    """测试所有自定义工具"""
    print("=== 工具测试 ===\n")

    # 测试词数统计
    result = word_count.invoke("Hello world this is a test")
    print(f"word_count: {result}")
    assert result == 6

    # 测试天气查询
    result = get_weather.invoke({"city": "北京", "unit": "celsius"})
    print(f"get_weather: {result}")
    assert "北京" in result

    # 测试数据库搜索
    result = database_tool.invoke({"table": "users", "query": "active=true", "limit": 5})
    print(f"database_search: {result}")

    # 查看工具的Schema
    print(f"\nweather工具Schema: {get_weather.args_schema.model_json_schema()}")

    print("\n所有工具测试通过!")

if __name__ == "__main__":
    test_tools()
```

---

## Agent记忆系统

### 记忆类型架构

```
┌──────────────────────────────────────────────────────────────────┐
│                      Agent 记忆系统                              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              短期记忆 (Short-term Memory)                 │    │
│  │  ┌──────────────────────────────────────────────────┐    │    │
│  │  │  对话历史   │  当前步骤  │  中间结果  │  工作区   │    │    │
│  │  └──────────────────────────────────────────────────┘    │    │
│  │  特点: 容量有限, 快速访问, 随会话消失                    │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              长期记忆 (Long-term Memory)                  │    │
│  │  ┌──────────────────────────────────────────────────┐    │    │
│  │  │  用户偏好   │  历史经验  │  学习知识  │  技能库   │    │    │
│  │  └──────────────────────────────────────────────────┘    │    │
│  │  特点: 持久存储, 跨会话, 需要检索机制                    │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              向量记忆 (Vector Memory)                     │    │
│  │  ┌──────────────────────────────────────────────────┐    │    │
│  │  │  文档嵌入   │  对话嵌入  │  知识嵌入  │ 语义搜索  │    │    │
│  │  └──────────────────────────────────────────────────┘    │    │
│  │  特点: 语义相似度检索, 适合大规模知识库                  │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### 完整记忆系统实现

```python
import json
import os
from datetime import datetime
from typing import Optional
from collections import deque

# ============================================================
# 向量记忆实现（使用ChromaDB）
# ============================================================
# pip install chromadb langchain-openai

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

class VectorMemory:
    """基于向量数据库的语义记忆"""

    def __init__(self, collection_name: str = "agent_memory", persist_dir: str = "./memory_db"):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir
        )

    def store(self, text: str, metadata: dict = None):
        """存储记忆"""
        meta = metadata or {}
        meta["timestamp"] = datetime.now().isoformat()
        self.vectorstore.add_texts(
            texts=[text],
            metadatas=[meta]
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """语义搜索相关记忆"""
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]

    def clear(self):
        """清空记忆"""
        self.vectorstore.delete_collection()


# ============================================================
# 综合记忆管理器
# ============================================================
class ComprehensiveMemory:
    """综合记忆管理器：整合短期、长期和向量记忆"""

    def __init__(self, short_term_limit: int = 20):
        self.short_term = deque(maxlen=short_term_limit)
        self.long_term_file = "./long_term_memory.json"
        self.long_term = self._load_long_term()
        self.vector_memory = VectorMemory()

    def _load_long_term(self) -> list:
        """从文件加载长期记忆"""
        if os.path.exists(self.long_term_file):
            with open(self.long_term_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_long_term(self):
        """保存长期记忆到文件"""
        with open(self.long_term_file, 'w', encoding='utf-8') as f:
            json.dump(self.long_term, f, ensure_ascii=False, indent=2)

    def add_interaction(self, role: str, content: str):
        """添加交互记录（短期记忆）"""
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.short_term.append(entry)

        # 同时存入向量记忆
        self.vector_memory.store(content, {"role": role})

    def promote_to_long_term(self, content: str, importance: str = "normal"):
        """将信息提升为长期记忆"""
        entry = {
            "content": content,
            "importance": importance,
            "timestamp": datetime.now().isoformat(),
            "access_count": 0
        }
        self.long_term.append(entry)
        self._save_long_term()

    def recall(self, query: str, top_k: int = 5) -> dict:
        """综合回忆：从所有记忆源检索相关信息"""
        # 从短期记忆中搜索
        short_results = [
            m for m in self.short_term
            if query.lower() in m["content"].lower()
        ]

        # 从向量记忆中语义搜索
        vector_results = self.vector_memory.search(query, top_k=top_k)

        return {
            "short_term": list(short_results)[-5:],
            "vector_search": vector_results,
            "long_term_count": len(self.long_term)
        }

    def get_context_messages(self) -> list[dict]:
        """获取用于LLM的上下文消息"""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.short_term
        ]

# 使用示例
if __name__ == "__main__":
    memory = ComprehensiveMemory()

    # 模拟交互
    memory.add_interaction("user", "我喜欢Python编程")
    memory.add_interaction("assistant", "Python是很棒的编程语言")
    memory.add_interaction("user", "帮我写一个排序算法")

    # 提升为长期记忆
    memory.promote_to_long_term("用户偏好Python编程", "high")

    # 回忆
    results = memory.recall("Python")
    print(json.dumps(results, ensure_ascii=False, indent=2))
```

---

## 完整Agent实战

### 多功能助手Agent（搜索+计算器+数据库查询）

```python
"""
完整Agent实现：集成搜索、计算器和数据库查询功能
"""
import os
import json
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field

load_dotenv()

# ============================================================
# 1. 数据库准备
# ============================================================
def setup_database():
    """创建并填充示例数据库"""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # 创建产品表
    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            stock INTEGER,
            rating REAL
        )
    """)

    # 插入示例数据
    products = [
        (1, "MacBook Pro 14", "电脑", 14999, 50, 4.8),
        (2, "iPhone 15 Pro", "手机", 8999, 200, 4.7),
        (3, "AirPods Pro 2", "配件", 1899, 500, 4.6),
        (4, "iPad Air", "平板", 4799, 150, 4.5),
        (5, "Apple Watch S9", "手表", 2999, 300, 4.4),
        (6, "ThinkPad X1 Carbon", "电脑", 9999, 80, 4.5),
        (7, "Galaxy S24 Ultra", "手机", 9499, 120, 4.6),
        (8, "Sony WH-1000XM5", "配件", 2499, 200, 4.7),
        (9, "Surface Pro 9", "平板", 8988, 60, 4.3),
        (10, "华为 MatePad Pro", "平板", 4699, 100, 4.4),
    ]
    cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", products)

    # 创建销售表
    cursor.execute("""
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            product_id INTEGER,
            quantity INTEGER,
            sale_date TEXT,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)

    sales = [
        (1, 1, 5, "2024-01-15"), (2, 2, 20, "2024-01-15"),
        (3, 3, 50, "2024-01-16"), (4, 1, 3, "2024-01-17"),
        (5, 2, 15, "2024-01-18"), (6, 4, 10, "2024-01-18"),
        (7, 5, 25, "2024-01-19"), (8, 6, 8, "2024-01-20"),
        (9, 7, 12, "2024-01-20"), (10, 8, 30, "2024-01-21"),
    ]
    cursor.executemany("INSERT INTO sales VALUES (?,?,?,?)", sales)

    conn.commit()
    return conn

# 全局数据库连接
db_conn = setup_database()


# ============================================================
# 2. 定义工具
# ============================================================

# 工具1：搜索
search_tool = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """在互联网上搜索信息。输入搜索关键词，返回搜索结果摘要。"""
    try:
        result = search_tool.run(query)
        return result[:1000]  # 限制返回长度
    except Exception as e:
        return f"搜索出错: {str(e)}"


# 工具2：计算器
class CalculatorInput(BaseModel):
    expression: str = Field(description="数学表达式，如 '(100 * 0.15) + 50'")

@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """计算数学表达式。支持加减乘除、幂运算、取余等。"""
    import math
    safe_dict = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "len": len,
        "sqrt": math.sqrt, "log": math.log, "pi": math.pi, "e": math.e,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "ceil": math.ceil, "floor": math.floor,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}。请检查表达式格式。"


# 工具3：数据库查询
class DBQueryInput(BaseModel):
    sql: str = Field(description="SQL查询语句（仅支持SELECT）")

@tool(args_schema=DBQueryInput)
def query_database(sql: str) -> str:
    """查询产品数据库。包含products表(id,name,category,price,stock,rating)和sales表(id,product_id,quantity,sale_date)。仅支持SELECT查询。"""
    # 安全检查
    sql_upper = sql.upper().strip()
    if not sql_upper.startswith("SELECT"):
        return "错误：仅支持SELECT查询"
    for forbidden in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]:
        if forbidden in sql_upper:
            return f"错误：不允许使用{forbidden}操作"

    try:
        cursor = db_conn.cursor()
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        if not rows:
            return "查询结果为空"

        # 格式化输出
        result_lines = [" | ".join(columns)]
        result_lines.append("-" * len(result_lines[0]))
        for row in rows[:20]:  # 最多20条
            result_lines.append(" | ".join(str(v) for v in row))

        return f"查询结果 ({len(rows)}条记录):\n" + "\n".join(result_lines)
    except Exception as e:
        return f"查询错误: {str(e)}"


# 工具4：日期时间
@tool
def datetime_tool() -> str:
    """获取当前日期时间和有用的时间信息"""
    now = datetime.now()
    return (
        f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"星期: {['一','二','三','四','五','六','日'][now.weekday()]}\n"
        f"今年第{now.timetuple().tm_yday}天\n"
        f"本月第{now.day}天"
    )


# ============================================================
# 3. 创建Agent
# ============================================================
tools = [web_search, calculator, query_database, datetime_tool]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """你是一个强大的AI助手，具备以下能力：

1. **互联网搜索**: 搜索最新信息和知识
2. **数学计算**: 执行复杂的数学运算
3. **数据库查询**: 查询产品和销售数据库
4. **时间查询**: 获取当前日期时间

数据库表结构：
- products: id, name, category, price, stock, rating
- sales: id, product_id, quantity, sale_date

工作原则：
- 复杂问题分步解决，逐步使用工具
- 数据分析时先查询数据，再进行计算
- 搜索时使用精确的关键词
- 始终用中文回答"""
    ),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=15,
    handle_parsing_errors=True,
)


# ============================================================
# 4. 交互式运行
# ============================================================
def run_interactive():
    """交互式Agent对话"""
    print("=" * 60)
    print("    多功能AI助手 (搜索 + 计算 + 数据库)")
    print("    输入 'quit' 退出")
    print("=" * 60)

    chat_history = []

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break
        if not user_input:
            continue

        try:
            result = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            response = result["output"]
            print(f"\n助手: {response}")

            # 更新对话历史
            chat_history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response}
            ])
        except Exception as e:
            print(f"\n错误: {str(e)}")


# ============================================================
# 5. 批量测试
# ============================================================
def run_tests():
    """测试Agent的多种能力"""
    test_cases = [
        "数据库里最贵的3个产品是什么？它们的总价是多少？",
        "各品类产品的平均价格是多少？哪个品类最贵？",
        "销量最好的产品是哪个？计算它的总销售额。",
        "现在几点了？帮我计算距离2025年元旦还有多少天。",
    ]

    for i, question in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}: {question}")
        print('='*60)

        result = agent_executor.invoke({"input": question})
        print(f"\n最终回答: {result['output']}")


if __name__ == "__main__":
    # run_interactive()  # 交互模式
    run_tests()           # 测试模式
```

---

## Agent执行流程深度剖析

### 完整执行时序图

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Agent 完整执行时序图                                │
│                                                                      │
│  User          Planner        Executor       Memory       Reflector  │
│   │               │              │              │              │     │
│   │──"分析报告"──►│              │              │              │     │
│   │               │              │              │              │     │
│   │               │──查询历史───►│              │              │     │
│   │               │◄──返回上下文─│              │              │     │
│   │               │              │              │              │     │
│   │               │──生成计划──┐ │              │              │     │
│   │               │            │ │              │              │     │
│   │               │◄───────────┘ │              │              │     │
│   │               │              │              │              │     │
│   │               │──子任务1────►│              │              │     │
│   │               │              │──搜索工具───►│              │     │
│   │               │              │◄──搜索结果──│              │     │
│   │               │              │──存储结果───────────►│     │     │
│   │               │◄──结果1──────│              │        │     │     │
│   │               │              │              │        │     │     │
│   │               │──子任务2────►│              │        │     │     │
│   │               │              │──计算工具───►│        │     │     │
│   │               │              │◄──计算结果──│        │     │     │
│   │               │◄──结果2──────│              │        │     │     │
│   │               │              │              │        │     │     │
│   │               │──评估请求──────────────────────────────►│     │
│   │               │◄──评估结果──────────────────────────────│     │
│   │               │              │              │              │     │
│   │               │  [质量达标]   │              │              │     │
│   │               │              │              │              │     │
│   │◄──最终报告───│              │              │              │     │
│   │               │              │              │              │     │
└──────────────────────────────────────────────────────────────────────┘
```

### Agent决策树可视化

```python
"""
Agent决策树：展示Agent在每个步骤的决策逻辑
"""
from openai import OpenAI
import json

client = OpenAI()

class AgentDecisionTree:
    """Agent决策树实现"""

    def __init__(self):
        self.decision_log = []

    def decide_next_action(self, state: dict) -> dict:
        """
        决策函数：根据当前状态决定下一步行动

        决策树逻辑:
        1. 是否有足够信息？ → 否 → 搜索/查询
        2. 是否需要计算？ → 是 → 调用计算器
        3. 是否需要验证？ → 是 → 交叉验证
        4. 是否达到质量标准？ → 否 → 反思/重试
        5. 全部通过 → 生成最终答案
        """
        question = state.get("question", "")
        gathered_info = state.get("gathered_info", [])
        iteration = state.get("iteration", 0)

        decision_prompt = f"""你是一个Agent的决策模块。根据当前状态，决定下一步行动。

当前状态:
- 用户问题: {question}
- 已收集信息: {json.dumps(gathered_info, ensure_ascii=False)}
- 当前迭代: {iteration}

请以JSON格式返回决策:
{{
    "action": "search|calculate|verify|reflect|answer",
    "reason": "决策原因",
    "parameters": {{}},
    "confidence": 0.0-1.0
}}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": decision_prompt}],
            response_format={"type": "json_object"}
        )

        decision = json.loads(response.choices[0].message.content)
        self.decision_log.append({
            "iteration": iteration,
            "state_summary": f"info_count={len(gathered_info)}",
            "decision": decision
        })
        return decision

    def print_decision_tree(self):
        """打印决策历史"""
        print("\n┌── Agent决策历史 ─────────────────────────┐")
        for entry in self.decision_log:
            i = entry["iteration"]
            d = entry["decision"]
            print(f"│ Step {i}: {d['action']:<12} "
                  f"(confidence: {d.get('confidence', 'N/A')})")
            print(f"│   原因: {d['reason'][:50]}")
            print(f"│   {'─'*44}")
        print("└──────────────────────────────────────────┘")
```

### Agent错误恢复机制

```python
"""
Agent错误恢复与容错机制
"""
import time
import logging
from typing import Callable, Any, Optional
from enum import Enum

logger = logging.getLogger("AgentErrorRecovery")


class RecoveryStrategy(Enum):
    """恢复策略"""
    RETRY = "retry"              # 重试相同操作
    FALLBACK = "fallback"        # 降级到备选方案
    SKIP = "skip"                # 跳过当前步骤
    REPLAN = "replan"            # 重新规划
    ESCALATE = "escalate"        # 升级处理（人工介入）


class AgentErrorHandler:
    """Agent错误处理器"""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_history = []
        self.fallback_tools = {}

    def register_fallback(self, tool_name: str, fallback_func: Callable):
        """注册工具的降级方案"""
        self.fallback_tools[tool_name] = fallback_func

    def classify_error(self, error: Exception) -> RecoveryStrategy:
        """根据错误类型决定恢复策略"""
        error_str = str(error).lower()

        if "rate_limit" in error_str or "429" in error_str:
            return RecoveryStrategy.RETRY  # 限流 → 重试
        elif "timeout" in error_str or "connection" in error_str:
            return RecoveryStrategy.RETRY  # 网络问题 → 重试
        elif "not found" in error_str or "404" in error_str:
            return RecoveryStrategy.FALLBACK  # 资源不存在 → 降级
        elif "permission" in error_str or "403" in error_str:
            return RecoveryStrategy.ESCALATE  # 权限问题 → 升级
        elif "parse" in error_str or "json" in error_str:
            return RecoveryStrategy.REPLAN  # 解析错误 → 重新规划
        else:
            return RecoveryStrategy.SKIP  # 未知错误 → 跳过

    def execute_with_recovery(
        self,
        func: Callable,
        tool_name: str,
        *args, **kwargs
    ) -> dict:
        """带错误恢复的工具执行"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                return {"status": "success", "result": result, "attempts": attempt + 1}
            except Exception as e:
                last_error = e
                strategy = self.classify_error(e)

                self.error_history.append({
                    "tool": tool_name,
                    "error": str(e),
                    "strategy": strategy.value,
                    "attempt": attempt + 1
                })

                logger.warning(
                    f"工具 {tool_name} 第{attempt+1}次失败: {e}, "
                    f"策略: {strategy.value}"
                )

                if strategy == RecoveryStrategy.RETRY:
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue

                elif strategy == RecoveryStrategy.FALLBACK:
                    if tool_name in self.fallback_tools:
                        try:
                            fallback_result = self.fallback_tools[tool_name](*args, **kwargs)
                            return {
                                "status": "fallback",
                                "result": fallback_result,
                                "original_error": str(e)
                            }
                        except Exception as fb_error:
                            logger.error(f"降级方案也失败: {fb_error}")
                    break

                elif strategy == RecoveryStrategy.SKIP:
                    return {
                        "status": "skipped",
                        "error": str(e),
                        "message": "该步骤已跳过，不影响最终结果"
                    }

                elif strategy == RecoveryStrategy.ESCALATE:
                    return {
                        "status": "escalated",
                        "error": str(e),
                        "message": "需要人工介入处理"
                    }

                elif strategy == RecoveryStrategy.REPLAN:
                    return {
                        "status": "replan",
                        "error": str(e),
                        "message": "需要重新规划执行策略"
                    }

        return {
            "status": "failed",
            "error": str(last_error),
            "attempts": self.max_retries
        }


# 使用示例
handler = AgentErrorHandler(max_retries=3, retry_delay=0.5)

# 注册降级方案
handler.register_fallback(
    "web_search",
    lambda query: f"[缓存数据] 关于'{query}'的历史搜索结果"
)

# 模拟工具执行
import random

def unreliable_search(query: str) -> str:
    """模拟不稳定的搜索工具"""
    if random.random() < 0.4:
        raise ConnectionError("搜索服务暂时不可用")
    return f"搜索到关于'{query}'的5条结果..."

result = handler.execute_with_recovery(
    unreliable_search, "web_search", "Python Agent框架"
)
print(f"执行结果: {result}")
```

### Agent安全防护模式

```python
"""
Agent安全防护：防止Agent执行危险操作
"""
import re
from typing import Callable

class AgentSafetyGuard:
    """Agent安全防护层"""

    def __init__(self):
        self.blocked_patterns = [
            r"rm\s+-rf",                # 危险的删除命令
            r"DROP\s+TABLE",            # SQL删表
            r"DELETE\s+FROM",           # SQL删数据
            r"os\.system\(",            # 系统命令执行
            r"subprocess\.",            # 子进程调用
            r"__import__\(",            # 动态导入
            r"eval\(",                  # 动态执行
            r"exec\(",                  # 动态执行
        ]
        self.sensitive_keywords = [
            "密码", "password", "token", "api_key", "secret",
            "信用卡", "身份证", "银行卡"
        ]
        self.action_log = []

    def check_tool_input(self, tool_name: str, input_str: str) -> dict:
        """检查工具输入是否安全"""
        issues = []

        # 检查危险模式
        for pattern in self.blocked_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                issues.append(f"检测到危险模式: {pattern}")

        # 检查敏感信息
        for keyword in self.sensitive_keywords:
            if keyword.lower() in input_str.lower():
                issues.append(f"包含敏感信息: {keyword}")

        # 检查输入长度（防止注入攻击）
        if len(input_str) > 10000:
            issues.append("输入长度异常（可能是注入攻击）")

        result = {
            "safe": len(issues) == 0,
            "tool": tool_name,
            "issues": issues
        }

        self.action_log.append(result)
        return result

    def check_tool_output(self, tool_name: str, output_str: str) -> dict:
        """检查工具输出是否包含敏感信息"""
        issues = []
        for keyword in self.sensitive_keywords:
            if keyword.lower() in output_str.lower():
                issues.append(f"输出包含敏感信息: {keyword}")
        return {"safe": len(issues) == 0, "issues": issues}

    def safe_execute(
        self,
        tool_func: Callable,
        tool_name: str,
        input_str: str
    ) -> dict:
        """安全执行工具"""
        # 输入检查
        input_check = self.check_tool_input(tool_name, input_str)
        if not input_check["safe"]:
            return {
                "status": "blocked",
                "reason": input_check["issues"],
                "message": "安全检查未通过，操作已阻止"
            }

        # 执行工具
        result = tool_func(input_str)
        result_str = str(result)

        # 输出检查
        output_check = self.check_tool_output(tool_name, result_str)
        if not output_check["safe"]:
            # 脱敏处理
            for keyword in self.sensitive_keywords:
                result_str = re.sub(
                    keyword, "[已脱敏]", result_str, flags=re.IGNORECASE
                )
            return {"status": "sanitized", "result": result_str}

        return {"status": "success", "result": result}


# 使用示例
guard = AgentSafetyGuard()

# 测试1: 正常输入
check = guard.check_tool_input("calculator", "2 + 3 * 4")
print(f"正常输入检查: {check}")  # safe: True

# 测试2: 危险输入
check = guard.check_tool_input("code_exec", "os.system('rm -rf /')")
print(f"危险输入检查: {check}")  # safe: False

# 测试3: 敏感信息
check = guard.check_tool_input("search", "查询password相关信息")
print(f"敏感信息检查: {check}")  # safe: False
```

---

## Agent性能优化

### Token优化策略

```
┌──────────────────────────────────────────────────────────────┐
│                Agent Token优化策略                             │
│                                                              │
│  1. 上下文压缩                                                │
│  ┌──────────────────────────────────────────────────┐        │
│  │ 原始: 5000 tokens → 压缩后: 1500 tokens          │        │
│  │                                                  │        │
│  │ 策略A: 只保留最近N轮对话                          │        │
│  │ 策略B: LLM摘要压缩历史消息                        │        │
│  │ 策略C: 向量检索相关历史（而非全部）                │        │
│  └──────────────────────────────────────────────────┘        │
│                                                              │
│  2. 工具返回值截断                                            │
│  ┌──────────────────────────────────────────────────┐        │
│  │ 搜索结果: 10000字 → 截断到: 2000字               │        │
│  │ 数据库结果: 100条 → 限制到: 20条                  │        │
│  │ 网页内容: 全文 → 提取关键段落                     │        │
│  └──────────────────────────────────────────────────┘        │
│                                                              │
│  3. 模型分层调用                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │ 路由/分类: gpt-4o-mini (低成本，快速)             │        │
│  │ 复杂推理: gpt-4o (高精度)                         │        │
│  │ 简单生成: gpt-4o-mini                             │        │
│  └──────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

```python
"""
Agent性能优化实践
"""
from openai import OpenAI
from typing import Optional

client = OpenAI()


class ContextCompressor:
    """上下文压缩器"""

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens

    def compress_history(self, messages: list, keep_last_n: int = 3) -> list:
        """压缩对话历史"""
        if len(messages) <= keep_last_n + 1:
            return messages

        # 保留系统消息
        system_msgs = [m for m in messages if m.get("role") == "system"]
        # 保留最近N轮
        recent_msgs = messages[-keep_last_n * 2:]

        # 摘要压缩中间消息
        middle_msgs = messages[len(system_msgs):-keep_last_n * 2]
        if middle_msgs:
            summary = self._summarize(middle_msgs)
            summary_msg = {
                "role": "system",
                "content": f"[历史对话摘要] {summary}"
            }
            return system_msgs + [summary_msg] + recent_msgs
        return system_msgs + recent_msgs

    def _summarize(self, messages: list) -> str:
        """使用小模型摘要消息"""
        content = "\n".join([
            f"{m['role']}: {m['content'][:200]}" for m in messages
        ])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"请用100字以内概括以下对话要点:\n{content}"
            }],
            max_tokens=200
        )
        return response.choices[0].message.content


class ModelRouter:
    """模型路由器：根据任务复杂度选择模型"""

    MODELS = {
        "simple": "gpt-4o-mini",     # 简单任务
        "medium": "gpt-4o",          # 中等任务
        "complex": "gpt-4o",         # 复杂任务
    }

    @classmethod
    def select_model(cls, task_type: str, token_count: int = 0) -> str:
        """选择合适的模型"""
        # 路由/分类/提取 → 小模型
        simple_tasks = ["classify", "extract", "route", "summarize"]
        if task_type in simple_tasks:
            return cls.MODELS["simple"]

        # 推理/规划/代码生成 → 大模型
        complex_tasks = ["reason", "plan", "code", "analyze"]
        if task_type in complex_tasks:
            return cls.MODELS["complex"]

        return cls.MODELS["medium"]

    @classmethod
    def estimate_cost(cls, model: str, input_tokens: int,
                      output_tokens: int) -> float:
        """估算API调用成本（USD）"""
        pricing = {
            "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.6 / 1_000_000},
            "gpt-4o": {"input": 2.5 / 1_000_000, "output": 10 / 1_000_000},
        }
        if model in pricing:
            cost = (input_tokens * pricing[model]["input"] +
                    output_tokens * pricing[model]["output"])
            return round(cost, 6)
        return 0.0


# 使用示例
router = ModelRouter()
print(f"分类任务模型: {router.select_model('classify')}")
print(f"推理任务模型: {router.select_model('reason')}")
print(f"GPT-4o-mini 1000 tokens 成本: ${router.estimate_cost('gpt-4o-mini', 500, 500)}")
print(f"GPT-4o 1000 tokens 成本: ${router.estimate_cost('gpt-4o', 500, 500)}")
```

### Agent生产部署清单

```
┌──────────────────────────────────────────────────────────────┐
│              Agent 生产部署检查清单                             │
│                                                              │
│  □ 安全                                                      │
│  ├─ □ 工具输入验证（防注入）                                  │
│  ├─ □ 输出脱敏（过滤敏感信息）                                │
│  ├─ □ API Key加密存储（使用环境变量/Vault）                   │
│  ├─ □ 速率限制（防止滥用）                                   │
│  └─ □ 操作审计日志                                           │
│                                                              │
│  □ 可靠性                                                    │
│  ├─ □ 错误重试机制（指数退避）                                │
│  ├─ □ 降级方案（工具不可用时的备选）                          │
│  ├─ □ 超时控制（每个工具调用设超时）                          │
│  ├─ □ 最大迭代限制（防无限循环）                              │
│  └─ □ 断路器模式（连续失败后停止调用）                        │
│                                                              │
│  □ 可观测性                                                  │
│  ├─ □ 结构化日志（JSON格式）                                 │
│  ├─ □ 追踪链路（LangSmith/OpenTelemetry）                   │
│  ├─ □ 指标监控（延迟/成功率/Token消耗）                      │
│  ├─ □ 告警规则（异常检测）                                   │
│  └─ □ 成本监控（按用户/会话统计）                            │
│                                                              │
│  □ 性能                                                      │
│  ├─ □ 上下文压缩（减少Token消耗）                            │
│  ├─ □ 模型路由（分级调用不同模型）                            │
│  ├─ □ 缓存策略（重复查询结果缓存）                            │
│  ├─ □ 并行工具执行（无依赖的工具并行）                        │
│  └─ □ 流式输出（提升用户体验）                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 总结

本教程涵盖了Agent基础概念的核心内容：

1. **Agent简介**: Agent是具备自主规划、工具使用和迭代执行能力的AI系统，超越了简单的"输入-输出"模式
2. **核心架构**: 包含规划、执行、记忆和反思四大模块，形成完整的认知循环
3. **Agent类型**: ReAct（交替思考行动）、Plan-Execute（先规划后执行）、Self-Ask（递归分解）、MRKL（模块化路由）各有适用场景
4. **LangChain实现**: 使用OpenAI Functions Agent可以快速构建功能强大的Agent
5. **自定义工具**: 通过@tool装饰器、StructuredTool和BaseTool子类三种方式开发工具
6. **记忆系统**: 短期记忆、长期记忆和向量记忆的组合使Agent具备上下文理解能力
7. **完整实战**: 集成搜索、计算和数据库查询的多功能Agent展示了工程化实践

## 最佳实践

1. **工具描述要清晰**: 工具的description直接影响LLM选择工具的准确性，务必描述清楚功能和输入格式
2. **控制迭代次数**: 设置合理的max_iterations防止Agent陷入死循环
3. **错误处理**: 所有工具都应有完善的异常处理，返回有意义的错误信息
4. **记忆管理**: 合理设置短期记忆窗口大小，避免context过长导致性能下降
5. **安全防护**: 数据库工具应严格限制SQL操作类型，计算器应限制可用函数
6. **日志记录**: verbose=True在开发阶段非常有用，上线后可关闭

## 参考资源

- [LangChain Agent文档](https://python.langchain.com/docs/modules/agents/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [ReAct论文](https://arxiv.org/abs/2210.03629)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangGraph文档](https://langchain-ai.github.io/langgraph/)

---

**文件大小目标**: 25KB
**创建时间**: 2024-01-01
**最后更新**: 2024-01-01
