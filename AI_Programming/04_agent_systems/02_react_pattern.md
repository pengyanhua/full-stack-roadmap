# ReAct模式详解

## 目录
1. [ReAct原理](#react原理)
2. [Thought-Action-Observation循环](#thought-action-observation循环)
3. [从零实现ReAct Agent](#从零实现react-agent)
4. [LangChain ReAct实现](#langchain-react实现)
5. [实战：Web搜索Agent](#实战web搜索agent)
6. [ReAct vs CoT vs Act-only对比](#react-vs-cot-vs-act-only对比)
7. [调试技巧与常见问题](#调试技巧与常见问题)

---

## ReAct原理

### 论文核心思想

ReAct（Reasoning + Acting）由Yao等人在2022年论文《ReAct: Synergizing Reasoning and Acting in Language Models》中提出。其核心思想是让语言模型**交替进行推理（Reasoning）和行动（Acting）**，形成一个相互增强的循环。

```
┌──────────────────────────────────────────────────────────────────┐
│                    ReAct 核心思想图解                              │
│                                                                  │
│   传统方法的问题:                                                │
│                                                                  │
│   ┌──────────┐    只推理,不行动         ┌──────────┐             │
│   │  CoT     │ ─────────────────────►  │ 幻觉/过时 │             │
│   │ (Chain   │    无法获取外部信息       │ 信息错误  │             │
│   │  of      │                         └──────────┘             │
│   │ Thought) │                                                   │
│   └──────────┘                                                   │
│                                                                  │
│   ┌──────────┐    只行动,不推理         ┌──────────┐             │
│   │ Act-only │ ─────────────────────►  │ 盲目试错  │             │
│   │          │    缺乏规划和反思         │ 效率低下  │             │
│   └──────────┘                         └──────────┘             │
│                                                                  │
│   ReAct的解决方案:                                               │
│                                                                  │
│   ┌─────────────────────────────────────────────────────┐       │
│   │                                                     │       │
│   │    Thought ──► Action ──► Observation ──► Thought   │       │
│   │       │                                     │       │       │
│   │       │          推理指导行动                 │       │       │
│   │       │          观察修正推理                 │       │       │
│   │       │                                     │       │       │
│   │       └─────────── 循环 ────────────────────┘       │       │
│   │                                                     │       │
│   │    ✓ 推理产生行动计划                                │       │
│   │    ✓ 行动获取真实信息                                │       │
│   │    ✓ 观察结果修正推理                                │       │
│   │    ✓ 形成正反馈循环                                  │       │
│   └─────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────┘
```

### ReAct的三大优势

| 优势 | 说明 | 对比 |
|------|------|------|
| **可解释性** | 思考过程完全可见，便于调试和理解 | CoT只有推理链，Action-only没有推理 |
| **准确性** | 通过工具获取真实信息，减少幻觉 | CoT容易产生幻觉 |
| **灵活性** | 可以根据观察结果动态调整策略 | Pipeline是固定流程 |

### ReAct与人类认知的类比

```
人类解决问题的过程:                    ReAct Agent:

"这个数学题怎么做？"                   Question: "2023年诺贝尔物理学奖是谁获得的？"
     │                                      │
     ▼                                      ▼
"让我想想，应该用公式..."              Thought: "我需要搜索最新的诺贝尔奖信息"
     │                                      │
     ▼                                      ▼
翻书/查公式表                          Action: Search("2023年诺贝尔物理学奖")
     │                                      │
     ▼                                      ▼
"找到了，公式是..."                    Observation: "Pierre Agostini, Ferenc Krausz..."
     │                                      │
     ▼                                      ▼
"把数字代入..."                        Thought: "我找到了获奖者信息，可以回答了"
     │                                      │
     ▼                                      ▼
"答案是42"                             Final Answer: "2023年诺贝尔物理学奖由..."
```

---

## Thought-Action-Observation循环

### 循环详解

```
┌──────────────────────────────────────────────────────────────┐
│              Thought-Action-Observation 循环                   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Step 1: THOUGHT (思考)                               │    │
│  │                                                      │    │
│  │  LLM根据问题和历史信息进行推理:                       │    │
│  │  - 分析当前状态                                       │    │
│  │  - 确定下一步需要什么信息                              │    │
│  │  - 选择合适的工具和策略                                │    │
│  │  - 评估已有信息是否足够回答                            │    │
│  └────────────────────────┬─────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Step 2: ACTION (行动)                                │    │
│  │                                                      │    │
│  │  根据思考结果选择并执行工具:                           │    │
│  │  - Search[query]: 搜索信息                            │    │
│  │  - Lookup[term]: 查找特定信息                         │    │
│  │  - Calculate[expr]: 计算表达式                        │    │
│  │  - Finish[answer]: 给出最终答案                       │    │
│  └────────────────────────┬─────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Step 3: OBSERVATION (观察)                           │    │
│  │                                                      │    │
│  │  工具返回结果，Agent观察并理解:                        │    │
│  │  - 搜索结果文本                                       │    │
│  │  - 计算结果数值                                       │    │
│  │  - API返回数据                                        │    │
│  │  - 错误信息                                           │    │
│  └────────────────────────┬─────────────────────────────┘    │
│                           │                                  │
│                    ┌──────┴──────┐                           │
│                    │ 是否完成？   │                           │
│                    └──────┬──────┘                           │
│                     否 /     \ 是                            │
│                      /         \                             │
│                     ▼           ▼                            │
│              回到 Step 1    输出最终答案                      │
└──────────────────────────────────────────────────────────────┘
```

### 完整示例追踪

```
Question: "苹果公司2023年的市值是多少？和微软比哪个更高？"

───────────────────────────────────────────────────────
Step 1:
  Thought: 我需要查找苹果公司2023年的市值信息。让我先搜索苹果公司的市值。
  Action: Search["Apple market cap 2023"]
  Observation: Apple's market capitalization reached approximately $3 trillion
               in 2023, making it the most valuable public company...

Step 2:
  Thought: 苹果市值约3万亿美元。现在我需要查找微软的市值进行比较。
  Action: Search["Microsoft market cap 2023"]
  Observation: Microsoft's market capitalization surpassed $2.8 trillion
               in late 2023, briefly overtaking Apple...

Step 3:
  Thought: 我已经有了两个公司的数据：
           - Apple: ~$3 trillion
           - Microsoft: ~$2.8 trillion
           苹果略高于微软，但两者非常接近，在2023年底有交替领先的情况。
           我现在可以给出完整的回答了。
  Action: Finish["2023年苹果公司市值约为3万亿美元，微软约为2.8万亿美元。
          苹果总体上略高于微软，但在2023年底两者非常接近，
          曾出现交替领先的情况。"]
───────────────────────────────────────────────────────
```

### Prompt模板设计

```python
REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
```

---

## 从零实现ReAct Agent

### 纯Python实现（不依赖框架）

```python
"""
从零实现ReAct Agent - 纯Python版本
不依赖LangChain等框架，帮助理解ReAct的底层原理
"""
import re
import json
from openai import OpenAI
from typing import Callable

client = OpenAI()


# ============================================================
# 1. 定义工具
# ============================================================
def search(query: str) -> str:
    """模拟搜索工具"""
    # 在实际应用中，这里调用搜索API
    knowledge_base = {
        "python创始人": "Python由Guido van Rossum于1991年创建",
        "python版本": "Python最新稳定版本是3.12，发布于2023年10月",
        "langchain": "LangChain是一个用于构建LLM应用的开源框架",
        "react论文": "ReAct论文由Yao等人在2022年发表于ICLR 2023",
        "地球半径": "地球平均半径约为6371公里",
        "光速": "光在真空中的速度约为299,792,458米/秒",
    }
    for key, value in knowledge_base.items():
        if key in query.lower():
            return value
    return f"未找到与'{query}'相关的信息，请尝试其他关键词"


def calculate(expression: str) -> str:
    """安全的数学计算"""
    import math
    safe_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "pow": pow, "sqrt": math.sqrt, "pi": math.pi,
        "log": math.log, "sin": math.sin, "cos": math.cos,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe_names)
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"


def lookup(term: str) -> str:
    """查找术语定义"""
    definitions = {
        "agent": "在AI领域，Agent是能够自主感知环境、做出决策并执行行动的系统",
        "react": "ReAct是Reasoning+Acting的缩写，是一种让LLM交替推理和行动的方法",
        "llm": "Large Language Model，大型语言模型，如GPT-4、Claude等",
        "prompt": "Prompt是给LLM的指令或输入文本，用于引导模型生成期望的输出",
    }
    return definitions.get(term.lower(), f"未找到术语'{term}'的定义")


# 工具注册表
TOOLS = {
    "Search": {
        "func": search,
        "description": "搜索互联网获取信息。输入: 搜索关键词"
    },
    "Calculate": {
        "func": calculate,
        "description": "计算数学表达式。输入: 数学表达式如 '2+3*4'"
    },
    "Lookup": {
        "func": lookup,
        "description": "查找术语定义。输入: 术语名称"
    },
}


# ============================================================
# 2. ReAct Agent核心实现
# ============================================================
class ReActAgent:
    """从零实现的ReAct Agent"""

    def __init__(
        self,
        tools: dict,
        model: str = "gpt-4o",
        max_steps: int = 8,
        verbose: bool = True
    ):
        self.tools = tools
        self.model = model
        self.max_steps = max_steps
        self.verbose = verbose
        self.client = OpenAI()

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        tool_descriptions = "\n".join([
            f"  {name}: {info['description']}"
            for name, info in self.tools.items()
        ])
        tool_names = ", ".join(self.tools.keys())

        return f"""You are a helpful assistant that answers questions by reasoning step-by-step and using tools.

Available tools:
{tool_descriptions}

You MUST follow this EXACT format for EVERY step:

Thought: <your reasoning about what to do next>
Action: <tool name, one of [{tool_names}]>
Action Input: <input for the tool>

When you have enough information to answer, use this format:

Thought: I now know the final answer
Final Answer: <your complete answer>

Important rules:
1. Always start with a Thought
2. Use tools to get factual information instead of guessing
3. You can use multiple tools in sequence
4. Always give Final Answer in Chinese (中文)"""

    def _parse_llm_output(self, text: str) -> dict:
        """解析LLM输出，提取Thought/Action/Action Input/Final Answer"""
        result = {"thought": "", "action": "", "action_input": "", "final_answer": ""}

        # 提取Thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|\Z)", text, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # 提取Final Answer
        final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        if final_match:
            result["final_answer"] = final_match.group(1).strip()
            return result

        # 提取Action和Action Input
        action_match = re.search(r"Action:\s*(.+?)(?=\n|$)", text)
        if action_match:
            result["action"] = action_match.group(1).strip()

        input_match = re.search(r"Action Input:\s*(.+?)(?=\n|$)", text)
        if input_match:
            result["action_input"] = input_match.group(1).strip()

        return result

    def _execute_tool(self, action: str, action_input: str) -> str:
        """执行工具调用"""
        if action not in self.tools:
            return f"错误: 未知工具 '{action}'。可用工具: {', '.join(self.tools.keys())}"
        try:
            result = self.tools[action]["func"](action_input)
            return result
        except Exception as e:
            return f"工具执行错误: {str(e)}"

    def run(self, question: str) -> str:
        """运行ReAct Agent"""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print('='*60)

        system_prompt = self._build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"}
        ]

        # 构建agent_scratchpad（累积的推理历史）
        scratchpad = ""

        for step in range(self.max_steps):
            if self.verbose:
                print(f"\n--- Step {step + 1} ---")

            # 调用LLM
            current_messages = messages.copy()
            if scratchpad:
                current_messages.append({
                    "role": "assistant",
                    "content": scratchpad
                })

            response = self.client.chat.completions.create(
                model=self.model,
                messages=current_messages,
                temperature=0,
                max_tokens=1000,
                stop=["\nObservation:"]  # 在Observation前停止，由我们来填充
            )

            llm_output = response.choices[0].message.content
            parsed = self._parse_llm_output(llm_output)

            if self.verbose:
                print(f"Thought: {parsed['thought']}")

            # 检查是否得到最终答案
            if parsed["final_answer"]:
                if self.verbose:
                    print(f"Final Answer: {parsed['final_answer']}")
                return parsed["final_answer"]

            # 执行工具
            if parsed["action"]:
                if self.verbose:
                    print(f"Action: {parsed['action']}")
                    print(f"Action Input: {parsed['action_input']}")

                observation = self._execute_tool(parsed["action"], parsed["action_input"])

                if self.verbose:
                    print(f"Observation: {observation}")

                # 更新scratchpad
                scratchpad += llm_output + f"\nObservation: {observation}\n"
            else:
                # LLM没有给出有效的Action或Final Answer
                if self.verbose:
                    print("Warning: LLM输出格式不正确，尝试重新引导...")
                scratchpad += llm_output + "\nPlease follow the correct format.\n"

        # 达到最大步数
        return "抱歉，在规定步数内未能得出答案。请尝试简化问题。"


# ============================================================
# 3. 运行测试
# ============================================================
def main():
    agent = ReActAgent(tools=TOOLS, verbose=True)

    # 测试1: 简单搜索
    print("\n" + "="*80)
    print("测试1: 简单事实查询")
    result = agent.run("Python是谁创建的？")
    print(f"\n最终答案: {result}")

    # 测试2: 多步推理
    print("\n" + "="*80)
    print("测试2: 需要搜索和计算")
    result = agent.run("地球的半径是多少公里？据此计算地球的周长（使用公式 2*pi*r）")
    print(f"\n最终答案: {result}")

    # 测试3: 多工具协作
    print("\n" + "="*80)
    print("测试3: 多工具协作")
    result = agent.run("什么是ReAct？它是哪一年提出的？")
    print(f"\n最终答案: {result}")


if __name__ == "__main__":
    main()
```

### 带有详细日志的增强版

```python
"""
增强版ReAct Agent - 带详细日志和错误恢复
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ReActAgent")


@dataclass
class AgentStep:
    """Agent执行步骤记录"""
    step_number: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None
    final_answer: Optional[str] = None
    duration_ms: float = 0
    error: Optional[str] = None


@dataclass
class AgentTrace:
    """Agent完整执行追踪"""
    question: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    total_duration_ms: float = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "question": self.question,
            "final_answer": self.final_answer,
            "total_steps": len(self.steps),
            "total_duration_ms": self.total_duration_ms,
            "total_llm_calls": self.total_llm_calls,
            "total_tool_calls": self.total_tool_calls,
            "steps": [
                {
                    "step": s.step_number,
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": s.observation,
                    "duration_ms": s.duration_ms,
                }
                for s in self.steps
            ]
        }

    def print_summary(self):
        """打印执行摘要"""
        print(f"\n{'='*60}")
        print(f"执行摘要")
        print(f"{'='*60}")
        print(f"问题: {self.question}")
        print(f"答案: {self.final_answer}")
        print(f"总步数: {len(self.steps)}")
        print(f"LLM调用次数: {self.total_llm_calls}")
        print(f"工具调用次数: {self.total_tool_calls}")
        print(f"总耗时: {self.total_duration_ms:.0f}ms")
        print(f"{'='*60}")


class EnhancedReActAgent(ReActAgent):
    """增强版ReAct Agent"""

    def __init__(self, *args, max_retries: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self.trace: Optional[AgentTrace] = None

    def run_with_trace(self, question: str) -> tuple[str, AgentTrace]:
        """运行Agent并返回完整追踪"""
        self.trace = AgentTrace(question=question)
        start_time = time.time()

        result = self.run(question)

        self.trace.final_answer = result
        self.trace.total_duration_ms = (time.time() - start_time) * 1000
        self.trace.print_summary()

        return result, self.trace


# 使用示例
if __name__ == "__main__":
    agent = EnhancedReActAgent(tools=TOOLS, verbose=True)
    answer, trace = agent.run_with_trace("Python最新版本是什么？有哪些新特性？")
    print(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2))
```

---

## LangChain ReAct实现

### 使用LangChain内置ReAct Agent

```python
"""
使用LangChain构建ReAct Agent
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

# ============================================================
# 1. 初始化工具
# ============================================================
search = DuckDuckGoSearchRun()
wikipedia = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)

@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息。输入搜索关键词。"""
    return search.run(query)

@tool
def wiki_search(query: str) -> str:
    """在维基百科中搜索信息。适合搜索人物、地点、事件等百科知识。"""
    return wikipedia.run(query)

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入有效的Python数学表达式。"""
    import math
    safe = {"abs": abs, "round": round, "sqrt": math.sqrt, "pi": math.pi}
    try:
        return str(eval(expression, {"__builtins__": {}}, safe))
    except Exception as e:
        return f"Error: {e}"

@tool
def string_reverse(text: str) -> str:
    """反转字符串。输入要反转的文本。"""
    return text[::-1]

tools = [web_search, wiki_search, calculator, string_reverse]

# ============================================================
# 2. 定义ReAct Prompt
# ============================================================
react_prompt = PromptTemplate.from_template(
"""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Always answer in Chinese (中文).

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

# ============================================================
# 3. 创建ReAct Agent
# ============================================================
llm = ChatOpenAI(model="gpt-4o", temperature=0)

agent = create_react_agent(llm, tools, react_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)

# ============================================================
# 4. 运行
# ============================================================
if __name__ == "__main__":
    # 测试1: 需要搜索的问题
    result = agent_executor.invoke({
        "input": "2024年奥运会在哪里举办？有多少个比赛项目？"
    })
    print(f"答案: {result['output']}")
    print(f"执行步骤数: {len(result['intermediate_steps'])}")

    # 查看中间步骤
    for i, (action, observation) in enumerate(result['intermediate_steps']):
        print(f"\n步骤 {i+1}:")
        print(f"  工具: {action.tool}")
        print(f"  输入: {action.tool_input}")
        print(f"  结果: {observation[:100]}...")
```

### 自定义ReAct Prompt（中文优化版）

```python
chinese_react_prompt = PromptTemplate.from_template(
"""你是一个善于分析和解决问题的AI助手。请根据以下步骤回答问题。

可用工具:
{tools}

请严格按照以下格式回答:

问题: 用户的问题
思考: 分析问题，思考需要什么信息，决定使用哪个工具
操作: 选择工具，必须是 [{tool_names}] 之一
操作输入: 工具的输入参数
观察结果: 工具返回的结果
... (思考/操作/操作输入/观察结果 可以重复多次)
思考: 我现在有足够的信息来回答问题了
最终答案: 完整的回答

注意事项:
1. 每一步都要先思考再行动
2. 如果一个工具返回的信息不够，尝试用不同的关键词或其他工具
3. 回答要全面、准确、使用中文

开始!

问题: {input}
思考:{agent_scratchpad}"""
)
```

---

## 实战：Web搜索Agent

### 集成Tavily搜索的ReAct Agent

```python
"""
实战：Web搜索ReAct Agent
集成Tavily搜索API（也支持SerpAPI）
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

load_dotenv()

# ============================================================
# 1. 搜索工具（支持多个搜索后端）
# ============================================================

# 方式A: Tavily搜索（推荐，专为AI Agent设计）
# pip install tavily-python
class TavilySearchInput(BaseModel):
    query: str = Field(description="搜索查询关键词")

@tool(args_schema=TavilySearchInput)
def tavily_search(query: str) -> str:
    """使用Tavily搜索引擎搜索最新信息。适合搜索新闻、技术文章、产品信息等。"""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True
        )
        # 格式化结果
        results = []
        if response.get("answer"):
            results.append(f"AI摘要: {response['answer']}")
        for item in response.get("results", [])[:3]:
            results.append(f"- {item['title']}: {item['content'][:200]}")
        return "\n".join(results)
    except Exception as e:
        return f"搜索错误: {str(e)}"


# 方式B: DuckDuckGo搜索（免费，无需API Key）
@tool
def ddg_search(query: str) -> str:
    """使用DuckDuckGo搜索引擎搜索信息。免费无限制。"""
    from langchain_community.tools import DuckDuckGoSearchRun
    search = DuckDuckGoSearchRun()
    try:
        return search.run(query)
    except Exception as e:
        return f"搜索错误: {str(e)}"


# ============================================================
# 2. 辅助工具
# ============================================================
@tool
def extract_url_content(url: str) -> str:
    """提取URL网页的文本内容。输入完整的URL地址。"""
    import requests
    from bs4 import BeautifulSoup
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        # 移除script和style
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:2000]  # 限制长度
    except Exception as e:
        return f"获取网页内容失败: {str(e)}"


@tool
def summarize_text(text: str) -> str:
    """总结长文本。输入要总结的文本。"""
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "用中文简洁总结以下文本，保留关键信息，不超过200字。"},
            {"role": "user", "content": text[:3000]}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。"""
    import math
    safe = {
        "abs": abs, "round": round, "sqrt": math.sqrt,
        "pi": math.pi, "e": math.e, "log": math.log,
        "pow": pow, "min": min, "max": max,
    }
    try:
        return f"结果: {eval(expression, {'__builtins__': {}}, safe)}"
    except Exception as e:
        return f"计算错误: {e}"


# ============================================================
# 3. 构建搜索Agent
# ============================================================
tools = [tavily_search, ddg_search, extract_url_content, summarize_text, calculator]

search_prompt = PromptTemplate.from_template(
"""You are an expert research assistant. You can search the web, read web pages, and analyze information.

Available tools:
{tools}

Strategy:
1. Start with a broad search to understand the topic
2. Use specific searches to find detailed information
3. Extract content from URLs when needed
4. Summarize long texts before presenting
5. Use calculator for any numerical analysis

Format:
Question: the input question
Thought: your reasoning
Action: one of [{tool_names}]
Action Input: the input
Observation: the result
... (repeat as needed)
Thought: I now know the final answer
Final Answer: comprehensive answer in Chinese

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_react_agent(llm, tools, search_prompt)

search_agent = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=8,
    handle_parsing_errors=True,
)


# ============================================================
# 4. 运行搜索任务
# ============================================================
def research(question: str) -> str:
    """执行研究任务"""
    result = search_agent.invoke({"input": question})
    return result["output"]


if __name__ == "__main__":
    # 研究任务示例
    questions = [
        "对比Python 3.12和3.11的主要区别，各有什么新特性？",
        "2024年AI领域最重要的三个突破是什么？",
        "LangChain和LlamaIndex有什么区别？各自的优势是什么？",
    ]

    for q in questions:
        print(f"\n{'='*70}")
        print(f"研究问题: {q}")
        print('='*70)
        answer = research(q)
        print(f"\n研究结果:\n{answer}")
```

---

## ReAct vs CoT vs Act-only对比

### 对比架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                   三种方法对比                                    │
│                                                                  │
│  ┌─────────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │  CoT (思维链)    │  │  Act-only      │  │  ReAct           │  │
│  │                 │  │                │  │                  │  │
│  │  Q: 问题        │  │  Q: 问题       │  │  Q: 问题         │  │
│  │  │              │  │  │             │  │  │               │  │
│  │  ▼              │  │  ▼             │  │  ▼               │  │
│  │  Think 1        │  │  Action 1      │  │  Thought 1       │  │
│  │  │              │  │  │             │  │  │               │  │
│  │  ▼              │  │  ▼             │  │  ▼               │  │
│  │  Think 2        │  │  Obs 1         │  │  Action 1        │  │
│  │  │              │  │  │             │  │  │               │  │
│  │  ▼              │  │  ▼             │  │  ▼               │  │
│  │  Think 3        │  │  Action 2      │  │  Observation 1   │  │
│  │  │              │  │  │             │  │  │               │  │
│  │  ▼              │  │  ▼             │  │  ▼               │  │
│  │  Answer         │  │  Obs 2         │  │  Thought 2       │  │
│  │                 │  │  │             │  │  │               │  │
│  │  ✗ 无法获取     │  │  ▼             │  │  ▼               │  │
│  │    外部信息     │  │  Answer        │  │  Action 2        │  │
│  │  ✗ 容易幻觉     │  │               │  │  │               │  │
│  │  ✓ 推理清晰     │  │  ✗ 缺乏推理   │  │  ▼               │  │
│  │                 │  │  ✗ 盲目行动   │  │  Observation 2   │  │
│  │                 │  │  ✓ 能用工具   │  │  │               │  │
│  │                 │  │               │  │  ▼               │  │
│  │                 │  │               │  │  Thought 3       │  │
│  │                 │  │               │  │  │               │  │
│  │                 │  │               │  │  ▼               │  │
│  │                 │  │               │  │  Answer          │  │
│  │                 │  │               │  │                  │  │
│  │                 │  │               │  │  ✓ 推理+行动     │  │
│  │                 │  │               │  │  ✓ 信息准确     │  │
│  │                 │  │               │  │  ✓ 可解释性强   │  │
│  └─────────────────┘  └────────────────┘  └──────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 对比实验代码

```python
"""
ReAct vs CoT vs Act-only 对比实验
"""
from openai import OpenAI

client = OpenAI()

# ============================================================
# 1. CoT (Chain of Thought) - 仅推理
# ============================================================
def cot_answer(question: str) -> str:
    """纯思维链推理，不使用工具"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "请一步步推理来回答问题。使用中文回答。"
            },
            {
                "role": "user",
                "content": f"请一步步思考并回答：{question}"
            }
        ],
        temperature=0
    )
    return response.choices[0].message.content


# ============================================================
# 2. Act-only - 仅行动，不显式推理
# ============================================================
def act_only_answer(question: str, tools: dict) -> str:
    """直接行动，不经过推理"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""You have these tools: {list(tools.keys())}.
                Directly choose and use a tool. Output format:
                Action: <tool>
                Action Input: <input>"""
            },
            {"role": "user", "content": question}
        ],
        temperature=0
    )

    text = response.choices[0].message.content
    # 解析并执行...（简化版）
    import re
    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match = re.search(r"Action Input:\s*(.+)", text)

    if action_match and input_match:
        action = action_match.group(1)
        action_input = input_match.group(1).strip()
        if action in tools:
            result = tools[action]["func"](action_input)
            return f"工具结果: {result}"
    return "无法执行"


# ============================================================
# 3. ReAct - 推理+行动（使用前面实现的ReActAgent）
# ============================================================
# react_agent = ReActAgent(tools=TOOLS) # 使用前面定义的


# ============================================================
# 4. 对比实验
# ============================================================
def run_comparison():
    """运行对比实验"""
    test_questions = [
        "地球到月球的距离是多少公里？以光速需要多少秒？",
        "Python最新稳定版本是什么？有哪些新特性？",
        "ReAct论文是哪一年发表的？核心贡献是什么？",
    ]

    for q in test_questions:
        print(f"\n{'='*70}")
        print(f"问题: {q}")
        print('='*70)

        # CoT
        print("\n[CoT 方法]")
        cot_result = cot_answer(q)
        print(f"结果: {cot_result[:200]}...")

        # Act-only
        print("\n[Act-only 方法]")
        act_result = act_only_answer(q, TOOLS)
        print(f"结果: {act_result[:200]}...")

        # ReAct
        print("\n[ReAct 方法]")
        react_agent = ReActAgent(tools=TOOLS, verbose=False)
        react_result = react_agent.run(q)
        print(f"结果: {react_result[:200]}...")


if __name__ == "__main__":
    run_comparison()
```

### 对比总结

| 维度 | CoT | Act-only | ReAct |
|------|-----|----------|-------|
| **信息准确性** | 低（依赖训练数据） | 中（有工具但缺乏判断） | 高（推理指导工具使用） |
| **推理深度** | 高 | 低 | 高 |
| **工具利用** | 无 | 有但盲目 | 有且智能 |
| **幻觉风险** | 高 | 中 | 低 |
| **效率** | 高（一次调用） | 中 | 低（多次调用） |
| **可解释性** | 中 | 低 | 高 |
| **适用场景** | 纯推理题 | 简单查询 | 复杂任务 |

---

## 调试技巧与常见问题

### 常见问题与解决方案

```
┌──────────────────────────────────────────────────────────────┐
│                ReAct Agent 常见问题诊断                       │
│                                                              │
│  问题1: Agent陷入循环                                        │
│  ├─ 症状: 重复相同的Thought和Action                          │
│  ├─ 原因: 工具返回的信息不够明确                              │
│  └─ 解决:                                                    │
│     ├─ 增加stop条件                                          │
│     ├─ 优化工具返回格式                                      │
│     └─ 设置max_iterations限制                                │
│                                                              │
│  问题2: 格式解析错误                                         │
│  ├─ 症状: 无法提取Action/Action Input                        │
│  ├─ 原因: LLM未按照格式输出                                  │
│  └─ 解决:                                                    │
│     ├─ 使用handle_parsing_errors=True                        │
│     ├─ 在prompt中加入更多示例                                │
│     └─ 使用更强的模型(GPT-4o)                                │
│                                                              │
│  问题3: 工具选择错误                                         │
│  ├─ 症状: 用错工具或传入错误参数                              │
│  ├─ 原因: 工具描述不清晰                                    │
│  └─ 解决:                                                    │
│     ├─ 优化工具的description                                 │
│     ├─ 减少工具数量(去除冗余)                                │
│     └─ 添加工具使用示例到prompt                              │
│                                                              │
│  问题4: Token消耗过多                                        │
│  ├─ 症状: 成本高，响应慢                                     │
│  ├─ 原因: scratchpad过长                                     │
│  └─ 解决:                                                    │
│     ├─ 限制observation长度                                   │
│     ├─ 减少max_iterations                                    │
│     └─ 使用摘要压缩历史                                      │
└──────────────────────────────────────────────────────────────┘
```

### 调试工具代码

```python
"""
ReAct Agent 调试工具箱
"""
import time
import json
from typing import Any
from functools import wraps


# ============================================================
# 1. 工具调用追踪装饰器
# ============================================================
def trace_tool(func):
    """追踪工具调用的装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        tool_name = func.__name__
        print(f"  [TOOL] 调用 {tool_name}")
        print(f"  [TOOL] 输入: {args}, {kwargs}")
        try:
            result = func(*args, **kwargs)
            duration = (time.time() - start) * 1000
            print(f"  [TOOL] 输出: {str(result)[:200]}")
            print(f"  [TOOL] 耗时: {duration:.0f}ms")
            return result
        except Exception as e:
            print(f"  [TOOL] 错误: {str(e)}")
            raise
    return wrapper


# ============================================================
# 2. Prompt调试器
# ============================================================
class PromptDebugger:
    """检查和优化ReAct Prompt"""

    @staticmethod
    def check_format(prompt_template: str) -> list[str]:
        """检查prompt模板是否包含必要元素"""
        issues = []
        required = [
            ("{tools}", "缺少工具描述占位符 {tools}"),
            ("{tool_names}", "缺少工具名称占位符 {tool_names}"),
            ("{input}", "缺少输入占位符 {input}"),
            ("{agent_scratchpad}", "缺少scratchpad占位符 {agent_scratchpad}"),
            ("Thought:", "缺少Thought格式说明"),
            ("Action:", "缺少Action格式说明"),
            ("Action Input:", "缺少Action Input格式说明"),
            ("Observation:", "缺少Observation格式说明"),
            ("Final Answer:", "缺少Final Answer格式说明"),
        ]
        for keyword, message in required:
            if keyword not in prompt_template:
                issues.append(message)
        return issues

    @staticmethod
    def estimate_tokens(prompt: str, scratchpad: str = "") -> dict:
        """估算token使用量"""
        # 粗略估算：英文约4字符/token，中文约2字符/token
        total_chars = len(prompt) + len(scratchpad)
        estimated_tokens = total_chars // 3  # 混合语言粗略估算
        return {
            "prompt_chars": len(prompt),
            "scratchpad_chars": len(scratchpad),
            "estimated_tokens": estimated_tokens,
            "warning": "接近token限制" if estimated_tokens > 3000 else "正常"
        }


# ============================================================
# 3. Agent执行可视化
# ============================================================
def visualize_agent_steps(intermediate_steps: list) -> str:
    """可视化Agent执行步骤"""
    output = []
    output.append("┌─ Agent 执行流程 ─────────────────────────────┐")

    for i, (action, observation) in enumerate(intermediate_steps):
        step_num = i + 1
        output.append(f"│                                              │")
        output.append(f"│  Step {step_num}:                                      │")
        output.append(f"│  ├─ Tool: {action.tool:<35}│")
        input_str = str(action.tool_input)[:35]
        output.append(f"│  ├─ Input: {input_str:<34}│")
        obs_str = str(observation)[:35]
        output.append(f"│  └─ Result: {obs_str:<33}│")
        if i < len(intermediate_steps) - 1:
            output.append(f"│       │                                      │")
            output.append(f"│       ▼                                      │")

    output.append(f"│                                              │")
    output.append("└──────────────────────────────────────────────┘")
    return "\n".join(output)


# ============================================================
# 4. 错误恢复策略
# ============================================================
class ErrorRecoveryAgent:
    """带错误恢复机制的Agent"""

    def __init__(self, agent_executor, max_retries: int = 3):
        self.agent = agent_executor
        self.max_retries = max_retries

    def run(self, question: str) -> str:
        """带重试的Agent执行"""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                result = self.agent.invoke({"input": question})
                return result["output"]
            except Exception as e:
                last_error = str(e)
                print(f"尝试 {attempt + 1}/{self.max_retries} 失败: {last_error}")

                # 根据错误类型调整策略
                if "rate_limit" in last_error.lower():
                    time.sleep(5 * (attempt + 1))  # 指数退避
                elif "parsing" in last_error.lower():
                    # 尝试简化问题
                    question = f"Please answer simply: {question}"
                else:
                    time.sleep(1)

        return f"所有重试均失败。最后错误: {last_error}"


# ============================================================
# 5. 性能监控
# ============================================================
class PerformanceMonitor:
    """Agent性能监控"""

    def __init__(self):
        self.records = []

    def record(self, question: str, steps: int, duration: float, success: bool):
        """记录执行数据"""
        self.records.append({
            "question": question,
            "steps": steps,
            "duration_ms": duration,
            "success": success,
            "timestamp": time.time()
        })

    def report(self) -> dict:
        """生成性能报告"""
        if not self.records:
            return {"message": "暂无数据"}

        total = len(self.records)
        successful = sum(1 for r in self.records if r["success"])
        avg_steps = sum(r["steps"] for r in self.records) / total
        avg_duration = sum(r["duration_ms"] for r in self.records) / total

        return {
            "total_queries": total,
            "success_rate": f"{successful/total*100:.1f}%",
            "avg_steps": f"{avg_steps:.1f}",
            "avg_duration_ms": f"{avg_duration:.0f}",
            "max_steps": max(r["steps"] for r in self.records),
            "max_duration_ms": f"{max(r['duration_ms'] for r in self.records):.0f}",
        }

    def print_report(self):
        """打印性能报告"""
        report = self.report()
        print("\n=== Agent性能报告 ===")
        for key, value in report.items():
            print(f"  {key}: {value}")


# 使用示例
if __name__ == "__main__":
    # Prompt检查
    debugger = PromptDebugger()
    issues = debugger.check_format("""
    Answer the question.
    Thought: think
    Action: act
    """)
    print("Prompt问题:", issues)

    # Token估算
    tokens = debugger.estimate_tokens("这是一个测试prompt" * 100)
    print("Token估算:", tokens)
```

---

## ReAct高级模式

### 多工具协同的ReAct Agent

在实际场景中，ReAct Agent往往需要在单次任务中协同使用多个工具。以下是一个完整的多工具ReAct实战案例。

```python
"""
高级ReAct Agent：多工具协同 + 上下文管理 + 流式输出
"""
import re
import json
import time
from openai import OpenAI
from typing import Callable, Optional

client = OpenAI()


class AdvancedReActAgent:
    """
    高级ReAct Agent实现
    特性:
    - 多工具协同
    - 上下文窗口管理
    - Observation截断
    - 思考链自我纠错
    - 结构化执行日志
    """

    def __init__(
        self,
        tools: dict,
        model: str = "gpt-4o",
        max_steps: int = 10,
        max_observation_length: int = 1500,
        verbose: bool = True
    ):
        self.tools = tools
        self.model = model
        self.max_steps = max_steps
        self.max_obs_len = max_observation_length
        self.verbose = verbose
        self.execution_log = []

    def _truncate_observation(self, obs: str) -> str:
        """截断过长的Observation，保留首尾"""
        if len(obs) <= self.max_obs_len:
            return obs
        half = self.max_obs_len // 2
        return (
            obs[:half] +
            f"\n... [截断了 {len(obs) - self.max_obs_len} 个字符] ...\n" +
            obs[-half:]
        )

    def _build_scratchpad(self, steps: list) -> str:
        """构建scratchpad，控制总长度"""
        parts = []
        for step in steps:
            parts.append(f"Thought: {step['thought']}")
            if step.get("action"):
                parts.append(f"Action: {step['action']}")
                parts.append(f"Action Input: {step['action_input']}")
                parts.append(f"Observation: {step['observation']}")
        return "\n".join(parts)

    def _parse_output(self, text: str) -> dict:
        """解析LLM输出"""
        result = {
            "thought": "", "action": "",
            "action_input": "", "final_answer": ""
        }

        # Final Answer
        final_match = re.search(
            r"Final Answer:\s*(.+)", text, re.DOTALL
        )
        if final_match:
            result["final_answer"] = final_match.group(1).strip()
            return result

        # Thought
        thought_match = re.search(
            r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|\Z)",
            text, re.DOTALL
        )
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # Action
        action_match = re.search(r"Action:\s*(.+?)(?=\n|$)", text)
        if action_match:
            result["action"] = action_match.group(1).strip()

        # Action Input
        input_match = re.search(r"Action Input:\s*(.+?)(?=\n|$)", text)
        if input_match:
            result["action_input"] = input_match.group(1).strip()

        return result

    def run(self, question: str) -> dict:
        """运行Agent，返回结构化结果"""
        start_time = time.time()
        steps = []

        tool_descriptions = "\n".join([
            f"  {name}: {info['description']}"
            for name, info in self.tools.items()
        ])
        tool_names = ", ".join(self.tools.keys())

        system_prompt = f"""You are a helpful assistant. Answer questions using tools.

Available tools:
{tool_descriptions}

STRICT FORMAT (follow exactly):

Thought: <reasoning>
Action: <one of [{tool_names}]>
Action Input: <input>

OR when done:

Thought: I now know the final answer
Final Answer: <complete answer in Chinese>

Rules:
1. Always think before acting
2. Use multiple tools if needed
3. Verify information across sources when possible
4. If a tool fails, try a different approach
5. Final answer must be comprehensive and in Chinese"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"}
        ]

        for step_num in range(self.max_steps):
            scratchpad = self._build_scratchpad(steps)

            current_messages = messages.copy()
            if scratchpad:
                current_messages.append({
                    "role": "assistant",
                    "content": scratchpad
                })

            response = client.chat.completions.create(
                model=self.model,
                messages=current_messages,
                temperature=0,
                max_tokens=1000,
                stop=["\nObservation:"]
            )

            output = response.choices[0].message.content
            parsed = self._parse_output(output)

            step_data = {
                "step": step_num + 1,
                "thought": parsed["thought"],
                "action": parsed.get("action", ""),
                "action_input": parsed.get("action_input", ""),
                "observation": "",
                "duration_ms": 0
            }

            if self.verbose:
                print(f"\n--- Step {step_num + 1} ---")
                print(f"Thought: {parsed['thought']}")

            # 最终答案
            if parsed["final_answer"]:
                if self.verbose:
                    print(f"Final Answer: {parsed['final_answer']}")

                total_time = (time.time() - start_time) * 1000
                return {
                    "answer": parsed["final_answer"],
                    "steps": steps,
                    "total_steps": len(steps),
                    "total_time_ms": round(total_time, 0),
                    "tools_used": list(set(
                        s["action"] for s in steps if s["action"]
                    ))
                }

            # 执行工具
            if parsed["action"]:
                tool_start = time.time()
                action = parsed["action"]
                action_input = parsed["action_input"]

                if self.verbose:
                    print(f"Action: {action}")
                    print(f"Action Input: {action_input}")

                if action in self.tools:
                    try:
                        obs = self.tools[action]["func"](action_input)
                    except Exception as e:
                        obs = f"工具执行错误: {str(e)}"
                else:
                    obs = f"未知工具: {action}。可用: {tool_names}"

                obs = self._truncate_observation(str(obs))
                step_data["observation"] = obs
                step_data["duration_ms"] = (time.time() - tool_start) * 1000

                if self.verbose:
                    print(f"Observation: {obs[:200]}...")

                steps.append(step_data)
            else:
                steps.append(step_data)

        total_time = (time.time() - start_time) * 1000
        return {
            "answer": "在规定步骤内未能得出答案",
            "steps": steps,
            "total_steps": len(steps),
            "total_time_ms": round(total_time, 0),
            "tools_used": []
        }


# ============================================================
# 工具定义
# ============================================================
import math

def search(query: str) -> str:
    """模拟搜索"""
    data = {
        "python 3.12": "Python 3.12于2023年10月发布，新特性包括：改进的f-string、"
                       "类型参数语法(PEP 695)、per-interpreter GIL(PEP 684)等",
        "langchain": "LangChain是最流行的LLM应用框架，支持Agent、RAG、Chain等模式",
        "react paper": "ReAct论文由Yao等人于2022年发表，提出了推理与行动交替的范式",
        "gpt-4o": "GPT-4o是OpenAI的多模态模型，支持文本、图像、音频输入",
    }
    for key, value in data.items():
        if key in query.lower():
            return value
    return f"关于'{query}'的搜索结果：找到相关信息若干条..."

def calculate(expression: str) -> str:
    safe = {
        "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
        "log": math.log, "sin": math.sin, "cos": math.cos,
        "abs": abs, "round": round, "pow": pow, "min": min, "max": max,
    }
    try:
        return str(eval(expression, {"__builtins__": {}}, safe))
    except Exception as e:
        return f"计算错误: {e}"

def lookup(term: str) -> str:
    defs = {
        "agent": "AI Agent是能自主感知环境、决策和执行的智能系统",
        "react": "ReAct=Reasoning+Acting，交替推理与行动的Agent范式",
        "rag": "RAG=Retrieval Augmented Generation，检索增强生成",
        "cot": "CoT=Chain of Thought，思维链推理方法",
    }
    return defs.get(term.lower(), f"未找到'{term}'的定义")

TOOLS = {
    "Search": {
        "func": search,
        "description": "搜索互联网信息。输入: 搜索关键词"
    },
    "Calculate": {
        "func": calculate,
        "description": "计算数学表达式。输入: 如 '2+3*4' 或 'sqrt(144)'"
    },
    "Lookup": {
        "func": lookup,
        "description": "查找术语定义。输入: 术语名称"
    },
}

# 运行测试
if __name__ == "__main__":
    agent = AdvancedReActAgent(tools=TOOLS, verbose=True)

    # 测试：需要多工具协同的问题
    result = agent.run(
        "什么是ReAct？它和CoT有什么区别？"
        "另外计算一下 pi * 6371^2 (地球表面积的一部分)"
    )

    print(f"\n{'='*60}")
    print(f"最终答案: {result['answer']}")
    print(f"总步骤: {result['total_steps']}")
    print(f"使用工具: {result['tools_used']}")
    print(f"总耗时: {result['total_time_ms']:.0f}ms")
```

### ReAct流式输出实现

```python
"""
ReAct Agent 流式输出：让用户实时看到思考过程
"""
from openai import OpenAI
import sys

client = OpenAI()


def stream_react_step(messages: list, model: str = "gpt-4o"):
    """流式输出ReAct的单步思考"""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=500,
        stream=True,
        stop=["\nObservation:"]
    )

    full_response = ""
    current_section = ""

    for chunk in stream:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_response += token

            # 高亮不同部分
            if "Thought:" in full_response and current_section != "thought":
                current_section = "thought"
                sys.stdout.write("\033[94m")  # 蓝色
            elif "Action:" in full_response and current_section != "action":
                current_section = "action"
                sys.stdout.write("\033[92m")  # 绿色
            elif "Final Answer:" in full_response and current_section != "answer":
                current_section = "answer"
                sys.stdout.write("\033[93m")  # 黄色

            sys.stdout.write(token)
            sys.stdout.flush()

    sys.stdout.write("\033[0m\n")  # 重置颜色
    return full_response


# 使用示例（需要终端支持ANSI颜色）
# stream_react_step([
#     {"role": "system", "content": "You are a ReAct agent..."},
#     {"role": "user", "content": "Question: Python最新版本?"}
# ])
```

### ReAct与Function Calling的融合

```python
"""
现代ReAct: 结合OpenAI Function Calling的ReAct实现
传统ReAct依赖文本解析，Function Calling提供结构化工具调用
"""
import json
from openai import OpenAI

client = OpenAI()


# 工具定义（JSON Schema格式）
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "搜索互联网信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


def modern_react_agent(question: str, max_iterations: int = 8) -> str:
    """
    现代ReAct Agent：结合Function Calling
    优势：不再需要文本解析，工具调用更可靠
    """
    messages = [
        {
            "role": "system",
            "content": """你是一个ReAct风格的Agent。
对于每个问题：
1. 先在回复中说明你的思考(Thought)
2. 然后决定是否需要使用工具
3. 如果不需要工具，直接给出最终答案
4. 用中文思考和回答

重要：每次回复时先说"思考："开头，阐述你的推理过程。"""
        },
        {"role": "user", "content": question}
    ]

    for i in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0
        )

        msg = response.choices[0].message

        # 显示思考过程
        if msg.content:
            print(f"\n[思考] {msg.content}")

        # 如果没有工具调用，说明已得到最终答案
        if not msg.tool_calls:
            return msg.content

        # 执行工具调用
        messages.append(msg)

        for tc in msg.tool_calls:
            func_name = tc.function.name
            func_args = json.loads(tc.function.arguments)

            print(f"[工具] {func_name}({json.dumps(func_args, ensure_ascii=False)})")

            # 执行工具（使用前面定义的函数）
            tool_functions = {
                "search": lambda q: search(q),
                "calculate": lambda e: calculate(e),
            }

            if func_name in tool_functions:
                result = tool_functions[func_name](**func_args)
            else:
                result = f"未知工具: {func_name}"

            print(f"[结果] {result[:200]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result)
            })

    return "达到最大迭代次数，未能得出答案。"


# 测试
if __name__ == "__main__":
    answer = modern_react_agent(
        "Python 3.12有什么新特性？计算一下如果每个特性节省5%开发时间，"
        "3个主要特性总共能节省百分之多少时间？"
    )
    print(f"\n最终答案: {answer}")
```

### ReAct Agent评估框架

```python
"""
ReAct Agent 评估：系统化测试Agent质量
"""
import time
import json
from typing import Callable
from dataclasses import dataclass, field


@dataclass
class TestCase:
    """测试用例"""
    question: str
    expected_tools: list[str]    # 期望使用的工具
    expected_keywords: list[str]  # 答案中应包含的关键词
    max_steps: int = 5           # 期望最大步数
    timeout_ms: int = 30000      # 超时时间


@dataclass
class TestResult:
    """测试结果"""
    test_case: TestCase
    passed: bool
    answer: str
    actual_tools: list[str]
    actual_steps: int
    duration_ms: float
    keyword_hits: int
    keyword_total: int
    errors: list[str] = field(default_factory=list)


class ReActEvaluator:
    """ReAct Agent评估器"""

    def __init__(self, agent_func: Callable):
        self.agent_func = agent_func
        self.results: list[TestResult] = []

    def evaluate(self, test_cases: list[TestCase]) -> dict:
        """运行所有测试用例"""
        print(f"\n{'='*60}")
        print(f"开始评估 ({len(test_cases)} 个测试用例)")
        print(f"{'='*60}")

        for i, tc in enumerate(test_cases, 1):
            print(f"\n--- 测试 {i}/{len(test_cases)} ---")
            print(f"问题: {tc.question[:60]}...")

            start = time.time()
            try:
                result = self.agent_func(tc.question)
                duration = (time.time() - start) * 1000

                answer = result.get("answer", "")
                tools = result.get("tools_used", [])
                steps = result.get("total_steps", 0)

                # 检查关键词
                keyword_hits = sum(
                    1 for kw in tc.expected_keywords
                    if kw.lower() in answer.lower()
                )

                # 检查工具使用
                tool_match = all(
                    t in tools for t in tc.expected_tools
                )

                # 判断是否通过
                passed = (
                    keyword_hits >= len(tc.expected_keywords) * 0.5 and
                    steps <= tc.max_steps and
                    duration <= tc.timeout_ms
                )

                test_result = TestResult(
                    test_case=tc,
                    passed=passed,
                    answer=answer[:200],
                    actual_tools=tools,
                    actual_steps=steps,
                    duration_ms=duration,
                    keyword_hits=keyword_hits,
                    keyword_total=len(tc.expected_keywords)
                )

            except Exception as e:
                duration = (time.time() - start) * 1000
                test_result = TestResult(
                    test_case=tc,
                    passed=False,
                    answer="",
                    actual_tools=[],
                    actual_steps=0,
                    duration_ms=duration,
                    keyword_hits=0,
                    keyword_total=len(tc.expected_keywords),
                    errors=[str(e)]
                )

            self.results.append(test_result)
            status = "PASS" if test_result.passed else "FAIL"
            print(f"  [{status}] 步骤:{test_result.actual_steps} "
                  f"关键词:{test_result.keyword_hits}/{test_result.keyword_total} "
                  f"耗时:{test_result.duration_ms:.0f}ms")

        return self._generate_report()

    def _generate_report(self) -> dict:
        """生成评估报告"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        avg_steps = sum(r.actual_steps for r in self.results) / max(total, 1)
        avg_time = sum(r.duration_ms for r in self.results) / max(total, 1)
        avg_keywords = (
            sum(r.keyword_hits / max(r.keyword_total, 1) for r in self.results)
            / max(total, 1) * 100
        )

        report = {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": f"{passed/max(total,1)*100:.1f}%",
            "avg_steps": f"{avg_steps:.1f}",
            "avg_duration_ms": f"{avg_time:.0f}",
            "avg_keyword_coverage": f"{avg_keywords:.1f}%",
        }

        print(f"\n{'='*60}")
        print("评估报告")
        print(f"{'='*60}")
        for k, v in report.items():
            print(f"  {k}: {v}")

        return report


# 定义测试集
BENCHMARK_TESTS = [
    TestCase(
        question="Python的创始人是谁？",
        expected_tools=["Search"],
        expected_keywords=["Guido", "van Rossum", "1991"],
        max_steps=3
    ),
    TestCase(
        question="计算地球周长，半径为6371公里",
        expected_tools=["Search", "Calculate"],
        expected_keywords=["6371", "40000", "周长"],
        max_steps=4
    ),
    TestCase(
        question="什么是ReAct模式？是哪年提出的？",
        expected_tools=["Search", "Lookup"],
        expected_keywords=["Reasoning", "Acting", "2022"],
        max_steps=4
    ),
]

# 使用评估器
# evaluator = ReActEvaluator(agent_func=agent.run)
# report = evaluator.evaluate(BENCHMARK_TESTS)
```

---

## 总结

本教程深入讲解了ReAct模式的核心内容：

1. **ReAct原理**: 将推理（Reasoning）和行动（Acting）交替进行，形成相互增强的循环，解决了CoT幻觉和Act-only盲目的问题
2. **Thought-Action-Observation循环**: 三步循环构成ReAct的核心，每步都有明确职责
3. **从零实现**: 纯Python实现帮助理解底层原理，包括prompt构建、输出解析和工具执行
4. **LangChain集成**: 利用LangChain框架可快速构建生产级ReAct Agent
5. **Web搜索Agent**: 集成Tavily/DuckDuckGo等搜索引擎构建研究型Agent
6. **对比实验**: ReAct在准确性和可解释性上优于CoT和Act-only
7. **调试技巧**: 追踪工具调用、检查prompt格式、错误恢复、性能监控

## 最佳实践

1. **精心设计工具描述**: 清晰、具体的工具描述是Agent正确选择工具的关键
2. **控制Observation长度**: 过长的工具返回会消耗大量token，建议截断或摘要
3. **设置合理的迭代上限**: max_iterations=5-10对大多数任务足够
4. **使用强模型**: ReAct需要较强的推理和指令遵循能力，推荐GPT-4o
5. **添加错误处理**: handle_parsing_errors=True可以自动处理格式错误
6. **监控和日志**: verbose=True在开发阶段非常有用，生产环境使用结构化日志

## 参考资源

- [ReAct论文](https://arxiv.org/abs/2210.03629) - 原始论文
- [LangChain ReAct文档](https://python.langchain.com/docs/modules/agents/agent_types/react)
- [Tavily搜索API](https://tavily.com/) - 专为AI Agent设计的搜索引擎
- [Chain-of-Thought论文](https://arxiv.org/abs/2201.11903) - CoT原始论文
- [LangSmith](https://smith.langchain.com/) - Agent调试和监控平台

---

**文件大小目标**: 25KB
**创建时间**: 2024-01-01
**最后更新**: 2024-01-01
