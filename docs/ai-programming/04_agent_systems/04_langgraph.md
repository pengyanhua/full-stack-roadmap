# LangGraph状态图

## 目录
1. [LangGraph简介](#langgraph简介)
2. [核心概念：StateGraph](#核心概念stategraph)
3. [State定义](#state定义)
4. [Node函数编写](#node函数编写)
5. [条件路由](#条件路由)
6. [循环与递归控制](#循环与递归控制)
7. [Human-in-the-Loop](#human-in-the-loop)
8. [Checkpointing断点续传](#checkpointing断点续传)
9. [完整实战：智能客服Agent](#完整实战智能客服agent)

---

## LangGraph简介

### 什么是LangGraph？

LangGraph是LangChain团队开发的一个用于构建**有状态、多步骤**AI Agent应用的框架。它基于**图（Graph）**的概念，将Agent的执行流程建模为一个状态图，其中节点（Node）执行操作，边（Edge）决定流转方向。

```
┌──────────────────────────────────────────────────────────────────┐
│                   LangGraph 核心概念                              │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │                    StateGraph                            │     │
│  │                                                         │     │
│  │   ┌──────────┐                                          │     │
│  │   │  START   │                                          │     │
│  │   └────┬─────┘                                          │     │
│  │        │                                                │     │
│  │        ▼                                                │     │
│  │   ┌──────────┐     Edge          ┌──────────┐          │     │
│  │   │  Node A  │──────────────────►│  Node B  │          │     │
│  │   │ (函数)   │                   │ (函数)   │          │     │
│  │   └──────────┘                   └────┬─────┘          │     │
│  │                                       │                │     │
│  │                          Conditional  │                │     │
│  │                              Edge     │                │     │
│  │                                       │                │     │
│  │                              ┌────────┴────────┐       │     │
│  │                              │   条件判断函数   │       │     │
│  │                              └───┬─────────┬───┘       │     │
│  │                                  │         │           │     │
│  │                           ┌──────┴─┐   ┌──┴──────┐    │     │
│  │                           │ Node C │   │ Node D  │    │     │
│  │                           └────┬───┘   └────┬────┘    │     │
│  │                                │            │         │     │
│  │                                ▼            ▼         │     │
│  │                           ┌──────────┐                │     │
│  │                           │   END    │                │     │
│  │                           └──────────┘                │     │
│  │                                                         │     │
│  │  State: 贯穿整个图的共享状态对象                        │     │
│  │  ┌─────────────────────────────────────────────────┐    │     │
│  │  │ {"messages": [...], "next_step": "...", ...}     │    │     │
│  │  └─────────────────────────────────────────────────┘    │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                  │
│  核心三要素:                                                     │
│  1. State  - 图中流转的状态数据                                  │
│  2. Node   - 执行具体操作的函数                                  │
│  3. Edge   - 连接节点的边（普通边/条件边）                        │
└──────────────────────────────────────────────────────────────────┘
```

### LangGraph vs LangChain Agent

| 特性 | LangChain Agent | LangGraph |
|------|----------------|-----------|
| **流程控制** | 隐式（由LLM决定） | 显式（图结构定义） |
| **状态管理** | 基础（消息历史） | 强大（自定义State） |
| **可视化** | 难以可视化 | 图结构可直接可视化 |
| **循环控制** | max_iterations | 精确的循环条件 |
| **人机协作** | 不原生支持 | 内置interrupt支持 |
| **持久化** | 需要额外实现 | 内置Checkpointing |
| **适用场景** | 简单Agent | 复杂工作流/多Agent |

### 环境准备

```bash
pip install langgraph langchain-openai langchain-core python-dotenv
```

---

## 核心概念：StateGraph

### 最简单的LangGraph示例

```python
"""
LangGraph 最简示例：Hello World
"""
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ============================================================
# 1. 定义State（状态）
# ============================================================
class State(TypedDict):
    """图的状态定义"""
    messages: Annotated[list, add_messages]  # 消息列表（自动追加）
    step_count: int                          # 步骤计数


# ============================================================
# 2. 定义Node（节点函数）
# ============================================================
def greet(state: State) -> dict:
    """问候节点"""
    return {
        "messages": [{"role": "assistant", "content": "你好！我是AI助手。"}],
        "step_count": state.get("step_count", 0) + 1
    }

def process(state: State) -> dict:
    """处理节点"""
    last_message = state["messages"][-1]
    return {
        "messages": [{"role": "assistant", "content": f"已处理你的请求: {last_message}"}],
        "step_count": state.get("step_count", 0) + 1
    }


# ============================================================
# 3. 构建图
# ============================================================
graph = StateGraph(State)

# 添加节点
graph.add_node("greet", greet)
graph.add_node("process", process)

# 添加边
graph.add_edge(START, "greet")       # 开始 → 问候
graph.add_edge("greet", "process")    # 问候 → 处理
graph.add_edge("process", END)        # 处理 → 结束

# 编译图
app = graph.compile()


# ============================================================
# 4. 运行
# ============================================================
result = app.invoke({
    "messages": [{"role": "user", "content": "你好"}],
    "step_count": 0
})

print(f"最终消息: {result['messages']}")
print(f"总步骤数: {result['step_count']}")

# 可视化图结构（需要安装graphviz）
# print(app.get_graph().draw_mermaid())
```

---

## State定义

### 使用TypedDict

```python
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages


# ============================================================
# 方式1: 基础TypedDict
# ============================================================
class BasicState(TypedDict):
    """基础状态"""
    messages: Annotated[list, add_messages]  # 消息自动追加
    user_input: str                          # 当前用户输入
    response: str                            # 当前回复


# ============================================================
# 方式2: 带Reducer的State
# ============================================================
def merge_dicts(existing: dict, new: dict) -> dict:
    """自定义reducer：合并字典"""
    return {**existing, **new}

class AdvancedState(TypedDict):
    messages: Annotated[list, add_messages]
    metadata: Annotated[dict, merge_dicts]      # 使用自定义reducer
    current_step: str
    error_count: int
    is_complete: bool
```

### 使用Pydantic模型

```python
from pydantic import BaseModel, Field
from typing import Literal


class CustomerServiceState(BaseModel):
    """客服系统状态（Pydantic版）"""
    messages: list = Field(default_factory=list, description="对话消息列表")
    intent: Optional[str] = Field(None, description="识别的用户意图")
    sentiment: Optional[str] = Field(None, description="情感分析结果")
    category: Optional[str] = Field(None, description="问题分类")
    resolution: Optional[str] = Field(None, description="解决方案")
    escalated: bool = Field(False, description="是否需要升级处理")
    satisfaction_score: Optional[int] = Field(None, description="满意度评分(1-5)")

    class Config:
        arbitrary_types_allowed = True
```

---

## Node函数编写

### 各类Node示例

```python
"""
LangGraph Node函数编写指南
"""
import os
import json
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    context: dict
    next_action: str


# ============================================================
# Node 1: LLM调用节点
# ============================================================
def chatbot_node(state: AgentState) -> dict:
    """基础聊天节点"""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# ============================================================
# Node 2: 意图识别节点
# ============================================================
def intent_classifier(state: AgentState) -> dict:
    """意图分类节点"""
    last_message = state["messages"][-1].content

    classification_prompt = f"""请分析以下用户消息的意图，返回以下类别之一:
- question: 咨询问题
- complaint: 投诉
- order: 订单相关
- general: 一般对话

用户消息: {last_message}

只返回类别名称，不要其他内容。"""

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    intent = response.content.strip().lower()

    return {"intent": intent}


# ============================================================
# Node 3: 工具调用节点
# ============================================================
def tool_executor(state: AgentState) -> dict:
    """工具执行节点"""
    # 根据上下文决定执行什么工具
    intent = state.get("intent", "general")

    if intent == "order":
        result = "订单查询结果: 订单#12345 状态为已发货"
    elif intent == "complaint":
        result = "已记录投诉，工单号: TK-98765"
    else:
        result = "已查询到相关信息"

    return {
        "messages": [AIMessage(content=result)],
        "context": {"tool_result": result}
    }


# ============================================================
# Node 4: 回复生成节点
# ============================================================
def response_generator(state: AgentState) -> dict:
    """生成最终回复"""
    context = state.get("context", {})
    intent = state.get("intent", "general")

    system_msg = SystemMessage(content=f"""你是一个专业的客服助手。
当前用户意图: {intent}
工具查询结果: {json.dumps(context, ensure_ascii=False)}
请基于以上信息生成友好的回复。""")

    messages = [system_msg] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# ============================================================
# 构建包含多种Node的图
# ============================================================
graph = StateGraph(AgentState)

graph.add_node("classify", intent_classifier)
graph.add_node("tool", tool_executor)
graph.add_node("respond", response_generator)

graph.add_edge(START, "classify")
graph.add_edge("classify", "tool")
graph.add_edge("tool", "respond")
graph.add_edge("respond", END)

app = graph.compile()

# 运行
result = app.invoke({
    "messages": [HumanMessage(content="我的订单12345到哪了？")],
    "intent": "",
    "context": {},
    "next_action": ""
})

for msg in result["messages"]:
    print(f"[{msg.type}] {msg.content}")
```

---

## 条件路由

### Conditional Edge详解

```
┌──────────────────────────────────────────────────────────────┐
│                    条件路由示意图                              │
│                                                              │
│                    ┌──────────┐                              │
│                    │  START   │                              │
│                    └────┬─────┘                              │
│                         │                                    │
│                         ▼                                    │
│                    ┌──────────┐                              │
│                    │  分类器   │                              │
│                    └────┬─────┘                              │
│                         │                                    │
│                  ┌──────┴──────┐                             │
│                  │ router函数  │  ← 条件路由函数              │
│                  └──┬──┬──┬───┘                             │
│                     │  │  │                                  │
│          "question" │  │  │ "complaint"                      │
│                     │  │  │                                  │
│                     ▼  │  ▼                                  │
│              ┌──────┐  │  ┌──────────┐                      │
│              │ 问答  │  │  │ 投诉处理  │                      │
│              │ 处理  │  │  │          │                      │
│              └──┬───┘  │  └────┬─────┘                      │
│                 │      │       │                             │
│                 │  "order"     │                             │
│                 │      │       │                             │
│                 │      ▼       │                             │
│                 │ ┌──────────┐ │                             │
│                 │ │ 订单查询  │ │                             │
│                 │ └────┬─────┘ │                             │
│                 │      │       │                             │
│                 ▼      ▼       ▼                             │
│                    ┌──────────┐                              │
│                    │  回复    │                              │
│                    └────┬─────┘                              │
│                         │                                    │
│                         ▼                                    │
│                    ┌──────────┐                              │
│                    │   END    │                              │
│                    └──────────┘                              │
└──────────────────────────────────────────────────────────────┘
```

```python
"""
LangGraph 条件路由完整示例
"""
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)


class RouterState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    category: str


# ============================================================
# 节点函数
# ============================================================
def classify_intent(state: RouterState) -> dict:
    """意图分类"""
    user_msg = state["messages"][-1].content
    prompt = f"""分析用户意图，返回以下之一: question, complaint, order, general
用户: {user_msg}
意图:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"intent": response.content.strip().lower()}

def handle_question(state: RouterState) -> dict:
    """处理问题"""
    response = llm.invoke([
        HumanMessage(content=f"作为客服回答问题: {state['messages'][-1].content}")
    ])
    return {"messages": [response]}

def handle_complaint(state: RouterState) -> dict:
    """处理投诉"""
    return {
        "messages": [AIMessage(content="非常抱歉给您带来不便，我已记录您的投诉，会尽快处理。工单号：TK-001")],
        "category": "complaint"
    }

def handle_order(state: RouterState) -> dict:
    """处理订单"""
    return {
        "messages": [AIMessage(content="正在为您查询订单信息，请稍候...")],
        "category": "order"
    }

def handle_general(state: RouterState) -> dict:
    """处理一般对话"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def generate_final_response(state: RouterState) -> dict:
    """生成最终回复"""
    return {"messages": [AIMessage(content="还有什么我可以帮您的吗？")]}


# ============================================================
# 路由函数
# ============================================================
def route_by_intent(state: RouterState) -> Literal["question", "complaint", "order", "general"]:
    """根据意图路由到不同处理节点"""
    intent = state.get("intent", "general")
    if intent in ["question", "complaint", "order"]:
        return intent
    return "general"


# ============================================================
# 构建图
# ============================================================
graph = StateGraph(RouterState)

# 添加节点
graph.add_node("classify", classify_intent)
graph.add_node("question", handle_question)
graph.add_node("complaint", handle_complaint)
graph.add_node("order", handle_order)
graph.add_node("general", handle_general)
graph.add_node("final", generate_final_response)

# 添加边
graph.add_edge(START, "classify")

# 条件路由
graph.add_conditional_edges(
    "classify",          # 源节点
    route_by_intent,     # 路由函数
    {                    # 路由映射
        "question": "question",
        "complaint": "complaint",
        "order": "order",
        "general": "general"
    }
)

# 所有处理节点 → 最终回复
graph.add_edge("question", "final")
graph.add_edge("complaint", "final")
graph.add_edge("order", "final")
graph.add_edge("general", "final")
graph.add_edge("final", END)

# 编译
app = graph.compile()

# 测试
test_cases = [
    "我的订单12345什么时候到？",
    "你们的服务太差了！",
    "Python和Java哪个更好？",
    "今天天气不错"
]

for msg in test_cases:
    print(f"\n{'='*50}")
    print(f"用户: {msg}")
    result = app.invoke({
        "messages": [HumanMessage(content=msg)],
        "intent": "",
        "category": ""
    })
    for m in result["messages"]:
        if hasattr(m, 'content'):
            print(f"[{m.type}] {m.content}")
```

---

## 循环与递归控制

### Agent循环模式

```python
"""
LangGraph 循环控制 - 实现Agent的思考-行动循环
"""
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ============================================================
# 定义工具
# ============================================================
@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    import math
    safe = {"sqrt": math.sqrt, "pi": math.pi, "abs": abs, "pow": pow}
    try:
        return str(eval(expression, {"__builtins__": {}}, safe))
    except Exception as e:
        return f"Error: {e}"

@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"关于'{query}'的搜索结果: 模拟数据"

tools_list = [calculator, search]
llm_with_tools = llm.bind_tools(tools_list)


# ============================================================
# State定义
# ============================================================
class LoopState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration: int
    max_iterations: int


# ============================================================
# 节点函数
# ============================================================
def agent_think(state: LoopState) -> dict:
    """Agent思考节点：调用LLM决定行动"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "iteration": state.get("iteration", 0) + 1
    }

def execute_tools(state: LoopState) -> dict:
    """工具执行节点"""
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # 找到并执行对应工具
        for t in tools_list:
            if t.name == tool_name:
                result = t.invoke(tool_args)
                tool_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )
                break

    return {"messages": tool_messages}


# ============================================================
# 路由函数：决定继续循环还是结束
# ============================================================
def should_continue(state: LoopState) -> Literal["tools", "end"]:
    """判断是否继续循环"""
    last_message = state["messages"][-1]
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 10)

    # 达到最大迭代次数，强制结束
    if iteration >= max_iter:
        return "end"

    # 如果LLM要调用工具，继续循环
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # 否则结束
    return "end"


# ============================================================
# 构建带循环的图
# ============================================================
graph = StateGraph(LoopState)

graph.add_node("agent", agent_think)
graph.add_node("tools", execute_tools)

# START → agent
graph.add_edge(START, "agent")

# agent → 条件判断（工具调用 or 结束）
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# tools → agent（形成循环！）
graph.add_edge("tools", "agent")

# 编译
app = graph.compile()


# ============================================================
# 运行
# ============================================================
result = app.invoke({
    "messages": [HumanMessage(content="计算 (25 * 4 + 30) / 5，然后搜索Python最新版本")],
    "iteration": 0,
    "max_iterations": 10
})

print(f"\n总迭代次数: {result['iteration']}")
for msg in result["messages"]:
    print(f"[{msg.type}] {msg.content[:100]}...")
```

---

## Human-in-the-Loop

### 人机协作模式

```
┌──────────────────────────────────────────────────────────────┐
│              Human-in-the-Loop 流程                           │
│                                                              │
│  ┌──────────┐                                                │
│  │  Agent    │                                                │
│  │  处理请求 │                                                │
│  └────┬─────┘                                                │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────┐    需要人工确认？                               │
│  │  判断    │─── 否 ──────────────► 继续自动执行             │
│  └────┬─────┘                                                │
│       │ 是                                                   │
│       ▼                                                      │
│  ┌──────────┐                                                │
│  │ INTERRUPT │  ◄── 暂停执行                                 │
│  │ (中断)    │                                                │
│  └────┬─────┘                                                │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────┐                                                │
│  │ 人工审核  │  ◄── 用户查看并决策                            │
│  │          │      - 批准                                    │
│  │          │      - 修改                                    │
│  │          │      - 拒绝                                    │
│  └────┬─────┘                                                │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────┐                                                │
│  │ 继续执行  │  ◄── 恢复Agent执行                            │
│  └──────────┘                                                │
└──────────────────────────────────────────────────────────────┘
```

```python
"""
LangGraph Human-in-the-Loop 实现
"""
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)


class HITLState(TypedDict):
    messages: Annotated[list, add_messages]
    proposed_action: str
    human_approved: bool
    action_result: str


def propose_action(state: HITLState) -> dict:
    """Agent提出行动方案"""
    user_msg = state["messages"][-1].content
    response = llm.invoke([
        HumanMessage(content=f"用户请求: {user_msg}\n请提出你的行动方案（不要执行）:")
    ])
    return {
        "proposed_action": response.content,
        "messages": [AIMessage(content=f"我建议: {response.content}\n等待您的确认...")]
    }

def human_review(state: HITLState) -> dict:
    """人工审核节点 - 使用interrupt实现"""
    # 这个节点会被interrupt，等待人工输入
    return state

def execute_action(state: HITLState) -> dict:
    """执行已批准的行动"""
    action = state["proposed_action"]
    return {
        "messages": [AIMessage(content=f"已执行: {action}")],
        "action_result": f"执行成功: {action}"
    }

def reject_action(state: HITLState) -> dict:
    """拒绝行动"""
    return {
        "messages": [AIMessage(content="好的，已取消该操作。请告诉我您想怎么做。")],
        "action_result": "已取消"
    }

def check_approval(state: HITLState) -> Literal["execute", "reject"]:
    """检查人工审批结果"""
    if state.get("human_approved", False):
        return "execute"
    return "reject"


# 构建图
graph = StateGraph(HITLState)

graph.add_node("propose", propose_action)
graph.add_node("review", human_review)
graph.add_node("execute", execute_action)
graph.add_node("reject", reject_action)

graph.add_edge(START, "propose")
graph.add_edge("propose", "review")
graph.add_conditional_edges("review", check_approval, {"execute": "execute", "reject": "reject"})
graph.add_edge("execute", END)
graph.add_edge("reject", END)

# 使用Checkpointer编译（支持中断和恢复）
memory = MemorySaver()
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["review"]  # 在review节点前中断
)


# ============================================================
# 使用：中断和恢复
# ============================================================
config = {"configurable": {"thread_id": "session_001"}}

# 第一次调用 - 会在review前中断
result = app.invoke(
    {
        "messages": [HumanMessage(content="帮我删除所有过期数据")],
        "proposed_action": "",
        "human_approved": False,
        "action_result": ""
    },
    config
)

print("Agent方案:", result["proposed_action"])
print("等待人工确认...")

# 模拟人工审批
# 方式1：批准
app.update_state(config, {"human_approved": True})
result = app.invoke(None, config)  # 继续执行
print("执行结果:", result["action_result"])

# 方式2：拒绝（取消注释使用）
# app.update_state(config, {"human_approved": False})
# result = app.invoke(None, config)
# print("结果:", result["action_result"])
```

---

## Checkpointing断点续传

### 持久化状态管理

```python
"""
LangGraph Checkpointing - 断点续传
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3


# ============================================================
# 1. 内存Checkpointer（开发用）
# ============================================================
memory_checkpointer = MemorySaver()

# 在编译时使用
# app = graph.compile(checkpointer=memory_checkpointer)


# ============================================================
# 2. SQLite Checkpointer（生产用）
# ============================================================
# 方式A: 内存SQLite
sqlite_memory = SqliteSaver.from_conn_string(":memory:")

# 方式B: 文件SQLite（持久化）
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
sqlite_file = SqliteSaver(conn)

# 在编译时使用
# app = graph.compile(checkpointer=sqlite_file)


# ============================================================
# 3. 使用Checkpoint恢复会话
# ============================================================
def demo_checkpoint_recovery():
    """演示断点续传"""
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict, Annotated
    from langgraph.graph.message import add_messages
    from langchain_core.messages import HumanMessage, AIMessage

    class ChatState(TypedDict):
        messages: Annotated[list, add_messages]
        turn_count: int

    def chat_node(state: ChatState) -> dict:
        turn = state.get("turn_count", 0) + 1
        return {
            "messages": [AIMessage(content=f"这是第{turn}轮回复")],
            "turn_count": turn
        }

    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    # 会话1
    config1 = {"configurable": {"thread_id": "user_alice"}}
    result = app.invoke({"messages": [HumanMessage(content="你好")], "turn_count": 0}, config1)
    print(f"Alice第1轮: {result['turn_count']}")

    result = app.invoke({"messages": [HumanMessage(content="今天天气")], "turn_count": result["turn_count"]}, config1)
    print(f"Alice第2轮: {result['turn_count']}")

    # 会话2（独立）
    config2 = {"configurable": {"thread_id": "user_bob"}}
    result = app.invoke({"messages": [HumanMessage(content="嗨")], "turn_count": 0}, config2)
    print(f"Bob第1轮: {result['turn_count']}")

    # 获取历史状态
    history = list(app.get_state_history(config1))
    print(f"\nAlice历史状态数: {len(history)}")
    for h in history:
        print(f"  Turn: {h.values.get('turn_count')}, Messages: {len(h.values.get('messages', []))}")

demo_checkpoint_recovery()
```

---

## 完整实战：智能客服Agent

### 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    智能客服Agent架构                               │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  用户输入                                               │      │
│  └───────────────────────┬────────────────────────────────┘      │
│                          │                                       │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────┐        │
│  │              意图识别 (Intent Classifier)              │        │
│  └───────────────────────┬──────────────────────────────┘        │
│                          │                                       │
│           ┌──────────────┼──────────────┐                        │
│           │              │              │                        │
│           ▼              ▼              ▼                        │
│  ┌──────────────┐ ┌──────────┐ ┌──────────────┐                │
│  │  FAQ问答专家  │ │ 订单专家  │ │  投诉处理专家 │                │
│  │              │ │          │ │              │                │
│  │ • 检索知识库  │ │ • 查订单  │ │ • 记录投诉   │                │
│  │ • 生成回答   │ │ • 修改订单 │ │ • 安抚用户   │                │
│  │              │ │ • 退款    │ │ • 升级处理   │                │
│  └──────┬───────┘ └────┬─────┘ └──────┬───────┘                │
│         │              │              │                        │
│         └──────────────┼──────────────┘                        │
│                        │                                       │
│                        ▼                                       │
│  ┌──────────────────────────────────────────────────────┐      │
│  │                  回复整合与输出                        │      │
│  └──────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘
```

### 完整代码实现

```python
"""
完整实战：基于LangGraph的智能客服Agent
功能：意图识别 → 路由 → 专家处理 → 回复生成
"""
import os
import json
from typing import TypedDict, Annotated, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ============================================================
# 1. State定义
# ============================================================
class CustomerServiceState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str                      # 用户意图
    sub_intent: str                  # 细分意图
    customer_id: str                 # 客户ID
    sentiment: str                   # 情感（positive/neutral/negative）
    context: dict                    # 上下文信息
    expert_response: str             # 专家回复
    needs_escalation: bool           # 是否需要升级
    resolution_status: str           # 解决状态


# ============================================================
# 2. 模拟数据库
# ============================================================
ORDER_DB = {
    "12345": {"status": "已发货", "product": "iPhone 15", "est_delivery": "2024-01-20", "price": 6999},
    "12346": {"status": "待发货", "product": "MacBook Pro", "est_delivery": "2024-01-25", "price": 14999},
    "12347": {"status": "已签收", "product": "AirPods Pro", "est_delivery": "2024-01-15", "price": 1899},
}

FAQ_DB = {
    "退货政策": "自签收之日起7天内可无理由退货，15天内可换货。",
    "运费": "订单满99元免运费，不满99元收取8元运费。",
    "支付方式": "支持支付宝、微信支付、银行卡、信用卡等多种支付方式。",
    "配送时间": "一般3-5个工作日送达，偏远地区5-7个工作日。",
    "会员权益": "会员享受9.5折优惠、免运费、专属客服等权益。",
}


# ============================================================
# 3. 节点函数
# ============================================================
def intent_classifier(state: CustomerServiceState) -> dict:
    """意图识别节点"""
    last_msg = state["messages"][-1].content

    prompt = f"""分析用户消息，返回JSON格式:
{{
    "intent": "faq|order|complaint|general",
    "sub_intent": "具体子意图",
    "sentiment": "positive|neutral|negative"
}}

用户消息: {last_msg}"""

    response = llm_fast.invoke([HumanMessage(content=prompt)])
    try:
        result = json.loads(response.content)
    except:
        result = {"intent": "general", "sub_intent": "chat", "sentiment": "neutral"}

    return {
        "intent": result.get("intent", "general"),
        "sub_intent": result.get("sub_intent", ""),
        "sentiment": result.get("sentiment", "neutral")
    }


def faq_expert(state: CustomerServiceState) -> dict:
    """FAQ问答专家"""
    user_msg = state["messages"][-1].content

    # 简单关键词匹配FAQ
    matched_faq = []
    for key, answer in FAQ_DB.items():
        if any(kw in user_msg for kw in key):
            matched_faq.append(f"**{key}**: {answer}")

    if matched_faq:
        context = "\n".join(matched_faq)
    else:
        context = "未找到完全匹配的FAQ条目"

    prompt = f"""你是FAQ客服专家。根据以下FAQ知识库回答用户问题。

FAQ匹配结果:
{context}

用户问题: {user_msg}

请用友好专业的语气回答。如果FAQ中没有答案，请如实告知并建议转人工。"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "expert_response": response.content,
        "context": {"faq_matched": len(matched_faq)},
        "resolution_status": "resolved" if matched_faq else "needs_human"
    }


def order_expert(state: CustomerServiceState) -> dict:
    """订单处理专家"""
    user_msg = state["messages"][-1].content

    # 提取订单号
    import re
    order_match = re.search(r'\d{5,}', user_msg)
    order_id = order_match.group() if order_match else None

    if order_id and order_id in ORDER_DB:
        order = ORDER_DB[order_id]
        order_info = json.dumps(order, ensure_ascii=False)
        prompt = f"""你是订单客服专家。根据订单信息回答用户。

订单号: {order_id}
订单信息: {order_info}

用户问题: {user_msg}

请提供准确的订单状态信息，语气友好专业。"""
    else:
        prompt = f"""用户询问订单相关问题，但未提供有效订单号或订单不存在。
用户消息: {user_msg}
请礼貌地请用户提供正确的订单号。"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "expert_response": response.content,
        "context": {"order_id": order_id, "order_found": order_id in ORDER_DB if order_id else False},
        "resolution_status": "resolved" if order_id in ORDER_DB else "pending"
    }


def complaint_expert(state: CustomerServiceState) -> dict:
    """投诉处理专家"""
    user_msg = state["messages"][-1].content
    sentiment = state.get("sentiment", "neutral")

    # 严重投诉需要升级
    needs_escalation = sentiment == "negative" and any(
        kw in user_msg for kw in ["投诉", "举报", "律师", "消费者协会", "退款"]
    )

    prompt = f"""你是投诉处理专家。用户情绪: {sentiment}

处理原则:
1. 先表示理解和歉意
2. 确认用户的问题
3. 提供解决方案
4. {"这是严重投诉，需要升级处理" if needs_escalation else "尝试在线解决"}

用户消息: {user_msg}

请用真诚、专业的语气回复。"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "expert_response": response.content,
        "needs_escalation": needs_escalation,
        "context": {"complaint_severity": "high" if needs_escalation else "normal"},
        "resolution_status": "escalated" if needs_escalation else "in_progress"
    }


def general_chat(state: CustomerServiceState) -> dict:
    """一般对话"""
    response = llm.invoke([
        SystemMessage(content="你是一个友好的客服助手。用中文回答。"),
        *state["messages"]
    ])
    return {
        "expert_response": response.content,
        "resolution_status": "resolved"
    }


def response_synthesizer(state: CustomerServiceState) -> dict:
    """回复整合节点 - 将专家回复包装成最终回复"""
    expert_resp = state.get("expert_response", "")
    needs_escalation = state.get("needs_escalation", False)

    suffix = ""
    if needs_escalation:
        suffix = "\n\n---\n*[系统提示: 此问题已升级至人工客服处理]*"

    return {
        "messages": [AIMessage(content=expert_resp + suffix)]
    }


# ============================================================
# 4. 路由函数
# ============================================================
def route_to_expert(state: CustomerServiceState) -> Literal["faq", "order", "complaint", "general"]:
    """根据意图路由到对应专家"""
    intent = state.get("intent", "general")
    if intent in ["faq", "order", "complaint"]:
        return intent
    return "general"


# ============================================================
# 5. 构建完整图
# ============================================================
graph = StateGraph(CustomerServiceState)

# 添加节点
graph.add_node("classify", intent_classifier)
graph.add_node("faq", faq_expert)
graph.add_node("order", order_expert)
graph.add_node("complaint", complaint_expert)
graph.add_node("general", general_chat)
graph.add_node("synthesize", response_synthesizer)

# 添加边
graph.add_edge(START, "classify")

graph.add_conditional_edges(
    "classify",
    route_to_expert,
    {"faq": "faq", "order": "order", "complaint": "complaint", "general": "general"}
)

graph.add_edge("faq", "synthesize")
graph.add_edge("order", "synthesize")
graph.add_edge("complaint", "synthesize")
graph.add_edge("general", "synthesize")
graph.add_edge("synthesize", END)

# 编译（带Checkpoint支持多轮对话）
memory = MemorySaver()
customer_service_app = graph.compile(checkpointer=memory)


# ============================================================
# 6. 运行客服系统
# ============================================================
def run_customer_service():
    """运行智能客服"""
    print("=" * 60)
    print("    智能客服系统 (基于LangGraph)")
    print("    输入 'quit' 退出")
    print("=" * 60)

    config = {"configurable": {"thread_id": "customer_001"}}

    while True:
        user_input = input("\n客户: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("感谢您的咨询，再见！")
            break
        if not user_input:
            continue

        result = customer_service_app.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "intent": "", "sub_intent": "", "customer_id": "C001",
                "sentiment": "", "context": {}, "expert_response": "",
                "needs_escalation": False, "resolution_status": ""
            },
            config
        )

        # 显示回复
        last_ai_msg = [m for m in result["messages"] if isinstance(m, AIMessage)]
        if last_ai_msg:
            print(f"\n客服: {last_ai_msg[-1].content}")

        # 显示调试信息
        print(f"  [DEBUG] 意图: {result['intent']} | 情感: {result['sentiment']} | 状态: {result['resolution_status']}")


def run_demo():
    """演示模式"""
    test_messages = [
        "你们的退货政策是什么？",
        "我的订单12345到哪了？",
        "你们的服务太差了，我要投诉！",
        "今天天气真不错",
        "订单12346什么时候发货？",
    ]

    config = {"configurable": {"thread_id": "demo"}}

    for msg in test_messages:
        print(f"\n{'='*60}")
        print(f"客户: {msg}")

        result = customer_service_app.invoke(
            {
                "messages": [HumanMessage(content=msg)],
                "intent": "", "sub_intent": "", "customer_id": "demo",
                "sentiment": "", "context": {}, "expert_response": "",
                "needs_escalation": False, "resolution_status": ""
            },
            config
        )

        last_ai = [m for m in result["messages"] if isinstance(m, AIMessage)]
        if last_ai:
            print(f"客服: {last_ai[-1].content}")
        print(f"[意图: {result['intent']} | 情感: {result['sentiment']}]")


if __name__ == "__main__":
    run_demo()
    # run_customer_service()  # 交互模式
```

---

## 流式输出与SubGraph

### LangGraph流式输出

```python
"""
LangGraph 流式输出：让用户实时看到Agent的思考过程
"""
import os
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)


class StreamState(TypedDict):
    messages: Annotated[list, add_messages]
    current_step: str


def thinking_node(state: StreamState) -> dict:
    """思考节点"""
    return {
        "messages": [AIMessage(content="让我来思考这个问题...")],
        "current_step": "thinking"
    }

def answer_node(state: StreamState) -> dict:
    """回答节点"""
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "current_step": "answering"
    }


# 构建图
graph = StateGraph(StreamState)
graph.add_node("think", thinking_node)
graph.add_node("answer", answer_node)
graph.add_edge(START, "think")
graph.add_edge("think", "answer")
graph.add_edge("answer", END)

app = graph.compile()


# ============================================================
# 流式输出方式1: stream_mode="values" - 输出每个步骤的完整状态
# ============================================================
def demo_stream_values():
    """值流模式：每步输出完整状态"""
    print("\n=== stream_mode='values' ===")
    input_state = {
        "messages": [HumanMessage(content="用三句话解释什么是LangGraph")],
        "current_step": ""
    }

    for state in app.stream(input_state, stream_mode="values"):
        step = state.get("current_step", "")
        msgs = state.get("messages", [])
        if msgs:
            last = msgs[-1]
            print(f"[{step}] [{last.type}] {last.content[:100]}")


# ============================================================
# 流式输出方式2: stream_mode="updates" - 仅输出状态变更
# ============================================================
def demo_stream_updates():
    """更新流模式：仅输出每步的变更"""
    print("\n=== stream_mode='updates' ===")
    input_state = {
        "messages": [HumanMessage(content="什么是StateGraph？")],
        "current_step": ""
    }

    for update in app.stream(input_state, stream_mode="updates"):
        for node_name, node_output in update.items():
            print(f"[Node: {node_name}]")
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    content = msg.content if hasattr(msg, 'content') else str(msg)
                    print(f"  {content[:150]}")


# ============================================================
# 流式输出方式3: astream_events - LLM token级别流式
# ============================================================
async def demo_stream_events():
    """事件流模式：获取LLM的逐token输出"""
    import asyncio

    input_state = {
        "messages": [HumanMessage(content="解释LangGraph的条件路由")],
        "current_step": ""
    }

    async for event in app.astream_events(input_state, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            # LLM逐token输出
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)
        elif kind == "on_chain_start":
            name = event.get("name", "")
            if name:
                print(f"\n--- 开始: {name} ---")


if __name__ == "__main__":
    demo_stream_values()
    demo_stream_updates()
    # asyncio.run(demo_stream_events())
```

### SubGraph子图嵌套

```
┌──────────────────────────────────────────────────────────────┐
│                   SubGraph 嵌套架构                            │
│                                                              │
│  ┌─────────────────── 主图 ──────────────────────────┐       │
│  │                                                   │       │
│  │  START → [路由器] ──┬── [子图A: 客服流程] ──┐      │       │
│  │                     │                       │      │       │
│  │                     ├── [子图B: 订单流程] ──┤      │       │
│  │                     │                       │      │       │
│  │                     └── [子图C: 技术支持] ──┘      │       │
│  │                                     │              │       │
│  │                                     ▼              │       │
│  │                              [合并节点] → END      │       │
│  └─────────────────────────────────────────────────────┘      │
│                                                              │
│  ┌──── 子图A详情 ────┐  ┌──── 子图B详情 ────┐               │
│  │                   │  │                   │               │
│  │  意图分析         │  │  订单查询         │               │
│  │     │             │  │     │             │               │
│  │     ▼             │  │     ▼             │               │
│  │  知识检索         │  │  状态检查         │               │
│  │     │             │  │     │             │               │
│  │     ▼             │  │     ▼             │               │
│  │  生成回复         │  │  生成回复         │               │
│  │                   │  │                   │               │
│  └───────────────────┘  └───────────────────┘               │
└──────────────────────────────────────────────────────────────┘
```

```python
"""
LangGraph SubGraph：模块化的子图嵌套
"""
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ============================================================
# 1. 定义共享State
# ============================================================
class MainState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str
    result: str


# ============================================================
# 2. 构建子图A：FAQ处理
# ============================================================
class FAQState(TypedDict):
    messages: Annotated[list, add_messages]
    faq_answer: str

def faq_search(state: FAQState) -> dict:
    """FAQ搜索"""
    user_msg = state["messages"][-1].content
    # 模拟FAQ检索
    faq_data = {
        "退货": "7天内可退货，15天内可换货",
        "运费": "满99免运费",
        "支付": "支持微信、支付宝、银行卡",
    }
    answer = "未找到相关FAQ"
    for key, val in faq_data.items():
        if key in user_msg:
            answer = val
            break
    return {"faq_answer": answer}

def faq_respond(state: FAQState) -> dict:
    """生成FAQ回复"""
    return {
        "messages": [AIMessage(content=f"FAQ回复: {state['faq_answer']}")]
    }

# 构建FAQ子图
faq_graph = StateGraph(FAQState)
faq_graph.add_node("search", faq_search)
faq_graph.add_node("respond", faq_respond)
faq_graph.add_edge(START, "search")
faq_graph.add_edge("search", "respond")
faq_graph.add_edge("respond", END)
faq_subgraph = faq_graph.compile()


# ============================================================
# 3. 构建子图B：订单处理
# ============================================================
class OrderState(TypedDict):
    messages: Annotated[list, add_messages]
    order_info: str

def order_lookup(state: OrderState) -> dict:
    """查询订单"""
    import re
    user_msg = state["messages"][-1].content
    order_match = re.search(r"\d{5,}", user_msg)
    if order_match:
        order_id = order_match.group()
        return {"order_info": f"订单{order_id}: 已发货，预计明天到达"}
    return {"order_info": "请提供有效的订单号"}

def order_respond(state: OrderState) -> dict:
    """生成订单回复"""
    return {
        "messages": [AIMessage(content=state["order_info"])]
    }

order_graph = StateGraph(OrderState)
order_graph.add_node("lookup", order_lookup)
order_graph.add_node("respond", order_respond)
order_graph.add_edge(START, "lookup")
order_graph.add_edge("lookup", "respond")
order_graph.add_edge("respond", END)
order_subgraph = order_graph.compile()


# ============================================================
# 4. 构建主图（使用子图作为节点）
# ============================================================
def router_node(state: MainState) -> dict:
    """路由节点：决定走哪个子图"""
    user_msg = state["messages"][-1].content
    if any(kw in user_msg for kw in ["订单", "物流", "发货"]):
        return {"route": "order"}
    elif any(kw in user_msg for kw in ["退货", "运费", "支付", "会员"]):
        return {"route": "faq"}
    else:
        return {"route": "general"}

def general_handler(state: MainState) -> dict:
    """一般问题处理"""
    response = llm.invoke(state["messages"])
    return {"messages": [response], "result": "general_handled"}

def faq_handler(state: MainState) -> dict:
    """调用FAQ子图"""
    result = faq_subgraph.invoke({
        "messages": state["messages"],
        "faq_answer": ""
    })
    return {
        "messages": result["messages"],
        "result": "faq_handled"
    }

def order_handler(state: MainState) -> dict:
    """调用订单子图"""
    result = order_subgraph.invoke({
        "messages": state["messages"],
        "order_info": ""
    })
    return {
        "messages": result["messages"],
        "result": "order_handled"
    }

def route_decision(state: MainState) -> Literal["faq", "order", "general"]:
    return state.get("route", "general")


# 构建主图
main_graph = StateGraph(MainState)

main_graph.add_node("router", router_node)
main_graph.add_node("faq", faq_handler)
main_graph.add_node("order", order_handler)
main_graph.add_node("general", general_handler)

main_graph.add_edge(START, "router")
main_graph.add_conditional_edges(
    "router",
    route_decision,
    {"faq": "faq", "order": "order", "general": "general"}
)
main_graph.add_edge("faq", END)
main_graph.add_edge("order", END)
main_graph.add_edge("general", END)

main_app = main_graph.compile()


# ============================================================
# 5. 测试
# ============================================================
test_cases = [
    "你们的退货政策是什么？",
    "我的订单12345到哪了？",
    "今天天气真好",
]

for msg in test_cases:
    print(f"\n{'='*50}")
    print(f"用户: {msg}")
    result = main_app.invoke({
        "messages": [HumanMessage(content=msg)],
        "route": "",
        "result": ""
    })
    last_ai = [m for m in result["messages"] if isinstance(m, AIMessage)]
    if last_ai:
        print(f"回复: {last_ai[-1].content}")
    print(f"路由: {result['route']} | 处理: {result['result']}")
```

### LangGraph高级模式：Map-Reduce

```python
"""
LangGraph Map-Reduce模式：并行处理多个子任务后合并
适用场景：同时分析多个文档、并行搜索多个来源
"""
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class MapReduceState(TypedDict):
    messages: Annotated[list, add_messages]
    topics: list[str]         # 待分析的主题列表
    analyses: list[str]       # 各主题的分析结果
    final_report: str         # 最终合并报告


def splitter_node(state: MapReduceState) -> dict:
    """拆分节点：将问题分解为多个子主题"""
    user_msg = state["messages"][-1].content

    response = llm.invoke([
        HumanMessage(content=f"""将以下问题分解为3-5个子主题，每行一个主题:
问题: {user_msg}

只列出主题名，不要编号，不要其他内容:""")
    ])

    topics = [
        t.strip() for t in response.content.strip().split("\n")
        if t.strip()
    ]
    return {"topics": topics[:5]}


def map_analyze(state: MapReduceState) -> dict:
    """Map节点：并行分析每个主题"""
    analyses = []
    for topic in state["topics"]:
        response = llm.invoke([
            HumanMessage(content=f"请用100字简要分析以下主题: {topic}")
        ])
        analyses.append(f"【{topic}】\n{response.content}")
    return {"analyses": analyses}


def reduce_merge(state: MapReduceState) -> dict:
    """Reduce节点：合并所有分析结果"""
    all_analyses = "\n\n".join(state["analyses"])

    response = llm.invoke([
        HumanMessage(content=f"""请基于以下各主题的分析结果，
撰写一份综合性的摘要报告（200字以内）:

{all_analyses}""")
    ])

    return {
        "final_report": response.content,
        "messages": [AIMessage(content=response.content)]
    }


# 构建图
graph = StateGraph(MapReduceState)
graph.add_node("split", splitter_node)
graph.add_node("analyze", map_analyze)
graph.add_node("merge", reduce_merge)

graph.add_edge(START, "split")
graph.add_edge("split", "analyze")
graph.add_edge("analyze", "merge")
graph.add_edge("merge", END)

map_reduce_app = graph.compile()


# 测试
result = map_reduce_app.invoke({
    "messages": [HumanMessage(content="分析2024年AI技术的发展趋势")],
    "topics": [],
    "analyses": [],
    "final_report": ""
})

print(f"子主题: {result['topics']}")
print(f"\n最终报告:\n{result['final_report']}")
```

### LangGraph错误处理与重试

```python
"""
LangGraph 节点级错误处理
"""
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage


class RobustState(TypedDict):
    messages: Annotated[list, add_messages]
    error_count: int
    last_error: str
    max_retries: int


def safe_node(func):
    """节点安全装饰器：自动捕获异常"""
    def wrapper(state):
        try:
            return func(state)
        except Exception as e:
            error_count = state.get("error_count", 0) + 1
            return {
                "error_count": error_count,
                "last_error": str(e),
                "messages": [AIMessage(
                    content=f"[系统] 处理出错: {str(e)[:100]}"
                )]
            }
    wrapper.__name__ = func.__name__
    return wrapper


@safe_node
def risky_operation(state: RobustState) -> dict:
    """可能失败的操作"""
    import random
    if random.random() < 0.5:
        raise RuntimeError("模拟的随机错误")
    return {
        "messages": [AIMessage(content="操作成功完成！")],
        "error_count": 0,
        "last_error": ""
    }


def error_checker(state: RobustState) -> Literal["retry", "give_up", "success"]:
    """检查是否需要重试"""
    if state.get("last_error") and state.get("error_count", 0) < state.get("max_retries", 3):
        return "retry"
    elif state.get("last_error"):
        return "give_up"
    else:
        return "success"


def give_up_node(state: RobustState) -> dict:
    """放弃节点：达到最大重试次数"""
    return {
        "messages": [AIMessage(
            content=f"抱歉，在{state['error_count']}次尝试后仍无法完成操作。"
            f"最后错误: {state['last_error']}"
        )]
    }


# 构建带重试的图
graph = StateGraph(RobustState)

graph.add_node("operation", risky_operation)
graph.add_node("give_up", give_up_node)

graph.add_edge(START, "operation")

graph.add_conditional_edges(
    "operation",
    error_checker,
    {
        "retry": "operation",    # 重试 → 回到操作节点
        "give_up": "give_up",   # 放弃 → 放弃节点
        "success": END           # 成功 → 结束
    }
)
graph.add_edge("give_up", END)

robust_app = graph.compile()

# 测试
result = robust_app.invoke({
    "messages": [HumanMessage(content="执行操作")],
    "error_count": 0,
    "last_error": "",
    "max_retries": 3
})

for msg in result["messages"]:
    if isinstance(msg, AIMessage):
        print(f"[AI] {msg.content}")
print(f"错误次数: {result['error_count']}")
```

### LangGraph与LangSmith集成

```
┌──────────────────────────────────────────────────────────────┐
│            LangGraph + LangSmith 可观测性架构                  │
│                                                              │
│  ┌──────────────────────────────────────────────┐            │
│  │              LangGraph 应用                    │            │
│  │                                              │            │
│  │  [Node1] → [Node2] → [Node3] → [Node4]      │            │
│  │     │         │         │         │          │            │
│  │     ▼         ▼         ▼         ▼          │            │
│  │  ┌──────────────────────────────────────┐    │            │
│  │  │         LangSmith Tracer             │    │            │
│  │  │  自动记录每个节点的:                  │    │            │
│  │  │  - 输入/输出                          │    │            │
│  │  │  - LLM调用详情                        │    │            │
│  │  │  - 工具调用结果                       │    │            │
│  │  │  - 执行时间                           │    │            │
│  │  │  - Token消耗                          │    │            │
│  │  └──────────────────────────────────────┘    │            │
│  └──────────────────────────────────────────────┘            │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────┐            │
│  │           LangSmith Dashboard                 │            │
│  │                                              │            │
│  │  - 完整执行追踪 (Trace)                      │            │
│  │  - 延迟分析 (Latency)                        │            │
│  │  - Token使用统计                              │            │
│  │  - 错误率监控                                 │            │
│  │  - A/B测试对比                                │            │
│  │  - 数据集评估                                 │            │
│  └──────────────────────────────────────────────┘            │
│                                                              │
│  开启方法:                                                    │
│  export LANGCHAIN_TRACING_V2=true                            │
│  export LANGCHAIN_API_KEY=your_key                           │
│  export LANGCHAIN_PROJECT=my_project                         │
│                                                              │
│  代码中无需任何修改，LangGraph自动上报追踪数据                  │
└──────────────────────────────────────────────────────────────┘
```

```python
# 启用LangSmith追踪只需设置环境变量
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-langgraph-project"

# 之后所有 LangGraph 执行自动被追踪
# 在 smith.langchain.com 查看完整执行详情
```

---

## 总结

本教程全面覆盖了LangGraph的核心内容：

1. **LangGraph简介**: 基于图结构的有状态Agent框架，提供显式流程控制
2. **State定义**: 使用TypedDict和Pydantic定义贯穿图的状态对象，支持自定义Reducer
3. **Node函数**: LLM调用、意图识别、工具执行、回复生成等各类节点的编写方法
4. **条件路由**: add_conditional_edges实现基于状态的动态分支
5. **循环控制**: 通过条件边实现Agent的思考-行动循环
6. **Human-in-the-Loop**: interrupt机制实现人机协作审批流程
7. **Checkpointing**: 内存/SQLite持久化支持断点续传和多会话管理
8. **智能客服实战**: 完整的意图识别到专家路由到回复生成的客服系统

## 最佳实践

1. **State设计**: 精心设计State结构，包含所有需要在节点间传递的信息
2. **节点职责单一**: 每个Node只做一件事，保持函数简洁
3. **使用Checkpoint**: 生产环境务必使用持久化Checkpointer
4. **控制循环**: 始终设置最大迭代次数，防止无限循环
5. **错误处理**: 在每个Node中添加try-except，确保图不会因单个节点失败而崩溃
6. **流式输出**: 使用stream模式提升用户体验

## 参考资源

- [LangGraph官方文档](https://langchain-ai.github.io/langgraph/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [LangGraph教程](https://langchain-ai.github.io/langgraph/tutorials/)
- [LangSmith调试](https://smith.langchain.com/)

---

**文件大小目标**: 30KB
**创建时间**: 2024-01-01
**最后更新**: 2024-01-01
