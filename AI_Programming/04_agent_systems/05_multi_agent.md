# 多Agent协作系统

## 目录
1. [多Agent概述](#多agent概述)
2. [协作模式](#协作模式)
3. [层级结构](#层级结构)
4. [对等协作](#对等协作)
5. [共享记忆](#共享记忆)
6. [AutoGen GroupChat](#autogen-groupchat)
7. [实战：AI研究团队](#实战ai研究团队)

---

## 多Agent概述

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    多Agent系统 总体架构                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │  Agent A     │    │  Agent B     │    │  Agent C     │             │
│  │ ┌─────────┐ │    │ ┌─────────┐ │    │ ┌─────────┐ │             │
│  │ │  LLM    │ │    │ │  LLM    │ │    │ │  LLM    │ │             │
│  │ ├─────────┤ │    │ ├─────────┤ │    │ ├─────────┤ │             │
│  │ │  Tools  │ │    │ │  Tools  │ │    │ │  Tools  │ │             │
│  │ ├─────────┤ │    │ ├─────────┤ │    │ ├─────────┤ │             │
│  │ │ Memory  │ │    │ │ Memory  │ │    │ │ Memory  │ │             │
│  │ └─────────┘ │    │ └─────────┘ │    │ └─────────┘ │             │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘             │
│         │                  │                  │                     │
│         └──────────┬───────┴──────────┬───────┘                     │
│                    │                  │                             │
│              ┌─────▼─────┐    ┌──────▼──────┐                      │
│              │ 消息总线   │    │  共享记忆    │                      │
│              │ Message    │    │  Shared     │                      │
│              │ Bus        │    │  Memory     │                      │
│              └────────────┘    └─────────────┘                      │
│                                                                     │
│  协作模式:                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ 层级结构  │  │ 对等协作  │  │ 辩论模式  │  │ 流水线   │           │
│  │Hierarchy │  │Peer2Peer│  │ Debate   │  │Pipeline │           │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**多Agent系统**（Multi-Agent System, MAS）是指多个具有独立决策能力的智能体协同工作，共同完成复杂任务的系统。与单一Agent不同，多Agent系统通过分工协作，能够处理更加复杂、多样化的任务。

**为什么需要多Agent？**

单一Agent面临以下局限性：
- **能力边界**：一个Agent难以精通所有领域
- **上下文限制**：LLM的上下文窗口有限，无法同时处理大量信息
- **可靠性**：单点失败会导致整个任务失败
- **效率**：串行处理无法充分利用并行计算能力

多Agent系统的优势：
- **专业分工**：每个Agent专注自己的专业领域
- **并行处理**：多个Agent可以同时工作
- **容错能力**：一个Agent失败不影响其他Agent
- **质量保证**：通过审查、辩论等机制提升输出质量

**多Agent系统的关键组件：**

| 组件 | 说明 | 示例 |
|------|------|------|
| Agent | 具有独立能力的智能体 | Researcher, Writer, Reviewer |
| 通信协议 | Agent间交换信息的方式 | 消息传递、共享状态、事件广播 |
| 协调机制 | 管理Agent协作的策略 | Supervisor、投票、拍卖 |
| 共享记忆 | Agent间共享的知识库 | 向量数据库、键值存储 |
| 任务分配 | 将任务分配给合适的Agent | 路由器、调度器 |

### 代码示例

```python
# 多Agent系统 - 基础框架
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json


class AgentRole(Enum):
    """Agent角色定义"""
    RESEARCHER = "researcher"
    WRITER = "writer"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"
    ANALYST = "analyst"


@dataclass
class Message:
    """Agent间通信消息"""
    sender: str
    receiver: str  # "all" 表示广播
    content: str
    msg_type: str = "text"  # text, task, result, feedback
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self):
        import time
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class Task:
    """任务定义"""
    task_id: str
    description: str
    assigned_to: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


class BaseAgent(ABC):
    """Agent基类"""

    def __init__(self, name: str, role: AgentRole,
                 system_prompt: str = ""):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.message_history: List[Message] = []
        self.inbox: asyncio.Queue = asyncio.Queue()

    @abstractmethod
    async def process_message(self, message: Message) -> Optional[Message]:
        """处理收到的消息"""
        pass

    async def send_message(self, receiver: str, content: str,
                           msg_type: str = "text",
                           metadata: Dict = None) -> Message:
        """发送消息"""
        msg = Message(
            sender=self.name,
            receiver=receiver,
            content=content,
            msg_type=msg_type,
            metadata=metadata or {}
        )
        self.message_history.append(msg)
        return msg

    async def receive_message(self, message: Message):
        """接收消息"""
        self.message_history.append(message)
        await self.inbox.put(message)

    def get_context(self, last_n: int = 10) -> str:
        """获取最近的消息上下文"""
        recent = self.message_history[-last_n:]
        context_lines = []
        for msg in recent:
            context_lines.append(
                f"[{msg.sender} -> {msg.receiver}] ({msg.msg_type}): "
                f"{msg.content[:200]}"
            )
        return "\n".join(context_lines)


class MessageBus:
    """消息总线 - 管理Agent间通信"""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_log: List[Message] = []

    def register_agent(self, agent: BaseAgent):
        """注册Agent"""
        self.agents[agent.name] = agent
        print(f"[MessageBus] 注册Agent: {agent.name} ({agent.role.value})")

    async def route_message(self, message: Message):
        """路由消息"""
        self.message_log.append(message)

        if message.receiver == "all":
            # 广播消息
            for name, agent in self.agents.items():
                if name != message.sender:
                    await agent.receive_message(message)
        elif message.receiver in self.agents:
            # 点对点消息
            await self.agents[message.receiver].receive_message(message)
        else:
            print(f"[MessageBus] 未找到接收者: {message.receiver}")

    async def broadcast(self, sender: str, content: str,
                        msg_type: str = "text"):
        """广播消息"""
        msg = Message(
            sender=sender,
            receiver="all",
            content=content,
            msg_type=msg_type
        )
        await self.route_message(msg)


# 使用示例
if __name__ == "__main__":
    bus = MessageBus()
    print("多Agent基础框架初始化完成")
    print(f"支持的Agent角色: {[r.value for r in AgentRole]}")
```

---

## 协作模式

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    四种核心协作模式对比                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 层级模式 (Supervisor)        2. 对等模式 (Peer-to-Peer)         │
│  ┌─────────────┐                ┌─────┐   ┌─────┐                 │
│  │ Supervisor  │                │  A  │◄──►│  B  │                 │
│  └──┬───┬───┬──┘                └──┬──┘   └──┬──┘                 │
│     │   │   │                      │         │                     │
│  ┌──▼┐ ┌▼──┐ ┌▼──┐              ┌──▼──┐   ┌──▼──┐                 │
│  │ A │ │ B │ │ C │              │  C  │◄──►│  D  │                 │
│  └───┘ └───┘ └───┘              └─────┘   └─────┘                 │
│  自上而下的任务分配              Agent间直接通信协商                  │
│                                                                     │
│  3. 辩论模式 (Debate)            4. 流水线模式 (Pipeline)            │
│  ┌─────┐    ┌─────┐            ┌───┐  ┌───┐  ┌───┐  ┌───┐       │
│  │正方 A│◄──►│反方 B│            │ A │─►│ B │─►│ C │─►│ D │       │
│  └──┬───┘    └──┬──┘            └───┘  └───┘  └───┘  └───┘       │
│     └─────┬─────┘               每个Agent处理后传给下一个            │
│     ┌─────▼─────┐                                                   │
│     │  裁判 C   │                                                   │
│     └───────────┘                                                   │
│  正反双方辩论,裁判决策                                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**协作模式**是多Agent系统设计的核心，不同的任务场景适合不同的协作模式。以下是四种核心模式的详细对比：

| 模式 | 适用场景 | 优势 | 劣势 | 典型应用 |
|------|----------|------|------|----------|
| **层级模式** | 任务可明确分解 | 控制清晰、易于管理 | 单点瓶颈、灵活性差 | 项目管理、客服系统 |
| **对等模式** | 需要协商决策 | 灵活、无单点故障 | 协调复杂、可能死锁 | 头脑风暴、设计评审 |
| **辩论模式** | 需要多角度分析 | 输出质量高、考虑全面 | 耗时长、成本高 | 决策支持、内容审核 |
| **流水线模式** | 任务有明确步骤 | 专业分工、易于扩展 | 串行瓶颈、依赖前序 | 内容生产、数据处理 |

**选择协作模式的决策流程：**

1. 任务是否可以分解为独立子任务？ --> 是 --> 层级模式
2. 任务是否需要多轮协商？ --> 是 --> 对等模式
3. 任务是否需要质量验证？ --> 是 --> 辩论模式
4. 任务是否有明确的处理流程？ --> 是 --> 流水线模式
5. 复合型任务 --> 混合模式

### 代码示例

```python
# 协作模式 - 模式选择器和基础实现
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import asyncio


class CollaborationMode(Enum):
    """协作模式枚举"""
    HIERARCHICAL = "hierarchical"    # 层级模式
    PEER_TO_PEER = "peer_to_peer"    # 对等模式
    DEBATE = "debate"                # 辩论模式
    PIPELINE = "pipeline"            # 流水线模式
    HYBRID = "hybrid"                # 混合模式


@dataclass
class TaskAnalysis:
    """任务分析结果"""
    is_decomposable: bool = False    # 可分解
    needs_negotiation: bool = False  # 需要协商
    needs_verification: bool = False # 需要验证
    has_clear_steps: bool = False    # 有明确步骤
    complexity: str = "medium"       # low, medium, high


class ModeSelector:
    """协作模式选择器"""

    @staticmethod
    def select_mode(analysis: TaskAnalysis) -> CollaborationMode:
        """基于任务分析选择协作模式"""
        if analysis.has_clear_steps and analysis.is_decomposable:
            return CollaborationMode.PIPELINE
        elif analysis.needs_verification and analysis.complexity == "high":
            return CollaborationMode.DEBATE
        elif analysis.needs_negotiation:
            return CollaborationMode.PEER_TO_PEER
        elif analysis.is_decomposable:
            return CollaborationMode.HIERARCHICAL
        else:
            return CollaborationMode.HYBRID

    @staticmethod
    def explain_selection(mode: CollaborationMode) -> str:
        """解释选择原因"""
        explanations = {
            CollaborationMode.HIERARCHICAL:
                "任务可明确分解为独立子任务，适合使用层级模式，"
                "由Supervisor统一调度各Worker Agent。",
            CollaborationMode.PEER_TO_PEER:
                "任务需要多Agent协商讨论，适合使用对等模式，"
                "Agent之间平等交流达成共识。",
            CollaborationMode.DEBATE:
                "任务复杂度高且需要质量验证，适合使用辩论模式，"
                "正反双方充分讨论后由裁判做出最终决策。",
            CollaborationMode.PIPELINE:
                "任务有明确的处理步骤，适合使用流水线模式，"
                "每个Agent负责一个环节，依次传递处理。",
            CollaborationMode.HYBRID:
                "任务特征复合，建议使用混合模式，"
                "结合多种协作模式的优势。"
        }
        return explanations.get(mode, "未知模式")


class CollaborationProtocol(ABC):
    """协作协议基类"""

    def __init__(self, mode: CollaborationMode):
        self.mode = mode
        self.agents: Dict[str, Any] = {}
        self.execution_log: List[Dict] = []

    @abstractmethod
    async def execute(self, task: str) -> str:
        """执行协作任务"""
        pass

    def log_step(self, step: str, agent: str, detail: str):
        """记录执行步骤"""
        self.execution_log.append({
            "step": step,
            "agent": agent,
            "detail": detail
        })


# 使用示例
if __name__ == "__main__":
    # 分析任务特征
    task_analysis = TaskAnalysis(
        is_decomposable=True,
        needs_negotiation=False,
        needs_verification=True,
        has_clear_steps=True,
        complexity="high"
    )

    # 选择协作模式
    mode = ModeSelector.select_mode(task_analysis)
    explanation = ModeSelector.explain_selection(mode)

    print(f"推荐的协作模式: {mode.value}")
    print(f"选择原因: {explanation}")
```

---

## 层级结构

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    层级协作模式 (Supervisor Pattern)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                    ┌─────────────────────┐                          │
│                    │    Supervisor Agent  │                          │
│                    │  ┌───────────────┐  │                          │
│                    │  │ 任务理解      │  │                          │
│                    │  │ 任务分解      │  │                          │
│                    │  │ 结果汇总      │  │                          │
│                    │  │ 质量检查      │  │                          │
│                    │  └───────────────┘  │                          │
│                    └─┬──────┬──────┬─────┘                          │
│                      │      │      │                                │
│            ┌─────────▼┐  ┌──▼──────┐  ┌▼─────────┐                │
│            │Worker A  │  │Worker B │  │Worker C  │                │
│            │(研究员)  │  │(分析师) │  │(写作者)  │                │
│            │          │  │         │  │          │                │
│            │ 搜索工具 │  │ 计算工具│  │ 编辑工具 │                │
│            │ 爬虫工具 │  │ 图表工具│  │ 格式工具 │                │
│            └──────────┘  └─────────┘  └──────────┘                │
│                                                                     │
│  执行流程:                                                          │
│  ┌────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐              │
│  │任务│──►│ 分解 │──►│ 分配 │──►│ 执行 │──►│ 汇总 │              │
│  └────┘   └──────┘   └──────┘   └──────┘   └──────┘              │
│             理解任务    选择Agent   并行/串行   合并结果              │
│             拆分子任务  分配子任务   执行子任务   质量检查              │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**层级协作模式**（Supervisor Pattern）是多Agent系统中最常见的协作模式。它模仿人类团队中"管理者-执行者"的组织结构：一个Supervisor Agent负责理解任务、分解子任务、分配给Worker Agent，并汇总最终结果。

**Supervisor Agent的职责：**
1. **任务理解**：解析用户输入，理解任务目标
2. **任务分解**：将复杂任务拆分为可管理的子任务
3. **Agent选择**：根据子任务特征选择合适的Worker Agent
4. **结果汇总**：收集各Worker的输出，合并为最终结果
5. **质量控制**：验证结果质量，必要时要求重做

**Worker Agent的职责：**
1. **接受任务**：从Supervisor接收分配的子任务
2. **执行任务**：使用自己的工具和能力完成任务
3. **报告结果**：将执行结果返回给Supervisor

### 代码示例

```python
# 层级结构 - Supervisor + Worker 完整实现
import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from openai import AsyncOpenAI


@dataclass
class SubTask:
    """子任务"""
    task_id: str
    description: str
    assigned_agent: str
    status: str = "pending"
    result: Optional[str] = None


class WorkerAgent:
    """Worker Agent - 执行具体任务"""

    def __init__(self, name: str, specialty: str,
                 system_prompt: str, tools: List[Dict] = None):
        self.name = name
        self.specialty = specialty
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.client = AsyncOpenAI()

    async def execute(self, task: str) -> str:
        """执行任务"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"请完成以下任务:\n{task}"}
        ]

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )

        return response.choices[0].message.content

    def __repr__(self):
        return f"Worker({self.name}, specialty={self.specialty})"


class SupervisorAgent:
    """Supervisor Agent - 管理和协调Worker"""

    def __init__(self, workers: List[WorkerAgent]):
        self.workers = {w.name: w for w in workers}
        self.client = AsyncOpenAI()
        self.execution_history: List[Dict] = []

    async def decompose_task(self, task: str) -> List[SubTask]:
        """分解任务为子任务"""
        worker_descriptions = "\n".join([
            f"- {name}: {w.specialty}"
            for name, w in self.workers.items()
        ])

        messages = [
            {"role": "system", "content": f"""你是一个任务协调者。你的团队成员有:
{worker_descriptions}

请将用户的任务分解为子任务，并分配给合适的团队成员。
以JSON格式返回，格式如下:
[
  {{"task_id": "1", "description": "子任务描述", "assigned_agent": "Agent名称"}},
  ...
]
只返回JSON数组，不要其他内容。"""},
            {"role": "user", "content": task}
        ]

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )

        try:
            subtasks_data = json.loads(response.choices[0].message.content)
            return [SubTask(**st) for st in subtasks_data]
        except json.JSONDecodeError:
            # 如果解析失败，创建单一任务
            return [SubTask(
                task_id="1",
                description=task,
                assigned_agent=list(self.workers.keys())[0]
            )]

    async def execute_subtask(self, subtask: SubTask) -> SubTask:
        """执行单个子任务"""
        if subtask.assigned_agent not in self.workers:
            subtask.status = "failed"
            subtask.result = f"未找到Agent: {subtask.assigned_agent}"
            return subtask

        worker = self.workers[subtask.assigned_agent]
        subtask.status = "in_progress"

        try:
            result = await worker.execute(subtask.description)
            subtask.result = result
            subtask.status = "completed"
        except Exception as e:
            subtask.status = "failed"
            subtask.result = f"执行错误: {str(e)}"

        return subtask

    async def synthesize_results(self, task: str,
                                  subtasks: List[SubTask]) -> str:
        """汇总子任务结果"""
        results_text = "\n\n".join([
            f"### 子任务 {st.task_id}: {st.description}\n"
            f"负责人: {st.assigned_agent}\n"
            f"状态: {st.status}\n"
            f"结果:\n{st.result}"
            for st in subtasks
        ])

        messages = [
            {"role": "system", "content":
                "你是一个结果汇总专家。请根据各子任务的执行结果，"
                "为用户提供一个完整、连贯的最终答案。"},
            {"role": "user", "content":
                f"原始任务: {task}\n\n各子任务结果:\n{results_text}"}
        ]

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=3000
        )

        return response.choices[0].message.content

    async def run(self, task: str) -> str:
        """运行完整的层级协作流程"""
        print(f"\n{'='*60}")
        print(f"[Supervisor] 收到任务: {task}")
        print(f"{'='*60}")

        # 第一步：分解任务
        print("\n[Supervisor] 正在分解任务...")
        subtasks = await self.decompose_task(task)
        for st in subtasks:
            print(f"  子任务 {st.task_id}: {st.description}")
            print(f"    -> 分配给: {st.assigned_agent}")

        # 第二步：并行执行子任务
        print("\n[Supervisor] 开始执行子任务...")
        completed_tasks = await asyncio.gather(
            *[self.execute_subtask(st) for st in subtasks]
        )

        for st in completed_tasks:
            status_icon = "OK" if st.status == "completed" else "FAIL"
            print(f"  [{status_icon}] 子任务 {st.task_id}: {st.status}")

        # 第三步：汇总结果
        print("\n[Supervisor] 正在汇总结果...")
        final_result = await self.synthesize_results(task, completed_tasks)

        # 记录执行历史
        self.execution_history.append({
            "task": task,
            "subtasks": [
                {"id": st.task_id, "agent": st.assigned_agent,
                 "status": st.status}
                for st in completed_tasks
            ],
            "final_result": final_result[:200]
        })

        return final_result


# 使用示例
async def run_hierarchical_example():
    # 创建Worker Agent
    researcher = WorkerAgent(
        name="researcher",
        specialty="网络搜索和信息收集",
        system_prompt="你是一个研究专家，擅长搜集和整理信息。请提供详细、准确的研究结果。"
    )

    analyst = WorkerAgent(
        name="analyst",
        specialty="数据分析和趋势总结",
        system_prompt="你是一个数据分析师，擅长从信息中提取关键洞察和趋势。"
    )

    writer = WorkerAgent(
        name="writer",
        specialty="内容创作和报告撰写",
        system_prompt="你是一个专业写作者，擅长将信息组织成清晰、有条理的文章。"
    )

    # 创建Supervisor
    supervisor = SupervisorAgent(
        workers=[researcher, analyst, writer]
    )

    # 执行任务
    result = await supervisor.run(
        "请分析2024年AI Agent技术的发展趋势，"
        "并撰写一份简要的分析报告。"
    )

    print(f"\n{'='*60}")
    print("最终结果:")
    print(f"{'='*60}")
    print(result)


if __name__ == "__main__":
    asyncio.run(run_hierarchical_example())
```

---

## 对等协作

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    对等协作模式 (Peer-to-Peer)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐                          ┌──────────┐                │
│  │ Agent A  │◄────── 消息通道 ────────►│ Agent B  │                │
│  │ (设计师) │                          │ (工程师) │                │
│  └────┬─────┘                          └────┬─────┘                │
│       │                                      │                     │
│       │    ┌──────────────────────┐          │                     │
│       └───►│    共享白板/状态     │◄─────────┘                     │
│            │  ┌────────────────┐ │                                  │
│       ┌───►│  │ 当前共识       │ │◄─────────┐                     │
│       │    │  │ 待解决分歧     │ │          │                     │
│       │    │  │ 投票记录       │ │          │                     │
│       │    │  └────────────────┘ │          │                     │
│       │    └──────────────────────┘          │                     │
│       │                                      │                     │
│  ┌────┴─────┐                          ┌────┴─────┐                │
│  │ Agent C  │◄────── 消息通道 ────────►│ Agent D  │                │
│  │ (测试员) │                          │ (审核者) │                │
│  └──────────┘                          └──────────┘                │
│                                                                     │
│  协作流程:                                                          │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐               │
│  │ 提议 │─►│ 讨论 │─►│ 修改 │─►│ 投票 │─►│ 共识 │               │
│  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘               │
│  某Agent    其他Agent  根据反馈   多数决或    记录最终               │
│  提出方案   给出意见   调整方案   一致同意    决策结果               │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**对等协作模式**（Peer-to-Peer）是一种去中心化的协作方式，所有Agent地位平等，通过协商、讨论和投票达成共识。这种模式模仿了人类团队中的头脑风暴和民主决策过程。

**对等协作的核心机制：**

1. **提议（Propose）**：任何Agent可以提出解决方案
2. **讨论（Discuss）**：其他Agent给出评价和建议
3. **修改（Revise）**：根据讨论结果修改方案
4. **投票（Vote）**：通过投票机制达成决策
5. **共识（Consensus）**：记录并执行最终决策

**投票策略对比：**

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| 多数决 | 超过半数同意即通过 | 一般性决策 |
| 一致同意 | 所有Agent都同意才通过 | 重要决策 |
| 加权投票 | 专家Agent投票权重更高 | 专业性决策 |
| 排名投票 | Agent对多个方案排序 | 多方案选择 |

### 代码示例

```python
# 对等协作 - Peer-to-Peer 完整实现
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from openai import AsyncOpenAI


class VoteType(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class Proposal:
    """提案"""
    proposer: str
    content: str
    version: int = 1
    votes: Dict[str, VoteType] = field(default_factory=dict)
    feedback: List[Dict[str, str]] = field(default_factory=list)
    status: str = "proposed"  # proposed, discussing, voting, accepted, rejected


@dataclass
class SharedWhiteboard:
    """共享白板 - Agent间共享状态"""
    topic: str = ""
    current_proposals: List[Proposal] = field(default_factory=list)
    consensus_history: List[str] = field(default_factory=list)
    discussion_log: List[Dict[str, str]] = field(default_factory=list)

    def add_discussion(self, agent: str, message: str):
        self.discussion_log.append({
            "agent": agent,
            "message": message
        })

    def get_context(self, last_n: int = 10) -> str:
        """获取最近的讨论上下文"""
        recent = self.discussion_log[-last_n:]
        return "\n".join([
            f"[{d['agent']}]: {d['message']}" for d in recent
        ])


class PeerAgent:
    """对等协作Agent"""

    def __init__(self, name: str, expertise: str,
                 personality: str = ""):
        self.name = name
        self.expertise = expertise
        self.personality = personality
        self.client = AsyncOpenAI()

    async def propose(self, topic: str, context: str = "") -> str:
        """提出方案"""
        messages = [
            {"role": "system", "content":
                f"你是{self.name}，专长是{self.expertise}。"
                f"{self.personality}\n"
                f"请针对讨论主题提出你的方案。"},
            {"role": "user", "content":
                f"讨论主题: {topic}\n\n"
                f"已有讨论:\n{context}\n\n"
                f"请提出你的方案:"}
        ]

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.8,
            max_tokens=1000
        )
        return response.choices[0].message.content

    async def review(self, proposal: str, topic: str,
                     context: str = "") -> Dict[str, str]:
        """评审方案"""
        messages = [
            {"role": "system", "content":
                f"你是{self.name}，专长是{self.expertise}。"
                f"{self.personality}\n"
                f"请从你的专业角度评审这个方案，指出优点和不足。"},
            {"role": "user", "content":
                f"讨论主题: {topic}\n"
                f"待评审方案:\n{proposal}\n\n"
                f"已有讨论:\n{context}\n\n"
                f"请给出你的评审意见:"}
        ]

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )

        return {
            "agent": self.name,
            "feedback": response.choices[0].message.content
        }

    async def vote(self, proposal: str, feedbacks: List[str],
                   topic: str) -> VoteType:
        """投票"""
        feedback_text = "\n".join(feedbacks)

        messages = [
            {"role": "system", "content":
                f"你是{self.name}，专长是{self.expertise}。\n"
                f"请基于方案内容和讨论反馈做出投票决定。\n"
                f"只回答: approve 或 reject 或 abstain"},
            {"role": "user", "content":
                f"主题: {topic}\n\n"
                f"方案:\n{proposal}\n\n"
                f"讨论反馈:\n{feedback_text}\n\n"
                f"你的投票 (approve/reject/abstain):"}
        ]

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=10
        )

        vote_text = response.choices[0].message.content.strip().lower()
        if "approve" in vote_text:
            return VoteType.APPROVE
        elif "reject" in vote_text:
            return VoteType.REJECT
        else:
            return VoteType.ABSTAIN


class PeerToPeerCollaboration:
    """对等协作协调器"""

    def __init__(self, agents: List[PeerAgent],
                 max_rounds: int = 3):
        self.agents = {a.name: a for a in agents}
        self.whiteboard = SharedWhiteboard()
        self.max_rounds = max_rounds

    async def run_discussion(self, topic: str) -> str:
        """运行完整的对等讨论流程"""
        self.whiteboard.topic = topic
        print(f"\n{'='*60}")
        print(f"[讨论] 主题: {topic}")
        print(f"[讨论] 参与者: {list(self.agents.keys())}")
        print(f"{'='*60}")

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n--- 第 {round_num} 轮讨论 ---")

            # 第一步：第一个Agent提出方案
            proposer = list(self.agents.values())[
                (round_num - 1) % len(self.agents)
            ]
            print(f"\n[提议] {proposer.name} 正在提出方案...")
            proposal_content = await proposer.propose(
                topic, self.whiteboard.get_context()
            )

            proposal = Proposal(
                proposer=proposer.name,
                content=proposal_content,
                version=round_num
            )
            self.whiteboard.current_proposals.append(proposal)
            self.whiteboard.add_discussion(
                proposer.name, f"[提案] {proposal_content[:200]}..."
            )
            print(f"  方案: {proposal_content[:100]}...")

            # 第二步：其他Agent评审
            print(f"\n[评审] 其他Agent正在评审...")
            reviewers = [a for a in self.agents.values()
                        if a.name != proposer.name]

            review_tasks = [
                reviewer.review(
                    proposal_content, topic,
                    self.whiteboard.get_context()
                )
                for reviewer in reviewers
            ]
            reviews = await asyncio.gather(*review_tasks)

            for review in reviews:
                proposal.feedback.append(review)
                self.whiteboard.add_discussion(
                    review["agent"],
                    f"[评审] {review['feedback'][:200]}..."
                )
                print(f"  {review['agent']}: "
                      f"{review['feedback'][:80]}...")

            # 第三步：投票
            print(f"\n[投票] 所有Agent正在投票...")
            feedback_texts = [r["feedback"] for r in reviews]

            vote_tasks = [
                agent.vote(proposal_content, feedback_texts, topic)
                for agent in self.agents.values()
            ]
            votes = await asyncio.gather(*vote_tasks)

            for agent, vote in zip(self.agents.values(), votes):
                proposal.votes[agent.name] = vote
                print(f"  {agent.name}: {vote.value}")

            # 统计投票
            approve_count = sum(
                1 for v in proposal.votes.values()
                if v == VoteType.APPROVE
            )
            total = len(proposal.votes)

            if approve_count > total / 2:
                proposal.status = "accepted"
                print(f"\n[结果] 方案通过！"
                      f"({approve_count}/{total}票赞成)")
                self.whiteboard.consensus_history.append(
                    proposal_content
                )
                return proposal_content
            else:
                proposal.status = "rejected"
                print(f"\n[结果] 方案未通过"
                      f"({approve_count}/{total}票赞成)，"
                      f"继续讨论...")

        # 如果所有轮次都没有达成共识
        print("\n[结果] 未能达成共识，返回最后一轮方案")
        return self.whiteboard.current_proposals[-1].content


# 使用示例
async def run_peer_collaboration():
    agents = [
        PeerAgent(
            "设计师小王", "UI/UX设计",
            "你注重用户体验和视觉美感。"
        ),
        PeerAgent(
            "工程师小李", "后端开发",
            "你注重技术可行性和系统性能。"
        ),
        PeerAgent(
            "产品经理小张", "产品规划",
            "你注重市场需求和商业价值。"
        ),
    ]

    collaboration = PeerToPeerCollaboration(agents, max_rounds=3)

    result = await collaboration.run_discussion(
        "设计一个AI驱动的客户服务聊天机器人的核心功能列表"
    )

    print(f"\n最终方案:\n{result}")


if __name__ == "__main__":
    asyncio.run(run_peer_collaboration())
```

---

## 共享记忆

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    共享记忆系统架构                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                         │
│  │ Agent A  │  │ Agent B  │  │ Agent C  │                         │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                         │
│       │  读/写       │  读/写       │  读/写                        │
│       └──────────┬───┴──────────┬───┘                               │
│                  │              │                                    │
│  ┌───────────────▼──────────────▼───────────────┐                  │
│  │              共享记忆层                        │                  │
│  ├──────────────────────────────────────────────┤                  │
│  │                                              │                  │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────┐  │                  │
│  │  │ 短期记忆  │  │ 长期记忆   │  │ 工作记忆  │  │                  │
│  │  │ (Queue)  │  │ (Vector   │  │ (KV     │  │                  │
│  │  │ 最近消息  │  │  Store)   │  │  Store)  │  │                  │
│  │  │ 对话历史  │  │ 知识库    │  │ 当前任务  │  │                  │
│  │  └──────────┘  │ 经验总结   │  │ 中间结果  │  │                  │
│  │                └───────────┘  └──────────┘  │                  │
│  │                                              │                  │
│  │  ┌────────────────────────────────────────┐  │                  │
│  │  │           访问控制层                     │  │                  │
│  │  │  Agent A: 读写短期+工作, 只读长期       │  │                  │
│  │  │  Agent B: 读写所有                      │  │                  │
│  │  │  Agent C: 只读短期, 读写长期            │  │                  │
│  │  └────────────────────────────────────────┘  │                  │
│  └──────────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**共享记忆**是多Agent系统中实现信息共享和协调的关键机制。它让不同的Agent能够访问共同的知识和状态，避免重复工作，确保信息一致性。

**三种记忆类型：**

| 记忆类型 | 存储内容 | 特点 | 实现方式 |
|----------|----------|------|----------|
| **短期记忆** | 最近的消息、对话历史 | 容量有限、FIFO | Queue、Ring Buffer |
| **长期记忆** | 知识库、经验总结、历史决策 | 持久化、可检索 | 向量数据库、SQL |
| **工作记忆** | 当前任务状态、中间结果 | 临时性、任务相关 | 键值存储、Redis |

**访问控制策略：**
- **完全共享**：所有Agent可读写所有记忆（简单但风险高）
- **角色限制**：根据Agent角色限制读写权限
- **命名空间隔离**：每个Agent有私有空间 + 共享空间

### 代码示例

```python
# 共享记忆系统 - 完整实现
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


class Permission(Enum):
    """权限级别"""
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"
    NONE = "none"


@dataclass
class MemoryEntry:
    """记忆条目"""
    key: str
    value: Any
    author: str
    timestamp: float = 0.0
    memory_type: str = "short_term"
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class ShortTermMemory:
    """短期记忆 - 基于队列的FIFO存储"""

    def __init__(self, max_size: int = 100):
        self.buffer: deque = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, entry: MemoryEntry):
        """添加记忆"""
        entry.memory_type = "short_term"
        self.buffer.append(entry)

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        """获取最近n条记忆"""
        return list(self.buffer)[-n:]

    def search(self, keyword: str) -> List[MemoryEntry]:
        """关键词搜索"""
        results = []
        for entry in self.buffer:
            if keyword.lower() in str(entry.value).lower():
                results.append(entry)
        return results

    def clear(self):
        """清空"""
        self.buffer.clear()

    @property
    def size(self) -> int:
        return len(self.buffer)


class LongTermMemory:
    """长期记忆 - 基于向量的语义检索"""

    def __init__(self):
        self.entries: Dict[str, MemoryEntry] = {}
        # 简化版：使用关键词索引代替向量检索
        self.tag_index: Dict[str, Set[str]] = {}

    def store(self, entry: MemoryEntry):
        """存储记忆"""
        entry.memory_type = "long_term"
        self.entries[entry.key] = entry

        # 更新标签索引
        for tag in entry.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(entry.key)

    def retrieve(self, key: str) -> Optional[MemoryEntry]:
        """按键检索"""
        return self.entries.get(key)

    def search_by_tags(self, tags: List[str]) -> List[MemoryEntry]:
        """按标签检索"""
        matching_keys = set()
        for tag in tags:
            if tag in self.tag_index:
                matching_keys.update(self.tag_index[tag])

        return [self.entries[k] for k in matching_keys
                if k in self.entries]

    def search_by_keyword(self, keyword: str) -> List[MemoryEntry]:
        """关键词搜索"""
        return [
            entry for entry in self.entries.values()
            if keyword.lower() in str(entry.value).lower() or
               keyword.lower() in entry.key.lower()
        ]

    def get_all(self) -> List[MemoryEntry]:
        """获取所有记忆"""
        return list(self.entries.values())

    @property
    def size(self) -> int:
        return len(self.entries)


class WorkingMemory:
    """工作记忆 - 当前任务的键值存储"""

    def __init__(self):
        self.store: Dict[str, Any] = {}
        self.task_context: Dict[str, Any] = {}

    def set(self, key: str, value: Any, author: str = ""):
        """设置值"""
        self.store[key] = {
            "value": value,
            "author": author,
            "timestamp": time.time()
        }

    def get(self, key: str, default: Any = None) -> Any:
        """获取值"""
        entry = self.store.get(key)
        if entry:
            return entry["value"]
        return default

    def delete(self, key: str):
        """删除值"""
        self.store.pop(key, None)

    def set_task_context(self, task_id: str, context: Dict):
        """设置任务上下文"""
        self.task_context[task_id] = context

    def get_task_context(self, task_id: str) -> Dict:
        """获取任务上下文"""
        return self.task_context.get(task_id, {})

    def clear_task(self, task_id: str):
        """清除任务相关数据"""
        self.task_context.pop(task_id, None)

    def get_summary(self) -> Dict[str, Any]:
        """获取工作记忆摘要"""
        return {
            "total_keys": len(self.store),
            "active_tasks": len(self.task_context),
            "keys": list(self.store.keys())
        }


class SharedMemorySystem:
    """共享记忆系统 - 统一管理三种记忆"""

    def __init__(self):
        self.short_term = ShortTermMemory(max_size=200)
        self.long_term = LongTermMemory()
        self.working = WorkingMemory()

        # 访问控制: agent_name -> {memory_type: permission}
        self.permissions: Dict[str, Dict[str, Permission]] = {}

    def set_permissions(self, agent_name: str,
                        permissions: Dict[str, Permission]):
        """设置Agent的记忆访问权限"""
        self.permissions[agent_name] = permissions

    def check_permission(self, agent_name: str,
                         memory_type: str,
                         action: str) -> bool:
        """检查权限"""
        if agent_name not in self.permissions:
            return True  # 默认允许

        perm = self.permissions[agent_name].get(
            memory_type, Permission.READ_WRITE
        )

        if perm == Permission.NONE:
            return False
        elif perm == Permission.READ and action == "write":
            return False
        elif perm == Permission.WRITE and action == "read":
            return False
        return True

    def add_to_short_term(self, key: str, value: Any,
                           author: str, tags: List[str] = None):
        """添加到短期记忆"""
        if not self.check_permission(author, "short_term", "write"):
            raise PermissionError(
                f"{author} 没有短期记忆的写权限"
            )

        entry = MemoryEntry(
            key=key, value=value, author=author,
            tags=tags or []
        )
        self.short_term.add(entry)

    def store_to_long_term(self, key: str, value: Any,
                            author: str, tags: List[str] = None):
        """存储到长期记忆"""
        if not self.check_permission(author, "long_term", "write"):
            raise PermissionError(
                f"{author} 没有长期记忆的写权限"
            )

        entry = MemoryEntry(
            key=key, value=value, author=author,
            tags=tags or []
        )
        self.long_term.store(entry)

    def set_working(self, key: str, value: Any, author: str):
        """设置工作记忆"""
        if not self.check_permission(author, "working", "write"):
            raise PermissionError(
                f"{author} 没有工作记忆的写权限"
            )
        self.working.set(key, value, author)

    def query(self, agent_name: str, memory_type: str,
              keyword: str = "", tags: List[str] = None,
              last_n: int = 10) -> List[Any]:
        """统一查询接口"""
        if not self.check_permission(agent_name, memory_type, "read"):
            raise PermissionError(
                f"{agent_name} 没有{memory_type}的读权限"
            )

        if memory_type == "short_term":
            if keyword:
                return self.short_term.search(keyword)
            return self.short_term.get_recent(last_n)

        elif memory_type == "long_term":
            if tags:
                return self.long_term.search_by_tags(tags)
            elif keyword:
                return self.long_term.search_by_keyword(keyword)
            return self.long_term.get_all()

        elif memory_type == "working":
            value = self.working.get(keyword)
            return [value] if value else []

        return []

    def get_status(self) -> Dict[str, Any]:
        """获取记忆系统状态"""
        return {
            "short_term_size": self.short_term.size,
            "long_term_size": self.long_term.size,
            "working_memory": self.working.get_summary(),
            "registered_agents": list(self.permissions.keys())
        }


# 使用示例
if __name__ == "__main__":
    # 创建共享记忆系统
    memory = SharedMemorySystem()

    # 设置权限
    memory.set_permissions("researcher", {
        "short_term": Permission.READ_WRITE,
        "long_term": Permission.READ_WRITE,
        "working": Permission.READ_WRITE
    })
    memory.set_permissions("writer", {
        "short_term": Permission.READ,
        "long_term": Permission.READ,
        "working": Permission.READ_WRITE
    })

    # Researcher存储研究结果
    memory.add_to_short_term(
        "search_1", "AI Agent市场预计2025年达到500亿美元",
        author="researcher", tags=["market", "ai"]
    )
    memory.store_to_long_term(
        "insight_1", "多Agent系统是AI应用的重要趋势",
        author="researcher", tags=["trend", "multi-agent"]
    )

    # Writer读取研究结果
    results = memory.query("writer", "short_term", last_n=5)
    print(f"Writer读取到 {len(results)} 条短期记忆")

    # 系统状态
    status = memory.get_status()
    print(f"记忆系统状态: {json.dumps(status, ensure_ascii=False, indent=2)}")
```

---

## AutoGen GroupChat

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AutoGen GroupChat 架构                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────┐           │
│  │                GroupChatManager                      │           │
│  │  ┌─────────────────────────────────────────────┐    │           │
│  │  │  Speaker Selection Strategy                  │    │           │
│  │  │  - round_robin (轮流发言)                    │    │           │
│  │  │  - auto (LLM自动选择)                        │    │           │
│  │  │  - manual (手动指定)                         │    │           │
│  │  │  - random (随机选择)                         │    │           │
│  │  └─────────────────────────────────────────────┘    │           │
│  │  ┌─────────────────────────────────────────────┐    │           │
│  │  │  Termination Condition                       │    │           │
│  │  │  - max_turns (最大轮数)                      │    │           │
│  │  │  - keyword (关键词触发)                      │    │           │
│  │  │  - function (自定义函数)                     │    │           │
│  │  └─────────────────────────────────────────────┘    │           │
│  └──────────────────────┬──────────────────────────────┘           │
│                         │ 管理                                      │
│     ┌───────────────────┼───────────────────┐                      │
│     │                   │                   │                      │
│  ┌──▼──────────┐  ┌────▼─────────┐  ┌─────▼────────┐             │
│  │AssistantAgent│  │AssistantAgent│  │ UserProxy    │             │
│  │  "Coder"    │  │  "Reviewer"  │  │  Agent       │             │
│  │             │  │              │  │              │             │
│  │ system_msg  │  │ system_msg   │  │ human_input  │             │
│  │ llm_config  │  │ llm_config   │  │ code_exec    │             │
│  └─────────────┘  └──────────────┘  └──────────────┘             │
│                                                                     │
│  消息流: User ──► Manager ──► Agent1 ──► Manager ──► Agent2 ...   │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**AutoGen** 是微软开源的多Agent框架，其核心概念是**GroupChat**——一种让多个Agent在聊天室中协同工作的模式。AutoGen的设计理念是让多Agent协作变得简单，就像人类在群聊中讨论问题一样。

**AutoGen 核心组件：**

| 组件 | 说明 | 功能 |
|------|------|------|
| **ConversableAgent** | 基础Agent类 | 所有Agent的父类 |
| **AssistantAgent** | 助手Agent | 由LLM驱动的Agent |
| **UserProxyAgent** | 用户代理Agent | 代表用户，可执行代码 |
| **GroupChat** | 群聊管理 | 管理多Agent对话 |
| **GroupChatManager** | 群聊管理器 | 控制发言顺序和终止条件 |

**Speaker Selection策略对比：**

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `round_robin` | 按固定顺序轮流发言 | 流程性任务 |
| `auto` | LLM根据对话内容自动选择下一个发言者 | 灵活讨论 |
| `manual` | 由特定Agent指定下一个发言者 | 受控流程 |
| `random` | 随机选择发言者 | 头脑风暴 |

### 代码示例

```python
# AutoGen GroupChat - 完整实现示例
# pip install pyautogen

import autogen
from typing import Dict, List, Any, Optional


# ===================== 基础配置 =====================

# LLM配置
llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini",
            "api_key": "your-api-key-here",
        }
    ],
    "temperature": 0.7,
    "timeout": 120,
}


# ===================== 示例1: 基础两人对话 =====================

def basic_two_agent_chat():
    """基础两人对话示例"""

    # 创建助手Agent
    assistant = autogen.AssistantAgent(
        name="assistant",
        system_message="""你是一个有帮助的AI助手。
        请清晰、简洁地回答问题。
        当你认为任务完成时，回复 TERMINATE。""",
        llm_config=llm_config,
    )

    # 创建用户代理Agent (不需要人类输入)
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",  # 不需要人类输入
        max_consecutive_auto_reply=5,
        is_termination_msg=lambda x: x.get(
            "content", ""
        ).rstrip().endswith("TERMINATE"),
        code_execution_config={
            "work_dir": "coding_workspace",
            "use_docker": False,
        },
    )

    # 发起对话
    user_proxy.initiate_chat(
        assistant,
        message="请写一个Python函数，计算斐波那契数列的第n项。"
    )


# ===================== 示例2: GroupChat多Agent协作 ===========

def groupchat_coding_team():
    """GroupChat编程团队示例"""

    # 产品经理Agent
    product_manager = autogen.AssistantAgent(
        name="ProductManager",
        system_message="""你是产品经理。你的职责是:
        1. 理解用户需求
        2. 将需求转化为技术需求文档
        3. 验收最终产品是否符合需求
        在讨论结束时确认需求是否被满足。""",
        llm_config=llm_config,
    )

    # 程序员Agent
    coder = autogen.AssistantAgent(
        name="Coder",
        system_message="""你是一个高级程序员。你的职责是:
        1. 根据需求编写高质量的Python代码
        2. 代码要包含完整的类型注解和文档字符串
        3. 代码要有适当的错误处理
        4. 响应代码审查的反馈并修改代码""",
        llm_config=llm_config,
    )

    # 代码审查Agent
    reviewer = autogen.AssistantAgent(
        name="CodeReviewer",
        system_message="""你是一个代码审查专家。你的职责是:
        1. 审查代码质量（可读性、可维护性）
        2. 检查潜在的bug和安全问题
        3. 提出改进建议
        4. 如果代码质量达标，明确表示"代码审查通过"
        审查标准：代码规范、错误处理、性能、安全性""",
        llm_config=llm_config,
    )

    # 测试员Agent
    tester = autogen.AssistantAgent(
        name="Tester",
        system_message="""你是一个测试工程师。你的职责是:
        1. 为代码编写单元测试
        2. 考虑边界情况和异常情况
        3. 报告测试结果
        当所有测试通过后，回复"所有测试通过 TERMINATE" """,
        llm_config=llm_config,
    )

    # 用户代理 (用于执行代码)
    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        human_input_mode="NEVER",
        code_execution_config={
            "work_dir": "team_workspace",
            "use_docker": False,
        },
        is_termination_msg=lambda x: "TERMINATE" in (
            x.get("content", "") or ""
        ),
    )

    # 创建GroupChat
    groupchat = autogen.GroupChat(
        agents=[user_proxy, product_manager, coder,
                reviewer, tester],
        messages=[],
        max_round=15,
        speaker_selection_method="auto",  # LLM自动选择
    )

    # 创建GroupChat管理器
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    # 发起讨论
    user_proxy.initiate_chat(
        manager,
        message="""请开发一个任务管理器类（TaskManager），要求：
        1. 支持添加、删除、更新任务
        2. 支持按优先级排序
        3. 支持按状态筛选（pending, in_progress, completed）
        4. 支持任务搜索
        5. 包含完整的测试代码"""
    )


# ===================== 示例3: 自定义Speaker选择 ===============

def custom_speaker_selection():
    """自定义发言者选择逻辑"""

    researcher = autogen.AssistantAgent(
        name="Researcher",
        system_message="你是研究员，负责收集和整理信息。",
        llm_config=llm_config,
    )

    analyst = autogen.AssistantAgent(
        name="Analyst",
        system_message="你是分析师，负责分析数据和趋势。",
        llm_config=llm_config,
    )

    writer = autogen.AssistantAgent(
        name="Writer",
        system_message="""你是写作者，负责撰写最终报告。
        当报告完成时，回复 TERMINATE。""",
        llm_config=llm_config,
    )

    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        human_input_mode="NEVER",
        is_termination_msg=lambda x: "TERMINATE" in (
            x.get("content", "") or ""
        ),
    )

    # 自定义发言顺序: 研究 -> 分析 -> 写作 (流水线模式)
    def custom_speaker_selection_func(
        last_speaker, groupchat
    ):
        """自定义发言者选择函数"""
        order = {
            "Admin": researcher,
            "Researcher": analyst,
            "Analyst": writer,
            "Writer": None  # 结束
        }

        next_speaker = order.get(last_speaker.name)
        if next_speaker is None:
            return researcher  # 循环
        return next_speaker

    groupchat = autogen.GroupChat(
        agents=[user_proxy, researcher, analyst, writer],
        messages=[],
        max_round=10,
        speaker_selection_method=custom_speaker_selection_func,
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
    )

    user_proxy.initiate_chat(
        manager,
        message="请研究并分析AI Agent技术在2024年的发展趋势，"
                "撰写一份简要报告。"
    )


# ===================== 示例4: 带工具调用的Agent ================

def agent_with_tools():
    """带工具调用的Agent示例"""

    # 定义工具函数
    def search_web(query: str) -> str:
        """模拟网络搜索"""
        mock_results = {
            "AI Agent": "AI Agent是能够自主决策和行动的智能系统...",
            "LangChain": "LangChain是一个用于构建LLM应用的框架...",
            "AutoGen": "AutoGen是微软开源的多Agent协作框架...",
        }
        for key, value in mock_results.items():
            if key.lower() in query.lower():
                return value
        return f"搜索'{query}'的结果: 未找到相关信息"

    def calculate(expression: str) -> str:
        """计算数学表达式"""
        try:
            result = eval(expression)
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"

    # 创建带工具的助手
    assistant = autogen.AssistantAgent(
        name="ToolAssistant",
        system_message="""你是一个有帮助的助手，可以使用搜索和计算工具。
        使用search_web搜索信息，使用calculate进行计算。
        完成任务后回复 TERMINATE。""",
        llm_config=llm_config,
    )

    # 注册工具函数
    user_proxy = autogen.UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        is_termination_msg=lambda x: "TERMINATE" in (
            x.get("content", "") or ""
        ),
    )

    # 使用register_for_llm和register_for_execution注册工具
    @user_proxy.register_for_execution()
    @assistant.register_for_llm(
        description="搜索网络获取信息"
    )
    def web_search(query: str) -> str:
        return search_web(query)

    @user_proxy.register_for_execution()
    @assistant.register_for_llm(
        description="计算数学表达式"
    )
    def calc(expression: str) -> str:
        return calculate(expression)

    # 发起对话
    user_proxy.initiate_chat(
        assistant,
        message="请搜索AutoGen相关信息，并计算123 * 456的结果。"
    )


# ===================== 运行示例 =====================

if __name__ == "__main__":
    print("=" * 60)
    print("AutoGen GroupChat 示例")
    print("=" * 60)

    print("\n可运行的示例:")
    print("1. basic_two_agent_chat() - 基础两人对话")
    print("2. groupchat_coding_team() - 编程团队协作")
    print("3. custom_speaker_selection() - 自定义发言顺序")
    print("4. agent_with_tools() - 带工具的Agent")

    # 取消注释以运行对应示例:
    # basic_two_agent_chat()
    # groupchat_coding_team()
    # custom_speaker_selection()
    # agent_with_tools()
```

---

## 实战：AI研究团队

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AI研究团队 系统架构                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  用户输入: "研究AI Agent技术发展趋势"                                │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────┐           │
│  │              Team Coordinator (协调者)                │           │
│  │  - 理解研究主题                                      │           │
│  │  - 制定研究计划                                      │           │
│  │  - 分配任务给团队成员                                 │           │
│  │  - 汇总最终研究报告                                   │           │
│  └───┬────────────────┬─────────────────┬──────────────┘           │
│      │                │                 │                           │
│      ▼                ▼                 ▼                           │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐                   │
│  │Researcher│   │ Analyst  │   │ Writer       │                   │
│  │ (研究员) │   │ (分析师) │   │ (写作者)     │                   │
│  │          │   │          │   │              │                   │
│  │ 工具:    │   │ 工具:    │   │ 工具:        │                   │
│  │ - 搜索   │   │ - 计算   │   │ - 格式化     │                   │
│  │ - 爬虫   │   │ - 图表   │   │ - 模板引擎   │                   │
│  │ - 文献   │   │ - 统计   │   │ - 审校工具   │                   │
│  └─────┬────┘   └────┬─────┘   └──────┬───────┘                   │
│        │             │                │                             │
│        ▼             ▼                ▼                             │
│  ┌──────────────────────────────────────────────┐                  │
│  │              共享记忆 (Shared Memory)         │                  │
│  │  - 研究资料库                                 │                  │
│  │  - 分析结果                                   │                  │
│  │  - 草稿版本                                   │                  │
│  └──────────────────────────────────────────────┘                  │
│                                                                     │
│  执行流程:                                                          │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐               │
│  │ 规划 │─►│ 研究 │─►│ 分析 │─►│ 写作 │─►│ 审核 │               │
│  │      │  │      │  │      │  │      │  │      │               │
│  │协调者│  │研究员│  │分析师│  │写作者│  │协调者│               │
│  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

本实战项目实现一个完整的**AI研究团队**，包含四个角色的Agent协作完成研究任务。我们使用LangGraph作为核心编排框架，结合共享记忆系统，实现从研究规划到最终报告的完整工作流。

**团队角色定义：**

| 角色 | 职责 | 能力 |
|------|------|------|
| **Coordinator** | 统筹规划、任务分配、结果审核 | 理解需求、分解任务、质量把控 |
| **Researcher** | 信息搜集、文献调研 | 网络搜索、论文检索、数据收集 |
| **Analyst** | 数据分析、趋势提取 | 统计分析、趋势识别、对比研究 |
| **Writer** | 报告撰写、内容组织 | 结构化写作、可视化呈现、审校 |

**执行流程详解：**

1. **规划阶段**（Coordinator）
   - 解析研究主题
   - 确定研究范围和目标
   - 制定研究计划，拆分子任务

2. **研究阶段**（Researcher）
   - 执行网络搜索
   - 收集相关论文和报告
   - 整理原始资料

3. **分析阶段**（Analyst）
   - 分析收集的资料
   - 提取关键趋势和洞察
   - 生成分析报告

4. **写作阶段**（Writer）
   - 组织报告结构
   - 撰写完整报告
   - 添加引用和参考

5. **审核阶段**（Coordinator）
   - 检查报告质量
   - 确认是否满足需求
   - 提出修改建议或确认完成

### 代码示例

```python
# 实战：AI研究团队 - 基于LangGraph的完整实现
# pip install langchain langgraph langchain-openai

import asyncio
import json
from typing import Dict, List, Any, TypedDict, Annotated, Literal
from dataclasses import dataclass, field
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END


# ===================== 1. 状态定义 =====================

class ResearchState(TypedDict):
    """研究团队共享状态"""
    # 原始输入
    topic: str

    # 研究计划
    research_plan: str
    sub_tasks: List[str]

    # 研究阶段输出
    raw_research: str
    sources: List[str]

    # 分析阶段输出
    analysis_report: str
    key_insights: List[str]

    # 写作阶段输出
    draft_report: str
    final_report: str

    # 流程控制
    current_phase: str
    feedback: str
    revision_count: int
    max_revisions: int
    is_complete: bool

    # 执行日志
    execution_log: Annotated[List[str], operator.add]


# ===================== 2. Agent定义 =====================

class TeamAgent:
    """团队Agent基类"""

    def __init__(self, role: str, system_prompt: str,
                 model: str = "gpt-4o-mini"):
        self.role = role
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.7
        )
        self.system_prompt = system_prompt

    async def think(self, prompt: str) -> str:
        """思考并生成回复"""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        response = await self.llm.ainvoke(messages)
        return response.content


# 创建团队成员
coordinator = TeamAgent(
    role="Coordinator",
    system_prompt="""你是研究团队的协调者。你的职责是:
    1. 理解研究主题，制定研究计划
    2. 将研究任务分解为明确的子任务
    3. 审核研究成果的质量
    4. 提出改进建议或确认完成

    在制定研究计划时，请以JSON格式输出子任务列表。
    在审核时，如果质量达标，回复"APPROVED"；
    否则给出具体改进建议。""",
    model="gpt-4o-mini"
)

researcher = TeamAgent(
    role="Researcher",
    system_prompt="""你是研究团队的研究员。你的职责是:
    1. 根据研究计划搜集相关信息
    2. 整理和归纳原始资料
    3. 标注信息来源

    请提供详细、有条理的研究资料，
    包含数据、案例和关键发现。
    每个信息点都标注来源。""",
    model="gpt-4o-mini"
)

analyst = TeamAgent(
    role="Analyst",
    system_prompt="""你是研究团队的分析师。你的职责是:
    1. 分析研究员收集的资料
    2. 提取关键趋势和洞察
    3. 进行对比分析和数据解读
    4. 生成结构化的分析报告

    请使用数据支持你的分析结论，
    提供清晰的逻辑推理过程。""",
    model="gpt-4o-mini"
)

writer = TeamAgent(
    role="Writer",
    system_prompt="""你是研究团队的写作者。你的职责是:
    1. 基于分析结果撰写完整的研究报告
    2. 组织清晰的报告结构
    3. 使用准确的语言表达
    4. 添加引用和参考资料

    报告应包含:摘要、引言、主体分析、
    结论和建议、参考资料。
    语言要专业、客观、有说服力。""",
    model="gpt-4o-mini"
)


# ===================== 3. 节点函数 =====================

async def planning_node(state: ResearchState) -> dict:
    """规划节点 - Coordinator制定研究计划"""
    topic = state["topic"]

    prompt = f"""请为以下研究主题制定研究计划:

主题: {topic}

请输出:
1. 研究目标和范围
2. 研究方法
3. 子任务列表（JSON数组格式）

格式要求:
首先用自然语言描述研究计划，
然后在最后用```json标记输出子任务列表，如:
```json
["子任务1", "子任务2", "子任务3"]
```"""

    result = await coordinator.think(prompt)

    # 提取子任务列表
    sub_tasks = []
    if "```json" in result:
        json_str = result.split("```json")[1].split("```")[0].strip()
        try:
            sub_tasks = json.loads(json_str)
        except json.JSONDecodeError:
            sub_tasks = [f"研究{topic}的各个方面"]

    return {
        "research_plan": result,
        "sub_tasks": sub_tasks,
        "current_phase": "researching",
        "execution_log": [
            f"[Coordinator] 完成研究计划制定，"
            f"分解为{len(sub_tasks)}个子任务"
        ]
    }


async def research_node(state: ResearchState) -> dict:
    """研究节点 - Researcher搜集资料"""
    topic = state["topic"]
    plan = state["research_plan"]
    sub_tasks = state.get("sub_tasks", [])
    feedback = state.get("feedback", "")

    feedback_section = ""
    if feedback:
        feedback_section = f"\n\n协调者反馈（请据此改进）:\n{feedback}"

    prompt = f"""请根据以下研究计划收集资料:

研究主题: {topic}

研究计划:
{plan}

子任务:
{json.dumps(sub_tasks, ensure_ascii=False, indent=2)}
{feedback_section}

请针对每个子任务提供详细的研究资料，
包含具体的数据、案例、技术细节。
为每个信息标注来源（可模拟合理来源）。"""

    result = await researcher.think(prompt)

    # 模拟提取来源
    sources = [
        "学术论文和技术博客",
        "行业研究报告",
        "官方文档和GitHub仓库"
    ]

    return {
        "raw_research": result,
        "sources": sources,
        "current_phase": "analyzing",
        "execution_log": [
            f"[Researcher] 完成资料搜集，"
            f"来源: {len(sources)}个渠道"
        ]
    }


async def analysis_node(state: ResearchState) -> dict:
    """分析节点 - Analyst分析资料"""
    topic = state["topic"]
    research = state["raw_research"]
    feedback = state.get("feedback", "")

    feedback_section = ""
    if feedback:
        feedback_section = f"\n\n协调者反馈（请据此改进）:\n{feedback}"

    prompt = f"""请分析以下研究资料:

研究主题: {topic}

原始研究资料:
{research}
{feedback_section}

请提供:
1. 关键发现和趋势（请以列表形式列出关键洞察）
2. 数据分析和解读
3. 对比分析
4. 结论和预测

在分析末尾，请用```json标记输出关键洞察列表:
```json
["洞察1", "洞察2", "洞察3"]
```"""

    result = await analyst.think(prompt)

    # 提取关键洞察
    insights = []
    if "```json" in result:
        json_str = result.split("```json")[1].split("```")[0].strip()
        try:
            insights = json.loads(json_str)
        except json.JSONDecodeError:
            insights = ["分析完成，详见报告"]

    return {
        "analysis_report": result,
        "key_insights": insights,
        "current_phase": "writing",
        "execution_log": [
            f"[Analyst] 完成分析，提取{len(insights)}个关键洞察"
        ]
    }


async def writing_node(state: ResearchState) -> dict:
    """写作节点 - Writer撰写报告"""
    topic = state["topic"]
    research = state["raw_research"]
    analysis = state["analysis_report"]
    insights = state.get("key_insights", [])
    sources = state.get("sources", [])
    feedback = state.get("feedback", "")

    feedback_section = ""
    if feedback:
        feedback_section = f"\n\n协调者反馈（请据此修改报告）:\n{feedback}"

    prompt = f"""请基于以下资料撰写完整的研究报告:

研究主题: {topic}

研究资料摘要:
{research[:2000]}

分析报告:
{analysis[:2000]}

关键洞察:
{json.dumps(insights, ensure_ascii=False)}

信息来源:
{json.dumps(sources, ensure_ascii=False)}
{feedback_section}

请按以下结构撰写报告:

# {topic} - 研究报告

## 摘要
(200字以内的核心摘要)

## 1. 引言
(研究背景和目的)

## 2. 研究方法
(简述研究方法)

## 3. 主体分析
### 3.1 (第一个主要发现)
### 3.2 (第二个主要发现)
### 3.3 (第三个主要发现)

## 4. 结论与建议
(总结和建议)

## 5. 参考资料
(来源列表)"""

    result = await writer.think(prompt)

    return {
        "draft_report": result,
        "current_phase": "reviewing",
        "execution_log": [
            f"[Writer] 完成报告撰写，"
            f"约{len(result)}字"
        ]
    }


async def review_node(state: ResearchState) -> dict:
    """审核节点 - Coordinator审核报告"""
    topic = state["topic"]
    report = state["draft_report"]
    revision_count = state.get("revision_count", 0)

    prompt = f"""请审核以下研究报告:

研究主题: {topic}

报告内容:
{report}

这是第 {revision_count + 1} 版报告。

请评估以下方面:
1. 内容完整性：是否覆盖了主题的关键方面
2. 分析深度：分析是否有深度和洞察力
3. 逻辑性：论述是否逻辑清晰
4. 可读性：语言是否专业、流畅
5. 引用规范：来源是否标注完整

如果报告质量达标，请回复: APPROVED
如果需要改进，请给出具体的修改建议。"""

    result = await coordinator.think(prompt)

    is_approved = "APPROVED" in result.upper()

    return {
        "feedback": "" if is_approved else result,
        "final_report": report if is_approved else "",
        "revision_count": revision_count + 1,
        "is_complete": is_approved,
        "current_phase": "complete" if is_approved else "revising",
        "execution_log": [
            f"[Coordinator] 审核完成: "
            f"{'通过' if is_approved else '需要修改'} "
            f"(第{revision_count + 1}版)"
        ]
    }


# ===================== 4. 路由函数 =====================

def should_revise(state: ResearchState) -> Literal["revise", "complete"]:
    """判断是否需要修改"""
    if state.get("is_complete", False):
        return "complete"
    if state.get("revision_count", 0) >= state.get("max_revisions", 2):
        return "complete"  # 达到最大修改次数，强制完成
    return "revise"


# ===================== 5. 构建工作流 =====================

def build_research_team_graph() -> StateGraph:
    """构建AI研究团队工作流"""

    # 创建StateGraph
    workflow = StateGraph(ResearchState)

    # 添加节点
    workflow.add_node("planning", planning_node)
    workflow.add_node("researching", research_node)
    workflow.add_node("analyzing", analysis_node)
    workflow.add_node("writing", writing_node)
    workflow.add_node("reviewing", review_node)

    # 设置入口
    workflow.set_entry_point("planning")

    # 添加边
    workflow.add_edge("planning", "researching")
    workflow.add_edge("researching", "analyzing")
    workflow.add_edge("analyzing", "writing")
    workflow.add_edge("writing", "reviewing")

    # 条件路由：审核后决定是否修改
    workflow.add_conditional_edges(
        "reviewing",
        should_revise,
        {
            "revise": "researching",  # 需要修改，重新研究
            "complete": END           # 审核通过，结束
        }
    )

    return workflow.compile()


# ===================== 6. 运行团队 =====================

async def run_research_team(topic: str) -> str:
    """运行AI研究团队"""

    print(f"\n{'='*70}")
    print(f"  AI研究团队 - 开始研究")
    print(f"  主题: {topic}")
    print(f"{'='*70}")

    # 构建工作流
    app = build_research_team_graph()

    # 初始化状态
    initial_state = {
        "topic": topic,
        "research_plan": "",
        "sub_tasks": [],
        "raw_research": "",
        "sources": [],
        "analysis_report": "",
        "key_insights": [],
        "draft_report": "",
        "final_report": "",
        "current_phase": "planning",
        "feedback": "",
        "revision_count": 0,
        "max_revisions": 2,
        "is_complete": False,
        "execution_log": [f"[System] 开始研究: {topic}"]
    }

    # 执行工作流
    final_state = await app.ainvoke(initial_state)

    # 输出执行日志
    print(f"\n{'='*70}")
    print("  执行日志:")
    print(f"{'='*70}")
    for log in final_state["execution_log"]:
        print(f"  {log}")

    # 输出关键洞察
    print(f"\n{'='*70}")
    print("  关键洞察:")
    print(f"{'='*70}")
    for i, insight in enumerate(final_state.get("key_insights", []), 1):
        print(f"  {i}. {insight}")

    # 返回最终报告
    final_report = final_state.get("final_report") or \
                   final_state.get("draft_report", "未生成报告")

    print(f"\n{'='*70}")
    print("  最终报告:")
    print(f"{'='*70}")
    print(final_report)

    return final_report


# ===================== CrewAI 实现 (对比) =====================

def crewai_research_team_example():
    """
    CrewAI研究团队实现 (代码示例)
    pip install crewai crewai-tools
    """
    from crewai import Agent, Task, Crew, Process
    from crewai_tools import SerperDevTool

    # 定义Agent
    researcher_agent = Agent(
        role="高级研究员",
        goal="搜集关于{topic}的全面信息",
        backstory="你是一个经验丰富的研究员，擅长在互联网上"
                  "找到最相关和最准确的信息。",
        tools=[SerperDevTool()],
        verbose=True,
        allow_delegation=False
    )

    analyst_agent = Agent(
        role="数据分析师",
        goal="分析研究数据并提取关键洞察",
        backstory="你是一个严谨的数据分析师，擅长从数据中"
                  "发现模式和趋势。",
        verbose=True,
        allow_delegation=False
    )

    writer_agent = Agent(
        role="技术写作专家",
        goal="撰写高质量的研究报告",
        backstory="你是一个技术写作专家，擅长将复杂的技术"
                  "内容转化为清晰、有条理的报告。",
        verbose=True,
        allow_delegation=False
    )

    # 定义任务
    research_task = Task(
        description="搜集关于{topic}的详细信息，"
                    "包括最新进展、关键技术和应用场景。",
        expected_output="详细的研究资料，包含数据和来源。",
        agent=researcher_agent
    )

    analysis_task = Task(
        description="分析收集的研究资料，提取关键趋势和洞察。",
        expected_output="结构化的分析报告，包含趋势和预测。",
        agent=analyst_agent,
        context=[research_task]  # 依赖研究任务
    )

    writing_task = Task(
        description="基于分析结果撰写完整的研究报告。",
        expected_output="完整的Markdown格式研究报告。",
        agent=writer_agent,
        context=[research_task, analysis_task]  # 依赖前两个任务
    )

    # 创建团队
    crew = Crew(
        agents=[researcher_agent, analyst_agent, writer_agent],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,  # 顺序执行
        verbose=True
    )

    # 执行
    result = crew.kickoff(
        inputs={"topic": "AI Agent技术发展趋势"}
    )

    return result


# ===================== LangGraph多Agent路由 (补充) =============

def langgraph_multi_agent_router_example():
    """
    LangGraph多Agent路由模式 - Supervisor路由到Worker
    """
    from typing import Literal
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from langgraph.graph import StateGraph, END, MessagesState

    llm = ChatOpenAI(model="gpt-4o-mini")

    # 定义Worker节点
    async def researcher_node(state: MessagesState):
        """研究员节点"""
        system = SystemMessage(
            content="你是研究员，专门搜集和整理信息。"
        )
        messages = [system] + state["messages"]
        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    async def analyst_node(state: MessagesState):
        """分析师节点"""
        system = SystemMessage(
            content="你是分析师，专门分析数据和趋势。"
        )
        messages = [system] + state["messages"]
        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    async def writer_node(state: MessagesState):
        """写作者节点"""
        system = SystemMessage(
            content="你是写作者，专门撰写报告和文章。"
        )
        messages = [system] + state["messages"]
        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    # Supervisor路由
    async def supervisor_node(state: MessagesState):
        """Supervisor决定下一步"""
        system = SystemMessage(content="""你是团队主管。
根据当前对话，决定接下来应该让谁工作。
可选: researcher, analyst, writer, FINISH
只回复一个选项名称。""")
        messages = [system] + state["messages"]
        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    def route_supervisor(state: MessagesState) -> Literal[
        "researcher", "analyst", "writer", "end"
    ]:
        """路由函数"""
        last_msg = state["messages"][-1].content.strip().lower()
        if "researcher" in last_msg:
            return "researcher"
        elif "analyst" in last_msg:
            return "analyst"
        elif "writer" in last_msg:
            return "writer"
        else:
            return "end"

    # 构建图
    workflow = StateGraph(MessagesState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("supervisor")

    # Worker完成后回到Supervisor
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("analyst", "supervisor")
    workflow.add_edge("writer", "supervisor")

    # Supervisor条件路由
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "end": END
        }
    )

    return workflow.compile()


# ===================== 主程序 =====================

if __name__ == "__main__":
    # 运行LangGraph研究团队
    topic = "AI Agent技术在企业中的应用现状与未来趋势"
    asyncio.run(run_research_team(topic))
```

---

## 辩论模式完整实现

### 架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                    辩论模式 (Debate Pattern)                       │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐      │
│  │                      辩论主持人                          │      │
│  │  - 宣布辩题                                             │      │
│  │  - 控制轮次                                             │      │
│  │  - 引导讨论方向                                         │      │
│  └───────────────────────┬────────────────────────────────┘      │
│                           │                                      │
│         ┌─────────────────┼─────────────────┐                    │
│         │                 │                 │                    │
│         ▼                 ▼                 ▼                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   正方Agent   │  │   反方Agent   │  │  裁判Agent    │           │
│  │              │  │              │  │              │           │
│  │ 目标:论证    │  │ 目标:质疑    │  │ 目标:裁决    │           │
│  │ 支持观点    │  │ 反对观点    │  │ 综合评价    │           │
│  │ 提供证据    │  │ 找出漏洞    │  │ 给出结论    │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                  │
│  辩论流程:                                                       │
│  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐         │
│  │正方   │──►│反方   │──►│正方   │──►│反方   │──►│裁判   │         │
│  │立论   │   │质疑   │   │反驳   │   │总结   │   │裁决   │         │
│  └──────┘   └──────┘   └──────┘   └──────┘   └──────┘         │
│                                                                  │
│  优势: 多角度分析→更全面 | 对抗验证→更准确 | 裁判综合→高质量     │
└──────────────────────────────────────────────────────────────────┘
```

### 完整辩论Agent实现

```python
"""
辩论模式完整实现：正方、反方、裁判三Agent辩论
适用于决策分析、方案评估、内容审核等场景
"""
import asyncio
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from openai import AsyncOpenAI

client = AsyncOpenAI()


@dataclass
class DebateRound:
    """辩论轮次记录"""
    round_num: int
    pro_argument: str = ""
    con_argument: str = ""


@dataclass
class DebateResult:
    """辩论结果"""
    topic: str
    rounds: List[DebateRound]
    verdict: str
    winner: str
    confidence: float
    key_points: List[str]


class DebateAgent:
    """辩论Agent"""

    def __init__(self, name: str, role: str, stance: str, personality: str):
        self.name = name
        self.role = role
        self.stance = stance
        self.personality = personality

    async def argue(
        self, topic: str, debate_history: str, instruction: str
    ) -> str:
        """发表论点"""
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""你是{self.name}，角色是{self.role}。
你的立场: {self.stance}
你的风格: {self.personality}

规则:
1. 必须围绕辩题展开论述
2. 引用具体事实和数据来支持论点
3. 回应对方的论点，不能回避质疑
4. 论述控制在200字以内
5. 用中文回答"""
                },
                {
                    "role": "user",
                    "content": f"""辩题: {topic}

已有辩论记录:
{debate_history}

当前任务: {instruction}

请发表你的论述:"""
                }
            ],
            temperature=0.8,
            max_tokens=500
        )
        return response.choices[0].message.content


class JudgeAgent:
    """裁判Agent"""

    def __init__(self):
        self.name = "裁判"

    async def evaluate(
        self, topic: str, debate_history: str
    ) -> Dict:
        """评估辩论并给出裁决"""
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """你是一位公正的辩论裁判。
请综合评估双方的论述质量，给出裁决。

请以JSON格式返回:
{
    "winner": "正方/反方/平局",
    "confidence": 0.0-1.0,
    "verdict": "综合裁决说明(200字以内)",
    "pro_strengths": ["正方的优势论点"],
    "con_strengths": ["反方的优势论点"],
    "key_insights": ["辩论中产生的关键洞察"]
}"""
                },
                {
                    "role": "user",
                    "content": f"""辩题: {topic}

完整辩论记录:
{debate_history}

请给出你的裁决:"""
                }
            ],
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)


class DebateArena:
    """辩论场"""

    def __init__(self, topic: str, max_rounds: int = 3):
        self.topic = topic
        self.max_rounds = max_rounds

        self.pro_agent = DebateAgent(
            name="正方辩手",
            role="支持者",
            stance=f"支持「{topic}」",
            personality="逻辑严密、善于举例、数据导向"
        )

        self.con_agent = DebateAgent(
            name="反方辩手",
            role="质疑者",
            stance=f"反对「{topic}」",
            personality="善于质疑、寻找漏洞、辩证思考"
        )

        self.judge = JudgeAgent()
        self.rounds: List[DebateRound] = []
        self.debate_log: List[str] = []

    def _get_history(self) -> str:
        """获取辩论历史"""
        return "\n\n".join(self.debate_log)

    async def run(self) -> DebateResult:
        """运行完整辩论"""
        print(f"\n{'='*60}")
        print(f"  辩论主题: {self.topic}")
        print(f"  正方: {self.pro_agent.name}")
        print(f"  反方: {self.con_agent.name}")
        print(f"  轮次: {self.max_rounds}")
        print(f"{'='*60}")

        for round_num in range(1, self.max_rounds + 1):
            debate_round = DebateRound(round_num=round_num)
            print(f"\n{'─'*40}")
            print(f"  第 {round_num} 轮")
            print(f"{'─'*40}")

            # 正方发言
            if round_num == 1:
                instruction = "请进行开场立论，阐述你的核心观点和论据"
            else:
                instruction = "请反驳对方上轮的论点，并补充新的论据"

            print(f"\n[正方] 发言中...")
            pro_arg = await self.pro_agent.argue(
                self.topic, self._get_history(), instruction
            )
            debate_round.pro_argument = pro_arg
            self.debate_log.append(f"[第{round_num}轮-正方] {pro_arg}")
            print(f"  {pro_arg[:150]}...")

            # 反方发言
            if round_num == 1:
                instruction = "请针对正方的立论进行质疑和反驳"
            else:
                instruction = "请反驳正方的论点，并总结你的核心主张"

            print(f"\n[反方] 发言中...")
            con_arg = await self.con_agent.argue(
                self.topic, self._get_history(), instruction
            )
            debate_round.con_argument = con_arg
            self.debate_log.append(f"[第{round_num}轮-反方] {con_arg}")
            print(f"  {con_arg[:150]}...")

            self.rounds.append(debate_round)

        # 裁判评判
        print(f"\n{'='*40}")
        print(f"  裁判评判中...")
        print(f"{'='*40}")

        verdict_data = await self.judge.evaluate(
            self.topic, self._get_history()
        )

        result = DebateResult(
            topic=self.topic,
            rounds=self.rounds,
            verdict=verdict_data.get("verdict", ""),
            winner=verdict_data.get("winner", "平局"),
            confidence=verdict_data.get("confidence", 0.5),
            key_points=verdict_data.get("key_insights", [])
        )

        # 输出结果
        print(f"\n{'='*60}")
        print(f"  裁决结果")
        print(f"{'='*60}")
        print(f"  胜方: {result.winner} (信心度: {result.confidence})")
        print(f"  裁决: {result.verdict}")
        print(f"\n  关键洞察:")
        for i, point in enumerate(result.key_points, 1):
            print(f"    {i}. {point}")

        return result


# 使用示例
async def run_debate_demo():
    """运行辩论演示"""
    topics = [
        "AI应该在所有企业中全面替代人工客服",
        "开源大模型比商业闭源模型更适合企业应用",
        "多Agent系统比单Agent系统在大多数场景下更优",
    ]

    arena = DebateArena(
        topic=topics[0],
        max_rounds=2
    )
    result = await arena.run()
    return result


if __name__ == "__main__":
    asyncio.run(run_debate_demo())
```

---

## 多Agent框架对比

### 框架选择决策图

```
┌──────────────────────────────────────────────────────────────────┐
│              多Agent框架选择指南                                    │
│                                                                  │
│  你的需求是什么？                                                 │
│       │                                                          │
│       ├── 需要精确控制执行流程？                                  │
│       │   └── 是 → LangGraph                                    │
│       │         优势: 图结构显式控制、checkpoint、HITL            │
│       │                                                          │
│       ├── 快速搭建Agent团队？                                    │
│       │   └── 是 → CrewAI                                       │
│       │         优势: 角色定义直观、任务依赖、开箱即用            │
│       │                                                          │
│       ├── 需要群聊式多Agent对话？                                │
│       │   └── 是 → AutoGen                                      │
│       │         优势: GroupChat、代码执行、灵活的发言策略         │
│       │                                                          │
│       └── 需要高度定制化？                                       │
│           └── 是 → 自建框架（基于OpenAI/Claude API）             │
│                 优势: 完全控制、无框架依赖、最轻量               │
│                                                                  │
│  详细对比:                                                       │
│  ┌──────────┬──────────┬──────────┬──────────┬─────────┐        │
│  │ 特性     │LangGraph │ CrewAI   │ AutoGen  │ 自建    │        │
│  ├──────────┼──────────┼──────────┼──────────┼─────────┤        │
│  │学习曲线  │  中等    │   低     │   中等   │  高     │        │
│  │灵活性    │  极高    │   中等   │   高     │  极高   │        │
│  │社区生态  │  大      │   中     │   大     │  -      │        │
│  │生产就绪  │  高      │   中     │   中     │  取决   │        │
│  │流程控制  │  显式图  │   任务链 │   群聊   │  完全   │        │
│  │状态管理  │  内置    │   有限   │   有限   │  自建   │        │
│  │可视化    │  Mermaid │   日志   │   日志   │  自建   │        │
│  │HITL      │  原生    │   回调   │   有限   │  自建   │        │
│  │Checkpoint│  原生    │   无     │   无     │  自建   │        │
│  └──────────┴──────────┴──────────┴──────────┴─────────┘        │
└──────────────────────────────────────────────────────────────────┘
```

### CrewAI深入示例

```python
"""
CrewAI 完整示例：产品研发团队
包含：产品经理、技术架构师、开发工程师、测试工程师
"""
# pip install crewai crewai-tools

from crewai import Agent, Task, Crew, Process

# ============================================================
# 1. 定义Agent（团队成员）
# ============================================================

product_manager = Agent(
    role="产品经理",
    goal="定义产品需求和优先级，确保产品方向正确",
    backstory="""你是一位经验丰富的产品经理，拥有10年互联网产品经验。
你擅长理解用户需求、定义产品功能、制定产品路线图。
你的决策基于数据和用户反馈，而非主观臆断。""",
    verbose=True,
    allow_delegation=True  # 允许委派任务给其他Agent
)

tech_architect = Agent(
    role="技术架构师",
    goal="设计稳健、可扩展的技术架构方案",
    backstory="""你是一位资深技术架构师，精通分布式系统、微服务、云原生架构。
你关注系统的可扩展性、可维护性和性能。
你会考虑技术选型的长期影响和团队的技术储备。""",
    verbose=True,
    allow_delegation=False
)

developer = Agent(
    role="高级开发工程师",
    goal="编写高质量的代码实现，确保功能正确实现",
    backstory="""你是一位全栈开发工程师，精通Python、JavaScript、Go等语言。
你遵循SOLID原则和Clean Code实践。
你注重代码的可读性、可测试性和性能。""",
    verbose=True,
    allow_delegation=False
)

qa_engineer = Agent(
    role="测试工程师",
    goal="确保产品质量，找出所有潜在问题",
    backstory="""你是一位严谨的QA工程师，擅长编写测试用例和自动化测试。
你善于从用户角度思考，找出产品中的问题和改进点。
你不仅测试功能正确性，还关注性能、安全和用户体验。""",
    verbose=True,
    allow_delegation=False
)


# ============================================================
# 2. 定义任务（Task）
# ============================================================

requirement_task = Task(
    description="""为以下产品需求编写详细的PRD（产品需求文档）:

需求: 为AI Agent管理平台设计一个多Agent协作调试功能。
用户可以：
1. 可视化Agent间的消息流转
2. 在任意节点设置断点
3. 回放历史执行记录
4. 修改中间状态并重新执行

请输出详细的功能描述、用户故事和验收标准。""",
    expected_output="完整的PRD文档，包含功能描述、用户故事和验收标准",
    agent=product_manager
)

architecture_task = Task(
    description="""基于产品需求文档，设计技术架构方案。
包括:
1. 整体架构图（文本描述）
2. 核心模块划分
3. 技术选型建议
4. 数据流设计
5. 关键技术难点和解决方案""",
    expected_output="完整的技术架构文档",
    agent=tech_architect,
    context=[requirement_task]  # 依赖需求任务
)

implementation_task = Task(
    description="""基于技术架构方案，编写核心模块的伪代码实现。
重点实现:
1. Agent消息追踪器
2. 断点管理器
3. 状态快照/恢复
请用Python编写，包含完整的类定义和关键方法。""",
    expected_output="核心模块的Python伪代码实现",
    agent=developer,
    context=[requirement_task, architecture_task]
)

testing_task = Task(
    description="""为核心模块编写测试方案。
包括:
1. 单元测试用例设计
2. 集成测试场景
3. 性能测试指标
4. 边界条件和异常测试""",
    expected_output="完整的测试方案文档",
    agent=qa_engineer,
    context=[requirement_task, implementation_task]
)


# ============================================================
# 3. 组建团队并执行
# ============================================================

crew = Crew(
    agents=[product_manager, tech_architect, developer, qa_engineer],
    tasks=[requirement_task, architecture_task,
           implementation_task, testing_task],
    process=Process.sequential,  # 顺序执行（有依赖关系）
    verbose=True,
    memory=True,                 # 启用记忆
    max_rpm=10                   # 每分钟最大请求数
)


def run_crew():
    """运行团队"""
    result = crew.kickoff()
    print(f"\n{'='*60}")
    print("团队协作完成！")
    print(f"{'='*60}")
    print(result)
    return result


# if __name__ == "__main__":
#     run_crew()
```

---

## 多Agent系统生产部署

### 生产架构设计

```
┌──────────────────────────────────────────────────────────────────┐
│              多Agent系统 生产部署架构                                │
│                                                                  │
│  ┌──────────────┐                                                │
│  │   API Gateway │                                                │
│  │   (认证/限流) │                                                │
│  └──────┬───────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────┐        │
│  │            Agent Orchestrator (协调器)                 │        │
│  │                                                      │        │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐             │        │
│  │  │任务队列   │ │状态管理   │ │日志追踪   │             │        │
│  │  │(Redis)   │ │(Redis)   │ │(ELK)     │             │        │
│  │  └──────────┘ └──────────┘ └──────────┘             │        │
│  └───────┬──────────────┬──────────────┬────────────────┘        │
│          │              │              │                          │
│     ┌────▼────┐   ┌────▼────┐   ┌────▼────┐                    │
│     │Agent Pod│   │Agent Pod│   │Agent Pod│   ← K8s部署         │
│     │         │   │         │   │         │                      │
│     │Researcher│  │Analyst │   │Writer   │                      │
│     │         │   │         │   │         │                      │
│     │ LLM API │   │ LLM API│   │ LLM API │                      │
│     │ Tools   │   │ Tools  │   │ Tools   │                      │
│     └─────────┘   └─────────┘   └─────────┘                    │
│          │              │              │                          │
│          └──────────────┼──────────────┘                         │
│                         │                                        │
│  ┌──────────────────────▼───────────────────────────────┐        │
│  │              共享基础设施层                             │        │
│  │                                                      │        │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │        │
│  │  │向量数据库 │ │关系数据库 │ │对象存储   │ │消息队列 │  │        │
│  │  │(Pinecone)│ │(Postgres)│ │(S3/OSS) │ │(Kafka) │  │        │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────┘  │        │
│  └──────────────────────────────────────────────────────┘        │
│                                                                  │
│  监控告警层:                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐           │
│  │Prometheus│ │ Grafana  │ │LangSmith │ │ PagerDuty │           │
│  │ 指标收集  │ │ 可视化   │ │ LLM追踪  │ │ 告警通知  │           │
│  └──────────┘ └──────────┘ └──────────┘ └───────────┘           │
└──────────────────────────────────────────────────────────────────┘
```

### 生产级多Agent成本控制

```python
"""
多Agent系统成本控制器
"""
import time
from typing import Dict
from dataclasses import dataclass, field


@dataclass
class AgentCostRecord:
    """Agent成本记录"""
    agent_name: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    total_cost_usd: float = 0.0


class CostController:
    """
    多Agent系统成本控制器
    功能:
    - 按Agent统计Token消耗
    - 设置预算上限
    - 成本超限告警
    - 模型降级策略
    """

    PRICING = {
        "gpt-4o": {"input": 2.5, "output": 10.0},        # per 1M tokens
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    }

    def __init__(self, budget_usd: float = 10.0):
        self.budget_usd = budget_usd
        self.agents: Dict[str, AgentCostRecord] = {}
        self.total_cost = 0.0
        self.budget_warnings = []

    def record_usage(
        self, agent_name: str, model: str,
        input_tokens: int, output_tokens: int
    ):
        """记录Token使用"""
        if agent_name not in self.agents:
            self.agents[agent_name] = AgentCostRecord(agent_name=agent_name)

        record = self.agents[agent_name]
        record.total_input_tokens += input_tokens
        record.total_output_tokens += output_tokens
        record.total_calls += 1

        # 计算成本
        pricing = self.PRICING.get(model, {"input": 1.0, "output": 3.0})
        cost = (
            input_tokens * pricing["input"] / 1_000_000 +
            output_tokens * pricing["output"] / 1_000_000
        )
        record.total_cost_usd += cost
        self.total_cost += cost

        # 检查预算
        if self.total_cost > self.budget_usd * 0.8:
            self.budget_warnings.append(
                f"警告: 已使用预算的{self.total_cost/self.budget_usd*100:.1f}%"
            )

    def should_downgrade(self) -> bool:
        """是否应该降级模型"""
        return self.total_cost > self.budget_usd * 0.7

    def get_recommended_model(self, default: str = "gpt-4o") -> str:
        """获取推荐模型（根据预算）"""
        if self.should_downgrade():
            return "gpt-4o-mini"
        return default

    def is_over_budget(self) -> bool:
        """是否超出预算"""
        return self.total_cost >= self.budget_usd

    def get_report(self) -> dict:
        """生成成本报告"""
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "budget_usd": self.budget_usd,
            "budget_used_pct": f"{self.total_cost/self.budget_usd*100:.1f}%",
            "over_budget": self.is_over_budget(),
            "agents": {
                name: {
                    "calls": r.total_calls,
                    "input_tokens": r.total_input_tokens,
                    "output_tokens": r.total_output_tokens,
                    "cost_usd": round(r.total_cost_usd, 4)
                }
                for name, r in self.agents.items()
            }
        }

    def print_report(self):
        """打印成本报告"""
        report = self.get_report()
        print(f"\n{'='*50}")
        print(f"  多Agent系统成本报告")
        print(f"{'='*50}")
        print(f"  总成本: ${report['total_cost_usd']}")
        print(f"  预算: ${report['budget_usd']}")
        print(f"  使用率: {report['budget_used_pct']}")
        print(f"  超预算: {'是' if report['over_budget'] else '否'}")
        print(f"\n  Agent明细:")
        for name, data in report["agents"].items():
            print(
                f"    {name}: "
                f"{data['calls']}次调用, "
                f"{data['input_tokens']+data['output_tokens']}tokens, "
                f"${data['cost_usd']}"
            )
        print(f"{'='*50}")


# 使用示例
cost_ctrl = CostController(budget_usd=5.0)

# 模拟Agent调用
cost_ctrl.record_usage("researcher", "gpt-4o", 2000, 1000)
cost_ctrl.record_usage("analyst", "gpt-4o", 1500, 800)
cost_ctrl.record_usage("writer", "gpt-4o-mini", 3000, 2000)

# 检查是否需要降级
print(f"推荐模型: {cost_ctrl.get_recommended_model()}")

# 打印报告
cost_ctrl.print_report()
```

---

## 总结

本教程涵盖了多Agent协作系统的核心内容:

1. **多Agent概述**: 多Agent系统通过多个专业化智能体的分工协作，克服了单一Agent的能力边界、上下文限制和可靠性问题，是构建复杂AI应用的重要架构模式。

2. **协作模式**: 四种核心协作模式各有优劣——层级模式适合明确的任务分解，对等模式适合协商决策，辩论模式保证输出质量，流水线模式适合有明确步骤的任务。应根据任务特征选择合适的模式。

3. **层级结构**: Supervisor Pattern是最常用的多Agent模式，Supervisor负责任务分解和结果汇总，Worker Agent负责具体执行。通过并行执行子任务可以显著提升效率。

4. **对等协作**: Peer-to-Peer模式通过提议-讨论-投票的流程达成共识，适合需要多角度分析的决策类任务。投票策略包括多数决、一致同意、加权投票等。

5. **共享记忆**: 三种记忆类型（短期/长期/工作记忆）配合访问控制机制，确保Agent间高效地共享信息同时保护数据安全。

6. **AutoGen GroupChat**: 微软AutoGen框架通过GroupChat机制让多Agent协作变得简单，支持多种Speaker选择策略和自定义工具集成。

7. **实战AI研究团队**: 基于LangGraph实现的完整研究团队，包含Coordinator、Researcher、Analyst、Writer四个角色，通过流水线模式协作完成研究任务，并支持审核-修改的迭代循环。

## 最佳实践

1. **从简单开始**: 先用2-3个Agent验证核心流程，再逐步增加Agent数量和复杂度
2. **明确职责边界**: 每个Agent的职责应该清晰且不重叠，避免角色混乱
3. **设计退出条件**: 避免无限循环，设置最大轮数、超时时间等终止条件
4. **实现错误处理**: 单个Agent失败不应导致整个系统崩溃，需要有降级策略
5. **监控和日志**: 记录详细的执行日志，方便调试和优化
6. **控制成本**: 多Agent意味着多次LLM调用，注意控制调用次数和Token使用量
7. **测试协作流程**: 不仅测试单个Agent，还要测试Agent间的协作是否顺畅
8. **共享记忆设计**: 合理设计共享记忆的结构和权限，避免信息冲突

## 参考资源

- [AutoGen 官方文档](https://microsoft.github.io/autogen/)
- [LangGraph Multi-Agent 指南](https://langchain-ai.github.io/langgraph/)
- [CrewAI 文档](https://docs.crewai.com/)
- [Multi-Agent Systems 论文综述](https://arxiv.org/abs/2308.08155)
- [LangChain Multi-Agent 教程](https://python.langchain.com/docs/use_cases/multi_agent)

---

**文件大小目标**: 30-35KB
**创建时间**: 2024-01-15
**最后更新**: 2024-01-15
