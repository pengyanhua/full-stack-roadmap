# AutoGen多Agent框架教程

## 目录
1. [AutoGen简介](#autogen简介)
2. [ConversableAgent](#conversableagent)
3. [GroupChat模式](#groupchat模式)
4. [代码执行Agent](#代码执行agent)
5. [完整多Agent协作示例](#完整多agent协作示例)

---

## AutoGen简介

### 什么是AutoGen

AutoGen是微软开发的多Agent对话框架，支持Agent之间的自动对话和协作。

### 安装

```bash
pip install pyautogen
```

### 核心概念

```
┌───────────────────────────────────────────────┐
│          AutoGen多Agent架构                    │
├───────────────────────────────────────────────┤
│                                               │
│  ┌──────────┐      ┌──────────┐              │
│  │ UserProxy│ ←──→ │Assistant │              │
│  │  Agent   │      │  Agent   │              │
│  └──────────┘      └──────────┘              │
│                                               │
│  ┌──────────────────────────────────────┐    │
│  │        GroupChat                      │    │
│  │  ┌────┐  ┌────┐  ┌────┐  ┌────┐     │    │
│  │  │ A1 │  │ A2 │  │ A3 │  │ A4 │     │    │
│  │  └────┘  └────┘  └────┘  └────┘     │    │
│  └──────────────────────────────────────┘    │
│                                               │
└───────────────────────────────────────────────┘
```

---

## ConversableAgent

### 基础对话

```python
from autogen import ConversableAgent

# 配置LLM
llm_config = {
    "model": "gpt-4",
    "api_key": "your-api-key"
}

# 创建Agent
assistant = ConversableAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="你是一个有帮助的AI助手。"
)

user_proxy = ConversableAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    llm_config=False
)

# 开始对话
user_proxy.initiate_chat(
    assistant,
    message="写一个Python函数计算斐波那契数列"
)
```

### 带代码执行的Agent

```python
from autogen import AssistantAgent, UserProxyAgent

# 助手Agent
assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config
)

# 用户代理(可执行代码)
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False
    }
)

# 对话
user_proxy.initiate_chat(
    assistant,
    message="绘制y=x^2在[-10,10]的图像"
)
```

---

## GroupChat模式

### 创建GroupChat

```python
from autogen import GroupChat, GroupChatManager

# 创建多个Agent
engineer = AssistantAgent(
    name="engineer",
    llm_config=llm_config,
    system_message="你是一个软件工程师，负责编写代码。"
)

product_manager = AssistantAgent(
    name="product_manager",
    llm_config=llm_config,
    system_message="你是产品经理，负责需求分析和规划。"
)

qa_tester = AssistantAgent(
    name="qa_tester",
    llm_config=llm_config,
    system_message="你是QA测试工程师，负责测试和质量保证。"
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    code_execution_config={"work_dir": "coding"}
)

# 创建GroupChat
groupchat = GroupChat(
    agents=[user_proxy, engineer, product_manager, qa_tester],
    messages=[],
    max_round=20
)

# 创建管理器
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# 开始讨论
user_proxy.initiate_chat(
    manager,
    message="开发一个待办事项应用"
)
```

---

## 完整多Agent协作示例

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

class AITeam:
    """AI团队协作系统"""

    def __init__(self, api_key):
        self.llm_config = {
            "model": "gpt-4",
            "api_key": api_key,
            "temperature": 0.7
        }

        # 创建团队成员
        self.researcher = AssistantAgent(
            name="researcher",
            llm_config=self.llm_config,
            system_message="""你是一个研究员，负责:
1. 收集和分析信息
2. 提供技术调研报告
3. 给出技术建议
"""
        )

        self.developer = AssistantAgent(
            name="developer",
            llm_config=self.llm_config,
            system_message="""你是开发工程师，负责:
1. 编写高质量代码
2. 实现功能需求
3. 代码注释和文档
"""
        )

        self.reviewer = AssistantAgent(
            name="reviewer",
            llm_config=self.llm_config,
            system_message="""你是代码审查员，负责:
1. 审查代码质量
2. 指出潜在问题
3. 提供改进建议
"""
        )

        self.user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="TERMINATE",
            max_consecutive_auto_reply=10,
            code_execution_config={
                "work_dir": "workspace",
                "use_docker": False
            }
        )

    def start_project(self, task):
        """开始项目"""
        # 创建GroupChat
        groupchat = GroupChat(
            agents=[
                self.user_proxy,
                self.researcher,
                self.developer,
                self.reviewer
            ],
            messages=[],
            max_round=30
        )

        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=self.llm_config
        )

        # 启动对话
        self.user_proxy.initiate_chat(
            manager,
            message=task
        )

# 使用示例
if __name__ == "__main__":
    team = AITeam(api_key="your-api-key")

    team.start_project(
        "创建一个Python脚本，读取CSV文件并生成数据统计报告"
    )
```

---

## 总结

AutoGen提供了强大的多Agent协作能力:

1. **简单易用**: API设计直观
2. **自动对话**: Agent之间自动交互
3. **代码执行**: 内置代码执行能力
4. **多Agent**: 支持复杂的团队协作

## 参考资源

- [AutoGen文档](https://microsoft.github.io/autogen/)
- [GitHub](https://github.com/microsoft/autogen)
