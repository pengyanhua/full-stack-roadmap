# Semantic Kernel完整教程

## 目录
1. [Semantic Kernel简介](#semantic-kernel简介)
2. [核心架构](#核心架构)
3. [Planner规划器](#planner规划器)
4. [Plugin插件系统](#plugin插件系统)
5. [Memory Connector](#memory-connector)
6. [与LangChain对比](#与langchain对比)

---

## Semantic Kernel简介

### 什么是Semantic Kernel

Semantic Kernel是微软开发的轻量级SDK，支持C#、Python和Java，用于将LLM集成到应用中。

### 安装

```bash
# Python
pip install semantic-kernel

# C#
dotnet add package Microsoft.SemanticKernel
```

### 核心概念

```
┌─────────────────────────────────────────────┐
│       Semantic Kernel架构                    │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │         Kernel (内核)                │   │
│  │  • 管理所有组件                      │   │
│  │  • 协调执行流程                      │   │
│  └───────┬─────────────────────────────┘   │
│          │                                  │
│    ┌─────┴─────┬──────────┬──────────┐    │
│    │           │          │          │    │
│  ┌─▼─┐     ┌──▼──┐   ┌───▼──┐   ┌──▼──┐  │
│  │LLM│     │Plugin│   │Memory│   │Plan │  │
│  │模型│     │插件  │   │记忆  │   │规划 │  │
│  └───┘     └─────┘   └──────┘   └─────┘  │
│                                             │
└─────────────────────────────────────────────┘
```

---

## 核心架构

### Python示例

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

# 创建Kernel
kernel = sk.Kernel()

# 添加LLM服务
kernel.add_service(
    OpenAIChatCompletion(
        service_id="gpt-4",
        ai_model_id="gpt-4"
    )
)

# 使用Kernel
result = await kernel.invoke_prompt("写一首诗")
print(result)
```

### C#示例

```csharp
using Microsoft.SemanticKernel;

// 创建Kernel
var kernel = Kernel.CreateBuilder()
    .AddOpenAIChatCompletion("gpt-4", "your-api-key")
    .Build();

// 调用
var result = await kernel.InvokePromptAsync("什么是AI?");
Console.WriteLine(result);
```

---

## Plugin插件系统

### 创建Plugin (Python)

```python
from semantic_kernel.functions import kernel_function

class MathPlugin:
    """数学插件"""

    @kernel_function(
        name="add",
        description="加法运算"
    )
    def add(self, a: int, b: int) -> int:
        return a + b

    @kernel_function(
        name="multiply",
        description="乘法运算"
    )
    def multiply(self, a: int, b: int) -> int:
        return a * b

# 注册Plugin
kernel.add_plugin(MathPlugin(), plugin_name="Math")

# 使用
result = await kernel.invoke_function(
    plugin_name="Math",
    function_name="add",
    a=5,
    b=3
)
print(result)  # 8
```

### Semantic Function

```python
from semantic_kernel.functions import KernelArguments

# 定义Semantic Function
translate_function = kernel.create_function_from_prompt(
    function_name="translate",
    plugin_name="Language",
    prompt="""
    将以下文本翻译成{{$language}}:
    {{$text}}
    """,
    description="翻译文本"
)

# 调用
result = await kernel.invoke(
    translate_function,
    KernelArguments(text="Hello World", language="中文")
)
print(result)
```

---

## Planner规划器

### Sequential Planner

```python
from semantic_kernel.planners import SequentialPlanner

# 创建规划器
planner = SequentialPlanner(kernel)

# 规划任务
plan = await planner.create_plan("查找Python的最新版本，然后总结它的新特性")

# 执行计划
result = await plan.invoke(kernel)
print(result)
```

---

## Memory Connector

```python
from semantic_kernel.memory import SemanticTextMemory
from semantic_kernel.connectors.memory.chroma import ChromaMemoryStore

# 创建Memory
memory_store = ChromaMemoryStore(persist_directory="./chroma_db")
memory = SemanticTextMemory(storage=memory_store)

# 保存记忆
await memory.save(
    collection="facts",
    id="fact1",
    text="Python是一种编程语言"
)

# 搜索记忆
results = await memory.search(
    collection="facts",
    query="什么是Python",
    limit=3
)
```

---

## 与LangChain对比

```
┌──────────────────────────────────────────────┐
│    Semantic Kernel vs LangChain               │
├──────────────────────────────────────────────┤
│                                              │
│  特性         Semantic Kernel    LangChain   │
│  ────────────────────────────────────────   │
│  语言支持     C#/Python/Java      Python     │
│  企业级       ✓                  部分         │
│  易用性       高                 中           │
│  生态系统     Microsoft生态      开源社区     │
│  最适合       .NET应用           Python应用   │
│                                              │
└──────────────────────────────────────────────┘
```

## 参考资源

- [Semantic Kernel文档](https://learn.microsoft.com/en-us/semantic-kernel/)
- [GitHub](https://github.com/microsoft/semantic-kernel)
