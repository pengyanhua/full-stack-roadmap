# LLM基础与API使用完整教程

## 目录
1. [LLM工作原理](#llm工作原理)
2. [Transformer架构简介](#transformer架构简介)
3. [OpenAI API完整教程](#openai-api完整教程)
4. [Claude API使用](#claude-api使用)
5. [Gemini API](#gemini-api)
6. [流式输出](#流式输出)
7. [Token计算与成本优化](#token计算与成本优化)
8. [完整代码示例](#完整代码示例)

---

## LLM工作原理

### 什么是LLM

大型语言模型(Large Language Model, LLM)是基于深度学习的自然语言处理模型，通过在海量文本数据上进行训练，学习语言的统计规律和语义表示。

### LLM的核心能力

```
┌─────────────────────────────────────────────────────────┐
│                    LLM核心能力图谱                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │  文本生成 │    │  语言理解 │    │  知识推理 │         │
│  │          │    │          │    │          │         │
│  │  续写    │    │  分类    │    │  逻辑推断 │         │
│  │  摘要    │    │  提取    │    │  常识推理 │         │
│  │  翻译    │    │  问答    │    │  数学计算 │         │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘         │
│       │               │               │               │
│       └───────────┬───┴───────────────┘               │
│                   │                                   │
│            ┌──────▼──────┐                           │
│            │  Transformer │                           │
│            │   架构核心    │                           │
│            └─────────────┘                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Transformer架构简介

### 架构总览

```
输入文本: "Hello, how are you?"
    │
    ▼
┌─────────────────────────────────────────┐
│      Tokenization (分词)                 │
│   ["Hello", ",", "how", "are", "you"]   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Token Embedding (词嵌入)            │
│   [0.2, 0.5, ...] [0.1, 0.3, ...]       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Position Encoding (位置编码)           │
│   添加位置信息到每个token                 │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Transformer Blocks (N层)               │
│                                          │
│  ┌────────────────────────────────┐     │
│  │  Self-Attention (自注意力)      │     │
│  │  计算token之间的关联             │     │
│  └────────────┬───────────────────┘     │
│               │                         │
│               ▼                         │
│  ┌────────────────────────────────┐     │
│  │  Feed Forward (前馈网络)         │     │
│  │  非线性转换                      │     │
│  └────────────────────────────────┘     │
│                                          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Output Layer (输出层)                  │
│   预测下一个token的概率分布               │
└──────────────┬──────────────────────────┘
               │
               ▼
输出文本: "I'm doing well, thanks!"
```

### Self-Attention机制

```python
# Self-Attention核心计算
def self_attention(Q, K, V, d_k):
    """
    Q: Query矩阵 (查询)
    K: Key矩阵 (键)
    V: Value矩阵 (值)
    d_k: Key的维度
    """
    # 计算注意力分数
    scores = Q @ K.T / math.sqrt(d_k)

    # Softmax归一化
    attention_weights = softmax(scores)

    # 加权求和
    output = attention_weights @ V

    return output
```

---

## OpenAI API完整教程

### API密钥配置

```python
import os
from openai import OpenAI

# 方法1: 环境变量
os.environ["OPENAI_API_KEY"] = "sk-your-api-key"
client = OpenAI()

# 方法2: 直接传入
client = OpenAI(api_key="sk-your-api-key")

# 方法3: 配置文件
from dotenv import load_dotenv
load_dotenv()  # 从.env文件加载
client = OpenAI()
```

### 基础对话示例

```python
from openai import OpenAI

client = OpenAI()

# 简单对话
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "你是一个专业的Python编程助手。"},
        {"role": "user", "content": "如何读取CSV文件？"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### 多轮对话管理

```python
class ChatSession:
    """OpenAI多轮对话管理器"""

    def __init__(self, model="gpt-4", system_prompt=None):
        self.client = OpenAI()
        self.model = model
        self.messages = []

        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": system_prompt
            })

    def chat(self, user_message, temperature=0.7, max_tokens=1000):
        """发送消息并获取回复"""
        # 添加用户消息
        self.messages.append({
            "role": "user",
            "content": user_message
        })

        # 调用API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # 提取助手回复
        assistant_message = response.choices[0].message.content

        # 添加到历史
        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message

    def clear_history(self, keep_system=True):
        """清空对话历史"""
        if keep_system and self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []

# 使用示例
session = ChatSession(
    model="gpt-4",
    system_prompt="你是一个友好的AI助手。"
)

# 多轮对话
print(session.chat("你好！"))
print(session.chat("我的名字是张三"))
print(session.chat("我叫什么名字？"))  # 能记住上下文
```

### 参数详解

```python
response = client.chat.completions.create(
    model="gpt-4",  # 模型选择
    messages=[...],  # 对话历史

    # 温度参数: 控制随机性 (0.0-2.0)
    # 0.0: 完全确定性，输出最可能的token
    # 1.0: 平衡创造力和一致性
    # 2.0: 高度随机和创造性
    temperature=0.7,

    # Top P采样: 核采样 (0.0-1.0)
    # 只考虑累积概率达到top_p的token
    top_p=1.0,

    # 频率惩罚: 减少重复 (-2.0-2.0)
    frequency_penalty=0.0,

    # 存在惩罚: 鼓励新话题 (-2.0-2.0)
    presence_penalty=0.0,

    # 最大token数
    max_tokens=1000,

    # 停止序列
    stop=["\n\n", "END"],

    # 返回多个结果
    n=1,

    # 流式输出
    stream=False
)
```

### 函数调用(Function Calling)

```python
import json

# 定义工具函数
def get_weather(location, unit="celsius"):
    """获取天气信息(模拟)"""
    return {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "description": "晴天"
    }

# 函数定义
functions = [
    {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名称，例如：北京、上海"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度单位"
                }
            },
            "required": ["location"]
        }
    }
]

# 调用API
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "北京今天天气怎么样？"}
    ],
    functions=functions,
    function_call="auto"
)

message = response.choices[0].message

# 检查是否需要调用函数
if message.function_call:
    function_name = message.function_call.name
    function_args = json.loads(message.function_call.arguments)

    # 执行函数
    if function_name == "get_weather":
        result = get_weather(**function_args)

        # 将结果返回给模型
        second_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "北京今天天气怎么样？"},
                message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(result)
                }
            ]
        )

        print(second_response.choices[0].message.content)
```

---

## Claude API使用

### 基础配置

```python
import anthropic

# 初始化客户端
client = anthropic.Anthropic(
    api_key="sk-ant-your-api-key"
)

# 基础对话
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)

print(response.content[0].text)
```

### Claude特色功能

```python
class ClaudeChat:
    """Claude对话管理器"""

    def __init__(self, model="claude-3-opus-20240229"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.messages = []

    def chat(self, user_message, system_prompt=None, max_tokens=2000):
        """发送消息"""
        # 添加用户消息
        self.messages.append({
            "role": "user",
            "content": user_message
        })

        # 构建请求
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": self.messages
        }

        # 添加系统提示(Claude的系统提示是独立参数)
        if system_prompt:
            kwargs["system"] = system_prompt

        # 调用API
        response = self.client.messages.create(**kwargs)

        # 提取回复
        assistant_message = response.content[0].text

        # 添加到历史
        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message

    def chat_with_thinking(self, user_message):
        """使用思考标签的对话"""
        prompt = f"""在回答之前，请在<thinking>标签中展示你的思考过程。

用户问题: {user_message}

请按以下格式回答:
<thinking>
1. 分析问题...
2. 考虑方案...
3. 得出结论...
</thinking>

<answer>
最终答案...
</answer>"""

        return self.chat(prompt)

# 使用示例
claude = ClaudeChat()
response = claude.chat_with_thinking("如何优化Python代码性能？")
print(response)
```

### Tool Use (Claude的函数调用)

```python
# 定义工具
tools = [
    {
        "name": "calculator",
        "description": "执行基础数学计算",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，例如: 2 + 2"
                }
            },
            "required": ["expression"]
        }
    }
]

# 调用
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    tools=tools,
    messages=[
        {"role": "user", "content": "125乘以37等于多少？"}
    ]
)

# 处理工具调用
if response.stop_reason == "tool_use":
    tool_use = next(block for block in response.content if block.type == "tool_use")
    tool_name = tool_use.name
    tool_input = tool_use.input

    # 执行工具
    if tool_name == "calculator":
        result = eval(tool_input["expression"])

        # 返回结果
        final_response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            tools=tools,
            messages=[
                {"role": "user", "content": "125乘以37等于多少？"},
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": str(result)
                        }
                    ]
                }
            ]
        )

        print(final_response.content[0].text)
```

---

## Gemini API

### 基础使用

```python
import google.generativeai as genai

# 配置API密钥
genai.configure(api_key="your-api-key")

# 创建模型
model = genai.GenerativeModel('gemini-pro')

# 简单对话
response = model.generate_content("什么是机器学习？")
print(response.text)

# 多轮对话
chat = model.start_chat(history=[])

response1 = chat.send_message("你好！")
print(response1.text)

response2 = chat.send_message("能介绍一下Python吗？")
print(response2.text)

# 查看历史
print(chat.history)
```

### Gemini Vision (图像理解)

```python
from PIL import Image

# 创建视觉模型
vision_model = genai.GenerativeModel('gemini-pro-vision')

# 加载图像
image = Image.open('example.jpg')

# 图像问答
response = vision_model.generate_content([
    "这张图片里有什么？请详细描述。",
    image
])

print(response.text)
```

---

## 流式输出

### OpenAI流式输出

```python
def stream_chat(messages, model="gpt-4"):
    """流式对话"""
    client = OpenAI()

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True  # 启用流式输出
    )

    full_response = ""

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            print(content, end="", flush=True)

    print()  # 换行
    return full_response

# 使用示例
messages = [
    {"role": "user", "content": "写一首关于AI的诗"}
]

response = stream_chat(messages)
```

### Claude流式输出

```python
def stream_claude(message):
    """Claude流式输出"""
    client = anthropic.Anthropic()

    with client.messages.stream(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[{"role": "user", "content": message}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    print()

# 使用示例
stream_claude("讲一个关于机器学习的故事")
```

---

## Token计算与成本优化

### Token计数

```python
import tiktoken

def count_tokens(text, model="gpt-4"):
    """计算文本的token数量"""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

# 示例
text = "Hello, how are you doing today?"
token_count = count_tokens(text)
print(f"Token数量: {token_count}")

# 计算对话的总token数
def count_message_tokens(messages, model="gpt-4"):
    """计算消息列表的token数"""
    encoding = tiktoken.encoding_for_model(model)

    total_tokens = 0
    for message in messages:
        # 每条消息有额外的开销
        total_tokens += 4  # 消息格式开销

        for key, value in message.items():
            total_tokens += len(encoding.encode(value))

    total_tokens += 2  # 回复提示

    return total_tokens

messages = [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "Hello!"}
]

print(f"总Token数: {count_message_tokens(messages)}")
```

### 成本计算

```python
class CostCalculator:
    """API成本计算器"""

    # 价格表 (USD per 1M tokens)
    PRICING = {
        "gpt-4": {
            "input": 30.00,
            "output": 60.00
        },
        "gpt-4-turbo": {
            "input": 10.00,
            "output": 30.00
        },
        "gpt-3.5-turbo": {
            "input": 0.50,
            "output": 1.50
        },
        "claude-3-opus": {
            "input": 15.00,
            "output": 75.00
        },
        "claude-3-sonnet": {
            "input": 3.00,
            "output": 15.00
        }
    }

    @staticmethod
    def calculate_cost(input_tokens, output_tokens, model):
        """计算单次请求成本"""
        pricing = CostCalculator.PRICING.get(model)
        if not pricing:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    @staticmethod
    def estimate_monthly_cost(daily_requests, avg_input_tokens,
                             avg_output_tokens, model):
        """估算月度成本"""
        cost_per_request = CostCalculator.calculate_cost(
            avg_input_tokens, avg_output_tokens, model
        )

        daily_cost = cost_per_request * daily_requests
        monthly_cost = daily_cost * 30

        return {
            "per_request": cost_per_request,
            "daily": daily_cost,
            "monthly": monthly_cost
        }

# 使用示例
costs = CostCalculator.estimate_monthly_cost(
    daily_requests=1000,
    avg_input_tokens=500,
    avg_output_tokens=300,
    model="gpt-4"
)

print(f"每次请求成本: ${costs['per_request']:.4f}")
print(f"每日成本: ${costs['daily']:.2f}")
print(f"每月成本: ${costs['monthly']:.2f}")
```

### 成本优化策略

```python
class SmartLLMRouter:
    """智能模型路由器 - 根据任务复杂度选择模型"""

    def __init__(self):
        self.client = OpenAI()

    def classify_complexity(self, prompt):
        """评估任务复杂度"""
        # 简单规则
        keywords_complex = ["分析", "推理", "创作", "复杂"]
        keywords_simple = ["翻译", "总结", "提取", "简单"]

        prompt_lower = prompt.lower()

        if any(kw in prompt_lower for kw in keywords_complex):
            return "complex"
        elif any(kw in prompt_lower for kw in keywords_simple):
            return "simple"
        else:
            # 根据长度判断
            return "complex" if len(prompt) > 500 else "simple"

    def route(self, prompt):
        """路由到合适的模型"""
        complexity = self.classify_complexity(prompt)

        if complexity == "complex":
            model = "gpt-4"
            temperature = 0.7
        else:
            model = "gpt-3.5-turbo"
            temperature = 0.3

        return model, temperature

    def chat(self, prompt):
        """智能对话"""
        model, temperature = self.route(prompt)

        print(f"[使用模型: {model}]")

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )

        return response.choices[0].message.content

# 使用示例
router = SmartLLMRouter()

# 简单任务 -> GPT-3.5
result1 = router.chat("将'Hello'翻译成中文")

# 复杂任务 -> GPT-4
result2 = router.chat("分析量子计算的未来发展趋势")
```

---

## 完整代码示例

### 多模型统一接口

```python
from abc import ABC, abstractmethod
from typing import List, Dict

class BaseLLM(ABC):
    """LLM基类"""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """发送对话"""
        pass

    @abstractmethod
    def stream_chat(self, messages: List[Dict[str, str]], **kwargs):
        """流式对话"""
        pass

class OpenAILLM(BaseLLM):
    """OpenAI实现"""

    def __init__(self, model="gpt-4", api_key=None):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, messages, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content

    def stream_chat(self, messages, **kwargs):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

class ClaudeLLM(BaseLLM):
    """Claude实现"""

    def __init__(self, model="claude-3-opus-20240229", api_key=None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def chat(self, messages, **kwargs):
        response = self.client.messages.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 2000)
        )
        return response.content[0].text

    def stream_chat(self, messages, **kwargs):
        with self.client.messages.stream(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 2000)
        ) as stream:
            for text in stream.text_stream:
                yield text

class UnifiedChatbot:
    """统一聊天机器人"""

    def __init__(self, provider="openai", **kwargs):
        if provider == "openai":
            self.llm = OpenAILLM(**kwargs)
        elif provider == "claude":
            self.llm = ClaudeLLM(**kwargs)
        else:
            raise ValueError(f"不支持的provider: {provider}")

        self.messages = []

    def add_message(self, role, content):
        """添加消息"""
        self.messages.append({"role": role, "content": content})

    def chat(self, user_message, stream=False):
        """发送消息"""
        self.add_message("user", user_message)

        if stream:
            full_response = ""
            for chunk in self.llm.stream_chat(self.messages):
                full_response += chunk
                print(chunk, end="", flush=True)
            print()

            self.add_message("assistant", full_response)
            return full_response
        else:
            response = self.llm.chat(self.messages)
            self.add_message("assistant", response)
            return response

# 使用示例
if __name__ == "__main__":
    # OpenAI
    bot_gpt = UnifiedChatbot(provider="openai", model="gpt-4")
    print("GPT-4:", bot_gpt.chat("你好！"))

    # Claude
    bot_claude = UnifiedChatbot(provider="claude")
    print("Claude:", bot_claude.chat("你好！"))

    # 流式输出
    print("流式输出:")
    bot_gpt.chat("讲一个笑话", stream=True)
```

### 错误处理与重试

```python
import time
from functools import wraps

def retry_with_exponential_backoff(
    max_retries=3,
    initial_delay=1,
    exponential_base=2,
    errors=(Exception,)
):
    """指数退避重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay

            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except errors as e:
                    if i == max_retries - 1:
                        raise

                    print(f"错误: {e}, {delay}秒后重试...")
                    time.sleep(delay)
                    delay *= exponential_base

            return None
        return wrapper
    return decorator

class RobustChatbot:
    """健壮的聊天机器人"""

    def __init__(self):
        self.client = OpenAI()

    @retry_with_exponential_backoff(
        max_retries=3,
        errors=(Exception,)
    )
    def chat(self, message):
        """带重试的对话"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": message}],
                timeout=30  # 30秒超时
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"API调用失败: {e}")
            raise

# 使用示例
bot = RobustChatbot()
try:
    result = bot.chat("Hello!")
    print(result)
except Exception as e:
    print(f"最终失败: {e}")
```

---

## 总结

本教程涵盖了LLM的基础知识和主流API的使用方法:

1. **LLM原理**: 理解Transformer架构和自注意力机制
2. **OpenAI API**: 掌握GPT-4的完整使用方法
3. **Claude API**: 学习Claude的特色功能
4. **Gemini API**: 了解Google的多模态能力
5. **流式输出**: 实现更好的用户体验
6. **成本优化**: 通过智能路由降低费用
7. **工程实践**: 统一接口、错误处理、重试机制

下一步学习建议:
- 深入学习Prompt工程技巧
- 探索Embedding和向量检索
- 构建完整的RAG应用
- 学习Agent开发

## 参考资源

- [OpenAI API文档](https://platform.openai.com/docs)
- [Anthropic Claude文档](https://docs.anthropic.com)
- [Google Gemini文档](https://ai.google.dev)
- [Tiktoken库](https://github.com/openai/tiktoken)
