# 工具调用完整教程

## 目录
1. [Function Calling概述](#function-calling概述)
2. [OpenAI Function Calling](#openai-function-calling)
3. [Claude Tool Use](#claude-tool-use)
4. [自定义工具开发](#自定义工具开发)
5. [工具链组合](#工具链组合)
6. [错误处理与重试](#错误处理与重试)
7. [完整示例：多工具Agent](#完整示例多工具agent)

---

## Function Calling概述

### 什么是Function Calling？

Function Calling（函数调用）是LLM的一项关键能力，允许模型根据用户的自然语言请求，**自动选择并调用预定义的函数/工具**，然后将函数的返回结果整合到回复中。

```
┌──────────────────────────────────────────────────────────────────┐
│                  Function Calling 完整流程                        │
│                                                                  │
│  ┌──────────┐                           ┌──────────┐            │
│  │  用户输入 │   "北京今天天气怎么样？"   │  开发者   │            │
│  └─────┬────┘                           │  定义工具 │            │
│        │                                └─────┬────┘            │
│        ▼                                      │                 │
│  ┌─────────────────────────────────────────────────────┐        │
│  │                    LLM 模型                          │        │
│  │                                                     │        │
│  │  1. 理解用户意图: 查询天气                          │        │
│  │  2. 匹配可用工具: get_weather                       │        │
│  │  3. 生成调用参数: {"city": "北京"}                  │        │
│  │  4. 返回工具调用请求（不是文本！）                   │        │
│  └────────────────────┬────────────────────────────────┘        │
│                       │                                         │
│                       ▼                                         │
│  ┌─────────────────────────────────────────────────────┐        │
│  │              应用层执行函数                           │        │
│  │                                                     │        │
│  │  result = get_weather(city="北京")                   │        │
│  │  → {"temp": 22, "condition": "晴", "humidity": 45}  │        │
│  └────────────────────┬────────────────────────────────┘        │
│                       │                                         │
│                       ▼                                         │
│  ┌─────────────────────────────────────────────────────┐        │
│  │              LLM 整合结果                            │        │
│  │                                                     │        │
│  │  "北京今天天气晴朗，气温22°C，湿度45%，              │        │
│  │   非常适合户外活动。"                                │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
│  关键要点:                                                       │
│  • LLM不直接执行函数，只是决定调用什么函数和参数                  │
│  • 实际执行在应用层完成                                          │
│  • LLM负责将结果转化为自然语言回复                               │
└──────────────────────────────────────────────────────────────────┘
```

### 主流平台对比

| 特性 | OpenAI | Anthropic (Claude) | Google (Gemini) |
|------|--------|-------------------|-----------------|
| **功能名称** | Function Calling / Tool Use | Tool Use | Function Calling |
| **参数定义** | JSON Schema | JSON Schema | JSON Schema |
| **并行调用** | 支持（parallel_tool_calls） | 支持 | 支持 |
| **强制调用** | tool_choice="required" | tool_choice={"type": "tool"} | - |
| **流式支持** | 支持 | 支持 | 支持 |
| **嵌套调用** | 支持 | 支持 | 支持 |

---

## OpenAI Function Calling

### 基础用法

```python
"""
OpenAI Function Calling 完整教程
"""
import json
from openai import OpenAI

client = OpenAI()

# ============================================================
# 1. 定义工具（JSON Schema格式）
# ============================================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如'北京'、'上海'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位",
                        "default": "celsius"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "在产品数据库中搜索产品",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["电子产品", "服装", "食品", "图书"],
                        "description": "产品类别"
                    },
                    "max_price": {
                        "type": "number",
                        "description": "最高价格"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果数量",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "发送电子邮件",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "收件人邮箱地址"
                    },
                    "subject": {
                        "type": "string",
                        "description": "邮件主题"
                    },
                    "body": {
                        "type": "string",
                        "description": "邮件正文内容"
                    }
                },
                "required": ["to", "subject", "body"]
            }
        }
    }
]


# ============================================================
# 2. 实际工具函数实现
# ============================================================
def get_weather(city: str, unit: str = "celsius") -> dict:
    """模拟天气API"""
    weather_db = {
        "北京": {"temp": 22, "condition": "晴", "humidity": 45, "wind": "北风3级"},
        "上海": {"temp": 26, "condition": "多云", "humidity": 65, "wind": "东风2级"},
        "广州": {"temp": 30, "condition": "雷阵雨", "humidity": 80, "wind": "南风4级"},
        "深圳": {"temp": 28, "condition": "阴", "humidity": 70, "wind": "东南风2级"},
    }
    data = weather_db.get(city, {"temp": 20, "condition": "未知", "humidity": 50, "wind": "微风"})
    if unit == "fahrenheit":
        data["temp"] = data["temp"] * 9/5 + 32
    return {**data, "city": city, "unit": unit}

def search_products(query: str, category: str = None, max_price: float = None, limit: int = 5) -> list:
    """模拟产品搜索"""
    products = [
        {"name": "MacBook Pro", "category": "电子产品", "price": 14999, "rating": 4.8},
        {"name": "iPhone 15", "category": "电子产品", "price": 6999, "rating": 4.7},
        {"name": "AirPods Pro", "category": "电子产品", "price": 1899, "rating": 4.6},
        {"name": "Python编程", "category": "图书", "price": 89, "rating": 4.5},
    ]
    results = [p for p in products if query.lower() in p["name"].lower()]
    if category:
        results = [p for p in results if p["category"] == category]
    if max_price:
        results = [p for p in results if p["price"] <= max_price]
    return results[:limit]

def send_email(to: str, subject: str, body: str) -> dict:
    """模拟发送邮件"""
    return {"status": "sent", "to": to, "subject": subject, "message_id": "msg_123456"}

# 工具函数映射
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "search_products": search_products,
    "send_email": send_email,
}


# ============================================================
# 3. 完整调用流程
# ============================================================
def chat_with_tools(user_message: str, tools: list = tools) -> str:
    """带工具调用的完整对话流程"""
    messages = [
        {"role": "system", "content": "你是一个有用的AI助手，可以查询天气、搜索产品和发送邮件。"},
        {"role": "user", "content": user_message}
    ]

    # 第一次调用：LLM决定是否需要工具
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"  # auto/none/required/指定工具
    )

    message = response.choices[0].message

    # 检查是否需要调用工具
    if message.tool_calls:
        # 将assistant的消息加入历史
        messages.append(message)

        # 执行所有工具调用
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"调用工具: {func_name}({func_args})")

            # 执行函数
            if func_name in TOOL_FUNCTIONS:
                result = TOOL_FUNCTIONS[func_name](**func_args)
            else:
                result = {"error": f"未知函数: {func_name}"}

            # 将工具结果加入消息
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })

        # 第二次调用：LLM基于工具结果生成回复
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        return final_response.choices[0].message.content
    else:
        # 不需要工具，直接返回
        return message.content


# ============================================================
# 4. 并行工具调用
# ============================================================
def demo_parallel_calls():
    """演示并行工具调用"""
    result = chat_with_tools("北京和上海的天气分别怎么样？")
    print(f"并行调用结果:\n{result}")


# ============================================================
# 5. 强制工具调用
# ============================================================
def force_tool_call():
    """强制使用指定工具"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "你好"}
        ],
        tools=tools,
        # 强制调用指定工具
        tool_choice={"type": "function", "function": {"name": "get_weather"}}
    )
    print(f"强制调用结果: {response.choices[0].message.tool_calls}")


if __name__ == "__main__":
    # 测试1: 单工具调用
    print("=== 单工具调用 ===")
    result = chat_with_tools("北京今天天气怎么样？")
    print(result)

    # 测试2: 并行调用
    print("\n=== 并行工具调用 ===")
    demo_parallel_calls()

    # 测试3: 多轮对话中的工具调用
    print("\n=== 产品搜索 ===")
    result = chat_with_tools("帮我搜索价格在2000以内的电子产品")
    print(result)
```

### 流式工具调用

```python
def stream_with_tools(user_message: str):
    """流式工具调用"""
    messages = [
        {"role": "system", "content": "你是有用的AI助手。"},
        {"role": "user", "content": user_message}
    ]

    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        stream=True
    )

    tool_calls_data = {}  # 累积工具调用数据
    content_parts = []

    for chunk in stream:
        delta = chunk.choices[0].delta

        # 累积文本内容
        if delta.content:
            print(delta.content, end="", flush=True)
            content_parts.append(delta.content)

        # 累积工具调用
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_data:
                    tool_calls_data[idx] = {"id": "", "name": "", "arguments": ""}
                if tc.id:
                    tool_calls_data[idx]["id"] = tc.id
                if tc.function and tc.function.name:
                    tool_calls_data[idx]["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    tool_calls_data[idx]["arguments"] += tc.function.arguments

    # 如果有工具调用，执行它们
    if tool_calls_data:
        for idx, tc_data in tool_calls_data.items():
            func_name = tc_data["name"]
            func_args = json.loads(tc_data["arguments"])
            print(f"\n[调用工具] {func_name}({func_args})")

            if func_name in TOOL_FUNCTIONS:
                result = TOOL_FUNCTIONS[func_name](**func_args)
                print(f"[工具结果] {json.dumps(result, ensure_ascii=False)}")
```

---

## Claude Tool Use

### Anthropic API工具调用

```python
"""
Claude Tool Use 完整教程
"""
import json
import anthropic

client = anthropic.Anthropic()

# ============================================================
# 1. 定义工具（Anthropic格式）
# ============================================================
tools = [
    {
        "name": "get_weather",
        "description": "获取指定城市的当前天气信息。返回温度、天气状况和湿度。",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如'北京'、'上海'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度单位，默认celsius"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "calculator",
        "description": "执行数学计算。支持基本运算和常见数学函数。",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如'2 + 3 * 4'或'sqrt(144)'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "search_knowledge",
        "description": "搜索知识库获取信息。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询"
                },
                "category": {
                    "type": "string",
                    "enum": ["science", "history", "technology", "general"],
                    "description": "搜索类别"
                }
            },
            "required": ["query"]
        }
    }
]


# ============================================================
# 2. 工具实现
# ============================================================
def get_weather(city: str, unit: str = "celsius") -> dict:
    weather_db = {
        "北京": {"temp": 22, "condition": "晴", "humidity": 45},
        "上海": {"temp": 26, "condition": "多云", "humidity": 65},
    }
    data = weather_db.get(city, {"temp": 20, "condition": "未知", "humidity": 50})
    if unit == "fahrenheit":
        data["temp"] = data["temp"] * 9/5 + 32
    return {**data, "city": city}

def calculator(expression: str) -> str:
    import math
    safe = {"sqrt": math.sqrt, "pi": math.pi, "e": math.e, "log": math.log,
            "sin": math.sin, "cos": math.cos, "abs": abs, "pow": pow}
    try:
        return str(eval(expression, {"__builtins__": {}}, safe))
    except Exception as e:
        return f"Error: {e}"

def search_knowledge(query: str, category: str = "general") -> str:
    return f"关于'{query}'的搜索结果：这是一个模拟的知识库搜索结果。"

TOOL_MAP = {
    "get_weather": get_weather,
    "calculator": calculator,
    "search_knowledge": search_knowledge,
}


# ============================================================
# 3. 完整调用流程
# ============================================================
def chat_with_claude_tools(user_message: str) -> str:
    """Claude工具调用完整流程"""
    messages = [{"role": "user", "content": user_message}]

    # 第一次调用
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        tools=tools,
        messages=messages,
        system="你是一个有用的AI助手，可以使用工具来帮助用户。用中文回答。"
    )

    # 处理响应
    while response.stop_reason == "tool_use":
        # 收集assistant的响应内容
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        # 执行所有工具调用
        tool_results = []
        for block in assistant_content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                tool_id = block.id

                print(f"  调用工具: {tool_name}({tool_input})")

                # 执行工具
                if tool_name in TOOL_MAP:
                    result = TOOL_MAP[tool_name](**tool_input)
                else:
                    result = f"未知工具: {tool_name}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
                })

        # 将工具结果传回
        messages.append({"role": "user", "content": tool_results})

        # 继续对话
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=tools,
            messages=messages,
            system="你是一个有用的AI助手。用中文回答。"
        )

    # 提取最终文本
    final_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            final_text += block.text

    return final_text


# ============================================================
# 4. 流式工具调用
# ============================================================
def stream_claude_tools(user_message: str):
    """Claude流式工具调用"""
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        tools=tools,
        messages=[{"role": "user", "content": user_message}]
    ) as stream:
        for event in stream:
            if hasattr(event, 'type'):
                if event.type == 'content_block_start':
                    if hasattr(event.content_block, 'text'):
                        print(event.content_block.text, end="")
                elif event.type == 'content_block_delta':
                    if hasattr(event.delta, 'text'):
                        print(event.delta.text, end="", flush=True)


if __name__ == "__main__":
    # 测试
    print("=== Claude Tool Use ===")
    result = chat_with_claude_tools("北京今天天气怎么样？另外帮我算一下 25 * 4 + 30")
    print(f"\n最终回答: {result}")
```

---

## 自定义工具开发

### 天气查询工具

```python
"""
实用自定义工具集合
"""
import os
import json
import requests
from typing import Optional
from datetime import datetime


# ============================================================
# 工具1: 真实天气查询（OpenWeatherMap API）
# ============================================================
class WeatherTool:
    """真实天气查询工具"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

    def get_weather(self, city: str, unit: str = "celsius") -> dict:
        """获取真实天气数据"""
        units = "metric" if unit == "celsius" else "imperial"
        try:
            response = requests.get(
                self.base_url,
                params={
                    "q": city,
                    "appid": self.api_key,
                    "units": units,
                    "lang": "zh_cn"
                },
                timeout=10
            )
            data = response.json()

            if response.status_code == 200:
                return {
                    "city": city,
                    "temperature": data["main"]["temp"],
                    "feels_like": data["main"]["feels_like"],
                    "humidity": data["main"]["humidity"],
                    "description": data["weather"][0]["description"],
                    "wind_speed": data["wind"]["speed"],
                    "unit": "°C" if unit == "celsius" else "°F"
                }
            else:
                return {"error": f"API错误: {data.get('message', '未知错误')}"}
        except Exception as e:
            return {"error": f"请求失败: {str(e)}"}

    def to_openai_tool(self) -> dict:
        """转换为OpenAI工具格式"""
        return {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的实时天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["city"]
                }
            }
        }


# ============================================================
# 工具2: 数据库查询工具
# ============================================================
import sqlite3

class DatabaseTool:
    """SQL数据库查询工具"""

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def query(self, sql: str) -> dict:
        """执行SELECT查询"""
        sql_clean = sql.strip().upper()
        if not sql_clean.startswith("SELECT"):
            return {"error": "仅支持SELECT查询"}

        dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "--", ";"]
        for keyword in dangerous:
            if keyword in sql_clean and keyword != "SELECT":
                return {"error": f"不允许的SQL操作: {keyword}"}

        try:
            cursor = self.conn.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows = [dict(row) for row in cursor.fetchall()]
            return {
                "columns": columns,
                "rows": rows[:50],
                "total_count": len(rows),
                "truncated": len(rows) > 50
            }
        except Exception as e:
            return {"error": f"查询错误: {str(e)}"}

    def get_schema(self) -> str:
        """获取数据库表结构"""
        cursor = self.conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table'"
        )
        tables = cursor.fetchall()
        return "\n".join([f"表: {t[0]}\n{t[1]}" for t in tables])

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "query_database",
                "description": f"查询SQL数据库。可用表结构:\n{self.get_schema()}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "SELECT SQL查询语句"}
                    },
                    "required": ["sql"]
                }
            }
        }


# ============================================================
# 工具3: 文件操作工具
# ============================================================
class FileOperationTool:
    """文件操作工具（沙箱化）"""

    def __init__(self, allowed_dir: str = "./workspace"):
        self.allowed_dir = os.path.abspath(allowed_dir)
        os.makedirs(self.allowed_dir, exist_ok=True)

    def _safe_path(self, path: str) -> str:
        """确保路径在允许目录内"""
        full_path = os.path.abspath(os.path.join(self.allowed_dir, path))
        if not full_path.startswith(self.allowed_dir):
            raise ValueError("路径越界")
        return full_path

    def read_file(self, path: str) -> dict:
        try:
            safe_path = self._safe_path(path)
            with open(safe_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {"content": content, "size": len(content), "path": path}
        except Exception as e:
            return {"error": str(e)}

    def write_file(self, path: str, content: str) -> dict:
        try:
            safe_path = self._safe_path(path)
            os.makedirs(os.path.dirname(safe_path), exist_ok=True)
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {"status": "success", "path": path, "size": len(content)}
        except Exception as e:
            return {"error": str(e)}

    def list_dir(self, path: str = ".") -> dict:
        try:
            safe_path = self._safe_path(path)
            items = []
            for item in os.listdir(safe_path):
                full = os.path.join(safe_path, item)
                items.append({
                    "name": item,
                    "type": "dir" if os.path.isdir(full) else "file",
                    "size": os.path.getsize(full) if os.path.isfile(full) else 0
                })
            return {"items": items, "count": len(items)}
        except Exception as e:
            return {"error": str(e)}


# ============================================================
# 工具4: API调用工具
# ============================================================
class APICallTool:
    """通用HTTP API调用工具"""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()

    def call(
        self,
        url: str,
        method: str = "GET",
        headers: dict = None,
        params: dict = None,
        body: dict = None
    ) -> dict:
        """发送HTTP请求"""
        try:
            response = self.session.request(
                method=method.upper(),
                url=url,
                headers=headers or {},
                params=params,
                json=body,
                timeout=self.timeout
            )
            # 尝试解析JSON
            try:
                data = response.json()
            except:
                data = response.text[:2000]

            return {
                "status_code": response.status_code,
                "data": data,
                "headers": dict(response.headers)
            }
        except Exception as e:
            return {"error": str(e)}

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "api_call",
                "description": "发送HTTP API请求",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "API URL"},
                        "method": {"type": "string", "enum": ["GET", "POST"], "default": "GET"},
                        "params": {"type": "object", "description": "URL查询参数"},
                        "body": {"type": "object", "description": "请求体（POST）"}
                    },
                    "required": ["url"]
                }
            }
        }
```

---

## 工具链组合

### 顺序工具链（Sequential）

```
┌──────────────────────────────────────────────────────────┐
│                工具链组合模式                              │
│                                                          │
│  顺序执行 (Sequential):                                  │
│  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐          │
│  │工具A  │───►│工具B  │───►│工具C  │───►│ 结果  │          │
│  └──────┘    └──────┘    └──────┘    └──────┘          │
│  搜索信息  → 提取数据  → 分析计算  → 生成报告           │
│                                                          │
│  并行执行 (Parallel):                                    │
│  ┌──────┐──►┌──────┐                                    │
│  │工具A  │   │ 结果A │──┐                                │
│  └──────┘   └──────┘  │  ┌──────┐    ┌──────┐          │
│  ┌──────┐──►┌──────┐  ├─►│ 合并  │───►│ 结果  │          │
│  │工具B  │   │ 结果B │──┘  └──────┘    └──────┘          │
│  └──────┘   └──────┘                                    │
│                                                          │
│  条件执行 (Conditional):                                 │
│  ┌──────┐   条件判断   ┌──────┐                          │
│  │工具A  │──► 结果 ───►│工具B  │  (满足条件)              │
│  └──────┘      │       └──────┘                          │
│                └──────►┌──────┐  (不满足条件)            │
│                        │工具C  │                          │
│                        └──────┘                          │
└──────────────────────────────────────────────────────────┘
```

```python
"""
工具链组合实现
"""
import asyncio
import json
from typing import Any, Callable
from concurrent.futures import ThreadPoolExecutor


# ============================================================
# 1. 顺序工具链
# ============================================================
class SequentialToolChain:
    """顺序执行多个工具"""

    def __init__(self):
        self.steps: list[dict] = []

    def add_step(self, name: str, func: Callable, input_mapper: Callable = None):
        """添加执行步骤"""
        self.steps.append({
            "name": name,
            "func": func,
            "input_mapper": input_mapper  # 将上一步输出映射为当前步输入
        })
        return self  # 链式调用

    def execute(self, initial_input: Any) -> dict:
        """执行整个工具链"""
        results = []
        current_input = initial_input

        for step in self.steps:
            # 映射输入
            if step["input_mapper"] and results:
                current_input = step["input_mapper"](results[-1]["output"])

            print(f"执行步骤: {step['name']}")
            try:
                output = step["func"](current_input)
                results.append({
                    "step": step["name"],
                    "input": current_input,
                    "output": output,
                    "status": "success"
                })
                current_input = output
            except Exception as e:
                results.append({
                    "step": step["name"],
                    "error": str(e),
                    "status": "error"
                })
                break

        return {
            "final_result": results[-1]["output"] if results and results[-1]["status"] == "success" else None,
            "steps": results
        }


# 使用示例
def search_info(query):
    return f"关于{query}的搜索结果：Python是一门编程语言..."

def extract_keywords(text):
    words = text.split("：")
    return words[-1] if len(words) > 1 else text

def translate_text(text):
    return f"[翻译] {text}"

chain = SequentialToolChain()
chain.add_step("搜索", search_info)
chain.add_step("提取关键词", extract_keywords)
chain.add_step("翻译", translate_text)

result = chain.execute("Python编程语言")
print(json.dumps(result, ensure_ascii=False, indent=2))


# ============================================================
# 2. 并行工具链
# ============================================================
class ParallelToolChain:
    """并行执行多个工具"""

    def __init__(self, max_workers: int = 5):
        self.tools: list[dict] = []
        self.max_workers = max_workers

    def add_tool(self, name: str, func: Callable, input_key: str = None):
        """添加并行工具"""
        self.tools.append({"name": name, "func": func, "input_key": input_key})
        return self

    def execute(self, inputs: dict) -> dict:
        """并行执行所有工具"""
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for tool in self.tools:
                tool_input = inputs.get(tool["input_key"], inputs) if tool["input_key"] else inputs
                future = executor.submit(tool["func"], tool_input)
                futures[tool["name"]] = future

            for name, future in futures.items():
                try:
                    results[name] = {"output": future.result(timeout=30), "status": "success"}
                except Exception as e:
                    results[name] = {"error": str(e), "status": "error"}

        return results


# 使用示例：同时查询多个城市天气
parallel = ParallelToolChain()
parallel.add_tool("北京天气", lambda _: get_weather("北京"), None)
parallel.add_tool("上海天气", lambda _: get_weather("上海"), None)

results = parallel.execute({})
for name, result in results.items():
    print(f"{name}: {result}")


# ============================================================
# 3. 条件工具链
# ============================================================
class ConditionalToolChain:
    """根据条件选择不同的工具执行"""

    def __init__(self):
        self.routes: list[dict] = []
        self.default_func: Callable = None

    def add_route(self, condition: Callable, func: Callable, name: str):
        """添加条件路由"""
        self.routes.append({"condition": condition, "func": func, "name": name})
        return self

    def set_default(self, func: Callable):
        """设置默认处理"""
        self.default_func = func
        return self

    def execute(self, input_data: Any) -> dict:
        """根据条件执行对应工具"""
        for route in self.routes:
            if route["condition"](input_data):
                print(f"匹配路由: {route['name']}")
                result = route["func"](input_data)
                return {"route": route["name"], "output": result}

        if self.default_func:
            result = self.default_func(input_data)
            return {"route": "default", "output": result}

        return {"error": "未匹配任何路由"}


# 使用示例
conditional = ConditionalToolChain()
conditional.add_route(
    condition=lambda x: "天气" in x,
    func=lambda x: get_weather("北京"),
    name="天气查询"
)
conditional.add_route(
    condition=lambda x: "计算" in x,
    func=lambda x: calculator("1+1"),
    name="数学计算"
)
conditional.set_default(lambda x: f"默认处理: {x}")

result = conditional.execute("今天天气怎么样")
print(result)
```

---

## 错误处理与重试

### 完善的错误处理框架

```python
"""
工具调用错误处理与重试机制
"""
import time
import json
import logging
from typing import Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """错误类型分类"""
    NETWORK = "network"        # 网络错误
    TIMEOUT = "timeout"        # 超时
    RATE_LIMIT = "rate_limit"  # 速率限制
    AUTH = "auth"              # 认证错误
    PARSE = "parse"            # 解析错误
    TOOL = "tool"              # 工具执行错误
    UNKNOWN = "unknown"        # 未知错误


@dataclass
class ToolCallResult:
    """工具调用结果"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    error_type: Optional[ErrorType] = None
    retries: int = 0
    duration_ms: float = 0


class RetryConfig:
    """重试配置"""
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_backoff: bool = True,
        retryable_errors: list[ErrorType] = None
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.retryable_errors = retryable_errors or [
            ErrorType.NETWORK, ErrorType.TIMEOUT, ErrorType.RATE_LIMIT
        ]

    def get_delay(self, attempt: int) -> float:
        """计算重试延迟（指数退避）"""
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** attempt)
        else:
            delay = self.base_delay
        return min(delay, self.max_delay)


class RobustToolExecutor:
    """健壮的工具执行器"""

    def __init__(self, retry_config: RetryConfig = None):
        self.retry_config = retry_config or RetryConfig()
        self.tools: dict[str, Callable] = {}
        self.fallbacks: dict[str, Callable] = {}

    def register(self, name: str, func: Callable, fallback: Callable = None):
        """注册工具及其降级方案"""
        self.tools[name] = func
        if fallback:
            self.fallbacks[name] = fallback

    def _classify_error(self, error: Exception) -> ErrorType:
        """分类错误类型"""
        error_str = str(error).lower()
        if "timeout" in error_str:
            return ErrorType.TIMEOUT
        elif "rate" in error_str or "429" in error_str:
            return ErrorType.RATE_LIMIT
        elif "auth" in error_str or "401" in error_str or "403" in error_str:
            return ErrorType.AUTH
        elif "connection" in error_str or "network" in error_str:
            return ErrorType.NETWORK
        elif "json" in error_str or "parse" in error_str:
            return ErrorType.PARSE
        else:
            return ErrorType.UNKNOWN

    def execute(self, tool_name: str, **kwargs) -> ToolCallResult:
        """执行工具（带重试和降级）"""
        if tool_name not in self.tools:
            return ToolCallResult(
                success=False,
                error=f"工具未找到: {tool_name}",
                error_type=ErrorType.TOOL
            )

        start_time = time.time()
        last_error = None

        # 重试循环
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                result = self.tools[tool_name](**kwargs)
                duration = (time.time() - start_time) * 1000
                return ToolCallResult(
                    success=True,
                    data=result,
                    retries=attempt,
                    duration_ms=duration
                )
            except Exception as e:
                error_type = self._classify_error(e)
                last_error = str(e)
                logger.warning(f"工具 {tool_name} 第{attempt+1}次执行失败: {last_error} (类型: {error_type.value})")

                # 判断是否可重试
                if error_type not in self.retry_config.retryable_errors:
                    break

                # 最后一次尝试不需要等待
                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.get_delay(attempt)
                    logger.info(f"等待 {delay:.1f}s 后重试...")
                    time.sleep(delay)

        # 所有重试失败，尝试降级方案
        if tool_name in self.fallbacks:
            logger.info(f"使用降级方案执行 {tool_name}")
            try:
                result = self.fallbacks[tool_name](**kwargs)
                duration = (time.time() - start_time) * 1000
                return ToolCallResult(
                    success=True,
                    data=result,
                    retries=self.retry_config.max_retries,
                    duration_ms=duration
                )
            except Exception as e:
                last_error = f"降级方案也失败: {str(e)}"

        duration = (time.time() - start_time) * 1000
        return ToolCallResult(
            success=False,
            error=last_error,
            error_type=self._classify_error(Exception(last_error)),
            retries=self.retry_config.max_retries,
            duration_ms=duration
        )


# ============================================================
# 使用示例
# ============================================================
def real_weather_api(city: str) -> dict:
    """模拟可能失败的真实API"""
    import random
    if random.random() < 0.3:
        raise ConnectionError("网络连接超时")
    return {"city": city, "temp": 22, "condition": "晴"}

def cached_weather(city: str) -> dict:
    """缓存降级方案"""
    return {"city": city, "temp": "N/A", "condition": "数据来自缓存", "cached": True}

# 创建执行器
executor = RobustToolExecutor(RetryConfig(max_retries=3, base_delay=0.5))
executor.register("weather", real_weather_api, fallback=cached_weather)

# 执行
result = executor.execute("weather", city="北京")
print(f"成功: {result.success}")
print(f"数据: {result.data}")
print(f"重试次数: {result.retries}")
print(f"耗时: {result.duration_ms:.0f}ms")
```

---

## 完整示例：多工具Agent

### 天气+新闻+计算+翻译 Agent

```python
"""
完整多工具Agent实现
集成天气查询、新闻搜索、数学计算和翻译功能
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


# ============================================================
# 1. 工具定义
# ============================================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取城市的实时天气信息，包括温度、湿度、风力等",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "days": {"type": "integer", "description": "预报天数(1-7)", "default": 1}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "搜索最新新闻和资讯",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "category": {
                        "type": "string",
                        "enum": ["科技", "财经", "体育", "娱乐", "国际"],
                        "description": "新闻类别"
                    },
                    "limit": {"type": "integer", "description": "结果数量", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算，支持基本运算、三角函数、对数等",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"},
                    "precision": {"type": "integer", "description": "小数精度", "default": 4}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "翻译文本，支持中英日韩法德等多种语言",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要翻译的文本"},
                    "source_lang": {"type": "string", "description": "源语言", "default": "auto"},
                    "target_lang": {"type": "string", "description": "目标语言", "default": "中文"}
                },
                "required": ["text", "target_lang"]
            }
        }
    }
]


# ============================================================
# 2. 工具实现
# ============================================================
def get_weather(city: str, days: int = 1) -> dict:
    """模拟天气API"""
    import random
    conditions = ["晴", "多云", "阴", "小雨", "大风"]
    forecast = []
    for i in range(days):
        forecast.append({
            "date": f"第{i+1}天",
            "temp_high": random.randint(15, 35),
            "temp_low": random.randint(5, 20),
            "condition": random.choice(conditions),
            "humidity": random.randint(30, 90),
            "wind": f"{random.choice(['北','南','东','西'])}风{random.randint(1,5)}级"
        })
    return {"city": city, "forecast": forecast}

def search_news(query: str, category: str = None, limit: int = 5) -> list:
    """模拟新闻搜索"""
    news = [
        {"title": f"[{category or '综合'}] {query}相关新闻{i+1}",
         "summary": f"这是关于{query}的最新报道...",
         "source": ["新华社", "人民日报", "央视新闻", "科技日报"][i % 4],
         "time": f"{i+1}小时前"}
        for i in range(limit)
    ]
    return news

def calculate(expression: str, precision: int = 4) -> dict:
    """数学计算"""
    import math
    safe = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
        "log": math.log, "log2": math.log2, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pow": pow, "ceil": math.ceil, "floor": math.floor,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe)
        return {"expression": expression, "result": round(result, precision)}
    except Exception as e:
        return {"expression": expression, "error": str(e)}

def translate(text: str, source_lang: str = "auto", target_lang: str = "中文") -> dict:
    """模拟翻译（实际可调用Google/DeepL API）"""
    # 简单模拟
    translations = {
        "Hello, World!": "你好，世界！",
        "Machine Learning": "机器学习",
        "Artificial Intelligence": "人工智能",
    }
    translated = translations.get(text, f"[{target_lang}翻译] {text}")
    return {"original": text, "translated": translated, "source": source_lang, "target": target_lang}


TOOL_MAP = {
    "get_weather": get_weather,
    "search_news": search_news,
    "calculate": calculate,
    "translate": translate,
}


# ============================================================
# 3. 多工具Agent
# ============================================================
class MultiToolAgent:
    """多工具Agent"""

    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model
        self.conversation_history = []
        self.system_prompt = """你是一个功能强大的AI助手，拥有以下能力：

1. 天气查询 - 查询任何城市的实时天气和预报
2. 新闻搜索 - 搜索最新新闻资讯
3. 数学计算 - 执行各种数学运算
4. 文本翻译 - 翻译多种语言

使用规则:
- 根据用户需求自动选择合适的工具
- 复杂问题可以组合使用多个工具
- 用中文回答所有问题
- 给出清晰、有条理的回答"""

    def chat(self, user_message: str) -> str:
        """与Agent对话"""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.conversation_history

        # 调用LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0
        )

        message = response.choices[0].message

        # 迭代处理工具调用
        max_iterations = 5
        iteration = 0

        while message.tool_calls and iteration < max_iterations:
            iteration += 1
            print(f"\n  [迭代 {iteration}] 需要调用 {len(message.tool_calls)} 个工具")

            # 添加assistant消息
            messages.append(message)

            # 执行所有工具
            for tc in message.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)
                print(f"  调用: {func_name}({json.dumps(func_args, ensure_ascii=False)})")

                if func_name in TOOL_MAP:
                    result = TOOL_MAP[func_name](**func_args)
                else:
                    result = {"error": f"未知工具: {func_name}"}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False)
                })

            # 再次调用LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            message = response.choices[0].message

        # 获取最终回复
        reply = message.content or "抱歉，处理过程中出现问题。"

        self.conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        return reply

    def reset(self):
        """重置对话历史"""
        self.conversation_history = []


# ============================================================
# 4. 交互运行
# ============================================================
def main():
    agent = MultiToolAgent()

    print("=" * 60)
    print("   多工具AI助手 (天气/新闻/计算/翻译)")
    print("   输入 'quit' 退出, 'reset' 重置对话")
    print("=" * 60)

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break
        if user_input.lower() == 'reset':
            agent.reset()
            print("对话已重置。")
            continue
        if not user_input:
            continue

        response = agent.chat(user_input)
        print(f"\n助手: {response}")


def run_demo():
    """演示多工具协作"""
    agent = MultiToolAgent()

    demos = [
        "北京今天天气怎么样？适合外出吗？",
        "帮我搜索最新的AI科技新闻",
        "计算圆的面积，半径为5.5厘米",
        "把'Artificial Intelligence is transforming the world'翻译成中文",
        "北京和上海哪个城市今天更热？温差是多少度？",
    ]

    for demo in demos:
        print(f"\n{'='*60}")
        print(f"用户: {demo}")
        print('-'*60)
        response = agent.chat(demo)
        print(f"\n助手: {response}")
        agent.reset()


if __name__ == "__main__":
    run_demo()
```

---

## 高级工具模式

### 类型安全的工具装饰器框架

```python
"""
类型安全的工具框架：自动从Python类型注解生成JSON Schema
"""
import inspect
import json
from typing import get_type_hints, Optional, Union, Literal
from functools import wraps


def auto_tool(func):
    """
    自动工具装饰器：从函数签名生成工具定义
    支持类型注解自动转JSON Schema
    """
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    doc = func.__doc__ or func.__name__

    # Python类型到JSON Schema类型映射
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        param_type = hints.get(param_name, str)
        param_desc = ""

        # 处理Optional类型
        is_optional = False
        if hasattr(param_type, "__origin__"):
            if param_type.__origin__ is Union:
                args = param_type.__args__
                if type(None) in args:
                    is_optional = True
                    param_type = args[0]

            # 处理Literal类型
            if param_type.__origin__ is Literal:
                properties[param_name] = {
                    "type": "string",
                    "enum": list(param_type.__args__),
                    "description": param_desc
                }
                if param.default is inspect.Parameter.empty and not is_optional:
                    required.append(param_name)
                continue

        json_type = type_map.get(param_type, "string")
        properties[param_name] = {
            "type": json_type,
            "description": param_desc
        }

        if param.default is not inspect.Parameter.empty:
            properties[param_name]["default"] = param.default
        elif not is_optional:
            required.append(param_name)

    # 构建工具定义
    tool_def = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc.strip().split("\n")[0],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.tool_definition = tool_def
    wrapper.tool_name = func.__name__
    return wrapper


# 使用示例
@auto_tool
def get_stock_price(symbol: str, currency: Literal["USD", "CNY"] = "USD") -> dict:
    """获取股票实时价格"""
    prices = {"AAPL": 185.5, "GOOGL": 140.2, "MSFT": 380.1}
    price = prices.get(symbol.upper(), 0)
    if currency == "CNY":
        price *= 7.2
    return {"symbol": symbol, "price": price, "currency": currency}


@auto_tool
def send_notification(
    recipient: str,
    message: str,
    priority: Literal["low", "medium", "high"] = "medium",
    channel: Optional[str] = None
) -> dict:
    """发送通知消息给指定用户"""
    return {
        "status": "sent",
        "recipient": recipient,
        "priority": priority,
        "channel": channel or "default"
    }


# 查看自动生成的工具定义
print(json.dumps(get_stock_price.tool_definition, indent=2, ensure_ascii=False))
print(json.dumps(send_notification.tool_definition, indent=2, ensure_ascii=False))
```

### 工具注册中心

```python
"""
工具注册中心：统一管理所有工具的注册、发现和执行
"""
import json
import time
import logging
from typing import Callable, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("ToolRegistry")


@dataclass
class ToolMetadata:
    """工具元数据"""
    name: str
    description: str
    category: str
    version: str = "1.0.0"
    author: str = ""
    call_count: int = 0
    avg_duration_ms: float = 0
    error_count: int = 0
    last_called: float = 0


class ToolRegistry:
    """
    工具注册中心
    功能:
    - 工具注册与发现
    - 调用统计
    - 版本管理
    - 健康检查
    """

    def __init__(self):
        self.tools: dict[str, dict] = {}
        self.metadata: dict[str, ToolMetadata] = {}
        self.categories: dict[str, list[str]] = {}

    def register(
        self,
        func: Callable,
        name: str = None,
        description: str = None,
        category: str = "general",
        tool_def: dict = None,
        version: str = "1.0.0"
    ):
        """注册工具"""
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip().split("\n")[0]

        self.tools[tool_name] = {
            "func": func,
            "definition": tool_def or {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_desc,
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        }

        self.metadata[tool_name] = ToolMetadata(
            name=tool_name,
            description=tool_desc,
            category=category,
            version=version
        )

        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(tool_name)

        logger.info(f"注册工具: {tool_name} (v{version}) [{category}]")

    def register_auto_tool(self, func, category: str = "general"):
        """注册使用@auto_tool装饰的函数"""
        if hasattr(func, "tool_definition"):
            self.register(
                func=func,
                name=func.tool_name,
                tool_def=func.tool_definition,
                category=category
            )
        else:
            raise ValueError(f"{func.__name__} 未使用 @auto_tool 装饰器")

    def execute(self, tool_name: str, **kwargs) -> dict:
        """执行工具（带统计）"""
        if tool_name not in self.tools:
            return {"error": f"工具未找到: {tool_name}"}

        start = time.time()
        meta = self.metadata[tool_name]

        try:
            result = self.tools[tool_name]["func"](**kwargs)
            duration = (time.time() - start) * 1000

            # 更新统计
            meta.call_count += 1
            meta.avg_duration_ms = (
                (meta.avg_duration_ms * (meta.call_count - 1) + duration)
                / meta.call_count
            )
            meta.last_called = time.time()

            return {
                "status": "success",
                "result": result,
                "duration_ms": round(duration, 2)
            }
        except Exception as e:
            meta.error_count += 1
            meta.call_count += 1
            logger.error(f"工具执行错误 {tool_name}: {e}")
            return {"status": "error", "error": str(e)}

    def get_tools_for_llm(
        self, category: str = None
    ) -> list[dict]:
        """获取LLM格式的工具列表"""
        tools = []
        for name, tool in self.tools.items():
            if category and self.metadata[name].category != category:
                continue
            tools.append(tool["definition"])
        return tools

    def get_tool_map(self) -> dict[str, Callable]:
        """获取工具名到函数的映射"""
        return {name: tool["func"] for name, tool in self.tools.items()}

    def list_tools(self, category: str = None) -> list[dict]:
        """列出所有工具信息"""
        result = []
        for name, meta in self.metadata.items():
            if category and meta.category != category:
                continue
            result.append({
                "name": meta.name,
                "description": meta.description,
                "category": meta.category,
                "version": meta.version,
                "calls": meta.call_count,
                "errors": meta.error_count,
                "avg_ms": f"{meta.avg_duration_ms:.1f}",
            })
        return result

    def health_check(self) -> dict:
        """工具健康检查"""
        total = len(self.tools)
        healthy = sum(
            1 for m in self.metadata.values()
            if m.error_count < m.call_count * 0.1 or m.call_count == 0
        )
        return {
            "total_tools": total,
            "healthy": healthy,
            "unhealthy": total - healthy,
            "categories": list(self.categories.keys()),
            "total_calls": sum(m.call_count for m in self.metadata.values()),
            "total_errors": sum(m.error_count for m in self.metadata.values()),
        }

    def print_dashboard(self):
        """打印工具仪表盘"""
        health = self.health_check()
        print(f"\n{'='*60}")
        print(f"  工具注册中心仪表盘")
        print(f"{'='*60}")
        print(f"  总工具数: {health['total_tools']}")
        print(f"  健康/异常: {health['healthy']}/{health['unhealthy']}")
        print(f"  总调用次数: {health['total_calls']}")
        print(f"  总错误次数: {health['total_errors']}")
        print(f"\n  {'─'*56}")

        for tool in self.list_tools():
            status = "OK" if tool["errors"] == 0 else "!!"
            print(
                f"  [{status}] {tool['name']:<20} "
                f"v{tool['version']:<8} "
                f"calls:{tool['calls']:<6} "
                f"err:{tool['errors']:<4} "
                f"avg:{tool['avg_ms']}ms"
            )
        print(f"{'='*60}")


# 使用示例
registry = ToolRegistry()

# 注册工具
registry.register_auto_tool(get_stock_price, category="finance")
registry.register_auto_tool(send_notification, category="communication")

# 执行工具
result = registry.execute("get_stock_price", symbol="AAPL", currency="CNY")
print(f"执行结果: {result}")

# 获取LLM格式工具列表
llm_tools = registry.get_tools_for_llm()
print(f"LLM工具数: {len(llm_tools)}")

# 仪表盘
registry.print_dashboard()
```

### 工具结果缓存

```python
"""
工具结果缓存：减少重复调用，降低成本和延迟
"""
import hashlib
import json
import time
from typing import Callable, Any, Optional


class ToolCache:
    """
    工具调用结果缓存
    支持:
    - TTL过期
    - LRU淘汰
    - 按工具分别配置
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: dict[str, dict] = {}
        self.access_order: list[str] = []
        self.tool_ttl: dict[str, int] = {}
        self.stats = {"hits": 0, "misses": 0}

    def set_tool_ttl(self, tool_name: str, ttl: int):
        """为特定工具设置TTL（秒）"""
        self.tool_ttl[tool_name] = ttl

    def _make_key(self, tool_name: str, kwargs: dict) -> str:
        """生成缓存键"""
        args_str = json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
        content = f"{tool_name}:{args_str}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, tool_name: str, kwargs: dict) -> Optional[Any]:
        """查询缓存"""
        key = self._make_key(tool_name, kwargs)
        if key in self.cache:
            entry = self.cache[key]
            ttl = self.tool_ttl.get(tool_name, self.default_ttl)
            if time.time() - entry["timestamp"] < ttl:
                self.stats["hits"] += 1
                # 更新LRU顺序
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                return entry["result"]
            else:
                # 已过期
                del self.cache[key]
        self.stats["misses"] += 1
        return None

    def set(self, tool_name: str, kwargs: dict, result: Any):
        """写入缓存"""
        key = self._make_key(tool_name, kwargs)
        # LRU淘汰
        if len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            self.cache.pop(oldest, None)

        self.cache[key] = {
            "result": result,
            "timestamp": time.time(),
            "tool": tool_name
        }
        self.access_order.append(key)

    def cached_execute(
        self,
        tool_name: str,
        func: Callable,
        **kwargs
    ) -> dict:
        """带缓存的工具执行"""
        # 查缓存
        cached = self.get(tool_name, kwargs)
        if cached is not None:
            return {"result": cached, "cached": True}

        # 执行工具
        result = func(**kwargs)

        # 写缓存
        self.set(tool_name, kwargs, result)
        return {"result": result, "cached": False}

    def get_stats(self) -> dict:
        """获取缓存统计"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / max(total, 1) * 100
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": f"{hit_rate:.1f}%"
        }


# 使用示例
cache = ToolCache(max_size=100, default_ttl=300)

# 天气数据缓存60秒（变化快）
cache.set_tool_ttl("get_weather", 60)
# 数据库查询缓存300秒
cache.set_tool_ttl("query_database", 300)

# 第一次调用（缓存未命中）
result1 = cache.cached_execute(
    "get_weather", get_weather, city="北京"
)
print(f"第1次: cached={result1['cached']}")  # False

# 第二次调用（缓存命中）
result2 = cache.cached_execute(
    "get_weather", get_weather, city="北京"
)
print(f"第2次: cached={result2['cached']}")  # True

print(f"缓存统计: {cache.get_stats()}")
```

### 工具调用完整流程图

```
┌──────────────────────────────────────────────────────────────────┐
│              工具调用生产级完整流程                                  │
│                                                                  │
│  用户请求                                                        │
│     │                                                            │
│     ▼                                                            │
│  ┌──────────┐                                                    │
│  │ 输入验证  │ → 检查长度/格式/安全性                              │
│  └────┬─────┘                                                    │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────┐    ┌──────────┐                                    │
│  │ LLM推理  │───►│ 缓存检查  │                                    │
│  │ 选择工具  │    │ 命中？   │                                    │
│  └────┬─────┘    └──┬───┬───┘                                    │
│       │          命中│   │未命中                                  │
│       │             │   │                                        │
│       │             ▼   ▼                                        │
│       │     ┌────────┐ ┌──────────┐                              │
│       │     │返回缓存│ │ 安全检查  │ → 参数注入检测                │
│       │     │ 结果   │ └────┬─────┘                              │
│       │     └────────┘      │                                    │
│       │                     ▼                                    │
│       │              ┌──────────┐                                │
│       │              │ 速率限制  │ → 令牌桶/滑动窗口               │
│       │              └────┬─────┘                                │
│       │                   │                                      │
│       │                   ▼                                      │
│       │              ┌──────────┐    失败    ┌──────────┐        │
│       │              │ 执行工具  │──────────►│ 重试/降级  │        │
│       │              │ (超时控制)│           │ 错误恢复  │        │
│       │              └────┬─────┘           └──────────┘        │
│       │                   │ 成功                                 │
│       │                   ▼                                      │
│       │              ┌──────────┐                                │
│       │              │ 输出检查  │ → 脱敏/截断/格式化               │
│       │              └────┬─────┘                                │
│       │                   │                                      │
│       │                   ▼                                      │
│       │              ┌──────────┐                                │
│       │              │ 写入缓存  │                                │
│       │              │ 记录日志  │                                │
│       │              └────┬─────┘                                │
│       │                   │                                      │
│       ▼                   ▼                                      │
│  ┌──────────────────────────────┐                                │
│  │    LLM整合结果，生成回复      │                                │
│  └──────────────────────────────┘                                │
│       │                                                          │
│       ▼                                                          │
│  最终回复给用户                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 总结

本教程全面覆盖了工具调用（Function Calling）的核心内容：

1. **Function Calling概述**: LLM通过JSON Schema定义的工具接口，自动选择并调用函数，再将结果整合到回复中
2. **OpenAI实现**: 完整的工具定义、调用、并行调用和流式调用教程
3. **Claude Tool Use**: Anthropic API的工具调用实现，包括多轮工具调用循环
4. **自定义工具**: 天气查询、数据库操作、文件操作、API调用等实用工具的完整实现
5. **工具链组合**: 顺序链、并行链和条件链三种组合模式
6. **错误处理**: 完善的重试机制、错误分类和降级方案
7. **多工具Agent**: 天气+新闻+计算+翻译的完整多工具Agent实战

## 最佳实践

1. **工具描述要精确**: 清晰描述功能、参数和返回值，直接影响LLM调用准确性
2. **参数使用enum约束**: 对于有限选项的参数，使用enum限制取值范围
3. **必须添加错误处理**: 所有工具都应捕获异常并返回有意义的错误信息
4. **限制返回数据大小**: 避免工具返回过多数据消耗token
5. **安全优先**: 数据库工具只允许SELECT，文件工具限制操作目录
6. **使用降级方案**: 关键工具应有备选方案，提高系统可靠性
7. **监控和日志**: 记录每次工具调用的耗时、成功率等指标

## 参考资源

- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [LangChain Tools](https://python.langchain.com/docs/modules/tools/)
- [JSON Schema规范](https://json-schema.org/)
- [Tavily API](https://tavily.com/) - AI优化搜索引擎

---

**文件大小目标**: 30KB
**创建时间**: 2024-01-01
**最后更新**: 2024-01-01
