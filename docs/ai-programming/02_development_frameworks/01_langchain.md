# LangChain完整开发教程

## 目录
1. [LangChain简介](#langchain简介)
2. [核心架构](#核心架构)
3. [LCEL表达式语言](#lcel表达式语言)
4. [Chains链式调用](#chains链式调用)
5. [Memory记忆系统](#memory记忆系统)
6. [Agents智能代理](#agents智能代理)
7. [完整电商客服示例](#完整电商客服示例)

---

## LangChain简介

### 什么是LangChain

LangChain是一个用于开发大语言模型应用的开源框架，提供了标准化的接口和工具链。

### 安装

```bash
pip install langchain langchain-openai langchain-community
pip install langchain-anthropic  # Claude支持
pip install chromadb  # 向量数据库
pip install faiss-cpu  # Facebook向量搜索
```

### 核心概念

```
┌─────────────────────────────────────────────────────┐
│           LangChain核心组件架构                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐ │
│  │  Models  │      │  Prompts │      │  Chains  │ │
│  │  模型封装 │      │  提示模板 │      │  流程编排 │ │
│  └─────┬────┘      └─────┬────┘      └─────┬────┘ │
│        │                 │                  │      │
│        └────────┬────────┴──────┬──────────┘      │
│                 │                │                 │
│          ┌──────▼─────┐   ┌──────▼──────┐         │
│          │   Memory   │   │   Agents    │         │
│          │   记忆管理  │   │   智能代理   │         │
│          └──────┬─────┘   └──────┬──────┘         │
│                 │                 │                │
│                 └────────┬────────┘                │
│                          │                         │
│                   ┌──────▼──────┐                  │
│                   │  Retrievers │                  │
│                   │  信息检索    │                  │
│                   └─────────────┘                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 核心架构

### Models模型层

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage

# OpenAI模型
llm_openai = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000
)

# Claude模型
llm_claude = ChatAnthropic(
    model="claude-3-opus-20240229",
    max_tokens=1000
)

# 基础调用
messages = [
    SystemMessage(content="你是一个Python专家"),
    HumanMessage(content="如何读取CSV文件？")
]

response = llm_openai.invoke(messages)
print(response.content)
```

### Prompts提示模板

```python
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts import FewShotPromptTemplate

# 简单模板
prompt = PromptTemplate(
    input_variables=["product", "language"],
    template="将{product}的产品描述翻译成{language}。"
)

formatted = prompt.format(product="iPhone", language="中文")
print(formatted)

# 对话模板
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}专家。"),
    ("human", "{question}"),
])

messages = chat_prompt.format_messages(
    role="数据分析",
    question="如何处理缺失值？"
)

# Few-shot模板
examples = [
    {
        "input": "happy",
        "output": "sad"
    },
    {
        "input": "tall",
        "output": "short"
    }
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="给出单词的反义词:",
    suffix="Input: {word}\nOutput:",
    input_variables=["word"]
)

print(few_shot_prompt.format(word="big"))
```

### 输出解析器

```python
from langchain.output_parsers import PydanticOutputParser, StructuredOutputParser
from langchain.output_parsers import ResponseSchema
from pydantic import BaseModel, Field

# Pydantic解析器
class Person(BaseModel):
    name: str = Field(description="人名")
    age: int = Field(description="年龄")
    occupation: str = Field(description="职业")

parser = PydanticOutputParser(pydantic_object=Person)

prompt = PromptTemplate(
    template="提取信息:\n{format_instructions}\n\n文本: {text}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 结构化解析器
response_schemas = [
    ResponseSchema(name="name", description="人名"),
    ResponseSchema(name="age", description="年龄"),
    ResponseSchema(name="occupation", description="职业")
]

structured_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = structured_parser.get_format_instructions()
```

---

## LCEL表达式语言

### 什么是LCEL

LangChain Expression Language (LCEL) 是一种声明式的链式调用语法。

### 基础链式调用

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# 创建组件
model = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
output_parser = StrOutputParser()

# LCEL链式组合
chain = prompt | model | output_parser

# 调用
result = chain.invoke({"topic": "程序员"})
print(result)
```

### 复杂链式调用

```python
from operator import itemgetter

# 多输入链
prompt1 = ChatPromptTemplate.from_template("将{text}翻译成{language}")
prompt2 = ChatPromptTemplate.from_template("将以下文本总结为一句话:\n{translated_text}")

# 组合链
chain = (
    {"text": itemgetter("text"), "language": itemgetter("language")}
    | prompt1
    | model
    | StrOutputParser()
    | (lambda translated: {"translated_text": translated})
    | prompt2
    | model
    | StrOutputParser()
)

result = chain.invoke({
    "text": "LangChain is a framework for developing applications powered by language models.",
    "language": "中文"
})

print(result)
```

### RunnablePassthrough

```python
from langchain.schema.runnable import RunnablePassthrough

# 传递上下文
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# RAG链
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

vectorstore = FAISS.from_texts(
    ["LangChain是一个开发框架", "它支持多种LLM"],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | ChatPromptTemplate.from_template(
        "基于以下上下文回答问题:\n{context}\n\n问题: {question}"
    )
    | model
    | StrOutputParser()
)

result = rag_chain.invoke("什么是LangChain?")
print(result)
```

### Runnable分支

```python
from langchain.schema.runnable import RunnableBranch

# 条件分支
branch = RunnableBranch(
    (
        lambda x: "代码" in x["topic"],
        ChatPromptTemplate.from_template("写一段{topic}代码") | model
    ),
    (
        lambda x: "故事" in x["topic"],
        ChatPromptTemplate.from_template("讲一个{topic}") | model
    ),
    ChatPromptTemplate.from_template("介绍{topic}") | model
)

# 测试不同分支
result1 = branch.invoke({"topic": "Python代码"})
print(result1.content)

result2 = branch.invoke({"topic": "童话故事"})
print(result2.content)
```

---

## Chains链式调用

### LLMChain

```python
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4")

prompt = PromptTemplate(
    input_variables=["product"],
    template="为{product}写一句宣传语"
)

chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run(product="智能手表")
print(result)
```

### SequentialChain

```python
from langchain.chains import SequentialChain

# 第一个链: 生成概要
synopsis_prompt = PromptTemplate(
    input_variables=["title"],
    template="为电影《{title}》写一句话概要"
)
synopsis_chain = LLMChain(
    llm=llm,
    prompt=synopsis_prompt,
    output_key="synopsis"
)

# 第二个链: 生成评论
review_prompt = PromptTemplate(
    input_variables=["synopsis"],
    template="基于以下概要写一篇影评:\n{synopsis}"
)
review_chain = LLMChain(
    llm=llm,
    prompt=review_prompt,
    output_key="review"
)

# 组合链
overall_chain = SequentialChain(
    chains=[synopsis_chain, review_chain],
    input_variables=["title"],
    output_variables=["synopsis", "review"],
    verbose=True
)

result = overall_chain({"title": "黑客帝国"})
print("概要:", result["synopsis"])
print("评论:", result["review"])
```

### TransformChain

```python
from langchain.chains import TransformChain

def transform_func(inputs):
    """自定义转换函数"""
    text = inputs["text"]
    # 文本清理
    cleaned = text.strip().lower()
    return {"cleaned_text": cleaned}

transform_chain = TransformChain(
    input_variables=["text"],
    output_variables=["cleaned_text"],
    transform=transform_func
)

result = transform_chain.run(text="  HELLO WORLD  ")
print(result)
```

### RouterChain

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate

# 定义不同的处理链
physics_template = """你是一个物理学专家。回答问题:
{input}"""

math_template = """你是一个数学专家。回答问题:
{input}"""

prompt_infos = [
    {
        "name": "physics",
        "description": "适合回答物理问题",
        "prompt_template": physics_template
    },
    {
        "name": "math",
        "description": "适合回答数学问题",
        "prompt_template": math_template
    }
]

# 创建目标链
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt = PromptTemplate(
        template=p_info["prompt_template"],
        input_variables=["input"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

# 默认链
default_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template="回答问题: {input}",
        input_variables=["input"]
    )
)

# 创建路由链
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

router_template = f"""给定用户问题，选择最合适的专家回答。

可选专家:
{destinations_str}

问题: {{input}}

选择:"""

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser()
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

# 组合多提示链
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True
)

# 测试路由
print(chain.run("什么是牛顿第一定律？"))
print(chain.run("求解方程 2x + 5 = 13"))
```

---

## Memory记忆系统

### ConversationBufferMemory

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# 创建记忆
memory = ConversationBufferMemory()

# 创建对话链
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 多轮对话
conversation.predict(input="你好，我叫张三")
conversation.predict(input="我喜欢编程")
conversation.predict(input="我叫什么名字？")

# 查看记忆
print(memory.buffer)
```

### ConversationBufferWindowMemory

```python
from langchain.memory import ConversationBufferWindowMemory

# 只保留最近K轮对话
memory = ConversationBufferWindowMemory(k=2)

conversation = ConversationChain(
    llm=llm,
    memory=memory
)

conversation.predict(input="第1轮对话")
conversation.predict(input="第2轮对话")
conversation.predict(input="第3轮对话")
conversation.predict(input="第4轮对话")

# 只会记住最近2轮
print(memory.buffer)
```

### ConversationSummaryMemory

```python
from langchain.memory import ConversationSummaryMemory

# 自动总结历史对话
memory = ConversationSummaryMemory(llm=llm)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.predict(input="你好，我是一名软件工程师，专注于后端开发")
conversation.predict(input="我最近在学习微服务架构")
conversation.predict(input="我对Docker和Kubernetes很感兴趣")

# 查看总结
print(memory.buffer)
```

### VectorStoreMemory

```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# 基于向量检索的记忆
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts([], embeddings)

memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# 保存对话
memory.save_context(
    {"input": "我最喜欢的编程语言是Python"},
    {"output": "很好！Python是一门非常流行的语言"}
)

memory.save_context(
    {"input": "我在学习机器学习"},
    {"output": "机器学习是AI的重要分支"}
)

# 基于语义检索记忆
relevant_memories = memory.load_memory_variables(
    {"input": "告诉我关于我学习的内容"}
)

print(relevant_memories)
```

---

## Agents智能代理

### Agent类型

```
┌────────────────────────────────────────────────┐
│            LangChain Agent类型                  │
├────────────────────────────────────────────────┤
│                                                │
│  1. Zero-shot ReAct                            │
│     • 无示例的推理+行动Agent                     │
│     • 最通用，但可能不够准确                     │
│                                                │
│  2. Structured Tool Chat                       │
│     • 支持多参数的结构化工具                     │
│     • 适合复杂工具调用                          │
│                                                │
│  3. OpenAI Functions                           │
│     • 使用OpenAI的Function Calling              │
│     • 最可靠，推荐使用                          │
│                                                │
│  4. Plan-and-Execute                           │
│     • 先规划再执行                              │
│     • 适合复杂任务                              │
│                                                │
└────────────────────────────────────────────────┘
```

### 创建工具

```python
from langchain.tools import Tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field

# 简单工具
def get_word_length(word: str) -> int:
    """返回单词的长度"""
    return len(word)

word_length_tool = Tool(
    name="WordLength",
    func=get_word_length,
    description="返回单词的长度"
)

# 结构化工具
class CalculatorInput(BaseModel):
    a: int = Field(description="第一个数字")
    b: int = Field(description="第二个数字")
    operation: str = Field(description="操作: add, subtract, multiply, divide")

def calculator(a: int, b: int, operation: str) -> float:
    """执行数学运算"""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b if b != 0 else "错误: 除数为0"

calculator_tool = StructuredTool.from_function(
    func=calculator,
    name="Calculator",
    description="执行基础数学运算",
    args_schema=CalculatorInput
)
```

### OpenAI Functions Agent

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# 定义工具
def search_database(query: str) -> str:
    """搜索产品数据库"""
    # 模拟数据库查询
    database = {
        "iPhone": "最新款iPhone 15，价格5999元",
        "MacBook": "MacBook Pro M3，价格12999元"
    }
    return database.get(query, "未找到产品")

def calculate_discount(price: float, discount: float) -> float:
    """计算折扣价格"""
    return price * (1 - discount)

tools = [
    Tool(
        name="SearchDatabase",
        func=search_database,
        description="搜索产品信息，输入产品名称"
    ),
    Tool(
        name="CalculateDiscount",
        func=calculate_discount,
        description="计算折扣后价格，输入原价和折扣比例(0-1)"
    )
]

# 创建Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的客服助手，帮助用户查询产品信息。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# 执行
result = agent_executor.invoke({
    "input": "查询iPhone的价格，并计算8折后的价格"
})

print(result["output"])
```

### 自定义Agent

```python
from langchain.agents import AgentExecutor, BaseAgent
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union

class CustomAgent(BaseAgent):
    """自定义Agent"""

    @property
    def input_keys(self):
        return ["input"]

    def plan(
        self,
        intermediate_steps: List[tuple],
        **kwargs
    ) -> Union[AgentAction, AgentFinish]:
        """规划下一步行动"""
        user_input = kwargs["input"]

        # 简单规则
        if "天气" in user_input:
            return AgentAction(
                tool="Weather",
                tool_input=user_input,
                log="需要查询天气"
            )
        elif "计算" in user_input:
            return AgentAction(
                tool="Calculator",
                tool_input=user_input,
                log="需要计算"
            )
        else:
            return AgentFinish(
                return_values={"output": "我不知道如何处理这个请求"},
                log="无法处理"
            )

    async def aplan(self, *args, **kwargs):
        """异步规划"""
        raise NotImplementedError()
```

---

## 完整电商客服示例

### 系统架构

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from typing import Dict, List
import json

class EcommerceCustomerService:
    """电商智能客服系统"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # 模拟数据库
        self.products = {
            "iPhone 15": {
                "price": 5999,
                "stock": 50,
                "description": "最新款苹果手机"
            },
            "MacBook Pro": {
                "price": 12999,
                "stock": 30,
                "description": "M3芯片笔记本电脑"
            },
            "AirPods Pro": {
                "price": 1999,
                "stock": 100,
                "description": "主动降噪无线耳机"
            }
        }

        self.orders = {}
        self.order_counter = 1000

        # 初始化Agent
        self.agent_executor = self._create_agent()

    def _create_tools(self) -> List[Tool]:
        """创建工具"""
        tools = [
            Tool(
                name="SearchProduct",
                func=self.search_product,
                description="搜索产品信息。输入: 产品名称"
            ),
            Tool(
                name="CheckStock",
                func=self.check_stock,
                description="检查产品库存。输入: 产品名称"
            ),
            Tool(
                name="CalculatePrice",
                func=self.calculate_price,
                description="计算订单总价。输入: JSON格式的产品列表，例如: {'iPhone 15': 2, 'AirPods Pro': 1}"
            ),
            Tool(
                name="CreateOrder",
                func=self.create_order,
                description="创建订单。输入: JSON格式的产品列表和用户信息"
            ),
            Tool(
                name="TrackOrder",
                func=self.track_order,
                description="查询订单状态。输入: 订单号"
            )
        ]
        return tools

    def search_product(self, product_name: str) -> str:
        """搜索产品"""
        product = self.products.get(product_name)
        if product:
            return json.dumps({
                "name": product_name,
                "price": product["price"],
                "description": product["description"],
                "in_stock": product["stock"] > 0
            }, ensure_ascii=False)
        return "未找到该产品"

    def check_stock(self, product_name: str) -> str:
        """检查库存"""
        product = self.products.get(product_name)
        if product:
            return f"{product_name}的库存数量: {product['stock']}"
        return "未找到该产品"

    def calculate_price(self, items_json: str) -> str:
        """计算总价"""
        try:
            items = json.loads(items_json)
            total = 0
            details = []

            for product_name, quantity in items.items():
                if product_name in self.products:
                    price = self.products[product_name]["price"]
                    subtotal = price * quantity
                    total += subtotal
                    details.append(f"{product_name} x {quantity} = ¥{subtotal}")

            result = "\n".join(details)
            result += f"\n总计: ¥{total}"
            return result

        except Exception as e:
            return f"计算错误: {str(e)}"

    def create_order(self, order_json: str) -> str:
        """创建订单"""
        try:
            order_data = json.loads(order_json)
            items = order_data.get("items", {})

            # 检查库存
            for product_name, quantity in items.items():
                if product_name not in self.products:
                    return f"错误: 产品{product_name}不存在"

                if self.products[product_name]["stock"] < quantity:
                    return f"错误: {product_name}库存不足"

            # 创建订单
            order_id = f"ORD{self.order_counter}"
            self.order_counter += 1

            total = sum(
                self.products[name]["price"] * qty
                for name, qty in items.items()
            )

            order = {
                "order_id": order_id,
                "items": items,
                "total": total,
                "status": "已确认",
                "user_info": order_data.get("user_info", {})
            }

            self.orders[order_id] = order

            # 扣减库存
            for product_name, quantity in items.items():
                self.products[product_name]["stock"] -= quantity

            return f"订单创建成功！订单号: {order_id}, 总金额: ¥{total}"

        except Exception as e:
            return f"创建订单失败: {str(e)}"

    def track_order(self, order_id: str) -> str:
        """追踪订单"""
        order = self.orders.get(order_id)
        if order:
            return json.dumps(order, ensure_ascii=False, indent=2)
        return "未找到该订单"

    def _create_agent(self) -> AgentExecutor:
        """创建Agent执行器"""
        tools = self._create_tools()

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的电商客服助手。你可以:
1. 查询产品信息和价格
2. 检查库存
3. 帮助用户计算订单金额
4. 创建订单
5. 查询订单状态

请始终保持礼貌和专业。当用户想要购买时，确认订单详情后再创建订单。
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_openai_functions_agent(self.llm, tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5
        )

    def chat(self, user_message: str) -> str:
        """处理用户消息"""
        response = self.agent_executor.invoke({"input": user_message})
        return response["output"]

# 使用示例
if __name__ == "__main__":
    customer_service = EcommerceCustomerService()

    print("电商智能客服系统已启动！\n")

    # 模拟对话
    conversations = [
        "你好，有什么产品推荐吗？",
        "iPhone 15多少钱？",
        "还有库存吗？",
        "我想买2台iPhone 15和1个AirPods Pro，一共多少钱？",
        "好的，帮我下单。我的地址是北京市朝阳区xxx"
    ]

    for msg in conversations:
        print(f"\n用户: {msg}")
        response = customer_service.chat(msg)
        print(f"客服: {response}")
```

---

## 总结

LangChain提供了构建LLM应用的完整工具链:

1. **Models**: 统一的模型接口
2. **Prompts**: 强大的模板系统
3. **LCEL**: 声明式链式调用
4. **Chains**: 多种预构建的处理链
5. **Memory**: 灵活的记忆管理
6. **Agents**: 智能任务规划和执行

## 最佳实践

1. 使用LCEL构建可维护的链
2. 根据需求选择合适的Memory类型
3. 优先使用OpenAI Functions Agent
4. 为工具提供清晰的描述
5. 合理设置max_iterations避免无限循环

## 参考资源

- [LangChain官方文档](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LCEL教程](https://python.langchain.com/docs/expression_language/)
