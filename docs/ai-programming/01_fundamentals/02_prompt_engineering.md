# Prompt工程完整教程

## 目录
1. [Prompt工程简介](#prompt工程简介)
2. [Prompt设计原则](#prompt设计原则)
3. [Few-shot Learning](#few-shot-learning)
4. [Chain-of-Thought (CoT)](#chain-of-thought-cot)
5. [ReAct模式](#react模式)
6. [Self-Consistency](#self-consistency)
7. [Tree of Thoughts](#tree-of-thoughts)
8. [50个实战Prompt模板](#50个实战prompt模板)

---

## Prompt工程简介

### 什么是Prompt工程

Prompt工程是设计和优化输入提示词的艺术和科学，目标是引导LLM产生更准确、更有用的输出。

### Prompt的组成部分

```
┌─────────────────────────────────────────────────────┐
│               完整Prompt结构                          │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. 角色定义 (Role)                                  │
│     "你是一个专业的Python开发专家"                     │
│                                                     │
│  2. 上下文 (Context)                                 │
│     "用户正在开发一个电商网站"                         │
│                                                     │
│  3. 任务描述 (Task)                                  │
│     "帮助优化数据库查询性能"                           │
│                                                     │
│  4. 输入数据 (Input)                                 │
│     "当前查询: SELECT * FROM orders..."              │
│                                                     │
│  5. 输出格式 (Format)                                │
│     "请以Markdown格式输出，包含代码示例"               │
│                                                     │
│  6. 约束条件 (Constraints)                           │
│     "不超过500字，使用简洁语言"                        │
│                                                     │
│  7. 示例 (Examples)                                 │
│     "输入: ... 输出: ..."                            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Prompt设计原则

### 1. 清晰明确原则

```python
# ❌ 不好的Prompt
prompt_bad = "告诉我关于Python的事情"

# ✅ 好的Prompt
prompt_good = """
作为Python专家，请解释以下概念:
1. 列表推导式的工作原理
2. 与传统for循环的性能对比
3. 提供3个实用示例

要求:
- 每个示例包含注释
- 说明适用场景
- 字数控制在300字以内
"""
```

### 2. 结构化原则

```python
def create_structured_prompt(task, context, constraints):
    """创建结构化Prompt"""
    return f"""
## 角色
你是一个经验丰富的软件架构师。

## 背景
{context}

## 任务
{task}

## 要求
{constraints}

## 输出格式
请按以下格式输出:
1. 问题分析
2. 解决方案
3. 代码示例
4. 注意事项
"""

# 使用示例
prompt = create_structured_prompt(
    task="设计一个高并发的订单系统",
    context="电商平台，日订单量100万+",
    constraints="使用微服务架构，保证数据一致性"
)
```

### 3. 分步引导原则

```python
# ❌ 一次性要求所有
prompt_bad = "创建一个完整的Web应用"

# ✅ 分步引导
prompt_good = """
让我们一步步创建Web应用:

步骤1: 需求分析
- 列出核心功能
- 确定技术栈

步骤2: 数据库设计
- 设计ER图
- 编写建表SQL

步骤3: API设计
- 定义RESTful接口
- 编写接口文档

步骤4: 前端实现
- 设计UI组件
- 实现交互逻辑

请从步骤1开始，完成后我会告诉你继续下一步。
"""
```

### 4. 示例驱动原则

```python
def create_example_driven_prompt(examples, task):
    """创建示例驱动的Prompt"""
    prompt = "请参考以下示例，完成类似任务:\n\n"

    for i, example in enumerate(examples, 1):
        prompt += f"## 示例{i}\n"
        prompt += f"输入: {example['input']}\n"
        prompt += f"输出: {example['output']}\n\n"

    prompt += f"## 当前任务\n{task}\n"

    return prompt

# 使用示例
examples = [
    {
        "input": "用户登录失败",
        "output": "ERROR_AUTH_001: 用户名或密码错误"
    },
    {
        "input": "数据库连接超时",
        "output": "ERROR_DB_002: 数据库连接超时，请稍后重试"
    }
]

prompt = create_example_driven_prompt(
    examples=examples,
    task="用户权限不足"
)
```

---

## Few-shot Learning

### Zero-shot (零样本)

```python
zero_shot_prompt = """
将以下文本分类为: 正面、负面或中性

文本: "这个产品质量一般，价格有点贵"
分类:
"""

# LLM会直接推理输出: 负面
```

### One-shot (单样本)

```python
one_shot_prompt = """
将文本分类为: 正面、负面或中性

示例:
文本: "非常满意，物超所值！"
分类: 正面

现在分类:
文本: "这个产品质量一般，价格有点贵"
分类:
"""
```

### Few-shot (多样本)

```python
few_shot_prompt = """
将文本分类为: 正面、负面或中性

示例1:
文本: "非常满意，物超所值！"
分类: 正面

示例2:
文本: "质量太差，退货了"
分类: 负面

示例3:
文本: "还可以吧，没什么特别的"
分类: 中性

现在分类:
文本: "这个产品质量一般，价格有点贵"
分类:
"""
```

### Few-shot实战: 实体提取

```python
class FewShotNER:
    """Few-shot命名实体识别"""

    def __init__(self):
        self.examples = [
            {
                "text": "苹果公司的CEO蒂姆·库克在加州宣布新产品",
                "entities": {
                    "组织": ["苹果公司"],
                    "人物": ["蒂姆·库克"],
                    "地点": ["加州"]
                }
            },
            {
                "text": "特斯拉在上海建立了超级工厂",
                "entities": {
                    "组织": ["特斯拉"],
                    "地点": ["上海"]
                }
            }
        ]

    def create_prompt(self, text):
        """创建Few-shot提示"""
        prompt = "从文本中提取命名实体 (组织、人物、地点):\n\n"

        # 添加示例
        for i, example in enumerate(self.examples, 1):
            prompt += f"示例{i}:\n"
            prompt += f"文本: {example['text']}\n"
            prompt += f"实体: {example['entities']}\n\n"

        # 添加待处理文本
        prompt += f"现在处理:\n"
        prompt += f"文本: {text}\n"
        prompt += f"实体:"

        return prompt

    def extract(self, text, client):
        """执行提取"""
        prompt = self.create_prompt(text)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return response.choices[0].message.content

# 使用示例
from openai import OpenAI

client = OpenAI()
ner = FewShotNER()

result = ner.extract(
    "微软的创始人比尔·盖茨在西雅图发表演讲"
)
print(result)
```

---

## Chain-of-Thought (CoT)

### 基础CoT

```python
# 没有CoT
prompt_no_cot = """
问题: 一个班级有30名学生，其中60%是女生。
女生中有40%参加了数学竞赛。有多少女生参加了数学竞赛？

答案:
"""

# 使用CoT
prompt_with_cot = """
问题: 一个班级有30名学生，其中60%是女生。
女生中有40%参加了数学竞赛。有多少女生参加了数学竞赛？

让我们一步步思考:
1. 首先，计算女生总数
2. 然后，计算参加竞赛的女生数

答案:
"""
```

### Zero-shot CoT

```python
zero_shot_cot = """
问题: 一个数的3倍加上5等于17，这个数是多少？

让我们一步步思考:
"""

# LLM会自动展开推理过程
```

### 复杂推理CoT

```python
class ChainOfThought:
    """CoT推理助手"""

    def __init__(self, client):
        self.client = client

    def solve_problem(self, problem, steps_hint=None):
        """使用CoT解决问题"""
        prompt = f"问题: {problem}\n\n"

        if steps_hint:
            prompt += "请按以下步骤思考:\n"
            for i, step in enumerate(steps_hint, 1):
                prompt += f"{i}. {step}\n"
            prompt += "\n"
        else:
            prompt += "让我们一步步思考:\n"

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return response.choices[0].message.content

# 使用示例
from openai import OpenAI

client = OpenAI()
cot = ChainOfThought(client)

# 数学问题
result = cot.solve_problem(
    problem="如果5个苹果花费10元，那么买8个苹果需要多少钱？",
    steps_hint=[
        "计算1个苹果的价格",
        "计算8个苹果的总价"
    ]
)
print(result)

# 逻辑推理
result = cot.solve_problem(
    problem="所有的狗都是动物。有些宠物是狗。那么有些宠物是动物吗？"
)
print(result)
```

### 多步推理示例

```python
multi_step_cot = """
问题: 一家公司去年收入100万，今年增长20%。
如果明年保持相同的增长率，明年收入是多少？

让我们分步计算:

步骤1: 计算今年收入
- 去年收入: 100万
- 增长率: 20%
- 今年收入 = 100万 × (1 + 20%) = 100万 × 1.2 = 120万

步骤2: 计算明年收入
- 今年收入: 120万
- 增长率: 20% (保持不变)
- 明年收入 = 120万 × (1 + 20%) = 120万 × 1.2 = 144万

答案: 明年收入是144万元。
"""
```

---

## ReAct模式

### ReAct原理

ReAct = Reasoning (推理) + Acting (行动)

```
┌──────────────────────────────────────────┐
│           ReAct循环流程                    │
├──────────────────────────────────────────┤
│                                          │
│  1. Thought (思考)                        │
│     ↓                                    │
│     "我需要先查找相关信息"                  │
│                                          │
│  2. Action (行动)                         │
│     ↓                                    │
│     调用工具: Search("Python教程")         │
│                                          │
│  3. Observation (观察)                    │
│     ↓                                    │
│     "找到了5个相关教程..."                 │
│                                          │
│  4. Thought (再思考)                      │
│     ↓                                    │
│     "现在我可以总结答案了"                  │
│                                          │
│  5. Final Answer (最终答案)               │
│                                          │
└──────────────────────────────────────────┘
```

### ReAct Prompt示例

```python
react_prompt = """
回答以下问题，你可以使用以下工具:

工具列表:
- Search(query): 搜索信息
- Calculator(expression): 计算数学表达式
- Wikipedia(topic): 查询维基百科

格式:
Thought: 思考当前需要做什么
Action: 工具名称[参数]
Observation: 工具返回的结果
... (重复Thought/Action/Observation)
Thought: 我现在知道最终答案了
Final Answer: 最终答案

问题: 北京和上海之间的距离是多少公里？两地的人口总和是多少？

Thought: 我需要先查找北京和上海之间的距离
Action: Search["北京上海距离"]
Observation: 北京和上海之间的直线距离约1067公里

Thought: 接下来查找两个城市的人口
Action: Wikipedia["北京人口"]
Observation: 北京市常住人口约2189万人

Thought: 继续查找上海人口
Action: Wikipedia["上海人口"]
Observation: 上海市常住人口约2489万人

Thought: 现在需要计算人口总和
Action: Calculator[2189 + 2489]
Observation: 4678

Thought: 我现在知道最终答案了
Final Answer: 北京和上海之间的距离约1067公里，两地人口总和约4678万人。
"""
```

### ReAct实现

```python
import re

class ReActAgent:
    """ReAct Agent实现"""

    def __init__(self, client, tools):
        self.client = client
        self.tools = tools  # {tool_name: function}

    def create_prompt(self, question, history=""):
        """创建ReAct提示"""
        tool_desc = "\n".join([
            f"- {name}: {func.__doc__}"
            for name, func in self.tools.items()
        ])

        return f"""
回答以下问题，你可以使用以下工具:

{tool_desc}

格式:
Thought: 思考当前需要做什么
Action: 工具名称[参数]
Observation: 工具返回的结果
... (重复Thought/Action/Observation)
Thought: 我现在知道最终答案了
Final Answer: 最终答案

{history}

问题: {question}
"""

    def parse_action(self, text):
        """解析Action"""
        match = re.search(r'Action:\s*(\w+)\[(.*?)\]', text)
        if match:
            tool_name = match.group(1)
            args = match.group(2).strip('\'"')
            return tool_name, args
        return None, None

    def run(self, question, max_steps=5):
        """运行ReAct循环"""
        history = ""

        for step in range(max_steps):
            # 生成Thought和Action
            prompt = self.create_prompt(question, history)

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                stop=["Observation:"]
            )

            text = response.choices[0].message.content
            history += text

            # 检查是否完成
            if "Final Answer:" in text:
                match = re.search(r'Final Answer:\s*(.*)', text, re.DOTALL)
                if match:
                    return match.group(1).strip()

            # 解析并执行Action
            tool_name, args = self.parse_action(text)

            if tool_name and tool_name in self.tools:
                result = self.tools[tool_name](args)
                history += f"\nObservation: {result}\n"
            else:
                history += "\nObservation: 工具调用失败\n"

        return "未能找到答案"

# 定义工具
def search(query):
    """搜索信息"""
    # 模拟搜索
    return f"关于'{query}'的搜索结果..."

def calculator(expression):
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except:
        return "计算错误"

# 使用示例
from openai import OpenAI

client = OpenAI()

agent = ReActAgent(
    client=client,
    tools={
        "Search": search,
        "Calculator": calculator
    }
)

answer = agent.run("100乘以250等于多少？")
print(answer)
```

---

## Self-Consistency

### 原理

通过多次采样，选择最一致的答案。

```
┌────────────────────────────────────────────┐
│         Self-Consistency流程                │
├────────────────────────────────────────────┤
│                                            │
│  问题 → 生成多个推理路径 (n次采样)           │
│                                            │
│  路径1: ... → 答案A                         │
│  路径2: ... → 答案A                         │
│  路径3: ... → 答案B                         │
│  路径4: ... → 答案A                         │
│  路径5: ... → 答案A                         │
│                                            │
│  投票: A出现4次，B出现1次                    │
│                                            │
│  最终答案: A (一致性最高)                    │
│                                            │
└────────────────────────────────────────────┘
```

### 实现

```python
from collections import Counter

class SelfConsistency:
    """Self-Consistency实现"""

    def __init__(self, client):
        self.client = client

    def generate_multiple_answers(self, question, n=5):
        """生成多个答案"""
        prompt = f"""
问题: {question}

让我们一步步思考，然后给出答案。

最终答案格式: ANSWER: [你的答案]
"""

        answers = []

        for i in range(n):
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7  # 增加多样性
            )

            text = response.choices[0].message.content

            # 提取答案
            match = re.search(r'ANSWER:\s*(.*)', text)
            if match:
                answer = match.group(1).strip()
                answers.append(answer)

        return answers

    def get_most_consistent(self, answers):
        """获取最一致的答案"""
        counter = Counter(answers)
        most_common = counter.most_common(1)[0]
        return most_common[0], most_common[1]

    def solve(self, question, n=5):
        """使用Self-Consistency解决问题"""
        answers = self.generate_multiple_answers(question, n)

        print(f"生成的{n}个答案:")
        for i, ans in enumerate(answers, 1):
            print(f"{i}. {ans}")

        final_answer, count = self.get_most_consistent(answers)

        print(f"\n最一致的答案 (出现{count}次): {final_answer}")

        return final_answer

# 使用示例
from openai import OpenAI

client = OpenAI()
sc = SelfConsistency(client)

answer = sc.solve(
    question="一个数的两倍加3等于11，这个数是多少？",
    n=5
)
```

---

## Tree of Thoughts

### 原理

探索多个思考路径，选择最优路径。

```
                        问题
                         │
         ┌───────────────┼───────────────┐
         │               │               │
       思路1            思路2            思路3
         │               │               │
    ┌────┴────┐     ┌────┴────┐     ┌────┴────┐
    │         │     │         │     │         │
  步骤1.1  步骤1.2 步骤2.1  步骤2.2 步骤3.1  步骤3.2
    │         │     │         │     │         │
  [评分]   [评分]  [评分]   [评分]  [评分]   [评分]
    │
  选择最优路径
    │
  继续扩展
    │
  最终答案
```

### 实现

```python
class TreeOfThoughts:
    """Tree of Thoughts实现"""

    def __init__(self, client):
        self.client = client

    def generate_thoughts(self, problem, context="", n=3):
        """生成多个思考路径"""
        prompt = f"""
问题: {problem}

{context}

请提供{n}个不同的解决思路。

格式:
思路1: ...
思路2: ...
思路3: ...
"""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )

        text = response.choices[0].message.content

        # 解析思路
        thoughts = []
        for i in range(1, n + 1):
            match = re.search(f'思路{i}:\s*(.*?)(?=思路{i+1}:|$)', text, re.DOTALL)
            if match:
                thoughts.append(match.group(1).strip())

        return thoughts

    def evaluate_thought(self, problem, thought):
        """评估思路质量"""
        prompt = f"""
问题: {problem}
思路: {thought}

请评估这个思路的质量，给出0-10分的评分。

评分标准:
- 可行性 (0-3分)
- 创新性 (0-3分)
- 完整性 (0-4分)

只需输出总分: [分数]
"""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        text = response.choices[0].message.content

        # 提取分数
        match = re.search(r'(\d+)', text)
        if match:
            return int(match.group(1))

        return 0

    def solve(self, problem, max_depth=2):
        """使用ToT解决问题"""
        print(f"问题: {problem}\n")

        # 第一层: 生成初始思路
        thoughts = self.generate_thoughts(problem)

        print("生成的思路:")
        for i, thought in enumerate(thoughts, 1):
            print(f"{i}. {thought}\n")

        # 评估思路
        scores = []
        for thought in thoughts:
            score = self.evaluate_thought(problem, thought)
            scores.append(score)
            print(f"思路{len(scores)}评分: {score}")

        # 选择最优思路
        best_idx = scores.index(max(scores))
        best_thought = thoughts[best_idx]

        print(f"\n选择的最优思路: {best_thought}\n")

        # 基于最优思路生成答案
        final_prompt = f"""
问题: {problem}
采用思路: {best_thought}

请详细展开这个思路，给出完整的解决方案。
"""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.7
        )

        return response.choices[0].message.content

# 使用示例
from openai import OpenAI

client = OpenAI()
tot = TreeOfThoughts(client)

solution = tot.solve(
    problem="如何在不增加服务器的情况下，将网站性能提升50%？"
)

print("\n最终解决方案:")
print(solution)
```

---

## 50个实战Prompt模板

### 1. 代码生成类

```python
code_templates = {
    "1. Python函数生成": """
编写一个Python函数:
- 功能: {功能描述}
- 输入参数: {参数列表}
- 返回值: {返回值说明}
- 要求: 包含类型提示、文档字符串、错误处理
""",

    "2. API接口设计": """
设计RESTful API接口:
- 资源: {资源名称}
- 操作: {CRUD操作}
- 请求格式: JSON
- 响应格式: JSON
- 包含: 路由、请求体、响应体、错误码
""",

    "3. SQL查询优化": """
优化以下SQL查询:
```sql
{原始SQL}
```

要求:
- 分析性能瓶颈
- 提供优化建议
- 给出优化后的SQL
- 说明性能提升
""",

    "4. 单元测试生成": """
为以下代码生成单元测试:
```python
{代码}
```

要求:
- 使用pytest框架
- 覆盖正常场景和边界情况
- 包含Mock对象
- 测试覆盖率>90%
""",

    "5. 代码审查": """
审查以下代码:
```{语言}
{代码}
```

请从以下方面分析:
1. 代码质量
2. 潜在Bug
3. 性能问题
4. 安全漏洞
5. 最佳实践建议
"""
}
```

### 2. 数据处理类

```python
data_templates = {
    "6. 数据清洗": """
清洗以下数据:
{数据示例}

要求:
- 处理缺失值
- 去除重复项
- 标准化格式
- 异常值检测
- 输出Python代码
""",

    "7. 数据分析": """
分析以下数据集:
{数据描述}

请提供:
1. 描述性统计
2. 数据分布可视化
3. 相关性分析
4. 异常点识别
5. 完整的Python代码
""",

    "8. 正则表达式": """
编写正则表达式提取:
- 目标内容: {目标描述}
- 样本文本: {文本示例}

要求:
- Python re模块语法
- 包含测试用例
- 处理边界情况
""",

    "9. 数据可视化": """
使用matplotlib/seaborn绘制:
- 图表类型: {图表类型}
- 数据: {数据描述}

要求:
- 美观的样式
- 清晰的标签
- 适当的颜色
- 完整代码
""",

    "10. ETL流程": """
设计ETL流程:
- 数据源: {数据源}
- 目标: {目标系统}
- 转换逻辑: {转换规则}

提供:
1. 流程图
2. Python实现
3. 错误处理
4. 性能优化
"""
}
```

### 3. 文档写作类

```python
doc_templates = {
    "11. README生成": """
为以下项目生成README.md:
- 项目名: {项目名}
- 功能: {功能列表}
- 技术栈: {技术栈}

包含:
1. 项目简介
2. 功能特性
3. 快速开始
4. API文档
5. 贡献指南
6. 许可证
""",

    "12. API文档": """
生成API文档:
- 端点: {API端点}
- 方法: {HTTP方法}
- 参数: {参数说明}

格式: OpenAPI 3.0规范
包含: 请求示例、响应示例、错误码
""",

    "13. 技术博客": """
撰写技术博客:
- 主题: {技术主题}
- 受众: {目标读者}
- 字数: 1500-2000字

结构:
1. 引言
2. 问题背景
3. 解决方案
4. 代码示例
5. 最佳实践
6. 总结
""",

    "14. 代码注释": """
为以下代码添加注释:
```{语言}
{代码}
```

要求:
- 函数级文档字符串
- 复杂逻辑行注释
- 参数和返回值说明
- 遵循{语言}注释规范
""",

    "15. 错误信息": """
设计用户友好的错误信息:
- 错误类型: {错误类型}
- 技术细节: {技术信息}

提供:
1. 用户可读的描述
2. 可能的原因
3. 解决建议
4. 帮助链接
"""
}
```

### 4. 问题解决类

```python
problem_templates = {
    "16. 调试助手": """
帮助调试以下问题:
- 错误信息: {错误信息}
- 相关代码: {代码片段}
- 环境: {环境信息}

请提供:
1. 错误原因分析
2. 调试步骤
3. 解决方案
4. 预防措施
""",

    "17. 性能优化": """
优化以下代码性能:
```{语言}
{代码}
```

当前性能: {性能指标}
目标: {目标指标}

分析:
1. 性能瓶颈
2. 优化方案
3. 优化后代码
4. 性能对比
""",

    "18. 架构设计": """
设计系统架构:
- 需求: {需求描述}
- 约束: {约束条件}
- 规模: {用户规模/数据量}

提供:
1. 架构图
2. 组件说明
3. 技术选型
4. 扩展方案
5. 风险评估
""",

    "19. 安全加固": """
对以下系统进行安全加固:
- 系统类型: {系统类型}
- 当前问题: {安全问题}

提供:
1. 漏洞分析
2. 加固方案
3. 代码示例
4. 安全检查清单
""",

    "20. 技术选型": """
为以下场景选择技术方案:
- 场景: {应用场景}
- 要求: {技术要求}
- 约束: {限制条件}

对比分析:
1. 候选方案
2. 优劣对比
3. 推荐方案
4. 实施建议
"""
}
```

### 5. 学习教学类

```python
learning_templates = {
    "21. 概念解释": """
解释以下技术概念:
- 概念: {概念名称}
- 受众: {受众水平}

要求:
1. 简单类比
2. 核心原理
3. 实际应用
4. 代码示例
5. 深入资源
""",

    "22. 教程编写": """
编写{主题}教程:
- 难度: {初级/中级/高级}
- 时长: {预计学习时间}

结构:
1. 学习目标
2. 前置知识
3. 分步讲解
4. 实战项目
5. 练习题
6. 进阶方向
""",

    "23. 代码讲解": """
讲解以下代码:
```{语言}
{代码}
```

面向: {目标群体}

内容:
1. 整体功能
2. 逐行解释
3. 关键概念
4. 运行示例
5. 扩展思考
""",

    "24. 面试准备": """
准备{主题}面试题:
- 岗位: {岗位名称}
- 级别: {职级}

提供:
1. 核心知识点
2. 高频面试题 (10题)
3. 详细答案
4. 延伸问题
5. 项目案例
""",

    "25. 学习路线": """
制定{领域}学习路线:
- 当前水平: {当前水平}
- 目标: {学习目标}
- 时间: {可用时间}

包含:
1. 阶段划分
2. 学习资源
3. 实战项目
4. 检验标准
5. 时间规划
"""
}
```

### 6-10. 更多模板

```python
# 25个额外模板
advanced_templates = {
    # 需求分析
    "26. 用户故事": "作为{角色}，我想要{功能}，以便{价值}",
    "27. 功能规格": "详细描述{功能}的输入、处理、输出",

    # 测试相关
    "28. 测试用例": "为{功能}设计测试用例，覆盖正常/异常/边界",
    "29. 性能测试": "设计{系统}的性能测试方案",
    "30. 安全测试": "设计{应用}的安全测试用例",

    # 运维相关
    "31. 部署脚本": "编写{应用}的自动化部署脚本",
    "32. 监控配置": "配置{系统}的监控告警",
    "33. 日志分析": "分析日志文件，发现{问题}",

    # 沟通协作
    "34. 技术方案": "撰写{功能}的技术方案文档",
    "35. 代码评审意见": "对{PR}提供建设性的评审意见",
    "36. 周报总结": "总结本周工作，包含{项目}进展",

    # 工具脚本
    "37. 自动化脚本": "编写自动化脚本完成{任务}",
    "38. 数据迁移": "设计从{源}到{目标}的数据迁移方案",
    "39. 批处理": "批量处理{数据类型}",

    # 前端相关
    "40. React组件": "创建{组件}的React组件",
    "41. CSS样式": "为{元素}编写响应式CSS",
    "42. 前端优化": "优化{页面}的加载性能",

    # 后端相关
    "43. 数据库设计": "为{业务}设计数据库表结构",
    "44. 缓存策略": "设计{场景}的缓存方案",
    "45. 消息队列": "使用{MQ}处理{任务}",

    # AI相关
    "46. Prompt优化": "优化以下Prompt: {原始Prompt}",
    "47. 模型选择": "为{任务}选择合适的AI模型",
    "48. RAG应用": "设计{领域}的RAG系统",

    # 综合应用
    "49. 系统集成": "将{系统A}与{系统B}集成",
    "50. 技术调研": "调研{技术}的应用场景和最佳实践"
}
```

---

## 完整示例: Prompt模板引擎

```python
class PromptTemplateEngine:
    """Prompt模板引擎"""

    def __init__(self):
        self.templates = {}
        self._load_templates()

    def _load_templates(self):
        """加载所有模板"""
        self.templates.update(code_templates)
        self.templates.update(data_templates)
        self.templates.update(doc_templates)
        self.templates.update(problem_templates)
        self.templates.update(learning_templates)

    def list_templates(self, category=None):
        """列出模板"""
        if category:
            return {k: v for k, v in self.templates.items()
                   if category in k}
        return list(self.templates.keys())

    def get_template(self, name):
        """获取模板"""
        return self.templates.get(name)

    def fill_template(self, name, **kwargs):
        """填充模板"""
        template = self.get_template(name)
        if not template:
            return f"模板'{name}'不存在"

        try:
            return template.format(**kwargs)
        except KeyError as e:
            return f"缺少参数: {e}"

    def generate_prompt(self, template_name, client, **params):
        """生成并执行Prompt"""
        prompt = self.fill_template(template_name, **params)

        if "不存在" in prompt or "缺少参数" in prompt:
            return prompt

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

# 使用示例
from openai import OpenAI

client = OpenAI()
engine = PromptTemplateEngine()

# 列出所有模板
print("可用模板:")
print(engine.list_templates())

# 使用模板生成代码
result = engine.generate_prompt(
    template_name="1. Python函数生成",
    client=client,
    功能描述="计算列表中所有偶数的和",
    参数列表="numbers: List[int]",
    返回值说明="int - 偶数之和"
)

print("\n生成的代码:")
print(result)

# 使用模板优化SQL
result = engine.generate_prompt(
    template_name="3. SQL查询优化",
    client=client,
    原始SQL="SELECT * FROM orders WHERE user_id = 123"
)

print("\n优化建议:")
print(result)
```

---

## 总结

Prompt工程的关键要点:

1. **结构化设计**: 明确角色、任务、格式、约束
2. **Few-shot学习**: 提供示例引导输出
3. **CoT推理**: 引导逐步思考
4. **ReAct模式**: 结合推理和工具调用
5. **Self-Consistency**: 通过多次采样提高准确性
6. **Tree of Thoughts**: 探索多个解决路径
7. **模板复用**: 建立Prompt模板库

## 最佳实践

1. 从简单开始，逐步优化
2. 使用具体示例而非抽象描述
3. 明确输出格式
4. 适当控制温度参数
5. 迭代改进Prompt

## 参考资源

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Library](https://docs.anthropic.com/claude/prompt-library)
- [PromptPerfect](https://promptperfect.jina.ai/)
