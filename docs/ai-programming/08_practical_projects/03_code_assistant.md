# AI编程助手 - 完整实战

## 目录
1. [项目概述](#项目概述)
2. [代码生成引擎](#代码生成引擎)
3. [Bug检测与修复](#bug检测与修复)
4. [代码重构建议](#代码重构建议)
5. [代码审查系统](#代码审查系统)
6. [FastAPI后端服务](#fastapi后端服务)
7. [VSCode插件开发](#vscode插件开发)
8. [测试与部署](#测试与部署)

---

## 项目概述

### 项目背景

AI编程助手通过大语言模型理解代码上下文，提供代码生成、Bug修复、重构建议、
代码审查等功能。本项目构建一个完整的AI编程助手后端服务，支持多种编程语言，
可通过API集成到IDE或命令行工具中。

### 系统总体架构

```
+============================================================================+
|                       AI编程助手 - 系统架构                                 |
+============================================================================+
|                                                                            |
|   接入层                                                                   |
|   +-----------------+  +-----------------+  +-----------------+            |
|   |  VSCode 插件    |  |  命令行工具     |  |  Web IDE        |            |
|   |  (Extension)    |  |  (CLI)          |  |  (Monaco)       |            |
|   +--------+--------+  +--------+--------+  +--------+--------+            |
|            |                     |                    |                     |
|            +----------+----------+----------+---------+                     |
|                       |                     |                               |
|                       v                     v                               |
|   服务层                                                                   |
|   +----------------------------------------------------------------+       |
|   |                    FastAPI 后端服务                              |       |
|   |                                                                 |       |
|   |  +--------------+  +--------------+  +--------------+          |       |
|   |  | 代码生成     |  | Bug修复      |  | 代码重构     |          |       |
|   |  | Code Gen     |  | Bug Fix      |  | Refactor     |          |       |
|   |  +------+-------+  +------+-------+  +------+-------+          |       |
|   |         |                  |                 |                  |       |
|   |  +------+-------+  +------+-------+  +------+-------+         |       |
|   |  | 代码审查     |  | 单元测试     |  | 文档生成     |         |       |
|   |  | Code Review  |  | Test Gen     |  | Doc Gen      |         |       |
|   |  +------+-------+  +------+-------+  +------+-------+         |       |
|   |         |                  |                 |                  |       |
|   |  +------+------------------+-----------------+------+          |       |
|   |  |            Prompt 工程引擎                       |          |       |
|   |  |  - 上下文构建  - Few-shot示例  - 输出解析        |          |       |
|   |  +------+------------------------------------------+          |       |
|   +---------|--------------------------------------------------+   |       |
|             |                                                       |       |
|   AI层     v                                                       |       |
|   +---------+--------------------------------------------------+   |       |
|   |  +------+------+  +-------------+  +------------------+    |   |       |
|   |  |   OpenAI    |  |  本地模型   |  | 代码分析工具     |    |   |       |
|   |  |  GPT-4      |  | DeepSeek    |  | AST解析/Lint    |    |   |       |
|   |  |  API        |  | CodeLlama   |  | 依赖分析        |    |   |       |
|   |  +-------------+  +-------------+  +------------------+    |   |       |
|   +-------------------------------------------------------------+   |       |
|                                                                      |       |
+============================================================================+
```

### 核心功能

| 功能 | 说明 | 输入 | 输出 |
|------|------|------|------|
| 代码生成 | 根据自然语言描述生成代码 | 需求描述 + 语言 | 完整代码 |
| Bug修复 | 分析错误代码并提供修复方案 | 代码 + 错误信息 | 修复后的代码 |
| 代码重构 | 识别代码坏味道并优化 | 源代码 | 重构后的代码 + 说明 |
| 代码审查 | 自动审查代码质量和安全性 | 源代码/diff | 审查报告 |
| 单元测试 | 自动生成测试用例 | 源代码 | 测试代码 |
| 文档生成 | 为函数/类生成文档字符串 | 源代码 | 带文档的代码 |

### 配置与依赖

```python
"""
AI编程助手 - 项目配置
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class CodeAssistantConfig:
    """编程助手配置"""

    # AI模型配置
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model: str = "gpt-4o-mini"
    max_tokens: int = 4096
    temperature: float = 0.2          # 代码生成用低温度保证准确性

    # 支持的编程语言
    supported_languages: List[str] = field(default_factory=lambda: [
        "python", "javascript", "typescript", "java", "go",
        "rust", "c", "cpp", "csharp", "sql", "bash",
    ])

    # 代码分析配置
    max_code_length: int = 10000      # 最大代码长度(字符)
    max_context_files: int = 5        # 最多关联文件数
    enable_ast_analysis: bool = True  # 启用AST分析

    # 安全配置
    blocked_patterns: List[str] = field(default_factory=lambda: [
        "os.system(", "subprocess.call(", "eval(",
        "exec(", "__import__(",
    ])


config = CodeAssistantConfig()

# requirements.txt
REQUIREMENTS = """
fastapi==0.104.1
uvicorn==0.24.0
openai==1.6.1
pydantic==2.5.3
httpx==0.25.2
python-multipart==0.0.6
"""

if __name__ == "__main__":
    print(f"[配置] 模型: {config.model}")
    print(f"[配置] 支持语言: {config.supported_languages}")
    print(f"[配置] 最大代码长度: {config.max_code_length}")
```

---

## 代码生成引擎

### 代码生成流程

```
+=========================================================================+
|                       代码生成引擎流程                                    |
+=========================================================================+
|                                                                         |
|  用户输入: "用Python写一个快速排序算法"                                 |
|       |                                                                 |
|       v                                                                 |
|  +----+-------------------------+                                       |
|  |     需求分析                  |                                       |
|  |  - 提取编程语言: Python       |                                       |
|  |  - 提取任务类型: 算法实现     |                                       |
|  |  - 提取关键需求: 快速排序     |                                       |
|  +----+-------------------------+                                       |
|       |                                                                 |
|       v                                                                 |
|  +----+-------------------------+                                       |
|  |     上下文构建                |                                       |
|  |  - 语言特定的编码规范         |                                       |
|  |  - Few-shot 示例              |                                       |
|  |  - 项目上下文(可选)           |                                       |
|  +----+-------------------------+                                       |
|       |                                                                 |
|       v                                                                 |
|  +----+-------------------------+                                       |
|  |     Prompt 组装               |                                       |
|  |  System: 你是资深Python工程师 |                                       |
|  |  User: 需求 + 规范 + 示例     |                                       |
|  +----+-------------------------+                                       |
|       |                                                                 |
|       v                                                                 |
|  +----+-------------------------+                                       |
|  |     LLM 生成代码              |                                       |
|  +----+-------------------------+                                       |
|       |                                                                 |
|       v                                                                 |
|  +----+-------------------------+                                       |
|  |     后处理                    |                                       |
|  |  - 提取代码块                 |                                       |
|  |  - 格式化(black/prettier)    |                                       |
|  |  - 安全检查(敏感API)          |                                       |
|  |  - 语法验证(AST解析)          |                                       |
|  +----+-------------------------+                                       |
|       |                                                                 |
|       v                                                                 |
|  返回: 代码 + 解释 + 使用示例                                          |
|                                                                         |
+=========================================================================+
```

### 代码生成器实现

```python
"""
代码生成引擎 - 根据自然语言描述生成高质量代码
"""
import re
import ast
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class GeneratedCode:
    """生成的代码结果"""
    code: str                          # 生成的代码
    language: str                      # 编程语言
    explanation: str = ""              # 代码解释
    usage_example: str = ""            # 使用示例
    dependencies: List[str] = field(default_factory=list)  # 依赖包
    warnings: List[str] = field(default_factory=list)      # 警告信息
    is_valid: bool = True              # 语法是否有效


# ============================================================
# 语言特定的Prompt模板
# ============================================================

LANGUAGE_PROMPTS = {
    "python": {
        "system": """你是一个资深Python工程师。请遵循以下规范:
- 使用Python 3.10+语法
- 遵循PEP 8编码规范
- 添加类型注解(Type Hints)
- 编写清晰的docstring
- 处理异常情况
- 代码简洁高效""",
        "example_style": """
# 示例风格:
def calculate_average(numbers: list[float]) -> float:
    \"\"\"计算数字列表的平均值。

    Args:
        numbers: 数字列表

    Returns:
        平均值

    Raises:
        ValueError: 列表为空时
    \"\"\"
    if not numbers:
        raise ValueError("列表不能为空")
    return sum(numbers) / len(numbers)
""",
    },
    "javascript": {
        "system": """你是一个资深JavaScript/TypeScript工程师。请遵循以下规范:
- 使用ES2022+语法
- 使用const/let，不使用var
- 添加JSDoc注释
- 处理异步操作使用async/await
- 适当使用箭头函数""",
        "example_style": """
// 示例风格:
/**
 * 计算数组平均值
 * @param {number[]} numbers - 数字数组
 * @returns {number} 平均值
 */
const calculateAverage = (numbers) => {
    if (!numbers.length) throw new Error('数组不能为空');
    return numbers.reduce((sum, n) => sum + n, 0) / numbers.length;
};
""",
    },
    "java": {
        "system": """你是一个资深Java工程师。请遵循以下规范:
- 使用Java 17+语法
- 遵循Google Java Style Guide
- 添加Javadoc注释
- 合理使用泛型
- 处理checked exception""",
        "example_style": "",
    },
}


class CodeExtractor:
    """从LLM输出中提取代码块"""

    @staticmethod
    def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
        """
        提取Markdown代码块

        返回: [(语言, 代码内容), ...]
        """
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return [(lang or "text", code.strip()) for lang, code in matches]

        # 如果没有代码块标记，尝试检测整段代码
        lines = text.strip().split("\n")
        code_lines = [l for l in lines if not l.startswith("#") or l.startswith("# ")]
        if code_lines:
            return [("text", "\n".join(code_lines))]

        return []

    @staticmethod
    def extract_main_code(text: str, target_language: str = "") -> str:
        """提取主要代码块"""
        blocks = CodeExtractor.extract_code_blocks(text)
        if not blocks:
            return text.strip()

        # 优先返回目标语言的代码
        for lang, code in blocks:
            if target_language and lang.lower() == target_language.lower():
                return code

        # 返回最长的代码块
        return max(blocks, key=lambda x: len(x[1]))[1]


class PythonValidator:
    """Python代码验证器"""

    @staticmethod
    def check_syntax(code: str) -> Tuple[bool, str]:
        """检查Python语法是否正确"""
        try:
            ast.parse(code)
            return True, "语法正确"
        except SyntaxError as e:
            return False, f"语法错误: 第{e.lineno}行 - {e.msg}"

    @staticmethod
    def check_security(code: str) -> List[str]:
        """检查代码安全性"""
        warnings = []
        dangerous_patterns = [
            ("os.system(", "使用os.system可能导致命令注入，建议使用subprocess.run"),
            ("eval(", "eval()执行任意代码，存在安全风险"),
            ("exec(", "exec()执行任意代码，存在安全风险"),
            ("pickle.loads(", "反序列化不可信数据可能导致远程代码执行"),
            ("__import__(", "动态导入可能被滥用"),
        ]
        for pattern, warning in dangerous_patterns:
            if pattern in code:
                warnings.append(f"[安全警告] {warning}")
        return warnings

    @staticmethod
    def extract_imports(code: str) -> List[str]:
        """提取代码中的import依赖"""
        imports = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split(".")[0])
        except SyntaxError:
            # 回退到正则匹配
            for line in code.split("\n"):
                m = re.match(r'^(?:from|import)\s+(\w+)', line.strip())
                if m:
                    imports.add(m.group(1))

        # 过滤标准库
        stdlib = {"os", "sys", "re", "json", "math", "time", "datetime",
                  "typing", "collections", "functools", "itertools",
                  "pathlib", "dataclasses", "abc", "enum", "uuid", "hashlib"}
        return sorted(imports - stdlib)


class CodeGenerator:
    """
    代码生成器

    核心功能:
    1. 根据自然语言描述生成代码
    2. 支持多种编程语言
    3. 自动验证语法和安全性
    4. 提取依赖信息
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        if HAS_OPENAI:
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", "your-key"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )
        else:
            self.client = None

    def generate(
        self,
        description: str,
        language: str = "python",
        context: str = "",
        style_guide: str = "",
    ) -> GeneratedCode:
        """
        生成代码

        参数:
            description: 自然语言描述
            language: 编程语言
            context: 项目上下文(现有代码/API说明)
            style_guide: 编码规范
        """
        # 1. 获取语言特定Prompt
        lang_config = LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS.get("python"))
        system_prompt = lang_config["system"]
        example_style = lang_config.get("example_style", "")

        # 2. 构建用户消息
        user_message = f"请用 {language} 实现以下功能:\n\n{description}"

        if context:
            user_message += f"\n\n项目上下文:\n{context}"
        if style_guide:
            user_message += f"\n\n编码规范:\n{style_guide}"
        if example_style:
            user_message += f"\n\n请参考以下代码风格:\n{example_style}"

        user_message += "\n\n请提供:\n1. 完整的可运行代码\n2. 简要的代码解释\n3. 使用示例"

        # 3. 调用LLM
        if self.client:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
                max_tokens=4096,
            )
            raw_output = response.choices[0].message.content
        else:
            raw_output = self._mock_generate(description, language)

        # 4. 提取代码
        code = CodeExtractor.extract_main_code(raw_output, language)

        # 5. 提取解释(代码块之外的文字)
        explanation = re.sub(r'```[\s\S]*?```', '', raw_output).strip()

        # 6. 验证和安全检查
        warnings = []
        is_valid = True

        if language == "python":
            syntax_ok, syntax_msg = PythonValidator.check_syntax(code)
            is_valid = syntax_ok
            if not syntax_ok:
                warnings.append(syntax_msg)

            security_warnings = PythonValidator.check_security(code)
            warnings.extend(security_warnings)

            dependencies = PythonValidator.extract_imports(code)
        else:
            dependencies = []

        return GeneratedCode(
            code=code,
            language=language,
            explanation=explanation[:500],
            usage_example="",
            dependencies=dependencies,
            warnings=warnings,
            is_valid=is_valid,
        )

    def _mock_generate(self, description: str, language: str) -> str:
        """无API时的模拟生成"""
        if "排序" in description or "sort" in description.lower():
            return """```python
def quick_sort(arr: list) -> list:
    \"\"\"快速排序算法

    Args:
        arr: 待排序的列表

    Returns:
        排序后的新列表

    Examples:
        >>> quick_sort([3, 6, 8, 10, 1, 2, 1])
        [1, 1, 2, 3, 6, 8, 10]
    \"\"\"
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)


# 使用示例
if __name__ == "__main__":
    data = [3, 6, 8, 10, 1, 2, 1]
    sorted_data = quick_sort(data)
    print(f"排序前: {data}")
    print(f"排序后: {sorted_data}")
```

这是一个标准的快速排序实现，使用了列表推导式使代码更加简洁。
时间复杂度: 平均O(n log n)，最坏O(n^2)。
空间复杂度: O(n)（因为创建了新列表）。"""

        elif "api" in description.lower() or "接口" in description:
            return """```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="示例API")

class Item(BaseModel):
    name: str
    price: float
    description: Optional[str] = None

items_db: dict[int, Item] = {}
counter = 0

@app.post("/items/", response_model=dict)
async def create_item(item: Item):
    global counter
    counter += 1
    items_db[counter] = item
    return {"id": counter, "item": item.model_dump()}

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"id": item_id, "item": items_db[item_id].model_dump()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

这是一个基础的FastAPI CRUD接口实现。"""

        return f"""```{language}
# TODO: 实现 {description}
# 请设置 OPENAI_API_KEY 环境变量以获取AI生成的代码
print("示例代码占位")
```

请配置API密钥以获得完整的代码生成。"""


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    print("=== AI代码生成器 测试 ===\n")

    generator = CodeGenerator()

    # 测试1: 生成排序算法
    print("[测试1] 生成快速排序")
    result = generator.generate("实现一个快速排序算法", language="python")
    print(f"  语言: {result.language}")
    print(f"  语法有效: {result.is_valid}")
    print(f"  依赖: {result.dependencies}")
    print(f"  警告: {result.warnings}")
    print(f"  代码预览:\n{result.code[:200]}...")
    print()

    # 测试2: 生成API
    print("[测试2] 生成API")
    result = generator.generate("创建一个商品管理的REST API接口", language="python")
    print(f"  语法有效: {result.is_valid}")
    print(f"  依赖: {result.dependencies}")
    print(f"  代码长度: {len(result.code)} 字符")
```

---

## Bug检测与修复

### Bug修复架构

```
+=========================================================================+
|                       Bug检测与修复流程                                   |
+=========================================================================+
|                                                                         |
|  输入: 代码 + 错误信息(可选)                                           |
|       |                                                                 |
|       v                                                                 |
|  +----+-----------------------------+                                   |
|  |     静态分析                      |                                   |
|  |  - AST语法树解析                  |                                   |
|  |  - 类型检查(mypy/pyright)        |                                   |
|  |  - 代码风格检查(flake8)          |                                   |
|  +----+-----------------------------+                                   |
|       |                                                                 |
|       v                                                                 |
|  +----+-----------------------------+                                   |
|  |     LLM Bug检测                   |                                   |
|  |  - 逻辑错误检测                   |                                   |
|  |  - 边界条件分析                   |                                   |
|  |  - 并发安全检查                   |                                   |
|  |  - 资源泄漏检测                   |                                   |
|  +----+-----------------------------+                                   |
|       |                                                                 |
|       v                                                                 |
|  +----+-----------------------------+                                   |
|  |     生成修复方案                   |                                   |
|  |  - 修复后的完整代码               |                                   |
|  |  - Bug原因说明                    |                                   |
|  |  - Diff对比(修改了哪些行)         |                                   |
|  +----+-----------------------------+                                   |
|       |                                                                 |
|       v                                                                 |
|  返回: 修复代码 + 原因分析 + Diff                                      |
|                                                                         |
+=========================================================================+
```

### Bug修复引擎

```python
"""
Bug检测与修复引擎 - 自动分析代码问题并提供修复方案
"""
import ast
import re
import difflib
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class BugSeverity(str, Enum):
    """Bug严重等级"""
    CRITICAL = "critical"       # 崩溃/安全漏洞
    HIGH = "high"               # 功能错误
    MEDIUM = "medium"           # 逻辑缺陷
    LOW = "low"                 # 代码风格
    INFO = "info"               # 建议


@dataclass
class BugReport:
    """Bug报告"""
    bug_id: int
    severity: BugSeverity
    line_number: int = -1
    description: str = ""
    suggestion: str = ""
    category: str = ""           # 类别: logic/security/performance/style


@dataclass
class FixResult:
    """修复结果"""
    original_code: str
    fixed_code: str
    bugs_found: List[BugReport] = field(default_factory=list)
    diff: str = ""
    explanation: str = ""
    is_fixed: bool = True


class StaticAnalyzer:
    """
    静态代码分析器

    使用AST和正则表达式检测常见问题
    """

    def analyze_python(self, code: str) -> List[BugReport]:
        """分析Python代码中的常见问题"""
        bugs: List[BugReport] = []
        lines = code.split("\n")

        # 1. 语法检查
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            bugs.append(BugReport(
                bug_id=1,
                severity=BugSeverity.CRITICAL,
                line_number=e.lineno or -1,
                description=f"语法错误: {e.msg}",
                category="syntax",
            ))
            return bugs  # 语法错误时无法继续分析

        # 2. 检查未使用的变量
        assigned_vars = set()
        used_vars = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned_vars.add(target.id)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_vars.add(node.id)

        unused = assigned_vars - used_vars - {"_", "__all__", "__name__"}
        for var in unused:
            bugs.append(BugReport(
                bug_id=len(bugs) + 1,
                severity=BugSeverity.LOW,
                description=f"变量 '{var}' 已赋值但未使用",
                category="style",
                suggestion=f"移除未使用的变量 '{var}'，或以 _ 开头表示有意忽略",
            ))

        # 3. 检查裸except
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                bugs.append(BugReport(
                    bug_id=len(bugs) + 1,
                    severity=BugSeverity.MEDIUM,
                    line_number=node.lineno,
                    description="使用了裸 except，会捕获所有异常包括 KeyboardInterrupt",
                    category="logic",
                    suggestion="指定具体异常类型，如 except ValueError:",
                ))

        # 4. 检查可变默认参数
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults + node.args.kw_defaults:
                    if default and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        bugs.append(BugReport(
                            bug_id=len(bugs) + 1,
                            severity=BugSeverity.HIGH,
                            line_number=node.lineno,
                            description=f"函数 '{node.name}' 使用了可变默认参数",
                            category="logic",
                            suggestion="使用 None 作为默认值，在函数体内初始化",
                        ))

        # 5. 检查常见安全问题
        security_patterns = [
            (r'\beval\s*\(', "使用eval()存在代码注入风险", BugSeverity.CRITICAL),
            (r'\bexec\s*\(', "使用exec()存在代码注入风险", BugSeverity.CRITICAL),
            (r'password\s*=\s*["\']', "代码中硬编码了密码", BugSeverity.CRITICAL),
            (r'api_key\s*=\s*["\'][a-zA-Z0-9]', "代码中硬编码了API密钥", BugSeverity.CRITICAL),
        ]
        for pattern, desc, severity in security_patterns:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line) and not line.strip().startswith("#"):
                    bugs.append(BugReport(
                        bug_id=len(bugs) + 1,
                        severity=severity,
                        line_number=i,
                        description=desc,
                        category="security",
                    ))

        return bugs


class BugFixer:
    """
    Bug修复引擎

    结合静态分析和LLM，检测并自动修复代码bug
    """

    FIX_SYSTEM_PROMPT = """你是一个资深代码审查专家。请分析以下代码中的Bug并提供修复。

要求:
1. 找出所有Bug(语法错误、逻辑错误、安全问题、性能问题)
2. 提供修复后的完整代码
3. 解释每个Bug的原因和修复方案
4. 保持代码风格一致

输出格式:
1. 先列出发现的问题
2. 然后给出修复后的完整代码(用```python代码块包裹)
3. 最后简要说明修复了什么"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.static_analyzer = StaticAnalyzer()
        if HAS_OPENAI:
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", "your-key"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )
        else:
            self.client = None

    def fix(
        self,
        code: str,
        error_message: str = "",
        language: str = "python",
    ) -> FixResult:
        """
        分析并修复代码Bug

        参数:
            code: 源代码
            error_message: 运行时错误信息(可选)
            language: 编程语言
        """
        # 1. 静态分析
        static_bugs = []
        if language == "python":
            static_bugs = self.static_analyzer.analyze_python(code)

        # 2. LLM分析修复
        user_message = f"请分析并修复以下 {language} 代码中的Bug:\n\n```{language}\n{code}\n```"
        if error_message:
            user_message += f"\n\n运行时报错信息:\n{error_message}"

        if self.client:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.FIX_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            llm_output = response.choices[0].message.content
        else:
            llm_output = self._mock_fix(code, error_message)

        # 3. 提取修复后的代码
        fixed_code = CodeExtractor.extract_main_code(llm_output, language)
        if not fixed_code or fixed_code == code:
            fixed_code = code  # 未修改

        # 4. 生成diff
        diff = self._generate_diff(code, fixed_code)

        # 5. 提取解释
        explanation = re.sub(r'```[\s\S]*?```', '', llm_output).strip()

        return FixResult(
            original_code=code,
            fixed_code=fixed_code,
            bugs_found=static_bugs,
            diff=diff,
            explanation=explanation[:800],
            is_fixed=(fixed_code != code),
        )

    def _generate_diff(self, original: str, fixed: str) -> str:
        """生成代码差异对比"""
        diff_lines = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            fixed.splitlines(keepends=True),
            fromfile="original.py",
            tofile="fixed.py",
            lineterm="",
        ))
        return "\n".join(diff_lines)

    def _mock_fix(self, code: str, error: str) -> str:
        """无API时的模拟修复"""
        # 模拟修复可变默认参数
        if "def " in code and "=[]" in code or "={}" in code:
            fixed = code.replace("=[]", "=None").replace("={}", "=None")
            # 查找函数并添加初始化
            return f"""发现的问题:
1. 使用了可变默认参数，可能导致意外的共享状态

修复后的代码:
```python
{fixed}
```

修复说明: 将可变默认参数(list/dict)替换为None，在函数体内初始化。"""

        return f"""未发现明显Bug。

```python
{code}
```

建议: 代码看起来没有明显的语法或逻辑错误。如有运行时问题，请提供错误信息。"""


# 复用之前定义的 CodeExtractor
class CodeExtractor:
    @staticmethod
    def extract_code_blocks(text: str) -> list:
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return [(lang or "text", code.strip()) for lang, code in matches] if matches else []

    @staticmethod
    def extract_main_code(text: str, target_language: str = "") -> str:
        blocks = CodeExtractor.extract_code_blocks(text)
        if not blocks:
            return text.strip()
        for lang, code in blocks:
            if target_language and lang.lower() == target_language.lower():
                return code
        return max(blocks, key=lambda x: len(x[1]))[1] if blocks else text.strip()


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    print("=== Bug修复引擎 测试 ===\n")

    fixer = BugFixer()

    # 测试: 带Bug的代码
    buggy_code = """
def process_items(items, result=[]):
    for item in items:
        result.append(item * 2)
    return result

def divide(a, b):
    return a / b

try:
    data = divide(10, 0)
except:
    print("发生错误")
    data = None

password = "admin123"
"""

    print("[测试] 分析有Bug的代码")
    result = fixer.fix(buggy_code.strip())

    print(f"\n发现 {len(result.bugs_found)} 个问题:")
    for bug in result.bugs_found:
        print(f"  [{bug.severity.value}] 第{bug.line_number}行: {bug.description}")
        if bug.suggestion:
            print(f"    建议: {bug.suggestion}")

    print(f"\n已修复: {result.is_fixed}")
    if result.diff:
        print(f"\nDiff:\n{result.diff[:500]}")
```

---

## 代码重构建议

### 重构分析架构

```
+=========================================================================+
|                        代码重构分析流程                                   |
+=========================================================================+
|                                                                         |
|  输入: 源代码                                                           |
|       |                                                                 |
|       v                                                                 |
|  +----+-------------------------------+                                 |
|  |     代码质量评估                    |                                 |
|  |  - 圈复杂度计算                     |                                 |
|  |  - 函数长度检查                     |                                 |
|  |  - 重复代码检测                     |                                 |
|  |  - 命名规范检查                     |                                 |
|  +----+-------------------------------+                                 |
|       |                                                                 |
|       v                                                                 |
|  +----+-------------------------------+                                 |
|  |     代码坏味道识别                  |                                 |
|  |                                     |                                 |
|  |  [长函数]      [深层嵌套]           |                                 |
|  |  [重复代码]    [过大的类]           |                                 |
|  |  [过长参数]    [数据泥团]           |                                 |
|  |  [魔术数字]    [注释掉的代码]       |                                 |
|  +----+-------------------------------+                                 |
|       |                                                                 |
|       v                                                                 |
|  +----+-------------------------------+                                 |
|  |     LLM 重构建议                    |                                 |
|  |  - 提取函数/方法                    |                                 |
|  |  - 引入设计模式                     |                                 |
|  |  - 简化条件逻辑                     |                                 |
|  |  - 优化数据结构                     |                                 |
|  +----+-------------------------------+                                 |
|       |                                                                 |
|       v                                                                 |
|  返回: 重构后代码 + 质量评分 + 改进说明                                 |
|                                                                         |
+=========================================================================+
```

### 重构引擎实现

```python
"""
代码重构引擎 - 分析代码质量并提供重构建议
"""
import ast
import re
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class SmellType(str, Enum):
    """代码坏味道类型"""
    LONG_FUNCTION = "long_function"
    DEEP_NESTING = "deep_nesting"
    DUPLICATE_CODE = "duplicate_code"
    MAGIC_NUMBER = "magic_number"
    LONG_PARAMETER = "long_parameter_list"
    DEAD_CODE = "dead_code"
    COMPLEX_CONDITION = "complex_condition"
    POOR_NAMING = "poor_naming"


@dataclass
class CodeSmell:
    """代码坏味道"""
    smell_type: SmellType
    location: str                   # 位置描述
    description: str
    severity: str = "medium"        # low/medium/high
    suggestion: str = ""


@dataclass
class QualityMetrics:
    """代码质量指标"""
    total_lines: int = 0
    code_lines: int = 0             # 不含空行和注释
    function_count: int = 0
    class_count: int = 0
    avg_function_length: float = 0.0
    max_function_length: int = 0
    max_nesting_depth: int = 0
    cyclomatic_complexity: int = 0   # 圈复杂度
    smells: List[CodeSmell] = field(default_factory=list)
    overall_score: float = 0.0       # 0-100 质量评分


@dataclass
class RefactorResult:
    """重构结果"""
    original_code: str
    refactored_code: str
    metrics_before: QualityMetrics
    improvements: List[str] = field(default_factory=list)
    explanation: str = ""


class CodeQualityAnalyzer:
    """
    代码质量分析器

    计算各种质量指标，识别代码坏味道
    """

    def __init__(
        self,
        max_function_lines: int = 50,
        max_nesting_depth: int = 4,
        max_params: int = 5,
    ):
        self.max_function_lines = max_function_lines
        self.max_nesting_depth = max_nesting_depth
        self.max_params = max_params

    def analyze(self, code: str) -> QualityMetrics:
        """分析Python代码质量"""
        metrics = QualityMetrics()
        lines = code.split("\n")
        metrics.total_lines = len(lines)
        metrics.code_lines = len([l for l in lines if l.strip() and not l.strip().startswith("#")])

        try:
            tree = ast.parse(code)
        except SyntaxError:
            metrics.overall_score = 0
            return metrics

        # 统计函数和类
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node)
                metrics.function_count += 1
            elif isinstance(node, ast.ClassDef):
                metrics.class_count += 1

        # 分析每个函数
        function_lengths = []
        for func in functions:
            # 函数长度
            func_lines = func.end_lineno - func.lineno + 1 if hasattr(func, 'end_lineno') and func.end_lineno else 0
            function_lengths.append(func_lines)

            # 长函数检测
            if func_lines > self.max_function_lines:
                metrics.smells.append(CodeSmell(
                    smell_type=SmellType.LONG_FUNCTION,
                    location=f"函数 '{func.name}' 第{func.lineno}行",
                    description=f"函数有 {func_lines} 行，超过 {self.max_function_lines} 行限制",
                    severity="high",
                    suggestion="考虑将函数拆分成多个小函数，每个函数只做一件事",
                ))

            # 参数过多
            param_count = len(func.args.args)
            if param_count > self.max_params:
                metrics.smells.append(CodeSmell(
                    smell_type=SmellType.LONG_PARAMETER,
                    location=f"函数 '{func.name}'",
                    description=f"函数有 {param_count} 个参数，超过 {self.max_params} 个限制",
                    severity="medium",
                    suggestion="考虑使用数据类(dataclass)或字典来封装参数",
                ))

        if function_lengths:
            metrics.avg_function_length = sum(function_lengths) / len(function_lengths)
            metrics.max_function_length = max(function_lengths)

        # 检查魔术数字
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if node.value not in (0, 1, -1, 2, 100, True, False, None):
                    # 检查是否在赋值语句中（常量定义）
                    metrics.smells.append(CodeSmell(
                        smell_type=SmellType.MAGIC_NUMBER,
                        location=f"第{node.lineno}行" if hasattr(node, 'lineno') else "未知位置",
                        description=f"魔术数字 {node.value}",
                        severity="low",
                        suggestion="将魔术数字提取为具名常量",
                    ))

        # 检测嵌套深度
        max_depth = self._calc_max_depth(tree)
        metrics.max_nesting_depth = max_depth
        if max_depth > self.max_nesting_depth:
            metrics.smells.append(CodeSmell(
                smell_type=SmellType.DEEP_NESTING,
                location="整体",
                description=f"最大嵌套深度为 {max_depth} 层，超过 {self.max_nesting_depth} 层限制",
                severity="high",
                suggestion="使用早返回(guard clause)或提取函数来降低嵌套深度",
            ))

        # 计算圈复杂度(简化版)
        metrics.cyclomatic_complexity = self._calc_complexity(tree)

        # 计算质量评分
        metrics.overall_score = self._calc_score(metrics)

        return metrics

    def _calc_max_depth(self, tree: ast.AST, depth: int = 0) -> int:
        """计算最大嵌套深度"""
        max_d = depth
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                child_depth = self._calc_max_depth(node, depth + 1)
                max_d = max(max_d, child_depth)
            else:
                child_depth = self._calc_max_depth(node, depth)
                max_d = max(max_d, child_depth)
        return max_d

    def _calc_complexity(self, tree: ast.AST) -> int:
        """计算圈复杂度(简化版)"""
        complexity = 1  # 基础复杂度
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.IfExp)):
                complexity += 1
            elif isinstance(node, (ast.For, ast.While)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity

    def _calc_score(self, metrics: QualityMetrics) -> float:
        """计算质量评分(0-100)"""
        score = 100.0

        # 扣分项
        high_smells = len([s for s in metrics.smells if s.severity == "high"])
        medium_smells = len([s for s in metrics.smells if s.severity == "medium"])
        low_smells = len([s for s in metrics.smells if s.severity == "low"])

        score -= high_smells * 15
        score -= medium_smells * 8
        score -= low_smells * 3

        # 复杂度扣分
        if metrics.cyclomatic_complexity > 20:
            score -= 15
        elif metrics.cyclomatic_complexity > 10:
            score -= 8

        return max(0.0, min(100.0, round(score, 1)))


class RefactorEngine:
    """
    代码重构引擎

    结合静态分析和LLM，提供代码重构建议
    """

    REFACTOR_PROMPT = """你是一个代码重构专家。请分析代码质量并提供重构建议。

要求:
1. 识别代码中的坏味道(长函数、深嵌套、重复代码等)
2. 提供重构后的完整代码
3. 说明每处重构的原因和收益
4. 保持功能不变，只改善结构

输出格式:
1. 质量评估(简要)
2. 重构后的完整代码(```python代码块)
3. 改进说明列表"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.analyzer = CodeQualityAnalyzer()
        if HAS_OPENAI:
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", "your-key"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )
        else:
            self.client = None

    def refactor(self, code: str, focus: str = "") -> RefactorResult:
        """
        分析并重构代码

        参数:
            code: 源代码
            focus: 重点关注方面(performance/readability/pattern)
        """
        # 1. 质量分析
        metrics = self.analyzer.analyze(code)

        # 2. LLM重构
        user_msg = f"请重构以下Python代码:\n\n```python\n{code}\n```"
        if focus:
            user_msg += f"\n\n重点关注: {focus}"
        if metrics.smells:
            smell_list = "\n".join(f"- {s.smell_type.value}: {s.description}" for s in metrics.smells[:5])
            user_msg += f"\n\n已检测到的问题:\n{smell_list}"

        if self.client:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.REFACTOR_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=4096,
            )
            output = response.choices[0].message.content
        else:
            output = self._mock_refactor(code, metrics)

        # 提取重构代码
        refactored = CodeExtractor.extract_main_code(output, "python")
        explanation = re.sub(r'```[\s\S]*?```', '', output).strip()

        # 提取改进列表
        improvements = re.findall(r'[-*]\s*(.+)', explanation)

        return RefactorResult(
            original_code=code,
            refactored_code=refactored if refactored != code else code,
            metrics_before=metrics,
            improvements=improvements[:10],
            explanation=explanation[:600],
        )

    def _mock_refactor(self, code: str, metrics: QualityMetrics) -> str:
        """无API时的模拟重构"""
        improvements = []
        if metrics.smells:
            for s in metrics.smells[:3]:
                improvements.append(f"- {s.suggestion}" if s.suggestion else f"- 修复 {s.description}")

        return f"""质量评估: {metrics.overall_score}/100

```python
{code}
```

改进建议:
{chr(10).join(improvements) if improvements else '- 代码质量良好，无需重大重构'}"""


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    print("=== 代码重构引擎 测试 ===\n")

    engine = RefactorEngine()

    # 测试代码
    sample_code = """
def process_data(data, flag, mode, threshold, output_path, verbose):
    results = []
    for item in data:
        if item is not None:
            if flag:
                if mode == "fast":
                    if item > threshold:
                        results.append(item * 2)
                    else:
                        results.append(item)
                elif mode == "slow":
                    if item > threshold:
                        results.append(item * 3)
                    else:
                        results.append(item * 1.5)
            else:
                results.append(item)
    if verbose:
        print(f"Processed {len(results)} items")
    return results
"""

    result = engine.refactor(sample_code.strip())

    print(f"质量评分: {result.metrics_before.overall_score}/100")
    print(f"代码坏味道: {len(result.metrics_before.smells)} 个")
    for smell in result.metrics_before.smells:
        print(f"  [{smell.severity}] {smell.smell_type.value}: {smell.description}")
    print(f"\n改进建议: {len(result.improvements)} 条")
    for imp in result.improvements:
        print(f"  {imp}")
```

---

## 代码审查系统

### 审查流程

```
+=========================================================================+
|                        AI代码审查流程                                     |
+=========================================================================+
|                                                                         |
|  输入: Git Diff / Pull Request / 代码文件                               |
|       |                                                                 |
|       v                                                                 |
|  +----+----------------------------------------------+                  |
|  |  审查维度                                          |                  |
|  |                                                    |                  |
|  |  +-----------+  +-----------+  +-----------+      |                  |
|  |  | 正确性    |  | 安全性    |  | 可维护性  |      |                  |
|  |  | - 逻辑Bug |  | - SQL注入 |  | - 可读性  |      |                  |
|  |  | - 边界    |  | - XSS     |  | - 命名    |      |                  |
|  |  | - 并发    |  | - 认证    |  | - 注释    |      |                  |
|  |  +-----------+  +-----------+  +-----------+      |                  |
|  |                                                    |                  |
|  |  +-----------+  +-----------+  +-----------+      |                  |
|  |  | 性能      |  | 测试      |  | 规范      |      |                  |
|  |  | - 复杂度  |  | - 覆盖率  |  | - PEP8    |      |                  |
|  |  | - 内存    |  | - 测试质量|  | - 类型注解|      |                  |
|  |  | - N+1     |  |           |  |           |      |                  |
|  |  +-----------+  +-----------+  +-----------+      |                  |
|  +----+----------------------------------------------+                  |
|       |                                                                 |
|       v                                                                 |
|  +----+----------------------------------------------+                  |
|  |  审查报告                                          |                  |
|  |  - 问题列表(严重/中等/轻微)                        |                  |
|  |  - 逐行批注                                        |                  |
|  |  - 总体评价                                        |                  |
|  |  - 改进建议                                        |                  |
|  +---------------------------------------------------+                  |
|                                                                         |
+=========================================================================+
```

### 代码审查引擎

```python
"""
AI代码审查系统 - 自动审查代码质量、安全性、性能
"""
import re
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class ReviewSeverity(str, Enum):
    BLOCKER = "blocker"         # 必须修改
    WARNING = "warning"         # 建议修改
    SUGGESTION = "suggestion"   # 可选优化
    PRAISE = "praise"           # 写得好的地方


@dataclass
class ReviewComment:
    """审查评论"""
    severity: ReviewSeverity
    line_number: int = -1
    category: str = ""           # correctness/security/performance/style
    message: str = ""
    suggestion: str = ""


@dataclass
class ReviewReport:
    """审查报告"""
    file_name: str = ""
    language: str = "python"
    comments: List[ReviewComment] = field(default_factory=list)
    summary: str = ""
    overall_rating: str = ""      # approve/changes_requested
    score: float = 0.0            # 0-10
    review_time: str = ""


class CodeReviewer:
    """
    AI代码审查器

    功能:
    1. 代码正确性检查
    2. 安全漏洞扫描
    3. 性能问题检测
    4. 编码规范检查
    5. 生成审查报告
    """

    REVIEW_PROMPT = """你是一个严格的高级代码审查员。请对以下代码进行全面审查。

审查维度:
1. 正确性: 逻辑错误、边界条件、异常处理
2. 安全性: SQL注入、XSS、认证绕过、数据泄露
3. 性能: 时间复杂度、内存使用、N+1查询、不必要的IO
4. 可维护性: 可读性、命名、注释、代码结构
5. 最佳实践: 设计模式、SOLID原则、DRY原则

输出格式(JSON):
{
  "score": 7.5,
  "overall_rating": "changes_requested",
  "summary": "总体评价...",
  "comments": [
    {"severity": "blocker", "line": 15, "category": "security", "message": "...", "suggestion": "..."},
    {"severity": "warning", "line": 23, "category": "performance", "message": "...", "suggestion": "..."},
    {"severity": "praise", "line": 5, "category": "style", "message": "好的命名规范"}
  ]
}"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        if HAS_OPENAI:
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", "your-key"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )
        else:
            self.client = None

    def review(self, code: str, file_name: str = "code.py", language: str = "python") -> ReviewReport:
        """对代码进行审查"""

        # 调用LLM或使用模拟
        if self.client:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.REVIEW_PROMPT},
                    {"role": "user", "content": f"请审查以下{language}代码:\n\n文件: {file_name}\n```{language}\n{code}\n```"},
                ],
                temperature=0.2,
                max_tokens=3000,
                response_format={"type": "json_object"},
            )
            import json
            result = json.loads(response.choices[0].message.content)
        else:
            result = self._mock_review(code)

        # 构建审查报告
        comments = []
        for c in result.get("comments", []):
            comments.append(ReviewComment(
                severity=ReviewSeverity(c.get("severity", "suggestion")),
                line_number=c.get("line", -1),
                category=c.get("category", ""),
                message=c.get("message", ""),
                suggestion=c.get("suggestion", ""),
            ))

        return ReviewReport(
            file_name=file_name,
            language=language,
            comments=comments,
            summary=result.get("summary", ""),
            overall_rating=result.get("overall_rating", "approve"),
            score=result.get("score", 7.0),
            review_time=datetime.now().isoformat(),
        )

    def _mock_review(self, code: str) -> Dict:
        """模拟审查结果"""
        comments = []

        # 检查简单的问题
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            if "except:" in line and "except Exception" not in line:
                comments.append({
                    "severity": "warning", "line": i, "category": "correctness",
                    "message": "裸except会捕获所有异常", "suggestion": "使用 except Exception as e:"
                })
            if len(line) > 120:
                comments.append({
                    "severity": "suggestion", "line": i, "category": "style",
                    "message": f"行长度 {len(line)} 字符，超过120字符限制",
                    "suggestion": "拆分为多行"
                })
            if re.search(r'password\s*=\s*["\']', line):
                comments.append({
                    "severity": "blocker", "line": i, "category": "security",
                    "message": "硬编码密码", "suggestion": "使用环境变量或密钥管理服务"
                })

        score = max(0, 10 - len([c for c in comments if c["severity"] == "blocker"]) * 3
                     - len([c for c in comments if c["severity"] == "warning"]) * 1.5)

        return {
            "score": round(score, 1),
            "overall_rating": "approve" if score >= 7 else "changes_requested",
            "summary": f"发现 {len(comments)} 个问题，整体评分 {score}/10",
            "comments": comments,
        }

    def format_report(self, report: ReviewReport) -> str:
        """格式化审查报告"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"  代码审查报告 - {report.file_name}")
        lines.append("=" * 60)
        lines.append(f"  评分: {report.score}/10")
        lines.append(f"  结论: {'通过' if report.overall_rating == 'approve' else '需要修改'}")
        lines.append(f"  审查时间: {report.review_time}")
        lines.append("-" * 60)
        lines.append(f"  摘要: {report.summary}")
        lines.append("-" * 60)

        # 按严重度分组
        for severity in ReviewSeverity:
            group = [c for c in report.comments if c.severity == severity]
            if group:
                label_map = {
                    "blocker": "必须修复",
                    "warning": "建议修改",
                    "suggestion": "可选优化",
                    "praise": "优秀之处",
                }
                lines.append(f"\n  [{label_map.get(severity.value, severity.value)}] ({len(group)}个)")
                for c in group:
                    loc = f"第{c.line_number}行" if c.line_number > 0 else ""
                    lines.append(f"    {loc} [{c.category}] {c.message}")
                    if c.suggestion:
                        lines.append(f"      -> {c.suggestion}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    print("=== AI代码审查 测试 ===\n")

    reviewer = CodeReviewer()

    sample_code = """
import os

password = "secret123"

def get_user_data(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    try:
        result = db.execute(query)
        return result
    except:
        return None

def process(data):
    temp = []
    for i in range(len(data)):
        if data[i] > 0:
            temp.append(data[i] * 2 + 3.14159)
    return temp
"""

    report = reviewer.review(sample_code.strip(), "user_service.py")
    print(reviewer.format_report(report))
```

---

## FastAPI后端服务

### API设计

```
+=========================================================================+
|                      AI编程助手 API 设计                                  |
+=========================================================================+
|                                                                         |
|  代码生成                                                               |
|  POST /api/v1/generate           根据描述生成代码                       |
|                                                                         |
|  Bug修复                                                                |
|  POST /api/v1/fix                分析并修复代码Bug                      |
|                                                                         |
|  代码重构                                                               |
|  POST /api/v1/refactor           代码重构建议                           |
|                                                                         |
|  代码审查                                                               |
|  POST /api/v1/review             AI代码审查                             |
|                                                                         |
|  工具接口                                                               |
|  POST /api/v1/explain            代码解释                               |
|  POST /api/v1/test/generate      生成单元测试                           |
|  POST /api/v1/doc/generate       生成文档注释                           |
|                                                                         |
|  GET  /api/v1/health             健康检查                               |
|  GET  /api/v1/languages          支持的语言列表                         |
|                                                                         |
+=========================================================================+
```

### FastAPI完整代码

```python
"""
AI编程助手 - FastAPI 后端服务
"""
import os
import time
from typing import Optional, List, Dict
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# 请求/响应模型
# ============================================================
class GenerateRequest(BaseModel):
    description: str = Field(..., min_length=5, max_length=2000, description="功能描述")
    language: str = Field(default="python", description="编程语言")
    context: str = Field(default="", description="项目上下文")
    style_guide: str = Field(default="", description="编码规范")

class GenerateResponse(BaseModel):
    code: str
    language: str
    explanation: str = ""
    dependencies: List[str] = []
    warnings: List[str] = []
    is_valid: bool = True
    processing_time_ms: float = 0.0

class FixRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=10000)
    error_message: str = Field(default="", description="错误信息")
    language: str = Field(default="python")

class FixResponse(BaseModel):
    fixed_code: str
    bugs_found: List[Dict] = []
    diff: str = ""
    explanation: str = ""
    is_fixed: bool = True

class RefactorRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=10000)
    focus: str = Field(default="", description="重点: readability/performance/pattern")
    language: str = Field(default="python")

class RefactorResponse(BaseModel):
    refactored_code: str
    quality_score: float = 0.0
    smells_count: int = 0
    improvements: List[str] = []
    explanation: str = ""

class ReviewRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=10000)
    file_name: str = Field(default="code.py")
    language: str = Field(default="python")

class ReviewResponse(BaseModel):
    score: float = 0.0
    overall_rating: str = "approve"
    summary: str = ""
    comments: List[Dict] = []

class ExplainRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=10000)
    language: str = Field(default="python")
    detail_level: str = Field(default="medium", description="简略/中等/详细")


# ============================================================
# 服务层(整合各引擎)
# ============================================================
class CodeAssistantService:
    """编程助手服务"""

    def __init__(self):
        self.request_count = 0
        print("[CodeAssistantService] 初始化完成")

    def generate_code(self, req: GenerateRequest) -> GenerateResponse:
        start = time.time()
        # 实际项目中调用 CodeGenerator
        code = self._mock_generate(req.description, req.language)
        elapsed = (time.time() - start) * 1000
        self.request_count += 1
        return GenerateResponse(
            code=code,
            language=req.language,
            explanation=f"根据描述 '{req.description[:50]}...' 生成的代码",
            dependencies=[],
            is_valid=True,
            processing_time_ms=round(elapsed, 1),
        )

    def fix_code(self, req: FixRequest) -> FixResponse:
        self.request_count += 1
        # 实际项目中调用 BugFixer
        return FixResponse(
            fixed_code=req.code,
            bugs_found=[],
            explanation="未发现明显Bug(模拟模式)",
            is_fixed=False,
        )

    def refactor_code(self, req: RefactorRequest) -> RefactorResponse:
        self.request_count += 1
        return RefactorResponse(
            refactored_code=req.code,
            quality_score=75.0,
            smells_count=0,
            improvements=["代码质量良好"],
            explanation="模拟模式 - 请配置API密钥以获取AI重构建议",
        )

    def review_code(self, req: ReviewRequest) -> ReviewResponse:
        self.request_count += 1
        return ReviewResponse(
            score=8.0,
            overall_rating="approve",
            summary="代码整体质量良好(模拟模式)",
            comments=[],
        )

    def _mock_generate(self, desc: str, lang: str) -> str:
        return f"# {desc}\n# 语言: {lang}\n# 请配置 OPENAI_API_KEY 以获取AI生成的代码\npass"


# ============================================================
# FastAPI 应用
# ============================================================
service = CodeAssistantService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[启动] AI编程助手后端服务")
    yield
    print("[关闭] 清理资源")


app = FastAPI(
    title="AI编程助手 API",
    description="提供代码生成、Bug修复、重构、审查等AI编程辅助功能",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/v1/health")
async def health():
    return {"status": "healthy", "requests": service.request_count, "timestamp": datetime.now().isoformat()}

@app.get("/api/v1/languages")
async def supported_languages():
    return {"languages": ["python", "javascript", "typescript", "java", "go", "rust", "c", "cpp", "sql"]}

@app.post("/api/v1/generate", response_model=GenerateResponse)
async def generate_code(req: GenerateRequest):
    """根据自然语言描述生成代码"""
    try:
        return service.generate_code(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/fix", response_model=FixResponse)
async def fix_code(req: FixRequest):
    """分析并修复代码Bug"""
    try:
        return service.fix_code(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/refactor", response_model=RefactorResponse)
async def refactor_code(req: RefactorRequest):
    """代码重构建议"""
    try:
        return service.refactor_code(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/review", response_model=ReviewResponse)
async def review_code(req: ReviewRequest):
    """AI代码审查"""
    try:
        return service.review_code(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/explain")
async def explain_code(req: ExplainRequest):
    """代码解释"""
    return {
        "explanation": f"这段{req.language}代码的功能是...(模拟模式)",
        "language": req.language,
        "line_count": len(req.code.split("\n")),
    }


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  AI编程助手")
    print("  API文档: http://localhost:8002/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8002, reload=True)
```

---

## VSCode插件开发

### 插件架构

```
+=========================================================================+
|                     VSCode 插件架构                                       |
+=========================================================================+
|                                                                         |
|  VSCode 编辑器                                                         |
|  +---------------------------------------------------------------+     |
|  |                                                                |     |
|  |  +--------------------+  +---------------------------------+  |     |
|  |  |  编辑器区域        |  |  侧边栏 - AI助手面板            |  |     |
|  |  |                    |  |                                  |  |     |
|  |  |  function foo():   |  |  [生成代码]                     |  |     |
|  |  |    // 选中的代码   |  |  描述: ___________              |  |     |
|  |  |    ...             |  |  语言: Python  v                 |  |     |
|  |  |                    |  |  [生成]                          |  |     |
|  |  |  右键菜单:         |  |                                  |  |     |
|  |  |  > AI: 修复Bug    |  |  [审查报告]                     |  |     |
|  |  |  > AI: 重构       |  |  评分: 8.5/10                   |  |     |
|  |  |  > AI: 生成测试   |  |  问题: 3个                       |  |     |
|  |  |  > AI: 解释代码   |  |                                  |  |     |
|  |  +--------------------+  +---------------------------------+  |     |
|  |                                                                |     |
|  |  状态栏: AI助手 | 已连接 | GPT-4o-mini                      |     |
|  +---------------------------------------------------------------+     |
|                          |                                              |
|                          | HTTP API                                     |
|                          v                                              |
|                 +--------+--------+                                     |
|                 | FastAPI 后端     |                                     |
|                 | localhost:8002   |                                     |
|                 +-----------------+                                     |
|                                                                         |
+=========================================================================+
```

### 插件核心代码

```python
"""
VSCode 插件配置和命令说明

注: VSCode插件使用TypeScript/JavaScript开发
以下展示 package.json 配置和核心逻辑的Python模拟版
"""

# ============================================================
# package.json 配置 (VSCode插件清单)
# ============================================================
PACKAGE_JSON = {
    "name": "ai-code-assistant",
    "displayName": "AI编程助手",
    "description": "AI驱动的代码生成、修复、重构、审查工具",
    "version": "1.0.0",
    "engines": {"vscode": "^1.85.0"},
    "categories": ["Programming Languages", "Linters"],
    "activationEvents": ["onStartupFinished"],
    "main": "./out/extension.js",
    "contributes": {
        "commands": [
            {"command": "ai-assistant.generateCode", "title": "AI: 生成代码"},
            {"command": "ai-assistant.fixBug", "title": "AI: 修复Bug"},
            {"command": "ai-assistant.refactor", "title": "AI: 重构代码"},
            {"command": "ai-assistant.review", "title": "AI: 代码审查"},
            {"command": "ai-assistant.explain", "title": "AI: 解释代码"},
            {"command": "ai-assistant.generateTest", "title": "AI: 生成测试"},
        ],
        "menus": {
            "editor/context": [
                {"command": "ai-assistant.fixBug", "group": "ai-assistant"},
                {"command": "ai-assistant.refactor", "group": "ai-assistant"},
                {"command": "ai-assistant.explain", "group": "ai-assistant"},
            ]
        },
        "configuration": {
            "title": "AI编程助手",
            "properties": {
                "ai-assistant.apiUrl": {
                    "type": "string",
                    "default": "http://localhost:8002",
                    "description": "后端API地址",
                },
                "ai-assistant.model": {
                    "type": "string",
                    "default": "gpt-4o-mini",
                    "description": "使用的AI模型",
                },
            },
        },
    },
}


# ============================================================
# extension.ts 核心逻辑 (Python模拟版)
# ============================================================

class VSCodeExtensionSimulator:
    """
    模拟VSCode插件的核心逻辑

    实际开发中使用TypeScript, 这里用Python展示逻辑流程
    """

    def __init__(self, api_url: str = "http://localhost:8002"):
        self.api_url = api_url

    def on_command_generate(self, description: str, language: str) -> str:
        """
        对应 ai-assistant.generateCode 命令

        流程:
        1. 弹出输入框让用户输入需求描述
        2. 调用后端 /api/v1/generate
        3. 将生成的代码插入编辑器
        """
        import json
        try:
            import httpx
            response = httpx.post(f"{self.api_url}/api/v1/generate", json={
                "description": description,
                "language": language,
            }, timeout=30)
            data = response.json()
            return data.get("code", "")
        except Exception as e:
            return f"# 生成失败: {e}"

    def on_command_fix(self, selected_code: str, language: str) -> str:
        """
        对应 ai-assistant.fixBug 命令

        流程:
        1. 获取编辑器中选中的代码
        2. 调用后端 /api/v1/fix
        3. 显示diff对比，让用户确认是否应用
        """
        import json
        try:
            import httpx
            response = httpx.post(f"{self.api_url}/api/v1/fix", json={
                "code": selected_code,
                "language": language,
            }, timeout=30)
            data = response.json()
            return data.get("fixed_code", selected_code)
        except Exception as e:
            return selected_code

    def on_command_review(self, file_content: str, file_name: str) -> dict:
        """
        对应 ai-assistant.review 命令

        流程:
        1. 获取当前打开文件的内容
        2. 调用后端 /api/v1/review
        3. 在Problems面板和侧边栏显示审查结果
        """
        try:
            import httpx
            response = httpx.post(f"{self.api_url}/api/v1/review", json={
                "code": file_content,
                "file_name": file_name,
            }, timeout=30)
            return response.json()
        except Exception:
            return {"score": 0, "comments": [], "summary": "审查服务不可用"}


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    import json
    print("=== VSCode插件配置 ===")
    print(f"插件名称: {PACKAGE_JSON['displayName']}")
    print(f"命令数量: {len(PACKAGE_JSON['contributes']['commands'])}")
    print("\n注册的命令:")
    for cmd in PACKAGE_JSON["contributes"]["commands"]:
        print(f"  {cmd['command']} -> {cmd['title']}")

    print("\n右键菜单:")
    for item in PACKAGE_JSON["contributes"]["menus"]["editor/context"]:
        print(f"  {item['command']}")
```

---

## 测试与部署

### 测试策略

```
+=========================================================================+
|                        测试策略                                          |
+=========================================================================+
|                                                                         |
|  单元测试 (Unit Tests)                                                  |
|  +--------------------------------------------------------------+      |
|  |  - 代码生成器: 输出格式、语法验证、安全检查                   |      |
|  |  - Bug修复器: 已知Bug的检测率、修复正确性                     |      |
|  |  - 重构引擎: 质量评分计算、坏味道检测                         |      |
|  |  - 代码审查: 安全问题检测率                                   |      |
|  +--------------------------------------------------------------+      |
|                                                                         |
|  集成测试 (Integration Tests)                                           |
|  +--------------------------------------------------------------+      |
|  |  - API端点: 请求/响应格式、状态码                             |      |
|  |  - 端到端流程: 生成 -> 审查 -> 修复 -> 通过                   |      |
|  +--------------------------------------------------------------+      |
|                                                                         |
|  评估测试 (Evaluation Tests)                                            |
|  +--------------------------------------------------------------+      |
|  |  - 生成代码的可执行率                                         |      |
|  |  - Bug修复的成功率                                            |      |
|  |  - 审查发现问题的精确率/召回率                                |      |
|  +--------------------------------------------------------------+      |
|                                                                         |
+=========================================================================+
```

### 测试代码

```python
"""
AI编程助手 - 单元测试
"""
import ast
import unittest
from typing import List


# ============================================================
# 被测试的工具函数(从前面模块引入)
# ============================================================

class PythonValidator:
    """Python代码验证器(简化版，用于测试)"""

    @staticmethod
    def check_syntax(code: str) -> tuple:
        try:
            ast.parse(code)
            return True, "语法正确"
        except SyntaxError as e:
            return False, f"语法错误: 第{e.lineno}行 - {e.msg}"

    @staticmethod
    def check_security(code: str) -> List[str]:
        warnings = []
        patterns = [
            ("eval(", "eval()存在安全风险"),
            ("exec(", "exec()存在安全风险"),
        ]
        for pattern, warning in patterns:
            if pattern in code:
                warnings.append(warning)
        return warnings

    @staticmethod
    def extract_imports(code: str) -> List[str]:
        imports = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split(".")[0])
        except SyntaxError:
            pass
        stdlib = {"os", "sys", "re", "json", "math", "typing", "datetime", "collections"}
        return sorted(imports - stdlib)


# ============================================================
# 测试用例
# ============================================================

class TestPythonValidator(unittest.TestCase):
    """Python验证器测试"""

    def test_valid_syntax(self):
        """测试有效的Python语法"""
        code = "def hello():\n    print('Hello, World!')"
        is_valid, msg = PythonValidator.check_syntax(code)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "语法正确")

    def test_invalid_syntax(self):
        """测试无效的Python语法"""
        code = "def hello(\n    print('missing paren'"
        is_valid, msg = PythonValidator.check_syntax(code)
        self.assertFalse(is_valid)
        self.assertIn("语法错误", msg)

    def test_security_eval(self):
        """测试eval安全检测"""
        code = "result = eval(user_input)"
        warnings = PythonValidator.check_security(code)
        self.assertTrue(len(warnings) > 0)
        self.assertIn("eval", warnings[0])

    def test_security_clean(self):
        """测试安全代码不应产生警告"""
        code = "result = int(user_input)"
        warnings = PythonValidator.check_security(code)
        self.assertEqual(len(warnings), 0)

    def test_extract_imports(self):
        """测试依赖提取"""
        code = """
import requests
import json
from fastapi import FastAPI
from pydantic import BaseModel
"""
        deps = PythonValidator.extract_imports(code)
        self.assertIn("requests", deps)
        self.assertIn("fastapi", deps)
        self.assertIn("pydantic", deps)
        # json是标准库，不应出现
        self.assertNotIn("json", deps)

    def test_empty_code(self):
        """测试空代码"""
        is_valid, msg = PythonValidator.check_syntax("")
        self.assertTrue(is_valid)

    def test_extract_imports_syntax_error(self):
        """语法错误的代码也应能提取部分import"""
        code = "import requests\ndef broken("
        deps = PythonValidator.extract_imports(code)
        # 语法错误时回退为空
        self.assertIsInstance(deps, list)


class TestCodeQuality(unittest.TestCase):
    """代码质量测试"""

    def test_function_count(self):
        """测试函数计数"""
        code = """
def func1():
    pass

def func2():
    pass

class MyClass:
    def method1(self):
        pass
"""
        tree = ast.parse(code)
        func_count = sum(1 for node in ast.walk(tree)
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
        self.assertEqual(func_count, 3)

    def test_class_count(self):
        """测试类计数"""
        code = """
class A:
    pass

class B(A):
    pass
"""
        tree = ast.parse(code)
        class_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        self.assertEqual(class_count, 2)


# ============================================================
# 运行测试
# ============================================================
if __name__ == "__main__":
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestPythonValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestCodeQuality))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 输出结果
    print(f"\n运行: {result.testsRun} 个测试")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
```

---

## 总结

本教程完整实现了一个AI编程助手系统，涵盖以下核心模块:

1. **代码生成引擎**: 多语言支持、Prompt工程、语法验证、安全检查、依赖提取
2. **Bug检测与修复**: AST静态分析、LLM智能检测、自动修复、Diff生成
3. **代码重构建议**: 质量指标计算、坏味道识别、LLM重构、评分体系
4. **代码审查系统**: 多维度审查(正确性/安全/性能/规范)、审查报告
5. **FastAPI后端**: 完整REST API(生成/修复/重构/审查/解释)
6. **VSCode插件**: 插件架构、命令注册、右键菜单、侧边栏面板
7. **测试**: 单元测试、验证器测试、质量测试

## 最佳实践

1. **Prompt工程**: 为每种语言定制System Prompt，包含编码规范和示例代码风格
2. **安全防护**: 对生成的代码进行安全扫描，阻止危险API调用
3. **语法验证**: 使用AST解析验证生成代码的语法正确性
4. **增量改进**: 结合静态分析(确定性)和LLM(灵活性)，互相补充
5. **用户体验**: 流式输出、进度提示、diff预览、一键应用

## 参考资源

- [OpenAI API 文档](https://platform.openai.com/docs/)
- [VSCode Extension API](https://code.visualstudio.com/api)
- [Python AST 模块](https://docs.python.org/3/library/ast.html)
- [FastAPI 文档](https://fastapi.tiangolo.com/)

---

**创建时间**: 2024-01-01
**最后更新**: 2024-01-01
