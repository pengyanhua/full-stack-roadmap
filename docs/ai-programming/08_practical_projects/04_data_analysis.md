# AI数据分析助手

## 目录
1. [项目概述](#项目概述)
2. [代码生成](#代码生成)
3. [可视化建议](#可视化建议)
4. [自动化报告](#自动化报告)
5. [Notebook集成](#notebook集成)
6. [完整项目](#完整项目)

---

## 项目概述

### 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                     AI 数据分析助手 架构                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  用户层                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Web界面  │  │ Notebook │  │ REST API │  │  CLI工具  │        │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘        │
│        │             │             │             │                │
│        └─────────────┴──────┬──────┴─────────────┘               │
│                             │                                    │
│  服务层                     ▼                                    │
│  ┌──────────────────────────────────────────────────┐            │
│  │              FastAPI 网关服务                      │            │
│  │  ┌────────┐ ┌─────────┐ ┌────────┐ ┌─────────┐  │           │
│  │  │会话管理│ │文件上传 │ │权限控制│ │速率限制 │  │            │
│  │  └────────┘ └─────────┘ └────────┘ └─────────┘  │           │
│  └──────────────────────┬───────────────────────────┘            │
│                         │                                        │
│  引擎层                 ▼                                        │
│  ┌────────────┐ ┌─────────────┐ ┌──────────────┐               │
│  │NL2Code引擎│ │可视化推荐   │ │报告生成引擎  │                │
│  │            │ │引擎         │ │              │                │
│  │自然语言    │ │数据特征分析 │ │模板渲染      │                │
│  │  → Pandas  │ │图表类型推荐 │ │HTML/PDF导出  │                │
│  │  → SQL     │ │参数自动配置 │ │定时任务      │                │
│  └─────┬──────┘ └──────┬──────┘ └──────┬───────┘               │
│        │               │               │                        │
│        └───────────────┼───────────────┘                        │
│                        ▼                                        │
│  数据层  ┌─────────────────────────────┐                        │
│          │  数据连接器                  │                        │
│          │  CSV / Excel / SQL / API     │                        │
│          └─────────────────────────────┘                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 核心工作流程

```
┌─────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ 上传数据 │────▶│ 数据概览  │────▶│ 自然语言  │────▶│ 代码生成  │
│ CSV/Excel│     │ 统计摘要  │     │ 提问分析  │     │ 执行结果  │
└─────────┘     └──────────┘     └──────────┘     └─────┬────┘
                                                        │
                     ┌──────────┐     ┌──────────┐      │
                     │ 导出报告  │◀────│ 可视化   │◀─────┘
                     │ HTML/PDF  │     │ 图表展示  │
                     └──────────┘     └──────────┘
```

### 详细说明

AI数据分析助手是一个将**自然语言**转换为**数据分析代码**的智能系统。用户只需用中文描述分析需求，系统即可自动生成Pandas代码、推荐合适的可视化图表、并生成结构化的分析报告。

**核心能力:**
- **NL2Code**: 自然语言转Pandas/SQL代码，支持复杂的数据处理操作
- **智能可视化**: 根据数据特征自动推荐最佳图表类型和配置
- **自动化报告**: 一键生成包含统计摘要、图表和洞察的完整分析报告
- **Notebook集成**: 无缝嵌入Jupyter Notebook工作流

**技术栈:**
- LLM: OpenAI GPT-4o-mini（代码生成与分析推理）
- 数据处理: Pandas, NumPy
- 可视化: Matplotlib, Plotly, Seaborn
- 后端: FastAPI + Uvicorn
- 报告: Jinja2模板 + WeasyPrint(PDF)
- Notebook: Jupyter Kernel Gateway

### 项目配置与数据模型

```python
"""
AI数据分析助手 - 配置与数据模型
"""
import os
import json
import hashlib
from enum import Enum
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

# ============================================================
# 项目配置
# ============================================================

@dataclass
class AnalysisConfig:
    """分析助手配置"""
    # OpenAI配置
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = "gpt-4o-mini"
    max_tokens: int = 4096
    temperature: float = 0.1  # 代码生成用低温度

    # 数据配置
    max_file_size_mb: int = 100
    max_rows_preview: int = 1000
    supported_formats: tuple = (".csv", ".xlsx", ".xls", ".json", ".parquet")

    # 安全配置
    sandbox_enabled: bool = True
    allowed_imports: tuple = ("pandas", "numpy", "matplotlib", "seaborn", "plotly")
    blocked_functions: tuple = ("exec", "eval", "os.system", "subprocess", "__import__")

    # 报告配置
    report_output_dir: str = "./reports"
    chart_output_dir: str = "./charts"
    report_template_dir: str = "./templates"


class ChartType(str, Enum):
    """图表类型枚举"""
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    PIE = "pie"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX = "box"
    AREA = "area"
    VIOLIN = "violin"
    TREEMAP = "treemap"


class ColumnType(str, Enum):
    """列数据类型"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"


@dataclass
class ColumnProfile:
    """列数据概况"""
    name: str
    dtype: str
    col_type: ColumnType
    non_null_count: int
    null_count: int
    unique_count: int
    sample_values: list = field(default_factory=list)
    # 数值列额外统计
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    median: Optional[float] = None
    # 分类列额外统计
    top_values: Optional[dict] = None


@dataclass
class DatasetProfile:
    """数据集概况"""
    filename: str
    row_count: int
    col_count: int
    columns: list  # List[ColumnProfile]
    memory_usage_mb: float
    has_missing: bool
    missing_summary: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_prompt_context(self) -> str:
        """转换为LLM提示上下文"""
        lines = [
            f"数据集: {self.filename}",
            f"行数: {self.row_count}, 列数: {self.col_count}",
            f"内存占用: {self.memory_usage_mb:.2f} MB",
            "",
            "列信息:"
        ]
        for col in self.columns:
            line = f"  - {col.name} ({col.dtype}, {col.col_type.value})"
            if col.col_type == ColumnType.NUMERIC and col.mean is not None:
                line += f" [均值={col.mean:.2f}, 标准差={col.std:.2f}]"
            if col.col_type == ColumnType.CATEGORICAL and col.top_values:
                top3 = list(col.top_values.items())[:3]
                line += f" [Top: {', '.join(f'{k}({v})' for k,v in top3)}]"
            if col.null_count > 0:
                line += f" [缺失: {col.null_count}]"
            lines.append(line)

        if self.has_missing:
            lines.append(f"\n缺失值: {json.dumps(self.missing_summary, ensure_ascii=False)}")
        return "\n".join(lines)


@dataclass
class AnalysisResult:
    """分析结果"""
    query: str
    generated_code: str
    execution_output: Any = None
    chart_paths: list = field(default_factory=list)
    error: Optional[str] = None
    execution_time: float = 0.0

    @property
    def success(self) -> bool:
        return self.error is None
```

### 数据加载与概况分析

```python
"""
数据加载器 - 支持多格式文件加载与自动概况分析
"""
import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    """多格式数据加载器"""

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def load(self, file_path: str, **kwargs) -> pd.DataFrame:
        """加载数据文件"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = path.suffix.lower()
        if suffix not in self.config.supported_formats:
            raise ValueError(f"不支持的格式: {suffix}")

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.config.max_file_size_mb:
            raise ValueError(f"文件过大: {size_mb:.1f}MB > {self.config.max_file_size_mb}MB")

        loaders = {
            ".csv": self._load_csv,
            ".xlsx": self._load_excel,
            ".xls": self._load_excel,
            ".json": self._load_json,
            ".parquet": self._load_parquet,
        }
        return loaders[suffix](file_path, **kwargs)

    def _load_csv(self, path: str, **kwargs) -> pd.DataFrame:
        """智能CSV加载 - 自动检测编码和分隔符"""
        encodings = ["utf-8", "gbk", "gb2312", "latin1"]
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc, **kwargs)
                print(f"  CSV加载成功, 编码: {enc}")
                return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        raise ValueError("无法识别CSV编码")

    def _load_excel(self, path: str, **kwargs) -> pd.DataFrame:
        return pd.read_excel(path, **kwargs)

    def _load_json(self, path: str, **kwargs) -> pd.DataFrame:
        return pd.read_json(path, **kwargs)

    def _load_parquet(self, path: str, **kwargs) -> pd.DataFrame:
        return pd.read_parquet(path, **kwargs)


class DataProfiler:
    """数据概况分析器"""

    def profile(self, df: pd.DataFrame, filename: str = "unknown") -> DatasetProfile:
        """生成数据集概况"""
        columns = []
        for col_name in df.columns:
            series = df[col_name]
            col_type = self._infer_column_type(series)

            cp = ColumnProfile(
                name=col_name,
                dtype=str(series.dtype),
                col_type=col_type,
                non_null_count=int(series.notna().sum()),
                null_count=int(series.isna().sum()),
                unique_count=int(series.nunique()),
                sample_values=series.dropna().head(5).tolist(),
            )

            if col_type == ColumnType.NUMERIC:
                cp.mean = float(series.mean()) if not series.empty else None
                cp.std = float(series.std()) if not series.empty else None
                cp.min_val = float(series.min()) if not series.empty else None
                cp.max_val = float(series.max()) if not series.empty else None
                cp.median = float(series.median()) if not series.empty else None

            if col_type == ColumnType.CATEGORICAL:
                cp.top_values = series.value_counts().head(10).to_dict()

            columns.append(cp)

        missing = {col: int(df[col].isna().sum())
                   for col in df.columns if df[col].isna().any()}

        return DatasetProfile(
            filename=filename,
            row_count=len(df),
            col_count=len(df.columns),
            columns=columns,
            memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
            has_missing=len(missing) > 0,
            missing_summary=missing,
        )

    def _infer_column_type(self, series: pd.Series) -> ColumnType:
        """推断列数据类型"""
        dtype = series.dtype

        if pd.api.types.is_bool_dtype(dtype):
            return ColumnType.BOOLEAN
        if pd.api.types.is_numeric_dtype(dtype):
            return ColumnType.NUMERIC
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return ColumnType.DATETIME

        # 尝试解析为日期
        if dtype == object:
            sample = series.dropna().head(20)
            try:
                pd.to_datetime(sample)
                return ColumnType.DATETIME
            except (ValueError, TypeError):
                pass

            # 判断分类 vs 文本
            nunique_ratio = series.nunique() / max(len(series), 1)
            if nunique_ratio < 0.3 or series.nunique() < 50:
                return ColumnType.CATEGORICAL
            avg_len = sample.astype(str).str.len().mean()
            if avg_len > 50:
                return ColumnType.TEXT

        return ColumnType.CATEGORICAL


# ============================================================
# 使用示例
# ============================================================

def demo_data_loading():
    """演示数据加载与概况分析"""
    # 创建示例数据
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "日期": pd.date_range("2024-01-01", periods=n, freq="D"),
        "产品": np.random.choice(["手机", "电脑", "平板", "耳机"], n),
        "地区": np.random.choice(["华东", "华南", "华北", "西南"], n),
        "销量": np.random.randint(10, 500, n),
        "单价": np.random.uniform(100, 5000, n).round(2),
        "满意度": np.random.uniform(3.0, 5.0, n).round(1),
    })
    df.loc[np.random.choice(n, 10), "满意度"] = np.nan  # 添加缺失值

    profiler = DataProfiler()
    profile = profiler.profile(df, "sales_data.csv")
    print(profile.to_prompt_context())
    return df, profile


if __name__ == "__main__":
    df, profile = demo_data_loading()
    print(f"\n数据形状: {df.shape}")
```

---

## 代码生成

### NL2Code引擎架构

```
┌──────────────────────────────────────────────────────────────┐
│                  NL2Code 引擎工作流                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ 用户问题  │───▶│ 意图识别      │───▶│ Prompt组装   │       │
│  │ "各产品   │    │              │    │              │       │
│  │  的平均   │    │ ·聚合统计    │    │ ·系统提示    │       │
│  │  销量"    │    │ ·筛选过滤    │    │ ·数据上下文  │       │
│  └──────────┘    │ ·排序        │    │ ·用户问题    │       │
│                  │ ·关联        │    │ ·输出格式    │       │
│                  └──────────────┘    └──────┬───────┘       │
│                                             │                │
│                                             ▼                │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ 返回结果  │◀───│ 安全沙箱执行  │◀───│ LLM生成代码  │       │
│  │ DataFrame │    │              │    │              │       │
│  │ + 图表    │    │ ·import检查  │    │ ·Pandas代码  │       │
│  └──────────┘    │ ·超时控制    │    │ ·注释说明    │       │
│                  │ ·内存限制    │    └──────────────┘       │
│                  └──────────────┘                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 详细说明

NL2Code引擎是数据分析助手的核心模块。它将用户的自然语言问题（如"按地区统计各产品的平均销量"）转换为可执行的Pandas代码。

**关键设计:**
1. **上下文感知Prompt**: 将数据集的列名、类型、统计摘要注入到提示中，让LLM精确生成代码
2. **安全沙箱**: 限制可用的import和函数调用，防止恶意代码执行
3. **代码提取与清洗**: 从LLM输出中提取代码块，修正常见语法问题
4. **结果捕获**: 自动捕获DataFrame输出、print内容和生成的图表文件

### 代码示例

```python
"""
NL2Code引擎 - 自然语言转Pandas代码
"""
import re
import ast
import time
import traceback
from io import StringIO
from contextlib import redirect_stdout

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# ============================================================
# 安全代码执行沙箱
# ============================================================

class CodeSandbox:
    """安全代码执行沙箱"""

    BLOCKED_PATTERNS = [
        r'\bexec\s*\(', r'\beval\s*\(',
        r'\bos\.\w+', r'\bsubprocess\b',
        r'\b__import__\b', r'\bopen\s*\(',
        r'\bglobals\s*\(', r'\blocals\s*\(',
        r'\bcompile\s*\(', r'\bgetattr\s*\(',
        r'\bdelattr\s*\(', r'\bsetattr\s*\(',
    ]

    ALLOWED_MODULES = {
        "pandas", "numpy", "matplotlib", "matplotlib.pyplot",
        "seaborn", "plotly", "plotly.express", "plotly.graph_objects",
        "datetime", "math", "statistics", "collections",
    }

    def validate_code(self, code: str) -> tuple:
        """验证代码安全性，返回 (is_safe, message)"""
        # 检查危险模式
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, code):
                return False, f"检测到危险调用: {pattern}"

        # 检查import语句
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"语法错误: {e}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in self.ALLOWED_MODULES:
                        return False, f"禁止导入: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] not in self.ALLOWED_MODULES:
                    return False, f"禁止导入: {node.module}"

        return True, "通过安全检查"

    def execute(self, code: str, local_vars: dict,
                timeout: float = 30.0) -> dict:
        """在受限环境中执行代码"""
        is_safe, msg = self.validate_code(code)
        if not is_safe:
            return {"success": False, "error": f"安全检查失败: {msg}"}

        import pandas as pd
        import numpy as np

        exec_globals = {
            "__builtins__": {
                "print": print, "len": len, "range": range,
                "int": int, "float": float, "str": str,
                "list": list, "dict": dict, "tuple": tuple,
                "sorted": sorted, "enumerate": enumerate,
                "zip": zip, "map": map, "filter": filter,
                "min": min, "max": max, "sum": sum,
                "abs": abs, "round": round, "isinstance": isinstance,
                "True": True, "False": False, "None": None,
            },
            "pd": pd, "np": np,
        }
        exec_globals.update(local_vars)

        stdout_capture = StringIO()
        start = time.time()

        try:
            with redirect_stdout(stdout_capture):
                exec(code, exec_globals)

            elapsed = time.time() - start
            # 收集结果
            result_df = exec_globals.get("result", None)
            output = stdout_capture.getvalue()

            return {
                "success": True,
                "output": output,
                "result": result_df,
                "time": elapsed,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "traceback": traceback.format_exc(),
            }


# ============================================================
# NL2Code 引擎
# ============================================================

class NL2CodeEngine:
    """自然语言转Pandas代码引擎"""

    SYSTEM_PROMPT = """你是一个专业的Python数据分析代码生成器。

规则:
1. 只使用pandas(pd)和numpy(np)，数据变量名为 df
2. 将最终结果赋值给变量 result
3. 如果需要打印信息用 print()
4. 代码简洁高效，添加中文注释
5. 只输出Python代码块，不要其他解释文字
6. 处理可能的缺失值(NaN)
7. 日期列用 pd.to_datetime() 转换
8. 中文列名直接使用，不要重命名"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.sandbox = CodeSandbox()
        if HAS_OPENAI and config.openai_api_key:
            self.client = OpenAI(api_key=config.openai_api_key)
        else:
            self.client = None

    def generate_code(self, question: str,
                      dataset_profile: DatasetProfile) -> str:
        """根据自然语言问题生成Pandas代码"""
        context = dataset_profile.to_prompt_context()
        user_prompt = f"""数据集信息:
{context}

用户问题: {question}

请生成完整的Pandas分析代码，将结果赋值给 result 变量。"""

        if self.client:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            raw = response.choices[0].message.content
        else:
            raw = self._mock_generate(question, dataset_profile)

        return self._extract_code(raw)

    def execute_analysis(self, question: str,
                         df, profile: DatasetProfile) -> AnalysisResult:
        """完整分析流程: 生成代码 -> 执行 -> 返回结果"""
        start = time.time()
        try:
            code = self.generate_code(question, profile)
            exec_result = self.sandbox.execute(code, {"df": df})
            elapsed = time.time() - start

            if exec_result["success"]:
                return AnalysisResult(
                    query=question,
                    generated_code=code,
                    execution_output=exec_result.get("result",
                                                     exec_result.get("output")),
                    execution_time=elapsed,
                )
            else:
                return AnalysisResult(
                    query=question,
                    generated_code=code,
                    error=exec_result["error"],
                    execution_time=elapsed,
                )
        except Exception as e:
            return AnalysisResult(
                query=question,
                generated_code="",
                error=str(e),
                execution_time=time.time() - start,
            )

    def _extract_code(self, raw_text: str) -> str:
        """从LLM输出中提取Python代码块"""
        pattern = r'```python\s*\n(.*?)```'
        matches = re.findall(pattern, raw_text, re.DOTALL)
        if matches:
            return matches[0].strip()
        pattern2 = r'```\s*\n(.*?)```'
        matches2 = re.findall(pattern2, raw_text, re.DOTALL)
        if matches2:
            return matches2[0].strip()
        return raw_text.strip()

    def _mock_generate(self, question: str,
                       profile: DatasetProfile) -> str:
        """无API时的模拟代码生成"""
        cols = [c.name for c in profile.columns]
        numeric_cols = [c.name for c in profile.columns
                        if c.col_type == ColumnType.NUMERIC]
        cat_cols = [c.name for c in profile.columns
                    if c.col_type == ColumnType.CATEGORICAL]

        q = question.lower()

        if any(kw in q for kw in ["平均", "均值", "mean"]):
            if cat_cols and numeric_cols:
                return f"""```python
# 按{cat_cols[0]}分组计算{numeric_cols[0]}的平均值
result = df.groupby('{cat_cols[0]}')['{numeric_cols[0]}'].mean().reset_index()
result.columns = ['{cat_cols[0]}', '平均{numeric_cols[0]}']
result = result.sort_values('平均{numeric_cols[0]}', ascending=False)
print(result.to_string(index=False))
```"""

        if any(kw in q for kw in ["趋势", "变化", "走势"]):
            date_cols = [c.name for c in profile.columns
                         if c.col_type == ColumnType.DATETIME]
            if date_cols and numeric_cols:
                return f"""```python
# 按时间分析{numeric_cols[0]}的变化趋势
df['{date_cols[0]}'] = pd.to_datetime(df['{date_cols[0]}'])
result = df.set_index('{date_cols[0]}')['{numeric_cols[0]}'].resample('M').sum().reset_index()
result.columns = ['月份', '合计{numeric_cols[0]}']
print(result.to_string(index=False))
```"""

        if any(kw in q for kw in ["分布", "直方图", "histogram"]):
            if numeric_cols:
                return f"""```python
# 分析{numeric_cols[0]}的分布
result = df['{numeric_cols[0]}'].describe().to_frame().T
print("统计摘要:")
print(result.to_string(index=False))
```"""

        if any(kw in q for kw in ["相关", "关系", "correlation"]):
            if len(numeric_cols) >= 2:
                return f"""```python
# 计算数值列之间的相关系数
result = df[{numeric_cols}].corr().round(3)
print("相关系数矩阵:")
print(result.to_string())
```"""

        # 默认: 基本统计
        return f"""```python
# 数据基本统计分析
result = df.describe(include='all').T
result = result.fillna('-')
print("数据概览:")
print(result.to_string())
```"""


# ============================================================
# 使用演示
# ============================================================

def demo_nl2code():
    """演示NL2Code引擎"""
    import pandas as pd
    import numpy as np

    # 创建示例数据
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "日期": pd.date_range("2024-01-01", periods=n, freq="D"),
        "产品": np.random.choice(["手机", "电脑", "平板", "耳机"], n),
        "地区": np.random.choice(["华东", "华南", "华北", "西南"], n),
        "销量": np.random.randint(10, 500, n),
        "单价": np.random.uniform(100, 5000, n).round(2),
        "满意度": np.random.uniform(3.0, 5.0, n).round(1),
    })

    config = AnalysisConfig()
    profiler = DataProfiler()
    profile = profiler.profile(df, "sales_data.csv")
    engine = NL2CodeEngine(config)

    questions = [
        "各产品的平均销量是多少？",
        "销量的月度变化趋势如何？",
        "单价的分布情况？",
        "销量和满意度之间有相关性吗？",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"问题: {q}")
        print("-"*60)
        result = engine.execute_analysis(q, df, profile)
        print(f"生成代码:\n{result.generated_code}")
        print(f"\n执行结果:")
        if result.success:
            print(result.execution_output)
        else:
            print(f"错误: {result.error}")
        print(f"耗时: {result.execution_time:.2f}s")


if __name__ == "__main__":
    demo_nl2code()
```

---

## 可视化建议

### 智能可视化推荐流程

```
┌──────────────────────────────────────────────────────────────┐
│                智能可视化推荐系统                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ 数据特征  │───▶│ 规则匹配      │───▶│ 图表候选集   │       │
│  │ 分析      │    │              │    │              │       │
│  │           │    │ 列类型       │    │ 柱状图 ★★★  │       │
│  │ ·列类型   │    │ 数据量       │    │ 折线图 ★★   │       │
│  │ ·唯一值数 │    │ 唯一值数     │    │ 饼图   ★    │       │
│  │ ·数据量   │    │ 缺失率       │    │              │       │
│  └──────────┘    └──────────────┘    └──────┬───────┘       │
│                                             │                │
│                                             ▼                │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ 渲染图表  │◀───│ 参数自动配置  │◀───│ LLM增强排序  │       │
│  │           │    │              │    │ (可选)       │       │
│  │ Matplotlib│    │ ·颜色方案    │    │              │       │
│  │ Plotly    │    │ ·标签格式    │    │ 根据分析意图  │       │
│  │ Seaborn   │    │ ·图表尺寸    │    │ 重新排序     │       │
│  └──────────┘    └──────────────┘    └──────────────┘       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 图表类型选择规则

```
┌────────────────┬──────────────────┬────────────────────────┐
│  数据模式      │  推荐图表         │  适用条件               │
├────────────────┼──────────────────┼────────────────────────┤
│ 分类+数值      │ 柱状图/条形图    │ 分类数 ≤ 15            │
│ 分类+占比      │ 饼图/环形图      │ 分类数 ≤ 8             │
│ 时间+数值      │ 折线图/面积图    │ 时间序列数据           │
│ 数值+数值      │ 散点图           │ 探索两变量关系          │
│ 单数值分布     │ 直方图/箱线图    │ 查看数据分布           │
│ 多数值比较     │ 热力图           │ 相关矩阵等             │
│ 分类+数值分布  │ 箱线图/小提琴图  │ 比较各组分布           │
│ 层级+数值      │ 树形图           │ 层级结构数据           │
└────────────────┴──────────────────┴────────────────────────┘
```

### 详细说明

可视化推荐系统根据用户选择的数据列和分析目标，自动推荐最合适的图表类型，并配置颜色方案、标签格式等参数。系统内置基于规则的推荐引擎，同时支持LLM增强以理解更复杂的可视化意图。

### 代码示例

```python
"""
智能可视化推荐与图表生成系统
"""
import io
import base64
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False


# ============================================================
# 图表推荐配置
# ============================================================

@dataclass
class ChartRecommendation:
    """图表推荐结果"""
    chart_type: ChartType
    score: float  # 0~1 推荐分数
    title: str
    description: str
    x_column: str = ""
    y_column: str = ""
    color_column: str = ""
    config: dict = field(default_factory=dict)


COLOR_PALETTES = {
    "default": ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
                 "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7"],
    "warm": ["#E63946", "#F1A208", "#F4845F", "#D62828",
             "#FCBF49", "#EAE2B7", "#F77F00", "#FCA311"],
    "cool": ["#264653", "#2A9D8F", "#4361EE", "#3A86FF",
             "#4895EF", "#4CC9F0", "#7209B7", "#560BAD"],
}


class ChartRecommender:
    """图表类型推荐器"""

    def recommend(self, df: pd.DataFrame,
                  x_col: str = None, y_col: str = None,
                  purpose: str = "explore") -> list:
        """根据数据和列推荐图表类型"""
        recommendations = []

        if x_col and y_col:
            recommendations = self._recommend_bivariate(
                df, x_col, y_col, purpose)
        elif x_col:
            recommendations = self._recommend_univariate(
                df, x_col, purpose)
        else:
            recommendations = self._recommend_overview(df, purpose)

        # 按分数排序
        recommendations.sort(key=lambda r: r.score, reverse=True)
        return recommendations[:5]

    def _recommend_univariate(self, df, col, purpose) -> list:
        """单变量推荐"""
        recs = []
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        nunique = series.nunique()

        if is_numeric:
            recs.append(ChartRecommendation(
                chart_type=ChartType.HISTOGRAM,
                score=0.9,
                title=f"{col}分布直方图",
                description=f"展示{col}的数值分布情况",
                x_column=col,
                config={"bins": min(30, nunique)},
            ))
            recs.append(ChartRecommendation(
                chart_type=ChartType.BOX,
                score=0.7,
                title=f"{col}箱线图",
                description=f"展示{col}的四分位数和异常值",
                x_column=col,
            ))
        else:
            if nunique <= 8:
                recs.append(ChartRecommendation(
                    chart_type=ChartType.PIE,
                    score=0.85,
                    title=f"{col}占比分布",
                    description=f"展示{col}各类别的占比",
                    x_column=col,
                ))
            if nunique <= 20:
                recs.append(ChartRecommendation(
                    chart_type=ChartType.BAR,
                    score=0.9 if nunique > 8 else 0.8,
                    title=f"{col}频次统计",
                    description=f"展示{col}各类别的频次",
                    x_column=col,
                ))
        return recs

    def _recommend_bivariate(self, df, x_col, y_col, purpose) -> list:
        """双变量推荐"""
        recs = []
        x_numeric = pd.api.types.is_numeric_dtype(df[x_col])
        y_numeric = pd.api.types.is_numeric_dtype(df[y_col])
        x_datetime = pd.api.types.is_datetime64_any_dtype(df[x_col])
        x_nunique = df[x_col].nunique()

        if x_datetime and y_numeric:
            recs.append(ChartRecommendation(
                chart_type=ChartType.LINE,
                score=0.95,
                title=f"{y_col}随时间变化趋势",
                description=f"展示{y_col}随{x_col}的变化趋势",
                x_column=x_col, y_column=y_col,
            ))
            recs.append(ChartRecommendation(
                chart_type=ChartType.AREA,
                score=0.8,
                title=f"{y_col}面积图",
                description=f"展示{y_col}随时间的累积趋势",
                x_column=x_col, y_column=y_col,
            ))

        elif x_numeric and y_numeric:
            recs.append(ChartRecommendation(
                chart_type=ChartType.SCATTER,
                score=0.9,
                title=f"{x_col} vs {y_col}散点图",
                description=f"探索{x_col}与{y_col}之间的关系",
                x_column=x_col, y_column=y_col,
            ))

        elif not x_numeric and y_numeric:
            if x_nunique <= 15:
                recs.append(ChartRecommendation(
                    chart_type=ChartType.BAR,
                    score=0.9,
                    title=f"各{x_col}的{y_col}对比",
                    description=f"比较不同{x_col}的{y_col}",
                    x_column=x_col, y_column=y_col,
                ))
            recs.append(ChartRecommendation(
                chart_type=ChartType.BOX,
                score=0.8,
                title=f"各{x_col}的{y_col}分布",
                description=f"比较各{x_col}类别下{y_col}的分布",
                x_column=x_col, y_column=y_col,
            ))

        return recs

    def _recommend_overview(self, df, purpose) -> list:
        """总览型推荐"""
        recs = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) >= 2:
            recs.append(ChartRecommendation(
                chart_type=ChartType.HEATMAP,
                score=0.85,
                title="数值列相关性热力图",
                description="展示各数值列之间的相关系数",
                config={"columns": numeric_cols},
            ))
        return recs


# ============================================================
# 图表渲染器
# ============================================================

class ChartRenderer:
    """图表渲染器 - 支持Matplotlib"""

    def __init__(self, palette: str = "default"):
        self.colors = COLOR_PALETTES.get(palette, COLOR_PALETTES["default"])

    def render(self, df: pd.DataFrame,
               rec: ChartRecommendation,
               save_path: str = None) -> str:
        """根据推荐配置渲染图表，返回base64或文件路径"""
        if not HAS_MPL:
            return "[matplotlib未安装，跳过图表渲染]"

        fig, ax = plt.subplots(figsize=(10, 6))

        render_map = {
            ChartType.BAR: self._render_bar,
            ChartType.LINE: self._render_line,
            ChartType.SCATTER: self._render_scatter,
            ChartType.PIE: self._render_pie,
            ChartType.HISTOGRAM: self._render_histogram,
            ChartType.HEATMAP: self._render_heatmap,
            ChartType.BOX: self._render_box,
            ChartType.AREA: self._render_area,
        }

        renderer = render_map.get(rec.chart_type)
        if renderer:
            renderer(df, rec, fig, ax)

        ax.set_title(rec.title, fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return save_path
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode()

    def _render_bar(self, df, rec, fig, ax):
        x, y = rec.x_column, rec.y_column
        if y:
            plot_data = df.groupby(x)[y].mean().sort_values(ascending=False)
        else:
            plot_data = df[x].value_counts()
        bars = ax.bar(plot_data.index.astype(str), plot_data.values,
                      color=self.colors[:len(plot_data)])
        ax.set_xlabel(x)
        ax.set_ylabel(y if y else "计数")
        # 添加数据标签
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=9)

    def _render_line(self, df, rec, fig, ax):
        x, y = rec.x_column, rec.y_column
        data = df.sort_values(x)
        ax.plot(data[x], data[y], color=self.colors[0],
                linewidth=2, marker='o', markersize=3)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.3)

    def _render_scatter(self, df, rec, fig, ax):
        x, y = rec.x_column, rec.y_column
        ax.scatter(df[x], df[y], c=self.colors[0],
                   alpha=0.6, edgecolors='white', linewidth=0.5)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        # 添加趋势线
        z = np.polyfit(df[x].dropna(), df[y].dropna(), 1)
        p = np.poly1d(z)
        x_sorted = np.sort(df[x].dropna())
        ax.plot(x_sorted, p(x_sorted), "--",
                color=self.colors[1], alpha=0.8, label="趋势线")
        ax.legend()

    def _render_pie(self, df, rec, fig, ax):
        data = df[rec.x_column].value_counts()
        colors = self.colors[:len(data)]
        wedges, texts, autotexts = ax.pie(
            data.values, labels=data.index,
            colors=colors, autopct='%1.1f%%',
            startangle=90, pctdistance=0.85)
        for text in autotexts:
            text.set_fontsize(9)

    def _render_histogram(self, df, rec, fig, ax):
        bins = rec.config.get("bins", 30)
        ax.hist(df[rec.x_column].dropna(), bins=bins,
                color=self.colors[0], edgecolor='white', alpha=0.8)
        ax.set_xlabel(rec.x_column)
        ax.set_ylabel("频次")
        # 添加均值线
        mean_val = df[rec.x_column].mean()
        ax.axvline(mean_val, color=self.colors[1],
                   linestyle='--', label=f'均值: {mean_val:.2f}')
        ax.legend()

    def _render_heatmap(self, df, rec, fig, ax):
        cols = rec.config.get("columns", df.select_dtypes(
            include=[np.number]).columns.tolist())
        corr = df[cols].corr()
        if HAS_SNS:
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlBu_r',
                        center=0, ax=ax, square=True)
        else:
            im = ax.imshow(corr.values, cmap='RdYlBu_r', aspect='auto')
            ax.set_xticks(range(len(cols)))
            ax.set_yticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=45, ha='right')
            ax.set_yticklabels(cols)
            plt.colorbar(im, ax=ax)

    def _render_box(self, df, rec, fig, ax):
        x, y = rec.x_column, rec.y_column
        if y:
            groups = df.groupby(x)[y].apply(list).to_dict()
            ax.boxplot(groups.values(), labels=groups.keys())
            ax.set_ylabel(y)
        else:
            ax.boxplot(df[x].dropna())
            ax.set_ylabel(x)

    def _render_area(self, df, rec, fig, ax):
        x, y = rec.x_column, rec.y_column
        data = df.sort_values(x)
        ax.fill_between(data[x], data[y],
                        alpha=0.4, color=self.colors[0])
        ax.plot(data[x], data[y],
                color=self.colors[0], linewidth=1.5)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.3)


# ============================================================
# 使用演示
# ============================================================

def demo_visualization():
    """演示可视化推荐与渲染"""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "日期": pd.date_range("2024-01-01", periods=n, freq="D"),
        "产品": np.random.choice(["手机", "电脑", "平板", "耳机"], n),
        "地区": np.random.choice(["华东", "华南", "华北", "西南"], n),
        "销量": np.random.randint(10, 500, n),
        "单价": np.random.uniform(100, 5000, n).round(2),
        "满意度": np.random.uniform(3.0, 5.0, n).round(1),
    })

    recommender = ChartRecommender()
    renderer = ChartRenderer(palette="default")

    # 场景1: 分类+数值
    print("=== 场景1: 产品 vs 销量 ===")
    recs = recommender.recommend(df, x_col="产品", y_col="销量")
    for i, r in enumerate(recs):
        print(f"  {i+1}. {r.chart_type.value} (分数={r.score:.2f}): {r.title}")

    # 场景2: 时间+数值
    print("\n=== 场景2: 日期 vs 销量 ===")
    recs = recommender.recommend(df, x_col="日期", y_col="销量")
    for i, r in enumerate(recs):
        print(f"  {i+1}. {r.chart_type.value} (分数={r.score:.2f}): {r.title}")

    # 场景3: 单变量
    print("\n=== 场景3: 单价分布 ===")
    recs = recommender.recommend(df, x_col="单价")
    for i, r in enumerate(recs):
        print(f"  {i+1}. {r.chart_type.value} (分数={r.score:.2f}): {r.title}")

    # 渲染示例
    if HAS_MPL:
        recs = recommender.recommend(df, x_col="产品", y_col="销量")
        if recs:
            path = renderer.render(df, recs[0], save_path="demo_chart.png")
            print(f"\n图表已保存: {path}")


if __name__ == "__main__":
    demo_visualization()
```

---

## 自动化报告

### 报告生成流程

```
┌──────────────────────────────────────────────────────────────┐
│                  自动化报告生成流程                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ 数据分析  │───▶│ 洞察提取      │───▶│ 模板渲染     │       │
│  │          │    │              │    │              │       │
│  │ ·描述统计 │    │ ·趋势发现    │    │ ·Jinja2模板  │       │
│  │ ·分组汇总 │    │ ·异常检测    │    │ ·图表嵌入    │       │
│  │ ·相关分析 │    │ ·排名变化    │    │ ·样式美化    │       │
│  └──────────┘    └──────────────┘    └──────┬───────┘       │
│                                             │                │
│                                             ▼                │
│                  ┌──────────────┐    ┌──────────────┐       │
│                  │ PDF导出      │◀───│ HTML报告     │       │
│                  │ (可选)       │    │              │       │
│                  └──────────────┘    └──────────────┘       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 详细说明

自动化报告模块将数据分析结果组装成结构化的HTML报告，包含统计摘要、图表和AI生成的洞察文字。支持Jinja2模板定制和PDF导出。

### 代码示例

```python
"""
自动化数据分析报告生成器
"""
import os
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# 数据洞察提取器
# ============================================================

class InsightExtractor:
    """从数据中提取关键洞察"""

    def extract_all(self, df: pd.DataFrame) -> list:
        """提取所有洞察"""
        insights = []
        insights.extend(self._basic_stats(df))
        insights.extend(self._find_outliers(df))
        insights.extend(self._find_trends(df))
        insights.extend(self._find_correlations(df))
        return insights

    def _basic_stats(self, df: pd.DataFrame) -> list:
        """基本统计洞察"""
        insights = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            mean_val = series.mean()
            std_val = series.std()
            cv = std_val / mean_val if mean_val != 0 else 0

            if cv > 0.5:
                insights.append({
                    "type": "variation",
                    "level": "warning",
                    "column": col,
                    "text": f"「{col}」变异系数较高({cv:.2f})，数据波动较大",
                    "value": round(cv, 3),
                })

        # 缺失值洞察
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        if len(missing_cols) > 0:
            worst_col = missing_cols.idxmax()
            worst_pct = missing_cols.max() / len(df) * 100
            insights.append({
                "type": "missing",
                "level": "info",
                "column": worst_col,
                "text": f"「{worst_col}」缺失率最高，达{worst_pct:.1f}%",
                "value": round(worst_pct, 1),
            })
        return insights

    def _find_outliers(self, df: pd.DataFrame) -> list:
        """异常值检测 (IQR方法)"""
        insights = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_count = ((series < lower) | (series > upper)).sum()
            outlier_pct = outlier_count / len(series) * 100

            if outlier_pct > 1:
                insights.append({
                    "type": "outlier",
                    "level": "warning",
                    "column": col,
                    "text": f"「{col}」存在{outlier_count}个异常值({outlier_pct:.1f}%)",
                    "value": outlier_count,
                })
        return insights

    def _find_trends(self, df: pd.DataFrame) -> list:
        """趋势发现"""
        insights = []
        date_cols = df.select_dtypes(include=["datetime64"]).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for dcol in date_cols:
            for ncol in numeric_cols:
                try:
                    monthly = df.set_index(dcol)[ncol].resample('M').mean()
                    if len(monthly) >= 3:
                        first_half = monthly[:len(monthly)//2].mean()
                        second_half = monthly[len(monthly)//2:].mean()
                        change_pct = (second_half - first_half) / first_half * 100 \
                            if first_half != 0 else 0

                        if abs(change_pct) > 10:
                            direction = "上升" if change_pct > 0 else "下降"
                            insights.append({
                                "type": "trend",
                                "level": "info",
                                "column": ncol,
                                "text": f"「{ncol}」呈{direction}趋势，"
                                        f"后期较前期变化{abs(change_pct):.1f}%",
                                "value": round(change_pct, 1),
                            })
                except Exception:
                    continue
        return insights

    def _find_correlations(self, df: pd.DataFrame) -> list:
        """相关性发现"""
        insights = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return insights

        corr = df[numeric_cols].corr()
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                val = corr.iloc[i, j]
                if abs(val) > 0.6:
                    direction = "正相关" if val > 0 else "负相关"
                    strength = "强" if abs(val) > 0.8 else "中等"
                    insights.append({
                        "type": "correlation",
                        "level": "info",
                        "column": f"{numeric_cols[i]}+{numeric_cols[j]}",
                        "text": f"「{numeric_cols[i]}」与「{numeric_cols[j]}」"
                                f"存在{strength}{direction}(r={val:.2f})",
                        "value": round(val, 3),
                    })
        return insights


# ============================================================
# HTML报告生成器
# ============================================================

class ReportGenerator:
    """HTML分析报告生成器"""

    HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
&lt;head&gt;
    <meta charset="UTF-8">
    &lt;title&gt;{{ title }}</title>
    &lt;style&gt;
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Microsoft YaHei', sans-serif;
               background: #f5f7fa; color: #333; padding: 40px; }
        .container { max-width: 1000px; margin: 0 auto;
                     background: white; border-radius: 12px;
                     box-shadow: 0 2px 12px rgba(0,0,0,0.08);
                     padding: 40px; }
        h1 { color: #1a1a2e; border-bottom: 3px solid #4E79A7;
             padding-bottom: 12px; margin-bottom: 24px; }
        h2 { color: #16213e; margin: 28px 0 16px; padding-left: 12px;
             border-left: 4px solid #4E79A7; }
        .meta { color: #888; font-size: 14px; margin-bottom: 20px; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 16px; margin: 20px 0; }
        .summary-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white; border-radius: 8px; padding: 20px; text-align: center; }
        .summary-card .value { font-size: 28px; font-weight: bold; }
        .summary-card .label { font-size: 13px; opacity: 0.85; margin-top: 4px; }
        table { width: 100%; border-collapse: collapse; margin: 16px 0;
                font-size: 14px; }
        th, td { padding: 10px 14px; text-align: left;
                 border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; color: #555; }
        tr:hover { background: #f5f8ff; }
        .insight { padding: 12px 16px; margin: 8px 0;
                   border-radius: 6px; font-size: 14px; }
        .insight.info { background: #e8f4fd; border-left: 4px solid #4E79A7; }
        .insight.warning { background: #fff3cd; border-left: 4px solid #F28E2B; }
        .chart-container { text-align: center; margin: 20px 0; }
        .chart-container img { max-width: 100%; border-radius: 8px;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .footer { text-align: center; color: #aaa; font-size: 12px;
                  margin-top: 30px; padding-top: 20px;
                  border-top: 1px solid #eee; }
    </style>
</head>
&lt;body&gt;
<div class="container">
    <h1>{{ title }}</h1>
    <div class="meta">生成时间: {{ generated_at }} | 数据文件: {{ filename }}</div>

    <h2>数据概览</h2>
    <div class="summary-grid">
        <div class="summary-card">
            <div class="value">{{ row_count }}</div>
            <div class="label">总行数</div>
        </div>
        <div class="summary-card">
            <div class="value">{{ col_count }}</div>
            <div class="label">总列数</div>
        </div>
        <div class="summary-card">
            <div class="value">{{ missing_count }}</div>
            <div class="label">缺失值数</div>
        </div>
        <div class="summary-card">
            <div class="value">{{ memory_mb }} MB</div>
            <div class="label">内存占用</div>
        </div>
    </div>

    <h2>列信息统计</h2>
    &lt;table&gt;
        &lt;thead&gt;&lt;tr&gt;
            &lt;th&gt;列名</th>&lt;th&gt;类型</th>&lt;th&gt;非空数</th>
            &lt;th&gt;唯一值</th>&lt;th&gt;统计摘要</th>
        </tr></thead>
        &lt;tbody&gt;
        {% for col in columns %}
        &lt;tr&gt;
            &lt;td&gt;&lt;strong&gt;{{ col.name }}</strong></td>
            &lt;td&gt;{{ col.col_type }}</td>
            &lt;td&gt;{{ col.non_null }}</td>
            &lt;td&gt;{{ col.unique }}</td>
            &lt;td&gt;{{ col.summary }}</td>
        </tr>
        {% endfor %}
        </tbody>
    </table>

    {% if insights %}
    <h2>数据洞察</h2>
    {% for insight in insights %}
    <div class="insight {{ insight.level }}">{{ insight.text }}</div>
    {% endfor %}
    {% endif %}

    {% if charts %}
    <h2>可视化分析</h2>
    {% for chart in charts %}
    <div class="chart-container">
        <h3>{{ chart.title }}</h3>
        <img src="{{ chart.src }}" alt="{{ chart.title }}">
    </div>
    {% endfor %}
    {% endif %}

    <div class="footer">
        由 AI数据分析助手 自动生成 | {{ generated_at }}
    </div>
</div>
</body>
</html>"""

    def generate(self, df: pd.DataFrame,
                 profile: DatasetProfile,
                 insights: list,
                 chart_paths: list = None,
                 title: str = "数据分析报告") -> str:
        """生成HTML报告"""
        try:
            from jinja2 import Template
            template = Template(self.HTML_TEMPLATE)
        except ImportError:
            return self._generate_simple_html(
                df, profile, insights, chart_paths, title)

        columns_data = []
        for col in profile.columns:
            summary = ""
            if col.col_type == ColumnType.NUMERIC and col.mean is not None:
                summary = (f"均值={col.mean:.2f}, 中位数={col.median:.2f}, "
                           f"范围=[{col.min_val:.2f}, {col.max_val:.2f}]")
            elif col.top_values:
                top3 = list(col.top_values.items())[:3]
                summary = ", ".join(f"{k}({v})" for k, v in top3)

            columns_data.append({
                "name": col.name,
                "col_type": col.col_type.value,
                "non_null": col.non_null_count,
                "unique": col.unique_count,
                "summary": summary,
            })

        charts_data = []
        if chart_paths:
            for i, path in enumerate(chart_paths):
                charts_data.append({
                    "title": f"图表 {i+1}",
                    "src": path,
                })

        total_missing = sum(profile.missing_summary.values()) \
            if profile.missing_summary else 0

        html = template.render(
            title=title,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filename=profile.filename,
            row_count=f"{profile.row_count:,}",
            col_count=profile.col_count,
            missing_count=f"{total_missing:,}",
            memory_mb=f"{profile.memory_usage_mb:.2f}",
            columns=columns_data,
            insights=insights,
            charts=charts_data,
        )
        return html

    def _generate_simple_html(self, df, profile, insights,
                               chart_paths, title):
        """无Jinja2时的简单HTML生成"""
        html = f"""<!DOCTYPE html>
&lt;html&gt;&lt;head&gt;<meta charset="UTF-8">&lt;title&gt;{title}</title>
&lt;style&gt;body{{font-family:sans-serif;padding:40px;}}
table{{border-collapse:collapse;width:100%;}}
th,td{{border:1px solid #ddd;padding:8px;text-align:left;}}
th{{background:#f5f5f5;}}
.insight{{padding:10px;margin:8px 0;background:#e8f4fd;border-radius:4px;}}
</style></head>&lt;body&gt;
<h1>{title}</h1>
&lt;p&gt;数据: {profile.filename} | 行数: {profile.row_count} | 列数: {profile.col_count}</p>
<h2>描述统计</h2>
{df.describe().to_html()}
<h2>洞察</h2>"""
        for ins in insights:
            html += f'<div class="insight">{ins["text"]}</div>\n'
        html += "</body></html>"
        return html

    def save_report(self, html: str, output_path: str):
        """保存HTML报告"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"报告已保存: {output_path}")


# ============================================================
# 使用演示
# ============================================================

def demo_report():
    """演示自动报告生成"""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "日期": pd.date_range("2024-01-01", periods=n, freq="D"),
        "产品": np.random.choice(["手机", "电脑", "平板", "耳机"], n),
        "地区": np.random.choice(["华东", "华南", "华北", "西南"], n),
        "销量": np.random.randint(10, 500, n),
        "单价": np.random.uniform(100, 5000, n).round(2),
        "满意度": np.random.uniform(3.0, 5.0, n).round(1),
    })
    df.loc[np.random.choice(n, 10), "满意度"] = np.nan

    profiler = DataProfiler()
    profile = profiler.profile(df, "sales_data.csv")

    extractor = InsightExtractor()
    insights = extractor.extract_all(df)

    print(f"发现 {len(insights)} 个洞察:")
    for ins in insights:
        print(f"  [{ins['level']}] {ins['text']}")

    generator = ReportGenerator()
    html = generator.generate(df, profile, insights,
                              title="销售数据分析报告")
    generator.save_report(html, "reports/sales_report.html")
    print(f"HTML大小: {len(html):,} 字节")


if __name__ == "__main__":
    demo_report()
```

---

## Notebook集成

### 集成架构

```
┌──────────────────────────────────────────────────────────────┐
│              Jupyter Notebook 集成架构                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌───────────────────────────────────────────────┐           │
│  │           Jupyter Notebook                     │           │
│  │                                               │           │
│  │  Cell 1: %load_ext ai_analyst                 │           │
│  │                                               │           │
│  │  Cell 2: %%ai_query                           │           │
│  │          各产品的月均销量趋势                    │           │
│  │                                               │           │
│  │  Cell 3: [自动生成的Pandas代码]                 │           │
│  │                                               │           │
│  │  Cell 4: [自动生成的图表]                       │           │
│  └────────────────────┬──────────────────────────┘           │
│                       │                                      │
│                       ▼                                      │
│  ┌────────────────────────────────────────────┐              │
│  │  AI Analyst Kernel Extension                │              │
│  │                                            │              │
│  │  ·Magic Commands (%ai, %%ai_query)         │              │
│  │  ·自动变量检测 (DataFrame发现)              │              │
│  │  ·代码注入 (生成代码插入新Cell)             │              │
│  │  ·上下文管理 (跟踪对话历史)                 │              │
│  └────────────────────┬───────────────────────┘              │
│                       │                                      │
│                       ▼                                      │
│  ┌────────────────────────────────────────────┐              │
│  │  NL2Code引擎 + 可视化推荐 + 报告生成        │              │
│  └────────────────────────────────────────────┘              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 详细说明

Notebook集成模块提供了两种方式将AI分析能力嵌入Jupyter工作流:
1. **IPython Magic Commands**: 通过 `%%ai_query` 魔法命令直接在Cell中提问
2. **交互式Widget**: 提供搜索框和按钮界面，适合非技术用户

### 代码示例

```python
"""
Jupyter Notebook 集成 - IPython Magic Commands
"""
import json
from datetime import datetime

import pandas as pd
import numpy as np

try:
    from IPython.core.magic import (
        Magics, magics_class, line_magic, cell_magic
    )
    from IPython.display import display, HTML, Markdown
    from IPython import get_ipython
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


# ============================================================
# Notebook 环境变量检测
# ============================================================

class NotebookContext:
    """Notebook环境上下文管理"""

    def __init__(self):
        self.history = []

    def find_dataframes(self, user_ns: dict) -> dict:
        """从Notebook命名空间中发现所有DataFrame"""
        dfs = {}
        for name, obj in user_ns.items():
            if name.startswith("_"):
                continue
            if isinstance(obj, pd.DataFrame) and len(obj) > 0:
                dfs[name] = {
                    "shape": obj.shape,
                    "columns": list(obj.columns),
                    "dtypes": {col: str(dt) for col, dt
                               in obj.dtypes.items()},
                    "memory_mb": obj.memory_usage(
                        deep=True).sum() / 1024 / 1024,
                }
        return dfs

    def select_best_dataframe(self, user_ns: dict,
                              question: str) -> tuple:
        """根据问题自动选择最相关的DataFrame"""
        dfs = self.find_dataframes(user_ns)
        if not dfs:
            return None, None

        # 简单策略: 选择列名与问题最相关的DataFrame
        best_name, best_score = None, -1
        for name, info in dfs.items():
            score = sum(1 for col in info["columns"]
                        if col in question)
            # 加上变量名匹配
            if name.lower() in question.lower():
                score += 5
            if score > best_score:
                best_score = score
                best_name = name

        if best_name is None:
            # 默认选最大的DataFrame
            best_name = max(dfs, key=lambda k: dfs[k]["shape"][0])

        return best_name, user_ns[best_name]

    def add_history(self, question: str, code: str, success: bool):
        """记录分析历史"""
        self.history.append({
            "question": question,
            "code": code,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        })


# ============================================================
# IPython Magic Extension (在Jupyter中使用)
# ============================================================

if HAS_IPYTHON:

    @magics_class
    class AIAnalystMagic(Magics):
        """AI数据分析 Magic Commands"""

        def __init__(self, shell):
            super().__init__(shell)
            self.context = NotebookContext()
            self.config = AnalysisConfig()
            self.engine = NL2CodeEngine(self.config)
            self.profiler = DataProfiler()
            self.recommender = ChartRecommender()
            self.renderer = ChartRenderer()

        @line_magic
        def ai_datasets(self, line):
            """列出当前可用的DataFrame: %ai_datasets"""
            dfs = self.context.find_dataframes(self.shell.user_ns)
            if not dfs:
                display(HTML("&lt;p&gt;未发现DataFrame变量</p>"))
                return

            rows = ""
            for name, info in dfs.items():
                cols = ", ".join(info["columns"][:8])
                if len(info["columns"]) > 8:
                    cols += f" ... (+{len(info['columns'])-8})"
                rows += f"""&lt;tr&gt;
                    &lt;td&gt;&lt;b&gt;{name}</b></td>
                    &lt;td&gt;{info['shape'][0]:,} x {info['shape'][1]}</td>
                    &lt;td&gt;{info['memory_mb']:.2f} MB</td>
                    &lt;td&gt;{cols}</td>
                </tr>"""

            html = f"""<table style="border-collapse:collapse;width:100%">
            <tr style="background:#f5f5f5">
                <th style="padding:8px;border:1px solid #ddd">变量名</th>
                <th style="padding:8px;border:1px solid #ddd">形状</th>
                <th style="padding:8px;border:1px solid #ddd">内存</th>
                <th style="padding:8px;border:1px solid #ddd">列名</th>
            </tr>{rows}</table>"""
            display(HTML(html))

        @cell_magic
        def ai_query(self, line, cell):
            """AI数据分析查询:
            %%ai_query [df_name]
            你的分析问题
            """
            question = cell.strip()
            df_name = line.strip() if line.strip() else None

            if df_name and df_name in self.shell.user_ns:
                df = self.shell.user_ns[df_name]
            else:
                df_name, df = self.context.select_best_dataframe(
                    self.shell.user_ns, question)

            if df is None:
                display(HTML("<p style='color:red'>未找到DataFrame</p>"))
                return

            display(HTML(
                f"&lt;p&gt;使用数据: &lt;b&gt;{df_name}</b> "
                f"({df.shape[0]:,} 行 x {df.shape[1]} 列)</p>"
            ))

            profile = self.profiler.profile(df, df_name)
            result = self.engine.execute_analysis(question, df, profile)

            # 显示生成的代码
            display(HTML(
                f"&lt;details&gt;&lt;summary&gt;生成的代码</summary>"
                f"&lt;pre&gt;{result.generated_code}</pre></details>"
            ))

            if result.success:
                if isinstance(result.execution_output, pd.DataFrame):
                    display(result.execution_output)
                elif result.execution_output:
                    display(HTML(f"&lt;pre&gt;{result.execution_output}</pre>"))
                display(HTML(
                    f"<p style='color:green'>执行成功 "
                    f"({result.execution_time:.2f}s)</p>"
                ))
            else:
                display(HTML(
                    f"<p style='color:red'>执行错误: "
                    f"{result.error}</p>"
                ))

            self.context.add_history(
                question, result.generated_code, result.success)

        @line_magic
        def ai_report(self, line):
            """生成数据分析报告: %ai_report df_name [output_path]"""
            parts = line.strip().split()
            if not parts:
                display(HTML("<p style='color:red'>用法: %ai_report df_name [output.html]</p>"))
                return

            df_name = parts[0]
            output = parts[1] if len(parts) > 1 else f"{df_name}_report.html"

            if df_name not in self.shell.user_ns:
                display(HTML(f"<p style='color:red'>变量 {df_name} 不存在</p>"))
                return

            df = self.shell.user_ns[df_name]
            profile = self.profiler.profile(df, df_name)
            extractor = InsightExtractor()
            insights = extractor.extract_all(df)

            generator = ReportGenerator()
            html = generator.generate(df, profile, insights,
                                      title=f"{df_name} 分析报告")
            generator.save_report(html, output)
            display(HTML(f"&lt;p&gt;报告已保存: <a href='{output}'>{output}</a></p>"))

        @line_magic
        def ai_viz(self, line):
            """智能可视化: %ai_viz df_name [x_col] [y_col]"""
            parts = line.strip().split()
            if not parts:
                display(HTML("<p style='color:red'>用法: %ai_viz df x_col [y_col]</p>"))
                return

            df_name = parts[0]
            x_col = parts[1] if len(parts) > 1 else None
            y_col = parts[2] if len(parts) > 2 else None

            df = self.shell.user_ns.get(df_name)
            if df is None:
                display(HTML(f"<p style='color:red'>变量 {df_name} 不存在</p>"))
                return

            recs = self.recommender.recommend(df, x_col=x_col, y_col=y_col)
            if recs:
                b64 = self.renderer.render(df, recs[0])
                display(HTML(
                    f"<h4>{recs[0].title}</h4>"
                    f"<img src='data:image/png;base64,{b64}'/>"
                ))

    def load_ipython_extension(ipython):
        """加载IPython扩展: %load_ext ai_analyst"""
        ipython.register_magics(AIAnalystMagic)


# ============================================================
# 非Notebook环境的模拟演示
# ============================================================

def demo_notebook_integration():
    """模拟Notebook集成功能"""
    print("=" * 60)
    print("Jupyter Notebook 集成演示 (非Notebook环境模拟)")
    print("=" * 60)

    # 模拟用户命名空间
    np.random.seed(42)
    n = 200
    user_ns = {
        "sales_df": pd.DataFrame({
            "日期": pd.date_range("2024-01-01", periods=n, freq="D"),
            "产品": np.random.choice(["手机", "电脑", "平板", "耳机"], n),
            "地区": np.random.choice(["华东", "华南", "华北", "西南"], n),
            "销量": np.random.randint(10, 500, n),
            "单价": np.random.uniform(100, 5000, n).round(2),
        }),
        "user_df": pd.DataFrame({
            "用户ID": range(100),
            "年龄": np.random.randint(18, 60, 100),
            "消费金额": np.random.uniform(0, 10000, 100).round(2),
        }),
        "some_string": "这不是DataFrame",
    }

    ctx = NotebookContext()

    # 1. 发现DataFrame
    print("\n1. 发现的DataFrame:")
    dfs = ctx.find_dataframes(user_ns)
    for name, info in dfs.items():
        print(f"  {name}: {info['shape']}, 列={info['columns']}")

    # 2. 自动选择DataFrame
    print("\n2. 根据问题选择DataFrame:")
    questions = [
        "各产品的平均销量",
        "用户年龄分布",
    ]
    for q in questions:
        name, _ = ctx.select_best_dataframe(user_ns, q)
        print(f"  问题: {q} -> 选择: {name}")

    # 3. 执行分析
    print("\n3. 模拟 %%ai_query:")
    config = AnalysisConfig()
    engine = NL2CodeEngine(config)
    profiler = DataProfiler()

    df = user_ns["sales_df"]
    profile = profiler.profile(df, "sales_df")
    result = engine.execute_analysis("各产品的平均销量是多少？", df, profile)

    print(f"  代码: {result.generated_code[:100]}...")
    if result.success:
        print(f"  结果: {result.execution_output}")
    print(f"  耗时: {result.execution_time:.3f}s")

    # 使用说明
    print("\n" + "=" * 60)
    print("在 Jupyter Notebook 中使用:")
    print("-" * 60)
    print("""
# 加载扩展
%load_ext ai_analyst

# 查看可用数据集
%ai_datasets

# AI分析查询
%%ai_query sales_df
各产品的月均销量趋势如何？

# 智能可视化
%ai_viz sales_df 产品 销量

# 生成报告
%ai_report sales_df output.html
""")


if __name__ == "__main__":
    demo_notebook_integration()
```

---

## 完整项目

### 完整系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                  AI数据分析助手 - 完整项目架构                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  前端层                                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              HTML/CSS/JavaScript Web界面                  │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │文件上传区│ │对话分析区│ │图表展示区│ │报告下载区│   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │ HTTP/SSE                           │
│  后端层                     ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    FastAPI 服务                            │   │
│  │  POST /upload   POST /analyze   GET /report   GET /chart │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                    │
│  引擎层                     ▼                                    │
│  ┌──────────┐ ┌──────────────┐ ┌───────────┐ ┌───────────┐    │
│  │DataLoader│ │NL2CodeEngine │ │ChartRecom.│ │ReportGen. │    │
│  │DataProf. │ │CodeSandbox   │ │ChartRend. │ │InsightExt.│    │
│  └──────────┘ └──────────────┘ └───────────┘ └───────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### FastAPI后端服务

```python
"""
AI数据分析助手 - FastAPI完整后端服务
"""
import os
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

try:
    from fastapi import FastAPI, UploadFile, File, HTTPException, Form
    from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# ============================================================
# 会话管理
# ============================================================

class SessionStore:
    """分析会话管理"""

    def __init__(self):
        self.sessions = {}

    def create_session(self, filename: str, df: pd.DataFrame,
                       profile: DatasetProfile) -> str:
        sid = str(uuid.uuid4())[:8]
        self.sessions[sid] = {
            "id": sid,
            "filename": filename,
            "df": df,
            "profile": profile,
            "history": [],
            "charts": [],
            "created_at": datetime.now().isoformat(),
        }
        return sid

    def get_session(self, sid: str) -> dict:
        if sid not in self.sessions:
            raise KeyError(f"会话不存在: {sid}")
        return self.sessions[sid]

    def add_history(self, sid: str, question: str,
                    code: str, result: str, success: bool):
        session = self.get_session(sid)
        session["history"].append({
            "question": question,
            "code": code,
            "result": result,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        })


# ============================================================
# FastAPI 应用
# ============================================================

UPLOAD_DIR = Path("./uploads")
REPORT_DIR = Path("./reports")
CHART_DIR = Path("./charts")

for d in [UPLOAD_DIR, REPORT_DIR, CHART_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if HAS_FASTAPI:
    app = FastAPI(title="AI数据分析助手", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 全局实例
    config = AnalysisConfig()
    loader = DataLoader(config)
    profiler = DataProfiler()
    engine = NL2CodeEngine(config)
    recommender = ChartRecommender()
    renderer = ChartRenderer()
    insight_extractor = InsightExtractor()
    report_generator = ReportGenerator()
    store = SessionStore()

    # --- Pydantic模型 ---
    class AnalyzeRequest(BaseModel):
        session_id: str
        question: str

    class VizRequest(BaseModel):
        session_id: str
        x_column: str
        y_column: Optional[str] = None

    # --- 接口 ---

    @app.post("/api/upload")
    async def upload_file(file: UploadFile = File(...)):
        """上传数据文件，创建分析会话"""
        suffix = Path(file.filename).suffix.lower()
        if suffix not in config.supported_formats:
            raise HTTPException(400, f"不支持的格式: {suffix}")

        # 保存文件
        file_id = str(uuid.uuid4())[:8]
        save_path = UPLOAD_DIR / f"{file_id}{suffix}"
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            df = loader.load(str(save_path))
            profile = profiler.profile(df, file.filename)
            sid = store.create_session(file.filename, df, profile)

            return {
                "session_id": sid,
                "filename": file.filename,
                "rows": profile.row_count,
                "columns": profile.col_count,
                "column_info": [
                    {
                        "name": c.name,
                        "type": c.col_type.value,
                        "non_null": c.non_null_count,
                        "unique": c.unique_count,
                    }
                    for c in profile.columns
                ],
                "preview": df.head(10).to_dict(orient="records"),
            }
        except Exception as e:
            raise HTTPException(500, f"文件处理失败: {str(e)}")

    @app.post("/api/analyze")
    async def analyze(req: AnalyzeRequest):
        """自然语言分析查询"""
        try:
            session = store.get_session(req.session_id)
        except KeyError:
            raise HTTPException(404, "会话不存在")

        df = session["df"]
        profile = session["profile"]

        result = engine.execute_analysis(req.question, df, profile)

        output_str = ""
        if result.success:
            if isinstance(result.execution_output, pd.DataFrame):
                output_str = result.execution_output.to_string(index=False)
            else:
                output_str = str(result.execution_output)

        store.add_history(
            req.session_id, req.question,
            result.generated_code, output_str, result.success)

        return {
            "success": result.success,
            "question": req.question,
            "code": result.generated_code,
            "output": output_str,
            "error": result.error,
            "execution_time": round(result.execution_time, 3),
        }

    @app.post("/api/visualize")
    async def visualize(req: VizRequest):
        """智能可视化"""
        try:
            session = store.get_session(req.session_id)
        except KeyError:
            raise HTTPException(404, "会话不存在")

        df = session["df"]
        recs = recommender.recommend(
            df, x_col=req.x_column, y_col=req.y_column)

        if not recs:
            return {"charts": [], "message": "无推荐图表"}

        charts = []
        for rec in recs[:3]:
            chart_id = str(uuid.uuid4())[:8]
            save_path = str(CHART_DIR / f"{chart_id}.png")
            try:
                renderer.render(df, rec, save_path=save_path)
                charts.append({
                    "id": chart_id,
                    "type": rec.chart_type.value,
                    "title": rec.title,
                    "description": rec.description,
                    "path": f"/charts/{chart_id}.png",
                    "score": rec.score,
                })
            except Exception as e:
                print(f"图表渲染失败: {e}")

        return {"charts": charts}

    @app.post("/api/report")
    async def generate_report(session_id: str = Form(...),
                              title: str = Form("数据分析报告")):
        """生成分析报告"""
        try:
            session = store.get_session(session_id)
        except KeyError:
            raise HTTPException(404, "会话不存在")

        df = session["df"]
        profile = session["profile"]
        insights = insight_extractor.extract_all(df)

        html = report_generator.generate(
            df, profile, insights,
            chart_paths=session.get("charts", []),
            title=title)

        report_id = str(uuid.uuid4())[:8]
        report_path = REPORT_DIR / f"{report_id}.html"
        report_generator.save_report(html, str(report_path))

        return {
            "report_id": report_id,
            "path": f"/reports/{report_id}.html",
            "insights_count": len(insights),
        }

    @app.get("/api/session/{session_id}/history")
    async def get_history(session_id: str):
        """获取分析历史"""
        try:
            session = store.get_session(session_id)
        except KeyError:
            raise HTTPException(404, "会话不存在")
        return {"history": session["history"]}


# ============================================================
# 前端页面
# ============================================================

FRONTEND_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
&lt;head&gt;
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
&lt;title&gt;AI数据分析助手</title>
&lt;style&gt;
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
       background: #f0f2f5; }
.header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white; padding: 20px 40px; }
.header h1 { font-size: 24px; }
.main { display: grid; grid-template-columns: 320px 1fr;
        gap: 20px; padding: 20px; max-width: 1400px; margin: 0 auto; }

/* 左侧面板 */
.sidebar { background: white; border-radius: 12px; padding: 20px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.08); height: fit-content; }
.upload-zone { border: 2px dashed #ccc; border-radius: 8px; padding: 30px;
               text-align: center; cursor: pointer; transition: all .3s; }
.upload-zone:hover { border-color: #667eea; background: #f8f7ff; }
.upload-zone.active { border-color: #667eea; background: #f0edff; }
.data-info { margin-top: 16px; font-size: 13px; }
.data-info table { width: 100%; margin-top: 8px; }
.data-info th, .data-info td { padding: 4px 8px; text-align: left;
                                font-size: 12px; border-bottom: 1px solid #f0f0f0; }

/* 右侧内容 */
.content { display: flex; flex-direction: column; gap: 16px; }
.chat-box { background: white; border-radius: 12px; padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); flex: 1;
            min-height: 400px; display: flex; flex-direction: column; }
.messages { flex: 1; overflow-y: auto; padding: 10px 0; }
.msg { margin: 8px 0; padding: 12px 16px; border-radius: 12px;
       max-width: 85%; font-size: 14px; line-height: 1.6; }
.msg.user { background: #667eea; color: white; margin-left: auto; }
.msg.bot { background: #f5f5f5; }
.msg pre { background: #1e1e1e; color: #d4d4d4; padding: 12px;
           border-radius: 6px; margin: 8px 0; overflow-x: auto;
           font-size: 13px; }
.msg .result { background: #e8f5e9; padding: 10px; border-radius: 6px;
               margin: 8px 0; font-family: monospace; font-size: 13px;
               white-space: pre-wrap; }
.msg .error { background: #ffebee; color: #c62828; padding: 10px;
              border-radius: 6px; margin: 8px 0; }
.input-area { display: flex; gap: 8px; padding-top: 12px;
              border-top: 1px solid #eee; }
.input-area input { flex: 1; padding: 12px 16px; border: 1px solid #ddd;
                    border-radius: 8px; font-size: 14px; outline: none; }
.input-area input:focus { border-color: #667eea; }
.btn { padding: 10px 20px; background: #667eea; color: white;
       border: none; border-radius: 8px; cursor: pointer;
       font-size: 14px; transition: all .2s; }
.btn:hover { background: #5a6fd6; }
.btn:disabled { background: #ccc; cursor: not-allowed; }
.btn-outline { background: white; color: #667eea; border: 1px solid #667eea; }
.btn-outline:hover { background: #f0edff; }

.quick-btns { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0; }
.quick-btn { padding: 6px 12px; background: #f0edff; color: #667eea;
             border: 1px solid #e0d8ff; border-radius: 16px;
             font-size: 12px; cursor: pointer; }
.quick-btn:hover { background: #667eea; color: white; }
</style>
</head>
&lt;body&gt;

<div class="header">
    <h1>AI 数据分析助手</h1>
    <p style="opacity:0.8; margin-top:4px; font-size:14px">
        上传数据，用自然语言进行分析</p>
</div>

<div class="main">
    <div class="sidebar">
        <div class="upload-zone" id="uploadZone" onclick="fileInput.click()">
            <div style="font-size:36px; margin-bottom:8px">📊</div>
            &lt;p&gt;点击或拖拽上传数据文件</p>
            <p style="color:#999; font-size:12px; margin-top:4px">
                支持 CSV, Excel, JSON, Parquet</p>
            <input type="file" id="fileInput"
                   accept=".csv,.xlsx,.xls,.json,.parquet"
                   style="display:none" onchange="uploadFile(this)">
        </div>
        <div class="data-info" id="dataInfo" style="display:none">
            <h3 id="fileName"></h3>
            <p id="dataShape"></p>
            <table id="colTable">&lt;thead&gt;&lt;tr&gt;
                &lt;th&gt;列名</th>&lt;th&gt;类型</th>&lt;th&gt;非空</th>
            </tr></thead>&lt;tbody&gt;</tbody></table>
        </div>

        <div style="margin-top:16px">
            <button class="btn btn-outline" style="width:100%;margin-top:8px"
                    onclick="generateReport()" id="reportBtn" disabled>
                生成分析报告
            </button>
        </div>
    </div>

    <div class="content">
        <div class="chat-box">
            <div class="messages" id="messages">
                <div class="msg bot">
                    欢迎使用AI数据分析助手！请先上传数据文件，
                    然后用自然语言描述您的分析需求。
                </div>
            </div>
            <div class="quick-btns" id="quickBtns" style="display:none">
                <span class="quick-btn" onclick="askQuestion('数据的基本统计信息')">
                    基本统计</span>
                <span class="quick-btn" onclick="askQuestion('各类别的平均值对比')">
                    分组对比</span>
                <span class="quick-btn" onclick="askQuestion('数据的变化趋势')">
                    趋势分析</span>
                <span class="quick-btn" onclick="askQuestion('各列之间的相关性')">
                    相关性分析</span>
                <span class="quick-btn" onclick="askQuestion('是否存在异常值')">
                    异常检测</span>
            </div>
            <div class="input-area">
                <input type="text" id="queryInput"
                       placeholder="输入分析问题，如：各产品的平均销量是多少？"
                       onkeypress="if(event.key==='Enter')askQuestion()"
                       disabled>
                <button class="btn" onclick="askQuestion()" id="askBtn" disabled>
                    分析
                </button>
            </div>
        </div>
    </div>
</div>

&lt;script&gt;
let sessionId = null;

async function uploadFile(input) {
    const file = input.files[0];
    if (!file) return;

    const zone = document.getElementById('uploadZone');
    zone.innerHTML = '&lt;p&gt;上传中...</p>';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const resp = await fetch('/api/upload', {method:'POST', body:formData});
        const data = await resp.json();

        if (!resp.ok) throw new Error(data.detail || '上传失败');

        sessionId = data.session_id;

        // 显示数据信息
        document.getElementById('fileName').textContent = data.filename;
        document.getElementById('dataShape').textContent =
            `${data.rows.toLocaleString()} 行 x ${data.columns} 列`;

        const tbody = document.querySelector('#colTable tbody');
        tbody.innerHTML = '';
        data.column_info.forEach(col => {
            tbody.innerHTML += `&lt;tr&gt;&lt;td&gt;${col.name}</td>
                &lt;td&gt;${col.type}</td>&lt;td&gt;${col.non_null}</td></tr>`;
        });

        document.getElementById('dataInfo').style.display = 'block';
        document.getElementById('quickBtns').style.display = 'flex';
        document.getElementById('queryInput').disabled = false;
        document.getElementById('askBtn').disabled = false;
        document.getElementById('reportBtn').disabled = false;

        zone.innerHTML = `&lt;p&gt;已加载: ${data.filename}</p>
            <p style="font-size:12px;color:#666">点击重新上传</p>
            <input type="file" id="fileInput"
                   accept=".csv,.xlsx,.xls,.json,.parquet"
                   style="display:none" onchange="uploadFile(this)">`;
        zone.onclick = () => zone.querySelector('input').click();

        addMessage('bot', `数据加载成功！${data.filename}，`
            + `共 ${data.rows.toLocaleString()} 行 ${data.columns} 列。请输入分析问题。`);
    } catch(e) {
        zone.innerHTML = `<p style="color:red">上传失败: ${e.message}</p>`;
    }
}

async function askQuestion(text) {
    const input = document.getElementById('queryInput');
    const question = text || input.value.trim();
    if (!question || !sessionId) return;

    input.value = '';
    addMessage('user', question);

    const thinking = addMessage('bot', '分析中...');

    try {
        const resp = await fetch('/api/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({session_id: sessionId, question: question})
        });
        const data = await resp.json();

        let html = '';
        if (data.code) {
            html += `&lt;pre&gt;${escapeHtml(data.code)}</pre>`;
        }
        if (data.success) {
            html += `<div class="result">${escapeHtml(data.output)}</div>`;
            html += `<small style="color:#888">耗时: ${data.execution_time}s</small>`;
        } else {
            html += `<div class="error">错误: ${escapeHtml(data.error)}</div>`;
        }

        thinking.innerHTML = html;
    } catch(e) {
        thinking.innerHTML = `<div class="error">请求失败: ${e.message}</div>`;
    }
}

async function generateReport() {
    if (!sessionId) return;
    addMessage('bot', '正在生成报告...');
    try {
        const formData = new FormData();
        formData.append('session_id', sessionId);
        formData.append('title', '数据分析报告');
        const resp = await fetch('/api/report', {method:'POST', body:formData});
        const data = await resp.json();
        addMessage('bot',
            `报告已生成！<a href="${data.path}" target="_blank">点击查看</a>`
            + ` (${data.insights_count} 个洞察)`);
    } catch(e) {
        addMessage('bot', `报告生成失败: ${e.message}`);
    }
}

function addMessage(role, text) {
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.innerHTML = text;
    document.getElementById('messages').appendChild(div);
    div.scrollIntoView({behavior: 'smooth'});
    return div;
}

function escapeHtml(s) {
    if (!s) return '';
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;')
            .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>
</body>
</html>"""


if HAS_FASTAPI:
    @app.get("/", response_class=HTMLResponse)
    async def index():
        return FRONTEND_HTML

    # 静态文件
    app.mount("/charts", StaticFiles(directory="charts"), name="charts")
    app.mount("/reports", StaticFiles(directory="reports"), name="reports")


# ============================================================
# 启动脚本
# ============================================================

def main():
    """启动完整服务"""
    if not HAS_FASTAPI:
        print("请先安装依赖: pip install fastapi uvicorn python-multipart")
        print("\n模拟运行核心功能...")
        demo_full_pipeline()
        return

    import uvicorn
    print("启动 AI数据分析助手...")
    print("访问: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


def demo_full_pipeline():
    """完整流程演示 (无需FastAPI)"""
    print("=" * 60)
    print("AI数据分析助手 - 完整流程演示")
    print("=" * 60)

    # 1. 创建数据
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "日期": dates,
        "产品": np.random.choice(["手机", "电脑", "平板", "耳机", "音箱"], n),
        "地区": np.random.choice(["华东", "华南", "华北", "西南", "西北"], n),
        "渠道": np.random.choice(["线上", "线下", "直播"], n),
        "销量": np.random.randint(10, 500, n),
        "单价": np.random.uniform(100, 8000, n).round(2),
        "成本": np.random.uniform(50, 4000, n).round(2),
        "满意度": np.random.uniform(2.5, 5.0, n).round(1),
    })
    df["利润"] = ((df["单价"] - df["成本"]) * df["销量"]).round(2)
    df.loc[np.random.choice(n, 15), "满意度"] = np.nan

    print(f"\n数据: {df.shape[0]} 行 x {df.shape[1]} 列")

    # 2. 数据概况
    profiler = DataProfiler()
    profile = profiler.profile(df, "sales_demo.csv")
    print(f"\n--- 数据概况 ---")
    print(profile.to_prompt_context())

    # 3. NL2Code分析
    config = AnalysisConfig()
    engine = NL2CodeEngine(config)

    questions = [
        "各产品的总销量排名",
        "各地区的平均利润对比",
        "销量和满意度的关系",
    ]

    print(f"\n--- NL2Code分析 ---")
    for q in questions:
        result = engine.execute_analysis(q, df, profile)
        print(f"\n问: {q}")
        print(f"代码: {result.generated_code[:80]}...")
        if result.success:
            output = str(result.execution_output)
            if len(output) > 200:
                output = output[:200] + "..."
            print(f"结果: {output}")
        else:
            print(f"错误: {result.error}")

    # 4. 可视化推荐
    recommender = ChartRecommender()
    print(f"\n--- 可视化推荐 ---")
    recs = recommender.recommend(df, x_col="产品", y_col="销量")
    for r in recs:
        print(f"  {r.chart_type.value}: {r.title} (分数={r.score:.2f})")

    # 5. 数据洞察
    extractor = InsightExtractor()
    insights = extractor.extract_all(df)
    print(f"\n--- 数据洞察 ({len(insights)}个) ---")
    for ins in insights:
        print(f"  [{ins['level']}] {ins['text']}")

    # 6. 生成报告
    generator = ReportGenerator()
    html = generator.generate(df, profile, insights,
                              title="销售数据综合分析报告")
    os.makedirs("reports", exist_ok=True)
    generator.save_report(html, "reports/demo_report.html")
    print(f"\n--- 报告 ---")
    print(f"HTML报告: reports/demo_report.html ({len(html):,} 字节)")

    print(f"\n{'='*60}")
    print("演示完成! 启动完整服务: python 04_data_analysis.py")
    print("依赖安装: pip install fastapi uvicorn pandas numpy")
    print("          pip install matplotlib seaborn openai")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
```

---

## 总结

本教程涵盖了AI数据分析助手的核心内容:

1. **项目概述**: 系统架构设计、配置管理、数据模型定义、多格式数据加载与自动概况分析
2. **代码生成**: NL2Code引擎、安全沙箱执行、LLM提示工程、代码提取与清洗
3. **可视化建议**: 基于规则的图表推荐、数据特征分析、Matplotlib/Seaborn图表渲染
4. **自动化报告**: 数据洞察提取(异常值/趋势/相关性)、Jinja2模板渲染、HTML报告生成
5. **Notebook集成**: IPython Magic Commands、DataFrame自动发现、交互式分析
6. **完整项目**: FastAPI后端服务、会话管理、前端Web界面、一键部署

## 最佳实践

1. **安全第一**: 始终使用沙箱执行LLM生成的代码，限制import和危险函数调用
2. **上下文注入**: 将数据集的列名、类型、统计摘要注入到LLM提示中，显著提高代码生成准确率
3. **优雅降级**: 关键依赖(OpenAI API、matplotlib)不可用时提供mock/模拟实现
4. **增量分析**: 保持会话状态，支持用户连续提问，上下文递进分析
5. **性能优化**: 大数据集采样预览，缓存数据概况，避免重复计算

## 参考资源

- [Pandas官方文档](https://pandas.pydata.org/docs/)
- [Matplotlib可视化库](https://matplotlib.org/)
- [Seaborn统计可视化](https://seaborn.pydata.org/)
- [FastAPI Web框架](https://fastapi.tiangolo.com/)
- [OpenAI API文档](https://platform.openai.com/docs/api-reference)
- [Jupyter IPython扩展](https://ipython.readthedocs.io/en/stable/config/custommagics.html)
- [Jinja2模板引擎](https://jinja.palletsprojects.com/)

---

**文件大小目标**: 35-40KB
**创建时间**: 2025-01-01
**最后更新**: 2025-01-01
