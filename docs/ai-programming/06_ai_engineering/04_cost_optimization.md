# 成本优化策略

## 目录
1. [成本分析](#成本分析)
2. [Token优化](#token优化)
3. [语义缓存](#语义缓存)
4. [模型路由](#模型路由)
5. [Batch处理](#batch处理)
6. [ROI计算](#roi计算)

---

## 成本分析

### LLM成本全景

大语言模型在生产环境中的成本主要来自API调用费用(Token计费)、GPU算力费用(自部署)、
以及围绕模型运行的基础设施开销。理解成本结构是优化的第一步。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     LLM应用成本全景图                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  直接成本                                                                   │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐      │
│  │  API调用费用       │  │  GPU算力费用       │  │  存储费用         │      │
│  │                   │  │                   │  │                   │      │
│  │  输入Token: $X/1M │  │  A100: ~$2/h      │  │  向量数据库       │      │
│  │  输出Token: $Y/1M │  │  H100: ~$4/h      │  │  对话历史         │      │
│  │  Embedding: $Z/1M │  │  推理实例          │  │  模型工件         │      │
│  └───────┬───────────┘  └───────┬───────────┘  └───────┬───────────┘      │
│          │                      │                      │                   │
│          └──────────────────────┼──────────────────────┘                   │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       总成本 = 直接 + 间接                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                 ▲                                          │
│          ┌──────────────────────┼──────────────────────┐                   │
│          │                      │                      │                   │
│  ┌───────┴───────────┐  ┌──────┴────────────┐  ┌──────┴────────────┐      │
│  │  网络带宽         │  │  运维人力         │  │  监控告警         │      │
│  │                   │  │                   │  │                   │      │
│  │  数据传输费       │  │  工程师工时       │  │  Prometheus/      │      │
│  │  CDN费用          │  │  值班支持         │  │  Grafana等        │      │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘      │
│  间接成本                                                                   │
│                                                                             │
│  主流模型API定价 (每百万Token, 2024):                                      │
│  ┌──────────────┬───────────┬───────────┬────────────┐                    │
│  │ 模型         │ 输入价格  │ 输出价格  │ 性价比     │                    │
│  ├──────────────┼───────────┼───────────┼────────────┤                    │
│  │ GPT-4o       │ $2.50     │ $10.00    │ 高质量     │                    │
│  │ GPT-4o-mini  │ $0.15     │ $0.60     │ 极高性价比 │                    │
│  │ Claude Sonnet│ $3.00     │ $15.00    │ 高质量     │                    │
│  │ Claude Haiku │ $0.25     │ $1.25     │ 高性价比   │                    │
│  │ Llama3-70B   │ 自部署    │ 自部署    │ GPU成本    │                    │
│  │ Qwen2-72B   │ 自部署    │ 自部署    │ GPU成本    │                    │
│  └──────────────┴───────────┴───────────┴────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 成本分析框架

```python
"""
LLM成本分析框架
包含: 成本计算、使用统计、预算预警、成本报告
"""

import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ============================================================
# 第一部分: 模型定价配置
# ============================================================

@dataclass
class ModelPricing:
    """模型定价信息"""
    model_name: str
    input_price_per_1m: float   # 输入Token价格 ($/1M tokens)
    output_price_per_1m: float  # 输出Token价格 ($/1M tokens)
    provider: str = ""
    quality_tier: str = ""      # high / medium / low
    max_context: int = 128000

    @property
    def input_price_per_token(self) -> float:
        return self.input_price_per_1m / 1_000_000

    @property
    def output_price_per_token(self) -> float:
        return self.output_price_per_1m / 1_000_000


# 主流模型定价表
MODEL_PRICING = {
    "gpt-4o": ModelPricing("gpt-4o", 2.50, 10.00, "OpenAI", "high"),
    "gpt-4o-mini": ModelPricing("gpt-4o-mini", 0.15, 0.60, "OpenAI", "medium"),
    "gpt-3.5-turbo": ModelPricing("gpt-3.5-turbo", 0.50, 1.50, "OpenAI", "low"),
    "claude-sonnet": ModelPricing("claude-sonnet", 3.00, 15.00, "Anthropic", "high"),
    "claude-haiku": ModelPricing("claude-haiku", 0.25, 1.25, "Anthropic", "medium"),
    "deepseek-v3": ModelPricing("deepseek-v3", 0.27, 1.10, "DeepSeek", "medium"),
}


@dataclass
class UsageRecord:
    """单次调用记录"""
    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    task_type: str = ""
    cached: bool = False
    latency_ms: float = 0


class CostAnalyzer:
    """
    LLM成本分析器
    跟踪每次API调用, 生成成本报告和预算预警
    """

    def __init__(self, daily_budget: float = 100.0):
        self.records: List[UsageRecord] = []
        self.daily_budget = daily_budget
        self.pricing = MODEL_PRICING.copy()

    def record_usage(self, model: str, input_tokens: int,
                     output_tokens: int, task_type: str = "",
                     cached: bool = False, latency_ms: float = 0):
        """记录一次API调用"""
        pricing = self.pricing.get(model)
        if not pricing:
            print(f"[警告] 未知模型: {model}, 使用默认定价")
            pricing = ModelPricing(model, 1.0, 3.0)

        cost = (input_tokens * pricing.input_price_per_token +
                output_tokens * pricing.output_price_per_token)

        record = UsageRecord(
            timestamp=time.time(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            task_type=task_type,
            cached=cached,
            latency_ms=latency_ms,
        )
        self.records.append(record)

        # 预算预警
        today_cost = self._get_today_cost()
        if today_cost > self.daily_budget * 0.8:
            pct = today_cost / self.daily_budget * 100
            print(f"[预算预警] 今日已消费 ${today_cost:.4f} "
                  f"({pct:.1f}% of ${self.daily_budget})")

        return cost

    def _get_today_cost(self) -> float:
        """获取今日总成本"""
        today_start = datetime.now().replace(
            hour=0, minute=0, second=0
        ).timestamp()
        return sum(
            r.cost for r in self.records if r.timestamp >= today_start
        )

    def get_cost_by_model(self, hours: int = 24) -> Dict[str, float]:
        """按模型分组统计成本"""
        cutoff = time.time() - hours * 3600
        costs = defaultdict(float)
        for r in self.records:
            if r.timestamp >= cutoff:
                costs[r.model] += r.cost
        return dict(costs)

    def get_cost_by_task(self, hours: int = 24) -> Dict[str, float]:
        """按任务类型分组统计成本"""
        cutoff = time.time() - hours * 3600
        costs = defaultdict(float)
        for r in self.records:
            if r.timestamp >= cutoff:
                task = r.task_type or "unknown"
                costs[task] += r.cost
        return dict(costs)

    def get_token_stats(self, hours: int = 24) -> Dict:
        """获取Token使用统计"""
        cutoff = time.time() - hours * 3600
        recent = [r for r in self.records if r.timestamp >= cutoff]

        if not recent:
            return {"total_requests": 0}

        total_input = sum(r.input_tokens for r in recent)
        total_output = sum(r.output_tokens for r in recent)
        total_cost = sum(r.cost for r in recent)
        cached_count = sum(1 for r in recent if r.cached)

        return {
            "total_requests": len(recent),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost": round(total_cost, 4),
            "avg_cost_per_request": round(total_cost / len(recent), 6),
            "avg_input_tokens": total_input // len(recent),
            "avg_output_tokens": total_output // len(recent),
            "cache_hit_rate": round(cached_count / len(recent) * 100, 1),
        }

    def generate_report(self, hours: int = 24) -> str:
        """生成成本分析报告"""
        stats = self.get_token_stats(hours)
        model_costs = self.get_cost_by_model(hours)
        task_costs = self.get_cost_by_task(hours)

        lines = [
            f"{'=' * 60}",
            f"LLM成本分析报告 (最近 {hours} 小时)",
            f"{'=' * 60}",
            "",
            f"总请求数:        {stats.get('total_requests', 0):,}",
            f"总输入Token:     {stats.get('total_input_tokens', 0):,}",
            f"总输出Token:     {stats.get('total_output_tokens', 0):,}",
            f"总Token:         {stats.get('total_tokens', 0):,}",
            f"总成本:          ${stats.get('total_cost', 0):.4f}",
            f"平均每请求成本:  ${stats.get('avg_cost_per_request', 0):.6f}",
            f"缓存命中率:      {stats.get('cache_hit_rate', 0):.1f}%",
            "",
            "按模型分组:",
        ]
        for model, cost in sorted(model_costs.items(),
                                   key=lambda x: -x[1]):
            lines.append(f"  {model:20s} ${cost:.4f}")

        lines.append("")
        lines.append("按任务分组:")
        for task, cost in sorted(task_costs.items(),
                                  key=lambda x: -x[1]):
            lines.append(f"  {task:20s} ${cost:.4f}")

        # 月度预估
        if stats.get("total_cost", 0) > 0:
            daily_rate = stats["total_cost"] / (hours / 24)
            monthly_est = daily_rate * 30
            lines.extend([
                "",
                f"日均成本预估:    ${daily_rate:.2f}",
                f"月度成本预估:    ${monthly_est:.2f}",
            ])

        return "\n".join(lines)


# ============================================================
# 演示: 成本分析
# ============================================================

def demo_cost_analysis():
    """演示成本分析功能"""
    analyzer = CostAnalyzer(daily_budget=50.0)

    # 模拟一批API调用
    import random
    tasks = ["chat", "summarize", "translate", "code_gen", "qa"]
    models = ["gpt-4o", "gpt-4o-mini", "claude-haiku"]

    for i in range(50):
        model = random.choice(models)
        task = random.choice(tasks)
        inp = random.randint(100, 3000)
        out = random.randint(50, 1500)
        cached = random.random() < 0.2
        analyzer.record_usage(
            model=model,
            input_tokens=inp,
            output_tokens=out,
            task_type=task,
            cached=cached,
            latency_ms=random.uniform(200, 2000),
        )

    print(analyzer.generate_report(hours=24))


if __name__ == "__main__":
    demo_cost_analysis()
```

---

## Token优化

### Token优化策略全景

Token是LLM的计费单位, 也是影响延迟的核心因素。减少不必要的Token消耗,
既能降低成本, 也能提升响应速度。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Token优化策略全景图                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  用户输入                                                                   │
│  ┌──────────────────────┐                                                  │
│  │ "请帮我把以下这段    │                                                  │
│  │  很长很长的文本...   │                                                  │
│  │  进行总结和提炼"     │  原始Token: ~5000                                │
│  └──────────┬───────────┘                                                  │
│             │                                                              │
│             ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Token优化 Pipeline                               │   │
│  │                                                                     │   │
│  │  Stage 1          Stage 2          Stage 3          Stage 4         │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │   │
│  │  │ Prompt     │  │ 上下文     │  │ 输出       │  │ 压缩       │   │   │
│  │  │ 工程优化   │  │ 窗口管理   │  │ 长度控制   │  │ 与摘要     │   │   │
│  │  │            │  │            │  │            │  │            │   │   │
│  │  │ -精简指令  │  │ -滑动窗口  │  │ -max_tokens│  │ -长文本    │   │   │
│  │  │ -模板复用  │  │ -对话裁剪  │  │ -stop序列  │  │  预处理    │   │   │
│  │  │ -Few-shot  │  │ -关键信息  │  │ -结构化    │  │ -分块处理  │   │   │
│  │  │  精选      │  │  提取      │  │  输出格式  │  │ -摘要缓存  │   │   │
│  │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘   │   │
│  │        │  -30%         │  -40%         │  -20%         │  -50%    │   │
│  │        └───────────────┴───────────────┴───────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│             │                                                              │
│             ▼                                                              │
│  ┌──────────────────────┐                                                  │
│  │  优化后Token: ~1500  │  节省约 70%                                      │
│  │  成本: $0.003→$0.001 │                                                  │
│  └──────────────────────┘                                                  │
│                                                                             │
│  关键指标:                                                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐     │
│  │ Token压缩率  │ │ 质量保持率   │ │ 延迟降低     │ │ 成本节省     │     │
│  │ 50-70%       │ │ >95%         │ │ 40-60%       │ │ 50-70%       │     │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Prompt优化器

```python
"""
Token优化工具集
包含: Prompt压缩、上下文管理、输出控制、Token计数
"""

import re
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================
# 第一部分: Token计数与估算
# ============================================================

class TokenEstimator:
    """
    Token数量估算器
    使用简化规则估算Token数(生产环境建议用tiktoken)

    规则 (近似):
      - 英文: 约1 token = 4字符 / 0.75个单词
      - 中文: 约1 token = 1-2个汉字
      - 代码: 约1 token = 3字符
    """

    # 常见模型的上下文窗口大小
    CONTEXT_WINDOWS = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-3.5-turbo": 16385,
        "claude-sonnet": 200000,
        "claude-haiku": 200000,
        "llama3-8b": 8192,
        "llama3-70b": 8192,
        "qwen2-72b": 131072,
    }

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """估算文本的Token数量"""
        if not text:
            return 0

        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        non_chinese = len(text) - chinese_chars

        # 中文约1.5字符/token, 英文约4字符/token
        tokens = chinese_chars / 1.5 + non_chinese / 4
        return max(1, int(tokens))

    @staticmethod
    def estimate_messages_tokens(messages: List[Dict]) -> int:
        """估算消息列表的Token数量"""
        total = 0
        for msg in messages:
            # 每条消息有额外开销 (~4 tokens)
            total += 4
            total += TokenEstimator.estimate_tokens(msg.get("role", ""))
            total += TokenEstimator.estimate_tokens(msg.get("content", ""))
        total += 2  # 会话开始/结束标记
        return total

    @staticmethod
    def estimate_cost(input_tokens: int, output_tokens: int,
                      model: str = "gpt-4o-mini") -> float:
        """估算调用成本"""
        pricing = {
            "gpt-4o": (2.50, 10.00),
            "gpt-4o-mini": (0.15, 0.60),
            "claude-sonnet": (3.00, 15.00),
            "claude-haiku": (0.25, 1.25),
        }
        inp_price, out_price = pricing.get(model, (1.0, 3.0))
        return (input_tokens * inp_price + output_tokens * out_price) / 1_000_000


# ============================================================
# 第二部分: Prompt压缩与优化
# ============================================================

class PromptOptimizer:
    """
    Prompt优化器
    通过多种策略压缩Prompt, 减少Token消耗同时保持质量
    """

    # 常见冗余短语映射 (长 -> 短)
    REDUNDANCY_MAP = {
        "请你帮我": "",
        "请帮我": "",
        "能不能帮我": "",
        "我想让你": "",
        "我希望你能": "",
        "以下是": "",
        "如下所示": "",
        "请注意": "注意:",
        "需要注意的是": "注意:",
        "首先,": "1.",
        "然后,": "2.",
        "最后,": "3.",
        "换句话说": "即",
        "也就是说": "即",
        "总而言之": "总之",
        "毫无疑问": "",
        "众所周知": "",
    }

    def __init__(self):
        self.estimator = TokenEstimator()

    def compress_prompt(self, prompt: str,
                        aggressive: bool = False) -> Tuple[str, Dict]:
        """
        压缩Prompt, 返回 (压缩后文本, 统计信息)
        """
        original_tokens = self.estimator.estimate_tokens(prompt)
        result = prompt

        # 策略1: 去除冗余短语
        for long, short in self.REDUNDANCY_MAP.items():
            result = result.replace(long, short)

        # 策略2: 合并连续空白
        result = re.sub(r'\n{3,}', '\n\n', result)
        result = re.sub(r' {2,}', ' ', result)

        # 策略3: 去除行首尾空白
        lines = [line.strip() for line in result.split('\n')]
        result = '\n'.join(line for line in lines if line)

        if aggressive:
            # 策略4: 移除标点冗余
            result = re.sub(r'[。！？]{2,}', '。', result)
            result = re.sub(r'[,.]{2,}', ',', result)
            # 策略5: 缩短常见指令
            result = result.replace("请用中文回答", "中文回答")
            result = result.replace("请详细解释", "解释")
            result = result.replace("请简要说明", "说明")

        compressed_tokens = self.estimator.estimate_tokens(result)
        savings = original_tokens - compressed_tokens
        pct = (savings / original_tokens * 100) if original_tokens > 0 else 0

        stats = {
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "tokens_saved": savings,
            "compression_rate": round(pct, 1),
        }

        return result.strip(), stats

    def optimize_few_shot(self, examples: List[Dict[str, str]],
                          max_examples: int = 3) -> List[Dict[str, str]]:
        """
        优化Few-shot示例
        选择最短且最具代表性的示例
        """
        # 按输入+输出长度排序, 取最短的
        scored = []
        for ex in examples:
            total_len = (self.estimator.estimate_tokens(ex.get("input", "")) +
                         self.estimator.estimate_tokens(ex.get("output", "")))
            scored.append((total_len, ex))

        scored.sort(key=lambda x: x[0])
        selected = [ex for _, ex in scored[:max_examples]]

        original_tokens = sum(
            self.estimator.estimate_tokens(e.get("input", "")) +
            self.estimator.estimate_tokens(e.get("output", ""))
            for e in examples
        )
        selected_tokens = sum(
            self.estimator.estimate_tokens(e.get("input", "")) +
            self.estimator.estimate_tokens(e.get("output", ""))
            for e in selected
        )

        print(f"[Few-shot优化] {len(examples)}个示例 -> {len(selected)}个")
        print(f"  Token: {original_tokens} -> {selected_tokens} "
              f"(节省 {original_tokens - selected_tokens})")

        return selected


# ============================================================
# 第三部分: 对话上下文管理
# ============================================================

class ContextWindowManager:
    """
    对话上下文窗口管理器
    在保持对话连贯性的同时控制Token数量
    """

    def __init__(self, max_tokens: int = 4000,
                 strategy: str = "sliding_window"):
        """
        Args:
            max_tokens: 上下文最大Token数
            strategy: 裁剪策略
              - sliding_window: 滑动窗口, 保留最近N轮
              - summarize: 对早期对话做摘要
              - importance: 按重要性保留
        """
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.estimator = TokenEstimator()
        self.system_prompt: str = ""
        self.messages: List[Dict] = []
        self.summary: str = ""

    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self.system_prompt = prompt

    def add_message(self, role: str, content: str):
        """添加消息"""
        self.messages.append({"role": role, "content": content})

    def get_context(self) -> List[Dict]:
        """
        获取裁剪后的上下文, 确保不超过Token限制
        """
        context = []

        # 系统提示词始终保留
        if self.system_prompt:
            context.append({"role": "system", "content": self.system_prompt})

        # 如果有历史摘要, 加入上下文
        if self.summary:
            context.append({
                "role": "system",
                "content": f"对话历史摘要: {self.summary}"
            })

        # 计算已用Token
        used_tokens = self.estimator.estimate_messages_tokens(context)
        remaining = self.max_tokens - used_tokens

        if self.strategy == "sliding_window":
            context.extend(
                self._sliding_window_trim(remaining)
            )
        elif self.strategy == "importance":
            context.extend(
                self._importance_trim(remaining)
            )
        else:
            # 默认: 从后往前填充
            context.extend(
                self._sliding_window_trim(remaining)
            )

        return context

    def _sliding_window_trim(self, max_tokens: int) -> List[Dict]:
        """滑动窗口裁剪: 保留最近的消息"""
        result = []
        total = 0
        # 从后往前遍历
        for msg in reversed(self.messages):
            msg_tokens = self.estimator.estimate_tokens(msg["content"]) + 4
            if total + msg_tokens > max_tokens:
                break
            result.insert(0, msg)
            total += msg_tokens
        return result

    def _importance_trim(self, max_tokens: int) -> List[Dict]:
        """按重要性裁剪: 优先保留用户首条+最近消息"""
        if not self.messages:
            return []

        # 始终保留第一条用户消息 (定义任务)
        important = [self.messages[0]] if self.messages else []
        # 始终保留最近4条消息
        recent = self.messages[-4:] if len(self.messages) > 4 else self.messages[1:]

        combined = important + recent
        # 去重
        seen = set()
        deduped = []
        for msg in combined:
            key = msg["content"][:50]
            if key not in seen:
                seen.add(key)
                deduped.append(msg)

        # Token裁剪
        total = 0
        result = []
        for msg in deduped:
            t = self.estimator.estimate_tokens(msg["content"]) + 4
            if total + t > max_tokens:
                break
            result.append(msg)
            total += t
        return result

    def get_stats(self) -> Dict:
        """获取上下文统计"""
        context = self.get_context()
        total_tokens = self.estimator.estimate_messages_tokens(context)
        all_tokens = self.estimator.estimate_messages_tokens(self.messages)

        return {
            "total_messages": len(self.messages),
            "context_messages": len(context),
            "context_tokens": total_tokens,
            "full_tokens": all_tokens,
            "tokens_saved": all_tokens - total_tokens,
            "utilization": round(total_tokens / self.max_tokens * 100, 1),
        }


# ============================================================
# 第四部分: 输出长度控制
# ============================================================

class OutputController:
    """
    输出长度与格式控制器
    通过结构化输出和长度限制减少输出Token
    """

    TASK_TEMPLATES = {
        "classification": {
            "instruction": "分类任务, 只输出类别名称, 不需要解释。",
            "max_tokens": 10,
            "format": "单行文本",
        },
        "sentiment": {
            "instruction": "情感分析, 只输出: positive/negative/neutral",
            "max_tokens": 5,
            "format": "单词",
        },
        "extraction": {
            "instruction": "信息提取, 以JSON格式输出, 不加额外说明。",
            "max_tokens": 200,
            "format": "JSON",
        },
        "summary": {
            "instruction": "用不超过3句话总结要点。",
            "max_tokens": 150,
            "format": "短文本",
        },
        "qa": {
            "instruction": "简洁回答问题, 不超过2句话。",
            "max_tokens": 100,
            "format": "短文本",
        },
        "translation": {
            "instruction": "只输出翻译结果, 不加注释。",
            "max_tokens": None,  # 与原文等长
            "format": "文本",
        },
    }

    def get_optimized_prompt(self, task_type: str,
                             user_content: str) -> Tuple[str, int]:
        """
        根据任务类型生成优化的Prompt

        Returns:
            (优化后的prompt, 建议max_tokens)
        """
        template = self.TASK_TEMPLATES.get(task_type)
        if not template:
            return user_content, 500

        prompt = f"{template['instruction']}\n\n{user_content}"
        max_tokens = template["max_tokens"]

        if max_tokens is None:
            est = TokenEstimator.estimate_tokens(user_content)
            max_tokens = int(est * 1.2)

        return prompt, max_tokens


# ============================================================
# 综合演示
# ============================================================

def demo_token_optimization():
    """演示Token优化全流程"""

    print("=" * 60)
    print("Token优化演示")
    print("=" * 60)

    # 1. Prompt压缩
    print("\n--- 1. Prompt压缩 ---")
    optimizer = PromptOptimizer()

    long_prompt = """请你帮我分析以下这段文本的情感倾向。
    我希望你能仔细阅读以下内容, 然后给出你的判断。
    需要注意的是, 请用中文回答, 请详细解释你的理由。
    以下是需要分析的文本:

    这款产品的质量真的太差了！！！
    买回来第二天就坏了, 客服态度也很差。
    强烈建议大家不要购买！！！
    """

    compressed, stats = optimizer.compress_prompt(long_prompt, aggressive=True)
    print(f"原始Prompt: {stats['original_tokens']} tokens")
    print(f"压缩后:     {stats['compressed_tokens']} tokens")
    print(f"节省:       {stats['tokens_saved']} tokens ({stats['compression_rate']}%)")
    print(f"\n压缩结果:\n{compressed}")

    # 2. 上下文窗口管理
    print("\n\n--- 2. 上下文窗口管理 ---")
    ctx = ContextWindowManager(max_tokens=500, strategy="sliding_window")
    ctx.set_system_prompt("你是一个智能助手。")

    # 模拟多轮对话
    conversations = [
        ("user", "什么是机器学习?"),
        ("assistant", "机器学习是人工智能的一个子领域, 它使计算机能从数据中学习。"),
        ("user", "深度学习呢?"),
        ("assistant", "深度学习是机器学习的一个子集, 使用多层神经网络。"),
        ("user", "Transformer是什么?"),
        ("assistant", "Transformer是一种基于自注意力机制的神经网络架构。"),
        ("user", "GPT和BERT有什么区别?"),
        ("assistant", "GPT是自回归模型, BERT是双向编码器。"),
        ("user", "什么是大语言模型?"),
    ]

    for role, content in conversations:
        ctx.add_message(role, content)

    context_stats = ctx.get_stats()
    print(f"总消息数:    {context_stats['total_messages']}")
    print(f"上下文消息:  {context_stats['context_messages']}")
    print(f"上下文Token: {context_stats['context_tokens']}")
    print(f"节省Token:   {context_stats['tokens_saved']}")
    print(f"窗口利用率:  {context_stats['utilization']}%")

    # 3. 输出控制
    print("\n\n--- 3. 输出长度控制 ---")
    controller = OutputController()

    tasks = [
        ("sentiment", "这家餐厅的菜品非常美味, 服务也很周到"),
        ("classification", "苹果发布了新款iPhone"),
        ("summary", "人工智能正在改变各行各业..."),
    ]

    for task_type, content in tasks:
        prompt, max_tok = controller.get_optimized_prompt(task_type, content)
        print(f"\n任务: {task_type}")
        print(f"  建议max_tokens: {max_tok}")
        print(f"  优化后Prompt: {prompt[:80]}...")

    # 4. 成本对比
    print("\n\n--- 4. 优化前后成本对比 ---")
    est = TokenEstimator()

    before_input = 5000
    before_output = 2000
    after_input = 1500
    after_output = 500

    for model in ["gpt-4o", "gpt-4o-mini", "claude-haiku"]:
        cost_before = est.estimate_cost(before_input, before_output, model)
        cost_after = est.estimate_cost(after_input, after_output, model)
        saved = cost_before - cost_after
        pct = saved / cost_before * 100
        print(f"  {model:15s}: ${cost_before:.4f} -> ${cost_after:.4f} "
              f"(节省 ${saved:.4f}, {pct:.0f}%)")


if __name__ == "__main__":
    demo_token_optimization()
```

---

## 语义缓存

### 语义缓存架构

传统缓存依赖精确匹配(相同输入 = 缓存命中), 但用户提问方式多样,
语义相同但表述不同的问题无法命中。语义缓存通过向量相似度匹配, 将
"意思相同"的查询路由到已有答案, 大幅提升缓存命中率。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      语义缓存 架构图                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  用户查询                                                                   │
│  ┌──────────────────────┐                                                  │
│  │ "Python怎么读文件?"  │                                                  │
│  └──────────┬───────────┘                                                  │
│             │                                                              │
│             ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Embedding层                                                       │   │
│  │  text-embedding-3-small / BGE-M3 / ...                             │   │
│  │  "Python怎么读文件?" ──> [0.12, -0.34, 0.56, ...]  (1536维向量)   │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  相似度检索                                                         │   │
│  │                                                                     │   │
│  │  缓存条目:                                                          │   │
│  │  ┌──────────────────────────┬────────────┬────────────────────────┐ │   │
│  │  │ 缓存的查询               │ 相似度     │ 状态                   │ │   │
│  │  ├──────────────────────────┼────────────┼────────────────────────┤ │   │
│  │  │ "如何用Python读取文件"   │ 0.95       │ --> 命中! (>=0.90)     │ │   │
│  │  │ "Java文件读写操作"       │ 0.62       │     未命中             │ │   │
│  │  │ "Python列表排序"         │ 0.45       │     未命中             │ │   │
│  │  └──────────────────────────┴────────────┴────────────────────────┘ │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                          │
│             ┌───────────────────┼───────────────────┐                      │
│             │ 命中              │                    │ 未命中              │
│             ▼                   │                    ▼                     │
│  ┌──────────────────┐           │         ┌──────────────────┐            │
│  │  返回缓存答案    │           │         │  调用LLM API     │            │
│  │  延迟: <10ms     │           │         │  延迟: 500-3000ms│            │
│  │  成本: $0        │           │         │  成本: $0.001+   │            │
│  └──────────────────┘           │         └────────┬─────────┘            │
│                                 │                  │                      │
│                                 │                  ▼                      │
│                                 │         ┌──────────────────┐            │
│                                 │         │  写入缓存        │            │
│                                 │         │  query_vec + 答案 │            │
│                                 │         └──────────────────┘            │
│                                 │                                          │
│  效果指标:                                                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐     │
│  │ 命中率       │ │ 延迟降低     │ │ 成本节省     │ │ 答案准确率   │     │
│  │ 30-60%       │ │ 95%+ (命中时)│ │ 30-60%       │ │ 需要>90%     │     │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 语义缓存实现

```python
"""
语义缓存完整实现
包含: 向量相似度缓存、TTL过期、LRU淘汰、命中率统计
"""

import time
import math
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict


# ============================================================
# 第一部分: 向量运算工具 (纯Python, 不依赖numpy)
# ============================================================

class VectorOps:
    """轻量级向量运算 (不依赖外部库)"""

    @staticmethod
    def cosine_similarity(vec_a: List[float],
                          vec_b: List[float]) -> float:
        """计算余弦相似度"""
        if len(vec_a) != len(vec_b):
            raise ValueError("向量维度不一致")
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def simple_text_embedding(text: str, dim: int = 64) -> List[float]:
        """
        简化文本嵌入 (仅用于演示, 生产环境请用专业Embedding模型)

        原理: 基于字符n-gram的哈希向量化
        - 将文本拆成字符级2-gram和3-gram
        - 对每个n-gram计算哈希, 映射到向量维度
        - 归一化为单位向量
        """
        vector = [0.0] * dim
        text = text.lower().strip()

        if not text:
            return vector

        # 2-gram和3-gram特征
        ngrams = []
        for n in (2, 3):
            for i in range(len(text) - n + 1):
                ngrams.append(text[i:i + n])

        # 词级特征
        words = text.split()
        ngrams.extend(words)

        # 哈希映射到向量
        for ng in ngrams:
            h = int(hashlib.md5(ng.encode()).hexdigest(), 16)
            idx = h % dim
            sign = 1 if (h // dim) % 2 == 0 else -1
            vector[idx] += sign

        # L2归一化
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector


# ============================================================
# 第二部分: 语义缓存核心
# ============================================================

@dataclass
class CacheEntry:
    """缓存条目"""
    query: str
    query_vector: List[float]
    response: str
    model: str
    created_at: float
    last_accessed: float
    access_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class SemanticCache:
    """
    语义缓存
    基于向量相似度的LLM响应缓存

    特性:
    - 语义匹配: 相似问题复用答案
    - TTL过期: 超时条目自动失效
    - LRU淘汰: 缓存满时淘汰最久未用条目
    - 命中统计: 跟踪命中率和节省成本
    """

    def __init__(self, max_size: int = 10000,
                 similarity_threshold: float = 0.90,
                 ttl_seconds: int = 3600,
                 embedding_dim: int = 64):
        """
        Args:
            max_size: 最大缓存条目数
            similarity_threshold: 相似度阈值 (0-1)
            ttl_seconds: 缓存过期时间 (秒)
            embedding_dim: 嵌入向量维度
        """
        self.max_size = max_size
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        self.dim = embedding_dim

        # 有序字典实现LRU
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # 统计
        self.stats = {
            "hits": 0, "misses": 0,
            "tokens_saved": 0, "cost_saved": 0.0,
        }

    def _get_embedding(self, text: str) -> List[float]:
        """获取文本嵌入向量 (生产环境替换为API调用)"""
        return VectorOps.simple_text_embedding(text, self.dim)

    def _evict_expired(self):
        """清除过期条目"""
        now = time.time()
        expired = [
            k for k, v in self.cache.items()
            if now - v.created_at > self.ttl
        ]
        for k in expired:
            del self.cache[k]

    def _evict_lru(self):
        """LRU淘汰: 移除最久未访问的条目"""
        while len(self.cache) >= self.max_size:
            # popitem(last=False) 移除最早插入的
            self.cache.popitem(last=False)

    def lookup(self, query: str) -> Optional[Tuple[str, float]]:
        """
        查找语义相似的缓存

        Returns:
            (缓存的响应, 相似度) 或 None
        """
        self._evict_expired()

        query_vec = self._get_embedding(query)
        best_match = None
        best_sim = 0.0

        for key, entry in self.cache.items():
            sim = VectorOps.cosine_similarity(query_vec, entry.query_vector)
            if sim > best_sim:
                best_sim = sim
                best_match = entry

        if best_match and best_sim >= self.threshold:
            # 命中: 更新访问信息
            best_match.last_accessed = time.time()
            best_match.access_count += 1
            # 移到末尾 (最近使用)
            key = hashlib.md5(best_match.query.encode()).hexdigest()
            if key in self.cache:
                self.cache.move_to_end(key)

            self.stats["hits"] += 1
            self.stats["tokens_saved"] += (best_match.input_tokens +
                                           best_match.output_tokens)
            return best_match.response, best_sim

        self.stats["misses"] += 1
        return None

    def store(self, query: str, response: str, model: str = "",
              input_tokens: int = 0, output_tokens: int = 0):
        """存储新的缓存条目"""
        self._evict_lru()

        query_vec = self._get_embedding(query)
        key = hashlib.md5(query.encode()).hexdigest()

        entry = CacheEntry(
            query=query,
            query_vector=query_vec,
            response=response,
            model=model,
            created_at=time.time(),
            last_accessed=time.time(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.cache[key] = entry

    def get_stats(self) -> Dict:
        """获取缓存统计"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0

        return {
            "total_queries": total,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": round(hit_rate, 1),
            "cache_size": len(self.cache),
            "tokens_saved": self.stats["tokens_saved"],
        }

    def print_stats(self):
        """打印缓存统计报告"""
        s = self.get_stats()
        print(f"\n{'=' * 50}")
        print("语义缓存统计报告")
        print(f"{'=' * 50}")
        print(f"总查询数:     {s['total_queries']}")
        print(f"缓存命中:     {s['hits']}")
        print(f"缓存未命中:   {s['misses']}")
        print(f"命中率:       {s['hit_rate']}%")
        print(f"缓存条目数:   {s['cache_size']}")
        print(f"节省Token:    {s['tokens_saved']:,}")


# ============================================================
# 第三部分: 带语义缓存的LLM客户端
# ============================================================

class CachedLLMClient:
    """
    带语义缓存的LLM调用客户端
    先查缓存, 未命中时调用LLM, 然后缓存结果
    """

    def __init__(self, cache: SemanticCache = None):
        self.cache = cache or SemanticCache()
        self.call_count = 0

    def _simulate_llm_call(self, query: str, model: str) -> Tuple[str, int, int]:
        """
        模拟LLM API调用 (演示用)
        实际项目中替换为openai.chat.completions.create()
        """
        self.call_count += 1
        # 模拟生成回答
        response = f"[{model}] 关于'{query[:20]}...'的回答: 这是一个模拟回答。"
        input_tokens = len(query) * 2
        output_tokens = len(response) * 2
        time.sleep(0.01)  # 模拟API延迟
        return response, input_tokens, output_tokens

    def query(self, query: str, model: str = "gpt-4o-mini") -> Dict:
        """
        智能查询: 优先使用缓存, 未命中时调用LLM
        """
        start = time.time()

        # 1. 查缓存
        cache_result = self.cache.lookup(query)
        if cache_result:
            response, similarity = cache_result
            elapsed = (time.time() - start) * 1000
            return {
                "response": response,
                "cached": True,
                "similarity": round(similarity, 3),
                "latency_ms": round(elapsed, 1),
                "cost": 0.0,
            }

        # 2. 未命中, 调用LLM
        response, in_tok, out_tok = self._simulate_llm_call(query, model)
        elapsed = (time.time() - start) * 1000

        # 3. 写入缓存
        self.cache.store(query, response, model, in_tok, out_tok)

        cost = (in_tok * 0.15 + out_tok * 0.60) / 1_000_000  # gpt-4o-mini价格

        return {
            "response": response,
            "cached": False,
            "similarity": 0.0,
            "latency_ms": round(elapsed, 1),
            "cost": round(cost, 6),
        }


# ============================================================
# 演示
# ============================================================

def demo_semantic_cache():
    """演示语义缓存效果"""

    print("=" * 60)
    print("语义缓存演示")
    print("=" * 60)

    client = CachedLLMClient(
        SemanticCache(similarity_threshold=0.85, ttl_seconds=600)
    )

    # 测试查询 (包含语义相似的问题)
    queries = [
        "Python怎么读取文件?",             # 首次查询 (miss)
        "如何用Python读取文件内容",          # 语义相似 (可能hit)
        "Python文件读取方法",               # 语义相似 (可能hit)
        "Java怎么读取文件?",               # 不同语言 (miss)
        "什么是机器学习?",                  # 新主题 (miss)
        "机器学习的定义是什么",              # 语义相似 (可能hit)
        "深度学习和机器学习的区别",           # 相关但不同 (miss)
        "Python怎么写入文件?",             # 相关但不同操作 (miss)
        "如何用Python读文件",              # 语义相似 (可能hit)
    ]

    for i, q in enumerate(queries, 1):
        result = client.query(q)
        status = "HIT " if result["cached"] else "MISS"
        sim = f"sim={result['similarity']:.2f}" if result["cached"] else ""
        print(f"  [{i}] {status} {sim:12s} | {q}")

    # 打印统计
    client.cache.print_stats()
    print(f"\n实际LLM调用次数: {client.call_count}")
    print(f"总查询次数:       {len(queries)}")
    print(f"节省调用比例:     "
          f"{(1 - client.call_count / len(queries)) * 100:.0f}%")


if __name__ == "__main__":
    demo_semantic_cache()
```

---

## 模型路由

### 智能模型路由架构

不同任务需要不同能力级别的模型。简单任务(情感分类、翻译)使用小模型即可,
复杂任务(推理、代码生成)需要大模型。智能路由器根据任务特征自动选择最优模型,
在保证质量的前提下大幅降低成本。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      智能模型路由架构                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  用户请求                                                                   │
│  ┌──────────────────────┐                                                  │
│  │ query + task_type    │                                                  │
│  └──────────┬───────────┘                                                  │
│             │                                                              │
│             ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     路由分析器                                       │   │
│  │                                                                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │   │
│  │  │ 复杂度评估  │  │ 任务分类    │  │ 质量要求    │                │   │
│  │  │             │  │             │  │             │                │   │
│  │  │ - 文本长度  │  │ - 分类      │  │ - 高精度    │                │   │
│  │  │ - 推理深度  │  │ - 生成      │  │ - 一般      │                │   │
│  │  │ - 领域难度  │  │ - 推理      │  │ - 快速响应  │                │   │
│  │  │ - 上下文量  │  │ - 代码      │  │ - 成本优先  │                │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                │   │
│  │         └────────────────┼────────────────┘                        │   │
│  │                          ▼                                         │   │
│  │              ┌───────────────────────┐                             │   │
│  │              │  路由决策引擎          │                             │   │
│  │              │  score = f(复杂度,     │                             │   │
│  │              │    任务类型, 质量要求) │                             │   │
│  │              └───────────┬───────────┘                             │   │
│  └──────────────────────────┼──────────────────────────────────────────┘   │
│                             │                                              │
│          ┌──────────────────┼──────────────────┐                           │
│          │                  │                  │                            │
│          ▼                  ▼                  ▼                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                     │
│  │  Tier 1      │  │  Tier 2      │  │  Tier 3      │                     │
│  │  大模型      │  │  中模型      │  │  小模型      │                     │
│  │              │  │              │  │              │                     │
│  │  GPT-4o      │  │  GPT-4o-mini │  │  GPT-3.5     │                     │
│  │  Claude      │  │  Claude Haiku│  │  本地 7B     │                     │
│  │  Sonnet      │  │  DeepSeek-V3 │  │  Qwen2-7B   │                     │
│  │              │  │              │  │              │                     │
│  │  适用:       │  │  适用:       │  │  适用:       │                     │
│  │  - 复杂推理  │  │  - 通用问答  │  │  - 简单分类  │                     │
│  │  - 代码生成  │  │  - 内容创作  │  │  - 情感分析  │                     │
│  │  - 专家分析  │  │  - 翻译总结  │  │  - 信息提取  │                     │
│  │              │  │              │  │              │                     │
│  │  成本: $$$   │  │  成本: $$    │  │  成本: $     │                     │
│  │  质量: ★★★  │  │  质量: ★★   │  │  质量: ★    │                     │
│  └──────────────┘  └──────────────┘  └──────────────┘                     │
│                                                                             │
│  路由效果 (示例):                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  不使用路由: 所有请求 -> GPT-4o        月成本: $3,000               │   │
│  │  使用路由:   30% GPT-4o + 40% Mini + 30% 3.5   月成本: $800        │   │
│  │  节省: 73%     质量下降: <5%                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 模型路由实现

```python
"""
智能模型路由系统
包含: 复杂度评估、任务分类、成本感知路由、Fallback策略
"""

import re
import time
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# 第一部分: 模型层级定义
# ============================================================

class ModelTier(Enum):
    TIER1_LARGE = "tier1_large"     # 大模型 (GPT-4o, Claude Sonnet)
    TIER2_MEDIUM = "tier2_medium"   # 中模型 (GPT-4o-mini, Claude Haiku)
    TIER3_SMALL = "tier3_small"     # 小模型 (GPT-3.5, 本地7B)


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    tier: ModelTier
    input_price: float     # $/1M tokens
    output_price: float    # $/1M tokens
    max_tokens: int
    avg_latency_ms: float
    quality_score: float   # 0-1, 模型能力评分
    available: bool = True


# 可用模型池
MODEL_POOL = {
    "gpt-4o": ModelConfig(
        "gpt-4o", ModelTier.TIER1_LARGE,
        2.50, 10.00, 128000, 1500, 0.95
    ),
    "claude-sonnet": ModelConfig(
        "claude-sonnet", ModelTier.TIER1_LARGE,
        3.00, 15.00, 200000, 1800, 0.94
    ),
    "gpt-4o-mini": ModelConfig(
        "gpt-4o-mini", ModelTier.TIER2_MEDIUM,
        0.15, 0.60, 128000, 800, 0.82
    ),
    "claude-haiku": ModelConfig(
        "claude-haiku", ModelTier.TIER2_MEDIUM,
        0.25, 1.25, 200000, 600, 0.80
    ),
    "deepseek-v3": ModelConfig(
        "deepseek-v3", ModelTier.TIER2_MEDIUM,
        0.27, 1.10, 65536, 900, 0.83
    ),
    "gpt-3.5-turbo": ModelConfig(
        "gpt-3.5-turbo", ModelTier.TIER3_SMALL,
        0.50, 1.50, 16385, 400, 0.70
    ),
    "local-qwen2-7b": ModelConfig(
        "local-qwen2-7b", ModelTier.TIER3_SMALL,
        0.0, 0.0, 8192, 300, 0.65
    ),
}


# ============================================================
# 第二部分: 任务复杂度评估
# ============================================================

class ComplexityAnalyzer:
    """
    任务复杂度评估器
    基于多维特征判断任务所需的模型能力
    """

    # 复杂任务关键词
    COMPLEX_KEYWORDS = [
        "推理", "分析", "为什么", "解释原理", "证明", "比较",
        "设计方案", "架构", "优化", "debug", "算法",
        "多步骤", "复杂", "深入", "详细分析", "数学",
        "代码重构", "系统设计", "trade-off",
    ]

    # 简单任务关键词
    SIMPLE_KEYWORDS = [
        "翻译", "分类", "是否", "是不是", "情感", "摘要",
        "提取", "格式化", "转换", "列出", "总结",
        "yes or no", "true or false",
    ]

    @staticmethod
    def estimate_complexity(query: str,
                            task_type: str = "") -> Dict:
        """
        评估查询复杂度

        Returns:
            {
                "score": 0-1 复杂度分数,
                "tier": 推荐模型层级,
                "reasons": 评估原因列表,
            }
        """
        score = 0.5  # 基础分
        reasons = []

        text = query.lower()

        # 1. 长度评估 (长文本通常更复杂)
        char_count = len(query)
        if char_count > 2000:
            score += 0.15
            reasons.append(f"长文本({char_count}字)")
        elif char_count < 50:
            score -= 0.1
            reasons.append("短文本")

        # 2. 关键词匹配
        complex_hits = sum(
            1 for kw in ComplexityAnalyzer.COMPLEX_KEYWORDS
            if kw in text
        )
        simple_hits = sum(
            1 for kw in ComplexityAnalyzer.SIMPLE_KEYWORDS
            if kw in text
        )

        if complex_hits > 0:
            score += complex_hits * 0.1
            reasons.append(f"含{complex_hits}个复杂关键词")
        if simple_hits > 0:
            score -= simple_hits * 0.08
            reasons.append(f"含{simple_hits}个简单关键词")

        # 3. 任务类型
        type_scores = {
            "reasoning": 0.3,
            "code_generation": 0.25,
            "analysis": 0.2,
            "creative_writing": 0.15,
            "translation": -0.15,
            "classification": -0.2,
            "sentiment": -0.25,
            "extraction": -0.15,
            "qa_simple": -0.1,
        }
        if task_type in type_scores:
            score += type_scores[task_type]
            reasons.append(f"任务类型: {task_type}")

        # 4. 是否包含代码
        if "```" in query or "def " in query or "class " in query:
            score += 0.15
            reasons.append("包含代码")

        # 5. 是否需要多步推理 (问号数量)
        question_count = query.count("?") + query.count("？")
        if question_count > 2:
            score += 0.1
            reasons.append(f"多个问题({question_count})")

        # 限制范围
        score = max(0.0, min(1.0, score))

        # 决定模型层级
        if score >= 0.7:
            tier = ModelTier.TIER1_LARGE
        elif score >= 0.4:
            tier = ModelTier.TIER2_MEDIUM
        else:
            tier = ModelTier.TIER3_SMALL

        return {
            "score": round(score, 2),
            "tier": tier,
            "reasons": reasons,
        }


# ============================================================
# 第三部分: 智能路由器
# ============================================================

class ModelRouter:
    """
    智能模型路由器
    根据任务复杂度和成本约束选择最优模型
    """

    def __init__(self, models: Dict[str, ModelConfig] = None,
                 cost_weight: float = 0.4,
                 quality_weight: float = 0.4,
                 latency_weight: float = 0.2):
        """
        Args:
            models: 可用模型配置
            cost_weight: 成本权重 (越高越倾向便宜模型)
            quality_weight: 质量权重 (越高越倾向高质量模型)
            latency_weight: 延迟权重 (越高越倾向快模型)
        """
        self.models = models or MODEL_POOL.copy()
        self.cost_w = cost_weight
        self.quality_w = quality_weight
        self.latency_w = latency_weight
        self.analyzer = ComplexityAnalyzer()
        self.routing_log: List[Dict] = []

    def route(self, query: str, task_type: str = "",
              prefer_tier: ModelTier = None,
              max_cost_per_1m: float = None) -> Tuple[str, Dict]:
        """
        路由请求到最优模型

        Args:
            query: 用户查询
            task_type: 任务类型
            prefer_tier: 强制指定模型层级
            max_cost_per_1m: 最大成本限制 ($/1M tokens)

        Returns:
            (模型名称, 路由详情)
        """
        # 1. 评估复杂度
        complexity = self.analyzer.estimate_complexity(query, task_type)
        target_tier = prefer_tier or complexity["tier"]

        # 2. 筛选候选模型
        candidates = []
        for name, cfg in self.models.items():
            if not cfg.available:
                continue
            if max_cost_per_1m and cfg.input_price > max_cost_per_1m:
                continue
            candidates.append((name, cfg))

        if not candidates:
            # Fallback: 使用任何可用模型
            candidates = [
                (n, c) for n, c in self.models.items() if c.available
            ]

        # 3. 评分排序
        scored = []
        for name, cfg in candidates:
            # 层级匹配分 (匹配目标层级得高分)
            tier_score = 1.0 if cfg.tier == target_tier else 0.5

            # 成本分 (越便宜越高, 归一化)
            max_price = max(c.input_price for _, c in candidates) or 1
            cost_score = 1.0 - (cfg.input_price / max_price)

            # 质量分 (直接用模型质量评分)
            quality_score = cfg.quality_score

            # 延迟分 (越快越高, 归一化)
            max_lat = max(c.avg_latency_ms for _, c in candidates) or 1
            latency_score = 1.0 - (cfg.avg_latency_ms / max_lat)

            # 综合得分
            if cfg.tier == target_tier:
                # 匹配层级的模型, 用正常权重
                total = (cost_score * self.cost_w +
                         quality_score * self.quality_w +
                         latency_score * self.latency_w)
            else:
                # 不匹配层级的, 打折
                total = (cost_score * self.cost_w +
                         quality_score * self.quality_w +
                         latency_score * self.latency_w) * 0.6

            scored.append((name, cfg, total))

        scored.sort(key=lambda x: -x[2])

        # 4. 选择最优
        best_name, best_cfg, best_score = scored[0]

        route_info = {
            "selected_model": best_name,
            "model_tier": best_cfg.tier.value,
            "complexity": complexity,
            "score": round(best_score, 3),
            "input_price": best_cfg.input_price,
            "output_price": best_cfg.output_price,
        }

        # 记录路由日志
        self.routing_log.append({
            "timestamp": time.time(),
            "query_preview": query[:50],
            **route_info,
        })

        return best_name, route_info

    def get_routing_stats(self) -> Dict:
        """获取路由统计"""
        if not self.routing_log:
            return {"total_routes": 0}

        model_counts = {}
        tier_counts = {}
        for log in self.routing_log:
            model = log["selected_model"]
            tier = log["model_tier"]
            model_counts[model] = model_counts.get(model, 0) + 1
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        total = len(self.routing_log)
        return {
            "total_routes": total,
            "model_distribution": {
                k: f"{v/total*100:.1f}%" for k, v in model_counts.items()
            },
            "tier_distribution": {
                k: f"{v/total*100:.1f}%" for k, v in tier_counts.items()
            },
        }

    def estimate_savings(self, baseline_model: str = "gpt-4o") -> Dict:
        """估算相比全部使用基线模型的成本节省"""
        if not self.routing_log:
            return {}

        baseline = self.models.get(baseline_model)
        if not baseline:
            return {}

        baseline_cost = 0.0
        routed_cost = 0.0
        avg_tokens = 1000  # 假设平均每次1000 tokens

        for log in self.routing_log:
            # 基线成本
            baseline_cost += (baseline.input_price + baseline.output_price) * avg_tokens / 1_000_000

            # 路由后成本
            routed_cost += (log["input_price"] + log["output_price"]) * avg_tokens / 1_000_000

        savings = baseline_cost - routed_cost
        pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0

        return {
            "baseline_model": baseline_model,
            "baseline_cost": round(baseline_cost, 4),
            "routed_cost": round(routed_cost, 4),
            "savings": round(savings, 4),
            "savings_pct": round(pct, 1),
        }


# ============================================================
# 演示
# ============================================================

def demo_model_routing():
    """演示智能模型路由"""

    print("=" * 60)
    print("智能模型路由演示")
    print("=" * 60)

    router = ModelRouter(
        cost_weight=0.4,
        quality_weight=0.4,
        latency_weight=0.2,
    )

    # 不同复杂度的任务
    test_queries = [
        ("这条评论是正面还是负面的: '产品质量很好'", "sentiment"),
        ("将以下中文翻译成英文: '今天天气很好'", "translation"),
        ("列出Python的5种数据类型", "qa_simple"),
        ("请详细分析GPT和BERT的架构差异, 并比较它们在不同NLP任务上的表现", "analysis"),
        ("设计一个高并发的消息队列系统, 要求支持百万级QPS", "reasoning"),
        ("请帮我重构这段代码, 使用设计模式优化架构:\n```python\ndef process():\n    pass\n```", "code_generation"),
        ("总结以下文章的主要观点", "extraction"),
        ("为什么Transformer比RNN更适合处理长序列? 请从数学角度推导注意力机制的优势", "reasoning"),
    ]

    print(f"\n{'序号':<4} {'任务类型':<16} {'复杂度':<8} {'选择模型':<18} {'层级'}")
    print("-" * 70)

    for i, (query, task_type) in enumerate(test_queries, 1):
        model, info = router.route(query, task_type)
        score = info["complexity"]["score"]
        tier = info["model_tier"]
        print(f"{i:<4} {task_type:<16} {score:<8.2f} {model:<18} {tier}")

    # 路由统计
    stats = router.get_routing_stats()
    print(f"\n路由统计:")
    print(f"  总路由次数: {stats['total_routes']}")
    print(f"  模型分布:")
    for model, pct in stats["model_distribution"].items():
        print(f"    {model}: {pct}")
    print(f"  层级分布:")
    for tier, pct in stats["tier_distribution"].items():
        print(f"    {tier}: {pct}")

    # 成本节省估算
    savings = router.estimate_savings("gpt-4o")
    print(f"\n成本节省 (相比全部使用 {savings.get('baseline_model', 'N/A')}):")
    print(f"  基线成本:   ${savings.get('baseline_cost', 0):.4f}")
    print(f"  路由后成本: ${savings.get('routed_cost', 0):.4f}")
    print(f"  节省:       ${savings.get('savings', 0):.4f} "
          f"({savings.get('savings_pct', 0):.1f}%)")


if __name__ == "__main__":
    demo_model_routing()
```

---

## Batch处理

### Batch处理架构

对于非实时场景(数据标注、批量翻译、离线分析), 使用Batch API可以获得50%折扣,
同时通过请求合并提升整体吞吐量。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Batch处理架构                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入数据                                                                   │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                  │
│  │ Task 1 │ │ Task 2 │ │ Task 3 │ │ Task 4 │ │ ...    │  共N个任务      │
│  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘                  │
│      └──────────┼──────────┼──────────┼──────────┘                        │
│                 ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Batch调度器                                                        │   │
│  │                                                                     │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │   │
│  │  │ 任务分组        │    │ 并发控制        │    │ 重试管理        │ │   │
│  │  │                 │    │                 │    │                 │ │   │
│  │  │ - 按模型分      │    │ - RPM限制       │    │ - 指数退避      │ │   │
│  │  │ - 按优先级分    │    │ - TPM限制       │    │ - 最大重试3次   │ │   │
│  │  │ - 按大小分批    │    │ - 信号量控制    │    │ - 死信队列      │ │   │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘ │   │
│  │                                                                     │   │
│  │  ┌────────────────────────────────────────────────────────────┐    │   │
│  │  │ Batch 1: [Task1, Task2, Task3]  (并行)                    │    │   │
│  │  │ Batch 2: [Task4, Task5, Task6]  (等Batch1完成后执行)      │    │   │
│  │  │ Batch 3: [Task7, Task8]         (等Batch2完成后执行)      │    │   │
│  │  └────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                 │                                                          │
│                 ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  结果聚合                                                           │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐                           │   │
│  │  │ 成功结果 │ │ 失败记录 │ │ 统计报告 │                           │   │
│  │  │ results[]│ │ errors[] │ │ cost/time│                           │   │
│  │  └──────────┘ └──────────┘ └──────────┘                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Batch API对比实时API:                                                     │
│  ┌──────────────┬──────────────┬──────────────┐                           │
│  │              │ 实时API      │ Batch API    │                           │
│  ├──────────────┼──────────────┼──────────────┤                           │
│  │ 价格         │ 标准价       │ 50%折扣      │                           │
│  │ 延迟         │ 秒级         │ 小时级       │                           │
│  │ 适用场景     │ 在线服务     │ 离线处理     │                           │
│  │ 并发限制     │ RPM/TPM      │ 更宽松       │                           │
│  └──────────────┴──────────────┴──────────────┘                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Batch处理器实现

```python
"""
Batch处理器
包含: 任务分批、并发控制、重试机制、进度跟踪
"""

import time
import random
import hashlib
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# 第一部分: 数据结构定义
# ============================================================

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchTask:
    """批处理任务"""
    task_id: str
    input_data: str
    task_type: str = "default"
    model: str = "gpt-4o-mini"
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0


@dataclass
class BatchResult:
    """批处理结果"""
    total_tasks: int = 0
    completed: int = 0
    failed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    total_time_seconds: float = 0.0
    results: List[Dict] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)


class BatchProcessor:
    """
    LLM批处理器
    支持并发控制、自动重试、进度跟踪
    """

    def __init__(self, max_concurrent: int = 5,
                 max_retries: int = 3,
                 batch_size: int = 10,
                 rate_limit_rpm: int = 60):
        """
        Args:
            max_concurrent: 最大并发数
            max_retries: 最大重试次数
            batch_size: 每批大小
            rate_limit_rpm: 每分钟请求限制
        """
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.rate_limit_rpm = rate_limit_rpm
        self.min_interval = 60.0 / rate_limit_rpm

    def _simulate_llm_call(self, task: BatchTask) -> BatchTask:
        """模拟LLM API调用 (演示用)"""
        start = time.time()

        # 模拟随机延迟和偶尔失败
        time.sleep(random.uniform(0.01, 0.05))

        if random.random() < 0.05:  # 5%失败率
            task.status = TaskStatus.FAILED
            task.error = "模拟API错误: Rate limit exceeded"
            return task

        # 模拟结果
        task.result = f"处理完成: {task.input_data[:30]}..."
        task.input_tokens = len(task.input_data) * 2
        task.output_tokens = random.randint(50, 300)
        task.cost = (task.input_tokens * 0.15 +
                     task.output_tokens * 0.60) / 1_000_000
        task.status = TaskStatus.COMPLETED
        task.latency_ms = (time.time() - start) * 1000

        return task

    def process_batch(self, tasks: List[BatchTask],
                      callback: Callable = None) -> BatchResult:
        """
        处理一批任务

        Args:
            tasks: 任务列表
            callback: 每完成一个任务的回调 fn(task, progress_pct)
        """
        result = BatchResult(total_tasks=len(tasks))
        start_time = time.time()

        # 分批处理
        batches = [
            tasks[i:i + self.batch_size]
            for i in range(0, len(tasks), self.batch_size)
        ]

        completed_count = 0

        for batch_idx, batch in enumerate(batches):
            # 并发执行当前批次
            with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
                futures = {
                    pool.submit(self._process_single, task): task
                    for task in batch
                }

                for future in as_completed(futures):
                    task = future.result()
                    completed_count += 1

                    if task.status == TaskStatus.COMPLETED:
                        result.completed += 1
                        result.total_input_tokens += task.input_tokens
                        result.total_output_tokens += task.output_tokens
                        result.total_cost += task.cost
                        result.results.append({
                            "task_id": task.task_id,
                            "result": task.result,
                        })
                    else:
                        result.failed += 1
                        result.errors.append({
                            "task_id": task.task_id,
                            "error": task.error,
                            "retries": task.retries,
                        })

                    # 进度回调
                    if callback:
                        pct = completed_count / len(tasks) * 100
                        callback(task, pct)

        result.total_time_seconds = time.time() - start_time
        return result

    def _process_single(self, task: BatchTask) -> BatchTask:
        """处理单个任务 (含重试)"""
        for attempt in range(self.max_retries + 1):
            task = self._simulate_llm_call(task)
            if task.status == TaskStatus.COMPLETED:
                return task

            task.retries = attempt + 1
            # 指数退避
            wait = min(2 ** attempt * 0.1, 5.0)
            time.sleep(wait)

        return task


# ============================================================
# 演示
# ============================================================

def demo_batch_processing():
    """演示Batch处理"""

    print("=" * 60)
    print("Batch处理演示")
    print("=" * 60)

    # 创建任务
    sample_texts = [
        "这款手机的拍照效果非常好, 电池续航也不错",
        "服务态度太差了, 等了一个小时才上菜",
        "物流很快, 包装完好, 好评",
        "质量不行, 用了两天就坏了",
        "性价比很高, 推荐购买",
        "客服响应很慢, 问题一直没解决",
        "使用体验超出预期, 非常满意",
        "产品和描述不符, 有些失望",
        "做工精细, 材质很好",
        "价格偏贵, 一般般吧",
        "非常好用的产品, 已经推荐给朋友了",
        "包装简陋, 到手有些划痕",
        "功能齐全, 操作简便",
        "退换货流程太复杂了",
        "五星好评, 下次还会回购",
    ]

    tasks = []
    for i, text in enumerate(sample_texts):
        task = BatchTask(
            task_id=f"task_{i+1:03d}",
            input_data=f"请分析以下评论的情感倾向(正面/负面/中性): {text}",
            task_type="sentiment",
            model="gpt-4o-mini",
        )
        tasks.append(task)

    print(f"\n总任务数: {len(tasks)}")
    print(f"批大小:   {5}")
    print(f"并发数:   {3}")

    # 进度回调
    def on_progress(task, pct):
        status = "OK" if task.status == TaskStatus.COMPLETED else "FAIL"
        print(f"  [{pct:5.1f}%] {task.task_id}: {status}")

    # 执行
    processor = BatchProcessor(
        max_concurrent=3,
        max_retries=2,
        batch_size=5,
        rate_limit_rpm=120,
    )

    result = processor.process_batch(tasks, callback=on_progress)

    # 报告
    print(f"\n{'=' * 50}")
    print("Batch处理报告")
    print(f"{'=' * 50}")
    print(f"总任务:       {result.total_tasks}")
    print(f"成功:         {result.completed}")
    print(f"失败:         {result.failed}")
    print(f"成功率:       {result.completed/result.total_tasks*100:.1f}%")
    print(f"总输入Token:  {result.total_input_tokens:,}")
    print(f"总输出Token:  {result.total_output_tokens:,}")
    print(f"总成本:       ${result.total_cost:.6f}")
    print(f"总耗时:       {result.total_time_seconds:.2f}s")
    print(f"平均速度:     {result.completed/max(result.total_time_seconds,0.01):.1f} tasks/s")

    # Batch API vs 实时API对比
    print(f"\n成本对比:")
    realtime_cost = result.total_cost
    batch_cost = realtime_cost * 0.5  # Batch API 50%折扣
    print(f"  实时API成本: ${realtime_cost:.6f}")
    print(f"  Batch API:   ${batch_cost:.6f} (50%折扣)")
    print(f"  节省:        ${realtime_cost - batch_cost:.6f}")

    if result.errors:
        print(f"\n失败任务:")
        for err in result.errors:
            print(f"  {err['task_id']}: {err['error']} (重试{err['retries']}次)")


if __name__ == "__main__":
    demo_batch_processing()
```

---

## ROI计算

### ROI评估框架

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    成本优化ROI评估框架                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  优化措施                  投入成本              预期收益                    │
│  ┌──────────────────┐    ┌───────────────┐    ┌───────────────────┐       │
│  │ 1. Token优化     │    │ 工程师: 1周   │    │ Token消耗 -30%    │       │
│  │    - Prompt压缩  │    │ 开发: 40h     │    │ 月省: $500-2000   │       │
│  │    - 输出控制    │    │ 成本: ~$4000  │    │ 回收期: 2-4个月   │       │
│  └──────────────────┘    └───────────────┘    └───────────────────┘       │
│                                                                             │
│  ┌──────────────────┐    ┌───────────────┐    ┌───────────────────┐       │
│  │ 2. 语义缓存     │    │ 工程师: 2周   │    │ API调用 -40%      │       │
│  │    - 向量缓存   │    │ 开发: 80h     │    │ 月省: $1000-5000  │       │
│  │    - Redis/向量DB│    │ 成本: ~$10000 │    │ 回收期: 2-6个月   │       │
│  └──────────────────┘    └───────────────┘    └───────────────────┘       │
│                                                                             │
│  ┌──────────────────┐    ┌───────────────┐    ┌───────────────────┐       │
│  │ 3. 模型路由     │    │ 工程师: 1周   │    │ 成本 -50%         │       │
│  │    - 复杂度分析 │    │ 开发: 40h     │    │ 月省: $2000-8000  │       │
│  │    - 分层路由   │    │ 成本: ~$4000  │    │ 回收期: 1-2个月   │       │
│  └──────────────────┘    └───────────────┘    └───────────────────┘       │
│                                                                             │
│  ┌──────────────────┐    ┌───────────────┐    ┌───────────────────┐       │
│  │ 4. Batch处理    │    │ 工程师: 3天   │    │ 离线任务 -50%     │       │
│  │    - 离线批量   │    │ 开发: 24h     │    │ 月省: $500-2000   │       │
│  │    - 异步队列   │    │ 成本: ~$2000  │    │ 回收期: 1个月     │       │
│  └──────────────────┘    └───────────────┘    └───────────────────┘       │
│                                                                             │
│  综合效果:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  优化前月成本: $10,000                                              │   │
│  │  优化后月成本: $3,000                                               │   │
│  │  月节省:       $7,000  (70%)                                        │   │
│  │  实施总投入:   $20,000 (一次性)                                     │   │
│  │  投资回收期:   ~3个月                                               │   │
│  │  年化ROI:      320%                                                 │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### ROI计算器

```python
"""
成本优化ROI计算器
包含: 投入产出分析、回收期计算、场景模拟、优化建议
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class OptimizationMeasure:
    """优化措施定义"""
    name: str
    description: str
    dev_hours: float          # 开发工时
    hourly_rate: float        # 工程师时薪 ($/h)
    infra_cost: float         # 额外基础设施成本 ($/月)
    savings_pct: float        # 预期成本节省比例 (0-1)
    quality_impact_pct: float # 质量影响 (负值=下降)
    applies_to_pct: float     # 适用请求比例 (0-1)
    confidence: float         # 估算信心度 (0-1)


@dataclass
class ROIResult:
    """ROI计算结果"""
    measure_name: str
    implementation_cost: float  # 一次性实施成本
    monthly_infra_cost: float   # 每月额外基础设施成本
    monthly_savings: float      # 每月节省
    net_monthly_savings: float  # 净月节省 (扣除基础设施)
    payback_months: float       # 回收期 (月)
    annual_roi_pct: float       # 年化ROI (%)
    quality_impact: str         # 质量影响评估


class ROICalculator:
    """
    成本优化ROI计算器
    帮助决策哪些优化措施值得投入
    """

    # 预定义优化措施模板
    MEASURE_TEMPLATES = {
        "token_optimization": OptimizationMeasure(
            name="Token优化",
            description="Prompt压缩、输出长度控制、上下文窗口管理",
            dev_hours=40, hourly_rate=100, infra_cost=0,
            savings_pct=0.30, quality_impact_pct=-2.0,
            applies_to_pct=1.0, confidence=0.85,
        ),
        "semantic_cache": OptimizationMeasure(
            name="语义缓存",
            description="基于向量相似度的查询结果缓存",
            dev_hours=80, hourly_rate=100, infra_cost=200,
            savings_pct=0.40, quality_impact_pct=-1.0,
            applies_to_pct=0.60, confidence=0.75,
        ),
        "model_routing": OptimizationMeasure(
            name="模型路由",
            description="根据任务复杂度自动选择成本最优模型",
            dev_hours=40, hourly_rate=100, infra_cost=50,
            savings_pct=0.50, quality_impact_pct=-3.0,
            applies_to_pct=0.80, confidence=0.80,
        ),
        "batch_processing": OptimizationMeasure(
            name="Batch处理",
            description="离线任务使用Batch API, 享受50%折扣",
            dev_hours=24, hourly_rate=100, infra_cost=30,
            savings_pct=0.50, quality_impact_pct=0.0,
            applies_to_pct=0.30, confidence=0.90,
        ),
    }

    def __init__(self, current_monthly_cost: float):
        """
        Args:
            current_monthly_cost: 当前每月LLM费用 ($)
        """
        self.monthly_cost = current_monthly_cost
        self.measures: List[OptimizationMeasure] = []
        self.results: List[ROIResult] = []

    def add_measure(self, measure: OptimizationMeasure):
        """添加优化措施"""
        self.measures.append(measure)

    def add_template_measure(self, template_name: str,
                             **overrides):
        """使用预定义模板添加优化措施"""
        template = self.MEASURE_TEMPLATES.get(template_name)
        if not template:
            print(f"[错误] 未知模板: {template_name}")
            return

        import copy
        measure = copy.deepcopy(template)
        for key, value in overrides.items():
            if hasattr(measure, key):
                setattr(measure, key, value)
        self.measures.append(measure)

    def calculate_roi(self, measure: OptimizationMeasure) -> ROIResult:
        """计算单个优化措施的ROI"""

        # 实施成本 = 开发工时 * 时薪
        impl_cost = measure.dev_hours * measure.hourly_rate

        # 月度节省 = 当前成本 * 节省比例 * 适用比例 * 信心度
        monthly_savings = (self.monthly_cost *
                           measure.savings_pct *
                           measure.applies_to_pct *
                           measure.confidence)

        # 净月度节省 = 月度节省 - 额外基础设施成本
        net_savings = monthly_savings - measure.infra_cost

        # 回收期
        if net_savings > 0:
            payback = impl_cost / net_savings
        else:
            payback = float('inf')

        # 年化ROI = (年净节省 - 实施成本) / 实施成本 * 100
        annual_net = net_savings * 12 - impl_cost
        annual_roi = (annual_net / impl_cost * 100) if impl_cost > 0 else 0

        # 质量影响
        qi = measure.quality_impact_pct
        if qi >= 0:
            quality_str = f"无负面影响 ({qi:+.1f}%)"
        elif qi > -2:
            quality_str = f"轻微影响 ({qi:+.1f}%)"
        elif qi > -5:
            quality_str = f"中等影响 ({qi:+.1f}%)"
        else:
            quality_str = f"显著影响 ({qi:+.1f}%), 需谨慎"

        return ROIResult(
            measure_name=measure.name,
            implementation_cost=impl_cost,
            monthly_infra_cost=measure.infra_cost,
            monthly_savings=round(monthly_savings, 2),
            net_monthly_savings=round(net_savings, 2),
            payback_months=round(payback, 1),
            annual_roi_pct=round(annual_roi, 1),
            quality_impact=quality_str,
        )

    def calculate_all(self) -> List[ROIResult]:
        """计算所有优化措施的ROI"""
        self.results = [
            self.calculate_roi(m) for m in self.measures
        ]
        # 按年化ROI排序
        self.results.sort(key=lambda r: -r.annual_roi_pct)
        return self.results

    def generate_report(self) -> str:
        """生成完整的ROI分析报告"""
        if not self.results:
            self.calculate_all()

        lines = [
            f"{'=' * 70}",
            f"成本优化ROI分析报告",
            f"{'=' * 70}",
            f"当前月度LLM成本: ${self.monthly_cost:,.2f}",
            f"分析措施数量:     {len(self.results)}",
            f"{'=' * 70}",
        ]

        total_savings = 0
        total_impl_cost = 0

        for i, r in enumerate(self.results, 1):
            lines.extend([
                f"\n{'─' * 60}",
                f"[{i}] {r.measure_name}",
                f"{'─' * 60}",
                f"  实施成本:        ${r.implementation_cost:,.0f}",
                f"  月度基础设施:    ${r.monthly_infra_cost:,.0f}/月",
                f"  月度节省 (毛):   ${r.monthly_savings:,.2f}/月",
                f"  月度节省 (净):   ${r.net_monthly_savings:,.2f}/月",
                f"  投资回收期:      {r.payback_months:.1f} 个月",
                f"  年化ROI:         {r.annual_roi_pct:.1f}%",
                f"  质量影响:        {r.quality_impact}",
                f"  推荐:            {'强烈推荐' if r.annual_roi_pct > 200 else '推荐' if r.annual_roi_pct > 100 else '考虑' if r.annual_roi_pct > 0 else '不推荐'}",
            ])
            total_savings += r.net_monthly_savings
            total_impl_cost += r.implementation_cost

        # 综合评估
        lines.extend([
            f"\n{'=' * 70}",
            f"综合评估",
            f"{'=' * 70}",
            f"  总实施成本:      ${total_impl_cost:,.0f} (一次性)",
            f"  总月度净节省:    ${total_savings:,.2f}",
            f"  优化后月度成本:  ${self.monthly_cost - total_savings:,.2f}",
            f"  成本降低比例:    {total_savings/self.monthly_cost*100:.1f}%",
            f"  综合回收期:      {total_impl_cost/max(total_savings,1):.1f} 个月",
        ])

        if total_savings > 0:
            annual = total_savings * 12 - total_impl_cost
            roi = annual / total_impl_cost * 100 if total_impl_cost > 0 else 0
            lines.append(f"  年化综合ROI:     {roi:.1f}%")

        return "\n".join(lines)

    def scenario_simulation(self, months: int = 12) -> str:
        """月度成本变化模拟"""
        if not self.results:
            self.calculate_all()

        lines = [
            f"\n{'=' * 60}",
            f"成本变化模拟 ({months}个月)",
            f"{'=' * 60}",
            f"{'月份':<6} {'无优化':>12} {'有优化':>12} {'累计节省':>12}",
            f"{'─' * 48}",
        ]

        total_no_opt = 0
        total_with_opt = 0
        total_savings = sum(r.net_monthly_savings for r in self.results)
        impl_cost = sum(r.implementation_cost for r in self.results)

        cumulative_savings = -impl_cost  # 初始投入

        for month in range(1, months + 1):
            no_opt = self.monthly_cost
            with_opt = self.monthly_cost - total_savings
            total_no_opt += no_opt
            total_with_opt += with_opt
            cumulative_savings += total_savings

            marker = " <-- 回收" if cumulative_savings >= 0 and cumulative_savings - total_savings < 0 else ""
            lines.append(
                f"  {month:<4} ${no_opt:>10,.2f} ${with_opt:>10,.2f} "
                f"${cumulative_savings:>10,.2f}{marker}"
            )

        lines.extend([
            f"{'─' * 48}",
            f"  {'合计':<4} ${total_no_opt:>10,.2f} ${total_with_opt:>10,.2f} "
            f"${cumulative_savings:>10,.2f}",
        ])

        return "\n".join(lines)


# ============================================================
# 演示
# ============================================================

def demo_roi_calculation():
    """演示ROI计算"""

    print("=" * 70)
    print("成本优化ROI计算演示")
    print("=" * 70)

    # 假设当前月度LLM成本为 $5000
    calc = ROICalculator(current_monthly_cost=5000)

    # 添加四种优化措施
    calc.add_template_measure("token_optimization")
    calc.add_template_measure("semantic_cache")
    calc.add_template_measure("model_routing")
    calc.add_template_measure("batch_processing")

    # 计算并生成报告
    calc.calculate_all()
    print(calc.generate_report())

    # 月度模拟
    print(calc.scenario_simulation(months=12))


if __name__ == "__main__":
    demo_roi_calculation()
```

---

## 总结

本教程涵盖了LLM成本优化策略的核心内容:

1. **成本分析**: API定价模型(Token计费)、成本追踪器、预算预警、按模型/任务维度的成本报告
2. **Token优化**: Prompt压缩(冗余短语去除)、上下文窗口管理(滑动窗口/重要性裁剪)、输出长度控制(任务模板)
3. **语义缓存**: 基于向量相似度的缓存匹配(余弦相似度)、LRU淘汰+TTL过期、命中率30-60%可节省同等比例成本
4. **模型路由**: 任务复杂度评估(关键词/长度/类型多维评分)、三层模型分级路由、成本节省50-70%
5. **Batch处理**: 离线任务分批并发处理、Batch API 50%折扣、自动重试与进度跟踪
6. **ROI计算**: 投入产出分析、回收期计算、质量影响评估、月度成本变化模拟

## 最佳实践

1. **先度量再优化**: 部署成本监控后再决定优化方向, 找出成本热点
2. **模型路由优先**: ROI最高, 实施简单, 质量影响可控
3. **语义缓存适合高重复场景**: FAQ、客服、文档问答命中率高
4. **Token优化持续进行**: 每个Prompt都值得审视是否有冗余
5. **Batch API用于离线任务**: 数据标注、批量翻译、报告生成等
6. **设定预算告警**: 避免意外成本飙升, 按日/周/月设定阈值
7. **质量与成本平衡**: 不要为省钱牺牲核心体验, 关键路径用好模型

## 参考资源

- [OpenAI API定价](https://openai.com/pricing)
- [Anthropic API定价](https://docs.anthropic.com/en/docs/about-claude/pricing)
- [GPTCache - 语义缓存框架](https://github.com/zilliztech/GPTCache)
- [LiteLLM - 多模型路由](https://github.com/BerriAI/litellm)
- [OpenAI Batch API文档](https://platform.openai.com/docs/guides/batch)
- [PromptCompressor - Prompt压缩](https://github.com/microsoft/LLMLingua)
- [Helicone - LLM成本监控](https://www.helicone.ai/)

---

**创建时间**: 2024-01-15
**最后更新**: 2024-01-15
