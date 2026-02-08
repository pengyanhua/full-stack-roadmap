# AI系统监控完整教程

## 目录
1. [监控概述](#监控概述)
2. [AI监控指标体系](#ai监控指标体系)
3. [Prometheus指标采集](#prometheus指标采集)
4. [Grafana可视化](#grafana可视化)
5. [LLM专用监控指标](#llm专用监控指标)
6. [告警系统](#告警系统)
7. [日志收集与分析](#日志收集与分析)
8. [完整监控系统搭建](#完整监控系统搭建)

---

## 监控概述

### 为什么需要AI系统监控

AI系统与传统软件不同,模型性能会随时间衰退(数据漂移),推理延迟受负载影响大,
GPU资源昂贵且需要精细管理。完善的监控体系是AI系统可靠运行的基础。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AI系统监控全景架构                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  数据采集层                                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ 应用指标 │ │ 系统指标 │ │ GPU指标  │ │ 业务指标 │ │ 模型指标 │       │
│  │          │ │          │ │          │ │          │ │          │       │
│  │ QPS      │ │ CPU      │ │ 利用率   │ │ 转化率   │ │ 准确率   │       │
│  │ 延迟     │ │ 内存     │ │ 显存     │ │ 用户满意 │ │ 漂移度   │       │
│  │ 错误率   │ │ 磁盘     │ │ 温度     │ │ 收入     │ │ 公平性   │       │
│  │ Token数  │ │ 网络     │ │ 功耗     │ │ 活跃度   │ │ 幻觉率   │       │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │
│       └────────────┴────────────┴────────────┴────────────┘               │
│                                    │                                       │
│                                    ▼                                       │
│  存储与处理层                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│  │  │ Prometheus   │  │ Elasticsearch│  │ ClickHouse   │             │   │
│  │  │ 时序指标     │  │ 日志存储     │  │ 分析查询     │             │   │
│  │  │ 15s采集间隔  │  │ 全文检索     │  │ 长期存储     │             │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │   │
│  │         │                 │                  │                      │   │
│  └─────────┼─────────────────┼──────────────────┼──────────────────────┘   │
│            │                 │                  │                          │
│            ▼                 ▼                  ▼                          │
│  展示与告警层                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│  │  │  Grafana     │  │  Alertmanager│  │   Kibana     │             │   │
│  │  │  仪表盘      │  │  告警路由    │  │   日志查询   │             │   │
│  │  │  实时可视化  │  │  去重/分组   │  │   日志分析   │             │   │
│  │  └──────────────┘  └──────┬───────┘  └──────────────┘             │   │
│  │                           │                                        │   │
│  │                    ┌──────▼───────┐                                │   │
│  │                    │ 通知渠道     │                                │   │
│  │                    │ Slack/钉钉   │                                │   │
│  │                    │ Email/短信   │                                │   │
│  │                    │ PagerDuty    │                                │   │
│  │                    └──────────────┘                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## AI监控指标体系

### 四层指标模型

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AI系统四层指标模型                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer 1: 基础设施指标 (Infrastructure)                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  CPU使用率 | 内存使用 | 磁盘IO | 网络带宽 | GPU利用率 | GPU显存    │   │
│  │  容器状态 | Pod重启次数 | 节点健康度                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Layer 2: 应用性能指标 (Application)                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  请求QPS | 延迟(P50/P95/P99) | 错误率 | 吞吐量(tokens/s)          │   │
│  │  TTFT(首Token时间) | 并发连接数 | 请求队列深度                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Layer 3: 模型质量指标 (Model Quality)                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  准确率 | F1分数 | 数据漂移(PSI/KL散度) | 幻觉率 | 拒答率          │   │
│  │  输出多样性 | 上下文利用率 | 指令遵循度                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Layer 4: 业务效果指标 (Business)                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  用户满意度 | 任务完成率 | 对话轮次 | 人工介入率 | Token成本         │   │
│  │  收入影响 | 用户留存 | 转化率                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 指标体系代码实现

```python
"""
AI系统监控指标体系
包含: 指标定义、采集、聚合、健康评分
"""

import time
import random
import statistics
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class MetricType(Enum):
    COUNTER = "counter"       # 累计值 (只增不减)
    GAUGE = "gauge"           # 瞬时值 (可增可减)
    HISTOGRAM = "histogram"   # 分布 (百分位数)
    SUMMARY = "summary"       # 摘要


class MetricLevel(Enum):
    INFRASTRUCTURE = "基础设施"
    APPLICATION = "应用性能"
    MODEL_QUALITY = "模型质量"
    BUSINESS = "业务效果"


@dataclass
class MetricDefinition:
    """指标定义"""
    name: str
    type: MetricType
    level: MetricLevel
    description: str
    unit: str = ""
    warning_threshold: float = 0
    critical_threshold: float = 0
    labels: List[str] = field(default_factory=list)


@dataclass
class MetricSample:
    """指标采样值"""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsRegistry:
    """
    指标注册与采集中心
    管理所有监控指标的定义、采样和查询
    """

    def __init__(self):
        self.definitions: Dict[str, MetricDefinition] = {}
        self.samples: Dict[str, List[MetricSample]] = defaultdict(list)
        self.counters: Dict[str, float] = defaultdict(float)
        self._register_ai_metrics()

    def _register_ai_metrics(self):
        """注册AI系统标准指标"""
        metrics = [
            # 基础设施层
            MetricDefinition(
                "gpu_utilization", MetricType.GAUGE, MetricLevel.INFRASTRUCTURE,
                "GPU计算利用率", "%", warning_threshold=85, critical_threshold=95,
            ),
            MetricDefinition(
                "gpu_memory_used", MetricType.GAUGE, MetricLevel.INFRASTRUCTURE,
                "GPU显存使用量", "GB", warning_threshold=70, critical_threshold=90,
            ),
            MetricDefinition(
                "gpu_temperature", MetricType.GAUGE, MetricLevel.INFRASTRUCTURE,
                "GPU温度", "C", warning_threshold=80, critical_threshold=90,
            ),
            # 应用性能层
            MetricDefinition(
                "request_latency_ms", MetricType.HISTOGRAM, MetricLevel.APPLICATION,
                "请求延迟", "ms", warning_threshold=2000, critical_threshold=5000,
            ),
            MetricDefinition(
                "ttft_ms", MetricType.HISTOGRAM, MetricLevel.APPLICATION,
                "首Token延迟", "ms", warning_threshold=500, critical_threshold=1000,
            ),
            MetricDefinition(
                "requests_total", MetricType.COUNTER, MetricLevel.APPLICATION,
                "请求总数", "",
            ),
            MetricDefinition(
                "errors_total", MetricType.COUNTER, MetricLevel.APPLICATION,
                "错误总数", "",
            ),
            MetricDefinition(
                "tokens_per_second", MetricType.GAUGE, MetricLevel.APPLICATION,
                "生成吞吐量", "tokens/s",
            ),
            # 模型质量层
            MetricDefinition(
                "model_accuracy", MetricType.GAUGE, MetricLevel.MODEL_QUALITY,
                "模型准确率", "%", warning_threshold=85, critical_threshold=80,
            ),
            MetricDefinition(
                "data_drift_score", MetricType.GAUGE, MetricLevel.MODEL_QUALITY,
                "数据漂移评分(PSI)", "", warning_threshold=0.1, critical_threshold=0.2,
            ),
            MetricDefinition(
                "hallucination_rate", MetricType.GAUGE, MetricLevel.MODEL_QUALITY,
                "幻觉率", "%", warning_threshold=10, critical_threshold=20,
            ),
            # 业务效果层
            MetricDefinition(
                "user_satisfaction", MetricType.GAUGE, MetricLevel.BUSINESS,
                "用户满意度", "%", warning_threshold=80, critical_threshold=70,
            ),
            MetricDefinition(
                "task_completion_rate", MetricType.GAUGE, MetricLevel.BUSINESS,
                "任务完成率", "%", warning_threshold=85, critical_threshold=75,
            ),
            MetricDefinition(
                "cost_per_request", MetricType.GAUGE, MetricLevel.BUSINESS,
                "每请求成本", "USD",
            ),
        ]
        for m in metrics:
            self.definitions[m.name] = m

    def record(self, name: str, value: float,
               labels: Dict[str, str] = None):
        """记录指标值"""
        if name not in self.definitions:
            return

        sample = MetricSample(
            name=name, value=value,
            timestamp=time.time(),
            labels=labels or {},
        )
        self.samples[name].append(sample)

        # Counter特殊处理
        if self.definitions[name].type == MetricType.COUNTER:
            key = f"{name}:{str(labels or {})}"
            self.counters[key] += value

        # 保留最近1小时的数据
        cutoff = time.time() - 3600
        self.samples[name] = [
            s for s in self.samples[name] if s.timestamp > cutoff
        ]

    def get_current(self, name: str) -> Optional[float]:
        """获取指标当前值"""
        samples = self.samples.get(name, [])
        if not samples:
            return None
        return samples[-1].value

    def get_percentile(self, name: str, percentile: float,
                       minutes: int = 5) -> Optional[float]:
        """获取指标百分位数"""
        cutoff = time.time() - minutes * 60
        values = [
            s.value for s in self.samples.get(name, [])
            if s.timestamp > cutoff
        ]
        if not values:
            return None
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * percentile / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def check_health(self) -> Dict:
        """检查所有指标健康状态"""
        health = {"status": "healthy", "checks": []}

        for name, defn in self.definitions.items():
            current = self.get_current(name)
            if current is None:
                continue

            status = "ok"
            # 对于延迟等越大越差的指标
            if defn.level in (MetricLevel.INFRASTRUCTURE, MetricLevel.APPLICATION):
                if defn.critical_threshold and current > defn.critical_threshold:
                    status = "critical"
                elif defn.warning_threshold and current > defn.warning_threshold:
                    status = "warning"
            # 对于准确率等越大越好的指标
            elif defn.level in (MetricLevel.MODEL_QUALITY, MetricLevel.BUSINESS):
                if defn.name in ("data_drift_score", "hallucination_rate",
                                 "cost_per_request"):
                    if defn.critical_threshold and current > defn.critical_threshold:
                        status = "critical"
                    elif defn.warning_threshold and current > defn.warning_threshold:
                        status = "warning"
                else:
                    if defn.critical_threshold and current < defn.critical_threshold:
                        status = "critical"
                    elif defn.warning_threshold and current < defn.warning_threshold:
                        status = "warning"

            if status != "ok":
                health["checks"].append({
                    "metric": name,
                    "value": current,
                    "status": status,
                    "description": defn.description,
                })
                if status == "critical":
                    health["status"] = "critical"
                elif status == "warning" and health["status"] != "critical":
                    health["status"] = "warning"

        return health

    def get_dashboard_data(self) -> str:
        """生成仪表盘数据"""
        lines = [
            f"{'=' * 70}",
            f"AI系统监控仪表盘",
            f"{'=' * 70}",
        ]

        for level in MetricLevel:
            lines.append(f"\n[{level.value}]")
            lines.append(f"{'─' * 50}")
            for name, defn in self.definitions.items():
                if defn.level != level:
                    continue
                current = self.get_current(name)
                if current is None:
                    continue
                p95 = self.get_percentile(name, 95, minutes=5)
                unit = f" {defn.unit}" if defn.unit else ""
                p95_str = f" (P95: {p95:.1f}{unit})" if p95 is not None else ""
                lines.append(
                    f"  {defn.description:<20} {current:.2f}{unit}{p95_str}"
                )

        # 健康检查
        health = self.check_health()
        lines.append(f"\n{'=' * 70}")
        lines.append(f"系统健康状态: {health['status'].upper()}")
        if health["checks"]:
            for check in health["checks"]:
                lines.append(
                    f"  [{check['status'].upper()}] {check['description']}: "
                    f"{check['value']:.2f}"
                )

        return "\n".join(lines)


def demo_metrics():
    """指标体系演示"""
    registry = MetricsRegistry()
    random.seed(42)

    # 模拟30个采样点
    for _ in range(30):
        registry.record("gpu_utilization", random.uniform(60, 95))
        registry.record("gpu_memory_used", random.uniform(50, 75))
        registry.record("gpu_temperature", random.uniform(65, 85))
        registry.record("request_latency_ms", random.uniform(200, 3000))
        registry.record("ttft_ms", random.uniform(50, 600))
        registry.record("tokens_per_second", random.uniform(80, 150))
        registry.record("model_accuracy", random.uniform(82, 95))
        registry.record("data_drift_score", random.uniform(0.01, 0.15))
        registry.record("hallucination_rate", random.uniform(3, 15))
        registry.record("user_satisfaction", random.uniform(75, 95))
        registry.record("task_completion_rate", random.uniform(80, 96))
        registry.record("cost_per_request", random.uniform(0.001, 0.01))

    print(registry.get_dashboard_data())


if __name__ == "__main__":
    demo_metrics()
```

---

## Prometheus指标采集

### Prometheus采集架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Prometheus 指标采集架构                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  应用暴露指标端点                       Prometheus Server                    │
│  ┌──────────────────┐                  ┌────────────────────────┐          │
│  │ vLLM Service     │  /metrics        │                        │          │
│  │ :8000/metrics    │<─────── pull ────│  Scrape Config         │          │
│  └──────────────────┘                  │  scrape_interval: 15s  │          │
│  ┌──────────────────┐                  │                        │          │
│  │ FastAPI App      │  /metrics        │  ┌──────────────────┐  │          │
│  │ :8080/metrics    │<─────── pull ────│  │  TSDB            │  │          │
│  └──────────────────┘                  │  │  时序数据库       │  │          │
│  ┌──────────────────┐                  │  │  15天数据保留     │  │          │
│  │ Node Exporter    │  /metrics        │  └──────────────────┘  │          │
│  │ :9100/metrics    │<─────── pull ────│                        │          │
│  └──────────────────┘                  │  ┌──────────────────┐  │          │
│  ┌──────────────────┐                  │  │  Alert Rules     │  │          │
│  │ NVIDIA DCGM      │  /metrics        │  │  告警规则引擎     │  │          │
│  │ :9400/metrics    │<─────── pull ────│  └────────┬─────────┘  │          │
│  └──────────────────┘                  └───────────┼────────────┘          │
│                                                    │                       │
│                                                    ▼                       │
│                                        ┌────────────────────────┐          │
│                                        │  Alertmanager          │          │
│                                        │  :9093                 │          │
│                                        │  告警路由/去重/静默     │          │
│                                        └────────────────────────┘          │
│                                                                             │
│  PromQL查询示例:                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ # 请求QPS                                                          │   │
│  │ rate(llm_requests_total[5m])                                       │   │
│  │                                                                     │   │
│  │ # P95延迟                                                           │   │
│  │ histogram_quantile(0.95, rate(llm_latency_bucket[5m]))             │   │
│  │                                                                     │   │
│  │ # 错误率                                                           │   │
│  │ rate(llm_errors_total[5m]) / rate(llm_requests_total[5m]) * 100   │   │
│  │                                                                     │   │
│  │ # GPU显存使用率                                                     │   │
│  │ gpu_memory_used_bytes / gpu_memory_total_bytes * 100               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Prometheus指标代码

```python
"""
Prometheus风格的指标暴露
支持: Counter、Gauge、Histogram、Summary
可直接与Prometheus集成
"""

import time
import math
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


class PrometheusCounter:
    """Prometheus Counter (只增不减)"""

    def __init__(self, name: str, description: str, labels: List[str] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[Tuple, float] = defaultdict(float)

    def inc(self, value: float = 1, labels: Dict[str, str] = None):
        key = tuple(sorted((labels or {}).items()))
        self._values[key] += value

    def get(self, labels: Dict[str, str] = None) -> float:
        key = tuple(sorted((labels or {}).items()))
        return self._values[key]

    def to_prometheus(self) -> str:
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} counter",
        ]
        for label_tuple, value in self._values.items():
            label_str = ",".join(f'{k}="{v}"' for k, v in label_tuple)
            if label_str:
                lines.append(f"{self.name}{{{label_str}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        return "\n".join(lines)


class PrometheusGauge:
    """Prometheus Gauge (可增可减)"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._value = 0.0

    def set(self, value: float):
        self._value = value

    def inc(self, value: float = 1):
        self._value += value

    def dec(self, value: float = 1):
        self._value -= value

    def get(self) -> float:
        return self._value

    def to_prometheus(self) -> str:
        return (
            f"# HELP {self.name} {self.description}\n"
            f"# TYPE {self.name} gauge\n"
            f"{self.name} {self._value}"
        )


class PrometheusHistogram:
    """Prometheus Histogram (分布)"""

    def __init__(self, name: str, description: str,
                 buckets: List[float] = None):
        self.name = name
        self.description = description
        self.buckets = sorted(buckets or [
            10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000
        ])
        self._bucket_counts: Dict[float, int] = {b: 0 for b in self.buckets}
        self._bucket_counts[float("inf")] = 0
        self._sum = 0.0
        self._count = 0

    def observe(self, value: float):
        self._sum += value
        self._count += 1
        for bucket in self.buckets:
            if value <= bucket:
                self._bucket_counts[bucket] += 1
        self._bucket_counts[float("inf")] += 1

    def to_prometheus(self) -> str:
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram",
        ]
        cumulative = 0
        for bucket in self.buckets:
            cumulative += self._bucket_counts[bucket]
            lines.append(f'{self.name}_bucket{{le="{bucket}"}} {cumulative}')
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {self._count}')
        lines.append(f"{self.name}_sum {self._sum}")
        lines.append(f"{self.name}_count {self._count}")
        return "\n".join(lines)


class LLMMetricsExporter:
    """
    LLM服务Prometheus指标导出器
    暴露 /metrics 端点供Prometheus抓取
    """

    def __init__(self):
        # 应用指标
        self.request_total = PrometheusCounter(
            "llm_requests_total", "Total LLM requests",
            labels=["model", "status"],
        )
        self.request_latency = PrometheusHistogram(
            "llm_request_latency_ms", "Request latency in milliseconds",
            buckets=[50, 100, 200, 500, 1000, 2000, 5000, 10000],
        )
        self.ttft = PrometheusHistogram(
            "llm_ttft_ms", "Time to first token in milliseconds",
            buckets=[20, 50, 100, 200, 500, 1000],
        )
        self.tokens_generated = PrometheusCounter(
            "llm_tokens_generated_total", "Total tokens generated",
        )
        self.active_requests = PrometheusGauge(
            "llm_active_requests", "Currently active requests",
        )

        # GPU指标
        self.gpu_utilization = PrometheusGauge(
            "llm_gpu_utilization_percent", "GPU utilization percentage",
        )
        self.gpu_memory = PrometheusGauge(
            "llm_gpu_memory_used_gb", "GPU memory used in GB",
        )

        # 模型指标
        self.model_accuracy = PrometheusGauge(
            "llm_model_accuracy_percent", "Model accuracy percentage",
        )

    def record_request(self, model: str, status: str,
                       latency_ms: float, ttft_ms: float,
                       tokens: int):
        """记录一次请求的所有指标"""
        self.request_total.inc(labels={"model": model, "status": status})
        self.request_latency.observe(latency_ms)
        self.ttft.observe(ttft_ms)
        self.tokens_generated.inc(tokens)

    def get_metrics_text(self) -> str:
        """输出Prometheus格式的指标文本"""
        parts = [
            self.request_total.to_prometheus(),
            self.request_latency.to_prometheus(),
            self.ttft.to_prometheus(),
            self.tokens_generated.to_prometheus(),
            self.active_requests.to_prometheus(),
            self.gpu_utilization.to_prometheus(),
            self.gpu_memory.to_prometheus(),
            self.model_accuracy.to_prometheus(),
        ]
        return "\n\n".join(parts) + "\n"

    def generate_prometheus_config(self) -> str:
        """生成Prometheus配置文件"""
        return """# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

scrape_configs:
  # LLM服务指标
  - job_name: 'llm-service'
    static_configs:
      - targets: ['vllm-server:8000']
    metrics_path: /metrics
    scrape_interval: 10s

  # FastAPI应用指标
  - job_name: 'llm-api'
    static_configs:
      - targets: ['api-server:8080']

  # GPU指标 (NVIDIA DCGM Exporter)
  - job_name: 'gpu-metrics'
    static_configs:
      - targets: ['dcgm-exporter:9400']

  # 节点指标
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
"""


def demo_prometheus():
    """Prometheus指标演示"""
    exporter = LLMMetricsExporter()
    import random
    random.seed(42)

    # 模拟请求
    print("=" * 60)
    print("模拟100个LLM请求...")
    print("=" * 60)

    for i in range(100):
        status = "success" if random.random() > 0.05 else "error"
        latency = random.uniform(200, 3000)
        ttft = random.uniform(30, 500)
        tokens = random.randint(50, 500)

        exporter.active_requests.inc()
        exporter.record_request(
            model="llama-8b", status=status,
            latency_ms=latency, ttft_ms=ttft, tokens=tokens,
        )
        exporter.active_requests.dec()

    # 设置GPU指标
    exporter.gpu_utilization.set(78.5)
    exporter.gpu_memory.set(52.3)
    exporter.model_accuracy.set(91.2)

    # 输出指标
    print("\n--- Prometheus 指标输出 ---")
    print(exporter.get_metrics_text()[:1500] + "\n... (截断)")

    # 输出配置
    print("\n--- Prometheus 配置 ---")
    print(exporter.generate_prometheus_config())


if __name__ == "__main__":
    demo_prometheus()
```

---

## Grafana可视化

### Grafana仪表盘配置

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Grafana LLM监控仪表盘布局                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Row 1: 概览 (Overview)                                                     │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐             │
│  │ 当前QPS    │ │ 平均延迟   │ │ 错误率     │ │ GPU利用率  │             │
│  │   ┌───┐   │ │   ┌───┐   │ │   ┌───┐   │ │   ┌───┐   │             │
│  │   │256│   │ │   │1.2s│   │ │   │2.1%│   │ │   │78%│   │             │
│  │   └───┘   │ │   └───┘   │ │   └───┘   │ │   └───┘   │             │
│  │  Stat Panel│ │  Stat Panel│ │  Stat Panel│ │  Stat Panel│             │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘             │
│                                                                             │
│  Row 2: 延迟分析 (Latency)                                                 │
│  ┌──────────────────────────────┐ ┌──────────────────────────────┐        │
│  │ 请求延迟 P50/P95/P99        │ │ TTFT分布                     │        │
│  │  ms                          │ │  ms                          │        │
│  │ 5000│                        │ │ 1000│                        │        │
│  │     │   ╱──P99               │ │     │   ╱──P99               │        │
│  │ 2000│  ╱──P95                │ │  500│  ╱──P95                │        │
│  │     │ ╱──P50                 │ │     │ ╱──P50                 │        │
│  │    0│╱────────────── time    │ │    0│╱────────────── time    │        │
│  │     Time Series Panel        │ │     Time Series Panel        │        │
│  └──────────────────────────────┘ └──────────────────────────────┘        │
│                                                                             │
│  Row 3: 吞吐量与GPU (Throughput & GPU)                                     │
│  ┌──────────────────────────────┐ ┌──────────────────────────────┐        │
│  │ Tokens/s 吞吐量             │ │ GPU利用率 & 显存              │        │
│  │ t/s                          │ │  %                           │        │
│  │ 3000│  ╱─╲                   │ │ 100│ ──GPU Util              │        │
│  │     │ ╱   ╲─────            │ │  80│╱────╲───                │        │
│  │ 1000│╱          ╲           │ │  60│  GPU Mem ──────         │        │
│  │    0│────────────── time    │ │    0│────────────── time     │        │
│  └──────────────────────────────┘ └──────────────────────────────┘        │
│                                                                             │
│  Row 4: 模型质量 (Model Quality)                                           │
│  ┌──────────────────────────────┐ ┌──────────────────────────────┐        │
│  │ 模型准确率趋势               │ │ 数据漂移评分 (PSI)           │        │
│  │  %                           │ │                              │        │
│  │ 95 │────────╲               │ │ 0.2│     ╱─╲ Critical       │        │
│  │ 90 │         ╲──────        │ │ 0.1│────╱   ╲── Warning     │        │
│  │ 85 │         ╱ threshold    │ │    0│────────────── time     │        │
│  │    0│────────────── time    │ │                              │        │
│  └──────────────────────────────┘ └──────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Grafana仪表盘代码

```python
"""
Grafana仪表盘配置生成器
生成JSON格式的Grafana Dashboard配置
"""

import json
from typing import Dict, List


class GrafanaDashboardGenerator:
    """生成Grafana仪表盘JSON配置"""

    @staticmethod
    def generate_llm_dashboard() -> Dict:
        """生成LLM监控仪表盘"""
        dashboard = {
            "dashboard": {
                "title": "LLM Service Monitor",
                "tags": ["llm", "ai", "monitoring"],
                "timezone": "browser",
                "refresh": "10s",
                "panels": [],
            }
        }

        panels = dashboard["dashboard"]["panels"]
        y_pos = 0

        # Row 1: 概览统计
        stat_panels = [
            ("当前QPS", 'rate(llm_requests_total[5m])', "requests/s"),
            ("平均延迟", 'avg(llm_request_latency_ms)', "ms"),
            ("错误率",
             'rate(llm_requests_total{status="error"}[5m])'
             '/rate(llm_requests_total[5m])*100', "%"),
            ("GPU利用率", 'llm_gpu_utilization_percent', "%"),
        ]
        for i, (title, expr, unit) in enumerate(stat_panels):
            panels.append({
                "type": "stat",
                "title": title,
                "gridPos": {"h": 4, "w": 6, "x": i * 6, "y": y_pos},
                "targets": [{"expr": expr}],
                "fieldConfig": {"defaults": {"unit": unit}},
            })
        y_pos += 4

        # Row 2: 延迟时序图
        panels.append({
            "type": "timeseries",
            "title": "请求延迟 P50/P95/P99",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": y_pos},
            "targets": [
                {"expr": 'histogram_quantile(0.5, rate(llm_request_latency_ms_bucket[5m]))',
                 "legendFormat": "P50"},
                {"expr": 'histogram_quantile(0.95, rate(llm_request_latency_ms_bucket[5m]))',
                 "legendFormat": "P95"},
                {"expr": 'histogram_quantile(0.99, rate(llm_request_latency_ms_bucket[5m]))',
                 "legendFormat": "P99"},
            ],
            "fieldConfig": {"defaults": {"unit": "ms"}},
        })
        panels.append({
            "type": "timeseries",
            "title": "TTFT分布 (首Token延迟)",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": y_pos},
            "targets": [
                {"expr": 'histogram_quantile(0.5, rate(llm_ttft_ms_bucket[5m]))',
                 "legendFormat": "P50"},
                {"expr": 'histogram_quantile(0.95, rate(llm_ttft_ms_bucket[5m]))',
                 "legendFormat": "P95"},
            ],
            "fieldConfig": {"defaults": {"unit": "ms"}},
        })
        y_pos += 8

        # Row 3: GPU监控
        panels.append({
            "type": "timeseries",
            "title": "GPU利用率 & 显存使用",
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": y_pos},
            "targets": [
                {"expr": "llm_gpu_utilization_percent",
                 "legendFormat": "GPU利用率"},
                {"expr": "llm_gpu_memory_used_gb / 80 * 100",
                 "legendFormat": "显存使用率"},
            ],
            "fieldConfig": {"defaults": {"unit": "percent", "max": 100}},
        })
        panels.append({
            "type": "timeseries",
            "title": "生成吞吐量 (tokens/s)",
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": y_pos},
            "targets": [
                {"expr": 'rate(llm_tokens_generated_total[5m])',
                 "legendFormat": "tokens/s"},
            ],
        })

        return dashboard

    @staticmethod
    def generate_docker_compose() -> str:
        """生成监控栈docker-compose"""
        return """# docker-compose-monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./alert_rules.yml:/etc/prometheus/alert_rules.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=15d'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml

  node-exporter:
    image: prom/node-exporter:latest
    ports:
      - "9100:9100"

volumes:
  prometheus_data:
  grafana_data:
"""


def demo_grafana():
    """Grafana仪表盘生成演示"""
    generator = GrafanaDashboardGenerator()

    dashboard = generator.generate_llm_dashboard()
    print("=" * 60)
    print("Grafana LLM Dashboard JSON (部分):")
    print("=" * 60)
    print(json.dumps(dashboard, indent=2, ensure_ascii=False)[:2000])
    print("\n... (截断)")

    print("\n--- Docker Compose监控栈 ---")
    print(generator.generate_docker_compose())


if __name__ == "__main__":
    demo_grafana()
```

---

## LLM专用监控指标

### LLM质量监控

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LLM专用监控指标体系                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  性能指标                          质量指标                                 │
│  ┌──────────────────────────┐     ┌──────────────────────────┐            │
│  │ TTFT (Time To First     │     │ 幻觉率                   │            │
│  │   Token)                │     │   - 事实性错误比例       │            │
│  │   目标: <200ms          │     │   - 阈值: <10%           │            │
│  │                         │     │                          │            │
│  │ TPOT (Time Per Output   │     │ 拒答率                   │            │
│  │   Token)                │     │   - 不当拒绝比例         │            │
│  │   目标: <50ms           │     │   - 阈值: <5%            │            │
│  │                         │     │                          │            │
│  │ TPS (Tokens Per Second) │     │ 指令遵循度               │            │
│  │   全系统吞吐量           │     │   - 格式/约束遵守率     │            │
│  │                         │     │   - 阈值: >90%           │            │
│  │ 并发请求数               │     │                          │            │
│  │   同时处理的请求          │     │ 输出多样性               │            │
│  │                         │     │   - 重复/模板化输出率    │            │
│  └──────────────────────────┘     └──────────────────────────┘            │
│                                                                             │
│  成本指标                          安全指标                                 │
│  ┌──────────────────────────┐     ┌──────────────────────────┐            │
│  │ Token消耗量               │     │ 有害内容检出率           │            │
│  │   - 输入token数           │     │   - 安全过滤触发频率     │            │
│  │   - 输出token数           │     │                          │            │
│  │                          │     │ PII泄露检测              │            │
│  │ 每请求成本                │     │   - 个人信息泄露率       │            │
│  │   - 按模型/用户统计      │     │                          │            │
│  │                          │     │ 越狱检测                 │            │
│  │ 缓存命中率               │     │   - 提示注入攻击频率     │            │
│  │   - 语义缓存有效性       │     │                          │            │
│  └──────────────────────────┘     └──────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### LLM质量监控代码

```python
"""
LLM专用质量监控
包含: 幻觉检测、数据漂移、输出质量评估
"""

import time
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class LLMRequest:
    """LLM请求记录"""
    request_id: str
    prompt: str
    response: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    ttft_ms: float
    timestamp: float = 0


class LLMQualityMonitor:
    """
    LLM输出质量监控
    检测: 幻觉、拒答、指令遵循、输出多样性
    """

    def __init__(self):
        self.request_log: List[LLMRequest] = []
        self.quality_scores: List[Dict] = []
        # 简单关键词检测(实际生产中使用专门的检测模型)
        self.refusal_patterns = [
            "我无法", "我不能", "作为AI", "我没有能力",
            "I cannot", "I'm unable", "As an AI",
        ]
        self.uncertainty_patterns = [
            "可能", "也许", "不确定", "我认为", "据我所知",
        ]

    def evaluate_response(self, request: LLMRequest) -> Dict:
        """评估单次响应的质量"""
        request.timestamp = time.time()
        self.request_log.append(request)

        scores = {
            "request_id": request.request_id,
            "timestamp": request.timestamp,
            "response_length": len(request.response),
            "is_refusal": self._check_refusal(request.response),
            "uncertainty_score": self._check_uncertainty(request.response),
            "repetition_score": self._check_repetition(request.response),
            "format_compliance": self._check_format(request.prompt, request.response),
            "response_relevance": self._check_relevance(request.prompt, request.response),
        }
        self.quality_scores.append(scores)
        return scores

    def _check_refusal(self, response: str) -> bool:
        """检查是否是拒答"""
        return any(p in response for p in self.refusal_patterns)

    def _check_uncertainty(self, response: str) -> float:
        """检查不确定性程度 (0-1)"""
        count = sum(1 for p in self.uncertainty_patterns if p in response)
        return min(1.0, count / 3)

    def _check_repetition(self, response: str) -> float:
        """检查重复率 (0-1)"""
        words = response.split()
        if len(words) < 10:
            return 0
        # N-gram重复率
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        unique_ratio = len(set(trigrams)) / len(trigrams) if trigrams else 1
        return 1 - unique_ratio

    def _check_format(self, prompt: str, response: str) -> float:
        """检查格式遵循度 (0-1)"""
        score = 1.0
        if "JSON" in prompt and "{" not in response:
            score -= 0.5
        if "列表" in prompt and ("1." not in response and "-" not in response):
            score -= 0.3
        if "代码" in prompt and "```" not in response and "def " not in response:
            score -= 0.3
        return max(0, score)

    def _check_relevance(self, prompt: str, response: str) -> float:
        """检查响应相关性 (简单词重叠, 0-1)"""
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        if not prompt_words:
            return 0
        overlap = len(prompt_words & response_words)
        return min(1.0, overlap / max(5, len(prompt_words) * 0.3))

    def get_quality_report(self, minutes: int = 60) -> str:
        """生成质量报告"""
        cutoff = time.time() - minutes * 60
        recent = [s for s in self.quality_scores if s["timestamp"] > cutoff]

        if not recent:
            return "无数据"

        total = len(recent)
        refusal_count = sum(1 for s in recent if s["is_refusal"])
        avg_uncertainty = sum(s["uncertainty_score"] for s in recent) / total
        avg_repetition = sum(s["repetition_score"] for s in recent) / total
        avg_format = sum(s["format_compliance"] for s in recent) / total
        avg_relevance = sum(s["response_relevance"] for s in recent) / total

        lines = [
            f"{'=' * 50}",
            f"LLM质量监控报告 (最近{minutes}分钟)",
            f"{'=' * 50}",
            f"总请求数: {total}",
            f"拒答率:   {refusal_count/total:.1%} ({refusal_count}/{total})",
            f"不确定性: {avg_uncertainty:.2f} (0=确定, 1=高度不确定)",
            f"重复率:   {avg_repetition:.2f} (0=无重复, 1=高度重复)",
            f"格式遵循: {avg_format:.2f} (1=完全遵循)",
            f"内容相关: {avg_relevance:.2f} (1=高度相关)",
            f"{'─' * 50}",
            f"综合质量评分: {(avg_format + avg_relevance + (1-avg_repetition)) / 3:.2f}",
        ]
        return "\n".join(lines)


class DataDriftDetector:
    """数据漂移检测器"""

    def __init__(self, window_size: int = 1000):
        self.reference_distribution: Dict[str, float] = {}
        self.current_samples: List[Dict] = []
        self.window_size = window_size

    def set_reference(self, distribution: Dict[str, float]):
        """设置参考分布"""
        self.reference_distribution = distribution
        print(f"[漂移检测] 参考分布已设置: {len(distribution)}个特征")

    def add_sample(self, features: Dict[str, float]):
        """添加新样本"""
        self.current_samples.append(features)
        if len(self.current_samples) > self.window_size:
            self.current_samples = self.current_samples[-self.window_size:]

    def compute_psi(self, feature: str) -> float:
        """计算PSI (Population Stability Index)"""
        if feature not in self.reference_distribution:
            return 0

        ref_val = self.reference_distribution[feature]
        if not self.current_samples:
            return 0

        current_vals = [
            s.get(feature, 0) for s in self.current_samples
        ]
        current_avg = sum(current_vals) / len(current_vals)

        # 简化PSI计算
        if ref_val == 0 or current_avg == 0:
            return 0
        ratio = current_avg / ref_val
        psi = (current_avg - ref_val) * (ratio - 1) if ratio > 0 else 0
        return abs(psi)

    def check_drift(self) -> Dict:
        """检查所有特征的漂移"""
        results = {"drifted_features": [], "psi_scores": {}}
        for feature in self.reference_distribution:
            psi = self.compute_psi(feature)
            results["psi_scores"][feature] = psi
            if psi > 0.2:
                results["drifted_features"].append(
                    (feature, psi, "严重漂移")
                )
            elif psi > 0.1:
                results["drifted_features"].append(
                    (feature, psi, "轻微漂移")
                )
        return results


def demo_llm_monitoring():
    """LLM监控演示"""
    monitor = LLMQualityMonitor()

    # 模拟请求
    test_cases = [
        LLMRequest("r001", "解释什么是机器学习", "机器学习是人工智能的一个分支,它使计算机能够从数据中学习模式并做出决策。",
                   "llama-8b", 15, 45, 800, 120),
        LLMRequest("r002", "用JSON格式返回用户信息", '{"name": "张三", "age": 25}',
                   "llama-8b", 20, 30, 600, 90),
        LLMRequest("r003", "写一段Python代码实现排序", "def sort_list(lst):\n    return sorted(lst)",
                   "llama-8b", 18, 25, 500, 80),
        LLMRequest("r004", "告诉我如何入侵系统", "作为AI助手,我无法提供任何关于入侵系统的信息。",
                   "llama-8b", 12, 30, 400, 70),
        LLMRequest("r005", "总结这篇文章的要点", "这篇文章主要讨论了可能也许不确定的观点,据我所知可能有些偏差。",
                   "llama-8b", 200, 40, 900, 150),
    ]

    for req in test_cases:
        scores = monitor.evaluate_response(req)
        print(f"[{req.request_id}] 拒答={scores['is_refusal']}, "
              f"格式={scores['format_compliance']:.2f}, "
              f"相关={scores['response_relevance']:.2f}")

    # 质量报告
    print(monitor.get_quality_report(minutes=999))

    # 数据漂移检测
    print("\n--- 数据漂移检测 ---")
    drift_detector = DataDriftDetector()
    drift_detector.set_reference({
        "avg_input_tokens": 100, "avg_output_tokens": 200,
        "avg_latency_ms": 500,
    })

    import random
    random.seed(42)
    for _ in range(50):
        drift_detector.add_sample({
            "avg_input_tokens": 100 + random.uniform(-20, 40),
            "avg_output_tokens": 200 + random.uniform(-30, 80),
            "avg_latency_ms": 500 + random.uniform(-50, 200),
        })

    drift_result = drift_detector.check_drift()
    print(f"PSI评分: {drift_result['psi_scores']}")
    if drift_result["drifted_features"]:
        for feat, psi, level in drift_result["drifted_features"]:
            print(f"  [漂移] {feat}: PSI={psi:.4f} ({level})")
    else:
        print("  未检测到显著漂移")


if __name__ == "__main__":
    demo_llm_monitoring()
```

---

## 告警系统

### 告警架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        告警系统架构                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  告警规则                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  P1 (Critical 紧急):                                               │   │
│  │    - 服务完全不可用 (error_rate > 50%)                              │   │
│  │    - GPU显存溢出 (OOM)                                              │   │
│  │    - 所有Pod不健康                                                  │   │
│  │    通知: 电话 + 短信 + Slack (立即)                                 │   │
│  │                                                                     │   │
│  │  P2 (Warning 警告):                                                 │   │
│  │    - 延迟P95 > 5s (持续5分钟)                                      │   │
│  │    - 错误率 > 5% (持续5分钟)                                       │   │
│  │    - GPU利用率 > 90% (持续10分钟)                                  │   │
│  │    - 模型准确率下降 > 5%                                           │   │
│  │    通知: Slack + 邮件 (5分钟内)                                    │   │
│  │                                                                     │   │
│  │  P3 (Info 提示):                                                    │   │
│  │    - 数据漂移PSI > 0.1                                             │   │
│  │    - 缓存命中率 < 30%                                              │   │
│  │    - 成本超出预算80%                                                │   │
│  │    通知: Slack (工作时间内)                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  告警流程:                                                                  │
│  ┌──────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────┐       │
│  │ 规则 │──>│  触发    │──>│  分组    │──>│  静默    │──>│ 通知 │       │
│  │ 评估 │   │  判断    │   │  去重    │   │  检查    │   │ 发送 │       │
│  └──────┘   └──────────┘   └──────────┘   └──────────┘   └──────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 告警系统代码

```python
"""
AI系统告警管理
支持: 多级告警、去重、静默、通知路由
"""

import time
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum


class AlertSeverity(Enum):
    CRITICAL = "critical"  # P1
    WARNING = "warning"    # P2
    INFO = "info"          # P3


class AlertState(Enum):
    FIRING = "firing"
    RESOLVED = "resolved"
    SILENCED = "silenced"


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    severity: AlertSeverity
    condition: str  # 描述
    check_func: Callable[[Dict], bool]  # 检查函数
    message_template: str
    for_duration: int = 300  # 持续时间(秒)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """告警实例"""
    rule_name: str
    severity: AlertSeverity
    state: AlertState
    message: str
    first_fired: float
    last_fired: float
    resolved_at: float = 0
    notification_sent: bool = False
    labels: Dict[str, str] = field(default_factory=dict)


class AlertManager:
    """
    告警管理器
    支持: 规则评估、去重、静默、通知路由
    """

    def __init__(self):
        self.rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.silences: Dict[str, float] = {}  # name -> until_timestamp
        self.notification_channels: Dict[str, List[Callable]] = {
            "critical": [],
            "warning": [],
            "info": [],
        }

    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.rules.append(rule)

    def add_notification_channel(self, severity: str,
                                 handler: Callable):
        """添加通知渠道"""
        if severity in self.notification_channels:
            self.notification_channels[severity].append(handler)

    def silence(self, rule_name: str, duration_seconds: int):
        """静默某条告警"""
        self.silences[rule_name] = time.time() + duration_seconds
        print(f"[告警] 已静默: {rule_name} ({duration_seconds}秒)")

    def evaluate(self, metrics: Dict) -> List[Alert]:
        """评估所有规则"""
        new_alerts = []
        now = time.time()

        for rule in self.rules:
            is_firing = rule.check_func(metrics)

            if is_firing:
                if rule.name in self.active_alerts:
                    # 更新已有告警
                    alert = self.active_alerts[rule.name]
                    alert.last_fired = now
                else:
                    # 创建新告警
                    alert = Alert(
                        rule_name=rule.name,
                        severity=rule.severity,
                        state=AlertState.FIRING,
                        message=rule.message_template.format(**metrics),
                        first_fired=now,
                        last_fired=now,
                        labels=rule.labels,
                    )
                    self.active_alerts[rule.name] = alert
                    new_alerts.append(alert)

                    # 检查静默
                    if rule.name in self.silences:
                        if now < self.silences[rule.name]:
                            alert.state = AlertState.SILENCED
                            continue

                    # 发送通知
                    self._notify(alert)
            else:
                # 告警恢复
                if rule.name in self.active_alerts:
                    alert = self.active_alerts.pop(rule.name)
                    alert.state = AlertState.RESOLVED
                    alert.resolved_at = now
                    self.alert_history.append(alert)
                    print(f"[告警恢复] {rule.name}")

        return new_alerts

    def _notify(self, alert: Alert):
        """发送告警通知"""
        if alert.notification_sent:
            return
        if alert.state == AlertState.SILENCED:
            return

        severity = alert.severity.value
        handlers = self.notification_channels.get(severity, [])
        for handler in handlers:
            handler(alert)
        alert.notification_sent = True

    def get_status(self) -> str:
        """获取告警状态"""
        lines = [
            f"{'=' * 60}",
            f"告警管理器状态",
            f"{'=' * 60}",
            f"活跃告警: {len(self.active_alerts)}",
            f"历史告警: {len(self.alert_history)}",
        ]

        if self.active_alerts:
            lines.append(f"\n活跃告警列表:")
            for name, alert in self.active_alerts.items():
                duration = time.time() - alert.first_fired
                lines.append(
                    f"  [{alert.severity.value.upper()}] {name} "
                    f"({alert.state.value}) 持续{duration:.0f}秒"
                )
                lines.append(f"    {alert.message}")

        return "\n".join(lines)

    def generate_alertmanager_config(self) -> str:
        """生成Alertmanager配置"""
        return """# alertmanager.yml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'critical-team'
      repeat_interval: 15m
    - match:
        severity: warning
      receiver: 'warning-team'
      repeat_interval: 1h

receivers:
  - name: 'default'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#ai-monitoring'

  - name: 'critical-team'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#ai-alerts-critical'
    pagerduty_configs:
      - service_key: 'your-pagerduty-key'

  - name: 'warning-team'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#ai-alerts-warning'
"""

    def generate_alert_rules(self) -> str:
        """生成Prometheus告警规则"""
        return """# alert_rules.yml
groups:
  - name: llm_alerts
    rules:
      # P1: 服务不可用
      - alert: LLMServiceDown
        expr: up{job="llm-service"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "LLM服务不可用"
          description: "LLM服务 {{ $labels.instance }} 已停止响应超过1分钟"

      # P2: 高延迟
      - alert: LLMHighLatency
        expr: histogram_quantile(0.95, rate(llm_request_latency_ms_bucket[5m])) > 5000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "LLM推理延迟过高"
          description: "P95延迟 {{ $value }}ms 超过5000ms阈值"

      # P2: 高错误率
      - alert: LLMHighErrorRate
        expr: rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "LLM服务错误率过高"
          description: "错误率 {{ $value | humanizePercentage }} 超过5%阈值"

      # P2: GPU显存高
      - alert: GPUMemoryHigh
        expr: llm_gpu_memory_used_gb / 80 > 0.9
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "GPU显存使用超过90%"

      # P3: 数据漂移
      - alert: DataDriftDetected
        expr: llm_data_drift_psi > 0.1
        for: 30m
        labels:
          severity: info
        annotations:
          summary: "检测到数据漂移"
          description: "PSI值 {{ $value }} 超过0.1阈值"
"""


def demo_alerts():
    """告警系统演示"""
    manager = AlertManager()

    # 定义告警规则
    manager.add_rule(AlertRule(
        name="高延迟", severity=AlertSeverity.WARNING,
        condition="P95延迟 > 2000ms",
        check_func=lambda m: m.get("latency_p95", 0) > 2000,
        message_template="P95延迟 {latency_p95:.0f}ms 超过阈值2000ms",
    ))
    manager.add_rule(AlertRule(
        name="高错误率", severity=AlertSeverity.CRITICAL,
        condition="错误率 > 10%",
        check_func=lambda m: m.get("error_rate", 0) > 0.1,
        message_template="错误率 {error_rate:.1%} 超过阈值10%",
    ))
    manager.add_rule(AlertRule(
        name="GPU显存高", severity=AlertSeverity.WARNING,
        condition="GPU显存 > 90%",
        check_func=lambda m: m.get("gpu_memory_pct", 0) > 90,
        message_template="GPU显存使用 {gpu_memory_pct:.0f}% 超过90%",
    ))

    # 添加通知渠道
    def slack_notify(alert: Alert):
        print(f"  [Slack通知] [{alert.severity.value}] {alert.message}")

    def pagerduty_notify(alert: Alert):
        print(f"  [PagerDuty] [{alert.severity.value}] {alert.message}")

    manager.add_notification_channel("critical", pagerduty_notify)
    manager.add_notification_channel("critical", slack_notify)
    manager.add_notification_channel("warning", slack_notify)

    # 模拟指标变化
    scenarios = [
        {"name": "正常", "latency_p95": 800, "error_rate": 0.02, "gpu_memory_pct": 70},
        {"name": "延迟升高", "latency_p95": 3500, "error_rate": 0.03, "gpu_memory_pct": 75},
        {"name": "故障", "latency_p95": 8000, "error_rate": 0.15, "gpu_memory_pct": 95},
        {"name": "恢复中", "latency_p95": 1500, "error_rate": 0.04, "gpu_memory_pct": 80},
        {"name": "恢复", "latency_p95": 500, "error_rate": 0.01, "gpu_memory_pct": 65},
    ]

    for scenario in scenarios:
        name = scenario.pop("name")
        print(f"\n--- 场景: {name} ---")
        alerts = manager.evaluate(scenario)
        if not alerts and not manager.active_alerts:
            print("  一切正常, 无活跃告警")
        scenario["name"] = name

    print(f"\n{manager.get_status()}")
    print(f"\n--- Prometheus告警规则 ---")
    print(manager.generate_alert_rules()[:1000] + "\n... (截断)")


if __name__ == "__main__":
    demo_alerts()
```

---

## 日志收集与分析

### 结构化日志

```python
"""
AI服务结构化日志系统
支持: JSON格式日志、请求追踪、性能日志
"""

import json
import time
import uuid
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class LLMLogEntry:
    """LLM请求日志条目"""
    timestamp: str
    level: str
    request_id: str
    event: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0
    ttft_ms: float = 0
    status: str = ""
    error: str = ""
    user_id: str = ""
    cost_usd: float = 0
    extra: Dict[str, Any] = None


class StructuredLogger:
    """结构化日志记录器"""

    def __init__(self, service_name: str = "llm-service"):
        self.service_name = service_name
        self.logs: list = []

    def _log(self, level: str, event: str, **kwargs) -> LLMLogEntry:
        entry = LLMLogEntry(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            level=level,
            request_id=kwargs.get("request_id", ""),
            event=event,
            **{k: v for k, v in kwargs.items() if k != "request_id"},
        )
        log_dict = {k: v for k, v in asdict(entry).items() if v}
        log_dict["service"] = self.service_name
        self.logs.append(log_dict)
        print(json.dumps(log_dict, ensure_ascii=False))
        return entry

    def info(self, event: str, **kwargs):
        return self._log("INFO", event, **kwargs)

    def warning(self, event: str, **kwargs):
        return self._log("WARNING", event, **kwargs)

    def error(self, event: str, **kwargs):
        return self._log("ERROR", event, **kwargs)

    def log_request(self, request_id: str, model: str,
                    input_tokens: int, output_tokens: int,
                    latency_ms: float, ttft_ms: float,
                    status: str, cost_usd: float = 0,
                    user_id: str = "", error: str = ""):
        """记录完整的LLM请求日志"""
        level = "INFO" if status == "success" else "ERROR"
        self._log(
            level, "llm_request_completed",
            request_id=request_id, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            latency_ms=latency_ms, ttft_ms=ttft_ms,
            status=status, cost_usd=cost_usd,
            user_id=user_id, error=error,
        )


def demo_logging():
    """日志系统演示"""
    logger = StructuredLogger("llm-api-v1")

    logger.info("service_started", extra={"version": "1.0.0", "gpu": "A100"})

    # 模拟请求日志
    for i in range(3):
        req_id = f"req-{uuid.uuid4().hex[:8]}"
        logger.log_request(
            request_id=req_id, model="llama-8b",
            input_tokens=150, output_tokens=200,
            latency_ms=800 + i * 100, ttft_ms=120,
            status="success", cost_usd=0.003,
            user_id=f"user_{i}",
        )

    # 错误日志
    logger.log_request(
        request_id="req-err001", model="llama-8b",
        input_tokens=500, output_tokens=0,
        latency_ms=5000, ttft_ms=0,
        status="error", error="GPU OOM: out of memory",
    )

    logger.warning("gpu_memory_high", extra={"gpu_id": 0, "usage_pct": 92})


if __name__ == "__main__":
    demo_logging()
```

---

## 完整监控系统搭建

### 一键部署配置

```python
"""
完整AI监控系统一键部署
整合: Prometheus + Grafana + Alertmanager + 结构化日志
"""


def generate_full_monitoring_stack() -> str:
    """生成完整监控栈部署配置"""
    return """
# ================================================================
# 完整AI监控系统部署指南
# ================================================================

# 1. 克隆配置
mkdir -p ai-monitoring && cd ai-monitoring

# 2. 启动监控栈
docker compose -f docker-compose-monitoring.yml up -d

# 3. 访问各服务
# - Grafana:      http://localhost:3000 (admin/admin)
# - Prometheus:   http://localhost:9090
# - Alertmanager: http://localhost:9093

# 4. 导入Grafana仪表盘
# 在Grafana中导入 dashboards/llm-monitor.json

# 5. 配置告警通知
# 编辑 alertmanager.yml 中的Slack webhook地址

# 6. 验证
curl http://localhost:9090/-/healthy  # Prometheus健康
curl http://localhost:3000/api/health  # Grafana健康
"""


if __name__ == "__main__":
    print(generate_full_monitoring_stack())
```

---

## 总结

本教程涵盖了AI系统监控的核心内容:

1. **监控概述**: 三层架构(采集/存储/展示), AI系统与传统系统的监控差异
2. **指标体系**: 四层模型(基础设施/应用性能/模型质量/业务效果), 指标定义与健康评分
3. **Prometheus**: 指标采集架构, Counter/Gauge/Histogram实现, PromQL查询
4. **Grafana**: 仪表盘布局设计, JSON配置生成, docker-compose监控栈
5. **LLM专用指标**: 幻觉检测/拒答率/指令遵循度/数据漂移PSI检测
6. **告警系统**: 三级告警(P1/P2/P3), 去重/静默/通知路由, Alertmanager配置
7. **日志收集**: 结构化JSON日志, 请求追踪, 性能日志记录
8. **完整系统**: Prometheus + Grafana + Alertmanager一键部署

## 参考资源

- [Prometheus官方文档](https://prometheus.io/docs/)
- [Grafana官方文档](https://grafana.com/docs/)
- [NVIDIA DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter)
- [Evidently AI (ML监控)](https://www.evidentlyai.com/)
- [Arize AI (LLM可观测性)](https://arize.com/)
- [OpenTelemetry](https://opentelemetry.io/)

---

**创建时间**: 2024-01-15
**最后更新**: 2024-01-15
