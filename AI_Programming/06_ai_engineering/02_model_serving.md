# 模型部署与服务完整教程

## 目录
1. [部署概述](#部署概述)
2. [vLLM高性能推理](#vllm高性能推理)
3. [TGI部署实战](#tgi部署实战)
4. [FastAPI + vLLM完整服务](#fastapi--vllm完整服务)
5. [负载均衡与高可用](#负载均衡与高可用)
6. [自动扩缩容](#自动扩缩容)
7. [Docker与K8s部署](#docker与k8s部署)
8. [生产部署最佳实践](#生产部署最佳实践)

---

## 部署概述

### LLM部署技术全景

大语言模型(LLM)的部署与传统ML模型有显著区别。LLM参数量巨大(7B-405B)，
推理需要大量GPU显存，且对延迟和吞吐量要求极高。本章介绍主流的LLM部署方案。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LLM部署技术全景图                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  客户端请求                                                                 │
│  ┌──────┐ ┌──────┐ ┌──────┐                                              │
│  │ Web  │ │ API  │ │ SDK  │                                              │
│  │ App  │ │Client│ │      │                                              │
│  └──┬───┘ └──┬───┘ └──┬───┘                                              │
│     └────────┼────────┘                                                   │
│              ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     API Gateway / Load Balancer                      │   │
│  │                  (Nginx / Traefik / Kong)                           │   │
│  └───────────────────────────┬─────────────────────────────────────────┘   │
│                              │                                             │
│              ┌───────────────┼───────────────┐                            │
│              ▼               ▼               ▼                            │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐                   │
│  │  推理引擎 A   │ │  推理引擎 B   │ │  推理引擎 C   │                   │
│  │               │ │               │ │               │                   │
│  │  ┌─────────┐  │ │  ┌─────────┐  │ │  ┌─────────┐  │                   │
│  │  │  vLLM   │  │ │  │   TGI   │  │ │  │ TRT-LLM │  │                   │
│  │  │         │  │ │  │         │  │ │  │         │  │                   │
│  │  │ PagedAttn│  │ │  │FlashAttn│  │ │  │TensorRT │  │                   │
│  │  │ Cont.Bat│  │ │  │ Cont.Bat│  │ │  │ INT8/FP8│  │                   │
│  │  └─────────┘  │ │  └─────────┘  │ │  └─────────┘  │                   │
│  │  GPU: A100    │ │  GPU: A100    │ │  GPU: H100    │                   │
│  └───────────────┘ └───────────────┘ └───────────────┘                   │
│                                                                             │
│  推理引擎对比:                                                              │
│  ┌───────────┬──────────┬──────────┬──────────┬──────────┐                │
│  │ 特性      │  vLLM    │   TGI    │ TRT-LLM  │ Ollama   │                │
│  ├───────────┼──────────┼──────────┼──────────┼──────────┤                │
│  │ 开发者    │ UC伯克利 │HuggingFace│ NVIDIA  │ Ollama   │                │
│  │ 吞吐量    │ 极高     │ 高       │ 极高     │ 中       │                │
│  │ 易用性    │ 高       │ 高       │ 中       │ 极高     │                │
│  │ 模型支持  │ 广泛     │ 广泛     │ 部分     │ 广泛     │                │
│  │ 量化支持  │ AWQ/GPTQ │ AWQ/GPTQ │ INT8/FP8 │ GGUF     │                │
│  │ 适用场景  │ 生产服务 │ 生产服务 │ 极致性能 │ 本地开发 │                │
│  └───────────┴──────────┴──────────┴──────────┴──────────┘                │
│                                                                             │
│  核心优化技术:                                                              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐     │
│  │ PagedAttention│ │ Continuous  │ │ Flash        │ │ 模型量化     │     │
│  │ 显存分页管理  │ │ Batching    │ │ Attention    │ │ AWQ/GPTQ     │     │
│  │ 减少碎片化    │ │ 动态批处理  │ │ IO优化       │ │ 减少显存     │     │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 部署方案选型

```python
"""
LLM部署方案选型指南
根据需求推荐最合适的部署方案
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class DeploymentOption:
    """部署选项"""
    name: str
    description: str
    pros: List[str]
    cons: List[str]
    best_for: str
    gpu_requirement: str
    throughput: str
    latency: str


# 主流部署方案对比
DEPLOYMENT_OPTIONS = [
    DeploymentOption(
        name="vLLM",
        description="高吞吐量LLM推理引擎, PagedAttention显存管理",
        pros=[
            "极高吞吐量 (PagedAttention)",
            "OpenAI兼容API",
            "支持连续批处理 (Continuous Batching)",
            "活跃的社区和生态",
        ],
        cons=[
            "需要GPU",
            "首次加载模型较慢",
            "部分模型架构不支持",
        ],
        best_for="生产环境高并发LLM服务",
        gpu_requirement="16GB+ (7B模型) / 80GB+ (70B模型)",
        throughput="高 (3000+ tokens/s)",
        latency="低 (首token <200ms)",
    ),
    DeploymentOption(
        name="TGI (Text Generation Inference)",
        description="HuggingFace出品的文本生成推理服务",
        pros=[
            "HuggingFace生态深度整合",
            "Docker一键部署",
            "支持流式输出",
            "内置Token流控",
        ],
        cons=[
            "吞吐量略低于vLLM",
            "配置项较多",
        ],
        best_for="HuggingFace模型快速部署",
        gpu_requirement="16GB+ (7B模型)",
        throughput="高 (2000+ tokens/s)",
        latency="低 (首token <300ms)",
    ),
    DeploymentOption(
        name="Ollama",
        description="本地运行LLM的极简工具",
        pros=[
            "极其简单的安装和使用",
            "支持CPU和GPU",
            "GGUF量化模型支持好",
            "跨平台 (Mac/Linux/Windows)",
        ],
        cons=[
            "不适合高并发生产环境",
            "性能不如专业推理引擎",
        ],
        best_for="本地开发和测试",
        gpu_requirement="可选 (支持CPU运行)",
        throughput="中 (100-500 tokens/s)",
        latency="中 (首token <500ms)",
    ),
]


def recommend_deployment(
    concurrent_users: int,
    model_size_b: float,
    gpu_memory_gb: int,
    need_streaming: bool = True,
) -> str:
    """根据需求推荐部署方案"""
    recommendations = []

    print(f"{'=' * 60}")
    print("LLM部署方案推荐")
    print(f"{'=' * 60}")
    print(f"并发用户: {concurrent_users}")
    print(f"模型大小: {model_size_b}B参数")
    print(f"GPU显存: {gpu_memory_gb}GB")
    print(f"需要流式输出: {need_streaming}")
    print(f"{'=' * 60}")

    # 判断逻辑
    if concurrent_users <= 5 and gpu_memory_gb <= 8:
        recommendations.append(("Ollama", "低并发 + 有限GPU, 适合本地开发"))
    if concurrent_users >= 10:
        recommendations.append(("vLLM", "高并发场景首选"))
    if concurrent_users >= 5:
        recommendations.append(("TGI", "中高并发, HuggingFace生态友好"))

    # 显存检查
    min_memory = model_size_b * 2  # FP16大约需要2x参数量GB
    if gpu_memory_gb < min_memory:
        print(f"\n[警告] GPU显存 {gpu_memory_gb}GB 可能不足以运行 "
              f"{model_size_b}B模型 (FP16需要约{min_memory:.0f}GB)")
        print("建议: 使用AWQ/GPTQ量化, 或使用更大显存的GPU")
        if gpu_memory_gb >= min_memory / 2:
            recommendations.append(
                ("vLLM + AWQ量化",
                 f"4-bit量化后约需{min_memory/4:.0f}GB显存")
            )

    print("\n推荐方案:")
    for i, (name, reason) in enumerate(recommendations, 1):
        print(f"  {i}. {name}: {reason}")

    return recommendations[0][0] if recommendations else "Ollama"


def print_deployment_comparison():
    """打印部署方案对比表"""
    print(f"\n{'=' * 70}")
    print("LLM部署方案详细对比")
    print(f"{'=' * 70}")

    for opt in DEPLOYMENT_OPTIONS:
        print(f"\n{'─' * 50}")
        print(f"方案: {opt.name}")
        print(f"描述: {opt.description}")
        print(f"吞吐量: {opt.throughput}")
        print(f"延迟: {opt.latency}")
        print(f"GPU需求: {opt.gpu_requirement}")
        print(f"最适合: {opt.best_for}")
        print(f"优点:")
        for p in opt.pros:
            print(f"  + {p}")
        print(f"缺点:")
        for c in opt.cons:
            print(f"  - {c}")


if __name__ == "__main__":
    print_deployment_comparison()
    print("\n")
    recommend_deployment(
        concurrent_users=50,
        model_size_b=7,
        gpu_memory_gb=24,
        need_streaming=True,
    )
```

---

## vLLM高性能推理

### vLLM架构原理

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       vLLM 核心架构                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  请求接入层                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  OpenAI Compatible API Server                                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │   │
│  │  │ /completions│  │ /chat/      │  │ /embeddings │                │   │
│  │  │             │  │ completions │  │             │                │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                │   │
│  │         └────────────────┼────────────────┘                        │   │
│  └──────────────────────────┼──────────────────────────────────────────┘   │
│                             ▼                                              │
│  调度层                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Scheduler (调度器)                                                 │   │
│  │  ┌──────────────────┐  ┌──────────────────┐                        │   │
│  │  │  等待队列         │  │  运行队列         │                        │   │
│  │  │  Waiting Queue   │  │  Running Queue   │                        │   │
│  │  │  [Req1][Req2]... │  │  [Req3][Req4]... │                        │   │
│  │  └──────────────────┘  └──────────────────┘                        │   │
│  │                                                                     │   │
│  │  策略: Continuous Batching (连续批处理)                              │   │
│  │  - 新请求可随时加入当前批次                                          │   │
│  │  - 完成的请求立即移出, 释放显存                                      │   │
│  │  - 比静态批处理吞吐量提升 2-10x                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                             │                                              │
│                             ▼                                              │
│  执行层                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PagedAttention Engine                                              │   │
│  │                                                                     │   │
│  │  GPU显存布局:                                                       │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │  模型权重 (固定)           KV Cache (动态分页)                │  │   │
│  │  │  ┌────────────────┐      ┌────┬────┬────┬────┬────┐        │  │   │
│  │  │  │  Transformer   │      │Blk1│Blk2│Blk3│Blk4│... │        │  │   │
│  │  │  │  Layers        │      ├────┼────┼────┼────┼────┤        │  │   │
│  │  │  │  (14GB for 7B) │      │Req1│Req1│Req2│Free│Req3│        │  │   │
│  │  │  └────────────────┘      └────┴────┴────┴────┴────┘        │  │   │
│  │  │                                                              │  │   │
│  │  │  Page Table (页表):                                          │  │   │
│  │  │  Req1: [Blk1, Blk2]  (token 0-31, 32-63)                   │  │   │
│  │  │  Req2: [Blk3]        (token 0-31)                           │  │   │
│  │  │  Req3: [Blk5]        (token 0-31)                           │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  │                                                                     │   │
│  │  优势:                                                              │   │
│  │  - 消除KV Cache显存碎片 (传统方式浪费60-80%)                       │   │
│  │  - 支持更大批次和更长序列                                           │   │
│  │  - 显存利用率接近100%                                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### vLLM启动与使用

```python
"""
vLLM 部署与使用完整示例
包含: 服务启动、API调用、流式输出、性能测试
"""

import json
import time
import asyncio
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass, field


# ============================================================
# 第一部分: vLLM启动配置
# ============================================================

VLLM_STARTUP_CONFIGS = {
    "基础启动": {
        "command": """
# 安装vLLM
pip install vllm

# 基础启动 (单GPU)
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --host 0.0.0.0 \\
    --port 8000
""",
        "说明": "最简启动方式, 默认使用FP16, 适合开发测试",
    },
    "生产配置": {
        "command": """
# 生产环境推荐配置
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --tensor-parallel-size 2 \\
    --max-model-len 8192 \\
    --max-num-seqs 256 \\
    --gpu-memory-utilization 0.90 \\
    --enable-chunked-prefill \\
    --disable-log-requests
""",
        "说明": "2GPU张量并行, 限制序列长度, 优化显存利用率",
    },
    "量化部署": {
        "command": """
# AWQ量化模型部署 (显存减少75%)
python -m vllm.entrypoints.openai.api_server \\
    --model TheBloke/Llama-3.1-8B-Instruct-AWQ \\
    --quantization awq \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --max-model-len 4096 \\
    --gpu-memory-utilization 0.85
""",
        "说明": "4-bit AWQ量化, 7B模型仅需约6GB显存",
    },
}


def print_vllm_configs():
    """打印vLLM启动配置"""
    print("=" * 60)
    print("vLLM 启动配置指南")
    print("=" * 60)
    for name, config in VLLM_STARTUP_CONFIGS.items():
        print(f"\n{'─' * 40}")
        print(f"配置: {name}")
        print(f"说明: {config['说明']}")
        print(f"命令:{config['command']}")


# ============================================================
# 第二部分: vLLM API客户端 (模拟, 无需实际服务)
# ============================================================

@dataclass
class ChatMessage:
    """聊天消息"""
    role: str
    content: str


@dataclass
class CompletionResponse:
    """补全响应"""
    id: str
    model: str
    choices: List[Dict]
    usage: Dict[str, int]
    created: float = 0


class VLLMClient:
    """
    vLLM API客户端
    兼容OpenAI API格式, 支持同步/流式调用
    """

    def __init__(self, base_url: str = "http://localhost:8000",
                 api_key: str = "not-needed"):
        self.base_url = base_url
        self.api_key = api_key
        self.default_params = {
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 0.95,
        }

    def chat_completion(
        self, messages: List[Dict[str, str]],
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> CompletionResponse:
        """
        聊天补全 (模拟响应用于演示)
        实际使用时调用 vLLM 的 OpenAI 兼容端点
        """
        # 构建请求体
        request_body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        print(f"[vLLM Client] 发送请求到 {self.base_url}/v1/chat/completions")
        print(f"  模型: {model}")
        print(f"  消息数: {len(messages)}")
        print(f"  温度: {temperature}, 最大token: {max_tokens}")

        # 模拟响应
        simulated_content = f"这是对'{messages[-1]['content']}'的模拟响应。"

        response = CompletionResponse(
            id="chatcmpl-demo123",
            model=model,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": simulated_content},
                "finish_reason": "stop",
            }],
            usage={
                "prompt_tokens": sum(len(m["content"]) for m in messages),
                "completion_tokens": len(simulated_content),
                "total_tokens": (
                    sum(len(m["content"]) for m in messages)
                    + len(simulated_content)
                ),
            },
            created=time.time(),
        )
        return response

    def generate_api_call_code(self) -> str:
        """生成实际API调用代码(可直接使用)"""
        return '''
# ---- 实际使用vLLM的代码 (需要运行vLLM服务) ----

# 方法1: 使用 OpenAI 官方SDK (推荐)
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # vLLM不需要真实key
)

# 同步调用
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "解释什么是机器学习"},
    ],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)

# 流式调用
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "用Python写一个快速排序"},
    ],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

# 方法2: 使用 requests
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "你好"}],
        "temperature": 0.7,
        "max_tokens": 256,
    },
)
result = response.json()
print(result["choices"][0]["message"]["content"])

# 方法3: 使用 aiohttp (异步高并发)
import aiohttp
import asyncio

async def batch_requests(prompts: list, concurrency: int = 10):
    """批量并发请求"""
    semaphore = asyncio.Semaphore(concurrency)

    async def single_request(prompt):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8000/v1/chat/completions",
                    json={
                        "model": "meta-llama/Llama-3.1-8B-Instruct",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 256,
                    },
                ) as resp:
                    return await resp.json()

    tasks = [single_request(p) for p in prompts]
    return await asyncio.gather(*tasks)

# 使用: asyncio.run(batch_requests(["问题1", "问题2", ...]))
'''


# ============================================================
# 第三部分: 性能测试工具
# ============================================================

class LLMBenchmark:
    """LLM推理性能基准测试"""

    def __init__(self):
        self.results: List[Dict] = []

    def simulate_benchmark(
        self, num_requests: int = 100,
        concurrency: int = 10,
        avg_input_tokens: int = 200,
        avg_output_tokens: int = 150,
    ) -> Dict:
        """模拟性能基准测试"""
        import random
        random.seed(42)

        print(f"\n{'=' * 60}")
        print("LLM推理性能基准测试")
        print(f"{'=' * 60}")
        print(f"请求数: {num_requests}")
        print(f"并发数: {concurrency}")
        print(f"平均输入tokens: {avg_input_tokens}")
        print(f"平均输出tokens: {avg_output_tokens}")

        # 模拟延迟数据
        latencies = []
        ttfts = []  # Time To First Token
        tps_list = []  # Tokens Per Second

        for _ in range(num_requests):
            input_tokens = avg_input_tokens + random.randint(-50, 50)
            output_tokens = avg_output_tokens + random.randint(-30, 30)

            # 模拟: TTFT与输入长度相关, 总延迟与输出长度相关
            ttft = 0.05 + input_tokens * 0.0002 + random.uniform(0, 0.05)
            generation_time = output_tokens * 0.008 + random.uniform(0, 0.1)
            total_latency = ttft + generation_time

            tps = output_tokens / generation_time if generation_time > 0 else 0

            ttfts.append(ttft * 1000)
            latencies.append(total_latency * 1000)
            tps_list.append(tps)

        # 计算统计指标
        def percentile(data, p):
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]

        results = {
            "总请求数": num_requests,
            "并发数": concurrency,
            "TTFT_avg_ms": sum(ttfts) / len(ttfts),
            "TTFT_p50_ms": percentile(ttfts, 50),
            "TTFT_p95_ms": percentile(ttfts, 95),
            "TTFT_p99_ms": percentile(ttfts, 99),
            "延迟_avg_ms": sum(latencies) / len(latencies),
            "延迟_p50_ms": percentile(latencies, 50),
            "延迟_p95_ms": percentile(latencies, 95),
            "延迟_p99_ms": percentile(latencies, 99),
            "吞吐量_avg_tps": sum(tps_list) / len(tps_list),
            "吞吐量_total_tps": (
                num_requests * avg_output_tokens
                / (sum(latencies) / 1000 / concurrency)
            ),
        }

        print(f"\n{'─' * 40}")
        print("测试结果:")
        print(f"{'─' * 40}")
        print(f"  TTFT (首Token延迟):")
        print(f"    平均: {results['TTFT_avg_ms']:.1f}ms")
        print(f"    P50:  {results['TTFT_p50_ms']:.1f}ms")
        print(f"    P95:  {results['TTFT_p95_ms']:.1f}ms")
        print(f"    P99:  {results['TTFT_p99_ms']:.1f}ms")
        print(f"  端到端延迟:")
        print(f"    平均: {results['延迟_avg_ms']:.1f}ms")
        print(f"    P50:  {results['延迟_p50_ms']:.1f}ms")
        print(f"    P95:  {results['延迟_p95_ms']:.1f}ms")
        print(f"    P99:  {results['延迟_p99_ms']:.1f}ms")
        print(f"  吞吐量:")
        print(f"    平均每请求: {results['吞吐量_avg_tps']:.1f} tokens/s")
        print(f"    系统总计: {results['吞吐量_total_tps']:.0f} tokens/s")

        return results


def demo_vllm():
    """vLLM部署演示"""
    # 1. 显示配置
    print_vllm_configs()

    # 2. API调用示例
    client = VLLMClient()
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "你是一个有用的AI助手。"},
            {"role": "user", "content": "解释什么是PagedAttention"},
        ],
        temperature=0.7,
    )
    print(f"\n响应: {response.choices[0]['message']['content']}")
    print(f"Token使用: {response.usage}")

    # 3. 打印实际API调用代码
    print("\n" + "=" * 60)
    print("实际API调用代码参考:")
    print("=" * 60)
    print(client.generate_api_call_code())

    # 4. 性能测试
    benchmark = LLMBenchmark()
    benchmark.simulate_benchmark(
        num_requests=100, concurrency=10,
        avg_input_tokens=200, avg_output_tokens=150,
    )


if __name__ == "__main__":
    demo_vllm()
```

---

## TGI部署实战

### TGI架构与配置

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TGI (Text Generation Inference) 架构                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Docker部署:                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  docker run --gpus all -p 8080:80                                   │   │
│  │    -v $PWD/data:/data                                               │   │
│  │    ghcr.io/huggingface/text-generation-inference:latest             │   │
│  │    --model-id meta-llama/Llama-3.1-8B-Instruct                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  内部架构:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TGI Server                                                         │   │
│  │  ┌──────────────┐                                                   │   │
│  │  │  HTTP Router  │  /generate           /generate_stream            │   │
│  │  │  (Rust)       │  (同步)              (SSE流式)                   │   │
│  │  └───────┬──────┘                                                   │   │
│  │          │                                                          │   │
│  │  ┌───────▼──────────────────────────────────────────────────────┐   │   │
│  │  │  Batcher (批处理器)                                          │   │   │
│  │  │  - Token Budget: 控制每批最大token数                         │   │   │
│  │  │  - Waiting Timeout: 等待凑批的超时时间                       │   │   │
│  │  │  - Max Batch Size: 最大批次大小                              │   │   │
│  │  └───────┬──────────────────────────────────────────────────────┘   │   │
│  │          │                                                          │   │
│  │  ┌───────▼──────────────────────────────────────────────────────┐   │   │
│  │  │  Model Backend (Python)                                      │   │   │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │   │   │
│  │  │  │Flash Attention│  │  Paged KV    │  │  Quantization│      │   │   │
│  │  │  │  v2           │  │  Cache       │  │  AWQ/GPTQ    │      │   │   │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘      │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  TGI vs vLLM 选型:                                                         │
│  ┌───────────────┬──────────────────┬──────────────────┐                   │
│  │ 维度          │ TGI              │ vLLM             │                   │
│  ├───────────────┼──────────────────┼──────────────────┤                   │
│  │ 部署方式      │ Docker优先       │ pip安装 + 命令行 │                   │
│  │ API格式       │ 自定义 + OpenAI  │ 纯OpenAI兼容     │                   │
│  │ 吞吐量        │ 高               │ 极高             │                   │
│  │ HF Hub集成    │ 深度集成         │ 支持             │                   │
│  │ 流式控制      │ 精细             │ 标准             │                   │
│  │ 监控          │ 内置Prometheus   │ 需额外配置       │                   │
│  └───────────────┴──────────────────┴──────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### TGI部署代码

```python
"""
TGI (Text Generation Inference) 部署配置与使用
包含: Docker配置、API调用、docker-compose编排
"""

from typing import Dict, List


class TGIDeployment:
    """TGI部署配置生成器"""

    @staticmethod
    def docker_run_command(
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        port: int = 8080,
        num_gpus: int = 1,
        quantize: str = None,
        max_input_length: int = 4096,
        max_total_tokens: int = 8192,
        max_batch_size: int = 32,
    ) -> str:
        """生成Docker启动命令"""
        cmd = f"""docker run --gpus '"device={",".join(str(i) for i in range(num_gpus))}"' \\
    -p {port}:80 \\
    -v $PWD/models:/data \\
    -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \\
    ghcr.io/huggingface/text-generation-inference:latest \\
    --model-id {model_id} \\
    --max-input-length {max_input_length} \\
    --max-total-tokens {max_total_tokens} \\
    --max-batch-prefill-tokens {max_input_length} \\
    --max-batch-size {max_batch_size}"""

        if quantize:
            cmd += f" \\\n    --quantize {quantize}"

        if num_gpus > 1:
            cmd += f" \\\n    --num-shard {num_gpus}"

        return cmd

    @staticmethod
    def docker_compose_config(
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    ) -> str:
        """生成docker-compose配置"""
        return f"""# docker-compose.yml
version: '3.8'

services:
  tgi:
    image: ghcr.io/huggingface/text-generation-inference:latest
    ports:
      - "8080:80"
    volumes:
      - ./models:/data
    environment:
      - HUGGING_FACE_HUB_TOKEN=${{HF_TOKEN}}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: >
      --model-id {model_id}
      --max-input-length 4096
      --max-total-tokens 8192
      --max-batch-size 32
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:80/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # 可选: Nginx反向代理
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      tgi:
        condition: service_healthy
"""

    @staticmethod
    def api_usage_example() -> str:
        """TGI API调用示例"""
        return '''
# ---- TGI API调用示例 ----

import requests

TGI_URL = "http://localhost:8080"

# 1. 健康检查
health = requests.get(f"{TGI_URL}/health")
print(f"服务状态: {health.status_code}")

# 2. 获取模型信息
info = requests.get(f"{TGI_URL}/info").json()
print(f"模型: {info['model_id']}")
print(f"最大输入长度: {info['max_input_length']}")

# 3. 文本生成 (同步)
response = requests.post(
    f"{TGI_URL}/generate",
    json={
        "inputs": "解释什么是Transformer架构:",
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True,
            "return_full_text": False,
        },
    },
)
result = response.json()
print(f"生成文本: {result['generated_text']}")

# 4. 流式生成 (SSE)
import sseclient

response = requests.post(
    f"{TGI_URL}/generate_stream",
    json={
        "inputs": "用Python实现二分查找:",
        "parameters": {"max_new_tokens": 512, "temperature": 0.3},
    },
    stream=True,
)
client = sseclient.SSEClient(response)
for event in client.events():
    import json
    data = json.loads(event.data)
    if data.get("token"):
        print(data["token"]["text"], end="", flush=True)

# 5. OpenAI兼容端点 (TGI v2+)
from openai import OpenAI

client = OpenAI(base_url=f"{TGI_URL}/v1", api_key="not-needed")
chat = client.chat.completions.create(
    model="tgi",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=256,
)
print(chat.choices[0].message.content)
'''


def demo_tgi():
    """TGI部署演示"""
    tgi = TGIDeployment()

    print("=" * 60)
    print("TGI 部署配置")
    print("=" * 60)

    # 基础配置
    print("\n--- 基础启动命令 ---")
    print(tgi.docker_run_command())

    # 量化配置
    print("\n--- 量化启动命令 (AWQ) ---")
    print(tgi.docker_run_command(
        model_id="TheBloke/Llama-3.1-8B-Instruct-AWQ",
        quantize="awq",
    ))

    # 多GPU配置
    print("\n--- 多GPU启动命令 ---")
    print(tgi.docker_run_command(num_gpus=2))

    # Docker Compose
    print("\n--- Docker Compose 配置 ---")
    print(tgi.docker_compose_config())

    # API示例
    print("\n--- API使用示例 ---")
    print(tgi.api_usage_example())


if __name__ == "__main__":
    demo_tgi()
```

---

## FastAPI + vLLM完整服务

### 生产级服务架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   FastAPI + vLLM 生产级服务架构                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        FastAPI Application                          │   │
│  │                                                                     │   │
│  │  中间件层:                                                          │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐     │   │
│  │  │ 认证中间件 │ │ 限流中间件 │ │ 日志中间件 │ │ CORS中间件 │     │   │
│  │  │ API Key    │ │ Rate Limit │ │ Request Log│ │            │     │   │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘     │   │
│  │                                                                     │   │
│  │  路由层:                                                            │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │  /v1/chat/completions   (聊天补全, 兼容OpenAI)              │  │   │
│  │  │  /v1/completions        (文本补全)                           │  │   │
│  │  │  /v1/models             (模型列表)                           │  │   │
│  │  │  /health                (健康检查)                           │  │   │
│  │  │  /metrics               (Prometheus指标)                     │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  │                                                                     │   │
│  │  业务逻辑层:                                                        │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐     │   │
│  │  │ 请求验证   │ │ 内容过滤   │ │ Token计数  │ │ 响应格式化 │     │   │
│  │  │ Pydantic   │ │ 安全检查   │ │ 计费统计   │ │ 流式/同步  │     │   │
│  │  └─────┬──────┘ └────────────┘ └────────────┘ └────────────┘     │   │
│  │        │                                                            │   │
│  │  ┌─────▼──────────────────────────────────────────────────────┐    │   │
│  │  │              vLLM Engine (后端推理引擎)                      │    │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │    │   │
│  │  │  │ Scheduler│  │ PagedAttn│  │ KV Cache │                 │    │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘                 │    │   │
│  │  └────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### FastAPI完整服务代码

```python
"""
FastAPI + vLLM 生产级LLM服务
包含: API认证、限流、流式输出、健康检查、指标收集
"""

import time
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass, field


# ============================================================
# 数据模型 (Pydantic模型模拟)
# ============================================================

@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatCompletionRequest:
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.95
    stream: bool = False
    stop: Optional[List[str]] = None


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ============================================================
# 中间件模拟
# ============================================================

class RateLimiter:
    """简易令牌桶限流器"""

    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.tokens: Dict[str, List[float]] = {}

    def is_allowed(self, api_key: str) -> bool:
        """检查请求是否被允许"""
        now = time.time()
        if api_key not in self.tokens:
            self.tokens[api_key] = []

        # 清理一分钟前的记录
        self.tokens[api_key] = [
            t for t in self.tokens[api_key] if now - t < 60
        ]

        if len(self.tokens[api_key]) >= self.rpm:
            return False

        self.tokens[api_key].append(now)
        return True

    def get_remaining(self, api_key: str) -> int:
        """获取剩余配额"""
        now = time.time()
        if api_key not in self.tokens:
            return self.rpm
        recent = [t for t in self.tokens[api_key] if now - t < 60]
        return max(0, self.rpm - len(recent))


class APIKeyAuth:
    """API Key认证"""

    def __init__(self):
        self.valid_keys = {
            "sk-demo-key-001": {"name": "测试用户", "rpm": 60},
            "sk-demo-key-002": {"name": "生产用户", "rpm": 600},
        }

    def validate(self, api_key: str) -> Optional[Dict]:
        """验证API Key"""
        return self.valid_keys.get(api_key)


class MetricsCollector:
    """Prometheus风格的指标收集器"""

    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_tokens = 0
        self.total_latency_ms = 0
        self.latencies: List[float] = []

    def record_request(self, latency_ms: float, tokens: int,
                       error: bool = False):
        """记录请求指标"""
        self.request_count += 1
        self.total_tokens += tokens
        self.total_latency_ms += latency_ms
        self.latencies.append(latency_ms)
        if error:
            self.error_count += 1

    def get_metrics(self) -> Dict:
        """获取当前指标"""
        avg_latency = (
            self.total_latency_ms / self.request_count
            if self.request_count > 0 else 0
        )
        sorted_lat = sorted(self.latencies) if self.latencies else [0]
        p95_idx = int(len(sorted_lat) * 0.95)

        return {
            "request_total": self.request_count,
            "error_total": self.error_count,
            "error_rate": (
                self.error_count / self.request_count
                if self.request_count > 0 else 0
            ),
            "tokens_total": self.total_tokens,
            "latency_avg_ms": avg_latency,
            "latency_p95_ms": sorted_lat[min(p95_idx, len(sorted_lat) - 1)],
        }

    def to_prometheus_format(self) -> str:
        """输出Prometheus格式"""
        m = self.get_metrics()
        lines = [
            "# HELP llm_requests_total Total number of requests",
            "# TYPE llm_requests_total counter",
            f'llm_requests_total {m["request_total"]}',
            "",
            "# HELP llm_errors_total Total number of errors",
            "# TYPE llm_errors_total counter",
            f'llm_errors_total {m["error_total"]}',
            "",
            "# HELP llm_tokens_total Total tokens processed",
            "# TYPE llm_tokens_total counter",
            f'llm_tokens_total {m["tokens_total"]}',
            "",
            "# HELP llm_latency_ms Request latency in ms",
            "# TYPE llm_latency_ms summary",
            f'llm_latency_ms{{quantile="0.5"}} {m["latency_avg_ms"]:.1f}',
            f'llm_latency_ms{{quantile="0.95"}} {m["latency_p95_ms"]:.1f}',
        ]
        return "\n".join(lines)


# ============================================================
# LLM服务核心
# ============================================================

class LLMService:
    """
    生产级LLM服务
    集成: 认证、限流、指标收集
    """

    def __init__(self):
        self.auth = APIKeyAuth()
        self.rate_limiter = RateLimiter(requests_per_minute=60)
        self.metrics = MetricsCollector()
        self.model_name = "meta-llama/Llama-3.1-8B-Instruct"
        self.start_time = time.time()

    def health_check(self) -> Dict:
        """健康检查"""
        uptime = time.time() - self.start_time
        return {
            "status": "healthy",
            "model": self.model_name,
            "uptime_seconds": round(uptime, 1),
            "metrics": self.metrics.get_metrics(),
        }

    def chat_completion(
        self, request: ChatCompletionRequest, api_key: str,
    ) -> Dict:
        """处理聊天补全请求"""
        start_time = time.time()

        # 1. 认证
        user = self.auth.validate(api_key)
        if user is None:
            self.metrics.record_request(0, 0, error=True)
            return {"error": "Invalid API key", "code": 401}

        # 2. 限流
        if not self.rate_limiter.is_allowed(api_key):
            remaining = self.rate_limiter.get_remaining(api_key)
            self.metrics.record_request(0, 0, error=True)
            return {
                "error": "Rate limit exceeded",
                "code": 429,
                "remaining": remaining,
            }

        # 3. 请求验证
        if not request.messages:
            return {"error": "Messages cannot be empty", "code": 400}

        if request.max_tokens > 4096:
            return {"error": "max_tokens exceeds limit (4096)", "code": 400}

        # 4. 推理 (模拟)
        prompt_tokens = sum(len(m["content"]) // 4 for m in request.messages)
        completion_text = f"[模拟响应] 收到{len(request.messages)}条消息"
        completion_tokens = len(completion_text) // 4

        # 5. 构建响应
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion_text,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

        # 6. 记录指标
        latency_ms = (time.time() - start_time) * 1000
        total_tokens = prompt_tokens + completion_tokens
        self.metrics.record_request(latency_ms, total_tokens)

        return response

    def generate_fastapi_code(self) -> str:
        """生成完整的FastAPI服务代码"""
        return '''
# ---- 完整FastAPI + vLLM服务代码 (需安装依赖) ----
# pip install fastapi uvicorn vllm openai

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import uvicorn
import time
import uuid

app = FastAPI(title="LLM API Service", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化vLLM引擎
engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    max_model_len=8192,
)
engine = AsyncLLMEngine.from_engine_args(engine_args)


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    stream: bool = False


@app.get("/health")
async def health():
    return {"status": "healthy", "model": engine_args.model}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    authorization: str = Header(None),
):
    # 构建prompt
    prompt = ""
    for msg in request.messages:
        prompt += f"<|{msg.role}|>\\n{msg.content}\\n"
    prompt += "<|assistant|>\\n"

    # 采样参数
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=0.95,
    )

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if request.stream:
        # 流式输出
        async def stream_generator():
            async for output in engine.generate(
                prompt, sampling_params, request_id
            ):
                text = output.outputs[0].text
                chunk = {
                    "id": request_id,
                    "choices": [{"delta": {"content": text}}],
                }
                yield f"data: {json.dumps(chunk)}\\n\\n"
            yield "data: [DONE]\\n\\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
        )
    else:
        # 同步输出
        results = await engine.generate(prompt, sampling_params, request_id)
        output = results[0].outputs[0]
        return {
            "id": request_id,
            "choices": [{
                "message": {"role": "assistant", "content": output.text},
                "finish_reason": output.finish_reason,
            }],
            "usage": {
                "prompt_tokens": len(results[0].prompt_token_ids),
                "completion_tokens": len(output.token_ids),
                "total_tokens": (
                    len(results[0].prompt_token_ids) + len(output.token_ids)
                ),
            },
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''


def demo_fastapi_service():
    """FastAPI + vLLM服务演示"""
    service = LLMService()

    # 1. 健康检查
    print("=" * 60)
    print("LLM Service 演示")
    print("=" * 60)
    health = service.health_check()
    print(f"\n健康检查: {health}")

    # 2. 正常请求
    print("\n--- 正常请求 ---")
    request = ChatCompletionRequest(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "你是一个有用的助手"},
            {"role": "user", "content": "什么是PagedAttention?"},
        ],
        temperature=0.7,
        max_tokens=256,
    )
    response = service.chat_completion(request, "sk-demo-key-001")
    print(f"响应: {response}")

    # 3. 无效API Key
    print("\n--- 认证失败 ---")
    response = service.chat_completion(request, "invalid-key")
    print(f"响应: {response}")

    # 4. 多次请求后查看指标
    for i in range(10):
        service.chat_completion(request, "sk-demo-key-001")

    print("\n--- Prometheus指标 ---")
    print(service.metrics.to_prometheus_format())

    # 5. 完整FastAPI代码
    print("\n--- 完整FastAPI服务代码 ---")
    print(service.generate_fastapi_code())


if __name__ == "__main__":
    demo_fastapi_service()
```

---

## 负载均衡与高可用

### 负载均衡架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LLM服务负载均衡架构                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  客户端请求                                                                 │
│       │                                                                    │
│       ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Layer 7 Load Balancer (Nginx / Traefik / HAProxy)                  │   │
│  │                                                                     │   │
│  │  策略:                                                              │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐               │   │
│  │  │ Round Robin  │ │ Least Conn   │ │ GPU感知路由  │               │   │
│  │  │ 轮询(默认)   │ │ 最少连接     │ │ 按显存负载   │               │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘               │   │
│  └─────────────────┬───────────────┬───────────────┬───────────────────┘   │
│                    │               │               │                       │
│                    ▼               ▼               ▼                       │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐          │
│  │  vLLM Instance 1 │ │  vLLM Instance 2 │ │  vLLM Instance 3 │          │
│  │  GPU: A100 (80G) │ │  GPU: A100 (80G) │ │  GPU: A100 (80G) │          │
│  │  Model: Llama-8B │ │  Model: Llama-8B │ │  Model: Llama-8B │          │
│  │  Port: 8001      │ │  Port: 8002      │ │  Port: 8003      │          │
│  │  状态: Active    │ │  状态: Active    │ │  状态: Standby   │          │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘          │
│                                                                             │
│  高可用机制:                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  1. 健康检查: 定期探测 /health, 不健康自动摘除                      │   │
│  │  2. 故障转移: 主节点故障, 自动切换到备用节点                         │   │
│  │  3. 会话保持: 流式请求绑定到同一后端                                 │   │
│  │  4. 优雅关闭: 等待进行中的请求完成后再下线                           │   │
│  │  5. 自动重启: 进程崩溃后自动重启并重新加入集群                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 负载均衡实现

```python
"""
LLM服务负载均衡器
支持: 多种均衡策略、健康检查、故障转移
"""

import time
import random
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class BalancerStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    GPU_AWARE = "gpu_aware"


@dataclass
class BackendServer:
    """后端服务器"""
    name: str
    url: str
    weight: int = 1
    healthy: bool = True
    active_connections: int = 0
    gpu_utilization: float = 0.0  # 0-1
    gpu_memory_used: float = 0.0  # GB
    gpu_memory_total: float = 80.0  # GB
    total_requests: int = 0
    error_count: int = 0
    last_health_check: float = 0


class LLMLoadBalancer:
    """
    LLM服务负载均衡器
    支持多种路由策略与健康检查
    """

    def __init__(self, strategy: BalancerStrategy = BalancerStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.backends: List[BackendServer] = []
        self.current_index = 0
        self.health_check_interval = 10  # 秒

    def add_backend(self, name: str, url: str, weight: int = 1,
                    gpu_memory_total: float = 80.0):
        """添加后端服务器"""
        backend = BackendServer(
            name=name, url=url, weight=weight,
            gpu_memory_total=gpu_memory_total,
        )
        self.backends.append(backend)
        print(f"[LB] 添加后端: {name} ({url}), 权重={weight}")

    def health_check(self):
        """执行健康检查"""
        for backend in self.backends:
            # 模拟健康检查
            was_healthy = backend.healthy
            backend.healthy = random.random() > 0.05  # 5%概率不健康
            backend.last_health_check = time.time()

            # 模拟GPU指标
            if backend.healthy:
                backend.gpu_utilization = random.uniform(0.3, 0.9)
                backend.gpu_memory_used = (
                    backend.gpu_memory_total * random.uniform(0.4, 0.85)
                )

            if was_healthy and not backend.healthy:
                print(f"[LB][ALERT] {backend.name} 变为不健康!")
            elif not was_healthy and backend.healthy:
                print(f"[LB] {backend.name} 恢复健康")

    def _get_healthy_backends(self) -> List[BackendServer]:
        """获取所有健康的后端"""
        return [b for b in self.backends if b.healthy]

    def select_backend(self) -> Optional[BackendServer]:
        """根据策略选择后端"""
        healthy = self._get_healthy_backends()
        if not healthy:
            print("[LB][ERROR] 没有健康的后端可用!")
            return None

        if self.strategy == BalancerStrategy.ROUND_ROBIN:
            idx = self.current_index % len(healthy)
            self.current_index += 1
            return healthy[idx]

        elif self.strategy == BalancerStrategy.LEAST_CONNECTIONS:
            return min(healthy, key=lambda b: b.active_connections)

        elif self.strategy == BalancerStrategy.WEIGHTED:
            # 加权随机
            total_weight = sum(b.weight for b in healthy)
            r = random.uniform(0, total_weight)
            cumulative = 0
            for b in healthy:
                cumulative += b.weight
                if r <= cumulative:
                    return b
            return healthy[-1]

        elif self.strategy == BalancerStrategy.GPU_AWARE:
            # GPU感知: 选择GPU利用率最低的
            return min(healthy, key=lambda b: b.gpu_utilization)

        return healthy[0]

    def route_request(self, request_id: str) -> Optional[str]:
        """路由请求到后端"""
        backend = self.select_backend()
        if backend is None:
            return None

        backend.active_connections += 1
        backend.total_requests += 1
        print(f"[LB] 请求 {request_id} -> {backend.name} "
              f"(连接数: {backend.active_connections}, "
              f"GPU: {backend.gpu_utilization:.0%})")
        return backend.url

    def release_connection(self, url: str):
        """释放连接"""
        for backend in self.backends:
            if backend.url == url:
                backend.active_connections = max(
                    0, backend.active_connections - 1
                )
                break

    def get_status(self) -> str:
        """获取负载均衡器状态"""
        lines = [
            f"{'=' * 65}",
            f"负载均衡器状态 (策略: {self.strategy.value})",
            f"{'=' * 65}",
            f"{'名称':<15}{'URL':<25}{'状态':<8}{'连接':<6}"
            f"{'GPU%':<8}{'显存':<12}{'请求数':<8}",
            f"{'-' * 65}",
        ]
        for b in self.backends:
            status = "健康" if b.healthy else "故障"
            mem = f"{b.gpu_memory_used:.0f}/{b.gpu_memory_total:.0f}GB"
            lines.append(
                f"{b.name:<15}{b.url:<25}{status:<8}"
                f"{b.active_connections:<6}{b.gpu_utilization:<8.0%}"
                f"{mem:<12}{b.total_requests:<8}"
            )
        return "\n".join(lines)


def generate_nginx_config(backends: List[Dict]) -> str:
    """生成Nginx负载均衡配置"""
    upstream_servers = "\n".join(
        f"    server {b['url']} weight={b.get('weight', 1)};"
        for b in backends
    )

    return f"""
# /etc/nginx/conf.d/llm-service.conf

upstream llm_backend {{
    least_conn;  # 最少连接策略
{upstream_servers}

    # 健康检查 (需要nginx-plus或第三方模块)
    # health_check interval=10s fails=3 passes=2;
}}

server {{
    listen 80;
    server_name llm-api.example.com;

    # 请求超时 (LLM生成可能较慢)
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;

    # 流式输出支持
    proxy_buffering off;
    proxy_cache off;

    location /v1/ {{
        proxy_pass http://llm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # SSE流式支持
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding on;
    }}

    location /health {{
        proxy_pass http://llm_backend;
    }}

    location /metrics {{
        proxy_pass http://llm_backend;
        # 限制内网访问
        allow 10.0.0.0/8;
        deny all;
    }}
}}
"""


def demo_load_balancer():
    """负载均衡器演示"""
    # 创建负载均衡器
    lb = LLMLoadBalancer(strategy=BalancerStrategy.LEAST_CONNECTIONS)

    # 添加后端
    lb.add_backend("vllm-node-1", "http://gpu-1:8001", weight=2)
    lb.add_backend("vllm-node-2", "http://gpu-2:8002", weight=2)
    lb.add_backend("vllm-node-3", "http://gpu-3:8003", weight=1)

    # 模拟健康检查
    lb.health_check()

    # 模拟请求路由
    print(f"\n模拟20个请求路由:")
    for i in range(20):
        url = lb.route_request(f"req-{i:03d}")
        if url and random.random() > 0.3:  # 模拟请求完成
            lb.release_connection(url)

    # 打印状态
    print(f"\n{lb.get_status()}")

    # 生成Nginx配置
    print("\n--- Nginx负载均衡配置 ---")
    print(generate_nginx_config([
        {"url": "gpu-1:8001", "weight": 2},
        {"url": "gpu-2:8002", "weight": 2},
        {"url": "gpu-3:8003", "weight": 1},
    ]))


if __name__ == "__main__":
    demo_load_balancer()
```

---

## 自动扩缩容

### 扩缩容策略

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     LLM服务自动扩缩容架构                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  监控指标                        扩缩容决策                                 │
│  ┌──────────────┐               ┌──────────────────────────────┐           │
│  │ GPU利用率    │───┐           │   Auto Scaler                │           │
│  │ (>80% 扩容) │   │           │                              │           │
│  └──────────────┘   │           │  规则:                       │           │
│  ┌──────────────┐   ├──────────>│  if GPU > 80% for 5min:     │           │
│  │ 请求队列深度 │───┤           │    scale_up(+1)             │           │
│  │ (>50 扩容)  │   │           │  if GPU < 30% for 10min:    │           │
│  └──────────────┘   │           │    scale_down(-1)           │           │
│  ┌──────────────┐   │           │  if queue > 50:             │           │
│  │ 响应延迟P95  │───┤           │    scale_up(+2)             │           │
│  │ (>2s 扩容)  │   │           │  min_replicas: 2            │           │
│  └──────────────┘   │           │  max_replicas: 10           │           │
│  ┌──────────────┐   │           └──────────────┬───────────────┘           │
│  │ 错误率       │───┘                          │                           │
│  │ (>5% 告警)  │                               │                           │
│  └──────────────┘                               ▼                           │
│                                                                             │
│  当前状态                     扩容操作                                      │
│  ┌────────┐                  ┌──────────┐                                  │
│  │ Pod 1  │  扩容时新增:     │ Pod 4    │  1. 调度GPU节点                 │
│  │ Pod 2  │ ─────────────>   │ (新增)   │  2. 拉取镜像                    │
│  │ Pod 3  │                  │          │  3. 加载模型 (~2min)            │
│  └────────┘                  │          │  4. 健康检查通过                 │
│                              └──────────┘  5. 接入流量                     │
│                                                                             │
│  Kubernetes HPA 配置:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  apiVersion: autoscaling/v2                                         │   │
│  │  kind: HorizontalPodAutoscaler                                      │   │
│  │  spec:                                                              │   │
│  │    scaleTargetRef: deployment/vllm-server                           │   │
│  │    minReplicas: 2                                                   │   │
│  │    maxReplicas: 10                                                  │   │
│  │    metrics:                                                         │   │
│  │    - type: Resource                                                 │   │
│  │      resource:                                                      │   │
│  │        name: nvidia.com/gpu                                         │   │
│  │        target: {type: Utilization, averageUtilization: 75}          │   │
│  │    - type: Pods                                                     │   │
│  │      pods:                                                          │   │
│  │        metric: {name: llm_queue_depth}                              │   │
│  │        target: {type: AverageValue, averageValue: 30}               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 自动扩缩容代码

```python
"""
LLM服务自动扩缩容控制器
支持: 基于指标的扩缩容、冷却期、最小/最大实例数
"""

import time
from typing import Dict, List
from dataclasses import dataclass, field
from enum import Enum


class ScaleAction(Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


@dataclass
class ScalingMetrics:
    """扩缩容指标快照"""
    timestamp: float
    gpu_utilization: float  # 0-1
    queue_depth: int
    latency_p95_ms: float
    error_rate: float  # 0-1
    active_replicas: int


@dataclass
class ScalingRule:
    """扩缩容规则"""
    name: str
    metric: str
    threshold_up: float
    threshold_down: float
    scale_up_amount: int = 1
    scale_down_amount: int = 1
    evaluation_minutes: int = 5


class AutoScaler:
    """
    LLM服务自动扩缩容控制器
    """

    def __init__(
        self, min_replicas: int = 2, max_replicas: int = 10,
        cooldown_seconds: int = 300,
    ):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.current_replicas = min_replicas
        self.cooldown_seconds = cooldown_seconds
        self.last_scale_time = 0
        self.metrics_history: List[ScalingMetrics] = []
        self.scale_events: List[Dict] = []
        self.rules: List[ScalingRule] = []

    def add_rule(self, rule: ScalingRule):
        """添加扩缩容规则"""
        self.rules.append(rule)
        print(f"[AutoScaler] 规则添加: {rule.name}")
        print(f"  扩容条件: {rule.metric} > {rule.threshold_up}")
        print(f"  缩容条件: {rule.metric} < {rule.threshold_down}")

    def record_metrics(self, metrics: ScalingMetrics):
        """记录指标"""
        self.metrics_history.append(metrics)
        # 只保留最近30分钟的数据
        cutoff = time.time() - 1800
        self.metrics_history = [
            m for m in self.metrics_history if m.timestamp > cutoff
        ]

    def _get_recent_avg(self, metric: str, minutes: int = 5) -> float:
        """获取最近N分钟的指标平均值"""
        cutoff = time.time() - minutes * 60
        recent = [
            m for m in self.metrics_history if m.timestamp > cutoff
        ]
        if not recent:
            return 0
        return sum(getattr(m, metric) for m in recent) / len(recent)

    def _in_cooldown(self) -> bool:
        """是否在冷却期内"""
        return time.time() - self.last_scale_time < self.cooldown_seconds

    def evaluate(self) -> ScaleAction:
        """评估是否需要扩缩容"""
        if self._in_cooldown():
            return ScaleAction.NO_ACTION

        for rule in self.rules:
            avg_value = self._get_recent_avg(
                rule.metric, rule.evaluation_minutes
            )

            # 检查是否需要扩容
            if avg_value > rule.threshold_up:
                if self.current_replicas < self.max_replicas:
                    new_count = min(
                        self.current_replicas + rule.scale_up_amount,
                        self.max_replicas,
                    )
                    self._scale(new_count, rule.name, avg_value, "up")
                    return ScaleAction.SCALE_UP

            # 检查是否需要缩容
            if avg_value < rule.threshold_down:
                if self.current_replicas > self.min_replicas:
                    new_count = max(
                        self.current_replicas - rule.scale_down_amount,
                        self.min_replicas,
                    )
                    self._scale(new_count, rule.name, avg_value, "down")
                    return ScaleAction.SCALE_DOWN

        return ScaleAction.NO_ACTION

    def _scale(self, new_count: int, rule_name: str,
               metric_value: float, direction: str):
        """执行扩缩容"""
        old_count = self.current_replicas
        self.current_replicas = new_count
        self.last_scale_time = time.time()

        event = {
            "timestamp": time.time(),
            "rule": rule_name,
            "direction": direction,
            "old_replicas": old_count,
            "new_replicas": new_count,
            "trigger_value": metric_value,
        }
        self.scale_events.append(event)

        arrow = "+++" if direction == "up" else "---"
        print(f"[AutoScaler] {arrow} {direction.upper()}: "
              f"{old_count} -> {new_count} 实例")
        print(f"  触发规则: {rule_name} (当前值: {metric_value:.2f})")

    def get_status(self) -> str:
        """获取扩缩容状态"""
        lines = [
            f"{'=' * 50}",
            f"AutoScaler 状态",
            f"{'=' * 50}",
            f"当前实例数: {self.current_replicas}",
            f"范围: [{self.min_replicas}, {self.max_replicas}]",
            f"冷却期: {self.cooldown_seconds}s",
            f"扩缩事件数: {len(self.scale_events)}",
        ]
        if self.scale_events:
            lines.append(f"\n最近扩缩事件:")
            for event in self.scale_events[-5:]:
                lines.append(
                    f"  [{event['direction'].upper()}] "
                    f"{event['old_replicas']} -> {event['new_replicas']} "
                    f"(规则: {event['rule']})"
                )
        return "\n".join(lines)


def demo_autoscaler():
    """自动扩缩容演示"""
    scaler = AutoScaler(min_replicas=2, max_replicas=8, cooldown_seconds=5)

    # 添加规则
    scaler.add_rule(ScalingRule(
        name="GPU利用率", metric="gpu_utilization",
        threshold_up=0.8, threshold_down=0.3,
        scale_up_amount=1, scale_down_amount=1,
        evaluation_minutes=5,
    ))
    scaler.add_rule(ScalingRule(
        name="请求队列深度", metric="queue_depth",
        threshold_up=50, threshold_down=5,
        scale_up_amount=2, scale_down_amount=1,
        evaluation_minutes=3,
    ))

    import random
    random.seed(42)

    # 模拟流量变化
    print("\n--- 模拟流量变化 ---")
    scenarios = [
        ("低流量", 0.2, 3, 200),
        ("正常流量", 0.5, 15, 500),
        ("高流量", 0.85, 60, 1500),
        ("流量高峰", 0.95, 100, 3000),
        ("高峰回落", 0.6, 20, 600),
        ("低谷", 0.15, 2, 150),
    ]

    for name, gpu, queue, latency in scenarios:
        print(f"\n场景: {name}")
        # 记录多个指标点(模拟5分钟)
        for _ in range(3):
            metrics = ScalingMetrics(
                timestamp=time.time(),
                gpu_utilization=gpu + random.uniform(-0.05, 0.05),
                queue_depth=int(queue + random.randint(-5, 5)),
                latency_p95_ms=latency + random.uniform(-50, 50),
                error_rate=random.uniform(0, 0.02),
                active_replicas=scaler.current_replicas,
            )
            scaler.record_metrics(metrics)

        # 评估
        action = scaler.evaluate()
        if action == ScaleAction.NO_ACTION:
            print(f"  操作: 保持 {scaler.current_replicas} 实例")

        time.sleep(0.1)  # 模拟时间流逝, 避免cooldown阻塞

    # 打印最终状态
    print(f"\n{scaler.get_status()}")


if __name__ == "__main__":
    demo_autoscaler()
```

---

## Docker与K8s部署

### 容器化部署配置

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Docker + Kubernetes 部署架构                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Kubernetes Cluster                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Namespace: llm-production                                          │   │
│  │                                                                     │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │ Ingress (TLS终止, 路由)                                       │  │   │
│  │  │ llm-api.example.com -> Service                                │  │   │
│  │  └────────────────────────────┬──────────────────────────────────┘  │   │
│  │                               │                                     │   │
│  │  ┌────────────────────────────▼──────────────────────────────────┐  │   │
│  │  │ Service: vllm-service (ClusterIP)                             │  │   │
│  │  │ Port: 8000 -> targetPort: 8000                                │  │   │
│  │  └────────────────────────────┬──────────────────────────────────┘  │   │
│  │                               │                                     │   │
│  │  ┌────────────────────────────▼──────────────────────────────────┐  │   │
│  │  │ Deployment: vllm-server (replicas: 3)                         │  │   │
│  │  │                                                               │  │   │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                      │  │   │
│  │  │  │  Pod 1  │  │  Pod 2  │  │  Pod 3  │                      │  │   │
│  │  │  │ vLLM    │  │ vLLM    │  │ vLLM    │                      │  │   │
│  │  │  │ GPU: 1  │  │ GPU: 1  │  │ GPU: 1  │                      │  │   │
│  │  │  └─────────┘  └─────────┘  └─────────┘                      │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                     │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                         │   │
│  │  │ HPA (自动扩缩容)│  │ PDB (中断预算) │                         │   │
│  │  │ min:2 max:10    │  │ minAvailable: 2 │                         │   │
│  │  └─────────────────┘  └─────────────────┘                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### K8s部署配置生成

```python
"""
Kubernetes部署配置生成器
生成: Dockerfile、Deployment、Service、HPA、Ingress
"""


def generate_dockerfile() -> str:
    """生成vLLM服务的Dockerfile"""
    return """# Dockerfile for vLLM LLM Service
FROM vllm/vllm-openai:latest

# 设置环境变量
ENV MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
ENV MAX_MODEL_LEN=8192
ENV GPU_MEMORY_UTILIZATION=0.90
ENV TENSOR_PARALLEL_SIZE=1

# 复制自定义配置(如有)
COPY ./config /app/config

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD python -m vllm.entrypoints.openai.api_server \\
    --model $MODEL_NAME \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --max-model-len $MAX_MODEL_LEN \\
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \\
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE
"""


def generate_k8s_deployment() -> str:
    """生成Kubernetes Deployment配置"""
    return """# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
  namespace: llm-production
  labels:
    app: vllm-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vllm-server
  template:
    metadata:
      labels:
        app: vllm-server
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: vllm
        image: registry.example.com/vllm-server:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_NAME
          value: "meta-llama/Llama-3.1-8B-Instruct"
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-credentials
              key: token
        resources:
          requests:
            cpu: "4"
            memory: "32Gi"
            nvidia.com/gpu: "1"
          limits:
            cpu: "8"
            memory: "64Gi"
            nvidia.com/gpu: "1"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 180
          periodSeconds: 30
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
      nodeSelector:
        gpu-type: a100

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  namespace: llm-production
spec:
  selector:
    app: vllm-server
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP

---
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
  namespace: llm-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: llm_request_queue_depth
      target:
        type: AverageValue
        averageValue: "30"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 120
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120
"""


def demo_k8s_deployment():
    """K8s部署配置演示"""
    print("=" * 60)
    print("Kubernetes LLM部署配置")
    print("=" * 60)

    print("\n--- Dockerfile ---")
    print(generate_dockerfile())

    print("\n--- Kubernetes配置 ---")
    print(generate_k8s_deployment())


if __name__ == "__main__":
    demo_k8s_deployment()
```

---

## 生产部署最佳实践

### 部署检查清单

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LLM部署生产最佳实践                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 性能优化                           2. 可靠性                            │
│  ┌──────────────────────────────┐     ┌──────────────────────────────┐     │
│  │ - 使用量化模型 (AWQ/GPTQ)   │     │ - 多副本部署 (>=2)           │     │
│  │ - 启用Continuous Batching   │     │ - 健康检查 + 自动重启        │     │
│  │ - 优化max_model_len         │     │ - 优雅关闭 (drain连接)       │     │
│  │ - GPU Memory Utilization 90%│     │ - 自动扩缩容 (HPA)           │     │
│  │ - 启用Flash Attention       │     │ - Pod中断预算 (PDB)          │     │
│  │ - Tensor Parallel (多GPU)   │     │ - 定期备份配置               │     │
│  └──────────────────────────────┘     └──────────────────────────────┘     │
│                                                                             │
│  3. 安全性                             4. 可观测性                          │
│  ┌──────────────────────────────┐     ┌──────────────────────────────┐     │
│  │ - API Key认证               │     │ - Prometheus指标暴露          │     │
│  │ - HTTPS / TLS终止           │     │ - 请求/响应日志              │     │
│  │ - 速率限制 (Rate Limit)     │     │ - GPU利用率监控              │     │
│  │ - 输入内容过滤              │     │ - 延迟/吞吐量仪表盘         │     │
│  │ - 输出内容审核              │     │ - 错误率告警                 │     │
│  │ - 网络隔离 (VPC)           │     │ - 成本追踪                   │     │
│  └──────────────────────────────┘     └──────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 总结

本教程涵盖了LLM模型部署与服务的核心内容:

1. **部署概述**: 推理引擎对比(vLLM/TGI/TRT-LLM/Ollama)、选型指南
2. **vLLM高性能推理**: PagedAttention原理、启动配置、API调用、性能基准测试
3. **TGI部署实战**: Docker部署、API格式、docker-compose编排
4. **FastAPI + vLLM完整服务**: 认证、限流、流式输出、指标收集的生产级服务
5. **负载均衡**: 多策略均衡(轮询/最少连接/GPU感知)、Nginx配置
6. **自动扩缩容**: 基于GPU/队列/延迟的扩缩策略、K8s HPA配置
7. **Docker与K8s部署**: Dockerfile、Deployment、Service、HPA完整配置
8. **最佳实践**: 性能优化/可靠性/安全性/可观测性检查清单

## 参考资源

- [vLLM官方文档](https://docs.vllm.ai/)
- [TGI官方文档](https://huggingface.co/docs/text-generation-inference/)
- [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [Ollama官方文档](https://ollama.ai/)
- [FastAPI文档](https://fastapi.tiangolo.com/)
- [Kubernetes GPU调度](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)

---

**创建时间**: 2024-01-15
**最后更新**: 2024-01-15
