# 模型优化技术

## 目录
1. [优化概述](#优化概述)
2. [模型量化](#模型量化)
3. [llama.cpp本地推理](#llamacpp本地推理)
4. [知识蒸馏](#知识蒸馏)
5. [模型剪枝](#模型剪枝)
6. [推理加速引擎](#推理加速引擎)
7. [性能对比实验](#性能对比实验)
8. [FP8量化与下一代精度](#fp8量化与下一代精度)
9. [投机解码](#投机解码)
10. [生产部署检查清单](#生产部署检查清单)

---

## 优化概述

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    模型优化技术全景图                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  原始模型: Llama-3.1-70B (FP16)                                        │
│  大小: 140GB  |  显存: 140GB+  |  延迟: ~500ms/token  |  成本: $$$    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    优化技术矩阵                                  │   │
│  │                                                                 │   │
│  │  1. 量化 (Quantization)          2. 蒸馏 (Distillation)        │   │
│  │  ┌──────────────────────┐       ┌──────────────────────┐      │   │
│  │  │ FP16 → INT8/INT4    │       │ 大模型 → 小模型       │      │   │
│  │  │ 减少模型体积 2-4x   │       │ 知识迁移             │      │   │
│  │  │ 精度损失可控        │       │ 保持大部分能力       │      │   │
│  │  │ 推理速度提升 1.5-3x │       │ 训练成本较高         │      │   │
│  │  └──────────────────────┘       └──────────────────────┘      │   │
│  │                                                                 │   │
│  │  3. 剪枝 (Pruning)              4. 推理引擎优化                │   │
│  │  ┌──────────────────────┐       ┌──────────────────────┐      │   │
│  │  │ 移除冗余参数/层     │       │ vLLM: PagedAttention │      │   │
│  │  │ 结构化/非结构化     │       │ TensorRT: 图优化     │      │   │
│  │  │ 模型体积减少 30-70% │       │ FlashAttention: IO   │      │   │
│  │  │ 需要微调恢复精度    │       │ 吞吐量提升 2-10x     │      │   │
│  │  └──────────────────────┘       └──────────────────────┘      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  优化后效果 (Llama-3.1-70B 为例):                                      │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  技术              │ 模型大小 │ 显存需求 │ 速度 │ 精度保留  │      │
│  ├────────────────────┼─────────┼─────────┼──────┼──────────┤      │
│  │  原始 FP16         │ 140 GB  │ 140 GB  │ 1x   │ 100%     │      │
│  │  INT8 量化         │  70 GB  │  70 GB  │ 1.5x │ ~99%     │      │
│  │  INT4 (GPTQ)      │  35 GB  │  35 GB  │ 2x   │ ~97%     │      │
│  │  INT4 (AWQ)       │  35 GB  │  35 GB  │ 2.5x │ ~98%     │      │
│  │  GGUF Q4_K_M      │  40 GB  │  40 GB  │ 2x   │ ~97%     │      │
│  │  INT4 + vLLM      │  35 GB  │  40 GB  │ 5x   │ ~97%     │      │
│  │  INT4 + TensorRT  │  35 GB  │  38 GB  │ 8x   │ ~96%     │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  优化策略决策树:                                                        │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │  需要本地/边缘部署?                                      │          │
│  │    是 → GGUF + llama.cpp (CPU/Apple Silicon)            │          │
│  │    否 ↓                                                  │          │
│  │  有GPU?                                                  │          │
│  │    否 → GGUF + llama.cpp (CPU)                          │          │
│  │    是 ↓                                                  │          │
│  │  需要高吞吐量?                                           │          │
│  │    是 → vLLM + AWQ/GPTQ量化                            │          │
│  │    否 ↓                                                  │          │
│  │  需要最低延迟?                                           │          │
│  │    是 → TensorRT-LLM + INT4                             │          │
│  │    否 → vLLM + FP16 或 INT8                             │          │
│  └──────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 详细说明

模型优化是将大语言模型部署到生产环境的关键步骤。原始的FP16模型体积大、推理慢、成本高，需要通过各种优化技术来降低部署成本并提升推理速度。

**为什么需要模型优化？**

1. **降低硬件成本**：量化后的模型显存需求减少2-4倍
2. **提升推理速度**：优化后的模型推理速度提升2-10倍
3. **支持边缘部署**：量化后的模型可以在消费级硬件上运行
4. **降低运营成本**：更少的GPU = 更低的云服务费用

**主流量化格式对比：**

| 格式 | 开发者 | 量化方式 | 优势 | 推理引擎 |
|------|--------|---------|------|---------|
| **GPTQ** | MIT | 后训练量化(PTQ) | 精度好 | vLLM, ExLlama |
| **AWQ** | MIT | 激活感知量化 | 速度快、精度高 | vLLM, TGI |
| **GGUF** | llama.cpp | 混合量化 | CPU友好、跨平台 | llama.cpp |
| **bitsandbytes** | HF | 动态量化 | 简单易用 | Transformers |
| **AQLM** | Yandex | 加性量化 | 极低比特(2-bit) | vLLM |

### 代码示例

```python
# 模型优化概述 - 环境检测与方案推荐
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class HardwareInfo:
    """硬件信息"""
    gpu_name: str = "无"
    gpu_memory_gb: float = 0
    gpu_count: int = 0
    cpu_cores: int = 0
    ram_gb: float = 0


def detect_hardware() -> HardwareInfo:
    """检测系统硬件"""
    import os
    import platform

    info = HardwareInfo()
    info.cpu_cores = os.cpu_count() or 1
    info.ram_gb = 16.0  # 默认值

    try:
        import torch
        if torch.cuda.is_available():
            info.gpu_count = torch.cuda.device_count()
            info.gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info.gpu_memory_gb = props.total_mem / (1024**3)
    except ImportError:
        pass

    try:
        import psutil
        info.ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

    return info


def recommend_optimization(model_size_b: float,
                            hardware: HardwareInfo) -> Dict:
    """根据硬件推荐优化方案"""
    model_fp16_gb = model_size_b * 2  # FP16模型大小

    recommendations = []

    if hardware.gpu_memory_gb >= model_fp16_gb * 1.2:
        recommendations.append({
            "方案": "FP16 + vLLM",
            "说明": "显存充足，可直接运行FP16获得最佳精度",
            "适合": "生产环境，高质量需求"
        })

    if hardware.gpu_memory_gb >= model_size_b * 0.6:
        recommendations.append({
            "方案": "AWQ/GPTQ INT4 + vLLM",
            "说明": f"量化后约{model_size_b*0.5:.0f}GB，当前显存可承载",
            "适合": "生产环境，成本敏感"
        })

    if hardware.ram_gb >= model_size_b * 0.6:
        recommendations.append({
            "方案": "GGUF Q4_K_M + llama.cpp",
            "说明": "CPU推理，无需GPU",
            "适合": "本地开发、边缘部署"
        })

    if not recommendations:
        recommendations.append({
            "方案": "使用更小的模型或云服务",
            "说明": f"当前硬件不足以运行{model_size_b}B模型",
            "适合": "考虑7B/3B模型或API服务"
        })

    return {
        "模型": f"{model_size_b}B 参数",
        "FP16大小": f"{model_fp16_gb:.0f} GB",
        "GPU": f"{hardware.gpu_name} ({hardware.gpu_memory_gb:.0f}GB)" if hardware.gpu_count > 0 else "无",
        "RAM": f"{hardware.ram_gb:.0f} GB",
        "推荐方案": recommendations
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  模型优化方案推荐")
    print("=" * 60)

    hw = detect_hardware()
    print(f"\n硬件信息:")
    print(f"  GPU: {hw.gpu_name}")
    print(f"  GPU显存: {hw.gpu_memory_gb:.1f} GB")
    print(f"  CPU核心: {hw.cpu_cores}")
    print(f"  内存: {hw.ram_gb:.1f} GB")

    for model_size in [7, 13, 70]:
        result = recommend_optimization(model_size, hw)
        print(f"\n{'='*50}")
        print(f"  {result['模型']}")
        print(f"{'='*50}")
        print(f"  FP16大小: {result['FP16大小']}")
        for i, rec in enumerate(result['推荐方案'], 1):
            print(f"\n  方案{i}: {rec['方案']}")
            print(f"    {rec['说明']}")
            print(f"    适合: {rec['适合']}")
```

---

## 模型量化

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    模型量化技术详解                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  量化原理: 用低精度数据类型表示模型权重                                  │
│                                                                         │
│  FP32 (32-bit):  [sign|exponent(8)|mantissa(23)]  精度最高 4字节/参数   │
│  FP16 (16-bit):  [sign|exponent(5)|mantissa(10)]  标准训练 2字节/参数   │
│  BF16 (16-bit):  [sign|exponent(8)|mantissa(7)]   大模型训练 2字节/参数 │
│  INT8 (8-bit):   [-128 ~ 127]                     8倍压缩  1字节/参数  │
│  INT4 (4-bit):   [-8 ~ 7]                         16倍压缩 0.5字节/参数│
│                                                                         │
│  量化方法分类:                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                 │   │
│  │  1. 后训练量化 (Post-Training Quantization, PTQ)                │   │
│  │  ┌────────────────────────────────────────────────────┐        │   │
│  │  │  训练完成后直接量化, 不需要额外训练                 │        │   │
│  │  │                                                    │        │   │
│  │  │  GPTQ: 基于Hessian矩阵的逐层量化                  │        │   │
│  │  │  ┌────────┐   ┌────────┐   ┌────────┐             │        │   │
│  │  │  │原始权重│──►│Hessian │──►│量化权重│             │        │   │
│  │  │  │ FP16   │   │近似求解│   │ INT4   │             │        │   │
│  │  │  └────────┘   └────────┘   └────────┘             │        │   │
│  │  │                                                    │        │   │
│  │  │  AWQ: 激活感知权重量化                             │        │   │
│  │  │  ┌────────┐   ┌────────┐   ┌────────┐             │        │   │
│  │  │  │原始权重│──►│分析激活│──►│重要通道│             │        │   │
│  │  │  │ FP16   │   │分布    │   │保护量化│             │        │   │
│  │  │  └────────┘   └────────┘   └────────┘             │        │   │
│  │  └────────────────────────────────────────────────────┘        │   │
│  │                                                                 │   │
│  │  2. 量化感知训练 (Quantization-Aware Training, QAT)             │   │
│  │  ┌────────────────────────────────────────────────────┐        │   │
│  │  │  训练时模拟量化效果, 模型学习适应低精度            │        │   │
│  │  │  精度更好但需要额外训练                            │        │   │
│  │  └────────────────────────────────────────────────────┘        │   │
│  │                                                                 │   │
│  │  3. 动态量化 (Dynamic Quantization)                              │   │
│  │  ┌────────────────────────────────────────────────────┐        │   │
│  │  │  推理时动态计算缩放因子                            │        │   │
│  │  │  bitsandbytes (HuggingFace) 使用此方法             │        │   │
│  │  └────────────────────────────────────────────────────┘        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  GGUF量化级别详解:                                                      │
│  ┌──────────┬───────────┬──────────┬───────────────────────┐           │
│  │  量化级别│ 每参数bit │ 7B大小   │ 精度/速度             │           │
│  ├──────────┼───────────┼──────────┼───────────────────────┤           │
│  │  Q2_K    │  2.6 bit  │  2.8 GB  │ 精度差, 最快最小      │           │
│  │  Q3_K_M  │  3.1 bit  │  3.3 GB  │ 精度较差              │           │
│  │  Q4_0    │  4.0 bit  │  3.8 GB  │ 基础4-bit量化         │           │
│  │  Q4_K_M  │  4.6 bit  │  4.1 GB  │ 推荐! 精度速度平衡    │           │
│  │  Q5_K_M  │  5.3 bit  │  4.8 GB  │ 较好精度              │           │
│  │  Q6_K    │  6.6 bit  │  5.5 GB  │ 接近FP16精度          │           │
│  │  Q8_0    │  8.0 bit  │  7.2 GB  │ 几乎无损              │           │
│  │  F16     │ 16.0 bit  │ 13.5 GB  │ 原始精度              │           │
│  └──────────┴───────────┴──────────┴───────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**模型量化** 是将模型权重从高精度（如FP16）转换为低精度（如INT8/INT4）的过程。这是部署大语言模型最重要的优化技术，可以将模型体积减少2-4倍，同时显著提升推理速度。

**GPTQ（GPT Quantization）：**
- 基于Hessian矩阵的逐层最优量化
- 使用少量校准数据（通常128条）来指导量化
- INT4量化下精度损失约1-3%
- 广泛支持：vLLM、ExLlama2、AutoGPTQ

**AWQ（Activation-aware Weight Quantization）：**
- 观察激活值分布，保护重要的权重通道
- 不需要反向传播，量化速度快
- INT4下精度通常优于GPTQ
- vLLM原生支持，推理速度更快

**GGUF（GPT-Generated Unified Format）：**
- llama.cpp专用格式，支持CPU推理
- 提供多种量化级别（Q2到Q8）
- 支持混合精度量化（关键层用更高精度）
- 跨平台：Windows/Linux/macOS/iOS/Android

### 代码示例

```python
# 模型量化 - 完整实现与使用
import torch
import torch.nn as nn
import time
from typing import Dict, List, Optional


# ===================== 1. 手动INT8量化实现 =====================

class INT8LinearLayer(nn.Module):
    """INT8量化线性层 (教学实现)"""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()

        # 计算per-channel缩放因子
        self.scale = weight.abs().max(dim=1, keepdim=True).values / 127.0
        self.scale = self.scale.float()

        # 量化权重到INT8
        quantized = torch.round(weight / self.scale).clamp(-128, 127)
        self.weight_int8 = quantized.to(torch.int8)

        # bias保持FP32
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 反量化 + 矩阵乘法
        weight_fp = self.weight_int8.float() * self.scale
        output = torch.nn.functional.linear(x, weight_fp, self.bias)
        return output

    @staticmethod
    def quantize_model(model: nn.Module) -> nn.Module:
        """将模型中的所有Linear层量化为INT8"""
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                quantized = INT8LinearLayer(
                    module.weight.data,
                    module.bias.data if module.bias is not None else None
                )
                setattr(model, name, quantized)
            else:
                INT8LinearLayer.quantize_model(module)
        return model


# ===================== 2. GPTQ量化使用 =====================

def quantize_with_gptq(model_name: str, output_dir: str,
                        bits: int = 4):
    """使用AutoGPTQ进行量化

    pip install auto-gptq optimum
    """
    code = f'''
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# 加载模型和分词器
model_name = "{model_name}"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 量化配置
quantize_config = BaseQuantizeConfig(
    bits={bits},                   # 量化位数
    group_size=128,                # 量化分组大小
    desc_act=True,                 # 激活感知排序
    sym=True,                      # 对称量化
)

# 加载模型
model = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config=quantize_config,
    torch_dtype=torch.float16,
)

# 准备校准数据 (128条)
calibration_data = []
texts = [
    "The meaning of life is",
    "In the beginning, there was",
    "Machine learning is a field of",
    # ... 更多校准文本
]
for text in texts:
    tokens = tokenizer(text, return_tensors="pt")
    calibration_data.append(tokens.input_ids)

# 执行量化
model.quantize(calibration_data)

# 保存量化模型
model.save_quantized("{output_dir}")
tokenizer.save_pretrained("{output_dir}")
print(f"GPTQ量化完成: {output_dir}")
'''
    return code


# ===================== 3. AWQ量化使用 =====================

def quantize_with_awq(model_name: str, output_dir: str):
    """使用AutoAWQ进行量化

    pip install autoawq
    """
    code = f'''
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(
    "{model_name}",
    safetensors=True,
)
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

# AWQ量化配置
quant_config = {{
    "zero_point": True,       # 零点量化
    "q_group_size": 128,      # 分组大小
    "w_bit": 4,               # 权重位数
    "version": "GEMM",        # 量化版本
}}

# 执行量化 (自动使用校准数据)
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data="pileval",      # 使用内置校准数据集
)

# 保存
model.save_quantized("{output_dir}")
tokenizer.save_pretrained("{output_dir}")
print(f"AWQ量化完成: {output_dir}")
'''
    return code


# ===================== 4. 量化效果对比实验 =====================

def compare_quantization_effects():
    """模拟量化效果对比"""
    print("=" * 70)
    print("  量化效果对比实验")
    print("=" * 70)

    # 创建模拟模型
    torch.manual_seed(42)
    input_dim, output_dim = 4096, 4096
    weight = torch.randn(output_dim, input_dim) * 0.02
    x = torch.randn(1, 32, input_dim)

    # FP16基准
    weight_fp16 = weight.half()
    output_fp16 = torch.nn.functional.linear(x.half(), weight_fp16)

    results = {}

    # INT8量化
    scale_int8 = weight.abs().max(dim=1, keepdim=True).values / 127.0
    weight_int8 = torch.round(weight / scale_int8).clamp(-128, 127)
    weight_dequant_int8 = weight_int8 * scale_int8
    output_int8 = torch.nn.functional.linear(x, weight_dequant_int8)

    error_int8 = (output_fp16.float() - output_int8).abs().mean().item()
    results["INT8"] = {
        "大小": f"{weight.numel() * 1 / 1024 / 1024:.1f} MB",
        "压缩比": "2x",
        "平均误差": f"{error_int8:.6f}",
        "相对误差": f"{error_int8 / output_fp16.float().abs().mean().item():.4%}"
    }

    # INT4量化 (per-channel)
    scale_int4 = weight.abs().max(dim=1, keepdim=True).values / 7.0
    weight_int4 = torch.round(weight / scale_int4).clamp(-8, 7)
    weight_dequant_int4 = weight_int4 * scale_int4
    output_int4 = torch.nn.functional.linear(x, weight_dequant_int4)

    error_int4 = (output_fp16.float() - output_int4).abs().mean().item()
    results["INT4"] = {
        "大小": f"{weight.numel() * 0.5 / 1024 / 1024:.1f} MB",
        "压缩比": "4x",
        "平均误差": f"{error_int4:.6f}",
        "相对误差": f"{error_int4 / output_fp16.float().abs().mean().item():.4%}"
    }

    # INT4分组量化 (group_size=128)
    group_size = 128
    weight_grouped = weight.reshape(-1, group_size)
    scale_group = weight_grouped.abs().max(dim=1, keepdim=True).values / 7.0
    weight_group_int4 = torch.round(weight_grouped / scale_group).clamp(-8, 7)
    weight_dequant_group = (weight_group_int4 * scale_group).reshape(weight.shape)
    output_group = torch.nn.functional.linear(x, weight_dequant_group)

    error_group = (output_fp16.float() - output_group).abs().mean().item()
    results["INT4-G128"] = {
        "大小": f"{(weight.numel() * 0.5 + weight.numel() / group_size * 2) / 1024 / 1024:.1f} MB",
        "压缩比": "~3.5x",
        "平均误差": f"{error_group:.6f}",
        "相对误差": f"{error_group / output_fp16.float().abs().mean().item():.4%}"
    }

    # 打印结果
    print(f"\n权重矩阵: {input_dim}x{output_dim}")
    print(f"FP16大小: {weight.numel() * 2 / 1024 / 1024:.1f} MB")
    print()

    for name, info in results.items():
        print(f"--- {name} ---")
        for k, v in info.items():
            print(f"  {k}: {v}")
        print()


if __name__ == "__main__":
    compare_quantization_effects()
```

---

## llama.cpp本地推理

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    llama.cpp 本地推理架构                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  llama.cpp: C/C++实现的LLM推理引擎, 支持CPU和GPU                       │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                    支持的硬件平台                             │      │
│  │                                                              │      │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │      │
│  │  │  x86 CPU │ │ARM CPU   │ │NVIDIA GPU│ │Apple     │       │      │
│  │  │ AVX2/512 │ │Neon/SVE  │ │CUDA      │ │Metal/ANE │       │      │
│  │  │ Intel/AMD│ │骁龙/A系列│ │RTX系列   │ │M1-M4     │       │      │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  GGUF格式工作流:                                                        │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐           │
│  │ HF模型   │──►│ 转换脚本 │──►│ GGUF文件 │──►│ llama.cpp│           │
│  │ safetensor│  │ convert  │   │ 量化版本 │   │ 推理引擎 │           │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘           │
│                                                                         │
│  部署方式:                                                              │
│  ┌───────────────────────────────────────────────────────┐             │
│  │  1. llama-cli: 命令行交互                             │             │
│  │     ./llama-cli -m model.gguf -p "Hello"             │             │
│  │                                                       │             │
│  │  2. llama-server: OpenAI兼容API服务                   │             │
│  │     ./llama-server -m model.gguf --port 8080         │             │
│  │     curl http://localhost:8080/v1/chat/completions   │             │
│  │                                                       │             │
│  │  3. Python绑定: llama-cpp-python                      │             │
│  │     from llama_cpp import Llama                       │             │
│  │     llm = Llama(model_path="model.gguf")             │             │
│  │                                                       │             │
│  │  4. Ollama: 一键部署 (基于llama.cpp)                  │             │
│  │     ollama run llama3.1:8b                            │             │
│  └───────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**llama.cpp** 是Georgi Gerganov开发的轻量级LLM推理引擎，用纯C/C++实现，无需Python依赖和GPU驱动。它通过GGUF格式的量化模型，让大语言模型可以在消费级硬件（如笔记本电脑、手机）上高效运行。

**llama.cpp的核心优势：**

1. **跨平台**：Windows/Linux/macOS/iOS/Android全平台支持
2. **CPU推理**：无需GPU，利用CPU的AVX2/AVX512/NEON指令集加速
3. **Apple Silicon**：原生Metal支持，M系列芯片上性能优异
4. **部分GPU卸载**：可将部分层放到GPU上加速
5. **低内存**：通过mmap实现按需加载，内存使用高效
6. **OpenAI兼容API**：内置HTTP服务器，可直接替换OpenAI API

**模型转换与量化流程：**

```bash
# 步骤1: 克隆llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j

# 步骤2: 转换HF模型为GGUF
python convert_hf_to_gguf.py /path/to/model --outtype f16 --outfile model-f16.gguf

# 步骤3: 量化
./llama-quantize model-f16.gguf model-q4_k_m.gguf q4_k_m

# 步骤4: 运行
./llama-cli -m model-q4_k_m.gguf -p "Hello, how are you?" -n 128
```

### 代码示例

```python
# llama.cpp Python绑定使用
# pip install llama-cpp-python


# ===================== 1. 基础使用 =====================

def basic_llama_cpp_usage():
    """llama-cpp-python基础使用"""
    code = '''
from llama_cpp import Llama

# 加载GGUF模型
llm = Llama(
    model_path="./models/llama-3.1-8b-instruct-q4_k_m.gguf",
    n_ctx=4096,           # 上下文窗口大小
    n_threads=8,          # CPU线程数
    n_gpu_layers=35,      # GPU层数 (0=纯CPU, -1=全部GPU)
    verbose=False,
)

# 文本补全
output = llm(
    "The meaning of life is",
    max_tokens=128,
    temperature=0.7,
    top_p=0.9,
    stop=["\\n\\n"],
)
print(output["choices"][0]["text"])

# Chat Completion (对话)
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "用Python写一个快速排序算法"},
    ],
    max_tokens=512,
    temperature=0.7,
)
print(response["choices"][0]["message"]["content"])
'''
    return code


# ===================== 2. OpenAI兼容服务 =====================

def llama_cpp_server():
    """llama.cpp服务器模式"""
    code = '''
# 启动服务器 (命令行)
# ./llama-server -m model.gguf --port 8080 --host 0.0.0.0

# Python客户端 (使用OpenAI SDK)
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

# 完全兼容OpenAI API
response = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "解释什么是量化"}
    ],
    temperature=0.7,
    max_tokens=256,
)
print(response.choices[0].message.content)

# 流式输出
stream = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "讲个笑话"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
'''
    return code


# ===================== 3. Ollama一键部署 =====================

def ollama_usage():
    """Ollama使用指南"""
    guide = """
# Ollama: 基于llama.cpp的一键部署工具

# 安装
# macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh
# Windows: 下载安装包 https://ollama.com/download

# 拉取并运行模型
ollama run llama3.1:8b            # 默认Q4_K_M量化
ollama run llama3.1:8b-instruct   # 指令微调版
ollama run llama3.1:70b           # 70B模型
ollama run qwen2.5:7b             # Qwen模型
ollama run codellama:7b           # 代码模型

# 列出本地模型
ollama list

# API调用
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "What is machine learning?",
  "stream": false
}'

# Python使用
import ollama

response = ollama.chat(
    model='llama3.1:8b',
    messages=[
        {'role': 'user', 'content': '什么是深度学习？'}
    ]
)
print(response['message']['content'])
"""
    return guide


if __name__ == "__main__":
    print("=" * 60)
    print("  llama.cpp 本地推理指南")
    print("=" * 60)

    print("\n--- 基础使用 ---")
    print(basic_llama_cpp_usage())

    print("\n--- OpenAI兼容服务 ---")
    print(llama_cpp_server())

    print("\n--- Ollama一键部署 ---")
    print(ollama_usage())
```

---

## 知识蒸馏

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    知识蒸馏 (Knowledge Distillation)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  核心思想: 让小模型(Student)学习大模型(Teacher)的"知识"                 │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                                                              │      │
│  │  Teacher Model (大模型, 已训练好)                            │      │
│  │  ┌──────────────────────┐                                   │      │
│  │  │  Llama-3.1-70B       │                                   │      │
│  │  │  输入 "猫是..." ──────┼──► Soft Labels (概率分布)        │      │
│  │  │                      │     [动物:0.6, 宠物:0.3, ...]     │      │
│  │  └──────────────────────┘                                   │      │
│  │           │                                                  │      │
│  │           │ 蒸馏损失 = KL散度(Teacher分布, Student分布)       │      │
│  │           │         + 交叉熵(Student输出, Hard Labels)       │      │
│  │           ▼                                                  │      │
│  │  Student Model (小模型, 待训练)                              │      │
│  │  ┌──────────────────────┐                                   │      │
│  │  │  Llama-3.1-8B        │                                   │      │
│  │  │  输入 "猫是..." ──────┼──► 学习Teacher的输出分布         │      │
│  │  │                      │     模仿大模型的推理能力          │      │
│  │  └──────────────────────┘                                   │      │
│  │                                                              │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  蒸馏损失函数:                                                          │
│  L = alpha * L_distill + (1-alpha) * L_task                            │
│                                                                         │
│  L_distill = KL(softmax(z_t/T), softmax(z_s/T)) * T^2                 │
│  L_task    = CrossEntropy(z_s, y_true)                                  │
│                                                                         │
│  T: 温度参数 (通常2-10), 越高越"软"                                    │
│  alpha: 蒸馏损失权重 (通常0.5-0.9)                                      │
│                                                                         │
│  LLM蒸馏方法:                                                           │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │  1. 输出蒸馏: Student学习Teacher的输出logits             │          │
│  │  2. 特征蒸馏: Student学习Teacher的中间层特征             │          │
│  │  3. 数据蒸馏: 用Teacher生成训练数据给Student              │          │
│  │     (Alpaca, Vicuna等均使用此方法)                        │          │
│  └──────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**知识蒸馏（Knowledge Distillation）** 是一种模型压缩技术，核心思想是用一个大模型（Teacher）的"知识"来训练一个小模型（Student），使小模型获得接近大模型的能力。

**在LLM领域，蒸馏有三种主要形式：**

1. **Logit蒸馏**：Student学习Teacher的输出概率分布
2. **特征蒸馏**：Student学习Teacher的隐藏层表示
3. **数据蒸馏**：用Teacher生成高质量训练数据，再用这些数据训练Student（最常用）

**温度参数T的作用：** 温度越高，Teacher的输出概率分布越"平滑"，能传递更多关于类别间关系的信息。例如T=1时输出可能是[0.9, 0.05, 0.05]，T=5时变为[0.4, 0.3, 0.3]，后者包含更丰富的类别相似度信息。

### 代码示例

```python
# 知识蒸馏 - 完整实现
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class DistillationLoss(nn.Module):
    """蒸馏损失函数"""

    def __init__(self, temperature: float = 4.0,
                 alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            student_logits: Student模型输出 (batch, seq_len, vocab_size)
            teacher_logits: Teacher模型输出 (batch, seq_len, vocab_size)
            labels: 真实标签 (batch, seq_len)
        """
        T = self.temperature

        # 蒸馏损失: KL散度
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)

        distill_loss = F.kl_div(
            student_soft.view(-1, student_soft.size(-1)),
            teacher_soft.view(-1, teacher_soft.size(-1)),
            reduction='batchmean'
        ) * (T * T)  # T^2 缩放

        # 任务损失: 交叉熵
        task_loss = self.ce_loss(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )

        # 总损失
        total_loss = (self.alpha * distill_loss +
                      (1 - self.alpha) * task_loss)

        return {
            "total_loss": total_loss,
            "distill_loss": distill_loss,
            "task_loss": task_loss,
        }


class DistillationTrainer:
    """蒸馏训练器"""

    def __init__(self, teacher_model, student_model,
                 temperature=4.0, alpha=0.7, lr=1e-4):
        self.teacher = teacher_model
        self.student = student_model
        self.criterion = DistillationLoss(temperature, alpha)
        self.optimizer = torch.optim.AdamW(
            student_model.parameters(), lr=lr
        )

        # Teacher设为评估模式, 不更新参数
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train_step(self, input_ids, labels):
        """单步训练"""
        self.student.train()

        # Teacher前向传播 (不计算梯度)
        with torch.no_grad():
            teacher_logits = self.teacher(input_ids)

        # Student前向传播
        student_logits = self.student(input_ids)

        # 计算蒸馏损失
        losses = self.criterion(student_logits, teacher_logits, labels)

        # 反向传播
        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            self.student.parameters(), max_norm=1.0
        )
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}


# ===================== 数据蒸馏示例 =====================

def data_distillation_example():
    """数据蒸馏: 用大模型生成训练数据"""
    code = '''
# 数据蒸馏: 用Teacher生成训练数据, 再训练Student
# 这是LLM领域最常用的蒸馏方法

from openai import OpenAI

client = OpenAI()  # 或本地部署的大模型

# 1. 定义种子任务
seed_tasks = [
    "解释什么是机器学习",
    "写一个Python快速排序",
    "翻译: Hello World",
]

# 2. 用Teacher生成高质量回答
training_data = []
for task in seed_tasks:
    response = client.chat.completions.create(
        model="gpt-4",  # Teacher模型
        messages=[
            {"role": "system", "content": "你是一个专业的AI助手"},
            {"role": "user", "content": task}
        ],
        temperature=0.7,
    )

    training_data.append({
        "instruction": task,
        "output": response.choices[0].message.content
    })

# 3. 用生成的数据微调Student模型
# 使用LoRA/QLoRA微调7B/8B模型
# (参见 03_fine_tuning.md)
'''
    return code


# ===================== 演示 =====================

if __name__ == "__main__":
    print("=" * 60)
    print("  知识蒸馏演示")
    print("=" * 60)

    # 创建简化模型
    vocab_size = 1000
    d_model = 256

    teacher = nn.Sequential(
        nn.Embedding(vocab_size, d_model),
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, vocab_size)
    )

    student = nn.Sequential(
        nn.Embedding(vocab_size, d_model // 2),
        nn.Linear(d_model // 2, d_model // 2),
        nn.ReLU(),
        nn.Linear(d_model // 2, vocab_size)
    )

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"\nTeacher参数: {teacher_params:,}")
    print(f"Student参数: {student_params:,}")
    print(f"压缩比: {teacher_params/student_params:.1f}x")

    # 蒸馏训练
    trainer = DistillationTrainer(teacher, student)

    input_ids = torch.randint(0, vocab_size, (4, 32))
    labels = torch.randint(0, vocab_size, (4, 32))

    print("\n蒸馏训练:")
    for step in range(10):
        losses = trainer.train_step(input_ids, labels)
        if (step + 1) % 2 == 0:
            print(f"  Step {step+1}: "
                  f"total={losses['total_loss']:.4f} "
                  f"distill={losses['distill_loss']:.4f} "
                  f"task={losses['task_loss']:.4f}")
```

---

## 模型剪枝

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    模型剪枝 (Model Pruning)                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  核心思想: 移除模型中不重要的参数/结构, 减少计算量                       │
│                                                                         │
│  剪枝类型:                                                              │
│  ┌───────────────────────────────────────────────────────────┐         │
│  │                                                           │         │
│  │  1. 非结构化剪枝 (Unstructured)                          │         │
│  │  ┌─────────────┐      ┌─────────────┐                   │         │
│  │  │ 1.2 0.3 0.8 │      │ 1.2  0  0.8 │                   │         │
│  │  │ 0.1 2.1 0.4 │ ──►  │  0  2.1  0  │  剪掉小权重      │         │
│  │  │ 0.9 0.2 1.5 │      │ 0.9  0  1.5 │  产生稀疏矩阵    │         │
│  │  └─────────────┘      └─────────────┘                   │         │
│  │  优点: 灵活, 压缩比高  缺点: 需要稀疏硬件支持          │         │
│  │                                                           │         │
│  │  2. 结构化剪枝 (Structured)                               │         │
│  │  ┌─────────────┐      ┌─────────┐                       │         │
│  │  │ ████ ████   │      │ ████    │                       │         │
│  │  │ ████ ████   │ ──►  │ ████    │  剪掉整个通道/层      │         │
│  │  │ ████ ████   │      │ ████    │  模型结构变小          │         │
│  │  │ ████ ████   │      └─────────┘                       │         │
│  │  └─────────────┘                                         │         │
│  │  优点: 直接加速  缺点: 粒度粗, 精度损失可能较大         │         │
│  │                                                           │         │
│  │  3. LLM专用: 层剪枝 / 注意力头剪枝                       │         │
│  │  ┌──────────────────────────────────────────┐            │         │
│  │  │ Layer1 Layer2 Layer3 ... Layer32          │            │         │
│  │  │   ↓                                      │            │         │
│  │  │ Layer1   -    Layer3 ... Layer32          │ 移除冗余层 │         │
│  │  └──────────────────────────────────────────┘            │         │
│  └───────────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**模型剪枝** 通过移除模型中不重要的参数或结构来减少模型大小和计算量。对于LLM，常见的剪枝方法包括：

1. **权重剪枝**：将小于阈值的权重设为零
2. **注意力头剪枝**：移除不重要的注意力头
3. **层剪枝**：移除整个Transformer层
4. **SparseGPT/Wanda**：LLM专用的一次性剪枝方法

### 代码示例

```python
# 模型剪枝 - PyTorch实现
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Tuple


# ===================== 1. 非结构化剪枝 =====================

def unstructured_pruning_demo():
    """非结构化剪枝演示"""
    print("=" * 60)
    print("  非结构化剪枝演示")
    print("=" * 60)

    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    # 统计原始参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"原始参数量: {total_params:,}")

    # 对所有Linear层进行L1非结构化剪枝 (30%)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.3)

    # 统计稀疏度
    zero_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            zeros = (module.weight == 0).sum().item()
            total = module.weight.numel()
            sparsity = zeros / total * 100
            zero_params += zeros
            print(f"  {name}: 稀疏度 {sparsity:.1f}%")

    overall_sparsity = zero_params / total_params * 100
    print(f"\n整体稀疏度: {overall_sparsity:.1f}%")

    # 永久化剪枝 (移除mask)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.remove(module, 'weight')


# ===================== 2. 结构化剪枝 =====================

def structured_pruning_demo():
    """结构化剪枝: 移除整个通道"""
    print("\n" + "=" * 60)
    print("  结构化剪枝演示")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    # 对第一个Linear层进行结构化剪枝 (移除50%的输出通道)
    layer = model[0]
    prune.ln_structured(layer, name='weight', amount=0.5, n=2, dim=0)

    # 查看被剪掉的通道
    mask = layer.weight_mask
    active_channels = mask.sum(dim=1) > 0
    print(f"原始输出通道: 256")
    print(f"保留通道数: {active_channels.sum().item()}")
    print(f"剪除通道数: {(~active_channels).sum().item()}")


# ===================== 3. LLM层剪枝 =====================

def llm_layer_pruning_analysis():
    """LLM层剪枝分析"""
    print("\n" + "=" * 60)
    print("  LLM层剪枝分析")
    print("=" * 60)

    # 模拟各层的重要性分数 (基于困惑度变化)
    n_layers = 32
    torch.manual_seed(42)

    # 模拟: 中间层通常最不重要
    importance = torch.zeros(n_layers)
    for i in range(n_layers):
        if i < 4:  # 前几层重要
            importance[i] = 5.0 + torch.randn(1).item() * 0.5
        elif i > n_layers - 4:  # 后几层重要
            importance[i] = 4.5 + torch.randn(1).item() * 0.5
        else:  # 中间层不太重要
            importance[i] = 1.0 + torch.randn(1).item() * 0.3

    print("\n各层重要性分数 (越高越重要):")
    for i, imp in enumerate(importance):
        bar = "#" * int(imp * 5)
        print(f"  Layer {i:2d}: {imp:.2f} {bar}")

    # 选择要剪掉的层 (重要性最低的25%)
    n_prune = n_layers // 4
    _, indices = importance.topk(n_prune, largest=False)
    prune_layers = sorted(indices.tolist())

    print(f"\n建议剪除的层 ({n_prune}层): {prune_layers}")
    print(f"剪枝后: {n_layers} → {n_layers - n_prune} 层")
    print(f"理论加速: {n_layers / (n_layers - n_prune):.2f}x")


if __name__ == "__main__":
    unstructured_pruning_demo()
    structured_pruning_demo()
    llm_layer_pruning_analysis()
```

---

## 推理加速引擎

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    推理加速引擎对比                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  vLLM                                                           │   │
│  │  ┌───────────────────────────────────────────────────────────┐ │   │
│  │  │  PagedAttention: KV Cache分页管理                         │ │   │
│  │  │  ┌────┐ ┌────┐ ┌────┐         不再需要预分配连续内存    │ │   │
│  │  │  │Page│ │Page│ │Page│  ──►   按需分配, 利用率95%+       │ │   │
│  │  │  │ 1  │ │ 2  │ │ 3  │         支持Prefix Caching       │ │   │
│  │  │  └────┘ └────┘ └────┘                                   │ │   │
│  │  │  Continuous Batching: 动态批处理                          │ │   │
│  │  │  吞吐量提升 5-24x  |  OpenAI兼容API                     │ │   │
│  │  └───────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  TensorRT-LLM (NVIDIA)                                         │   │
│  │  ┌───────────────────────────────────────────────────────────┐ │   │
│  │  │  图优化: 算子融合、内存优化、Kernel自动调优              │ │   │
│  │  │  FP8量化: Hopper架构(H100)原生支持                       │ │   │
│  │  │  Inflight Batching: 请求级别的批处理                     │ │   │
│  │  │  延迟最低, NVIDIA GPU专用                                │ │   │
│  │  └───────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  TGI (Text Generation Inference) - Hugging Face                 │   │
│  │  ┌───────────────────────────────────────────────────────────┐ │   │
│  │  │  Flash Attention 2 集成                                   │ │   │
│  │  │  Token流式输出                                            │ │   │
│  │  │  Tensor并行 (多GPU)                                       │ │   │
│  │  │  Docker一键部署, HuggingFace生态集成                      │ │   │
│  │  └───────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  性能对比 (Llama-3.1-8B, A100 80GB, INT4):                            │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │  引擎          │ 吞吐量(tok/s) │ 首Token延迟│ 使用难度  │          │
│  ├────────────────┼───────────────┼────────────┼──────────┤          │
│  │  Transformers  │    ~50        │   ~200ms   │ 简单     │          │
│  │  vLLM          │   ~500        │    ~80ms   │ 简单     │          │
│  │  TGI           │   ~400        │   ~100ms   │ 中等     │          │
│  │  TensorRT-LLM  │   ~800        │    ~50ms   │ 复杂     │          │
│  │  llama.cpp(GPU)│   ~200        │   ~150ms   │ 简单     │          │
│  └──────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**vLLM** 是UC Berkeley开发的高性能LLM推理引擎，其核心创新PagedAttention通过分页管理KV Cache，将内存利用率从20-40%提升到95%以上。它支持Continuous Batching（连续批处理），能动态管理并发请求，极大提升吞吐量。

**TensorRT-LLM** 是NVIDIA官方的LLM推理优化库，通过图级别的优化（算子融合、内存优化）获得最低延迟。适合对性能要求极高的生产环境，但部署复杂度也最高。

**选择建议：**
- 通用生产环境 --> **vLLM**（简单高效）
- NVIDIA环境追求极致性能 --> **TensorRT-LLM**
- HuggingFace生态集成 --> **TGI**
- 本地/边缘部署 --> **llama.cpp**

### 代码示例

```python
# 推理加速引擎使用示例


# ===================== 1. vLLM使用 =====================

def vllm_example():
    """vLLM部署示例"""
    code = '''
# pip install vllm

from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="auto",              # 自动选择精度
    quantization="awq",        # 使用AWQ量化 (可选)
    tensor_parallel_size=1,    # GPU数量
    gpu_memory_utilization=0.9,# GPU显存利用率
    max_model_len=4096,        # 最大序列长度
)

# 批量推理
prompts = [
    "What is machine learning?",
    "Explain quantum computing",
    "Write a Python hello world",
]

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt[:50]}...")
    print(f"Output: {output.outputs[0].text[:100]}...")
    print()

# ===== vLLM OpenAI兼容服务器 =====
# 命令行启动:
# python -m vllm.entrypoints.openai.api_server \\
#     --model meta-llama/Llama-3.1-8B-Instruct \\
#     --port 8000 \\
#     --dtype auto \\
#     --api-key your-api-key

# Python客户端:
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-api-key")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=256,
)
'''
    return code


# ===================== 2. TGI使用 =====================

def tgi_example():
    """TGI部署示例"""
    code = '''
# Docker部署 TGI
# docker run --gpus all --shm-size 1g -p 8080:80 \\
#     -v /data/models:/data \\
#     ghcr.io/huggingface/text-generation-inference:latest \\
#     --model-id meta-llama/Llama-3.1-8B-Instruct \\
#     --quantize awq \\
#     --max-input-length 4096 \\
#     --max-total-tokens 8192

# Python客户端
from huggingface_hub import InferenceClient

client = InferenceClient("http://localhost:8080")

# 文本生成
output = client.text_generation(
    "What is deep learning?",
    max_new_tokens=256,
    temperature=0.7,
)
print(output)

# 流式输出
for token in client.text_generation(
    "Explain transformers",
    max_new_tokens=256,
    stream=True,
):
    print(token, end="", flush=True)
'''
    return code


# ===================== 3. 性能基准测试 =====================

def benchmark_template():
    """推理性能基准测试模板"""
    code = '''
import time
import torch
from typing import List, Dict


def benchmark_inference(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 128,
    num_runs: int = 5,
) -> Dict:
    """推理性能基准测试"""

    # 预热
    inputs = tokenizer(prompts[0], return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10)

    # 同步GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 测试
    total_tokens = 0
    total_time = 0
    first_token_times = []

    for run in range(num_runs):
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()

            gen_tokens = outputs.shape[1] - input_len
            total_tokens += gen_tokens
            total_time += (end - start)

    # 计算指标
    throughput = total_tokens / total_time
    avg_latency = total_time / (num_runs * len(prompts))

    return {
        "吞吐量": f"{throughput:.1f} tokens/s",
        "平均延迟": f"{avg_latency*1000:.1f} ms",
        "总Token数": total_tokens,
        "总耗时": f"{total_time:.2f}s",
    }
'''
    return code


if __name__ == "__main__":
    print("=" * 60)
    print("  推理加速引擎使用指南")
    print("=" * 60)

    print("\n--- vLLM ---")
    print(vllm_example())

    print("\n--- TGI ---")
    print(tgi_example())
```

---

## 性能对比实验

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    性能对比实验结果                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  测试环境: NVIDIA A100 80GB, Llama-3.1-8B-Instruct                     │
│  测试任务: 128 token输入, 256 token输出, 并发数=16                      │
│                                                                         │
│  吞吐量对比 (tokens/second):                                            │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  TensorRT-LLM INT4  ████████████████████████████████████ 850│      │
│  │  vLLM AWQ           █████████████████████████████       520 │      │
│  │  vLLM FP16          ███████████████████████             420 │      │
│  │  TGI AWQ            ██████████████████████              390 │      │
│  │  llama.cpp GPU Q4   ████████████                        220 │      │
│  │  Transformers FP16  ████                                 60 │      │
│  │  llama.cpp CPU Q4   ██                                   30 │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  首Token延迟 (TTFT, ms):                                               │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  TensorRT-LLM INT4  ██           45ms                       │      │
│  │  vLLM AWQ           ████         75ms                       │      │
│  │  vLLM FP16          █████        95ms                       │      │
│  │  TGI AWQ            ██████       110ms                      │      │
│  │  llama.cpp GPU Q4   ████████     150ms                      │      │
│  │  Transformers FP16  ███████████  200ms                      │      │
│  │  llama.cpp CPU Q4   ███████████████████████████  500ms      │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  显存使用 (GB):                                                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  Transformers FP16  ████████████████████████████  16.5 GB   │      │
│  │  vLLM FP16          ████████████████████          14.5 GB   │      │
│  │  TGI FP16           ████████████████████          14.2 GB   │      │
│  │  vLLM AWQ           ███████████                    6.8 GB   │      │
│  │  TensorRT-LLM INT4  ██████████                     6.2 GB   │      │
│  │  llama.cpp GPU Q4   █████████                      5.8 GB   │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  精度对比 (MMLU基准分数):                                               │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  FP16 (基准)        ████████████████████████████  67.2      │      │
│  │  INT8               ███████████████████████████   66.8      │      │
│  │  AWQ INT4           ██████████████████████████    65.9      │      │
│  │  GPTQ INT4          █████████████████████████     65.1      │      │
│  │  GGUF Q4_K_M        █████████████████████████     65.3      │      │
│  │  GGUF Q3_K_M        ███████████████████████       63.5      │      │
│  │  GGUF Q2_K          █████████████████             58.2      │      │
│  └──────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 详细说明

通过系统的性能对比实验，可以帮助选择最适合自身场景的优化方案。关键指标包括：

1. **吞吐量（Throughput）**：每秒生成的token数，衡量服务能力
2. **首Token延迟（TTFT）**：从请求到第一个token输出的时间，影响用户体验
3. **显存使用**：决定了需要的GPU规格和成本
4. **精度保留**：量化后模型能力的保持程度

### 代码示例

```python
# 性能对比实验 - 完整基准测试框架
import time
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    engine: str
    quantization: str
    throughput_tps: float      # tokens/second
    ttft_ms: float             # 首token延迟
    tpot_ms: float             # 每token生成时间
    memory_gb: float           # 显存使用
    model_size_gb: float       # 模型文件大小
    accuracy_score: float      # 精度分数 (0-100)


class BenchmarkRunner:
    """基准测试运行器"""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult):
        """添加测试结果"""
        self.results.append(result)

    def generate_report(self) -> str:
        """生成对比报告"""
        report = []
        report.append("=" * 75)
        report.append("  模型优化性能对比报告")
        report.append("=" * 75)

        # 按吞吐量排序
        sorted_results = sorted(
            self.results,
            key=lambda x: x.throughput_tps,
            reverse=True
        )

        # 吞吐量对比
        report.append("\n吞吐量排名 (tokens/second):")
        report.append("-" * 60)
        max_tp = max(r.throughput_tps for r in sorted_results)

        for r in sorted_results:
            bar_len = int(r.throughput_tps / max_tp * 40)
            bar = "#" * bar_len
            report.append(
                f"  {r.engine:20s} {r.quantization:8s} "
                f"{bar} {r.throughput_tps:>6.0f}"
            )

        # 延迟对比
        report.append("\n首Token延迟排名 (ms, 越低越好):")
        report.append("-" * 60)
        sorted_by_latency = sorted(
            self.results,
            key=lambda x: x.ttft_ms
        )

        for r in sorted_by_latency:
            bar_len = int(r.ttft_ms / 500 * 40)
            bar = "#" * min(bar_len, 40)
            report.append(
                f"  {r.engine:20s} {r.quantization:8s} "
                f"{bar} {r.ttft_ms:>6.0f}ms"
            )

        # 显存对比
        report.append("\n显存使用 (GB):")
        report.append("-" * 60)
        sorted_by_mem = sorted(
            self.results,
            key=lambda x: x.memory_gb
        )

        for r in sorted_by_mem:
            bar_len = int(r.memory_gb / 20 * 40)
            bar = "#" * bar_len
            report.append(
                f"  {r.engine:20s} {r.quantization:8s} "
                f"{bar} {r.memory_gb:>5.1f}GB"
            )

        # 精度对比
        report.append("\n精度分数 (越高越好):")
        report.append("-" * 60)
        sorted_by_acc = sorted(
            self.results,
            key=lambda x: x.accuracy_score,
            reverse=True
        )

        for r in sorted_by_acc:
            bar_len = int(r.accuracy_score / 100 * 40)
            bar = "#" * bar_len
            report.append(
                f"  {r.engine:20s} {r.quantization:8s} "
                f"{bar} {r.accuracy_score:>5.1f}"
            )

        # 综合推荐
        report.append("\n" + "=" * 60)
        report.append("  综合推荐")
        report.append("=" * 60)
        report.append("  最高吞吐: " + sorted_results[0].engine)
        report.append("  最低延迟: " + sorted_by_latency[0].engine)
        report.append("  最省显存: " + sorted_by_mem[0].engine)
        report.append("  最高精度: " + sorted_by_acc[0].engine)

        return "\n".join(report)

    def save_results(self, filepath: str):
        """保存结果为JSON"""
        data = [asdict(r) for r in self.results]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# ===================== 模拟基准测试结果 =====================

def run_simulated_benchmark():
    """使用模拟数据生成对比报告"""
    runner = BenchmarkRunner()

    # 模拟测试结果 (基于公开基准数据)
    results = [
        BenchmarkResult("Transformers", "FP16", 60, 200, 17, 16.5, 15.0, 67.2),
        BenchmarkResult("vLLM", "FP16", 420, 95, 2.4, 14.5, 15.0, 67.2),
        BenchmarkResult("vLLM", "AWQ-INT4", 520, 75, 1.9, 6.8, 4.5, 65.9),
        BenchmarkResult("TGI", "AWQ-INT4", 390, 110, 2.6, 7.2, 4.5, 65.9),
        BenchmarkResult("TensorRT-LLM", "INT4", 850, 45, 1.2, 6.2, 4.3, 65.5),
        BenchmarkResult("llama.cpp", "Q4_K_M", 220, 150, 4.5, 5.8, 4.1, 65.3),
        BenchmarkResult("llama.cpp-CPU", "Q4_K_M", 30, 500, 33, 0, 4.1, 65.3),
    ]

    for r in results:
        runner.add_result(r)

    report = runner.generate_report()
    print(report)

    return runner


if __name__ == "__main__":
    runner = run_simulated_benchmark()
```

---

## FP8量化与下一代精度

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FP8 量化技术详解                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  浮点数精度演进:                                                        │
│                                                                         │
│  FP32  [1|8|23]  ████████████████████████████████  32 bit  精度极高     │
│  FP16  [1|5|10]  ████████████████                  16 bit  标准推理     │
│  BF16  [1|8|7]   ████████████████                  16 bit  大范围       │
│  FP8   [1|4|3]   ████████                           8 bit  下一代标准   │
│  INT8  [-128~127] ████████                          8 bit  整数量化     │
│  INT4  [-8~7]    ████                               4 bit  极致压缩     │
│                                                                         │
│  FP8 两种格式:                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                 │   │
│  │  E4M3 (4位指数, 3位尾数)                                       │   │
│  │  ┌──────┬──────────┬─────────┐                                 │   │
│  │  │ sign │ exponent │mantissa │                                 │   │
│  │  │  1   │   4      │   3     │                                 │   │
│  │  └──────┴──────────┴─────────┘                                 │   │
│  │  范围: ±448      精度: 较高                                     │   │
│  │  用途: 权重存储, 前向传播                                       │   │
│  │                                                                 │   │
│  │  E5M2 (5位指数, 2位尾数)                                       │   │
│  │  ┌──────┬──────────┬─────────┐                                 │   │
│  │  │ sign │ exponent │mantissa │                                 │   │
│  │  │  1   │   5      │   2     │                                 │   │
│  │  └──────┴──────────┴─────────┘                                 │   │
│  │  范围: ±57344    精度: 较低                                     │   │
│  │  用途: 梯度存储, 反向传播                                       │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  FP8 vs INT8 vs INT4 对比:                                              │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐              │
│  │  特性    │   FP8    │  INT8    │  INT4    │  FP16    │              │
│  ├──────────┼──────────┼──────────┼──────────┼──────────┤              │
│  │ 每参数   │  1 字节  │  1 字节  │ 0.5字节  │  2 字节  │              │
│  │ 动态范围 │   高     │   低     │  极低    │  很高    │              │
│  │ 精度损失 │  <0.5%   │  ~1%     │  ~3%     │   0%     │              │
│  │ 需校准   │  是      │  是      │   是     │   否     │              │
│  │ 硬件支持 │ H100/H200│  广泛    │  广泛    │  广泛    │              │
│  │ 训练支持 │   是     │   否     │   否     │   是     │              │
│  │ 推理速度 │  ~2x     │  ~1.5x   │  ~2.5x   │  1x     │              │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘              │
│                                                                         │
│  硬件支持:                                                              │
│  ┌────────────────────────────────────────────────────────────┐        │
│  │  NVIDIA H100/H200  → 原生FP8 Tensor Core, 最佳性能       │        │
│  │  NVIDIA L40S       → 支持FP8推理                          │        │
│  │  AMD MI300X        → 原生FP8支持                          │        │
│  │  Intel Gaudi 2/3   → FP8支持                              │        │
│  │  NVIDIA A100       → 不支持FP8 (使用INT8替代)             │        │
│  └────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**FP8（8-bit Floating Point）** 是NVIDIA在Hopper架构（H100）中引入的下一代量化精度格式。与INT8量化不同，FP8保留了浮点数的动态范围特性，因此在精度损失极小的情况下实现了显著的推理加速。

**为什么FP8比INT8更好？**

1. **动态范围更大**：FP8使用指数位，可以表示更大范围的数值，避免了INT8中常见的溢出/截断问题
2. **精度损失更小**：在主流LLM基准测试中，FP8量化的精度损失通常<0.5%，而INT8约为1%
3. **硬件原生支持**：H100的FP8 Tensor Core提供了最优的计算效率
4. **支持训练**：FP8可以用于训练（E5M2用于梯度），而INT8只能用于推理

**FP8量化工作流程：**

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 收集校准数据 | 128-512条代表性数据 |
| 2 | 运行校准 | 统计每层激活值分布 |
| 3 | 计算缩放因子 | per-tensor或per-channel |
| 4 | 量化权重 | FP16 -> E4M3 |
| 5 | 量化激活 | 动态或静态缩放 |
| 6 | 导出模型 | TensorRT-LLM或vLLM格式 |

### 代码示例

```python
# FP8 量化完整实现与使用
import torch
import torch.nn as nn
import struct
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ===================== 1. FP8格式模拟实现 =====================

class FP8Simulator:
    """FP8浮点格式模拟器 (教学实现)

    在不支持FP8硬件的环境下模拟FP8的精度特性
    """

    # E4M3: 4位指数, 3位尾数
    E4M3_EXPONENT_BITS = 4
    E4M3_MANTISSA_BITS = 3
    E4M3_BIAS = 7  # 2^(4-1) - 1
    E4M3_MAX = 448.0  # 最大可表示值
    E4M3_MIN = 2**(-9)  # 最小正规数

    # E5M2: 5位指数, 2位尾数
    E5M2_EXPONENT_BITS = 5
    E5M2_MANTISSA_BITS = 2
    E5M2_BIAS = 15  # 2^(5-1) - 1
    E5M2_MAX = 57344.0
    E5M2_MIN = 2**(-16)

    @staticmethod
    def quantize_e4m3(tensor: torch.Tensor,
                       scale: Optional[torch.Tensor] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """将FP32/FP16张量量化为E4M3格式

        Args:
            tensor: 输入张量
            scale: 可选的缩放因子, 为None则自动计算

        Returns:
            (量化后的张量, 缩放因子)
        """
        if scale is None:
            # 自动计算缩放因子: 将最大值映射到E4M3_MAX
            amax = tensor.abs().max()
            scale = amax / FP8Simulator.E4M3_MAX
            scale = scale.clamp(min=1e-12)  # 避免除零

        # 缩放到E4M3范围
        scaled = tensor / scale

        # 裁剪到E4M3范围
        clipped = scaled.clamp(-FP8Simulator.E4M3_MAX,
                                FP8Simulator.E4M3_MAX)

        # 模拟E4M3精度: 3位尾数 = 8个尾数级别
        # 通过round到最近的可表示值来模拟
        sign = clipped.sign()
        abs_val = clipped.abs()

        # 找到最近的2的幂次 (指数部分)
        log2 = torch.floor(torch.log2(abs_val.clamp(min=FP8Simulator.E4M3_MIN)))
        exponent = torch.pow(2.0, log2)

        # 量化尾数部分 (3位 = 8级)
        mantissa = abs_val / exponent  # 1.0 ~ 2.0
        mantissa_quantized = torch.round(mantissa * 8) / 8  # 量化到8级

        # 重建量化值
        quantized = sign * mantissa_quantized * exponent

        # 处理零和极小值
        quantized = torch.where(
            abs_val < FP8Simulator.E4M3_MIN,
            torch.zeros_like(quantized),
            quantized
        )

        return quantized, scale

    @staticmethod
    def quantize_e5m2(tensor: torch.Tensor,
                       scale: Optional[torch.Tensor] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """将张量量化为E5M2格式 (用于梯度)"""
        if scale is None:
            amax = tensor.abs().max()
            scale = amax / FP8Simulator.E5M2_MAX
            scale = scale.clamp(min=1e-12)

        scaled = tensor / scale
        clipped = scaled.clamp(-FP8Simulator.E5M2_MAX,
                                FP8Simulator.E5M2_MAX)

        sign = clipped.sign()
        abs_val = clipped.abs()

        log2 = torch.floor(torch.log2(abs_val.clamp(min=FP8Simulator.E5M2_MIN)))
        exponent = torch.pow(2.0, log2)

        # 2位尾数 = 4级
        mantissa = abs_val / exponent
        mantissa_quantized = torch.round(mantissa * 4) / 4

        quantized = sign * mantissa_quantized * exponent
        quantized = torch.where(
            abs_val < FP8Simulator.E5M2_MIN,
            torch.zeros_like(quantized),
            quantized
        )

        return quantized, scale

    @staticmethod
    def compare_precision():
        """对比不同精度格式的量化误差"""
        print("=" * 70)
        print("  FP8 精度对比实验")
        print("=" * 70)

        # 创建模拟权重分布 (正态分布, 类似真实LLM权重)
        torch.manual_seed(42)
        weights = torch.randn(1024, 1024) * 0.02  # 标准LLM权重范围

        # FP8 E4M3 量化
        e4m3_q, e4m3_scale = FP8Simulator.quantize_e4m3(weights)
        e4m3_dequant = e4m3_q * e4m3_scale
        e4m3_error = (weights - e4m3_dequant).abs().mean()

        # FP8 E5M2 量化
        e5m2_q, e5m2_scale = FP8Simulator.quantize_e5m2(weights)
        e5m2_dequant = e5m2_q * e5m2_scale
        e5m2_error = (weights - e5m2_dequant).abs().mean()

        # INT8 对称量化 (对比)
        int8_scale = weights.abs().max() / 127
        int8_q = torch.round(weights / int8_scale).clamp(-128, 127)
        int8_dequant = int8_q * int8_scale
        int8_error = (weights - int8_dequant).abs().mean()

        # INT4 对称量化 (对比)
        int4_scale = weights.abs().max() / 7
        int4_q = torch.round(weights / int4_scale).clamp(-8, 7)
        int4_dequant = int4_q * int4_scale
        int4_error = (weights - int4_dequant).abs().mean()

        print(f"\n权重分布: mean={weights.mean():.6f}, "
              f"std={weights.std():.6f}, "
              f"range=[{weights.min():.4f}, {weights.max():.4f}]")

        print(f"\n{'格式':<15} {'平均误差':<15} {'相对误差%':<15} {'最大误差':<15}")
        print("-" * 60)

        for name, q, s, err in [
            ("FP8 E4M3", e4m3_q, e4m3_scale, e4m3_error),
            ("FP8 E5M2", e5m2_q, e5m2_scale, e5m2_error),
            ("INT8", int8_q, int8_scale, int8_error),
            ("INT4", int4_q, int4_scale, int4_error),
        ]:
            dequant = q * s if name.startswith("FP8") else q * s
            max_err = (weights - dequant).abs().max()
            rel_err = err / weights.abs().mean() * 100
            print(f"{name:<15} {err.item():<15.8f} {rel_err.item():<15.4f} "
                  f"{max_err.item():<15.8f}")


# ===================== 2. FP8量化线性层 =====================

class FP8Linear(nn.Module):
    """FP8量化线性层

    使用E4M3格式量化权重, 推理时反量化计算
    """

    def __init__(self, weight: torch.Tensor,
                 bias: Optional[torch.Tensor] = None):
        super().__init__()

        # 量化权重为E4M3
        quantized, scale = FP8Simulator.quantize_e4m3(weight)
        self.register_buffer('weight_fp8', quantized)
        self.register_buffer('weight_scale', scale)

        if bias is not None:
            self.register_buffer('bias', bias.clone())
        else:
            self.bias = None

        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 反量化权重
        weight = self.weight_fp8 * self.weight_scale
        return nn.functional.linear(x, weight, self.bias)

    @staticmethod
    def from_linear(linear: nn.Linear) -> 'FP8Linear':
        """从标准Linear层转换"""
        return FP8Linear(
            linear.weight.data,
            linear.bias.data if linear.bias is not None else None
        )


# ===================== 3. FP8模型量化器 =====================

class FP8Quantizer:
    """FP8模型量化器

    支持per-tensor和per-channel两种缩放模式
    """

    def __init__(self, mode: str = "per_tensor"):
        """
        Args:
            mode: "per_tensor" 或 "per_channel"
        """
        assert mode in ("per_tensor", "per_channel")
        self.mode = mode
        self.calibration_data = {}

    def calibrate(self, model: nn.Module, calibration_loader,
                  num_batches: int = 32):
        """收集校准数据, 统计每层激活值分布

        Args:
            model: 待量化模型
            calibration_loader: 校准数据加载器
            num_batches: 使用的校准批次数
        """
        print("开始FP8校准...")
        hooks = []
        activation_ranges = {}

        def make_hook(name):
            def hook_fn(module, input, output):
                if name not in activation_ranges:
                    activation_ranges[name] = {
                        'min': float('inf'),
                        'max': float('-inf'),
                        'amax': 0,
                    }
                if isinstance(output, torch.Tensor):
                    amax = output.abs().max().item()
                    activation_ranges[name]['amax'] = max(
                        activation_ranges[name]['amax'], amax
                    )
                    activation_ranges[name]['min'] = min(
                        activation_ranges[name]['min'],
                        output.min().item()
                    )
                    activation_ranges[name]['max'] = max(
                        activation_ranges[name]['max'],
                        output.max().item()
                    )
            return hook_fn

        # 注册Hook
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(make_hook(name))
                hooks.append(h)

        # 运行校准
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= num_batches:
                    break
                if isinstance(batch, dict):
                    model(**batch)
                else:
                    model(batch)
                print(f"  校准进度: {i+1}/{num_batches}", end='\r')

        # 移除Hook
        for h in hooks:
            h.remove()

        # 计算缩放因子
        for name, ranges in activation_ranges.items():
            self.calibration_data[name] = {
                'activation_scale': ranges['amax'] / FP8Simulator.E4M3_MAX,
                'activation_range': (ranges['min'], ranges['max']),
            }

        print(f"\n校准完成! 共收集 {len(activation_ranges)} 层的激活分布")
        return self.calibration_data

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """将模型中的Linear层替换为FP8Linear"""
        quantized_count = 0
        total_params_original = 0
        total_params_quantized = 0

        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                original_size = module.weight.nelement() * 2  # FP16 = 2B
                fp8_layer = FP8Linear.from_linear(module)
                setattr(model, name, fp8_layer)
                quantized_size = fp8_layer.weight_fp8.nelement() * 1  # FP8 = 1B
                total_params_original += original_size
                total_params_quantized += quantized_size
                quantized_count += 1
            else:
                child_orig, child_quant, child_count = self._quantize_recursive(
                    module
                )
                total_params_original += child_orig
                total_params_quantized += child_quant
                quantized_count += child_count

        compression = total_params_original / max(total_params_quantized, 1)
        print(f"\nFP8量化完成:")
        print(f"  量化层数: {quantized_count}")
        print(f"  原始大小: {total_params_original / 1024**2:.1f} MB")
        print(f"  量化大小: {total_params_quantized / 1024**2:.1f} MB")
        print(f"  压缩比: {compression:.1f}x")

        return model

    def _quantize_recursive(self, module):
        """递归量化子模块"""
        total_orig = 0
        total_quant = 0
        count = 0
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                orig = child.weight.nelement() * 2
                fp8_layer = FP8Linear.from_linear(child)
                setattr(module, name, fp8_layer)
                quant = fp8_layer.weight_fp8.nelement() * 1
                total_orig += orig
                total_quant += quant
                count += 1
            else:
                o, q, c = self._quantize_recursive(child)
                total_orig += o
                total_quant += q
                count += c
        return total_orig, total_quant, count


# ===================== 4. TensorRT-LLM FP8部署模板 =====================

def tensorrt_fp8_deployment_template():
    """TensorRT-LLM FP8量化部署模板代码"""

    code = '''
# === TensorRT-LLM FP8 量化与部署 ===
# 环境: NVIDIA H100 GPU + TensorRT-LLM

# Step 1: 安装依赖
# pip install tensorrt-llm

# Step 2: 模型量化 (使用modelopt)
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_tensorrt_llm_checkpoint

# 加载模型
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# FP8量化配置
quant_config = mtq.FP8_DEFAULT_CFG

# 准备校准数据
def calibrate_loop(model):
    """校准回调函数"""
    calibration_texts = [
        "The capital of France is",
        "Machine learning algorithms can",
        "In the year 2024, technology",
        # ... 128条校准文本
    ]
    for text in calibration_texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        model(**inputs)

# 执行FP8量化
model = mtq.quantize(model, quant_config, forward_loop=calibrate_loop)

# 导出为TensorRT-LLM格式
export_tensorrt_llm_checkpoint(
    model,
    decoder_type="llama",
    dtype=torch.float16,
    export_dir="./llama-8b-fp8",
    inference_tensor_parallel=1,
)

# Step 3: 构建TensorRT引擎
# trtllm-build \\
#     --checkpoint_dir ./llama-8b-fp8 \\
#     --output_dir ./llama-8b-fp8-engine \\
#     --gemm_plugin fp8 \\
#     --max_batch_size 64 \\
#     --max_input_len 2048 \\
#     --max_seq_len 4096

# Step 4: 运行推理
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir("./llama-8b-fp8-engine")
outputs = runner.generate(
    ["What is machine learning?"],
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
'''
    return code


# ===================== 5. vLLM FP8推理模板 =====================

def vllm_fp8_template():
    """vLLM FP8推理模板"""

    code = '''
# === vLLM FP8 推理 ===
# vLLM 0.4.0+ 原生支持FP8量化

from vllm import LLM, SamplingParams

# 方法1: 加载已量化的FP8模型
llm = LLM(
    model="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
    dtype="auto",
    max_model_len=4096,
    gpu_memory_utilization=0.9,
)

# 方法2: 在线FP8量化 (需要H100)
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantization="fp8",  # 自动FP8量化
    max_model_len=4096,
)

# 推理
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)

prompts = [
    "解释什么是量化技术",
    "FP8和INT8有什么区别",
]

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Output: {output.outputs[0].text}")
    print()
'''
    return code


# ===================== 使用示例 =====================

if __name__ == "__main__":
    # 运行FP8精度对比
    FP8Simulator.compare_precision()

    # 量化示例模型
    print("\n" + "=" * 70)
    print("  FP8 模型量化示例")
    print("=" * 70)

    # 创建示例模型
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 256),
    )

    # FP8量化
    quantizer = FP8Quantizer(mode="per_tensor")
    quantized_model = quantizer.quantize_model(model)

    # 精度对比
    test_input = torch.randn(1, 512)
    with torch.no_grad():
        original_model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
        )
        # 注意: 这里仅展示量化流程, 实际需要相同权重
        output_quantized = quantized_model(test_input)
        print(f"\n量化模型输出shape: {output_quantized.shape}")
```

---

## 投机解码

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    投机解码 (Speculative Decoding)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  核心思想: 用小模型快速猜测, 大模型并行验证                              │
│                                                                         │
│  传统自回归解码 (逐token生成, 慢):                                      │
│  ┌────────────────────────────────────────────────────────────┐        │
│  │  大模型                                                    │        │
│  │  Step 1: "The"  → forward → "cat"                         │        │
│  │  Step 2: "The cat" → forward → "sat"                      │        │
│  │  Step 3: "The cat sat" → forward → "on"                   │        │
│  │  Step 4: "The cat sat on" → forward → "the"               │        │
│  │  Step 5: "The cat sat on the" → forward → "mat"           │        │
│  │                                                            │        │
│  │  总计: 5次大模型forward pass (串行, 每次~50ms)            │        │
│  │  总时间: ~250ms                                            │        │
│  └────────────────────────────────────────────────────────────┘        │
│                                                                         │
│  投机解码 (批量验证, 快):                                               │
│  ┌────────────────────────────────────────────────────────────┐        │
│  │  小模型 (Draft Model, 1B参数):                             │        │
│  │  快速生成K个候选token (K=5):                               │        │
│  │  "The" → "cat" → "sat" → "on" → "the" → "mat"           │        │
│  │  (5次小模型forward, 每次~5ms = 25ms)                      │        │
│  │                                                            │        │
│  │  大模型 (Target Model, 70B参数):                           │        │
│  │  一次forward验证所有候选:                                   │        │
│  │  "The [cat] [sat] [on] [the] [mat]" → 并行验证             │        │
│  │  (1次大模型forward = ~50ms)                                │        │
│  │                                                            │        │
│  │  验证结果: ✓cat ✓sat ✓on ✓the ✓mat  → 全部接受!          │        │
│  │  总时间: 25ms + 50ms = 75ms (3.3x加速!)                   │        │
│  └────────────────────────────────────────────────────────────┘        │
│                                                                         │
│  算法流程:                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐          │
│  │ 1. Draft │──►│ 2. 验证  │──►│ 3. 接受/ │──►│ 4. 继续/ │          │
│  │ 小模型   │   │ 大模型   │   │    拒绝   │   │    回退   │          │
│  │ 生成K个  │   │ 并行验证 │   │ 逐个检查 │   │ 从拒绝处  │          │
│  │ 候选token│   │ K个token │   │ 概率比较 │   │ 重新开始  │          │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘          │
│                                                                         │
│  验证准则 (保证输出分布不变):                                           │
│  ┌────────────────────────────────────────────────────────────┐        │
│  │  对每个候选token x_i:                                      │        │
│  │                                                            │        │
│  │  接受概率 = min(1, P_target(x_i) / P_draft(x_i))          │        │
│  │                                                            │        │
│  │  如果 P_target >= P_draft:  100%接受 (大模型也会这么选)    │        │
│  │  如果 P_target < P_draft:   按比例随机接受                 │        │
│  │  如果被拒绝: 从修正分布中重新采样, 丢弃后续候选            │        │
│  │                                                            │        │
│  │  关键性质: 输出分布与只用大模型解码完全一致!               │        │
│  └────────────────────────────────────────────────────────────┘        │
│                                                                         │
│  加速效果 (取决于接受率):                                               │
│  ┌────────────────────────────────────────────────────────────┐        │
│  │  接受率    │ K=5 加速比 │ 适用场景                         │        │
│  ├────────────┼────────────┼──────────────────────────────────┤        │
│  │  >90%      │  3-4x      │ 模板化文本, 代码补全            │        │
│  │  70-90%    │  2-3x      │ 通用对话, 翻译                  │        │
│  │  50-70%    │  1.5-2x    │ 创意写作                        │        │
│  │  <50%      │  <1.5x     │ 高度创造性任务 (可能不值得)     │        │
│  └────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**投机解码（Speculative Decoding）** 是一种无损加速LLM推理的技术。它利用一个小型"草稿模型"快速生成多个候选token，然后让大型"目标模型"一次性并行验证这些候选token。由于Transformer对多个token的并行验证效率远高于逐个生成，因此可以在不改变输出质量的前提下实现2-3倍加速。

**核心优势：**

1. **无损加速**：数学证明输出分布与原始模型完全一致
2. **无需训练**：直接使用现有的大小模型配对
3. **与量化互补**：可以叠加使用（量化+投机=更快）
4. **适用于低并发**：在batch_size=1时加速效果最显著

**常见的模型配对：**

| 目标模型 | 草稿模型 | 接受率 | 预期加速 |
|----------|----------|--------|---------|
| Llama-3.1-70B | Llama-3.1-8B | ~75% | 2-2.5x |
| Llama-3.1-8B | Llama-3.2-1B | ~70% | 2-2.5x |
| GPT-4 | GPT-3.5 | ~80% | 2-3x |
| CodeLlama-34B | CodeLlama-7B | ~85% | 3-4x |

### 代码示例

```python
# 投机解码 - 完整实现
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time


# ===================== 1. 投机解码核心算法 =====================

@dataclass
class SpeculativeConfig:
    """投机解码配置"""
    num_speculative_tokens: int = 5   # 每次猜测的token数 K
    temperature: float = 1.0           # 采样温度
    top_p: float = 0.9                 # Top-p采样


class SpeculativeDecoder:
    """投机解码器

    使用小模型(draft)快速生成候选, 大模型(target)并行验证

    数学保证: 输出分布与直接用目标模型采样完全一致
    """

    def __init__(self, target_model: nn.Module,
                 draft_model: nn.Module,
                 config: SpeculativeConfig = None):
        """
        Args:
            target_model: 大型目标模型
            draft_model: 小型草稿模型
            config: 投机解码配置
        """
        self.target = target_model
        self.draft = draft_model
        self.config = config or SpeculativeConfig()

        # 统计信息
        self.total_draft_tokens = 0
        self.accepted_tokens = 0
        self.total_target_calls = 0
        self.total_draft_calls = 0

    def sample_token(self, logits: torch.Tensor,
                     temperature: float = 1.0,
                     top_p: float = 0.9) -> Tuple[int, float]:
        """从logits中采样一个token

        Returns:
            (token_id, token_probability)
        """
        if temperature == 0:
            # 贪心解码
            token_id = logits.argmax(dim=-1).item()
            prob = F.softmax(logits, dim=-1)[token_id].item()
            return token_id, prob

        # 温度缩放
        scaled_logits = logits / temperature

        # Top-p (nucleus) 采样
        probs = F.softmax(scaled_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 移除累计概率超过top_p的token
        sorted_mask = cumulative_probs - sorted_probs > top_p
        sorted_probs[sorted_mask] = 0
        sorted_probs = sorted_probs / sorted_probs.sum()

        # 采样
        idx = torch.multinomial(sorted_probs, num_samples=1)
        token_id = sorted_indices[idx].item()
        token_prob = probs[token_id].item()

        return token_id, token_prob

    def get_logits(self, model: nn.Module,
                   input_ids: torch.Tensor) -> torch.Tensor:
        """获取模型logits (最后一个token位置)"""
        with torch.no_grad():
            outputs = model(input_ids)
            if hasattr(outputs, 'logits'):
                return outputs.logits[:, -1, :]
            return outputs[:, -1, :]

    def speculative_step(self, input_ids: torch.Tensor
                         ) -> Tuple[List[int], int]:
        """执行一步投机解码

        1. Draft模型生成K个候选token
        2. Target模型一次forward验证所有候选
        3. 按接受/拒绝准则确定接受多少个

        Args:
            input_ids: 当前输入序列 [1, seq_len]

        Returns:
            (accepted_tokens, num_target_forward_calls)
        """
        K = self.config.num_speculative_tokens
        draft_tokens = []
        draft_probs = []

        # === Phase 1: Draft模型快速生成K个候选 ===
        current_ids = input_ids.clone()
        for _ in range(K):
            draft_logits = self.get_logits(self.draft, current_ids)
            token_id, prob = self.sample_token(
                draft_logits.squeeze(0),
                self.config.temperature,
                self.config.top_p
            )
            draft_tokens.append(token_id)
            draft_probs.append(prob)
            current_ids = torch.cat([
                current_ids,
                torch.tensor([[token_id]], device=input_ids.device)
            ], dim=1)
            self.total_draft_calls += 1

        # === Phase 2: Target模型一次性验证 ===
        # 构建包含所有候选的序列
        candidate_ids = torch.cat([
            input_ids,
            torch.tensor([draft_tokens], device=input_ids.device)
        ], dim=1)

        # Target模型一次forward得到所有位置的logits
        with torch.no_grad():
            target_outputs = self.target(candidate_ids)
            if hasattr(target_outputs, 'logits'):
                all_logits = target_outputs.logits
            else:
                all_logits = target_outputs

        self.total_target_calls += 1

        # === Phase 3: 验证与接受/拒绝 ===
        accepted = []
        seq_len = input_ids.shape[1]

        for i in range(K):
            # Target模型在位置 (seq_len - 1 + i) 预测 token i
            target_logits = all_logits[:, seq_len - 1 + i, :]
            target_probs = F.softmax(
                target_logits.squeeze(0) / self.config.temperature,
                dim=-1
            )

            draft_token = draft_tokens[i]
            p_target = target_probs[draft_token].item()
            p_draft = draft_probs[i]

            # 接受准则: accept with probability min(1, p_target/p_draft)
            accept_prob = min(1.0, p_target / max(p_draft, 1e-10))

            if torch.rand(1).item() < accept_prob:
                accepted.append(draft_token)
                self.accepted_tokens += 1
            else:
                # 拒绝: 从修正分布中采样一个token
                # 修正分布: max(0, p_target - p_draft) (归一化)
                correction = target_probs - F.softmax(
                    self.get_logits(self.draft, input_ids).squeeze(0)
                    / self.config.temperature,
                    dim=-1
                ).squeeze(0)
                correction = F.relu(correction)
                if correction.sum() > 0:
                    correction = correction / correction.sum()
                    corrected_token = torch.multinomial(
                        correction, num_samples=1
                    ).item()
                else:
                    corrected_token = target_probs.argmax().item()
                accepted.append(corrected_token)
                break  # 拒绝后停止, 丢弃后续候选

        self.total_draft_tokens += K

        # 如果全部接受, 还可以从target的最后一个位置再采样一个
        if len(accepted) == K:
            bonus_logits = all_logits[:, seq_len - 1 + K, :]
            bonus_token, _ = self.sample_token(
                bonus_logits.squeeze(0),
                self.config.temperature,
                self.config.top_p
            )
            accepted.append(bonus_token)

        return accepted, 1

    def generate(self, input_ids: torch.Tensor,
                 max_new_tokens: int = 128,
                 eos_token_id: int = 2) -> List[int]:
        """使用投机解码生成文本

        Args:
            input_ids: 输入token序列 [1, seq_len]
            max_new_tokens: 最大生成token数
            eos_token_id: 结束token ID

        Returns:
            生成的token列表
        """
        generated = []
        current_ids = input_ids.clone()

        while len(generated) < max_new_tokens:
            # 一步投机解码
            new_tokens, _ = self.speculative_step(current_ids)

            for token in new_tokens:
                if token == eos_token_id:
                    return generated
                generated.append(token)
                if len(generated) >= max_new_tokens:
                    break

            # 更新输入序列
            new_tensor = torch.tensor(
                [new_tokens[:len(generated) - len(generated) + len(new_tokens)]],
                device=input_ids.device
            )
            current_ids = torch.cat([current_ids, new_tensor], dim=1)

        return generated

    def get_stats(self) -> Dict:
        """获取投机解码统计信息"""
        accept_rate = (self.accepted_tokens / max(self.total_draft_tokens, 1)
                       * 100)
        tokens_per_target_call = (
            (self.accepted_tokens + self.total_target_calls)
            / max(self.total_target_calls, 1)
        )

        return {
            "总draft token数": self.total_draft_tokens,
            "接受token数": self.accepted_tokens,
            "接受率": f"{accept_rate:.1f}%",
            "Target forward次数": self.total_target_calls,
            "Draft forward次数": self.total_draft_calls,
            "每次Target生成token数": f"{tokens_per_target_call:.1f}",
            "理论加速比": f"{tokens_per_target_call:.1f}x",
        }


# ===================== 2. 投机解码模拟器 =====================

class SpeculativeSimulator:
    """投机解码性能模拟器

    模拟不同接受率和K值下的加速效果
    """

    @staticmethod
    def simulate_speedup(accept_rate: float, K: int,
                         target_latency_ms: float = 50,
                         draft_latency_ms: float = 5) -> Dict:
        """模拟投机解码的加速比

        Args:
            accept_rate: 平均接受率 (0-1)
            K: 每次猜测的token数
            target_latency_ms: 大模型单次forward延迟
            draft_latency_ms: 小模型单次forward延迟

        Returns:
            包含加速比等指标的字典
        """
        # 传统自回归: 每个token需要一次target forward
        baseline_per_token = target_latency_ms

        # 投机解码: K次draft + 1次target
        # 平均接受的token数 = sum(accept_rate^i, i=0..K-1) + 1(bonus if all accepted)
        expected_accepted = sum(accept_rate ** i for i in range(K))
        if accept_rate ** K > 0.5:  # 全部接受时的bonus token
            expected_accepted += accept_rate ** K

        speculative_time = K * draft_latency_ms + target_latency_ms
        speculative_per_token = speculative_time / expected_accepted

        speedup = baseline_per_token / speculative_per_token

        return {
            "接受率": f"{accept_rate*100:.0f}%",
            "K值": K,
            "期望接受数": f"{expected_accepted:.2f}",
            "传统延迟(ms/token)": f"{baseline_per_token:.1f}",
            "投机延迟(ms/token)": f"{speculative_per_token:.1f}",
            "加速比": f"{speedup:.2f}x",
        }

    @staticmethod
    def print_speedup_table():
        """打印不同参数组合的加速比表格"""
        print("=" * 75)
        print("  投机解码加速比模拟 (Target=50ms, Draft=5ms)")
        print("=" * 75)

        print(f"\n{'接受率':<10} ", end="")
        for K in [3, 5, 7, 10]:
            print(f"{'K='+str(K):<12}", end="")
        print()
        print("-" * 60)

        for rate in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            print(f"{rate*100:>5.0f}%     ", end="")
            for K in [3, 5, 7, 10]:
                result = SpeculativeSimulator.simulate_speedup(rate, K)
                print(f"{result['加速比']:<12}", end="")
            print()

        print("\n注: 实际加速比还受GPU利用率、内存带宽等因素影响")


# ===================== 3. vLLM投机解码使用模板 =====================

def vllm_speculative_template():
    """vLLM投机解码配置模板"""

    code = '''
# === vLLM 投机解码配置 ===

from vllm import LLM, SamplingParams

# 方法1: 使用独立的draft模型
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="meta-llama/Llama-3.2-1B-Instruct",
    num_speculative_tokens=5,       # 每次猜测5个token
    max_model_len=4096,
    tensor_parallel_size=4,         # 4卡并行
    gpu_memory_utilization=0.9,
)

# 方法2: 使用ngram匹配 (无需额外模型)
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="[ngram]",    # 基于n-gram匹配的投机
    num_speculative_tokens=5,
    ngram_prompt_lookup_max=4,      # 最大n-gram窗口
    max_model_len=4096,
    tensor_parallel_size=4,
)

# 方法3: MLP投机 (Medusa风格)
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="[mlp_speculator]",
    num_speculative_tokens=3,
    max_model_len=4096,
    tensor_parallel_size=4,
)

# 推理
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

outputs = llm.generate(
    ["请详细解释投机解码的工作原理"],
    sampling_params,
)

for output in outputs:
    print(output.outputs[0].text)
'''
    return code


# ===================== 使用示例 =====================

if __name__ == "__main__":
    # 打印加速比模拟表格
    SpeculativeSimulator.print_speedup_table()

    # 模拟具体场景
    print("\n" + "=" * 60)
    print("  具体场景模拟")
    print("=" * 60)

    scenarios = [
        ("代码补全 (高接受率)", 0.90, 5),
        ("通用对话", 0.75, 5),
        ("创意写作 (低接受率)", 0.55, 5),
        ("翻译任务", 0.80, 7),
    ]

    for name, rate, k in scenarios:
        result = SpeculativeSimulator.simulate_speedup(rate, k)
        print(f"\n场景: {name}")
        for key, value in result.items():
            print(f"  {key}: {value}")
```

---

## 生产部署检查清单

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LLM 生产部署检查清单                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  部署流程:                                                              │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐        │
│  │ 模型 │─►│ 量化 │─►│ 基准 │─►│ 服务 │─►│ 监控 │─►│ 运维 │        │
│  │ 选型 │  │ 优化 │  │ 测试 │  │ 部署 │  │ 告警 │  │ 迭代 │        │
│  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘        │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  阶段1: 模型选型与优化                                          │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │  □ 确定任务需求 (对话/代码/翻译/RAG)                    │   │   │
│  │  │  □ 选择基础模型 (参数量 vs 精度 vs 成本)                │   │   │
│  │  │  □ 确定量化方案 (FP16/INT8/INT4/FP8)                   │   │   │
│  │  │  □ 在目标任务上评估量化精度损失                          │   │   │
│  │  │  □ 确定是否需要微调                                     │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  │                                                                 │   │
│  │  阶段2: 性能基准测试                                            │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │  □ TTFT (首Token延迟) < 目标值                          │   │   │
│  │  │  □ TPS (生成速度) > 目标值                              │   │   │
│  │  │  □ 并发吞吐量满足峰值需求                               │   │   │
│  │  │  □ 显存使用在安全范围内 (<85%)                          │   │   │
│  │  │  □ 长文本场景 (4K/8K/32K) 性能测试                     │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  │                                                                 │   │
│  │  阶段3: 服务化部署                                              │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │  □ 选择推理引擎 (vLLM/TensorRT-LLM/TGI)               │   │   │
│  │  │  □ 配置API服务 (OpenAI兼容)                             │   │   │
│  │  │  □ 设置负载均衡和自动扩缩容                             │   │   │
│  │  │  □ 实现请求队列和限流                                   │   │   │
│  │  │  □ 配置健康检查和就绪探针                               │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  │                                                                 │   │
│  │  阶段4: 安全与合规                                              │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │  □ 输入内容过滤 (注入攻击/敏感内容)                     │   │   │
│  │  │  □ 输出内容审核 (有害内容/幻觉检测)                     │   │   │
│  │  │  □ 速率限制 (每用户/每IP)                               │   │   │
│  │  │  □ 认证授权 (API Key/OAuth)                              │   │   │
│  │  │  □ 数据隐私 (日志脱敏/PII过滤)                          │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  │                                                                 │   │
│  │  阶段5: 监控与运维                                              │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │  □ 延迟监控 (P50/P95/P99)                              │   │   │
│  │  │  □ 吞吐量监控 (QPS/TPS)                                │   │   │
│  │  │  □ GPU利用率和显存监控                                  │   │   │
│  │  │  □ 错误率和超时监控                                     │   │   │
│  │  │  □ 成本监控 (每请求/每token)                            │   │   │
│  │  │  □ 模型质量监控 (用户反馈/自动评估)                     │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 详细说明

将LLM从实验环境部署到生产环境需要考虑性能、稳定性、安全性和成本等多个维度。以下是一份完整的生产部署检查清单，涵盖从模型选型到持续运维的全流程。

**关键性能指标(KPI)：**

| 指标 | 说明 | 对话场景目标 | 批处理目标 |
|------|------|-------------|-----------|
| TTFT | 首Token延迟 | <500ms | <2s |
| TPS | 每秒生成token数 | >30 tps | >1000 tps(批) |
| QPS | 每秒请求数 | >10 | >100 |
| P99延迟 | 99%请求延迟 | <3s | <30s |
| 可用性 | 服务在线率 | >99.9% | >99.5% |
| GPU利用率 | GPU计算利用 | >70% | >85% |

### 代码示例

```python
# 生产部署检查清单 - 自动化验证工具
import time
import json
import statistics
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum


# ===================== 1. 检查项定义 =====================

class CheckStatus(Enum):
    PASS = "通过"
    WARN = "警告"
    FAIL = "失败"
    SKIP = "跳过"


@dataclass
class CheckResult:
    """单项检查结果"""
    name: str
    category: str
    status: CheckStatus
    message: str
    value: Optional[str] = None
    threshold: Optional[str] = None


@dataclass
class DeploymentReport:
    """部署检查报告"""
    model_name: str
    engine: str
    timestamp: str = ""
    results: List[CheckResult] = field(default_factory=list)

    def add_result(self, result: CheckResult):
        self.results.append(result)

    def summary(self) -> Dict:
        """统计各状态数量"""
        counts = {s: 0 for s in CheckStatus}
        for r in self.results:
            counts[r.status] += 1
        return {s.value: c for s, c in counts.items()}

    def is_ready(self) -> bool:
        """是否满足上线条件 (无FAIL)"""
        return all(r.status != CheckStatus.FAIL for r in self.results)

    def print_report(self):
        """打印检查报告"""
        print("=" * 75)
        print(f"  LLM 生产部署检查报告")
        print(f"  模型: {self.model_name}")
        print(f"  引擎: {self.engine}")
        print(f"  时间: {self.timestamp}")
        print("=" * 75)

        # 按类别分组
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = []
            categories[r.category].append(r)

        for category, checks in categories.items():
            print(f"\n--- {category} ---")
            for check in checks:
                icon = {
                    CheckStatus.PASS: "[OK]",
                    CheckStatus.WARN: "[!!]",
                    CheckStatus.FAIL: "[XX]",
                    CheckStatus.SKIP: "[--]",
                }[check.status]

                print(f"  {icon} {check.name}")
                if check.value:
                    print(f"       当前值: {check.value}", end="")
                    if check.threshold:
                        print(f"  (阈值: {check.threshold})", end="")
                    print()
                if check.status in (CheckStatus.WARN, CheckStatus.FAIL):
                    print(f"       {check.message}")

        # 总结
        summary = self.summary()
        print(f"\n{'='*75}")
        print(f"  总结: ", end="")
        for status, count in summary.items():
            if count > 0:
                print(f"{status}={count}  ", end="")
        print()

        if self.is_ready():
            print("  结论: 满足上线条件")
        else:
            print("  结论: 存在未通过项, 请修复后重新检查")
        print("=" * 75)


# ===================== 2. 性能检查器 =====================

class PerformanceChecker:
    """性能基准检查"""

    def __init__(self, api_url: str = "http://localhost:8000/v1"):
        self.api_url = api_url

    def check_ttft(self, report: DeploymentReport,
                   threshold_ms: float = 500,
                   num_requests: int = 20):
        """检查首Token延迟 (TTFT)"""
        # 模拟TTFT测量 (实际部署时替换为真实API调用)
        ttft_samples = []

        for i in range(num_requests):
            # 模拟: 实际应使用streaming API测量首token时间
            start = time.time()
            # response = requests.post(api_url + "/chat/completions", ...)
            time.sleep(0.05 + 0.02 * (i % 5))  # 模拟延迟
            ttft = (time.time() - start) * 1000
            ttft_samples.append(ttft)

        p50 = statistics.median(ttft_samples)
        p95 = sorted(ttft_samples)[int(len(ttft_samples) * 0.95)]
        p99 = sorted(ttft_samples)[int(len(ttft_samples) * 0.99)]

        status = CheckStatus.PASS
        msg = "TTFT在目标范围内"
        if p95 > threshold_ms:
            status = CheckStatus.WARN
            msg = f"P95 TTFT ({p95:.0f}ms) 超过阈值"
        if p99 > threshold_ms * 2:
            status = CheckStatus.FAIL
            msg = f"P99 TTFT ({p99:.0f}ms) 严重超标"

        report.add_result(CheckResult(
            name="首Token延迟 (TTFT)",
            category="性能指标",
            status=status,
            message=msg,
            value=f"P50={p50:.0f}ms P95={p95:.0f}ms P99={p99:.0f}ms",
            threshold=f"P95 < {threshold_ms}ms"
        ))

    def check_throughput(self, report: DeploymentReport,
                         threshold_tps: float = 30,
                         num_requests: int = 10):
        """检查生成吞吐量 (TPS)"""
        tps_samples = []

        for _ in range(num_requests):
            tokens_generated = 100  # 模拟生成100 tokens
            start = time.time()
            time.sleep(0.1 + 0.05 * (len(tps_samples) % 3))  # 模拟
            elapsed = time.time() - start
            tps = tokens_generated / elapsed
            tps_samples.append(tps)

        avg_tps = statistics.mean(tps_samples)
        min_tps = min(tps_samples)

        status = CheckStatus.PASS if avg_tps >= threshold_tps else (
            CheckStatus.WARN if avg_tps >= threshold_tps * 0.8
            else CheckStatus.FAIL
        )

        report.add_result(CheckResult(
            name="生成吞吐量 (TPS)",
            category="性能指标",
            status=status,
            message=f"平均TPS {'满足' if status == CheckStatus.PASS else '未达到'}目标",
            value=f"平均={avg_tps:.1f} 最低={min_tps:.1f} tps",
            threshold=f"> {threshold_tps} tps"
        ))

    def check_concurrent(self, report: DeploymentReport,
                         target_qps: float = 10,
                         num_concurrent: int = 20):
        """检查并发处理能力"""
        # 模拟并发请求
        start = time.time()
        time.sleep(num_concurrent * 0.01)  # 模拟并发处理
        elapsed = time.time() - start
        actual_qps = num_concurrent / elapsed

        status = CheckStatus.PASS if actual_qps >= target_qps else (
            CheckStatus.WARN if actual_qps >= target_qps * 0.7
            else CheckStatus.FAIL
        )

        report.add_result(CheckResult(
            name="并发吞吐量 (QPS)",
            category="性能指标",
            status=status,
            message=f"并发QPS {'满足' if status == CheckStatus.PASS else '低于'}目标",
            value=f"{actual_qps:.1f} QPS ({num_concurrent}并发)",
            threshold=f"> {target_qps} QPS"
        ))


# ===================== 3. 资源检查器 =====================

class ResourceChecker:
    """系统资源检查"""

    @staticmethod
    def check_gpu_memory(report: DeploymentReport,
                         threshold_pct: float = 85):
        """检查GPU显存使用"""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_mem / 1024**3
                pct = allocated / total * 100
            else:
                allocated, total, pct = 0, 0, 0
        except ImportError:
            allocated, total, pct = 8.5, 24.0, 35.4  # 模拟值

        status = CheckStatus.PASS
        msg = "显存使用在安全范围"
        if pct > threshold_pct:
            status = CheckStatus.WARN
            msg = f"显存使用率 {pct:.1f}% 偏高, 可能导致OOM"
        if pct > 95:
            status = CheckStatus.FAIL
            msg = "显存即将耗尽, 有OOM风险"

        report.add_result(CheckResult(
            name="GPU显存使用",
            category="系统资源",
            status=status,
            message=msg,
            value=f"{allocated:.1f}GB / {total:.1f}GB ({pct:.1f}%)",
            threshold=f"< {threshold_pct}%"
        ))

    @staticmethod
    def check_disk_space(report: DeploymentReport,
                         model_dir: str = "./model",
                         min_free_gb: float = 50):
        """检查磁盘空间"""
        # 模拟磁盘检查
        total_gb = 500.0
        free_gb = 120.0

        status = CheckStatus.PASS if free_gb >= min_free_gb else (
            CheckStatus.WARN if free_gb >= min_free_gb * 0.5
            else CheckStatus.FAIL
        )

        report.add_result(CheckResult(
            name="磁盘空间",
            category="系统资源",
            status=status,
            message=f"剩余空间 {'充足' if status == CheckStatus.PASS else '不足'}",
            value=f"剩余 {free_gb:.0f}GB / 总计 {total_gb:.0f}GB",
            threshold=f"剩余 > {min_free_gb}GB"
        ))


# ===================== 4. 安全检查器 =====================

class SecurityChecker:
    """安全合规检查"""

    @staticmethod
    def check_rate_limiting(report: DeploymentReport,
                            api_url: str = "http://localhost:8000"):
        """检查速率限制"""
        # 模拟: 检查API是否配置了速率限制
        has_rate_limit = True  # 实际应检查nginx/API网关配置

        status = CheckStatus.PASS if has_rate_limit else CheckStatus.FAIL
        report.add_result(CheckResult(
            name="API速率限制",
            category="安全合规",
            status=status,
            message="已配置速率限制" if has_rate_limit else "未配置速率限制, 有滥用风险",
            value="100 req/min/user" if has_rate_limit else "未设置",
        ))

    @staticmethod
    def check_auth(report: DeploymentReport):
        """检查认证授权"""
        has_auth = True  # 实际应检查API是否需要认证

        status = CheckStatus.PASS if has_auth else CheckStatus.FAIL
        report.add_result(CheckResult(
            name="认证授权",
            category="安全合规",
            status=status,
            message="已配置API认证" if has_auth else "API未设置认证, 任何人可访问",
            value="API Key认证" if has_auth else "无认证",
        ))

    @staticmethod
    def check_content_filter(report: DeploymentReport):
        """检查内容过滤"""
        has_input_filter = True
        has_output_filter = True

        if has_input_filter and has_output_filter:
            status = CheckStatus.PASS
            msg = "输入输出过滤均已配置"
        elif has_input_filter or has_output_filter:
            status = CheckStatus.WARN
            msg = "部分过滤未配置"
        else:
            status = CheckStatus.FAIL
            msg = "未配置任何内容过滤"

        report.add_result(CheckResult(
            name="内容安全过滤",
            category="安全合规",
            status=status,
            message=msg,
            value=f"输入过滤={'是' if has_input_filter else '否'}, "
                  f"输出过滤={'是' if has_output_filter else '否'}",
        ))

    @staticmethod
    def check_log_privacy(report: DeploymentReport):
        """检查日志隐私"""
        has_pii_filter = True

        status = CheckStatus.PASS if has_pii_filter else CheckStatus.WARN
        report.add_result(CheckResult(
            name="日志隐私保护",
            category="安全合规",
            status=status,
            message="PII数据已脱敏" if has_pii_filter else "日志中可能包含用户隐私数据",
            value="已配置PII过滤" if has_pii_filter else "未配置",
        ))


# ===================== 5. 监控检查器 =====================

class MonitoringChecker:
    """监控告警检查"""

    @staticmethod
    def check_metrics_collection(report: DeploymentReport):
        """检查指标收集"""
        metrics = {
            "延迟监控 (TTFT/TPS)": True,
            "GPU利用率监控": True,
            "错误率监控": True,
            "QPS监控": True,
            "成本监控": False,
        }

        configured = sum(1 for v in metrics.values() if v)
        total = len(metrics)

        if configured == total:
            status = CheckStatus.PASS
        elif configured >= total * 0.7:
            status = CheckStatus.WARN
        else:
            status = CheckStatus.FAIL

        missing = [k for k, v in metrics.items() if not v]

        report.add_result(CheckResult(
            name="监控指标收集",
            category="监控告警",
            status=status,
            message=f"已配置 {configured}/{total} 项监控" +
                    (f", 缺少: {', '.join(missing)}" if missing else ""),
            value=f"{configured}/{total} 项",
        ))

    @staticmethod
    def check_alerting(report: DeploymentReport):
        """检查告警配置"""
        alerts = {
            "GPU OOM告警": True,
            "高延迟告警 (P99>3s)": True,
            "高错误率告警 (>1%)": True,
            "服务不可用告警": True,
        }

        configured = sum(1 for v in alerts.values() if v)
        total = len(alerts)

        status = CheckStatus.PASS if configured == total else (
            CheckStatus.WARN if configured >= 2 else CheckStatus.FAIL
        )

        report.add_result(CheckResult(
            name="告警规则",
            category="监控告警",
            status=status,
            message=f"已配置 {configured}/{total} 条告警规则",
            value=f"{configured}/{total} 条",
        ))

    @staticmethod
    def check_health_endpoint(report: DeploymentReport,
                              api_url: str = "http://localhost:8000"):
        """检查健康检查端点"""
        has_health = True  # 实际应请求 /health 端点

        report.add_result(CheckResult(
            name="健康检查端点",
            category="监控告警",
            status=CheckStatus.PASS if has_health else CheckStatus.FAIL,
            message="健康检查端点可用" if has_health else "未配置健康检查",
            value=f"{api_url}/health" if has_health else "未配置",
        ))


# ===================== 6. 完整部署检查 =====================

def run_deployment_check(model_name: str = "Llama-3.1-8B-Instruct",
                          engine: str = "vLLM",
                          api_url: str = "http://localhost:8000"
                          ) -> DeploymentReport:
    """运行完整的部署前检查"""

    report = DeploymentReport(
        model_name=model_name,
        engine=engine,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    print("开始生产部署检查...\n")

    # 性能检查
    print("  [1/5] 性能基准测试...")
    perf = PerformanceChecker(api_url)
    perf.check_ttft(report)
    perf.check_throughput(report)
    perf.check_concurrent(report)

    # 资源检查
    print("  [2/5] 系统资源检查...")
    ResourceChecker.check_gpu_memory(report)
    ResourceChecker.check_disk_space(report)

    # 安全检查
    print("  [3/5] 安全合规检查...")
    SecurityChecker.check_rate_limiting(report)
    SecurityChecker.check_auth(report)
    SecurityChecker.check_content_filter(report)
    SecurityChecker.check_log_privacy(report)

    # 监控检查
    print("  [4/5] 监控告警检查...")
    MonitoringChecker.check_metrics_collection(report)
    MonitoringChecker.check_alerting(report)
    MonitoringChecker.check_health_endpoint(report)

    print("  [5/5] 生成报告...\n")

    # 打印报告
    report.print_report()

    return report


# ===================== 运行示例 =====================

if __name__ == "__main__":
    report = run_deployment_check(
        model_name="Llama-3.1-8B-Instruct-AWQ",
        engine="vLLM 0.6.0",
        api_url="http://localhost:8000"
    )

    # 导出报告为JSON
    print("\n导出检查报告...")
    report_data = {
        "model": report.model_name,
        "engine": report.engine,
        "timestamp": report.timestamp,
        "ready": report.is_ready(),
        "summary": report.summary(),
        "checks": [
            {
                "name": r.name,
                "category": r.category,
                "status": r.status.value,
                "message": r.message,
                "value": r.value,
                "threshold": r.threshold,
            }
            for r in report.results
        ]
    }
    print(json.dumps(report_data, ensure_ascii=False, indent=2))
```

---

## 总结

本教程涵盖了模型优化技术的核心内容:

1. **优化概述**: 量化、蒸馏、剪枝和推理引擎优化构成了完整的优化技术栈。根据硬件条件和性能需求选择合适的优化方案。

2. **模型量化**: GPTQ、AWQ和GGUF是三种主流量化格式。INT4量化可将模型体积减少4倍，精度损失通常在1-3%以内。AWQ在速度和精度上均优于GPTQ。

3. **llama.cpp本地推理**: 基于GGUF格式的轻量推理引擎，支持CPU/GPU/Apple Silicon等多种硬件。通过Ollama可以一键部署，通过llama-server提供OpenAI兼容API。

4. **知识蒸馏**: 让小模型学习大模型的知识，包括Logit蒸馏、特征蒸馏和数据蒸馏三种方式。数据蒸馏（用大模型生成训练数据）是LLM领域最常用的方法。

5. **模型剪枝**: 通过移除不重要的参数或结构减少计算量，包括非结构化剪枝（权重级别）和结构化剪枝（通道/层级别）。LLM专用方法如SparseGPT和Wanda可实现一次性剪枝。

6. **推理加速引擎**: vLLM（PagedAttention）、TensorRT-LLM（图优化）和TGI（HF生态）是三大主流引擎。vLLM适合通用场景，TensorRT-LLM适合极致性能需求。

7. **性能对比**: 完整的基准测试框架覆盖吞吐量、延迟、显存和精度四个维度，帮助做出数据驱动的技术选型决策。

8. **FP8量化与下一代精度**: FP8是H100/H200原生支持的8-bit浮点格式，分为E4M3（权重/前向）和E5M2（梯度/反向）两种格式。相比INT8，FP8保留浮点动态范围，精度损失更小（<0.5%），且支持训练。提供了FP8模拟器、量化器和TensorRT-LLM/vLLM部署模板。

9. **投机解码**: 使用小型草稿模型快速生成K个候选token，大型目标模型一次性并行验证。数学保证输出分布不变（无损加速），在代码补全等高接受率场景可达3-4倍加速。实现了完整的投机解码算法和性能模拟器。

10. **生产部署检查清单**: 覆盖模型选型、性能基准、服务部署、安全合规和监控运维五大阶段。提供了自动化检查工具，涵盖TTFT/TPS/QPS性能检查、GPU显存检查、安全过滤检查、监控告警检查等，并生成可导出的检查报告。

## 最佳实践

1. **先量化再选引擎**: AWQ INT4 + vLLM 是性价比最高的组合
2. **分级部署**: 高并发用vLLM集群，低延迟用TensorRT-LLM，本地用llama.cpp
3. **精度验证**: 量化后务必在目标任务上评估精度损失
4. **Q4_K_M是GGUF的最佳平衡点**: 体积适中、精度好、速度快
5. **Continuous Batching**: 使用vLLM/TGI等支持动态批处理的引擎
6. **KV Cache管理**: 合理设置max_model_len避免OOM
7. **多GPU时使用Tensor Parallel**: 将模型分布到多个GPU上
8. **监控性能指标**: 持续监控TTFT、吞吐量和显存使用
9. **H100优先用FP8**: 在H100/H200上优先选择FP8量化，精度损失比INT8更小且速度更快
10. **低并发场景用投机解码**: batch_size=1时投机解码加速最显著，高并发场景收益递减
11. **投机解码选择合适的K值**: K=5是通用推荐，代码等高接受率场景可增大到7-10
12. **部署前跑完检查清单**: 性能/安全/监控三项全部通过才可上线

## 参考资源

- [vLLM 官方文档](https://docs.vllm.ai/)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [Ollama](https://ollama.com/)
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference)
- [GPTQ 论文](https://arxiv.org/abs/2210.17323)
- [AWQ 论文](https://arxiv.org/abs/2306.00978)
- [SparseGPT 论文](https://arxiv.org/abs/2301.00774)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) - Micikevicius et al., 2022
- [Speculative Decoding 论文](https://arxiv.org/abs/2211.17192) - Leviathan et al., 2022
- [SpecInfer 论文](https://arxiv.org/abs/2305.09781) - Miao et al., 2023
- [NVIDIA modelopt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) - FP8量化工具

---

**文件大小目标**: 30-35KB
**创建时间**: 2024-01-15
**最后更新**: 2024-01-15
