# 模型微调完整教程

## 目录
1. [微调概述](#微调概述)
2. [LoRA原理与实现](#lora原理与实现)
3. [QLoRA量化微调](#qlora量化微调)
4. [PEFT框架使用](#peft框架使用)
5. [Hugging Face Trainer](#hugging-face-trainer)
6. [完整实战：微调Llama模型](#完整实战微调llama模型)
7. [DPO与RLHF对齐](#dpo与rlhf对齐)
8. [多GPU与分布式微调](#多gpu与分布式微调)

---

## 微调概述

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    模型微调方法全景图                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  预训练模型 (Pre-trained Model)                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  参数量: 7B ~ 405B                                               │  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐     ┌────────┐               │  │
│  │  │Layer 1 │ │Layer 2 │ │Layer 3 │ ... │Layer N │               │  │
│  │  │        │ │        │ │        │     │        │               │  │
│  │  │ Q K V  │ │ Q K V  │ │ Q K V  │     │ Q K V  │               │  │
│  │  │ FFN    │ │ FFN    │ │ FFN    │     │ FFN    │               │  │
│  │  └────────┘ └────────┘ └────────┘     └────────┘               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│       │              │              │              │                    │
│       ▼              ▼              ▼              ▼                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    微调方法 (Fine-Tuning Methods)                 │  │
│  │                                                                  │  │
│  │  方法1: Full Fine-Tuning (全量微调)                              │  │
│  │  ┌──────────────────────────────────────┐                       │  │
│  │  │ 更新所有参数 (100%)                   │  显存需求: ~4x模型大小│  │
│  │  │ ████████████████████████████████████ │  7B模型需要~120GB     │  │
│  │  └──────────────────────────────────────┘                       │  │
│  │                                                                  │  │
│  │  方法2: LoRA (Low-Rank Adaptation)                               │  │
│  │  ┌──────────────────────────────────────┐                       │  │
│  │  │ 冻结原始参数, 添加低秩适配器 (~0.1%) │  显存需求: ~1.2x模型  │  │
│  │  │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░██░░ │  7B模型需要~16GB      │  │
│  │  └──────────────────────────────────────┘                       │  │
│  │                                                                  │  │
│  │  方法3: QLoRA (Quantized LoRA)                                   │  │
│  │  ┌──────────────────────────────────────┐                       │  │
│  │  │ 4-bit量化 + LoRA适配器              │  显存需求: ~0.3x模型  │  │
│  │  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██▒▒ │  7B模型需要~6GB       │  │
│  │  └──────────────────────────────────────┘                       │  │
│  │                                                                  │  │
│  │  方法4: Adapter Tuning                                           │  │
│  │  ┌──────────────────────────────────────┐                       │  │
│  │  │ 在层间插入小型适配器模块 (~1-5%)     │  显存需求: ~1.5x模型  │  │
│  │  │ ░░░░░░██░░░░░░██░░░░░░██░░░░░░██░░ │                       │  │
│  │  └──────────────────────────────────────┘                       │  │
│  │                                                                  │  │
│  │  方法5: Prefix Tuning / Prompt Tuning                            │  │
│  │  ┌──────────────────────────────────────┐                       │  │
│  │  │ 只训练前缀/提示向量 (~0.01%)         │  显存需求: 极低       │  │
│  │  │ ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │                       │  │
│  │  └──────────────────────────────────────┘                       │  │
│  │                                                                  │  │
│  │  ██ = 可训练参数    ░░ = 冻结参数    ▒▒ = 量化冻结参数          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  微调数据流:                                                            │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐           │
│  │ 原始数据 │──►│ 数据格式化│──►│ Tokenize │──►│ 微调训练 │           │
│  │ JSON/CSV │   │ Instruct │   │ 分词处理 │   │ LoRA/QLoRA│           │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘           │
│       │                                              │                 │
│       ▼                                              ▼                 │
│  数据格式:                                     ┌──────────┐           │
│  {"instruction": "...",                        │ 合并权重 │           │
│   "input": "...",                              │ 导出模型 │           │
│   "output": "..."}                             └──────────┘           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**模型微调（Fine-Tuning）** 是将预训练大语言模型适配到特定任务或领域的过程。预训练模型在海量通用数据上学习了语言理解能力，但可能不擅长特定领域（如医疗、法律、代码）的任务。通过在少量领域数据上微调，可以大幅提升模型在目标任务上的表现。

**为什么需要微调？**

1. **领域适配**：让通用模型理解行业术语和知识
2. **任务优化**：提升特定任务（分类、摘要、翻译）的准确率
3. **风格控制**：让模型生成符合特定风格的输出
4. **成本效益**：比从头训练便宜数千倍
5. **数据隐私**：在私有数据上微调，无需上传到第三方

**微调方法对比：**

| 方法 | 可训练参数比例 | 7B模型显存 | 训练速度 | 效果 | 适用场景 |
|------|---------------|-----------|----------|------|----------|
| **Full Fine-Tuning** | 100% | ~120GB | 慢 | 最好 | 有充足GPU资源 |
| **LoRA** | ~0.1-1% | ~16GB | 快 | 接近Full FT | 单GPU微调 |
| **QLoRA** | ~0.1-1% | ~6GB | 较快 | 接近LoRA | 消费级GPU |
| **Adapter** | ~1-5% | ~20GB | 中等 | 好 | 多任务切换 |
| **Prefix Tuning** | ~0.01% | ~10GB | 最快 | 一般 | 轻量级适配 |
| **Prompt Tuning** | ~0.001% | ~10GB | 最快 | 较弱 | 极简场景 |

**微调数据格式 -- Alpaca格式（最常用）：**

```json
[
  {
    "instruction": "将以下英文翻译成中文",
    "input": "Hello, how are you?",
    "output": "你好，你好吗？"
  },
  {
    "instruction": "写一首关于春天的五言绝句",
    "input": "",
    "output": "春风吹绿柳，细雨润红花。燕子归来日，人间处处家。"
  }
]
```

**微调数据格式 -- ChatML格式（对话模型）：**

```json
{
  "messages": [
    {"role": "system", "content": "你是一个专业的医疗助手。"},
    {"role": "user", "content": "头疼怎么办？"},
    {"role": "assistant", "content": "头疼有多种可能的原因..."}
  ]
}
```

### 代码示例

```python
# 微调概述 - 环境准备与数据处理
import json
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


# ===================== 1. 微调配置 =====================

@dataclass
class FineTuningConfig:
    """微调配置"""
    # 模型
    model_name: str = "meta-llama/Llama-3.1-8B"
    # 数据
    train_data: str = "train.json"
    val_data: str = "val.json"
    max_seq_length: int = 2048
    # LoRA
    lora_r: int = 16               # LoRA秩
    lora_alpha: int = 32           # LoRA缩放因子
    lora_dropout: float = 0.05     # LoRA Dropout
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )
    # 训练
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3
    # 量化 (QLoRA)
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True
    # 输出
    output_dir: str = "./output"


# ===================== 2. 数据处理 =====================

class FineTuningDataProcessor:
    """微调数据处理器"""

    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_data(self, file_path: str) -> List[Dict]:
        """加载JSON数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"加载了 {len(data)} 条数据")
        return data

    def format_alpaca(self, sample: Dict) -> str:
        """Alpaca格式转为训练文本"""
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output = sample.get("output", "")

        if input_text:
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
        else:
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output}"
            )
        return prompt

    def format_chatml(self, sample: Dict) -> str:
        """ChatML格式转为训练文本"""
        messages = sample.get("messages", [])
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return text

    def tokenize(self, text: str) -> Dict:
        """分词处理"""
        result = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        result["labels"] = result["input_ids"].clone()
        return result

    def prepare_dataset(self, data: List[Dict],
                        format_type: str = "alpaca") -> List[Dict]:
        """准备数据集"""
        formatter = (self.format_alpaca if format_type == "alpaca"
                     else self.format_chatml)

        processed = []
        for sample in data:
            text = formatter(sample)
            tokenized = self.tokenize(text)
            processed.append(tokenized)

        print(f"处理完成: {len(processed)} 条样本")
        return processed


# ===================== 3. 显存估算 =====================

def estimate_memory_usage(model_params_b: float,
                          method: str = "qlora") -> Dict[str, float]:
    """估算微调显存需求

    Args:
        model_params_b: 模型参数量（十亿）
        method: 微调方法 (full/lora/qlora)

    Returns:
        各项显存开销（GB）
    """
    # 模型参数大小 (FP16 = 2 bytes per param)
    model_size_fp16 = model_params_b * 2  # GB

    if method == "full":
        # Full FT: 模型(FP16) + 梯度(FP16) + 优化器状态(FP32x2) + 激活
        model_mem = model_size_fp16
        grad_mem = model_size_fp16
        optimizer_mem = model_params_b * 4 * 2  # Adam: m + v (FP32)
        activation_mem = model_size_fp16 * 0.5  # 估算

    elif method == "lora":
        # LoRA: 模型(FP16) + LoRA参数 + 梯度 + 优化器
        model_mem = model_size_fp16
        lora_ratio = 0.01  # ~1% 可训练参数
        lora_params = model_params_b * lora_ratio
        grad_mem = lora_params * 2
        optimizer_mem = lora_params * 4 * 2
        activation_mem = model_size_fp16 * 0.2

    elif method == "qlora":
        # QLoRA: 模型(4-bit) + LoRA参数(FP16) + 梯度 + 优化器
        model_mem = model_params_b * 0.5  # 4-bit = 0.5 bytes
        lora_ratio = 0.01
        lora_params = model_params_b * lora_ratio
        grad_mem = lora_params * 2
        optimizer_mem = lora_params * 4 * 2
        activation_mem = model_params_b * 0.3

    else:
        raise ValueError(f"未知方法: {method}")

    total = model_mem + grad_mem + optimizer_mem + activation_mem

    result = {
        "模型参数": f"{model_mem:.1f} GB",
        "梯度": f"{grad_mem:.1f} GB",
        "优化器状态": f"{optimizer_mem:.1f} GB",
        "激活值": f"{activation_mem:.1f} GB",
        "总计": f"{total:.1f} GB"
    }

    return result


# ===================== 使用示例 =====================

if __name__ == "__main__":
    print("=" * 60)
    print("  微调显存需求估算")
    print("=" * 60)

    for model_size in [7, 13, 70]:
        print(f"\n{'='*40}")
        print(f"  模型大小: {model_size}B 参数")
        print(f"{'='*40}")

        for method in ["full", "lora", "qlora"]:
            mem = estimate_memory_usage(model_size, method)
            print(f"\n  {method.upper()} 微调:")
            for k, v in mem.items():
                print(f"    {k}: {v}")
```

---

## LoRA原理与实现

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LoRA (Low-Rank Adaptation) 原理图                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  核心思想: 权重更新矩阵 Delta_W 是低秩的                                │
│                                                                         │
│  原始前向传播:        LoRA前向传播:                                      │
│  h = W * x             h = W * x + (B * A) * x                         │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                 │   │
│  │          原始权重 W (冻结)              LoRA适配器               │   │
│  │         ┌───────────────┐                                      │   │
│  │         │               │           ┌───┐                      │   │
│  │   x ───►│   W (d x d)  │───┐   x ──►│ A │──► r维 ──►┌───┐    │   │
│  │         │   (frozen)    │   │        │dxr│          │ B │    │   │
│  │         └───────────────┘   │        └───┘          │rxd│    │   │
│  │                             │     (降维)            └─┬─┘    │   │
│  │                             │                   (升维) │      │   │
│  │                             │                         │      │   │
│  │                             ▼         × alpha/r       ▼      │   │
│  │                          ┌─────┐                  ┌─────┐    │   │
│  │                          │  +  │◄─────────────────│scale│    │   │
│  │                          └──┬──┘                  └─────┘    │   │
│  │                             │                                │   │
│  │                             ▼                                │   │
│  │                          h = W*x + (alpha/r) * B*A*x         │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  参数量对比:                                                            │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │  原始权重 W:  d_in × d_out = 4096 × 4096 = 16,777,216  │          │
│  │  LoRA参数:    A: d_in × r  = 4096 × 16   = 65,536      │          │
│  │              B: r × d_out = 16 × 4096   = 65,536      │          │
│  │  合计:        131,072  (占原始的 0.78%)                  │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                         │
│  LoRA应用位置 (以Llama为例):                                            │
│  ┌──────────────────────────────────────────────────┐                  │
│  │  Transformer Layer                                │                  │
│  │  ├── Self-Attention                              │                  │
│  │  │   ├── q_proj  ← LoRA (推荐)                  │                  │
│  │  │   ├── k_proj  ← LoRA (推荐)                  │                  │
│  │  │   ├── v_proj  ← LoRA (推荐)                  │                  │
│  │  │   └── o_proj  ← LoRA (推荐)                  │                  │
│  │  └── MLP                                         │                  │
│  │      ├── gate_proj ← LoRA (可选)                 │                  │
│  │      ├── up_proj   ← LoRA (可选)                 │                  │
│  │      └── down_proj ← LoRA (可选)                 │                  │
│  └──────────────────────────────────────────────────┘                  │
│                                                                         │
│  LoRA超参数选择指南:                                                    │
│  ┌──────────┬──────────┬───────────────────────────┐                   │
│  │  参数    │  推荐值  │  说明                     │                   │
│  ├──────────┼──────────┼───────────────────────────┤                   │
│  │  r       │  8~64    │  秩，越大表达能力越强     │                   │
│  │  alpha   │  2*r     │  缩放因子，通常为r的2倍   │                   │
│  │  dropout │  0.05    │  LoRA层的dropout          │                   │
│  │  target  │ q,k,v,o  │  至少包含attention投影    │                   │
│  └──────────┴──────────┴───────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**LoRA（Low-Rank Adaptation，低秩适配）** 是微软在2021年提出的参数高效微调方法。其核心洞察是：微调时权重的更新矩阵 Delta_W 具有低秩（low-rank）特性，即可以用两个小矩阵的乘积来近似。

**LoRA的数学原理：**

原始前向传播：`h = W * x`

LoRA修改后：`h = W * x + (alpha/r) * B * A * x`

其中：
- `W`: 原始权重矩阵 (d_in x d_out)，冻结不更新
- `A`: 降维矩阵 (d_in x r)，用高斯随机初始化
- `B`: 升维矩阵 (r x d_out)，初始化为零
- `r`: 秩（rank），远小于d_in和d_out
- `alpha`: 缩放因子，控制LoRA的影响程度

**为什么LoRA有效？**

1. **低秩假设**：大模型微调时的权重更新是低秩的，不需要更新所有参数
2. **初始化策略**：B初始化为零确保训练开始时LoRA不改变原始模型行为
3. **无额外推理延迟**：训练完成后可将LoRA权重合并回原始模型 `W' = W + (alpha/r) * B * A`
4. **可组合性**：不同任务的LoRA适配器可以热切换，一个基础模型服务多个任务

**LoRA关键超参数：**

| 参数 | 说明 | 推荐值 | 影响 |
|------|------|--------|------|
| `r` | 秩/维度 | 8-64 | 越大越好但参数越多 |
| `lora_alpha` | 缩放因子 | 2*r | 控制LoRA权重的影响力 |
| `lora_dropout` | Dropout率 | 0.05 | 防止过拟合 |
| `target_modules` | 应用的层 | q,k,v,o投影 | 覆盖越多效果越好 |

### 代码示例

```python
# LoRA 从零实现
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Dict


# ===================== 1. LoRA线性层实现 =====================

class LoRALinear(nn.Module):
    """LoRA线性层 -- 从零实现

    将原始的nn.Linear替换为带LoRA适配器的版本。
    原始权重冻结，只训练A和B两个小矩阵。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        merge_weights: bool = False
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.merged = False

        # 原始权重 (冻结)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

        # 冻结原始权重
        self.weight.requires_grad = False
        self.bias.requires_grad = False

        # LoRA参数 (可训练)
        if r > 0:
            # A: 降维矩阵 (in_features, r) -- 高斯初始化
            self.lora_A = nn.Parameter(
                torch.randn(in_features, r) * (1 / math.sqrt(r))
            )
            # B: 升维矩阵 (r, out_features) -- 零初始化
            self.lora_B = nn.Parameter(
                torch.zeros(r, out_features)
            )
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播: h = Wx + b + (alpha/r) * B * A * x"""
        # 原始线性变换
        result = F.linear(x, self.weight, self.bias)

        # 添加LoRA
        if self.r > 0 and not self.merged:
            lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B
            result = result + lora_out * self.scaling

        return result

    def merge(self):
        """将LoRA权重合并到原始权重中 (部署时使用)"""
        if self.r > 0 and not self.merged:
            # W' = W + (alpha/r) * B^T * A^T
            # 注意: weight是(out, in), 所以需要转置
            delta_w = (self.lora_B.T @ self.lora_A.T) * self.scaling
            self.weight.data += delta_w
            self.merged = True
            print(f"LoRA权重已合并: {self.in_features}x{self.out_features}")

    def unmerge(self):
        """从原始权重中移除LoRA (切换适配器时使用)"""
        if self.r > 0 and self.merged:
            delta_w = (self.lora_B.T @ self.lora_A.T) * self.scaling
            self.weight.data -= delta_w
            self.merged = False

    @property
    def lora_parameters(self) -> int:
        """LoRA可训练参数数量"""
        if self.r > 0:
            return self.lora_A.numel() + self.lora_B.numel()
        return 0


# ===================== 2. 将LoRA注入到现有模型 =====================

def inject_lora(
    model: nn.Module,
    target_modules: List[str],
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05
) -> nn.Module:
    """将LoRA注入到模型的指定模块中

    Args:
        model: 原始模型
        target_modules: 要注入LoRA的模块名称列表
        r: LoRA秩
        lora_alpha: LoRA缩放因子
        lora_dropout: LoRA Dropout率

    Returns:
        注入LoRA后的模型
    """
    # 冻结所有原始参数
    for param in model.parameters():
        param.requires_grad = False

    # 替换目标模块
    replaced_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 检查是否是目标模块
            module_name = name.split(".")[-1]
            if module_name in target_modules:
                # 创建LoRA替代层
                lora_layer = LoRALinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout
                )
                # 复制原始权重
                lora_layer.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_layer.bias.data = module.bias.data.clone()

                # 替换模块
                parent = _get_parent_module(model, name)
                child_name = name.split(".")[-1]
                setattr(parent, child_name, lora_layer)
                replaced_count += 1

    print(f"已将LoRA注入到 {replaced_count} 个模块")

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable:,} ({trainable/total_params*100:.2f}%)")

    return model


def _get_parent_module(model: nn.Module, name: str) -> nn.Module:
    """获取父模块"""
    parts = name.split(".")
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
    return module


# ===================== 3. 示例: 给Transformer模型注入LoRA =====================

class SimpleTransformerBlock(nn.Module):
    """简化的Transformer Block (用于演示)"""

    def __init__(self, d_model: int = 512, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model

        # Self-Attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # FFN
        self.gate_proj = nn.Linear(d_model, d_model * 4)
        self.up_proj = nn.Linear(d_model, d_model * 4)
        self.down_proj = nn.Linear(d_model * 4, d_model)

        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 简化版前向传播 (省略attention计算细节)
        residual = x
        x = self.norm1(x)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 简化: 直接用v作为attention输出
        attn_out = self.o_proj(v)
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = self.down_proj(gate * up)
        x = residual + x
        return x


class SimpleLanguageModel(nn.Module):
    """简化的语言模型 (用于演示LoRA)"""

    def __init__(self, vocab_size=32000, d_model=512,
                 n_layers=4, n_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


# ===================== 演示 =====================

def lora_demo():
    """LoRA完整演示"""
    print("=" * 60)
    print("  LoRA 从零实现演示")
    print("=" * 60)

    # 1. 创建基础模型
    model = SimpleLanguageModel(
        vocab_size=32000, d_model=256,
        n_layers=4, n_heads=8
    )

    total_before = sum(p.numel() for p in model.parameters())
    print(f"\n原始模型参数量: {total_before:,}")

    # 2. 注入LoRA
    print("\n注入LoRA...")
    target = ["q_proj", "k_proj", "v_proj", "o_proj"]
    model = inject_lora(model, target, r=16, lora_alpha=32)

    # 3. 验证只有LoRA参数可训练
    print("\n可训练参数:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}")

    # 4. 模拟训练
    print("\n模拟训练...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4
    )

    input_ids = torch.randint(0, 32000, (2, 32))
    labels = torch.randint(0, 32000, (2, 32))

    for step in range(5):
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, 32000), labels.view(-1)
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  Step {step+1}: loss = {loss.item():.4f}")

    # 5. 合并LoRA权重 (部署)
    print("\n合并LoRA权重...")
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.merge()

    print("\nLoRA演示完成!")


if __name__ == "__main__":
    lora_demo()
```

---

## QLoRA量化微调

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    QLoRA (Quantized LoRA) 原理图                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  核心创新: 4-bit量化模型 + LoRA适配器 + 分页优化器                       │
│                                                                         │
│  标准LoRA:                      QLoRA:                                  │
│  ┌─────────────┐                ┌─────────────┐                        │
│  │ W (FP16)    │  16GB          │ W (NF4)     │  3.5GB                 │
│  │ 冻结        │                │ 4-bit量化冻结│                        │
│  └──────┬──────┘                └──────┬──────┘                        │
│         │                              │                                │
│  ┌──────┴──────┐                ┌──────┴──────┐                        │
│  │LoRA A (FP16)│                │LoRA A (BF16)│                        │
│  │LoRA B (FP16)│                │LoRA B (BF16)│                        │
│  └─────────────┘                └─────────────┘                        │
│  总计: ~16GB                    总计: ~6GB                              │
│                                                                         │
│  QLoRA的三大核心技术:                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                 │   │
│  │  1. NF4 量化 (4-bit NormalFloat)                               │   │
│  │  ┌────────────────────────────────────────────────────┐        │   │
│  │  │  FP16: 1.2345 → NF4: 量化为4-bit (16个离散值)     │        │   │
│  │  │  基于正态分布优化的量化格式:                        │        │   │
│  │  │  [-1.0, -0.69, -0.52, -0.39, -0.28, -0.18, -0.09, │        │   │
│  │  │    0.0, 0.08, 0.17, 0.27, 0.38, 0.51, 0.68, 1.0]  │        │   │
│  │  │  比均匀INT4量化精度更高（针对正态分布权重优化）      │        │   │
│  │  └────────────────────────────────────────────────────┘        │   │
│  │                                                                 │   │
│  │  2. 双重量化 (Double Quantization)                              │   │
│  │  ┌────────────────────────────────────────────────────┐        │   │
│  │  │  第一次: 权重 FP16 → NF4 (产生量化常数 FP32)       │        │   │
│  │  │  第二次: 量化常数 FP32 → FP8 (进一步压缩)          │        │   │
│  │  │  节省约 0.37 bit/参数 ≈ 每个参数节省约3%存储        │        │   │
│  │  └────────────────────────────────────────────────────┘        │   │
│  │                                                                 │   │
│  │  3. 分页优化器 (Paged Optimizer)                                │   │
│  │  ┌────────────────────────────────────────────────────┐        │   │
│  │  │  GPU显存不足时 → 自动将优化器状态转移到CPU内存      │        │   │
│  │  │  基于NVIDIA统一内存 (Unified Memory) 功能           │        │   │
│  │  │  GPU ◄────► CPU 自动分页，避免OOM                  │        │   │
│  │  └────────────────────────────────────────────────────┘        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  量化精度对比:                                                          │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐              │
│  │          │  FP32    │  FP16    │  INT8    │  NF4     │              │
│  ├──────────┼──────────┼──────────┼──────────┼──────────┤              │
│  │ 每参数   │ 4 bytes  │ 2 bytes  │ 1 byte  │ 0.5 byte │              │
│  │ 7B模型   │  28 GB   │  14 GB   │  7 GB   │  3.5 GB  │              │
│  │ 70B模型  │ 280 GB   │ 140 GB   │  70 GB  │  35 GB   │              │
│  │ 精度损失 │ 基准     │ 极小     │ 小      │  可接受  │              │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**QLoRA** 是华盛顿大学在2023年提出的高效微调方法，通过将模型权重量化为4-bit（NF4格式），配合LoRA适配器，使得在消费级GPU（如RTX 3090/4090的24GB显存）上也能微调65B参数的大模型。

**QLoRA的关键创新点：**

1. **NF4量化（4-bit NormalFloat）**：基于正态分布优化的量化格式。由于预训练模型的权重通常呈正态分布，NF4比均匀量化（INT4）精度更高。

2. **双重量化（Double Quantization）**：对量化过程中产生的缩放因子再做一次量化，进一步减少存储。

3. **分页优化器（Paged Optimizer）**：利用NVIDIA的统一内存管理，在GPU显存不足时自动将优化器状态卸载到CPU。

**QLoRA显存节约对比（7B模型）：**

| 方法 | 模型权重 | LoRA参数 | 优化器 | 总计 |
|------|---------|----------|--------|------|
| Full FT (FP16) | 14GB | - | 28GB+ | ~56GB |
| LoRA (FP16) | 14GB | ~100MB | ~400MB | ~16GB |
| QLoRA (NF4) | 3.5GB | ~100MB | ~400MB | ~6GB |

### 代码示例

```python
# QLoRA - 使用bitsandbytes实现4-bit量化微调
import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass


# ===================== 1. 量化配置 =====================

@dataclass
class QuantizationConfig:
    """量化配置"""
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"     # nf4 或 fp4
    bnb_4bit_use_double_quant: bool = True  # 双重量化


def get_bnb_config(config: QuantizationConfig):
    """获取bitsandbytes量化配置

    需要安装: pip install bitsandbytes>=0.41.0
    """
    try:
        from transformers import BitsAndBytesConfig

        compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        )
        return bnb_config

    except ImportError:
        print("请安装: pip install bitsandbytes transformers")
        return None


# ===================== 2. QLoRA模型加载 =====================

def load_model_qlora(model_name: str, config: QuantizationConfig):
    """加载4-bit量化模型

    Args:
        model_name: 模型名称或路径
        config: 量化配置

    Returns:
        model, tokenizer
    """
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        # 量化配置
        bnb_config = get_bnb_config(config)

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right"
        )

        # 确保有pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 加载4-bit量化模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",           # 自动分配到GPU
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # 禁用缓存 (训练时不需要)
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        print(f"模型加载完成: {model_name}")
        print(f"量化类型: {config.bnb_4bit_quant_type}")
        print(f"计算精度: {config.bnb_4bit_compute_dtype}")
        print(f"双重量化: {config.bnb_4bit_use_double_quant}")

        return model, tokenizer

    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请安装: pip install transformers bitsandbytes accelerate")
        return None, None


# ===================== 3. 添加LoRA适配器 =====================

def add_lora_adapter(
    model,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list = None,
    task_type: str = "CAUSAL_LM"
):
    """给量化模型添加LoRA适配器

    需要安装: pip install peft>=0.6.0
    """
    try:
        from peft import (
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
            TaskType
        )

        # 准备模型 (处理量化模型的梯度检查点等)
        model = prepare_model_for_kbit_training(model)

        # 默认目标模块
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        # LoRA配置
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # 添加LoRA
        model = get_peft_model(model, lora_config)

        # 打印可训练参数统计
        model.print_trainable_parameters()

        return model, lora_config

    except ImportError:
        print("请安装: pip install peft")
        return model, None


# ===================== 4. 模拟NF4量化过程 =====================

def simulate_nf4_quantization(tensor: torch.Tensor) -> Dict:
    """模拟NF4量化过程（教学用途）

    NF4量化的核心步骤:
    1. 将权重分为block
    2. 每个block计算缩放因子
    3. 归一化后映射到NF4离散值
    """
    # NF4的16个离散值 (基于标准正态分布的分位数)
    nf4_values = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949,
        -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379,
        0.4407, 0.5626, 0.7230, 1.0
    ])

    # 原始统计
    original_size = tensor.numel() * tensor.element_size()  # bytes
    original_mean = tensor.mean().item()
    original_std = tensor.std().item()

    # 量化过程
    block_size = 64  # 每个block的大小
    flat = tensor.flatten().float()

    # 分block处理
    n_blocks = (flat.numel() + block_size - 1) // block_size
    padded_size = n_blocks * block_size
    padded = torch.zeros(padded_size)
    padded[:flat.numel()] = flat

    quantized_indices = torch.zeros(padded_size, dtype=torch.uint8)
    scales = torch.zeros(n_blocks)

    for i in range(n_blocks):
        block = padded[i * block_size: (i + 1) * block_size]
        # 缩放因子 = max(abs(block))
        scale = block.abs().max()
        scales[i] = scale

        if scale > 0:
            # 归一化到 [-1, 1]
            normalized = block / scale
            # 映射到最近的NF4值
            distances = (normalized.unsqueeze(-1) -
                        nf4_values.unsqueeze(0)).abs()
            quantized_indices[i * block_size: (i + 1) * block_size] = \
                distances.argmin(dim=-1).to(torch.uint8)

    # 反量化
    dequantized = torch.zeros(padded_size)
    for i in range(n_blocks):
        indices = quantized_indices[i * block_size: (i + 1) * block_size]
        dequantized[i * block_size: (i + 1) * block_size] = \
            nf4_values[indices.long()] * scales[i]

    dequantized = dequantized[:flat.numel()].reshape(tensor.shape)

    # 计算量化误差
    quant_error = (tensor.float() - dequantized).abs().mean().item()
    relative_error = quant_error / (tensor.float().abs().mean().item() + 1e-8)

    # 量化后大小: 每个参数 4bit + 缩放因子开销
    quantized_size = (flat.numel() * 4 / 8 +  # 4-bit参数
                      n_blocks * 4)             # FP32缩放因子
    compression_ratio = original_size / quantized_size

    return {
        "原始大小": f"{original_size / 1024:.1f} KB",
        "量化大小": f"{quantized_size / 1024:.1f} KB",
        "压缩比": f"{compression_ratio:.2f}x",
        "平均量化误差": f"{quant_error:.6f}",
        "相对误差": f"{relative_error:.4%}",
        "块数": n_blocks,
    }


# ===================== 使用示例 =====================

if __name__ == "__main__":
    print("=" * 60)
    print("  QLoRA 量化微调演示")
    print("=" * 60)

    # 模拟NF4量化
    print("\n--- NF4量化模拟 ---")
    # 创建一个模拟的权重矩阵 (正态分布)
    weight = torch.randn(1024, 1024) * 0.02  # 模拟预训练权重

    result = simulate_nf4_quantization(weight)
    for k, v in result.items():
        print(f"  {k}: {v}")

    # QLoRA完整流程代码模板
    print("\n--- QLoRA完整流程 (代码模板) ---")
    print("""
    # 步骤1: 加载4-bit量化模型
    model, tokenizer = load_model_qlora(
        "meta-llama/Llama-3.1-8B-Instruct",
        QuantizationConfig()
    )

    # 步骤2: 添加LoRA适配器
    model, lora_config = add_lora_adapter(
        model, r=16, lora_alpha=32
    )

    # 步骤3: 配置训练参数 (见Trainer部分)

    # 步骤4: 训练

    # 步骤5: 合并并保存
    model = model.merge_and_unload()
    model.save_pretrained("./merged_model")
    tokenizer.save_pretrained("./merged_model")
    """)
```

---

## PEFT框架使用

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PEFT (Parameter-Efficient Fine-Tuning) 框架           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PEFT 是 Hugging Face 推出的参数高效微调统一框架                         │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    PEFT 支持的方法                               │   │
│  │                                                                 │   │
│  │  ┌──────────────────┐  ┌──────────────────┐                    │   │
│  │  │    LoRA          │  │    IA3            │                    │   │
│  │  │ 低秩适配         │  │ 抑制/放大激活     │                    │   │
│  │  │ 最常用           │  │ 参数更少          │                    │   │
│  │  └──────────────────┘  └──────────────────┘                    │   │
│  │  ┌──────────────────┐  ┌──────────────────┐                    │   │
│  │  │  AdaLoRA         │  │  Prefix Tuning   │                    │   │
│  │  │ 自适应秩分配     │  │ 前缀调优         │                    │   │
│  │  │ 自动选择最优r    │  │ 在注意力前添加   │                    │   │
│  │  └──────────────────┘  └──────────────────┘                    │   │
│  │  ┌──────────────────┐  ┌──────────────────┐                    │   │
│  │  │  Prompt Tuning   │  │  P-Tuning v2     │                    │   │
│  │  │ 软提示调优       │  │ 深度提示调优     │                    │   │
│  │  │ 只在输入添加     │  │ 每层都添加       │                    │   │
│  │  └──────────────────┘  └──────────────────┘                    │   │
│  │  ┌──────────────────┐  ┌──────────────────┐                    │   │
│  │  │  LoftQ           │  │  QLoRA           │                    │   │
│  │  │ 量化感知LoRA     │  │ 4-bit量化+LoRA   │                    │   │
│  │  │ 更好的初始化     │  │ 最节省显存       │                    │   │
│  │  └──────────────────┘  └──────────────────┘                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  PEFT工作流程:                                                          │
│  ┌────────┐  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌─────────┐   │
│  │ 加载   │─►│ 选择    │─►│ 配置     │─►│ 训练    │─►│ 保存/   │   │
│  │ 基模型 │  │ PEFT方法│  │ get_peft │  │ 只更新  │  │ 合并    │   │
│  │        │  │ LoRA等  │  │ _model() │  │ 适配器  │  │ 权重    │   │
│  └────────┘  └─────────┘  └──────────┘  └─────────┘  └─────────┘   │
│                                                                         │
│  适配器管理 (一个基座模型 + 多个适配器):                                 │
│  ┌─────────────────────────────────────────────────┐                   │
│  │          Base Model (Llama-3.1-8B)              │                   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐            │                   │
│  │  │ adapter│  │ adapter│  │ adapter│            │                   │
│  │  │ 中文   │  │ 代码   │  │ 医疗   │            │                   │
│  │  │ ~50MB  │  │ ~50MB  │  │ ~50MB  │            │                   │
│  │  └────────┘  └────────┘  └────────┘            │                   │
│  │  model.set_adapter("chinese")  ← 热切换       │                   │
│  └─────────────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**PEFT（Parameter-Efficient Fine-Tuning）** 是Hugging Face推出的参数高效微调框架，提供了统一的API来使用各种高效微调方法。PEFT的核心优势是：只需几行代码就能给任意Hugging Face模型添加参数高效的微调能力。

**PEFT的核心API：**

| API | 功能 | 使用场景 |
|-----|------|----------|
| `LoraConfig` | 配置LoRA超参数 | 定义LoRA设置 |
| `get_peft_model()` | 给模型添加PEFT适配器 | 初始化微调 |
| `prepare_model_for_kbit_training()` | 准备量化模型 | QLoRA场景 |
| `model.print_trainable_parameters()` | 打印可训练参数 | 检查配置 |
| `model.save_pretrained()` | 保存适配器 | 保存训练结果 |
| `PeftModel.from_pretrained()` | 加载适配器 | 推理/继续训练 |
| `model.merge_and_unload()` | 合并权重 | 部署模型 |

**选择PEFT方法的决策树：**

1. 需要接近全量微调的效果？ --> **LoRA** (r=64, 所有线性层)
2. 显存极度受限(消费级GPU)？ --> **QLoRA** (4-bit量化+LoRA)
3. 需要多任务快速切换？ --> **LoRA** (多适配器)
4. 数据极少(<100条)？ --> **Prompt Tuning**
5. 需要最少参数？ --> **IA3**

### 代码示例

```python
# PEFT框架使用 - 完整示例
import torch
from typing import Dict, List, Optional


# ===================== 1. PEFT LoRA配置示例 =====================

def create_lora_config_examples():
    """各种场景的LoRA配置示例"""

    configs = {}

    # 场景1: 通用对话微调 (推荐配置)
    configs["通用对话"] = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    # 场景2: 轻量级适配 (参数最少)
    configs["轻量适配"] = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    # 场景3: 高质量微调 (接近Full FT)
    configs["高质量"] = {
        "r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj",
                           "lm_head"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    # 场景4: 分类任务
    configs["分类任务"] = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
        "task_type": "SEQ_CLS",
        "modules_to_save": ["classifier"],  # 分类头全量训练
    }

    return configs


# ===================== 2. 完整PEFT微调流程 =====================

def peft_finetune_template():
    """PEFT微调完整代码模板"""

    code = '''
# === 完整的PEFT微调代码模板 ===

# Step 1: 安装依赖
# pip install transformers peft datasets accelerate bitsandbytes

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)
from datasets import load_dataset
from trl import SFTTrainer  # pip install trl

# Step 2: 配置量化 (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Step 3: 加载模型和分词器
model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Step 4: 准备模型
model = prepare_model_for_kbit_training(model)

# Step 5: 配置LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Step 6: 加载数据集
dataset = load_dataset("json", data_files="train.json", split="train")

def format_instruction(sample):
    """格式化训练数据"""
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample.get('input', '')}

### Response:
{sample['output']}"""

# Step 7: 训练配置
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.001,
    max_grad_norm=0.3,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
)

# Step 8: 创建训练器
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    formatting_func=format_instruction,
    max_seq_length=2048,
    packing=True,   # 将短样本打包在一起
)

# Step 9: 训练
trainer.train()

# Step 10: 保存LoRA适配器
model.save_pretrained("./lora_adapter")
tokenizer.save_pretrained("./lora_adapter")

# Step 11: 推理 (加载适配器)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./lora_adapter")

# Step 12: 合并权重 (可选, 用于部署)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
'''
    return code


# ===================== 3. 适配器管理 =====================

def adapter_management_demo():
    """适配器管理示例代码"""

    code = '''
# === 多适配器管理 ===

from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

# 加载第一个适配器 (中文对话)
model = PeftModel.from_pretrained(
    base_model,
    "./adapters/chinese_chat",
    adapter_name="chinese"
)

# 添加第二个适配器 (代码生成)
model.load_adapter("./adapters/code_gen", adapter_name="code")

# 添加第三个适配器 (医疗问答)
model.load_adapter("./adapters/medical_qa", adapter_name="medical")

# 切换适配器
model.set_adapter("chinese")   # 使用中文对话适配器
output1 = model.generate(...)

model.set_adapter("code")      # 切换到代码生成适配器
output2 = model.generate(...)

model.set_adapter("medical")   # 切换到医疗问答适配器
output3 = model.generate(...)

# 禁用所有适配器 (使用原始模型)
with model.disable_adapter():
    output_base = model.generate(...)
'''
    return code


# ===================== 使用示例 =====================

if __name__ == "__main__":
    print("=" * 60)
    print("  PEFT 框架使用指南")
    print("=" * 60)

    # 打印各场景配置
    configs = create_lora_config_examples()
    for name, config in configs.items():
        print(f"\n--- {name} 配置 ---")
        for k, v in config.items():
            print(f"  {k}: {v}")

    # 打印完整代码模板
    print("\n" + "=" * 60)
    print("  完整微调代码模板")
    print("=" * 60)
    print(peft_finetune_template())
```

---

## Hugging Face Trainer

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Hugging Face Trainer 训练流程                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Trainer是Hugging Face提供的高级训练API, 封装了完整的训练循环            │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Trainer 组件                                  │   │
│  │                                                                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │    Model     │  │  Tokenizer   │  │   Dataset    │          │   │
│  │  │  待训练模型  │  │  分词器      │  │  训练数据    │          │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │   │
│  │         │                 │                  │                  │   │
│  │         └────────────┬────┴──────────────────┘                  │   │
│  │                      ▼                                          │   │
│  │  ┌──────────────────────────────────────────────────────┐      │   │
│  │  │              TrainingArguments                        │      │   │
│  │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐       │      │   │
│  │  │  │ 学习率     │ │ Batch大小  │ │ Epoch数    │       │      │   │
│  │  │  │ 优化器     │ │ 梯度累积   │ │ 混合精度   │       │      │   │
│  │  │  │ 调度器     │ │ 梯度检查点 │ │ 保存策略   │       │      │   │
│  │  │  └────────────┘ └────────────┘ └────────────┘       │      │   │
│  │  └──────────────────────────┬───────────────────────────┘      │   │
│  │                             ▼                                   │   │
│  │  ┌──────────────────────────────────────────────────────┐      │   │
│  │  │              Trainer / SFTTrainer                     │      │   │
│  │  │                                                      │      │   │
│  │  │  trainer.train()     → 自动执行训练循环               │      │   │
│  │  │  trainer.evaluate()  → 评估模型                      │      │   │
│  │  │  trainer.predict()   → 批量预测                      │      │   │
│  │  │  trainer.save_model() → 保存模型                     │      │   │
│  │  │                                                      │      │   │
│  │  │  内置功能:                                           │      │   │
│  │  │  - 分布式训练 (DDP/FSDP/DeepSpeed)                  │      │   │
│  │  │  - 混合精度 (FP16/BF16)                             │      │   │
│  │  │  - 梯度累积/梯度检查点                              │      │   │
│  │  │  - 早停/最佳模型保存                                │      │   │
│  │  │  - WandB/TensorBoard日志                            │      │   │
│  │  │  - 断点续训                                         │      │   │
│  │  └──────────────────────────────────────────────────────┘      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  SFTTrainer vs Trainer:                                                 │
│  ┌──────────────────┬──────────────────────────────────────────┐       │
│  │  Trainer         │ 通用训练器, 需要自行处理数据格式         │       │
│  │  SFTTrainer      │ 监督微调专用, 自动处理对话格式/打包      │       │
│  │  DPOTrainer      │ RLHF直接偏好优化训练器                  │       │
│  │  PPOTrainer      │ RLHF PPO训练器                          │       │
│  └──────────────────┴──────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**Hugging Face Trainer** 是一个功能强大的高级训练API，它封装了完整的训练循环，让用户只需关注模型、数据和超参数的配置。**SFTTrainer**（来自trl库）是专门为监督微调（Supervised Fine-Tuning）设计的Trainer变体，内置了对话格式处理、序列打包等微调专用功能。

**TrainingArguments 关键参数：**

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `per_device_train_batch_size` | 每GPU批次大小 | 4 (显存允许时更大) |
| `gradient_accumulation_steps` | 梯度累积步数 | 4-8 (等效batch=16-32) |
| `learning_rate` | 学习率 | 2e-4 (LoRA), 2e-5 (Full FT) |
| `lr_scheduler_type` | 学习率调度 | "cosine" |
| `warmup_ratio` | 预热比例 | 0.03-0.1 |
| `num_train_epochs` | 训练轮数 | 1-3 (微调通常不需要太多) |
| `bf16` | BFloat16混合精度 | True (A100/4090) |
| `fp16` | Float16混合精度 | True (V100/3090) |
| `gradient_checkpointing` | 梯度检查点 | True (节省显存) |
| `optim` | 优化器 | "paged_adamw_32bit" (QLoRA) |
| `save_strategy` | 保存策略 | "epoch" 或 "steps" |
| `logging_steps` | 日志间隔 | 10-50 |

### 代码示例

```python
# Hugging Face Trainer - 完整训练代码
import json
from typing import Dict, List
from dataclasses import dataclass


# ===================== 1. 数据准备工具 =====================

def create_sample_dataset(output_path: str = "train_data.json",
                          n_samples: int = 100):
    """创建示例训练数据"""
    samples = []

    # 示例1: 翻译任务
    translations = [
        ("Hello", "你好"),
        ("Thank you", "谢谢"),
        ("Good morning", "早上好"),
        ("How are you", "你好吗"),
        ("Goodbye", "再见"),
    ]
    for en, zh in translations:
        samples.append({
            "instruction": "将以下英文翻译成中文",
            "input": en,
            "output": zh
        })

    # 示例2: 摘要任务
    for i in range(20):
        samples.append({
            "instruction": "用一句话总结以下内容",
            "input": f"这是第{i+1}篇需要摘要的长文本内容...",
            "output": f"这是第{i+1}篇文本的精简摘要。"
        })

    # 示例3: 问答任务
    qa_pairs = [
        ("什么是深度学习？", "深度学习是机器学习的一个子领域，使用多层神经网络从数据中学习特征表示。"),
        ("PyTorch是什么？", "PyTorch是由Meta开发的开源深度学习框架，以动态计算图著称。"),
        ("什么是Transformer？", "Transformer是一种基于自注意力机制的神经网络架构，是现代LLM的基础。"),
    ]
    for q, a in qa_pairs:
        samples.append({
            "instruction": q,
            "input": "",
            "output": a
        })

    # 补充到n_samples
    while len(samples) < n_samples:
        idx = len(samples) % len(samples)
        samples.append(samples[idx].copy())

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples[:n_samples], f, ensure_ascii=False, indent=2)

    print(f"创建了 {n_samples} 条训练数据: {output_path}")
    return output_path


# ===================== 2. 完整训练脚本 =====================

FULL_TRAINING_SCRIPT = '''
#!/usr/bin/env python3
"""
完整的QLoRA微调脚本
使用方法: python train.py
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import load_dataset
from trl import SFTTrainer
import os

# ==================== 配置 ====================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_PATH = "train_data.json"
OUTPUT_DIR = "./output"

# LoRA配置
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# 训练配置
EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 2048

# ==================== 主流程 ====================

def main():
    print("=" * 60)
    print("  QLoRA 微调训练")
    print("=" * 60)

    # 1. 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. 加载分词器
    print("\\n[1/5] 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. 加载模型
    print("[2/5] 加载4-bit量化模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # 4. 添加LoRA
    print("[3/5] 配置LoRA...")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. 加载数据集
    print("[4/5] 加载数据集...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"  数据集大小: {len(dataset)}")

    # 格式化函数
    def format_instruction(sample):
        text = f"""### Instruction:
{sample['instruction']}

### Input:
{sample.get('input', '')}

### Response:
{sample['output']}"""
        return text

    # 6. 训练配置
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.001,
        max_grad_norm=0.3,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,
        report_to="none",  # 或 "wandb"
        seed=42,
    )

    # 7. 创建训练器
    print("[5/5] 开始训练...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        formatting_func=format_instruction,
        max_seq_length=MAX_SEQ_LEN,
        packing=True,
    )

    # 8. 训练
    trainer.train()

    # 9. 保存
    print("\\n保存模型...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

    print("\\n训练完成!")
    print(f"模型保存在: {OUTPUT_DIR}/final")

if __name__ == "__main__":
    main()
'''


# ===================== 3. 推理脚本 =====================

INFERENCE_SCRIPT = '''
#!/usr/bin/env python3
"""微调模型推理脚本"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def load_finetuned_model(base_model_name, adapter_path):
    """加载微调后的模型"""

    # 量化配置 (与训练时一致)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 加载base模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # 加载LoRA适配器
    model = PeftModel.from_pretrained(model, adapter_path)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    return model, tokenizer


def generate_response(model, tokenizer, instruction, input_text="",
                      max_new_tokens=512, temperature=0.7):
    """生成回复"""

    prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()
'''


if __name__ == "__main__":
    print("=" * 60)
    print("  Hugging Face Trainer 使用指南")
    print("=" * 60)

    # 创建示例数据
    create_sample_dataset("sample_train.json", 50)

    print("\n完整训练脚本已准备就绪")
    print("使用方法: python train.py")
```

---

## 完整实战：微调Llama模型

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    完整微调实战流程                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  端到端流程:                                                            │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐         │
│  │ 数据 │─►│ 预处理│─►│ 配置 │─►│ 训练 │─►│ 评估 │─►│ 部署 │         │
│  │ 收集 │  │ 清洗  │  │ 模型 │  │ 微调 │  │ 测试 │  │ 推理 │         │
│  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘         │
│      │          │          │          │          │          │           │
│      ▼          ▼          ▼          ▼          ▼          ▼           │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐         │
│  │JSON  │  │格式化│  │QLoRA │  │SFT   │  │BLEU  │  │vLLM  │         │
│  │收集  │  │标准化│  │4-bit │  │Trainer│  │Rouge │  │合并  │         │
│  │标注  │  │去重  │  │LoRA  │  │训练   │  │人工  │  │部署  │         │
│  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘         │
│                                                                         │
│  硬件需求 (QLoRA):                                                      │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  模型      │  最低GPU          │  推荐GPU                   │      │
│  │  7/8B     │  RTX 3060 (12GB)  │  RTX 3090/4090 (24GB)     │      │
│  │  13B      │  RTX 3090 (24GB)  │  RTX 4090 (24GB)          │      │
│  │  70B      │  A100 (40GB)      │  A100 (80GB) x 2          │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  常见问题排查:                                                          │
│  ┌──────────────────────────────────────────────────────┐              │
│  │  OOM         → 减小batch_size / 增加gradient_accum  │              │
│  │  Loss不降    → 检查数据格式 / 调整学习率            │              │
│  │  Loss突增    → 降低学习率 / 增加warmup              │              │
│  │  过拟合      → 减少epochs / 增加dropout / 更多数据  │              │
│  │  生成质量差  → 增加r值 / 覆盖更多target_modules     │              │
│  └──────────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 详细说明

本节将展示一个完整的端到端微调流程，从数据准备到模型部署。以微调Llama-3.1-8B-Instruct为例，使用QLoRA方法，让模型学会特定领域的问答能力。

**微调步骤清单：**

1. **数据收集与标注**：收集领域数据，整理为标准格式
2. **数据预处理**：清洗、去重、格式化、分词
3. **模型配置**：选择基础模型、配置量化和LoRA参数
4. **训练执行**：使用SFTTrainer进行监督微调
5. **模型评估**：自动指标（loss/perplexity）+ 人工评估
6. **模型导出**：合并LoRA权重、导出为标准格式
7. **部署推理**：使用vLLM或TGI进行高性能推理

**微调数据质量准则：**

| 准则 | 说明 | 重要性 |
|------|------|--------|
| 多样性 | 覆盖各种问题类型和表述方式 | 高 |
| 准确性 | 答案必须正确、无歧义 | 极高 |
| 一致性 | 格式和风格统一 | 高 |
| 数据量 | 通常500-5000条高质量数据即可 | 中 |
| 去重 | 移除重复和近似重复样本 | 高 |
| 长度 | 答案长度适中，避免过短或过长 | 中 |

### 代码示例

```python
# 完整微调实战 - 端到端流程
import json
import os
import hashlib
from typing import List, Dict, Tuple, Optional
from collections import Counter


# ===================== 1. 数据收集与预处理 =====================

class DataPreprocessor:
    """微调数据预处理器"""

    def __init__(self, min_instruction_len: int = 5,
                 min_output_len: int = 10,
                 max_output_len: int = 2048):
        self.min_instruction_len = min_instruction_len
        self.min_output_len = min_output_len
        self.max_output_len = max_output_len

    def load_raw_data(self, file_path: str) -> List[Dict]:
        """加载原始数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"加载了 {len(data)} 条原始数据")
        return data

    def clean_text(self, text: str) -> str:
        """文本清洗"""
        # 移除多余空白
        text = ' '.join(text.split())
        # 移除特殊字符
        text = text.strip()
        return text

    def validate_sample(self, sample: Dict) -> Tuple[bool, str]:
        """验证单条数据"""
        instruction = sample.get("instruction", "")
        output = sample.get("output", "")

        if len(instruction) < self.min_instruction_len:
            return False, "指令太短"
        if len(output) < self.min_output_len:
            return False, "输出太短"
        if len(output) > self.max_output_len:
            return False, "输出太长"
        if not instruction.strip():
            return False, "指令为空"
        if not output.strip():
            return False, "输出为空"
        return True, "通过"

    def deduplicate(self, data: List[Dict]) -> List[Dict]:
        """基于内容hash去重"""
        seen = set()
        unique = []

        for sample in data:
            # 基于instruction+input的hash去重
            content = sample.get("instruction", "") + \
                     sample.get("input", "")
            content_hash = hashlib.md5(
                content.encode('utf-8')
            ).hexdigest()

            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(sample)

        removed = len(data) - len(unique)
        if removed > 0:
            print(f"去重: 移除了 {removed} 条重复数据")
        return unique

    def process(self, data: List[Dict]) -> List[Dict]:
        """完整预处理流程"""
        print(f"\n开始预处理 {len(data)} 条数据...")

        # 步骤1: 清洗
        for sample in data:
            sample["instruction"] = self.clean_text(
                sample.get("instruction", ""))
            sample["input"] = self.clean_text(
                sample.get("input", ""))
            sample["output"] = self.clean_text(
                sample.get("output", ""))

        # 步骤2: 验证
        valid_data = []
        error_counts = Counter()
        for sample in data:
            is_valid, reason = self.validate_sample(sample)
            if is_valid:
                valid_data.append(sample)
            else:
                error_counts[reason] += 1

        if error_counts:
            print(f"过滤掉的数据:")
            for reason, count in error_counts.items():
                print(f"  {reason}: {count} 条")

        # 步骤3: 去重
        valid_data = self.deduplicate(valid_data)

        print(f"预处理完成: {len(valid_data)} 条有效数据")
        return valid_data

    def split_data(self, data: List[Dict],
                   val_ratio: float = 0.1) -> Tuple[List, List]:
        """分割训练集和验证集"""
        import random
        random.seed(42)
        random.shuffle(data)

        split_idx = int(len(data) * (1 - val_ratio))
        train = data[:split_idx]
        val = data[split_idx:]

        print(f"训练集: {len(train)} 条")
        print(f"验证集: {len(val)} 条")
        return train, val

    def save_data(self, data: List[Dict], file_path: str):
        """保存数据"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"数据已保存: {file_path}")


# ===================== 2. 微调配置生成器 =====================

class FineTuneConfigGenerator:
    """根据硬件自动生成最优配置"""

    @staticmethod
    def detect_gpu() -> Dict:
        """检测GPU信息"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_mem
                gpu_mem_gb = gpu_mem / (1024**3)
                return {
                    "name": gpu_name,
                    "memory_gb": gpu_mem_gb,
                    "count": torch.cuda.device_count()
                }
        except Exception:
            pass
        return {"name": "CPU", "memory_gb": 0, "count": 0}

    @staticmethod
    def recommend_config(gpu_memory_gb: float,
                         model_size_b: float = 8) -> Dict:
        """根据GPU显存推荐配置"""

        if gpu_memory_gb >= 80:
            # A100 80GB
            return {
                "method": "LoRA (FP16)",
                "batch_size": 8,
                "grad_accum": 2,
                "lora_r": 64,
                "max_seq_len": 4096,
                "quantization": "none",
                "note": "充足显存, 可使用FP16 LoRA获得最佳效果"
            }
        elif gpu_memory_gb >= 40:
            # A100 40GB
            return {
                "method": "LoRA (FP16)",
                "batch_size": 4,
                "grad_accum": 4,
                "lora_r": 32,
                "max_seq_len": 2048,
                "quantization": "none",
                "note": "较充足显存, FP16 LoRA"
            }
        elif gpu_memory_gb >= 24:
            # RTX 3090/4090
            return {
                "method": "QLoRA (4-bit)",
                "batch_size": 4,
                "grad_accum": 4,
                "lora_r": 16,
                "max_seq_len": 2048,
                "quantization": "nf4",
                "note": "消费级GPU最佳选择"
            }
        elif gpu_memory_gb >= 12:
            # RTX 3060
            return {
                "method": "QLoRA (4-bit)",
                "batch_size": 1,
                "grad_accum": 16,
                "lora_r": 8,
                "max_seq_len": 1024,
                "quantization": "nf4",
                "note": "低显存, 需要减小batch和序列长度"
            }
        else:
            return {
                "method": "不推荐本地微调",
                "note": "建议使用云服务(如AutoDL、RunPod)"
            }


# ===================== 3. 模型评估 =====================

class ModelEvaluator:
    """微调模型评估工具"""

    @staticmethod
    def compute_perplexity(model, tokenizer, texts: List[str],
                           max_length: int = 512) -> float:
        """计算困惑度 (Perplexity)"""
        import torch

        model.eval()
        total_loss = 0
        total_tokens = 0

        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=max_length
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity

    @staticmethod
    def evaluate_generation(model, tokenizer,
                            test_samples: List[Dict],
                            max_new_tokens: int = 256) -> List[Dict]:
        """评估生成质量"""
        import torch

        results = []
        model.eval()

        for sample in test_samples:
            instruction = sample["instruction"]
            input_text = sample.get("input", "")
            expected = sample["output"]

            prompt = f"### Instruction:\n{instruction}\n\n"
            if input_text:
                prompt += f"### Input:\n{input_text}\n\n"
            prompt += "### Response:\n"

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )

            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            results.append({
                "instruction": instruction,
                "expected": expected,
                "generated": generated,
            })

        return results

    @staticmethod
    def print_evaluation_report(results: List[Dict]):
        """打印评估报告"""
        print("\n" + "=" * 60)
        print("  模型评估报告")
        print("=" * 60)

        for i, r in enumerate(results, 1):
            print(f"\n--- 样本 {i} ---")
            print(f"指令: {r['instruction'][:80]}...")
            print(f"期望: {r['expected'][:100]}...")
            print(f"生成: {r['generated'][:100]}...")
            print()


# ===================== 4. 模型导出 =====================

def export_model_guide():
    """模型导出指南"""
    guide = """
# ===================== 模型导出方式 =====================

# 方式1: 保存LoRA适配器 (推荐, 体积小)
model.save_pretrained("./lora_adapter")
tokenizer.save_pretrained("./lora_adapter")
# 适配器大小约 50-200MB

# 方式2: 合并权重后保存 (用于部署)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model", safe_serialization=True)
tokenizer.save_pretrained("./merged_model")
# 合并后约 16GB (FP16)

# 方式3: 转换为GGUF格式 (用于llama.cpp本地推理)
# 使用llama.cpp的转换脚本:
# python convert_hf_to_gguf.py ./merged_model --outtype f16
# ./llama-quantize merged_model.gguf merged_model_q4.gguf q4_k_m
# 量化后约 4-5GB

# 方式4: 推送到Hugging Face Hub
model.push_to_hub("your-username/your-model-name")
tokenizer.push_to_hub("your-username/your-model-name")
"""
    return guide


# ===================== 使用示例 =====================

if __name__ == "__main__":
    print("=" * 60)
    print("  完整微调实战演示")
    print("=" * 60)

    # 1. 数据预处理演示
    print("\n--- 1. 数据预处理 ---")
    preprocessor = DataPreprocessor()

    # 创建模拟数据
    sample_data = [
        {"instruction": "什么是深度学习？", "input": "",
         "output": "深度学习是机器学习的一个分支，它使用多层神经网络来从数据中自动学习特征和模式。"},
        {"instruction": "解释LoRA微调", "input": "",
         "output": "LoRA通过添加低秩矩阵来实现参数高效微调，只需训练原始参数量的0.1%即可获得接近全量微调的效果。"},
        {"instruction": "短", "input": "", "output": "短"},  # 会被过滤
        {"instruction": "什么是深度学习？", "input": "",
         "output": "重复数据"},  # 会被去重
    ]

    processed = preprocessor.process(sample_data)
    train, val = preprocessor.split_data(processed, val_ratio=0.2)

    # 2. 硬件检测与配置推荐
    print("\n--- 2. 硬件检测 ---")
    config_gen = FineTuneConfigGenerator()
    gpu_info = config_gen.detect_gpu()
    print(f"GPU: {gpu_info['name']}")
    print(f"显存: {gpu_info['memory_gb']:.1f} GB")

    config = config_gen.recommend_config(
        gpu_info['memory_gb'], model_size_b=8
    )
    print(f"\n推荐配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # 3. 导出指南
    print("\n--- 3. 模型导出 ---")
    print(export_model_guide())
```

---

## DPO与RLHF对齐

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LLM对齐 (Alignment) 技术全景                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  对齐目标: 让LLM的输出符合人类偏好 (有用、无害、诚实)                   │
│                                                                         │
│  训练阶段:                                                              │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐                       │
│  │ 阶段1:    │───►│ 阶段2:    │───►│ 阶段3:    │                       │
│  │ 预训练    │    │ SFT       │    │ 对齐      │                       │
│  │ (无监督)  │    │ (监督微调)│    │ (RLHF/DPO)│                       │
│  └───────────┘    └───────────┘    └───────────┘                       │
│  海量文本         指令数据          人类偏好数据                         │
│                                                                         │
│  RLHF (Reinforcement Learning from Human Feedback):                    │
│  ┌───────────────────────────────────────────────────────┐             │
│  │                                                       │             │
│  │  步骤1: 收集人类偏好数据                              │             │
│  │  ┌──────────────────────────────────────┐             │             │
│  │  │  Prompt: "解释量子计算"               │             │             │
│  │  │  Response A: "量子计算是一种..."  ← 更好           │             │
│  │  │  Response B: "量子就是量子..."  ← 较差             │             │
│  │  │  人类标注: A > B                      │             │             │
│  │  └──────────────────────────────────────┘             │             │
│  │                                                       │             │
│  │  步骤2: 训练奖励模型 (Reward Model)                   │             │
│  │  ┌──────────────────────────────────────┐             │             │
│  │  │  (prompt, response) → 分数            │             │             │
│  │  │  RM(prompt, A) > RM(prompt, B)        │             │             │
│  │  └──────────────────────────────────────┘             │             │
│  │                                                       │             │
│  │  步骤3: PPO训练策略模型                               │             │
│  │  ┌──────────────────────────────────────┐             │             │
│  │  │  最大化奖励 + KL约束(不偏离原模型)    │             │             │
│  │  │  复杂! 需要4个模型同时运行             │             │             │
│  │  └──────────────────────────────────────┘             │             │
│  └───────────────────────────────────────────────────────┘             │
│                                                                         │
│  DPO (Direct Preference Optimization) -- 更简单的替代方案:             │
│  ┌───────────────────────────────────────────────────────┐             │
│  │                                                       │             │
│  │  核心思想: 跳过奖励模型, 直接从偏好数据优化策略       │             │
│  │                                                       │             │
│  │  RLHF: 偏好数据 → 奖励模型 → PPO → 策略模型          │             │
│  │  DPO:  偏好数据 → 直接优化策略模型  (简单!)           │             │
│  │                                                       │             │
│  │  DPO损失函数:                                         │             │
│  │  L = -log(sigmoid(beta * (                            │             │
│  │      log(pi(y_w|x)/pi_ref(y_w|x)) -                  │             │
│  │      log(pi(y_l|x)/pi_ref(y_l|x))                    │             │
│  │  )))                                                  │             │
│  │                                                       │             │
│  │  y_w: 偏好(winning)回复    y_l: 非偏好(losing)回复   │             │
│  │  pi: 当前模型    pi_ref: 参考模型(SFT后的模型)        │             │
│  │  beta: 温度参数, 控制偏离程度                          │             │
│  │                                                       │             │
│  │  优势:                                                │             │
│  │  - 不需要训练奖励模型                                 │             │
│  │  - 不需要PPO(强化学习), 只需简单的分类损失            │             │
│  │  - 训练更稳定, 超参数更少                             │             │
│  │  - 效果与RLHF相当甚至更好                             │             │
│  └───────────────────────────────────────────────────────┘             │
│                                                                         │
│  RLHF vs DPO 对比:                                                     │
│  ┌───────────────┬───────────────────┬───────────────────┐             │
│  │               │ RLHF (PPO)        │ DPO               │             │
│  ├───────────────┼───────────────────┼───────────────────┤             │
│  │ 需要模型数    │ 4 (策略+参考+RM   │ 2 (策略+参考)     │             │
│  │               │  +critic)         │                   │             │
│  │ 训练复杂度    │ 高 (RL不稳定)     │ 低 (类似SFT)      │             │
│  │ 显存需求      │ 极高              │ 中等              │             │
│  │ 超参数        │ 多且敏感          │ 少 (主要是beta)   │             │
│  │ 效果          │ 好                │ 相当或更好        │             │
│  │ 使用者        │ OpenAI, Anthropic │ Meta (Llama 3)    │             │
│  └───────────────┴───────────────────┴───────────────────┘             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 代码示例

```python
# DPO 对齐训练 -- 完整实现
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ===================== 1. DPO偏好数据格式 =====================

@dataclass
class PreferenceData:
    """DPO偏好数据"""
    prompt: str
    chosen: str      # 人类偏好的回复
    rejected: str     # 不偏好的回复


def create_preference_dataset():
    """创建示例偏好数据集"""
    data = [
        PreferenceData(
            prompt="什么是深度学习？",
            chosen="深度学习是机器学习的一个分支，使用多层神经网络从数据中"
                   "自动学习层次化的特征表示。它在图像识别、自然语言处理等"
                   "领域取得了突破性进展。",
            rejected="深度学习就是AI的一种。"
        ),
        PreferenceData(
            prompt="如何学习编程？",
            chosen="学习编程建议按以下步骤进行：1）选择一门语言（如Python）"
                   "从基础语法开始；2）通过实际项目练习，如做小工具或网站；"
                   "3）学习数据结构和算法；4）参与开源项目积累经验。",
            rejected="去网上搜一下就行了。"
        ),
    ]
    return data


# ===================== 2. DPO损失函数实现 =====================

class DPOLoss(nn.Module):
    """DPO (Direct Preference Optimization) 损失函数

    L_DPO = -log(sigmoid(beta * (
        log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x))
    )))
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,    # log P(y_w|x) 策略模型
        policy_rejected_logps: torch.Tensor,   # log P(y_l|x) 策略模型
        ref_chosen_logps: torch.Tensor,        # log P(y_w|x) 参考模型
        ref_rejected_logps: torch.Tensor,      # log P(y_l|x) 参考模型
    ) -> Dict[str, torch.Tensor]:
        """
        计算DPO损失

        所有输入形状: (batch,) -- 每个样本的对数概率
        """
        # 计算log ratio
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        # DPO损失
        logits = self.beta * (chosen_logratios - rejected_logratios)
        loss = -F.logsigmoid(logits).mean()

        # 统计信息
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        reward_margin = (chosen_rewards - rejected_rewards).mean()
        accuracy = (logits > 0).float().mean()

        return {
            "loss": loss,
            "chosen_rewards": chosen_rewards.mean(),
            "rejected_rewards": rejected_rewards.mean(),
            "reward_margin": reward_margin,
            "accuracy": accuracy,
        }


# ===================== 3. DPO训练器 =====================

class DPOTrainer:
    """简化的DPO训练器"""

    def __init__(self, policy_model, ref_model,
                 beta: float = 0.1, lr: float = 5e-7):
        self.policy = policy_model
        self.ref = ref_model
        self.criterion = DPOLoss(beta=beta)
        self.optimizer = torch.optim.AdamW(
            policy_model.parameters(), lr=lr
        )

        # 冻结参考模型
        self.ref.eval()
        for p in self.ref.parameters():
            p.requires_grad = False

    def compute_logps(self, model, input_ids, labels):
        """计算序列的对数概率"""
        with torch.set_grad_enabled(model.training):
            logits = model(input_ids)
            log_probs = F.log_softmax(logits, dim=-1)

            # 收集label位置的log概率
            per_token_logps = torch.gather(
                log_probs[:, :-1, :], 2,
                labels[:, 1:].unsqueeze(2)
            ).squeeze(2)

            # 求和得到序列级log概率
            return per_token_logps.sum(dim=-1)

    def train_step(self, chosen_ids, rejected_ids,
                   chosen_labels, rejected_labels):
        """DPO训练步"""
        self.policy.train()

        # 策略模型的log概率
        policy_chosen_logps = self.compute_logps(
            self.policy, chosen_ids, chosen_labels)
        policy_rejected_logps = self.compute_logps(
            self.policy, rejected_ids, rejected_labels)

        # 参考模型的log概率
        with torch.no_grad():
            ref_chosen_logps = self.compute_logps(
                self.ref, chosen_ids, chosen_labels)
            ref_rejected_logps = self.compute_logps(
                self.ref, rejected_ids, rejected_labels)

        # DPO损失
        metrics = self.criterion(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps
        )

        # 反向传播
        self.optimizer.zero_grad()
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), max_norm=1.0
        )
        self.optimizer.step()

        return {k: v.item() for k, v in metrics.items()}


# ===================== 4. 使用TRL库的DPO训练 =====================

def trl_dpo_training_template():
    """使用TRL库的DPO训练代码模板"""
    code = '''
# pip install trl peft transformers datasets

from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from datasets import load_dataset

# 加载模型
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRA配置
peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# 加载偏好数据集
# 格式: {"prompt": "...", "chosen": "...", "rejected": "..."}
dataset = load_dataset("json", data_files="preferences.json")

# DPO训练配置
dpo_config = DPOConfig(
    output_dir="./dpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    beta=0.1,                    # DPO温度参数
    bf16=True,
    logging_steps=10,
    optim="paged_adamw_32bit",
    max_length=1024,
    max_prompt_length=512,
)

# DPO训练器
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    peft_config=peft_config,
)

trainer.train()
trainer.save_model("./dpo_model")
'''
    return code


# ===================== 演示 =====================

if __name__ == "__main__":
    print("=" * 60)
    print("  DPO 对齐训练演示")
    print("=" * 60)

    # 简化模型演示
    vocab_size = 1000
    d_model = 128

    # 创建策略模型和参考模型
    policy = nn.Sequential(
        nn.Embedding(vocab_size, d_model),
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, vocab_size)
    )

    ref = nn.Sequential(
        nn.Embedding(vocab_size, d_model),
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, vocab_size)
    )
    # 参考模型初始化为策略模型的副本
    ref.load_state_dict(policy.state_dict())

    # 训练
    trainer = DPOTrainer(policy, ref, beta=0.1, lr=1e-4)

    print("\nDPO训练:")
    for step in range(10):
        chosen_ids = torch.randint(0, vocab_size, (4, 32))
        rejected_ids = torch.randint(0, vocab_size, (4, 32))

        metrics = trainer.train_step(
            chosen_ids, rejected_ids,
            chosen_ids, rejected_ids
        )

        if (step + 1) % 2 == 0:
            print(f"  Step {step+1}: "
                  f"loss={metrics['loss']:.4f}, "
                  f"acc={metrics['accuracy']:.2f}, "
                  f"margin={metrics['reward_margin']:.4f}")

    print("\n完整DPO训练代码模板:")
    print(trl_dpo_training_template())
```

---

## 多GPU与分布式微调

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    多GPU微调策略                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  根据模型大小和GPU数量选择策略:                                         │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  单GPU + QLoRA:                                              │      │
│  │  适用: 7B-13B模型, 1x RTX 3090/4090 (24GB)                  │      │
│  │  命令: python train.py                                       │      │
│  │  效率: 基准线                                                │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  多GPU + DDP + QLoRA:                                        │      │
│  │  适用: 7B-13B模型, 多卡并行加速                              │      │
│  │  命令: torchrun --nproc_per_node=4 train.py                  │      │
│  │  效率: 近线性加速 (4卡 ≈ 3.5x)                              │      │
│  │                                                              │      │
│  │  GPU0: 模型副本 + 数据1/4    GPU1: 模型副本 + 数据2/4       │      │
│  │  GPU2: 模型副本 + 数据3/4    GPU3: 模型副本 + 数据4/4       │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  DeepSpeed ZeRO + LoRA:                                      │      │
│  │  适用: 13B-70B模型, 多卡分片                                 │      │
│  │  命令: deepspeed --num_gpus=4 train.py --deepspeed ds.json   │      │
│  │                                                              │      │
│  │  ZeRO-1: 分片优化器状态                                     │      │
│  │  ZeRO-2: 分片优化器状态 + 梯度                              │      │
│  │  ZeRO-3: 分片优化器状态 + 梯度 + 模型参数 (最节省显存)      │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  FSDP + QLoRA:                                               │      │
│  │  适用: 70B+模型, PyTorch原生分布式                           │      │
│  │  命令: torchrun --nproc_per_node=8 train.py                  │      │
│  │  特点: 无需DeepSpeed依赖, PyTorch 2.0+原生支持              │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  各策略显存需求对比 (Llama-3.1-8B, QLoRA):                             │
│  ┌───────────────────┬───────────────┬──────────────────────┐         │
│  │ 策略              │ 每卡显存      │ 总显存               │         │
│  ├───────────────────┼───────────────┼──────────────────────┤         │
│  │ 单卡QLoRA         │ 6 GB          │ 6 GB                 │         │
│  │ 4卡DDP+QLoRA     │ 6 GB/卡       │ 24 GB                │         │
│  │ 4卡ZeRO-2+QLoRA  │ 4 GB/卡       │ 16 GB                │         │
│  │ 4卡ZeRO-3+QLoRA  │ 3 GB/卡       │ 12 GB                │         │
│  └───────────────────┴───────────────┴──────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 代码示例

```python
# 多GPU微调配置模板

# ===================== 1. Accelerate配置 =====================

ACCELERATE_CONFIG = '''
# accelerate_config.yaml
# 使用命令: accelerate launch --config_file accelerate_config.yaml train.py

compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 1
num_processes: 4          # GPU数量
mixed_precision: bf16
gpu_ids: 0,1,2,3

# DeepSpeed配置 (可选)
# deepspeed_config:
#   zero_stage: 2
#   offload_optimizer_device: cpu
'''

# ===================== 2. DeepSpeed ZeRO-2配置 =====================

DEEPSPEED_CONFIG = '''
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "overlap_comm": true,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
'''

# ===================== 3. 多GPU训练启动命令 =====================

LAUNCH_COMMANDS = '''
# 方法1: torchrun (PyTorch原生)
torchrun --nproc_per_node=4 train.py

# 方法2: accelerate (Hugging Face)
accelerate launch --num_processes=4 train.py

# 方法3: DeepSpeed
deepspeed --num_gpus=4 train.py \\
    --deepspeed ds_config.json

# 方法4: 使用Hugging Face Trainer内置支持
# 只需在TrainingArguments中设置:
#   deepspeed="ds_config.json"
# 或
#   fsdp="full_shard auto_wrap"
'''


def print_multi_gpu_guide():
    """打印多GPU微调指南"""
    print("=" * 60)
    print("  多GPU微调配置指南")
    print("=" * 60)

    print("\n--- Accelerate配置 ---")
    print(ACCELERATE_CONFIG)

    print("\n--- DeepSpeed ZeRO-2配置 ---")
    print(DEEPSPEED_CONFIG)

    print("\n--- 启动命令 ---")
    print(LAUNCH_COMMANDS)

    # 使用Trainer的多GPU微调
    print("\n--- TrainingArguments多GPU设置 ---")
    print('''
    training_args = TrainingArguments(
        output_dir="./output",

        # DeepSpeed方式
        deepspeed="ds_config.json",

        # 或 FSDP方式 (PyTorch原生)
        # fsdp="full_shard auto_wrap",
        # fsdp_config={
        #     "fsdp_min_num_params": 1000,
        #     "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
        # },

        # 通用设置
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        bf16=True,
        gradient_checkpointing=True,
    )
    ''')


if __name__ == "__main__":
    print_multi_gpu_guide()
```

---

## 总结

本教程涵盖了模型微调的核心内容:

1. **微调概述**: 详细对比了Full Fine-Tuning、LoRA、QLoRA、Adapter、Prefix Tuning等方法的原理、显存需求和适用场景。微调让我们用少量数据将通用大模型适配到特定领域。

2. **LoRA原理与实现**: 深入讲解了LoRA的低秩分解数学原理，从零实现了LoRALinear层，包含权重注入、训练和合并的完整流程。关键参数r控制秩大小，alpha控制缩放。

3. **QLoRA量化微调**: 通过NF4量化、双重量化和分页优化器三大技术，将7B模型的显存需求从16GB降至6GB。模拟了NF4量化的完整过程，展示了压缩比和精度损失。

4. **PEFT框架使用**: Hugging Face的PEFT框架提供了统一的参数高效微调API，支持LoRA、AdaLoRA、Prefix Tuning等多种方法。支持多适配器热切换，一个基座模型服务多个任务。

5. **Hugging Face Trainer**: SFTTrainer封装了完整的监督微调训练循环，支持混合精度、梯度累积、分布式训练等高级功能。通过TrainingArguments精细控制训练过程。

6. **完整微调实战**: 端到端流程覆盖数据预处理（清洗/验证/去重）、硬件检测与配置推荐、训练执行、模型评估和多种导出格式（LoRA适配器/合并权重/GGUF）。

7. **DPO与RLHF对齐**: 对比了RLHF（PPO）和DPO两种对齐方法。RLHF需要训练奖励模型+PPO强化学习，流程复杂但效果稳定；DPO直接用偏好数据优化策略模型，无需奖励模型，训练更简单高效。实现了完整的DPO损失函数和训练器，以及基于TRL库的快速对齐方案。

8. **多GPU与分布式微调**: 对比了单GPU QLoRA、多GPU DDP+QLoRA、DeepSpeed ZeRO和FSDP四种策略。提供了Accelerate和DeepSpeed的完整配置模板，以及多GPU环境下的启动命令和TrainingArguments关键参数。

## 最佳实践

1. **数据质量 > 数据数量**: 500条高质量数据的效果往往优于5000条低质量数据
2. **从小r开始**: LoRA的r从8开始尝试，效果不佳再增大到16/32/64
3. **target_modules覆盖所有线性层**: 不仅是attention的q/v，还包括gate/up/down_proj
4. **使用QLoRA节省显存**: 对于7B-13B模型，QLoRA在消费级GPU上即可微调
5. **学习率建议**: LoRA使用2e-4到1e-4，QLoRA使用2e-4
6. **训练轮数**: 微调通常1-3个epoch即可，过多容易过拟合
7. **梯度检查点**: 始终开启gradient_checkpointing以节省显存
8. **评估**: 除了自动指标（loss/perplexity），务必进行人工评估
9. **对齐优先用DPO**: 资源有限时优先尝试DPO，比RLHF简单且显存需求低一半
10. **偏好数据质量**: DPO/RLHF的效果高度依赖偏好数据质量，确保chosen/rejected差异明显
11. **多GPU从DDP开始**: 多卡训练首选DDP+QLoRA，模型放不下再用DeepSpeed ZeRO-3或FSDP
12. **分布式调试**: 多GPU训练前先用单GPU验证代码正确性，避免分布式调试困难

## 参考资源

- [LoRA 原始论文](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
- [QLoRA 原始论文](https://arxiv.org/abs/2305.14314) - Dettmers et al., 2023
- [DPO 原始论文](https://arxiv.org/abs/2305.18290) - Rafailov et al., 2023
- [RLHF/InstructGPT 论文](https://arxiv.org/abs/2203.02155) - Ouyang et al., 2022
- [PEFT 官方文档](https://huggingface.co/docs/peft)
- [TRL 官方文档](https://huggingface.co/docs/trl)
- [Hugging Face Trainer 文档](https://huggingface.co/docs/transformers/main_classes/trainer)
- [DeepSpeed 官方文档](https://www.deepspeed.ai/)
- [Accelerate 官方文档](https://huggingface.co/docs/accelerate)
- [Axolotl 微调框架](https://github.com/OpenAccess-AI-Collective/axolotl)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - 一站式微调框架

---

**文件大小目标**: 30-35KB
**创建时间**: 2024-01-15
**最后更新**: 2024-01-15
