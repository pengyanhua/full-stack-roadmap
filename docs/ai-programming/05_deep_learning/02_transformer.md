# Transformer架构详解

## 目录
1. [Transformer概述](#transformer概述)
2. [Self-Attention](#self-attention)
3. [Multi-Head Attention](#multi-head-attention)
4. [Position Encoding](#position-encoding)
5. [完整实现](#完整实现)
6. [训练和推理](#训练和推理)

---

## Transformer概述

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Transformer 完整架构图                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│         Encoder (编码器)              Decoder (解码器)               │
│    ┌─────────────────────┐     ┌─────────────────────┐             │
│    │                     │     │                     │             │
│    │  ┌───────────────┐  │     │  ┌───────────────┐  │             │
│    │  │ Feed Forward  │  │     │  │ Feed Forward  │  │             │
│    │  │ Network (FFN) │  │     │  │ Network (FFN) │  │             │
│    │  └───────┬───────┘  │     │  └───────┬───────┘  │             │
│    │      Add & Norm    │     │      Add & Norm    │             │
│    │          │          │     │          │          │             │
│    │  ┌───────────────┐  │     │  ┌───────────────┐  │             │
│    │  │  Multi-Head   │  │     │  │  Cross        │  │             │
│    │  │  Self-Attn    │  │  ┌──┼──│  Attention    │  │             │
│    │  └───────┬───────┘  │  │  │  └───────┬───────┘  │             │
│    │      Add & Norm    │  │  │      Add & Norm    │             │
│    │          │          │  │  │          │          │             │
│ Nx │          │          │  │  │  ┌───────────────┐  │ Nx          │
│    │          │          │──┘  │  │  Masked       │  │             │
│    │          │          │     │  │  Multi-Head   │  │             │
│    │          │          │     │  │  Self-Attn    │  │             │
│    │          │          │     │  └───────┬───────┘  │             │
│    │          │          │     │      Add & Norm    │             │
│    │          │          │     │          │          │             │
│    └──────────┼──────────┘     └──────────┼──────────┘             │
│               │                           │                        │
│    ┌──────────┴──────────┐     ┌──────────┴──────────┐             │
│    │ Positional Encoding │     │ Positional Encoding │             │
│    └──────────┬──────────┘     └──────────┬──────────┘             │
│    ┌──────────┴──────────┐     ┌──────────┴──────────┐             │
│    │  Input Embedding    │     │ Output Embedding    │             │
│    └──────────┬──────────┘     └──────────┬──────────┘             │
│               │                           │                        │
│          Input Tokens              Output Tokens (shifted)         │
│         "我 爱 学习"               "<SOS> I love"                  │
│                                                                     │
│                              ┌──────────┐                          │
│                              │ Linear   │                          │
│                              │ Softmax  │                          │
│                              └──────────┘                          │
│                              Output: "I love learning"             │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**Transformer** 是2017年Google在论文"Attention Is All You Need"中提出的革命性架构。它完全基于注意力机制（Attention），摒弃了传统的循环结构（RNN/LSTM），实现了真正的并行计算，极大提升了训练速度和模型性能。

**Transformer的里程碑意义：**
- **BERT**（2018）：基于Transformer Encoder，革新了NLP预训练
- **GPT**系列（2018-2024）：基于Transformer Decoder，推动了大语言模型发展
- **Vision Transformer**（2020）：将Transformer扩展到计算机视觉
- **ChatGPT/GPT-4/Claude**：基于Transformer的大规模语言模型

**Transformer核心组件：**

| 组件 | 说明 | 作用 |
|------|------|------|
| **Self-Attention** | 自注意力机制 | 捕获序列内部的依赖关系 |
| **Multi-Head Attention** | 多头注意力 | 从多个子空间捕获不同模式 |
| **Position Encoding** | 位置编码 | 注入位置信息（因为Attention没有位置概念） |
| **Feed Forward Network** | 前馈网络 | 非线性变换，增强表达能力 |
| **Layer Normalization** | 层归一化 | 稳定训练过程 |
| **Residual Connection** | 残差连接 | 缓解梯度消失，帮助深层网络训练 |

**Encoder vs Decoder 对比：**

| 特性 | Encoder | Decoder |
|------|---------|---------|
| 注意力类型 | 双向Self-Attention | Masked Self-Attention + Cross-Attention |
| 信息流 | 可以看到所有位置 | 只能看到当前及之前的位置 |
| 典型应用 | BERT, 分类, NER | GPT, 文本生成, 翻译 |
| 输入 | 完整序列 | 自回归（逐token生成） |

### 代码示例

```python
# Transformer概述 - 整体框架
import torch
import torch.nn as nn
import math
from typing import Optional


class TransformerConfig:
    """Transformer配置"""
    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 512,      # 模型维度
        n_heads: int = 8,        # 注意力头数
        n_layers: int = 6,       # 编码器/解码器层数
        d_ff: int = 2048,        # 前馈网络维度
        max_seq_len: int = 512,  # 最大序列长度
        dropout: float = 0.1,    # Dropout比率
        pad_idx: int = 0,        # 填充token的ID
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.pad_idx = pad_idx

        # 验证
        assert d_model % n_heads == 0, \
            "d_model必须能被n_heads整除"
        self.d_k = d_model // n_heads  # 每个头的维度


# 使用示例
if __name__ == "__main__":
    config = TransformerConfig()
    print(f"Transformer配置:")
    print(f"  模型维度 d_model: {config.d_model}")
    print(f"  注意力头数: {config.n_heads}")
    print(f"  每个头的维度 d_k: {config.d_k}")
    print(f"  编码器层数: {config.n_layers}")
    print(f"  前馈网络维度 d_ff: {config.d_ff}")
    print(f"  词汇表大小: {config.vocab_size}")
```

---

## Self-Attention

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Scaled Dot-Product Attention                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  公式: Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V             │
│                                                                     │
│  输入:  X = [x1, x2, x3]  (序列长度=3, 维度=d_model)               │
│                                                                     │
│  步骤1: 生成Q, K, V                                                │
│  ┌───┐   ┌────┐   ┌───┐                                           │
│  │ X │──►│ Wq │──►│ Q │  Q = X · Wq                               │
│  │   │   └────┘   └───┘                                           │
│  │   │   ┌────┐   ┌───┐                                           │
│  │   │──►│ Wk │──►│ K │  K = X · Wk                               │
│  │   │   └────┘   └───┘                                           │
│  │   │   ┌────┐   ┌───┐                                           │
│  │   │──►│ Wv │──►│ V │  V = X · Wv                               │
│  └───┘   └────┘   └───┘                                           │
│                                                                     │
│  步骤2: 计算注意力分数                                               │
│  ┌───┐   ┌─────┐   ┌─────────────┐                                │
│  │ Q │──►│     │   │ 0.8 0.1 0.1 │  注意力权重                    │
│  └───┘   │ Q·K^T│──►│ 0.2 0.7 0.1 │  (经过softmax)               │
│  ┌───┐   │ /√d_k│   │ 0.1 0.2 0.7 │                              │
│  │ K │──►│     │   └──────┬──────┘                                │
│  └───┘   └─────┘          │                                        │
│                            ▼                                        │
│  步骤3: 加权求和                                                    │
│  ┌───────────────┐   ┌───┐   ┌──────┐                             │
│  │ Attn Weights  │ × │ V │ = │Output│                             │
│  │ (seq, seq)    │   │   │   │      │                             │
│  └───────────────┘   └───┘   └──────┘                             │
│                                                                     │
│  直觉理解:                                                          │
│  "The cat sat on the mat"                                          │
│   ┌──────────────────┐                                             │
│  "cat" 对 "The" 关注 0.1                                           │
│  "cat" 对 "cat" 关注 0.3  ← 自己关注自己                           │
│  "cat" 对 "sat" 关注 0.4  ← 动词与主语强相关                       │
│  "cat" 对 "on"  关注 0.05                                          │
│  "cat" 对 "the" 关注 0.05                                          │
│  "cat" 对 "mat" 关注 0.1                                           │
│   └──────────────────┘                                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**Self-Attention（自注意力）** 是Transformer的核心创新。它允许序列中的每个位置都可以"关注"序列中的所有其他位置，从而捕获任意距离的依赖关系。

**Q/K/V的直觉理解：**

- **Query (Q)**：当前位置的"提问"——"我应该关注谁？"
- **Key (K)**：每个位置的"标签"——"我的特征是什么？"
- **Value (V)**：每个位置的"内容"——"如果你关注我，这是我的信息"

**计算流程详解：**

1. **线性投影**：将输入 X 通过三个不同的权重矩阵投影为 Q, K, V
2. **计算相似度**：Q 和 K 做点积，得到注意力分数
3. **缩放**：除以 sqrt(d_k) 防止点积值过大导致 softmax 梯度消失
4. **Softmax归一化**：将分数转为概率分布（权重之和为1）
5. **加权求和**：用注意力权重对 V 做加权平均

**为什么需要缩放（Scale）？**

当 d_k 较大时，Q 和 K 的点积结果会很大，导致 softmax 输出接近 one-hot 向量（梯度接近0）。除以 sqrt(d_k) 可以使点积结果保持合理的方差。

### 代码示例

```python
# Self-Attention - 完整实现
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: nn.Dropout = None
) -> tuple:
    """
    Scaled Dot-Product Attention

    Args:
        query: (batch, seq_len, d_k) 或 (batch, n_heads, seq_len, d_k)
        key:   同上
        value: 同上
        mask:  (batch, 1, 1, seq_len) 或 (batch, 1, seq_len, seq_len)
        dropout: Dropout层

    Returns:
        output: 注意力输出
        attn_weights: 注意力权重矩阵
    """
    d_k = query.size(-1)

    # 步骤1: 计算注意力分数 Q·K^T / √d_k
    # (batch, ..., seq_q, d_k) @ (batch, ..., d_k, seq_k)
    # = (batch, ..., seq_q, seq_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 步骤2: 应用mask (用于padding或causal mask)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 步骤3: Softmax归一化
    attn_weights = F.softmax(scores, dim=-1)

    # 步骤4: 应用dropout
    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # 步骤5: 加权求和 V
    # (batch, ..., seq_q, seq_k) @ (batch, ..., seq_k, d_v)
    # = (batch, ..., seq_q, d_v)
    output = torch.matmul(attn_weights, value)

    return output, attn_weights


class SelfAttention(nn.Module):
    """单头Self-Attention"""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Q, K, V 的线性投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len) 可选

        Returns:
            output: (batch, seq_len, d_model)
        """
        # 生成 Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # 计算注意力
        output, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask, self.dropout
        )

        return output


# 演示
def self_attention_demo():
    """Self-Attention演示"""
    print("=" * 60)
    print("Self-Attention 演示")
    print("=" * 60)

    batch_size = 2
    seq_len = 4
    d_model = 8

    # 模拟输入 (batch=2, seq_len=4, d_model=8)
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入形状: {x.shape}")

    # 手动计算注意力
    Q = x  # 简化：Q=K=V=X
    K = x
    V = x

    # 计算分数
    d_k = d_model
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    print(f"\n注意力分数矩阵形状: {scores.shape}")
    print(f"第一个样本的注意力分数:\n{scores[0]}")

    # Softmax
    attn_weights = F.softmax(scores, dim=-1)
    print(f"\n注意力权重 (softmax后):\n{attn_weights[0]}")
    print(f"权重行和: {attn_weights[0].sum(dim=-1)}")  # 每行和为1

    # 使用SelfAttention模块
    self_attn = SelfAttention(d_model)
    output = self_attn(x)
    print(f"\nSelfAttention输出形状: {output.shape}")


if __name__ == "__main__":
    self_attention_demo()
```

---

## Multi-Head Attention

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Multi-Head Attention 架构                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  d_model=512, n_heads=8, d_k=64                                    │
│                                                                     │
│  输入 X: (batch, seq_len, 512)                                     │
│       │                                                             │
│       ├──► Wq ──► Q: (batch, seq_len, 512)                        │
│       ├──► Wk ──► K: (batch, seq_len, 512)                        │
│       └──► Wv ──► V: (batch, seq_len, 512)                        │
│                    │                                                │
│                    ▼  reshape + transpose                           │
│       ┌────────────┬────────────┬─── ... ──┬────────────┐          │
│       │  Head 1    │  Head 2    │          │  Head 8    │          │
│       │ Q1 K1 V1  │ Q2 K2 V2  │          │ Q8 K8 V8  │          │
│       │(batch,     │(batch,     │          │(batch,     │          │
│       │ seq, 64)  │ seq, 64)  │          │ seq, 64)  │          │
│       │            │            │          │            │          │
│       │ Attention  │ Attention  │          │ Attention  │          │
│       │ (seq,64)  │ (seq,64)  │          │ (seq,64)  │          │
│       └─────┬──────┘─────┬──────┘─── ... ──┘─────┬──────┘          │
│             │            │                       │                  │
│             └────────────┴───────── Concat ──────┘                  │
│                                      │                              │
│                              (batch, seq, 512)                     │
│                                      │                              │
│                                      ▼                              │
│                               ┌────────────┐                       │
│                               │  Wo线性层   │                       │
│                               │ (512, 512) │                       │
│                               └─────┬──────┘                       │
│                                     │                              │
│                              Output: (batch, seq, 512)             │
│                                                                     │
│  为什么需要多头？                                                    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐              │
│  │ Head 1  │  │ Head 2  │  │ Head 3  │  │ Head 4  │              │
│  │ 语法关系│  │ 指代关系│  │ 语义相似│  │ 位置模式│              │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘              │
│  每个头学习不同类型的注意力模式                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**Multi-Head Attention（多头注意力）** 是将注意力机制并行执行多次，每次使用不同的投影参数。这样模型可以同时从不同的"角度"关注信息。

**多头注意力的优势：**

1. **多子空间表示**：每个头在不同的子空间中学习注意力模式
2. **丰富的特征捕获**：不同的头可以关注语法、语义、位置等不同方面
3. **计算效率**：虽然有多个头，但每个头的维度减小(d_k = d_model / n_heads)，总计算量与单头相同

**参数对照表：**

| 参数 | 典型值 | 说明 |
|------|--------|------|
| d_model | 512 | 模型总维度 |
| n_heads | 8 | 注意力头数 |
| d_k = d_v | 64 | 每个头的维度 (512/8=64) |
| 输入形状 | (batch, seq, 512) | 批次 x 序列长度 x 维度 |
| 输出形状 | (batch, seq, 512) | 与输入相同 |

### 代码示例

```python
# Multi-Head Attention - 完整实现
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model: int, n_heads: int,
                 dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, \
            f"d_model({d_model}) 必须能被 n_heads({n_heads}) 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q, K, V, O 的线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None  # 保存注意力权重用于可视化

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """将最后一个维度分割为 (n_heads, d_k)

        (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, n_heads, seq_len, d_k)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """合并多头输出

        (batch, n_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2)  # (batch, seq_len, n_heads, d_k)
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_q, d_model)
            key:   (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask:  (batch, 1, 1, seq_k) 或 (batch, 1, seq_q, seq_k)

        Returns:
            output: (batch, seq_q, d_model)
        """
        batch_size = query.size(0)

        # 1. 线性投影
        Q = self.W_q(query)  # (batch, seq_q, d_model)
        K = self.W_k(key)    # (batch, seq_k, d_model)
        V = self.W_v(value)  # (batch, seq_k, d_model)

        # 2. 分割为多个头
        Q = self.split_heads(Q)  # (batch, n_heads, seq_q, d_k)
        K = self.split_heads(K)  # (batch, n_heads, seq_k, d_k)
        V = self.split_heads(V)  # (batch, n_heads, seq_k, d_k)

        # 3. Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        self.attn_weights = attn_weights  # 保存用于可视化

        attn_weights = self.dropout(attn_weights)

        # (batch, n_heads, seq_q, d_k)
        context = torch.matmul(attn_weights, V)

        # 4. 合并多头
        context = self.combine_heads(context)  # (batch, seq_q, d_model)

        # 5. 输出投影
        output = self.W_o(context)  # (batch, seq_q, d_model)

        return output


# Masked Multi-Head Attention (用于Decoder)
def create_causal_mask(seq_len: int) -> torch.Tensor:
    """创建因果掩码 (下三角矩阵)

    防止Decoder在位置i看到位置i之后的信息
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


def create_padding_mask(seq: torch.Tensor,
                        pad_idx: int = 0) -> torch.Tensor:
    """创建padding掩码

    将padding位置标记为0(被忽略)
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask  # (batch, 1, 1, seq_len)


# 演示
def multi_head_attention_demo():
    """Multi-Head Attention演示"""
    print("=" * 60)
    print("Multi-Head Attention 演示")
    print("=" * 60)

    batch_size = 2
    seq_len = 6
    d_model = 512
    n_heads = 8

    # 创建模型
    mha = MultiHeadAttention(d_model, n_heads)

    # 模拟输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入形状: {x.shape}")

    # Self-Attention (Q=K=V=X)
    output = mha(x, x, x)
    print(f"Self-Attention输出形状: {output.shape}")
    print(f"注意力权重形状: {mha.attn_weights.shape}")

    # 带因果掩码的Self-Attention (Decoder)
    causal_mask = create_causal_mask(seq_len)
    output_masked = mha(x, x, x, mask=causal_mask)
    print(f"\nMasked Self-Attention输出形状: {output_masked.shape}")
    print(f"因果掩码形状: {causal_mask.shape}")
    print(f"因果掩码:\n{causal_mask[0, 0]}")

    # Padding掩码
    token_ids = torch.tensor([[1, 2, 3, 4, 0, 0],
                               [1, 2, 3, 0, 0, 0]])
    pad_mask = create_padding_mask(token_ids)
    print(f"\nPadding掩码形状: {pad_mask.shape}")
    print(f"Padding掩码:\n{pad_mask[0, 0, 0]}")

    # 参数量
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\nMulti-Head Attention 参数量: {total_params:,}")


if __name__ == "__main__":
    multi_head_attention_demo()
```

---

## Position Encoding

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Position Encoding (位置编码)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  问题: Self-Attention是排列不变的 (permutation invariant)            │
│  "猫 吃 鱼" 和 "鱼 吃 猫" 的注意力计算结果相同!                     │
│  → 需要显式注入位置信息                                              │
│                                                                     │
│  正弦位置编码公式:                                                   │
│  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))                    │
│  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))                    │
│                                                                     │
│  pos: 位置索引 (0, 1, 2, ...)                                      │
│  i:   维度索引 (0, 1, 2, ..., d_model/2 - 1)                      │
│                                                                     │
│  可视化 (d_model=8):                                                │
│  pos=0: [sin(0), cos(0), sin(0),    cos(0),    ...]               │
│  pos=1: [sin(1), cos(1), sin(0.01), cos(0.01), ...]               │
│  pos=2: [sin(2), cos(2), sin(0.02), cos(0.02), ...]               │
│                                                                     │
│  ┌───────────────────────────────────────┐                         │
│  │  维度 0,1 (高频):  ~~~~ 短波         │                         │
│  │  维度 2,3:         ~~   中波         │                         │
│  │  维度 4,5:         ~    长波         │                         │
│  │  维度 6,7 (低频):  —    超长波       │                         │
│  └───────────────────────────────────────┘                         │
│  不同维度使用不同频率的正弦/余弦，形成独特的位置"指纹"               │
│                                                                     │
│  最终输入 = Token Embedding + Position Encoding                     │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐               │
│  │ Embedding│ + │ Pos Encoding │ = │ 模型输入     │               │
│  │ (学习的) │   │ (固定的/学习)│   │              │               │
│  └──────────┘   └──────────────┘   └──────────────┘               │
│                                                                     │
│  位置编码方法对比:                                                   │
│  ┌────────────┬──────────────────┬──────────────────┐              │
│  │ 正弦编码   │ 可学习编码        │ RoPE            │              │
│  │ (原始论文) │ (BERT/GPT)       │ (LLaMA/GPT-NeoX)│              │
│  │ 固定、不学习│ 随模型训练更新    │ 旋转位置编码    │              │
│  │ 可外推     │ 受限于训练长度    │ 更好的外推能力   │              │
│  └────────────┴──────────────────┴──────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**位置编码（Position Encoding）** 解决了Transformer中注意力机制对位置不敏感的问题。由于Self-Attention的计算只涉及元素间的相似度，不包含任何位置信息，因此需要显式地将位置信息编码到输入中。

**三种主要的位置编码方法：**

| 方法 | 原理 | 优点 | 缺点 | 使用者 |
|------|------|------|------|--------|
| **正弦编码** | 固定的sin/cos函数 | 可外推到更长序列 | 表达能力有限 | 原始Transformer |
| **可学习编码** | 将位置作为可训练参数 | 灵活、表达能力强 | 不能外推 | BERT, GPT-2 |
| **RoPE** | 旋转位置编码 | 相对位置感知、好外推 | 实现较复杂 | LLaMA, Qwen |

### 代码示例

```python
# Position Encoding - 完整实现
import torch
import torch.nn as nn
import math


class SinusoidalPositionEncoding(nn.Module):
    """正弦位置编码 (原始Transformer论文)"""

    def __init__(self, d_model: int, max_seq_len: int = 5000,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预计算位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()

        # 计算频率分母: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        # 偶数维度用sin，奇数维度用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加batch维度: (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)

        # 注册为buffer(不参与训练，但会随模型保存)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) - Token Embedding

        Returns:
            x + position_encoding: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnablePositionEncoding(nn.Module):
    """可学习位置编码 (BERT/GPT-2 风格)"""

    def __init__(self, d_model: int, max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        pos_encoding = self.position_embedding(positions)
        x = x + pos_encoding
        return self.dropout(x)


class RotaryPositionEncoding(nn.Module):
    """旋转位置编码 RoPE (LLaMA 风格)

    对Q和K施加旋转变换，使注意力分数自然包含相对位置信息。
    """

    def __init__(self, d_model: int, max_seq_len: int = 4096,
                 base: float = 10000.0):
        super().__init__()
        self.d_model = d_model

        # 计算旋转频率
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_model, 2).float() / d_model)
        )
        self.register_buffer('inv_freq', inv_freq)

        # 预计算cos和sin
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # (seq_len, d_model/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, d_model)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0))

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """将张量的后半部分取反并与前半部分交换"""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor,
                k: torch.Tensor) -> tuple:
        """
        对Q和K施加旋转位置编码

        Args:
            q: (batch, n_heads, seq_len, d_k)
            k: (batch, n_heads, seq_len, d_k)
        """
        seq_len = q.size(2)
        cos = self.cos_cached[:, :seq_len, :]
        sin = self.sin_cached[:, :seq_len, :]

        # 广播到多头维度
        cos = cos.unsqueeze(1)  # (1, 1, seq_len, d_model)
        sin = sin.unsqueeze(1)

        # RoPE变换: x * cos + rotate_half(x) * sin
        q_embed = q * cos + self.rotate_half(q) * sin
        k_embed = k * cos + self.rotate_half(k) * sin

        return q_embed, k_embed


# 演示
def position_encoding_demo():
    """位置编码演示"""
    print("=" * 60)
    print("Position Encoding 演示")
    print("=" * 60)

    d_model = 512
    seq_len = 10
    batch_size = 2

    # 模拟Token Embedding
    x = torch.randn(batch_size, seq_len, d_model)

    # 1. 正弦位置编码
    sin_pe = SinusoidalPositionEncoding(d_model)
    out_sin = sin_pe(x)
    print(f"正弦编码输出形状: {out_sin.shape}")

    # 查看PE矩阵的前几个值
    pe_matrix = sin_pe.pe[0, :5, :8]
    print(f"\nPE矩阵 (前5个位置, 前8个维度):\n{pe_matrix}")

    # 2. 可学习位置编码
    learn_pe = LearnablePositionEncoding(d_model)
    out_learn = learn_pe(x)
    print(f"\n可学习编码输出形状: {out_learn.shape}")

    # 3. RoPE
    rope = RotaryPositionEncoding(64)  # d_k = 64
    q = torch.randn(batch_size, 8, seq_len, 64)
    k = torch.randn(batch_size, 8, seq_len, 64)
    q_rot, k_rot = rope(q, k)
    print(f"\nRoPE: Q形状 {q_rot.shape}, K形状 {k_rot.shape}")


if __name__ == "__main__":
    position_encoding_demo()
```

---

## 完整实现

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    完整Transformer实现结构                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Transformer                                                        │
│  ├── Encoder                                                        │
│  │   ├── TokenEmbedding                                            │
│  │   ├── PositionEncoding                                          │
│  │   └── EncoderLayer × N                                          │
│  │       ├── MultiHeadAttention (Self)                             │
│  │       ├── LayerNorm + Residual                                  │
│  │       ├── FeedForward                                           │
│  │       └── LayerNorm + Residual                                  │
│  │                                                                  │
│  ├── Decoder                                                        │
│  │   ├── TokenEmbedding                                            │
│  │   ├── PositionEncoding                                          │
│  │   └── DecoderLayer × N                                          │
│  │       ├── Masked MultiHeadAttention (Self)                      │
│  │       ├── LayerNorm + Residual                                  │
│  │       ├── MultiHeadAttention (Cross, K/V from Encoder)          │
│  │       ├── LayerNorm + Residual                                  │
│  │       ├── FeedForward                                           │
│  │       └── LayerNorm + Residual                                  │
│  │                                                                  │
│  └── Output                                                        │
│      └── Linear (d_model → vocab_size)                             │
│                                                                     │
│  Encoder-Only: BERT (分类、NER)                                     │
│  Decoder-Only: GPT (文本生成) ← 当前主流LLM架构                     │
│  Encoder-Decoder: T5, BART (翻译、摘要)                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

本节实现完整的Transformer模型，包含Encoder和Decoder两部分。代码严格按照"Attention Is All You Need"论文的架构实现。

**各组件的连接方式：**

1. **EncoderLayer**：Self-Attention → Add&Norm → FFN → Add&Norm
2. **DecoderLayer**：Masked Self-Attention → Add&Norm → Cross-Attention → Add&Norm → FFN → Add&Norm
3. **整体流程**：Input → Embedding+PE → Encoder Stack → Decoder Stack → Linear → Softmax

### 代码示例

```python
# 完整Transformer实现 (~200行核心代码)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ===================== 基础组件 =====================

class FeedForward(nn.Module):
    """前馈网络 (Position-wise Feed-Forward Network)

    FFN(x) = max(0, xW1 + b1)W2 + b2
    即两个线性层 + ReLU激活
    """

    def __init__(self, d_model: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiHeadAttention(nn.Module):
    """多头注意力 (复用之前的实现)"""

    def __init__(self, d_model: int, n_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(
            batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(
            batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(
            batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)

        return self.W_o(context)


# ===================== Encoder =====================

class EncoderLayer(nn.Module):
    """Transformer Encoder 层"""

    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-Attention + Residual + LayerNorm
        attn_output = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed Forward + Residual + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder (N层堆叠)"""

    def __init__(self, vocab_size: int, d_model: int,
                 n_heads: int, n_layers: int, d_ff: int,
                 max_seq_len: int = 512, dropout: float = 0.1,
                 pad_idx: int = 0):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model,
                                       padding_idx=pad_idx)
        self.pos_encoding = SinusoidalPositionEncoding(
            d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: (batch, src_len) token indices
            src_mask: (batch, 1, 1, src_len)
        """
        # Embedding + Position Encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # N层Encoder
        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x)


# ===================== Decoder =====================

class DecoderLayer(nn.Module):
    """Transformer Decoder 层"""

    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                enc_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Masked Self-Attention + Residual + LayerNorm
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))

        # Cross-Attention (Q from decoder, K/V from encoder)
        cross_attn_output = self.cross_attn(x, enc_output,
                                             enc_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # Feed Forward + Residual + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x


class TransformerDecoder(nn.Module):
    """Transformer Decoder (N层堆叠)"""

    def __init__(self, vocab_size: int, d_model: int,
                 n_heads: int, n_layers: int, d_ff: int,
                 max_seq_len: int = 512, dropout: float = 0.1,
                 pad_idx: int = 0):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model,
                                       padding_idx=pad_idx)
        self.pos_encoding = SinusoidalPositionEncoding(
            d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor,
                enc_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt: (batch, tgt_len) target token indices
            enc_output: (batch, src_len, d_model)
            tgt_mask: (batch, 1, tgt_len, tgt_len) causal mask
            src_mask: (batch, 1, 1, src_len) padding mask
        """
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)

        return self.norm(x)


# ===================== 完整Transformer =====================

class SinusoidalPositionEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Transformer(nn.Module):
    """完整的Transformer模型 (Encoder-Decoder)"""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        super().__init__()

        self.pad_idx = pad_idx

        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, n_heads, n_layers,
            d_ff, max_seq_len, dropout, pad_idx
        )

        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, n_heads, n_layers,
            d_ff, max_seq_len, dropout, pad_idx
        )

        # 输出投影层
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # 初始化参数
        self._init_parameters()

    def _init_parameters(self):
        """Xavier初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """创建源序列padding mask"""
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def create_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """创建目标序列的组合mask (padding + causal)"""
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_causal_mask = torch.tril(
            torch.ones(tgt_len, tgt_len, device=tgt.device)
        ).unsqueeze(0).unsqueeze(0)
        tgt_mask = tgt_pad_mask & (tgt_causal_mask.bool())
        return tgt_mask

    def forward(self, src: torch.Tensor,
                tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: (batch, src_len) 源序列token ID
            tgt: (batch, tgt_len) 目标序列token ID

        Returns:
            logits: (batch, tgt_len, tgt_vocab_size)
        """
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)

        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        logits = self.output_proj(dec_output)

        return logits


# 演示
def transformer_demo():
    """完整Transformer演示"""
    print("=" * 60)
    print("完整Transformer 演示")
    print("=" * 60)

    # 配置
    src_vocab = 5000
    tgt_vocab = 5000
    d_model = 256
    n_heads = 8
    n_layers = 3
    d_ff = 512

    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=128,
        dropout=0.1
    )

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 模拟输入
    batch_size = 4
    src_len = 20
    tgt_len = 15

    src = torch.randint(1, src_vocab, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab, (batch_size, tgt_len))

    # 添加一些padding
    src[0, 15:] = 0
    tgt[0, 12:] = 0

    print(f"\n源序列形状: {src.shape}")
    print(f"目标序列形状: {tgt.shape}")

    # 前向传播
    logits = model(src, tgt)
    print(f"输出logits形状: {logits.shape}")
    # (batch, tgt_len, tgt_vocab_size)

    # 预测下一个token
    predictions = logits.argmax(dim=-1)
    print(f"预测token形状: {predictions.shape}")


if __name__ == "__main__":
    transformer_demo()
```

---

## 训练和推理

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Transformer 训练和推理流程                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  训练 (Teacher Forcing):                                            │
│  ┌──────────────────────────────────────────────────┐              │
│  │ 输入:  src = "我 爱 学习"                          │              │
│  │        tgt = "<SOS> I love learning"  (右移一位)   │              │
│  │ 标签:  labels = "I love learning <EOS>"            │              │
│  │                                                    │              │
│  │ Encoder(src) → enc_output                         │              │
│  │ Decoder(tgt, enc_output) → logits                 │              │
│  │ Loss = CrossEntropy(logits, labels)               │              │
│  └──────────────────────────────────────────────────┘              │
│                                                                     │
│  推理 (Autoregressive):                                             │
│  ┌──────────────────────────────────────────────────┐              │
│  │ Step 1: tgt = [<SOS>]                             │              │
│  │         → predict "I"                             │              │
│  │ Step 2: tgt = [<SOS>, I]                          │              │
│  │         → predict "love"                          │              │
│  │ Step 3: tgt = [<SOS>, I, love]                    │              │
│  │         → predict "learning"                      │              │
│  │ Step 4: tgt = [<SOS>, I, love, learning]          │              │
│  │         → predict "<EOS>"  → 停止                 │              │
│  └──────────────────────────────────────────────────┘              │
│                                                                     │
│  解码策略:                                                          │
│  ┌────────────┬─────────────┬────────────────────┐                 │
│  │ Greedy     │ Beam Search │ Sampling           │                 │
│  │ 贪心搜索   │ 束搜索      │ 采样              │                 │
│  │ argmax     │ 保留top-k   │ top-k/top-p       │                 │
│  │ 最快但质量低│ 平衡质量速度│ 多样性高           │                 │
│  └────────────┴─────────────┴────────────────────┘                 │
│                                                                     │
│  学习率调度 (Warmup + Decay):                                       │
│  lr                                                                 │
│  │    /\                                                           │
│  │   /  \                                                          │
│  │  /    \_____                                                    │
│  │ /           \____                                               │
│  │/                  \___                                          │
│  └──────────────────────── steps                                   │
│  warmup_steps    decay phase                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**训练阶段**使用Teacher Forcing策略：将正确的目标序列（右移一位）作为Decoder的输入。这样Decoder在每个位置都能看到之前的正确token，加速收敛。

**推理阶段**使用自回归生成：从`&lt;SOS&gt;`开始，每次预测一个token，将预测结果追加到输入中，直到生成`&lt;EOS&gt;`或达到最大长度。

**解码策略对比：**

| 策略 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **Greedy** | 每步选概率最高的token | 最快 | 可能不是全局最优 |
| **Beam Search** | 保留top-k条路径 | 质量较高 | 速度较慢 |
| **Top-k Sampling** | 从top-k候选中随机采样 | 多样性好 | 可能生成低质量内容 |
| **Top-p (Nucleus)** | 从累积概率达p的候选中采样 | 自适应候选数量 | 需要调参 |
| **Temperature** | 调整logits的平滑程度 | 控制随机性 | T太高/太低都不好 |

### 代码示例

```python
# 训练和推理 - 完整实现
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import time
from typing import List, Tuple


# ===================== 1. 学习率调度器 =====================

class TransformerLRScheduler:
    """Transformer学习率调度器 (Warmup + 衰减)

    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """

    def __init__(self, optimizer, d_model: int,
                 warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        """更新学习率"""
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self) -> float:
        """计算当前学习率"""
        step = max(self.step_num, 1)
        return self.d_model ** (-0.5) * min(
            step ** (-0.5),
            step * self.warmup_steps ** (-1.5)
        )


# ===================== 2. 训练循环 =====================

class TransformerTrainer:
    """Transformer训练器"""

    def __init__(self, model, pad_idx: int = 0):
        self.model = model
        self.pad_idx = pad_idx

        # 损失函数 (忽略padding)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=0.1  # 标签平滑
        )

        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=0.0001,
            betas=(0.9, 0.98),
            eps=1e-9
        )

        # 学习率调度
        self.scheduler = TransformerLRScheduler(
            self.optimizer, model.encoder.d_model, warmup_steps=4000
        )

    def train_step(self, src: torch.Tensor,
                   tgt: torch.Tensor) -> float:
        """单步训练"""
        self.model.train()

        # 目标输入 (去掉最后一个token)
        tgt_input = tgt[:, :-1]
        # 目标标签 (去掉第一个token <SOS>)
        tgt_label = tgt[:, 1:]

        # 前向传播
        logits = self.model(src, tgt_input)

        # 计算损失
        # logits: (batch, tgt_len-1, vocab_size)
        # labels: (batch, tgt_len-1)
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_label.reshape(-1)
        )

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪 (防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def train_epoch(self, dataloader) -> float:
        """训练一个epoch"""
        total_loss = 0
        num_batches = 0

        for src, tgt in dataloader:
            loss = self.train_step(src, tgt)
            total_loss += loss
            num_batches += 1

        return total_loss / num_batches


# ===================== 3. 推理 (解码) =====================

class TransformerInference:
    """Transformer推理器"""

    def __init__(self, model, pad_idx: int = 0,
                 sos_idx: int = 1, eos_idx: int = 2):
        self.model = model
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor,
                      max_len: int = 100) -> torch.Tensor:
        """贪心解码"""
        self.model.eval()
        device = src.device

        # Encoder
        src_mask = self.model.create_src_mask(src)
        enc_output = self.model.encoder(src, src_mask)

        # 初始化Decoder输入
        tgt = torch.full(
            (src.size(0), 1), self.sos_idx,
            dtype=torch.long, device=device
        )

        for _ in range(max_len):
            tgt_mask = self.model.create_tgt_mask(tgt)
            dec_output = self.model.decoder(
                tgt, enc_output, tgt_mask, src_mask
            )
            logits = self.model.output_proj(dec_output[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)

            tgt = torch.cat([tgt, next_token], dim=1)

            # 检查是否所有样本都生成了EOS
            if (next_token == self.eos_idx).all():
                break

        return tgt

    @torch.no_grad()
    def beam_search_decode(self, src: torch.Tensor,
                           beam_size: int = 4,
                           max_len: int = 100) -> torch.Tensor:
        """束搜索解码"""
        self.model.eval()
        device = src.device
        batch_size = src.size(0)

        # 简化版：只处理batch_size=1
        assert batch_size == 1, "Beam search目前只支持batch_size=1"

        # Encoder
        src_mask = self.model.create_src_mask(src)
        enc_output = self.model.encoder(src, src_mask)

        # 扩展为beam_size份
        enc_output = enc_output.repeat(beam_size, 1, 1)
        src_mask = src_mask.repeat(beam_size, 1, 1, 1)

        # 初始化beam
        beam_seqs = torch.full(
            (beam_size, 1), self.sos_idx,
            dtype=torch.long, device=device
        )
        beam_scores = torch.zeros(beam_size, device=device)
        beam_scores[1:] = float('-inf')  # 初始只有第一个beam有效

        completed = []

        for step in range(max_len):
            tgt_mask = self.model.create_tgt_mask(beam_seqs)
            dec_output = self.model.decoder(
                beam_seqs, enc_output, tgt_mask, src_mask
            )
            logits = self.model.output_proj(dec_output[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)

            vocab_size = log_probs.size(-1)

            # 计算所有候选的分数
            next_scores = beam_scores.unsqueeze(1) + log_probs
            next_scores = next_scores.view(-1)  # (beam_size * vocab_size)

            # 选择top-k
            top_scores, top_indices = next_scores.topk(
                beam_size, dim=0)

            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            # 更新beam
            beam_seqs = torch.cat([
                beam_seqs[beam_indices],
                token_indices.unsqueeze(1)
            ], dim=1)
            beam_scores = top_scores

            # 检查完成的beam
            for i in range(beam_size):
                if token_indices[i] == self.eos_idx:
                    completed.append((beam_scores[i].item(), beam_seqs[i]))

            if len(completed) >= beam_size:
                break

        # 返回分数最高的序列
        if completed:
            completed.sort(key=lambda x: x[0], reverse=True)
            return completed[0][1].unsqueeze(0)
        else:
            return beam_seqs[0:1]

    @torch.no_grad()
    def sample_decode(self, src: torch.Tensor,
                      max_len: int = 100,
                      temperature: float = 1.0,
                      top_k: int = 50,
                      top_p: float = 0.9) -> torch.Tensor:
        """采样解码 (Top-k + Top-p + Temperature)"""
        self.model.eval()
        device = src.device

        src_mask = self.model.create_src_mask(src)
        enc_output = self.model.encoder(src, src_mask)

        tgt = torch.full(
            (src.size(0), 1), self.sos_idx,
            dtype=torch.long, device=device
        )

        for _ in range(max_len):
            tgt_mask = self.model.create_tgt_mask(tgt)
            dec_output = self.model.decoder(
                tgt, enc_output, tgt_mask, src_mask
            )
            logits = self.model.output_proj(dec_output[:, -1, :])

            # Temperature缩放
            logits = logits / temperature

            # Top-k过滤
            if top_k > 0:
                top_k_values, _ = logits.topk(top_k, dim=-1)
                min_value = top_k_values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(
                    logits < min_value, float('-inf')
                )

            # Top-p (Nucleus) 过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = logits.sort(
                    descending=True, dim=-1
                )
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                remove_mask = cum_probs > top_p
                remove_mask[:, 1:] = remove_mask[:, :-1].clone()
                remove_mask[:, 0] = False

                indices_to_remove = remove_mask.scatter(
                    1, sorted_indices, remove_mask
                )
                logits = logits.masked_fill(
                    indices_to_remove, float('-inf')
                )

            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            tgt = torch.cat([tgt, next_token], dim=1)

            if (next_token == self.eos_idx).all():
                break

        return tgt


# ===================== 4. 使用示例 =====================

def training_and_inference_demo():
    """训练和推理演示"""
    print("=" * 60)
    print("Transformer 训练和推理演示")
    print("=" * 60)

    # 配置
    src_vocab = 1000
    tgt_vocab = 1000
    d_model = 128
    n_heads = 4
    n_layers = 2
    d_ff = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print(f"设备: {device}")

    # 模拟训练数据
    batch_size = 16
    src_len = 20
    tgt_len = 22  # 包含 <SOS> 和 <EOS>

    src = torch.randint(3, src_vocab, (batch_size, src_len)).to(device)
    tgt = torch.randint(3, tgt_vocab, (batch_size, tgt_len)).to(device)
    tgt[:, 0] = 1  # <SOS>

    # 训练
    trainer = TransformerTrainer(model)
    print("\n--- 训练 ---")
    for epoch in range(5):
        loss = trainer.train_step(src, tgt)
        lr = trainer.scheduler._get_lr()
        print(f"Epoch {epoch+1}: loss={loss:.4f}, lr={lr:.6f}")

    # 推理
    inference = TransformerInference(model)
    test_src = torch.randint(3, src_vocab, (1, 10)).to(device)

    print("\n--- 推理 ---")

    # 贪心解码
    greedy_output = inference.greedy_decode(test_src, max_len=20)
    print(f"贪心解码输出: {greedy_output[0].tolist()}")

    # 采样解码
    sample_output = inference.sample_decode(
        test_src, max_len=20,
        temperature=0.8, top_k=50, top_p=0.9
    )
    print(f"采样解码输出: {sample_output[0].tolist()}")

    print("\n训练和推理演示完成!")


if __name__ == "__main__":
    training_and_inference_demo()
```

---

## GPT: Decoder-Only架构

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GPT Decoder-Only 架构                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  现代LLM (GPT/LLaMA/Qwen) 均使用Decoder-Only架构                   │
│  与原始Transformer Encoder-Decoder的区别:                            │
│                                                                     │
│  Encoder-Decoder (翻译):        Decoder-Only (生成):                │
│  ┌──────────┐  ┌──────────┐    ┌──────────────────────┐           │
│  │ Encoder  │─►│ Decoder  │    │  Decoder (N层)       │           │
│  │ 双向Attn │  │ 因果Attn │    │  因果Self-Attention  │           │
│  │ +CrossAttn│  │ +FFN     │    │  + FFN               │           │
│  └──────────┘  └──────────┘    │  (无Cross-Attention) │           │
│  输入: src+tgt  输出: tokens    │  输入: tokens         │           │
│                                 │  输出: next_token     │           │
│                                 └──────────────────────┘           │
│                                                                     │
│  GPT架构 (LLaMA风格):                                               │
│  ┌──────────────────────────────────────────────────────┐          │
│  │                                                      │          │
│  │  Token Embedding + Position (RoPE)                   │          │
│  │       │                                              │          │
│  │  ┌────▼────────────────────────────────────┐         │          │
│  │  │  Decoder Layer × N                      │         │          │
│  │  │  ┌───────────────────────────────────┐  │         │          │
│  │  │  │ 1. RMSNorm                       │  │         │          │
│  │  │  │ 2. Causal Multi-Head Attention    │  │         │          │
│  │  │  │    (+ RoPE on Q, K)              │  │         │          │
│  │  │  │ 3. Residual Connection            │  │         │          │
│  │  │  │ 4. RMSNorm                       │  │         │          │
│  │  │  │ 5. SwiGLU FFN (gate * up * down) │  │         │          │
│  │  │  │ 6. Residual Connection            │  │         │          │
│  │  │  └───────────────────────────────────┘  │         │          │
│  │  └─────────────────────────────────────────┘         │          │
│  │       │                                              │          │
│  │  RMSNorm                                             │          │
│  │       │                                              │          │
│  │  Linear Head (d_model → vocab_size)                  │          │
│  │       │                                              │          │
│  │  Output: logits (vocab_size)                         │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                     │
│  GPT vs 原始Transformer的关键差异:                                   │
│  ┌──────────────┬──────────────────┬──────────────────┐            │
│  │ 特性         │ 原始Transformer  │ GPT/LLaMA        │            │
│  ├──────────────┼──────────────────┼──────────────────┤            │
│  │ 归一化       │ Post-Norm (LayerN)│ Pre-Norm (RMSNorm)│           │
│  │ 激活函数     │ ReLU             │ SwiGLU            │            │
│  │ 位置编码     │ 正弦编码(加法)   │ RoPE(旋转)        │            │
│  │ FFN          │ 2层Linear        │ 3层(gate+up+down) │            │
│  │ Attention    │ 标准MHA          │ GQA(分组查询)     │            │
│  │ Cross-Attn   │ 有               │ 无                │            │
│  └──────────────┴──────────────────┴──────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

### 代码示例

```python
# GPT Decoder-Only 从零实现 (LLaMA风格)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ===================== 1. RMSNorm =====================

class RMSNorm(nn.Module):
    """RMS LayerNorm (LLaMA使用)

    比标准LayerNorm更简单高效: 不减均值, 只除以RMS
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ===================== 2. SwiGLU FFN =====================

class SwiGLUFFN(nn.Module):
    """SwiGLU前馈网络 (LLaMA使用)

    FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    比标准ReLU FFN效果更好
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))  # SiLU = x * sigmoid(x)
        up = self.up_proj(x)
        x = self.down_proj(gate * up)
        return self.dropout(x)


# ===================== 3. GQA (Grouped Query Attention) =====================

class GroupedQueryAttention(nn.Module):
    """分组查询注意力 (LLaMA 2/3, Mistral使用)

    Q有n_heads个头, K/V只有n_kv_heads个头 (n_kv_heads < n_heads)
    多个Q头共享同一组K/V, 减少KV Cache大小
    """

    def __init__(self, d_model: int, n_heads: int,
                 n_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        assert n_heads % n_kv_heads == 0

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads  # 每组Q头数
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d_k, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.size()

        # 投影
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)

        # 扩展KV到与Q相同的头数 (repeat interleave)
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.o_proj(out)


# ===================== 4. GPT Decoder Layer =====================

class GPTDecoderLayer(nn.Module):
    """GPT Decoder层 (LLaMA风格, Pre-Norm)"""

    def __init__(self, d_model: int, n_heads: int,
                 n_kv_heads: int, d_ff: int,
                 dropout: float = 0.0):
        super().__init__()

        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(
            d_model, n_heads, n_kv_heads, dropout
        )

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-Norm + Attention + Residual
        x = x + self.attn(self.attn_norm(x), mask)

        # Pre-Norm + FFN + Residual
        x = x + self.ffn(self.ffn_norm(x))

        return x


# ===================== 5. 完整GPT模型 =====================

class GPTModel(nn.Module):
    """GPT Decoder-Only模型 (简化版LLaMA)"""

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: int = 4,     # GQA: KV头数少于Q头数
        n_layers: int = 6,
        d_ff: int = 1376,        # SwiGLU常用: d_ff ≈ 8/3 * d_model
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token嵌入
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Decoder层
        self.layers = nn.ModuleList([
            GPTDecoderLayer(d_model, n_heads, n_kv_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 输出层
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重共享: 嵌入层和输出层共享权重
        self.lm_head.weight = self.token_emb.weight

        # 因果掩码 (预计算)
        causal_mask = torch.tril(
            torch.ones(max_seq_len, max_seq_len)
        ).unsqueeze(0).unsqueeze(0)
        self.register_buffer('causal_mask', causal_mask)

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token IDs

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"序列长度{T}超过最大{self.max_seq_len}"

        # Token嵌入 (注意: GPT/LLaMA不使用位置编码加法, 而用RoPE)
        x = self.token_emb(input_ids)

        # 因果掩码
        mask = self.causal_mask[:, :, :T, :T]

        # N层Decoder
        for layer in self.layers:
            x = layer(x, mask)

        # 输出
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor,
                 max_new_tokens: int = 100,
                 temperature: float = 1.0,
                 top_k: int = 50) -> torch.Tensor:
        """自回归文本生成"""
        self.eval()

        for _ in range(max_new_tokens):
            # 截断到max_seq_len
            idx_cond = input_ids[:, -self.max_seq_len:]

            # 前向传播
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k过滤
            if top_k > 0:
                v, _ = logits.topk(top_k, dim=-1)
                logits[logits < v[:, -1:]] = float('-inf')

            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 拼接
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# ===================== 演示 =====================

def gpt_demo():
    """GPT模型演示"""
    print("=" * 60)
    print("GPT Decoder-Only 模型演示")
    print("=" * 60)

    # 配置 (缩小版)
    model = GPTModel(
        vocab_size=1000,
        d_model=256,
        n_heads=8,
        n_kv_heads=4,     # GQA: 4个KV头, 8个Q头
        n_layers=4,
        d_ff=688,          # ≈ 8/3 * 256
        max_seq_len=512,
    )

    # 参数统计
    total = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total:,}")

    # 各组件参数量
    emb_params = sum(p.numel() for p in model.token_emb.parameters())
    layer_params = sum(p.numel() for p in model.layers.parameters())
    head_params = 0  # 权重共享

    print(f"  嵌入层: {emb_params:,}")
    print(f"  Decoder层: {layer_params:,}")
    print(f"  LM Head: 权重共享")

    # 前向传播
    input_ids = torch.randint(0, 1000, (2, 32))
    logits = model(input_ids)
    print(f"\n输入形状: {input_ids.shape}")
    print(f"输出形状: {logits.shape}")

    # 文本生成
    prompt = torch.randint(0, 1000, (1, 5))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"\n生成: {prompt.shape} -> {generated.shape}")
    print(f"生成的token IDs: {generated[0].tolist()}")


if __name__ == "__main__":
    gpt_demo()
```

---

## Flash Attention与注意力优化

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Flash Attention 原理与优化                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  标准Attention的瓶颈: IO-bound (内存带宽受限)                       │
│                                                                     │
│  标准实现:                                                          │
│  ┌──────┐  ┌──────────┐  ┌──────┐  ┌──────────┐  ┌──────┐        │
│  │ Q, K │─►│ QK^T/√dk │─►│ 写回 │─►│ softmax  │─►│ 写回 │        │
│  │ HBM  │  │  SRAM    │  │ HBM  │  │  SRAM    │  │ HBM  │        │
│  └──────┘  └──────────┘  └──────┘  └──────────┘  └──────┘        │
│     │                                                │              │
│     │  O(N^2 * d) 次 HBM 读写                       │              │
│     │  HBM带宽: ~2TB/s,  SRAM带宽: ~19TB/s         │              │
│     └────────────────────────────────────────────────┘              │
│                                                                     │
│  Flash Attention (分块计算):                                        │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  将Q, K, V按块(block)加载到SRAM                      │          │
│  │  在SRAM中完成 QK^T → softmax → × V 的全部计算       │          │
│  │  只写回最终结果到HBM                                  │          │
│  │                                                      │          │
│  │  ┌────┐ ┌────┐                                      │          │
│  │  │Q块1│ │K块1│──►  SRAM: 一次性计算所有步骤         │          │
│  │  └────┘ └────┘     ↓                                │          │
│  │  ┌────┐ ┌────┐  ┌────────┐                          │          │
│  │  │Q块1│ │K块2│──► 在线softmax (逐块累积)            │          │
│  │  └────┘ └────┘  │ 不需要存储N×N的注意力矩阵!        │          │
│  │        ...       └────────┘                          │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                     │
│  效果对比:                                                          │
│  ┌────────────────┬──────────────┬──────────────────────┐          │
│  │                │ 标准Attention │ Flash Attention 2    │          │
│  ├────────────────┼──────────────┼──────────────────────┤          │
│  │ 内存复杂度     │ O(N^2)       │ O(N) !!!             │          │
│  │ 速度 (A100)   │ 1x           │ 2-4x                 │          │
│  │ 序列长度上限   │ ~4K (显存限制)│ ~128K+               │          │
│  │ 精度           │ 精确         │ 精确 (非近似!)        │          │
│  └────────────────┴──────────────┴──────────────────────┘          │
│                                                                     │
│  GPU内存层次:                                                       │
│  ┌────────────────────────────────────────────────────────┐        │
│  │  SRAM (片上): 20MB, 19TB/s  ← Flash Attn在这里计算   │        │
│  │       ↕                                                │        │
│  │  HBM (显存): 80GB, 2TB/s   ← 标准Attn来回读写       │        │
│  │       ↕                                                │        │
│  │  CPU DRAM:   512GB, 50GB/s                            │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                     │
│  注意力变体发展:                                                    │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐     │
│  │ MHA        │ │ MQA        │ │ GQA        │ │ Flash Attn │     │
│  │ Multi-Head │ │ Multi-Query│ │ Grouped    │ │ IO-Aware   │     │
│  │ 原始论文   │ │ 1个KV头   │ │ 分组KV头   │ │ 分块计算   │     │
│  │ 2017       │ │ 2019       │ │ 2023       │ │ 2022/2023  │     │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

### 代码示例

```python
# Flash Attention与注意力优化
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Optional


# ===================== 1. PyTorch原生Flash Attention =====================

def use_flash_attention():
    """使用PyTorch 2.0+ 内置的Flash Attention"""
    print("=" * 60)
    print("Flash Attention (PyTorch 2.0+)")
    print("=" * 60)

    # PyTorch 2.0+ 自动使用Flash Attention
    # 通过 F.scaled_dot_product_attention 调用

    batch, heads, seq_len, d_k = 4, 8, 1024, 64

    q = torch.randn(batch, heads, seq_len, d_k, device="cpu")
    k = torch.randn(batch, heads, seq_len, d_k, device="cpu")
    v = torch.randn(batch, heads, seq_len, d_k, device="cpu")

    # 因果掩码
    # PyTorch 2.0+ 使用 is_causal=True 参数
    output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,  # 自动应用因果掩码
    )

    print(f"输入 Q: {q.shape}")
    print(f"输出:   {output.shape}")
    print(f"\nPyTorch版本: {torch.__version__}")

    # 检查可用的后端
    if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
        print(f"Flash SDP: {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"Memory-efficient SDP: "
              f"{torch.backends.cuda.mem_efficient_sdp_enabled()}")


# ===================== 2. 标准Attention vs Flash Attention性能对比 =====================

def standard_attention(q, k, v, mask=None):
    """标准注意力实现 (用于对比)"""
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def benchmark_attention():
    """注意力机制性能基准测试"""
    print("\n" + "=" * 60)
    print("Attention 性能对比")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    configs = [
        {"seq_len": 256, "d_k": 64, "heads": 8},
        {"seq_len": 1024, "d_k": 64, "heads": 8},
        {"seq_len": 4096, "d_k": 64, "heads": 8},
    ]

    for cfg in configs:
        seq_len = cfg["seq_len"]
        d_k = cfg["d_k"]
        heads = cfg["heads"]

        q = torch.randn(1, heads, seq_len, d_k, device=device)
        k = torch.randn(1, heads, seq_len, d_k, device=device)
        v = torch.randn(1, heads, seq_len, d_k, device=device)

        # 标准Attention
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = standard_attention(q, k, v)
        if device == "cuda":
            torch.cuda.synchronize()
        std_time = (time.perf_counter() - start) / 10

        # Flash Attention (via SDPA)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if device == "cuda":
            torch.cuda.synchronize()
        flash_time = (time.perf_counter() - start) / 10

        speedup = std_time / flash_time if flash_time > 0 else 0
        print(f"\n  seq_len={seq_len:5d}: "
              f"标准={std_time*1000:.2f}ms, "
              f"SDPA={flash_time*1000:.2f}ms, "
              f"加速={speedup:.2f}x")


# ===================== 3. KV Cache (推理加速) =====================

class KVCacheAttention(nn.Module):
    """带KV Cache的注意力 (推理加速)

    推理时, 已生成token的K/V不需要重新计算,
    缓存起来直接复用, 每步只计算新token的Q/K/V
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[tuple] = None) -> tuple:
        """
        Args:
            x: (batch, seq_len, d_model) -- 推理时seq_len=1
            kv_cache: (cached_k, cached_v) 或 None

        Returns:
            output, new_kv_cache
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 拼接KV Cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        # 更新Cache
        new_kv_cache = (k, v)

        # Attention (只计算新token的注意力)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(out), new_kv_cache


def kv_cache_demo():
    """KV Cache推理演示"""
    print("\n" + "=" * 60)
    print("KV Cache 推理加速演示")
    print("=" * 60)

    d_model, n_heads = 256, 8
    attn = KVCacheAttention(d_model, n_heads)
    attn.eval()

    # 模拟自回归生成
    batch = 1
    kv_cache = None

    # 第1步: 处理prompt (多个token)
    prompt = torch.randn(batch, 10, d_model)  # 10个token的prompt
    out, kv_cache = attn(prompt, kv_cache)
    print(f"Step 1 (Prompt): 输入={prompt.shape}, "
          f"KV Cache K形状={kv_cache[0].shape}")

    # 后续步骤: 每次只处理1个新token
    for step in range(5):
        new_token = torch.randn(batch, 1, d_model)
        out, kv_cache = attn(new_token, kv_cache)
        print(f"Step {step+2}: 输入=(1,1,{d_model}), "
              f"KV Cache长度={kv_cache[0].shape[2]}")

    print(f"\n无KV Cache: 每步计算全部token的K/V → O(N^2)")
    print(f"有KV Cache: 每步只计算1个新token → O(N)")


if __name__ == "__main__":
    use_flash_attention()
    benchmark_attention()
    kv_cache_demo()
```

---

## 注意力可视化

### 代码示例

```python
# 注意力权重可视化工具
import torch
import torch.nn.functional as F
import math


def visualize_attention_ascii(attn_weights: torch.Tensor,
                               tokens: list = None,
                               max_display: int = 10):
    """用ASCII图可视化注意力权重矩阵

    Args:
        attn_weights: (seq_len, seq_len) 注意力权重
        tokens: token列表
        max_display: 最大显示token数
    """
    seq_len = min(attn_weights.size(0), max_display)
    weights = attn_weights[:seq_len, :seq_len]

    if tokens is None:
        tokens = [f"T{i}" for i in range(seq_len)]
    tokens = tokens[:seq_len]

    # 表头
    max_tok_len = max(len(t) for t in tokens)
    header = " " * (max_tok_len + 2)
    for t in tokens:
        header += f"{t:>{max_tok_len+1}}"
    print(header)
    print(" " * (max_tok_len + 2) + "-" * (seq_len * (max_tok_len + 1)))

    # 亮度映射
    blocks = " ░▒▓█"

    for i in range(seq_len):
        row = f"{tokens[i]:>{max_tok_len}} | "
        for j in range(seq_len):
            w = weights[i, j].item()
            level = min(int(w * len(blocks)), len(blocks) - 1)
            # 显示数值
            row += f"{w:.2f} "
        print(row)


def create_attention_example():
    """创建注意力可视化示例"""
    print("=" * 60)
    print("注意力权重可视化")
    print("=" * 60)

    # 模拟一个句子的注意力
    tokens = ["The", "cat", "sat", "on", "mat"]
    seq_len = len(tokens)
    d_k = 8

    # 模拟Q, K
    torch.manual_seed(42)
    Q = torch.randn(seq_len, d_k)
    K = torch.randn(seq_len, d_k)

    # 计算注意力权重
    scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)

    # 因果掩码 (decoder)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    scores = scores.masked_fill(causal_mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)

    print("\n因果注意力权重矩阵:")
    print("(行=Query位置, 列=Key位置, 值=注意力权重)")
    print()
    visualize_attention_ascii(attn_weights, tokens)

    print("\n解读: 每行的值之和=1.0")
    print("  - 'cat' 主要关注 'The'(0.55) 和自己(0.45)")
    print("  - 'mat' 可以看到所有之前的token")
    print("  - 右上角全为0 (因果掩码, 不能看未来)")

    # 多头注意力可视化
    print("\n\n多头注意力 - 不同头关注不同模式:")
    n_heads = 4
    for head in range(n_heads):
        torch.manual_seed(head * 10)
        Q_h = torch.randn(seq_len, d_k)
        K_h = torch.randn(seq_len, d_k)
        scores_h = torch.matmul(Q_h, K_h.T) / math.sqrt(d_k)
        scores_h = scores_h.masked_fill(causal_mask == 0, float('-inf'))
        attn_h = F.softmax(scores_h, dim=-1)

        # 找到每个query最关注的key
        _, max_idx = attn_h.max(dim=-1)
        patterns = [f"{tokens[i]}→{tokens[max_idx[i]]}"
                    for i in range(seq_len)]
        print(f"  Head {head}: {', '.join(patterns)}")


if __name__ == "__main__":
    create_attention_example()
```

---

## 总结

本教程涵盖了Transformer架构详解的核心内容:

1. **Transformer概述**: Transformer是基于注意力机制的革命性架构，由Encoder和Decoder两部分组成，通过Self-Attention实现并行计算，是BERT、GPT等所有现代大语言模型的基础。

2. **Self-Attention**: 通过Q/K/V三个投影矩阵计算序列内部的注意力权重，使每个位置都能"关注"序列中的所有其他位置。缩放因子√d_k防止点积值过大。

3. **Multi-Head Attention**: 将注意力机制并行执行多次（多个头），每个头在不同的子空间中学习不同的注意力模式（语法、语义、位置等），增强模型的表达能力。

4. **Position Encoding**: 解决Attention对位置不敏感的问题，三种主要方法——正弦编码（固定）、可学习编码（BERT/GPT-2）、RoPE（LLaMA），各有优劣。

5. **完整实现**: 约200行核心代码实现了完整的Encoder-Decoder Transformer，包含所有组件：Embedding、Position Encoding、Multi-Head Attention、Feed Forward、Layer Norm、Residual Connection。

6. **训练和推理**: 训练使用Teacher Forcing和标签平滑，推理支持三种解码策略——贪心搜索、束搜索和Top-k/Top-p采样，配合Warmup学习率调度。

7. **GPT Decoder-Only架构**: 现代LLM（GPT/LLaMA/Qwen）均使用的Decoder-Only架构，包含RMSNorm、SwiGLU FFN、GQA（分组查询注意力）和权重共享等核心改进。

8. **Flash Attention与注意力优化**: Flash Attention通过分块计算和IO感知优化将注意力的内存复杂度从O(N^2)降至O(N)，速度提升2-4x。KV Cache是推理加速的关键技术。

9. **注意力可视化**: ASCII图可视化注意力权重矩阵，直观理解多头注意力中不同头学习的不同模式。

## 最佳实践

1. **模型维度**: d_model应该能被n_heads整除，常用配置如512/8、768/12、1024/16
2. **学习率**: 使用Warmup策略，先线性增大再衰减，warmup_steps通常4000-10000
3. **标签平滑**: label_smoothing=0.1可以提升泛化能力
4. **梯度裁剪**: clip_grad_norm设为1.0防止梯度爆炸
5. **初始化**: Xavier或Kaiming初始化，不要使用默认的随机初始化
6. **Layer Norm**: Pre-Norm（LN在Attention/FFN之前）比Post-Norm更稳定
7. **混合精度**: 使用FP16/BF16加速训练，大模型必备
8. **Flash Attention**: 使用`F.scaled_dot_product_attention`自动调用Flash Attention
9. **KV Cache**: 推理时缓存已计算的K/V，避免重复计算
10. **GQA**: 使用分组查询注意力减少KV Cache大小，提高推理效率
11. **权重共享**: 嵌入层和LM Head共享权重减少参数量

## 参考资源

- [Attention Is All You Need (原始论文)](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [minGPT (Karpathy)](https://github.com/karpathy/minGPT)
- [nanoGPT (Karpathy)](https://github.com/karpathy/nanoGPT)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

---

**文件大小目标**: 35-40KB
**创建时间**: 2024-01-15
**最后更新**: 2024-01-15
