# PyTorch基础教程

## 目录
1. [PyTorch简介](#pytorch简介)
2. [Tensor操作](#tensor操作)
3. [Autograd自动求导](#autograd自动求导)
4. [nn.Module](#nnmodule)
5. [训练循环](#训练循环)
6. [DataLoader](#dataloader)
7. [完整项目：MNIST手写数字识别](#完整项目mnist手写数字识别)

---

## PyTorch简介

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PyTorch 生态系统架构                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    应用层 (Applications)                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │   │
│  │  │ 计算机视觉│  │ 自然语言 │  │ 语音处理 │  │ 强化学习 │   │   │
│  │  │ torchvision│ │处理 (NLP)│  │torchaudio│  │          │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    核心层 (Core)                              │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │   │
│  │  │ Tensor   │  │ Autograd │  │ nn.Module│  │ Optim    │   │   │
│  │  │ 张量计算 │  │ 自动求导 │  │ 神经网络 │  │ 优化器   │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │   │
│  │  │DataLoader│  │ Distrib  │  │ JIT/TorchScript│            │   │
│  │  │ 数据加载 │  │ 分布式   │  │ 编译优化  │                │   │
│  │  └──────────┘  └──────────┘  └──────────┘                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    硬件层 (Hardware)                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │   │
│  │  │ CPU      │  │ CUDA GPU │  │ MPS      │  │ XPU      │   │   │
│  │  │ (默认)   │  │ (NVIDIA) │  │ (Apple)  │  │ (Intel)  │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**PyTorch** 是由 Meta（原Facebook）AI研究团队开发的开源深度学习框架。它以**动态计算图**（Define-by-Run）和**Pythonic**的设计理念著称，已成为学术研究和工业应用中最受欢迎的深度学习框架之一。

**PyTorch vs TensorFlow 对比：**

| 特性 | PyTorch | TensorFlow |
|------|---------|------------|
| **计算图** | 动态图（Eager Mode） | 静态图 + Eager Mode |
| **调试** | 原生Python调试 | 需要tf.debugging |
| **学术论文** | ~70%使用PyTorch | ~30%使用TensorFlow |
| **部署** | TorchServe, ONNX | TF Serving, TF Lite |
| **API风格** | Pythonic, 面向对象 | 函数式 + Keras |
| **社区** | 研究社区强大 | 工业部署生态完善 |
| **学习曲线** | 较平缓 | 较陡峭 |

**安装PyTorch：**

```bash
# CPU版本
pip install torch torchvision torchaudio

# CUDA 12.1 GPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon (MPS)
pip install torch torchvision torchaudio  # macOS自动支持MPS
```

### 代码示例

```python
# PyTorch简介 - 环境验证和基础操作
import torch
import sys


def check_pytorch_environment():
    """检查PyTorch环境"""
    print("=" * 50)
    print("PyTorch 环境信息")
    print("=" * 50)

    print(f"PyTorch 版本: {torch.__version__}")
    print(f"Python 版本: {sys.version}")

    # CUDA信息
    print(f"\nCUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            mem = torch.cuda.get_device_properties(i).total_mem
            print(f"  显存: {mem / 1024**3:.1f} GB")

    # MPS信息 (Apple Silicon)
    if hasattr(torch.backends, "mps"):
        print(f"\nMPS 可用: {torch.backends.mps.is_available()}")

    # 选择最佳设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\n当前使用设备: {device}")
    return device


def basic_tensor_demo():
    """基础Tensor演示"""
    # 创建Tensor
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"\n基础Tensor: {x}")
    print(f"  形状: {x.shape}")
    print(f"  数据类型: {x.dtype}")
    print(f"  设备: {x.device}")

    # 简单计算
    y = x * 2 + 1
    print(f"  x * 2 + 1 = {y}")

    # 矩阵运算
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    C = torch.matmul(A, B)
    print(f"\n矩阵乘法结果:\n{C}")


if __name__ == "__main__":
    device = check_pytorch_environment()
    basic_tensor_demo()
```

---

## Tensor操作

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Tensor（张量）概念图                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  维度(Rank)示意:                                                     │
│                                                                     │
│  0维: 标量      1维: 向量        2维: 矩阵          3维: 张量       │
│  ┌───┐         ┌─────────┐    ┌─────────┐      ┌─────────┐       │
│  │ 5 │         │ 1 2 3 4 │    │ 1 2 3   │      │┌───────┐│       │
│  └───┘         └─────────┘    │ 4 5 6   │      ││ 1 2 3 ││       │
│  shape: ()     shape: (4,)    │ 7 8 9   │      ││ 4 5 6 ││       │
│                               └─────────┘      │└───────┘│       │
│                               shape: (3,3)     │┌───────┐│       │
│                                                ││ 7 8 9 ││       │
│                                                ││10 11 12││       │
│                                                │└───────┘│       │
│                                                └─────────┘       │
│                                                shape: (2,2,3)    │
│                                                                     │
│  Tensor属性:                                                        │
│  ┌──────────┬──────────┬──────────┬──────────┐                     │
│  │  shape   │  dtype   │  device  │ requires │                     │
│  │  形状    │ 数据类型  │  设备    │  _grad   │                     │
│  │ (3,4,5) │ float32  │ cuda:0  │  True    │                     │
│  └──────────┴──────────┴──────────┴──────────┘                     │
│                                                                     │
│  常见数据类型:                                                       │
│  float32(默认) | float16(半精度) | float64 | int32 | int64 | bool  │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**Tensor（张量）** 是PyTorch中最基本的数据结构，类似于NumPy的ndarray，但增加了GPU加速和自动求导的能力。

**Tensor的四个核心属性：**

| 属性 | 说明 | 示例 |
|------|------|------|
| `shape` | 张量的维度和大小 | `torch.Size([3, 4])` |
| `dtype` | 数据类型 | `torch.float32` |
| `device` | 存储设备 | `cpu` 或 `cuda:0` |
| `requires_grad` | 是否需要计算梯度 | `True` / `False` |

### 代码示例

```python
# Tensor操作 - 完整示例
import torch
import numpy as np


# ===================== 1. Tensor创建 =====================

def tensor_creation():
    """Tensor创建的各种方式"""
    print("=" * 50)
    print("1. Tensor创建方式")
    print("=" * 50)

    # 从Python列表创建
    t1 = torch.tensor([1, 2, 3, 4])
    print(f"从列表创建: {t1}")

    # 从NumPy数组创建
    np_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    t2 = torch.from_numpy(np_arr)
    print(f"从NumPy创建: {t2}")

    # 指定数据类型
    t3 = torch.tensor([1, 2, 3], dtype=torch.float32)
    print(f"指定float32: {t3}")

    # 特殊Tensor
    zeros = torch.zeros(3, 4)         # 全零
    ones = torch.ones(2, 3)           # 全一
    eye = torch.eye(3)                # 单位矩阵
    rand = torch.rand(2, 3)           # 均匀分布 [0, 1)
    randn = torch.randn(2, 3)         # 标准正态分布
    arange = torch.arange(0, 10, 2)   # 等差序列
    linspace = torch.linspace(0, 1, 5)  # 等间隔

    print(f"\n全零矩阵 (3x4):\n{zeros}")
    print(f"随机矩阵 (2x3):\n{rand}")
    print(f"等差序列: {arange}")
    print(f"等间隔序列: {linspace}")

    # 根据已有Tensor创建（形状和设备一致）
    t4 = torch.zeros_like(t2)
    t5 = torch.randn_like(t3)
    print(f"\nzeros_like: {t4}")
    print(f"randn_like: {t5}")


# ===================== 2. Tensor形状操作 =====================

def tensor_shape_operations():
    """Tensor形状变换"""
    print("\n" + "=" * 50)
    print("2. Tensor形状操作")
    print("=" * 50)

    x = torch.arange(12).float()
    print(f"原始Tensor: {x}")
    print(f"  形状: {x.shape}")

    # reshape / view
    x_2d = x.reshape(3, 4)
    print(f"\nreshape(3, 4):\n{x_2d}")

    x_view = x.view(4, 3)
    print(f"\nview(4, 3):\n{x_view}")

    # -1 自动推断维度
    x_auto = x.reshape(2, -1)
    print(f"\nreshape(2, -1):\n{x_auto}")
    print(f"  形状: {x_auto.shape}")  # (2, 6)

    # 增加/删除维度
    x_3d = x_2d.unsqueeze(0)     # 增加批次维度
    print(f"\nunsqueeze(0) 形状: {x_3d.shape}")  # (1, 3, 4)

    x_squeeze = x_3d.squeeze(0)   # 删除大小为1的维度
    print(f"squeeze(0) 形状: {x_squeeze.shape}")  # (3, 4)

    # 转置
    x_t = x_2d.t()               # 2D转置
    print(f"\n转置:\n{x_t}")

    # permute - 多维转置
    batch = torch.randn(2, 3, 4)
    batch_p = batch.permute(0, 2, 1)  # (2, 4, 3)
    print(f"\npermute(0,2,1): {batch.shape} -> {batch_p.shape}")

    # 拼接
    a = torch.ones(2, 3)
    b = torch.zeros(2, 3)
    cat_0 = torch.cat([a, b], dim=0)  # 沿行拼接
    cat_1 = torch.cat([a, b], dim=1)  # 沿列拼接
    print(f"\ncat(dim=0) 形状: {cat_0.shape}")  # (4, 3)
    print(f"cat(dim=1) 形状: {cat_1.shape}")  # (2, 6)

    # stack - 创建新维度
    stacked = torch.stack([a, b], dim=0)
    print(f"stack(dim=0) 形状: {stacked.shape}")  # (2, 2, 3)


# ===================== 3. Tensor数学运算 =====================

def tensor_math_operations():
    """Tensor数学运算"""
    print("\n" + "=" * 50)
    print("3. Tensor数学运算")
    print("=" * 50)

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    # 逐元素运算
    print(f"a + b =\n{a + b}")
    print(f"a * b =\n{a * b}")        # 逐元素乘法
    print(f"a / b =\n{a / b}")
    print(f"a ** 2 =\n{a ** 2}")

    # 矩阵乘法
    mm = torch.matmul(a, b)           # 或 a @ b
    print(f"\n矩阵乘法 (a @ b):\n{mm}")

    # 批量矩阵乘法
    batch_a = torch.randn(4, 3, 2)   # batch=4, 3x2矩阵
    batch_b = torch.randn(4, 2, 5)   # batch=4, 2x5矩阵
    batch_mm = torch.bmm(batch_a, batch_b)
    print(f"\n批量矩阵乘法: {batch_a.shape} @ {batch_b.shape}"
          f" = {batch_mm.shape}")

    # 统计运算
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"\n统计运算:")
    print(f"  sum: {x.sum()}")
    print(f"  mean: {x.mean()}")
    print(f"  std: {x.std()}")
    print(f"  max: {x.max()}")
    print(f"  min: {x.min()}")
    print(f"  argmax: {x.argmax()}")

    # 沿维度运算
    matrix = torch.tensor([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]])
    print(f"\n沿dim=0求和 (列): {matrix.sum(dim=0)}")
    print(f"沿dim=1求和 (行): {matrix.sum(dim=1)}")
    print(f"沿dim=1求最大值: {matrix.max(dim=1)}")

    # 广播机制 (Broadcasting)
    print(f"\n广播机制:")
    a = torch.tensor([[1.0], [2.0], [3.0]])  # (3, 1)
    b = torch.tensor([10.0, 20.0, 30.0])     # (3,)
    result = a + b  # 广播为 (3, 3)
    print(f"  {a.shape} + {b.shape} = {result.shape}")
    print(f"  结果:\n{result}")


# ===================== 4. Tensor索引和切片 =====================

def tensor_indexing():
    """Tensor索引和切片"""
    print("\n" + "=" * 50)
    print("4. Tensor索引和切片")
    print("=" * 50)

    x = torch.arange(20).reshape(4, 5).float()
    print(f"原始矩阵 (4x5):\n{x}")

    # 基础索引
    print(f"\nx[0]: {x[0]}")           # 第一行
    print(f"x[0, 0]: {x[0, 0]}")      # 第一个元素
    print(f"x[-1]: {x[-1]}")           # 最后一行

    # 切片
    print(f"\nx[:2]: \n{x[:2]}")       # 前两行
    print(f"x[:, :3]: \n{x[:, :3]}")   # 前三列
    print(f"x[1:3, 2:4]: \n{x[1:3, 2:4]}")  # 子矩阵

    # 布尔索引
    mask = x > 10
    print(f"\nx > 10 的元素: {x[mask]}")

    # fancy indexing
    indices = torch.tensor([0, 2, 3])
    print(f"\n选择第0,2,3行:\n{x[indices]}")

    # where - 条件选择
    result = torch.where(x > 10, x, torch.zeros_like(x))
    print(f"\nwhere(x>10, x, 0):\n{result}")


# ===================== 5. GPU操作 =====================

def tensor_gpu_operations():
    """Tensor GPU操作"""
    print("\n" + "=" * 50)
    print("5. GPU操作")
    print("=" * 50)

    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {device}")

    # 在GPU上创建Tensor
    x = torch.randn(3, 3, device=device)
    print(f"\n在{device}上创建: {x.device}")

    # CPU和GPU之间移动
    x_cpu = torch.randn(3, 3)
    x_gpu = x_cpu.to(device)           # 移到GPU
    x_back = x_gpu.cpu()               # 移回CPU
    print(f"CPU -> {device}: {x_gpu.device}")
    print(f"{device} -> CPU: {x_back.device}")

    # GPU运算速度对比
    if torch.cuda.is_available():
        import time

        size = 5000
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()

        # CPU计算
        start = time.time()
        _ = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start

        # GPU计算 (含同步)
        torch.cuda.synchronize()
        start = time.time()
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        print(f"\n{size}x{size} 矩阵乘法:")
        print(f"  CPU 耗时: {cpu_time:.4f}秒")
        print(f"  GPU 耗时: {gpu_time:.4f}秒")
        print(f"  加速比: {cpu_time / gpu_time:.1f}x")


# 运行所有示例
if __name__ == "__main__":
    tensor_creation()
    tensor_shape_operations()
    tensor_math_operations()
    tensor_indexing()
    tensor_gpu_operations()
```

---

## Autograd自动求导

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Autograd 自动求导原理                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  前向传播 (Forward Pass):                                            │
│  ┌───┐     ┌──────┐     ┌──────┐     ┌──────┐     ┌───┐          │
│  │ x │────►│ z=Wx │────►│ a=σ(z)│────►│ L=loss│────►│ L │          │
│  └───┘     └──────┘     └──────┘     └──────┘     └───┘          │
│    ↓          ↓            ↓            ↓                          │
│  输入       线性变换      激活函数      损失计算      标量损失        │
│                                                                     │
│  反向传播 (Backward Pass) - 链式法则:                                │
│  ┌───┐     ┌──────┐     ┌──────┐     ┌──────┐     ┌───┐          │
│  │∂L │◄────│∂L/∂z │◄────│∂L/∂a │◄────│∂L/∂L │◄────│ 1 │          │
│  │∂W │     │=∂L/∂a│     │=∂L/∂L│     │ = 1  │     └───┘          │
│  └───┘     │·∂a/∂z│     │·∂L/∂a│     └──────┘                     │
│  梯度       └──────┘     └──────┘                                   │
│                                                                     │
│  计算图 (Computational Graph):                                       │
│  ┌─┐   ┌─┐                                                         │
│  │x│   │W│  requires_grad=True                                     │
│  └┬┘   └┬┘                                                         │
│   │     │                                                           │
│   └──┬──┘                                                           │
│      ▼                                                              │
│  ┌──────┐   grad_fn=MulBackward                                    │
│  │z=W*x │                                                           │
│  └──┬───┘                                                           │
│     ▼                                                               │
│  ┌──────┐   grad_fn=SigmoidBackward                                │
│  │a=σ(z)│                                                           │
│  └──┬───┘                                                           │
│     ▼                                                               │
│  ┌──────┐   grad_fn=MSELossBackward                                │
│  │ Loss │                                                           │
│  └──────┘                                                           │
│     │  .backward()  自动计算所有梯度                                 │
│     ▼                                                               │
│  W.grad = ∂Loss/∂W  (存储在参数的.grad属性中)                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**Autograd（自动求导）** 是PyTorch的核心特性之一，它通过动态构建计算图来自动计算梯度。这是训练神经网络的基础——通过反向传播算法计算损失函数对每个参数的梯度，然后使用优化器更新参数。

**Autograd的工作原理：**

1. **前向传播**：执行计算操作时，PyTorch自动构建计算图
2. **记录操作**：每个操作（加、乘、激活函数等）都会被记录为计算图中的节点
3. **反向传播**：调用`.backward()`时，通过链式法则自动计算所有梯度
4. **梯度存储**：梯度被存储在每个`requires_grad=True`的Tensor的`.grad`属性中

**关键概念对照表：**

| 概念 | 说明 | API |
|------|------|-----|
| `requires_grad` | 标记Tensor需要计算梯度 | `tensor.requires_grad_(True)` |
| `grad_fn` | 记录创建该Tensor的操作 | `tensor.grad_fn` |
| `backward()` | 触发反向传播 | `loss.backward()` |
| `grad` | 存储计算得到的梯度 | `tensor.grad` |
| `no_grad()` | 临时禁用梯度计算 | `with torch.no_grad():` |
| `detach()` | 从计算图中分离 | `tensor.detach()` |
| `zero_grad()` | 清零梯度 | `optimizer.zero_grad()` |

### 代码示例

```python
# Autograd自动求导 - 完整示例
import torch
import torch.nn as nn


# ===================== 1. 基础自动求导 =====================

def basic_autograd():
    """基础自动求导演示"""
    print("=" * 50)
    print("1. 基础自动求导")
    print("=" * 50)

    # 创建需要梯度的Tensor
    x = torch.tensor(2.0, requires_grad=True)
    w = torch.tensor(3.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)

    # 前向计算: y = w*x + b
    y = w * x + b
    print(f"y = w*x + b = {w}*{x} + {b} = {y}")
    print(f"y.grad_fn: {y.grad_fn}")

    # 反向传播
    y.backward()

    # 查看梯度
    print(f"\n梯度:")
    print(f"  dy/dx = w = {x.grad}")      # 3.0
    print(f"  dy/dw = x = {w.grad}")      # 2.0
    print(f"  dy/db = 1 = {b.grad}")      # 1.0


# ===================== 2. 复杂函数的梯度 =====================

def complex_gradient():
    """复杂函数的梯度计算"""
    print("\n" + "=" * 50)
    print("2. 复杂函数的梯度")
    print("=" * 50)

    # f(x) = x^3 + 2x^2 + 3x + 4
    x = torch.tensor(2.0, requires_grad=True)
    f = x**3 + 2 * x**2 + 3 * x + 4
    f.backward()
    # df/dx = 3x^2 + 4x + 3 = 3*4 + 4*2 + 3 = 23
    print(f"f(x) = x^3 + 2x^2 + 3x + 4")
    print(f"f(2) = {f.item()}")
    print(f"f'(x) = 3x^2 + 4x + 3")
    print(f"f'(2) = {x.grad.item()}")  # 23.0

    # 多变量函数
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x * x).sum()  # y = x1^2 + x2^2 + x3^2
    y.backward()
    print(f"\ny = sum(x^2), x = {[1, 2, 3]}")
    print(f"dy/dx = 2x = {x.grad}")  # [2, 4, 6]


# ===================== 3. 梯度累积和清零 =====================

def gradient_accumulation():
    """梯度累积和清零"""
    print("\n" + "=" * 50)
    print("3. 梯度累积和清零")
    print("=" * 50)

    w = torch.tensor(1.0, requires_grad=True)

    # 第一次backward
    y1 = w * 2
    y1.backward()
    print(f"第1次backward后 w.grad = {w.grad}")  # 2.0

    # 第二次backward - 梯度会累积！
    y2 = w * 3
    y2.backward()
    print(f"第2次backward后 w.grad = {w.grad}")  # 5.0 (2+3)

    # 清零梯度
    w.grad.zero_()
    print(f"清零后 w.grad = {w.grad}")  # 0.0

    # 第三次backward
    y3 = w * 4
    y3.backward()
    print(f"第3次backward后 w.grad = {w.grad}")  # 4.0

    print("\n注意: 在训练循环中，每次backward前必须清零梯度！")


# ===================== 4. 禁用梯度计算 =====================

def disable_gradient():
    """禁用梯度计算"""
    print("\n" + "=" * 50)
    print("4. 禁用梯度计算")
    print("=" * 50)

    x = torch.randn(3, requires_grad=True)

    # 方法1: torch.no_grad() 上下文管理器
    with torch.no_grad():
        y = x * 2
        print(f"no_grad中: y.requires_grad = {y.requires_grad}")

    # 方法2: detach()
    y_detached = x.detach()
    print(f"detach后: y.requires_grad = {y_detached.requires_grad}")

    # 方法3: @torch.no_grad() 装饰器
    @torch.no_grad()
    def inference(model_input):
        return model_input * 2

    result = inference(x)
    print(f"装饰器: result.requires_grad = {result.requires_grad}")

    print("\n使用场景: 推理时、评估时、计算指标时")


# ===================== 5. 实际应用：线性回归 =====================

def linear_regression_autograd():
    """用Autograd实现线性回归"""
    print("\n" + "=" * 50)
    print("5. 线性回归 (纯Autograd)")
    print("=" * 50)

    # 生成数据: y = 3x + 2 + noise
    torch.manual_seed(42)
    X = torch.randn(100, 1)
    y_true = 3 * X + 2 + torch.randn(100, 1) * 0.3

    # 初始化参数
    w = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    learning_rate = 0.1
    epochs = 100

    for epoch in range(epochs):
        # 前向传播
        y_pred = X * w + b

        # 计算MSE损失
        loss = ((y_pred - y_true) ** 2).mean()

        # 反向传播
        loss.backward()

        # 手动更新参数 (不记录梯度)
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        # 清零梯度
        w.grad.zero_()
        b.grad.zero_()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: "
                  f"loss={loss.item():.4f}, "
                  f"w={w.item():.4f}, "
                  f"b={b.item():.4f}")

    print(f"\n最终结果: y = {w.item():.4f}x + {b.item():.4f}")
    print(f"真实关系: y = 3.0000x + 2.0000")


if __name__ == "__main__":
    basic_autograd()
    complex_gradient()
    gradient_accumulation()
    disable_gradient()
    linear_regression_autograd()
```

---

## nn.Module

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    nn.Module 继承体系                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────┐                                                │
│  │   nn.Module     │  所有神经网络的基类                             │
│  │  ┌────────────┐│                                                │
│  │  │ parameters()││  所有可训练参数                                 │
│  │  │ forward()   ││  前向传播逻辑                                  │
│  │  │ train()     ││  设置训练模式                                  │
│  │  │ eval()      ││  设置评估模式                                  │
│  │  │ to(device)  ││  移动到指定设备                                │
│  │  │ state_dict()││  保存/加载模型                                 │
│  │  └────────────┘│                                                │
│  └───────┬────────┘                                                │
│          │ 继承                                                     │
│     ┌────┼────────────────────────────┐                            │
│     ▼    ▼                            ▼                            │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐                    │
│  │ nn.Linear│  │ nn.Conv2d│  │ 自定义Module  │                    │
│  │ 全连接层 │  │ 卷积层   │  │ class MyModel │                    │
│  └──────────┘  └──────────┘  │   (nn.Module) │                    │
│                              │  def __init__  │                    │
│  常用预定义层:                │  def forward   │                    │
│  ┌──────────────────┐       └───────────────┘                    │
│  │ nn.Linear        │  全连接层                                   │
│  │ nn.Conv1d/2d/3d  │  卷积层                                    │
│  │ nn.LSTM/GRU      │  循环层                                    │
│  │ nn.BatchNorm     │  批归一化                                   │
│  │ nn.Dropout       │  随机丢弃                                   │
│  │ nn.Embedding     │  嵌入层                                    │
│  │ nn.MultiheadAttn │  多头注意力                                 │
│  │ nn.Transformer   │  Transformer                               │
│  └──────────────────┘                                             │
│                                                                     │
│  模型构建方式:                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ nn.Sequential│  │ 手动定义     │  │ nn.ModuleList│             │
│  │ 顺序堆叠    │  │ forward方法  │  │ 动态层列表   │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**nn.Module** 是PyTorch中所有神经网络模块的基类。自定义模型需要继承`nn.Module`，并实现`__init__`和`forward`方法。

**nn.Module的核心方法：**

| 方法 | 说明 | 使用场景 |
|------|------|----------|
| `__init__()` | 定义网络层 | 初始化时 |
| `forward()` | 定义前向传播 | 自动调用 |
| `parameters()` | 返回所有可训练参数 | 传给优化器 |
| `named_parameters()` | 返回参数名和值 | 调试检查 |
| `train()` | 设置训练模式 | 训练时 |
| `eval()` | 设置评估模式 | 推理时 |
| `to(device)` | 移动到指定设备 | GPU训练 |
| `state_dict()` | 获取模型参数字典 | 保存模型 |
| `load_state_dict()` | 加载模型参数 | 加载模型 |

### 代码示例

```python
# nn.Module - 完整示例
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== 1. 基础模型定义 =====================

class SimpleLinearModel(nn.Module):
    """简单的线性模型"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ===================== 2. 多层感知机 (MLP) =====================

class MLP(nn.Module):
    """多层感知机"""

    def __init__(self, input_dim: int, hidden_dim: int,
                 output_dim: int, dropout: float = 0.1):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一层: 线性 + BN + ReLU + Dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 第二层: 线性 + BN + ReLU + Dropout
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 输出层
        x = self.fc3(x)
        return x


# ===================== 3. nn.Sequential =====================

def sequential_model():
    """使用nn.Sequential快速构建模型"""

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
    )

    return model


# ===================== 4. 卷积神经网络 (CNN) =====================

class SimpleCNN(nn.Module):
    """简单的卷积神经网络"""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 卷积层
        self.conv_layers = nn.Sequential(
            # Conv1: 1 -> 32 通道, 3x3卷积
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14

            # Conv2: 32 -> 64 通道
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
        )

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),              # 64*7*7 = 3136
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ===================== 5. 模型工具函数 =====================

def model_utilities():
    """模型实用功能演示"""
    print("=" * 50)
    print("模型实用功能")
    print("=" * 50)

    model = MLP(input_dim=784, hidden_dim=256, output_dim=10)

    # 查看模型结构
    print(f"\n模型结构:\n{model}")

    # 查看参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数: {trainable:,}")

    # 查看各层参数
    print("\n各层参数:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    print("\n模型已保存到 model.pth")

    # 加载模型
    loaded_model = MLP(input_dim=784, hidden_dim=256, output_dim=10)
    loaded_model.load_state_dict(torch.load("model.pth"))
    print("模型已从 model.pth 加载")

    # 模型模式切换
    model.train()   # 训练模式 (Dropout和BN正常工作)
    model.eval()    # 评估模式 (Dropout关闭, BN使用运行统计量)

    # 移到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"\n模型已移至: {device}")

    # 冻结层 (迁移学习常用)
    for param in model.fc1.parameters():
        param.requires_grad = False
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    print(f"冻结了 {frozen} 个参数张量")


if __name__ == "__main__":
    model_utilities()

    # 测试前向传播
    model = SimpleCNN(num_classes=10)
    x = torch.randn(4, 1, 28, 28)  # batch=4, 1通道, 28x28
    output = model(x)
    print(f"\nCNN输出形状: {output.shape}")  # (4, 10)
```

---

## 训练循环

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PyTorch 标准训练循环                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────┐           │
│  │                   训练循环 (Training Loop)            │           │
│  │                                                       │           │
│  │  for epoch in range(num_epochs):                      │           │
│  │    for batch in dataloader:                           │           │
│  │      ┌─────────────────────────────────────────┐     │           │
│  │      │ 1. optimizer.zero_grad()  # 清零梯度    │     │           │
│  │      │         ↓                               │     │           │
│  │      │ 2. output = model(input)  # 前向传播    │     │           │
│  │      │         ↓                               │     │           │
│  │      │ 3. loss = criterion(output, target)     │     │           │
│  │      │                           # 计算损失    │     │           │
│  │      │         ↓                               │     │           │
│  │      │ 4. loss.backward()        # 反向传播    │     │           │
│  │      │         ↓                               │     │           │
│  │      │ 5. optimizer.step()       # 更新参数    │     │           │
│  │      └─────────────────────────────────────────┘     │           │
│  │                                                       │           │
│  │    validate(model, val_dataloader)  # 验证            │           │
│  │    save_checkpoint(model, epoch)    # 保存检查点       │           │
│  └─────────────────────────────────────────────────────┘           │
│                                                                     │
│  损失函数:                        优化器:                            │
│  ┌──────────────────┐           ┌──────────────────┐               │
│  │ CrossEntropyLoss │           │ SGD              │               │
│  │ MSELoss          │           │ Adam (最常用)    │               │
│  │ BCELoss          │           │ AdamW            │               │
│  │ L1Loss           │           │ RMSprop          │               │
│  │ NLLLoss          │           │ LBFGS            │               │
│  └──────────────────┘           └──────────────────┘               │
│                                                                     │
│  学习率调度器:                                                      │
│  ┌──────────────────────────────────────────────┐                  │
│  │ StepLR | CosineAnnealingLR | ReduceLROnPlateau│                  │
│  │ OneCycleLR | WarmupCosineSchedule              │                  │
│  └──────────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**训练循环**是深度学习中最核心的流程，包含五个关键步骤：清零梯度、前向传播、计算损失、反向传播、更新参数。

**常用损失函数对照表：**

| 损失函数 | 适用场景 | 输入要求 |
|----------|----------|----------|
| `nn.CrossEntropyLoss` | 多分类 | logits (未softmax) + 类别索引 |
| `nn.BCEWithLogitsLoss` | 二分类 | logits + 0/1标签 |
| `nn.MSELoss` | 回归 | 预测值 + 真实值 |
| `nn.L1Loss` | 回归(鲁棒) | 预测值 + 真实值 |
| `nn.NLLLoss` | 多分类 | log概率 + 类别索引 |

**常用优化器对照表：**

| 优化器 | 特点 | 推荐场景 |
|--------|------|----------|
| `SGD` | 简单、需要精心调参 | 经典CV模型 |
| `Adam` | 自适应学习率、通用 | 大多数任务 |
| `AdamW` | Adam + 权重衰减解耦 | Transformer |
| `RMSprop` | 适合非平稳目标 | RNN |

### 代码示例

```python
# 训练循环 - 完整的训练框架
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Tuple, Optional
import time


class Trainer:
    """通用训练器"""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        scheduler: Optional[object] = None
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        # 训练历史
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

    def train_one_epoch(self, dataloader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # ===== 核心五步 =====
            # 1. 清零梯度
            self.optimizer.zero_grad()

            # 2. 前向传播
            outputs = self.model(inputs)

            # 3. 计算损失
            loss = self.criterion(outputs, targets)

            # 4. 反向传播
            loss.backward()

            # 5. 更新参数
            self.optimizer.step()
            # ====================

            # 统计
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, dataloader) -> Tuple[float, float]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def fit(self, train_loader, val_loader,
            num_epochs: int, verbose: bool = True) -> Dict:
        """完整训练流程"""
        best_val_acc = 0.0
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # 训练
            train_loss, train_acc = self.train_one_epoch(train_loader)

            # 验证
            val_loss, val_acc = self.validate(val_loader)

            # 更新学习率
            if self.scheduler:
                self.scheduler.step()

            # 记录历史
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), "best_model.pth")

            # 打印进度
            if verbose:
                epoch_time = time.time() - epoch_start
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch+1:3d}/{num_epochs} "
                    f"| Train Loss: {train_loss:.4f} "
                    f"| Train Acc: {train_acc:.4f} "
                    f"| Val Loss: {val_loss:.4f} "
                    f"| Val Acc: {val_acc:.4f} "
                    f"| LR: {current_lr:.6f} "
                    f"| Time: {epoch_time:.1f}s"
                )

        total_time = time.time() - start_time
        print(f"\n训练完成! 总耗时: {total_time:.1f}s")
        print(f"最佳验证准确率: {best_val_acc:.4f}")

        return self.history


# 使用示例
def training_demo():
    """训练演示"""
    print("=" * 60)
    print("训练循环演示")
    print("=" * 60)

    # 创建模型
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 10)
    )

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建训练器
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )

    print(f"\n模型参数量: "
          f"{sum(p.numel() for p in model.parameters()):,}")
    print(f"设备: {device}")
    print(f"优化器: {type(optimizer).__name__}")
    print(f"调度器: {type(scheduler).__name__}")

    print("\n训练器已准备就绪，可调用trainer.fit()开始训练")


if __name__ == "__main__":
    training_demo()
```

---

## DataLoader

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DataLoader 数据管道                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────┐     ┌───────────┐     ┌───────────┐                │
│  │ 原始数据  │────►│ Dataset   │────►│DataLoader │                │
│  │ 文件/API  │     │ 数据集类  │     │ 批次迭代  │                │
│  └───────────┘     └───────────┘     └───────────┘                │
│                         │                  │                       │
│                    ┌────┴────┐        ┌────┴────┐                  │
│                    │__len__  │        │batch_size│                  │
│                    │__getitem│        │shuffle   │                  │
│                    │transform│        │num_workers│                  │
│                    └─────────┘        │pin_memory│                  │
│                                       │collate_fn│                  │
│                                       └─────────┘                  │
│                                                                     │
│  数据流:                                                            │
│  ┌──────┐    ┌─────────┐    ┌──────────┐    ┌────────┐           │
│  │ 磁盘 │───►│ Dataset │───►│Transform │───►│ Batch  │           │
│  │ 存储 │    │ __getitem│   │ 数据增强 │    │ 打包   │           │
│  └──────┘    └─────────┘    └──────────┘    └────┬───┘           │
│                                                   │                │
│                                              ┌────▼───┐           │
│                                              │ 模型   │           │
│                                              │ 训练   │           │
│                                              └────────┘           │
│                                                                     │
│  常用数据增强 (torchvision.transforms):                              │
│  ┌─────────────────────────────────────────────┐                   │
│  │ RandomCrop | RandomHorizontalFlip | Resize  │                   │
│  │ Normalize | ToTensor | ColorJitter           │                   │
│  │ RandomRotation | GaussianBlur               │                   │
│  └─────────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

**DataLoader** 是PyTorch的数据加载管道，由两个核心组件组成：

1. **Dataset**: 定义如何获取单个数据样本
2. **DataLoader**: 将Dataset封装为可迭代的批次

**Dataset类型对比：**

| 类型 | 基类 | 需要实现 | 适用场景 |
|------|------|----------|----------|
| Map-style | `Dataset` | `__len__`, `__getitem__` | 可随机访问 |
| Iterable-style | `IterableDataset` | `__iter__` | 流式数据 |

**DataLoader关键参数：**

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `batch_size` | 每批次样本数 | 32-128 |
| `shuffle` | 是否打乱顺序 | 训练True, 验证False |
| `num_workers` | 数据加载进程数 | CPU核心数/2 |
| `pin_memory` | 固定内存(GPU训练) | True |
| `drop_last` | 是否丢弃最后不完整批次 | 训练True |
| `collate_fn` | 自定义批次打包函数 | 特殊数据格式时 |

### 代码示例

```python
# DataLoader - 完整示例
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, List, Dict, Any, Optional
import numpy as np


# ===================== 1. 自定义Dataset =====================

class CustomDataset(Dataset):
    """自定义数据集"""

    def __init__(self, data: np.ndarray, labels: np.ndarray,
                 transform=None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """获取单个样本"""
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


# ===================== 2. 图像数据集 =====================

class ImageFolderDataset(Dataset):
    """图像文件夹数据集"""

    def __init__(self, image_paths: List[str],
                 labels: List[int],
                 transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        from PIL import Image

        # 加载图像
        image = Image.open(self.image_paths[idx]).convert("RGB")

        # 应用变换
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


# ===================== 3. 数据变换 =====================

def get_transforms(train: bool = True):
    """获取数据变换"""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# ===================== 4. MNIST数据集加载 =====================

def load_mnist(batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """加载MNIST数据集"""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"MNIST 数据集:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    print(f"  批次大小: {batch_size}")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  测试批次数: {len(test_loader)}")

    # 查看一个批次
    images, labels = next(iter(train_loader))
    print(f"\n  批次图像形状: {images.shape}")  # (64, 1, 28, 28)
    print(f"  批次标签形状: {labels.shape}")    # (64,)

    return train_loader, test_loader


# ===================== 5. 数据集分割 =====================

def split_dataset_demo():
    """数据集分割演示"""
    print("\n" + "=" * 50)
    print("数据集分割")
    print("=" * 50)

    # 创建一个简单数据集
    data = np.random.randn(1000, 10)
    labels = np.random.randint(0, 3, size=1000)
    dataset = CustomDataset(data, labels)

    # 按比例分割: 80% 训练, 10% 验证, 10% 测试
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"数据集大小: {total}")
    print(f"训练集: {len(train_set)}")
    print(f"验证集: {len(val_set)}")
    print(f"测试集: {len(test_set)}")

    # 创建DataLoader
    train_loader = DataLoader(train_set, batch_size=32,
                              shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32,
                            shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32,
                             shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    split_dataset_demo()

    print("\n" + "=" * 50)
    print("MNIST数据集加载")
    print("=" * 50)
    train_loader, test_loader = load_mnist(batch_size=64)
```

---

## 完整项目：MNIST手写数字识别

### 核心概念

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MNIST手写数字识别 完整流程                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐    │
│  │ 数据 │─►│ 预处理│─►│ 模型 │─►│ 训练 │─►│ 评估 │─►│ 部署 │    │
│  │ 加载 │  │ 增强  │  │ 定义 │  │ 循环 │  │ 测试 │  │ 推理 │    │
│  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘    │
│                                                                     │
│  模型架构:                                                          │
│  ┌───────────────────────────────────────┐                         │
│  │  Input: (batch, 1, 28, 28)            │                         │
│  │      ↓                                │                         │
│  │  Conv2d(1→32, 3x3) + BN + ReLU       │                         │
│  │  MaxPool2d(2x2)     → (batch, 32, 14, 14)                      │
│  │      ↓                                │                         │
│  │  Conv2d(32→64, 3x3) + BN + ReLU      │                         │
│  │  MaxPool2d(2x2)     → (batch, 64, 7, 7)                        │
│  │      ↓                                │                         │
│  │  Flatten            → (batch, 3136)   │                         │
│  │      ↓                                │                         │
│  │  Linear(3136→128) + ReLU + Dropout    │                         │
│  │      ↓                                │                         │
│  │  Linear(128→10)                       │                         │
│  │      ↓                                │                         │
│  │  Output: (batch, 10)  → 10个类别      │                         │
│  └───────────────────────────────────────┘                         │
│                                                                     │
│  目标: 准确率 > 99%                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 详细说明

MNIST是深度学习入门的经典数据集，包含60,000张训练图像和10,000张测试图像，每张图像是28x28像素的灰度手写数字（0-9）。

本项目整合了前面所学的所有知识：Tensor操作、Autograd、nn.Module、训练循环和DataLoader，构建一个完整的端到端深度学习项目。

### 代码示例

```python
# 完整项目：MNIST手写数字识别
# 整合所有PyTorch基础知识的端到端项目

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
from typing import Dict, Tuple


# ===================== 1. 配置 =====================

class Config:
    """训练配置"""
    # 数据
    batch_size = 128
    num_workers = 2

    # 模型
    num_classes = 10
    dropout = 0.25

    # 训练
    epochs = 15
    learning_rate = 0.001
    weight_decay = 1e-4

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 保存
    save_path = "mnist_best.pth"


# ===================== 2. 数据准备 =====================

def prepare_data(config: Config) -> Tuple[DataLoader, DataLoader]:
    """准备数据"""
    # 训练集变换 (含数据增强)
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),           # 随机旋转
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 测试集变换 (仅标准化)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载和加载数据集
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True,
        download=True, transform=train_transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False,
        download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


# ===================== 3. 模型定义 =====================

class MNISTNet(nn.Module):
    """MNIST分类网络"""

    def __init__(self, num_classes: int = 10,
                 dropout: float = 0.25):
        super().__init__()

        # 卷积特征提取器
        self.features = nn.Sequential(
            # Block 1: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),        # 28x28 -> 14x14
            nn.Dropout2d(dropout),

            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),        # 14x14 -> 7x7
            nn.Dropout2d(dropout),
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ===================== 4. 训练和评估 =====================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ===================== 5. 完整训练流程 =====================

def train_mnist():
    """完整的MNIST训练流程"""
    config = Config()

    print("=" * 60)
    print("  MNIST 手写数字识别")
    print("=" * 60)
    print(f"  设备: {config.device}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  训练轮数: {config.epochs}")

    # 1. 数据准备
    print("\n[1/4] 准备数据...")
    train_loader, test_loader = prepare_data(config)
    print(f"  训练集: {len(train_loader.dataset)} 样本")
    print(f"  测试集: {len(test_loader.dataset)} 样本")

    # 2. 创建模型
    print("\n[2/4] 创建模型...")
    model = MNISTNet(
        num_classes=config.num_classes,
        dropout=config.dropout
    ).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")

    # 3. 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    # 4. 训练循环
    print("\n[3/4] 开始训练...")
    print("-" * 75)
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} "
          f"| {'Test Loss':>9} | {'Test Acc':>8} | {'LR':>10} "
          f"| {'Time':>5}")
    print("-" * 75)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device
        )

        # 评估
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, config.device
        )

        # 更新学习率
        scheduler.step()

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), config.save_path)

        # 打印进度
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.4f}% "
            f"| {test_loss:9.4f} | {test_acc:7.4f}% | "
            f"{current_lr:10.6f} | {epoch_time:4.1f}s"
        )

    total_time = time.time() - start_time
    print("-" * 75)
    print(f"\n训练完成!")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  最佳测试准确率: {best_acc:.4f}%")
    print(f"  最佳模型已保存到: {config.save_path}")

    # 5. 详细评估
    print("\n[4/4] 详细评估...")
    model.load_state_dict(torch.load(config.save_path))

    # 每个数字的准确率
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1

    print("\n  各数字识别准确率:")
    for i in range(10):
        acc = class_correct[i] / class_total[i] * 100
        print(f"    数字 {i}: {acc:6.2f}% "
              f"({class_correct[i]}/{class_total[i]})")

    return model


# ===================== 6. 推理函数 =====================

@torch.no_grad()
def predict(model: nn.Module, image: torch.Tensor,
            device: torch.device) -> Tuple[int, float]:
    """单张图像预测"""
    model.eval()
    image = image.to(device)

    if image.dim() == 3:
        image = image.unsqueeze(0)  # 增加batch维度

    output = model(image)
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted = probabilities.max(1)

    return predicted.item(), confidence.item()


# ===================== 主程序 =====================

if __name__ == "__main__":
    model = train_mnist()
```

---

## 进阶技巧

### 混合精度训练

```
┌─────────────────────────────────────────────────────────────────────┐
│                    混合精度训练 (AMP) 原理                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  FP32 vs FP16 vs BF16:                                             │
│  ┌──────────┬──────────┬──────────┬──────────────────────────┐     │
│  │  类型    │ 位数     │ 范围     │ 用途                     │     │
│  ├──────────┼──────────┼──────────┼──────────────────────────┤     │
│  │  FP32    │ 32 bit   │ 很大     │ 默认精度, 梯度累积      │     │
│  │  FP16    │ 16 bit   │ 较小     │ 前向/反向传播加速        │     │
│  │  BF16    │ 16 bit   │ 同FP32   │ A100/4090推荐           │     │
│  └──────────┴──────────┴──────────┴──────────────────────────┘     │
│                                                                     │
│  混合精度工作流程:                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │ 权重     │───►│ 前向传播 │───►│ 损失计算 │───►│ 反向传播 │     │
│  │ (FP32)   │    │ (FP16)   │    │ (FP32)   │    │ (FP16)   │     │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘     │
│       ▲                                               │            │
│       │           ┌──────────┐    ┌──────────┐        │            │
│       └───────────│ 参数更新 │◄───│ 梯度缩放 │◄───────┘            │
│                   │ (FP32)   │    │ (Scaler) │                     │
│                   └──────────┘    └──────────┘                     │
│                                                                     │
│  优势:                                                              │
│  - 显存减少 ~40% (激活值用FP16存储)                                 │
│  - 训练速度提升 ~1.5-2x (Tensor Core加速)                          │
│  - 精度损失极小 (通过GradScaler补偿)                                │
└─────────────────────────────────────────────────────────────────────┘
```

```python
# 混合精度训练完整示例
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler


class AMPTrainer:
    """混合精度训练器"""

    def __init__(self, model: nn.Module, lr: float = 1e-3):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scaler = GradScaler()  # 梯度缩放器
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, inputs, targets):
        """混合精度训练步"""
        self.optimizer.zero_grad()

        # 自动混合精度上下文
        with autocast():
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

        # 缩放损失并反向传播
        self.scaler.scale(loss).backward()

        # 反缩放梯度并更新
        self.scaler.step(self.optimizer)

        # 更新缩放因子
        self.scaler.update()

        return loss.item()


def amp_training_demo():
    """混合精度训练演示"""
    if not torch.cuda.is_available():
        print("混合精度训练需要CUDA GPU")
        return

    device = torch.device("cuda")

    # 创建模型
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)

    trainer = AMPTrainer(model)

    # 模拟数据
    x = torch.randn(128, 784, device=device)
    y = torch.randint(0, 10, (128,), device=device)

    # 训练
    for epoch in range(10):
        loss = trainer.train_step(x, y)
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}: loss={loss:.4f}")

    # 对比显存使用
    print(f"\n已分配显存: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"最大显存峰值: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")


# BF16训练 (适用于Ampere及以上架构)
def bf16_training_example():
    """BFloat16训练示例"""
    code = '''
    # BF16比FP16更安全: 指数位与FP32相同, 不需要GradScaler
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    '''
    return code


if __name__ == "__main__":
    amp_training_demo()
```

### 模型调试与性能分析

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PyTorch 调试与性能分析工具                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  常见问题诊断:                                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │ 问题              │ 症状           │ 解决方法            │      │
│  ├───────────────────┼────────────────┼─────────────────────┤      │
│  │ 梯度爆炸          │ loss=NaN/Inf   │ 梯度裁剪+降低LR    │      │
│  │ 梯度消失          │ loss不下降     │ 残差连接+BN/LN      │      │
│  │ 显存溢出(OOM)     │ CUDA OOM       │ 减batch/梯度检查点  │      │
│  │ 数据泄露          │ val_acc异常高  │ 检查数据划分        │      │
│  │ 过拟合            │ train好val差   │ Dropout/数据增强    │      │
│  │ 欠拟合            │ 都差           │ 增大模型/更多数据   │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│  性能分析工具链:                                                    │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐                │
│  │ PyTorch  │  │ torch.profiler│  │ NVIDIA Nsight│                │
│  │ Autograd │  │ 算子级分析   │  │ GPU级分析    │                │
│  │ Profiler │  │ TensorBoard  │  │ 系统级优化   │                │
│  └──────────┘  └──────────────┘  └──────────────┘                │
│                                                                     │
│  显存优化策略 (优先级排序):                                         │
│  1. 减小batch_size                                                 │
│  2. 使用混合精度训练 (AMP)                                         │
│  3. 梯度累积 (gradient accumulation)                                │
│  4. 梯度检查点 (gradient checkpointing)                             │
│  5. 模型并行 (model parallelism)                                    │
│  6. 数据并行 (DDP)                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

```python
# 模型调试与性能分析工具
import torch
import torch.nn as nn
import time


# ===================== 1. 梯度检查工具 =====================

class GradientMonitor:
    """梯度监控器 -- 检测梯度爆炸/消失"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.grad_history = {}

    def check_gradients(self) -> dict:
        """检查所有参数的梯度状态"""
        stats = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                stats[name] = {
                    "mean": grad.mean().item(),
                    "std": grad.std().item(),
                    "max": grad.max().item(),
                    "min": grad.min().item(),
                    "has_nan": bool(torch.isnan(grad).any()),
                    "has_inf": bool(torch.isinf(grad).any()),
                }
        return stats

    def print_gradient_report(self):
        """打印梯度报告"""
        stats = self.check_gradients()
        print("=" * 60)
        print("  梯度监控报告")
        print("=" * 60)

        for name, s in stats.items():
            status = "OK"
            if s["has_nan"]:
                status = "NaN!"
            elif s["has_inf"]:
                status = "Inf!"
            elif abs(s["max"]) > 100:
                status = "可能爆炸"
            elif abs(s["max"]) < 1e-7:
                status = "可能消失"

            print(f"  {name:30s} | max={s['max']:>10.4f} "
                  f"| mean={s['mean']:>10.6f} | {status}")


# ===================== 2. 显存监控 =====================

class MemoryTracker:
    """GPU显存追踪器"""

    @staticmethod
    def get_memory_info() -> dict:
        """获取当前显存使用情况"""
        if not torch.cuda.is_available():
            return {"status": "无GPU"}

        return {
            "已分配": f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB",
            "已缓存": f"{torch.cuda.memory_reserved() / 1024**2:.1f} MB",
            "最大分配": f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MB",
            "最大缓存": f"{torch.cuda.max_memory_reserved() / 1024**2:.1f} MB",
        }

    @staticmethod
    def estimate_model_memory(model: nn.Module) -> dict:
        """估算模型显存占用"""
        param_size = sum(p.numel() * p.element_size()
                        for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size()
                         for b in model.buffers())

        # 估算: 训练时需要 参数 + 梯度 + 优化器状态(2x for Adam)
        total_training = param_size * 4  # 参数+梯度+Adam(m+v)

        return {
            "参数大小": f"{param_size / 1024**2:.1f} MB",
            "缓冲区": f"{buffer_size / 1024**2:.1f} MB",
            "训练估算(含优化器)": f"{total_training / 1024**2:.1f} MB",
            "参数量": f"{sum(p.numel() for p in model.parameters()):,}",
        }


# ===================== 3. 梯度累积训练 =====================

def gradient_accumulation_training():
    """梯度累积 -- 用小显存模拟大batch"""
    print("=" * 60)
    print("  梯度累积训练示例")
    print("=" * 60)

    model = nn.Linear(100, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 等效batch_size = micro_batch * accumulation_steps = 4 * 8 = 32
    micro_batch_size = 4
    accumulation_steps = 8
    effective_batch = micro_batch_size * accumulation_steps

    print(f"微批次大小: {micro_batch_size}")
    print(f"累积步数: {accumulation_steps}")
    print(f"等效批次: {effective_batch}")

    # 训练循环
    for epoch in range(3):
        total_loss = 0
        optimizer.zero_grad()

        for step in range(accumulation_steps):
            x = torch.randn(micro_batch_size, 100)
            y = torch.randint(0, 10, (micro_batch_size,))

            outputs = model(x)
            loss = criterion(outputs, y) / accumulation_steps  # 除以累积步数
            loss.backward()

            total_loss += loss.item()

        # 累积完成后统一更新
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch+1}: loss={total_loss:.4f}")


# ===================== 4. 梯度检查点 =====================

def gradient_checkpointing_demo():
    """梯度检查点 -- 用计算换显存"""
    from torch.utils.checkpoint import checkpoint

    class LargeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                ) for _ in range(10)
            ])

        def forward(self, x, use_checkpoint=False):
            for layer in self.layers:
                if use_checkpoint:
                    # 不保存中间激活值, 反向传播时重新计算
                    x = checkpoint(layer, x, use_reentrant=False)
                else:
                    x = layer(x)
            return x

    model = LargeModel()

    print("=" * 50)
    print("梯度检查点演示")
    print("=" * 50)

    mem_info = MemoryTracker.estimate_model_memory(model)
    for k, v in mem_info.items():
        print(f"  {k}: {v}")

    print("\n使用梯度检查点可节省 ~60-70% 的激活值显存")
    print("代价是训练速度降低约 ~30% (需要重新计算)")


# ===================== 5. 模型参数统计 =====================

def count_parameters_by_layer(model: nn.Module):
    """按层统计参数量"""
    print("=" * 60)
    print("  模型参数分布")
    print("=" * 60)

    total = 0
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            print(f"  {name:40s} {params:>12,} 参数")
            total += params

    print(f"  {'总计':40s} {total:>12,} 参数")
    print(f"  模型大小 (FP32): {total * 4 / 1024**2:.1f} MB")
    print(f"  模型大小 (FP16): {total * 2 / 1024**2:.1f} MB")


if __name__ == "__main__":
    gradient_accumulation_training()
    gradient_checkpointing_demo()

    # 参数统计
    from torchvision.models import resnet18
    try:
        model = resnet18()
        count_parameters_by_layer(model)
    except ImportError:
        print("需要安装torchvision")
```

### 分布式训练基础

```
┌─────────────────────────────────────────────────────────────────────┐
│                    分布式训练策略对比                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 数据并行 (Data Parallel / DDP)                                  │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐                   │
│  │ GPU 0  │  │ GPU 1  │  │ GPU 2  │  │ GPU 3  │                   │
│  │ 模型副本│  │ 模型副本│  │ 模型副本│  │ 模型副本│                   │
│  │ 数据1/4│  │ 数据2/4│  │ 数据3/4│  │ 数据4/4│                   │
│  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘                   │
│      │           │           │           │                          │
│      └───────────┴─────┬─────┴───────────┘                          │
│                        │ AllReduce (梯度同步)                        │
│                        ▼                                             │
│                   参数更新 (每GPU独立更新)                            │
│                                                                     │
│  2. 模型并行 (Model Parallel / Tensor Parallel)                     │
│  ┌────────────────────────────────────────────┐                     │
│  │ Layer 1-8    │ Layer 9-16  │ Layer 17-24  │ Layer 25-32         │
│  │   GPU 0      │   GPU 1    │   GPU 2      │   GPU 3             │
│  └──────────────┴────────────┴──────────────┴─────────────┘        │
│  适用于: 单GPU放不下整个模型                                        │
│                                                                     │
│  3. FSDP (Fully Sharded Data Parallel)                              │
│  ┌──────────────────────────────────────────┐                       │
│  │ 每个GPU只存储 1/N 的参数+梯度+优化器状态 │                       │
│  │ 前向/反向传播时按需通信                  │                       │
│  │ 显存效率最高, PyTorch 2.0+ 原生支持      │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                     │
│  推荐选择:                                                          │
│  ┌──────────────────────────────────────────────────┐               │
│  │ 多GPU + 模型放得下单GPU → DDP                   │               │
│  │ 多GPU + 模型放不下单GPU → FSDP 或 DeepSpeed     │               │
│  │ 超大模型(70B+) → DeepSpeed ZeRO-3 或 FSDP       │               │
│  └──────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

```python
# 分布式训练基础 -- DDP示例
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import os


def setup_ddp(rank, world_size):
    """初始化分布式训练"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """清理分布式训练"""
    dist.destroy_process_group()


def ddp_train(rank, world_size):
    """DDP训练主函数 (每个GPU运行一份)"""
    setup_ddp(rank, world_size)

    # 创建模型 (每个GPU一份)
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(rank)

    # DDP包装
    model = DDP(model, device_ids=[rank])

    # 分布式数据采样器
    dataset = torch.utils.data.TensorDataset(
        torch.randn(1000, 784),
        torch.randint(0, 10, (1000,))
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    dataloader = DataLoader(
        dataset, batch_size=32, sampler=sampler
    )

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(5):
        sampler.set_epoch(epoch)  # 重要: 确保每个epoch的数据顺序不同

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()  # DDP自动同步梯度
            optimizer.step()

        if rank == 0:  # 只在主进程打印
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

    cleanup_ddp()


# 启动DDP训练:
# torchrun --nproc_per_node=4 train_ddp.py

# 或在脚本中:
# import torch.multiprocessing as mp
# mp.spawn(ddp_train, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    print("=" * 60)
    print("  分布式训练(DDP)代码模板")
    print("=" * 60)
    print("\n启动命令: torchrun --nproc_per_node=<GPU数> train.py")
    print("\n注意: DDP训练需要多个GPU, 此处仅展示代码结构")
```

---

## 总结

本教程涵盖了PyTorch基础教程的核心内容:

1. **PyTorch简介**: PyTorch是以动态计算图和Pythonic设计著称的深度学习框架，支持CPU、CUDA GPU、Apple MPS等多种硬件加速。

2. **Tensor操作**: Tensor是PyTorch的核心数据结构，支持丰富的创建方式、形状变换、数学运算、索引切片和GPU加速操作。

3. **Autograd自动求导**: 通过动态构建计算图，PyTorch能够自动计算任意复杂函数的梯度，是训练神经网络的基础。需要注意梯度累积问题和适时禁用梯度计算。

4. **nn.Module**: 所有神经网络的基类，通过继承并实现`__init__`和`forward`方法来定义自定义模型。支持参数管理、模型保存加载、设备迁移等功能。

5. **训练循环**: 标准五步流程（清零梯度、前向传播、计算损失、反向传播、更新参数）配合验证、学习率调度和模型保存构成完整训练流程。

6. **DataLoader**: 由Dataset和DataLoader组成的数据管道，支持自定义数据集、数据增强、多进程加载和批次管理。

7. **MNIST完整项目**: 整合所有知识构建端到端的手写数字识别系统，包含数据准备、模型定义（CNN）、训练循环、评估和推理的完整流程。

8. **进阶技巧**: 混合精度训练（AMP/BF16）提升速度节省显存、梯度累积模拟大batch、梯度检查点用计算换显存、模型调试工具链和分布式训练（DDP/FSDP）基础。

## 最佳实践

1. **始终设置随机种子**以保证实验可重复: `torch.manual_seed(42)`
2. **合理使用GPU**: 用`model.to(device)`和`data.to(device)`统一管理设备
3. **训练前必须清零梯度**: `optimizer.zero_grad()`
4. **推理时关闭梯度**: `with torch.no_grad():` 和 `model.eval()`
5. **使用BatchNorm和Dropout时注意train/eval模式切换**
6. **用`pin_memory=True`加速CPU到GPU的数据传输**
7. **保存模型用`state_dict()`而非直接保存模型对象**
8. **使用混合精度训练加速**: `torch.cuda.amp.autocast()`
9. **梯度累积模拟大batch**: `loss = loss / accumulation_steps`
10. **大模型训练用DDP**: `torchrun --nproc_per_node=N` 启动多GPU训练
11. **定期监控梯度**: 使用GradientMonitor检测NaN/Inf和梯度爆炸/消失

## 参考资源

- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [PyTorch 文档](https://pytorch.org/docs/stable/)
- [Deep Learning with PyTorch (官方书籍)](https://pytorch.org/deep-learning-with-pytorch)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [torchvision 模型库](https://pytorch.org/vision/stable/models.html)

---

**文件大小目标**: 30-35KB
**创建时间**: 2024-01-15
**最后更新**: 2024-01-15
