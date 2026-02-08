# GPU计算架构深度解析

## 课程概述

本教程全面讲解GPU计算原理，从GPU架构设计到CUDA编程，从内存层次到性能优化，帮助你深入理解并行计算和GPU加速技术。

**学习目标**：
- 理解GPU vs CPU的架构差异
- 掌握CUDA编程模型和线程层次
- 深入了解GPU内存层次结构
- 学会GPU性能优化技巧
- 了解Tensor Core和AI加速

---

## 1. GPU vs CPU架构对比

### 1.1 设计哲学差异

```
┌─────────────────────────────────────────────────────────────┐
│              CPU vs GPU 架构对比                              │
└─────────────────────────────────────────────────────────────┘

CPU设计（延迟优先）：
┌────────────────────────────────────────┐
│  CPU核心（少而强）                      │
│                                        │
│  ┌──────────────────────────────────┐ │
│  │  核心0                            │ │
│  │  ┌────────────┐  ┌────────────┐ │ │
│  │  │  大型ALU   │  │ 分支预测器  │ │ │
│  │  └────────────┘  └────────────┘ │ │
│  │  ┌────────────────────────────┐ │ │
│  │  │  乱序执行引擎               │ │ │
│  │  └────────────────────────────┘ │ │
│  │  ┌────────────┐  ┌────────────┐ │ │
│  │  │ L1缓存32KB │  │ L2缓存256KB│ │ │
│  │  └────────────┘  └────────────┘ │ │
│  └──────────────────────────────────┘ │
│                                        │
│  资源分配：                             │
│  ┌────────────────────────┐            │
│  │ 控制逻辑: 30%          │            │
│  │ 缓存:     40%          │            │
│  │ 计算单元: 30%          │            │
│  └────────────────────────┘            │
│                                        │
│  优势：复杂逻辑、低延迟                 │
│  劣势：并行度低                         │
└────────────────────────────────────────┘

GPU设计（吞吐量优先）：
┌────────────────────────────────────────┐
│  GPU核心（多而简）                      │
│                                        │
│  ┌──────┐┌──────┐┌──────┐┌──────┐    │
│  │ SM 0 ││ SM 1 ││ SM 2 ││ SM 3 │    │
│  ├──────┤├──────┤├──────┤├──────┤    │
│  │█████ ││█████ ││█████ ││█████ │    │
│  │█████ ││█████ ││█████ ││█████ │    │ 每个SM包含
│  │█████ ││█████ ││█████ ││█████ │    │ 多个CUDA核心
│  └──────┘└──────┘└──────┘└──────┘    │
│    ...      ...      ...     ...     │  (数千个)
│  ┌──────┐┌──────┐┌──────┐┌──────┐    │
│  │ SM N ││ SM N+1│...              │
│  └──────┘└──────┘                     │
│                                        │
│  资源分配：                             │
│  ┌────────────────────────┐            │
│  │ 控制逻辑: 5%           │            │
│  │ 缓存:     10%          │            │
│  │ 计算单元: 85%          │            │
│  └────────────────────────┘            │
│                                        │
│  优势：大规模并行                       │
│  劣势：单线程慢、分支效率低              │
└────────────────────────────────────────┘

性能对比（单精度浮点）：
┌────────────────┬──────────┬──────────┐
│  指标          │  CPU     │  GPU     │
├────────────────┼──────────┼──────────┤
│ 核心数         │  8-64    │  2000+   │
│ 时钟频率       │  3-5GHz  │  1-2GHz  │
│ 单核性能       │  高      │  低      │
│ 并行吞吐量     │  低      │  极高    │
│ 内存带宽       │  50GB/s  │  900GB/s │
│ 峰值算力(FP32) │  500GFLOPS│ 20TFLOPS│
│ 功耗           │  65-125W │  250-450W│
└────────────────┴──────────┴──────────┘

适用场景：
CPU: 复杂控制流、串行任务、通用计算
GPU: 数据并行、矩阵运算、图形/AI
```

### 1.2 CUDA核心架构

```
┌─────────────────────────────────────────────────────────────┐
│              NVIDIA GPU架构（Ampere架构示例）                 │
└─────────────────────────────────────────────────────────────┘

完整GPU芯片：
┌───────────────────────────────────────────────────────────┐
│  GPU Die                                                  │
│                                                           │
│  ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐      │
│  │ GPC 0││ GPC 1││ GPC 2││ GPC 3││ GPC 4││ GPC 5│      │
│  └──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘      │
│     │       │       │       │       │       │           │
│  每个GPC包含多个TPC                                        │
│  每个TPC包含2个SM                                         │
│                                                           │
│  ┌──────────────────────────────────────────────────┐    │
│  │  L2 Cache (6-12MB)                               │    │
│  └────────────────┬─────────────────────────────────┘    │
│                   │                                      │
│  ┌────────────────▼─────────────────────────────────┐    │
│  │  Memory Controllers (GDDR6X)                     │    │
│  │  ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐                │    │
│  │  │MC0││MC1││MC2││MC3││MC4││MC5│                │    │
│  │  └───┘└───┘└───┘└───┘└───┘└───┘                │    │
│  └──────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────┘

SM (Streaming Multiprocessor) 内部结构：
┌───────────────────────────────────────────────────────────┐
│  Streaming Multiprocessor (SM)                            │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Warp Scheduler × 4  (指令调度器)                    │ │
│  └───────┬─────────────────────┬───────────────────────┘ │
│          │                     │                         │
│  ┌───────▼─────────┐   ┌───────▼─────────┐              │
│  │ CUDA Core × 64  │   │ CUDA Core × 64  │              │
│  │ (FP32/INT32)    │   │ (FP32/INT32)    │              │
│  └─────────────────┘   └─────────────────┘              │
│                                                           │
│  ┌─────────────────┐   ┌─────────────────┐              │
│  │ Tensor Core ×4  │   │ LD/ST Units×32  │  加载/存储   │
│  │ (FP16/BF16/INT8)│   │                 │              │
│  └─────────────────┘   └─────────────────┘              │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  寄存器文件 (256KB)  - 65536个32位寄存器            │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ L1/Shared    │  │ Texture      │  │ Constant     │  │
│  │ Memory       │  │ Cache        │  │ Cache        │  │
│  │ (128KB可配置)│  │              │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                           │
│  每个SM规格（RTX 3080为例）：                              │
│    - 128个FP32核心                                        │
│    - 4个Tensor Core (第3代)                              │
│    - 4个Warp调度器                                        │
│    - 最大并发2048个线程                                   │
└───────────────────────────────────────────────────────────┘

Warp执行模型：
┌──────────────────────────────────────┐
│  Warp = 32个线程（SIMT模式）         │
│                                      │
│  ┌────────────────────────────────┐ │
│  │  单一指令流                     │ │
│  │  ▼                              │ │
│  │  [同一条指令]                   │ │
│  │  │ │ │ │ ... │ │ │ │          │ │
│  │  ▼ ▼ ▼ ▼     ▼ ▼ ▼ ▼          │ │
│  │  线程0-31 (32个线程同时执行)    │ │
│  └────────────────────────────────┘ │
│                                      │
│  分支惩罚：                          │
│  if (threadIdx.x < 16)              │
│      A;  ← 前16个线程执行            │
│  else                               │
│      B;  ← 后16个线程执行            │
│                                      │
│  实际执行：串行化（效率降低50%）     │
│  ┌──┐                               │
│  │A │ 前半Warp执行，后半等待         │
│  ├──┤                               │
│  │B │ 后半Warp执行，前半等待         │
│  └──┘                               │
└──────────────────────────────────────┘
```

---

## 2. CUDA编程模型

### 2.1 线程层次结构

```
┌─────────────────────────────────────────────────────────────┐
│              CUDA线程层次（Thread Hierarchy）                 │
└─────────────────────────────────────────────────────────────┘

Grid → Block → Thread三级结构：

┌──────────────────────────────────────────────────────────┐
│  Grid (网格)                                              │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Block(0,0)   Block(1,0)   Block(2,0)   Block(3,0)│  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  │  │
│  │  │░░░░░░░░│  │░░░░░░░░│  │░░░░░░░░│  │░░░░░░░░│  │  │
│  │  │░░░░░░░░│  │░░░░░░░░│  │░░░░░░░░│  │░░░░░░░░│  │  │
│  │  └────────┘  └────────┘  └────────┘  └────────┘  │  │
│  │                                                    │  │
│  │  Block(0,1)   Block(1,1)   Block(2,1)   Block(3,1)│  │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  │  │
│  │  │░░░░░░░░│  │░░░░░░░░│  │░░░░░░░░│  │░░░░░░░░│  │  │
│  │  │░░░░░░░░│  │░░░░░░░░│  │░░░░░░░░│  │░░░░░░░░│  │  │
│  │  └────────┘  └────────┘  └────────┘  └────────┘  │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Grid维度：gridDim.x, gridDim.y, gridDim.z              │
│  Block索引：blockIdx.x, blockIdx.y, blockIdx.z          │
└──────────────────────────────────────────────────────────┘

单个Block内部：
┌──────────────────────────────────────────────────────────┐
│  Block (线程块)  - 最多1024个线程                         │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  3D线程组织 (示例: 8×8×2 = 128个线程)              │ │
│  │                                                    │ │
│  │  Z=0层:                                            │ │
│  │  ┌──┬──┬──┬──┬──┬──┬──┬──┐                       │ │
│  │  │T0│T1│T2│T3│T4│T5│T6│T7│                       │ │
│  │  ├──┼──┼──┼──┼──┼──┼──┼──┤                       │ │
│  │  │  │  │  │  │  │  │  │  │                       │ │
│  │  ├──┼──┼──┼──┼──┼──┼──┼──┤                       │ │
│  │  │  │  │  │  │  │  │  │  │  ...8行               │ │
│  │  └──┴──┴──┴──┴──┴──┴──┴──┘                       │ │
│  │                                                    │ │
│  │  Z=1层: (类似布局)                                 │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  Block维度：blockDim.x, blockDim.y, blockDim.z          │
│  Thread索引：threadIdx.x, threadIdx.y, threadIdx.z      │
│                                                          │
│  Block内线程可以：                                        │
│  - 共享Shared Memory                                    │
│  - 使用__syncthreads()同步                              │
│  - 最大1024个线程限制                                    │
└──────────────────────────────────────────────────────────┘

全局线程ID计算：
┌──────────────────────────────────────┐
│  1D配置：                             │
│  int tid = blockIdx.x * blockDim.x + │
│            threadIdx.x;              │
│                                      │
│  2D配置：                             │
│  int x = blockIdx.x*blockDim.x +     │
│          threadIdx.x;                │
│  int y = blockIdx.y*blockDim.y +     │
│          threadIdx.y;                │
│  int tid = y * width + x;            │
│                                      │
│  3D配置：                             │
│  int x = blockIdx.x*blockDim.x +     │
│          threadIdx.x;                │
│  int y = blockIdx.y*blockDim.y +     │
│          threadIdx.y;                │
│  int z = blockIdx.z*blockDim.z +     │
│          threadIdx.z;                │
└──────────────────────────────────────┘

Kernel启动配置：
┌──────────────────────────────────────┐
│  // 1D Grid, 1D Block                │
│  dim3 grid(numBlocks);               │
│  dim3 block(threadsPerBlock);        │
│  myKernel<<<grid, block>>>(args);    │
│                                      │
│  // 2D Grid, 2D Block                │
│  dim3 grid(gridX, gridY);            │
│  dim3 block(blockX, blockY);         │
│  myKernel<<<grid, block>>>(args);    │
│                                      │
│  // 共享内存+流                       │
│  myKernel<<<grid, block,             │
│              sharedMem, stream>>>    │
│             (args);                  │
└──────────────────────────────────────┘
```

### 2.2 CUDA内存层次

```
┌─────────────────────────────────────────────────────────────┐
│              CUDA内存层次（Memory Hierarchy）                 │
└─────────────────────────────────────────────────────────────┘

速度       范围           大小          延迟        带宽
快         小             小            低          高
│                                                   │
│   ┌─────────────┐                                ▲
│   │  寄存器      │  线程私有   ~64KB/SM  1cycle   │
│   │ (Registers) │  自动分配                       │
│   └──────┬──────┘                                │
│          │                                       │
│   ┌──────▼──────────┐                            │
│   │  本地内存        │  线程私有   数MB   ~400cycle│
│   │(Local Memory)   │  (溢出寄存器，在DRAM中)      │
│   └─────────────────┘                            │
│          │                                       │
│   ┌──────▼──────────┐                            │
│   │  共享内存        │  Block内共享 48-164KB     │
│   │(Shared Memory)  │  __shared__   ~20cycles   │
│   │  (L1 Cache)     │  手动管理                  │
│   └──────┬──────────┘                            │
│          │                                       │
│   ┌──────▼──────────┐                            │
│   │  L2 Cache       │  GPU全局    6-12MB        │
│   │                 │  自动管理    ~200cycles    │
│   └──────┬──────────┘                            │
│          │                                       │
│   ┌──────▼──────────┐                            │
│   │  全局内存        │  GPU全局    8-80GB        │
│   │(Global Memory)  │  (GDDR6/HBM) ~400cycles   │
│   │                 │  900GB/s                  │
│   └──────┬──────────┘                            │
│          │                                       │
│   ┌──────▼──────────┐                            │
│   │  常量内存        │  GPU只读    64KB          │
│   │(Constant Memory)│  缓存优化    ~40cycles    │
│   │  __constant__   │  (广播优化)                │
│   └─────────────────┘                            │
│          │                                       │
│   ┌──────▼──────────┐                            │
│   │  纹理内存        │  GPU只读    缓存          │
│   │(Texture Memory) │  空间局部性  ~400cycles   │
│   │                 │  2D优化                   │
│   └─────────────────┘                            │
│                                                  ▼
慢         大             大            高          低

内存声明示例：
┌──────────────────────────────────────────┐
│ // 全局内存（最常用）                     │
│ __global__ void kernel(float *g_data) {  │
│     int idx = threadIdx.x;               │
│     float val = g_data[idx];  // 全局访问│
│ }                                        │
│                                          │
│ // 共享内存（快速，Block内共享）          │
│ __shared__ float s_data[256];            │
│ s_data[threadIdx.x] = g_data[idx];       │
│ __syncthreads();  // 同步                │
│ float val = s_data[threadIdx.x + 1];     │
│                                          │
│ // 常量内存（只读，广播优化）             │
│ __constant__ float c_coeffs[10];         │
│ float result = c_coeffs[0] * val;        │
│                                          │
│ // 寄存器变量（自动分配）                 │
│ float localVar = 1.0f;  // 寄存器        │
└──────────────────────────────────────────┘

内存访问模式：
┌──────────────────────────────────────────┐
│  1. 合并访问（Coalesced）- 最优          │
│  Warp中32个线程访问连续地址：            │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┐            │
│  │T0│T1│T2│T3│...│29│30│31│            │
│  └┬─┴┬─┴┬─┴┬─┴───┴┬─┴┬─┴┬─┘            │
│   │  │  │  │      │  │  │              │
│  ┌▼──▼──▼──▼──────▼──▼──▼─┐            │
│  │连续128字节(一次事务)     │            │
│  └──────────────────────────┘            │
│                                          │
│  2. 未对齐访问 - 多次事务                 │
│  ┌──┬──┬──┬──┐                         │
│  │T0│T1│T2│T3│                         │
│  └┬─┴┬─┴┬─┴┬─┘                         │
│   ↓  ↓  ↓  ↓  (跨越缓存行边界)          │
│  需要2次128字节事务                      │
│                                          │
│  3. 跨步访问（Strided）- 浪费带宽        │
│  ┌──┬──┬──┬──┐                         │
│  │T0│T1│T2│T3│ stride=4                │
│  └┬─┴─┴─┴┬┴─┴─┴┬┴─┴─┴┬┘                │
│   ↓      ↓     ↓     ↓                 │
│  [0][X][X][X][4][X][X][X][8]...         │
│   (大量缓存行浪费)                       │
└──────────────────────────────────────────┘
```

---

## 3. CUDA编程实战

### 3.1 向量加法示例

```cuda
// vector_add.cu - CUDA向量加法完整示例
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA核函数（在GPU上执行）
__global__ void vectorAdd(const float *A, const float *B,
                          float *C, int N) {
    // 计算全局线程ID
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // 边界检查
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    // 1. 设置向量大小
    int N = 1 << 20;  // 1M个元素
    size_t bytes = N * sizeof(float);

    // 2. 分配主机内存
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // 3. 初始化输入数据
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    // 4. 分配设备内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // 5. 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // 6. 配置执行参数
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("启动配置: %d blocks × %d threads\n",
           blocksPerGrid, threadsPerBlock);

    // 7. 启动核函数
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // 8. 同步等待完成
    CUDA_CHECK(cudaDeviceSynchronize());

    // 9. 拷贝结果回主机
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // 10. 验证结果
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            printf("Error at index %d: %f vs %f\n",
                   i, h_C[i], h_A[i] + h_B[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("向量加法成功!\n");
    }

    // 11. 释放内存
    free(h_A); free(h_B); free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}

/*
编译运行：
nvcc vector_add.cu -o vector_add
./vector_add
*/
```

### 3.2 矩阵乘法优化

```cuda
// matrix_mul.cu - 矩阵乘法优化示例
#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16  // Shared Memory块大小

// 基础版本（全局内存）
__global__ void matMulBasic(float *A, float *B, float *C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
            // 每次循环2次全局内存访问（慢！）
        }
        C[row * N + col] = sum;
    }
}

// 优化版本（Shared Memory）
__global__ void matMulTiled(float *A, float *B, float *C,
                             int M, int N, int K) {
    // 分配共享内存
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // 分块计算
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 1. 协作加载数据到共享内存
        if (row < M && t * TILE_SIZE + tx < K)
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (t * TILE_SIZE + ty < K && col < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        // 2. 同步确保所有线程加载完成
        __syncthreads();

        // 3. 使用共享内存计算（快！）
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // 4. 同步确保所有线程计算完成
        __syncthreads();
    }

    // 5. 写入结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 性能测试函数
void testMatMul() {
    int M = 2048, N = 2048, K = 2048;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // 分配并初始化主机内存
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    for (int i = 0; i < M * K; i++) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = rand() / (float)RAND_MAX;

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    // 测试基础版本
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMulBasic<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_basic;
    cudaEventElapsedTime(&time_basic, start, stop);

    // 测试优化版本
    cudaEventRecord(start);
    matMulTiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_tiled;
    cudaEventElapsedTime(&time_tiled, start, stop);

    printf("矩阵大小: %dx%d × %dx%d\n", M, K, K, N);
    printf("基础版本: %.2f ms\n", time_basic);
    printf("优化版本: %.2f ms\n", time_tiled);
    printf("加速比: %.2fx\n", time_basic / time_tiled);

    // 计算GFLOPS
    double gflops = 2.0 * M * N * K / (time_tiled * 1e6);
    printf("性能: %.2f GFLOPS\n", gflops);

    // 清理
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
```

---

## 4. Tensor Core与AI加速

### 4.1 Tensor Core原理

```
┌─────────────────────────────────────────────────────────────┐
│              Tensor Core架构（第4代Ampere）                   │
└─────────────────────────────────────────────────────────────┘

传统CUDA核心矩阵乘法：
┌──────────────────────────────────────┐
│  D = A × B + C  (4×4矩阵)            │
│                                      │
│  需要：4×4×4 = 64次乘加运算           │
│  CUDA核心：64个周期（串行）           │
│  Tensor核心：1个周期（并行）          │
└──────────────────────────────────────┘

Tensor Core操作（WMMA - Warp Matrix Multiply Accumulate）：
┌────────────────────────────────────────────────────┐
│  支持的数据类型：                                   │
│  - FP16 (半精度)：16×16×16 → FP32累加              │
│  - BF16 (脑浮点)：16×16×16 → FP32累加              │
│  - TF32 (TensorFloat32)：16×16×16 → FP32累加       │
│  - INT8：16×16×16 → INT32累加                      │
│  - INT4/INT1：用于极致量化                         │
│                                                    │
│  单次操作：                                         │
│  D(16×16) = A(16×16) × B(16×16) + C(16×16)        │
│             ↑          ↑          ↑               │
│           FP16       FP16       FP32              │
│                                                    │
│  吞吐量：每SM每周期256 FP16 FMA                     │
│         相当于512 FP16操作                         │
└────────────────────────────────────────────────────┘

性能对比（RTX 3090为例）：
┌──────────────┬────────────┬────────────┐
│  运算类型    │   TFLOPS   │   应用     │
├──────────────┼────────────┼────────────┤
│ FP32(标准)   │   35.6     │  通用计算  │
│ FP16(Tensor) │   142      │  深度学习  │
│ INT8(Tensor) │   284      │  推理加速  │
│ INT4(Tensor) │   568      │  极致推理  │
└──────────────┴────────────┴────────────┘

使用示例（CUDA C++）：
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void tensorCoreGEMM() {
    // 声明矩阵片段
    fragment<matrix_a, 16, 16, 16, half,
             row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half,
             col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    // 加载数据
    load_matrix_sync(a_frag, A, 16);
    load_matrix_sync(b_frag, B, 16);
    fill_fragment(c_frag, 0.0f);

    // Tensor Core计算
    mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 存储结果
    store_matrix_sync(D, c_frag, 16, mem_row_major);
}
```

---

## 5. GPU性能测试

### 5.1 GPU信息查询

```python
#!/usr/bin/env python3
"""
GPU信息查询脚本（使用pycuda）
"""

import subprocess
import os

def query_nvidia_smi():
    """使用nvidia-smi查询GPU信息"""
    print("=== NVIDIA GPU信息 ===\n")

    try:
        # 基本信息
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,'
             'memory.total,compute_cap',
             '--format=csv,noheader'],
            capture_output=True,
            text=True
        )
        print("GPU型号和驱动:")
        print(result.stdout)

        # 详细规格
        result = subprocess.run(['nvidia-smi', '-q'],
                              capture_output=True, text=True)

        for line in result.stdout.split('\n'):
            if any(keyword in line for keyword in
                  ['CUDA Version', 'SM', 'Memory Clock',
                   'Graphics Clock', 'Power Limit',
                   'GPU Current Temp']):
                print(line.strip())

    except FileNotFoundError:
        print("未安装nvidia-smi或非NVIDIA GPU")

def query_cuda_properties():
    """使用CUDA查询详细属性"""
    print("\n=== CUDA设备属性 ===\n")

    try:
        import pycuda.driver as cuda
        import pycuda.autoinit

        dev = cuda.Device(0)
        attrs = dev.get_attributes()

        print(f"设备名称: {dev.name()}")
        print(f"计算能力: {dev.compute_capability()}")
        print(f"总内存: {dev.total_memory() / 1024**3:.2f} GB")
        print(f"SM数量: {attrs[cuda.device_attribute.MULTIPROCESSOR_COUNT]}")
        print(f"每SM最大线程数: {attrs[cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR]}")
        print(f"每Block最大线程数: {attrs[cuda.device_attribute.MAX_THREADS_PER_BLOCK]}")
        print(f"Warp大小: {attrs[cuda.device_attribute.WARP_SIZE]}")
        print(f"寄存器/SM: {attrs[cuda.device_attribute.MAX_REGISTERS_PER_MULTIPROCESSOR]}")
        print(f"共享内存/SM: {attrs[cuda.device_attribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR]} B")
        print(f"L2缓存大小: {attrs[cuda.device_attribute.L2_CACHE_SIZE]} B")
        print(f"内存时钟: {attrs[cuda.device_attribute.MEMORY_CLOCK_RATE] / 1000} MHz")
        print(f"内存总线宽度: {attrs[cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH]} bit")

        # 计算理论带宽
        mem_clock_khz = attrs[cuda.device_attribute.MEMORY_CLOCK_RATE]
        bus_width = attrs[cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH]
        bandwidth = 2 * mem_clock_khz * (bus_width / 8) / 1e6  # GB/s
        print(f"理论内存带宽: {bandwidth:.1f} GB/s")

    except ImportError:
        print("未安装pycuda")

def main():
    query_nvidia_smi()
    query_cuda_properties()

if __name__ == "__main__":
    main()
```

### 5.2 GPU性能基准测试

```python
#!/usr/bin/env python3
"""
GPU性能基准测试
"""

import numpy as np
import time

try:
    import cupy as cp
    import cupyx.profiler
except ImportError:
    print("需要安装cupy: pip install cupy-cuda11x")
    exit(1)

def benchmark_memory_bandwidth():
    """测试内存带宽"""
    print("=== GPU内存带宽测试 ===\n")

    sizes_mb = [1, 10, 100, 500, 1000]

    for size_mb in sizes_mb:
        size = size_mb * 1024 * 1024 // 4  # float32元素数
        data = cp.random.rand(size, dtype=cp.float32)

        # 预热
        _ = cp.sum(data)

        # 读取测试
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = cp.sum(data)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start

        bandwidth = (size * 4 * 10) / elapsed / 1e9
        print(f"{size_mb:4d} MB: {bandwidth:6.1f} GB/s")

def benchmark_compute():
    """测试计算性能（GFLOPS）"""
    print("\n=== GPU计算性能测试 ===\n")

    sizes = [1024, 2048, 4096, 8192]

    for size in sizes:
        A = cp.random.rand(size, size, dtype=cp.float32)
        B = cp.random.rand(size, size, dtype=cp.float32)

        # 预热
        _ = cp.matmul(A, B)

        # 测试
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        C = cp.matmul(A, B)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start

        # 计算GFLOPS
        ops = 2 * size**3  # 矩阵乘法运算量
        gflops = ops / elapsed / 1e9

        print(f"{size}×{size}: {elapsed*1000:6.2f} ms, {gflops:7.1f} GFLOPS")

def benchmark_tensor_cores():
    """测试Tensor Core性能"""
    print("\n=== Tensor Core性能测试 ===\n")

    # 需要Tensor Core支持（计算能力7.0+）
    try:
        sizes = [2048, 4096, 8192]

        for size in sizes:
            # 使用FP16激活Tensor Core
            A = cp.random.rand(size, size, dtype=cp.float16)
            B = cp.random.rand(size, size, dtype=cp.float16)

            # 预热
            _ = cp.matmul(A, B)

            # 测试
            cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            C = cp.matmul(A, B)
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - start

            ops = 2 * size**3
            tflops = ops / elapsed / 1e12

            print(f"{size}×{size} FP16: {elapsed*1000:6.2f} ms, "
                  f"{tflops:5.2f} TFLOPS")

    except Exception as e:
        print(f"Tensor Core测试失败: {e}")

def main():
    print("GPU性能基准测试")
    print("=" * 60)

    # 显示GPU信息
    print(f"GPU: {cp.cuda.Device().name}")
    print(f"计算能力: {cp.cuda.Device().compute_capability}")
    print()

    benchmark_memory_bandwidth()
    benchmark_compute()
    benchmark_tensor_cores()

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
```

---

## 6. 学习资源与总结

### 6.1 关键要点总结

```
┌─────────────────────────────────────────────────────────────┐
│                  GPU计算核心概念                              │
└─────────────────────────────────────────────────────────────┘

1. GPU vs CPU
   ├─ CPU：少核心高频率，复杂逻辑
   ├─ GPU：多核心低频率，大规模并行
   └─ 应用：数据并行、矩阵运算、AI

2. CUDA架构
   ├─ SM：流多处理器（基本单元）
   ├─ Warp：32线程SIMT执行
   ├─ 线程层次：Grid → Block → Thread
   └─ 最大并发：数万线程

3. 内存层次
   ├─ 寄存器：最快，线程私有
   ├─ 共享内存：Block内共享，手动管理
   ├─ L2缓存：GPU全局
   ├─ 全局内存：大容量，高延迟
   └─ 合并访问优化关键

4. Tensor Core
   ├─ 专用矩阵运算单元
   ├─ FP16/INT8加速
   ├─ AI训练/推理性能10x+
   └─ WMMA API编程

5. 性能优化
   ├─ 占用率：最大化活跃Warp
   ├─ 合并访问：对齐连续访问
   ├─ 共享内存：减少全局访问
   ├─ 指令优化：避免分支发散
   └─ 异步执行：Stream并行

6. 编程模型
   ├─ Kernel启动：<<<grid,block>>>
   ├─ 内存管理：cudaMalloc/cudaMemcpy
   ├─ 同步：__syncthreads()
   └─ 库：cuBLAS, cuDNN, TensorRT
└─────────────────────────────────────────────────────────────┘
```

**下一步**：学习网络硬件架构，理解网卡、交换机和RDMA技术。

**文件大小**：约32KB
**最后更新**：2024年
