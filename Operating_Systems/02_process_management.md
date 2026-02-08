# 进程管理

## 课程概述

本教程深入讲解操作系统的进程管理机制,从进程的基本概念到CPU调度算法,从进程间通信到线程同步,帮助你全面掌握进程管理的核心技术。

**学习目标**:
- 理解进程与线程的本质区别
- 掌握进程状态转换与生命周期
- 深入了解CPU调度算法的设计与权衡
- 掌握各种进程间通信机制
- 理解死锁的产生与预防
- 学会协程与用户态调度

---

## 1. 进程基本概念

### 1.1 进程定义

```
┌─────────────────────────────────────────────────────────────┐
│              进程（Process）的本质                            │
└─────────────────────────────────────────────────────────────┘

进程 = 程序 + 执行状态 + 系统资源

程序（Program）                  进程（Process)
┌────────────────┐              ┌─────────────────────────────┐
│  静态代码文件   │              │  运行中的程序实例            │
│  存储在磁盘    │  执行         │                             │
│  .text段       │  ────▶       │  ┌────────────────────────┐ │
│  .data段       │              │  │ 程序代码(.text)         │ │
│  .bss段        │              │  ├────────────────────────┤ │
└────────────────┘              │  │ 初始化数据(.data)       │ │
                                │  ├────────────────────────┤ │
                                │  │ 未初始化数据(.bss)      │ │
                                │  ├────────────────────────┤ │
                                │  │ 堆(Heap) ↓             │ │
                                │  │                        │ │
                                │  │         ...            │ │
                                │  │                        │ │
                                │  │ 栈(Stack) ↑            │ │
                                │  └────────────────────────┘ │
                                │                             │
                                │  执行状态:                   │
                                │  - PC(程序计数器)           │
                                │  - 寄存器值                 │
                                │  - CPU状态字                │
                                │                             │
                                │  系统资源:                   │
                                │  - 打开的文件               │
                                │  - 信号处理表               │
                                │  - 内存映射                 │
                                │  - 网络连接                 │
                                └─────────────────────────────┘

进程控制块(PCB - Process Control Block):
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  进程标识信息                                             │
│  ┌────────────────────────────────────┐                 │
│  │ PID: 进程ID                         │                 │
│  │ PPID: 父进程ID                      │                 │
│  │ UID: 用户ID                         │                 │
│  │ GID: 组ID                           │                 │
│  └────────────────────────────────────┘                 │
│                                                          │
│  处理器状态信息                                           │
│  ┌────────────────────────────────────┐                 │
│  │ 程序计数器(PC)                      │                 │
│  │ 栈指针(SP)                          │                 │
│  │ 通用寄存器                          │                 │
│  │ PSW(程序状态字)                     │                 │
│  └────────────────────────────────────┘                 │
│                                                          │
│  进程调度信息                                             │
│  ┌────────────────────────────────────┐                 │
│  │ 进程状态(运行/就绪/阻塞)            │                 │
│  │ 优先级                              │                 │
│  │ 调度参数                            │                 │
│  │ CPU时间片                           │                 │
│  └────────────────────────────────────┘                 │
│                                                          │
│  内存管理信息                                             │
│  ┌────────────────────────────────────┐                 │
│  │ 页表指针                            │                 │
│  │ 代码段起始地址                      │                 │
│  │ 数据段起始地址                      │                 │
│  │ 堆栈段起始地址                      │                 │
│  └────────────────────────────────────┘                 │
│                                                          │
│  文件管理信息                                             │
│  ┌────────────────────────────────────┐                 │
│  │ 打开文件描述符表                    │                 │
│  │ 当前工作目录                        │                 │
│  │ 根目录                              │                 │
│  └────────────────────────────────────┘                 │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 1.2 进程与线程对比

```
┌─────────────────────────────────────────────────────────────┐
│              进程 vs 线程                                     │
└─────────────────────────────────────────────────────────────┘

单线程进程:
┌───────────────────────────────────────┐
│         进程A                          │
│  ┌─────────────────────────────────┐  │
│  │  代码段(共享)                    │  │
│  ├─────────────────────────────────┤  │
│  │  数据段                          │  │
│  ├─────────────────────────────────┤  │
│  │  堆                              │  │
│  ├─────────────────────────────────┤  │
│  │  栈                              │  │
│  └─────────────────────────────────┘  │
│                                       │
│  ┌─────────────────────────────────┐  │
│  │  PCB(进程控制块)                 │  │
│  │  - 进程状态                      │  │
│  │  - PC/寄存器                     │  │
│  │  - 打开文件                      │  │
│  └─────────────────────────────────┘  │
└───────────────────────────────────────┘

多线程进程:
┌───────────────────────────────────────┐
│         进程B                          │
│  ┌─────────────────────────────────┐  │
│  │  代码段(所有线程共享)            │  │
│  ├─────────────────────────────────┤  │
│  │  数据段(所有线程共享)            │  │
│  ├─────────────────────────────────┤  │
│  │  堆(所有线程共享)                │  │
│  ├─────────────────────────────────┤  │
│  │  线程1栈  │ 线程2栈  │ 线程3栈  │  │
│  └─────────────────────────────────┘  │
│                                       │
│  ┌──────────┐┌──────────┐┌──────────┐│
│  │ TCB1     ││ TCB2     ││ TCB3     ││
│  │ PC/寄存器││ PC/寄存器││ PC/寄存器││
│  │ 线程状态 ││ 线程状态 ││ 线程状态 ││
│  └──────────┘└──────────┘└──────────┘│
│                                       │
│  ┌─────────────────────────────────┐  │
│  │  PCB(共享)                       │  │
│  │  - 打开文件                      │  │
│  │  - 信号处理                      │  │
│  │  - 内存映射                      │  │
│  └─────────────────────────────────┘  │
└───────────────────────────────────────┘

对比表:
┌──────────┬─────────────────┬─────────────────┐
│ 特性     │     进程         │     线程         │
├──────────┼─────────────────┼─────────────────┤
│ 定义     │ 资源分配单位     │ CPU调度单位      │
│ 地址空间 │ 独立             │ 共享             │
│ 资源     │ 独立拥有         │ 共享进程资源     │
│ 通信     │ IPC(复杂)        │ 共享内存(简单)   │
│ 开销     │ 大(~10ms)        │ 小(~100μs)       │
│ 切换     │ 页表切换         │ 无需切换页表     │
│ 数据保护 │ 强(隔离)         │ 弱(需同步)       │
│ 崩溃影响 │ 独立             │ 整个进程崩溃     │
└──────────┴─────────────────┴─────────────────┘

使用场景:
进程: 独立应用、安全隔离、浏览器标签页
线程: 并行计算、I/O并发、Web服务器请求处理
```

---

## 2. 进程状态与转换

### 2.1 五状态模型

```
┌─────────────────────────────────────────────────────────────┐
│              进程状态转换图（5-State Model）                  │
└─────────────────────────────────────────────────────────────┘

                    创建进程
                       │
                       ▼
              ┌──────────────┐
              │  新建(New)   │  进程刚创建，尚未进入就绪队列
              └──────┬───────┘
                     │ 准入(Admit)
                     │
                     ▼
              ┌──────────────┐
              │  就绪(Ready) │  等待CPU，一切准备就绪
              └──────┬───────┘
                     │ 调度(Dispatch)
                     │
                     ▼
              ┌──────────────┐
         ┌────│ 运行(Running)│────┐  正在CPU上执行
         │    └──────┬───────┘    │
         │           │            │
    超时  │           │ I/O请求    │  退出
  (Timeout)│         │ 或等待事件 │ (Exit)
         │           │            │
         │           ▼            ▼
         │    ┌──────────────┐  ┌──────────────┐
         │    │ 阻塞(Blocked)│  │ 终止(Terminated)│
         │    └──────┬───────┘  └──────────────┘
         │           │                   进程结束
         │           │ I/O完成或
         │           │ 事件发生
         │           │
         │           ▼
         │    ┌──────────────┐
         └───▶│  就绪(Ready) │
              └──────────────┘

状态详细说明:

1. 新建(New)
   ┌────────────────────────────────────┐
   │ • 进程正在被创建                    │
   │ • 分配PCB                          │
   │ • 尚未加载到内存                   │
   │ • fork()刚返回                     │
   └────────────────────────────────────┘

2. 就绪(Ready)
   ┌────────────────────────────────────┐
   │ • 万事俱备，只欠CPU                │
   │ • 在就绪队列中等待                 │
   │ • 可以被调度执行                   │
   │ • 所需资源已分配                   │
   └────────────────────────────────────┘

3. 运行(Running)
   ┌────────────────────────────────────┐
   │ • 正在CPU上执行指令                │
   │ • 单核系统同时只有一个              │
   │ • 多核系统可以有多个               │
   │ • 占用CPU时间片                    │
   └────────────────────────────────────┘

4. 阻塞(Blocked/Waiting)
   ┌────────────────────────────────────┐
   │ • 等待某个事件发生                 │
   │ • 等待I/O操作完成                  │
   │ • 等待信号量                       │
   │ • 等待消息到达                     │
   │ • sleep()调用                      │
   └────────────────────────────────────┘

5. 终止(Terminated/Zombie)
   ┌────────────────────────────────────┐
   │ • 进程执行结束                     │
   │ • 等待父进程回收                   │
   │ • 保留退出状态                     │
   │ • PCB尚未释放                      │
   └────────────────────────────────────┘
```

### 2.2 七状态模型（包含挂起）

```
┌─────────────────────────────────────────────────────────────┐
│              包含挂起状态的进程模型                           │
└─────────────────────────────────────────────────────────────┘

                                内存中
    ┌─────────────────────────────────────────────────┐
    │                                                 │
    │   ┌──────────┐  调度   ┌──────────┐            │
    │   │  就绪    │────────▶│  运行    │            │
    │   │ (Ready)  │◀────────│(Running) │            │
    │   └─────┬────┘  超时   └────┬─────┘            │
    │         │                   │                  │
    │         │                   │ 等待事件          │
    │         │ 激活              │                  │
    │         │                   ▼                  │
    │         │            ┌──────────┐              │
    │         └────────────│  阻塞    │              │
    │                      │(Blocked) │              │
    │                      └─────┬────┘              │
    │                            │                   │
    └────────────────────────────┼───────────────────┘
                                 │
                    挂起(Suspend) │
                                 │
    ┌────────────────────────────┼───────────────────┐
    │                            │                   │
    │                      交换到磁盘                 │
    │                            │                   │
    │   ┌──────────┐             ▼                   │
    │   │ 就绪挂起  │      ┌──────────┐              │
    │   │(Ready    │◀─────│ 阻塞挂起  │              │
    │   │Suspend)  │ 事件 │(Blocked  │              │
    │   └─────┬────┘ 完成 │Suspend)  │              │
    │         │           └──────────┘              │
    │         │ 激活                                │
    │         │                                     │
    └─────────┼─────────────────────────────────────┘
              │
              ▼ 换入内存
           就绪状态

挂起的原因:
1. 系统负载过高，需要释放内存
2. 用户主动挂起（调试）
3. 父进程请求挂起子进程
4. 周期性执行的进程（cron）
5. 交换(Swapping)策略

Linux中的状态(ps命令):
┌──────┬─────────────────────────────┐
│ 状态 │ 含义                         │
├──────┼─────────────────────────────┤
│  R   │ Running/Runnable (运行/就绪) │
│  S   │ Sleeping (可中断睡眠)        │
│  D   │ Disk Sleep (不可中断睡眠)    │
│  T   │ Stopped (暂停)               │
│  t   │ Tracing Stop (跟踪暂停)      │
│  Z   │ Zombie (僵尸)                │
│  X   │ Dead (死亡)                  │
└──────┴─────────────────────────────┘
```

---

## 3. CPU调度算法

### 3.1 调度算法分类

```
┌─────────────────────────────────────────────────────────────┐
│              CPU调度算法分类                                  │
└─────────────────────────────────────────────────────────────┘

调度算法树:
CPU调度算法
├── 非抢占式(Non-preemptive)
│   ├── FCFS (First Come First Served)
│   ├── SJF (Shortest Job First)
│   └── 优先级调度
│
└── 抢占式(Preemptive)
    ├── SRTF (Shortest Remaining Time First)
    ├── RR (Round Robin)
    ├── 多级队列调度
    ├── 多级反馈队列调度
    └── CFS (Completely Fair Scheduler)

调度时机:
1. 进程从运行态切换到阻塞态（非抢占）
2. 进程从运行态切换到就绪态（抢占）
3. 进程从阻塞态切换到就绪态（抢占）
4. 进程终止（非抢占）

评价指标:
┌────────────────────────────────────────┐
│ 1. CPU利用率 (CPU Utilization)        │
│    尽可能让CPU忙碌                     │
│                                        │
│ 2. 吞吐量 (Throughput)                │
│    单位时间完成的进程数                │
│                                        │
│ 3. 周转时间 (Turnaround Time)        │
│    从提交到完成的总时间                │
│    = 完成时间 - 到达时间               │
│                                        │
│ 4. 等待时间 (Waiting Time)            │
│    在就绪队列中等待的总时间            │
│                                        │
│ 5. 响应时间 (Response Time)           │
│    从提交到首次响应的时间              │
│    对交互式系统重要                    │
└────────────────────────────────────────┘
```

### 3.2 经典调度算法详解

```
┌─────────────────────────────────────────────────────────────┐
│          1. FCFS (先来先服务 - First Come First Served)      │
└─────────────────────────────────────────────────────────────┘

原理: 按到达顺序依次执行

示例:
进程  到达时间  执行时间
P1      0        24
P2      1         3
P3      2         3

甘特图:
0        24   27   30
├────────┼────┼────┤
│   P1   │ P2 │ P3 │
└────────┴────┴────┘

平均等待时间 = (0 + 23 + 25) / 3 = 16ms

优点: 简单、公平
缺点: 护航效应(Convoy Effect)、平均等待时间长

┌─────────────────────────────────────────────────────────────┐
│          2. SJF (最短作业优先 - Shortest Job First)          │
└─────────────────────────────────────────────────────────────┘

原理: 选择执行时间最短的进程

示例(非抢占):
进程  到达时间  执行时间
P1      0        7
P2      2        4
P3      4        1
P4      5        4

甘特图:
0     7  8    12      16
├─────┼──┼─────┼───────┤
│ P1  │P3│ P2  │  P4   │
└─────┴──┴─────┴───────┘

平均等待时间 = (0 + 6 + 3 + 7) / 4 = 4ms

优点: 最优化平均等待时间
缺点: 无法准确预测执行时间、饥饿问题

┌─────────────────────────────────────────────────────────────┐
│          3. RR (时间片轮转 - Round Robin)                    │
└─────────────────────────────────────────────────────────────┘

原理: 每个进程分配固定时间片(如10ms)，超时则切换

示例(时间片=4):
进程  到达时间  执行时间
P1      0        24
P2      0         3
P3      0         3

甘特图:
0   4   7  10      14  18      22  26  30
├───┼───┼───┼───────┼───┼───────┼───┼───┤
│P1 │P2 │P3 │  P1   │P1 │  P1   │P1 │P1 │
└───┴───┴───┴───────┴───┴───────┴───┴───┘

平均等待时间 = (6 + 4 + 7) / 3 = 5.67ms

优点: 响应时间好、公平
缺点: 上下文切换开销、时间片选择困难

时间片选择:
- 太大: 退化为FCFS
- 太小: 上下文切换开销大
- 经验值: 10-100ms

┌─────────────────────────────────────────────────────────────┐
│          4. 多级反馈队列 (Multilevel Feedback Queue)         │
└─────────────────────────────────────────────────────────────┘

原理: 多个优先级队列，动态调整进程优先级

队列结构:
┌────────────────────────────────────┐
│ Q0: 优先级最高，时间片=8ms          │  新进程
│     [P1] [P2] [P3]                 │    ↓
├────────────────────────────────────┤  进入Q0
│ Q1: 优先级中等，时间片=16ms         │    ↓
│     [P4] [P5]                      │  超时降到Q1
├────────────────────────────────────┤    ↓
│ Q2: 优先级最低，FCFS                │  超时降到Q2
│     [P6] [P7] [P8]                 │
└────────────────────────────────────┘

调度规则:
1. 优先调度Q0
2. Q0为空则调度Q1
3. Q1为空则调度Q2
4. 进程在当前队列用完时间片→降到下一级队列
5. I/O完成→提升到上一级队列

优点:
- 响应时间快(新进程在高优先级)
- CPU密集型进程逐渐降级
- I/O密集型进程保持高优先级

┌─────────────────────────────────────────────────────────────┐
│          5. CFS (完全公平调度器 - Linux默认)                  │
└─────────────────────────────────────────────────────────────┘

原理: 基于虚拟运行时间(vruntime)，红黑树实现

虚拟运行时间计算:
vruntime += 实际运行时间 × (NICE_0_LOAD / 进程权重)

红黑树结构(按vruntime排序):
           ┌──────┐
           │ P3   │  vruntime=100
           │ (最小)│
           └──┬───┘
              │
       ┌──────┴──────┐
    ┌──▼───┐      ┌──▼───┐
    │ P1   │      │ P5   │
    │ 120  │      │ 150  │
    └──────┘      └──┬───┘
                     │
                  ┌──▼───┐
                  │ P7   │
                  │ 180  │
                  └──────┘

调度流程:
1. 选择vruntime最小的进程(最左节点) - O(1)
2. 运行一个周期(6ms)
3. 更新vruntime
4. 重新插入红黑树 - O(log n)

优点:
✓ 完全公平(按权重分配CPU)
✓ 高效(O(log n))
✓ 支持多核
✓ 实时性好

特性:
- sched_latency: 调度周期(默认6ms)
- min_granularity: 最小调度粒度(0.75ms)
- nice值: -20(最高优先级) 到 +19(最低)
```

### 3.3 实时调度

```
┌─────────────────────────────────────────────────────────────┐
│              实时调度算法                                     │
└─────────────────────────────────────────────────────────────┘

实时系统分类:
1. 硬实时(Hard Real-Time)
   - 必须在截止期前完成
   - 超时即失败
   - 例: 飞行控制、医疗设备

2. 软实时(Soft Real-Time)
   - 尽量在截止期前完成
   - 超时性能下降
   - 例: 视频播放、游戏

Linux实时调度策略:
┌────────────────────────────────────────┐
│ SCHED_FIFO  (先进先出)                 │
│ - 静态优先级(1-99)                     │
│ - 无时间片限制                         │
│ - 运行到完成或阻塞                     │
│                                        │
│ SCHED_RR    (轮转)                     │
│ - 静态优先级(1-99)                     │
│ - 有时间片(默认100ms)                  │
│ - 同优先级轮转                         │
│                                        │
│ SCHED_DEADLINE (截止期调度)            │
│ - EDF算法(Earliest Deadline First)    │
│ - 保证截止期                           │
│ - Linux 3.14+                         │
└────────────────────────────────────────┘

优先级关系:
┌────────────────────────────────────┐
│ 优先级 1-99: 实时进程               │  高
│  ├─ SCHED_FIFO                     │  ↑
│  ├─ SCHED_RR                       │  │
│  └─ SCHED_DEADLINE                 │  │
├────────────────────────────────────┤  │
│ 优先级 100-139: 普通进程            │  │
│  └─ SCHED_NORMAL (CFS)             │  │
└────────────────────────────────────┘  低
```

---

## 4. 进程间通信(IPC)

### 4.1 IPC机制概览

```
┌─────────────────────────────────────────────────────────────┐
│              IPC机制分类                                      │
└─────────────────────────────────────────────────────────────┘

IPC机制
├── 管道(Pipe)
│   ├── 匿名管道(pipe)      - 父子进程
│   └── 命名管道(FIFO)      - 无关进程
│
├── 信号(Signal)             - 异步通知
│
├── 消息队列(Message Queue)  - 消息传递
│
├── 共享内存(Shared Memory)  - 最快，需同步
│
├── 信号量(Semaphore)        - 同步原语
│
├── Socket                   - 网络通信
│
└── 内存映射文件(mmap)       - 文件共享

性能对比:
┌──────────────┬──────────┬──────────┬─────────┐
│ 机制         │ 速度     │ 数据量   │ 用途    │
├──────────────┼──────────┼──────────┼─────────┤
│ 管道         │ 中       │ 流数据   │ 简单通信│
│ 消息队列     │ 中       │ 结构化   │ 解耦    │
│ 共享内存     │ 最快     │ 大数据   │ 高性能  │
│ 信号         │ 最快     │ 无数据   │ 通知    │
│ Socket       │ 慢       │ 任意     │ 网络    │
└──────────────┴──────────┴──────────┴─────────┘
```

### 4.2 管道详解

```
┌─────────────────────────────────────────────────────────────┐
│              管道(Pipe)                                       │
└─────────────────────────────────────────────────────────────┘

匿名管道原理:
┌─────────────┐              ┌─────────────┐
│  进程A      │              │  进程B      │
│             │              │             │
│  write()────┼──┐      ┌────┼──read()     │
│    fd[1]    │  │      │    │    fd[0]    │
└─────────────┘  │      │    └─────────────┘
                 │      │
                 ▼      ▼
          ┌────────────────────┐
          │   内核缓冲区(4KB)   │
          │  ┌──┬──┬──┬──┬──┐  │
          │  │D1│D2│D3│D4│  │  │  FIFO队列
          │  └──┴──┴──┴──┴──┘  │
          └────────────────────┘

特点:
• 半双工(单向通信)
• 只能用于父子进程或兄弟进程
• 读取后数据从管道移除
• 写满阻塞，读空阻塞
• 进程结束自动关闭

命名管道(FIFO):
┌─────────────┐              ┌─────────────┐
│ 无关进程A   │              │ 无关进程B   │
│             │              │             │
│ open("/tmp/ │              │ open("/tmp/ │
│  myfifo")───┼──┐      ┌────┼─  myfifo") │
│ write()     │  │      │    │ read()      │
└─────────────┘  │      │    └─────────────┘
                 │      │
                 ▼      ▼
          ┌────────────────────┐
          │  文件系统中的特殊文件│
          │  /tmp/myfifo       │
          │  (FIFO类型)        │
          └────────────────────┘
```

### 4.3 共享内存详解

```
┌─────────────────────────────────────────────────────────────┐
│              共享内存(Shared Memory)                          │
└─────────────────────────────────────────────────────────────┘

进程A虚拟地址空间        进程B虚拟地址空间
┌─────────────────┐      ┌─────────────────┐
│  0xFFFFFFFF     │      │  0xFFFFFFFF     │
│     内核空间    │      │     内核空间    │
├─────────────────┤      ├─────────────────┤
│                 │      │                 │
│   栈            │      │   栈            │
│                 │      │                 │
│   ...           │      │   ...           │
│                 │      │                 │
│   共享内存段    │      │   共享内存段    │
│  0xA0000000────┐│      │  0xB0000000────┐│
│  [共享数据]     ││      │  [共享数据]     ││
│                ││      │                ││
└────────────────┘│      └────────────────┘│
                  │                        │
                  └────────┬───────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │  物理内存中的   │
                  │  同一块区域     │
                  │  [共享数据]     │
                  └────────────────┘

创建流程:
1. shmget() - 创建/获取共享内存标识符
2. shmat()  - 将共享内存映射到进程地址空间
3. 使用共享内存(直接读写)
4. shmdt()  - 分离共享内存
5. shmctl() - 控制/删除共享内存

优点:
✓ 最快的IPC(无需拷贝)
✓ 适合大量数据传输

缺点:
✗ 需要同步机制(信号量/互斥锁)
✗ 多进程访问需要协调
```

### 4.4 IPC实战代码

```c
/*
 * IPC综合示例
 * 编译: gcc -o ipc_demo ipc_demo.c -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/msg.h>
#include <signal.h>

// ============ 1. 管道示例 ============
void demo_pipe() {
    printf("\n=== 管道(Pipe)示例 ===\n");

    int pipefd[2];
    pid_t pid;
    char buf[100];

    // 创建管道
    if (pipe(pipefd) == -1) {
        perror("pipe");
        return;
    }

    pid = fork();

    if (pid == 0) {
        // 子进程：读取端
        close(pipefd[1]);  // 关闭写端

        ssize_t n = read(pipefd[0], buf, sizeof(buf));
        printf("子进程收到: %s", buf);

        close(pipefd[0]);
        exit(0);
    } else {
        // 父进程：写入端
        close(pipefd[0]);  // 关闭读端

        const char *msg = "Hello from parent!\n";
        write(pipefd[1], msg, strlen(msg));

        close(pipefd[1]);
        wait(NULL);
    }
}

// ============ 2. 信号示例 ============
volatile sig_atomic_t got_signal = 0;

void signal_handler(int signo) {
    printf("\n收到信号: %d\n", signo);
    got_signal = 1;
}

void demo_signal() {
    printf("\n=== 信号(Signal)示例 ===\n");

    // 设置信号处理函数
    signal(SIGUSR1, signal_handler);

    pid_t pid = fork();

    if (pid == 0) {
        // 子进程：等待信号
        printf("子进程等待信号...\n");
        while (!got_signal) {
            pause();  // 等待信号
        }
        printf("子进程收到信号，退出\n");
        exit(0);
    } else {
        // 父进程：发送信号
        sleep(1);
        printf("父进程发送SIGUSR1信号\n");
        kill(pid, SIGUSR1);
        wait(NULL);
    }
}

// ============ 3. 消息队列示例 ============
struct msg_buffer {
    long msg_type;
    char msg_text[100];
};

void demo_message_queue() {
    printf("\n=== 消息队列(Message Queue)示例 ===\n");

    key_t key = ftok(".", 'a');
    int msgid = msgget(key, 0666 | IPC_CREAT);

    if (msgid == -1) {
        perror("msgget");
        return;
    }

    pid_t pid = fork();

    if (pid == 0) {
        // 子进程：接收消息
        struct msg_buffer message;
        msgrcv(msgid, &message, sizeof(message.msg_text), 1, 0);
        printf("子进程收到消息: %s\n", message.msg_text);
        exit(0);
    } else {
        // 父进程：发送消息
        struct msg_buffer message;
        message.msg_type = 1;
        strcpy(message.msg_text, "Hello from message queue!");

        msgsnd(msgid, &message, sizeof(message.msg_text), 0);
        printf("父进程发送消息\n");

        wait(NULL);

        // 删除消息队列
        msgctl(msgid, IPC_RMID, NULL);
    }
}

// ============ 4. 共享内存示例 ============
void demo_shared_memory() {
    printf("\n=== 共享内存(Shared Memory)示例 ===\n");

    key_t key = ftok(".", 'b');
    int shmid = shmget(key, 1024, 0666 | IPC_CREAT);

    if (shmid == -1) {
        perror("shmget");
        return;
    }

    pid_t pid = fork();

    if (pid == 0) {
        // 子进程：读取共享内存
        sleep(1);  // 等待父进程写入
        char *str = (char *)shmat(shmid, NULL, 0);
        printf("子进程读取: %s\n", str);
        shmdt(str);
        exit(0);
    } else {
        // 父进程：写入共享内存
        char *str = (char *)shmat(shmid, NULL, 0);
        strcpy(str, "Hello from shared memory!");
        printf("父进程写入共享内存\n");

        wait(NULL);

        shmdt(str);
        shmctl(shmid, IPC_RMID, NULL);  // 删除共享内存
    }
}

// ============ 主函数 ============
int main() {
    printf("IPC机制综合示例\n");
    printf("==========================================\n");

    demo_pipe();
    demo_signal();
    demo_message_queue();
    demo_shared_memory();

    return 0;
}
```

---

## 5. 线程与并发

### 5.1 POSIX线程(pthread)

```c
/*
 * POSIX线程示例
 * 编译: gcc -o pthread_demo pthread_demo.c -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

// 全局变量(线程间共享)
int shared_counter = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// 线程函数
void *thread_function(void *arg) {
    int thread_id = *(int *)arg;

    printf("线程%d开始执行\n", thread_id);

    // 使用互斥锁保护共享变量
    for (int i = 0; i < 100000; i++) {
        pthread_mutex_lock(&mutex);
        shared_counter++;
        pthread_mutex_unlock(&mutex);
    }

    printf("线程%d完成\n", thread_id);
    return NULL;
}

// 生产者-消费者问题
#define BUFFER_SIZE 10

int buffer[BUFFER_SIZE];
int count = 0;  // 缓冲区中的项数

pthread_mutex_t buffer_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t not_full = PTHREAD_COND_INITIALIZER;
pthread_cond_t not_empty = PTHREAD_COND_INITIALIZER;

void *producer(void *arg) {
    for (int i = 0; i < 20; i++) {
        pthread_mutex_lock(&buffer_mutex);

        // 等待缓冲区不满
        while (count == BUFFER_SIZE) {
            pthread_cond_wait(&not_full, &buffer_mutex);
        }

        // 生产
        buffer[count++] = i;
        printf("生产: %d, 缓冲区数量: %d\n", i, count);

        // 通知消费者
        pthread_cond_signal(&not_empty);
        pthread_mutex_unlock(&buffer_mutex);

        usleep(100000);  // 模拟生产时间
    }
    return NULL;
}

void *consumer(void *arg) {
    for (int i = 0; i < 20; i++) {
        pthread_mutex_lock(&buffer_mutex);

        // 等待缓冲区不空
        while (count == 0) {
            pthread_cond_wait(&not_empty, &buffer_mutex);
        }

        // 消费
        int item = buffer[--count];
        printf("           消费: %d, 缓冲区数量: %d\n", item, count);

        // 通知生产者
        pthread_cond_signal(&not_full);
        pthread_mutex_unlock(&buffer_mutex);

        usleep(150000);  // 模拟消费时间
    }
    return NULL;
}

int main() {
    printf("=== 基本线程示例 ===\n");

    pthread_t threads[4];
    int thread_ids[4];

    // 创建4个线程
    for (int i = 0; i < 4; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, thread_function, &thread_ids[i]);
    }

    // 等待所有线程完成
    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("最终计数器值: %d (预期: 400000)\n", shared_counter);

    printf("\n=== 生产者-消费者示例 ===\n");

    pthread_t prod_thread, cons_thread;
    pthread_create(&prod_thread, NULL, producer, NULL);
    pthread_create(&cons_thread, NULL, consumer, NULL);

    pthread_join(prod_thread, NULL);
    pthread_join(cons_thread, NULL);

    pthread_mutex_destroy(&mutex);
    pthread_mutex_destroy(&buffer_mutex);
    pthread_cond_destroy(&not_full);
    pthread_cond_destroy(&not_empty);

    return 0;
}
```

### 5.2 协程(Coroutine)

```python
#!/usr/bin/env python3
"""
协程示例：用户态调度
Python asyncio实现
"""

import asyncio
import time

# 传统同步方式
def sync_download(url):
    print(f"开始下载: {url}")
    time.sleep(2)  # 模拟网络延迟
    print(f"完成下载: {url}")
    return f"数据来自 {url}"

# 协程异步方式
async def async_download(url):
    print(f"开始下载: {url}")
    await asyncio.sleep(2)  # 模拟网络延迟（不阻塞）
    print(f"完成下载: {url}")
    return f"数据来自 {url}"

async def main():
    print("=== 同步下载（串行）===")
    start = time.time()
    sync_download("http://site1.com")
    sync_download("http://site2.com")
    sync_download("http://site3.com")
    print(f"同步总耗时: {time.time() - start:.2f}秒\n")

    print("=== 异步下载（并发）===")
    start = time.time()
    # 并发执行3个协程
    results = await asyncio.gather(
        async_download("http://site1.com"),
        async_download("http://site2.com"),
        async_download("http://site3.com")
    )
    print(f"异步总耗时: {time.time() - start:.2f}秒\n")
    print(f"结果: {results}")

if __name__ == "__main__":
    asyncio.run(main())

"""
输出示例:
=== 同步下载（串行）===
开始下载: http://site1.com
完成下载: http://site1.com
开始下载: http://site2.com
完成下载: http://site2.com
开始下载: http://site3.com
完成下载: http://site3.com
同步总耗时: 6.01秒

=== 异步下载（并发）===
开始下载: http://site1.com
开始下载: http://site2.com
开始下载: http://site3.com
完成下载: http://site1.com
完成下载: http://site2.com
完成下载: http://site3.com
异步总耗时: 2.00秒
"""
```

---

## 6. 实战：进程监控与管理

```bash
#!/bin/bash
# 进程监控脚本

cat << 'EOF'
=== 进程信息查看 ===
EOF

# 1. 查看所有进程
echo -e "\n1. 查看所有进程"
ps aux | head -10

# 2. 查看进程树
echo -e "\n2. 进程树"
pstree -p $$ | head -20

# 3. 查看线程
echo -e "\n3. 查看线程"
ps -eLf | grep $$ | head -5

# 4. 实时进程监控
echo -e "\n4. 实时监控（top）"
cat << 'SCRIPT'
top -n 1 -b | head -20
SCRIPT

# 5. 查看进程详细信息
echo -e "\n5. 进程详细信息"
cat /proc/$$/status | head -20

# 6. CPU亲和性
echo -e "\n6. CPU亲和性"
taskset -cp $$

# 7. 进程优先级
echo -e "\n7. nice值"
ps -o pid,ni,comm -p $$

# 8. 进程打开的文件
echo -e "\n8. 打开的文件描述符"
ls -l /proc/$$/fd | head -10

# 9. 进程调度策略
echo -e "\n9. 调度策略"
chrt -p $$
```

```python
#!/usr/bin/env python3
"""
高级进程监控工具
"""

import psutil
import time
from datetime import datetime

def monitor_process(pid):
    """监控指定进程"""
    try:
        proc = psutil.Process(pid)

        print(f"=== 进程信息 (PID: {pid}) ===\n")

        # 基本信息
        print(f"进程名: {proc.name()}")
        print(f"状态: {proc.status()}")
        print(f"创建时间: {datetime.fromtimestamp(proc.create_time())}")
        print(f"父进程PID: {proc.ppid()}")
        print(f"用户: {proc.username()}")

        # CPU信息
        cpu_percent = proc.cpu_percent(interval=1)
        cpu_times = proc.cpu_times()
        print(f"\nCPU使用率: {cpu_percent}%")
        print(f"用户态时间: {cpu_times.user}s")
        print(f"内核态时间: {cpu_times.system}s")
        print(f"CPU亲和性: {proc.cpu_affinity()}")

        # 内存信息
        mem_info = proc.memory_info()
        print(f"\n内存RSS: {mem_info.rss / 1024 / 1024:.2f} MB")
        print(f"虚拟内存: {mem_info.vms / 1024 / 1024:.2f} MB")
        print(f"内存百分比: {proc.memory_percent():.2f}%")

        # 线程信息
        threads = proc.threads()
        print(f"\n线程数: {proc.num_threads()}")
        for thread in threads[:3]:  # 显示前3个线程
            print(f"  线程{thread.id}: 用户态={thread.user_time}s, "
                  f"内核态={thread.system_time}s")

        # 文件描述符
        open_files = proc.open_files()
        print(f"\n打开文件数: {len(open_files)}")
        for f in open_files[:5]:  # 显示前5个
            print(f"  {f.path}")

        # 连接信息
        connections = proc.connections()
        print(f"\n网络连接数: {len(connections)}")
        for conn in connections[:3]:  # 显示前3个
            print(f"  {conn.laddr} -> {conn.raddr} ({conn.status})")

    except psutil.NoSuchProcess:
        print(f"进程 {pid} 不存在")
    except psutil.AccessDenied:
        print(f"无权访问进程 {pid}")

def system_overview():
    """系统整体概况"""
    print("=== 系统概况 ===\n")

    # CPU
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    print(f"CPU核心数: {psutil.cpu_count(logical=False)} 物理, "
          f"{psutil.cpu_count()} 逻辑")
    for i, percent in enumerate(cpu_percent):
        print(f"  CPU{i}: {percent}%")

    # 内存
    mem = psutil.virtual_memory()
    print(f"\n内存总量: {mem.total / 1024 / 1024 / 1024:.2f} GB")
    print(f"已用: {mem.used / 1024 / 1024 / 1024:.2f} GB ({mem.percent}%)")
    print(f"可用: {mem.available / 1024 / 1024 / 1024:.2f} GB")

    # 进程统计
    print(f"\n总进程数: {len(psutil.pids())}")
    print(f"运行中: {len([p for p in psutil.process_iter(['status']) "
          f"if p.info['status'] == psutil.STATUS_RUNNING])}")

    # CPU密集型进程
    print("\nTop 5 CPU密集型进程:")
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
        processes.append(proc.info)

    processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
    for proc in processes[:5]:
        print(f"  PID {proc['pid']:6} {proc['name']:20} "
              f"{proc['cpu_percent']:5.1f}%")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pid = int(sys.argv[1])
        monitor_process(pid)
    else:
        system_overview()
```

---

## 7. 延伸阅读

### 7.1 关键概念总结

```
┌─────────────────────────────────────────────────────────────┐
│              进程管理核心概念                                 │
└─────────────────────────────────────────────────────────────┘

进程管理
├── 进程概念
│   ├── PCB（进程控制块）
│   ├── 进程状态（新建/就绪/运行/阻塞/终止）
│   ├── 进程vs线程（资源 vs 调度）
│   └── 上下文切换（~1500周期）
│
├── CPU调度
│   ├── FCFS - 简单但护航效应
│   ├── SJF - 最优平均等待时间
│   ├── RR - 时间片轮转，响应时间好
│   ├── 多级反馈队列 - 动态优先级
│   └── CFS - Linux默认，红黑树
│
├── 进程间通信
│   ├── 管道 - 单向，父子进程
│   ├── 消息队列 - 结构化消息
│   ├── 共享内存 - 最快，需同步
│   ├── 信号 - 异步通知
│   └── Socket - 网络通信
│
└── 并发控制
    ├── 互斥锁 - 临界区保护
    ├── 信号量 - 资源计数
    ├── 条件变量 - 等待/通知
    └── 读写锁 - 读共享写独占
```

---

**下一步**: 学习[内存管理](03_memory_management.md),深入理解虚拟内存和页面置换算法。

**相关模块**:
- [OS概述](01_os_overview.md) - 操作系统基础
- [并发控制](06_concurrency.md) - 锁与死锁
- [Linux](../Linux/) - Linux进程管理实战

**文件大小**: 约35KB
**最后更新**: 2024年
