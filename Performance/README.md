# 性能测试与调优

性能优化是架构师的核心能力之一,需要系统性的方法论和工具链。

## 目录

1. [负载测试](01_load_testing.md) - JMeter、Gatling、K6
2. [性能分析](02_profiling.md) - CPU/内存/火焰图
3. [瓶颈定位](03_bottleneck_analysis.md) - 系统诊断方法
4. [优化案例](04_optimization_cases.md) - 真实案例复盘

## 性能指标

```
┌────────────────────────────────────────────────────┐
│              关键性能指标                          │
├────────────────────────────────────────────────────┤
│  响应时间  RT (Response Time)                     │
│  吞吐量    TPS/QPS (Transactions/Queries Per Sec) │
│  并发数    Concurrency                             │
│  错误率    Error Rate                              │
│  资源利用率 CPU/Memory/Disk/Network               │
└────────────────────────────────────────────────────┘
```

## 相关模块

性能调优需要理解底层原理:

```
┌─────────────────────────────────────────────────┐
│          性能优化的基础知识                      │
├─────────────────────────────────────────────────┤
│                                                 │
│  [Computer_Hardware](../Computer_Hardware/)    │
│  └─▶ CPU架构、内存系统、存储设备               │
│      理解硬件瓶颈与性能特性                     │
│                                                 │
│  [Operating_Systems](../Operating_Systems/)    │
│  └─▶ 进程调度、内存管理、I/O管理               │
│      理解OS层面的性能影响                       │
│                                                 │
│  Performance (本模块)                           │
│  └─▶ 性能测试、分析、调优实战                   │
│      应用层性能优化方法                         │
└─────────────────────────────────────────────────┘
```

**推荐学习路径**:
1. [Computer_Hardware](../Computer_Hardware/) - 了解硬件性能特性
2. [Operating_Systems](../Operating_Systems/09_performance_tuning.md) - 学习OS性能调优
3. **Performance (本模块)** - 应用层性能优化实战

开始学习 → [01_load_testing.md](01_load_testing.md)
