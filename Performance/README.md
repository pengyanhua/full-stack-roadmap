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

开始学习 → [01_load_testing.md](01_load_testing.md)
