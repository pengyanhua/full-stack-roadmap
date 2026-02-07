# 架构师知识体系

## 概述

本知识体系从架构师视角出发，系统性地整理了必须掌握的核心技术点。

**设计原则**：
- 融会贯通：将语言、数据库、中间件等知识串联
- 实战导向：每个知识点都有实战案例
- 避坑指南：详解常见问题和解决方案
- 架构思维：培养系统性思考能力

## 目录结构

```
Architecture/
│
├── 01_system_design/              # 系统设计基础
│   ├── 01_design_principles.md        # 设计原则 (SOLID, DRY, KISS, YAGNI)
│   ├── 02_architecture_patterns.md    # 架构模式 (分层、六边形、CQRS、事件驱动)
│   └── 03_capacity_planning.md        # 容量规划与估算
│
├── 02_distributed/                # 分布式系统
│   ├── 01_cap_base.md                 # CAP/BASE 理论与一致性模型
│   ├── 02_distributed_lock.md         # 分布式锁 (Redis/ZK/MySQL实现)
│   └── 03_distributed_transaction.md  # 分布式事务 (2PC/TCC/Saga)
│
├── 03_high_availability/          # 高可用架构
│   ├── 01_ha_principles.md            # 高可用原则与指标 (SLA/RTO/RPO)
│   ├── 02_rate_limiting.md            # 限流熔断降级
│   ├── 03_failover.md                 # 故障转移
│   └── 04_disaster_recovery.md        # 容灾与多活架构
│
├── 04_high_performance/           # 高性能优化
│   ├── 01_performance_metrics.md      # 性能指标与分析
│   ├── 02_concurrency.md              # 并发编程模型 (CSP/锁/无锁)
│   ├── 03_io_optimization.md          # I/O 与网络优化
│   └── 04_pool_pattern.md             # 池化技术 (对象池/连接池/协程池)
│
├── 05_microservices/              # 微服务架构
│   ├── 01_service_splitting.md        # 服务拆分原则 (DDD/绞杀者模式)
│   ├── 02_api_design.md               # API 设计规范 (REST/gRPC)
│   ├── 03_service_governance.md       # 服务治理 (注册发现/负载均衡/配置中心)
│   └── 04_service_mesh.md             # Service Mesh (Istio/Envoy)
│
├── 06_database_architecture/      # 数据库架构
│   ├── 01_mysql_optimization.md       # MySQL 优化 (索引/SQL/锁)
│   ├── 02_sharding.md                 # 分库分表 (策略/实现/问题)
│   └── 03_read_write_splitting.md     # 读写分离 (主从复制/延迟处理)
│
├── 07_cache_architecture/         # 缓存架构
│   └── 01_cache_patterns.md           # 缓存模式 (Cache Aside/穿透/击穿/雪崩)
│
├── 08_message_queue/              # 消息队列
│   └── 01_mq_patterns.md              # 消息模式 (可靠性/顺序/延迟/事务消息)
│
├── 09_security/                   # 安全架构
│   └── 01_security_fundamentals.md    # 安全基础 (认证授权/加密/OWASP)
│
└── 10_observability/              # 可观测性
    └── 01_observability.md            # 可观测性三支柱 (Metrics/Logs/Traces)
```

## 架构师能力模型

```
                    ┌─────────────────────┐
                    │     业务理解        │
                    │  Business Acumen    │
                    └─────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌───────────────┐
│   技术深度    │   │    架构能力      │   │   软技能      │
│ Tech Depth   │   │  Architecture   │   │ Soft Skills  │
├───────────────┤   ├─────────────────┤   ├───────────────┤
│ • 编程语言    │   │ • 系统设计      │   │ • 沟通协调    │
│ • 数据库      │   │ • 性能优化      │   │ • 技术领导力  │
│ • 中间件      │   │ • 高可用设计    │   │ • 项目管理    │
│ • 云原生      │   │ • 安全设计      │   │ • 技术决策    │
└───────────────┘   └─────────────────┘   └───────────────┘
```

## 学习路径

### Level 1: 基础夯实
1. 系统设计原则与模式
2. 单体应用优化
3. 数据库深入理解

### Level 2: 分布式入门
1. CAP 理论与实践
2. 缓存架构设计
3. 消息队列应用

### Level 3: 架构进阶
1. 微服务架构设计
2. 高可用架构
3. 性能优化

### Level 4: 架构大师
1. 多活架构
2. 云原生架构
3. 技术决策与权衡

## 核心原则

### 1. 简单优先 (KISS)
> 能用简单方案解决的问题，不要过度设计

### 2. 演进式架构
> 架构应该随着业务增长而演进，而非一步到位

### 3. 适度冗余
> 适当的冗余是高可用的基础，但过度冗余增加复杂性

### 4. 故障设计
> 假设一切都会失败，设计系统来处理失败

### 5. 数据驱动
> 基于数据和指标做决策，而非直觉
