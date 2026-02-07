# Kafka 基础教程

## 一、概述

Apache Kafka 是一个分布式流处理平台，具有以下核心能力：

- **消息队列**：高吞吐量的发布/订阅消息系统
- **存储系统**：持久化消息，支持数据回溯
- **流处理**：实时处理数据流

### 核心特点

| 特点 | 说明 |
|------|------|
| 高吞吐量 | 单机可达百万级 TPS |
| 持久化 | 消息持久化到磁盘，支持数据回溯 |
| 分布式 | 支持水平扩展，天然分布式 |
| 高可用 | 副本机制保证数据不丢失 |
| 顺序性 | 分区内消息严格有序 |

### 使用场景

1. **消息队列**：系统解耦、异步处理、削峰填谷
2. **日志收集**：统一收集各服务日志
3. **流处理**：实时数据分析、ETL
4. **事件溯源**：记录状态变更历史
5. **指标收集**：运维监控数据聚合

## 二、核心概念

### 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Kafka Cluster                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Broker 0  │  │   Broker 1  │  │   Broker 2  │             │
│  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  │             │
│  │  │Topic A│  │  │  │Topic A│  │  │  │Topic A│  │             │
│  │  │ P0(L) │  │  │  │ P1(L) │  │  │  │ P2(L) │  │             │
│  │  │ P1(F) │  │  │  │ P2(F) │  │  │  │ P0(F) │  │             │
│  │  └───────┘  │  │  └───────┘  │  │  └───────┘  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                  │
│  ┌──────────────────────────────────────────────────┐           │
│  │                   ZooKeeper                       │           │
│  │    (元数据管理、Controller 选举、配置管理)         │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
        ▲                                        │
        │                                        ▼
┌───────────────┐                      ┌───────────────┐
│   Producer    │                      │   Consumer    │
│  (生产者)      │                      │  (消费者)      │
└───────────────┘                      └───────────────┘

P0(L) = Partition 0 Leader    P0(F) = Partition 0 Follower
```

### 核心组件

#### 1. Broker（消息代理）

Kafka 集群中的一台服务器就是一个 Broker。

```
Broker 职责：
├── 接收生产者消息
├── 存储消息到磁盘
├── 响应消费者拉取请求
├── 副本同步
└── 分区管理
```

#### 2. Topic（主题）

消息的逻辑分类，类似于数据库中的表。

```
Topic 特点：
├── 一个 Topic 可以有多个分区
├── 一个 Topic 可以有多个订阅者
├── 消息发送时必须指定 Topic
└── 不同 Topic 的消息互相隔离
```

#### 3. Partition（分区）

Topic 的物理分片，是 Kafka 并行处理的基本单位。

```
┌─────────────────────────────────────────────────────┐
│                    Topic: orders                     │
├─────────────────────────────────────────────────────┤
│  Partition 0: [msg0] [msg3] [msg6] [msg9]  ...      │
│  Partition 1: [msg1] [msg4] [msg7] [msg10] ...      │
│  Partition 2: [msg2] [msg5] [msg8] [msg11] ...      │
└─────────────────────────────────────────────────────┘

分区特点：
├── 分区内消息有序（FIFO）
├── 分区间消息无序
├── 每个分区是一个有序的、不可变的消息序列
├── 消息通过 offset（偏移量）唯一标识
└── 分区数决定了消费者的最大并行度
```

#### 4. Offset（偏移量）

分区中每条消息的唯一标识，从 0 开始递增。

```
Partition 0:
┌─────┬─────┬─────┬─────┬─────┬─────┐
│  0  │  1  │  2  │  3  │  4  │  5  │  ← Offset
├─────┼─────┼─────┼─────┼─────┼─────┤
│ msgA│ msgB│ msgC│ msgD│ msgE│ msgF│  ← Message
└─────┴─────┴─────┴─────┴─────┴─────┘
                          ▲
                          │
                    Current Offset (消费者当前位置)
```

#### 5. Replica（副本）

分区的备份，用于高可用。

```
副本类型：
├── Leader Replica（领导者副本）
│   ├── 负责处理所有读写请求
│   └── 每个分区有且仅有一个 Leader
│
└── Follower Replica（跟随者副本）
    ├── 从 Leader 同步数据
    ├── 不处理客户端请求
    └── Leader 故障时可被选举为新 Leader

ISR (In-Sync Replicas)：
├── 与 Leader 保持同步的副本集合
├── 只有 ISR 中的副本才有资格被选为 Leader
└── 副本落后太多会被踢出 ISR
```

#### 6. Producer（生产者）

向 Topic 发送消息的客户端。

```
发送流程：
1. 序列化消息
2. 根据分区策略选择目标分区
3. 将消息添加到批次（Batch）
4. 发送批次到对应的 Broker
5. 等待确认（根据 acks 配置）

分区策略：
├── 指定分区：直接发送到指定分区
├── 按 Key 哈希：相同 Key 的消息发送到同一分区
└── 轮询（Round Robin）：没有 Key 时轮询分配
```

#### 7. Consumer（消费者）

从 Topic 拉取消息的客户端。

```
消费模式：
├── 订阅模式（Subscribe）：自动分配分区
└── 分配模式（Assign）：手动指定分区

拉取流程：
1. 向 Coordinator 发送加入组请求
2. 等待分区分配
3. 从 Leader 拉取消息
4. 处理消息
5. 提交 Offset
```

#### 8. Consumer Group（消费者组）

多个消费者组成的逻辑分组。

```
┌─────────────────────────────────────────────────────────────┐
│                    Topic: orders (3 partitions)             │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│    │ P0       │    │ P1       │    │ P2       │            │
│    └────┬─────┘    └────┬─────┘    └────┬─────┘            │
└─────────┼───────────────┼───────────────┼──────────────────┘
          │               │               │
          ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│               Consumer Group A (group.id=A)                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │Consumer 1│    │Consumer 2│    │Consumer 3│              │
│  │  (P0)    │    │  (P1)    │    │  (P2)    │              │
│  └──────────┘    └──────────┘    └──────────┘              │
└─────────────────────────────────────────────────────────────┘

消费者组特点：
├── 同一组内，一个分区只能被一个消费者消费
├── 不同组之间，消费互不影响（广播效果）
├── 消费者数量 > 分区数时，多余消费者空闲
└── 消费者数量 < 分区数时，一个消费者消费多个分区
```

## 三、安装与配置

### Docker 快速启动

```bash
# 创建 docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
EOF

# 启动服务
docker-compose up -d

# 验证
docker exec -it kafka kafka-topics --bootstrap-server localhost:9092 --list
```

### KRaft 模式（无 ZooKeeper）

Kafka 3.0+ 支持 KRaft 模式，移除了对 ZooKeeper 的依赖。

```bash
# KRaft 单节点配置
cat > docker-compose-kraft.yml << 'EOF'
version: '3'
services:
  kafka:
    image: confluentinc/cp-kafka:7.5.0
    ports:
      - "9092:9092"
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@localhost:9093
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      CLUSTER_ID: 'MkU3OEVBNTcwNTJENDM2Qk'
EOF
```

### 主要配置（server.properties）

```properties
# ============================================================
#                    Broker 基础配置
# ============================================================

# Broker 唯一标识
broker.id=0

# 监听地址
listeners=PLAINTEXT://0.0.0.0:9092

# 对外暴露的地址（客户端连接使用）
advertised.listeners=PLAINTEXT://192.168.1.100:9092

# 数据存储目录（可配置多个，逗号分隔）
log.dirs=/var/kafka-logs

# ZooKeeper 连接地址
zookeeper.connect=localhost:2181

# ============================================================
#                    Topic 默认配置
# ============================================================

# 默认分区数
num.partitions=3

# 默认副本因子
default.replication.factor=1

# 是否自动创建 Topic
auto.create.topics.enable=true

# ============================================================
#                    日志配置
# ============================================================

# 单个日志段文件大小（1GB）
log.segment.bytes=1073741824

# 日志保留时间（7天）
log.retention.hours=168

# 日志保留大小（-1 表示无限制）
log.retention.bytes=-1

# 日志清理策略：delete（删除）或 compact（压缩）
log.cleanup.policy=delete

# ============================================================
#                    副本配置
# ============================================================

# ISR 最小副本数（低于此值将拒绝写入）
min.insync.replicas=1

# 副本拉取最大字节数
replica.fetch.max.bytes=1048576

# 副本落后多少条消息会被踢出 ISR
replica.lag.max.messages=4000

# ============================================================
#                    网络配置
# ============================================================

# 处理网络请求的线程数
num.network.threads=3

# 处理磁盘 IO 的线程数
num.io.threads=8

# 请求队列大小
queued.max.requests=500

# Socket 缓冲区大小
socket.send.buffer.bytes=102400
socket.receive.buffer.bytes=102400
socket.request.max.bytes=104857600
```

## 四、消息模型

### 消息结构

```
┌─────────────────────────────────────────────────────────────┐
│                      Kafka Record                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┬────────────────────────────────────────┐   │
│  │  Headers    │  [key1:value1, key2:value2, ...]      │   │
│  ├─────────────┼────────────────────────────────────────┤   │
│  │  Key        │  (可选) 用于分区路由和日志压缩         │   │
│  ├─────────────┼────────────────────────────────────────┤   │
│  │  Value      │  消息体（实际业务数据）                 │   │
│  ├─────────────┼────────────────────────────────────────┤   │
│  │  Timestamp  │  消息时间戳                             │   │
│  ├─────────────┼────────────────────────────────────────┤   │
│  │  Offset     │  分区内唯一偏移量（由 Broker 分配）    │   │
│  └─────────────┴────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 消息保证

#### 1. 生产者可靠性（acks 配置）

```
acks=0：不等待确认
├── 最高吞吐量
├── 可能丢失消息
└── 适用于日志等允许丢失的场景

acks=1：Leader 确认
├── Leader 写入成功即返回
├── Leader 故障可能丢失
└── 平衡了吞吐量和可靠性

acks=all/-1：所有 ISR 确认
├── 所有 ISR 副本都确认才返回
├── 最高可靠性
└── 配合 min.insync.replicas 使用
```

#### 2. 消费者可靠性（Offset 提交）

```
自动提交（enable.auto.commit=true）：
├── 定期自动提交已拉取的 Offset
├── 可能重复消费或丢失消息
└── 适用于允许少量重复/丢失的场景

手动提交（enable.auto.commit=false）：
├── 同步提交：commitSync()，阻塞等待
├── 异步提交：commitAsync()，不阻塞
└── 精确控制提交时机，最高可靠性
```

#### 3. 消息语义

```
At Most Once（最多一次）：
├── 先提交 Offset，再处理消息
├── 消息可能丢失，不会重复
└── 生产者 acks=0

At Least Once（至少一次）：
├── 先处理消息，再提交 Offset
├── 消息可能重复，不会丢失
└── 生产者 acks=all + 消费者手动提交

Exactly Once（精确一次）：
├── 幂等生产者 + 事务
├── enable.idempotence=true
└── 最严格的语义保证
```

## 五、数据存储

### 日志结构

```
/var/kafka-logs/
├── topic-orders-0/                    # Topic: orders, Partition: 0
│   ├── 00000000000000000000.log       # 日志段文件
│   ├── 00000000000000000000.index     # 偏移量索引
│   ├── 00000000000000000000.timeindex # 时间戳索引
│   ├── 00000000000000123456.log       # 新的日志段
│   ├── 00000000000000123456.index
│   ├── 00000000000000123456.timeindex
│   └── leader-epoch-checkpoint        # Leader 纪元检查点
│
├── topic-orders-1/
│   └── ...
│
└── __consumer_offsets-0/              # 消费者 Offset 存储（内部 Topic）
    └── ...
```

### 日志段（Log Segment）

```
日志段文件（.log）：
├── 存储实际消息数据
├── 只追加写入（Append Only）
├── 达到阈值（大小/时间）后创建新段
└── 旧段可被清理或压缩

偏移量索引（.index）：
├── 稀疏索引（每隔一定字节建一个索引）
├── 格式：[offset -> file position]
└── 支持快速定位消息

时间戳索引（.timeindex）：
├── 格式：[timestamp -> offset]
└── 支持按时间查找消息
```

### 日志清理策略

```
Delete（删除）：
├── 直接删除过期的日志段
├── 基于时间：log.retention.hours
└── 基于大小：log.retention.bytes

Compact（压缩）：
├── 保留每个 Key 的最新 Value
├── 适用于 Key-Value 场景（如用户状态）
├── 后台线程异步执行
└── 设置：log.cleanup.policy=compact

Delete + Compact：
├── 同时启用两种策略
└── log.cleanup.policy=delete,compact
```

## 六、集群管理

### Controller 选举

```
Controller 职责：
├── 管理分区状态机
├── 管理副本状态机
├── 监听 Broker 上下线
├── 处理分区 Leader 选举
└── 同步元数据到其他 Broker

选举过程：
1. Broker 启动时在 ZK 创建 /controller 临时节点
2. 第一个创建成功的 Broker 成为 Controller
3. 其他 Broker 监听该节点变化
4. Controller 下线后，节点删除，触发重新选举
```

### 分区 Leader 选举

```
触发时机：
├── Broker 上下线
├── 手动触发 Preferred Leader 选举
└── ISR 变化

选举策略：
├── 从 ISR 中选择第一个副本作为 Leader
├── 如果 ISR 为空，根据配置决定：
│   ├── unclean.leader.election.enable=false：等待 ISR 恢复
│   └── unclean.leader.election.enable=true：从 OSR 选举（可能丢数据）
```

### Rebalance（重平衡）

消费者组内的分区重新分配。

```
触发时机：
├── 消费者加入或离开组
├── 消费者崩溃（心跳超时）
├── 订阅的 Topic 分区数变化
└── 手动触发

分配策略：
├── Range：按 Topic 分区范围分配
├── RoundRobin：轮询分配所有分区
├── Sticky：尽量保持原有分配
└── CooperativeSticky：增量式重平衡（减少 Stop-The-World）

避免频繁 Rebalance：
├── 增加 session.timeout.ms
├── 增加 heartbeat.interval.ms
├── 增加 max.poll.interval.ms
└── 控制单次 poll 返回的消息数量
```

## 七、性能优化

### 生产者优化

```properties
# 批量发送（减少网络请求）
batch.size=16384                  # 批次大小（字节）
linger.ms=5                       # 等待时间（毫秒）

# 压缩
compression.type=lz4              # none/gzip/snappy/lz4/zstd

# 缓冲区
buffer.memory=33554432            # 发送缓冲区大小

# 重试
retries=3                         # 重试次数
retry.backoff.ms=100              # 重试间隔
```

### 消费者优化

```properties
# 拉取配置
fetch.min.bytes=1                 # 最小拉取字节数
fetch.max.wait.ms=500             # 最大等待时间
max.poll.records=500              # 单次 poll 最大消息数
max.partition.fetch.bytes=1048576 # 每个分区最大拉取字节数

# 心跳与超时
session.timeout.ms=10000          # 会话超时
heartbeat.interval.ms=3000        # 心跳间隔
max.poll.interval.ms=300000       # poll 间隔超时
```

### Broker 优化

```properties
# 磁盘 IO
num.io.threads=16                 # IO 线程数
log.flush.interval.messages=10000 # 刷盘消息数阈值
log.flush.interval.ms=1000        # 刷盘时间阈值

# 网络
num.network.threads=8             # 网络线程数
socket.send.buffer.bytes=102400   # 发送缓冲区
socket.receive.buffer.bytes=102400 # 接收缓冲区

# 副本
num.replica.fetchers=4            # 副本拉取线程数
```

## 八、监控指标

### 关键指标

```
Broker 指标：
├── kafka.server:type=BrokerTopicMetrics,name=MessagesInPerSec
├── kafka.server:type=BrokerTopicMetrics,name=BytesInPerSec
├── kafka.server:type=BrokerTopicMetrics,name=BytesOutPerSec
├── kafka.server:type=ReplicaManager,name=UnderReplicatedPartitions
└── kafka.server:type=ReplicaManager,name=IsrShrinksPerSec

生产者指标：
├── record-send-rate：发送速率
├── record-error-rate：错误率
├── request-latency-avg：平均延迟
└── batch-size-avg：平均批次大小

消费者指标：
├── records-consumed-rate：消费速率
├── records-lag-max：最大消费延迟
├── fetch-latency-avg：拉取延迟
└── commit-latency-avg：提交延迟
```

### 常用监控命令

```bash
# 查看消费者组消费进度
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --describe --group my-group

# 输出示例：
# GROUP    TOPIC     PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
# my-group orders    0          1000            1050            50
# my-group orders    1          2000            2010            10
# my-group orders    2          1500            1500            0
```
