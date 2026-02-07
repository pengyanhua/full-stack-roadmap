# Kafka 高级特性

## 一、幂等生产者

### 概念

幂等性（Idempotence）：多次执行相同操作，结果与执行一次相同。

```
问题场景：
生产者发送消息 → Broker 写入成功 → ACK 丢失 → 生产者重试 → 消息重复

幂等生产者解决方案：
├── Producer ID (PID)：生产者唯一标识
├── Sequence Number：消息序列号
└── Broker 去重：相同 PID + Sequence 的消息只保留一份
```

### 配置

```properties
# 开启幂等性
enable.idempotence=true

# 幂等性要求以下配置：
acks=all                    # 必须
retries > 0                 # 必须（默认 Integer.MAX_VALUE）
max.in.flight.requests.per.connection <= 5  # 必须
```

### 限制

```
幂等性范围：
├── 单分区幂等：只能保证单个分区内的幂等
├── 单会话幂等：Producer 重启后 PID 改变，无法去重
└── 不能跨分区：不同分区的消息无法去重

跨分区 + 跨会话幂等需要使用事务
```

## 二、事务

### 概念

Kafka 事务提供跨分区的原子性写入。

```
事务语义：
├── 原子性：事务中的所有消息要么全部成功，要么全部失败
├── 隔离性：消费者可以选择只读取已提交的消息
└── 精确一次：配合幂等性，实现 Exactly Once

使用场景：
├── 跨 Topic/分区的原子写入
├── Consume-Transform-Produce 模式
└── 流处理中的精确一次语义
```

### 事务流程

```
┌──────────────────────────────────────────────────────────────────┐
│                        事务执行流程                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   Producer                   Broker                               │
│      │                          │                                 │
│      │  1. initTransactions()   │                                 │
│      │ -----------------------> │  分配 PID，初始化事务状态        │
│      │                          │                                 │
│      │  2. beginTransaction()   │                                 │
│      │ -----------------------> │  标记事务开始                    │
│      │                          │                                 │
│      │  3. send(record)         │                                 │
│      │ -----------------------> │  写入消息（标记为未提交）         │
│      │                          │                                 │
│      │  4. sendOffsetsToTxn()   │                                 │
│      │ -----------------------> │  提交消费位点到事务               │
│      │                          │                                 │
│      │  5. commitTransaction()  │                                 │
│      │ -----------------------> │  提交事务，消息对消费者可见       │
│      │                          │                                 │
│      │  (或 abortTransaction()) │                                 │
│      │ -----------------------> │  回滚事务，消息被丢弃            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 配置

```properties
# 生产者配置
transactional.id=my-transactional-id   # 事务 ID（必须唯一且固定）
enable.idempotence=true                 # 自动开启

# 消费者配置
isolation.level=read_committed          # 只读取已提交的消息
# 可选值：
# read_uncommitted（默认）：读取所有消息
# read_committed：只读取已提交的事务消息
```

### 代码示例（Java）

```java
// 生产者事务示例
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("transactional.id", "my-transactional-id");
props.put("enable.idempotence", "true");

Producer<String, String> producer = new KafkaProducer<>(props);

// 初始化事务
producer.initTransactions();

try {
    // 开始事务
    producer.beginTransaction();

    // 发送多条消息（跨 Topic/分区）
    producer.send(new ProducerRecord<>("topic1", "key1", "value1"));
    producer.send(new ProducerRecord<>("topic2", "key2", "value2"));

    // 提交事务
    producer.commitTransaction();
} catch (Exception e) {
    // 回滚事务
    producer.abortTransaction();
}
```

## 三、消费者组协调

### Coordinator

```
Group Coordinator 职责：
├── 管理消费者组成员
├── 处理加入/离开请求
├── 分配分区
├── 管理消费位点
└── 处理心跳

Coordinator 选择：
├── 通过 __consumer_offsets 分区确定
├── 公式：abs(group.id.hashCode()) % numPartitions
└── 该分区的 Leader 所在 Broker 即为 Coordinator
```

### Rebalance 协议

```
Rebalance 流程（Eager 协议）：
1. 触发 Rebalance（成员变化/分区变化）
2. 所有消费者停止消费，放弃分区
3. 发送 JoinGroup 请求
4. Coordinator 选择 Leader Consumer
5. Leader 执行分区分配
6. 所有消费者发送 SyncGroup 请求获取分配结果
7. 恢复消费

问题：Stop-The-World，所有消费者暂停

增量协作式 Rebalance（Cooperative）：
1. 只撤销需要移动的分区
2. 其他分区继续消费
3. 减少 Rebalance 影响范围
```

### 分区分配策略

```
Range（默认）：
├── 按 Topic 维度分配
├── 每个 Topic 的分区连续分配给消费者
└── 可能导致不均衡

示例：3 个分区，2 个消费者
Topic A: P0, P1, P2
Consumer 1: P0, P1
Consumer 2: P2

RoundRobin：
├── 所有 Topic 的分区混在一起
├── 轮询分配
└── 更均衡，但可能打散同一 Topic 的分区

Sticky：
├── 尽量保持原有分配
├── 减少 Rebalance 时的分区移动
└── 在均衡性和稳定性间平衡

CooperativeSticky：
├── Sticky 的增量版本
├── 配合 Cooperative Rebalance 使用
└── Kafka 2.4+ 推荐
```

### 配置

```properties
# 分区分配策略
partition.assignment.strategy=org.apache.kafka.clients.consumer.CooperativeStickyAssignor

# 会话超时（检测消费者失败）
session.timeout.ms=45000

# 心跳间隔
heartbeat.interval.ms=3000

# poll 间隔超时（处理时间过长会被踢出）
max.poll.interval.ms=300000

# 单次 poll 最大消息数
max.poll.records=500
```

## 四、日志压缩（Log Compaction）

### 概念

日志压缩保留每个 Key 的最新 Value，删除旧值。

```
原始日志：
Offset  Key     Value
0       K1      V1
1       K2      V2
2       K1      V3      ← K1 的新值
3       K3      V4
4       K2      null    ← K2 的墓碑消息
5       K1      V5      ← K1 的最新值

压缩后：
Offset  Key     Value
3       K3      V4
5       K1      V5

特点：
├── K1 只保留最新值 V5
├── K2 被删除（墓碑消息）
├── Offset 不连续但保持顺序
└── 墓碑消息保留一段时间后删除
```

### 使用场景

```
适用场景：
├── 数据库 CDC（变更数据捕获）
├── 用户状态/配置存储
├── Key-Value 缓存
└── 事件溯源中的快照

不适用场景：
├── 需要完整历史记录
├── 无 Key 的消息
└── Key 基数过大
```

### 配置

```properties
# Topic 级别配置
log.cleanup.policy=compact              # 或 delete,compact
min.cleanable.dirty.ratio=0.5           # 脏数据比例阈值
log.cleaner.min.compaction.lag.ms=0     # 压缩前最小等待时间
delete.retention.ms=86400000            # 墓碑消息保留时间
min.compaction.lag.ms=0                 # 消息写入后多久才能被压缩
```

## 五、配额管理（Quota）

### 概念

限制客户端的资源使用，防止单个客户端影响整个集群。

```
配额类型：
├── 生产者吞吐量（bytes/sec）
├── 消费者吞吐量（bytes/sec）
├── 请求处理时间（%）
└── 连接数（Kafka 2.7+）
```

### 配置

```bash
# 设置用户级别配额
kafka-configs.sh --bootstrap-server localhost:9092 \
  --alter \
  --add-config 'producer_byte_rate=1048576,consumer_byte_rate=2097152' \
  --entity-type users \
  --entity-name alice

# 设置客户端级别配额
kafka-configs.sh --bootstrap-server localhost:9092 \
  --alter \
  --add-config 'producer_byte_rate=1048576' \
  --entity-type clients \
  --entity-name my-producer

# 设置用户 + 客户端组合配额
kafka-configs.sh --bootstrap-server localhost:9092 \
  --alter \
  --add-config 'producer_byte_rate=1048576' \
  --entity-type users \
  --entity-name alice \
  --entity-type clients \
  --entity-name my-producer

# 设置默认配额
kafka-configs.sh --bootstrap-server localhost:9092 \
  --alter \
  --add-config 'producer_byte_rate=1048576' \
  --entity-type users \
  --entity-default

# 查看配额
kafka-configs.sh --bootstrap-server localhost:9092 \
  --describe \
  --entity-type users \
  --entity-name alice

# 删除配额
kafka-configs.sh --bootstrap-server localhost:9092 \
  --alter \
  --delete-config 'producer_byte_rate' \
  --entity-type users \
  --entity-name alice
```

## 六、跨数据中心复制

### MirrorMaker 2.0

Kafka 自带的跨集群复制工具。

```
架构：
┌─────────────────┐                    ┌─────────────────┐
│  Source Cluster │                    │  Target Cluster │
│    (US-East)    │                    │    (US-West)    │
│                 │                    │                 │
│  ┌───────────┐  │   MirrorMaker 2   │  ┌───────────┐  │
│  │  Topic A  │  │ =================>│  │  Topic A  │  │
│  │  Topic B  │  │                    │  │  Topic B  │  │
│  └───────────┘  │                    │  └───────────┘  │
└─────────────────┘                    └─────────────────┘

特点：
├── 保留原始 Topic 名称（或添加前缀）
├── 支持 Offset 同步
├── 支持 ACL 同步
├── 支持消费者组位点同步
└── 基于 Kafka Connect
```

### 配置示例

```properties
# mm2.properties

# 集群定义
clusters=source,target
source.bootstrap.servers=source-kafka:9092
target.bootstrap.servers=target-kafka:9092

# 复制配置
source->target.enabled=true
source->target.topics=.*            # 复制所有 Topic
source->target.topics.blacklist=internal.*  # 排除内部 Topic

# 同步配置
sync.topic.configs.enabled=true     # 同步 Topic 配置
sync.topic.acls.enabled=true        # 同步 ACL
emit.checkpoints.enabled=true       # 发送检查点
emit.heartbeats.enabled=true        # 发送心跳

# 复制策略
replication.factor=3                # 目标集群副本因子
refresh.topics.interval.seconds=60  # Topic 刷新间隔
refresh.groups.interval.seconds=60  # 消费者组刷新间隔

# 消费者组 Offset 同步
sync.group.offsets.enabled=true
sync.group.offsets.interval.seconds=60
```

### 启动命令

```bash
# 启动 MirrorMaker 2
connect-mirror-maker.sh mm2.properties

# 或作为 Kafka Connect 运行
# 需要配置 Connect Worker 和 MM2 Connector
```

## 七、Kafka Streams

### 概念

Kafka Streams 是一个流处理库，用于构建实时数据处理应用。

```
特点：
├── 轻量级：只是一个库，不需要独立集群
├── 状态存储：内置 RocksDB 状态存储
├── 容错：状态自动备份到 Kafka
├── 精确一次：支持 Exactly Once 语义
└── 时间语义：支持事件时间和处理时间

核心概念：
├── KStream：记录流（每条消息都是独立事件）
├── KTable：变更日志流（Key 的最新状态）
├── GlobalKTable：全局表（所有实例都有完整数据）
└── Processor API：底层 API，更灵活
```

### 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      Kafka Streams App                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │   Source    │ -> │  Processor  │ -> │    Sink     │        │
│   │ (KStream)   │    │ (Transform) │    │ (KStream)   │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│         │                   │                   │               │
│         │            ┌──────┴──────┐            │               │
│         │            │ State Store │            │               │
│         │            │ (RocksDB)   │            │               │
│         │            └──────┬──────┘            │               │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│   ┌─────────────────────────────────────────────────────┐      │
│   │                  Kafka Cluster                       │      │
│   │  Input Topic    Changelog Topic    Output Topic     │      │
│   └─────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### 代码示例（Java）

```java
// 配置
Properties props = new Properties();
props.put(StreamsConfig.APPLICATION_ID_CONFIG, "word-count");
props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
props.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
props.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass());

// 构建拓扑
StreamsBuilder builder = new StreamsBuilder();

// 从输入 Topic 读取
KStream<String, String> textLines = builder.stream("input-topic");

// 处理：拆分单词、分组、计数
KTable<String, Long> wordCounts = textLines
    .flatMapValues(line -> Arrays.asList(line.toLowerCase().split("\\W+")))
    .groupBy((key, word) -> word)
    .count(Materialized.as("word-counts-store"));

// 写入输出 Topic
wordCounts.toStream().to("output-topic",
    Produced.with(Serdes.String(), Serdes.Long()));

// 启动应用
KafkaStreams streams = new KafkaStreams(builder.build(), props);
streams.start();

// 优雅关闭
Runtime.getRuntime().addShutdownHook(new Thread(streams::close));
```

### 窗口操作

```java
// 滚动窗口（Tumbling Window）
KTable<Windowed<String>, Long> tumblingCounts = stream
    .groupByKey()
    .windowedBy(TimeWindows.of(Duration.ofMinutes(5)))
    .count();

// 跳跃窗口（Hopping Window）
KTable<Windowed<String>, Long> hoppingCounts = stream
    .groupByKey()
    .windowedBy(TimeWindows.of(Duration.ofMinutes(5)).advanceBy(Duration.ofMinutes(1)))
    .count();

// 会话窗口（Session Window）
KTable<Windowed<String>, Long> sessionCounts = stream
    .groupByKey()
    .windowedBy(SessionWindows.with(Duration.ofMinutes(5)))
    .count();

// 滑动窗口（Sliding Window）- Kafka 2.7+
KTable<Windowed<String>, Long> slidingCounts = stream
    .groupByKey()
    .windowedBy(SlidingWindows.withTimeDifferenceAndGrace(
        Duration.ofMinutes(5),
        Duration.ofMinutes(1)))
    .count();
```

## 八、Kafka Connect

### 概念

Kafka Connect 是数据集成框架，用于在 Kafka 和外部系统间移动数据。

```
组件：
├── Connector：定义如何与外部系统交互
│   ├── Source Connector：从外部系统导入数据到 Kafka
│   └── Sink Connector：从 Kafka 导出数据到外部系统
├── Task：实际执行数据传输的工作单元
├── Worker：运行 Connector 和 Task 的进程
└── Converter：数据格式转换（JSON、Avro 等）

运行模式：
├── Standalone：单进程运行，适合开发测试
└── Distributed：多 Worker 集群，适合生产环境
```

### 常用 Connector

```
Source Connector：
├── debezium-connector-mysql：MySQL CDC
├── debezium-connector-postgres：PostgreSQL CDC
├── kafka-connect-jdbc：JDBC 数据源
├── kafka-connect-elasticsearch：从 ES 读取
└── kafka-connect-s3-source：从 S3 读取

Sink Connector：
├── kafka-connect-jdbc：写入关系数据库
├── kafka-connect-elasticsearch：写入 ES
├── kafka-connect-s3：写入 S3
├── kafka-connect-hdfs：写入 HDFS
└── kafka-connect-redis：写入 Redis
```

### 配置示例

```properties
# connect-distributed.properties

# Kafka 连接
bootstrap.servers=localhost:9092
group.id=connect-cluster

# 存储配置
config.storage.topic=connect-configs
offset.storage.topic=connect-offsets
status.storage.topic=connect-status

# 数据格式
key.converter=org.apache.kafka.connect.json.JsonConverter
value.converter=org.apache.kafka.connect.json.JsonConverter
key.converter.schemas.enable=false
value.converter.schemas.enable=false

# REST API 端口
rest.port=8083
```

### 启动和管理

```bash
# 启动 Connect Worker
connect-distributed.sh connect-distributed.properties

# 查看已安装的 Connector
curl http://localhost:8083/connector-plugins

# 创建 Connector
curl -X POST http://localhost:8083/connectors \
  -H "Content-Type: application/json" \
  -d '{
    "name": "jdbc-source",
    "config": {
      "connector.class": "io.confluent.connect.jdbc.JdbcSourceConnector",
      "connection.url": "jdbc:mysql://localhost:3306/mydb",
      "connection.user": "user",
      "connection.password": "password",
      "table.whitelist": "users,orders",
      "mode": "incrementing",
      "incrementing.column.name": "id",
      "topic.prefix": "mysql-"
    }
  }'

# 查看 Connector 状态
curl http://localhost:8083/connectors/jdbc-source/status

# 暂停 Connector
curl -X PUT http://localhost:8083/connectors/jdbc-source/pause

# 恢复 Connector
curl -X PUT http://localhost:8083/connectors/jdbc-source/resume

# 删除 Connector
curl -X DELETE http://localhost:8083/connectors/jdbc-source
```

## 九、安全配置

### 认证机制

```
支持的认证方式：
├── SASL/PLAIN：用户名密码（明文，需配合 TLS）
├── SASL/SCRAM：加盐哈希密码
├── SASL/GSSAPI：Kerberos
├── SASL/OAUTHBEARER：OAuth 2.0
└── mTLS：双向 TLS 证书认证
```

### SASL/SCRAM 配置

```bash
# 创建 SCRAM 用户
kafka-configs.sh --bootstrap-server localhost:9092 \
  --alter \
  --add-config 'SCRAM-SHA-256=[password=secret],SCRAM-SHA-512=[password=secret]' \
  --entity-type users \
  --entity-name alice
```

```properties
# server.properties（Broker 配置）
listeners=SASL_PLAINTEXT://0.0.0.0:9092
security.inter.broker.protocol=SASL_PLAINTEXT
sasl.mechanism.inter.broker.protocol=SCRAM-SHA-256
sasl.enabled.mechanisms=SCRAM-SHA-256

# client.properties（客户端配置）
security.protocol=SASL_PLAINTEXT
sasl.mechanism=SCRAM-SHA-256
sasl.jaas.config=org.apache.kafka.common.security.scram.ScramLoginModule required \
  username="alice" \
  password="secret";
```

### TLS 加密

```properties
# server.properties
listeners=SSL://0.0.0.0:9093
ssl.keystore.location=/path/to/keystore.jks
ssl.keystore.password=keystorepassword
ssl.key.password=keypassword
ssl.truststore.location=/path/to/truststore.jks
ssl.truststore.password=truststorepassword

# 双向 TLS（mTLS）
ssl.client.auth=required

# client.properties
security.protocol=SSL
ssl.truststore.location=/path/to/truststore.jks
ssl.truststore.password=truststorepassword
# 如果是 mTLS
ssl.keystore.location=/path/to/client-keystore.jks
ssl.keystore.password=keystorepassword
ssl.key.password=keypassword
```

### ACL 授权

```bash
# 允许用户 alice 读写 topic1
kafka-acls.sh --bootstrap-server localhost:9092 \
  --add \
  --allow-principal User:alice \
  --operation Read \
  --operation Write \
  --topic topic1

# 允许用户 alice 使用消费者组
kafka-acls.sh --bootstrap-server localhost:9092 \
  --add \
  --allow-principal User:alice \
  --operation Read \
  --group my-group

# 拒绝用户 bob 的所有操作
kafka-acls.sh --bootstrap-server localhost:9092 \
  --add \
  --deny-principal User:bob \
  --operation All \
  --topic '*'
```

## 十、故障排查

### 常见问题

```
1. 消息丢失
├── 检查 acks 配置
├── 检查 min.insync.replicas
├── 检查 unclean.leader.election.enable
└── 检查生产者重试配置

2. 消息重复
├── 开启幂等性：enable.idempotence=true
├── 使用事务
└── 消费者端做幂等处理

3. 消费延迟（Lag）
├── 增加消费者数量（不超过分区数）
├── 优化消费者处理逻辑
├── 增加 max.poll.records
└── 检查是否频繁 Rebalance

4. Rebalance 频繁
├── 增加 session.timeout.ms
├── 增加 max.poll.interval.ms
├── 减少单次处理时间
└── 使用 CooperativeSticky 分配策略

5. Broker 内存溢出
├── 检查 page cache 使用
├── 调整 JVM 堆大小
├── 检查日志段大小
└── 清理过期数据
```

### 诊断命令

```bash
# 查看 Topic 的 ISR
kafka-topics.sh --bootstrap-server localhost:9092 \
  --describe \
  --topic my-topic

# 查看消费延迟
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --describe \
  --group my-group

# 查看 Broker 日志目录使用情况
kafka-log-dirs.sh --bootstrap-server localhost:9092 \
  --describe \
  --broker-list 0

# 验证副本同步
kafka-replica-verification.sh \
  --broker-list localhost:9092 \
  --topic-white-list my-topic

# 查看控制器信息
kafka-metadata.sh --snapshot /var/kafka-logs/__cluster_metadata-0/00000000000000000000.log \
  --command "cat"

# 检查 Broker 健康
kafka-broker-api-versions.sh --bootstrap-server localhost:9092
```

### JMX 监控指标

```
关键监控指标：

# 消息吞吐
kafka.server:type=BrokerTopicMetrics,name=MessagesInPerSec
kafka.server:type=BrokerTopicMetrics,name=BytesInPerSec
kafka.server:type=BrokerTopicMetrics,name=BytesOutPerSec

# 请求延迟
kafka.network:type=RequestMetrics,name=TotalTimeMs,request=Produce
kafka.network:type=RequestMetrics,name=TotalTimeMs,request=FetchConsumer

# 副本状态
kafka.server:type=ReplicaManager,name=UnderReplicatedPartitions
kafka.server:type=ReplicaManager,name=IsrShrinksPerSec
kafka.server:type=ReplicaManager,name=IsrExpandsPerSec

# 控制器状态
kafka.controller:type=KafkaController,name=ActiveControllerCount
kafka.controller:type=KafkaController,name=OfflinePartitionsCount

# 日志状态
kafka.log:type=LogFlushStats,name=LogFlushRateAndTimeMs
kafka.server:type=LogManager,name=LogDirectoryOffline
```
