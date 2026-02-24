# 大数据数据采集与同步实战

## 1. 数据采集架构概览

### 1.1 数据源分类

在大数据平台中，数据采集是整个数据链路的起点。根据数据的结构化程度，可以将数据源分为三大类：

| 分类 | 数据源类型 | 典型代表 | 格式特征 | 采集方式 |
|------|-----------|---------|---------|---------|
| 结构化数据 | RDBMS | MySQL, PostgreSQL, Oracle, SQL Server | 行列表结构，有严格Schema | Sqoop, DataX, Canal, Debezium |
| 半结构化数据 | 日志/消息 | Nginx日志, App日志, JSON, XML, CSV | 有一定格式但非严格Schema | Flume, Filebeat, Logstash |
| 非结构化数据 | 多媒体/文档 | 图片, 视频, PDF, Word | 无固定格式，二进制为主 | 自定义爬虫, SDK上传, API采集 |

**完整数据采集架构**：

```
数据采集平台全景架构
┌─────────────────────────────────────────────────────────────────────────────┐
│                            数据源层 (Data Sources)                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  MySQL   │  │ Oracle   │  │ App Logs │  │ Nginx    │  │ REST API │    │
│  │  (CDC)   │  │ (Batch)  │  │ (Stream) │  │ (Stream) │  │ (Pull)   │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │              │              │              │              │          │
└───────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────┘
        │              │              │              │              │
        ↓              ↓              ↓              ↓              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          采集工具层 (Ingestion)                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Canal   │  │  Sqoop   │  │  Flume   │  │ Filebeat │  │  Custom  │    │
│  │ Debezium │  │  DataX   │  │          │  │ Logstash │  │  Script  │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │              │              │              │              │          │
└───────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────┘
        │              │              │              │              │
        ↓              ↓              ↓              ↓              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      消息中间件层 (Message Bus)                              │
│                    ┌──────────────────────────┐                              │
│                    │      Apache Kafka        │                              │
│                    │  (统一数据汇聚与缓冲)      │                              │
│                    └─────────────┬────────────┘                              │
│                                  │                                           │
└──────────────────────────────────┼───────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ↓              ↓              ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        存储与计算层 (Storage & Compute)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  HDFS/Hive   │  │Elasticsearch │  │  HBase       │  │  ClickHouse  │   │
│  │  (数据湖)     │  │  (搜索引擎)   │  │  (宽表存储)   │  │  (OLAP分析)   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

核心设计原则：

- **统一汇聚**：所有数据源优先写入Kafka，实现数据总线解耦
- **多路分发**：从Kafka消费数据写入不同下游存储，按场景选择
- **分层采集**：实时数据走CDC/日志流，离线数据走批量同步
- **容错保障**：每个环节具备断点续传和数据校验能力

### 1.2 采集工具选型

| 工具 | 类型 | 协议/接口 | 吞吐量 | Exactly-Once | 语言 | 典型场景 |
|------|------|----------|--------|-------------|------|---------|
| **Flume** | 日志采集 | Avro, Thrift, HTTP | 高(万级EPS) | Channel级别 | Java | Hadoop生态日志采集 |
| **Logstash** | 日志处理 | 多种Input/Output | 中(依赖Filter) | At-Least-Once | JRuby | ELK日志分析 |
| **Filebeat** | 日志转发 | Registry机制 | 高(轻量) | At-Least-Once | Go | 轻量日志转发到Logstash/Kafka |
| **Sqoop** | 批量同步 | JDBC/MapReduce | 高(并行MR) | 不支持 | Java | RDBMS ↔ HDFS离线同步 |
| **DataX** | 批量同步 | Reader/Writer插件 | 高(多线程) | 不支持 | Java/Python | 异构数据源离线同步 |
| **Canal** | 实时CDC | MySQL Binlog | 高(毫秒延迟) | At-Least-Once | Java | MySQL实时变更捕获 |
| **Debezium** | 实时CDC | Kafka Connect | 高(毫秒延迟) | Exactly-Once* | Java | 多数据库CDC(PostgreSQL/MongoDB等) |

> *Debezium配合Kafka事务可实现Exactly-Once语义。

**选型核心指标**：

- **延迟要求**：秒级以内选CDC工具（Canal/Debezium），分钟级选日志采集，小时级选批量同步
- **数据源类型**：数据库变更选CDC，应用日志选Flume/Filebeat，跨源批量选DataX
- **团队技术栈**：Hadoop生态优先Flume/Sqoop，ELK生态优先Filebeat/Logstash，Kafka生态优先Debezium

---

## 2. Apache Flume日志采集

### 2.1 Flume架构

Apache Flume是Hadoop生态中最经典的日志采集框架，采用Agent-based架构，每个Agent由Source、Channel、Sink三个核心组件组成。

**单Agent基础架构**：

```
Flume Agent 基础架构
┌─────────────────────────────────────────────────────────┐
│                      Flume Agent                         │
│                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Source   │───→│   Channel    │───→│    Sink      │   │
│  │ (数据输入) │    │  (缓冲通道)   │    │  (数据输出)   │   │
│  └──────────┘    └──────────────┘    └──────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │            Interceptor Chain                      │   │
│  │  [Timestamp] → [Host] → [Regex Filter] → ...     │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**多Agent串联（Agent Chaining）**：

```
多Agent串联架构 —— 用于跨网络/跨机房传输
┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
│   Agent 1 (采集)   │      │   Agent 2 (汇聚)   │      │   Agent 3 (存储)   │
│                    │      │                    │      │                    │
│ Taildir  → File   │─Avro→│  Avro   → File    │─Avro→│  Avro  → HDFS     │
│ Source     Channel │  RPC │  Source   Channel  │  RPC │  Source  Sink     │
└───────────────────┘      └───────────────────┘      └───────────────────┘
  (边缘节点/App服务器)         (汇聚层/中转节点)           (存储层/Hadoop集群)
```

**多路复用扇出（Multiplexing Fan-out）**：

```
Flume 多路复用扇出架构
                         ┌──────────────┐    ┌──────────────┐
                    ┌───→│ Channel 1    │───→│ Sink 1       │──→ HDFS
                    │    │ (Memory)     │    │ (hdfs)       │
┌──────────┐        │    └──────────────┘    └──────────────┘
│  Source   │───(Selector)
│ (Taildir) │        │    ┌──────────────┐    ┌──────────────┐
│           │        └───→│ Channel 2    │───→│ Sink 2       │──→ Kafka
└──────────┘              │ (Kafka)      │    │ (kafka)      │
                          └──────────────┘    └──────────────┘

Selector类型：
  - replicating: 复制模式，Event发送到所有Channel（默认）
  - multiplexing: 路由模式，根据Header值路由到不同Channel
```

### 2.2 核心组件

**Source类型**：

| Source | 描述 | 持久化 | 适用场景 |
|--------|------|--------|---------|
| **Avro Source** | 接收Avro RPC调用 | 依赖Channel | 多Agent串联，跨节点传输 |
| **Exec Source** | 执行Unix命令（如tail -F） | 不支持 | 简单测试，不推荐生产用 |
| **Taildir Source** | 监控目录下文件变化（支持断点续传） | 支持（position file） | 生产环境日志采集首选 |
| **Kafka Source** | 从Kafka消费数据 | Kafka Offset | Kafka → HDFS/HBase等 |
| **Syslog Source** | 接收Syslog协议数据 | 依赖Channel | 系统日志/网络设备日志 |
| **Spooling Dir Source** | 监控目录新文件（整文件消费） | 文件重命名标记 | 文件落地后消费 |
| **HTTP Source** | 接收HTTP POST请求 | 依赖Channel | SDK埋点数据上报 |

**Channel类型**：

| Channel | 描述 | 容量 | 可靠性 | 性能 | 适用场景 |
|---------|------|------|--------|------|---------|
| **Memory Channel** | 内存队列 | 受JVM堆限制 | 低（进程崩溃丢数据） | 高 | 容许少量丢失的高吞吐场景 |
| **File Channel** | 本地磁盘WAL日志 | 受磁盘限制 | 高（事务写磁盘） | 中 | 生产环境，要求不丢数据 |
| **Kafka Channel** | Kafka作为Channel | Kafka集群容量 | 高（Kafka副本机制） | 高 | 可直接省略Sink，Kafka既做Channel也做输出 |
| **JDBC Channel** | 数据库存储 | 受DB限制 | 高 | 低 | 特殊需求，很少使用 |

**Sink类型**：

| Sink | 描述 | 批量写入 | 适用场景 |
|------|------|---------|---------|
| **HDFS Sink** | 写入HDFS文件 | 支持（roll策略） | 日志入数据湖/数据仓库 |
| **HBase Sink** | 写入HBase表 | 支持（批量Put） | 实时查询宽表 |
| **Kafka Sink** | 写入Kafka Topic | 支持（batch.size） | 数据总线中转 |
| **Avro Sink** | 发送Avro RPC到下游Agent | 支持 | 多Agent串联 |
| **ElasticSearch Sink** | 写入Elasticsearch | 支持（Bulk API） | 日志搜索分析 |
| **Hive Sink** | 写入Hive事务表 | 支持 | ORC事务表实时写入 |
| **Null Sink** | 丢弃数据 | - | 测试/调试 |

### 2.3 配置实战

**示例1：Taildir Source → File Channel → HDFS Sink（生产级配置）**

```properties
# flume-taildir-hdfs.conf
# 定义Agent组件名称
a1.sources = s1
a1.channels = c1
a1.sinks = k1

# ========== Source配置：Taildir Source ==========
a1.sources.s1.type = TAILDIR
# 文件组配置，可监控多个目录
a1.sources.s1.filegroups = fg1 fg2
# 采集应用日志（支持正则匹配）
a1.sources.s1.filegroups.fg1 = /data/logs/app/.*\\.log
a1.sources.s1.headers.fg1.logType = app
# 采集Nginx访问日志
a1.sources.s1.filegroups.fg2 = /data/logs/nginx/access\\.log.*
a1.sources.s1.headers.fg2.logType = nginx
# Position文件（断点续传的关键，记录每个文件的读取偏移量）
a1.sources.s1.positionFile = /data/flume/position/taildir_position.json
# 每次读取的最大行数
a1.sources.s1.batchSize = 1000
# 文件检测间隔（毫秒）
a1.sources.s1.fileHeader = true
a1.sources.s1.fileHeaderKey = file

# Interceptor拦截器链
a1.sources.s1.interceptors = i1 i2 i3
# 添加时间戳Header
a1.sources.s1.interceptors.i1.type = timestamp
# 添加主机名Header
a1.sources.s1.interceptors.i2.type = host
a1.sources.s1.interceptors.i2.hostHeader = hostname
# 正则过滤：排除空行和注释行
a1.sources.s1.interceptors.i3.type = regex_filter
a1.sources.s1.interceptors.i3.regex = ^\\s*$|^#.*
a1.sources.s1.interceptors.i3.excludeEvents = true

a1.sources.s1.channels = c1

# ========== Channel配置：File Channel ==========
a1.channels.c1.type = file
# Checkpoint目录
a1.channels.c1.checkpointDir = /data/flume/checkpoint
# 数据目录（支持多磁盘提高IO）
a1.channels.c1.dataDirs = /data1/flume/data /data2/flume/data
# Channel最大容量（Event个数）
a1.channels.c1.capacity = 10000000
# 每次事务最大Event数
a1.channels.c1.transactionCapacity = 10000
# Checkpoint间隔（秒）
a1.channels.c1.checkpointInterval = 300000

# ========== Sink配置：HDFS Sink ==========
a1.sinks.k1.type = hdfs
a1.sinks.k1.channel = c1
# HDFS路径（按日志类型和日期分区）
a1.sinks.k1.hdfs.path = hdfs://nameservice1/data/raw/%{logType}/%Y-%m-%d/%H
# 文件前缀
a1.sinks.k1.hdfs.filePrefix = events
# 文件后缀
a1.sinks.k1.hdfs.fileSuffix = .log
# 文件滚动策略
a1.sinks.k1.hdfs.rollInterval = 3600
a1.sinks.k1.hdfs.rollSize = 134217728
a1.sinks.k1.hdfs.rollCount = 0
# 时间取整（按小时取整路径中的时间）
a1.sinks.k1.hdfs.round = true
a1.sinks.k1.hdfs.roundValue = 1
a1.sinks.k1.hdfs.roundUnit = hour
# 文件格式（推荐压缩格式）
a1.sinks.k1.hdfs.fileType = CompressedStream
a1.sinks.k1.hdfs.codeC = lzop
# 使用本地时间戳（而非Event Header中的时间戳）
a1.sinks.k1.hdfs.useLocalTimeStamp = false
# 批量写入大小
a1.sinks.k1.hdfs.batchSize = 1000
# 空闲超时关闭文件（秒），防止小文件
a1.sinks.k1.hdfs.idleTimeout = 120
```

**示例2：Kafka Source → Memory Channel → HDFS Sink**

```properties
# flume-kafka-hdfs.conf
a2.sources = kafka_source
a2.channels = mem_channel
a2.sinks = hdfs_sink

# ========== Kafka Source ==========
a2.sources.kafka_source.type = org.apache.flume.source.kafka.KafkaSource
a2.sources.kafka_source.kafka.bootstrap.servers = kafka01:9092,kafka02:9092,kafka03:9092
a2.sources.kafka_source.kafka.topics = user_behavior,order_events
a2.sources.kafka_source.kafka.consumer.group.id = flume-hdfs-consumer
# 每次Poll拉取的最大消息数
a2.sources.kafka_source.batchSize = 2000
# 每批次最大等待时间（毫秒）
a2.sources.kafka_source.batchDurationMillis = 2000
# 从最早的Offset开始消费
a2.sources.kafka_source.kafka.consumer.auto.offset.reset = earliest
# 将Kafka Topic名称放入Header
a2.sources.kafka_source.setTopicHeader = true
a2.sources.kafka_source.topicHeader = topic
a2.sources.kafka_source.channels = mem_channel

# ========== Memory Channel ==========
a2.channels.mem_channel.type = memory
a2.channels.mem_channel.capacity = 100000
a2.channels.mem_channel.transactionCapacity = 5000
# 每个Event的最大字节数
a2.channels.mem_channel.byteCapacityBufferPercentage = 20
a2.channels.mem_channel.byteCapacity = 800000

# ========== HDFS Sink ==========
a2.sinks.hdfs_sink.type = hdfs
a2.sinks.hdfs_sink.channel = mem_channel
a2.sinks.hdfs_sink.hdfs.path = hdfs://nameservice1/data/kafka/%{topic}/%Y-%m-%d/%H
a2.sinks.hdfs_sink.hdfs.filePrefix = kafka
a2.sinks.hdfs_sink.hdfs.fileSuffix = .gz
a2.sinks.hdfs_sink.hdfs.rollInterval = 1800
a2.sinks.hdfs_sink.hdfs.rollSize = 268435456
a2.sinks.hdfs_sink.hdfs.rollCount = 0
a2.sinks.hdfs_sink.hdfs.fileType = CompressedStream
a2.sinks.hdfs_sink.hdfs.codeC = gzip
a2.sinks.hdfs_sink.hdfs.batchSize = 2000
a2.sinks.hdfs_sink.hdfs.round = true
a2.sinks.hdfs_sink.hdfs.roundValue = 1
a2.sinks.hdfs_sink.hdfs.roundUnit = hour
```

**Interceptor拦截器详解**：

```properties
# 拦截器链配置示例（可组合多个拦截器）
a1.sources.s1.interceptors = ts host regex_filter static_header

# 1. Timestamp拦截器 —— 在Header中添加时间戳
a1.sources.s1.interceptors.ts.type = timestamp
a1.sources.s1.interceptors.ts.preserveExisting = true

# 2. Host拦截器 —— 在Header中添加主机信息
a1.sources.s1.interceptors.host.type = host
a1.sources.s1.interceptors.host.hostHeader = hostname
a1.sources.s1.interceptors.host.useIP = false

# 3. Regex Filter拦截器 —— 正则匹配过滤Event
a1.sources.s1.interceptors.regex_filter.type = regex_filter
# 只保留包含ERROR或WARN的日志
a1.sources.s1.interceptors.regex_filter.regex = .*(ERROR|WARN).*
a1.sources.s1.interceptors.regex_filter.excludeEvents = false

# 4. Static拦截器 —— 添加固定Header
a1.sources.s1.interceptors.static_header.type = static
a1.sources.s1.interceptors.static_header.key = datacenter
a1.sources.s1.interceptors.static_header.value = beijing-dc1
```

### 2.4 高可用

**Failover Sink Processor（故障转移）**：

```properties
# 多Agent故障转移配置
a1.sinkgroups = g1
a1.sinkgroups.g1.sinks = k1 k2
a1.sinkgroups.g1.processor.type = failover
# 优先级：数值越大优先级越高
a1.sinkgroups.g1.processor.priority.k1 = 10
a1.sinkgroups.g1.processor.priority.k2 = 5
# 最大惩罚时间（毫秒），Sink故障后的退避上限
a1.sinkgroups.g1.processor.maxpenalty = 10000

# 主Sink：发往主集群
a1.sinks.k1.type = avro
a1.sinks.k1.hostname = collector-primary.example.com
a1.sinks.k1.port = 4545
a1.sinks.k1.channel = c1

# 备Sink：发往备集群
a1.sinks.k2.type = avro
a1.sinks.k2.hostname = collector-standby.example.com
a1.sinks.k2.port = 4545
a1.sinks.k2.channel = c1
```

**Load Balancing Sink Processor（负载均衡）**：

```properties
# 负载均衡配置
a1.sinkgroups = g1
a1.sinkgroups.g1.sinks = k1 k2 k3
a1.sinkgroups.g1.processor.type = load_balance
# 负载均衡策略：round_robin 或 random
a1.sinkgroups.g1.processor.selector = round_robin
# 在所有Sink间均匀分配
a1.sinkgroups.g1.processor.backoff = true
a1.sinkgroups.g1.processor.selector.maxTimeOut = 30000

a1.sinks.k1.type = avro
a1.sinks.k1.hostname = collector01.example.com
a1.sinks.k1.port = 4545
a1.sinks.k1.channel = c1

a1.sinks.k2.type = avro
a1.sinks.k2.hostname = collector02.example.com
a1.sinks.k2.port = 4545
a1.sinks.k2.channel = c1

a1.sinks.k3.type = avro
a1.sinks.k3.hostname = collector03.example.com
a1.sinks.k3.port = 4545
a1.sinks.k3.channel = c1
```

---

## 3. Logstash/Filebeat日志采集

### 3.1 ELK架构

ELK（Elasticsearch + Logstash + Kibana）及其扩展EFK（加入Filebeat）是日志分析领域的主流方案。在大数据场景中，通常将Kafka作为缓冲层引入，形成完整的日志采集分析链路。

```
ELK/EFK 日志采集架构
┌───────────────────────────────────────────────────────────────────────┐
│                        应用服务器集群                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ App 01   │  │ App 02   │  │ App 03   │  │ App N    │             │
│  │┌────────┐│  │┌────────┐│  │┌────────┐│  │┌────────┐│             │
│  ││Filebeat││  ││Filebeat││  ││Filebeat││  ││Filebeat││             │
│  │└───┬────┘│  │└───┬────┘│  │└───┬────┘│  │└───┬────┘│             │
│  └────┼─────┘  └────┼─────┘  └────┼─────┘  └────┼─────┘             │
│       │              │              │              │                   │
└───────┼──────────────┼──────────────┼──────────────┼───────────────────┘
        │              │              │              │
        └──────────────┼──────────────┼──────────────┘
                       ↓              ↓
              ┌─────────────────────────────┐
              │        Apache Kafka         │
              │   (日志缓冲 & 削峰填谷)      │
              └─────────────┬───────────────┘
                            │
              ┌─────────────┼───────────────┐
              ↓                             ↓
  ┌───────────────────┐         ┌───────────────────┐
  │    Logstash        │         │    Flink/Spark    │
  │  (ETL处理)         │         │  (流计算分析)      │
  │ Grok → Mutate      │         │                   │
  │   → GeoIP          │         │                   │
  └─────────┬─────────┘         └─────────┬─────────┘
            │                             │
            ↓                             ↓
  ┌───────────────────┐         ┌───────────────────┐
  │  Elasticsearch    │         │    HDFS / Hive    │
  │  (实时搜索/分析)   │         │  (离线数据仓库)    │
  └─────────┬─────────┘         └───────────────────┘
            │
            ↓
  ┌───────────────────┐
  │     Kibana        │
  │  (可视化仪表盘)    │
  └───────────────────┘
```

### 3.2 Logstash Pipeline

Logstash的核心是Pipeline，由Input → Filter → Output三个阶段构成。

**完整Logstash配置示例**：

```ruby
# /etc/logstash/conf.d/main-pipeline.conf

# ========== Input: 多数据源输入 ==========
input {
  # 1. 从Kafka消费日志
  kafka {
    bootstrap_servers => "kafka01:9092,kafka02:9092,kafka03:9092"
    topics => ["app-logs", "nginx-logs"]
    group_id => "logstash-consumer-group"
    consumer_threads => 3
    codec => json
    # 将Kafka Topic名称记录到metadata
    decorate_events => true
    auto_offset_reset => "latest"
  }

  # 2. 从文件读取日志（备用/测试）
  file {
    path => ["/var/log/app/*.log"]
    start_position => "beginning"
    sincedb_path => "/var/lib/logstash/sincedb_app"
    codec => multiline {
      pattern => "^%{TIMESTAMP_ISO8601}"
      negate => true
      what => "previous"
    }
  }

  # 3. 从数据库增量读取（JDBC Input）
  jdbc {
    jdbc_driver_library => "/opt/logstash/drivers/mysql-connector-java-8.0.28.jar"
    jdbc_driver_class => "com.mysql.cj.jdbc.Driver"
    jdbc_connection_string => "jdbc:mysql://mysql-master:3306/business_db"
    jdbc_user => "logstash_reader"
    jdbc_password => "${MYSQL_PASSWORD}"
    schedule => "*/5 * * * *"
    statement => "SELECT * FROM orders WHERE update_time > :sql_last_value ORDER BY update_time"
    use_column_value => true
    tracking_column => "update_time"
    tracking_column_type => "timestamp"
    last_run_metadata_path => "/var/lib/logstash/jdbc_last_run"
  }
}

# ========== Filter: 数据清洗与转换 ==========
filter {
  # 根据来源Topic分别处理
  if [@metadata][kafka][topic] == "nginx-logs" {
    # Grok解析Nginx访问日志
    grok {
      match => {
        "message" => '%{IPORHOST:client_ip} - %{DATA:user_name} \[%{HTTPDATE:request_time}\] "%{WORD:http_method} %{DATA:request_uri} HTTP/%{NUMBER:http_version}" %{NUMBER:response_code} %{NUMBER:body_bytes_sent} "%{DATA:http_referer}" "%{DATA:user_agent}" %{NUMBER:request_duration}'
      }
      remove_field => ["message"]
    }

    # 日期解析
    date {
      match => ["request_time", "dd/MMM/yyyy:HH:mm:ss Z"]
      target => "@timestamp"
      remove_field => ["request_time"]
    }

    # GeoIP地理位置解析
    geoip {
      source => "client_ip"
      target => "geo"
      database => "/opt/logstash/GeoLite2-City.mmdb"
      fields => ["city_name", "country_name", "latitude", "longitude", "region_name"]
    }

    # 类型转换
    mutate {
      convert => {
        "response_code" => "integer"
        "body_bytes_sent" => "integer"
        "request_duration" => "float"
      }
      # 添加自定义字段
      add_field => { "log_type" => "nginx_access" }
      # 移除不需要的字段
      remove_field => ["@version", "host"]
    }

    # UA解析
    useragent {
      source => "user_agent"
      target => "ua"
    }
  }

  else if [@metadata][kafka][topic] == "app-logs" {
    # 应用日志通常是JSON格式
    json {
      source => "message"
      remove_field => ["message"]
    }

    # 日期处理
    date {
      match => ["timestamp", "yyyy-MM-dd HH:mm:ss.SSS", "ISO8601"]
      target => "@timestamp"
      remove_field => ["timestamp"]
    }

    mutate {
      add_field => { "log_type" => "application" }
    }
  }

  # 通用处理：丢弃健康检查请求
  if [request_uri] =~ /^\/health/ {
    drop { }
  }
}

# ========== Output: 多目标输出 ==========
output {
  # 1. 写入Elasticsearch（按日期索引）
  elasticsearch {
    hosts => ["es01:9200", "es02:9200", "es03:9200"]
    index => "%{log_type}-%{+YYYY.MM.dd}"
    user => "elastic"
    password => "${ES_PASSWORD}"
    # 使用ILM生命周期管理
    ilm_enabled => true
    ilm_rollover_alias => "logs"
    ilm_pattern => "000001"
    ilm_policy => "logs-lifecycle-policy"
  }

  # 2. 错误日志额外写入Kafka告警Topic
  if [level] == "ERROR" {
    kafka {
      bootstrap_servers => "kafka01:9092,kafka02:9092,kafka03:9092"
      topic_id => "alert-error-logs"
      codec => json
    }
  }

  # 3. 调试输出到控制台（生产环境注释掉）
  # stdout {
  #   codec => rubydebug
  # }
}
```

### 3.3 Filebeat轻量采集

Filebeat用Go语言编写，资源占用极低（通常仅需10-30MB内存），适合部署在每台应用服务器上作为日志采集Agent。

```yaml
# /etc/filebeat/filebeat.yml

# ========== Input配置 ==========
filebeat.inputs:

  # 1. 应用日志采集
  - type: log
    id: app-logs
    enabled: true
    paths:
      - /data/logs/app/*.log
      - /data/logs/app/**/*.log
    # 多行合并（Java异常堆栈）
    multiline.type: pattern
    multiline.pattern: '^\d{4}-\d{2}-\d{2}'
    multiline.negate: true
    multiline.match: after
    multiline.max_lines: 50
    # 自定义字段
    fields:
      log_type: application
      env: production
      service: order-service
    fields_under_root: true
    # 排除特定行
    exclude_lines: ['^DEBUG', '^\s*$']
    # 文件过期清理（7天未更新的文件停止监控）
    close_inactive: 5m
    clean_inactive: 168h
    ignore_older: 168h
    # 采集速率控制
    harvester_buffer_size: 16384
    max_bytes: 10485760

  # 2. Nginx访问日志
  - type: log
    id: nginx-access
    enabled: true
    paths:
      - /var/log/nginx/access.log*
    fields:
      log_type: nginx_access
    fields_under_root: true
    exclude_files: ['\.gz$']

  # 3. Nginx错误日志
  - type: log
    id: nginx-error
    enabled: true
    paths:
      - /var/log/nginx/error.log*
    fields:
      log_type: nginx_error
    fields_under_root: true
    multiline.type: pattern
    multiline.pattern: '^\d{4}/\d{2}/\d{2}'
    multiline.negate: true
    multiline.match: after

  # 4. 系统日志
  - type: log
    id: syslog
    enabled: true
    paths:
      - /var/log/syslog
      - /var/log/messages
    fields:
      log_type: system
    fields_under_root: true

# ========== 处理器（轻量ETL） ==========
processors:
  # 添加主机元数据
  - add_host_metadata:
      when.not.contains.tags: forwarded
  # 添加云服务商元数据（如果在云上）
  - add_cloud_metadata: ~
  # 丢弃调试事件
  - drop_event:
      when:
        regexp:
          message: "^DEBUG"
  # 重命名字段
  - rename:
      fields:
        - from: "host.name"
          to: "hostname"
      ignore_missing: true

# ========== Output配置 ==========
# 方案1：输出到Kafka（推荐生产使用）
output.kafka:
  hosts: ["kafka01:9092", "kafka02:9092", "kafka03:9092"]
  topic: '%{[log_type]}-logs'
  partition.round_robin:
    reachable_only: true
  required_acks: 1
  compression: gzip
  max_message_bytes: 10485760
  # Kafka Producer配置
  bulk_max_size: 2048
  worker: 3

# 方案2：输出到Logstash（需注释掉上面的Kafka配置）
# output.logstash:
#   hosts: ["logstash01:5044", "logstash02:5044"]
#   loadbalance: true
#   bulk_max_size: 1024
#   compression_level: 3

# ========== 通用配置 ==========
# Registry文件路径（记录采集进度）
filebeat.registry.path: /var/lib/filebeat/registry
filebeat.registry.flush: 5s

# 日志配置
logging.level: info
logging.to_files: true
logging.files:
  path: /var/log/filebeat
  name: filebeat.log
  keepfiles: 7
  permissions: 0644

# 监控（可选：将指标发送到Elasticsearch）
monitoring.enabled: true
monitoring.elasticsearch:
  hosts: ["es01:9200"]
  username: "beats_system"
  password: "${ES_MONITOR_PASSWORD}"
```

### 3.4 Grok模式匹配

Grok是Logstash中最核心的Filter插件，使用命名正则表达式模式来解析非结构化日志。

**Apache访问日志解析**：

```ruby
# Apache Combined Log Format 示例日志：
# 192.168.1.100 - frank [10/Oct/2024:13:55:36 -0700] "GET /api/users HTTP/1.1" 200 2326 "http://example.com" "Mozilla/5.0"

filter {
  grok {
    match => {
      "message" => '%{COMBINEDAPACHELOG}'
    }
  }
}

# 解析结果:
# clientip: 192.168.1.100
# ident: -
# auth: frank
# timestamp: 10/Oct/2024:13:55:36 -0700
# verb: GET
# request: /api/users
# httpversion: 1.1
# response: 200
# bytes: 2326
# referrer: http://example.com
# agent: Mozilla/5.0
```

**Nginx访问日志解析（自定义格式）**：

```ruby
# Nginx log_format：
# '$remote_addr - $remote_user [$time_local] "$request" $status $body_bytes_sent
#  "$http_referer" "$http_user_agent" $request_time $upstream_response_time'

# 示例日志：
# 10.0.0.1 - - [24/Feb/2026:10:30:15 +0800] "POST /api/order/create HTTP/1.1" 200 1523 "https://www.example.com/cart" "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0)" 0.152 0.148

filter {
  grok {
    match => {
      "message" => '%{IPORHOST:client_ip} - %{DATA:remote_user} \[%{HTTPDATE:time_local}\] "%{WORD:method} %{URIPATHPARAM:request_uri} HTTP/%{NUMBER:http_version}" %{NUMBER:status:int} %{NUMBER:body_bytes:int} "%{DATA:referer}" "%{DATA:user_agent}" %{NUMBER:request_time:float} %{NUMBER:upstream_time:float}'
    }
  }

  date {
    match => ["time_local", "dd/MMM/yyyy:HH:mm:ss Z"]
    target => "@timestamp"
  }

  # 从request_uri中提取API路径
  grok {
    match => {
      "request_uri" => '%{URIPATH:api_path}(?:\?%{GREEDYDATA:query_string})?'
    }
  }
}
```

**JSON结构化日志解析**：

```ruby
# 结构化JSON日志示例：
# {"timestamp":"2026-02-24T10:30:15.123Z","level":"ERROR","service":"order-service","traceId":"abc123","message":"Payment failed","exception":"java.lang.RuntimeException: Connection refused"}

filter {
  json {
    source => "message"
    target => "log"
  }

  date {
    match => ["[log][timestamp]", "ISO8601"]
    target => "@timestamp"
  }

  mutate {
    rename => {
      "[log][level]" => "level"
      "[log][service]" => "service"
      "[log][traceId]" => "trace_id"
      "[log][message]" => "log_message"
      "[log][exception]" => "exception"
    }
    remove_field => ["message", "log"]
  }
}
```

**Java异常堆栈解析**：

```ruby
# Java堆栈日志示例：
# 2026-02-24 10:30:15.123 ERROR [order-service,abc123] c.e.OrderService - Failed to process order
# java.lang.NullPointerException: Order item is null
#     at com.example.OrderService.process(OrderService.java:42)
#     at com.example.OrderController.create(OrderController.java:28)
#     at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
# Caused by: java.sql.SQLException: Connection pool exhausted
#     at com.zaxxer.hikari.HikariPool.getConnection(HikariPool.java:155)
#     ... 35 more

# 首先在Filebeat中配置多行合并
# multiline.pattern: '^\d{4}-\d{2}-\d{2}'
# multiline.negate: true
# multiline.match: after

filter {
  grok {
    match => {
      "message" => '%{TIMESTAMP_ISO8601:timestamp} %{LOGLEVEL:level} \[%{DATA:service},%{DATA:trace_id}\] %{DATA:logger} - %{GREEDYDATA:log_message}'
    }
  }

  # 提取异常类名
  if [log_message] =~ /Exception|Error/ {
    grok {
      match => {
        "log_message" => '%{JAVACLASS:exception_class}: %{GREEDYDATA:exception_message}'
      }
      tag_on_failure => ["_no_exception_parsed"]
    }
  }

  date {
    match => ["timestamp", "yyyy-MM-dd HH:mm:ss.SSS"]
    target => "@timestamp"
    remove_field => ["timestamp"]
  }
}
```

---

## 4. Sqoop离线数据同步

### 4.1 架构与原理

Apache Sqoop（SQL-to-Hadoop）是用于在RDBMS和Hadoop生态之间进行批量数据传输的工具。它的底层利用MapReduce实现并行数据传输。

```
Sqoop 工作原理
┌─────────────────────────────────────────────────────────────────────┐
│                        Sqoop Client                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  1. 连接数据库，获取表元数据（列名、类型、主键）               │   │
│  │  2. 生成Java Bean类（ORM映射）                               │   │
│  │  3. 根据--split-by列计算数据分片边界                          │   │
│  │  4. 提交MapReduce Job                                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
└───────────────────┬─────────────────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    YARN / MapReduce                                  │
│                                                                     │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐       │
│  │ Mapper 1  │  │ Mapper 2  │  │ Mapper 3  │  │ Mapper 4  │       │
│  │ id: 1~25k │  │id:25k~50k │  │id:50k~75k │  │id:75k~100k│       │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘       │
│        │               │               │               │             │
│   SELECT ...      SELECT ...      SELECT ...      SELECT ...        │
│   WHERE id>=1     WHERE id>=     WHERE id>=      WHERE id>=         │
│   AND id<25000    25000 AND...   50000 AND...    75000 AND...       │
│        │               │               │               │             │
└────────┼───────────────┼───────────────┼───────────────┼─────────────┘
         │               │               │               │
         ↓               ↓               ↓               ↓
┌─────────────────┐                          ┌────────────────────────┐
│     MySQL       │ ←── JDBC SELECT ───────→ │    HDFS / Hive         │
│  (Source DB)    │                           │ /user/hive/warehouse/  │
│                 │ ←── JDBC INSERT ───────→ │  (Target Storage)      │
│  (Target DB)   │     (Sqoop Export)        │                        │
└─────────────────┘                          └────────────────────────┘
```

**Mapper并行机制**：

- 默认4个Mapper（`-m 4`），每个Mapper执行一个数据分片的SELECT查询
- 分片依据`--split-by`指定的列（默认使用主键），要求该列数值均匀分布
- 如果分片列分布不均，可使用`--boundary-query`手动指定边界

### 4.2 Import（数据导入）

**全量导入MySQL到HDFS**：

```bash
# 全量导入orders表到HDFS
sqoop import \
  --connect "jdbc:mysql://mysql-master:3306/business_db?useSSL=false&serverTimezone=Asia/Shanghai" \
  --username sqoop_reader \
  --password-file hdfs:///user/sqoop/mysql.password \
  --table orders \
  --target-dir /data/raw/orders/full/2026-02-24 \
  --delete-target-dir \
  --fields-terminated-by '\001' \
  --lines-terminated-by '\n' \
  --null-string '\\N' \
  --null-non-string '\\N' \
  --num-mappers 8 \
  --split-by id \
  --compress \
  --compression-codec org.apache.hadoop.io.compress.SnappyCodec \
  --fetch-size 10000
```

**导入MySQL到Hive（自动建表）**：

```bash
# 导入到Hive，自动建表和加载数据
sqoop import \
  --connect "jdbc:mysql://mysql-master:3306/business_db" \
  --username sqoop_reader \
  --password-file hdfs:///user/sqoop/mysql.password \
  --table orders \
  --hive-import \
  --create-hive-table \
  --hive-database ods \
  --hive-table ods_orders_full \
  --hive-overwrite \
  --hive-partition-key dt \
  --hive-partition-value "2026-02-24" \
  --fields-terminated-by '\001' \
  --null-string '\\N' \
  --null-non-string '\\N' \
  --num-mappers 8 \
  --split-by id

# 使用SQL查询导入（更灵活的字段选择和过滤）
sqoop import \
  --connect "jdbc:mysql://mysql-master:3306/business_db" \
  --username sqoop_reader \
  --password-file hdfs:///user/sqoop/mysql.password \
  --query 'SELECT id, user_id, amount, status, create_time, update_time FROM orders WHERE status != "DELETED" AND $CONDITIONS' \
  --target-dir /data/raw/orders/filtered/2026-02-24 \
  --hive-import \
  --hive-database ods \
  --hive-table ods_orders_active \
  --fields-terminated-by '\001' \
  --num-mappers 4 \
  --split-by id
```

**增量导入（Append模式 —— 适用于自增ID）**：

```bash
# 增量导入 —— 基于自增ID追加新数据
sqoop import \
  --connect "jdbc:mysql://mysql-master:3306/business_db" \
  --username sqoop_reader \
  --password-file hdfs:///user/sqoop/mysql.password \
  --table orders \
  --target-dir /data/raw/orders/incremental \
  --incremental append \
  --check-column id \
  --last-value 1000000 \
  --fields-terminated-by '\001' \
  --num-mappers 4 \
  --split-by id

# 增量导入 —— 基于时间戳（lastmodified模式，适用于有update_time列的表）
sqoop import \
  --connect "jdbc:mysql://mysql-master:3306/business_db" \
  --username sqoop_reader \
  --password-file hdfs:///user/sqoop/mysql.password \
  --table orders \
  --target-dir /data/raw/orders/incremental \
  --incremental lastmodified \
  --check-column update_time \
  --last-value "2026-02-23 00:00:00" \
  --merge-key id \
  --fields-terminated-by '\001' \
  --num-mappers 4 \
  --split-by id
```

**Sqoop Job（保存增量状态，定时调度）**：

```bash
# 创建Sqoop Job（自动记录last-value）
sqoop job --create orders_incremental_job -- import \
  --connect "jdbc:mysql://mysql-master:3306/business_db" \
  --username sqoop_reader \
  --password-file hdfs:///user/sqoop/mysql.password \
  --table orders \
  --target-dir /data/raw/orders/incremental \
  --incremental append \
  --check-column id \
  --last-value 0 \
  --fields-terminated-by '\001' \
  --num-mappers 4

# 执行Job（每次运行会自动更新last-value）
sqoop job --exec orders_incremental_job

# 查看Job列表
sqoop job --list

# 查看Job配置
sqoop job --show orders_incremental_job
```

### 4.3 Export（数据导出）

```bash
# 从Hive/HDFS导出到MySQL
sqoop export \
  --connect "jdbc:mysql://mysql-master:3306/report_db" \
  --username sqoop_writer \
  --password-file hdfs:///user/sqoop/mysql.password \
  --table report_daily_orders \
  --export-dir /user/hive/warehouse/dws.db/dws_daily_orders/dt=2026-02-24 \
  --input-fields-terminated-by '\001' \
  --input-null-string '\\N' \
  --input-null-non-string '\\N' \
  --num-mappers 4 \
  --batch \
  --update-mode allowinsert \
  --update-key "dt,product_id" \
  --columns "dt,product_id,order_count,total_amount,user_count"

# --update-mode 参数说明:
# updateonly   : 只更新已存在的行（默认）
# allowinsert  : 存在则更新，不存在则插入（upsert语义）
```

### 4.4 DataX对比

| 对比维度 | Sqoop | DataX |
|---------|-------|-------|
| **开发方** | Apache社区 | 阿里巴巴开源 |
| **运行方式** | MapReduce Job（依赖Hadoop） | 单机多线程（不依赖Hadoop） |
| **配置方式** | 命令行参数 | JSON配置文件 |
| **插件体系** | Connector较少 | Reader/Writer插件丰富 |
| **数据源** | 主要RDBMS ↔ HDFS | 几乎所有异构数据源 |
| **运维成本** | 依赖YARN集群 | 独立运行，部署简单 |
| **吞吐量** | 高（分布式MR） | 高（JVM多线程） |
| **社区活跃度** | 社区不太活跃 | 活跃，阿里持续维护 |
| **增量同步** | 内置支持 | 需手动实现 |
| **脏数据处理** | 较弱 | 内置脏数据管理 |

**DataX完整Job配置示例（MySQL → HDFS）**：

```json
{
  "job": {
    "setting": {
      "speed": {
        "channel": 8,
        "byte": 10485760,
        "record": 100000
      },
      "errorLimit": {
        "record": 0,
        "percentage": 0.02
      }
    },
    "content": [
      {
        "reader": {
          "name": "mysqlreader",
          "parameter": {
            "username": "datax_reader",
            "password": "your_password_here",
            "column": [
              "id",
              "user_id",
              "product_id",
              "amount",
              "status",
              "create_time",
              "update_time"
            ],
            "splitPk": "id",
            "connection": [
              {
                "table": ["orders"],
                "jdbcUrl": [
                  "jdbc:mysql://mysql-master:3306/business_db?useSSL=false&characterEncoding=utf8"
                ]
              }
            ],
            "where": "update_time >= '2026-02-24 00:00:00' AND update_time < '2026-02-25 00:00:00'"
          }
        },
        "writer": {
          "name": "hdfswriter",
          "parameter": {
            "defaultFS": "hdfs://nameservice1",
            "fileType": "orc",
            "path": "/data/raw/orders/dt=2026-02-24",
            "fileName": "orders",
            "column": [
              {"name": "id", "type": "BIGINT"},
              {"name": "user_id", "type": "BIGINT"},
              {"name": "product_id", "type": "BIGINT"},
              {"name": "amount", "type": "DECIMAL(10,2)"},
              {"name": "status", "type": "STRING"},
              {"name": "create_time", "type": "TIMESTAMP"},
              {"name": "update_time", "type": "TIMESTAMP"}
            ],
            "writeMode": "nonConflict",
            "fieldDelimiter": "\u0001",
            "compress": "SNAPPY",
            "hadoopConfig": {
              "dfs.nameservices": "nameservice1",
              "dfs.ha.namenodes.nameservice1": "nn1,nn2",
              "dfs.namenode.rpc-address.nameservice1.nn1": "namenode01:8020",
              "dfs.namenode.rpc-address.nameservice1.nn2": "namenode02:8020",
              "dfs.client.failover.proxy.provider.nameservice1": "org.apache.hadoop.hdfs.server.namenode.ha.ConfiguredFailoverProxyProvider"
            }
          }
        }
      }
    ]
  }
}
```

**DataX执行命令**：

```bash
# 执行DataX任务
python /opt/datax/bin/datax.py /opt/datax/jobs/mysql2hdfs_orders.json

# 指定JVM参数
python /opt/datax/bin/datax.py \
  -j "-Xms4g -Xmx4g -XX:+UseG1GC" \
  /opt/datax/jobs/mysql2hdfs_orders.json

# 传递动态参数
python /opt/datax/bin/datax.py \
  -p "-Ddt=2026-02-24 -Ddb=business_db" \
  /opt/datax/jobs/mysql2hdfs_orders.json
```

---

## 5. Canal实时CDC

### 5.1 Canal原理

Canal是阿里巴巴开源的MySQL Binlog增量订阅和消费组件。它通过伪装成MySQL Slave来接收Master的Binlog推送，从而实现实时数据变更捕获（CDC，Change Data Capture）。

```
Canal CDC 工作原理
┌──────────────────────────────────────────────────────────────────┐
│                     MySQL Master                                 │
│                                                                  │
│  ┌───────────────────────────────────────────────────────┐      │
│  │                  Binlog (ROW格式)                      │      │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │      │
│  │  │ INSERT  │  │ UPDATE  │  │ DELETE  │  │  DDL    │ │      │
│  │  │ event   │  │ event   │  │ event   │  │ event   │ │      │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │      │
│  └────────────────────────┬──────────────────────────────┘      │
│                           │                                      │
│  MySQL Replication协议:    │  Dump Request                       │
│  1. Canal发送DUMP请求      │  (COM_BINLOG_DUMP)                  │
│  2. Master推送Binlog       ↓                                     │
└───────────────────────────┼──────────────────────────────────────┘
                            │ Binlog Events Stream
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│                    Canal Server                                   │
│                                                                   │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐ │
│  │  EventParser │→ │ EventSink        │→ │ EventStore         │ │
│  │              │  │                  │  │ (Ring Buffer)      │ │
│  │ • 接收Binlog │  │ • 数据过滤       │  │ • 存储解析后的数据  │ │
│  │ • 解析协议   │  │ • 数据路由       │  │ • 支持ACK机制      │ │
│  │ • HA切换     │  │ • 格式转换       │  │ • 有序消费         │ │
│  └──────────────┘  └──────────────────┘  └─────────┬──────────┘ │
│                                                     │            │
└─────────────────────────────────────────────────────┼────────────┘
                                                      │
                    ┌─────────────────────────────────┼──────┐
                    │                                 │      │
                    ↓                                 ↓      ↓
          ┌─────────────────┐  ┌──────────────┐  ┌──────────────┐
          │  Canal Client   │  │ Canal Kafka  │  │ Canal RabbitMQ│
          │  (TCP直连)       │  │  Connector   │  │  Connector   │
          └────────┬────────┘  └──────┬───────┘  └──────┬───────┘
                   │                  │                  │
                   ↓                  ↓                  ↓
          ┌──────────────┐    ┌──────────┐      ┌──────────────┐
          │  自定义应用   │    │  Kafka   │      │  RabbitMQ   │
          │  (Java程序)  │    │  Topic   │      │  Queue      │
          └──────────────┘    └──────────┘      └──────────────┘
```

**MySQL Replication协议流程**：

1. Canal Server向MySQL Master注册为Slave（需要配置server-id）
2. Canal发送`COM_BINLOG_DUMP`命令，指定Binlog文件名和位点
3. MySQL Master推送Binlog Event流
4. Canal解析Binlog Event，转换为Canal的Entry结构
5. 下游Client/Connector消费Entry，实现数据同步

**前置要求 —— MySQL需开启ROW格式Binlog**：

```bash
# MySQL配置 (/etc/my.cnf)
[mysqld]
# 开启Binlog
log-bin=mysql-bin
# ROW格式（Canal必须要求）
binlog-format=ROW
# 记录完整的行变更前后镜像
binlog-row-image=FULL
# Server ID（在集群中唯一）
server-id=1
# Binlog保留天数
expire_logs_days=7

# 验证Binlog配置
mysql> SHOW VARIABLES LIKE 'binlog_format';
# +---------------+-------+
# | Variable_name | Value |
# +---------------+-------+
# | binlog_format | ROW   |
# +---------------+-------+

# 创建Canal专用账号
mysql> CREATE USER 'canal'@'%' IDENTIFIED BY 'canal_password';
mysql> GRANT SELECT, REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'canal'@'%';
mysql> FLUSH PRIVILEGES;
```

### 5.2 部署与配置

**下载与安装**：

```bash
# 下载Canal（以1.1.7为例）
wget https://github.com/alibaba/canal/releases/download/canal-1.1.7/canal.deployer-1.1.7.tar.gz

# 解压
mkdir -p /opt/canal
tar -xzf canal.deployer-1.1.7.tar.gz -C /opt/canal

# 目录结构
# /opt/canal/
# ├── bin/          # 启动脚本
# │   ├── startup.sh
# │   └── stop.sh
# ├── conf/         # 配置文件
# │   ├── canal.properties          # Canal Server全局配置
# │   └── example/
# │       └── instance.properties   # 实例配置（每个MySQL对应一个实例）
# ├── lib/          # 依赖JAR包
# └── logs/         # 日志目录
```

**Canal Server全局配置**：

```properties
# /opt/canal/conf/canal.properties

#################################################
####         Canal Server 全局配置             ####
#################################################

# Canal Server绑定的IP和端口
canal.ip =
canal.port = 11111
canal.metrics.pull.port = 11112

# Canal Admin管理端口
canal.admin.manager = 127.0.0.1:8089
canal.admin.port = 11110

# 实例列表（逗号分隔，可配置多个）
canal.destinations = example,order_db,user_db

# 实例配置文件加载方式（spring/manager）
canal.instance.global.mode = spring

# 全局实例的Spring配置
canal.instance.global.spring.xml = classpath:spring/default-instance.xml

# ZooKeeper配置（HA模式必须配置）
canal.zkServers = zk01:2181,zk02:2181,zk03:2181

# 持久化模式（memory/file/zookeeper）
canal.instance.global.lazy = false

#################################################
####         Server Mode（TCP/Kafka/RocketMQ） ####
#################################################

# 服务模式：tcp（Client直连）/ kafka / rocketMQ / rabbitMQ
canal.serverMode = tcp

# Kafka模式配置
# canal.serverMode = kafka
# kafka.bootstrap.servers = kafka01:9092,kafka02:9092,kafka03:9092
# kafka.acks = all
# kafka.compression.type = snappy
# kafka.batch.size = 16384
# kafka.max.request.size = 1048576
# kafka.buffer.memory = 33554432
# kafka.retries = 3
# kafka.max.in.flight.requests.per.connection = 1
```

**实例配置**：

```properties
# /opt/canal/conf/example/instance.properties

#################################################
####         MySQL连接配置                      ####
#################################################

# MySQL Master地址
canal.instance.master.address = mysql-master:3306
# Binlog文件名（为空表示从最新位点开始）
canal.instance.master.journal.name =
# Binlog偏移量
canal.instance.master.position =
# Binlog时间戳（从指定时间开始解析，优先级低于journal.name+position）
canal.instance.master.timestamp =
# GTID模式
canal.instance.master.gtid =

# MySQL Slave配置（HA切换用）
# canal.instance.standby.address = mysql-slave:3306

# 数据库账号密码
canal.instance.dbUsername = canal
canal.instance.dbPassword = canal_password
canal.instance.connectionCharset = UTF-8

# Canal伪装的Slave server-id（需保证在MySQL集群中唯一）
canal.instance.mysql.slaveId = 1234

#################################################
####         数据过滤配置                       ####
#################################################

# 白名单过滤（支持正则）
# 格式: schema.table (使用.*匹配所有)
# 示例: business_db.orders,business_db.users,business_db.products
canal.instance.filter.regex = business_db\\..*

# 黑名单过滤（排除不需要的表）
canal.instance.filter.black.regex = business_db\\.temp_.*,business_db\\.log_.*

# DDL过滤
canal.instance.filter.druid.ddl = true
canal.instance.filter.query.dcl = false
canal.instance.filter.query.dml = false
canal.instance.filter.query.ddl = false

#################################################
####         MQ相关配置（Kafka模式时生效）       ####
#################################################

# Kafka Topic名称
canal.mq.topic = canal_business_db
# 分区数
canal.mq.partitionsNum = 6
# 按表名hash分区（保证同一表的变更有序）
canal.mq.partitionHash = business_db.orders:id,business_db.users:id
# 动态Topic（按表名路由到不同Topic）
# canal.mq.dynamicTopic = business_db\\.orders:topic_orders,business_db\\.users:topic_users
```

**启动Canal**：

```bash
# 启动Canal Server
/opt/canal/bin/startup.sh

# 查看日志
tail -f /opt/canal/logs/canal/canal.log
tail -f /opt/canal/logs/example/example.log

# 停止Canal Server
/opt/canal/bin/stop.sh
```

### 5.3 Canal Client编程

使用Java编写Canal Client程序，实时消费MySQL的Binlog变更。

```java
import com.alibaba.otter.canal.client.CanalConnector;
import com.alibaba.otter.canal.client.CanalConnectors;
import com.alibaba.otter.canal.protocol.CanalEntry;
import com.alibaba.otter.canal.protocol.CanalEntry.*;
import com.alibaba.otter.canal.protocol.Message;
import com.google.protobuf.InvalidProtocolBufferException;

import java.net.InetSocketAddress;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Canal Client 示例
 * 实时消费MySQL Binlog，处理INSERT/UPDATE/DELETE变更
 */
public class CanalClientExample {

    private static final String CANAL_HOST = "canal-server";
    private static final int CANAL_PORT = 11111;
    private static final String DESTINATION = "example";
    private static final int BATCH_SIZE = 1000;

    public static void main(String[] args) {
        // 创建Canal连接器
        // 单机模式
        CanalConnector connector = CanalConnectors.newSingleConnector(
                new InetSocketAddress(CANAL_HOST, CANAL_PORT),
                DESTINATION,
                "canal",      // Canal用户名（可选）
                "canal"       // Canal密码（可选）
        );

        // HA模式（基于ZooKeeper自动发现）
        // CanalConnector connector = CanalConnectors.newClusterConnector(
        //         "zk01:2181,zk02:2181,zk03:2181",
        //         DESTINATION,
        //         "canal",
        //         "canal"
        // );

        try {
            // 建立连接
            connector.connect();
            // 订阅数据库表（正则匹配）
            connector.subscribe("business_db\\.orders,business_db\\.users");
            // 回滚到上次ACK的位点（确保消息不丢失）
            connector.rollback();

            System.out.println("Canal Client 已启动，等待消费Binlog...");

            while (true) {
                // 批量获取数据（阻塞等待，超时1秒）
                Message message = connector.getWithoutAck(BATCH_SIZE);
                long batchId = message.getId();
                int size = message.getEntries().size();

                if (batchId == -1 || size == 0) {
                    // 无数据，等待后重试
                    TimeUnit.MILLISECONDS.sleep(200);
                    continue;
                }

                try {
                    // 处理Binlog Entry
                    processEntries(message.getEntries());
                    // 确认消费成功
                    connector.ack(batchId);
                } catch (Exception e) {
                    System.err.println("处理失败，回滚: " + e.getMessage());
                    // 处理失败，回滚到上次ACK位点
                    connector.rollback(batchId);
                }
            }
        } catch (Exception e) {
            System.err.println("Canal Client异常: " + e.getMessage());
            e.printStackTrace();
        } finally {
            connector.disconnect();
        }
    }

    /**
     * 处理Binlog Entry列表
     */
    private static void processEntries(List<Entry> entries)
            throws InvalidProtocolBufferException {

        for (Entry entry : entries) {
            // 跳过事务开始/结束标记
            if (entry.getEntryType() == EntryType.TRANSACTIONBEGIN
                    || entry.getEntryType() == EntryType.TRANSACTIONEND) {
                continue;
            }

            // 解析RowChange
            RowChange rowChange = RowChange.parseFrom(entry.getStoreValue());
            EventType eventType = rowChange.getEventType();

            String schemaName = entry.getHeader().getSchemaName();
            String tableName = entry.getHeader().getTableName();
            long executeTime = entry.getHeader().getExecuteTime();
            String logFileName = entry.getHeader().getLogfileName();
            long logFileOffset = entry.getHeader().getLogfileOffset();

            System.out.printf("[%s] %s.%s | binlog: %s:%d | eventType: %s%n",
                    new java.util.Date(executeTime),
                    schemaName, tableName,
                    logFileName, logFileOffset,
                    eventType);

            // 处理DDL语句
            if (rowChange.getIsDdl()) {
                System.out.println("DDL: " + rowChange.getSql());
                continue;
            }

            // 处理DML（INSERT/UPDATE/DELETE）
            for (RowData rowData : rowChange.getRowDatasList()) {
                switch (eventType) {
                    case INSERT:
                        handleInsert(schemaName, tableName, rowData);
                        break;
                    case UPDATE:
                        handleUpdate(schemaName, tableName, rowData);
                        break;
                    case DELETE:
                        handleDelete(schemaName, tableName, rowData);
                        break;
                    default:
                        break;
                }
            }
        }
    }

    private static void handleInsert(String schema, String table, RowData rowData) {
        System.out.println("  >>> INSERT <<<");
        List<Column> columns = rowData.getAfterColumnsList();
        StringBuilder sb = new StringBuilder();
        for (Column column : columns) {
            sb.append(String.format("    %s = %s (updated=%b)%n",
                    column.getName(), column.getValue(), column.getUpdated()));
        }
        System.out.print(sb);

        // 实际业务处理示例：写入Kafka、更新缓存、同步到ES等
        // kafkaProducer.send(buildMessage(schema, table, "INSERT", columns));
    }

    private static void handleUpdate(String schema, String table, RowData rowData) {
        System.out.println("  >>> UPDATE <<<");
        List<Column> beforeColumns = rowData.getBeforeColumnsList();
        List<Column> afterColumns = rowData.getAfterColumnsList();

        // 对比变更前后的值
        for (int i = 0; i < afterColumns.size(); i++) {
            Column before = beforeColumns.get(i);
            Column after = afterColumns.get(i);
            if (after.getUpdated()) {
                System.out.printf("    %s: %s -> %s%n",
                        after.getName(), before.getValue(), after.getValue());
            }
        }
    }

    private static void handleDelete(String schema, String table, RowData rowData) {
        System.out.println("  >>> DELETE <<<");
        List<Column> columns = rowData.getBeforeColumnsList();
        for (Column column : columns) {
            System.out.printf("    %s = %s%n", column.getName(), column.getValue());
        }
    }
}
```

**Maven依赖**：

```xml
<!-- pom.xml -->
<dependencies>
    <dependency>
        <groupId>com.alibaba.otter</groupId>
        <artifactId>canal.client</artifactId>
        <version>1.1.7</version>
    </dependency>
    <dependency>
        <groupId>com.alibaba.otter</groupId>
        <artifactId>canal.protocol</artifactId>
        <version>1.1.7</version>
    </dependency>
</dependencies>
```

### 5.4 Canal + Kafka集成

**Canal Kafka模式配置**：

将`canal.properties`中的`canal.serverMode`设为`kafka`，Canal Server会自动将Binlog变更推送到Kafka Topic。

```properties
# canal.properties 关键配置
canal.serverMode = kafka

# Kafka连接配置
kafka.bootstrap.servers = kafka01:9092,kafka02:9092,kafka03:9092
kafka.acks = all
kafka.compression.type = snappy
kafka.batch.size = 16384
kafka.linger.ms = 100
kafka.max.request.size = 1048576
kafka.buffer.memory = 33554432
kafka.retries = 3
# 保证分区内有序
kafka.max.in.flight.requests.per.connection = 1
```

```properties
# instance.properties 关键配置
# 目标Kafka Topic
canal.mq.topic = canal_business_db
# 分区策略
canal.mq.partitionsNum = 6
# 按主键hash分区（同一行的变更保证顺序）
canal.mq.partitionHash = business_db.orders:id,business_db.users:id
# 或使用动态Topic（按表名分发到不同Topic）
# canal.mq.dynamicTopic = business_db\\.orders:topic_orders,business_db\\.users:topic_users
# 是否扁平化消息格式（推荐true，JSON格式更易解析）
canal.mq.flatMessage = true
```

**Canal推送到Kafka的消息格式（flatMessage=true）**：

```json
{
  "data": [
    {
      "id": "10001",
      "user_id": "1001",
      "product_id": "2001",
      "amount": "299.00",
      "status": "PAID",
      "create_time": "2026-02-24 10:30:15",
      "update_time": "2026-02-24 10:35:20"
    }
  ],
  "old": [
    {
      "status": "CREATED",
      "update_time": "2026-02-24 10:30:15"
    }
  ],
  "database": "business_db",
  "table": "orders",
  "type": "UPDATE",
  "es": 1740370520000,
  "ts": 1740370520123,
  "id": 15,
  "isDdl": false,
  "mysqlType": {
    "id": "bigint(20)",
    "user_id": "bigint(20)",
    "amount": "decimal(10,2)",
    "status": "varchar(32)",
    "create_time": "datetime",
    "update_time": "datetime"
  },
  "sqlType": {
    "id": -5,
    "user_id": -5,
    "amount": 3,
    "status": 12,
    "create_time": 93,
    "update_time": 93
  },
  "pkNames": ["id"]
}
```

**下游Flink SQL消费Canal Kafka数据**：

```sql
-- Flink SQL: 使用canal-json格式消费CDC数据
CREATE TABLE orders_cdc (
    id BIGINT,
    user_id BIGINT,
    product_id BIGINT,
    amount DECIMAL(10, 2),
    status STRING,
    create_time TIMESTAMP(3),
    update_time TIMESTAMP(3),
    PRIMARY KEY (id) NOT ENFORCED
) WITH (
    'connector' = 'kafka',
    'topic' = 'canal_business_db',
    'properties.bootstrap.servers' = 'kafka01:9092,kafka02:9092,kafka03:9092',
    'properties.group.id' = 'flink-canal-consumer',
    'scan.startup.mode' = 'earliest-offset',
    'format' = 'canal-json',
    'canal-json.ignore-parse-errors' = 'true'
);

-- 实时统计每小时订单金额（物化视图效果）
CREATE TABLE order_hourly_stats (
    window_start TIMESTAMP(3),
    window_end TIMESTAMP(3),
    order_count BIGINT,
    total_amount DECIMAL(18, 2),
    paid_count BIGINT,
    PRIMARY KEY (window_start) NOT ENFORCED
) WITH (
    'connector' = 'jdbc',
    'url' = 'jdbc:mysql://mysql-report:3306/report_db',
    'table-name' = 'order_hourly_stats',
    'driver' = 'com.mysql.cj.jdbc.Driver',
    'username' = 'flink_writer',
    'password' = 'your_password'
);

INSERT INTO order_hourly_stats
SELECT
    TUMBLE_START(update_time, INTERVAL '1' HOUR) AS window_start,
    TUMBLE_END(update_time, INTERVAL '1' HOUR) AS window_end,
    COUNT(*) AS order_count,
    SUM(amount) AS total_amount,
    COUNT(CASE WHEN status = 'PAID' THEN 1 END) AS paid_count
FROM orders_cdc
WHERE status IN ('PAID', 'COMPLETED')
GROUP BY TUMBLE(update_time, INTERVAL '1' HOUR);
```

---

## 6. 实战案例：企业级数据采集平台

### 6.1 需求分析

一个典型的中大型互联网企业，需要构建统一的数据采集平台来覆盖多种数据源。以下是一个电商企业的真实数据采集需求：

```
企业级数据采集平台架构
┌─────────────────────────────────────────────────────────────────────────┐
│                          数据源层                                       │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ MySQL集群    │  │ App日志     │  │ Nginx日志    │  │ 第三方API   │   │
│  │ (订单/用户/  │  │ (业务日志/  │  │ (访问日志/   │  │ (支付回调/  │   │
│  │  商品/库存)  │  │  埋点日志)  │  │  错误日志)   │  │  物流状态)  │   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │
│         │                │                │                │            │
└─────────┼────────────────┼────────────────┼────────────────┼────────────┘
          │                │                │                │
          ↓                ↓                ↓                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                        采集层                                           │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │   Canal     │  │   Flume     │  │  Filebeat   │  │ Custom ETL  │   │
│  │  (实时CDC)  │  │  (日志采集)  │  │  (轻量采集)  │  │ (API采集)   │   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │
│         │                │                │                │            │
└─────────┼────────────────┼────────────────┼────────────────┼────────────┘
          │                │                │                │
          └────────────────┼────────────────┼────────────────┘
                           ↓                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      消息总线层                                          │
│                                                                         │
│                    ┌────────────────────────────────┐                    │
│                    │          Apache Kafka           │                    │
│                    │                                │                    │
│                    │  topic: canal_orders           │                    │
│                    │  topic: canal_users            │                    │
│                    │  topic: app_logs               │                    │
│                    │  topic: nginx_access_logs      │                    │
│                    │  topic: api_callback           │                    │
│                    └───────────────┬────────────────┘                    │
│                                   │                                      │
└───────────────────────────────────┼──────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
┌──────────────────────┐ ┌──────────────┐ ┌──────────────────────┐
│  HDFS / Hive         │ │Elasticsearch │ │  Flink实时计算        │
│  (离线数据仓库)       │ │ (日志搜索)    │ │  (实时指标/实时报表)   │
│                      │ │              │ │                      │
│  ODS → DWD → DWS    │ │ Kibana可视化  │ │  结果写入MySQL/Redis  │
│       → ADS         │ │              │ │                      │
└──────────────────────┘ └──────────────┘ └──────────────────────┘
```

**各数据源的采集需求**：

| 数据源 | 采集方式 | 延迟要求 | 数据量(日) | 目标 |
|--------|---------|---------|-----------|------|
| MySQL订单表 | Canal CDC | 秒级 | ~500万条 | Kafka → Flink实时 + HDFS离线 |
| MySQL用户表 | Canal CDC | 秒级 | ~50万条 | Kafka → ES(搜索) + HDFS(离线) |
| MySQL商品表 | DataX全量 | T+1 | ~100万条 | HDFS → Hive维度表 |
| App业务日志 | Flume | 分钟级 | ~10亿条/50TB | Kafka → HDFS |
| App埋点日志 | Flume | 分钟级 | ~20亿条/100TB | Kafka → HDFS + Flink实时 |
| Nginx访问日志 | Filebeat | 分钟级 | ~5亿条/20TB | Kafka → ES + HDFS |
| 支付回调API | 自定义服务 | 秒级 | ~200万条 | Kafka → Flink |

### 6.2 实现方案

**1. Canal CDC采集MySQL变更**：

```properties
# /opt/canal/conf/order_db/instance.properties
canal.instance.master.address = mysql-order-master:3306
canal.instance.dbUsername = canal
canal.instance.dbPassword = canal_password_encrypted
canal.instance.mysql.slaveId = 2001

# 采集订单相关的所有表
canal.instance.filter.regex = order_db\\..*
canal.instance.filter.black.regex = order_db\\.undo_log,order_db\\.temp_.*

canal.mq.topic = canal_order_db
canal.mq.partitionsNum = 12
canal.mq.partitionHash = order_db.orders:id,order_db.order_items:order_id
canal.mq.flatMessage = true
```

**2. Flume采集App日志**：

```properties
# /opt/flume/conf/app-log-agent.conf
agent.sources = taildir_source
agent.channels = kafka_channel
agent.sinks = hdfs_sink

# 使用Kafka Channel（既做Channel又做输出到Kafka）
agent.channels.kafka_channel.type = org.apache.flume.channel.kafka.KafkaChannel
agent.channels.kafka_channel.kafka.bootstrap.servers = kafka01:9092,kafka02:9092,kafka03:9092
agent.channels.kafka_channel.kafka.topic = app_logs
agent.channels.kafka_channel.kafka.consumer.group.id = flume-app-log
agent.channels.kafka_channel.parseAsFlumeEvent = false
agent.channels.kafka_channel.kafka.producer.acks = 1
agent.channels.kafka_channel.kafka.producer.compression.type = lz4

agent.sources.taildir_source.type = TAILDIR
agent.sources.taildir_source.filegroups = f1 f2
agent.sources.taildir_source.filegroups.f1 = /data/logs/order-service/.*\\.log
agent.sources.taildir_source.headers.f1.service = order-service
agent.sources.taildir_source.filegroups.f2 = /data/logs/user-service/.*\\.log
agent.sources.taildir_source.headers.f2.service = user-service
agent.sources.taildir_source.positionFile = /data/flume/position/app_position.json
agent.sources.taildir_source.batchSize = 2000
agent.sources.taildir_source.channels = kafka_channel

# HDFS Sink（从Kafka Channel消费写入HDFS）
agent.sinks.hdfs_sink.type = hdfs
agent.sinks.hdfs_sink.channel = kafka_channel
agent.sinks.hdfs_sink.hdfs.path = hdfs://nameservice1/data/raw/app_logs/%Y-%m-%d/%H
agent.sinks.hdfs_sink.hdfs.filePrefix = app
agent.sinks.hdfs_sink.hdfs.fileSuffix = .lzo
agent.sinks.hdfs_sink.hdfs.rollInterval = 3600
agent.sinks.hdfs_sink.hdfs.rollSize = 268435456
agent.sinks.hdfs_sink.hdfs.rollCount = 0
agent.sinks.hdfs_sink.hdfs.fileType = CompressedStream
agent.sinks.hdfs_sink.hdfs.codeC = lzop
agent.sinks.hdfs_sink.hdfs.batchSize = 2000
```

**3. Filebeat采集Nginx日志**：

```yaml
# /etc/filebeat/filebeat-nginx.yml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /var/log/nginx/access.log
    fields:
      log_type: nginx_access
    fields_under_root: true
    close_inactive: 5m

output.kafka:
  hosts: ["kafka01:9092", "kafka02:9092", "kafka03:9092"]
  topic: "nginx_access_logs"
  partition.round_robin:
    reachable_only: true
  required_acks: 1
  compression: gzip
  worker: 2

logging.level: warning
logging.to_files: true
logging.files:
  path: /var/log/filebeat
  name: filebeat-nginx.log
  keepfiles: 3
```

**4. 自定义API数据采集（Python示例）**：

```python
"""
第三方API数据采集服务
定时拉取支付状态回调、物流状态更新等外部数据，推送到Kafka
"""
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from confluent_kafka import Producer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kafka Producer配置
kafka_config = {
    'bootstrap.servers': 'kafka01:9092,kafka02:9092,kafka03:9092',
    'client.id': 'api-collector',
    'acks': 'all',
    'retries': 3,
    'compression.type': 'snappy',
    'batch.size': 16384,
    'linger.ms': 100,
}

producer = Producer(kafka_config)

def delivery_callback(err, msg):
    """Kafka消息发送回调"""
    if err:
        logger.error(f"消息发送失败: {err}")
    else:
        logger.debug(f"消息已发送到 {msg.topic()} [{msg.partition()}] offset={msg.offset()}")

def fetch_payment_callbacks(last_check_time: str) -> list:
    """拉取支付回调数据"""
    url = "https://api.payment-provider.com/v1/callbacks"
    headers = {
        "Authorization": "Bearer YOUR_API_TOKEN",
        "Content-Type": "application/json"
    }
    params = {
        "start_time": last_check_time,
        "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "page_size": 500
    }

    all_records = []
    page = 1

    while True:
        params["page"] = page
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        records = data.get("records", [])
        if not records:
            break

        all_records.extend(records)
        if len(records) < params["page_size"]:
            break
        page += 1

    return all_records

def send_to_kafka(topic: str, records: list):
    """批量发送到Kafka"""
    for record in records:
        key = str(record.get("order_id", "")).encode("utf-8")
        value = json.dumps(record, ensure_ascii=False).encode("utf-8")

        producer.produce(
            topic=topic,
            key=key,
            value=value,
            callback=delivery_callback
        )

    producer.flush(timeout=30)
    logger.info(f"已发送 {len(records)} 条记录到 {topic}")

def main():
    """主循环：每30秒拉取一次API数据"""
    topic = "api_payment_callbacks"
    last_check_time = (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")

    while True:
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            records = fetch_payment_callbacks(last_check_time)

            if records:
                send_to_kafka(topic, records)
                logger.info(f"[{current_time}] 采集到 {len(records)} 条支付回调")
            else:
                logger.info(f"[{current_time}] 无新增数据")

            last_check_time = current_time
        except requests.RequestException as e:
            logger.error(f"API请求异常: {e}")
        except Exception as e:
            logger.error(f"采集异常: {e}")

        time.sleep(30)

if __name__ == "__main__":
    main()
```

### 6.3 数据质量保障

**Exactly-Once语义保障策略**：

| 组件 | 保障机制 | 实现方式 |
|------|---------|---------|
| Canal | At-Least-Once + 下游幂等 | Binlog位点持久化到ZK；下游按主键Upsert |
| Flume | File Channel事务 | Source写入和Sink读取分属不同事务，Channel持久化保证 |
| Filebeat | Registry记录偏移量 | 每条日志消费后更新Registry文件 |
| Kafka | 事务Producer + 消费者Offset管理 | `enable.idempotence=true`，手动Offset提交 |
| Flink | Checkpoint + Two-Phase Commit | Kafka Source精确一次 + JDBC/HBase Sink幂等写入 |

**数据校验规则**：

```python
# 数据质量校验框架示例
class DataQualityChecker:
    """数据质量校验"""

    def check_completeness(self, source_count: int, target_count: int,
                           threshold: float = 0.001) -> bool:
        """完整性校验：对比源端和目标端数据条数"""
        if source_count == 0:
            return True
        diff_rate = abs(source_count - target_count) / source_count
        passed = diff_rate <= threshold
        if not passed:
            print(f"  completeness FAILED: source={source_count}, "
                  f"target={target_count}, diff_rate={diff_rate:.4f}")
        return passed

    def check_timeliness(self, max_delay_seconds: int,
                         latest_event_time: str) -> bool:
        """时效性校验：最新数据的延迟不超过阈值"""
        from datetime import datetime
        latest = datetime.strptime(latest_event_time, "%Y-%m-%d %H:%M:%S")
        delay = (datetime.now() - latest).total_seconds()
        passed = delay <= max_delay_seconds
        if not passed:
            print(f"  timeliness FAILED: delay={delay}s, threshold={max_delay_seconds}s")
        return passed

    def check_uniqueness(self, table: str, key_column: str) -> bool:
        """唯一性校验：主键无重复"""
        # 在Hive中执行: SELECT COUNT(*) - COUNT(DISTINCT key_column) FROM table
        # 差值应为0
        pass

    def check_accuracy(self, source_sum: float, target_sum: float,
                       threshold: float = 0.01) -> bool:
        """准确性校验：金额等关键指标的汇总值比对"""
        if source_sum == 0:
            return target_sum == 0
        diff_rate = abs(source_sum - target_sum) / abs(source_sum)
        passed = diff_rate <= threshold
        if not passed:
            print(f"  accuracy FAILED: source_sum={source_sum}, "
                  f"target_sum={target_sum}, diff_rate={diff_rate:.6f}")
        return passed
```

**监控与告警指标**：

| 指标 | 含义 | 告警阈值 | 告警级别 |
|------|------|---------|---------|
| `canal.delay` | Canal解析延迟（秒） | > 60s | P1 |
| `canal.binlog.lag` | Binlog消费落后字节数 | > 1GB | P1 |
| `flume.channel.fill_percentage` | Channel填充率 | > 80% | P2 |
| `flume.sink.connection_failed_count` | Sink连接失败次数 | > 0 | P2 |
| `kafka.consumer.lag` | 消费者Lag（消息积压） | > 100000 | P1 |
| `filebeat.harvester.open_files` | 打开的文件数 | > 1000 | P3 |
| `data.completeness.diff_rate` | 数据完整性差异率 | > 0.1% | P1 |
| `data.timeliness.max_delay` | 数据最大延迟 | > 300s | P2 |

---

## 7. 最佳实践总结

### 7.1 选型决策树

根据数据源类型、延迟要求和技术栈选择合适的采集工具：

```
数据采集工具选型决策树

                        ┌─────────────────────┐
                        │   需要采集什么数据？   │
                        └──────────┬──────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ↓                    ↓                    ↓
     ┌────────────────┐  ┌────────────────┐   ┌────────────────┐
     │  数据库变更数据  │  │   日志/文件数据  │   │  API/消息数据   │
     └───────┬────────┘  └───────┬────────┘   └───────┬────────┘
             │                   │                     │
        ┌────┴────┐         ┌───┴────┐            ┌───┴────┐
        ↓         ↓         ↓        ↓            ↓        ↓
   ┌────────┐ ┌────────┐ ┌──────┐ ┌──────┐  ┌────────┐ ┌────────┐
   │实时(CDC)│ │离线(批量)│ │实时  │ │离线  │  │推送模式│ │拉取模式│
   └───┬────┘ └───┬────┘ └──┬───┘ └──┬───┘  └───┬────┘ └───┬────┘
       │          │          │        │          │          │
       ↓          ↓          ↓        ↓          ↓          ↓
  ┌─────────┐┌────────┐┌────────┐┌───────┐┌─────────┐┌─────────┐
  │ MySQL?  ││ Hadoop ││ Hadoop ││ ELK   ││ Kafka   ││ 自定义   │
  │→ Canal  ││ 生态?  ││ 生态?  ││ 生态? ││Consumer ││ 定时拉取 │
  │         ││→ Sqoop ││→ Flume ││→ File-││ 接收推送 ││ Python/ │
  │ 多种DB? ││        ││        ││  beat ││         ││ Java    │
  │→Debezium││ 通用?  ││ 通用?  ││      ││         ││         │
  │         ││→ DataX ││→ File- ││      ││         ││         │
  │         ││        ││  beat  ││      ││         ││         │
  └─────────┘└────────┘└────────┘└───────┘└─────────┘└─────────┘
```

**快速选型指南**：

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| MySQL实时同步到数据湖 | Canal + Kafka + Flink | 毫秒级延迟，支持DDL同步 |
| PostgreSQL/MongoDB CDC | Debezium + Kafka Connect | 原生Kafka生态，多数据库支持 |
| MySQL全量同步到Hive | Sqoop / DataX | 成熟稳定，支持并行导入 |
| 多源异构数据批量同步 | DataX | 插件丰富，配置简单 |
| App日志采集到Hadoop | Flume (Taildir + Kafka Channel) | 与Hadoop深度集成 |
| 日志采集到ELK | Filebeat + Logstash | 轻量高效，Filter能力强 |
| 大规模容器日志采集 | Filebeat (DaemonSet) | 资源占用低，K8s友好 |

### 7.2 性能调优

**Flume性能调优参数**：

```properties
# ---- Source调优 ----
# Taildir Source: 增大批量读取量
a1.sources.s1.batchSize = 2000
# 减少文件检测间隔（毫秒）
a1.sources.s1.pollDelay = 500

# ---- Channel调优 ----
# Memory Channel: 增大容量
a1.channels.c1.capacity = 200000
a1.channels.c1.transactionCapacity = 10000
# File Channel: 使用多磁盘
a1.channels.c1.dataDirs = /ssd1/flume/data,/ssd2/flume/data
a1.channels.c1.checkpointDir = /ssd1/flume/checkpoint
# Kafka Channel: 调优Producer
a1.channels.c1.kafka.producer.batch.size = 32768
a1.channels.c1.kafka.producer.linger.ms = 50
a1.channels.c1.kafka.producer.buffer.memory = 67108864

# ---- Sink调优 ----
# HDFS Sink: 增大批量写入
a1.sinks.k1.hdfs.batchSize = 5000
# 使用多线程Sink（Sink Runner线程池）
a1.sinks.k1.hdfs.threadsPoolSize = 20
a1.sinks.k1.hdfs.rollTimerPoolSize = 5
# 减少HDFS RPC调用（增大roll阈值）
a1.sinks.k1.hdfs.rollSize = 268435456
a1.sinks.k1.hdfs.rollInterval = 3600
```

**Logstash性能调优参数**：

```yaml
# /etc/logstash/logstash.yml

# Pipeline线程数（建议等于CPU核心数）
pipeline.workers: 8
# 每个Worker的批量处理大小
pipeline.batch.size: 500
# 批量等待延迟（毫秒）
pipeline.batch.delay: 50
# Output Worker数
pipeline.output.workers: 4

# JVM配置 (/etc/logstash/jvm.options)
# -Xms4g
# -Xmx4g
# -XX:+UseG1GC
# -XX:G1HeapRegionSize=16m
```

**Sqoop性能调优参数**：

```bash
# 增加Mapper并行度（根据源库承受能力调整）
--num-mappers 16

# 选择分布均匀的分片列
--split-by id

# 增大JDBC Fetch Size（减少网络往返）
--fetch-size 50000

# 使用Direct模式（部分数据库支持，绕过JDBC）
--direct

# 指定压缩编解码器
--compress --compression-codec org.apache.hadoop.io.compress.SnappyCodec

# Mapper内存调优
-D mapreduce.map.memory.mb=4096
-D mapreduce.map.java.opts=-Xmx3584m
```

**Canal性能调优参数**：

```properties
# canal.properties

# Parser线程数（并行解析Binlog）
canal.instance.parser.parallelThreadSize = 8

# 内部Ring Buffer大小（Event缓冲区）
canal.instance.memory.buffer.size = 32768
canal.instance.memory.buffer.memunit = 1024

# 批量获取大小
canal.instance.memory.batch.mode = MEMSIZE
canal.instance.memory.rawEntry = false

# 网络相关
canal.instance.network.receiveBufferSize = 16384
canal.instance.network.sendBufferSize = 16384

# Kafka Producer调优
kafka.batch.size = 32768
kafka.linger.ms = 50
kafka.buffer.memory = 67108864
kafka.max.request.size = 5242880
```

### 7.3 监控与告警

**各组件核心监控指标**：

| 组件 | 指标名 | 含义 | 采集方式 |
|------|--------|------|---------|
| **Canal** | canal_instance_delay | 解析延迟(ms) | Prometheus Metrics端口 |
| **Canal** | canal_instance_store_produce_seq | 产生序列号 | Canal Admin API |
| **Canal** | canal_instance_store_consume_seq | 消费序列号 | Canal Admin API |
| **Flume** | CHANNEL.ChannelFillPercentage | Channel填充百分比 | JMX / Flume Metrics |
| **Flume** | CHANNEL.EventPutAttemptCount | Source尝试写入次数 | JMX |
| **Flume** | CHANNEL.EventTakeSuccessCount | Sink成功读取次数 | JMX |
| **Flume** | SINK.ConnectionFailedCount | Sink连接失败次数 | JMX |
| **Filebeat** | filebeat_harvester_running | 活跃采集器数量 | Filebeat Metrics |
| **Filebeat** | filebeat_input_log_files_truncated | 文件截断次数 | Filebeat Metrics |
| **Logstash** | jvm.mem.heap_used_percent | JVM堆使用率 | Logstash API |
| **Logstash** | pipeline.events.filtered | 过滤后事件数 | Logstash API |
| **Sqoop** | MapReduce Job状态 | 任务成功/失败 | YARN ResourceManager |
| **DataX** | 读取速度/写入速度 | 吞吐量 | DataX Job统计日志 |

**告警策略配置示例（Prometheus AlertManager）**：

```yaml
# alertmanager-rules.yml
groups:
  - name: data_ingestion_alerts
    rules:
      # Canal延迟告警
      - alert: CanalHighDelay
        expr: canal_instance_delay > 60000
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Canal解析延迟超过60秒"
          description: "实例 {{ $labels.instance }} 延迟 {{ $value }}ms"

      # Flume Channel积压告警
      - alert: FlumeChannelBacklog
        expr: flume_channel_fill_percentage > 80
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "Flume Channel填充率超过80%"
          description: "Agent {{ $labels.agent }} Channel {{ $labels.channel }} 填充率 {{ $value }}%"

      # Kafka消费者Lag告警
      - alert: KafkaConsumerHighLag
        expr: kafka_consumer_group_lag > 500000
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Kafka消费者积压超过50万"
          description: "消费组 {{ $labels.consumer_group }} Topic {{ $labels.topic }} 积压 {{ $value }}"

      # Filebeat采集器停止告警
      - alert: FilebeatHarvesterStopped
        expr: filebeat_harvester_running == 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Filebeat无活跃采集器"
          description: "节点 {{ $labels.host }} 上Filebeat已无活跃采集进程"

      # 数据完整性告警
      - alert: DataCompletenessFailure
        expr: data_quality_completeness_diff_rate > 0.001
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "数据完整性校验未通过"
          description: "表 {{ $labels.table }} 源端与目标端差异率 {{ $value }}"
```

**运维Checklist**：

| 类别 | 检查项 | 频率 | 说明 |
|------|--------|------|------|
| **日常巡检** | Canal延迟 | 每5分钟 | 超过30秒立即告警 |
| **日常巡检** | Kafka消费者Lag | 每5分钟 | 超过10万条关注，超过50万告警 |
| **日常巡检** | Flume Channel使用率 | 每1分钟 | 超过80%需扩容或排查下游 |
| **日常巡检** | HDFS小文件数量 | 每天 | 单分区超过1000个文件需合并 |
| **定期维护** | Binlog磁盘空间 | 每天 | 确保MySQL Binlog不会撑满磁盘 |
| **定期维护** | Sqoop/DataX Job成功率 | 每天 | T+1离线任务必须100%成功 |
| **定期维护** | 数据完整性对账 | 每天 | 源端和目标端条数/金额比对 |
| **应急预案** | Canal主备切换演练 | 每月 | 验证ZK HA切换正常 |
| **应急预案** | Kafka集群扩容演练 | 每季度 | 验证Topic Partition扩展和数据迁移 |

**数据采集常见问题排查**：

| 问题现象 | 可能原因 | 解决方案 |
|---------|---------|---------|
| Canal延迟持续增大 | Binlog事件量突增/下游消费慢 | 增加Parser线程；检查Kafka写入性能 |
| Flume Channel满 | Sink写入HDFS失败/速度不够 | 检查HDFS连接；增大Sink线程池；增大Channel容量 |
| Filebeat日志采集遗漏 | 日志文件轮转被清理 | 增大`clean_inactive`；加快采集速度 |
| Sqoop导入数据重复 | 分片列不均匀/任务重试 | 选择均匀分片列；目标表增加去重逻辑 |
| DataX脏数据比例高 | 源端数据格式不规范 | 配置脏数据限制和记录文件；修复源端数据 |
| HDFS小文件过多 | Flume roll参数设置过小 | 增大rollSize/rollInterval；定期合并小文件 |

---

> **总结**：数据采集是大数据平台的"生命线"。选型时应根据数据源类型（DB/日志/API）、延迟要求（实时/准实时/离线）和团队技术栈综合考量。无论选择哪种工具，都必须重视数据质量保障（完整性、唯一性、时效性校验）和运维监控（延迟、积压、成功率告警），确保数据从采集到入仓全链路可控、可追溯。
