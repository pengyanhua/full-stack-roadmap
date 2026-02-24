# Presto/Trino交互式查询引擎实战

## 1. Presto/Trino架构

### 1.1 发展历史

Presto诞生于2012年Facebook内部，最初是为了替代Hive在交互式查询场景下响应过慢的问题。
Facebook的数据团队需要一个能在秒级到分钟级完成PB级数据查询的引擎，于是Martin Traverso、
Dain Sundstrom、David Phillips和Eric Hwang四位工程师创建了Presto项目。

**发展时间线**：

```
Presto/Trino发展历程
┌──────────────────────────────────────────────────────────────────────┐
│  2012          2013         2018          2019         2020+         │
│   │             │            │             │            │            │
│   ↓             ↓            ↓             ↓            ↓            │
│ ┌──────┐    ┌──────┐    ┌──────┐    ┌──────────┐  ┌──────────┐     │
│ │FB内部 │ →  │ 开源  │ →  │Presto│ →  │  社区分裂 │→ │  Trino   │    │
│ │研发   │    │GitHub │    │基金会 │    │PrestoDB/ │  │  稳定发展 │    │
│ └──────┘    └──────┘    └──────┘    │PrestoSQL │  └──────────┘     │
│                                      └──────────┘                   │
│                                                                      │
│ 2012: Facebook内部开发Presto，解决Hive交互式查询慢的问题             │
│ 2013: Facebook将Presto开源到GitHub                                   │
│ 2018: Facebook将Presto捐赠给Linux基金会，成立Presto Software Foundation│
│ 2019: 原始创始人离开Facebook，fork项目为PrestoSQL                    │
│ 2020: PrestoSQL正式更名为Trino，避免商标争议                         │
│ 2021+: Trino社区快速发展，成为联邦查询引擎事实标准                   │
└──────────────────────────────────────────────────────────────────────┘
```

2019年的社区分裂是Presto历史上最重要的事件。原始创建者因为与Facebook在项目治理方向上
产生分歧，选择离开并创建了PrestoSQL（后更名为Trino）。此后两个项目走上了不同的发展道路。

**PrestoDB vs Trino 对比**：

| 维度 | PrestoDB | Trino (原PrestoSQL) |
|------|----------|---------------------|
| **治理模式** | Linux基金会 Presto Foundation | 社区驱动，Trino Software Foundation |
| **核心维护者** | Facebook/Meta工程师为主 | 原始创始人团队 (Martin, Dain, David) |
| **发版节奏** | 每几个月发布一次大版本 | 每周发布一个新版本 |
| **社区活跃度** | GitHub Star ~16k，较少外部贡献 | GitHub Star ~10k+，大量外部贡献者 |
| **商业支持** | Ahana (现已被IBM收购) | Starburst (创始人创立的公司) |
| **连接器数量** | ~20+ 内置连接器 | ~40+ 内置连接器 |
| **新特性** | 侧重Facebook内部需求 (Velox引擎) | 侧重社区需求 (多样化连接器、SQL扩展) |
| **Java版本** | Java 11+ | Java 17+ (积极跟进新版本) |
| **容错执行** | Project Aria (内部) | 内置容错执行模式 (Fault-tolerant execution) |
| **部署生态** | 与Facebook生态紧密 | Kubernetes Helm Chart、Docker原生支持 |

> **实践建议**：对于新项目，推荐选择Trino。Trino社区更活跃，连接器更丰富，文档更完善，
> 且原始创始人持续投入开发。本教程后续内容以Trino为主，相关概念同样适用于PrestoDB。

### 1.2 整体架构

Trino采用经典的MPP（大规模并行处理）架构，由一个Coordinator节点和多个Worker节点组成。
与Hive等引擎不同，Trino不依赖中间磁盘存储——所有数据处理都在内存中以Pipeline方式完成。

```
Trino整体架构
┌─────────────────────────────────────────────────────────────────────────────┐
│                              客户端层                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Trino CLI│  │   JDBC   │  │   ODBC   │  │ REST API │  │ Superset │    │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘    │
│        └──────────────┴──────────────┴──────────────┴──────────────┘        │
│                                     │ HTTP/HTTPS                            │
├─────────────────────────────────────┼───────────────────────────────────────┤
│                                     ↓                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Coordinator (协调节点)                            │  │
│  │  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌──────────────────┐   │  │
│  │  │  Parser  │→ │ Analyzer  │→ │ Planner & │→ │    Scheduler     │   │  │
│  │  │ SQL解析   │  │ 语义分析   │  │ Optimizer │  │ 分布式任务调度     │   │  │
│  │  │          │  │           │  │ 逻辑/物理  │  │                  │   │  │
│  │  └──────────┘  └───────────┘  │ 计划优化   │  └────────┬─────────┘   │  │
│  │                               └───────────┘           │             │  │
│  │  ┌────────────────────┐  ┌─────────────────────────┐  │             │  │
│  │  │  Metadata Manager  │  │    Discovery Service    │  │             │  │
│  │  │  (Catalog管理)     │  │    (Worker注册发现)      │  │             │  │
│  │  └────────────────────┘  └─────────────────────────┘  │             │  │
│  └───────────────────────────────────────────────────────┼─────────────┘  │
│                                                          │                 │
├──────────────────────────────────────────────────────────┼─────────────────┤
│                                                          ↓                 │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                       Worker节点集群 (N个)                             │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │ │
│  │  │   Worker #1     │  │   Worker #2     │  │   Worker #N     │      │ │
│  │  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │      │ │
│  │  │ │ Task Runner  │ │  │ │ Task Runner  │ │  │ │ Task Runner  │ │      │ │
│  │  │ │ ┌─────────┐ │ │  │ │ ┌─────────┐ │ │  │ │ ┌─────────┐ │ │      │ │
│  │  │ │ │Operator │ │ │  │ │ │Operator │ │ │  │ │ │Operator │ │ │      │ │
│  │  │ │ │Pipeline │ │ │  │ │ │Pipeline │ │ │  │ │ │Pipeline │ │ │      │ │
│  │  │ │ └─────────┘ │ │  │ │ └─────────┘ │ │  │ │ └─────────┘ │ │      │ │
│  │  │ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │      │ │
│  │  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │      │ │
│  │  │ │ Memory Pool │ │  │ │ Memory Pool │ │  │ │ Memory Pool │ │      │ │
│  │  │ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │      │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘      │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                     │                                      │
├─────────────────────────────────────┼──────────────────────────────────────┤
│                                     ↓                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                        Connector层 (插件化)                            │ │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ │ │
│  │  │  Hive  │ │ MySQL  │ │  ES    │ │ Kafka  │ │Iceberg │ │ Delta  │ │ │
│  │  │Connector│ │Connector│ │Connector│ │Connector│ │Connector│ │Connector│ │ │
│  │  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ │ │
│  └──────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────┘ │
│         ↓          ↓          ↓          ↓          ↓          ↓         │
│  ┌──────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐ ┌────────┐ │
│  │HDFS/S3/  │ │MySQL/  │ │Elastic │ │ Kafka  │ │ S3/HDFS  │ │ S3/HDFS│ │
│  │OSS       │ │PG/..   │ │search  │ │Cluster │ │ Iceberg  │ │ Delta  │ │
│  └──────────┘ └────────┘ └────────┘ └────────┘ └──────────┘ └────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

**MPP Pipeline执行模型**：

Trino的执行模型与传统MapReduce的"批处理"模型有本质区别。Trino采用Pipeline方式处理数据，
数据在各个Operator之间以流式方式传递，无需落盘。这使得Trino能够在数据扫描的同时就开始
进行聚合、过滤等操作，大幅降低查询延迟。

```
Pipeline执行 vs 批处理执行
┌─────────────────────────────────────────────────────────┐
│ Trino Pipeline执行（数据流式传递，无中间落盘）           │
│                                                          │
│  Scan ──→ Filter ──→ Join ──→ Aggregate ──→ Output      │
│   ↑         ↑         ↑         ↑            ↑          │
│   │ 数据    │ 数据    │ 数据    │  数据      │ 数据     │
│   │ 持续    │ 持续    │ 持续    │  持续      │ 持续     │
│   │ 流入    │ 流出    │ 流出    │  流出      │ 返回     │
│   时间 ─────────────────────────────────→               │
│   总延迟: 秒级~分钟级                                    │
├─────────────────────────────────────────────────────────┤
│ Hive/MR批处理执行（每个Stage写入HDFS再读取）            │
│                                                          │
│  Scan → [HDFS] → Map → [HDFS] → Reduce → [HDFS] → Out │
│          写盘          写盘            写盘              │
│   时间 ─────────────────────────────────→               │
│   总延迟: 分钟级~小时级                                  │
└─────────────────────────────────────────────────────────┘
```

### 1.3 核心概念

**Catalog/Schema/Table三级命名空间**：

Trino使用三级命名空间来组织数据，这是实现联邦查询的基础。每个外部数据源注册为一个Catalog，
Catalog下是Schema（对应数据库），Schema下是Table。

```
Trino三级命名空间层次结构
┌──────────────────────────────────────────────────────────────┐
│                        Trino Cluster                         │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Catalog: hive                                          │  │
│  │  ├── Schema: dw (数据仓库)                             │  │
│  │  │    ├── Table: user_events                           │  │
│  │  │    ├── Table: order_facts                           │  │
│  │  │    └── Table: product_dim                           │  │
│  │  ├── Schema: ods (原始数据层)                          │  │
│  │  │    ├── Table: raw_logs                              │  │
│  │  │    └── Table: raw_transactions                      │  │
│  │  └── Schema: ads (应用数据层)                          │  │
│  │       └── Table: daily_report                          │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Catalog: mysql_app                                     │  │
│  │  ├── Schema: user_center                               │  │
│  │  │    ├── Table: users                                 │  │
│  │  │    └── Table: user_profiles                         │  │
│  │  └── Schema: order_service                             │  │
│  │       ├── Table: orders                                │  │
│  │       └── Table: payments                              │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Catalog: es_search                                     │  │
│  │  └── Schema: default                                   │  │
│  │       ├── Table: product_index                         │  │
│  │       └── Table: log_index                             │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  完全限定表名: catalog.schema.table                          │
│  示例: hive.dw.user_events                                   │
│        mysql_app.user_center.users                           │
│        es_search.default.product_index                       │
└──────────────────────────────────────────────────────────────┘
```

**Connector插件架构**：

Connector是Trino连接外部数据源的核心抽象。每个Connector实现了一组SPI（Service Provider Interface），
包括元数据管理、数据读取、数据写入和谓词下推等能力。

```java
// Connector SPI核心接口
public interface Connector {
    // 获取事务处理器
    ConnectorTransactionHandle beginTransaction(
        IsolationLevel isolationLevel, boolean readOnly);

    // 获取元数据管理器 — 提供表、列、分区等元信息
    ConnectorMetadata getMetadata(ConnectorTransactionHandle transaction);

    // 获取Split管理器 — 决定数据如何被切分为并行任务
    ConnectorSplitManager getSplitManager();

    // 获取数据读取器工厂 — 从Split中读取数据页(Page)
    ConnectorPageSourceProvider getPageSourceProvider();

    // 获取数据写入器工厂 — 将数据写入目标数据源
    ConnectorPageSinkProvider getPageSinkProvider();
}
```

**查询生命周期**：

一条SQL从提交到返回结果，在Trino内部经历以下完整流程：

```
查询生命周期（Query Lifecycle）
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ① SQL提交                                                       │
│     │  "SELECT * FROM hive.dw.orders WHERE dt='2024-01-01'"     │
│     ↓                                                            │
│  ② Parse（语法解析）                                             │
│     │  SQL文本 → AST（抽象语法树）                               │
│     │  使用ANTLR语法解析器                                       │
│     ↓                                                            │
│  ③ Analyze（语义分析）                                           │
│     │  AST → 带类型信息的IR                                      │
│     │  验证表/列是否存在、类型是否匹配、权限是否足够             │
│     │  从Connector获取元数据信息                                  │
│     ↓                                                            │
│  ④ Plan（逻辑计划生成）                                          │
│     │  IR → 逻辑计划树                                           │
│     │  TableScan → Filter → Project → Aggregate → ...            │
│     ↓                                                            │
│  ⑤ Optimize（计划优化）                                          │
│     │  逻辑计划 → 优化后逻辑计划                                 │
│     │  谓词下推、列裁剪、Join重排序、常量折叠                    │
│     │  基于规则(RBO) + 基于代价(CBO)的混合优化                   │
│     ↓                                                            │
│  ⑥ Fragment & Schedule（物理计划 & 调度）                        │
│     │  逻辑计划 → 分布式物理计划（多个Stage/Fragment）           │
│     │  Coordinator将Task分配到各Worker节点                       │
│     ↓                                                            │
│  ⑦ Execute（分布式执行）                                         │
│     │  各Worker并行执行Task                                      │
│     │  数据以Page为单位在Operator之间流式传递                    │
│     │  Stage之间通过Exchange（网络Shuffle）传输数据              │
│     ↓                                                            │
│  ⑧ Return（结果返回）                                            │
│     │  最终Stage的输出汇聚到Coordinator                          │
│     │  以分页方式流式返回给客户端                                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. 部署与配置

### 2.1 集群部署

**下载与安装**：

```bash
# 下载Trino Server（以433版本为例）
wget https://repo1.maven.org/maven2/io/trino/trino-server/433/trino-server-433.tar.gz

# 解压
tar -xzf trino-server-433.tar.gz
mv trino-server-433 /opt/trino

# 下载Trino CLI（可选，用于命令行查询）
wget https://repo1.maven.org/maven2/io/trino/trino-cli/433/trino-cli-433-executable.jar
mv trino-cli-433-executable.jar /usr/local/bin/trino
chmod +x /usr/local/bin/trino
```

**目录结构**：

```
/opt/trino/
├── bin/
│   └── launcher              # 启动脚本
├── lib/                      # Trino核心jar包
├── plugin/                   # 各Connector插件目录
│   ├── hive/
│   ├── mysql/
│   ├── elasticsearch/
│   ├── kafka/
│   ├── iceberg/
│   └── ...
├── etc/                      # 配置文件目录（需手动创建）
│   ├── config.properties     # 主配置文件
│   ├── jvm.config            # JVM参数
│   ├── node.properties       # 节点标识
│   ├── log.properties        # 日志级别
│   └── catalog/              # 各数据源连接器配置
│       ├── hive.properties
│       ├── mysql.properties
│       └── ...
└── data/                     # 运行时数据目录
    └── var/
        ├── log/              # 服务日志
        └── run/              # PID文件
```

**Coordinator节点配置**：

```properties
# etc/config.properties (Coordinator)
coordinator=true
node-scheduler.include-coordinator=false
http-server.http.port=8080
query.max-memory=50GB
query.max-memory-per-node=10GB
query.max-total-memory-per-node=12GB
discovery.uri=http://coordinator-host:8080
```

**Worker节点配置**：

```properties
# etc/config.properties (Worker)
coordinator=false
http-server.http.port=8080
query.max-memory=50GB
query.max-memory-per-node=10GB
query.max-total-memory-per-node=12GB
discovery.uri=http://coordinator-host:8080
```

**JVM配置（Coordinator和Worker通用）**：

```properties
# etc/jvm.config
-server
-Xmx16G
-XX:InitialRAMPercentage=80
-XX:MaxRAMPercentage=80
-XX:G1HeapRegionSize=32M
-XX:+ExplicitGCInvokesConcurrent
-XX:+ExitOnOutOfMemoryError
-XX:+HeapDumpOnOutOfMemoryError
-XX:-OmitStackTraceInFastThrow
-XX:ReservedCodeCacheSize=512M
-XX:PerMethodRecompilationCutoff=10000
-XX:PerBytecodeRecompilationCutoff=10000
-Djdk.attach.allowAttachSelf=true
-Djdk.nio.maxCachedBufferSize=2000000
-Dfile.encoding=UTF-8
-XX:+UnlockDiagnosticVMOptions
-XX:+UseAESCTRIntrinsics
```

**节点属性配置**：

```properties
# etc/node.properties
node.environment=production
node.id=ffffffff-ffff-ffff-ffff-ffffffffffff
node.data-dir=/opt/trino/data
```

> **注意**: 每个节点的 `node.id` 必须唯一。可使用 `uuidgen` 命令生成。

**启动与管理**：

```bash
# 后台启动
/opt/trino/bin/launcher start

# 前台启动（用于调试）
/opt/trino/bin/launcher run

# 停止
/opt/trino/bin/launcher stop

# 查看状态
/opt/trino/bin/launcher status

# 使用CLI连接
trino --server http://coordinator-host:8080 --catalog hive --schema dw

# 验证集群状态
trino> SELECT * FROM system.runtime.nodes;
```

### 2.2 Connector配置

**Hive Connector（连接Hive Metastore + HDFS/S3）**：

```properties
# etc/catalog/hive.properties
connector.name=hive

# Hive Metastore配置
hive.metastore.uri=thrift://metastore-host:9083

# HDFS配置
hive.config.resources=/opt/trino/etc/core-site.xml,/opt/trino/etc/hdfs-site.xml

# S3配置（如果数据在S3上）
# hive.s3.aws-access-key=YOUR_ACCESS_KEY
# hive.s3.aws-secret-key=YOUR_SECRET_KEY
# hive.s3.endpoint=https://s3.amazonaws.com

# 性能优化
hive.max-partitions-per-scan=100000
hive.allow-drop-table=true
hive.allow-rename-table=true
hive.allow-add-column=true
hive.parquet.use-column-names=true
hive.orc.use-column-names=true

# 缓存配置
hive.metastore-cache-ttl=2m
hive.metastore-refresh-interval=1m
hive.file-status-cache-expire-time=10m
hive.file-status-cache-tables=*
```

**MySQL Connector**：

```properties
# etc/catalog/mysql.properties
connector.name=mysql

# 连接配置
connection-url=jdbc:mysql://mysql-host:3306
connection-user=trino_reader
connection-password=your_password

# 连接池配置
# 注意: MySQL Connector使用JDBC连接池
jdbc.connection-pool.max-size=30
jdbc.connection-pool.min-size=5

# 类型映射
mysql.auto-reconnect=true
mysql.jdbc.use-information-schema=true
```

**Elasticsearch Connector**：

```properties
# etc/catalog/elasticsearch.properties
connector.name=elasticsearch

# ES集群配置
elasticsearch.host=es-host
elasticsearch.port=9200
elasticsearch.default-schema-name=default

# 安全配置（如果启用了x-pack security）
# elasticsearch.security=PASSWORD
# elasticsearch.auth.user=elastic
# elasticsearch.auth.password=your_password

# TLS配置
# elasticsearch.tls.enabled=true
# elasticsearch.tls.verify-hostnames=false

# 性能配置
elasticsearch.scroll-size=1000
elasticsearch.scroll-timeout=1m
elasticsearch.request-timeout=30s
elasticsearch.max-hits=1000000
```

**Kafka Connector**：

```properties
# etc/catalog/kafka.properties
connector.name=kafka

# Kafka集群配置
kafka.nodes=kafka-broker1:9092,kafka-broker2:9092,kafka-broker3:9092

# Schema Registry配置（用于Avro/Protobuf格式）
# kafka.confluent-schema-registry-url=http://schema-registry:8081

# 默认schema
kafka.default-schema=default

# 主题映射 — 通过JSON文件定义Topic到Table的映射
kafka.table-description-dir=/opt/trino/etc/kafka
kafka.hide-internal-columns=false
```

**Kafka Topic映射文件示例**：

```json
// etc/kafka/user_events.json
{
  "tableName": "user_events",
  "schemaName": "default",
  "topicName": "app.user.events",
  "key": {
    "dataFormat": "raw",
    "fields": [
      {
        "name": "kafka_key",
        "dataFormat": "raw",
        "type": "VARCHAR",
        "hidden": "false"
      }
    ]
  },
  "message": {
    "dataFormat": "json",
    "fields": [
      {
        "name": "user_id",
        "mapping": "user_id",
        "type": "BIGINT"
      },
      {
        "name": "event_type",
        "mapping": "event_type",
        "type": "VARCHAR"
      },
      {
        "name": "event_time",
        "mapping": "event_time",
        "type": "TIMESTAMP",
        "dataFormat": "iso8601"
      },
      {
        "name": "properties",
        "mapping": "properties",
        "type": "VARCHAR"
      }
    ]
  }
}
```

**Iceberg Connector（现代数据湖表格式）**：

```properties
# etc/catalog/iceberg.properties
connector.name=iceberg

# Iceberg元数据管理方式
iceberg.catalog.type=hive_metastore
hive.metastore.uri=thrift://metastore-host:9083

# S3存储配置
# hive.s3.aws-access-key=YOUR_ACCESS_KEY
# hive.s3.aws-secret-key=YOUR_SECRET_KEY

# Iceberg特性
iceberg.file-format=PARQUET
iceberg.compression-codec=ZSTD
iceberg.max-partitions-per-writer=100
```

### 2.3 资源管理

Trino通过Resource Groups机制对查询进行分组管理和资源限制，防止单个用户或查询耗尽集群资源。

**Resource Groups JSON配置**：

```json
// etc/resource-groups.json
{
  "rootGroups": [
    {
      "name": "global",
      "softMemoryLimit": "90%",
      "hardConcurrencyLimit": 100,
      "maxQueued": 500,
      "subGroups": [
        {
          "name": "etl",
          "softMemoryLimit": "50%",
          "hardConcurrencyLimit": 20,
          "maxQueued": 100,
          "schedulingWeight": 3,
          "schedulingPolicy": "weighted_fair"
        },
        {
          "name": "adhoc",
          "softMemoryLimit": "30%",
          "hardConcurrencyLimit": 50,
          "maxQueued": 200,
          "schedulingWeight": 1,
          "subGroups": [
            {
              "name": "analyst",
              "softMemoryLimit": "60%",
              "hardConcurrencyLimit": 30,
              "maxQueued": 100
            },
            {
              "name": "developer",
              "softMemoryLimit": "40%",
              "hardConcurrencyLimit": 20,
              "maxQueued": 50
            }
          ]
        },
        {
          "name": "dashboard",
          "softMemoryLimit": "20%",
          "hardConcurrencyLimit": 30,
          "maxQueued": 300,
          "schedulingWeight": 2
        }
      ]
    }
  ],
  "selectors": [
    {
      "source": "etl-scheduler",
      "group": "global.etl"
    },
    {
      "source": "superset",
      "group": "global.dashboard"
    },
    {
      "user": "analyst_.*",
      "group": "global.adhoc.analyst"
    },
    {
      "group": "global.adhoc.developer"
    }
  ]
}
```

**Resource Groups配置加载**：

```properties
# etc/resource-groups.properties
resource-groups.configuration-manager=file
resource-groups.config-file=/opt/trino/etc/resource-groups.json
```

**内存管理参数详解**：

```properties
# etc/config.properties — 内存相关参数

# 单个查询在整个集群可使用的最大总内存
query.max-memory=50GB

# 单个查询在单个Worker可使用的最大用户内存
query.max-memory-per-node=10GB

# 单个查询在单个Worker可使用的最大总内存（用户内存 + 系统内存）
query.max-total-memory-per-node=12GB

# 内存不足时的kill策略：total-reservation（杀最大的）或 total-reservation-on-blocked-nodes
query.low-memory-killer.policy=total-reservation-on-blocked-nodes

# 低内存kill延迟（避免误杀短查询）
query.low-memory-killer.delay=5m
```

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `query.max-memory` | 20GB | 单查询集群总内存上限 | 设为集群总内存的50%-70% |
| `query.max-memory-per-node` | JVM max * 0.3 | 单查询单节点内存上限 | 设为JVM堆的40%-60% |
| `memory.heap-headroom-per-node` | JVM max * 0.3 | 预留给非查询的堆内存 | 保持默认或降至20% |
| `query.max-total-memory-per-node` | query.max-memory-per-node * 2 | 含系统内存的上限 | 设为query.max-memory-per-node的1.2x |

---

## 3. SQL查询实战

### 3.1 基础查询

Trino兼容大部分ANSI SQL标准，并提供丰富的内置函数。以下是常用查询模式。

**基础SELECT与过滤**：

```sql
-- 基础查询
SELECT
    user_id,
    user_name,
    email,
    created_at
FROM mysql_app.user_center.users
WHERE created_at >= DATE '2024-01-01'
  AND status = 'active'
ORDER BY created_at DESC
LIMIT 100;

-- 多条件过滤
SELECT *
FROM hive.dw.order_facts
WHERE dt BETWEEN '2024-01-01' AND '2024-01-31'
  AND order_status IN ('completed', 'shipped')
  AND amount > 100.00;
```

**各种JOIN操作**：

```sql
-- INNER JOIN: 只返回两表匹配的行
SELECT
    o.order_id,
    o.amount,
    u.user_name,
    u.email
FROM hive.dw.order_facts o
INNER JOIN mysql_app.user_center.users u
    ON o.user_id = u.user_id
WHERE o.dt = '2024-01-15';

-- LEFT JOIN: 返回左表所有行，右表不匹配则为NULL
SELECT
    u.user_id,
    u.user_name,
    COALESCE(SUM(o.amount), 0) AS total_amount,
    COUNT(o.order_id) AS order_count
FROM mysql_app.user_center.users u
LEFT JOIN hive.dw.order_facts o
    ON u.user_id = o.user_id
    AND o.dt >= '2024-01-01'
GROUP BY u.user_id, u.user_name;

-- FULL OUTER JOIN: 返回两表所有行
SELECT
    COALESCE(a.user_id, b.user_id) AS user_id,
    a.page_views,
    b.purchases
FROM hive.dw.daily_page_views a
FULL OUTER JOIN hive.dw.daily_purchases b
    ON a.user_id = b.user_id AND a.dt = b.dt
WHERE COALESCE(a.dt, b.dt) = '2024-01-15';

-- CROSS JOIN: 笛卡尔积（慎用，通常配合UNNEST使用）
SELECT
    d.dt,
    c.category_name
FROM (SELECT DISTINCT dt FROM hive.dw.dim_date WHERE year = 2024) d
CROSS JOIN hive.dw.dim_category c;

-- SEMI JOIN (使用 EXISTS/IN): 只要右表存在匹配就返回左表行
SELECT u.*
FROM mysql_app.user_center.users u
WHERE EXISTS (
    SELECT 1 FROM hive.dw.order_facts o
    WHERE o.user_id = u.user_id AND o.dt >= '2024-01-01'
);
```

**GROUP BY与聚合**：

```sql
-- 基础聚合
SELECT
    product_category,
    COUNT(*) AS order_count,
    SUM(amount) AS total_revenue,
    AVG(amount) AS avg_order_value,
    MIN(amount) AS min_order,
    MAX(amount) AS max_order
FROM hive.dw.order_facts
WHERE dt BETWEEN '2024-01-01' AND '2024-03-31'
GROUP BY product_category
HAVING SUM(amount) > 10000
ORDER BY total_revenue DESC;

-- GROUPING SETS: 多维度聚合
SELECT
    COALESCE(region, '全部区域') AS region,
    COALESCE(product_category, '全部品类') AS category,
    SUM(amount) AS revenue,
    COUNT(*) AS order_count
FROM hive.dw.order_facts
WHERE dt = '2024-01-15'
GROUP BY GROUPING SETS (
    (region, product_category),   -- 按区域+品类
    (region),                      -- 按区域汇总
    (product_category),            -- 按品类汇总
    ()                             -- 总计
);
```

**窗口函数（Window Functions）**：

```sql
-- ROW_NUMBER: 分组排序编号
SELECT *
FROM (
    SELECT
        user_id,
        order_id,
        amount,
        order_time,
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY order_time DESC
        ) AS rn
    FROM hive.dw.order_facts
    WHERE dt >= '2024-01-01'
) t
WHERE rn <= 3;  -- 每个用户最近3笔订单

-- RANK / DENSE_RANK: 排名（含并列）
SELECT
    product_id,
    product_name,
    total_sales,
    RANK() OVER (ORDER BY total_sales DESC) AS sales_rank,
    DENSE_RANK() OVER (ORDER BY total_sales DESC) AS dense_rank
FROM (
    SELECT
        product_id,
        product_name,
        SUM(amount) AS total_sales
    FROM hive.dw.order_facts
    WHERE dt BETWEEN '2024-01-01' AND '2024-01-31'
    GROUP BY product_id, product_name
) t;

-- LAG / LEAD: 前后行对比
SELECT
    dt,
    daily_revenue,
    LAG(daily_revenue, 1) OVER (ORDER BY dt) AS prev_day_revenue,
    LEAD(daily_revenue, 1) OVER (ORDER BY dt) AS next_day_revenue,
    daily_revenue - LAG(daily_revenue, 1) OVER (ORDER BY dt) AS day_over_day,
    ROUND(
        (daily_revenue - LAG(daily_revenue, 1) OVER (ORDER BY dt))
        * 100.0 / LAG(daily_revenue, 1) OVER (ORDER BY dt), 2
    ) AS growth_rate_pct
FROM (
    SELECT dt, SUM(amount) AS daily_revenue
    FROM hive.dw.order_facts
    WHERE dt BETWEEN '2024-01-01' AND '2024-01-31'
    GROUP BY dt
) daily_stats
ORDER BY dt;

-- SUM / AVG窗口累计
SELECT
    dt,
    daily_revenue,
    SUM(daily_revenue) OVER (
        ORDER BY dt
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_revenue,
    AVG(daily_revenue) OVER (
        ORDER BY dt
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_avg
FROM (
    SELECT dt, SUM(amount) AS daily_revenue
    FROM hive.dw.order_facts
    GROUP BY dt
) t
ORDER BY dt;
```

**ANSI SQL兼容性**：

| SQL特性 | Trino支持 | 说明 |
|---------|-----------|------|
| SELECT/WHERE/ORDER BY | 完全支持 | 标准SQL语法 |
| JOIN (INNER/LEFT/RIGHT/FULL/CROSS) | 完全支持 | 含跨Catalog JOIN |
| GROUP BY / HAVING | 完全支持 | 含GROUPING SETS/CUBE/ROLLUP |
| 窗口函数 | 完全支持 | ROW_NUMBER, RANK, LAG, LEAD, NTILE等 |
| CTE (WITH) | 完全支持 | 支持递归CTE |
| 子查询 | 完全支持 | 相关/非相关子查询 |
| UNION / INTERSECT / EXCEPT | 完全支持 | 集合操作 |
| VALUES | 完全支持 | 内联数据 |
| LATERAL JOIN | 完全支持 | 横向连接 |
| INSERT / UPDATE / DELETE | 部分支持 | 依赖Connector能力 |
| CREATE TABLE AS SELECT | 完全支持 | CTAS语句 |
| MERGE | 部分支持 | 仅Iceberg/Delta等Connector |

### 3.2 跨数据源联邦查询

联邦查询是Trino最强大的能力之一——在一条SQL中同时查询多个不同的数据源，无需ETL。

**场景一：Hive数仓 + MySQL业务库联合查询**：

```sql
-- 用户行为分析：将行为日志（Hive）与用户信息（MySQL）关联
-- hive.dw.user_events: 存储在HDFS/S3上的用户行为事件，按天分区
-- mysql_app.user_center.users: MySQL中的用户基础信息表
SELECT
    u.user_id,
    u.user_name,
    u.city,
    u.register_channel,
    COUNT(DISTINCT e.session_id) AS session_count,
    COUNT(*) AS event_count,
    COUNT(DISTINCT e.dt) AS active_days,
    MIN(e.event_time) AS first_event_time,
    MAX(e.event_time) AS last_event_time
FROM hive.dw.user_events e
JOIN mysql_app.user_center.users u
    ON e.user_id = u.user_id
WHERE e.dt BETWEEN '2024-01-01' AND '2024-01-31'
  AND u.status = 'active'
  AND u.created_at < DATE '2024-01-01'  -- 仅老用户
GROUP BY u.user_id, u.user_name, u.city, u.register_channel
HAVING COUNT(*) >= 10  -- 活跃度筛选
ORDER BY event_count DESC
LIMIT 1000;
```

**场景二：Hive + Elasticsearch联合查询**：

```sql
-- 商品搜索增强：ES全文检索 + Hive统计数据
-- es_search.default.product_index: Elasticsearch中的商品索引
-- hive.dw.product_stats: Hive中的商品销售统计表
SELECT
    es.product_id,
    es.product_name,
    es.category,
    es._score AS search_score,
    stats.total_sales,
    stats.avg_rating,
    stats.review_count
FROM es_search.default.product_index es
JOIN hive.dw.product_stats stats
    ON es.product_id = stats.product_id
WHERE es.query = 'keyword:手机 AND brand:华为'
  AND stats.total_sales > 100
ORDER BY es._score * LOG10(stats.total_sales + 1) DESC
LIMIT 50;
```

**场景三：三源联合查询（Hive + MySQL + Iceberg）**：

```sql
-- 实时订单分析：结合实时数据（Iceberg）与历史数据（Hive）和用户维度（MySQL）
WITH recent_orders AS (
    -- Iceberg表：近实时更新的订单数据
    SELECT order_id, user_id, amount, order_status, order_time
    FROM iceberg.ods.orders_realtime
    WHERE order_time >= CURRENT_TIMESTAMP - INTERVAL '1' HOUR
),
user_segments AS (
    -- MySQL表：用户分层标签
    SELECT user_id, vip_level, lifetime_value_tier
    FROM mysql_app.user_center.user_profiles
    WHERE vip_level >= 3  -- 高价值用户
),
history_stats AS (
    -- Hive表：用户历史消费统计
    SELECT user_id, SUM(amount) AS history_total
    FROM hive.dw.order_facts
    WHERE dt >= '2024-01-01'
    GROUP BY user_id
)
SELECT
    ro.order_id,
    ro.user_id,
    us.vip_level,
    us.lifetime_value_tier,
    ro.amount AS current_order_amount,
    hs.history_total,
    ro.amount / NULLIF(hs.history_total, 0) * 100 AS pct_of_history
FROM recent_orders ro
JOIN user_segments us ON ro.user_id = us.user_id
LEFT JOIN history_stats hs ON ro.user_id = hs.user_id
ORDER BY ro.amount DESC;
```

### 3.3 高级函数

**UNNEST — 展开数组和MAP**：

```sql
-- 展开数组
SELECT
    order_id,
    tag
FROM hive.dw.orders
CROSS JOIN UNNEST(tags) AS t(tag)
WHERE dt = '2024-01-15';

-- 展开MAP为键值对
SELECT
    event_id,
    prop_key,
    prop_value
FROM hive.dw.user_events
CROSS JOIN UNNEST(properties) AS t(prop_key, prop_value)
WHERE dt = '2024-01-15'
  AND prop_key = 'page_name';

-- 展开带序号
SELECT
    user_id,
    idx,
    item_id
FROM hive.dw.user_cart
CROSS JOIN UNNEST(item_ids) WITH ORDINALITY AS t(item_id, idx)
WHERE dt = '2024-01-15';
```

**JSON函数**：

```sql
-- JSON_EXTRACT: 从JSON字符串中提取值
SELECT
    event_id,
    JSON_EXTRACT(payload, '$.user_id') AS user_id_json,
    JSON_EXTRACT_SCALAR(payload, '$.action') AS action,
    CAST(JSON_EXTRACT_SCALAR(payload, '$.amount') AS DOUBLE) AS amount,
    JSON_EXTRACT(payload, '$.items') AS items_array
FROM hive.ods.raw_events
WHERE dt = '2024-01-15';

-- JSON_ARRAY_LENGTH + JSON_EXTRACT 结合
SELECT
    order_id,
    JSON_ARRAY_LENGTH(JSON_EXTRACT(order_detail, '$.items')) AS item_count,
    JSON_EXTRACT_SCALAR(
        JSON_ARRAY_GET(JSON_EXTRACT(order_detail, '$.items'), 0),
        '$.name'
    ) AS first_item_name
FROM hive.ods.raw_orders
WHERE dt = '2024-01-15';

-- JSON_FORMAT / JSON_PARSE
SELECT
    JSON_FORMAT(JSON_PARSE('{"key": "value"}')) AS formatted,
    JSON_ARRAY_CONTAINS(JSON_PARSE('[1, 2, 3]'), 2) AS contains_two;
```

**Lambda表达式（数组/MAP高阶函数）**：

```sql
-- TRANSFORM: 对数组每个元素进行转换
SELECT
    user_id,
    scores,
    TRANSFORM(scores, x -> x * 1.1) AS adjusted_scores,
    TRANSFORM(scores, x -> CAST(x AS VARCHAR)) AS score_strings
FROM hive.dw.user_scores
WHERE dt = '2024-01-15';

-- FILTER: 过滤数组元素
SELECT
    order_id,
    items,
    FILTER(items, x -> x.price > 100) AS expensive_items,
    CARDINALITY(FILTER(items, x -> x.price > 100)) AS expensive_count
FROM hive.dw.order_details
WHERE dt = '2024-01-15';

-- REDUCE: 数组聚合
SELECT
    user_id,
    amounts,
    REDUCE(
        amounts,
        0.0,
        (s, x) -> s + x,     -- 累加器
        s -> s                 -- 最终转换
    ) AS total_amount,
    REDUCE(
        amounts,
        CAST(ROW(0.0, 0) AS ROW(sum DOUBLE, count INTEGER)),
        (s, x) -> CAST(ROW(s.sum + x, s.count + 1) AS ROW(sum DOUBLE, count INTEGER)),
        s -> s.sum / s.count
    ) AS avg_amount
FROM hive.dw.user_transactions;

-- MAP_FILTER / MAP_TRANSFORM_KEYS / MAP_TRANSFORM_VALUES
SELECT
    event_id,
    properties,
    MAP_FILTER(properties, (k, v) -> k LIKE 'utm_%') AS utm_params,
    MAP_TRANSFORM_VALUES(properties, (k, v) -> UPPER(v)) AS upper_values
FROM hive.dw.user_events
WHERE dt = '2024-01-15';
```

**近似函数（适用于海量数据快速估算）**：

```sql
-- approx_distinct: 基于HyperLogLog的近似去重计数
-- 比COUNT(DISTINCT)快数倍，误差约2.3%
SELECT
    dt,
    approx_distinct(user_id) AS approx_uv,
    COUNT(DISTINCT user_id) AS exact_uv  -- 对比精确值
FROM hive.dw.user_events
WHERE dt BETWEEN '2024-01-01' AND '2024-01-31'
GROUP BY dt
ORDER BY dt;

-- approx_percentile: 近似百分位数
SELECT
    product_category,
    approx_percentile(amount, 0.5) AS median_amount,       -- P50
    approx_percentile(amount, 0.9) AS p90_amount,           -- P90
    approx_percentile(amount, 0.99) AS p99_amount,          -- P99
    approx_percentile(amount, ARRAY[0.25, 0.5, 0.75]) AS quartiles  -- 四分位
FROM hive.dw.order_facts
WHERE dt = '2024-01-15'
GROUP BY product_category;

-- approx_most_frequent: 近似TopN高频值
SELECT
    approx_most_frequent(10, search_keyword, 10000) AS top_keywords
FROM hive.dw.search_logs
WHERE dt = '2024-01-15';
```

### 3.4 EXPLAIN分析

EXPLAIN是理解查询执行计划和进行性能调优的核心工具。

**基础EXPLAIN**：

```sql
-- 逻辑执行计划
EXPLAIN
SELECT
    u.city,
    COUNT(*) AS order_count,
    SUM(o.amount) AS total_amount
FROM hive.dw.order_facts o
JOIN mysql_app.user_center.users u ON o.user_id = u.user_id
WHERE o.dt = '2024-01-15'
GROUP BY u.city;
```

输出示例（简化）：

```
- Output[city, order_count, total_amount]
    - Aggregate(FINAL)[city] => [count:bigint, sum:double]
        - LocalExchange[HASH][$hashvalue] (city)
            - RemoteExchange[REPARTITION][$hashvalue_0] (city)
                - Aggregate(PARTIAL)[city]
                    - InnerJoin[o.user_id = u.user_id]
                        Distribution: REPLICATED
                        - FilteredTableScan[hive:dw.order_facts]
                            dt = '2024-01-15'  (谓词下推到分区)
                        - LocalExchange[HASH][$hashvalue_1] (user_id)
                            - RemoteExchange[REPLICATE]
                                - TableScan[mysql:user_center.users]
```

**分布式执行计划**：

```sql
-- 查看分布式物理计划（含Stage划分和数据交换方式）
EXPLAIN (TYPE DISTRIBUTED)
SELECT
    u.city,
    COUNT(*) AS order_count,
    SUM(o.amount) AS total_amount
FROM hive.dw.order_facts o
JOIN mysql_app.user_center.users u ON o.user_id = u.user_id
WHERE o.dt = '2024-01-15'
GROUP BY u.city;
```

**如何解读执行计划**：

```
分布式执行计划阶段解析
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Fragment 0 [SINGLE]                    ← 输出Stage(Coordinator) │
│  │  Output[city, order_count, total_amount]                      │
│  │  └── Aggregate(FINAL)[city]                                   │
│  │      └── RemoteSource[1]             ← 从Fragment 1接收数据   │
│  │                                                               │
│  Fragment 1 [HASH: city]                ← 中间Stage(按city分区)  │
│  │  Aggregate(PARTIAL)[city]                                     │
│  │  └── InnerJoin[user_id]                                       │
│  │      ├── RemoteSource[2]             ← 左表数据               │
│  │      └── RemoteSource[3]             ← 右表数据(REPLICATE)    │
│  │                                                               │
│  Fragment 2 [SOURCE: hive]              ← 源Stage(读Hive)        │
│  │  TableScan[hive:dw.order_facts]                               │
│  │  Predicate: dt = '2024-01-15'        ← 谓词已下推             │
│  │                                                               │
│  Fragment 3 [SOURCE: mysql]             ← 源Stage(读MySQL)       │
│  │  TableScan[mysql:user_center.users]                           │
│  │  Distribution: REPLICATE             ← 小表广播到所有Worker    │
│  │                                                               │
│  数据流向:                                                        │
│  Fragment 2 ──┐                                                   │
│               ├──→ Fragment 1 ──→ Fragment 0 ──→ Client          │
│  Fragment 3 ──┘    (Exchange)      (Exchange)                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**关键执行计划元素含义**：

| 计划元素 | 含义 | 性能提示 |
|----------|------|----------|
| `Fragment [SINGLE]` | 单节点执行的Stage | 最终汇聚Stage，避免返回太多数据 |
| `Fragment [HASH: col]` | 按col哈希分区执行 | 数据按key重分布 |
| `Fragment [SOURCE]` | 数据源读取Stage | 并行度取决于Split数量 |
| `RemoteExchange[REPARTITION]` | Shuffle数据交换 | 网络开销大，关注数据量 |
| `RemoteExchange[REPLICATE]` | 广播数据交换 | 小表广播到所有Worker |
| `LocalExchange[HASH]` | 本地数据重分区 | Worker内部线程间数据交换 |
| `Aggregate(PARTIAL)` | 局部预聚合 | 在数据交换前减少数据量 |
| `Aggregate(FINAL)` | 最终聚合 | 汇合所有局部聚合结果 |
| `Distribution: REPLICATED` | Join时小表广播 | 自动选择或手动指定 |
| `Distribution: PARTITIONED` | Join时双表Shuffle | 两个大表Join |

**EXPLAIN ANALYZE — 含实际执行统计的计划**：

```sql
-- 实际执行并收集运行时统计
EXPLAIN ANALYZE
SELECT
    dt,
    COUNT(*) AS cnt
FROM hive.dw.user_events
WHERE dt BETWEEN '2024-01-01' AND '2024-01-07'
GROUP BY dt;

-- 输出中包含实际的行数、数据量、耗时等信息：
-- TableScan: rows=12345678, size=2.3GB, cpu=15.2s, wall=3.1s
-- Aggregate(PARTIAL): rows=7, cpu=0.5s
-- 这些信息帮助定位实际的性能瓶颈
```

---

## 4. 性能优化

### 4.1 查询优化

**谓词下推（Predicate Pushdown）**：

谓词下推是Trino最重要的优化之一，它将过滤条件推送到数据源层执行，减少传输到Trino的数据量。

```sql
-- ✅ 正确：分区谓词可以下推，只扫描指定分区的数据
SELECT * FROM hive.dw.order_facts
WHERE dt = '2024-01-15'    -- 分区列，直接下推到Hive Metastore
  AND amount > 100;         -- 列过滤，可下推到ORC/Parquet Reader

-- ❌ 错误：对分区列使用函数，导致无法下推
SELECT * FROM hive.dw.order_facts
WHERE YEAR(dt) = 2024       -- 函数包裹分区列，无法进行分区裁剪
  AND MONTH(dt) = 1;        -- 将扫描全部分区！

-- ✅ 正确改写：使用范围条件替代函数
SELECT * FROM hive.dw.order_facts
WHERE dt BETWEEN '2024-01-01' AND '2024-01-31';  -- 直接下推

-- ✅ 正确：MySQL Connector的谓词下推
SELECT * FROM mysql_app.user_center.users
WHERE user_id = 12345;      -- 等值条件下推到MySQL执行
-- Trino生成的SQL: SELECT ... FROM users WHERE user_id = 12345

-- ❌ 注意：复杂表达式可能无法下推
SELECT * FROM mysql_app.user_center.users
WHERE UPPER(email) = 'TEST@EXAMPLE.COM';  -- 函数在Trino端执行
-- MySQL端扫描全表后在Trino端过滤

-- ✅ Elasticsearch的谓词下推
SELECT * FROM es_search.default.product_index
WHERE query = 'category:电子产品 AND price:[100 TO 500]';
-- 查询条件直接下推到ES执行
```

**分区裁剪（Partition Pruning）**：

```sql
-- ✅ 高效：直接使用分区列进行过滤
SELECT COUNT(*)
FROM hive.dw.user_events
WHERE dt = '2024-01-15'        -- 一级分区
  AND hour = '14';              -- 二级分区
-- 只扫描 dt=2024-01-15/hour=14 这一个分区

-- ❌ 低效：不使用分区列或使用不当
SELECT COUNT(*)
FROM hive.dw.user_events
WHERE event_time BETWEEN TIMESTAMP '2024-01-15 14:00:00'
                      AND TIMESTAMP '2024-01-15 14:59:59';
-- event_time不是分区列，将扫描所有分区后再过滤

-- ✅ 动态分区裁剪（Dynamic Partition Pruning）
-- Trino可以利用Join中的过滤条件动态裁剪分区
SELECT e.*
FROM hive.dw.user_events e
JOIN (
    SELECT DISTINCT user_id
    FROM mysql_app.user_center.users
    WHERE city = '上海'
) u ON e.user_id = u.user_id
WHERE e.dt = '2024-01-15';
```

**Join优化策略**：

```sql
-- 查看当前Join分布式策略
SHOW SESSION LIKE 'join_distribution_type';

-- 强制使用广播Join（当一侧表较小时）
SET SESSION join_distribution_type = 'BROADCAST';

-- 强制使用分区Hash Join（当两表都很大时）
SET SESSION join_distribution_type = 'PARTITIONED';

-- 自动选择（推荐，Trino根据统计信息自动决定）
SET SESSION join_distribution_type = 'AUTOMATIC';

-- ✅ 好的写法：小表放在JOIN的右侧（Build Side）
SELECT o.*, u.user_name
FROM hive.dw.order_facts o         -- 大表作为Probe Side
JOIN mysql_app.user_center.users u -- 小表作为Build Side (被广播)
    ON o.user_id = u.user_id
WHERE o.dt = '2024-01-15';

-- ❌ 差的写法：两个大表直接Join且无过滤条件
SELECT a.*, b.*
FROM hive.dw.user_events a         -- 10亿行
JOIN hive.dw.page_views b          -- 50亿行
    ON a.session_id = b.session_id; -- 两个大表Shuffle，极其耗资源

-- ✅ 改进：先过滤再Join
SELECT a.*, b.*
FROM (SELECT * FROM hive.dw.user_events WHERE dt = '2024-01-15') a
JOIN (SELECT * FROM hive.dw.page_views WHERE dt = '2024-01-15') b
    ON a.session_id = b.session_id;
```

**Join重排序**：

```sql
-- 开启基于代价的Join重排序（CBO需要表统计信息）
SET SESSION join_reordering_strategy = 'AUTOMATIC';

-- 手动收集表统计信息（Hive Connector）
ANALYZE hive.dw.order_facts;

-- 收集特定列的统计信息
ANALYZE hive.dw.order_facts
WITH (columns = ARRAY['user_id', 'amount', 'product_category']);
```

### 4.2 数据格式优化

数据文件格式对Trino查询性能有决定性影响。列式存储格式（ORC/Parquet）能显著提升查询效率。

**格式性能对比**：

| 维度 | ORC | Parquet | CSV/JSON | Avro |
|------|-----|---------|----------|------|
| **存储类型** | 列式 | 列式 | 行式文本 | 行式二进制 |
| **压缩率** | 极高 (ZLIB/ZSTD) | 极高 (Snappy/ZSTD) | 低 | 中等 |
| **列裁剪** | 支持 | 支持 | 不支持 | 不支持 |
| **谓词下推** | 支持 (Bloom Filter + Min/Max) | 支持 (Row Group Statistics) | 不支持 | 不支持 |
| **Trino读取速度** | 极快 | 极快 | 慢 | 中等 |
| **适用场景** | Hive生态首选 | 跨引擎通用 | 临时数据 | Kafka/流处理 |
| **嵌套类型支持** | 良好 | 优秀 | 有限 | 优秀 |

```sql
-- ✅ 高效：Parquet格式 + 谓词下推
-- Trino只读取需要的列，且利用Parquet的统计信息跳过不匹配的Row Group
SELECT user_id, amount, order_status
FROM hive.dw.order_facts       -- Parquet格式，按dt分区
WHERE dt = '2024-01-15'        -- 分区裁剪
  AND amount > 1000            -- 下推到Parquet Reader（利用Min/Max统计）
  AND order_status = 'completed'; -- 下推到Parquet Reader
-- 实际数据读取量可能只有全量数据的1%

-- ❌ 低效：CSV格式全表扫描
SELECT user_id, amount, order_status
FROM hive.ods.orders_csv       -- CSV格式
WHERE dt = '2024-01-15'
  AND amount > 1000;
-- CSV无法列裁剪，无法谓词下推，必须读取全部列全部行
-- 数据读取量 = 全量数据的100%

-- 创建ORC格式表（推荐用于Hive场景）
CREATE TABLE hive.dw.order_facts_orc
WITH (
    format = 'ORC',
    partitioned_by = ARRAY['dt'],
    orc_bloom_filter_columns = ARRAY['user_id', 'product_id'],
    orc_bloom_filter_fpp = 0.05
) AS
SELECT * FROM hive.ods.raw_orders;

-- 创建Parquet格式表（推荐用于跨引擎场景）
CREATE TABLE hive.dw.order_facts_parquet
WITH (
    format = 'PARQUET',
    partitioned_by = ARRAY['dt']
) AS
SELECT * FROM hive.ods.raw_orders;
```

### 4.3 内存管理

Trino是纯内存计算引擎，内存管理直接影响查询成功率和集群稳定性。

**内存池架构**：

```
Trino Worker内存结构
┌─────────────────────────────────────────────────────────┐
│                     JVM Heap (-Xmx16G)                  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │              User Memory Pool (用户内存)            │  │
│  │  - Hash表（Join/Aggregation）                       │  │
│  │  - 排序缓冲区                                       │  │
│  │  - 窗口函数缓冲区                                   │  │
│  │  受 query.max-memory-per-node 限制                  │  │
│  │  通常占 JVM Heap 的 60%-70%                         │  │
│  ├────────────────────────────────────────────────────┤  │
│  │              System Memory Pool (系统内存)           │  │
│  │  - 读取缓冲区（从Connector读取的数据页）            │  │
│  │  - 网络传输缓冲区                                   │  │
│  │  - Exchange缓冲区                                   │  │
│  │  通常占 JVM Heap 的 20%-30%                         │  │
│  ├────────────────────────────────────────────────────┤  │
│  │              Heap Headroom (堆内存预留)              │  │
│  │  - GC需要的工作空间                                  │  │
│  │  - Trino内部对象、线程栈等                          │  │
│  │  memory.heap-headroom-per-node 控制                  │  │
│  │  通常占 JVM Heap 的 10%-20%                         │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Spill-to-Disk配置（内存不足时溢写磁盘）**：

```properties
# etc/config.properties

# 开启Spill功能
spill-enabled=true

# 溢写文件路径（建议使用SSD）
spiller-spill-path=/opt/trino/data/spill

# 单个查询最大溢写空间
spiller-max-used-space-threshold=0.7

# 触发溢写的内存使用比例
# 当查询内存使用超过此比例时开始溢写
experimental.spill-threshold-per-operator=0.7

# 支持溢写的操作：
# - ORDER BY (排序)
# - Window Functions (窗口函数)
# - Aggregation (聚合)
# - Join (Hash Join)
```

**内存调优最佳实践**：

```properties
# 16GB JVM堆的推荐配置
# etc/config.properties

query.max-memory=50GB                     # 集群级别: Worker数 * 单Worker配额
query.max-memory-per-node=8GB             # 单Worker: JVM堆的50%
query.max-total-memory-per-node=10GB      # 含系统内存: JVM堆的62.5%
memory.heap-headroom-per-node=3GB         # 预留: JVM堆的18.75%

# 查询超时
query.max-execution-time=30m
query.max-run-time=30m

# 内存不足时的策略
query.low-memory-killer.policy=total-reservation-on-blocked-nodes
query.low-memory-killer.delay=5m
```

### 4.4 Connector优化

**Hive Connector优化**：

```properties
# etc/catalog/hive.properties

# Metastore缓存 — 减少对Hive Metastore的RPC调用
hive.metastore-cache-ttl=2m
hive.metastore-refresh-interval=1m
hive.per-transaction-metastore-cache-maximum-size=1000

# 文件列表缓存 — 避免每次查询都扫描HDFS目录
hive.file-status-cache-expire-time=10m
hive.file-status-cache-tables=dw.*,ads.*

# 文件列表并行度
hive.max-concurrent-file-renames=20
hive.max-initial-splits=200
hive.max-initial-split-size=32MB

# Split大小优化
hive.max-split-size=64MB
hive.max-splits-per-second=1000

# 数据格式优化
hive.parquet.use-column-names=true
hive.orc.use-column-names=true

# 动态过滤
hive.dynamic-filtering.wait-timeout=10s
```

**MySQL Connector优化**：

```properties
# etc/catalog/mysql.properties

# 连接池配置
connection-url=jdbc:mysql://mysql-host:3306?useSSL=false&serverTimezone=Asia/Shanghai
connection-user=trino_reader
connection-password=your_password

# 谓词下推优化
mysql.jdbc.pushdown.enabled=true

# 并行度
mysql.domain-compaction-threshold=500
```

**各Connector调优参数汇总**：

| Connector | 关键参数 | 默认值 | 推荐值 | 说明 |
|-----------|----------|--------|--------|------|
| **Hive** | `hive.max-split-size` | 64MB | 32MB-128MB | Split越小并行度越高，但调度开销增加 |
| **Hive** | `hive.metastore-cache-ttl` | 0 (禁用) | 1m-5m | 元数据缓存时间，减少Metastore压力 |
| **Hive** | `hive.max-initial-splits` | 200 | 100-500 | 初始Split数，影响查询启动速度 |
| **Hive** | `hive.dynamic-filtering.wait-timeout` | 0s | 5s-15s | 等待动态过滤条件的时间 |
| **MySQL** | `jdbc.connection-pool.max-size` | 30 | 20-50 | JDBC连接池大小 |
| **ES** | `elasticsearch.scroll-size` | 1000 | 1000-5000 | 每次scroll请求返回的文档数 |
| **ES** | `elasticsearch.request-timeout` | 10s | 30s-60s | ES请求超时时间 |
| **Kafka** | `kafka.messages-per-split` | 100000 | 50000-200000 | 每个Split包含的消息数 |
| **Iceberg** | `iceberg.max-partitions-per-writer` | 100 | 100-500 | 写入时最大分区数 |

---

## 5. 与其他查询引擎对比

### 5.1 Trino vs Hive

Hive和Trino虽然都可以查询HDFS/S3上的数据，但架构和适用场景完全不同。

| 维度 | Trino | Hive (on Tez/Spark) |
|------|-------|---------------------|
| **执行模型** | MPP内存Pipeline | DAG批处理（中间结果落盘） |
| **查询延迟** | 秒级 ~ 分钟级 | 分钟级 ~ 小时级 |
| **吞吐量** | 中等（受内存限制） | 高（可处理PB级数据） |
| **并发能力** | 高（适合多用户交互查询） | 低（资源占用大，适合ETL） |
| **SQL兼容性** | ANSI SQL标准，丰富 | HiveQL，部分ANSI SQL |
| **联邦查询** | 原生支持40+数据源 | 不支持（仅HDFS/S3） |
| **容错能力** | 弱（查询失败需重跑） | 强（Stage级别重试） |
| **资源使用** | 纯内存，对内存要求高 | CPU+磁盘+内存均衡 |
| **数据写入** | 有限（CTAS，部分Connector） | 完整（INSERT/UPDATE/MERGE） |
| **适用场景** | Ad-hoc查询、BI报表、数据探索 | ETL管道、大规模批处理 |
| **数据量建议** | 单查询 < 数百GB | 单查询可达TB/PB |
| **UDF支持** | Java SPI插件 | Java UDF/UDAF/UDTF |

**典型使用场景分工**：

```
数据处理流水线中的角色分工
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  数据源 ──→ [Hive ETL Pipeline] ──→ 数据仓库 ──→ [Trino查询]   │
│              (批处理、T+1)           (ORC/Parquet)   (交互式)     │
│                                                                  │
│  详细流程:                                                        │
│  ┌────────┐    ┌──────────────┐    ┌───────────┐    ┌─────────┐ │
│  │ ODS层   │ →  │  Hive ETL    │ →  │  DW/ADS层 │ →  │ Trino   │ │
│  │原始数据 │    │ (清洗/转换/  │    │ (聚合/建模│    │交互查询 │ │
│  │         │    │  加载/建模)  │    │ 后的结果) │    │BI报表   │ │
│  └────────┘    └──────────────┘    └───────────┘    └─────────┘ │
│                    ↑                                     ↑       │
│              适合Hive的场景              适合Trino的场景          │
│              - TB级数据清洗              - 秒级响应查询           │
│              - 复杂ETL转换              - 多数据源联合            │
│              - 数仓分层建设              - 数据探索分析           │
│                                         - 仪表盘查询             │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 Trino vs Spark SQL

| 维度 | Trino | Spark SQL |
|------|-------|-----------|
| **架构模型** | MPP (Massively Parallel Processing) | DAG (Directed Acyclic Graph) |
| **执行方式** | Pipeline流式处理 | Stage批处理（Shuffle写盘） |
| **进程模型** | 常驻进程，即时执行 | 需启动Executor，有冷启动延迟 |
| **查询延迟** | 毫秒级~分钟级 | 秒级~小时级（含启动时间） |
| **大数据处理** | 受内存限制，超大查询可能OOM | 支持Spill，可处理超内存数据集 |
| **容错能力** | 弱（查询级别重启） | 强（Stage级别重试，RDD Lineage） |
| **编程模型** | 纯SQL | SQL + DataFrame + Dataset + RDD |
| **ML/图计算** | 不支持 | MLlib, GraphX |
| **流处理** | 不支持 | Structured Streaming |
| **联邦查询** | 原生支持，核心能力 | 通过JDBC数据源支持，非核心 |
| **部署复杂度** | 简单（单个进程） | 较复杂（Driver + Executor + 资源管理） |
| **社区生态** | 查询引擎专注 | 大数据全栈（批/流/ML） |
| **适用场景** | 交互式查询、联邦查询 | ETL、ML、复杂数据处理 |

**选择建议**：
- 如果需求是**交互式SQL查询**、**多数据源联邦查询**、**BI报表** → 选Trino
- 如果需求是**复杂ETL**、**机器学习**、**流批一体** → 选Spark SQL
- 两者可以共存互补，Spark做ETL，Trino做查询

### 5.3 Trino vs ClickHouse/Doris

| 维度 | Trino | ClickHouse | Apache Doris |
|------|-------|------------|--------------|
| **定位** | 联邦查询引擎 | 列式OLAP数据库 | MPP分析型数据库 |
| **数据模型** | 无存储，连接外部源 | 自有MergeTree存储 | 自有列式存储 |
| **数据写入** | 有限（依赖Connector） | 高性能批量写入 | 实时/批量写入 |
| **单表查询** | 快（取决于数据源） | 极快（本地列存+向量化） | 很快（向量化+CBO） |
| **多表Join** | 强（MPP分布式Join） | 弱（分布式Join开销大） | 强（MPP + Colocation Join） |
| **联邦查询** | 核心能力（40+源） | 有限（外部表功能） | 有限（Multi-Catalog） |
| **实时更新** | 不适用 | 支持（MergeTree变种） | 支持（Unique Key模型） |
| **SQL兼容性** | 高（ANSI SQL） | 中（ClickHouse SQL方言） | 高（MySQL协议兼容） |
| **数据规模** | 无限（取决于数据源） | TB级本地数据 | TB级本地数据 |
| **运维复杂度** | 中（无状态，易扩缩容） | 较高（分片/副本管理） | 低（FE/BE自动管理） |

**引擎选型决策矩阵**：

```
查询引擎选型决策流程
┌──────────────────────────────────────────────────────────────────┐
│                    你的核心需求是什么？                            │
│                          │                                       │
│            ┌─────────────┼─────────────┐                        │
│            ↓             ↓             ↓                        │
│     ┌───────────┐  ┌──────────┐  ┌──────────────┐              │
│     │ 需要查询   │  │ 需要超高  │  │ 需要实时写入 │              │
│     │ 多个不同   │  │ 性能的单  │  │ + 快速查询   │              │
│     │ 数据源？   │  │ 数据源分  │  │ 的OLAP库？   │              │
│     └─────┬─────┘  │ 析查询？  │  └──────┬───────┘              │
│           │        └────┬─────┘         │                       │
│           ↓             ↓               ↓                       │
│     ┌──────────┐  ┌──────────┐   ┌──────────────┐              │
│     │  Trino   │  │ClickHouse│   │ Apache Doris │              │
│     │          │  │          │   │ / StarRocks  │              │
│     └──────────┘  └──────────┘   └──────────────┘              │
│                                                                  │
│  组合使用建议:                                                    │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  数据湖(Hive/Iceberg) ──→ Trino (联邦查询/探索)       │     │
│  │                       └──→ ClickHouse/Doris (高性能查询)│     │
│  │  业务DB(MySQL) ─────────→ Trino (跨源关联)             │     │
│  │  实时数据(Kafka) ───────→ Doris (实时导入+即时查询)     │     │
│  └────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. 实战案例：跨数据源联邦查询平台

### 6.1 需求分析

某电商公司的数据分布在多个系统中，分析师和业务人员需要跨数据源进行联合分析。
传统方案需要将所有数据ETL到一个数据仓库中，链路长、时效性差。
使用Trino构建联邦查询平台可以直接查询各数据源，实现T+0分析。

```
跨数据源联邦查询平台架构
┌──────────────────────────────────────────────────────────────────────────┐
│                             应用层                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  Apache      │  │  Spring Boot │  │  Jupyter     │                  │
│  │  Superset    │  │  API服务     │  │  Notebook    │                  │
│  │  (BI可视化)   │  │  (数据服务)   │  │  (数据探索)   │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         └─────────────────┼─────────────────┘                           │
│                           │ JDBC / REST API                             │
├───────────────────────────┼──────────────────────────────────────────────┤
│                           ↓                                              │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                    Trino集群 (联邦查询引擎)                        │  │
│  │                                                                    │  │
│  │  Coordinator ──→ Worker #1                                        │  │
│  │       │     └──→ Worker #2                                        │  │
│  │       │     └──→ Worker #3                                        │  │
│  │       │     └──→ Worker #N                                        │  │
│  │       │                                                            │  │
│  │  ┌────┴────────────────────────────────────────────────────────┐   │  │
│  │  │                    Catalog配置层                             │   │  │
│  │  │  ┌──────┐ ┌───────┐ ┌────┐ ┌──────┐ ┌───────┐ ┌────────┐ │   │  │
│  │  │  │ hive │ │iceberg│ │mysql│ │  pg  │ │  es   │ │ kafka  │ │   │  │
│  │  │  └──┬───┘ └───┬───┘ └──┬─┘ └──┬───┘ └───┬───┘ └───┬────┘ │   │  │
│  │  └─────┼─────────┼────────┼──────┼─────────┼─────────┼───────┘   │  │
│  └────────┼─────────┼────────┼──────┼─────────┼─────────┼────────────┘  │
│           ↓         ↓        ↓      ↓         ↓         ↓               │
├───────────┼─────────┼────────┼──────┼─────────┼─────────┼────────────────┤
│           ↓         ↓        ↓      ↓         ↓         ↓               │
│  ┌────────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ HDFS/S3    │ │Iceberg │ │  MySQL   │ │  Elastic │ │  Kafka   │      │
│  │ 数据湖     │ │ 湖仓   │ │  业务库  │ │  search  │ │  消息队列│      │
│  │            │ │        │ │ PostgreSQL│ │          │ │          │      │
│  │ - 行为日志 │ │- 近实时│ │ - 用户表 │ │ - 商品索引│ │ - 实时事件│      │
│  │ - 交易明细 │ │  订单  │ │ - 配置表 │ │ - 日志索引│ │ - 点击流 │      │
│  │ - 历史归档 │ │- 库存  │ │ - 字典表 │ │          │ │          │      │
│  └────────────┘ └────────┘ └──────────┘ └──────────┘ └──────────┘      │
│       数据湖         湖仓一体     业务数据库     搜索引擎      消息系统    │
└──────────────────────────────────────────────────────────────────────────┘
```

### 6.2 架构设计

**各数据源Catalog配置**：

```properties
# etc/catalog/hive_lake.properties — 数据湖（历史数据）
connector.name=hive
hive.metastore.uri=thrift://metastore:9083
hive.config.resources=/opt/trino/etc/core-site.xml,/opt/trino/etc/hdfs-site.xml
hive.max-partitions-per-scan=100000
hive.parquet.use-column-names=true
hive.metastore-cache-ttl=2m
hive.file-status-cache-expire-time=10m
```

```properties
# etc/catalog/iceberg_rt.properties — 湖仓一体（近实时数据）
connector.name=iceberg
iceberg.catalog.type=hive_metastore
hive.metastore.uri=thrift://metastore:9083
iceberg.file-format=PARQUET
iceberg.compression-codec=ZSTD
```

```properties
# etc/catalog/mysql_biz.properties — MySQL业务库
connector.name=mysql
connection-url=jdbc:mysql://mysql-master:3306?useSSL=false
connection-user=trino_reader
connection-password=secure_password_here
jdbc.connection-pool.max-size=30
```

```properties
# etc/catalog/pg_biz.properties — PostgreSQL业务库
connector.name=postgresql
connection-url=jdbc:postgresql://pg-host:5432/business
connection-user=trino_reader
connection-password=secure_password_here
jdbc.connection-pool.max-size=20
```

```properties
# etc/catalog/es_search.properties — Elasticsearch
connector.name=elasticsearch
elasticsearch.host=es-cluster
elasticsearch.port=9200
elasticsearch.default-schema-name=default
elasticsearch.scroll-size=5000
elasticsearch.request-timeout=30s
```

**Spring Boot数据服务接入示例**：

```java
// pom.xml 依赖
// <dependency>
//     <groupId>io.trino</groupId>
//     <artifactId>trino-jdbc</artifactId>
//     <version>433</version>
// </dependency>

// application.yml
// spring:
//   datasource:
//     trino:
//       url: jdbc:trino://coordinator-host:8080/hive_lake/dw
//       username: api_service
//       driver-class-name: io.trino.jdbc.TrinoDriver

import java.sql.*;
import java.util.Properties;

public class TrinoQueryService {

    private static final String TRINO_URL = "jdbc:trino://coordinator:8080";

    public void executeQuery(String sql) throws SQLException {
        Properties properties = new Properties();
        properties.setProperty("user", "api_service");
        properties.setProperty("source", "spring-boot-api");
        // 可选：设置Catalog和Schema
        properties.setProperty("catalog", "hive_lake");
        properties.setProperty("schema", "dw");
        // 可选：SSL配置
        // properties.setProperty("SSL", "true");
        // properties.setProperty("SSLTrustStorePath", "/path/to/truststore.jks");

        try (Connection connection = DriverManager.getConnection(TRINO_URL, properties);
             Statement statement = connection.createStatement();
             ResultSet resultSet = statement.executeQuery(sql)) {

            ResultSetMetaData metadata = resultSet.getMetaData();
            int columnCount = metadata.getColumnCount();

            while (resultSet.next()) {
                for (int i = 1; i <= columnCount; i++) {
                    System.out.printf("%s: %s  ",
                        metadata.getColumnName(i),
                        resultSet.getString(i));
                }
                System.out.println();
            }
        }
    }
}
```

**Python数据探索接入示例**：

```python
# pip install trino

from trino.dbapi import connect
from trino.auth import BasicAuthentication
import pandas as pd

def query_trino(sql, catalog="hive_lake", schema="dw"):
    """执行Trino查询并返回DataFrame"""
    conn = connect(
        host="coordinator-host",
        port=8080,
        user="data_analyst",
        catalog=catalog,
        schema=schema,
        source="jupyter-notebook",
        # 可选：认证配置
        # auth=BasicAuthentication("user", "password"),
        # http_scheme="https",
    )
    cursor = conn.cursor()
    cursor.execute(sql)

    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return pd.DataFrame(rows, columns=columns)


# 使用示例：跨数据源联邦查询
df = query_trino("""
    SELECT
        u.user_id,
        u.user_name,
        u.city,
        COUNT(DISTINCT e.session_id) AS sessions,
        SUM(CASE WHEN e.event_type = 'purchase' THEN 1 ELSE 0 END) AS purchases
    FROM hive_lake.dw.user_events e
    JOIN mysql_biz.user_center.users u ON e.user_id = u.user_id
    WHERE e.dt = '2024-01-15'
    GROUP BY u.user_id, u.user_name, u.city
    ORDER BY purchases DESC
    LIMIT 100
""")

print(df.describe())
```

### 6.3 典型查询场景

**场景一：用户分析 — 行为日志(Hive) + 用户画像(MySQL)**

```sql
-- 需求：分析不同渠道注册用户的留存和消费行为
-- 数据源：
--   hive_lake.dw.user_events — Hive上的用户行为事件（日志量大，按天分区）
--   mysql_biz.user_center.users — MySQL中的用户注册信息
--   mysql_biz.user_center.user_profiles — MySQL中的用户画像标签

WITH user_cohort AS (
    -- Step 1: 从MySQL获取目标用户群（1月新注册用户）
    SELECT
        u.user_id,
        u.user_name,
        u.register_channel,
        u.created_at AS register_date,
        p.age_group,
        p.gender
    FROM mysql_biz.user_center.users u
    LEFT JOIN mysql_biz.user_center.user_profiles p
        ON u.user_id = p.user_id
    WHERE u.created_at BETWEEN DATE '2024-01-01' AND DATE '2024-01-31'
      AND u.status = 'active'
),
user_behavior AS (
    -- Step 2: 从Hive获取这些用户的行为数据
    SELECT
        user_id,
        COUNT(DISTINCT dt) AS active_days,
        COUNT(DISTINCT session_id) AS total_sessions,
        COUNT(*) AS total_events,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchase_count,
        SUM(CASE WHEN event_type = 'add_to_cart' THEN 1 ELSE 0 END) AS cart_count,
        MIN(dt) AS first_active_date,
        MAX(dt) AS last_active_date
    FROM hive_lake.dw.user_events
    WHERE dt BETWEEN '2024-01-01' AND '2024-02-29'
    GROUP BY user_id
)
SELECT
    uc.register_channel,
    uc.age_group,
    COUNT(*) AS user_count,
    AVG(ub.active_days) AS avg_active_days,
    AVG(ub.total_sessions) AS avg_sessions,
    SUM(CASE WHEN ub.active_days >= 7 THEN 1 ELSE 0 END) * 100.0
        / COUNT(*) AS seven_day_retention_pct,
    SUM(ub.purchase_count) AS total_purchases,
    SUM(ub.purchase_count) * 100.0 / NULLIF(SUM(ub.cart_count), 0)
        AS cart_to_purchase_rate
FROM user_cohort uc
LEFT JOIN user_behavior ub ON uc.user_id = ub.user_id
GROUP BY uc.register_channel, uc.age_group
ORDER BY user_count DESC;
```

**场景二：搜索增强 — 全文检索(ES) + 业务数据(Hive)**

```sql
-- 需求：商品搜索结果中融合销售数据和评价信息
-- 数据源：
--   es_search.default.product_index — Elasticsearch商品全文索引
--   hive_lake.dw.product_sales_30d — Hive上的近30天销售统计
--   mysql_biz.product_service.product_reviews_summary — MySQL评价汇总

SELECT
    es.product_id,
    es.product_name,
    es.category,
    es.brand,
    es._score AS relevance_score,
    COALESCE(sales.total_quantity, 0) AS sold_30d,
    COALESCE(sales.total_revenue, 0) AS revenue_30d,
    COALESCE(reviews.avg_rating, 0) AS avg_rating,
    COALESCE(reviews.review_count, 0) AS review_count,
    -- 综合排序分 = 搜索相关性 * 销量权重 * 评分权重
    es._score
        * (1 + LN(COALESCE(sales.total_quantity, 0) + 1))
        * (COALESCE(reviews.avg_rating, 3.0) / 5.0)
    AS composite_score
FROM es_search.default.product_index es
LEFT JOIN hive_lake.dw.product_sales_30d sales
    ON es.product_id = sales.product_id
LEFT JOIN mysql_biz.product_service.product_reviews_summary reviews
    ON es.product_id = reviews.product_id
WHERE es.query = 'keyword:无线耳机 AND status:在售'
ORDER BY composite_score DESC
LIMIT 20;
```

**场景三：实时仪表盘 — 多数据源聚合**

```sql
-- 需求：运营实时大盘，汇聚多个数据源的核心指标
-- 数据源：
--   iceberg_rt.ods.orders_realtime — Iceberg近实时订单（分钟级延迟）
--   hive_lake.dw.order_facts — Hive历史订单（T+1）
--   mysql_biz.user_center.users — MySQL用户总量

WITH today_metrics AS (
    -- 今日实时指标（来自Iceberg近实时表）
    SELECT
        COUNT(*) AS today_orders,
        COUNT(DISTINCT user_id) AS today_buyers,
        SUM(amount) AS today_gmv,
        AVG(amount) AS today_avg_order_value,
        SUM(CASE WHEN order_status = 'completed' THEN amount ELSE 0 END) AS today_revenue
    FROM iceberg_rt.ods.orders_realtime
    WHERE order_time >= CURRENT_DATE
),
yesterday_metrics AS (
    -- 昨日指标（来自Hive T+1数据）
    SELECT
        COUNT(*) AS yesterday_orders,
        COUNT(DISTINCT user_id) AS yesterday_buyers,
        SUM(amount) AS yesterday_gmv
    FROM hive_lake.dw.order_facts
    WHERE dt = CAST(CURRENT_DATE - INTERVAL '1' DAY AS VARCHAR)
),
last_week_metrics AS (
    -- 上周同期指标（来自Hive历史数据）
    SELECT
        COUNT(*) AS last_week_orders,
        SUM(amount) AS last_week_gmv
    FROM hive_lake.dw.order_facts
    WHERE dt = CAST(CURRENT_DATE - INTERVAL '7' DAY AS VARCHAR)
),
user_metrics AS (
    -- 用户总量（来自MySQL）
    SELECT
        COUNT(*) AS total_users,
        SUM(CASE WHEN created_at >= CURRENT_DATE THEN 1 ELSE 0 END) AS today_new_users
    FROM mysql_biz.user_center.users
)
SELECT
    -- 今日指标
    t.today_orders,
    t.today_buyers,
    t.today_gmv,
    t.today_avg_order_value,
    t.today_revenue,
    -- 环比（vs昨日）
    ROUND((t.today_gmv - y.yesterday_gmv) * 100.0
        / NULLIF(y.yesterday_gmv, 0), 2) AS gmv_dod_pct,
    ROUND((t.today_orders - y.yesterday_orders) * 100.0
        / NULLIF(y.yesterday_orders, 0), 2) AS orders_dod_pct,
    -- 同比（vs上周同期）
    ROUND((t.today_gmv - lw.last_week_gmv) * 100.0
        / NULLIF(lw.last_week_gmv, 0), 2) AS gmv_wow_pct,
    -- 用户指标
    u.total_users,
    u.today_new_users
FROM today_metrics t
CROSS JOIN yesterday_metrics y
CROSS JOIN last_week_metrics lw
CROSS JOIN user_metrics u;
```

---

## 7. 运维与最佳实践

### 7.1 监控

**Web UI监控**：

Trino自带Web UI，默认端口与HTTP端口相同（通常为8080）。通过浏览器访问
`http://coordinator-host:8080` 可以查看集群状态、正在执行的查询、历史查询等信息。

**关键监控指标**：

| 指标分类 | 指标名称 | 含义 | 告警阈值建议 |
|----------|----------|------|--------------|
| **集群** | Active Workers | 活跃Worker节点数 | 低于预期节点数时告警 |
| **集群** | Running Queries | 正在执行的查询数 | > 并发上限的80%时预警 |
| **集群** | Queued Queries | 排队等待的查询数 | > 50时告警 |
| **集群** | Blocked Queries | 被阻塞的查询数 | > 0持续5分钟告警 |
| **性能** | Query Throughput | 每分钟完成查询数 | 低于基线50%时告警 |
| **性能** | P50 Latency | 查询延迟中位数 | > 10s（取决于业务场景） |
| **性能** | P99 Latency | 查询延迟99分位 | > 60s |
| **内存** | Cluster Memory Usage | 集群总内存使用率 | > 85% 预警，> 95% 告警 |
| **内存** | Per-Node Memory Usage | 单节点内存使用率 | > 90% 告警 |
| **错误** | Failed Queries/min | 每分钟失败查询数 | > 5 告警 |
| **错误** | User Error Rate | 用户错误占比 | 区分用户错误和系统错误 |

**JMX指标采集**：

```properties
# etc/catalog/jmx.properties
connector.name=jmx
```

```sql
-- 通过JMX Connector查询Trino内部指标
-- 查看所有可用的JMX MBean
SELECT * FROM jmx.current."java.lang:type=memory";

-- 查询查询执行统计
SELECT *
FROM jmx.current."trino.execution:name=querymanager";

-- 查询内存使用情况
SELECT *
FROM jmx.current."trino.memory:type=clustermemorymanager,name=clustermemorymanager";
```

**Prometheus + Grafana监控集成**：

```bash
# 下载JMX Exporter
wget https://repo1.maven.org/maven2/io/prometheus/jmx/jmx_prometheus_javaagent/0.19.0/jmx_prometheus_javaagent-0.19.0.jar \
  -O /opt/trino/lib/jmx_exporter.jar
```

```properties
# etc/jvm.config 中添加JMX Exporter
-javaagent:/opt/trino/lib/jmx_exporter.jar=9090:/opt/trino/etc/jmx_exporter_config.yml
```

```yaml
# etc/jmx_exporter_config.yml
---
lowercaseOutputName: true
lowercaseOutputLabelNames: true
rules:
  - pattern: "trino.execution<name=QueryManager><>(running_queries|queued_queries|failed_queries_total|completed_queries_total)"
    name: "trino_$1"
    type: GAUGE
  - pattern: "trino.memory<type=ClusterMemoryManager, name=ClusterMemoryManager><>(cluster_memory_bytes|free_memory_bytes)"
    name: "trino_memory_$1"
    type: GAUGE
  - pattern: "java.lang<type=Memory><HeapMemoryUsage>(used|max|committed)"
    name: "jvm_heap_memory_$1_bytes"
    type: GAUGE
```

### 7.2 故障排查

**常见错误与解决方案**：

| 错误码/信息 | 原因分析 | 解决方案 |
|-------------|----------|----------|
| `EXCEEDED_GLOBAL_MEMORY_LIMIT` | 单个查询超出集群总内存限制 `query.max-memory` | 1. 增加集群内存或Worker数量 2. 优化查询减少数据量 3. 调大 `query.max-memory` |
| `EXCEEDED_LOCAL_MEMORY_LIMIT` | 单个查询超出单Worker内存限制 | 1. 调大 `query.max-memory-per-node` 2. 开启Spill-to-disk 3. 减少Join/聚合数据量 |
| `NO_NODES_AVAILABLE` | 无可用Worker节点 | 1. 检查Worker进程是否存活 2. 检查Worker到Coordinator的网络 3. 检查Discovery Service |
| `REMOTE_TASK_ERROR` | Worker执行Task时出错 | 1. 检查Worker日志 2. 可能是数据格式问题 3. 可能是连接器配置错误 |
| `HIVE_METASTORE_ERROR` | 无法连接Hive Metastore | 1. 检查Metastore服务状态 2. 验证 `hive.metastore.uri` 配置 3. 检查防火墙 |
| `HIVE_PARTITION_SCHEMA_MISMATCH` | 分区Schema不一致 | 1. 修复分区Schema 2. 设置 `hive.parquet.use-column-names=true` |
| `GENERIC_INTERNAL_ERROR` | 通用内部错误 | 1. 查看Coordinator/Worker完整日志 2. 检查查询计划 3. 尝试简化查询 |
| `QUERY_EXPIRED` / `ABANDONED_QUERY` | 查询超时被终止 | 1. 优化查询 2. 调大 `query.max-execution-time` |
| `TOO_MANY_REQUESTS_FAILED` | Connector请求失败过多 | 1. 检查数据源状态 2. 增加重试配置 3. 降低并行度 |
| `JDBC_ERROR` | JDBC连接器通信错误 | 1. 检查数据库连接 2. 增加连接池大小 3. 检查网络超时 |
| `CORRUPT_PAGE` | 数据页损坏 | 1. 检查数据文件完整性 2. 重新生成数据文件 |
| `INSUFFICIENT_RESOURCES` | 集群资源不足以启动查询 | 1. 等待其他查询结束 2. 增加集群资源 3. 调整Resource Group |

**日志分析**：

```bash
# Coordinator日志位置
tail -f /opt/trino/data/var/log/server.log

# 查看特定查询的执行日志
# 方法1: 通过Web UI查看查询详情页面
# 方法2: 通过system表查询
```

```sql
-- 查询最近失败的查询
SELECT
    query_id,
    state,
    error_code,
    error_type,
    LEFT(query, 200) AS query_preview,
    created,
    "end",
    DATE_DIFF('second', created, "end") AS duration_seconds
FROM system.runtime.queries
WHERE state = 'FAILED'
ORDER BY created DESC
LIMIT 20;

-- 查看正在运行的查询及其资源使用
SELECT
    query_id,
    state,
    LEFT(query, 100) AS query_preview,
    user,
    source,
    cumulative_memory / 1024 / 1024 / 1024 AS memory_gb,
    total_cpu_time_seconds,
    created
FROM system.runtime.queries
WHERE state = 'RUNNING'
ORDER BY cumulative_memory DESC;

-- 查看Worker节点状态
SELECT
    node_id,
    http_uri,
    node_version,
    state,
    coordinator
FROM system.runtime.nodes;
```

### 7.3 最佳实践检查清单

以下是Trino生产环境部署和使用的完整检查清单：

**数据格式与存储**：

- [ ] 使用列式存储格式（ORC或Parquet），不使用CSV/JSON作为分析表格式
- [ ] 对大表启用压缩（推荐ZSTD或Snappy）
- [ ] ORC/Parquet文件大小控制在128MB-512MB之间（避免过多小文件）
- [ ] Hive表设置 `hive.parquet.use-column-names=true` 防止列顺序变更导致的问题

**分区设计**：

- [ ] 大表（>1亿行）必须建立分区，优先使用日期分区
- [ ] 分区粒度适中（避免过多分区，单表分区数不超过10万）
- [ ] 查询SQL中必须包含分区过滤条件
- [ ] 分区列不要使用函数包裹（防止分区裁剪失效）

**Catalog与Connector**：

- [ ] 每个Connector使用独立的只读账号连接数据源
- [ ] JDBC连接器配置连接池大小（避免数据库连接耗尽）
- [ ] Hive Connector开启Metastore缓存和文件列表缓存
- [ ] Elasticsearch Connector设置合理的scroll-size和request-timeout
- [ ] 敏感信息（密码）使用Secret管理，不在配置文件中明文存储

**内存与资源**：

- [ ] JVM堆大小设置为物理内存的70%-80%
- [ ] `query.max-memory-per-node` 设置为JVM堆的40%-60%
- [ ] 生产环境开启Spill-to-disk功能，Spill路径使用SSD
- [ ] 配置Resource Groups限制不同用户/场景的资源使用
- [ ] 设置 `query.max-execution-time` 防止查询无限运行

**安全**：

- [ ] 生产环境开启HTTPS（TLS加密传输）
- [ ] 配置认证机制（LDAP/Kerberos/OAuth2）
- [ ] 配置授权机制（System Access Control或Ranger集成）
- [ ] Coordinator Web UI限制访问（或开启认证）
- [ ] 审计日志开启，记录所有查询执行记录

**查询编写规范**：

```sql
-- ✅ 好的查询实践

-- 1. 始终指定分区条件
SELECT * FROM hive_lake.dw.user_events
WHERE dt = '2024-01-15';             -- 明确分区条件

-- 2. 只查询需要的列（避免 SELECT *）
SELECT user_id, event_type, event_time
FROM hive_lake.dw.user_events
WHERE dt = '2024-01-15';

-- 3. 小表放在Join右侧
SELECT o.*, u.user_name
FROM hive_lake.dw.order_facts o      -- 大表
JOIN mysql_biz.user_center.users u   -- 小表
    ON o.user_id = u.user_id;

-- 4. 使用approx_distinct替代COUNT(DISTINCT)（对精度要求不高时）
SELECT dt, approx_distinct(user_id) AS uv
FROM hive_lake.dw.user_events
WHERE dt >= '2024-01-01'
GROUP BY dt;

-- 5. 使用LIMIT限制返回结果数
SELECT * FROM large_table LIMIT 1000;


-- ❌ 差的查询实践

-- 1. 不带分区条件全表扫描
SELECT * FROM hive_lake.dw.user_events;  -- 可能扫描TB级数据

-- 2. SELECT * 不指定列
SELECT * FROM hive_lake.dw.wide_table;   -- 可能有几百列

-- 3. 对分区列使用函数
SELECT * FROM hive_lake.dw.user_events
WHERE SUBSTR(dt, 1, 7) = '2024-01';     -- 分区裁剪失效

-- 4. 无限制的CROSS JOIN
SELECT * FROM table_a CROSS JOIN table_b; -- 笛卡尔积爆炸

-- 5. 在WHERE中对大表列使用不可下推的函数
SELECT * FROM mysql_biz.user_center.users
WHERE MD5(email) = 'abc123...';           -- 无法下推到MySQL
```

**运维检查**：

- [ ] 配置监控告警（Prometheus + Grafana）
- [ ] 定期检查慢查询日志（通过system.runtime.queries表）
- [ ] 定期收集表统计信息（ANALYZE语句）提升CBO优化效果
- [ ] 制定Trino版本升级计划（建议每季度跟进社区版本）
- [ ] 准备故障恢复预案（Coordinator单点需做好高可用或快速恢复方案）
- [ ] 文档化Catalog配置和Resource Group策略
- [ ] Worker节点配置相同的硬件规格，避免数据倾斜导致木桶效应
