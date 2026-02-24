# 数据湖技术：Iceberg、Delta Lake与Hudi实战

## 1. 数据湖概述

### 1.1 数据湖vs数据仓库

数据湖（Data Lake）是一种以原始格式存储海量数据的架构，支持结构化、半结构化和非结构化数据。
与传统数据仓库不同，数据湖采用Schema-on-Read策略，数据写入时无需预定义模式。

近年来，**Data Lakehouse** 架构融合了两者优势，在数据湖之上提供数据仓库级别的事务保障和治理能力。

```
数据架构演进

  Data Warehouse            Data Lake              Data Lakehouse
  ┌─────────────┐      ┌─────────────┐      ┌──────────────────────┐
  │  BI / SQL   │      │  ML / SQL   │      │  BI / ML / SQL / API │
  ├─────────────┤      ├─────────────┤      ├──────────────────────┤
  │  Schema-on  │      │  Schema-on  │      │  Table Format Layer  │
  │  -Write     │      │  -Read      │      │  (Iceberg/Delta/Hudi)│
  ├─────────────┤      ├─────────────┤      ├──────────────────────┤
  │  Proprietary│      │  Raw Files  │      │  Open File Formats   │
  │  Storage    │      │  (S3/HDFS)  │      │  (Parquet/ORC/Avro)  │
  └─────────────┘      └─────────────┘      ├──────────────────────┤
                                             │  Object Storage      │
                                             │  (S3/ADLS/GCS/HDFS)  │
                                             └──────────────────────┘
```

**三种架构全面对比**：

| 对比维度 | Data Warehouse | Data Lake | Data Lakehouse |
|---------|---------------|-----------|----------------|
| **存储格式** | 专有格式 | 开放格式（Parquet/ORC） | 开放格式 + 表格式 |
| **Schema策略** | Schema-on-Write | Schema-on-Read | 两者兼备 |
| **ACID事务** | ✅ 完整支持 | ❌ 不支持 | ✅ 完整支持 |
| **存储成本** | 高（专用存储） | 低（对象存储） | 低（对象存储） |
| **数据灵活性** | 低（需预定义） | 高（任意格式） | 高（带治理） |
| **数据治理** | ✅ 强治理 | ❌ 弱治理（易成"数据沼泽"） | ✅ 强治理 |
| **查询性能** | 极高 | 中等 | 高（接近数仓） |
| **并发控制** | ✅ 支持 | ❌ 不支持 | ✅ 支持 |
| **Time Travel** | 有限支持 | ❌ 不支持 | ✅ 完整支持 |
| **代表产品** | Teradata, Redshift | HDFS + Hive | Iceberg/Delta/Hudi |

### 1.2 数据湖挑战

传统数据湖（如HDFS + Hive）面临五大核心挑战：

**1) Schema Evolution（模式演进）**

业务变更时需要增删改字段。传统Hive表修改Schema极为不便，往往需要重写全部数据。

```
❌ 传统Hive的Schema演进问题

-- 添加字段只能追加在最后
ALTER TABLE orders ADD COLUMNS (discount DOUBLE);

-- 无法安全地：
--   删除字段
--   重命名字段
--   调整字段顺序
--   修改字段类型（int→long）

-- 修改分区字段更是灾难性操作，需要重建整张表
```

**2) ACID Transactions（事务支持）**

数据湖缺乏原子性保障，并发写入可能导致数据不一致。

```
❌ 无ACID的问题场景

Writer A: 正在写入 part-00001.parquet ... part-00100.parquet
Writer B: 同时读取该目录
Reader:   读到了 part-00001 ~ part-00050（不完整数据）

-- 更严重的情况：
Writer A: 写入过程中失败
结果:     目录中存在部分文件（脏数据），无法自动回滚
```

**3) Time Travel（时间旅行）**

传统数据湖无法查询历史版本数据，误操作后无法回滚。

**4) Partition Evolution（分区演进）**

Hive表的分区方案一旦确定，修改分区需要完全重写数据。例如从按天分区改为按小时分区，
对PB级数据来说是不可接受的。

**5) Data Swamp（数据沼泽）问题**

```
数据沼泽形成过程

    健康数据湖                         数据沼泽
  ┌─────────────┐                 ┌─────────────┐
  │ 有目录结构   │                 │ 无人维护     │
  │ 有数据质量   │   缺乏治理      │ 数据重复     │
  │ 有访问控制   │ ──────────→    │ Schema混乱   │
  │ 有文档说明   │                 │ 无法理解     │
  │ 有生命周期   │                 │ 存储浪费     │
  └─────────────┘                 └─────────────┘
```

### 1.3 三大表格式对比

Apache Iceberg、Delta Lake和Apache Hudi是当前最主流的三大数据湖表格式（Table Format），
它们都为数据湖带来了ACID事务、Schema演进、Time Travel等数仓级别的能力。

| 对比维度 | Apache Iceberg | Delta Lake | Apache Hudi |
|---------|---------------|------------|-------------|
| **起源** | Netflix（2017开源） | Databricks（2019开源） | Uber（2019进入Apache） |
| **治理方** | Apache基金会 | Linux基金会（2024年迁移） | Apache基金会 |
| **Spark支持** | ✅ 完整 | ✅ 原生最优 | ✅ 完整 |
| **Flink支持** | ✅ 完整 | ✅ 支持（Flink/Delta Connector） | ✅ 完整 |
| **Trino/Presto** | ✅ 原生最优 | ✅ 支持 | ✅ 支持 |
| **Hive支持** | ✅ 支持 | ❌ 有限 | ✅ 支持 |
| **ACID事务** | ✅ Snapshot Isolation | ✅ Serializable | ✅ Snapshot Isolation |
| **Schema演进** | ✅ 完整（按ID追踪） | ✅ 完整（按名称追踪） | ✅ 完整 |
| **分区演进** | ✅ 原生支持（无需重写） | ❌ 不支持 | ❌ 有限支持 |
| **隐式分区** | ✅ 支持 | ❌ 不支持 | ❌ 不支持 |
| **Time Travel** | ✅ 快照和时间戳 | ✅ 版本和时间戳 | ✅ 时间线 |
| **流式读写** | ✅ 支持 | ✅ 支持 | ✅ 原生最优（增量Pull） |
| **Upsert性能** | 中等 | 中等 | ✅ 最优（Record-level索引） |
| **社区活跃度** | 极高（增长最快） | 高（Databricks主导） | 高 |
| **云厂商支持** | AWS/GCP/Azure全覆盖 | Azure/Databricks原生 | AWS EMR原生 |

```
三大表格式定位

  写入频率高 / 近实时Upsert          通用数据湖 / 分析优先
  ┌─────────────────┐              ┌─────────────────┐
  │   Apache Hudi   │              │ Apache Iceberg  │
  │                 │              │                 │
  │  CDC / 流式写入  │              │  大规模分析查询   │
  │  增量ETL        │              │  分区演进        │
  │  近实时数仓      │              │  多引擎互操作     │
  └─────────────────┘              └─────────────────┘

             Databricks生态 / 统一平台
             ┌─────────────────┐
             │   Delta Lake    │
             │                 │
             │  Databricks深度  │
             │  集成            │
             │  Unity Catalog  │
             └─────────────────┘
```

---

## 2. Apache Iceberg

### 2.1 架构

Apache Iceberg采用三层架构设计：Catalog层、Metadata层和Data层。
这种分层设计实现了元数据与数据的解耦，支持高效的快照隔离和原子提交。

```
Apache Iceberg 架构

┌──────────────────────────────────────────────────────────┐
│                     Catalog Layer                        │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│   │   Hive   │  │   REST   │  │ AWS Glue │  │ Nessie │ │
│   │ Metastore│  │  Catalog │  │  Catalog │  │Catalog │ │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬────┘ │
│        │              │              │             │      │
│        └──────────────┴──────┬───────┴─────────────┘      │
│                              │                            │
│                   指向当前 metadata.json                   │
└──────────────────────────────┼────────────────────────────┘
                               │
┌──────────────────────────────┼────────────────────────────┐
│                     Metadata Layer                        │
│                              ↓                            │
│              ┌───────────────────────────┐                │
│              │     metadata.json (v3)    │                │
│              │  ┌─────────────────────┐  │                │
│              │  │ table schema        │  │                │
│              │  │ partition spec      │  │                │
│              │  │ snapshot list       │  │                │
│              │  │ current-snapshot-id │  │                │
│              │  └─────────────────────┘  │                │
│              └─────────────┬─────────────┘                │
│                            │                              │
│                            ↓                              │
│              ┌───────────────────────────┐                │
│              │    Manifest List          │                │
│              │  (snap-xxx-m0.avro)       │                │
│              │  ┌─────────────────────┐  │                │
│              │  │ manifest-path-1     │  │                │
│              │  │ manifest-path-2     │  │                │
│              │  │ partition-summary   │  │                │
│              │  └─────────────────────┘  │                │
│              └─────────────┬─────────────┘                │
│                     ┌──────┴──────┐                       │
│                     ↓             ↓                       │
│          ┌──────────────┐ ┌──────────────┐                │
│          │ Manifest File│ │ Manifest File│                │
│          │  (m0.avro)   │ │  (m1.avro)   │                │
│          │ ┌──────────┐ │ │ ┌──────────┐ │                │
│          │ │file-path │ │ │ │file-path │ │                │
│          │ │partition  │ │ │ │partition  │ │                │
│          │ │col stats  │ │ │ │col stats  │ │                │
│          │ │row count  │ │ │ │row count  │ │                │
│          │ └──────────┘ │ │ └──────────┘ │                │
│          └──────────────┘ └──────────────┘                │
└──────────────────────────────┼────────────────────────────┘
                               │
┌──────────────────────────────┼────────────────────────────┐
│                      Data Layer                           │
│                     ┌────┴────┐                           │
│                     ↓         ↓                           │
│           ┌──────────────┐ ┌──────────────┐               │
│           │ data-001.    │ │ data-002.    │               │
│           │ parquet      │ │ parquet      │    ...        │
│           └──────────────┘ └──────────────┘               │
│                                                           │
│   支持格式: Apache Parquet / ORC / Avro                    │
└───────────────────────────────────────────────────────────┘
```

**Snapshot Isolation（快照隔离）机制**：

每次写入操作都会创建一个新的Snapshot，包含一个新的Manifest List，
指向变更和未变更的Manifest Files。读取操作始终基于某个完整的Snapshot，
因此读写之间互不干扰。

```
快照隔离示意

  Snapshot S1 (current)              Snapshot S2 (new write)
  ┌───────────────────┐             ┌───────────────────┐
  │ manifest-list-s1  │             │ manifest-list-s2  │
  │  ├─ manifest-A ───┼─ file1.pq  │  ├─ manifest-A ───┼─ file1.pq (复用)
  │  └─ manifest-B ───┼─ file2.pq  │  ├─ manifest-B ───┼─ file2.pq (复用)
  └───────────────────┘             │  └─ manifest-C ───┼─ file3.pq (新增)
                                    └───────────────────┘
  Reader读取S1: 看到 file1 + file2
  Writer提交S2: 原子性地更新 current-snapshot → S2
  新Reader读S2: 看到 file1 + file2 + file3
```

### 2.2 核心特性

**1) Schema Evolution（模式演进）**

Iceberg使用唯一的列ID追踪每一列，而非依赖列名或位置。这意味着可以安全地进行以下操作：

- **ADD** - 添加新列（在任意位置）
- **DROP** - 删除列（不影响已有数据文件）
- **RENAME** - 重命名列（通过ID关联，旧文件无需改变）
- **REORDER** - 调整列顺序
- **UPDATE** - 修改类型（支持安全的类型提升，如int→long、float→double）

```sql
-- ✅ Iceberg Schema Evolution 示例

-- 添加新列
ALTER TABLE db.orders ADD COLUMN discount DOUBLE AFTER price;

-- 重命名列（底层按ID追踪，无需重写数据）
ALTER TABLE db.orders RENAME COLUMN cust_name TO customer_name;

-- 删除列（惰性删除，旧Parquet文件不变）
ALTER TABLE db.orders DROP COLUMN deprecated_field;

-- 调整列顺序
ALTER TABLE db.orders ALTER COLUMN email AFTER customer_name;

-- 安全类型提升
ALTER TABLE db.orders ALTER COLUMN quantity TYPE BIGINT;  -- int → bigint
```

```
❌ 不安全的类型修改（Iceberg会拒绝）

ALTER TABLE db.orders ALTER COLUMN price TYPE INT;  -- double → int (精度丢失)
-- 错误: Cannot change column type: double is not compatible with int
```

**2) Partition Evolution（分区演进）**

这是Iceberg最独特的功能之一。可以在不重写任何历史数据的情况下更改分区方案。

```sql
-- 初始分区：按天分区
CREATE TABLE db.events (
    event_id   BIGINT,
    event_time TIMESTAMP,
    event_type STRING,
    user_id    BIGINT,
    payload    STRING
) USING iceberg
PARTITIONED BY (days(event_time));

-- 数据量增长后，改为按小时分区（零数据重写！）
ALTER TABLE db.events ADD PARTITION FIELD hours(event_time);
ALTER TABLE db.events DROP PARTITION FIELD days(event_time);

-- 查询引擎会自动：
-- 对旧数据使用 day 分区进行裁剪
-- 对新数据使用 hour 分区进行裁剪
```

```
分区演进原理

  旧数据 (day分区)                 新数据 (hour分区)
  ┌──────────────────┐            ┌──────────────────┐
  │ event_time_day=  │            │ event_time_hour=  │
  │  2024-01-01/     │            │  2024-06-15-08/   │
  │    data.parquet  │            │    data.parquet   │
  │  2024-01-02/     │            │  2024-06-15-09/   │
  │    data.parquet  │            │    data.parquet   │
  └──────────────────┘            └──────────────────┘

  查询: WHERE event_time = '2024-06-15 09:30:00'
  → 引擎自动选择 hour 分区裁剪新数据，day 分区裁剪旧数据
```

**3) Hidden Partitioning（隐式分区）**

用户查询时不需要知道分区列和分区函数，Iceberg自动处理分区裁剪。

```sql
-- 建表时指定分区转换
CREATE TABLE db.logs (
    log_time TIMESTAMP,
    level    STRING,
    message  STRING
) USING iceberg
PARTITIONED BY (hours(log_time), bucket(16, level));

-- ✅ 用户查询无需关心分区结构，直接过滤即可
SELECT * FROM db.logs
WHERE log_time BETWEEN '2024-06-15 08:00:00' AND '2024-06-15 10:00:00'
  AND level = 'ERROR';
-- Iceberg自动将 log_time 过滤条件映射到 hours 分区
-- 自动将 level = 'ERROR' 映射到 bucket 分区
```

```
❌ Hive分区查询 — 用户必须知道分区列

-- Hive要求显式指定分区列
SELECT * FROM logs
WHERE log_date = '2024-06-15'      -- 必须用分区列名
  AND log_hour IN (8, 9, 10);      -- 必须拆解时间到分区列

-- 如果忘记写分区条件 → 全表扫描
```

**4) Time Travel（时间旅行）**

```sql
-- 按时间戳查询历史版本
SELECT * FROM db.orders TIMESTAMP AS OF '2024-06-01 00:00:00';

-- 按快照ID查询
SELECT * FROM db.orders VERSION AS OF 5678901234;

-- 查看快照历史
SELECT * FROM db.orders.snapshots;

-- 查看所有历史变更
SELECT * FROM db.orders.history;
```

### 2.3 操作实战

**创建和操作Iceberg表**：

```sql
-- 创建Iceberg表
CREATE TABLE lakehouse.db.orders (
    order_id     BIGINT,
    customer_id  BIGINT,
    product_name STRING,
    quantity     INT,
    price        DOUBLE,
    order_time   TIMESTAMP,
    status       STRING
) USING iceberg
PARTITIONED BY (days(order_time), bucket(8, customer_id))
TBLPROPERTIES (
    'write.format.default'          = 'parquet',
    'write.parquet.compression-codec' = 'zstd',
    'write.target-file-size-bytes'  = '536870912',  -- 512MB
    'read.split.target-size'        = '134217728'    -- 128MB
);

-- 插入数据
INSERT INTO lakehouse.db.orders VALUES
    (1001, 501, 'Laptop',     1, 5999.00, TIMESTAMP '2024-06-15 10:30:00', 'PAID'),
    (1002, 502, 'Phone',      2, 3999.00, TIMESTAMP '2024-06-15 11:00:00', 'PAID'),
    (1003, 503, 'Headphones', 3, 299.00,  TIMESTAMP '2024-06-15 11:30:00', 'PENDING');

-- MERGE INTO (Upsert模式)
MERGE INTO lakehouse.db.orders AS target
USING (
    SELECT * FROM staging.order_updates
) AS source
ON target.order_id = source.order_id
WHEN MATCHED AND source.status = 'CANCELLED' THEN
    DELETE
WHEN MATCHED THEN
    UPDATE SET
        status     = source.status,
        quantity   = source.quantity,
        price      = source.price
WHEN NOT MATCHED THEN
    INSERT (order_id, customer_id, product_name, quantity, price, order_time, status)
    VALUES (source.order_id, source.customer_id, source.product_name,
            source.quantity, source.price, source.order_time, source.status);
```

**Time Travel操作**：

```sql
-- 查看快照列表
SELECT snapshot_id, committed_at, operation, summary
FROM lakehouse.db.orders.snapshots;

-- 按时间戳回查
SELECT COUNT(*), SUM(price * quantity) AS total_amount
FROM lakehouse.db.orders
TIMESTAMP AS OF '2024-06-15 12:00:00';

-- 按版本号回查
SELECT * FROM lakehouse.db.orders VERSION AS OF 4519283746251;

-- 回滚到指定快照（调用存储过程）
CALL lakehouse.system.rollback_to_snapshot('db.orders', 4519283746251);

-- 回滚到指定时间
CALL lakehouse.system.rollback_to_timestamp('db.orders', TIMESTAMP '2024-06-15 12:00:00');

-- 查看两个快照之间的增量变更
SELECT * FROM lakehouse.db.orders.changes
WHERE _change_type IN ('insert', 'delete', 'update_before', 'update_after')
  AND snapshot_id BETWEEN 100 AND 200;
```

**Schema Evolution DDL**：

```sql
-- 添加嵌套结构字段
ALTER TABLE lakehouse.db.orders
    ADD COLUMN shipping STRUCT<
        address: STRING,
        city: STRING,
        zip_code: STRING,
        carrier: STRING
    >;

-- 在结构体内部添加字段
ALTER TABLE lakehouse.db.orders
    ADD COLUMN shipping.country STRING AFTER shipping.zip_code;

-- 重命名列
ALTER TABLE lakehouse.db.orders RENAME COLUMN product_name TO item_name;

-- 类型提升
ALTER TABLE lakehouse.db.orders ALTER COLUMN quantity TYPE BIGINT;
```

**PyIceberg API基础**：

```python
from pyiceberg.catalog import load_catalog

# 加载Catalog
catalog = load_catalog(
    "my_catalog",
    **{
        "type": "rest",
        "uri": "http://localhost:8181",
        "s3.endpoint": "http://localhost:9000",
        "s3.access-key-id": "admin",
        "s3.secret-access-key": "password",
    }
)

# 列出命名空间
namespaces = catalog.list_namespaces()
print(f"命名空间列表: {namespaces}")

# 列出表
tables = catalog.list_tables("db")
print(f"表列表: {tables}")

# 加载表
table = catalog.load_table("db.orders")

# 查看Schema
print(f"Schema: {table.schema()}")
print(f"Partition Spec: {table.spec()}")
print(f"当前快照: {table.current_snapshot()}")

# 查看快照历史
for snapshot in table.metadata.snapshots:
    print(f"  Snapshot {snapshot.snapshot_id} at {snapshot.timestamp_ms}")

# 使用PyArrow读取数据
arrow_table = table.scan(
    row_filter="status = 'PAID' AND price > 1000",
    selected_fields=("order_id", "item_name", "price", "status"),
    limit=100
).to_arrow()

print(f"查询结果行数: {len(arrow_table)}")
print(arrow_table.to_pandas())

# Schema演进
from pyiceberg.schema import Schema
from pyiceberg.types import NestedField, StringType, DoubleType

with table.update_schema() as update:
    update.add_column("discount", DoubleType(), doc="折扣金额")
    update.add_column("coupon_code", StringType(), doc="优惠券代码")

# Time Travel — 扫描历史快照
snapshot_id = table.metadata.snapshots[0].snapshot_id
historical_scan = table.scan(snapshot_id=snapshot_id)
historical_data = historical_scan.to_arrow()
print(f"历史快照 {snapshot_id} 数据行数: {len(historical_data)}")
```

### 2.4 Spark集成

**Spark Session配置**：

```scala
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder()
    .appName("Iceberg-Demo")
    .config("spark.sql.extensions",
        "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
    .config("spark.sql.catalog.lakehouse", "org.apache.iceberg.spark.SparkCatalog")
    .config("spark.sql.catalog.lakehouse.type", "hive")
    .config("spark.sql.catalog.lakehouse.uri", "thrift://hive-metastore:9083")
    .config("spark.sql.catalog.lakehouse.warehouse", "s3a://my-bucket/warehouse")
    // REST Catalog 配置示例
    // .config("spark.sql.catalog.rest_catalog", "org.apache.iceberg.spark.SparkCatalog")
    // .config("spark.sql.catalog.rest_catalog.type", "rest")
    // .config("spark.sql.catalog.rest_catalog.uri", "http://rest-catalog:8181")
    .getOrCreate()
```

**Scala DataFrame读写**：

```scala
import org.apache.spark.sql.functions._

// 读取Iceberg表
val ordersDF = spark.read
    .format("iceberg")
    .load("lakehouse.db.orders")

// 基本分析
ordersDF
    .filter(col("status") === "PAID")
    .groupBy(window(col("order_time"), "1 hour"))
    .agg(
        count("*").as("order_count"),
        sum(col("price") * col("quantity")).as("total_revenue")
    )
    .orderBy("window")
    .show(truncate = false)

// 写入Iceberg表
val newOrdersDF = spark.read.parquet("s3a://staging/new-orders/")

newOrdersDF.writeTo("lakehouse.db.orders")
    .option("merge-schema", "true")   // 自动合并Schema
    .append()

// 覆盖指定分区（动态覆盖）
newOrdersDF.writeTo("lakehouse.db.orders")
    .overwritePartitions()

// Time Travel读取
val historicalDF = spark.read
    .option("snapshot-id", 5678901234L)
    .format("iceberg")
    .load("lakehouse.db.orders")

val asOfDF = spark.read
    .option("as-of-timestamp", "1718438400000")  // epoch millis
    .format("iceberg")
    .load("lakehouse.db.orders")

// 增量读取（两个快照之间的变更）
val changesDF = spark.read
    .format("iceberg")
    .option("start-snapshot-id", 1000L)
    .option("end-snapshot-id", 2000L)
    .load("lakehouse.db.orders")
```

**Spark SQL操作**：

```sql
-- 使用Spark SQL创建Iceberg表
CREATE TABLE lakehouse.db.user_events (
    event_id   BIGINT,
    user_id    BIGINT,
    event_type STRING,
    properties MAP<STRING, STRING>,
    event_time TIMESTAMP
) USING iceberg
PARTITIONED BY (days(event_time))
TBLPROPERTIES (
    'format-version' = '2',
    'write.delete.mode' = 'merge-on-read',
    'write.update.mode' = 'merge-on-read'
);

-- 插入数据
INSERT INTO lakehouse.db.user_events VALUES
    (1, 1001, 'page_view', MAP('page', '/home'), TIMESTAMP '2024-06-15 10:00:00'),
    (2, 1002, 'click', MAP('button', 'buy_now'), TIMESTAMP '2024-06-15 10:05:00');

-- 分析查询
SELECT event_type,
       COUNT(*)                           AS event_count,
       COUNT(DISTINCT user_id)            AS unique_users,
       DATE_FORMAT(event_time, 'yyyy-MM-dd HH:00') AS hour_bucket
FROM lakehouse.db.user_events
WHERE event_time >= '2024-06-15'
GROUP BY event_type, DATE_FORMAT(event_time, 'yyyy-MM-dd HH:00')
ORDER BY hour_bucket, event_count DESC;

-- 元数据查询
SELECT * FROM lakehouse.db.user_events.snapshots;
SELECT * FROM lakehouse.db.user_events.manifests;
SELECT * FROM lakehouse.db.user_events.files;    -- 数据文件统计
SELECT * FROM lakehouse.db.user_events.partitions;

-- 表维护操作
-- 过期快照清理（保留最近7天）
CALL lakehouse.system.expire_snapshots(
    table => 'db.user_events',
    older_than => TIMESTAMP '2024-06-08 00:00:00',
    retain_last => 10
);

-- 合并小文件（Compaction）
CALL lakehouse.system.rewrite_data_files(
    table => 'db.user_events',
    strategy => 'sort',
    sort_order => 'user_id ASC, event_time ASC',
    options => MAP(
        'target-file-size-bytes', '536870912',
        'min-file-size-bytes', '67108864',
        'max-file-size-bytes', '1073741824'
    )
);

-- 清理孤立文件
CALL lakehouse.system.remove_orphan_files(
    table => 'db.user_events',
    older_than => TIMESTAMP '2024-06-10 00:00:00'
);
```

---

## 3. Delta Lake

### 3.1 架构

Delta Lake由Databricks创建，在Parquet文件之上增加了一个事务日志层（_delta_log），
提供ACID事务、可伸缩元数据处理和Time Travel等能力。

```
Delta Lake 架构

┌──────────────────────────────────────────────────────────────┐
│                    Delta Table 目录结构                       │
│                                                              │
│  s3://bucket/warehouse/db/orders/                            │
│  │                                                           │
│  ├── _delta_log/                   ← 事务日志目录             │
│  │   ├── 00000000000000000000.json ← 第0次提交              │
│  │   ├── 00000000000000000001.json ← 第1次提交              │
│  │   ├── 00000000000000000002.json ← 第2次提交              │
│  │   ├── ...                                                │
│  │   ├── 00000000000000000010.checkpoint.parquet             │
│  │   │                              ↑ 每10次提交生成Checkpoint│
│  │   └── _last_checkpoint           ← 指向最新Checkpoint     │
│  │                                                           │
│  ├── part-00000-xxx.snappy.parquet  ← 数据文件               │
│  ├── part-00001-xxx.snappy.parquet                           │
│  ├── year=2024/month=06/            ← Hive风格分区（可选）    │
│  │   ├── part-00000-yyy.snappy.parquet                       │
│  │   └── part-00001-yyy.snappy.parquet                       │
│  └── ...                                                     │
└──────────────────────────────────────────────────────────────┘
```

**事务日志（_delta_log）详解**：

```
JSON事务日志内容示例（第2次提交）

┌────────────────────────────────────────────────────────┐
│  00000000000000000002.json                              │
│  ┌──────────────────────────────────────────────────┐  │
│  │ { "commitInfo": {                                │  │
│  │     "timestamp": 1718452800000,                  │  │
│  │     "operation": "MERGE",                        │  │
│  │     "operationParameters": {...}                 │  │
│  │ }}                                               │  │
│  │ { "remove": {                                    │  │
│  │     "path": "part-00000-old.parquet",            │  │
│  │     "deletionTimestamp": 1718452800000            │  │
│  │ }}                                               │  │
│  │ { "add": {                                       │  │
│  │     "path": "part-00000-new.parquet",            │  │
│  │     "size": 53687091,                            │  │
│  │     "partitionValues": {"year":"2024","month":"06"},│ │
│  │     "stats": "{\"numRecords\":50000,             │  │
│  │               \"minValues\":{\"price\":9.99},    │  │
│  │               \"maxValues\":{\"price\":9999.00}}"│  │
│  │ }}                                               │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

**MVCC与乐观并发控制**：

```
乐观并发控制流程

  Writer A                   Writer B                  _delta_log
  ────────                   ────────                  ──────────
     │                          │
     │  读取当前版本 v5          │
     │ ←────────────────────────┼────── version 5
     │                          │
     │                          │  读取当前版本 v5
     │                          │ ←──── version 5
     │                          │
     │  提交 v6 (成功)          │
     │ ─────────────────────────┼────→ version 6  ✅
     │                          │
     │                          │  提交 v6 (冲突!)
     │                          │ ────→ CONFLICT  ❌
     │                          │
     │                          │  重新读取 v6
     │                          │ ←──── version 6
     │                          │
     │                          │  Rebase + 提交 v7
     │                          │ ────→ version 7  ✅
     │                          │
```

### 3.2 核心特性

**1) ACID Transactions**

Delta Lake通过事务日志保证所有操作的原子性、一致性、隔离性和持久性。

```python
# ✅ 原子性写入 — 要么全部成功，要么全部失败
df.write.format("delta").mode("append").save("/data/orders")

# ✅ 并发写入安全
# Writer A 和 Writer B 可以同时 append，不会产生冲突
# 冲突检测在 commit 阶段进行
```

**2) Schema Enforcement & Evolution**

```python
# ✅ Schema Enforcement（写入时校验）
# 如果新数据Schema不匹配，写入会被拒绝
try:
    bad_df.write.format("delta").mode("append").save("/data/orders")
except Exception as e:
    print(f"Schema不匹配: {e}")
    # AnalysisException: A schema mismatch detected when writing to the Delta table.

# ✅ Schema Evolution（允许Schema变更）
new_df.write.format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .save("/data/orders")

# 或者覆盖写入时替换Schema
new_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save("/data/orders")
```

**3) Time Travel**

```sql
-- 按版本号查询
SELECT * FROM orders VERSION AS OF 5;

-- 按时间戳查询
SELECT * FROM orders TIMESTAMP AS OF '2024-06-15 10:00:00';

-- 查看变更历史
DESCRIBE HISTORY orders;

-- 结果示例:
-- +---------+-------------------+---------+------------------+
-- | version | timestamp         | operation| operationParams |
-- +---------+-------------------+---------+------------------+
-- |    5    | 2024-06-15 12:00  | MERGE   | {predicate:...}  |
-- |    4    | 2024-06-15 11:00  | WRITE   | {mode:Append}    |
-- |    3    | 2024-06-15 10:00  | DELETE  | {predicate:...}  |
-- +---------+-------------------+---------+------------------+
```

**4) Change Data Feed (CDF)**

CDF允许下游消费者增量读取表的变更数据，非常适合构建CDC管道。

```sql
-- 启用 Change Data Feed
ALTER TABLE orders SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true');
```

```python
# 读取变更数据
changes_df = spark.read.format("delta") \
    .option("readChangeDataFeed", "true") \
    .option("startingVersion", 3) \
    .option("endingVersion", 5) \
    .table("orders")

# 变更数据包含特殊列:
#   _change_type: insert / update_preimage / update_postimage / delete
#   _commit_version: 变更发生的版本号
#   _commit_timestamp: 变更发生的时间戳

changes_df.show()
# +--------+------+-------+--------------+-----------------+-------------------+
# |order_id|status| price |_change_type  |_commit_version  |_commit_timestamp  |
# +--------+------+-------+--------------+-----------------+-------------------+
# |  1001  | PAID | 5999.0|update_preimage|       5        |2024-06-15 12:00:00|
# |  1001  |SHIPPED|5999.0|update_postimage|      5        |2024-06-15 12:00:00|
# |  1004  | PAID | 799.0 |   insert     |       4        |2024-06-15 11:00:00|
# +--------+------+-------+--------------+-----------------+-------------------+

# 流式读取变更（用于实时ETL下游）
streaming_changes = spark.readStream.format("delta") \
    .option("readChangeDataFeed", "true") \
    .option("startingVersion", 0) \
    .table("orders")

streaming_changes.writeStream \
    .format("delta") \
    .option("checkpointLocation", "/checkpoints/orders_downstream") \
    .trigger(processingTime="30 seconds") \
    .start("/data/orders_derived")
```

### 3.3 操作实战

**PySpark + Delta Lake完整操作**：

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from delta.tables import DeltaTable

# 创建SparkSession（带Delta Lake支持）
spark = SparkSession.builder \
    .appName("DeltaLake-Demo") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.databricks.delta.retentionDurationCheck.enabled", "false") \
    .getOrCreate()

# ===== 创建Delta表 =====
data = [
    (1001, 501, "Laptop",     1, 5999.00, "2024-06-15 10:30:00", "PAID"),
    (1002, 502, "Phone",      2, 3999.00, "2024-06-15 11:00:00", "PAID"),
    (1003, 503, "Headphones", 3, 299.00,  "2024-06-15 11:30:00", "PENDING"),
]
columns = ["order_id", "customer_id", "product_name",
           "quantity", "price", "order_time", "status"]

df = spark.createDataFrame(data, columns) \
    .withColumn("order_time", to_timestamp("order_time"))

df.write.format("delta") \
    .mode("overwrite") \
    .partitionBy("status") \
    .save("/data/delta/orders")

# ===== 读取Delta表 =====
orders = spark.read.format("delta").load("/data/delta/orders")
orders.show()

# ===== MERGE / UPSERT =====
delta_table = DeltaTable.forPath(spark, "/data/delta/orders")

updates = spark.createDataFrame([
    (1001, 501, "Laptop",     1, 5999.00, "2024-06-15 10:30:00", "SHIPPED"),
    (1004, 504, "Tablet",     1, 2999.00, "2024-06-15 14:00:00", "PAID"),
], columns).withColumn("order_time", to_timestamp("order_time"))

delta_table.alias("target").merge(
    updates.alias("source"),
    "target.order_id = source.order_id"
).whenMatchedUpdate(set={
    "status":   col("source.status"),
    "quantity": col("source.quantity"),
    "price":    col("source.price"),
}).whenNotMatchedInsertAll().execute()

# ===== UPDATE =====
delta_table.update(
    condition=expr("status = 'PENDING' AND order_time < '2024-06-15 12:00:00'"),
    set={"status": lit("CANCELLED")}
)

# ===== DELETE =====
delta_table.delete(condition=expr("status = 'CANCELLED'"))

# ===== 查看历史 =====
delta_table.history().select(
    "version", "timestamp", "operation", "operationParameters"
).show(truncate=False)
```

**表维护操作（SQL）**：

```sql
-- VACUUM: 清理不再被引用的旧文件
-- 默认保留7天内的文件（防止正在运行的查询失败）
VACUUM orders RETAIN 168 HOURS;

-- 干跑模式（只查看会删除哪些文件）
VACUUM orders RETAIN 168 HOURS DRY RUN;

-- OPTIMIZE: 合并小文件，提升查询性能
OPTIMIZE orders;

-- 针对特定分区优化
OPTIMIZE orders WHERE status = 'PAID';

-- Z-ORDER: 多维数据聚簇，加速多列过滤查询
OPTIMIZE orders ZORDER BY (customer_id, order_time);

-- Z-ORDER原理:
-- 将 customer_id 和 order_time 的值交错排列
-- 使得按这两列过滤时，能跳过更多不相关的文件
```

```
Z-ORDER 效果对比

  无Z-ORDER (文件内容随机分布)
  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │ file1        │ │ file2        │ │ file3        │
  │ cust: 1~1000 │ │ cust: 1~1000 │ │ cust: 1~1000 │
  │ time: 混合   │ │ time: 混合   │ │ time: 混合   │
  └──────────────┘ └──────────────┘ └──────────────┘
  查询 WHERE customer_id = 500 → 扫描全部3个文件 ❌

  有Z-ORDER (数据按customer_id+time聚簇)
  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │ file1        │ │ file2        │ │ file3        │
  │ cust: 1~300  │ │ cust: 301~700│ │ cust: 701~1000│
  │ time: Jan-Mar│ │ time: Jan-Jun│ │ time: Apr-Jun │
  └──────────────┘ └──────────────┘ └──────────────┘
  查询 WHERE customer_id = 500 → 只扫描file2 ✅
```

**DESCRIBE HISTORY**：

```sql
-- 查看完整变更历史
DESCRIBE HISTORY orders;

-- 限制返回条目数
DESCRIBE HISTORY orders LIMIT 10;

-- 输出字段说明:
-- version          - 版本号
-- timestamp        - 提交时间
-- userId           - 执行用户
-- userName         - 用户名
-- operation        - 操作类型 (WRITE, MERGE, DELETE, UPDATE, OPTIMIZE, VACUUM等)
-- operationParameters - 操作参数
-- operationMetrics - 操作指标 (numOutputRows, numTargetRowsInserted等)
-- readVersion      - 读取的基础版本
-- isolationLevel   - 隔离级别

-- 基于历史进行恢复
RESTORE TABLE orders TO VERSION AS OF 3;
RESTORE TABLE orders TO TIMESTAMP AS OF '2024-06-15 10:00:00';
```

### 3.4 Delta vs Iceberg对比

| 对比维度 | Delta Lake | Apache Iceberg |
|---------|-----------|---------------|
| **治理方** | Linux基金会（原Databricks） | Apache基金会（Netflix发起） |
| **元数据管理** | _delta_log事务日志（JSON+Checkpoint） | metadata.json + Manifest List + Manifest Files |
| **Catalog** | Unity Catalog / Hive Metastore | REST / Hive / Glue / Nessie（更灵活） |
| **分区演进** | ❌ 不支持（需重写数据） | ✅ 原生支持（零重写） |
| **隐式分区** | ❌ 不支持 | ✅ 支持 |
| **引擎支持** | Spark最优，Flink/Trino可用 | Spark/Flink/Trino/Presto/Dremio均良好 |
| **云集成** | Azure/Databricks原生最优 | AWS/GCP/Azure均良好 |
| **MERGE性能** | 文件级（Copy-on-Write） | 文件级（v2支持Merge-on-Read） |
| **Change Data Feed** | ✅ 原生CDF | ✅ 增量扫描 |
| **小文件合并** | OPTIMIZE命令 | rewrite_data_files存储过程 |
| **数据聚簇** | Z-ORDER BY | sort order + rewrite |
| **Deletion Vectors** | ✅ 支持（v2.4+） | ✅ 支持（Format v2） |
| **UniForm** | ✅ 可生成Iceberg元数据 | N/A（原生格式） |
| **生态锁定** | Databricks生态偏向 | 厂商中立，社区驱动 |
| **适用场景** | Databricks用户首选 | 多引擎、多云环境首选 |

---

## 4. Apache Hudi

### 4.1 架构

Apache Hudi（Hadoop Upserts Deletes and Incrementals）由Uber开发，
专为高频率数据摄入和增量处理场景优化。其核心架构围绕Timeline、File Group和File Slice构建。

```
Apache Hudi 架构

┌──────────────────────────────────────────────────────────────┐
│                        Timeline                              │
│   记录表上所有操作的时间线                                     │
│                                                              │
│   ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐            │
│   │commit  │→ │commit  │→ │delta   │→ │compaction│           │
│   │  001   │  │  002   │  │commit  │  │   004   │           │
│   │COMPLETE│  │COMPLETE│  │  003   │  │COMPLETE │           │
│   └────────┘  └────────┘  │COMPLETE│  └────────┘            │
│                           └────────┘                         │
│   操作类型: COMMITS / DELTA_COMMITS / COMPACTIONS /           │
│            CLEANS / SAVEPOINTS / ROLLBACKS                   │
└──────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                    Partition (可选)                           │
│   例如: /data/hudi/orders/dt=2024-06-15/                     │
│                                                              │
│   ┌──────────────────────────────────────────────┐           │
│   │              File Group 1                     │           │
│   │   (由File ID唯一标识，绑定到同一组记录)         │           │
│   │                                               │           │
│   │   File Slice (commit 001)                     │           │
│   │   ┌───────────────────┐                       │           │
│   │   │ base_file_001.pq  │  Base File (Parquet)  │           │
│   │   └───────────────────┘                       │           │
│   │                                               │           │
│   │   File Slice (commit 002)                     │           │
│   │   ┌───────────────────┐  ┌─────────────────┐  │           │
│   │   │ base_file_002.pq  │  │ .log_001.avro   │  │           │
│   │   └───────────────────┘  │ (delta log file) │  │           │
│   │                          └─────────────────┘  │           │
│   │                                               │           │
│   │   File Slice (commit 003)                     │           │
│   │   ┌───────────────────┐  ┌─────────────────┐  │           │
│   │   │ base_file_002.pq  │  │ .log_001.avro   │  │           │
│   │   │ (同上，未更新)     │  │ .log_002.avro   │  │           │
│   │   └───────────────────┘  │ (追加新delta)    │  │           │
│   │                          └─────────────────┘  │           │
│   └──────────────────────────────────────────────┘           │
│                                                              │
│   ┌──────────────────────────────────────────────┐           │
│   │              File Group 2                     │           │
│   │   ...                                        │           │
│   └──────────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────┘
```

**关键概念**：

- **Timeline（时间线）**：记录表上所有变更操作的有序日志，每个操作有三种状态：
  REQUESTED → INFLIGHT → COMPLETED（或FAILED）
- **File Group**：一组物理文件，由唯一的File ID标识。同一Record Key的数据始终路由到同一File Group
- **File Slice**：File Group在某个commit时间点的快照，包含一个Base File和零或多个Log Files
- **Base File**：Parquet格式的基础数据文件
- **Log File**：Avro格式的增量变更日志（仅MoR表类型使用）

### 4.2 核心特性

**Copy-on-Write (CoW) vs Merge-on-Read (MoR)**：

这是Hudi最核心的设计抉择，两种表类型适用于不同场景。

```
Copy-on-Write (CoW) 工作流程

  写入/更新操作:
  ┌────────────────┐     ┌────────────────┐
  │ base_v1.parquet│     │ base_v2.parquet│   ← 重写整个文件
  │ record A: 100  │ ──→ │ record A: 200  │   (即使只改一条)
  │ record B: 200  │     │ record B: 200  │
  │ record C: 300  │     │ record C: 300  │
  └────────────────┘     └────────────────┘

  读取操作:
  直接读取最新的 base_v2.parquet，无需额外合并 ✅


Merge-on-Read (MoR) 工作流程

  写入/更新操作:
  ┌────────────────┐   ┌──────────────┐
  │ base_v1.parquet│ + │ log_001.avro │   ← 只写增量日志
  │ record A: 100  │   │ A → 200      │   (写入速度快)
  │ record B: 200  │   └──────────────┘
  │ record C: 300  │   ┌──────────────┐
  └────────────────┘ + │ log_002.avro │   ← 继续追加日志
                       │ C → 350      │
                       └──────────────┘

  读取操作 (Snapshot Query):
  读取 base + 合并 log → 得到最新视图（读取较慢）

  读取操作 (Read Optimized Query):
  只读取 base_v1.parquet（快但不包含最新修改）

  Compaction（压实）:
  base + log → 新的 base_v2.parquet（后台任务）
```

| 对比维度 | Copy-on-Write (CoW) | Merge-on-Read (MoR) |
|---------|--------------------|--------------------|
| **写入延迟** | 高（重写整个文件） | 低（仅追加日志） |
| **写入放大** | 高 | 低 |
| **读取延迟** | 低（直接读Parquet） | 中等（需合并日志） |
| **快照查询** | 快 | 较慢（需merge） |
| **Read Optimized查询** | 等同快照查询 | 快（但数据不是最新） |
| **存储开销** | 较高（每次重写） | 较低（增量日志） |
| **是否需要Compaction** | 否 | 是 |
| **适用场景** | 读多写少、批量更新 | 写多读少、近实时摄入 |
| **典型用例** | 数仓维度表、报表 | CDC流水、日志摄入、实时大屏 |

**Record-level Indexing（记录级索引）**：

Hudi的高效Upsert依赖于索引快速定位记录所在的File Group。

| 索引类型 | 原理 | 适用场景 |
|---------|------|---------|
| **Bloom Index** | 基于Bloom Filter，存储在Parquet Footer | 默认索引，适合大部分场景 |
| **Simple Index** | 全表扫描匹配 | 小表或测试 |
| **HBase Index** | 使用外部HBase存储索引映射 | 超大规模表，需要全局索引 |
| **Bucket Index** | 一致性Hash分桶，Record Key直接映射到File Group | 高吞吐写入，无需索引查找 |
| **Record Index** | Hudi 0.14+，基于HFile的内建全局索引 | 替代HBase Index，无外部依赖 |

### 4.3 操作实战

**Spark + Hudi写入**：

```scala
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SaveMode
import org.apache.hudi.DataSourceWriteOptions._
import org.apache.hudi.config.HoodieWriteConfig._
import org.apache.hudi.QuickstartUtils._

val spark = SparkSession.builder()
    .appName("Hudi-Demo")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.sql.extensions", "org.apache.spark.sql.hudi.HoodieSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.hudi.catalog.HoodieCatalog")
    .getOrCreate()

import spark.implicits._

// 准备数据
val ordersDF = Seq(
    (1001, 501, "Laptop",     1, 5999.00, "2024-06-15", "PAID"),
    (1002, 502, "Phone",      2, 3999.00, "2024-06-15", "PAID"),
    (1003, 503, "Headphones", 3, 299.00,  "2024-06-15", "PENDING")
).toDF("order_id", "customer_id", "product_name",
       "quantity", "price", "dt", "status")

// UPSERT写入（CoW表）
ordersDF.write.format("hudi")
    .option(PRECOMBINE_FIELD.key(), "dt")           // 预合并字段（决定保留哪条记录）
    .option(RECORDKEY_FIELD.key(), "order_id")       // 记录主键
    .option(PARTITIONPATH_FIELD.key(), "dt")         // 分区字段
    .option(TBL_NAME.key(), "orders")                // 表名
    .option(OPERATION.key(), "upsert")               // 操作类型
    .option(TABLE_TYPE.key(), "COPY_ON_WRITE")       // 表类型 CoW
    .option("hoodie.index.type", "BLOOM")            // 索引类型
    .option("hoodie.upsert.shuffle.parallelism", "2")
    .mode(SaveMode.Append)
    .save("/data/hudi/orders")

// INSERT写入（跳过索引查找，纯追加，更快）
newRecordsDF.write.format("hudi")
    .option(PRECOMBINE_FIELD.key(), "dt")
    .option(RECORDKEY_FIELD.key(), "order_id")
    .option(PARTITIONPATH_FIELD.key(), "dt")
    .option(TBL_NAME.key(), "orders")
    .option(OPERATION.key(), "insert")
    .mode(SaveMode.Append)
    .save("/data/hudi/orders")

// BULK_INSERT（首次全量写入，跳过索引和upsert逻辑）
historicalDF.write.format("hudi")
    .option(PRECOMBINE_FIELD.key(), "dt")
    .option(RECORDKEY_FIELD.key(), "order_id")
    .option(PARTITIONPATH_FIELD.key(), "dt")
    .option(TBL_NAME.key(), "orders")
    .option(OPERATION.key(), "bulk_insert")
    .option("hoodie.bulkinsert.sort.mode", "PARTITION_SORT")
    .mode(SaveMode.Overwrite)
    .save("/data/hudi/orders")

// MoR表写入
streamDF.write.format("hudi")
    .option(PRECOMBINE_FIELD.key(), "event_time")
    .option(RECORDKEY_FIELD.key(), "event_id")
    .option(PARTITIONPATH_FIELD.key(), "dt")
    .option(TBL_NAME.key(), "events_mor")
    .option(TABLE_TYPE.key(), "MERGE_ON_READ")       // MoR表
    .option(OPERATION.key(), "upsert")
    .option("hoodie.compact.inline", "true")          // 内联compaction
    .option("hoodie.compact.inline.max.delta.commits", "5")  // 每5次delta提交触发
    .mode(SaveMode.Append)
    .save("/data/hudi/events_mor")
```

**查询操作**：

```scala
// Snapshot Query（读取最新完整数据）
val snapshotDF = spark.read.format("hudi")
    .load("/data/hudi/orders")

snapshotDF.filter("status = 'PAID'")
    .groupBy("dt")
    .agg(
        count("*").as("order_count"),
        sum("price").as("total_revenue")
    )
    .show()

// Incremental Query（增量查询，只读取某commit之后的变更）
val incrementalDF = spark.read.format("hudi")
    .option(QUERY_TYPE.key(), QUERY_TYPE_INCREMENTAL_OPT_VAL)
    .option(BEGIN_INSTANTTIME.key(), "20240615100000")  // 起始commit时间
    .option(END_INSTANTTIME.key(), "20240615140000")    // 结束commit时间（可选）
    .load("/data/hudi/orders")

println(s"增量变更记录数: ${incrementalDF.count()}")
incrementalDF.show()

// Point-in-Time Query（时间点查询）
val pitDF = spark.read.format("hudi")
    .option("as.of.instant", "20240615120000")
    .load("/data/hudi/orders")

pitDF.show()

// MoR表 — Read Optimized Query（只读Base文件，不合并Log）
val readOptDF = spark.read.format("hudi")
    .option(QUERY_TYPE.key(), QUERY_TYPE_READ_OPTIMIZED_OPT_VAL)
    .load("/data/hudi/events_mor")
```

**Hudi SQL扩展**：

```sql
-- 创建Hudi表（CoW）
CREATE TABLE hudi_orders (
    order_id     BIGINT,
    customer_id  BIGINT,
    product_name STRING,
    quantity     INT,
    price        DOUBLE,
    status       STRING,
    dt           STRING
) USING hudi
TBLPROPERTIES (
    type = 'cow',
    primaryKey = 'order_id',
    preCombineField = 'dt'
)
PARTITIONED BY (dt)
LOCATION '/data/hudi/orders_sql';

-- 插入数据
INSERT INTO hudi_orders VALUES
    (1001, 501, 'Laptop',     1, 5999.00, 'PAID',    '2024-06-15'),
    (1002, 502, 'Phone',      2, 3999.00, 'PAID',    '2024-06-15'),
    (1003, 503, 'Headphones', 3, 299.00,  'PENDING', '2024-06-15');

-- MERGE INTO (Upsert)
MERGE INTO hudi_orders AS target
USING (
    SELECT 1001 AS order_id, 501 AS customer_id, 'Laptop' AS product_name,
           1 AS quantity, 5999.00 AS price, 'SHIPPED' AS status, '2024-06-16' AS dt
    UNION ALL
    SELECT 1004, 504, 'Tablet', 1, 2999.00, 'PAID', '2024-06-16'
) AS source
ON target.order_id = source.order_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *;

-- UPDATE
UPDATE hudi_orders
SET status = 'CANCELLED'
WHERE status = 'PENDING' AND dt = '2024-06-15';

-- DELETE
DELETE FROM hudi_orders WHERE status = 'CANCELLED';

-- Time Travel
SELECT * FROM hudi_orders TIMESTAMP AS OF '2024-06-15 12:00:00';

-- 查看提交历史
CALL show_commits(table => 'hudi_orders', limit => 10);

-- 查看Compaction计划（MoR表）
CALL show_compaction(table => 'hudi_events_mor');

-- 手动触发Compaction
CALL run_compaction(table => 'hudi_events_mor', op => 'schedule');
CALL run_compaction(table => 'hudi_events_mor', op => 'execute');
```

### 4.4 增量ETL

Hudi的增量查询（Incremental Query）是构建高效ETL管道的利器，
避免了传统全表扫描的低效模式。

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder \
    .appName("Hudi-Incremental-ETL") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.extensions",
            "org.apache.spark.sql.hudi.HoodieSparkSessionExtension") \
    .getOrCreate()

# ===== 增量ETL管道 =====
# 场景：从orders源表增量读取，加工后写入orders_daily聚合表

def run_incremental_etl(begin_time: str, end_time: str):
    """
    增量ETL：只处理 [begin_time, end_time) 范围内的变更数据

    Args:
        begin_time: 起始commit时间，如 "20240615100000"
        end_time:   结束commit时间，如 "20240615140000"
    """

    # 第1步：增量读取源表变更
    incremental_df = spark.read.format("hudi") \
        .option("hoodie.datasource.query.type", "incremental") \
        .option("hoodie.datasource.read.begin.instanttime", begin_time) \
        .option("hoodie.datasource.read.end.instanttime", end_time) \
        .load("/data/hudi/orders")

    record_count = incremental_df.count()
    print(f"增量读取 {begin_time} ~ {end_time}: {record_count} 条变更")

    if record_count == 0:
        print("无新增数据，跳过本轮ETL")
        return

    # 第2步：数据加工（聚合计算每日订单统计）
    daily_stats = incremental_df \
        .filter("status IN ('PAID', 'SHIPPED')") \
        .groupBy("dt") \
        .agg(
            count("*").alias("order_count"),
            countDistinct("customer_id").alias("unique_customers"),
            sum("price").alias("total_revenue"),
            avg("price").alias("avg_order_value"),
            max("price").alias("max_order_value")
        ) \
        .withColumn("etl_time", current_timestamp())

    # 第3步：写入目标表（Upsert模式确保幂等性）
    daily_stats.write.format("hudi") \
        .option("hoodie.datasource.write.precombine.field", "etl_time") \
        .option("hoodie.datasource.write.recordkey.field", "dt") \
        .option("hoodie.datasource.write.partitionpath.field", "dt") \
        .option("hoodie.table.name", "orders_daily_stats") \
        .option("hoodie.datasource.write.operation", "upsert") \
        .mode("append") \
        .save("/data/hudi/orders_daily_stats")

    print(f"ETL完成，已更新 {daily_stats.count()} 个分区")


# ===== 自动化增量调度 =====
def get_last_checkpoint(checkpoint_path: str) -> str:
    """从checkpoint文件读取上次处理的commit时间"""
    try:
        with open(checkpoint_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "00000000000000"  # 首次运行，从最早开始

def save_checkpoint(checkpoint_path: str, instant_time: str):
    """保存本次处理的最新commit时间"""
    with open(checkpoint_path, "w") as f:
        f.write(instant_time)

def get_latest_commit(table_path: str) -> str:
    """获取Hudi表最新的commit时间"""
    timeline_df = spark.read.format("hudi") \
        .load(table_path) \
        .select("_hoodie_commit_time") \
        .distinct() \
        .orderBy(col("_hoodie_commit_time").desc()) \
        .limit(1)
    return timeline_df.first()[0]


# 主流程
CHECKPOINT_PATH = "/data/checkpoints/orders_etl_checkpoint.txt"
TABLE_PATH = "/data/hudi/orders"

begin_time = get_last_checkpoint(CHECKPOINT_PATH)
end_time = get_latest_commit(TABLE_PATH)

print(f"本轮ETL范围: {begin_time} → {end_time}")

run_incremental_etl(begin_time, end_time)
save_checkpoint(CHECKPOINT_PATH, end_time)
```

```
增量ETL vs 全量ETL 效率对比

  全量ETL (传统方式)
  ┌────────────┐    扫描全表      ┌──────────┐    全量写入     ┌──────────┐
  │ 源表       │ ──────────────→ │ ETL加工  │ ────────────→ │ 目标表   │
  │ 10亿行     │  读取10亿行      │          │  写入聚合结果  │          │
  └────────────┘  (30分钟)       └──────────┘               └──────────┘

  增量ETL (Hudi Incremental Query)
  ┌────────────┐    增量读取      ┌──────────┐    Upsert      ┌──────────┐
  │ 源表       │ ──────────────→ │ ETL加工  │ ────────────→ │ 目标表   │
  │ 10亿行     │  只读1万行变更   │          │  只更新变更    │          │
  └────────────┘  (10秒)         └──────────┘  部分          └──────────┘

  性能提升: 从30分钟 → 10秒 (180倍提升) ✅
```

---

## 5. Lakehouse架构实践

### 5.1 架构设计

现代Lakehouse架构采用分层设计，将存储、表格式、计算和应用解耦，
实现最大的灵活性和可扩展性。

```
Lakehouse 完整架构

┌────────────────────────────────────────────────────────────────┐
│                      应用层 (Application Layer)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │    BI    │  │    ML    │  │   API    │  │ Real-time│      │
│  │ Tableau  │  │ MLflow   │  │ REST    │  │Dashboard │      │
│  │ Superset │  │ SageMaker│  │ GraphQL │  │ Grafana  │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       └──────────────┴──────┬──────┴──────────────┘            │
└─────────────────────────────┼──────────────────────────────────┘
                              │
┌─────────────────────────────┼──────────────────────────────────┐
│                      计算层 (Compute Layer)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  Spark   │  │  Flink   │  │  Trino   │  │  Presto  │      │
│  │  (批处理) │  │  (流处理) │  │ (交互查询)│  │  (联邦)  │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       └──────────────┴──────┬──────┴──────────────┘            │
└─────────────────────────────┼──────────────────────────────────┘
                              │
┌─────────────────────────────┼──────────────────────────────────┐
│                   表格式层 (Table Format Layer)                │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Apache Iceberg  /  Delta Lake  /  Apache Hudi          │  │
│  │                                                          │  │
│  │  提供: ACID事务 | Schema演进 | Time Travel | 分区管理    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Catalog服务                                             │  │
│  │  Hive Metastore / AWS Glue / Nessie / Unity Catalog     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────┼──────────────────────────────────┘
                              │
┌─────────────────────────────┼──────────────────────────────────┐
│                  文件格式层 (File Format Layer)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │ Parquet  │  │   ORC    │  │  Avro    │                     │
│  │ (列存储) │  │ (列存储) │  │ (行存储) │                     │
│  └──────────┘  └──────────┘  └──────────┘                     │
└─────────────────────────────┼──────────────────────────────────┘
                              │
┌─────────────────────────────┼──────────────────────────────────┐
│                  存储层 (Storage Layer)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  AWS S3  │  │Azure ADLS│  │ GCS      │  │  HDFS    │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
└────────────────────────────────────────────────────────────────┘
```

**Medallion Architecture（奖章架构）**：

Lakehouse中最常用的数据分层模式，将数据处理分为Bronze、Silver、Gold三层。

```
Medallion Architecture (Bronze → Silver → Gold)

  数据源                Bronze层            Silver层             Gold层
  ┌──────┐           ┌───────────┐      ┌────────────┐      ┌───────────┐
  │ MySQL│──CDC──→  │ 原始数据   │      │ 清洗数据    │      │ 聚合指标   │
  │ Kafka│──流式──→ │ 未清洗     │ ──→  │ 去重/规范化 │ ──→  │ 宽表/立方体│
  │ API  │──批量──→ │ Schema原样 │      │ Schema统一  │      │ BI就绪     │
  │ Files│──导入──→ │ 分区: 摄入时间│   │ 分区: 业务键│      │ 分区: 查询优│
  └──────┘           └───────────┘      └────────────┘      └───────────┘
                     表格式: Iceberg     表格式: Iceberg      表格式: Iceberg
                     保留: 90天          保留: 1年            保留: 3年
                     质量: 无校验        质量: 基础校验        质量: 严格校验
```

### 5.2 统一数据治理

**Catalog管理**：

不同的Catalog服务适用于不同场景：

| Catalog | 适用场景 | 特点 |
|---------|---------|------|
| **Hive Metastore** | 传统Hadoop生态 | 成熟稳定，广泛支持 |
| **AWS Glue Catalog** | AWS云原生 | 与AWS服务深度集成 |
| **Nessie** | 多分支数据治理 | Git-like版本控制，支持分支/合并 |
| **Unity Catalog** | Databricks生态 | 统一治理，细粒度权限 |
| **Polaris Catalog** | 开源REST Catalog | Snowflake开源，Iceberg原生 |

**Nessie Catalog — Git-like数据版本管理**：

```
Nessie分支模型

  main分支 (生产)
  ──●──────●──────●──────●──────●──── (稳定数据)
           │                   ↑
           │                   │ merge
           ↓                   │
  etl-dev分支 (开发)            │
  ─────────●──────●──────●─────┘
           │      │      │
           │  修改Schema  │
           │  添加新表    │
           │  测试验证    │

  好处:
  ✅ 开发环境修改不影响生产
  ✅ 验证通过后一键merge到main
  ✅ 出问题可以回滚到任意commit
  ✅ 多团队并行开发互不干扰
```

**访问控制**：

```sql
-- 列级别访问控制（以Trino + Iceberg为例）
-- 创建策略：普通分析师不能看到PII字段
CREATE SECURITY POLICY pii_policy
    ON lakehouse.db.customers
    FOR SELECT
    DENY COLUMNS (ssn, phone_number, email)
    TO ROLE analyst;

-- 行级别访问控制
-- 区域经理只能看到自己区域的数据
CREATE ROW ACCESS POLICY region_policy
    ON lakehouse.db.orders
    AS (region_filter)
    USING (region = current_user_region());

-- 数据脱敏（动态掩码）
-- 对敏感字段进行掩码处理
CREATE MASKING POLICY email_mask
    AS (val STRING) RETURNS STRING ->
    CASE
        WHEN current_role() IN ('admin', 'data_engineer')
        THEN val
        ELSE CONCAT(LEFT(val, 2), '***@***.com')
    END;

ALTER TABLE lakehouse.db.customers
    ALTER COLUMN email SET MASKING POLICY email_mask;
```

**数据血缘与审计**：

```python
# 利用Iceberg快照元数据构建数据血缘
from pyiceberg.catalog import load_catalog

catalog = load_catalog("lakehouse", type="rest", uri="http://catalog:8181")
table = catalog.load_table("db.orders")

# 审计：查看所有历史操作
print("=== 数据变更审计日志 ===")
for snapshot in table.metadata.snapshots:
    summary = snapshot.summary
    print(f"  时间: {snapshot.timestamp_ms}")
    print(f"  操作: {summary.get('operation', 'unknown')}")
    print(f"  新增文件: {summary.get('added-data-files', 0)}")
    print(f"  删除文件: {summary.get('deleted-data-files', 0)}")
    print(f"  新增记录: {summary.get('added-records', 0)}")
    print(f"  删除记录: {summary.get('deleted-records', 0)}")
    print(f"  总记录数: {summary.get('total-records', 0)}")
    print("---")
```

---

## 6. 实战案例：Lakehouse替代传统数仓

### 6.1 需求背景

某电商平台拥有以下数据规模：
- 日增订单量：500万条
- 历史数据总量：50亿条（约30TB Parquet）
- 分区方式：按天分区（dt=yyyy-MM-dd）
- 存储：HDFS + Hive外部表
- 查询引擎：Spark SQL + Presto

**面临的问题**：

```
迁移前架构 (Hive + 手动分区管理)

┌──────────────────────────────────────────────────────────┐
│                    传统Hive数仓                           │
│                                                          │
│  问题1: Schema变更需要重写全部分区                         │
│  ┌──────────────────────────────┐                        │
│  │ ALTER TABLE orders            │                        │
│  │ ADD COLUMNS (discount DOUBLE);│  ← 只能加在末尾       │
│  │ -- 旧分区中该字段全部为NULL     │  ← 新旧数据Schema不一致│
│  └──────────────────────────────┘                        │
│                                                          │
│  问题2: 分区变更 = 灾难                                   │
│  ┌──────────────────────────────┐                        │
│  │ -- 从按天分区改为按小时分区     │                        │
│  │ -- 需要重写全部30TB数据！       │  ← 预计耗时48小时     │
│  │ -- 期间表不可用                 │  ← 业务中断           │
│  └──────────────────────────────┘                        │
│                                                          │
│  问题3: 并发写入冲突                                      │
│  ┌──────────────────────────────┐                        │
│  │ -- ETL写入和即席查询并发执行    │                        │
│  │ -- 经常出现脏读或写入失败       │                        │
│  │ -- 无ACID保障                  │                        │
│  └──────────────────────────────┘                        │
│                                                          │
│  问题4: 无法回溯历史数据                                   │
│  ┌──────────────────────────────┐                        │
│  │ -- 误删数据后无法恢复          │                        │
│  │ -- 无Time Travel能力          │                        │
│  │ -- 只能依赖HDFS快照（粒度太粗）│                        │
│  └──────────────────────────────┘                        │
└──────────────────────────────────────────────────────────┘
```

```
迁移后架构 (Iceberg Lakehouse)

┌──────────────────────────────────────────────────────────┐
│                   Iceberg Lakehouse                       │
│                                                          │
│  ✅ Schema演进: 按列ID追踪，安全添加/删除/重命名           │
│  ┌──────────────────────────────────────────────┐        │
│  │ ALTER TABLE orders ADD COLUMN discount DOUBLE │        │
│  │ AFTER price;                                  │        │
│  │ -- 旧数据文件不变，读取时自动填充NULL            │        │
│  │ -- 无需重写任何数据                             │        │
│  └──────────────────────────────────────────────┘        │
│                                                          │
│  ✅ 分区演进: 零停机切换分区方案                            │
│  ┌──────────────────────────────────────────────┐        │
│  │ ALTER TABLE orders ADD PARTITION FIELD        │        │
│  │ hours(order_time);                            │        │
│  │ -- 旧数据保持天分区，新数据使用小时分区           │        │
│  │ -- 查询引擎自动选择正确的分区裁剪策略            │        │
│  └──────────────────────────────────────────────┘        │
│                                                          │
│  ✅ ACID事务: 快照隔离，读写互不干扰                       │
│  ┌──────────────────────────────────────────────┐        │
│  │ -- ETL写入创建新快照                           │        │
│  │ -- 查询基于已提交的快照                         │        │
│  │ -- 乐观并发控制处理写入冲突                      │        │
│  └──────────────────────────────────────────────┘        │
│                                                          │
│  ✅ Time Travel: 完整历史审计和数据回溯                    │
│  ┌──────────────────────────────────────────────┐        │
│  │ SELECT * FROM orders TIMESTAMP AS OF           │        │
│  │ '2024-06-01 00:00:00';                         │        │
│  │ CALL system.rollback_to_snapshot('orders', id);│        │
│  └──────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────┘
```

### 6.2 迁移方案

**方案一：原地迁移（In-place Migration）**

适用于Hive表直接升级为Iceberg表，无需数据拷贝。

```sql
-- 步骤1：在Hive Metastore中将Hive表迁移为Iceberg表
-- 此操作只修改元数据，不移动或重写数据文件
CALL lakehouse.system.migrate('hive_db.orders');

-- 或者使用 ALTER TABLE（Spark SQL）
ALTER TABLE hive_db.orders
SET TBLPROPERTIES (
    'storage_handler' = 'org.apache.iceberg.mr.hive.HiveIcebergStorageHandler'
);

-- 步骤2：验证迁移结果
DESCRIBE EXTENDED lakehouse.hive_db.orders;
SELECT COUNT(*) FROM lakehouse.hive_db.orders;

-- 步骤3：设置表属性
ALTER TABLE lakehouse.hive_db.orders SET TBLPROPERTIES (
    'format-version'                  = '2',
    'write.format.default'            = 'parquet',
    'write.parquet.compression-codec' = 'zstd',
    'write.target-file-size-bytes'    = '536870912'
);

-- 步骤4：（可选）执行分区演进
ALTER TABLE lakehouse.hive_db.orders
    ADD PARTITION FIELD hours(order_time);
```

```
✅ 原地迁移优势:
   - 零数据拷贝，秒级完成
   - 原始Parquet文件不变
   - 迁移后立即获得ACID和Time Travel

❌ 原地迁移限制:
   - 需要Hive Metastore兼容
   - 历史快照从迁移时刻开始（迁移前无历史）
   - 表在迁移期间需要短暂锁定
```

**方案二：影子迁移（Shadow Migration）**

适用于需要更高安全性的场景，新旧表并行运行。

```sql
-- 步骤1：创建新的Iceberg表（Schema与原表一致）
CREATE TABLE lakehouse.db.orders_iceberg (
    order_id     BIGINT,
    customer_id  BIGINT,
    product_name STRING,
    quantity     INT,
    price        DOUBLE,
    order_time   TIMESTAMP,
    status       STRING
) USING iceberg
PARTITIONED BY (days(order_time))
TBLPROPERTIES (
    'format-version' = '2',
    'write.format.default' = 'parquet',
    'write.parquet.compression-codec' = 'zstd'
);

-- 步骤2：全量数据迁移（批量插入）
INSERT INTO lakehouse.db.orders_iceberg
SELECT * FROM hive_db.orders;

-- 步骤3：增量同步（双写期间）
-- 使用Spark Streaming从Kafka同时写入新旧两张表
-- 或使用CDC工具（如Debezium）将变更同步到Iceberg表

-- 步骤4：数据一致性校验
SELECT 'hive' AS source, COUNT(*) AS cnt, SUM(price) AS total
FROM hive_db.orders
WHERE dt = '2024-06-15'
UNION ALL
SELECT 'iceberg' AS source, COUNT(*) AS cnt, SUM(price) AS total
FROM lakehouse.db.orders_iceberg
WHERE order_time >= '2024-06-15' AND order_time < '2024-06-16';

-- 步骤5：切换流量（修改视图指向）
CREATE OR REPLACE VIEW db.orders AS
SELECT * FROM lakehouse.db.orders_iceberg;

-- 步骤6：下线旧Hive表（确认无问题后）
-- DROP TABLE hive_db.orders;
```

```
影子迁移时间线

  Day 1          Day 2-7          Day 8           Day 9+
  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────┐
  │ 全量迁移  │  │ 双写+增量同步 │  │ 一致性   │  │ 切换流量  │
  │ Hive →   │  │ Hive + Ice   │  │ 校验     │  │ 下线Hive │
  │ Iceberg  │  │ 并行运行      │  │ 通过     │  │          │
  └──────────┘  └──────────────┘  └──────────┘  └──────────┘
```

**分区演进（零停机）**：

```sql
-- 迁移完成后，根据业务需求进行分区演进

-- 原始分区: days(order_time)
-- 新需求: 近期数据按小时分区（查询更快），历史数据保持天分区

-- 添加小时分区（对新写入数据生效）
ALTER TABLE lakehouse.db.orders_iceberg
    ADD PARTITION FIELD hours(order_time);

-- 删除天分区（不影响已有数据文件的分区结构）
ALTER TABLE lakehouse.db.orders_iceberg
    DROP PARTITION FIELD days(order_time);

-- 验证分区规范
SELECT * FROM lakehouse.db.orders_iceberg.partitions;

-- 查询时引擎自动处理混合分区
-- ✅ 旧数据 → 按天裁剪
-- ✅ 新数据 → 按小时裁剪
SELECT * FROM lakehouse.db.orders_iceberg
WHERE order_time BETWEEN '2024-06-15 08:00:00' AND '2024-06-15 10:00:00';
```

### 6.3 性能对比

迁移前后的实际性能对比（基于50亿条数据、30TB存储的生产环境测试）：

| 对比维度 | Hive (迁移前) | Iceberg (迁移后) | 提升幅度 |
|---------|-------------|-----------------|---------|
| **点查询延迟** | 45秒 | 8秒 | 5.6倍 |
| **范围扫描（1天）** | 120秒 | 25秒 | 4.8倍 |
| **聚合查询（1月）** | 600秒 | 180秒 | 3.3倍 |
| **写入延迟（500万行）** | 300秒 | 180秒 | 1.7倍 |
| **存储大小** | 30TB | 22TB（zstd压缩） | 节省27% |
| **添加列** | 需重写30TB（48小时） | 秒级DDL | 极大提升 |
| **修改分区** | 需重写30TB（48小时） | 秒级DDL | 极大提升 |
| **并发写入** | ❌ 经常冲突失败 | ✅ 乐观并发控制 | 质变 |
| **Time Travel** | ❌ 不支持 | ✅ 任意历史版本 | 质变 |
| **数据回溯** | 依赖HDFS快照 | ✅ 秒级回滚 | 质变 |

**查询性能提升的关键因素**：

```
为什么Iceberg查询更快？

  1. 列统计信息裁剪 (Column-level Min/Max Statistics)
  ┌────────────────────────────────────────────────────────┐
  │ Manifest File 中记录每个数据文件的列统计信息:           │
  │   file: data-001.parquet                               │
  │     price: min=9.99, max=999.00, null_count=0          │
  │     customer_id: min=1, max=500, null_count=0          │
  │                                                        │
  │ 查询 WHERE price > 5000                                │
  │ → 跳过 data-001.parquet (max=999 < 5000) ✅           │
  └────────────────────────────────────────────────────────┘

  2. 分区裁剪 (Partition Pruning)
  ┌────────────────────────────────────────────────────────┐
  │ Manifest List 中记录每个Manifest覆盖的分区范围:         │
  │   manifest-001: partition day in [2024-06-01, 2024-06-15]│
  │   manifest-002: partition day in [2024-06-16, 2024-06-30]│
  │                                                        │
  │ 查询 WHERE order_time = '2024-06-20'                   │
  │ → 跳过 manifest-001 (不包含6月20日) ✅                 │
  └────────────────────────────────────────────────────────┘

  3. 文件裁剪效果对比
  Hive:    查询扫描 3000 个文件 / 读取 500GB
  Iceberg: 查询扫描  150 个文件 / 读取  25GB  (裁剪95%)
```

---

## 7. 技术选型与最佳实践

### 7.1 选型决策

```
数据湖表格式选型决策树

                        开始选型
                          │
                          ↓
              ┌───────────────────────┐
              │ 是否使用Databricks？   │
              └───────┬───────┬───────┘
                 是    │       │  否
                      ↓       ↓
            ┌──────────┐  ┌──────────────────────┐
            │Delta Lake│  │ 是否需要高频Upsert    │
            │(首选)    │  │ (CDC / 近实时摄入)?   │
            └──────────┘  └───────┬───────┬───────┘
                             是   │       │  否
                                  ↓       ↓
                       ┌───────────┐  ┌──────────────────────┐
                       │Apache Hudi│  │ 是否需要分区演进 /    │
                       │(首选)     │  │ 多引擎互操作?        │
                       └───────────┘  └───────┬───────┬──────┘
                                         是   │       │  否
                                              ↓       ↓
                                  ┌──────────────┐ ┌────────────┐
                                  │Apache Iceberg│ │三者均可选择 │
                                  │(首选)        │ │按团队熟悉度 │
                                  └──────────────┘ └────────────┘
```

**按使用场景推荐**：

| 使用场景 | 推荐方案 | 理由 |
|---------|---------|------|
| **大规模数据湖分析** | Apache Iceberg | 最好的分区演进、多引擎支持、厂商中立 |
| **Databricks平台** | Delta Lake | 原生集成最优，Unity Catalog一站式治理 |
| **CDC实时入湖** | Apache Hudi | Record-level索引、增量查询、低延迟Upsert |
| **流批一体** | Apache Iceberg / Hudi | Flink集成成熟 |
| **多云多引擎** | Apache Iceberg | REST Catalog标准、Trino/Spark/Flink均原生支持 |
| **数据仓库迁移** | Apache Iceberg | In-place migration从Hive最成熟 |
| **近实时数仓** | Apache Hudi (MoR) | 分钟级延迟、增量拉取、Compaction |
| **机器学习特征存储** | Delta Lake / Iceberg | Time Travel、Schema演进、版本管理 |

**引擎兼容性矩阵**：

| 引擎 | Iceberg | Delta Lake | Hudi |
|------|---------|-----------|------|
| **Apache Spark** | ✅ 完整 | ✅ 原生最优 | ✅ 完整 |
| **Apache Flink** | ✅ 完整 | ✅ Connector | ✅ 完整 |
| **Trino** | ✅ 原生最优 | ✅ Connector | ✅ Connector |
| **PrestoDB** | ✅ 原生 | ❌ 有限 | ✅ Connector |
| **Dremio** | ✅ 原生最优 | ❌ 不支持 | ❌ 不支持 |
| **StarRocks** | ✅ 外表 | ✅ 外表 | ✅ 外表 |
| **ClickHouse** | ✅ 外表 | ✅ 外表 | ❌ 有限 |
| **AWS Athena** | ✅ 原生 | ❌ 不支持 | ❌ 不支持 |
| **BigQuery** | ✅ 外表 | ❌ 不支持 | ❌ 不支持 |

### 7.2 最佳实践

**1) Compaction策略（小文件合并）**

小文件问题是数据湖性能的头号杀手。合理的Compaction策略至关重要。

```
小文件问题

  流式写入产生大量小文件:
  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
  │ 1MB  │ │ 2MB  │ │ 0.5MB│ │ 3MB  │ │ 1MB  │ │ 0.8MB│  ...
  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘
  → 每个文件都需要一次IO操作，100个小文件 = 100次IO

  Compaction后:
  ┌──────────────────────────────┐ ┌──────────────────────────────┐
  │          512MB               │ │          512MB               │
  └──────────────────────────────┘ └──────────────────────────────┘
  → 2次IO操作，查询性能提升50倍+
```

```sql
-- Iceberg Compaction
CALL lakehouse.system.rewrite_data_files(
    table   => 'db.orders',
    strategy => 'binpack',        -- 或 'sort'
    options  => MAP(
        'target-file-size-bytes', '536870912',   -- 目标512MB
        'min-file-size-bytes',    '67108864',    -- 最小64MB
        'max-file-size-bytes',    '1073741824',  -- 最大1GB
        'min-input-files',        '5',           -- 至少5个文件才合并
        'partial-progress.enabled', 'true',      -- 大表分批合并
        'partial-progress.max-commits', '10'
    )
);

-- Delta Lake Compaction
OPTIMIZE orders
WHERE dt >= '2024-06-01'
ZORDER BY (customer_id);

-- Hudi Compaction (MoR表)
-- 内联Compaction（写入时触发）
-- hoodie.compact.inline=true
-- hoodie.compact.inline.max.delta.commits=5

-- 异步Compaction（独立作业）
CALL run_compaction(table => 'orders_mor', op => 'schedule');
CALL run_compaction(table => 'orders_mor', op => 'execute');
```

**2) Snapshot过期策略**

```sql
-- Iceberg: 清理过期快照
CALL lakehouse.system.expire_snapshots(
    table      => 'db.orders',
    older_than => TIMESTAMP '2024-06-08 00:00:00',  -- 保留7天
    retain_last => 100,       -- 至少保留最近100个快照
    stream_results => true    -- 流式返回结果（避免内存溢出）
);

-- Delta Lake: VACUUM清理旧文件
VACUUM orders RETAIN 168 HOURS;   -- 保留7天

-- Hudi: Clean操作
-- 自动清理由 hoodie.cleaner.policy 控制
-- KEEP_LATEST_COMMITS: 保留最近N个commit
-- KEEP_LATEST_FILE_VERSIONS: 保留最近N个文件版本
```

**3) 元数据表维护**

```sql
-- Iceberg: 清理孤立文件（数据文件存在但无元数据引用）
CALL lakehouse.system.remove_orphan_files(
    table      => 'db.orders',
    older_than => TIMESTAMP '2024-06-10 00:00:00',
    dry_run    => true   -- 先干跑确认，再实际删除
);

-- Iceberg: 重写Manifest文件（优化元数据结构）
CALL lakehouse.system.rewrite_manifests(
    table            => 'db.orders',
    use_caching      => true
);
```

**4) 文件大小调优指南**

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| **目标文件大小** | 256MB - 1GB | 太小增加IO次数，太大降低并行度 |
| **分析负载** | 512MB - 1GB | 大文件利于顺序扫描 |
| **流式负载** | 128MB - 256MB | 小文件降低写入延迟 |
| **混合负载** | 256MB - 512MB | 平衡读写性能 |
| **Parquet Row Group** | 128MB | 与Parquet内部Row Group对齐 |
| **最小文件阈值** | 目标大小的75% | 低于此值触发合并 |

**5) 运维检查清单**

```
数据湖运维日常检查清单

日常 (每天):
  □ 检查小文件数量（每个分区的文件数）
  □ 监控写入延迟和成功率
  □ 检查Compaction任务是否正常完成 (Hudi MoR)
  □ 确认增量ETL管道checkpoint正常推进

周常 (每周):
  □ 执行Compaction / OPTIMIZE（分析型表）
  □ 清理过期快照（保留7-30天）
  □ 检查存储用量趋势
  □ 审查慢查询日志，识别缺失分区裁剪的查询

月常 (每月):
  □ 清理孤立文件 (remove_orphan_files)
  □ 重写Manifest文件 (rewrite_manifests)
  □ 评估分区方案是否需要调整
  □ 审计数据访问权限
  □ 更新表统计信息
  □ 评估数据生命周期策略（老数据归档或删除）

季度:
  □ 性能基准测试（对比历史指标）
  □ 评估表格式版本升级
  □ 容量规划和成本优化
  □ 数据质量全面审计
```

```bash
#!/bin/bash
# 数据湖健康检查脚本示例

echo "====== 数据湖健康检查 ======"
echo "检查时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. 检查小文件（Iceberg表）
echo "--- 小文件检查 ---"
spark-sql -e "
SELECT partition,
       COUNT(*)              AS file_count,
       SUM(file_size_in_bytes) / 1024 / 1024 / 1024 AS total_size_gb,
       AVG(file_size_in_bytes) / 1024 / 1024          AS avg_file_size_mb,
       MIN(file_size_in_bytes) / 1024 / 1024          AS min_file_size_mb
FROM lakehouse.db.orders.files
GROUP BY partition
HAVING AVG(file_size_in_bytes) < 67108864   -- 平均小于64MB的分区
ORDER BY file_count DESC
LIMIT 20;
"

# 2. 检查快照数量
echo ""
echo "--- 快照数量检查 ---"
spark-sql -e "
SELECT COUNT(*) AS snapshot_count,
       MIN(committed_at) AS oldest_snapshot,
       MAX(committed_at) AS newest_snapshot
FROM lakehouse.db.orders.snapshots;
"

# 3. 检查表总大小
echo ""
echo "--- 存储用量 ---"
spark-sql -e "
SELECT SUM(file_size_in_bytes) / 1024 / 1024 / 1024 AS total_size_gb,
       COUNT(*) AS total_files,
       SUM(record_count) AS total_records
FROM lakehouse.db.orders.files;
"

echo ""
echo "====== 检查完成 ======"
```

---

> **总结**：数据湖表格式（Iceberg、Delta Lake、Hudi）为传统数据湖带来了数仓级别的
> 事务保障、Schema管理和时间旅行能力。选择哪种表格式取决于具体场景：Iceberg适合多引擎、
> 多云环境；Delta Lake适合Databricks生态；Hudi适合高频Upsert和近实时场景。
> 无论选择哪种，都需要做好Compaction、快照清理和文件大小调优等日常运维工作。

---

> 返回目录: [README.md](README.md) | 上一篇: [05_hadoop_hdfs.md](05_hadoop_hdfs.md)
