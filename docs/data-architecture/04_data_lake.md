# 数据湖架构设计

## 目录
- [概述](#概述)
- [Delta Lake详解](#delta-lake详解)
- [Apache Iceberg详解](#apache-iceberg详解)
- [Apache Hudi详解](#apache-hudi详解)
- [表格式对比](#表格式对比)
- [数据湖vs数据仓库](#数据湖vs数据仓库)
- [实战案例](#实战案例)

## 概述

### 数据湖架构演进

```
传统数据仓库        数据湖1.0          数据湖2.0 (Lakehouse)
┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐
│   Oracle    │    │    HDFS     │    │   Delta Lake       │
│   Teradata  │    │ 原始文件存储 │    │   Iceberg          │
│   结构化     │ -> │ Schema-on-   │ -> │   Hudi             │
│   ETL复杂    │    │ Read        │    │   ACID事务         │
│   扩展困难   │    │ 无事务支持   │    │   时间旅行         │
└─────────────┘    └─────────────┘    └─────────────────────┘
     昂贵              灵活但             最佳实践
     刚性           缺乏治理           BI + ML统一
```

### Lakehouse架构

```
┌──────────────────────────────────────────────────────────────┐
│                      应用层                                   │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│  BI工具  │ ML训练   │ 实时分析  │ 数据科学  │  流式处理        │
│ Tableau  │ MLflow   │ Presto   │ Jupyter  │  Flink          │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴─────┬───────────┘
     └──────────┴──────────┴──────────┴───────────┘
                          │
┌─────────────────────────▼─────────────────────────────────────┐
│                   查询引擎层                                   │
├──────────┬──────────┬──────────┬──────────┬──────────────────┤
│  Spark   │  Presto  │  Trino   │  Hive    │   Flink SQL      │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────────────┘
     └──────────┴──────────┴──────────┴──────────┘
                          │
┌─────────────────────────▼─────────────────────────────────────┐
│                 表格式层 (Metadata Layer)                      │
├──────────────────┬────────────────┬───────────────────────────┤
│   Delta Lake     │  Apache Iceberg │    Apache Hudi          │
│   ┌───────────┐  │  ┌───────────┐ │   ┌───────────┐          │
│   │ Transaction│  │  │ Metadata  │ │   │ Timeline  │          │
│   │ Log       │  │  │ Tree      │ │   │ Service   │          │
│   └───────────┘  │  └───────────┘ │   └───────────┘          │
└──────┬───────────┴────────┬───────┴──────────┬────────────────┘
       │                    │                  │
┌──────▼────────────────────▼──────────────────▼────────────────┐
│                     存储层                                     │
├──────────────────┬────────────────┬───────────────────────────┤
│  对象存储         │  HDFS          │   本地文件系统             │
│  S3/OSS/COS     │  Hadoop        │   Local FS                │
│  ┌────┬────┬───┐│  ┌────┬────┐   │   ┌────┬────┐             │
│  │Par-│ORC │Avro││  │Par-│ORC │   │   │Par-│JSON│             │
│  │quet│    │    ││  │quet│    │   │   │quet│    │             │
│  └────┴────┴───┘│  └────┴────┘   │   └────┴────┘             │
└──────────────────┴────────────────┴───────────────────────────┘
```

## Delta Lake详解

### Delta Lake架构

```
┌──────────────────────────────────────────────────────────────┐
│                    Delta Lake Architecture                   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  应用层                                                        │
│  ┌──────────┬──────────┬──────────┬──────────┐              │
│  │ Spark SQL│ PySpark  │ Delta DML│ Streaming│              │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┘              │
│       └──────────┴──────────┴──────────┘                     │
│                      │                                        │
│  ┌───────────────────▼────────────────────────────────┐      │
│  │         Delta Lake Transaction Layer               │      │
│  │  ┌──────────────────────────────────────────────┐ │      │
│  │  │  Transaction Log (_delta_log/)               │ │      │
│  │  │  ┌─────┬─────┬─────┬─────┬─────┬─────┐      │ │      │
│  │  │  │  0  │  1  │  2  │ ... │ 10  │ ... │      │ │      │
│  │  │  │.json│.json│.json│     │.chk │     │      │ │      │
│  │  │  └─────┴─────┴─────┴─────┴─────┴─────┘      │ │      │
│  │  │  Atomic Commits, Version Control             │ │      │
│  │  └──────────────────────────────────────────────┘ │      │
│  │                                                    │      │
│  │  ┌──────────────────────────────────────────────┐ │      │
│  │  │  Optimistic Concurrency Control              │ │      │
│  │  │  - Read: Snapshot Isolation                  │ │      │
│  │  │  - Write: Conflict Detection                 │ │      │
│  │  └──────────────────────────────────────────────┘ │      │
│  └────────────────────────────────────────────────────┘      │
│                      │                                        │
│  ┌───────────────────▼────────────────────────────────┐      │
│  │           Data Files (Parquet)                     │      │
│  │  ┌──────┬──────┬──────┬──────┬──────┐            │      │
│  │  │part-0│part-1│part-2│part-3│ ...  │            │      │
│  │  │.parq │.parq │.parq │.parq │      │            │      │
│  │  └──────┴──────┴──────┴──────┴──────┘            │      │
│  └────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

### Delta Lake核心特性

#### 1. ACID事务

```scala
import org.apache.spark.sql.SparkSession
import io.delta.tables._

val spark = SparkSession.builder()
  .appName("DeltaLakeExample")
  .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
  .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
  .getOrCreate()

import spark.implicits._

// 创建Delta表
val data = Seq(
  (1, "Alice", 1000),
  (2, "Bob", 2000),
  (3, "Charlie", 3000)
).toDF("id", "name", "amount")

data.write
  .format("delta")
  .mode("overwrite")
  .save("/delta/users")

// ACID更新操作
val deltaTable = DeltaTable.forPath(spark, "/delta/users")

// 1. UPDATE
deltaTable.update(
  condition = expr("id = 1"),
  set = Map("amount" -> expr("amount + 500"))
)

// 2. DELETE
deltaTable.delete(condition = expr("amount < 1500"))

// 3. MERGE (UPSERT)
val updates = Seq(
  (1, "Alice", 1600),  // 更新
  (4, "David", 4000)   // 插入
).toDF("id", "name", "amount")

deltaTable.alias("target")
  .merge(
    updates.alias("source"),
    "target.id = source.id"
  )
  .whenMatched()
  .updateAll()
  .whenNotMatched()
  .insertAll()
  .execute()

// 查询结果
spark.read.format("delta").load("/delta/users").show()
/*
+---+-------+------+
| id|   name|amount|
+---+-------+------+
|  1|  Alice|  1600|
|  3|Charlie|  3000|
|  4|  David|  4000|
+---+-------+------+
*/
```

#### 2. 时间旅行

```scala
// 按版本查询
spark.read
  .format("delta")
  .option("versionAsOf", 0)
  .load("/delta/users")
  .show()

// 按时间戳查询
spark.read
  .format("delta")
  .option("timestampAsOf", "2026-02-01 10:00:00")
  .load("/delta/users")
  .show()

// 查看历史版本
deltaTable.history().show()
/*
+-------+-------------------+------+--------+---------+--------------------+
|version|          timestamp|userId|userName|operation|       operationMetrics|
+-------+-------------------+------+--------+---------+--------------------+
|      3|2026-02-07 14:30:00|  user1|   alice|    MERGE|{numTargetRowsIns...|
|      2|2026-02-07 14:20:00|  user1|   alice|   DELETE|{numDeletedRows: 1}  |
|      1|2026-02-07 14:10:00|  user1|   alice|   UPDATE|{numUpdatedRows: 1}  |
|      0|2026-02-07 14:00:00|  user1|   alice|    WRITE|{numFiles: 3}        |
+-------+-------------------+------+--------+---------+--------------------+
*/

// 回滚到指定版本
deltaTable.restoreToVersion(1)
```

#### 3. Schema演化

```scala
// 自动Schema演化
val newData = Seq(
  (5, "Eve", 5000, "eve@email.com")  // 新增email列
).toDF("id", "name", "amount", "email")

newData.write
  .format("delta")
  .mode("append")
  .option("mergeSchema", "true")  // 启用Schema合并
  .save("/delta/users")

// 显式Schema演化
deltaTable.toDF
  .withColumn("email", lit(null).cast("string"))
  .write
  .format("delta")
  .mode("overwrite")
  .option("overwriteSchema", "true")
  .save("/delta/users")
```

#### 4. 数据压缩与优化

```scala
// OPTIMIZE - 合并小文件
deltaTable.optimize()
  .executeCompaction()

// Z-ORDER - 多维度聚类
deltaTable.optimize()
  .where("date >= '2026-01-01'")
  .executeZOrderBy("city", "category")

// VACUUM - 清理过期文件
deltaTable.vacuum(168)  // 保留7天的历史版本

// 查看表详情
spark.sql("DESCRIBE DETAIL delta.`/delta/users`").show()

// 查看表统计信息
spark.sql("DESCRIBE EXTENDED delta.`/delta/users`").show()
```

### Delta Lake实战：CDC数据湖

```python
from delta import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# 创建Spark会话
spark = SparkSession.builder \
    .appName("CDC_DeltaLake") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# 读取CDC流数据
cdc_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka1:9092") \
    .option("subscribe", "cdc.orders") \
    .option("startingOffsets", "latest") \
    .load()

# 解析CDC事件
parsed_stream = cdc_stream \
    .selectExpr("CAST(value AS STRING) as json") \
    .select(from_json(col("json"), schema).alias("data")) \
    .select("data.*")

# CDC处理逻辑
def upsert_to_delta(batch_df, batch_id):
    """
    处理CDC变更数据
    op: c=create, u=update, d=delete, r=read
    """
    # 创建临时视图
    batch_df.createOrReplaceTempView("updates")

    # 获取Delta表
    delta_table = DeltaTable.forPath(spark, "/delta/orders")

    # 分别处理INSERT/UPDATE/DELETE

    # 1. DELETE操作
    delete_df = batch_df.filter(col("op") == "d")
    if delete_df.count() > 0:
        delta_table.alias("target") \
            .merge(
                delete_df.alias("source"),
                "target.order_id = source.before.order_id"
            ) \
            .whenMatchedDelete() \
            .execute()

    # 2. INSERT/UPDATE操作
    upsert_df = batch_df.filter(col("op").isin("c", "u", "r"))
    if upsert_df.count() > 0:
        delta_table.alias("target") \
            .merge(
                upsert_df.select("after.*").alias("source"),
                "target.order_id = source.order_id"
            ) \
            .whenMatchedUpdateAll() \
            .whenNotMatchedInsertAll() \
            .execute()

    print(f"Batch {batch_id} processed successfully")

# 启动流式写入
query = parsed_stream.writeStream \
    .foreachBatch(upsert_to_delta) \
    .outputMode("update") \
    .option("checkpointLocation", "/delta/checkpoints/orders") \
    .start()

query.awaitTermination()
```

## Apache Iceberg详解

### Iceberg架构

```
┌──────────────────────────────────────────────────────────────┐
│                  Apache Iceberg Architecture                 │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  查询引擎                                                      │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┐        │
│  │  Spark  │  Flink  │  Presto │  Trino  │   Hive  │        │
│  └────┬────┴────┬────┴────┬────┴────┬────┴────┬────┘        │
│       └─────────┴─────────┴─────────┴─────────┘              │
│                         │                                     │
│  ┌──────────────────────▼──────────────────────────────┐     │
│  │         Iceberg Table API                           │     │
│  └──────────────────────┬──────────────────────────────┘     │
│                         │                                     │
│  ┌──────────────────────▼──────────────────────────────┐     │
│  │              Metadata Tree                          │     │
│  │                                                      │     │
│  │  当前元数据 (metadata.json)                          │     │
│  │  ┌────────────────────────────────────────────┐    │     │
│  │  │ schema, partition-spec, sort-order         │    │     │
│  │  │ current-snapshot-id: 12345                 │    │     │
│  │  │ snapshots: [...]                           │    │     │
│  │  └──────────────────┬─────────────────────────┘    │     │
│  │                     │                               │     │
│  │  快照 (Snapshot)    ▼                               │     │
│  │  ┌────────────────────────────────────────────┐    │     │
│  │  │ snapshot-id: 12345                         │    │     │
│  │  │ timestamp: 2026-02-07T14:30:00            │    │     │
│  │  │ manifest-list: s3://bucket/snap-12345.avro│    │     │
│  │  └──────────────────┬─────────────────────────┘    │     │
│  │                     │                               │     │
│  │  清单列表            ▼                               │     │
│  │  ┌────────────────────────────────────────────┐    │     │
│  │  │ Manifest List (snap-12345.avro)           │    │     │
│  │  │ ┌──────────┬──────────┬──────────┐        │    │     │
│  │  │ │manifest-1│manifest-2│manifest-3│        │    │     │
│  │  │ └────┬─────┴────┬─────┴────┬─────┘        │    │     │
│  │  └──────┼──────────┼──────────┼──────────────┘    │     │
│  │         │          │          │                    │     │
│  │  清单文件▼          ▼          ▼                    │     │
│  │  ┌───────────────────────────────────────────┐    │     │
│  │  │ Manifest Files (manifest-*.avro)          │    │     │
│  │  │ ┌──────┬──────┬──────┬──────┐            │    │     │
│  │  │ │file-1│file-2│file-3│ ...  │            │    │     │
│  │  │ │path  │path  │path  │      │            │    │     │
│  │  │ │stats │stats │stats │      │            │    │     │
│  │  │ └──────┴──────┴──────┴──────┘            │    │     │
│  │  └───────────────────────────────────────────┘    │     │
│  └────────────────────┬──────────────────────────────┘     │
│                       │                                     │
│  ┌────────────────────▼──────────────────────────────┐     │
│  │         Data Files (Parquet/ORC/Avro)             │     │
│  │  ┌──────┬──────┬──────┬──────┬──────┐            │     │
│  │  │part-1│part-2│part-3│part-4│ ...  │            │     │
│  │  └──────┴──────┴──────┴──────┴──────┘            │     │
│  └────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

### Iceberg核心特性

#### 1. 表创建与管理

```sql
-- 使用Spark SQL创建Iceberg表
CREATE TABLE orders (
    order_id BIGINT,
    user_id BIGINT,
    product_id BIGINT,
    quantity INT,
    price DECIMAL(10,2),
    order_date DATE,
    order_status STRING
)
USING iceberg
PARTITIONED BY (days(order_date))
TBLPROPERTIES (
    'write.format.default'='parquet',
    'write.parquet.compression-codec'='snappy',
    'commit.retry.num-retries'='3'
);

-- 插入数据
INSERT INTO orders VALUES
(1, 101, 201, 2, 99.99, DATE '2026-02-07', 'paid'),
(2, 102, 202, 1, 199.99, DATE '2026-02-07', 'pending');

-- 查询表快照
SELECT * FROM orders.snapshots;

-- 查询表历史
SELECT * FROM orders.history;

-- 查询表文件
SELECT * FROM orders.files;
```

#### 2. 时间旅行与回滚

```sql
-- 按快照ID查询
SELECT * FROM orders VERSION AS OF 3821550127947089987;

-- 按时间戳查询
SELECT * FROM orders TIMESTAMP AS OF '2026-02-07 10:00:00';

-- 增量读取
SELECT * FROM orders
WHERE _file_timestamp > TIMESTAMP '2026-02-06 00:00:00';

-- 回滚到指定快照
CALL catalog_name.system.rollback_to_snapshot('orders', 3821550127947089987);

-- 回滚到指定时间
CALL catalog_name.system.rollback_to_timestamp('orders', TIMESTAMP '2026-02-06');
```

#### 3. Schema演化

```sql
-- 添加列
ALTER TABLE orders ADD COLUMN email STRING;

-- 删除列
ALTER TABLE orders DROP COLUMN email;

-- 重命名列
ALTER TABLE orders RENAME COLUMN order_status TO status;

-- 修改列类型 (向上兼容)
ALTER TABLE orders ALTER COLUMN quantity TYPE BIGINT;

-- 更新列注释
ALTER TABLE orders ALTER COLUMN price COMMENT '订单价格(单位:元)';
```

#### 4. 分区演化

```sql
-- 更改分区策略 (无需重写数据)
ALTER TABLE orders
SET PARTITION SPEC (bucket(16, user_id), days(order_date));

-- 查看分区演化历史
SELECT * FROM orders.partition_specs;
```

#### 5. 表维护

```scala
import org.apache.iceberg.spark.actions.SparkActions

val spark: SparkSession = ...

// 1. 合并小文件
SparkActions
  .get(spark)
  .rewriteDataFiles(
    spark.table("orders")
  )
  .option("target-file-size-bytes", "536870912") // 512MB
  .option("min-file-size-bytes", "134217728")    // 128MB
  .execute()

// 2. 删除过期快照
SparkActions
  .get(spark)
  .expireSnapshots(
    spark.table("orders")
  )
  .expireOlderThan(System.currentTimeMillis() - 7 * 24 * 60 * 60 * 1000) // 7天
  .execute()

// 3. 删除孤儿文件
SparkActions
  .get(spark)
  .deleteOrphanFiles(
    spark.table("orders")
  )
  .olderThan(System.currentTimeMillis() - 3 * 24 * 60 * 60 * 1000) // 3天
  .execute()

// 4. 重写Manifest文件
SparkActions
  .get(spark)
  .rewriteManifests(
    spark.table("orders")
  )
  .execute()
```

### Iceberg实战：多引擎访问

```python
# Spark写入
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.sql.catalog.my_catalog", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.my_catalog.type", "hadoop") \
    .config("spark.sql.catalog.my_catalog.warehouse", "s3://bucket/warehouse") \
    .getOrCreate()

df = spark.read.parquet("s3://source/orders/")
df.writeTo("my_catalog.db.orders").create()

# Flink读写
from pyflink.table import EnvironmentSettings, TableEnvironment

env_settings = EnvironmentSettings.in_streaming_mode()
table_env = TableEnvironment.create(env_settings)

table_env.execute_sql("""
    CREATE CATALOG my_catalog WITH (
        'type' = 'iceberg',
        'catalog-type' = 'hadoop',
        'warehouse' = 's3://bucket/warehouse'
    )
""")

table_env.execute_sql("""
    INSERT INTO my_catalog.db.orders
    SELECT * FROM kafka_source
""")

# Presto查询
# 在Presto配置catalog/iceberg.properties:
# connector.name=iceberg
# iceberg.catalog.type=hadoop
# iceberg.catalog.warehouse=s3://bucket/warehouse

# 然后在Presto中查询:
# SELECT * FROM iceberg.db.orders WHERE order_date = DATE '2026-02-07';
```

## Apache Hudi详解

### Hudi架构

```
┌──────────────────────────────────────────────────────────────┐
│                  Apache Hudi Architecture                    │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  应用层                                                        │
│  ┌─────────┬─────────┬─────────┬─────────┐                  │
│  │  Spark  │  Flink  │  Presto │   Hive  │                  │
│  └────┬────┴────┬────┴────┬────┴────┬────┘                  │
│       └─────────┴─────────┴─────────┘                        │
│                    │                                          │
│  ┌─────────────────▼──────────────────────────────────┐      │
│  │            Hudi Table API                          │      │
│  │  ┌──────────────────────────────────────────┐     │      │
│  │  │  Table Types                             │     │      │
│  │  │  ┌────────────┐    ┌───────────────┐    │     │      │
│  │  │  │ Copy-on-   │    │ Merge-on-     │    │     │      │
│  │  │  │ Write(CoW) │    │ Read (MoR)    │    │     │      │
│  │  │  └────────────┘    └───────────────┘    │     │      │
│  │  └──────────────────────────────────────────┘     │      │
│  └─────────────────┬──────────────────────────────────┘      │
│                    │                                          │
│  ┌─────────────────▼──────────────────────────────────┐      │
│  │         Timeline Service                           │      │
│  │  ┌──────────────────────────────────────────┐     │      │
│  │  │  Timeline (时间轴)                        │     │      │
│  │  │  ┌────┬────┬────┬────┬────┬────┬────┐   │     │      │
│  │  │  │ C1 │ C2 │ C3 │ C4 │ C5 │ C6 │... │   │     │      │
│  │  │  │.cmt│.cmt│.cmt│.cmt│.cmt│.cmt│    │   │     │      │
│  │  │  └────┴────┴────┴────┴────┴────┴────┘   │     │      │
│  │  │  Instant: 提交时间点                     │     │      │
│  │  └──────────────────────────────────────────┘     │      │
│  │                                                    │      │
│  │  ┌──────────────────────────────────────────┐     │      │
│  │  │  Hoodie.properties (表元数据)             │     │      │
│  │  │  - Table Type (CoW/MoR)                  │     │      │
│  │  │  - Record Key, Partition Path            │     │      │
│  │  │  - Precombine Field                      │     │      │
│  │  └──────────────────────────────────────────┘     │      │
│  └─────────────────┬──────────────────────────────────┘      │
│                    │                                          │
│  ┌─────────────────▼──────────────────────────────────┐      │
│  │              Data Files                            │      │
│  │                                                     │      │
│  │  CoW模式:                                           │      │
│  │  ┌──────────┬──────────┬──────────┐               │      │
│  │  │ Parquet  │ Parquet  │ Parquet  │               │      │
│  │  │  File    │  File    │  File    │               │      │
│  │  └──────────┴──────────┴──────────┘               │      │
│  │                                                     │      │
│  │  MoR模式:                                           │      │
│  │  ┌──────────────────────────────────────────┐     │      │
│  │  │ Base Files (Parquet)                     │     │      │
│  │  │ ┌──────┬──────┬──────┐                   │     │      │
│  │  │ │base-1│base-2│base-3│                   │     │      │
│  │  │ └──────┴──────┴──────┘                   │     │      │
│  │  │                                           │     │      │
│  │  │ Log Files (Avro)                         │     │      │
│  │  │ ┌──────┬──────┬──────┐                   │     │      │
│  │  │ │ log-1│ log-2│ log-3│                   │     │      │
│  │  │ └──────┴──────┴──────┘                   │     │      │
│  │  └──────────────────────────────────────────┘     │      │
│  └────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

### Hudi表类型对比

#### Copy-on-Write (CoW)

```scala
import org.apache.hudi.QuickstartUtils._
import org.apache.hudi.DataSourceWriteOptions._
import org.apache.hudi.config.HoodieWriteConfig._

val spark = SparkSession.builder()
  .appName("HudiCoWExample")
  .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
  .getOrCreate()

import spark.implicits._

// 创建CoW表
val tableName = "hudi_cow_orders"
val basePath = "s3://bucket/hudi/orders"

val inserts = Seq(
  (1, "Alice", 100, "2026-02-07"),
  (2, "Bob", 200, "2026-02-07"),
  (3, "Charlie", 300, "2026-02-07")
).toDF("id", "name", "amount", "date")

inserts.write
  .format("hudi")
  .option(RECORDKEY_FIELD.key, "id")
  .option(PARTITIONPATH_FIELD.key, "date")
  .option(PRECOMBINE_FIELD.key, "amount")
  .option(TBL_NAME.key, tableName)
  .option("hoodie.datasource.write.table.type", "COPY_ON_WRITE")
  .mode("overwrite")
  .save(basePath)

// 更新数据 (Copy-on-Write: 重写整个Parquet文件)
val updates = Seq(
  (1, "Alice", 150, "2026-02-07"),  // 更新
  (4, "David", 400, "2026-02-07")   // 新增
).toDF("id", "name", "amount", "date")

updates.write
  .format("hudi")
  .option(RECORDKEY_FIELD.key, "id")
  .option(PARTITIONPATH_FIELD.key, "date")
  .option(PRECOMBINE_FIELD.key, "amount")
  .option(TBL_NAME.key, tableName)
  .mode("append")
  .save(basePath)

// 查询数据 (快速读取)
spark.read.format("hudi").load(basePath).show()

/*
特点:
- 写入慢: 每次更新都重写整个文件
- 读取快: 直接读取Parquet文件
- 适用场景: 读多写少
*/
```

#### Merge-on-Read (MoR)

```scala
// 创建MoR表
val tableNameMoR = "hudi_mor_orders"
val basePathMoR = "s3://bucket/hudi/orders_mor"

inserts.write
  .format("hudi")
  .option(RECORDKEY_FIELD.key, "id")
  .option(PARTITIONPATH_FIELD.key, "date")
  .option(PRECOMBINE_FIELD.key, "amount")
  .option(TBL_NAME.key, tableNameMoR)
  .option("hoodie.datasource.write.table.type", "MERGE_ON_READ")
  .option("hoodie.compact.inline", "false")  // 禁用自动压缩
  .mode("overwrite")
  .save(basePathMoR)

// 更新数据 (Merge-on-Read: 追加到Log文件)
updates.write
  .format("hudi")
  .option(RECORDKEY_FIELD.key, "id")
  .option(PARTITIONPATH_FIELD.key, "date")
  .option(PRECOMBINE_FIELD.key, "amount")
  .option(TBL_NAME.key, tableNameMoR)
  .mode("append")
  .save(basePathMoR)

// Snapshot查询 (合并Base + Log)
spark.read
  .format("hudi")
  .load(basePathMoR)
  .show()

// Read Optimized查询 (只读Base文件)
spark.read
  .format("hudi")
  .option("hoodie.datasource.query.type", "read_optimized")
  .load(basePathMoR)
  .show()

// 增量查询
spark.read
  .format("hudi")
  .option("hoodie.datasource.query.type", "incremental")
  .option("hoodie.datasource.read.begin.instanttime", "20260207000000")
  .load(basePathMoR)
  .show()

/*
特点:
- 写入快: 追加到Log文件
- 读取慢: 需要合并Base + Log
- 适用场景: 写多读少
*/
```

### Hudi表维护

```scala
import org.apache.hudi.client.HoodieSparkCompactor

// 1. 压缩 (Compaction) - 合并Log到Base
spark.read.format("hudi")
  .option("hoodie.compact.inline", "true")
  .option("hoodie.compact.inline.max.delta.commits", "5")
  .load(basePathMoR)

// 手动触发压缩
spark.sql(s"""
  CALL compact('$tableNameMoR', 'schedule')
""")

// 2. 清理 (Cleaning) - 删除旧版本
spark.sql(s"""
  CALL run_clean(table => '$tableNameMoR', retain_commits => 10)
""")

// 3. 聚簇 (Clustering) - 优化数据布局
spark.sql(s"""
  CALL run_clustering(table => '$tableNameMoR', order => 'id')
""")

// 4. 归档 (Archiving) - 归档旧时间轴
// 自动归档,超过50次提交后归档
.option("hoodie.keep.min.commits", "30")
.option("hoodie.keep.max.commits", "50")
```

## 表格式对比

### 功能对比表

```
┌────────────────┬──────────────┬──────────────┬──────────────┐
│    特性        │ Delta Lake   │  Iceberg     │    Hudi      │
├────────────────┼──────────────┼──────────────┼──────────────┤
│ ACID事务       │      ✓       │      ✓       │      ✓       │
│ 时间旅行       │      ✓       │      ✓       │      ✓       │
│ Schema演化     │      ✓       │      ✓       │      ✓       │
│ 分区演化       │      ✗       │      ✓       │      ✗       │
│ 行级更新       │      ✓       │      ✓       │      ✓       │
│ 增量读取       │      ✓       │      ✓       │      ✓       │
│ 多引擎支持     │   中等       │     最好      │     中等      │
│ 流式写入       │      ✓       │      ✓       │      ✓       │
│ 写入性能       │     好       │     好       │    最好       │
│ 读取性能       │     好       │     好       │     中等      │
│ CDC支持        │     好       │     中等      │    最好       │
│ 小文件问题     │   需优化      │   需优化      │  自动处理     │
│ 成熟度         │     高       │     中       │     中       │
│ 社区活跃度     │     高       │     高       │     中       │
└────────────────┴──────────────┴──────────────┴──────────────┘
```

### 选型建议

```
Delta Lake 最适合:
✓ Databricks平台用户
✓ 以Spark为主的工作负载
✓ 需要强ACID保证
✓ 批处理为主的场景

Apache Iceberg 最适合:
✓ 多引擎访问需求 (Spark/Flink/Presto/Trino)
✓ 需要分区演化
✓ 大规模数据湖
✓ 开放标准重视度高

Apache Hudi 最适合:
✓ 近实时数据摄入
✓ CDC场景
✓ 需要增量处理
✓ 流批一体架构
```

## 数据湖vs数据仓库

### 架构对比

```
传统数据仓库 (Data Warehouse)
┌─────────────────────────────────────────────────────────┐
│  应用层: BI工具, 报表                                     │
├─────────────────────────────────────────────────────────┤
│  数据集市层 (Data Marts)                                 │
│  ┌──────────┬──────────┬──────────┬──────────┐         │
│  │   销售    │   财务   │   HR     │  客服    │         │
│  └──────────┴──────────┴──────────┴──────────┘         │
├─────────────────────────────────────────────────────────┤
│  数据仓库层 (DW)                                         │
│  - 星型模型/雪花模型                                      │
│  - 严格Schema                                           │
│  - SQL查询                                              │
├─────────────────────────────────────────────────────────┤
│  ETL层                                                  │
│  - Extract, Transform, Load                            │
│  - 数据清洗、转换、加载                                   │
├─────────────────────────────────────────────────────────┤
│  数据源                                                  │
│  ┌──────────┬──────────┬──────────┐                    │
│  │  OLTP DB │   ERP    │   CRM    │                    │
│  └──────────┴──────────┴──────────┘                    │
└─────────────────────────────────────────────────────────┘

缺点:
- 存储和计算耦合,扩展性差
- Schema刚性,难以适应变化
- ETL流程复杂,延迟高
- 成本高昂


数据湖 (Data Lake)
┌─────────────────────────────────────────────────────────┐
│  应用层                                                  │
│  ┌──────────┬──────────┬──────────┬──────────┐         │
│  │    BI    │ 机器学习  │ 数据科学  │  流式分析│         │
│  └──────────┴──────────┴──────────┴──────────┘         │
├─────────────────────────────────────────────────────────┤
│  计算引擎层                                              │
│  ┌──────────┬──────────┬──────────┬──────────┐         │
│  │  Spark   │  Presto  │   Hive   │  Flink   │         │
│  └──────────┴──────────┴──────────┴──────────┘         │
├─────────────────────────────────────────────────────────┤
│  表格式层 (Lakehouse)                                    │
│  ┌──────────┬──────────┬──────────┐                    │
│  │  Delta   │ Iceberg  │   Hudi   │                    │
│  └──────────┴──────────┴──────────┘                    │
├─────────────────────────────────────────────────────────┤
│  存储层                                                  │
│  ┌────────────────────────────────────────────┐        │
│  │  对象存储 (S3/OSS/COS/ADLS)                │        │
│  │  ┌──────┬──────┬──────┬──────┬──────┐     │        │
│  │  │原始层 │清洗层 │加工层 │应用层 │归档层│     │        │
│  │  │ Raw  │Bronze│Silver│ Gold │Archive│     │        │
│  │  └──────┴──────┴──────┴──────┴──────┘     │        │
│  └────────────────────────────────────────────┘        │
├─────────────────────────────────────────────────────────┤
│  数据源 (ELT)                                            │
│  ┌──────────┬──────────┬──────────┬──────────┐         │
│  │  Database│   API    │   日志   │   IoT    │         │
│  └──────────┴──────────┴──────────┴──────────┘         │
└─────────────────────────────────────────────────────────┘

优点:
- 存储计算分离,弹性扩展
- Schema灵活,支持半结构化/非结构化
- 成本低(对象存储)
- 支持多种工作负载
```

### 选型决策树

```
                开始
                 │
        是否需要实时查询?
         ┌───────┴───────┐
         是              否
         │               │
   数据量 > 100TB?    是否已有数仓?
   ┌─────┴─────┐    ┌─────┴─────┐
   是         否     是          否
   │          │      │           │
 数据湖+     传统     保持       看预算
 Lakehouse  数仓     现状       ┌──┴──┐
   │                           高    低
   │                           │     │
 选择表格式                    数仓   数据湖
 ┌──┴──┐
 │     │
多引擎? Spark为主?
 │      │
 是     否
 │      │
Iceberg Delta
```

## 实战案例

### 案例1：构建实时数据湖

```python
"""
需求:
1. 实时采集业务数据库变更
2. 存储到数据湖
3. 支持SQL分析
4. 支持时间旅行
"""

from pyspark.sql import SparkSession
from delta import *

# 1. 配置Spark
spark = SparkSession.builder \
    .appName("RealtimeDataLake") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# 2. 读取CDC流
cdc_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "mysql.inventory.orders") \
    .option("startingOffsets", "latest") \
    .load()

# 3. 解析并转换
from pyspark.sql.functions import from_json, col

schema = "order_id LONG, user_id LONG, amount DECIMAL(10,2), status STRING, ts TIMESTAMP"

parsed = cdc_stream \
    .selectExpr("CAST(value AS STRING) as json") \
    .select(from_json(col("json"), schema).alias("data")) \
    .select("data.*") \
    .withWatermark("ts", "10 seconds")

# 4. 写入Delta Lake
query = parsed.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "/tmp/checkpoint/orders") \
    .option("mergeSchema", "true") \
    .partitionBy("DATE(ts)") \
    .start("/datalake/orders")

# 5. 实时查询
spark.sql("""
    SELECT
        DATE(ts) as date,
        status,
        COUNT(*) as order_count,
        SUM(amount) as total_amount
    FROM delta.`/datalake/orders`
    WHERE DATE(ts) = CURRENT_DATE()
    GROUP BY DATE(ts), status
""").show()

# 6. 时间旅行
spark.read \
    .format("delta") \
    .option("versionAsOf", 10) \
    .load("/datalake/orders") \
    .show()
```

### 案例2：数据湖分层架构

```
/datalake/
├── bronze/          # 原始层 (Raw Data)
│   ├── orders/      # 原始订单数据
│   ├── users/       # 原始用户数据
│   └── products/    # 原始产品数据
│
├── silver/          # 清洗层 (Cleaned Data)
│   ├── orders/      # 清洗后订单
│   ├── users/       # 清洗后用户
│   └── products/    # 清洗后产品
│
└── gold/            # 应用层 (Business Data)
    ├── fact_orders/ # 订单事实表
    ├── dim_users/   # 用户维度表
    └── agg_daily/   # 日汇总表
```

```python
# Bronze层: 原样存储原始数据
bronze_df.write \
    .format("delta") \
    .mode("append") \
    .save("/datalake/bronze/orders")

# Silver层: 清洗转换
silver_df = spark.read.format("delta").load("/datalake/bronze/orders") \
    .dropDuplicates(["order_id"]) \
    .filter(col("amount") > 0) \
    .withColumn("order_date", to_date(col("created_at")))

silver_df.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("order_date") \
    .save("/datalake/silver/orders")

# Gold层: 业务聚合
gold_df = spark.sql("""
    SELECT
        order_date,
        user_id,
        COUNT(*) as order_count,
        SUM(amount) as total_spent
    FROM delta.`/datalake/silver/orders`
    GROUP BY order_date, user_id
""")

gold_df.write \
    .format("delta") \
    .mode("overwrite") \
    .save("/datalake/gold/user_daily_stats")
```

## 总结

数据湖关键技术选型:

1. **表格式选择**
   - Delta Lake: Databricks生态,成熟稳定
   - Iceberg: 多引擎支持最好,分区演化
   - Hudi: CDC场景最优,写入性能好

2. **架构设计**
   - 分层存储: Bronze/Silver/Gold
   - 计算存储分离: 弹性伸缩
   - 元数据管理: Catalog统一管理

3. **性能优化**
   - 分区策略: 合理分区,避免小文件
   - 数据压缩: Snappy/Gzip/Zstd
   - 文件格式: Parquet列式存储
   - Z-Order/Clustering: 数据排序

4. **数据治理**
   - Schema管理: 版本控制,演化策略
   - 数据质量: 自动化检查
   - 访问控制: 细粒度权限
   - 审计日志: 操作追踪

数据湖 vs 数据仓库:
- 数据湖: 灵活、低成本、支持多种工作负载
- 数据仓库: 性能好、SQL优化、BI分析
- Lakehouse: 融合两者优势,统一架构
