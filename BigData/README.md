# 大数据架构

大数据架构处理PB级数据的存储、计算和分析。

## 目录

1. [批处理](01_batch_processing.md) - Hadoop、Spark
2. [流处理](02_stream_processing.md) - Flink、Kafka Streams
3. [实时数仓](03_realtime_warehouse.md) - Lambda/Kappa架构
4. [OLAP引擎](04_olap.md) - ClickHouse、Druid
5. [Hadoop与HDFS](05_hadoop_hdfs.md) - Hadoop生态、HDFS存储
6. [Hive数据仓库](06_hive.md) - HiveQL、数据仓库
7. [HBase](07_hbase.md) - NoSQL列式数据库
8. [数据采集](08_data_collection.md) - Flume、Logstash、Sqoop、Canal
9. [Presto/Trino](09_presto_trino.md) - 交互式查询引擎
10. [数据湖](10_data_lake.md) - Iceberg、Delta Lake、Hudi
11. [Airflow](11_airflow.md) - 工作流调度

## 大数据技术栈

```
┌────────────────────────────────────────────────────┐
│              大数据技术栈                          │
├────────────────────────────────────────────────────┤
│  数据采集   Flume、Kafka、Logstash、Canal、Sqoop  │
│  数据存储   HDFS、S3、HBase、Cassandra             │
│  数据湖     Iceberg、Delta Lake、Hudi              │
│  批处理     Hadoop MR、Spark、Flink                │
│  流处理     Flink、Storm、Spark Streaming          │
│  数据仓库   Hive、Doris                            │
│  查询引擎   Presto/Trino、ClickHouse               │
│  OLAP       ClickHouse、Doris、StarRocks           │
│  调度       Airflow、Oozie、DolphinScheduler       │
└────────────────────────────────────────────────────┘
```

开始学习 → [01_batch_processing.md](01_batch_processing.md)
