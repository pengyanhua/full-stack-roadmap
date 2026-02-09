# 大数据架构

大数据架构处理PB级数据的存储、计算和分析。

## 目录

1. [批处理](01_batch_processing.md) - Hadoop、Spark
2. [流处理](02_stream_processing.md) - Flink、Kafka Streams
3. [实时数仓](03_realtime_warehouse.md) - Lambda/Kappa架构
4. [OLAP引擎](04_olap.md) - ClickHouse、Druid

## 大数据技术栈

```
┌────────────────────────────────────────────────────┐
│              大数据技术栈                          │
├────────────────────────────────────────────────────┤
│  数据采集   Flume、Kafka、Logstash                │
│  数据存储   HDFS、S3、HBase、Cassandra             │
│  批处理     Hadoop MR、Spark、Flink                │
│  流处理     Flink、Storm、Spark Streaming          │
│  查询引擎   Hive、Presto、ClickHouse               │
│  调度       Airflow、Oozie                         │
└────────────────────────────────────────────────────┘
```

开始学习 → [01_batch_processing.md](01_batch_processing.md)
