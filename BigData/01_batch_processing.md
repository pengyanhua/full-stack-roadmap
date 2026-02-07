# Apache Spark批处理实战

## 1. Spark架构详解

### 1.1 整体架构

```
Spark集群架构
┌─────────────────────────────────────────────────────────┐
│                    Client Application                   │
│                   (Spark Driver)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ SparkContext │→ │ DAG Scheduler│→ │Task Scheduler│  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
         │                    │                    │
         │                    ↓                    ↓
         │          ┌──────────────────────────────┐
         │          │    Cluster Manager           │
         │          │  (YARN/Mesos/Standalone)     │
         │          └──────────────────────────────┘
         │                    │
         ↓                    ↓
┌─────────────────────────────────────────────────────────┐
│                    Worker Nodes                         │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │  Executor 1     │  │  Executor 2     │  ...         │
│  │  ┌───────────┐  │  │  ┌───────────┐  │              │
│  │  │ Task 1.1  │  │  │  │ Task 2.1  │  │              │
│  │  ├───────────┤  │  │  ├───────────┤  │              │
│  │  │ Task 1.2  │  │  │  │ Task 2.2  │  │              │
│  │  ├───────────┤  │  │  ├───────────┤  │              │
│  │  │ Cache     │  │  │  │ Cache     │  │              │
│  │  └───────────┘  │  │  └───────────┘  │              │
│  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

### 1.2 核心组件详解

**Driver进程**：
- 运行应用的main()函数
- 创建SparkContext
- 将用户程序转换为Task
- 调度Task到Executor
- 跟踪Executor状态

**Executor进程**：
- 运行Task并返回结果
- 为应用缓存数据（内存或磁盘）
- 每个应用有独立的Executor

**Cluster Manager**：
- Standalone: Spark自带集群管理器
- YARN: Hadoop资源管理器
- Mesos: Apache Mesos
- Kubernetes: K8s容器编排

### 1.3 执行流程

```
Job执行流程
┌──────────────┐
│ 用户代码     │
│ (Action操作) │
└──────────────┘
       │
       ↓
┌──────────────┐
│ DAG Scheduler│ → 划分Stage (按Shuffle边界)
└──────────────┘
       │
       ↓
┌──────────────┐
│Task Scheduler│ → 生成TaskSet
└──────────────┘
       │
       ↓
┌──────────────┐
│   Executor   │ → 执行Task
└──────────────┘

Stage划分示例：
RDD1 → map → RDD2 → filter → RDD3 [Stage 0]
               ↓
          reduceByKey (Shuffle)
               ↓
         RDD4 → map → RDD5 [Stage 1]
```

## 2. RDD、DataFrame、Dataset对比

### 2.1 三者对比表

| 特性 | RDD | DataFrame | Dataset |
|------|-----|-----------|---------|
| **类型安全** | 编译时检查 | 运行时检查 | 编译时检查 |
| **API风格** | 函数式 | 声明式SQL | 混合 |
| **性能** | 较低 | 高（Catalyst优化） | 高（Catalyst优化） |
| **序列化** | Java序列化 | Tungsten二进制 | Tungsten二进制 |
| **适用场景** | 非结构化数据 | 结构化数据分析 | 类型安全的结构化数据 |
| **Python支持** | 是 | 是 | 否（仅Scala/Java） |

### 2.2 API使用对比

```scala
// Scala示例 - 同一操作的三种实现

// 1. RDD API
val rdd = sc.textFile("users.csv")
  .map(line => line.split(","))
  .filter(arr => arr(2).toInt > 18)
  .map(arr => (arr(0), arr(2).toInt))

// 2. DataFrame API
val df = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("users.csv")
  .filter($"age" > 18)
  .select($"name", $"age")

// 3. Dataset API
case class User(name: String, email: String, age: Int)

val ds = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("users.csv")
  .as[User]
  .filter(_.age > 18)
  .map(u => (u.name, u.age))
```

### 2.3 性能对比

```python
# Python示例 - 性能测试
from pyspark.sql import SparkSession
import time

spark = SparkSession.builder.appName("PerformanceTest").getOrCreate()

# 生成测试数据
df = spark.range(0, 100000000).toDF("id")

# RDD API
start = time.time()
rdd_result = df.rdd.map(lambda row: row.id * 2).filter(lambda x: x > 1000).count()
print(f"RDD Time: {time.time() - start:.2f}s")

# DataFrame API (使用Catalyst优化器)
start = time.time()
df_result = df.selectExpr("id * 2 as doubled").filter("doubled > 1000").count()
print(f"DataFrame Time: {time.time() - start:.2f}s")

# 典型结果：DataFrame快30-40%
```

## 3. 完整WordCount示例

### 3.1 Scala版本

```scala
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

object WordCount {

  def main(args: Array[String]): Unit = {
    // 1. 使用RDD API
    wordCountRDD()

    // 2. 使用DataFrame API
    wordCountDataFrame()

    // 3. 使用Dataset API
    wordCountDataset()
  }

  // 方法1: RDD API (经典方式)
  def wordCountRDD(): Unit = {
    val conf = new SparkConf()
      .setAppName("WordCount-RDD")
      .setMaster("local[*]")

    val sc = new SparkContext(conf)

    try {
      val textRDD = sc.textFile("hdfs://namenode:9000/input/text.txt")

      val wordCounts = textRDD
        .flatMap(line => line.split("\\s+"))  // 分词
        .filter(word => word.nonEmpty)         // 过滤空字符串
        .map(word => (word.toLowerCase, 1))    // 转小写并计数1
        .reduceByKey(_ + _)                    // 聚合
        .sortBy(_._2, ascending = false)       // 按计数降序排序

      // 输出Top 10
      wordCounts.take(10).foreach {
        case (word, count) => println(s"$word: $count")
      }

      // 保存结果
      wordCounts.saveAsTextFile("hdfs://namenode:9000/output/wordcount")

    } finally {
      sc.stop()
    }
  }

  // 方法2: DataFrame API (SQL风格)
  def wordCountDataFrame(): Unit = {
    val spark = SparkSession.builder()
      .appName("WordCount-DataFrame")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    try {
      val df = spark.read.text("hdfs://namenode:9000/input/text.txt")

      // 使用SQL表达式
      val wordCounts = df
        .selectExpr("explode(split(lower(value), '\\\\s+')) as word")
        .filter($"word" =!= "")
        .groupBy("word")
        .count()
        .orderBy($"count".desc)

      wordCounts.show(10)

      // 保存为Parquet格式
      wordCounts.write
        .mode("overwrite")
        .parquet("hdfs://namenode:9000/output/wordcount.parquet")

    } finally {
      spark.stop()
    }
  }

  // 方法3: Dataset API (类型安全)
  def wordCountDataset(): Unit = {
    val spark = SparkSession.builder()
      .appName("WordCount-Dataset")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    try {
      case class WordCount(word: String, count: Long)

      val ds = spark.read
        .textFile("hdfs://namenode:9000/input/text.txt")
        .flatMap(_.split("\\s+"))
        .filter(_.nonEmpty)
        .map(_.toLowerCase)
        .groupByKey(identity)
        .count()
        .map { case (word, count) => WordCount(word, count) }
        .orderBy($"count".desc)

      ds.show(10)

      // 保存为JSON
      ds.write
        .mode("overwrite")
        .json("hdfs://namenode:9000/output/wordcount.json")

    } finally {
      spark.stop()
    }
  }
}
```

### 3.2 Python版本

```python
#!/usr/bin/env python3
"""
Apache Spark WordCount - Python实现
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, lower, col, desc
import sys

def wordcount_rdd(spark):
    """使用RDD API"""
    sc = spark.sparkContext

    # 读取文件
    text_rdd = sc.textFile("hdfs://namenode:9000/input/text.txt")

    # WordCount逻辑
    word_counts = (
        text_rdd
        .flatMap(lambda line: line.split())
        .filter(lambda word: word.strip())
        .map(lambda word: (word.lower(), 1))
        .reduceByKey(lambda a, b: a + b)
        .sortBy(lambda x: x[1], ascending=False)
    )

    # 输出Top 10
    top_10 = word_counts.take(10)
    for word, count in top_10:
        print(f"{word}: {count}")

    # 保存结果
    word_counts.saveAsTextFile("hdfs://namenode:9000/output/wordcount_rdd")

def wordcount_dataframe(spark):
    """使用DataFrame API"""
    # 读取文件
    df = spark.read.text("hdfs://namenode:9000/input/text.txt")

    # WordCount逻辑
    word_counts = (
        df
        .select(explode(split(lower(col("value")), r"\s+")).alias("word"))
        .filter(col("word") != "")
        .groupBy("word")
        .count()
        .orderBy(desc("count"))
    )

    # 显示Top 10
    word_counts.show(10)

    # 保存为Parquet
    word_counts.write.mode("overwrite").parquet(
        "hdfs://namenode:9000/output/wordcount_df.parquet"
    )

    return word_counts

def advanced_wordcount(spark):
    """高级WordCount - 包含停用词过滤"""
    # 停用词列表
    stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'])

    df = spark.read.text("hdfs://namenode:9000/input/text.txt")

    # 广播停用词列表（优化性能）
    broadcast_stopwords = spark.sparkContext.broadcast(stopwords)

    # WordCount with stopwords removal
    word_counts = (
        df
        .select(explode(split(lower(col("value")), r"[^a-z]+")).alias("word"))
        .filter(col("word") != "")
        .filter(~col("word").isin(broadcast_stopwords.value))  # 过滤停用词
        .groupBy("word")
        .count()
        .orderBy(desc("count"))
    )

    word_counts.show(20)

    # 统计信息
    total_words = word_counts.selectExpr("sum(count) as total").first()["total"]
    unique_words = word_counts.count()

    print(f"\n统计信息:")
    print(f"总词数: {total_words}")
    print(f"不重复词数: {unique_words}")

    return word_counts

def main():
    # 创建SparkSession
    spark = SparkSession.builder \
        .appName("WordCount-Python") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

    try:
        print("=== RDD API ===")
        wordcount_rdd(spark)

        print("\n=== DataFrame API ===")
        wordcount_dataframe(spark)

        print("\n=== Advanced WordCount ===")
        advanced_wordcount(spark)

    finally:
        spark.stop()

if __name__ == "__main__":
    main()
```

## 4. 调优参数详解

### 4.1 资源配置参数

```bash
# spark-submit完整配置示例
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  \
  # ===== Driver配置 =====
  --driver-memory 4g \              # Driver内存
  --driver-cores 2 \                # Driver CPU核数
  \
  # ===== Executor配置 =====
  --num-executors 10 \              # Executor数量
  --executor-memory 8g \            # 每个Executor内存
  --executor-cores 4 \              # 每个Executor核数
  \
  # ===== 内存管理 =====
  --conf spark.memory.fraction=0.8 \           # 执行和存储内存占比
  --conf spark.memory.storageFraction=0.5 \    # 存储内存占比
  --conf spark.memory.offHeap.enabled=true \   # 启用堆外内存
  --conf spark.memory.offHeap.size=2g \
  \
  # ===== Shuffle配置 =====
  --conf spark.sql.shuffle.partitions=200 \    # Shuffle分区数
  --conf spark.shuffle.file.buffer=64k \       # Shuffle写缓冲
  --conf spark.reducer.maxSizeInFlight=96m \   # Reduce拉取数据大小
  --conf spark.shuffle.compress=true \         # Shuffle压缩
  --conf spark.shuffle.spill.compress=true \   # Spill压缩
  \
  # ===== 序列化 =====
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.kryoserializer.buffer.max=256m \
  \
  # ===== 并行度 =====
  --conf spark.default.parallelism=200 \       # 默认并行度
  \
  # ===== 动态资源分配 =====
  --conf spark.dynamicAllocation.enabled=true \
  --conf spark.dynamicAllocation.minExecutors=5 \
  --conf spark.dynamicAllocation.maxExecutors=20 \
  --conf spark.dynamicAllocation.initialExecutors=10 \
  \
  my-application.jar
```

### 4.2 内存模型详解

```
Executor内存分配 (总内存: 8GB)
┌─────────────────────────────────────────┐
│   Executor总内存 (8GB)                  │
├─────────────────────────────────────────┤
│ Reserved Memory (300MB)                 │ 系统保留
├─────────────────────────────────────────┤
│ Usable Memory (7.7GB)                   │
│ ┌─────────────────────────────────────┐ │
│ │ Unified Memory (6.16GB)             │ │ spark.memory.fraction=0.8
│ │ ┌─────────────────┬─────────────────┤ │
│ │ │ Storage Memory  │ Execution Memory│ │
│ │ │    (3.08GB)     │    (3.08GB)     │ │ spark.memory.storageFraction=0.5
│ │ │                 │                 │ │
│ │ │ - Cache RDD     │ - Shuffle       │ │
│ │ │ - Broadcast变量 │ - Join          │ │
│ │ │                 │ - Sort          │ │
│ │ │                 │ - Aggregation   │ │
│ │ └─────────────────┴─────────────────┘ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │ User Memory (1.54GB)                │ │ 用户数据结构
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### 4.3 调优实战案例

```python
"""
Spark性能调优实战
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, broadcast

def optimize_join():
    """Join优化示例"""
    spark = SparkSession.builder.appName("JoinOptimization").getOrCreate()

    # 大表
    large_df = spark.read.parquet("hdfs:///data/large_table")  # 10GB

    # 小表
    small_df = spark.read.parquet("hdfs:///data/small_table")  # 100MB

    # ❌ 不好的做法：普通Join（会产生Shuffle）
    result_bad = large_df.join(small_df, "user_id")

    # ✅ 好的做法：Broadcast Join（小表广播到所有节点）
    result_good = large_df.join(broadcast(small_df), "user_id")

    # 查看执行计划
    result_good.explain(True)

    return result_good

def optimize_partitions():
    """分区优化示例"""
    spark = SparkSession.builder.appName("PartitionOptimization").getOrCreate()

    df = spark.read.parquet("hdfs:///data/events")

    # 检查当前分区数
    print(f"Current partitions: {df.rdd.getNumPartitions()}")

    # ❌ 分区过多（每个分区太小，调度开销大）
    df_over_partitioned = df.repartition(1000)

    # ❌ 分区过少（并行度不足）
    df_under_partitioned = df.coalesce(2)

    # ✅ 合适的分区数（每个分区128MB左右）
    # 假设数据总大小25GB，目标分区大小128MB
    # 分区数 = 25 * 1024 / 128 ≈ 200
    df_optimized = df.repartition(200)

    # 按业务字段分区（用于后续GroupBy优化）
    df_partitioned_by_key = df.repartition(200, "user_id")

    return df_optimized

def optimize_cache():
    """缓存优化示例"""
    spark = SparkSession.builder.appName("CacheOptimization").getOrCreate()

    df = spark.read.parquet("hdfs:///data/users")

    # ❌ 不好的做法：缓存整个大表
    # df.cache()

    # ✅ 好的做法：只缓存需要多次使用的中间结果
    filtered_df = df.filter(col("age") > 18).filter(col("country") == "US")

    # 缓存过滤后的数据（数据量大幅减少）
    filtered_df.cache()

    # 多次使用缓存的数据
    count_by_gender = filtered_df.groupBy("gender").count()
    count_by_state = filtered_df.groupBy("state").count()

    # 使用完后释放缓存
    filtered_df.unpersist()

def optimize_udf():
    """UDF优化示例"""
    from pyspark.sql.functions import udf, pandas_udf
    from pyspark.sql.types import IntegerType
    import pandas as pd

    spark = SparkSession.builder.appName("UDFOptimization").getOrCreate()

    df = spark.range(0, 1000000)

    # ❌ 普通UDF（慢，需要序列化/反序列化）
    @udf(returnType=IntegerType())
    def slow_udf(x):
        return x * 2

    # ✅ Pandas UDF（快，向量化操作）
    @pandas_udf(IntegerType())
    def fast_udf(x: pd.Series) -> pd.Series:
        return x * 2

    # 性能对比
    import time

    start = time.time()
    df.select(slow_udf(col("id"))).count()
    print(f"Normal UDF: {time.time() - start:.2f}s")

    start = time.time()
    df.select(fast_udf(col("id"))).count()
    print(f"Pandas UDF: {time.time() - start:.2f}s")
    # Pandas UDF通常快5-10倍
```

## 5. 数据倾斜解决方案

### 5.1 数据倾斜诊断

```python
"""
数据倾斜检测
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, mean, stddev

def detect_skew(df, key_column):
    """检测数据倾斜"""
    spark = df.sparkSession

    # 统计每个key的数据量
    key_stats = df.groupBy(key_column).count()

    # 计算统计指标
    stats = key_stats.select(
        mean("count").alias("mean"),
        stddev("count").alias("stddev")
    ).first()

    # 找出异常的key
    threshold = stats["mean"] + 3 * stats["stddev"]
    skewed_keys = key_stats.filter(col("count") > threshold)

    print(f"平均值: {stats['mean']:.0f}")
    print(f"标准差: {stats['stddev']:.0f}")
    print(f"倾斜阈值: {threshold:.0f}")
    print(f"\n倾斜的key:")
    skewed_keys.show()

    return skewed_keys
```

### 5.2 解决方案1：加盐法

```scala
// Scala - 加盐法解决数据倾斜
import org.apache.spark.sql.functions._
import scala.util.Random

object SaltedJoin {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("SaltedJoin").getOrCreate()
    import spark.implicits._

    // 大表（有倾斜）
    val largeDF = spark.read.parquet("hdfs:///data/large")

    // 小表
    val smallDF = spark.read.parquet("hdfs:///data/small")

    // 盐值数量（根据倾斜程度调整）
    val saltNum = 10

    // 步骤1：小表加盐扩容
    val saltedSmallDF = smallDF
      .withColumn("salt", explode(array((0 until saltNum).map(lit): _*)))
      .withColumn("salted_key", concat($"user_id", lit("_"), $"salt"))

    // 步骤2：大表加随机盐
    val saltedLargeDF = largeDF
      .withColumn("salt", (rand() * saltNum).cast("int"))
      .withColumn("salted_key", concat($"user_id", lit("_"), $"salt"))

    // 步骤3：使用加盐后的key进行Join
    val result = saltedLargeDF
      .join(saltedSmallDF, Seq("salted_key"), "left")
      .drop("salt", "salted_key")

    result.write.parquet("hdfs:///output/salted_join")
  }
}
```

### 5.3 解决方案2：两阶段聚合

```python
"""
解决方案2：两阶段聚合
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, rand, concat, lit

def two_stage_aggregation(df, key_column, value_column):
    """两阶段聚合解决数据倾斜"""
    spark = df.sparkContext

    # 加盐数量
    salt_num = 10

    # 第一阶段：局部聚合（加盐）
    stage1 = (
        df
        .withColumn("salt", (rand() * salt_num).cast("int"))
        .withColumn("salted_key", concat(col(key_column), lit("_"), col("salt")))
        .groupBy("salted_key")
        .agg(spark_sum(value_column).alias("partial_sum"))
    )

    # 第二阶段：全局聚合（去盐）
    stage2 = (
        stage1
        .withColumn(key_column,
                    expr(f"split(salted_key, '_')[0]"))
        .groupBy(key_column)
        .agg(spark_sum("partial_sum").alias("total_sum"))
    )

    return stage2

# 使用示例
spark = SparkSession.builder.appName("TwoStageAgg").getOrCreate()

df = spark.read.parquet("hdfs:///data/sales")
result = two_stage_aggregation(df, "product_id", "amount")
result.show()
```

### 5.4 解决方案3：拆分Join

```python
"""
解决方案3：拆分倾斜key和正常key
"""
from pyspark.sql.functions import broadcast

def split_skewed_join(large_df, small_df, join_key, skewed_keys_list):
    """拆分倾斜key单独处理"""

    # 分离倾斜数据和正常数据
    skewed_large = large_df.filter(col(join_key).isin(skewed_keys_list))
    normal_large = large_df.filter(~col(join_key).isin(skewed_keys_list))

    skewed_small = small_df.filter(col(join_key).isin(skewed_keys_list))
    normal_small = small_df.filter(~col(join_key).isin(skewed_keys_list))

    # 正常数据：Broadcast Join
    normal_result = normal_large.join(broadcast(normal_small), join_key)

    # 倾斜数据：加盐处理
    salt_num = 20

    # 小表扩容
    salted_small = (
        skewed_small
        .withColumn("salt", explode(array([lit(i) for i in range(salt_num)])))
        .withColumn("salted_key", concat(col(join_key), lit("_"), col("salt")))
    )

    # 大表加盐
    salted_large = (
        skewed_large
        .withColumn("salt", (rand() * salt_num).cast("int"))
        .withColumn("salted_key", concat(col(join_key), lit("_"), col("salt")))
    )

    # Join倾斜数据
    skewed_result = (
        salted_large
        .join(salted_small, "salted_key")
        .drop("salt", "salted_key")
    )

    # 合并结果
    final_result = normal_result.union(skewed_result)

    return final_result

# 使用示例
skewed_keys = ["key1", "key2", "key3"]  # 从detect_skew()获取
result = split_skewed_join(large_df, small_df, "user_id", skewed_keys)
```

### 5.5 解决方案4：动态分区裁剪

```scala
// Scala - 启用动态分区裁剪（Spark 3.0+）
spark.conf.set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")

// 示例：Join分区表
val fact = spark.read.parquet("hdfs:///warehouse/sales")
  .filter($"date" >= "2026-01-01")

val dimension = spark.read.parquet("hdfs:///warehouse/products")
  .filter($"category" === "Electronics")

// 动态分区裁剪会自动优化
val result = fact.join(dimension, "product_id")
```

### 5.6 解决方案5：使用AQE自适应查询

```python
"""
解决方案5：自适应查询执行（AQE）
"""
spark = SparkSession.builder \
    .appName("AQEOptimization") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5") \
    .config("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB") \
    .getOrCreate()

# AQE会在运行时：
# 1. 自动调整分区数
# 2. 自动处理倾斜Join
# 3. 动态切换Join策略

df1 = spark.read.parquet("hdfs:///data/large")
df2 = spark.read.parquet("hdfs:///data/small")

result = df1.join(df2, "key")
result.write.parquet("hdfs:///output/aqe_result")
```

## 6. 实战案例：电商日志分析

```python
#!/usr/bin/env python3
"""
电商日志分析 - 完整示例
数据：用户行为日志（点击、加购、购买）
目标：计算各类指标
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

def analyze_ecommerce_logs():
    spark = SparkSession.builder \
        .appName("EcommerceAnalysis") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

    # 读取日志数据
    logs = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv("hdfs:///data/ecommerce/logs/*.csv")

    # 数据清洗
    clean_logs = logs \
        .filter(col("user_id").isNotNull()) \
        .filter(col("timestamp").isNotNull()) \
        .withColumn("date", to_date("timestamp")) \
        .withColumn("hour", hour("timestamp"))

    # 指标1：每日PV、UV
    daily_stats = clean_logs \
        .groupBy("date") \
        .agg(
            count("*").alias("pv"),
            countDistinct("user_id").alias("uv")
        ) \
        .orderBy("date")

    daily_stats.show()

    # 指标2：转化漏斗
    funnel = clean_logs \
        .groupBy("user_id") \
        .agg(
            sum(when(col("event_type") == "view", 1).otherwise(0)).alias("views"),
            sum(when(col("event_type") == "add_to_cart", 1).otherwise(0)).alias("add_to_cart"),
            sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias("purchases")
        )

    funnel_stats = funnel.agg(
        count("*").alias("total_users"),
        sum(when(col("views") > 0, 1).otherwise(0)).alias("viewed_users"),
        sum(when(col("add_to_cart") > 0, 1).otherwise(0)).alias("cart_users"),
        sum(when(col("purchases") > 0, 1).otherwise(0)).alias("purchased_users")
    ).first()

    print("\n转化漏斗:")
    print(f"浏览用户: {funnel_stats['viewed_users']}")
    print(f"加购用户: {funnel_stats['cart_users']} ({funnel_stats['cart_users']/funnel_stats['viewed_users']*100:.2f}%)")
    print(f"购买用户: {funnel_stats['purchased_users']} ({funnel_stats['purchased_users']/funnel_stats['cart_users']*100:.2f}%)")

    # 指标3：用户留存率
    user_first_visit = clean_logs \
        .groupBy("user_id") \
        .agg(min("date").alias("first_visit_date"))

    retention = clean_logs \
        .join(user_first_visit, "user_id") \
        .withColumn("days_since_first", datediff("date", "first_visit_date")) \
        .groupBy("first_visit_date", "days_since_first") \
        .agg(countDistinct("user_id").alias("retained_users"))

    # 计算Day 1, Day 7, Day 30留存
    retention_rates = retention \
        .filter(col("days_since_first").isin([0, 1, 7, 30])) \
        .show()

    # 指标4：商品热度排行
    product_ranking = clean_logs \
        .filter(col("event_type") == "purchase") \
        .groupBy("product_id", "product_name") \
        .agg(
            count("*").alias("purchase_count"),
            sum("price").alias("revenue"),
            countDistinct("user_id").alias("unique_buyers")
        ) \
        .orderBy(desc("revenue")) \
        .limit(20)

    print("\nTop 20商品:")
    product_ranking.show()

    # 保存结果
    daily_stats.write.mode("overwrite").parquet("hdfs:///output/daily_stats")
    product_ranking.write.mode("overwrite").parquet("hdfs:///output/product_ranking")

    spark.stop()

if __name__ == "__main__":
    analyze_ecommerce_logs()
```

## 7. 最佳实践总结

### 7.1 性能优化检查清单

```
✅ 资源配置
  ├── Executor内存设置合理（避免GC频繁）
  ├── 并行度充足（spark.default.parallelism）
  └── 启用动态资源分配

✅ 数据处理
  ├── 选择合适的文件格式（Parquet > ORC > JSON > CSV）
  ├── 数据分区（按业务字段分区）
  ├── 合理使用缓存（cache高频访问数据）
  └── 过滤操作尽早执行

✅ Join优化
  ├── 小表使用Broadcast Join
  ├── 大表Join前先过滤
  ├── 检测并处理数据倾斜
  └── 选择合适的Join类型

✅ Shuffle优化
  ├── 减少Shuffle操作
  ├── 调整shuffle分区数
  ├── 启用Shuffle压缩
  └── 考虑使用coalesce替代repartition

✅ 代码优化
  ├── 使用DataFrame/Dataset API（而非RDD）
  ├── 避免使用UDF（或使用Pandas UDF）
  ├── 使用内置函数
  └── 启用Tungsten优化
```

### 7.2 监控指标

- **CPU使用率**: 70-80%为佳
- **内存使用率**: < 90%
- **GC时间**: < 10%总时间
- **Shuffle Read/Write**: 越小越好
- **任务倾斜度**: 最慢任务 < 中位数的2倍
