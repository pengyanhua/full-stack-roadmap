# 实时数仓架构

## 1. Lambda架构详解

### 1.1 Lambda架构概览

```
Lambda架构三层模型
┌───────────────────────────────────────────────────────────────┐
│                        数据源层                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ 业务数据库│  │ 日志文件 │  │  埋点    │  │  IoT设备 │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
└───────────────────────────────────────────────────────────────┘
            │                           │
            ↓                           ↓
┌──────────────────────┐    ┌──────────────────────┐
│   Batch Layer        │    │   Speed Layer        │
│   (批处理层)         │    │   (速度层)           │
│                      │    │                      │
│  ┌────────────────┐ │    │  ┌────────────────┐ │
│  │  Kafka → HDFS  │ │    │  │  Kafka → Flink │ │
│  └────────────────┘ │    │  └────────────────┘ │
│         ↓            │    │         ↓            │
│  ┌────────────────┐ │    │  ┌────────────────┐ │
│  │  Spark Batch   │ │    │  │ Stream Process │ │
│  │  (每日T+1)     │ │    │  │  (实时计算)    │ │
│  └────────────────┘ │    │  └────────────────┘ │
│         ↓            │    │         ↓            │
│  ┌────────────────┐ │    │  ┌────────────────┐ │
│  │  Batch Views   │ │    │  │ Realtime Views │ │
│  │  (Hive/Iceberg)│ │    │  │  (Redis/HBase) │ │
│  └────────────────┘ │    │  └────────────────┘ │
└──────────────────────┘    └──────────────────────┘
            │                           │
            └──────────┬────────────────┘
                       ↓
        ┌──────────────────────────┐
        │   Serving Layer          │
        │   (服务层)               │
        │  ┌────────────────────┐  │
        │  │ Query Merger       │  │
        │  │ (合并批量+实时)    │  │
        │  └────────────────────┘  │
        │         ↓                 │
        │  ┌────────────────────┐  │
        │  │  API Gateway       │  │
        │  └────────────────────┘  │
        └──────────────────────────┘
                  ↓
        ┌──────────────────────┐
        │  应用层              │
        │  - BI报表            │
        │  - 数据大屏          │
        │  - 算法模型          │
        └──────────────────────┘
```

### 1.2 批处理层实现

```scala
/**
 * 批处理层 - Spark Batch
 * 每日T+1全量计算
 */
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._

object BatchLayer {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Lambda-BatchLayer")
      .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
      .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog")
      .getOrCreate()

    import spark.implicits._

    // 1. 读取原始数据（从HDFS/S3）
    val rawData = spark.read
      .format("parquet")
      .load("hdfs:///warehouse/raw/user_behavior/dt=20260207")

    // 2. 数据清洗
    val cleanedData = rawData
      .filter($"user_id".isNotNull)
      .filter($"timestamp".isNotNull)
      .withColumn("date", to_date($"timestamp"))
      .withColumn("hour", hour($"timestamp"))

    // 3. 计算DWD层（明细层）
    val dwdUserBehavior = cleanedData
      .select(
        $"user_id",
        $"item_id",
        $"behavior_type",
        $"timestamp",
        $"date",
        $"hour"
      )

    // 保存到Iceberg表
    dwdUserBehavior.write
      .format("iceberg")
      .mode("overwrite")
      .save("warehouse.dwd_user_behavior")

    // 4. 计算DWS层（汇总层）
    // 每日用户行为汇总
    val dwsUserDaily = dwdUserBehavior
      .groupBy("user_id", "date")
      .agg(
        count(when($"behavior_type" === "pv", 1)).alias("pv_count"),
        count(when($"behavior_type" === "cart", 1)).alias("cart_count"),
        count(when($"behavior_type" === "fav", 1)).alias("fav_count"),
        count(when($"behavior_type" === "buy", 1)).alias("buy_count")
      )

    dwsUserDaily.write
      .format("iceberg")
      .mode("overwrite")
      .save("warehouse.dws_user_daily")

    // 5. 计算ADS层（应用层）
    // 商品热度排行
    val adsItemRanking = dwdUserBehavior
      .filter($"behavior_type" === "pv")
      .groupBy("item_id", "date")
      .agg(count("*").alias("pv_count"))
      .withColumn("rank", rank().over(
        Window.partitionBy("date").orderBy($"pv_count".desc)
      ))
      .filter($"rank" <= 100)

    adsItemRanking.write
      .format("iceberg")
      .mode("overwrite")
      .save("warehouse.ads_item_ranking")

    spark.stop()
  }
}
```

### 1.3 速度层实现

```java
/**
 * 速度层 - Flink Streaming
 * 实时增量计算
 */
import org.apache.flink.api.common.eventtime.*;
import org.apache.flink.api.common.state.*;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;

import java.time.Duration;

public class SpeedLayer {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.enableCheckpointing(10000);

        // 1. 从Kafka读取实时数据
        DataStream<UserBehavior> stream = env
            .addSource(new FlinkKafkaConsumer<>(
                "user_behavior",
                new UserBehaviorSchema(),
                kafkaProperties()
            ))
            .assignTimestampsAndWatermarks(
                WatermarkStrategy
                    .<UserBehavior>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                    .withTimestampAssigner((event, ts) -> event.timestamp)
            );

        // 2. 实时计算用户行为指标
        DataStream<UserMetrics> userMetrics = stream
            .keyBy(UserBehavior::getUserId)
            .process(new UserBehaviorProcessor());

        // 3. 写入Redis（实时查询层）
        userMetrics.addSink(new RedisSink<UserMetrics>()
            .withKey(metric -> "user_metrics:" + metric.userId)
            .withExpiration(24 * 3600)  // 24小时过期
        );

        // 4. 实时商品热度Top-N
        DataStream<ItemRanking> itemRanking = stream
            .filter(b -> "pv".equals(b.behaviorType))
            .keyBy(UserBehavior::getItemId)
            .window(TumblingEventTimeWindows.of(Time.minutes(5)))
            .aggregate(new CountAggregate(), new WindowResultFunction())
            .keyBy(ItemViewCount::getWindowEnd)
            .process(new TopNHotItems(100));

        // 5. 写入HBase（实时明细层）
        stream.addSink(new HBaseSink<>("dwd_user_behavior_realtime"));

        env.execute("Lambda-SpeedLayer");
    }

    /**
     * 用户行为处理器 - 维护实时状态
     */
    public static class UserBehaviorProcessor
            extends KeyedProcessFunction<Long, UserBehavior, UserMetrics> {

        private ValueState<UserMetrics> metricsState;

        @Override
        public void open(Configuration parameters) {
            ValueStateDescriptor<UserMetrics> descriptor =
                new ValueStateDescriptor<>("user-metrics", UserMetrics.class);

            // 配置状态TTL（24小时）
            StateTtlConfig ttlConfig = StateTtlConfig
                .newBuilder(Time.hours(24))
                .setUpdateType(StateTtlConfig.UpdateType.OnCreateAndWrite)
                .setStateVisibility(StateTtlConfig.StateVisibility.NeverReturnExpired)
                .build();

            descriptor.enableTimeToLive(ttlConfig);
            metricsState = getRuntimeContext().getState(descriptor);
        }

        @Override
        public void processElement(UserBehavior value,
                                  Context ctx,
                                  Collector<UserMetrics> out) throws Exception {

            UserMetrics metrics = metricsState.value();
            if (metrics == null) {
                metrics = new UserMetrics(value.userId);
            }

            // 更新指标
            switch (value.behaviorType) {
                case "pv":
                    metrics.pvCount++;
                    break;
                case "cart":
                    metrics.cartCount++;
                    break;
                case "fav":
                    metrics.favCount++;
                    break;
                case "buy":
                    metrics.buyCount++;
                    break;
            }

            metrics.lastUpdateTime = System.currentTimeMillis();
            metricsState.update(metrics);

            out.collect(metrics);
        }
    }

    // 数据模型
    public static class UserBehavior {
        public long userId;
        public long itemId;
        public String behaviorType;
        public long timestamp;

        public long getUserId() { return userId; }
        public long getItemId() { return itemId; }
    }

    public static class UserMetrics {
        public long userId;
        public long pvCount;
        public long cartCount;
        public long favCount;
        public long buyCount;
        public long lastUpdateTime;

        public UserMetrics(long userId) {
            this.userId = userId;
        }
    }
}
```

### 1.4 服务层实现

```java
/**
 * 服务层 - 合并批量视图和实时视图
 */
@RestController
@RequestMapping("/api/metrics")
public class ServingLayer {

    @Autowired
    private RedisTemplate<String, UserMetrics> redisTemplate;

    @Autowired
    private HiveJdbcTemplate hiveTemplate;

    /**
     * 查询用户行为指标（Lambda架构合并查询）
     */
    @GetMapping("/user/{userId}")
    public UserMetricsResponse getUserMetrics(
            @PathVariable Long userId,
            @RequestParam String startDate,
            @RequestParam String endDate) {

        // 1. 查询批处理层（历史数据）
        String batchQuery = String.format(
            "SELECT pv_count, cart_count, fav_count, buy_count " +
            "FROM warehouse.dws_user_daily " +
            "WHERE user_id = %d AND date >= '%s' AND date < CURRENT_DATE",
            userId, startDate
        );

        UserMetrics batchMetrics = hiveTemplate.queryForObject(
            batchQuery,
            (rs, rowNum) -> new UserMetrics(
                rs.getLong("pv_count"),
                rs.getLong("cart_count"),
                rs.getLong("fav_count"),
                rs.getLong("buy_count")
            )
        );

        // 2. 查询速度层（今日实时数据）
        String redisKey = "user_metrics:" + userId;
        UserMetrics realtimeMetrics = redisTemplate.opsForValue().get(redisKey);

        // 3. 合并结果
        UserMetricsResponse response = new UserMetricsResponse();
        response.userId = userId;
        response.pvCount = batchMetrics.pvCount + (realtimeMetrics != null ? realtimeMetrics.pvCount : 0);
        response.cartCount = batchMetrics.cartCount + (realtimeMetrics != null ? realtimeMetrics.cartCount : 0);
        response.favCount = batchMetrics.favCount + (realtimeMetrics != null ? realtimeMetrics.favCount : 0);
        response.buyCount = batchMetrics.buyCount + (realtimeMetrics != null ? realtimeMetrics.buyCount : 0);

        return response;
    }

    @Data
    public static class UserMetricsResponse {
        private Long userId;
        private Long pvCount;
        private Long cartCount;
        private Long favCount;
        private Long buyCount;
    }
}
```

## 2. Kappa架构简化方案

### 2.1 Kappa架构概览

```
Kappa架构（纯流式架构）
┌───────────────────────────────────────────────────────┐
│                    数据源层                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ 业务数据库│  │ 日志文件 │  │  埋点    │            │
│  └──────────┘  └──────────┘  └──────────┘            │
└───────────────────────────────────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────┐
        │    Kafka (消息队列)      │
        │  - 数据持久化            │
        │  - 支持重新消费          │
        │  - 保留7天历史           │
        └──────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────┐
        │   Stream Processing      │
        │   (统一流处理层)         │
        │                          │
        │  ┌────────────────────┐  │
        │  │  Flink/Kafka       │  │
        │  │  Streams           │  │
        │  │                    │  │
        │  │ - 实时计算         │  │
        │  │ - 历史回溯         │  │
        │  │ - 版本迭代         │  │
        │  └────────────────────┘  │
        └──────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │  Redis   │ │  HBase   │ │ ClickHouse│
   │ (实时KV) │ │(明细数据)│ │ (OLAP)    │
   └──────────┘ └──────────┘ └──────────┘
          │            │            │
          └────────────┼────────────┘
                       ↓
          ┌──────────────────────┐
          │   Serving Layer      │
          │   (API服务)          │
          └──────────────────────┘

Kappa vs Lambda:
✅ 优势: 架构简单、维护成本低、无需数据合并
❌ 劣势: 对流处理引擎要求高、重新计算成本大
```

### 2.2 Kappa架构实现

```java
/**
 * Kappa架构 - Flink实现
 * 统一流处理，支持历史回溯
 */
import org.apache.flink.api.common.state.*;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;

public class KappaArchitecture {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // 关键配置：允许从任意时间点重新消费
        Properties kafkaProps = new Properties();
        kafkaProps.setProperty("bootstrap.servers", "kafka:9092");
        kafkaProps.setProperty("group.id", "kappa-processor-v2");  // 版本化GroupID

        FlinkKafkaConsumer<Event> source = new FlinkKafkaConsumer<>(
            "events",
            new EventSchema(),
            kafkaProps
        );

        // 支持三种启动模式：
        // 1. 从最新位置开始（实时处理）
        // source.setStartFromLatest();

        // 2. 从最早位置开始（重新计算全量）
        // source.setStartFromEarliest();

        // 3. 从指定时间戳开始（修复特定时间段）
        source.setStartFromTimestamp(1704067200000L);  // 2026-01-01 00:00:00

        DataStream<Event> stream = env.addSource(source);

        // 统一的流处理逻辑
        DataStream<Metrics> metrics = stream
            .keyBy(Event::getKey)
            .process(new UnifiedProcessor());

        // 写入多个存储系统
        metrics.addSink(new RedisSink<>());       // 实时KV
        metrics.addSink(new HBaseSink<>());       // 明细数据
        metrics.addSink(new ClickHouseSink<>());  // OLAP分析

        env.execute("Kappa-Architecture");
    }

    /**
     * 统一处理器 - 处理实时和历史数据
     */
    public static class UnifiedProcessor
            extends KeyedProcessFunction<String, Event, Metrics> {

        private MapState<String, Long> counters;

        @Override
        public void open(Configuration parameters) {
            // 使用状态存储所有维度的计数器
            MapStateDescriptor<String, Long> descriptor =
                new MapStateDescriptor<>("counters", String.class, Long.class);

            counters = getRuntimeContext().getMapState(descriptor);
        }

        @Override
        public void processElement(Event event,
                                  Context ctx,
                                  Collector<Metrics> out) throws Exception {

            // 更新各维度计数器
            String[] dimensions = {"pv", "cart", "fav", "buy"};
            for (String dimension : dimensions) {
                Long count = counters.get(dimension);
                if (count == null) count = 0L;

                if (dimension.equals(event.eventType)) {
                    count++;
                    counters.put(dimension, count);
                }
            }

            // 输出指标
            Metrics metrics = new Metrics();
            metrics.key = ctx.getCurrentKey();
            metrics.pvCount = counters.get("pv");
            metrics.cartCount = counters.get("cart");
            metrics.favCount = counters.get("fav");
            metrics.buyCount = counters.get("buy");
            metrics.timestamp = event.timestamp;

            out.collect(metrics);
        }
    }
}
```

## 3. Apache Doris实时OLAP

### 3.1 Doris架构

```
Apache Doris架构
┌────────────────────────────────────────────────┐
│              MySQL Client / JDBC               │
└────────────────────────────────────────────────┘
                      │
                      ↓
┌────────────────────────────────────────────────┐
│              FE (Frontend)                     │
│  ┌──────────────┐  ┌──────────────┐           │
│  │ Query Parser │  │ Query Planner│           │
│  └──────────────┘  └──────────────┘           │
│  ┌──────────────┐  ┌──────────────┐           │
│  │ Meta Manager │  │ Load Manager │           │
│  └──────────────┘  └──────────────┘           │
└────────────────────────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   BE 1       │ │   BE 2       │ │   BE 3       │
│ (Backend)    │ │ (Backend)    │ │ (Backend)    │
│              │ │              │ │              │
│ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │
│ │ Tablet 1 │ │ │ │ Tablet 2 │ │ │ │ Tablet 3 │ │
│ ├──────────┤ │ │ ├──────────┤ │ │ ├──────────┤ │
│ │ Tablet 4 │ │ │ │ Tablet 5 │ │ │ │ Tablet 6 │ │
│ └──────────┘ │ │ └──────────┘ │ │ └──────────┘ │
└──────────────┘ └──────────────┘ └──────────────┘

特性:
- MPP架构，查询并行执行
- 列式存储，压缩比高
- 支持实时写入和查询
- 兼容MySQL协议
```

### 3.2 Doris建表与数据导入

```sql
-- 1. 创建数据库
CREATE DATABASE IF NOT EXISTS realtime_dw;
USE realtime_dw;

-- 2. 创建明细表（Duplicate Key模型）
CREATE TABLE dwd_user_behavior (
    user_id BIGINT COMMENT '用户ID',
    item_id BIGINT COMMENT '商品ID',
    behavior_type VARCHAR(20) COMMENT '行为类型',
    event_time DATETIME COMMENT '事件时间',
    date DATE COMMENT '日期分区',
    province VARCHAR(50) COMMENT '省份',
    city VARCHAR(50) COMMENT '城市'
)
DUPLICATE KEY(user_id, item_id)
PARTITION BY RANGE(date) (
    PARTITION p20260201 VALUES [('2026-02-01'), ('2026-02-02')),
    PARTITION p20260202 VALUES [('2026-02-02'), ('2026-02-03')),
    PARTITION p20260203 VALUES [('2026-02-03'), ('2026-02-04'))
)
DISTRIBUTED BY HASH(user_id) BUCKETS 32
PROPERTIES (
    "replication_num" = "3",
    "storage_medium" = "SSD",
    "storage_cooldown_time" = "2026-03-01 00:00:00"
);

-- 3. 创建聚合表（Aggregate Key模型）
CREATE TABLE dws_user_daily (
    user_id BIGINT COMMENT '用户ID',
    date DATE COMMENT '日期',
    pv_count BIGINT SUM DEFAULT "0" COMMENT '浏览次数',
    cart_count BIGINT SUM DEFAULT "0" COMMENT '加购次数',
    fav_count BIGINT SUM DEFAULT "0" COMMENT '收藏次数',
    buy_count BIGINT SUM DEFAULT "0" COMMENT '购买次数'
)
AGGREGATE KEY(user_id, date)
PARTITION BY RANGE(date) ()
DISTRIBUTED BY HASH(user_id) BUCKETS 32
PROPERTIES (
    "replication_num" = "3"
);

-- 4. 创建物化视图（加速查询）
CREATE MATERIALIZED VIEW mv_item_hourly_pv AS
SELECT
    item_id,
    DATE_FORMAT(event_time, '%Y-%m-%d %H:00:00') AS hour,
    COUNT(*) AS pv_count
FROM dwd_user_behavior
WHERE behavior_type = 'pv'
GROUP BY item_id, DATE_FORMAT(event_time, '%Y-%m-%d %H:00:00');

-- 5. Flink实时写入Doris
-- flink-doris-connector配置
```

```java
/**
 * Flink写入Doris
 */
import org.apache.flink.connector.jdbc.JdbcConnectionOptions;
import org.apache.flink.connector.jdbc.JdbcExecutionOptions;
import org.apache.flink.connector.jdbc.JdbcSink;
import org.apache.flink.streaming.api.datastream.DataStream;

public class FlinkToDoris {

    public static void writeToDoris(DataStream<UserBehavior> stream) {
        stream.addSink(
            JdbcSink.sink(
                // INSERT语句
                "INSERT INTO dwd_user_behavior " +
                "(user_id, item_id, behavior_type, event_time, date, province, city) " +
                "VALUES (?, ?, ?, ?, ?, ?, ?)",

                // PreparedStatement设置参数
                (ps, behavior) -> {
                    ps.setLong(1, behavior.userId);
                    ps.setLong(2, behavior.itemId);
                    ps.setString(3, behavior.behaviorType);
                    ps.setTimestamp(4, new Timestamp(behavior.timestamp));
                    ps.setDate(5, new Date(behavior.timestamp));
                    ps.setString(6, behavior.province);
                    ps.setString(7, behavior.city);
                },

                // 执行选项
                JdbcExecutionOptions.builder()
                    .withBatchSize(1000)
                    .withBatchIntervalMs(200)
                    .withMaxRetries(3)
                    .build(),

                // 连接选项
                new JdbcConnectionOptions.JdbcConnectionOptionsBuilder()
                    .withUrl("jdbc:mysql://doris-fe:9030/realtime_dw")
                    .withDriverName("com.mysql.cj.jdbc.Driver")
                    .withUsername("root")
                    .withPassword("password")
                    .build()
            )
        );
    }
}
```

### 3.3 Doris查询优化

```sql
-- 查询1：热门商品Top-N（使用物化视图加速）
SELECT
    item_id,
    SUM(pv_count) AS total_pv
FROM mv_item_hourly_pv
WHERE hour >= '2026-02-07 00:00:00'
  AND hour < '2026-02-08 00:00:00'
GROUP BY item_id
ORDER BY total_pv DESC
LIMIT 100;

-- 查询2：用户行为漏斗分析
SELECT
    date,
    COUNT(DISTINCT user_id) AS total_users,
    COUNT(DISTINCT CASE WHEN pv_count > 0 THEN user_id END) AS pv_users,
    COUNT(DISTINCT CASE WHEN cart_count > 0 THEN user_id END) AS cart_users,
    COUNT(DISTINCT CASE WHEN buy_count > 0 THEN user_id END) AS buy_users,
    COUNT(DISTINCT CASE WHEN cart_count > 0 THEN user_id END) * 100.0 /
        COUNT(DISTINCT CASE WHEN pv_count > 0 THEN user_id END) AS pv_to_cart_rate,
    COUNT(DISTINCT CASE WHEN buy_count > 0 THEN user_id END) * 100.0 /
        COUNT(DISTINCT CASE WHEN cart_count > 0 THEN user_id END) AS cart_to_buy_rate
FROM dws_user_daily
WHERE date >= '2026-02-01'
GROUP BY date
ORDER BY date;

-- 查询3：地域分析（利用分区裁剪）
SELECT
    province,
    city,
    COUNT(*) AS event_count,
    COUNT(DISTINCT user_id) AS user_count
FROM dwd_user_behavior
WHERE date = '2026-02-07'  -- 分区裁剪
  AND behavior_type = 'buy'
GROUP BY province, city
ORDER BY event_count DESC
LIMIT 50;
```

## 4. 实战：实时用户画像系统

### 4.1 系统架构

```
实时用户画像系统
┌────────────────────────────────────────────────────┐
│                 数据采集层                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ 埋点数据  │  │ 订单数据  │  │ CRM数据  │         │
│  └──────────┘  └──────────┘  └──────────┘         │
└────────────────────────────────────────────────────┘
                     │
                     ↓ (Kafka)
┌────────────────────────────────────────────────────┐
│              标签计算层 (Flink)                     │
│                                                    │
│  ┌────────────────┐  ┌────────────────┐           │
│  │ 规则型标签     │  │ 统计型标签     │           │
│  │ - 性别         │  │ - 消费金额     │           │
│  │ - 年龄段       │  │ - 购买频次     │           │
│  │ - 会员等级     │  │ - 活跃度       │           │
│  └────────────────┘  └────────────────┘           │
│  ┌────────────────┐  ┌────────────────┐           │
│  │ 预测型标签     │  │ 挖掘型标签     │           │
│  │ - 流失倾向     │  │ - 兴趣偏好     │           │
│  │ - 购买意向     │  │ - 价格敏感度   │           │
│  └────────────────┘  └────────────────┘           │
└────────────────────────────────────────────────────┘
                     │
                     ↓
┌────────────────────────────────────────────────────┐
│              标签存储层                             │
│  ┌────────────────┐  ┌────────────────┐           │
│  │ HBase          │  │ Redis          │           │
│  │ (全量标签)     │  │ (热点标签)     │           │
│  └────────────────┘  └────────────────┘           │
└────────────────────────────────────────────────────┘
                     │
                     ↓
┌────────────────────────────────────────────────────┐
│              应用层                                 │
│  - 个性化推荐                                       │
│  - 精准营销                                         │
│  - 用户分群                                         │
└────────────────────────────────────────────────────┘
```

### 4.2 标签计算实现

```java
/**
 * 实时用户画像标签计算
 */
import org.apache.flink.api.common.state.*;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.co.KeyedCoProcessFunction;
import org.apache.flink.util.Collector;

import java.util.*;

public class UserProfileTagging {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // 1. 用户行为流
        DataStream<UserBehavior> behaviorStream = env
            .addSource(new KafkaSource<>("user_behavior"));

        // 2. 用户属性流
        DataStream<UserProfile> profileStream = env
            .addSource(new KafkaSource<>("user_profile"));

        // 3. 计算标签
        DataStream<UserTags> tagStream = behaviorStream
            .keyBy(UserBehavior::getUserId)
            .connect(profileStream.keyBy(UserProfile::getUserId))
            .process(new UserTagProcessor());

        // 4. 写入HBase
        tagStream.addSink(new HBaseTagSink());

        // 5. 热点标签写入Redis
        tagStream
            .filter(tags -> tags.isHotUser())
            .addSink(new RedisTagSink());

        env.execute("User Profile Tagging");
    }

    /**
     * 用户标签处理器
     */
    public static class UserTagProcessor
            extends KeyedCoProcessFunction<Long, UserBehavior, UserProfile, UserTags> {

        // 状态：用户行为统计
        private MapState<String, BehaviorStats> behaviorStatsState;

        // 状态：用户属性
        private ValueState<UserProfile> profileState;

        @Override
        public void open(Configuration parameters) {
            // 行为统计状态（保留30天）
            MapStateDescriptor<String, BehaviorStats> behaviorDescriptor =
                new MapStateDescriptor<>("behavior-stats", String.class, BehaviorStats.class);
            StateTtlConfig ttlConfig = StateTtlConfig
                .newBuilder(Time.days(30))
                .setUpdateType(StateTtlConfig.UpdateType.OnCreateAndWrite)
                .build();
            behaviorDescriptor.enableTimeToLive(ttlConfig);
            behaviorStatsState = getRuntimeContext().getMapState(behaviorDescriptor);

            // 用户属性状态
            profileState = getRuntimeContext().getState(
                new ValueStateDescriptor<>("profile", UserProfile.class)
            );
        }

        @Override
        public void processElement1(UserBehavior behavior,
                                   Context ctx,
                                   Collector<UserTags> out) throws Exception {

            // 更新行为统计
            String period = getPeriod(behavior.timestamp);  // 例如：2026-02-07
            BehaviorStats stats = behaviorStatsState.get(period);
            if (stats == null) {
                stats = new BehaviorStats();
            }

            stats.update(behavior);
            behaviorStatsState.put(period, stats);

            // 计算标签
            UserTags tags = calculateTags(ctx.getCurrentKey(), stats, profileState.value());
            out.collect(tags);
        }

        @Override
        public void processElement2(UserProfile profile,
                                   Context ctx,
                                   Collector<UserTags> out) throws Exception {
            // 更新用户属性
            profileState.update(profile);

            // 重新计算标签
            Iterator<Map.Entry<String, BehaviorStats>> iterator = behaviorStatsState.entries().iterator();
            BehaviorStats latestStats = iterator.hasNext() ? iterator.next().getValue() : new BehaviorStats();

            UserTags tags = calculateTags(ctx.getCurrentKey(), latestStats, profile);
            out.collect(tags);
        }

        /**
         * 标签计算逻辑
         */
        private UserTags calculateTags(Long userId, BehaviorStats stats, UserProfile profile) {
            UserTags tags = new UserTags(userId);

            // 规则型标签
            if (profile != null) {
                tags.addTag("gender", profile.gender);
                tags.addTag("age_group", getAgeGroup(profile.age));
                tags.addTag("member_level", profile.memberLevel);
            }

            // 统计型标签
            tags.addTag("total_purchase", String.valueOf(stats.totalPurchase));
            tags.addTag("purchase_frequency", getPurchaseFrequency(stats.purchaseCount));
            tags.addTag("avg_price", String.valueOf(stats.totalAmount / Math.max(stats.purchaseCount, 1)));

            // 行为偏好标签
            tags.addTag("active_level", getActiveLevel(stats.pvCount, stats.cartCount));
            tags.addTag("conversion_rate", String.format("%.2f",
                stats.purchaseCount * 100.0 / Math.max(stats.pvCount, 1)));

            // 预测型标签（简化示例）
            boolean isChurnRisk = stats.daysSinceLastPurchase > 30;
            tags.addTag("churn_risk", isChurnRisk ? "high" : "low");

            return tags;
        }

        private String getAgeGroup(int age) {
            if (age < 18) return "未成年";
            if (age < 25) return "18-24岁";
            if (age < 35) return "25-34岁";
            if (age < 45) return "35-44岁";
            return "45岁以上";
        }

        private String getPurchaseFrequency(long count) {
            if (count == 0) return "无购买";
            if (count <= 3) return "低频";
            if (count <= 10) return "中频";
            return "高频";
        }

        private String getActiveLevel(long pvCount, long cartCount) {
            long totalActions = pvCount + cartCount;
            if (totalActions < 10) return "低活跃";
            if (totalActions < 50) return "中活跃";
            return "高活跃";
        }

        private String getPeriod(long timestamp) {
            // 转换为日期字符串
            return new java.text.SimpleDateFormat("yyyy-MM-dd")
                .format(new Date(timestamp));
        }
    }

    // 数据模型
    public static class UserBehavior {
        public long userId;
        public String behaviorType;
        public double amount;
        public long timestamp;

        public long getUserId() { return userId; }
    }

    public static class UserProfile {
        public long userId;
        public String gender;
        public int age;
        public String memberLevel;

        public long getUserId() { return userId; }
    }

    public static class BehaviorStats {
        public long pvCount = 0;
        public long cartCount = 0;
        public long purchaseCount = 0;
        public double totalAmount = 0;
        public long totalPurchase = 0;
        public long lastPurchaseTime = 0;
        public long daysSinceLastPurchase = 0;

        public void update(UserBehavior behavior) {
            switch (behavior.behaviorType) {
                case "pv":
                    pvCount++;
                    break;
                case "cart":
                    cartCount++;
                    break;
                case "buy":
                    purchaseCount++;
                    totalAmount += behavior.amount;
                    totalPurchase++;
                    lastPurchaseTime = behavior.timestamp;
                    break;
            }

            if (lastPurchaseTime > 0) {
                daysSinceLastPurchase = (System.currentTimeMillis() - lastPurchaseTime) / (24 * 3600 * 1000);
            }
        }
    }

    public static class UserTags {
        public long userId;
        public Map<String, String> tags = new HashMap<>();
        public long updateTime;

        public UserTags(long userId) {
            this.userId = userId;
            this.updateTime = System.currentTimeMillis();
        }

        public void addTag(String key, String value) {
            tags.put(key, value);
        }

        public boolean isHotUser() {
            // 定义热点用户：高活跃 + 高频购买
            return "高活跃".equals(tags.get("active_level")) &&
                   "高频".equals(tags.get("purchase_frequency"));
        }
    }
}
```

### 4.3 标签查询服务

```java
/**
 * 用户画像查询服务
 */
@RestController
@RequestMapping("/api/profile")
public class UserProfileService {

    @Autowired
    private RedisTemplate<String, Map<String, String>> redisTemplate;

    @Autowired
    private HBaseTemplate hbaseTemplate;

    /**
     * 获取用户标签
     */
    @GetMapping("/tags/{userId}")
    public UserTagsResponse getUserTags(@PathVariable Long userId) {
        String redisKey = "user_tags:" + userId;

        // 1. 先查Redis（热点数据）
        Map<String, String> tags = redisTemplate.opsForValue().get(redisKey);

        // 2. Redis未命中，查HBase
        if (tags == null) {
            tags = hbaseTemplate.get("user_profile", String.valueOf(userId), "tags");

            // 回写Redis（过期时间1小时）
            if (tags != null) {
                redisTemplate.opsForValue().set(redisKey, tags, 1, TimeUnit.HOURS);
            }
        }

        return new UserTagsResponse(userId, tags);
    }

    /**
     * 用户分群查询
     */
    @PostMapping("/segment")
    public List<Long> getUserSegment(@RequestBody SegmentQuery query) {
        // 使用Doris进行用户分群查询
        String sql = buildSegmentSQL(query);
        return dorisTemplate.queryForList(sql, Long.class);
    }

    private String buildSegmentSQL(SegmentQuery query) {
        // 构建SQL：例如查询"25-34岁、高活跃、近30天有购买"的用户
        return String.format(
            "SELECT user_id FROM user_profile_tags " +
            "WHERE age_group = '%s' AND active_level = '%s' AND purchase_count_30d > 0",
            query.ageGroup, query.activeLevel
        );
    }
}
```

## 5. 总结

### 5.1 架构选型对比

| 维度 | Lambda | Kappa | 适用场景 |
|------|--------|-------|----------|
| **复杂度** | 高（双路径） | 低（单路径） | Kappa更简单 |
| **数据一致性** | 需要合并 | 自然一致 | Kappa更优 |
| **历史回溯** | 批处理层重跑 | Kafka重放 | Lambda成本低 |
| **实时性** | 毫秒级 | 毫秒级 | 相同 |
| **维护成本** | 高 | 低 | Kappa更低 |
| **计算成本** | 低（批处理便宜） | 高（流处理贵） | Lambda更省 |

### 5.2 最佳实践

```
1. 小团队、简单场景 → Kappa架构
2. 大规模、复杂场景 → Lambda架构
3. 实时OLAP → Doris/ClickHouse
4. 用户画像 → Flink + HBase + Redis
```
