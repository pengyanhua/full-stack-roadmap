# 特征存储系统（Feature Store）

## 1. Feature Store架构

### 1.1 核心概念

```
Feature Store完整架构
┌────────────────────────────────────────────────────────────┐
│                    数据源层                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ 数据库   │  │ 数据仓库 │  │ 实时流   │  │ 日志文件 │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└────────────────────────────────────────────────────────────┘
                         │
                         ↓
┌────────────────────────────────────────────────────────────┐
│                 特征工程层                                  │
│  ┌────────────────┐         ┌────────────────┐            │
│  │ 批处理特征计算 │         │ 流式特征计算   │            │
│  │ (Spark/Flink)  │         │ (Flink)        │            │
│  └────────────────┘         └────────────────┘            │
└────────────────────────────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ↓                         ↓
┌──────────────────────┐    ┌──────────────────────┐
│   离线特征存储        │    │   在线特征存储        │
│   (Parquet/Delta)    │    │   (Redis/DynamoDB)   │
│                      │    │                      │
│  - 历史特征数据      │    │  - 低延迟查询        │
│  - 模型训练          │    │  - 实时推理          │
│  - 特征探索          │    │  - 毫秒级响应        │
└──────────────────────┘    └──────────────────────┘
            │                         │
            └────────────┬────────────┘
                         ↓
┌────────────────────────────────────────────────────────────┐
│                  Feature Store服务层                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ 特征注册    │  │ 特征版本    │  │ 特征血缘    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└────────────────────────────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            ↓                         ↓
┌──────────────────────┐    ┌──────────────────────┐
│    离线消费           │    │    在线消费           │
│  - 模型训练           │    │  - 实时预测           │
│  - 批量预测           │    │  - A/B测试            │
└──────────────────────┘    └──────────────────────┘
```

### 1.2 为什么需要Feature Store

```
传统问题:
❌ 特征工程代码重复（训练和推理不一致）
❌ 特征计算效率低（重复计算）
❌ 特征共享困难（团队间无法复用）
❌ 特征版本管理混乱
❌ 训练-推理偏差（Training-Serving Skew）

Feature Store解决方案:
✅ 统一特征定义（一次定义，到处使用）
✅ 特征复用（团队共享特征）
✅ 离线在线一致性（消除偏差）
✅ 特征版本管理（Git for Features）
✅ 特征血缘追踪（可解释性）
```

## 2. Feast架构与组件

### 2.1 Feast核心组件

```
Feast架构
┌────────────────────────────────────────────────────┐
│            Feature Repository (Git)                │
│  - feature_definition.py                          │
│  - feature_store.yaml                             │
└────────────────────────────────────────────────────┘
                      │
                      ↓
┌────────────────────────────────────────────────────┐
│              Feast Registry                        │
│  (存储特征元数据)                                  │
│  - S3/GCS/Local                                    │
└────────────────────────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ↓            ↓            ↓
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Offline     │ │ Online      │ │ Streaming   │
│ Store       │ │ Store       │ │ Source      │
│ (Parquet)   │ │ (Redis)     │ │ (Kafka)     │
└─────────────┘ └─────────────┘ └─────────────┘
```

### 2.2 Feast特征定义

```python
"""
Feast特征定义示例
文件: features/user_features.py
"""
from feast import Entity, Feature, FeatureView, ValueType
from feast.data_source import FileSource, KafkaSource
from datetime import timedelta

# 1. 定义Entity（实体）
user = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="用户ID"
)

item = Entity(
    name="item_id",
    value_type=ValueType.INT64,
    description="商品ID"
)

# 2. 定义离线数据源（批处理）
user_stats_source = FileSource(
    path="s3://my-bucket/features/user_stats.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp"
)

# 3. 定义在线数据源（流式）
user_activity_stream = KafkaSource(
    name="user_activity_stream",
    event_timestamp_column="event_timestamp",
    bootstrap_servers="localhost:9092",
    message_format=AvroFormat("user_activity_schema"),
    topic="user_activity",
    batch_source=user_stats_source  # 用于历史数据
)

# 4. 定义Feature View（特征视图）
user_features = FeatureView(
    name="user_features",
    entities=["user_id"],
    ttl=timedelta(days=30),  # 特征有效期
    features=[
        Feature(name="total_orders", dtype=ValueType.INT64),
        Feature(name="total_spent", dtype=ValueType.DOUBLE),
        Feature(name="avg_order_value", dtype=ValueType.DOUBLE),
        Feature(name="days_since_last_order", dtype=ValueType.INT32),
        Feature(name="favorite_category", dtype=ValueType.STRING),
        Feature(name="user_lifetime_days", dtype=ValueType.INT32),
    ],
    online=True,  # 支持在线查询
    source=user_stats_source,
    tags={"team": "data-science", "version": "v1"}
)

# 5. 定义Stream Feature View（实时特征）
from feast import Field
from feast.types import Float32, Int64

user_realtime_features = FeatureView(
    name="user_realtime_features",
    entities=["user_id"],
    ttl=timedelta(hours=1),
    schema=[
        Field(name="clicks_last_hour", dtype=Int64),
        Field(name="purchases_last_hour", dtype=Int64),
        Field(name="avg_session_duration", dtype=Float32),
    ],
    online=True,
    source=user_activity_stream,
)

# 6. 定义特征服务（Feature Service）
from feast import FeatureService

recommendation_features = FeatureService(
    name="recommendation_v1",
    features=[
        user_features[["total_orders", "avg_order_value", "favorite_category"]],
        user_realtime_features[["clicks_last_hour"]],
    ],
    tags={"model": "recommendation", "version": "v1"}
)
```

### 2.3 配置Feature Store

```yaml
# feature_store.yaml
project: ecommerce_ml
registry: s3://my-bucket/feast/registry.db
provider: aws  # 或local, gcp

online_store:
  type: redis
  connection_string: "redis:6379"

offline_store:
  type: file  # 或 snowflake, bigquery, redshift

entity_key_serialization_version: 2
```

### 2.4 初始化和应用Feature Store

```bash
# 1. 初始化Feature Store
feast init my_feature_repo
cd my_feature_repo

# 2. 定义特征（见上面的Python代码）

# 3. 应用到Registry
feast apply

# 4. 物化特征到在线存储
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")

# 5. 启动Feature Server（可选）
feast serve
```

## 3. 在线特征服务（Redis）

### 3.1 物化特征到Redis

```python
"""
将离线特征物化到在线存储
"""
from feast import FeatureStore
from datetime import datetime, timedelta

# 初始化Feature Store
store = FeatureStore(repo_path=".")

# 物化特征到Redis（增量）
store.materialize_incremental(end_date=datetime.now())

# 或物化指定时间范围
store.materialize(
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now()
)
```

### 3.2 在线特征查询

```python
"""
实时获取特征用于推理
"""
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path=".")

# 1. 单个实体查询
entity_rows = pd.DataFrame({
    "user_id": [1001, 1002, 1003]
})

# 获取特征
features = store.get_online_features(
    features=[
        "user_features:total_orders",
        "user_features:avg_order_value",
        "user_features:favorite_category",
        "user_realtime_features:clicks_last_hour",
    ],
    entity_rows=entity_rows
).to_df()

print(features)
#    user_id  total_orders  avg_order_value  favorite_category  clicks_last_hour
# 0     1001           45            128.50        Electronics                12
# 1     1002           12             89.20              Books                 3
# 2     1003           78            256.80             Sports                 7

# 2. 使用Feature Service
features = store.get_online_features(
    feature_service="recommendation_v1",
    entity_rows=entity_rows
).to_df()

# 3. 推理时获取特征
def predict(user_id):
    """实时预测函数"""
    # 获取特征
    entity_rows = pd.DataFrame({"user_id": [user_id]})
    features = store.get_online_features(
        feature_service="recommendation_v1",
        entity_rows=entity_rows
    ).to_df()

    # 转换为模型输入
    X = features[["total_orders", "avg_order_value", "clicks_last_hour"]].values

    # 模型推理
    prediction = model.predict(X)

    return prediction[0]
```

## 4. 离线特征存储（Parquet/Delta）

### 4.1 批量特征计算

```python
"""
使用Spark批量计算特征
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()

# 读取原始数据
orders = spark.read.parquet("s3://data/orders/")
users = spark.read.parquet("s3://data/users/")

# 计算用户特征
user_features = orders.groupBy("user_id").agg(
    count("*").alias("total_orders"),
    sum("amount").alias("total_spent"),
    avg("amount").alias("avg_order_value"),
    max("order_date").alias("last_order_date")
)

# 计算衍生特征
current_date = current_date()
user_features = user_features.withColumn(
    "days_since_last_order",
    datediff(current_date, col("last_order_date"))
)

# 计算最喜欢的类别
favorite_category = orders \
    .join(items, "item_id") \
    .groupBy("user_id", "category") \
    .agg(count("*").alias("category_count")) \
    .withColumn(
        "rank",
        row_number().over(
            Window.partitionBy("user_id").orderBy(col("category_count").desc())
        )
    ) \
    .filter(col("rank") == 1) \
    .select("user_id", col("category").alias("favorite_category"))

# 合并特征
user_features = user_features.join(favorite_category, "user_id", "left")

# 添加时间戳
user_features = user_features.withColumn(
    "event_timestamp",
    current_timestamp()
).withColumn(
    "created_timestamp",
    current_timestamp()
)

# 保存为Parquet
user_features.write.mode("overwrite").parquet(
    "s3://my-bucket/features/user_stats.parquet"
)
```

### 4.2 训练数据集生成

```python
"""
使用Feast生成训练数据集
"""
from feast import FeatureStore
from datetime import datetime, timedelta
import pandas as pd

store = FeatureStore(repo_path=".")

# 1. 准备训练样本（带标签）
training_samples = pd.DataFrame({
    "user_id": [1001, 1002, 1003, 1004],
    "item_id": [5001, 5002, 5003, 5004],
    "event_timestamp": [
        datetime(2026, 2, 1, 10, 0),
        datetime(2026, 2, 1, 11, 0),
        datetime(2026, 2, 1, 12, 0),
        datetime(2026, 2, 1, 13, 0),
    ],
    "purchased": [1, 0, 1, 0]  # 标签
})

# 2. Point-in-time correct join（时间点正确连接）
training_data = store.get_historical_features(
    entity_df=training_samples,
    features=[
        "user_features:total_orders",
        "user_features:avg_order_value",
        "user_features:favorite_category",
        "item_features:price",
        "item_features:category",
        "item_features:popularity_score",
    ]
).to_df()

print(training_data)
#    user_id  item_id  event_timestamp  purchased  total_orders  avg_order_value  ...
# 0     1001     5001  2026-02-01 10:00          1            45            128.50  ...
# 1     1002     5002  2026-02-01 11:00          0            12             89.20  ...

# 3. 保存训练数据
training_data.to_parquet("training_data.parquet")

# 4. 训练模型
from sklearn.ensemble import RandomForestClassifier

# 特征列
feature_cols = [
    "total_orders", "avg_order_value", "price", "popularity_score"
]

X_train = training_data[feature_cols]
y_train = training_data["purchased"]

model = RandomForestClassifier()
model.fit(X_train, y_train)
```

## 5. 特征血缘与版本管理

### 5.1 特征血缘追踪

```python
"""
特征血缘系统
"""
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime

@dataclass
class FeatureLineage:
    """特征血缘"""
    feature_name: str
    version: str
    source_tables: List[str]
    transformation_sql: str
    dependencies: List[str]
    created_at: datetime
    created_by: str

# 记录特征血缘
lineage = FeatureLineage(
    feature_name="user_features.avg_order_value",
    version="v2.1",
    source_tables=[
        "warehouse.orders",
        "warehouse.users"
    ],
    transformation_sql="""
        SELECT
            user_id,
            AVG(amount) as avg_order_value,
            CURRENT_TIMESTAMP() as event_timestamp
        FROM warehouse.orders
        WHERE order_status = 'completed'
        GROUP BY user_id
    """,
    dependencies=[
        "user_features.total_spent",
        "user_features.total_orders"
    ],
    created_at=datetime.now(),
    created_by="data-team"
)

# 保存到元数据存储
save_lineage(lineage)

# 查询特征血缘
def get_feature_lineage(feature_name):
    """查询特征的完整血缘"""
    lineage = load_lineage(feature_name)

    print(f"特征: {lineage.feature_name}")
    print(f"版本: {lineage.version}")
    print(f"数据源: {', '.join(lineage.source_tables)}")
    print(f"依赖特征: {', '.join(lineage.dependencies)}")
    print(f"创建时间: {lineage.created_at}")
```

### 5.2 特征版本管理

```python
"""
特征版本控制
"""
from feast import FeatureView

# 版本1
user_features_v1 = FeatureView(
    name="user_features",
    entities=["user_id"],
    features=[
        Feature(name="total_orders", dtype=ValueType.INT64),
        Feature(name="total_spent", dtype=ValueType.DOUBLE),
    ],
    tags={"version": "v1", "deprecated": "true"}
)

# 版本2（添加新特征）
user_features_v2 = FeatureView(
    name="user_features",
    entities=["user_id"],
    features=[
        Feature(name="total_orders", dtype=ValueType.INT64),
        Feature(name="total_spent", dtype=ValueType.DOUBLE),
        Feature(name="avg_order_value", dtype=ValueType.DOUBLE),  # 新增
        Feature(name="favorite_category", dtype=ValueType.STRING),  # 新增
    ],
    tags={"version": "v2", "stable": "true"}
)

# 使用特定版本的特征
def get_features_by_version(version):
    """按版本获取特征"""
    if version == "v1":
        return user_features_v1
    elif version == "v2":
        return user_features_v2
    else:
        raise ValueError(f"Unknown version: {version}")

# 特征版本兼容性检查
def check_compatibility(old_version, new_version):
    """检查版本兼容性"""
    old_features = set(f.name for f in old_version.features)
    new_features = set(f.name for f in new_version.features)

    removed = old_features - new_features
    added = new_features - old_features

    if removed:
        print(f"⚠️ 警告: 以下特征被移除: {removed}")
        return False

    if added:
        print(f"✅ 新增特征: {added}")

    return True
```

## 6. 实时特征计算（Flink）

### 6.1 Flink实时特征计算

```python
"""
使用Flink计算实时特征
"""
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, EnvironmentSettings

# 创建环境
env = StreamExecutionEnvironment.get_execution_environment()
settings = EnvironmentSettings.new_instance().in_streaming_mode().build()
table_env = StreamTableEnvironment.create(env, environment_settings=settings)

# 1. 定义Kafka Source
table_env.execute_sql("""
    CREATE TABLE user_events (
        user_id BIGINT,
        event_type STRING,
        item_id BIGINT,
        timestamp BIGINT,
        event_time AS TO_TIMESTAMP(FROM_UNIXTIME(timestamp)),
        WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'user_events',
        'properties.bootstrap.servers' = 'localhost:9092',
        'properties.group.id' = 'feature_computation',
        'scan.startup.mode' = 'latest-offset',
        'format' = 'json'
    )
""")

# 2. 计算实时特征
table_env.execute_sql("""
    CREATE TABLE user_realtime_features (
        user_id BIGINT,
        window_start TIMESTAMP(3),
        window_end TIMESTAMP(3),
        clicks_last_hour BIGINT,
        purchases_last_hour BIGINT,
        unique_items_viewed BIGINT,
        PRIMARY KEY (user_id) NOT ENFORCED
    ) WITH (
        'connector' = 'upsert-kafka',
        'topic' = 'user_features',
        'properties.bootstrap.servers' = 'localhost:9092',
        'key.format' = 'json',
        'value.format' = 'json'
    )
""")

# 3. 实时聚合
table_env.execute_sql("""
    INSERT INTO user_realtime_features
    SELECT
        user_id,
        TUMBLE_START(event_time, INTERVAL '1' HOUR) as window_start,
        TUMBLE_END(event_time, INTERVAL '1' HOUR) as window_end,
        SUM(CASE WHEN event_type = 'click' THEN 1 ELSE 0 END) as clicks_last_hour,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as purchases_last_hour,
        COUNT(DISTINCT item_id) as unique_items_viewed
    FROM user_events
    GROUP BY
        user_id,
        TUMBLE(event_time, INTERVAL '1' HOUR)
""")
```

### 6.2 实时特征写入Redis

```python
"""
Flink计算的实时特征写入Redis
"""
from pyflink.datastream.connectors import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema
import redis
import json

# Redis连接
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def process_feature(value):
    """处理特征并写入Redis"""
    feature = json.loads(value)

    user_id = feature['user_id']

    # 构造Redis key
    key = f"feast:user_realtime_features:{user_id}"

    # 写入Redis（Hash结构）
    redis_client.hmset(key, {
        'clicks_last_hour': feature['clicks_last_hour'],
        'purchases_last_hour': feature['purchases_last_hour'],
        'unique_items_viewed': feature['unique_items_viewed'],
        '_timestamp': feature['window_end']
    })

    # 设置TTL（1小时）
    redis_client.expire(key, 3600)

# Flink DataStream处理
def main():
    env = StreamExecutionEnvironment.get_execution_environment()

    # Kafka Consumer
    kafka_consumer = FlinkKafkaConsumer(
        topics='user_features',
        deserialization_schema=SimpleStringSchema(),
        properties={'bootstrap.servers': 'localhost:9092'}
    )

    # 处理流
    stream = env.add_source(kafka_consumer)
    stream.map(process_feature)

    env.execute("Feature Store Writer")

if __name__ == '__main__':
    main()
```

## 7. 最佳实践

### 7.1 特征工程规范

```python
"""
特征工程最佳实践
"""

# 1. 特征命名规范
# ✅ 好的命名
"user_total_orders_30d"        # 明确时间窗口
"item_price_normalized"        # 明确处理方式
"user_category_preference_top3"  # 明确含义

# ❌ 不好的命名
"feature1"
"x"
"tmp_var"

# 2. 特征文档化
user_features = FeatureView(
    name="user_features",
    features=[
        Feature(
            name="total_orders",
            dtype=ValueType.INT64,
            description="用户历史订单总数（所有时间）",
            labels={"category": "transaction", "sensitivity": "low"}
        ),
        Feature(
            name="avg_order_value",
            dtype=ValueType.DOUBLE,
            description="用户平均订单金额（RMB），计算方式: total_spent / total_orders",
            labels={"category": "transaction", "sensitivity": "medium"}
        ),
    ]
)

# 3. 特征验证
def validate_features(features_df):
    """特征质量检查"""
    # 检查空值比例
    null_ratio = features_df.isnull().sum() / len(features_df)
    if (null_ratio > 0.1).any():
        print(f"⚠️ 警告: 以下特征空值过多: {null_ratio[null_ratio > 0.1]}")

    # 检查数据分布
    for col in features_df.select_dtypes(include=['float64', 'int64']).columns:
        if features_df[col].std() == 0:
            print(f"⚠️ 警告: 特征 {col} 方差为0")

    # 检查异常值
    for col in features_df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = features_df[col].quantile(0.25)
        Q3 = features_df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((features_df[col] < Q1 - 3*IQR) | (features_df[col] > Q3 + 3*IQR)).sum()
        if outliers > len(features_df) * 0.05:
            print(f"⚠️ 警告: 特征 {col} 有 {outliers} 个异常值")
```

### 7.2 性能优化

```
Feature Store性能优化清单:
├── 离线存储
│   ├── 使用分区（按日期/用户ID）
│   ├── 列式存储（Parquet/ORC）
│   ├── 数据压缩（Snappy/ZSTD）
│   └── 预聚合常用特征
│
├── 在线存储
│   ├── Redis集群（高可用）
│   ├── 特征TTL设置（避免内存溢出）
│   ├── 批量查询（减少网络开销）
│   └── 缓存热点特征
│
└── 特征计算
    ├── 增量计算（而非全量）
    ├── 异步物化（后台任务）
    ├── 并行计算（Spark/Flink）
    └── 特征复用（避免重复计算）
```

Feature Store完整教程完成！
