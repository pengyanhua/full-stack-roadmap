# 容量规划与估算

## 一、核心指标

### 流量指标

```
QPS (Queries Per Second)
├── 平均 QPS = 日请求量 / 86400
├── 峰值 QPS = 平均 QPS × 峰值系数（通常 2-10 倍）
└── 设计 QPS = 峰值 QPS × 冗余系数（通常 1.5-2 倍）

示例：电商系统
- 日活用户：100 万
- 人均请求：50 次/天
- 日请求量：5000 万
- 平均 QPS：5000万 / 86400 ≈ 580
- 峰值 QPS（10倍）：5800
- 设计 QPS（2倍冗余）：11600 ≈ 12000
```

### 数据量指标

```
存储量估算：
├── 单条数据大小
├── 日增量 = 日新增条数 × 单条大小
├── 年增量 = 日增量 × 365
└── 总存储 = 历史数据 + 预留增长（通常 2-3 年）

示例：订单系统
- 日订单量：100 万
- 单条订单：2 KB（含索引）
- 日增量：2 GB
- 年增量：730 GB ≈ 0.7 TB
- 3 年预估：2.1 TB
- 含副本（3副本）：6.3 TB
```

### 带宽指标

```
带宽 = QPS × 平均响应大小

示例：
- QPS：10000
- 平均响应：10 KB
- 带宽需求：10000 × 10 KB = 100 MB/s = 800 Mbps
- 冗余系数 2 倍：1.6 Gbps
```

---

## 二、估算方法论

### 2 的幂次估算法

```
常用数值速记：

2^10 = 1024 ≈ 1 千 (K)
2^20 = 1048576 ≈ 1 百万 (M)
2^30 ≈ 10 亿 (G)
2^40 ≈ 1 万亿 (T)

时间转换：
1 天 = 86400 秒 ≈ 10^5 秒
1 年 ≈ 3 × 10^7 秒

网络延迟参考：
- 内存访问：100 ns
- SSD 随机读：100 μs
- 机械硬盘：10 ms
- 同机房网络：0.5 ms
- 跨机房网络：30-100 ms
- 跨国网络：100-300 ms
```

### 估算实战：设计 Twitter

```
需求：
- 3 亿月活用户
- 50% 日活 = 1.5 亿
- 每人每天发 2 条推文
- 每人每天读 100 条推文

写入量：
- 日发推量 = 1.5 亿 × 2 = 3 亿条/天
- QPS = 3 亿 / 86400 ≈ 3500
- 峰值 QPS ≈ 35000

读取量：
- 日读取量 = 1.5 亿 × 100 = 150 亿条/天
- QPS = 150 亿 / 86400 ≈ 170000
- 峰值 QPS ≈ 1700000 (170万)

存储量：
- 推文大小：280 字符 × 2 + 元数据 ≈ 1 KB
- 日增量 = 3 亿 × 1 KB = 300 GB
- 年增量 = 300 GB × 365 ≈ 110 TB
- 5 年 + 副本 ≈ 1.6 PB

结论：
- 读写比 ≈ 50:1（典型读多写少）
- 需要强大的缓存层
- 需要分布式存储
```

### 估算实战：设计 URL 短链服务

```
需求：
- 每月新增 1 亿短链
- 读写比 100:1

写入：
- 月写入 = 1 亿
- QPS = 1 亿 / (30 × 86400) ≈ 40
- 峰值 ≈ 400

读取：
- 月读取 = 100 亿
- QPS = 100 亿 / (30 × 86400) ≈ 4000
- 峰值 ≈ 40000

存储：
- 短链长度：7 字符（可表示 62^7 ≈ 3.5 万亿）
- 单条记录：短链(7B) + 长链(100B) + 元数据(50B) ≈ 200B
- 月增量 = 1 亿 × 200B = 20 GB
- 5 年 = 1.2 TB

URL 编码方案：
- Base62: [a-zA-Z0-9]
- 7 位可表示：62^7 = 3,521,614,606,208 (3.5万亿)
- 足够使用
```

---

## 三、服务器容量规划

### 单机容量评估

```python
# 估算单机处理能力
class ServerCapacity:
    """
    典型配置：8核16G

    CPU 密集型（计算、加密）：
    - 理论 QPS = 核数 × 单核处理能力
    - 实际按 70% 利用率：8 × 1000 × 0.7 = 5600

    IO 密集型（Web服务）：
    - 取决于 IO 等待时间
    - 通常 1000-5000 QPS

    内存密集型（缓存）：
    - 取决于数据量和命中率
    - Redis 单机 10万+ QPS
    """

    @staticmethod
    def estimate_web_server(
        qps_target: int,
        single_server_qps: int = 2000,
        redundancy: float = 1.5
    ) -> int:
        """估算 Web 服务器数量"""
        return math.ceil(qps_target * redundancy / single_server_qps)

    @staticmethod
    def estimate_db_server(
        qps_target: int,
        read_ratio: float = 0.8,
        single_read_qps: int = 5000,
        single_write_qps: int = 1000
    ) -> dict:
        """估算数据库服务器"""
        read_qps = qps_target * read_ratio
        write_qps = qps_target * (1 - read_ratio)

        return {
            "read_replicas": math.ceil(read_qps / single_read_qps),
            "write_masters": math.ceil(write_qps / single_write_qps)
        }
```

### 常见组件容量参考

```
┌─────────────────────────────────────────────────────────────┐
│                    单机容量参考值                            │
├──────────────────┬──────────────────────────────────────────┤
│ 组件              │ 容量参考                                 │
├──────────────────┼──────────────────────────────────────────┤
│ Nginx            │ 50000+ QPS（静态）/ 10000+ QPS（反代）    │
│ Tomcat           │ 1000-3000 QPS                            │
│ Go HTTP Server   │ 10000-50000 QPS                          │
│ Node.js          │ 5000-15000 QPS                           │
├──────────────────┼──────────────────────────────────────────┤
│ MySQL            │ 3000-8000 QPS（取决于查询复杂度）         │
│ PostgreSQL       │ 3000-10000 QPS                           │
│ Redis            │ 100000+ QPS                              │
│ MongoDB          │ 10000-50000 QPS                          │
├──────────────────┼──────────────────────────────────────────┤
│ Kafka            │ 100000+ msg/s（单分区）                  │
│ RabbitMQ         │ 20000-50000 msg/s                        │
│ Elasticsearch    │ 5000-20000 QPS（搜索）                   │
└──────────────────┴──────────────────────────────────────────┘

注意：以上为参考值，实际取决于：
- 硬件配置
- 数据量大小
- 查询复杂度
- 网络环境
```

---

## 四、数据库容量规划

### MySQL 容量规划

```sql
-- 表数据量评估
-- 单表建议：500万-2000万行（需要根据实际查询优化）

-- 查看表大小
SELECT
    table_name,
    ROUND(data_length / 1024 / 1024, 2) AS data_mb,
    ROUND(index_length / 1024 / 1024, 2) AS index_mb,
    table_rows
FROM information_schema.tables
WHERE table_schema = 'your_database';

-- 估算索引大小
-- 一般索引大小 ≈ 数据大小 × 0.3-0.5
```

```python
# MySQL 容量规划
class MySQLCapacityPlanner:

    @staticmethod
    def estimate_table_size(
        row_count: int,
        avg_row_size: int,  # 字节
        index_ratio: float = 0.4
    ) -> dict:
        """估算表大小"""
        data_size = row_count * avg_row_size
        index_size = data_size * index_ratio

        return {
            "data_size_gb": data_size / (1024 ** 3),
            "index_size_gb": index_size / (1024 ** 3),
            "total_size_gb": (data_size + index_size) / (1024 ** 3)
        }

    @staticmethod
    def estimate_sharding(
        total_rows: int,
        rows_per_table: int = 10_000_000,  # 单表 1000 万行
        growth_years: int = 3
    ) -> dict:
        """估算分表数量"""
        current_tables = math.ceil(total_rows / rows_per_table)

        # 假设每年增长 50%
        future_rows = total_rows * (1.5 ** growth_years)
        future_tables = math.ceil(future_rows / rows_per_table)

        # 建议分表数为 2 的幂次
        recommended = 2 ** math.ceil(math.log2(future_tables))

        return {
            "current_need": current_tables,
            "future_need": future_tables,
            "recommended": recommended
        }
```

### Redis 容量规划

```python
# Redis 内存估算
class RedisCapacityPlanner:

    # Redis 数据结构内存开销（近似值）
    OVERHEAD = {
        "string": 56,      # key + value 基础开销
        "hash_entry": 64,  # 每个字段开销
        "list_entry": 32,  # 每个元素开销
        "set_entry": 48,   # 每个成员开销
        "zset_entry": 64,  # 每个成员开销
    }

    @staticmethod
    def estimate_string_memory(
        key_count: int,
        avg_key_size: int,
        avg_value_size: int
    ) -> float:
        """估算 String 类型内存（MB）"""
        per_key = 56 + avg_key_size + avg_value_size
        total = key_count * per_key
        return total / (1024 * 1024)

    @staticmethod
    def estimate_hash_memory(
        key_count: int,
        avg_fields: int,
        avg_field_size: int,
        avg_value_size: int
    ) -> float:
        """估算 Hash 类型内存（MB）"""
        per_field = 64 + avg_field_size + avg_value_size
        per_key = 56 + (per_field * avg_fields)
        total = key_count * per_key
        return total / (1024 * 1024)

# 示例：用户 Session 缓存
# 100 万用户，每个 Session 500 字节
memory = RedisCapacityPlanner.estimate_string_memory(
    key_count=1_000_000,
    avg_key_size=30,      # "session:user:123456"
    avg_value_size=500
)
print(f"预估内存: {memory:.2f} MB")  # ≈ 560 MB
```

---

## 五、成本估算

### 云服务成本模型

```python
class CloudCostEstimator:
    """云服务成本估算（以阿里云为参考）"""

    # 价格参考（元/月，按量付费可能更高）
    PRICES = {
        # ECS 实例（包年包月参考价）
        "ecs_4c8g": 300,
        "ecs_8c16g": 600,
        "ecs_16c32g": 1200,

        # RDS MySQL（高可用版）
        "rds_4c8g": 800,
        "rds_8c16g": 1600,
        "rds_16c64g": 4000,
        "rds_storage_gb": 1,  # 每 GB 每月

        # Redis
        "redis_8g": 800,
        "redis_16g": 1500,
        "redis_32g": 3000,

        # 带宽
        "bandwidth_mbps": 25,  # 每 Mbps 每月

        # OSS 存储
        "oss_storage_gb": 0.12,  # 每 GB 每月
        "oss_request_10k": 0.01,  # 每万次请求
    }

    def estimate_monthly_cost(
        self,
        web_servers: int,
        web_spec: str,
        db_spec: str,
        db_storage_gb: int,
        redis_spec: str,
        bandwidth_mbps: int
    ) -> dict:
        """估算月度成本"""
        costs = {
            "ecs": web_servers * self.PRICES[f"ecs_{web_spec}"],
            "rds": self.PRICES[f"rds_{db_spec}"] + db_storage_gb * self.PRICES["rds_storage_gb"],
            "redis": self.PRICES[f"redis_{redis_spec}"],
            "bandwidth": bandwidth_mbps * self.PRICES["bandwidth_mbps"],
        }
        costs["total"] = sum(costs.values())
        return costs
```

### 成本优化策略

```
┌──────────────────────────────────────────────────────────────┐
│                      成本优化策略                             │
├───────────────┬──────────────────────────────────────────────┤
│ 策略           │ 说明                                         │
├───────────────┼──────────────────────────────────────────────┤
│ 预留实例       │ 包年比按量省 30-50%                          │
│ 竞价实例       │ 可省 50-90%，适合可中断任务                   │
│ 自动伸缩       │ 低峰期自动缩容                               │
│ 冷热分离       │ 冷数据用低成本存储（OSS/S3）                  │
│ CDN 加速       │ 减少源站带宽                                 │
│ 数据压缩       │ 减少存储和传输成本                           │
│ 定期清理       │ 删除过期数据和日志                           │
└───────────────┴──────────────────────────────────────────────┘
```

---

## 六、容量规划流程

### 完整流程

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 业务分析                                                  │
│    • 用户量预估                                              │
│    • 业务场景梳理                                            │
│    • 数据生命周期                                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 指标计算                                                  │
│    • QPS 估算                                                │
│    • 存储量估算                                              │
│    • 带宽估算                                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 架构设计                                                  │
│    • 技术选型                                                │
│    • 服务拆分                                                │
│    • 数据分布                                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 资源规划                                                  │
│    • 服务器数量                                              │
│    • 数据库规格                                              │
│    • 缓存容量                                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. 压测验证                                                  │
│    • 单机压测                                                │
│    • 集群压测                                                │
│    • 瓶颈分析                                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. 持续优化                                                  │
│    • 监控告警                                                │
│    • 定期复盘                                                │
│    • 容量预警                                                │
└─────────────────────────────────────────────────────────────┘
```

### 容量规划模板

```yaml
# 容量规划文档模板
project: "电商平台"
version: "1.0"
date: "2024-01-01"

# 业务预估
business:
  daily_active_users: 1000000
  peak_multiplier: 5
  growth_rate_yearly: 50%

# 流量预估
traffic:
  average_qps: 5000
  peak_qps: 25000
  design_qps: 50000  # 2倍冗余

# 存储预估
storage:
  daily_increment_gb: 10
  retention_years: 3
  total_required_tb: 15

# 资源规划
resources:
  web_servers:
    spec: "8c16g"
    count: 10
    load_balancer: "SLB"

  database:
    type: "MySQL 8.0"
    spec: "16c64g"
    storage_gb: 2000
    replicas: 2
    sharding: 16

  cache:
    type: "Redis 6.0"
    spec: "32g"
    cluster_nodes: 6

  message_queue:
    type: "Kafka"
    brokers: 3
    partitions: 32

# 成本预估（月）
cost:
  compute: 15000
  database: 8000
  cache: 6000
  bandwidth: 5000
  storage: 2000
  total: 36000
```
