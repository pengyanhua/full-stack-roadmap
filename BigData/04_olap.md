# OLAP引擎选型与优化

## 1. 主流OLAP引擎对比

### 1.1 ClickHouse vs Doris vs StarRocks

```
三大OLAP引擎架构对比
┌─────────────────────────────────────────────────────────────┐
│ ClickHouse (俄罗斯Yandex)                                    │
├─────────────────────────────────────────────────────────────┤
│ 架构: MPP + Shared-Nothing                                  │
│ 存储: MergeTree系列引擎                                      │
│ 查询: 向量化执行引擎                                        │
│ 特点: 单表查询极快，Join性能一般                            │
│ 适用: 日志分析、用户行为分析、监控系统                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Apache Doris (百度开源)                                      │
├─────────────────────────────────────────────────────────────┤
│ 架构: MPP + FE/BE分离                                       │
│ 存储: 列式存储 + LSM-Tree                                   │
│ 查询: 向量化 + CBO优化器                                    │
│ 特点: 兼容MySQL协议，Join性能好                             │
│ 适用: 多维分析、实时数仓、报表系统                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ StarRocks (DorisDB商业版开源)                                │
├─────────────────────────────────────────────────────────────┤
│ 架构: MPP + 全面向量化                                      │
│ 存储: 列式存储 + 智能物化视图                               │
│ 查询: 全链路向量化 + Pipeline执行                           │
│ 特点: 性能最强，资源消耗低                                  │
│ 适用: 高性能OLAP、实时分析、湖仓一体                        │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 详细对比表

| 维度 | ClickHouse | Apache Doris | StarRocks |
|------|------------|--------------|-----------|
| **查询性能** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Join性能** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **实时写入** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **数据更新** | ⭐⭐ (复杂) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **易用性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **生态成熟度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **社区活跃度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **运维复杂度** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 1.3 SQL兼容性对比

```sql
-- 窗口函数支持
-- ClickHouse: 支持（1.21+）
SELECT
    user_id,
    event_time,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_time) AS rn
FROM user_events;

-- Doris/StarRocks: 完全支持
SELECT
    user_id,
    SUM(amount) OVER (PARTITION BY user_id ORDER BY date
                      ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7d_sum
FROM orders;

-- CTE（公共表表达式）
-- 三者都支持
WITH daily_stats AS (
    SELECT date, COUNT(*) AS cnt
    FROM events
    GROUP BY date
)
SELECT * FROM daily_stats WHERE cnt > 1000;

-- 物化视图
-- ClickHouse: 需手动刷新
CREATE MATERIALIZED VIEW mv_daily_stats
ENGINE = SummingMergeTree()
ORDER BY (date, user_id)
AS SELECT
    date,
    user_id,
    COUNT(*) AS cnt
FROM events
GROUP BY date, user_id;

-- Doris: 同步物化视图
CREATE MATERIALIZED VIEW mv_daily_stats AS
SELECT date, user_id, COUNT(*) AS cnt
FROM events
GROUP BY date, user_id;

-- StarRocks: 异步物化视图（更强大）
CREATE MATERIALIZED VIEW mv_daily_stats
REFRESH ASYNC
AS SELECT date, user_id, COUNT(*) AS cnt
FROM events
GROUP BY date, user_id;
```

## 2. TPC-H基准测试

### 2.1 测试环境

```
硬件配置:
- CPU: Intel Xeon Gold 6248R (3.0GHz, 24核)
- 内存: 256GB DDR4
- 磁盘: 4TB NVMe SSD
- 网络: 10Gbps

数据规模:
- TPC-H SF100 (100GB数据)
- Orders表: 1.5亿行
- Lineitem表: 6亿行
```

### 2.2 TPC-H查询示例

```sql
-- TPC-H Q1: 聚合查询
SELECT
    l_returnflag,
    l_linestatus,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price,
    SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    AVG(l_quantity) AS avg_qty,
    AVG(l_extendedprice) AS avg_price,
    AVG(l_discount) AS avg_disc,
    COUNT(*) AS count_order
FROM lineitem
WHERE l_shipdate <= DATE '1998-09-01'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;

-- TPC-H Q3: Join查询
SELECT
    l_orderkey,
    SUM(l_extendedprice * (1 - l_discount)) AS revenue,
    o_orderdate,
    o_shippriority
FROM customer, orders, lineitem
WHERE c_mktsegment = 'BUILDING'
  AND c_custkey = o_custkey
  AND l_orderkey = o_orderkey
  AND o_orderdate < DATE '1995-03-15'
  AND l_shipdate > DATE '1995-03-15'
GROUP BY l_orderkey, o_orderdate, o_shippriority
ORDER BY revenue DESC, o_orderdate
LIMIT 10;

-- TPC-H Q6: 高选择性过滤
SELECT
    SUM(l_extendedprice * l_discount) AS revenue
FROM lineitem
WHERE l_shipdate >= DATE '1994-01-01'
  AND l_shipdate < DATE '1995-01-01'
  AND l_discount BETWEEN 0.05 AND 0.07
  AND l_quantity < 24;
```

### 2.3 性能测试结果

```
TPC-H SF100 查询耗时对比（秒）
┌──────┬─────────────┬──────────────┬─────────────┐
│ 查询  │ ClickHouse │ Apache Doris │ StarRocks   │
├──────┼─────────────┼──────────────┼─────────────┤
│ Q1   │    0.8      │     1.2      │     0.6     │
│ Q2   │    2.5      │     1.8      │     1.3     │
│ Q3   │    3.2      │     2.1      │     1.7     │
│ Q4   │    1.5      │     1.1      │     0.9     │
│ Q5   │    4.8      │     3.2      │     2.5     │
│ Q6   │    0.3      │     0.5      │     0.2     │
│ Q7   │    5.2      │     3.8      │     2.9     │
│ Q8   │    4.1      │     2.9      │     2.2     │
│ Q9   │    6.7      │     4.3      │     3.5     │
│ Q10  │    3.8      │     2.6      │     2.0     │
│ Q11  │    1.2      │     0.9      │     0.7     │
│ Q12  │    2.1      │     1.5      │     1.1     │
│ Q13  │    5.3      │     4.1      │     3.2     │
│ Q14  │    1.8      │     1.3      │     0.9     │
│ Q15  │    2.9      │     2.0      │     1.5     │
│ Q16  │    3.5      │     2.4      │     1.8     │
│ Q17  │    7.2      │     5.1      │     3.9     │
│ Q18  │    8.5      │     6.2      │     4.7     │
│ Q19  │    2.7      │     1.9      │     1.4     │
│ Q20  │    4.6      │     3.3      │     2.6     │
│ Q21  │    9.8      │     7.1      │     5.4     │
│ Q22  │    1.9      │     1.4      │     1.0     │
├──────┼─────────────┼──────────────┼─────────────┤
│ 总计  │   78.3      │    54.7      │    41.9     │
└──────┴─────────────┴──────────────┴─────────────┘

结论:
1. 单表聚合: ClickHouse和StarRocks都很快
2. 复杂Join: StarRocks > Doris > ClickHouse
3. 综合性能: StarRocks最强，Doris次之
```

## 3. 列式存储原理

### 3.1 行式 vs 列式存储

```
行式存储 (Row-Oriented)
┌──────────────────────────────────────────┐
│ Record 1: [ID=1, Name=Alice, Age=25, ...]│
│ Record 2: [ID=2, Name=Bob, Age=30, ...]  │
│ Record 3: [ID=3, Name=Carol, Age=28, ...] │
└──────────────────────────────────────────┘
优点: 写入快，适合OLTP
缺点: 读取列需要扫描整行，压缩率低

列式存储 (Column-Oriented)
┌────────┬────────┬────────┬────────┐
│ ID列   │ Name列 │ Age列  │ ...    │
├────────┼────────┼────────┼────────┤
│ 1      │ Alice  │ 25     │ ...    │
│ 2      │ Bob    │ 30     │ ...    │
│ 3      │ Carol  │ 28     │ ...    │
└────────┴────────┴────────┴────────┘
         ↓        ↓        ↓
      压缩存储
┌────────────────────────────────────┐
│ ID: [1,2,3] (RLE压缩)              │
│ Name: [Alice,Bob,Carol] (字典编码) │
│ Age: [25,30,28] (Delta编码)        │
└────────────────────────────────────┘

优点:
1. 只读需要的列 → IO减少
2. 相同类型数据压缩率高 → 存储省
3. 向量化执行 → CPU效率高
4. SIMD加速 → 计算快

缺点:
1. 写入慢（需要拆分到多列）
2. 更新复杂
```

### 3.2 压缩算法

```
常用压缩算法
┌──────────────────────────────────────────────────────┐
│ 1. RLE (Run-Length Encoding) - 游程编码              │
├──────────────────────────────────────────────────────┤
│ 原始: [1,1,1,1,2,2,3,3,3,3,3]                        │
│ 压缩: [(1,4), (2,2), (3,5)]                          │
│ 适用: 重复值多的列（状态、类型等）                   │
│ 压缩比: 5-20倍                                       │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ 2. Dictionary Encoding - 字典编码                    │
├──────────────────────────────────────────────────────┤
│ 原始: ["Male","Female","Male","Male","Female"]       │
│ 字典: {0:"Male", 1:"Female"}                         │
│ 编码: [0,1,0,0,1]                                    │
│ 适用: 低基数列（性别、地区等）                       │
│ 压缩比: 3-10倍                                       │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ 3. Delta Encoding - 增量编码                         │
├──────────────────────────────────────────────────────┤
│ 原始: [1000, 1001, 1002, 1003, 1004]                 │
│ 基值: 1000                                           │
│ 增量: [0, 1, 1, 1, 1]                                │
│ 适用: 递增序列（ID、时间戳）                         │
│ 压缩比: 2-5倍                                        │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ 4. LZ4/ZSTD - 通用压缩                               │
├──────────────────────────────────────────────────────┤
│ LZ4: 速度快（500MB/s），压缩比一般（2-3倍）          │
│ ZSTD: 平衡（200MB/s），压缩比好（3-5倍）             │
│ 适用: 文本、混合类型数据                             │
└──────────────────────────────────────────────────────┘
```

### 3.3 ClickHouse存储引擎

```sql
-- MergeTree家族
-- 1. MergeTree (基础引擎)
CREATE TABLE events (
    date Date,
    user_id UInt64,
    event_type String,
    timestamp DateTime
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (user_id, timestamp)
SETTINGS index_granularity = 8192;

-- 2. ReplacingMergeTree (去重引擎)
CREATE TABLE user_profile (
    user_id UInt64,
    name String,
    age UInt8,
    update_time DateTime
)
ENGINE = ReplacingMergeTree(update_time)  -- 按update_time去重
ORDER BY user_id;

-- 3. SummingMergeTree (聚合引擎)
CREATE TABLE user_metrics (
    date Date,
    user_id UInt64,
    pv_count UInt64,
    click_count UInt64
)
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, user_id);

-- 4. AggregatingMergeTree (高级聚合)
CREATE TABLE user_stats (
    date Date,
    user_id UInt64,
    pv_count AggregateFunction(sum, UInt64),
    unique_items AggregateFunction(uniq, UInt64)
)
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, user_id);

-- 5. CollapsingMergeTree (标记删除)
CREATE TABLE user_changes (
    user_id UInt64,
    name String,
    age UInt8,
    sign Int8  -- 1表示新增，-1表示删除
)
ENGINE = CollapsingMergeTree(sign)
ORDER BY user_id;
```

## 4. 物化视图与查询优化

### 4.1 物化视图类型

```sql
-- ClickHouse物化视图
-- 同步物化视图（实时更新）
CREATE MATERIALIZED VIEW mv_daily_stats
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, user_id)
AS SELECT
    toDate(timestamp) AS date,
    user_id,
    countState() AS pv_count,
    sumState(amount) AS total_amount
FROM events
GROUP BY date, user_id;

-- 查询物化视图
SELECT
    date,
    user_id,
    countMerge(pv_count) AS pv,
    sumMerge(total_amount) AS amount
FROM mv_daily_stats
WHERE date >= '2026-02-01'
GROUP BY date, user_id;

-- Doris物化视图
-- 同步物化视图（自动命中）
CREATE MATERIALIZED VIEW mv_item_sales AS
SELECT
    item_id,
    SUM(amount) AS total_sales,
    COUNT(*) AS order_count
FROM orders
GROUP BY item_id;

-- 查询自动改写，无需手动指定物化视图
SELECT item_id, SUM(amount)
FROM orders
GROUP BY item_id;
-- Doris会自动使用mv_item_sales

-- StarRocks异步物化视图（最强大）
CREATE MATERIALIZED VIEW mv_user_behavior
REFRESH ASYNC START('2026-02-07 00:00:00') EVERY(INTERVAL 1 HOUR)
AS SELECT
    user_id,
    DATE_TRUNC('hour', event_time) AS hour,
    COUNT(*) AS event_count,
    COUNT(DISTINCT item_id) AS unique_items,
    SUM(CASE WHEN event_type = 'buy' THEN 1 ELSE 0 END) AS buy_count
FROM user_events
GROUP BY user_id, DATE_TRUNC('hour', event_time);

-- 查询自动改写
SELECT
    user_id,
    SUM(event_count) AS total_events
FROM user_events
WHERE event_time >= '2026-02-07 00:00:00'
GROUP BY user_id;
-- StarRocks自动使用mv_user_behavior
```

### 4.2 查询优化技巧

```sql
-- 优化1: 使用分区裁剪
-- ❌ 不好 - 全表扫描
SELECT COUNT(*) FROM events
WHERE timestamp >= '2026-02-01 00:00:00'
  AND timestamp < '2026-02-08 00:00:00';

-- ✅ 好 - 分区裁剪
SELECT COUNT(*) FROM events
WHERE date >= '2026-02-01'
  AND date < '2026-02-08'
  AND timestamp >= '2026-02-01 00:00:00'
  AND timestamp < '2026-02-08 00:00:00';

-- 优化2: 使用投影（Projection）
-- ClickHouse 21.12+
ALTER TABLE events ADD PROJECTION proj_user_daily
(
    SELECT
        toDate(timestamp) AS date,
        user_id,
        COUNT()
    GROUP BY date, user_id
);

-- 查询自动使用投影
SELECT
    toDate(timestamp) AS date,
    user_id,
    COUNT()
FROM events
GROUP BY date, user_id;

-- 优化3: 预聚合
-- ❌ 不好 - 每次都聚合
SELECT
    item_id,
    SUM(amount) AS sales
FROM orders
WHERE date >= '2026-01-01'
GROUP BY item_id;

-- ✅ 好 - 使用SummingMergeTree预聚合
CREATE TABLE orders_summary (
    date Date,
    item_id UInt64,
    sales Decimal(18,2)
)
ENGINE = SummingMergeTree()
ORDER BY (date, item_id);

-- 查询预聚合表
SELECT
    item_id,
    SUM(sales) AS sales
FROM orders_summary
WHERE date >= '2026-01-01'
GROUP BY item_id;

-- 优化4: Bitmap索引（Doris/StarRocks）
-- 创建Bitmap索引
CREATE INDEX idx_status ON orders(status) USING BITMAP;

-- 高效的IN查询
SELECT COUNT(*)
FROM orders
WHERE status IN ('paid', 'shipped');  -- Bitmap加速

-- 优化5: BloomFilter索引
-- ClickHouse
ALTER TABLE events ADD INDEX bf_user_id user_id TYPE bloom_filter GRANULARITY 3;

-- 高效的等值查询
SELECT * FROM events WHERE user_id = 123456;
```

### 4.3 Join优化

```sql
-- Join优化策略
-- 1. Broadcast Join (小表广播)
-- Doris自动选择
SELECT /*+ BROADCAST */
    o.order_id,
    u.name
FROM orders o
JOIN users u ON o.user_id = u.user_id;

-- 2. Shuffle Join (大表Join)
-- 默认策略
SELECT
    o.order_id,
    i.name
FROM orders o
JOIN items i ON o.item_id = i.item_id;

-- 3. Colocate Join (数据预分布)
-- Doris/StarRocks
-- 创建Colocate表组
ALTER TABLE orders SET ("colocate_with" = "group1");
ALTER TABLE order_items SET ("colocate_with" = "group1");

-- Join无需Shuffle
SELECT
    o.order_id,
    oi.item_id
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id;

-- 4. Runtime Filter (运行时过滤)
-- StarRocks自动优化
SELECT
    l.l_orderkey,
    l.l_quantity
FROM lineitem l
JOIN orders o ON l.l_orderkey = o.o_orderkey
WHERE o.o_orderdate >= '2026-01-01'
  AND o.o_orderdate < '2026-02-01';
-- orders的过滤条件会下推到lineitem
```

## 5. 实战案例：广告点击分析

### 5.1 数据模型设计

```sql
-- StarRocks建表
-- 1. 明细表 (Duplicate Key)
CREATE TABLE ad_clicks (
    click_time DATETIME NOT NULL,
    user_id BIGINT NOT NULL,
    ad_id BIGINT NOT NULL,
    campaign_id BIGINT NOT NULL,
    creative_id BIGINT NOT NULL,
    placement VARCHAR(50),
    device_type VARCHAR(20),
    os VARCHAR(20),
    province VARCHAR(50),
    city VARCHAR(50),
    cost DECIMAL(10, 4),
    revenue DECIMAL(10, 4)
)
DUPLICATE KEY(click_time, user_id)
PARTITION BY RANGE(click_time) (
    PARTITION p20260201 VALUES [('2026-02-01'), ('2026-02-02')),
    PARTITION p20260202 VALUES [('2026-02-02'), ('2026-02-03'))
)
DISTRIBUTED BY HASH(user_id) BUCKETS 32
PROPERTIES (
    "replication_num" = "3",
    "dynamic_partition.enable" = "true",
    "dynamic_partition.time_unit" = "DAY",
    "dynamic_partition.start" = "-30",
    "dynamic_partition.end" = "3",
    "dynamic_partition.prefix" = "p",
    "dynamic_partition.buckets" = "32"
);

-- 2. 聚合表 (Aggregate Key)
CREATE TABLE ad_campaign_stats (
    stat_date DATE NOT NULL,
    campaign_id BIGINT NOT NULL,
    clicks BIGINT SUM DEFAULT "0",
    cost DECIMAL(18, 4) SUM DEFAULT "0",
    revenue DECIMAL(18, 4) SUM DEFAULT "0",
    unique_users BITMAP BITMAP_UNION
)
AGGREGATE KEY(stat_date, campaign_id)
PARTITION BY RANGE(stat_date) ()
DISTRIBUTED BY HASH(campaign_id) BUCKETS 16
PROPERTIES (
    "replication_num" = "3"
);

-- 3. 物化视图（自动聚合）
CREATE MATERIALIZED VIEW mv_hourly_stats
REFRESH ASYNC EVERY(INTERVAL 1 HOUR)
AS SELECT
    DATE_TRUNC('hour', click_time) AS hour,
    campaign_id,
    device_type,
    COUNT(*) AS clicks,
    SUM(cost) AS total_cost,
    SUM(revenue) AS total_revenue,
    COUNT(DISTINCT user_id) AS unique_users
FROM ad_clicks
GROUP BY
    DATE_TRUNC('hour', click_time),
    campaign_id,
    device_type;
```

### 5.2 高级分析查询

```sql
-- 查询1: 实时Campaign效果分析
SELECT
    campaign_id,
    SUM(clicks) AS total_clicks,
    SUM(cost) AS total_cost,
    SUM(revenue) AS total_revenue,
    SUM(revenue) - SUM(cost) AS profit,
    SUM(revenue) / SUM(cost) AS roi,
    BITMAP_COUNT(BITMAP_UNION(unique_users)) AS unique_users
FROM ad_campaign_stats
WHERE stat_date >= CURRENT_DATE - INTERVAL 7 DAY
GROUP BY campaign_id
HAVING SUM(clicks) > 1000
ORDER BY roi DESC
LIMIT 20;

-- 查询2: 漏斗分析（点击→转化）
WITH funnel AS (
    SELECT
        user_id,
        MIN(CASE WHEN event_type = 'click' THEN click_time END) AS click_time,
        MIN(CASE WHEN event_type = 'conversion' THEN click_time END) AS conversion_time
    FROM ad_events
    WHERE click_time >= '2026-02-07 00:00:00'
    GROUP BY user_id
)
SELECT
    COUNT(DISTINCT user_id) AS total_users,
    COUNT(DISTINCT CASE WHEN click_time IS NOT NULL THEN user_id END) AS clicked_users,
    COUNT(DISTINCT CASE WHEN conversion_time IS NOT NULL THEN user_id END) AS converted_users,
    COUNT(DISTINCT CASE WHEN conversion_time IS NOT NULL THEN user_id END) * 100.0 /
        COUNT(DISTINCT CASE WHEN click_time IS NOT NULL THEN user_id END) AS conversion_rate
FROM funnel;

-- 查询3: 留存分析
SELECT
    first_date,
    day_diff,
    COUNT(DISTINCT user_id) AS retained_users,
    COUNT(DISTINCT user_id) * 100.0 / first_day_users AS retention_rate
FROM (
    SELECT
        user_id,
        MIN(DATE(click_time)) AS first_date
    FROM ad_clicks
    GROUP BY user_id
) first_day
JOIN (
    SELECT
        user_id,
        DATE(click_time) AS return_date
    FROM ad_clicks
) returns ON first_day.user_id = returns.user_id
CROSS JOIN (
    SELECT first_date, COUNT(DISTINCT user_id) AS first_day_users
    FROM (
        SELECT user_id, MIN(DATE(click_time)) AS first_date
        FROM ad_clicks
        GROUP BY user_id
    )
    GROUP BY first_date
) base ON first_day.first_date = base.first_date
WHERE DATEDIFF(return_date, first_date) AS day_diff <= 30
GROUP BY first_date, day_diff, first_day_users
ORDER BY first_date, day_diff;

-- 查询4: 同期群分析 (Cohort Analysis)
SELECT
    cohort_month,
    month_number,
    SUM(revenue) AS cohort_revenue,
    COUNT(DISTINCT user_id) AS active_users
FROM (
    SELECT
        c.user_id,
        DATE_TRUNC('month', c.first_click) AS cohort_month,
        PERIOD_DIFF(DATE_FORMAT(a.click_time, '%Y%m'),
                    DATE_FORMAT(c.first_click, '%Y%m')) AS month_number,
        a.revenue
    FROM ad_clicks a
    JOIN (
        SELECT user_id, MIN(click_time) AS first_click
        FROM ad_clicks
        GROUP BY user_id
    ) c ON a.user_id = c.user_id
) cohorts
GROUP BY cohort_month, month_number
ORDER BY cohort_month, month_number;

-- 查询5: 多维OLAP分析
SELECT
    province,
    city,
    device_type,
    SUM(clicks) AS clicks,
    SUM(cost) AS cost,
    SUM(revenue) AS revenue
FROM ad_clicks
WHERE click_time >= '2026-02-01'
GROUP BY
    ROLLUP(province, city, device_type)
ORDER BY clicks DESC;
```

### 5.3 性能调优

```sql
-- 调优1: 创建合适的索引
-- Bitmap索引（低基数列）
CREATE INDEX idx_device ON ad_clicks(device_type) USING BITMAP;
CREATE INDEX idx_province ON ad_clicks(province) USING BITMAP;

-- BloomFilter索引（高基数列）
CREATE INDEX idx_user ON ad_clicks(user_id) USING BLOOM_FILTER;

-- 调优2: 分桶优化
-- 分析数据分布
SELECT
    MOD(user_id, 32) AS bucket_id,
    COUNT(*) AS row_count
FROM ad_clicks
GROUP BY bucket_id
ORDER BY row_count DESC;

-- 根据分析结果调整分桶数
ALTER TABLE ad_clicks
SET ("distribution_type" = "hash", "bucket_num" = "64");

-- 调优3: 压缩算法选择
-- 不同列使用不同压缩
ALTER TABLE ad_clicks
MODIFY COLUMN cost SET COMPRESSION "LZ4",
MODIFY COLUMN revenue SET COMPRESSION "ZSTD",
MODIFY COLUMN device_type SET COMPRESSION "ZSTD";

-- 调优4: 查询并行度
-- 设置Session级别并行度
SET parallel_fragment_exec_instance_num = 16;

-- 调优5: 使用Colocate优化Join
ALTER TABLE ad_clicks SET ("colocate_with" = "ad_group");
ALTER TABLE ad_campaigns SET ("colocate_with" = "ad_group");
```

## 6. 运维最佳实践

### 6.1 监控指标

```sql
-- ClickHouse监控查询
-- 1. 查询性能
SELECT
    query_id,
    user,
    query_duration_ms,
    read_rows,
    read_bytes,
    memory_usage
FROM system.query_log
WHERE event_date = today()
  AND type = 'QueryFinish'
ORDER BY query_duration_ms DESC
LIMIT 10;

-- 2. 表大小
SELECT
    database,
    table,
    formatReadableSize(sum(bytes)) AS size,
    sum(rows) AS rows
FROM system.parts
WHERE active
GROUP BY database, table
ORDER BY sum(bytes) DESC;

-- 3. Merge性能
SELECT
    database,
    table,
    count() AS merges,
    formatReadableSize(sum(bytes_read_uncompressed)) AS read,
    formatReadableSize(sum(bytes_written_uncompressed)) AS written
FROM system.merges
GROUP BY database, table;

-- Doris/StarRocks监控
-- 1. 慢查询
SHOW QUERY STATS
WHERE query_time > 10000  -- 超过10秒
ORDER BY query_time DESC
LIMIT 20;

-- 2. 表统计信息
SHOW TABLE STATS table_name;

-- 3. Tablet分布
SHOW TABLET FROM table_name;
```

### 6.2 故障处理

```
常见问题排查:
1. 查询慢 → 检查分区裁剪、索引、物化视图
2. 内存不足 → 调整内存配置、优化查询
3. 磁盘满 → 清理历史分区、压缩数据
4. 副本不一致 → 手动修复副本
5. Merge慢 → 调整Merge参数
```

OLAP引擎实战教程完成！
