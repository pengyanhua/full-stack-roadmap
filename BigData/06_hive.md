# Hive数据仓库实战

## 1. Hive架构与原理

### 1.1 整体架构

Hive是构建在Hadoop之上的数据仓库工具，它将结构化的数据文件映射为数据库表，并提供类SQL查询语言HiveQL。
Hive的核心思想是将SQL转换为MapReduce/Tez/Spark作业，在Hadoop集群上执行大规模数据分析。

**Hive整体架构图**：

```
Hive整体架构
┌──────────────────────────────────────────────────────────────────────┐
│                           客户端层                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │ Beeline  │  │  JDBC    │  │  ODBC    │  │  Hive CLI (废弃) │    │
│  │ 命令行    │  │ Java连接 │  │ BI工具   │  │  旧版命令行       │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───────┬──────────┘    │
│       │              │              │                │               │
├───────┼──────────────┼──────────────┼────────────────┼───────────────┤
│       ↓              ↓              ↓                ↓               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    HiveServer2 (Thrift)                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │
│  │  │  SQL Parser  │→ │  Optimizer  │→ │  Execution Engine   │  │   │
│  │  │  语法解析器   │  │  查询优化器  │  │  MR / Tez / Spark  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                             │                                       │
├─────────────────────────────┼───────────────────────────────────────┤
│                             ↓                                       │
│  ┌─────────────────────────────────────────┐                        │
│  │           Metastore Service             │                        │
│  │  ┌───────────┐    ┌─────────────────┐   │                        │
│  │  │ Thrift API│    │  Backend RDBMS   │   │                        │
│  │  │ 元数据接口 │    │  MySQL/PostgreSQL│   │                        │
│  │  └───────────┘    └─────────────────┘   │                        │
│  └─────────────────────────────────────────┘                        │
│                             │                                       │
├─────────────────────────────┼───────────────────────────────────────┤
│                             ↓                                       │
│  ┌─────────────────┐  ┌─────────────────┐                           │
│  │      HDFS       │  │      YARN       │                           │
│  │  数据存储        │  │  资源调度与计算   │                           │
│  └─────────────────┘  └─────────────────┘                           │
└──────────────────────────────────────────────────────────────────────┘
```

**SQL执行流程**：

```
SQL查询执行流程
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  SQL    │ →  │  Parser  │ →  │ Semantic │ →  │ Logical  │ →  │ Physical │
│  Input  │    │  解析AST  │    │ Analyzer │    │ Plan     │    │ Plan     │
└─────────┘    └──────────┘    │ 语义分析  │    │ 逻辑计划  │    │ 物理计划  │
                               └──────────┘    └──────────┘    └────┬─────┘
                                                                    │
                                                                    ↓
                                                              ┌──────────┐
                                                              │ MR/Tez/  │
                                                              │ Spark Job│
                                                              └──────────┘
```

**Metastore三种部署模式**：

| 模式 | 描述 | Metastore位置 | RDBMS | 适用场景 |
|------|------|--------------|-------|---------|
| **Embedded** | 内嵌模式 | 与HiveServer2同进程 | Derby (内嵌) | 开发测试，单用户 |
| **Local** | 本地模式 | 与HiveServer2同进程 | MySQL/PostgreSQL | 小规模集群 |
| **Remote** | 远程模式 | 独立进程/服务 | MySQL/PostgreSQL | 生产环境，多客户端 |

```
三种Metastore模式对比
┌─────────────────────────────────────────────────────────────────┐
│  Embedded模式                                                   │
│  ┌────────────────────────────────┐                             │
│  │ HiveServer2 + Metastore + Derby│  ← 全部在一个JVM进程中      │
│  └────────────────────────────────┘                             │
├─────────────────────────────────────────────────────────────────┤
│  Local模式                                                      │
│  ┌────────────────────────────┐    ┌─────────┐                  │
│  │ HiveServer2 + Metastore   │ →  │  MySQL  │                  │
│  └────────────────────────────┘    └─────────┘                  │
├─────────────────────────────────────────────────────────────────┤
│  Remote模式 (推荐生产使用)                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────┐           │
│  │ HiveServer2  │ →  │  Metastore   │ →  │  MySQL  │           │
│  │  (可多个)     │    │  (独立服务)   │    │         │           │
│  └──────────────┘    └──────────────┘    └─────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

**Remote模式核心配置** (hive-site.xml)：

```xml
<!-- Metastore服务端配置 -->
<property>
  <name>hive.metastore.warehouse.dir</name>
  <value>/user/hive/warehouse</value>
  <description>Hive数据仓库在HDFS上的默认路径</description>
</property>
<property>
  <name>javax.jdo.option.ConnectionURL</name>
  <value>jdbc:mysql://metastore-host:3306/hivemeta?useSSL=false</value>
</property>
<property>
  <name>javax.jdo.option.ConnectionDriverName</name>
  <value>com.mysql.cj.jdbc.Driver</value>
</property>
<property>
  <name>javax.jdo.option.ConnectionUserName</name>
  <value>hive</value>
</property>
<property>
  <name>javax.jdo.option.ConnectionPassword</name>
  <value>hive_password</value>
</property>

<!-- HiveServer2客户端配置 -->
<property>
  <name>hive.metastore.uris</name>
  <value>thrift://metastore-host:9083</value>
  <description>远程Metastore的Thrift地址</description>
</property>
```

### 1.2 执行引擎对比

Hive支持三种执行引擎，通过 `hive.execution.engine` 参数切换。

| 维度 | MapReduce | Tez | Spark |
|------|-----------|-----|-------|
| **执行模型** | Map → Shuffle → Reduce | DAG (有向无环图) | DAG + 内存计算 |
| **中间数据** | 落盘到HDFS | 内存 + 本地磁盘 | 内存优先，溢出到磁盘 |
| **延迟** | 高 (Job启动开销大) | 中 (容器复用) | 低 (内存常驻) |
| **吞吐量** | 高 | 高 | 高 |
| **资源利用** | 低 (Map/Reduce间有间隔) | 高 (Pipeline执行) | 高 (内存缓存) |
| **适合场景** | 超大规模批处理 | 交互式查询 + 批处理 | 交互式查询 + 迭代计算 |
| **稳定性** | 最成熟 | 成熟 | 需要额外调优 |
| **社区支持** | 维护模式 | Apache活跃 | Apache活跃 |
| **Hive版本** | 所有版本 | Hive 0.13+ | Hive 1.1+ |

**引擎切换配置**：

```sql
-- 使用Tez引擎 (Hive 2.x+默认)
SET hive.execution.engine=tez;

-- 使用Spark引擎
SET hive.execution.engine=spark;

-- 使用MapReduce引擎 (旧版默认)
SET hive.execution.engine=mr;
```

**Tez引擎关键配置**：

```xml
<property>
  <name>hive.execution.engine</name>
  <value>tez</value>
</property>
<property>
  <name>tez.am.resource.memory.mb</name>
  <value>4096</value>
</property>
<property>
  <name>tez.task.resource.memory.mb</name>
  <value>2048</value>
</property>
<property>
  <name>tez.grouping.min-size</name>
  <value>16777216</value>
  <description>Tez任务最小分组大小: 16MB</description>
</property>
<property>
  <name>tez.grouping.max-size</name>
  <value>1073741824</value>
  <description>Tez任务最大分组大小: 1GB</description>
</property>
```

```
MapReduce vs Tez 执行对比
┌──────────────────────────────────────────────────────────────┐
│  MapReduce: 多个独立Job，中间结果落盘                          │
│                                                              │
│  Job1                    Job2                    Job3        │
│  ┌────┐    ┌─────┐      ┌────┐    ┌─────┐      ┌────┐      │
│  │Map │→ →│Reduce│→HDFS→│Map │→ →│Reduce│→HDFS→│Map │→...  │
│  └────┘    └─────┘      └────┘    └─────┘      └────┘      │
│            磁盘IO↑                 磁盘IO↑                   │
├──────────────────────────────────────────────────────────────┤
│  Tez: 单个DAG，Pipeline执行，中间数据在内存                    │
│                                                              │
│         ┌──────┐                                             │
│         │Map-1 │──┐                                          │
│         └──────┘  │   ┌─────────┐   ┌──────────┐            │
│                   ├──→│Reduce-1 │──→│ Reduce-2 │            │
│         ┌──────┐  │   └─────────┘   └──────────┘            │
│         │Map-2 │──┘       内存传输 ↑                          │
│         └──────┘                                             │
└──────────────────────────────────────────────────────────────┘
```

### 1.3 数据模型

Hive的数据模型包括Database、Table、Partition和Bucket四个层次。

```
Hive数据模型层次
┌──────────────────────────────────────────────────────┐
│                    Database                          │
│  ┌────────────────────────────────────────────────┐  │
│  │                   Table                        │  │
│  │  ┌──────────────────────────────────────────┐  │  │
│  │  │         Partition (dt=2024-01-01)         │  │  │
│  │  │  ┌────────┐ ┌────────┐ ┌────────┐       │  │  │
│  │  │  │Bucket 0│ │Bucket 1│ │Bucket 2│       │  │  │
│  │  │  │ 文件0  │ │ 文件1  │ │ 文件2   │       │  │  │
│  │  │  └────────┘ └────────┘ └────────┘       │  │  │
│  │  └──────────────────────────────────────────┘  │  │
│  │  ┌──────────────────────────────────────────┐  │  │
│  │  │         Partition (dt=2024-01-02)         │  │  │
│  │  │  ┌────────┐ ┌────────┐ ┌────────┐       │  │  │
│  │  │  │Bucket 0│ │Bucket 1│ │Bucket 2│       │  │  │
│  │  │  └────────┘ └────────┘ └────────┘       │  │  │
│  │  └──────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

**Internal Table (内部表/管理表) vs External Table (外部表)**：

| 维度 | Internal Table | External Table |
|------|---------------|----------------|
| **关键字** | CREATE TABLE | CREATE EXTERNAL TABLE |
| **数据管理** | Hive管理数据生命周期 | Hive仅管理元数据 |
| **DROP行为** | 删除元数据 + 删除HDFS数据 | 仅删除元数据，数据保留 |
| **默认路径** | /user/hive/warehouse/db/table | 可自定义LOCATION |
| **适用场景** | 中间表、临时表 | 原始数据、共享数据 |
| **数据安全** | 低 (误删会丢数据) | 高 (数据独立于Hive) |

```sql
-- 内部表: DROP TABLE时数据会被删除
CREATE TABLE orders_internal (
    order_id    BIGINT,
    user_id     BIGINT,
    amount      DECIMAL(10,2),
    status      STRING
)
PARTITIONED BY (dt STRING)
STORED AS ORC;

-- 外部表: DROP TABLE时仅删除元数据，HDFS数据保留
CREATE EXTERNAL TABLE orders_external (
    order_id    BIGINT,
    user_id     BIGINT,
    amount      DECIMAL(10,2),
    status      STRING
)
PARTITIONED BY (dt STRING)
STORED AS ORC
LOCATION '/data/warehouse/orders';
```

**分区 (Partition)**：将表按某列的值拆分到不同的HDFS子目录，查询时可跳过无关分区。

```
HDFS目录结构 (分区表)
/user/hive/warehouse/orders/
├── dt=2024-01-01/
│   ├── 000000_0        ← ORC数据文件
│   └── 000001_0
├── dt=2024-01-02/
│   ├── 000000_0
│   └── 000001_0
└── dt=2024-01-03/
    └── 000000_0
```

**分桶 (Bucket)**：将分区内数据按某列的Hash值均匀分布到固定数量的文件中，适合采样和优化JOIN。

```sql
-- 创建分桶表: 按user_id哈希分成32个桶
CREATE TABLE orders_bucketed (
    order_id    BIGINT,
    user_id     BIGINT,
    amount      DECIMAL(10,2),
    status      STRING
)
PARTITIONED BY (dt STRING)
CLUSTERED BY (user_id) INTO 32 BUCKETS
STORED AS ORC;
```

---

## 2. HiveQL核心语法

### 2.1 DDL (数据定义语言)

**Hive数据类型速查表**：

| 分类 | 数据类型 | 说明 | 示例 |
|------|---------|------|------|
| **整数** | TINYINT | 1字节有符号整数 | 127 |
| | SMALLINT | 2字节有符号整数 | 32767 |
| | INT | 4字节有符号整数 | 2147483647 |
| | BIGINT | 8字节有符号整数 | 9223372036854775807 |
| **浮点** | FLOAT | 4字节单精度 | 3.14 |
| | DOUBLE | 8字节双精度 | 3.141592653589793 |
| | DECIMAL(p,s) | 精确小数 | DECIMAL(10,2) |
| **字符串** | STRING | 不限长度字符串 | 'hello' |
| | VARCHAR(n) | 变长字符串 | VARCHAR(255) |
| | CHAR(n) | 定长字符串 | CHAR(10) |
| **日期时间** | DATE | 日期 | '2024-01-15' |
| | TIMESTAMP | 时间戳 | '2024-01-15 10:30:00' |
| | INTERVAL | 时间间隔 | INTERVAL '1' DAY |
| **布尔** | BOOLEAN | 布尔值 | TRUE / FALSE |
| **二进制** | BINARY | 二进制数据 | — |
| **复杂类型** | ARRAY\<T\> | 数组 | ARRAY\<STRING\> |
| | MAP\<K,V\> | 键值对 | MAP\<STRING,INT\> |
| | STRUCT\<a:T,...\> | 结构体 | STRUCT\<name:STRING,age:INT\> |
| | UNIONTYPE\<T,...\> | 联合类型 | UNIONTYPE\<INT,STRING\> |

**CREATE TABLE完整语法示例**：

```sql
-- 1. 标准分区分桶表
CREATE TABLE IF NOT EXISTS dw.fact_page_view (
    session_id      STRING      COMMENT '会话ID',
    user_id         BIGINT      COMMENT '用户ID',
    page_url        STRING      COMMENT '页面URL',
    referrer_url    STRING      COMMENT '来源URL',
    duration_sec    INT         COMMENT '停留时长(秒)',
    device_info     STRUCT<
        os:         STRING,
        browser:    STRING,
        resolution: STRING
    >                           COMMENT '设备信息',
    utm_params      MAP<STRING, STRING> COMMENT 'UTM追踪参数',
    event_tags      ARRAY<STRING>       COMMENT '事件标签',
    created_at      TIMESTAMP   COMMENT '事件时间'
)
COMMENT '页面浏览事实表'
PARTITIONED BY (dt STRING COMMENT '日期分区', hour STRING COMMENT '小时分区')
CLUSTERED BY (user_id) SORTED BY (created_at) INTO 64 BUCKETS
ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.orc.OrcSerde'
STORED AS ORC
TBLPROPERTIES (
    'orc.compress'          = 'SNAPPY',
    'orc.bloom.filter.columns' = 'user_id,session_id',
    'transactional'         = 'false'
);

-- 2. 外部表关联JSON数据
CREATE EXTERNAL TABLE ods.raw_app_log (
    event_id    STRING,
    event_type  STRING,
    user_id     BIGINT,
    properties  MAP<STRING, STRING>,
    timestamp   BIGINT
)
PARTITIONED BY (dt STRING)
ROW FORMAT SERDE 'org.apache.hive.hcatalog.data.JsonSerDe'
STORED AS TEXTFILE
LOCATION '/data/raw/app_log'
TBLPROPERTIES ('serialization.null.format' = '');

-- 3. 使用CSV SerDe
CREATE EXTERNAL TABLE ods.csv_orders (
    order_id    STRING,
    product_id  STRING,
    quantity    INT,
    price       DOUBLE
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
    'separatorChar' = ',',
    'quoteChar'     = '\"',
    'escapeChar'    = '\\'
)
STORED AS TEXTFILE
LOCATION '/data/raw/csv_orders';
```

**ALTER TABLE常用操作**：

```sql
-- 添加分区
ALTER TABLE dw.fact_page_view ADD IF NOT EXISTS
    PARTITION (dt='2024-01-15', hour='10')
    PARTITION (dt='2024-01-15', hour='11');

-- 删除分区
ALTER TABLE dw.fact_page_view DROP IF EXISTS PARTITION (dt='2024-01-01');

-- 修改列
ALTER TABLE dw.fact_page_view CHANGE COLUMN duration_sec stay_seconds INT COMMENT '停留秒数';

-- 添加列
ALTER TABLE dw.fact_page_view ADD COLUMNS (
    is_bounce BOOLEAN COMMENT '是否跳出',
    load_time FLOAT   COMMENT '加载时间(秒)'
);

-- 修改表属性
ALTER TABLE dw.fact_page_view SET TBLPROPERTIES ('orc.compress' = 'ZLIB');

-- 修改表类型: 内部表 → 外部表
ALTER TABLE orders_internal SET TBLPROPERTIES ('EXTERNAL' = 'TRUE');
```

**DESCRIBE查看元数据**：

```sql
-- 查看表结构
DESCRIBE dw.fact_page_view;

-- 查看详细信息 (含存储格式、表属性等)
DESCRIBE FORMATTED dw.fact_page_view;

-- 查看分区信息
SHOW PARTITIONS dw.fact_page_view;

-- 查看建表语句
SHOW CREATE TABLE dw.fact_page_view;
```

### 2.2 DML (数据操作语言)

```sql
-- 1. LOAD DATA: 从HDFS或本地加载文件到表
-- 从本地文件系统加载 (复制文件)
LOAD DATA LOCAL INPATH '/opt/data/orders.csv'
INTO TABLE ods.csv_orders PARTITION (dt='2024-01-15');

-- 从HDFS加载 (移动文件)
LOAD DATA INPATH '/staging/orders/2024-01-15/'
INTO TABLE ods.csv_orders PARTITION (dt='2024-01-15');

-- OVERWRITE: 覆盖已有数据
LOAD DATA INPATH '/staging/orders/2024-01-15/'
OVERWRITE INTO TABLE ods.csv_orders PARTITION (dt='2024-01-15');

-- 2. INSERT INTO: 追加数据
INSERT INTO TABLE dw.fact_orders PARTITION (dt='2024-01-15')
SELECT
    order_id,
    user_id,
    product_id,
    amount,
    status,
    created_at
FROM ods.raw_orders
WHERE dt = '2024-01-15';

-- 3. INSERT OVERWRITE: 覆盖写入 (先删后写)
INSERT OVERWRITE TABLE dw.fact_orders PARTITION (dt='2024-01-15')
SELECT
    order_id,
    user_id,
    product_id,
    amount,
    status,
    created_at
FROM ods.raw_orders
WHERE dt = '2024-01-15';

-- 4. 多表插入 (一次扫描源表，写入多个目标)
FROM ods.raw_orders src
INSERT OVERWRITE TABLE dw.fact_orders PARTITION (dt='2024-01-15')
    SELECT order_id, user_id, product_id, amount, status, created_at
    WHERE src.dt = '2024-01-15' AND src.status = 'PAID'
INSERT OVERWRITE TABLE dw.fact_refunds PARTITION (dt='2024-01-15')
    SELECT order_id, user_id, product_id, amount, status, created_at
    WHERE src.dt = '2024-01-15' AND src.status = 'REFUNDED';

-- 5. CTAS: CREATE TABLE AS SELECT
CREATE TABLE dw.user_order_summary
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY')
AS
SELECT
    user_id,
    COUNT(*)            AS order_cnt,
    SUM(amount)         AS total_amount,
    MIN(created_at)     AS first_order_time,
    MAX(created_at)     AS last_order_time
FROM dw.fact_orders
WHERE dt >= '2024-01-01'
GROUP BY user_id;

-- 6. INSERT DIRECTORY: 导出数据到HDFS
INSERT OVERWRITE DIRECTORY '/export/user_order_summary'
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
SELECT * FROM dw.user_order_summary
WHERE order_cnt > 10;

-- 导出到本地文件系统
INSERT OVERWRITE LOCAL DIRECTORY '/tmp/export/orders'
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
SELECT order_id, user_id, amount FROM dw.fact_orders WHERE dt = '2024-01-15';
```

### 2.3 查询语法

**SELECT基础查询**：

```sql
-- 基本查询
SELECT
    o.order_id,
    o.user_id,
    u.username,
    o.amount,
    o.status,
    o.created_at
FROM dw.fact_orders o
JOIN dw.dim_user u ON o.user_id = u.user_id
WHERE o.dt = '2024-01-15'
  AND o.amount > 100
  AND u.is_active = TRUE
ORDER BY o.amount DESC
LIMIT 100;
```

**JOIN类型**：

```sql
-- 1. INNER JOIN: 两表都匹配
SELECT a.*, b.product_name
FROM fact_orders a
JOIN dim_product b ON a.product_id = b.product_id;

-- 2. LEFT OUTER JOIN: 保留左表全部
SELECT a.*, b.product_name
FROM fact_orders a
LEFT JOIN dim_product b ON a.product_id = b.product_id;

-- 3. RIGHT OUTER JOIN: 保留右表全部
SELECT a.*, b.product_name
FROM fact_orders a
RIGHT JOIN dim_product b ON a.product_id = b.product_id;

-- 4. FULL OUTER JOIN: 保留两表全部
SELECT a.*, b.product_name
FROM fact_orders a
FULL OUTER JOIN dim_product b ON a.product_id = b.product_id;

-- 5. LEFT SEMI JOIN: 等价于IN/EXISTS子查询，只返回左表列
SELECT a.*
FROM fact_orders a
LEFT SEMI JOIN dim_user b ON a.user_id = b.user_id;
-- 等价于: SELECT a.* FROM fact_orders a WHERE a.user_id IN (SELECT user_id FROM dim_user)

-- 6. CROSS JOIN: 笛卡尔积 (慎用)
SELECT a.*, b.*
FROM dim_date a
CROSS JOIN dim_channel b;

-- 7. MAP JOIN (小表广播): 小表加载到内存
SELECT /*+ MAPJOIN(b) */ a.*, b.product_name
FROM fact_orders a
JOIN dim_product b ON a.product_id = b.product_id;
```

**ORDER BY / SORT BY / DISTRIBUTE BY / CLUSTER BY 对比**：

| 关键字 | 作用范围 | 排序保证 | Reducer数量 | 性能 | 使用场景 |
|--------|---------|---------|-------------|------|---------|
| **ORDER BY** | 全局排序 | 全局有序 | 只有1个 | 慢 (大数据量) | 最终输出需要全局有序 |
| **SORT BY** | 单个Reducer内排序 | 局部有序 | 多个 | 快 | 每个Reducer输出有序即可 |
| **DISTRIBUTE BY** | 控制数据分发 | 不排序 | 多个 | 快 | 相同key发到同一个Reducer |
| **CLUSTER BY** | DISTRIBUTE BY + SORT BY | 局部有序 | 多个 | 快 | 同字段分发且排序 |

```sql
-- ORDER BY: 全局排序，数据量大时极慢
SELECT user_id, amount FROM fact_orders ORDER BY amount DESC;

-- SORT BY: 每个Reducer内部排序，不保证全局
SET mapreduce.job.reduces = 4;
SELECT user_id, amount FROM fact_orders SORT BY amount DESC;

-- DISTRIBUTE BY: 控制数据分发到同一Reducer
SELECT user_id, amount FROM fact_orders DISTRIBUTE BY user_id;

-- DISTRIBUTE BY + SORT BY: 组合使用
SELECT user_id, amount
FROM fact_orders
DISTRIBUTE BY user_id
SORT BY user_id ASC, amount DESC;

-- CLUSTER BY: 等价于 DISTRIBUTE BY col SORT BY col ASC
SELECT user_id, amount FROM fact_orders CLUSTER BY user_id;
```

**GROUP BY与HAVING**：

```sql
-- 按用户分组统计，过滤累计金额超过1万的高价值用户
SELECT
    user_id,
    COUNT(*)            AS order_cnt,
    SUM(amount)         AS total_amount,
    AVG(amount)         AS avg_amount,
    MAX(amount)         AS max_amount,
    MIN(created_at)     AS first_order_time
FROM dw.fact_orders
WHERE dt BETWEEN '2024-01-01' AND '2024-01-31'
  AND status = 'PAID'
GROUP BY user_id
HAVING SUM(amount) > 10000
ORDER BY total_amount DESC
LIMIT 1000;

-- GROUPING SETS: 多维度聚合
SELECT
    dt,
    channel,
    COUNT(*) AS order_cnt,
    SUM(amount) AS total_amount
FROM dw.fact_orders
WHERE dt BETWEEN '2024-01-01' AND '2024-01-07'
GROUP BY dt, channel
GROUPING SETS (
    (dt, channel),   -- 按日期+渠道
    (dt),            -- 仅按日期
    (channel),       -- 仅按渠道
    ()               -- 全局汇总
);
```

### 2.4 窗口函数

窗口函数是Hive中最强大的分析功能之一，在不减少行数的情况下进行聚合计算。

```sql
-- ROW_NUMBER: 为每一行分配唯一序号 (去重常用)
-- 场景: 取每个用户最近一笔订单
SELECT *
FROM (
    SELECT
        user_id,
        order_id,
        amount,
        created_at,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC) AS rn
    FROM dw.fact_orders
    WHERE dt = '2024-01-15'
) t
WHERE rn = 1;

-- RANK / DENSE_RANK: 排名 (有并列时行为不同)
-- RANK:       1, 2, 2, 4 (跳过)
-- DENSE_RANK: 1, 2, 2, 3 (不跳过)
SELECT
    product_id,
    category_id,
    sales_amount,
    RANK()       OVER (PARTITION BY category_id ORDER BY sales_amount DESC) AS rank_val,
    DENSE_RANK() OVER (PARTITION BY category_id ORDER BY sales_amount DESC) AS dense_rank_val
FROM dw.product_daily_sales
WHERE dt = '2024-01-15';

-- LAG / LEAD: 访问前/后N行数据
-- 场景: 计算订单间隔天数
SELECT
    user_id,
    order_id,
    created_at,
    LAG(created_at, 1)  OVER (PARTITION BY user_id ORDER BY created_at) AS prev_order_time,
    LEAD(created_at, 1) OVER (PARTITION BY user_id ORDER BY created_at) AS next_order_time,
    DATEDIFF(
        created_at,
        LAG(created_at, 1) OVER (PARTITION BY user_id ORDER BY created_at)
    ) AS days_since_last_order
FROM dw.fact_orders
WHERE dt BETWEEN '2024-01-01' AND '2024-01-31';

-- SUM OVER: 累计求和
-- 场景: 计算每日累计GMV
SELECT
    dt,
    daily_gmv,
    SUM(daily_gmv) OVER (ORDER BY dt ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_gmv,
    SUM(daily_gmv) OVER (ORDER BY dt ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7d_gmv
FROM (
    SELECT dt, SUM(amount) AS daily_gmv
    FROM dw.fact_orders
    WHERE dt BETWEEN '2024-01-01' AND '2024-01-31'
      AND status = 'PAID'
    GROUP BY dt
) daily;

-- NTILE: 将数据均匀分成N组
-- 场景: 用户按消费金额分成10个档次
SELECT
    user_id,
    total_amount,
    NTILE(10) OVER (ORDER BY total_amount DESC) AS decile
FROM dw.user_order_summary;

-- FIRST_VALUE / LAST_VALUE
SELECT
    user_id,
    order_id,
    amount,
    FIRST_VALUE(amount) OVER (PARTITION BY user_id ORDER BY created_at) AS first_order_amount,
    LAST_VALUE(amount) OVER (
        PARTITION BY user_id ORDER BY created_at
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS last_order_amount
FROM dw.fact_orders
WHERE dt BETWEEN '2024-01-01' AND '2024-01-31';

-- 综合实战: 同环比分析
SELECT
    dt,
    daily_gmv,
    LAG(daily_gmv, 1)  OVER (ORDER BY dt) AS prev_day_gmv,
    LAG(daily_gmv, 7)  OVER (ORDER BY dt) AS same_day_last_week_gmv,
    ROUND((daily_gmv - LAG(daily_gmv, 1) OVER (ORDER BY dt))
        / LAG(daily_gmv, 1) OVER (ORDER BY dt) * 100, 2) AS day_over_day_pct,
    ROUND((daily_gmv - LAG(daily_gmv, 7) OVER (ORDER BY dt))
        / LAG(daily_gmv, 7) OVER (ORDER BY dt) * 100, 2) AS week_over_week_pct
FROM (
    SELECT dt, SUM(amount) AS daily_gmv
    FROM dw.fact_orders
    WHERE status = 'PAID'
    GROUP BY dt
) daily
ORDER BY dt;
```

---

## 3. 存储格式与压缩

### 3.1 文件格式对比

| 维度 | TextFile | SequenceFile | ORC | Parquet |
|------|----------|-------------|-----|---------|
| **存储方式** | 行式 | 行式 | 列式 | 列式 |
| **可读性** | 人类可读 | 二进制 | 二进制 | 二进制 |
| **可分割** | 是 (无压缩时) | 是 | 是 | 是 |
| **压缩** | 文件级 | Block/Record级 | Stripe级 | RowGroup级 |
| **Schema演化** | 不支持 | 不支持 | 有限支持 | 完整支持 |
| **谓词下推** | 不支持 | 不支持 | 支持 (Min/Max/Bloom) | 支持 (Min/Max) |
| **默认压缩** | 无 | 无 | ZLIB | Snappy |
| **写入速度** | 快 | 中 | 慢 | 中 |
| **读取速度** | 慢 | 中 | 快 (列裁剪) | 快 (列裁剪) |
| **存储空间** | 大 | 中 | 小 | 小 |
| **适用场景** | 原始数据导入 | MapReduce中间结果 | Hive数仓 (推荐) | 跨引擎共享 (Spark/Impala) |
| **Hive默认** | 是 | 否 | 否 | 否 |

```sql
-- 创建不同格式的表
CREATE TABLE t_textfile   (...) STORED AS TEXTFILE;
CREATE TABLE t_sequence   (...) STORED AS SEQUENCEFILE;
CREATE TABLE t_orc        (...) STORED AS ORC;
CREATE TABLE t_parquet    (...) STORED AS PARQUET;

-- ORC表指定压缩
CREATE TABLE t_orc_snappy (...)
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

-- Parquet表指定压缩
CREATE TABLE t_parquet_snappy (...)
STORED AS PARQUET
TBLPROPERTIES ('parquet.compression' = 'SNAPPY');
```

### 3.2 压缩算法

| 维度 | Snappy | GZIP | LZO | ZSTD | Bzip2 |
|------|--------|------|-----|------|-------|
| **压缩比** | 中 (~2-3x) | 高 (~4-5x) | 中 (~2-3x) | 高 (~4-5x) | 最高 (~5-6x) |
| **压缩速度** | 极快 (~250MB/s) | 慢 (~25MB/s) | 快 (~135MB/s) | 快 (~100MB/s) | 极慢 (~10MB/s) |
| **解压速度** | 极快 (~500MB/s) | 中 (~100MB/s) | 快 (~400MB/s) | 快 (~300MB/s) | 慢 (~30MB/s) |
| **CPU消耗** | 低 | 高 | 中 | 中 | 极高 |
| **可分割** | 否 | 否 | 是 (需索引) | 否 | 是 |
| **Hadoop原生** | 是 | 是 | 需安装 | Hadoop 3.0.1+ | 是 |
| **适用场景** | 中间数据、热数据 | 冷数据归档 | 需要分割的MR | 替代GZIP | 归档存储 |

```sql
-- 设置MapReduce中间结果压缩
SET hive.exec.compress.intermediate = true;
SET mapreduce.map.output.compress = true;
SET mapreduce.map.output.compress.codec = org.apache.hadoop.io.compress.SnappyCodec;

-- 设置最终输出压缩
SET hive.exec.compress.output = true;
SET mapreduce.output.fileoutputformat.compress = true;
SET mapreduce.output.fileoutputformat.compress.codec = org.apache.hadoop.io.compress.SnappyCodec;
```

### 3.3 ORC与Parquet深度对比

```
ORC文件内部结构
┌──────────────────────────────────────────────────────┐
│                    ORC File                          │
│  ┌────────────────────────────────────────────────┐  │
│  │ Stripe 1 (默认250MB)                           │  │
│  │  ┌──────────────┐                              │  │
│  │  │ Index Data   │  ← 列的Min/Max/Bloom Filter  │  │
│  │  ├──────────────┤                              │  │
│  │  │ Row Data     │  ← 按列存储的实际数据         │  │
│  │  │  Column 1    │                              │  │
│  │  │  Column 2    │                              │  │
│  │  │  Column ...  │                              │  │
│  │  ├──────────────┤                              │  │
│  │  │ Stripe Footer│  ← 每列的编码、长度信息       │  │
│  │  └──────────────┘                              │  │
│  ├────────────────────────────────────────────────┤  │
│  │ Stripe 2                                       │  │
│  │  ┌──────────────┐                              │  │
│  │  │ Index Data   │                              │  │
│  │  ├──────────────┤                              │  │
│  │  │ Row Data     │                              │  │
│  │  ├──────────────┤                              │  │
│  │  │ Stripe Footer│                              │  │
│  │  └──────────────┘                              │  │
│  ├────────────────────────────────────────────────┤  │
│  │ File Footer                                    │  │
│  │  ┌──────────────────────────────────────────┐  │  │
│  │  │ 每个Stripe的统计信息 (行数、Min/Max)      │  │  │
│  │  │ Schema定义                                │  │  │
│  │  │ 每列的统计信息                            │  │  │
│  │  └──────────────────────────────────────────┘  │  │
│  ├────────────────────────────────────────────────┤  │
│  │ PostScript: 压缩算法、Footer长度              │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘

Parquet文件内部结构
┌──────────────────────────────────────────────────────┐
│                  Parquet File                        │
│  ┌────────────────────────────────────────────────┐  │
│  │ Magic Number: PAR1                             │  │
│  ├────────────────────────────────────────────────┤  │
│  │ Row Group 1 (默认128MB)                        │  │
│  │  ┌─────────────────────────────────────────┐   │  │
│  │  │ Column Chunk 1                          │   │  │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │  │
│  │  │  │ Page 1  │ │ Page 2  │ │ Page 3  │   │   │  │
│  │  │  │(Data)   │ │(Data)   │ │(Dict)   │   │   │  │
│  │  │  └─────────┘ └─────────┘ └─────────┘   │   │  │
│  │  ├─────────────────────────────────────────┤   │  │
│  │  │ Column Chunk 2                          │   │  │
│  │  │  ┌─────────┐ ┌─────────┐               │   │  │
│  │  │  │ Page 1  │ │ Page 2  │               │   │  │
│  │  │  └─────────┘ └─────────┘               │   │  │
│  │  └─────────────────────────────────────────┘   │  │
│  ├────────────────────────────────────────────────┤  │
│  │ Row Group 2                                    │  │
│  │  (同上结构...)                                  │  │
│  ├────────────────────────────────────────────────┤  │
│  │ File Metadata (Thrift编码)                     │  │
│  │  ┌──────────────────────────────────────────┐  │  │
│  │  │ Schema定义                                │  │  │
│  │  │ 每个Row Group的Column Chunk偏移量          │  │  │
│  │  │ 每列的统计信息 (Min/Max/Null Count)       │  │  │
│  │  │ Key-Value Metadata                        │  │  │
│  │  └──────────────────────────────────────────┘  │  │
│  ├────────────────────────────────────────────────┤  │
│  │ Footer Length (4 bytes)                        │  │
│  ├────────────────────────────────────────────────┤  │
│  │ Magic Number: PAR1                             │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

**ORC vs Parquet深度对比**：

| 维度 | ORC | Parquet |
|------|-----|---------|
| **起源** | Facebook (Hortonworks主导) | Twitter + Cloudera |
| **数据块** | Stripe (默认250MB) | Row Group (默认128MB) |
| **索引** | 三级: File / Stripe / Row (1万行) | 两级: File / Row Group |
| **Bloom Filter** | 内建支持 | 1.2.0+ 支持 |
| **嵌套类型** | 扁平化存储 | Dremel编码 (更好) |
| **ACID** | 完整支持 (Hive 3.x) | 不支持 |
| **Schema演化** | 添加列、合并Schema | 添加/删除/重命名列 |
| **生态支持** | Hive最佳 | Spark/Impala/Flink/Arrow |
| **向量化** | Hive原生支持 | 需要额外适配 |
| **典型压缩比** | 约为原始数据的10-15% | 约为原始数据的15-20% |

**性能基准参考** (1TB TPC-DS，Hive on Tez):

| 指标 | ORC + Snappy | ORC + ZLIB | Parquet + Snappy | TextFile |
|------|-------------|------------|-----------------|----------|
| **存储大小** | 105GB | 85GB | 120GB | 1000GB |
| **全表扫描** | 45s | 55s | 50s | 350s |
| **列投影查询** | 12s | 15s | 14s | 320s |
| **谓词下推查询** | 8s | 10s | 11s | 310s |

---

## 4. Hive性能优化

### 4.1 分区裁剪与谓词下推

**静态分区 vs 动态分区**：

```sql
-- ✅ 静态分区: 手动指定分区值
INSERT OVERWRITE TABLE dw.fact_orders PARTITION (dt='2024-01-15')
SELECT order_id, user_id, amount, status, created_at
FROM ods.raw_orders
WHERE dt = '2024-01-15';

-- ✅ 动态分区: 根据查询结果自动创建分区
SET hive.exec.dynamic.partition = true;
SET hive.exec.dynamic.partition.mode = nonstrict;

INSERT OVERWRITE TABLE dw.fact_orders PARTITION (dt)
SELECT order_id, user_id, amount, status, created_at,
       dt  -- 最后一列作为动态分区列
FROM ods.raw_orders
WHERE dt BETWEEN '2024-01-01' AND '2024-01-31';

-- ✅ 混合分区: 静态 + 动态
INSERT OVERWRITE TABLE dw.fact_orders PARTITION (year='2024', month)
SELECT order_id, user_id, amount, status, created_at,
       MONTH(created_at) AS month
FROM ods.raw_orders;
```

**分区裁剪 (Partition Pruning)**：

```sql
-- ✅ 正确: 直接使用分区列过滤，Hive只扫描对应分区
SELECT * FROM dw.fact_orders WHERE dt = '2024-01-15';

-- ✅ 正确: 范围过滤也支持分区裁剪
SELECT * FROM dw.fact_orders WHERE dt BETWEEN '2024-01-01' AND '2024-01-07';

-- ✅ 正确: IN列表支持分区裁剪
SELECT * FROM dw.fact_orders WHERE dt IN ('2024-01-15', '2024-01-16');

-- ❌ 错误: 对分区列使用函数，导致无法裁剪
SELECT * FROM dw.fact_orders WHERE SUBSTR(dt, 1, 7) = '2024-01';

-- ❌ 错误: 隐式类型转换导致全表扫描
SELECT * FROM dw.fact_orders WHERE dt = 20240115;  -- dt是STRING类型

-- ❌ 错误: 关联子查询中的分区列过滤可能失效
SELECT * FROM dw.fact_orders
WHERE dt IN (SELECT dt FROM dim_date WHERE is_holiday = TRUE);
-- 优化方法: 先将子查询结果收集为常量列表
```

**谓词下推 (Predicate Pushdown)**：

```sql
-- 开启谓词下推
SET hive.optimize.ppd = true;              -- 默认true
SET hive.optimize.ppd.storage = true;       -- 存储层下推 (ORC/Parquet)

-- ✅ ORC谓词下推: WHERE条件会被下推到ORC Reader
-- Hive利用ORC文件的 Stripe/Row Index (Min/Max/Count) 跳过不匹配的数据块
SELECT user_id, amount
FROM dw.fact_orders
WHERE amount > 1000 AND user_id = 12345;
-- ORC Reader: 跳过 amount 最大值 < 1000 的Stripe
-- ORC Reader: 跳过 user_id 不包含 12345 的Stripe (若有Bloom Filter)

-- ✅ 建表时添加Bloom Filter加速等值查询
CREATE TABLE dw.fact_orders (...)
STORED AS ORC
TBLPROPERTIES (
    'orc.bloom.filter.columns' = 'user_id,order_id',
    'orc.bloom.filter.fpp'     = '0.05'
);
```

### 4.2 数据倾斜处理

数据倾斜是Hive查询中最常见的性能问题：少数Reducer处理了绝大部分数据。

```
数据倾斜示意
┌──────────────────────────────────────────────────────┐
│  正常分布                   倾斜分布                   │
│  ┌───┐ ┌───┐ ┌───┐        ┌───┐ ┌───┐ ┌──────────┐ │
│  │ R0│ │ R1│ │ R2│        │ R0│ │ R1│ │    R2    │ │
│  │   │ │   │ │   │        │   │ │   │ │          │ │
│  │100│ │110│ │ 90│        │ 50│ │ 30│ │  10000   │ │
│  │ 行│ │ 行│ │ 行│        │ 行│ │ 行│ │   行     │ │
│  └───┘ └───┘ └───┘        └───┘ └───┘ └──────────┘ │
│   ≈均匀                     R2严重倾斜，拖慢整个任务   │
└──────────────────────────────────────────────────────┘
```

**方案1: Skew Join**：

```sql
-- 开启倾斜JOIN优化
SET hive.optimize.skewjoin = true;
SET hive.skewjoin.key = 100000;  -- 超过10万行的key被视为倾斜key

-- Hive会自动将倾斜key和非倾斜key分两个Job处理:
-- Job1: 非倾斜key走普通JOIN
-- Job2: 倾斜key走Map Join (广播小表)
SELECT a.*, b.user_name
FROM fact_orders a
JOIN dim_user b ON a.user_id = b.user_id;
```

**方案2: Map端聚合**：

```sql
-- 开启Map端预聚合 (Combiner)
SET hive.map.aggr = true;                    -- 默认true
SET hive.groupby.mapaggr.checkinterval = 100000;
SET hive.groupby.skewindata = true;          -- 关键参数: 倾斜时自动两阶段聚合

-- 原理: 启用后Hive会自动将GROUP BY拆成两个MR Job
-- Job1: 随机分发到Reducer做部分聚合 (打散热点key)
-- Job2: 按key分发做最终聚合
SELECT user_id, COUNT(*) AS cnt
FROM fact_page_view
GROUP BY user_id;
```

**方案3: 手动两阶段聚合 (加盐打散)**：

```sql
-- 场景: 统计每个城市的订单量，某些热门城市数据极度倾斜
-- ❌ 直接聚合: 热门城市所在Reducer处理大量数据
SELECT city, COUNT(*) AS order_cnt FROM fact_orders GROUP BY city;

-- ✅ 两阶段聚合: 先加随机前缀打散，再去前缀最终聚合
SELECT
    city,
    SUM(partial_cnt) AS order_cnt
FROM (
    -- 第一阶段: 加随机前缀打散数据
    SELECT
        REGEXP_REPLACE(salted_city, '^\\d+_', '') AS city,
        COUNT(*) AS partial_cnt
    FROM (
        SELECT CONCAT(CAST(FLOOR(RAND() * 10) AS STRING), '_', city) AS salted_city
        FROM fact_orders
    ) t1
    GROUP BY salted_city
) t2
GROUP BY city;
```

**方案4: 大表JOIN大表倾斜处理**：

```sql
-- 场景: fact_orders JOIN fact_payment，user_id=-1 (未登录) 大量数据
-- ✅ 将倾斜key和非倾斜key分开处理
SELECT * FROM (
    -- 非倾斜key走正常JOIN
    SELECT a.*, b.pay_time
    FROM fact_orders a
    JOIN fact_payment b ON a.order_id = b.order_id
    WHERE a.user_id <> -1

    UNION ALL

    -- 倾斜key走MAP JOIN
    SELECT /*+ MAPJOIN(b) */ a.*, b.pay_time
    FROM fact_orders a
    JOIN fact_payment b ON a.order_id = b.order_id
    WHERE a.user_id = -1
) combined;
```

### 4.3 小文件合并

大量小文件会导致NameNode内存压力大、Map任务过多。

```sql
-- 1. 输入合并: 将多个小文件合并为一个Map输入
SET hive.input.format = org.apache.hadoop.hive.ql.io.CombineHiveInputFormat;
SET mapreduce.input.fileinputformat.split.maxsize = 268435456;      -- 256MB
SET mapreduce.input.fileinputformat.split.minsize = 67108864;       -- 64MB
SET mapreduce.input.fileinputformat.split.minsize.per.node = 67108864;
SET mapreduce.input.fileinputformat.split.minsize.per.rack = 67108864;

-- 2. 输出合并: Map-Only任务后自动合并小文件
SET hive.merge.mapfiles = true;               -- Map任务输出合并 (默认true)
SET hive.merge.mapredfiles = true;            -- MapReduce任务输出合并 (默认false)
SET hive.merge.tezfiles = true;               -- Tez任务输出合并
SET hive.merge.size.per.task = 268435456;     -- 合并后文件目标大小: 256MB
SET hive.merge.smallfiles.avgsize = 16777216; -- 平均文件小于16MB时触发合并

-- 3. 手动合并已有ORC表的小文件
ALTER TABLE dw.fact_orders PARTITION (dt='2024-01-15') CONCATENATE;

-- 4. 使用INSERT OVERWRITE重写分区 (最彻底)
INSERT OVERWRITE TABLE dw.fact_orders PARTITION (dt='2024-01-15')
SELECT * FROM dw.fact_orders WHERE dt = '2024-01-15';
```

### 4.4 向量化执行与CBO

**向量化执行 (Vectorization)**：一次处理1024行数据而非逐行处理。

```sql
-- 开启向量化执行
SET hive.vectorized.execution.enabled = true;           -- 默认true (Hive 2.x+)
SET hive.vectorized.execution.reduce.enabled = true;    -- Reduce端向量化
SET hive.vectorized.execution.reduce.groupby.enabled = true;
SET hive.vectorized.use.vectorized.input.format = true;
SET hive.vectorized.use.checked.expressions = true;

-- 注意: 向量化执行仅支持ORC/Parquet格式
-- 某些复杂UDF可能不支持向量化，会自动回退到行模式
```

**CBO (Cost-Based Optimizer)**：基于统计信息选择最优执行计划。

```sql
-- 开启CBO
SET hive.cbo.enable = true;                    -- 默认true (Hive 1.1+)
SET hive.compute.query.using.stats = true;
SET hive.stats.fetch.column.stats = true;
SET hive.stats.fetch.partition.stats = true;

-- 收集表级统计信息
ANALYZE TABLE dw.fact_orders COMPUTE STATISTICS;

-- 收集分区级统计信息
ANALYZE TABLE dw.fact_orders PARTITION (dt='2024-01-15') COMPUTE STATISTICS;

-- 收集列级统计信息 (CBO关键)
ANALYZE TABLE dw.fact_orders COMPUTE STATISTICS FOR COLUMNS
    order_id, user_id, amount, status;

-- 查看统计信息
DESCRIBE FORMATTED dw.fact_orders;
DESCRIBE FORMATTED dw.fact_orders PARTITION (dt='2024-01-15');
```

**核心优化参数速查表**：

| 参数 | 默认值 | 推荐值 | 说明 |
|------|-------|-------|------|
| `hive.execution.engine` | mr | tez | 执行引擎 |
| `hive.vectorized.execution.enabled` | true | true | 向量化执行 |
| `hive.cbo.enable` | true | true | CBO优化器 |
| `hive.optimize.ppd` | true | true | 谓词下推 |
| `hive.auto.convert.join` | true | true | 自动Map Join |
| `hive.auto.convert.join.noconditionaltask.size` | 10MB | 512MB | Map Join小表阈值 |
| `hive.exec.parallel` | false | true | 并行执行Stage |
| `hive.exec.parallel.thread.number` | 8 | 16 | 并行线程数 |
| `hive.map.aggr` | true | true | Map端聚合 |
| `hive.fetch.task.conversion` | more | more | 简单查询免MR |
| `hive.tez.auto.reducer.parallelism` | true | true | 自动调整Reducer数 |
| `hive.exec.compress.output` | false | true | 输出压缩 |
| `hive.merge.mapredfiles` | false | true | 输出小文件合并 |

---

## 5. 数仓分层建模实战

### 5.1 分层架构

数据仓库通常采用分层架构，每一层有明确的职责，便于管理、复用和问题排查。

```
数仓分层架构
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  数据源              ODS层           DWD层          DWS层     ADS层  │
│                                                                      │
│  ┌─────────┐     ┌──────────┐    ┌──────────┐  ┌────────┐ ┌──────┐ │
│  │ MySQL   │────→│          │    │          │  │        │ │      │ │
│  │ 业务库   │     │          │    │          │  │        │ │ 用户 │ │
│  └─────────┘     │  原始     │    │  明细    │  │  汇总  │ │ 画像 │ │
│                  │  数据     │───→│  数据    │─→│  数据  │→│      │ │
│  ┌─────────┐     │  层      │    │  层      │  │  层    │ │ GMV  │ │
│  │ 日志     │────→│          │    │          │  │        │ │ 报表 │ │
│  │ Kafka   │     │  (外部表) │    │  (ORC)  │  │ (ORC) │ │      │ │
│  └─────────┘     │          │    │          │  │        │ │ 漏斗 │ │
│                  │          │    │          │  │        │ │      │ │
│  ┌─────────┐     │          │    │          │  │        │ │      │ │
│  │ 第三方   │────→│          │    │          │  │        │ │      │ │
│  │ API     │     │          │    │          │  │        │ │      │ │
│  └─────────┘     └──────────┘    └──────────┘  └────────┘ └──────┘ │
│                                                                      │
│  ← 贴源存储 →    ← 清洗去重 →    ← 轻度聚合 →  ← 面向主题 →         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

| 层次 | 全称 | 职责 | 存储格式 | 数据特点 |
|------|------|------|---------|---------|
| **ODS** | Operational Data Store | 原始数据存储，贴源层 | TextFile/JSON | 未清洗，与源系统一致 |
| **DWD** | Data Warehouse Detail | 明细数据，清洗转换 | ORC + Snappy | 去重、脱敏、标准化 |
| **DWS** | Data Warehouse Summary | 轻度汇总，按主题域聚合 | ORC + Snappy | 宽表、日/周/月聚合 |
| **ADS** | Application Data Store | 应用数据，面向报表/API | ORC / MySQL | 直接对接BI/应用 |

### 5.2 ODS层

ODS层直接对接数据源，使用外部表保证数据安全。

```sql
-- 创建ODS层数据库
CREATE DATABASE IF NOT EXISTS ods COMMENT 'ODS原始数据层';

-- 用户行为日志 (JSON格式, Kafka → HDFS)
CREATE EXTERNAL TABLE ods.user_action_log (
    event_id        STRING      COMMENT '事件ID',
    event_type      STRING      COMMENT '事件类型: page_view/click/scroll',
    user_id         BIGINT      COMMENT '用户ID, 未登录为-1',
    session_id      STRING      COMMENT '会话ID',
    page_url        STRING      COMMENT '当前页面URL',
    referrer_url    STRING      COMMENT '来源URL',
    device_type     STRING      COMMENT '设备类型: mobile/pc/tablet',
    os              STRING      COMMENT '操作系统',
    browser         STRING      COMMENT '浏览器',
    ip              STRING      COMMENT 'IP地址',
    properties      STRING      COMMENT '扩展属性(JSON字符串)',
    event_time      BIGINT      COMMENT '事件时间戳(毫秒)'
)
PARTITIONED BY (dt STRING COMMENT '日期分区')
ROW FORMAT SERDE 'org.apache.hive.hcatalog.data.JsonSerDe'
STORED AS TEXTFILE
LOCATION '/data/ods/user_action_log'
TBLPROPERTIES ('serialization.null.format' = '');

-- 订单数据 (MySQL Binlog → HDFS, 增量同步)
CREATE EXTERNAL TABLE ods.order_info (
    id              BIGINT      COMMENT '自增ID',
    order_id        STRING      COMMENT '订单编号',
    user_id         BIGINT      COMMENT '用户ID',
    product_id      BIGINT      COMMENT '商品ID',
    product_name    STRING      COMMENT '商品名称',
    category_id     INT         COMMENT '品类ID',
    quantity        INT         COMMENT '购买数量',
    unit_price      DECIMAL(10,2) COMMENT '单价',
    total_amount    DECIMAL(10,2) COMMENT '订单总金额',
    discount_amount DECIMAL(10,2) COMMENT '优惠金额',
    pay_amount      DECIMAL(10,2) COMMENT '实付金额',
    order_status    STRING      COMMENT '订单状态',
    pay_type        STRING      COMMENT '支付方式',
    province        STRING      COMMENT '省份',
    city            STRING      COMMENT '城市',
    created_at      STRING      COMMENT '创建时间',
    updated_at      STRING      COMMENT '更新时间'
)
PARTITIONED BY (dt STRING COMMENT '日期分区')
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
LOCATION '/data/ods/order_info';

-- 添加分区 (Airflow/Azkaban调度脚本中执行)
ALTER TABLE ods.user_action_log ADD IF NOT EXISTS PARTITION (dt='${bizdate}');
ALTER TABLE ods.order_info ADD IF NOT EXISTS PARTITION (dt='${bizdate}');
```

### 5.3 DWD层

DWD层负责数据清洗、去重、类型转换、脱敏，并转换为高效的ORC格式。

```sql
CREATE DATABASE IF NOT EXISTS dwd COMMENT 'DWD明细数据层';

-- DWD用户行为明细表
CREATE TABLE dwd.fact_user_action (
    event_id        STRING      COMMENT '事件ID',
    event_type      STRING      COMMENT '事件类型',
    user_id         BIGINT      COMMENT '用户ID',
    session_id      STRING      COMMENT '会话ID',
    page_url        STRING      COMMENT '页面URL',
    page_path       STRING      COMMENT '页面路径 (从URL提取)',
    referrer_url    STRING      COMMENT '来源URL',
    channel         STRING      COMMENT '渠道 (从referrer_url解析)',
    device_type     STRING      COMMENT '设备类型',
    os              STRING      COMMENT '操作系统',
    browser         STRING      COMMENT '浏览器',
    province        STRING      COMMENT '省份 (IP解析)',
    city            STRING      COMMENT '城市 (IP解析)',
    event_time      TIMESTAMP   COMMENT '事件时间'
)
COMMENT 'DWD用户行为明细宽表'
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

-- ETL: ODS → DWD 清洗转换
INSERT OVERWRITE TABLE dwd.fact_user_action PARTITION (dt='${bizdate}')
SELECT
    event_id,
    event_type,
    -- 清洗: user_id 为空或非法值统一为 -1
    CASE WHEN user_id IS NULL OR user_id <= 0 THEN -1 ELSE user_id END AS user_id,
    session_id,
    page_url,
    -- 提取URL路径
    PARSE_URL(page_url, 'PATH') AS page_path,
    referrer_url,
    -- 解析渠道
    CASE
        WHEN referrer_url LIKE '%baidu.com%'   THEN 'baidu'
        WHEN referrer_url LIKE '%google.com%'  THEN 'google'
        WHEN referrer_url LIKE '%weixin%'      THEN 'wechat'
        WHEN referrer_url LIKE '%douyin%'      THEN 'douyin'
        WHEN referrer_url IS NULL OR referrer_url = '' THEN 'direct'
        ELSE 'other'
    END AS channel,
    device_type,
    os,
    browser,
    -- IP解析为省市 (假设有UDF: ip_to_province, ip_to_city)
    ip_to_province(ip) AS province,
    ip_to_city(ip) AS city,
    -- 时间戳转换
    FROM_UNIXTIME(CAST(event_time / 1000 AS BIGINT)) AS event_time
FROM ods.user_action_log
WHERE dt = '${bizdate}'
  -- 数据质量过滤
  AND event_id IS NOT NULL
  AND event_type IN ('page_view', 'click', 'scroll', 'submit')
  AND event_time > 0;

-- DWD订单明细表 (去重)
CREATE TABLE dwd.fact_order_detail (
    order_id        STRING      COMMENT '订单编号',
    user_id         BIGINT      COMMENT '用户ID',
    product_id      BIGINT      COMMENT '商品ID',
    product_name    STRING      COMMENT '商品名称',
    category_id     INT         COMMENT '品类ID',
    quantity        INT         COMMENT '购买数量',
    unit_price      DECIMAL(10,2) COMMENT '单价',
    total_amount    DECIMAL(10,2) COMMENT '订单总金额',
    discount_amount DECIMAL(10,2) COMMENT '优惠金额',
    pay_amount      DECIMAL(10,2) COMMENT '实付金额',
    order_status    STRING      COMMENT '订单状态',
    pay_type        STRING      COMMENT '支付方式',
    province        STRING      COMMENT '省份',
    city            STRING      COMMENT '城市',
    created_at      TIMESTAMP   COMMENT '创建时间',
    updated_at      TIMESTAMP   COMMENT '更新时间'
)
COMMENT 'DWD订单明细表'
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

-- ETL: ODS → DWD (去重: 取同一order_id最新一条)
INSERT OVERWRITE TABLE dwd.fact_order_detail PARTITION (dt='${bizdate}')
SELECT
    order_id, user_id, product_id, product_name, category_id,
    quantity, unit_price, total_amount, discount_amount, pay_amount,
    order_status, pay_type, province, city,
    CAST(created_at AS TIMESTAMP) AS created_at,
    CAST(updated_at AS TIMESTAMP) AS updated_at
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY updated_at DESC) AS rn
    FROM ods.order_info
    WHERE dt = '${bizdate}'
      AND order_id IS NOT NULL
      AND total_amount >= 0
) dedup
WHERE rn = 1;
```

### 5.4 DWS层

DWS层按主题域进行轻度聚合，构建日/周/月汇总宽表。

```sql
CREATE DATABASE IF NOT EXISTS dws COMMENT 'DWS汇总数据层';

-- 每日用户行为汇总宽表
CREATE TABLE dws.user_action_daily (
    user_id             BIGINT      COMMENT '用户ID',
    pv_cnt              BIGINT      COMMENT '浏览次数',
    uv_session_cnt      BIGINT      COMMENT '独立会话数',
    click_cnt           BIGINT      COMMENT '点击次数',
    avg_stay_seconds    DOUBLE      COMMENT '平均停留时长(秒)',
    page_cnt            INT         COMMENT '浏览页面数',
    first_visit_time    TIMESTAMP   COMMENT '当日首次访问时间',
    last_visit_time     TIMESTAMP   COMMENT '当日末次访问时间',
    main_channel        STRING      COMMENT '主要来源渠道',
    main_device         STRING      COMMENT '主要设备类型',
    province            STRING      COMMENT '省份'
)
COMMENT '用户每日行为汇总'
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

INSERT OVERWRITE TABLE dws.user_action_daily PARTITION (dt='${bizdate}')
SELECT
    user_id,
    SUM(CASE WHEN event_type = 'page_view' THEN 1 ELSE 0 END) AS pv_cnt,
    COUNT(DISTINCT session_id) AS uv_session_cnt,
    SUM(CASE WHEN event_type = 'click' THEN 1 ELSE 0 END) AS click_cnt,
    AVG(stay_seconds) AS avg_stay_seconds,
    COUNT(DISTINCT page_path) AS page_cnt,
    MIN(event_time) AS first_visit_time,
    MAX(event_time) AS last_visit_time,
    -- 主要渠道: 取出现次数最多的渠道
    MAX(channel) AS main_channel,
    MAX(device_type) AS main_device,
    MAX(province) AS province
FROM dwd.fact_user_action
WHERE dt = '${bizdate}'
  AND user_id > 0
GROUP BY user_id;

-- 每日交易汇总宽表
CREATE TABLE dws.trade_daily (
    dt                  STRING      COMMENT '日期',
    total_order_cnt     BIGINT      COMMENT '总订单数',
    paid_order_cnt      BIGINT      COMMENT '支付订单数',
    total_gmv           DECIMAL(15,2) COMMENT '总GMV',
    total_pay_amount    DECIMAL(15,2) COMMENT '总实付金额',
    total_discount      DECIMAL(15,2) COMMENT '总优惠金额',
    buyer_cnt           BIGINT      COMMENT '下单人数',
    paid_buyer_cnt      BIGINT      COMMENT '支付人数',
    avg_order_amount    DECIMAL(10,2) COMMENT '客单价',
    refund_cnt          BIGINT      COMMENT '退款订单数',
    refund_amount       DECIMAL(15,2) COMMENT '退款金额'
)
COMMENT '每日交易汇总'
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

INSERT OVERWRITE TABLE dws.trade_daily
SELECT
    dt,
    COUNT(DISTINCT order_id) AS total_order_cnt,
    COUNT(DISTINCT CASE WHEN order_status IN ('PAID','SHIPPED','COMPLETED') THEN order_id END) AS paid_order_cnt,
    SUM(total_amount) AS total_gmv,
    SUM(CASE WHEN order_status IN ('PAID','SHIPPED','COMPLETED') THEN pay_amount ELSE 0 END) AS total_pay_amount,
    SUM(discount_amount) AS total_discount,
    COUNT(DISTINCT user_id) AS buyer_cnt,
    COUNT(DISTINCT CASE WHEN order_status IN ('PAID','SHIPPED','COMPLETED') THEN user_id END) AS paid_buyer_cnt,
    ROUND(
        SUM(CASE WHEN order_status IN ('PAID','SHIPPED','COMPLETED') THEN pay_amount ELSE 0 END)
        / NULLIF(COUNT(DISTINCT CASE WHEN order_status IN ('PAID','SHIPPED','COMPLETED') THEN user_id END), 0),
        2
    ) AS avg_order_amount,
    COUNT(DISTINCT CASE WHEN order_status = 'REFUNDED' THEN order_id END) AS refund_cnt,
    SUM(CASE WHEN order_status = 'REFUNDED' THEN pay_amount ELSE 0 END) AS refund_amount
FROM dwd.fact_order_detail
WHERE dt = '${bizdate}'
GROUP BY dt;
```

### 5.5 ADS层

ADS层面向业务应用，产出直接供Dashboard、API、推荐系统使用的指标表。

```sql
CREATE DATABASE IF NOT EXISTS ads COMMENT 'ADS应用数据层';

-- 用户画像标签表
CREATE TABLE ads.user_profile (
    user_id             BIGINT      COMMENT '用户ID',
    reg_days            INT         COMMENT '注册天数',
    total_order_cnt     BIGINT      COMMENT '累计订单数',
    total_pay_amount    DECIMAL(15,2) COMMENT '累计消费金额',
    recent_30d_order    BIGINT      COMMENT '近30天订单数',
    recent_30d_amount   DECIMAL(15,2) COMMENT '近30天消费金额',
    avg_order_amount    DECIMAL(10,2) COMMENT '平均客单价',
    favorite_category   STRING      COMMENT '最常购买品类',
    rfm_level           STRING      COMMENT 'RFM等级: high/medium/low',
    churn_risk          STRING      COMMENT '流失风险: high/medium/low',
    last_order_date     STRING      COMMENT '最近下单日期',
    last_visit_date     STRING      COMMENT '最近访问日期'
)
COMMENT '用户画像标签表'
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS ORC;

INSERT OVERWRITE TABLE ads.user_profile PARTITION (dt='${bizdate}')
SELECT
    o.user_id,
    DATEDIFF('${bizdate}', u.reg_date) AS reg_days,
    o.total_order_cnt,
    o.total_pay_amount,
    o.recent_30d_order,
    o.recent_30d_amount,
    ROUND(o.total_pay_amount / NULLIF(o.total_order_cnt, 0), 2) AS avg_order_amount,
    o.favorite_category,
    CASE
        WHEN o.recent_30d_amount >= 5000 AND o.recent_30d_order >= 5 THEN 'high'
        WHEN o.recent_30d_amount >= 1000 OR o.recent_30d_order >= 2  THEN 'medium'
        ELSE 'low'
    END AS rfm_level,
    CASE
        WHEN DATEDIFF('${bizdate}', o.last_order_date) > 90 THEN 'high'
        WHEN DATEDIFF('${bizdate}', o.last_order_date) > 30 THEN 'medium'
        ELSE 'low'
    END AS churn_risk,
    o.last_order_date,
    v.last_visit_date
FROM (
    SELECT
        user_id,
        COUNT(*) AS total_order_cnt,
        SUM(pay_amount) AS total_pay_amount,
        SUM(CASE WHEN DATEDIFF('${bizdate}', dt) <= 30 THEN 1 ELSE 0 END) AS recent_30d_order,
        SUM(CASE WHEN DATEDIFF('${bizdate}', dt) <= 30 THEN pay_amount ELSE 0 END) AS recent_30d_amount,
        -- 最常购买品类
        MAX(category_id) AS favorite_category,
        MAX(dt) AS last_order_date
    FROM dwd.fact_order_detail
    WHERE order_status IN ('PAID','SHIPPED','COMPLETED')
    GROUP BY user_id
) o
LEFT JOIN dim.dim_user u ON o.user_id = u.user_id
LEFT JOIN (
    SELECT user_id, MAX(dt) AS last_visit_date
    FROM dws.user_action_daily
    GROUP BY user_id
) v ON o.user_id = v.user_id;

-- 每日核心经营指标表 (对接Dashboard)
CREATE TABLE ads.daily_business_metrics (
    dt                      STRING      COMMENT '日期',
    gmv                     DECIMAL(15,2) COMMENT 'GMV',
    gmv_day_over_day_pct    DECIMAL(8,4) COMMENT 'GMV日环比(%)',
    gmv_week_over_week_pct  DECIMAL(8,4) COMMENT 'GMV周同比(%)',
    paid_order_cnt          BIGINT      COMMENT '支付订单数',
    paid_buyer_cnt          BIGINT      COMMENT '支付人数',
    avg_order_amount        DECIMAL(10,2) COMMENT '客单价',
    new_user_cnt            BIGINT      COMMENT '新增注册用户',
    dau                     BIGINT      COMMENT '日活跃用户数',
    conversion_rate         DECIMAL(8,4) COMMENT '转化率(%)'
)
COMMENT '每日核心经营指标'
STORED AS ORC;
```

---

## 6. 实战案例：电商数仓建设

### 6.1 需求分析与数据流

```
电商数仓全景数据流架构
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                            │
│  │ MySQL     │  │ App/Web   │  │ 第三方    │                            │
│  │ 业务库    │  │ 日志      │  │ 数据      │                            │
│  │           │  │           │  │           │                            │
│  │ ·用户表   │  │ ·点击流   │  │ ·天气     │                            │
│  │ ·商品表   │  │ ·曝光日志 │  │ ·物流     │                            │
│  │ ·订单表   │  │ ·启动日志 │  │ ·支付平台 │                            │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘                            │
│        │              │              │                                   │
│        ↓              ↓              ↓                                   │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐                              │
│  │ Sqoop /  │  │ Flume /   │  │ API /    │                              │
│  │ Canal    │  │ Kafka     │  │ DataX    │                              │
│  │ 增量同步  │  │ 实时采集  │  │ 批量拉取  │                              │
│  └─────┬────┘  └─────┬─────┘  └─────┬────┘                              │
│        │             │              │                                    │
│        ↓             ↓              ↓                                    │
│  ┌────────────────────────────────────────┐                              │
│  │              HDFS 数据湖                │                              │
│  ├────────────────────────────────────────┤                              │
│  │ ODS层: 外部表, TextFile/JSON           │                              │
│  ├────────────────────────────────────────┤                              │
│  │ DWD层: 清洗去重, ORC+Snappy           │                              │
│  ├────────────────────────────────────────┤                              │
│  │ DWS层: 主题汇总, ORC+Snappy           │                              │
│  ├────────────────────────────────────────┤                              │
│  │ ADS层: 指标产出, ORC                   │                              │
│  └───────────────────┬────────────────────┘                              │
│                      │                                                   │
│                      ↓                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ BI报表   │  │ 数据API  │  │ 推荐系统  │  │ 数据大屏  │                │
│  │ Tableau  │  │ Service  │  │ Feature  │  │ Grafana  │                │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘                │
│                                                                          │
│  调度: Airflow / DolphinScheduler        监控: Prometheus + Grafana     │
│  质量: Griffin / Great Expectations       血缘: Atlas / DataHub         │
└──────────────────────────────────────────────────────────────────────────┘
```

### 6.2 维度建模

采用星型模型 (Star Schema)，以事实表为中心，维度表环绕。

```
星型模型 (Star Schema)
                    ┌──────────────────┐
                    │   dim_date       │
                    │ ─────────────    │
                    │ date_key (PK)    │
                    │ full_date        │
                    │ year / quarter   │
                    │ month / week     │
                    │ day_of_week      │
                    │ is_weekend       │
                    │ is_holiday       │
                    └────────┬─────────┘
                             │
┌──────────────────┐         │         ┌──────────────────┐
│   dim_user       │         │         │   dim_product    │
│ ─────────────    │         │         │ ─────────────    │
│ user_id (PK)     │    ┌────┴────┐    │ product_id (PK)  │
│ username         │    │         │    │ product_name     │
│ gender           ├───→│  fact   │←───┤ category_id      │
│ age_group        │    │ _orders │    │ category_name    │
│ reg_date         │    │         │    │ brand            │
│ phone_masked     │    └────┬────┘    │ price            │
│ province / city  │         │         │ status           │
│ vip_level        │         │         └──────────────────┘
└──────────────────┘         │
                             │
                    ┌────────┴─────────┐
                    │ dim_channel      │
                    │ ─────────────    │
                    │ channel_id (PK)  │
                    │ channel_name     │
                    │ channel_type     │
                    │ platform         │
                    └──────────────────┘
```

**维度表DDL**：

```sql
-- 日期维度表 (一次性生成)
CREATE TABLE dim.dim_date (
    date_key        STRING      COMMENT '日期键 yyyy-MM-dd',
    full_date       DATE        COMMENT '完整日期',
    year            INT         COMMENT '年',
    quarter         INT         COMMENT '季度 1-4',
    month           INT         COMMENT '月 1-12',
    week_of_year    INT         COMMENT '年内第几周',
    day_of_month    INT         COMMENT '月内第几天',
    day_of_week     INT         COMMENT '周几 1=Monday',
    day_name        STRING      COMMENT '星期名称',
    is_weekend      BOOLEAN     COMMENT '是否周末',
    is_holiday      BOOLEAN     COMMENT '是否节假日',
    holiday_name    STRING      COMMENT '节假日名称'
)
COMMENT '日期维度表'
STORED AS ORC;

-- 用户维度表 (每日全量快照)
CREATE TABLE dim.dim_user (
    user_id         BIGINT      COMMENT '用户ID',
    username        STRING      COMMENT '用户名',
    gender          STRING      COMMENT '性别: M/F/U',
    birthday        STRING      COMMENT '生日',
    age_group       STRING      COMMENT '年龄段',
    phone_masked    STRING      COMMENT '脱敏手机号 138****1234',
    email_masked    STRING      COMMENT '脱敏邮箱',
    province        STRING      COMMENT '省份',
    city            STRING      COMMENT '城市',
    reg_date        STRING      COMMENT '注册日期',
    reg_channel     STRING      COMMENT '注册渠道',
    vip_level       INT         COMMENT 'VIP等级 0-5'
)
COMMENT '用户维度表'
PARTITIONED BY (dt STRING COMMENT '快照日期')
STORED AS ORC;

-- 商品维度表
CREATE TABLE dim.dim_product (
    product_id      BIGINT      COMMENT '商品ID',
    product_name    STRING      COMMENT '商品名称',
    category_id     INT         COMMENT '品类ID',
    category_name   STRING      COMMENT '品类名称',
    category_l1     STRING      COMMENT '一级品类',
    category_l2     STRING      COMMENT '二级品类',
    category_l3     STRING      COMMENT '三级品类',
    brand_id        INT         COMMENT '品牌ID',
    brand_name      STRING      COMMENT '品牌名称',
    price           DECIMAL(10,2) COMMENT '价格',
    status          STRING      COMMENT '状态: on_sale/off_sale',
    created_at      TIMESTAMP   COMMENT '上架时间'
)
COMMENT '商品维度表'
PARTITIONED BY (dt STRING COMMENT '快照日期')
STORED AS ORC;
```

**事实表DDL**：

```sql
-- 订单事实表
CREATE TABLE dwd.fact_orders (
    order_id        STRING      COMMENT '订单编号',
    user_id         BIGINT      COMMENT '用户ID (FK → dim_user)',
    product_id      BIGINT      COMMENT '商品ID (FK → dim_product)',
    channel_id      STRING      COMMENT '渠道ID (FK → dim_channel)',
    quantity        INT         COMMENT '数量',
    unit_price      DECIMAL(10,2) COMMENT '单价',
    total_amount    DECIMAL(10,2) COMMENT '总金额 (度量)',
    discount_amount DECIMAL(10,2) COMMENT '优惠金额 (度量)',
    pay_amount      DECIMAL(10,2) COMMENT '实付金额 (度量)',
    order_status    STRING      COMMENT '订单状态',
    pay_type        STRING      COMMENT '支付方式',
    created_at      TIMESTAMP   COMMENT '创建时间'
)
COMMENT '订单事实表'
PARTITIONED BY (dt STRING COMMENT '日期分区 (FK → dim_date)')
CLUSTERED BY (user_id) INTO 32 BUCKETS
STORED AS ORC
TBLPROPERTIES (
    'orc.compress' = 'SNAPPY',
    'orc.bloom.filter.columns' = 'user_id,order_id'
);

-- 页面浏览事实表
CREATE TABLE dwd.fact_page_views (
    event_id        STRING      COMMENT '事件ID',
    user_id         BIGINT      COMMENT '用户ID (FK → dim_user)',
    session_id      STRING      COMMENT '会话ID',
    page_path       STRING      COMMENT '页面路径',
    referrer_path   STRING      COMMENT '来源路径',
    channel         STRING      COMMENT '渠道',
    device_type     STRING      COMMENT '设备类型',
    stay_seconds    INT         COMMENT '停留时长',
    event_time      TIMESTAMP   COMMENT '事件时间'
)
COMMENT '页面浏览事实表'
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');
```

### 6.3 指标计算

**GMV (成交总额) 计算**：

```sql
-- 每日GMV及同环比
SELECT
    t.dt,
    t.daily_gmv,
    t.paid_order_cnt,
    t.paid_buyer_cnt,
    ROUND(t.daily_gmv / NULLIF(t.paid_buyer_cnt, 0), 2) AS avg_per_buyer,
    -- 日环比
    LAG(t.daily_gmv, 1) OVER (ORDER BY t.dt) AS prev_day_gmv,
    ROUND(
        (t.daily_gmv - LAG(t.daily_gmv, 1) OVER (ORDER BY t.dt))
        / NULLIF(LAG(t.daily_gmv, 1) OVER (ORDER BY t.dt), 0) * 100, 2
    ) AS day_over_day_pct,
    -- 周同比
    LAG(t.daily_gmv, 7) OVER (ORDER BY t.dt) AS same_day_last_week_gmv,
    ROUND(
        (t.daily_gmv - LAG(t.daily_gmv, 7) OVER (ORDER BY t.dt))
        / NULLIF(LAG(t.daily_gmv, 7) OVER (ORDER BY t.dt), 0) * 100, 2
    ) AS week_over_week_pct,
    -- 累计GMV
    SUM(t.daily_gmv) OVER (ORDER BY t.dt ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_gmv
FROM (
    SELECT
        dt,
        SUM(pay_amount) AS daily_gmv,
        COUNT(DISTINCT order_id) AS paid_order_cnt,
        COUNT(DISTINCT user_id) AS paid_buyer_cnt
    FROM dwd.fact_orders
    WHERE order_status IN ('PAID', 'SHIPPED', 'COMPLETED')
      AND dt BETWEEN '2024-01-01' AND '2024-01-31'
    GROUP BY dt
) t
ORDER BY t.dt;
```

**转化漏斗分析 (浏览 → 加购 → 下单 → 支付)**：

```sql
-- 电商核心转化漏斗
WITH funnel AS (
    -- Step 1: 浏览商品详情页的用户
    SELECT DISTINCT user_id
    FROM dwd.fact_page_views
    WHERE dt = '${bizdate}'
      AND page_path LIKE '/product/detail/%'
),
cart AS (
    -- Step 2: 加入购物车的用户
    SELECT DISTINCT user_id
    FROM dwd.fact_user_action
    WHERE dt = '${bizdate}'
      AND event_type = 'add_to_cart'
),
ordered AS (
    -- Step 3: 提交订单的用户
    SELECT DISTINCT user_id
    FROM dwd.fact_orders
    WHERE dt = '${bizdate}'
),
paid AS (
    -- Step 4: 完成支付的用户
    SELECT DISTINCT user_id
    FROM dwd.fact_orders
    WHERE dt = '${bizdate}'
      AND order_status IN ('PAID', 'SHIPPED', 'COMPLETED')
)
SELECT
    '${bizdate}'                                AS dt,
    f.view_cnt                                  AS step1_view,
    c.cart_cnt                                  AS step2_cart,
    o.order_cnt                                 AS step3_order,
    p.paid_cnt                                  AS step4_paid,
    -- 各步骤转化率
    ROUND(c.cart_cnt  / NULLIF(f.view_cnt, 0) * 100, 2) AS view_to_cart_pct,
    ROUND(o.order_cnt / NULLIF(c.cart_cnt, 0) * 100, 2) AS cart_to_order_pct,
    ROUND(p.paid_cnt  / NULLIF(o.order_cnt, 0) * 100, 2) AS order_to_paid_pct,
    -- 全链路转化率
    ROUND(p.paid_cnt  / NULLIF(f.view_cnt, 0) * 100, 2) AS overall_conversion_pct
FROM
    (SELECT COUNT(*) AS view_cnt FROM funnel) f,
    (SELECT COUNT(*) AS cart_cnt FROM cart) c,
    (SELECT COUNT(*) AS order_cnt FROM ordered) o,
    (SELECT COUNT(*) AS paid_cnt FROM paid) p;
```

**7日留存率计算**：

```sql
-- 新用户7日留存分析
WITH new_users AS (
    -- 当日新注册用户
    SELECT user_id, reg_date
    FROM dim.dim_user
    WHERE dt = '${bizdate}'
      AND reg_date = '${bizdate}'
),
retention AS (
    SELECT
        n.reg_date,
        n.user_id,
        -- 次日留存
        MAX(CASE WHEN v.dt = DATE_ADD(n.reg_date, 1)  THEN 1 ELSE 0 END) AS day1_active,
        -- 3日留存
        MAX(CASE WHEN v.dt = DATE_ADD(n.reg_date, 3)  THEN 1 ELSE 0 END) AS day3_active,
        -- 7日留存
        MAX(CASE WHEN v.dt = DATE_ADD(n.reg_date, 7)  THEN 1 ELSE 0 END) AS day7_active,
        -- 14日留存
        MAX(CASE WHEN v.dt = DATE_ADD(n.reg_date, 14) THEN 1 ELSE 0 END) AS day14_active,
        -- 30日留存
        MAX(CASE WHEN v.dt = DATE_ADD(n.reg_date, 30) THEN 1 ELSE 0 END) AS day30_active
    FROM new_users n
    LEFT JOIN dws.user_action_daily v
        ON n.user_id = v.user_id
        AND v.dt BETWEEN DATE_ADD(n.reg_date, 1) AND DATE_ADD(n.reg_date, 30)
    GROUP BY n.reg_date, n.user_id
)
SELECT
    reg_date,
    COUNT(*)                                    AS new_user_cnt,
    SUM(day1_active)                            AS day1_retained,
    ROUND(SUM(day1_active)  / COUNT(*) * 100, 2) AS day1_retention_pct,
    SUM(day3_active)                            AS day3_retained,
    ROUND(SUM(day3_active)  / COUNT(*) * 100, 2) AS day3_retention_pct,
    SUM(day7_active)                            AS day7_retained,
    ROUND(SUM(day7_active)  / COUNT(*) * 100, 2) AS day7_retention_pct,
    SUM(day14_active)                           AS day14_retained,
    ROUND(SUM(day14_active) / COUNT(*) * 100, 2) AS day14_retention_pct,
    SUM(day30_active)                           AS day30_retained,
    ROUND(SUM(day30_active) / COUNT(*) * 100, 2) AS day30_retention_pct
FROM retention
GROUP BY reg_date;
```

---

## 7. Hive 3.x新特性与总结

### 7.1 ACID事务

Hive 3.x原生支持完整的ACID事务，包括UPDATE、DELETE和MERGE操作。

```sql
-- 创建事务表 (必须是ORC格式 + 分桶)
CREATE TABLE dw.user_profile_transactional (
    user_id         BIGINT,
    username        STRING,
    email           STRING,
    vip_level       INT,
    total_amount    DECIMAL(15,2),
    last_login      TIMESTAMP,
    updated_at      TIMESTAMP
)
CLUSTERED BY (user_id) INTO 16 BUCKETS
STORED AS ORC
TBLPROPERTIES (
    'transactional' = 'true',
    'orc.compress'  = 'SNAPPY'
);

-- UPDATE: 更新满足条件的行
UPDATE dw.user_profile_transactional
SET vip_level = 5,
    updated_at = CURRENT_TIMESTAMP()
WHERE total_amount > 100000;

-- DELETE: 删除满足条件的行
DELETE FROM dw.user_profile_transactional
WHERE last_login < '2023-01-01';

-- MERGE (UPSERT): 合并更新，存在则更新，不存在则插入
MERGE INTO dw.user_profile_transactional AS target
USING (
    SELECT user_id, username, email, vip_level, total_amount, last_login
    FROM staging.user_update_batch
) AS source
ON target.user_id = source.user_id
WHEN MATCHED THEN
    UPDATE SET
        username    = source.username,
        email       = source.email,
        vip_level   = source.vip_level,
        total_amount = source.total_amount,
        last_login  = source.last_login,
        updated_at  = CURRENT_TIMESTAMP()
WHEN NOT MATCHED THEN
    INSERT VALUES (
        source.user_id, source.username, source.email,
        source.vip_level, source.total_amount, source.last_login,
        CURRENT_TIMESTAMP()
    );
```

**ACID相关配置**：

```xml
<property>
  <name>hive.support.concurrency</name>
  <value>true</value>
</property>
<property>
  <name>hive.enforce.bucketing</name>
  <value>true</value>
</property>
<property>
  <name>hive.exec.dynamic.partition.mode</name>
  <value>nonstrict</value>
</property>
<property>
  <name>hive.txn.manager</name>
  <value>org.apache.hadoop.hive.ql.lockmgr.DbTxnManager</value>
</property>
<property>
  <name>hive.compactor.initiator.on</name>
  <value>true</value>
  <description>开启自动Compaction</description>
</property>
<property>
  <name>hive.compactor.worker.threads</name>
  <value>4</value>
  <description>Compaction工作线程数</description>
</property>
```

**Compaction (压缩合并)**：

```
ACID表的Delta文件合并
┌──────────────────────────────────────────────────────────┐
│  写入过程: 每次INSERT/UPDATE/DELETE生成delta文件           │
│                                                          │
│  base_0000001/          ← 初始基础数据                    │
│  delta_0000002_0000002/ ← INSERT事务                     │
│  delta_0000003_0000003/ ← UPDATE事务                     │
│  delete_delta_0000004/  ← DELETE事务                     │
│                                                          │
│  Minor Compaction: 合并delta文件 → 大delta                │
│  base_0000001/                                           │
│  delta_0000002_0000004/ ← 合并后的delta                   │
│                                                          │
│  Major Compaction: 合并base + delta → 新base              │
│  base_0000004/          ← 全新的基础数据                  │
└──────────────────────────────────────────────────────────┘
```

```sql
-- 手动触发Compaction
ALTER TABLE dw.user_profile_transactional COMPACT 'minor';
ALTER TABLE dw.user_profile_transactional COMPACT 'major';

-- 查看Compaction状态
SHOW COMPACTIONS;
```

### 7.2 LLAP实时查询

LLAP (Live Long and Process) 是Hive 2.0+引入的持久化查询服务，在Hive 3.x中成为推荐的交互式查询方案。

```
LLAP架构
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  ┌──────────────┐                                            │
│  │  HiveServer2 │                                            │
│  │  (协调器)     │                                            │
│  └──────┬───────┘                                            │
│         │ 任务分发                                            │
│         ↓                                                    │
│  ┌────────────────────────────────────────────────┐          │
│  │            LLAP Daemon集群                      │          │
│  │                                                 │          │
│  │  ┌──────────────┐  ┌──────────────┐            │          │
│  │  │  LLAP Daemon │  │  LLAP Daemon │   ...      │          │
│  │  │  ┌────────┐  │  │  ┌────────┐  │            │          │
│  │  │  │Executor│  │  │  │Executor│  │            │          │
│  │  │  │ Pool   │  │  │  │ Pool   │  │            │          │
│  │  │  ├────────┤  │  │  ├────────┤  │            │          │
│  │  │  │  I/O   │  │  │  │  I/O   │  │            │          │
│  │  │  │ Thread │  │  │  │ Thread │  │            │          │
│  │  │  ├────────┤  │  │  ├────────┤  │            │          │
│  │  │  │ Cache  │  │  │  │ Cache  │  │            │          │
│  │  │  │(堆外)  │  │  │  │(堆外)  │  │            │          │
│  │  │  └────────┘  │  │  └────────┘  │            │          │
│  │  └──────────────┘  └──────────────┘            │          │
│  └────────────────────────────────────────────────┘          │
│         │                                                    │
│         ↓                                                    │
│  ┌────────────┐                                              │
│  │   HDFS     │  ← 热数据缓存在LLAP，冷数据从HDFS读取        │
│  └────────────┘                                              │
└──────────────────────────────────────────────────────────────┘
```

**LLAP核心特性**：

| 特性 | 说明 |
|------|------|
| **常驻进程** | Daemon长期运行，避免容器启动开销 |
| **列式缓存** | 堆外内存缓存ORC列数据，热数据秒级响应 |
| **向量化** | 原生向量化执行，SIMD指令加速 |
| **多租户** | 支持队列隔离，YARN资源管理 |
| **安全** | 细粒度列级权限控制 |

**LLAP配置**：

```xml
<property>
  <name>hive.llap.execution.mode</name>
  <value>all</value>
  <description>all: 所有查询走LLAP; auto: 优化器决定; none: 不使用</description>
</property>
<property>
  <name>hive.llap.daemon.memory.per.instance.mb</name>
  <value>32768</value>
  <description>每个LLAP Daemon内存: 32GB</description>
</property>
<property>
  <name>hive.llap.io.memory.size</name>
  <value>24576</value>
  <description>I/O缓存大小: 24GB (堆外)</description>
</property>
<property>
  <name>hive.llap.daemon.num.executors</name>
  <value>12</value>
  <description>每个Daemon的执行器数量</description>
</property>
<property>
  <name>hive.llap.daemon.yarn.container.mb</name>
  <value>36864</value>
  <description>YARN容器总内存: 36GB</description>
</property>
```

```bash
# 启动LLAP服务
hive --service llap --name llap_cluster \
  --instances 10 \
  --size 36864m \
  --executors 12 \
  --cache 24576m \
  --xmx 8192m \
  --loglevel INFO \
  --startImmediately

# 查看LLAP状态
hive --service llapstatus --name llap_cluster
```

### 7.3 最佳实践检查清单

以下是Hive数仓项目的生产级最佳实践清单：

**建表规范**：

- ✅ ODS层使用外部表 (EXTERNAL TABLE)，防止误删数据
- ✅ DWD/DWS/ADS层使用内部表，Hive管理生命周期
- ✅ 所有表使用ORC格式 + Snappy压缩
- ✅ 合理设置分区 (按天/小时)，避免分区过多 (>10000)
- ✅ 大表按高频JOIN列分桶，桶数为2的幂次 (16/32/64)
- ✅ 为每列添加COMMENT注释
- ✅ 表名使用 `层_主题_描述` 命名规范: ods_trade_order, dwd_fact_orders
- ❌ 不要使用TextFile作为DWD/DWS/ADS层的存储格式
- ❌ 不要在同一列上既分区又分桶
- ❌ 不要创建无分区的大表 (>1TB)

**查询优化**：

- ✅ 查询必须带分区过滤条件，严禁全表扫描
- ✅ 开启CBO并定期收集列统计信息 (ANALYZE TABLE)
- ✅ 小表JOIN大表时使用Map Join (Hive会自动判断)
- ✅ 使用ORC的Bloom Filter加速等值查询
- ✅ 开启向量化执行 (`hive.vectorized.execution.enabled = true`)
- ✅ 使用Tez引擎替代MapReduce
- ✅ 开启并行执行 (`hive.exec.parallel = true`)
- ❌ 不要对分区列使用函数 (会导致分区裁剪失效)
- ❌ 不要在大数据量上使用ORDER BY (只有1个Reducer)
- ❌ 不要使用SELECT *，只查询需要的列

**数据倾斜**：

- ✅ 使用 `hive.groupby.skewindata = true` 处理GROUP BY倾斜
- ✅ 使用 `hive.optimize.skewjoin = true` 处理JOIN倾斜
- ✅ 大表JOIN大表时对倾斜key加盐打散
- ✅ 定期检查数据分布: `SELECT key, COUNT(*) FROM t GROUP BY key ORDER BY 2 DESC LIMIT 20`
- ❌ 不要忽略NULL值导致的倾斜 (NULL会聚集到同一个Reducer)

**存储与运维**：

- ✅ 开启小文件合并 (`hive.merge.mapredfiles = true`)
- ✅ 定期执行 `ALTER TABLE ... CONCATENATE` 合并ORC小文件
- ✅ ACID表定期执行Major Compaction
- ✅ 设置数据保留策略，定期清理过期分区
- ✅ 使用Airflow/DolphinScheduler管理ETL调度，设置依赖和重试
- ✅ 对关键指标表添加数据质量校验 (行数、空值率、值域)
- ❌ 不要手动修改HDFS上的Hive表数据文件
- ❌ 不要在生产环境使用Embedded Metastore模式

**安全与权限**：

- ✅ 生产环境开启Ranger/Sentry做细粒度权限控制
- ✅ ODS层敏感字段 (手机号、身份证) 在DWD层脱敏处理
- ✅ 使用Kerberos认证保护Metastore和HiveServer2
- ✅ 启用审计日志追踪数据访问行为
- ❌ 不要在表属性中明文存储密码
- ❌ 不要给普通用户授予DROP DATABASE/TABLE权限

---

> **总结**：Hive作为大数据生态中最成熟的数据仓库工具，通过SQL-on-Hadoop的方式大幅降低了大数据分析的门槛。
> 掌握Hive的核心在于三个方面：(1) 深入理解存储格式与分区分桶模型以优化存储；(2) 熟练运用窗口函数与JOIN优化以提升查询性能；(3) 建立规范的ODS→DWD→DWS→ADS分层体系以支撑业务需求。
> 随着Hive 3.x引入ACID事务和LLAP实时查询能力，Hive已从纯离线批处理工具演进为支持准实时分析的现代数据仓库平台。
