# 数据建模与维度建模

## 目录
- [数据建模概述](#数据建模概述)
- [Kimball维度建模](#kimball维度建模)
- [星型模型](#星型模型)
- [雪花模型](#雪花模型)
- [实战案例](#实战案例)
- [建模最佳实践](#建模最佳实践)

## 数据建模概述

### 数据建模层次

```
┌────────────────────────────────────────────────────────┐
│              数据仓库分层架构                           │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────────────────────────────────┐     │
│  │  ODS (Operational Data Store)                │     │
│  │  原始数据层 - 源系统数据直接同步              │     │
│  └──────────────────┬───────────────────────────┘     │
│                     │                                  │
│                     ▼                                  │
│  ┌──────────────────────────────────────────────┐     │
│  │  DWD (Data Warehouse Detail)                 │     │
│  │  明细数据层 - 清洗、标准化                   │     │
│  └──────────────────┬───────────────────────────┘     │
│                     │                                  │
│                     ▼                                  │
│  ┌──────────────────────────────────────────────┐     │
│  │  DWS (Data Warehouse Summary)                │     │
│  │  汇总数据层 - 轻度聚合                       │     │
│  └──────────────────┬───────────────────────────┘     │
│                     │                                  │
│                     ▼                                  │
│  ┌──────────────────────────────────────────────┐     │
│  │  ADS (Application Data Store)                │     │
│  │  应用数据层 - 面向应用的数据集市              │     │
│  └──────────────────────────────────────────────┘     │
│                                                         │
└────────────────────────────────────────────────────────┘
```

### OLTP vs OLAP建模

| 特性 | OLTP (事务型) | OLAP (分析型) |
|------|--------------|---------------|
| 目标 | 支持日常业务操作 | 支持分析决策 |
| 数据组织 | 高度规范化(3NF) | 反规范化(星型/雪花) |
| 查询类型 | 简单、频繁、小数据量 | 复杂、大数据量聚合 |
| 更新频率 | 高频实时更新 | 批量定期更新 |
| 历史数据 | 保留短期 | 保留长期 |
| 示例 | MySQL订单表 | ClickHouse销售分析表 |

## Kimball维度建模

### 四步维度建模法

```
Kimball维度建模四步法
=====================

步骤1: 选择业务过程
   ├─ 确定要分析的业务活动
   └─ 例如: 销售、库存、客服

步骤2: 声明粒度
   ├─ 确定事实表中一行数据代表什么
   └─ 例如: 每一笔订单、每一次点击

步骤3: 确认维度
   ├─ 确定分析的角度(Who/What/Where/When/Why/How)
   └─ 例如: 时间、地点、产品、客户

步骤4: 确认事实
   ├─ 确定要度量的指标
   └─ 例如: 销售额、数量、利润
```

### 维度表设计原则

```sql
-- ============================================
-- 维度表设计示例
-- ============================================

-- 时间维度表 (dim_date)
CREATE TABLE dim_date (
    date_key INT PRIMARY KEY COMMENT '日期键 YYYYMMDD',
    full_date DATE NOT NULL COMMENT '完整日期',

    -- 年
    year INT NOT NULL COMMENT '年份',
    year_name VARCHAR(10) COMMENT '年份名称 2024年',

    -- 季度
    quarter INT NOT NULL COMMENT '季度 1-4',
    quarter_name VARCHAR(10) COMMENT '季度名称 Q1',
    year_quarter VARCHAR(10) COMMENT '年季度 2024-Q1',

    -- 月
    month INT NOT NULL COMMENT '月份 1-12',
    month_name VARCHAR(10) COMMENT '月份名称 一月',
    year_month VARCHAR(10) COMMENT '年月 2024-01',
    month_days INT COMMENT '当月天数',

    -- 周
    week_of_year INT COMMENT '年中第几周',
    week_of_month INT COMMENT '月中第几周',
    week_name VARCHAR(20) COMMENT '周名称',

    -- 日
    day_of_year INT COMMENT '年中第几天',
    day_of_month INT COMMENT '月中第几天',
    day_of_week INT COMMENT '周中第几天 1-7',
    day_name VARCHAR(10) COMMENT '星期名称',

    -- 标识
    is_weekend TINYINT COMMENT '是否周末',
    is_holiday TINYINT COMMENT '是否节假日',
    holiday_name VARCHAR(50) COMMENT '节假日名称',
    is_workday TINYINT COMMENT '是否工作日',

    -- 财年
    fiscal_year INT COMMENT '财年',
    fiscal_quarter INT COMMENT '财季',
    fiscal_month INT COMMENT '财月',

    -- 其他
    season VARCHAR(10) COMMENT '季节',

    INDEX idx_full_date (full_date),
    INDEX idx_year_month (year, month),
    INDEX idx_year_quarter (year, quarter)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='时间维度表';

-- 生成时间维度数据
DELIMITER $$

CREATE PROCEDURE generate_dim_date(
    IN start_date DATE,
    IN end_date DATE
)
BEGIN
    DECLARE current_date DATE;
    SET current_date = start_date;

    WHILE current_date <= end_date DO
        INSERT INTO dim_date (
            date_key,
            full_date,
            year,
            year_name,
            quarter,
            quarter_name,
            year_quarter,
            month,
            month_name,
            year_month,
            day_of_year,
            day_of_month,
            day_of_week,
            day_name,
            is_weekend,
            is_workday
        ) VALUES (
            DATE_FORMAT(current_date, '%Y%m%d'),
            current_date,
            YEAR(current_date),
            CONCAT(YEAR(current_date), '年'),
            QUARTER(current_date),
            CONCAT('Q', QUARTER(current_date)),
            CONCAT(YEAR(current_date), '-Q', QUARTER(current_date)),
            MONTH(current_date),
            ELT(MONTH(current_date), '一月','二月','三月','四月','五月','六月',
                '七月','八月','九月','十月','十一月','十二月'),
            DATE_FORMAT(current_date, '%Y-%m'),
            DAYOFYEAR(current_date),
            DAY(current_date),
            DAYOFWEEK(current_date),
            ELT(DAYOFWEEK(current_date), '周日','周一','周二','周三','周四','周五','周六'),
            IF(DAYOFWEEK(current_date) IN (1,7), 1, 0),
            IF(DAYOFWEEK(current_date) IN (1,7), 0, 1)
        );

        SET current_date = DATE_ADD(current_date, INTERVAL 1 DAY);
    END WHILE;
END$$

DELIMITER ;

-- 调用存储过程生成2020-2030年数据
CALL generate_dim_date('2020-01-01', '2030-12-31');


-- 产品维度表 (dim_product)
CREATE TABLE dim_product (
    product_key INT AUTO_INCREMENT PRIMARY KEY COMMENT '产品代理键',
    product_id VARCHAR(50) NOT NULL COMMENT '产品业务键',

    -- 产品信息
    product_name VARCHAR(200) NOT NULL COMMENT '产品名称',
    product_code VARCHAR(50) COMMENT '产品编码',
    product_desc TEXT COMMENT '产品描述',

    -- 分类信息（多级分类）
    category_level1 VARCHAR(50) COMMENT '一级分类',
    category_level2 VARCHAR(50) COMMENT '二级分类',
    category_level3 VARCHAR(50) COMMENT '三级分类',

    -- 品牌信息
    brand_name VARCHAR(100) COMMENT '品牌名称',
    brand_country VARCHAR(50) COMMENT '品牌国家',

    -- 属性
    color VARCHAR(50) COMMENT '颜色',
    size VARCHAR(50) COMMENT '尺寸',
    weight DECIMAL(10,2) COMMENT '重量(kg)',
    unit VARCHAR(20) COMMENT '单位',

    -- 价格
    standard_price DECIMAL(10,2) COMMENT '标准价格',
    cost_price DECIMAL(10,2) COMMENT '成本价格',

    -- 状态
    status VARCHAR(20) COMMENT '状态: 在售/停售/缺货',
    is_active TINYINT DEFAULT 1 COMMENT '是否有效',

    -- SCD Type 2 (渐变维度)
    effective_date DATE NOT NULL COMMENT '生效日期',
    expiry_date DATE DEFAULT '9999-12-31' COMMENT '失效日期',
    is_current TINYINT DEFAULT 1 COMMENT '是否当前版本',
    version INT DEFAULT 1 COMMENT '版本号',

    -- 审计字段
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY uk_product_id_version (product_id, version),
    INDEX idx_product_id (product_id),
    INDEX idx_category (category_level1, category_level2, category_level3),
    INDEX idx_brand (brand_name),
    INDEX idx_effective_date (effective_date, expiry_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='产品维度表(SCD Type 2)';


-- 客户维度表 (dim_customer)
CREATE TABLE dim_customer (
    customer_key INT AUTO_INCREMENT PRIMARY KEY COMMENT '客户代理键',
    customer_id VARCHAR(50) NOT NULL COMMENT '客户业务键',

    -- 基本信息
    customer_name VARCHAR(100) NOT NULL COMMENT '客户姓名',
    gender VARCHAR(10) COMMENT '性别',
    birth_date DATE COMMENT '出生日期',
    age_group VARCHAR(20) COMMENT '年龄段: 18-25/26-35/36-45/46+',

    -- 联系方式
    email VARCHAR(100) COMMENT '邮箱',
    phone VARCHAR(20) COMMENT '电话',

    -- 地址信息（多级地域）
    country VARCHAR(50) COMMENT '国家',
    province VARCHAR(50) COMMENT '省份',
    city VARCHAR(50) COMMENT '城市',
    district VARCHAR(50) COMMENT '区县',
    address VARCHAR(200) COMMENT '详细地址',
    postal_code VARCHAR(20) COMMENT '邮编',

    -- 客户分层
    customer_level VARCHAR(20) COMMENT '客户等级: VIP/金卡/银卡/普通',
    customer_segment VARCHAR(50) COMMENT '客户细分',

    -- 注册信息
    registration_date DATE COMMENT '注册日期',
    registration_channel VARCHAR(50) COMMENT '注册渠道',

    -- SCD Type 1 (覆盖) + Type 2 (历史)
    -- Type 1: 联系方式等可以覆盖
    -- Type 2: 客户等级等需要保留历史
    effective_date DATE NOT NULL,
    expiry_date DATE DEFAULT '9999-12-31',
    is_current TINYINT DEFAULT 1,
    version INT DEFAULT 1,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY uk_customer_id_version (customer_id, version),
    INDEX idx_customer_id (customer_id),
    INDEX idx_location (country, province, city),
    INDEX idx_customer_level (customer_level)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客户维度表';


-- 渠道维度表 (dim_channel)
CREATE TABLE dim_channel (
    channel_key INT AUTO_INCREMENT PRIMARY KEY,
    channel_id VARCHAR(50) NOT NULL,

    channel_name VARCHAR(100) NOT NULL COMMENT '渠道名称',
    channel_type VARCHAR(50) COMMENT '渠道类型: 线上/线下',
    channel_category VARCHAR(50) COMMENT '渠道类别',

    -- 线上渠道
    platform VARCHAR(50) COMMENT '平台: 官网/APP/小程序',
    device_type VARCHAR(50) COMMENT '设备类型: PC/Mobile/Tablet',

    -- 线下渠道
    store_id VARCHAR(50) COMMENT '门店ID',
    store_name VARCHAR(100) COMMENT '门店名称',
    store_type VARCHAR(50) COMMENT '门店类型',

    region VARCHAR(50) COMMENT '区域',
    city VARCHAR(50) COMMENT '城市',

    is_active TINYINT DEFAULT 1,

    UNIQUE KEY uk_channel_id (channel_id),
    INDEX idx_channel_type (channel_type),
    INDEX idx_platform (platform)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='渠道维度表';
```

### 事实表设计

```sql
-- ============================================
-- 事实表设计示例
-- ============================================

-- 销售事实表 (fact_sales)
CREATE TABLE fact_sales (
    -- 主键
    sales_key BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '销售流水号',

    -- 维度外键
    date_key INT NOT NULL COMMENT '日期键',
    customer_key INT NOT NULL COMMENT '客户键',
    product_key INT NOT NULL COMMENT '产品键',
    channel_key INT NOT NULL COMMENT '渠道键',

    -- 退化维度（无需单独建维度表）
    order_id VARCHAR(50) NOT NULL COMMENT '订单ID',
    order_item_id VARCHAR(50) NOT NULL COMMENT '订单明细ID',

    -- 度量值（事实）
    -- 可加性度量
    quantity INT NOT NULL COMMENT '销售数量',
    sales_amount DECIMAL(12,2) NOT NULL COMMENT '销售金额',
    discount_amount DECIMAL(12,2) DEFAULT 0 COMMENT '折扣金额',
    cost_amount DECIMAL(12,2) NOT NULL COMMENT '成本金额',
    profit_amount DECIMAL(12,2) NOT NULL COMMENT '利润金额',

    -- 半可加性度量（不能跨时间累加）
    unit_price DECIMAL(10,2) NOT NULL COMMENT '单价',

    -- 非可加性度量（比率，不能累加）
    discount_rate DECIMAL(5,2) COMMENT '折扣率',
    profit_rate DECIMAL(5,2) COMMENT '利润率',

    -- 事实类型标识
    transaction_type VARCHAR(20) COMMENT '交易类型: 销售/退货',
    payment_method VARCHAR(20) COMMENT '支付方式',

    -- 审计字段
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- 索引
    INDEX idx_date (date_key),
    INDEX idx_customer (customer_key),
    INDEX idx_product (product_key),
    INDEX idx_channel (channel_key),
    INDEX idx_order (order_id),
    INDEX idx_date_customer (date_key, customer_key),
    INDEX idx_date_product (date_key, product_key),

    -- 外键约束
    FOREIGN KEY (date_key) REFERENCES dim_date(date_key),
    FOREIGN KEY (customer_key) REFERENCES dim_customer(customer_key),
    FOREIGN KEY (product_key) REFERENCES dim_product(product_key),
    FOREIGN KEY (channel_key) REFERENCES dim_channel(channel_key)

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
PARTITION BY RANGE (date_key) (
    PARTITION p202401 VALUES LESS THAN (20240201),
    PARTITION p202402 VALUES LESS THAN (20240301),
    PARTITION p202403 VALUES LESS THAN (20240401),
    PARTITION p_future VALUES LESS THAN MAXVALUE
)
COMMENT='销售事实表';


-- 库存快照事实表 (fact_inventory_snapshot)
-- 周期快照：每天拍一次快照
CREATE TABLE fact_inventory_snapshot (
    snapshot_key BIGINT AUTO_INCREMENT PRIMARY KEY,

    date_key INT NOT NULL COMMENT '快照日期键',
    product_key INT NOT NULL COMMENT '产品键',
    warehouse_key INT NOT NULL COMMENT '仓库键',

    -- 度量值
    quantity_on_hand INT NOT NULL COMMENT '在库数量',
    quantity_available INT NOT NULL COMMENT '可用数量',
    quantity_reserved INT NOT NULL COMMENT '预留数量',
    quantity_in_transit INT NOT NULL COMMENT '在途数量',

    inventory_value DECIMAL(12,2) COMMENT '库存价值',

    -- 计算字段
    days_of_supply INT COMMENT '可供应天数',
    turnover_rate DECIMAL(5,2) COMMENT '周转率',

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY uk_snapshot (date_key, product_key, warehouse_key),
    INDEX idx_date (date_key),
    INDEX idx_product (product_key),
    INDEX idx_warehouse (warehouse_key)

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='库存快照事实表';


-- 累积快照事实表 (fact_order_accumulating)
-- 跟踪订单生命周期
CREATE TABLE fact_order_accumulating (
    order_key BIGINT AUTO_INCREMENT PRIMARY KEY,

    order_id VARCHAR(50) NOT NULL COMMENT '订单ID',

    -- 多个日期维度（生命周期各阶段）
    order_date_key INT COMMENT '下单日期键',
    payment_date_key INT COMMENT '支付日期键',
    shipment_date_key INT COMMENT '发货日期键',
    delivery_date_key INT COMMENT '收货日期键',

    customer_key INT NOT NULL,
    product_key INT NOT NULL,

    -- 度量值
    quantity INT NOT NULL,
    order_amount DECIMAL(12,2) NOT NULL,

    -- 时间间隔度量
    payment_lag_days INT COMMENT '下单到支付天数',
    shipment_lag_days INT COMMENT '支付到发货天数',
    delivery_lag_days INT COMMENT '发货到收货天数',
    total_cycle_days INT COMMENT '总周期天数',

    -- 状态
    current_status VARCHAR(20) COMMENT '当前状态',

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY uk_order_id (order_id),
    INDEX idx_order_date (order_date_key),
    INDEX idx_customer (customer_key)

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='订单累积快照事实表';
```

## 星型模型

### 星型模型架构

```
           星型模型示例 - 销售分析
┌──────────────────────────────────────────────────┐
│                                                   │
│       ┌─────────────┐                            │
│       │  dim_date   │                            │
│       │  时间维度   │                            │
│       └──────┬──────┘                            │
│              │                                    │
│              │                                    │
│   ┌──────────┼──────────┐                       │
│   │                     │                        │
│   ▼                     ▼                        │
│┌────────┐         ┌──────────┐                  │
││dim_    │         │          │                  │
││customer│◀────────│fact_sales│─────────▶┌──────┐│
││客户维度│         │ 销售事实 │          │dim_  ││
│└────────┘         │          │          │product│
│                   └──────────┘          │产品  ││
│                        │                │维度  ││
│                        │                └──────┘│
│                        ▼                        │
│                   ┌─────────┐                   │
│                   │dim_     │                   │
│                   │channel  │                   │
│                   │渠道维度  │                   │
│                   └─────────┘                   │
│                                                   │
└──────────────────────────────────────────────────┘

特点:
- 中心是事实表
- 维度表直接连接到事实表
- 查询简单，JOIN少
- 数据冗余，但查询性能好
```

### 星型模型查询示例

```sql
-- ============================================
-- 星型模型查询示例
-- ============================================

-- 1. 按时间、产品分析销售
SELECT
    d.year_month,
    p.category_level1,
    p.product_name,
    COUNT(f.sales_key) AS order_count,
    SUM(f.quantity) AS total_quantity,
    SUM(f.sales_amount) AS total_sales,
    SUM(f.profit_amount) AS total_profit,
    AVG(f.unit_price) AS avg_price
FROM fact_sales f
JOIN dim_date d ON f.date_key = d.date_key
JOIN dim_product p ON f.product_key = p.product_key
WHERE d.year = 2024
  AND d.quarter = 1
  AND p.is_current = 1
GROUP BY d.year_month, p.category_level1, p.product_name
ORDER BY total_sales DESC;


-- 2. 客户分群分析
SELECT
    c.customer_level,
    c.age_group,
    c.city,
    COUNT(DISTINCT c.customer_key) AS customer_count,
    SUM(f.sales_amount) AS total_sales,
    AVG(f.sales_amount) AS avg_sales_per_order,
    SUM(f.sales_amount) / COUNT(DISTINCT c.customer_key) AS avg_sales_per_customer
FROM fact_sales f
JOIN dim_customer c ON f.customer_key = c.customer_key
JOIN dim_date d ON f.date_key = d.date_key
WHERE d.year = 2024
  AND c.is_current = 1
GROUP BY c.customer_level, c.age_group, c.city
ORDER BY total_sales DESC;


-- 3. 渠道效能分析
SELECT
    ch.channel_type,
    ch.platform,
    d.year_month,
    COUNT(f.sales_key) AS order_count,
    SUM(f.sales_amount) AS total_sales,
    SUM(f.quantity) AS total_quantity,
    SUM(f.sales_amount) / COUNT(f.sales_key) AS avg_order_value
FROM fact_sales f
JOIN dim_channel ch ON f.channel_key = ch.channel_key
JOIN dim_date d ON f.date_key = d.date_key
WHERE d.year = 2024
  AND ch.is_active = 1
GROUP BY ch.channel_type, ch.platform, d.year_month
ORDER BY d.year_month, total_sales DESC;


-- 4. 同比、环比分析
WITH monthly_sales AS (
    SELECT
        d.year,
        d.month,
        d.year_month,
        SUM(f.sales_amount) AS sales_amount
    FROM fact_sales f
    JOIN dim_date d ON f.date_key = d.date_key
    GROUP BY d.year, d.month, d.year_month
)
SELECT
    curr.year_month,
    curr.sales_amount AS current_sales,
    prev.sales_amount AS prev_month_sales,
    (curr.sales_amount - prev.sales_amount) / prev.sales_amount * 100 AS mom_growth_rate,
    last_year.sales_amount AS last_year_sales,
    (curr.sales_amount - last_year.sales_amount) / last_year.sales_amount * 100 AS yoy_growth_rate
FROM monthly_sales curr
LEFT JOIN monthly_sales prev
    ON curr.year = prev.year
    AND curr.month = prev.month + 1
LEFT JOIN monthly_sales last_year
    ON curr.month = last_year.month
    AND curr.year = last_year.year + 1
ORDER BY curr.year_month;


-- 5. RFM客户分析
SELECT
    customer_key,
    DATEDIFF('2024-12-31', MAX(full_date)) AS recency,
    COUNT(DISTINCT order_id) AS frequency,
    SUM(sales_amount) AS monetary,
    CASE
        WHEN DATEDIFF('2024-12-31', MAX(full_date)) <= 30 THEN '高'
        WHEN DATEDIFF('2024-12-31', MAX(full_date)) <= 90 THEN '中'
        ELSE '低'
    END AS recency_score,
    CASE
        WHEN COUNT(DISTINCT order_id) >= 10 THEN '高'
        WHEN COUNT(DISTINCT order_id) >= 5 THEN '中'
        ELSE '低'
    END AS frequency_score,
    CASE
        WHEN SUM(sales_amount) >= 10000 THEN '高'
        WHEN SUM(sales_amount) >= 5000 THEN '中'
        ELSE '低'
    END AS monetary_score
FROM fact_sales f
JOIN dim_date d ON f.date_key = d.date_key
WHERE d.year = 2024
GROUP BY customer_key;
```

## 雪花模型

### 雪花模型架构

```
           雪花模型示例 - 销售分析
┌──────────────────────────────────────────────────┐
│                                                   │
│       ┌─────────┐                                │
│       │dim_date │                                │
│       └────┬────┘                                │
│            │                                      │
│            │                                      │
│   ┌────────┴────────┐                           │
│   │                 │                            │
│   ▼                 ▼                            │
│┌──────┐       ┌──────────┐       ┌─────────┐   │
││dim_  │       │          │       │dim_     │   │
││customer│◀───│fact_sales│──────▶│product  │   │
│└───┬──┘       │          │       └────┬────┘   │
│    │          └──────────┘            │         │
│    │               │                  │         │
│    ▼               ▼                  ▼         │
│┌──────┐       ┌─────────┐       ┌─────────┐   │
││dim_  │       │dim_     │       │dim_     │   │
││city  │       │channel  │       │category │   │
│└──┬───┘       └────┬────┘       └────┬────┘   │
│   │                │                  │         │
│   ▼                ▼                  ▼         │
│┌──────┐       ┌─────────┐       ┌─────────┐   │
││dim_  │       │dim_     │       │dim_     │   │
││province│     │store    │       │brand    │   │
│└──────┘       └─────────┘       └─────────┘   │
│                                                 │
└─────────────────────────────────────────────────┘

特点:
- 维度表进一步规范化
- 减少数据冗余
- 查询需要更多JOIN
- 维护更新更容易
```

### 雪花模型实现

```sql
-- ============================================
-- 雪花模型维度表（规范化）
-- ============================================

-- 产品维度表（主表）
CREATE TABLE dim_product_snow (
    product_key INT AUTO_INCREMENT PRIMARY KEY,
    product_id VARCHAR(50) NOT NULL,
    product_name VARCHAR(200) NOT NULL,
    product_code VARCHAR(50),

    category_key INT NOT NULL,  -- 指向分类表
    brand_key INT NOT NULL,     -- 指向品牌表

    color VARCHAR(50),
    size VARCHAR(50),
    standard_price DECIMAL(10,2),

    effective_date DATE NOT NULL,
    expiry_date DATE DEFAULT '9999-12-31',
    is_current TINYINT DEFAULT 1,

    INDEX idx_category_key (category_key),
    INDEX idx_brand_key (brand_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 产品分类维度表（独立表）
CREATE TABLE dim_category (
    category_key INT AUTO_INCREMENT PRIMARY KEY,
    category_id VARCHAR(50) NOT NULL,

    category_level1 VARCHAR(50),
    category_level2 VARCHAR(50),
    category_level3 VARCHAR(50),

    category_path VARCHAR(200),  -- 如: 电子产品/手机/智能手机

    UNIQUE KEY uk_category_id (category_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 品牌维度表（独立表）
CREATE TABLE dim_brand (
    brand_key INT AUTO_INCREMENT PRIMARY KEY,
    brand_id VARCHAR(50) NOT NULL,

    brand_name VARCHAR(100) NOT NULL,
    brand_country VARCHAR(50),
    brand_segment VARCHAR(50),  -- 高端/中端/低端

    UNIQUE KEY uk_brand_id (brand_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- 客户维度表（主表）
CREATE TABLE dim_customer_snow (
    customer_key INT AUTO_INCREMENT PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,

    customer_name VARCHAR(100) NOT NULL,
    gender VARCHAR(10),
    birth_date DATE,

    city_key INT NOT NULL,  -- 指向城市表

    customer_level VARCHAR(20),
    registration_date DATE,

    effective_date DATE NOT NULL,
    expiry_date DATE DEFAULT '9999-12-31',
    is_current TINYINT DEFAULT 1,

    INDEX idx_city_key (city_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 城市维度表
CREATE TABLE dim_city (
    city_key INT AUTO_INCREMENT PRIMARY KEY,
    city_id VARCHAR(50) NOT NULL,

    city_name VARCHAR(50) NOT NULL,
    province_key INT NOT NULL,  -- 指向省份表

    city_level VARCHAR(20),  -- 一线/二线/三线

    INDEX idx_province_key (province_key),
    UNIQUE KEY uk_city_id (city_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 省份维度表
CREATE TABLE dim_province (
    province_key INT AUTO_INCREMENT PRIMARY KEY,
    province_id VARCHAR(50) NOT NULL,

    province_name VARCHAR(50) NOT NULL,
    country_key INT NOT NULL,  -- 指向国家表

    region VARCHAR(50),  -- 华东/华南/...

    INDEX idx_country_key (country_key),
    UNIQUE KEY uk_province_id (province_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 国家维度表
CREATE TABLE dim_country (
    country_key INT AUTO_INCREMENT PRIMARY KEY,
    country_id VARCHAR(50) NOT NULL,

    country_name VARCHAR(50) NOT NULL,
    continent VARCHAR(50),

    UNIQUE KEY uk_country_id (country_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- 雪花模型查询（需要多个JOIN）
SELECT
    d.year_month,
    cat.category_level1,
    cat.category_level2,
    br.brand_name,
    p.product_name,
    co.country_name,
    pr.province_name,
    ci.city_name,
    SUM(f.sales_amount) AS total_sales
FROM fact_sales f
JOIN dim_date d ON f.date_key = d.date_key
JOIN dim_product_snow p ON f.product_key = p.product_key
JOIN dim_category cat ON p.category_key = cat.category_key
JOIN dim_brand br ON p.brand_key = br.brand_key
JOIN dim_customer_snow c ON f.customer_key = c.customer_key
JOIN dim_city ci ON c.city_key = ci.city_key
JOIN dim_province pr ON ci.province_key = pr.province_key
JOIN dim_country co ON pr.country_key = co.country_key
WHERE d.year = 2024
GROUP BY
    d.year_month,
    cat.category_level1,
    cat.category_level2,
    br.brand_name,
    p.product_name,
    co.country_name,
    pr.province_name,
    ci.city_name;
```

## 实战案例

### 案例：电商数据仓库建模

```python
# dw_modeling_example.py
"""
电商数据仓库建模示例
演示从ODS到DWD到DWS到ADS的完整流程
"""

from datetime import datetime, timedelta
from typing import List, Dict
import pymysql

class DataWarehouseETL:
    """数据仓库ETL示例"""

    def __init__(self, db_config: Dict):
        self.conn = pymysql.connect(**db_config)
        self.cursor = self.conn.cursor()

    def extract_from_ods(self, date: str):
        """从ODS层提取数据"""
        # ODS层通常是业务系统的镜像
        sql = f"""
        SELECT
            order_id,
            user_id,
            product_id,
            quantity,
            price,
            amount,
            order_time,
            payment_time,
            status
        FROM ods_orders
        WHERE DATE(order_time) = '{date}'
        """
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def transform_to_dwd(self, ods_data: List):
        """转换到DWD层（清洗、标准化）"""
        dwd_data = []

        for row in ods_data:
            # 数据清洗
            order_id, user_id, product_id, quantity, price, amount, order_time, payment_time, status = row

            # 标准化处理
            cleaned_row = {
                'order_id': order_id.strip(),
                'user_id': int(user_id),
                'product_id': int(product_id),
                'quantity': max(0, int(quantity)),  # 确保非负
                'price': round(float(price), 2),
                'amount': round(float(amount), 2),
                'order_time': order_time,
                'payment_time': payment_time if payment_time else None,
                'status': status.strip().upper(),
                # 增加计算字段
                'order_date_key': int(order_time.strftime('%Y%m%d')),
                'payment_date_key': int(payment_time.strftime('%Y%m%d')) if payment_time else None,
                'payment_lag_hours': self.calculate_lag_hours(order_time, payment_time)
            }

            dwd_data.append(cleaned_row)

        return dwd_data

    def calculate_lag_hours(self, start_time, end_time):
        """计算时间差（小时）"""
        if not end_time:
            return None
        return int((end_time - start_time).total_seconds() / 3600)

    def load_to_dwd(self, dwd_data: List):
        """加载到DWD层"""
        sql = """
        INSERT INTO dwd_order_detail (
            order_id, user_id, product_id, quantity, price, amount,
            order_time, payment_time, status,
            order_date_key, payment_date_key, payment_lag_hours,
            etl_date
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        for row in dwd_data:
            self.cursor.execute(sql, (
                row['order_id'], row['user_id'], row['product_id'],
                row['quantity'], row['price'], row['amount'],
                row['order_time'], row['payment_time'], row['status'],
                row['order_date_key'], row['payment_date_key'], row['payment_lag_hours'],
                datetime.now()
            ))

        self.conn.commit()

    def aggregate_to_dws(self, date: str):
        """聚合到DWS层（汇总）"""
        # 用户日汇总
        sql_user_daily = f"""
        INSERT INTO dws_user_daily_summary (
            date_key, user_id,
            order_count, total_amount, avg_amount,
            total_quantity, etl_date
        )
        SELECT
            order_date_key AS date_key,
            user_id,
            COUNT(DISTINCT order_id) AS order_count,
            SUM(amount) AS total_amount,
            AVG(amount) AS avg_amount,
            SUM(quantity) AS total_quantity,
            NOW() AS etl_date
        FROM dwd_order_detail
        WHERE order_date_key = {date.replace('-', '')}
        GROUP BY order_date_key, user_id
        """
        self.cursor.execute(sql_user_daily)

        # 产品日汇总
        sql_product_daily = f"""
        INSERT INTO dws_product_daily_summary (
            date_key, product_id,
            sales_count, total_quantity, total_amount,
            avg_price, etl_date
        )
        SELECT
            order_date_key AS date_key,
            product_id,
            COUNT(*) AS sales_count,
            SUM(quantity) AS total_quantity,
            SUM(amount) AS total_amount,
            AVG(price) AS avg_price,
            NOW() AS etl_date
        FROM dwd_order_detail
        WHERE order_date_key = {date.replace('-', '')}
        GROUP BY order_date_key, product_id
        """
        self.cursor.execute(sql_product_daily)

        self.conn.commit()

    def build_ads_report(self, date: str):
        """构建ADS层报表"""
        # 每日销售概览报表
        sql = f"""
        INSERT INTO ads_daily_sales_report (
            report_date,
            total_orders,
            total_amount,
            total_users,
            avg_order_value,
            new_users,
            repeat_purchase_rate,
            etl_date
        )
        SELECT
            '{date}' AS report_date,
            COUNT(DISTINCT order_id) AS total_orders,
            SUM(amount) AS total_amount,
            COUNT(DISTINCT user_id) AS total_users,
            AVG(amount) AS avg_order_value,
            (SELECT COUNT(*) FROM dws_user_daily_summary
             WHERE date_key = {date.replace('-', '')}
             AND order_count = 1) AS new_users,
            (SELECT COUNT(*) FROM dws_user_daily_summary
             WHERE date_key = {date.replace('-', '')}
             AND order_count > 1) * 1.0 / COUNT(DISTINCT user_id) AS repeat_purchase_rate,
            NOW() AS etl_date
        FROM dwd_order_detail
        WHERE order_date_key = {date.replace('-', '')}
        """
        self.cursor.execute(sql)
        self.conn.commit()

    def run_daily_etl(self, date: str):
        """执行每日ETL流程"""
        print(f"开始处理日期: {date}")

        # 1. Extract从ODS提取
        print("1. 从ODS提取数据...")
        ods_data = self.extract_from_ods(date)
        print(f"   提取 {len(ods_data)} 条记录")

        # 2. Transform转换到DWD
        print("2. 转换数据到DWD...")
        dwd_data = self.transform_to_dwd(ods_data)

        # 3. Load加载到DWD
        print("3. 加载数据到DWD...")
        self.load_to_dwd(dwd_data)

        # 4. 聚合到DWS
        print("4. 聚合数据到DWS...")
        self.aggregate_to_dws(date)

        # 5. 构建ADS报表
        print("5. 构建ADS报表...")
        self.build_ads_report(date)

        print(f"完成处理日期: {date}")

    def close(self):
        """关闭连接"""
        self.cursor.close()
        self.conn.close()

# 使用示例
if __name__ == "__main__":
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'password',
        'database': 'data_warehouse',
        'charset': 'utf8mb4'
    }

    etl = DataWarehouseETL(db_config)

    # 处理昨天的数据
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    etl.run_daily_etl(yesterday)

    etl.close()
```

## 建模最佳实践

### SCD（渐变维度）处理

```sql
-- ============================================
-- SCD Type 2 实现示例
-- ============================================

-- 更新产品价格（保留历史）
DELIMITER $$

CREATE PROCEDURE update_product_price(
    IN p_product_id VARCHAR(50),
    IN p_new_price DECIMAL(10,2)
)
BEGIN
    DECLARE v_product_key INT;
    DECLARE v_current_version INT;
    DECLARE v_today DATE;

    SET v_today = CURDATE();

    -- 获取当前版本
    SELECT product_key, version
    INTO v_product_key, v_current_version
    FROM dim_product
    WHERE product_id = p_product_id
      AND is_current = 1;

    -- 关闭当前版本
    UPDATE dim_product
    SET expiry_date = v_today,
        is_current = 0
    WHERE product_key = v_product_key;

    -- 插入新版本
    INSERT INTO dim_product (
        product_id, product_name, category_level1, category_level2,
        brand_name, standard_price,
        effective_date, version, is_current
    )
    SELECT
        product_id, product_name, category_level1, category_level2,
        brand_name, p_new_price AS standard_price,
        v_today AS effective_date,
        v_current_version + 1 AS version,
        1 AS is_current
    FROM dim_product
    WHERE product_key = v_product_key;
END$$

DELIMITER ;


-- 查询历史价格变化
SELECT
    product_id,
    product_name,
    standard_price,
    effective_date,
    expiry_date,
    version,
    DATEDIFF(expiry_date, effective_date) AS days_effective
FROM dim_product
WHERE product_id = 'P001'
ORDER BY version;
```

### 数据质量检查

```sql
-- ============================================
-- 数据质量检查规则
-- ============================================

-- 1. 空值检查
SELECT
    '空值检查' AS check_type,
    COUNT(*) AS error_count
FROM fact_sales
WHERE customer_key IS NULL
   OR product_key IS NULL
   OR date_key IS NULL
   OR sales_amount IS NULL;

-- 2. 数据一致性检查
SELECT
    '金额一致性检查' AS check_type,
    COUNT(*) AS error_count
FROM fact_sales
WHERE ABS(sales_amount - (quantity * unit_price - discount_amount)) > 0.01;

-- 3. 参照完整性检查
SELECT
    '维度外键检查' AS check_type,
    COUNT(*) AS error_count
FROM fact_sales f
LEFT JOIN dim_customer c ON f.customer_key = c.customer_key
WHERE c.customer_key IS NULL;

-- 4. 重复数据检查
SELECT
    '重复订单检查' AS check_type,
    order_id,
    COUNT(*) AS duplicate_count
FROM fact_sales
GROUP BY order_id, order_item_id
HAVING COUNT(*) > 1;

-- 5. 数据范围检查
SELECT
    '日期范围检查' AS check_type,
    COUNT(*) AS error_count
FROM fact_sales f
JOIN dim_date d ON f.date_key = d.date_key
WHERE d.full_date > CURDATE()
   OR d.full_date < '2020-01-01';

-- 6. 业务逻辑检查
SELECT
    '负数销售检查' AS check_type,
    COUNT(*) AS error_count
FROM fact_sales
WHERE quantity < 0
   OR sales_amount < 0
   OR (transaction_type = '销售' AND profit_amount > sales_amount);
```

### 性能优化

```sql
-- ============================================
-- 数据仓库性能优化
-- ============================================

-- 1. 分区表
ALTER TABLE fact_sales
PARTITION BY RANGE (date_key) (
    PARTITION p202401 VALUES LESS THAN (20240201),
    PARTITION p202402 VALUES LESS THAN (20240301),
    -- ... 更多分区
);

-- 2. 汇总表/物化视图
CREATE TABLE dws_product_monthly_summary AS
SELECT
    CONCAT(d.year, LPAD(d.month, 2, '0')) AS month_key,
    p.product_key,
    p.product_name,
    p.category_level1,
    SUM(f.quantity) AS total_quantity,
    SUM(f.sales_amount) AS total_sales,
    AVG(f.unit_price) AS avg_price,
    COUNT(DISTINCT f.customer_key) AS customer_count
FROM fact_sales f
JOIN dim_date d ON f.date_key = d.date_key
JOIN dim_product p ON f.product_key = p.product_key
WHERE p.is_current = 1
GROUP BY month_key, p.product_key, p.product_name, p.category_level1;

-- 添加索引
CREATE INDEX idx_month_key ON dws_product_monthly_summary(month_key);
CREATE INDEX idx_product_key ON dws_product_monthly_summary(product_key);


-- 3. 列式存储（如ClickHouse）
CREATE TABLE fact_sales_clickhouse (
    sales_key UInt64,
    date_key UInt32,
    customer_key UInt32,
    product_key UInt32,
    quantity UInt32,
    sales_amount Decimal(12, 2),
    profit_amount Decimal(12, 2)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(toDate(toString(date_key), '%Y%m%d'))
ORDER BY (date_key, customer_key, product_key);
```

## 总结

数据建模是数据仓库的核心，Kimball维度建模提供了一套成熟的方法论。

**关键要点**:
1. 遵循维度建模四步法
2. 星型模型优先（性能好）
3. 合理使用SCD处理历史变化
4. 分层架构（ODS/DWD/DWS/ADS）
5. 重视数据质量
6. 性能优化（分区、索引、汇总表）

**进阶阅读**:
- 《数据仓库工具箱》- Ralph Kimball
- 《数据仓库》- Inmon
- ClickHouse文档
- Hive文档
