# practical examples

::: info 文件信息
- 📄 原文件：`06_practical_examples.sql`
- 🔤 语言：SQL
:::

## SQL 脚本

```sql
-- ============================================================
--                    PostgreSQL 实战示例
-- ============================================================
-- 本文件包含常见业务场景的 SQL 实现。
-- ============================================================

-- \c learn_postgresql

-- ============================================================
--                    1. 电商订单系统
-- ============================================================

-- --- 创建表结构 ---

-- 商品表
CREATE TABLE IF NOT EXISTS shop_products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    category_id INTEGER,
    price NUMERIC(10, 2) NOT NULL,
    stock INTEGER DEFAULT 0 CHECK (stock >= 0),
    attributes JSONB DEFAULT '{}',
    tags TEXT[],
    status VARCHAR(20) DEFAULT 'on_sale'
        CHECK (status IN ('on_sale', 'off_sale', 'deleted')),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 用户表
CREATE TABLE IF NOT EXISTS shop_users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL,
    phone VARCHAR(20),
    balance NUMERIC(10, 2) DEFAULT 0.00 CHECK (balance >= 0),
    vip_level SMALLINT DEFAULT 0,
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 订单表
CREATE TABLE IF NOT EXISTS shop_orders (
    id SERIAL PRIMARY KEY,
    order_no VARCHAR(32) NOT NULL UNIQUE,
    user_id INTEGER NOT NULL REFERENCES shop_users(id),
    total_amount NUMERIC(10, 2) NOT NULL,
    discount_amount NUMERIC(10, 2) DEFAULT 0.00,
    pay_amount NUMERIC(10, 2) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending'
        CHECK (status IN ('pending', 'paid', 'shipped', 'completed', 'cancelled', 'refunded')),
    shipping_address JSONB,
    paid_at TIMESTAMPTZ,
    shipped_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 订单明细表
CREATE TABLE IF NOT EXISTS shop_order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES shop_orders(id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL,
    product_snapshot JSONB NOT NULL,  -- 商品快照
    price NUMERIC(10, 2) NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    amount NUMERIC(10, 2) NOT NULL
);

-- 创建索引
CREATE INDEX idx_products_category ON shop_products(category_id);
CREATE INDEX idx_products_tags ON shop_products USING GIN(tags);
CREATE INDEX idx_products_attrs ON shop_products USING GIN(attributes);
CREATE INDEX idx_orders_user ON shop_orders(user_id);
CREATE INDEX idx_orders_status ON shop_orders(status);
CREATE INDEX idx_orders_created ON shop_orders(created_at);

-- 插入测试数据
INSERT INTO shop_products (name, category_id, price, stock, attributes, tags) VALUES
    ('iPhone 15', 1, 5999.00, 100,
     '{"brand": "Apple", "storage": "128GB", "color": "黑色"}',
     ARRAY['5G', '旗舰', 'iOS']),
    ('MacBook Pro', 2, 14999.00, 50,
     '{"brand": "Apple", "cpu": "M3", "ram": "16GB"}',
     ARRAY['专业', '创意', 'macOS']),
    ('AirPods Pro', 3, 1899.00, 200,
     '{"brand": "Apple", "type": "TWS", "anc": true}',
     ARRAY['降噪', '无线']);

INSERT INTO shop_users (username, email, phone, balance, vip_level) VALUES
    ('user1', 'user1@example.com', '13800001111', 10000.00, 3),
    ('user2', 'user2@example.com', '13800002222', 5000.00, 2),
    ('user3', 'user3@example.com', '13800003333', 2000.00, 1);

-- --- 业务函数 ---

-- 生成订单号函数
CREATE OR REPLACE FUNCTION generate_order_no()
RETURNS VARCHAR AS $$
BEGIN
    RETURN TO_CHAR(NOW(), 'YYYYMMDDHH24MISS') ||
           LPAD(FLOOR(RANDOM() * 1000000)::TEXT, 6, '0');
END;
$$ LANGUAGE plpgsql;

-- 创建订单函数
CREATE OR REPLACE FUNCTION create_order(
    p_user_id INTEGER,
    p_product_id INTEGER,
    p_quantity INTEGER
)
RETURNS TABLE(order_no VARCHAR, message TEXT) AS $$
DECLARE
    v_product RECORD;
    v_order_no VARCHAR;
    v_order_id INTEGER;
    v_total NUMERIC;
BEGIN
    -- 检查库存（加锁）
    SELECT id, name, price, stock, attributes
    INTO v_product
    FROM shop_products
    WHERE id = p_product_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RETURN QUERY SELECT NULL::VARCHAR, '商品不存在'::TEXT;
        RETURN;
    END IF;

    IF v_product.stock < p_quantity THEN
        RETURN QUERY SELECT NULL::VARCHAR, '库存不足'::TEXT;
        RETURN;
    END IF;

    -- 生成订单号
    v_order_no := generate_order_no();
    v_total := v_product.price * p_quantity;

    -- 创建订单
    INSERT INTO shop_orders (order_no, user_id, total_amount, pay_amount)
    VALUES (v_order_no, p_user_id, v_total, v_total)
    RETURNING id INTO v_order_id;

    -- 创建订单明细（包含商品快照）
    INSERT INTO shop_order_items (order_id, product_id, product_snapshot, price, quantity, amount)
    VALUES (
        v_order_id,
        p_product_id,
        jsonb_build_object(
            'name', v_product.name,
            'price', v_product.price,
            'attributes', v_product.attributes
        ),
        v_product.price,
        p_quantity,
        v_total
    );

    -- 扣减库存
    UPDATE shop_products SET stock = stock - p_quantity WHERE id = p_product_id;

    RETURN QUERY SELECT v_order_no, '订单创建成功'::TEXT;
END;
$$ LANGUAGE plpgsql;

-- 调用创建订单
SELECT * FROM create_order(1, 1, 2);

-- --- 高级查询 ---

-- 商品搜索（使用 JSONB 和数组）
SELECT name, price, attributes, tags
FROM shop_products
WHERE attributes @> '{"brand": "Apple"}'
  AND tags && ARRAY['旗舰', '专业'];

-- 订单统计（按日期和状态）
SELECT
    DATE_TRUNC('day', created_at) AS order_date,
    status,
    COUNT(*) AS order_count,
    SUM(pay_amount) AS total_amount
FROM shop_orders
GROUP BY GROUPING SETS (
    (DATE_TRUNC('day', created_at), status),
    (DATE_TRUNC('day', created_at)),
    (status),
    ()
)
ORDER BY order_date, status;

-- 用户消费排行（窗口函数）
SELECT
    username,
    vip_level,
    order_count,
    total_spent,
    RANK() OVER (ORDER BY total_spent DESC) AS rank,
    total_spent * 100.0 / SUM(total_spent) OVER () AS percentage
FROM (
    SELECT
        u.username,
        u.vip_level,
        COUNT(o.id) AS order_count,
        COALESCE(SUM(o.pay_amount), 0) AS total_spent
    FROM shop_users u
    LEFT JOIN shop_orders o ON u.id = o.user_id AND o.status IN ('paid', 'completed')
    GROUP BY u.id, u.username, u.vip_level
) stats
ORDER BY rank;


-- ============================================================
--                    2. 时序数据处理
-- ============================================================

-- 创建时序数据表
CREATE TABLE IF NOT EXISTS metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC NOT NULL,
    tags JSONB DEFAULT '{}',
    recorded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
) PARTITION BY RANGE (recorded_at);

-- 创建分区
CREATE TABLE metrics_2024_01 PARTITION OF metrics
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE metrics_2024_02 PARTITION OF metrics
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- 创建索引
CREATE INDEX idx_metrics_name_time ON metrics(metric_name, recorded_at DESC);
CREATE INDEX idx_metrics_tags ON metrics USING GIN(tags);

-- 插入测试数据
INSERT INTO metrics (metric_name, metric_value, tags, recorded_at)
SELECT
    'cpu_usage',
    RANDOM() * 100,
    '{"host": "server1", "region": "cn-north"}'::JSONB,
    TIMESTAMP '2024-01-01' + (n || ' minutes')::INTERVAL
FROM generate_series(1, 1000) n;

-- 时间聚合查询
SELECT
    DATE_TRUNC('hour', recorded_at) AS hour,
    metric_name,
    AVG(metric_value) AS avg_value,
    MAX(metric_value) AS max_value,
    MIN(metric_value) AS min_value,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY metric_value) AS p95
FROM metrics
WHERE metric_name = 'cpu_usage'
  AND recorded_at >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', recorded_at), metric_name
ORDER BY hour;

-- 移动平均
SELECT
    recorded_at,
    metric_value,
    AVG(metric_value) OVER (
        ORDER BY recorded_at
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) AS moving_avg_10
FROM metrics
WHERE metric_name = 'cpu_usage'
ORDER BY recorded_at
LIMIT 100;


-- ============================================================
--                    3. 全文搜索应用
-- ============================================================

-- 文章表
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    author_id INTEGER,
    tags TEXT[],
    view_count INTEGER DEFAULT 0,
    search_vector TSVECTOR,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 创建触发器自动更新搜索向量
CREATE OR REPLACE FUNCTION update_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('simple', COALESCE(NEW.title, '') || ' ' || COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_search_vector
    BEFORE INSERT OR UPDATE ON articles
    FOR EACH ROW
    EXECUTE FUNCTION update_search_vector();

-- 创建全文搜索索引
CREATE INDEX idx_articles_search ON articles USING GIN(search_vector);
CREATE INDEX idx_articles_tags ON articles USING GIN(tags);

-- 插入测试数据
INSERT INTO articles (title, content, tags) VALUES
    ('PostgreSQL 入门指南', 'PostgreSQL 是一款强大的开源关系数据库...', ARRAY['数据库', 'PostgreSQL', '教程']),
    ('SQL 优化技巧', '本文介绍常用的 SQL 查询优化方法...', ARRAY['SQL', '优化', '性能']),
    ('全文搜索实战', '使用 PostgreSQL 实现全文搜索功能...', ARRAY['搜索', 'PostgreSQL', '实战']);

-- 全文搜索查询
SELECT
    id,
    title,
    ts_rank(search_vector, query) AS rank,
    ts_headline('simple', content, query, 'StartSel=<b>, StopSel=</b>') AS excerpt
FROM articles,
     to_tsquery('simple', 'PostgreSQL & 数据库') AS query
WHERE search_vector @@ query
ORDER BY rank DESC;

-- 组合搜索（全文 + 标签）
SELECT id, title, tags
FROM articles
WHERE search_vector @@ to_tsquery('simple', 'PostgreSQL')
  AND tags && ARRAY['教程', '实战'];


-- ============================================================
--                    4. 地理位置查询
-- ============================================================

-- 启用 PostGIS（需要安装扩展）
-- CREATE EXTENSION IF NOT EXISTS postgis;

-- 简单地理查询（使用内置类型）
CREATE TABLE IF NOT EXISTS stores (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    latitude NUMERIC(10, 7) NOT NULL,
    longitude NUMERIC(10, 7) NOT NULL,
    address TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 插入测试数据
INSERT INTO stores (name, latitude, longitude, address) VALUES
    ('门店A', 39.9042, 116.4074, '北京市东城区'),
    ('门店B', 39.9139, 116.3972, '北京市西城区'),
    ('门店C', 31.2304, 121.4737, '上海市浦东新区');

-- 计算距离（Haversine 公式）
CREATE OR REPLACE FUNCTION haversine_distance(
    lat1 NUMERIC, lon1 NUMERIC,
    lat2 NUMERIC, lon2 NUMERIC
)
RETURNS NUMERIC AS $$
DECLARE
    R NUMERIC := 6371;  -- 地球半径（公里）
    dlat NUMERIC;
    dlon NUMERIC;
    a NUMERIC;
    c NUMERIC;
BEGIN
    dlat := RADIANS(lat2 - lat1);
    dlon := RADIANS(lon2 - lon1);
    a := SIN(dlat/2) * SIN(dlat/2) +
         COS(RADIANS(lat1)) * COS(RADIANS(lat2)) *
         SIN(dlon/2) * SIN(dlon/2);
    c := 2 * ATAN2(SQRT(a), SQRT(1-a));
    RETURN R * c;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- 查找附近门店
SELECT
    name,
    address,
    ROUND(haversine_distance(39.9, 116.4, latitude, longitude)::NUMERIC, 2) AS distance_km
FROM stores
ORDER BY haversine_distance(39.9, 116.4, latitude, longitude)
LIMIT 5;


-- ============================================================
--                    5. 审计日志
-- ============================================================

-- 审计日志表
CREATE TABLE IF NOT EXISTS audit_logs (
    id BIGSERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    record_id INTEGER NOT NULL,
    action VARCHAR(10) NOT NULL CHECK (action IN ('INSERT', 'UPDATE', 'DELETE')),
    old_data JSONB,
    new_data JSONB,
    changed_fields TEXT[],
    changed_by VARCHAR(100),
    changed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_table_record ON audit_logs(table_name, record_id);
CREATE INDEX idx_audit_time ON audit_logs(changed_at);

-- 通用审计触发器函数
CREATE OR REPLACE FUNCTION audit_trigger()
RETURNS TRIGGER AS $$
DECLARE
    v_old_data JSONB;
    v_new_data JSONB;
    v_changed_fields TEXT[];
    v_key TEXT;
BEGIN
    IF TG_OP = 'INSERT' THEN
        v_new_data := to_jsonb(NEW);
        INSERT INTO audit_logs (table_name, record_id, action, new_data, changed_by)
        VALUES (TG_TABLE_NAME, NEW.id, 'INSERT', v_new_data, current_user);

    ELSIF TG_OP = 'UPDATE' THEN
        v_old_data := to_jsonb(OLD);
        v_new_data := to_jsonb(NEW);

        -- 找出变更的字段
        FOR v_key IN SELECT jsonb_object_keys(v_new_data)
        LOOP
            IF v_old_data -> v_key IS DISTINCT FROM v_new_data -> v_key THEN
                v_changed_fields := array_append(v_changed_fields, v_key);
            END IF;
        END LOOP;

        INSERT INTO audit_logs (table_name, record_id, action, old_data, new_data, changed_fields, changed_by)
        VALUES (TG_TABLE_NAME, NEW.id, 'UPDATE', v_old_data, v_new_data, v_changed_fields, current_user);

    ELSIF TG_OP = 'DELETE' THEN
        v_old_data := to_jsonb(OLD);
        INSERT INTO audit_logs (table_name, record_id, action, old_data, changed_by)
        VALUES (TG_TABLE_NAME, OLD.id, 'DELETE', v_old_data, current_user);
    END IF;

    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- 为表添加审计触发器
CREATE TRIGGER audit_shop_products
    AFTER INSERT OR UPDATE OR DELETE ON shop_products
    FOR EACH ROW EXECUTE FUNCTION audit_trigger();

-- 查询审计日志
SELECT
    table_name,
    record_id,
    action,
    changed_fields,
    old_data ->> 'price' AS old_price,
    new_data ->> 'price' AS new_price,
    changed_at
FROM audit_logs
WHERE table_name = 'shop_products'
ORDER BY changed_at DESC;


-- ============================================================
--                    6. 高级统计分析
-- ============================================================

-- 漏斗分析
WITH funnel AS (
    SELECT
        user_id,
        MAX(CASE WHEN action = 'view' THEN 1 ELSE 0 END) AS viewed,
        MAX(CASE WHEN action = 'cart' THEN 1 ELSE 0 END) AS added_cart,
        MAX(CASE WHEN action = 'order' THEN 1 ELSE 0 END) AS ordered,
        MAX(CASE WHEN action = 'pay' THEN 1 ELSE 0 END) AS paid
    FROM (
        -- 模拟用户行为数据
        VALUES
            (1, 'view'), (1, 'cart'), (1, 'order'), (1, 'pay'),
            (2, 'view'), (2, 'cart'), (2, 'order'),
            (3, 'view'), (3, 'cart'),
            (4, 'view')
    ) AS actions(user_id, action)
    GROUP BY user_id
)
SELECT
    '浏览' AS stage,
    SUM(viewed) AS users,
    100.0 AS rate
FROM funnel
UNION ALL
SELECT
    '加购' AS stage,
    SUM(added_cart) AS users,
    ROUND(SUM(added_cart) * 100.0 / NULLIF(SUM(viewed), 0), 2) AS rate
FROM funnel
UNION ALL
SELECT
    '下单' AS stage,
    SUM(ordered) AS users,
    ROUND(SUM(ordered) * 100.0 / NULLIF(SUM(viewed), 0), 2) AS rate
FROM funnel
UNION ALL
SELECT
    '支付' AS stage,
    SUM(paid) AS users,
    ROUND(SUM(paid) * 100.0 / NULLIF(SUM(viewed), 0), 2) AS rate
FROM funnel;

-- 用户分群（RFM 模型）
WITH rfm_data AS (
    SELECT
        user_id,
        MAX(created_at) AS last_order_date,
        COUNT(*) AS frequency,
        SUM(pay_amount) AS monetary
    FROM shop_orders
    WHERE status IN ('paid', 'completed')
    GROUP BY user_id
),
rfm_scores AS (
    SELECT
        user_id,
        NTILE(5) OVER (ORDER BY last_order_date) AS r_score,
        NTILE(5) OVER (ORDER BY frequency) AS f_score,
        NTILE(5) OVER (ORDER BY monetary) AS m_score
    FROM rfm_data
)
SELECT
    user_id,
    r_score,
    f_score,
    m_score,
    CASE
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN '重要价值客户'
        WHEN r_score >= 4 AND f_score < 4 AND m_score >= 4 THEN '重要发展客户'
        WHEN r_score < 4 AND f_score >= 4 AND m_score >= 4 THEN '重要保持客户'
        WHEN r_score < 4 AND f_score < 4 AND m_score >= 4 THEN '重要挽留客户'
        ELSE '一般客户'
    END AS customer_segment
FROM rfm_scores;


-- ============================================================
--                    总结
-- ============================================================

/*
PostgreSQL 实战技巧：

电商系统：
- JSONB 存储商品属性和快照
- 数组存储标签
- 事务保证订单一致性
- 函数封装业务逻辑

时序数据：
- 表分区按时间分割
- 时间聚合函数
- 窗口函数计算移动平均
- 百分位数统计

全文搜索：
- TSVECTOR 和 TSQUERY
- 触发器自动更新搜索向量
- ts_rank 计算相关度
- ts_headline 高亮显示

地理位置：
- PostGIS 扩展（可选）
- Haversine 公式计算距离
- 空间索引

审计日志：
- 通用审计触发器
- JSONB 存储变更数据
- 自动记录变更字段

统计分析：
- 漏斗分析
- RFM 用户分群
- 窗口函数高级应用
- GROUPING SETS 多维聚合
*/

```
