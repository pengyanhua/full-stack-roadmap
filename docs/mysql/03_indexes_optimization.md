# indexes optimization

::: info 文件信息
- 📄 原文件：`03_indexes_optimization.sql`
- 🔤 语言：SQL
:::

## SQL 脚本

```sql
-- ============================================================
--                    MySQL 索引与优化
-- ============================================================
-- 本文件介绍 MySQL 索引原理和查询优化技巧。
-- ============================================================

USE learn_mysql;

-- ============================================================
--                    1. 索引基础
-- ============================================================

/*
索引类型：
1. B-Tree 索引（默认）：适合范围查询、排序
2. Hash 索引：只支持等值查询（Memory 引擎）
3. 全文索引（FULLTEXT）：文本搜索
4. 空间索引（SPATIAL）：地理数据

按功能分类：
- 主键索引（PRIMARY KEY）：唯一且非空
- 唯一索引（UNIQUE）：值唯一，可有 NULL
- 普通索引（INDEX/KEY）：无约束
- 组合索引（复合索引）：多列组成
- 全文索引（FULLTEXT）：文本搜索
- 前缀索引：只索引字符串前 N 个字符
*/

-- 创建测试表
CREATE TABLE IF NOT EXISTS products (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    sku VARCHAR(50) NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    category_id INT UNSIGNED,
    brand VARCHAR(100),
    price DECIMAL(10, 2) NOT NULL,
    stock INT UNSIGNED DEFAULT 0,
    status ENUM('active', 'inactive', 'deleted') DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 插入测试数据
INSERT INTO products (sku, name, description, category_id, brand, price, stock) VALUES
    ('SKU001', 'iPhone 15', 'Apple iPhone 15 128GB', 1, 'Apple', 5999.00, 100),
    ('SKU002', 'MacBook Pro', 'Apple MacBook Pro 14"', 2, 'Apple', 14999.00, 50),
    ('SKU003', 'Galaxy S24', 'Samsung Galaxy S24 256GB', 1, 'Samsung', 4999.00, 80),
    ('SKU004', 'ThinkPad X1', 'Lenovo ThinkPad X1 Carbon', 2, 'Lenovo', 9999.00, 30),
    ('SKU005', 'AirPods Pro', 'Apple AirPods Pro 2', 3, 'Apple', 1899.00, 200);


-- ============================================================
--                    2. 创建索引
-- ============================================================

-- --- 创建表时定义索引 ---
CREATE TABLE IF NOT EXISTS articles (
    id INT UNSIGNED AUTO_INCREMENT,
    title VARCHAR(200) NOT NULL,
    content TEXT,
    author_id INT UNSIGNED,
    category VARCHAR(50),
    tags VARCHAR(200),
    view_count INT UNSIGNED DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 主键索引
    PRIMARY KEY (id),

    -- 唯一索引
    UNIQUE KEY uk_title (title),

    -- 普通索引
    INDEX idx_author (author_id),

    -- 组合索引
    INDEX idx_category_created (category, created_at),

    -- 全文索引
    FULLTEXT INDEX ft_content (title, content)
) ENGINE=InnoDB;

-- --- 创建表后添加索引 ---

-- 方式1：CREATE INDEX
CREATE INDEX idx_brand ON products(brand);
CREATE UNIQUE INDEX uk_sku ON products(sku);

-- 方式2：ALTER TABLE
ALTER TABLE products ADD INDEX idx_price (price);
ALTER TABLE products ADD INDEX idx_category_price (category_id, price);

-- 前缀索引（只索引前 N 个字符）
ALTER TABLE products ADD INDEX idx_name_prefix (name(50));

-- --- 查看索引 ---
SHOW INDEX FROM products;
SHOW INDEX FROM articles;

-- 查看表的索引信息
SELECT
    INDEX_NAME,
    COLUMN_NAME,
    SEQ_IN_INDEX,
    NON_UNIQUE,
    INDEX_TYPE
FROM INFORMATION_SCHEMA.STATISTICS
WHERE TABLE_SCHEMA = 'learn_mysql' AND TABLE_NAME = 'products';

-- --- 删除索引 ---
DROP INDEX idx_brand ON products;
-- 或
ALTER TABLE products DROP INDEX idx_name_prefix;


-- ============================================================
--                    3. 索引使用原则
-- ============================================================

/*
【最左前缀原则】
组合索引 (a, b, c) 可以支持以下查询：
- WHERE a = ?
- WHERE a = ? AND b = ?
- WHERE a = ? AND b = ? AND c = ?
- WHERE a = ? AND b > ?

不能使用索引：
- WHERE b = ?（缺少最左列）
- WHERE a = ? AND c = ?（跳过了 b）

【索引选择性】
选择性 = 不同值数量 / 总行数
选择性越高，索引效果越好
- 高选择性：身份证号、邮箱
- 低选择性：性别、状态（不适合单独建索引）

【索引建议】
1. 主键自动建立索引
2. 频繁作为查询条件的列
3. 外键列
4. 经常用于 JOIN 的列
5. 经常用于 ORDER BY、GROUP BY 的列
6. 选择性高的列

【不建议索引】
1. 很少查询的列
2. 选择性低的列
3. TEXT、BLOB 等大字段（考虑前缀索引）
4. 频繁更新的列（索引维护成本高）
*/


-- ============================================================
--                    4. EXPLAIN 执行计划
-- ============================================================

/*
EXPLAIN 输出列说明：

id：查询序号
select_type：查询类型
  - SIMPLE：简单查询
  - PRIMARY：最外层查询
  - SUBQUERY：子查询
  - DERIVED：派生表
  - UNION：UNION 中的第二个及之后的查询

table：访问的表

type：访问类型（性能从好到差）
  - system：系统表，只有一行
  - const：常量，通过主键或唯一索引
  - eq_ref：唯一索引扫描，用于 JOIN
  - ref：非唯一索引扫描
  - range：索引范围扫描
  - index：全索引扫描
  - ALL：全表扫描（最差）

possible_keys：可能使用的索引
key：实际使用的索引
key_len：索引使用长度
ref：与索引比较的列或常量
rows：预估扫描行数
filtered：过滤百分比
Extra：额外信息
  - Using index：覆盖索引
  - Using where：使用 WHERE 过滤
  - Using temporary：使用临时表
  - Using filesort：文件排序
*/

-- 基本执行计划
EXPLAIN SELECT * FROM products WHERE id = 1;

-- 查看详细格式
EXPLAIN FORMAT=JSON SELECT * FROM products WHERE brand = 'Apple';

-- 查看实际执行信息（MySQL 8.0+）
EXPLAIN ANALYZE SELECT * FROM products WHERE price > 5000;

-- --- 各种查询的执行计划示例 ---

-- const：主键查询
EXPLAIN SELECT * FROM products WHERE id = 1;

-- ref：普通索引
EXPLAIN SELECT * FROM products WHERE category_id = 1;

-- range：范围查询
EXPLAIN SELECT * FROM products WHERE price BETWEEN 1000 AND 5000;

-- ALL：全表扫描（无合适索引）
EXPLAIN SELECT * FROM products WHERE description LIKE '%Apple%';


-- ============================================================
--                    5. 索引优化技巧
-- ============================================================

-- --- 覆盖索引 ---
-- 查询的列都在索引中，无需回表
CREATE INDEX idx_cover ON products(category_id, price, name);

-- 使用覆盖索引
EXPLAIN SELECT category_id, price, name FROM products WHERE category_id = 1;
-- Extra: Using index

-- --- 索引条件下推（ICP）---
-- MySQL 5.6+ 在存储引擎层过滤数据
EXPLAIN SELECT * FROM products WHERE category_id = 1 AND price > 5000;
-- Extra: Using index condition

-- --- 避免索引失效 ---

-- 1. 函数操作导致失效
EXPLAIN SELECT * FROM products WHERE YEAR(created_at) = 2024;  -- 索引失效
EXPLAIN SELECT * FROM products
WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01';  -- 可用索引

-- 2. 类型转换导致失效
EXPLAIN SELECT * FROM products WHERE sku = 123;  -- 隐式转换，索引可能失效
EXPLAIN SELECT * FROM products WHERE sku = '123';  -- 正确

-- 3. LIKE 前导通配符失效
EXPLAIN SELECT * FROM products WHERE name LIKE '%Phone%';  -- 索引失效
EXPLAIN SELECT * FROM products WHERE name LIKE 'Phone%';   -- 可用索引

-- 4. OR 可能导致失效
EXPLAIN SELECT * FROM products WHERE category_id = 1 OR stock > 100;

-- 5. NOT IN、<> 可能失效
EXPLAIN SELECT * FROM products WHERE category_id NOT IN (1, 2);

-- --- 优化 ORDER BY ---

-- 使用索引排序
CREATE INDEX idx_price_desc ON products(price DESC);
EXPLAIN SELECT * FROM products ORDER BY price DESC LIMIT 10;

-- 组合索引用于 WHERE + ORDER BY
CREATE INDEX idx_cat_price ON products(category_id, price);
EXPLAIN SELECT * FROM products WHERE category_id = 1 ORDER BY price;


-- ============================================================
--                    6. 查询优化
-- ============================================================

-- --- 分页优化 ---

-- 传统分页（深分页性能差）
SELECT * FROM products ORDER BY id LIMIT 100000, 10;

-- 优化方案1：延迟关联
SELECT p.*
FROM products p
JOIN (SELECT id FROM products ORDER BY id LIMIT 100000, 10) tmp
ON p.id = tmp.id;

-- 优化方案2：使用游标（记住上次位置）
SELECT * FROM products WHERE id > 100000 ORDER BY id LIMIT 10;

-- --- COUNT 优化 ---

-- 精确计数（慢）
SELECT COUNT(*) FROM products;

-- 近似计数（快）
SELECT TABLE_ROWS
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = 'learn_mysql' AND TABLE_NAME = 'products';

-- --- JOIN 优化 ---

-- 确保 JOIN 列有索引
EXPLAIN SELECT e.name, d.name
FROM employees e
JOIN departments d ON e.department_id = d.id;

-- 小表驱动大表
-- MySQL 优化器会自动选择，但可以用 STRAIGHT_JOIN 强制
SELECT STRAIGHT_JOIN e.name, d.name
FROM departments d
JOIN employees e ON e.department_id = d.id;

-- --- 子查询优化 ---

-- 改写为 JOIN（通常更快）
-- 子查询写法
SELECT * FROM employees WHERE department_id IN (
    SELECT id FROM departments WHERE location = '北京'
);

-- JOIN 写法
SELECT e.* FROM employees e
JOIN departments d ON e.department_id = d.id
WHERE d.location = '北京';


-- ============================================================
--                    7. 性能分析工具
-- ============================================================

-- --- 慢查询日志 ---
-- 查看慢查询配置
SHOW VARIABLES LIKE 'slow_query%';
SHOW VARIABLES LIKE 'long_query_time';

-- 开启慢查询日志（临时）
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 1;  -- 超过1秒记录

-- --- 性能模式 ---
-- MySQL 5.6+ Performance Schema

-- 查看当前执行的查询
SELECT * FROM performance_schema.events_statements_current
WHERE SQL_TEXT IS NOT NULL;

-- 查看最耗时的查询
SELECT
    DIGEST_TEXT,
    COUNT_STAR AS exec_count,
    SUM_TIMER_WAIT/1000000000000 AS total_time_sec,
    AVG_TIMER_WAIT/1000000000000 AS avg_time_sec
FROM performance_schema.events_statements_summary_by_digest
ORDER BY SUM_TIMER_WAIT DESC
LIMIT 10;

-- --- SHOW PROFILE ---
-- 查看查询各阶段耗时
SET profiling = 1;
SELECT * FROM products WHERE category_id = 1;
SHOW PROFILES;
SHOW PROFILE FOR QUERY 1;
SHOW PROFILE CPU, BLOCK IO FOR QUERY 1;


-- ============================================================
--                    8. 锁与并发
-- ============================================================

/*
InnoDB 锁类型：
- 共享锁（S锁）：读锁，多个事务可同时持有
- 排他锁（X锁）：写锁，独占访问
- 意向锁：表级锁，表示事务想要获取行锁
- 间隙锁：锁定索引间隙，防止幻读
- 临键锁：记录锁 + 间隙锁
*/

-- 查看当前锁
SELECT * FROM performance_schema.data_locks;

-- 查看锁等待
SELECT * FROM performance_schema.data_lock_waits;

-- 查看 InnoDB 状态（包含锁信息）
SHOW ENGINE INNODB STATUS;

-- --- 锁定读 ---

-- 共享锁
SELECT * FROM products WHERE id = 1 LOCK IN SHARE MODE;
-- MySQL 8.0+
SELECT * FROM products WHERE id = 1 FOR SHARE;

-- 排他锁
SELECT * FROM products WHERE id = 1 FOR UPDATE;

-- 跳过被锁的行（MySQL 8.0+）
SELECT * FROM products WHERE category_id = 1 FOR UPDATE SKIP LOCKED;

-- 不等待锁（MySQL 8.0+）
SELECT * FROM products WHERE id = 1 FOR UPDATE NOWAIT;


-- ============================================================
--                    9. 分区表
-- ============================================================

/*
分区类型：
- RANGE：按范围分区
- LIST：按列表分区
- HASH：按哈希分区
- KEY：按键分区
*/

-- 创建范围分区表
CREATE TABLE IF NOT EXISTS orders_partitioned (
    id INT UNSIGNED AUTO_INCREMENT,
    user_id INT UNSIGNED,
    total_amount DECIMAL(10, 2),
    order_date DATE NOT NULL,
    status VARCHAR(20),
    PRIMARY KEY (id, order_date)
) ENGINE=InnoDB
PARTITION BY RANGE (YEAR(order_date)) (
    PARTITION p2022 VALUES LESS THAN (2023),
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);

-- 查看分区信息
SELECT
    PARTITION_NAME,
    TABLE_ROWS,
    AVG_ROW_LENGTH,
    DATA_LENGTH
FROM INFORMATION_SCHEMA.PARTITIONS
WHERE TABLE_SCHEMA = 'learn_mysql'
AND TABLE_NAME = 'orders_partitioned';

-- 添加分区
ALTER TABLE orders_partitioned ADD PARTITION (
    PARTITION p2025 VALUES LESS THAN (2026)
);

-- 删除分区
-- ALTER TABLE orders_partitioned DROP PARTITION p2022;

-- 重组分区
-- ALTER TABLE orders_partitioned REORGANIZE PARTITION p_future INTO (
--     PARTITION p2025 VALUES LESS THAN (2026),
--     PARTITION p_future VALUES LESS THAN MAXVALUE
-- );


-- ============================================================
--                    总结
-- ============================================================

/*
索引类型：
- B-Tree（默认）、Hash、全文、空间
- 主键、唯一、普通、组合、前缀

索引使用原则：
- 最左前缀原则
- 高选择性列优先
- 覆盖索引减少回表

EXPLAIN 关键指标：
- type：const > eq_ref > ref > range > index > ALL
- Extra：Using index（好）、Using filesort（可优化）

索引失效场景：
- 函数操作列
- 隐式类型转换
- LIKE 前导通配符
- OR 连接不同索引列

查询优化：
- 深分页使用延迟关联或游标
- 子查询改写为 JOIN
- 确保 JOIN 列有索引

性能分析工具：
- EXPLAIN / EXPLAIN ANALYZE
- 慢查询日志
- Performance Schema
- SHOW PROFILE
*/

```
