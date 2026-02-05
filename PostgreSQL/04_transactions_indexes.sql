-- ============================================================
--                    PostgreSQL 事务与索引
-- ============================================================
-- 本文件介绍 PostgreSQL 事务处理和索引优化。
-- ============================================================

-- \c learn_postgresql

-- ============================================================
--                    1. 事务基础
-- ============================================================

/*
PostgreSQL 事务特性：
- 完整的 ACID 支持
- 多种隔离级别
- 保存点支持
- 可延迟约束
*/

-- 创建测试表
CREATE TABLE IF NOT EXISTS accounts (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    balance NUMERIC(10, 2) NOT NULL DEFAULT 0
        CHECK (balance >= 0)
);

TRUNCATE TABLE accounts RESTART IDENTITY;
INSERT INTO accounts (name, balance) VALUES
    ('Alice', 1000.00),
    ('Bob', 500.00),
    ('Charlie', 2000.00);

-- --- 基本事务 ---

-- 开始事务
BEGIN;
-- 或 START TRANSACTION;

-- 转账操作
UPDATE accounts SET balance = balance - 200 WHERE name = 'Alice';
UPDATE accounts SET balance = balance + 200 WHERE name = 'Bob';

-- 检查结果
SELECT * FROM accounts;

-- 提交
COMMIT;
-- 或 END;

-- --- 回滚 ---
BEGIN;
UPDATE accounts SET balance = balance - 500 WHERE name = 'Charlie';
-- 发现问题，回滚
ROLLBACK;

-- --- 保存点 ---
BEGIN;

UPDATE accounts SET balance = balance + 100 WHERE name = 'Alice';
SAVEPOINT sp1;

UPDATE accounts SET balance = balance + 100 WHERE name = 'Bob';
SAVEPOINT sp2;

UPDATE accounts SET balance = balance + 100 WHERE name = 'Charlie';

-- 回滚到 sp2
ROLLBACK TO SAVEPOINT sp2;

-- 只有 Alice 和 Bob 的更新保留
COMMIT;


-- ============================================================
--                    2. 隔离级别
-- ============================================================

/*
PostgreSQL 隔离级别：
1. READ UNCOMMITTED：在 PG 中等同于 READ COMMITTED
2. READ COMMITTED：默认级别，每条语句看到最新已提交数据
3. REPEATABLE READ：事务期间看到一致的快照
4. SERIALIZABLE：完全串行化，最严格

PostgreSQL 使用 MVCC（多版本并发控制）实现隔离
*/

-- 查看当前隔离级别
SHOW transaction_isolation;

-- 设置事务隔离级别
BEGIN ISOLATION LEVEL REPEATABLE READ;
-- 执行操作
COMMIT;

-- 设置会话级别
SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- --- SERIALIZABLE 隔离级别 ---
-- 最严格，可能抛出序列化错误，需要重试

BEGIN ISOLATION LEVEL SERIALIZABLE;
-- 如果检测到序列化冲突
-- ERROR: could not serialize access due to read/write dependencies
COMMIT;

-- --- 只读事务 ---
BEGIN READ ONLY;
SELECT * FROM accounts;
-- UPDATE accounts SET balance = 0;  -- 错误：只读事务
COMMIT;

-- --- 可延迟事务 ---
-- 适用于长时间只读事务
BEGIN ISOLATION LEVEL SERIALIZABLE READ ONLY DEFERRABLE;
-- 系统会等待一个安全的快照点
COMMIT;


-- ============================================================
--                    3. 锁机制
-- ============================================================

/*
PostgreSQL 锁类型：
- 行级锁：FOR UPDATE, FOR SHARE 等
- 表级锁：LOCK TABLE
- 咨询锁：自定义锁

MVCC 减少了锁冲突，读写不阻塞
*/

-- --- 行级锁 ---

-- FOR UPDATE：排他锁
BEGIN;
SELECT * FROM accounts WHERE name = 'Alice' FOR UPDATE;
-- 其他事务无法更新此行
UPDATE accounts SET balance = balance + 50 WHERE name = 'Alice';
COMMIT;

-- FOR SHARE：共享锁
BEGIN;
SELECT * FROM accounts WHERE name = 'Bob' FOR SHARE;
-- 其他事务可以读取，但不能更新
COMMIT;

-- FOR NO KEY UPDATE：较弱的排他锁
-- 不阻塞只修改非键列的操作

-- FOR KEY SHARE：较弱的共享锁
-- 不阻塞 FOR NO KEY UPDATE

-- NOWAIT：获取不到锁立即报错
BEGIN;
SELECT * FROM accounts WHERE name = 'Alice' FOR UPDATE NOWAIT;
COMMIT;

-- SKIP LOCKED：跳过已锁定的行
BEGIN;
SELECT * FROM accounts FOR UPDATE SKIP LOCKED LIMIT 1;
COMMIT;

-- --- 表级锁 ---

-- 显式锁表
BEGIN;
LOCK TABLE accounts IN SHARE MODE;  -- 共享模式
-- LOCK TABLE accounts IN EXCLUSIVE MODE;  -- 排他模式
COMMIT;

-- --- 咨询锁（应用级锁）---

-- 获取咨询锁
SELECT pg_advisory_lock(12345);

-- 尝试获取（非阻塞）
SELECT pg_try_advisory_lock(12345);

-- 释放咨询锁
SELECT pg_advisory_unlock(12345);

-- 事务级咨询锁（事务结束自动释放）
SELECT pg_advisory_xact_lock(12345);

-- --- 查看锁信息 ---

-- 当前锁
SELECT * FROM pg_locks;

-- 锁等待
SELECT
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity
    ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity
    ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;


-- ============================================================
--                    4. 索引类型
-- ============================================================

/*
PostgreSQL 索引类型：
- B-tree：默认，适合比较和范围查询
- Hash：等值比较（PostgreSQL 10+ 已可靠）
- GiST：通用搜索树，几何、全文搜索
- SP-GiST：空间分区 GiST
- GIN：倒排索引，数组、JSONB、全文
- BRIN：块范围索引，大表按顺序存储的数据
*/

-- 创建测试表
CREATE TABLE IF NOT EXISTS logs (
    id SERIAL PRIMARY KEY,
    level VARCHAR(10),
    message TEXT,
    metadata JSONB,
    tags TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- --- B-tree 索引（默认）---
CREATE INDEX idx_logs_level ON logs(level);
CREATE INDEX idx_logs_created ON logs(created_at DESC);

-- 多列索引
CREATE INDEX idx_logs_level_created ON logs(level, created_at);

-- 部分索引
CREATE INDEX idx_logs_errors ON logs(created_at)
WHERE level = 'ERROR';

-- 表达式索引
CREATE INDEX idx_logs_date ON logs(DATE(created_at));
CREATE INDEX idx_logs_lower_level ON logs(LOWER(level));

-- 唯一索引
CREATE UNIQUE INDEX idx_logs_unique ON logs(id);

-- --- Hash 索引 ---
CREATE INDEX idx_logs_level_hash ON logs USING HASH (level);

-- --- GIN 索引（用于 JSONB 和数组）---
CREATE INDEX idx_logs_metadata ON logs USING GIN (metadata);
CREATE INDEX idx_logs_tags ON logs USING GIN (tags);

-- JSONB 路径操作索引
CREATE INDEX idx_logs_metadata_ops ON logs USING GIN (metadata jsonb_path_ops);

-- --- GiST 索引（用于几何和全文）---
-- CREATE INDEX idx_posts_search ON posts USING GiST (search_vector);

-- --- BRIN 索引（块范围索引）---
-- 适合大表，数据按物理顺序与索引列相关
CREATE INDEX idx_logs_created_brin ON logs USING BRIN (created_at);

-- --- 并发创建索引 ---
-- 不阻塞写操作
CREATE INDEX CONCURRENTLY idx_logs_message ON logs(message);

-- --- 覆盖索引（INCLUDE）---
-- PostgreSQL 11+
CREATE INDEX idx_logs_level_include ON logs(level) INCLUDE (message, created_at);


-- ============================================================
--                    5. 索引维护
-- ============================================================

-- 查看索引
SELECT
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'logs';

-- 查看索引大小
SELECT
    indexrelname AS index_name,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE relname = 'logs';

-- 查看索引使用情况
SELECT
    indexrelname AS index_name,
    idx_scan AS times_used,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched
FROM pg_stat_user_indexes
WHERE relname = 'logs';

-- 重建索引
REINDEX INDEX idx_logs_level;
REINDEX TABLE logs;

-- 并发重建（PostgreSQL 12+）
REINDEX INDEX CONCURRENTLY idx_logs_level;

-- 删除索引
DROP INDEX IF EXISTS idx_logs_message;


-- ============================================================
--                    6. 执行计划
-- ============================================================

-- 基本 EXPLAIN
EXPLAIN SELECT * FROM employees WHERE department_id = 1;

-- 详细分析（实际执行）
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT * FROM employees WHERE department_id = 1;

-- JSON 格式
EXPLAIN (ANALYZE, FORMAT JSON)
SELECT * FROM employees WHERE salary > 20000;

/*
EXPLAIN 输出解读：
- Seq Scan：顺序扫描（全表扫描）
- Index Scan：索引扫描
- Index Only Scan：仅索引扫描（覆盖索引）
- Bitmap Heap Scan：位图堆扫描
- Nested Loop：嵌套循环连接
- Hash Join：哈希连接
- Merge Join：合并连接

关键指标：
- cost：估计成本（启动成本..总成本）
- rows：估计行数
- actual time：实际执行时间
- loops：循环次数
*/

-- 强制使用/不使用索引
SET enable_seqscan = off;  -- 禁用顺序扫描
EXPLAIN SELECT * FROM employees WHERE department_id = 1;
SET enable_seqscan = on;


-- ============================================================
--                    7. 表分区
-- ============================================================

/*
PostgreSQL 10+ 原生分区支持：
- RANGE：范围分区
- LIST：列表分区
- HASH：哈希分区（PostgreSQL 11+）
*/

-- --- 范围分区 ---
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL,
    user_id INTEGER,
    total_amount NUMERIC(10, 2),
    order_date DATE NOT NULL,
    status VARCHAR(20)
) PARTITION BY RANGE (order_date);

-- 创建分区
CREATE TABLE orders_2023 PARTITION OF orders
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

CREATE TABLE orders_2024 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- 默认分区
CREATE TABLE orders_default PARTITION OF orders DEFAULT;

-- 在分区上创建索引（自动在所有分区创建）
CREATE INDEX idx_orders_date ON orders(order_date);

-- 插入数据（自动路由到正确分区）
INSERT INTO orders (user_id, total_amount, order_date, status)
VALUES (1, 100.00, '2024-03-15', 'completed');

-- 查询分区信息
SELECT
    parent.relname AS parent_table,
    child.relname AS partition_name,
    pg_get_expr(child.relpartbound, child.oid) AS partition_expression
FROM pg_inherits
JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
JOIN pg_class child ON pg_inherits.inhrelid = child.oid
WHERE parent.relname = 'orders';

-- --- 列表分区 ---
CREATE TABLE IF NOT EXISTS sales (
    id SERIAL,
    region VARCHAR(50) NOT NULL,
    amount NUMERIC(10, 2),
    sale_date DATE
) PARTITION BY LIST (region);

CREATE TABLE sales_east PARTITION OF sales
    FOR VALUES IN ('北京', '上海', '天津');

CREATE TABLE sales_south PARTITION OF sales
    FOR VALUES IN ('广州', '深圳', '海南');

-- --- 哈希分区 ---
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL,
    user_id INTEGER NOT NULL,
    data JSONB,
    created_at TIMESTAMP
) PARTITION BY HASH (user_id);

CREATE TABLE sessions_0 PARTITION OF sessions
    FOR VALUES WITH (MODULUS 4, REMAINDER 0);

CREATE TABLE sessions_1 PARTITION OF sessions
    FOR VALUES WITH (MODULUS 4, REMAINDER 1);

CREATE TABLE sessions_2 PARTITION OF sessions
    FOR VALUES WITH (MODULUS 4, REMAINDER 2);

CREATE TABLE sessions_3 PARTITION OF sessions
    FOR VALUES WITH (MODULUS 4, REMAINDER 3);

-- --- 分区维护 ---

-- 分离分区
ALTER TABLE orders DETACH PARTITION orders_2023;

-- 附加分区
ALTER TABLE orders ATTACH PARTITION orders_2023
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

-- 删除分区
-- DROP TABLE orders_2023;


-- ============================================================
--                    8. 性能优化
-- ============================================================

-- --- 统计信息 ---

-- 更新统计信息
ANALYZE employees;
ANALYZE accounts;

-- 查看表统计信息
SELECT
    relname AS table_name,
    n_live_tup AS live_rows,
    n_dead_tup AS dead_rows,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
WHERE relname IN ('employees', 'accounts');

-- --- VACUUM ---

-- 回收空间
VACUUM employees;

-- 详细信息
VACUUM VERBOSE employees;

-- 完全清理（锁表）
VACUUM FULL employees;

-- 自动 vacuum 配置
SHOW autovacuum;
-- ALTER TABLE employees SET (autovacuum_vacuum_threshold = 50);

-- --- 配置优化 ---

-- 查看配置
SHOW shared_buffers;
SHOW work_mem;
SHOW effective_cache_size;

-- 会话级调整
SET work_mem = '256MB';

-- --- 连接池 ---
-- 使用 PgBouncer 或内置连接池

-- 查看当前连接
SELECT * FROM pg_stat_activity;

-- 终止连接
-- SELECT pg_terminate_backend(pid);


-- ============================================================
--                    总结
-- ============================================================

/*
事务特性：
- BEGIN / COMMIT / ROLLBACK
- SAVEPOINT
- 隔离级别：READ COMMITTED（默认）、REPEATABLE READ、SERIALIZABLE
- 只读事务、可延迟事务

锁机制：
- FOR UPDATE / FOR SHARE
- NOWAIT / SKIP LOCKED
- 咨询锁 pg_advisory_lock
- MVCC 减少锁冲突

索引类型：
- B-tree：默认，比较和范围
- Hash：等值比较
- GIN：数组、JSONB、全文
- GiST：几何、全文
- BRIN：大表顺序数据

索引特性：
- 部分索引
- 表达式索引
- 覆盖索引（INCLUDE）
- 并发创建（CONCURRENTLY）

分区：
- RANGE：范围分区
- LIST：列表分区
- HASH：哈希分区
- 自动路由

性能优化：
- EXPLAIN ANALYZE
- VACUUM / ANALYZE
- 合理配置参数
- 使用连接池
*/
