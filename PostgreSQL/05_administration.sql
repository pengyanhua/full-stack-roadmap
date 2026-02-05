-- ============================================================
--                    PostgreSQL 管理与运维
-- ============================================================
-- 本文件介绍 PostgreSQL 数据库管理和运维知识。
-- ============================================================

-- ============================================================
--                    1. 用户与权限管理
-- ============================================================

-- --- 创建角色/用户 ---
-- PostgreSQL 中用户和角色是相同的概念

-- 创建角色
CREATE ROLE app_user WITH LOGIN PASSWORD 'secure_password';

-- 创建超级用户
CREATE ROLE admin_user WITH SUPERUSER LOGIN PASSWORD 'admin_pass';

-- 创建只读用户
CREATE ROLE readonly_user WITH LOGIN PASSWORD 'readonly_pass';

-- 带选项创建
CREATE ROLE dev_user WITH
    LOGIN
    PASSWORD 'dev_pass'
    CREATEDB
    CREATEROLE
    VALID UNTIL '2025-12-31';

-- --- 修改角色 ---
ALTER ROLE app_user WITH PASSWORD 'new_password';
ALTER ROLE app_user VALID UNTIL '2025-06-30';
ALTER ROLE app_user RENAME TO application_user;

-- --- 删除角色 ---
-- DROP ROLE IF EXISTS app_user;

-- --- 角色属性 ---
/*
LOGIN：可以登录
SUPERUSER：超级用户
CREATEDB：可以创建数据库
CREATEROLE：可以创建角色
REPLICATION：可以进行流复制
INHERIT：继承组权限
*/

-- --- 组角色 ---
CREATE ROLE developers;
CREATE ROLE testers;

-- 将用户加入组
GRANT developers TO dev_user;
GRANT testers TO readonly_user;

-- 从组中移除
REVOKE developers FROM dev_user;

-- --- 数据库权限 ---

-- 授予数据库连接权限
GRANT CONNECT ON DATABASE learn_postgresql TO app_user;

-- 授予 Schema 使用权限
GRANT USAGE ON SCHEMA public TO app_user;

-- 授予表权限
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE employees TO app_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;

-- 授予序列权限
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;

-- 授予所有权限
GRANT ALL PRIVILEGES ON TABLE employees TO admin_user;

-- --- 默认权限 ---
-- 自动为新创建的对象设置权限
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO readonly_user;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO developers;

-- --- 撤销权限 ---
REVOKE DELETE ON TABLE employees FROM app_user;
REVOKE ALL PRIVILEGES ON TABLE employees FROM app_user;

-- --- 行级安全（RLS）---
-- PostgreSQL 9.5+

-- 启用行级安全
ALTER TABLE employees ENABLE ROW LEVEL SECURITY;

-- 创建策略
CREATE POLICY employee_isolation ON employees
    FOR ALL
    TO app_user
    USING (department_id = current_setting('app.department_id')::INTEGER);

-- 设置应用上下文
SET app.department_id = '1';

-- 查看策略
SELECT * FROM pg_policies WHERE tablename = 'employees';

-- --- 查看权限 ---

-- 表权限
SELECT
    grantee,
    table_name,
    privilege_type
FROM information_schema.table_privileges
WHERE table_schema = 'public';

-- 角色信息
SELECT
    rolname,
    rolsuper,
    rolcreatedb,
    rolcreaterole,
    rolreplication
FROM pg_roles
WHERE rolname NOT LIKE 'pg_%';


-- ============================================================
--                    2. 备份与恢复
-- ============================================================

/*
备份方式：
1. pg_dump：逻辑备份，导出 SQL 或自定义格式
2. pg_basebackup：物理备份，用于流复制
3. 文件系统备份：需要停机或使用 PITR
*/

-- --- pg_dump 命令行示例 ---
/*
# 导出整个数据库（SQL 格式）
pg_dump -U postgres -d learn_postgresql > backup.sql

# 导出为自定义格式（支持并行恢复）
pg_dump -U postgres -Fc -d learn_postgresql > backup.dump

# 导出特定表
pg_dump -U postgres -t employees -d learn_postgresql > employees.sql

# 仅导出结构
pg_dump -U postgres -s -d learn_postgresql > schema.sql

# 仅导出数据
pg_dump -U postgres -a -d learn_postgresql > data.sql

# 导出时排除表
pg_dump -U postgres -T logs -d learn_postgresql > backup.sql

# 并行导出（目录格式）
pg_dump -U postgres -Fd -j 4 -d learn_postgresql -f backup_dir
*/

-- --- pg_restore 恢复 ---
/*
# 从 SQL 文件恢复
psql -U postgres -d learn_postgresql < backup.sql

# 从自定义格式恢复
pg_restore -U postgres -d learn_postgresql backup.dump

# 并行恢复
pg_restore -U postgres -j 4 -d learn_postgresql backup.dump

# 仅恢复特定表
pg_restore -U postgres -t employees -d learn_postgresql backup.dump
*/

-- --- COPY 命令 ---
-- 快速导出/导入数据

-- 导出到 CSV
COPY employees TO '/tmp/employees.csv' WITH CSV HEADER;

-- 导出查询结果
COPY (SELECT * FROM employees WHERE salary > 20000)
TO '/tmp/high_salary.csv' WITH CSV HEADER;

-- 从 CSV 导入
COPY employees FROM '/tmp/employees.csv' WITH CSV HEADER;

-- 使用 \copy（客户端命令，不需要超级用户权限）
-- \copy employees TO '/tmp/employees.csv' WITH CSV HEADER


-- ============================================================
--                    3. 监控与诊断
-- ============================================================

-- --- 数据库状态 ---

-- 数据库大小
SELECT
    datname AS database,
    pg_size_pretty(pg_database_size(datname)) AS size
FROM pg_database
ORDER BY pg_database_size(datname) DESC;

-- 表大小
SELECT
    relname AS table_name,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
    pg_size_pretty(pg_relation_size(relid)) AS table_size,
    pg_size_pretty(pg_indexes_size(relid)) AS index_size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;

-- --- 活动连接 ---

-- 当前连接
SELECT
    pid,
    usename,
    application_name,
    client_addr,
    state,
    query,
    query_start,
    state_change
FROM pg_stat_activity
WHERE datname = current_database();

-- 连接数统计
SELECT
    state,
    COUNT(*) AS count
FROM pg_stat_activity
GROUP BY state;

-- --- 慢查询 ---

-- 启用 pg_stat_statements 扩展
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- 查看慢查询
SELECT
    query,
    calls,
    total_exec_time / calls AS avg_time_ms,
    rows / calls AS avg_rows
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;

-- --- 锁等待 ---

SELECT
    blocked.pid AS blocked_pid,
    blocked.query AS blocked_query,
    blocking.pid AS blocking_pid,
    blocking.query AS blocking_query
FROM pg_stat_activity blocked
JOIN pg_stat_activity blocking ON blocking.pid = ANY(pg_blocking_pids(blocked.pid))
WHERE blocked.pid != blocking.pid;

-- --- 复制状态 ---

-- 主库视图
SELECT
    client_addr,
    state,
    sent_lsn,
    write_lsn,
    flush_lsn,
    replay_lsn
FROM pg_stat_replication;

-- 从库视图
SELECT
    status,
    receive_start_lsn,
    received_lsn,
    latest_end_lsn
FROM pg_stat_wal_receiver;


-- ============================================================
--                    4. 性能调优参数
-- ============================================================

-- --- 内存配置 ---

-- 共享缓冲区（一般设为物理内存的 25%）
SHOW shared_buffers;
-- ALTER SYSTEM SET shared_buffers = '4GB';

-- 工作内存（每个操作的内存）
SHOW work_mem;
-- SET work_mem = '256MB';

-- 维护工作内存
SHOW maintenance_work_mem;
-- ALTER SYSTEM SET maintenance_work_mem = '1GB';

-- 有效缓存大小（查询计划器参考）
SHOW effective_cache_size;
-- ALTER SYSTEM SET effective_cache_size = '12GB';

-- --- 检查点配置 ---

-- WAL 缓冲区
SHOW wal_buffers;

-- 检查点超时
SHOW checkpoint_timeout;

-- 最大 WAL 大小
SHOW max_wal_size;

-- --- 并行查询 ---

-- 最大并行工作进程
SHOW max_parallel_workers_per_gather;
-- SET max_parallel_workers_per_gather = 4;

-- 并行表扫描最小大小
SHOW min_parallel_table_scan_size;

-- --- 日志配置 ---

-- 日志目标
SHOW log_destination;

-- 慢查询日志
SHOW log_min_duration_statement;
-- ALTER SYSTEM SET log_min_duration_statement = '1000';  -- 1秒

-- 记录所有语句
-- ALTER SYSTEM SET log_statement = 'all';

-- --- 应用配置更改 ---
-- 需要重启的参数修改后执行：
-- SELECT pg_reload_conf();
-- 或重启 PostgreSQL 服务


-- ============================================================
--                    5. 高可用与复制
-- ============================================================

/*
PostgreSQL 复制方式：
1. 流复制（Streaming Replication）：主从同步
2. 逻辑复制（Logical Replication）：选择性复制
3. 同步/异步复制
*/

-- --- 创建复制用户 ---
CREATE ROLE replication_user WITH REPLICATION LOGIN PASSWORD 'repl_pass';

-- --- 发布订阅（逻辑复制）---

-- 主库：创建发布
CREATE PUBLICATION my_publication FOR TABLE employees, departments;

-- 创建所有表的发布
CREATE PUBLICATION all_tables FOR ALL TABLES;

-- 从库：创建订阅
-- CREATE SUBSCRIPTION my_subscription
--     CONNECTION 'host=primary_host dbname=learn_postgresql user=replication_user password=repl_pass'
--     PUBLICATION my_publication;

-- 查看发布
SELECT * FROM pg_publication;
SELECT * FROM pg_publication_tables;

-- 查看订阅
SELECT * FROM pg_subscription;

-- --- 复制槽 ---

-- 创建复制槽
SELECT pg_create_physical_replication_slot('replica1');
SELECT pg_create_logical_replication_slot('logical1', 'pgoutput');

-- 查看复制槽
SELECT * FROM pg_replication_slots;

-- 删除复制槽
SELECT pg_drop_replication_slot('replica1');


-- ============================================================
--                    6. 维护任务
-- ============================================================

-- --- VACUUM ---

-- 标准 VACUUM（释放空间供重用）
VACUUM employees;

-- VACUUM FULL（完全重组表，需要排他锁）
VACUUM FULL employees;

-- VACUUM ANALYZE（同时更新统计信息）
VACUUM ANALYZE employees;

-- 自动 VACUUM 配置
SELECT
    name,
    setting,
    short_desc
FROM pg_settings
WHERE name LIKE '%autovacuum%';

-- --- 更新统计信息 ---
ANALYZE employees;
ANALYZE;  -- 整个数据库

-- --- 重建索引 ---
REINDEX TABLE employees;
REINDEX DATABASE learn_postgresql;

-- 并发重建
REINDEX TABLE CONCURRENTLY employees;

-- --- 表空间 ---

-- 创建表空间
-- CREATE TABLESPACE fast_storage LOCATION '/ssd/postgresql/data';

-- 移动表到表空间
-- ALTER TABLE employees SET TABLESPACE fast_storage;

-- 查看表空间
SELECT * FROM pg_tablespace;


-- ============================================================
--                    7. 故障排除
-- ============================================================

-- --- 常见问题诊断 ---

-- 检查表膨胀
SELECT
    relname AS table_name,
    n_live_tup AS live_tuples,
    n_dead_tup AS dead_tuples,
    ROUND(100 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0), 2) AS dead_ratio
FROM pg_stat_user_tables
WHERE n_dead_tup > 0
ORDER BY dead_ratio DESC;

-- 检查索引使用情况
SELECT
    relname AS table_name,
    indexrelname AS index_name,
    idx_scan AS scans,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched
FROM pg_stat_user_indexes
ORDER BY idx_scan;

-- 未使用的索引
SELECT
    relname AS table_name,
    indexrelname AS index_name,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
AND indexrelname NOT LIKE '%pkey%';

-- 检查序列
SELECT
    sequencename,
    last_value
FROM pg_sequences;

-- 检查约束
SELECT
    conname AS constraint_name,
    contype AS type,
    conrelid::regclass AS table_name
FROM pg_constraint
WHERE conrelid IN (
    SELECT oid FROM pg_class WHERE relname IN ('employees', 'departments')
);

-- --- 终止问题查询 ---

-- 取消查询（温和）
SELECT pg_cancel_backend(pid);

-- 终止连接（强制）
SELECT pg_terminate_backend(pid);

-- 终止所有非超级用户连接
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'learn_postgresql'
AND pid <> pg_backend_pid()
AND usename != 'postgres';


-- ============================================================
--                    8. 最佳实践
-- ============================================================

/*
安全最佳实践：
1. 使用强密码
2. 限制超级用户使用
3. 使用最小权限原则
4. 启用 SSL 连接
5. 配置 pg_hba.conf 限制访问
6. 定期审计权限

性能最佳实践：
1. 合理配置内存参数
2. 使用连接池
3. 定期 VACUUM 和 ANALYZE
4. 监控慢查询
5. 合理设计索引
6. 分区大表

备份最佳实践：
1. 定期全量备份
2. 启用 WAL 归档
3. 测试恢复流程
4. 异地存储备份
5. 监控备份状态

高可用最佳实践：
1. 使用流复制
2. 配置同步复制（关键数据）
3. 监控复制延迟
4. 自动故障转移
5. 定期测试切换

监控最佳实践：
1. 启用 pg_stat_statements
2. 设置慢查询日志
3. 监控连接数
4. 监控磁盘空间
5. 设置告警阈值
*/


-- ============================================================
--                    总结
-- ============================================================

/*
用户权限：
- CREATE ROLE / USER
- GRANT / REVOKE
- 行级安全（RLS）
- 默认权限

备份恢复：
- pg_dump / pg_restore
- COPY 命令
- WAL 归档

监控：
- pg_stat_activity
- pg_stat_statements
- pg_stat_user_tables
- pg_stat_user_indexes

性能参数：
- shared_buffers
- work_mem
- effective_cache_size
- max_parallel_workers

复制：
- 流复制
- 逻辑复制（发布/订阅）
- 复制槽

维护：
- VACUUM / ANALYZE
- REINDEX
- 表空间管理

故障排除：
- 表膨胀检查
- 索引使用分析
- 问题查询处理
*/
