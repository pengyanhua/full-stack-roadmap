# basics

::: info 文件信息
- 📄 原文件：`01_basics.sql`
- 🔤 语言：SQL
:::

## SQL 脚本

```sql
-- ============================================================
--                    PostgreSQL 基础教程
-- ============================================================
-- 本文件介绍 PostgreSQL 数据库的基础操作。
--
-- 运行方式：
--   psql -U postgres -f 01_basics.sql
--   或在 psql 客户端中逐条执行
-- ============================================================

-- ============================================================
--                    1. 数据库操作
-- ============================================================

-- 查看所有数据库
-- \l 或 \list (psql 命令)
SELECT datname FROM pg_database;

-- 创建数据库
CREATE DATABASE learn_postgresql
    WITH ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TEMPLATE = template0;

-- 连接数据库
-- \c learn_postgresql (psql 命令)

-- 查看当前数据库
SELECT current_database();

-- 查看数据库信息
-- \conninfo (psql 命令)

-- 删除数据库（谨慎使用！）
-- DROP DATABASE IF EXISTS learn_postgresql;


-- ============================================================
--                    2. 数据类型
-- ============================================================

/*
PostgreSQL 数据类型丰富：

【数值类型】
- SMALLINT      : 2字节，-32768 到 32767
- INTEGER/INT   : 4字节，约 ±21亿
- BIGINT        : 8字节
- DECIMAL(p,s)  : 精确小数，p总位数，s小数位
- NUMERIC(p,s)  : 同 DECIMAL
- REAL          : 4字节浮点数
- DOUBLE PRECISION : 8字节浮点数
- SERIAL        : 自增整数（4字节）
- BIGSERIAL     : 自增大整数（8字节）

【字符串类型】
- CHAR(n)       : 定长字符串
- VARCHAR(n)    : 变长字符串
- TEXT          : 无限长度文本

【布尔类型】
- BOOLEAN       : true/false/null

【日期时间类型】
- DATE          : 日期
- TIME          : 时间
- TIMESTAMP     : 日期时间
- TIMESTAMPTZ   : 带时区的日期时间
- INTERVAL      : 时间间隔

【特殊类型】
- UUID          : 通用唯一标识符
- JSON/JSONB    : JSON 数据（JSONB 更高效）
- ARRAY         : 数组
- HSTORE        : 键值对
- INET/CIDR     : IP 地址
- BYTEA         : 二进制数据
- ENUM          : 枚举类型

【几何类型】
- POINT, LINE, CIRCLE, POLYGON 等
*/


-- ============================================================
--                    3. 模式（Schema）
-- ============================================================

-- PostgreSQL 使用 Schema 组织数据库对象
-- 默认 Schema 是 public

-- 创建 Schema
CREATE SCHEMA IF NOT EXISTS app;

-- 设置搜索路径
SET search_path TO app, public;

-- 查看当前搜索路径
SHOW search_path;

-- 查看所有 Schema
-- \dn (psql 命令)
SELECT schema_name FROM information_schema.schemata;


-- ============================================================
--                    4. 表操作
-- ============================================================

-- 创建枚举类型
CREATE TYPE user_status AS ENUM ('active', 'inactive', 'banned');

-- 创建表
CREATE TABLE IF NOT EXISTS users (
    -- 自增主键（推荐使用 IDENTITY）
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- 或使用 SERIAL
    -- id SERIAL PRIMARY KEY,

    -- 用户名，非空，唯一
    username VARCHAR(50) NOT NULL UNIQUE,

    -- 邮箱
    email VARCHAR(100) NOT NULL,

    -- 密码哈希
    password_hash VARCHAR(255) NOT NULL,

    -- 年龄，可为空
    age SMALLINT CHECK (age >= 0 AND age <= 150),

    -- 余额，精确小数
    balance NUMERIC(10, 2) DEFAULT 0.00,

    -- 状态枚举
    status user_status DEFAULT 'active',

    -- 标签数组
    tags TEXT[],

    -- 元数据 JSON
    metadata JSONB DEFAULT '{}',

    -- 创建时间
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- 更新时间
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 添加表注释
COMMENT ON TABLE users IS '用户表';
COMMENT ON COLUMN users.username IS '用户名，唯一标识';
COMMENT ON COLUMN users.balance IS '账户余额';

-- 查看表结构
-- \d users (psql 命令)
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_name = 'users';

-- --- 修改表 ---

-- 添加列
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- 修改列类型
ALTER TABLE users ALTER COLUMN phone TYPE VARCHAR(30);

-- 重命名列
ALTER TABLE users RENAME COLUMN phone TO mobile;

-- 删除列
ALTER TABLE users DROP COLUMN IF EXISTS mobile;

-- 添加约束
ALTER TABLE users ADD CONSTRAINT chk_balance CHECK (balance >= 0);

-- 删除约束
ALTER TABLE users DROP CONSTRAINT IF EXISTS chk_balance;

-- 创建索引
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_created ON users(created_at);
CREATE INDEX idx_users_tags ON users USING GIN(tags);  -- 数组索引
CREATE INDEX idx_users_metadata ON users USING GIN(metadata);  -- JSONB 索引

-- 删除索引
DROP INDEX IF EXISTS idx_users_created;


-- ============================================================
--                    5. CRUD 操作
-- ============================================================

-- --- INSERT 插入 ---

-- 单行插入
INSERT INTO users (username, email, password_hash, age, balance, tags)
VALUES ('alice', 'alice@example.com', 'hash123', 25, 100.00, ARRAY['vip', 'early-adopter']);

-- 多行插入
INSERT INTO users (username, email, password_hash, age, balance, tags) VALUES
    ('bob', 'bob@example.com', 'hash456', 30, 200.50, ARRAY['regular']),
    ('charlie', 'charlie@example.com', 'hash789', 28, 150.00, ARRAY['vip']),
    ('diana', 'diana@example.com', 'hashabc', 35, 500.00, ARRAY['premium', 'vip']),
    ('eve', 'eve@example.com', 'hashdef', 22, 50.00, NULL);

-- 插入并返回数据
INSERT INTO users (username, email, password_hash)
VALUES ('frank', 'frank@example.com', 'hashghi')
RETURNING id, username, created_at;

-- 冲突处理（UPSERT）
INSERT INTO users (username, email, password_hash)
VALUES ('alice', 'alice_new@example.com', 'newhash')
ON CONFLICT (username) DO UPDATE SET
    email = EXCLUDED.email,
    updated_at = CURRENT_TIMESTAMP;

-- 冲突时不做任何事
INSERT INTO users (username, email, password_hash)
VALUES ('alice', 'alice@example.com', 'hash123')
ON CONFLICT DO NOTHING;

-- --- SELECT 查询 ---

-- 查询所有列
SELECT * FROM users;

-- 查询指定列
SELECT id, username, email, balance FROM users;

-- 使用别名
SELECT
    id AS "用户ID",
    username AS "用户名",
    balance AS "余额"
FROM users;

-- 去重
SELECT DISTINCT status FROM users;

-- --- WHERE 条件 ---

-- 比较运算符
SELECT * FROM users WHERE age >= 25;
SELECT * FROM users WHERE status = 'active';
SELECT * FROM users WHERE balance != 0;  -- 或 <>

-- 逻辑运算符
SELECT * FROM users WHERE age >= 25 AND balance > 100;
SELECT * FROM users WHERE age < 25 OR balance > 300;
SELECT * FROM users WHERE NOT status = 'banned';

-- BETWEEN 范围
SELECT * FROM users WHERE age BETWEEN 25 AND 35;
SELECT * FROM users WHERE created_at BETWEEN '2024-01-01' AND '2024-12-31';

-- IN 列表
SELECT * FROM users WHERE status IN ('active', 'inactive');

-- LIKE 模糊匹配
SELECT * FROM users WHERE username LIKE 'a%';      -- 以 a 开头
SELECT * FROM users WHERE email LIKE '%@example.com';
SELECT * FROM users WHERE username ILIKE '%LI%';   -- 不区分大小写

-- 正则表达式
SELECT * FROM users WHERE username ~ '^[a-c]';     -- 以 a-c 开头
SELECT * FROM users WHERE email ~* 'example';      -- 不区分大小写

-- NULL 判断
SELECT * FROM users WHERE age IS NULL;
SELECT * FROM users WHERE tags IS NOT NULL;

-- 数组操作
SELECT * FROM users WHERE 'vip' = ANY(tags);       -- 包含 vip
SELECT * FROM users WHERE tags @> ARRAY['vip'];    -- 包含 vip
SELECT * FROM users WHERE tags && ARRAY['vip', 'premium'];  -- 有交集

-- JSONB 操作
SELECT * FROM users WHERE metadata @> '{"level": "gold"}';
SELECT * FROM users WHERE metadata ? 'level';      -- 存在键
SELECT * FROM users WHERE metadata ->> 'level' = 'gold';

-- --- ORDER BY 排序 ---

-- 升序（默认）
SELECT * FROM users ORDER BY age ASC;

-- 降序
SELECT * FROM users ORDER BY balance DESC;

-- NULL 排序位置
SELECT * FROM users ORDER BY age NULLS FIRST;
SELECT * FROM users ORDER BY age NULLS LAST;

-- 多列排序
SELECT * FROM users ORDER BY status ASC, balance DESC;

-- --- LIMIT 和 OFFSET 分页 ---

-- 获取前 N 条
SELECT * FROM users LIMIT 3;

-- 分页
SELECT * FROM users ORDER BY id LIMIT 3 OFFSET 0;   -- 第1页
SELECT * FROM users ORDER BY id LIMIT 3 OFFSET 3;   -- 第2页

-- FETCH（SQL 标准语法）
SELECT * FROM users ORDER BY id FETCH FIRST 3 ROWS ONLY;
SELECT * FROM users ORDER BY id OFFSET 3 FETCH NEXT 3 ROWS ONLY;

-- --- UPDATE 更新 ---

-- 更新单条
UPDATE users SET balance = balance + 50 WHERE id = 1;

-- 更新多列
UPDATE users SET
    age = 26,
    status = 'active',
    balance = 200.00,
    updated_at = CURRENT_TIMESTAMP
WHERE username = 'alice';

-- 更新并返回
UPDATE users SET balance = balance * 1.1
WHERE status = 'active'
RETURNING id, username, balance;

-- 使用子查询更新
UPDATE users SET balance = (
    SELECT AVG(balance) FROM users
)
WHERE balance IS NULL;

-- --- DELETE 删除 ---

-- 删除指定记录
DELETE FROM users WHERE id = 5;

-- 删除并返回
DELETE FROM users WHERE status = 'banned'
RETURNING *;

-- 删除所有（保留表结构）
-- DELETE FROM users;

-- 清空表（更快，重置序列）
-- TRUNCATE TABLE users RESTART IDENTITY;


-- ============================================================
--                    6. 聚合函数
-- ============================================================

-- 计数
SELECT COUNT(*) AS total_users FROM users;
SELECT COUNT(age) AS users_with_age FROM users;
SELECT COUNT(DISTINCT status) AS status_count FROM users;

-- 求和
SELECT SUM(balance) AS total_balance FROM users;

-- 平均值
SELECT AVG(age)::NUMERIC(10,2) AS average_age FROM users;
SELECT ROUND(AVG(balance), 2) AS average_balance FROM users;

-- 最大/最小值
SELECT MAX(balance) AS max_balance FROM users;
SELECT MIN(age) AS min_age FROM users;

-- 字符串聚合
SELECT STRING_AGG(username, ', ') AS all_usernames FROM users;
SELECT STRING_AGG(username, ', ' ORDER BY username) AS sorted_usernames FROM users;

-- 数组聚合
SELECT ARRAY_AGG(username) AS username_array FROM users;

-- 组合使用
SELECT
    COUNT(*) AS 用户数,
    SUM(balance) AS 总余额,
    ROUND(AVG(balance), 2) AS 平均余额,
    MAX(balance) AS 最高余额,
    MIN(balance) AS 最低余额
FROM users
WHERE status = 'active';


-- ============================================================
--                    7. GROUP BY 分组
-- ============================================================

-- 按状态分组统计
SELECT
    status,
    COUNT(*) AS user_count,
    SUM(balance) AS total_balance,
    ROUND(AVG(balance), 2) AS avg_balance
FROM users
GROUP BY status;

-- HAVING 过滤分组
SELECT
    status,
    COUNT(*) AS user_count,
    AVG(balance) AS avg_balance
FROM users
GROUP BY status
HAVING COUNT(*) >= 2 AND AVG(balance) > 100;

-- GROUPING SETS（多维分组）
SELECT
    status,
    CASE WHEN age < 30 THEN '青年' ELSE '中年' END AS age_group,
    COUNT(*) AS user_count
FROM users
GROUP BY GROUPING SETS (
    (status),
    (CASE WHEN age < 30 THEN '青年' ELSE '中年' END),
    ()  -- 总计
);

-- ROLLUP（层级汇总）
SELECT
    COALESCE(status::TEXT, '总计') AS status,
    COUNT(*) AS user_count,
    SUM(balance) AS total_balance
FROM users
GROUP BY ROLLUP(status);

-- CUBE（所有组合）
SELECT
    status,
    CASE WHEN age < 30 THEN '青年' ELSE '中年' END AS age_group,
    COUNT(*) AS user_count
FROM users
GROUP BY CUBE(status, CASE WHEN age < 30 THEN '青年' ELSE '中年' END);


-- ============================================================
--                    8. 字符串函数
-- ============================================================

SELECT
    -- 连接字符串
    username || ' <' || email || '>' AS user_info,
    CONCAT(username, ' - ', status) AS combined,
    CONCAT_WS('-', id::TEXT, username, status::TEXT) AS ws_combined,

    -- 大小写转换
    UPPER(username) AS upper_name,
    LOWER(email) AS lower_email,
    INITCAP(username) AS init_cap,

    -- 截取
    LEFT(email, 5) AS left_5,
    RIGHT(email, 10) AS right_10,
    SUBSTRING(email FROM 1 FOR 5) AS sub_str,
    SUBSTRING(email FROM '@(.*)$') AS domain,  -- 正则提取

    -- 长度
    LENGTH(username) AS char_length,
    OCTET_LENGTH(username) AS byte_length,

    -- 查找
    POSITION('@' IN email) AS at_position,
    STRPOS(email, '@') AS at_pos,

    -- 替换
    REPLACE(email, '@example.com', '@test.com') AS new_email,
    REGEXP_REPLACE(email, '@.*$', '@new.com') AS regex_replace,

    -- 去空格
    TRIM('  hello  ') AS trimmed,
    LTRIM('  hello') AS left_trimmed,
    RTRIM('hello  ') AS right_trimmed,
    TRIM(BOTH 'x' FROM 'xxxhelloxx') AS trim_char,

    -- 填充
    LPAD(id::TEXT, 5, '0') AS padded_id,
    RPAD(username, 10, '.') AS padded_name,

    -- 反转
    REVERSE(username) AS reversed,

    -- 分割
    SPLIT_PART(email, '@', 1) AS email_user,
    SPLIT_PART(email, '@', 2) AS email_domain
FROM users
LIMIT 1;


-- ============================================================
--                    9. 日期时间函数
-- ============================================================

SELECT
    -- 当前日期时间
    NOW() AS now,
    CURRENT_TIMESTAMP AS current_ts,
    CURRENT_DATE AS today,
    CURRENT_TIME AS current_time,
    LOCALTIME AS local_time,
    LOCALTIMESTAMP AS local_timestamp,

    -- 提取部分
    EXTRACT(YEAR FROM created_at) AS year,
    EXTRACT(MONTH FROM created_at) AS month,
    EXTRACT(DAY FROM created_at) AS day,
    EXTRACT(HOUR FROM created_at) AS hour,
    EXTRACT(DOW FROM created_at) AS day_of_week,  -- 0=周日
    EXTRACT(DOY FROM created_at) AS day_of_year,
    EXTRACT(WEEK FROM created_at) AS week_number,
    EXTRACT(EPOCH FROM created_at) AS unix_timestamp,

    -- 截断
    DATE_TRUNC('month', created_at) AS month_start,
    DATE_TRUNC('year', created_at) AS year_start,
    DATE_TRUNC('hour', created_at) AS hour_start,

    -- 格式化
    TO_CHAR(created_at, 'YYYY年MM月DD日 HH24:MI:SS') AS formatted,
    TO_CHAR(created_at, 'YYYY-MM-DD') AS date_only,
    TO_CHAR(created_at, 'Day, Month DD, YYYY') AS long_format,

    -- 日期计算
    created_at + INTERVAL '7 days' AS plus_7_days,
    created_at - INTERVAL '1 month' AS minus_1_month,
    created_at + INTERVAL '1 hour 30 minutes' AS plus_90_min,

    -- 日期差
    AGE(NOW(), created_at) AS age,
    NOW() - created_at AS interval_diff,
    DATE_PART('day', NOW() - created_at) AS days_ago,

    -- 从字符串解析
    TO_TIMESTAMP('2024-01-15 10:30:00', 'YYYY-MM-DD HH24:MI:SS') AS parsed,
    TO_DATE('2024-01-15', 'YYYY-MM-DD') AS parsed_date

FROM users
WHERE id = 1;


-- ============================================================
--                    10. 条件表达式
-- ============================================================

SELECT
    username,
    balance,

    -- CASE WHEN
    CASE status
        WHEN 'active' THEN '活跃'
        WHEN 'inactive' THEN '不活跃'
        WHEN 'banned' THEN '已封禁'
        ELSE '未知'
    END AS status_cn,

    -- CASE 搜索形式
    CASE
        WHEN balance >= 500 THEN 'VIP'
        WHEN balance >= 200 THEN '高级'
        WHEN balance >= 100 THEN '普通'
        ELSE '新用户'
    END AS user_level,

    -- COALESCE（返回第一个非空值）
    COALESCE(age, 0) AS age_or_zero,
    COALESCE(tags, ARRAY['default']) AS tags_or_default,

    -- NULLIF（相等则返回 NULL）
    NULLIF(balance, 0) AS null_if_zero,

    -- GREATEST / LEAST
    GREATEST(balance, 100) AS at_least_100,
    LEAST(balance, 500) AS at_most_500

FROM users;


-- ============================================================
--                    总结
-- ============================================================

/*
PostgreSQL 特色功能：

Schema：
- 组织数据库对象
- 设置 search_path

高级数据类型：
- ARRAY：数组类型
- JSONB：高效 JSON
- UUID：唯一标识符
- 自定义 ENUM

CRUD 增强：
- RETURNING：返回修改的行
- ON CONFLICT：UPSERT 操作
- FETCH：SQL 标准分页

聚合与分组：
- STRING_AGG / ARRAY_AGG
- GROUPING SETS / ROLLUP / CUBE

字符串处理：
- || 连接运算符
- 正则表达式支持
- SPLIT_PART

日期时间：
- EXTRACT 提取部分
- DATE_TRUNC 截断
- INTERVAL 计算
- AGE 函数

与 MySQL 主要区别：
- 使用 SERIAL/IDENTITY 代替 AUTO_INCREMENT
- 字符串连接用 || 代替 CONCAT（也支持）
- 更强大的日期函数
- 原生支持数组和 JSON
- Schema 概念
- 更严格的类型检查
*/

```
