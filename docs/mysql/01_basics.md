# basics

::: info 文件信息
- 📄 原文件：`01_basics.sql`
- 🔤 语言：SQL
:::

## SQL 脚本

```sql
-- ============================================================
--                    MySQL 基础教程
-- ============================================================
-- 本文件介绍 MySQL 数据库的基础操作。
--
-- 运行方式：
--   mysql -u root -p < 01_basics.sql
--   或在 MySQL 客户端中逐条执行
-- ============================================================

-- ============================================================
--                    1. 数据库操作
-- ============================================================

-- 查看所有数据库
SHOW DATABASES;

-- 创建数据库
-- CHARACTER SET: 字符集，推荐 utf8mb4（支持完整 Unicode，包括 emoji）
-- COLLATE: 排序规则，utf8mb4_unicode_ci 大小写不敏感
CREATE DATABASE IF NOT EXISTS learn_mysql
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

-- 使用数据库
USE learn_mysql;

-- 查看当前数据库
SELECT DATABASE();

-- 查看数据库创建语句
SHOW CREATE DATABASE learn_mysql;

-- 删除数据库（谨慎使用！）
-- DROP DATABASE IF EXISTS learn_mysql;


-- ============================================================
--                    2. 数据类型
-- ============================================================

/*
MySQL 主要数据类型：

【数值类型】
- TINYINT      : 1字节，-128 到 127（或 0-255 UNSIGNED）
- SMALLINT     : 2字节，-32768 到 32767
- MEDIUMINT    : 3字节
- INT/INTEGER  : 4字节，约 ±21亿
- BIGINT       : 8字节，约 ±922京
- DECIMAL(M,D) : 精确小数，M总位数，D小数位数
- FLOAT        : 4字节浮点数
- DOUBLE       : 8字节浮点数

【字符串类型】
- CHAR(N)      : 定长字符串，N=0-255
- VARCHAR(N)   : 变长字符串，N=0-65535
- TEXT         : 长文本，最大 65535 字节
- MEDIUMTEXT   : 中等长度文本，最大 16MB
- LONGTEXT     : 长文本，最大 4GB
- ENUM         : 枚举类型
- SET          : 集合类型

【日期时间类型】
- DATE         : 日期，'YYYY-MM-DD'
- TIME         : 时间，'HH:MM:SS'
- DATETIME     : 日期时间，'YYYY-MM-DD HH:MM:SS'
- TIMESTAMP    : 时间戳，自动更新
- YEAR         : 年份

【二进制类型】
- BINARY(N)    : 定长二进制
- VARBINARY(N) : 变长二进制
- BLOB         : 二进制大对象
- JSON         : JSON 数据（MySQL 5.7+）
*/


-- ============================================================
--                    3. 表操作
-- ============================================================

-- --- 创建表 ---
CREATE TABLE IF NOT EXISTS users (
    -- 主键，自增
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,

    -- 用户名，非空，唯一
    username VARCHAR(50) NOT NULL UNIQUE,

    -- 邮箱
    email VARCHAR(100) NOT NULL,

    -- 密码哈希
    password_hash VARCHAR(255) NOT NULL,

    -- 年龄，可为空
    age TINYINT UNSIGNED,

    -- 余额，精确小数
    balance DECIMAL(10, 2) DEFAULT 0.00,

    -- 状态枚举
    status ENUM('active', 'inactive', 'banned') DEFAULT 'active',

    -- 创建时间，默认当前时间
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 更新时间，自动更新
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    -- 索引
    INDEX idx_email (email),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户表';

-- 查看表结构
DESCRIBE users;
-- 或
SHOW COLUMNS FROM users;

-- 查看表创建语句
SHOW CREATE TABLE users;

-- --- 修改表 ---

-- 添加列
ALTER TABLE users ADD COLUMN phone VARCHAR(20) AFTER email;

-- 修改列类型
ALTER TABLE users MODIFY COLUMN phone VARCHAR(30);

-- 重命名列
ALTER TABLE users CHANGE COLUMN phone mobile VARCHAR(30);

-- 删除列
ALTER TABLE users DROP COLUMN mobile;

-- 添加索引
ALTER TABLE users ADD INDEX idx_created (created_at);

-- 删除索引
ALTER TABLE users DROP INDEX idx_created;

-- 重命名表
-- RENAME TABLE users TO members;
-- RENAME TABLE members TO users;


-- ============================================================
--                    4. CRUD 操作
-- ============================================================

-- --- INSERT 插入 ---

-- 单行插入
INSERT INTO users (username, email, password_hash, age, balance)
VALUES ('alice', 'alice@example.com', 'hash123', 25, 100.00);

-- 多行插入
INSERT INTO users (username, email, password_hash, age, balance) VALUES
    ('bob', 'bob@example.com', 'hash456', 30, 200.50),
    ('charlie', 'charlie@example.com', 'hash789', 28, 150.00),
    ('diana', 'diana@example.com', 'hashabc', 35, 500.00),
    ('eve', 'eve@example.com', 'hashdef', 22, 50.00);

-- 插入或更新（主键或唯一键冲突时更新）
INSERT INTO users (username, email, password_hash)
VALUES ('alice', 'alice_new@example.com', 'newhash')
ON DUPLICATE KEY UPDATE
    email = VALUES(email),
    updated_at = CURRENT_TIMESTAMP;

-- 插入忽略（主键冲突时忽略）
INSERT IGNORE INTO users (username, email, password_hash)
VALUES ('alice', 'alice@example.com', 'hash123');

-- --- SELECT 查询 ---

-- 查询所有列
SELECT * FROM users;

-- 查询指定列
SELECT id, username, email, balance FROM users;

-- 使用别名
SELECT
    id AS 用户ID,
    username AS 用户名,
    balance AS 余额
FROM users;

-- 去重
SELECT DISTINCT status FROM users;

-- --- WHERE 条件 ---

-- 比较运算符
SELECT * FROM users WHERE age >= 25;
SELECT * FROM users WHERE status = 'active';
SELECT * FROM users WHERE balance <> 0;  -- 不等于

-- 逻辑运算符
SELECT * FROM users WHERE age >= 25 AND balance > 100;
SELECT * FROM users WHERE age < 25 OR balance > 300;
SELECT * FROM users WHERE NOT status = 'banned';

-- BETWEEN 范围
SELECT * FROM users WHERE age BETWEEN 25 AND 35;
SELECT * FROM users WHERE created_at BETWEEN '2024-01-01' AND '2024-12-31';

-- IN 列表
SELECT * FROM users WHERE status IN ('active', 'inactive');
SELECT * FROM users WHERE id IN (1, 3, 5);

-- LIKE 模糊匹配
SELECT * FROM users WHERE username LIKE 'a%';      -- 以 a 开头
SELECT * FROM users WHERE email LIKE '%@example.com';  -- 以 @example.com 结尾
SELECT * FROM users WHERE username LIKE '%li%';    -- 包含 li
SELECT * FROM users WHERE username LIKE '_ob';     -- _ 匹配单个字符

-- NULL 判断
SELECT * FROM users WHERE age IS NULL;
SELECT * FROM users WHERE age IS NOT NULL;

-- --- ORDER BY 排序 ---

-- 升序（默认）
SELECT * FROM users ORDER BY age ASC;

-- 降序
SELECT * FROM users ORDER BY balance DESC;

-- 多列排序
SELECT * FROM users ORDER BY status ASC, balance DESC;

-- --- LIMIT 分页 ---

-- 获取前 N 条
SELECT * FROM users LIMIT 3;

-- 分页：LIMIT offset, count
SELECT * FROM users LIMIT 0, 3;   -- 第1页，每页3条
SELECT * FROM users LIMIT 3, 3;   -- 第2页
SELECT * FROM users LIMIT 6, 3;   -- 第3页

-- 或使用 OFFSET
SELECT * FROM users LIMIT 3 OFFSET 0;

-- --- UPDATE 更新 ---

-- 更新单条
UPDATE users SET balance = balance + 50 WHERE id = 1;

-- 更新多列
UPDATE users
SET
    age = 26,
    status = 'active',
    balance = 200.00
WHERE username = 'alice';

-- 批量更新
UPDATE users SET balance = balance * 1.1 WHERE status = 'active';

-- 使用 CASE 条件更新
UPDATE users SET status = CASE
    WHEN balance >= 500 THEN 'active'
    WHEN balance >= 100 THEN 'inactive'
    ELSE 'inactive'
END;

-- --- DELETE 删除 ---

-- 删除指定记录
DELETE FROM users WHERE id = 5;

-- 删除多条
DELETE FROM users WHERE status = 'banned';

-- 删除所有（保留表结构）
-- DELETE FROM users;

-- 清空表（更快，重置自增）
-- TRUNCATE TABLE users;


-- ============================================================
--                    5. 聚合函数
-- ============================================================

-- 计数
SELECT COUNT(*) AS total_users FROM users;
SELECT COUNT(age) AS users_with_age FROM users;  -- 不计算 NULL
SELECT COUNT(DISTINCT status) AS status_count FROM users;

-- 求和
SELECT SUM(balance) AS total_balance FROM users;

-- 平均值
SELECT AVG(age) AS average_age FROM users;
SELECT AVG(balance) AS average_balance FROM users;

-- 最大/最小值
SELECT MAX(balance) AS max_balance FROM users;
SELECT MIN(age) AS min_age FROM users;

-- 组合使用
SELECT
    COUNT(*) AS 用户数,
    SUM(balance) AS 总余额,
    AVG(balance) AS 平均余额,
    MAX(balance) AS 最高余额,
    MIN(balance) AS 最低余额
FROM users
WHERE status = 'active';


-- ============================================================
--                    6. GROUP BY 分组
-- ============================================================

-- 按状态分组统计
SELECT
    status,
    COUNT(*) AS user_count,
    SUM(balance) AS total_balance,
    AVG(balance) AS avg_balance
FROM users
GROUP BY status;

-- 多列分组
SELECT
    status,
    CASE
        WHEN age < 25 THEN '青年'
        WHEN age < 35 THEN '中年'
        ELSE '其他'
    END AS age_group,
    COUNT(*) AS count
FROM users
GROUP BY status, age_group;

-- HAVING 过滤分组结果（WHERE 过滤行，HAVING 过滤组）
SELECT
    status,
    COUNT(*) AS user_count,
    AVG(balance) AS avg_balance
FROM users
GROUP BY status
HAVING COUNT(*) >= 2 AND AVG(balance) > 100;

-- WITH ROLLUP 添加汇总行
SELECT
    COALESCE(status, '总计') AS status,
    COUNT(*) AS user_count,
    SUM(balance) AS total_balance
FROM users
GROUP BY status WITH ROLLUP;


-- ============================================================
--                    7. 字符串函数
-- ============================================================

SELECT
    -- 连接字符串
    CONCAT(username, ' <', email, '>') AS user_info,
    CONCAT_WS('-', id, username, status) AS combined,

    -- 大小写转换
    UPPER(username) AS upper_name,
    LOWER(email) AS lower_email,

    -- 截取
    LEFT(email, 5) AS left_5,
    RIGHT(email, 10) AS right_10,
    SUBSTRING(email, 1, 5) AS sub_str,

    -- 长度
    LENGTH(username) AS byte_length,
    CHAR_LENGTH(username) AS char_length,

    -- 查找
    LOCATE('@', email) AS at_position,
    INSTR(email, '@') AS at_pos,

    -- 替换
    REPLACE(email, '@example.com', '@test.com') AS new_email,

    -- 去空格
    TRIM('  hello  ') AS trimmed,
    LTRIM('  hello') AS left_trimmed,
    RTRIM('hello  ') AS right_trimmed,

    -- 填充
    LPAD(id, 5, '0') AS padded_id,
    RPAD(username, 10, '.') AS padded_name,

    -- 反转
    REVERSE(username) AS reversed
FROM users
LIMIT 1;


-- ============================================================
--                    8. 日期时间函数
-- ============================================================

SELECT
    -- 当前日期时间
    NOW() AS now,
    CURRENT_TIMESTAMP AS current_ts,
    CURDATE() AS today,
    CURTIME() AS current_time,

    -- 提取部分
    YEAR(created_at) AS year,
    MONTH(created_at) AS month,
    DAY(created_at) AS day,
    HOUR(created_at) AS hour,
    MINUTE(created_at) AS minute,
    SECOND(created_at) AS second,
    DAYOFWEEK(created_at) AS day_of_week,  -- 1=周日
    DAYOFYEAR(created_at) AS day_of_year,
    WEEK(created_at) AS week_number,

    -- 格式化
    DATE_FORMAT(created_at, '%Y年%m月%d日 %H:%i:%s') AS formatted,
    DATE_FORMAT(created_at, '%Y-%m-%d') AS date_only,

    -- 日期计算
    DATE_ADD(created_at, INTERVAL 7 DAY) AS plus_7_days,
    DATE_SUB(created_at, INTERVAL 1 MONTH) AS minus_1_month,
    DATE_ADD(created_at, INTERVAL '1:30' HOUR_MINUTE) AS plus_90_min,

    -- 日期差
    DATEDIFF(NOW(), created_at) AS days_ago,
    TIMESTAMPDIFF(HOUR, created_at, NOW()) AS hours_ago,

    -- Unix 时间戳
    UNIX_TIMESTAMP(created_at) AS unix_ts,
    FROM_UNIXTIME(1704067200) AS from_unix
FROM users
WHERE id = 1;


-- ============================================================
--                    9. 数值函数
-- ============================================================

SELECT
    -- 四舍五入
    ROUND(balance, 1) AS rounded,
    ROUND(123.456, 0) AS rounded_int,

    -- 向上/向下取整
    CEIL(balance) AS ceiling,
    FLOOR(balance) AS floored,

    -- 截断
    TRUNCATE(balance, 1) AS truncated,

    -- 绝对值
    ABS(-100) AS absolute,

    -- 取模
    MOD(balance, 100) AS remainder,

    -- 幂运算
    POW(2, 10) AS power,
    SQRT(16) AS square_root,

    -- 随机数
    RAND() AS random_0_1,
    FLOOR(RAND() * 100) AS random_0_99,

    -- 符号
    SIGN(-10) AS sign_negative,
    SIGN(10) AS sign_positive
FROM users
WHERE id = 1;


-- ============================================================
--                    10. 条件函数
-- ============================================================

SELECT
    username,
    balance,

    -- IF 函数
    IF(balance > 200, '高余额', '低余额') AS balance_level,

    -- IFNULL 空值替换
    IFNULL(age, 0) AS age_or_zero,

    -- NULLIF 相等则返回 NULL
    NULLIF(status, 'active') AS null_if_active,

    -- COALESCE 返回第一个非空值
    COALESCE(age, 0) AS coalesced_age,

    -- CASE WHEN（简单形式）
    CASE status
        WHEN 'active' THEN '活跃'
        WHEN 'inactive' THEN '不活跃'
        WHEN 'banned' THEN '已封禁'
        ELSE '未知'
    END AS status_cn,

    -- CASE WHEN（搜索形式）
    CASE
        WHEN balance >= 500 THEN 'VIP'
        WHEN balance >= 200 THEN '高级'
        WHEN balance >= 100 THEN '普通'
        ELSE '新用户'
    END AS user_level
FROM users;


-- ============================================================
--                    总结
-- ============================================================

/*
MySQL 基础知识点：

数据库操作：
- CREATE/DROP DATABASE
- USE database_name
- SHOW DATABASES

表操作：
- CREATE TABLE（数据类型、约束、索引）
- ALTER TABLE（ADD/MODIFY/DROP COLUMN）
- DROP/TRUNCATE TABLE

CRUD：
- INSERT：单行、多行、ON DUPLICATE KEY UPDATE
- SELECT：列选择、WHERE、ORDER BY、LIMIT
- UPDATE：条件更新、批量更新
- DELETE：条件删除

条件与运算：
- 比较：=, <>, <, >, <=, >=
- 逻辑：AND, OR, NOT
- 范围：BETWEEN, IN
- 模糊：LIKE（% 和 _）
- 空值：IS NULL, IS NOT NULL

聚合与分组：
- 聚合函数：COUNT, SUM, AVG, MAX, MIN
- GROUP BY 分组
- HAVING 过滤组
- WITH ROLLUP 汇总

常用函数：
- 字符串：CONCAT, SUBSTRING, REPLACE, TRIM
- 日期：NOW, DATE_FORMAT, DATE_ADD, DATEDIFF
- 数值：ROUND, CEIL, FLOOR, RAND
- 条件：IF, IFNULL, COALESCE, CASE
*/

```
