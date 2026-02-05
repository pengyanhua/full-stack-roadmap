# MySQL 优化

## 一、索引优化

### 1. 索引类型与选择

```
┌─────────────────────────────────────────────────────────────────┐
│                      MySQL 索引类型                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   B+ Tree 索引 (默认)                                           │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   适用场景:                                               │  │
│   │   • 等值查询 (=, IN)                                     │  │
│   │   • 范围查询 (>, <, BETWEEN)                             │  │
│   │   • 排序 (ORDER BY)                                      │  │
│   │   • 分组 (GROUP BY)                                      │  │
│   │   • 前缀匹配 (LIKE 'abc%')                               │  │
│   │                                                          │  │
│   │   不适用:                                                 │  │
│   │   • 后缀匹配 (LIKE '%abc')                               │  │
│   │   • 函数计算后的列                                       │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   Hash 索引 (Memory引擎)                                        │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │   适用: 等值查询 (=)                                      │  │
│   │   不适用: 范围查询、排序                                  │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   全文索引 (FULLTEXT)                                           │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │   适用: 文本搜索 (MATCH AGAINST)                         │  │
│   │   建议: 大规模搜索用 Elasticsearch 替代                   │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 索引设计原则

```sql
-- ❌ 坑点 1: 在区分度低的列上建索引
CREATE INDEX idx_status ON orders(status);  -- 状态只有几种值
-- 查询时可能全表扫描比索引更快

-- ✅ 正确: 在区分度高的列上建索引
CREATE INDEX idx_order_no ON orders(order_no);  -- 订单号唯一性高


-- ❌ 坑点 2: 索引列顺序错误
-- 查询: WHERE user_id = ? AND created_at > ?
CREATE INDEX idx_time_user ON orders(created_at, user_id);  -- 顺序错误

-- ✅ 正确: 等值查询列在前，范围查询列在后
CREATE INDEX idx_user_time ON orders(user_id, created_at);


-- ❌ 坑点 3: 索引列参与计算
SELECT * FROM users WHERE YEAR(created_at) = 2024;  -- 无法使用索引

-- ✅ 正确: 改写为范围查询
SELECT * FROM users
WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01';


-- ❌ 坑点 4: 隐式类型转换
-- user_id 是 varchar 类型
SELECT * FROM users WHERE user_id = 123;  -- 数字会导致全表扫描

-- ✅ 正确: 类型一致
SELECT * FROM users WHERE user_id = '123';


-- 联合索引最左前缀原则
CREATE INDEX idx_abc ON t(a, b, c);
-- ✅ 可以使用索引
SELECT * FROM t WHERE a = 1;
SELECT * FROM t WHERE a = 1 AND b = 2;
SELECT * FROM t WHERE a = 1 AND b = 2 AND c = 3;

-- ❌ 无法使用索引
SELECT * FROM t WHERE b = 2;           -- 缺少最左列 a
SELECT * FROM t WHERE b = 2 AND c = 3; -- 缺少最左列 a

-- 部分使用索引
SELECT * FROM t WHERE a = 1 AND c = 3; -- 只用到 a，c 无法用到
```

### 3. 覆盖索引

```sql
-- 覆盖索引: 查询的列都在索引中，无需回表
CREATE INDEX idx_user_name_email ON users(user_id, name, email);

-- ✅ 覆盖索引，不需要回表
SELECT user_id, name, email FROM users WHERE user_id = 123;

-- ❌ 需要回表
SELECT * FROM users WHERE user_id = 123;  -- 需要查询其他列


-- 使用 EXPLAIN 查看
EXPLAIN SELECT user_id, name, email FROM users WHERE user_id = 123;
-- Extra: Using index  表示使用了覆盖索引
```

### 4. 索引失效场景

```sql
-- 1. 使用 OR (部分索引列没有索引)
SELECT * FROM users WHERE name = 'test' OR age = 20;
-- 如果 age 没有索引，整个查询不走索引

-- 2. 使用 NOT, !=, <>
SELECT * FROM users WHERE status != 'deleted';
-- 优化: 改为 IN 条件

-- 3. IS NULL / IS NOT NULL (视情况)
SELECT * FROM users WHERE deleted_at IS NULL;
-- 如果 NULL 值很多，可能不走索引

-- 4. LIKE 前缀通配符
SELECT * FROM users WHERE name LIKE '%张';  -- 不走索引
SELECT * FROM users WHERE name LIKE '张%';  -- 走索引

-- 5. 索引列参与运算
SELECT * FROM users WHERE id + 1 = 10;     -- 不走索引
SELECT * FROM users WHERE id = 9;          -- 走索引
```

---

## 二、SQL 优化

### 1. 慢查询分析

```sql
-- 开启慢查询日志
SET GLOBAL slow_query_log = ON;
SET GLOBAL long_query_time = 1;  -- 超过1秒记录
SET GLOBAL slow_query_log_file = '/var/log/mysql/slow.log';

-- 使用 EXPLAIN 分析
EXPLAIN SELECT * FROM orders WHERE user_id = 123;

/*
+----+-------------+--------+------+---------------+------+---------+------+------+-------------+
| id | select_type | table  | type | possible_keys | key  | key_len | ref  | rows | Extra       |
+----+-------------+--------+------+---------------+------+---------+------+------+-------------+
| 1  | SIMPLE      | orders | ref  | idx_user_id   | ...  | 4       | ...  | 100  | Using where |
+----+-------------+--------+------+---------------+------+---------+------+------+-------------+

type 类型 (性能从好到差):
- system/const: 主键或唯一索引等值查询
- eq_ref: 关联查询使用主键或唯一索引
- ref: 使用普通索引
- range: 范围扫描
- index: 索引全扫描
- ALL: 全表扫描 (最差，需要优化)
*/

-- EXPLAIN ANALYZE (MySQL 8.0+)
EXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = 123;
-- 显示实际执行时间
```

### 2. 常见优化技巧

```sql
-- 1. 分页优化
-- ❌ 深分页性能差
SELECT * FROM orders ORDER BY id LIMIT 1000000, 20;

-- ✅ 使用游标分页
SELECT * FROM orders WHERE id > 1000000 ORDER BY id LIMIT 20;

-- ✅ 延迟关联
SELECT * FROM orders o
INNER JOIN (SELECT id FROM orders ORDER BY id LIMIT 1000000, 20) t
ON o.id = t.id;


-- 2. COUNT 优化
-- ❌ 全表统计慢
SELECT COUNT(*) FROM orders;

-- ✅ 使用近似值或缓存
-- 方案1: 使用 Redis 缓存计数
-- 方案2: 单独维护计数表
-- 方案3: 使用 information_schema (不精确)
SELECT TABLE_ROWS FROM information_schema.TABLES
WHERE TABLE_NAME = 'orders';


-- 3. JOIN 优化
-- ❌ 大表 JOIN
SELECT * FROM orders o LEFT JOIN users u ON o.user_id = u.id;

-- ✅ 小表驱动大表
SELECT * FROM users u  -- 小表在前
LEFT JOIN orders o ON u.id = o.user_id;

-- ✅ 确保 JOIN 列有索引
CREATE INDEX idx_user_id ON orders(user_id);


-- 4. 批量插入
-- ❌ 逐条插入
INSERT INTO users (name) VALUES ('a');
INSERT INTO users (name) VALUES ('b');
INSERT INTO users (name) VALUES ('c');

-- ✅ 批量插入
INSERT INTO users (name) VALUES ('a'), ('b'), ('c');

-- ✅ 使用 LOAD DATA INFILE (最快)
LOAD DATA INFILE '/tmp/data.csv' INTO TABLE users;


-- 5. 批量更新
-- ❌ 逐条更新
UPDATE users SET status = 'active' WHERE id = 1;
UPDATE users SET status = 'active' WHERE id = 2;

-- ✅ 批量更新
UPDATE users SET status = 'active' WHERE id IN (1, 2, 3);

-- ✅ CASE WHEN 批量更新不同值
UPDATE users SET status = CASE id
    WHEN 1 THEN 'active'
    WHEN 2 THEN 'inactive'
    WHEN 3 THEN 'deleted'
END WHERE id IN (1, 2, 3);
```

---

## 三、表结构设计

### 1. 数据类型选择

```sql
-- 整数类型
TINYINT      -- 1字节, -128~127 (状态、类型)
SMALLINT     -- 2字节
INT          -- 4字节 (主键)
BIGINT       -- 8字节 (超大ID、时间戳)

-- 字符串类型
CHAR(n)      -- 定长，适合固定长度 (如手机号)
VARCHAR(n)   -- 变长，适合可变长度 (如姓名)
TEXT         -- 大文本，避免在索引中使用

-- 时间类型
DATE         -- 日期
DATETIME     -- 日期时间，不受时区影响
TIMESTAMP    -- 时间戳，自动转换时区，最大2038年

-- ✅ 最佳实践
CREATE TABLE users (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    phone CHAR(11),
    status TINYINT NOT NULL DEFAULT 0,
    balance DECIMAL(10,2) NOT NULL DEFAULT 0.00,  -- 金额用 DECIMAL
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 2. 范式与反范式

```sql
-- 第三范式 (规范化)
-- 优点: 数据不冗余，更新简单
-- 缺点: 查询需要 JOIN

-- 用户表
CREATE TABLE users (
    id BIGINT PRIMARY KEY,
    name VARCHAR(50)
);

-- 订单表 (只存 user_id)
CREATE TABLE orders (
    id BIGINT PRIMARY KEY,
    user_id BIGINT,
    amount DECIMAL(10,2)
);

-- 查询需要 JOIN
SELECT o.*, u.name FROM orders o JOIN users u ON o.user_id = u.id;


-- 反范式 (冗余)
-- 优点: 查询快，无需 JOIN
-- 缺点: 数据冗余，更新复杂

CREATE TABLE orders (
    id BIGINT PRIMARY KEY,
    user_id BIGINT,
    user_name VARCHAR(50),  -- 冗余用户名
    amount DECIMAL(10,2)
);

-- 查询无需 JOIN
SELECT * FROM orders WHERE user_id = 123;


-- ✅ 实践建议:
-- 1. 读多写少场景，适度冗余
-- 2. 冗余字段应该是稳定的 (如用户名很少改)
-- 3. 冗余后要考虑数据同步 (异步更新或双写)
```

---

## 四、锁与事务

### 1. 锁类型

```
┌─────────────────────────────────────────────────────────────────┐
│                     MySQL 锁分类                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   按粒度分:                                                      │
│   ─────────────────────────────────────────────────────────    │
│   表锁: LOCK TABLES ... (MyISAM 默认，开销小，并发低)           │
│   行锁: InnoDB 默认 (开销大，并发高)                             │
│   间隙锁: 锁定索引间隙，防止幻读                                 │
│                                                                 │
│   按类型分:                                                      │
│   ─────────────────────────────────────────────────────────    │
│   共享锁 (S锁): SELECT ... LOCK IN SHARE MODE                   │
│   排他锁 (X锁): SELECT ... FOR UPDATE                           │
│                                                                 │
│   意向锁:                                                        │
│   ─────────────────────────────────────────────────────────    │
│   IS: 意向共享锁 (表级)                                         │
│   IX: 意向排他锁 (表级)                                         │
│   用于快速判断表中是否有行锁                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 死锁处理

```sql
-- 死锁示例
-- 事务 A                    -- 事务 B
BEGIN;                       BEGIN;
UPDATE t SET ... WHERE id=1; -- 锁住 id=1
                             UPDATE t SET ... WHERE id=2; -- 锁住 id=2
UPDATE t SET ... WHERE id=2; -- 等待 id=2 的锁
                             UPDATE t SET ... WHERE id=1; -- 等待 id=1 的锁
-- 死锁!


-- ✅ 预防死锁
-- 1. 按固定顺序访问表和行
-- 2. 大事务拆成小事务
-- 3. 降低隔离级别
-- 4. 为表添加合理的索引

-- 检查死锁
SHOW ENGINE INNODB STATUS;

-- 查看锁等待
SELECT * FROM information_schema.INNODB_LOCK_WAITS;

-- 设置锁等待超时
SET innodb_lock_wait_timeout = 10;  -- 10秒
```

### 3. 事务隔离级别

```sql
-- 查看当前隔离级别
SELECT @@transaction_isolation;

-- 设置隔离级别
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;

/*
隔离级别           脏读    不可重复读   幻读
────────────────────────────────────────────
READ UNCOMMITTED    ✓         ✓         ✓
READ COMMITTED      ✗         ✓         ✓     <- 推荐生产使用
REPEATABLE READ     ✗         ✗         ✓     <- MySQL 默认
SERIALIZABLE        ✗         ✗         ✗
*/

-- ✅ 生产建议: 使用 READ COMMITTED
-- 原因:
-- 1. 减少锁竞争
-- 2. 避免 Gap Lock 导致的死锁
-- 3. 与其他数据库行为一致
```

---

## 五、检查清单

### 索引优化检查

- [ ] 慢查询是否都有对应索引？
- [ ] 索引列顺序是否正确？
- [ ] 是否存在索引失效的情况？
- [ ] 是否有冗余索引？

### SQL 优化检查

- [ ] 是否避免了 SELECT *？
- [ ] 分页查询是否优化？
- [ ] 批量操作是否合并？
- [ ] JOIN 是否有合适的索引？

### 表设计检查

- [ ] 数据类型是否合适？
- [ ] 是否适度反范式？
- [ ] 大字段是否拆分？
- [ ] 是否有适当的约束？
