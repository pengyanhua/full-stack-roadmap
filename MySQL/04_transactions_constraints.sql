-- ============================================================
--                    MySQL 事务与约束
-- ============================================================
-- 本文件介绍 MySQL 事务处理和数据完整性约束。
-- ============================================================

USE learn_mysql;

-- ============================================================
--                    1. 事务基础
-- ============================================================

/*
事务特性 ACID：
- Atomicity（原子性）：事务中的操作要么全部成功，要么全部回滚
- Consistency（一致性）：事务前后数据库状态保持一致
- Isolation（隔离性）：并发事务之间相互隔离
- Durability（持久性）：事务提交后数据永久保存

InnoDB 支持事务，MyISAM 不支持
*/

-- 查看当前自动提交状态
SELECT @@autocommit;

-- 关闭自动提交
SET autocommit = 0;

-- 开启自动提交
SET autocommit = 1;

-- --- 基本事务操作 ---

-- 创建测试表
CREATE TABLE IF NOT EXISTS accounts (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    balance DECIMAL(10, 2) NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 清空并插入测试数据
TRUNCATE TABLE accounts;
INSERT INTO accounts (name, balance) VALUES
    ('Alice', 1000.00),
    ('Bob', 500.00),
    ('Charlie', 2000.00);

-- 开始事务
START TRANSACTION;
-- 或
BEGIN;

-- 转账操作
UPDATE accounts SET balance = balance - 200 WHERE name = 'Alice';
UPDATE accounts SET balance = balance + 200 WHERE name = 'Bob';

-- 检查结果
SELECT * FROM accounts;

-- 提交事务
COMMIT;

-- --- 回滚示例 ---
START TRANSACTION;

UPDATE accounts SET balance = balance - 500 WHERE name = 'Charlie';
UPDATE accounts SET balance = balance + 500 WHERE name = 'Alice';

-- 检查余额是否为负
SELECT * FROM accounts WHERE balance < 0;

-- 如果有问题，回滚
ROLLBACK;

-- 验证数据未变
SELECT * FROM accounts;

-- --- 保存点（Savepoint）---
START TRANSACTION;

UPDATE accounts SET balance = balance + 100 WHERE name = 'Alice';
SAVEPOINT sp1;  -- 创建保存点

UPDATE accounts SET balance = balance + 100 WHERE name = 'Bob';
SAVEPOINT sp2;

UPDATE accounts SET balance = balance + 100 WHERE name = 'Charlie';

-- 回滚到保存点 sp2
ROLLBACK TO SAVEPOINT sp2;

-- 只有 Alice 和 Bob 的更新保留
COMMIT;

SELECT * FROM accounts;


-- ============================================================
--                    2. 隔离级别
-- ============================================================

/*
并发问题：
1. 脏读（Dirty Read）：读取到其他事务未提交的数据
2. 不可重复读（Non-repeatable Read）：同一事务中多次读取结果不同
3. 幻读（Phantom Read）：同一事务中多次查询返回不同的行数

隔离级别（从低到高）：
1. READ UNCOMMITTED：可能脏读、不可重复读、幻读
2. READ COMMITTED：防止脏读
3. REPEATABLE READ：防止脏读、不可重复读（MySQL 默认）
4. SERIALIZABLE：防止所有问题，但性能最差
*/

-- 查看当前隔离级别
SELECT @@transaction_isolation;
-- MySQL 5.7 及之前
-- SELECT @@tx_isolation;

-- 设置会话隔离级别
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;
SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SET SESSION TRANSACTION ISOLATION LEVEL SERIALIZABLE;
SET SESSION TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;

-- 设置全局隔离级别
SET GLOBAL TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- --- 演示不同隔离级别 ---

-- 【脏读演示】（需要两个会话）
-- 会话 1：READ UNCOMMITTED
SET SESSION TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
START TRANSACTION;
SELECT balance FROM accounts WHERE name = 'Alice';  -- 读取到会话2未提交的数据

-- 会话 2：
-- START TRANSACTION;
-- UPDATE accounts SET balance = 9999 WHERE name = 'Alice';
-- （不提交）

-- 【不可重复读演示】
-- 会话 1：READ COMMITTED
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;
START TRANSACTION;
SELECT balance FROM accounts WHERE name = 'Alice';  -- 第一次读取
-- 会话 2 更新并提交
SELECT balance FROM accounts WHERE name = 'Alice';  -- 第二次读取，结果不同
COMMIT;

-- 【REPEATABLE READ 防止不可重复读】
SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ;
START TRANSACTION;
SELECT balance FROM accounts WHERE name = 'Alice';  -- 第一次读取
-- 会话 2 更新并提交
SELECT balance FROM accounts WHERE name = 'Alice';  -- 第二次读取，结果相同
COMMIT;


-- ============================================================
--                    3. 约束类型
-- ============================================================

/*
MySQL 约束类型：
1. PRIMARY KEY：主键约束（唯一 + 非空）
2. UNIQUE：唯一约束
3. NOT NULL：非空约束
4. DEFAULT：默认值约束
5. CHECK：检查约束（MySQL 8.0.16+）
6. FOREIGN KEY：外键约束
*/

-- 创建带完整约束的表
CREATE TABLE IF NOT EXISTS customers (
    id INT UNSIGNED AUTO_INCREMENT,
    email VARCHAR(100) NOT NULL,
    username VARCHAR(50) NOT NULL,
    age TINYINT UNSIGNED,
    gender ENUM('M', 'F', 'O'),
    phone VARCHAR(20),
    credit_score INT DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 主键约束
    PRIMARY KEY (id),

    -- 唯一约束
    UNIQUE KEY uk_email (email),
    UNIQUE KEY uk_username (username),

    -- 检查约束（MySQL 8.0.16+）
    CONSTRAINT chk_age CHECK (age >= 0 AND age <= 150),
    CONSTRAINT chk_credit CHECK (credit_score >= 0 AND credit_score <= 1000),
    CONSTRAINT chk_status CHECK (status IN ('active', 'inactive', 'suspended'))
) ENGINE=InnoDB;

-- --- 添加约束 ---

-- 添加唯一约束
ALTER TABLE customers ADD CONSTRAINT uk_phone UNIQUE (phone);

-- 添加检查约束
ALTER TABLE customers ADD CONSTRAINT chk_gender
    CHECK (gender IN ('M', 'F', 'O'));

-- --- 删除约束 ---

-- 删除唯一约束
ALTER TABLE customers DROP INDEX uk_phone;

-- 删除检查约束
ALTER TABLE customers DROP CHECK chk_gender;

-- --- 查看约束 ---
SELECT
    CONSTRAINT_NAME,
    CONSTRAINT_TYPE
FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
WHERE TABLE_SCHEMA = 'learn_mysql'
AND TABLE_NAME = 'customers';

-- 查看检查约束详情
SELECT * FROM INFORMATION_SCHEMA.CHECK_CONSTRAINTS
WHERE CONSTRAINT_SCHEMA = 'learn_mysql';


-- ============================================================
--                    4. 外键约束
-- ============================================================

-- 创建父表
CREATE TABLE IF NOT EXISTS categories (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    parent_id INT UNSIGNED,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 自引用外键
    FOREIGN KEY (parent_id) REFERENCES categories(id)
        ON DELETE SET NULL
        ON UPDATE CASCADE
) ENGINE=InnoDB;

-- 创建子表
CREATE TABLE IF NOT EXISTS items (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    category_id INT UNSIGNED NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 外键约束
    CONSTRAINT fk_items_category
    FOREIGN KEY (category_id) REFERENCES categories(id)
        ON DELETE RESTRICT      -- 删除时：拒绝
        ON UPDATE CASCADE       -- 更新时：级联更新
) ENGINE=InnoDB;

/*
外键动作选项：
- RESTRICT：拒绝操作（默认）
- CASCADE：级联操作
- SET NULL：设置为 NULL
- SET DEFAULT：设置为默认值
- NO ACTION：等同于 RESTRICT
*/

-- 插入测试数据
INSERT INTO categories (name, parent_id) VALUES
    ('电子产品', NULL),
    ('手机', 1),
    ('电脑', 1),
    ('服装', NULL);

INSERT INTO items (name, category_id, price) VALUES
    ('iPhone 15', 2, 5999.00),
    ('MacBook Pro', 3, 14999.00);

-- 测试外键约束
-- 删除有子记录的分类（失败）
-- DELETE FROM categories WHERE id = 2;

-- 更新分类 ID（子表自动更新）
UPDATE categories SET id = 20 WHERE id = 2;
SELECT * FROM items;  -- category_id 变为 20

-- --- 添加外键 ---
ALTER TABLE items
ADD CONSTRAINT fk_items_category_new
FOREIGN KEY (category_id) REFERENCES categories(id);

-- --- 删除外键 ---
ALTER TABLE items DROP FOREIGN KEY fk_items_category;

-- --- 临时禁用外键检查 ---
SET FOREIGN_KEY_CHECKS = 0;
-- 执行批量操作...
SET FOREIGN_KEY_CHECKS = 1;


-- ============================================================
--                    5. 数据完整性
-- ============================================================

-- --- 使用触发器增强完整性 ---

DELIMITER //

-- 插入前检查
CREATE TRIGGER trg_accounts_before_insert
BEFORE INSERT ON accounts
FOR EACH ROW
BEGIN
    IF NEW.balance < 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = '余额不能为负数';
    END IF;
END//

-- 更新前检查
CREATE TRIGGER trg_accounts_before_update
BEFORE UPDATE ON accounts
FOR EACH ROW
BEGIN
    IF NEW.balance < 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = '余额不能为负数';
    END IF;
END//

DELIMITER ;

-- 测试触发器
-- INSERT INTO accounts (name, balance) VALUES ('Test', -100);  -- 失败

-- --- 存储过程实现原子操作 ---

DELIMITER //

CREATE PROCEDURE transfer_money(
    IN from_account VARCHAR(50),
    IN to_account VARCHAR(50),
    IN amount DECIMAL(10, 2),
    OUT result VARCHAR(100)
)
BEGIN
    DECLARE from_balance DECIMAL(10, 2);
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        SET result = '转账失败：数据库错误';
    END;

    START TRANSACTION;

    -- 检查转出账户余额
    SELECT balance INTO from_balance
    FROM accounts
    WHERE name = from_account
    FOR UPDATE;  -- 加锁

    IF from_balance IS NULL THEN
        ROLLBACK;
        SET result = '转账失败：转出账户不存在';
    ELSEIF from_balance < amount THEN
        ROLLBACK;
        SET result = '转账失败：余额不足';
    ELSE
        -- 执行转账
        UPDATE accounts SET balance = balance - amount WHERE name = from_account;
        UPDATE accounts SET balance = balance + amount WHERE name = to_account;

        COMMIT;
        SET result = CONCAT('转账成功：', from_account, ' -> ', to_account, ' ', amount);
    END IF;
END//

DELIMITER ;

-- 调用存储过程
CALL transfer_money('Alice', 'Bob', 100, @result);
SELECT @result;

-- 验证结果
SELECT * FROM accounts;


-- ============================================================
--                    6. 乐观锁与悲观锁
-- ============================================================

-- --- 悲观锁 ---
-- 在查询时锁定记录，其他事务无法修改

START TRANSACTION;

-- 使用 FOR UPDATE 加排他锁
SELECT * FROM accounts WHERE name = 'Alice' FOR UPDATE;

-- 进行更新操作
UPDATE accounts SET balance = balance - 50 WHERE name = 'Alice';

COMMIT;

-- --- 乐观锁 ---
-- 使用版本号或时间戳检测冲突

-- 添加版本号列
ALTER TABLE accounts ADD COLUMN version INT DEFAULT 0;

-- 乐观锁更新（CAS 模式）
UPDATE accounts
SET balance = balance - 100, version = version + 1
WHERE name = 'Alice' AND version = 0;

-- 检查影响行数
-- 如果为 0，说明发生冲突，需要重试

-- 乐观锁示例（使用存储过程）
DELIMITER //

CREATE PROCEDURE optimistic_update(
    IN account_name VARCHAR(50),
    IN amount DECIMAL(10, 2),
    IN expected_version INT,
    OUT success BOOLEAN
)
BEGIN
    DECLARE affected_rows INT;

    UPDATE accounts
    SET balance = balance + amount, version = version + 1
    WHERE name = account_name AND version = expected_version;

    SET affected_rows = ROW_COUNT();
    SET success = (affected_rows > 0);
END//

DELIMITER ;


-- ============================================================
--                    7. 死锁处理
-- ============================================================

/*
死锁：两个或多个事务相互等待对方释放锁

InnoDB 死锁检测：
- 自动检测死锁
- 回滚代价较小的事务
- 设置超时时间

预防死锁：
1. 按固定顺序访问表和行
2. 保持事务简短
3. 使用适当的隔离级别
4. 避免大事务
*/

-- 查看死锁超时设置
SHOW VARIABLES LIKE 'innodb_lock_wait_timeout';

-- 设置锁等待超时（秒）
SET innodb_lock_wait_timeout = 10;

-- 查看最近的死锁信息
SHOW ENGINE INNODB STATUS;

-- 查看当前锁信息
SELECT * FROM performance_schema.data_locks;
SELECT * FROM performance_schema.data_lock_waits;

-- 查看正在执行的事务
SELECT * FROM information_schema.INNODB_TRX;


-- ============================================================
--                    8. 日志与恢复
-- ============================================================

/*
MySQL 日志类型：
1. Redo Log（重做日志）：保证持久性，用于崩溃恢复
2. Undo Log（回滚日志）：保证原子性，用于回滚和 MVCC
3. Binary Log（二进制日志）：用于复制和增量备份
4. Error Log（错误日志）：记录错误和警告
5. Slow Query Log（慢查询日志）：记录慢查询
6. General Log（通用日志）：记录所有查询
*/

-- 查看日志配置
SHOW VARIABLES LIKE '%log%';

-- 查看二进制日志
SHOW BINARY LOGS;
SHOW BINLOG EVENTS IN 'mysql-bin.000001' LIMIT 10;

-- 查看二进制日志内容（命令行）
-- mysqlbinlog mysql-bin.000001

-- 使用二进制日志恢复（命令行）
-- mysqlbinlog mysql-bin.000001 | mysql -u root -p


-- ============================================================
--                    总结
-- ============================================================

/*
事务操作：
- START TRANSACTION / BEGIN
- COMMIT / ROLLBACK
- SAVEPOINT / ROLLBACK TO SAVEPOINT

隔离级别：
- READ UNCOMMITTED：最低，可脏读
- READ COMMITTED：防止脏读
- REPEATABLE READ：防止不可重复读（默认）
- SERIALIZABLE：最高，性能最差

约束类型：
- PRIMARY KEY：主键
- UNIQUE：唯一
- NOT NULL：非空
- DEFAULT：默认值
- CHECK：检查（MySQL 8.0.16+）
- FOREIGN KEY：外键

外键动作：
- RESTRICT：拒绝
- CASCADE：级联
- SET NULL：置空
- NO ACTION：同 RESTRICT

锁机制：
- 悲观锁：FOR UPDATE / LOCK IN SHARE MODE
- 乐观锁：版本号/时间戳

最佳实践：
- 保持事务简短
- 选择适当的隔离级别
- 按固定顺序访问资源
- 处理死锁异常
- 使用适当的锁粒度
*/
