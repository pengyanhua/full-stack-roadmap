-- ============================================================
--                    MySQL 实战示例
-- ============================================================
-- 本文件包含常见业务场景的 SQL 实现。
-- ============================================================

USE learn_mysql;

-- ============================================================
--                    1. 电商订单系统
-- ============================================================

-- --- 创建表结构 ---

-- 商品表
CREATE TABLE IF NOT EXISTS shop_products (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    category_id INT UNSIGNED,
    price DECIMAL(10, 2) NOT NULL,
    stock INT UNSIGNED DEFAULT 0,
    status ENUM('on_sale', 'off_sale', 'deleted') DEFAULT 'on_sale',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_category (category_id),
    INDEX idx_status (status)
) ENGINE=InnoDB;

-- 用户表
CREATE TABLE IF NOT EXISTS shop_users (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL,
    phone VARCHAR(20),
    balance DECIMAL(10, 2) DEFAULT 0.00,
    vip_level TINYINT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 订单表
CREATE TABLE IF NOT EXISTS shop_orders (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    order_no VARCHAR(32) NOT NULL UNIQUE,
    user_id INT UNSIGNED NOT NULL,
    total_amount DECIMAL(10, 2) NOT NULL,
    discount_amount DECIMAL(10, 2) DEFAULT 0.00,
    pay_amount DECIMAL(10, 2) NOT NULL,
    status ENUM('pending', 'paid', 'shipped', 'completed', 'cancelled', 'refunded') DEFAULT 'pending',
    paid_at TIMESTAMP NULL,
    shipped_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user (user_id),
    INDEX idx_status (status),
    INDEX idx_created (created_at)
) ENGINE=InnoDB;

-- 订单明细表
CREATE TABLE IF NOT EXISTS shop_order_items (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    order_id INT UNSIGNED NOT NULL,
    product_id INT UNSIGNED NOT NULL,
    product_name VARCHAR(200) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    quantity INT UNSIGNED NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    INDEX idx_order (order_id),
    FOREIGN KEY (order_id) REFERENCES shop_orders(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- 插入测试数据
INSERT INTO shop_products (name, category_id, price, stock) VALUES
    ('iPhone 15', 1, 5999.00, 100),
    ('MacBook Pro', 2, 14999.00, 50),
    ('AirPods Pro', 3, 1899.00, 200),
    ('iPad Air', 2, 4599.00, 80),
    ('Apple Watch', 3, 2999.00, 150);

INSERT INTO shop_users (username, email, phone, balance, vip_level) VALUES
    ('user1', 'user1@example.com', '13800001111', 10000.00, 3),
    ('user2', 'user2@example.com', '13800002222', 5000.00, 2),
    ('user3', 'user3@example.com', '13800003333', 2000.00, 1);

-- --- 业务查询示例 ---

-- 1. 生成订单号
SELECT CONCAT(DATE_FORMAT(NOW(), '%Y%m%d%H%i%s'), LPAD(FLOOR(RAND() * 1000000), 6, '0')) AS order_no;

-- 2. 创建订单存储过程
DELIMITER //

CREATE PROCEDURE create_order(
    IN p_user_id INT,
    IN p_product_id INT,
    IN p_quantity INT,
    OUT p_order_no VARCHAR(32),
    OUT p_result VARCHAR(100)
)
BEGIN
    DECLARE v_price DECIMAL(10, 2);
    DECLARE v_stock INT;
    DECLARE v_product_name VARCHAR(200);
    DECLARE v_total DECIMAL(10, 2);
    DECLARE v_order_id INT;

    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        SET p_result = '订单创建失败';
    END;

    START TRANSACTION;

    -- 检查库存（加锁）
    SELECT price, stock, name INTO v_price, v_stock, v_product_name
    FROM shop_products
    WHERE id = p_product_id
    FOR UPDATE;

    IF v_stock < p_quantity THEN
        ROLLBACK;
        SET p_result = '库存不足';
    ELSE
        -- 生成订单号
        SET p_order_no = CONCAT(DATE_FORMAT(NOW(), '%Y%m%d%H%i%s'),
                                LPAD(FLOOR(RAND() * 1000000), 6, '0'));

        -- 计算总价
        SET v_total = v_price * p_quantity;

        -- 创建订单
        INSERT INTO shop_orders (order_no, user_id, total_amount, pay_amount)
        VALUES (p_order_no, p_user_id, v_total, v_total);

        SET v_order_id = LAST_INSERT_ID();

        -- 创建订单明细
        INSERT INTO shop_order_items (order_id, product_id, product_name, price, quantity, amount)
        VALUES (v_order_id, p_product_id, v_product_name, v_price, p_quantity, v_total);

        -- 扣减库存
        UPDATE shop_products SET stock = stock - p_quantity WHERE id = p_product_id;

        COMMIT;
        SET p_result = '订单创建成功';
    END IF;
END//

DELIMITER ;

-- 调用创建订单
CALL create_order(1, 1, 2, @order_no, @result);
SELECT @order_no, @result;

-- 3. 订单统计报表
SELECT
    DATE(created_at) AS order_date,
    COUNT(*) AS order_count,
    COUNT(DISTINCT user_id) AS user_count,
    SUM(pay_amount) AS total_sales,
    AVG(pay_amount) AS avg_order_amount
FROM shop_orders
WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
GROUP BY DATE(created_at)
ORDER BY order_date DESC;

-- 4. 用户消费排行
SELECT
    u.username,
    u.vip_level,
    COUNT(o.id) AS order_count,
    SUM(o.pay_amount) AS total_spent,
    AVG(o.pay_amount) AS avg_spent
FROM shop_users u
LEFT JOIN shop_orders o ON u.id = o.user_id AND o.status IN ('paid', 'completed')
GROUP BY u.id, u.username, u.vip_level
ORDER BY total_spent DESC
LIMIT 10;

-- 5. 商品销售排行
SELECT
    p.name AS product_name,
    p.price,
    COALESCE(SUM(oi.quantity), 0) AS total_sold,
    COALESCE(SUM(oi.amount), 0) AS total_revenue
FROM shop_products p
LEFT JOIN shop_order_items oi ON p.id = oi.product_id
LEFT JOIN shop_orders o ON oi.order_id = o.id AND o.status IN ('paid', 'completed')
GROUP BY p.id, p.name, p.price
ORDER BY total_sold DESC;


-- ============================================================
--                    2. 用户签到系统
-- ============================================================

CREATE TABLE IF NOT EXISTS user_checkins (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNSIGNED NOT NULL,
    checkin_date DATE NOT NULL,
    points_earned INT DEFAULT 10,
    consecutive_days INT DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_user_date (user_id, checkin_date),
    INDEX idx_user (user_id)
) ENGINE=InnoDB;

-- 签到存储过程
DELIMITER //

CREATE PROCEDURE user_checkin(
    IN p_user_id INT,
    OUT p_points INT,
    OUT p_consecutive INT,
    OUT p_message VARCHAR(100)
)
BEGIN
    DECLARE v_last_date DATE;
    DECLARE v_last_consecutive INT;
    DECLARE v_today DATE DEFAULT CURDATE();

    -- 检查今日是否已签到
    IF EXISTS (SELECT 1 FROM user_checkins WHERE user_id = p_user_id AND checkin_date = v_today) THEN
        SET p_message = '今日已签到';
        SELECT points_earned, consecutive_days INTO p_points, p_consecutive
        FROM user_checkins WHERE user_id = p_user_id AND checkin_date = v_today;
    ELSE
        -- 获取上次签到信息
        SELECT checkin_date, consecutive_days INTO v_last_date, v_last_consecutive
        FROM user_checkins
        WHERE user_id = p_user_id
        ORDER BY checkin_date DESC
        LIMIT 1;

        -- 计算连续天数
        IF v_last_date = DATE_SUB(v_today, INTERVAL 1 DAY) THEN
            SET p_consecutive = v_last_consecutive + 1;
        ELSE
            SET p_consecutive = 1;
        END IF;

        -- 计算积分（连续签到奖励）
        SET p_points = 10 + LEAST(p_consecutive - 1, 7) * 5;

        -- 插入签到记录
        INSERT INTO user_checkins (user_id, checkin_date, points_earned, consecutive_days)
        VALUES (p_user_id, v_today, p_points, p_consecutive);

        SET p_message = CONCAT('签到成功，获得 ', p_points, ' 积分');
    END IF;
END//

DELIMITER ;

-- 签到
CALL user_checkin(1, @points, @consecutive, @msg);
SELECT @points AS 积分, @consecutive AS 连续天数, @msg AS 消息;

-- 签到排行榜
SELECT
    u.username,
    COUNT(*) AS total_checkins,
    SUM(c.points_earned) AS total_points,
    MAX(c.consecutive_days) AS max_consecutive
FROM shop_users u
JOIN user_checkins c ON u.id = c.user_id
WHERE c.checkin_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
GROUP BY u.id, u.username
ORDER BY total_points DESC
LIMIT 10;


-- ============================================================
--                    3. 消息队列表
-- ============================================================

CREATE TABLE IF NOT EXISTS message_queue (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    queue_name VARCHAR(50) NOT NULL,
    payload JSON NOT NULL,
    status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending',
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 3,
    scheduled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_queue_status (queue_name, status, scheduled_at),
    INDEX idx_scheduled (scheduled_at)
) ENGINE=InnoDB;

-- 添加消息
INSERT INTO message_queue (queue_name, payload, scheduled_at)
VALUES ('email', '{"to": "user@example.com", "subject": "Welcome", "body": "Hello!"}', NOW());

-- 获取并锁定消息（消费者）
DELIMITER //

CREATE PROCEDURE consume_message(
    IN p_queue_name VARCHAR(50),
    OUT p_message_id BIGINT,
    OUT p_payload JSON
)
BEGIN
    DECLARE v_id BIGINT;

    -- 查找并锁定一条消息
    SELECT id INTO v_id
    FROM message_queue
    WHERE queue_name = p_queue_name
      AND status = 'pending'
      AND scheduled_at <= NOW()
      AND retry_count < max_retries
    ORDER BY scheduled_at
    LIMIT 1
    FOR UPDATE SKIP LOCKED;

    IF v_id IS NOT NULL THEN
        -- 更新状态
        UPDATE message_queue
        SET status = 'processing', started_at = NOW()
        WHERE id = v_id;

        -- 返回消息
        SELECT id, payload INTO p_message_id, p_payload
        FROM message_queue WHERE id = v_id;
    ELSE
        SET p_message_id = NULL;
        SET p_payload = NULL;
    END IF;
END//

-- 完成消息
CREATE PROCEDURE complete_message(IN p_message_id BIGINT)
BEGIN
    UPDATE message_queue
    SET status = 'completed', completed_at = NOW()
    WHERE id = p_message_id;
END//

-- 失败重试
CREATE PROCEDURE fail_message(IN p_message_id BIGINT, IN p_error TEXT)
BEGIN
    UPDATE message_queue
    SET
        status = IF(retry_count + 1 >= max_retries, 'failed', 'pending'),
        retry_count = retry_count + 1,
        error_message = p_error,
        scheduled_at = DATE_ADD(NOW(), INTERVAL POW(2, retry_count) MINUTE)
    WHERE id = p_message_id;
END//

DELIMITER ;


-- ============================================================
--                    4. 树形结构（邻接表）
-- ============================================================

CREATE TABLE IF NOT EXISTS categories (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    parent_id INT UNSIGNED,
    level INT DEFAULT 1,
    path VARCHAR(500),
    sort_order INT DEFAULT 0,
    INDEX idx_parent (parent_id),
    INDEX idx_path (path)
) ENGINE=InnoDB;

INSERT INTO categories (name, parent_id, level, path) VALUES
    ('电子产品', NULL, 1, '/1/'),
    ('手机', 1, 2, '/1/2/'),
    ('电脑', 1, 2, '/1/3/'),
    ('智能手机', 2, 3, '/1/2/4/'),
    ('功能手机', 2, 3, '/1/2/5/'),
    ('笔记本', 3, 3, '/1/3/6/'),
    ('台式机', 3, 3, '/1/3/7/');

-- 查询所有子分类（使用路径）
SELECT * FROM categories WHERE path LIKE '/1/2/%';

-- 查询所有父分类
SELECT * FROM categories
WHERE FIND_IN_SET(id, (
    SELECT REPLACE(TRIM(BOTH '/' FROM path), '/', ',')
    FROM categories WHERE id = 4
));

-- 递归查询（MySQL 8.0+）
WITH RECURSIVE category_tree AS (
    SELECT id, name, parent_id, level, path
    FROM categories
    WHERE parent_id IS NULL

    UNION ALL

    SELECT c.id, c.name, c.parent_id, c.level, c.path
    FROM categories c
    JOIN category_tree ct ON c.parent_id = ct.id
)
SELECT * FROM category_tree;


-- ============================================================
--                    5. 数据统计与报表
-- ============================================================

-- 按时间维度统计
SELECT
    DATE_FORMAT(created_at, '%Y-%m') AS month,
    COUNT(*) AS order_count,
    SUM(pay_amount) AS revenue,
    COUNT(DISTINCT user_id) AS unique_users
FROM shop_orders
WHERE status IN ('paid', 'completed')
GROUP BY DATE_FORMAT(created_at, '%Y-%m')
ORDER BY month;

-- 同比环比分析
WITH monthly_stats AS (
    SELECT
        DATE_FORMAT(created_at, '%Y-%m') AS month,
        SUM(pay_amount) AS revenue
    FROM shop_orders
    WHERE status IN ('paid', 'completed')
    GROUP BY DATE_FORMAT(created_at, '%Y-%m')
)
SELECT
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) AS last_month_revenue,
    revenue - LAG(revenue) OVER (ORDER BY month) AS mom_change,
    ROUND((revenue - LAG(revenue) OVER (ORDER BY month)) /
          NULLIF(LAG(revenue) OVER (ORDER BY month), 0) * 100, 2) AS mom_rate
FROM monthly_stats;

-- 用户留存分析
WITH first_order AS (
    SELECT user_id, MIN(DATE(created_at)) AS first_date
    FROM shop_orders
    GROUP BY user_id
),
retention AS (
    SELECT
        fo.first_date,
        DATEDIFF(DATE(o.created_at), fo.first_date) AS days_after,
        COUNT(DISTINCT o.user_id) AS users
    FROM first_order fo
    JOIN shop_orders o ON fo.user_id = o.user_id
    GROUP BY fo.first_date, days_after
)
SELECT
    first_date AS 注册日期,
    MAX(CASE WHEN days_after = 0 THEN users END) AS 当日,
    MAX(CASE WHEN days_after = 1 THEN users END) AS 次日,
    MAX(CASE WHEN days_after = 7 THEN users END) AS 七日,
    MAX(CASE WHEN days_after = 30 THEN users END) AS 三十日
FROM retention
GROUP BY first_date
ORDER BY first_date;


-- ============================================================
--                    6. 数据清洗与迁移
-- ============================================================

-- 批量更新（避免锁表过久）
DELIMITER //

CREATE PROCEDURE batch_update(
    IN p_batch_size INT,
    OUT p_total_updated INT
)
BEGIN
    DECLARE v_affected INT DEFAULT 1;
    SET p_total_updated = 0;

    WHILE v_affected > 0 DO
        UPDATE shop_products
        SET status = 'off_sale'
        WHERE status = 'on_sale' AND stock = 0
        LIMIT p_batch_size;

        SET v_affected = ROW_COUNT();
        SET p_total_updated = p_total_updated + v_affected;

        -- 短暂休息，避免锁竞争
        DO SLEEP(0.1);
    END WHILE;
END//

DELIMITER ;

-- 数据归档
INSERT INTO shop_orders_archive
SELECT * FROM shop_orders
WHERE created_at < DATE_SUB(CURDATE(), INTERVAL 1 YEAR)
  AND status IN ('completed', 'cancelled', 'refunded');

DELETE FROM shop_orders
WHERE created_at < DATE_SUB(CURDATE(), INTERVAL 1 YEAR)
  AND status IN ('completed', 'cancelled', 'refunded')
LIMIT 10000;


-- ============================================================
--                    总结
-- ============================================================

/*
实战技巧：

订单系统：
- 使用事务保证数据一致性
- 库存扣减加锁防止超卖
- 订单号生成策略

签到系统：
- 唯一索引防止重复签到
- 连续签到计算逻辑

消息队列：
- SKIP LOCKED 实现并发消费
- 指数退避重试策略

树形结构：
- 路径枚举法快速查询
- 递归 CTE（MySQL 8.0+）

数据统计：
- 窗口函数计算同比环比
- 用户留存分析

批量操作：
- 分批处理避免锁表
- 数据归档策略
*/
