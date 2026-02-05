-- ============================================================
--                    PostgreSQL 高级特性
-- ============================================================
-- 本文件介绍 PostgreSQL 的高级特性。
-- ============================================================

-- \c learn_postgresql

-- ============================================================
--                    1. JSON/JSONB 操作
-- ============================================================

/*
JSON vs JSONB:
- JSON: 存储原始文本，保留格式和顺序
- JSONB: 二进制格式，更高效，支持索引
- 推荐使用 JSONB
*/

-- 创建包含 JSONB 的表
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    category VARCHAR(50),
    price NUMERIC(10, 2),
    attributes JSONB DEFAULT '{}',
    tags JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 插入 JSON 数据
INSERT INTO products (name, category, price, attributes, tags) VALUES
    ('iPhone 15', 'phone', 5999.00,
     '{"brand": "Apple", "storage": "128GB", "color": "黑色", "specs": {"screen": "6.1寸", "chip": "A16"}}',
     '["5G", "iOS", "旗舰"]'),
    ('MacBook Pro', 'laptop', 14999.00,
     '{"brand": "Apple", "storage": "512GB", "ram": "16GB", "specs": {"screen": "14寸", "chip": "M3"}}',
     '["macOS", "专业", "创意"]'),
    ('Galaxy S24', 'phone', 4999.00,
     '{"brand": "Samsung", "storage": "256GB", "color": "白色", "specs": {"screen": "6.2寸", "chip": "骁龙8"}}',
     '["5G", "Android", "旗舰"]');

-- --- JSON 操作符 ---

-- -> 获取 JSON 对象（返回 JSON）
SELECT name, attributes -> 'brand' AS brand_json FROM products;

-- ->> 获取 JSON 文本（返回 TEXT）
SELECT name, attributes ->> 'brand' AS brand FROM products;

-- 嵌套访问
SELECT name, attributes -> 'specs' ->> 'screen' AS screen FROM products;

-- #> 路径访问（返回 JSON）
SELECT name, attributes #> '{specs, chip}' AS chip FROM products;

-- #>> 路径访问（返回 TEXT）
SELECT name, attributes #>> '{specs, chip}' AS chip FROM products;

-- --- JSONB 特有操作符 ---

-- @> 包含
SELECT name FROM products WHERE attributes @> '{"brand": "Apple"}';

-- <@ 被包含
SELECT name FROM products WHERE '{"brand": "Apple"}' <@ attributes;

-- ? 存在键
SELECT name FROM products WHERE attributes ? 'color';

-- ?| 存在任意键
SELECT name FROM products WHERE attributes ?| array['color', 'ram'];

-- ?& 存在所有键
SELECT name FROM products WHERE attributes ?& array['brand', 'storage'];

-- || 合并
SELECT name, attributes || '{"warranty": "1年"}' AS updated FROM products;

-- - 删除键
SELECT name, attributes - 'color' AS without_color FROM products;

-- #- 删除路径
SELECT name, attributes #- '{specs, chip}' AS without_chip FROM products;

-- --- JSONB 函数 ---

-- jsonb_set 设置值
SELECT name, jsonb_set(attributes, '{color}', '"红色"') AS updated FROM products;

-- jsonb_insert 插入值
SELECT name, jsonb_insert(tags, '{0}', '"新品"') AS updated FROM products;

-- jsonb_each 展开为行
SELECT name, key, value
FROM products, jsonb_each(attributes)
WHERE category = 'phone';

-- jsonb_array_elements 展开数组
SELECT name, tag
FROM products, jsonb_array_elements_text(tags) AS tag;

-- jsonb_object_keys 获取所有键
SELECT DISTINCT key
FROM products, jsonb_object_keys(attributes) AS key;

-- jsonb_typeof 获取类型
SELECT name, jsonb_typeof(attributes), jsonb_typeof(tags) FROM products;

-- jsonb_agg 聚合为 JSON 数组
SELECT category, jsonb_agg(name) AS products FROM products GROUP BY category;

-- jsonb_object_agg 聚合为 JSON 对象
SELECT jsonb_object_agg(name, price) AS price_map FROM products;

-- --- JSONB 索引 ---
CREATE INDEX idx_products_attributes ON products USING GIN (attributes);
CREATE INDEX idx_products_tags ON products USING GIN (tags);

-- 针对特定路径的索引
CREATE INDEX idx_products_brand ON products ((attributes ->> 'brand'));


-- ============================================================
--                    2. 数组操作
-- ============================================================

-- 创建包含数组的表
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    tags TEXT[],
    scores INTEGER[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO articles (title, tags, scores) VALUES
    ('PostgreSQL 入门', ARRAY['数据库', 'SQL', 'PostgreSQL'], ARRAY[95, 88, 92]),
    ('Python 数据分析', ARRAY['Python', '数据分析', 'Pandas'], ARRAY[90, 85, 88]),
    ('Web 开发指南', ARRAY['Web', 'JavaScript', 'React'], ARRAY[88, 92, 90]);

-- --- 数组操作符 ---

-- = 相等
SELECT * FROM articles WHERE tags = ARRAY['数据库', 'SQL', 'PostgreSQL'];

-- @> 包含
SELECT * FROM articles WHERE tags @> ARRAY['Python'];

-- <@ 被包含
SELECT * FROM articles WHERE ARRAY['数据库'] <@ tags;

-- && 有交集
SELECT * FROM articles WHERE tags && ARRAY['Python', 'SQL'];

-- || 连接
SELECT title, tags || ARRAY['推荐'] AS new_tags FROM articles;

-- 索引访问（从1开始）
SELECT title, tags[1] AS first_tag, scores[1] AS first_score FROM articles;

-- 切片
SELECT title, tags[1:2] AS first_two_tags FROM articles;

-- --- 数组函数 ---

-- array_length 长度
SELECT title, array_length(tags, 1) AS tag_count FROM articles;

-- array_dims 维度
SELECT title, array_dims(tags) FROM articles;

-- array_position 查找位置
SELECT title, array_position(tags, 'Python') AS python_pos FROM articles;

-- array_remove 删除元素
SELECT title, array_remove(tags, 'SQL') AS without_sql FROM articles;

-- array_replace 替换元素
SELECT title, array_replace(tags, 'SQL', 'MySQL') AS replaced FROM articles;

-- array_cat 连接数组
SELECT title, array_cat(tags, ARRAY['新标签']) AS extended FROM articles;

-- array_append / array_prepend
SELECT title, array_append(tags, '热门') AS appended FROM articles;
SELECT title, array_prepend('精选', tags) AS prepended FROM articles;

-- unnest 展开数组
SELECT title, unnest(tags) AS tag FROM articles;

-- 使用 unnest 统计标签
SELECT tag, COUNT(*) AS count
FROM articles, unnest(tags) AS tag
GROUP BY tag
ORDER BY count DESC;

-- array_agg 聚合为数组
SELECT array_agg(title) AS all_titles FROM articles;

-- --- 数组与聚合 ---

-- 计算平均分
SELECT title, (SELECT AVG(s) FROM unnest(scores) AS s) AS avg_score FROM articles;

-- ANY / ALL
SELECT * FROM articles WHERE 90 = ANY(scores);
SELECT * FROM articles WHERE 80 < ALL(scores);


-- ============================================================
--                    3. 全文搜索
-- ============================================================

-- 创建文章内容表
CREATE TABLE IF NOT EXISTS posts (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    content TEXT,
    search_vector TSVECTOR,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO posts (title, content) VALUES
    ('PostgreSQL 全文搜索', 'PostgreSQL 提供了强大的全文搜索功能，支持中文分词和多种搜索模式。'),
    ('数据库索引优化', '合理使用索引可以显著提升数据库查询性能，包括 B-tree、GIN、GiST 等索引类型。'),
    ('SQL 查询技巧', '掌握 SQL 查询技巧可以写出高效的数据库查询语句，提高开发效率。');

-- 更新搜索向量
UPDATE posts SET search_vector = to_tsvector('simple', title || ' ' || content);

-- 创建全文搜索索引
CREATE INDEX idx_posts_search ON posts USING GIN (search_vector);

-- --- 全文搜索查询 ---

-- 基本搜索
SELECT title FROM posts
WHERE search_vector @@ to_tsquery('simple', 'PostgreSQL');

-- 使用 plainto_tsquery（自动处理空格）
SELECT title FROM posts
WHERE search_vector @@ plainto_tsquery('simple', '数据库 索引');

-- 使用 phraseto_tsquery（短语搜索）
SELECT title FROM posts
WHERE search_vector @@ phraseto_tsquery('simple', '全文搜索');

-- 布尔操作
SELECT title FROM posts
WHERE search_vector @@ to_tsquery('simple', 'PostgreSQL | SQL');  -- OR

SELECT title FROM posts
WHERE search_vector @@ to_tsquery('simple', '数据库 & 索引');  -- AND

SELECT title FROM posts
WHERE search_vector @@ to_tsquery('simple', 'SQL & !PostgreSQL');  -- NOT

-- --- 搜索排名 ---

-- ts_rank 计算相关度
SELECT
    title,
    ts_rank(search_vector, to_tsquery('simple', '数据库')) AS rank
FROM posts
WHERE search_vector @@ to_tsquery('simple', '数据库')
ORDER BY rank DESC;

-- ts_headline 高亮显示
SELECT
    title,
    ts_headline('simple', content, to_tsquery('simple', '索引'),
        'StartSel=<b>, StopSel=</b>') AS highlighted
FROM posts
WHERE search_vector @@ to_tsquery('simple', '索引');


-- ============================================================
--                    4. 范围类型
-- ============================================================

-- PostgreSQL 支持范围类型
CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    during TSTZRANGE,  -- 时间戳范围
    capacity INT4RANGE  -- 整数范围
);

INSERT INTO events (name, during, capacity) VALUES
    ('会议A', '[2024-01-15 09:00, 2024-01-15 12:00)', '[10, 50)'),
    ('会议B', '[2024-01-15 14:00, 2024-01-15 17:00)', '[20, 100)'),
    ('会议C', '[2024-01-15 10:00, 2024-01-15 11:00)', '[5, 30)');

-- 范围操作符
-- @> 包含元素
SELECT * FROM events WHERE during @> '2024-01-15 10:30'::TIMESTAMPTZ;

-- && 重叠
SELECT * FROM events
WHERE during && '[2024-01-15 11:00, 2024-01-15 15:00)'::TSTZRANGE;

-- << 完全在左边
SELECT * FROM events
WHERE during << '[2024-01-15 13:00, 2024-01-15 14:00)'::TSTZRANGE;

-- 范围函数
SELECT
    name,
    lower(during) AS start_time,
    upper(during) AS end_time,
    upper(during) - lower(during) AS duration
FROM events;

-- 排除约束（防止时间重叠）
ALTER TABLE events
ADD CONSTRAINT no_overlap EXCLUDE USING GIST (during WITH &&);


-- ============================================================
--                    5. 继承
-- ============================================================

-- 父表
CREATE TABLE IF NOT EXISTS vehicles (
    id SERIAL PRIMARY KEY,
    brand VARCHAR(50) NOT NULL,
    model VARCHAR(50) NOT NULL,
    year INTEGER
);

-- 子表继承父表
CREATE TABLE IF NOT EXISTS cars (
    doors INTEGER,
    fuel_type VARCHAR(20)
) INHERITS (vehicles);

CREATE TABLE IF NOT EXISTS motorcycles (
    engine_cc INTEGER
) INHERITS (vehicles);

-- 插入数据
INSERT INTO cars (brand, model, year, doors, fuel_type)
VALUES ('Toyota', 'Camry', 2023, 4, 'hybrid');

INSERT INTO motorcycles (brand, model, year, engine_cc)
VALUES ('Honda', 'CBR', 2023, 650);

-- 查询所有车辆（包括子表）
SELECT * FROM vehicles;

-- 只查询父表
SELECT * FROM ONLY vehicles;

-- 查询特定子表
SELECT * FROM cars;
SELECT * FROM motorcycles;


-- ============================================================
--                    6. 存储过程与函数
-- ============================================================

-- --- 创建函数 ---

-- 返回标量值
CREATE OR REPLACE FUNCTION get_employee_count(dept_id INTEGER)
RETURNS INTEGER AS $$
DECLARE
    emp_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO emp_count
    FROM employees
    WHERE department_id = dept_id;
    RETURN emp_count;
END;
$$ LANGUAGE plpgsql;

-- 调用函数
SELECT get_employee_count(1);

-- 返回表
CREATE OR REPLACE FUNCTION get_employees_by_dept(dept_id INTEGER)
RETURNS TABLE(id INTEGER, name VARCHAR, salary NUMERIC) AS $$
BEGIN
    RETURN QUERY
    SELECT e.id, e.name, e.salary
    FROM employees e
    WHERE e.department_id = dept_id
    ORDER BY e.salary DESC;
END;
$$ LANGUAGE plpgsql;

-- 调用
SELECT * FROM get_employees_by_dept(1);

-- 返回 SETOF
CREATE OR REPLACE FUNCTION get_high_salary_employees(min_salary NUMERIC)
RETURNS SETOF employees AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM employees WHERE salary >= min_salary;
END;
$$ LANGUAGE plpgsql;

-- --- 存储过程 (PostgreSQL 11+) ---
CREATE OR REPLACE PROCEDURE transfer_money(
    from_account VARCHAR,
    to_account VARCHAR,
    amount NUMERIC
)
LANGUAGE plpgsql
AS $$
BEGIN
    -- 扣款
    UPDATE users SET balance = balance - amount WHERE username = from_account;

    -- 存款
    UPDATE users SET balance = balance + amount WHERE username = to_account;

    -- 自动提交（过程中可以使用事务控制）
    COMMIT;
END;
$$;

-- 调用存储过程
CALL transfer_money('alice', 'bob', 100);

-- --- 触发器 ---

-- 创建更新时间触发器函数
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 创建触发器
CREATE TRIGGER trigger_users_updated
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();


-- ============================================================
--                    7. 视图
-- ============================================================

-- 普通视图
CREATE OR REPLACE VIEW employee_details AS
SELECT
    e.id,
    e.name,
    e.email,
    e.salary,
    d.name AS department,
    m.name AS manager
FROM employees e
LEFT JOIN departments d ON e.department_id = d.id
LEFT JOIN employees m ON e.manager_id = m.id;

-- 查询视图
SELECT * FROM employee_details;

-- 物化视图（缓存结果）
CREATE MATERIALIZED VIEW department_stats AS
SELECT
    d.id,
    d.name,
    COUNT(e.id) AS employee_count,
    ROUND(AVG(e.salary), 2) AS avg_salary,
    SUM(e.salary) AS total_salary
FROM departments d
LEFT JOIN employees e ON d.id = e.department_id
GROUP BY d.id, d.name
WITH DATA;

-- 查询物化视图
SELECT * FROM department_stats;

-- 刷新物化视图
REFRESH MATERIALIZED VIEW department_stats;

-- 并发刷新（需要唯一索引）
CREATE UNIQUE INDEX ON department_stats (id);
REFRESH MATERIALIZED VIEW CONCURRENTLY department_stats;


-- ============================================================
--                    8. 扩展
-- ============================================================

-- 查看已安装扩展
SELECT * FROM pg_extension;

-- 查看可用扩展
SELECT * FROM pg_available_extensions WHERE name LIKE '%uuid%';

-- 安装扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 使用 UUID
SELECT uuid_generate_v4();

-- 使用加密
SELECT crypt('password', gen_salt('bf'));

-- 验证密码
SELECT crypt('password', '$2a$06$...') = '$2a$06$...';


-- ============================================================
--                    总结
-- ============================================================

/*
PostgreSQL 高级特性：

JSONB：
- 操作符：->, ->>, #>, @>, ?, ||
- 函数：jsonb_set, jsonb_each, jsonb_agg
- GIN 索引支持

数组：
- 操作符：@>, <@, &&, ||
- 函数：array_agg, unnest, array_length
- ANY/ALL 比较

全文搜索：
- TSVECTOR 和 TSQUERY
- GIN 索引
- ts_rank 排名
- ts_headline 高亮

范围类型：
- INT4RANGE, TSTZRANGE 等
- 重叠和包含操作
- 排除约束

继承：
- 表继承
- ONLY 关键字

函数与过程：
- CREATE FUNCTION
- CREATE PROCEDURE
- 触发器

视图：
- 普通视图
- 物化视图 (MATERIALIZED VIEW)
- REFRESH MATERIALIZED VIEW

扩展：
- uuid-ossp
- pgcrypto
- 众多社区扩展
*/
