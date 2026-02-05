-- ============================================================
--                    PostgreSQL 连接与子查询
-- ============================================================
-- 本文件介绍 PostgreSQL 中的表连接和子查询操作。
-- ============================================================

-- \c learn_postgresql

-- ============================================================
--                    准备测试数据
-- ============================================================

-- 部门表
CREATE TABLE IF NOT EXISTS departments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    location VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 员工表
CREATE TABLE IF NOT EXISTS employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    department_id INTEGER REFERENCES departments(id)
        ON DELETE SET NULL ON UPDATE CASCADE,
    manager_id INTEGER REFERENCES employees(id)
        ON DELETE SET NULL ON UPDATE CASCADE,
    salary NUMERIC(10, 2) NOT NULL,
    hire_date DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 项目表
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    budget NUMERIC(12, 2),
    start_date DATE,
    end_date DATE,
    status VARCHAR(20) DEFAULT 'planning'
        CHECK (status IN ('planning', 'active', 'completed', 'cancelled'))
);

-- 员工-项目关联表
CREATE TABLE IF NOT EXISTS employee_projects (
    employee_id INTEGER REFERENCES employees(id) ON DELETE CASCADE,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    role VARCHAR(50),
    joined_at DATE,
    PRIMARY KEY (employee_id, project_id)
);

-- 清空并插入测试数据
TRUNCATE TABLE employee_projects, employees, departments, projects RESTART IDENTITY CASCADE;

INSERT INTO departments (name, location) VALUES
    ('技术部', '北京'),
    ('市场部', '上海'),
    ('财务部', '北京'),
    ('人事部', '深圳'),
    ('运营部', '广州');

INSERT INTO employees (name, email, department_id, manager_id, salary, hire_date) VALUES
    ('张三', 'zhangsan@company.com', 1, NULL, 25000.00, '2020-01-15'),
    ('李四', 'lisi@company.com', 1, 1, 18000.00, '2021-03-20'),
    ('王五', 'wangwu@company.com', 1, 1, 20000.00, '2020-08-10'),
    ('赵六', 'zhaoliu@company.com', 2, NULL, 22000.00, '2019-06-01'),
    ('钱七', 'qianqi@company.com', 2, 4, 15000.00, '2022-01-10'),
    ('孙八', 'sunba@company.com', 3, NULL, 28000.00, '2018-04-15'),
    ('周九', 'zhoujiu@company.com', 3, 6, 16000.00, '2023-02-01'),
    ('吴十', 'wushi@company.com', NULL, NULL, 12000.00, '2023-06-15');

INSERT INTO projects (name, budget, start_date, end_date, status) VALUES
    ('网站重构', 500000.00, '2024-01-01', '2024-06-30', 'active'),
    ('APP开发', 800000.00, '2024-03-01', '2024-12-31', 'active'),
    ('数据分析平台', 300000.00, '2023-06-01', '2023-12-31', 'completed'),
    ('市场推广', 200000.00, '2024-02-01', NULL, 'planning');

INSERT INTO employee_projects (employee_id, project_id, role, joined_at) VALUES
    (1, 1, '项目经理', '2024-01-01'),
    (2, 1, '开发工程师', '2024-01-15'),
    (3, 1, '开发工程师', '2024-01-15'),
    (1, 2, '技术顾问', '2024-03-01'),
    (2, 2, '开发工程师', '2024-03-15'),
    (4, 4, '项目经理', '2024-02-01'),
    (5, 4, '市场专员', '2024-02-15'),
    (6, 3, '项目经理', '2023-06-01');


-- ============================================================
--                    1. JOIN 连接
-- ============================================================

-- --- INNER JOIN ---
SELECT
    e.id,
    e.name AS employee_name,
    e.salary,
    d.name AS department_name,
    d.location
FROM employees e
INNER JOIN departments d ON e.department_id = d.id;

-- --- LEFT JOIN ---
SELECT
    e.id,
    e.name AS employee_name,
    d.name AS department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.id;

-- 查找没有部门的员工
SELECT e.name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.id
WHERE d.id IS NULL;

-- --- RIGHT JOIN ---
SELECT
    d.name AS department_name,
    e.name AS employee_name
FROM employees e
RIGHT JOIN departments d ON e.department_id = d.id;

-- --- FULL OUTER JOIN ---
-- PostgreSQL 支持完整外连接（MySQL 不支持）
SELECT
    e.name AS employee_name,
    d.name AS department_name
FROM employees e
FULL OUTER JOIN departments d ON e.department_id = d.id;

-- --- CROSS JOIN ---
SELECT e.name, p.name AS project
FROM employees e
CROSS JOIN projects p
LIMIT 20;

-- --- 自连接 ---
SELECT
    e.name AS employee,
    e.salary AS employee_salary,
    m.name AS manager,
    m.salary AS manager_salary
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;

-- --- NATURAL JOIN ---
-- 自动使用同名列连接（谨慎使用）
-- SELECT * FROM employees NATURAL JOIN departments;

-- --- USING 子句 ---
-- 当连接列名相同时
-- SELECT * FROM employees JOIN departments USING (department_id);


-- ============================================================
--                    2. LATERAL JOIN
-- ============================================================

/*
LATERAL JOIN 是 PostgreSQL 的强大特性：
- 子查询可以引用外层查询的列
- 类似于相关子查询，但可以返回多行
*/

-- 查询每个部门薪资最高的 2 名员工
SELECT
    d.name AS department,
    top_emp.name AS employee,
    top_emp.salary
FROM departments d
LEFT JOIN LATERAL (
    SELECT name, salary
    FROM employees e
    WHERE e.department_id = d.id
    ORDER BY salary DESC
    LIMIT 2
) top_emp ON true
ORDER BY d.name, top_emp.salary DESC;

-- 查询每个员工参与的项目数量
SELECT
    e.name AS employee,
    proj_count.count AS project_count
FROM employees e
LEFT JOIN LATERAL (
    SELECT COUNT(*) AS count
    FROM employee_projects ep
    WHERE ep.employee_id = e.id
) proj_count ON true;


-- ============================================================
--                    3. 子查询
-- ============================================================

-- --- 标量子查询 ---
SELECT
    e.name,
    e.salary,
    (SELECT AVG(salary) FROM employees) AS avg_salary,
    e.salary - (SELECT AVG(salary) FROM employees) AS diff_from_avg
FROM employees e;

-- --- WHERE 子查询 ---
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- --- IN 子查询 ---
SELECT name, department_id
FROM employees
WHERE department_id IN (
    SELECT id FROM departments WHERE location = '北京'
);

-- --- ANY / ALL ---
SELECT name, salary
FROM employees
WHERE salary > ANY (
    SELECT salary FROM employees WHERE department_id = 1
);

SELECT name, salary
FROM employees
WHERE salary > ALL (
    SELECT salary FROM employees WHERE department_id = 2
);

-- --- EXISTS ---
SELECT d.name
FROM departments d
WHERE EXISTS (
    SELECT 1 FROM employees e WHERE e.department_id = d.id
);

SELECT d.name
FROM departments d
WHERE NOT EXISTS (
    SELECT 1 FROM employees e WHERE e.department_id = d.id
);


-- ============================================================
--                    4. CTE (公用表表达式)
-- ============================================================

-- --- 基本 CTE ---
WITH dept_stats AS (
    SELECT
        department_id,
        COUNT(*) AS emp_count,
        AVG(salary) AS avg_salary,
        SUM(salary) AS total_salary
    FROM employees
    WHERE department_id IS NOT NULL
    GROUP BY department_id
)
SELECT
    d.name AS department,
    ds.emp_count,
    ROUND(ds.avg_salary, 2) AS avg_salary,
    ds.total_salary
FROM dept_stats ds
JOIN departments d ON ds.department_id = d.id
ORDER BY ds.avg_salary DESC;

-- --- 多个 CTE ---
WITH
high_salary AS (
    SELECT * FROM employees WHERE salary >= 20000
),
low_salary AS (
    SELECT * FROM employees WHERE salary < 20000
)
SELECT
    'High Salary' AS category,
    COUNT(*) AS count,
    ROUND(AVG(salary), 2) AS avg_salary
FROM high_salary
UNION ALL
SELECT
    'Low Salary' AS category,
    COUNT(*) AS count,
    ROUND(AVG(salary), 2) AS avg_salary
FROM low_salary;


-- ============================================================
--                    5. 递归 CTE
-- ============================================================

-- 组织层级结构
WITH RECURSIVE employee_hierarchy AS (
    -- 基础查询：顶级员工（无经理）
    SELECT
        id,
        name,
        manager_id,
        1 AS level,
        ARRAY[name] AS path,
        name::TEXT AS path_string
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- 递归查询：下级员工
    SELECT
        e.id,
        e.name,
        e.manager_id,
        eh.level + 1,
        eh.path || e.name,
        eh.path_string || ' -> ' || e.name
    FROM employees e
    JOIN employee_hierarchy eh ON e.manager_id = eh.id
)
SELECT
    id,
    name,
    level,
    path_string AS hierarchy
FROM employee_hierarchy
ORDER BY path;

-- 数字序列生成
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < 10
)
SELECT * FROM numbers;

-- 日期序列生成
WITH RECURSIVE dates AS (
    SELECT DATE '2024-01-01' AS date
    UNION ALL
    SELECT date + 1 FROM dates WHERE date < '2024-01-10'
)
SELECT * FROM dates;


-- ============================================================
--                    6. 窗口函数
-- ============================================================

/*
窗口函数在 PostgreSQL 中功能强大：
- 不会减少行数
- 可以访问当前行的"窗口"内的其他行
*/

-- --- 排名函数 ---
SELECT
    name,
    department_id,
    salary,
    -- ROW_NUMBER: 行号，不重复
    ROW_NUMBER() OVER (ORDER BY salary DESC) AS row_num,
    -- RANK: 排名，相同值相同排名，有间隙
    RANK() OVER (ORDER BY salary DESC) AS rank,
    -- DENSE_RANK: 排名，相同值相同排名，无间隙
    DENSE_RANK() OVER (ORDER BY salary DESC) AS dense_rank,
    -- NTILE: 分成 N 组
    NTILE(4) OVER (ORDER BY salary DESC) AS quartile
FROM employees;

-- --- 分区排名 ---
SELECT
    name,
    department_id,
    salary,
    ROW_NUMBER() OVER (
        PARTITION BY department_id
        ORDER BY salary DESC
    ) AS dept_rank
FROM employees
WHERE department_id IS NOT NULL;

-- 取每个部门薪资最高的员工
SELECT * FROM (
    SELECT
        name,
        department_id,
        salary,
        ROW_NUMBER() OVER (
            PARTITION BY department_id
            ORDER BY salary DESC
        ) AS rn
    FROM employees
    WHERE department_id IS NOT NULL
) ranked
WHERE rn = 1;

-- --- 聚合窗口函数 ---
SELECT
    name,
    department_id,
    salary,
    SUM(salary) OVER (PARTITION BY department_id) AS dept_total,
    AVG(salary) OVER (PARTITION BY department_id) AS dept_avg,
    COUNT(*) OVER (PARTITION BY department_id) AS dept_count,
    salary - AVG(salary) OVER (PARTITION BY department_id) AS diff_from_avg,
    salary * 100.0 / SUM(salary) OVER (PARTITION BY department_id) AS pct_of_dept
FROM employees
WHERE department_id IS NOT NULL;

-- --- 累计计算 ---
SELECT
    name,
    salary,
    hire_date,
    SUM(salary) OVER (ORDER BY hire_date) AS running_total,
    AVG(salary) OVER (ORDER BY hire_date) AS running_avg,
    COUNT(*) OVER (ORDER BY hire_date) AS cumulative_count
FROM employees
ORDER BY hire_date;

-- --- 偏移函数 ---
SELECT
    name,
    salary,
    hire_date,
    -- LAG: 前一行
    LAG(name) OVER (ORDER BY hire_date) AS prev_employee,
    LAG(salary) OVER (ORDER BY hire_date) AS prev_salary,
    -- LEAD: 后一行
    LEAD(name) OVER (ORDER BY hire_date) AS next_employee,
    -- 指定偏移量和默认值
    LAG(salary, 2, 0) OVER (ORDER BY hire_date) AS salary_2_before,
    -- FIRST_VALUE / LAST_VALUE
    FIRST_VALUE(name) OVER (ORDER BY salary DESC) AS highest_paid,
    LAST_VALUE(name) OVER (
        ORDER BY salary DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS lowest_paid
FROM employees;

-- --- 窗口帧定义 ---
SELECT
    name,
    hire_date,
    salary,
    -- 前后各1行的移动平均
    AVG(salary) OVER (
        ORDER BY hire_date
        ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
    ) AS moving_avg_3,
    -- 前2行到当前行
    AVG(salary) OVER (
        ORDER BY hire_date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg_prev_2,
    -- 当前行到末尾
    SUM(salary) OVER (
        ORDER BY hire_date
        ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
    ) AS remaining_sum
FROM employees
ORDER BY hire_date;


-- ============================================================
--                    7. UNION / INTERSECT / EXCEPT
-- ============================================================

-- UNION (合并，去重)
SELECT name, 'employee' AS type FROM employees
UNION
SELECT name, 'department' AS type FROM departments;

-- UNION ALL (合并，保留重复)
SELECT department_id FROM employees WHERE salary > 20000
UNION ALL
SELECT department_id FROM employees WHERE hire_date > '2022-01-01';

-- INTERSECT (交集)
SELECT department_id FROM employees WHERE salary > 18000
INTERSECT
SELECT department_id FROM employees WHERE hire_date > '2020-01-01';

-- EXCEPT (差集)
SELECT department_id FROM employees
EXCEPT
SELECT id FROM departments WHERE location = '上海';


-- ============================================================
--                    8. 高级查询技巧
-- ============================================================

-- --- DISTINCT ON (PostgreSQL 特有) ---
-- 返回每个分组的第一行
SELECT DISTINCT ON (department_id)
    department_id,
    name,
    salary
FROM employees
WHERE department_id IS NOT NULL
ORDER BY department_id, salary DESC;

-- --- FILTER 子句 ---
SELECT
    department_id,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE salary > 20000) AS high_salary_count,
    AVG(salary) FILTER (WHERE hire_date > '2021-01-01') AS recent_avg
FROM employees
WHERE department_id IS NOT NULL
GROUP BY department_id;

-- --- 数组聚合与展开 ---
-- 聚合为数组
SELECT
    department_id,
    ARRAY_AGG(name ORDER BY salary DESC) AS employees
FROM employees
WHERE department_id IS NOT NULL
GROUP BY department_id;

-- 展开数组
SELECT
    d.name AS department,
    UNNEST(ARRAY_AGG(e.name)) AS employee
FROM departments d
JOIN employees e ON d.id = e.department_id
GROUP BY d.id, d.name;


-- ============================================================
--                    总结
-- ============================================================

/*
PostgreSQL 连接特性：
- 支持 FULL OUTER JOIN
- LATERAL JOIN 强大灵活
- NATURAL JOIN 和 USING

CTE 特性：
- WITH 语句
- 递归 CTE (WITH RECURSIVE)
- 多个 CTE 定义

窗口函数：
- ROW_NUMBER, RANK, DENSE_RANK, NTILE
- LAG, LEAD, FIRST_VALUE, LAST_VALUE
- PARTITION BY 分区
- 窗口帧 (ROWS BETWEEN)
- FILTER 子句

集合操作：
- UNION / UNION ALL
- INTERSECT
- EXCEPT

PostgreSQL 特有：
- DISTINCT ON
- LATERAL JOIN
- ARRAY_AGG / UNNEST
- FILTER 子句
- 丰富的窗口帧选项
*/
