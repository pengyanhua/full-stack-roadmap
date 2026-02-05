-- ============================================================
--                    MySQL 连接与子查询
-- ============================================================
-- 本文件介绍 MySQL 中的表连接和子查询操作。
-- ============================================================

USE learn_mysql;

-- ============================================================
--                    准备测试数据
-- ============================================================

-- 部门表
CREATE TABLE IF NOT EXISTS departments (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    location VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 员工表
CREATE TABLE IF NOT EXISTS employees (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    department_id INT UNSIGNED,
    manager_id INT UNSIGNED,
    salary DECIMAL(10, 2) NOT NULL,
    hire_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (department_id) REFERENCES departments(id)
        ON DELETE SET NULL ON UPDATE CASCADE,
    FOREIGN KEY (manager_id) REFERENCES employees(id)
        ON DELETE SET NULL ON UPDATE CASCADE,

    INDEX idx_department (department_id),
    INDEX idx_manager (manager_id)
) ENGINE=InnoDB;

-- 项目表
CREATE TABLE IF NOT EXISTS projects (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    budget DECIMAL(12, 2),
    start_date DATE,
    end_date DATE,
    status ENUM('planning', 'active', 'completed', 'cancelled') DEFAULT 'planning'
) ENGINE=InnoDB;

-- 员工-项目关联表（多对多）
CREATE TABLE IF NOT EXISTS employee_projects (
    employee_id INT UNSIGNED,
    project_id INT UNSIGNED,
    role VARCHAR(50),
    joined_at DATE,

    PRIMARY KEY (employee_id, project_id),
    FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- 订单表
CREATE TABLE IF NOT EXISTS orders (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNSIGNED,
    total_amount DECIMAL(10, 2) NOT NULL,
    status ENUM('pending', 'paid', 'shipped', 'completed', 'cancelled') DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- 插入测试数据
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

INSERT INTO orders (user_id, total_amount, status) VALUES
    (1, 199.00, 'completed'),
    (1, 599.00, 'completed'),
    (2, 299.00, 'shipped'),
    (2, 899.00, 'paid'),
    (3, 1299.00, 'pending'),
    (NULL, 99.00, 'cancelled');


-- ============================================================
--                    1. INNER JOIN 内连接
-- ============================================================

/*
内连接：返回两个表中匹配的记录
只有在连接条件成立时才返回结果
*/

-- 基本内连接
SELECT
    e.id,
    e.name AS employee_name,
    e.salary,
    d.name AS department_name,
    d.location
FROM employees e
INNER JOIN departments d ON e.department_id = d.id;

-- 简写（省略 INNER）
SELECT
    e.name,
    d.name AS department
FROM employees e
JOIN departments d ON e.department_id = d.id;

-- 多条件连接
SELECT
    e.name,
    d.name AS department
FROM employees e
JOIN departments d ON e.department_id = d.id AND d.location = '北京';

-- 使用 WHERE（效果相同）
SELECT
    e.name,
    d.name AS department
FROM employees e
JOIN departments d ON e.department_id = d.id
WHERE d.location = '北京';


-- ============================================================
--                    2. LEFT JOIN 左连接
-- ============================================================

/*
左连接：返回左表所有记录，右表匹配的记录
右表无匹配时，相应列为 NULL
*/

-- 基本左连接（显示所有员工，包括无部门的）
SELECT
    e.id,
    e.name AS employee_name,
    d.name AS department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.id;

-- 查找没有部门的员工
SELECT
    e.name AS employee_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.id
WHERE d.id IS NULL;

-- 查找没有员工的部门
SELECT
    d.name AS department_name
FROM departments d
LEFT JOIN employees e ON d.id = e.department_id
WHERE e.id IS NULL;


-- ============================================================
--                    3. RIGHT JOIN 右连接
-- ============================================================

/*
右连接：返回右表所有记录，左表匹配的记录
等价于调换顺序的左连接
*/

-- 显示所有部门及其员工
SELECT
    d.name AS department_name,
    e.name AS employee_name
FROM employees e
RIGHT JOIN departments d ON e.department_id = d.id;


-- ============================================================
--                    4. CROSS JOIN 交叉连接
-- ============================================================

/*
交叉连接：返回两个表的笛卡尔积
结果行数 = 表1行数 × 表2行数
*/

-- 所有员工与所有项目的组合
SELECT
    e.name AS employee,
    p.name AS project
FROM employees e
CROSS JOIN projects p
LIMIT 20;

-- 等价写法（旧语法）
SELECT e.name, p.name
FROM employees e, projects p
LIMIT 20;


-- ============================================================
--                    5. 自连接
-- ============================================================

/*
自连接：表与自身连接
常用于树形结构（如员工-经理关系）
*/

-- 查询员工及其经理
SELECT
    e.name AS employee,
    e.salary AS employee_salary,
    m.name AS manager,
    m.salary AS manager_salary
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;

-- 查询薪资高于其经理的员工
SELECT
    e.name AS employee,
    e.salary AS employee_salary,
    m.name AS manager,
    m.salary AS manager_salary
FROM employees e
JOIN employees m ON e.manager_id = m.id
WHERE e.salary > m.salary;

-- 查询同部门的员工对
SELECT
    e1.name AS employee1,
    e2.name AS employee2,
    d.name AS department
FROM employees e1
JOIN employees e2 ON e1.department_id = e2.department_id AND e1.id < e2.id
JOIN departments d ON e1.department_id = d.id;


-- ============================================================
--                    6. 多表连接
-- ============================================================

-- 三表连接：员工、部门、项目
SELECT
    e.name AS employee,
    d.name AS department,
    p.name AS project,
    ep.role
FROM employees e
JOIN departments d ON e.department_id = d.id
JOIN employee_projects ep ON e.id = ep.employee_id
JOIN projects p ON ep.project_id = p.id
ORDER BY e.name, p.name;

-- 四表连接：用户、订单、员工、部门
SELECT
    u.username AS customer,
    o.total_amount,
    o.status AS order_status,
    e.name AS handler,
    d.name AS department
FROM users u
JOIN orders o ON u.id = o.user_id
LEFT JOIN employees e ON e.department_id = 2  -- 假设市场部处理订单
LEFT JOIN departments d ON e.department_id = d.id
WHERE e.id = 4;


-- ============================================================
--                    7. UNION 联合查询
-- ============================================================

/*
UNION：合并多个 SELECT 结果
- UNION：去除重复行
- UNION ALL：保留所有行（更快）
- 列数和数据类型必须兼容
*/

-- 合并两个查询（去重）
SELECT name, 'employee' AS type FROM employees
UNION
SELECT name, 'department' AS type FROM departments;

-- 保留重复（UNION ALL）
SELECT department_id FROM employees WHERE salary > 20000
UNION ALL
SELECT department_id FROM employees WHERE hire_date > '2022-01-01';

-- UNION 后排序
SELECT name, salary, 'high' AS level FROM employees WHERE salary >= 20000
UNION
SELECT name, salary, 'low' AS level FROM employees WHERE salary < 20000
ORDER BY salary DESC;


-- ============================================================
--                    8. 标量子查询
-- ============================================================

/*
标量子查询：返回单个值
可用于 SELECT、WHERE、HAVING 中
*/

-- 在 SELECT 中
SELECT
    e.name,
    e.salary,
    (SELECT AVG(salary) FROM employees) AS avg_salary,
    e.salary - (SELECT AVG(salary) FROM employees) AS diff_from_avg
FROM employees e;

-- 在 WHERE 中
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- 在 HAVING 中
SELECT
    department_id,
    AVG(salary) AS avg_salary
FROM employees
GROUP BY department_id
HAVING AVG(salary) > (SELECT AVG(salary) FROM employees);


-- ============================================================
--                    9. 列子查询
-- ============================================================

/*
列子查询：返回一列多行
常与 IN、ANY、ALL 配合使用
*/

-- IN 子查询
SELECT name, department_id
FROM employees
WHERE department_id IN (
    SELECT id FROM departments WHERE location = '北京'
);

-- NOT IN 子查询
SELECT name
FROM employees
WHERE department_id NOT IN (
    SELECT id FROM departments WHERE location = '上海'
);

-- ANY/SOME：满足子查询中任一值
SELECT name, salary
FROM employees
WHERE salary > ANY (
    SELECT salary FROM employees WHERE department_id = 1
);

-- ALL：满足子查询中所有值
SELECT name, salary
FROM employees
WHERE salary > ALL (
    SELECT salary FROM employees WHERE department_id = 2
);


-- ============================================================
--                    10. 行子查询
-- ============================================================

/*
行子查询：返回一行多列
用于比较多个列
*/

-- 查找与某员工相同部门和经理的员工
SELECT name, department_id, manager_id
FROM employees
WHERE (department_id, manager_id) = (
    SELECT department_id, manager_id
    FROM employees
    WHERE name = '李四'
);


-- ============================================================
--                    11. 表子查询
-- ============================================================

/*
表子查询：返回多行多列
用于 FROM 子句（派生表）
*/

-- 派生表（必须有别名）
SELECT
    dept_stats.department_name,
    dept_stats.emp_count,
    dept_stats.avg_salary
FROM (
    SELECT
        d.name AS department_name,
        COUNT(e.id) AS emp_count,
        AVG(e.salary) AS avg_salary
    FROM departments d
    LEFT JOIN employees e ON d.id = e.department_id
    GROUP BY d.id, d.name
) AS dept_stats
WHERE dept_stats.emp_count > 0
ORDER BY dept_stats.avg_salary DESC;


-- ============================================================
--                    12. 关联子查询
-- ============================================================

/*
关联子查询：子查询引用外层查询的列
每行都执行一次子查询
*/

-- 查询薪资高于所在部门平均薪资的员工
SELECT
    e.name,
    e.salary,
    e.department_id
FROM employees e
WHERE e.salary > (
    SELECT AVG(e2.salary)
    FROM employees e2
    WHERE e2.department_id = e.department_id
);

-- 查询每个部门薪资最高的员工
SELECT
    e.name,
    e.salary,
    e.department_id
FROM employees e
WHERE e.salary = (
    SELECT MAX(e2.salary)
    FROM employees e2
    WHERE e2.department_id = e.department_id
);


-- ============================================================
--                    13. EXISTS 子查询
-- ============================================================

/*
EXISTS：检查子查询是否返回任何行
- 返回 TRUE 或 FALSE
- 通常比 IN 更高效（尤其是大数据集）
*/

-- 查询有员工的部门
SELECT d.name
FROM departments d
WHERE EXISTS (
    SELECT 1 FROM employees e WHERE e.department_id = d.id
);

-- 查询没有员工的部门
SELECT d.name
FROM departments d
WHERE NOT EXISTS (
    SELECT 1 FROM employees e WHERE e.department_id = d.id
);

-- 查询参与了项目的员工
SELECT e.name
FROM employees e
WHERE EXISTS (
    SELECT 1 FROM employee_projects ep WHERE ep.employee_id = e.id
);


-- ============================================================
--                    14. 公用表表达式 CTE
-- ============================================================

/*
CTE (Common Table Expression)：
- 临时命名的结果集
- 提高可读性
- 可递归
- MySQL 8.0+ 支持
*/

-- 基本 CTE
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
    ds.avg_salary,
    ds.total_salary
FROM dept_stats ds
JOIN departments d ON ds.department_id = d.id
ORDER BY ds.avg_salary DESC;

-- 多个 CTE
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
    AVG(salary) AS avg_salary
FROM high_salary
UNION ALL
SELECT
    'Low Salary' AS category,
    COUNT(*) AS count,
    AVG(salary) AS avg_salary
FROM low_salary;

-- 递归 CTE（组织层级）
WITH RECURSIVE employee_hierarchy AS (
    -- 基础查询：顶级员工（无经理）
    SELECT
        id,
        name,
        manager_id,
        1 AS level,
        CAST(name AS CHAR(500)) AS path
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- 递归查询：下级员工
    SELECT
        e.id,
        e.name,
        e.manager_id,
        eh.level + 1,
        CONCAT(eh.path, ' -> ', e.name)
    FROM employees e
    JOIN employee_hierarchy eh ON e.manager_id = eh.id
)
SELECT
    id,
    name,
    level,
    path
FROM employee_hierarchy
ORDER BY path;


-- ============================================================
--                    总结
-- ============================================================

/*
连接类型：
- INNER JOIN：只返回匹配的行
- LEFT JOIN：返回左表所有行
- RIGHT JOIN：返回右表所有行
- CROSS JOIN：笛卡尔积
- 自连接：表与自身连接

子查询类型：
- 标量子查询：返回单个值
- 列子查询：返回一列（IN, ANY, ALL）
- 行子查询：返回一行
- 表子查询：返回多行多列（派生表）
- 关联子查询：引用外层查询

关键字：
- UNION/UNION ALL：合并结果集
- EXISTS/NOT EXISTS：存在性检查
- IN/NOT IN：列表匹配
- ANY/SOME/ALL：比较运算

CTE (WITH)：
- 提高可读性
- 可重复引用
- 支持递归（MySQL 8.0+）

性能提示：
- JOIN 通常比子查询更高效
- EXISTS 比 IN 在大数据集上更快
- 关联子查询可能较慢，考虑改写为 JOIN
- 使用 EXPLAIN 分析查询计划
*/
