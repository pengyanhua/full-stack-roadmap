-- ============================================================
--                    MySQL 存储过程与函数
-- ============================================================
-- 本文件介绍 MySQL 存储过程、函数和触发器。
-- ============================================================

USE learn_mysql;

-- ============================================================
--                    1. 存储过程基础
-- ============================================================

/*
存储过程优点：
- 减少网络传输
- 提高性能（预编译）
- 代码复用
- 安全性（限制直接访问表）

存储过程 vs 函数：
- 存储过程：可有多个输出，可执行 DML/DDL，CALL 调用
- 函数：只能返回一个值，可在 SQL 中使用
*/

-- 更改语句结束符（存储过程内部使用分号）
DELIMITER //

-- --- 基本存储过程 ---

-- 无参数存储过程
CREATE PROCEDURE get_all_employees()
BEGIN
    SELECT * FROM employees;
END//

-- 调用存储过程
DELIMITER ;
CALL get_all_employees();

-- --- 带参数的存储过程 ---

DELIMITER //

-- IN 参数（输入）
CREATE PROCEDURE get_employee_by_id(
    IN emp_id INT
)
BEGIN
    SELECT * FROM employees WHERE id = emp_id;
END//

-- OUT 参数（输出）
CREATE PROCEDURE get_employee_count(
    OUT total INT
)
BEGIN
    SELECT COUNT(*) INTO total FROM employees;
END//

-- INOUT 参数（输入输出）
CREATE PROCEDURE increase_salary(
    INOUT salary DECIMAL(10, 2),
    IN percentage DECIMAL(5, 2)
)
BEGIN
    SET salary = salary * (1 + percentage / 100);
END//

DELIMITER ;

-- 调用示例
CALL get_employee_by_id(1);

CALL get_employee_count(@count);
SELECT @count AS employee_count;

SET @sal = 10000;
CALL increase_salary(@sal, 10);
SELECT @sal AS new_salary;  -- 11000


-- ============================================================
--                    2. 变量与流程控制
-- ============================================================

DELIMITER //

-- --- 变量声明和赋值 ---
CREATE PROCEDURE variable_demo()
BEGIN
    -- 声明变量
    DECLARE emp_name VARCHAR(50);
    DECLARE emp_salary DECIMAL(10, 2) DEFAULT 0;
    DECLARE total_count INT;

    -- 赋值方式1：SET
    SET emp_name = 'Test';
    SET emp_salary = 5000;

    -- 赋值方式2：SELECT INTO
    SELECT COUNT(*) INTO total_count FROM employees;

    -- 赋值方式3：SELECT INTO 多个变量
    SELECT name, salary INTO emp_name, emp_salary
    FROM employees WHERE id = 1;

    -- 输出结果
    SELECT emp_name, emp_salary, total_count;
END//

-- --- IF 条件 ---
CREATE PROCEDURE check_salary_level(
    IN emp_id INT,
    OUT level VARCHAR(20)
)
BEGIN
    DECLARE emp_salary DECIMAL(10, 2);

    SELECT salary INTO emp_salary
    FROM employees WHERE id = emp_id;

    IF emp_salary >= 25000 THEN
        SET level = '高级';
    ELSEIF emp_salary >= 15000 THEN
        SET level = '中级';
    ELSE
        SET level = '初级';
    END IF;
END//

-- --- CASE 语句 ---
CREATE PROCEDURE get_department_info(
    IN dept_id INT,
    OUT dept_type VARCHAR(20)
)
BEGIN
    DECLARE dept_name VARCHAR(50);

    SELECT name INTO dept_name
    FROM departments WHERE id = dept_id;

    CASE dept_name
        WHEN '技术部' THEN SET dept_type = '研发';
        WHEN '市场部' THEN SET dept_type = '销售';
        WHEN '财务部' THEN SET dept_type = '管理';
        WHEN '人事部' THEN SET dept_type = '管理';
        ELSE SET dept_type = '其他';
    END CASE;
END//

-- --- CASE 搜索形式 ---
CREATE PROCEDURE evaluate_performance(
    IN emp_id INT,
    OUT evaluation VARCHAR(20)
)
BEGIN
    DECLARE emp_salary DECIMAL(10, 2);
    DECLARE hire_years INT;

    SELECT salary,
           TIMESTAMPDIFF(YEAR, hire_date, CURDATE())
    INTO emp_salary, hire_years
    FROM employees WHERE id = emp_id;

    CASE
        WHEN emp_salary > 20000 AND hire_years > 3 THEN
            SET evaluation = '优秀';
        WHEN emp_salary > 15000 OR hire_years > 2 THEN
            SET evaluation = '良好';
        ELSE
            SET evaluation = '待提升';
    END CASE;
END//

DELIMITER ;

-- 测试
CALL variable_demo();

CALL check_salary_level(1, @level);
SELECT @level;

CALL evaluate_performance(1, @eval);
SELECT @eval;


-- ============================================================
--                    3. 循环结构
-- ============================================================

DELIMITER //

-- --- WHILE 循环 ---
CREATE PROCEDURE while_demo()
BEGIN
    DECLARE i INT DEFAULT 1;
    DECLARE result VARCHAR(100) DEFAULT '';

    WHILE i <= 5 DO
        SET result = CONCAT(result, i, ' ');
        SET i = i + 1;
    END WHILE;

    SELECT result AS numbers;
END//

-- --- REPEAT 循环（类似 do-while）---
CREATE PROCEDURE repeat_demo()
BEGIN
    DECLARE i INT DEFAULT 1;
    DECLARE total INT DEFAULT 0;

    REPEAT
        SET total = total + i;
        SET i = i + 1;
    UNTIL i > 10
    END REPEAT;

    SELECT total AS sum_1_to_10;  -- 55
END//

-- --- LOOP 循环（无限循环，需要 LEAVE 退出）---
CREATE PROCEDURE loop_demo()
BEGIN
    DECLARE i INT DEFAULT 0;

    loop_label: LOOP
        SET i = i + 1;

        -- 跳过偶数
        IF i MOD 2 = 0 THEN
            ITERATE loop_label;  -- 类似 continue
        END IF;

        -- 大于 10 退出
        IF i > 10 THEN
            LEAVE loop_label;  -- 类似 break
        END IF;

        SELECT i AS odd_number;
    END LOOP loop_label;
END//

-- --- 实际应用：批量更新 ---
CREATE PROCEDURE batch_update_salary(
    IN percentage DECIMAL(5, 2),
    IN batch_size INT
)
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE total_updated INT DEFAULT 0;
    DECLARE rows_affected INT;

    batch_loop: LOOP
        UPDATE employees
        SET salary = salary * (1 + percentage / 100)
        WHERE salary < 20000
        LIMIT batch_size;

        SET rows_affected = ROW_COUNT();
        SET total_updated = total_updated + rows_affected;

        IF rows_affected = 0 THEN
            LEAVE batch_loop;
        END IF;

        -- 可以在这里添加延迟或事务控制
        COMMIT;
    END LOOP;

    SELECT total_updated AS employees_updated;
END//

DELIMITER ;

-- 测试
CALL while_demo();
CALL repeat_demo();


-- ============================================================
--                    4. 游标（Cursor）
-- ============================================================

DELIMITER //

-- 基本游标使用
CREATE PROCEDURE cursor_demo()
BEGIN
    -- 声明变量
    DECLARE done INT DEFAULT 0;
    DECLARE emp_id INT;
    DECLARE emp_name VARCHAR(50);
    DECLARE emp_salary DECIMAL(10, 2);

    -- 声明游标
    DECLARE emp_cursor CURSOR FOR
        SELECT id, name, salary FROM employees WHERE department_id = 1;

    -- 声明处理器（当游标到达末尾时设置 done = 1）
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;

    -- 创建临时表存储结果
    CREATE TEMPORARY TABLE IF NOT EXISTS cursor_results (
        id INT,
        name VARCHAR(50),
        salary DECIMAL(10, 2),
        bonus DECIMAL(10, 2)
    );

    TRUNCATE TABLE cursor_results;

    -- 打开游标
    OPEN emp_cursor;

    -- 循环读取
    read_loop: LOOP
        FETCH emp_cursor INTO emp_id, emp_name, emp_salary;

        IF done THEN
            LEAVE read_loop;
        END IF;

        -- 处理每一行
        INSERT INTO cursor_results VALUES (
            emp_id,
            emp_name,
            emp_salary,
            emp_salary * 0.1  -- 10% 奖金
        );
    END LOOP;

    -- 关闭游标
    CLOSE emp_cursor;

    -- 返回结果
    SELECT * FROM cursor_results;
END//

-- 复杂游标示例：生成报表
CREATE PROCEDURE generate_dept_report()
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE dept_id INT;
    DECLARE dept_name VARCHAR(50);

    DECLARE dept_cursor CURSOR FOR
        SELECT id, name FROM departments;

    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;

    -- 创建报表临时表
    DROP TEMPORARY TABLE IF EXISTS dept_report;
    CREATE TEMPORARY TABLE dept_report (
        department VARCHAR(50),
        employee_count INT,
        total_salary DECIMAL(12, 2),
        avg_salary DECIMAL(10, 2),
        max_salary DECIMAL(10, 2),
        min_salary DECIMAL(10, 2)
    );

    OPEN dept_cursor;

    dept_loop: LOOP
        FETCH dept_cursor INTO dept_id, dept_name;

        IF done THEN
            LEAVE dept_loop;
        END IF;

        INSERT INTO dept_report
        SELECT
            dept_name,
            COUNT(*),
            SUM(salary),
            AVG(salary),
            MAX(salary),
            MIN(salary)
        FROM employees
        WHERE department_id = dept_id;
    END LOOP;

    CLOSE dept_cursor;

    SELECT * FROM dept_report;
END//

DELIMITER ;

-- 测试游标
CALL cursor_demo();
CALL generate_dept_report();


-- ============================================================
--                    5. 自定义函数
-- ============================================================

DELIMITER //

-- 标量函数
CREATE FUNCTION get_full_name(
    first_name VARCHAR(50),
    last_name VARCHAR(50)
)
RETURNS VARCHAR(100)
DETERMINISTIC  -- 相同输入总是返回相同结果
BEGIN
    RETURN CONCAT(first_name, ' ', last_name);
END//

-- 计算函数
CREATE FUNCTION calculate_tax(
    salary DECIMAL(10, 2)
)
RETURNS DECIMAL(10, 2)
DETERMINISTIC
BEGIN
    DECLARE tax DECIMAL(10, 2);

    IF salary <= 5000 THEN
        SET tax = 0;
    ELSEIF salary <= 10000 THEN
        SET tax = (salary - 5000) * 0.1;
    ELSEIF salary <= 20000 THEN
        SET tax = 500 + (salary - 10000) * 0.2;
    ELSE
        SET tax = 2500 + (salary - 20000) * 0.3;
    END IF;

    RETURN tax;
END//

-- 查询相关函数
CREATE FUNCTION get_department_name(
    dept_id INT
)
RETURNS VARCHAR(50)
READS SQL DATA  -- 函数读取数据但不修改
BEGIN
    DECLARE dept_name VARCHAR(50);

    SELECT name INTO dept_name
    FROM departments
    WHERE id = dept_id;

    RETURN COALESCE(dept_name, '未分配');
END//

-- 业务逻辑函数
CREATE FUNCTION get_employee_level(
    emp_id INT
)
RETURNS VARCHAR(20)
READS SQL DATA
BEGIN
    DECLARE emp_salary DECIMAL(10, 2);
    DECLARE hire_years INT;

    SELECT salary, TIMESTAMPDIFF(YEAR, hire_date, CURDATE())
    INTO emp_salary, hire_years
    FROM employees WHERE id = emp_id;

    IF emp_salary IS NULL THEN
        RETURN '未知';
    END IF;

    IF emp_salary >= 25000 AND hire_years >= 5 THEN
        RETURN '资深专家';
    ELSEIF emp_salary >= 20000 AND hire_years >= 3 THEN
        RETURN '高级';
    ELSEIF emp_salary >= 15000 THEN
        RETURN '中级';
    ELSE
        RETURN '初级';
    END IF;
END//

DELIMITER ;

-- 使用函数
SELECT get_full_name('John', 'Doe') AS full_name;

SELECT
    name,
    salary,
    calculate_tax(salary) AS tax,
    salary - calculate_tax(salary) AS net_salary
FROM employees;

SELECT
    name,
    get_department_name(department_id) AS department,
    get_employee_level(id) AS level
FROM employees;


-- ============================================================
--                    6. 触发器
-- ============================================================

-- 创建日志表
CREATE TABLE IF NOT EXISTS employee_audit (
    id INT AUTO_INCREMENT PRIMARY KEY,
    employee_id INT,
    action VARCHAR(10),
    old_data JSON,
    new_data JSON,
    changed_by VARCHAR(50),
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

DELIMITER //

-- BEFORE INSERT 触发器
CREATE TRIGGER trg_employee_before_insert
BEFORE INSERT ON employees
FOR EACH ROW
BEGIN
    -- 自动设置创建时间
    IF NEW.created_at IS NULL THEN
        SET NEW.created_at = NOW();
    END IF;

    -- 数据验证
    IF NEW.salary < 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = '薪资不能为负数';
    END IF;
END//

-- AFTER INSERT 触发器
CREATE TRIGGER trg_employee_after_insert
AFTER INSERT ON employees
FOR EACH ROW
BEGIN
    INSERT INTO employee_audit (employee_id, action, new_data, changed_by)
    VALUES (
        NEW.id,
        'INSERT',
        JSON_OBJECT('name', NEW.name, 'salary', NEW.salary),
        CURRENT_USER()
    );
END//

-- BEFORE UPDATE 触发器
CREATE TRIGGER trg_employee_before_update
BEFORE UPDATE ON employees
FOR EACH ROW
BEGIN
    -- 限制薪资涨幅不超过 50%
    IF NEW.salary > OLD.salary * 1.5 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = '薪资涨幅不能超过50%';
    END IF;
END//

-- AFTER UPDATE 触发器
CREATE TRIGGER trg_employee_after_update
AFTER UPDATE ON employees
FOR EACH ROW
BEGIN
    INSERT INTO employee_audit (employee_id, action, old_data, new_data, changed_by)
    VALUES (
        NEW.id,
        'UPDATE',
        JSON_OBJECT('name', OLD.name, 'salary', OLD.salary),
        JSON_OBJECT('name', NEW.name, 'salary', NEW.salary),
        CURRENT_USER()
    );
END//

-- AFTER DELETE 触发器
CREATE TRIGGER trg_employee_after_delete
AFTER DELETE ON employees
FOR EACH ROW
BEGIN
    INSERT INTO employee_audit (employee_id, action, old_data, changed_by)
    VALUES (
        OLD.id,
        'DELETE',
        JSON_OBJECT('name', OLD.name, 'salary', OLD.salary),
        CURRENT_USER()
    );
END//

DELIMITER ;

-- 查看触发器
SHOW TRIGGERS;

SELECT * FROM information_schema.TRIGGERS
WHERE TRIGGER_SCHEMA = 'learn_mysql';


-- ============================================================
--                    7. 错误处理
-- ============================================================

DELIMITER //

CREATE PROCEDURE error_handling_demo()
BEGIN
    -- 声明变量
    DECLARE error_occurred BOOLEAN DEFAULT FALSE;
    DECLARE error_message VARCHAR(255);

    -- 声明错误处理器
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        GET DIAGNOSTICS CONDITION 1
            error_message = MESSAGE_TEXT;
        SET error_occurred = TRUE;
        ROLLBACK;
        SELECT CONCAT('错误: ', error_message) AS error;
    END;

    -- 特定错误处理
    DECLARE CONTINUE HANDLER FOR 1062  -- 重复键错误
    BEGIN
        SET error_message = '记录已存在';
    END;

    START TRANSACTION;

    -- 尝试插入数据
    INSERT INTO employees (name, email, salary, hire_date)
    VALUES ('Test', 'test@test.com', 10000, CURDATE());

    IF error_occurred THEN
        SELECT '操作失败' AS result;
    ELSE
        COMMIT;
        SELECT '操作成功' AS result;
    END IF;
END//

-- 使用 SIGNAL 抛出自定义错误
CREATE PROCEDURE validate_and_insert(
    IN emp_name VARCHAR(50),
    IN emp_salary DECIMAL(10, 2)
)
BEGIN
    IF emp_name IS NULL OR emp_name = '' THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = '员工姓名不能为空';
    END IF;

    IF emp_salary < 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = '薪资不能为负数',
        MYSQL_ERRNO = 1644;
    END IF;

    INSERT INTO employees (name, email, salary, hire_date)
    VALUES (emp_name, CONCAT(emp_name, '@company.com'), emp_salary, CURDATE());

    SELECT '插入成功' AS result;
END//

DELIMITER ;


-- ============================================================
--                    8. 管理存储过程和函数
-- ============================================================

-- 查看存储过程
SHOW PROCEDURE STATUS WHERE Db = 'learn_mysql';

-- 查看函数
SHOW FUNCTION STATUS WHERE Db = 'learn_mysql';

-- 查看存储过程创建语句
SHOW CREATE PROCEDURE get_employee_by_id;

-- 查看函数创建语句
SHOW CREATE FUNCTION calculate_tax;

-- 删除存储过程
DROP PROCEDURE IF EXISTS variable_demo;

-- 删除函数
DROP FUNCTION IF EXISTS get_full_name;

-- 删除触发器
DROP TRIGGER IF EXISTS trg_employee_before_insert;


-- ============================================================
--                    总结
-- ============================================================

/*
存储过程：
- DELIMITER 更改结束符
- IN/OUT/INOUT 参数
- CALL 调用

变量与流程：
- DECLARE 声明变量
- SET / SELECT INTO 赋值
- IF/ELSEIF/ELSE/END IF
- CASE/WHEN/THEN/END CASE
- WHILE/REPEAT/LOOP

游标：
- DECLARE CURSOR FOR
- OPEN/FETCH/CLOSE
- HANDLER FOR NOT FOUND

函数：
- CREATE FUNCTION ... RETURNS
- DETERMINISTIC / READS SQL DATA
- 可在 SQL 中使用

触发器：
- BEFORE/AFTER INSERT/UPDATE/DELETE
- NEW.column / OLD.column
- 用于审计、验证、级联操作

错误处理：
- DECLARE HANDLER
- SIGNAL SQLSTATE
- GET DIAGNOSTICS

最佳实践：
- 保持存储过程简洁
- 使用事务保证一致性
- 适当的错误处理
- 注意性能影响
- 文档化复杂逻辑
*/
