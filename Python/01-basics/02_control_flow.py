#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
                    Python 控制流
============================================================
本文件介绍 Python 中的条件语句、循环、异常处理等控制流结构。
============================================================
"""


def main01_if_else():
    """
    ============================================================
                    1. 条件语句 if/elif/else
    ============================================================
    """
    print("=" * 60)
    print("1. 条件语句 if/elif/else")
    print("=" * 60)

    # 【基本 if 语句】
    age = 18
    if age >= 18:
        print(f"年龄 {age}：成年人")

    # 【if-else】
    score = 75
    if score >= 60:
        print(f"分数 {score}：及格")
    else:
        print(f"分数 {score}：不及格")

    # 【if-elif-else】
    score = 85
    if score >= 90:
        grade = 'A'
    elif score >= 80:
        grade = 'B'
    elif score >= 70:
        grade = 'C'
    elif score >= 60:
        grade = 'D'
    else:
        grade = 'F'
    print(f"分数 {score} 的等级是: {grade}")

    # 【三元表达式】（条件表达式）
    x = 10
    result = "正数" if x > 0 else "非正数"
    print(f"\n三元表达式: x={x}, 结果: {result}")

    # 嵌套三元表达式（不推荐，可读性差）
    x = 0
    result = "正数" if x > 0 else ("零" if x == 0 else "负数")
    print(f"嵌套三元: x={x}, 结果: {result}")

    # 【海象运算符 :=】（Python 3.8+）
    # 在表达式中同时赋值和使用
    print("\n--- 海象运算符 := ---")
    data = [1, 2, 3, 4, 5]
    if (n := len(data)) > 3:
        print(f"列表长度 {n} 大于 3")

    # 【技巧】利用短路求值
    print("\n--- 短路求值 ---")
    name = ""
    display_name = name or "匿名用户"  # 空字符串为 False
    print(f"显示名称: {display_name}")

    # and 返回第一个 False 值或最后一个值
    result = 1 and 2 and 3
    print(f"1 and 2 and 3 = {result}")  # 3

    # or 返回第一个 True 值或最后一个值
    result = 0 or "" or "default"
    print(f"0 or '' or 'default' = {result}")  # default


def main02_match():
    """
    ============================================================
                    2. 模式匹配 match（Python 3.10+）
    ============================================================
    【重要】match 是 Python 3.10 新增的结构化模式匹配
    比 switch/case 更强大！
    """
    print("\n" + "=" * 60)
    print("2. 模式匹配 match（Python 3.10+）")
    print("=" * 60)

    # 【基本 match】
    def get_day_type(day):
        match day:
            case "Saturday" | "Sunday":  # 使用 | 匹配多个值
                return "周末"
            case "Monday" | "Tuesday" | "Wednesday" | "Thursday" | "Friday":
                return "工作日"
            case _:  # 通配符，匹配所有
                return "未知"

    print(f"Saturday: {get_day_type('Saturday')}")
    print(f"Monday: {get_day_type('Monday')}")

    # 【序列模式匹配】
    def process_point(point):
        match point:
            case (0, 0):
                return "原点"
            case (0, y):
                return f"Y轴上，y={y}"
            case (x, 0):
                return f"X轴上，x={x}"
            case (x, y):
                return f"点 ({x}, {y})"
            case _:
                return "不是有效的点"

    print(f"\n(0, 0): {process_point((0, 0))}")
    print(f"(0, 5): {process_point((0, 5))}")
    print(f"(3, 4): {process_point((3, 4))}")

    # 【字典模式匹配】
    def process_command(command):
        match command:
            case {"action": "quit"}:
                return "退出程序"
            case {"action": "move", "direction": direction}:
                return f"移动方向: {direction}"
            case {"action": "attack", "target": target, "damage": damage}:
                return f"攻击 {target}，伤害 {damage}"
            case _:
                return "未知命令"

    print(f"\n命令1: {process_command({'action': 'quit'})}")
    print(f"命令2: {process_command({'action': 'move', 'direction': 'north'})}")
    print(f"命令3: {process_command({'action': 'attack', 'target': 'dragon', 'damage': 100})}")

    # 【守卫条件】使用 if 添加额外条件
    def check_number(n):
        match n:
            case x if x < 0:
                return "负数"
            case x if x == 0:
                return "零"
            case x if x < 10:
                return "个位正数"
            case _:
                return "大于等于10"

    print(f"\n-5: {check_number(-5)}")
    print(f"0: {check_number(0)}")
    print(f"7: {check_number(7)}")
    print(f"15: {check_number(15)}")


def main03_for_loop():
    """
    ============================================================
                        3. for 循环
    ============================================================
    Python 的 for 循环是 for-each 风格，遍历可迭代对象
    """
    print("\n" + "=" * 60)
    print("3. for 循环")
    print("=" * 60)

    # 【遍历列表】
    fruits = ["apple", "banana", "cherry"]
    print("遍历列表:")
    for fruit in fruits:
        print(f"  {fruit}")

    # 【遍历字符串】
    print("\n遍历字符串:")
    for char in "Hello":
        print(f"  {char}")

    # 【range 函数】
    print("\n使用 range:")
    print("range(5):", list(range(5)))           # [0, 1, 2, 3, 4]
    print("range(2, 5):", list(range(2, 5)))     # [2, 3, 4]
    print("range(0, 10, 2):", list(range(0, 10, 2)))  # [0, 2, 4, 6, 8]
    print("range(5, 0, -1):", list(range(5, 0, -1)))  # [5, 4, 3, 2, 1]

    # 【enumerate】同时获取索引和值
    print("\n使用 enumerate:")
    for i, fruit in enumerate(fruits):
        print(f"  {i}: {fruit}")

    # 指定起始索引
    print("\nenumerate 指定起始:")
    for i, fruit in enumerate(fruits, start=1):
        print(f"  {i}: {fruit}")

    # 【zip】并行遍历多个序列
    print("\n使用 zip:")
    names = ["Alice", "Bob", "Charlie"]
    ages = [25, 30, 35]
    for name, age in zip(names, ages):
        print(f"  {name}: {age}岁")

    # 【技巧】zip_longest 处理不等长序列
    from itertools import zip_longest
    a = [1, 2, 3]
    b = ['a', 'b']
    print("\nzip_longest:")
    for x, y in zip_longest(a, b, fillvalue='N/A'):
        print(f"  ({x}, {y})")

    # 【遍历字典】
    print("\n遍历字典:")
    person = {"name": "Alice", "age": 25, "city": "Beijing"}

    print("遍历键:")
    for key in person:
        print(f"  {key}")

    print("遍历键值对:")
    for key, value in person.items():
        print(f"  {key}: {value}")

    print("遍历值:")
    for value in person.values():
        print(f"  {value}")

    # 【列表推导式】（更 Pythonic！）
    print("\n列表推导式:")
    squares = [x**2 for x in range(5)]
    print(f"平方: {squares}")

    evens = [x for x in range(10) if x % 2 == 0]
    print(f"偶数: {evens}")

    # 嵌套推导式
    matrix = [[i*j for j in range(1, 4)] for i in range(1, 4)]
    print(f"乘法表: {matrix}")

    # 【字典推导式】
    print("\n字典推导式:")
    squares_dict = {x: x**2 for x in range(5)}
    print(f"平方字典: {squares_dict}")

    # 【集合推导式】
    print("\n集合推导式:")
    squares_set = {x**2 for x in range(-3, 4)}
    print(f"平方集合: {squares_set}")

    # 【生成器表达式】（节省内存）
    print("\n生成器表达式:")
    gen = (x**2 for x in range(5))
    print(f"生成器: {gen}")
    print(f"转为列表: {list(gen)}")


def main04_while_loop():
    """
    ============================================================
                        4. while 循环
    ============================================================
    """
    print("\n" + "=" * 60)
    print("4. while 循环")
    print("=" * 60)

    # 【基本 while】
    print("基本 while:")
    count = 0
    while count < 5:
        print(f"  count = {count}")
        count += 1

    # 【while-else】
    # 【重要】else 在循环正常结束（没有 break）时执行
    print("\nwhile-else (正常结束):")
    n = 0
    while n < 3:
        print(f"  n = {n}")
        n += 1
    else:
        print("  循环正常结束！")

    print("\nwhile-else (break 退出):")
    n = 0
    while n < 10:
        if n == 3:
            print("  找到 3，退出！")
            break
        n += 1
    else:
        print("  这行不会执行")

    # 【无限循环】
    print("\n无限循环示例:")
    count = 0
    while True:
        count += 1
        if count >= 3:
            print(f"  达到 {count}，退出")
            break


def main05_break_continue():
    """
    ============================================================
                    5. break 和 continue
    ============================================================
    """
    print("\n" + "=" * 60)
    print("5. break 和 continue")
    print("=" * 60)

    # 【break】跳出整个循环
    print("break 示例:")
    for i in range(10):
        if i == 5:
            print(f"  i={i}, 跳出循环")
            break
        print(f"  i={i}")

    # 【continue】跳过本次迭代
    print("\ncontinue 示例:")
    for i in range(5):
        if i == 2:
            print(f"  i={i}, 跳过")
            continue
        print(f"  i={i}")

    # 【for-else】
    print("\nfor-else (查找示例):")

    def find_item(items, target):
        for item in items:
            if item == target:
                print(f"  找到 {target}!")
                break
        else:
            print(f"  未找到 {target}")

    find_item([1, 2, 3, 4, 5], 3)
    find_item([1, 2, 3, 4, 5], 10)


def main06_pass():
    """
    ============================================================
                        6. pass 语句
    ============================================================
    pass 是空操作，用作占位符
    """
    print("\n" + "=" * 60)
    print("6. pass 语句")
    print("=" * 60)

    # 【用于空函数】
    def not_implemented_yet():
        pass  # TODO: 以后实现

    # 【用于空类】
    class EmptyClass:
        pass

    # 【用于空的条件分支】
    x = 10
    if x > 0:
        pass  # 暂时不处理
    else:
        print("负数")

    # 【技巧】使用 ... 代替 pass（更现代的写法）
    def another_placeholder():
        ...  # 等价于 pass

    print("pass 和 ... 都可用作占位符")


def main07_exception():
    """
    ============================================================
                    7. 异常处理 try/except
    ============================================================
    """
    print("\n" + "=" * 60)
    print("7. 异常处理 try/except")
    print("=" * 60)

    # 【基本 try-except】
    print("基本异常处理:")
    try:
        result = 10 / 0
    except ZeroDivisionError:
        print("  捕获到除零错误！")

    # 【捕获多种异常】
    print("\n捕获多种异常:")
    try:
        # 可能引发不同类型的异常
        num = int("abc")
    except ValueError:
        print("  捕获到 ValueError")
    except TypeError:
        print("  捕获到 TypeError")

    # 【捕获多种异常（合并）】
    print("\n合并捕获多种异常:")
    try:
        num = int("abc")
    except (ValueError, TypeError) as e:
        print(f"  捕获到异常: {type(e).__name__}: {e}")

    # 【获取异常信息】
    print("\n获取异常信息:")
    try:
        result = 10 / 0
    except ZeroDivisionError as e:
        print(f"  异常类型: {type(e).__name__}")
        print(f"  异常信息: {e}")

    # 【try-except-else】
    # else 在没有异常时执行
    print("\ntry-except-else:")
    try:
        result = 10 / 2
    except ZeroDivisionError:
        print("  除零错误")
    else:
        print(f"  计算成功: {result}")

    # 【try-except-finally】
    # finally 无论如何都执行
    print("\ntry-except-finally:")
    try:
        result = 10 / 2
    except ZeroDivisionError:
        print("  除零错误")
    finally:
        print("  清理工作（finally 总是执行）")

    # 【完整结构】
    print("\n完整的异常处理结构:")
    try:
        result = 10 / 2
    except ZeroDivisionError:
        print("  除零错误")
    except Exception as e:
        print(f"  其他错误: {e}")
    else:
        print(f"  成功: {result}")
    finally:
        print("  清理完成")

    # 【raise】主动抛出异常
    print("\nraise 抛出异常:")

    def check_age(age):
        if age < 0:
            raise ValueError("年龄不能为负数")
        return age

    try:
        check_age(-5)
    except ValueError as e:
        print(f"  捕获到: {e}")

    # 【自定义异常】
    print("\n自定义异常:")

    class MyError(Exception):
        """自定义异常类"""
        def __init__(self, message, code):
            super().__init__(message)
            self.code = code

    try:
        raise MyError("自定义错误", code=404)
    except MyError as e:
        print(f"  异常: {e}, 错误码: {e.code}")

    # 【异常链】
    print("\n异常链:")
    try:
        try:
            result = 10 / 0
        except ZeroDivisionError as e:
            raise ValueError("计算失败") from e
    except ValueError as e:
        print(f"  捕获: {e}")
        print(f"  原因: {e.__cause__}")


def main08_context_manager():
    """
    ============================================================
                    8. 上下文管理器 with
    ============================================================
    with 语句用于资源管理，确保资源被正确释放
    """
    print("\n" + "=" * 60)
    print("8. 上下文管理器 with")
    print("=" * 60)

    # 【文件操作】
    print("文件操作示例:")
    # 传统方式（需要手动关闭）
    # f = open('file.txt')
    # try:
    #     content = f.read()
    # finally:
    #     f.close()

    # 使用 with（推荐）
    # with open('file.txt') as f:
    #     content = f.read()
    # 文件自动关闭
    print("  with open('file.txt') as f: ...")

    # 【多个上下文管理器】
    print("\n多个上下文管理器:")
    # with open('input.txt') as fin, open('output.txt', 'w') as fout:
    #     fout.write(fin.read())
    print("  with open('a') as f1, open('b') as f2: ...")

    # 【自定义上下文管理器】
    print("\n自定义上下文管理器:")

    class MyContext:
        def __enter__(self):
            print("    进入上下文")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            print("    退出上下文")
            if exc_type:
                print(f"    处理异常: {exc_val}")
            return False  # 不抑制异常

        def do_something(self):
            print("    执行操作")

    with MyContext() as ctx:
        ctx.do_something()

    # 【使用 contextlib】
    print("\n使用 contextlib:")
    from contextlib import contextmanager

    @contextmanager
    def my_context():
        print("    设置资源")
        try:
            yield "资源对象"
        finally:
            print("    清理资源")

    with my_context() as resource:
        print(f"    使用: {resource}")


if __name__ == "__main__":
    main01_if_else()
    main02_match()
    main03_for_loop()
    main04_while_loop()
    main05_break_continue()
    main06_pass()
    main07_exception()
    main08_context_manager()
