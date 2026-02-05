#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
                    Python 闭包与作用域
============================================================
本文件介绍 Python 中的作用域规则和闭包（Closure）概念。

闭包：内部函数引用了外部函数的变量，并且外部函数返回了内部函数。
============================================================
"""


def main01_scope():
    """
    ============================================================
                    1. 变量作用域 LEGB 规则
    ============================================================
    Python 变量查找顺序：
    L - Local: 局部作用域（函数内部）
    E - Enclosing: 闭包作用域（外层函数）
    G - Global: 全局作用域（模块级别）
    B - Built-in: 内置作用域（Python 内置）
    """
    print("=" * 60)
    print("1. 变量作用域 LEGB 规则")
    print("=" * 60)

    # 全局变量
    global_var = "我是全局变量"

    def outer():
        # 闭包变量（外层函数的局部变量）
        enclosing_var = "我是闭包变量"

        def inner():
            # 局部变量
            local_var = "我是局部变量"
            # 按 LEGB 顺序查找
            print(f"  Local: {local_var}")
            print(f"  Enclosing: {enclosing_var}")
            print(f"  Global: {global_var}")
            print(f"  Built-in: {len}")  # 内置函数

        inner()

    outer()

    # 【演示变量遮蔽】
    print("\n--- 变量遮蔽 ---")
    x = "全局 x"

    def shadow_demo():
        x = "局部 x"  # 遮蔽全局变量
        print(f"函数内的 x: {x}")

    shadow_demo()
    print(f"全局的 x: {x}")  # 全局变量不受影响


def main02_global_nonlocal():
    """
    ============================================================
                2. global 和 nonlocal 关键字
    ============================================================
    """
    print("\n" + "=" * 60)
    print("2. global 和 nonlocal 关键字")
    print("=" * 60)

    # 【global】修改全局变量
    print("--- global 关键字 ---")

    counter = 0

    def increment():
        global counter  # 声明使用全局变量
        counter += 1
        print(f"counter = {counter}")

    increment()
    increment()
    print(f"最终 counter = {counter}")

    # 【警告】滥用 global 会导致代码难以维护
    # 推荐使用类或函数参数/返回值

    # 【nonlocal】修改闭包变量
    print("\n--- nonlocal 关键字 ---")

    def make_counter():
        count = 0

        def increment():
            nonlocal count  # 声明使用外层函数的变量
            count += 1
            return count

        return increment

    counter1 = make_counter()
    counter2 = make_counter()

    print(f"counter1: {counter1()}, {counter1()}, {counter1()}")
    print(f"counter2: {counter2()}, {counter2()}")  # 独立的计数器

    # 【对比：没有 nonlocal 会怎样】
    print("\n--- 没有 nonlocal ---")

    def broken_counter():
        count = 0

        def increment():
            # count += 1  # 报错！UnboundLocalError
            # 因为 count = 在左边，Python 认为是局部变量
            local_count = count + 1  # 只能读取，不能修改
            return local_count

        return increment


def main03_closure_basics():
    """
    ============================================================
                        3. 闭包基础
    ============================================================
    闭包 = 内部函数 + 对外部变量的引用 + 外部函数返回内部函数
    """
    print("\n" + "=" * 60)
    print("3. 闭包基础")
    print("=" * 60)

    # 【基本闭包】
    def make_multiplier(factor):
        """创建一个乘法器"""
        def multiply(n):
            return n * factor  # 引用外部变量 factor
        return multiply

    double = make_multiplier(2)
    triple = make_multiplier(3)

    print(f"double(5) = {double(5)}")
    print(f"triple(5) = {triple(5)}")

    # 【查看闭包变量】
    print(f"\ndouble.__closure__: {double.__closure__}")
    print(f"闭包变量值: {double.__closure__[0].cell_contents}")

    # 【闭包保持状态】
    def make_accumulator(start=0):
        """创建一个累加器"""
        total = start

        def add(n):
            nonlocal total
            total += n
            return total

        return add

    acc = make_accumulator(100)
    print(f"\n累加器: {acc(10)}, {acc(20)}, {acc(30)}")

    # 【闭包作为配置】
    def make_logger(prefix):
        """创建一个带前缀的日志记录器"""
        def log(message):
            print(f"[{prefix}] {message}")
        return log

    error_log = make_logger("ERROR")
    info_log = make_logger("INFO")

    print("\n日志记录器:")
    error_log("Something went wrong!")
    info_log("Everything is fine.")


def main04_closure_trap():
    """
    ============================================================
                    4. 闭包陷阱（常见错误）
    ============================================================
    """
    print("\n" + "=" * 60)
    print("4. 闭包陷阱（常见错误）")
    print("=" * 60)

    # 【陷阱：循环中的闭包】
    print("--- 陷阱：循环中的闭包 ---")

    # 错误示例
    def create_functions_wrong():
        functions = []
        for i in range(3):
            def func():
                return i  # 闭包捕获的是变量 i，不是值！
            functions.append(func)
        return functions

    funcs = create_functions_wrong()
    print("错误结果（都返回最后的值）:")
    for f in funcs:
        print(f"  {f()}")  # 全部输出 2！

    # 【原因分析】
    # 所有闭包共享同一个变量 i
    # 循环结束时 i = 2，所以所有闭包返回 2

    # 【解决方案1：使用默认参数】
    print("\n解决方案1：默认参数")

    def create_functions_fixed1():
        functions = []
        for i in range(3):
            def func(i=i):  # 默认参数在定义时求值
                return i
            functions.append(func)
        return functions

    funcs = create_functions_fixed1()
    for f in funcs:
        print(f"  {f()}")

    # 【解决方案2：使用工厂函数】
    print("\n解决方案2：工厂函数")

    def create_functions_fixed2():
        def make_func(i):
            def func():
                return i
            return func

        functions = []
        for i in range(3):
            functions.append(make_func(i))
        return functions

    funcs = create_functions_fixed2()
    for f in funcs:
        print(f"  {f()}")

    # 【解决方案3：使用 lambda + 默认参数】
    print("\n解决方案3：lambda + 默认参数")
    funcs = [lambda i=i: i for i in range(3)]
    for f in funcs:
        print(f"  {f()}")

    # 【解决方案4：使用 functools.partial】
    print("\n解决方案4：functools.partial")
    from functools import partial

    def return_value(i):
        return i

    funcs = [partial(return_value, i) for i in range(3)]
    for f in funcs:
        print(f"  {f()}")


def main05_practical_closures():
    """
    ============================================================
                    5. 闭包的实际应用
    ============================================================
    """
    print("\n" + "=" * 60)
    print("5. 闭包的实际应用")
    print("=" * 60)

    # 【应用1：计数器】
    print("--- 应用1：计数器 ---")

    def make_counter():
        count = 0

        def counter():
            nonlocal count
            count += 1
            return count

        def get_count():
            return count

        def reset():
            nonlocal count
            count = 0

        counter.get = get_count
        counter.reset = reset
        return counter

    c = make_counter()
    print(f"调用: {c()}, {c()}, {c()}")
    print(f"获取: {c.get()}")
    c.reset()
    print(f"重置后: {c()}")

    # 【应用2：缓存/记忆化】
    print("\n--- 应用2：缓存/记忆化 ---")

    def memoize(func):
        cache = {}

        def wrapper(*args):
            if args not in cache:
                cache[args] = func(*args)
                print(f"  计算 {args} = {cache[args]}")
            else:
                print(f"  缓存命中 {args} = {cache[args]}")
            return cache[args]

        wrapper.cache = cache
        return wrapper

    @memoize
    def fibonacci(n):
        if n < 2:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    print("计算 fibonacci(5):")
    result = fibonacci(5)
    print(f"结果: {result}")

    # 【应用3：延迟求值】
    print("\n--- 应用3：延迟求值 ---")

    def lazy(func):
        """创建一个延迟求值的包装器"""
        result = None
        computed = False

        def wrapper():
            nonlocal result, computed
            if not computed:
                result = func()
                computed = True
            return result

        return wrapper

    @lazy
    def expensive_computation():
        print("  执行昂贵的计算...")
        return sum(range(1000))

    print("第一次调用:")
    print(f"  结果: {expensive_computation()}")
    print("第二次调用（使用缓存）:")
    print(f"  结果: {expensive_computation()}")

    # 【应用4：配置工厂】
    print("\n--- 应用4：配置工厂 ---")

    def create_validator(min_val, max_val):
        """创建一个范围验证器"""
        def validate(value):
            if min_val <= value <= max_val:
                return True, f"{value} 在范围 [{min_val}, {max_val}] 内"
            return False, f"{value} 超出范围 [{min_val}, {max_val}]"

        return validate

    age_validator = create_validator(0, 150)
    score_validator = create_validator(0, 100)

    print(f"年龄验证 25: {age_validator(25)}")
    print(f"年龄验证 200: {age_validator(200)}")
    print(f"分数验证 85: {score_validator(85)}")

    # 【应用5：私有数据】
    print("\n--- 应用5：私有数据 ---")

    def create_bank_account(initial_balance):
        """创建一个银行账户（数据私有化）"""
        balance = initial_balance

        def deposit(amount):
            nonlocal balance
            if amount > 0:
                balance += amount
                return f"存入 {amount}，余额: {balance}"
            return "存款金额必须为正"

        def withdraw(amount):
            nonlocal balance
            if 0 < amount <= balance:
                balance -= amount
                return f"取出 {amount}，余额: {balance}"
            return "取款失败"

        def get_balance():
            return balance

        return {
            "deposit": deposit,
            "withdraw": withdraw,
            "get_balance": get_balance
        }

    account = create_bank_account(1000)
    print(account["deposit"](500))
    print(account["withdraw"](200))
    print(f"当前余额: {account['get_balance']()}")
    # balance 变量无法直接访问，实现了数据私有化


def main06_closure_vs_class():
    """
    ============================================================
                    6. 闭包 vs 类
    ============================================================
    闭包和类都可以用来维护状态，各有优劣
    """
    print("\n" + "=" * 60)
    print("6. 闭包 vs 类")
    print("=" * 60)

    # 【使用闭包实现计数器】
    print("--- 闭包实现 ---")

    def make_counter_closure():
        count = 0

        def counter():
            nonlocal count
            count += 1
            return count

        return counter

    counter1 = make_counter_closure()
    print(f"闭包计数器: {counter1()}, {counter1()}")

    # 【使用类实现计数器】
    print("\n--- 类实现 ---")

    class Counter:
        def __init__(self):
            self.count = 0

        def __call__(self):
            self.count += 1
            return self.count

    counter2 = Counter()
    print(f"类计数器: {counter2()}, {counter2()}")

    # 【比较】
    print("\n--- 比较 ---")
    print("""
    闭包优点：
    - 代码简洁
    - 真正的数据私有化
    - 适合简单场景

    类优点：
    - 更好的可扩展性
    - 支持继承
    - 更适合复杂场景
    - 代码更易理解（对于大多数人）
    """)


if __name__ == "__main__":
    main01_scope()
    main02_global_nonlocal()
    main03_closure_basics()
    main04_closure_trap()
    main05_practical_closures()
    main06_closure_vs_class()
