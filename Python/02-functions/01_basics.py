#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
                    Python 函数基础
============================================================
本文件介绍 Python 函数的定义、参数、返回值等基本概念。

Python 中函数是一等公民（First-class Citizen）：
- 可以赋值给变量
- 可以作为参数传递
- 可以作为返回值
- 可以存储在数据结构中
============================================================
"""


def main01_function_definition():
    """
    ============================================================
                    1. 函数定义与调用
    ============================================================
    """
    print("=" * 60)
    print("1. 函数定义与调用")
    print("=" * 60)

    # 【基本函数定义】
    def greet():
        """简单的问候函数（这是文档字符串 docstring）"""
        print("Hello, World!")

    greet()

    # 【带参数的函数】
    def greet_person(name):
        """向指定的人问候"""
        print(f"Hello, {name}!")

    greet_person("Alice")

    # 【带返回值的函数】
    def add(a, b):
        """返回两个数的和"""
        return a + b

    result = add(3, 5)
    print(f"add(3, 5) = {result}")

    # 【多个 return 语句】
    def absolute(n):
        """返回绝对值"""
        if n >= 0:
            return n
        return -n

    print(f"absolute(-5) = {absolute(-5)}")

    # 【没有 return 语句返回 None】
    def do_nothing():
        pass

    result = do_nothing()
    print(f"do_nothing() 返回: {result}")

    # 【函数文档字符串】
    def well_documented_function(param1, param2):
        """
        这是一个有良好文档的函数。

        Args:
            param1: 第一个参数的描述
            param2: 第二个参数的描述

        Returns:
            返回值的描述

        Raises:
            ValueError: 当参数无效时抛出

        Examples:
            >>> well_documented_function(1, 2)
            3
        """
        return param1 + param2

    # 查看文档
    print(f"\n函数文档: {well_documented_function.__doc__[:50]}...")
    help(well_documented_function)


def main02_parameters():
    """
    ============================================================
                    2. 函数参数类型
    ============================================================
    """
    print("\n" + "=" * 60)
    print("2. 函数参数类型")
    print("=" * 60)

    # 【位置参数】按顺序传递
    def power(base, exponent):
        return base ** exponent

    print(f"power(2, 3) = {power(2, 3)}")

    # 【关键字参数】按名称传递
    print(f"power(exponent=3, base=2) = {power(exponent=3, base=2)}")

    # 【默认参数】
    def greet(name, greeting="Hello"):
        return f"{greeting}, {name}!"

    print(f"\ngreet('Alice') = {greet('Alice')}")
    print(f"greet('Alice', 'Hi') = {greet('Alice', 'Hi')}")

    # 【警告】默认参数陷阱 - 可变默认值！
    def bad_append(item, lst=[]):  # 不要这样做！
        lst.append(item)
        return lst

    print(f"\n【警告】可变默认参数陷阱:")
    print(f"bad_append(1): {bad_append(1)}")
    print(f"bad_append(2): {bad_append(2)}")  # [1, 2]! 不是 [2]!

    # 【正确做法】使用 None 作为默认值
    def good_append(item, lst=None):
        if lst is None:
            lst = []
        lst.append(item)
        return lst

    print(f"\n正确做法:")
    print(f"good_append(1): {good_append(1)}")
    print(f"good_append(2): {good_append(2)}")  # [2]

    # 【仅位置参数】Python 3.8+
    # / 之前的参数只能通过位置传递
    def positional_only(a, b, /, c):
        return a + b + c

    print(f"\n仅位置参数: positional_only(1, 2, c=3) = {positional_only(1, 2, c=3)}")
    # positional_only(a=1, b=2, c=3)  # 报错！

    # 【仅关键字参数】
    # * 之后的参数只能通过关键字传递
    def keyword_only(a, *, b, c):
        return a + b + c

    print(f"仅关键字参数: keyword_only(1, b=2, c=3) = {keyword_only(1, b=2, c=3)}")
    # keyword_only(1, 2, 3)  # 报错！

    # 【组合使用】
    def combined(pos_only, /, standard, *, kw_only):
        return f"{pos_only}, {standard}, {kw_only}"

    print(f"组合: {combined(1, 2, kw_only=3)}")
    print(f"组合: {combined(1, standard=2, kw_only=3)}")


def main03_variadic():
    """
    ============================================================
                    3. 可变参数 *args 和 **kwargs
    ============================================================
    """
    print("\n" + "=" * 60)
    print("3. 可变参数 *args 和 **kwargs")
    print("=" * 60)

    # 【*args】收集位置参数为元组
    def sum_all(*args):
        """接受任意数量的位置参数"""
        print(f"args = {args}, 类型: {type(args)}")
        return sum(args)

    print(f"sum_all(1, 2, 3) = {sum_all(1, 2, 3)}")
    print(f"sum_all(1, 2, 3, 4, 5) = {sum_all(1, 2, 3, 4, 5)}")

    # 【**kwargs】收集关键字参数为字典
    def print_info(**kwargs):
        """接受任意数量的关键字参数"""
        print(f"kwargs = {kwargs}, 类型: {type(kwargs)}")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")

    print("\nprint_info(name='Alice', age=25):")
    print_info(name="Alice", age=25)

    # 【同时使用 *args 和 **kwargs】
    def flexible_function(*args, **kwargs):
        """同时接受位置参数和关键字参数"""
        print(f"位置参数: {args}")
        print(f"关键字参数: {kwargs}")

    print("\nflexible_function(1, 2, 3, a=4, b=5):")
    flexible_function(1, 2, 3, a=4, b=5)

    # 【参数顺序】
    # 普通参数 -> *args -> 默认参数 -> **kwargs
    def full_signature(a, b, *args, c=10, **kwargs):
        print(f"a={a}, b={b}, args={args}, c={c}, kwargs={kwargs}")

    print("\nfull_signature(1, 2, 3, 4, c=5, d=6, e=7):")
    full_signature(1, 2, 3, 4, c=5, d=6, e=7)

    # 【解包参数】
    print("\n--- 解包参数 ---")

    def add(a, b, c):
        return a + b + c

    # * 解包列表/元组
    numbers = [1, 2, 3]
    print(f"*解包: add(*{numbers}) = {add(*numbers)}")

    # ** 解包字典
    params = {"a": 1, "b": 2, "c": 3}
    print(f"**解包: add(**{params}) = {add(**params)}")

    # 【实际应用：装饰器】
    def log_call(func):
        def wrapper(*args, **kwargs):
            print(f"调用 {func.__name__}，参数: args={args}, kwargs={kwargs}")
            return func(*args, **kwargs)
        return wrapper

    @log_call
    def multiply(a, b):
        return a * b

    print(f"\nmultiply(3, 4) = {multiply(3, 4)}")


def main04_return_values():
    """
    ============================================================
                    4. 返回值
    ============================================================
    """
    print("\n" + "=" * 60)
    print("4. 返回值")
    print("=" * 60)

    # 【返回单个值】
    def square(n):
        return n ** 2

    print(f"square(5) = {square(5)}")

    # 【返回多个值】（实际返回元组）
    def min_max(numbers):
        return min(numbers), max(numbers)

    result = min_max([3, 1, 4, 1, 5, 9])
    print(f"\nmin_max 返回: {result}, 类型: {type(result)}")

    # 解包接收
    minimum, maximum = min_max([3, 1, 4, 1, 5, 9])
    print(f"解包: min={minimum}, max={maximum}")

    # 【返回 None】
    def maybe_return(condition):
        if condition:
            return "有值"
        # 没有 return 语句，返回 None

    print(f"\nmaybe_return(True) = {maybe_return(True)}")
    print(f"maybe_return(False) = {maybe_return(False)}")

    # 【提前返回】（守卫子句）
    def divide(a, b):
        if b == 0:
            return None  # 提前返回
        return a / b

    print(f"\ndivide(10, 2) = {divide(10, 2)}")
    print(f"divide(10, 0) = {divide(10, 0)}")

    # 【返回函数】
    def make_multiplier(factor):
        def multiply(n):
            return n * factor
        return multiply

    double = make_multiplier(2)
    triple = make_multiplier(3)
    print(f"\ndouble(5) = {double(5)}")
    print(f"triple(5) = {triple(5)}")


def main05_first_class():
    """
    ============================================================
                5. 函数作为一等公民
    ============================================================
    """
    print("\n" + "=" * 60)
    print("5. 函数作为一等公民")
    print("=" * 60)

    # 【函数赋值给变量】
    def greet(name):
        return f"Hello, {name}!"

    say_hello = greet  # 函数赋值给变量
    print(f"say_hello('Alice') = {say_hello('Alice')}")

    # 【函数存储在数据结构中】
    def add(a, b): return a + b
    def sub(a, b): return a - b
    def mul(a, b): return a * b
    def div(a, b): return a / b if b != 0 else None

    operations = {
        "+": add,
        "-": sub,
        "*": mul,
        "/": div
    }

    print("\n函数存储在字典中:")
    for op, func in operations.items():
        print(f"  10 {op} 3 = {func(10, 3)}")

    # 【函数作为参数】
    def apply_operation(func, a, b):
        """高阶函数：接受函数作为参数"""
        return func(a, b)

    print(f"\napply_operation(add, 5, 3) = {apply_operation(add, 5, 3)}")
    print(f"apply_operation(mul, 5, 3) = {apply_operation(mul, 5, 3)}")

    # 【函数作为返回值】
    def create_greeting(greeting):
        """返回一个定制的问候函数"""
        def greet(name):
            return f"{greeting}, {name}!"
        return greet

    hello = create_greeting("Hello")
    ni_hao = create_greeting("你好")
    print(f"\nhello('World') = {hello('World')}")
    print(f"ni_hao('世界') = {ni_hao('世界')}")

    # 【内置高阶函数】
    print("\n--- 内置高阶函数 ---")
    numbers = [1, 2, 3, 4, 5]

    # map
    squared = list(map(lambda x: x**2, numbers))
    print(f"map (平方): {squared}")

    # filter
    evens = list(filter(lambda x: x % 2 == 0, numbers))
    print(f"filter (偶数): {evens}")

    # sorted with key
    words = ["banana", "Apple", "cherry"]
    sorted_words = sorted(words, key=str.lower)
    print(f"sorted (忽略大小写): {sorted_words}")

    # 【函数属性】
    print("\n--- 函数属性 ---")
    def example_function(a, b):
        """示例函数的文档"""
        return a + b

    print(f"__name__: {example_function.__name__}")
    print(f"__doc__: {example_function.__doc__}")
    print(f"__module__: {example_function.__module__}")


def main06_lambda():
    """
    ============================================================
                    6. Lambda 匿名函数
    ============================================================
    lambda 是创建小型匿名函数的方式
    语法: lambda 参数: 表达式
    """
    print("\n" + "=" * 60)
    print("6. Lambda 匿名函数")
    print("=" * 60)

    # 【基本 lambda】
    square = lambda x: x ** 2
    print(f"square(5) = {square(5)}")

    # 等价于：
    # def square(x):
    #     return x ** 2

    # 【多个参数】
    add = lambda a, b: a + b
    print(f"add(3, 5) = {add(3, 5)}")

    # 【默认参数】
    greet = lambda name, greeting="Hello": f"{greeting}, {name}!"
    print(f"greet('Alice') = {greet('Alice')}")

    # 【在高阶函数中使用】
    print("\n--- 在高阶函数中使用 ---")
    numbers = [1, 2, 3, 4, 5]

    # map
    doubled = list(map(lambda x: x * 2, numbers))
    print(f"map: {doubled}")

    # filter
    evens = list(filter(lambda x: x % 2 == 0, numbers))
    print(f"filter: {evens}")

    # sorted
    pairs = [(1, 'b'), (2, 'a'), (3, 'c')]
    sorted_pairs = sorted(pairs, key=lambda x: x[1])
    print(f"sorted by second element: {sorted_pairs}")

    # 【条件表达式】
    is_even = lambda x: "偶数" if x % 2 == 0 else "奇数"
    print(f"\nis_even(4) = {is_even(4)}")
    print(f"is_even(5) = {is_even(5)}")

    # 【IIFE: 立即调用的 lambda】
    result = (lambda x, y: x + y)(3, 5)
    print(f"\nIIFE: {result}")

    # 【lambda 的限制】
    # - 只能包含单个表达式
    # - 不能包含语句（如 if 语句、循环等）
    # - 不适合复杂逻辑

    # 【技巧】使用 lambda 创建简单的回调
    print("\n--- lambda 作为回调 ---")

    def process_data(data, callback):
        return [callback(item) for item in data]

    data = [1, 2, 3, 4, 5]
    result = process_data(data, lambda x: x ** 2)
    print(f"处理结果: {result}")


def main07_type_hints():
    """
    ============================================================
                    7. 类型提示（Type Hints）
    ============================================================
    Python 3.5+ 支持类型提示，提高代码可读性
    """
    print("\n" + "=" * 60)
    print("7. 类型提示（Type Hints）")
    print("=" * 60)

    # 【基本类型提示】
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    print(f"greet('Alice') = {greet('Alice')}")

    # 【多个参数和返回值】
    def add(a: int, b: int) -> int:
        return a + b

    print(f"add(3, 5) = {add(3, 5)}")

    # 【可选参数】
    from typing import Optional

    def find_user(user_id: int) -> Optional[str]:
        """返回用户名，找不到返回 None"""
        users = {1: "Alice", 2: "Bob"}
        return users.get(user_id)

    print(f"\nfind_user(1) = {find_user(1)}")
    print(f"find_user(99) = {find_user(99)}")

    # 【复杂类型】
    from typing import List, Dict, Tuple, Union

    def process_items(items: List[int]) -> List[int]:
        return [x * 2 for x in items]

    def get_config() -> Dict[str, int]:
        return {"timeout": 30, "retry": 3}

    def get_bounds() -> Tuple[int, int]:
        return (0, 100)

    def stringify(value: Union[int, float]) -> str:
        return str(value)

    print(f"\nprocess_items([1, 2, 3]) = {process_items([1, 2, 3])}")
    print(f"get_config() = {get_config()}")
    print(f"get_bounds() = {get_bounds()}")
    print(f"stringify(42) = {stringify(42)}")

    # 【Python 3.10+ 简化语法】
    # 可以用 | 替代 Union
    def stringify_new(value: int | float) -> str:
        return str(value)

    # 可以直接用 list, dict 等
    def process_new(items: list[int]) -> list[int]:
        return items

    # 【Callable 类型】
    from typing import Callable

    def apply_func(func: Callable[[int, int], int], a: int, b: int) -> int:
        return func(a, b)

    result = apply_func(lambda x, y: x + y, 3, 5)
    print(f"\napply_func(lambda x, y: x + y, 3, 5) = {result}")

    # 【TypeVar 泛型】
    from typing import TypeVar

    T = TypeVar('T')

    def first(items: list[T]) -> T | None:
        return items[0] if items else None

    print(f"first([1, 2, 3]) = {first([1, 2, 3])}")
    print(f"first(['a', 'b']) = {first(['a', 'b'])}")

    # 【注意】类型提示只是提示，不会强制检查
    # 使用 mypy 等工具进行静态类型检查
    print("\n【注意】类型提示不强制检查，需要用 mypy 等工具")


if __name__ == "__main__":
    main01_function_definition()
    main02_parameters()
    main03_variadic()
    main04_return_values()
    main05_first_class()
    main06_lambda()
    main07_type_hints()
