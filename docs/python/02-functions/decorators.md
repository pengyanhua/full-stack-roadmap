# decorators.py

::: info 文件信息
- 📄 原文件：`03_decorators.py`
- 🔤 语言：python
:::

Python 装饰器
本文件详细介绍 Python 中的装饰器（Decorator）。

装饰器是一种设计模式，用于在不修改原函数代码的情况下，
动态地扩展函数的功能。本质上是一个接受函数并返回新函数的函数。

## 完整代码

```python
import functools
import time
from typing import Callable, Any


def main01_decorator_basics():
    """
    ============================================================
                    1. 装饰器基础
    ============================================================
    """
    print("=" * 60)
    print("1. 装饰器基础")
    print("=" * 60)

    # 【装饰器的本质】
    # 装饰器就是一个接受函数作为参数，返回新函数的高阶函数

    def my_decorator(func):
        def wrapper():
            print("函数执行前")
            func()
            print("函数执行后")
        return wrapper

    # 手动应用装饰器
    def say_hello():
        print("Hello!")

    decorated = my_decorator(say_hello)
    print("手动应用装饰器:")
    decorated()

    # 【@ 语法糖】
    print("\n使用 @ 语法糖:")

    @my_decorator
    def say_goodbye():
        print("Goodbye!")

    say_goodbye()

    # 等价于：say_goodbye = my_decorator(say_goodbye)

    # 【带参数的函数】
    print("\n--- 装饰带参数的函数 ---")

    def decorator_with_args(func):
        def wrapper(*args, **kwargs):
            print(f"参数: args={args}, kwargs={kwargs}")
            result = func(*args, **kwargs)
            print(f"返回值: {result}")
            return result
        return wrapper

    @decorator_with_args
    def add(a, b):
        return a + b

    result = add(3, 5)
    print(f"最终结果: {result}")


def main02_functools_wraps():
    """
    ============================================================
                2. 使用 functools.wraps 保留元信息
    ============================================================
    """
    print("\n" + "=" * 60)
    print("2. 使用 functools.wraps 保留元信息")
    print("=" * 60)

    # 【问题：装饰器会丢失原函数信息】
    def bad_decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    @bad_decorator
    def greet(name):
        """向某人问好"""
        return f"Hello, {name}!"

    print(f"没有 @wraps 时:")
    print(f"  __name__: {greet.__name__}")    # wrapper，不是 greet
    print(f"  __doc__: {greet.__doc__}")      # None

    # 【解决方案：使用 @functools.wraps】
    def good_decorator(func):
        @functools.wraps(func)  # 保留原函数的元信息
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    @good_decorator
    def farewell(name):
        """向某人告别"""
        return f"Goodbye, {name}!"

    print(f"\n使用 @wraps 后:")
    print(f"  __name__: {farewell.__name__}")  # farewell
    print(f"  __doc__: {farewell.__doc__}")    # 向某人告别

    # 【还可以访问原函数】
    print(f"  __wrapped__: {farewell.__wrapped__}")


def main03_decorator_with_args():
    """
    ============================================================
                    3. 带参数的装饰器
    ============================================================
    """
    print("\n" + "=" * 60)
    print("3. 带参数的装饰器")
    print("=" * 60)

    # 【基本结构：三层嵌套】
    def repeat(times):
        """让函数重复执行指定次数"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                result = None
                for _ in range(times):
                    result = func(*args, **kwargs)
                return result
            return wrapper
        return decorator

    @repeat(times=3)
    def greet(name):
        print(f"Hello, {name}!")

    print("重复3次:")
    greet("Alice")

    # 【带默认参数的装饰器】
    def log(level="INFO"):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                print(f"[{level}] 调用 {func.__name__}")
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @log(level="DEBUG")
    def process():
        print("处理中...")

    @log()  # 使用默认级别
    def another_process():
        print("另一个处理...")

    print("\n带级别的日志:")
    process()
    another_process()

    # 【灵活的装饰器：可以带参数也可以不带】
    def flexible_decorator(func=None, *, prefix=">>>"):
        def decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                print(f"{prefix} 开始执行 {f.__name__}")
                return f(*args, **kwargs)
            return wrapper

        if func is not None:
            # 不带参数调用 @flexible_decorator
            return decorator(func)
        # 带参数调用 @flexible_decorator(prefix="---")
        return decorator

    @flexible_decorator  # 不带参数
    def task1():
        print("任务1")

    @flexible_decorator(prefix="---")  # 带参数
    def task2():
        print("任务2")

    print("\n灵活的装饰器:")
    task1()
    task2()


def main04_class_decorator():
    """
    ============================================================
                    4. 类装饰器
    ============================================================
    """
    print("\n" + "=" * 60)
    print("4. 类装饰器")
    print("=" * 60)

    # 【使用类实现装饰器】
    class Timer:
        """计时装饰器（类实现）"""
        def __init__(self, func):
            functools.update_wrapper(self, func)
            self.func = func

        def __call__(self, *args, **kwargs):
            start = time.perf_counter()
            result = self.func(*args, **kwargs)
            end = time.perf_counter()
            print(f"{self.func.__name__} 耗时: {end - start:.4f}秒")
            return result

    @Timer
    def slow_function():
        time.sleep(0.1)
        return "完成"

    print("类装饰器:")
    slow_function()

    # 【带参数的类装饰器】
    class Retry:
        """重试装饰器"""
        def __init__(self, max_retries=3, delay=0.1):
            self.max_retries = max_retries
            self.delay = delay

        def __call__(self, func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(self.max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        print(f"  尝试 {attempt + 1} 失败: {e}")
                        time.sleep(self.delay)
                raise last_exception

            return wrapper

    attempt_count = [0]

    @Retry(max_retries=3, delay=0.01)
    def unreliable_function():
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise ValueError("随机错误")
        return "成功！"

    print("\n带参数的类装饰器 (重试):")
    try:
        result = unreliable_function()
        print(f"结果: {result}")
    except Exception as e:
        print(f"最终失败: {e}")


def main05_stacking_decorators():
    """
    ============================================================
                    5. 装饰器堆叠
    ============================================================
    """
    print("\n" + "=" * 60)
    print("5. 装饰器堆叠")
    print("=" * 60)

    def bold(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return f"<b>{func(*args, **kwargs)}</b>"
        return wrapper

    def italic(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return f"<i>{func(*args, **kwargs)}</i>"
        return wrapper

    def underline(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return f"<u>{func(*args, **kwargs)}</u>"
        return wrapper

    # 【堆叠顺序：从下往上应用，从上往下执行】
    @bold
    @italic
    @underline
    def hello():
        return "Hello"

    # 等价于：hello = bold(italic(underline(hello)))

    print(f"堆叠结果: {hello()}")
    # 输出: <b><i><u>Hello</u></i></b>

    print("\n装饰器应用顺序：从下往上")
    print("装饰器执行顺序：从上往下")


def main06_practical_decorators():
    """
    ============================================================
                6. 实用装饰器示例
    ============================================================
    """
    print("\n" + "=" * 60)
    print("6. 实用装饰器示例")
    print("=" * 60)

    # 【1. 计时装饰器】
    def timer(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            print(f"  {func.__name__} 耗时: {(end - start) * 1000:.2f}ms")
            return result
        return wrapper

    # 【2. 缓存装饰器】
    def cache(func):
        """简单的缓存装饰器"""
        cached = {}

        @functools.wraps(func)
        def wrapper(*args):
            if args not in cached:
                cached[args] = func(*args)
            return cached[args]

        wrapper.cache = cached
        wrapper.clear_cache = cached.clear
        return wrapper

    # 【3. 验证装饰器】
    def validate_positive(func):
        """确保所有参数都是正数"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for arg in args:
                if isinstance(arg, (int, float)) and arg <= 0:
                    raise ValueError(f"参数必须为正数，收到: {arg}")
            return func(*args, **kwargs)
        return wrapper

    # 【4. 日志装饰器】
    def log_calls(func):
        """记录函数调用"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            print(f"  调用: {func.__name__}({signature})")
            result = func(*args, **kwargs)
            print(f"  返回: {result!r}")
            return result
        return wrapper

    # 【5. 单例装饰器】
    def singleton(cls):
        """确保类只有一个实例"""
        instances = {}

        @functools.wraps(cls)
        def get_instance(*args, **kwargs):
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
            return instances[cls]

        return get_instance

    # 【6. 类型检查装饰器】
    def enforce_types(func):
        """根据类型提示检查参数类型"""
        hints = func.__annotations__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 检查位置参数
            import inspect
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            for i, arg in enumerate(args):
                if i < len(params):
                    param_name = params[i]
                    if param_name in hints:
                        expected_type = hints[param_name]
                        if not isinstance(arg, expected_type):
                            raise TypeError(
                                f"参数 {param_name} 应为 {expected_type.__name__}，"
                                f"收到 {type(arg).__name__}"
                            )

            return func(*args, **kwargs)

        return wrapper

    # 【应用示例】
    print("--- 计时装饰器 ---")

    @timer
    def slow_sum(n):
        return sum(range(n))

    slow_sum(100000)

    print("\n--- 缓存装饰器 ---")

    @cache
    def fibonacci(n):
        if n < 2:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    @timer
    def test_fib():
        return fibonacci(30)

    test_fib()
    print(f"  缓存大小: {len(fibonacci.cache)}")

    print("\n--- 日志装饰器 ---")

    @log_calls
    def add(a, b):
        return a + b

    add(3, 5)

    print("\n--- 单例装饰器 ---")

    @singleton
    class Database:
        def __init__(self):
            print("  创建数据库连接")

    db1 = Database()
    db2 = Database()
    print(f"  db1 is db2: {db1 is db2}")

    print("\n--- 类型检查装饰器 ---")

    @enforce_types
    def greet(name: str, times: int):
        return f"{name}! " * times

    print(f"  greet('Hello', 3) = {greet('Hello', 3)}")
    try:
        greet(123, "abc")
    except TypeError as e:
        print(f"  类型错误: {e}")


def main07_builtin_decorators():
    """
    ============================================================
                7. 内置装饰器
    ============================================================
    """
    print("\n" + "=" * 60)
    print("7. 内置装饰器")
    print("=" * 60)

    # 【@staticmethod】静态方法
    print("--- @staticmethod ---")

    class MathUtils:
        @staticmethod
        def add(a, b):
            return a + b

    print(f"MathUtils.add(3, 5) = {MathUtils.add(3, 5)}")

    # 【@classmethod】类方法
    print("\n--- @classmethod ---")

    class Person:
        count = 0

        def __init__(self, name):
            self.name = name
            Person.count += 1

        @classmethod
        def from_string(cls, s):
            """从字符串创建实例"""
            name = s.split(":")[1]
            return cls(name)

        @classmethod
        def get_count(cls):
            return cls.count

    p = Person.from_string("name:Alice")
    print(f"Person: {p.name}, 总数: {Person.get_count()}")

    # 【@property】属性装饰器
    print("\n--- @property ---")

    class Circle:
        def __init__(self, radius):
            self._radius = radius

        @property
        def radius(self):
            """获取半径"""
            return self._radius

        @radius.setter
        def radius(self, value):
            """设置半径"""
            if value <= 0:
                raise ValueError("半径必须为正")
            self._radius = value

        @property
        def area(self):
            """计算面积（只读属性）"""
            return 3.14159 * self._radius ** 2

    c = Circle(5)
    print(f"半径: {c.radius}, 面积: {c.area:.2f}")
    c.radius = 10
    print(f"新半径: {c.radius}, 新面积: {c.area:.2f}")

    # 【@functools.lru_cache】LRU 缓存
    print("\n--- @functools.lru_cache ---")

    @functools.lru_cache(maxsize=128)
    def fib(n):
        if n < 2:
            return n
        return fib(n - 1) + fib(n - 2)

    result = fib(100)
    print(f"fib(100) = {result}")
    print(f"缓存信息: {fib.cache_info()}")

    # 【@functools.total_ordering】自动生成比较方法
    print("\n--- @functools.total_ordering ---")

    @functools.total_ordering
    class Student:
        def __init__(self, name, score):
            self.name = name
            self.score = score

        def __eq__(self, other):
            return self.score == other.score

        def __lt__(self, other):
            return self.score < other.score

    s1 = Student("Alice", 85)
    s2 = Student("Bob", 90)
    print(f"Alice < Bob: {s1 < s2}")
    print(f"Alice <= Bob: {s1 <= s2}")
    print(f"Alice > Bob: {s1 > s2}")


def main08_dataclass():
    """
    ============================================================
                8. @dataclass 装饰器
    ============================================================
    Python 3.7+ 的数据类装饰器
    """
    print("\n" + "=" * 60)
    print("8. @dataclass 装饰器")
    print("=" * 60)

    from dataclasses import dataclass, field

    # 【基本数据类】
    @dataclass
    class Point:
        x: float
        y: float

    p = Point(3, 4)
    print(f"Point: {p}")
    print(f"自动生成 __repr__: Point(x=3, y=4)")

    # 【带默认值】
    @dataclass
    class Person:
        name: str
        age: int = 0
        city: str = "Unknown"

    person = Person("Alice", 25)
    print(f"\nPerson: {person}")

    # 【不可变数据类】
    @dataclass(frozen=True)
    class ImmutablePoint:
        x: float
        y: float

    ip = ImmutablePoint(1, 2)
    print(f"\nImmutablePoint: {ip}")
    # ip.x = 10  # 报错！FrozenInstanceError

    # 【字段选项】
    @dataclass
    class Config:
        name: str
        values: list = field(default_factory=list)  # 可变默认值
        _internal: str = field(default="", repr=False)  # 不在 repr 中显示

    config = Config("test")
    config.values.append(1)
    print(f"\nConfig: {config}")

    # 【继承】
    @dataclass
    class Employee(Person):
        department: str = "Engineering"

    emp = Employee("Bob", 30, "Shanghai", "Sales")
    print(f"\nEmployee: {emp}")

    # 【自动生成比较方法】
    @dataclass(order=True)
    class Student:
        score: int
        name: str = field(compare=False)  # 不参与比较

    students = [
        Student(85, "Alice"),
        Student(90, "Bob"),
        Student(78, "Charlie"),
    ]
    print(f"\n排序前: {[s.name for s in students]}")
    students.sort()
    print(f"排序后: {[s.name for s in students]}")


if __name__ == "__main__":
    main01_decorator_basics()
    main02_functools_wraps()
    main03_decorator_with_args()
    main04_class_decorator()
    main05_stacking_decorators()
    main06_practical_decorators()
    main07_builtin_decorators()
    main08_dataclass()
```
