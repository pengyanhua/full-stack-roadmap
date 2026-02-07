# variables.py

::: info 文件信息
- 📄 原文件：`01_variables.py`
- 🔤 语言：python
:::

Python 变量与数据类型
本文件介绍 Python 中的变量声明、基本数据类型、类型转换等核心概念。

Python 是动态类型语言，变量不需要声明类型，类型在运行时确定。

## 完整代码

```python
def main01_variables():
    """
    ============================================================
                        1. 变量声明与赋值
    ============================================================
    Python 变量特点：
    - 不需要声明类型（动态类型）
    - 变量名区分大小写
    - 变量本质是对象的引用（一切皆对象）
    """
    print("=" * 60)
    print("1. 变量声明与赋值")
    print("=" * 60)

    # 【基本赋值】直接赋值，类型自动推断
    name = "Alice"          # str 类型
    age = 25                # int 类型
    height = 1.75           # float 类型
    is_student = True       # bool 类型

    print(f"name = {name}, 类型: {type(name)}")
    print(f"age = {age}, 类型: {type(age)}")
    print(f"height = {height}, 类型: {type(height)}")
    print(f"is_student = {is_student}, 类型: {type(is_student)}")

    # 【多重赋值】一行赋值多个变量
    x, y, z = 1, 2, 3
    print(f"\n多重赋值: x={x}, y={y}, z={z}")

    # 【链式赋值】多个变量指向同一对象
    a = b = c = 100
    print(f"链式赋值: a={a}, b={b}, c={c}")
    print(f"a is b: {a is b}")  # True，指向同一对象

    # 【变量交换】Python 特有的优雅写法
    x, y = 10, 20
    print(f"\n交换前: x={x}, y={y}")
    x, y = y, x  # 无需临时变量！
    print(f"交换后: x={x}, y={y}")

    # 【解包赋值】从序列中解包
    coordinates = (100, 200, 300)
    x, y, z = coordinates
    print(f"\n解包赋值: x={x}, y={y}, z={z}")

    # 【星号解包】收集剩余元素
    first, *rest, last = [1, 2, 3, 4, 5]
    print(f"星号解包: first={first}, rest={rest}, last={last}")


def main02_numbers():
    """
    ============================================================
                        2. 数字类型
    ============================================================
    Python 数字类型：
    - int: 整数（无限精度！）
    - float: 浮点数（双精度）
    - complex: 复数
    """
    print("\n" + "=" * 60)
    print("2. 数字类型")
    print("=" * 60)

    # 【整数 int】
    # 【重要】Python 3 的整数没有长度限制！
    small_int = 42
    big_int = 123456789012345678901234567890  # 任意大的整数
    print(f"小整数: {small_int}")
    print(f"大整数: {big_int}")

    # 不同进制表示
    binary = 0b1010       # 二进制，值为 10
    octal = 0o17          # 八进制，值为 15
    hexadecimal = 0xFF    # 十六进制，值为 255
    print(f"\n二进制 0b1010 = {binary}")
    print(f"八进制 0o17 = {octal}")
    print(f"十六进制 0xFF = {hexadecimal}")

    # 数字分隔符（Python 3.6+）
    million = 1_000_000
    print(f"使用下划线分隔: {million}")

    # 【浮点数 float】
    pi = 3.14159
    scientific = 1.23e-4  # 科学计数法，值为 0.000123
    print(f"\n浮点数 pi = {pi}")
    print(f"科学计数法 1.23e-4 = {scientific}")

    # 【警告】浮点数精度问题！
    result = 0.1 + 0.2
    print(f"\n【警告】0.1 + 0.2 = {result}")  # 不是 0.3！
    print(f"0.1 + 0.2 == 0.3: {result == 0.3}")  # False!

    # 【技巧】使用 decimal 模块处理精确计算
    from decimal import Decimal
    d1 = Decimal('0.1')
    d2 = Decimal('0.2')
    print(f"Decimal: 0.1 + 0.2 = {d1 + d2}")  # 精确的 0.3

    # 【复数 complex】
    c1 = 3 + 4j
    c2 = complex(1, 2)
    print(f"\n复数: c1 = {c1}, 实部 = {c1.real}, 虚部 = {c1.imag}")
    print(f"复数运算: {c1} + {c2} = {c1 + c2}")

    # 【数学运算】
    print("\n--- 数学运算 ---")
    a, b = 17, 5
    print(f"a={a}, b={b}")
    print(f"加法: a + b = {a + b}")
    print(f"减法: a - b = {a - b}")
    print(f"乘法: a * b = {a * b}")
    print(f"除法: a / b = {a / b}")      # 总是返回 float
    print(f"整除: a // b = {a // b}")    # 向下取整
    print(f"取模: a % b = {a % b}")
    print(f"幂运算: a ** b = {a ** b}")  # 17^5
    print(f"divmod: divmod(a, b) = {divmod(a, b)}")  # 同时返回商和余数

    # 【内置数学函数】
    print("\n--- 内置数学函数 ---")
    print(f"abs(-10) = {abs(-10)}")
    print(f"round(3.7) = {round(3.7)}")
    print(f"round(3.14159, 2) = {round(3.14159, 2)}")
    print(f"pow(2, 10) = {pow(2, 10)}")
    print(f"max(1, 5, 3) = {max(1, 5, 3)}")
    print(f"min(1, 5, 3) = {min(1, 5, 3)}")


def main03_strings():
    """
    ============================================================
                        3. 字符串类型
    ============================================================
    Python 字符串特点：
    - 不可变（immutable）
    - 支持 Unicode（UTF-8）
    - 单引号和双引号等价
    """
    print("\n" + "=" * 60)
    print("3. 字符串类型")
    print("=" * 60)

    # 【字符串创建】
    s1 = 'Hello'           # 单引号
    s2 = "World"           # 双引号
    s3 = '''多行
字符串'''                   # 三引号（多行）
    s4 = """也可以用
双引号"""

    print(f"单引号: {s1}")
    print(f"双引号: {s2}")
    print(f"三引号: {s3}")

    # 【转义字符】
    escaped = "Hello\tWorld\n换行了"
    print(f"\n转义字符: {escaped}")

    # 【原始字符串】r前缀，不处理转义
    raw = r"C:\Users\name\file.txt"
    print(f"原始字符串: {raw}")

    # 【字符串拼接】
    print("\n--- 字符串拼接 ---")
    greeting = s1 + " " + s2
    print(f"+ 拼接: {greeting}")

    repeated = "Ha" * 3
    print(f"* 重复: {repeated}")

    # 【格式化字符串】
    print("\n--- 字符串格式化 ---")
    name = "Alice"
    age = 25

    # f-string（推荐，Python 3.6+）
    print(f"f-string: Hello, {name}! You are {age} years old.")
    print(f"表达式: 2 + 3 = {2 + 3}")
    print(f"格式化数字: pi = {3.14159:.2f}")
    print(f"填充对齐: |{name:>10}| |{name:<10}| |{name:^10}|")

    # format() 方法
    template = "Hello, {}! You are {} years old."
    print(f"format(): {template.format(name, age)}")

    # % 格式化（旧式，不推荐）
    print("%%格式化: Hello, %s! You are %d years old." % (name, age))

    # 【字符串索引和切片】
    print("\n--- 索引和切片 ---")
    s = "Hello, Python!"
    print(f"字符串: '{s}'")
    print(f"s[0] = '{s[0]}'")       # 第一个字符
    print(f"s[-1] = '{s[-1]}'")     # 最后一个字符
    print(f"s[0:5] = '{s[0:5]}'")   # 切片 [start:end)
    print(f"s[7:] = '{s[7:]}'")     # 从索引7到末尾
    print(f"s[:5] = '{s[:5]}'")     # 从开头到索引5
    print(f"s[::2] = '{s[::2]}'")   # 步长为2
    print(f"s[::-1] = '{s[::-1]}'") # 反转字符串

    # 【常用字符串方法】
    print("\n--- 常用字符串方法 ---")
    text = "  Hello, World!  "
    print(f"原字符串: '{text}'")
    print(f"strip(): '{text.strip()}'")
    print(f"lower(): '{text.lower()}'")
    print(f"upper(): '{text.upper()}'")
    print(f"replace(): '{text.replace('World', 'Python')}'")
    print(f"split(): {text.split(',')}")
    print(f"find('World'): {text.find('World')}")
    print(f"startswith('  Hello'): {text.startswith('  Hello')}")
    print(f"endswith('!  '): {text.endswith('!  ')}")

    # 【字符串判断方法】
    print("\n--- 字符串判断 ---")
    print(f"'123'.isdigit(): {'123'.isdigit()}")
    print(f"'abc'.isalpha(): {'abc'.isalpha()}")
    print(f"'abc123'.isalnum(): {'abc123'.isalnum()}")
    print(f"'   '.isspace(): {'   '.isspace()}")

    # 【join 方法】
    words = ["Hello", "World", "Python"]
    joined = " ".join(words)
    print(f"\njoin: {joined}")

    # 【注意】字符串是不可变的！
    # s[0] = 'h'  # 这会报错！TypeError


def main04_bool_none():
    """
    ============================================================
                    4. 布尔值和 None
    ============================================================
    """
    print("\n" + "=" * 60)
    print("4. 布尔值和 None")
    print("=" * 60)

    # 【布尔值】True 和 False
    is_valid = True
    is_empty = False
    print(f"is_valid = {is_valid}, 类型: {type(is_valid)}")

    # 【布尔运算】
    print("\n--- 布尔运算 ---")
    print(f"True and False = {True and False}")
    print(f"True or False = {True or False}")
    print(f"not True = {not True}")

    # 【比较运算】
    print("\n--- 比较运算 ---")
    a, b = 10, 20
    print(f"a={a}, b={b}")
    print(f"a == b: {a == b}")
    print(f"a != b: {a != b}")
    print(f"a < b: {a < b}")
    print(f"a >= b: {a >= b}")

    # 【链式比较】Python 特有！
    x = 15
    print(f"\n链式比较: 10 < {x} < 20 = {10 < x < 20}")

    # 【真值判断】
    # 以下值被视为 False：
    # - None
    # - False
    # - 数字零：0, 0.0, 0j
    # - 空序列：'', [], (), {}
    # - 空集合：set()
    # - 自定义类的 __bool__() 或 __len__() 返回 False 或 0
    print("\n--- 真值判断（Truthy/Falsy）---")
    falsy_values = [None, False, 0, 0.0, '', [], {}, set()]
    for val in falsy_values:
        print(f"bool({repr(val):10}) = {bool(val)}")

    print("\n非空值都为 True:")
    truthy_values = [1, -1, "hello", [1], {"a": 1}]
    for val in truthy_values:
        print(f"bool({repr(val):15}) = {bool(val)}")

    # 【None 类型】
    print("\n--- None 类型 ---")
    result = None
    print(f"result = {result}, 类型: {type(result)}")

    # 【重要】使用 is 而不是 == 来判断 None
    if result is None:
        print("result is None: True（使用 is 判断）")

    # 【技巧】函数没有显式 return 时返回 None
    def no_return():
        pass

    print(f"无返回值函数: {no_return()}")


def main05_type_conversion():
    """
    ============================================================
                        5. 类型转换
    ============================================================
    """
    print("\n" + "=" * 60)
    print("5. 类型转换")
    print("=" * 60)

    # 【显式类型转换】
    print("--- 显式类型转换 ---")

    # 转换为整数
    print(f"int('123') = {int('123')}")
    print(f"int(3.9) = {int(3.9)}")      # 截断，不是四舍五入
    print(f"int('0xFF', 16) = {int('0xFF', 16)}")  # 指定进制

    # 转换为浮点数
    print(f"\nfloat('3.14') = {float('3.14')}")
    print(f"float(10) = {float(10)}")

    # 转换为字符串
    print(f"\nstr(123) = '{str(123)}'")
    print(f"str(3.14) = '{str(3.14)}'")
    print(f"str(True) = '{str(True)}'")

    # 转换为布尔
    print(f"\nbool(1) = {bool(1)}")
    print(f"bool(0) = {bool(0)}")
    print(f"bool('') = {bool('')}")
    print(f"bool('hello') = {bool('hello')}")

    # 转换为列表/元组/集合
    print(f"\nlist('abc') = {list('abc')}")
    print(f"tuple([1, 2, 3]) = {tuple([1, 2, 3])}")
    print(f"set([1, 2, 2, 3]) = {set([1, 2, 2, 3])}")

    # 【类型检查】
    print("\n--- 类型检查 ---")
    x = 42
    print(f"type(x) = {type(x)}")
    print(f"type(x) == int: {type(x) == int}")
    print(f"isinstance(x, int): {isinstance(x, int)}")
    print(f"isinstance(x, (int, float)): {isinstance(x, (int, float))}")

    # 【注意】isinstance 更推荐，因为它支持继承
    print(f"\nisinstance(True, int): {isinstance(True, int)}")  # True! bool 是 int 的子类


def main06_constants():
    """
    ============================================================
                        6. 常量（约定）
    ============================================================
    【注意】Python 没有真正的常量！
    只能通过命名约定（全大写）来表示常量
    """
    print("\n" + "=" * 60)
    print("6. 常量（约定）")
    print("=" * 60)

    # 【约定】全大写表示常量
    PI = 3.14159
    MAX_SIZE = 100
    DEFAULT_NAME = "Unknown"

    print(f"PI = {PI}")
    print(f"MAX_SIZE = {MAX_SIZE}")
    print(f"DEFAULT_NAME = {DEFAULT_NAME}")

    # 【警告】Python 不会阻止你修改"常量"
    # PI = 3.0  # 这在语法上是允许的！

    # 【技巧】使用 Enum 创建真正的常量
    from enum import Enum

    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3

    print(f"\nEnum 常量: {Color.RED}, 值: {Color.RED.value}")

    # 【技巧】使用 typing.Final（Python 3.8+）
    from typing import Final
    MAX_CONNECTIONS: Final = 10
    print(f"Final 类型提示: MAX_CONNECTIONS = {MAX_CONNECTIONS}")
    # 类型检查器会警告对 Final 变量的重新赋值


def main07_id_and_is():
    """
    ============================================================
                    7. 对象标识与相等性
    ============================================================
    - id(): 返回对象的唯一标识（内存地址）
    - is: 判断是否是同一对象
    - ==: 判断值是否相等
    """
    print("\n" + "=" * 60)
    print("7. 对象标识与相等性")
    print("=" * 60)

    # 【is vs ==】
    a = [1, 2, 3]
    b = [1, 2, 3]
    c = a

    print(f"a = {a}, id(a) = {id(a)}")
    print(f"b = {b}, id(b) = {id(b)}")
    print(f"c = a, id(c) = {id(c)}")

    print(f"\na == b: {a == b}")  # True，值相等
    print(f"a is b: {a is b}")   # False，不是同一对象
    print(f"a is c: {a is c}")   # True，是同一对象

    # 【小整数缓存】
    # Python 缓存 -5 到 256 的整数
    print("\n--- 小整数缓存 ---")
    x = 100
    y = 100
    print(f"x = 100, y = 100")
    print(f"x is y: {x is y}")  # True，使用缓存

    x = 1000
    y = 1000
    print(f"\nx = 1000, y = 1000")
    print(f"x is y: {x is y}")  # 可能是 False

    # 【字符串驻留】
    print("\n--- 字符串驻留 ---")
    s1 = "hello"
    s2 = "hello"
    print(f"s1 = 'hello', s2 = 'hello'")
    print(f"s1 is s2: {s1 is s2}")  # True，字符串驻留

    s3 = "hello world"
    s4 = "hello world"
    print(f"\ns3 = 'hello world', s4 = 'hello world'")
    print(f"s3 is s4: {s3 is s4}")  # 可能是 True 或 False


if __name__ == "__main__":
    main01_variables()
    main02_numbers()
    main03_strings()
    main04_bool_none()
    main05_type_conversion()
    main06_constants()
    main07_id_and_is()
```
