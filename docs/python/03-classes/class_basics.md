# class basics.py

::: info 文件信息
- 📄 原文件：`01_class_basics.py`
- 🔤 语言：python
:::

Python 类基础
本文件介绍 Python 中类的定义、实例化、属性和方法。

Python 是一门面向对象的语言，类是创建对象的蓝图。

## 完整代码

```python
def main01_class_definition():
    """
    ============================================================
                    1. 类的定义与实例化
    ============================================================
    """
    print("=" * 60)
    print("1. 类的定义与实例化")
    print("=" * 60)

    # 【最简单的类】
    class Empty:
        pass

    obj = Empty()
    print(f"空类实例: {obj}")

    # 【带属性的类】
    class Person:
        # 类属性（所有实例共享）
        species = "Homo sapiens"

        # 初始化方法（构造函数）
        def __init__(self, name, age):
            # 实例属性（每个实例独立）
            self.name = name
            self.age = age

        # 实例方法
        def introduce(self):
            return f"我是 {self.name}，今年 {self.age} 岁"

    # 创建实例
    alice = Person("Alice", 25)
    bob = Person("Bob", 30)

    print(f"\nalice.name = {alice.name}")
    print(f"bob.age = {bob.age}")
    print(f"alice.introduce() = {alice.introduce()}")

    # 【类属性 vs 实例属性】
    print(f"\n--- 类属性 vs 实例属性 ---")
    print(f"Person.species = {Person.species}")
    print(f"alice.species = {alice.species}")

    # 修改类属性会影响所有实例
    Person.species = "Human"
    print(f"修改后 bob.species = {bob.species}")

    # 但给实例赋值会创建实例属性，遮蔽类属性
    alice.species = "Alien"
    print(f"alice.species = {alice.species}")  # Alien
    print(f"bob.species = {bob.species}")      # Human

    # 【动态添加属性】
    print(f"\n--- 动态添加属性 ---")
    alice.email = "alice@example.com"  # 动态添加
    print(f"alice.email = {alice.email}")
    # print(bob.email)  # AttributeError，bob 没有 email


def main02_methods():
    """
    ============================================================
                    2. 方法类型
    ============================================================
    """
    print("\n" + "=" * 60)
    print("2. 方法类型")
    print("=" * 60)

    class MyClass:
        class_var = "类变量"

        def __init__(self, value):
            self.instance_var = value

        # 【实例方法】第一个参数是 self
        def instance_method(self):
            return f"实例方法，访问实例变量: {self.instance_var}"

        # 【类方法】使用 @classmethod，第一个参数是 cls
        @classmethod
        def class_method(cls):
            return f"类方法，访问类变量: {cls.class_var}"

        # 【静态方法】使用 @staticmethod，不需要 self 或 cls
        @staticmethod
        def static_method(x, y):
            return f"静态方法，计算: {x} + {y} = {x + y}"

    obj = MyClass("实例值")

    print(f"实例方法: {obj.instance_method()}")
    print(f"类方法: MyClass.class_method() = {MyClass.class_method()}")
    print(f"类方法: obj.class_method() = {obj.class_method()}")  # 也可以通过实例调用
    print(f"静态方法: {MyClass.static_method(3, 5)}")

    # 【类方法的常见用途：工厂方法】
    print(f"\n--- 类方法作为工厂方法 ---")

    class Date:
        def __init__(self, year, month, day):
            self.year = year
            self.month = month
            self.day = day

        @classmethod
        def from_string(cls, date_string):
            """从字符串创建日期"""
            year, month, day = map(int, date_string.split('-'))
            return cls(year, month, day)

        @classmethod
        def today(cls):
            """创建今天的日期"""
            import datetime
            today = datetime.date.today()
            return cls(today.year, today.month, today.day)

        def __str__(self):
            return f"{self.year}-{self.month:02d}-{self.day:02d}"

    d1 = Date(2024, 1, 15)
    d2 = Date.from_string("2024-06-20")
    d3 = Date.today()

    print(f"直接创建: {d1}")
    print(f"从字符串: {d2}")
    print(f"今天: {d3}")


def main03_special_methods():
    """
    ============================================================
                3. 特殊方法（魔法方法）
    ============================================================
    特殊方法以双下划线开头和结尾，用于自定义类的行为
    """
    print("\n" + "=" * 60)
    print("3. 特殊方法（魔法方法）")
    print("=" * 60)

    class Vector:
        """二维向量类"""

        def __init__(self, x, y):
            self.x = x
            self.y = y

        # 【字符串表示】
        def __str__(self):
            """用户友好的字符串表示（print 时调用）"""
            return f"Vector({self.x}, {self.y})"

        def __repr__(self):
            """开发者友好的字符串表示（调试时使用）"""
            return f"Vector(x={self.x}, y={self.y})"

        # 【算术运算】
        def __add__(self, other):
            """加法: v1 + v2"""
            return Vector(self.x + other.x, self.y + other.y)

        def __sub__(self, other):
            """减法: v1 - v2"""
            return Vector(self.x - other.x, self.y - other.y)

        def __mul__(self, scalar):
            """标量乘法: v * n"""
            return Vector(self.x * scalar, self.y * scalar)

        def __rmul__(self, scalar):
            """反向乘法: n * v"""
            return self.__mul__(scalar)

        def __neg__(self):
            """取负: -v"""
            return Vector(-self.x, -self.y)

        # 【比较运算】
        def __eq__(self, other):
            """相等: v1 == v2"""
            return self.x == other.x and self.y == other.y

        def __lt__(self, other):
            """小于: v1 < v2（按长度比较）"""
            return self.length() < other.length()

        # 【其他】
        def __abs__(self):
            """绝对值/长度: abs(v)"""
            return self.length()

        def __bool__(self):
            """布尔值: bool(v)"""
            return self.x != 0 or self.y != 0

        def __len__(self):
            """长度: len(v)"""
            return 2  # 向量有两个分量

        def length(self):
            """计算向量长度"""
            return (self.x ** 2 + self.y ** 2) ** 0.5

    v1 = Vector(3, 4)
    v2 = Vector(1, 2)

    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v1 + v2 = {v1 + v2}")
    print(f"v1 - v2 = {v1 - v2}")
    print(f"v1 * 2 = {v1 * 2}")
    print(f"3 * v2 = {3 * v2}")
    print(f"-v1 = {-v1}")
    print(f"v1 == Vector(3, 4): {v1 == Vector(3, 4)}")
    print(f"abs(v1) = {abs(v1)}")
    print(f"bool(Vector(0, 0)) = {bool(Vector(0, 0))}")

    # 【容器类特殊方法】
    print(f"\n--- 容器类特殊方法 ---")

    class MyList:
        """自定义列表类"""

        def __init__(self, items=None):
            self._items = list(items) if items else []

        def __len__(self):
            """len(obj)"""
            return len(self._items)

        def __getitem__(self, index):
            """obj[index]"""
            return self._items[index]

        def __setitem__(self, index, value):
            """obj[index] = value"""
            self._items[index] = value

        def __delitem__(self, index):
            """del obj[index]"""
            del self._items[index]

        def __contains__(self, item):
            """item in obj"""
            return item in self._items

        def __iter__(self):
            """for item in obj"""
            return iter(self._items)

        def __repr__(self):
            return f"MyList({self._items})"

    ml = MyList([1, 2, 3, 4, 5])
    print(f"MyList: {ml}")
    print(f"len(ml) = {len(ml)}")
    print(f"ml[2] = {ml[2]}")
    print(f"3 in ml: {3 in ml}")
    ml[0] = 10
    print(f"修改后: {ml}")


def main04_property():
    """
    ============================================================
                4. 属性访问控制
    ============================================================
    """
    print("\n" + "=" * 60)
    print("4. 属性访问控制")
    print("=" * 60)

    # 【使用 @property】
    class Circle:
        def __init__(self, radius):
            self._radius = radius  # 约定：单下划线表示内部使用

        @property
        def radius(self):
            """获取半径"""
            return self._radius

        @radius.setter
        def radius(self, value):
            """设置半径"""
            if value <= 0:
                raise ValueError("半径必须为正数")
            self._radius = value

        @radius.deleter
        def radius(self):
            """删除半径"""
            print("删除半径")
            del self._radius

        @property
        def diameter(self):
            """直径（只读属性）"""
            return self._radius * 2

        @property
        def area(self):
            """面积（只读属性）"""
            return 3.14159 * self._radius ** 2

    c = Circle(5)
    print(f"radius = {c.radius}")
    print(f"diameter = {c.diameter}")
    print(f"area = {c.area:.2f}")

    c.radius = 10
    print(f"新 radius = {c.radius}")
    print(f"新 area = {c.area:.2f}")

    try:
        c.radius = -5
    except ValueError as e:
        print(f"设置负半径: {e}")

    # 【__slots__】限制实例属性
    print(f"\n--- __slots__ ---")

    class Point:
        __slots__ = ['x', 'y']  # 只允许这些属性

        def __init__(self, x, y):
            self.x = x
            self.y = y

    p = Point(3, 4)
    print(f"Point: ({p.x}, {p.y})")
    # p.z = 5  # AttributeError: 'Point' object has no attribute 'z'

    # __slots__ 的好处：
    # 1. 节省内存（不创建 __dict__）
    # 2. 访问属性更快
    # 3. 防止意外添加属性

    # 【私有属性】双下划线触发名称改写
    print(f"\n--- 私有属性（名称改写）---")

    class Secret:
        def __init__(self):
            self.__private = "私有"
            self._protected = "受保护"

        def get_private(self):
            return self.__private

    s = Secret()
    print(f"_protected = {s._protected}")  # 可以访问（但不推荐）
    # print(s.__private)  # AttributeError
    print(f"get_private() = {s.get_private()}")
    print(f"名称改写后: _Secret__private = {s._Secret__private}")  # 实际上改名了


def main05_descriptors():
    """
    ============================================================
                    5. 描述符
    ============================================================
    描述符是实现了特定协议的类，用于控制属性访问
    """
    print("\n" + "=" * 60)
    print("5. 描述符")
    print("=" * 60)

    # 【数据描述符】实现 __get__ 和 __set__
    class TypedField:
        """类型检查描述符"""

        def __init__(self, name, expected_type):
            self.name = name
            self.expected_type = expected_type

        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
            return obj.__dict__.get(self.name)

        def __set__(self, obj, value):
            if not isinstance(value, self.expected_type):
                raise TypeError(
                    f"{self.name} 必须是 {self.expected_type.__name__}，"
                    f"收到 {type(value).__name__}"
                )
            obj.__dict__[self.name] = value

    class Person:
        name = TypedField('name', str)
        age = TypedField('age', int)

        def __init__(self, name, age):
            self.name = name
            self.age = age

    p = Person("Alice", 25)
    print(f"Person: {p.name}, {p.age}")

    try:
        p.age = "二十五"
    except TypeError as e:
        print(f"类型错误: {e}")

    # 【范围检查描述符】
    class RangeField:
        """范围检查描述符"""

        def __init__(self, name, min_val, max_val):
            self.name = name
            self.min_val = min_val
            self.max_val = max_val

        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
            return obj.__dict__.get(self.name)

        def __set__(self, obj, value):
            if not self.min_val <= value <= self.max_val:
                raise ValueError(
                    f"{self.name} 必须在 [{self.min_val}, {self.max_val}] 范围内"
                )
            obj.__dict__[self.name] = value

    class Student:
        score = RangeField('score', 0, 100)

        def __init__(self, name, score):
            self.name = name
            self.score = score

    print(f"\n范围检查描述符:")
    s = Student("Bob", 85)
    print(f"Student: {s.name}, 分数: {s.score}")

    try:
        s.score = 120
    except ValueError as e:
        print(f"范围错误: {e}")


def main06_callable_objects():
    """
    ============================================================
                    6. 可调用对象
    ============================================================
    """
    print("\n" + "=" * 60)
    print("6. 可调用对象")
    print("=" * 60)

    # 实现 __call__ 使对象可以像函数一样调用
    class Adder:
        """累加器"""

        def __init__(self, start=0):
            self.total = start

        def __call__(self, value):
            self.total += value
            return self.total

    adder = Adder(100)
    print(f"adder(10) = {adder(10)}")
    print(f"adder(20) = {adder(20)}")
    print(f"adder(30) = {adder(30)}")

    # 【带状态的函数】
    class Counter:
        """调用计数器"""

        def __init__(self, func):
            self.func = func
            self.count = 0

        def __call__(self, *args, **kwargs):
            self.count += 1
            return self.func(*args, **kwargs)

    @Counter
    def greet(name):
        return f"Hello, {name}!"

    print(f"\n{greet('Alice')}")
    print(f"{greet('Bob')}")
    print(f"调用次数: {greet.count}")

    # 【策略模式】
    print(f"\n--- 策略模式 ---")

    class Discount:
        """折扣策略基类"""

        def __call__(self, price):
            raise NotImplementedError

    class NoDiscount(Discount):
        def __call__(self, price):
            return price

    class PercentDiscount(Discount):
        def __init__(self, percent):
            self.percent = percent

        def __call__(self, price):
            return price * (1 - self.percent / 100)

    class FixedDiscount(Discount):
        def __init__(self, amount):
            self.amount = amount

        def __call__(self, price):
            return max(0, price - self.amount)

    # 使用
    discounts = [
        ("无折扣", NoDiscount()),
        ("8折", PercentDiscount(20)),
        ("减50", FixedDiscount(50)),
    ]

    price = 200
    print(f"原价: {price}")
    for name, discount in discounts:
        print(f"  {name}: {discount(price)}")


if __name__ == "__main__":
    main01_class_definition()
    main02_methods()
    main03_special_methods()
    main04_property()
    main05_descriptors()
    main06_callable_objects()
```
