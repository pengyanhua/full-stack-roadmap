# inheritance.py

::: info 文件信息
- 📄 原文件：`02_inheritance.py`
- 🔤 语言：python
:::

Python 继承与多态
本文件介绍 Python 中的类继承、多态、抽象类等概念。

## 完整代码

```python
from abc import ABC, abstractmethod


def main01_basic_inheritance():
    """
    ============================================================
                    1. 基本继承
    ============================================================
    """
    print("=" * 60)
    print("1. 基本继承")
    print("=" * 60)

    # 【基类（父类）】
    class Animal:
        def __init__(self, name):
            self.name = name

        def speak(self):
            return "..."

        def info(self):
            return f"我是 {self.name}"

    # 【派生类（子类）】
    class Dog(Animal):
        def speak(self):  # 重写父类方法
            return "汪汪!"

        def fetch(self):  # 新增方法
            return f"{self.name} 正在捡球"

    class Cat(Animal):
        def speak(self):
            return "喵喵!"

        def climb(self):
            return f"{self.name} 正在爬树"

    # 使用
    dog = Dog("旺财")
    cat = Cat("咪咪")

    print(f"dog.info(): {dog.info()}")      # 继承的方法
    print(f"dog.speak(): {dog.speak()}")    # 重写的方法
    print(f"dog.fetch(): {dog.fetch()}")    # 新增的方法

    print(f"\ncat.info(): {cat.info()}")
    print(f"cat.speak(): {cat.speak()}")

    # 【isinstance 和 issubclass】
    print(f"\n--- 类型检查 ---")
    print(f"isinstance(dog, Dog): {isinstance(dog, Dog)}")
    print(f"isinstance(dog, Animal): {isinstance(dog, Animal)}")
    print(f"issubclass(Dog, Animal): {issubclass(Dog, Animal)}")


def main02_super():
    """
    ============================================================
                    2. super() 调用父类方法
    ============================================================
    """
    print("\n" + "=" * 60)
    print("2. super() 调用父类方法")
    print("=" * 60)

    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age

        def introduce(self):
            return f"我是 {self.name}，{self.age} 岁"

    class Employee(Person):
        def __init__(self, name, age, department):
            super().__init__(name, age)  # 调用父类的 __init__
            self.department = department

        def introduce(self):
            # 调用父类方法并扩展
            base_intro = super().introduce()
            return f"{base_intro}，在 {self.department} 部门工作"

    emp = Employee("Alice", 30, "Engineering")
    print(f"emp.introduce(): {emp.introduce()}")

    # 【不使用 super 的问题】
    print(f"\n--- 为什么要用 super() ---")

    class A:
        def method(self):
            print("A.method")

    class B(A):
        def method(self):
            print("B.method start")
            super().method()  # 推荐
            # A.method(self)  # 不推荐：硬编码父类名
            print("B.method end")

    B().method()


def main03_multiple_inheritance():
    """
    ============================================================
                    3. 多重继承
    ============================================================
    """
    print("\n" + "=" * 60)
    print("3. 多重继承")
    print("=" * 60)

    # 【多重继承基础】
    class Flyable:
        def fly(self):
            return "我能飞!"

    class Swimmable:
        def swim(self):
            return "我能游泳!"

    class Duck(Flyable, Swimmable):
        def quack(self):
            return "嘎嘎!"

    duck = Duck()
    print(f"duck.fly(): {duck.fly()}")
    print(f"duck.swim(): {duck.swim()}")
    print(f"duck.quack(): {duck.quack()}")

    # 【MRO：方法解析顺序】
    print(f"\n--- MRO (方法解析顺序) ---")

    class A:
        def method(self):
            return "A"

    class B(A):
        def method(self):
            return "B"

    class C(A):
        def method(self):
            return "C"

    class D(B, C):
        pass

    d = D()
    print(f"d.method(): {d.method()}")  # B（按 MRO 顺序）
    print(f"D.mro(): {[cls.__name__ for cls in D.mro()]}")
    # MRO: D -> B -> C -> A -> object

    # 【菱形继承问题】
    print(f"\n--- 菱形继承 ---")

    class Base:
        def __init__(self):
            print("Base.__init__")

    class Left(Base):
        def __init__(self):
            print("Left.__init__")
            super().__init__()

    class Right(Base):
        def __init__(self):
            print("Right.__init__")
            super().__init__()

    class Child(Left, Right):
        def __init__(self):
            print("Child.__init__")
            super().__init__()

    print("创建 Child 实例:")
    child = Child()
    # super() 按 MRO 顺序调用，Base 只被调用一次


def main04_mixins():
    """
    ============================================================
                    4. Mixin 模式
    ============================================================
    Mixin 是一种特殊的多重继承用法，用于为类添加功能
    """
    print("\n" + "=" * 60)
    print("4. Mixin 模式")
    print("=" * 60)

    # 【Mixin 类】提供额外功能，通常不单独使用
    class JsonMixin:
        """提供 JSON 序列化功能"""
        def to_json(self):
            import json
            return json.dumps(self.__dict__)

        @classmethod
        def from_json(cls, json_str):
            import json
            data = json.loads(json_str)
            return cls(**data)

    class LogMixin:
        """提供日志功能"""
        def log(self, message):
            print(f"[{self.__class__.__name__}] {message}")

    class ValidateMixin:
        """提供验证功能"""
        def validate(self):
            for field, rules in getattr(self, '_validation_rules', {}).items():
                value = getattr(self, field, None)
                for rule in rules:
                    if not rule(value):
                        raise ValueError(f"{field} 验证失败")
            return True

    # 【使用 Mixin】
    class Person(JsonMixin, LogMixin):
        def __init__(self, name, age):
            self.name = name
            self.age = age

    p = Person("Alice", 25)
    print(f"JSON: {p.to_json()}")
    p.log("Person 创建完成")

    p2 = Person.from_json('{"name": "Bob", "age": 30}')
    print(f"从 JSON 创建: {p2.name}, {p2.age}")

    # 【Mixin 命名约定】
    # - 以 Mixin 结尾
    # - 不定义 __init__（或调用 super().__init__）
    # - 只提供方法，不提供状态


def main05_abstract_class():
    """
    ============================================================
                    5. 抽象基类 ABC
    ============================================================
    """
    print("\n" + "=" * 60)
    print("5. 抽象基类 ABC")
    print("=" * 60)

    # 【定义抽象基类】
    class Shape(ABC):
        """形状抽象基类"""

        @abstractmethod
        def area(self):
            """计算面积"""
            pass

        @abstractmethod
        def perimeter(self):
            """计算周长"""
            pass

        def describe(self):
            """通用方法（非抽象）"""
            return f"面积: {self.area():.2f}, 周长: {self.perimeter():.2f}"

    # 【实现抽象类】
    class Rectangle(Shape):
        def __init__(self, width, height):
            self.width = width
            self.height = height

        def area(self):
            return self.width * self.height

        def perimeter(self):
            return 2 * (self.width + self.height)

    class Circle(Shape):
        def __init__(self, radius):
            self.radius = radius

        def area(self):
            return 3.14159 * self.radius ** 2

        def perimeter(self):
            return 2 * 3.14159 * self.radius

    # 使用
    rect = Rectangle(4, 5)
    circle = Circle(3)

    print(f"Rectangle: {rect.describe()}")
    print(f"Circle: {circle.describe()}")

    # 【不能实例化抽象类】
    try:
        shape = Shape()
    except TypeError as e:
        print(f"\n不能实例化抽象类: {e}")

    # 【抽象属性】
    print(f"\n--- 抽象属性 ---")

    class Animal(ABC):
        @property
        @abstractmethod
        def species(self):
            pass

        @abstractmethod
        def speak(self):
            pass

    class Dog(Animal):
        @property
        def species(self):
            return "犬科"

        def speak(self):
            return "汪!"

    dog = Dog()
    print(f"Dog species: {dog.species}")


def main06_polymorphism():
    """
    ============================================================
                    6. 多态
    ============================================================
    """
    print("\n" + "=" * 60)
    print("6. 多态")
    print("=" * 60)

    class Animal(ABC):
        @abstractmethod
        def speak(self):
            pass

    class Dog(Animal):
        def speak(self):
            return "汪汪!"

    class Cat(Animal):
        def speak(self):
            return "喵喵!"

    class Duck(Animal):
        def speak(self):
            return "嘎嘎!"

    # 【多态：同一接口，不同实现】
    def make_animals_speak(animals):
        """让所有动物说话"""
        for animal in animals:
            print(f"  {animal.__class__.__name__}: {animal.speak()}")

    animals = [Dog(), Cat(), Duck()]
    print("让所有动物说话:")
    make_animals_speak(animals)

    # 【鸭子类型】Python 不需要继承也能实现多态
    print(f"\n--- 鸭子类型 ---")

    class Robot:
        """Robot 不继承 Animal，但有 speak 方法"""
        def speak(self):
            return "哔哔!"

    # 只要有 speak 方法就能工作
    everything = [Dog(), Cat(), Robot()]
    print("鸭子类型（不需要继承）:")
    for thing in everything:
        print(f"  {thing.__class__.__name__}: {thing.speak()}")


def main07_composition():
    """
    ============================================================
                7. 组合优于继承
    ============================================================
    """
    print("\n" + "=" * 60)
    print("7. 组合优于继承")
    print("=" * 60)

    # 【继承的问题】
    print("--- 继承的问题 ---")
    print("1. 紧耦合：子类依赖父类实现细节")
    print("2. 脆弱基类：修改父类可能破坏子类")
    print("3. 单继承限制（虽然 Python 支持多继承）")

    # 【使用组合】
    print("\n--- 使用组合 ---")

    class Engine:
        def start(self):
            return "引擎启动"

        def stop(self):
            return "引擎停止"

    class Wheels:
        def __init__(self, count=4):
            self.count = count

        def rotate(self):
            return f"{self.count} 个轮子转动"

    class Car:
        """组合：Car 有 Engine 和 Wheels"""

        def __init__(self):
            self.engine = Engine()
            self.wheels = Wheels(4)

        def drive(self):
            return f"{self.engine.start()}, {self.wheels.rotate()}"

        def park(self):
            return self.engine.stop()

    car = Car()
    print(f"car.drive(): {car.drive()}")
    print(f"car.park(): {car.park()}")

    # 【策略模式：运行时切换行为】
    print("\n--- 策略模式 ---")

    class FlyBehavior(ABC):
        @abstractmethod
        def fly(self):
            pass

    class FlyWithWings(FlyBehavior):
        def fly(self):
            return "扇动翅膀飞行"

    class FlyNoWay(FlyBehavior):
        def fly(self):
            return "不会飞"

    class FlyWithRocket(FlyBehavior):
        def fly(self):
            return "火箭飞行!"

    class Bird:
        def __init__(self, fly_behavior: FlyBehavior):
            self.fly_behavior = fly_behavior

        def perform_fly(self):
            return self.fly_behavior.fly()

        def set_fly_behavior(self, fb: FlyBehavior):
            self.fly_behavior = fb

    # 使用
    duck = Bird(FlyWithWings())
    print(f"Duck: {duck.perform_fly()}")

    penguin = Bird(FlyNoWay())
    print(f"Penguin: {penguin.perform_fly()}")

    # 运行时改变行为
    penguin.set_fly_behavior(FlyWithRocket())
    print(f"Penguin with rocket: {penguin.perform_fly()}")


def main08_dataclass_inheritance():
    """
    ============================================================
                8. dataclass 继承
    ============================================================
    """
    print("\n" + "=" * 60)
    print("8. dataclass 继承")
    print("=" * 60)

    from dataclasses import dataclass, field

    @dataclass
    class Person:
        name: str
        age: int

    @dataclass
    class Employee(Person):
        department: str
        salary: float = 0.0

    emp = Employee("Alice", 30, "Engineering", 50000.0)
    print(f"Employee: {emp}")

    # 【注意】父类有默认值时，子类必须也有默认值
    @dataclass
    class Base:
        x: int = 0  # 有默认值

    @dataclass
    class Derived(Base):
        y: int = 0  # 子类字段也必须有默认值
        # z: int  # 这会报错！

    d = Derived(x=1, y=2)
    print(f"Derived: {d}")


if __name__ == "__main__":
    main01_basic_inheritance()
    main02_super()
    main03_multiple_inheritance()
    main04_mixins()
    main05_abstract_class()
    main06_polymorphism()
    main07_composition()
    main08_dataclass_inheritance()
```
