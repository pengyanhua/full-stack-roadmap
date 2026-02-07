# protocols.py

::: info 文件信息
- 📄 原文件：`03_protocols.py`
- 🔤 语言：python
:::

Python 协议与鸭子类型
本文件介绍 Python 中的协议（Protocol）、鸭子类型和结构子类型。

"如果它走起来像鸭子，叫起来也像鸭子，那它就是鸭子。"

## 完整代码

```python
from typing import Protocol, runtime_checkable, Iterable, Iterator
from abc import ABC, abstractmethod


def main01_duck_typing():
    """
    ============================================================
                    1. 鸭子类型
    ============================================================
    Python 是动态类型语言，关注对象的行为而非类型
    """
    print("=" * 60)
    print("1. 鸭子类型")
    print("=" * 60)

    # 【鸭子类型基本概念】
    class Duck:
        def quack(self):
            return "嘎嘎!"

        def walk(self):
            return "摇摇摆摆走"

    class Person:
        def quack(self):
            return "我在模仿鸭子叫：嘎嘎!"

        def walk(self):
            return "正常走路"

    class Robot:
        def quack(self):
            return "合成声音：嘎嘎"

        def walk(self):
            return "机械行走"

    def make_it_quack(thing):
        """不关心类型，只关心有没有 quack 方法"""
        print(f"  {thing.__class__.__name__}: {thing.quack()}")

    print("鸭子类型演示:")
    for obj in [Duck(), Person(), Robot()]:
        make_it_quack(obj)

    # 【文件类对象】
    print(f"\n--- 文件类对象 ---")

    class StringWriter:
        """像文件一样的字符串写入器"""
        def __init__(self):
            self.content = []

        def write(self, text):
            self.content.append(text)

        def read(self):
            return ''.join(self.content)

    def write_greeting(file_like):
        """接受任何有 write 方法的对象"""
        file_like.write("Hello, ")
        file_like.write("World!")

    sw = StringWriter()
    write_greeting(sw)
    print(f"StringWriter 内容: {sw.read()}")


def main02_builtin_protocols():
    """
    ============================================================
                2. 内置协议（特殊方法）
    ============================================================
    Python 通过特殊方法实现内置协议
    """
    print("\n" + "=" * 60)
    print("2. 内置协议（特殊方法）")
    print("=" * 60)

    # 【可迭代协议】__iter__
    print("--- 可迭代协议 ---")

    class Countdown:
        def __init__(self, start):
            self.start = start

        def __iter__(self):
            n = self.start
            while n > 0:
                yield n
                n -= 1

    print(f"Countdown(5): {list(Countdown(5))}")

    # 【序列协议】__len__, __getitem__
    print(f"\n--- 序列协议 ---")

    class Sentence:
        def __init__(self, text):
            self.words = text.split()

        def __len__(self):
            return len(self.words)

        def __getitem__(self, index):
            return self.words[index]

    s = Sentence("Hello World Python")
    print(f"len(s) = {len(s)}")
    print(f"s[0] = {s[0]}")
    print(f"s[-1] = {s[-1]}")
    print(f"for word in s: {[w for w in s]}")  # 自动支持迭代

    # 【上下文管理协议】__enter__, __exit__
    print(f"\n--- 上下文管理协议 ---")

    class Timer:
        def __enter__(self):
            import time
            self.start = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            import time
            self.elapsed = time.perf_counter() - self.start
            print(f"  耗时: {self.elapsed*1000:.2f}ms")
            return False

    with Timer():
        sum(range(100000))

    # 【可调用协议】__call__
    print(f"\n--- 可调用协议 ---")

    class Multiplier:
        def __init__(self, factor):
            self.factor = factor

        def __call__(self, x):
            return x * self.factor

    double = Multiplier(2)
    triple = Multiplier(3)
    print(f"double(5) = {double(5)}")
    print(f"triple(5) = {triple(5)}")

    # 【哈希协议】__hash__, __eq__
    print(f"\n--- 哈希协议 ---")

    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __hash__(self):
            return hash((self.x, self.y))

        def __eq__(self, other):
            return self.x == other.x and self.y == other.y

    p1 = Point(1, 2)
    p2 = Point(1, 2)
    p3 = Point(3, 4)

    # 可以用作字典键和集合元素
    points = {p1: "origin", p3: "other"}
    print(f"p1 in points: {p1 in points}")
    print(f"p2 in points: {p2 in points}")  # True，因为 p1 == p2


def main03_typing_protocol():
    """
    ============================================================
                3. typing.Protocol（结构子类型）
    ============================================================
    Python 3.8+ 引入的正式协议支持
    """
    print("\n" + "=" * 60)
    print("3. typing.Protocol（结构子类型）")
    print("=" * 60)

    # 【定义协议】
    class Drawable(Protocol):
        """可绘制协议"""
        def draw(self) -> str:
            ...

    class Resizable(Protocol):
        """可调整大小协议"""
        def resize(self, factor: float) -> None:
            ...

    # 【实现协议（隐式）】
    class Circle:
        def __init__(self, radius: float):
            self.radius = radius

        def draw(self) -> str:
            return f"绘制圆形，半径={self.radius}"

        def resize(self, factor: float) -> None:
            self.radius *= factor

    class Rectangle:
        def __init__(self, width: float, height: float):
            self.width = width
            self.height = height

        def draw(self) -> str:
            return f"绘制矩形，{self.width}x{self.height}"

        def resize(self, factor: float) -> None:
            self.width *= factor
            self.height *= factor

    # 【使用协议作为类型提示】
    def render(shape: Drawable) -> None:
        print(f"  {shape.draw()}")

    def scale_up(shape: Resizable) -> None:
        shape.resize(2.0)

    print("协议类型检查（类型检查器使用）:")
    circle = Circle(5)
    rect = Rectangle(4, 3)

    render(circle)
    render(rect)

    # 【运行时可检查的协议】
    print(f"\n--- 运行时检查 ---")

    @runtime_checkable
    class Speakable(Protocol):
        def speak(self) -> str:
            ...

    class Dog:
        def speak(self) -> str:
            return "汪!"

    class Cat:
        def speak(self) -> str:
            return "喵!"

    class Rock:
        pass

    dog = Dog()
    rock = Rock()

    print(f"isinstance(dog, Speakable): {isinstance(dog, Speakable)}")
    print(f"isinstance(rock, Speakable): {isinstance(rock, Speakable)}")


def main04_protocol_vs_abc():
    """
    ============================================================
                4. Protocol vs ABC
    ============================================================
    """
    print("\n" + "=" * 60)
    print("4. Protocol vs ABC")
    print("=" * 60)

    # 【ABC：名义类型】需要显式继承
    class AnimalABC(ABC):
        @abstractmethod
        def speak(self) -> str:
            pass

    class DogABC(AnimalABC):  # 必须继承
        def speak(self) -> str:
            return "汪!"

    # 【Protocol：结构类型】不需要继承
    class AnimalProtocol(Protocol):
        def speak(self) -> str:
            ...

    class DogProtocol:  # 不需要继承，只要实现方法
        def speak(self) -> str:
            return "汪!"

    print("""
    ABC（抽象基类）:
    - 需要显式继承
    - 运行时强制检查
    - 适合定义接口规范

    Protocol（协议）:
    - 不需要继承（结构子类型）
    - 主要用于静态类型检查
    - 更灵活，适合鸭子类型
    """)

    # 【组合使用】
    @runtime_checkable
    class Closeable(Protocol):
        def close(self) -> None:
            ...

    class FileWrapper:
        def __init__(self, name):
            self.name = name

        def close(self) -> None:
            print(f"    关闭 {self.name}")

    def cleanup(resource: Closeable) -> None:
        resource.close()

    print("组合使用:")
    fw = FileWrapper("test.txt")
    print(f"  isinstance(fw, Closeable): {isinstance(fw, Closeable)}")
    cleanup(fw)


def main05_common_protocols():
    """
    ============================================================
                5. 常用协议示例
    ============================================================
    """
    print("\n" + "=" * 60)
    print("5. 常用协议示例")
    print("=" * 60)

    # 【Comparable 协议】
    print("--- Comparable 协议 ---")

    class Comparable(Protocol):
        def __lt__(self, other) -> bool: ...
        def __le__(self, other) -> bool: ...
        def __gt__(self, other) -> bool: ...
        def __ge__(self, other) -> bool: ...

    class Score:
        def __init__(self, value: int):
            self.value = value

        def __lt__(self, other: 'Score') -> bool:
            return self.value < other.value

        def __le__(self, other: 'Score') -> bool:
            return self.value <= other.value

        def __gt__(self, other: 'Score') -> bool:
            return self.value > other.value

        def __ge__(self, other: 'Score') -> bool:
            return self.value >= other.value

        def __repr__(self):
            return f"Score({self.value})"

    scores = [Score(85), Score(92), Score(78)]
    print(f"排序前: {scores}")
    print(f"排序后: {sorted(scores)}")

    # 【Hashable 协议】
    print(f"\n--- Hashable 协议 ---")

    @runtime_checkable
    class Hashable(Protocol):
        def __hash__(self) -> int: ...

    class ImmutablePoint:
        def __init__(self, x: int, y: int):
            self._x = x
            self._y = y

        def __hash__(self) -> int:
            return hash((self._x, self._y))

        def __eq__(self, other) -> bool:
            return self._x == other._x and self._y == other._y

    p = ImmutablePoint(1, 2)
    print(f"isinstance(p, Hashable): {isinstance(p, Hashable)}")
    print(f"可以用作字典键: {{{p}: 'point'}}")

    # 【SupportsAdd 协议】
    print(f"\n--- SupportsAdd 协议 ---")

    class SupportsAdd(Protocol):
        def __add__(self, other): ...

    def double(x: SupportsAdd):
        return x + x

    print(f"double(5) = {double(5)}")
    print(f"double('hello') = {double('hello')}")
    print(f"double([1, 2]) = {double([1, 2])}")


def main06_generic_protocols():
    """
    ============================================================
                6. 泛型协议
    ============================================================
    """
    print("\n" + "=" * 60)
    print("6. 泛型协议")
    print("=" * 60)

    from typing import TypeVar, Generic

    T = TypeVar('T')
    T_co = TypeVar('T_co', covariant=True)  # 协变

    # 【泛型协议】
    class Container(Protocol[T]):
        def get(self) -> T: ...
        def set(self, value: T) -> None: ...

    class Box(Generic[T]):
        def __init__(self, value: T):
            self._value = value

        def get(self) -> T:
            return self._value

        def set(self, value: T) -> None:
            self._value = value

    box: Container[int] = Box(42)
    print(f"box.get() = {box.get()}")
    box.set(100)
    print(f"box.get() = {box.get()}")

    # 【协变协议】
    print(f"\n--- 协变协议 ---")

    class Reader(Protocol[T_co]):
        def read(self) -> T_co: ...

    class StringReader:
        def read(self) -> str:
            return "Hello"

    def process_reader(reader: Reader[str]) -> str:
        return reader.read()

    sr = StringReader()
    print(f"process_reader(sr) = {process_reader(sr)}")


def main07_practical_examples():
    """
    ============================================================
                7. 实际应用示例
    ============================================================
    """
    print("\n" + "=" * 60)
    print("7. 实际应用示例")
    print("=" * 60)

    # 【存储协议】
    print("--- 存储协议 ---")

    class Storage(Protocol):
        def save(self, key: str, data: dict) -> None: ...
        def load(self, key: str) -> dict | None: ...
        def delete(self, key: str) -> None: ...

    class MemoryStorage:
        def __init__(self):
            self._data = {}

        def save(self, key: str, data: dict) -> None:
            self._data[key] = data
            print(f"    [Memory] 保存: {key}")

        def load(self, key: str) -> dict | None:
            return self._data.get(key)

        def delete(self, key: str) -> None:
            self._data.pop(key, None)
            print(f"    [Memory] 删除: {key}")

    class FileStorage:
        def __init__(self, directory: str):
            self.directory = directory

        def save(self, key: str, data: dict) -> None:
            print(f"    [File] 保存到 {self.directory}/{key}.json")

        def load(self, key: str) -> dict | None:
            print(f"    [File] 从 {self.directory}/{key}.json 加载")
            return {"mock": "data"}

        def delete(self, key: str) -> None:
            print(f"    [File] 删除 {self.directory}/{key}.json")

    class UserService:
        def __init__(self, storage: Storage):
            self.storage = storage

        def create_user(self, user_id: str, name: str) -> None:
            self.storage.save(user_id, {"name": name})

        def get_user(self, user_id: str) -> dict | None:
            return self.storage.load(user_id)

    # 可以轻松切换存储实现
    print("使用内存存储:")
    service1 = UserService(MemoryStorage())
    service1.create_user("1", "Alice")

    print("\n使用文件存储:")
    service2 = UserService(FileStorage("/data"))
    service2.create_user("1", "Alice")

    # 【日志协议】
    print(f"\n--- 日志协议 ---")

    class Logger(Protocol):
        def info(self, msg: str) -> None: ...
        def error(self, msg: str) -> None: ...

    class ConsoleLogger:
        def info(self, msg: str) -> None:
            print(f"    [INFO] {msg}")

        def error(self, msg: str) -> None:
            print(f"    [ERROR] {msg}")

    class FileLogger:
        def __init__(self, filename: str):
            self.filename = filename

        def info(self, msg: str) -> None:
            print(f"    [INFO -> {self.filename}] {msg}")

        def error(self, msg: str) -> None:
            print(f"    [ERROR -> {self.filename}] {msg}")

    def process_data(data: list, logger: Logger) -> None:
        logger.info(f"开始处理 {len(data)} 条数据")
        # 处理逻辑
        logger.info("处理完成")

    print("使用控制台日志:")
    process_data([1, 2, 3], ConsoleLogger())

    print("\n使用文件日志:")
    process_data([1, 2, 3], FileLogger("app.log"))


if __name__ == "__main__":
    main01_duck_typing()
    main02_builtin_protocols()
    main03_typing_protocol()
    main04_protocol_vs_abc()
    main05_common_protocols()
    main06_generic_protocols()
    main07_practical_examples()
```
