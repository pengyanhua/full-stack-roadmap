# sequences.py

::: info 文件信息
- 📄 原文件：`03_sequences.py`
- 🔤 语言：python
:::

Python 序列类型：列表、元组、集合
本文件介绍 Python 中的主要序列类型及其操作。

主要序列类型：
- list: 可变有序序列
- tuple: 不可变有序序列
- set: 可变无序集合（元素唯一）
- frozenset: 不可变集合

## 完整代码

```python
def main01_list_basics():
    """
    ============================================================
                        1. 列表基础
    ============================================================
    列表（list）是 Python 中最常用的数据结构
    - 可变（mutable）
    - 有序
    - 可包含任意类型元素
    """
    print("=" * 60)
    print("1. 列表基础")
    print("=" * 60)

    # 【创建列表】
    empty_list = []              # 空列表
    empty_list2 = list()         # 使用 list()
    numbers = [1, 2, 3, 4, 5]    # 数字列表
    mixed = [1, "hello", 3.14, True]  # 混合类型
    nested = [[1, 2], [3, 4]]    # 嵌套列表

    print(f"空列表: {empty_list}")
    print(f"数字列表: {numbers}")
    print(f"混合列表: {mixed}")
    print(f"嵌套列表: {nested}")

    # 【从其他类型转换】
    from_string = list("hello")
    from_range = list(range(5))
    print(f"\n从字符串: {from_string}")
    print(f"从 range: {from_range}")

    # 【列表索引】
    fruits = ["apple", "banana", "cherry", "date"]
    print(f"\n列表: {fruits}")
    print(f"fruits[0] = {fruits[0]}")      # 第一个元素
    print(f"fruits[-1] = {fruits[-1]}")    # 最后一个元素
    print(f"fruits[-2] = {fruits[-2]}")    # 倒数第二个

    # 【列表切片】
    print(f"\n切片操作:")
    print(f"fruits[1:3] = {fruits[1:3]}")    # [banana, cherry]
    print(f"fruits[:2] = {fruits[:2]}")      # 前两个
    print(f"fruits[2:] = {fruits[2:]}")      # 从索引2到末尾
    print(f"fruits[::2] = {fruits[::2]}")    # 步长为2
    print(f"fruits[::-1] = {fruits[::-1]}")  # 反转

    # 【修改元素】
    fruits[0] = "apricot"
    print(f"\n修改后: {fruits}")

    # 【切片赋值】
    numbers = [1, 2, 3, 4, 5]
    numbers[1:4] = [20, 30]  # 替换多个元素
    print(f"切片赋值: {numbers}")


def main02_list_methods():
    """
    ============================================================
                        2. 列表方法
    ============================================================
    """
    print("\n" + "=" * 60)
    print("2. 列表方法")
    print("=" * 60)

    # 【添加元素】
    fruits = ["apple", "banana"]
    print(f"原列表: {fruits}")

    fruits.append("cherry")       # 末尾添加单个元素
    print(f"append: {fruits}")

    fruits.insert(1, "apricot")   # 在指定位置插入
    print(f"insert(1, ...): {fruits}")

    fruits.extend(["date", "elderberry"])  # 扩展列表
    print(f"extend: {fruits}")

    # 【技巧】+ 运算符创建新列表，extend 修改原列表
    new_list = fruits + ["fig"]
    print(f"+ 运算符: {new_list}")

    # 【删除元素】
    print("\n--- 删除元素 ---")
    numbers = [1, 2, 3, 2, 4, 2, 5]
    print(f"原列表: {numbers}")

    numbers.remove(2)  # 删除第一个匹配的值
    print(f"remove(2): {numbers}")

    popped = numbers.pop()  # 删除并返回最后一个元素
    print(f"pop(): 返回 {popped}, 列表 {numbers}")

    popped = numbers.pop(0)  # 删除并返回指定位置的元素
    print(f"pop(0): 返回 {popped}, 列表 {numbers}")

    del numbers[0]  # 使用 del 删除
    print(f"del numbers[0]: {numbers}")

    numbers.clear()  # 清空列表
    print(f"clear(): {numbers}")

    # 【查找和计数】
    print("\n--- 查找和计数 ---")
    letters = ['a', 'b', 'c', 'b', 'd', 'b']
    print(f"列表: {letters}")
    print(f"index('b'): {letters.index('b')}")     # 第一个 'b' 的索引
    print(f"count('b'): {letters.count('b')}")     # 'b' 出现的次数
    print(f"'b' in letters: {'b' in letters}")     # 成员检测

    # 【排序】
    print("\n--- 排序 ---")
    numbers = [3, 1, 4, 1, 5, 9, 2, 6]
    print(f"原列表: {numbers}")

    # sort() 原地排序，修改原列表
    numbers.sort()
    print(f"sort(): {numbers}")

    numbers.sort(reverse=True)  # 降序
    print(f"sort(reverse=True): {numbers}")

    # sorted() 返回新列表，不修改原列表
    words = ["banana", "Apple", "cherry"]
    sorted_words = sorted(words)
    print(f"\nsorted(): {sorted_words}")

    # 自定义排序键
    sorted_by_len = sorted(words, key=len)
    print(f"按长度排序: {sorted_by_len}")

    sorted_lower = sorted(words, key=str.lower)
    print(f"忽略大小写: {sorted_lower}")

    # 【反转】
    print("\n--- 反转 ---")
    numbers = [1, 2, 3, 4, 5]
    numbers.reverse()  # 原地反转
    print(f"reverse(): {numbers}")

    # reversed() 返回迭代器
    print(f"reversed(): {list(reversed(numbers))}")

    # 【复制】
    print("\n--- 复制 ---")
    original = [1, 2, [3, 4]]

    # 浅拷贝
    shallow1 = original.copy()
    shallow2 = original[:]
    shallow3 = list(original)

    print(f"原列表: {original}")
    print(f"浅拷贝: {shallow1}")

    # 【警告】浅拷贝只复制一层！
    shallow1[2][0] = 99
    print(f"修改浅拷贝后，原列表也变了: {original}")

    # 【技巧】深拷贝
    import copy
    original = [1, 2, [3, 4]]
    deep = copy.deepcopy(original)
    deep[2][0] = 99
    print(f"\n深拷贝后修改，原列表不变: {original}")


def main03_tuple():
    """
    ============================================================
                        3. 元组 tuple
    ============================================================
    元组是不可变的有序序列
    - 不可变（immutable）
    - 有序
    - 可哈希（可作为字典键）
    """
    print("\n" + "=" * 60)
    print("3. 元组 tuple")
    print("=" * 60)

    # 【创建元组】
    empty_tuple = ()
    empty_tuple2 = tuple()
    single = (1,)          # 【注意】单元素元组需要逗号！
    not_tuple = (1)        # 这是整数，不是元组！
    numbers = (1, 2, 3)
    mixed = (1, "hello", 3.14)

    print(f"空元组: {empty_tuple}")
    print(f"单元素元组: {single}, 类型: {type(single)}")
    print(f"(1) 的类型: {type(not_tuple)}")  # int!
    print(f"数字元组: {numbers}")

    # 【元组解包】
    print("\n--- 元组解包 ---")
    point = (10, 20, 30)
    x, y, z = point
    print(f"解包: x={x}, y={y}, z={z}")

    # 星号解包
    first, *rest = (1, 2, 3, 4, 5)
    print(f"星号解包: first={first}, rest={rest}")

    *start, last = (1, 2, 3, 4, 5)
    print(f"星号解包: start={start}, last={last}")

    # 【元组操作】
    print("\n--- 元组操作 ---")
    t1 = (1, 2, 3)
    t2 = (4, 5, 6)

    print(f"拼接: {t1 + t2}")
    print(f"重复: {t1 * 2}")
    print(f"长度: {len(t1)}")
    print(f"成员检测: {2 in t1}")
    print(f"索引: {t1[1]}")
    print(f"切片: {t1[1:]}")

    # 【元组方法】（只有两个）
    t = (1, 2, 3, 2, 4, 2)
    print(f"\nindex(2): {t.index(2)}")
    print(f"count(2): {t.count(2)}")

    # 【命名元组】更具可读性
    print("\n--- 命名元组 ---")
    from collections import namedtuple

    Point = namedtuple('Point', ['x', 'y', 'z'])
    p = Point(10, 20, 30)

    print(f"命名元组: {p}")
    print(f"p.x = {p.x}")
    print(f"p[0] = {p[0]}")
    print(f"解包: {p.x}, {p.y}, {p.z}")

    # 转换为字典
    print(f"_asdict(): {p._asdict()}")

    # 【技巧】使用 typing.NamedTuple（Python 3.6+）
    from typing import NamedTuple

    class Person(NamedTuple):
        name: str
        age: int
        city: str = "Unknown"  # 带默认值

    person = Person("Alice", 25)
    print(f"\nPerson: {person}")
    print(f"person.name = {person.name}")

    # 【元组 vs 列表】
    print("\n--- 元组 vs 列表 ---")
    print("元组优势:")
    print("  1. 不可变，更安全")
    print("  2. 可哈希，能作为字典键")
    print("  3. 内存占用更小")
    print("  4. 访问速度略快")


def main04_set():
    """
    ============================================================
                        4. 集合 set
    ============================================================
    集合是无序的、元素唯一的可变容器
    - 元素唯一（自动去重）
    - 无序
    - 元素必须是可哈希的
    """
    print("\n" + "=" * 60)
    print("4. 集合 set")
    print("=" * 60)

    # 【创建集合】
    empty_set = set()    # 【注意】{} 是空字典，不是空集合！
    numbers = {1, 2, 3, 4, 5}
    from_list = set([1, 2, 2, 3, 3, 3])  # 自动去重

    print(f"空集合: {empty_set}")
    print(f"数字集合: {numbers}")
    print(f"从列表创建（去重）: {from_list}")

    # 【添加和删除】
    print("\n--- 添加和删除 ---")
    s = {1, 2, 3}
    print(f"原集合: {s}")

    s.add(4)
    print(f"add(4): {s}")

    s.update([5, 6, 7])
    print(f"update([5, 6, 7]): {s}")

    s.remove(7)  # 元素不存在会报错
    print(f"remove(7): {s}")

    s.discard(100)  # 元素不存在不会报错
    print(f"discard(100): {s}")

    popped = s.pop()  # 随机删除一个元素
    print(f"pop(): 返回 {popped}, 集合 {s}")

    # 【集合运算】（集合论）
    print("\n--- 集合运算 ---")
    a = {1, 2, 3, 4, 5}
    b = {4, 5, 6, 7, 8}
    print(f"a = {a}")
    print(f"b = {b}")

    # 并集
    print(f"\n并集 a | b: {a | b}")
    print(f"并集 a.union(b): {a.union(b)}")

    # 交集
    print(f"\n交集 a & b: {a & b}")
    print(f"交集 a.intersection(b): {a.intersection(b)}")

    # 差集
    print(f"\n差集 a - b: {a - b}")
    print(f"差集 a.difference(b): {a.difference(b)}")

    # 对称差集（异或）
    print(f"\n对称差集 a ^ b: {a ^ b}")
    print(f"对称差集 a.symmetric_difference(b): {a.symmetric_difference(b)}")

    # 【子集和超集】
    print("\n--- 子集和超集 ---")
    small = {1, 2}
    big = {1, 2, 3, 4, 5}

    print(f"small = {small}")
    print(f"big = {big}")
    print(f"small <= big (子集): {small <= big}")
    print(f"small < big (真子集): {small < big}")
    print(f"big >= small (超集): {big >= small}")
    print(f"small.issubset(big): {small.issubset(big)}")
    print(f"big.issuperset(small): {big.issuperset(small)}")

    # 【不相交判断】
    c = {10, 20}
    print(f"\nc = {c}")
    print(f"a.isdisjoint(c): {a.isdisjoint(c)}")  # 无交集返回 True

    # 【frozenset 不可变集合】
    print("\n--- frozenset 不可变集合 ---")
    fs = frozenset([1, 2, 3])
    print(f"frozenset: {fs}")
    # fs.add(4)  # 报错！不能修改

    # frozenset 可以作为字典键或集合元素
    d = {fs: "value"}
    print(f"frozenset 作为字典键: {d}")


def main05_common_operations():
    """
    ============================================================
                    5. 序列通用操作
    ============================================================
    """
    print("\n" + "=" * 60)
    print("5. 序列通用操作")
    print("=" * 60)

    # 以下操作适用于所有序列类型

    # 【成员测试】
    print("--- 成员测试 ---")
    seq = [1, 2, 3, 4, 5]
    print(f"3 in {seq}: {3 in seq}")
    print(f"10 not in {seq}: {10 not in seq}")

    # 【拼接和重复】
    print("\n--- 拼接和重复 ---")
    print(f"[1, 2] + [3, 4]: {[1, 2] + [3, 4]}")
    print(f"(1, 2) + (3, 4): {(1, 2) + (3, 4)}")
    print(f"[1, 2] * 3: {[1, 2] * 3}")

    # 【长度、最大、最小、求和】
    print("\n--- 聚合函数 ---")
    numbers = [3, 1, 4, 1, 5, 9, 2, 6]
    print(f"列表: {numbers}")
    print(f"len(): {len(numbers)}")
    print(f"max(): {max(numbers)}")
    print(f"min(): {min(numbers)}")
    print(f"sum(): {sum(numbers)}")

    # 【any 和 all】
    print("\n--- any 和 all ---")
    bools = [True, True, False]
    print(f"列表: {bools}")
    print(f"all(): {all(bools)}")  # 所有为 True 才返回 True
    print(f"any(): {any(bools)}")  # 有一个 True 就返回 True

    # 空列表
    print(f"all([]): {all([])}")  # True（空序列）
    print(f"any([]): {any([])}")  # False

    # 【enumerate 和 zip】
    print("\n--- enumerate 和 zip ---")
    names = ["Alice", "Bob"]
    ages = [25, 30]

    for i, name in enumerate(names):
        print(f"  {i}: {name}")

    for name, age in zip(names, ages):
        print(f"  {name}: {age}岁")

    # 【技巧】同时获取最大值和索引
    numbers = [3, 1, 4, 1, 5, 9, 2, 6]
    max_idx, max_val = max(enumerate(numbers), key=lambda x: x[1])
    print(f"\n最大值 {max_val} 在索引 {max_idx}")

    # 【filter 和 map】
    print("\n--- filter 和 map ---")
    numbers = [1, 2, 3, 4, 5, 6]

    # filter: 过滤
    evens = list(filter(lambda x: x % 2 == 0, numbers))
    print(f"filter 偶数: {evens}")

    # map: 映射
    squares = list(map(lambda x: x**2, numbers))
    print(f"map 平方: {squares}")

    # 【技巧】推导式通常更 Pythonic
    evens = [x for x in numbers if x % 2 == 0]
    squares = [x**2 for x in numbers]

    # 【reduce】
    print("\n--- reduce ---")
    from functools import reduce
    numbers = [1, 2, 3, 4, 5]
    product = reduce(lambda x, y: x * y, numbers)
    print(f"reduce 乘积: {product}")


if __name__ == "__main__":
    main01_list_basics()
    main02_list_methods()
    main03_tuple()
    main04_set()
    main05_common_operations()
```
