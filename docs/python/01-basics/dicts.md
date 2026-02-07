# dicts.py

::: info 文件信息
- 📄 原文件：`04_dicts.py`
- 🔤 语言：python
:::

Python 字典 dict
本文件介绍 Python 中的字典类型及其操作。

字典（dict）是 Python 中最重要的数据结构之一：
- 键值对映射
- 键必须是可哈希的（不可变类型）
- 值可以是任意类型
- Python 3.7+ 保证插入顺序

## 完整代码

```python
def main01_dict_basics():
    """
    ============================================================
                        1. 字典基础
    ============================================================
    """
    print("=" * 60)
    print("1. 字典基础")
    print("=" * 60)

    # 【创建字典】
    empty_dict = {}
    empty_dict2 = dict()

    # 花括号语法
    person = {
        "name": "Alice",
        "age": 25,
        "city": "Beijing"
    }

    # dict() 构造函数
    person2 = dict(name="Bob", age=30, city="Shanghai")

    # 从键值对列表创建
    items = [("a", 1), ("b", 2), ("c", 3)]
    from_pairs = dict(items)

    # 使用 dict.fromkeys() 创建
    keys = ["x", "y", "z"]
    from_keys = dict.fromkeys(keys, 0)  # 所有键的值都是 0

    print(f"空字典: {empty_dict}")
    print(f"person: {person}")
    print(f"person2: {person2}")
    print(f"从键值对: {from_pairs}")
    print(f"fromkeys: {from_keys}")

    # 【访问值】
    print("\n--- 访问值 ---")
    print(f"person['name']: {person['name']}")

    # 【警告】键不存在会抛出 KeyError
    # print(person['salary'])  # KeyError!

    # 【技巧】使用 get() 安全访问
    print(f"person.get('name'): {person.get('name')}")
    print(f"person.get('salary'): {person.get('salary')}")  # None
    print(f"person.get('salary', 0): {person.get('salary', 0)}")  # 默认值

    # 【修改和添加】
    print("\n--- 修改和添加 ---")
    person["age"] = 26          # 修改
    person["email"] = "a@b.com"  # 添加
    print(f"修改后: {person}")

    # 【setdefault】如果键不存在则设置默认值
    person.setdefault("country", "China")
    person.setdefault("age", 100)  # 已存在，不会修改
    print(f"setdefault 后: {person}")

    # 【删除】
    print("\n--- 删除 ---")
    del person["email"]
    print(f"del email: {person}")

    age = person.pop("age")
    print(f"pop('age'): 返回 {age}, 字典 {person}")

    # pop 带默认值，键不存在不会报错
    salary = person.pop("salary", 0)
    print(f"pop('salary', 0): {salary}")

    # popitem() 删除并返回最后一个键值对
    d = {"a": 1, "b": 2, "c": 3}
    item = d.popitem()
    print(f"popitem(): 返回 {item}, 字典 {d}")


def main02_dict_operations():
    """
    ============================================================
                        2. 字典操作
    ============================================================
    """
    print("\n" + "=" * 60)
    print("2. 字典操作")
    print("=" * 60)

    person = {"name": "Alice", "age": 25, "city": "Beijing"}

    # 【遍历】
    print("--- 遍历 ---")

    print("遍历键:")
    for key in person:
        print(f"  {key}")

    print("\n遍历键（显式）:")
    for key in person.keys():
        print(f"  {key}")

    print("\n遍历值:")
    for value in person.values():
        print(f"  {value}")

    print("\n遍历键值对:")
    for key, value in person.items():
        print(f"  {key}: {value}")

    # 【成员测试】
    print("\n--- 成员测试 ---")
    print(f"'name' in person: {'name' in person}")
    print(f"'salary' in person: {'salary' in person}")
    print(f"'Alice' in person.values(): {'Alice' in person.values()}")

    # 【合并字典】
    print("\n--- 合并字典 ---")
    d1 = {"a": 1, "b": 2}
    d2 = {"b": 3, "c": 4}

    # update() 方法（修改 d1）
    d1_copy = d1.copy()
    d1_copy.update(d2)
    print(f"update: {d1_copy}")

    # | 运算符（Python 3.9+）
    merged = d1 | d2
    print(f"| 运算符: {merged}")

    # |= 运算符（Python 3.9+）
    d1_copy = d1.copy()
    d1_copy |= d2
    print(f"|= 运算符: {d1_copy}")

    # 解包合并
    merged = {**d1, **d2}
    print(f"** 解包: {merged}")

    # 【字典视图】
    print("\n--- 字典视图 ---")
    d = {"a": 1, "b": 2, "c": 3}
    keys = d.keys()
    values = d.values()
    items = d.items()

    print(f"keys(): {keys}, 类型: {type(keys)}")
    print(f"values(): {values}")
    print(f"items(): {items}")

    # 【重要】视图是动态的！
    d["d"] = 4
    print(f"添加元素后，keys 视图自动更新: {keys}")

    # 【键的视图支持集合运算】
    d1 = {"a": 1, "b": 2}
    d2 = {"b": 3, "c": 4}
    print(f"\n两个字典共有的键: {d1.keys() & d2.keys()}")
    print(f"d1 独有的键: {d1.keys() - d2.keys()}")


def main03_dict_comprehension():
    """
    ============================================================
                        3. 字典推导式
    ============================================================
    """
    print("\n" + "=" * 60)
    print("3. 字典推导式")
    print("=" * 60)

    # 【基本推导式】
    squares = {x: x**2 for x in range(5)}
    print(f"平方字典: {squares}")

    # 【带条件】
    even_squares = {x: x**2 for x in range(10) if x % 2 == 0}
    print(f"偶数平方: {even_squares}")

    # 【从列表创建】
    names = ["alice", "bob", "charlie"]
    name_lengths = {name: len(name) for name in names}
    print(f"名字长度: {name_lengths}")

    # 【键值互换】
    original = {"a": 1, "b": 2, "c": 3}
    swapped = {v: k for k, v in original.items()}
    print(f"原字典: {original}")
    print(f"键值互换: {swapped}")

    # 【过滤字典】
    scores = {"Alice": 85, "Bob": 72, "Charlie": 90, "David": 65}
    passed = {name: score for name, score in scores.items() if score >= 70}
    print(f"及格学生: {passed}")

    # 【嵌套推导式】
    matrix = {
        i: {j: i * j for j in range(1, 4)}
        for i in range(1, 4)
    }
    print(f"乘法表: {matrix}")


def main04_defaultdict():
    """
    ============================================================
                    4. defaultdict 默认字典
    ============================================================
    defaultdict 在访问不存在的键时自动创建默认值
    """
    print("\n" + "=" * 60)
    print("4. defaultdict 默认字典")
    print("=" * 60)

    from collections import defaultdict

    # 【默认值为 int（0）】
    word_count = defaultdict(int)
    text = "apple banana apple cherry banana apple"
    for word in text.split():
        word_count[word] += 1  # 不存在的键自动创建为 0
    print(f"单词计数: {dict(word_count)}")

    # 【默认值为 list】
    groups = defaultdict(list)
    students = [
        ("Alice", "A"),
        ("Bob", "B"),
        ("Charlie", "A"),
        ("David", "B"),
    ]
    for name, grade in students:
        groups[grade].append(name)
    print(f"\n按成绩分组: {dict(groups)}")

    # 【默认值为 set】
    tags = defaultdict(set)
    articles = [
        ("Article1", "python"),
        ("Article1", "programming"),
        ("Article2", "python"),
        ("Article1", "python"),  # 重复，会被去重
    ]
    for article, tag in articles:
        tags[article].add(tag)
    print(f"\n文章标签: {dict(tags)}")

    # 【自定义默认值】
    def default_person():
        return {"name": "Unknown", "age": 0}

    people = defaultdict(default_person)
    print(f"\n访问不存在的键: {people['new_person']}")

    # 【嵌套 defaultdict】
    nested = defaultdict(lambda: defaultdict(int))
    nested["fruit"]["apple"] += 1
    nested["fruit"]["banana"] += 2
    nested["vegetable"]["carrot"] += 3
    print(f"\n嵌套 defaultdict: {dict(nested['fruit'])}")


def main05_counter():
    """
    ============================================================
                    5. Counter 计数器
    ============================================================
    Counter 是专门用于计数的字典子类
    """
    print("\n" + "=" * 60)
    print("5. Counter 计数器")
    print("=" * 60)

    from collections import Counter

    # 【创建 Counter】
    # 从可迭代对象
    text = "abracadabra"
    counter = Counter(text)
    print(f"字符计数: {counter}")

    # 从列表
    words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
    word_counter = Counter(words)
    print(f"单词计数: {word_counter}")

    # 从字典
    c = Counter({"a": 3, "b": 2})
    print(f"从字典创建: {c}")

    # 【访问计数】
    print(f"\ncounter['a']: {counter['a']}")
    print(f"counter['x']: {counter['x']}")  # 不存在返回 0，不报错！

    # 【most_common】获取最常见元素
    print(f"\n最常见的 2 个字符: {counter.most_common(2)}")
    print(f"所有元素按频率排序: {counter.most_common()}")

    # 【elements】返回迭代器，按计数重复元素
    c = Counter(a=3, b=2)
    print(f"elements(): {list(c.elements())}")

    # 【Counter 运算】
    print("\n--- Counter 运算 ---")
    c1 = Counter(a=3, b=1)
    c2 = Counter(a=1, b=2)
    print(f"c1 = {c1}")
    print(f"c2 = {c2}")

    print(f"c1 + c2: {c1 + c2}")  # 相加
    print(f"c1 - c2: {c1 - c2}")  # 相减（只保留正数）
    print(f"c1 & c2: {c1 & c2}")  # 交集（取最小）
    print(f"c1 | c2: {c1 | c2}")  # 并集（取最大）

    # 【更新计数】
    print("\n--- 更新计数 ---")
    c = Counter(a=3, b=2)
    c.update("aab")  # 增加计数
    print(f"update 后: {c}")

    c.subtract("ab")  # 减少计数
    print(f"subtract 后: {c}")

    # 【total】计数总和（Python 3.10+）
    c = Counter(a=3, b=2)
    print(f"\ntotal(): {c.total()}")


def main06_ordereddict():
    """
    ============================================================
                    6. OrderedDict 有序字典
    ============================================================
    【注意】Python 3.7+ 普通 dict 已保证顺序
    OrderedDict 仍有一些特殊功能
    """
    print("\n" + "=" * 60)
    print("6. OrderedDict 有序字典")
    print("=" * 60)

    from collections import OrderedDict

    # 【创建】
    od = OrderedDict()
    od["first"] = 1
    od["second"] = 2
    od["third"] = 3
    print(f"OrderedDict: {od}")

    # 【move_to_end】移动到末尾或开头
    od.move_to_end("first")  # 移到末尾
    print(f"move_to_end('first'): {od}")

    od.move_to_end("third", last=False)  # 移到开头
    print(f"move_to_end('third', last=False): {od}")

    # 【popitem】可以指定从开头还是末尾弹出
    item = od.popitem(last=False)  # 弹出第一个
    print(f"popitem(last=False): {item}")

    # 【比较相等】OrderedDict 比较时考虑顺序
    d1 = OrderedDict([("a", 1), ("b", 2)])
    d2 = OrderedDict([("b", 2), ("a", 1)])
    print(f"\nOrderedDict 相等比较（考虑顺序）:")
    print(f"d1: {d1}")
    print(f"d2: {d2}")
    print(f"d1 == d2: {d1 == d2}")  # False

    # 普通 dict 不考虑顺序
    d3 = {"a": 1, "b": 2}
    d4 = {"b": 2, "a": 1}
    print(f"\n普通 dict 相等比较（不考虑顺序）:")
    print(f"d3 == d4: {d3 == d4}")  # True


def main07_chainmap():
    """
    ============================================================
                    7. ChainMap 链式映射
    ============================================================
    ChainMap 将多个字典链接成一个视图
    """
    print("\n" + "=" * 60)
    print("7. ChainMap 链式映射")
    print("=" * 60)

    from collections import ChainMap

    # 【创建】
    defaults = {"color": "red", "size": "medium"}
    user_settings = {"color": "blue"}

    settings = ChainMap(user_settings, defaults)
    print(f"defaults: {defaults}")
    print(f"user_settings: {user_settings}")
    print(f"ChainMap: {dict(settings)}")

    # 【查找顺序】从前往后
    print(f"\nsettings['color']: {settings['color']}")   # blue（来自 user_settings）
    print(f"settings['size']: {settings['size']}")       # medium（来自 defaults）

    # 【修改只影响第一个字典】
    settings["theme"] = "dark"
    print(f"\n添加 'theme' 后:")
    print(f"user_settings: {user_settings}")
    print(f"defaults: {defaults}")

    # 【maps 属性】
    print(f"\nmaps: {settings.maps}")

    # 【new_child】创建新的子上下文
    child = settings.new_child({"color": "green"})
    print(f"\nnew_child: {dict(child)}")

    # 【parents】获取父链
    print(f"parents: {dict(child.parents)}")


def main08_practical_examples():
    """
    ============================================================
                    8. 实用示例
    ============================================================
    """
    print("\n" + "=" * 60)
    print("8. 实用示例")
    print("=" * 60)

    # 【分组数据】
    print("--- 分组数据 ---")
    from collections import defaultdict

    data = [
        {"name": "Alice", "department": "Engineering"},
        {"name": "Bob", "department": "Sales"},
        {"name": "Charlie", "department": "Engineering"},
        {"name": "David", "department": "Sales"},
        {"name": "Eve", "department": "Engineering"},
    ]

    by_dept = defaultdict(list)
    for item in data:
        by_dept[item["department"]].append(item["name"])

    print(f"按部门分组: {dict(by_dept)}")

    # 【统计词频】
    print("\n--- 统计词频 ---")
    from collections import Counter

    text = """
    Python is a great programming language.
    Python is easy to learn.
    Python has great libraries.
    """
    words = text.lower().split()
    word_freq = Counter(words)
    print(f"前 3 高频词: {word_freq.most_common(3)}")

    # 【倒排索引】
    print("\n--- 倒排索引 ---")
    documents = [
        "Python is great",
        "Java is also great",
        "Python and Java are programming languages"
    ]

    index = defaultdict(set)
    for doc_id, doc in enumerate(documents):
        for word in doc.lower().split():
            index[word].add(doc_id)

    print(f"'python' 出现在文档: {index['python']}")
    print(f"'great' 出现在文档: {index['great']}")

    # 【缓存/记忆化】
    print("\n--- 简单缓存 ---")

    cache = {}

    def fibonacci(n):
        if n in cache:
            return cache[n]
        if n <= 1:
            result = n
        else:
            result = fibonacci(n - 1) + fibonacci(n - 2)
        cache[n] = result
        return result

    print(f"fibonacci(10) = {fibonacci(10)}")
    print(f"缓存: {cache}")

    # 【技巧】使用 functools.lru_cache 更简洁
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def fib(n):
        if n <= 1:
            return n
        return fib(n - 1) + fib(n - 2)

    print(f"\nfib(30) = {fib(30)}")
    print(f"缓存信息: {fib.cache_info()}")


if __name__ == "__main__":
    main01_dict_basics()
    main02_dict_operations()
    main03_dict_comprehension()
    main04_defaultdict()
    main05_counter()
    main06_ordereddict()
    main07_chainmap()
    main08_practical_examples()
```
