# re collections.py

::: info 文件信息
- 📄 原文件：`03_re_collections.py`
- 🔤 语言：python
:::

Python 标准库：re 和 collections 模块
本文件介绍正则表达式和高级集合类型。

## 完整代码

```python
import re
from collections import (
    Counter, defaultdict, OrderedDict,
    namedtuple, deque, ChainMap
)


def main01_re_basics():
    """
    ============================================================
                    1. 正则表达式基础
    ============================================================
    """
    print("=" * 60)
    print("1. 正则表达式基础")
    print("=" * 60)

    # 【基本匹配】
    print("--- 基本匹配 ---")
    text = "Hello, World! Hello, Python!"

    # match: 从开头匹配
    match = re.match(r"Hello", text)
    print(f"match('Hello'): {match.group() if match else None}")

    # search: 搜索第一个匹配
    search = re.search(r"World", text)
    print(f"search('World'): {search.group() if search else None}")

    # findall: 找出所有匹配
    all_matches = re.findall(r"Hello", text)
    print(f"findall('Hello'): {all_matches}")

    # finditer: 返回迭代器
    print("finditer('Hello'):")
    for m in re.finditer(r"Hello", text):
        print(f"  位置 {m.start()}-{m.end()}: {m.group()}")


def main02_re_patterns():
    """
    ============================================================
                    2. 正则表达式模式
    ============================================================
    """
    print("\n" + "=" * 60)
    print("2. 正则表达式模式")
    print("=" * 60)

    print("""
    【常用元字符】
    .       匹配任意字符（除换行符）
    ^       匹配开头
    $       匹配结尾
    *       匹配 0 次或多次
    +       匹配 1 次或多次
    ?       匹配 0 次或 1 次
    {n}     匹配 n 次
    {n,m}   匹配 n 到 m 次
    []      字符集
    |       或
    ()      分组

    【特殊序列】
    \d      数字 [0-9]
    \D      非数字
    \w      单词字符 [a-zA-Z0-9_]
    \W      非单词字符
    \s      空白字符
    \S      非空白字符
    \b      单词边界
    """)

    # 【示例】
    text = "My email is test@example.com and phone is 123-456-7890"

    # 匹配邮箱
    email = re.search(r'\w+@\w+\.\w+', text)
    print(f"邮箱: {email.group() if email else None}")

    # 匹配电话
    phone = re.search(r'\d{3}-\d{3}-\d{4}', text)
    print(f"电话: {phone.group() if phone else None}")

    # 【字符集】
    print(f"\n--- 字符集 ---")
    text = "abc123ABC"
    print(f"[a-z]+: {re.findall(r'[a-z]+', text)}")
    print(f"[A-Z]+: {re.findall(r'[A-Z]+', text)}")
    print(f"[0-9]+: {re.findall(r'[0-9]+', text)}")
    print(f"[^a-z]+: {re.findall(r'[^a-z]+', text)}")  # 非小写字母


def main03_re_groups():
    """
    ============================================================
                    3. 分组和捕获
    ============================================================
    """
    print("\n" + "=" * 60)
    print("3. 分组和捕获")
    print("=" * 60)

    # 【基本分组】
    print("--- 基本分组 ---")
    text = "John Smith, Jane Doe, Bob Johnson"

    # 捕获名字
    pattern = r'(\w+)\s(\w+)'
    matches = re.findall(pattern, text)
    print(f"findall: {matches}")

    # 使用 match 对象
    match = re.search(pattern, text)
    if match:
        print(f"group(0): {match.group(0)}")  # 整个匹配
        print(f"group(1): {match.group(1)}")  # 第一组
        print(f"group(2): {match.group(2)}")  # 第二组
        print(f"groups(): {match.groups()}")

    # 【命名分组】
    print(f"\n--- 命名分组 ---")
    pattern = r'(?P<first>\w+)\s(?P<last>\w+)'
    match = re.search(pattern, text)
    if match:
        print(f"first: {match.group('first')}")
        print(f"last: {match.group('last')}")
        print(f"groupdict(): {match.groupdict()}")

    # 【非捕获分组】
    print(f"\n--- 非捕获分组 ---")
    text = "apple123orange456"
    pattern = r'(?:\w+)(\d+)'  # (?:...) 不捕获
    matches = re.findall(pattern, text)
    print(f"非捕获分组: {matches}")  # 只有数字


def main04_re_replace():
    """
    ============================================================
                    4. 替换和分割
    ============================================================
    """
    print("\n" + "=" * 60)
    print("4. 替换和分割")
    print("=" * 60)

    # 【替换 sub】
    print("--- 替换 ---")
    text = "Hello World Hello Python"

    # 简单替换
    result = re.sub(r'Hello', 'Hi', text)
    print(f"sub: {result}")

    # 限制替换次数
    result = re.sub(r'Hello', 'Hi', text, count=1)
    print(f"sub(count=1): {result}")

    # 使用函数替换
    def upper_match(match):
        return match.group().upper()

    result = re.sub(r'\w+', upper_match, "hello world")
    print(f"函数替换: {result}")

    # 使用反向引用
    text = "John Smith"
    result = re.sub(r'(\w+) (\w+)', r'\2, \1', text)
    print(f"反向引用: {result}")

    # 【分割 split】
    print(f"\n--- 分割 ---")
    text = "apple,banana;cherry orange"

    # 使用多种分隔符
    result = re.split(r'[,;\s]+', text)
    print(f"split: {result}")


def main05_re_flags():
    """
    ============================================================
                    5. 正则表达式标志
    ============================================================
    """
    print("\n" + "=" * 60)
    print("5. 正则表达式标志")
    print("=" * 60)

    # 【re.IGNORECASE】忽略大小写
    print("--- 忽略大小写 ---")
    text = "Hello HELLO hello"
    matches = re.findall(r'hello', text, re.IGNORECASE)
    print(f"re.IGNORECASE: {matches}")

    # 【re.MULTILINE】多行模式
    print(f"\n--- 多行模式 ---")
    text = "line1\nline2\nline3"
    matches = re.findall(r'^line\d', text, re.MULTILINE)
    print(f"re.MULTILINE: {matches}")

    # 【re.DOTALL】让 . 匹配换行
    print(f"\n--- DOTALL ---")
    text = "Hello\nWorld"
    match = re.search(r'Hello.World', text, re.DOTALL)
    print(f"re.DOTALL: {match.group() if match else None}")

    # 【re.VERBOSE】详细模式（可加注释）
    print(f"\n--- VERBOSE ---")
    pattern = re.compile(r'''
        \d{3}       # 区号
        [-.]?       # 可选分隔符
        \d{3}       # 前缀
        [-.]?       # 可选分隔符
        \d{4}       # 号码
    ''', re.VERBOSE)
    print(f"匹配电话: {pattern.search('123-456-7890').group()}")


def main06_collections_counter():
    """
    ============================================================
                    6. Counter 计数器
    ============================================================
    """
    print("\n" + "=" * 60)
    print("6. Counter 计数器")
    print("=" * 60)

    # 【创建 Counter】
    print("--- 创建 Counter ---")
    c1 = Counter(['a', 'b', 'c', 'a', 'b', 'a'])
    c2 = Counter("abracadabra")
    c3 = Counter({'red': 4, 'blue': 2})

    print(f"从列表: {c1}")
    print(f"从字符串: {c2}")
    print(f"从字典: {c3}")

    # 【常用方法】
    print(f"\n--- 常用方法 ---")
    c = Counter("abracadabra")
    print(f"most_common(3): {c.most_common(3)}")
    print(f"elements: {list(c.elements())}")
    print(f"total: {c.total()}")

    # 【Counter 运算】
    print(f"\n--- Counter 运算 ---")
    c1 = Counter(a=3, b=1)
    c2 = Counter(a=1, b=2)
    print(f"c1 + c2: {c1 + c2}")
    print(f"c1 - c2: {c1 - c2}")
    print(f"c1 & c2: {c1 & c2}")
    print(f"c1 | c2: {c1 | c2}")


def main07_collections_deque():
    """
    ============================================================
                    7. deque 双端队列
    ============================================================
    """
    print("\n" + "=" * 60)
    print("7. deque 双端队列")
    print("=" * 60)

    # 【创建 deque】
    print("--- 创建 deque ---")
    d = deque([1, 2, 3])
    print(f"deque: {d}")

    # 【两端操作】
    print(f"\n--- 两端操作 ---")
    d.append(4)         # 右端添加
    d.appendleft(0)     # 左端添加
    print(f"添加后: {d}")

    d.pop()             # 右端删除
    d.popleft()         # 左端删除
    print(f"删除后: {d}")

    # 【旋转】
    print(f"\n--- 旋转 ---")
    d = deque([1, 2, 3, 4, 5])
    d.rotate(2)         # 右旋转 2 步
    print(f"右旋转 2: {d}")
    d.rotate(-2)        # 左旋转 2 步
    print(f"左旋转 2: {d}")

    # 【有界队列】
    print(f"\n--- 有界队列 ---")
    d = deque(maxlen=3)
    for i in range(5):
        d.append(i)
        print(f"添加 {i}: {list(d)}")


def main08_collections_others():
    """
    ============================================================
                    8. 其他集合类型
    ============================================================
    """
    print("\n" + "=" * 60)
    print("8. 其他集合类型")
    print("=" * 60)

    # 【defaultdict】
    print("--- defaultdict ---")
    dd = defaultdict(list)
    dd['fruits'].append('apple')
    dd['fruits'].append('banana')
    dd['vegetables'].append('carrot')
    print(f"defaultdict: {dict(dd)}")

    # 【namedtuple】
    print(f"\n--- namedtuple ---")
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(3, 4)
    print(f"Point: {p}")
    print(f"x={p.x}, y={p.y}")
    print(f"_asdict: {p._asdict()}")

    # 【OrderedDict】
    print(f"\n--- OrderedDict ---")
    od = OrderedDict()
    od['first'] = 1
    od['second'] = 2
    od.move_to_end('first')
    print(f"move_to_end: {od}")

    # 【ChainMap】
    print(f"\n--- ChainMap ---")
    defaults = {'color': 'red', 'size': 'medium'}
    user_settings = {'color': 'blue'}
    settings = ChainMap(user_settings, defaults)
    print(f"color: {settings['color']}")  # blue
    print(f"size: {settings['size']}")    # medium


if __name__ == "__main__":
    main01_re_basics()
    main02_re_patterns()
    main03_re_groups()
    main04_re_replace()
    main05_re_flags()
    main06_collections_counter()
    main07_collections_deque()
    main08_collections_others()
```
