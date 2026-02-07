# generators.py

::: info 文件信息
- 📄 原文件：`04_generators.py`
- 🔤 语言：python
:::

Python 生成器与迭代器
本文件介绍 Python 中的迭代器（Iterator）和生成器（Generator）。

生成器是一种特殊的迭代器，通过 yield 语句实现惰性求值，
非常适合处理大数据集或无限序列。

## 完整代码

```python
def main01_iterator_basics():
    """
    ============================================================
                    1. 迭代器基础
    ============================================================
    迭代器协议：
    - __iter__(): 返回迭代器对象本身
    - __next__(): 返回下一个值，没有更多值时抛出 StopIteration
    """
    print("=" * 60)
    print("1. 迭代器基础")
    print("=" * 60)

    # 【可迭代对象 vs 迭代器】
    my_list = [1, 2, 3]
    print(f"列表是可迭代对象: {hasattr(my_list, '__iter__')}")

    # 获取迭代器
    iterator = iter(my_list)
    print(f"迭代器: {iterator}")
    print(f"迭代器有 __next__: {hasattr(iterator, '__next__')}")

    # 手动迭代
    print(f"\n手动迭代:")
    print(f"  next(iterator) = {next(iterator)}")
    print(f"  next(iterator) = {next(iterator)}")
    print(f"  next(iterator) = {next(iterator)}")
    # print(next(iterator))  # StopIteration

    # 【for 循环的本质】
    print("\nfor 循环的本质:")
    print("  1. 调用 iter() 获取迭代器")
    print("  2. 反复调用 next() 获取值")
    print("  3. 捕获 StopIteration 结束循环")

    # 【自定义迭代器】
    print("\n--- 自定义迭代器 ---")

    class CountDown:
        """倒计时迭代器"""
        def __init__(self, start):
            self.current = start

        def __iter__(self):
            return self

        def __next__(self):
            if self.current <= 0:
                raise StopIteration
            self.current -= 1
            return self.current + 1

    countdown = CountDown(5)
    print(f"倒计时: {list(countdown)}")

    # 【重要】迭代器只能遍历一次
    countdown = CountDown(3)
    print(f"\n第一次遍历: {list(countdown)}")
    print(f"第二次遍历: {list(countdown)}")  # 空！


def main02_generator_basics():
    """
    ============================================================
                    2. 生成器基础
    ============================================================
    生成器是使用 yield 语句的函数
    """
    print("\n" + "=" * 60)
    print("2. 生成器基础")
    print("=" * 60)

    # 【基本生成器】
    def count_up(n):
        """从 1 数到 n 的生成器"""
        i = 1
        while i <= n:
            yield i  # yield 暂停函数并返回值
            i += 1

    print("生成器函数:")
    gen = count_up(5)
    print(f"  生成器对象: {gen}")
    print(f"  逐个获取: {next(gen)}, {next(gen)}, {next(gen)}")
    print(f"  转为列表: {list(count_up(5))}")

    # 【生成器的执行流程】
    print("\n--- 生成器执行流程 ---")

    def simple_gen():
        print("  开始")
        yield 1
        print("  第一个 yield 之后")
        yield 2
        print("  第二个 yield 之后")
        yield 3
        print("  结束")

    gen = simple_gen()
    print(f"创建生成器（函数体未执行）")
    print(f"第一次 next: {next(gen)}")
    print(f"第二次 next: {next(gen)}")
    print(f"第三次 next: {next(gen)}")
    # next(gen)  # StopIteration

    # 【生成器表达式】
    print("\n--- 生成器表达式 ---")

    # 列表推导式（立即求值，占用内存）
    list_comp = [x**2 for x in range(5)]
    print(f"列表推导式: {list_comp}")

    # 生成器表达式（惰性求值，节省内存）
    gen_exp = (x**2 for x in range(5))
    print(f"生成器表达式: {gen_exp}")
    print(f"转为列表: {list(gen_exp)}")

    # 【内存对比】
    import sys
    list_1m = [x for x in range(1000000)]
    gen_1m = (x for x in range(1000000))
    print(f"\n内存对比:")
    print(f"  100万元素列表: {sys.getsizeof(list_1m):,} bytes")
    print(f"  100万元素生成器: {sys.getsizeof(gen_1m)} bytes")


def main03_yield_from():
    """
    ============================================================
                    3. yield from 语法
    ============================================================
    yield from 用于委托给子生成器
    """
    print("\n" + "=" * 60)
    print("3. yield from 语法")
    print("=" * 60)

    # 【没有 yield from】
    def chain_old(*iterables):
        for iterable in iterables:
            for item in iterable:
                yield item

    # 【使用 yield from】
    def chain_new(*iterables):
        for iterable in iterables:
            yield from iterable  # 委托给子迭代器

    print("yield from 链接多个迭代器:")
    result = list(chain_new([1, 2], [3, 4], [5, 6]))
    print(f"  chain([1,2], [3,4], [5,6]) = {result}")

    # 【递归生成器】
    def flatten(nested):
        """展平嵌套列表"""
        for item in nested:
            if isinstance(item, list):
                yield from flatten(item)  # 递归委托
            else:
                yield item

    nested = [1, [2, 3, [4, 5]], 6, [7, [8, 9]]]
    print(f"\n展平嵌套列表:")
    print(f"  输入: {nested}")
    print(f"  输出: {list(flatten(nested))}")

    # 【树遍历】
    class Node:
        def __init__(self, value, children=None):
            self.value = value
            self.children = children or []

    def traverse(node):
        """前序遍历树"""
        yield node.value
        for child in node.children:
            yield from traverse(child)

    tree = Node(1, [
        Node(2, [Node(4), Node(5)]),
        Node(3, [Node(6)])
    ])
    print(f"\n树遍历: {list(traverse(tree))}")


def main04_generator_methods():
    """
    ============================================================
                4. 生成器方法：send, throw, close
    ============================================================
    """
    print("\n" + "=" * 60)
    print("4. 生成器方法")
    print("=" * 60)

    # 【send() 方法】向生成器发送值
    print("--- send() 方法 ---")

    def accumulator():
        """累加器，接收发送的值"""
        total = 0
        while True:
            value = yield total  # yield 可以接收 send 的值
            if value is not None:
                total += value

    acc = accumulator()
    print(f"初始化: {next(acc)}")  # 必须先 next() 启动
    print(f"send(10): {acc.send(10)}")
    print(f"send(20): {acc.send(20)}")
    print(f"send(30): {acc.send(30)}")

    # 【throw() 方法】向生成器抛出异常
    print("\n--- throw() 方法 ---")

    def gen_with_exception():
        try:
            while True:
                try:
                    value = yield
                    print(f"  收到: {value}")
                except ValueError as e:
                    print(f"  捕获 ValueError: {e}")
        except GeneratorExit:
            print("  生成器关闭")

    gen = gen_with_exception()
    next(gen)  # 启动
    gen.send("Hello")
    gen.throw(ValueError, "自定义错误")
    gen.send("Still working")

    # 【close() 方法】关闭生成器
    print("\n--- close() 方法 ---")
    gen.close()  # 触发 GeneratorExit


def main05_practical_generators():
    """
    ============================================================
                5. 生成器实际应用
    ============================================================
    """
    print("\n" + "=" * 60)
    print("5. 生成器实际应用")
    print("=" * 60)

    # 【应用1：无限序列】
    print("--- 无限序列 ---")

    def infinite_counter(start=0):
        """无限计数器"""
        n = start
        while True:
            yield n
            n += 1

    counter = infinite_counter()
    print(f"前10个数: {[next(counter) for _ in range(10)]}")

    # 【应用2：斐波那契数列】
    print("\n--- 斐波那契数列 ---")

    def fibonacci():
        """无限斐波那契数列"""
        a, b = 0, 1
        while True:
            yield a
            a, b = b, a + b

    fib = fibonacci()
    print(f"前10个斐波那契数: {[next(fib) for _ in range(10)]}")

    # 【应用3：文件读取】
    print("\n--- 文件逐行读取（模拟）---")

    def read_large_file(lines):
        """逐行读取大文件（模拟）"""
        for line in lines:
            yield line.strip()

    # 模拟大文件
    fake_file = ["line 1\n", "line 2\n", "line 3\n"]
    for line in read_large_file(fake_file):
        print(f"  处理: {line}")

    # 【应用4：管道/流水线】
    print("\n--- 数据管道 ---")

    def numbers(n):
        """生成数字"""
        for i in range(n):
            yield i

    def square(nums):
        """平方"""
        for n in nums:
            yield n ** 2

    def filter_even(nums):
        """过滤偶数"""
        for n in nums:
            if n % 2 == 0:
                yield n

    # 组合管道
    pipeline = filter_even(square(numbers(10)))
    print(f"管道结果: {list(pipeline)}")

    # 【应用5：滑动窗口】
    print("\n--- 滑动窗口 ---")

    def sliding_window(iterable, size):
        """生成滑动窗口"""
        from collections import deque
        it = iter(iterable)
        window = deque(maxlen=size)

        # 填充初始窗口
        for _ in range(size):
            window.append(next(it))
        yield tuple(window)

        # 滑动
        for item in it:
            window.append(item)
            yield tuple(window)

    data = [1, 2, 3, 4, 5, 6, 7]
    print(f"窗口大小 3: {list(sliding_window(data, 3))}")

    # 【应用6：分块处理】
    print("\n--- 分块处理 ---")

    def chunked(iterable, size):
        """将可迭代对象分成固定大小的块"""
        chunk = []
        for item in iterable:
            chunk.append(item)
            if len(chunk) == size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    data = list(range(10))
    print(f"分块（大小3）: {list(chunked(data, 3))}")


def main06_itertools():
    """
    ============================================================
                6. itertools 模块
    ============================================================
    itertools 提供了高效的迭代器工具
    """
    print("\n" + "=" * 60)
    print("6. itertools 模块")
    print("=" * 60)

    import itertools

    # 【无限迭代器】
    print("--- 无限迭代器 ---")

    # count: 无限计数
    counter = itertools.count(10, 2)  # 从10开始，步长2
    print(f"count(10, 2): {[next(counter) for _ in range(5)]}")

    # cycle: 无限循环
    cycler = itertools.cycle(['A', 'B', 'C'])
    print(f"cycle(['A','B','C']): {[next(cycler) for _ in range(7)]}")

    # repeat: 重复
    print(f"repeat('X', 3): {list(itertools.repeat('X', 3))}")

    # 【终止迭代器】
    print("\n--- 终止迭代器 ---")

    # chain: 链接多个迭代器
    print(f"chain([1,2], [3,4]): {list(itertools.chain([1, 2], [3, 4]))}")

    # compress: 根据选择器过滤
    data = ['A', 'B', 'C', 'D']
    selectors = [1, 0, 1, 0]
    print(f"compress: {list(itertools.compress(data, selectors))}")

    # dropwhile: 丢弃满足条件的前导元素
    print(f"dropwhile(<5): {list(itertools.dropwhile(lambda x: x < 5, [1,3,5,2,1]))}")

    # takewhile: 获取满足条件的前导元素
    print(f"takewhile(<5): {list(itertools.takewhile(lambda x: x < 5, [1,3,5,2,1]))}")

    # islice: 切片
    print(f"islice(range(10), 2, 8, 2): {list(itertools.islice(range(10), 2, 8, 2))}")

    # filterfalse: 过滤掉满足条件的
    print(f"filterfalse(is_even): {list(itertools.filterfalse(lambda x: x % 2 == 0, range(10)))}")

    # 【组合迭代器】
    print("\n--- 组合迭代器 ---")

    # product: 笛卡尔积
    print(f"product([1,2], ['a','b']): {list(itertools.product([1, 2], ['a', 'b']))}")

    # permutations: 排列
    print(f"permutations('AB', 2): {list(itertools.permutations('AB', 2))}")

    # combinations: 组合
    print(f"combinations('ABC', 2): {list(itertools.combinations('ABC', 2))}")

    # combinations_with_replacement: 可重复组合
    print(f"combinations_with_replacement('AB', 2): {list(itertools.combinations_with_replacement('AB', 2))}")

    # 【分组】
    print("\n--- 分组 ---")

    # groupby: 按键分组（需要先排序！）
    data = [('A', 1), ('A', 2), ('B', 3), ('B', 4), ('A', 5)]
    data.sort(key=lambda x: x[0])  # 必须先排序
    for key, group in itertools.groupby(data, key=lambda x: x[0]):
        print(f"  {key}: {list(group)}")

    # 【累积】
    print("\n--- 累积 ---")

    # accumulate: 累积
    print(f"accumulate([1,2,3,4]): {list(itertools.accumulate([1, 2, 3, 4]))}")
    print(f"accumulate(乘法): {list(itertools.accumulate([1, 2, 3, 4], lambda x, y: x * y))}")

    # 【pairwise】Python 3.10+
    print(f"\npairwise([1,2,3,4]): {list(itertools.pairwise([1, 2, 3, 4]))}")

    # 【batched】Python 3.12+
    # print(f"batched([1,2,3,4,5], 2): {list(itertools.batched([1,2,3,4,5], 2))}")


def main07_generator_context():
    """
    ============================================================
                7. 生成器作为上下文管理器
    ============================================================
    """
    print("\n" + "=" * 60)
    print("7. 生成器作为上下文管理器")
    print("=" * 60)

    from contextlib import contextmanager

    @contextmanager
    def timer(name):
        """计时上下文管理器"""
        import time
        print(f"  开始 {name}")
        start = time.perf_counter()
        try:
            yield  # 这里会执行 with 块中的代码
        finally:
            end = time.perf_counter()
            print(f"  结束 {name}，耗时: {(end - start)*1000:.2f}ms")

    print("使用生成器上下文管理器:")
    with timer("测试"):
        # 模拟一些工作
        sum(range(100000))

    # 【带返回值的上下文管理器】
    @contextmanager
    def open_file(filename, mode='r'):
        """模拟文件打开（带返回值）"""
        print(f"  打开文件: {filename}")
        f = {"name": filename, "mode": mode, "content": "文件内容"}
        try:
            yield f
        finally:
            print(f"  关闭文件: {filename}")

    print("\n带返回值的上下文管理器:")
    with open_file("test.txt") as f:
        print(f"  读取: {f['content']}")

    # 【异常处理】
    @contextmanager
    def error_handler(name):
        """错误处理上下文"""
        try:
            yield
        except Exception as e:
            print(f"  {name} 捕获异常: {e}")

    print("\n异常处理上下文:")
    with error_handler("测试"):
        raise ValueError("测试异常")
    print("  继续执行")


if __name__ == "__main__":
    main01_iterator_basics()
    main02_generator_basics()
    main03_yield_from()
    main04_generator_methods()
    main05_practical_generators()
    main06_itertools()
    main07_generator_context()
```
