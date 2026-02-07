# implementation

::: info 文件信息
- 📄 原文件：`implementation.py`
- 🔤 语言：python
:::

高级数据结构实现
包含 Trie、线段树、树状数组、跳表、布隆过滤器等实现。

## 完整代码

```python
from typing import List, Dict, Optional, Any, Tuple
import random
import math


# ============================================================
#                    Trie（前缀树）
# ============================================================

class TrieNode:
    """Trie 节点"""

    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end: bool = False  # 是否是单词结尾
        self.count: int = 0        # 以此为前缀的单词数量


class Trie:
    """
    前缀树（字典树）

    用于高效的字符串存储和检索
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        插入单词

        时间复杂度: O(m)，m 为单词长度
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        node.is_end = True

    def search(self, word: str) -> bool:
        """
        搜索完整单词

        时间复杂度: O(m)
        """
        node = self._find_node(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        """
        检查是否存在以 prefix 为前缀的单词

        时间复杂度: O(m)
        """
        return self._find_node(prefix) is not None

    def count_prefix(self, prefix: str) -> int:
        """
        统计以 prefix 为前缀的单词数量
        """
        node = self._find_node(prefix)
        return node.count if node else 0

    def get_words_with_prefix(self, prefix: str) -> List[str]:
        """
        获取所有以 prefix 为前缀的单词
        """
        node = self._find_node(prefix)
        if not node:
            return []

        result = []
        self._collect_words(node, prefix, result)
        return result

    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """找到前缀对应的节点"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def _collect_words(self, node: TrieNode, prefix: str, result: List[str]) -> None:
        """收集所有单词"""
        if node.is_end:
            result.append(prefix)
        for char, child in node.children.items():
            self._collect_words(child, prefix + char, result)

    def delete(self, word: str) -> bool:
        """
        删除单词

        Returns:
            是否成功删除
        """
        def _delete(node: TrieNode, word: str, depth: int) -> bool:
            if depth == len(word):
                if not node.is_end:
                    return False
                node.is_end = False
                return len(node.children) == 0

            char = word[depth]
            if char not in node.children:
                return False

            should_delete = _delete(node.children[char], word, depth + 1)

            if should_delete:
                del node.children[char]
                return not node.is_end and len(node.children) == 0

            node.children[char].count -= 1
            return False

        return _delete(self.root, word, 0)


# ============================================================
#                    线段树
# ============================================================

class SegmentTree:
    """
    线段树（区间和）

    支持单点更新和区间查询
    """

    def __init__(self, nums: List[int]):
        self.n = len(nums)
        # 使用 4n 空间确保足够
        self.tree = [0] * (4 * self.n)
        if self.n > 0:
            self._build(nums, 0, 0, self.n - 1)

    def _build(self, nums: List[int], node: int, start: int, end: int) -> None:
        """构建线段树"""
        if start == end:
            self.tree[node] = nums[start]
            return

        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2

        self._build(nums, left_child, start, mid)
        self._build(nums, right_child, mid + 1, end)

        self.tree[node] = self.tree[left_child] + self.tree[right_child]

    def update(self, index: int, value: int) -> None:
        """
        单点更新

        时间复杂度: O(log n)
        """
        self._update(0, 0, self.n - 1, index, value)

    def _update(self, node: int, start: int, end: int, index: int, value: int) -> None:
        if start == end:
            self.tree[node] = value
            return

        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2

        if index <= mid:
            self._update(left_child, start, mid, index, value)
        else:
            self._update(right_child, mid + 1, end, index, value)

        self.tree[node] = self.tree[left_child] + self.tree[right_child]

    def query(self, left: int, right: int) -> int:
        """
        区间查询

        时间复杂度: O(log n)
        """
        return self._query(0, 0, self.n - 1, left, right)

    def _query(self, node: int, start: int, end: int, left: int, right: int) -> int:
        # 完全不相交
        if right < start or left > end:
            return 0

        # 完全包含
        if left <= start and end <= right:
            return self.tree[node]

        # 部分相交
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2

        left_sum = self._query(left_child, start, mid, left, right)
        right_sum = self._query(right_child, mid + 1, end, left, right)

        return left_sum + right_sum


class SegmentTreeLazy:
    """
    线段树（带懒惰传播）

    支持区间更新和区间查询
    """

    def __init__(self, nums: List[int]):
        self.n = len(nums)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)  # 懒惰标记
        if self.n > 0:
            self._build(nums, 0, 0, self.n - 1)

    def _build(self, nums: List[int], node: int, start: int, end: int) -> None:
        if start == end:
            self.tree[node] = nums[start]
            return

        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2

        self._build(nums, left_child, start, mid)
        self._build(nums, right_child, mid + 1, end)

        self.tree[node] = self.tree[left_child] + self.tree[right_child]

    def _push_down(self, node: int, start: int, end: int) -> None:
        """下推懒惰标记"""
        if self.lazy[node] != 0:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2

            # 更新子节点的值
            self.tree[left_child] += self.lazy[node] * (mid - start + 1)
            self.tree[right_child] += self.lazy[node] * (end - mid)

            # 传递懒惰标记
            self.lazy[left_child] += self.lazy[node]
            self.lazy[right_child] += self.lazy[node]

            # 清除当前懒惰标记
            self.lazy[node] = 0

    def update_range(self, left: int, right: int, value: int) -> None:
        """
        区间更新：[left, right] 范围内的元素都加上 value

        时间复杂度: O(log n)
        """
        self._update_range(0, 0, self.n - 1, left, right, value)

    def _update_range(self, node: int, start: int, end: int,
                      left: int, right: int, value: int) -> None:
        if right < start or left > end:
            return

        if left <= start and end <= right:
            self.tree[node] += value * (end - start + 1)
            self.lazy[node] += value
            return

        self._push_down(node, start, end)

        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2

        self._update_range(left_child, start, mid, left, right, value)
        self._update_range(right_child, mid + 1, end, left, right, value)

        self.tree[node] = self.tree[left_child] + self.tree[right_child]

    def query(self, left: int, right: int) -> int:
        """区间查询"""
        return self._query(0, 0, self.n - 1, left, right)

    def _query(self, node: int, start: int, end: int, left: int, right: int) -> int:
        if right < start or left > end:
            return 0

        if left <= start and end <= right:
            return self.tree[node]

        self._push_down(node, start, end)

        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2

        return (self._query(left_child, start, mid, left, right) +
                self._query(right_child, mid + 1, end, left, right))


# ============================================================
#                    树状数组
# ============================================================

class BinaryIndexedTree:
    """
    树状数组（Fenwick Tree）

    支持单点更新和前缀查询
    """

    def __init__(self, n: int):
        """
        Args:
            n: 数组大小
        """
        self.n = n
        self.tree = [0] * (n + 1)  # 1-indexed

    @classmethod
    def from_array(cls, nums: List[int]) -> 'BinaryIndexedTree':
        """从数组构建树状数组"""
        bit = cls(len(nums))
        for i, num in enumerate(nums):
            bit.update(i, num)
        return bit

    def _lowbit(self, x: int) -> int:
        """获取最低位的 1"""
        return x & (-x)

    def update(self, index: int, delta: int) -> None:
        """
        单点更新：a[index] += delta

        时间复杂度: O(log n)
        """
        index += 1  # 转为 1-indexed
        while index <= self.n:
            self.tree[index] += delta
            index += self._lowbit(index)

    def prefix_sum(self, index: int) -> int:
        """
        前缀和查询：sum(a[0:index+1])

        时间复杂度: O(log n)
        """
        index += 1  # 转为 1-indexed
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= self._lowbit(index)
        return result

    def range_sum(self, left: int, right: int) -> int:
        """
        区间和查询：sum(a[left:right+1])

        时间复杂度: O(log n)
        """
        if left == 0:
            return self.prefix_sum(right)
        return self.prefix_sum(right) - self.prefix_sum(left - 1)


class BinaryIndexedTree2D:
    """二维树状数组"""

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]

    def _lowbit(self, x: int) -> int:
        return x & (-x)

    def update(self, row: int, col: int, delta: int) -> None:
        """单点更新"""
        row += 1
        col += 1
        i = row
        while i <= self.rows:
            j = col
            while j <= self.cols:
                self.tree[i][j] += delta
                j += self._lowbit(j)
            i += self._lowbit(i)

    def prefix_sum(self, row: int, col: int) -> int:
        """前缀和：(0,0) 到 (row,col) 的矩形和"""
        row += 1
        col += 1
        result = 0
        i = row
        while i > 0:
            j = col
            while j > 0:
                result += self.tree[i][j]
                j -= self._lowbit(j)
            i -= self._lowbit(i)
        return result

    def range_sum(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """区间和：(r1,c1) 到 (r2,c2) 的矩形和"""
        result = self.prefix_sum(r2, c2)
        if r1 > 0:
            result -= self.prefix_sum(r1 - 1, c2)
        if c1 > 0:
            result -= self.prefix_sum(r2, c1 - 1)
        if r1 > 0 and c1 > 0:
            result += self.prefix_sum(r1 - 1, c1 - 1)
        return result


# ============================================================
#                    跳表
# ============================================================

class SkipListNode:
    """跳表节点"""

    def __init__(self, value: float, level: int):
        self.value = value
        # forward[i] 是第 i 层的下一个节点
        self.forward: List[Optional['SkipListNode']] = [None] * level


class SkipList:
    """
    跳表

    支持快速的有序集合操作
    """

    MAX_LEVEL = 16  # 最大层数
    P = 0.5         # 上升概率

    def __init__(self):
        self.head = SkipListNode(float('-inf'), self.MAX_LEVEL)
        self.level = 1  # 当前最大层数
        self.size = 0

    def _random_level(self) -> int:
        """随机生成层数"""
        level = 1
        while random.random() < self.P and level < self.MAX_LEVEL:
            level += 1
        return level

    def search(self, value: float) -> bool:
        """
        搜索元素

        时间复杂度: O(log n) 期望
        """
        current = self.head

        # 从最高层开始搜索
        for i in range(self.level - 1, -1, -1):
            while current.forward[i] and current.forward[i].value < value:
                current = current.forward[i]

        current = current.forward[0]
        return current is not None and current.value == value

    def insert(self, value: float) -> None:
        """
        插入元素

        时间复杂度: O(log n) 期望
        """
        # 记录每层的前驱节点
        update = [None] * self.MAX_LEVEL
        current = self.head

        for i in range(self.level - 1, -1, -1):
            while current.forward[i] and current.forward[i].value < value:
                current = current.forward[i]
            update[i] = current

        # 生成新节点的层数
        new_level = self._random_level()

        # 如果新层数超过当前最大层数
        if new_level > self.level:
            for i in range(self.level, new_level):
                update[i] = self.head
            self.level = new_level

        # 创建新节点
        new_node = SkipListNode(value, new_level)

        # 在每层插入新节点
        for i in range(new_level):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

        self.size += 1

    def delete(self, value: float) -> bool:
        """
        删除元素

        时间复杂度: O(log n) 期望

        Returns:
            是否成功删除
        """
        update = [None] * self.MAX_LEVEL
        current = self.head

        for i in range(self.level - 1, -1, -1):
            while current.forward[i] and current.forward[i].value < value:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        if current is None or current.value != value:
            return False

        # 在每层删除节点
        for i in range(self.level):
            if update[i].forward[i] != current:
                break
            update[i].forward[i] = current.forward[i]

        # 更新最大层数
        while self.level > 1 and self.head.forward[self.level - 1] is None:
            self.level -= 1

        self.size -= 1
        return True

    def range_query(self, low: float, high: float) -> List[float]:
        """
        范围查询：返回 [low, high] 范围内的所有元素
        """
        result = []
        current = self.head

        # 找到第一个 >= low 的节点
        for i in range(self.level - 1, -1, -1):
            while current.forward[i] and current.forward[i].value < low:
                current = current.forward[i]

        current = current.forward[0]

        # 收集范围内的所有元素
        while current and current.value <= high:
            result.append(current.value)
            current = current.forward[0]

        return result

    def to_list(self) -> List[float]:
        """转换为有序列表"""
        result = []
        current = self.head.forward[0]
        while current:
            result.append(current.value)
            current = current.forward[0]
        return result

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"SkipList({self.to_list()})"


# ============================================================
#                    布隆过滤器
# ============================================================

class BloomFilter:
    """
    布隆过滤器

    用于快速判断元素是否可能存在
    """

    def __init__(self, expected_elements: int, false_positive_rate: float = 0.01):
        """
        Args:
            expected_elements: 预期元素数量
            false_positive_rate: 期望的误判率
        """
        # 计算最优位数组大小
        self.size = self._optimal_size(expected_elements, false_positive_rate)
        # 计算最优哈希函数数量
        self.hash_count = self._optimal_hash_count(self.size, expected_elements)
        # 位数组
        self.bit_array = [False] * self.size
        self.count = 0

    def _optimal_size(self, n: int, p: float) -> int:
        """计算最优位数组大小"""
        m = -n * math.log(p) / (math.log(2) ** 2)
        return int(m) + 1

    def _optimal_hash_count(self, m: int, n: int) -> int:
        """计算最优哈希函数数量"""
        k = (m / n) * math.log(2)
        return max(1, int(k))

    def _hash(self, item: str, seed: int) -> int:
        """
        生成哈希值

        使用双重哈希模拟多个哈希函数
        """
        h1 = hash(item)
        h2 = hash(item + str(seed))
        return (h1 + seed * h2) % self.size

    def add(self, item: str) -> None:
        """
        添加元素

        时间复杂度: O(k)，k 为哈希函数数量
        """
        for i in range(self.hash_count):
            index = self._hash(item, i)
            self.bit_array[index] = True
        self.count += 1

    def contains(self, item: str) -> bool:
        """
        检查元素是否可能存在

        返回 True：元素可能存在（有误判可能）
        返回 False：元素一定不存在

        时间复杂度: O(k)
        """
        for i in range(self.hash_count):
            index = self._hash(item, i)
            if not self.bit_array[index]:
                return False
        return True

    def estimated_false_positive_rate(self) -> float:
        """估算当前误判率"""
        # 使用公式: (1 - e^(-kn/m))^k
        m = self.size
        n = self.count
        k = self.hash_count

        if n == 0:
            return 0.0

        return (1 - math.exp(-k * n / m)) ** k

    def __len__(self) -> int:
        return self.count


class CountingBloomFilter:
    """
    计数布隆过滤器

    支持删除操作
    """

    def __init__(self, expected_elements: int, false_positive_rate: float = 0.01):
        self.size = int(-expected_elements * math.log(false_positive_rate) /
                       (math.log(2) ** 2)) + 1
        self.hash_count = max(1, int((self.size / expected_elements) * math.log(2)))
        # 使用计数器数组替代位数组
        self.counters = [0] * self.size
        self.count = 0

    def _hash(self, item: str, seed: int) -> int:
        h1 = hash(item)
        h2 = hash(item + str(seed))
        return (h1 + seed * h2) % self.size

    def add(self, item: str) -> None:
        """添加元素"""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            self.counters[index] += 1
        self.count += 1

    def remove(self, item: str) -> bool:
        """
        删除元素

        注意：只有在确定元素存在时才应该删除
        """
        if not self.contains(item):
            return False

        for i in range(self.hash_count):
            index = self._hash(item, i)
            if self.counters[index] > 0:
                self.counters[index] -= 1
        self.count -= 1
        return True

    def contains(self, item: str) -> bool:
        """检查元素是否可能存在"""
        for i in range(self.hash_count):
            index = self._hash(item, i)
            if self.counters[index] == 0:
                return False
        return True


# ============================================================
#                    测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Trie（前缀树）")
    print("=" * 60)

    trie = Trie()
    words = ["apple", "app", "application", "apply", "banana", "band"]

    for word in words:
        trie.insert(word)

    print(f"插入单词: {words}")
    print(f"搜索 'app': {trie.search('app')}")
    print(f"搜索 'ap': {trie.search('ap')}")
    print(f"前缀 'app' 存在: {trie.starts_with('app')}")
    print(f"前缀 'app' 的单词数: {trie.count_prefix('app')}")
    print(f"前缀 'app' 的所有单词: {trie.get_words_with_prefix('app')}")

    trie.delete('app')
    print(f"删除 'app' 后搜索: {trie.search('app')}")
    print(f"前缀 'app' 的所有单词: {trie.get_words_with_prefix('app')}")

    print("\n" + "=" * 60)
    print("线段树")
    print("=" * 60)

    nums = [1, 3, 5, 7, 9, 11]
    st = SegmentTree(nums)

    print(f"数组: {nums}")
    print(f"区间 [1,4] 的和: {st.query(1, 4)}")

    st.update(2, 10)  # 将 index=2 的值改为 10
    print(f"更新 index=2 为 10 后:")
    print(f"区间 [1,4] 的和: {st.query(1, 4)}")

    print("\n带懒惰传播的线段树:")
    nums2 = [1, 2, 3, 4, 5]
    st_lazy = SegmentTreeLazy(nums2)

    print(f"数组: {nums2}")
    print(f"区间 [0,4] 的和: {st_lazy.query(0, 4)}")

    st_lazy.update_range(1, 3, 10)  # [1,3] 范围内的元素都加 10
    print(f"区间 [1,3] 加 10 后:")
    print(f"区间 [0,4] 的和: {st_lazy.query(0, 4)}")

    print("\n" + "=" * 60)
    print("树状数组")
    print("=" * 60)

    nums3 = [1, 2, 3, 4, 5, 6, 7, 8]
    bit = BinaryIndexedTree.from_array(nums3)

    print(f"数组: {nums3}")
    print(f"前缀和 [0,4]: {bit.prefix_sum(4)}")
    print(f"区间和 [2,5]: {bit.range_sum(2, 5)}")

    bit.update(3, 10)  # a[3] += 10
    print(f"a[3] += 10 后:")
    print(f"区间和 [2,5]: {bit.range_sum(2, 5)}")

    print("\n二维树状数组:")
    bit2d = BinaryIndexedTree2D(3, 3)
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    for i in range(3):
        for j in range(3):
            bit2d.update(i, j, matrix[i][j])

    print(f"矩阵: {matrix}")
    print(f"子矩阵 (0,0)-(1,1) 的和: {bit2d.range_sum(0, 0, 1, 1)}")
    print(f"子矩阵 (1,1)-(2,2) 的和: {bit2d.range_sum(1, 1, 2, 2)}")

    print("\n" + "=" * 60)
    print("跳表")
    print("=" * 60)

    skip_list = SkipList()
    values = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]

    for v in values:
        skip_list.insert(v)

    print(f"插入: {values}")
    print(f"跳表: {skip_list}")
    print(f"搜索 5: {skip_list.search(5)}")
    print(f"搜索 7: {skip_list.search(7)}")
    print(f"范围查询 [2,5]: {skip_list.range_query(2, 5)}")

    skip_list.delete(4)
    print(f"删除 4 后: {skip_list}")

    print("\n" + "=" * 60)
    print("布隆过滤器")
    print("=" * 60)

    bloom = BloomFilter(expected_elements=1000, false_positive_rate=0.01)

    # 添加一些单词
    words_to_add = ["apple", "banana", "cherry", "date", "elderberry"]
    for word in words_to_add:
        bloom.add(word)

    print(f"位数组大小: {bloom.size}")
    print(f"哈希函数数量: {bloom.hash_count}")
    print(f"添加的单词: {words_to_add}")

    # 测试查询
    test_words = ["apple", "banana", "fig", "grape", "cherry"]
    for word in test_words:
        result = bloom.contains(word)
        actual = word in words_to_add
        status = "✓" if result == actual else "✗ (误判)"
        print(f"查询 '{word}': {result} {status}")

    print(f"当前估算误判率: {bloom.estimated_false_positive_rate():.4%}")

    print("\n计数布隆过滤器（支持删除）:")
    counting_bloom = CountingBloomFilter(expected_elements=1000)

    counting_bloom.add("hello")
    counting_bloom.add("world")
    print(f"添加 'hello', 'world'")
    print(f"包含 'hello': {counting_bloom.contains('hello')}")

    counting_bloom.remove("hello")
    print(f"删除 'hello' 后:")
    print(f"包含 'hello': {counting_bloom.contains('hello')}")
    print(f"包含 'world': {counting_bloom.contains('world')}")
```
