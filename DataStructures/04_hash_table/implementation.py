"""
============================================================
                    哈希表实现
============================================================
包含链地址法和开放地址法的哈希表实现，以及 LRU 缓存。
============================================================
"""

from typing import TypeVar, Generic, Optional, List, Iterator, Tuple
from collections import OrderedDict

K = TypeVar('K')
V = TypeVar('V')


# ============================================================
#                    链地址法哈希表
# ============================================================

class HashTableChaining(Generic[K, V]):
    """
    哈希表实现（链地址法）

    使用链表解决冲突
    """

    class Node:
        """链表节点"""
        def __init__(self, key: K, value: V, next=None):
            self.key = key
            self.value = value
            self.next = next

    def __init__(self, capacity: int = 16, load_factor: float = 0.75):
        self._capacity = capacity
        self._load_factor = load_factor
        self._size = 0
        self._buckets: List[Optional[self.Node]] = [None] * capacity

    def _hash(self, key: K) -> int:
        """计算哈希值"""
        return hash(key) % self._capacity

    def put(self, key: K, value: V) -> None:
        """
        插入或更新键值对

        时间复杂度: 平均 O(1)
        """
        # 检查是否需要扩容
        if self._size >= self._capacity * self._load_factor:
            self._resize(self._capacity * 2)

        index = self._hash(key)
        node = self._buckets[index]

        # 查找是否已存在
        while node:
            if node.key == key:
                node.value = value  # 更新
                return
            node = node.next

        # 头插法插入新节点
        new_node = self.Node(key, value, self._buckets[index])
        self._buckets[index] = new_node
        self._size += 1

    def get(self, key: K) -> Optional[V]:
        """
        获取值

        时间复杂度: 平均 O(1)
        """
        index = self._hash(key)
        node = self._buckets[index]

        while node:
            if node.key == key:
                return node.value
            node = node.next

        return None

    def remove(self, key: K) -> Optional[V]:
        """
        删除键值对

        时间复杂度: 平均 O(1)
        """
        index = self._hash(key)
        node = self._buckets[index]
        prev = None

        while node:
            if node.key == key:
                if prev:
                    prev.next = node.next
                else:
                    self._buckets[index] = node.next
                self._size -= 1
                return node.value
            prev = node
            node = node.next

        return None

    def contains(self, key: K) -> bool:
        """检查键是否存在"""
        return self.get(key) is not None

    def _resize(self, new_capacity: int) -> None:
        """扩容"""
        old_buckets = self._buckets
        self._capacity = new_capacity
        self._buckets = [None] * new_capacity
        self._size = 0

        # 重新哈希所有元素
        for bucket in old_buckets:
            node = bucket
            while node:
                self.put(node.key, node.value)
                node = node.next

    def __setitem__(self, key: K, value: V) -> None:
        self.put(key, value)

    def __getitem__(self, key: K) -> V:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __delitem__(self, key: K) -> None:
        if self.remove(key) is None:
            raise KeyError(key)

    def __contains__(self, key: K) -> bool:
        return self.contains(key)

    def __len__(self) -> int:
        return self._size

    def keys(self) -> List[K]:
        """返回所有键"""
        result = []
        for bucket in self._buckets:
            node = bucket
            while node:
                result.append(node.key)
                node = node.next
        return result

    def values(self) -> List[V]:
        """返回所有值"""
        result = []
        for bucket in self._buckets:
            node = bucket
            while node:
                result.append(node.value)
                node = node.next
        return result

    def items(self) -> List[Tuple[K, V]]:
        """返回所有键值对"""
        result = []
        for bucket in self._buckets:
            node = bucket
            while node:
                result.append((node.key, node.value))
                node = node.next
        return result

    def __repr__(self) -> str:
        items = [f"{k}: {v}" for k, v in self.items()]
        return "{" + ", ".join(items) + "}"


# ============================================================
#                    开放地址法哈希表
# ============================================================

class HashTableOpenAddressing(Generic[K, V]):
    """
    哈希表实现（开放地址法 - 线性探测）
    """

    # 标记已删除的槽位
    _DELETED = object()

    def __init__(self, capacity: int = 16, load_factor: float = 0.5):
        self._capacity = capacity
        self._load_factor = load_factor
        self._size = 0
        self._keys: List = [None] * capacity
        self._values: List = [None] * capacity

    def _hash(self, key: K) -> int:
        """计算哈希值"""
        return hash(key) % self._capacity

    def _find_slot(self, key: K) -> Tuple[int, bool]:
        """
        查找槽位

        返回: (索引, 是否找到)
        """
        index = self._hash(key)
        first_deleted = -1

        for _ in range(self._capacity):
            if self._keys[index] is None:
                # 空槽位
                if first_deleted != -1:
                    return first_deleted, False
                return index, False

            if self._keys[index] is self._DELETED:
                # 记录第一个删除位置
                if first_deleted == -1:
                    first_deleted = index

            elif self._keys[index] == key:
                # 找到了
                return index, True

            # 线性探测
            index = (index + 1) % self._capacity

        # 表满了
        if first_deleted != -1:
            return first_deleted, False
        raise RuntimeError("哈希表已满")

    def put(self, key: K, value: V) -> None:
        """插入或更新"""
        if self._size >= self._capacity * self._load_factor:
            self._resize(self._capacity * 2)

        index, found = self._find_slot(key)

        if not found:
            self._size += 1

        self._keys[index] = key
        self._values[index] = value

    def get(self, key: K) -> Optional[V]:
        """获取值"""
        index, found = self._find_slot(key)
        if found:
            return self._values[index]
        return None

    def remove(self, key: K) -> Optional[V]:
        """删除"""
        index, found = self._find_slot(key)
        if found:
            value = self._values[index]
            self._keys[index] = self._DELETED
            self._values[index] = None
            self._size -= 1
            return value
        return None

    def _resize(self, new_capacity: int) -> None:
        """扩容"""
        old_keys = self._keys
        old_values = self._values

        self._capacity = new_capacity
        self._keys = [None] * new_capacity
        self._values = [None] * new_capacity
        self._size = 0

        for i, key in enumerate(old_keys):
            if key is not None and key is not self._DELETED:
                self.put(key, old_values[i])

    def __setitem__(self, key: K, value: V) -> None:
        self.put(key, value)

    def __getitem__(self, key: K) -> V:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __contains__(self, key: K) -> bool:
        _, found = self._find_slot(key)
        return found

    def __len__(self) -> int:
        return self._size


# ============================================================
#                    LRU 缓存
# ============================================================

class LRUCache:
    """
    LRU 缓存实现

    使用双向链表 + 哈希表
    - 双向链表维护访问顺序
    - 哈希表实现 O(1) 查找

    时间复杂度: get 和 put 都是 O(1)
    """

    class Node:
        def __init__(self, key: int = 0, value: int = 0):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> Node

        # 使用哨兵节点
        self.head = self.Node()
        self.tail = self.Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_head(self, node: Node) -> None:
        """添加到头部（最近使用）"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: Node) -> None:
        """从链表中删除节点"""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: Node) -> None:
        """移动到头部"""
        self._remove_node(node)
        self._add_to_head(node)

    def _remove_tail(self) -> Node:
        """删除尾部节点（最久未使用）"""
        node = self.tail.prev
        self._remove_node(node)
        return node

    def get(self, key: int) -> int:
        """获取值"""
        if key not in self.cache:
            return -1

        node = self.cache[key]
        self._move_to_head(node)  # 移到最近使用
        return node.value

    def put(self, key: int, value: int) -> None:
        """插入或更新"""
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._move_to_head(node)
        else:
            node = self.Node(key, value)
            self.cache[key] = node
            self._add_to_head(node)

            if len(self.cache) > self.capacity:
                # 删除最久未使用
                removed = self._remove_tail()
                del self.cache[removed.key]


class LRUCacheSimple:
    """
    LRU 缓存简化实现（使用 OrderedDict）
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)  # 移到末尾（最近使用）
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # 删除最旧的


# ============================================================
#                    常见哈希表算法
# ============================================================

def two_sum(nums: List[int], target: int) -> List[int]:
    """
    两数之和

    时间复杂度: O(n)
    空间复杂度: O(n)
    """
    seen = {}  # 值 -> 索引

    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i

    return []


def group_anagrams(strs: List[str]) -> List[List[str]]:
    """
    字母异位词分组

    输入: ["eat","tea","tan","ate","nat","bat"]
    输出: [["eat","tea","ate"],["tan","nat"],["bat"]]

    时间复杂度: O(n * k log k)，k 是字符串平均长度
    空间复杂度: O(n * k)
    """
    groups = {}

    for s in strs:
        # 排序后的字符串作为键
        key = ''.join(sorted(s))
        if key not in groups:
            groups[key] = []
        groups[key].append(s)

    return list(groups.values())


def longest_consecutive(nums: List[int]) -> int:
    """
    最长连续序列

    输入: [100, 4, 200, 1, 3, 2]
    输出: 4 (序列 [1, 2, 3, 4])

    时间复杂度: O(n)
    空间复杂度: O(n)
    """
    if not nums:
        return 0

    num_set = set(nums)
    max_length = 0

    for num in num_set:
        # 只从序列起点开始计算
        if num - 1 not in num_set:
            current_num = num
            current_length = 1

            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1

            max_length = max(max_length, current_length)

    return max_length


def contains_duplicate(nums: List[int]) -> bool:
    """
    存在重复元素

    时间复杂度: O(n)
    空间复杂度: O(n)
    """
    return len(nums) != len(set(nums))


def is_anagram(s: str, t: str) -> bool:
    """
    有效的字母异位词

    时间复杂度: O(n)
    空间复杂度: O(1) - 最多26个字母
    """
    if len(s) != len(t):
        return False

    count = {}
    for c in s:
        count[c] = count.get(c, 0) + 1

    for c in t:
        if c not in count:
            return False
        count[c] -= 1
        if count[c] < 0:
            return False

    return True


def find_duplicate(nums: List[int]) -> int:
    """
    寻找重复数（不使用额外空间的方法：快慢指针）

    数组包含 n+1 个整数，范围 [1, n]
    利用索引作为指针，形成环

    时间复杂度: O(n)
    空间复杂度: O(1)
    """
    slow = fast = nums[0]

    # 找到相遇点
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break

    # 找到环入口
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]

    return slow


def subarray_sum(nums: List[int], k: int) -> int:
    """
    和为 K 的子数组数量

    使用前缀和 + 哈希表

    时间复杂度: O(n)
    空间复杂度: O(n)
    """
    count = 0
    prefix_sum = 0
    prefix_count = {0: 1}  # 前缀和 -> 出现次数

    for num in nums:
        prefix_sum += num

        # 查找是否存在前缀和使得差为 k
        if prefix_sum - k in prefix_count:
            count += prefix_count[prefix_sum - k]

        prefix_count[prefix_sum] = prefix_count.get(prefix_sum, 0) + 1

    return count


# ============================================================
#                    测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("链地址法哈希表测试")
    print("=" * 60)

    ht = HashTableChaining()

    print("\n--- 插入操作 ---")
    ht["apple"] = "苹果"
    ht["banana"] = "香蕉"
    ht["cherry"] = "樱桃"
    ht["date"] = "枣"
    print(f"哈希表: {ht}")

    print("\n--- 查找操作 ---")
    print(f"apple: {ht['apple']}")
    print(f"banana: {ht.get('banana')}")
    print(f"grape exists: {'grape' in ht}")

    print("\n--- 删除操作 ---")
    del ht["date"]
    print(f"删除 date 后: {ht}")

    print("\n--- 遍历操作 ---")
    print(f"keys: {ht.keys()}")
    print(f"values: {ht.values()}")

    print("\n" + "=" * 60)
    print("LRU 缓存测试")
    print("=" * 60)

    lru = LRUCache(3)

    print("\n--- 操作序列 ---")
    operations = [
        ("put", 1, 1),
        ("put", 2, 2),
        ("put", 3, 3),
        ("get", 1, None),
        ("put", 4, 4),  # 淘汰 2
        ("get", 2, None),
        ("get", 3, None),
    ]

    for op in operations:
        if op[0] == "put":
            lru.put(op[1], op[2])
            print(f"put({op[1]}, {op[2]})")
        else:
            result = lru.get(op[1])
            print(f"get({op[1]}) = {result}")

    print("\n" + "=" * 60)
    print("算法测试")
    print("=" * 60)

    print("\n--- 两数之和 ---")
    nums = [2, 7, 11, 15]
    target = 9
    print(f"nums: {nums}, target: {target}")
    print(f"结果: {two_sum(nums, target)}")

    print("\n--- 字母异位词分组 ---")
    strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    print(f"输入: {strs}")
    print(f"分组: {group_anagrams(strs)}")

    print("\n--- 最长连续序列 ---")
    nums = [100, 4, 200, 1, 3, 2]
    print(f"输入: {nums}")
    print(f"最长连续序列长度: {longest_consecutive(nums)}")

    print("\n--- 和为K的子数组 ---")
    nums = [1, 1, 1]
    k = 2
    print(f"nums: {nums}, k: {k}")
    print(f"子数组数量: {subarray_sum(nums, k)}")
