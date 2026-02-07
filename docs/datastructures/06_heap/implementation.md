# implementation

::: info 文件信息
- 📄 原文件：`implementation.py`
- 🔤 语言：python
:::

堆实现
包含最大堆、最小堆、优先队列和常见堆算法。

## 完整代码

```python
from typing import TypeVar, Generic, List, Optional, Callable
import heapq

T = TypeVar('T')


# ============================================================
#                    最大堆实现
# ============================================================

class MaxHeap(Generic[T]):
    """
    最大堆实现

    特性：父节点 >= 子节点
    """

    def __init__(self, items: List[T] = None):
        self._heap: List[T] = []
        if items:
            self._build_heap(items)

    def _build_heap(self, items: List[T]) -> None:
        """
        建堆（Floyd 算法）

        时间复杂度: O(n)
        """
        self._heap = items.copy()
        # 从最后一个非叶节点开始下沉
        for i in range(len(self._heap) // 2 - 1, -1, -1):
            self._sift_down(i)

    def push(self, item: T) -> None:
        """
        插入元素

        时间复杂度: O(log n)
        """
        self._heap.append(item)
        self._sift_up(len(self._heap) - 1)

    def pop(self) -> T:
        """
        删除并返回最大元素

        时间复杂度: O(log n)
        """
        if not self._heap:
            raise IndexError("堆为空")

        max_val = self._heap[0]

        # 将末尾元素移到堆顶
        last = self._heap.pop()
        if self._heap:
            self._heap[0] = last
            self._sift_down(0)

        return max_val

    def peek(self) -> T:
        """
        查看最大元素

        时间复杂度: O(1)
        """
        if not self._heap:
            raise IndexError("堆为空")
        return self._heap[0]

    def _sift_up(self, index: int) -> None:
        """上浮操作"""
        while index > 0:
            parent = (index - 1) // 2
            if self._heap[index] > self._heap[parent]:
                self._heap[index], self._heap[parent] = \
                    self._heap[parent], self._heap[index]
                index = parent
            else:
                break

    def _sift_down(self, index: int) -> None:
        """下沉操作"""
        size = len(self._heap)

        while True:
            largest = index
            left = 2 * index + 1
            right = 2 * index + 2

            if left < size and self._heap[left] > self._heap[largest]:
                largest = left
            if right < size and self._heap[right] > self._heap[largest]:
                largest = right

            if largest != index:
                self._heap[index], self._heap[largest] = \
                    self._heap[largest], self._heap[index]
                index = largest
            else:
                break

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def __repr__(self) -> str:
        return f"MaxHeap({self._heap})"


# ============================================================
#                    最小堆实现
# ============================================================

class MinHeap(Generic[T]):
    """
    最小堆实现

    特性：父节点 <= 子节点
    """

    def __init__(self, items: List[T] = None):
        self._heap: List[T] = []
        if items:
            self._build_heap(items)

    def _build_heap(self, items: List[T]) -> None:
        """建堆 O(n)"""
        self._heap = items.copy()
        for i in range(len(self._heap) // 2 - 1, -1, -1):
            self._sift_down(i)

    def push(self, item: T) -> None:
        """插入元素 O(log n)"""
        self._heap.append(item)
        self._sift_up(len(self._heap) - 1)

    def pop(self) -> T:
        """删除最小元素 O(log n)"""
        if not self._heap:
            raise IndexError("堆为空")

        min_val = self._heap[0]
        last = self._heap.pop()
        if self._heap:
            self._heap[0] = last
            self._sift_down(0)

        return min_val

    def peek(self) -> T:
        """查看最小元素 O(1)"""
        if not self._heap:
            raise IndexError("堆为空")
        return self._heap[0]

    def _sift_up(self, index: int) -> None:
        while index > 0:
            parent = (index - 1) // 2
            if self._heap[index] < self._heap[parent]:
                self._heap[index], self._heap[parent] = \
                    self._heap[parent], self._heap[index]
                index = parent
            else:
                break

    def _sift_down(self, index: int) -> None:
        size = len(self._heap)

        while True:
            smallest = index
            left = 2 * index + 1
            right = 2 * index + 2

            if left < size and self._heap[left] < self._heap[smallest]:
                smallest = left
            if right < size and self._heap[right] < self._heap[smallest]:
                smallest = right

            if smallest != index:
                self._heap[index], self._heap[smallest] = \
                    self._heap[smallest], self._heap[index]
                index = smallest
            else:
                break

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0


# ============================================================
#                    优先队列
# ============================================================

class PriorityQueue(Generic[T]):
    """
    优先队列（基于最小堆）

    支持自定义优先级
    """

    def __init__(self, key: Callable[[T], any] = None, reverse: bool = False):
        """
        Args:
            key: 优先级函数
            reverse: True 为最大优先
        """
        self._heap = []
        self._key = key or (lambda x: x)
        self._reverse = reverse
        self._counter = 0  # 保证稳定性

    def push(self, item: T) -> None:
        """入队"""
        priority = self._key(item)
        if self._reverse:
            priority = -priority if isinstance(priority, (int, float)) else priority

        heapq.heappush(self._heap, (priority, self._counter, item))
        self._counter += 1

    def pop(self) -> T:
        """出队"""
        if not self._heap:
            raise IndexError("队列为空")
        return heapq.heappop(self._heap)[2]

    def peek(self) -> T:
        """查看队首"""
        if not self._heap:
            raise IndexError("队列为空")
        return self._heap[0][2]

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0


# ============================================================
#                    堆排序
# ============================================================

def heap_sort(arr: List[T]) -> List[T]:
    """
    堆排序

    时间复杂度: O(n log n)
    空间复杂度: O(1)
    不稳定排序
    """
    result = arr.copy()
    n = len(result)

    def sift_down(heap_size: int, root: int):
        while True:
            largest = root
            left = 2 * root + 1
            right = 2 * root + 2

            if left < heap_size and result[left] > result[largest]:
                largest = left
            if right < heap_size and result[right] > result[largest]:
                largest = right

            if largest != root:
                result[root], result[largest] = result[largest], result[root]
                root = largest
            else:
                break

    # 建堆
    for i in range(n // 2 - 1, -1, -1):
        sift_down(n, i)

    # 排序
    for i in range(n - 1, 0, -1):
        result[0], result[i] = result[i], result[0]
        sift_down(i, 0)

    return result


# ============================================================
#                    常见堆算法
# ============================================================

def find_kth_largest(nums: List[int], k: int) -> int:
    """
    数组中的第 K 个最大元素

    方法：维护大小为 k 的最小堆

    时间复杂度: O(n log k)
    空间复杂度: O(k)
    """
    heap = []

    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)

    return heap[0]


def top_k_frequent(nums: List[int], k: int) -> List[int]:
    """
    前 K 个高频元素

    时间复杂度: O(n log k)
    """
    from collections import Counter

    count = Counter(nums)

    # 最小堆，按频率排序
    heap = []
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)

    return [item[1] for item in heap]


def merge_k_sorted_lists(lists: List[List[int]]) -> List[int]:
    """
    合并 K 个有序数组

    时间复杂度: O(n log k)，n 为总元素数
    """
    heap = []
    result = []

    # 初始化堆（值，列表索引，元素索引）
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))

    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)

        # 将下一个元素入堆
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))

    return result


class MedianFinder:
    """
    数据流的中位数

    使用两个堆：
    - 最大堆存储较小的一半
    - 最小堆存储较大的一半

    时间复杂度:
    - addNum: O(log n)
    - findMedian: O(1)
    """

    def __init__(self):
        self.max_heap = []  # 存储较小的一半（取负实现最大堆）
        self.min_heap = []  # 存储较大的一半

    def addNum(self, num: int) -> None:
        # 先加入最大堆
        heapq.heappush(self.max_heap, -num)

        # 将最大堆的最大值移到最小堆
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))

        # 平衡两个堆的大小（最大堆可以多一个）
        if len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def findMedian(self) -> float:
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        return (-self.max_heap[0] + self.min_heap[0]) / 2


def last_stone_weight(stones: List[int]) -> int:
    """
    最后一块石头的重量

    每次取最重的两块石头粉碎

    时间复杂度: O(n log n)
    """
    # 最大堆（取负）
    heap = [-s for s in stones]
    heapq.heapify(heap)

    while len(heap) > 1:
        first = -heapq.heappop(heap)
        second = -heapq.heappop(heap)

        if first != second:
            heapq.heappush(heap, -(first - second))

    return -heap[0] if heap else 0


def k_closest_points(points: List[List[int]], k: int) -> List[List[int]]:
    """
    最接近原点的 K 个点

    时间复杂度: O(n log k)
    """
    heap = []

    for x, y in points:
        dist = x * x + y * y
        heapq.heappush(heap, (-dist, x, y))
        if len(heap) > k:
            heapq.heappop(heap)

    return [[x, y] for _, x, y in heap]


# ============================================================
#                    测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("最大堆测试")
    print("=" * 60)

    max_heap = MaxHeap()

    print("\n--- 插入操作 ---")
    for val in [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]:
        max_heap.push(val)
    print(f"堆: {max_heap}")
    print(f"最大值: {max_heap.peek()}")

    print("\n--- 删除操作 ---")
    while not max_heap.is_empty():
        print(f"pop: {max_heap.pop()}", end=" ")
    print()

    print("\n--- 建堆测试 ---")
    arr = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
    max_heap = MaxHeap(arr)
    print(f"数组 {arr} 建堆后: {max_heap}")

    print("\n" + "=" * 60)
    print("堆排序测试")
    print("=" * 60)

    arr = [64, 34, 25, 12, 22, 11, 90]
    print(f"原数组: {arr}")
    print(f"堆排序: {heap_sort(arr)}")

    print("\n" + "=" * 60)
    print("算法测试")
    print("=" * 60)

    print("\n--- 第 K 大元素 ---")
    nums = [3, 2, 1, 5, 6, 4]
    k = 2
    print(f"数组: {nums}, k={k}")
    print(f"第 {k} 大: {find_kth_largest(nums, k)}")

    print("\n--- Top-K 高频 ---")
    nums = [1, 1, 1, 2, 2, 3]
    k = 2
    print(f"数组: {nums}, k={k}")
    print(f"前 {k} 高频: {top_k_frequent(nums, k)}")

    print("\n--- 合并 K 个有序数组 ---")
    lists = [[1, 4, 5], [1, 3, 4], [2, 6]]
    print(f"输入: {lists}")
    print(f"合并: {merge_k_sorted_lists(lists)}")

    print("\n--- 数据流中位数 ---")
    mf = MedianFinder()
    for num in [1, 2, 3, 4, 5]:
        mf.addNum(num)
        print(f"添加 {num}, 中位数: {mf.findMedian()}")

    print("\n--- 最后一块石头 ---")
    stones = [2, 7, 4, 1, 8, 1]
    print(f"石头: {stones}")
    print(f"最后重量: {last_stone_weight(stones)}")
```
