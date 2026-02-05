"""
============================================================
                    动态数组实现
============================================================
从零实现一个动态数组，理解底层原理。
============================================================
"""

from typing import TypeVar, Generic, Iterator, Optional
import ctypes

T = TypeVar('T')


class DynamicArray(Generic[T]):
    """
    动态数组实现

    特性：
    - 自动扩容/缩容
    - 支持负索引
    - 支持迭代
    - 泛型支持
    """

    def __init__(self, capacity: int = 10):
        """
        初始化动态数组

        Args:
            capacity: 初始容量
        """
        self._size = 0                    # 实际元素数量
        self._capacity = capacity          # 当前容量
        self._array = self._make_array(capacity)  # 底层数组

    def _make_array(self, capacity: int):
        """创建底层数组（模拟C数组）"""
        return (capacity * ctypes.py_object)()

    # ==================== 基本操作 ====================

    def __len__(self) -> int:
        """返回元素数量"""
        return self._size

    def __getitem__(self, index: int) -> T:
        """
        获取元素（支持负索引）

        时间复杂度: O(1)
        """
        index = self._validate_index(index)
        return self._array[index]

    def __setitem__(self, index: int, value: T) -> None:
        """
        设置元素

        时间复杂度: O(1)
        """
        index = self._validate_index(index)
        self._array[index] = value

    def _validate_index(self, index: int) -> int:
        """验证并转换索引"""
        if index < 0:
            index += self._size
        if not 0 <= index < self._size:
            raise IndexError(f"索引 {index} 超出范围 [0, {self._size})")
        return index

    # ==================== 添加操作 ====================

    def append(self, value: T) -> None:
        """
        在末尾添加元素

        时间复杂度: 均摊 O(1)
        """
        if self._size == self._capacity:
            self._resize(2 * self._capacity)  # 扩容 2 倍

        self._array[self._size] = value
        self._size += 1

    def insert(self, index: int, value: T) -> None:
        """
        在指定位置插入元素

        时间复杂度: O(n)

        Args:
            index: 插入位置
            value: 插入的值
        """
        # 允许在末尾插入
        if index < 0:
            index += self._size
        if not 0 <= index <= self._size:
            raise IndexError(f"插入位置 {index} 无效")

        if self._size == self._capacity:
            self._resize(2 * self._capacity)

        # 后移元素：从后往前移动
        for i in range(self._size, index, -1):
            self._array[i] = self._array[i - 1]

        self._array[index] = value
        self._size += 1

    def extend(self, iterable) -> None:
        """添加多个元素"""
        for item in iterable:
            self.append(item)

    # ==================== 删除操作 ====================

    def pop(self, index: int = -1) -> T:
        """
        删除并返回指定位置的元素

        时间复杂度:
            - 末尾: O(1)
            - 其他位置: O(n)
        """
        index = self._validate_index(index)
        value = self._array[index]

        # 前移元素
        for i in range(index, self._size - 1):
            self._array[i] = self._array[i + 1]

        self._size -= 1
        self._array[self._size] = None  # 帮助垃圾回收

        # 缩容：当元素数量少于容量的 1/4 时
        if self._size > 0 and self._size <= self._capacity // 4:
            self._resize(self._capacity // 2)

        return value

    def remove(self, value: T) -> None:
        """
        删除第一个匹配的元素

        时间复杂度: O(n)
        """
        for i in range(self._size):
            if self._array[i] == value:
                self.pop(i)
                return
        raise ValueError(f"{value} 不在数组中")

    def clear(self) -> None:
        """清空数组"""
        self._size = 0
        self._capacity = 10
        self._array = self._make_array(self._capacity)

    # ==================== 查找操作 ====================

    def index(self, value: T) -> int:
        """
        查找元素的索引

        时间复杂度: O(n)
        """
        for i in range(self._size):
            if self._array[i] == value:
                return i
        raise ValueError(f"{value} 不在数组中")

    def count(self, value: T) -> int:
        """统计元素出现次数"""
        return sum(1 for i in range(self._size) if self._array[i] == value)

    def __contains__(self, value: T) -> bool:
        """检查元素是否存在"""
        for i in range(self._size):
            if self._array[i] == value:
                return True
        return False

    # ==================== 内部方法 ====================

    def _resize(self, new_capacity: int) -> None:
        """
        调整容量

        时间复杂度: O(n)
        """
        new_array = self._make_array(new_capacity)
        for i in range(self._size):
            new_array[i] = self._array[i]
        self._array = new_array
        self._capacity = new_capacity

    # ==================== 迭代器 ====================

    def __iter__(self) -> Iterator[T]:
        """支持 for 循环"""
        for i in range(self._size):
            yield self._array[i]

    def __reversed__(self) -> Iterator[T]:
        """支持反向迭代"""
        for i in range(self._size - 1, -1, -1):
            yield self._array[i]

    # ==================== 其他方法 ====================

    def __repr__(self) -> str:
        """字符串表示"""
        items = [str(self._array[i]) for i in range(self._size)]
        return f"DynamicArray([{', '.join(items)}])"

    def __eq__(self, other) -> bool:
        """判断相等"""
        if not isinstance(other, DynamicArray):
            return False
        if len(self) != len(other):
            return False
        for i in range(self._size):
            if self._array[i] != other._array[i]:
                return False
        return True

    @property
    def capacity(self) -> int:
        """返回当前容量"""
        return self._capacity

    def is_empty(self) -> bool:
        """检查是否为空"""
        return self._size == 0


# ============================================================
#                    常用数组算法
# ============================================================

def binary_search(arr: list, target) -> int:
    """
    二分查找（要求有序数组）

    时间复杂度: O(log n)
    空间复杂度: O(1)

    Returns:
        找到返回索引，否则返回 -1
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2  # 防止溢出

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


def two_sum(arr: list, target: int) -> tuple:
    """
    两数之和

    时间复杂度: O(n)
    空间复杂度: O(n)

    Returns:
        返回两个数的索引，如 (0, 2)
    """
    seen = {}  # 值 -> 索引

    for i, num in enumerate(arr):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i

    return (-1, -1)


def remove_duplicates(arr: list) -> int:
    """
    原地删除排序数组中的重复项

    时间复杂度: O(n)
    空间复杂度: O(1)

    Returns:
        不重复元素的数量
    """
    if not arr:
        return 0

    # 双指针：slow 指向不重复区域的末尾
    slow = 0

    for fast in range(1, len(arr)):
        if arr[fast] != arr[slow]:
            slow += 1
            arr[slow] = arr[fast]

    return slow + 1


def rotate_array(arr: list, k: int) -> None:
    """
    旋转数组（向右移动 k 步）

    方法：三次反转
    [1,2,3,4,5,6,7] k=3
    -> [7,6,5,4,3,2,1] 全部反转
    -> [5,6,7,4,3,2,1] 反转前 k 个
    -> [5,6,7,1,2,3,4] 反转后 n-k 个

    时间复杂度: O(n)
    空间复杂度: O(1)
    """
    n = len(arr)
    k = k % n  # 处理 k > n 的情况

    def reverse(left: int, right: int):
        while left < right:
            arr[left], arr[right] = arr[right], arr[left]
            left += 1
            right -= 1

    reverse(0, n - 1)      # 全部反转
    reverse(0, k - 1)      # 反转前 k 个
    reverse(k, n - 1)      # 反转后 n-k 个


def max_subarray(arr: list) -> int:
    """
    最大子数组和（Kadane 算法）

    时间复杂度: O(n)
    空间复杂度: O(1)
    """
    if not arr:
        return 0

    max_sum = current_sum = arr[0]

    for num in arr[1:]:
        # 要么加入当前子数组，要么从当前位置重新开始
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)

    return max_sum


def merge_sorted_arrays(arr1: list, m: int, arr2: list, n: int) -> None:
    """
    合并两个有序数组（arr1 有足够空间）

    方法：逆向双指针，从后往前填充

    时间复杂度: O(m + n)
    空间复杂度: O(1)
    """
    p1 = m - 1  # arr1 的指针
    p2 = n - 1  # arr2 的指针
    p = m + n - 1  # 合并后的指针

    while p2 >= 0:
        if p1 >= 0 and arr1[p1] > arr2[p2]:
            arr1[p] = arr1[p1]
            p1 -= 1
        else:
            arr1[p] = arr2[p2]
            p2 -= 1
        p -= 1


# ============================================================
#                    测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("动态数组测试")
    print("=" * 60)

    # 基本操作测试
    arr = DynamicArray()

    print("\n--- 添加元素 ---")
    for i in range(1, 6):
        arr.append(i * 10)
    print(f"添加后: {arr}")
    print(f"长度: {len(arr)}, 容量: {arr.capacity}")

    print("\n--- 插入元素 ---")
    arr.insert(2, 99)
    print(f"在索引2插入99: {arr}")

    print("\n--- 访问元素 ---")
    print(f"arr[0] = {arr[0]}")
    print(f"arr[-1] = {arr[-1]}")

    print("\n--- 删除元素 ---")
    popped = arr.pop()
    print(f"pop(): {popped}, 数组: {arr}")
    popped = arr.pop(0)
    print(f"pop(0): {popped}, 数组: {arr}")

    print("\n--- 迭代 ---")
    print("正向:", list(arr))
    print("反向:", list(reversed(arr)))

    print("\n--- 查找 ---")
    print(f"99 in arr: {99 in arr}")
    print(f"index(99): {arr.index(99)}")

    print("\n" + "=" * 60)
    print("算法测试")
    print("=" * 60)

    print("\n--- 二分查找 ---")
    sorted_arr = [1, 3, 5, 7, 9, 11, 13]
    print(f"数组: {sorted_arr}")
    print(f"查找 7: 索引 {binary_search(sorted_arr, 7)}")
    print(f"查找 4: 索引 {binary_search(sorted_arr, 4)}")

    print("\n--- 两数之和 ---")
    nums = [2, 7, 11, 15]
    target = 9
    print(f"数组: {nums}, 目标: {target}")
    print(f"结果: {two_sum(nums, target)}")

    print("\n--- 删除重复项 ---")
    nums = [1, 1, 2, 2, 3, 4, 4, 5]
    print(f"原数组: {nums}")
    new_len = remove_duplicates(nums)
    print(f"去重后: {nums[:new_len]}")

    print("\n--- 旋转数组 ---")
    nums = [1, 2, 3, 4, 5, 6, 7]
    print(f"原数组: {nums}")
    rotate_array(nums, 3)
    print(f"右移3位: {nums}")

    print("\n--- 最大子数组和 ---")
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(f"数组: {nums}")
    print(f"最大子数组和: {max_subarray(nums)}")

    print("\n--- 合并有序数组 ---")
    arr1 = [1, 3, 5, 0, 0, 0]
    arr2 = [2, 4, 6]
    print(f"arr1: {arr1[:3]}, arr2: {arr2}")
    merge_sorted_arrays(arr1, 3, arr2, 3)
    print(f"合并后: {arr1}")
