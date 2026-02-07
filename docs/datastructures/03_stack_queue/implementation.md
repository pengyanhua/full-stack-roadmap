# implementation

::: info 文件信息
- 📄 原文件：`implementation.py`
- 🔤 语言：python
:::

栈和队列实现
包含栈、队列、循环队列、双端队列、单调栈等实现。

## 完整代码

```python
from typing import TypeVar, Generic, List, Optional, Iterator
from collections import deque

T = TypeVar('T')


# ============================================================
#                    栈实现
# ============================================================

class Stack(Generic[T]):
    """
    栈实现（基于数组）

    时间复杂度：所有操作 O(1)
    """

    def __init__(self):
        self._data: List[T] = []

    def push(self, item: T) -> None:
        """入栈"""
        self._data.append(item)

    def pop(self) -> T:
        """出栈"""
        if self.is_empty():
            raise IndexError("栈为空")
        return self._data.pop()

    def peek(self) -> T:
        """查看栈顶"""
        if self.is_empty():
            raise IndexError("栈为空")
        return self._data[-1]

    def is_empty(self) -> bool:
        """是否为空"""
        return len(self._data) == 0

    def size(self) -> int:
        """元素数量"""
        return len(self._data)

    def __len__(self) -> int:
        return self.size()

    def __repr__(self) -> str:
        return f"Stack({self._data})"


class LinkedStack(Generic[T]):
    """
    栈实现（基于链表）

    时间复杂度：所有操作 O(1)
    """

    class Node:
        def __init__(self, val, next=None):
            self.val = val
            self.next = next

    def __init__(self):
        self._top: Optional[self.Node] = None
        self._size = 0

    def push(self, item: T) -> None:
        """入栈"""
        self._top = self.Node(item, self._top)
        self._size += 1

    def pop(self) -> T:
        """出栈"""
        if self.is_empty():
            raise IndexError("栈为空")
        val = self._top.val
        self._top = self._top.next
        self._size -= 1
        return val

    def peek(self) -> T:
        """查看栈顶"""
        if self.is_empty():
            raise IndexError("栈为空")
        return self._top.val

    def is_empty(self) -> bool:
        return self._size == 0

    def size(self) -> int:
        return self._size


class MinStack:
    """
    最小栈：支持 O(1) 获取最小值

    思路：使用辅助栈同步存储当前最小值
    """

    def __init__(self):
        self._data = []
        self._min_stack = []

    def push(self, val: int) -> None:
        self._data.append(val)
        # 如果当前值 <= 最小值，入辅助栈
        if not self._min_stack or val <= self._min_stack[-1]:
            self._min_stack.append(val)

    def pop(self) -> None:
        val = self._data.pop()
        # 如果弹出的是最小值，辅助栈也弹出
        if val == self._min_stack[-1]:
            self._min_stack.pop()

    def top(self) -> int:
        return self._data[-1]

    def get_min(self) -> int:
        return self._min_stack[-1]


# ============================================================
#                    队列实现
# ============================================================

class Queue(Generic[T]):
    """
    队列实现（基于 deque）

    时间复杂度：所有操作 O(1)
    """

    def __init__(self):
        self._data = deque()

    def enqueue(self, item: T) -> None:
        """入队"""
        self._data.append(item)

    def dequeue(self) -> T:
        """出队"""
        if self.is_empty():
            raise IndexError("队列为空")
        return self._data.popleft()

    def front(self) -> T:
        """查看队首"""
        if self.is_empty():
            raise IndexError("队列为空")
        return self._data[0]

    def rear(self) -> T:
        """查看队尾"""
        if self.is_empty():
            raise IndexError("队列为空")
        return self._data[-1]

    def is_empty(self) -> bool:
        return len(self._data) == 0

    def size(self) -> int:
        return len(self._data)

    def __len__(self) -> int:
        return self.size()

    def __repr__(self) -> str:
        return f"Queue({list(self._data)})"


class CircularQueue:
    """
    循环队列（基于数组）

    解决普通数组队列的"假溢出"问题
    """

    def __init__(self, capacity: int):
        # 多分配一个空间用于区分空和满
        self._capacity = capacity + 1
        self._data = [None] * self._capacity
        self._front = 0
        self._rear = 0

    def enqueue(self, item) -> bool:
        """入队，成功返回 True"""
        if self.is_full():
            return False
        self._data[self._rear] = item
        self._rear = (self._rear + 1) % self._capacity
        return True

    def dequeue(self):
        """出队"""
        if self.is_empty():
            raise IndexError("队列为空")
        item = self._data[self._front]
        self._front = (self._front + 1) % self._capacity
        return item

    def front(self):
        """查看队首"""
        if self.is_empty():
            raise IndexError("队列为空")
        return self._data[self._front]

    def rear(self):
        """查看队尾"""
        if self.is_empty():
            raise IndexError("队列为空")
        # 队尾指针的前一个位置
        return self._data[(self._rear - 1 + self._capacity) % self._capacity]

    def is_empty(self) -> bool:
        return self._front == self._rear

    def is_full(self) -> bool:
        return (self._rear + 1) % self._capacity == self._front

    def size(self) -> int:
        return (self._rear - self._front + self._capacity) % self._capacity

    def __repr__(self) -> str:
        if self.is_empty():
            return "CircularQueue([])"
        items = []
        i = self._front
        while i != self._rear:
            items.append(str(self._data[i]))
            i = (i + 1) % self._capacity
        return f"CircularQueue([{', '.join(items)}])"


class Deque(Generic[T]):
    """
    双端队列实现

    两端都可以进行插入和删除操作
    """

    def __init__(self):
        self._data = deque()

    def add_first(self, item: T) -> None:
        """头部添加"""
        self._data.appendleft(item)

    def add_last(self, item: T) -> None:
        """尾部添加"""
        self._data.append(item)

    def remove_first(self) -> T:
        """头部删除"""
        if self.is_empty():
            raise IndexError("双端队列为空")
        return self._data.popleft()

    def remove_last(self) -> T:
        """尾部删除"""
        if self.is_empty():
            raise IndexError("双端队列为空")
        return self._data.pop()

    def first(self) -> T:
        """查看头部"""
        if self.is_empty():
            raise IndexError("双端队列为空")
        return self._data[0]

    def last(self) -> T:
        """查看尾部"""
        if self.is_empty():
            raise IndexError("双端队列为空")
        return self._data[-1]

    def is_empty(self) -> bool:
        return len(self._data) == 0

    def size(self) -> int:
        return len(self._data)

    def __len__(self) -> int:
        return self.size()


# ============================================================
#                    用栈实现队列
# ============================================================

class QueueUsingStacks:
    """
    用两个栈实现队列

    思路：
    - inStack 负责入队
    - outStack 负责出队
    - 出队时如果 outStack 为空，将 inStack 倒入 outStack

    均摊时间复杂度：O(1)
    """

    def __init__(self):
        self._in_stack = []
        self._out_stack = []

    def push(self, x: int) -> None:
        """入队"""
        self._in_stack.append(x)

    def pop(self) -> int:
        """出队"""
        self._move_if_needed()
        return self._out_stack.pop()

    def peek(self) -> int:
        """查看队首"""
        self._move_if_needed()
        return self._out_stack[-1]

    def empty(self) -> bool:
        return not self._in_stack and not self._out_stack

    def _move_if_needed(self) -> None:
        """将 inStack 倒入 outStack"""
        if not self._out_stack:
            while self._in_stack:
                self._out_stack.append(self._in_stack.pop())


# ============================================================
#                    用队列实现栈
# ============================================================

class StackUsingQueue:
    """
    用一个队列实现栈

    思路：入栈时，将队列中已有元素依次出队再入队
    使新元素始终在队首
    """

    def __init__(self):
        self._queue = deque()

    def push(self, x: int) -> None:
        """入栈"""
        self._queue.append(x)
        # 将之前的元素移到后面
        for _ in range(len(self._queue) - 1):
            self._queue.append(self._queue.popleft())

    def pop(self) -> int:
        """出栈"""
        return self._queue.popleft()

    def top(self) -> int:
        """查看栈顶"""
        return self._queue[0]

    def empty(self) -> bool:
        return len(self._queue) == 0


# ============================================================
#                    单调栈
# ============================================================

class MonotonicStack:
    """
    单调栈示例：下一个更大元素

    对于数组中的每个元素，找到它右边第一个比它大的元素
    """

    @staticmethod
    def next_greater_element(nums: List[int]) -> List[int]:
        """
        下一个更大元素

        输入: [2, 1, 2, 4, 3]
        输出: [4, 2, 4, -1, -1]

        时间复杂度: O(n)
        空间复杂度: O(n)
        """
        n = len(nums)
        result = [-1] * n
        stack = []  # 存储索引

        for i in range(n):
            # 当前元素大于栈顶，说明找到了栈顶的下一个更大元素
            while stack and nums[i] > nums[stack[-1]]:
                idx = stack.pop()
                result[idx] = nums[i]
            stack.append(i)

        return result

    @staticmethod
    def daily_temperatures(temperatures: List[int]) -> List[int]:
        """
        每日温度：需要等待几天才能等到更暖和的一天

        输入: [73, 74, 75, 71, 69, 72, 76, 73]
        输出: [1, 1, 4, 2, 1, 1, 0, 0]

        时间复杂度: O(n)
        空间复杂度: O(n)
        """
        n = len(temperatures)
        result = [0] * n
        stack = []  # 存储索引

        for i in range(n):
            while stack and temperatures[i] > temperatures[stack[-1]]:
                idx = stack.pop()
                result[idx] = i - idx
            stack.append(i)

        return result

    @staticmethod
    def largest_rectangle_in_histogram(heights: List[int]) -> int:
        """
        柱状图中最大的矩形

        使用单调递增栈

        时间复杂度: O(n)
        空间复杂度: O(n)
        """
        stack = []  # 存储索引，对应高度单调递增
        max_area = 0
        heights = [0] + heights + [0]  # 添加哨兵

        for i, h in enumerate(heights):
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                width = i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)

        return max_area


# ============================================================
#                    单调队列
# ============================================================

class MonotonicQueue:
    """
    单调队列示例：滑动窗口最大值
    """

    @staticmethod
    def max_sliding_window(nums: List[int], k: int) -> List[int]:
        """
        滑动窗口最大值

        输入: nums = [1,3,-1,-3,5,3,6,7], k = 3
        输出: [3,3,5,5,6,7]

        使用单调递减队列，队首始终是窗口最大值

        时间复杂度: O(n)
        空间复杂度: O(k)
        """
        if not nums or k == 0:
            return []

        result = []
        queue = deque()  # 存储索引，对应值单调递减

        for i, num in enumerate(nums):
            # 移除窗口外的元素
            while queue and queue[0] < i - k + 1:
                queue.popleft()

            # 保持单调递减：移除比当前元素小的
            while queue and nums[queue[-1]] < num:
                queue.pop()

            queue.append(i)

            # 窗口形成后开始记录结果
            if i >= k - 1:
                result.append(nums[queue[0]])

        return result


# ============================================================
#                    经典栈算法
# ============================================================

def is_valid_parentheses(s: str) -> bool:
    """
    有效的括号

    输入: "([{}])"
    输出: True

    时间复杂度: O(n)
    空间复杂度: O(n)
    """
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}

    for char in s:
        if char in mapping:
            # 右括号：检查是否匹配
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            # 左括号：入栈
            stack.append(char)

    return len(stack) == 0


def eval_rpn(tokens: List[str]) -> int:
    """
    逆波兰表达式求值（后缀表达式）

    输入: ["2", "1", "+", "3", "*"]
    输出: 9  ((2 + 1) * 3)

    时间复杂度: O(n)
    空间复杂度: O(n)
    """
    stack = []

    for token in tokens:
        if token in '+-*/':
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            else:
                # 注意：Python 的整除需要特殊处理负数
                stack.append(int(a / b))
        else:
            stack.append(int(token))

    return stack[0]


def decode_string(s: str) -> str:
    """
    字符串解码

    输入: "3[a2[c]]"
    输出: "accaccacc"

    时间复杂度: O(n)
    空间复杂度: O(n)
    """
    stack = []
    current_num = 0
    current_str = ""

    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            # 保存当前状态
            stack.append((current_str, current_num))
            current_str = ""
            current_num = 0
        elif char == ']':
            # 恢复并展开
            prev_str, num = stack.pop()
            current_str = prev_str + current_str * num
        else:
            current_str += char

    return current_str


# ============================================================
#                    测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("栈测试")
    print("=" * 60)

    stack = Stack()
    for i in [1, 2, 3, 4, 5]:
        stack.push(i)
    print(f"栈: {stack}")
    print(f"pop: {stack.pop()}, peek: {stack.peek()}")
    print(f"size: {stack.size()}, isEmpty: {stack.is_empty()}")

    print("\n--- 最小栈 ---")
    min_stack = MinStack()
    for val in [3, 5, 2, 1, 4]:
        min_stack.push(val)
        print(f"push {val}, min = {min_stack.get_min()}")

    print("\n" + "=" * 60)
    print("队列测试")
    print("=" * 60)

    queue = Queue()
    for i in [1, 2, 3, 4, 5]:
        queue.enqueue(i)
    print(f"队列: {queue}")
    print(f"dequeue: {queue.dequeue()}, front: {queue.front()}")

    print("\n--- 循环队列 ---")
    cq = CircularQueue(5)
    for i in [1, 2, 3, 4, 5]:
        cq.enqueue(i)
    print(f"循环队列: {cq}")
    cq.dequeue()
    cq.dequeue()
    cq.enqueue(6)
    print(f"出队两个后入队6: {cq}")

    print("\n" + "=" * 60)
    print("单调栈测试")
    print("=" * 60)

    nums = [2, 1, 2, 4, 3]
    print(f"数组: {nums}")
    print(f"下一个更大元素: {MonotonicStack.next_greater_element(nums)}")

    temps = [73, 74, 75, 71, 69, 72, 76, 73]
    print(f"\n温度: {temps}")
    print(f"每日温度: {MonotonicStack.daily_temperatures(temps)}")

    heights = [2, 1, 5, 6, 2, 3]
    print(f"\n柱状图: {heights}")
    print(f"最大矩形面积: {MonotonicStack.largest_rectangle_in_histogram(heights)}")

    print("\n" + "=" * 60)
    print("滑动窗口测试")
    print("=" * 60)

    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    print(f"数组: {nums}, k = {k}")
    print(f"滑动窗口最大值: {MonotonicQueue.max_sliding_window(nums, k)}")

    print("\n" + "=" * 60)
    print("经典算法测试")
    print("=" * 60)

    print("\n--- 括号匹配 ---")
    tests = ["([{}])", "([)]", "((()))", ""]
    for s in tests:
        print(f"'{s}' -> {is_valid_parentheses(s)}")

    print("\n--- 逆波兰表达式 ---")
    tokens = ["2", "1", "+", "3", "*"]
    print(f"{tokens} = {eval_rpn(tokens)}")

    print("\n--- 字符串解码 ---")
    s = "3[a2[c]]"
    print(f"'{s}' -> '{decode_string(s)}'")
```
