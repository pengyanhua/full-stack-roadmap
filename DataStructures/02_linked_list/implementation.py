"""
============================================================
                    链表实现
============================================================
包含单向链表、双向链表的完整实现，以及常见算法。
============================================================
"""

from typing import TypeVar, Generic, Optional, Iterator, List

T = TypeVar('T')


# ============================================================
#                    单向链表
# ============================================================

class ListNode(Generic[T]):
    """单向链表节点"""

    def __init__(self, val: T, next: 'ListNode[T]' = None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"ListNode({self.val})"


class SinglyLinkedList(Generic[T]):
    """
    单向链表实现

    特性：
    - 维护 head 和 tail 指针
    - 支持头尾 O(1) 操作
    - 支持迭代
    """

    def __init__(self):
        self._head: Optional[ListNode[T]] = None
        self._tail: Optional[ListNode[T]] = None
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    # ==================== 添加操作 ====================

    def append(self, val: T) -> None:
        """
        尾部添加

        时间复杂度: O(1)
        """
        new_node = ListNode(val)

        if self.is_empty():
            self._head = new_node
            self._tail = new_node
        else:
            self._tail.next = new_node
            self._tail = new_node

        self._size += 1

    def prepend(self, val: T) -> None:
        """
        头部添加

        时间复杂度: O(1)
        """
        new_node = ListNode(val)
        new_node.next = self._head
        self._head = new_node

        if self._tail is None:
            self._tail = new_node

        self._size += 1

    def insert(self, index: int, val: T) -> None:
        """
        在指定位置插入

        时间复杂度: O(n)
        """
        if index < 0 or index > self._size:
            raise IndexError(f"索引 {index} 超出范围")

        if index == 0:
            self.prepend(val)
        elif index == self._size:
            self.append(val)
        else:
            # 找到前驱节点
            prev = self._get_node(index - 1)
            new_node = ListNode(val)
            new_node.next = prev.next
            prev.next = new_node
            self._size += 1

    # ==================== 删除操作 ====================

    def pop_first(self) -> T:
        """
        删除头部元素

        时间复杂度: O(1)
        """
        if self.is_empty():
            raise IndexError("链表为空")

        val = self._head.val
        self._head = self._head.next
        self._size -= 1

        if self.is_empty():
            self._tail = None

        return val

    def pop_last(self) -> T:
        """
        删除尾部元素

        时间复杂度: O(n) - 需要找到倒数第二个节点
        """
        if self.is_empty():
            raise IndexError("链表为空")

        if self._size == 1:
            val = self._head.val
            self._head = None
            self._tail = None
            self._size = 0
            return val

        # 找到倒数第二个节点
        current = self._head
        while current.next != self._tail:
            current = current.next

        val = self._tail.val
        current.next = None
        self._tail = current
        self._size -= 1

        return val

    def remove(self, val: T) -> bool:
        """
        删除第一个匹配的元素

        时间复杂度: O(n)
        """
        if self.is_empty():
            return False

        # 删除头节点
        if self._head.val == val:
            self.pop_first()
            return True

        # 查找并删除
        current = self._head
        while current.next:
            if current.next.val == val:
                if current.next == self._tail:
                    self._tail = current
                current.next = current.next.next
                self._size -= 1
                return True
            current = current.next

        return False

    def delete_at(self, index: int) -> T:
        """
        删除指定位置的元素

        时间复杂度: O(n)
        """
        if index < 0 or index >= self._size:
            raise IndexError(f"索引 {index} 超出范围")

        if index == 0:
            return self.pop_first()

        prev = self._get_node(index - 1)
        val = prev.next.val

        if prev.next == self._tail:
            self._tail = prev

        prev.next = prev.next.next
        self._size -= 1

        return val

    # ==================== 访问操作 ====================

    def get(self, index: int) -> T:
        """
        获取指定位置的元素

        时间复杂度: O(n)
        """
        return self._get_node(index).val

    def _get_node(self, index: int) -> ListNode[T]:
        """获取指定位置的节点"""
        if index < 0 or index >= self._size:
            raise IndexError(f"索引 {index} 超出范围")

        current = self._head
        for _ in range(index):
            current = current.next
        return current

    def __getitem__(self, index: int) -> T:
        return self.get(index)

    def first(self) -> T:
        """获取头部元素"""
        if self.is_empty():
            raise IndexError("链表为空")
        return self._head.val

    def last(self) -> T:
        """获取尾部元素"""
        if self.is_empty():
            raise IndexError("链表为空")
        return self._tail.val

    # ==================== 查找操作 ====================

    def find(self, val: T) -> int:
        """
        查找元素的索引

        时间复杂度: O(n)
        返回: 索引，未找到返回 -1
        """
        current = self._head
        index = 0
        while current:
            if current.val == val:
                return index
            current = current.next
            index += 1
        return -1

    def __contains__(self, val: T) -> bool:
        return self.find(val) != -1

    # ==================== 其他操作 ====================

    def reverse(self) -> None:
        """
        原地反转链表

        时间复杂度: O(n)
        空间复杂度: O(1)
        """
        if self._size <= 1:
            return

        self._tail = self._head

        prev = None
        current = self._head

        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node

        self._head = prev

    def to_list(self) -> List[T]:
        """转换为 Python 列表"""
        result = []
        current = self._head
        while current:
            result.append(current.val)
            current = current.next
        return result

    def __iter__(self) -> Iterator[T]:
        current = self._head
        while current:
            yield current.val
            current = current.next

    def __repr__(self) -> str:
        if self.is_empty():
            return "SinglyLinkedList()"
        values = " -> ".join(str(val) for val in self)
        return f"SinglyLinkedList({values})"

    def clear(self) -> None:
        """清空链表"""
        self._head = None
        self._tail = None
        self._size = 0


# ============================================================
#                    双向链表
# ============================================================

class DoublyListNode(Generic[T]):
    """双向链表节点"""

    def __init__(self, val: T, prev: 'DoublyListNode[T]' = None,
                 next: 'DoublyListNode[T]' = None):
        self.val = val
        self.prev = prev
        self.next = next


class DoublyLinkedList(Generic[T]):
    """
    双向链表实现

    特性：
    - 使用哨兵节点简化边界处理
    - 头尾操作都是 O(1)
    - 已知节点的删除是 O(1)
    """

    def __init__(self):
        # 哨兵节点
        self._head = DoublyListNode(None)  # 虚拟头
        self._tail = DoublyListNode(None)  # 虚拟尾
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    # ==================== 添加操作 ====================

    def _insert_between(self, val: T, prev_node: DoublyListNode,
                        next_node: DoublyListNode) -> DoublyListNode:
        """在两个节点之间插入"""
        new_node = DoublyListNode(val, prev_node, next_node)
        prev_node.next = new_node
        next_node.prev = new_node
        self._size += 1
        return new_node

    def append(self, val: T) -> None:
        """尾部添加 - O(1)"""
        self._insert_between(val, self._tail.prev, self._tail)

    def prepend(self, val: T) -> None:
        """头部添加 - O(1)"""
        self._insert_between(val, self._head, self._head.next)

    def insert(self, index: int, val: T) -> None:
        """指定位置插入 - O(n)"""
        if index < 0 or index > self._size:
            raise IndexError(f"索引 {index} 超出范围")

        node = self._get_node(index) if index < self._size else self._tail
        self._insert_between(val, node.prev, node)

    # ==================== 删除操作 ====================

    def _delete_node(self, node: DoublyListNode) -> T:
        """删除指定节点 - O(1)"""
        val = node.val
        node.prev.next = node.next
        node.next.prev = node.prev
        self._size -= 1
        return val

    def pop_first(self) -> T:
        """删除头部 - O(1)"""
        if self.is_empty():
            raise IndexError("链表为空")
        return self._delete_node(self._head.next)

    def pop_last(self) -> T:
        """删除尾部 - O(1)"""
        if self.is_empty():
            raise IndexError("链表为空")
        return self._delete_node(self._tail.prev)

    def delete_at(self, index: int) -> T:
        """删除指定位置 - O(n)"""
        node = self._get_node(index)
        return self._delete_node(node)

    # ==================== 访问操作 ====================

    def _get_node(self, index: int) -> DoublyListNode:
        """获取节点（优化：从较近的端开始）"""
        if index < 0 or index >= self._size:
            raise IndexError(f"索引 {index} 超出范围")

        # 从较近的一端开始遍历
        if index < self._size // 2:
            current = self._head.next
            for _ in range(index):
                current = current.next
        else:
            current = self._tail.prev
            for _ in range(self._size - 1 - index):
                current = current.prev

        return current

    def get(self, index: int) -> T:
        return self._get_node(index).val

    def __getitem__(self, index: int) -> T:
        return self.get(index)

    def first(self) -> T:
        if self.is_empty():
            raise IndexError("链表为空")
        return self._head.next.val

    def last(self) -> T:
        if self.is_empty():
            raise IndexError("链表为空")
        return self._tail.prev.val

    # ==================== 其他操作 ====================

    def to_list(self) -> List[T]:
        return list(self)

    def __iter__(self) -> Iterator[T]:
        current = self._head.next
        while current != self._tail:
            yield current.val
            current = current.next

    def __reversed__(self) -> Iterator[T]:
        current = self._tail.prev
        while current != self._head:
            yield current.val
            current = current.prev

    def __repr__(self) -> str:
        if self.is_empty():
            return "DoublyLinkedList()"
        values = " <-> ".join(str(val) for val in self)
        return f"DoublyLinkedList({values})"


# ============================================================
#                    常见链表算法
# ============================================================

def reverse_list(head: ListNode) -> ListNode:
    """
    反转链表（迭代法）

    时间复杂度: O(n)
    空间复杂度: O(1)
    """
    prev = None
    current = head

    while current:
        next_node = current.next  # 保存下一个节点
        current.next = prev       # 反转指针
        prev = current            # 移动 prev
        current = next_node       # 移动 current

    return prev


def reverse_list_recursive(head: ListNode) -> ListNode:
    """
    反转链表（递归法）

    时间复杂度: O(n)
    空间复杂度: O(n) - 递归栈
    """
    # 基本情况
    if not head or not head.next:
        return head

    # 递归反转后面的部分
    new_head = reverse_list_recursive(head.next)

    # 处理当前节点
    head.next.next = head
    head.next = None

    return new_head


def find_middle(head: ListNode) -> ListNode:
    """
    找链表中点（快慢指针）

    时间复杂度: O(n)
    空间复杂度: O(1)

    奇数个节点返回正中间
    偶数个节点返回中间偏左
    """
    slow = fast = head

    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    return slow


def has_cycle(head: ListNode) -> bool:
    """
    检测链表是否有环（快慢指针）

    时间复杂度: O(n)
    空间复杂度: O(1)
    """
    if not head or not head.next:
        return False

    slow = head
    fast = head.next

    while slow != fast:
        if not fast or not fast.next:
            return False
        slow = slow.next
        fast = fast.next.next

    return True


def find_cycle_start(head: ListNode) -> Optional[ListNode]:
    """
    找环的起点

    原理：
    - 设链表头到环入口距离为 a
    - 环入口到相遇点距离为 b
    - 相遇点到环入口距离为 c
    - 快指针走的距离是慢指针的两倍
    - 2(a + b) = a + b + n(b + c)
    - a = c + (n-1)(b + c)
    - 所以从头和相遇点同时出发，会在入口相遇
    """
    if not head or not head.next:
        return None

    slow = fast = head

    # 检测是否有环
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # 无环

    # 找环入口
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next

    return slow


def merge_two_lists(l1: ListNode, l2: ListNode) -> ListNode:
    """
    合并两个有序链表

    时间复杂度: O(m + n)
    空间复杂度: O(1)
    """
    dummy = ListNode(0)
    current = dummy

    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next

    # 连接剩余部分
    current.next = l1 if l1 else l2

    return dummy.next


def remove_nth_from_end(head: ListNode, n: int) -> ListNode:
    """
    删除链表的倒数第 N 个节点

    方法：快慢指针，快指针先走 n 步

    时间复杂度: O(n)
    空间复杂度: O(1)
    """
    dummy = ListNode(0)
    dummy.next = head
    fast = slow = dummy

    # 快指针先走 n+1 步
    for _ in range(n + 1):
        fast = fast.next

    # 同时移动
    while fast:
        fast = fast.next
        slow = slow.next

    # 删除节点
    slow.next = slow.next.next

    return dummy.next


def is_palindrome(head: ListNode) -> bool:
    """
    判断链表是否是回文

    方法：快慢指针找中点 + 反转后半部分

    时间复杂度: O(n)
    空间复杂度: O(1)
    """
    if not head or not head.next:
        return True

    # 找中点
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    # 反转后半部分
    second_half = reverse_list(slow.next)

    # 比较
    first_half = head
    result = True
    while second_half:
        if first_half.val != second_half.val:
            result = False
            break
        first_half = first_half.next
        second_half = second_half.next

    # 恢复链表（可选）
    slow.next = reverse_list(slow.next)

    return result


def get_intersection(headA: ListNode, headB: ListNode) -> Optional[ListNode]:
    """
    找两个链表的交点

    方法：双指针同步
    - 指针 A 遍历完 A 链表后遍历 B 链表
    - 指针 B 遍历完 B 链表后遍历 A 链表
    - 如果有交点，会在交点相遇
    - 否则会同时到达 None

    时间复杂度: O(m + n)
    空间复杂度: O(1)
    """
    if not headA or not headB:
        return None

    pA, pB = headA, headB

    while pA != pB:
        pA = pA.next if pA else headB
        pB = pB.next if pB else headA

    return pA


# ============================================================
#                    辅助函数
# ============================================================

def create_linked_list(values: List) -> Optional[ListNode]:
    """从列表创建链表"""
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    return head


def print_linked_list(head: ListNode) -> str:
    """打印链表"""
    values = []
    while head:
        values.append(str(head.val))
        head = head.next
    return " -> ".join(values) + " -> None"


# ============================================================
#                    测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("单向链表测试")
    print("=" * 60)

    sll = SinglyLinkedList()

    print("\n--- 添加元素 ---")
    for i in range(1, 6):
        sll.append(i)
    print(f"append 1-5: {sll}")

    sll.prepend(0)
    print(f"prepend 0: {sll}")

    sll.insert(3, 99)
    print(f"insert(3, 99): {sll}")

    print("\n--- 访问元素 ---")
    print(f"first: {sll.first()}, last: {sll.last()}")
    print(f"get(3): {sll.get(3)}")

    print("\n--- 删除元素 ---")
    print(f"pop_first: {sll.pop_first()}, 链表: {sll}")
    print(f"pop_last: {sll.pop_last()}, 链表: {sll}")
    print(f"remove(99): {sll.remove(99)}, 链表: {sll}")

    print("\n--- 反转 ---")
    sll.reverse()
    print(f"反转后: {sll}")

    print("\n" + "=" * 60)
    print("双向链表测试")
    print("=" * 60)

    dll = DoublyLinkedList()

    print("\n--- 添加元素 ---")
    for i in range(1, 6):
        dll.append(i)
    print(f"append 1-5: {dll}")

    dll.prepend(0)
    print(f"prepend 0: {dll}")

    print("\n--- 双向遍历 ---")
    print(f"正向: {list(dll)}")
    print(f"反向: {list(reversed(dll))}")

    print("\n--- 删除 ---")
    print(f"pop_first: {dll.pop_first()}")
    print(f"pop_last: {dll.pop_last()}")
    print(f"当前: {dll}")

    print("\n" + "=" * 60)
    print("链表算法测试")
    print("=" * 60)

    print("\n--- 反转链表 ---")
    head = create_linked_list([1, 2, 3, 4, 5])
    print(f"原链表: {print_linked_list(head)}")
    head = reverse_list(head)
    print(f"反转后: {print_linked_list(head)}")

    print("\n--- 找中点 ---")
    head = create_linked_list([1, 2, 3, 4, 5])
    mid = find_middle(head)
    print(f"链表: {print_linked_list(head)}")
    print(f"中点: {mid.val}")

    print("\n--- 合并有序链表 ---")
    l1 = create_linked_list([1, 3, 5])
    l2 = create_linked_list([2, 4, 6])
    print(f"l1: {print_linked_list(l1)}")
    print(f"l2: {print_linked_list(l2)}")
    merged = merge_two_lists(l1, l2)
    print(f"合并: {print_linked_list(merged)}")

    print("\n--- 回文判断 ---")
    head1 = create_linked_list([1, 2, 3, 2, 1])
    head2 = create_linked_list([1, 2, 3, 4, 5])
    print(f"{print_linked_list(head1)} 是回文: {is_palindrome(head1)}")
    print(f"{print_linked_list(head2)} 是回文: {is_palindrome(head2)}")

    print("\n--- 删除倒数第N个 ---")
    head = create_linked_list([1, 2, 3, 4, 5])
    print(f"原链表: {print_linked_list(head)}")
    head = remove_nth_from_end(head, 2)
    print(f"删除倒数第2个: {print_linked_list(head)}")
