# 栈和队列 (Stack & Queue)

## 栈 (Stack)

### 概念

栈是一种**后进先出 (LIFO, Last In First Out)** 的线性数据结构。

```
栈的操作：

    Push(3)    Push(5)    Push(7)     Pop()
    ───────    ───────    ───────    ───────
    │   │      │   │      │ 7 │ ←   │   │
    │   │      │ 5 │      │ 5 │      │ 5 │ ← top
    │ 3 │ ← top│ 3 │      │ 3 │      │ 3 │
    └───┘      └───┘      └───┘      └───┘

类比：一摞盘子、浏览器后退、撤销操作
```

### 核心操作

| 操作 | 说明 | 时间复杂度 |
|------|------|-----------|
| push(x) | 入栈 | O(1) |
| pop() | 出栈 | O(1) |
| peek/top() | 查看栈顶 | O(1) |
| isEmpty() | 是否为空 | O(1) |
| size() | 元素数量 | O(1) |

### 实现方式

```
1. 数组实现：
┌───┬───┬───┬───┬───┬───┐
│ 3 │ 5 │ 7 │   │   │   │
└───┴───┴───┴───┴───┴───┘
              ↑
            top = 2

2. 链表实现：
    head
     ↓
┌───┐   ┌───┐   ┌───┐
│ 7 │──>│ 5 │──>│ 3 │──> None
└───┘   └───┘   └───┘
  ↑
 top
```

### 适用场景

- ✅ 函数调用栈（递归）
- ✅ 表达式求值（中缀转后缀）
- ✅ 括号匹配
- ✅ 浏览器前进后退
- ✅ 撤销/重做功能
- ✅ DFS 深度优先搜索

---

## 队列 (Queue)

### 概念

队列是一种**先进先出 (FIFO, First In First Out)** 的线性数据结构。

```
队列的操作：

Enqueue(入队)                    Dequeue(出队)
    ↓                                ↓
┌───┬───┬───┬───┬───┐         ┌───┬───┬───┬───┬───┐
│   │ 7 │ 5 │ 3 │   │   =>    │   │ 7 │ 5 │   │   │
└───┴───┴───┴───┴───┘         └───┴───┴───┴───┴───┘
        ↑       ↑                     ↑       ↑
       rear   front                  rear   front

类比：排队、打印任务队列、消息队列
```

### 核心操作

| 操作 | 说明 | 时间复杂度 |
|------|------|-----------|
| enqueue(x) | 入队 | O(1) |
| dequeue() | 出队 | O(1) |
| front/peek() | 查看队首 | O(1) |
| isEmpty() | 是否为空 | O(1) |
| size() | 元素数量 | O(1) |

---

## 队列变体

### 1. 循环队列 (Circular Queue)

```
解决数组实现队列的"假溢出"问题

普通数组队列的问题：
┌───┬───┬───┬───┬───┐
│   │   │ 5 │ 3 │ 1 │  头部空间浪费
└───┴───┴───┴───┴───┘
        ↑           ↑
      front       rear

循环队列：
        rear
          ↓
    ┌───┬───┐
   7│   │   │6
    ├───┼───┤
   3│   │   │5
    └───┴───┘
          ↑
        front

公式：
- 入队：rear = (rear + 1) % capacity
- 出队：front = (front + 1) % capacity
- 判满：(rear + 1) % capacity == front
- 判空：front == rear
```

### 2. 双端队列 (Deque)

```
两端都可以进行插入和删除操作

┌─────────────────────────────────┐
│                                 │
↓         ↓               ↓       ↓
addFirst  removeFirst     removeLast  addLast
          ┌───┬───┬───┬───┐
          │ 1 │ 2 │ 3 │ 4 │
          └───┴───┴───┴───┘
          front         rear

应用：滑动窗口最大值
```

### 3. 优先队列 (Priority Queue)

```
元素按优先级出队，通常用堆实现

入队：[3, 1, 4, 1, 5, 9]
出队顺序：1, 1, 3, 4, 5, 9 （最小堆）

时间复杂度：
- 入队：O(log n)
- 出队：O(log n)
- 查看最值：O(1)

应用：任务调度、Dijkstra 算法、哈夫曼编码
```

---

## 单调栈 (Monotonic Stack)

### 概念

栈内元素保持单调递增或递减。

```
单调递减栈示例：
入栈序列：[3, 1, 4, 1, 5, 9, 2, 6]

操作过程：
3 入栈: [3]
1 入栈: [3, 1]
4 入栈: 1,3 出栈，[4]
1 入栈: [4, 1]
5 入栈: 1 出栈，[4, 5]
9 入栈: 4,5 出栈，[9]
2 入栈: [9, 2]
6 入栈: 2 出栈，[9, 6]
```

### 应用

- 下一个更大元素
- 每日温度
- 柱状图中最大矩形
- 接雨水

---

## 单调队列 (Monotonic Queue)

### 概念

队列内元素保持单调，用于滑动窗口问题。

```
滑动窗口最大值：
数组: [1, 3, -1, -3, 5, 3, 6, 7]
窗口大小: 3

窗口位置                最大值
[1  3  -1] -3  5  3  6  7      3
 1 [3  -1  -3] 5  3  6  7      3
 1  3 [-1  -3  5] 3  6  7      5
 1  3  -1 [-3  5  3] 6  7      5
 1  3  -1  -3 [5  3  6] 7      6
 1  3  -1  -3  5 [3  6  7]     7

使用单调递减队列，队首始终是窗口最大值
```

---

## 栈和队列的互相实现

### 用栈实现队列

```
使用两个栈：入栈 inStack，出栈 outStack

入队：push 到 inStack
出队：
  1. 如果 outStack 为空，将 inStack 全部倒入 outStack
  2. 从 outStack pop

均摊时间复杂度：O(1)
```

### 用队列实现栈

```
使用两个队列，或一个队列

方法1：两个队列
- push：入队到非空队列
- pop：将 n-1 个元素移到另一队列，取最后一个

方法2：一个队列
- push：入队后，将前 n-1 个元素依次出队再入队
```

---

## 时间复杂度总结

| 数据结构 | push/enqueue | pop/dequeue | peek | 底层实现 |
|----------|--------------|-------------|------|----------|
| 栈 | O(1) | O(1) | O(1) | 数组/链表 |
| 队列 | O(1) | O(1) | O(1) | 数组/链表 |
| 循环队列 | O(1) | O(1) | O(1) | 数组 |
| 双端队列 | O(1) | O(1) | O(1) | 数组/链表 |
| 优先队列 | O(log n) | O(log n) | O(1) | 堆 |

---

## 相关算法题

| 难度 | 题目 | 核心技巧 |
|------|------|----------|
| 简单 | 有效的括号 | 栈匹配 |
| 简单 | 用栈实现队列 | 双栈 |
| 简单 | 用队列实现栈 | 单/双队列 |
| 中等 | 最小栈 | 辅助栈 |
| 中等 | 每日温度 | 单调栈 |
| 中等 | 下一个更大元素 | 单调栈 |
| 困难 | 柱状图中最大的矩形 | 单调栈 |
| 困难 | 滑动窗口最大值 | 单调队列 |
| 困难 | 接雨水 | 单调栈/双指针 |

---

## 多语言对照

### 栈

```python
# Python - 使用 list
stack = []
stack.append(1)    # push
stack.pop()        # pop
stack[-1]          # peek
len(stack) == 0    # isEmpty
```

```go
// Go - 使用 slice
var stack []int
stack = append(stack, 1)           // push
stack = stack[:len(stack)-1]       // pop
stack[len(stack)-1]                // peek
```

```java
// Java - 使用 Deque
Deque<Integer> stack = new ArrayDeque<>();
stack.push(1);     // push
stack.pop();       // pop
stack.peek();      // peek
```

### 队列

```python
# Python - 使用 collections.deque
from collections import deque
queue = deque()
queue.append(1)     # enqueue
queue.popleft()     # dequeue
queue[0]            # front
```

```go
// Go - 使用 slice (简单实现)
var queue []int
queue = append(queue, 1)    // enqueue
queue = queue[1:]           // dequeue (效率低)
```

```java
// Java - 使用 Deque
Deque<Integer> queue = new ArrayDeque<>();
queue.offer(1);    // enqueue
queue.poll();      // dequeue
queue.peek();      // front
```
