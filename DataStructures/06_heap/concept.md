# 堆 (Heap)

## 概念

堆是一种特殊的**完全二叉树**，满足**堆序性质**。

```
最大堆（Max Heap）：父节点 ≥ 子节点
最小堆（Min Heap）：父节点 ≤ 子节点

最大堆示例：
           90
          /  \
        85    80
       /  \   /
      70  60 50

最小堆示例：
           10
          /  \
        20    15
       /  \   /
      30  40 50
```

## 数组表示

```
完全二叉树可以用数组高效存储：

       90
      /  \
    85    80
   /  \   /
  70  60 50

数组: [90, 85, 80, 70, 60, 50]
索引:  0   1   2   3   4   5

索引关系（从 0 开始）：
- 父节点: (i - 1) / 2
- 左子节点: 2 * i + 1
- 右子节点: 2 * i + 2
```

## 核心操作

### 1. 上浮 (Sift Up / Bubble Up)

```
插入新元素后，将其上浮到正确位置

插入 95 到最大堆：
Step 1: 添加到末尾
       90
      /  \
    85    80
   /  \   /  \
  70  60 50  95

Step 2: 与父节点比较，95 > 80，交换
       90
      /  \
    85    95
   /  \   /  \
  70  60 50  80

Step 3: 95 > 90，交换
       95
      /  \
    85    90
   /  \   /  \
  70  60 50  80

时间复杂度: O(log n)
```

### 2. 下沉 (Sift Down / Heapify)

```
删除堆顶后，将末尾元素放到堆顶并下沉

删除最大值 95：
Step 1: 用末尾元素 80 替换堆顶
       80
      /  \
    85    90
   /  \   /
  70  60 50

Step 2: 与较大的子节点比较，80 < 90，交换
       90
      /  \
    85    80
   /  \   /
  70  60 50

时间复杂度: O(log n)
```

## 时间复杂度

| 操作 | 时间复杂度 |
|------|-----------|
| 获取最值 | O(1) |
| 插入 | O(log n) |
| 删除最值 | O(log n) |
| 建堆 | O(n) |
| 堆排序 | O(n log n) |

## 建堆

### 自顶向下（插入法）

```
依次插入每个元素，每次 O(log n)
总时间复杂度: O(n log n)
```

### 自底向上（Floyd 算法）

```
从最后一个非叶节点开始，依次下沉

数组: [4, 10, 3, 5, 1]

       4
      / \
    10   3
   /  \
  5    1

从索引 1 (值为10) 开始下沉...
       4
      / \
    10   3
   /  \
  5    1
  ↓
       4
      / \
    10   3
   /  \
  5    1

从索引 0 (值为4) 开始下沉...
       10
      /  \
     5    3
    / \
   4   1

时间复杂度: O(n)
```

## 应用场景

### 1. 优先队列

```
按优先级处理任务

tasks = [(3, "低优先级"), (1, "高优先级"), (2, "中优先级")]
最小堆按优先级出队: "高优先级" → "中优先级" → "低优先级"
```

### 2. Top-K 问题

```
找第 K 大的元素：
- 维护大小为 K 的最小堆
- 遍历数组，大于堆顶则替换
- 最终堆顶就是第 K 大

时间复杂度: O(n log k)
```

### 3. 堆排序

```
Step 1: 建堆 O(n)
Step 2: 依次取出堆顶 O(n log n)

特点：
- 原地排序，空间 O(1)
- 不稳定排序
- 时间 O(n log n)
```

### 4. 合并 K 个有序链表

```
使用最小堆维护 K 个链表的当前节点
每次取最小的加入结果链表

时间复杂度: O(n log k)，n 为总节点数
```

### 5. 中位数维护

```
使用两个堆：
- 最大堆存储较小的一半
- 最小堆存储较大的一半

中位数 = (最大堆顶 + 最小堆顶) / 2
```

## 相关算法题

| 难度 | 题目 | 核心技巧 |
|------|------|----------|
| 简单 | 最后一块石头的重量 | 最大堆 |
| 中等 | 数组中的第K个最大元素 | 最小堆/快速选择 |
| 中等 | 前K个高频元素 | 最小堆 |
| 中等 | 数据流的中位数 | 双堆 |
| 困难 | 合并K个升序链表 | 最小堆 |
| 困难 | 滑动窗口最大值 | 单调队列/堆 |

## 多语言实现

```python
# Python - heapq（最小堆）
import heapq

# 创建堆
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)

# 弹出最小值
min_val = heapq.heappop(heap)  # 1

# 最大堆（取负）
max_heap = []
heapq.heappush(max_heap, -3)
max_val = -heapq.heappop(max_heap)

# 建堆
arr = [4, 1, 3, 2]
heapq.heapify(arr)  # O(n)

# Top-K
top_k = heapq.nlargest(2, arr)
bottom_k = heapq.nsmallest(2, arr)
```

```go
// Go - container/heap
import "container/heap"

// 实现 heap.Interface
type IntHeap []int
func (h IntHeap) Len() int           { return len(h) }
func (h IntHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h IntHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *IntHeap) Push(x any)        { *h = append(*h, x.(int)) }
func (h *IntHeap) Pop() any {
    old := *h
    n := len(old)
    x := old[n-1]
    *h = old[:n-1]
    return x
}

// 使用
h := &IntHeap{3, 1, 2}
heap.Init(h)
heap.Push(h, 0)
min := heap.Pop(h).(int)
```

```java
// Java - PriorityQueue（最小堆）
PriorityQueue<Integer> minHeap = new PriorityQueue<>();
minHeap.offer(3);
minHeap.offer(1);
int min = minHeap.poll();  // 1

// 最大堆
PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
// 或
PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a, b) -> b - a);
```
