# 内存管理

## 课程概述

本教程深入讲解操作系统的内存管理机制，包括虚拟内存、分页分段、页面置换算法、内存分配等核心技术。

**学习目标**：
- 理解虚拟内存的原理和作用
- 掌握分页与分段机制
- 熟悉各种页面置换算法
- 理解TLB和多级页表
- 掌握内存分配算法
- 了解Swap和OOM机制

---

## 1. 虚拟内存原理

### 1.1 为什么需要虚拟内存

```
┌─────────────────────────────────────────────────────────────┐
│              物理内存 vs 虚拟内存                              │
└─────────────────────────────────────────────────────────────┘

物理内存直接访问的问题：
┌─────────────────────────┐
│  进程A  │  进程B  │  OS │  物理内存 (有限)
└─────────────────────────┘
问题:
• 内存碎片
• 地址冲突
• 无法隔离
• 容量受限

虚拟内存解决方案：
    进程A地址空间           进程B地址空间
   ┌──────────────┐        ┌──────────────┐
   │ 0x00000000   │        │ 0x00000000   │
   │     ...      │        │     ...      │
   │ 0xFFFFFFFF   │        │ 0xFFFFFFFF   │
   └──────┬───────┘        └──────┬───────┘
          │                       │
          │   MMU (内存管理单元)   │
          │   ┌─────────────┐    │
          └───┤  页表映射    ├────┘
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │  物理内存    │
              │ ┌──────────┐│
              │ │  页框1   ││
              │ │  页框2   ││
              │ │   ...    ││
              │ └──────────┘│
              └─────────────┘

虚拟内存的优势：
1. 地址空间隔离 - 每个进程独立地址空间
2. 内存保护 - 进程间互不干扰
3. 容量扩展 - 使用磁盘扩展内存
4. 内存共享 - 多进程共享物理页
```

### 1.2 地址转换机制

```c
// 虚拟地址到物理地址转换示例
#include <stdio.h>
#include <stdint.h>

// 假设: 32位地址, 4KB页大小
#define PAGE_SIZE 4096
#define PAGE_BITS 12  // log2(4096)

typedef struct {
    uint32_t frame_number : 20;  // 物理页框号
    uint32_t present : 1;        // 是否在内存
    uint32_t writable : 1;       // 是否可写
    uint32_t user : 1;           // 用户模式可访问
    uint32_t accessed : 1;       // 是否被访问过
    uint32_t dirty : 1;          // 是否被修改过
    uint32_t reserved : 7;
} page_table_entry_t;

// 虚拟地址结构
typedef struct {
    uint32_t offset : 12;        // 页内偏移 (0-4095)
    uint32_t page_number : 20;   // 虚拟页号
} virtual_address_t;

// 物理地址结构
typedef struct {
    uint32_t offset : 12;        // 页内偏移
    uint32_t frame_number : 20;  // 物理页框号
} physical_address_t;

// 地址转换函数
physical_address_t translate_address(
    virtual_address_t va,
    page_table_entry_t *page_table
) {
    physical_address_t pa;
    page_table_entry_t pte = page_table[va.page_number];

    if (!pte.present) {
        printf("缺页异常! 虚拟页 %u 不在内存中\n", va.page_number);
        // 触发缺页中断，从磁盘加载页面
        return (physical_address_t){0, 0};
    }

    pa.frame_number = pte.frame_number;
    pa.offset = va.offset;

    printf("虚拟地址转换:\n");
    printf("  虚拟页号: %u, 偏移: %u\n", va.page_number, va.offset);
    printf("  物理页框: %u, 偏移: %u\n", pa.frame_number, pa.offset);

    return pa;
}

int main() {
    // 模拟页表（简化版）
    page_table_entry_t page_table[1024] = {0};

    // 设置一些映射
    page_table[0] = (page_table_entry_t){.frame_number = 5, .present = 1};
    page_table[1] = (page_table_entry_t){.frame_number = 10, .present = 1};
    page_table[2] = (page_table_entry_t){.frame_number = 15, .present = 0};

    // 测试地址转换
    virtual_address_t va1 = {.page_number = 0, .offset = 100};
    translate_address(va1, page_table);

    virtual_address_t va2 = {.page_number = 2, .offset = 200};
    translate_address(va2, page_table);  // 会触发缺页

    return 0;
}
```

## 2. 分页与分段

### 2.1 分页机制

```
┌─────────────────────────────────────────────────────────────┐
│                    分页内存管理                               │
└─────────────────────────────────────────────────────────────┘

虚拟地址空间              页表               物理内存
┌────────────┐         ┌──────┐          ┌────────────┐
│   页 0     │───────▶│ 页框5 │────────▶│   页框0    │
├────────────┤         ├──────┤          ├────────────┤
│   页 1     │───────▶│ 页框2 │          │   页框1    │
├────────────┤         ├──────┤          ├────────────┤
│   页 2     │───────▶│ 页框7 │────────▶│   页框2    │
├────────────┤         ├──────┤          ├────────────┤
│   页 3     │         │  缺页 │          │   页框3    │
├────────────┤         ├──────┤          ├────────────┤
│   ...      │         │ ...  │          │   ...      │
└────────────┘         └──────┘          └────────────┘

页表项结构 (x86-64):
┌──────────────────────────────────────────────────────┐
│ 物理页框号 (52位) │ 保留 │ 标志位 (12位)             │
└──────────────────────────────────────────────────────┘
                           │
                           ├─ P: Present (是否在内存)
                           ├─ R/W: Read/Write (读写权限)
                           ├─ U/S: User/Supervisor (用户/内核)
                           ├─ A: Accessed (是否被访问)
                           └─ D: Dirty (是否被修改)
```

### 2.2 多级页表

```
┌─────────────────────────────────────────────────────────────┐
│              四级页表 (x86-64)                                │
└─────────────────────────────────────────────────────────────┘

64位虚拟地址结构:
┌──────┬──────┬──────┬──────┬──────┬────────────┐
│ 未用 │ PML4 │ PDPT │  PD  │  PT  │   Offset   │
│ 16位 │ 9位  │ 9位  │ 9位  │ 9位  │   12位     │
└──────┴──────┴──────┴──────┴──────┴────────────┘

转换过程:
CR3寄存器 ──┐
            │
            ▼
      ┌──────────┐
      │ PML4表   │  ◀── 使用PML4索引
      └────┬─────┘
           │
           ▼
      ┌──────────┐
      │ PDPT表   │  ◀── 使用PDPT索引
      └────┬─────┘
           │
           ▼
      ┌──────────┐
      │  PD表    │  ◀── 使用PD索引
      └────┬─────┘
           │
           ▼
      ┌──────────┐
      │  PT表    │  ◀── 使用PT索引
      └────┬─────┘
           │
           ▼
      物理页框号 + 页内偏移 = 物理地址

优势:
• 节省内存 (稀疏地址空间不需要完整页表)
• 支持大地址空间 (理论上2^48字节 = 256TB)
```

## 3. 页面置换算法

### 3.1 常见置换算法对比

```
┌─────────────────────────────────────────────────────────────┐
│              页面置换算法比较                                 │
└─────────────────────────────────────────────────────────────┘

访问序列: 1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5
物理页框数: 3

1. FIFO (先进先出):
   步骤    内存状态      缺页?
   1      [1, -, -]      ✓
   2      [1, 2, -]      ✓
   3      [1, 2, 3]      ✓
   4      [4, 2, 3]      ✓  (淘汰1)
   1      [4, 1, 3]      ✓  (淘汰2)
   2      [4, 1, 2]      ✓  (淘汰3)
   5      [5, 1, 2]      ✓  (淘汰4)
   ...
   缺页次数: 9

2. LRU (最近最少使用):
   步骤    内存状态      缺页?
   1      [1, -, -]      ✓
   2      [1, 2, -]      ✓
   3      [1, 2, 3]      ✓
   4      [4, 2, 3]      ✓  (淘汰1)
   1      [4, 2, 1]      ✓  (淘汰3)
   2      [4, 2, 1]      ✗  (命中)
   5      [5, 2, 1]      ✓  (淘汰4)
   ...
   缺页次数: 7

3. Clock (时钟/二次机会):
   使用访问位，给页面"第二次机会"
   缺页次数: 通常介于FIFO和LRU之间

4. OPT (最优置换):
   淘汰未来最长时间不使用的页面
   缺页次数: 6 (理论最优)
```

### 3.2 LRU算法实现

```python
# LRU页面置换算法实现
from collections import OrderedDict

class LRUCache:
    """LRU缓存实现 (可用于页面置换模拟)"""

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.page_faults = 0
        self.hits = 0

    def access(self, page: int) -> bool:
        """
        访问页面
        返回: True表示缺页, False表示命中
        """
        if page in self.cache:
            # 命中: 移动到末尾(最近使用)
            self.cache.move_to_end(page)
            self.hits += 1
            return False
        else:
            # 缺页
            self.page_faults += 1

            if len(self.cache) >= self.capacity:
                # 淘汰最久未使用的页面(第一个)
                victim = next(iter(self.cache))
                self.cache.pop(victim)
                print(f"  淘汰页面: {victim}")

            # 加载新页面
            self.cache[page] = True
            print(f"  加载页面: {page}")
            return True

    def get_stats(self):
        """获取统计信息"""
        total = self.page_faults + self.hits
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'page_faults': self.page_faults,
            'hits': self.hits,
            'hit_rate': hit_rate
        }

# 测试
if __name__ == "__main__":
    # 访问序列
    reference_string = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]

    # 创建LRU缓存(3个页框)
    lru = LRUCache(capacity=3)

    print("LRU页面置换模拟:")
    print("=" * 50)

    for i, page in enumerate(reference_string, 1):
        print(f"步骤 {i}: 访问页面 {page}")
        is_fault = lru.access(page)
        print(f"  当前内存: {list(lru.cache.keys())}")
        print(f"  缺页: {'是' if is_fault else '否'}")
        print()

    # 打印统计
    stats = lru.get_stats()
    print("=" * 50)
    print(f"缺页次数: {stats['page_faults']}")
    print(f"命中次数: {stats['hits']}")
    print(f"命中率: {stats['hit_rate']:.2%}")
```

## 4. TLB与多级页表

### 4.1 TLB (Translation Lookaside Buffer)

```
┌─────────────────────────────────────────────────────────────┐
│              TLB加速地址转换                                  │
└─────────────────────────────────────────────────────────────┘

没有TLB:
虚拟地址 ──▶ 页表(内存) ──▶ 物理地址 ──▶ 数据(内存)
           │1次内存访问│              │1次内存访问│
           └───────────┴──────────────────────────┘
           总共: 2次内存访问

有TLB:
虚拟地址 ──▶ TLB(缓存) ──命中──▶ 物理地址 ──▶ 数据
              │很快!│                  │1次内存访问│
              │                        │
              └─未命中──▶ 页表(内存) ───┘
                        总共: 1次内存访问(命中) 或 2次(未命中)

TLB结构:
┌────────────┬────────────┬──────────────┐
│  虚拟页号  │  物理页框  │   标志位     │
├────────────┼────────────┼──────────────┤
│   0x1234   │   0x5678   │ V=1, R/W=1  │
│   0x2345   │   0x6789   │ V=1, R/W=0  │
│   ...      │   ...      │   ...       │
└────────────┴────────────┴──────────────┘

TLB命中率影响:
• TLB命中率 99% + TLB访问 1ns + 内存访问 100ns
• 平均访问时间 = 0.99×(1+100) + 0.01×(1+100+100) = 102ns
• 相比无TLB的200ns，提升近2倍
```

### 4.2 查看TLB信息

```bash
# 查看TLB配置 (Linux)
cat /proc/cpuinfo | grep -E "TLB|cache"

# 查看页表统计
cat /proc/vmstat | grep -E "pgfault|pgmajfault"

# 使用perf分析TLB miss
perf stat -e dTLB-load-misses,dTLB-store-misses,iTLB-load-misses ./your_program
```

## 5. 内存分配算法

### 5.1 Buddy System (伙伴系统)

```
┌─────────────────────────────────────────────────────────────┐
│              Buddy System内存分配                             │
└─────────────────────────────────────────────────────────────┘

初始状态: 1MB空闲块
┌─────────────────────────────────────────────────────────┐
│                       1MB (空闲)                         │
└─────────────────────────────────────────────────────────┘

分配70KB:
1. 1MB分裂 → 512KB + 512KB
┌──────────────────────────┬──────────────────────────┐
│     512KB (空闲)         │     512KB (空闲)         │
└──────────────────────────┴──────────────────────────┘

2. 第一个512KB分裂 → 256KB + 256KB
┌─────────────┬─────────────┬──────────────────────────┐
│ 256KB (空闲)│ 256KB (空闲)│     512KB (空闲)         │
└─────────────┴─────────────┴──────────────────────────┘

3. 第一个256KB分裂 → 128KB + 128KB
┌──────┬──────┬─────────────┬──────────────────────────┐
│128KB │128KB │ 256KB (空闲)│     512KB (空闲)         │
│(分配)│(空闲)│             │                          │
└──────┴──────┴─────────────┴──────────────────────────┘

释放128KB时，与相邻的128KB合并:
┌─────────────┬─────────────┬──────────────────────────┐
│ 256KB (空闲)│ 256KB (空闲)│     512KB (空闲)         │
└─────────────┴─────────────┴──────────────────────────┘

继续合并:
┌──────────────────────────┬──────────────────────────┐
│     512KB (空闲)         │     512KB (空闲)         │
└──────────────────────────┴──────────────────────────┘

最终合并:
┌─────────────────────────────────────────────────────────┐
│                       1MB (空闲)                         │
└─────────────────────────────────────────────────────────┘

优点:
• 快速合并相邻块
• 减少外部碎片

缺点:
• 内部碎片 (分配64KB需要128KB块)
```

### 5.2 Slab Allocator

```
┌─────────────────────────────────────────────────────────────┐
│              Slab分配器 (Linux内核)                           │
└─────────────────────────────────────────────────────────────┘

分层结构:
Cache (per object type)
  │
  ├─▶ Slab (full)     ──▶ [●][●][●][●] 全部已分配
  │
  ├─▶ Slab (partial)  ──▶ [●][○][●][○] 部分已分配
  │
  └─▶ Slab (empty)    ──▶ [○][○][○][○] 全部空闲

示例: task_struct缓存
┌──────────────────────────────────────┐
│  task_struct Cache                   │
│  (进程描述符缓存)                     │
├──────────────────────────────────────┤
│  Slab 1: [task][task][task][task]   │
│  Slab 2: [task][free][task][free]   │
│  Slab 3: [free][free][free][free]   │
└──────────────────────────────────────┘

优点:
• 减少碎片
• 快速分配/释放
• 对象重用 (保持初始化状态)
```

## 6. Swap与OOM

### 6.1 Swap机制

```
┌─────────────────────────────────────────────────────────────┐
│              Swap交换空间                                     │
└─────────────────────────────────────────────────────────────┘

内存压力处理流程:
     物理内存
   ┌────────────┐
   │   活跃页   │
   ├────────────┤
   │   非活跃页 │  ◀─── 当内存不足时
   ├────────────┤       │
   │   缓存页   │       │ 回收算法
   └──────┬─────┘       │
          │             │
          │ Swap Out    │
          ▼             │
    ┌──────────┐        │
    │ 交换空间  │ ───────┘
    │ (磁盘)   │
    └──────────┘

页面回收优先级:
1. 清空页缓存 (page cache)
2. 回收匿名页 → Swap
3. 回收文件页 (干净页直接丢弃)
4. OOM Killer
```

### 6.2 OOM Killer

```bash
# 查看OOM配置
cat /proc/sys/vm/overcommit_memory
# 0: 启发式overcommit
# 1: 总是overcommit
# 2: 不overcommit

# 查看OOM分数
cat /proc/<pid>/oom_score

# 调整进程OOM优先级
echo -1000 > /proc/<pid>/oom_adj  # -1000到1000, 越小越不容易被杀
```

```python
# 监控内存并模拟OOM情况
import psutil
import time

def monitor_memory():
    """监控系统内存使用"""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()

    print(f"物理内存: {mem.percent}% 使用")
    print(f"  总量: {mem.total / (1024**3):.2f} GB")
    print(f"  已用: {mem.used / (1024**3):.2f} GB")
    print(f"  可用: {mem.available / (1024**3):.2f} GB")
    print(f"Swap: {swap.percent}% 使用")
    print(f"  总量: {swap.total / (1024**3):.2f} GB")

    if mem.percent > 90:
        print("⚠️  内存使用过高，可能触发OOM!")

    if swap.percent > 50:
        print("⚠️  Swap使用过高，性能严重下降!")

if __name__ == "__main__":
    while True:
        monitor_memory()
        print("-" * 50)
        time.sleep(5)
```

## 7. 实战: 内存性能分析

### 7.1 内存泄漏检测

```c
// 使用Valgrind检测内存泄漏
#include <stdlib.h>

void memory_leak_example() {
    int *array = malloc(100 * sizeof(int));
    // 忘记释放!
}

void correct_usage() {
    int *array = malloc(100 * sizeof(int));
    // ... 使用array ...
    free(array);  // 正确释放
}

int main() {
    memory_leak_example();
    correct_usage();
    return 0;
}

// 编译并检测:
// gcc -g -o test test.c
// valgrind --leak-check=full ./test
```

### 7.2 内存性能优化

```bash
# 查看进程内存映射
pmap -x <pid>

# 查看内存详细统计
cat /proc/meminfo

# 查看进程内存使用
ps aux --sort=-rss | head

# 使用perf分析缺页
perf record -e page-faults -p <pid>
perf report

# 查看NUMA节点内存
numastat

# 清理缓存 (测试用)
echo 3 > /proc/sys/vm/drop_caches
```

## 总结

内存管理是操作系统的核心功能之一：

**关键技术**：
1. 虚拟内存 - 地址空间隔离与扩展
2. 分页机制 - 灵活的内存映射
3. 多级页表 - 支持大地址空间
4. TLB - 加速地址转换
5. 页面置换 - LRU/Clock算法
6. 内存分配 - Buddy/Slab系统
7. Swap - 扩展物理内存
8. OOM - 内存不足保护

**性能优化**：
- 减少缺页中断
- 提高TLB命中率
- 合理配置Swap
- 预防内存泄漏
- NUMA亲和性优化

## 下一步

- 学习 [04_file_systems.md](04_file_systems.md) - 文件系统管理
- 实践: 编写内存分配器
- 深入: 阅读Linux内存管理源码

## 延伸阅读

- 《深入理解Linux内核》第8章 - 内存管理
- 《Operating Systems: Three Easy Pieces》- Virtual Memory
- Linux源码: mm/目录
