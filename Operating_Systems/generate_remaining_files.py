#!/usr/bin/env python3
"""
批量生成Operating_Systems模块的剩余教程文件
"""

import os

# 文件模板定义
files_content = {
    "03_memory_management.md": """# 内存管理

## 课程概述

本教程深入讲解操作系统的内存管理机制,从虚拟内存的基本原理到页面置换算法,从内存分配策略到内存保护机制,帮助你全面掌握内存管理的核心技术。

**学习目标**:
- 理解虚拟内存的工作原理和优势
- 掌握分页、分段和段页式内存管理
- 深入了解页面置换算法的设计与权衡
- 理解TLB和多级页表的优化机制
- 掌握Buddy、Slab等内存分配算法
- 学会内存性能分析与优化

---

## 1. 虚拟内存基础

### 1.1 虚拟内存概念

```
┌─────────────────────────────────────────────────────────────┐
│              虚拟内存系统架构                                 │
└─────────────────────────────────────────────────────────────┘

进程A                         进程B
虚拟地址空间                   虚拟地址空间
┌──────────────┐              ┌──────────────┐
│ 0xFFFFFFFF   │              │ 0xFFFFFFFF   │
│   内核空间   │              │   内核空间   │
├──────────────┤              ├──────────────┤
│              │              │              │
│   栈         │              │   栈         │
│   ↓          │              │   ↓          │
│              │              │              │
│              │              │              │
│   ↑          │              │   ↑          │
│   堆         │              │   堆         │
│              │              │              │
│   数据段     │              │   数据段     │
│   代码段     │              │   代码段     │
│ 0x00000000   │              │ 0x00000000   │
└──────┬───────┘              └──────┬───────┘
       │                             │
       └─────────┬───────────────────┘
                 │ MMU(内存管理单元)
                 │ 地址转换
                 ▼
        ┌─────────────────┐
        │   物理内存(RAM)  │
        │  ┌────────────┐  │
        │  │ 进程A页框   │  │
        │  ├────────────┤  │
        │  │ 内核        │  │
        │  ├────────────┤  │
        │  │ 进程B页框   │  │
        │  ├────────────┤  │
        │  │ 空闲        │  │
        │  └────────────┘  │
        └─────────────────┘

虚拟内存的优势:
✓ 进程隔离(每个进程独立地址空间)
✓ 内存保护(防止越界访问)
✓ 共享内存(多进程共享库)
✓ 需求分页(按需加载)
✓ 地址空间大于物理内存
```

### 1.2 地址转换机制

```
┌─────────────────────────────────────────────────────────────┐
│              虚拟地址到物理地址的转换                         │
└─────────────────────────────────────────────────────────────┘

32位虚拟地址 (分页系统):
┌────────────────┬──────────────────┐
│  页号(20位)     │  页内偏移(12位)   │
└───────┬────────┴─────────┬────────┘
        │                  │
        ▼                  │
   ┌────────────┐          │
   │  页表查询   │          │
   │  (Page     │          │
   │   Table)   │          │
   └────┬───────┘          │
        │                  │
        ▼                  │
   ┌────────────┐          │
   │ 页框号      │          │
   │ (20位)     │          │
   └────┬───────┘          │
        │                  │
        └──────────┬───────┘
                   │
                   ▼
            ┌──────────────────┐
            │ 物理地址(32位)    │
            │ 页框号 + 页内偏移 │
            └──────────────────┘

页表项(PTE - Page Table Entry)结构:
┌─────┬──────┬──────┬──────┬──────┬──────┬─────────┐
│有效位│修改位│访问位│保护位│缓存位│present│页框号    │
│  V  │  D  │  R  │  W  │  C  │   P  │   PFN   │
└─────┴──────┴──────┴──────┴──────┴──────┴─────────┘
 1bit  1bit   1bit   2bit   1bit   1bit    20bit

标志位说明:
- V (Valid): 页表项有效
- D (Dirty): 页面被修改过
- R (Reference): 页面被访问过
- W (Write): 写权限
- P (Present): 页面在内存中
```

---

## 2. 分页机制

### 2.1 单级页表

```c
/*
 * 页表模拟实现
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define PAGE_SIZE 4096        // 4KB页面
#define PAGE_ENTRIES 1024     // 页表项数
#define VIRTUAL_BITS 32       // 32位地址

// 页表项
typedef struct {
    uint32_t present : 1;     // 在内存中
    uint32_t writable : 1;    // 可写
    uint32_t user : 1;        // 用户态可访问
    uint32_t accessed : 1;    // 已访问
    uint32_t dirty : 1;       // 已修改
    uint32_t unused : 7;      // 未使用
    uint32_t frame : 20;      // 页框号(物理页号)
} page_table_entry;

// 地址转换
uint32_t translate_address(page_table_entry *page_table,
                           uint32_t virtual_addr) {
    // 提取页号和偏移
    uint32_t page_num = virtual_addr >> 12;     // 高20位
    uint32_t offset = virtual_addr & 0xFFF;     // 低12位

    // 查页表
    page_table_entry pte = page_table[page_num];

    if (!pte.present) {
        printf("缺页异常! 虚拟页: 0x%x\n", page_num);
        return 0xFFFFFFFF;  // 缺页
    }

    // 计算物理地址
    uint32_t physical_addr = (pte.frame << 12) | offset;

    printf("虚拟地址: 0x%08x -> 物理地址: 0x%08x\n",
           virtual_addr, physical_addr);

    return physical_addr;
}

int main() {
    // 模拟页表
    page_table_entry *page_table = calloc(PAGE_ENTRIES, sizeof(page_table_entry));

    // 设置几个页表项
    page_table[0].present = 1;
    page_table[0].frame = 10;    // 虚拟页0 -> 物理页10

    page_table[1].present = 1;
    page_table[1].frame = 20;    // 虚拟页1 -> 物理页20

    // 测试地址转换
    translate_address(page_table, 0x00000100);  // 第0页
    translate_address(page_table, 0x00001200);  // 第1页
    translate_address(page_table, 0x00002000);  // 第2页(未映射)

    free(page_table);
    return 0;
}
```

### 2.2 多级页表

```
┌─────────────────────────────────────────────────────────────┐
│              二级页表结构 (x86 32位)                          │
└─────────────────────────────────────────────────────────────┘

虚拟地址(32位):
┌────────────┬────────────┬──────────────┐
│ 页目录索引  │ 页表索引    │  页内偏移     │
│  (10位)    │  (10位)    │   (12位)     │
└─────┬──────┴──────┬─────┴──────┬───────┘
      │             │            │
      ▼             │            │
 ┌─────────────┐    │            │
 │ 页目录(PD)   │    │            │
 │ 1024项      │    │            │
 │ ┌─────────┐ │    │            │
 │ │ PDE[0]  │ │    │            │
 │ ├─────────┤ │    │            │
 │ │ PDE[1]──┼─┼────┘            │
 │ ├─────────┤ │                 │
 │ │ PDE[2]  │ │                 │
 │ └─────────┘ │                 │
 └─────────────┘                 │
        │                        │
        ▼                        │
 ┌─────────────┐                 │
 │ 页表(PT)     │                 │
 │ 1024项      │                 │
 │ ┌─────────┐ │                 │
 │ │ PTE[0]  │ │                 │
 │ ├─────────┤ │                 │
 │ │ PTE[1]──┼─┼─────────────────┘
 │ ├─────────┤ │
 │ │ PTE[2]  │ │
 │ └─────────┘ │
 └─────────────┘
        │
        ▼
   物理页框

x86-64 四级页表:
┌────┬────┬────┬────┬────────┐
│PML4│PDPT│ PD │ PT │ Offset │
│ 9位│ 9位│ 9位│ 9位│  12位  │
└────┴────┴────┴────┴────────┘

优势:
✓ 节省内存(稀疏地址空间)
✓ 支持大地址空间
✓ 按需分配
```

---

## 3. 页面置换算法

### 3.1 经典置换算法

```python
#!/usr/bin/env python3
"""
页面置换算法模拟
"""

def fifo_replacement(pages, frames):
    """先进先出(FIFO)算法"""
    memory = []
    page_faults = 0

    for page in pages:
        if page not in memory:
            page_faults += 1
            if len(memory) >= frames:
                memory.pop(0)  # 移除最早的页面
            memory.append(page)
            print(f"缺页: {page}, 内存: {memory}")
        else:
            print(f"命中: {page}, 内存: {memory}")

    return page_faults

def lru_replacement(pages, frames):
    """最近最少使用(LRU)算法"""
    memory = []
    page_faults = 0

    for page in pages:
        if page not in memory:
            page_faults += 1
            if len(memory) >= frames:
                memory.pop(0)  # 移除最久未使用的
            memory.append(page)
            print(f"缺页: {page}, 内存: {memory}")
        else:
            # 更新访问时间(移到末尾)
            memory.remove(page)
            memory.append(page)
            print(f"命中: {page}, 内存: {memory}")

    return page_faults

def optimal_replacement(pages, frames):
    """最优置换(OPT)算法"""
    memory = []
    page_faults = 0

    for i, page in enumerate(pages):
        if page not in memory:
            page_faults += 1
            if len(memory) >= frames:
                # 找到将来最晚使用的页面
                future_uses = {}
                for mem_page in memory:
                    try:
                        future_uses[mem_page] = pages[i+1:].index(mem_page)
                    except ValueError:
                        future_uses[mem_page] = float('inf')

                victim = max(future_uses, key=future_uses.get)
                memory.remove(victim)

            memory.append(page)
            print(f"缺页: {page}, 内存: {memory}")
        else:
            print(f"命中: {page}, 内存: {memory}")

    return page_faults

def clock_replacement(pages, frames):
    """时钟置换(Clock)算法"""
    memory = [None] * frames
    use_bit = [0] * frames
    pointer = 0
    page_faults = 0

    for page in pages:
        if page in memory:
            # 命中,设置使用位
            idx = memory.index(page)
            use_bit[idx] = 1
            print(f"命中: {page}, 内存: {[p for p in memory if p]}")
        else:
            page_faults += 1
            # 寻找替换页面
            while True:
                if memory[pointer] is None:
                    # 空位
                    memory[pointer] = page
                    use_bit[pointer] = 1
                    pointer = (pointer + 1) % frames
                    break
                elif use_bit[pointer] == 0:
                    # 使用位为0,替换
                    memory[pointer] = page
                    use_bit[pointer] = 1
                    pointer = (pointer + 1) % frames
                    break
                else:
                    # 使用位为1,置0并继续
                    use_bit[pointer] = 0
                    pointer = (pointer + 1) % frames

            print(f"缺页: {page}, 内存: {[p for p in memory if p]}, "
                  f"指针: {pointer}")

    return page_faults

# 测试
if __name__ == "__main__":
    pages = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    frames = 3

    print("=== FIFO算法 ===")
    faults = fifo_replacement(pages, frames)
    print(f"缺页次数: {faults}\n")

    print("=== LRU算法 ===")
    faults = lru_replacement(pages, frames)
    print(f"缺页次数: {faults}\n")

    print("=== OPT算法(理论最优) ===")
    faults = optimal_replacement(pages, frames)
    print(f"缺页次数: {faults}\n")

    print("=== Clock算法 ===")
    faults = clock_replacement(pages, frames)
    print(f"缺页次数: {faults}")
```

### 3.2 工作集模型

```
┌─────────────────────────────────────────────────────────────┐
│              工作集(Working Set)模型                          │
└─────────────────────────────────────────────────────────────┘

定义: 进程在时间窗口Δ内访问的页面集合

时间轴:
t=0     t=10    t=20    t=30    t=40    t=50
│───────│───────│───────│───────│───────│
访问序列: 1 2 3 1 4 2 5 1 2 3 6 ...

工作集计算(Δ=10):
时刻t=30, 向前看10个引用:
WS(30, 10) = {1, 2, 3, 6}  // 这4个页面

工作集大小变化:
│
4├─┐   ┌───┐
 │ │   │   │
3│ └───┘   └───┐
 │             │
2│             └───
 │
1└────────────────────▶ 时间

颠簸(Thrashing):
当分配的页框数 < 工作集大小时
→ 频繁缺页 → CPU利用率下降

CPU利用率
│     ┌─────────── 最优点
│    ╱ │
│   ╱  │
│  ╱   │  颠簸区域
│ ╱    │    ╲
│╱     │     ╲___
└──────┼──────────▶ 并发度
     最优值
```

---

## 4. TLB与缓存

### 4.1 TLB工作原理

```
┌─────────────────────────────────────────────────────────────┐
│       TLB (Translation Lookaside Buffer) 快表                │
└─────────────────────────────────────────────────────────────┘

地址转换流程:
虚拟地址
   │
   ▼
┌──────────┐
│  TLB查询  │  ← 并行查找,极快(~1 cycle)
└────┬─────┘
     │
  命中?├─────YES──→ 直接获得物理地址
     │
     NO
     │
     ▼
┌──────────┐
│ 页表查询  │  ← 慢(需访问内存)
└────┬─────┘
     │
     ▼
  更新TLB
     │
     ▼
  物理地址

TLB表项结构:
┌──────┬──────┬─────┬──────┬──────┐
│ 虚拟页│物理页│Valid│ Dirty│ ASID │
│  VPN │ PFN  │  V  │  D   │(进程)│
└──────┴──────┴─────┴──────┴──────┘

TLB类型:
1. 全相联TLB
   - 任意虚拟页可映射到任意TLB项
   - 查找快但硬件复杂

2. 组相联TLB
   - 折中方案
   - 现代处理器常用

性能影响:
有效访问时间 = TLB命中率 × TLB时间
             + TLB失效率 × (TLB时间 + 页表时间)

示例:
TLB命中率 = 98%
TLB时间 = 1ns
页表时间 = 100ns

EAT = 0.98 × 1 + 0.02 × (1 + 100)
    = 0.98 + 2.02
    = 3ns  (相比无TLB的101ns，加速33倍)
```

---

## 5. 内存分配算法

### 5.1 Buddy System

```c
/*
 * Buddy内存分配器实现
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define MAX_ORDER 11  // 最大2^11 = 2048页
#define PAGE_SIZE 4096

typedef struct block {
    struct block *next;
    int order;
} block_t;

// 空闲链表(每个order一个链表)
block_t *free_lists[MAX_ORDER];

// 初始化
void buddy_init(void *memory, size_t size) {
    int order = (int)log2(size / PAGE_SIZE);
    free_lists[order] = (block_t *)memory;
    free_lists[order]->next = NULL;
    free_lists[order]->order = order;
}

// 分配
void *buddy_alloc(int order) {
    // 找到足够大的块
    int current_order = order;
    while (current_order < MAX_ORDER && !free_lists[current_order]) {
        current_order++;
    }

    if (current_order >= MAX_ORDER) {
        return NULL;  // 内存不足
    }

    // 从空闲链表移除
    block_t *block = free_lists[current_order];
    free_lists[current_order] = block->next;

    // 分裂块
    while (current_order > order) {
        current_order--;
        size_t block_size = PAGE_SIZE * (1 << current_order);

        // 创建buddy块
        block_t *buddy = (block_t *)((char *)block + block_size);
        buddy->order = current_order;
        buddy->next = free_lists[current_order];
        free_lists[current_order] = buddy;
    }

    return block;
}

// 释放
void buddy_free(void *ptr, int order) {
    block_t *block = (block_t *)ptr;

    // 尝试合并buddy
    while (order < MAX_ORDER - 1) {
        size_t block_size = PAGE_SIZE * (1 << order);
        uintptr_t buddy_addr = (uintptr_t)block ^ block_size;

        // 查找buddy是否空闲
        block_t **prev = &free_lists[order];
        block_t *curr = free_lists[order];

        int found = 0;
        while (curr) {
            if ((uintptr_t)curr == buddy_addr) {
                // 找到buddy,合并
                *prev = curr->next;

                // 合并为更大的块
                if ((uintptr_t)block > buddy_addr) {
                    block = curr;
                }
                order++;
                found = 1;
                break;
            }
            prev = &curr->next;
            curr = curr->next;
        }

        if (!found) break;
    }

    // 插入空闲链表
    block->order = order;
    block->next = free_lists[order];
    free_lists[order] = block;
}
```

### 5.2 Slab分配器

```
┌─────────────────────────────────────────────────────────────┐
│              Slab分配器（Linux内核使用）                      │
└─────────────────────────────────────────────────────────────┘

三层结构:
┌─────────────────────────────────────────┐
│ Cache (针对特定对象类型)                 │
│ 例: task_struct cache, inode cache      │
└────────┬────────────────────────────────┘
         │
         ├── Slab (一个或多个连续页)
         │   ┌──────────────────────────┐
         │   │  完全空闲(Empty)          │
         │   │  [obj][obj][obj][obj]    │
         │   └──────────────────────────┘
         │
         ├── Slab
         │   ┌──────────────────────────┐
         │   │  部分使用(Partial)        │
         │   │  [obj][X][obj][X]        │
         │   └──────────────────────────┘
         │
         └── Slab
             ┌──────────────────────────┐
             │  完全使用(Full)           │
             │  [X][X][X][X]            │
             └──────────────────────────┘

优点:
✓ 减少碎片
✓ 缓存热对象
✓ 对象初始化开销小
✓ 硬件缓存友好

分配流程:
1. 从Partial链表获取slab
2. 如无Partial,从Empty获取
3. 如无Empty,分配新slab
4. 返回空闲对象

释放流程:
1. 标记对象为空闲
2. 如slab变为空,移到Empty
3. 如Empty过多,释放部分slab
```

---

## 6. Linux内存管理实战

```bash
#!/bin/bash
# Linux内存管理分析

echo "=== 系统内存信息 ==="
free -h
echo ""

echo "=== 详细内存统计 ==="
cat /proc/meminfo | head -20
echo ""

echo "=== 虚拟内存统计 ==="
vmstat 1 5
echo ""

echo "=== 内存映射 ==="
cat /proc/$$/maps | head -20
echo ""

echo "=== Slab信息 ==="
sudo slabtop -o | head -20
echo ""

echo "=== 内存压力 ==="
cat /proc/pressure/memory
echo ""

echo "=== OOM信息 ==="
dmesg | grep -i "out of memory" | tail -5
```

```python
#!/usr/bin/env python3
"""
内存分析工具
"""

import psutil
import os

def analyze_memory():
    # 虚拟内存
    vm = psutil.virtual_memory()
    print("=== 虚拟内存 ===")
    print(f"总量: {vm.total / 1024**3:.2f} GB")
    print(f"可用: {vm.available / 1024**3:.2f} GB")
    print(f"已用: {vm.used / 1024**3:.2f} GB ({vm.percent}%)")
    print(f"空闲: {vm.free / 1024**3:.2f} GB")
    print(f"缓冲: {vm.buffers / 1024**3:.2f} GB")
    print(f"缓存: {vm.cached / 1024**3:.2f} GB")

    # Swap
    swap = psutil.swap_memory()
    print(f"\n=== Swap ===")
    print(f"总量: {swap.total / 1024**3:.2f} GB")
    print(f"已用: {swap.used / 1024**3:.2f} GB ({swap.percent}%)")
    print(f"换入: {swap.sin / 1024**2:.2f} MB")
    print(f"换出: {swap.sout / 1024**2:.2f} MB")

    # 进程内存使用
    print(f"\n=== Top 5 内存占用进程 ===")
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            processes.append((
                proc.info['pid'],
                proc.info['name'],
                proc.info['memory_info'].rss
            ))
        except:
            pass

    processes.sort(key=lambda x: x[2], reverse=True)
    for pid, name, rss in processes[:5]:
        print(f"PID {pid:6} {name:20} {rss/1024**2:8.2f} MB")

if __name__ == "__main__":
    analyze_memory()
```

---

## 7. 关键概念总结

```
┌─────────────────────────────────────────────────────────────┐
│              内存管理核心概念                                 │
└─────────────────────────────────────────────────────────────┘

内存管理
├── 虚拟内存
│   ├── 地址转换(MMU)
│   ├── 进程隔离
│   ├── 内存保护
│   └── 共享内存
│
├── 分页机制
│   ├── 单级页表(简单但占用大)
│   ├── 多级页表(x86:2级, x64:4级)
│   ├── TLB加速(~98%命中率)
│   └── 大页支持(2MB/1GB)
│
├── 页面置换
│   ├── FIFO(简单,有Belady异常)
│   ├── LRU(最优近似,开销大)
│   ├── Clock(LRU近似,高效)
│   └── OPT(理论最优)
│
├── 内存分配
│   ├── Buddy(伙伴系统,2^n块)
│   ├── Slab(对象缓存)
│   └── SLUB(简化Slab)
│
└── 性能优化
    ├── 工作集模型
    ├── 预取(Prefetch)
    ├── 写时复制(COW)
    └── 内存压缩
```

---

**下一步**: 学习[文件系统](04_file_systems.md),理解文件组织与管理。

**相关模块**:
- [进程管理](02_process_management.md) - 内存与进程的关系
- [性能调优](09_performance_tuning.md) - 内存性能优化
- [Computer_Hardware](../Computer_Hardware/) - DRAM原理

**文件大小**: 约30KB
**最后更新**: 2024年
""",

    "04_file_systems.md": """# 文件系统

## 课程概述

本教程深入讲解文件系统的设计与实现,从文件系统的基本概念到ext4、NTFS等现代文件系统的内部结构,帮助你全面理解文件组织、存储管理和性能优化。

**学习目标**:
- 理解文件系统的层次结构
- 掌握inode、目录、数据块的组织方式
- 深入了解ext4文件系统的设计
- 理解日志文件系统的原理
- 掌握文件缓存与性能优化
- 对比主流文件系统的特点

---

## 1. 文件系统基础

### 1.1 文件系统层次结构

```
┌─────────────────────────────────────────────────────────────┐
│              文件系统层次结构                                 │
└─────────────────────────────────────────────────────────────┘

应用层
┌────────────────────────────────────┐
│   应用程序                          │
│   open/read/write/close            │
└───────────────┬────────────────────┘
                │
          系统调用接口
                │
┌───────────────▼────────────────────┐
│   VFS(Virtual File System)        │  统一接口层
│   - 通用文件操作                   │
│   - 文件描述符管理                 │
│   - 目录缓存(dcache)               │
│   - inode缓存(icache)              │
└───────────────┬────────────────────┘
                │
        ┌───────┼───────┬────────┐
        │       │       │        │
┌───────▼──┐ ┌──▼───┐ ┌▼─────┐ ┌▼──────┐
│  ext4    │ │NTFS  │ │ XFS  │ │  NFS  │  具体文件系统
│          │ │      │ │      │ │(网络) │
└───────┬──┘ └──┬───┘ └┬─────┘ └┬──────┘
        │       │      │        │
┌───────▼───────▼──────▼────────▼───────┐
│   块设备层(Block Layer)                │
│   - I/O调度                            │
│   - 请求合并                           │
└───────────────┬────────────────────────┘
                │
┌───────────────▼────────────────────────┐
│   设备驱动(Device Driver)              │
│   - SATA/NVMe/SCSI驱动                │
└───────────────┬────────────────────────┘
                │
            ┌───▼────┐
            │  硬盘  │
            └────────┘
```

### 1.2 文件系统基本概念

```
┌─────────────────────────────────────────────────────────────┐
│              文件与目录组织                                   │
└─────────────────────────────────────────────────────────────┘

文件(File):
- 命名的数据集合
- 元数据: 大小、权限、时间戳
- 数据块: 实际内容

目录(Directory):
- 特殊的文件
- 包含文件名到inode的映射
- 目录项(dentry)

目录树结构:
/
├── bin/           (二进制文件)
│   ├── ls
│   └── cat
├── etc/           (配置文件)
│   ├── passwd
│   └── fstab
├── home/          (用户目录)
│   └── user/
│       ├── documents/
│       └── downloads/
├── usr/           (用户程序)
│   ├── bin/
│   └── lib/
└── var/           (可变数据)
    ├── log/
    └── tmp/

文件类型:
- 普通文件 (-)
- 目录 (d)
- 符号链接 (l)
- 字符设备 (c)
- 块设备 (b)
- 管道 (p)
- 套接字 (s)
```

---

## 2. inode详解

### 2.1 inode结构

```
┌─────────────────────────────────────────────────────────────┐
│              inode (Index Node) 结构                         │
└─────────────────────────────────────────────────────────────┘

inode包含文件元数据,但不包含文件名:

┌──────────────────────────────────────┐
│ inode结构                             │
├──────────────────────────────────────┤
│ • inode号 (唯一标识)                  │
│ • 文件类型 (普通/目录/链接...)        │
│ • 权限 (rwxrwxrwx)                   │
│ • UID/GID (所有者)                   │
│ • 文件大小 (字节)                     │
│ • 时间戳:                             │
│   - atime (访问时间)                 │
│   - mtime (修改时间)                 │
│   - ctime (状态改变时间)             │
│ • 链接数 (硬链接计数)                │
│ • 数据块指针:                         │
│   - 12个直接指针                     │
│   - 1个一级间接指针                  │
│   - 1个二级间接指针                  │
│   - 1个三级间接指针                  │
└──────────────────────────────────────┘

数据块寻址(以4KB块为例):
┌─────────────────────────────────────┐
│ 直接指针 (12个)                      │
│ ├─→ 数据块0 (4KB)                   │
│ ├─→ 数据块1 (4KB)                   │
│ ...                                 │
│ └─→ 数据块11 (4KB)                  │
│ 总计: 48KB                          │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│ 一级间接指针                         │
│ └─→ 指针块 (1024个指针)             │
│     ├─→ 数据块                      │
│     ├─→ 数据块                      │
│     ...                             │
│ 总计: 4MB (1024 × 4KB)             │
└─────────────────────────────────────┘
                ↓
┌─────────────────────────────────────┐
│ 二级间接指针                         │
│ └─→ 指针块                          │
│     ├─→ 指针块 → 数据块...          │
│     ├─→ 指针块 → 数据块...          │
│ 总计: 4GB (1024² × 4KB)            │
└─────────────────────────────────────┘

最大文件大小:
12×4KB + 1024×4KB + 1024²×4KB + 1024³×4KB
= 48KB + 4MB + 4GB + 4TB
≈ 4TB
```

### 2.2 硬链接与符号链接

```
┌─────────────────────────────────────────────────────────────┐
│              硬链接 vs 符号链接                               │
└─────────────────────────────────────────────────────────────┘

硬链接(Hard Link):
目录项                 inode        数据块
┌────────┐          ┌──────┐     ┌─────────┐
│file1   │─────────→│inode │────→│ 数据    │
└────────┘          │  123 │     └─────────┘
┌────────┐          │links=2│
│file2   │─────────→│      │
└────────┘          └──────┘
(硬链接指向同一inode)

特点:
- 共享inode和数据
- 删除一个不影响另一个
- 只有最后一个被删除时数据才删除
- 不能跨文件系统
- 不能链接目录

符号链接(Symbolic Link/Soft Link):
目录项                 inode        数据块
┌────────┐          ┌──────┐     ┌─────────┐
│original│─────────→│inode │────→│ 数据    │
└────────┘          │  123 │     └─────────┘
                    └──────┘
┌────────┐          ┌──────┐     ┌─────────────┐
│symlink │─────────→│inode │────→│/path/to/    │
└────────┘          │  456 │     │original     │
                    └──────┘     └─────────────┘
(符号链接是特殊文件,存储目标路径)

特点:
- 独立的inode
- 可以跨文件系统
- 可以链接目录
- 原文件删除后,符号链接失效(dangling link)
```

---

## 3. ext4文件系统

### 3.1 ext4布局

```
┌─────────────────────────────────────────────────────────────┐
│              ext4文件系统布局                                 │
└─────────────────────────────────────────────────────────────┘

整个分区布局:
┌────────────┬──────────────────────────────────┐
│ 引导块     │  块组0  │  块组1  │ ... │ 块组N  │
│ (1024字节) │         │         │     │        │
└────────────┴──────────────────────────────────┘

每个块组(Block Group)结构:
┌─────────────────────────────────────────────┐
│ 超级块(Superblock)                          │
│ - 文件系统元信息                            │
│ - 块大小、inode数量、挂载次数等             │
│ (备份在多个块组中)                          │
├─────────────────────────────────────────────┤
│ 组描述符表(Group Descriptor Table)         │
│ - 各块组的统计信息                          │
│ - 空闲块数、inode数等                       │
├─────────────────────────────────────────────┤
│ 数据块位图(Data Block Bitmap)              │
│ - 标记数据块是否空闲(每位对应一个块)        │
│ - 1=已用, 0=空闲                           │
├─────────────────────────────────────────────┤
│ inode位图(inode Bitmap)                    │
│ - 标记inode是否空闲                         │
├─────────────────────────────────────────────┤
│ inode表(inode Table)                       │
│ ┌──────┬──────┬──────┬──────┐              │
│ │inode1│inode2│inode3│ ...  │              │
│ └──────┴──────┴──────┴──────┘              │
│ (每个inode: 256字节)                        │
├─────────────────────────────────────────────┤
│ 数据块(Data Blocks)                        │
│ ┌────┬────┬────┬────┬────┐                │
│ │DB1 │DB2 │DB3 │... │DBN │                │
│ └────┴────┴────┴────┴────┘                │
│ (实际文件数据)                              │
└─────────────────────────────────────────────┘

ext4新特性:
1. Extents (区段)
   替代间接块指针,提高大文件性能
   ┌─────────────────────────┐
   │ Extent Header           │
   ├─────────────────────────┤
   │ Extent 1: 逻辑块0-999   │
   │          物理块5000-5999│
   ├─────────────────────────┤
   │ Extent 2: 逻辑块1000-1999│
   │          物理块8000-8999│
   └─────────────────────────┘

2. 日志校验和(Journal Checksum)
3. 延迟分配(Delayed Allocation)
4. 多块分配(Multiblock Allocation)
5. 在线碎片整理
6. 快速fsck
7. 纳秒级时间戳
```

### 3.2 文件操作流程

```c
/*
 * 文件系统操作示例
 */

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

void demonstrate_file_operations() {
    printf("=== 文件操作流程 ===\n");

    // 1. 创建文件
    int fd = open("test.txt", O_CREAT | O_RDWR, 0644);
    printf("创建文件,文件描述符: %d\n", fd);

    // 2. 写入数据
    const char *data = "Hello, ext4!\n";
    ssize_t written = write(fd, data, 13);
    printf("写入 %zd 字节\n", written);

    // 3. 同步到磁盘
    fsync(fd);  // 确保数据写入磁盘

    // 4. 获取文件信息
    struct stat st;
    fstat(fd, &st);
    printf("文件大小: %ld 字节\n", st.st_size);
    printf("inode号: %ld\n", st.st_ino);
    printf("硬链接数: %ld\n", st.st_nlink);
    printf("块数: %ld (512字节块)\n", st.st_blocks);

    // 5. 创建硬链接
    link("test.txt", "test_hard.txt");
    printf("创建硬链接\n");

    fstat(fd, &st);
    printf("硬链接数变为: %ld\n", st.st_nlink);

    // 6. 创建符号链接
    symlink("test.txt", "test_sym.txt");
    printf("创建符号链接\n");

    // 7. 关闭文件
    close(fd);
}

void demonstrate_directory_operations() {
    printf("\n=== 目录操作 ===\n");

    // 创建目录
    mkdir("testdir", 0755);

    // 改变目录
    chdir("testdir");

    // 获取当前目录
    char cwd[256];
    getcwd(cwd, sizeof(cwd));
    printf("当前目录: %s\n", cwd);

    // 返回上级目录
    chdir("..");

    // 删除目录
    rmdir("testdir");
}

int main() {
    demonstrate_file_operations();
    demonstrate_directory_operations();

    // 清理
    unlink("test.txt");
    unlink("test_hard.txt");
    unlink("test_sym.txt");

    return 0;
}
```

---

## 4. 日志文件系统

### 4.1 日志原理

```
┌─────────────────────────────────────────────────────────────┐
│              日志文件系统(Journaling)                         │
└─────────────────────────────────────────────────────────────┘

问题: 系统崩溃时文件系统可能不一致
解决: 使用日志记录操作,崩溃后可以恢复

日志区域:
┌────────────────────────────────────┐
│ 日志超级块                          │
├────────────────────────────────────┤
│ 日志项1: 开始(Transaction Begin)   │
│ 日志项2: 元数据更新                │
│ 日志项3: 数据更新                  │
│ 日志项4: 提交(Transaction Commit)  │
├────────────────────────────────────┤
│ 日志项5: 开始                       │
│ ...                                │
└────────────────────────────────────┘

写入流程(Ordered模式):
1. 写数据到磁盘
   ┌─────────┐
   │ 数据块   │ ← 先写数据
   └─────────┘

2. 写元数据到日志
   ┌─────────┐
   │ 日志区   │ ← 记录元数据变化
   │ Begin   │
   │ Metadata│
   │ Commit  │
   └─────────┘

3. Checkpoint: 写元数据到磁盘
   ┌─────────┐
   │ 元数据   │ ← 更新inode/bitmap等
   └─────────┘

4. 释放日志空间

三种日志模式:
┌──────────┬──────────┬──────────┬─────────┐
│ 模式     │ 日志内容  │ 性能     │ 安全性  │
├──────────┼──────────┼──────────┼─────────┤
│ Journal  │ 数据+元数据│ 慢      │ 最高    │
│ Ordered  │ 仅元数据  │ 中      │ 高      │
│(默认)    │ 数据先写   │         │         │
│ Writeback│ 仅元数据  │ 快      │ 低      │
│          │ 无顺序保证│         │         │
└──────────┴──────────┴──────────┴─────────┘

恢复流程:
1. 扫描日志
2. 重放未完成的事务
3. 文件系统恢复到一致状态
```

---

## 5. 文件系统性能优化

### 5.1 缓存机制

```
┌─────────────────────────────────────────────────────────────┐
│              Linux文件系统缓存                                │
└─────────────────────────────────────────────────────────────┘

页缓存(Page Cache):
┌────────────────────────────────────────┐
│ 应用程序                                │
│ read(fd, buf, size)                    │
└───────────────┬────────────────────────┘
                │
                ▼
        ┌────────────────┐
        │  VFS层         │
        └───────┬────────┘
                │
                ▼
        ┌────────────────┐
        │ Page Cache     │  缓存最近访问的文件页
        │ ┌────┬────┬───┐│
        │ │Page│Page│...││
        │ └────┴────┴───┘│
        └───────┬────────┘
                │
          命中? ├──YES──→ 直接返回(快)
                │
                NO
                │
                ▼
        ┌────────────────┐
        │ 块设备层        │
        │ (读取磁盘)      │
        └────────────────┘

目录项缓存(Dentry Cache):
加速路径名查找
/usr/bin/ls → inode查找过程被缓存

inode缓存(Inode Cache):
缓存常用的inode结构

预读(Readahead):
顺序读时提前读取后续数据
┌────┬────┬────┬────┬────┐
│读取│预读│预读│预读│预读│
└────┴────┴────┴────┴────┘
  ↑
 应用请求

写回(Write Back):
延迟写入,批量刷新
- 减少磁盘操作
- 提高性能
- 风险: 崩溃可能丢数据
```

### 5.2 性能测试

```bash
#!/bin/bash
# 文件系统性能测试

echo "=== 顺序写测试 ==="
dd if=/dev/zero of=testfile bs=1M count=1024 conv=fdatasync
# 1GB文件,同步写入

echo -e "\n=== 顺序读测试 ==="
dd if=testfile of=/dev/null bs=1M

echo -e "\n=== 随机读写测试(fio) ==="
fio --name=randread --ioengine=libaio --iodepth=16 \
    --rw=randread --bs=4k --direct=1 \
    --size=1G --numjobs=4 --runtime=60 \
    --group_reporting

echo -e "\n=== 元数据操作测试 ==="
time for i in {1..10000}; do
    touch file_$i
done

time rm file_*

echo -e "\n=== fsync性能测试 ==="
python3 << 'EOF'
import time
import os

# 测试不同同步策略的性能
def test_no_sync():
    start = time.time()
    with open('test_nosync', 'w') as f:
        for i in range(10000):
            f.write(f"Line {i}\n")
    print(f"无同步: {time.time() - start:.3f}秒")
    os.remove('test_nosync')

def test_fsync():
    start = time.time()
    fd = os.open('test_fsync', os.O_CREAT | os.O_WRONLY)
    for i in range(10000):
        os.write(fd, f"Line {i}\n".encode())
    os.fsync(fd)  # 强制同步
    os.close(fd)
    print(f"fsync: {time.time() - start:.3f}秒")
    os.remove('test_fsync')

test_no_sync()
test_fsync()
EOF

rm -f testfile
```

---

## 6. 主流文件系统对比

```
┌─────────────────────────────────────────────────────────────┐
│              文件系统对比                                     │
└─────────────────────────────────────────────────────────────┘

┌──────────┬────────┬────────┬────────┬────────┐
│ 特性     │ ext4   │ XFS    │ Btrfs  │ NTFS   │
├──────────┼────────┼────────┼────────┼────────┤
│ 最大文件 │ 16TB   │ 8EB    │ 16EB   │ 16EB   │
│ 最大卷   │ 1EB    │ 8EB    │ 16EB   │ 256TB  │
│ 日志     │ 是     │ 是     │ COW    │ 是     │
│ 快照     │ 否     │ 否     │ 是     │ 否     │
│ 压缩     │ 否     │ 否     │ 是     │ 是     │
│ 碎片整理 │ 在线   │ 在线   │ 自动   │ 在线   │
│ 校验和   │ 元数据 │ 元数据 │ 全部   │ 无     │
│ 最佳场景 │ 通用   │ 大文件 │ 新系统 │ Windows│
└──────────┴────────┴────────┴────────┴────────┘

ext4:
+ 成熟稳定
+ 性能均衡
+ 广泛支持
- 缺少高级特性

XFS:
+ 大文件性能优秀
+ 并行I/O好
+ 可扩展性强
- 不支持缩小

Btrfs:
+ 快照/子卷
+ 数据校验
+ 写时复制
- 相对年轻

ZFS:
+ 企业级特性
+ 数据完整性
+ 快照/克隆
- Linux支持复杂
```

---

## 7. 实战工具

```bash
#!/bin/bash
# 文件系统管理工具

echo "=== 查看文件系统信息 ==="
df -Th

echo -e "\n=== inode使用情况 ==="
df -i

echo -e "\n=== 文件系统详细信息 ==="
dumpe2fs -h /dev/sda1 2>/dev/null | head -20

echo -e "\n=== 查看文件的物理块 ==="
echo "test file" > testfile
filefrag -v testfile
rm testfile

echo -e "\n=== 监控I/O ==="
iostat -x 1 5

echo -e "\n=== 查看打开文件 ==="
lsof | head -20

echo -e "\n=== 查看文件系统挂载选项 ==="
cat /proc/mounts | grep "^/dev"

echo -e "\n=== 文件系统性能调优参数 ==="
cat /sys/block/sda/queue/scheduler
cat /sys/block/sda/queue/read_ahead_kb
```

---

## 8. 关键概念总结

```
┌─────────────────────────────────────────────────────────────┐
│              文件系统核心概念                                 │
└─────────────────────────────────────────────────────────────┘

文件系统
├── 基本概念
│   ├── VFS(统一接口)
│   ├── inode(文件元数据)
│   ├── 目录(文件名映射)
│   └── 数据块(实际内容)
│
├── ext4架构
│   ├── 超级块(元信息)
│   ├── 块组(组织单位)
│   ├── Extents(连续块)
│   └── 日志(一致性)
│
├── 性能优化
│   ├── Page Cache(页缓存)
│   ├── Dentry Cache(目录缓存)
│   ├── Readahead(预读)
│   └── Write Back(延迟写)
│
└── 高级特性
    ├── 日志(崩溃恢复)
    ├── 快照(Btrfs/ZFS)
    ├── 校验和(数据完整性)
    └── 在线碎片整理
```

---

**下一步**: 学习[I/O管理](05_io_management.md),理解I/O子系统和调度算法。

**相关模块**:
- [存储设备](../Computer_Hardware/03_storage_devices.md) - 硬件基础
- [Linux](../Linux/) - 文件系统实战
- [性能调优](09_performance_tuning.md) - I/O性能优化

**文件大小**: 约28KB
**最后更新**: 2024年
"""
}

# 生成文件
def generate_files():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for filename, content in files_content.items():
        filepath = os.path.join(base_dir, filename)
        print(f"生成 {filename}...")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ {filename} 已创建")

    print(f"\n成功生成 {len(files_content)} 个文件!")

if __name__ == "__main__":
    generate_files()
