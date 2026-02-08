# 操作系统原理

深入理解操作系统的核心机制，掌握进程管理、内存管理、文件系统、并发控制等关键技术。

## 📚 学习路径

```
┌────────────────────────────────────────────────────┐
│          操作系统学习路线图                        │
├────────────────────────────────────────────────────┤
│                                                    │
│  基础 ────▶ OS概述 ────▶ 进程管理 ────▶ 内存管理 │
│            (架构/内核)  (调度/IPC)   (虚拟内存)   │
│                                                    │
│  核心 ────▶ 文件系统 ───▶ I/O管理 ───▶ 并发控制  │
│            (ext4/NTFS)   (设备驱动)  (锁/死锁)    │
│                                                    │
│  进阶 ────▶ 虚拟化 ─────▶ 安全 ──────▶ 性能调优  │
│            (KVM/Docker)  (隔离/沙箱) (系统分析)   │
└────────────────────────────────────────────────────┘
```

## 📖 目录

### 01. [OS概述](01_os_overview.md)
- 操作系统的演进历史
- OS的作用与功能
- 内核架构（宏内核/微内核/混合内核）
- 用户态与内核态
- 系统调用机制
- Linux/Windows/macOS对比

### 02. [进程管理](02_process_management.md)
- 进程与线程概念
- 进程状态转换（就绪/运行/阻塞）
- CPU调度算法（FIFO/SJF/RR/CFS）
- 进程间通信（管道/信号/共享内存/消息队列）
- 线程同步原语
- 协程与用户态调度

### 03. [内存管理](03_memory_management.md)
- 虚拟内存原理
- 分页与分段
- 页面置换算法（LRU/Clock/OPT）
- TLB与多级页表
- 内存分配算法（Buddy/Slab）
- Swap与OOM机制

### 04. [文件系统](04_file_systems.md)
- 文件系统层次结构
- inode与文件元数据
- ext4文件系统详解
- 日志文件系统（Journaling）
- B+树索引
- VFS虚拟文件系统
- NTFS/APFS/XFS对比

### 05. [I/O管理](05_io_management.md)
- I/O子系统架构
- I/O调度算法（CFQ/Deadline/Noop）
- 缓冲与缓存
- 设备驱动模型
- 异步I/O（AIO/io_uring）
- 零拷贝技术

### 06. [并发控制](06_concurrency.md)
- 临界区与竞态条件
- 互斥锁与信号量
- 读写锁与自旋锁
- 死锁检测与预防
- 无锁编程（CAS/原子操作）
- RCU机制

### 07. [虚拟化技术](07_virtualization.md)
- 虚拟化类型（全虚拟化/半虚拟化/硬件辅助）
- KVM架构与原理
- Hypervisor设计（Type 1/Type 2）
- 容器技术（Docker/Podman）
- cgroup与namespace
- 虚拟化vs容器对比

### 08. [系统安全](08_security.md)
- 权限与访问控制（DAC/MAC）
- SELinux与AppArmor
- 沙箱技术（seccomp/pledge）
- 内核安全机制（ASLR/DEP/SMEP）
- 漏洞与攻击（Buffer Overflow/提权）
- 安全审计

### 09. [性能调优](09_performance_tuning.md)
- 性能分析方法论
- CPU性能分析（perf/火焰图）
- 内存性能分析（valgrind/heaptrack）
- I/O性能分析（iostat/iotop）
- 系统调优参数（sysctl）
- BPF与eBPF追踪

### 10. [现代操作系统](10_modern_os.md)
- Linux内核架构
- Windows NT内核
- macOS内核（XNU）
- 实时操作系统（RTOS）
- 移动操作系统（Android/iOS）
- 微内核新趋势（Fuchsia/seL4）

## 🎯 学习目标

### 初级目标（2-3个月）
- 理解OS的基本概念和作用
- 掌握进程、线程、内存的核心机制
- 能够使用基本的系统调用

### 中级目标（3-6个月）
- 深入理解虚拟内存和文件系统
- 掌握并发编程与同步机制
- 能够进行基本的性能分析

### 高级目标（6个月+）
- 理解内核源码
- 掌握虚拟化技术
- 能够进行系统级性能调优

## 🔗 与其他模块的关系

```
┌─────────────────────────────────────────────────┐
│          模块关系图谱                           │
├─────────────────────────────────────────────────┤
│                                                 │
│  Computer_Hardware                              │
│         │                                       │
│         ▼                                       │
│  Operating_Systems (本模块)                    │
│         │                                       │
│         ├─▶ Linux                               │
│         │   (Linux系统实战)                     │
│         │                                       │
│         ├─▶ Container                           │
│         │   (容器技术详解)                      │
│         │                                       │
│         ├─▶ Networking                          │
│         │   (网络协议栈)                        │
│         │                                       │
│         └─▶ Performance                         │
│             (系统性能优化)                      │
└─────────────────────────────────────────────────┘
```

**与Linux模块的区别：**
- **Operating_Systems（本模块）**: 重原理、算法、内核机制
- **Linux**: 重实战、命令、运维配置

**推荐学习顺序：**
1. **Computer_Hardware** → 理解硬件基础
2. **Operating_Systems（本模块）** → 理解OS原理
3. **Linux** → 实战Linux系统
4. **Container** → 容器编排实践
5. **Performance** → 性能调优综合应用

## 💡 学习建议

### 理论与实践结合

```bash
# 查看进程信息
ps aux | head
top -H  # 查看线程

# 查看内存使用
free -h
vmstat 1
cat /proc/meminfo

# 查看文件系统
df -Th
lsblk
mount | grep ext4

# 查看I/O
iostat -x 1
iotop

# 系统调用追踪
strace ls
ltrace ls

# 内核参数
sysctl -a | grep vm
cat /proc/sys/vm/swappiness
```

### 推荐实验

1. **编写简单Shell** - 理解进程创建与管道
2. **实现内存分配器** - 理解内存管理
3. **文件系统模拟器** - 理解inode与数据块
4. **生产者消费者** - 理解并发同步
5. **性能分析实战** - 使用perf/eBPF

### 推荐工具

```
┌────────────────────────────────────────────┐
│        系统分析工具集                      │
├────────────────────────────────────────────┤
│                                            │
│  CPU:    top, htop, perf, flamegraph      │
│  内存:   free, vmstat, pmap, valgrind     │
│  I/O:    iostat, iotop, blktrace          │
│  网络:   netstat, ss, tcpdump, wireshark  │
│  追踪:   strace, ltrace, ftrace, bpftrace │
│  综合:   sysstat, dstat, atop, glances    │
└────────────────────────────────────────────┘
```

## 📊 OS技术演进

```
┌────────────────────────────────────────────────────┐
│          操作系统技术发展                          │
├────────────────────────────────────────────────────┤
│                                                    │
│  1960s  分时系统 (UNIX诞生)                       │
│     ↓                                              │
│  1980s  图形界面 (Windows/Mac)                    │
│     ↓                                              │
│  1990s  网络时代 (Linux崛起)                      │
│     ↓                                              │
│  2000s  虚拟化技术 (VMware/Xen)                   │
│     ↓                                              │
│  2010s  容器革命 (Docker/Kubernetes)              │
│     ↓                                              │
│  2020s  云原生OS (Serverless/eBPF)                │
│                                                    │
│  未来:  微内核 (Rust OS/形式验证)                 │
└────────────────────────────────────────────────────┘
```

## 📚 经典资源

### 必读书籍
- **《操作系统概念》(恐龙书)** - OS经典教材
- **《现代操作系统》(Tanenbaum)** - 深入浅出
- **《深入理解Linux内核》** - Linux内核详解
- **《Linux内核设计与实现》** - 内核机制精讲

### 在线课程
- [MIT 6.828: Operating System Engineering](https://pdos.csail.mit.edu/6.828/)
- [CS162 Operating Systems (Berkeley)](https://cs162.org/)
- [xv6操作系统实验](https://github.com/mit-pdos/xv6-public)

### 内核源码
- [Linux Kernel](https://github.com/torvalds/linux)
- [xv6 (教学OS)](https://github.com/mit-pdos/xv6-public)
- [seL4 (微内核)](https://github.com/seL4/seL4)

### 社区资源
- [LWN.net](https://lwn.net/) - Linux内核新闻
- [Brendan Gregg's Blog](https://www.brendangregg.com/) - 性能分析大师
- [Linux Performance](http://www.brendangregg.com/linuxperf.html) - 性能工具图谱

## 🎓 实验项目

### 初级项目
1. **Shell实现** - 进程管理与I/O重定向
2. **内存池** - 内存分配与管理
3. **线程同步** - 生产者消费者问题

### 中级项目
1. **简易文件系统** - inode/数据块/目录
2. **用户态线程库** - 协程调度
3. **虚拟内存模拟器** - 页表/TLB/缺页处理

### 高级项目
1. **xv6内核扩展** - 添加系统调用
2. **eBPF性能追踪** - 内核态性能分析
3. **微内核设计** - 模块化OS架构

## 🚀 开始学习

选择适合你的起点：
- **零基础**: 从 [01_os_overview.md](01_os_overview.md) 开始
- **有基础**: 直接学习 [02_process_management.md](02_process_management.md)
- **系统编程经验**: 跳到 [09_performance_tuning.md](09_performance_tuning.md)
- **内核开发**: 重点学习 [10_modern_os.md](10_modern_os.md)

祝学习愉快！🎉
