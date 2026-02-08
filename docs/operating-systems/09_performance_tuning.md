# 性能分析与调优

## 课程概述

本教程深入讲解操作系统性能分析与调优技术,从perf工具到火焰图,从eBPF到系统调优,帮助你全面掌握性能问题定位和优化方法。

**学习目标**:
- 掌握Linux性能分析工具(perf、eBPF、ftrace)
- 理解火焰图的生成与分析
- 学习CPU、内存、I/O性能调优
- 掌握系统瓶颈定位方法
- 理解性能监控指标
- 学习内核参数调优

---

## 1. 性能分析基础

### 1.1 性能指标

```
┌─────────────────────────────────────────────────────────────┐
│              性能分析的黄金指标                               │
└─────────────────────────────────────────────────────────────┘

USE方法 (Utilization, Saturation, Errors):
┌─────────────────────────────────────────────────────────────┐
│ 资源类型    │ 利用率          │ 饱和度          │ 错误      │
├─────────────┼─────────────────┼─────────────────┼──────────┤
│ CPU         │ %user+%system   │ 运行队列长度    │ 指令错误  │
│             │ (vmstat, mpstat)│ (vmstat r列)    │          │
├─────────────┼─────────────────┼─────────────────┼──────────┤
│ Memory      │ 已用内存比例    │ swap使用率      │ OOM kill │
│             │ (free)          │ (vmstat si/so)  │ (dmesg)  │
├─────────────┼─────────────────┼─────────────────┼──────────┤
│ Disk        │ %util           │ 平均队列长度    │ I/O错误  │
│             │ (iostat)        │ (iostat avgqu-sz│ (dmesg)  │
├─────────────┼─────────────────┼─────────────────┼──────────┤
│ Network     │ 带宽使用率      │ socket缓冲溢出  │ 丢包     │
│             │ (sar -n DEV)    │ (netstat -s)    │ (ifconfig│
└─────────────┴─────────────────┴─────────────────┴──────────┘

RED方法 (Rate, Errors, Duration) - 面向服务:
┌─────────────────────────────────────────────────────────────┐
│ • Rate (速率): 每秒请求数 (QPS/RPS)                          │
│ • Errors (错误率): 失败请求比例                              │
│ • Duration (延迟): 请求响应时间 (P50/P95/P99)               │
└─────────────────────────────────────────────────────────────┘

性能分析流程:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. 问题识别                                                 │
│     ┌──────────────────────────────────────────────┐       │
│     │ • 用户反馈慢                                  │       │
│     │ • 监控告警                                    │       │
│     │ • 容量规划                                    │       │
│     └────────────────┬─────────────────────────────┘       │
│                      │                                      │
│                      ▼                                      │
│  2. 快速检查 (60秒)                                          │
│     ┌──────────────────────────────────────────────┐       │
│     │ uptime    - 负载均衡                          │       │
│     │ dmesg -T  - 内核错误                          │       │
│     │ vmstat 1  - CPU/内存/swap                     │       │
│     │ mpstat -P ALL 1 - CPU各核心                  │       │
│     │ pidstat 1 - 进程资源使用                      │       │
│     │ iostat -xz 1 - 磁盘I/O                        │       │
│     │ free -m   - 内存使用                          │       │
│     │ sar -n DEV 1 - 网络I/O                        │       │
│     │ sar -n TCP,ETCP 1 - TCP统计                  │       │
│     │ top       - 进程概览                          │       │
│     └────────────────┬─────────────────────────────┘       │
│                      │                                      │
│                      ▼                                      │
│  3. 深入分析                                                 │
│     ┌──────────────────────────────────────────────┐       │
│     │ • CPU: perf, eBPF                             │       │
│     │ • 内存: valgrind, heaptrack                   │       │
│     │ • I/O: iotop, blktrace                        │       │
│     │ • 网络: tcpdump, wireshark                    │       │
│     │ • 系统调用: strace                            │       │
│     └────────────────┬─────────────────────────────┘       │
│                      │                                      │
│                      ▼                                      │
│  4. 优化与验证                                               │
│     ┌──────────────────────────────────────────────┐       │
│     │ • 应用层优化                                  │       │
│     │ • 内核参数调优                                │       │
│     │ • 硬件升级                                    │       │
│     │ • 架构调整                                    │       │
│     └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 系统性能观测工具全景

```
┌─────────────────────────────────────────────────────────────┐
│              Linux性能观测工具                                │
└─────────────────────────────────────────────────────────────┘

                        Application
                             │
                   ┌─────────┼─────────┐
                   │         │         │
               ┌───▼───┐ ┌───▼───┐ ┌───▼───┐
               │ strace│ │ ltrace│ │ gdb   │
               └───────┘ └───────┘ └───────┘
                             │
                        System Call
                             │
      ═══════════════════════╪═══════════════════════
                             │ Kernel
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐         ┌────▼────┐        ┌────▼────┐
    │ Scheduler│        │ Memory  │        │   I/O   │
    │          │        │ Manager │        │ Manager │
    └────┬────┘         └────┬────┘        └────┬────┘
         │                   │                   │
    ┌────▼────┐         ┌────▼────┐        ┌────▼────┐
    │ mpstat  │         │ free    │        │ iostat  │
    │ pidstat │         │ vmstat  │        │ iotop   │
    │ top     │         │ slabtop │        │ blktrace│
    │ perf    │         │ /proc   │        │ biosnoop│
    └─────────┘         └─────────┘        └─────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                        ┌────▼────┐
                        │ Hardware│
                        │         │
                        │ CPU     │
                        │ Memory  │
                        │ Disk    │
                        │ Network │
                        └─────────┘

工具分类:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│ 基础工具 (Basic):                                            │
│ • uptime, dmesg, top, htop, ps, vmstat, iostat, free        │
│                                                             │
│ 中级工具 (Intermediate):                                     │
│ • perf, strace, tcpdump, iotop, sysstat (sar), nicstat      │
│                                                             │
│ 高级工具 (Advanced):                                         │
│ • eBPF (bcc tools), ftrace, SystemTap                       │
│                                                             │
│ 可视化 (Visualization):                                      │
│ • FlameGraph, Grafana, Prometheus                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. CPU性能分析

### 2.1 Perf工具

```bash
#!/bin/bash
# Perf性能分析完整示例

echo "=== CPU性能快速检查 ==="

# 1. 查看系统负载
uptime
# 输出: load average: 2.51, 3.13, 2.72
# 解读: 1分钟/5分钟/15分钟平均负载
# 如果 > CPU核心数,说明有任务在等待

# 2. CPU利用率
mpstat -P ALL 1 3
# %usr: 用户空间CPU时间
# %sys: 内核空间CPU时间
# %iowait: 等待I/O的时间(高则I/O瓶颈)
# %idle: 空闲时间

echo ""
echo "=== Perf Top实时分析 ==="
# 实时查看CPU热点
perf top -g
# -g: 显示调用栈

echo ""
echo "=== Perf Record采样 ==="
# 采样30秒
perf record -F 99 -a -g -- sleep 30
# -F 99: 采样频率99Hz (避免与定时器冲突)
# -a: 所有CPU
# -g: 调用栈

# 查看报告
perf report --stdio

# 或交互式查看
# perf report

echo ""
echo "=== Perf Stat统计 ==="
# 统计性能计数器
perf stat ./your_program

# 输出示例:
# Performance counter stats for './your_program':
#       1234.56 msec task-clock         # 0.998 CPUs utilized
#            12      context-switches   # 9.717 /sec
#             0      cpu-migrations     # 0.000 /sec
#           456      page-faults        # 369.414 /sec
#   4,567,890,123    cycles             # 3.699 GHz
#   2,345,678,901    instructions       # 0.51 insn per cycle
#     456,789,012    branches           # 370.000 M/sec
#       1,234,567    branch-misses      # 0.27% of all branches

echo ""
echo "=== Perf火焰图 ==="
# 生成火焰图
perf script | ./FlameGraph/stackcollapse-perf.pl | \
    ./FlameGraph/flamegraph.pl > flamegraph.svg
```

```
┌─────────────────────────────────────────────────────────────┐
│              Perf事件类型                                     │
└─────────────────────────────────────────────────────────────┘

硬件事件 (Hardware Events):
┌────────────────────────────────────────────────────┐
│ • cpu-cycles / cycles       - CPU周期              │
│ • instructions              - 执行的指令数         │
│ • cache-references          - 缓存引用             │
│ • cache-misses              - 缓存未命中           │
│ • branch-instructions       - 分支指令             │
│ • branch-misses             - 分支预测失败         │
│ • bus-cycles                - 总线周期             │
└────────────────────────────────────────────────────┘

软件事件 (Software Events):
┌────────────────────────────────────────────────────┐
│ • cpu-clock                 - CPU时钟              │
│ • task-clock                - 任务时钟             │
│ • page-faults / faults      - 页错误               │
│ • context-switches / cs     - 上下文切换           │
│ • cpu-migrations            - CPU迁移              │
│ • minor-faults              - 次要页错误           │
│ • major-faults              - 主要页错误           │
└────────────────────────────────────────────────────┘

跟踪点 (Tracepoints):
┌────────────────────────────────────────────────────┐
│ • sched:sched_switch        - 进程调度             │
│ • syscalls:sys_enter_*      - 系统调用进入         │
│ • syscalls:sys_exit_*       - 系统调用退出         │
│ • block:block_rq_*          - 块I/O请求            │
│ • net:net_dev_*             - 网络设备             │
└────────────────────────────────────────────────────┘

使用示例:
# 监控缓存未命中
perf stat -e cache-misses,cache-references ./program

# 监控系统调用
perf record -e 'syscalls:sys_enter_*' -a

# 监控调度事件
perf record -e sched:sched_switch -a
```

### 2.2 火焰图分析

```
┌─────────────────────────────────────────────────────────────┐
│              火焰图 (Flame Graph)                             │
└─────────────────────────────────────────────────────────────┘

火焰图结构:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  func_a (1%)                                                │
│  ┌─┐                                                        │
│  └─┘                                                        │
│                                                             │
│  func_b (15%)                                               │
│  ┌───────────────┐                                         │
│  │ subfunc_b1 (8%)  subfunc_b2 (7%)                        │
│  │ ┌──────────┐     ┌────────┐                            │
│  └─┴──────────┴─────┴────────┘                            │
│                                                             │
│  func_c (84%)  ◀─ 热点函数!                                 │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ subfunc_c1 (60%)         subfunc_c2 (24%)             │ │
│  │ ┌─────────────────────┐  ┌──────────────┐            │ │
│  │ │ deeper_c1 (40%)     │  │ deeper_c2    │            │ │
│  │ │ ┌────────────────┐  │  │ (24%)        │            │ │
│  │ │ │ hot_loop (40%) │  │  │ ┌──────────┐ │            │ │
│  │ │ └────────────────┘  │  │ └──────────┘ │            │ │
│  └─┴─────────────────────┴──┴──────────────┴────────────┘ │
│                                                             │
│  X轴: 函数占用CPU比例 (宽度)                                 │
│  Y轴: 调用栈深度 (高度)                                      │
│  颜色: 随机 (相同函数相同颜色)                               │
└─────────────────────────────────────────────────────────────┘

火焰图类型:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│ 1. On-CPU火焰图                                             │
│    • 显示CPU执行时间                                        │
│    • 找热点函数                                             │
│    • perf record -F 99 -a -g                                │
│                                                             │
│ 2. Off-CPU火焰图                                            │
│    • 显示阻塞时间                                           │
│    • 找I/O等待、锁等待                                      │
│    • offcputime (eBPF工具)                                  │
│                                                             │
│ 3. Memory火焰图                                             │
│    • 显示内存分配                                           │
│    • 找内存泄漏                                             │
│                                                             │
│ 4. Differential火焰图                                       │
│    • 对比两次采样                                           │
│    • 找性能回归                                             │
└─────────────────────────────────────────────────────────────┘

分析技巧:
┌─────────────────────────────────────────────────────────────┐
│ 1. 宽平台: CPU密集型,可能需要优化                           │
│ 2. 高塔: 深调用栈,可能过度抽象                              │
│ 3. 锯齿: 多个小函数,考虑内联                                │
│ 4. 单一热点: 优化此函数                                     │
│ 5. 分散热点: 系统性问题                                     │
└─────────────────────────────────────────────────────────────┘
```

```bash
#!/bin/bash
# 火焰图生成脚本

DURATION=30  # 采样时长(秒)

# 1. 安装FlameGraph工具
if [ ! -d "FlameGraph" ]; then
    echo "Cloning FlameGraph..."
    git clone https://github.com/brendangregg/FlameGraph
fi

echo "=== 生成On-CPU火焰图 ==="
# 采样
perf record -F 99 -a -g -- sleep $DURATION

# 生成火焰图
perf script | ./FlameGraph/stackcollapse-perf.pl | \
    ./FlameGraph/flamegraph.pl --title="On-CPU Flame Graph" \
    > oncpu-flamegraph.svg

echo "On-CPU火焰图已生成: oncpu-flamegraph.svg"

# 针对特定进程
# perf record -F 99 -p $PID -g -- sleep $DURATION

echo ""
echo "=== 生成Off-CPU火焰图 (需要eBPF) ==="
# 使用offcputime工具
if command -v offcputime-bpfcc &> /dev/null; then
    offcputime-bpfcc -df -p $PID $DURATION > offcpu.stacks
    ./FlameGraph/flamegraph.pl --title="Off-CPU Flame Graph" \
        --colors=io offcpu.stacks > offcpu-flamegraph.svg
    echo "Off-CPU火焰图已生成: offcpu-flamegraph.svg"
else
    echo "offcputime工具未安装,跳过"
fi

echo ""
echo "=== 差异火焰图 ==="
# 对比两个perf.data
# perf script -i perf.data.old | ./FlameGraph/stackcollapse-perf.pl > old.folded
# perf script -i perf.data.new | ./FlameGraph/stackcollapse-perf.pl > new.folded
# ./FlameGraph/difffolded.pl old.folded new.folded | \
#     ./FlameGraph/flamegraph.pl > diff-flamegraph.svg
```

---

## 3. eBPF性能分析

### 3.1 eBPF工具集(BCC)

```
┌─────────────────────────────────────────────────────────────┐
│              eBPF (Extended Berkeley Packet Filter)          │
└─────────────────────────────────────────────────────────────┘

eBPF架构:
┌─────────────────────────────────────────────────────────────┐
│ 用户空间                                                     │
│ ┌───────────────────────────────────────────────────────┐  │
│ │ BCC/bpftrace工具                                       │  │
│ │ execsnoop, opensnoop, biolatency, tcplife...          │  │
│ └──────────────────┬────────────────────────────────────┘  │
│                    │ 编译eBPF程序                           │
│                    ▼                                        │
│ ════════════════════════════════════════════════════════    │
│ 内核空间                                                     │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ eBPF虚拟机                                            │   │
│ │ ┌──────────────────────────────────────────────────┐ │   │
│ │ │ Verifier (验证器)                                 │ │   │
│ │ │ • 检查内存访问                                    │ │   │
│ │ │ • 确保程序会终止                                  │ │   │
│ │ │ • 无死循环                                        │ │   │
│ │ └──────────────────────────────────────────────────┘ │   │
│ │ ┌──────────────────────────────────────────────────┐ │   │
│ │ │ JIT Compiler (即时编译)                           │ │   │
│ │ │ eBPF字节码 → 本地机器码                           │ │   │
│ │ └──────────────────────────────────────────────────┘ │   │
│ └───────────────────┬──────────────────────────────────┘   │
│                     │ 挂载到Hook点                          │
│                     ▼                                       │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Hook点                                                │   │
│ │ • Tracepoints (sched:sched_switch)                   │   │
│ │ • Kprobes (内核函数入口)                              │   │
│ │ • Uprobes (用户函数入口)                              │   │
│ │ • XDP (网络数据包)                                    │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

常用BCC工具:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│ CPU分析:                                                     │
│ • execsnoop     - 跟踪新进程执行                            │
│ • profile       - CPU分析(类似perf)                         │
│ • offcputime    - 测量阻塞时间                              │
│ • cpudist       - CPU运行时间分布                           │
│                                                             │
│ 内存分析:                                                    │
│ • memleak       - 内存泄漏检测                              │
│ • cachestat     - 页缓存统计                                │
│ • oomkill       - 跟踪OOM killer                            │
│                                                             │
│ I/O分析:                                                     │
│ • biolatency    - 块I/O延迟分布                             │
│ • biosnoop      - 跟踪块I/O                                 │
│ • ext4slower    - 跟踪慢ext4操作                            │
│ • opensnoop     - 跟踪open()系统调用                        │
│                                                             │
│ 网络分析:                                                    │
│ • tcpconnect    - 跟踪TCP连接                               │
│ • tcpaccept     - 跟踪TCP接受                               │
│ • tcpretrans    - 跟踪TCP重传                               │
│ • tcplife       - TCP连接生命周期                           │
│                                                             │
│ 文件系统:                                                    │
│ • fileslower    - 跟踪慢文件操作                            │
│ • filelife      - 文件生命周期                              │
│ • vfsstat       - VFS统计                                   │
└─────────────────────────────────────────────────────────────┘
```

```bash
#!/bin/bash
# eBPF工具使用示例

echo "=== 安装BCC工具 ==="
# Ubuntu/Debian
# apt-get install bpfcc-tools linux-headers-$(uname -r)

# CentOS/RHEL
# yum install bcc-tools

echo ""
echo "=== 1. 跟踪进程执行 ==="
# 查看所有新进程
execsnoop-bpfcc
# 输出: 时间戳, PID, PPID, 返回值, 命令

echo ""
echo "=== 2. 块I/O延迟分析 ==="
# 查看块I/O延迟直方图
biolatency-bpfcc 10 1
# 输出:
#      usecs               : count     distribution
#          0 -> 1          : 0        |                    |
#          2 -> 3          : 0        |                    |
#          4 -> 7          : 0        |                    |
#          8 -> 15         : 0        |                    |
#         16 -> 31         : 0        |                    |
#         32 -> 63         : 10       |**                  |
#         64 -> 127        : 200      |********            |
#        128 -> 255        : 500      |********************|
#        256 -> 511        : 150      |******              |

echo ""
echo "=== 3. 跟踪慢文件I/O ==="
# 跟踪超过10ms的文件操作
fileslower-bpfcc 10
# 输出: 时间, 进程, 操作类型, 文件名, 延迟(ms)

echo ""
echo "=== 4. TCP连接跟踪 ==="
# 跟踪新的TCP连接
tcpconnect-bpfcc
# 输出: PID, 进程名, 目标IP, 目标端口

# 跟踪TCP连接生命周期
tcplife-bpfcc
# 输出: PID, 进程, 本地IP:端口, 远程IP:端口, 发送/接收字节, 连接时长(ms)

echo ""
echo "=== 5. 内存泄漏检测 ==="
# 跟踪内存分配(5秒)
memleak-bpfcc -p $PID 5
# 输出: 分配但未释放的内存,带调用栈

echo ""
echo "=== 6. CPU分析 ==="
# CPU火焰图采样(类似perf)
profile-bpfcc -adf -p $PID 30 > profile.stacks
# -a: 所有CPU
# -d: 包含时间戳
# -f: 输出折叠的调用栈

echo ""
echo "=== 7. Off-CPU分析 ==="
# 跟踪阻塞时间
offcputime-bpfcc -df -p $PID 30 > offcpu.stacks
# 找出程序在等什么

echo ""
echo "=== 8. 系统调用统计 ==="
# 统计系统调用次数和延迟
syscount-bpfcc -p $PID
# 输出: 系统调用名, 次数

echo ""
echo "=== 9. 页缓存统计 ==="
# 实时页缓存命中率
cachestat-bpfcc 1
# 输出: HITS, MISSES, DIRTIES, 命中率
```

### 3.2 bpftrace脚本

```bash
#!/usr/bin/env bpftrace
# bpftrace示例脚本

# 1. 跟踪open系统调用
tracepoint:syscalls:sys_enter_open
{
    printf("%-6d %-16s %s\n", pid, comm, str(args->filename));
}

# 2. 统计读取的字节数
tracepoint:syscalls:sys_exit_read
/args->ret > 0/
{
    @bytes[comm] = sum(args->ret);
}

END
{
    printf("\nRead bytes by process:\n");
    print(@bytes);
}

# 3. 测量函数延迟
kprobe:do_sys_open
{
    @start[tid] = nsecs;
}

kretprobe:do_sys_open
/@start[tid]/
{
    $duration = nsecs - @start[tid];
    @latency_us = hist($duration / 1000);
    delete(@start[tid]);
}

END
{
    printf("\nopen() latency distribution (microseconds):\n");
    print(@latency_us);
}

# 4. 跟踪TCP连接
kprobe:tcp_v4_connect
{
    printf("TCP connect by PID %d (%s)\n", pid, comm);
}

# 5. 跟踪内存分配
tracepoint:kmem:kmalloc
{
    @alloc_bytes[comm] = sum(args->bytes_alloc);
}

interval:s:1
{
    printf("\nMemory allocation by process (last 1s):\n");
    print(@alloc_bytes);
    clear(@alloc_bytes);
}
```

```bash
#!/bin/bash
# bpftrace一行命令示例

# 统计系统调用
bpftrace -e 'tracepoint:raw_syscalls:sys_enter { @[comm] = count(); }'

# VFS读取字节数
bpftrace -e 'kretprobe:vfs_read /retval > 0/ { @bytes = hist(retval); }'

# TCP连接延迟
bpftrace -e 'kprobe:tcp_v4_connect { @start[tid] = nsecs; }
             kretprobe:tcp_v4_connect /@start[tid]/ {
                 @connect_latency_ms = hist((nsecs - @start[tid]) / 1000000);
                 delete(@start[tid]);
             }'

# 进程CPU时间
bpftrace -e 'profile:hz:99 { @[comm] = count(); }'

# 页错误统计
bpftrace -e 'software:page-faults:1 { @[comm, pid] = count(); }'
```

---

## 4. 内存性能分析

```bash
#!/bin/bash
# 内存性能分析脚本

echo "=== 内存使用概览 ==="
free -h
#               total        used        free      shared  buff/cache   available
# Mem:           15Gi       8.0Gi       2.0Gi       500Mi       5.0Gi       6.5Gi
# Swap:          8.0Gi       1.0Gi       7.0Gi

# available: 可用内存(包含可回收的缓存)
# 如果available很低,可能有内存压力

echo ""
echo "=== 内存详细信息 ==="
cat /proc/meminfo | head -30

echo ""
echo "=== 页面换出/换入 ==="
vmstat 1 5
# si: swap in (从swap读入内存)
# so: swap out (从内存写入swap)
# 如果si/so频繁,说明内存不足

echo ""
echo "=== 进程内存使用 ==="
ps aux --sort=-%mem | head -10

echo ""
echo "=== Slab缓存 ==="
slabtop -o
# 内核slab分配器统计

echo ""
echo "=== 内存泄漏检测 ==="
# 使用valgrind
# valgrind --leak-check=full --show-leak-kinds=all ./program

# 使用memleak (eBPF)
# memleak-bpfcc -p $PID 30

echo ""
echo "=== OOM Killer日志 ==="
dmesg -T | grep -i "out of memory"

echo ""
echo "=== 大页内存 ==="
cat /proc/meminfo | grep -i huge

echo ""
echo "=== 内存性能计数器 ==="
perf stat -e 'cache-misses,cache-references,page-faults' ./program
```

---

## 5. I/O性能分析

```bash
#!/bin/bash
# I/O性能分析脚本

echo "=== 磁盘I/O统计 ==="
iostat -xz 1 5
# %util: 设备利用率 (接近100%表示饱和)
# await: 平均等待时间(ms)
# avgqu-sz: 平均队列长度
# r/s, w/s: 每秒读写次数

echo ""
echo "=== 进程I/O统计 ==="
iotop -b -n 1 -o
# -o: 只显示有I/O的进程

echo ""
echo "=== 块I/O跟踪 ==="
# 使用blktrace
# blktrace -d /dev/sda -o - | blkparse -i -

# 使用eBPF biosnoop
biosnoop-bpfcc
# 实时显示块I/O请求

echo ""
echo "=== I/O延迟分布 ==="
biolatency-bpfcc -D 10 1
# -D: 按磁盘分组

echo ""
echo "=== 文件系统I/O ==="
# 慢操作跟踪
fileslower-bpfcc 10  # 超过10ms的操作

# VFS统计
vfsstat-bpfcc 1

echo ""
echo "=== I/O调度器 ==="
for disk in /sys/block/sd*/queue/scheduler; do
    echo "$disk: $(cat $disk)"
done

echo ""
echo "=== 磁盘队列深度 ==="
for disk in /sys/block/sd*/queue/nr_requests; do
    echo "$disk: $(cat $disk)"
done
```

---

## 6. 系统调优

### 6.1 内核参数调优

```bash
#!/bin/bash
# 系统性能调优脚本

echo "=== 网络调优 ==="
# TCP连接优化
sysctl -w net.ipv4.tcp_tw_reuse=1
sysctl -w net.ipv4.tcp_fin_timeout=30
sysctl -w net.ipv4.tcp_max_syn_backlog=8192
sysctl -w net.ipv4.tcp_max_tw_buckets=5000

# TCP缓冲区
sysctl -w net.core.rmem_max=16777216
sysctl -w net.core.wmem_max=16777216
sysctl -w net.ipv4.tcp_rmem='4096 87380 16777216'
sysctl -w net.ipv4.tcp_wmem='4096 87380 16777216'

# 连接数
sysctl -w net.core.somaxconn=4096
sysctl -w net.ipv4.ip_local_port_range='10000 65535'

echo ""
echo "=== 内存调优 ==="
# Swappiness (0-100)
sysctl -w vm.swappiness=10  # 减少swap使用

# 脏页回写
sysctl -w vm.dirty_ratio=10
sysctl -w vm.dirty_background_ratio=5
sysctl -w vm.dirty_writeback_centisecs=100
sysctl -w vm.dirty_expire_centisecs=200

# Overcommit
sysctl -w vm.overcommit_memory=1
sysctl -w vm.overcommit_ratio=50

# 大页内存
echo 1024 > /proc/sys/vm/nr_hugepages

echo ""
echo "=== CPU调优 ==="
# CPU Governor
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > $cpu  # 性能模式
done

echo ""
echo "=== I/O调优 ==="
# I/O调度器
for disk in /sys/block/sd*/queue/scheduler; do
    echo mq-deadline > $disk  # HDD用deadline, SSD用none
done

# 预读大小
for disk in /sys/block/sd*/queue/read_ahead_kb; do
    echo 512 > $disk
done

# 队列深度
for disk in /sys/block/sd*/queue/nr_requests; do
    echo 256 > $disk
done

echo ""
echo "=== 文件系统调优 ==="
# 挂载选项示例
# mount -o noatime,nodiratime,data=writeback /dev/sda1 /mnt

echo ""
echo "=== 保存设置 ==="
# 将设置写入/etc/sysctl.conf
# sysctl -p  # 加载配置
```

### 6.2 应用层优化

```python
#!/usr/bin/env python3
"""
应用层性能优化示例
"""
import time
import cProfile
import pstats
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# 1. 使用缓存
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 2. 使用生成器(节省内存)
def read_large_file_bad(filename):
    """不好的做法: 一次读取整个文件"""
    with open(filename) as f:
        return f.readlines()  # 占用大量内存

def read_large_file_good(filename):
    """好的做法: 使用生成器"""
    with open(filename) as f:
        for line in f:
            yield line  # 逐行处理

# 3. 并发处理
def process_item(item):
    # 模拟耗时操作
    time.sleep(0.1)
    return item * 2

def parallel_processing(items):
    """并行处理"""
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_item, items))
    return results

# 4. 性能分析
def profile_function():
    """使用cProfile分析函数"""
    profiler = cProfile.Profile()
    profiler.enable()

    # 要分析的代码
    result = fibonacci(30)

    profiler.disable()

    # 打印结果
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)

if __name__ == '__main__':
    # 运行性能分析
    profile_function()

    # 命令行分析:
    # python -m cProfile -s cumulative script.py
```

---

## 7. 性能监控与报警

```bash
#!/bin/bash
# 性能监控脚本

THRESHOLD_CPU=80
THRESHOLD_MEM=90
THRESHOLD_DISK=85

monitor() {
    while true; do
        # CPU使用率
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
        if (( $(echo "$cpu_usage > $THRESHOLD_CPU" | bc -l) )); then
            echo "[ALERT] High CPU usage: $cpu_usage%"
            # 发送告警
        fi

        # 内存使用率
        mem_usage=$(free | grep Mem | awk '{print ($3/$2) * 100.0}')
        if (( $(echo "$mem_usage > $THRESHOLD_MEM" | bc -l) )); then
            echo "[ALERT] High memory usage: $mem_usage%"
        fi

        # 磁盘使用率
        df -h | awk '$5 > "'$THRESHOLD_DISK'%" {print "[ALERT] High disk usage on "$6": "$5}'

        sleep 60
    done
}

# Prometheus导出器示例
cat > /tmp/metrics.prom << 'EOF'
# HELP node_cpu_usage CPU usage percentage
# TYPE node_cpu_usage gauge
node_cpu_usage{cpu="all"} 45.2

# HELP node_memory_usage Memory usage percentage
# TYPE node_memory_usage gauge
node_memory_usage 62.5

# HELP node_disk_io_time Disk I/O time in ms
# TYPE node_disk_io_time counter
node_disk_io_time{device="sda"} 12345
EOF

echo "监控指标已导出到 /tmp/metrics.prom"
```

---

## 8. 总结

本教程深入讲解了操作系统性能分析与调优:

**核心知识点**:
1. 性能指标: USE方法、RED方法
2. 工具链: perf、eBPF、ftrace
3. 火焰图: On-CPU、Off-CPU分析
4. eBPF: 零开销跟踪、动态插桩
5. 系统调优: 内核参数、应用优化

**实战技能**:
- 使用perf定位CPU热点
- 生成和分析火焰图
- 使用eBPF工具进行深度分析
- 进行内存、I/O性能调优
- 配置性能监控

**最佳实践**:
1. 先测量再优化
2. 关注瓶颈而非平均值
3. 使用P99/P999而非平均延迟
4. 火焰图可视化分析
5. 持续监控性能指标
6. 逐步调优并验证

性能优化是一个持续迭代的过程!
