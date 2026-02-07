# Linux 进程管理详解

## 一、进程概念

### 进程与线程

```
进程与线程的关系:
┌─────────────────────────────────────────────────────────────┐
│                         进程 (Process)                       │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  进程地址空间                                          │ │
│  │  ┌──────────┬──────────┬──────────┬──────────────┐    │ │
│  │  │   代码段  │  数据段  │   堆     │     栈       │    │ │
│  │  └──────────┴──────────┴──────────┴──────────────┘    │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  线程:                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   线程1      │  │   线程2      │  │   线程3      │     │
│  │  (独立栈)    │  │  (独立栈)    │  │  (独立栈)    │     │
│  │  共享地址空间 │  │  共享地址空间 │  │  共享地址空间 │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘

进程 vs 线程:
┌──────────┬─────────────────────┬─────────────────────┐
│   特性   │        进程         │        线程         │
├──────────┼─────────────────────┼─────────────────────┤
│ 定义     │ 资源分配的基本单位  │ CPU调度的基本单位   │
│ 地址空间 │ 独立                │ 共享进程地址空间    │
│ 资源开销 │ 大                  │ 小                  │
│ 通信     │ 进程间通信(IPC)     │ 直接访问共享内存    │
│ 创建销毁 │ 慢                  │ 快                  │
│ 独立性   │ 独立                │ 不独立              │
└──────────┴─────────────────────┴─────────────────────┘
```

### 进程状态

```
进程状态转换图:
┌─────────┐
│  新建   │
│  (New)  │
└────┬────┘
     │
     ▼
┌─────────┐     调度器选中      ┌─────────┐
│  就绪   │ ──────────────────> │  运行   │
│ (Ready) │ <────────────────── │(Running)│
└────▲────┘      时间片用完      └────┬────┘
     │                               │
     │                               │ I/O或事件等待
     │          I/O或事件完成         │
     │                               ▼
     └─────────────────────────┌─────────┐
                               │  等待   │
                               │(Waiting)│
                               └────┬────┘
                                    │
                                    │ 进程结束
                                    ▼
                               ┌─────────┐
                               │  终止   │
                               │  (Exit) │
                               └─────────┘

Linux 进程状态 (ps 输出的 STAT 列):
┌──────┬──────────────────────────────────────┐
│ 状态 │                说明                  │
├──────┼──────────────────────────────────────┤
│  R   │ Running - 运行或就绪                 │
│  S   │ Sleeping - 可中断睡眠 (等待事件)     │
│  D   │ Disk Sleep - 不可中断睡眠 (通常是I/O)│
│  T   │ Stopped - 停止 (Ctrl+Z 或 SIGSTOP)   │
│  t   │ Tracing Stop - 被调试器追踪停止      │
│  Z   │ Zombie - 僵尸进程 (已结束,等待回收)  │
│  X   │ Dead - 已死亡                        │
│  I   │ Idle - 空闲内核线程                  │
│  <   │ 高优先级                             │
│  N   │ 低优先级                             │
│  L   │ 页面锁定在内存中                     │
│  s   │ 会话领导者                           │
│  l   │ 多线程                               │
│  +   │ 前台进程组                           │
└──────┴──────────────────────────────────────┘
```

## 二、进程查看

### ps 命令详解

```bash
# ============================================================
#                   基本用法
# ============================================================

# 当前终端进程
ps

# 所有进程 (BSD 风格,推荐)
ps aux

# 所有进程 (System V 风格)
ps -ef

# ps aux 输出详解:
# USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND
# root         1  0.0  0.1 169416 11484 ?        Ss   10:00   0:01 /sbin/init
#  ↓          ↓   ↓    ↓     ↓     ↓   ↓         ↓      ↓      ↓      ↓
# 用户      进程ID CPU  内存  虚拟  物理 终端   状态  启动时间 CPU时间 命令

# 各列说明:
# USER: 进程所有者
# PID:  进程ID
# %CPU: CPU使用率
# %MEM: 内存使用率
# VSZ:  虚拟内存大小 (KB)
# RSS:  物理内存大小 (KB)
# TTY:  终端 (? 表示无终端)
# STAT: 进程状态
# START: 启动时间
# TIME: 累计CPU时间
# COMMAND: 命令

# ============================================================
#                   过滤和搜索
# ============================================================

# 按用户过滤
ps -u username
ps aux | grep username

# 按进程名过滤
ps aux | grep nginx
ps -C nginx                    # 更精确

# 按 PID 查看
ps -p 1234
ps -p 1234,5678,9012           # 多个 PID

# 按父进程 ID 查看
ps -f --ppid 1234

# 查看进程树
ps auxf                        # f: forest (树形)
ps -ejH                        # H: hierarchy (层次)
pstree                         # 专门的进程树命令
pstree -p                      # 显示 PID
pstree -u                      # 显示用户

# ============================================================
#                   排序
# ============================================================

# 按 CPU 使用率排序
ps aux --sort=-pcpu | head     # - 表示降序
ps aux --sort=pcpu | head      # 升序

# 按内存使用率排序
ps aux --sort=-pmem | head

# 按启动时间排序
ps aux --sort=start_time

# 多字段排序
ps aux --sort=-pcpu,+pmem

# ============================================================
#                   自定义输出
# ============================================================

# 自定义列
ps -eo pid,user,pcpu,pmem,comm

# 常用列:
# pid, ppid, user, group, command, comm
# pcpu, pmem, vsz, rss, tty, stat
# start, etime (运行时间), time (CPU时间)
# nice, pri (优先级)

# 格式化输出
ps -eo pid,user,comm,pcpu,pmem --sort=-pmem | head -10

# 查看特定信息
ps -eo pid,comm,etime          # PID、命令、运行时间
ps -eo pid,comm,rss --sort=-rss | head  # 内存使用最多的进程

# ============================================================
#                   实用示例
# ============================================================

# 查找占用内存最多的10个进程
ps aux --sort=-pmem | head -11

# 查找占用 CPU 最多的10个进程
ps aux --sort=-pcpu | head -11

# 查找僵尸进程
ps aux | grep 'Z'
ps aux | awk '$8=="Z" {print}'

# 统计进程数
ps aux | wc -l
ps -u username | wc -l         # 特定用户的进程数

# 查看线程
ps -eLf                        # 显示所有线程
ps -T -p 1234                  # 查看特定进程的线程

# 查看进程的环境变量
ps e -p 1234
cat /proc/1234/environ | tr '\0' '\n'

# 查看进程的启动命令
ps -o cmd= -p 1234
cat /proc/1234/cmdline | tr '\0' ' '

# 监控进程变化
watch -n 1 'ps aux --sort=-pmem | head -10'
```

### top/htop 实时监控

```bash
# ============================================================
#                   top 命令
# ============================================================

# 基本使用
top                            # 实时监控

# 常用选项
top -u username                # 只显示特定用户
top -p 1234                    # 只显示特定 PID
top -p 1234,5678               # 监控多个进程
top -d 2                       # 刷新间隔2秒 (默认3秒)
top -n 10                      # 迭代10次后退出
top -b > top.log               # 批处理模式,输出到文件

# top 交互命令 (在 top 运行时按键):
# ──────────────────────────────────────────
# 通用:
#   q      - 退出
#   h, ?   - 帮助
#   Space  - 立即刷新
#   d      - 更改刷新间隔
#   k      - 杀死进程
#   r      - 重新设置 nice 值
#
# 显示:
#   1      - 显示每个 CPU 核心
#   t      - 切换任务/CPU 信息显示
#   m      - 切换内存信息显示
#   c      - 显示完整命令
#   V      - 树形显示
#   i      - 切换显示空闲进程
#   H      - 显示线程
#
# 排序:
#   P      - 按 CPU 排序
#   M      - 按内存排序
#   T      - 按运行时间排序
#   N      - 按 PID 排序
#
# 过滤:
#   u      - 按用户过滤
#   o      - 按字段过滤
#   W      - 保存当前设置

# top 输出详解:
# ┌─────────────────────────────────────────────────────────────┐
# │ top - 10:30:00 up 5 days,  2:15,  3 users,  load average:   │
# │                                              0.50, 0.75, 0.60│
# │ Tasks: 200 total,   1 running, 199 sleeping,   0 stopped    │
# │ %Cpu(s):  5.0 us,  2.0 sy,  0.0 ni, 93.0 id,  0.0 wa        │
# │ MiB Mem :  15926 total,   8234 free,   4567 used,  3125 buff│
# │ MiB Swap:   2048 total,   2048 free,      0 used.  10234 av │
# └─────────────────────────────────────────────────────────────┘
#
# 第1行: 系统信息
#   当前时间 | 运行时间 | 登录用户数 | 平均负载 (1, 5, 15分钟)
#
# 第2行: 任务信息
#   总任务数 | 运行 | 睡眠 | 停止 | 僵尸
#
# 第3行: CPU 使用率
#   us: 用户空间   sy: 内核空间   ni: nice值调整
#   id: 空闲       wa: I/O等待    hi: 硬中断
#   si: 软中断     st: 虚拟化窃取
#
# 第4行: 内存信息
#   total: 总内存   free: 空闲   used: 使用   buff/cache: 缓存
#
# 第5行: 交换分区信息
#   类似内存信息

# ============================================================
#                   htop (需要安装)
# ============================================================

# 安装
sudo apt install htop          # Debian/Ubuntu
sudo yum install htop          # CentOS

# 使用
htop                           # 启动 htop

# htop 特点:
# - 彩色界面,更直观
# - 支持鼠标操作
# - 可以横向和纵向滚动
# - 树形显示进程
# - 可以直接杀死进程
# - 显示所有 CPU 核心

# htop 快捷键:
#   F1      - 帮助
#   F2      - 设置
#   F3      - 搜索进程
#   F4      - 过滤
#   F5      - 树形视图
#   F6      - 排序
#   F9      - 杀死进程
#   F10     - 退出
#   Space   - 标记进程
#   U       - 取消所有标记
#   /       - 搜索
#   u       - 显示特定用户
#   k       - 杀死进程
#   t       - 树形视图
#   H       - 显示/隐藏线程
#   K       - 隐藏内核线程
#   P       - 按 CPU 排序
#   M       - 按内存排序
#   T       - 按时间排序

# ============================================================
#                   其他监控工具
# ============================================================

# atop - 高级监控
sudo apt install atop
atop                           # 类似 top,但更详细
atop -d 5                      # 5秒刷新一次

# glances - Python 监控工具
sudo apt install glances
glances                        # 漂亮的监控界面
glances -w                     # Web 界面模式 (http://localhost:61208)

# nmon - 性能监控
sudo apt install nmon
nmon                           # 交互式监控
nmon -f -s 10 -c 60            # 10秒采样一次,共60次,输出到文件

# dstat - 统计工具
sudo apt install dstat
dstat                          # 实时系统统计
dstat -cdngy                   # CPU、磁盘、网络、分页、系统
dstat -t 5                     # 5秒刷新,带时间戳
```

## 三、进程控制

### 信号机制

```bash
# ============================================================
#                   信号列表
# ============================================================

# 查看所有信号
kill -l

# 常用信号:
┌──────┬────────┬─────────────────────────────────────────┐
│ 信号 │  说明  │              用途                       │
├──────┼────────┼─────────────────────────────────────────┤
│  1   │ SIGHUP │ 挂起,通常用于重新加载配置文件           │
│  2   │ SIGINT │ 中断 (Ctrl+C)                           │
│  3   │ SIGQUIT│ 退出 (Ctrl+\), 生成 core dump          │
│  9   │ SIGKILL│ 强制终止,不能被捕获或忽略               │
│ 15   │ SIGTERM│ 终止 (默认),可以被捕获,正常清理         │
│ 17   │ SIGCHLD│ 子进程状态改变                          │
│ 18   │ SIGCONT│ 继续运行 (从停止状态恢复)               │
│ 19   │ SIGSTOP│ 停止,不能被捕获或忽略                   │
│ 20   │ SIGTSTP│ 停止 (Ctrl+Z), 可以被捕获               │
│ 10   │ SIGUSR1│ 用户自定义信号1                         │
│ 12   │ SIGUSR2│ 用户自定义信号2                         │
└──────┴────────┴─────────────────────────────────────────┘

# ============================================================
#                   发送信号
# ============================================================

# kill 命令
kill PID                       # 发送 SIGTERM (15)
kill -15 PID                   # 同上
kill -TERM PID                 # 同上

kill -9 PID                    # 发送 SIGKILL (强制终止)
kill -KILL PID                 # 同上

kill -HUP PID                  # 重新加载配置 (如 nginx, sshd)
kill -1 PID                    # 同上

kill -STOP PID                 # 暂停进程
kill -CONT PID                 # 继续进程

# 示例:重新加载 Nginx 配置
sudo nginx -t                  # 测试配置
sudo kill -HUP $(cat /var/run/nginx.pid)
# 或
sudo nginx -s reload

# killall 命令 (按进程名)
killall nginx                  # 杀死所有 nginx 进程
killall -9 nginx               # 强制杀死
killall -HUP nginx             # 重新加载配置
killall -u username            # 杀死用户的所有进程
killall -i nginx               # 交互式确认

# pkill 命令 (按模式匹配)
pkill nginx                    # 匹配进程名
pkill -9 nginx                 # 强制终止
pkill -u username              # 杀死用户的进程
pkill -f "python script.py"    # 匹配完整命令行
pkill -P 1234                  # 杀死父进程的所有子进程

# 示例:杀死所有 Python 脚本
pkill -f "python.*\.py"

# ============================================================
#                   批量杀死进程
# ============================================================

# 杀死所有匹配的进程
ps aux | grep nginx | grep -v grep | awk '{print $2}' | xargs kill

# 或使用 pgrep (更简洁)
kill $(pgrep nginx)
kill -9 $(pgrep -f "python script.py")

# 杀死进程树 (包括子进程)
kill -- -$(ps -o pgid= -p PID)   # 杀死进程组

# 安全杀死进程的脚本
kill_process() {
    local name=$1
    local pid=$(pgrep -f "$name")

    if [ -z "$pid" ]; then
        echo "Process not found"
        return 1
    fi

    echo "Killing process: $name (PID: $pid)"
    kill $pid
    sleep 2

    if ps -p $pid > /dev/null; then
        echo "Process still running, force killing..."
        kill -9 $pid
    fi
}

kill_process "nginx"

# ============================================================
#                   捕获信号 (Shell 脚本)
# ============================================================

# trap 命令捕获信号
#!/bin/bash

# 清理函数
cleanup() {
    echo "Cleaning up..."
    rm -f /tmp/script.$$
    exit 0
}

# 捕获 SIGINT (Ctrl+C) 和 SIGTERM
trap cleanup INT TERM

# 捕获 EXIT (脚本退出时)
trap "echo 'Script exiting...'" EXIT

# 忽略信号
trap '' HUP          # 忽略 SIGHUP
trap '' INT          # 忽略 SIGINT (Ctrl+C 无效)

# 恢复默认处理
trap - INT           # 恢复 SIGINT 默认处理

# 示例:长时间运行的脚本
#!/bin/bash

# 捕获 Ctrl+C
trap 'echo "Interrupted!"; exit 130' INT

while true; do
    echo "Running... (Press Ctrl+C to stop)"
    sleep 1
done
```

### 进程优先级

```bash
# ============================================================
#                   Nice 值
# ============================================================

# Nice 值范围: -20 到 19
# - -20: 最高优先级 (只有 root 可以设置)
# -  0 : 默认优先级
# - 19: 最低优先级

# 查看进程优先级
ps -eo pid,ni,comm             # NI 列显示 nice 值
ps -l                          # PRI 列显示优先级

# 启动时设置 nice 值
nice -n 10 command             # nice 值为 10
nice -10 command               # 同上
nice command                   # 默认 nice 值为 10

# 后台低优先级任务
nice -n 19 tar -czf backup.tar.gz /data &

# 修改运行中进程的 nice 值
renice 5 -p 1234               # 设置 PID 1234 的 nice 值为 5
renice 10 -u username          # 设置用户所有进程
renice -5 -g groupname         # 设置组所有进程 (需要 root)

# 在 top 中修改 (按 r 键,输入 PID 和新的 nice 值)

# 示例:高优先级任务
sudo nice -n -10 important_task

# 示例:后台编译 (低优先级)
nice -n 15 make -j4 &

# ============================================================
#                   ionice (I/O 优先级)
# ============================================================

# I/O 调度类:
# 0: None (使用默认)
# 1: Real-time (实时)
# 2: Best-effort (尽力而为,默认)
# 3: Idle (空闲)

# 启动时设置 ionice
ionice -c 3 command            # 空闲优先级
ionice -c 2 -n 7 command       # Best-effort,优先级7 (0-7)
ionice -c 1 -n 0 command       # 实时,优先级0 (需要 root)

# 修改运行中进程
ionice -c 3 -p 1234            # 设置为空闲优先级

# 查看进程 I/O 优先级
ionice -p 1234

# 示例:后台备份 (低 I/O 优先级)
ionice -c 3 nice -n 19 rsync -av /data /backup

# ============================================================
#                   cgroups (控制组)
# ============================================================

# cgroups 可以限制进程的资源使用

# 查看 cgroup 信息
cat /proc/1234/cgroup

# 使用 systemd-run 限制资源
sudo systemd-run --scope -p CPUQuota=50% command
sudo systemd-run --scope -p MemoryLimit=500M command

# 限制 CPU 和内存
sudo systemd-run --scope \
    -p CPUQuota=50% \
    -p MemoryLimit=1G \
    command

# 示例:限制编译进程
sudo systemd-run --scope -p CPUQuota=200% make -j4
```

## 四、进程间通信 (IPC)

### 管道

```bash
# ============================================================
#                   匿名管道
# ============================================================

# 基本管道
command1 | command2

# 多级管道
ps aux | grep nginx | grep -v grep | awk '{print $2}'

# 管道 + tee (同时输出到文件和屏幕)
ps aux | tee processes.txt | grep nginx

# ============================================================
#                   命名管道 (FIFO)
# ============================================================

# 创建命名管道
mkfifo mypipe

# 向管道写入 (阻塞直到有读取者)
echo "Hello" > mypipe &

# 从管道读取
cat < mypipe

# 实用示例:进程间通信
# 终端1:
mkfifo /tmp/mypipe
while true; do
    if read line < /tmp/mypipe; then
        echo "Received: $line"
    fi
done

# 终端2:
echo "Message 1" > /tmp/mypipe
echo "Message 2" > /tmp/mypipe

# 清理
rm /tmp/mypipe
```

### 信号量与共享内存

```bash
# ============================================================
#                   查看 IPC 资源
# ============================================================

# 查看所有 IPC 资源
ipcs

# 查看共享内存
ipcs -m

# 查看信号量
ipcs -s

# 查看消息队列
ipcs -q

# 查看详细信息
ipcs -m -i shmid              # 查看特定共享内存

# 删除 IPC 资源
ipcrm -m shmid                # 删除共享内存
ipcrm -s semid                # 删除信号量
ipcrm -q msgid                # 删除消息队列

# 删除所有 (危险!)
ipcrm -a                      # 删除所有 IPC 资源

# 查看限制
ipcs -l
```

### 进程跟踪

```bash
# ============================================================
#                   strace - 系统调用跟踪
# ============================================================

# 跟踪程序执行
strace command

# 跟踪运行中的进程
strace -p 1234

# 只跟踪特定系统调用
strace -e open,read,write command
strace -e trace=file command  # 所有文件相关调用
strace -e trace=network command  # 所有网络相关调用

# 统计系统调用
strace -c command             # 显示统计信息

# 跟踪子进程
strace -f command             # 跟踪 fork 的子进程

# 输出到文件
strace -o output.txt command

# 显示时间戳
strace -t command             # 显示时间
strace -T command             # 显示每个调用的耗时

# 实用示例:
# 查看程序打开了哪些文件
strace -e open ls /home

# 查看程序读取了哪些配置文件
strace -e open,read nginx 2>&1 | grep conf

# 诊断程序卡死
strace -p 1234

# ============================================================
#                   ltrace - 库函数调用跟踪
# ============================================================

# 跟踪库函数调用
ltrace command

# 跟踪运行中的进程
ltrace -p 1234

# 只跟踪特定函数
ltrace -e malloc,free command

# 统计
ltrace -c command

# 输出到文件
ltrace -o output.txt command

# ============================================================
#                   lsof - 列出打开的文件
# ============================================================

# 查看进程打开的文件
lsof -p 1234

# 查看文件被哪些进程打开
lsof /var/log/syslog

# 查看网络连接
lsof -i                       # 所有网络连接
lsof -i :80                   # 端口 80
lsof -i TCP                   # TCP 连接
lsof -i UDP                   # UDP 连接
lsof -i TCP:80-443            # 端口范围

# 查看用户打开的文件
lsof -u username

# 查看程序打开的文件
lsof -c nginx

# 组合查询
lsof -u username -c nginx     # 用户 AND 程序
lsof -u username -i :80       # 用户的网络连接

# 实用示例:
# 查找被删除但仍被占用的文件
lsof | grep deleted

# 恢复被删除的文件
lsof | grep deleted
# 找到 FD (文件描述符),然后:
cp /proc/PID/fd/FD recovered_file

# 查看哪个进程占用了磁盘
lsof /mnt/disk
```

这是进程管理教程,涵盖了进程概念、查看、控制、通信和跟踪。
