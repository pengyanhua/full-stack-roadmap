# Linux 基础教程

## 一、Linux 概述

### 什么是 Linux?

Linux 是一个开源的类 Unix 操作系统内核,由 Linus Torvalds 于 1991 年创建。

```
Linux 生态系统:
┌─────────────────────────────────────────────────────────────┐
│                         应用层                               │
│  (Web服务器、数据库、开发工具、桌面应用等)                    │
├─────────────────────────────────────────────────────────────┤
│                         工具层                               │
│  (Shell、GNU工具集、包管理器等)                              │
├─────────────────────────────────────────────────────────────┤
│                       Linux 内核                             │
│  (进程管理、内存管理、设备驱动、文件系统、网络协议栈)           │
├─────────────────────────────────────────────────────────────┤
│                         硬件层                               │
│  (CPU、内存、硬盘、网卡等)                                   │
└─────────────────────────────────────────────────────────────┘
```

### 主流 Linux 发行版

| 发行版 | 特点 | 适用场景 | 包管理器 |
|--------|------|----------|----------|
| **Ubuntu** | 用户友好、社区活跃 | 桌面、开发、服务器 | apt/dpkg |
| **Debian** | 稳定可靠、纯开源 | 服务器、嵌入式 | apt/dpkg |
| **CentOS/Rocky** | 企业级稳定性 | 企业服务器 | yum/dnf |
| **Fedora** | 前沿技术、快速更新 | 开发、测试 | dnf |
| **Arch Linux** | 滚动更新、高度定制 | 极客、学习 | pacman |
| **Alpine** | 极简、安全 | 容器、嵌入式 | apk |

### Linux 核心哲学

```
Unix/Linux 设计哲学:
├── 一切皆文件 (Everything is a file)
│   ├── 普通文件、目录、设备、socket 都是文件
│   └── 统一的文件操作接口
│
├── 小即是美 (Small is beautiful)
│   ├── 每个程序只做一件事并做好
│   └── 通过管道组合完成复杂任务
│
├── 可移植性 (Portability)
│   ├── 使用纯文本存储配置
│   └── 跨平台兼容
│
└── 提供机制而非策略 (Mechanism, not policy)
    └── 给用户选择权和灵活性
```

## 二、系统架构

### 目录结构

```
/                          # 根目录
├── bin/                   # 基本命令二进制文件 (ls, cp, mv)
├── boot/                  # 启动文件 (内核、grub)
├── dev/                   # 设备文件
│   ├── sda                # 第一块硬盘
│   ├── tty                # 终端设备
│   └── null               # 空设备
├── etc/                   # 系统配置文件
│   ├── passwd             # 用户信息
│   ├── group              # 组信息
│   ├── fstab              # 文件系统挂载配置
│   ├── hosts              # 主机名解析
│   └── systemd/           # systemd 配置
├── home/                  # 用户家目录
│   ├── user1/
│   └── user2/
├── lib/                   # 共享库文件
├── media/                 # 可移动设备挂载点
├── mnt/                   # 临时挂载点
├── opt/                   # 可选应用程序
├── proc/                  # 进程和内核信息 (虚拟文件系统)
│   ├── cpuinfo            # CPU 信息
│   ├── meminfo            # 内存信息
│   └── [pid]/             # 进程信息
├── root/                  # root 用户家目录
├── run/                   # 运行时数据 (PID 文件、socket)
├── sbin/                  # 系统管理命令 (需要 root 权限)
├── srv/                   # 服务数据
├── sys/                   # 设备和驱动信息 (虚拟文件系统)
├── tmp/                   # 临时文件 (重启后清空)
├── usr/                   # 用户程序和数据
│   ├── bin/               # 用户命令
│   ├── lib/               # 应用程序库
│   ├── local/             # 本地安装的软件
│   ├── share/             # 共享数据
│   └── src/               # 源代码
└── var/                   # 可变数据
    ├── log/               # 日志文件
    ├── cache/             # 应用缓存
    ├── spool/             # 任务队列
    └── www/               # Web 内容
```

### 关键目录说明

```
/etc - 配置文件目录
├── 所有系统和应用配置文件
├── 纯文本格式,可手动编辑
├── 重要文件:
│   ├── /etc/passwd          # 用户账户信息
│   ├── /etc/shadow          # 用户密码哈希
│   ├── /etc/group           # 组信息
│   ├── /etc/sudoers         # sudo 权限配置
│   ├── /etc/ssh/sshd_config # SSH 服务器配置
│   ├── /etc/hostname        # 主机名
│   └── /etc/fstab           # 文件系统挂载表

/var - 可变数据目录
├── 存储运行时变化的数据
├── 重要子目录:
│   ├── /var/log/            # 系统和应用日志
│   │   ├── syslog           # 系统日志
│   │   ├── auth.log         # 认证日志
│   │   ├── kern.log         # 内核日志
│   │   └── nginx/           # Nginx 日志
│   ├── /var/cache/          # 应用缓存
│   └── /var/spool/          # 打印、邮件队列

/proc - 进程信息 (虚拟文件系统)
├── 不占用实际磁盘空间
├── 实时反映系统状态
├── 重要文件:
│   ├── /proc/cpuinfo        # CPU 详细信息
│   ├── /proc/meminfo        # 内存使用情况
│   ├── /proc/uptime         # 系统运行时间
│   ├── /proc/loadavg        # 系统负载
│   └── /proc/[pid]/         # 进程详细信息
│       ├── cmdline          # 启动命令
│       ├── environ          # 环境变量
│       └── status           # 进程状态

/sys - 设备和驱动信息 (虚拟文件系统)
├── 提供内核对象的结构化视图
├── 用于与设备和驱动交互
└── 通过读写文件控制硬件
```

## 三、用户与权限

### 用户管理

```bash
# ============================================================
#                      用户操作
# ============================================================

# 创建新用户
sudo useradd -m -s /bin/bash john       # -m 创建家目录, -s 指定 shell
sudo useradd -m -G sudo,docker john     # 同时加入多个附加组

# 设置密码
sudo passwd john

# 修改用户信息
sudo usermod -l newname oldname         # 修改用户名
sudo usermod -aG groupname john         # 添加到附加组
sudo usermod -s /bin/zsh john           # 修改默认 shell

# 删除用户
sudo userdel john                       # 只删除用户
sudo userdel -r john                    # 同时删除家目录和邮件

# 查看用户信息
id john                                 # 查看 UID、GID、所属组
whoami                                  # 当前用户
who                                     # 已登录用户
w                                       # 更详细的登录信息
last                                    # 登录历史

# ============================================================
#                      组操作
# ============================================================

# 创建组
sudo groupadd developers

# 修改组
sudo groupmod -n newgroup oldgroup      # 重命名组

# 删除组
sudo groupdel developers

# 查看组信息
groups john                             # 查看用户所属组
cat /etc/group                          # 所有组信息
getent group developers                 # 查看组成员
```

### 文件权限

```
权限表示法:
┌──────────┬─────┬─────┬─────┬─────────────────┐
│ 文件类型 │ 所有者│  组  │ 其他 │    说明          │
├──────────┼─────┼─────┼─────┼─────────────────┤
│    -     │ rwx │ r-x │ r-- │ 普通文件         │
│    d     │ rwx │ rwx │ r-x │ 目录             │
│    l     │ rwx │ rwx │ rwx │ 符号链接(总是777) │
│    b     │ rw- │ rw- │ --- │ 块设备           │
│    c     │ rw- │ rw- │ --- │ 字符设备         │
└──────────┴─────┴─────┴─────┴─────────────────┘

文件类型:
  - : 普通文件
  d : 目录
  l : 符号链接
  b : 块设备 (如硬盘)
  c : 字符设备 (如终端)
  s : socket
  p : 管道

权限位:
  r : read (读)     - 4
  w : write (写)    - 2
  x : execute (执行) - 1
  - : 无权限        - 0

权限组:
  第1组: 所有者 (owner)
  第2组: 所属组 (group)
  第3组: 其他人 (others)
```

### 权限管理

```bash
# ============================================================
#                   修改权限 (chmod)
# ============================================================

# 符号模式
chmod u+x file.sh                # 所有者添加执行权限
chmod g-w file.txt               # 组移除写权限
chmod o=r file.txt               # 其他人设置为只读
chmod a+x script.sh              # 所有人添加执行权限 (a=all)
chmod u+x,g+x,o-w file           # 组合操作

# 数字模式 (常用)
chmod 644 file.txt               # rw-r--r-- (文件常用)
chmod 755 script.sh              # rwxr-xr-x (可执行文件常用)
chmod 700 private/               # rwx------ (私有目录)
chmod 777 shared/                # rwxrwxrwx (所有人可读写执行,不推荐)

# 递归修改
chmod -R 755 directory/          # 递归修改目录及其内容

# 权限计算:
# 644 = 110 100 100 = rw-r--r--
#       ↓   ↓   ↓
#       所有者 组 其他
# 6 = 4(r) + 2(w)
# 4 = 4(r)
# 4 = 4(r)

# ============================================================
#                   修改所有者 (chown)
# ============================================================

# 修改所有者
sudo chown john file.txt         # 修改所有者为 john
sudo chown john:developers file  # 同时修改所有者和组
sudo chown :developers file      # 只修改组

# 递归修改
sudo chown -R john:developers /var/www/

# ============================================================
#                   修改组 (chgrp)
# ============================================================

# 修改文件所属组
sudo chgrp developers file.txt
sudo chgrp -R developers directory/

# ============================================================
#                   特殊权限
# ============================================================

# SUID (Set User ID) - 4000
# 执行文件时以文件所有者身份运行
chmod u+s /usr/bin/passwd        # passwd 命令需要 SUID
chmod 4755 file                  # rwsr-xr-x

# SGID (Set Group ID) - 2000
# 文件: 以文件所属组身份运行
# 目录: 新建文件继承目录的组
chmod g+s /shared                # 共享目录常用
chmod 2755 directory             # rwxr-sr-x

# Sticky Bit - 1000
# 目录中只有文件所有者可以删除自己的文件
chmod +t /tmp                    # /tmp 目录使用
chmod 1777 /shared               # rwxrwxrwt

# 查看特殊权限
ls -l /usr/bin/passwd            # -rwsr-xr-x (SUID)
ls -ld /tmp                      # drwxrwxrwt (Sticky)
```

### 默认权限 (umask)

```bash
# 查看当前 umask
umask                            # 输出: 0022

# umask 计算:
# 文件默认权限 = 666 - umask = 666 - 022 = 644
# 目录默认权限 = 777 - umask = 777 - 022 = 755

# 设置 umask
umask 027                        # 文件:640, 目录:750
umask 077                        # 文件:600, 目录:700 (更安全)

# 永久设置 (添加到 ~/.bashrc 或 /etc/profile)
echo "umask 027" >> ~/.bashrc
```

## 四、包管理

### APT (Debian/Ubuntu)

```bash
# ============================================================
#                   软件包操作
# ============================================================

# 更新软件包列表
sudo apt update                  # 更新本地软件包索引

# 升级软件包
sudo apt upgrade                 # 升级已安装的软件包
sudo apt full-upgrade            # 升级并处理依赖变化
sudo apt dist-upgrade            # 发行版升级

# 安装软件包
sudo apt install nginx           # 安装单个软件包
sudo apt install pkg1 pkg2 pkg3  # 安装多个
sudo apt install -y nginx        # 自动回答 yes
sudo apt install nginx=1.18.0-1  # 安装特定版本

# 删除软件包
sudo apt remove nginx            # 删除软件包(保留配置)
sudo apt purge nginx             # 删除软件包及配置
sudo apt autoremove              # 删除不需要的依赖

# 搜索软件包
apt search keyword               # 搜索软件包
apt search --names-only nginx    # 只搜索包名

# 查看软件包信息
apt show nginx                   # 详细信息
apt list --installed             # 已安装的包
apt list --upgradable            # 可升级的包

# ============================================================
#                   软件源管理
# ============================================================

# 软件源配置文件
/etc/apt/sources.list            # 主配置文件
/etc/apt/sources.list.d/         # 额外源配置目录

# 常用国内源 (Ubuntu 22.04)
# 编辑 /etc/apt/sources.list

# 清华源
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse

# 阿里源
deb https://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse

# 添加 PPA (Personal Package Archive)
sudo add-apt-repository ppa:user/ppa-name
sudo apt update

# 删除 PPA
sudo add-apt-repository --remove ppa:user/ppa-name

# ============================================================
#                   缓存管理
# ============================================================

# 清理缓存
sudo apt clean                   # 清理下载的包文件
sudo apt autoclean               # 清理过期的包文件

# 修复损坏的依赖
sudo apt --fix-broken install
sudo dpkg --configure -a
```

### YUM/DNF (CentOS/Fedora)

```bash
# ============================================================
#                   软件包操作 (DNF)
# ============================================================

# DNF 是 YUM 的下一代,命令基本兼容

# 更新
sudo dnf check-update            # 检查更新
sudo dnf update                  # 更新所有软件包
sudo dnf upgrade                 # 同 update

# 安装
sudo dnf install nginx
sudo dnf install -y nginx        # 自动回答 yes

# 删除
sudo dnf remove nginx
sudo dnf autoremove              # 删除不需要的依赖

# 搜索
dnf search nginx
dnf list installed               # 已安装的包
dnf list available               # 可用的包

# 查看信息
dnf info nginx

# ============================================================
#                   仓库管理
# ============================================================

# 仓库配置目录
/etc/yum.repos.d/

# 列出仓库
dnf repolist
dnf repolist all

# 启用/禁用仓库
sudo dnf config-manager --enable epel
sudo dnf config-manager --disable epel

# 添加 EPEL 仓库 (Extra Packages for Enterprise Linux)
sudo dnf install epel-release

# 清理缓存
sudo dnf clean all
sudo dnf makecache                # 重建缓存
```

### 编译安装

```bash
# ============================================================
#                   从源码编译安装
# ============================================================

# 1. 安装编译工具
# Ubuntu/Debian
sudo apt install build-essential

# CentOS/RHEL
sudo yum groupinstall "Development Tools"

# 2. 下载源码
wget https://example.com/software-1.0.tar.gz
tar -xzf software-1.0.tar.gz
cd software-1.0

# 3. 配置、编译、安装
./configure --prefix=/usr/local   # 配置安装路径
make                              # 编译
make test                         # 测试 (可选)
sudo make install                 # 安装

# 4. 卸载
sudo make uninstall               # 如果支持

# 常用配置选项
./configure --help                # 查看所有选项
./configure \
  --prefix=/usr/local \           # 安装路径
  --enable-feature \              # 启用功能
  --disable-feature \             # 禁用功能
  --with-library=/path            # 指定库路径
```

## 五、系统服务管理

### systemd 基础

```
systemd 架构:
┌─────────────────────────────────────────────────────────────┐
│                         systemd                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   System    │  │    User     │  │   Session   │          │
│  │   Units     │  │    Units    │  │   Units     │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│  Unit 类型:                                                  │
│  ├── .service    # 服务                                     │
│  ├── .socket     # Socket                                   │
│  ├── .target     # 目标(类似运行级别)                       │
│  ├── .timer      # 定时器                                   │
│  ├── .mount      # 挂载点                                   │
│  └── .device     # 设备                                     │
└─────────────────────────────────────────────────────────────┘
```

### systemctl 命令

```bash
# ============================================================
#                   服务管理
# ============================================================

# 启动服务
sudo systemctl start nginx
sudo systemctl start nginx.service  # 效果相同

# 停止服务
sudo systemctl stop nginx

# 重启服务
sudo systemctl restart nginx

# 重新加载配置 (不重启服务)
sudo systemctl reload nginx

# 重启或重载 (优先重载)
sudo systemctl reload-or-restart nginx

# ============================================================
#                   开机自启
# ============================================================

# 设置开机自启
sudo systemctl enable nginx

# 禁用开机自启
sudo systemctl disable nginx

# 启动并设置开机自启
sudo systemctl enable --now nginx

# 查看是否开机自启
systemctl is-enabled nginx

# ============================================================
#                   状态查看
# ============================================================

# 查看服务状态
systemctl status nginx
# 输出示例:
# ● nginx.service - A high performance web server
#    Loaded: loaded (/lib/systemd/system/nginx.service; enabled)
#    Active: active (running) since Mon 2024-01-01 10:00:00 UTC
#    Main PID: 1234 (nginx)
#    Tasks: 2 (limit: 4915)
#    Memory: 10.5M
#    CGroup: /system.slice/nginx.service
#            ├─1234 nginx: master process
#            └─1235 nginx: worker process

# 查看服务是否运行
systemctl is-active nginx

# 查看服务是否失败
systemctl is-failed nginx

# 列出所有服务
systemctl list-units --type=service          # 运行中的服务
systemctl list-units --type=service --all    # 所有服务
systemctl list-unit-files --type=service     # 所有服务文件

# ============================================================
#                   日志查看 (journalctl)
# ============================================================

# 查看服务日志
journalctl -u nginx                  # 所有日志
journalctl -u nginx -n 50            # 最后50行
journalctl -u nginx -f               # 实时跟踪
journalctl -u nginx --since today    # 今天的日志
journalctl -u nginx --since "2024-01-01" --until "2024-01-02"

# 查看系统日志
journalctl                           # 所有日志
journalctl -b                        # 本次启动的日志
journalctl -b -1                     # 上次启动的日志
journalctl -p err                    # 只看错误日志
journalctl -f                        # 实时跟踪所有日志

# 日志管理
journalctl --disk-usage              # 查看日志占用空间
sudo journalctl --vacuum-size=500M   # 清理日志,保留500M
sudo journalctl --vacuum-time=7d     # 清理7天前的日志

# ============================================================
#                   系统管理
# ============================================================

# 重启系统
sudo systemctl reboot

# 关机
sudo systemctl poweroff

# 休眠
sudo systemctl hibernate

# 重新加载 systemd 配置
sudo systemctl daemon-reload
```

### 创建自定义服务

```bash
# 创建服务文件
sudo nano /etc/systemd/system/myapp.service
```

```ini
# /etc/systemd/system/myapp.service
[Unit]
Description=My Application Service
Documentation=https://example.com/docs
After=network.target                    # 在网络启动后启动
Wants=postgresql.service                # 依赖但不强制

[Service]
Type=simple                             # 服务类型
User=myapp                              # 运行用户
Group=myapp                             # 运行组
WorkingDirectory=/opt/myapp             # 工作目录
ExecStart=/opt/myapp/bin/start.sh       # 启动命令
ExecReload=/bin/kill -HUP $MAINPID      # 重载命令
ExecStop=/opt/myapp/bin/stop.sh         # 停止命令
Restart=on-failure                      # 失败时重启
RestartSec=10s                          # 重启间隔

# 环境变量
Environment="NODE_ENV=production"
Environment="PORT=3000"
EnvironmentFile=/opt/myapp/.env         # 从文件加载环境变量

# 资源限制
LimitNOFILE=65536                       # 最大文件描述符
LimitNPROC=4096                         # 最大进程数

# 日志
StandardOutput=journal                  # 标准输出到 journal
StandardError=journal                   # 标准错误到 journal
SyslogIdentifier=myapp                  # 日志标识

[Install]
WantedBy=multi-user.target              # 多用户模式启动
```

```bash
# 服务类型说明
Type=simple      # 默认,ExecStart 启动的进程是主进程
Type=forking     # ExecStart 会 fork 子进程,父进程退出
Type=oneshot     # 执行一次就结束的任务
Type=notify      # 服务启动后会发送通知
Type=dbus        # 通过 D-Bus 启动

# 重启策略
Restart=no              # 不重启
Restart=on-failure      # 失败时重启(推荐)
Restart=always          # 总是重启
Restart=on-abnormal     # 异常退出时重启

# 应用服务
sudo systemctl daemon-reload             # 重新加载配置
sudo systemctl enable myapp              # 设置开机自启
sudo systemctl start myapp               # 启动服务
sudo systemctl status myapp              # 查看状态
```

## 六、系统信息查看

```bash
# ============================================================
#                   系统信息
# ============================================================

# 系统版本
uname -a                         # 所有系统信息
uname -r                         # 内核版本
cat /etc/os-release              # 发行版信息
lsb_release -a                   # LSB 版本信息 (需安装)

# 主机信息
hostname                         # 主机名
hostnamectl                      # 详细主机信息

# 系统运行时间
uptime                           # 运行时间和负载
uptime -p                        # 简洁格式

# 系统负载
cat /proc/loadavg                # 1分钟、5分钟、15分钟平均负载
# 输出: 0.50 0.75 0.60 1/200 12345
#       ↓    ↓    ↓    ↓     ↓
#       1分钟 5分钟 15分钟 运行/总进程 最新PID

# ============================================================
#                   硬件信息
# ============================================================

# CPU 信息
lscpu                            # CPU 详细信息
cat /proc/cpuinfo                # CPU 信息文件
nproc                            # CPU 核心数

# 内存信息
free -h                          # 人类可读格式
free -m                          # MB 单位
cat /proc/meminfo                # 详细内存信息

# 磁盘信息
lsblk                            # 块设备列表
lsblk -f                         # 包含文件系统信息
fdisk -l                         # 磁盘分区信息 (需要 root)
df -h                            # 磁盘使用情况
df -i                            # inode 使用情况

# PCI 设备
lspci                            # PCI 设备列表
lspci -v                         # 详细信息

# USB 设备
lsusb                            # USB 设备列表
lsusb -v                         # 详细信息

# ============================================================
#                   性能监控
# ============================================================

# 实时进程监控
top                              # 经典进程监控
htop                             # 更友好的界面 (需安装)

# 快捷键 (top):
#   q    - 退出
#   k    - 杀死进程
#   M    - 按内存排序
#   P    - 按 CPU 排序
#   1    - 显示每个 CPU 核心

# I/O 监控
iostat                           # CPU 和 I/O 统计
iostat -x 2                      # 每2秒刷新,显示扩展信息
iotop                            # 类似 top 的 I/O 监控 (需要 root)

# 网络监控
iftop                            # 实时网络流量监控
nethogs                          # 按进程显示网络使用

# 综合监控
vmstat 2                         # 每2秒显示虚拟内存统计
sar -u 2 5                       # 每2秒采样CPU,共5次 (需安装 sysstat)
```

## 七、启动流程

```
Linux 启动流程:
┌────────────────────────────────────────────────────────────┐
│  1. BIOS/UEFI                                               │
│     ├── 加电自检 (POST)                                    │
│     └── 加载引导程序                                       │
├────────────────────────────────────────────────────────────┤
│  2. Bootloader (GRUB2)                                      │
│     ├── 显示启动菜单                                       │
│     ├── 加载 Linux 内核                                    │
│     └── 加载 initramfs (初始化内存文件系统)               │
├────────────────────────────────────────────────────────────┤
│  3. Linux Kernel                                            │
│     ├── 初始化硬件                                         │
│     ├── 挂载根文件系统                                     │
│     └── 启动 init 进程 (PID 1)                            │
├────────────────────────────────────────────────────────────┤
│  4. systemd (init 系统)                                    │
│     ├── 加载 default.target                                │
│     ├── 并行启动服务                                       │
│     └── 到达 multi-user.target 或 graphical.target        │
├────────────────────────────────────────────────────────────┤
│  5. 登录提示                                                │
│     ├── 图形界面: Display Manager (GDM, LightDM)           │
│     └── 命令行: getty (登录提示符)                         │
└────────────────────────────────────────────────────────────┘
```

### GRUB 配置

```bash
# GRUB 配置文件
/etc/default/grub                # 主配置
/boot/grub/grub.cfg              # 生成的配置 (不要手动编辑)

# 编辑配置
sudo nano /etc/default/grub

# 常用配置项
GRUB_DEFAULT=0                   # 默认启动项 (0=第一个)
GRUB_TIMEOUT=5                   # 菜单超时时间(秒)
GRUB_CMDLINE_LINUX=""            # 内核参数
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"  # 默认内核参数

# 更新 GRUB 配置
sudo update-grub                 # Debian/Ubuntu
sudo grub2-mkconfig -o /boot/grub2/grub.cfg  # CentOS

# 查看可用内核
ls /boot/vmlinuz-*
```

### 运行级别 (Runlevel) 与 Target

```
传统运行级别 → systemd Target:
┌────────┬────────────────┬──────────────────────────────┐
│ 运行级别│  systemd Target│          说明                │
├────────┼────────────────┼──────────────────────────────┤
│   0    │ poweroff       │ 关机                          │
│   1    │ rescue         │ 单用户模式(救援模式)           │
│   2    │ multi-user     │ 多用户模式(无网络)             │
│   3    │ multi-user     │ 多用户模式(命令行)             │
│   4    │ multi-user     │ 未使用                        │
│   5    │ graphical      │ 多用户模式(图形界面)           │
│   6    │ reboot         │ 重启                          │
└────────┴────────────────┴──────────────────────────────┘

# 查看当前 target
systemctl get-default

# 设置默认 target
sudo systemctl set-default multi-user.target    # 命令行模式
sudo systemctl set-default graphical.target     # 图形界面模式

# 切换 target
sudo systemctl isolate multi-user.target        # 切换到命令行
sudo systemctl isolate graphical.target         # 切换到图形界面

# 救援模式
sudo systemctl rescue           # 进入救援模式
sudo systemctl emergency        # 进入紧急模式
```

## 八、Shell 基础

### 常用 Shell

```bash
# 查看可用 shell
cat /etc/shells

# 输出:
# /bin/sh       # POSIX shell
# /bin/bash     # Bourne Again Shell (最常用)
# /bin/zsh      # Z Shell (功能强大)
# /bin/fish     # Friendly Interactive Shell
# /bin/dash     # Debian Almquist Shell (轻量)

# 查看当前 shell
echo $SHELL
echo $0

# 切换 shell
chsh -s /bin/zsh               # 修改默认 shell
```

### 环境变量

```bash
# 查看环境变量
env                            # 所有环境变量
printenv                       # 同 env
echo $PATH                     # 查看特定变量

# 重要环境变量
$HOME                          # 用户家目录
$USER                          # 当前用户名
$SHELL                         # 当前 shell
$PATH                          # 可执行文件搜索路径
$PWD                           # 当前工作目录
$OLDPWD                        # 之前的工作目录
$LANG                          # 语言设置
$EDITOR                        # 默认编辑器

# 设置环境变量
export VAR="value"             # 临时设置 (当前会话)
export PATH=$PATH:/new/path    # 添加到 PATH

# 永久设置 (按优先级)
~/.bashrc                      # 用户级 Bash 配置 (交互式 shell)
~/.bash_profile                # 用户级 Bash 配置 (登录 shell)
~/.profile                     # 用户级通用配置
/etc/profile                   # 系统级通用配置
/etc/bash.bashrc               # 系统级 Bash 配置
/etc/environment               # 系统级环境变量

# 添加到 ~/.bashrc
echo 'export MY_VAR="value"' >> ~/.bashrc
source ~/.bashrc               # 重新加载配置

# 取消设置
unset VAR
```

### 命令别名

```bash
# 查看别名
alias                          # 显示所有别名
alias ll                       # 查看特定别名

# 设置别名
alias ll='ls -lah'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias grep='grep --color=auto'
alias rm='rm -i'               # 删除前确认

# 永久设置 (添加到 ~/.bashrc)
echo "alias ll='ls -lah'" >> ~/.bashrc
source ~/.bashrc

# 取消别名
unalias ll

# 忽略别名执行原命令
\rm file.txt                   # 使用 \ 前缀
/bin/rm file.txt               # 使用完整路径
```

## 九、常用快捷键

```bash
# ============================================================
#                   命令行编辑
# ============================================================

Ctrl + A         # 移到行首
Ctrl + E         # 移到行尾
Ctrl + U         # 删除光标前的内容
Ctrl + K         # 删除光标后的内容
Ctrl + W         # 删除光标前的单词
Ctrl + Y         # 粘贴之前删除的内容
Ctrl + L         # 清屏 (同 clear 命令)
Ctrl + R         # 搜索历史命令 (反向搜索)
Ctrl + C         # 终止当前命令
Ctrl + D         # 退出当前 shell (同 exit)
Ctrl + Z         # 暂停当前程序 (后台运行用 bg,前台运行用 fg)

# ============================================================
#                   历史命令
# ============================================================

!!               # 执行上一条命令
!n               # 执行历史中第 n 条命令
!-n              # 执行倒数第 n 条命令
!string          # 执行最近以 string 开头的命令
!$               # 上一条命令的最后一个参数
!*               # 上一条命令的所有参数

# 查看历史
history          # 显示历史命令
history 10       # 显示最后10条
history -c       # 清空历史

# ============================================================
#                   Tab 补全
# ============================================================

Tab              # 自动补全
Tab Tab          # 显示所有可能的补全
```

这是 Linux 基础教程的第一部分,涵盖了系统架构、用户权限、包管理、服务管理等核心概念。
