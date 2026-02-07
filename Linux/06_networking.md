# Linux 网络管理

## 一、网络基础

### OSI 模型与 TCP/IP

```
OSI 七层模型 vs TCP/IP 四层模型:
┌──────────────────┬─────────────────┬────────────────────┐
│   OSI 模型       │  TCP/IP 模型    │      协议/示例      │
├──────────────────┼─────────────────┼────────────────────┤
│ 7. 应用层        │                 │ HTTP, FTP, DNS,    │
│ 6. 表示层        │  应用层         │ SMTP, SSH, Telnet  │
│ 5. 会话层        │                 │                    │
├──────────────────┼─────────────────┼────────────────────┤
│ 4. 传输层        │  传输层         │ TCP, UDP           │
├──────────────────┼─────────────────┼────────────────────┤
│ 3. 网络层        │  网络层         │ IP, ICMP, ARP      │
├──────────────────┼─────────────────┼────────────────────┤
│ 2. 数据链路层    │  网络接口层     │ Ethernet, WiFi,    │
│ 1. 物理层        │                 │ MAC, PPP           │
└──────────────────┴─────────────────┴────────────────────┘

TCP/IP 协议栈数据封装:
┌────────────────────────────────────────────────────────────┐
│                        应用层数据                           │
└───────────────────┬────────────────────────────────────────┘
                    │ 添加传输层头部
                    ▼
┌────────────────────────────────────────────────────────────┐
│ TCP/UDP 头部      │              应用层数据                │
└───────────────────┼────────────────────────────────────────┘
                    │ 添加网络层头部
                    ▼
┌────────────────────────────────────────────────────────────┐
│ IP 头部           │ TCP 头部      │     应用层数据         │
└───────────────────┼───────────────┼────────────────────────┘
                    │ 添加数据链路层头部和尾部
                    ▼
┌────────────────────────────────────────────────────────────┐
│Eth头│ IP头 │TCP头│  数据  │ Eth尾 │                        │
└─────┴──────┴─────┴────────┴───────┘                        │
      帧                                                       │
└────────────────────────────────────────────────────────────┘
```

### 网络配置文件

```bash
# ============================================================
#                   重要配置文件
# ============================================================

# 网络接口配置 (Debian/Ubuntu)
/etc/network/interfaces

# 示例:
auto eth0
iface eth0 inet static
    address 192.168.1.100
    netmask 255.255.255.0
    gateway 192.168.1.1
    dns-nameservers 8.8.8.8 8.8.4.4

# DHCP 配置:
auto eth0
iface eth0 inet dhcp

# ────────────────────────────────────────────────────────────

# 网络配置 (CentOS/RHEL)
/etc/sysconfig/network-scripts/ifcfg-eth0

# 示例:
DEVICE=eth0
BOOTPROTO=static        # static 或 dhcp
ONBOOT=yes
IPADDR=192.168.1.100
NETMASK=255.255.255.0
GATEWAY=192.168.1.1
DNS1=8.8.8.8
DNS2=8.8.4.4

# ────────────────────────────────────────────────────────────

# DNS 配置
/etc/resolv.conf

# 示例:
nameserver 8.8.8.8
nameserver 8.8.4.4
search example.com
options timeout:2 attempts:3

# ────────────────────────────────────────────────────────────

# 主机名
/etc/hostname           # 主机名
/etc/hosts              # 本地 DNS 解析

# /etc/hosts 示例:
127.0.0.1   localhost
127.0.1.1   myhostname
192.168.1.10 server1.example.com server1
192.168.1.20 server2.example.com server2

# ────────────────────────────────────────────────────────────

# 路由表
/etc/iproute2/rt_tables

# ────────────────────────────────────────────────────────────

# NetworkManager 配置
/etc/NetworkManager/NetworkManager.conf
/etc/NetworkManager/system-connections/
```

## 二、网络接口管理

### ip 命令 (推荐)

```bash
# ============================================================
#                   查看网络接口
# ============================================================

# 查看所有接口
ip link show                   # 数据链路层
ip addr show                   # 网络层 (包含 IP 地址)
ip -s link                     # 显示统计信息
ip -s -h link                  # 人类可读格式

# 查看特定接口
ip addr show eth0
ip link show eth0

# 简洁输出
ip -br addr                    # brief 格式
ip -br link

# ============================================================
#                   配置 IP 地址
# ============================================================

# 添加 IP 地址
sudo ip addr add 192.168.1.100/24 dev eth0

# 删除 IP 地址
sudo ip addr del 192.168.1.100/24 dev eth0

# 刷新接口的所有 IP
sudo ip addr flush dev eth0

# ============================================================
#                   启用/禁用接口
# ============================================================

# 启用接口
sudo ip link set eth0 up

# 禁用接口
sudo ip link set eth0 down

# ============================================================
#                   修改 MAC 地址
# ============================================================

# 查看当前 MAC
ip link show eth0

# 修改 MAC 地址
sudo ip link set eth0 down
sudo ip link set eth0 address 00:11:22:33:44:55
sudo ip link set eth0 up

# ============================================================
#                   修改 MTU
# ============================================================

# 查看 MTU
ip link show eth0 | grep mtu

# 修改 MTU
sudo ip link set eth0 mtu 1400

# ============================================================
#                   虚拟接口
# ============================================================

# 创建虚拟接口
sudo ip link add link eth0 name eth0.10 type vlan id 10

# 删除虚拟接口
sudo ip link delete eth0.10
```

### 传统 ifconfig (已弃用)

```bash
# ifconfig 已弃用,但很多系统仍在使用

# 查看接口
ifconfig
ifconfig eth0

# 启用/禁用接口
sudo ifconfig eth0 up
sudo ifconfig eth0 down

# 配置 IP 地址
sudo ifconfig eth0 192.168.1.100 netmask 255.255.255.0

# 添加别名 IP
sudo ifconfig eth0:0 192.168.1.101 netmask 255.255.255.0

# 修改 MAC 地址
sudo ifconfig eth0 hw ether 00:11:22:33:44:55

# 修改 MTU
sudo ifconfig eth0 mtu 1400
```

### nmcli (NetworkManager)

```bash
# ============================================================
#                   基本操作
# ============================================================

# 查看连接
nmcli connection show
nmcli con show                 # 简写

# 查看设备
nmcli device status
nmcli dev status               # 简写

# 查看详细信息
nmcli con show "Wired connection 1"
nmcli dev show eth0

# ============================================================
#                   创建连接
# ============================================================

# 创建静态 IP 连接
nmcli con add \
    type ethernet \
    con-name mycon \
    ifname eth0 \
    ip4 192.168.1.100/24 \
    gw4 192.168.1.1

# 添加 DNS
nmcli con mod mycon ipv4.dns "8.8.8.8 8.8.4.4"

# 创建 DHCP 连接
nmcli con add \
    type ethernet \
    con-name mycon \
    ifname eth0

# ============================================================
#                   修改连接
# ============================================================

# 修改 IP 地址
nmcli con mod mycon ipv4.addresses 192.168.1.101/24
nmcli con mod mycon ipv4.gateway 192.168.1.1
nmcli con mod mycon ipv4.method manual    # static

# 修改 DNS
nmcli con mod mycon ipv4.dns "8.8.8.8"
nmcli con mod mycon +ipv4.dns "8.8.4.4"   # 添加额外DNS

# 修改为 DHCP
nmcli con mod mycon ipv4.method auto

# ============================================================
#                   激活连接
# ============================================================

# 启用连接
nmcli con up mycon

# 禁用连接
nmcli con down mycon

# 重新加载连接
nmcli con reload

# ============================================================
#                   删除连接
# ============================================================

# 删除连接
nmcli con delete mycon

# ============================================================
#                   WiFi 管理
# ============================================================

# 扫描 WiFi
nmcli dev wifi list

# 连接 WiFi
nmcli dev wifi connect SSID password PASSWORD

# 连接隐藏 WiFi
nmcli con add \
    type wifi \
    con-name mywifi \
    ifname wlan0 \
    ssid SSID \
    wifi-sec.key-mgmt wpa-psk \
    wifi-sec.psk PASSWORD

# 断开 WiFi
nmcli dev disconnect wlan0

# 启用/禁用 WiFi
nmcli radio wifi on
nmcli radio wifi off
```

## 三、路由管理

### 路由表

```bash
# ============================================================
#                   查看路由
# ============================================================

# 使用 ip 命令 (推荐)
ip route show
ip route list
ip r                           # 简写

# 查看特定目的地的路由
ip route get 8.8.8.8

# 使用 route 命令 (已弃用)
route -n                       # -n 显示数字地址

# 使用 netstat
netstat -rn

# ============================================================
#                   添加路由
# ============================================================

# 添加默认网关
sudo ip route add default via 192.168.1.1 dev eth0

# 添加网络路由
sudo ip route add 10.0.0.0/8 via 192.168.1.254 dev eth0

# 添加主机路由
sudo ip route add 10.0.0.5 via 192.168.1.254 dev eth0

# 添加直连路由
sudo ip route add 192.168.2.0/24 dev eth1

# ============================================================
#                   删除路由
# ============================================================

# 删除默认网关
sudo ip route del default

# 删除网络路由
sudo ip route del 10.0.0.0/8

# 删除主机路由
sudo ip route del 10.0.0.5

# ============================================================
#                   修改路由
# ============================================================

# 修改默认网关
sudo ip route replace default via 192.168.1.2 dev eth0

# 修改网络路由
sudo ip route replace 10.0.0.0/8 via 192.168.1.253 dev eth0

# ============================================================
#                   策略路由
# ============================================================

# 查看路由表
ip rule list

# 添加策略路由
sudo ip rule add from 192.168.1.0/24 table 100
sudo ip route add default via 192.168.2.1 table 100

# 删除策略路由
sudo ip rule del from 192.168.1.0/24 table 100

# 刷新路由缓存
sudo ip route flush cache

# ============================================================
#                   永久路由 (Debian/Ubuntu)
# ============================================================

# 编辑 /etc/network/interfaces
auto eth0
iface eth0 inet static
    address 192.168.1.100
    netmask 255.255.255.0
    gateway 192.168.1.1
    up ip route add 10.0.0.0/8 via 192.168.1.254
    down ip route del 10.0.0.0/8

# ============================================================
#                   永久路由 (CentOS/RHEL)
# ============================================================

# 创建 /etc/sysconfig/network-scripts/route-eth0
10.0.0.0/8 via 192.168.1.254
```

## 四、DNS

### DNS 配置

```bash
# ============================================================
#                   配置 DNS 服务器
# ============================================================

# 编辑 /etc/resolv.conf (临时,重启后丢失)
nameserver 8.8.8.8
nameserver 8.8.4.4
search example.com
options timeout:2 attempts:3

# ────────────────────────────────────────────────────────────

# systemd-resolved (Ubuntu 18.04+)
# 编辑 /etc/systemd/resolved.conf
[Resolve]
DNS=8.8.8.8 8.8.4.4
FallbackDNS=1.1.1.1
Domains=example.com
DNSSEC=no

# 重启服务
sudo systemctl restart systemd-resolved

# 查看状态
resolvectl status

# ────────────────────────────────────────────────────────────

# NetworkManager
nmcli con mod mycon ipv4.dns "8.8.8.8 8.8.4.4"
nmcli con up mycon

# ============================================================
#                   DNS 查询工具
# ============================================================

# dig (推荐)
dig example.com                # 查询 A 记录
dig example.com MX             # 查询 MX 记录
dig example.com NS             # 查询 NS 记录
dig example.com TXT            # 查询 TXT 记录
dig example.com ANY            # 查询所有记录
dig @8.8.8.8 example.com       # 使用指定 DNS 服务器
dig +short example.com         # 简洁输出
dig +trace example.com         # 追踪解析过程
dig -x 8.8.8.8                 # 反向解析

# nslookup
nslookup example.com
nslookup example.com 8.8.8.8   # 指定 DNS 服务器

# host
host example.com
host -t MX example.com         # 查询特定记录类型
host -a example.com            # 查询所有记录

# ============================================================
#                   DNS 缓存
# ============================================================

# systemd-resolved 缓存
# 查看缓存统计
resolvectl statistics

# 清空缓存
sudo resolvectl flush-caches

# ────────────────────────────────────────────────────────────

# nscd (Name Service Cache Daemon)
sudo systemctl restart nscd    # 重启清空缓存

# ────────────────────────────────────────────────────────────

# dnsmasq
sudo systemctl restart dnsmasq
```

## 五、网络诊断

### 连通性测试

```bash
# ============================================================
#                   ping
# ============================================================

# 基本 ping
ping google.com
ping -c 4 google.com           # 只ping 4次
ping -i 0.5 google.com         # 间隔0.5秒
ping -W 2 google.com           # 超时时间2秒
ping -s 1000 google.com        # 数据包大小1000字节

# 快速 ping (需要 root)
sudo ping -f google.com        # flood ping

# IPv6 ping
ping6 google.com

# ============================================================
#                   traceroute / tracepath
# ============================================================

# 追踪路由
traceroute google.com
traceroute -n google.com       # 不解析主机名
traceroute -m 15 google.com    # 最大跳数15

# tracepath (不需要 root)
tracepath google.com

# mtr (结合 ping 和 traceroute)
sudo apt install mtr
mtr google.com                 # 实时追踪
mtr -r google.com              # 报告模式

# ============================================================
#                   telnet / nc (netcat)
# ============================================================

# 测试端口连通性
telnet google.com 80
nc -zv google.com 80           # -z 扫描, -v 详细
nc -zv google.com 80-443       # 扫描端口范围

# 监听端口
nc -l 8080                     # 监听 8080 端口

# 文件传输
# 接收端:
nc -l 8080 > received_file

# 发送端:
nc target_host 8080 < file_to_send

# ============================================================
#                   curl / wget
# ============================================================

# 测试 HTTP 连接
curl -I https://google.com     # 只获取头部
curl -v https://google.com     # 详细输出
curl -o /dev/null -s -w "Time: %{time_total}s\n" https://google.com

# 测试 HTTPS 证书
curl -vI https://google.com 2>&1 | grep -A 5 "SSL certificate"

# 下载文件
wget https://example.com/file.tar.gz
wget -c https://example.com/file.tar.gz  # 断点续传
wget -O output.txt https://example.com   # 指定输出文件名
```

### 端口扫描

```bash
# ============================================================
#                   查看本地端口
# ============================================================

# 使用 ss (推荐)
ss -tuln                       # TCP/UDP, 监听端口, 数字格式
ss -tulpn                      # 包含进程信息
ss -antp                       # 所有 TCP 连接,包含进程

# 参数说明:
# -t: TCP
# -u: UDP
# -l: 监听端口
# -n: 数字格式 (不解析)
# -p: 显示进程
# -a: 所有连接
# -4: IPv4
# -6: IPv6

# 常用组合:
ss -tulpn                      # 监听端口 + 进程
ss -antp                       # 所有TCP连接 + 进程
ss -s                          # 统计信息

# 过滤:
ss state established           # 已建立的连接
ss dst 192.168.1.10            # 目标地址
ss sport = :80                 # 源端口
ss dport = :80                 # 目标端口

# ────────────────────────────────────────────────────────────

# 使用 netstat (已弃用,但常用)
netstat -tuln                  # 监听端口
netstat -tulpn                 # 包含进程
netstat -antp                  # 所有 TCP 连接

# 常用组合:
netstat -tulpn | grep LISTEN   # 监听端口
netstat -antp | grep ESTABLISHED  # 已建立连接

# ────────────────────────────────────────────────────────────

# 使用 lsof
lsof -i                        # 所有网络连接
lsof -i :80                    # 端口 80
lsof -i TCP                    # TCP 连接
lsof -i UDP                    # UDP 连接

# ============================================================
#                   扫描远程端口
# ============================================================

# nmap (需要安装)
sudo apt install nmap

# 基本扫描
nmap 192.168.1.10              # 扫描常用端口
nmap -p 1-65535 192.168.1.10   # 扫描所有端口
nmap -p 80,443 192.168.1.10    # 扫描特定端口
nmap -p- 192.168.1.10          # 同 -p 1-65535

# 扫描类型
nmap -sT 192.168.1.10          # TCP 连接扫描
nmap -sS 192.168.1.10          # SYN 扫描 (需要 root)
nmap -sU 192.168.1.10          # UDP 扫描
nmap -sP 192.168.1.0/24        # Ping 扫描 (主机发现)

# 服务版本检测
nmap -sV 192.168.1.10          # 检测服务版本
nmap -O 192.168.1.10           # 检测操作系统 (需要 root)
nmap -A 192.168.1.10           # 综合扫描 (OS, 版本, 脚本, traceroute)

# 快速扫描
nmap -F 192.168.1.10           # 快速扫描 (100个常用端口)
nmap -T4 192.168.1.10          # 加快速度 (T0-T5)

# 扫描整个网段
nmap 192.168.1.0/24            # 扫描整个C类网段
nmap 192.168.1.1-254           # 扫描范围

# 输出到文件
nmap -oN output.txt 192.168.1.10    # 普通格式
nmap -oX output.xml 192.168.1.10    # XML 格式

# 示例:快速扫描网段内活跃主机
nmap -sP 192.168.1.0/24

# 示例:扫描 Web 服务器
nmap -p 80,443,8080 -sV 192.168.1.10
```

### 网络流量

```bash
# ============================================================
#                   tcpdump - 抓包
# ============================================================

# 基本抓包
sudo tcpdump                   # 抓取所有流量
sudo tcpdump -i eth0           # 指定接口
sudo tcpdump -i any            # 所有接口

# 过滤
sudo tcpdump host 192.168.1.10             # 特定主机
sudo tcpdump src 192.168.1.10              # 源地址
sudo tcpdump dst 192.168.1.10              # 目标地址
sudo tcpdump net 192.168.1.0/24            # 网段
sudo tcpdump port 80                       # 端口
sudo tcpdump portrange 80-443              # 端口范围

# 协议过滤
sudo tcpdump tcp                           # TCP
sudo tcpdump udp                           # UDP
sudo tcpdump icmp                          # ICMP

# 组合过滤
sudo tcpdump 'host 192.168.1.10 and port 80'
sudo tcpdump 'src 192.168.1.10 or dst 192.168.1.20'
sudo tcpdump 'tcp and not port 22'

# 输出选项
sudo tcpdump -n                # 不解析主机名
sudo tcpdump -nn               # 不解析主机名和端口
sudo tcpdump -v                # 详细输出
sudo tcpdump -vv               # 更详细
sudo tcpdump -X                # 显示十六进制和 ASCII
sudo tcpdump -A                # 只显示 ASCII
sudo tcpdump -c 100            # 抓取100个数据包后停止

# 保存到文件
sudo tcpdump -w capture.pcap
sudo tcpdump -r capture.pcap   # 读取文件

# 实用示例:
# 抓取 HTTP 流量
sudo tcpdump -i eth0 -A 'tcp port 80'

# 抓取 DNS 查询
sudo tcpdump -i any -nn 'udp port 53'

# 抓取 SSH 连接
sudo tcpdump -i any 'tcp port 22'

# ============================================================
#                   iftop - 实时流量监控
# ============================================================

sudo apt install iftop
sudo iftop                     # 实时显示连接和流量
sudo iftop -i eth0             # 指定接口
sudo iftop -n                  # 不解析主机名
sudo iftop -P                  # 显示端口
sudo iftop -f "port 80"        # 过滤

# iftop 快捷键:
#   t: 切换显示模式
#   n: 切换主机名/IP
#   p: 显示/隐藏端口
#   P: 暂停显示
#   q: 退出

# ============================================================
#                   nethogs - 按进程监控流量
# ============================================================

sudo apt install nethogs
sudo nethogs                   # 按进程显示流量
sudo nethogs eth0              # 指定接口

# ============================================================
#                   bmon - 带宽监控
# ============================================================

sudo apt install bmon
bmon                           # 图形化带宽监控

# ============================================================
#                   vnstat - 流量统计
# ============================================================

sudo apt install vnstat

# 安装后需要初始化
sudo vnstat -u -i eth0         # 初始化接口

# 查看统计
vnstat                         # 月度统计
vnstat -d                      # 日统计
vnstat -m                      # 月统计
vnstat -h                      # 小时统计
vnstat -l                      # 实时监控
vnstat -t                      # 前10天

# ============================================================
#                   查看连接统计
# ============================================================

# 统计各种状态的连接数
ss -s
netstat -s                     # 详细统计

# 统计每个 IP 的连接数
ss -antp | awk '{print $5}' | cut -d: -f1 | sort | uniq -c | sort -rn

# 统计 ESTABLISHED 连接数
ss -antp | grep ESTABLISHED | wc -l

# 查看最多连接的 IP
ss -antp | grep ESTABLISHED | awk '{print $5}' | cut -d: -f1 | sort | uniq -c | sort -rn | head
```

这是网络管理教程的主要内容,涵盖了网络配置、路由、DNS、诊断和流量监控。

接下来让我创建系统安全教程和速查手册。
