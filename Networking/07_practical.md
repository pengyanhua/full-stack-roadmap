# 网络实践与工具

## 一、Wireshark 抓包分析

```
Wireshark - 网络协议分析器:

基本操作:
1. 选择网卡
2. 开始捕获
3. 应用过滤器
4. 分析数据包

常用显示过滤器:
# 协议过滤
tcp                        # TCP 数据包
udp                        # UDP 数据包
icmp                       # ICMP 数据包
http                       # HTTP 数据包
dns                        # DNS 查询
tls                        # TLS/SSL 数据包

# IP 地址过滤
ip.addr == 192.168.1.10    # 源或目标
ip.src == 192.168.1.10     # 源地址
ip.dst == 192.168.1.10     # 目标地址
ip.addr == 192.168.1.0/24  # 网段

# 端口过滤
tcp.port == 80             # TCP 端口 80
tcp.srcport == 80          # 源端口
tcp.dstport == 80          # 目标端口
tcp.port == 80 || tcp.port == 443  # 多端口

# 组合过滤
ip.addr == 192.168.1.10 && tcp.port == 80
tcp.flags.syn == 1 && tcp.flags.ack == 0    # SYN 包
tcp.analysis.retransmission                  # TCP 重传

# HTTP 过滤
http.request.method == "GET"
http.request.uri contains "api"
http.response.code == 200
http.host == "example.com"

# 内容过滤
frame contains "password"
tcp contains "error"

捕获过滤器 (BPF 语法):
# 主机
host 192.168.1.10
src host 192.168.1.10
dst host 192.168.1.10

# 网络
net 192.168.1.0/24
net 192.168.1.0 mask 255.255.255.0

# 端口
port 80
src port 1234
dst port 80
portrange 80-443

# 协议
tcp
udp
icmp

# 组合
host 192.168.1.10 and port 80
tcp port 80 or tcp port 443

分析 TCP 三次握手:
┌────────────────────────────────────────────────────────────┐
│ No. │ Time │ Source      │ Destination │ Protocol │ Info   │
├─────┼──────┼─────────────┼─────────────┼──────────┼────────┤
│ 1   │ 0.0  │192.168.1.10 │192.168.1.20 │   TCP    │[SYN]   │
│     │      │             │             │          │Seq=0   │
│ 2   │ 0.01 │192.168.1.20 │192.168.1.10 │   TCP    │[SYN,ACK│
│     │      │             │             │          │Seq=0   │
│     │      │             │             │          │Ack=1   │
│ 3   │ 0.02 │192.168.1.10 │192.168.1.20 │   TCP    │[ACK]   │
│     │      │             │             │          │Seq=1   │
│     │      │             │             │          │Ack=1   │
└─────┴──────┴─────────────┴─────────────┴──────────┴────────┘

分析 HTTP 请求:
1. 找到 HTTP GET/POST 包
2. 查看请求头 (右键 → Follow → HTTP Stream)
3. 查看响应内容
4. 分析时间 (Time to First Byte)

分析 DNS 查询:
1. 过滤: dns
2. 查看查询 (Query) 和响应 (Response)
3. 检查响应时间
4. 识别 DNS 劫持 (响应 IP 异常)

统计功能:
├─ Statistics → Protocol Hierarchy (协议分布)
├─ Statistics → Conversations (会话统计)
├─ Statistics → Endpoints (端点统计)
├─ Statistics → I/O Graphs (流量图表)
└─ Analyze → Expert Information (专家信息)

导出对象:
File → Export Objects → HTTP
# 可导出 HTTP 传输的文件 (图片、CSS、JS等)

命令行版本 (tshark):
# 捕获并保存
tshark -i eth0 -w capture.pcap

# 显示特定协议
tshark -r capture.pcap -Y http

# 统计
tshark -r capture.pcap -q -z io,phs

# 提取字段
tshark -r capture.pcap -T fields -e ip.src -e ip.dst -e tcp.port
```

## 二、tcpdump 命令行抓包

```
tcpdump - 命令行数据包分析工具:

基本语法:
tcpdump [选项] [表达式]

常用选项:
-i eth0               # 指定网卡
-c 100                # 捕获 100 个包后停止
-w file.pcap          # 保存到文件
-r file.pcap          # 读取文件
-n                    # 不解析主机名
-nn                   # 不解析主机名和端口
-v, -vv, -vvv         # 详细输出
-A                    # ASCII 格式显示
-X                    # 十六进制和 ASCII 显示
-s 0                  # 捕获完整数据包 (默认68字节)
-q                    # 简洁输出

过滤表达式:
# 主机
tcpdump host 192.168.1.10
tcpdump src host 192.168.1.10
tcpdump dst host 192.168.1.10

# 网络
tcpdump net 192.168.1.0/24

# 端口
tcpdump port 80
tcpdump src port 1234
tcpdump dst port 80
tcpdump portrange 80-443

# 协议
tcpdump tcp
tcpdump udp
tcpdump icmp

# 组合条件
tcpdump host 192.168.1.10 and port 80
tcpdump tcp and dst port 80
tcpdump 'tcp[tcpflags] & tcp-syn != 0'  # SYN 包

实用示例:
# 抓取 HTTP 流量
tcpdump -i eth0 -A -s 0 'tcp port 80 and (((ip[2:2] - ((ip[0]&0xf)<<2)) - ((tcp[12]&0xf0)>>2)) != 0)'

# 抓取 HTTPS 握手
tcpdump -i eth0 -nn 'tcp port 443 and (tcp[((tcp[12]&0xf0)>>2)] = 0x16)'

# 抓取 DNS 查询
tcpdump -i eth0 -nn 'udp port 53'

# 抓取 ICMP (ping)
tcpdump -i eth0 icmp

# 抓取特定 IP 的所有流量
tcpdump -i eth0 host 192.168.1.10

# 抓取 SYN 包
tcpdump -i eth0 'tcp[tcpflags] & tcp-syn != 0'

# 抓取 HTTP GET 请求
tcpdump -i eth0 -A -s 0 'tcp dst port 80 and tcp[((tcp[12]&0xf0)>>2):4] = 0x47455420'

# 保存抓包
tcpdump -i eth0 -w capture.pcap

# 读取并过滤
tcpdump -r capture.pcap port 80

# 抓取特定大小的包
tcpdump -i eth0 'greater 1000'  # 大于 1000 字节
tcpdump -i eth0 'less 100'      # 小于 100 字节

# 抓取特定 MAC 地址
tcpdump -i eth0 ether host aa:bb:cc:dd:ee:ff

# 抓取非 SSH 流量 (避免干扰)
tcpdump -i eth0 not port 22

# 抓取到多个文件 (轮转)
tcpdump -i eth0 -w file.pcap -C 100 -W 5
# -C 100: 每 100MB 一个文件
# -W 5: 保留 5 个文件

# 实时显示 HTTP Host
tcpdump -i eth0 -n -s 0 -A tcp port 80 | grep -i 'Host:'
```

## 三、网络诊断工具

```
常用网络诊断工具:

1. ping - 连通性测试
ping -c 4 google.com           # 发送 4 个包
ping -i 0.5 google.com         # 间隔 0.5 秒
ping -s 1000 google.com        # 包大小 1000 字节
ping -W 2 google.com           # 超时 2 秒

2. traceroute / tracepath - 路由追踪
traceroute google.com
traceroute -I google.com       # 使用 ICMP
traceroute -T -p 80 google.com # 使用 TCP 端口 80
tracepath google.com           # 不需要 root
mtr google.com                 # 实时追踪

3. nslookup / dig / host - DNS 查询
nslookup google.com
dig google.com
dig @8.8.8.8 google.com        # 指定 DNS 服务器
dig +short google.com          # 简洁输出
dig +trace google.com          # 追踪解析过程
host google.com

4. netstat / ss - 网络统计
netstat -tuln                  # 监听端口
netstat -antp                  # 所有 TCP 连接
netstat -s                     # 统计信息
netstat -r                     # 路由表

ss -tuln                       # 监听端口 (更快)
ss -antp                       # 所有 TCP 连接
ss -s                          # 统计
ss state established           # 已建立的连接

5. lsof - 列出打开的文件
lsof -i                        # 所有网络连接
lsof -i :80                    # 端口 80
lsof -i TCP                    # TCP 连接
lsof -i UDP                    # UDP 连接
lsof -u username               # 用户打开的文件

6. nc (netcat) - 网络瑞士军刀
# 端口扫描
nc -zv google.com 80-443

# 监听端口
nc -l 8080

# 连接测试
nc -v google.com 80
# 输入: GET / HTTP/1.1

# 文件传输
# 接收端:
nc -l 8080 > received_file
# 发送端:
nc target_host 8080 < file_to_send

# Banner 抓取
nc -v target 22

7. nmap - 网络扫描
# 端口扫描
nmap 192.168.1.10
nmap -p 1-65535 192.168.1.10   # 全端口
nmap -p 80,443 192.168.1.10    # 特定端口

# 扫描类型
nmap -sT 192.168.1.10          # TCP 连接扫描
nmap -sS 192.168.1.10          # SYN 扫描 (需要 root)
nmap -sU 192.168.1.10          # UDP 扫描
nmap -sP 192.168.1.0/24        # Ping 扫描

# 服务版本检测
nmap -sV 192.168.1.10

# OS 检测
nmap -O 192.168.1.10

# 综合扫描
nmap -A 192.168.1.10

# 快速扫描
nmap -F 192.168.1.10

# 扫描网段
nmap 192.168.1.0/24

8. iperf3 - 带宽测试
# 服务器端:
iperf3 -s

# 客户端:
iperf3 -c server_ip
iperf3 -c server_ip -t 30      # 测试 30 秒
iperf3 -c server_ip -R         # 反向测试 (下载)
iperf3 -c server_ip -u -b 100M # UDP 测试,100Mbps

9. curl / wget - HTTP 客户端
# curl
curl https://example.com
curl -I https://example.com    # 只获取头部
curl -v https://example.com    # 详细输出
curl -o file.html https://example.com  # 保存到文件
curl -L https://example.com    # 跟随重定向
curl -X POST -d "data" https://api.example.com  # POST 请求
curl -H "Authorization: Bearer token" https://api.example.com  # 自定义头部

# wget
wget https://example.com/file.tar.gz
wget -c https://example.com/file.tar.gz  # 断点续传
wget -O output.txt https://example.com   # 指定文件名
wget -r https://example.com              # 递归下载

10. iftop / nethogs - 流量监控
# iftop - 实时流量监控
sudo iftop
sudo iftop -i eth0
sudo iftop -n                  # 不解析主机名

# nethogs - 按进程显示流量
sudo nethogs
sudo nethogs eth0
```

## 四、故障排查流程

```
网络故障排查步骤:

1. 物理层检查:
├─ 检查网线是否插好
├─ 检查网卡灯是否亮
├─ 检查网卡状态: ip link show
└─ 检查交换机/路由器状态

2. 数据链路层检查:
├─ 检查 MAC 地址: ip link show
├─ 检查 ARP 缓存: arp -a
├─ 检查交换机端口: show mac address-table
└─ 检查 VLAN 配置

3. 网络层检查:
├─ 检查 IP 配置: ip addr show
├─ ping 网关: ping 192.168.1.1
├─ ping 外网: ping 8.8.8.8
├─ 检查路由: ip route show
└─ traceroute 追踪路径

4. 传输层检查:
├─ 检查端口监听: ss -tuln
├─ 检查连接状态: ss -antp
├─ 检查防火墙: iptables -L
└─ telnet/nc 测试端口: nc -zv host port

5. 应用层检查:
├─ DNS 解析: dig example.com
├─ HTTP 测试: curl -v https://example.com
├─ 检查服务日志: journalctl -u service
└─ 检查应用配置

常见问题排查:

问题: 无法访问网站
├─ ping IP 成功,ping 域名失败 → DNS 问题
│  └─ 检查 /etc/resolv.conf
│  └─ 尝试其他 DNS: dig @8.8.8.8 example.com
│
├─ ping 网关失败 → 本地网络问题
│  └─ 检查 IP 配置
│  └─ 检查网线/网卡
│
├─ ping 外网失败 → 路由问题
│  └─ 检查默认网关
│  └─ traceroute 追踪
│
└─ ping 成功,HTTP 失败 → 应用层问题
   └─ 检查防火墙
   └─ 检查服务状态

问题: 网络慢
├─ ping 延迟高 → 网络拥塞或链路问题
│  └─ mtr 分析每跳延迟
│
├─ 带宽不足 → iftop 查看流量
│  └─ nethogs 找出占用带宽的进程
│
└─ 丢包严重 → 链路质量问题
   └─ ping -c 100 检查丢包率
   └─ 检查网线/网卡

问题: 间歇性断网
├─ 检查系统日志: journalctl -f
├─ 检查网卡日志: dmesg | grep eth0
├─ 检查 DHCP 租约: cat /var/lib/dhcp/dhclient.leases
└─ 监控连接状态: watch -n 1 'ping -c 1 8.8.8.8'

抓包分析:
# 抓取问题发生时的数据包
tcpdump -i any -w problem.pcap

# 用 Wireshark 分析
wireshark problem.pcap

# 查找异常:
├─ TCP 重传 (tcp.analysis.retransmission)
├─ 连接超时
├─ RST 包 (tcp.flags.reset == 1)
└─ ICMP 错误 (icmp.type == 3)
```

## 五、性能优化

```
网络性能优化:

1. TCP 参数调优 (/etc/sysctl.conf):
# TCP 缓冲区
net.core.rmem_max = 134217728          # 最大接收缓冲 128MB
net.core.wmem_max = 134217728          # 最大发送缓冲 128MB
net.ipv4.tcp_rmem = 4096 87380 67108864  # TCP 接收缓冲
net.ipv4.tcp_wmem = 4096 65536 67108864  # TCP 发送缓冲

# TCP 连接队列
net.core.somaxconn = 1024              # 最大连接队列
net.ipv4.tcp_max_syn_backlog = 2048    # SYN 队列

# TIME_WAIT 优化
net.ipv4.tcp_tw_reuse = 1              # 重用 TIME_WAIT
net.ipv4.tcp_fin_timeout = 30          # FIN_WAIT 超时

# TCP 拥塞控制
net.ipv4.tcp_congestion_control = bbr  # 使用 BBR 算法

# TCP keepalive
net.ipv4.tcp_keepalive_time = 600      # 开始发送 keepalive 的时间
net.ipv4.tcp_keepalive_intvl = 60      # keepalive 间隔
net.ipv4.tcp_keepalive_probes = 3      # keepalive 探测次数

# 应用更改
sysctl -p

2. 网卡优化:
# 查看网卡参数
ethtool eth0

# 关闭自动协商 (千兆网络)
ethtool -s eth0 speed 1000 duplex full autoneg off

# 调整 Ring Buffer
ethtool -g eth0                        # 查看当前值
ethtool -G eth0 rx 4096 tx 4096        # 设置

# 启用 TCP 分段卸载 (TSO)
ethtool -K eth0 tso on

# 启用大接收卸载 (LRO)
ethtool -K eth0 lro on

3. 并发连接优化:
# 增加文件描述符限制
ulimit -n 65535

# 永久修改 (/etc/security/limits.conf)
* soft nofile 65535
* hard nofile 65535

4. DNS 优化:
# 使用本地 DNS 缓存
sudo apt install dnsmasq

# 配置 dnsmasq (/etc/dnsmasq.conf)
cache-size=10000
no-resolv
server=8.8.8.8
server=8.8.4.4

5. CDN 加速:
├─ 使用 CDN 加速静态资源
├─ 启用 HTTP/2
├─ 启用 gzip/brotli 压缩
└─ 使用 Keep-Alive

6. 负载均衡:
├─ 使用 Nginx / HAProxy
├─ DNS 负载均衡
└─ 云负载均衡 (AWS ELB, GCP LB)

性能测试工具:
# Apache Bench
ab -n 1000 -c 100 http://example.com/

# wrk (更强大)
wrk -t 12 -c 400 -d 30s http://example.com/

# siege
siege -c 100 -t 30s http://example.com/

# 带宽测试
iperf3 -c server_ip -t 60 -P 10

# 延迟测试
ping -c 100 server_ip

# HTTP 性能测试
curl -w "@curl-format.txt" -o /dev/null -s https://example.com/

# curl-format.txt 内容:
time_namelookup:  %{time_namelookup}s\n
time_connect:  %{time_connect}s\n
time_appconnect:  %{time_appconnect}s\n
time_pretransfer:  %{time_pretransfer}s\n
time_redirect:  %{time_redirect}s\n
time_starttransfer:  %{time_starttransfer}s\n
time_total:  %{time_total}s\n
```

这是网络实践与工具教程,涵盖了Wireshark、tcpdump、诊断工具、故障排查和性能优化。
