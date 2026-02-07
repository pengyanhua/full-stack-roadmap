# 应用层协议

## 一、HTTP/HTTPS

### HTTP 协议

```
HTTP (HyperText Transfer Protocol) 超文本传输协议:

HTTP 请求格式:
┌────────────────────────────────────────────────────────────┐
│ 请求行                                                      │
│ GET /index.html HTTP/1.1                                   │
├────────────────────────────────────────────────────────────┤
│ 请求头 (Headers)                                           │
│ Host: www.example.com                                      │
│ User-Agent: Mozilla/5.0                                    │
│ Accept: text/html                                          │
│ Connection: keep-alive                                     │
├────────────────────────────────────────────────────────────┤
│ 空行                                                        │
├────────────────────────────────────────────────────────────┤
│ 请求体 (Body, 可选)                                        │
│ username=admin&password=123456                             │
└────────────────────────────────────────────────────────────┘

HTTP 响应格式:
┌────────────────────────────────────────────────────────────┐
│ 状态行                                                      │
│ HTTP/1.1 200 OK                                            │
├────────────────────────────────────────────────────────────┤
│ 响应头 (Headers)                                           │
│ Content-Type: text/html; charset=UTF-8                     │
│ Content-Length: 1234                                       │
│ Server: nginx/1.18.0                                       │
│ Date: Mon, 01 Jan 2024 12:00:00 GMT                       │
├────────────────────────────────────────────────────────────┤
│ 空行                                                        │
├────────────────────────────────────────────────────────────┤
│ 响应体 (Body)                                              │
│ <!DOCTYPE html>                                            │
│ <html>...</html>                                           │
└────────────────────────────────────────────────────────────┘

HTTP 方法:
├─ GET: 获取资源
├─ POST: 提交数据
├─ PUT: 更新资源 (完整)
├─ PATCH: 更新资源 (部分)
├─ DELETE: 删除资源
├─ HEAD: 获取头部信息
├─ OPTIONS: 查询支持的方法
└─ TRACE: 回显请求 (调试用)

状态码分类:
├─ 1xx 信息性: 100 Continue, 101 Switching Protocols
├─ 2xx 成功: 200 OK, 201 Created, 204 No Content
├─ 3xx 重定向: 301 Moved Permanently, 302 Found, 304 Not Modified
├─ 4xx 客户端错误: 400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found
└─ 5xx 服务器错误: 500 Internal Server Error, 502 Bad Gateway, 503 Service Unavailable

常用请求头:
├─ Host: 目标主机 (必需)
├─ User-Agent: 客户端信息
├─ Accept: 接受的内容类型
├─ Accept-Language: 接受的语言
├─ Accept-Encoding: 接受的编码 (gzip, deflate)
├─ Cookie: Cookie 数据
├─ Referer: 来源页面
├─ Authorization: 认证信息
└─ Content-Type: 请求体类型

常用响应头:
├─ Content-Type: 响应体类型
├─ Content-Length: 响应体长度
├─ Set-Cookie: 设置 Cookie
├─ Cache-Control: 缓存控制
├─ ETag: 资源标识
├─ Location: 重定向地址
└─ Server: 服务器信息

HTTP/1.0 vs HTTP/1.1 vs HTTP/2:
┌────────────────┬──────────┬──────────┬──────────┐
│      特性      │ HTTP/1.0 │ HTTP/1.1 │ HTTP/2   │
├────────────────┼──────────┼──────────┼──────────┤
│   连接         │ 短连接   │ 长连接   │ 多路复用 │
│   管道化       │ 不支持   │ 支持     │ 原生支持 │
│   头部压缩     │ 无       │ 无       │ HPACK    │
│   服务器推送   │ 无       │ 无       │ 支持     │
│   二进制协议   │ 文本     │ 文本     │ 二进制   │
└────────────────┴──────────┴──────────┴──────────┘

HTTP/2 特性:
├─ 二进制分帧: 数据以帧为单位传输
├─ 多路复用: 单连接并行请求
├─ 头部压缩: HPACK 算法
├─ 服务器推送: 主动推送资源
└─ 流优先级: 关键资源优先

HTTP/3 (基于 QUIC):
├─ 基于 UDP
├─ 更快的连接建立
├─ 更好的拥塞控制
└─ 连接迁移 (IP 变化不断开)
```

### HTTPS

```
HTTPS = HTTP + TLS/SSL

HTTPS 握手过程:
客户端                                         服务器
   │                                              │
   │ [1] ClientHello                              │
   │  ├─ 支持的 TLS 版本                          │
   │  ├─ 支持的加密套件                           │
   │  └─ 随机数 (Client Random)                  │
   │──────────────────────────────────────────►   │
   │                                              │
   │ [2] ServerHello                              │
   │  ├─ 选择的 TLS 版本                          │
   │  ├─ 选择的加密套件                           │
   │  ├─ 随机数 (Server Random)                  │
   │  ├─ 服务器证书                               │
   │  └─ ServerHelloDone                          │
   │ ◄──────────────────────────────────────────  │
   │                                              │
   │ [3] 客户端验证证书                           │
   │                                              │
   │ [4] ClientKeyExchange                        │
   │  └─ 预主密钥 (用服务器公钥加密)              │
   │──────────────────────────────────────────►   │
   │                                              │
   │ [5] ChangeCipherSpec + Finished              │
   │──────────────────────────────────────────►   │
   │                                              │
   │ [6] ChangeCipherSpec + Finished              │
   │ ◄──────────────────────────────────────────  │
   │                                              │
   │ ◄═════════ 加密的应用数据传输 ══════════════► │

密钥生成:
├─ Client Random (客户端生成)
├─ Server Random (服务器生成)
├─ Pre-Master Secret (客户端生成,用服务器公钥加密传输)
└─ Master Secret = PRF(Pre-Master, Client Random, Server Random)
   ├─ 客户端加密密钥
   ├─ 服务器加密密钥
   ├─ 客户端 MAC 密钥
   └─ 服务器 MAC 密钥

证书链验证:
服务器证书 ← 中间CA证书 ← 根CA证书 (浏览器内置)

HTTPS 优势:
├─ 数据加密: 防止窃听
├─ 完整性: 防止篡改
├─ 身份验证: 防止冒充
└─ SEO 优势: Google 排名加分

常见端口:
├─ HTTP: 80
└─ HTTPS: 443
```

## 二、DNS

```
DNS (Domain Name System) 域名系统:

DNS 查询流程:
┌──────────────────────────────────────────────────────────┐
│  1. 客户端查询: www.example.com                          │
│     │                                                    │
│     ▼                                                    │
│  2. 检查本地缓存 (浏览器/系统)                           │
│     │ 未命中                                             │
│     ▼                                                    │
│  3. 递归查询 DNS 服务器 (如 8.8.8.8)                    │
│     │                                                    │
│     ├──► 4. 查询根 DNS (.): 返回 .com DNS 地址          │
│     │                                                    │
│     ├──► 5. 查询 .com DNS: 返回 example.com DNS 地址    │
│     │                                                    │
│     └──► 6. 查询 example.com DNS: 返回 www.example.com IP│
│                                                          │
│  7. 返回 IP 地址给客户端: 93.184.216.34                 │
└──────────────────────────────────────────────────────────┘

DNS 记录类型:
├─ A: IPv4 地址
├─ AAAA: IPv6 地址
├─ CNAME: 别名 (规范名称)
├─ MX: 邮件服务器
├─ NS: 域名服务器
├─ TXT: 文本记录 (SPF, DKIM等)
├─ PTR: 反向解析 (IP → 域名)
├─ SOA: 授权起始
└─ SRV: 服务定位

DNS 查询命令:
# dig (推荐)
dig example.com
dig example.com MX
dig @8.8.8.8 example.com      # 指定DNS服务器
dig +short example.com         # 简洁输出
dig +trace example.com         # 追踪解析过程

# nslookup
nslookup example.com
nslookup example.com 8.8.8.8

# host
host example.com
host -t MX example.com

DNS 缓存:
├─ 浏览器缓存: 几分钟
├─ 系统缓存: OS DNS 缓存
├─ DNS 服务器缓存: 根据 TTL
└─ 清除缓存:
   ├─ Windows: ipconfig /flushdns
   ├─ Linux: sudo systemd-resolve --flush-caches
   └─ macOS: sudo dscacheutil -flushcache

DNS 安全:
├─ DNSSEC: DNS 安全扩展 (防止劫持)
├─ DNS over HTTPS (DoH): 加密 DNS 查询
└─ DNS over TLS (DoT): TLS 加密

常见 DNS 服务器:
├─ Google: 8.8.8.8, 8.8.4.4
├─ Cloudflare: 1.1.1.1, 1.0.0.1
├─ Quad9: 9.9.9.9
└─ OpenDNS: 208.67.222.222, 208.67.220.220
```

## 三、FTP

```
FTP (File Transfer Protocol) 文件传输协议:

工作模式:
1. 主动模式 (Active):
   ├─ 客户端: 随机端口 → 服务器:21 (控制连接)
   └─ 服务器:20 → 客户端:随机端口 (数据连接)

2. 被动模式 (Passive, 推荐):
   ├─ 客户端: 随机端口 → 服务器:21 (控制连接)
   └─ 客户端: 随机端口 → 服务器:随机端口 (数据连接)

FTP 命令:
├─ USER: 用户名
├─ PASS: 密码
├─ LIST: 列出目录
├─ RETR: 下载文件
├─ STOR: 上传文件
├─ DELE: 删除文件
├─ MKD: 创建目录
├─ PWD: 当前目录
└─ QUIT: 退出

使用 FTP:
# 命令行 FTP 客户端
ftp ftp.example.com
> user username
> pass password
> ls
> get file.txt
> put file.txt
> bye

# lftp (更强大)
lftp ftp://username:password@ftp.example.com
lftp> ls
lftp> mirror -R local/ remote/  # 同步上传
lftp> mirror remote/ local/     # 同步下载
lftp> exit

FTPS vs SFTP:
├─ FTPS: FTP + TLS/SSL (端口 990)
└─ SFTP: SSH File Transfer Protocol (端口 22, 基于 SSH)

现代替代方案:
├─ SFTP: 更安全
├─ SCP: 简单文件复制
└─ rsync: 增量同步
```

## 四、SMTP/POP3/IMAP

```
电子邮件协议:

┌────────────────────────────────────────────────────────────┐
│  发送方                                        接收方       │
│  ┌────────┐                                  ┌────────┐    │
│  │ 客户端 │                                  │ 客户端 │    │
│  └───┬────┘                                  └───▲────┘    │
│      │ SMTP (25/465/587)      POP3/IMAP (110/995/143/993) │
│      ▼                                          │          │
│  ┌────────┐      SMTP (25)     ┌────────┐     │          │
│  │邮件服务│ ──────────────────► │邮件服务│ ────┘          │
│  │  器A   │                     │  器B   │                │
│  └────────┘                     └────────┘                │
└────────────────────────────────────────────────────────────┘

SMTP (Simple Mail Transfer Protocol):
├─ 端口:
│  ├─ 25: 服务器间传输
│  ├─ 465: SMTPS (SSL)
│  └─ 587: 提交 (STARTTLS)
├─ 只负责发送
└─ 命令: HELO, MAIL FROM, RCPT TO, DATA, QUIT

POP3 (Post Office Protocol 3):
├─ 端口: 110 (明文), 995 (SSL)
├─ 下载并删除 (默认)
├─ 不同步状态
└─ 适合单设备

IMAP (Internet Message Access Protocol):
├─ 端口: 143 (明文), 993 (SSL)
├─ 服务器端管理
├─ 同步状态 (已读/未读/文件夹)
└─ 适合多设备

POP3 vs IMAP:
┌────────────┬─────────────┬─────────────────┐
│    特性    │    POP3     │      IMAP       │
├────────────┼─────────────┼─────────────────┤
│ 邮件存储   │ 本地        │ 服务器          │
│ 多设备同步 │ 不支持      │ 支持            │
│ 离线访问   │ 支持        │ 需缓存          │
│ 服务器负载 │ 低          │ 高              │
│ 带宽       │ 下载后低    │ 持续占用        │
└────────────┴─────────────┴─────────────────┘

使用 telnet 测试 SMTP:
telnet smtp.example.com 25
> HELO client.example.com
> MAIL FROM:<sender@example.com>
> RCPT TO:<receiver@example.com>
> DATA
> Subject: Test Email
>
> Hello, this is a test email.
> .
> QUIT
```

## 五、DHCP

```
DHCP (Dynamic Host Configuration Protocol) 动态主机配置协议:

DHCP 四步握手 (DORA):
客户端                                         DHCP服务器
   │                                              │
   │ [1] DHCP Discover (广播)                     │
   │ "有没有 DHCP 服务器?"                        │
   │──────────────────────────────────────────►   │
   │                                              │
   │ [2] DHCP Offer (单播/广播)                   │
   │ "我可以给你 192.168.1.100"                   │
   │ ◄────────────────────────────────────────── │
   │                                              │
   │ [3] DHCP Request (广播)                      │
   │ "我接受 192.168.1.100"                       │
   │──────────────────────────────────────────►   │
   │                                              │
   │ [4] DHCP Ack (单播)                          │
   │ "已分配 192.168.1.100, 租期 24h"             │
   │ ◄────────────────────────────────────────── │
   │                                              │
   │ 使用 IP 地址                                 │

DHCP 分配信息:
├─ IP 地址
├─ 子网掩码
├─ 默认网关
├─ DNS 服务器
├─ 租期 (Lease Time)
└─ 其他选项 (域名、NTP 服务器等)

租期更新:
├─ 租期到 50% 时,客户端尝试续租 (单播)
├─ 租期到 87.5% 时,再次尝试 (广播)
└─ 租期到期,释放 IP

端口:
├─ 服务器: UDP 67
└─ 客户端: UDP 68

查看 DHCP 租约:
# Linux
cat /var/lib/dhcp/dhclient.leases

# Windows
ipconfig /all

# 释放/更新 IP
# Linux
sudo dhclient -r      # 释放
sudo dhclient         # 获取

# Windows
ipconfig /release
ipconfig /renew
```

## 六、SSH

```
SSH (Secure Shell) 安全外壳协议:

端口: 22 (TCP)

SSH 连接建立:
1. TCP 三次握手
2. SSH 协议版本协商
3. 密钥交换
4. 认证
5. 会话建立

认证方式:
├─ 密码认证: 用户名 + 密码
├─ 公钥认证: SSH 密钥对 (推荐)
└─ 主机密钥认证

SSH 连接:
# 基本连接
ssh user@host

# 指定端口
ssh -p 2222 user@host

# 指定密钥
ssh -i ~/.ssh/id_rsa user@host

# 执行命令
ssh user@host 'ls -la /var/log'

# 远程命令输出
ssh user@host 'cat /etc/os-release'

SSH 隧道 (端口转发):
# 本地端口转发 (访问远程服务)
ssh -L 8080:localhost:80 user@remote
# 访问 localhost:8080 → remote:80

# 远程端口转发 (暴露本地服务)
ssh -R 8080:localhost:80 user@remote
# remote:8080 → localhost:80

# 动态端口转发 (SOCKS 代理)
ssh -D 1080 user@remote
# 浏览器设置 SOCKS5: localhost:1080

SSH 配置 (~/.ssh/config):
Host myserver
    HostName 192.168.1.100
    User admin
    Port 22
    IdentityFile ~/.ssh/id_rsa
    ServerAliveInterval 60

# 使用配置
ssh myserver

SSH 密钥管理:
# 生成密钥对
ssh-keygen -t ed25519 -C "email@example.com"

# 复制公钥到服务器
ssh-copy-id user@host

# 手动添加
cat ~/.ssh/id_rsa.pub | ssh user@host "cat >> ~/.ssh/authorized_keys"

SSH 安全:
├─ 禁用 root 登录
├─ 禁用密码认证 (只用密钥)
├─ 修改默认端口
├─ 使用 fail2ban
└─ 定期更新
```

这是应用层协议教程,涵盖了HTTP/HTTPS、DNS、FTP、邮件协议、DHCP和SSH。
