# 网络安全协议

## 一、TLS/SSL

```
TLS (Transport Layer Security) 传输层安全协议
SSL (Secure Sockets Layer) 安全套接字层

历史版本:
├─ SSL 1.0: 未发布
├─ SSL 2.0: 1995年,已废弃
├─ SSL 3.0: 1996年,已废弃 (POODLE 攻击)
├─ TLS 1.0: 1999年,即将废弃
├─ TLS 1.1: 2006年,即将废弃
├─ TLS 1.2: 2008年,目前主流
└─ TLS 1.3: 2018年,最新标准 (推荐)

TLS 1.2 握手过程:
客户端                                         服务器
   │                                              │
   │ ① ClientHello                                │
   │  ├─ TLS版本 (1.2)                           │
   │  ├─ 支持的加密套件列表                       │
   │  ├─ 支持的压缩方法                           │
   │  └─ Client Random (28字节随机数)            │
   │──────────────────────────────────────────►   │
   │                                              │
   │ ② ServerHello                                │
   │  ├─ 选择的TLS版本                            │
   │  ├─ 选择的加密套件                           │
   │  ├─ Server Random                            │
   │  ├─ 服务器证书 (Certificate)                │
   │  ├─ ServerKeyExchange (可选)                │
   │  ├─ CertificateRequest (可选,双向认证)      │
   │  └─ ServerHelloDone                          │
   │ ◄────────────────────────────────────────── │
   │                                              │
   │ ③ 客户端验证证书                             │
   │                                              │
   │ ④ ClientKeyExchange                          │
   │  └─ Pre-Master Secret (用服务器公钥加密)    │
   │──────────────────────────────────────────►   │
   │                                              │
   │ ⑤ ChangeCipherSpec                           │
   │ ⑥ Finished (加密握手消息的哈希)              │
   │──────────────────────────────────────────►   │
   │                                              │
   │ ⑦ ChangeCipherSpec                           │
   │ ⑧ Finished                                   │
   │ ◄────────────────────────────────────────── │
   │                                              │
   │ ◄═════════ 加密应用数据传输 ══════════════►  │

TLS 1.3 改进:
├─ 握手延迟: 2-RTT → 1-RTT
├─ 0-RTT 恢复: 会话恢复更快
├─ 简化密码套件: 移除不安全算法
├─ 强制前向安全 (Perfect Forward Secrecy)
└─ 加密握手消息: 更多握手信息被加密

加密套件示例:
TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
 │    │     │      │     │   │    │
 │    │     │      │     │   │    └─ 消息认证码: SHA256
 │    │     │      │     │   └────── 模式: GCM
 │    │     │      │     └────────── 加密长度: 128位
 │    │     │      └──────────────── 对称加密: AES
 │    │     └─────────────────────── 身份验证: RSA
 │    └───────────────────────────── 密钥交换: ECDHE
 └────────────────────────────────── 协议: TLS

密钥交换算法:
├─ RSA: 传统,不支持前向安全
├─ DH (Diffie-Hellman): 支持前向安全
├─ ECDHE (Elliptic Curve DH Ephemeral): 推荐
└─ PSK (Pre-Shared Key): 物联网常用

对称加密算法:
├─ AES (Advanced Encryption Standard): 主流
├─ ChaCha20: 移动设备优化
└─ 3DES: 已淘汰

消息认证:
├─ HMAC-SHA256
├─ HMAC-SHA384
└─ AEAD (AES-GCM, ChaCha20-Poly1305)

证书验证:
1. 检查证书有效期
2. 检查域名匹配
3. 检查证书签名 (CA 验证)
4. 检查证书吊销状态 (CRL/OCSP)

查看证书:
# OpenSSL
openssl s_client -connect example.com:443

# 查看证书详情
openssl x509 -in cert.pem -text -noout

# 测试 TLS 配置
# 在线工具: SSL Labs (ssllabs.com/ssltest)
```

## 二、IPSec

```
IPSec (Internet Protocol Security):

工作模式:
├─ 传输模式 (Transport Mode):
│  ├─ 只加密 IP 数据包的载荷
│  ├─ IP 头部不加密
│  └─ 用于端到端通信
│
└─ 隧道模式 (Tunnel Mode):
   ├─ 加密整个 IP 数据包
   ├─ 添加新的 IP 头部
   └─ 用于 VPN 网关间通信

IPSec 协议:
├─ AH (Authentication Header, 协议号 51):
│  ├─ 提供完整性验证和身份认证
│  ├─ 不提供加密
│  └─ 很少单独使用
│
└─ ESP (Encapsulating Security Payload, 协议号 50):
   ├─ 提供加密、完整性验证和身份认证
   ├─ 可选认证
   └─ 最常用

IKE (Internet Key Exchange) 密钥协商:
├─ IKEv1: 传统版本
└─ IKEv2: 改进版本,更快更安全

IKEv2 交换过程:
阶段1 (IKE_SA_INIT):
├─ 协商加密算法
├─ DH 密钥交换
└─ 交换随机数

阶段2 (IKE_AUTH):
├─ 认证双方身份
├─ 建立第一个 Child SA (ESP/AH)
└─ 分配 IP 地址

配置示例 (strongSwan):
# /etc/ipsec.conf
conn myvpn
    left=%any
    leftauth=pubkey
    leftcert=serverCert.pem
    right=%any
    rightauth=pubkey
    rightca=%same
    auto=add
    ike=aes256-sha256-modp2048!
    esp=aes256-sha256!
```

## 三、VPN 协议

```
VPN (Virtual Private Network) 虚拟专用网:

主流 VPN 协议:

1. OpenVPN:
├─ 基于: SSL/TLS
├─ 端口: 1194 UDP/TCP (可配置)
├─ 优点: 开源、安全、灵活、跨平台
├─ 缺点: 速度较慢、配置复杂
└─ 适用: 企业 VPN

2. WireGuard:
├─ 基于: 自定义加密协议
├─ 端口: 51820 UDP
├─ 优点: 极简、快速、现代密码学
├─ 缺点: 较新、生态系统还在发展
└─ 适用: 现代化 VPN (推荐)

3. IPSec/IKEv2:
├─ 基于: IPSec
├─ 端口: 500 UDP (IKE), 4500 UDP (NAT-T)
├─ 优点: 原生支持(iOS/macOS)、快速、稳定
├─ 缺点: 配置复杂
└─ 适用: 移动设备

4. PPTP:
├─ 基于: GRE
├─ 端口: 1723 TCP
├─ 优点: 简单、快速
├─ 缺点: 不安全 (已被破解)
└─ 不推荐使用

5. L2TP/IPSec:
├─ 基于: L2TP + IPSec
├─ 端口: 1701 UDP, 500 UDP, 4500 UDP
├─ 优点: 原生支持、较安全
├─ 缺点: 速度较慢、易被屏蔽
└─ 适用: 传统企业 VPN

6. SSTP:
├─ 基于: SSL/TLS
├─ 端口: 443 TCP
├─ 优点: 难以被屏蔽、Windows 原生支持
├─ 缺点: Windows 专有
└─ 适用: Windows 环境

协议对比:
┌──────────┬────────┬────────┬────────┬────────┐
│   协议   │  安全  │  速度  │  易用  │  推荐  │
├──────────┼────────┼────────┼────────┼────────┤
│ OpenVPN  │  ★★★★★│  ★★★☆☆│  ★★★☆☆│  ★★★★☆│
│WireGuard │  ★★★★★│  ★★★★★│  ★★★★☆│  ★★★★★│
│IPSec/IKEv2│  ★★★★☆│  ★★★★☆│  ★★★☆☆│  ★★★★☆│
│   PPTP   │  ★☆☆☆☆│  ★★★★★│  ★★★★★│  ☆☆☆☆☆│
│L2TP/IPSec│  ★★★☆☆│  ★★★☆☆│  ★★★☆☆│  ★★☆☆☆│
└──────────┴────────┴────────┴────────┴────────┘

WireGuard 配置示例:
# 服务器配置
[Interface]
PrivateKey = <server-private-key>
Address = 10.0.0.1/24
ListenPort = 51820

[Peer]
PublicKey = <client-public-key>
AllowedIPs = 10.0.0.2/32

# 客户端配置
[Interface]
PrivateKey = <client-private-key>
Address = 10.0.0.2/24
DNS = 1.1.1.1

[Peer]
PublicKey = <server-public-key>
Endpoint = server.example.com:51820
AllowedIPs = 0.0.0.0/0
PersistentKeepalive = 25
```

## 四、认证协议

```
认证协议:

1. Kerberos:
├─ 用途: 网络身份认证
├─ 特点: 基于票据、单点登录
├─ 组件:
│  ├─ KDC (Key Distribution Center)
│  ├─ TGT (Ticket Granting Ticket)
│  └─ Service Ticket
└─ 使用: Windows Active Directory

2. RADIUS (Remote Authentication Dial-In User Service):
├─ 端口: 1812 (认证), 1813 (计费)
├─ 用途: AAA (认证、授权、计费)
├─ 特点: 集中式认证
└─ 使用: 企业网络、WiFi、VPN

3. OAuth 2.0:
├─ 用途: 授权框架
├─ 角色:
│  ├─ Resource Owner (用户)
│  ├─ Client (应用)
│  ├─ Authorization Server (授权服务器)
│  └─ Resource Server (资源服务器)
├─ 授权类型:
│  ├─ Authorization Code (最常用)
│  ├─ Implicit (已废弃)
│  ├─ Resource Owner Password
│  └─ Client Credentials
└─ 使用: 第三方登录 (Google、GitHub)

4. OpenID Connect (OIDC):
├─ 基于: OAuth 2.0
├─ 用途: 身份认证 (Authentication)
├─ 特点: OAuth 2.0 + 身份层
└─ 使用: 单点登录 (SSO)

5. SAML (Security Assertion Markup Language):
├─ 用途: 单点登录 (SSO)
├─ 特点: 基于 XML
├─ 角色:
│  ├─ Identity Provider (IdP)
│  └─ Service Provider (SP)
└─ 使用: 企业级 SSO

OAuth 2.0 授权码流程:
用户                应用           授权服务器      资源服务器
 │                   │                │               │
 │ ① 请求登录         │                │               │
 │────────────────►   │                │               │
 │                   │ ② 重定向到授权页│               │
 │                   │────────────────►               │
 │ ◄──────────────────                │               │
 │                   │                │               │
 │ ③ 用户同意授权     │                │               │
 │────────────────────────────────────►               │
 │                   │                │               │
 │ ④ 返回授权码       │                │               │
 │ ◄──────────────────────────────────                │
 │────────────────►   │                │               │
 │                   │ ⑤ 用授权码换取token            │
 │                   │────────────────►               │
 │                   │ ⑥ 返回access_token             │
 │                   │ ◄──────────────                │
 │                   │                │               │
 │                   │ ⑦ 使用token访问资源            │
 │                   │────────────────────────────────►
 │                   │ ⑧ 返回资源                     │
 │                   │ ◄──────────────────────────────│
```

## 五、无线安全

```
WiFi 安全协议:

历史演进:
├─ WEP (Wired Equivalent Privacy):
│  ├─ 1997年,已废弃
│  ├─ 使用 RC4 加密
│  └─ 严重安全漏洞 (数分钟可破解)
│
├─ WPA (Wi-Fi Protected Access):
│  ├─ 2003年,过渡方案
│  ├─ TKIP (临时密钥完整性协议)
│  └─ 已过时
│
├─ WPA2 (2004年):
│  ├─ AES-CCMP 加密
│  ├─ 目前主流
│  └─ 存在 KRACK 攻击风险
│
└─ WPA3 (2018年):
   ├─ SAE (Simultaneous Authentication of Equals)
   ├─ 前向安全
   ├─ 更强的加密 (192位)
   └─ 最新标准 (推荐)

WPA2/WPA3 认证模式:
├─ PSK (Pre-Shared Key): 家庭/小型办公
└─ Enterprise (802.1X): 企业环境
   └─ 使用 RADIUS 服务器

WPA2 四次握手:
客户端                             接入点 (AP)
   │                                  │
   │ ① ANonce (AP 随机数)             │
   │ ◄──────────────────────────────  │
   │                                  │
   │ ② SNonce + MIC                   │
   │ ─────────────────────────────►   │
   │                                  │
   │ ③ GTK + MIC                      │
   │ ◄──────────────────────────────  │
   │                                  │
   │ ④ ACK                            │
   │ ─────────────────────────────►   │
   │                                  │
   │ ◄═══════ 加密数据传输 ═══════►   │

WPA3 改进:
├─ SAE: 抵抗离线字典攻击
├─ 前向安全: 即使密钥泄露,历史数据仍安全
├─ 更强加密: 192位安全套件 (企业版)
└─ 简化配置: Wi-Fi Easy Connect (DPP)

最佳实践:
├─ 使用 WPA3 (或至少 WPA2)
├─ 使用强密码 (20+ 字符)
├─ 隐藏 SSID (轻微提升安全性)
├─ 禁用 WPS (存在漏洞)
├─ 定期更新路由器固件
└─ 企业使用 802.1X
```

这是网络安全协议教程,涵盖了TLS/SSL、IPSec、VPN、认证协议和无线安全。
