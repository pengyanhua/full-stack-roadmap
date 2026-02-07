[English](README.md) | [中文](README_zh.md)

# 全栈开发学习路线

一个全面的全栈开发学习资源库，涵盖编程语言、框架、数据库、系统架构和数据结构，包含实用的代码示例和详细的中文注释。

## 📖 在线文档

🌐 **访问网站**: [https://t.tecfav.com](https://t.tecfav.com)

文档网站提供：
- 🎨 美观的响应式界面
- 🔍 全文搜索功能
- 💡 代码高亮和行号显示
- 📱 移动端友好设计
- 🌙 暗色模式支持

## 🚀 快速开始

### 在线浏览

访问[文档网站](https://t.tecfav.com)以获得更好的阅读体验。

### 本地开发

```bash
# 克隆仓库
git clone https://github.com/pengyanhua/full-stack-roadmap.git
cd full-stack-roadmap

# 安装依赖
npm install

# 启动开发服务器
npm run docs:dev

# 构建生产版本
npm run docs:build

# 预览生产构建
npm run docs:preview
```

### 运行代码示例

每种编程语言都有可运行的示例：

```bash
# Python
python Python/02-functions/02_closure.py

# Go
go run Go/04-concurrency/01_goroutines.go

# Java
javac Java/01-basics/Variables.java && java Variables

# JavaScript
node JavaScript/01-basics/01_variables.js
```

## 目录

### 编程语言

| 语言 | 主题 |
|------|------|
| **Go** | 变量、流程控制、函数、结构体、并发、包管理、测试、标准库、项目实战 |
| **Python** | 变量、流程控制、函数、类、异步、模块、测试、标准库、项目实战 |
| **Java** | 基础、面向对象、集合、并发、I/O、函数式、现代特性（Records、模式匹配、虚拟线程）、项目实战 |
| **JavaScript** | 变量、流程控制、对象与数组、函数、闭包、异步、ES6+、DOM、项目实战 |

### 前端框架

| 框架 | 主题 |
|------|------|
| **React** | JSX、组件、Hooks（useState、useEffect）、Context |
| **Vue** | 模板语法、组件、组合式 API、响应式、Composables、Router、Pinia |

### 数据库

| 数据库 | 主题 |
|--------|------|
| **MySQL** | SQL 基础、性能优化 |
| **PostgreSQL** | 高级 SQL 特性 |
| **Redis** | 数据结构、缓存模式 |
| **Elasticsearch** | 全文搜索、聚合查询 |
| **VectorDB** | 向量嵌入、相似度搜索 |

### 消息队列

| 技术 | 主题 |
|------|------|
| **Kafka** | 生产者、消费者、主题、分区 |

### 系统架构

| 分类 | 主题 |
|------|------|
| **系统设计** | 设计原则、架构模式、容量规划 |
| **分布式系统** | CAP/BASE 理论、分布式锁、分布式事务 |
| **高可用** | 高可用原则、限流、故障转移、容灾 |
| **高性能** | 性能指标、并发、I/O 优化、池化模式 |
| **微服务** | 服务拆分、API 设计、服务治理、Service Mesh |
| **数据库架构** | MySQL 优化、分库分表、读写分离 |
| **缓存架构** | 缓存模式、缓存策略 |
| **消息队列** | MQ 模式、可靠性保证 |
| **安全** | 安全基础 |
| **可观测性** | 日志、指标、链路追踪 |

### 数据结构

| 数据结构 | 实现 |
|----------|------|
| 数组 | 概念 + Python 实现 |
| 链表 | 概念 + Python 实现 |
| 栈与队列 | 概念 + Python 实现 |
| 哈希表 | 概念 + Python 实现 |
| 树 | 概念 + Python 实现 |
| 堆 | 概念 + Python 实现 |
| 图 | 概念 + Python 实现 |
| 高级数据结构 | Trie、并查集等 |

### 计算机网络

| 主题 | 内容 |
|------|------|
| **网络基础** | OSI 模型、TCP/IP 协议栈、网络分层架构 |
| **链路层** | 以太网、MAC 地址、ARP、交换机、VLAN |
| **网络层** | IP 协议、路由、子网划分、ICMP、NAT |
| **传输层** | TCP、UDP、三次握手、四次挥手、流量控制、拥塞控制 |
| **应用层** | HTTP/HTTPS、DNS、FTP、SMTP、WebSocket |
| **安全协议** | SSL/TLS、证书、加密、认证 |
| **实践应用** | 网络诊断、抓包分析、性能优化 |

### 容器与运维

| 技术 | 主题 |
|------|------|
| **Docker** | 基础、镜像、容器、Dockerfile、Docker Compose |
| **Kubernetes** | 基础、Deployments、Services、实战示例 |
| **Linux** | 基础、文件系统、命令、Shell 脚本、进程管理、网络、安全 |

## 项目结构

```
.
├── Architecture/          # 系统设计与架构模式
├── Container/             # Docker 与 Kubernetes
├── DataStructures/        # 数据结构与实现
├── Elasticsearch/         # Elasticsearch 教程
├── Go/                    # Go 语言学习路径
├── Java/                  # Java 语言学习路径
├── JavaScript/            # JavaScript 学习路径
├── Kafka/                 # Apache Kafka 教程
├── Linux/                 # Linux 基础与运维
├── MySQL/                 # MySQL 数据库教程
├── Networking/            # 计算机网络协议
├── PostgreSQL/            # PostgreSQL 教程
├── Python/                # Python 语言学习路径
├── React/                 # React 框架教程
├── Redis/                 # Redis 教程
├── VectorDB/              # 向量数据库教程
└── Vue/                   # Vue 框架教程
```

## 特点

- ✅ **结构化学习路径** - 从基础到高级的系统化学习
- ✅ **详细中文注释** - 实用的代码示例配有详细注释
- ✅ **理论与实践结合** - 概念讲解 + 代码实现
- ✅ **真实项目示例** - 每种语言都包含项目实战
- ✅ **系统架构深度** - 分布式、微服务、高可用等最佳实践
- ✅ **精美文档网站** - 支持搜索和暗色模式
- ✅ **移动端友好** - 随时随地学习

## 🛠️ 开发

### 代码转 Markdown

项目包含自动化脚本，可将代码文件转换为格式化的 Markdown 文档：

```bash
npm run convert
```

脚本功能：
- 🔍 扫描所有 `.py`, `.go`, `.java`, `.js` 文件
- 📝 解析代码结构和注释
- ✨ 生成带语法高亮的 Markdown
- 💬 保留详细的注释和说明

### 添加新内容

1. 在相应目录添加代码文件（如 `Python/02-functions/`）
2. 运行 `npm run convert` 生成 Markdown
3. 查看 `docs/` 目录中生成的文件
4. 提交并推送 - GitHub Actions 会自动部署！

## 学习指南

每个目录包含按学习顺序编号的子目录：

```
Go/
├── 01-basics/       # 从这里开始
├── 02-functions/
├── 03-structs/
├── 04-concurrency/
├── 05-packages/
├── 06-testing/
├── 07-stdlib/
└── 08-projects/     # 以项目实战结束
```

**学习方式**：
1. 访问[在线文档](https://t.tecfav.com)获得最佳体验
2. 或直接浏览代码仓库，按编号顺序学习
3. 运行代码示例，动手实践
4. 完成每个模块的项目实战

## 许可证

MIT
