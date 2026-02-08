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

### 系统基础

| 主题 | 内容 |
|------|------|
| **计算机硬件** | CPU架构、内存系统、存储设备、I/O系统、GPU计算、网络硬件、电源散热、硬件选型 |
| **操作系统** | OS概述、进程管理、内存管理、文件系统、I/O管理、并发控制、虚拟化、安全、性能调优、现代OS |

### 软件架构（架构师必备）

| 分类 | 主题 |
|------|------|
| **云原生** | 云计算、Serverless、多云架构、成本优化 |
| **DevOps** | CI/CD、GitOps、基础设施即代码、部署策略 |
| **API网关** | 网关设计、路由策略、认证授权、性能优化 |
| **领域驱动设计** | 战略设计、战术设计、事件风暴 |
| **性能调优** | 负载测试、性能分析、瓶颈定位、优化实践 |
| **技术治理** | 技术债务、架构评审、ADR、技术标准 |
| **数据架构** | 数据建模、数据治理、数据管道、数据湖 |
| **安全进阶** | 零信任、密钥管理、合规性、安全测试 |
| **大数据** | 批处理、流处理、实时数仓、OLAP |
| **AI架构** | ML流水线、模型服务、特征平台 |
| **软技能** | 技术决策、架构文档、技术沟通 |

### AI编程

| 主题 | 内容 |
|------|------|
| **基础知识** | LLM基础、Prompt工程、Embedding向量 |
| **开发框架** | LangChain、LlamaIndex、Semantic Kernel、AutoGen |
| **RAG系统** | RAG架构、向量数据库、分块策略、检索优化 |
| **Agent系统** | Agent基础、ReAct模式、Tool Calling、LangGraph、多Agent协作 |
| **深度学习** | PyTorch、Transformer、模型微调、模型优化 |
| **AI工程** | MLOps、模型服务、监控、成本优化 |
| **AI辅助编程** | GitHub Copilot、Cursor、代码审查、效率提升 |
| **实战项目** | 智能客服、文档问答、代码助手、数据分析 |

## 项目结构

```
.
├── AI_Architecture/       # AI系统架构与模型服务
├── AI_Programming/        # AI辅助编程、RAG、Agent、深度学习
├── API_Gateway/           # API网关设计与实现
├── Architecture/          # 系统设计与架构模式
├── BigData/               # 大数据处理与分析
├── Cloud_Native/          # 云计算与Serverless
├── Computer_Hardware/     # 计算机硬件基础与性能
├── Container/             # Docker 与 Kubernetes
├── Data_Architecture/     # 数据建模与治理
├── DataStructures/        # 数据结构与实现
├── DDD/                   # 领域驱动设计
├── DevOps/                # CI/CD、GitOps、IaC
├── Elasticsearch/         # Elasticsearch 教程
├── Go/                    # Go 语言学习路径
├── Governance/            # 技术治理与标准
├── Java/                  # Java 语言学习路径
├── JavaScript/            # JavaScript 学习路径
├── Kafka/                 # Apache Kafka 教程
├── Linux/                 # Linux 基础与运维
├── MySQL/                 # MySQL 数据库教程
├── Networking/            # 计算机网络协议
├── Operating_Systems/     # 操作系统原理与机制
├── Performance/           # 性能测试与调优
├── PostgreSQL/            # PostgreSQL 教程
├── Python/                # Python 语言学习路径
├── React/                 # React 框架教程
├── Redis/                 # Redis 教程
├── Security_Advanced/     # 高级安全实践
├── Soft_Skills/           # 技术领导力与沟通
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
