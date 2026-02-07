---
layout: home

hero:
  name: "全栈开发学习路线"
  text: "从基础到进阶的系统化学习资源"
  tagline: 涵盖编程语言、框架、数据库、系统架构和数据结构，配有详细的中文注释和实战项目
  actions:
    - theme: brand
      text: 开始学习
      link: /guide/getting-started
    - theme: alt
      text: 查看 GitHub
      link: https://github.com/pengyanhua/full-stack-roadmap
  image:
    src: /logo.svg
    alt: 全栈开发学习路线

features:
  - icon: 🐍
    title: Python
    details: 从基础到高级，涵盖函数式编程、异步编程、测试等，配有实战项目
    link: /python/

  - icon: 🔷
    title: Go
    details: 学习 Go 的并发模型、包管理、测试框架，掌握现代化后端开发
    link: /go/

  - icon: ☕
    title: Java
    details: 深入 Java 核心，包括现代特性（Records、虚拟线程）和企业级开发
    link: /java/

  - icon: 💛
    title: JavaScript
    details: ES6+、TypeScript、Node.js，全栈 JavaScript 开发必备
    link: /javascript/

  - icon: ⚛️
    title: React
    details: 组件化开发、Hooks、状态管理，构建现代化前端应用
    link: /react/

  - icon: 💚
    title: Vue
    details: 渐进式前端框架，从基础到高级，快速构建用户界面
    link: /vue/

  - icon: 🗄️
    title: 数据库
    details: MySQL、PostgreSQL、Redis、Elasticsearch 等，掌握数据存储与查询优化
    link: /mysql/

  - icon: 📨
    title: 消息队列
    details: Kafka 生产者/消费者模型，构建高性能分布式系统
    link: /kafka/

  - icon: 🏗️
    title: 系统架构
    details: 设计原则、分布式系统、微服务、高可用、高性能架构实践
    link: /architecture/

  - icon: 🔢
    title: 数据结构
    details: 数组、链表、树、图等核心数据结构，配 Python 实现
    link: /datastructures/

  - icon: 🐳
    title: 容器化
    details: Docker、Kubernetes 实战，掌握现代化部署和运维
    link: /container/

  - icon: 🎯
    title: 项目实战
    details: 每种语言都包含实战项目，理论与实践相结合
    link: /python/08-projects/
---

## 🚀 快速开始

### 学习路径建议

1. **编程语言基础**
   - 选择一门主力语言（Python/Go/Java）
   - 按目录编号顺序学习（01-基础 → 08-项目）
   - 完成每个章节的练习

2. **前端开发**
   - 学习 JavaScript 基础
   - 掌握 React 或 Vue 框架
   - 构建完整的前端项目

3. **后端开发**
   - 学习数据库（MySQL、Redis）
   - 掌握消息队列（Kafka）
   - 理解系统架构原则

4. **系统架构**
   - 学习分布式系统理论
   - 掌握微服务架构
   - 理解高可用和高性能设计

5. **容器化部署**
   - Docker 容器化应用
   - Kubernetes 编排部署
   - CI/CD 自动化流程

### 特色亮点

- ✅ **结构化学习路径**：从基础到高级，循序渐进
- ✅ **详细中文注释**：代码示例配有详细的中文注释
- ✅ **理论与实践结合**：概念讲解 + 代码实现
- ✅ **实战项目导向**：每种语言都包含真实项目
- ✅ **系统架构深度**：涵盖分布式、微服务、高可用等

## 📚 学习资源

### 编程语言

<div class="language-grid">

| 语言 | 学习模块 | 难度 |
|------|---------|------|
| [Python](/python/) | 基础、函数、类、异步、模块、测试、项目 | ⭐⭐ |
| [Go](/go/) | 基础、并发、包管理、测试、项目 | ⭐⭐⭐ |
| [Java](/java/) | 基础、OOP、集合、并发、现代特性、项目 | ⭐⭐⭐ |
| [JavaScript](/javascript/) | 基础、ES6+、异步、TypeScript、Node.js | ⭐⭐ |

</div>

### 系统架构

| 主题 | 内容 |
|------|------|
| [系统设计](/architecture/) | SOLID 原则、架构模式、容量规划 |
| [分布式系统](/architecture/) | CAP/BASE 理论、分布式锁、分布式事务 |
| [微服务](/architecture/) | 服务拆分、API 设计、Service Mesh |
| [高可用](/architecture/) | 限流、熔断、故障转移、容灾 |

## 🎯 适用人群

- 🎓 **计算机专业学生**：系统化学习全栈技术
- 💼 **初级开发者**：快速提升技术深度和广度
- 🚀 **面试准备者**：全面复习核心概念和最佳实践
- 🔧 **技术爱好者**：探索不同语言和技术栈

## 📖 使用指南

### 如何学习

1. **选择学习路径**：根据目标选择编程语言或技术栈
2. **按顺序学习**：每个目录按编号组织，建议按顺序学习
3. **动手实践**：运行代码示例，修改并实验
4. **完成项目**：在 08-projects 目录找到实战项目
5. **深入架构**：学习系统架构和设计模式

### 代码运行

每个代码文件都可以直接运行：

```bash
# Python
python Python/02-functions/02_closure.py

# Go
go run Go/04-concurrency/01_goroutines.go

# Java
javac Java/01-basics/Variables.java && java Variables
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

- 报告错误或提出建议
- 完善文档和示例
- 分享学习心得

## 📄 许可证

本项目采用 [MIT 许可证](https://github.com/pengyanhua/full-stack-roadmap/blob/main/LICENSE)。

---

<div style="text-align: center; margin-top: 40px;">
  <p>⭐ 如果这个项目对你有帮助，请给一个 Star！</p>
  <p>💬 有问题？欢迎在 GitHub 讨论区交流</p>
</div>

<style>
.language-grid {
  margin: 20px 0;
}

.language-grid table {
  width: 100%;
}
</style>
