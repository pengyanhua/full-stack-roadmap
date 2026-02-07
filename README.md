[English](README.md) | [中文](README_zh.md)

# Full Stack Learning Roadmap

A comprehensive full-stack development learning resource covering programming languages, frameworks, databases, system architecture, and data structures, with practical code examples and detailed Chinese comments.

## 📖 Online Documentation

🌐 **Visit the website**: [https://pengyanhua.github.io/full-stack-roadmap](https://pengyanhua.github.io/full-stack-roadmap)

The documentation website provides:
- 🎨 Beautiful and responsive UI
- 🔍 Full-text search capability
- 💡 Syntax highlighting with line numbers
- 📱 Mobile-friendly design
- 🌙 Dark mode support

## 🚀 Quick Start

### View Online

Visit the [documentation website](https://pengyanhua.github.io/full-stack-roadmap) to browse all content with enhanced readability.

### Local Development

```bash
# Clone the repository
git clone https://github.com/pengyanhua/full-stack-roadmap.git
cd full-stack-roadmap

# Install dependencies
npm install

# Start development server
npm run docs:dev

# Build for production
npm run docs:build

# Preview production build
npm run docs:preview
```

### Run Code Examples

Each programming language has runnable examples:

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

## Contents

### Programming Languages

| Language | Topics |
|----------|--------|
| **Go** | Variables, Control Flow, Functions, Structs, Concurrency, Packages, Testing, Stdlib, Projects |
| **Python** | Variables, Control Flow, Functions, Classes, Async, Modules, Testing, Stdlib, Projects |
| **Java** | Basics, OOP, Collections, Concurrency, I/O, Functional, Modern Features (Records, Pattern Matching, Virtual Threads), Projects |
| **JavaScript** | Variables, Control Flow, Objects & Arrays, Functions, Closures, Async, ES6+, DOM, Projects |

### Frontend Frameworks

| Framework | Topics |
|-----------|--------|
| **React** | JSX, Components, Hooks (useState, useEffect), Context |
| **Vue** | Template Syntax, Components, Composition API, Reactivity, Composables, Router, Pinia |

### Databases

| Database | Topics |
|----------|--------|
| **MySQL** | SQL fundamentals, optimization |
| **PostgreSQL** | Advanced SQL features |
| **Redis** | Data structures, caching patterns |
| **Elasticsearch** | Full-text search, aggregations |
| **VectorDB** | Vector embeddings, similarity search |

### Message Queue

| Technology | Topics |
|------------|--------|
| **Kafka** | Producers, consumers, topics, partitions |

### System Architecture

| Category | Topics |
|----------|--------|
| **System Design** | Design Principles, Architecture Patterns, Capacity Planning |
| **Distributed Systems** | CAP/BASE Theorem, Distributed Locks, Distributed Transactions |
| **High Availability** | HA Principles, Rate Limiting, Failover, Disaster Recovery |
| **High Performance** | Performance Metrics, Concurrency, I/O Optimization, Pool Patterns |
| **Microservices** | Service Splitting, API Design, Service Governance, Service Mesh |
| **Database Architecture** | MySQL Optimization, Sharding, Read/Write Splitting |
| **Cache Architecture** | Cache Patterns, Cache Strategies |
| **Message Queue** | MQ Patterns, Reliability |
| **Security** | Security Fundamentals |
| **Observability** | Logging, Metrics, Tracing |

### Data Structures

| Structure | Implementation |
|-----------|----------------|
| Array | Concept + Python |
| Linked List | Concept + Python |
| Stack & Queue | Concept + Python |
| Hash Table | Concept + Python |
| Tree | Concept + Python |
| Heap | Concept + Python |
| Graph | Concept + Python |
| Advanced | Trie, Union-Find, etc. |

### Computer Networking

| Topic | Contents |
|-------|----------|
| **Network Fundamentals** | OSI Model, TCP/IP Protocol Stack, Network Layering |
| **Link Layer** | Ethernet, MAC Address, ARP, Switches, VLAN |
| **Network Layer** | IP Protocol, Routing, Subnetting, ICMP, NAT |
| **Transport Layer** | TCP, UDP, Three-way Handshake, Flow Control, Congestion Control |
| **Application Layer** | HTTP/HTTPS, DNS, FTP, SMTP, WebSocket |
| **Security Protocols** | SSL/TLS, Certificates, Encryption, Authentication |
| **Practical Applications** | Network Diagnostics, Packet Analysis, Performance Optimization |

### Containers & Operations

| Technology | Topics |
|------------|--------|
| **Docker** | Basics, Images, Containers, Dockerfile, Docker Compose |
| **Kubernetes** | Basics, Deployments, Services, Practical Examples |
| **Linux** | Basics, File System, Commands, Shell Scripting, Process Management, Networking, Security |

## Project Structure

```
.
├── Architecture/          # System design & architecture patterns
├── Container/             # Docker & Kubernetes
├── DataStructures/        # Data structures with implementations
├── Elasticsearch/         # Elasticsearch tutorials
├── Go/                    # Go language learning path
├── Java/                  # Java language learning path
├── JavaScript/            # JavaScript learning path
├── Kafka/                 # Apache Kafka tutorials
├── Linux/                 # Linux basics & operations
├── MySQL/                 # MySQL database tutorials
├── Networking/            # Computer networking protocols
├── PostgreSQL/            # PostgreSQL tutorials
├── Python/                # Python language learning path
├── React/                 # React framework tutorials
├── Redis/                 # Redis tutorials
├── VectorDB/              # Vector database tutorials
└── Vue/                   # Vue framework tutorials
```

## 🎯 Features

- ✅ **Structured learning paths** from basics to advanced topics
- ✅ **Practical code examples** with detailed Chinese comments
- ✅ **Theory + Practice**: Covers both concepts and implementations
- ✅ **Real-world projects** for each language
- ✅ **System architecture** best practices
- ✅ **Beautiful documentation** website with search and dark mode
- ✅ **Mobile-friendly** design for learning on the go

## 🛠️ Development

### Convert Code to Markdown

The project includes an automated script to convert code files into well-formatted Markdown documentation:

```bash
npm run convert
```

This script will:
- 🔍 Scan all `.py`, `.go`, `.java`, `.js` files
- 📝 Parse code structure and section comments
- ✨ Generate formatted Markdown with syntax highlighting
- 💬 Preserve detailed comments and explanations

### Adding New Content

1. Add your code files to the appropriate directory (e.g., `Python/02-functions/`)
2. Run `npm run convert` to generate Markdown
3. Review the generated files in `docs/`
4. Commit and push - GitHub Actions will auto-deploy!

## Learning Guide

Each directory contains numbered subdirectories representing the learning sequence:

```
Go/
├── 01-basics/       # Start here
├── 02-functions/
├── 03-structs/
├── 04-concurrency/
├── 05-packages/
├── 06-testing/
├── 07-stdlib/
└── 08-projects/     # End with practical projects
```

**How to learn**:
1. Visit the [documentation website](https://pengyanhua.github.io/full-stack-roadmap) for the best experience
2. Or browse the code repository directly, following the numbered order
3. Run the code examples and practice hands-on
4. Complete the project exercises in each module

## License

MIT
