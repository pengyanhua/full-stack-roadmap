---
name: stats
description: 展示项目统计信息。当用户问"项目有多少内容"、"学习进度"、"模块概览"、"stats"时使用。
argument-hint: [overview|<模块名>]
allowed-tools: Bash, Glob, Grep, Read
context: fork
agent: Explore
---

# 项目统计

生成项目的统计概览信息。

## 参数

- `$ARGUMENTS`:
  - 空 / `overview` — 全局概览
  - `<模块名>` — 单个模块的详细统计（如 `Go`、`Python`）

## 全局概览统计

需要统计并展示：

### 1. 模块概览表

```
| 模块 | 类型 | 课程数 | 代码行数 | 文档页数 |
|------|------|--------|----------|----------|
| Go   | 语言 | 29     | 2,340    | 29       |
| ...  | ...  | ...    | ...      | ...      |
```

统计方式：
- 课程数：源码目录中的代码文件数量（`.go`, `.py`, `.java`, `.js`, `.jsx`, `.vue`, `.ts`）
- 代码行数：使用 `wc -l` 统计
- 文档页数：`docs/<模块>/` 下的 `.md` 文件数量（不含 index.md）

### 2. 分类汇总

按类别分组：
- 编程语言：Go, Python, Java, JavaScript
- 前端框架：React, Vue
- 数据库：MySQL, PostgreSQL, Redis, Elasticsearch, VectorDB
- 架构设计：Architecture, DDD, API_Gateway, Data_Architecture, Performance, Governance
- 云原生/DevOps：Cloud_Native, Container, DevOps
- 基础设施：Computer_Hardware, Operating_Systems, Linux, Networking
- 数据/AI：AI_Programming, AI_Architecture, BigData, Kafka
- 其他：Security_Advanced, Soft_Skills, DataStructures

### 3. 总计

- 总模块数
- 总课程/文档数
- 总代码行数

## 单模块详细统计

当指定了模块名时，展示：

1. 各子目录的文件列表
2. 每个文件的行数和主要主题
3. 与其他同类模块的对比
4. 可能缺失的内容建议

## 输出格式

使用清晰的 Markdown 表格和列表，方便阅读。在最后给出一句话总结项目规模。
