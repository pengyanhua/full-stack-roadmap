# PostgreSQL 学习路径

::: tip 🐘 世界上最先进的开源数据库
PostgreSQL 是一个功能强大的开源对象关系数据库系统，以可靠性、功能健壮性和性能著称。
:::

## 📚 学习内容

### PostgreSQL 基础
- 安装与配置
- psql 命令行工具
- 数据库与模式
- 角色与权限

### 高级数据类型
- ARRAY 数组
- JSON/JSONB
- hstore 键值对
- 自定义类型

### 高级特性
- 窗口函数
- CTE (公共表表达式)
- 递归查询
- 全文搜索

### 性能优化
- 查询计划分析
- 索引策略
- 分区表
- 物化视图

### 扩展功能
- PostGIS (地理信息)
- pg_stat_statements
- 外部数据包装器 (FDW)
- 触发器与存储过程

## 🎯 与 MySQL 的区别

| 特性 | PostgreSQL | MySQL |
|------|-----------|-------|
| ACID 支持 | ✅ 完整支持 | ⚠️ 部分引擎支持 |
| JSON 支持 | ✅ JSONB 高性能 | ⚠️ 基础支持 |
| 窗口函数 | ✅ 完整支持 | ⚠️ 较新版本支持 |
| 全文搜索 | ✅ 内置强大 | ⚠️ 基础支持 |
| 复杂查询 | ✅ 性能更好 | ⚠️ 相对较慢 |

## 📖 推荐资源

- [PostgreSQL 官方文档](https://www.postgresql.org/docs/)
- [PostgreSQL 中文文档](http://www.postgres.cn/docs/)
- 《PostgreSQL 修炼之道》

## 🔗 相关学习

- 对比学习 [MySQL](/mysql/) 特性差异
- 结合 [Redis](/redis/) 缓存方案
- 了解 [系统架构](/architecture/) 数据库选型

## 💡 实战建议

1. **充分利用特性**：使用 PostgreSQL 独有功能
2. **JSON 存储**：适合半结构化数据场景
3. **地理信息**：PostGIS 是地理应用首选
4. **企业级应用**：适合复杂业务场景

---

::: warning 🚧 内容正在完善中
PostgreSQL 详细教程和代码示例正在编写中，敬请期待！

如果你有任何建议或想学习的内容，欢迎在 [GitHub Discussions](https://github.com/pengyanhua/full-stack-roadmap/discussions) 中讨论。
:::
