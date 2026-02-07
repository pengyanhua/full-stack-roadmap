# Cloudflare Pages vs Workers 选择指南

## 当前项目：推荐使用 Pages ✅

### 为什么？

当前是**静态文档网站**（VitePress），Cloudflare Pages 是最佳选择：

| 特性 | Pages | Workers | 说明 |
|------|-------|---------|------|
| 静态站点托管 | ✅ 原生支持 | ⚠️ 需手动配置 | Pages 专为此设计 |
| Git 自动部署 | ✅ 内置 | ❌ 需配置 Actions | 推送即部署 |
| 免费额度 | ✅ 无限请求 | ⚠️ 10万/天 | Pages 更慷慨 |
| 配置复杂度 | ✅ 简单 | ⚠️ 较复杂 | 几分钟搞定 |
| 构建日志 | ✅ 可视化 | ⚠️ 需自己实现 | 方便调试 |

## 添加讨论功能：不需要 Workers！

### 方案对比

#### ✅ 推荐：Giscus（基于 GitHub Discussions）

```
用户 → 前端 → GitHub API → GitHub Discussions
```

**优势：**
- 无需后端服务器
- 完全免费
- 5 分钟集成
- 功能完整（回复、点赞、Markdown）
- 自动适配主题

**劣势：**
- 用户需要 GitHub 账号
- 依赖 GitHub 服务

#### ⚠️ Pages + Functions

```
用户 → 前端 → Pages Function → 数据库
```

**说明：**
- Cloudflare Pages 支持 Functions（本质是 Workers）
- 可以在 Pages 项目中添加 API endpoints
- 适合简单的后端逻辑

**使用场景：**
- 表单提交
- 简单的数据查询
- 第三方 API 代理

#### ⚠️ Workers + D1 数据库

```
用户 → 前端 → Worker → D1 数据库
```

**适用场景：**
- 需要完全控制数据
- 复杂的业务逻辑
- 不依赖第三方服务

**成本：**
- 需要编写后端代码
- 管理数据库
- 实现认证、权限等

## 实际决策流程

### 当前阶段：使用 Pages

```bash
# 部署到 Cloudflare Pages
1. 连接 GitHub 仓库
2. 配置构建命令
3. 完成！
```

### 添加评论：使用 Giscus

```bash
# 安装 Giscus
npm install @giscus/vue

# 在 VitePress 主题中集成
# 5 分钟完成！
```

### 未来扩展（如果需要）

**场景 1：添加简单 API**

使用 **Pages Functions**：

```javascript
// functions/api/hello.js
export async function onRequest(context) {
  return new Response(JSON.stringify({ message: "Hello" }))
}
```

访问：`https://f.tecfav.com/api/hello`

**场景 2：复杂的后端逻辑**

这时才考虑 **Workers + D1**：

```javascript
// worker.js
import { Hono } from 'hono'

const app = new Hono()

app.get('/api/comments', async (c) => {
  const db = c.env.DB
  const comments = await db.prepare('SELECT * FROM comments').all()
  return c.json(comments)
})

export default app
```

## 技术栈演进路径

### 阶段 1：纯静态（当前）✅

```
Pages + VitePress
```

**时间：** 1 天
**成本：** 免费

### 阶段 2：添加评论 ✅

```
Pages + VitePress + Giscus
```

**时间：** + 1 小时
**成本：** 免费

### 阶段 3：简单 API（如需要）

```
Pages + Functions
```

**示例：**
- 表单提交
- 邮件通知
- 访问统计

**时间：** + 几小时
**成本：** 免费（10万请求/天）

### 阶段 4：完整后端（很少需要）

```
Workers + D1 + R2
```

**示例：**
- 用户认证系统
- 复杂的数据处理
- 文件上传存储

**时间：** + 数天
**成本：** 小额付费

## 推荐决策

### 现在：Cloudflare Pages

```bash
# 1. 部署静态站点
优点：简单、快速、免费
时间：30 分钟
```

### 添加评论时：Giscus

```bash
# 2. 集成 GitHub Discussions
优点：无需后端、免费、功能完整
时间：30 分钟
```

### 如果未来需要自定义 API：Pages Functions

```bash
# 3. 在 Pages 中添加 Functions
优点：渐进式增强、无需迁移
时间：按需
```

### 只有在以下情况才用 Workers：

- ❌ 需要复杂的服务端逻辑
- ❌ 需要自己的数据库
- ❌ 需要完全控制后端
- ❌ 不想依赖第三方服务

## 总结

### 当前最佳方案

```
Cloudflare Pages + Giscus
```

**理由：**
1. ✅ Pages 完美适配静态文档
2. ✅ Giscus 解决评论需求
3. ✅ 完全免费
4. ✅ 配置简单
5. ✅ 性能优秀
6. ✅ 无需维护后端

### 什么时候迁移到 Workers？

**答案：可能永远不需要！**

- Pages + Functions 可以覆盖 95% 的场景
- 只有非常复杂的应用才需要纯 Workers
- 即使需要，也可以混用（Pages 前端 + Workers API）

## 立即行动

1. **现在**：使用 Pages 部署文档站（查看 [QUICK_DEPLOY_CF.md](QUICK_DEPLOY_CF.md)）
2. **稍后**：添加 Giscus 评论（查看 [docs/guide/add-comments.md](docs/guide/add-comments.md)）
3. **未来**：根据需要考虑 Functions 或 Workers

---

**结论：选择 Pages！** 🎉
