# Cloudflare Pages 部署指南

将文档网站部署到 Cloudflare Pages 并绑定自定义域名 `f.tecfav.com`

## 方式一：通过 Cloudflare Dashboard（推荐）

### 1. 创建 Cloudflare Pages 项目

1. 登录 [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. 选择 **Workers & Pages** → **Create application** → **Pages** → **Connect to Git**
3. 授权 GitHub 并选择 `full-stack-roadmap` 仓库
4. 配置构建设置：
   - **Project name**: `full-stack-roadmap`
   - **Production branch**: `main`
   - **Build command**: `npm run docs:build`
   - **Build output directory**: `docs/.vitepress/dist`
   - **Node version**: `20`

5. 点击 **Save and Deploy**

### 2. 绑定自定义域名

1. 部署完成后，进入项目设置
2. 点击 **Custom domains** → **Set up a custom domain**
3. 输入域名：`f.tecfav.com`
4. 根据提示添加 DNS 记录：

   **如果使用 Cloudflare DNS**（自动配置）：
   - Cloudflare 会自动添加 CNAME 记录

   **如果使用其他 DNS 提供商**：
   - 添加 CNAME 记录：
     ```
     名称: f
     类型: CNAME
     值: full-stack-roadmap.pages.dev
     ```

5. 等待 DNS 生效（通常 1-5 分钟）

### 3. 启用 HTTPS

Cloudflare Pages 会自动为自定义域名提供免费的 SSL 证书

## 方式二：通过 GitHub Actions 自动部署

### 1. 获取 Cloudflare API 凭证

1. 在 Cloudflare Dashboard 获取：
   - **Account ID**: 在 Workers & Pages 页面右侧找到
   - **API Token**:
     - 进入 [API Tokens](https://dash.cloudflare.com/profile/api-tokens)
     - 创建自定义 Token，权限：`Cloudflare Pages - Edit`

### 2. 配置 GitHub Secrets

在 GitHub 仓库设置中添加 Secrets：

- `CLOUDFLARE_API_TOKEN`: 你的 API Token
- `CLOUDFLARE_ACCOUNT_ID`: 你的 Account ID

### 3. 启用 GitHub Actions

项目已包含 `.github/workflows/cloudflare-pages.yml` 配置文件。
推送到 main 分支后会自动触发部署。

## 方式三：使用 Wrangler CLI 手动部署

### 1. 安装 Wrangler

```bash
npm install -g wrangler
```

### 2. 登录 Cloudflare

```bash
wrangler login
```

### 3. 构建项目

```bash
npm run docs:build
```

### 4. 部署到 Pages

```bash
wrangler pages deploy docs/.vitepress/dist --project-name=full-stack-roadmap
```

### 5. 绑定自定义域名

```bash
wrangler pages domain add f.tecfav.com --project-name=full-stack-roadmap
```

## 验证部署

部署完成后，访问以下 URL 验证：

- ✅ Cloudflare Pages URL: `https://full-stack-roadmap.pages.dev`
- ✅ 自定义域名: `https://f.tecfav.com`

## 常见问题

### Q: 构建失败怎么办？

A: 检查构建日志，常见原因：
- Node.js 版本不匹配（确保使用 Node 18+）
- 依赖安装失败（检查 package.json）
- Markdown 文件语法错误

### Q: 自定义域名无法访问？

A: 检查：
1. DNS 记录是否正确配置
2. DNS 是否生效（使用 `nslookup f.tecfav.com`）
3. SSL 证书是否颁发完成（可能需要 15 分钟）

### Q: 如何更新网站？

A: 推送代码到 main 分支：
```bash
git add .
git commit -m "update content"
git push origin main
```

Cloudflare Pages 会自动检测变更并重新部署。

## 性能优化

Cloudflare Pages 自带的优化功能：

- ✅ 全球 CDN 加速（275+ 数据中心）
- ✅ 自动 HTTPS
- ✅ HTTP/2 和 HTTP/3 支持
- ✅ 无限带宽
- ✅ DDoS 保护

## 额外配置

### 配置自定义 Headers

创建 `docs/public/_headers` 文件：

```
/*
  X-Frame-Options: DENY
  X-Content-Type-Options: nosniff
  Referrer-Policy: no-referrer-when-downgrade
  Cache-Control: public, max-age=3600
```

### 配置重定向

创建 `docs/public/_redirects` 文件：

```
/old-page  /new-page  301
/docs/*    /:splat    301
```

## 监控和分析

Cloudflare Pages 提供免费的 Web Analytics：

1. 在项目设置中启用 **Web Analytics**
2. 查看访问量、页面浏览量、访客来源等数据

---

## 下一步

- [ ] 完成 Cloudflare Pages 项目创建
- [ ] 绑定自定义域名 f.tecfav.com
- [ ] 验证网站访问
- [ ] 配置 GitHub Actions 自动部署
- [ ] 启用 Web Analytics 监控
