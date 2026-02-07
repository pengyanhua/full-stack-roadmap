# 快速部署到 Cloudflare Pages

## 最简单的方式（推荐）

###  1. 登录 Cloudflare

访问 [Cloudflare Dashboard](https://dash.cloudflare.com/)

### 2. 创建 Pages 项目

1. 点击 **Workers & Pages**
2. 点击 **Create application**
3. 选择 **Pages** 标签
4. 点击 **Connect to Git**

### 3. 连接 GitHub

1. 点击 **Connect GitHub**
2. 授权 Cloudflare 访问你的 GitHub
3. 选择 `pengyanhua/full-stack-roadmap` 仓库

### 4. 配置构建

填写以下信息：

| 配置项 | 值 |
|--------|-------------|
| **Project name** | `full-stack-roadmap` |
| **Production branch** | `main` |
| **Framework preset** | `VitePress` |
| **Build command** | `npm run docs:build` |
| **Build output directory** | `docs/.vitepress/dist` |

**环境变量**（点击 **Add environment variable**）：

| 名称 | 值 |
|------|----------|
| `NODE_VERSION` | `20` |

### 5. 开始部署

点击 **Save and Deploy**

Cloudflare 会自动：
- ✅ 克隆你的仓库
- ✅ 安装依赖
- ✅ 构建网站
- ✅ 部署到全球 CDN

### 6. 查看网站

部署完成后，你会看到：
- **URL**: `https://full-stack-roadmap.pages.dev`

点击链接访问你的网站！

### 7. 绑定自定义域名 `f.tecfav.com`

1. 在项目页面，点击 **Custom domains**
2. 点击 **Set up a custom domain**
3. 输入：`f.tecfav.com`
4. 点击 **Continue**

**如果域名在 Cloudflare**：
- Cloudflare 会自动添加 DNS 记录
- 点击 **Activate domain**

**如果域名不在 Cloudflare**：
- 添加以下 CNAME 记录到你的 DNS 提供商：
  ```
  类型: CNAME
  名称: f
  目标: full-stack-roadmap.pages.dev
  ```
- 等待 DNS 生效（1-5 分钟）

### 8. 启用 HTTPS

Cloudflare 会自动为 `f.tecfav.com` 颁发免费 SSL 证书（大约 15 分钟）

## 访问网站

- ✅ Pages URL: https://full-stack-roadmap.pages.dev
- ✅ 自定义域名: https://f.tecfav.com

## 自动更新

每次推送代码到 `main` 分支，Cloudflare 会自动重新构建和部署！

```bash
git add .
git commit -m "update content"
git push origin main
```

## 故障排除

### 构建失败？

1. 在 Cloudflare Dashboard 查看**构建日志**
2. 检查错误信息
3. 常见问题：
   - Markdown 语法错误
   - Node.js 版本不匹配
   - 依赖安装失败

### 域名无法访问？

1. 检查 DNS 记录是否正确
2. 使用 `nslookup f.tecfav.com` 验证
3. 等待 SSL 证书颁发（最多 15 分钟）

---

## 就这么简单！ 🎉

总共只需要 **3 步**：
1. 连接 GitHub
2. 配置构建设置
3. 绑定域名

其他一切都由 Cloudflare 自动处理！
