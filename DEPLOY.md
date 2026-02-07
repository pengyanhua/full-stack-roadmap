# 部署指南

本文档说明如何将项目部署到 GitHub Pages。

## 📋 前置要求

- Node.js 18+
- npm 或 yarn
- GitHub 账号
- Git

## 🚀 部署到 GitHub Pages

### 1. 准备工作

确保你的代码已推送到 GitHub 仓库。

### 2. 配置 GitHub Pages

1. 访问你的 GitHub 仓库
2. 点击 **Settings** → **Pages**
3. **Source** 选择 **GitHub Actions**

### 3. 推送代码

推送代码到 main 分支会自动触发部署：

```bash
git add .
git commit -m "Setup VitePress documentation"
git push origin main
```

### 4. 查看部署状态

1. 访问仓库的 **Actions** 标签页
2. 等待 "Deploy VitePress site to Pages" 工作流完成
3. 部署成功后，访问 `https://pengyanhua.github.io/full-stack-roadmap`

## 🔧 本地开发

### 安装依赖

```bash
npm install
```

### 启动开发服务器

```bash
npm run docs:dev
```

访问 http://localhost:5173

### 构建生产版本

```bash
npm run docs:build
```

### 预览生产构建

```bash
npm run docs:preview
```

## 📝 更新内容工作流

### 1. 添加新代码

在相应目录添加代码文件：

```bash
# 例如：添加 Python 装饰器教程
# Python/02-functions/03_decorator.py
```

### 2. 运行转换脚本

```bash
npm run convert
```

这会自动将代码文件转换为 Markdown。

### 3. 检查生成的文档

```bash
# 启动开发服务器查看效果
npm run docs:dev
```

### 4. 提交并推送

```bash
git add .
git commit -m "Add decorator tutorial"
git push
```

GitHub Actions 会自动构建并部署！

## ⚙️ 配置说明

### VitePress 配置

主配置文件：`docs/.vitepress/config.ts`

关键配置项：

```typescript
{
  title: "网站标题",
  description: "网站描述",
  themeConfig: {
    nav: [...],      // 顶部导航
    sidebar: {...},  // 侧边栏
    search: {...}    // 搜索配置
  }
}
```

### 导航栏配置

编辑 `config.ts` 中的 `nav` 数组：

```typescript
nav: [
  { text: '首页', link: '/' },
  {
    text: '编程语言',
    items: [
      { text: 'Python', link: '/python/' },
      // ...
    ]
  }
]
```

### 侧边栏配置

编辑 `config.ts` 中的 `sidebar` 对象：

```typescript
sidebar: {
  '/python/': [
    {
      text: 'Python 学习路径',
      items: [
        { text: '基础', link: '/python/01-basics/' },
        // ...
      ]
    }
  ]
}
```

## 🎨 自定义样式

### 覆盖默认样式

创建 `docs/.vitepress/theme/index.ts`:

```typescript
import DefaultTheme from 'vitepress/theme'
import './custom.css'

export default {
  extends: DefaultTheme,
  // 添加自定义组件或逻辑
}
```

创建 `docs/.vitepress/theme/custom.css`:

```css
:root {
  --vp-c-brand: #42b883;
  --vp-c-brand-light: #42d392;
}
```

## 📊 SEO 优化

### 1. 配置 sitemap

已在 `config.ts` 中配置：

```typescript
sitemap: {
  hostname: 'https://pengyanhua.github.io/full-stack-roadmap'
}
```

### 2. Meta 标签

每个 Markdown 文件可以添加 frontmatter：

```markdown
---
title: Python 闭包详解
description: 深入理解 Python 闭包和作用域
head:
  - - meta
    - name: keywords
      content: python, closure, 闭包, 作用域
---
```

### 3. 结构化数据

VitePress 会自动生成 sitemap.xml。

## 🔍 搜索配置

### 本地搜索（默认）

已配置本地搜索：

```typescript
search: {
  provider: 'local',
  options: {
    translations: {
      button: {
        buttonText: '搜索文档'
      }
    }
  }
}
```

### Algolia 搜索（可选）

如果需要更强大的搜索功能：

1. 申请 Algolia DocSearch
2. 配置：

```typescript
search: {
  provider: 'algolia',
  options: {
    appId: 'YOUR_APP_ID',
    apiKey: 'YOUR_API_KEY',
    indexName: 'full-stack-roadmap'
  }
}
```

## 🐛 常见问题

### Q: 推送后网站没有更新？

**A:** 检查：
1. GitHub Actions 是否运行成功（Actions 标签页）
2. Pages 设置是否选择了 GitHub Actions
3. 等待几分钟让 CDN 刷新

### Q: 本地开发服务器启动失败？

**A:**
```bash
# 删除依赖重新安装
rm -rf node_modules package-lock.json
npm install

# 清除缓存
rm -rf docs/.vitepress/cache docs/.vitepress/dist
```

### Q: 代码高亮不正确？

**A:** 检查代码块语言标识：

```markdown
\`\`\`python  ← 确保语言标识正确
def hello():
    pass
\`\`\`
```

### Q: 侧边栏链接 404？

**A:** 确保：
1. Markdown 文件存在于对应路径
2. 链接路径正确（区分大小写）
3. 文件名与链接匹配

## 📚 参考资源

- [VitePress 官方文档](https://vitepress.dev/)
- [GitHub Pages 文档](https://docs.github.com/en/pages)
- [GitHub Actions 文档](https://docs.github.com/en/actions)

## 💡 最佳实践

1. **定期更新**：保持内容新鲜
2. **响应式测试**：在不同设备上测试
3. **性能优化**：图片压缩、懒加载
4. **SEO 优化**：合理使用标题、描述
5. **可访问性**：语义化 HTML、Alt 文本

---

如有问题，请提交 Issue！🚀
