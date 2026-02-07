# 🚀 快速开始指南

本指南帮助你快速启动 VitePress 文档网站。

## ✅ 已完成的配置

项目已经配置好以下内容：

- ✅ VitePress 项目结构
- ✅ 完整的导航和侧边栏配置
- ✅ 代码转 Markdown 自动化脚本
- ✅ GitHub Actions 自动部署
- ✅ 首页和模块入口页面
- ✅ 搜索功能（本地搜索）
- ✅ 暗色模式支持

## 📦 第一步：安装依赖

```bash
npm install
```

## 🎯 第二步：启动开发服务器

```bash
npm run docs:dev
```

访问 http://localhost:5173 查看效果！

## 📝 第三步：转换现有代码（可选）

如果你想将现有的代码文件转换为 Markdown：

```bash
npm run convert
```

这会自动扫描以下目录的代码文件并生成 Markdown：
- `Python/` → `docs/python/`
- `Go/` → `docs/go/`
- `Java/` → `docs/java/`
- `JavaScript/` → `docs/javascript/`

## 🎨 第四步：自定义配置

### 修改网站标题和描述

编辑 `docs/.vitepress/config.ts`:

```typescript
export default defineConfig({
  title: "你的网站标题",
  description: "你的网站描述",
  // ...
})
```

### 修改 GitHub 链接

1. 在 `config.ts` 中更新：
```typescript
socialLinks: [
  { icon: 'github', link: 'https://github.com/你的用户名/full-stack-roadmap' }
]
```

2. 在 `docs/index.md` 中更新 GitHub 按钮链接

### 添加 Google Analytics（可选）

```typescript
// config.ts
export default defineConfig({
  // ...
  head: [
    [
      'script',
      { async: '', src: 'https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX' }
    ],
    [
      'script',
      {},
      `window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-XXXXXXXXXX');`
    ]
  ]
})
```

## 🌐 第五步：部署到 GitHub Pages

### 1. 创建 GitHub 仓库

```bash
# 如果还没有初始化 git
git init
git add .
git commit -m "Initial commit: VitePress documentation"

# 关联远程仓库
git remote add origin https://github.com/你的用户名/full-stack-roadmap.git
git branch -M main
git push -u origin main
```

### 2. 配置 GitHub Pages

1. 访问仓库设置：**Settings** → **Pages**
2. **Source** 选择 **GitHub Actions**

### 3. 推送触发部署

```bash
git add .
git commit -m "Setup VitePress site"
git push
```

### 4. 等待部署完成

访问 **Actions** 标签页查看部署进度。

完成后访问：`https://你的用户名.github.io/full-stack-roadmap`

## 📚 常用命令

```bash
# 开发模式（热更新）
npm run docs:dev

# 构建生产版本
npm run docs:build

# 预览生产构建
npm run docs:preview

# 转换代码为 Markdown
npm run convert
```

## 📖 添加新内容

### 方式1：直接写 Markdown（推荐）

在 `docs/` 目录下创建 Markdown 文件：

```bash
# 例如：添加 Go 并发教程
docs/go/04-concurrency/goroutines.md
```

### 方式2：转换代码文件

1. 在对应目录添加代码文件：
```bash
Go/04-concurrency/01_goroutines.go
```

2. 运行转换脚本：
```bash
npm run convert
```

3. 检查生成的文档：
```bash
docs/go/04-concurrency/goroutines.md
```

### 更新侧边栏

编辑 `docs/.vitepress/config.ts` 的 `sidebar` 部分：

```typescript
sidebar: {
  '/go/': [
    {
      text: 'Go 学习路径',
      items: [
        { text: 'Goroutines', link: '/go/04-concurrency/goroutines' },
        // 添加新链接
      ]
    }
  ]
}
```

## 🎨 Markdown 功能示例

### 代码高亮

\`\`\`python{2,4-6}
def hello():
    print("Hello")  # 高亮此行

    # 高亮这几行
    for i in range(3):
        print(i)
\`\`\`

### 提示框

```markdown
::: tip 提示
这是一个提示框
:::

::: warning 警告
这是一个警告框
:::

::: danger 危险
这是一个危险提示
:::

::: info 信息
这是一个信息框
:::
```

### 代码组

```markdown
::: code-group

\`\`\`python [Python]
def hello():
    print("Hello")
\`\`\`

\`\`\`go [Go]
func hello() {
    fmt.Println("Hello")
}
\`\`\`

:::
```

## 🔧 故障排除

### 开发服务器启动失败

```bash
# 清除缓存
rm -rf docs/.vitepress/cache docs/.vitepress/dist node_modules
npm install
```

### 构建失败

检查：
1. 所有 Markdown 文件语法正确
2. 链接路径正确（区分大小写）
3. 图片路径正确

### 部署后 404

检查：
1. GitHub Pages 是否选择了 "GitHub Actions"
2. 仓库名称是否正确
3. base 路径配置（如果不是根路径部署）

## 📊 项目结构

```
.
├── docs/                      # VitePress 文档源文件
│   ├── .vitepress/
│   │   ├── config.ts         # 配置文件
│   │   ├── theme/            # 自定义主题（可选）
│   │   ├── dist/             # 构建输出（gitignore）
│   │   └── cache/            # 缓存（gitignore）
│   ├── public/               # 静态资源
│   │   └── logo.svg
│   ├── index.md              # 首页
│   ├── guide/                # 指南
│   ├── python/               # Python 文档
│   ├── go/                   # Go 文档
│   └── ...
├── scripts/
│   └── convert-to-markdown.js  # 转换脚本
├── .github/
│   └── workflows/
│       └── deploy.yml        # GitHub Actions 配置
├── Python/                   # 原始 Python 代码
├── Go/                       # 原始 Go 代码
├── package.json
└── README.md
```

## 💡 下一步

1. ✅ 查看示例页面：http://localhost:5173/python/02-functions/closure
2. ✅ 自定义首页内容
3. ✅ 添加更多文档内容
4. ✅ 配置 SEO 和 Analytics
5. ✅ 推送到 GitHub 并部署

## 📚 参考文档

- [VitePress 官方文档](https://vitepress.dev/)
- [Markdown 扩展功能](https://vitepress.dev/guide/markdown)
- [主题配置](https://vitepress.dev/reference/default-theme-config)

---

有问题？查看 [DEPLOY.md](DEPLOY.md) 获取详细部署说明！
