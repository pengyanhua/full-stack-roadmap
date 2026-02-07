# Giscus 配置步骤

## ✅ 已完成的配置

- ✅ 安装了 `@giscus/vue` 依赖
- ✅ 创建了评论组件 `Comment.vue`
- ✅ 集成到 VitePress 主题
- ✅ 配置了智能显示规则（首页和指南页不显示）
- ✅ 主题自动跟随深浅色模式

## 📝 需要你完成的步骤

### 1. 启用 GitHub Discussions

1. 打开仓库：https://github.com/pengyanhua/full-stack-roadmap
2. 点击 **Settings** (设置)
3. 找到 **Features** (功能) 部分
4. 勾选 ✅ **Discussions**
5. 点击 **Set up discussions**

### 2. 安装 Giscus App

1. 访问：https://github.com/apps/giscus
2. 点击 **Install**
3. 选择 **Only select repositories**
4. 选中 `full-stack-roadmap`
5. 点击 **Install**

### 3. 获取配置参数

1. 访问：https://giscus.app/zh-CN

2. **仓库**部分：
   - 输入：`pengyanhua/full-stack-roadmap`
   - 等待验证通过（会显示绿色✓）

3. **Discussion 分类**：
   - 建议选择：`General`（通用讨论）
   - 或者：`Announcements`（公告 - 只有维护者可发起讨论）

4. **特性**（保持默认即可）：
   - ✅ 启用主评论区的回复
   - ✅ 在主评论上方放置评论框
   - ✅ 延迟加载评论

5. **主题**：
   - 选择：`preferred_color_scheme`
   - （已在代码中配置为动态跟随）

6. **滚动到页面底部**，你会看到类似这样的配置：

```html
<script src="https://giscus.app/client.js"
        data-repo="pengyanhua/full-stack-roadmap"
        data-repo-id="R_kgDO..."      <!-- 复制这个值 -->
        data-category="General"
        data-category-id="DIC_kwDO..." <!-- 复制这个值 -->
        ...>
</script>
```

7. **复制两个重要的值**：
   - `data-repo-id`: 类似 `R_kgDONd3yOA`
   - `data-category-id`: 类似 `DIC_kwDONd3yOM4ClXkK`

### 4. 更新配置文件

编辑文件：`docs/.vitepress/theme/components/Comment.vue`

找到这几行并替换为你的值：

```vue
<Giscus
  repo="pengyanhua/full-stack-roadmap"
  repo-id="R_kgDO..."           <!-- 替换为你的 repo-id -->
  category="General"             <!-- 确认分类名称 -->
  category-id="DIC_kwDO..."     <!-- 替换为你的 category-id -->
  ...
/>
```

**示例**：

```vue
<Giscus
  repo="pengyanhua/full-stack-roadmap"
  repo-id="R_kgDONd3yOA"
  category="General"
  category-id="DIC_kwDONd3yOM4ClXkK"
  ...
/>
```

### 5. 测试

```bash
npm run docs:dev
```

打开浏览器访问任意文档页面（例如：http://localhost:5173/python/01-basics/variables）

滚动到页面底部，你应该看到：
- 💬 讨论
- 使用 GitHub 账号登录后即可参与讨论
- Giscus 评论区

### 6. 登录测试

1. 点击 **Sign in with GitHub**
2. 授权 Giscus 应用
3. 输入评论测试
4. 评论会出现在 GitHub Discussions 中

## 🎨 已配置的功能

### 智能显示规则

评论区会在以下页面**自动隐藏**：
- ✅ 首页 (`/`)
- ✅ 指南页面 (`/guide/*`)

其他所有文档页面都会显示评论区。

### 禁用特定页面的评论

如果想在某个页面禁用评论，在文件顶部添加：

```markdown
---
comment: false
---

# 页面标题
```

### 主题自动跟随

评论区会自动跟随 VitePress 的深浅色模式：
- 浅色模式 → 浅色评论区
- 深色模式 → 深色评论区

## 🔍 验证配置

### 检查清单

- [ ] GitHub Discussions 已启用
- [ ] Giscus App 已安装到仓库
- [ ] 已获取 `repo-id` 和 `category-id`
- [ ] 已更新 `Comment.vue` 配置
- [ ] 本地测试成功显示评论区
- [ ] 可以登录并发表评论

### 常见问题

**Q: 评论区不显示？**
A: 检查：
1. 是否在首页或指南页（这些页面默认隐藏）
2. 浏览器控制台是否有错误
3. `repo-id` 和 `category-id` 是否正确

**Q: 无法加载评论？**
A: 确认：
1. Discussions 是否已启用
2. Giscus App 是否已安装
3. 网络是否能访问 GitHub

**Q: 主题颜色不对？**
A: 已配置为自动跟随，切换 VitePress 主题后评论区会自动切换

## 📦 部署到 Cloudflare Pages

配置完成后，推送代码：

```bash
git add .
git commit -m "Add Giscus comment system"
git push origin main
```

Cloudflare Pages 会自动重新构建和部署，评论功能也会在线上生效！

## 🎉 完成！

访问你的网站，滚动到任意文档页面底部，就能看到评论区了！

所有评论都会保存在：
https://github.com/pengyanhua/full-stack-roadmap/discussions

---

需要帮助？查看：
- [Giscus 官方文档](https://giscus.app/zh-CN)
- [VitePress 主题自定义](https://vitepress.dev/guide/extending-default-theme)
