# 更新 Giscus 配置

如果从 https://giscus.app/zh-CN 获取的配置与代码中不同，按以下步骤更新：

## 1. 记录你的配置

从 Giscus 网站复制：
- `data-repo-id`: `R_kgDO...`
- `data-category-id`: `DIC_kwDO...`

## 2. 更新配置文件

编辑文件：`docs/.vitepress/theme/components/Comment.vue`

找到第 9-11 行：

```vue
<Giscus
  repo="pengyanhua/full-stack-roadmap"
  repo-id="R_kgDONd3yOA"           ← 替换为你的 repo-id
  category="General"                ← 确认分类名称
  category-id="DIC_kwDONd3yOM4ClXkK" ← 替换为你的 category-id
  ...
/>
```

## 3. 保存并提交

```bash
git add docs/.vitepress/theme/components/Comment.vue
git commit -m "Update Giscus configuration"
git push origin main
```

## 4. 等待部署

Cloudflare Pages 会自动重新部署，大约 1-2 分钟后生效。

---

**注意**：如果你已经正确安装了 Giscus App，现有的配置应该就能工作！
