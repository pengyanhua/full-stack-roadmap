# 添加评论功能

使用 Giscus 为文档网站添加基于 GitHub Discussions 的评论系统。

## 为什么选择 Giscus？

- ✅ **无需后端** - 完全基于 GitHub Discussions
- ✅ **完全免费** - 利用 GitHub 基础设施
- ✅ **隐私友好** - 无广告、无追踪
- ✅ **功能完整** - Markdown、回复、emoji、点赞
- ✅ **主题适配** - 自动跟随网站深浅色模式
- ✅ **SEO 友好** - 评论可被搜索引擎索引

## 配置步骤

### 1. 启用 GitHub Discussions

1. 进入仓库 `pengyanhua/full-stack-roadmap`
2. **Settings** → **Features** → 勾选 **Discussions**

### 2. 安装 Giscus App

访问 [Giscus App](https://github.com/apps/giscus) 并安装到你的仓库。

### 3. 获取配置参数

访问 [Giscus 配置页面](https://giscus.app/zh-CN)：

1. **仓库**：输入 `pengyanhua/full-stack-roadmap`
2. **Discussion 分类**：选择 `Announcements`（推荐）
3. **特性**：
   - ✅ 启用主评论区的回复
   - ✅ 在主评论上方放置评论框
   - ✅ 延迟加载评论
4. **主题**：选择 `preferred_color_scheme`（自动适配）

复制页面底部生成的配置参数。

### 4. 安装依赖

```bash
npm install @giscus/vue
```

### 5. 更新主题配置

编辑 `docs/.vitepress/theme/index.ts`：

```typescript
import DefaultTheme from 'vitepress/theme'
import Comment from './components/Comment.vue'
import { h } from 'vue'

export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      'doc-after': () => h(Comment)
    })
  }
}
```

### 6. 更新评论组件

编辑 `docs/.vitepress/theme/components/Comment.vue`，填入从 Giscus 网站获取的配置：

```vue
<template>
  <div class="comment-container">
    <Giscus
      repo="pengyanhua/full-stack-roadmap"
      repo-id="你的仓库ID"
      category="Announcements"
      category-id="你的分类ID"
      mapping="pathname"
      strict="0"
      reactions-enabled="1"
      emit-metadata="0"
      input-position="top"
      theme="preferred_color_scheme"
      lang="zh-CN"
      loading="lazy"
    />
  </div>
</template>
```

### 7. 测试

```bash
npm run docs:dev
```

打开任意文档页面，底部应该显示评论区！

## 效果预览

评论区会显示在每篇文档底部：

- 用户需要登录 GitHub 才能评论
- 支持 Markdown 格式
- 支持回复、点赞、emoji 反应
- 自动跟随网站主题（浅色/深色）

## 管理评论

所有评论都在 GitHub Discussions 中管理：

- 访问：`https://github.com/pengyanhua/full-stack-roadmap/discussions`
- 可以编辑、删除、标记为答案
- 可以锁定讨论、设置分类

## 高级配置

### 按页面禁用评论

在特定页面的 frontmatter 中添加：

```yaml
---
comment: false
---
```

然后更新 `Comment.vue`：

```vue
<script setup>
import { useData } from 'vitepress'
import { computed } from 'vue'

const { frontmatter } = useData()
const showComment = computed(() => frontmatter.value.comment !== false)
</script>

<template>
  <div v-if="showComment" class="comment-container">
    <!-- Giscus 组件 -->
  </div>
</template>
```

### 自定义样式

在 `Comment.vue` 中添加：

```vue
<style scoped>
.comment-container {
  margin-top: 4rem;
  padding: 2rem;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
}
</style>
```

## 迁移到其他评论系统

如果未来需要更复杂的功能，可以考虑：

1. **Waline** - 支持匿名评论、阅读统计
2. **Twikoo** - 私有部署、反垃圾
3. **Cloudflare Pages Functions** - 自己实现评论 API

但对于大多数文档站点，**Giscus 已经足够**！
