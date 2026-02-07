<template>
  <div v-if="showComment" class="comment-container">
    <div class="comment-header">
      <h2>💬 讨论</h2>
      <p>使用 GitHub 账号登录后即可参与讨论</p>
    </div>
    <Giscus
      repo="pengyanhua/full-stack-roadmap"
      repo-id="R_kgDONd3yOA"
      category="General"
      category-id="DIC_kwDONd3yOM4ClXkK"
      mapping="pathname"
      strict="0"
      reactions-enabled="1"
      emit-metadata="0"
      input-position="top"
      :theme="giscusTheme"
      lang="zh-CN"
      loading="lazy"
    />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useData, useRoute } from 'vitepress'
import Giscus from '@giscus/vue'

const { frontmatter, isDark } = useData()
const route = useRoute()

// 根据页面配置决定是否显示评论
const showComment = computed(() => {
  // 首页不显示评论
  if (route.path === '/') return false

  // 指南页面不显示评论
  if (route.path.startsWith('/guide/')) return false

  // 可以在 frontmatter 中设置 comment: false 来禁用评论
  if (frontmatter.value.comment === false) return false

  return true
})

// 主题跟随 VitePress 深浅色模式
const giscusTheme = computed(() => {
  return isDark.value ? 'dark' : 'light'
})
</script>

<style scoped>
.comment-container {
  margin-top: 4rem;
  padding-top: 2rem;
  border-top: 1px solid var(--vp-c-divider);
}

.comment-header {
  margin-bottom: 2rem;
}

.comment-header h2 {
  margin: 0 0 0.5rem 0;
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.comment-header p {
  margin: 0;
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
}

@media (max-width: 768px) {
  .comment-container {
    margin-top: 3rem;
  }

  .comment-header h2 {
    font-size: 1.25rem;
  }
}
</style>
