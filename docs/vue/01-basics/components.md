# components.vue

::: info 文件信息
- 📄 原文件：`02_components.vue`
- 🔤 语言：vue
:::

组件基础知识
【单文件组件 (SFC)】
.vue 文件包含三个部分：
- `<template>`: HTML 模板
- `<script>`: JavaScript 逻辑
- `<style>`: CSS 样式
【script setup】
Vue 3.2+ 推荐的语法：
- 顶层绑定自动暴露给模板
- 更简洁的代码
- 更好的运行时性能

## 完整代码

```vue
<!--
============================================================
                    Vue 3 组件基础
============================================================
组件是 Vue 应用的基本构建块。
通过组件可以将 UI 拆分为独立、可复用的部分。
============================================================
-->

<template>
    <div class="tutorial">
        <h1>Vue 3 组件教程</h1>

        <!-- 使用子组件 -->
        <UserCard
            :user="currentUser"
            :show-email="true"
            @update="handleUserUpdate"
            @delete="handleUserDelete"
        >
            <!-- 插槽内容 -->
            <template #actions>
                <button>自定义操作</button>
            </template>
        </UserCard>

        <!-- 列表渲染组件 -->
        <UserCard
            v-for="user in users"
            :key="user.id"
            :user="user"
            @update="handleUserUpdate"
        />
    </div>
</template>

<script setup>
/**
 * ============================================================
 *                    组件基础知识
 * ============================================================
 *
 * 【单文件组件 (SFC)】
 * .vue 文件包含三个部分：
 * - <template>: HTML 模板
 * - <script>: JavaScript 逻辑
 * - <style>: CSS 样式
 *
 * 【script setup】
 * Vue 3.2+ 推荐的语法：
 * - 顶层绑定自动暴露给模板
 * - 更简洁的代码
 * - 更好的运行时性能
 * ============================================================
 */

import { ref, reactive } from 'vue';

// 导入子组件（自动注册）
import UserCard from './UserCard.vue';

// 响应式数据
const currentUser = ref({
    id: 1,
    name: 'Alice',
    email: 'alice@example.com',
    avatar: '/avatars/alice.png',
});

const users = ref([
    { id: 1, name: 'Alice', email: 'alice@example.com' },
    { id: 2, name: 'Bob', email: 'bob@example.com' },
    { id: 3, name: 'Charlie', email: 'charlie@example.com' },
]);

// 事件处理
function handleUserUpdate(user) {
    console.log('更新用户:', user);
}

function handleUserDelete(userId) {
    console.log('删除用户:', userId);
    users.value = users.value.filter(u => u.id !== userId);
}
</script>
```
