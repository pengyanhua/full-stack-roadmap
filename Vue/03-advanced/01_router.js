/**
 * ============================================================
 *                    Vue Router 路由配置
 * ============================================================
 * Vue Router 是 Vue.js 的官方路由管理器。
 * 用于构建单页面应用（SPA）的页面导航。
 *
 * 安装：npm install vue-router@4
 * ============================================================
 */

import { createRouter, createWebHistory, createWebHashHistory } from 'vue-router';

// ============================================================
//                    1. 基本路由配置
// ============================================================

/**
 * 【路由配置】
 *
 * 每个路由对象包含：
 * - path: URL 路径
 * - name: 路由名称（可选，用于编程式导航）
 * - component: 对应的组件
 * - meta: 元信息（可选）
 * - children: 嵌套路由（可选）
 */

// 导入页面组件
// 方式1：静态导入（适合首屏需要的页面）
import Home from '@/views/Home.vue';
import About from '@/views/About.vue';

// 方式2：动态导入（路由懒加载，按需加载）
const User = () => import('@/views/User.vue');
const UserProfile = () => import('@/views/UserProfile.vue');
const UserSettings = () => import('@/views/UserSettings.vue');
const NotFound = () => import('@/views/NotFound.vue');

// 路由配置数组
const routes = [
    // ============================================================
    //                    基本路由
    // ============================================================

    {
        path: '/',                    // URL 路径
        name: 'Home',                 // 路由名称
        component: Home,              // 对应组件
        meta: {                       // 路由元信息
            title: '首页',
            requiresAuth: false,
        },
    },

    {
        path: '/about',
        name: 'About',
        component: About,
        meta: { title: '关于我们' },
    },

    // ============================================================
    //                    动态路由
    // ============================================================

    /**
     * 【动态路由参数】
     *
     * 使用 :参数名 定义动态段
     * - 在组件中通过 $route.params.参数名 访问
     * - 或使用 useRoute().params.参数名
     */

    {
        path: '/user/:id',            // :id 是动态参数
        name: 'User',
        component: User,
        meta: { title: '用户详情' },

        // 将路由参数作为 props 传递给组件
        props: true,  // 组件会收到 props: { id: '...' }
    },

    {
        // 多个动态参数
        path: '/post/:category/:postId',
        name: 'Post',
        component: () => import('@/views/Post.vue'),
        props: true,  // 组件收到 { category, postId }
    },

    {
        // 可选参数（在参数后加 ?）
        path: '/search/:query?',
        name: 'Search',
        component: () => import('@/views/Search.vue'),
    },

    {
        // 正则约束参数
        path: '/article/:id(\\d+)',   // 只匹配数字
        name: 'Article',
        component: () => import('@/views/Article.vue'),
    },

    {
        // 重复参数（匹配多段）
        path: '/files/:path+',        // 匹配 /files/a/b/c
        name: 'Files',
        component: () => import('@/views/Files.vue'),
    },

    // ============================================================
    //                    嵌套路由
    // ============================================================

    /**
     * 【嵌套路由】
     *
     * 使用 children 定义子路由
     * 父组件需要有 <router-view> 来渲染子路由
     */

    {
        path: '/dashboard',
        name: 'Dashboard',
        component: () => import('@/views/Dashboard.vue'),
        meta: {
            title: '控制台',
            requiresAuth: true,
        },
        // 子路由
        children: [
            {
                path: '',              // 默认子路由 /dashboard
                name: 'DashboardHome',
                component: () => import('@/views/DashboardHome.vue'),
            },
            {
                path: 'analytics',     // /dashboard/analytics
                name: 'Analytics',
                component: () => import('@/views/Analytics.vue'),
            },
            {
                path: 'settings',      // /dashboard/settings
                name: 'Settings',
                component: () => import('@/views/Settings.vue'),
            },
        ],
    },

    // 用户相关嵌套路由
    {
        path: '/users/:id',
        component: () => import('@/views/UserLayout.vue'),
        props: true,
        children: [
            {
                path: '',
                name: 'UserOverview',
                component: () => import('@/views/UserOverview.vue'),
            },
            {
                path: 'profile',
                name: 'UserProfile',
                component: UserProfile,
            },
            {
                path: 'settings',
                name: 'UserSettings',
                component: UserSettings,
            },
        ],
    },

    // ============================================================
    //                    命名视图
    // ============================================================

    /**
     * 【命名视图】
     *
     * 同一个路由可以有多个视图
     * 使用 components (注意是复数) 而不是 component
     */

    {
        path: '/layout',
        components: {
            default: () => import('@/views/MainContent.vue'),
            sidebar: () => import('@/views/Sidebar.vue'),
            header: () => import('@/views/Header.vue'),
        },
    },

    // ============================================================
    //                    重定向和别名
    // ============================================================

    /**
     * 【重定向】
     * 访问某路径时自动跳转到另一个路径
     */

    {
        path: '/home',
        redirect: '/',  // 重定向到首页
    },

    {
        path: '/old-user/:id',
        redirect: to => {
            // 动态重定向
            return { name: 'User', params: { id: to.params.id } };
        },
    },

    /**
     * 【别名】
     * 访问别名路径时显示原路径的内容，但 URL 不变
     */

    {
        path: '/settings',
        alias: ['/preferences', '/config'],  // 三个路径都显示同一个组件
        component: () => import('@/views/Settings.vue'),
    },

    // ============================================================
    //                    404 和通配路由
    // ============================================================

    {
        // 匹配所有未匹配的路由
        path: '/:pathMatch(.*)*',
        name: 'NotFound',
        component: NotFound,
        meta: { title: '页面不存在' },
    },
];


// ============================================================
//                    2. 创建路由实例
// ============================================================

const router = createRouter({
    /**
     * 【历史模式】
     *
     * createWebHistory(): HTML5 History 模式
     * - URL 更美观：/user/123
     * - 需要服务器配置支持
     *
     * createWebHashHistory(): Hash 模式
     * - URL 带 #：/#/user/123
     * - 不需要服务器配置
     */
    history: createWebHistory(import.meta.env.BASE_URL),

    // 路由配置
    routes,

    /**
     * 【滚动行为】
     * 控制路由切换时的滚动位置
     */
    scrollBehavior(to, from, savedPosition) {
        // 如果有保存的位置（浏览器前进/后退），恢复位置
        if (savedPosition) {
            return savedPosition;
        }

        // 如果有锚点，滚动到锚点
        if (to.hash) {
            return {
                el: to.hash,
                behavior: 'smooth',
            };
        }

        // 默认滚动到顶部
        return { top: 0, behavior: 'smooth' };
    },

    /**
     * 【严格模式】
     * 是否严格匹配路径末尾的斜杠
     */
    strict: false,

    /**
     * 【链接激活类名】
     */
    linkActiveClass: 'router-link-active',
    linkExactActiveClass: 'router-link-exact-active',
});


// ============================================================
//                    3. 导航守卫
// ============================================================

/**
 * 【全局前置守卫】
 * 在路由跳转前执行
 * 用于权限验证、登录检查等
 */
router.beforeEach((to, from, next) => {
    console.log('导航从', from.path, '到', to.path);

    // 设置页面标题
    document.title = to.meta.title || 'Vue App';

    // 检查是否需要认证
    if (to.meta.requiresAuth) {
        // 检查登录状态（这里简化处理）
        const isAuthenticated = localStorage.getItem('token');

        if (!isAuthenticated) {
            // 未登录，重定向到登录页
            next({
                path: '/login',
                query: { redirect: to.fullPath },  // 保存目标路径
            });
            return;
        }
    }

    // 继续导航
    next();
});

/**
 * 【全局解析守卫】
 * 在所有组件内守卫和异步路由组件被解析之后调用
 */
router.beforeResolve(async (to) => {
    // 可以在这里进行数据预取
    if (to.meta.requiresData) {
        try {
            // await fetchData();
        } catch (error) {
            // 处理错误，可以取消导航
            return false;
        }
    }
});

/**
 * 【全局后置钩子】
 * 在导航完成后执行
 * 不接收 next 函数，不能改变导航
 */
router.afterEach((to, from, failure) => {
    if (failure) {
        console.error('导航失败:', failure);
        return;
    }

    // 发送页面访问统计
    // analytics.trackPageView(to.fullPath);

    // 隐藏加载状态
    // hideLoading();
});

/**
 * 【错误处理】
 */
router.onError((error) => {
    console.error('路由错误:', error);
});


// ============================================================
//                    4. 路由元信息用法
// ============================================================

/**
 * 【路由元信息 meta】
 *
 * 可以在 meta 中存储任意信息：
 * - requiresAuth: 是否需要登录
 * - roles: 允许的角色
 * - title: 页面标题
 * - breadcrumb: 面包屑
 */

// 在守卫中使用 meta
router.beforeEach((to, from, next) => {
    // 检查路由及其所有父路由的 meta
    const requiresAuth = to.matched.some(record => record.meta.requiresAuth);

    // 检查角色权限
    const requiredRoles = to.meta.roles;
    if (requiredRoles) {
        const userRole = 'admin';  // 假设从 store 获取
        if (!requiredRoles.includes(userRole)) {
            next({ name: 'Forbidden' });
            return;
        }
    }

    next();
});


// ============================================================
//                    5. 导出路由实例
// ============================================================

export default router;


// ============================================================
//                    6. 组件中使用路由（示例代码）
// ============================================================

/**
 * 在组件中使用路由
 *
 * ```vue
 * <template>
 *   <!-- 声明式导航 -->
 *   <router-link to="/">首页</router-link>
 *   <router-link :to="{ name: 'User', params: { id: 123 }}">用户</router-link>
 *   <router-link to="/about" active-class="active">关于</router-link>
 *
 *   <!-- 路由出口 -->
 *   <router-view></router-view>
 *
 *   <!-- 带过渡动画的路由 -->
 *   <router-view v-slot="{ Component }">
 *     <transition name="fade" mode="out-in">
 *       <component :is="Component" />
 *     </transition>
 *   </router-view>
 *
 *   <!-- 缓存路由组件 -->
 *   <router-view v-slot="{ Component }">
 *     <keep-alive>
 *       <component :is="Component" />
 *     </keep-alive>
 *   </router-view>
 * </template>
 *
 * <script setup>
 * import { useRouter, useRoute } from 'vue-router';
 *
 * // 路由器实例（用于导航）
 * const router = useRouter();
 *
 * // 当前路由信息
 * const route = useRoute();
 *
 * // 获取路由参数
 * console.log('当前路径:', route.path);
 * console.log('路由参数:', route.params);
 * console.log('查询参数:', route.query);
 * console.log('路由名称:', route.name);
 * console.log('路由元信息:', route.meta);
 *
 * // 编程式导航
 * function goToUser(id) {
 *   // 方式1：字符串路径
 *   router.push(`/user/${id}`);
 *
 *   // 方式2：对象
 *   router.push({ path: `/user/${id}` });
 *
 *   // 方式3：命名路由 + 参数
 *   router.push({ name: 'User', params: { id } });
 *
 *   // 方式4：带查询参数
 *   router.push({ path: '/search', query: { q: 'vue' }});
 * }
 *
 * // 替换当前路由（不会留下历史记录）
 * function replaceRoute() {
 *   router.replace({ name: 'Home' });
 * }
 *
 * // 前进后退
 * function goBack() {
 *   router.back();     // 等同于 router.go(-1)
 * }
 *
 * function goForward() {
 *   router.forward();  // 等同于 router.go(1)
 * }
 * </script>
 * ```
 */
