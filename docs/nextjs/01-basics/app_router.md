# app_router.tsx

::: info 文件信息
- 📄 原文件：`01_app_router.tsx`
- 🔤 语言：TypeScript (Next.js / React)
:::

Next.js 13+ 引入了基于 React Server Components 的 App Router，采用文件系统路由、嵌套布局、流式渲染等现代特性，是 Next.js 推荐的路由方案。

## 完整代码

```tsx
/**
 * ============================================================
 *                Next.js App Router 路由系统
 * ============================================================
 * Next.js 13+ 引入了基于 React Server Components 的 App Router，
 * 采用文件系统路由、嵌套布局、流式渲染等现代特性，
 * 是 Next.js 推荐的路由方案。
 *
 * 适用版本：Next.js 14 / 15 (App Router)
 * ============================================================
 */

import Link from 'next/link';
import { redirect, notFound } from 'next/navigation';
import { useRouter, usePathname, useSearchParams } from 'next/navigation';
import { Suspense } from 'react';
import type { Metadata, ResolvingMetadata } from 'next';

// ============================================================
//                    1. App Router 基础概念
// ============================================================

/**
 * 【App Router 与 Pages Router 对比】
 *
 * Next.js 有两套路由系统：
 *
 * Pages Router (旧版 - pages/ 目录)：
 * - 基于页面的路由
 * - getServerSideProps / getStaticProps 获取数据
 * - _app.tsx / _document.tsx 全局配置
 * - 所有组件默认是客户端组件
 *
 * App Router (新版 - app/ 目录)：
 * - 基于文件夹的路由，支持嵌套布局
 * - React Server Components 为默认
 * - 内置 loading / error / not-found 状态处理
 * - 支持并行路由、拦截路由等高级模式
 * - 使用 fetch 的扩展 API 进行数据获取和缓存
 *
 * 【app/ 目录结构】
 *
 * app/
 * ├── layout.tsx          // 根布局（必需）
 * ├── page.tsx            // 首页 → /
 * ├── loading.tsx         // 加载状态
 * ├── error.tsx           // 错误处理
 * ├── not-found.tsx       // 404 页面
 * ├── globals.css         // 全局样式
 * ├── about/
 * │   └── page.tsx        // → /about
 * ├── blog/
 * │   ├── page.tsx        // → /blog
 * │   ├── layout.tsx      // 博客区域布局
 * │   └── [slug]/
 * │       └── page.tsx    // → /blog/:slug
 * └── dashboard/
 *     ├── layout.tsx      // 仪表盘布局
 *     ├── page.tsx        // → /dashboard
 *     └── settings/
 *         └── page.tsx    // → /dashboard/settings
 *
 * 【关键规则】
 * - 只有 page.tsx 会生成可访问的路由
 * - layout.tsx 在导航时不会重新渲染
 * - 文件夹名称直接映射为 URL 路径段
 */

// --- 根布局（每个 App Router 项目都必须有） ---
// app/layout.tsx
export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="zh-CN">
            <body>
                {/* 根布局包裹所有页面 */}
                <header>全站导航栏</header>
                <main>{children}</main>
                <footer>全站页脚</footer>
            </body>
        </html>
    );
}

// --- 首页 ---
// app/page.tsx
function HomePage() {
    // page.tsx 导出的组件就是该路由的页面
    return (
        <div>
            <h1>欢迎来到我的网站</h1>
            <p>这是首页，对应路由 /</p>
        </div>
    );
}


// ============================================================
//                    2. 页面与布局
// ============================================================

/**
 * 【page.tsx — 页面文件】
 *
 * page.tsx 是路由的核心文件，只有它能使路由可访问：
 * - 必须 default export 一个 React 组件
 * - 自动接收 params 和 searchParams 作为 props
 * - 每次导航都会重新渲染
 *
 * 【layout.tsx — 布局文件】
 *
 * layout.tsx 提供共享的 UI 结构：
 * - 在路由切换时保持状态（不会重新挂载）
 * - 接收 children prop，嵌套渲染子路由
 * - 支持嵌套布局：子布局自动嵌入父布局中
 *
 * 【template.tsx — 模板文件】
 *
 * 与 layout.tsx 类似，但区别在于：
 * - 每次导航都会重新创建实例（重新挂载）
 * - 不保持状态
 * - 适用于需要每次导航都重新初始化的场景
 *   （如进入/退出动画、每次导航都要记录日志）
 */

// --- 博客列表页面 ---
// app/blog/page.tsx
interface BlogPageProps {
    searchParams: Promise<{ page?: string; category?: string }>;
}

async function BlogPage({ searchParams }: BlogPageProps) {
    // Next.js 15 中 searchParams 是一个 Promise
    const params = await searchParams;
    const page = Number(params.page) || 1;
    const category = params.category || 'all';

    return (
        <div>
            <h1>博客文章列表</h1>
            <p>当前页码: {page}，分类: {category}</p>
        </div>
    );
}

// --- 嵌套布局 ---
// app/dashboard/layout.tsx
function DashboardLayout({ children }: { children: React.ReactNode }) {
    return (
        <div className="dashboard">
            {/* 侧边栏在所有仪表盘子页面共享 */}
            <aside className="sidebar">
                <nav>
                    <Link href="/dashboard">概览</Link>
                    <Link href="/dashboard/analytics">分析</Link>
                    <Link href="/dashboard/settings">设置</Link>
                </nav>
            </aside>

            {/* children 是当前激活的子页面 */}
            <section className="content">
                {children}
            </section>
        </div>
    );
}

// --- template.tsx 示例 ---
// app/dashboard/template.tsx
function DashboardTemplate({ children }: { children: React.ReactNode }) {
    // 每次导航到 /dashboard/* 下的任何页面时
    // 这个模板都会重新挂载（layout 则不会）
    console.log('模板重新挂载 — 可以用于记录页面浏览');

    return (
        <div className="template-wrapper">
            {/* 每次导航都会触发动画 */}
            <div className="page-transition">
                {children}
            </div>
        </div>
    );
}


// ============================================================
//                    3. 加载与错误处理
// ============================================================

/**
 * 【loading.tsx — 加载状态】
 *
 * 当页面或布局在加载数据时，自动显示 loading.tsx 的内容：
 * - 基于 React Suspense 实现
 * - 在路由段级别自动创建 Suspense boundary
 * - 即时显示，提升用户体验
 * - 可以是骨架屏、加载动画等
 *
 * 【error.tsx — 错误处理】
 *
 * 当页面渲染出错时，自动显示 error.tsx 的内容：
 * - 基于 React Error Boundary 实现
 * - 必须是客户端组件（'use client'）
 * - 只捕获子组件的错误，不捕获同级 layout 的错误
 * - 提供 reset 函数用于重试
 *
 * 【not-found.tsx — 404 页面】
 *
 * 当调用 notFound() 函数时显示：
 * - app/not-found.tsx 处理根级 404
 * - 也可以在子路由段中定义局部 not-found
 * - 自动返回 404 HTTP 状态码
 */

// --- loading.tsx ---
// app/blog/loading.tsx
function BlogLoading() {
    return (
        <div className="loading-skeleton">
            {/* 骨架屏 */}
            <div className="skeleton-title" />
            <div className="skeleton-card" />
            <div className="skeleton-card" />
            <div className="skeleton-card" />
        </div>
    );
}

// --- error.tsx（必须是客户端组件）---
// app/blog/error.tsx
// 'use client';  // 实际文件中需要加这个指令

function BlogError({
    error,
    reset,
}: {
    error: Error & { digest?: string };
    reset: () => void;
}) {
    return (
        <div className="error-page">
            <h2>博客加载失败</h2>
            <p>错误信息: {error.message}</p>
            {/* reset 会重新渲染该路由段 */}
            <button onClick={reset}>重试</button>
        </div>
    );
}

// --- not-found.tsx ---
// app/not-found.tsx
function NotFoundPage() {
    return (
        <div className="not-found">
            <h1>404 - 页面不存在</h1>
            <p>您访问的页面不存在或已被移除。</p>
            <Link href="/">返回首页</Link>
        </div>
    );
}

// --- 在页面中使用 notFound() ---
// app/blog/[slug]/page.tsx
async function BlogPostPage({
    params,
}: {
    params: Promise<{ slug: string }>;
}) {
    const { slug } = await params;
    const post = await fetchPost(slug);

    // 当文章不存在时，触发 not-found.tsx
    if (!post) {
        notFound();
    }

    return (
        <article>
            <h1>{post.title}</h1>
            <div>{post.content}</div>
        </article>
    );
}


// ============================================================
//                    4. 路由分组与并行路由
// ============================================================

/**
 * 【路由分组 (group)】
 *
 * 使用 (folderName) 括号包裹文件夹名，可以在不影响 URL 的情况下
 * 组织路由结构：
 *
 * app/
 * ├── (marketing)/
 * │   ├── layout.tsx      // 营销页面共用布局
 * │   ├── about/page.tsx  // → /about（URL 中没有 marketing）
 * │   └── blog/page.tsx   // → /blog
 * ├── (shop)/
 * │   ├── layout.tsx      // 商城页面共用布局
 * │   ├── cart/page.tsx   // → /cart
 * │   └── products/page.tsx // → /products
 * └── layout.tsx          // 根布局
 *
 * 用途：
 * - 为不同功能区域使用不同的布局
 * - 按团队/功能模块组织代码
 * - 创建多个根布局
 *
 * 【并行路由 (@slot)】
 *
 * 使用 @folderName 定义具名插槽，实现在同一页面同时渲染多个路由段：
 *
 * app/dashboard/
 * ├── layout.tsx          // 接收 @analytics 和 @team 作为 props
 * ├── page.tsx            // 默认页面
 * ├── @analytics/
 * │   └── page.tsx        // 分析面板（独立路由段）
 * └── @team/
 *     └── page.tsx        // 团队面板（独立路由段）
 *
 * 特点：
 * - 每个 slot 可以独立加载/错误处理
 * - 支持条件渲染（基于权限等）
 * - slot 不影响 URL 结构
 */

// --- 路由分组示例 ---
// app/(marketing)/layout.tsx
function MarketingLayout({ children }: { children: React.ReactNode }) {
    return (
        <div className="marketing-theme">
            {/* 营销页面专用导航 */}
            <nav className="marketing-nav">
                <Link href="/about">关于我们</Link>
                <Link href="/blog">博客</Link>
                <Link href="/pricing">定价</Link>
            </nav>
            {children}
        </div>
    );
}

// --- 并行路由示例 ---
// app/dashboard/layout.tsx（使用并行路由）
function DashboardLayoutWithSlots({
    children,
    analytics,
    team,
}: {
    children: React.ReactNode;
    analytics: React.ReactNode;
    team: React.ReactNode;
}) {
    return (
        <div className="dashboard-grid">
            {/* 主内容区 */}
            <div className="main">{children}</div>

            {/* 分析面板 — 来自 @analytics/page.tsx */}
            <div className="analytics-panel">{analytics}</div>

            {/* 团队面板 — 来自 @team/page.tsx */}
            <div className="team-panel">{team}</div>
        </div>
    );
}

// --- 并行路由的条件渲染 ---
// app/dashboard/layout.tsx（基于角色）
function ConditionalDashboard({
    children,
    admin,
    user,
}: {
    children: React.ReactNode;
    admin: React.ReactNode;
    user: React.ReactNode;
}) {
    const role = getCurrentUserRole();

    return (
        <div>
            {children}
            {/* 根据用户角色显示不同的并行路由 */}
            {role === 'admin' ? admin : user}
        </div>
    );
}


// ============================================================
//                    5. 动态路由
// ============================================================

/**
 * 【动态路由段】
 *
 * Next.js 支持三种动态路由模式：
 *
 * 1. [slug] — 单段动态路由
 *    - /blog/[slug]  →  匹配 /blog/hello-world
 *    - params: { slug: 'hello-world' }
 *
 * 2. [...slug] — 全捕获路由（Catch-all）
 *    - /docs/[...slug]  →  匹配 /docs/a/b/c
 *    - params: { slug: ['a', 'b', 'c'] }
 *    - 不匹配 /docs（无参数时不匹配）
 *
 * 3. [[...slug]] — 可选全捕获路由（Optional Catch-all）
 *    - /docs/[[...slug]]  →  匹配 /docs 和 /docs/a/b/c
 *    - params: { slug: undefined } 或 { slug: ['a', 'b', 'c'] }
 *    - /docs 也会匹配（参数为 undefined）
 *
 * 【generateStaticParams】
 *
 * 用于在构建时预生成静态路由（替代 getStaticPaths）：
 * - 返回一个参数对象数组
 * - 配合动态路由使用
 * - 支持增量静态生成（ISR）
 */

// --- 单段动态路由 ---
// app/blog/[slug]/page.tsx
interface PostPageProps {
    params: Promise<{ slug: string }>;
}

async function PostPage({ params }: PostPageProps) {
    const { slug } = await params;

    // 在服务端组件中直接获取数据
    const post = await fetch(`https://api.example.com/posts/${slug}`);
    const data = await post.json();

    return (
        <article>
            <h1>{data.title}</h1>
            <p>URL 参数: {slug}</p>
        </article>
    );
}

// --- 预生成静态参数 ---
// app/blog/[slug]/page.tsx
async function generateStaticParams() {
    // 构建时获取所有文章的 slug
    const posts = await fetch('https://api.example.com/posts').then(r => r.json());

    return posts.map((post: { slug: string }) => ({
        slug: post.slug,
    }));
    // 返回: [{ slug: 'first-post' }, { slug: 'second-post' }, ...]
}

// --- 全捕获路由 ---
// app/docs/[...slug]/page.tsx
interface DocsPageProps {
    params: Promise<{ slug: string[] }>;
}

async function DocsPage({ params }: DocsPageProps) {
    const { slug } = await params;
    // /docs/getting-started       → slug: ['getting-started']
    // /docs/api/reference/auth    → slug: ['api', 'reference', 'auth']

    const breadcrumbs = slug.map((segment, index) => ({
        label: segment,
        href: '/docs/' + slug.slice(0, index + 1).join('/'),
    }));

    return (
        <div>
            {/* 面包屑导航 */}
            <nav>
                {breadcrumbs.map((crumb) => (
                    <Link key={crumb.href} href={crumb.href}>
                        {crumb.label}
                    </Link>
                ))}
            </nav>
            <h1>文档: {slug.join(' / ')}</h1>
        </div>
    );
}

// --- 可选全捕获路由 ---
// app/shop/[[...categories]]/page.tsx
interface ShopPageProps {
    params: Promise<{ categories?: string[] }>;
}

async function ShopPage({ params }: ShopPageProps) {
    const { categories } = await params;

    // /shop                     → categories: undefined (显示全部)
    // /shop/electronics         → categories: ['electronics']
    // /shop/electronics/phones  → categories: ['electronics', 'phones']

    if (!categories) {
        return <h1>全部商品</h1>;
    }

    return (
        <div>
            <h1>分类: {categories.join(' > ')}</h1>
        </div>
    );
}


// ============================================================
//                    6. 链接与导航
// ============================================================

/**
 * 【Link 组件】
 *
 * Next.js 的 <Link> 组件提供客户端导航：
 * - 自动预取可见链接指向的路由（生产环境）
 * - 不会导致整页刷新
 * - 支持 prefetch 控制
 * - 替代原生 <a> 标签进行应用内导航
 *
 * 【useRouter Hook】
 *
 * 用于程序化导航（只能在客户端组件中使用）：
 * - router.push(url)    → 导航到新页面
 * - router.replace(url) → 替换当前历史记录
 * - router.refresh()    → 刷新当前路由（重新获取数据）
 * - router.back()       → 返回上一页
 * - router.prefetch(url) → 预取路由
 *
 * 【usePathname / useSearchParams】
 *
 * 客户端组件中读取 URL 信息：
 * - usePathname() → 当前路径（如 /blog/hello）
 * - useSearchParams() → 查询参数（如 ?page=1&sort=date）
 *
 * 【redirect() 函数】
 *
 * 服务端重定向：
 * - 在 Server Component 或 Server Action 中使用
 * - 会抛出 NEXT_REDIRECT 错误（内部机制）
 * - 默认 307 临时重定向，可指定 308 永久重定向
 */

// --- Link 组件 ---
function Navigation() {
    return (
        <nav>
            {/* 基本链接 */}
            <Link href="/">首页</Link>

            {/* 动态路由链接 */}
            <Link href="/blog/hello-world">文章详情</Link>

            {/* 使用对象形式 */}
            <Link
                href={{
                    pathname: '/blog',
                    query: { page: '2', sort: 'date' },
                }}
            >
                博客第二页
            </Link>

            {/* 控制预取行为 */}
            <Link href="/heavy-page" prefetch={false}>
                不预取的页面
            </Link>

            {/* 替换历史记录（而非推入） */}
            <Link href="/new-page" replace>
                替换导航
            </Link>

            {/* 滚动到页面顶部（默认为 true） */}
            <Link href="/about" scroll={false}>
                不滚动到顶部
            </Link>
        </nav>
    );
}

// --- useRouter 程序化导航 ---
// 'use client';  // 实际文件中需要加这个指令

function SearchForm() {
    const router = useRouter();

    function handleSearch(term: string) {
        // 程序化导航
        router.push(`/search?q=${encodeURIComponent(term)}`);
    }

    function handleLogout() {
        // 清除登录状态后重定向
        clearSession();
        router.replace('/login');  // replace 不会在历史中留下记录
    }

    function handleDataUpdate() {
        // 刷新当前路由（重新执行 Server Component）
        router.refresh();
    }

    return (
        <div>
            <input onKeyDown={(e) => {
                if (e.key === 'Enter') {
                    handleSearch((e.target as HTMLInputElement).value);
                }
            }} />
            <button onClick={handleLogout}>退出登录</button>
            <button onClick={handleDataUpdate}>刷新数据</button>
            <button onClick={() => router.back()}>返回上一页</button>
        </div>
    );
}

// --- usePathname 和 useSearchParams ---
// 'use client';

function ActiveLink({ href, children }: { href: string; children: React.ReactNode }) {
    const pathname = usePathname();
    const isActive = pathname === href;

    return (
        <Link
            href={href}
            className={isActive ? 'text-blue-600 font-bold' : 'text-gray-600'}
        >
            {children}
        </Link>
    );
}

// 'use client';
function SearchFilters() {
    const searchParams = useSearchParams();
    const router = useRouter();
    const pathname = usePathname();

    // 读取当前搜索参数
    const currentSort = searchParams.get('sort') || 'newest';
    const currentPage = Number(searchParams.get('page')) || 1;

    function updateFilter(key: string, value: string) {
        // 创建新的 URLSearchParams
        const params = new URLSearchParams(searchParams.toString());
        params.set(key, value);

        // 更新 URL（不刷新页面）
        router.push(`${pathname}?${params.toString()}`);
    }

    return (
        <div>
            <select
                value={currentSort}
                onChange={(e) => updateFilter('sort', e.target.value)}
            >
                <option value="newest">最新</option>
                <option value="popular">最热</option>
            </select>
            <p>当前页码: {currentPage}</p>
        </div>
    );
}

// --- 服务端重定向 ---
// app/old-blog/[slug]/page.tsx（服务端组件中使用 redirect）
async function OldBlogRedirect({
    params,
}: {
    params: Promise<{ slug: string }>;
}) {
    const { slug } = await params;

    // 永久重定向到新的 URL
    redirect(`/blog/${slug}`);
    // 308 永久重定向：redirect(`/blog/${slug}`, RedirectType.permanent);
}


// ============================================================
//                    7. 路由拦截
// ============================================================

/**
 * 【拦截路由（Intercepting Routes）】
 *
 * 拦截路由可以在当前布局中加载另一个路由的内容，
 * 而不需要切换到目标路由的完整上下文。
 *
 * 典型场景：
 * - 在信息流中点击照片，弹出模态框显示大图
 * - 直接访问照片 URL 时，显示完整的照片页面
 * - 用户分享照片链接时，看到的是完整页面
 *
 * 【拦截约定】
 *
 * (.)   — 匹配同级路由段
 * (..)  — 匹配上一级路由段
 * (..)(..) — 匹配上两级路由段
 * (...) — 匹配根路由（app 目录）
 *
 * 注意：这些约定基于路由段层级，而非文件系统目录。
 *
 * 【目录结构示例 — 照片模态框】
 *
 * app/
 * ├── feed/
 * │   ├── page.tsx               // 信息流页面
 * │   └── (..)photo/[id]/
 * │       └── page.tsx           // 拦截路由 → 显示为模态框
 * └── photo/[id]/
 *     └── page.tsx               // 真实路由 → 显示完整页面
 *
 * 当从 /feed 点击链接到 /photo/123：
 * - 拦截路由介入 → 在当前页面弹出模态框
 * 当直接访问 /photo/123：
 * - 正常渲染 → 显示完整的照片页面
 */

// --- 信息流页面 ---
// app/feed/page.tsx
function FeedPage() {
    const photos = [
        { id: '1', url: '/images/photo1.jpg', title: '日落' },
        { id: '2', url: '/images/photo2.jpg', title: '山川' },
        { id: '3', url: '/images/photo3.jpg', title: '大海' },
    ];

    return (
        <div className="photo-grid">
            {photos.map((photo) => (
                // 点击时，拦截路由会在模态框中显示
                <Link key={photo.id} href={`/photo/${photo.id}`}>
                    <img src={photo.url} alt={photo.title} />
                </Link>
            ))}
        </div>
    );
}

// --- 拦截路由（模态框形式）---
// app/feed/(..)photo/[id]/page.tsx
async function InterceptedPhotoPage({
    params,
}: {
    params: Promise<{ id: string }>;
}) {
    const { id } = await params;

    return (
        <div className="modal-overlay">
            <div className="modal-content">
                <img src={`/images/photo${id}.jpg`} alt={`照片 ${id}`} />
                <Link href="/feed">关闭</Link>
            </div>
        </div>
    );
}

// --- 真实路由（完整页面）---
// app/photo/[id]/page.tsx
async function FullPhotoPage({
    params,
}: {
    params: Promise<{ id: string }>;
}) {
    const { id } = await params;

    return (
        <div className="photo-detail">
            <img src={`/images/photo${id}.jpg`} alt={`照片 ${id}`} />
            <h1>照片详情 #{id}</h1>
            <p>直接访问或分享此链接时显示完整页面</p>
        </div>
    );
}


// ============================================================
//                    8. 元数据
// ============================================================

/**
 * 【元数据 API】
 *
 * Next.js App Router 提供两种定义元数据的方式：
 *
 * 1. 静态元数据 — 导出 metadata 对象
 *    适用于不依赖动态数据的页面
 *
 * 2. 动态元数据 — 导出 generateMetadata 函数
 *    适用于依赖路由参数或外部数据的页面
 *
 * 【元数据合并规则】
 *
 * - 元数据从根布局到页面逐层合并
 * - 子级元数据会覆盖父级的同名字段
 * - title.template 可以在布局中定义模板
 *
 * 【支持的元数据字段】
 *
 * - title: 页面标题（支持 template）
 * - description: 页面描述
 * - keywords: 关键词
 * - openGraph: Open Graph 社交分享
 * - twitter: Twitter 卡片
 * - robots: 搜索引擎指令
 * - icons: 网站图标
 * - manifest: PWA manifest
 */

// --- 静态元数据 ---
// app/about/page.tsx
export const metadata: Metadata = {
    title: '关于我们',
    description: '了解我们的团队和使命',
    keywords: ['关于', '团队', '使命'],

    // Open Graph（用于社交媒体分享）
    openGraph: {
        title: '关于我们 — MyApp',
        description: '了解我们的团队和使命',
        url: 'https://myapp.com/about',
        siteName: 'MyApp',
        images: [
            {
                url: 'https://myapp.com/og-about.jpg',
                width: 1200,
                height: 630,
                alt: '关于我们',
            },
        ],
        locale: 'zh_CN',
        type: 'website',
    },

    // 搜索引擎指令
    robots: {
        index: true,
        follow: true,
    },
};

// --- 根布局中的 title 模板 ---
// app/layout.tsx
export const rootMetadata: Metadata = {
    title: {
        default: 'MyApp',            // 当子页面没有定义 title 时使用
        template: '%s | MyApp',       // 子页面的 title 会替换 %s
    },
    description: '全栈学习路线图',
};
// 效果：about 页面标题会是 "关于我们 | MyApp"

// --- 动态元数据 ---
// app/blog/[slug]/page.tsx
async function generateMetadataForPost(
    { params }: { params: Promise<{ slug: string }> },
    parent: ResolvingMetadata,
): Promise<Metadata> {
    const { slug } = await params;

    // 获取文章数据
    const post = await fetch(`https://api.example.com/posts/${slug}`).then(
        (r) => r.json()
    );

    // 获取父级元数据（可选）
    const previousImages = (await parent).openGraph?.images || [];

    return {
        title: post.title,
        description: post.excerpt,
        openGraph: {
            title: post.title,
            description: post.excerpt,
            images: [post.coverImage, ...previousImages],
        },
    };
}

// --- 文件约定元数据 ---
/**
 * 除了代码定义，还可以通过特殊文件名定义元数据：
 *
 * app/
 * ├── favicon.ico           // 网站图标
 * ├── icon.png              // 应用图标
 * ├── apple-icon.png        // Apple 触摸图标
 * ├── opengraph-image.jpg   // Open Graph 默认图片
 * ├── twitter-image.jpg     // Twitter 卡片图片
 * ├── sitemap.ts            // 站点地图（可动态生成）
 * └── robots.ts             // robots.txt（可动态生成）
 */


// ============================================================
//                    9. 最佳实践
// ============================================================

/**
 * 【App Router 路由最佳实践】
 *
 * ✅ 推荐做法：
 * 1. 优先使用 App Router 而非 Pages Router（新项目）
 * 2. 合理使用嵌套布局，将共享 UI 放在 layout.tsx 中
 * 3. 使用 loading.tsx 提供即时加载反馈
 * 4. 使用 error.tsx 优雅处理每个路由段的错误
 * 5. 使用路由分组 (group) 组织代码结构，保持 URL 简洁
 * 6. 动态路由配合 generateStaticParams 预生成静态页面
 * 7. 使用 <Link> 组件而非 <a> 标签进行应用内导航
 * 8. 服务端组件中使用 redirect()，客户端组件中使用 useRouter
 * 9. 利用 metadata API 做好 SEO 优化
 * 10. 使用 Suspense 包裹异步组件实现流式渲染
 *
 * ❌ 避免做法：
 * 1. 在客户端组件中使用 redirect() → 应使用 useRouter
 * 2. 在 layout.tsx 中放置需要每次导航都刷新的逻辑 → 使用 template.tsx
 * 3. 过度使用并行路由增加复杂性 → 简单场景用组件组合即可
 * 4. 在 page.tsx 中定义布局相关的 UI → 放到 layout.tsx 中
 * 5. 忘记为 error.tsx 添加 'use client' 指令 → Error Boundary 必须是客户端组件
 * 6. 滥用动态路由 → 能用静态路由就用静态路由
 * 7. 在服务端组件中使用 useRouter/usePathname → 这些是客户端 Hook
 * 8. 将大量数据通过 searchParams 传递 → 使用服务端状态或数据库
 */

// --- 示例：结构良好的路由设计 ---
/**
 * 一个电商项目的路由结构示例：
 *
 * app/
 * ├── layout.tsx                    // 全局布局（导航 + 页脚）
 * ├── page.tsx                      // 首页
 * ├── loading.tsx                   // 全局加载状态
 * ├── error.tsx                     // 全局错误处理
 * ├── not-found.tsx                 // 全局 404
 * │
 * ├── (marketing)/                  // 营销页面组（共享营销布局）
 * │   ├── layout.tsx
 * │   ├── about/page.tsx
 * │   └── pricing/page.tsx
 * │
 * ├── (shop)/                       // 商城功能组
 * │   ├── layout.tsx                // 商城布局（带分类侧栏）
 * │   ├── products/
 * │   │   ├── page.tsx              // 商品列表
 * │   │   ├── loading.tsx           // 列表加载骨架屏
 * │   │   └── [id]/
 * │   │       ├── page.tsx          // 商品详情
 * │   │       └── loading.tsx
 * │   └── cart/page.tsx             // 购物车
 * │
 * └── dashboard/                    // 用户中心
 *     ├── layout.tsx                // 仪表盘布局
 *     ├── page.tsx                  // 概览
 *     ├── @orders/page.tsx          // 订单面板（并行路由）
 *     ├── @notifications/page.tsx   // 通知面板（并行路由）
 *     └── settings/page.tsx         // 设置
 */


// ============================================================
//                    辅助函数（示例用）
// ============================================================

// 模拟数据获取
async function fetchPost(slug: string) {
    return { title: `文章: ${slug}`, content: '文章内容...' };
}

// 模拟获取用户角色
function getCurrentUserRole(): string {
    return 'admin';
}

// 模拟清除会话
function clearSession() {
    // 清除登录状态
}
```
