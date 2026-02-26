/**
 * ============================================================
 *              Next.js 数据获取与缓存
 * ============================================================
 * 本文件介绍 Next.js App Router 中的数据获取策略和缓存机制。
 *
 * Next.js 扩展了原生 fetch API，提供自动缓存和重验证能力，
 * 支持静态生成（SSG）、增量静态再生（ISR）和动态渲染。
 *
 * 核心概念：
 * - 服务端组件中直接 fetch，无需 getStaticProps / getServerSideProps
 * - Next.js 14 默认缓存 fetch；Next.js 15 默认不缓存
 * - 增量静态再生（ISR）通过 revalidate 选项实现
 * ============================================================
 */

import { Suspense } from 'react';
import { revalidatePath, revalidateTag } from 'next/cache';
import { unstable_cache, unstable_noStore as noStore } from 'next/cache';
import { cookies, headers } from 'next/headers';

// ============================================================
//               1. fetch 扩展
// ============================================================

/**
 * 【Next.js 扩展的 fetch】
 *
 * Next.js 在服务端组件中扩展了原生 fetch API：
 * - 自动去重：同一渲染过程中，相同请求只执行一次
 * - 缓存控制：通过 cache 和 next 选项控制缓存行为
 * - 标签系统：通过 next.tags 标记请求，支持按需重验证
 *
 * 版本差异：
 *   Next.js 14：默认 force-cache（自动缓存）
 *   Next.js 15：默认 no-store（不缓存）
 */

// --- 基本用法：在服务端组件中直接 fetch ---
async function ProductPage() {
    // 服务端组件中直接使用 async/await，无需 useEffect / useState
    const res = await fetch('https://api.example.com/products');
    const products = await res.json();

    return (
        <div>
            <h1>商品列表</h1>
            {products.map((product: any) => (
                <div key={product.id}>
                    <h2>{product.name}</h2>
                    <p>价格：¥{product.price}</p>
                </div>
            ))}
        </div>
    );
}

// --- fetch 选项扩展 ---
async function FetchOptionsDemo() {
    const res = await fetch('https://api.example.com/data', {
        method: 'GET',
        headers: { 'Authorization': 'Bearer token' },
        // Next.js 扩展选项
        cache: 'force-cache',          // 缓存策略
        next: {
            revalidate: 3600,           // 重验证间隔（秒）
            tags: ['products'],          // 缓存标签
        },
    });
    return res.json();
}


// ============================================================
//               2. 静态数据获取
// ============================================================

/**
 * 【静态数据获取 — Static Data Fetching】
 *
 * 使用 cache: 'force-cache' 实现构建时数据获取：
 * - 数据在构建时获取并缓存，后续请求直接返回
 * - 适用于不经常变化的数据（博客、文档、配置等）
 * - 等价于 Pages Router 中的 getStaticProps
 */

async function StaticBlogPage() {
    const res = await fetch('https://api.example.com/posts', {
        cache: 'force-cache',  // Next.js 14 中这是默认值
    });
    const posts = await res.json();

    return (
        <div>
            {posts.map((post: any) => (
                <article key={post.id}>
                    <h2>{post.title}</h2>
                    <p>{post.excerpt}</p>
                </article>
            ))}
        </div>
    );
}

// --- generateStaticParams 预生成静态页面（等价于 getStaticPaths）---
export async function generateStaticParams() {
    const res = await fetch('https://api.example.com/posts');
    const posts = await res.json();
    return posts.map((post: any) => ({ slug: post.slug }));
}

async function BlogPostPage({ params }: { params: Promise<{ slug: string }> }) {
    const { slug } = await params;
    const res = await fetch(`https://api.example.com/posts/${slug}`, {
        cache: 'force-cache',
    });
    const post = await res.json();

    return (
        <article>
            <h1>{post.title}</h1>
            <div dangerouslySetInnerHTML={{ __html: post.content }} />
        </article>
    );
}


// ============================================================
//               3. 动态数据获取
// ============================================================

/**
 * 【动态数据获取 — Dynamic Data Fetching】
 *
 * 使用 cache: 'no-store' 每次请求时重新获取数据：
 * - 请求结果不被缓存，每次访问都执行新请求
 * - 适用于实时数据（用户仪表盘、实时报价等）
 * - 等价于 Pages Router 中的 getServerSideProps
 */

async function DashboardPage() {
    const res = await fetch('https://api.example.com/dashboard', {
        cache: 'no-store',  // Next.js 15 中这是默认值
    });
    const data = await res.json();

    return (
        <div>
            <h1>实时仪表盘</h1>
            <p>活跃用户：{data.activeUsers}</p>
            <p>今日订单：{data.todayOrders}</p>
        </div>
    );
}

// --- segment config 控制整个路由的渲染模式 ---
export const dynamic = 'force-dynamic';        // 强制动态渲染
// export const dynamic = 'force-static';       // 强制静态渲染
// export const dynamic = 'auto';               // 默认，自动判断
export const runtime = 'nodejs';                // 'nodejs' | 'edge'

// --- 使用动态函数自动切换为动态渲染 ---
async function DynamicByHeaders() {
    // cookies() 或 headers() 会自动触发动态渲染
    const cookieStore = await cookies();
    const theme = cookieStore.get('theme')?.value ?? 'light';
    const headersList = await headers();
    const userAgent = headersList.get('user-agent');

    return <div>主题：{theme}，UA：{userAgent}</div>;
}


// ============================================================
//               4. ISR 增量静态再生
// ============================================================

/**
 * 【增量静态再生 — Incremental Static Regeneration】
 *
 * ISR 结合了静态生成和动态渲染的优点：
 * - 首次请求返回静态缓存页面
 * - 超过 revalidate 时间后，后台重新生成页面
 * - 采用 stale-while-revalidate 策略：
 *   1. 用户 A 请求 → 返回缓存页面（即使过期也先返回）
 *   2. 后台触发重新生成
 *   3. 用户 B 请求 → 返回新生成的页面
 */

async function ISRProductPage() {
    const res = await fetch('https://api.example.com/products', {
        next: { revalidate: 60 },   // 60 秒后过期
    });
    const products = await res.json();

    return (
        <div>
            <h1>商品列表（每分钟更新）</h1>
            {products.map((p: any) => <div key={p.id}>{p.name} - ¥{p.price}</div>)}
        </div>
    );
}

// --- 页面级 revalidate ---
export const revalidate = 60;   // 整个路由段每 60 秒重验证

// --- 不同数据不同刷新频率 ---
async function MixedRevalidationPage() {
    // 商品信息：每小时刷新
    const products = await fetch('https://api.example.com/products', {
        next: { revalidate: 3600 },
    }).then(r => r.json());

    // 评论信息：每 5 分钟刷新（页面整体 revalidate 取最短值 300 秒）
    const reviews = await fetch('https://api.example.com/reviews', {
        next: { revalidate: 300 },
    }).then(r => r.json());

    return <div>{/* 商品和评论展示 */}</div>;
}


// ============================================================
//               5. 按需重验证
// ============================================================

/**
 * 【按需重验证 — On-Demand Revalidation】
 *
 * 除了基于时间的自动重验证，还可以主动触发：
 * - revalidatePath(path)：重验证指定路径的页面
 * - revalidateTag(tag)：重验证带有指定标签的所有请求
 * - 典型场景：CMS 内容更新后通知网站刷新
 */

// --- 步骤 1：fetch 时打标签 ---
async function TaggedFetchPage() {
    const res = await fetch('https://api.example.com/articles', {
        next: { tags: ['articles'] },
    });
    const articles = await res.json();
    return (
        <div>
            {articles.map((a: any) => <article key={a.id}>{a.title}</article>)}
        </div>
    );
}

// --- 步骤 2：在 Server Action 中触发重验证 ---
async function publishArticle(formData: FormData) {
    'use server';
    await saveArticleToDB(formData);
    revalidateTag('articles');   // 所有带 'articles' 标签的缓存失效
}

// --- revalidatePath 示例 ---
async function updateProduct(formData: FormData) {
    'use server';
    const id = formData.get('id') as string;
    await updateProductInDB(id, formData);

    revalidatePath('/products');             // 重验证 /products 页面
    revalidatePath('/products/[id]', 'page'); // 重验证动态路由
    revalidatePath('/', 'layout');            // 重验证整个站点
}

// --- Webhook 触发重验证 ---
async function handleWebhook(request: Request) {
    const secret = request.headers.get('x-webhook-secret');
    if (secret !== process.env.WEBHOOK_SECRET) {
        return new Response('Unauthorized', { status: 401 });
    }
    const body = await request.json();
    if (body.type === 'article.updated') revalidateTag('articles');
    if (body.type === 'product.updated') revalidateTag('products');
    return Response.json({ revalidated: true });
}


// ============================================================
//               6. 并行数据获取
// ============================================================

/**
 * 【并行数据获取 — Parallel Data Fetching】
 *
 * 多个独立的数据请求应并行执行：
 * - 使用 Promise.all() 同时发起多个请求
 * - 避免请求瀑布流（waterfall），减少总等待时间
 *   顺序：A(200ms) → B(300ms) → C(150ms) = 650ms
 *   并行：A + B + C = 300ms（取最长）
 */

// --- 错误示范：串行请求 ---
async function WaterfallPage() {
    const user = await (await fetch('/api/user')).json();
    const posts = await (await fetch('/api/posts')).json();         // 等 user 完成才开始
    const notifications = await (await fetch('/api/notifications')).json();
    return <div>{/* 渲染数据 */}</div>;
}

// --- 正确做法：Promise.all 并行 ---
async function ParallelPage() {
    const [user, posts, notifications] = await Promise.all([
        fetch('/api/user').then(r => r.json()),
        fetch('/api/posts').then(r => r.json()),
        fetch('/api/notifications').then(r => r.json()),
    ]);
    return (
        <div>
            <h1>欢迎，{user.name}</h1>
            <p>文章数：{posts.length}，通知：{notifications.length}</p>
        </div>
    );
}

// --- Suspense 渐进式加载 ---
async function DashboardLayout() {
    return (
        <div>
            <h1>仪表盘</h1>
            <Suspense fallback={<p>加载用户信息...</p>}>
                <UserProfile />
            </Suspense>
            <Suspense fallback={<p>加载统计数据...</p>}>
                <StatsPanel />
            </Suspense>
        </div>
    );
}

async function UserProfile() {
    const user = await fetch('/api/user', { next: { tags: ['user'] } }).then(r => r.json());
    return <div>用户：{user.name}</div>;
}

async function StatsPanel() {
    const stats = await fetch('/api/stats', { next: { revalidate: 60 } }).then(r => r.json());
    return <div>总访问量：{stats.visits}</div>;
}


// ============================================================
//               7. 缓存层级
// ============================================================

/**
 * 【Next.js 四层缓存体系】
 *
 *   层级                │ 位置    │ 目的               │ 持续时间
 *   ─────────────────────────────────────────────────────────
 *   Request Memoization │ 服务端  │ 同一渲染去重请求     │ 单次渲染
 *   Data Cache          │ 服务端  │ 跨请求/部署缓存数据  │ 持久化
 *   Full Route Cache    │ 服务端  │ 缓存整个 HTML/RSC   │ 持久化
 *   Router Cache        │ 客户端  │ 减少导航时的请求     │ 会话级
 */

// --- Request Memoization：同一渲染周期内自动去重 ---
async function getUser(id: string) {
    const res = await fetch(`https://api.example.com/users/${id}`);
    return res.json();
}

async function UserNameDisplay({ userId }: { userId: string }) {
    const user = await getUser(userId);   // 第一次：实际网络请求
    return <h1>{user.name}</h1>;
}

async function UserEmailDisplay({ userId }: { userId: string }) {
    const user = await getUser(userId);   // 第二次：自动去重，复用结果
    return <p>{user.email}</p>;
}

// --- unstable_cache：缓存非 fetch 数据（如数据库查询）---
const getCachedUser = unstable_cache(
    async (userId: string) => {
        // const user = await db.user.findUnique({ where: { id: userId } });
        return { id: userId, name: '示例用户' };
    },
    ['user-cache'],                  // 缓存键前缀
    { tags: ['users'], revalidate: 3600 }
);

// --- noStore：确保数据完全不被缓存 ---
async function FullyDynamicComponent() {
    noStore();
    const data = await fetchSensitiveData();
    return <div>{/* 实时敏感数据 */}</div>;
}


// ============================================================
//               8. 最佳实践
// ============================================================

/**
 * 【数据获取最佳实践】
 *
 * ✅ 推荐做法：
 * - 在服务端组件中获取数据，减少客户端 bundle 大小
 * - 使用 Promise.all() 并行请求独立数据
 * - 使用 Suspense 实现渐进式加载体验
 * - 为 fetch 请求添加 tags，方便按需重验证
 * - 封装数据获取函数，利用 Request Memoization 自动去重
 * - 在 Next.js 15 中显式指定 cache: 'force-cache' 以启用缓存
 *
 * ❌ 避免做法：
 * - 避免在客户端组件中获取可在服务端获取的数据
 * - 避免串行请求导致瀑布流，增加页面加载时间
 * - 避免将整个页面设为动态渲染，仅让需要动态的部分动态
 * - 避免在循环中单独 await fetch，应收集后 Promise.all
 * - 避免缓存敏感数据（如包含用户隐私的响应）
 *
 * 【缓存策略选择指南】
 *
 *   数据类型            │ 推荐策略
 *   ─────────────────────────────────────
 *   静态内容（文档）      │ force-cache
 *   定期变化（商品列表）   │ revalidate: 60~3600
 *   用户相关（仪表盘）    │ no-store
 *   CMS 内容            │ revalidateTag 按需
 *
 * 【fetch vs 数据库直连】
 *
 *   调用外部 API         → fetch（自动缓存）
 *   访问自身数据库        → 直接查询 + unstable_cache
 *   同一 Next.js 应用 API → 直接调函数，不要 fetch 自己
 */

// --- 封装数据层示例 ---
async function getProducts(category?: string) {
    const url = category
        ? `https://api.example.com/products?category=${category}`
        : 'https://api.example.com/products';

    const res = await fetch(url, {
        next: { tags: ['products'], revalidate: 300 },
    });

    if (!res.ok) throw new Error(`获取商品失败: ${res.status}`);
    return res.json();
}

// --- 错误处理 ---
async function SafeDataPage() {
    try {
        const data = await getProducts();
        return <div>{/* 正常渲染 */}</div>;
    } catch (error) {
        console.error('数据获取失败:', error);
        return <div>数据加载失败，请稍后重试</div>;
    }
}

// 占位函数声明（仅用于类型推断，不可直接运行）
declare function saveArticleToDB(formData: FormData): Promise<void>;
declare function updateProductInDB(id: string, formData: FormData): Promise<void>;
declare function fetchSensitiveData(): Promise<any>;
