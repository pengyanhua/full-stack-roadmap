# server_client_components.tsx

::: info 文件信息
- 📄 原文件：`02_server_client_components.tsx`
- 🔤 语言：TypeScript (Next.js / React)
:::

Next.js App Router 引入了 React Server Components（RSC），从根本上改变了组件的渲染方式和数据获取模式。理解 Server Component 和 Client Component 的边界是掌握 App Router 的关键。

## 完整代码

```tsx
/**
 * ============================================================
 *            Next.js 服务端组件与客户端组件
 * ============================================================
 * Next.js App Router 引入了 React Server Components（RSC），
 * 从根本上改变了组件的渲染方式和数据获取模式。
 * 理解 Server Component 和 Client Component 的边界是掌握
 * App Router 的关键。
 *
 * 适用版本：Next.js 14 / 15 (App Router)
 * ============================================================
 */

import { Suspense } from 'react';
import Link from 'next/link';
import { cookies, headers } from 'next/headers';
import { notFound } from 'next/navigation';

// ============================================================
//                    1. 服务端组件基础
// ============================================================

/**
 * 【什么是 Server Component】
 *
 * 在 App Router 中，所有组件默认是 Server Component：
 * - 在服务器上渲染，不会被发送到客户端
 * - 可以直接访问后端资源（数据库、文件系统、环境变量）
 * - 可以使用 async/await 直接获取数据
 * - 不会增加客户端 JavaScript 包体积
 * - 不能使用浏览器 API（window、document）
 * - 不能使用 React Hook（useState、useEffect 等）
 * - 不能使用事件处理器（onClick、onChange 等）
 *
 * 【Server Component 的优势】
 *
 * 1. 零客户端 JavaScript
 *    → 组件代码不发送到浏览器，减少包体积
 * 2. 直接访问后端
 *    → 无需创建 API 路由，直接查询数据库
 * 3. 自动代码分割
 *    → 每个 Server Component 自动进行代码分割
 * 4. 流式渲染
 *    → 可以逐步发送 HTML，提升首屏速度
 * 5. 安全性
 *    → 敏感逻辑和密钥留在服务端，不会泄露
 */

// --- 基本的 Server Component ---
// app/users/page.tsx（默认就是 Server Component，无需声明）
async function UsersPage() {
    // 直接在组件中获取数据 — 这在客户端组件中不可能做到
    const users = await fetch('https://api.example.com/users', {
        // Next.js 扩展的 fetch 选项
        cache: 'force-cache',  // 静态数据：构建时获取并缓存
    }).then(r => r.json());

    return (
        <div>
            <h1>用户列表</h1>
            <ul>
                {users.map((user: { id: number; name: string }) => (
                    <li key={user.id}>{user.name}</li>
                ))}
            </ul>
        </div>
    );
}

// --- 直接访问数据库 ---
// 在 Server Component 中可以直接查询数据库
async function ProductsPage() {
    // 模拟直接数据库查询（如 Prisma、Drizzle 等 ORM）
    // const products = await db.product.findMany();
    const products = await getProductsFromDB();

    return (
        <div>
            <h1>商品列表</h1>
            {products.map((product) => (
                <div key={product.id}>
                    <h2>{product.name}</h2>
                    <p>价格: ¥{product.price}</p>
                </div>
            ))}
        </div>
    );
}

// --- 访问服务端专有 API ---
async function ServerOnlyPage() {
    // 读取 HTTP 请求头
    const headersList = await headers();
    const userAgent = headersList.get('user-agent') || '';

    // 读取 Cookie
    const cookieStore = await cookies();
    const theme = cookieStore.get('theme')?.value || 'light';

    // 使用环境变量（服务端专用，不会泄露到客户端）
    const apiSecret = process.env.API_SECRET_KEY;

    return (
        <div>
            <p>用户代理: {userAgent.slice(0, 50)}...</p>
            <p>主题偏好: {theme}</p>
            {/* apiSecret 永远不会出现在客户端 */}
        </div>
    );
}


// ============================================================
//                    2. 客户端组件
// ============================================================

/**
 * 【什么是 Client Component】
 *
 * 通过在文件顶部添加 'use client' 指令声明客户端组件：
 * - 在客户端（浏览器）上执行交互逻辑
 * - 可以使用 React Hook（useState、useEffect 等）
 * - 可以使用事件处理器（onClick、onChange 等）
 * - 可以访问浏览器 API（window、localStorage 等）
 * - 仍然会在服务端进行初始 HTML 渲染（SSR）
 *
 * 【'use client' 的工作原理】
 *
 * 'use client' 是一个声明"入口点"的指令：
 * - 标记了服务端和客户端的边界
 * - 该文件及其所有导入的模块都成为客户端代码
 * - 不需要在每个客户端组件文件中都添加
 * - 只需在边界入口点添加即可
 *
 * 【注意】
 * Client Component 并不意味着"只在客户端渲染"：
 * - 初始 HTML 仍然在服务端生成（SSR）
 * - 然后在客户端进行"水合"（hydration）
 * - 水合后才能响应用户交互
 */

// --- 客户端组件基础 ---
// components/Counter.tsx
// 'use client';  // 实际文件中需要放在最顶部

function Counter() {
    // useState 只能在客户端组件中使用
    const [count, setCount] = useState(0);

    return (
        <div>
            <p>计数: {count}</p>
            {/* onClick 事件处理器需要客户端组件 */}
            <button onClick={() => setCount(count + 1)}>+1</button>
            <button onClick={() => setCount(count - 1)}>-1</button>
        </div>
    );
}

// --- 需要浏览器 API 的组件 ---
// components/ThemeToggle.tsx
// 'use client';

function ThemeToggle() {
    const [theme, setTheme] = useState<'light' | 'dark'>('light');

    // useEffect 只能在客户端组件中使用
    useEffect(() => {
        // 访问 localStorage — 浏览器 API
        const saved = localStorage.getItem('theme');
        if (saved === 'light' || saved === 'dark') {
            setTheme(saved);
        }
    }, []);

    function toggleTheme() {
        const newTheme = theme === 'light' ? 'dark' : 'light';
        setTheme(newTheme);
        localStorage.setItem('theme', newTheme);
        // 修改 DOM — 浏览器 API
        document.documentElement.classList.toggle('dark', newTheme === 'dark');
    }

    return (
        <button onClick={toggleTheme}>
            当前主题: {theme === 'light' ? '浅色' : '深色'}
        </button>
    );
}

// --- 表单交互组件 ---
// components/SearchInput.tsx
// 'use client';

function SearchInput({ onSearch }: { onSearch?: (query: string) => void }) {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<string[]>([]);

    // 防抖搜索
    useEffect(() => {
        if (!query.trim()) {
            setResults([]);
            return;
        }

        const timer = setTimeout(async () => {
            const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
            const data = await res.json();
            setResults(data.results);
        }, 300);

        return () => clearTimeout(timer);
    }, [query]);

    return (
        <div>
            <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="搜索..."
            />
            <ul>
                {results.map((result, i) => (
                    <li key={i}>{result}</li>
                ))}
            </ul>
        </div>
    );
}


// ============================================================
//                    3. 渲染模式对比
// ============================================================

/**
 * 【Next.js 渲染模式全景】
 *
 * 1. SSG（静态站点生成 — Static Site Generation）
 *    - 构建时生成 HTML
 *    - 适用于不经常变化的内容（博客、文档）
 *    - App Router 中：默认行为（当没有动态数据时）
 *    - 最快的加载速度，可部署到 CDN
 *
 * 2. SSR（服务端渲染 — Server-Side Rendering）
 *    - 每次请求时在服务端生成 HTML
 *    - 适用于个性化内容（仪表盘、用户首页）
 *    - App Router 中：使用动态函数（cookies、headers）或
 *      设置 { cache: 'no-store' } 时自动启用
 *
 * 3. CSR（客户端渲染 — Client-Side Rendering）
 *    - 在浏览器中渲染（传统 SPA 模式）
 *    - 适用于高度交互的组件
 *    - App Router 中：Client Component + useEffect 获取数据
 *
 * 4. ISR（增量静态再生 — Incremental Static Regeneration）
 *    - 静态生成 + 定时重新验证
 *    - App Router 中：fetch 的 next.revalidate 选项
 *
 * 5. Streaming（流式渲染）
 *    - 逐步发送 HTML 到客户端
 *    - 配合 Suspense 使用，先显示骨架再填充内容
 *    - App Router 的核心特性之一
 *
 * 【App Router 中的缓存策略】
 *
 * fetch('url', { cache: 'force-cache' })     → 静态（SSG）
 * fetch('url', { cache: 'no-store' })         → 动态（SSR）
 * fetch('url', { next: { revalidate: 60 } })  → ISR（60秒重新验证）
 */

// --- SSG: 静态生成 ---
// 没有动态数据的页面自动成为静态页面
async function StaticPage() {
    // force-cache 是默认行为，构建时获取数据
    const data = await fetch('https://api.example.com/static-content', {
        cache: 'force-cache',
    }).then(r => r.json());

    return <div>{data.content}</div>;
}

// --- SSR: 每次请求动态渲染 ---
async function DynamicPage() {
    // no-store 表示每次请求都重新获取
    const data = await fetch('https://api.example.com/realtime-data', {
        cache: 'no-store',
    }).then(r => r.json());

    // 或者使用动态函数（自动触发动态渲染）
    const cookieStore = await cookies();
    const userId = cookieStore.get('userId')?.value;

    return <div>用户 {userId} 的数据: {JSON.stringify(data)}</div>;
}

// --- ISR: 增量静态再生 ---
async function ISRPage() {
    // 每 60 秒重新验证一次
    const data = await fetch('https://api.example.com/products', {
        next: { revalidate: 60 },
    }).then(r => r.json());

    return (
        <div>
            <p>数据每 60 秒更新一次</p>
            <pre>{JSON.stringify(data, null, 2)}</pre>
        </div>
    );
}

// --- CSR: 客户端渲染 ---
// 'use client';
function CSRComponent() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch('/api/client-data')
            .then(r => r.json())
            .then(d => {
                setData(d);
                setLoading(false);
            });
    }, []);

    if (loading) return <p>加载中...</p>;

    return <div>{JSON.stringify(data)}</div>;
}


// ============================================================
//                    4. 组件组合模式
// ============================================================

/**
 * 【Server 包裹 Client 模式】
 *
 * 最常见的组合模式：Server Component 作为父组件，
 * 在服务端获取数据后通过 props 传递给 Client Component。
 *
 * Server Component 可以：
 * - 导入和渲染 Client Component ✅
 * - 将可序列化的数据作为 props 传递 ✅
 * - 将 Server Component 作为 children 传递给 Client Component ✅
 *
 * Client Component 不能：
 * - 直接导入 Server Component ❌
 *   （但可以通过 children 或其他 ReactNode prop 接收）
 *
 * 【组合模式的意义】
 *
 * 将交互逻辑下推到叶子节点（Leaf），
 * 让尽可能多的组件树保持在服务端，
 * 只在真正需要交互的地方使用 Client Component。
 */

// --- 模式1：Server 获取数据，Client 处理交互 ---

// Server Component（无需 'use client'）
async function ProductList() {
    // 在服务端获取数据
    const products = await fetch('https://api.example.com/products', {
        next: { revalidate: 3600 },
    }).then(r => r.json());

    return (
        <div>
            <h1>商品列表</h1>
            {/* 将数据传递给客户端组件 */}
            {products.map((product: Product) => (
                <ProductCard key={product.id} product={product} />
            ))}
        </div>
    );
}

// Client Component — 处理交互逻辑
// 'use client';
function ProductCard({ product }: { product: Product }) {
    const [isWished, setIsWished] = useState(false);

    return (
        <div className="product-card">
            <h3>{product.name}</h3>
            <p>¥{product.price}</p>
            <button onClick={() => setIsWished(!isWished)}>
                {isWished ? '已收藏' : '收藏'}
            </button>
        </div>
    );
}

// --- 模式2：通过 children 传递 Server Component ---

// Client Component — 提供交互式布局
// 'use client';
function InteractivePanel({ children, title }: {
    children: React.ReactNode;
    title: string;
}) {
    const [isExpanded, setIsExpanded] = useState(true);

    return (
        <div className="panel">
            <div
                className="panel-header"
                onClick={() => setIsExpanded(!isExpanded)}
            >
                <h2>{title}</h2>
                <span>{isExpanded ? '收起' : '展开'}</span>
            </div>

            {/* children 可以是 Server Component！ */}
            {isExpanded && (
                <div className="panel-content">
                    {children}
                </div>
            )}
        </div>
    );
}

// Server Component — 在服务端获取和渲染
async function ServerContent() {
    const data = await fetch('https://api.example.com/content').then(r => r.json());
    return <div>{data.html}</div>;
}

// 组合使用（在 Server Component 中）
function PageWithPanel() {
    return (
        <InteractivePanel title="服务端内容面板">
            {/* ServerContent 是 Server Component */}
            {/* 通过 children 传入 Client Component 中 */}
            <ServerContent />
        </InteractivePanel>
    );
}

// --- 模式3：多个 slot 组合 ---

// Client Component — 带标签页的容器
// 'use client';
function TabContainer({
    tabs,
}: {
    tabs: { label: string; content: React.ReactNode }[];
}) {
    const [activeTab, setActiveTab] = useState(0);

    return (
        <div>
            <div className="tab-bar">
                {tabs.map((tab, index) => (
                    <button
                        key={index}
                        className={activeTab === index ? 'active' : ''}
                        onClick={() => setActiveTab(index)}
                    >
                        {tab.label}
                    </button>
                ))}
            </div>
            <div className="tab-content">
                {tabs[activeTab].content}
            </div>
        </div>
    );
}

// Server Component — 组合使用
async function DashboardPage() {
    return (
        <TabContainer
            tabs={[
                {
                    label: '概览',
                    content: <OverviewPanel />,   // Server Component
                },
                {
                    label: '统计',
                    content: <StatsPanel />,       // Server Component
                },
            ]}
        />
    );
}


// ============================================================
//                    5. 数据序列化
// ============================================================

/**
 * 【可序列化 Props 限制】
 *
 * Server Component 传递给 Client Component 的 props
 * 必须是可序列化的（能被 JSON.stringify 处理）。
 *
 * ✅ 可以传递：
 * - 基本类型：string, number, boolean, null, undefined
 * - 纯对象和数组（值也需要可序列化）
 * - Date 对象（会被序列化为字符串）
 * - Map 和 Set（React 支持的序列化类型）
 * - TypedArray 和 ArrayBuffer
 * - React 元素（JSX）/ ReactNode
 * - Promise（配合 use() Hook）
 *
 * ❌ 不能传递：
 * - 函数 / 方法
 * - 类实例（会丢失原型链和方法）
 * - DOM 节点
 * - Symbol
 * - 循环引用的对象
 *
 * 【常见错误与解决方案】
 *
 * 错误：将回调函数作为 prop 传递给 Client Component
 * 解决：使用 Server Action 替代，或在客户端定义函数
 */

// --- 正确：传递可序列化数据 ---

interface UserData {
    id: number;
    name: string;
    email: string;
    createdAt: string;  // Date 序列化为字符串
}

// Server Component
async function UserProfile() {
    const user: UserData = await fetchUser(1);

    // ✅ 传递纯数据对象
    return <UserCard user={user} />;
}

// Client Component
// 'use client';
function UserCard({ user }: { user: UserData }) {
    const [isEditing, setIsEditing] = useState(false);
    return (
        <div>
            <h2>{user.name}</h2>
            <p>{user.email}</p>
            <button onClick={() => setIsEditing(!isEditing)}>
                {isEditing ? '取消' : '编辑'}
            </button>
        </div>
    );
}

// --- 错误示范：不能传递函数 ---

/**
 * ❌ 以下代码会报错：
 *
 * // Server Component
 * function ParentServer() {
 *     function handleClick() {
 *         console.log('clicked');
 *     }
 *     // 错误！不能将函数传递给 Client Component
 *     return <ChildClient onClick={handleClick} />;
 * }
 *
 * ✅ 正确做法 — 使用 Server Action：
 */

// Server Component + Server Action
async function FormPage() {
    // Server Action — 在服务端执行的函数
    async function submitForm(formData: FormData) {
        'use server';
        const name = formData.get('name') as string;
        // 直接操作数据库
        // await db.user.create({ data: { name } });
        console.log('服务端处理:', name);
    }

    return (
        <form action={submitForm}>
            <input name="name" placeholder="用户名" />
            <SubmitButton />
        </form>
    );
}

// Client Component — 处理表单交互状态
// 'use client';
function SubmitButton() {
    // useFormStatus 需要在 <form> 内的 Client Component 中使用
    // const { pending } = useFormStatus();

    return (
        <button type="submit">
            提交
        </button>
    );
}

// --- 传递 React 元素（可序列化） ---

// Server Component
async function CardWithActions() {
    const data = await fetchData();

    // ✅ JSX 元素是可序列化的
    return (
        <ClientWrapper
            header={<h1>{data.title}</h1>}
            footer={<p>最后更新: {data.updatedAt}</p>}
        >
            <p>{data.description}</p>
        </ClientWrapper>
    );
}

// Client Component
// 'use client';
function ClientWrapper({
    header,
    footer,
    children,
}: {
    header: React.ReactNode;
    footer: React.ReactNode;
    children: React.ReactNode;
}) {
    const [showFooter, setShowFooter] = useState(true);

    return (
        <div className="card">
            {header}
            {children}
            <button onClick={() => setShowFooter(!showFooter)}>
                {showFooter ? '隐藏' : '显示'}页脚
            </button>
            {showFooter && footer}
        </div>
    );
}


// ============================================================
//                    6. 流式渲染
// ============================================================

/**
 * 【流式渲染（Streaming SSR）】
 *
 * 传统 SSR：服务端必须完成所有数据获取后，
 * 才能发送完整的 HTML → 用户看到白屏等待。
 *
 * 流式渲染：服务端可以逐步发送 HTML 片段：
 * 1. 先发送页面骨架（布局、静态内容）
 * 2. 异步数据就绪后，发送对应的 HTML 片段
 * 3. 客户端逐步填充内容，用户更早看到页面
 *
 * 【Suspense Boundary】
 *
 * React <Suspense> 是流式渲染的关键：
 * - 包裹异步组件
 * - fallback 属性提供加载占位符
 * - 异步内容就绪后自动替换占位符
 * - 可以嵌套多层，实现精细化的加载控制
 *
 * 【loading.tsx 与 Suspense 的关系】
 *
 * loading.tsx 本质上就是自动创建的 Suspense boundary：
 *
 * // Next.js 内部等价于：
 * <Layout>
 *   <Suspense fallback={<Loading />}>
 *     <Page />
 *   </Suspense>
 * </Layout>
 *
 * 手动使用 <Suspense> 可以实现更精细的控制。
 */

// --- 基本流式渲染 ---
// app/dashboard/page.tsx
async function StreamingDashboard() {
    return (
        <div className="dashboard">
            <h1>仪表盘</h1>

            {/* 快速内容：立即显示 */}
            <WelcomeMessage />

            {/* 慢速内容：包裹在 Suspense 中，逐步加载 */}
            <Suspense fallback={<p>加载统计数据...</p>}>
                <SlowStatistics />
            </Suspense>

            <Suspense fallback={<p>加载最近活动...</p>}>
                <SlowRecentActivity />
            </Suspense>

            <Suspense fallback={<p>加载推荐内容...</p>}>
                <SlowRecommendations />
            </Suspense>
        </div>
    );
}

// 快速组件 — 不需要异步数据
function WelcomeMessage() {
    return <p>欢迎回来！以下是您的仪表盘概览。</p>;
}

// 慢速组件 — 需要从 API 获取数据
async function SlowStatistics() {
    // 模拟慢速 API 调用
    const stats = await fetch('https://api.example.com/stats', {
        cache: 'no-store',
    }).then(r => r.json());

    return (
        <div className="stats-grid">
            <div>总用户: {stats.totalUsers}</div>
            <div>今日活跃: {stats.dailyActive}</div>
            <div>收入: ¥{stats.revenue}</div>
        </div>
    );
}

async function SlowRecentActivity() {
    const activities = await fetch('https://api.example.com/activities').then(r => r.json());

    return (
        <ul>
            {activities.map((a: { id: number; text: string }) => (
                <li key={a.id}>{a.text}</li>
            ))}
        </ul>
    );
}

async function SlowRecommendations() {
    const recs = await fetch('https://api.example.com/recommendations').then(r => r.json());

    return (
        <div className="recommendations">
            {recs.map((r: { id: number; title: string }) => (
                <div key={r.id}>{r.title}</div>
            ))}
        </div>
    );
}

// --- 嵌套 Suspense（细粒度控制）---
async function ProductDetailPage({
    params,
}: {
    params: Promise<{ id: string }>;
}) {
    const { id } = await params;

    return (
        <div>
            {/* 第一层：产品基本信息先加载 */}
            <Suspense fallback={<ProductSkeleton />}>
                <ProductInfo id={id} />

                {/* 第二层：评论在产品信息之后加载 */}
                <Suspense fallback={<p>加载评论...</p>}>
                    <ProductReviews id={id} />
                </Suspense>

                {/* 第二层：推荐商品独立加载 */}
                <Suspense fallback={<p>加载推荐...</p>}>
                    <RelatedProducts id={id} />
                </Suspense>
            </Suspense>
        </div>
    );
}

function ProductSkeleton() {
    return (
        <div className="skeleton">
            <div className="skeleton-image" />
            <div className="skeleton-title" />
            <div className="skeleton-price" />
        </div>
    );
}


// ============================================================
//                    7. 选择策略
// ============================================================

/**
 * 【何时使用 Server Component】
 *
 * 使用场景：
 * - 数据获取（直接查询数据库或调用 API）
 * - 访问后端资源（文件系统、环境变量）
 * - 保护敏感信息（API 密钥、数据库连接）
 * - 渲染大量静态内容（减少客户端 JS）
 * - 使用服务端专用库（如 Node.js 原生模块）
 *
 * 【何时使用 Client Component】
 *
 * 使用场景：
 * - 需要用户交互（onClick、onChange 等事件）
 * - 需要状态管理（useState、useReducer）
 * - 需要生命周期/副作用（useEffect）
 * - 需要浏览器 API（localStorage、geolocation）
 * - 使用依赖状态/效果的自定义 Hook
 * - 使用 React Class Component
 *
 * 【决策流程图】
 *
 * 组件是否需要交互？
 * ├── 否 → Server Component ✅
 * └── 是 → 能否将交互部分提取为子组件？
 *     ├── 能 → 父组件 Server + 子组件 Client ✅
 *     └── 不能 → Client Component ✅
 *
 * 【常见组件的选择建议】
 *
 * | 组件类型       | 推荐     | 原因                   |
 * |---------------|---------|----------------------|
 * | 页面布局       | Server  | 纯展示，无需交互          |
 * | 导航栏         | Client  | 可能有菜单展开、活跃状态    |
 * | 数据表格       | Server  | 服务端获取并渲染          |
 * | 排序/筛选按钮   | Client  | 需要交互                |
 * | 文章内容       | Server  | 大量静态内容             |
 * | 评论表单       | Client  | 需要输入和提交           |
 * | 侧边栏         | 混合    | 布局 Server + 折叠 Client |
 * | 图片轮播       | Client  | 需要滑动交互             |
 * | SEO 元数据     | Server  | 服务端生成              |
 */

// --- 示例：混合组件设计 ---

// Server Component — 文章页面
async function ArticlePage({
    params,
}: {
    params: Promise<{ slug: string }>;
}) {
    const { slug } = await params;
    const article = await fetchArticle(slug);

    if (!article) notFound();

    return (
        <article>
            {/* 静态内容 — Server Component */}
            <h1>{article.title}</h1>
            <p className="meta">
                作者: {article.author} | 发布: {article.date}
            </p>
            <div className="content">{article.content}</div>

            {/* 交互部分 — Client Component */}
            <LikeButton articleId={article.id} initialCount={article.likes} />

            {/* 评论区 — Server 获取数据 + Client 处理交互 */}
            <Suspense fallback={<p>加载评论...</p>}>
                <CommentsSection articleId={article.id} />
            </Suspense>
        </article>
    );
}

// Client Component — 点赞按钮
// 'use client';
function LikeButton({
    articleId,
    initialCount,
}: {
    articleId: number;
    initialCount: number;
}) {
    const [likes, setLikes] = useState(initialCount);
    const [liked, setLiked] = useState(false);

    async function handleLike() {
        setLiked(!liked);
        setLikes(prev => liked ? prev - 1 : prev + 1);

        // 调用 API
        await fetch(`/api/articles/${articleId}/like`, {
            method: 'POST',
        });
    }

    return (
        <button onClick={handleLike} className={liked ? 'liked' : ''}>
            {liked ? '已赞' : '点赞'} ({likes})
        </button>
    );
}

// Server Component — 获取评论数据
async function CommentsSection({ articleId }: { articleId: number }) {
    const comments = await fetch(
        `https://api.example.com/articles/${articleId}/comments`,
        { cache: 'no-store' }
    ).then(r => r.json());

    return (
        <div className="comments">
            <h3>评论 ({comments.length})</h3>
            {comments.map((c: Comment) => (
                <div key={c.id} className="comment">
                    <strong>{c.author}</strong>
                    <p>{c.text}</p>
                </div>
            ))}
            {/* 评论表单需要客户端交互 */}
            <CommentForm articleId={articleId} />
        </div>
    );
}


// ============================================================
//                    8. 最佳实践
// ============================================================

/**
 * 【Server Component 与 Client Component 最佳实践】
 *
 * ✅ 推荐做法：
 * 1. 默认使用 Server Component，只在需要交互时才使用 Client Component
 * 2. 将交互逻辑下推到组件树的叶子节点
 * 3. 通过 children/ReactNode props 在 Client 中渲染 Server 内容
 * 4. 使用 Server Action 处理表单提交和数据变更
 * 5. 利用 Suspense 实现流式渲染，提升用户体验
 * 6. 敏感数据（API 密钥、数据库 URL）只在 Server Component 中使用
 * 7. 大型依赖库（markdown 解析、语法高亮）放在 Server Component 中
 * 8. 合理使用 fetch 缓存策略（force-cache / no-store / revalidate）
 * 9. 为异步组件提供有意义的 loading 状态（骨架屏优于 spinner）
 *
 * ❌ 避免做法：
 * 1. 在 Client Component 中获取可以在服务端获取的数据
 *    → 增加客户端 JS 体积和请求瀑布
 * 2. 将整个页面标记为 'use client'
 *    → 丧失 Server Component 的所有优势
 * 3. 向 Client Component 传递不可序列化的 props（函数、类实例）
 *    → 会导致运行时错误
 * 4. 在 Server Component 中使用 useState / useEffect
 *    → 这些 Hook 只能在客户端使用
 * 5. 在 Client Component 中使用 cookies() / headers()
 *    → 这些是服务端专用 API
 * 6. 过度使用 'use client'，给不需要的文件都加上该指令
 *    → 分析真正需要交互的组件，精准标记
 * 7. 忽略流式渲染和 Suspense
 *    → 用户会面对长时间白屏
 * 8. 在 Server Component 之间通过全局变量共享状态
 *    → 每个请求应该是独立的
 */

// --- 示例：组件边界划分 ---

/**
 * 一个典型的电商产品页面组件树：
 *
 * ProductPage (Server)            ← 获取产品数据
 * ├── ProductBreadcrumb (Server)  ← 纯展示
 * ├── ProductImages (Client)      ← 图片轮播需要交互
 * ├── ProductInfo (Server)        ← 纯展示
 * │   ├── ProductTitle (Server)   ← 纯展示
 * │   ├── ProductPrice (Server)   ← 纯展示
 * │   └── AddToCart (Client)      ← 按钮交互 + 状态
 * ├── ProductTabs (Client)        ← 标签切换需要交互
 * │   ├── Description (Server)    ← 通过 children 传入
 * │   ├── Specifications (Server) ← 通过 children 传入
 * │   └── Reviews (Server)        ← 通过 children 传入
 * │       └── ReviewForm (Client) ← 表单交互
 * └── RelatedProducts (Server)    ← 服务端获取推荐数据
 *     └── ProductCard (Client)    ← 收藏按钮需要交互
 */


// ============================================================
//                    辅助类型与函数（示例用）
// ============================================================

// 引入客户端 Hook（实际使用时通过 'use client' 指令启用）
import { useState, useEffect } from 'react';

// 类型定义
interface Product {
    id: number;
    name: string;
    price: number;
    description: string;
}

interface Comment {
    id: number;
    author: string;
    text: string;
}

interface Article {
    id: number;
    title: string;
    author: string;
    date: string;
    content: string;
    likes: number;
}

// 模拟数据获取函数
async function getProductsFromDB(): Promise<Product[]> {
    return [
        { id: 1, name: '机械键盘', price: 299, description: 'Cherry MX 轴体' },
        { id: 2, name: '无线鼠标', price: 199, description: '人体工学设计' },
    ];
}

async function fetchUser(id: number): Promise<UserData> {
    return { id, name: '张三', email: 'zhangsan@example.com', createdAt: '2024-01-01' };
}

async function fetchData(): Promise<{ title: string; description: string; updatedAt: string; html: string }> {
    return { title: '标题', description: '描述', updatedAt: '2024-12-01', html: '<p>内容</p>' };
}

async function fetchArticle(slug: string): Promise<Article | null> {
    return { id: 1, title: `文章: ${slug}`, author: '作者', date: '2024-12-01', content: '内容...', likes: 42 };
}

// 占位组件
async function ProductInfo({ id }: { id: string }) {
    return <div>产品信息 #{id}</div>;
}

async function ProductReviews({ id }: { id: string }) {
    return <div>产品评论 #{id}</div>;
}

async function RelatedProducts({ id }: { id: string }) {
    return <div>相关产品 #{id}</div>;
}

async function OverviewPanel() {
    return <div>概览面板</div>;
}

async function StatsPanel() {
    return <div>统计面板</div>;
}

function CommentForm({ articleId }: { articleId: number }) {
    return <div>评论表单 #{articleId}</div>;
}
```
