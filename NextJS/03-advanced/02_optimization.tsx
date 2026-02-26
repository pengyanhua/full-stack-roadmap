/**
 * ============================================================
 *                    Next.js 性能优化
 * ============================================================
 * Next.js 内置了大量性能优化工具和策略。
 * 本文件涵盖 Image、Font、Script、Metadata 等核心优化手段，
 * 以及打包分析和 Core Web Vitals 优化策略。
 *
 * 适用版本：Next.js 14 / 15（App Router）
 * ============================================================
 */

import Image from 'next/image';
import Script from 'next/script';
import { Inter, Noto_Sans_SC } from 'next/font/google';
import localFont from 'next/font/local';
import dynamic from 'next/dynamic';
import type { Metadata, Viewport } from 'next';

// ============================================================
//                    1. Image 组件
// ============================================================

/**
 * 【next/image 图片优化】
 * - 自动 WebP/AVIF 格式转换
 * - 响应式图片（自动生成多种尺寸）
 * - 懒加载 — 进入视口才加载
 * - 防止布局偏移（CLS）— 自动占位
 * - 图片优化 API（/_next/image）按需压缩
 */

import heroImage from '@/public/images/hero.jpg';

function ImageExamples() {
    return (
        <div>
            {/* 静态导入 — 自动获取 width/height，支持 blur 占位 */}
            <Image
                src={heroImage}
                alt="首页横幅图片"
                placeholder="blur"    // 自动生成模糊占位图
                priority              // 首屏图片：禁用懒加载，预加载
            />

            {/* 远程图片 — 必须指定 width 和 height */}
            <Image
                src="https://example.com/photo.jpg"
                alt="用户头像"
                width={200}
                height={200}
            />

            {/* fill 模式 — 填满父容器 */}
            <div style={{ position: 'relative', width: '100%', height: 400 }}>
                <Image
                    src="/images/banner.jpg"
                    alt="横幅"
                    fill
                    style={{ objectFit: 'cover' }}
                    sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
                    // sizes 帮助浏览器选择合适的图片尺寸
                />
            </div>
        </div>
    );
}

// next.config.js 图片配置
const nextConfigImages = {
    images: {
        remotePatterns: [
            { protocol: 'https' as const, hostname: 'cdn.example.com', pathname: '/images/**' },
        ],
        formats: ['image/avif', 'image/webp'],
    },
};


// ============================================================
//                    2. Font 优化
// ============================================================

/**
 * 【next/font 字体优化】
 * - 构建时下载字体文件，零运行时请求
 * - 自动使用 CSS size-adjust，消除布局偏移 (CLS = 0)
 * - 字体文件自托管，不向 Google 发送请求
 * - 支持 Google Fonts、本地字体、variable fonts
 */

// Google Fonts
const inter = Inter({
    subsets: ['latin'],
    display: 'swap',
    variable: '--font-inter',
});

// 中文字体
const notoSansSC = Noto_Sans_SC({
    subsets: ['latin'],
    weight: ['400', '700'],
    variable: '--font-noto-sans',
    preload: false,               // 中文字体较大，不预加载
});

// 本地字体
const customFont = localFont({
    src: [
        { path: '../fonts/Custom-Regular.woff2', weight: '400', style: 'normal' },
        { path: '../fonts/Custom-Bold.woff2', weight: '700', style: 'normal' },
    ],
    variable: '--font-custom',
    display: 'swap',
    fallback: ['system-ui', 'Arial'],
});

// 在 Layout 中应用字体
function RootLayout({ children }: { children: React.ReactNode }) {
    return (
        <html lang="zh-CN" className={`${inter.variable} ${notoSansSC.variable}`}>
            <body className={inter.className}>{children}</body>
        </html>
    );
}

// Tailwind CSS 中使用字体变量
// tailwind.config.ts → theme.extend.fontFamily:
// sans: ['var(--font-inter)'], chinese: ['var(--font-noto-sans)']


// ============================================================
//                    3. Script 管理
// ============================================================

/**
 * 【next/script 脚本管理】
 * - 精确控制第三方脚本的加载时机
 *
 * 【加载策略】
 * - beforeInteractive: 页面水合之前（polyfill 等关键脚本）
 * - afterInteractive: 页面水合之后（分析工具，默认值）
 * - lazyOnload: 浏览器空闲时（聊天插件等低优先级）
 */

function ScriptExample() {
    return (
        <>
            {/* 关键脚本 — 最先加载 */}
            <Script
                src="https://cdn.example.com/polyfill.min.js"
                strategy="beforeInteractive"
            />

            {/* 分析工具 — 页面可交互后加载 */}
            <Script
                src="https://www.googletagmanager.com/gtag/js?id=GA_ID"
                strategy="afterInteractive"
            />
            <Script id="google-analytics" strategy="afterInteractive">
                {`
                    window.dataLayer = window.dataLayer || [];
                    function gtag(){dataLayer.push(arguments);}
                    gtag('js', new Date());
                    gtag('config', 'GA_MEASUREMENT_ID');
                `}
            </Script>

            {/* 低优先级 — 浏览器空闲时加载 */}
            <Script
                src="https://widget.intercom.io/widget/APP_ID"
                strategy="lazyOnload"
            />

            {/* 回调函数 */}
            <Script
                src="https://maps.googleapis.com/maps/api/js"
                strategy="afterInteractive"
                onLoad={() => console.log('Maps 加载完成')}
                onError={(e) => console.error('Maps 加载失败:', e)}
            />
        </>
    );
}


// ============================================================
//                    4. Metadata API
// ============================================================

/**
 * 【Metadata API】
 * - App Router 使用 Metadata API 管理 SEO 信息
 * - 支持静态和动态两种定义方式
 * - 自动处理 <head> 中的 meta 标签
 */

// 静态 Metadata（在 layout.tsx 或 page.tsx 中导出）
export const metadata_example: Metadata = {
    title: {
        default: '我的网站',
        template: '%s | 我的网站',     // 子页面：文章标题 | 我的网站
    },
    description: '一个使用 Next.js 构建的现代化网站',
    openGraph: {
        title: '我的网站',
        description: '使用 Next.js 构建',
        url: 'https://example.com',
        images: [{ url: '/og-image.jpg', width: 1200, height: 630 }],
        locale: 'zh_CN',
        type: 'website',
    },
    twitter: {
        card: 'summary_large_image',
        title: '我的网站',
        images: ['https://example.com/twitter-image.jpg'],
    },
    robots: { index: true, follow: true },
    icons: { icon: '/favicon.ico', apple: '/apple-touch-icon.png' },
};

// 动态 Metadata（根据路由参数生成）
type BlogProps = { params: Promise<{ slug: string }> };

export async function generateMetadata({ params }: BlogProps): Promise<Metadata> {
    const { slug } = await params;
    const post = await fetch(`https://api.example.com/posts/${slug}`).then(r => r.json());

    return {
        title: post.title,
        description: post.excerpt,
        openGraph: {
            title: post.title,
            images: [{ url: post.coverImage }],
            type: 'article',
            publishedTime: post.createdAt,
        },
    };
}

// Viewport 配置（Next.js 14+ 独立导出）
export const viewport: Viewport = {
    width: 'device-width',
    initialScale: 1,
    themeColor: [
        { media: '(prefers-color-scheme: light)', color: '#ffffff' },
        { media: '(prefers-color-scheme: dark)', color: '#000000' },
    ],
};


// ============================================================
//                    5. 静态资源
// ============================================================

/**
 * 【public/ 目录】
 * - public/ 中的文件通过 / 路径直接访问
 * - 不经过构建工具处理
 * - 适合 favicon、robots.txt、sitemap.xml
 *
 * 【目录结构】
 * public/
 * ├── favicon.ico
 * ├── robots.txt
 * └── images/logo.svg
 */

function StaticAssetExample() {
    return (
        <div>
            <Image src="/images/logo.svg" alt="Logo" width={120} height={40} priority />
            <a href="/documents/guide.pdf" download>下载指南</a>
        </div>
    );
}

// robots.ts（Next.js 14+ 使用 TypeScript 生成）
function robotsConfig() {
    return {
        rules: [{ userAgent: '*', allow: '/', disallow: ['/api/', '/admin/'] }],
        sitemap: 'https://example.com/sitemap.xml',
    };
}

// sitemap.ts
async function sitemapConfig() {
    const posts = await fetch('https://api.example.com/posts').then(r => r.json());
    return [
        { url: 'https://example.com', lastModified: new Date(), priority: 1 },
        ...posts.map((p: any) => ({
            url: `https://example.com/blog/${p.slug}`,
            lastModified: new Date(p.updatedAt),
        })),
    ];
}


// ============================================================
//                    6. 打包优化
// ============================================================

/**
 * 【Tree Shaking】
 * - 自动删除未引用的导出（需 ES Module 格式）
 *
 * 【Dynamic Import 动态导入】
 * - next/dynamic 实现组件级代码分割
 *
 * 【Bundle Analyzer】
 * - @next/bundle-analyzer 可视化打包结果
 * - 运行：ANALYZE=true npm run build
 */

// 动态导入重型组件
const RichTextEditor = dynamic(() => import('@/components/RichTextEditor'), {
    loading: () => <div>编辑器加载中...</div>,
    ssr: false,  // 仅客户端渲染
});

const ChartDashboard = dynamic(() => import('@/components/ChartDashboard'), {
    loading: () => <p>图表加载中...</p>,
});

const AdminPanel = dynamic(() => import('@/components/AdminPanel'));

function DynamicImportExample({ isAdmin }: { isAdmin: boolean }) {
    return (
        <div>
            <RichTextEditor />
            <ChartDashboard />
            {isAdmin && <AdminPanel />}  {/* 非管理员不会加载代码 */}
        </div>
    );
}

// next.config.js 优化配置
const nextConfigOptimization = {
    experimental: {
        optimizePackageImports: [
            'lucide-react', '@heroicons/react', 'lodash', 'date-fns',
        ],
    },
};


// ============================================================
//                    7. Core Web Vitals
// ============================================================

/**
 * 【Core Web Vitals 核心指标】
 * - LCP (Largest Contentful Paint): 最大内容绘制 → 目标 < 2.5s
 * - INP (Interaction to Next Paint): 交互响应 → 目标 < 200ms
 * - CLS (Cumulative Layout Shift): 布局偏移 → 目标 < 0.1
 */

// LCP 优化：首屏图片预加载
function LCPOptimization() {
    return (
        <Image
            src="/images/hero.jpg"
            alt="首屏大图"
            width={1200} height={600}
            priority         // 预加载，不懒加载
            sizes="100vw"
        />
        // 另外在 layout 中添加 preconnect:
        // <link rel="preconnect" href="https://cdn.example.com" />
    );
}

// CLS 优化：预留空间
function CLSOptimization() {
    return (
        <div>
            {/* 图片始终指定宽高 */}
            <Image src="/photo.jpg" alt="照片" width={400} height={300} />

            {/* 嵌入内容使用 aspect-ratio */}
            <div style={{ aspectRatio: '16/9', width: '100%' }}>
                <iframe src="https://www.youtube.com/embed/xxx" />
            </div>

            {/* 动态内容预留 min-height */}
            <div style={{ minHeight: 200 }}>{/* 异步内容 */}</div>
        </div>
    );
}

// 性能监控
// 'use client'
// import { useReportWebVitals } from 'next/web-vitals';
// export function WebVitals() {
//     useReportWebVitals((metric) => {
//         // metric: { name, value, rating }
//         // rating: 'good' | 'needs-improvement' | 'poor'
//         fetch('/api/analytics', {
//             method: 'POST',
//             body: JSON.stringify(metric),
//         });
//     });
// }


// ============================================================
//                    8. 最佳实践
// ============================================================

/**
 * 【性能优化最佳实践】
 *
 * ✅ 推荐做法：
 * 1. 首屏图片加 priority，非首屏使用默认懒加载
 * 2. 使用 next/font 加载字体，消除 FOUT/CLS
 * 3. 第三方脚本分级：关键用 afterInteractive，非关键用 lazyOnload
 * 4. 使用 next/dynamic 对重型组件进行代码分割
 * 5. 为 Image 提供 sizes 属性优化响应式加载
 * 6. 定期用 bundle-analyzer 分析打包体积
 * 7. 使用 generateMetadata 为动态页面生成 SEO 信息
 *
 * ❌ 避免做法：
 * 1. 所有图片都加 priority → 等于全部不优先
 * 2. 使用 <img> 替代 next/image → 失去自动优化
 * 3. 在 <head> 手动插入 <link> 加载字体 → 外部请求 + CLS
 * 4. 将大型库完整导入 → 应按需导入（如 lodash/debounce）
 * 5. 所有脚本用 beforeInteractive → 阻塞首屏渲染
 * 6. 忽略 CLS → 图片/字体/动态内容未预留空间
 */
