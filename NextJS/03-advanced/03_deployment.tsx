/**
 * ============================================================
 *                    Next.js 部署
 * ============================================================
 * 本文件介绍 Next.js 应用的构建与部署策略。
 * 涵盖静态导出、Node.js 服务器、Vercel 平台、
 * Docker 容器化以及环境变量管理。
 *
 * 适用版本：Next.js 14 / 15（App Router）
 * ============================================================
 */

import type { NextConfig } from 'next';

// ============================================================
//                    1. 构建过程
// ============================================================

/**
 * 【next build 构建命令】
 * - 执行 `next build` 将应用编译为生产版本
 * - 自动进行静态分析、代码分割、Tree Shaking
 * - 预渲染所有静态页面
 *
 * 【构建输出标记】
 * ○ (Static)   — 静态 HTML
 * ● (SSG)      — 静态生成（带 generateStaticParams）
 * λ (Dynamic)  — 服务端动态渲染
 *
 * 【.next/ 目录结构】
 * .next/
 * ├── cache/         — 构建缓存
 * ├── server/        — 服务端渲染代码
 * │   └── app/       — App Router 页面
 * ├── static/        — 静态资源（JS/CSS/字体）
 * └── BUILD_ID       — 构建唯一标识
 */

// package.json 常用脚本
const packageScripts = {
    scripts: {
        'dev':     'next dev',
        'dev:turbo': 'next dev --turbo',  // Turbopack 加速
        'build':   'next build',
        'start':   'next start',          // 启动生产服务器
        'lint':    'next lint',
        'analyze': 'ANALYZE=true next build',
    },
};

// 跳过类型检查（不推荐，CI 中应急可用）
const skipCheckConfig: NextConfig = {
    typescript: { ignoreBuildErrors: true },
    eslint: { ignoreDuringBuilds: true },
};


// ============================================================
//                    2. 静态导出
// ============================================================

/**
 * 【output: 'export' 静态导出】
 * - 导出为纯静态 HTML/CSS/JS，部署到任何静态托管
 * - 不需要 Node.js 服务器
 *
 * 【限制条件 — 不支持以下功能】
 * - 服务端渲染（动态 SSR）
 * - API Routes（/api/*）
 * - 中间件（middleware.ts）
 * - 增量静态再生（ISR / revalidate）
 * - next/image 优化 API（需自定义 loader）
 */

const staticExportConfig: NextConfig = {
    output: 'export',
    images: { unoptimized: true },   // 禁用图片优化 API
    trailingSlash: true,             // /about → /about/index.html
};

// 自定义图片 loader（用于 CDN）
function customImageLoader({ src, width, quality }: {
    src: string; width: number; quality?: number;
}) {
    return `https://cdn.example.com${src}?w=${width}&q=${quality || 75}`;
}

// 静态导出的动态路由 — 必须提供 generateStaticParams
export async function generateStaticParams() {
    const posts = await fetch('https://api.example.com/posts').then(r => r.json());
    return posts.map((post: any) => ({ slug: post.slug }));
}

// 构建：next build → 输出 out/ 目录
// 部署：将 out/ 上传到 Nginx、GitHub Pages 等


// ============================================================
//                    3. Node.js 服务器
// ============================================================

/**
 * 【next start — Node.js 服务器部署】
 * - 支持全部 Next.js 功能（SSR、API、ISR、中间件）
 * - 需要 Node.js 运行环境
 *
 * 【standalone 输出模式】
 * - 自动收集依赖，生成独立可运行的目录
 * - 不需要完整 node_modules/，大幅减小部署体积
 * - 特别适合 Docker 容器化部署
 */

const standaloneConfig: NextConfig = {
    output: 'standalone',
};

// standalone 目录结构：
// .next/standalone/
// ├── node_modules/   — 仅必要依赖
// ├── server.js       — 入口文件
// └── .next/server/
//
// 启动：node .next/standalone/server.js
// 注意：需手动复制 public/ 和 .next/static/
// cp -r public .next/standalone/public
// cp -r .next/static .next/standalone/.next/static

// PM2 进程管理
const pm2Config = {
    apps: [{
        name: 'nextjs-app',
        script: '.next/standalone/server.js',
        instances: 'max',            // 所有 CPU 核心
        exec_mode: 'cluster',        // 集群模式
        env: {
            PORT: 3000,
            NODE_ENV: 'production',
        },
        max_memory_restart: '1G',
    }],
};
// 启动：pm2 start ecosystem.config.js
// 日志：pm2 logs nextjs-app


// ============================================================
//                    4. Vercel 部署
// ============================================================

/**
 * 【Vercel — Next.js 官方托管平台】
 * - 零配置部署：推送代码自动构建
 * - 自动 CDN 分发、HTTPS、域名管理
 * - Preview Deployment — 每个 PR 独立预览环境
 * - 内置 Edge Functions、Analytics
 *
 * 【部署流程】
 * 1. 推送代码到 GitHub
 * 2. Vercel 导入项目
 * 3. 自动检测 Next.js 并配置构建
 * 4. 每次推送自动部署
 */

// vercel.json（可选配置）
const vercelConfig = {
    redirects: [
        { source: '/old-path', destination: '/new-path', permanent: true },
    ],
    headers: [
        {
            source: '/api/(.*)',
            headers: [
                { key: 'Access-Control-Allow-Origin', value: '*' },
                { key: 'Cache-Control', value: 'no-store' },
            ],
        },
        {
            source: '/(.*)',
            headers: [
                { key: 'X-Frame-Options', value: 'DENY' },
            ],
        },
    ],
    functions: {
        'app/api/**': { maxDuration: 30, memory: 1024 },
    },
    regions: ['hkg1', 'sin1'],  // 香港、新加坡
};

// Vercel CLI 命令
// npm i -g vercel
// vercel              — 部署预览环境
// vercel --prod       — 部署生产环境
// vercel env pull     — 拉取环境变量到 .env.local


// ============================================================
//                    5. Docker 部署
// ============================================================

/**
 * 【Docker 多阶段构建】
 * - 搭配 output: 'standalone' 效果最佳
 * - 三个阶段：安装依赖 → 构建 → 运行
 * - 最终镜像通常 100-200MB
 */

// --- 推荐的 Dockerfile ---
const dockerfile = `
# 阶段 1：安装依赖
FROM node:20-alpine AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci

# 阶段 2：构建应用
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production
RUN npm run build

# 阶段 3：生产运行（最小化镜像）
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# 安全：创建非 root 用户
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"
CMD ["node", "server.js"]
`;

// .dockerignore
const dockerignore = `
node_modules
.next
.git
*.md
.env*.local
`;

// docker-compose.yml
const dockerCompose = `
version: '3.8'
services:
  nextjs:
    build: .
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
      - NEXT_PUBLIC_API_URL=https://api.example.com
    depends_on:
      - db
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mydb
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
`;

// Docker 常用命令：
// docker build -t my-nextjs-app .
// docker run -p 3000:3000 my-nextjs-app
// docker compose up -d


// ============================================================
//                    6. 环境变量
// ============================================================

/**
 * 【环境变量体系】
 *
 * 加载优先级（高 → 低）：
 * 1. process.env（系统/平台注入）
 * 2. .env.$(NODE_ENV).local
 * 3. .env.local（本地覆盖，git 忽略）
 * 4. .env.$(NODE_ENV)
 * 5. .env（默认值）
 *
 * 【NEXT_PUBLIC_ 前缀】
 * - 带前缀：内联到客户端 JS，浏览器可见
 * - 不带前缀：仅服务端可用（Server Components / API）
 * - 敏感信息绝不使用 NEXT_PUBLIC_！
 *
 * 【构建时 vs 运行时】
 * - NEXT_PUBLIC_ 在构建时替换为字面量
 * - 服务端变量在运行时从 process.env 读取
 */

// .env 文件示例
// DATABASE_URL=postgresql://localhost:5432/mydb     ← 仅服务端
// API_SECRET_KEY=sk_test_xxxxx                      ← 仅服务端
// NEXT_PUBLIC_APP_NAME=我的应用                      ← 客户端可见
// NEXT_PUBLIC_API_URL=https://api.example.com       ← 客户端可见

// Server Component（可访问所有变量）
async function ServerPage() {
    const dbUrl = process.env.DATABASE_URL;              // ✅ 仅服务端
    const secret = process.env.API_SECRET_KEY;           // ✅ 仅服务端
    const appName = process.env.NEXT_PUBLIC_APP_NAME;    // ✅ 两端均可

    const data = await fetch(process.env.NEXT_PUBLIC_API_URL + '/data', {
        headers: { 'Authorization': `Bearer ${secret}` },
    });
    return <div>{appName}</div>;
}

// Client Component（仅公开变量）
// 'use client'
function ClientComponent() {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL;      // ✅ 可用
    // const secret = process.env.API_SECRET_KEY;         // ❌ undefined
    return <p>API: {apiUrl}</p>;
}

// 类型安全的环境变量校验（使用 zod）
// import { z } from 'zod';
// const envSchema = z.object({
//     DATABASE_URL: z.string().url(),
//     API_SECRET_KEY: z.string().min(10),
//     NEXT_PUBLIC_API_URL: z.string().url(),
//     NEXT_PUBLIC_APP_NAME: z.string(),
// });
// export const env = envSchema.parse(process.env);


// ============================================================
//                    7. 最佳实践
// ============================================================

/**
 * 【部署最佳实践】
 *
 * ✅ 推荐做法：
 * 1. 使用 output: 'standalone' 减小部署体积
 * 2. Docker 多阶段构建 + 非 root 用户运行
 * 3. 敏感信息通过平台环境变量注入，不提交到 Git
 * 4. NEXT_PUBLIC_ 变量仅用于非敏感的公开信息
 * 5. 使用 .env.local 管理本地开发配置
 * 6. 生产构建前执行类型检查和 Lint
 * 7. 使用 PM2 或 systemd 管理 Node.js 进程
 * 8. 配置健康检查端点 /api/health
 *
 * ❌ 避免做法：
 * 1. 将 .env.local 或含密钥的 .env 提交到 Git
 * 2. 在 NEXT_PUBLIC_ 中存放 API 密钥 → 浏览器可见
 * 3. 生产环境使用 next dev → 无性能优化
 * 4. Docker 中用 root 用户运行 → 安全风险
 * 5. 忽略构建时的类型错误 → 隐藏 bug
 * 6. 不配置 .dockerignore → 镜像体积膨胀
 * 7. 硬编码环境配置 → 难以多环境切换
 */
