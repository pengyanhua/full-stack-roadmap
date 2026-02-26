# route_handlers.tsx

::: info 文件信息
- 📄 原文件：`03_route_handlers.tsx`
- 🔤 语言：TypeScript (Next.js / React)
:::

本文件介绍 Next.js App Router 中的 Route Handlers。Route Handlers 使用 Web API 的 Request/Response 对象创建自定义请求处理程序，替代 Pages Router 的 API Routes。

## 完整代码

```tsx
/**
 * ============================================================
 *          Next.js Route Handlers (API 路由)
 * ============================================================
 * 本文件介绍 Next.js App Router 中的 Route Handlers。
 *
 * Route Handlers 使用 Web API 的 Request/Response 对象
 * 创建自定义请求处理程序，替代 Pages Router 的 API Routes。
 *
 * 核心概念：
 * - 定义在 route.ts 文件中（不是 page.tsx）
 * - 导出 HTTP 方法函数：GET, POST, PUT, PATCH, DELETE
 * - 使用标准 Web API：Request, Response, Headers
 * ============================================================
 */

import { NextRequest, NextResponse } from 'next/server';
import { cookies, headers } from 'next/headers';
import { revalidateTag } from 'next/cache';

// ============================================================
//               1. Route Handler 基础
// ============================================================

/**
 * 【Route Handler — 路由处理器】
 *
 * 文件约定：app/api/xxx/route.ts → /api/xxx
 * 注意：route.ts 和 page.tsx 不能在同一目录中共存！
 * 支持的方法：GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS
 */

// --- 基本 GET / POST ---
export async function GET() {
    return Response.json({ message: '你好，世界！' });
}

export async function POST(request: Request) {
    const body = await request.json();
    // const post = await db.post.create({ data: body });
    return Response.json({ message: '创建成功', data: body }, { status: 201 });
}

// --- 动态路由 CRUD ---
// app/api/products/[id]/route.ts
async function handleGetProduct(
    request: Request,
    { params }: { params: Promise<{ id: string }> }
) {
    const { id } = await params;
    // const product = await db.product.findUnique({ where: { id } });
    // if (!product) return Response.json({ error: '不存在' }, { status: 404 });
    return Response.json({ id, name: '示例商品', price: 99 });
}

async function handleUpdateProduct(
    request: Request,
    { params }: { params: Promise<{ id: string }> }
) {
    const { id } = await params;
    const body = await request.json();
    // await db.product.update({ where: { id }, data: body });
    return Response.json({ id, ...body });
}

async function handleDeleteProduct(
    request: Request,
    { params }: { params: Promise<{ id: string }> }
) {
    const { id } = await params;
    // await db.product.delete({ where: { id } });
    return new Response(null, { status: 204 });
}


// ============================================================
//               2. 请求处理
// ============================================================

/**
 * 【请求处理 — Request Handling】
 *
 * 使用标准 Web Request 或 NextRequest（扩展版）：
 * - nextUrl：解析后的 URL（方便获取 searchParams）
 * - cookies：Cookie 操作方法
 * - geo / ip：地理位置信息（Vercel 部署时可用）
 */

// --- URL 参数 ---
async function handleSearch(request: NextRequest) {
    const searchParams = request.nextUrl.searchParams;
    const query = searchParams.get('q');            // ?q=关键词
    const page = searchParams.get('page') ?? '1';
    const limit = searchParams.get('limit') ?? '10';

    // const results = await db.product.findMany({
    //     where: { name: { contains: query } },
    //     skip: (Number(page) - 1) * Number(limit),
    //     take: Number(limit),
    // });

    return Response.json({ query, page: Number(page), results: [] });
}

// --- 请求头 ---
async function handleWithHeaders(request: NextRequest) {
    // 方式一：从 request 读取
    const authorization = request.headers.get('authorization');
    // 方式二：使用 next/headers
    const headersList = await headers();
    const userAgent = headersList.get('user-agent');
    return Response.json({ authorization, userAgent });
}

// --- Cookies ---
async function handleWithCookies(request: NextRequest) {
    // 方式一：从 NextRequest 读取
    const theme = request.cookies.get('theme')?.value;
    // 方式二：使用 next/headers
    const cookieStore = await cookies();
    const token = cookieStore.get('session-token')?.value;
    return Response.json({ theme, token });
}

// --- 请求体的多种读取方式 ---
async function handleRequestBody(request: Request) {
    // const json = await request.json();         // JSON
    // const form = await request.formData();     // FormData
    // const text = await request.text();         // 纯文本
    // const buf  = await request.arrayBuffer();  // 二进制
    return Response.json({ received: true });
}


// ============================================================
//               3. 响应构建
// ============================================================

/**
 * 【响应构建 — Response Building】
 *
 * - Response.json()：标准 JSON 响应
 * - NextResponse.json()：扩展版，支持 cookies 设置等
 * - NextResponse.redirect()：重定向
 * - ReadableStream：流式响应
 */

// --- NextResponse 设置 Cookie ---
async function jsonWithCookie() {
    const res = NextResponse.json({ data: 'hello' }, { status: 200 });
    res.cookies.set('visited', 'true', {
        httpOnly: true,
        secure: true,
        sameSite: 'lax',
        maxAge: 60 * 60 * 24 * 7,   // 7 天
    });
    return res;
}

// --- 重定向 ---
async function handleRedirect(request: NextRequest) {
    return NextResponse.redirect(new URL('/login', request.url));
}

// --- 流式响应 ---
async function handleStream() {
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
        async start(controller) {
            for (const chunk of ['第一部分\n', '第二部分\n', '第三部分\n']) {
                controller.enqueue(encoder.encode(chunk));
                await new Promise(r => setTimeout(r, 1000));
            }
            controller.close();
        },
    });
    return new Response(stream, {
        headers: { 'Content-Type': 'text/plain; charset=utf-8' },
    });
}

// --- SSE (Server-Sent Events) ---
async function handleSSE() {
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
        async start(controller) {
            let count = 0;
            const interval = setInterval(() => {
                count++;
                const data = JSON.stringify({ count, time: new Date().toISOString() });
                controller.enqueue(encoder.encode(`data: ${data}\n\n`));
                if (count >= 10) { clearInterval(interval); controller.close(); }
            }, 1000);
        },
    });
    return new Response(stream, {
        headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' },
    });
}


// ============================================================
//               4. 动态与静态
// ============================================================

/**
 * 【动态与静态 Route Handler】
 *
 * 仅 GET + 不使用动态函数 → 构建时静态缓存
 *
 * 以下情况自动切换为动态：
 * - 使用 Request 对象读取 headers / cookies
 * - 使用 POST / PUT / DELETE 等方法
 * - 使用 cookies()、headers() 等动态函数
 * - 配置 dynamic = 'force-dynamic'
 */

// --- 静态 Route Handler ---
async function handleStaticConfig() {
    return Response.json({
        version: '1.0.0',
        features: ['dark-mode', 'i18n'],
    });
}

// --- 配置选项 ---
// export const dynamic = 'force-dynamic';   // 强制动态
// export const revalidate = 60;              // 每 60 秒重验证
// export const runtime = 'edge';             // Edge Runtime

// --- 带缓存标签 ---
async function handleTaggedGet() {
    const data = await fetch('https://api.example.com/data', {
        next: { tags: ['api-data'] },
    });
    return Response.json(await data.json());
}

// 触发重验证
async function handleRevalidate(request: Request) {
    const body = await request.json();
    if (body.secret !== process.env.REVALIDATION_SECRET) {
        return Response.json({ error: '未授权' }, { status: 401 });
    }
    revalidateTag('api-data');
    return Response.json({ revalidated: true });
}


// ============================================================
//               5. CORS 处理
// ============================================================

/**
 * 【CORS 跨域配置】
 *
 * API 被其他域名调用时需配置 CORS：
 * 1. Route Handler 中手动设置响应头
 * 2. 封装 withCors 工具函数
 * 3. next.config.js 全局配置
 */

const corsHeaders = {
    'Access-Control-Allow-Origin': 'https://example.com',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Max-Age': '86400',
};

// --- OPTIONS 预检 ---
export async function OPTIONS() {
    return new Response(null, { status: 204, headers: corsHeaders });
}

// --- 带 CORS 的响应 ---
async function handleCorsGet() {
    return Response.json({ items: ['a', 'b'] }, { headers: corsHeaders });
}

// --- 封装 CORS 工具函数 ---
function withCors(response: Response, origin?: string): Response {
    const h = new Headers(response.headers);
    h.set('Access-Control-Allow-Origin', origin ?? '*');
    h.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    h.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    return new Response(response.body, { status: response.status, headers: h });
}

// --- next.config.js 全局方式 ---
// const nextConfig = {
//     async headers() {
//         return [{
//             source: '/api/:path*',
//             headers: [
//                 { key: 'Access-Control-Allow-Origin', value: '*' },
//                 { key: 'Access-Control-Allow-Methods', value: 'GET,POST,PUT,DELETE' },
//             ],
//         }];
//     },
// };


// ============================================================
//               6. 认证集成
// ============================================================

/**
 * 【认证集成 — Authentication】
 *
 * 常见认证模式：
 * - JWT Token 验证
 * - Session Cookie 检查
 * - 封装认证中间件函数
 * - 与 NextAuth.js / Auth.js 集成
 */

// --- JWT Token 验证 ---
async function handleProtectedApi(request: NextRequest) {
    const authHeader = request.headers.get('authorization');
    if (!authHeader?.startsWith('Bearer ')) {
        return Response.json({ error: '缺少认证令牌' }, { status: 401 });
    }

    const token = authHeader.split(' ')[1];
    try {
        // const payload = await verifyJWT(token);
        const payload = { userId: '123', role: 'admin' };
        return Response.json({ user: payload });
    } catch {
        return Response.json({ error: '令牌无效或已过期' }, { status: 401 });
    }
}

// --- 封装认证中间件 ---
type AuthHandler = (
    request: NextRequest,
    context: { params: Promise<any>; user: any }
) => Promise<Response>;

function withAuth(handler: AuthHandler) {
    return async (request: NextRequest, context: { params: Promise<any> }) => {
        const cookieStore = await cookies();
        const sessionToken = cookieStore.get('session-token')?.value;
        if (!sessionToken) {
            return Response.json({ error: '未登录' }, { status: 401 });
        }
        // const session = await verifySession(sessionToken);
        const session = { userId: '123', role: 'user' };
        return handler(request, { ...context, user: session });
    };
}

// 使用认证中间件
const protectedGet = withAuth(async (request, { user }) => {
    return Response.json({ message: `欢迎，用户 ${user.userId}` });
});

// --- 角色权限控制 ---
function withRole(roles: string[], handler: AuthHandler) {
    return withAuth(async (request, context) => {
        if (!roles.includes(context.user.role)) {
            return Response.json({ error: '权限不足' }, { status: 403 });
        }
        return handler(request, context);
    });
}

const adminOnly = withRole(['admin'], async (request, { user }) => {
    return Response.json({ adminData: '机密内容' });
});


// ============================================================
//               7. 最佳实践
// ============================================================

/**
 * 【Route Handler 最佳实践】
 *
 * ✅ 推荐做法：
 * - 使用 NextRequest/NextResponse 获取扩展功能
 * - 为外部访问的 API 正确配置 CORS
 * - 统一错误响应格式 { error: string, code?: string }
 * - 封装认证逻辑为可复用中间件函数
 * - 对输入参数做验证（Zod / 手动校验）
 * - 合理使用缓存，静态数据默认缓存，动态数据 force-dynamic
 * - 为 webhook 端点验证签名防止伪造
 *
 * ❌ 避免做法：
 * - 避免处理可用 Server Actions 替代的表单提交
 * - 避免将 route.ts 和 page.tsx 放在同一目录
 * - 避免在 Edge Runtime 中使用 Node.js 原生模块
 * - 避免不处理错误导致 500 泄漏堆栈信息
 * - 避免在 GET handler 中修改数据（违反 HTTP 语义）
 * - 避免生产环境 CORS origin 硬编码为 '*'
 *
 * 【路由组织结构推荐】
 *
 *   app/api/
 *   ├── auth/
 *   │   ├── login/route.ts          # POST 登录
 *   │   └── [...nextauth]/route.ts  # NextAuth
 *   ├── products/
 *   │   ├── route.ts                # GET 列表, POST 创建
 *   │   └── [id]/route.ts           # GET/PUT/DELETE
 *   ├── upload/route.ts             # POST 文件上传
 *   └── webhook/stripe/route.ts     # POST 回调
 *
 * 【Route Handler vs Server Action vs Middleware】
 *
 *   特性             │ Route Handler │ Server Action │ Middleware
 *   ────────────────────────────────────────────────────────
 *   位置             │ route.ts      │ 'use server'  │ middleware.ts
 *   HTTP 方法        │ 任意          │ 仅 POST       │ 任意（拦截）
 *   外部客户端调用    │ 是            │ 否            │ 否
 *   返回值           │ Response      │ 可序列化值     │ NextResponse
 *   典型用途         │ RESTful API   │ 数据变更       │ 重定向/认证
 */
```
