/**
 * ============================================================
 *                    Next.js 中间件
 * ============================================================
 * 中间件 (Middleware) 是 Next.js 提供的请求拦截机制。
 * 它在请求到达页面或 API 路由之前执行，可用于
 * 认证、重定向、请求改写、国际化等场景。
 *
 * 适用版本：Next.js 14 / 15（App Router）
 * 文件位置：middleware.ts 放在项目根目录（与 app/ 同级）
 * ============================================================
 */

import { NextRequest, NextResponse } from 'next/server';
import { headers } from 'next/headers';

// ============================================================
//                    1. 中间件基础
// ============================================================

/**
 * 【什么是中间件】
 * - 中间件运行在 Edge Runtime（非 Node.js 运行时）
 * - 每个请求到达服务器后，先经过中间件处理
 * - 可以修改请求/响应的 headers、cookies
 * - 可以重定向 (redirect) 或重写 (rewrite) 请求
 *
 * 【文件位置】
 * ├── app/
 * ├── middleware.ts    ← 必须放在项目根目录
 * └── next.config.js
 *
 * 【执行顺序】
 * 请求 → middleware → 路由匹配 → 页面/API 渲染
 */

// --- 最简单的中间件 ---
export function middleware(request: NextRequest) {
    console.log('请求路径:', request.nextUrl.pathname);
    console.log('请求方法:', request.method);

    // NextResponse.next() 表示继续处理请求，不做拦截
    return NextResponse.next();
}

// --- NextRequest 常用属性 ---
function showNextRequestAPI(request: NextRequest) {
    const url = request.nextUrl;           // NextURL 对象
    const pathname = url.pathname;         // /dashboard/settings
    const searchParams = url.searchParams; // 查询参数
    const geo = request.geo;              // 地理位置（Vercel 可用）
    const ip = request.ip;               // 客户端 IP
    const cookieStore = request.cookies;  // Cookie 存储
}

// --- NextResponse 常用方法 ---
function showNextResponseAPI() {
    const next = NextResponse.next();                                    // 放行
    const redirect = NextResponse.redirect(new URL('/login', 'http://localhost:3000'));  // 重定向
    const rewrite = NextResponse.rewrite(new URL('/api/proxy', 'http://localhost:3000')); // 重写
    const json = NextResponse.json({ error: '未授权' }, { status: 401 });  // JSON 响应
}


// ============================================================
//                    2. 路由匹配
// ============================================================

/**
 * 【config.matcher 配置】
 * - 通过 matcher 指定中间件要拦截的路由
 * - 不配置 matcher 时，中间件会拦截所有请求
 * - 支持字符串、正则、数组等多种匹配模式
 */

// 方式1：单个路径匹配
export const config_single = {
    matcher: '/dashboard/:path*',
};

// 方式2：多个路径匹配
export const config_multiple = {
    matcher: ['/dashboard/:path*', '/admin/:path*', '/api/protected/:path*'],
};

// 方式3：排除特定路径（正则）
export const config_exclude = {
    matcher: ['/((?!api|_next/static|_next/image|favicon.ico|robots.txt).*)'],
};

// 方式4：在中间件内部条件判断
function middlewareWithConditional(request: NextRequest) {
    const pathname = request.nextUrl.pathname;

    // 跳过静态资源
    if (pathname.startsWith('/_next') || pathname.includes('.')) {
        return NextResponse.next();
    }

    if (pathname.startsWith('/dashboard')) return checkAuth(request);
    if (pathname.startsWith('/api/'))      return rateLimit(request);

    return NextResponse.next();
}

function checkAuth(req: NextRequest) { return NextResponse.next(); }
function rateLimit(req: NextRequest) { return NextResponse.next(); }


// ============================================================
//                    3. 重定向与重写
// ============================================================

/**
 * 【重定向 vs 重写】
 *
 * redirect：浏览器地址栏 URL 改变，返回 3xx
 *   → 适用于：登录跳转、旧 URL 迁移
 *
 * rewrite：浏览器地址栏 URL 不变，服务器内部转发
 *   → 适用于：A/B 测试、代理转发、多租户
 */

function redirectMiddleware(request: NextRequest) {
    const { pathname } = request.nextUrl;

    // 旧路径永久重定向
    if (pathname === '/old-blog') {
        return NextResponse.redirect(new URL('/blog', request.url), 301);
    }

    // 未登录用户重定向到登录页
    const token = request.cookies.get('session-token');
    if (!token && pathname.startsWith('/dashboard')) {
        const loginUrl = new URL('/login', request.url);
        loginUrl.searchParams.set('callbackUrl', pathname);
        return NextResponse.redirect(loginUrl);
    }

    return NextResponse.next();
}

function rewriteMiddleware(request: NextRequest) {
    const { pathname } = request.nextUrl;

    // A/B 测试：50% 的用户看到新版页面
    if (pathname === '/landing') {
        const bucket = Math.random() < 0.5 ? 'a' : 'b';
        return NextResponse.rewrite(new URL(`/landing/${bucket}`, request.url));
        // 用户地址栏仍然是 /landing，但实际渲染 /landing/a 或 /landing/b
    }

    // 多租户：根据子域名路由
    const hostname = request.headers.get('host') || '';
    const subdomain = hostname.split('.')[0];
    if (subdomain !== 'www' && subdomain !== 'localhost') {
        return NextResponse.rewrite(
            new URL(`/tenants/${subdomain}${pathname}`, request.url)
        );
    }

    return NextResponse.next();
}


// ============================================================
//                    4. 请求头操作
// ============================================================

/**
 * 【Headers 和 Cookies 操作】
 * - 中间件可以读取和修改请求/响应的 headers
 * - 可以设置、删除、读取 cookies
 * - 常用于传递上下文信息给 Server Components
 */

function headersMiddleware(request: NextRequest) {
    const requestHeaders = new Headers(request.headers);
    requestHeaders.set('x-pathname', request.nextUrl.pathname);
    requestHeaders.set('x-request-id', crypto.randomUUID());

    const response = NextResponse.next({
        request: { headers: requestHeaders },
    });

    // 安全相关响应头
    response.headers.set('X-Frame-Options', 'DENY');
    response.headers.set('X-Content-Type-Options', 'nosniff');
    return response;
}

function cookiesMiddleware(request: NextRequest) {
    const theme = request.cookies.get('theme');       // { name, value }
    const hasToken = request.cookies.has('token');     // 是否存在

    const response = NextResponse.next();

    // 设置 cookies（附带安全选项）
    response.cookies.set('visited', 'true', {
        httpOnly: true,        // 仅服务器端可访问
        secure: true,          // 仅 HTTPS
        sameSite: 'lax',       // CSRF 保护
        maxAge: 60 * 60 * 24,  // 1 天
        path: '/',
    });

    response.cookies.delete('old-cookie');  // 删除
    return response;
}

// 在 Server Component 中读取中间件设置的 headers
async function DashboardPage() {
    const headersList = await headers();
    const pathname = headersList.get('x-pathname');
    return <div>当前路径: {pathname}</div>;
}


// ============================================================
//                    5. 认证检查
// ============================================================

/**
 * 【中间件认证模式】
 * - 中间件是实现路由保护的理想位置
 * - 运行在 Edge Runtime，响应速度快
 *
 * 【注意事项】
 * - Edge Runtime 不支持 Node.js 原生模块
 * - 不能使用 jsonwebtoken 库，需用 jose（兼容 Edge）
 */

import { jwtVerify } from 'jose';

const JWT_SECRET = new TextEncoder().encode(process.env.JWT_SECRET || 'secret');
const protectedRoutes = ['/dashboard', '/profile', '/settings'];
const authRoutes = ['/login', '/register'];

async function authMiddleware(request: NextRequest) {
    const { pathname } = request.nextUrl;
    const token = request.cookies.get('auth-token')?.value;

    const isProtected = protectedRoutes.some(r => pathname.startsWith(r));
    const isAuthPage = authRoutes.some(r => pathname.startsWith(r));

    // 验证 JWT
    let isValid = false;
    let payload: any = null;
    if (token) {
        try {
            const verified = await jwtVerify(token, JWT_SECRET, { algorithms: ['HS256'] });
            payload = verified.payload;
            isValid = true;
        } catch { isValid = false; }
    }

    // 未登录 → 重定向登录页
    if (isProtected && !isValid) {
        const url = new URL('/login', request.url);
        url.searchParams.set('from', pathname);
        return NextResponse.redirect(url);
    }

    // 已登录 → 跳过登录/注册页
    if (isAuthPage && isValid) {
        return NextResponse.redirect(new URL('/dashboard', request.url));
    }

    // 将用户信息传递给后续路由
    if (isValid && payload) {
        const headers = new Headers(request.headers);
        headers.set('x-user-id', payload.userId as string);
        headers.set('x-user-role', payload.role as string);
        return NextResponse.next({ request: { headers } });
    }

    return NextResponse.next();
}

// --- 基于角色的访问控制 (RBAC) ---
async function rbacMiddleware(request: NextRequest) {
    const { pathname } = request.nextUrl;
    const userRole = request.headers.get('x-user-role');

    const roleRoutes: Record<string, string[]> = {
        '/api/users':   ['admin'],
        '/api/reports': ['admin', 'manager'],
        '/api/content': ['admin', 'manager', 'editor'],
    };

    for (const [route, roles] of Object.entries(roleRoutes)) {
        if (pathname.startsWith(route) && (!userRole || !roles.includes(userRole))) {
            return NextResponse.json({ error: '权限不足' }, { status: 403 });
        }
    }
    return NextResponse.next();
}


// ============================================================
//                    6. 国际化路由
// ============================================================

/**
 * 【中间件实现 i18n】
 * - 检测用户首选语言（Accept-Language 头）
 * - 根据 cookie 或路径前缀判断当前语言
 * - 自动重定向到正确的语言路径
 *
 * 【URL 结构】
 * /zh/about → 中文    /en/about → 英文    /about → 自动检测
 */

const locales = ['zh', 'en', 'ja', 'ko'];
const defaultLocale = 'zh';

function getPreferredLocale(request: NextRequest): string {
    const acceptLang = request.headers.get('Accept-Language') || '';
    // 解析 zh-CN,zh;q=0.9,en;q=0.8 格式
    const preferred = acceptLang.split(',').map(lang => {
        const [code, q] = lang.trim().split(';q=');
        return { code: code.split('-')[0].toLowerCase(), quality: q ? parseFloat(q) : 1.0 };
    }).sort((a, b) => b.quality - a.quality);

    for (const { code } of preferred) {
        if (locales.includes(code)) return code;
    }
    return defaultLocale;
}

function i18nMiddleware(request: NextRequest) {
    const { pathname } = request.nextUrl;

    // 已包含语言前缀 → 放行
    const hasLocale = locales.some(
        l => pathname.startsWith(`/${l}/`) || pathname === `/${l}`
    );
    if (hasLocale) return NextResponse.next();

    // 优先 cookie，其次 Accept-Language
    const cookieLocale = request.cookies.get('NEXT_LOCALE')?.value;
    const locale = (cookieLocale && locales.includes(cookieLocale))
        ? cookieLocale
        : getPreferredLocale(request);

    // /about → /zh/about
    const newUrl = new URL(`/${locale}${pathname}`, request.url);
    newUrl.search = request.nextUrl.search;

    const response = NextResponse.redirect(newUrl);
    response.cookies.set('NEXT_LOCALE', locale, { maxAge: 60 * 60 * 24 * 365 });
    return response;
}

// --- 语言切换处理 ---
// 点击语言切换按钮 → /set-locale?locale=en&redirect=/about
function handleLocaleSwitch(request: NextRequest) {
    const { pathname, searchParams } = request.nextUrl;

    if (pathname === '/set-locale') {
        const newLocale = searchParams.get('locale');
        const redirect = searchParams.get('redirect') || '/';

        if (newLocale && locales.includes(newLocale)) {
            const response = NextResponse.redirect(
                new URL(`/${newLocale}${redirect}`, request.url)
            );
            response.cookies.set('NEXT_LOCALE', newLocale, { maxAge: 60 * 60 * 24 * 365 });
            return response;
        }
    }
    return i18nMiddleware(request);
}


// ============================================================
//                    7. 最佳实践
// ============================================================

/**
 * 【中间件最佳实践】
 *
 * ✅ 推荐做法：
 * 1. 使用 config.matcher 精确限制中间件作用范围
 * 2. 保持中间件轻量 — Edge Runtime 有执行时间限制
 * 3. 使用 jose 库替代 jsonwebtoken（兼容 Edge Runtime）
 * 4. 通过 headers 传递上下文信息给 Server Components
 * 5. 合理使用 redirect 和 rewrite 的场景区分
 * 6. 为安全 cookies 设置 httpOnly、secure、sameSite
 * 7. 中间件只做「门卫」工作：认证、重定向、头信息
 *
 * ❌ 避免做法：
 * 1. 在中间件中进行复杂的数据库查询 → 使用 API 路由
 * 2. 不配置 matcher 导致静态资源也被拦截 → 性能浪费
 * 3. 在中间件中使用 Node.js 专有模块 → Edge Runtime 不支持
 * 4. 忽略 Token 过期处理 → 用户体验差
 * 5. 在中间件中渲染 UI → 中间件只负责请求层逻辑
 */

// 推荐的 config.matcher（排除静态资源）
export const config = {
    matcher: [
        '/((?!api|_next/static|_next/image|favicon.ico|sitemap.xml|robots.txt|.*\\.(?:svg|png|jpg|jpeg|gif|webp|ico)$).*)',
    ],
};
