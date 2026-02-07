# http server.js

::: info 文件信息
- 📄 原文件：`02_http_server.js`
- 🔤 语言：javascript
:::

============================================================
               Node.js HTTP 服务器
============================================================
本文件介绍 Node.js 中的 HTTP 服务器开发。
============================================================

## 完整代码

```javascript
/**
 * ============================================================
 *                Node.js HTTP 服务器
 * ============================================================
 * 本文件介绍 Node.js 中的 HTTP 服务器开发。
 * ============================================================
 */

const http = require("http");
const https = require("https");
const url = require("url");
const querystring = require("querystring");
const fs = require("fs");
const path = require("path");

// ============================================================
//                    1. 基础 HTTP 服务器
// ============================================================

console.log("=".repeat(60));
console.log("Node.js HTTP 服务器示例");
console.log("=".repeat(60));

/**
 * 创建基础服务器
 */
function createBasicServer() {
    const server = http.createServer((req, res) => {
        // 设置响应头
        res.setHeader("Content-Type", "text/plain; charset=utf-8");

        // 发送响应
        res.statusCode = 200;
        res.end("Hello, World!\n你好，世界！");
    });

    return server;
}

// ============================================================
//                    2. 路由处理
// ============================================================

/**
 * 简单的路由器
 */
class Router {
    constructor() {
        this.routes = {
            GET: {},
            POST: {},
            PUT: {},
            DELETE: {}
        };
    }

    // 注册路由
    get(path, handler) {
        this.routes.GET[path] = handler;
    }

    post(path, handler) {
        this.routes.POST[path] = handler;
    }

    put(path, handler) {
        this.routes.PUT[path] = handler;
    }

    delete(path, handler) {
        this.routes.DELETE[path] = handler;
    }

    // 处理请求
    async handle(req, res) {
        const parsedUrl = url.parse(req.url, true);
        const pathname = parsedUrl.pathname;
        const method = req.method;

        // 添加解析后的信息到请求对象
        req.query = parsedUrl.query;
        req.pathname = pathname;

        // 查找路由
        const handler = this.routes[method]?.[pathname];

        if (handler) {
            try {
                await handler(req, res);
            } catch (error) {
                console.error("Handler error:", error);
                res.statusCode = 500;
                res.setHeader("Content-Type", "application/json");
                res.end(JSON.stringify({ error: "Internal Server Error" }));
            }
        } else {
            res.statusCode = 404;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ error: "Not Found" }));
        }
    }
}

// ============================================================
//                    3. 请求体解析
// ============================================================

/**
 * 解析 JSON 请求体
 */
function parseJsonBody(req) {
    return new Promise((resolve, reject) => {
        let body = "";

        req.on("data", chunk => {
            body += chunk.toString();

            // 防止请求体过大
            if (body.length > 1e6) {
                req.destroy();
                reject(new Error("Request body too large"));
            }
        });

        req.on("end", () => {
            try {
                const data = body ? JSON.parse(body) : {};
                resolve(data);
            } catch (error) {
                reject(new Error("Invalid JSON"));
            }
        });

        req.on("error", reject);
    });
}

/**
 * 解析 URL 编码的请求体
 */
function parseUrlEncodedBody(req) {
    return new Promise((resolve, reject) => {
        let body = "";

        req.on("data", chunk => {
            body += chunk.toString();
        });

        req.on("end", () => {
            const data = querystring.parse(body);
            resolve(data);
        });

        req.on("error", reject);
    });
}

// ============================================================
//                    4. 中间件模式
// ============================================================

/**
 * 中间件管理器
 */
class MiddlewareManager {
    constructor() {
        this.middlewares = [];
    }

    use(middleware) {
        this.middlewares.push(middleware);
    }

    async execute(req, res) {
        let index = 0;

        const next = async () => {
            if (index < this.middlewares.length) {
                const middleware = this.middlewares[index++];
                await middleware(req, res, next);
            }
        };

        await next();
    }
}

// --- 常用中间件 ---

/**
 * 日志中间件
 */
function loggerMiddleware(req, res, next) {
    const start = Date.now();
    const { method, url } = req;

    res.on("finish", () => {
        const duration = Date.now() - start;
        console.log(`${method} ${url} ${res.statusCode} - ${duration}ms`);
    });

    next();
}

/**
 * CORS 中间件
 */
function corsMiddleware(req, res, next) {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

    if (req.method === "OPTIONS") {
        res.statusCode = 204;
        res.end();
        return;
    }

    next();
}

/**
 * JSON 解析中间件
 */
async function jsonParserMiddleware(req, res, next) {
    if (req.headers["content-type"]?.includes("application/json")) {
        try {
            req.body = await parseJsonBody(req);
        } catch (error) {
            res.statusCode = 400;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ error: error.message }));
            return;
        }
    }
    next();
}

// ============================================================
//                    5. RESTful API 示例
// ============================================================

/**
 * 创建 RESTful API 服务器
 */
function createRestApiServer() {
    // 模拟数据库
    const todos = new Map();
    let nextId = 1;

    const router = new Router();
    const middleware = new MiddlewareManager();

    // 注册中间件
    middleware.use(loggerMiddleware);
    middleware.use(corsMiddleware);
    middleware.use(jsonParserMiddleware);

    // --- API 路由 ---

    // 获取所有 Todo
    router.get("/api/todos", (req, res) => {
        const list = Array.from(todos.values());
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify(list));
    });

    // 获取单个 Todo
    router.get("/api/todo", (req, res) => {
        const id = parseInt(req.query.id);
        const todo = todos.get(id);

        res.setHeader("Content-Type", "application/json");

        if (todo) {
            res.end(JSON.stringify(todo));
        } else {
            res.statusCode = 404;
            res.end(JSON.stringify({ error: "Todo not found" }));
        }
    });

    // 创建 Todo
    router.post("/api/todos", (req, res) => {
        const { title, completed = false } = req.body;

        if (!title) {
            res.statusCode = 400;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ error: "Title is required" }));
            return;
        }

        const todo = {
            id: nextId++,
            title,
            completed,
            createdAt: new Date().toISOString()
        };

        todos.set(todo.id, todo);

        res.statusCode = 201;
        res.setHeader("Content-Type", "application/json");
        res.end(JSON.stringify(todo));
    });

    // 更新 Todo
    router.put("/api/todo", (req, res) => {
        const id = parseInt(req.query.id);
        const todo = todos.get(id);

        res.setHeader("Content-Type", "application/json");

        if (!todo) {
            res.statusCode = 404;
            res.end(JSON.stringify({ error: "Todo not found" }));
            return;
        }

        const { title, completed } = req.body;
        if (title !== undefined) todo.title = title;
        if (completed !== undefined) todo.completed = completed;
        todo.updatedAt = new Date().toISOString();

        res.end(JSON.stringify(todo));
    });

    // 删除 Todo
    router.delete("/api/todo", (req, res) => {
        const id = parseInt(req.query.id);
        const deleted = todos.delete(id);

        res.setHeader("Content-Type", "application/json");

        if (deleted) {
            res.end(JSON.stringify({ success: true }));
        } else {
            res.statusCode = 404;
            res.end(JSON.stringify({ error: "Todo not found" }));
        }
    });

    // 最后注册路由处理
    middleware.use((req, res) => router.handle(req, res));

    // 创建服务器
    const server = http.createServer((req, res) => {
        middleware.execute(req, res);
    });

    return server;
}

// ============================================================
//                    6. 静态文件服务
// ============================================================

/**
 * 静态文件服务器
 */
function createStaticServer(staticDir) {
    // MIME 类型映射
    const mimeTypes = {
        ".html": "text/html",
        ".css": "text/css",
        ".js": "text/javascript",
        ".json": "application/json",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
        ".txt": "text/plain",
        ".pdf": "application/pdf"
    };

    const server = http.createServer((req, res) => {
        // 解析 URL
        const parsedUrl = url.parse(req.url);
        let pathname = parsedUrl.pathname;

        // 默认页面
        if (pathname === "/") {
            pathname = "/index.html";
        }

        // 防止目录遍历攻击
        const safePath = path.normalize(pathname).replace(/^(\.\.[\/\\])+/, "");
        const filePath = path.join(staticDir, safePath);

        // 检查文件是否在静态目录内
        if (!filePath.startsWith(staticDir)) {
            res.statusCode = 403;
            res.end("Forbidden");
            return;
        }

        // 读取文件
        fs.readFile(filePath, (err, data) => {
            if (err) {
                if (err.code === "ENOENT") {
                    res.statusCode = 404;
                    res.end("Not Found");
                } else {
                    res.statusCode = 500;
                    res.end("Internal Server Error");
                }
                return;
            }

            // 设置 MIME 类型
            const ext = path.extname(filePath).toLowerCase();
            const mimeType = mimeTypes[ext] || "application/octet-stream";

            res.setHeader("Content-Type", mimeType);
            res.setHeader("Content-Length", data.length);
            res.end(data);
        });
    });

    return server;
}

// ============================================================
//                    7. 流式响应
// ============================================================

/**
 * 演示流式响应
 */
function createStreamServer() {
    const server = http.createServer((req, res) => {
        if (req.url === "/stream") {
            // 流式发送数据
            res.setHeader("Content-Type", "text/plain");
            res.setHeader("Transfer-Encoding", "chunked");

            let count = 0;
            const interval = setInterval(() => {
                res.write(`数据块 ${++count}\n`);

                if (count >= 5) {
                    clearInterval(interval);
                    res.end("流结束\n");
                }
            }, 500);

            // 客户端断开连接时清理
            req.on("close", () => {
                clearInterval(interval);
            });
        } else if (req.url === "/file") {
            // 流式发送文件
            const filePath = __filename;
            const stat = fs.statSync(filePath);

            res.setHeader("Content-Type", "text/plain");
            res.setHeader("Content-Length", stat.size);

            const readStream = fs.createReadStream(filePath);
            readStream.pipe(res);
        } else {
            res.setHeader("Content-Type", "text/html; charset=utf-8");
            res.end(`
                <h1>流式响应示例</h1>
                <ul>
                    <li><a href="/stream">分块数据流</a></li>
                    <li><a href="/file">文件流</a></li>
                </ul>
            `);
        }
    });

    return server;
}

// ============================================================
//                    8. HTTP 客户端
// ============================================================

/**
 * 简单的 HTTP GET 请求
 */
function httpGet(urlString) {
    return new Promise((resolve, reject) => {
        const parsedUrl = new URL(urlString);
        const protocol = parsedUrl.protocol === "https:" ? https : http;

        protocol.get(urlString, (res) => {
            let data = "";

            res.on("data", chunk => {
                data += chunk;
            });

            res.on("end", () => {
                resolve({
                    statusCode: res.statusCode,
                    headers: res.headers,
                    body: data
                });
            });
        }).on("error", reject);
    });
}

/**
 * HTTP POST 请求
 */
function httpPost(urlString, body, headers = {}) {
    return new Promise((resolve, reject) => {
        const parsedUrl = new URL(urlString);
        const protocol = parsedUrl.protocol === "https:" ? https : http;

        const options = {
            method: "POST",
            hostname: parsedUrl.hostname,
            port: parsedUrl.port,
            path: parsedUrl.pathname + parsedUrl.search,
            headers: {
                "Content-Type": "application/json",
                "Content-Length": Buffer.byteLength(body),
                ...headers
            }
        };

        const req = protocol.request(options, (res) => {
            let data = "";

            res.on("data", chunk => {
                data += chunk;
            });

            res.on("end", () => {
                resolve({
                    statusCode: res.statusCode,
                    headers: res.headers,
                    body: data
                });
            });
        });

        req.on("error", reject);
        req.write(body);
        req.end();
    });
}

// ============================================================
//                    主程序
// ============================================================

async function main() {
    const PORT = process.env.PORT || 3000;

    // 选择要启动的服务器
    const serverType = process.argv[2] || "rest";

    let server;

    switch (serverType) {
        case "basic":
            server = createBasicServer();
            console.log("启动基础服务器...");
            break;

        case "rest":
            server = createRestApiServer();
            console.log("启动 REST API 服务器...");
            console.log(`
API 端点：
  GET    /api/todos     - 获取所有 Todo
  GET    /api/todo?id=1 - 获取单个 Todo
  POST   /api/todos     - 创建 Todo
  PUT    /api/todo?id=1 - 更新 Todo
  DELETE /api/todo?id=1 - 删除 Todo
`);
            break;

        case "static":
            server = createStaticServer(__dirname);
            console.log("启动静态文件服务器...");
            break;

        case "stream":
            server = createStreamServer();
            console.log("启动流式响应服务器...");
            break;

        default:
            console.log("未知服务器类型。可用: basic, rest, static, stream");
            process.exit(1);
    }

    // 启动服务器
    server.listen(PORT, () => {
        console.log(`服务器运行在 http://localhost:${PORT}`);
        console.log("按 Ctrl+C 停止服务器");
    });

    // 优雅关闭
    process.on("SIGINT", () => {
        console.log("\n正在关闭服务器...");
        server.close(() => {
            console.log("服务器已关闭");
            process.exit(0);
        });
    });
}

// 只在直接运行时启动服务器
if (require.main === module) {
    main();
}

// 导出模块
module.exports = {
    Router,
    MiddlewareManager,
    createBasicServer,
    createRestApiServer,
    createStaticServer,
    createStreamServer,
    httpGet,
    httpPost
};
```
