# HttpServer

::: info 文件信息
- 📄 原文件：`HttpServer.java`
- 🔤 语言：java
:::

简单 HTTP 服务器
使用 Java 内置的 HttpServer 实现简单的 REST API。
功能：
- 基本路由
- JSON 响应
- 请求日志
- 错误处理
运行后访问：
- http://localhost:8080/         首页
- http://localhost:8080/api/time 当前时间
- http://localhost:8080/api/echo 回显请求
- http://localhost:8080/api/users 用户列表

## 完整代码

```java
import com.sun.net.httpserver.*;
import java.io.*;
import java.net.*;
import java.nio.charset.*;
import java.util.*;
import java.util.concurrent.*;
import java.time.*;
import java.time.format.*;

/**
 * ============================================================
 *                    简单 HTTP 服务器
 * ============================================================
 * 使用 Java 内置的 HttpServer 实现简单的 REST API。
 *
 * 功能：
 * - 基本路由
 * - JSON 响应
 * - 请求日志
 * - 错误处理
 *
 * 运行后访问：
 * - http://localhost:8080/         首页
 * - http://localhost:8080/api/time 当前时间
 * - http://localhost:8080/api/echo 回显请求
 * - http://localhost:8080/api/users 用户列表
 * ============================================================
 */
public class HttpServer {

    private static final int PORT = 8080;
    private static final DateTimeFormatter FORMATTER =
        DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    // 模拟数据库
    private static final List<User> users = new CopyOnWriteArrayList<>(List.of(
        new User(1, "Alice", "alice@example.com"),
        new User(2, "Bob", "bob@example.com"),
        new User(3, "Charlie", "charlie@example.com")
    ));

    public static void main(String[] args) throws IOException {
        com.sun.net.httpserver.HttpServer server =
            com.sun.net.httpserver.HttpServer.create(new InetSocketAddress(PORT), 0);

        // 配置路由
        server.createContext("/", HttpServer::handleHome);
        server.createContext("/api/time", HttpServer::handleTime);
        server.createContext("/api/echo", HttpServer::handleEcho);
        server.createContext("/api/users", HttpServer::handleUsers);

        // 使用虚拟线程（Java 21+）
        server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());

        server.start();
        System.out.println("服务器启动在 http://localhost:" + PORT);
        System.out.println("\n可用端点:");
        System.out.println("  GET  /           - 首页");
        System.out.println("  GET  /api/time   - 当前时间");
        System.out.println("  POST /api/echo   - 回显请求体");
        System.out.println("  GET  /api/users  - 获取用户列表");
        System.out.println("  POST /api/users  - 创建用户");
        System.out.println("\n按 Ctrl+C 停止服务器");
    }

    /**
     * 首页处理
     */
    private static void handleHome(HttpExchange exchange) throws IOException {
        logRequest(exchange);

        String html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Java HTTP Server</title>
                <style>
                    body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                    h1 { color: #333; }
                    .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
                    code { background: #e0e0e0; padding: 2px 6px; border-radius: 3px; }
                </style>
            </head>
            <body>
                <h1>Java HTTP Server</h1>
                <p>欢迎使用 Java 内置 HTTP 服务器！</p>

                <h2>可用端点</h2>
                <div class="endpoint">
                    <code>GET /api/time</code> - 获取当前时间
                </div>
                <div class="endpoint">
                    <code>POST /api/echo</code> - 回显请求体
                </div>
                <div class="endpoint">
                    <code>GET /api/users</code> - 获取用户列表
                </div>
                <div class="endpoint">
                    <code>POST /api/users</code> - 创建用户
                </div>

                <h2>示例</h2>
                <pre>
curl http://localhost:8080/api/time
curl -X POST -d "Hello" http://localhost:8080/api/echo
curl http://localhost:8080/api/users
                </pre>
            </body>
            </html>
            """;

        sendResponse(exchange, 200, "text/html", html);
    }

    /**
     * 时间 API
     */
    private static void handleTime(HttpExchange exchange) throws IOException {
        logRequest(exchange);

        if (!exchange.getRequestMethod().equals("GET")) {
            sendError(exchange, 405, "Method Not Allowed");
            return;
        }

        String json = """
            {
                "timestamp": "%s",
                "unix": %d,
                "timezone": "%s"
            }
            """.formatted(
                LocalDateTime.now().format(FORMATTER),
                System.currentTimeMillis() / 1000,
                ZoneId.systemDefault().getId()
            );

        sendResponse(exchange, 200, "application/json", json);
    }

    /**
     * Echo API
     */
    private static void handleEcho(HttpExchange exchange) throws IOException {
        logRequest(exchange);

        if (!exchange.getRequestMethod().equals("POST")) {
            sendError(exchange, 405, "Method Not Allowed");
            return;
        }

        String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
        String contentType = exchange.getRequestHeaders().getFirst("Content-Type");

        String json = """
            {
                "method": "%s",
                "path": "%s",
                "contentType": "%s",
                "contentLength": %d,
                "body": "%s",
                "headers": %s
            }
            """.formatted(
                exchange.getRequestMethod(),
                exchange.getRequestURI().getPath(),
                contentType != null ? contentType : "none",
                body.length(),
                escapeJson(body),
                headersToJson(exchange.getRequestHeaders())
            );

        sendResponse(exchange, 200, "application/json", json);
    }

    /**
     * 用户 API
     */
    private static void handleUsers(HttpExchange exchange) throws IOException {
        logRequest(exchange);

        String method = exchange.getRequestMethod();

        switch (method) {
            case "GET" -> handleGetUsers(exchange);
            case "POST" -> handleCreateUser(exchange);
            default -> sendError(exchange, 405, "Method Not Allowed");
        }
    }

    private static void handleGetUsers(HttpExchange exchange) throws IOException {
        // 解析查询参数
        String query = exchange.getRequestURI().getQuery();
        Map<String, String> params = parseQuery(query);

        List<User> result = users;

        // 简单过滤
        if (params.containsKey("name")) {
            String name = params.get("name");
            result = users.stream()
                .filter(u -> u.name().toLowerCase().contains(name.toLowerCase()))
                .toList();
        }

        StringBuilder json = new StringBuilder("[\n");
        for (int i = 0; i < result.size(); i++) {
            User u = result.get(i);
            json.append("""
                    {
                        "id": %d,
                        "name": "%s",
                        "email": "%s"
                    }""".formatted(u.id(), u.name(), u.email()));
            if (i < result.size() - 1) json.append(",");
            json.append("\n");
        }
        json.append("]");

        sendResponse(exchange, 200, "application/json", json.toString());
    }

    private static void handleCreateUser(HttpExchange exchange) throws IOException {
        String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);

        // 简单解析（实际应使用 JSON 库）
        String name = extractJsonValue(body, "name");
        String email = extractJsonValue(body, "email");

        if (name == null || email == null) {
            sendError(exchange, 400, "Missing name or email");
            return;
        }

        int newId = users.stream().mapToInt(User::id).max().orElse(0) + 1;
        User newUser = new User(newId, name, email);
        users.add(newUser);

        String json = """
            {
                "id": %d,
                "name": "%s",
                "email": "%s",
                "message": "User created successfully"
            }
            """.formatted(newUser.id(), newUser.name(), newUser.email());

        sendResponse(exchange, 201, "application/json", json);
    }

    // ============================================================
    //                    辅助方法
    // ============================================================

    private static void sendResponse(HttpExchange exchange, int code, String contentType, String body)
            throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);

        exchange.getResponseHeaders().set("Content-Type", contentType + "; charset=utf-8");
        exchange.sendResponseHeaders(code, bytes.length);

        try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static void sendError(HttpExchange exchange, int code, String message) throws IOException {
        String json = """
            {
                "error": true,
                "code": %d,
                "message": "%s"
            }
            """.formatted(code, message);

        sendResponse(exchange, code, "application/json", json);
    }

    private static void logRequest(HttpExchange exchange) {
        String time = LocalDateTime.now().format(FORMATTER);
        String method = exchange.getRequestMethod();
        String path = exchange.getRequestURI().getPath();
        String remote = exchange.getRemoteAddress().toString();

        System.out.printf("[%s] %s %s from %s%n", time, method, path, remote);
    }

    private static Map<String, String> parseQuery(String query) {
        Map<String, String> params = new HashMap<>();
        if (query == null || query.isEmpty()) return params;

        for (String pair : query.split("&")) {
            String[] kv = pair.split("=", 2);
            if (kv.length == 2) {
                params.put(
                    URLDecoder.decode(kv[0], StandardCharsets.UTF_8),
                    URLDecoder.decode(kv[1], StandardCharsets.UTF_8)
                );
            }
        }
        return params;
    }

    private static String escapeJson(String s) {
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    private static String headersToJson(Headers headers) {
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, List<String>> entry : headers.entrySet()) {
            if (!first) sb.append(", ");
            sb.append("\"").append(entry.getKey()).append("\": \"")
              .append(String.join(", ", entry.getValue())).append("\"");
            first = false;
        }
        sb.append("}");
        return sb.toString();
    }

    private static String extractJsonValue(String json, String key) {
        // 简单的 JSON 值提取（实际应使用 JSON 库）
        String pattern = "\"" + key + "\"\\s*:\\s*\"([^\"]*)\"";
        java.util.regex.Matcher matcher = java.util.regex.Pattern.compile(pattern).matcher(json);
        return matcher.find() ? matcher.group(1) : null;
    }
}

record User(int id, String name, String email) {}
```
