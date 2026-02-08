# API 网关架构设计

## 目录
- [API网关概述](#api网关概述)
- [核心功能](#核心功能)
- [架构模式](#架构模式)
- [Kong网关实战](#kong网关实战)
- [性能优化](#性能优化)
- [高可用设计](#高可用设计)
- [实战案例](#实战案例)

---

## API网关概述

### 什么是API网关

```
传统架构                              API网关架构
┌────────────┐                       ┌────────────┐
│   客户端   │                       │   客户端   │
└──┬──┬──┬──┘                       └──────┬─────┘
   │  │  │                                 │
   │  │  │                                 ▼
   │  │  │                          ┌────────────┐
   │  │  │                          │ API Gateway│
   │  │  │                          └──────┬─────┘
   │  │  │                                 │
   ▼  ▼  ▼                          ┌──────┴─────┐
┌──────────┐                        │            │
│ Service A│                        ▼            ▼
│ Service B│                   ┌─────────┐  ┌─────────┐
│ Service C│                   │Service A│  │Service B│
└──────────┘                   └─────────┘  └─────────┘
                                     ▼
                                ┌─────────┐
                                │Service C│
                                └─────────┘

问题:                               优势:
- 多个端点管理                      - 统一入口
- 重复的认证授权                    - 集中认证
- 跨域处理分散                      - 流量控制
- 监控困难                          - 统一监控
```

### API网关核心职责

```
┌──────────────────────────────────────────────────────┐
│              API 网关核心功能模块                    │
├──────────────────────────────────────────────────────┤
│                                                      │
│ 1️⃣ 流量管理 (Traffic Management)                    │
│    ├─ 路由转发                                       │
│    ├─ 负载均衡                                       │
│    ├─ 熔断降级                                       │
│    └─ 限流控制                                       │
│                                                      │
│ 2️⃣ 安全防护 (Security)                              │
│    ├─ 身份认证 (OAuth2, JWT)                        │
│    ├─ 权限授权 (RBAC, ABAC)                         │
│    ├─ API密钥管理                                    │
│    └─ 防重放攻击                                     │
│                                                      │
│ 3️⃣ 协议转换 (Protocol Translation)                  │
│    ├─ REST ↔ gRPC                                   │
│    ├─ HTTP ↔ WebSocket                              │
│    └─ 版本适配                                       │
│                                                      │
│ 4️⃣ 可观测性 (Observability)                         │
│    ├─ 访问日志                                       │
│    ├─ 指标监控                                       │
│    ├─ 链路追踪                                       │
│    └─ 告警通知                                       │
│                                                      │
│ 5️⃣ 增强功能 (Enhancements)                          │
│    ├─ 请求/响应转换                                  │
│    ├─ 缓存                                           │
│    ├─ API组合                                        │
│    └─ Mock响应                                       │
└──────────────────────────────────────────────────────┘
```

---

## 核心功能

### 路由配置

```yaml
# Kong路由配置示例
apiVersion: configuration.konghq.com/v1
kind: KongIngress
metadata:
  name: api-routes
spec:
  route:
    - name: user-service
      paths:
        - /api/v1/users
      methods:
        - GET
        - POST
      strip_path: true
      protocols:
        - http
        - https

    - name: order-service
      paths:
        - /api/v1/orders
      strip_path: true
      hosts:
        - api.example.com

  upstream:
    hash_on: ip
    hash_fallback: none
    slots: 1000
    healthchecks:
      active:
        http_path: /health
        healthy:
          interval: 10
          successes: 2
        unhealthy:
          interval: 5
          http_failures: 3
```

### 认证授权

```lua
-- Kong JWT插件配置
local jwt_decoder = require "kong.plugins.jwt.jwt_parser"

local function verify_token(token)
  local jwt, err = jwt_decoder:new(token)
  if err then
    return false, "Invalid token"
  end

  -- 验证签名
  if not jwt:verify_signature(jwt_secret) then
    return false, "Invalid signature"
  end

  -- 验证过期时间
  if jwt.claims.exp and jwt.claims.exp < ngx.time() then
    return false, "Token expired"
  end

  -- 验证权限
  if not has_permission(jwt.claims.scope, ngx.var.request_uri) then
    return false, "Insufficient permissions"
  end

  return true, jwt.claims
end

-- OAuth2配置
{
  "name": "oauth2",
  "config": {
    "enable_client_credentials": true,
    "enable_authorization_code": true,
    "token_expiration": 7200,
    "refresh_token_ttl": 1209600,
    "mandatory_scope": true,
    "scopes": ["read", "write", "admin"]
  }
}
```

### 限流策略

```yaml
# 多维度限流配置
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: rate-limiting
config:
  # 基于IP限流
  second: 10
  minute: 100
  hour: 1000

  # 基于用户限流
  limit_by: consumer

  # Redis集群存储
  policy: redis
  redis_host: redis-cluster
  redis_port: 6379
  redis_database: 0

  # 限流响应
  fault_tolerant: true
  hide_client_headers: false
```

---

## 架构模式

### 单体网关 vs 微网关

```
1. 单体API网关 (Monolithic Gateway)
┌────────────────────────────────────────┐
│          单一API网关实例               │
│  ┌──────────────────────────────────┐ │
│  │      所有路由规则                │ │
│  │      所有认证逻辑                │ │
│  │      所有限流配置                │ │
│  └──────────────────────────────────┘ │
└───────────┬────────────────────────────┘
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
 Service Service Service
    A       B       C

优点: 简单、集中管理
缺点: 单点故障、性能瓶颈

────────────────────────────────────────

2. 微网关 (Micro Gateway)
┌────────┐  ┌────────┐  ┌────────┐
│Gateway │  │Gateway │  │Gateway │
│   A    │  │   B    │  │   C    │
└────┬───┘  └────┬───┘  └────┬───┘
     │           │           │
     ▼           ▼           ▼
 Service     Service     Service
    A           B           C

优点: 独立部署、故障隔离
缺点: 管理复杂、配置分散

────────────────────────────────────────

3. 边缘网关 + 内部网关
             外部流量
                │
                ▼
         ┌────────────┐
         │ 边缘网关   │  (安全、限流)
         └──────┬─────┘
                │
       ┌────────┼────────┐
       │        │        │
       ▼        ▼        ▼
    ┌─────┐ ┌─────┐ ┌─────┐
    │内部 │ │内部 │ │内部 │  (路由、聚合)
    │网关A│ │网关B│ │网关C│
    └──┬──┘ └──┬──┘ └──┬──┘
       │       │       │
       ▼       ▼       ▼
    Service Service Service

优点: 职责分离、灵活扩展
缺点: 架构复杂
```

### BFF模式 (Backend For Frontend)

```
┌──────────────────────────────────────────────┐
│         BFF 架构模式                         │
├──────────────────────────────────────────────┤
│                                              │
│  Web客户端          移动客户端      IoT设备 │
│     │                  │              │      │
│     ▼                  ▼              ▼      │
│ ┌────────┐       ┌────────┐      ┌────────┐│
│ │Web BFF │       │App BFF │      │IoT BFF ││
│ └────┬───┘       └────┬───┘      └────┬───┘│
│      │                │               │     │
│      └────────┬───────┴───────┬───────┘     │
│               │               │              │
│          ┌────┴────┐     ┌────┴────┐       │
│          │ User    │     │ Order   │        │
│          │ Service │     │ Service │        │
│          └─────────┘     └─────────┘        │
└──────────────────────────────────────────────┘

每个BFF专门服务一种客户端:
- 定制化数据聚合
- 优化响应格式
- 适配不同协议
```

---

## Kong网关实战

### Kong部署配置

```yaml
# kong-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kong
  namespace: api-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kong
  template:
    metadata:
      labels:
        app: kong
    spec:
      containers:
      - name: kong
        image: kong:3.4
        env:
        - name: KONG_DATABASE
          value: "postgres"
        - name: KONG_PG_HOST
          value: "postgres"
        - name: KONG_PG_PASSWORD
          valueFrom:
            secretKeyRef:
              name: kong-postgres
              key: password
        - name: KONG_PROXY_ACCESS_LOG
          value: "/dev/stdout"
        - name: KONG_ADMIN_ACCESS_LOG
          value: "/dev/stdout"
        - name: KONG_PROXY_ERROR_LOG
          value: "/dev/stderr"
        - name: KONG_ADMIN_ERROR_LOG
          value: "/dev/stderr"
        - name: KONG_ADMIN_LISTEN
          value: "0.0.0.0:8001"
        ports:
        - name: proxy
          containerPort: 8000
        - name: proxy-ssl
          containerPort: 8443
        - name: admin
          containerPort: 8001
        livenessProbe:
          httpGet:
            path: /status
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /status
            port: 8001
          initialDelaySeconds: 10
          periodSeconds: 5
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 2000m
            memory: 2Gi

---
apiVersion: v1
kind: Service
metadata:
  name: kong-proxy
  namespace: api-gateway
spec:
  type: LoadBalancer
  ports:
  - name: proxy
    port: 80
    targetPort: 8000
  - name: proxy-ssl
    port: 443
    targetPort: 8443
  selector:
    app: kong
```

### Kong声明式配置

```yaml
# kong.yaml - 声明式配置
_format_version: "3.0"

services:
  - name: user-service
    url: http://user-service:8080
    routes:
      - name: user-api
        paths:
          - /api/v1/users
        methods:
          - GET
          - POST
          - PUT
          - DELETE
    plugins:
      - name: jwt
        config:
          secret_is_base64: false
      - name: rate-limiting
        config:
          minute: 100
          policy: redis
          redis_host: redis
      - name: cors
        config:
          origins:
            - https://example.com
          methods:
            - GET
            - POST
          headers:
            - Authorization
            - Content-Type

  - name: order-service
    url: http://order-service:8080
    routes:
      - name: order-api
        paths:
          - /api/v1/orders
    plugins:
      - name: request-transformer
        config:
          add:
            headers:
              - X-Service:order
      - name: response-transformer
        config:
          remove:
            headers:
              - X-Internal-Secret

# 全局插件
plugins:
  - name: prometheus
    config:
      per_consumer: true

  - name: correlation-id
    config:
      header_name: X-Request-ID
      generator: uuid

  - name: request-size-limiting
    config:
      allowed_payload_size: 10
```

---

## 性能优化

### 缓存策略

```lua
-- Kong缓存插件
local cache = kong.cache

local function get_user(user_id)
  local cache_key = "user:" .. user_id

  -- 尝试从缓存获取
  local user, err = cache:get(cache_key, {
    ttl = 300,  -- 5分钟过期
    neg_ttl = 60  -- 负缓存1分钟
  }, function()
    -- 缓存未命中,从后端服务获取
    local res = kong.service.request()
    if res.status == 200 then
      return res.body
    else
      return nil, "user not found"
    end
  end)

  return user, err
end

-- HTTP缓存头
kong.response.set_header("Cache-Control", "max-age=300, public")
kong.response.set_header("ETag", etag)
```

### 连接池优化

```nginx
# Kong Nginx配置优化
upstream user_service {
    server user-service:8080 max_fails=3 fail_timeout=30s;
    keepalive 32;
    keepalive_requests 100;
    keepalive_timeout 60s;
}

proxy_http_version 1.1;
proxy_set_header Connection "";

# 连接池大小
lua_socket_pool_size 30;
lua_socket_keepalive_timeout 60s;
```

---

## 高可用设计

### 多区域部署

```yaml
# 多可用区部署
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kong
spec:
  replicas: 6
  template:
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - kong
            topologyKey: topology.kubernetes.io/zone

      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: kong
```

### 健康检查

```yaml
# 主动健康检查
healthchecks:
  active:
    type: http
    http_path: /health
    timeout: 1
    concurrency: 10
    healthy:
      interval: 5
      successes: 2
      http_statuses: [200, 302]
    unhealthy:
      interval: 3
      http_failures: 3
      tcp_failures: 2
      timeouts: 3
      http_statuses: [429, 500, 503]

  passive:
    type: http
    healthy:
      successes: 5
      http_statuses: [200, 201, 202, 203, 204, 205, 206, 207, 208, 226, 300, 301, 302, 303, 304, 305, 306, 307, 308]
    unhealthy:
      http_failures: 3
      tcp_failures: 2
      timeouts: 2
      http_statuses: [429, 500, 503]
```

---

## 实战案例

### 案例: 电商API网关

```yaml
# ecommerce-gateway.yaml
services:
  # 用户服务
  - name: user-service
    url: http://user-service
    routes:
      - name: user-login
        paths: [/api/v1/auth/login]
        methods: [POST]
        plugins:
          - name: rate-limiting
            config:
              minute: 5  # 登录限流
          - name: bot-detection

      - name: user-profile
        paths: [/api/v1/users/me]
        methods: [GET, PUT]
        plugins:
          - name: jwt
          - name: response-transformer
            config:
              remove:
                json: ["password", "salt"]

  # 商品服务
  - name: product-service
    url: http://product-service
    routes:
      - name: product-list
        paths: [/api/v1/products]
        methods: [GET]
        plugins:
          - name: proxy-cache
            config:
              strategy: memory
              cache_ttl: 300
          - name: response-transformer
            config:
              add:
                headers:
                  - X-Cache-Status:HIT

  # 订单服务
  - name: order-service
    url: http://order-service
    routes:
      - name: create-order
        paths: [/api/v1/orders]
        methods: [POST]
        plugins:
          - name: jwt
          - name: request-validator
            config:
              body_schema: |
                {
                  "type": "object",
                  "required": ["items", "address"],
                  "properties": {
                    "items": {"type": "array"},
                    "address": {"type": "string"}
                  }
                }
          - name: pre-function
            config:
              functions: |
                return function()
                  -- 幂等性检查
                  local request_id = kong.request.get_header("X-Idempotency-Key")
                  if request_id then
                    local cache_key = "idempotency:" .. request_id
                    local cached = kong.cache:get(cache_key)
                    if cached then
                      return kong.response.exit(200, cached)
                    end
                  end
                end

  # 支付服务
  - name: payment-service
    url: http://payment-service
    routes:
      - name: payment
        paths: [/api/v1/payments]
        methods: [POST]
        plugins:
          - name: jwt
          - name: request-termination  # 维护模式
            config:
              status_code: 503
              message: "Payment service under maintenance"
            enabled: false  # 可动态开启
```

---

## 总结

### API网关选型对比

```
┌────────────────────────────────────────────────────┐
│          主流API网关对比                           │
├──────────┬──────────┬──────────┬─────────────────┤
│ 网关     │ Kong     │ APISIX   │ Nginx/OpenResty │
├──────────┼──────────┼──────────┼─────────────────┤
│ 语言     │ Lua      │ Lua      │ Lua/C           │
│ 性能     │ ⭐⭐⭐  │ ⭐⭐⭐⭐│ ⭐⭐⭐⭐⭐      │
│ 插件     │ 丰富     │ 丰富     │ 需自开发        │
│ 易用性   │ ⭐⭐⭐⭐│ ⭐⭐⭐  │ ⭐⭐            │
│ 社区     │ 活跃     │ 活跃     │ 成熟            │
│ 动态配置 │ ✅       │ ✅       │ ❌              │
│ 开源协议 │ Apache   │ Apache   │ BSD             │
└──────────┴──────────┴──────────┴─────────────────┘
```

### 下一步学习
- [02_routing_strategies.md](02_routing_strategies.md) - 路由策略
- [03_authentication.md](03_authentication.md) - 认证授权
- [04_gateway_comparison.md](04_gateway_comparison.md) - 网关对比
