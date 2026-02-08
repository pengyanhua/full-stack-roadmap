# API 网关路由与流量控制

## 目录
- [路由策略](#路由策略)
- [负载均衡](#负载均衡)
- [流量控制](#流量控制)
- [熔断降级](#熔断降级)
- [灰度发布](#灰度发布)
- [流量镜像](#流量镜像)
- [实战案例](#实战案例)

---

## 路由策略

### 路由匹配规则

```
┌──────────────────────────────────────────────────┐
│            API 网关路由匹配优先级                │
├──────────────────────────────────────────────────┤
│                                                  │
│  1. 精确路径匹配                                 │
│     /api/v1/users/123  →  精确匹配               │
│                                                  │
│  2. 前缀匹配                                     │
│     /api/v1/users/*    →  前缀匹配               │
│                                                  │
│  3. 正则表达式匹配                               │
│     /api/v[0-9]+/.*    →  正则匹配               │
│                                                  │
│  4. Header匹配                                   │
│     X-API-Version: v1  →  Header路由             │
│                                                  │
│  5. Query参数匹配                                │
│     ?version=v1        →  Query路由              │
│                                                  │
│  6. 主机名匹配                                   │
│     api.example.com    →  Host路由               │
└──────────────────────────────────────────────────┘
```

### Kong路由配置

```yaml
# 基于路径的路由
routes:
  - name: user-exact
    paths:
      - /api/v1/users/profile  # 精确匹配
    strip_path: false
    service: user-service

  - name: user-prefix
    paths:
      - /api/v1/users  # 前缀匹配
    strip_path: true
    service: user-service

  - name: user-regex
    paths:
      - ~/api/v\d+/users  # 正则匹配
    strip_path: true
    service: user-service

# 基于Header的路由
  - name: mobile-api
    paths:
      - /api/users
    headers:
      User-Agent:
        - ".*Mobile.*"
    service: mobile-user-service

# 基于Method的路由
  - name: user-read
    paths:
      - /api/v1/users
    methods:
      - GET
      - HEAD
    service: user-read-service

  - name: user-write
    paths:
      - /api/v1/users
    methods:
      - POST
      - PUT
      - DELETE
    service: user-write-service

# 基于域名的路由
  - name: api-v1
    hosts:
      - api-v1.example.com
    service: legacy-service

  - name: api-v2
    hosts:
      - api-v2.example.com
    service: new-service
```

---

## 负载均衡

### 负载均衡算法

```
┌──────────────────────────────────────────────────┐
│           负载均衡算法对比                       │
├──────────────────────────────────────────────────┤
│                                                  │
│  Round Robin (轮询)                              │
│  ┌────┐  ┌────┐  ┌────┐                         │
│  │ S1 │←→│ S2 │←→│ S3 │                         │
│  └────┘  └────┘  └────┘                         │
│    ↑       ↑       ↑                             │
│    1       2       3  (依次分配)                 │
│                                                  │
│  优点: 简单、公平                                │
│  缺点: 不考虑服务器负载                          │
│                                                  │
│ ────────────────────────────────────────────    │
│                                                  │
│  Least Connections (最少连接)                   │
│  ┌────┐  ┌────┐  ┌────┐                         │
│  │ S1 │  │ S2 │  │ S3 │                         │
│  │ 5  │  │ 2  │← │ 8  │  (选择S2)               │
│  └────┘  └────┘  └────┘                         │
│                                                  │
│  优点: 考虑实际负载                              │
│  缺点: 需要维护连接状态                          │
│                                                  │
│ ────────────────────────────────────────────    │
│                                                  │
│  Weighted (加权)                                 │
│  ┌────┐  ┌────┐  ┌────┐                         │
│  │ S1 │  │ S2 │  │ S3 │                         │
│  │ w:3│  │ w:2│  │ w:1│                         │
│  └────┘  └────┘  └────┘                         │
│    50%    33%    17%  (按权重分配)               │
│                                                  │
│  优点: 适应不同性能服务器                        │
│  缺点: 需要配置权重                              │
│                                                  │
│ ────────────────────────────────────────────    │
│                                                  │
│  IP Hash (IP哈希)                                │
│  Client IP → Hash → Server                      │
│  192.168.1.1 → hash % 3 = 1 → S2                │
│                                                  │
│  优点: 会话保持                                  │
│  缺点: 分布可能不均                              │
└──────────────────────────────────────────────────┘
```

### Kong负载均衡配置

```yaml
# upstream配置
upstreams:
  - name: user-service-upstream
    algorithm: round-robin  # 或 least-connections, consistent-hashing
    hash_on: ip  # 基于IP哈希
    hash_fallback: none
    slots: 10000

    targets:
      - target: user-service-1:8080
        weight: 100
      - target: user-service-2:8080
        weight: 100
      - target: user-service-3:8080
        weight: 50  # 性能较低,降低权重

    # 健康检查
    healthchecks:
      active:
        type: http
        http_path: /health
        timeout: 1
        concurrency: 10
        healthy:
          interval: 5
          successes: 2
        unhealthy:
          interval: 3
          http_failures: 3
          timeouts: 3

      passive:
        type: http
        healthy:
          successes: 5
        unhealthy:
          http_failures: 3
          timeouts: 2
```

---

## 流量控制

### 限流策略

```yaml
# 多级限流配置
plugins:
  # 全局限流
  - name: rate-limiting
    config:
      second: 100
      minute: 5000
      hour: 100000
      policy: redis
      redis_host: redis-cluster
      fault_tolerant: true

  # 基于消费者限流
  - name: rate-limiting
    consumer: premium-user
    config:
      second: 1000  # VIP用户更高限额
      minute: 50000

  # 基于IP限流
  - name: ip-restriction
    config:
      allow:
        - 10.0.0.0/8
        - 192.168.0.0/16
      deny:
        - 1.2.3.4

# 并发限流
  - name: request-size-limiting
    config:
      allowed_payload_size: 10  # 10MB
      size_unit: megabytes

# 连接数限流
  - name: request-termination
    config:
      status_code: 429
      message: "Too many requests"
```

### 限流算法实现

```lua
-- Kong 令牌桶算法实现
local function token_bucket_limit(key, rate, capacity)
  local redis = require "resty.redis"
  local red = redis:new()
  red:connect("redis", 6379)

  local bucket_key = "rate_limit:" .. key
  local now = ngx.now()

  -- 获取当前桶状态
  local res, err = red:hmget(bucket_key, "tokens", "last_refill")
  local tokens = tonumber(res[1]) or capacity
  local last_refill = tonumber(res[2]) or now

  -- 计算应该补充的令牌
  local time_passed = now - last_refill
  local tokens_to_add = time_passed * rate
  tokens = math.min(capacity, tokens + tokens_to_add)

  -- 尝试消费一个令牌
  if tokens >= 1 then
    tokens = tokens - 1

    -- 更新桶状态
    red:hmset(bucket_key,
      "tokens", tokens,
      "last_refill", now
    )
    red:expire(bucket_key, 3600)

    return true, tokens
  else
    return false, 0
  end
end

-- 使用示例
local allowed, remaining = token_bucket_limit(
  ngx.var.remote_addr,
  10,  -- 每秒10个令牌
  100  -- 桶容量100
)

if not allowed then
  return kong.response.exit(429, {
    message = "Rate limit exceeded",
    retry_after = 1
  })
end

kong.response.set_header("X-RateLimit-Remaining", remaining)
```

---

## 熔断降级

### 熔断器模式

```
┌──────────────────────────────────────────────────┐
│             熔断器状态机                         │
├──────────────────────────────────────────────────┤
│                                                  │
│           失败率 > 阈值                          │
│   Closed ──────────────▶ Open                    │
│     │                      │                     │
│     │                      │ 超时后               │
│     │                      ▼                     │
│     │                  Half-Open                 │
│     │ 成功 ◀───────────────┘                     │
│     │                                            │
│                                                  │
│  Closed (闭合):                                  │
│    - 正常处理请求                                │
│    - 统计失败率                                  │
│                                                  │
│  Open (断开):                                    │
│    - 快速失败,不调用后端                         │
│    - 返回降级响应                                │
│    - 等待恢复窗口                                │
│                                                  │
│  Half-Open (半开):                               │
│    - 允许部分请求通过                            │
│    - 测试后端是否恢复                            │
│    - 成功则转Closed,失败则转Open                │
└──────────────────────────────────────────────────┘
```

### Kong熔断配置

```yaml
# Circuit Breaker插件
plugins:
  - name: circuit-breaker
    config:
      # 触发条件
      failure_threshold: 10  # 失败次数阈值
      failure_rate_threshold: 0.5  # 失败率50%
      window_size: 60  # 统计窗口60秒

      # 熔断时间
      open_duration: 30  # 熔断30秒

      # 半开状态
      half_open_requests: 3  # 半开时允许3个请求

      # 降级响应
      fallback_response:
        status: 503
        message: "Service temporarily unavailable"
        headers:
          Retry-After: "30"
```

### 自定义降级逻辑

```lua
-- Kong 降级插件
local kong = kong
local circuit_breaker = {}

function circuit_breaker:access(conf)
  local service_key = "cb:" .. kong.router.get_service().id
  local state = get_circuit_state(service_key)

  if state == "open" then
    -- 熔断状态,返回降级响应
    return kong.response.exit(503, {
      message = conf.fallback_message or "Service unavailable",
      fallback_data = get_cached_response(service_key)
    })

  elseif state == "half_open" then
    -- 半开状态,限制请求数
    if not can_pass_half_open(service_key) then
      return kong.response.exit(503, {
        message = "Circuit breaker is recovering"
      })
    end
  end
end

function circuit_breaker:log(conf)
  local service_key = "cb:" .. kong.router.get_service().id
  local status = kong.response.get_status()

  -- 记录请求结果
  if status >= 500 then
    record_failure(service_key)
  else
    record_success(service_key)
  end

  -- 检查是否需要熔断
  check_and_update_state(service_key, conf)
end

return circuit_breaker
```

---

## 灰度发布

### 基于权重的灰度

```yaml
# Kong灰度发布配置
upstreams:
  - name: user-service
    targets:
      # 旧版本 90%
      - target: user-service-v1:8080
        weight: 900

      # 新版本 10%
      - target: user-service-v2:8080
        weight: 100

# 逐步调整权重
# 阶段1: 90/10
# 阶段2: 70/30
# 阶段3: 50/50
# 阶段4: 30/70
# 阶段5: 0/100
```

### 基于Header的灰度

```lua
-- Kong 灰度插件
local function canary_routing(conf)
  local headers = kong.request.get_headers()

  -- 1. Header灰度
  if headers["X-Canary"] == "true" then
    return "canary"
  end

  -- 2. 用户白名单
  local user_id = headers["X-User-ID"]
  if user_id and is_in_whitelist(user_id) then
    return "canary"
  end

  -- 3. 百分比灰度
  local random = math.random(100)
  if random <= conf.canary_percentage then
    return "canary"
  end

  return "stable"
end

-- 路由到不同上游
local version = canary_routing(conf)
if version == "canary" then
  kong.service.set_upstream("user-service-canary")
else
  kong.service.set_upstream("user-service-stable")
end
```

---

## 流量镜像

### 流量复制配置

```yaml
# Kong流量镜像插件
plugins:
  - name: request-mirror
    config:
      mirror_upstream: http://test-service:8080
      mirror_percentage: 10  # 复制10%流量
      async: true  # 异步发送,不影响主请求
      preserve_host: true
```

### 自定义镜像逻辑

```lua
-- 流量镜像实现
local http = require "resty.http"

local function mirror_request(conf)
  -- 只镜像特定比例的流量
  if math.random(100) > conf.percentage then
    return
  end

  -- 异步镜像请求
  ngx.timer.at(0, function()
    local httpc = http.new()

    -- 复制请求
    local res, err = httpc:request_uri(conf.mirror_url, {
      method = ngx.var.request_method,
      body = kong.request.get_raw_body(),
      headers = kong.request.get_headers(),
      keepalive_timeout = 60,
      keepalive_pool = 10
    })

    -- 记录镜像结果
    if not res then
      ngx.log(ngx.ERR, "Mirror request failed: ", err)
    end
  end)
end
```

---

## 实战案例

### 案例: 微服务API网关路由

```yaml
# microservices-gateway.yaml
_format_version: "3.0"

# 服务定义
services:
  # 用户服务 (读写分离)
  - name: user-read-service
    url: http://user-read-cluster
    routes:
      - name: user-query
        paths:
          - /api/v1/users
        methods: [GET, HEAD]
    plugins:
      - name: proxy-cache
        config:
          cache_ttl: 300
          strategy: memory

  - name: user-write-service
    url: http://user-write-cluster
    routes:
      - name: user-mutation
        paths:
          - /api/v1/users
        methods: [POST, PUT, DELETE]
    plugins:
      - name: rate-limiting
        config:
          minute: 100

  # 订单服务 (灰度发布)
  - name: order-service
    url: http://order-service-upstream
    routes:
      - name: order-api
        paths:
          - /api/v1/orders
    plugins:
      - name: canary
        config:
          upstream_host: order-service-canary
          percentage: 10
          hash: header
          hash_header: X-User-ID

  # 支付服务 (熔断保护)
  - name: payment-service
    url: http://payment-service
    routes:
      - name: payment-api
        paths:
          - /api/v1/payments
    plugins:
      - name: circuit-breaker
        config:
          failure_threshold: 5
          open_duration: 60
          fallback_response:
            status: 503
            message: "Payment service unavailable, please try again later"

# 上游配置
upstreams:
  - name: user-read-cluster
    algorithm: least-connections
    targets:
      - target: user-read-1:8080
        weight: 100
      - target: user-read-2:8080
        weight: 100
      - target: user-read-3:8080
        weight: 100

  - name: order-service-upstream
    algorithm: consistent-hashing
    hash_on: header
    hash_on_header: X-User-ID
    targets:
      - target: order-v1:8080
        weight: 900  # 90%
      - target: order-v2:8080
        weight: 100  # 10%
```

---

## 总结

### 路由策略选择指南

```
┌────────────────────────────────────────────────┐
│          路由策略选择决策树                    │
├────────────────────────────────────────────────┤
│                                                │
│  需要会话保持?                                 │
│    ├─ 是 → IP Hash / Cookie Hash              │
│    └─ 否                                       │
│         │                                      │
│         └─ 服务器性能不同?                     │
│              ├─ 是 → Weighted Round Robin      │
│              └─ 否 → Round Robin / Least Conn  │
│                                                │
│  需要灰度发布?                                 │
│    └─ 是 → Header路由 + 权重调整               │
│                                                │
│  高可用要求?                                   │
│    └─ 是 → 熔断器 + 降级策略                   │
└────────────────────────────────────────────────┘
```

### 关键要点

1. **路由匹配**: 精确 > 前缀 > 正则
2. **负载均衡**: 根据场景选择算法
3. **流量控制**: 多级限流保护系统
4. **熔断降级**: 防止级联故障
5. **灰度发布**: 降低发布风险

### 下一步学习
- [03_authentication.md](03_authentication.md) - 统一认证授权
- [04_gateway_comparison.md](04_gateway_comparison.md) - 网关对比
