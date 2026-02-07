# API 网关架构设计

## 目录
- [API 网关概述](#api-网关概述)
- [核心功能](#核心功能)
- [架构模式](#架构模式)
- [Kong 完整配置](#kong-完整配置)
- [APISIX 完整配置](#apisix-完整配置)
- [性能优化](#性能优化)
- [实战案例](#实战案例)

---

## API 网关概述

### 什么是 API 网关

API 网关是微服务架构中的关键组件，作为所有客户端请求的统一入口。

```
传统架构                          API 网关架构
┌──────────┐                    ┌──────────┐
│  客户端  │                    │  客户端  │
└────┬─────┘                    └────┬─────┘
     │                                │
     ├───────────────┐                │
     │               │                ▼
┌────▼────┐   ┌─────▼────┐    ┌──────────────┐
│ 服务 A  │   │  服务 B  │    │  API Gateway │
│         │   │          │    │  ┌─────────┐ │
│ ✗ 认证  │   │  ✗ 认证  │    │  │  认证   │ │
│ ✗ 限流  │   │  ✗ 限流  │    │  │  限流   │ │
│ ✗ 日志  │   │  ✗ 日志  │    │  │  日志   │ │
└─────────┘   └──────────┘    │  │  监控   │ │
                               │  └─────────┘ │
  重复代码                     └──────┬───────┘
  分散管理                            │
                               ┌──────┴───────┐
                               │              │
                         ┌─────▼────┐  ┌─────▼────┐
                         │  服务 A  │  │  服务 B  │
                         │  (纯业务)│  │  (纯业务)│
                         └──────────┘  └──────────┘

                         集中管理、代码解耦
```

### API 网关的价值

```
┌─────────────────────────────────────────────────────┐
│              API 网关核心价值                        │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. 统一入口 (Single Entry Point)                   │
│     ├─ 客户端只需知道网关地址                       │
│     └─ 简化客户端配置                               │
│                                                      │
│  2. 协议转换 (Protocol Translation)                 │
│     ├─ HTTP/gRPC/WebSocket 互转                     │
│     └─ REST 转 GraphQL                              │
│                                                      │
│  3. 安全防护 (Security)                             │
│     ├─ 认证授权 (JWT/OAuth2)                        │
│     ├─ IP 黑白名单                                  │
│     ├─ WAF 防护                                     │
│     └─ 限流防刷                                     │
│                                                      │
│  4. 流量管理 (Traffic Management)                   │
│     ├─ 负载均衡                                     │
│     ├─ 灰度发布                                     │
│     ├─ 熔断降级                                     │
│     └─ 超时重试                                     │
│                                                      │
│  5. 可观测性 (Observability)                        │
│     ├─ 访问日志                                     │
│     ├─ 指标采集                                     │
│     └─ 链路追踪                                     │
│                                                      │
│  6. 服务编排 (Service Orchestration)                │
│     ├─ 请求聚合                                     │
│     └─ 响应转换                                     │
└─────────────────────────────────────────────────────┘
```

---

## 核心功能

### 1. 路由转发

```
                   路由匹配流程
┌─────────────────────────────────────────────┐
│  客户端请求: GET /api/v1/users/123          │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────┐
│  路由规则匹配                               │
│  ┌──────────────────────────────────────┐  │
│  │  Rule 1: /api/v1/users/*  ✓ 匹配    │  │
│  │  Rule 2: /api/v1/orders/* ✗ 不匹配  │  │
│  │  Rule 3: /api/v2/*        ✗ 不匹配  │  │
│  └──────────────────────────────────────┘  │
└────────────────┬───────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────┐
│  负载均衡选择后端                           │
│  ┌────────────┐  ┌────────────┐           │
│  │ user-svc-1 │  │ user-svc-2 │           │
│  │  (健康)    │  │  (健康)    │           │
│  └─────┬──────┘  └────────────┘           │
│        │ 选中                               │
└────────┼────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│  转发到: http://user-svc-1:8080/v1/users/123│
└────────────────────────────────────────────┘
```

### 2. 认证授权流程

```
         完整的认证授权流程
┌─────────────────────────────────┐
│  1. 客户端请求                  │
│     Authorization: Bearer <JWT> │
└──────────────┬──────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  2. API Gateway 验证             │
│     ├─ JWT 签名验证              │
│     ├─ 过期时间检查              │
│     └─ 解析 Claims               │
└──────────────┬───────────────────┘
               │
          ┌────┴────┐
          │         │
         失败      成功
          │         │
          │         ▼
          │    ┌────────────────────┐
          │    │  3. 权限检查       │
          │    │     RBAC/ABAC      │
          │    └────┬───────────────┘
          │         │
          │    ┌────┴────┐
          │    │         │
          │   拒绝      允许
          │    │         │
          ▼    ▼         ▼
┌─────────────────┐  ┌──────────────┐
│  4. 返回 401    │  │  5. 转发请求 │
│     或 403      │  │     到后端   │
└─────────────────┘  └──────────────┘
```

---

## 架构模式

### 1. 单体网关 vs 微网关

```
┌───────────────────────────────────────────────────┐
│                  单体网关架构                     │
├───────────────────────────────────────────────────┤
│                                                   │
│         所有流量经过同一个网关集群                │
│                                                   │
│   ┌─────────────────────────────────────┐        │
│   │        API Gateway Cluster          │        │
│   │  ┌────┐  ┌────┐  ┌────┐  ┌────┐   │        │
│   │  │ GW1│  │ GW2│  │ GW3│  │ GW4│   │        │
│   │  └─┬──┘  └─┬──┘  └─┬──┘  └─┬──┘   │        │
│   └────┼───────┼───────┼───────┼────────┘        │
│        │       │       │       │                 │
│   ┌────┴───────┴───────┴───────┴────┐            │
│   │                                  │            │
│   ▼           ▼           ▼          ▼            │
│ 用户服务   订单服务   支付服务   物流服务         │
│                                                   │
│  优点: 统一管理、配置简单                         │
│  缺点: 单点瓶颈、扩展困难                         │
└───────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────┐
│                  微网关架构 (BFF)                  │
├───────────────────────────────────────────────────┤
│                                                   │
│      Web 客户端          移动客户端               │
│          │                   │                    │
│          ▼                   ▼                    │
│    ┌──────────┐        ┌──────────┐              │
│    │  Web GW  │        │ Mobile GW│              │
│    └────┬─────┘        └────┬─────┘              │
│         │                   │                    │
│         └───────┬───────────┘                    │
│                 │                                │
│    ┌────────────┼────────────┐                  │
│    ▼            ▼            ▼                   │
│  用户服务    订单服务    支付服务                 │
│                                                   │
│  优点: 针对性优化、隔离故障                       │
│  缺点: 维护成本高、配置分散                       │
└───────────────────────────────────────────────────┘
```

### 2. 网关集群高可用架构

```
                  高可用网关架构
┌─────────────────────────────────────────────┐
│              客户端 (全球分布)              │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│              DNS / Global LB                │
│         (GeoDNS + 健康检查)                 │
└──────┬──────────────────────┬───────────────┘
       │                      │
       ▼                      ▼
┌──────────────┐      ┌──────────────┐
│  Region A    │      │  Region B    │
│  (主区域)    │      │  (备区域)    │
│              │      │              │
│  ┌────────┐  │      │  ┌────────┐  │
│  │  LB    │  │      │  │  LB    │  │
│  └───┬────┘  │      │  └───┬────┘  │
│      │       │      │      │       │
│  ┌───┴───┐   │      │  ┌───┴───┐   │
│  │Gateway│   │      │  │Gateway│   │
│  │ ┌──┬──┤   │      │  │ ┌──┬──┤   │
│  │ │GW│GW│   │      │  │ │GW│GW│   │
│  │ └──┴──┘   │      │  │ └──┴──┘   │
│  └───┬───┘   │      │  └───┬───┘   │
│      │       │      │      │       │
│  ┌───┴───┐   │      │  ┌───┴───┐   │
│  │ Redis │   │      │  │ Redis │   │
│  └───────┘   │      │  └───────┘   │
└──────┬───────┘      └──────┬────────┘
       │                     │
       └──────────┬──────────┘
                  │
          ┌───────┴────────┐
          │                │
     ┌────▼────┐      ┌────▼────┐
     │ 服务 A  │      │ 服务 B  │
     │(多实例) │      │(多实例) │
     └─────────┘      └─────────┘
```

---

## Kong 完整配置

### Kong 架构

```
┌────────────────────────────────────────────┐
│              Kong 架构                     │
├────────────────────────────────────────────┤
│                                            │
│  ┌──────────────────────────────────┐     │
│  │       Admin API (8001)           │     │
│  │    (配置管理接口)                │     │
│  └──────────────┬───────────────────┘     │
│                 │                          │
│  ┌──────────────▼───────────────────┐     │
│  │         Kong Core                │     │
│  │  ┌────────────────────────────┐  │     │
│  │  │     Plugin System           │  │     │
│  │  │  ┌──────┐  ┌──────┐        │  │     │
│  │  │  │ 认证 │  │ 限流 │  ...   │  │     │
│  │  │  └──────┘  └──────┘        │  │     │
│  │  └────────────────────────────┘  │     │
│  └──────────────┬───────────────────┘     │
│                 │                          │
│  ┌──────────────▼───────────────────┐     │
│  │       Proxy (8000/8443)          │     │
│  │    (数据平面 - 处理请求)         │     │
│  └──────────────┬───────────────────┘     │
│                 │                          │
│  ┌──────────────▼───────────────────┐     │
│  │       PostgreSQL / Cassandra     │     │
│  │    (存储配置和状态)              │     │
│  └──────────────────────────────────┘     │
└────────────────────────────────────────────┘
```

### Kong 声明式配置

```yaml
# kong.yml - Kong 完整配置文件
_format_version: "3.0"

services:
  # 用户服务
  - name: user-service
    url: http://user-service:8080
    protocol: http
    connect_timeout: 5000
    write_timeout: 60000
    read_timeout: 60000
    retries: 3

    routes:
      - name: user-api
        paths:
          - /api/v1/users
        methods:
          - GET
          - POST
          - PUT
          - DELETE
        strip_path: false
        preserve_host: false

    plugins:
      # JWT 认证
      - name: jwt
        config:
          claims_to_verify:
            - exp
          key_claim_name: kid
          secret_is_base64: false

      # 限流插件
      - name: rate-limiting
        config:
          minute: 100
          hour: 10000
          policy: redis
          redis_host: redis.default.svc.cluster.local
          redis_port: 6379
          redis_database: 0
          fault_tolerant: true

      # CORS 跨域
      - name: cors
        config:
          origins:
            - https://example.com
          methods:
            - GET
            - POST
            - PUT
            - DELETE
          headers:
            - Authorization
            - Content-Type
          exposed_headers:
            - X-Auth-Token
          credentials: true
          max_age: 3600

      # 请求响应转换
      - name: request-transformer
        config:
          add:
            headers:
              - X-Gateway-ID:kong
              - X-Request-ID:$(uuid)
          remove:
            headers:
              - X-Internal-Header

      # Prometheus 指标
      - name: prometheus
        config:
          per_consumer: true

      # 日志记录
      - name: file-log
        config:
          path: /var/log/kong/access.log
          reopen: true

  # 订单服务
  - name: order-service
    url: http://order-service:8080

    routes:
      - name: order-api
        paths:
          - /api/v1/orders

    plugins:
      # OAuth2 认证
      - name: oauth2
        config:
          scopes:
            - read
            - write
          mandatory_scope: true
          token_expiration: 7200
          enable_authorization_code: true
          enable_client_credentials: true
          enable_implicit_grant: false

      # IP 限制
      - name: ip-restriction
        config:
          allow:
            - 10.0.0.0/8
            - 172.16.0.0/12

      # 熔断器
      - name: circuit-breaker
        config:
          threshold: 10
          window_size: 60
          timeout: 30

# 全局插件
plugins:
  # 全局请求大小限制
  - name: request-size-limiting
    config:
      allowed_payload_size: 10
      size_unit: megabytes

  # 全局请求终止 (维护模式)
  - name: request-termination
    enabled: false
    config:
      status_code: 503
      message: Service temporarily unavailable

# 消费者 (API 客户端)
consumers:
  - username: mobile-app
    custom_id: mobile-client-001

    # JWT 凭证
    jwt_secrets:
      - key: mobile-app-key
        algorithm: HS256
        secret: your-256-bit-secret

    # ACL 分组
    acls:
      - group: mobile-clients

  - username: web-app
    custom_id: web-client-001

    jwt_secrets:
      - key: web-app-key
        algorithm: RS256
        rsa_public_key: |
          -----BEGIN PUBLIC KEY-----
          MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA...
          -----END PUBLIC KEY-----

# 上游健康检查
upstreams:
  - name: user-upstream
    algorithm: round-robin
    hash_on: none
    hash_fallback: none

    healthchecks:
      active:
        type: http
        http_path: /health
        timeout: 1
        concurrency: 10
        healthy:
          interval: 10
          successes: 2
        unhealthy:
          interval: 5
          http_failures: 3
          timeouts: 3

      passive:
        healthy:
          successes: 5
        unhealthy:
          http_failures: 3
          timeouts: 2

    targets:
      - target: user-service-1:8080
        weight: 100
      - target: user-service-2:8080
        weight: 100
```

### Kong Docker Compose 部署

```yaml
# docker-compose.yml
version: '3.8'

services:
  kong-database:
    image: postgres:15-alpine
    container_name: kong-database
    environment:
      POSTGRES_USER: kong
      POSTGRES_DB: kong
      POSTGRES_PASSWORD: kong
    volumes:
      - kong_data:/var/lib/postgresql/data
    networks:
      - kong-net
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "kong"]
      interval: 10s
      timeout: 5s
      retries: 5

  kong-migration:
    image: kong:3.5
    container_name: kong-migration
    command: kong migrations bootstrap
    environment:
      KONG_DATABASE: postgres
      KONG_PG_HOST: kong-database
      KONG_PG_DATABASE: kong
      KONG_PG_USER: kong
      KONG_PG_PASSWORD: kong
    depends_on:
      kong-database:
        condition: service_healthy
    networks:
      - kong-net

  kong:
    image: kong:3.5
    container_name: kong
    environment:
      KONG_DATABASE: postgres
      KONG_PG_HOST: kong-database
      KONG_PG_DATABASE: kong
      KONG_PG_USER: kong
      KONG_PG_PASSWORD: kong
      KONG_PROXY_ACCESS_LOG: /dev/stdout
      KONG_ADMIN_ACCESS_LOG: /dev/stdout
      KONG_PROXY_ERROR_LOG: /dev/stderr
      KONG_ADMIN_ERROR_LOG: /dev/stderr
      KONG_ADMIN_LISTEN: 0.0.0.0:8001
      KONG_ADMIN_GUI_URL: http://localhost:8002
    ports:
      - "8000:8000"   # Proxy HTTP
      - "8443:8443"   # Proxy HTTPS
      - "8001:8001"   # Admin API HTTP
      - "8444:8444"   # Admin API HTTPS
    depends_on:
      kong-database:
        condition: service_healthy
      kong-migration:
        condition: service_completed_successfully
    networks:
      - kong-net
    healthcheck:
      test: ["CMD", "kong", "health"]
      interval: 10s
      timeout: 10s
      retries: 10

  konga:
    image: pantsel/konga:latest
    container_name: konga
    environment:
      NODE_ENV: production
      DB_ADAPTER: postgres
      DB_HOST: kong-database
      DB_PORT: 5432
      DB_USER: kong
      DB_PASSWORD: kong
      DB_DATABASE: konga
    ports:
      - "1337:1337"
    depends_on:
      - kong-database
    networks:
      - kong-net

volumes:
  kong_data:

networks:
  kong-net:
    driver: bridge
```

---

## APISIX 完整配置

### APISIX 架构

```
┌────────────────────────────────────────────┐
│             APISIX 架构                    │
├────────────────────────────────────────────┤
│                                            │
│  ┌──────────────────────────────────┐     │
│  │      Admin API (9180)            │     │
│  │   (RESTful 配置接口)             │     │
│  └──────────────┬───────────────────┘     │
│                 │                          │
│  ┌──────────────▼───────────────────┐     │
│  │         etcd Cluster             │     │
│  │    (配置中心 - 毫秒级同步)       │     │
│  └──────────────┬───────────────────┘     │
│                 │ Watch                    │
│  ┌──────────────▼───────────────────┐     │
│  │        APISIX Core               │     │
│  │  ┌────────────────────────────┐  │     │
│  │  │   Plugin Runner (多语言)   │  │     │
│  │  │  ┌──────┐  ┌──────┐        │  │     │
│  │  │  │ Lua  │  │ Java │  ...   │  │     │
│  │  │  └──────┘  └──────┘        │  │     │
│  │  └────────────────────────────┘  │     │
│  └──────────────┬───────────────────┘     │
│                 │                          │
│  ┌──────────────▼───────────────────┐     │
│  │       Proxy (9080/9443)          │     │
│  │    (数据平面 - 高性能转发)       │     │
│  └──────────────────────────────────┘     │
└────────────────────────────────────────────┘
```

### APISIX 配置文件

```yaml
# config.yaml - APISIX 主配置
apisix:
  node_listen:
    - 9080
  enable_admin: true
  enable_admin_cors: true
  enable_debug: false
  enable_dev_mode: false
  enable_reuseport: true
  enable_ipv6: true

  config_center: etcd

  proxy_cache:
    zones:
      - name: disk_cache_one
        memory_size: 50m
        disk_size: 1G
        disk_path: /tmp/disk_cache_one
        cache_levels: 1:2

  delete_uri_tail_slash: false
  router:
    http: radixtree_uri
    ssl: radixtree_sni

  stream_proxy:
    only: false
    tcp:
      - 9100
    udp:
      - 9200

nginx_config:
  error_log: logs/error.log
  error_log_level: warn
  worker_processes: auto
  worker_rlimit_nofile: 20480

  event:
    worker_connections: 10620

  http:
    access_log: logs/access.log
    keepalive_timeout: 60s
    client_header_timeout: 60s
    client_body_timeout: 60s
    send_timeout: 10s

    underscores_in_headers: on
    real_ip_header: X-Real-IP

    upstream:
      keepalive: 320
      keepalive_requests: 1000
      keepalive_timeout: 60s

etcd:
  host:
    - "http://etcd:2379"
  prefix: /apisix
  timeout: 30
  startup_retry: 2

plugins:
  - real-ip
  - client-control
  - proxy-rewrite
  - ext-plugin-pre-req
  - request-id
  - zipkin
  - prometheus
  - key-auth
  - jwt-auth
  - basic-auth
  - authz-keycloak
  - wolf-rbac
  - openid-connect
  - hmac-auth
  - ip-restriction
  - ua-restriction
  - referer-restriction
  - uri-blocker
  - request-validation
  - cors
  - limit-req
  - limit-conn
  - limit-count
  - proxy-cache
  - proxy-mirror
  - kafka-logger
  - http-logger
  - tcp-logger
  - udp-logger
  - syslog
  - fault-injection
  - serverless-pre-function
  - serverless-post-function
  - grpc-transcode
  - response-rewrite

stream_plugins:
  - mqtt-proxy
  - ip-restriction
  - limit-conn

plugin_attr:
  prometheus:
    export_addr:
      ip: 0.0.0.0
      port: 9091

  zipkin:
    endpoint: http://zipkin:9411/api/v2/spans
    sample_ratio: 1
```

### APISIX 路由配置 (通过 Admin API)

```bash
# 创建路由脚本
#!/bin/bash

ADMIN_API="http://localhost:9180"
API_KEY="edd1c9f034335f136f87ad84b625c8f1"

# 1. 创建上游 (Upstream)
curl -X PUT "${ADMIN_API}/apisix/admin/upstreams/1" \
  -H "X-API-KEY: ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "user-service-upstream",
    "type": "roundrobin",
    "scheme": "http",
    "discovery_type": "dns",
    "pass_host": "pass",
    "nodes": {
      "user-service-1:8080": 1,
      "user-service-2:8080": 1
    },
    "checks": {
      "active": {
        "type": "http",
        "http_path": "/health",
        "timeout": 1,
        "healthy": {
          "interval": 2,
          "successes": 2
        },
        "unhealthy": {
          "interval": 1,
          "http_failures": 2
        }
      },
      "passive": {
        "healthy": {
          "http_statuses": [200, 201, 202],
          "successes": 3
        },
        "unhealthy": {
          "http_statuses": [500, 502, 503, 504],
          "http_failures": 3,
          "tcp_failures": 3
        }
      }
    },
    "timeout": {
      "connect": 6,
      "send": 6,
      "read": 6
    },
    "retries": 2,
    "keepalive_pool": {
      "size": 320,
      "idle_timeout": 60,
      "requests": 1000
    }
  }'

# 2. 创建路由 (Route)
curl -X PUT "${ADMIN_API}/apisix/admin/routes/1" \
  -H "X-API-KEY: ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "user-api-route",
    "desc": "用户服务路由",
    "uri": "/api/v1/users/*",
    "methods": ["GET", "POST", "PUT", "DELETE"],
    "upstream_id": "1",
    "plugins": {
      "jwt-auth": {
        "header": "Authorization",
        "query": "jwt",
        "cookie": "jwt"
      },
      "limit-req": {
        "rate": 100,
        "burst": 50,
        "key": "remote_addr",
        "rejected_code": 429,
        "rejected_msg": "Too many requests"
      },
      "limit-count": {
        "count": 1000,
        "time_window": 60,
        "key": "remote_addr",
        "rejected_code": 429,
        "policy": "redis",
        "redis_host": "redis",
        "redis_port": 6379,
        "redis_database": 1
      },
      "cors": {
        "allow_origins": "https://example.com",
        "allow_methods": "GET,POST,PUT,DELETE",
        "allow_headers": "Authorization,Content-Type",
        "expose_headers": "X-Request-ID",
        "max_age": 3600,
        "allow_credential": true
      },
      "prometheus": {},
      "zipkin": {
        "endpoint": "http://zipkin:9411/api/v2/spans",
        "sample_ratio": 0.1
      },
      "request-id": {
        "header_name": "X-Request-ID",
        "include_in_response": true
      }
    },
    "status": 1
  }'

# 3. 创建消费者 (Consumer)
curl -X PUT "${ADMIN_API}/apisix/admin/consumers" \
  -H "X-API-KEY: ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "mobile-app",
    "desc": "移动端应用",
    "plugins": {
      "jwt-auth": {
        "key": "mobile-app-key",
        "secret": "my-secret-key",
        "algorithm": "HS256",
        "exp": 86400
      }
    }
  }'
```

### APISIX Docker Compose 部署

```yaml
# docker-compose-apisix.yml
version: '3.8'

services:
  etcd:
    image: bitnami/etcd:3.5
    container_name: apisix-etcd
    environment:
      ETCD_ENABLE_V2: "true"
      ALLOW_NONE_AUTHENTICATION: "yes"
      ETCD_ADVERTISE_CLIENT_URLS: "http://etcd:2379"
      ETCD_LISTEN_CLIENT_URLS: "http://0.0.0.0:2379"
    ports:
      - "2379:2379"
      - "2380:2380"
    volumes:
      - etcd_data:/bitnami/etcd
    networks:
      - apisix-net

  apisix:
    image: apache/apisix:3.7.0-debian
    container_name: apisix
    volumes:
      - ./config.yaml:/usr/local/apisix/conf/config.yaml:ro
      - ./apisix_log:/usr/local/apisix/logs
    ports:
      - "9080:9080"    # HTTP
      - "9443:9443"    # HTTPS
      - "9180:9180"    # Admin API
      - "9091:9091"    # Prometheus Metrics
    depends_on:
      - etcd
    networks:
      - apisix-net
    environment:
      - APISIX_STAND_ALONE=false

  apisix-dashboard:
    image: apache/apisix-dashboard:3.0.1-alpine
    container_name: apisix-dashboard
    volumes:
      - ./dashboard_conf/conf.yaml:/usr/local/apisix-dashboard/conf/conf.yaml:ro
    ports:
      - "9000:9000"
    depends_on:
      - apisix
    networks:
      - apisix-net

  prometheus:
    image: prom/prometheus:latest
    container_name: apisix-prometheus
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - apisix-net
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    container_name: apisix-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - apisix-net
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  etcd_data:
  prometheus_data:
  grafana_data:

networks:
  apisix-net:
    driver: bridge
```

---

## 性能优化

### 性能优化清单

```
┌─────────────────────────────────────────────┐
│         API 网关性能优化策略                │
├─────────────────────────────────────────────┤
│                                             │
│ 1. 连接池优化                               │
│    ├─ Keepalive 连接复用                    │
│    ├─ 连接池大小调整                        │
│    └─ 超时时间优化                          │
│                                             │
│ 2. 缓存策略                                 │
│    ├─ 响应缓存 (Proxy Cache)                │
│    ├─ 限流计数器缓存 (Redis)                │
│    └─ 配置热加载                            │
│                                             │
│ 3. 并发控制                                 │
│    ├─ Worker 进程数 = CPU 核数              │
│    ├─ Worker 连接数优化                     │
│    └─ 限流熔断                              │
│                                             │
│ 4. 网络优化                                 │
│    ├─ TCP Fast Open                         │
│    ├─ HTTP/2 启用                           │
│    └─ gRPC 支持                             │
│                                             │
│ 5. 插件优化                                 │
│    ├─ 只启用必需插件                        │
│    ├─ Lua JIT 编译                          │
│    └─ 插件执行顺序优化                      │
└─────────────────────────────────────────────┘
```

### 压测脚本

```python
# load_test.py - K6 压测脚本 (Python 版本)
from locust import HttpUser, task, between
import json

class APIGatewayUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        """获取 JWT Token"""
        response = self.client.post("/auth/login", json={
            "username": "test@example.com",
            "password": "password123"
        })

        if response.status_code == 200:
            data = response.json()
            self.token = data["token"]
        else:
            self.token = None

    @task(3)
    def get_users(self):
        """查询用户列表 - 权重 3"""
        headers = {"Authorization": f"Bearer {self.token}"}
        self.client.get("/api/v1/users", headers=headers)

    @task(2)
    def get_user_detail(self):
        """查询用户详情 - 权重 2"""
        headers = {"Authorization": f"Bearer {self.token}"}
        self.client.get("/api/v1/users/123", headers=headers)

    @task(1)
    def create_user(self):
        """创建用户 - 权重 1"""
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        payload = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "password123"
        }

        self.client.post("/api/v1/users",
                        headers=headers,
                        json=payload)

# 运行命令:
# locust -f load_test.py --host=http://localhost:9080 --users=1000 --spawn-rate=50
```

---

## 实战案例

### 案例: 电商平台 API 网关

**场景**: 日均 1000 万请求，高峰 QPS 5000+

```
              电商 API 网关架构
┌────────────────────────────────────────────┐
│            客户端层                        │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐   │
│  │  Web    │  │  iOS    │  │ Android  │   │
│  └────┬────┘  └────┬────┘  └────┬─────┘   │
└───────┼───────────┼────────────┼──────────┘
        │           │            │
        └───────────┴────────────┘
                    │
┌───────────────────▼────────────────────────┐
│        CDN (CloudFlare)                    │
│        WAF + DDoS 防护                     │
└───────────────────┬────────────────────────┘
                    │
┌───────────────────▼────────────────────────┐
│      负载均衡 (ALB)                        │
└───────────────────┬────────────────────────┘
                    │
        ┌───────────┴────────────┐
        │                        │
┌───────▼──────┐       ┌────────▼─────┐
│  APISIX (主)  │       │ APISIX (备)  │
│  ┌─────────┐ │       │  ┌─────────┐ │
│  │ Plugins │ │       │  │ Plugins │ │
│  │ ┌─────┐ │ │       │  │ ┌─────┐ │ │
│  │ │JWT  │ │ │       │  │ │JWT  │ │ │
│  │ │限流 │ │ │       │  │ │限流 │ │ │
│  │ │日志 │ │ │       │  │ │日志 │ │ │
│  │ └─────┘ │ │       │  │ └─────┘ │ │
│  └─────────┘ │       │  └─────────┘ │
└──────┬───────┘       └──────────────┘
       │
       │          ┌─────────┐
       ├─────────▶│  Redis  │ (限流、缓存)
       │          └─────────┘
       │
┌──────┴──────────────────────────────┐
│                                     │
│  微服务集群                         │
│  ┌─────────┐  ┌─────────┐          │
│  │ 用户服务│  │ 商品服务│  ...     │
│  │ (50实例)│  │ (80实例)│          │
│  └─────────┘  └─────────┘          │
└─────────────────────────────────────┘

配置要点:
- APISIX Worker 进程: 16 (32核CPU)
- Keepalive 连接池: 1000
- 限流: Redis 集群 (5主5从)
- 监控: Prometheus + Grafana
- 日志: Kafka + ELK
```

### 监控配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'apisix'
    static_configs:
      - targets: ['apisix:9091']
    metrics_path: '/apisix/prometheus/metrics'

  - job_name: 'kong'
    static_configs:
      - targets: ['kong:8001']
    metrics_path: '/metrics'
```

---

## 总结

### API 网关选型对比

```
┌──────────────────────────────────────────────┐
│          Kong vs APISIX 对比                 │
├──────────┬───────────────┬──────────────────┤
│  特性    │     Kong      │      APISIX      │
├──────────┼───────────────┼──────────────────┤
│ 性能     │ 高 (20K RPS)  │ 极高 (50K+ RPS)  │
│ 配置方式 │ Admin API     │ Admin API + etcd │
│ 配置同步 │ PostgreSQL    │ etcd (毫秒级)    │
│ 插件语言 │ Lua           │ Lua + 多语言     │
│ 学习曲线 │ 中等          │ 较陡             │
│ 社区     │ 成熟          │ 快速增长         │
│ Dashboard│ Konga (社区)  │ 官方支持         │
│ 适用场景 │ 中小型企业    │ 高性能、大规模   │
└──────────┴───────────────┴──────────────────┘
```

### 关键要点

1. **统一入口**: API 网关是微服务的唯一入口
2. **安全第一**: JWT/OAuth2 认证 + 限流防护
3. **高可用**: 多实例部署 + 健康检查
4. **可观测**: 日志、指标、链路追踪
5. **性能优化**: 连接池、缓存、异步处理

### 下一步学习

- [02_routing_strategies.md](02_routing_strategies.md) - 路由策略与灰度发布
- [03_authentication.md](03_authentication.md) - 认证授权深入
- [04_gateway_comparison.md](04_gateway_comparison.md) - 网关对比与选型
