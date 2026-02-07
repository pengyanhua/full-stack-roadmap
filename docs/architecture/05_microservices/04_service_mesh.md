# Service Mesh

## 一、Service Mesh 概述

### 什么是 Service Mesh？

```
┌─────────────────────────────────────────────────────────────────┐
│                   Service Mesh 架构演进                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   阶段 1: 服务直接通信                                           │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   ┌─────────┐              ┌─────────┐                  │  │
│   │   │Service A│─────────────▶│Service B│                  │  │
│   │   │ 重试    │              │         │                  │  │
│   │   │ 超时    │              │         │                  │  │
│   │   │ 熔断    │              │         │                  │  │
│   │   └─────────┘              └─────────┘                  │  │
│   │                                                          │  │
│   │   问题: 每个服务都要实现通信逻辑 (侵入式)                  │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   阶段 2: 公共库                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   ┌─────────┐              ┌─────────┐                  │  │
│   │   │Service A│─────────────▶│Service B│                  │  │
│   │   │         │              │         │                  │  │
│   │   │ ┌─────┐ │              │ ┌─────┐ │                  │  │
│   │   │ │ SDK │ │              │ │ SDK │ │                  │  │
│   │   │ └─────┘ │              │ └─────┘ │                  │  │
│   │   └─────────┘              └─────────┘                  │  │
│   │                                                          │  │
│   │   问题: 多语言支持困难、升级需要重新部署                   │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   阶段 3: Service Mesh (Sidecar 模式)                           │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   ┌─────────────────┐      ┌─────────────────┐          │  │
│   │   │      Pod A       │      │      Pod B       │          │  │
│   │   │ ┌─────┐ ┌─────┐ │      │ ┌─────┐ ┌─────┐ │          │  │
│   │   │ │Svc A│ │Proxy│ │─────▶│ │Proxy│ │Svc B│ │          │  │
│   │   │ └─────┘ └─────┘ │      │ └─────┘ └─────┘ │          │  │
│   │   └─────────────────┘      └─────────────────┘          │  │
│   │                                                          │  │
│   │   优点:                                                   │  │
│   │   • 非侵入式 (应用无感知)                                 │  │
│   │   • 语言无关                                             │  │
│   │   • 统一治理                                             │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Service Mesh 核心能力

```
┌─────────────────────────────────────────────────────────────────┐
│                  Service Mesh 功能矩阵                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   流量管理                        可观测性                       │
│   ─────────────────              ─────────────────              │
│   • 负载均衡                     • 分布式追踪                    │
│   • 流量路由                     • 指标收集                      │
│   • 流量镜像                     • 访问日志                      │
│   • 灰度发布                     • 健康检查                      │
│   • A/B 测试                                                    │
│                                                                 │
│   安全                           弹性                           │
│   ─────────────────              ─────────────────              │
│   • mTLS 加密                    • 超时控制                      │
│   • 身份认证                     • 重试策略                      │
│   • 访问控制                     • 熔断降级                      │
│   • 策略执行                     • 限流                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、Istio 实战

### 1. Istio 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      Istio 架构                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                     ┌─────────────────────┐                     │
│                     │      Istiod         │                     │
│                     │  (控制平面)          │                     │
│                     │  ┌───────────────┐  │                     │
│                     │  │ Pilot (配置)  │  │                     │
│                     │  │ Citadel (安全)│  │                     │
│                     │  │ Galley (验证) │  │                     │
│                     │  └───────────────┘  │                     │
│                     └──────────┬──────────┘                     │
│                                │ xDS API                        │
│         ┌──────────────────────┼──────────────────────┐         │
│         ▼                      ▼                      ▼         │
│   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐  │
│   │    Pod A    │       │    Pod B    │       │    Pod C    │  │
│   │ ┌─────────┐ │       │ ┌─────────┐ │       │ ┌─────────┐ │  │
│   │ │ Service │ │       │ │ Service │ │       │ │ Service │ │  │
│   │ └────┬────┘ │       │ └────┬────┘ │       │ └────┬────┘ │  │
│   │      │      │       │      │      │       │      │      │  │
│   │ ┌────▼────┐ │       │ ┌────▼────┐ │       │ ┌────▼────┐ │  │
│   │ │  Envoy  │◀┼───────┼▶│  Envoy  │◀┼───────┼▶│  Envoy  │ │  │
│   │ │(Sidecar)│ │       │ │(Sidecar)│ │       │ │(Sidecar)│ │  │
│   │ └─────────┘ │       │ └─────────┘ │       │ └─────────┘ │  │
│   └─────────────┘       └─────────────┘       └─────────────┘  │
│                           数据平面                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 流量管理

```yaml
# VirtualService - 流量路由
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: reviews-route
spec:
  hosts:
    - reviews
  http:
    # 灰度发布: 10% 流量到 v2
    - match:
        - headers:
            end-user:
              exact: "test-user"  # 测试用户走新版本
      route:
        - destination:
            host: reviews
            subset: v2
    - route:
        - destination:
            host: reviews
            subset: v1
          weight: 90
        - destination:
            host: reviews
            subset: v2
          weight: 10

---
# DestinationRule - 目标规则
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: reviews-destination
spec:
  host: reviews
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        h2UpgradePolicy: UPGRADE
        http1MaxPendingRequests: 100
        http2MaxRequests: 1000
    loadBalancer:
      simple: ROUND_ROBIN
    outlierDetection:  # 熔断配置
      consecutive5xxErrors: 5
      interval: 5s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
  subsets:
    - name: v1
      labels:
        version: v1
    - name: v2
      labels:
        version: v2
```

### 3. 弹性配置

```yaml
# 超时和重试
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ratings
spec:
  hosts:
    - ratings
  http:
    - route:
        - destination:
            host: ratings
      timeout: 10s  # 超时时间
      retries:
        attempts: 3          # 重试次数
        perTryTimeout: 2s    # 每次重试超时
        retryOn: 5xx,reset,connect-failure,retriable-4xx

---
# 熔断
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: ratings
spec:
  host: ratings
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 1
        http2MaxRequests: 100
        maxRequestsPerConnection: 10
    outlierDetection:
      consecutiveGatewayErrors: 5
      consecutive5xxErrors: 5
      interval: 10s
      baseEjectionTime: 30s
      maxEjectionPercent: 100
```

### 4. 安全配置

```yaml
# mTLS 配置
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: istio-system
spec:
  mtls:
    mode: STRICT  # 强制 mTLS

---
# 授权策略
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: reviews-viewer
  namespace: default
spec:
  selector:
    matchLabels:
      app: reviews
  action: ALLOW
  rules:
    - from:
        - source:
            principals: ["cluster.local/ns/default/sa/productpage"]
      to:
        - operation:
            methods: ["GET"]
```

---

## 三、Envoy 配置

### 基础配置

```yaml
# envoy.yaml
static_resources:
  listeners:
    - name: listener_0
      address:
        socket_address:
          address: 0.0.0.0
          port_value: 10000
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                stat_prefix: ingress_http
                route_config:
                  name: local_route
                  virtual_hosts:
                    - name: backend
                      domains: ["*"]
                      routes:
                        - match:
                            prefix: "/api/v1"
                          route:
                            cluster: service_backend
                            timeout: 30s
                            retry_policy:
                              retry_on: "5xx,reset,connect-failure"
                              num_retries: 3
                http_filters:
                  - name: envoy.filters.http.router
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router

  clusters:
    - name: service_backend
      connect_timeout: 5s
      type: STRICT_DNS
      lb_policy: ROUND_ROBIN
      health_checks:
        - timeout: 5s
          interval: 10s
          unhealthy_threshold: 3
          healthy_threshold: 2
          http_health_check:
            path: "/health"
      load_assignment:
        cluster_name: service_backend
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address:
                      address: backend-service
                      port_value: 8080
```

---

## 四、何时使用 Service Mesh？

```
┌─────────────────────────────────────────────────────────────────┐
│                 Service Mesh 适用场景评估                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ✅ 适合使用                                                    │
│   ─────────────────────────────────────────────────────────    │
│   □ 微服务数量 > 20 个                                          │
│   □ 多语言技术栈                                                │
│   □ 需要细粒度流量控制                                          │
│   □ 有零信任安全需求                                            │
│   □ 需要统一的可观测性                                          │
│   □ 已经使用 Kubernetes                                         │
│                                                                 │
│   ❌ 不适合使用                                                  │
│   ─────────────────────────────────────────────────────────    │
│   □ 微服务数量 < 10 个                                          │
│   □ 单一语言技术栈 (已有成熟 SDK)                               │
│   □ 对延迟极度敏感 (Sidecar 增加 ~1ms)                          │
│   □ 团队没有 K8s 运维经验                                       │
│   □ 资源有限 (Sidecar 占用内存 ~50MB)                           │
│                                                                 │
│   复杂度权衡:                                                    │
│   ─────────────────────────────────────────────────────────    │
│   Service Mesh 解决的问题:  运维复杂度 ↓  开发复杂度 ↓           │
│   Service Mesh 带来的问题:  架构复杂度 ↑  调试难度 ↑             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五、检查清单

### 引入前评估

- [ ] 微服务数量是否足够多？
- [ ] 是否有多语言技术栈？
- [ ] 团队是否熟悉 Kubernetes？
- [ ] 是否有足够的资源开销？

### 实施检查

- [ ] 是否配置了合理的重试策略？
- [ ] 是否配置了熔断保护？
- [ ] 是否开启了 mTLS？
- [ ] 是否接入了可观测性组件？
