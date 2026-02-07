# 路由策略与负载均衡

## 目录
- [路由策略概述](#路由策略概述)
- [负载均衡算法](#负载均衡算法)
- [灰度发布](#灰度发布)
- [流量控制](#流量控制)
- [动态路由](#动态路由)
- [实战案例](#实战案例)

---

## 路由策略概述

### 路由匹配规则

```
        API 网关路由匹配流程
┌─────────────────────────────────────┐
│  请求: GET /api/v1/users/123?page=1 │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  1. 精确匹配 (Exact Match)           │
│     /api/v1/users/123  ✗ 不匹配      │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  2. 前缀匹配 (Prefix Match)          │
│     /api/v1/users  ✓ 匹配            │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  3. 正则匹配 (Regex Match)           │
│     ^/api/v[0-9]+/users/\d+  ✓ 匹配 │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  4. 主机匹配 (Host Match)            │
│     api.example.com  ✓ 匹配          │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  5. 方法匹配 (Method Match)          │
│     GET  ✓ 匹配                      │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  6. Header 匹配                      │
│     X-API-Version: v1  ✓ 匹配        │
└──────────────┬───────────────────────┘
               │
               ▼
         转发到目标服务
```

### Kong 高级路由配置

```yaml
# Kong 路由配置示例
services:
  - name: user-service-v1
    url: http://user-service-v1:8080

    routes:
      # 1. 基于路径的路由
      - name: user-api-v1
        paths:
          - /api/v1/users
        strip_path: false
        preserve_host: false
        protocols:
          - http
          - https

      # 2. 基于 Header 的路由
      - name: user-api-mobile
        paths:
          - /api/users
        headers:
          X-Client-Type:
            - mobile
        strip_path: true

      # 3. 基于正则表达式的路由
      - name: user-api-regex
        paths:
          - ~/api/v[0-9]+/users/[0-9]+$
        regex_priority: 10

      # 4. 基于查询参数的路由
      - name: user-api-beta
        paths:
          - /api/users
        headers:
          X-Beta-User:
            - "true"

  - name: user-service-v2
    url: http://user-service-v2:8080

    routes:
      # 版本 2 路由
      - name: user-api-v2
        paths:
          - /api/v2/users
        strip_path: false
```

### APISIX 路由配置

```bash
#!/bin/bash
# APISIX 复杂路由配置

ADMIN_API="http://localhost:9180"
API_KEY="your-api-key"

# 1. 基于 URI 参数的路由
curl -X PUT "${ADMIN_API}/apisix/admin/routes/1" \
  -H "X-API-KEY: ${API_KEY}" \
  -d '{
    "uri": "/api/v1/users/*",
    "methods": ["GET", "POST"],
    "vars": [
      ["arg_version", "==", "v1"]
    ],
    "upstream": {
      "type": "roundrobin",
      "nodes": {
        "user-service-v1:8080": 1
      }
    }
  }'

# 2. 基于 Cookie 的路由 (A/B 测试)
curl -X PUT "${ADMIN_API}/apisix/admin/routes/2" \
  -H "X-API-KEY: ${API_KEY}" \
  -d '{
    "uri": "/api/v1/products/*",
    "methods": ["GET"],
    "vars": [
      ["cookie_ab_test", "==", "group_b"]
    ],
    "upstream": {
      "type": "roundrobin",
      "nodes": {
        "product-service-new:8080": 1
      }
    }
  }'

# 3. 基于请求头的路由 (移动端优化)
curl -X PUT "${ADMIN_API}/apisix/admin/routes/3" \
  -H "X-API-KEY: ${API_KEY}" \
  -d '{
    "uri": "/api/v1/orders/*",
    "methods": ["GET", "POST"],
    "vars": [
      ["http_user_agent", "~~", ".*Mobile.*"]
    ],
    "upstream": {
      "type": "roundrobin",
      "nodes": {
        "order-service-mobile:8080": 1
      }
    },
    "timeout": {
      "connect": 3,
      "send": 3,
      "read": 10
    }
  }'

# 4. 基于地理位置的路由
curl -X PUT "${ADMIN_API}/apisix/admin/routes/4" \
  -H "X-API-KEY: ${API_KEY}" \
  -d '{
    "uri": "/api/v1/content/*",
    "vars": [
      ["http_x_country_code", "in", ["CN", "HK", "TW"]]
    ],
    "upstream": {
      "type": "roundrobin",
      "nodes": {
        "content-service-asia:8080": 1
      }
    }
  }'
```

---

## 负载均衡算法

### 常见负载均衡算法

```
┌─────────────────────────────────────────────┐
│         负载均衡算法对比                    │
├──────────────┬──────────────┬───────────────┤
│  算法        │  特点        │  适用场景     │
├──────────────┼──────────────┼───────────────┤
│ Round Robin  │ 轮询         │ 服务器性能均等│
│              │ 简单公平     │ 无状态服务    │
│              │              │               │
│ Weighted RR  │ 加权轮询     │ 服务器性能不同│
│              │ 按权重分配   │ 灰度发布      │
│              │              │               │
│ Least Conn   │ 最少连接     │ 长连接场景    │
│              │ 动态均衡     │ WebSocket     │
│              │              │               │
│ IP Hash      │ IP 哈希      │ 会话保持      │
│              │ 同一客户端   │ 有状态服务    │
│              │ 固定服务器   │               │
│              │              │               │
│ Consistent   │ 一致性哈希   │ 缓存场景      │
│ Hash         │ 节点变化影响小│ 分布式缓存   │
│              │              │               │
│ Random       │ 随机选择     │ 无特殊要求    │
│              │ 简单快速     │ 快速测试      │
└──────────────┴──────────────┴───────────────┘
```

### 负载均衡算法图解

```
1. Round Robin (轮询)
┌────────────────────────────────┐
│  请求序列: 1  2  3  4  5  6    │
│           │  │  │  │  │  │     │
│           ▼  ▼  ▼  ▼  ▼  ▼     │
│  Server A  ●     ●     ●        │
│  Server B     ●     ●     ●     │
│                                 │
│  特点: 均匀分配                 │
└────────────────────────────────┘

2. Weighted Round Robin (加权轮询)
┌────────────────────────────────┐
│  Server A (Weight: 3)           │
│  Server B (Weight: 1)           │
│                                 │
│  请求: 1  2  3  4  5  6  7  8   │
│        │  │  │  │  │  │  │  │   │
│        ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼   │
│  A     ●  ●  ●     ●  ●  ●      │
│  B              ●           ●    │
│                                 │
│  A:B = 3:1                      │
└────────────────────────────────┘

3. Least Connections (最少连接)
┌────────────────────────────────┐
│  当前连接数:                    │
│  Server A: 5 connections        │
│  Server B: 3 connections ← 选中 │
│  Server C: 7 connections        │
│                                 │
│  新请求 → 分配到 Server B       │
└────────────────────────────────┘

4. IP Hash (IP 哈希)
┌────────────────────────────────┐
│  Client IP: 192.168.1.100       │
│  Hash(192.168.1.100) = 12345    │
│  12345 % 3 = 0 → Server A       │
│                                 │
│  同一 IP 始终路由到同一服务器   │
└────────────────────────────────┘

5. Consistent Hash (一致性哈希)
┌────────────────────────────────┐
│         Hash Ring               │
│                                 │
│       Server A (0°)             │
│          │                      │
│    ┌─────┼─────┐                │
│    │     │     │                │
│  S C   Key1   S B               │
│  (270°)       (120°)            │
│    │           │                │
│    └───────────┘                │
│                                 │
│  Key1 → 顺时针找到 Server B     │
│  新增 Server D 只影响部分 Key   │
└────────────────────────────────┘
```

### Kong 负载均衡配置

```yaml
# Kong Upstream 配置
upstreams:
  # 1. 轮询负载均衡
  - name: user-service-roundrobin
    algorithm: round-robin
    hash_on: none
    hash_fallback: none

    targets:
      - target: user-service-1:8080
        weight: 100
      - target: user-service-2:8080
        weight: 100
      - target: user-service-3:8080
        weight: 100

  # 2. 加权负载均衡 (灰度发布)
  - name: user-service-weighted
    algorithm: round-robin

    targets:
      - target: user-service-stable:8080
        weight: 90   # 90% 流量
      - target: user-service-canary:8080
        weight: 10   # 10% 流量

  # 3. 一致性哈希 (基于 Cookie)
  - name: user-service-consistent-hash
    algorithm: consistent-hashing
    hash_on: cookie
    hash_on_cookie: session_id
    hash_fallback: ip

    targets:
      - target: user-service-1:8080
        weight: 100
      - target: user-service-2:8080
        weight: 100

  # 4. 最少连接
  - name: user-service-least-connections
    algorithm: least-connections

    targets:
      - target: user-service-1:8080
      - target: user-service-2:8080

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
          http_statuses: [200, 201, 202, 204]
          successes: 5
        unhealthy:
          http_statuses: [500, 502, 503, 504]
          http_failures: 3
          timeouts: 2
```

### APISIX 负载均衡 Lua 脚本

```lua
-- custom_balancer.lua - 自定义负载均衡器
local balancer = require("ngx.balancer")
local core = require("apisix.core")

local _M = {}

-- 带权重的随机选择
function _M.weighted_random(nodes)
    local total_weight = 0
    local weighted_nodes = {}

    -- 计算总权重
    for addr, weight in pairs(nodes) do
        total_weight = total_weight + weight
        table.insert(weighted_nodes, {
            addr = addr,
            weight = weight,
            cumulative = total_weight
        })
    end

    -- 随机选择
    local random_value = math.random(1, total_weight)

    for _, node in ipairs(weighted_nodes) do
        if random_value <= node.cumulative then
            return node.addr
        end
    end

    return weighted_nodes[1].addr
end

-- 基于响应时间的负载均衡
local response_times = {}

function _M.least_response_time(nodes)
    local min_time = math.huge
    local selected_node = nil

    for addr, weight in pairs(nodes) do
        local avg_time = response_times[addr] or 0

        if avg_time < min_time then
            min_time = avg_time
            selected_node = addr
        end
    end

    return selected_node or next(nodes)
end

-- 记录响应时间
function _M.record_response_time(addr, time)
    if not response_times[addr] then
        response_times[addr] = time
    else
        -- 指数移动平均
        response_times[addr] = response_times[addr] * 0.7 + time * 0.3
    end
end

return _M
```

---

## 灰度发布

### 灰度发布策略

```
            灰度发布演进过程
┌─────────────────────────────────────┐
│  阶段 1: 初始状态 (100% 稳定版)    │
│  ┌─────────────────────────────┐   │
│  │  用户流量                   │   │
│  │  100%  │                    │   │
│  │        ▼                    │   │
│  │   ┌──────────┐              │   │
│  │   │ v1.0     │              │   │
│  │   │ (稳定版) │              │   │
│  │   └──────────┘              │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  阶段 2: 金丝雀发布 (5% 灰度)      │
│  ┌─────────────────────────────┐   │
│  │  用户流量                   │   │
│  │   95%  │  5%                │   │
│  │        ▼   ▼                │   │
│  │   ┌──────────┐  ┌────────┐ │   │
│  │   │ v1.0     │  │ v2.0   │ │   │
│  │   │ (稳定版) │  │(灰度版)│ │   │
│  │   └──────────┘  └────────┘ │   │
│  │                  观察指标   │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  阶段 3: 逐步扩大 (50% 灰度)       │
│  ┌─────────────────────────────┐   │
│  │  用户流量                   │   │
│  │   50%  │  50%               │   │
│  │        ▼   ▼                │   │
│  │   ┌──────────┐  ┌────────┐ │   │
│  │   │ v1.0     │  │ v2.0   │ │   │
│  │   │          │  │        │ │   │
│  │   └──────────┘  └────────┘ │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  阶段 4: 全量发布 (100% 新版)      │
│  ┌─────────────────────────────┐   │
│  │  用户流量                   │   │
│  │       100% │                │   │
│  │            ▼                │   │
│  │       ┌────────┐            │   │
│  │       │ v2.0   │            │   │
│  │       │ (新版) │            │   │
│  │       └────────┘            │   │
│  │   (v1.0 已下线)             │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### Kong 灰度发布配置

```bash
#!/bin/bash
# Kong 灰度发布脚本

ADMIN_API="http://localhost:8001"

# 1. 创建 Upstream (初始: 100% 稳定版)
curl -X POST ${ADMIN_API}/upstreams \
  -d "name=user-service-canary" \
  -d "algorithm=round-robin"

# 添加稳定版目标
curl -X POST ${ADMIN_API}/upstreams/user-service-canary/targets \
  -d "target=user-service-v1:8080" \
  -d "weight=1000"

# 2. 开始灰度: 添加新版本 (5% 流量)
curl -X POST ${ADMIN_API}/upstreams/user-service-canary/targets \
  -d "target=user-service-v2:8080" \
  -d "weight=50"   # 50/(1000+50) ≈ 5%

# 3. 观察指标后,逐步增加权重到 20%
curl -X PATCH ${ADMIN_API}/upstreams/user-service-canary/targets/user-service-v2:8080 \
  -d "weight=250"  # 250/(1000+250) = 20%

# 4. 继续增加到 50%
curl -X PATCH ${ADMIN_API}/upstreams/user-service-canary/targets/user-service-v2:8080 \
  -d "weight=1000" # 1000/(1000+1000) = 50%

# 5. 全量切换到新版本
curl -X PATCH ${ADMIN_API}/upstreams/user-service-canary/targets/user-service-v1:8080 \
  -d "weight=0"    # 关闭旧版本流量

# 6. 清理旧版本
curl -X DELETE ${ADMIN_API}/upstreams/user-service-canary/targets/user-service-v1:8080
```

### APISIX 灰度发布 (基于用户属性)

```bash
#!/bin/bash
# APISIX 精细化灰度发布

ADMIN_API="http://localhost:9180"
API_KEY="your-api-key"

# 1. 创建 v1 路由 (默认稳定版)
curl -X PUT "${ADMIN_API}/apisix/admin/routes/user-api-v1" \
  -H "X-API-KEY: ${API_KEY}" \
  -d '{
    "uri": "/api/v1/users/*",
    "priority": 1,
    "upstream": {
      "type": "roundrobin",
      "nodes": {
        "user-service-v1:8080": 1
      }
    }
  }'

# 2. 创建 v2 路由 (灰度版 - 内部员工)
curl -X PUT "${ADMIN_API}/apisix/admin/routes/user-api-v2-internal" \
  -H "X-API-KEY: ${API_KEY}" \
  -d '{
    "uri": "/api/v1/users/*",
    "priority": 10,
    "vars": [
      ["http_x_user_type", "==", "internal"]
    ],
    "upstream": {
      "type": "roundrobin",
      "nodes": {
        "user-service-v2:8080": 1
      }
    }
  }'

# 3. 创建 v2 路由 (灰度版 - Beta 用户)
curl -X PUT "${ADMIN_API}/apisix/admin/routes/user-api-v2-beta" \
  -H "X-API-KEY: ${API_KEY}" \
  -d '{
    "uri": "/api/v1/users/*",
    "priority": 9,
    "vars": [
      ["cookie_beta_user", "==", "true"]
    ],
    "upstream": {
      "type": "roundrobin",
      "nodes": {
        "user-service-v2:8080": 1
      }
    }
  }'

# 4. 创建 v2 路由 (灰度版 - 5% 随机流量)
curl -X PUT "${ADMIN_API}/apisix/admin/routes/user-api-v2-random" \
  -H "X-API-KEY: ${API_KEY}" \
  -d '{
    "uri": "/api/v1/users/*",
    "priority": 5,
    "plugins": {
      "traffic-split": {
        "rules": [
          {
            "weighted_upstreams": [
              {
                "upstream": {
                  "type": "roundrobin",
                  "nodes": {
                    "user-service-v2:8080": 1
                  }
                },
                "weight": 5
              },
              {
                "weight": 95
              }
            ]
          }
        ]
      }
    },
    "upstream": {
      "type": "roundrobin",
      "nodes": {
        "user-service-v1:8080": 1
      }
    }
  }'
```

### 自动化灰度发布脚本

```python
# canary_deploy.py - 自动化灰度发布
import requests
import time
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CanaryDeployment:
    def __init__(self, admin_api: str, api_key: str):
        self.admin_api = admin_api
        self.headers = {"X-API-KEY": api_key}
        self.prometheus_url = "http://prometheus:9090"

    def get_error_rate(self, service: str) -> float:
        """从 Prometheus 获取错误率"""
        query = f'rate(http_requests_total{{service="{service}",status=~"5.."}}[5m])'

        response = requests.get(
            f"{self.prometheus_url}/api/v1/query",
            params={"query": query}
        )

        if response.status_code == 200:
            result = response.json()["data"]["result"]
            if result:
                return float(result[0]["value"][1])

        return 0.0

    def get_latency_p99(self, service: str) -> float:
        """获取 P99 延迟"""
        query = f'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{{service="{service}"}}[5m]))'

        response = requests.get(
            f"{self.prometheus_url}/api/v1/query",
            params={"query": query}
        )

        if response.status_code == 200:
            result = response.json()["data"]["result"]
            if result:
                return float(result[0]["value"][1])

        return 0.0

    def update_traffic_weight(self, route_id: str, canary_weight: int):
        """更新灰度流量权重"""
        config = {
            "plugins": {
                "traffic-split": {
                    "rules": [{
                        "weighted_upstreams": [
                            {
                                "upstream": {
                                    "type": "roundrobin",
                                    "nodes": {"user-service-v2:8080": 1}
                                },
                                "weight": canary_weight
                            },
                            {
                                "weight": 100 - canary_weight
                            }
                        ]
                    }]
                }
            }
        }

        response = requests.patch(
            f"{self.admin_api}/apisix/admin/routes/{route_id}",
            headers=self.headers,
            json=config
        )

        if response.status_code == 200:
            logger.info(f"Updated canary weight to {canary_weight}%")
            return True
        else:
            logger.error(f"Failed to update weight: {response.text}")
            return False

    def check_health(self, service: str) -> bool:
        """健康检查"""
        error_rate = self.get_error_rate(service)
        latency_p99 = self.get_latency_p99(service)

        logger.info(f"{service} - Error Rate: {error_rate:.4f}, P99 Latency: {latency_p99:.3f}s")

        # 健康标准
        if error_rate > 0.01:  # 错误率 > 1%
            logger.warning(f"{service} error rate too high: {error_rate:.2%}")
            return False

        if latency_p99 > 1.0:  # P99 延迟 > 1s
            logger.warning(f"{service} latency too high: {latency_p99:.3f}s")
            return False

        return True

    def gradual_rollout(self, route_id: str, stages: list):
        """渐进式灰度发布"""
        for stage in stages:
            weight = stage["weight"]
            duration = stage["duration"]

            logger.info(f"Starting stage: {weight}% traffic for {duration}s")

            # 更新权重
            if not self.update_traffic_weight(route_id, weight):
                logger.error("Failed to update traffic weight, rolling back")
                self.rollback(route_id)
                return False

            # 观察期
            time.sleep(duration)

            # 健康检查
            if not self.check_health("user-service-v2"):
                logger.error("Health check failed, rolling back")
                self.rollback(route_id)
                return False

        logger.info("Canary deployment completed successfully!")
        return True

    def rollback(self, route_id: str):
        """回滚到稳定版本"""
        logger.warning("Rolling back to stable version")
        self.update_traffic_weight(route_id, 0)

if __name__ == "__main__":
    deployer = CanaryDeployment(
        admin_api="http://localhost:9180",
        api_key="your-api-key"
    )

    # 灰度发布阶段配置
    stages = [
        {"weight": 5, "duration": 300},    # 5% 流量,观察 5 分钟
        {"weight": 10, "duration": 300},   # 10% 流量,观察 5 分钟
        {"weight": 25, "duration": 600},   # 25% 流量,观察 10 分钟
        {"weight": 50, "duration": 600},   # 50% 流量,观察 10 分钟
        {"weight": 100, "duration": 0}     # 100% 全量
    ]

    deployer.gradual_rollout("user-api-canary", stages)
```

---

## 流量控制

### 限流策略

```
        多层限流架构
┌──────────────────────────────┐
│  Layer 1: CDN/WAF 限流       │
│  ├─ 防 DDoS                  │
│  └─ IP 黑名单                │
└──────────┬───────────────────┘
           │
┌──────────▼───────────────────┐
│  Layer 2: API 网关限流       │
│  ├─ 全局限流 (10000 QPS)    │
│  ├─ 服务限流 (1000 QPS/服务)│
│  └─ 用户限流 (100 QPS/用户) │
└──────────┬───────────────────┘
           │
┌──────────▼───────────────────┐
│  Layer 3: 应用层限流         │
│  ├─ 接口限流                 │
│  └─ 资源限流                 │
└──────────────────────────────┘
```

### Kong 限流配置

```yaml
plugins:
  # 1. 全局限流 (令牌桶算法)
  - name: rate-limiting
    config:
      second: 100
      minute: 5000
      hour: 200000
      policy: redis
      redis_host: redis
      redis_port: 6379
      redis_database: 0
      fault_tolerant: true
      hide_client_headers: false

  # 2. 消费者级别限流
  - name: rate-limiting
    consumer: mobile-app
    config:
      minute: 1000
      hour: 50000
      policy: cluster

  # 3. 并发限流 (漏桶算法)
  - name: rate-limiting
    config:
      limit_by: ip
      policy: local
      second: 10
      fault_tolerant: false
```

### APISIX 限流插件

```lua
-- limit-req-custom.lua - 自定义限流插件
local core = require("apisix.core")
local plugin_name = "limit-req-custom"

local schema = {
    type = "object",
    properties = {
        rate = {type = "number", minimum = 0},
        burst = {type = "number", minimum = 0},
        key = {type = "string"},
        rejected_code = {type = "integer", minimum = 200, default = 429}
    },
    required = {"rate", "burst", "key"}
}

local _M = {
    version = 0.1,
    priority = 1001,
    name = plugin_name,
    schema = schema
}

function _M.check_schema(conf)
    return core.schema.check(schema, conf)
end

function _M.access(conf, ctx)
    local key = conf.key
    local rate = conf.rate
    local burst = conf.burst

    -- 从请求中获取限流键值
    local limit_key
    if key == "remote_addr" then
        limit_key = ctx.var.remote_addr
    elseif key == "consumer_name" then
        limit_key = ctx.consumer_name
    elseif key:sub(1, 5) == "http_" then
        limit_key = ctx.var[key]
    end

    if not limit_key then
        core.log.error("failed to fetch limit key")
        return 500
    end

    local key_name = plugin_name .. ":" .. limit_key

    -- 使用 Redis 实现限流
    local red = redis:new()
    local ok, err = red:connect("redis", 6379)
    if not ok then
        core.log.error("failed to connect redis: ", err)
        return
    end

    -- 令牌桶算法
    local current_time = ngx.now()
    local capacity = burst
    local fill_rate = rate

    local last_time, err = red:get(key_name .. ":time")
    if not last_time or last_time == ngx.null then
        last_time = current_time
    else
        last_time = tonumber(last_time)
    end

    local tokens, err = red:get(key_name .. ":tokens")
    if not tokens or tokens == ngx.null then
        tokens = capacity
    else
        tokens = tonumber(tokens)
    end

    -- 计算新增令牌
    local time_passed = current_time - last_time
    tokens = math.min(capacity, tokens + time_passed * fill_rate)

    if tokens < 1 then
        -- 触发限流
        core.response.set_header("X-RateLimit-Limit", rate)
        core.response.set_header("X-RateLimit-Remaining", 0)
        core.response.set_header("X-RateLimit-Reset", math.ceil(1 / fill_rate))

        return conf.rejected_code, {
            error = "Too many requests",
            retry_after = math.ceil(1 / fill_rate)
        }
    end

    -- 消耗令牌
    tokens = tokens - 1

    -- 更新 Redis
    red:set(key_name .. ":tokens", tokens)
    red:set(key_name .. ":time", current_time)
    red:expire(key_name .. ":tokens", 3600)
    red:expire(key_name .. ":time", 3600)

    -- 设置响应头
    core.response.set_header("X-RateLimit-Limit", rate)
    core.response.set_header("X-RateLimit-Remaining", math.floor(tokens))

    red:set_keepalive(10000, 100)
end

return _M
```

---

## 动态路由

### 基于服务发现的动态路由

```go
// service_discovery.go - 基于 Consul 的动态路由
package main

import (
    "fmt"
    "log"
    "time"

    consulapi "github.com/hashicorp/consul/api"
)

type ServiceDiscovery struct {
    client *consulapi.Client
}

func NewServiceDiscovery(consulAddr string) (*ServiceDiscovery, error) {
    config := consulapi.DefaultConfig()
    config.Address = consulAddr

    client, err := consulapi.NewClient(config)
    if err != nil {
        return nil, err
    }

    return &ServiceDiscovery{client: client}, nil
}

// 注册服务
func (sd *ServiceDiscovery) RegisterService(name, addr string, port int) error {
    registration := &consulapi.AgentServiceRegistration{
        ID:      fmt.Sprintf("%s-%s-%d", name, addr, port),
        Name:    name,
        Address: addr,
        Port:    port,
        Check: &consulapi.AgentServiceCheck{
            HTTP:     fmt.Sprintf("http://%s:%d/health", addr, port),
            Interval: "10s",
            Timeout:  "5s",
        },
        Tags: []string{"v1", "production"},
    }

    return sd.client.Agent().ServiceRegister(registration)
}

// 发现服务
func (sd *ServiceDiscovery) DiscoverService(name string) ([]string, error) {
    services, _, err := sd.client.Health().Service(name, "", true, nil)
    if err != nil {
        return nil, err
    }

    var endpoints []string
    for _, service := range services {
        endpoint := fmt.Sprintf("%s:%d",
            service.Service.Address,
            service.Service.Port)
        endpoints = append(endpoints, endpoint)
    }

    return endpoints, nil
}

// 监听服务变化
func (sd *ServiceDiscovery) WatchService(name string, callback func([]string)) {
    var lastIndex uint64

    for {
        services, meta, err := sd.client.Health().Service(
            name, "", true,
            &consulapi.QueryOptions{WaitIndex: lastIndex})

        if err != nil {
            log.Printf("Error watching service: %v", err)
            time.Sleep(5 * time.Second)
            continue
        }

        lastIndex = meta.LastIndex

        var endpoints []string
        for _, service := range services {
            endpoint := fmt.Sprintf("%s:%d",
                service.Service.Address,
                service.Service.Port)
            endpoints = append(endpoints, endpoint)
        }

        callback(endpoints)
    }
}

// 更新 APISIX 路由
func updateAPISIXUpstream(serviceName string, endpoints []string) error {
    // 构建 APISIX upstream 配置
    nodes := make(map[string]int)
    for _, endpoint := range endpoints {
        nodes[endpoint] = 1
    }

    config := map[string]interface{}{
        "type": "roundrobin",
        "nodes": nodes,
    }

    // 调用 APISIX Admin API 更新
    // ...实现略...

    return nil
}

func main() {
    sd, err := NewServiceDiscovery("localhost:8500")
    if err != nil {
        log.Fatal(err)
    }

    // 监听服务变化并更新路由
    go sd.WatchService("user-service", func(endpoints []string) {
        log.Printf("Service endpoints updated: %v", endpoints)
        updateAPISIXUpstream("user-service", endpoints)
    })

    select {}
}
```

---

## 实战案例

### 案例: 电商大促灰度发布

**场景**: 双11 大促前发布新版本,采用多维度灰度策略

```python
# multi_dimension_canary.py
import requests
import hashlib

class MultiDimensionCanary:
    def __init__(self, admin_api, api_key):
        self.admin_api = admin_api
        self.api_key = api_key

    def create_canary_routes(self):
        """创建多维度灰度路由"""

        # 1. 内部员工路由 (100% 新版)
        self.create_route("internal-employees", {
            "uri": "/api/v1/products/*",
            "priority": 100,
            "vars": [
                ["http_x_employee_id", "~~", ".*"]
            ],
            "upstream": "product-service-v2:8080"
        })

        # 2. 白名单用户路由 (100% 新版)
        self.create_route("whitelist-users", {
            "uri": "/api/v1/products/*",
            "priority": 90,
            "vars": [
                ["cookie_whitelist", "==", "true"]
            ],
            "upstream": "product-service-v2:8080"
        })

        # 3. 地域灰度 (上海地区 50%)
        self.create_route("region-shanghai", {
            "uri": "/api/v1/products/*",
            "priority": 80,
            "vars": [
                ["http_x_city", "==", "Shanghai"]
            ],
            "plugins": {
                "traffic-split": {
                    "rules": [{
                        "weighted_upstreams": [
                            {"upstream": {"nodes": {"product-service-v2:8080": 1}}, "weight": 50},
                            {"weight": 50}
                        ]
                    }]
                }
            },
            "upstream": "product-service-v1:8080"
        })

        # 4. 用户 ID 哈希灰度 (10%)
        self.create_route("user-hash-canary", {
            "uri": "/api/v1/products/*",
            "priority": 70,
            "vars": [
                ["lua", "return tonumber(ngx.var.cookie_user_id) % 10 == 0"]
            ],
            "upstream": "product-service-v2:8080"
        })

        # 5. 默认路由 (稳定版)
        self.create_route("default-stable", {
            "uri": "/api/v1/products/*",
            "priority": 1,
            "upstream": "product-service-v1:8080"
        })

    def create_route(self, name, config):
        """创建路由"""
        url = f"{self.admin_api}/apisix/admin/routes/{name}"
        headers = {"X-API-KEY": self.api_key}

        response = requests.put(url, headers=headers, json=config)
        if response.status_code == 200:
            print(f"Route {name} created successfully")
        else:
            print(f"Failed to create route {name}: {response.text}")

if __name__ == "__main__":
    canary = MultiDimensionCanary(
        "http://localhost:9180",
        "your-api-key"
    )
    canary.create_canary_routes()
```

---

## 总结

### 路由策略选择指南

```
┌─────────────────────────────────────────┐
│         路由策略决策树                  │
│                                         │
│         需要会话保持?                   │
│            │                            │
│      ┌─────┴─────┐                     │
│     是           否                     │
│      │            │                    │
│  IP Hash      需要均衡性能?            │
│  一致性哈希       │                    │
│            ┌──────┴──────┐             │
│           是             否             │
│            │              │             │
│      Least Conn    需要加权?          │
│      响应时间          │               │
│                  ┌─────┴─────┐        │
│                 是           否        │
│                  │            │        │
│           Weighted RR   Round Robin    │
└─────────────────────────────────────────┘
```

### 关键要点

1. **路由优先级**: 精确匹配 > 正则匹配 > 前缀匹配
2. **负载均衡**: 根据场景选择合适算法
3. **灰度发布**: 多维度、渐进式、可回滚
4. **流量控制**: 多层限流、令牌桶 + 漏桶
5. **动态路由**: 结合服务发现实现自动化

### 下一步学习

- [03_authentication.md](03_authentication.md) - 认证授权实现
- [04_gateway_comparison.md](04_gateway_comparison.md) - 网关选型对比
