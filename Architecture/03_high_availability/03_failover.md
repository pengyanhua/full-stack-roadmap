# 故障转移与容灾

## 一、故障类型与应对

### 故障分类

```
┌─────────────────────────────────────────────────────────────────┐
│                        故障类型金字塔                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                          /\                                     │
│                         /  \   数据中心级故障                    │
│                        / DC \  (自然灾害、电力故障)               │
│                       /______\                                  │
│                      /        \                                 │
│                     /   机房   \  机房级故障                     │
│                    /   Room    \  (网络故障、空调故障)            │
│                   /______________\                              │
│                  /                \                             │
│                 /    机架级故障    \                             │
│                /      Rack        \  (交换机故障、PDU故障)        │
│               /____________________\                            │
│              /                      \                           │
│             /      服务器级故障       \                          │
│            /        Server          \  (硬件故障、OS崩溃)         │
│           /__________________________\                          │
│          /                            \                         │
│         /        进程/服务级故障         \                        │
│        /          Process             \  (OOM、死锁、Bug)         │
│       /________________________________\                        │
│                                                                 │
│  频率:  高 ◀───────────────────────────────────────────▶ 低     │
│  影响:  低 ◀───────────────────────────────────────────▶ 高     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 应对策略矩阵

| 故障类型 | 检测手段 | 恢复时间 | 应对策略 |
|---------|---------|---------|---------|
| 进程故障 | 心跳检测 | 秒级 | 进程重启、服务切换 |
| 服务器故障 | 健康检查 | 秒~分钟级 | 负载均衡摘除 |
| 机架故障 | 网络监控 | 分钟级 | 跨机架冗余 |
| 机房故障 | 多点探测 | 分钟~小时级 | 同城双活 |
| 数据中心故障 | 全局监控 | 小时级 | 异地多活 |

---

## 二、健康检查

### 1. 检查类型

```
┌─────────────────────────────────────────────────────────────────┐
│                      健康检查层次                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Level 1: TCP 端口检查                                         │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  检测端口是否可连接                                       │  │
│   │  优点: 简单、快速                                         │  │
│   │  缺点: 无法检测应用状态                                   │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   Level 2: HTTP 健康端点                                        │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  GET /health → 200 OK                                    │  │
│   │  优点: 可检测应用基本状态                                 │  │
│   │  缺点: 无法检测依赖健康                                   │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   Level 3: 深度健康检查                                         │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  GET /health/deep → 检查所有依赖                         │  │
│   │  优点: 全面检测                                           │  │
│   │  缺点: 耗时、可能级联失败                                 │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   Level 4: 业务探针                                             │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  模拟真实业务请求                                         │  │
│   │  优点: 最真实的检测                                       │  │
│   │  缺点: 复杂、有副作用                                     │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 健康检查实现

```go
// 多层次健康检查
type HealthChecker struct {
    db       *sql.DB
    redis    *redis.Client
    services map[string]string  // 下游服务地址
}

// 基础健康检查（用于负载均衡）
func (h *HealthChecker) LivenessCheck() HealthResult {
    return HealthResult{
        Status: "UP",
        Time:   time.Now(),
    }
}

// 就绪检查（用于 K8s readiness probe）
func (h *HealthChecker) ReadinessCheck() HealthResult {
    result := HealthResult{
        Status: "UP",
        Time:   time.Now(),
        Checks: make(map[string]CheckResult),
    }

    // 检查数据库
    if err := h.db.Ping(); err != nil {
        result.Status = "DOWN"
        result.Checks["database"] = CheckResult{Status: "DOWN", Error: err.Error()}
    } else {
        result.Checks["database"] = CheckResult{Status: "UP"}
    }

    // 检查 Redis
    if err := h.redis.Ping(context.Background()).Err(); err != nil {
        result.Status = "DOWN"
        result.Checks["redis"] = CheckResult{Status: "DOWN", Error: err.Error()}
    } else {
        result.Checks["redis"] = CheckResult{Status: "UP"}
    }

    return result
}

// 深度健康检查（用于诊断）
func (h *HealthChecker) DeepHealthCheck(ctx context.Context) HealthResult {
    result := h.ReadinessCheck()

    // 检查下游服务
    var wg sync.WaitGroup
    var mu sync.Mutex

    for name, url := range h.services {
        wg.Add(1)
        go func(name, url string) {
            defer wg.Done()

            checkResult := h.checkService(ctx, url)

            mu.Lock()
            result.Checks[name] = checkResult
            if checkResult.Status == "DOWN" {
                result.Status = "DEGRADED"  // 下游异常标记为降级
            }
            mu.Unlock()
        }(name, url)
    }

    wg.Wait()
    return result
}

func (h *HealthChecker) checkService(ctx context.Context, url string) CheckResult {
    ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
    defer cancel()

    req, _ := http.NewRequestWithContext(ctx, "GET", url+"/health", nil)
    resp, err := http.DefaultClient.Do(req)

    if err != nil {
        return CheckResult{Status: "DOWN", Error: err.Error()}
    }
    defer resp.Body.Close()

    if resp.StatusCode == 200 {
        return CheckResult{Status: "UP", Latency: "10ms"}
    }
    return CheckResult{Status: "DOWN", Error: fmt.Sprintf("status: %d", resp.StatusCode)}
}

// HTTP Handler
func (h *HealthChecker) RegisterEndpoints(mux *http.ServeMux) {
    // 存活检查 - 快速响应
    mux.HandleFunc("/health/live", func(w http.ResponseWriter, r *http.Request) {
        json.NewEncoder(w).Encode(h.LivenessCheck())
    })

    // 就绪检查 - 检查核心依赖
    mux.HandleFunc("/health/ready", func(w http.ResponseWriter, r *http.Request) {
        result := h.ReadinessCheck()
        if result.Status != "UP" {
            w.WriteHeader(http.StatusServiceUnavailable)
        }
        json.NewEncoder(w).Encode(result)
    })

    // 深度检查 - 诊断用
    mux.HandleFunc("/health/deep", func(w http.ResponseWriter, r *http.Request) {
        result := h.DeepHealthCheck(r.Context())
        json.NewEncoder(w).Encode(result)
    })
}
```

### 3. K8s 健康检查配置

```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: app
        # 存活探针：检测容器是否需要重启
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 10    # 启动后等待时间
          periodSeconds: 10          # 检查间隔
          timeoutSeconds: 3          # 超时时间
          failureThreshold: 3        # 失败次数后重启

        # 就绪探针：检测是否可以接收流量
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3        # 失败后从 Service 摘除

        # 启动探针：保护慢启动应用
        startupProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 0
          periodSeconds: 5
          failureThreshold: 30       # 最多等待 150 秒启动
```

---

## 三、故障转移策略

### 1. 服务级故障转移

```
┌─────────────────────────────────────────────────────────────────┐
│                   服务故障转移流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   正常状态:                                                      │
│   ┌─────────┐     ┌────────────────┐                            │
│   │  客户端  │────▶│  负载均衡器    │                            │
│   └─────────┘     └────────────────┘                            │
│                          │                                      │
│               ┌──────────┼──────────┐                           │
│               ▼          ▼          ▼                           │
│          ┌────────┐ ┌────────┐ ┌────────┐                       │
│          │实例 A  │ │实例 B  │ │实例 C  │                       │
│          │ (健康) │ │ (健康) │ │ (健康) │                       │
│          └────────┘ └────────┘ └────────┘                       │
│                                                                 │
│   故障检测:                                                      │
│          ┌────────┐ ┌────────┐ ┌────────┐                       │
│          │实例 A  │ │实例 B  │ │实例 C  │                       │
│          │  ✓     │ │   ✗   │ │  ✓     │  ← 健康检查失败        │
│          └────────┘ └────────┘ └────────┘                       │
│                                                                 │
│   故障转移后:                                                    │
│   ┌─────────┐     ┌────────────────┐                            │
│   │  客户端  │────▶│  负载均衡器    │                            │
│   └─────────┘     └────────────────┘                            │
│                          │                                      │
│               ┌──────────┴──────────┐                           │
│               ▼                     ▼                           │
│          ┌────────┐            ┌────────┐                       │
│          │实例 A  │            │实例 C  │  ← 实例 B 被摘除       │
│          │ 50%    │            │ 50%    │                       │
│          └────────┘            └────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 数据库故障转移

```
┌─────────────────────────────────────────────────────────────────┐
│                MySQL 主从故障转移                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   正常状态:                                                      │
│                      ┌─────────────┐                            │
│      写入 ─────────▶ │  Master     │                            │
│                      └─────────────┘                            │
│                            │ 复制                               │
│                  ┌─────────┴─────────┐                          │
│                  ▼                   ▼                          │
│           ┌─────────────┐    ┌─────────────┐                    │
│   读取◀───│  Slave 1    │    │  Slave 2    │───▶ 读取           │
│           └─────────────┘    └─────────────┘                    │
│                                                                 │
│   故障发生:                                                      │
│                      ┌─────────────┐                            │
│                      │  Master ✗   │  ← 主库故障                │
│                      └─────────────┘                            │
│                                                                 │
│   故障转移:                                                      │
│                      ┌─────────────┐                            │
│      写入 ─────────▶ │  Slave 1    │  ← 提升为新主库            │
│                      │ (New Master)│                            │
│                      └─────────────┘                            │
│                            │ 复制                               │
│                            ▼                                    │
│                     ┌─────────────┐                             │
│             读取◀───│  Slave 2    │                             │
│                     └─────────────┘                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3. 自动故障转移实现

```go
// 数据库连接管理器（支持故障转移）
type DBManager struct {
    master       *sql.DB
    slaves       []*sql.DB
    healthySlaves []*sql.DB
    mu           sync.RWMutex

    // 故障检测配置
    checkInterval time.Duration
    failThreshold int
}

func NewDBManager(masterDSN string, slaveDSNs []string) (*DBManager, error) {
    master, err := sql.Open("mysql", masterDSN)
    if err != nil {
        return nil, err
    }

    slaves := make([]*sql.DB, 0, len(slaveDSNs))
    for _, dsn := range slaveDSNs {
        slave, err := sql.Open("mysql", dsn)
        if err != nil {
            continue  // 单个从库失败不影响整体
        }
        slaves = append(slaves, slave)
    }

    mgr := &DBManager{
        master:        master,
        slaves:        slaves,
        healthySlaves: slaves,
        checkInterval: 5 * time.Second,
        failThreshold: 3,
    }

    // 启动健康检查
    go mgr.healthCheckLoop()

    return mgr, nil
}

func (m *DBManager) healthCheckLoop() {
    ticker := time.NewTicker(m.checkInterval)
    failCounts := make(map[*sql.DB]int)

    for range ticker.C {
        var healthy []*sql.DB

        for _, slave := range m.slaves {
            if err := slave.Ping(); err != nil {
                failCounts[slave]++
                if failCounts[slave] >= m.failThreshold {
                    log.Printf("Slave marked as unhealthy: %v", err)
                    continue
                }
            } else {
                failCounts[slave] = 0
            }
            healthy = append(healthy, slave)
        }

        m.mu.Lock()
        m.healthySlaves = healthy
        m.mu.Unlock()
    }
}

// 获取写连接（主库）
func (m *DBManager) Master() *sql.DB {
    return m.master
}

// 获取读连接（从库，负载均衡）
func (m *DBManager) Slave() *sql.DB {
    m.mu.RLock()
    defer m.mu.RUnlock()

    if len(m.healthySlaves) == 0 {
        // 没有健康的从库，降级到主库
        log.Println("No healthy slaves, falling back to master")
        return m.master
    }

    // 简单轮询
    idx := rand.Intn(len(m.healthySlaves))
    return m.healthySlaves[idx]
}
```

---

## 四、Redis Sentinel 与 Cluster 故障转移

### 1. Redis Sentinel

```
┌─────────────────────────────────────────────────────────────────┐
│                   Redis Sentinel 架构                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐              │
│   │ Sentinel1 │────│ Sentinel2 │────│ Sentinel3 │              │
│   └───────────┘    └───────────┘    └───────────┘              │
│         │               │                │                      │
│         │      监控/选举/通知             │                      │
│         │               │                │                      │
│         ▼               ▼                ▼                      │
│   ┌───────────────────────────────────────────────────┐        │
│   │                    Redis 实例                      │        │
│   │  ┌─────────┐    ┌─────────┐    ┌─────────┐       │        │
│   │  │ Master  │───▶│ Slave 1 │    │ Slave 2 │       │        │
│   │  └─────────┘    └─────────┘    └─────────┘       │        │
│   └───────────────────────────────────────────────────┘        │
│                                                                 │
│   故障转移流程:                                                  │
│   1. Sentinel 检测到 Master 不可用                               │
│   2. 多个 Sentinel 投票确认 (Quorum)                             │
│   3. 选举 Leader Sentinel 执行故障转移                           │
│   4. 选择最合适的 Slave 提升为 Master                            │
│   5. 通知其他 Slave 复制新 Master                                │
│   6. 通知客户端新的 Master 地址                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Sentinel 配置与使用

```bash
# sentinel.conf
sentinel monitor mymaster 192.168.1.100 6379 2  # 2 表示 quorum
sentinel down-after-milliseconds mymaster 5000   # 5秒无响应视为下线
sentinel failover-timeout mymaster 60000         # 故障转移超时
sentinel parallel-syncs mymaster 1               # 同时同步的 slave 数
```

```go
import "github.com/go-redis/redis/v8"

func NewSentinelClient() *redis.Client {
    return redis.NewFailoverClient(&redis.FailoverOptions{
        MasterName:    "mymaster",
        SentinelAddrs: []string{
            "192.168.1.101:26379",
            "192.168.1.102:26379",
            "192.168.1.103:26379",
        },
        // 连接池配置
        PoolSize:     100,
        MinIdleConns: 10,
        // 超时配置
        DialTimeout:  5 * time.Second,
        ReadTimeout:  3 * time.Second,
        WriteTimeout: 3 * time.Second,
    })
}

// 使用时自动处理故障转移
func Example() {
    client := NewSentinelClient()

    // 写操作自动路由到 Master
    client.Set(ctx, "key", "value", 0)

    // 读操作也路由到 Master（如需读写分离需额外配置）
    client.Get(ctx, "key")
}
```

### 3. Redis Cluster 故障转移

```go
import "github.com/go-redis/redis/v8"

func NewClusterClient() *redis.ClusterClient {
    return redis.NewClusterClient(&redis.ClusterOptions{
        Addrs: []string{
            "192.168.1.101:7000",
            "192.168.1.101:7001",
            "192.168.1.102:7000",
            "192.168.1.102:7001",
            "192.168.1.103:7000",
            "192.168.1.103:7001",
        },
        // 故障转移配置
        MaxRedirects: 8,              // 最大重定向次数
        ReadOnly:     false,          // 是否只读
        RouteByLatency: true,         // 按延迟路由读请求

        // 连接池
        PoolSize:     100,
        MinIdleConns: 10,
    })
}

// Cluster 故障转移是自动的
// 当某个 Master 节点故障时，其 Slave 会自动提升
```

---

## 五、多层故障转移架构

### 1. 完整的高可用架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     多层故障转移架构                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Layer 1: DNS 故障转移                                         │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  DNS (GSLB)                                              │  │
│   │  api.example.com → 健康的数据中心 IP                      │  │
│   │  TTL: 60s，故障时自动切换                                 │  │
│   └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│   Layer 2: 负载均衡故障转移                                      │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                    ┌─────────────┐                       │  │
│   │   ┌─────────┐      │     LB      │      ┌─────────┐     │  │
│   │   │  LB-1   │◀─HA─▶│   (VIP)     │◀─HA─▶│  LB-2   │     │  │
│   │   │ Master  │      │ Keepalived  │      │ Backup  │     │  │
│   │   └─────────┘      └─────────────┘      └─────────┘     │  │
│   └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│   Layer 3: 应用层故障转移                                        │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │     ┌─────────┐  ┌─────────┐  ┌─────────┐              │  │
│   │     │ App-1   │  │ App-2   │  │ App-3   │              │  │
│   │     │  ✓      │  │   ✓     │  │   ✗     │ ← 自动摘除  │  │
│   │     └─────────┘  └─────────┘  └─────────┘              │  │
│   └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│   Layer 4: 数据层故障转移                                        │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  ┌──────────────────┐    ┌──────────────────┐          │  │
│   │  │   MySQL Master   │───▶│   MySQL Slave    │          │  │
│   │  │  (MHA/Orchestr)  │    │  (auto failover) │          │  │
│   │  └──────────────────┘    └──────────────────┘          │  │
│   │                                                         │  │
│   │  ┌──────────────────┐    ┌──────────────────┐          │  │
│   │  │  Redis Sentinel  │───▶│  Redis Cluster   │          │  │
│   │  │  (auto failover) │    │  (auto failover) │          │  │
│   │  └──────────────────┘    └──────────────────┘          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Keepalived 配置 (LB 故障转移)

```bash
# /etc/keepalived/keepalived.conf (Master)
vrrp_script check_nginx {
    script "/usr/local/bin/check_nginx.sh"
    interval 2
    weight -20
}

vrrp_instance VI_1 {
    state MASTER
    interface eth0
    virtual_router_id 51
    priority 100
    advert_int 1

    authentication {
        auth_type PASS
        auth_pass secret
    }

    virtual_ipaddress {
        192.168.1.100/24    # VIP
    }

    track_script {
        check_nginx
    }

    notify_master "/usr/local/bin/notify_master.sh"
    notify_backup "/usr/local/bin/notify_backup.sh"
    notify_fault "/usr/local/bin/notify_fault.sh"
}
```

```bash
#!/bin/bash
# /usr/local/bin/check_nginx.sh
if ! pgrep -x nginx > /dev/null; then
    exit 1
fi
if ! curl -s http://localhost/health > /dev/null; then
    exit 1
fi
exit 0
```

---

## 六、故障转移最佳实践

### 1. 坑点与避免

```go
// ❌ 坑点 1: 健康检查太激进
// 网络抖动导致频繁故障转移
healthCheck:
  interval: 1s          # 太频繁
  failureThreshold: 1   # 一次失败就切换

// ✅ 正确配置
healthCheck:
  interval: 5s
  failureThreshold: 3   # 连续3次失败才切换
  successThreshold: 2   # 恢复需要连续2次成功


// ❌ 坑点 2: 脑裂问题
// 网络分区时，两个节点都认为自己是 Master
// 导致数据不一致

// ✅ 解决方案: Quorum 机制
sentinel monitor mymaster 192.168.1.100 6379 2  # 需要2个Sentinel同意
// 或使用 STONITH (Shoot The Other Node In The Head)


// ❌ 坑点 3: 数据丢失
// 异步复制时，Master 故障可能丢失未同步的数据

// ✅ 解决方案
// 1. 使用半同步复制
set global rpl_semi_sync_master_enabled = 1;
// 2. 或使用 Raft 协议的强一致性存储（etcd, TiKV）


// ❌ 坑点 4: 故障转移风暴
// 多个组件同时故障转移，系统雪崩

// ✅ 解决方案: 错峰故障转移
type FailoverCoordinator struct {
    lastFailover map[string]time.Time
    minInterval  time.Duration  // 最小故障转移间隔
    mu           sync.Mutex
}

func (c *FailoverCoordinator) CanFailover(component string) bool {
    c.mu.Lock()
    defer c.mu.Unlock()

    if last, ok := c.lastFailover[component]; ok {
        if time.Since(last) < c.minInterval {
            return false  // 太频繁，拒绝
        }
    }

    c.lastFailover[component] = time.Now()
    return true
}
```

### 2. 故障转移演练

```go
// 混沌工程 - 故障注入
type ChaosEngine struct {
    targets []Target
}

func (c *ChaosEngine) InjectFailure(target Target, failure FailureType) error {
    switch failure {
    case NetworkDelay:
        return exec.Command("tc", "qdisc", "add", "dev", "eth0",
            "root", "netem", "delay", "100ms").Run()

    case NetworkPartition:
        return exec.Command("iptables", "-A", "INPUT",
            "-s", target.IP, "-j", "DROP").Run()

    case ProcessKill:
        return exec.Command("kill", "-9", target.PID).Run()

    case DiskFull:
        return exec.Command("dd", "if=/dev/zero",
            "of=/tmp/fill", "bs=1M").Run()
    }
    return nil
}

// 定期演练脚本
func ScheduledDrill() {
    // 1. 通知相关人员
    notify("Starting failover drill")

    // 2. 记录当前状态
    before := captureMetrics()

    // 3. 注入故障
    engine := NewChaosEngine()
    engine.InjectFailure(redisTarget, ProcessKill)

    // 4. 等待故障转移
    time.Sleep(30 * time.Second)

    // 5. 验证系统状态
    after := captureMetrics()
    if !validateFailover(before, after) {
        alert("Failover drill failed!")
    }

    // 6. 恢复故障
    engine.Recover()

    // 7. 生成报告
    generateReport(before, after)
}
```

---

## 七、检查清单

### 故障检测检查

- [ ] 健康检查覆盖所有关键组件？
- [ ] 健康检查间隔和阈值是否合理？
- [ ] 是否区分了存活检查和就绪检查？
- [ ] 是否有独立的监控系统检测？

### 故障转移检查

- [ ] 所有有状态服务是否有冗余？
- [ ] 故障转移是否自动化？
- [ ] 是否考虑了脑裂问题？
- [ ] 故障转移后是否有告警？

### 演练检查

- [ ] 是否定期进行故障转移演练？
- [ ] 是否有演练回滚预案？
- [ ] 演练结果是否有文档记录？
- [ ] 是否根据演练结果改进系统？
