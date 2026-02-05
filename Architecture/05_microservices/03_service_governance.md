# 服务治理

## 一、服务注册与发现

### 1. 服务注册中心

```
┌─────────────────────────────────────────────────────────────────┐
│                   服务注册与发现架构                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                     ┌─────────────────┐                         │
│                     │   注册中心       │                         │
│                     │  (Consul/Nacos) │                         │
│                     └─────────────────┘                         │
│                            │                                    │
│         ┌──────────────────┼──────────────────┐                 │
│         │                  │                  │                 │
│    ①注册│             ②心跳│             ③发现│                 │
│         ▼                  │                  ▼                 │
│   ┌─────────────┐          │          ┌─────────────┐          │
│   │ Service A   │          │          │ Service B   │          │
│   │ 192.168.1.1 │          │          │ 查询Service │          │
│   │ 192.168.1.2 │          │          │   A的地址   │          │
│   └─────────────┘          │          └─────────────┘          │
│                            │                  │                 │
│                            │                  │④调用            │
│                            │                  ▼                 │
│                            │          ┌─────────────┐          │
│                            │          │ Service A   │          │
│                            │          │ 192.168.1.1 │          │
│                            │          └─────────────┘          │
│                                                                 │
│   流程:                                                         │
│   ① 服务启动时向注册中心注册                                    │
│   ② 定期发送心跳保持注册状态                                    │
│   ③ 调用方查询目标服务地址                                      │
│   ④ 调用方直接访问目标服务                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 注册中心对比

| 特性 | Consul | Nacos | etcd | ZooKeeper |
|------|--------|-------|------|-----------|
| CAP | CP/AP可选 | CP+AP | CP | CP |
| 健康检查 | TCP/HTTP/gRPC | TCP/HTTP | 心跳 | 心跳 |
| 配置中心 | KV存储 | 内置 | KV存储 | 需开发 |
| 易用性 | 高 | 高 | 中 | 低 |
| 生态 | HashiCorp | 阿里巴巴 | K8s | Hadoop |

### 3. Go 服务注册示例

```go
import (
    "github.com/hashicorp/consul/api"
)

type ServiceRegistry struct {
    client     *api.Client
    serviceID  string
    serviceName string
    address    string
    port       int
}

func NewServiceRegistry(consulAddr, serviceName, address string, port int) (*ServiceRegistry, error) {
    config := api.DefaultConfig()
    config.Address = consulAddr

    client, err := api.NewClient(config)
    if err != nil {
        return nil, err
    }

    return &ServiceRegistry{
        client:      client,
        serviceID:   fmt.Sprintf("%s-%s-%d", serviceName, address, port),
        serviceName: serviceName,
        address:     address,
        port:        port,
    }, nil
}

// 注册服务
func (r *ServiceRegistry) Register() error {
    registration := &api.AgentServiceRegistration{
        ID:      r.serviceID,
        Name:    r.serviceName,
        Address: r.address,
        Port:    r.port,
        Tags:    []string{"api", "v1"},
        Check: &api.AgentServiceCheck{
            HTTP:                           fmt.Sprintf("http://%s:%d/health", r.address, r.port),
            Interval:                       "10s",
            Timeout:                        "5s",
            DeregisterCriticalServiceAfter: "30s",
        },
    }

    return r.client.Agent().ServiceRegister(registration)
}

// 注销服务
func (r *ServiceRegistry) Deregister() error {
    return r.client.Agent().ServiceDeregister(r.serviceID)
}

// 服务发现
func (r *ServiceRegistry) Discover(serviceName string) ([]*api.ServiceEntry, error) {
    entries, _, err := r.client.Health().Service(serviceName, "", true, nil)
    return entries, err
}

// 使用示例
func main() {
    registry, _ := NewServiceRegistry("consul:8500", "user-service", "192.168.1.10", 8080)

    // 启动时注册
    registry.Register()

    // 关闭时注销
    defer registry.Deregister()

    // 发现其他服务
    services, _ := registry.Discover("order-service")
    for _, svc := range services {
        fmt.Printf("%s:%d\n", svc.Service.Address, svc.Service.Port)
    }
}
```

---

## 二、负载均衡

### 1. 负载均衡策略

```
┌─────────────────────────────────────────────────────────────────┐
│                    负载均衡策略对比                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. 轮询 (Round Robin)                                         │
│   ─────────────────────────────────────────────────────────    │
│   请求依次分配: A → B → C → A → B → C ...                       │
│   优点: 简单、公平    缺点: 不考虑服务器性能差异                  │
│                                                                 │
│   2. 加权轮询 (Weighted Round Robin)                            │
│   ─────────────────────────────────────────────────────────    │
│   按权重分配: A(w=3) → A → A → B(w=1) → A ...                  │
│   优点: 考虑性能差异   缺点: 需要预设权重                        │
│                                                                 │
│   3. 随机 (Random)                                              │
│   ─────────────────────────────────────────────────────────    │
│   随机选择服务器                                                │
│   优点: 简单    缺点: 可能不均匀                                │
│                                                                 │
│   4. 最少连接 (Least Connections)                               │
│   ─────────────────────────────────────────────────────────    │
│   选择当前连接数最少的服务器                                    │
│   优点: 动态平衡    缺点: 需要维护连接计数                       │
│                                                                 │
│   5. 一致性哈希 (Consistent Hash)                               │
│   ─────────────────────────────────────────────────────────    │
│   相同key总是路由到同一服务器                                   │
│   优点: 会话保持、缓存友好   缺点: 可能负载不均                  │
│                                                                 │
│   6. P2C (Power of Two Choices)                                 │
│   ─────────────────────────────────────────────────────────    │
│   随机选两个，选负载较低的                                      │
│   优点: 简单且效果好   缺点: gRPC默认使用                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 负载均衡实现

```go
// 负载均衡器接口
type LoadBalancer interface {
    Select(instances []*Instance) *Instance
}

type Instance struct {
    Address     string
    Weight      int
    Connections int32
}

// 轮询
type RoundRobin struct {
    counter uint64
}

func (r *RoundRobin) Select(instances []*Instance) *Instance {
    n := atomic.AddUint64(&r.counter, 1)
    return instances[n%uint64(len(instances))]
}

// 加权轮询
type WeightedRoundRobin struct {
    mu      sync.Mutex
    current int
    weights []int
}

func (w *WeightedRoundRobin) Select(instances []*Instance) *Instance {
    w.mu.Lock()
    defer w.mu.Unlock()

    totalWeight := 0
    for _, inst := range instances {
        totalWeight += inst.Weight
    }

    w.current = (w.current + 1) % totalWeight

    sum := 0
    for _, inst := range instances {
        sum += inst.Weight
        if w.current < sum {
            return inst
        }
    }
    return instances[0]
}

// 最少连接
type LeastConnections struct{}

func (l *LeastConnections) Select(instances []*Instance) *Instance {
    var selected *Instance
    minConns := int32(math.MaxInt32)

    for _, inst := range instances {
        conns := atomic.LoadInt32(&inst.Connections)
        if conns < minConns {
            minConns = conns
            selected = inst
        }
    }
    return selected
}

// P2C
type P2C struct{}

func (p *P2C) Select(instances []*Instance) *Instance {
    if len(instances) <= 2 {
        return instances[0]
    }

    // 随机选两个
    i := rand.Intn(len(instances))
    j := rand.Intn(len(instances))
    for j == i {
        j = rand.Intn(len(instances))
    }

    // 选负载较低的
    if instances[i].Connections <= instances[j].Connections {
        return instances[i]
    }
    return instances[j]
}
```

---

## 三、配置中心

### 1. 配置管理

```go
// Nacos 配置中心
import "github.com/nacos-group/nacos-sdk-go/v2/clients/config_client"

type ConfigManager struct {
    client config_client.IConfigClient
}

// 获取配置
func (m *ConfigManager) GetConfig(dataId, group string) (string, error) {
    return m.client.GetConfig(vo.ConfigParam{
        DataId: dataId,
        Group:  group,
    })
}

// 监听配置变更
func (m *ConfigManager) WatchConfig(dataId, group string, onChange func(string)) error {
    return m.client.ListenConfig(vo.ConfigParam{
        DataId: dataId,
        Group:  group,
        OnChange: func(namespace, group, dataId, data string) {
            onChange(data)
        },
    })
}

// 配置热更新
type AppConfig struct {
    mu          sync.RWMutex
    MaxConns    int    `yaml:"max_conns"`
    Timeout     int    `yaml:"timeout"`
    FeatureFlag bool   `yaml:"feature_flag"`
}

var config = &AppConfig{}

func InitConfig(manager *ConfigManager) {
    // 初始加载
    data, _ := manager.GetConfig("app.yaml", "DEFAULT_GROUP")
    yaml.Unmarshal([]byte(data), config)

    // 监听变更
    manager.WatchConfig("app.yaml", "DEFAULT_GROUP", func(data string) {
        newConfig := &AppConfig{}
        if err := yaml.Unmarshal([]byte(data), newConfig); err == nil {
            config.mu.Lock()
            *config = *newConfig
            config.mu.Unlock()
            log.Println("Config updated")
        }
    })
}

func GetConfig() *AppConfig {
    config.mu.RLock()
    defer config.mu.RUnlock()
    return config
}
```

---

## 四、服务网关

### 1. 网关职责

```
┌─────────────────────────────────────────────────────────────────┐
│                      API 网关架构                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                        客户端                                    │
│                          │                                      │
│                          ▼                                      │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                    API Gateway                           │  │
│   │  ┌─────────────────────────────────────────────────┐   │  │
│   │  │                    功能职责                      │   │  │
│   │  │  • 路由转发      • 负载均衡                     │   │  │
│   │  │  • 认证鉴权      • 限流熔断                     │   │  │
│   │  │  • 协议转换      • 请求聚合                     │   │  │
│   │  │  • 日志监控      • 灰度发布                     │   │  │
│   │  └─────────────────────────────────────────────────┘   │  │
│   └─────────────────────────────────────────────────────────┘  │
│                          │                                      │
│          ┌───────────────┼───────────────┐                      │
│          ▼               ▼               ▼                      │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│   │ User Service│ │Order Service│ │Product Svc  │              │
│   └─────────────┘ └─────────────┘ └─────────────┘              │
│                                                                 │
│   常见网关:                                                      │
│   • Kong - 基于 Nginx，插件丰富                                  │
│   • APISIX - 国产，性能好                                        │
│   • Spring Cloud Gateway - Java 生态                            │
│   • Envoy - 云原生，Service Mesh                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 路由配置示例

```yaml
# Kong 路由配置
services:
  - name: user-service
    url: http://user-service:8080
    routes:
      - name: user-route
        paths:
          - /api/users
        methods:
          - GET
          - POST
        plugins:
          - name: rate-limiting
            config:
              minute: 100
          - name: jwt
            config:
              secret_is_base64: false

  - name: order-service
    url: http://order-service:8080
    routes:
      - name: order-route
        paths:
          - /api/orders
        plugins:
          - name: rate-limiting
            config:
              minute: 50
```

---

## 五、链路追踪

### OpenTelemetry 集成

```go
import (
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/exporters/jaeger"
    "go.opentelemetry.io/otel/sdk/trace"
)

func InitTracer(serviceName string) (*trace.TracerProvider, error) {
    exporter, err := jaeger.New(jaeger.WithCollectorEndpoint(
        jaeger.WithEndpoint("http://jaeger:14268/api/traces"),
    ))
    if err != nil {
        return nil, err
    }

    tp := trace.NewTracerProvider(
        trace.WithBatcher(exporter),
        trace.WithResource(resource.NewWithAttributes(
            semconv.SchemaURL,
            semconv.ServiceNameKey.String(serviceName),
        )),
    )

    otel.SetTracerProvider(tp)
    return tp, nil
}

// HTTP 中间件
func TracingMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        tracer := otel.Tracer("http-server")
        ctx, span := tracer.Start(c.Request.Context(), c.FullPath())
        defer span.End()

        span.SetAttributes(
            attribute.String("http.method", c.Request.Method),
            attribute.String("http.url", c.Request.URL.String()),
        )

        c.Request = c.Request.WithContext(ctx)
        c.Next()

        span.SetAttributes(
            attribute.Int("http.status_code", c.Writer.Status()),
        )
    }
}

// 传播 trace context
func CallDownstream(ctx context.Context, url string) error {
    tracer := otel.Tracer("http-client")
    ctx, span := tracer.Start(ctx, "call-downstream")
    defer span.End()

    req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)

    // 注入 trace header
    otel.GetTextMapPropagator().Inject(ctx, propagation.HeaderCarrier(req.Header))

    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        span.RecordError(err)
        return err
    }
    defer resp.Body.Close()

    return nil
}
```

---

## 六、检查清单

### 服务治理检查

- [ ] 是否有服务注册发现机制？
- [ ] 负载均衡策略是否合适？
- [ ] 是否有配置中心？
- [ ] 是否有统一网关？

### 可观测性检查

- [ ] 是否接入链路追踪？
- [ ] 是否有统一日志平台？
- [ ] 是否有监控告警？
- [ ] 是否能快速定位问题？
