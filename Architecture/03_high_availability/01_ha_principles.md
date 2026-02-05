# 高可用架构原则

## 一、可用性指标

### SLA 与可用性

```
SLA (Service Level Agreement) 服务等级协议

┌─────────────────────────────────────────────────────────────┐
│                     可用性等级对照表                         │
├───────────┬──────────────┬────────────────────────────────┤
│  可用性    │   年故障时间  │            说明                │
├───────────┼──────────────┼────────────────────────────────┤
│  99%      │   87.6 小时   │  约 3.6 天                     │
│  99.9%    │   8.76 小时   │  三个9，大部分互联网服务       │
│  99.99%   │   52.6 分钟   │  四个9，核心业务系统           │
│  99.999%  │   5.26 分钟   │  五个9，金融/电信级            │
│  99.9999% │   31.5 秒     │  六个9，几乎无故障             │
└───────────┴──────────────┴────────────────────────────────┘

计算公式：
可用性 = (总时间 - 故障时间) / 总时间 × 100%
       = MTBF / (MTBF + MTTR)

MTBF: Mean Time Between Failures（平均故障间隔）
MTTR: Mean Time To Repair（平均修复时间）
```

### 核心指标

```
┌─────────────────────────────────────────────────────────────┐
│                       高可用核心指标                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  RTO (Recovery Time Objective) 恢复时间目标                  │
│  └── 从故障发生到系统恢复的最大可接受时间                    │
│  └── 例：电商系统 RTO = 5 分钟                               │
│                                                             │
│  RPO (Recovery Point Objective) 恢复点目标                   │
│  └── 可接受的最大数据丢失量（时间度量）                      │
│  └── 例：每小时备份，RPO = 1 小时                            │
│                                                             │
│                      故障发生                                │
│                         │                                   │
│  ──────────────────────┼────────────────────────▶ 时间     │
│         │              │                │                   │
│         │◀── RPO ────▶│◀──── RTO ────▶│                   │
│         │              │                │                   │
│      最后备份        故障点           恢复完成              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 二、高可用设计原则

### 1. 冗余设计

```
┌─────────────────────────────────────────────────────────────┐
│                       冗余设计                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  应用层冗余：                                                │
│  ┌──────────────────────────────────────────┐              │
│  │             Load Balancer                 │              │
│  │           (主备或集群)                     │              │
│  └──────────────────────────────────────────┘              │
│                      │                                      │
│      ┌───────────────┼───────────────┐                     │
│      ▼               ▼               ▼                     │
│  ┌────────┐     ┌────────┐     ┌────────┐                 │
│  │ App 1  │     │ App 2  │     │ App 3  │  ← 至少 2 个    │
│  └────────┘     └────────┘     └────────┘                 │
│                                                             │
│  数据层冗余：                                                │
│  ┌──────────────────────────────────────────┐              │
│  │  Master ◀──同步/异步复制──▶ Slave        │              │
│  │    │                          │          │              │
│  │    └────────────────────────┬─┘          │              │
│  │                             ▼            │              │
│  │                      故障自动切换         │              │
│  └──────────────────────────────────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. 故障隔离

```python
# 故障隔离策略

# 1. 线程池隔离
class ServiceIsolation:
    def __init__(self):
        # 不同服务使用独立线程池
        self.order_pool = ThreadPoolExecutor(max_workers=50)
        self.payment_pool = ThreadPoolExecutor(max_workers=30)
        self.inventory_pool = ThreadPoolExecutor(max_workers=20)

    def call_order_service(self, request):
        # 订单服务故障不影响支付服务
        return self.order_pool.submit(order_client.call, request)

    def call_payment_service(self, request):
        return self.payment_pool.submit(payment_client.call, request)


# 2. 进程隔离（微服务）
# 每个服务独立部署，故障不蔓延

# 3. 机房隔离
# 同城双活、异地多活
```

```
舱壁模式（Bulkhead）:

┌─────────────────────────────────────────────────────────────┐
│                        API Gateway                          │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   ┌──────────┐         ┌──────────┐         ┌──────────┐
   │ 订单服务  │         │ 用户服务  │         │ 商品服务  │
   │┌────────┐│         │┌────────┐│         │┌────────┐│
   ││线程池50 ││         ││线程池30 ││         ││线程池40 ││
   │└────────┘│         │└────────┘│         │└────────┘│
   │ 独立连接池│         │ 独立连接池│         │ 独立连接池│
   └──────────┘         └──────────┘         └──────────┘

订单服务故障时，不会耗尽其他服务的资源
```

### 3. 快速失败

```go
// 超时控制
func CallServiceWithTimeout(ctx context.Context) error {
    // 设置超时
    ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
    defer cancel()

    result := make(chan error, 1)

    go func() {
        result <- callRemoteService()
    }()

    select {
    case err := <-result:
        return err
    case <-ctx.Done():
        return errors.New("service timeout")
    }
}

// 快速失败原则
// 1. 合理的超时时间（不要太长）
// 2. 重试有上限
// 3. 失败立即返回，不阻塞
```

### 4. 优雅降级

```python
class ProductService:
    def get_product_detail(self, product_id: str) -> ProductDetail:
        """
        降级策略：
        1. 优先返回完整数据
        2. 核心服务不可用时返回缓存数据
        3. 缓存也没有时返回兜底数据
        """
        try:
            # Level 1: 完整数据
            product = self.product_repo.get(product_id)
            reviews = self.review_service.get_reviews(product_id)
            recommendations = self.recommend_service.get(product_id)

            return ProductDetail(
                product=product,
                reviews=reviews,
                recommendations=recommendations
            )
        except ReviewServiceError:
            # Level 2: 评论服务故障，返回产品+空评论
            product = self.product_repo.get(product_id)
            return ProductDetail(
                product=product,
                reviews=[],  # 降级：不展示评论
                recommendations=[]
            )
        except ProductRepoError:
            # Level 3: 数据库故障，返回缓存
            cached = self.cache.get(f"product:{product_id}")
            if cached:
                return ProductDetail(product=cached, reviews=[], recommendations=[])

            # Level 4: 缓存也没有，返回默认
            return ProductDetail(
                product=Product(id=product_id, name="商品加载中...", status="loading"),
                reviews=[],
                recommendations=[]
            )
```

### 5. 无状态设计

```
❌ 有状态服务：
┌─────────────────┐
│    Server A     │
│  Session: {     │  ← 用户 Session 存在服务器内存
│    user1: ...   │     Server A 宕机，用户数据丢失
│  }              │
└─────────────────┘

✅ 无状态服务：
┌─────────────────┐     ┌─────────────────┐
│    Server A     │     │    Server B     │
│   (无状态)       │     │   (无状态)       │
└─────────────────┘     └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
              ┌─────────────┐
              │   Redis     │  ← Session 集中存储
              │  Cluster    │     任意服务器都能处理请求
              └─────────────┘
```

## 三、故障检测与恢复

### 健康检查

```yaml
# Kubernetes 健康检查配置
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: app
    livenessProbe:        # 存活检查：失败则重启容器
      httpGet:
        path: /health/live
        port: 8080
      initialDelaySeconds: 30
      periodSeconds: 10
      failureThreshold: 3

    readinessProbe:       # 就绪检查：失败则不接收流量
      httpGet:
        path: /health/ready
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 5
      failureThreshold: 3

    startupProbe:         # 启动检查：启动期间不做存活检查
      httpGet:
        path: /health/startup
        port: 8080
      failureThreshold: 30
      periodSeconds: 10
```

```go
// 健康检查端点实现
func (s *Server) healthHandler(w http.ResponseWriter, r *http.Request) {
    checks := map[string]func() error{
        "database": s.checkDatabase,
        "redis":    s.checkRedis,
        "kafka":    s.checkKafka,
    }

    status := "healthy"
    results := make(map[string]string)

    for name, check := range checks {
        if err := check(); err != nil {
            status = "unhealthy"
            results[name] = err.Error()
        } else {
            results[name] = "ok"
        }
    }

    response := map[string]interface{}{
        "status":  status,
        "checks":  results,
        "version": s.version,
    }

    if status == "unhealthy" {
        w.WriteHeader(http.StatusServiceUnavailable)
    }

    json.NewEncoder(w).Encode(response)
}
```

### 故障转移

```
数据库故障转移（MySQL + MHA）：

正常状态：
┌─────────────┐           ┌─────────────┐
│   Master    │◀──同步──▶│   Slave 1   │
│  (写+读)    │           │   (只读)    │
└─────────────┘           └─────────────┘
                          ┌─────────────┐
                          │   Slave 2   │
                          │   (只读)    │
                          └─────────────┘

Master 故障后：
┌─────────────┐           ┌─────────────┐
│   Master    │     ╳     │  New Master │ ← Slave 1 提升
│  (故障)     │           │  (写+读)    │
└─────────────┘           └─────────────┘
                          ┌─────────────┐
                          │   Slave 2   │ ← 指向新 Master
                          │   (只读)    │
                          └─────────────┘

自动故障转移工具：
- MySQL: MHA, Orchestrator, ProxySQL
- PostgreSQL: Patroni, repmgr
- Redis: Sentinel, Cluster
```

## 四、高可用架构模式

### 1. 主从模式

```
┌───────────────────────────────────────────────────────────┐
│                      主从模式                              │
│                                                           │
│            写请求                    读请求               │
│              │                         │                  │
│              ▼                         ▼                  │
│        ┌──────────┐            ┌───────────────┐         │
│        │  Master  │───复制────▶│    Slaves     │         │
│        │  (主库)   │            │ (从库1,2,3)   │         │
│        └──────────┘            └───────────────┘         │
│                                                           │
│  优点：简单、读性能可扩展                                  │
│  缺点：主库单点、主从延迟                                  │
│  适用：读多写少、延迟不敏感                                │
└───────────────────────────────────────────────────────────┘
```

### 2. 主主模式

```
┌───────────────────────────────────────────────────────────┐
│                      主主模式                              │
│                                                           │
│        写请求 A                     写请求 B              │
│           │                            │                  │
│           ▼                            ▼                  │
│     ┌──────────┐    双向复制    ┌──────────┐            │
│     │ Master A │◀─────────────▶│ Master B │            │
│     └──────────┘                └──────────┘            │
│                                                           │
│  问题：数据冲突、循环复制                                  │
│  解决：                                                    │
│  1. 奇偶 ID（A用奇数，B用偶数）                            │
│  2. 同一时刻只有一个写入                                   │
│  3. 使用全局唯一 ID                                        │
└───────────────────────────────────────────────────────────┘
```

### 3. 集群模式

```
┌───────────────────────────────────────────────────────────┐
│                    集群模式（如Redis Cluster）             │
│                                                           │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐              │
│  │ Node 1  │    │ Node 2  │    │ Node 3  │              │
│  │ 主+从   │    │ 主+从   │    │ 主+从   │              │
│  │ Slot    │    │ Slot    │    │ Slot    │              │
│  │ 0-5460  │    │5461-10922│   │10923-16383│             │
│  └─────────┘    └─────────┘    └─────────┘              │
│                                                           │
│  特点：                                                    │
│  1. 数据分片，每个节点存储部分数据                         │
│  2. 每个主节点有从节点，故障自动切换                       │
│  3. 去中心化，无单点                                       │
└───────────────────────────────────────────────────────────┘
```

## 五、高可用检查清单

```
┌─────────────────────────────────────────────────────────────┐
│                    高可用检查清单                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 架构设计：                                                   │
│ □ 服务是否无状态？                                          │
│ □ 是否有冗余部署（至少 2 个实例）？                          │
│ □ 是否有负载均衡？                                          │
│ □ 数据库是否有主从/集群？                                    │
│ □ 缓存是否有集群/哨兵？                                      │
│                                                             │
│ 故障处理：                                                   │
│ □ 是否有超时控制？                                          │
│ □ 是否有重试机制？                                          │
│ □ 是否有熔断降级？                                          │
│ □ 是否有限流保护？                                          │
│ □ 服务间是否做了隔离？                                      │
│                                                             │
│ 监控告警：                                                   │
│ □ 是否有健康检查？                                          │
│ □ 是否有性能监控？                                          │
│ □ 是否有日志收集？                                          │
│ □ 是否有告警通知？                                          │
│ □ 是否有 On-Call 机制？                                     │
│                                                             │
│ 容灾恢复：                                                   │
│ □ 是否有数据备份？                                          │
│ □ 是否定期演练恢复？                                        │
│ □ RTO/RPO 是否达标？                                        │
│ □ 是否有应急预案？                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
