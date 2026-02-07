# 性能优化实战案例：从1K到10K QPS

## 1. 案例背景

### 1.1 系统概况

**电商订单API服务**

```
技术栈:
- 语言: Go 1.21
- 框架: Gin
- 数据库: MySQL 8.0 + Redis 7.0
- 部署: Kubernetes (3副本)
- 规格: 4C8G per pod

初始性能:
- QPS: 1000
- P50响应时间: 150ms
- P99响应时间: 800ms
- CPU使用率: 65%
- 内存使用: 3GB

目标:
- QPS: 10000 (10x)
- P99响应时间: < 200ms (4x faster)
- 错误率: < 0.01%
```

## 2. 第一轮优化 (1K → 3K QPS)

### 2.1 问题诊断

```bash
# CPU Profiling
curl http://localhost:6060/debug/pprof/profile?seconds=30 > cpu.prof
go tool pprof cpu.prof

(pprof) top10
      flat  flat%   sum%        cum   cum%
    12.50s 50.00% 50.00%     12.50s 50.00%  encoding/json.Marshal
     5.00s 20.00% 70.00%     17.50s 70.00%  handleCreateOrder
     3.00s 12.00% 82.00%      3.00s 12.00%  database/sql.(*DB).Query
```

**发现: JSON序列化是最大瓶颈**

### 2.2 优化方案1: 更快的JSON库

```go
// 优化前
import "encoding/json"

func respondJSON(c *gin.Context, data interface{}) {
    c.JSON(200, data)  // 使用标准库
}

// Benchmark
// BenchmarkStandardJSON-8    10000    115234 ns/op

// 优化后
import "github.com/bytedance/sonic"

func respondJSON(c *gin.Context, data interface{}) {
    bytes, _ := sonic.Marshal(data)
    c.Data(200, "application/json", bytes)
}

// BenchmarkSonicJSON-8       50000     23456 ns/op  (5x faster!)

// 效果:
// - CPU使用率: 65% → 45%
// - QPS: 1000 → 2000
```

### 2.3 优化方案2: 数据库连接池

```go
// 优化前
db, _ := sql.Open("mysql", dsn)
// 默认配置: MaxOpenConns=0 (无限)

// 问题: 连接创建销毁开销大

// 优化后
db, _ := sql.Open("mysql", dsn)
db.SetMaxOpenConns(100)        // 最大连接数
db.SetMaxIdleConns(50)         // 最大空闲连接
db.SetConnMaxLifetime(time.Hour)  // 连接最大存活时间

// 效果:
// - 数据库连接创建次数: 100次/秒 → 5次/秒
// - QPS: 2000 → 2500
```

### 2.4 优化方案3: 预编译SQL语句

```go
// 优化前
func getOrder(orderID string) (*Order, error) {
    query := "SELECT * FROM orders WHERE id = ?"
    row := db.QueryRow(query, orderID)
    // 每次都编译SQL
}

// 优化后
var (
    getOrderStmt *sql.Stmt
)

func init() {
    getOrderStmt, _ = db.Prepare("SELECT * FROM orders WHERE id = ?")
}

func getOrder(orderID string) (*Order, error) {
    row := getOrderStmt.QueryRow(orderID)
    // 复用预编译语句
}

// 效果:
// - SQL编译时间: 减少90%
// - QPS: 2500 → 3000
```

**第一轮总结:**
- QPS: 1000 → 3000 (3x)
- P99: 800ms → 400ms
- CPU: 65% → 50%

## 3. 第二轮优化 (3K → 6K QPS)

### 3.1 引入Redis缓存

```go
// Cache-Aside模式
type OrderCache struct {
    rdb *redis.Client
}

func (c *OrderCache) Get(orderID string) (*Order, error) {
    // 1. 查Redis
    cached, err := c.rdb.Get(ctx, "order:"+orderID).Result()
    if err == nil {
        var order Order
        json.Unmarshal([]byte(cached), &order)
        return &order, nil
    }

    // 2. 查数据库
    order, err := getOrderFromDB(orderID)
    if err != nil {
        return nil, err
    }

    // 3. 写Redis (异步)
    go func() {
        bytes, _ := json.Marshal(order)
        c.rdb.SetEX(ctx, "order:"+orderID, bytes, 10*time.Minute)
    }()

    return order, nil
}

// 效果:
// - 缓存命中率: 75%
// - 数据库QPS: 3000 → 750
// - API QPS: 3000 → 5000
```

### 3.2 批量查询优化

```go
// 优化前: N+1查询
func getOrderDetails(orderID string) (*OrderDetails, error) {
    order, _ := getOrder(orderID)

    // N次查询
    for _, itemID := range order.ItemIDs {
        item, _ := getItem(itemID)
        order.Items = append(order.Items, item)
    }
    return order, nil
}
// 问题: 100个item = 101次查询

// 优化后: IN查询
func getOrderDetails(orderID string) (*OrderDetails, error) {
    order, _ := getOrder(orderID)

    // 1次批量查询
    items, _ := getItemsBatch(order.ItemIDs)
    order.Items = items

    return order, nil
}

func getItemsBatch(ids []string) ([]*Item, error) {
    query := "SELECT * FROM items WHERE id IN (?)"
    args := strings.Join(ids, ",")
    rows, _ := db.Query(query, args)
    // 解析结果...
}

// 效果:
// - 查询次数: 101 → 2
// - P99响应时间: 400ms → 250ms
// - QPS: 5000 → 6000
```

### 3.3 并发查询

```go
// 优化前: 串行查询
func getOrderFullInfo(orderID string) (*OrderFullInfo, error) {
    order, _ := getOrder(orderID)           // 50ms
    user, _ := getUser(order.UserID)        // 30ms
    products, _ := getProducts(order.Items) // 70ms
    // 总耗时: 150ms
}

// 优化后: 并发查询
func getOrderFullInfo(orderID string) (*OrderFullInfo, error) {
    var wg sync.WaitGroup
    var order *Order
    var user *User
    var products []*Product

    wg.Add(3)

    go func() {
        defer wg.Done()
        order, _ = getOrder(orderID)
    }()

    go func() {
        defer wg.Done()
        user, _ = getUser(order.UserID)
    }()

    go func() {
        defer wg.Done()
        products, _ = getProducts(order.Items)
    }()

    wg.Wait()  // 等待最长的 (70ms)

    return &OrderFullInfo{order, user, products}, nil
}

// 效果:
// - 响应时间: 150ms → 70ms (2x faster)
// - QPS: 6000 → 6500
```

**第二轮总结:**
- QPS: 3000 → 6500 (2.2x)
- P99: 400ms → 200ms
- 数据库负载: 降低75%

## 4. 第三轮优化 (6.5K → 10K QPS)

### 4.1 对象池减少GC

```go
// 优化前: 频繁创建对象
func handleRequest(c *gin.Context) {
    req := &CreateOrderRequest{}  // 每次new
    c.BindJSON(req)

    resp := &Response{}           // 每次new
    // ...
    c.JSON(200, resp)
}

// GC压力大:
// gc 123 @45.678s 5%: 0.012+1.5+0.003 ms clock

// 优化后: 使用sync.Pool
var requestPool = sync.Pool{
    New: func() interface{} {
        return &CreateOrderRequest{}
    },
}

var responsePool = sync.Pool{
    New: func() interface{} {
        return &Response{}
    },
}

func handleRequest(c *gin.Context) {
    req := requestPool.Get().(*CreateOrderRequest)
    defer func() {
        req.Reset()
        requestPool.Put(req)
    }()

    c.BindJSON(req)

    resp := responsePool.Get().(*Response)
    defer responsePool.Put(resp)

    // 处理请求...
    c.JSON(200, resp)
}

// 效果:
// - GC频率: 降低60%
// - 堆分配: 减少40%
// - QPS: 6500 → 8000
```

### 4.2 热点数据本地缓存

```go
// 优化前: 每次查Redis
func getProduct(productID string) (*Product, error) {
    cached, _ := rdb.Get(ctx, "product:"+productID).Result()
    // 网络IO: 1-2ms
}

// 优化后: 本地缓存 + Redis
import "github.com/patrickmn/go-cache"

var (
    localCache = cache.New(5*time.Minute, 10*time.Minute)
    rdb        *redis.Client
)

func getProduct(productID string) (*Product, error) {
    // 1. 查本地缓存 (0.001ms)
    if cached, found := localCache.Get(productID); found {
        return cached.(*Product), nil
    }

    // 2. 查Redis (1ms)
    if cached, err := rdb.Get(ctx, "product:"+productID).Result(); err == nil {
        var p Product
        json.Unmarshal([]byte(cached), &p)
        localCache.Set(productID, &p, cache.DefaultExpiration)
        return &p, nil
    }

    // 3. 查数据库 (10ms)
    p, _ := getProductFromDB(productID)
    localCache.Set(productID, p, cache.DefaultExpiration)
    return p, nil
}

// 缓存层次:
// L1 (本地): 0.001ms, 命中率60%
// L2 (Redis): 1ms, 命中率35%
// L3 (DB): 10ms, 命中率5%

// 效果:
// - 平均查询时间: 1ms → 0.4ms
// - QPS: 8000 → 9500
```

### 4.3 协议优化 - HTTP/2

```yaml
# Kubernetes Ingress配置
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: order-api
  annotations:
    nginx.ingress.kubernetes.io/backend-protocol: "HTTP2"
    nginx.ingress.kubernetes.io/http2-push-preload: "true"
spec:
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: order-service
            port:
              number: 8080
```

```go
// 客户端启用HTTP/2
import "golang.org/x/net/http2"

httpClient := &http.Client{
    Transport: &http.Transport{
        MaxIdleConns:        100,
        MaxIdleConnsPerHost: 10,
    },
}
http2.ConfigureTransport(httpClient.Transport.(*http.Transport))

// 效果:
// - 连接复用率: 提升80%
// - 响应时间: 降低15%
// - QPS: 9500 → 10500
```

**第三轮总结:**
- QPS: 6500 → 10500 (1.6x)
- P99: 200ms → 120ms
- 内存: 3GB → 2GB

## 5. 关键优化总结

### 5.1 性能提升对比

```
优化项              QPS提升    响应时间    实施难度
──────────────────────────────────────────
JSON库替换          2x         -20%       低
连接池优化          1.25x      -10%       低
SQL预编译           1.2x       -5%        低
Redis缓存           1.67x      -35%       中
批量查询            1.2x       -15%       中
并发查询            1.08x      -30%       中
对象池              1.23x      -10%       中
本地缓存            1.19x      -20%       中
HTTP/2              1.11x      -15%       低
──────────────────────────────────────────
总计                10.5x      -75%       -
```

### 5.2 性能优化ROI

```
投入产出比排序 (从高到低):

1. JSON库替换
   工作量: 0.5人天
   提升: 2x QPS
   ROI: ⭐⭐⭐⭐⭐

2. 连接池配置
   工作量: 0.2人天
   提升: 1.25x QPS
   ROI: ⭐⭐⭐⭐⭐

3. Redis缓存
   工作量: 2人天
   提升: 1.67x QPS
   ROI: ⭐⭐⭐⭐

4. 对象池
   工作量: 1人天
   提升: 1.23x QPS
   ROI: ⭐⭐⭐⭐

5. 本地缓存
   工作量: 1人天
   提升: 1.19x QPS
   ROI: ⭐⭐⭐⭐
```

## 6. 其他优化案例

### 6.1 案例2: 图片服务优化 (CDN + WebP)

```
问题:
- 用户上传图片占用大量带宽
- 图片加载慢影响用户体验

优化方案:
```

```go
// 1. 图片压缩
import "github.com/disintegration/imaging"

func compressImage(src image.Image) ([]byte, error) {
    // 调整大小
    resized := imaging.Resize(src, 800, 0, imaging.Lanczos)

    // 转WebP格式 (比JPEG小30%)
    var buf bytes.Buffer
    webp.Encode(&buf, resized, &webp.Options{Quality: 80})

    return buf.Bytes(), nil
}

// 2. CDN加速
// 图片URL: https://cdn.example.com/images/abc.webp
// 源站: https://img.example.com/images/abc.webp

// Nginx配置
location /images/ {
    proxy_cache image_cache;
    proxy_cache_valid 200 30d;
    proxy_cache_key $uri;

    add_header X-Cache-Status $upstream_cache_status;
    proxy_pass http://origin_server;
}

// 效果:
// - 图片大小: 2MB → 200KB (10x)
// - CDN命中率: 95%
// - 加载时间: 3s → 0.3s
// - 带宽成本: 降低90%
```

### 6.2 案例3: 大数据查询优化 (分区表)

```sql
-- 问题: 订单表10亿行, 查询慢

-- 优化前
SELECT * FROM orders WHERE user_id = 123 AND created_at > '2024-01-01';
-- 执行时间: 15秒

-- 优化方案: 按月分区
CREATE TABLE orders (
    id BIGINT,
    user_id BIGINT,
    created_at DATETIME,
    ...
) PARTITION BY RANGE (YEAR(created_at)*100 + MONTH(created_at)) (
    PARTITION p202401 VALUES LESS THAN (202402),
    PARTITION p202402 VALUES LESS THAN (202403),
    PARTITION p202403 VALUES LESS THAN (202404),
    ...
);

-- 创建索引
CREATE INDEX idx_user_created ON orders(user_id, created_at);

-- 优化后
SELECT * FROM orders WHERE user_id = 123 AND created_at > '2024-01-01';
-- 执行时间: 0.5秒 (30x faster!)

-- 效果:
-- - 查询时间: 15s → 0.5s
-- - 扫描行数: 1亿 → 300万 (只扫描相关分区)
```

### 6.3 案例4: 消息队列削峰填谷

```go
// 问题: 秒杀场景瞬时流量10万QPS, 数据库扛不住

// 优化方案: 消息队列异步处理
import "github.com/streadway/amqp"

// 1. 接收请求 (同步, 快速返回)
func createOrder(c *gin.Context) {
    var req CreateOrderRequest
    c.BindJSON(&req)

    // 参数验证
    if err := validate(req); err != nil {
        c.JSON(400, gin.H{"error": err.Error()})
        return
    }

    // 推送到消息队列
    orderID := generateOrderID()
    msg := OrderMessage{
        OrderID: orderID,
        UserID:  req.UserID,
        Items:   req.Items,
    }
    publishToQueue("orders", msg)

    // 快速返回
    c.JSON(202, gin.H{
        "orderID": orderID,
        "status":  "processing",
    })
}

// 2. 消费者异步处理 (慢速, 稳定)
func consumeOrders() {
    msgs, _ := ch.Consume("orders", "", false, false, false, false, nil)

    for msg := range msgs {
        var order OrderMessage
        json.Unmarshal(msg.Body, &order)

        // 处理订单 (可重试)
        if err := processOrder(order); err != nil {
            msg.Nack(false, true)  // 重新入队
        } else {
            msg.Ack(false)
        }
    }
}

// 效果:
// - 接口响应时间: 1000ms → 50ms (20x)
// - 处理能力: 1000 QPS → 10000 QPS (削峰后均匀消费)
// - 系统稳定性: 提升90%
```

## 7. 性能监控看板

### 7.1 Grafana Dashboard配置

```yaml
# Prometheus采集指标
- job_name: 'order-service'
  static_configs:
    - targets: ['order-service:8080']
  metrics_path: '/metrics'

# 关键指标
# 1. QPS
rate(http_requests_total[1m])

# 2. 响应时间分位数
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[1m]))

# 3. 错误率
rate(http_requests_total{status=~"5.."}[1m]) / rate(http_requests_total[1m])

# 4. CPU使用率
rate(process_cpu_seconds_total[1m])

# 5. 内存使用
process_resident_memory_bytes
```

### 7.2 告警规则

```yaml
groups:
- name: performance_alerts
  rules:
  - alert: HighResponseTime
    expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 0.5
    for: 5m
    annotations:
      summary: "P99响应时间超过500ms"

  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.01
    for: 2m
    annotations:
      summary: "错误率超过1%"

  - alert: HighCPU
    expr: rate(process_cpu_seconds_total[5m]) > 0.8
    for: 10m
    annotations:
      summary: "CPU使用率超过80%"
```

## 8. 经验总结

### 8.1 性能优化黄金法则

```
1. 测量优先
   - 没有数据支撑不要优化
   - 建立baseline
   - 验证效果

2. 关注瓶颈
   - 80/20原则
   - 优化热点代码
   - 忽略不重要的部分

3. 权衡取舍
   - 性能 vs 可维护性
   - 复杂度 vs 收益
   - 成本 vs 价值

4. 渐进式优化
   - 先易后难
   - 快速迭代
   - 持续改进
```

### 8.2 常见性能模式

```
缓存模式:
- Cache-Aside
- Write-Through
- Write-Behind

并发模式:
- 并行查询
- 批量处理
- 流水线

资源池模式:
- 连接池
- 线程池
- 对象池

异步模式:
- 消息队列
- 事件驱动
- Future/Promise
```

## 9. 总结

从1K到10K QPS的关键:

1. **架构层面** - 缓存、分库分表、CDN
2. **代码层面** - 算法优化、并发、对象池
3. **基础设施** - HTTP/2、连接池、负载均衡
4. **监控运维** - 持续监控、快速定位、及时优化

最重要的是: **数据驱动 + 持续优化**
