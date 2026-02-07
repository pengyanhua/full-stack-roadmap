# 限流熔断降级

## 一、为什么需要限流熔断降级？

### 系统过载的连锁反应

```
正常状态:
┌─────────┐    100 QPS    ┌─────────┐    100 QPS    ┌─────────┐
│  用户    │ ───────────▶ │ 服务 A  │ ───────────▶ │ 服务 B  │
└─────────┘               └─────────┘               └─────────┘
                          响应时间: 50ms            响应时间: 30ms

过载状态 (无保护):
┌─────────┐   1000 QPS    ┌─────────┐   1000 QPS    ┌─────────┐
│  用户    │ ───────────▶ │ 服务 A  │ ───────────▶ │ 服务 B  │
└─────────┘               └─────────┘               └─────────┘
                          响应时间: 5000ms          响应时间: 3000ms
                          线程池满，拒绝连接         OOM，服务崩溃
                                │
                                ▼
                          服务 A 也崩溃 (级联故障)
```

### 三种保护机制

```
┌─────────────────────────────────────────────────────────────────┐
│                        流量保护体系                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │   限流      │    │   熔断      │    │   降级      │        │
│   │ Rate Limit  │    │  Circuit    │    │  Degrade    │        │
│   │             │    │  Breaker    │    │             │        │
│   ├─────────────┤    ├─────────────┤    ├─────────────┤        │
│   │ 控制流入    │    │ 控制流出    │    │ 降低服务    │        │
│   │ 请求速率    │    │ 快速失败    │    │ 质量换取    │        │
│   │             │    │ 防止级联    │    │ 可用性      │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                 │
│   应用场景:         应用场景:          应用场景:                │
│   - 突发流量       - 下游服务故障      - 系统过载               │
│   - 防止过载       - 网络问题          - 非核心功能             │
│   - 公平使用       - 超时积压          - 资源不足               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、限流 (Rate Limiting)

### 1. 限流算法对比

| 算法 | 原理 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| 计数器 | 固定窗口计数 | 实现简单 | 临界问题 | 简单场景 |
| 滑动窗口 | 滑动时间窗口 | 平滑限流 | 内存占用 | 精确限流 |
| 漏桶 | 固定速率流出 | 流量整形 | 无法应对突发 | 匀速处理 |
| 令牌桶 | 固定速率生成令牌 | 允许突发 | 实现复杂 | 通用场景 |

### 2. 计数器算法

```go
// ❌ 简单计数器 - 存在临界问题
type SimpleCounter struct {
    count     int64
    limit     int64
    window    time.Duration
    startTime time.Time
    mu        sync.Mutex
}

func (c *SimpleCounter) Allow() bool {
    c.mu.Lock()
    defer c.mu.Unlock()

    now := time.Now()
    // 新窗口，重置计数
    if now.Sub(c.startTime) > c.window {
        c.count = 0
        c.startTime = now
    }

    if c.count < c.limit {
        c.count++
        return true
    }
    return false
}

// 临界问题示例:
// 窗口: 1秒，限制: 100
// 0.9秒时: 100个请求 ✓
// 1.1秒时: 100个请求 ✓
// 实际 0.2秒内通过了 200个请求！
```

### 3. 滑动窗口算法

```go
// ✅ 滑动窗口 - 解决临界问题
type SlidingWindow struct {
    buckets    []int64       // 子窗口计数
    bucketSize time.Duration // 子窗口大小
    numBuckets int           // 子窗口数量
    limit      int64         // 限制
    lastBucket int           // 最后更新的桶
    lastTime   time.Time     // 最后更新时间
    mu         sync.Mutex
}

func NewSlidingWindow(window time.Duration, numBuckets int, limit int64) *SlidingWindow {
    return &SlidingWindow{
        buckets:    make([]int64, numBuckets),
        bucketSize: window / time.Duration(numBuckets),
        numBuckets: numBuckets,
        limit:      limit,
        lastTime:   time.Now(),
    }
}

func (s *SlidingWindow) Allow() bool {
    s.mu.Lock()
    defer s.mu.Unlock()

    now := time.Now()
    s.slideWindow(now)

    // 计算总数
    var total int64
    for _, count := range s.buckets {
        total += count
    }

    if total < s.limit {
        s.buckets[s.lastBucket]++
        return true
    }
    return false
}

func (s *SlidingWindow) slideWindow(now time.Time) {
    elapsed := now.Sub(s.lastTime)
    bucketsToSlide := int(elapsed / s.bucketSize)

    if bucketsToSlide > 0 {
        // 清除过期的桶
        for i := 0; i < bucketsToSlide && i < s.numBuckets; i++ {
            s.lastBucket = (s.lastBucket + 1) % s.numBuckets
            s.buckets[s.lastBucket] = 0
        }
        s.lastTime = now
    }
}
```

### 4. 漏桶算法

```go
// 漏桶算法 - 流量整形
type LeakyBucket struct {
    capacity   int64         // 桶容量
    remaining  int64         // 剩余容量
    leakRate   float64       // 漏出速率 (每秒)
    lastLeak   time.Time     // 上次漏水时间
    mu         sync.Mutex
}

func NewLeakyBucket(capacity int64, leakRate float64) *LeakyBucket {
    return &LeakyBucket{
        capacity:  capacity,
        remaining: capacity,
        leakRate:  leakRate,
        lastLeak:  time.Now(),
    }
}

func (l *LeakyBucket) Allow() bool {
    l.mu.Lock()
    defer l.mu.Unlock()

    now := time.Now()
    l.leak(now)

    if l.remaining > 0 {
        l.remaining--
        return true
    }
    return false
}

func (l *LeakyBucket) leak(now time.Time) {
    elapsed := now.Sub(l.lastLeak).Seconds()
    leaked := int64(elapsed * l.leakRate)

    if leaked > 0 {
        l.remaining = min(l.capacity, l.remaining+leaked)
        l.lastLeak = now
    }
}
```

### 5. 令牌桶算法

```go
// ✅ 令牌桶 - 推荐使用
type TokenBucket struct {
    capacity   int64         // 桶容量
    tokens     int64         // 当前令牌数
    rate       float64       // 令牌生成速率 (每秒)
    lastUpdate time.Time     // 上次更新时间
    mu         sync.Mutex
}

func NewTokenBucket(capacity int64, rate float64) *TokenBucket {
    return &TokenBucket{
        capacity:   capacity,
        tokens:     capacity,  // 初始满桶
        rate:       rate,
        lastUpdate: time.Now(),
    }
}

func (t *TokenBucket) Allow() bool {
    return t.AllowN(1)
}

func (t *TokenBucket) AllowN(n int64) bool {
    t.mu.Lock()
    defer t.mu.Unlock()

    now := time.Now()
    t.refill(now)

    if t.tokens >= n {
        t.tokens -= n
        return true
    }
    return false
}

func (t *TokenBucket) refill(now time.Time) {
    elapsed := now.Sub(t.lastUpdate).Seconds()
    newTokens := int64(elapsed * t.rate)

    if newTokens > 0 {
        t.tokens = min(t.capacity, t.tokens+newTokens)
        t.lastUpdate = now
    }
}

// 使用 Go 官方库
import "golang.org/x/time/rate"

func main() {
    // 每秒 100 个请求，允许突发 50
    limiter := rate.NewLimiter(100, 50)

    if limiter.Allow() {
        // 处理请求
    }

    // 等待直到获取令牌
    ctx, cancel := context.WithTimeout(context.Background(), time.Second)
    defer cancel()
    if err := limiter.Wait(ctx); err != nil {
        // 超时
    }
}
```

### 6. 分布式限流

```go
// Redis + Lua 实现分布式限流
const slidingWindowLua = `
local key = KEYS[1]
local window = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

-- 移除窗口外的请求
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

-- 获取当前窗口请求数
local count = redis.call('ZCARD', key)

if count < limit then
    -- 添加当前请求
    redis.call('ZADD', key, now, now .. '-' .. math.random())
    redis.call('EXPIRE', key, window / 1000 + 1)
    return 1
else
    return 0
end
`

type DistributedRateLimiter struct {
    client *redis.Client
    script *redis.Script
    key    string
    window time.Duration
    limit  int64
}

func NewDistributedRateLimiter(client *redis.Client, key string,
    window time.Duration, limit int64) *DistributedRateLimiter {
    return &DistributedRateLimiter{
        client: client,
        script: redis.NewScript(slidingWindowLua),
        key:    key,
        window: window,
        limit:  limit,
    }
}

func (r *DistributedRateLimiter) Allow(ctx context.Context) (bool, error) {
    result, err := r.script.Run(ctx, r.client,
        []string{r.key},
        r.window.Milliseconds(),
        r.limit,
        time.Now().UnixMilli(),
    ).Int()

    if err != nil {
        return false, err
    }
    return result == 1, nil
}
```

### 7. 多维度限流

```go
// 组合限流器 - 多维度保护
type MultiDimensionLimiter struct {
    global    *rate.Limiter              // 全局限流
    perUser   map[string]*rate.Limiter   // 用户级限流
    perIP     map[string]*rate.Limiter   // IP级限流
    perAPI    map[string]*rate.Limiter   // API级限流
    mu        sync.RWMutex
}

func (m *MultiDimensionLimiter) Allow(ctx context.Context,
    userID, ip, api string) bool {

    // 1. 全局限流
    if !m.global.Allow() {
        return false
    }

    // 2. 用户限流
    if userLimiter := m.getUserLimiter(userID); userLimiter != nil {
        if !userLimiter.Allow() {
            return false
        }
    }

    // 3. IP 限流
    if ipLimiter := m.getIPLimiter(ip); ipLimiter != nil {
        if !ipLimiter.Allow() {
            return false
        }
    }

    // 4. API 限流
    if apiLimiter := m.getAPILimiter(api); apiLimiter != nil {
        if !apiLimiter.Allow() {
            return false
        }
    }

    return true
}

// 配置示例
var limitConfig = map[string]struct {
    Rate  float64
    Burst int
}{
    "global":         {10000, 1000},     // 全局: 10000 QPS
    "user:normal":    {100, 20},         // 普通用户: 100 QPS
    "user:vip":       {1000, 100},       // VIP用户: 1000 QPS
    "ip:default":     {50, 10},          // 默认IP: 50 QPS
    "api:/api/login": {10, 5},           // 登录接口: 10 QPS
    "api:/api/order": {100, 20},         // 订单接口: 100 QPS
}
```

---

## 三、熔断 (Circuit Breaker)

### 1. 熔断器状态机

```
        ┌──────────────────────────────────────────────────────────┐
        │                     熔断器状态转换                         │
        └──────────────────────────────────────────────────────────┘

                    失败率 > 阈值
        ┌───────┐ ──────────────────▶ ┌───────┐
        │ 关闭  │                      │ 打开  │
        │ CLOSED│ ◀────────────────── │ OPEN  │
        └───────┘    探测请求成功      └───────┘
            ▲                              │
            │                              │ 超时后
            │        ┌──────────┐          │
            └─────── │ 半开     │ ◀────────┘
              成功   │HALF-OPEN │
                     └──────────┘
                          │
                          │ 失败
                          ▼
                     回到 OPEN

状态说明:
- CLOSED: 正常状态，请求正常通过
- OPEN:   熔断状态，请求快速失败
- HALF_OPEN: 探测状态，允许少量请求通过测试
```

### 2. 熔断器实现

```go
type State int

const (
    StateClosed State = iota
    StateOpen
    StateHalfOpen
)

type CircuitBreaker struct {
    name            string
    state           State
    failureCount    int64
    successCount    int64
    failureThreshold int64         // 失败阈值
    successThreshold int64         // 半开状态成功阈值
    timeout         time.Duration  // 熔断持续时间
    lastFailure     time.Time      // 最后失败时间
    mu              sync.RWMutex

    // 统计窗口
    windowSize      time.Duration
    windowStart     time.Time
    windowRequests  int64
    windowFailures  int64
}

func NewCircuitBreaker(name string, opts ...Option) *CircuitBreaker {
    cb := &CircuitBreaker{
        name:             name,
        state:            StateClosed,
        failureThreshold: 5,           // 默认5次失败
        successThreshold: 3,           // 半开状态3次成功
        timeout:          30 * time.Second,
        windowSize:       10 * time.Second,
        windowStart:      time.Now(),
    }
    for _, opt := range opts {
        opt(cb)
    }
    return cb
}

func (cb *CircuitBreaker) Execute(fn func() error) error {
    if !cb.allowRequest() {
        return ErrCircuitOpen
    }

    err := fn()
    cb.recordResult(err)
    return err
}

func (cb *CircuitBreaker) allowRequest() bool {
    cb.mu.Lock()
    defer cb.mu.Unlock()

    switch cb.state {
    case StateClosed:
        return true

    case StateOpen:
        // 检查是否超时，可以尝试半开
        if time.Since(cb.lastFailure) > cb.timeout {
            cb.state = StateHalfOpen
            cb.successCount = 0
            return true
        }
        return false

    case StateHalfOpen:
        // 半开状态限制请求数
        return true
    }

    return false
}

func (cb *CircuitBreaker) recordResult(err error) {
    cb.mu.Lock()
    defer cb.mu.Unlock()

    // 重置过期的统计窗口
    if time.Since(cb.windowStart) > cb.windowSize {
        cb.windowStart = time.Now()
        cb.windowRequests = 0
        cb.windowFailures = 0
    }

    cb.windowRequests++

    if err != nil {
        cb.windowFailures++
        cb.onFailure()
    } else {
        cb.onSuccess()
    }
}

func (cb *CircuitBreaker) onFailure() {
    cb.lastFailure = time.Now()

    switch cb.state {
    case StateClosed:
        // 计算失败率
        if cb.windowRequests >= 10 { // 最小请求数
            failureRate := float64(cb.windowFailures) / float64(cb.windowRequests)
            if failureRate >= 0.5 { // 失败率 >= 50%
                cb.state = StateOpen
                log.Printf("[CircuitBreaker] %s: CLOSED -> OPEN", cb.name)
            }
        }

    case StateHalfOpen:
        // 半开状态下任何失败都回到打开状态
        cb.state = StateOpen
        log.Printf("[CircuitBreaker] %s: HALF_OPEN -> OPEN", cb.name)
    }
}

func (cb *CircuitBreaker) onSuccess() {
    switch cb.state {
    case StateHalfOpen:
        cb.successCount++
        if cb.successCount >= cb.successThreshold {
            cb.state = StateClosed
            cb.windowFailures = 0
            cb.windowRequests = 0
            log.Printf("[CircuitBreaker] %s: HALF_OPEN -> CLOSED", cb.name)
        }
    }
}

// 使用示例
var breaker = NewCircuitBreaker("payment-service",
    WithFailureThreshold(5),
    WithTimeout(30*time.Second),
)

func CallPaymentService() error {
    return breaker.Execute(func() error {
        resp, err := http.Get("http://payment-service/pay")
        if err != nil {
            return err
        }
        if resp.StatusCode >= 500 {
            return errors.New("server error")
        }
        return nil
    })
}
```

### 3. 使用 sony/gobreaker

```go
import "github.com/sony/gobreaker"

func main() {
    settings := gobreaker.Settings{
        Name:        "payment-service",
        MaxRequests: 3,                    // 半开状态最大请求数
        Interval:    10 * time.Second,     // 统计窗口
        Timeout:     30 * time.Second,     // 熔断持续时间
        ReadyToTrip: func(counts gobreaker.Counts) bool {
            // 自定义熔断条件
            failureRatio := float64(counts.TotalFailures) / float64(counts.Requests)
            return counts.Requests >= 10 && failureRatio >= 0.5
        },
        OnStateChange: func(name string, from, to gobreaker.State) {
            log.Printf("[CircuitBreaker] %s: %s -> %s", name, from, to)
        },
    }

    cb := gobreaker.NewCircuitBreaker(settings)

    result, err := cb.Execute(func() (interface{}, error) {
        return callPaymentService()
    })
}
```

### 4. 熔断器的坑点

```go
// ❌ 坑点 1: 所有错误都触发熔断
func (cb *CircuitBreaker) Execute(fn func() error) error {
    err := fn()
    if err != nil {
        cb.recordFailure()  // 业务错误也触发熔断
    }
    return err
}

// ✅ 正确: 区分系统错误和业务错误
type CircuitError struct {
    Err        error
    ShouldTrip bool  // 是否应该触发熔断
}

func (cb *CircuitBreaker) Execute(fn func() error) error {
    err := fn()
    if err != nil {
        var ce *CircuitError
        if errors.As(err, &ce) && ce.ShouldTrip {
            cb.recordFailure()
        }
        // 业务错误（如参数错误）不触发熔断
    }
    return err
}

// ❌ 坑点 2: 熔断器粒度不当
var globalBreaker = NewCircuitBreaker("all-services")  // 粒度太大

// ✅ 正确: 按服务/接口粒度
var breakers = map[string]*CircuitBreaker{
    "payment-service":     NewCircuitBreaker("payment-service"),
    "inventory-service":   NewCircuitBreaker("inventory-service"),
    "user-service:/login": NewCircuitBreaker("user-service:/login"),
}

// ❌ 坑点 3: 忘记处理熔断后的请求
func handler(w http.ResponseWriter, r *http.Request) {
    err := breaker.Execute(func() error {
        return callDownstream()
    })
    if err == gobreaker.ErrOpenState {
        // 熔断了，但没有降级处理
        http.Error(w, "service unavailable", 503)
    }
}

// ✅ 正确: 熔断后降级处理
func handler(w http.ResponseWriter, r *http.Request) {
    result, err := breaker.Execute(func() (interface{}, error) {
        return callDownstream()
    })

    if err == gobreaker.ErrOpenState {
        // 降级：返回缓存/默认值
        result = getCachedResult()
    }

    json.NewEncoder(w).Encode(result)
}
```

---

## 四、降级 (Degradation)

### 1. 降级策略

```
┌─────────────────────────────────────────────────────────────────┐
│                       降级策略类型                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   返回降级数据   │  │   功能降级      │  │   服务降级      │ │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤ │
│  │ • 缓存数据      │  │ • 关闭非核心    │  │ • 同步转异步    │ │
│  │ • 静态数据      │  │ • 简化流程      │  │ • 精简返回      │ │
│  │ • 默认值        │  │ • 降低精度      │  │ • 队列排队      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                 │
│  示例:                 示例:               示例:                │
│  推荐列表返回热门      关闭评论功能        下单成功但延迟发货   │
│  搜索返回缓存结果      简化风控校验        异步处理积分         │
│  库存返回"有货"        降低实时性          排队支付             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 降级实现

```go
// 降级配置
type DegradeConfig struct {
    Enabled     bool              // 是否启用降级
    Level       int               // 降级级别 1-3
    Features    map[string]bool   // 功能开关
    Strategy    DegradeStrategy   // 降级策略
}

type DegradeStrategy interface {
    Degrade(ctx context.Context, req interface{}) (interface{}, error)
}

// 返回缓存策略
type CacheDegradeStrategy struct {
    cache   Cache
    ttl     time.Duration
}

func (s *CacheDegradeStrategy) Degrade(ctx context.Context,
    req interface{}) (interface{}, error) {
    key := generateKey(req)
    if data, ok := s.cache.Get(key); ok {
        return data, nil
    }
    return nil, ErrNoCache
}

// 返回默认值策略
type DefaultValueStrategy struct {
    defaultValue interface{}
}

func (s *DefaultValueStrategy) Degrade(ctx context.Context,
    req interface{}) (interface{}, error) {
    return s.defaultValue, nil
}

// 降级服务
type DegradeService struct {
    config    *DegradeConfig
    strategy  DegradeStrategy
}

func (s *DegradeService) Execute(ctx context.Context,
    normalFn func() (interface{}, error)) (interface{}, error) {

    // 检查是否需要降级
    if s.shouldDegrade(ctx) {
        return s.strategy.Degrade(ctx, nil)
    }

    // 正常执行
    result, err := normalFn()
    if err != nil {
        // 执行失败也可以降级
        return s.strategy.Degrade(ctx, nil)
    }

    return result, nil
}

func (s *DegradeService) shouldDegrade(ctx context.Context) bool {
    // 1. 手动开关
    if s.config.Enabled {
        return true
    }

    // 2. 系统负载判断
    if systemLoad() > 0.9 {
        return true
    }

    // 3. 错误率判断
    if errorRate() > 0.5 {
        return true
    }

    return false
}
```

### 3. 多级降级

```go
// 多级降级示例：商品详情页
type ProductDetailService struct {
    productService   *ProductService
    reviewService    *ReviewService
    recommendService *RecommendService
    cache            Cache
    degradeLevel     int  // 0=正常, 1=轻度, 2=中度, 3=重度
}

func (s *ProductDetailService) GetProductDetail(productID string) (*ProductDetail, error) {
    detail := &ProductDetail{}

    // 核心信息：始终获取
    product, err := s.getProductWithFallback(productID)
    if err != nil {
        return nil, err  // 核心信息获取失败，直接返回错误
    }
    detail.Product = product

    // 级别 1: 评论降级
    if s.degradeLevel < 1 {
        reviews, err := s.reviewService.GetReviews(productID)
        if err == nil {
            detail.Reviews = reviews
        } else {
            // 降级：显示缓存的评论或"暂无评论"
            detail.Reviews = s.getCachedReviews(productID)
        }
    } else {
        detail.Reviews = nil  // 完全不显示评论
    }

    // 级别 2: 推荐降级
    if s.degradeLevel < 2 {
        recommends, err := s.recommendService.GetRecommends(productID)
        if err == nil {
            detail.Recommends = recommends
        } else {
            // 降级：显示热门商品
            detail.Recommends = s.getHotProducts()
        }
    } else {
        detail.Recommends = nil  // 完全不显示推荐
    }

    // 级别 3: 只返回基础信息
    if s.degradeLevel >= 3 {
        return &ProductDetail{
            Product: &BasicProduct{
                ID:    product.ID,
                Name:  product.Name,
                Price: product.Price,
            },
        }, nil
    }

    return detail, nil
}

func (s *ProductDetailService) getProductWithFallback(productID string) (*Product, error) {
    // 尝试从服务获取
    product, err := s.productService.GetProduct(productID)
    if err == nil {
        // 更新缓存
        s.cache.Set("product:"+productID, product, 5*time.Minute)
        return product, nil
    }

    // 降级：从缓存获取
    if cached, ok := s.cache.Get("product:" + productID); ok {
        return cached.(*Product), nil
    }

    return nil, err
}
```

### 4. 开关降级

```go
// 功能开关
type FeatureToggle struct {
    store  FeatureStore  // Redis/Nacos/Apollo
    cache  sync.Map      // 本地缓存
}

func (f *FeatureToggle) IsEnabled(feature string) bool {
    // 优先本地缓存
    if v, ok := f.cache.Load(feature); ok {
        return v.(bool)
    }

    // 从配置中心获取
    enabled := f.store.Get(feature)
    f.cache.Store(feature, enabled)
    return enabled
}

// 使用示例
var toggle = &FeatureToggle{}

func handler(w http.ResponseWriter, r *http.Request) {
    // 运营活动开关
    if toggle.IsEnabled("promotion_banner") {
        showPromotionBanner()
    }

    // 新功能灰度
    if toggle.IsEnabled("new_checkout_flow") {
        newCheckoutFlow()
    } else {
        oldCheckoutFlow()
    }

    // 紧急降级开关
    if toggle.IsEnabled("degrade_recommendation") {
        showStaticRecommendation()
    } else {
        showDynamicRecommendation()
    }
}

// 通过配置中心动态调整
// Apollo/Nacos 配置:
// {
//     "promotion_banner": true,
//     "new_checkout_flow": false,
//     "degrade_recommendation": false
// }
```

---

## 五、Sentinel 实战

### 1. Sentinel 核心概念

```
┌─────────────────────────────────────────────────────────────────┐
│                     Sentinel 架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                    Sentinel Core                         │  │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │  │
│   │  │ 流量控制 │ │ 熔断降级 │ │ 系统保护 │ │ 热点参数 │   │  │
│   │  │  Flow    │ │ Degrade  │ │  System  │ │ ParamFlow│   │  │
│   │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │  │
│   │              ↑                                          │  │
│   │         ┌────┴────┐                                    │  │
│   │         │  Slot   │ ← 责任链模式                        │  │
│   │         │  Chain  │                                    │  │
│   │         └─────────┘                                    │  │
│   └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                    Dashboard                             │  │
│   │               (可视化管理控制台)                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Go 版本 Sentinel

```go
import (
    sentinel "github.com/alibaba/sentinel-golang/api"
    "github.com/alibaba/sentinel-golang/core/base"
    "github.com/alibaba/sentinel-golang/core/flow"
    "github.com/alibaba/sentinel-golang/core/circuitbreaker"
)

func main() {
    // 初始化 Sentinel
    err := sentinel.InitDefault()
    if err != nil {
        log.Fatal(err)
    }

    // 配置流控规则
    _, err = flow.LoadRules([]*flow.Rule{
        {
            Resource:               "order-service",
            Threshold:              100,                    // QPS 阈值
            TokenCalculateStrategy: flow.Direct,           // 直接计算
            ControlBehavior:        flow.Reject,           // 拒绝策略
            StatIntervalInMs:       1000,                  // 统计窗口 1s
        },
        {
            Resource:               "payment-service",
            Threshold:              50,
            TokenCalculateStrategy: flow.Direct,
            ControlBehavior:        flow.Throttling,       // 匀速排队
            MaxQueueingTimeMs:      500,                   // 最大排队时间
        },
    })

    // 配置熔断规则
    _, err = circuitbreaker.LoadRules([]*circuitbreaker.Rule{
        {
            Resource:                     "inventory-service",
            Strategy:                     circuitbreaker.SlowRequestRatio,
            RetryTimeoutMs:              5000,              // 熔断恢复时间
            MinRequestAmount:            10,                // 最小请求数
            StatIntervalMs:              10000,             // 统计窗口
            MaxAllowedRtMs:              100,               // 慢调用阈值(ms)
            Threshold:                   0.5,               // 慢调用比例阈值
        },
        {
            Resource:                     "user-service",
            Strategy:                     circuitbreaker.ErrorRatio,
            RetryTimeoutMs:              10000,
            MinRequestAmount:            10,
            StatIntervalMs:              10000,
            Threshold:                   0.5,               // 错误率阈值
        },
    })
}

// 在业务代码中使用
func CallOrderService() error {
    entry, blockErr := sentinel.Entry("order-service")
    if blockErr != nil {
        // 被限流/熔断
        return handleBlock(blockErr)
    }
    defer entry.Exit()

    // 正常业务逻辑
    return doBusinessLogic()
}

func handleBlock(blockErr *base.BlockError) error {
    switch blockErr.BlockType() {
    case base.BlockTypeFlow:
        return errors.New("rate limited")
    case base.BlockTypeCircuitBreaking:
        return errors.New("circuit breaker open")
    default:
        return errors.New("blocked")
    }
}
```

### 3. HTTP 中间件集成

```go
import (
    sentinelPlugin "github.com/alibaba/sentinel-golang/pkg/adapters/gin"
    "github.com/gin-gonic/gin"
)

func main() {
    r := gin.New()

    // 使用 Sentinel 中间件
    r.Use(sentinelPlugin.SentinelMiddleware(
        // 自定义资源名提取
        sentinelPlugin.WithResourceExtractor(func(c *gin.Context) string {
            return c.Request.Method + ":" + c.FullPath()
        }),
        // 自定义 Block 处理
        sentinelPlugin.WithBlockFallback(func(c *gin.Context) {
            c.JSON(429, gin.H{
                "code":    429,
                "message": "Too many requests",
            })
        }),
    ))

    r.GET("/api/orders", getOrders)
    r.POST("/api/orders", createOrder)

    r.Run(":8080")
}
```

---

## 六、实战检查清单

### 限流检查

- [ ] 选择了合适的限流算法？
- [ ] 限流阈值是否通过压测确定？
- [ ] 是否实现了多维度限流（用户/IP/API）？
- [ ] 分布式限流是否考虑了 Redis 故障？
- [ ] 限流后的响应是否友好？

### 熔断检查

- [ ] 熔断器粒度是否合适？
- [ ] 是否区分了系统错误和业务错误？
- [ ] 熔断阈值和恢复时间是否合理？
- [ ] 熔断后是否有降级方案？
- [ ] 是否有熔断状态监控和告警？

### 降级检查

- [ ] 是否识别了核心和非核心功能？
- [ ] 降级策略是否可以快速切换？
- [ ] 降级数据是否有兜底方案？
- [ ] 降级开关是否支持动态调整？
- [ ] 是否定期演练降级流程？

### 整体架构检查

```
                    请求入口
                       │
                       ▼
              ┌────────────────┐
              │   API 网关     │ ← 全局限流、黑白名单
              └────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │   负载均衡     │ ← 健康检查、流量调度
              └────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │   服务实例     │ ← 接口限流、熔断
              │                │
              │  ┌──────────┐  │
              │  │ 限流器   │  │
              │  └──────────┘  │
              │  ┌──────────┐  │
              │  │ 熔断器   │  │
              │  └──────────┘  │
              │  ┌──────────┐  │
              │  │ 降级开关 │  │
              │  └──────────┘  │
              └────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │   下游服务     │
              └────────────────┘
```
