# 池化技术

## 一、池化概述

### 为什么需要池化？

```
┌─────────────────────────────────────────────────────────────────┐
│                    池化技术解决的问题                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   问题 1: 频繁创建销毁的开销                                     │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   无池化:    创建 → 使用 → 销毁 → 创建 → 使用 → 销毁     │  │
│   │              ↑        ↑          ↑                       │  │
│   │            开销大    可用      开销大                      │  │
│   │                                                          │  │
│   │   有池化:    创建 → 使用 → 归还 → 获取 → 使用 → 归还     │  │
│   │              ↑                    ↑                       │  │
│   │          只需一次               几乎无开销                  │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   问题 2: 资源数量无法控制                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   无池化: 高并发时可能创建过多连接，导致资源耗尽            │  │
│   │                                                          │  │
│   │   有池化: 限制最大资源数量，超出时排队等待                  │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   适用池化的资源:                                                │
│   - 数据库连接 (建立 TCP + 认证 = 几十~几百ms)                   │
│   - HTTP 连接 (TCP 三次握手 + TLS 握手)                         │
│   - 线程/协程 (创建线程需要分配栈空间)                           │
│   - 对象实例 (大对象分配 + GC 压力)                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 池化模式对比

```
┌─────────────────────────────────────────────────────────────────┐
│                      池化模式对比                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. 固定大小池                                                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐                         │  │
│   │  │ 1 │ │ 2 │ │ 3 │ │ 4 │ │ 5 │  固定 5 个资源          │  │
│   │  └───┘ └───┘ └───┘ └───┘ └───┘                         │  │
│   │                                                          │  │
│   │  优点: 简单、资源可控                                     │  │
│   │  缺点: 无法应对突发流量                                   │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   2. 弹性池                                                      │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  核心数量: ┌───┐ ┌───┐ ┌───┐                            │  │
│   │           │ 1 │ │ 2 │ │ 3 │                            │  │
│   │           └───┘ └───┘ └───┘                            │  │
│   │                                                          │  │
│   │  弹性扩展: ┌───┐ ┌───┐                                  │  │
│   │           │ 4 │ │ 5 │  ← 高负载时创建                   │  │
│   │           └───┘ └───┘  → 空闲时销毁                     │  │
│   │                                                          │  │
│   │  优点: 灵活、资源利用率高                                 │  │
│   │  缺点: 实现复杂                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   3. 分片池                                                      │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │     Shard 0        Shard 1        Shard 2               │  │
│   │   ┌───┬───┬───┐  ┌───┬───┬───┐  ┌───┬───┬───┐         │  │
│   │   │ 1 │ 2 │ 3 │  │ 4 │ 5 │ 6 │  │ 7 │ 8 │ 9 │         │  │
│   │   └───┴───┴───┘  └───┴───┴───┘  └───┴───┴───┘         │  │
│   │                                                          │  │
│   │  根据 key hash 到不同分片，减少锁竞争                     │  │
│   │  优点: 高并发性能好                                       │  │
│   │  缺点: 实现复杂                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、对象池

### 1. sync.Pool 原理与使用

```go
// sync.Pool 基本使用
var bufferPool = sync.Pool{
    New: func() interface{} {
        return make([]byte, 4096)
    },
}

func processData(data []byte) {
    // 从池中获取
    buf := bufferPool.Get().([]byte)

    // 使用完后归还
    defer func() {
        // 清理状态后归还
        buf = buf[:0]
        bufferPool.Put(buf)
    }()

    // 使用 buf 处理数据
    copy(buf, data)
    // ...
}

// sync.Pool 特性:
// 1. 对象可能被 GC 回收（不保证持久化）
// 2. 每个 P 有私有池 + 共享池
// 3. Get 时: 私有池 → 共享池 → 其他 P 的共享池 → New
// 4. Put 时: 放入私有池
```

### 2. 自定义对象池

```go
// 带容量限制的对象池
type BoundedPool struct {
    pool    chan interface{}
    factory func() interface{}
}

func NewBoundedPool(capacity int, factory func() interface{}) *BoundedPool {
    return &BoundedPool{
        pool:    make(chan interface{}, capacity),
        factory: factory,
    }
}

func (p *BoundedPool) Get() interface{} {
    select {
    case obj := <-p.pool:
        return obj
    default:
        return p.factory()
    }
}

func (p *BoundedPool) Put(obj interface{}) {
    select {
    case p.pool <- obj:
        // 放入池中
    default:
        // 池满，丢弃对象
    }
}

// 带生命周期管理的对象池
type ManagedPool struct {
    mu       sync.Mutex
    idle     []*poolObject
    active   int
    maxSize  int
    factory  func() (interface{}, error)
    validate func(interface{}) bool
    maxAge   time.Duration
}

type poolObject struct {
    obj       interface{}
    createdAt time.Time
}

func (p *ManagedPool) Get(ctx context.Context) (interface{}, error) {
    p.mu.Lock()

    // 尝试从空闲池获取
    for len(p.idle) > 0 {
        po := p.idle[len(p.idle)-1]
        p.idle = p.idle[:len(p.idle)-1]

        // 检查对象是否过期
        if time.Since(po.createdAt) > p.maxAge {
            continue  // 过期对象丢弃
        }

        // 验证对象是否可用
        if p.validate != nil && !p.validate(po.obj) {
            continue
        }

        p.active++
        p.mu.Unlock()
        return po.obj, nil
    }

    // 检查是否可以创建新对象
    if p.active >= p.maxSize {
        p.mu.Unlock()
        return nil, errors.New("pool exhausted")
    }

    p.active++
    p.mu.Unlock()

    // 创建新对象
    return p.factory()
}

func (p *ManagedPool) Put(obj interface{}) {
    p.mu.Lock()
    defer p.mu.Unlock()

    p.active--

    // 验证后放入空闲池
    if p.validate == nil || p.validate(obj) {
        p.idle = append(p.idle, &poolObject{
            obj:       obj,
            createdAt: time.Now(),
        })
    }
}
```

### 3. bytes.Buffer 池

```go
// 高性能 bytes.Buffer 池
type BufferPool struct {
    pool sync.Pool
}

func NewBufferPool(initialCap int) *BufferPool {
    return &BufferPool{
        pool: sync.Pool{
            New: func() interface{} {
                return bytes.NewBuffer(make([]byte, 0, initialCap))
            },
        },
    }
}

func (p *BufferPool) Get() *bytes.Buffer {
    return p.pool.Get().(*bytes.Buffer)
}

func (p *BufferPool) Put(buf *bytes.Buffer) {
    buf.Reset()
    p.pool.Put(buf)
}

// 使用示例
var jsonBufferPool = NewBufferPool(1024)

func marshalJSON(v interface{}) ([]byte, error) {
    buf := jsonBufferPool.Get()
    defer jsonBufferPool.Put(buf)

    encoder := json.NewEncoder(buf)
    if err := encoder.Encode(v); err != nil {
        return nil, err
    }

    // 注意: 返回前必须复制，因为 buf 会被重用
    result := make([]byte, buf.Len())
    copy(result, buf.Bytes())
    return result, nil
}
```

---

## 三、连接池

### 1. 通用连接池

```go
// 通用连接池实现
type ConnectionPool struct {
    mu          sync.Mutex
    idle        []*idleConn
    numOpen     int
    maxOpen     int
    maxIdle     int
    maxLifetime time.Duration
    maxIdleTime time.Duration
    factory     func() (Conn, error)
    waiting     []chan connRequest
}

type Conn interface {
    Close() error
    Ping() error
}

type idleConn struct {
    conn      Conn
    createdAt time.Time
    idleAt    time.Time
}

type connRequest struct {
    conn Conn
    err  error
}

func NewConnectionPool(opts PoolOptions) *ConnectionPool {
    pool := &ConnectionPool{
        maxOpen:     opts.MaxOpen,
        maxIdle:     opts.MaxIdle,
        maxLifetime: opts.MaxLifetime,
        maxIdleTime: opts.MaxIdleTime,
        factory:     opts.Factory,
    }

    // 启动清理协程
    go pool.cleaner()

    return pool
}

func (p *ConnectionPool) Get(ctx context.Context) (Conn, error) {
    p.mu.Lock()

    // 1. 尝试从空闲连接获取
    for len(p.idle) > 0 {
        ic := p.idle[len(p.idle)-1]
        p.idle = p.idle[:len(p.idle)-1]

        // 检查连接是否过期
        if p.isExpired(ic) {
            ic.conn.Close()
            p.numOpen--
            continue
        }

        p.mu.Unlock()

        // 验证连接是否可用
        if err := ic.conn.Ping(); err != nil {
            ic.conn.Close()
            p.mu.Lock()
            p.numOpen--
            p.mu.Unlock()
            continue
        }

        return ic.conn, nil
    }

    // 2. 检查是否可以创建新连接
    if p.numOpen < p.maxOpen {
        p.numOpen++
        p.mu.Unlock()

        conn, err := p.factory()
        if err != nil {
            p.mu.Lock()
            p.numOpen--
            p.mu.Unlock()
            return nil, err
        }
        return conn, nil
    }

    // 3. 等待连接释放
    req := make(chan connRequest, 1)
    p.waiting = append(p.waiting, req)
    p.mu.Unlock()

    select {
    case result := <-req:
        return result.conn, result.err
    case <-ctx.Done():
        p.mu.Lock()
        // 从等待队列移除
        for i, r := range p.waiting {
            if r == req {
                p.waiting = append(p.waiting[:i], p.waiting[i+1:]...)
                break
            }
        }
        p.mu.Unlock()
        return nil, ctx.Err()
    }
}

func (p *ConnectionPool) Put(conn Conn) {
    p.mu.Lock()
    defer p.mu.Unlock()

    // 如果有等待者，直接给等待者
    if len(p.waiting) > 0 {
        req := p.waiting[0]
        p.waiting = p.waiting[1:]
        req <- connRequest{conn: conn}
        return
    }

    // 放入空闲池
    if len(p.idle) < p.maxIdle {
        p.idle = append(p.idle, &idleConn{
            conn:      conn,
            createdAt: time.Now(),
            idleAt:    time.Now(),
        })
        return
    }

    // 池满，关闭连接
    conn.Close()
    p.numOpen--
}

func (p *ConnectionPool) isExpired(ic *idleConn) bool {
    now := time.Now()

    // 检查连接生命周期
    if p.maxLifetime > 0 && now.Sub(ic.createdAt) > p.maxLifetime {
        return true
    }

    // 检查空闲时间
    if p.maxIdleTime > 0 && now.Sub(ic.idleAt) > p.maxIdleTime {
        return true
    }

    return false
}

func (p *ConnectionPool) cleaner() {
    ticker := time.NewTicker(time.Minute)
    for range ticker.C {
        p.mu.Lock()

        // 清理过期连接
        valid := make([]*idleConn, 0, len(p.idle))
        for _, ic := range p.idle {
            if p.isExpired(ic) {
                ic.conn.Close()
                p.numOpen--
            } else {
                valid = append(valid, ic)
            }
        }
        p.idle = valid

        p.mu.Unlock()
    }
}
```

### 2. 数据库连接池最佳实践

```go
import (
    "database/sql"
    "time"
)

type DBConfig struct {
    DSN             string
    MaxOpenConns    int
    MaxIdleConns    int
    ConnMaxLifetime time.Duration
    ConnMaxIdleTime time.Duration
}

func NewDBPool(cfg DBConfig) (*sql.DB, error) {
    db, err := sql.Open("mysql", cfg.DSN)
    if err != nil {
        return nil, err
    }

    // 连接池参数配置
    db.SetMaxOpenConns(cfg.MaxOpenConns)       // 最大打开连接数
    db.SetMaxIdleConns(cfg.MaxIdleConns)       // 最大空闲连接数
    db.SetConnMaxLifetime(cfg.ConnMaxLifetime) // 连接最大生命周期
    db.SetConnMaxIdleTime(cfg.ConnMaxIdleTime) // 空闲连接最大生命周期

    // 验证连接
    if err := db.Ping(); err != nil {
        return nil, err
    }

    return db, nil
}

// 配置建议
// MaxOpenConns: 根据数据库配置和业务需求，一般 20-100
// MaxIdleConns: MaxOpenConns 的 10%-25%
// ConnMaxLifetime: 小于数据库 wait_timeout，建议 5-30 分钟
// ConnMaxIdleTime: 建议 5-10 分钟

// 计算公式（经验值）:
// MaxOpenConns ≈ (CPU核数 * 2) + 有效磁盘数
// 例: 8核CPU + 1块SSD → 约 17-20 连接

// 监控连接池状态
func MonitorDBPool(db *sql.DB) {
    go func() {
        ticker := time.NewTicker(30 * time.Second)
        for range ticker.C {
            stats := db.Stats()
            metrics.Gauge("db.open_connections", float64(stats.OpenConnections))
            metrics.Gauge("db.idle", float64(stats.Idle))
            metrics.Gauge("db.in_use", float64(stats.InUse))
            metrics.Counter("db.wait_count", float64(stats.WaitCount))
            metrics.Gauge("db.wait_duration_ms", float64(stats.WaitDuration.Milliseconds()))
        }
    }()
}
```

---

## 四、线程/协程池

### 1. Goroutine 池

```go
// 高性能 Goroutine 池
type GoPool struct {
    workers   int
    taskQueue chan func()
    wg        sync.WaitGroup
}

func NewGoPool(workers int) *GoPool {
    pool := &GoPool{
        workers:   workers,
        taskQueue: make(chan func(), workers*10),
    }

    for i := 0; i < workers; i++ {
        go pool.worker()
    }

    return pool
}

func (p *GoPool) worker() {
    for task := range p.taskQueue {
        task()
        p.wg.Done()
    }
}

func (p *GoPool) Submit(task func()) {
    p.wg.Add(1)
    p.taskQueue <- task
}

func (p *GoPool) Wait() {
    p.wg.Wait()
}

func (p *GoPool) Shutdown() {
    close(p.taskQueue)
}

// 使用示例
func main() {
    pool := NewGoPool(runtime.NumCPU())
    defer pool.Shutdown()

    for i := 0; i < 1000; i++ {
        n := i
        pool.Submit(func() {
            fmt.Println(n)
        })
    }

    pool.Wait()
}
```

### 2. 使用 ants 库

```go
import "github.com/panjf2000/ants/v2"

func main() {
    // 创建固定大小的协程池
    pool, _ := ants.NewPool(1000,
        ants.WithPreAlloc(true),                    // 预分配
        ants.WithExpiryDuration(10*time.Second),   // 空闲过期时间
        ants.WithPanicHandler(func(i interface{}) {
            log.Printf("panic: %v", i)
        }),
    )
    defer pool.Release()

    // 提交任务
    for i := 0; i < 10000; i++ {
        n := i
        _ = pool.Submit(func() {
            time.Sleep(10 * time.Millisecond)
            fmt.Println(n)
        })
    }

    // 等待任务完成
    for pool.Running() > 0 {
        time.Sleep(100 * time.Millisecond)
    }
}

// 带参数的任务池
func withArgs() {
    pool, _ := ants.NewPoolWithFunc(1000, func(i interface{}) {
        n := i.(int)
        fmt.Println(n * 2)
    })
    defer pool.Release()

    for i := 0; i < 1000; i++ {
        _ = pool.Invoke(i)
    }
}
```

---

## 五、池化最佳实践

### 1. 常见坑点

```go
// ❌ 坑点 1: sync.Pool 对象被 GC 回收
var pool = sync.Pool{
    New: func() interface{} {
        return &BigObject{data: make([]byte, 1<<20)}  // 1MB
    },
}

func bad() {
    // sync.Pool 不保证对象持久化
    // GC 时对象可能被回收，导致频繁创建
}

// ✅ 解决: 对于需要保证数量的场景，使用 channel 实现
var pool = make(chan *BigObject, 100)

func init() {
    for i := 0; i < 100; i++ {
        pool <- &BigObject{data: make([]byte, 1<<20)}
    }
}


// ❌ 坑点 2: 归还对象前未清理状态
var bufPool = sync.Pool{
    New: func() interface{} { return new(bytes.Buffer) },
}

func bad() {
    buf := bufPool.Get().(*bytes.Buffer)
    buf.WriteString("sensitive data")
    bufPool.Put(buf)  // 未清理!

    // 下次获取可能拿到包含敏感数据的 buffer
}

// ✅ 解决: 归还前重置状态
func good() {
    buf := bufPool.Get().(*bytes.Buffer)
    defer func() {
        buf.Reset()  // 清理状态
        bufPool.Put(buf)
    }()

    buf.WriteString("data")
}


// ❌ 坑点 3: 连接池参数配置不当
db.SetMaxOpenConns(1000)   // 太大
db.SetMaxIdleConns(1000)   // 空闲连接过多
db.SetConnMaxLifetime(0)   // 连接永不过期

// ✅ 解决: 合理配置
db.SetMaxOpenConns(20)
db.SetMaxIdleConns(5)
db.SetConnMaxLifetime(30 * time.Minute)
db.SetConnMaxIdleTime(5 * time.Minute)


// ❌ 坑点 4: 连接泄漏
func bad(db *sql.DB) {
    rows, _ := db.Query("SELECT * FROM users")
    // 忘记 rows.Close()
    // 连接不会被归还到池中
}

// ✅ 解决: 使用 defer 确保释放
func good(db *sql.DB) {
    rows, err := db.Query("SELECT * FROM users")
    if err != nil {
        return err
    }
    defer rows.Close()  // 确保关闭

    for rows.Next() {
        // ...
    }
    return rows.Err()
}
```

### 2. 监控指标

```go
// 连接池健康检查
type PoolMetrics struct {
    // 容量指标
    TotalConnections    int64  // 总连接数
    IdleConnections     int64  // 空闲连接数
    ActiveConnections   int64  // 活跃连接数
    WaitingRequests     int64  // 等待中的请求

    // 性能指标
    GetLatencyP99       time.Duration  // 获取连接延迟
    WaitDuration        time.Duration  // 等待时间
    ConnectionsCreated  int64  // 创建的连接数
    ConnectionsReused   int64  // 复用的连接数
    ConnectionsClosed   int64  // 关闭的连接数

    // 错误指标
    TimeoutCount        int64  // 超时次数
    ErrorCount          int64  // 错误次数
}

// Prometheus 指标
var (
    poolConnections = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "pool_connections",
            Help: "Number of connections in pool",
        },
        []string{"pool_name", "state"},  // state: idle, active
    )

    poolWaitDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "pool_wait_duration_seconds",
            Help:    "Time spent waiting for a connection",
            Buckets: []float64{.001, .005, .01, .025, .05, .1, .25, .5, 1},
        },
        []string{"pool_name"},
    )
)
```

---

## 六、检查清单

### 池化设计检查

- [ ] 是否确实需要池化（创建成本 > 池化开销）？
- [ ] 池大小是否根据实际负载测试确定？
- [ ] 是否有池容量监控和告警？
- [ ] 是否处理了资源泄漏？

### 对象池检查

- [ ] 对象归还前是否清理了状态？
- [ ] 是否处理了 GC 回收的情况？
- [ ] 是否有对象验证机制？

### 连接池检查

- [ ] 连接参数是否合理配置？
- [ ] 是否有连接健康检查？
- [ ] 是否有连接超时处理？
- [ ] 是否监控连接池使用情况？
