# 并发编程模型

## 一、并发模型概述

### 并发 vs 并行

```
┌─────────────────────────────────────────────────────────────────┐
│                   并发 (Concurrency) vs 并行 (Parallelism)       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   并发 - 同时处理多件事 (结构)                                    │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                        单核 CPU                          │  │
│   │                                                          │  │
│   │   时间 ─▶                                                │  │
│   │   ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐                   │  │
│   │   │A │  │B │  │A │  │C │  │B │  │A │  ← 时间片轮转      │  │
│   │   └──┘  └──┘  └──┘  └──┘  └──┘  └──┘                   │  │
│   │                                                          │  │
│   │   任务交替执行，看起来"同时"进行                           │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   并行 - 同时做多件事 (执行)                                     │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                        多核 CPU                          │  │
│   │                                                          │  │
│   │   时间 ─▶                                                │  │
│   │   Core1: ┌───────────────────────────────┐              │  │
│   │          │           任务 A              │              │  │
│   │          └───────────────────────────────┘              │  │
│   │   Core2: ┌───────────────────────────────┐              │  │
│   │          │           任务 B              │              │  │
│   │          └───────────────────────────────┘              │  │
│   │   Core3: ┌───────────────────────────────┐              │  │
│   │          │           任务 C              │              │  │
│   │          └───────────────────────────────┘              │  │
│   │                                                          │  │
│   │   任务真正同时执行                                        │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   关系: 并发是并行的前提，并行是并发的一种执行方式                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 主流并发模型

```
┌─────────────────────────────────────────────────────────────────┐
│                      并发模型对比                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. 多线程模型 (共享内存)                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  Thread1   Thread2   Thread3                             │  │
│   │     │         │         │                                │  │
│   │     └─────────┼─────────┘                                │  │
│   │               ▼                                          │  │
│   │        ┌──────────────┐                                  │  │
│   │        │  共享内存     │  ← 需要锁保护                     │  │
│   │        └──────────────┘                                  │  │
│   │  语言: Java, C++, C#                                     │  │
│   │  优点: 直观，性能好                                       │  │
│   │  缺点: 容易死锁、竞态条件                                  │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   2. CSP 模型 (通信顺序进程)                                     │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  Goroutine1 ──────▶ Channel ──────▶ Goroutine2          │  │
│   │                                                          │  │
│   │  "Don't communicate by sharing memory;                   │  │
│   │   share memory by communicating."                        │  │
│   │                                                          │  │
│   │  语言: Go                                                │  │
│   │  优点: 简单、安全                                         │  │
│   │  缺点: Channel 本身有开销                                 │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   3. Actor 模型                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │    ┌─────────┐         ┌─────────┐                      │  │
│   │    │ Actor A │ ──msg──▶│ Actor B │                      │  │
│   │    │ Mailbox │         │ Mailbox │                      │  │
│   │    └─────────┘         └─────────┘                      │  │
│   │                                                          │  │
│   │  每个 Actor 有自己的状态和邮箱                            │  │
│   │  语言: Erlang, Scala(Akka)                               │  │
│   │  优点: 天然分布式、容错                                   │  │
│   │  缺点: 编程范式转变大                                     │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   4. 协程/异步模型                                               │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  async/await, Promise, Future                            │  │
│   │                                                          │  │
│   │  async function fetchData() {                            │  │
│   │      const a = await fetch('/api/a');                    │  │
│   │      const b = await fetch('/api/b');                    │  │
│   │      return process(a, b);                               │  │
│   │  }                                                       │  │
│   │                                                          │  │
│   │  语言: JavaScript, Python, Rust                          │  │
│   │  优点: I/O密集型场景效率高                                │  │
│   │  缺点: 传染性 (async 函数调用也需要 async)                │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、Go 并发编程

### 1. Goroutine 与 Channel

```go
// 基础使用
func main() {
    ch := make(chan int, 10)  // 带缓冲的 channel

    // 生产者
    go func() {
        for i := 0; i < 100; i++ {
            ch <- i
        }
        close(ch)  // 关闭 channel
    }()

    // 消费者
    for v := range ch {
        fmt.Println(v)
    }
}

// 多个 goroutine 同步
func processItems(items []Item) []Result {
    results := make([]Result, len(items))
    var wg sync.WaitGroup

    for i, item := range items {
        wg.Add(1)
        go func(idx int, it Item) {
            defer wg.Done()
            results[idx] = process(it)
        }(i, item)
    }

    wg.Wait()
    return results
}
```

### 2. 并发模式

```go
// 扇出扇入 (Fan-out/Fan-in)
func fanOutFanIn(inputs []int) []int {
    numWorkers := runtime.NumCPU()

    // Fan-out: 分发到多个 worker
    inputCh := make(chan int, len(inputs))
    for _, v := range inputs {
        inputCh <- v
    }
    close(inputCh)

    // 每个 worker 的输出 channel
    workerChs := make([]<-chan int, numWorkers)
    for i := 0; i < numWorkers; i++ {
        workerChs[i] = worker(inputCh)
    }

    // Fan-in: 合并所有 worker 的输出
    return collect(merge(workerChs...))
}

func worker(input <-chan int) <-chan int {
    output := make(chan int)
    go func() {
        defer close(output)
        for v := range input {
            output <- process(v)
        }
    }()
    return output
}

func merge(chs ...<-chan int) <-chan int {
    merged := make(chan int)
    var wg sync.WaitGroup

    for _, ch := range chs {
        wg.Add(1)
        go func(c <-chan int) {
            defer wg.Done()
            for v := range c {
                merged <- v
            }
        }(ch)
    }

    go func() {
        wg.Wait()
        close(merged)
    }()

    return merged
}
```

```go
// 工作池模式 (Worker Pool)
type WorkerPool struct {
    workers   int
    taskQueue chan Task
    results   chan Result
    wg        sync.WaitGroup
}

type Task func() Result
type Result struct {
    Value interface{}
    Err   error
}

func NewWorkerPool(workers, queueSize int) *WorkerPool {
    pool := &WorkerPool{
        workers:   workers,
        taskQueue: make(chan Task, queueSize),
        results:   make(chan Result, queueSize),
    }

    // 启动 workers
    for i := 0; i < workers; i++ {
        pool.wg.Add(1)
        go pool.worker()
    }

    return pool
}

func (p *WorkerPool) worker() {
    defer p.wg.Done()
    for task := range p.taskQueue {
        p.results <- task()
    }
}

func (p *WorkerPool) Submit(task Task) {
    p.taskQueue <- task
}

func (p *WorkerPool) Shutdown() {
    close(p.taskQueue)
    p.wg.Wait()
    close(p.results)
}

// 使用示例
func main() {
    pool := NewWorkerPool(10, 100)

    // 提交任务
    for i := 0; i < 50; i++ {
        id := i
        pool.Submit(func() Result {
            time.Sleep(100 * time.Millisecond)
            return Result{Value: id * 2}
        })
    }

    // 收集结果
    go func() {
        for result := range pool.results {
            fmt.Println(result.Value)
        }
    }()

    pool.Shutdown()
}
```

### 3. 并发控制

```go
// 限制并发数
type Semaphore struct {
    ch chan struct{}
}

func NewSemaphore(n int) *Semaphore {
    return &Semaphore{
        ch: make(chan struct{}, n),
    }
}

func (s *Semaphore) Acquire() {
    s.ch <- struct{}{}
}

func (s *Semaphore) Release() {
    <-s.ch
}

// 使用示例
func processWithLimit(items []Item, limit int) {
    sem := NewSemaphore(limit)
    var wg sync.WaitGroup

    for _, item := range items {
        wg.Add(1)
        sem.Acquire()

        go func(it Item) {
            defer wg.Done()
            defer sem.Release()
            process(it)
        }(item)
    }

    wg.Wait()
}

// 使用 errgroup 处理错误
import "golang.org/x/sync/errgroup"

func processWithErrGroup(ctx context.Context, items []Item) error {
    g, ctx := errgroup.WithContext(ctx)

    // 限制并发数
    g.SetLimit(10)

    for _, item := range items {
        item := item  // 捕获变量
        g.Go(func() error {
            return process(ctx, item)
        })
    }

    return g.Wait()  // 返回第一个错误
}
```

### 4. Context 使用

```go
// 超时控制
func fetchWithTimeout(url string, timeout time.Duration) ([]byte, error) {
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    defer cancel()

    req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    return io.ReadAll(resp.Body)
}

// 取消传播
func processWithCancel(ctx context.Context) error {
    // 子 goroutine 监听取消信号
    errCh := make(chan error, 1)

    go func() {
        errCh <- doWork(ctx)
    }()

    select {
    case err := <-errCh:
        return err
    case <-ctx.Done():
        return ctx.Err()
    }
}

// 传递请求相关数据
type contextKey string

const (
    requestIDKey contextKey = "request_id"
    userIDKey    contextKey = "user_id"
)

func middleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        ctx := r.Context()
        ctx = context.WithValue(ctx, requestIDKey, uuid.New().String())
        ctx = context.WithValue(ctx, userIDKey, r.Header.Get("X-User-ID"))
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}

func getRequestID(ctx context.Context) string {
    if v := ctx.Value(requestIDKey); v != nil {
        return v.(string)
    }
    return ""
}
```

---

## 三、锁与同步原语

### 1. 锁的类型

```go
// 互斥锁 (Mutex)
type SafeCounter struct {
    mu    sync.Mutex
    count int
}

func (c *SafeCounter) Inc() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.count++
}

// 读写锁 (RWMutex) - 读多写少场景
type SafeMap struct {
    mu   sync.RWMutex
    data map[string]interface{}
}

func (m *SafeMap) Get(key string) (interface{}, bool) {
    m.mu.RLock()
    defer m.mu.RUnlock()
    v, ok := m.data[key]
    return v, ok
}

func (m *SafeMap) Set(key string, value interface{}) {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.data[key] = value
}
```

### 2. 原子操作

```go
import "sync/atomic"

// 原子计数器 - 比锁更高效
type AtomicCounter struct {
    count int64
}

func (c *AtomicCounter) Inc() int64 {
    return atomic.AddInt64(&c.count, 1)
}

func (c *AtomicCounter) Get() int64 {
    return atomic.LoadInt64(&c.count)
}

// 原子指针 - 实现无锁数据结构
type AtomicValue struct {
    v atomic.Value
}

func (a *AtomicValue) Store(val interface{}) {
    a.v.Store(val)
}

func (a *AtomicValue) Load() interface{} {
    return a.v.Load()
}

// CAS 操作
func (c *AtomicCounter) CompareAndSwap(old, new int64) bool {
    return atomic.CompareAndSwapInt64(&c.count, old, new)
}

// 使用 CAS 实现自旋锁
type SpinLock struct {
    flag int32
}

func (l *SpinLock) Lock() {
    for !atomic.CompareAndSwapInt32(&l.flag, 0, 1) {
        runtime.Gosched()  // 让出 CPU
    }
}

func (l *SpinLock) Unlock() {
    atomic.StoreInt32(&l.flag, 0)
}
```

### 3. sync.Once 与 sync.Pool

```go
// 单例模式 - 懒加载
var (
    instance *Database
    once     sync.Once
)

func GetDatabase() *Database {
    once.Do(func() {
        instance = connectToDatabase()
    })
    return instance
}

// 对象池 - 减少 GC 压力
var bufferPool = sync.Pool{
    New: func() interface{} {
        return make([]byte, 4096)
    },
}

func processRequest(data []byte) {
    buf := bufferPool.Get().([]byte)
    defer bufferPool.Put(buf)

    // 使用 buf 处理数据
    copy(buf, data)
    // ...
}

// 实际应用: bytes.Buffer 池
var bufPool = sync.Pool{
    New: func() interface{} {
        return new(bytes.Buffer)
    },
}

func formatJSON(v interface{}) string {
    buf := bufPool.Get().(*bytes.Buffer)
    defer func() {
        buf.Reset()
        bufPool.Put(buf)
    }()

    json.NewEncoder(buf).Encode(v)
    return buf.String()
}
```

### 4. 避免锁竞争

```go
// ❌ 坑点 1: 锁粒度太大
type BadCache struct {
    mu   sync.Mutex
    data map[string]interface{}
}

func (c *BadCache) Get(key string) interface{} {
    c.mu.Lock()
    defer c.mu.Unlock()  // 读操作也加互斥锁
    return c.data[key]
}

// ✅ 解决: 使用读写锁
type GoodCache struct {
    mu   sync.RWMutex
    data map[string]interface{}
}

func (c *GoodCache) Get(key string) interface{} {
    c.mu.RLock()
    defer c.mu.RUnlock()
    return c.data[key]
}


// ❌ 坑点 2: 单一锁保护所有数据
type BadMap struct {
    mu   sync.Mutex
    data map[string]interface{}
}

// ✅ 解决: 分片锁 (Sharded Lock)
const shardCount = 32

type ShardedMap struct {
    shards [shardCount]*shard
}

type shard struct {
    mu   sync.RWMutex
    data map[string]interface{}
}

func NewShardedMap() *ShardedMap {
    m := &ShardedMap{}
    for i := 0; i < shardCount; i++ {
        m.shards[i] = &shard{
            data: make(map[string]interface{}),
        }
    }
    return m
}

func (m *ShardedMap) getShard(key string) *shard {
    hash := fnv.New32a()
    hash.Write([]byte(key))
    return m.shards[hash.Sum32()%shardCount]
}

func (m *ShardedMap) Get(key string) (interface{}, bool) {
    shard := m.getShard(key)
    shard.mu.RLock()
    defer shard.mu.RUnlock()
    v, ok := shard.data[key]
    return v, ok
}

func (m *ShardedMap) Set(key string, value interface{}) {
    shard := m.getShard(key)
    shard.mu.Lock()
    defer shard.mu.Unlock()
    shard.data[key] = value
}
```

---

## 四、无锁编程

### 1. Lock-Free 队列

```go
// 无锁队列 (基于 CAS)
type LockFreeQueue struct {
    head unsafe.Pointer
    tail unsafe.Pointer
}

type node struct {
    value interface{}
    next  unsafe.Pointer
}

func NewLockFreeQueue() *LockFreeQueue {
    n := unsafe.Pointer(&node{})
    return &LockFreeQueue{head: n, tail: n}
}

func (q *LockFreeQueue) Enqueue(value interface{}) {
    newNode := &node{value: value}
    newPtr := unsafe.Pointer(newNode)

    for {
        tail := atomic.LoadPointer(&q.tail)
        tailNode := (*node)(tail)
        next := atomic.LoadPointer(&tailNode.next)

        if tail == atomic.LoadPointer(&q.tail) {
            if next == nil {
                // 尝试将新节点链接到队尾
                if atomic.CompareAndSwapPointer(&tailNode.next, nil, newPtr) {
                    // 更新 tail 指针
                    atomic.CompareAndSwapPointer(&q.tail, tail, newPtr)
                    return
                }
            } else {
                // tail 落后了，帮助推进
                atomic.CompareAndSwapPointer(&q.tail, tail, next)
            }
        }
    }
}

func (q *LockFreeQueue) Dequeue() (interface{}, bool) {
    for {
        head := atomic.LoadPointer(&q.head)
        tail := atomic.LoadPointer(&q.tail)
        headNode := (*node)(head)
        next := atomic.LoadPointer(&headNode.next)

        if head == atomic.LoadPointer(&q.head) {
            if head == tail {
                if next == nil {
                    return nil, false  // 队列为空
                }
                // tail 落后，帮助推进
                atomic.CompareAndSwapPointer(&q.tail, tail, next)
            } else {
                value := (*node)(next).value
                if atomic.CompareAndSwapPointer(&q.head, head, next) {
                    return value, true
                }
            }
        }
    }
}
```

### 2. Copy-on-Write

```go
// COW Map - 读多写少场景
type COWMap struct {
    v atomic.Value  // 存储 map[string]interface{}
}

func NewCOWMap() *COWMap {
    m := &COWMap{}
    m.v.Store(make(map[string]interface{}))
    return m
}

func (m *COWMap) Get(key string) (interface{}, bool) {
    data := m.v.Load().(map[string]interface{})
    v, ok := data[key]
    return v, ok
}

func (m *COWMap) Set(key string, value interface{}) {
    // 复制整个 map
    for {
        oldData := m.v.Load().(map[string]interface{})
        newData := make(map[string]interface{}, len(oldData)+1)
        for k, v := range oldData {
            newData[k] = v
        }
        newData[key] = value

        // 原子替换
        if m.v.CompareAndSwap(oldData, newData) {
            return
        }
    }
}

// 适用场景: 配置中心、路由表等读多写少的场景
```

---

## 五、并发陷阱

### 常见问题

```go
// ❌ 陷阱 1: goroutine 泄漏
func leak() {
    ch := make(chan int)
    go func() {
        val := <-ch  // 永远阻塞，因为没人发送
        fmt.Println(val)
    }()
    // 函数返回，ch 和 goroutine 都泄漏了
}

// ✅ 解决: 使用 context 取消
func noLeak(ctx context.Context) {
    ch := make(chan int)
    go func() {
        select {
        case val := <-ch:
            fmt.Println(val)
        case <-ctx.Done():
            return
        }
    }()
}


// ❌ 陷阱 2: 在循环中捕获迭代变量
func badCapture() {
    for i := 0; i < 5; i++ {
        go func() {
            fmt.Println(i)  // 可能全部打印 5
        }()
    }
}

// ✅ 解决: 通过参数传递
func goodCapture() {
    for i := 0; i < 5; i++ {
        go func(n int) {
            fmt.Println(n)  // 正确打印 0-4
        }(i)
    }
}


// ❌ 陷阱 3: 忘记关闭 channel
func badClose() <-chan int {
    ch := make(chan int)
    go func() {
        for i := 0; i < 10; i++ {
            ch <- i
        }
        // 忘记 close(ch)
    }()
    return ch
}

// 消费者会永远阻塞
for v := range badClose() {  // 死循环
    fmt.Println(v)
}


// ❌ 陷阱 4: 数据竞争
type BadCounter struct {
    count int  // 没有保护
}

func (c *BadCounter) Inc() {
    c.count++  // 数据竞争!
}

// ✅ 检测: go run -race main.go
// ✅ 解决: 使用 atomic 或 mutex


// ❌ 陷阱 5: select 中的 default 导致忙等
func badSelect(ch chan int) {
    for {
        select {
        case v := <-ch:
            process(v)
        default:
            // CPU 100%!
        }
    }
}

// ✅ 解决: 使用 time.Sleep 或去掉 default
func goodSelect(ch chan int) {
    for {
        select {
        case v := <-ch:
            process(v)
        case <-time.After(100 * time.Millisecond):
            // 超时处理
        }
    }
}
```

---

## 六、检查清单

### 并发设计检查

- [ ] 是否选择了合适的并发模型？
- [ ] goroutine 生命周期是否可控？
- [ ] 是否有 goroutine 泄漏的风险？
- [ ] 并发数是否有限制？

### 同步机制检查

- [ ] 锁的粒度是否合适？
- [ ] 是否使用了合适的锁类型（mutex vs rwmutex）？
- [ ] 是否可以使用原子操作代替锁？
- [ ] 是否存在死锁风险？

### 数据竞争检查

- [ ] 是否运行过 race detector？
- [ ] 共享数据是否都有保护？
- [ ] channel 是否正确关闭？
- [ ] context 是否正确传递和取消？
