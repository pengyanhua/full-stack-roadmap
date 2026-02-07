# 性能瓶颈分析方法论

## 1. 瓶颈分析概述

### 1.1 性能瓶颈分类

```
系统瓶颈四象限:

           CPU密集型          内存密集型
              │                  │
    高计算    │    大数据         │   大对象
    复杂算法  │    批处理         │   缓存
              │                  │
─────────────┼──────────────────┼─────────────
              │                  │
    慢查询    │    网络传输       │   带宽
    磁盘IO    │    第三方调用     │   延迟
              │                  │
           IO密集型          网络密集型
```

### 1.2 USE方法论

**USE = Utilization + Saturation + Errors**

```
资源分析矩阵:

资源类型     利用率(U)      饱和度(S)       错误(E)
─────────────────────────────────────────────
CPU         90%+          运行队列>CPU数   -
内存        85%+          Swap使用         OOM
磁盘        IOPS 90%+     IO wait高        读写错误
网络        带宽90%+      丢包重传         连接错误
数据库      连接池90%+    慢查询堆积       超时错误
```

## 2. CPU瓶颈分析

### 2.1 CPU指标监控

```bash
# Linux - top命令
top -H -p <pid>  # 查看线程级CPU
# 关注:
# - %CPU: CPU使用率
# - wa: IO等待
# - sy: 系统调用

# pidstat - 进程统计
pidstat -p <pid> 1 10  # 每秒采样,共10次
# 输出:
# %usr: 用户态CPU
# %system: 内核态CPU
# %wait: IO等待

# perf - 性能事件
perf top -p <pid>  # 实时查看热点函数
perf record -p <pid> -g -- sleep 30  # 记录30秒
perf report  # 分析结果
```

### 2.2 火焰图分析

```bash
# 生成CPU火焰图
git clone https://github.com/brendangregg/FlameGraph
cd FlameGraph

# Go应用
curl http://localhost:6060/debug/pprof/profile?seconds=30 > cpu.prof
go tool pprof -raw cpu.prof | ./stackcollapse-go.pl | ./flamegraph.pl > cpu.svg

# Java应用
./profiler.sh -d 60 -f flamegraph.html <pid>

# Python应用
py-spy record -o flamegraph.svg --pid <pid>
```

**火焰图阅读技巧:**

```
火焰图结构:
┌─────────────────────────────────────┐
│        main (100%)                  │ ← 顶层: 入口函数
├─────────────────────────────────────┤
│  handleRequest (95%)   │ other(5%)  │ ← 中层: 主要调用
├────────────────┬────────────────────┤
│ dbQuery (60%)  │ jsonEncode (30%) │ ← 底层: 热点函数
└────────────────┴────────────────────┘

分析要点:
1. 宽度 = CPU时间占比
2. 高度 = 调用栈深度
3. 平顶 = CPU热点 (需优化!)
4. 尖顶 = 调用链长 (可能正常)
```

### 2.3 CPU优化案例

**案例: JSON序列化瓶颈**

```go
// 优化前: 标准库 encoding/json
func handler(w http.ResponseWriter, r *http.Request) {
    data := getData()
    json.NewEncoder(w).Encode(data)  // CPU热点!
}

// Benchmark:
// BenchmarkStandardJSON-8    10000    105234 ns/op

// 优化方案1: 使用更快的JSON库
import "github.com/bytedance/sonic"

func handler(w http.ResponseWriter, r *http.Request) {
    data := getData()
    sonic.NewEncoder(w).Encode(data)
}
// BenchmarkSonicJSON-8       50000     21045 ns/op  (5x faster)

// 优化方案2: 预序列化 + 缓存
var responseCache sync.Map

func handler(w http.ResponseWriter, r *http.Request) {
    cacheKey := getCacheKey(r)
    if cached, ok := responseCache.Load(cacheKey); ok {
        w.Write(cached.([]byte))
        return
    }

    data := getData()
    bytes, _ := sonic.Marshal(data)
    responseCache.Store(cacheKey, bytes)
    w.Write(bytes)
}
// BenchmarkCachedJSON-8     200000      5123 ns/op  (20x faster)
```

## 3. 内存瓶颈分析

### 3.1 内存指标

```bash
# 查看进程内存
ps aux | grep <process>
# VSZ: 虚拟内存
# RSS: 物理内存

# 详细内存映射
pmap -x <pid>

# Go - 内存统计
curl http://localhost:6060/debug/pprof/heap

# Java - 堆内存
jmap -heap <pid>
jstat -gc <pid> 1000

# Python - 内存快照
import tracemalloc
tracemalloc.start()
# ... code ...
snapshot = tracemalloc.take_snapshot()
```

### 3.2 内存泄漏检测

**Go - 内存泄漏案例**

```go
// 泄漏代码
type Cache struct {
    data map[string][]byte
    mu   sync.RWMutex
}

func (c *Cache) Set(key string, value []byte) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.data[key] = value  // 永不删除!
}

// 检测方法
// 1. pprof heap分析
go tool pprof http://localhost:6060/debug/pprof/heap
(pprof) top
(pprof) list Cache.Set

// 2. 对比两次heap快照
curl http://localhost:6060/debug/pprof/heap > heap1.prof
# 运行一段时间
curl http://localhost:6060/debug/pprof/heap > heap2.prof
go tool pprof -base heap1.prof heap2.prof

// 修复方案
import "github.com/patrickmn/go-cache"

func NewCache() *cache.Cache {
    return cache.New(5*time.Minute, 10*time.Minute)
}
```

**Java - 堆dump分析**

```bash
# 生成heap dump
jmap -dump:format=b,file=heap.bin <pid>

# 使用MAT分析
# 1. 下载Eclipse MAT
# 2. File -> Open Heap Dump -> heap.bin
# 3. 查看Leak Suspects
# 4. Dominator Tree找大对象
```

### 3.3 GC调优

**Go GC调优**

```bash
# 查看GC统计
GODEBUG=gctrace=1 ./app
# 输出:
# gc 1 @0.005s 0%: 0.018+1.2+0.004 ms clock, 0.15+0/1.2/3.0+0.03 ms cpu, 4->4->0 MB, 5 MB goal, 8 P

# 调整GC目标百分比
GOGC=200 ./app  # 默认100, 提高到200减少GC频率

# 内存限制(Go 1.19+)
GOMEMLIMIT=8GiB ./app
```

**Java GC调优**

```bash
# G1 GC (推荐)
java -XX:+UseG1GC \
     -XX:MaxGCPauseMillis=200 \
     -XX:G1HeapRegionSize=16m \
     -Xms4g -Xmx4g \
     -XX:+PrintGCDetails \
     -XX:+PrintGCDateStamps \
     -Xloggc:gc.log \
     -jar app.jar

# ZGC (低延迟)
java -XX:+UseZGC \
     -Xms8g -Xmx8g \
     -XX:+UnlockExperimentalVMOptions \
     -jar app.jar
```

## 4. IO瓶颈分析

### 4.1 磁盘IO监控

```bash
# iostat - 磁盘统计
iostat -x 1 10
# 关注:
# %util: 磁盘利用率 (>80%瓶颈)
# await: 平均等待时间
# svctm: 平均服务时间

# iotop - IO top
iotop -o  # 只显示有IO的进程

# 文件IO跟踪
strace -e trace=file -p <pid>
lsof -p <pid>  # 查看打开的文件
```

### 4.2 数据库慢查询

**MySQL慢查询分析**

```sql
-- 开启慢查询日志
SET GLOBAL slow_query_log = 'ON';
SET GLOBAL long_query_time = 1;  -- 1秒以上

-- 分析慢查询
EXPLAIN SELECT * FROM orders WHERE user_id = 123;

-- 优化前
+----+-------------+--------+------+---------------+------+---------+------+--------+-------------+
| id | select_type | table  | type | possible_keys | key  | key_len | ref  | rows   | Extra       |
+----+-------------+--------+------+---------------+------+---------+------+--------+-------------+
|  1 | SIMPLE      | orders | ALL  | NULL          | NULL | NULL    | NULL | 100000 | Using where |
+----+-------------+--------+------+---------------+------+---------+------+--------+-------------+

-- 添加索引
CREATE INDEX idx_user_id ON orders(user_id);

-- 优化后
+----+-------------+--------+------+---------------+-------------+---------+-------+------+-------+
| id | select_type | table  | type | possible_keys | key         | key_len | ref   | rows | Extra |
+----+-------------+--------+------+---------------+-------------+---------+-------+------+-------+
|  1 | SIMPLE      | orders | ref  | idx_user_id   | idx_user_id | 8       | const |  100 | NULL  |
+----+-------------+--------+------+---------------+-------------+---------+-------+------+-------+
```

**查询优化技巧**

```sql
-- 避免SELECT *
SELECT id, name, price FROM products;  -- 好
SELECT * FROM products;  -- 坏

-- 使用LIMIT
SELECT * FROM orders ORDER BY created_at DESC LIMIT 10;

-- 避免子查询
-- 坏
SELECT * FROM orders WHERE user_id IN (SELECT id FROM users WHERE city='北京');
-- 好
SELECT o.* FROM orders o INNER JOIN users u ON o.user_id = u.id WHERE u.city='北京';

-- 使用覆盖索引
CREATE INDEX idx_user_status ON orders(user_id, status);
SELECT status FROM orders WHERE user_id = 123;  -- 索引覆盖
```

### 4.3 缓存优化

**Redis缓存模式**

```python
import redis
import json
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379)

def cache_aside(ttl=3600):
    """Cache-Aside模式装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存key
            cache_key = f"{func.__name__}:{json.dumps(args)}:{json.dumps(kwargs)}"

            # 1. 查缓存
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)

            # 2. 查数据库
            result = func(*args, **kwargs)

            # 3. 写缓存
            redis_client.setex(cache_key, ttl, json.dumps(result))

            return result
        return wrapper
    return decorator

@cache_aside(ttl=600)
def get_user(user_id):
    return db.query(f"SELECT * FROM users WHERE id={user_id}")
```

## 5. 网络瓶颈分析

### 5.1 网络监控

```bash
# 网络流量
iftop -i eth0  # 实时流量
nethogs        # 按进程显示

# 连接状态
netstat -ant | awk '{print $6}' | sort | uniq -c
ss -s  # socket统计

# TCP参数优化
sysctl -w net.core.somaxconn=4096
sysctl -w net.ipv4.tcp_max_syn_backlog=4096
sysctl -w net.ipv4.tcp_tw_reuse=1
```

### 5.2 HTTP优化

```
优化技术:

1. 连接复用
   ┌────────┐    Keep-Alive    ┌────────┐
   │ Client │ ◄──────────────► │ Server │
   └────────┘   一个连接多请求  └────────┘

2. HTTP/2 多路复用
   ┌────────┐    Stream 1,2,3  ┌────────┐
   │ Client │ ◄──────────────► │ Server │
   └────────┘   并行请求        └────────┘

3. 压缩
   Response: 1MB → gzip → 100KB

4. CDN
   Client → CDN(就近) → Origin
```

**代码示例**

```go
// HTTP客户端优化
var httpClient = &http.Client{
    Transport: &http.Transport{
        MaxIdleConns:        100,              // 连接池大小
        MaxIdleConnsPerHost: 10,               // 每个host最大连接
        IdleConnTimeout:     90 * time.Second, // 空闲连接超时
        DisableCompression:  false,            // 启用压缩
    },
    Timeout: 30 * time.Second,
}

// 使用HTTP/2
import "golang.org/x/net/http2"
http2.ConfigureTransport(httpClient.Transport.(*http.Transport))
```

## 6. 分布式系统瓶颈

### 6.1 服务依赖分析

```
依赖链路追踪:

User Request
    ↓
[API Gateway] 50ms
    ↓
[Order Service] 200ms
    ├─► [Product Service] 50ms
    ├─► [Inventory Service] 100ms (慢!)
    ├─► [User Service] 30ms
    └─► [Payment Service] 20ms
         ↓
     Total: 250ms

优化: 并行调用
[Order Service] 100ms
    ├─┬─► [Product Service] 50ms
    │ ├─► [Inventory Service] 100ms
    │ ├─► [User Service] 30ms
    │ └─► [Payment Service] 20ms
    └─ 等待最长的完成 (100ms)

Total: 150ms (提升40%)
```

### 6.2 限流与熔断

```go
// 限流 - Token Bucket
import "golang.org/x/time/rate"

limiter := rate.NewLimiter(100, 200)  // 100 QPS, burst 200

func handler(w http.ResponseWriter, r *http.Request) {
    if !limiter.Allow() {
        http.Error(w, "Too Many Requests", http.StatusTooManyRequests)
        return
    }
    // 处理请求
}

// 熔断 - Circuit Breaker
import "github.com/sony/gobreaker"

cb := gobreaker.NewCircuitBreaker(gobreaker.Settings{
    Name:        "HTTP GET",
    MaxRequests: 3,
    Timeout:     10 * time.Second,
    ReadyToTrip: func(counts gobreaker.Counts) bool {
        failureRatio := float64(counts.TotalFailures) / float64(counts.Requests)
        return counts.Requests >= 3 && failureRatio >= 0.6
    },
})

func callExternalService() (interface{}, error) {
    return cb.Execute(func() (interface{}, error) {
        resp, err := http.Get("http://external-service/api")
        if err != nil {
            return nil, err
        }
        return resp, nil
    })
}
```

## 7. 综合案例

### 7.1 电商系统性能优化

**初始状态:**
- QPS: 500
- P99: 2000ms
- CPU: 80%
- 内存: 4GB

**瓶颈识别:**

```bash
# 1. CPU Profiling
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=30
# 发现: JSON序列化占50% CPU

# 2. 慢查询分析
# 发现: 订单查询无索引, 平均500ms

# 3. 内存分析
# 发现: 商品信息缓存占3GB, 但命中率只有30%
```

**优化方案:**

```go
// 1. JSON序列化优化 (CPU)
// 使用sonic替换标准库
import "github.com/bytedance/sonic"

// 2. 数据库优化 (IO)
CREATE INDEX idx_user_id_created ON orders(user_id, created_at DESC);

// 3. 缓存优化 (内存)
// 热点数据放Redis, 冷数据降级
type ProductCache struct {
    hot  *cache.Cache  // 内存缓存 (top 10%)
    cold redis.Client  // Redis缓存 (90%)
}

func (c *ProductCache) Get(id string) *Product {
    // 先查热数据
    if p, found := c.hot.Get(id); found {
        return p.(*Product)
    }

    // 再查Redis
    if cached, err := c.cold.Get(ctx, "product:"+id).Result(); err == nil {
        var p Product
        json.Unmarshal([]byte(cached), &p)
        return &p
    }

    // 最后查数据库
    p := db.GetProduct(id)
    c.hot.Set(id, p, cache.DefaultExpiration)
    return p
}

// 4. 并行查询 (响应时间)
func getOrderDetails(orderID string) *OrderDetails {
    var wg sync.WaitGroup
    var order *Order
    var items []*OrderItem
    var user *User

    wg.Add(3)

    // 并行查询
    go func() {
        defer wg.Done()
        order = orderRepo.Get(orderID)
    }()

    go func() {
        defer wg.Done()
        items = itemRepo.GetByOrderID(orderID)
    }()

    go func() {
        defer wg.Done()
        user = userRepo.Get(order.UserID)
    }()

    wg.Wait()

    return &OrderDetails{Order: order, Items: items, User: user}
}
```

**优化结果:**
- QPS: 2000 (4x)
- P99: 500ms (4x faster)
- CPU: 45% (降低35%)
- 内存: 2GB (降低50%)

## 8. 最佳实践

### 8.1 性能优化流程

```
1. 监控告警
   ↓
2. 定位瓶颈 (USE方法)
   ↓
3. 分析原因 (Profiling)
   ↓
4. 制定方案
   ↓
5. 验证效果 (A/B测试)
   ↓
6. 上线发布
   ↓
7. 持续监控
```

### 8.2 常见误区

```
❌ 过早优化
   先保证功能正确, 再优化性能

❌ 盲目优化
   没有profiling数据支撑

❌ 局部优化
   忽略整体架构问题

❌ 只优化代码
   忽略配置、架构、基础设施

✓ 数据驱动
✓ 关注瓶颈
✓ 持续迭代
✓ 全栈思维
```

## 9. 总结

性能瓶颈分析核心方法:
1. **USE方法** - 系统级资源分析
2. **RED方法** - 服务级监控(Rate/Error/Duration)
3. **Profiling** - 代码级热点分析
4. **分布式追踪** - 链路级延迟分析

工具推荐:
- 监控: Prometheus + Grafana
- Profiling: pprof, py-spy, async-profiler
- 追踪: Jaeger, Zipkin
- 压测: K6, Gatling
