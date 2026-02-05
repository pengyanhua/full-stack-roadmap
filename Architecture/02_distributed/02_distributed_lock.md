# 分布式锁

## 一、为什么需要分布式锁

```
单机环境：
┌─────────────────────────────────────┐
│           JVM / 进程                │
│  Thread A ──┐                       │
│             │──▶ synchronized/Lock  │
│  Thread B ──┘                       │
└─────────────────────────────────────┘

分布式环境：
┌──────────────┐     ┌──────────────┐
│   Server A   │     │   Server B   │
│  Request 1   │     │  Request 2   │
└──────────────┘     └──────────────┘
        │                   │
        └─────────┬─────────┘
                  ▼
           ┌──────────────┐
           │   Database   │  ← 同时操作，数据不一致！
           └──────────────┘

需要分布式锁来协调跨进程/跨机器的并发访问
```

## 二、分布式锁的核心要求

```
┌─────────────────────────────────────────────────────────────┐
│                    分布式锁核心特性                          │
├─────────────────────────────────────────────────────────────┤
│ 1. 互斥性      │ 同一时刻只有一个客户端持有锁                │
│ 2. 防死锁      │ 客户端崩溃后锁能自动释放（超时机制）        │
│ 3. 安全性      │ 只有锁持有者才能释放锁                      │
│ 4. 可用性      │ 锁服务高可用，不是单点                      │
│ 5. 可重入      │ 同一客户端可多次获取同一把锁（可选）        │
│ 6. 公平性      │ 按请求顺序获取锁（可选）                    │
└─────────────────────────────────────────────────────────────┘
```

## 三、Redis 分布式锁

### 基础实现

```go
// 加锁
func (l *RedisLock) Lock(ctx context.Context) (bool, error) {
    // SET key value NX EX seconds
    // NX: 只在 key 不存在时设置
    // EX: 设置过期时间
    result, err := l.client.SetNX(ctx, l.key, l.value, l.expiration).Result()
    return result, err
}

// 解锁（Lua 脚本保证原子性）
func (l *RedisLock) Unlock(ctx context.Context) error {
    script := `
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("DEL", KEYS[1])
        else
            return 0
        end
    `
    _, err := l.client.Eval(ctx, script, []string{l.key}, l.value).Result()
    return err
}
```

### 完整实现（含续期）

```go
package distlock

import (
    "context"
    "github.com/go-redis/redis/v8"
    "github.com/google/uuid"
    "time"
)

type RedisLock struct {
    client     *redis.Client
    key        string
    value      string        // 唯一标识，防止误删
    expiration time.Duration
    cancelFunc context.CancelFunc
}

func NewRedisLock(client *redis.Client, key string, expiration time.Duration) *RedisLock {
    return &RedisLock{
        client:     client,
        key:        "lock:" + key,
        value:      uuid.New().String(),
        expiration: expiration,
    }
}

// Lock 获取锁
func (l *RedisLock) Lock(ctx context.Context) (bool, error) {
    // 尝试获取锁
    success, err := l.client.SetNX(ctx, l.key, l.value, l.expiration).Result()
    if err != nil || !success {
        return false, err
    }

    // 启动看门狗续期
    l.startWatchdog(ctx)
    return true, nil
}

// LockWithRetry 带重试的获取锁
func (l *RedisLock) LockWithRetry(ctx context.Context, retryTimes int, retryDelay time.Duration) (bool, error) {
    for i := 0; i < retryTimes; i++ {
        success, err := l.Lock(ctx)
        if err != nil {
            return false, err
        }
        if success {
            return true, nil
        }

        select {
        case <-ctx.Done():
            return false, ctx.Err()
        case <-time.After(retryDelay):
            continue
        }
    }
    return false, nil
}

// Unlock 释放锁
func (l *RedisLock) Unlock(ctx context.Context) error {
    // 停止看门狗
    if l.cancelFunc != nil {
        l.cancelFunc()
    }

    // Lua 脚本保证原子性
    script := `
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("DEL", KEYS[1])
        else
            return 0
        end
    `
    _, err := l.client.Eval(ctx, script, []string{l.key}, l.value).Result()
    return err
}

// startWatchdog 看门狗自动续期
func (l *RedisLock) startWatchdog(parentCtx context.Context) {
    ctx, cancel := context.WithCancel(parentCtx)
    l.cancelFunc = cancel

    go func() {
        ticker := time.NewTicker(l.expiration / 3) // 每 1/3 过期时间续期一次
        defer ticker.Stop()

        for {
            select {
            case <-ctx.Done():
                return
            case <-ticker.C:
                l.extend(ctx)
            }
        }
    }()
}

// extend 续期
func (l *RedisLock) extend(ctx context.Context) {
    script := `
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("PEXPIRE", KEYS[1], ARGV[2])
        else
            return 0
        end
    `
    l.client.Eval(ctx, script, []string{l.key}, l.value, l.expiration.Milliseconds())
}
```

### Redlock 算法（多节点）

```go
/*
Redlock 算法步骤：
1. 获取当前时间戳 T1
2. 依次向 N 个 Redis 节点请求锁
3. 如果获取锁的节点数 >= N/2+1，且总耗时 < 锁过期时间，则获取成功
4. 如果获取失败，向所有节点发送释放锁请求
*/

type Redlock struct {
    clients    []*redis.Client
    quorum     int
    key        string
    value      string
    expiration time.Duration
}

func NewRedlock(clients []*redis.Client, key string, expiration time.Duration) *Redlock {
    return &Redlock{
        clients:    clients,
        quorum:     len(clients)/2 + 1,
        key:        "lock:" + key,
        value:      uuid.New().String(),
        expiration: expiration,
    }
}

func (r *Redlock) Lock(ctx context.Context) (bool, error) {
    start := time.Now()
    successCount := 0

    // 向所有节点请求锁
    for _, client := range r.clients {
        success, err := client.SetNX(ctx, r.key, r.value, r.expiration).Result()
        if err == nil && success {
            successCount++
        }
    }

    // 计算获取锁耗时
    elapsed := time.Since(start)

    // 判断是否获取成功
    if successCount >= r.quorum && elapsed < r.expiration {
        return true, nil
    }

    // 获取失败，释放所有节点
    r.Unlock(ctx)
    return false, nil
}

func (r *Redlock) Unlock(ctx context.Context) {
    script := `
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("DEL", KEYS[1])
        end
        return 0
    `
    for _, client := range r.clients {
        client.Eval(ctx, script, []string{r.key}, r.value)
    }
}
```

## 四、ZooKeeper 分布式锁

```
ZooKeeper 锁原理：利用临时顺序节点

/locks/order_lock
├── lock_0000000001  (Client A)  ← 最小序号，获得锁
├── lock_0000000002  (Client B)  ← 监听 0001
└── lock_0000000003  (Client C)  ← 监听 0002

流程：
1. Client 在 /locks/order_lock 下创建临时顺序节点
2. 获取所有子节点，判断自己是否最小
3. 如果是最小，获得锁
4. 如果不是，监听比自己小的节点
5. 被监听节点删除时，重新判断
```

```java
// Java 实现（使用 Curator）
public class ZkDistributedLock {
    private final InterProcessMutex lock;

    public ZkDistributedLock(CuratorFramework client, String lockPath) {
        this.lock = new InterProcessMutex(client, lockPath);
    }

    public boolean acquire(long timeout, TimeUnit unit) throws Exception {
        return lock.acquire(timeout, unit);
    }

    public void release() throws Exception {
        lock.release();
    }
}

// 使用
CuratorFramework client = CuratorFrameworkFactory.newClient(
    "localhost:2181",
    new RetryNTimes(3, 1000)
);
client.start();

ZkDistributedLock lock = new ZkDistributedLock(client, "/locks/order");
try {
    if (lock.acquire(10, TimeUnit.SECONDS)) {
        // 执行业务逻辑
    }
} finally {
    lock.release();
}
```

## 五、MySQL 分布式锁

```sql
-- 方式1：利用唯一索引
CREATE TABLE distributed_lock (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    lock_name VARCHAR(64) NOT NULL,
    lock_value VARCHAR(255) NOT NULL,
    expire_time DATETIME NOT NULL,
    UNIQUE KEY uk_lock_name (lock_name)
);

-- 加锁
INSERT INTO distributed_lock (lock_name, lock_value, expire_time)
VALUES ('order_lock', 'uuid-xxx', DATE_ADD(NOW(), INTERVAL 30 SECOND));

-- 解锁
DELETE FROM distributed_lock
WHERE lock_name = 'order_lock' AND lock_value = 'uuid-xxx';

-- 定时清理过期锁
DELETE FROM distributed_lock WHERE expire_time < NOW();
```

```sql
-- 方式2：利用 FOR UPDATE
-- 需要先创建锁记录

-- 加锁（阻塞等待）
BEGIN;
SELECT * FROM distributed_lock WHERE lock_name = 'order_lock' FOR UPDATE;
-- 执行业务逻辑
COMMIT;

-- 非阻塞尝试
SELECT * FROM distributed_lock
WHERE lock_name = 'order_lock'
FOR UPDATE NOWAIT;  -- 获取不到立即返回错误
```

## 六、方案对比

```
┌────────────────────────────────────────────────────────────────────────┐
│                         分布式锁方案对比                                │
├──────────────┬──────────────┬──────────────┬──────────────┬───────────┤
│              │    Redis     │  ZooKeeper   │    MySQL     │   etcd    │
├──────────────┼──────────────┼──────────────┼──────────────┼───────────┤
│ 性能         │ 高           │ 中           │ 低           │ 中        │
│ 可靠性       │ 中           │ 高           │ 高           │ 高        │
│ 复杂度       │ 低           │ 中           │ 低           │ 中        │
│ 一致性       │ AP           │ CP           │ CP           │ CP        │
├──────────────┼──────────────┼──────────────┼──────────────┼───────────┤
│ 公平性       │ 无           │ 有           │ 无           │ 有        │
│ 可重入       │ 需自实现     │ 内置         │ 需自实现     │ 需自实现  │
│ 自动续期     │ 需自实现     │ 临时节点     │ 无           │ Lease     │
├──────────────┼──────────────┼──────────────┼──────────────┼───────────┤
│ 适用场景     │ 高并发、     │ 强一致、     │ 已有MySQL、  │ K8s环境、 │
│              │ 对一致性     │ 公平锁       │ 并发不高     │ 强一致    │
│              │ 要求不高     │              │              │           │
└──────────────┴──────────────┴──────────────┴──────────────┴───────────┘
```

## 七、实战避坑

### 坑1：锁过期但业务未完成

```go
// ❌ 问题场景
func ProcessOrder(orderID string) {
    lock := NewRedisLock(client, orderID, 10*time.Second)
    lock.Lock(ctx)
    defer lock.Unlock(ctx)

    // 业务执行了 15 秒，锁已过期
    // 其他请求可能已经获取了锁
    doSlowBusiness()  // 超过 10 秒
}

// ✅ 解决方案：看门狗自动续期
func ProcessOrder(orderID string) {
    lock := NewRedisLock(client, orderID, 30*time.Second)
    lock.Lock(ctx)  // 内部启动看门狗
    defer lock.Unlock(ctx)

    doSlowBusiness()  // 看门狗会自动续期
}
```

### 坑2：锁被其他客户端误删

```go
// ❌ 错误：直接删除 key
func Unlock() {
    client.Del(ctx, lockKey)  // 可能删除别人的锁！
}

// ✅ 正确：验证后再删除
func Unlock() {
    script := `
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("DEL", KEYS[1])
        end
        return 0
    `
    client.Eval(ctx, script, []string{lockKey}, lockValue)
}
```

### 坑3：Redis 主从切换导致锁丢失

```
时序：
1. Client A 在 Master 获取锁
2. Master 尚未同步到 Slave 就宕机
3. Slave 被提升为新 Master
4. Client B 在新 Master 获取了同一把锁
5. Client A 和 B 同时持有锁！

解决方案：
1. 使用 Redlock 算法（多节点）
2. 使用 ZooKeeper（强一致）
3. 业务层做幂等处理
```

### 坑4：可重入锁实现错误

```go
// ✅ 正确的可重入锁实现
type ReentrantLock struct {
    client    *redis.Client
    key       string
    value     string  // 客户端标识
    lockCount int     // 重入次数（本地记录）
    mu        sync.Mutex
}

func (l *ReentrantLock) Lock(ctx context.Context) error {
    l.mu.Lock()
    defer l.mu.Unlock()

    // 已经持有锁，增加计数
    if l.lockCount > 0 {
        l.lockCount++
        return nil
    }

    // 尝试获取锁
    success, err := l.client.SetNX(ctx, l.key, l.value, 30*time.Second).Result()
    if err != nil {
        return err
    }
    if !success {
        return errors.New("failed to acquire lock")
    }

    l.lockCount = 1
    return nil
}

func (l *ReentrantLock) Unlock(ctx context.Context) error {
    l.mu.Lock()
    defer l.mu.Unlock()

    if l.lockCount == 0 {
        return errors.New("lock not held")
    }

    l.lockCount--
    if l.lockCount > 0 {
        return nil  // 还有重入，不释放
    }

    // 释放锁
    return l.release(ctx)
}
```

### 坑5：死锁检测

```go
// 锁获取超时机制
func (l *Lock) LockWithTimeout(ctx context.Context, timeout time.Duration) error {
    deadline := time.Now().Add(timeout)

    for time.Now().Before(deadline) {
        success, err := l.tryLock(ctx)
        if err != nil {
            return err
        }
        if success {
            return nil
        }

        // 指数退避
        time.Sleep(time.Duration(rand.Intn(100)) * time.Millisecond)
    }

    return errors.New("lock timeout")
}
```

## 八、最佳实践

```
1. 锁粒度
   - 尽量细粒度：lock:order:{orderId} 而非 lock:order
   - 避免大范围锁导致性能问题

2. 超时时间
   - 根据业务最长执行时间设置
   - 使用看门狗自动续期
   - 留足余量但不要过长

3. 重试策略
   - 使用指数退避
   - 设置最大重试次数
   - 避免惊群效应

4. 异常处理
   - finally 中释放锁
   - 考虑锁续期失败的情况
   - 业务幂等兜底

5. 监控告警
   - 锁等待时间
   - 锁持有时间
   - 锁获取失败率
```
