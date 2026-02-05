# 消息队列模式

## 一、消息队列概述

### 为什么需要消息队列？

```
┌─────────────────────────────────────────────────────────────────┐
│                   消息队列解决的问题                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. 异步处理 - 提升响应速度                                     │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   同步: 用户下单 → 扣库存 → 发短信 → 发邮件 → 返回        │  │
│   │         总耗时: 100ms + 50ms + 100ms + 100ms = 350ms     │  │
│   │                                                          │  │
│   │   异步: 用户下单 → 扣库存 → 发消息 → 返回                 │  │
│   │         总耗时: 100ms + 50ms + 5ms = 155ms               │  │
│   │         (短信邮件异步处理)                                │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   2. 削峰填谷 - 保护下游                                        │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   峰值: 10000 QPS ──▶ MQ ──▶ 下游按 1000 QPS 消费       │  │
│   │                                                          │  │
│   │   请求量    ████████                                     │  │
│   │            ████████                                      │  │
│   │   处理量   ────────────────────                          │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   3. 系统解耦 - 降低依赖                                        │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   紧耦合: 订单服务 → 库存服务 → 物流服务 → 通知服务      │  │
│   │          (一个服务故障，全链路不可用)                     │  │
│   │                                                          │  │
│   │   松耦合: 订单服务 → MQ ← 库存/物流/通知服务              │  │
│   │          (服务独立，互不影响)                             │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 消息队列对比

| 特性 | Kafka | RocketMQ | RabbitMQ | Pulsar |
|------|-------|----------|----------|--------|
| 吞吐量 | 百万级 | 十万级 | 万级 | 百万级 |
| 延迟 | ms级 | ms级 | us级 | ms级 |
| 可靠性 | 高 | 高 | 高 | 高 |
| 顺序消息 | 分区内 | 支持 | 支持 | 支持 |
| 事务消息 | 支持 | 支持 | 支持 | 支持 |
| 延迟消息 | 不支持 | 支持 | 支持 | 支持 |
| 适用场景 | 大数据/日志 | 电商/交易 | 企业应用 | 云原生 |

---

## 二、消息模式

### 1. 点对点 (P2P)

```
┌─────────────────────────────────────────────────────────────────┐
│                   点对点模式                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Producer ────▶ Queue ────▶ Consumer                          │
│                                                                 │
│   特点:                                                         │
│   • 一条消息只被一个消费者消费                                   │
│   • 多个消费者竞争消费 (负载均衡)                                │
│   • 消费后消息被删除                                            │
│                                                                 │
│   示例: 订单处理、任务分发                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 发布订阅 (Pub/Sub)

```
┌─────────────────────────────────────────────────────────────────┐
│                   发布订阅模式                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                            ┌────▶ Consumer Group A             │
│   Producer ────▶ Topic ────┼────▶ Consumer Group B             │
│                            └────▶ Consumer Group C             │
│                                                                 │
│   特点:                                                         │
│   • 一条消息被多个消费组消费                                     │
│   • 同一消费组内竞争消费                                        │
│   • 消息保留一定时间                                            │
│                                                                 │
│   示例: 事件广播、数据同步                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、可靠性保证

### 1. 消息丢失场景

```
┌─────────────────────────────────────────────────────────────────┐
│                   消息丢失的三个阶段                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   阶段 1: 生产者 → Broker                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   丢失原因: 网络故障、Broker 未持久化                     │  │
│   │   解决方案:                                               │  │
│   │   • 同步发送 + 重试                                      │  │
│   │   • 确认机制 (acks=all)                                  │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   阶段 2: Broker 存储                                           │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   丢失原因: Broker 宕机、磁盘故障                         │  │
│   │   解决方案:                                               │  │
│   │   • 同步刷盘                                             │  │
│   │   • 多副本复制                                           │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   阶段 3: Broker → 消费者                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   丢失原因: 消费者处理失败后自动确认                      │  │
│   │   解决方案:                                               │  │
│   │   • 手动确认 (处理成功后再 ACK)                          │  │
│   │   • 失败重试                                             │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Go 可靠消息实现

```go
// Kafka 可靠生产者
func NewReliableProducer(brokers []string) (*kafka.Writer, error) {
    return &kafka.Writer{
        Addr:         kafka.TCP(brokers...),
        Balancer:     &kafka.LeastBytes{},
        RequiredAcks: kafka.RequireAll,  // 所有副本确认
        Async:        false,              // 同步发送
        MaxAttempts:  3,                  // 重试次数
    }, nil
}

// 带重试的发送
func SendWithRetry(ctx context.Context, w *kafka.Writer, msg kafka.Message) error {
    var err error
    for i := 0; i < 3; i++ {
        err = w.WriteMessages(ctx, msg)
        if err == nil {
            return nil
        }
        time.Sleep(time.Duration(i+1) * 100 * time.Millisecond)
    }
    return err
}

// 可靠消费者
func ReliableConsumer(ctx context.Context, r *kafka.Reader, handler func(msg kafka.Message) error) {
    for {
        msg, err := r.FetchMessage(ctx)
        if err != nil {
            continue
        }

        // 处理消息
        err = handler(msg)
        if err != nil {
            // 处理失败，不提交 offset
            // 消息会被重新消费
            log.Printf("process failed: %v", err)
            continue
        }

        // 手动提交 offset
        if err := r.CommitMessages(ctx, msg); err != nil {
            log.Printf("commit failed: %v", err)
        }
    }
}
```

### 3. 幂等消费

```go
// 消息幂等处理
type IdempotentConsumer struct {
    cache   *redis.Client
    handler func(msg Message) error
}

func (c *IdempotentConsumer) Process(msg Message) error {
    // 检查是否已处理
    key := fmt.Sprintf("msg:processed:%s", msg.ID)
    exists, _ := c.cache.Exists(context.Background(), key).Result()
    if exists > 0 {
        // 已处理，跳过
        return nil
    }

    // 处理消息
    err := c.handler(msg)
    if err != nil {
        return err
    }

    // 标记已处理 (设置过期时间防止无限增长)
    c.cache.Set(context.Background(), key, "1", 24*time.Hour)
    return nil
}

// 数据库层面幂等
func ProcessOrder(tx *sql.Tx, msg OrderMessage) error {
    // 使用唯一索引保证幂等
    _, err := tx.Exec(`
        INSERT INTO order_process_log (message_id, processed_at)
        VALUES (?, NOW())
        ON DUPLICATE KEY UPDATE processed_at = processed_at
    `, msg.ID)

    if err != nil {
        return err
    }

    // 检查是否是重复消息
    var count int
    tx.QueryRow("SELECT ROW_COUNT()").Scan(&count)
    if count == 0 {
        // 已处理过
        return nil
    }

    // 执行业务逻辑
    return doBusinessLogic(tx, msg)
}
```

---

## 四、顺序消息

### 顺序保证

```
┌─────────────────────────────────────────────────────────────────┐
│                   顺序消息实现                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   全局顺序:                                                      │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   所有消息进入同一分区                                    │  │
│   │   缺点: 吞吐量低                                          │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   分区顺序 (推荐):                                               │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   相同 key 的消息进入同一分区                             │  │
│   │                                                          │  │
│   │   Order-1 ────┐                                          │  │
│   │   Order-1 ────┼────▶ Partition 0 ────▶ Consumer 0       │  │
│   │   Order-1 ────┘                                          │  │
│   │                                                          │  │
│   │   Order-2 ────┐                                          │  │
│   │   Order-2 ────┼────▶ Partition 1 ────▶ Consumer 1       │  │
│   │   Order-2 ────┘                                          │  │
│   │                                                          │  │
│   │   同一订单的消息保证顺序                                  │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```go
// 按 key 路由到同一分区
func SendOrderMessage(w *kafka.Writer, orderID string, msg []byte) error {
    return w.WriteMessages(context.Background(), kafka.Message{
        Key:   []byte(orderID),  // 相同 orderID 进入同一分区
        Value: msg,
    })
}

// 顺序消费
func ConsumeInOrder(r *kafka.Reader) {
    for {
        msg, err := r.FetchMessage(context.Background())
        if err != nil {
            continue
        }

        // 单线程处理保证顺序
        process(msg)

        r.CommitMessages(context.Background(), msg)
    }
}
```

---

## 五、延迟消息

### 延迟消息实现

```go
// 方案 1: RocketMQ 原生支持
// 延迟级别: 1s 5s 10s 30s 1m 2m 3m 4m 5m 6m 7m 8m 9m 10m 20m 30m 1h 2h

// 方案 2: Redis 实现延迟队列
type DelayQueue struct {
    client *redis.Client
    key    string
}

func (q *DelayQueue) Push(msg Message, delay time.Duration) error {
    data, _ := json.Marshal(msg)
    score := float64(time.Now().Add(delay).Unix())

    return q.client.ZAdd(context.Background(), q.key, &redis.Z{
        Score:  score,
        Member: data,
    }).Err()
}

func (q *DelayQueue) Poll() (*Message, error) {
    now := float64(time.Now().Unix())

    // 获取到期的消息
    results, err := q.client.ZRangeByScore(context.Background(), q.key, &redis.ZRangeBy{
        Min:   "-inf",
        Max:   fmt.Sprintf("%f", now),
        Count: 1,
    }).Result()

    if err != nil || len(results) == 0 {
        return nil, nil
    }

    // 删除并返回
    if q.client.ZRem(context.Background(), q.key, results[0]).Val() > 0 {
        var msg Message
        json.Unmarshal([]byte(results[0]), &msg)
        return &msg, nil
    }

    return nil, nil
}

// 方案 3: 时间轮
type TimeWheel struct {
    interval time.Duration
    slots    int
    current  int
    buckets  [][]Task
    ticker   *time.Ticker
}

func (tw *TimeWheel) AddTask(delay time.Duration, task Task) {
    ticks := int(delay / tw.interval)
    slot := (tw.current + ticks) % tw.slots
    tw.buckets[slot] = append(tw.buckets[slot], task)
}

func (tw *TimeWheel) Start() {
    tw.ticker = time.NewTicker(tw.interval)
    go func() {
        for range tw.ticker.C {
            tw.current = (tw.current + 1) % tw.slots
            for _, task := range tw.buckets[tw.current] {
                go task.Execute()
            }
            tw.buckets[tw.current] = nil
        }
    }()
}
```

---

## 六、事务消息

### 本地消息表

```go
// 本地消息表方案
func CreateOrderWithMessage(tx *sql.Tx, order *Order) error {
    // 1. 创建订单
    _, err := tx.Exec("INSERT INTO orders (...) VALUES (...)", ...)
    if err != nil {
        return err
    }

    // 2. 插入本地消息表
    msg := OrderCreatedMessage{OrderID: order.ID}
    data, _ := json.Marshal(msg)
    _, err = tx.Exec(`
        INSERT INTO outbox_messages (id, topic, payload, status, created_at)
        VALUES (?, ?, ?, 'PENDING', NOW())
    `, uuid.New().String(), "order-created", data)

    return err
}

// 后台任务扫描发送
func SendPendingMessages() {
    for {
        messages := db.Query("SELECT * FROM outbox_messages WHERE status = 'PENDING' LIMIT 100")

        for _, msg := range messages {
            err := mq.Send(msg.Topic, msg.Payload)
            if err == nil {
                db.Exec("UPDATE outbox_messages SET status = 'SENT' WHERE id = ?", msg.ID)
            }
        }

        time.Sleep(time.Second)
    }
}
```

---

## 七、检查清单

### 生产者检查

- [ ] 是否使用同步发送确认？
- [ ] 是否配置了重试机制？
- [ ] 是否有消息发送失败的处理？

### 消费者检查

- [ ] 是否使用手动确认？
- [ ] 是否实现了幂等消费？
- [ ] 是否有死信队列处理？

### 可靠性检查

- [ ] Broker 是否多副本？
- [ ] 是否配置了刷盘策略？
- [ ] 是否有消息积压监控？
