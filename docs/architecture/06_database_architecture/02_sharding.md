# 分库分表

## 一、何时分库分表？

### 分库分表指标

```
┌─────────────────────────────────────────────────────────────────┐
│                    分库分表时机判断                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   单表数据量参考:                                                │
│   ─────────────────────────────────────────────────────────    │
│   • < 500万行: 通常不需要分表                                   │
│   • 500万-2000万: 可以考虑分表                                  │
│   • > 2000万行: 建议分表                                        │
│   • > 5000万行: 必须分表                                        │
│                                                                 │
│   注意: 以上只是参考，实际取决于:                                │
│   • 单行数据大小                                                │
│   • 查询复杂度                                                  │
│   • 索引数量和大小                                              │
│   • 服务器配置                                                  │
│                                                                 │
│   QPS 参考:                                                     │
│   ─────────────────────────────────────────────────────────    │
│   • 单机 MySQL 读: 3000-5000 QPS                                │
│   • 单机 MySQL 写: 1000-2000 QPS                                │
│   • 超过时考虑分库                                              │
│                                                                 │
│   ⚠️ 分库分表不是银弹:                                          │
│   ─────────────────────────────────────────────────────────    │
│   优先考虑:                                                      │
│   1. 硬件升级 (SSD、内存)                                       │
│   2. 读写分离                                                   │
│   3. 缓存                                                       │
│   4. SQL 优化                                                   │
│   以上都无法解决再考虑分库分表                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、分片策略

### 1. 水平分片 vs 垂直分片

```
┌─────────────────────────────────────────────────────────────────┐
│                   分片策略对比                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   垂直分库 - 按业务拆分                                          │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   拆分前:                                                 │  │
│   │   ┌─────────────────────────────────────────┐           │  │
│   │   │              电商数据库                  │           │  │
│   │   │  users | orders | products | payments   │           │  │
│   │   └─────────────────────────────────────────┘           │  │
│   │                                                          │  │
│   │   拆分后:                                                 │  │
│   │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │  │
│   │   │ user_db  │ │ order_db │ │product_db│ │payment_db│  │  │
│   │   │  users   │ │  orders  │ │ products │ │ payments │  │  │
│   │   └──────────┘ └──────────┘ └──────────┘ └──────────┘  │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   垂直分表 - 拆分宽表                                            │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   拆分前: users (id, name, avatar, profile, ...)        │  │
│   │                                                          │  │
│   │   拆分后:                                                 │  │
│   │   users (id, name)              -- 常用字段              │  │
│   │   user_profiles (user_id, avatar, profile, ...)         │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   水平分表 - 拆分行数据                                          │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   拆分前: orders (1亿条数据)                              │  │
│   │                                                          │  │
│   │   拆分后:                                                 │  │
│   │   orders_0 (2500万)  -- user_id % 4 == 0                │  │
│   │   orders_1 (2500万)  -- user_id % 4 == 1                │  │
│   │   orders_2 (2500万)  -- user_id % 4 == 2                │  │
│   │   orders_3 (2500万)  -- user_id % 4 == 3                │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 分片键选择

```
┌─────────────────────────────────────────────────────────────────┐
│                     分片键选择原则                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   原则 1: 数据分布均匀                                           │
│   ─────────────────────────────────────────────────────────    │
│   ❌ 按地区分片: 北京 80%，其他 20% → 数据倾斜                   │
│   ✅ 按用户ID分片: 分布均匀                                      │
│                                                                 │
│   原则 2: 查询收敛                                               │
│   ─────────────────────────────────────────────────────────    │
│   ❌ 不带分片键查询需要扫描所有分片                              │
│   ✅ 带分片键查询只访问一个分片                                  │
│                                                                 │
│   原则 3: 避免跨分片操作                                         │
│   ─────────────────────────────────────────────────────────    │
│   ❌ 跨分片 JOIN、跨分片事务                                     │
│   ✅ 相关数据在同一分片                                         │
│                                                                 │
│   常见分片键:                                                    │
│   ─────────────────────────────────────────────────────────    │
│   • 用户ID: 用户相关数据 (订单、收藏、消息)                      │
│   • 租户ID: SaaS 多租户场景                                     │
│   • 时间: 日志、流水数据                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3. 分片算法

```go
// 1. 取模分片
func HashShard(userID int64, shardCount int) int {
    return int(userID % int64(shardCount))
}
// 优点: 简单、均匀
// 缺点: 扩容需要迁移数据


// 2. 范围分片
func RangeShard(userID int64) int {
    if userID < 1000000 {
        return 0
    } else if userID < 2000000 {
        return 1
    } else {
        return 2
    }
}
// 优点: 扩容简单
// 缺点: 数据可能不均匀、热点问题


// 3. 一致性哈希
type ConsistentHash struct {
    ring       []uint32
    nodes      map[uint32]string
    replicas   int
}

func (c *ConsistentHash) GetNode(key string) string {
    hash := c.hash(key)
    idx := sort.Search(len(c.ring), func(i int) bool {
        return c.ring[i] >= hash
    })
    if idx >= len(c.ring) {
        idx = 0
    }
    return c.nodes[c.ring[idx]]
}
// 优点: 扩容只迁移部分数据
// 缺点: 实现复杂


// 4. 时间分片 (按月分表)
func TimeShard(createTime time.Time) string {
    return fmt.Sprintf("orders_%s", createTime.Format("200601"))
}
// 优点: 历史数据可归档
// 缺点: 跨时间范围查询复杂
```

---

## 三、分库分表实现

### 1. 使用 ShardingSphere

```yaml
# shardingsphere-proxy config-sharding.yaml
schemaName: sharding_db

dataSources:
  ds_0:
    url: jdbc:mysql://mysql-0:3306/db_0
    username: root
    password: root
  ds_1:
    url: jdbc:mysql://mysql-1:3306/db_1
    username: root
    password: root

rules:
  - !SHARDING
    tables:
      orders:
        actualDataNodes: ds_${0..1}.orders_${0..3}
        tableStrategy:
          standard:
            shardingColumn: user_id
            shardingAlgorithmName: orders_table_mod
        databaseStrategy:
          standard:
            shardingColumn: user_id
            shardingAlgorithmName: orders_db_mod
        keyGenerateStrategy:
          column: order_id
          keyGeneratorName: snowflake

    shardingAlgorithms:
      orders_db_mod:
        type: MOD
        props:
          sharding-count: 2
      orders_table_mod:
        type: MOD
        props:
          sharding-count: 4

    keyGenerators:
      snowflake:
        type: SNOWFLAKE
```

### 2. Go 代码实现

```go
// 分片路由器
type ShardRouter struct {
    dbCount    int
    tableCount int
    dataSources map[int]*sql.DB
}

func NewShardRouter(dsns []string, tableCount int) *ShardRouter {
    router := &ShardRouter{
        dbCount:     len(dsns),
        tableCount:  tableCount,
        dataSources: make(map[int]*sql.DB),
    }

    for i, dsn := range dsns {
        db, _ := sql.Open("mysql", dsn)
        router.dataSources[i] = db
    }

    return router
}

// 计算分片
func (r *ShardRouter) Route(userID int64) (dbIndex int, tableIndex int) {
    // 先计算库
    dbIndex = int(userID % int64(r.dbCount))
    // 再计算表
    tableIndex = int(userID % int64(r.tableCount))
    return
}

// 获取表名
func (r *ShardRouter) GetTableName(userID int64) string {
    _, tableIndex := r.Route(userID)
    return fmt.Sprintf("orders_%d", tableIndex)
}

// 获取数据源
func (r *ShardRouter) GetDB(userID int64) *sql.DB {
    dbIndex, _ := r.Route(userID)
    return r.dataSources[dbIndex]
}

// 执行查询
func (r *ShardRouter) QueryOrder(userID int64, orderID string) (*Order, error) {
    db := r.GetDB(userID)
    table := r.GetTableName(userID)

    query := fmt.Sprintf("SELECT * FROM %s WHERE order_id = ?", table)
    row := db.QueryRow(query, orderID)

    var order Order
    err := row.Scan(&order.ID, &order.UserID, &order.Amount)
    return &order, err
}

// 跨分片查询
func (r *ShardRouter) QueryAllOrders(orderIDs []string) ([]*Order, error) {
    // 需要查询所有分片
    var wg sync.WaitGroup
    results := make(chan *Order, len(orderIDs))

    for dbIdx, db := range r.dataSources {
        for tableIdx := 0; tableIdx < r.tableCount; tableIdx++ {
            wg.Add(1)
            go func(db *sql.DB, table string) {
                defer wg.Done()

                query := fmt.Sprintf("SELECT * FROM %s WHERE order_id IN (?)", table)
                // ... 执行查询
            }(db, fmt.Sprintf("orders_%d", tableIdx))
        }
    }

    wg.Wait()
    close(results)

    var orders []*Order
    for order := range results {
        orders = append(orders, order)
    }
    return orders, nil
}
```

---

## 四、分布式 ID

### 1. ID 生成方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| UUID | 简单、无依赖 | 无序、太长 | 分布式场景 |
| 自增ID | 有序、简单 | 需要协调 | 单库 |
| Snowflake | 有序、高性能 | 时钟回拨 | 通用 |
| 号段模式 | 高可用 | 实现复杂 | 高并发 |

### 2. Snowflake 实现

```go
// Snowflake ID 结构:
// 1位符号位 + 41位时间戳 + 10位机器ID + 12位序列号
// 总共 64 位

type Snowflake struct {
    mu        sync.Mutex
    epoch     int64  // 起始时间戳
    machineID int64  // 机器ID
    sequence  int64  // 序列号
    lastTime  int64  // 上次生成时间
}

const (
    machineBits   = 10
    sequenceBits  = 12
    machineMax    = -1 ^ (-1 << machineBits)
    sequenceMax   = -1 ^ (-1 << sequenceBits)
    timeShift     = machineBits + sequenceBits
    machineShift  = sequenceBits
)

func NewSnowflake(machineID int64) *Snowflake {
    return &Snowflake{
        epoch:     1704067200000, // 2024-01-01 00:00:00 UTC
        machineID: machineID & machineMax,
    }
}

func (s *Snowflake) Generate() int64 {
    s.mu.Lock()
    defer s.mu.Unlock()

    now := time.Now().UnixMilli()

    if now < s.lastTime {
        // 时钟回拨处理
        panic("clock moved backwards")
    }

    if now == s.lastTime {
        s.sequence = (s.sequence + 1) & sequenceMax
        if s.sequence == 0 {
            // 序列号用尽，等待下一毫秒
            for now <= s.lastTime {
                now = time.Now().UnixMilli()
            }
        }
    } else {
        s.sequence = 0
    }

    s.lastTime = now

    id := ((now - s.epoch) << timeShift) |
        (s.machineID << machineShift) |
        s.sequence

    return id
}

// 解析 ID
func (s *Snowflake) Parse(id int64) (timestamp, machineID, sequence int64) {
    timestamp = (id >> timeShift) + s.epoch
    machineID = (id >> machineShift) & machineMax
    sequence = id & sequenceMax
    return
}
```

---

## 五、分库分表问题与解决

### 常见问题

```
┌─────────────────────────────────────────────────────────────────┐
│                   分库分表常见问题                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   问题 1: 跨分片查询                                             │
│   ─────────────────────────────────────────────────────────    │
│   场景: 查询所有订单 (无分片键)                                  │
│   方案:                                                         │
│   • 并行查询所有分片 + 内存聚合                                  │
│   • 使用 ES 等搜索引擎做二级索引                                 │
│   • 冗余数据到汇总表                                            │
│                                                                 │
│   问题 2: 跨分片 JOIN                                           │
│   ─────────────────────────────────────────────────────────    │
│   场景: orders JOIN users                                       │
│   方案:                                                         │
│   • 字段冗余 (订单表冗余用户名)                                  │
│   • 应用层 JOIN (多次查询组装)                                   │
│   • 全局表广播 (小表复制到每个分片)                              │
│                                                                 │
│   问题 3: 跨分片事务                                             │
│   ─────────────────────────────────────────────────────────    │
│   场景: 扣减不同分片的库存                                       │
│   方案:                                                         │
│   • 最终一致性 (消息队列 + 补偿)                                 │
│   • TCC / Saga 分布式事务                                       │
│   • 业务设计避免跨分片事务                                       │
│                                                                 │
│   问题 4: 扩容                                                   │
│   ─────────────────────────────────────────────────────────    │
│   场景: 2 个分片扩到 4 个                                        │
│   方案:                                                         │
│   • 一致性哈希 (只迁移部分数据)                                  │
│   • 翻倍扩容 (2→4→8)                                            │
│   • 预分片 (提前分更多表)                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 六、检查清单

### 分片设计检查

- [ ] 是否真的需要分库分表？
- [ ] 分片键选择是否合理？
- [ ] 数据分布是否均匀？
- [ ] 查询是否都带分片键？

### 实现检查

- [ ] 分布式 ID 方案是否确定？
- [ ] 跨分片查询如何处理？
- [ ] 跨分片事务如何处理？
- [ ] 扩容方案是否考虑？
