# 读写分离

## 一、读写分离架构

### 基本架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    读写分离架构                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                       应用服务                                   │
│                          │                                      │
│               ┌──────────┴──────────┐                           │
│               │     数据访问层      │                           │
│               │   (路由策略)        │                           │
│               └──────────┬──────────┘                           │
│                          │                                      │
│            ┌─────────────┴─────────────┐                        │
│            │                           │                        │
│         写操作                      读操作                       │
│            │                           │                        │
│            ▼                           ▼                        │
│   ┌─────────────────┐      ┌─────────────────────────────┐     │
│   │     Master      │      │         Slave Pool          │     │
│   │   (主库-写)     │─────▶│  ┌─────┐ ┌─────┐ ┌─────┐   │     │
│   │                 │ 复制 │  │Slave│ │Slave│ │Slave│   │     │
│   └─────────────────┘      │  │  1  │ │  2  │ │  3  │   │     │
│                            │  └─────┘ └─────┘ └─────┘   │     │
│                            └─────────────────────────────┘     │
│                                                                 │
│   优点:                                                         │
│   • 分担主库压力                                                │
│   • 读性能线性扩展                                              │
│   • 主库故障可切换从库                                          │
│                                                                 │
│   注意:                                                         │
│   • 主从延迟问题                                                │
│   • 数据一致性                                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 复制方式

```
┌─────────────────────────────────────────────────────────────────┐
│                    MySQL 主从复制                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   异步复制 (默认)                                                │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   Master ─────写入────▶ Binlog ─────▶ Slave              │  │
│   │      │                                   │                │  │
│   │      │                                   │                │  │
│   │    返回成功                          应用 Relay Log       │  │
│   │   (不等待)                                               │  │
│   │                                                          │  │
│   │   优点: 性能好                                            │  │
│   │   缺点: 可能丢数据                                        │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   半同步复制                                                     │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   Master ─────写入────▶ Binlog ─────▶ Slave              │  │
│   │      │                                   │                │  │
│   │      │◀─────────── ACK ─────────────────┘                │  │
│   │      │                                                    │  │
│   │    返回成功                                               │  │
│   │   (等待至少1个Slave确认)                                  │  │
│   │                                                          │  │
│   │   优点: 数据安全                                          │  │
│   │   缺点: 性能略有下降                                      │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   组复制 (Group Replication)                                    │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   基于 Paxos 协议，多主架构                               │  │
│   │   强一致性，自动故障转移                                  │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、读写分离实现

### 1. 基于中间件

```yaml
# ShardingSphere 读写分离配置
schemaName: readwrite_splitting_db

dataSources:
  primary_ds:
    url: jdbc:mysql://master:3306/db
    username: root
    password: root
  replica_ds_0:
    url: jdbc:mysql://slave1:3306/db
    username: root
    password: root
  replica_ds_1:
    url: jdbc:mysql://slave2:3306/db
    username: root
    password: root

rules:
  - !READWRITE_SPLITTING
    dataSources:
      readwrite_ds:
        staticStrategy:
          writeDataSourceName: primary_ds
          readDataSourceNames:
            - replica_ds_0
            - replica_ds_1
        loadBalancerName: round_robin
    loadBalancers:
      round_robin:
        type: ROUND_ROBIN
```

### 2. Go 代码实现

```go
// 读写分离数据源
type ReadWriteDataSource struct {
    master  *sql.DB
    slaves  []*sql.DB
    current uint64
}

func NewReadWriteDataSource(masterDSN string, slaveDSNs []string) (*ReadWriteDataSource, error) {
    master, err := sql.Open("mysql", masterDSN)
    if err != nil {
        return nil, err
    }

    slaves := make([]*sql.DB, 0, len(slaveDSNs))
    for _, dsn := range slaveDSNs {
        slave, err := sql.Open("mysql", dsn)
        if err != nil {
            continue // 单个从库失败不影响整体
        }
        slaves = append(slaves, slave)
    }

    return &ReadWriteDataSource{
        master: master,
        slaves: slaves,
    }, nil
}

// 获取写库
func (ds *ReadWriteDataSource) Master() *sql.DB {
    return ds.master
}

// 获取读库 (轮询)
func (ds *ReadWriteDataSource) Slave() *sql.DB {
    if len(ds.slaves) == 0 {
        return ds.master // 无从库时降级到主库
    }

    idx := atomic.AddUint64(&ds.current, 1)
    return ds.slaves[idx%uint64(len(ds.slaves))]
}

// 使用示例
type UserRepository struct {
    ds *ReadWriteDataSource
}

func (r *UserRepository) Create(user *User) error {
    // 写操作使用主库
    _, err := r.ds.Master().Exec(
        "INSERT INTO users (name, email) VALUES (?, ?)",
        user.Name, user.Email,
    )
    return err
}

func (r *UserRepository) GetByID(id string) (*User, error) {
    // 读操作使用从库
    row := r.ds.Slave().QueryRow(
        "SELECT id, name, email FROM users WHERE id = ?", id,
    )
    var user User
    err := row.Scan(&user.ID, &user.Name, &user.Email)
    return &user, err
}
```

### 3. 强制走主库

```go
// 上下文标记强制走主库
type ctxKey string

const forceMasterKey ctxKey = "force_master"

func WithForceMaster(ctx context.Context) context.Context {
    return context.WithValue(ctx, forceMasterKey, true)
}

func IsForceMaster(ctx context.Context) bool {
    v, ok := ctx.Value(forceMasterKey).(bool)
    return ok && v
}

// 智能路由
type SmartDataSource struct {
    ds *ReadWriteDataSource
}

func (s *SmartDataSource) GetDB(ctx context.Context, isWrite bool) *sql.DB {
    // 写操作或强制走主库
    if isWrite || IsForceMaster(ctx) {
        return s.ds.Master()
    }
    return s.ds.Slave()
}

// 使用场景: 写后立即读
func (r *UserRepository) CreateAndGet(ctx context.Context, user *User) (*User, error) {
    // 创建用户
    err := r.Create(user)
    if err != nil {
        return nil, err
    }

    // 强制走主库读取 (避免主从延迟)
    ctx = WithForceMaster(ctx)
    return r.GetByIDWithCtx(ctx, user.ID)
}
```

---

## 三、主从延迟处理

### 延迟检测

```go
// 监控主从延迟
func MonitorReplicationLag(master, slave *sql.DB) (int64, error) {
    // 方法1: 通过 SHOW SLAVE STATUS
    var secondsBehindMaster sql.NullInt64
    row := slave.QueryRow("SHOW SLAVE STATUS")
    // 解析 Seconds_Behind_Master 字段

    // 方法2: 心跳表
    // 主库定期更新心跳表，从库读取计算差值
    var masterTime, slaveTime time.Time

    master.QueryRow("SELECT heartbeat_time FROM heartbeat WHERE id = 1").Scan(&masterTime)
    slave.QueryRow("SELECT heartbeat_time FROM heartbeat WHERE id = 1").Scan(&slaveTime)

    lag := masterTime.Sub(slaveTime).Milliseconds()
    return lag, nil
}

// 心跳表
/*
CREATE TABLE heartbeat (
    id INT PRIMARY KEY,
    heartbeat_time TIMESTAMP(3) NOT NULL
);

-- 主库定期执行
UPDATE heartbeat SET heartbeat_time = NOW(3) WHERE id = 1;
*/
```

### 延迟解决方案

```
┌─────────────────────────────────────────────────────────────────┐
│                   主从延迟解决方案                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   方案 1: 强制走主库                                             │
│   ─────────────────────────────────────────────────────────    │
│   场景: 写操作后立即读取                                         │
│   实现: 通过 hint 或上下文标记                                   │
│                                                                 │
│   方案 2: 延迟时间内走主库                                       │
│   ─────────────────────────────────────────────────────────    │
│   实现: 写操作后记录时间，一定时间内读走主库                      │
│                                                                 │
│   方案 3: 等待复制完成                                          │
│   ─────────────────────────────────────────────────────────    │
│   实现: 使用 GTID 等待从库复制到指定位置                         │
│                                                                 │
│   方案 4: 缓存一致性                                             │
│   ─────────────────────────────────────────────────────────    │
│   实现: 写后更新缓存，读先查缓存                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```go
// 方案 2: 延迟时间内走主库
type WriteTracker struct {
    mu         sync.RWMutex
    lastWrite  map[string]time.Time
    lagWindow  time.Duration
}

func NewWriteTracker(lagWindow time.Duration) *WriteTracker {
    return &WriteTracker{
        lastWrite: make(map[string]time.Time),
        lagWindow: lagWindow,
    }
}

// 记录写操作
func (t *WriteTracker) RecordWrite(key string) {
    t.mu.Lock()
    t.lastWrite[key] = time.Now()
    t.mu.Unlock()
}

// 检查是否需要走主库
func (t *WriteTracker) ShouldUseMaster(key string) bool {
    t.mu.RLock()
    lastWrite, ok := t.lastWrite[key]
    t.mu.RUnlock()

    if !ok {
        return false
    }

    return time.Since(lastWrite) < t.lagWindow
}

// 使用示例
var tracker = NewWriteTracker(500 * time.Millisecond)

func UpdateUser(user *User) error {
    err := masterDB.Exec("UPDATE users SET name = ? WHERE id = ?", user.Name, user.ID)
    if err == nil {
        tracker.RecordWrite(fmt.Sprintf("user:%s", user.ID))
    }
    return err
}

func GetUser(id string) (*User, error) {
    var db *sql.DB
    if tracker.ShouldUseMaster(fmt.Sprintf("user:%s", id)) {
        db = masterDB
    } else {
        db = slaveDB
    }
    // 查询...
}
```

---

## 四、高可用方案

### MHA (MySQL High Availability)

```
┌─────────────────────────────────────────────────────────────────┐
│                      MHA 架构                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    ┌─────────────┐                              │
│                    │ MHA Manager │                              │
│                    └──────┬──────┘                              │
│                           │ 监控                                │
│            ┌──────────────┼──────────────┐                      │
│            ▼              ▼              ▼                      │
│     ┌──────────┐   ┌──────────┐   ┌──────────┐                 │
│     │  Master  │   │  Slave1  │   │  Slave2  │                 │
│     │   Node   │   │   Node   │   │   Node   │                 │
│     └──────────┘   └──────────┘   └──────────┘                 │
│                                                                 │
│   故障转移流程:                                                  │
│   1. Manager 检测到 Master 故障                                 │
│   2. 保存 binlog (从宕机 Master)                                │
│   3. 选择最新的 Slave 提升为 Master                             │
│   4. 其他 Slave 指向新 Master                                   │
│   5. 更新 VIP 或 DNS                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Orchestrator

```go
// Orchestrator API 调用示例
type Orchestrator struct {
    baseURL string
    client  *http.Client
}

// 获取集群拓扑
func (o *Orchestrator) GetTopology(clusterName string) (*Topology, error) {
    resp, err := o.client.Get(fmt.Sprintf("%s/api/cluster/%s", o.baseURL, clusterName))
    // ...
}

// 手动故障转移
func (o *Orchestrator) Failover(clusterName string) error {
    resp, err := o.client.Post(
        fmt.Sprintf("%s/api/graceful-master-takeover/%s", o.baseURL, clusterName),
        "application/json",
        nil,
    )
    // ...
}
```

---

## 五、检查清单

### 架构检查

- [ ] 读写比例是否适合读写分离？
- [ ] 从库数量是否足够？
- [ ] 复制方式是否满足数据安全？
- [ ] 高可用方案是否完善？

### 延迟处理检查

- [ ] 是否监控主从延迟？
- [ ] 关键业务是否处理了延迟问题？
- [ ] 是否有强制走主库的机制？
- [ ] 从库故障是否能自动降级？
