# 容灾与多活架构

## 一、容灾概述

### 灾难类型

```
┌─────────────────────────────────────────────────────────────────┐
│                       灾难类型分级                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Level 1: 局部故障                                             │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  • 单机硬件故障    • 进程崩溃    • 网络抖动             │  │
│   │  影响范围: 单实例   恢复时间: 秒~分钟级                   │  │
│   │  应对: 冗余部署、健康检查、自动重启                       │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   Level 2: 机房故障                                             │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  • 机房断电    • 网络中断    • 空调故障                  │  │
│   │  影响范围: 整个机房   恢复时间: 分钟~小时级               │  │
│   │  应对: 同城双机房、同城双活                               │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   Level 3: 城市级灾难                                           │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  • 地震    • 洪水    • 大规模停电                        │  │
│   │  影响范围: 整个城市   恢复时间: 小时~天级                  │  │
│   │  应对: 异地灾备、两地三中心                               │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   Level 4: 区域性灾难                                           │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  • 大规模自然灾害    • 区域网络故障                      │  │
│   │  影响范围: 区域/国家   恢复时间: 天~周级                  │  │
│   │  应对: 跨区域多活、全球化部署                             │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 容灾指标

```
┌─────────────────────────────────────────────────────────────────┐
│                    核心容灾指标                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   RPO (Recovery Point Objective) - 恢复点目标                    │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │    最后备份          故障发生         数据恢复           │  │
│   │        │               │               │                │  │
│   │    ────┼───────────────┼───────────────┼────▶ 时间      │  │
│   │        │◀─── RPO ────▶│                                 │  │
│   │        │   数据损失量   │                                │  │
│   │                                                          │  │
│   │   RPO = 0: 零数据丢失 (同步复制)                         │  │
│   │   RPO < 1分钟: 异步复制 + 频繁备份                       │  │
│   │   RPO < 1小时: 定期备份                                   │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   RTO (Recovery Time Objective) - 恢复时间目标                   │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │       故障发生          服务恢复                         │  │
│   │           │               │                             │  │
│   │    ───────┼───────────────┼─────────────────▶ 时间      │  │
│   │           │◀─── RTO ────▶│                              │  │
│   │           │   服务中断时间 │                             │  │
│   │                                                          │  │
│   │   RTO = 0: 零中断 (多活架构)                             │  │
│   │   RTO < 1分钟: 热备自动切换                              │  │
│   │   RTO < 1小时: 温备手动切换                              │  │
│   │   RTO < 24小时: 冷备恢复                                  │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   不同业务的 RPO/RTO 要求:                                       │
│   ┌────────────────────┬──────────┬──────────┐                │
│   │      业务类型       │   RPO    │   RTO    │                │
│   ├────────────────────┼──────────┼──────────┤                │
│   │ 金融交易            │    0     │  秒级    │                │
│   │ 电商订单            │  分钟级  │  分钟级  │                │
│   │ 内容发布            │  小时级  │  小时级  │                │
│   │ 数据分析            │  天级    │  天级    │                │
│   └────────────────────┴──────────┴──────────┘                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、容灾架构模式

### 1. 冷备 (Cold Standby)

```
┌─────────────────────────────────────────────────────────────────┐
│                       冷备架构                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   主数据中心 (活跃)                     备用数据中心 (关闭)      │
│   ┌───────────────────┐               ┌───────────────────┐    │
│   │                   │   定期备份     │                   │    │
│   │  ┌─────────────┐  │  ─────────▶   │  ┌─────────────┐  │    │
│   │  │   应用服务   │  │               │  │  应用服务   │  │    │
│   │  │   (运行中)   │  │               │  │  (已关闭)   │  │    │
│   │  └─────────────┘  │               │  └─────────────┘  │    │
│   │                   │               │                   │    │
│   │  ┌─────────────┐  │  数据同步     │  ┌─────────────┐  │    │
│   │  │   数据库     │  │  (每日/每周)  │  │   数据库     │  │    │
│   │  │   (主库)     │  │ ─────────▶   │  │   (冷备份)   │  │    │
│   │  └─────────────┘  │               │  └─────────────┘  │    │
│   │                   │               │                   │    │
│   └───────────────────┘               └───────────────────┘    │
│                                                                 │
│   特点:                                                         │
│   • RPO: 小时~天级 (取决于备份频率)                              │
│   • RTO: 小时~天级 (需要启动服务、恢复数据)                       │
│   • 成本: 低 (备用资源可以关闭)                                  │
│   • 适用: 非核心业务、成本敏感场景                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 温备 (Warm Standby)

```
┌─────────────────────────────────────────────────────────────────┐
│                       温备架构                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   主数据中心 (活跃)                     备用数据中心 (待命)      │
│   ┌───────────────────┐               ┌───────────────────┐    │
│   │                   │               │                   │    │
│   │  ┌─────────────┐  │               │  ┌─────────────┐  │    │
│   │  │   应用服务   │  │               │  │  应用服务   │  │    │
│   │  │  (处理流量)  │  │               │  │ (已启动,空闲)│  │    │
│   │  └─────────────┘  │               │  └─────────────┘  │    │
│   │                   │               │                   │    │
│   │  ┌─────────────┐  │   实时复制    │  ┌─────────────┐  │    │
│   │  │   数据库     │  │  ─────────▶  │  │   数据库     │  │    │
│   │  │   (主库)     │  │   (异步)     │  │   (从库)     │  │    │
│   │  └─────────────┘  │               │  └─────────────┘  │    │
│   │                   │               │                   │    │
│   └───────────────────┘               └───────────────────┘    │
│                                                                 │
│   故障切换流程:                                                  │
│   1. 检测主数据中心故障                                          │
│   2. 将从库提升为主库                                            │
│   3. DNS/负载均衡切换到备用中心                                  │
│   4. 服务开始处理流量                                            │
│                                                                 │
│   特点:                                                         │
│   • RPO: 秒~分钟级 (异步复制延迟)                                │
│   • RTO: 分钟级 (主要是切换时间)                                 │
│   • 成本: 中等 (备用资源需要运行)                                │
│   • 适用: 重要业务、可接受短暂中断                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3. 热备 / 双活 (Hot Standby / Active-Active)

```
┌─────────────────────────────────────────────────────────────────┐
│                     同城双活架构                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                      ┌─────────────┐                            │
│                      │    GSLB     │                            │
│                      │  (全局负载)  │                            │
│                      └─────────────┘                            │
│                            │                                    │
│              ┌─────────────┴─────────────┐                      │
│              ▼                           ▼                      │
│   ┌───────────────────┐       ┌───────────────────┐            │
│   │   数据中心 A       │       │   数据中心 B       │            │
│   │   (同城机房1)      │       │   (同城机房2)      │            │
│   ├───────────────────┤       ├───────────────────┤            │
│   │  ┌─────────────┐  │       │  ┌─────────────┐  │            │
│   │  │  应用服务    │  │◀─────▶│  │  应用服务    │  │            │
│   │  │  50% 流量   │  │       │  │  50% 流量   │  │            │
│   │  └─────────────┘  │       │  └─────────────┘  │            │
│   │                   │       │                   │            │
│   │  ┌─────────────┐  │  同步  │  ┌─────────────┐  │            │
│   │  │  MySQL      │  │◀─────▶│  │  MySQL      │  │            │
│   │  │  Master     │  │  复制  │  │  Slave      │  │            │
│   │  └─────────────┘  │       │  └─────────────┘  │            │
│   │                   │       │                   │            │
│   │  ┌─────────────┐  │  双向  │  ┌─────────────┐  │            │
│   │  │  Redis      │  │◀─────▶│  │  Redis      │  │            │
│   │  │  Cluster    │  │  同步  │  │  Cluster    │  │            │
│   │  └─────────────┘  │       │  └─────────────┘  │            │
│   └───────────────────┘       └───────────────────┘            │
│                                                                 │
│   特点:                                                         │
│   • 两个机房都处理流量                                           │
│   • 数据实时同步 (延迟 < 10ms)                                   │
│   • 单机房故障时另一机房接管全部流量                              │
│   • RPO ≈ 0, RTO ≈ 0                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、两地三中心

### 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                    两地三中心架构                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                         用户请求                                 │
│                            │                                    │
│                      ┌─────┴─────┐                              │
│                      │   GSLB    │                              │
│                      └─────┬─────┘                              │
│                            │                                    │
│         ┌──────────────────┼──────────────────┐                 │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│   ┌───────────┐      ┌───────────┐      ┌───────────┐          │
│   │  北京     │      │  北京     │      │  上海     │          │
│   │  机房 A   │◀────▶│  机房 B   │      │  灾备中心 │          │
│   │  (生产)   │ 同城 │  (生产)   │      │  (灾备)   │          │
│   └───────────┘ 双活 └───────────┘      └───────────┘          │
│         │                  │                  │                 │
│         │              ┌───┴───┐              │                 │
│         │              │       │              │                 │
│         ▼              ▼       ▼              ▼                 │
│   ┌─────────────────────────────────────────────────┐          │
│   │                   数据同步                       │          │
│   │                                                  │          │
│   │  北京A ◀──同步复制──▶ 北京B ──异步复制──▶ 上海   │          │
│   │  Master              Slave               Slave   │          │
│   │                                                  │          │
│   │  延迟: < 1ms         延迟: < 1ms        延迟: ~50ms         │
│   │                                                  │          │
│   └─────────────────────────────────────────────────┘          │
│                                                                 │
│   容灾级别:                                                      │
│   ├─ 单机房故障: 同城切换 (RTO: 秒级)                            │
│   ├─ 同城故障: 异地切换 (RTO: 分钟级, RPO: ~50ms数据)            │
│   └─ 区域故障: 需人工介入评估                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### MySQL 两地三中心配置

```sql
-- 北京机房A (Master)
-- my.cnf
[mysqld]
server-id = 1
log-bin = mysql-bin
binlog-format = ROW
sync_binlog = 1                        -- 同步写binlog
innodb_flush_log_at_trx_commit = 1     -- 每次事务刷盘

-- 半同步复制配置
plugin-load = "rpl_semi_sync_master=semisync_master.so"
rpl_semi_sync_master_enabled = 1
rpl_semi_sync_master_timeout = 1000     -- 超时降级为异步

-- 北京机房B (Slave, 半同步)
[mysqld]
server-id = 2
relay-log = relay-bin
read_only = 1
plugin-load = "rpl_semi_sync_slave=semisync_slave.so"
rpl_semi_sync_slave_enabled = 1

-- 上海灾备 (Slave, 异步)
[mysqld]
server-id = 3
relay-log = relay-bin
read_only = 1
-- 异步复制，不配置半同步
```

```go
// 数据库路由中间件
type DBRouter struct {
    master     *sql.DB      // 北京A
    localSlave *sql.DB      // 北京B
    remoteSlave *sql.DB     // 上海 (仅灾备使用)

    currentRegion string    // 当前区域
}

func (r *DBRouter) GetReadDB() *sql.DB {
    // 优先读本地从库
    if r.localSlave != nil && r.isHealthy(r.localSlave) {
        return r.localSlave
    }
    // 降级读主库
    return r.master
}

func (r *DBRouter) GetWriteDB() *sql.DB {
    return r.master
}

// 灾备切换
func (r *DBRouter) Failover(newMaster string) error {
    // 1. 停止旧主库写入
    // 2. 等待从库追上
    // 3. 提升新主库
    // 4. 切换路由
    return nil
}
```

---

## 四、异地多活

### 1. 异地多活挑战

```
┌─────────────────────────────────────────────────────────────────┐
│                   异地多活核心挑战                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   挑战 1: 网络延迟                                               │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  北京 ←──────────── 50~100ms ──────────────▶ 上海        │  │
│   │                                                          │  │
│   │  问题: 同步复制导致写入延迟                               │  │
│   │  方案: 单元化部署，本地写入                               │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   挑战 2: 数据一致性                                             │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  场景: 用户在北京下单，上海查询订单                        │  │
│   │                                                          │  │
│   │  问题: 异步复制导致数据延迟                               │  │
│   │  方案: 1. 按用户分片，路由到固定单元                       │  │
│   │        2. 必须跨单元访问时，查主库                        │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   挑战 3: 数据冲突                                               │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  场景: 同一用户同时在两地修改数据                          │  │
│   │                                                          │  │
│   │  问题: 双写导致数据冲突                                   │  │
│   │  方案: 1. 分片保证单写                                    │  │
│   │        2. 冲突检测 + 自动合并                             │  │
│   │        3. 最终一致性 (Last Write Wins)                    │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 单元化架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     单元化多活架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                         用户请求                                 │
│                            │                                    │
│                      ┌─────┴─────┐                              │
│                      │   路由层   │                              │
│                      │ (按用户ID) │                              │
│                      └─────┬─────┘                              │
│                            │                                    │
│           ┌────────────────┼────────────────┐                   │
│           │                │                │                   │
│           ▼                ▼                ▼                   │
│   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐        │
│   │    单元 A     │ │    单元 B     │ │    单元 C     │        │
│   │   (北京)      │ │   (上海)      │ │   (深圳)      │        │
│   │  用户 0-33%   │ │  用户 34-66%  │ │  用户 67-100% │        │
│   ├───────────────┤ ├───────────────┤ ├───────────────┤        │
│   │ ┌───────────┐ │ │ ┌───────────┐ │ │ ┌───────────┐ │        │
│   │ │  应用服务  │ │ │ │  应用服务  │ │ │ │  应用服务  │ │        │
│   │ └───────────┘ │ │ └───────────┘ │ │ └───────────┘ │        │
│   │ ┌───────────┐ │ │ ┌───────────┐ │ │ ┌───────────┐ │        │
│   │ │  数据库    │ │ │ │  数据库    │ │ │ │  数据库    │ │        │
│   │ │ (本单元数据)│ │ │ │ (本单元数据)│ │ │ │ (本单元数据)│ │        │
│   │ └───────────┘ │ │ └───────────┘ │ │ └───────────┘ │        │
│   │ ┌───────────┐ │ │ ┌───────────┐ │ │ ┌───────────┐ │        │
│   │ │  缓存      │ │ │ │  缓存      │ │ │ │  缓存      │ │        │
│   │ └───────────┘ │ │ └───────────┘ │ │ └───────────┘ │        │
│   └───────────────┘ └───────────────┘ └───────────────┘        │
│           │                │                │                   │
│           └────────────────┼────────────────┘                   │
│                            │                                    │
│                      异步数据同步                                │
│                 (用于灾备和全局查询)                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

核心原则:
1. 用户路由: 根据用户ID路由到固定单元
2. 本地闭环: 单元内完成所有读写操作
3. 数据隔离: 每个单元只存储本单元用户数据
4. 异步同步: 单元间异步同步数据用于灾备
```

### 3. 路由策略实现

```go
// 单元路由器
type UnitRouter struct {
    units       []Unit
    hashRing    *consistent.HashRing
    userMapping map[string]string  // 用户ID -> 单元ID 映射缓存
}

type Unit struct {
    ID       string
    Region   string
    Endpoint string
    Weight   int
}

func NewUnitRouter(units []Unit) *UnitRouter {
    router := &UnitRouter{
        units:       units,
        userMapping: make(map[string]string),
    }

    // 构建一致性哈希环
    members := make([]consistent.Member, len(units))
    for i, u := range units {
        members[i] = unitMember{u}
    }
    router.hashRing = consistent.New(members, consistent.Config{
        PartitionCount:    271,
        ReplicationFactor: 20,
    })

    return router
}

// 路由用户到指定单元
func (r *UnitRouter) RouteUser(userID string) *Unit {
    // 检查缓存
    if unitID, ok := r.userMapping[userID]; ok {
        return r.getUnit(unitID)
    }

    // 一致性哈希计算
    member := r.hashRing.LocateKey([]byte(userID))
    unit := member.(unitMember).Unit

    // 缓存映射
    r.userMapping[userID] = unit.ID

    return &unit
}

// HTTP 中间件
func UnitRoutingMiddleware(router *UnitRouter) gin.HandlerFunc {
    return func(c *gin.Context) {
        userID := c.GetHeader("X-User-ID")
        if userID == "" {
            userID = c.Query("user_id")
        }

        if userID != "" {
            unit := router.RouteUser(userID)

            // 检查是否需要跨单元转发
            if !isLocalUnit(unit) {
                // 转发到目标单元
                proxyToUnit(c, unit)
                c.Abort()
                return
            }
        }

        c.Next()
    }
}

// 跨单元请求转发
func proxyToUnit(c *gin.Context, unit *Unit) {
    proxy := httputil.NewSingleHostReverseProxy(&url.URL{
        Scheme: "http",
        Host:   unit.Endpoint,
    })

    // 添加追踪头
    c.Request.Header.Set("X-Forwarded-From", localUnit.ID)
    c.Request.Header.Set("X-Original-Request-ID", c.GetHeader("X-Request-ID"))

    proxy.ServeHTTP(c.Writer, c.Request)
}
```

### 4. 数据同步策略

```go
// 异步数据同步服务
type DataSyncService struct {
    localDB    *sql.DB
    syncQueue  MessageQueue     // Kafka/RocketMQ
    conflictResolver ConflictResolver
}

// 数据变更事件
type DataChangeEvent struct {
    ID        string    `json:"id"`
    Table     string    `json:"table"`
    Type      string    `json:"type"`  // INSERT/UPDATE/DELETE
    Data      []byte    `json:"data"`
    UnitID    string    `json:"unit_id"`
    Timestamp time.Time `json:"timestamp"`
    Version   int64     `json:"version"`
}

// 发送数据变更到同步队列
func (s *DataSyncService) PublishChange(event DataChangeEvent) error {
    data, _ := json.Marshal(event)
    return s.syncQueue.Publish("data-sync", data)
}

// 消费并应用数据变更
func (s *DataSyncService) ConsumeChanges() {
    s.syncQueue.Subscribe("data-sync", func(msg []byte) error {
        var event DataChangeEvent
        json.Unmarshal(msg, &event)

        // 跳过本单元的事件
        if event.UnitID == localUnitID {
            return nil
        }

        return s.applyChange(event)
    })
}

func (s *DataSyncService) applyChange(event DataChangeEvent) error {
    // 检查冲突
    localData, err := s.getLocalData(event.Table, event.ID)
    if err != nil && err != sql.ErrNoRows {
        return err
    }

    if localData != nil && localData.Version >= event.Version {
        // 本地版本更新，使用冲突解决策略
        resolved := s.conflictResolver.Resolve(localData, event.Data)
        return s.updateLocal(event.Table, event.ID, resolved)
    }

    // 直接应用变更
    switch event.Type {
    case "INSERT":
        return s.insertLocal(event.Table, event.Data)
    case "UPDATE":
        return s.updateLocal(event.Table, event.ID, event.Data)
    case "DELETE":
        return s.deleteLocal(event.Table, event.ID)
    }
    return nil
}

// 冲突解决器
type ConflictResolver interface {
    Resolve(local, remote []byte) []byte
}

// Last Write Wins 策略
type LWWResolver struct{}

func (r *LWWResolver) Resolve(local, remote []byte) []byte {
    localTime := extractTimestamp(local)
    remoteTime := extractTimestamp(remote)

    if remoteTime.After(localTime) {
        return remote
    }
    return local
}

// 合并策略 (适用于计数器等场景)
type MergeResolver struct{}

func (r *MergeResolver) Resolve(local, remote []byte) []byte {
    // 合并两边的增量
    localCounter := parseCounter(local)
    remoteCounter := parseCounter(remote)

    merged := Counter{
        Value: localCounter.Value + remoteCounter.Delta,
    }
    return toBytes(merged)
}
```

---

## 五、全局数据服务

### 全局数据处理

```go
// 某些数据需要全局唯一或全局可见
// 例如: 用户名、手机号、订单号

type GlobalDataService struct {
    globalDB    *sql.DB           // 全局数据库 (强一致性)
    localUnits  map[string]*Unit  // 本地单元
    coordinator DistributedCoordinator
}

// 全局唯一性检查
func (s *GlobalDataService) CheckGlobalUnique(table, field, value string) (bool, error) {
    // 使用分布式锁
    lockKey := fmt.Sprintf("unique:%s:%s:%s", table, field, value)
    lock, err := s.coordinator.Lock(lockKey, 10*time.Second)
    if err != nil {
        return false, err
    }
    defer lock.Unlock()

    // 查询全局数据库
    var count int
    err = s.globalDB.QueryRow(
        fmt.Sprintf("SELECT COUNT(*) FROM %s WHERE %s = ?", table, field),
        value,
    ).Scan(&count)

    return count == 0, err
}

// 全局ID生成 (保证全局唯一)
type GlobalIDGenerator struct {
    unitID     uint16    // 单元ID (0-1023)
    sequence   uint32    // 序列号
    lastTime   int64     // 上次时间戳
    mu         sync.Mutex
}

func (g *GlobalIDGenerator) Generate() int64 {
    g.mu.Lock()
    defer g.mu.Unlock()

    now := time.Now().UnixMilli()

    if now == g.lastTime {
        g.sequence++
        if g.sequence >= 4096 {
            // 序列号用尽，等待下一毫秒
            for now <= g.lastTime {
                now = time.Now().UnixMilli()
            }
            g.sequence = 0
        }
    } else {
        g.sequence = 0
        g.lastTime = now
    }

    // 64位ID: 时间戳(41) + 单元ID(10) + 序列号(12)
    id := (now-epoch)<<22 | int64(g.unitID)<<12 | int64(g.sequence)
    return id
}
```

---

## 六、容灾演练

### 演练计划

```go
// 容灾演练框架
type DRDrill struct {
    name        string
    drillType   DrillType
    targetUnit  string
    duration    time.Duration
    rollback    func() error
}

type DrillType int

const (
    DrillTypeNetworkPartition DrillType = iota  // 网络分区
    DrillTypeServiceFailure                      // 服务故障
    DrillTypeDBFailover                          // 数据库故障转移
    DrillTypeFullDCFailure                       // 整个数据中心故障
)

// 执行演练
func (d *DRDrill) Execute() error {
    log.Printf("Starting DR drill: %s", d.name)

    // 1. 演练前检查
    if err := d.preCheck(); err != nil {
        return fmt.Errorf("pre-check failed: %w", err)
    }

    // 2. 通知相关人员
    d.notify("Drill starting")

    // 3. 记录基线指标
    baseline := captureMetrics()

    // 4. 注入故障
    if err := d.injectFailure(); err != nil {
        d.rollback()
        return fmt.Errorf("inject failure failed: %w", err)
    }

    // 5. 等待系统反应
    time.Sleep(d.duration)

    // 6. 验证容灾效果
    result := d.verify(baseline)

    // 7. 恢复
    if err := d.rollback(); err != nil {
        d.notify("CRITICAL: Rollback failed!")
        return fmt.Errorf("rollback failed: %w", err)
    }

    // 8. 生成报告
    d.generateReport(result)

    return nil
}

// 网络分区演练
func NetworkPartitionDrill(sourceUnit, targetUnit string) *DRDrill {
    return &DRDrill{
        name:       fmt.Sprintf("Network partition %s -> %s", sourceUnit, targetUnit),
        drillType:  DrillTypeNetworkPartition,
        targetUnit: targetUnit,
        duration:   5 * time.Minute,
        rollback: func() error {
            // 恢复网络
            return exec.Command("iptables", "-D", "OUTPUT",
                "-d", getUnitIP(targetUnit), "-j", "DROP").Run()
        },
    }
}

// 验证清单
type DrillVerification struct {
    ServiceAvailable  bool     // 服务是否可用
    DataConsistent    bool     // 数据是否一致
    FailoverTime      time.Duration  // 故障转移时间
    DataLoss          int64    // 数据丢失量
    ErrorRate         float64  // 错误率
    Latency           time.Duration  // 延迟
}
```

---

## 七、检查清单

### 容灾架构检查

- [ ] RPO/RTO 目标是否明确定义？
- [ ] 数据复制策略是否满足 RPO 要求？
- [ ] 故障切换流程是否满足 RTO 要求？
- [ ] 是否有完整的灾备文档？

### 多活架构检查

- [ ] 单元划分是否合理？
- [ ] 路由策略是否正确？
- [ ] 数据同步是否可靠？
- [ ] 冲突解决策略是否明确？

### 演练检查

- [ ] 是否定期进行容灾演练？
- [ ] 演练是否覆盖所有故障场景？
- [ ] 演练结果是否有改进跟踪？
- [ ] 是否有演练失败的应急预案？
