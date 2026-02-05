# 架构模式

## 一、分层架构 (Layered Architecture)

### 经典三层架构

```
┌─────────────────────────────────────────┐
│           表现层 (Presentation)          │
│         Controller / API / View          │
├─────────────────────────────────────────┤
│           业务逻辑层 (Business)           │
│              Service / Manager           │
├─────────────────────────────────────────┤
│           数据访问层 (Data Access)        │
│            Repository / DAO              │
└─────────────────────────────────────────┘
                    │
                    ▼
            ┌───────────────┐
            │   Database    │
            └───────────────┘
```

### 代码示例 (Go)

```go
// ===== 数据访问层 =====
type UserRepository interface {
    FindByID(ctx context.Context, id int64) (*User, error)
    Save(ctx context.Context, user *User) error
}

type mysqlUserRepository struct {
    db *sql.DB
}

func (r *mysqlUserRepository) FindByID(ctx context.Context, id int64) (*User, error) {
    row := r.db.QueryRowContext(ctx, "SELECT id, name, email FROM users WHERE id = ?", id)
    user := &User{}
    err := row.Scan(&user.ID, &user.Name, &user.Email)
    return user, err
}

// ===== 业务逻辑层 =====
type UserService struct {
    userRepo UserRepository
    cache    Cache
}

func (s *UserService) GetUser(ctx context.Context, id int64) (*User, error) {
    // 先查缓存
    if user, err := s.cache.GetUser(id); err == nil {
        return user, nil
    }

    // 查数据库
    user, err := s.userRepo.FindByID(ctx, id)
    if err != nil {
        return nil, err
    }

    // 回写缓存
    s.cache.SetUser(user)
    return user, nil
}

// ===== 表现层 =====
type UserHandler struct {
    userService *UserService
}

func (h *UserHandler) GetUser(w http.ResponseWriter, r *http.Request) {
    id, _ := strconv.ParseInt(r.URL.Query().Get("id"), 10, 64)

    user, err := h.userService.GetUser(r.Context(), id)
    if err != nil {
        http.Error(w, err.Error(), 500)
        return
    }

    json.NewEncoder(w).Encode(user)
}
```

### 优缺点

| 优点 | 缺点 |
|------|------|
| 职责清晰 | 层间调用开销 |
| 易于理解 | 可能导致贫血模型 |
| 易于测试 | 业务逻辑分散 |
| 团队分工明确 | 修改需要跨层 |

### 避坑指南

**坑1：跨层调用**
```go
// ❌ 表现层直接访问数据层
func (h *UserHandler) GetUser(w http.ResponseWriter, r *http.Request) {
    user, _ := h.userRepo.FindByID(r.Context(), id)  // 跳过 Service
}

// ✅ 严格遵循层次
func (h *UserHandler) GetUser(w http.ResponseWriter, r *http.Request) {
    user, _ := h.userService.GetUser(r.Context(), id)
}
```

**坑2：Service 层过于臃肿**
```go
// ❌ 一个 Service 做太多事
type UserService struct {
    // 用户相关
    // 订单相关
    // 支付相关
    // 通知相关
}

// ✅ 按领域拆分
type UserService struct { /* 用户相关 */ }
type OrderService struct { /* 订单相关 */ }
```

---

## 二、六边形架构 (Hexagonal Architecture)

也叫端口-适配器架构 (Ports and Adapters)

### 核心思想

```
                    ┌─────────────────────────────────┐
     HTTP API ──────▶│                                 │◀────── CLI
                    │     ┌─────────────────────┐     │
    gRPC API ──────▶│     │                     │     │◀────── 定时任务
                    │     │    Application      │     │
                    │     │    Core / Domain    │     │
   Message Queue ──▶│     │    (业务逻辑)        │     │
                    │     │                     │     │
                    │     └─────────────────────┘     │
                    │                                 │
                    │           Ports (接口)           │
                    │                                 │
                    └─────────────────────────────────┘
                              │           │
                    ┌─────────┘           └─────────┐
                    ▼                               ▼
              ┌──────────┐                   ┌──────────┐
              │  MySQL   │                   │  Redis   │
              │ Adapter  │                   │ Adapter  │
              └──────────┘                   └──────────┘
```

### 代码结构

```
project/
├── cmd/                        # 启动入口
│   ├── api/
│   └── worker/
├── internal/
│   ├── domain/                 # 领域层（核心）
│   │   ├── user/
│   │   │   ├── entity.go       # 实体
│   │   │   ├── repository.go   # 仓储接口（Port）
│   │   │   └── service.go      # 领域服务
│   │   └── order/
│   ├── application/            # 应用层
│   │   └── usecase/
│   │       └── create_order.go
│   └── infrastructure/         # 基础设施层（Adapters）
│       ├── persistence/
│       │   ├── mysql/
│       │   └── redis/
│       └── messaging/
│           └── kafka/
├── api/                        # API 适配器
│   ├── http/
│   └── grpc/
└── pkg/                        # 公共包
```

### 代码示例

```go
// ===== 领域层：定义接口（Port） =====
// internal/domain/user/repository.go
package user

type Repository interface {
    FindByID(ctx context.Context, id string) (*User, error)
    Save(ctx context.Context, user *User) error
}

// 领域服务
type Service struct {
    repo Repository
}

func (s *Service) Register(ctx context.Context, cmd RegisterCommand) (*User, error) {
    // 业务逻辑：验证、创建用户
    user := NewUser(cmd.Email, cmd.Name)
    if err := user.Validate(); err != nil {
        return nil, err
    }
    return user, s.repo.Save(ctx, user)
}

// ===== 基础设施层：实现接口（Adapter） =====
// internal/infrastructure/persistence/mysql/user_repo.go
package mysql

type UserRepository struct {
    db *sql.DB
}

func (r *UserRepository) FindByID(ctx context.Context, id string) (*user.User, error) {
    // MySQL 具体实现
}

func (r *UserRepository) Save(ctx context.Context, u *user.User) error {
    // MySQL 具体实现
}

// ===== API 层：另一个 Adapter =====
// api/http/user_handler.go
type UserHandler struct {
    userService *user.Service
}

func (h *UserHandler) Register(w http.ResponseWriter, r *http.Request) {
    var cmd user.RegisterCommand
    json.NewDecoder(r.Body).Decode(&cmd)

    u, err := h.userService.Register(r.Context(), cmd)
    // ...
}
```

### 优势

1. **可测试性**：核心逻辑不依赖外部系统，可用 Mock
2. **可替换性**：更换数据库只需新增 Adapter
3. **业务聚焦**：领域逻辑与技术细节分离

### 适用场景

- 复杂业务系统
- 需要对接多种外部系统
- 需要高测试覆盖率的项目

---

## 三、CQRS 模式

Command Query Responsibility Segregation（命令查询职责分离）

### 核心思想

```
                    ┌─────────────────────────────────────┐
                    │           API Gateway               │
                    └─────────────────────────────────────┘
                              │              │
              ┌───────────────┘              └───────────────┐
              ▼                                              ▼
    ┌─────────────────────┐                    ┌─────────────────────┐
    │   Command Service   │                    │   Query Service     │
    │   (写操作)           │                    │   (读操作)           │
    │                     │                    │                     │
    │  • 创建订单          │                    │  • 查询订单列表      │
    │  • 更新订单          │                    │  • 订单详情          │
    │  • 取消订单          │                    │  • 订单统计          │
    └─────────────────────┘                    └─────────────────────┘
              │                                              │
              ▼                                              ▼
    ┌─────────────────────┐                    ┌─────────────────────┐
    │   Write Database    │  ═══ 同步 ═══▶    │   Read Database     │
    │   (MySQL 主库)       │                    │   (ES / Redis)      │
    └─────────────────────┘                    └─────────────────────┘
```

### 代码示例

```go
// ===== Command 端 =====
type CreateOrderCommand struct {
    UserID    string
    ProductID string
    Quantity  int
}

type OrderCommandService struct {
    orderRepo  OrderRepository
    eventBus   EventBus
}

func (s *OrderCommandService) CreateOrder(ctx context.Context, cmd CreateOrderCommand) (string, error) {
    // 1. 业务逻辑
    order := NewOrder(cmd.UserID, cmd.ProductID, cmd.Quantity)

    // 2. 持久化
    if err := s.orderRepo.Save(ctx, order); err != nil {
        return "", err
    }

    // 3. 发布事件（用于同步到读库）
    s.eventBus.Publish(OrderCreatedEvent{
        OrderID:   order.ID,
        UserID:    order.UserID,
        CreatedAt: order.CreatedAt,
    })

    return order.ID, nil
}

// ===== Query 端 =====
type OrderQueryService struct {
    es *elasticsearch.Client
}

func (s *OrderQueryService) SearchOrders(ctx context.Context, query SearchQuery) ([]OrderView, error) {
    // 从 ES 查询，支持复杂搜索
    result, err := s.es.Search(
        s.es.Search.WithIndex("orders"),
        s.es.Search.WithQuery(buildQuery(query)),
    )
    // ...
}

// ===== 事件处理（同步读库） =====
type OrderEventHandler struct {
    es *elasticsearch.Client
}

func (h *OrderEventHandler) HandleOrderCreated(event OrderCreatedEvent) {
    // 同步到 ES
    h.es.Index(
        "orders",
        strings.NewReader(toJSON(event)),
        h.es.Index.WithDocumentID(event.OrderID),
    )
}
```

### CQRS + Event Sourcing

```
┌─────────────┐
│   Command   │
└─────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│           Event Store                    │
│  ┌─────────────────────────────────┐    │
│  │ OrderCreated { orderId: 1, ... }│    │
│  │ OrderPaid { orderId: 1, ... }   │    │
│  │ OrderShipped { orderId: 1, ... }│    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
       │
       ▼ 投影（Projection）
┌─────────────────────────────────────────┐
│              Read Model                  │
│   ┌───────────────────────────────┐     │
│   │  Order { id: 1, status: ... } │     │
│   └───────────────────────────────┘     │
└─────────────────────────────────────────┘
```

### 适用场景

| 适用 | 不适用 |
|------|--------|
| 读写比例差异大（读多写少） | 简单 CRUD 系统 |
| 读写模型差异大 | 数据一致性要求极高 |
| 需要复杂查询 | 团队经验不足 |
| 高并发读取 | 小型项目 |

### 避坑指南

**坑1：数据一致性**
```
问题：写库和读库数据不一致

解决方案：
1. 最终一致性：接受短暂不一致
2. 版本号：乐观锁检测冲突
3. 刷新策略：关键查询强制读主库
```

**坑2：复杂度过高**
```
问题：简单项目使用 CQRS 增加维护成本

建议：
1. 先用简单架构
2. 当读写模型差异明显时再引入
3. 可以只在部分模块使用 CQRS
```

---

## 四、事件驱动架构 (EDA)

### 基本模式

```
┌─────────────┐      Event      ┌─────────────┐
│  Producer   │ ═══════════════▶│   Broker    │
│ (事件生产者) │                 │ (消息中间件) │
└─────────────┘                 └─────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
              ┌──────────┐      ┌──────────┐      ┌──────────┐
              │Consumer A│      │Consumer B│      │Consumer C│
              └──────────┘      └──────────┘      └──────────┘
```

### 事件类型

```go
// 1. 领域事件（Domain Event）- 业务发生的事实
type OrderCreatedEvent struct {
    OrderID   string    `json:"order_id"`
    UserID    string    `json:"user_id"`
    Amount    float64   `json:"amount"`
    CreatedAt time.Time `json:"created_at"`
}

// 2. 集成事件（Integration Event）- 跨服务通信
type PaymentCompletedEvent struct {
    PaymentID string    `json:"payment_id"`
    OrderID   string    `json:"order_id"`
    Status    string    `json:"status"`
    PaidAt    time.Time `json:"paid_at"`
}

// 3. 命令事件（Command Event）- 请求执行操作
type SendEmailCommand struct {
    To      string `json:"to"`
    Subject string `json:"subject"`
    Body    string `json:"body"`
}
```

### 实战：订单系统

```
                        ┌───────────────┐
                        │  订单服务      │
                        │ Order Service │
                        └───────────────┘
                               │
                               │ OrderCreated
                               ▼
                        ┌───────────────┐
                        │    Kafka      │
                        └───────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │ 库存服务     │     │ 通知服务     │     │ 数据分析    │
    │ 扣减库存     │     │ 发送通知     │     │ 更新报表    │
    └─────────────┘     └─────────────┘     └─────────────┘
```

```python
# 订单服务 - 发布事件
class OrderService:
    def create_order(self, user_id: str, items: List[Item]) -> Order:
        order = Order.create(user_id, items)
        self.order_repo.save(order)

        # 发布事件（异步、解耦）
        self.event_bus.publish("order.created", OrderCreatedEvent(
            order_id=order.id,
            user_id=user_id,
            items=items,
            total_amount=order.total_amount
        ))

        return order

# 库存服务 - 订阅事件
class InventoryConsumer:
    @subscribe("order.created")
    def handle_order_created(self, event: OrderCreatedEvent):
        for item in event.items:
            self.inventory_service.deduct(item.product_id, item.quantity)

# 通知服务 - 订阅事件
class NotificationConsumer:
    @subscribe("order.created")
    def handle_order_created(self, event: OrderCreatedEvent):
        self.email_service.send_order_confirmation(event.user_id, event.order_id)
```

### Saga 模式（分布式事务）

```
订单创建 Saga:

┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ 创建订单 │───▶│ 扣减库存 │───▶│ 扣减余额 │───▶│ 完成订单 │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
     │              │              │
     │              │              ▼ 失败
     │              │         ┌─────────┐
     │              │◀────────│ 回滚余额 │
     │              │         └─────────┘
     │              ▼
     │         ┌─────────┐
     │◀────────│ 回滚库存 │
     │         └─────────┘
     ▼
┌─────────┐
│ 取消订单 │
└─────────┘
```

### 避坑指南

**坑1：事件顺序**
```
问题：事件到达顺序与发送顺序不一致

解决：
1. 同一实体的事件发到同一分区
2. 事件携带版本号/时间戳
3. 消费端做幂等处理
```

**坑2：事件丢失**
```
问题：服务宕机导致事件丢失

解决：
1. 本地事件表（Outbox Pattern）
2. 事务消息
3. 消费确认机制
```

**坑3：事件风暴**
```
问题：一个事件触发大量下游事件

解决：
1. 限流
2. 优先级队列
3. 熔断降级
```

---

## 五、架构模式选择指南

| 场景 | 推荐架构 |
|------|---------|
| 小型项目 / MVP | 分层架构 |
| 复杂业务领域 | 六边形架构 + DDD |
| 读写差异大 | CQRS |
| 服务间解耦 | 事件驱动 |
| 高并发读 | CQRS + 缓存 |
| 复杂流程编排 | Saga |
