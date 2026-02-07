# 分布式事务

## 一、分布式事务问题

```
单体架构（本地事务）：
┌─────────────────────────────────────────┐
│              Application                │
│                                         │
│    BEGIN TRANSACTION                    │
│    ├── 扣减库存                          │
│    ├── 创建订单                          │
│    └── 扣减余额                          │
│    COMMIT / ROLLBACK                    │
│                                         │
└─────────────────────────────────────────┘
                    │
                    ▼
            ┌───────────────┐
            │   Database    │  ← 同一个数据库，ACID 保证
            └───────────────┘


微服务架构（分布式事务）：
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  库存服务     │  │  订单服务     │  │  账户服务     │
│  扣减库存     │  │  创建订单     │  │  扣减余额     │
└──────────────┘  └──────────────┘  └──────────────┘
        │                │                │
        ▼                ▼                ▼
   ┌─────────┐      ┌─────────┐      ┌─────────┐
   │ MySQL A │      │ MySQL B │      │ MySQL C │
   └─────────┘      └─────────┘      └─────────┘

问题：三个独立数据库，如何保证一致性？
```

## 二、解决方案概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                      分布式事务解决方案                              │
├─────────────────────────────────────────────────────────────────────┤
│  方案          │ 一致性   │ 性能  │ 复杂度  │ 适用场景              │
├─────────────────────────────────────────────────────────────────────┤
│  2PC           │ 强一致   │ 低    │ 中      │ 数据库间事务          │
│  3PC           │ 强一致   │ 低    │ 高      │ 理论价值              │
│  TCC           │ 最终一致 │ 中    │ 高      │ 高并发、强一致需求    │
│  Saga          │ 最终一致 │ 高    │ 中      │ 长事务、跨服务        │
│  本地消息表    │ 最终一致 │ 高    │ 低      │ 异步场景              │
│  事务消息      │ 最终一致 │ 高    │ 低      │ 消息驱动              │
│  最大努力通知  │ 最终一致 │ 高    │ 低      │ 跨系统通知            │
└─────────────────────────────────────────────────────────────────────┘
```

## 三、2PC（两阶段提交）

### 原理

```
                    ┌─────────────────┐
                    │   协调者 (TM)    │
                    └─────────────────┘
                            │
           ┌────────────────┼────────────────┐
           │                │                │
           ▼                ▼                ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ 参与者 A    │  │ 参与者 B    │  │ 参与者 C    │
    └─────────────┘  └─────────────┘  └─────────────┘

阶段一：Prepare（准备）
┌──────────────────────────────────────────────────────────────┐
│  协调者 ──Prepare──▶ 参与者 A  ──▶ 执行事务，写 undo/redo log │
│          ──Prepare──▶ 参与者 B  ──▶ 执行事务，写 undo/redo log │
│          ──Prepare──▶ 参与者 C  ──▶ 执行事务，写 undo/redo log │
│                                                              │
│  参与者 ──Vote Yes/No──▶ 协调者                               │
└──────────────────────────────────────────────────────────────┘

阶段二：Commit/Rollback
┌──────────────────────────────────────────────────────────────┐
│  如果全部 Yes:                                                │
│    协调者 ──Commit──▶ 所有参与者 ──▶ 提交事务                 │
│                                                              │
│  如果有 No:                                                   │
│    协调者 ──Rollback──▶ 所有参与者 ──▶ 回滚事务               │
└──────────────────────────────────────────────────────────────┘
```

### XA 事务实现

```java
// Java XA 事务示例
public class XATransactionExample {

    public void transfer(String fromAccount, String toAccount, BigDecimal amount) throws Exception {
        // 获取 XA 数据源
        XADataSource xaDS1 = getXADataSource("db1");
        XADataSource xaDS2 = getXADataSource("db2");

        // 获取 XA 连接
        XAConnection xaConn1 = xaDS1.getXAConnection();
        XAConnection xaConn2 = xaDS2.getXAConnection();

        // 获取 XA 资源
        XAResource xaRes1 = xaConn1.getXAResource();
        XAResource xaRes2 = xaConn2.getXAResource();

        // 创建事务 ID
        Xid xid1 = createXid(1);
        Xid xid2 = createXid(2);

        try {
            // 阶段一：Prepare
            xaRes1.start(xid1, XAResource.TMNOFLAGS);
            // 执行扣款 SQL
            xaRes1.end(xid1, XAResource.TMSUCCESS);
            int prepare1 = xaRes1.prepare(xid1);

            xaRes2.start(xid2, XAResource.TMNOFLAGS);
            // 执行加款 SQL
            xaRes2.end(xid2, XAResource.TMSUCCESS);
            int prepare2 = xaRes2.prepare(xid2);

            // 阶段二：Commit
            if (prepare1 == XAResource.XA_OK && prepare2 == XAResource.XA_OK) {
                xaRes1.commit(xid1, false);
                xaRes2.commit(xid2, false);
            } else {
                xaRes1.rollback(xid1);
                xaRes2.rollback(xid2);
            }
        } catch (Exception e) {
            xaRes1.rollback(xid1);
            xaRes2.rollback(xid2);
            throw e;
        }
    }
}
```

### 2PC 问题

```
1. 同步阻塞
   - Prepare 阶段所有参与者阻塞等待
   - 性能差，不适合高并发

2. 单点故障
   - 协调者宕机，参与者一直阻塞

3. 数据不一致
   - 阶段二部分参与者收到 Commit，部分没收到
   - 收到的提交了，没收到的超时后可能回滚
```

## 四、TCC（Try-Confirm-Cancel）

### 原理

```
TCC 把事务分成三个操作：
┌─────────────────────────────────────────────────────────────┐
│  Try     │ 资源预留                                         │
│          │ 检查 + 预留资源，不真正执行                        │
├──────────┼──────────────────────────────────────────────────┤
│  Confirm │ 确认执行                                         │
│          │ 真正执行业务，使用 Try 阶段预留的资源              │
├──────────┼──────────────────────────────────────────────────┤
│  Cancel  │ 取消执行                                         │
│          │ 释放 Try 阶段预留的资源                           │
└──────────┴──────────────────────────────────────────────────┘

示例：转账 100 元
┌─────────────────────────────────────────────────────────────┐
│  Try:                                                       │
│    账户 A: 冻结 100 元（余额不变，冻结金额 +100）              │
│    账户 B: 无操作（或预增加）                                 │
│                                                             │
│  Confirm:                                                   │
│    账户 A: 扣减冻结金额 100，余额 -100                        │
│    账户 B: 余额 +100                                         │
│                                                             │
│  Cancel:                                                    │
│    账户 A: 释放冻结金额 100                                   │
│    账户 B: 无操作                                            │
└─────────────────────────────────────────────────────────────┘
```

### 代码实现

```go
// TCC 接口定义
type TCCService interface {
    Try(ctx context.Context, req interface{}) error
    Confirm(ctx context.Context, req interface{}) error
    Cancel(ctx context.Context, req interface{}) error
}

// 库存服务 TCC 实现
type InventoryTCCService struct {
    db *sql.DB
}

func (s *InventoryTCCService) Try(ctx context.Context, req *DeductRequest) error {
    // 冻结库存
    _, err := s.db.ExecContext(ctx, `
        UPDATE inventory
        SET frozen = frozen + ?
        WHERE product_id = ? AND (stock - frozen) >= ?
    `, req.Quantity, req.ProductID, req.Quantity)

    return err
}

func (s *InventoryTCCService) Confirm(ctx context.Context, req *DeductRequest) error {
    // 扣减库存和冻结
    _, err := s.db.ExecContext(ctx, `
        UPDATE inventory
        SET stock = stock - ?, frozen = frozen - ?
        WHERE product_id = ?
    `, req.Quantity, req.Quantity, req.ProductID)

    return err
}

func (s *InventoryTCCService) Cancel(ctx context.Context, req *DeductRequest) error {
    // 释放冻结
    _, err := s.db.ExecContext(ctx, `
        UPDATE inventory
        SET frozen = frozen - ?
        WHERE product_id = ? AND frozen >= ?
    `, req.Quantity, req.ProductID, req.Quantity)

    return err
}

// TCC 事务协调器
type TCCCoordinator struct {
    inventoryTCC  *InventoryTCCService
    accountTCC    *AccountTCCService
    txLogRepo     *TxLogRepository
}

func (c *TCCCoordinator) Execute(ctx context.Context, order *Order) error {
    txID := uuid.New().String()

    // 记录事务日志
    c.txLogRepo.Create(txID, "TRYING")

    // Try 阶段
    if err := c.inventoryTCC.Try(ctx, order.Items); err != nil {
        c.txLogRepo.Update(txID, "TRY_FAILED")
        return err
    }
    if err := c.accountTCC.Try(ctx, order.Amount); err != nil {
        c.txLogRepo.Update(txID, "TRY_FAILED")
        c.inventoryTCC.Cancel(ctx, order.Items)  // 补偿
        return err
    }

    c.txLogRepo.Update(txID, "TRIED")

    // Confirm 阶段
    if err := c.inventoryTCC.Confirm(ctx, order.Items); err != nil {
        // Confirm 失败需要重试
        c.txLogRepo.Update(txID, "CONFIRM_FAILED")
        return err
    }
    if err := c.accountTCC.Confirm(ctx, order.Amount); err != nil {
        c.txLogRepo.Update(txID, "CONFIRM_FAILED")
        return err
    }

    c.txLogRepo.Update(txID, "CONFIRMED")
    return nil
}
```

### TCC 要点

```
1. 空回滚
   - Try 未执行，直接收到 Cancel
   - Cancel 需要判断 Try 是否执行过

2. 幂等性
   - Confirm/Cancel 可能被重复调用
   - 需要保证幂等

3. 悬挂
   - Cancel 先于 Try 执行
   - 需要防止 Try 在 Cancel 后执行

解决方案：事务状态记录
┌──────────────────────────────────────────────┐
│  CREATE TABLE tcc_transaction (             │
│      tx_id VARCHAR(64) PRIMARY KEY,         │
│      status VARCHAR(20),  -- TRYING/TRIED/  │
│                          -- CONFIRMING/     │
│                          -- CONFIRMED/      │
│                          -- CANCELLING/     │
│                          -- CANCELLED       │
│      created_at TIMESTAMP,                  │
│      updated_at TIMESTAMP                   │
│  );                                         │
└──────────────────────────────────────────────┘
```

## 五、Saga 模式

### 原理

```
Saga 把长事务拆分成多个本地事务，每个事务有对应的补偿操作

正向操作：T1 → T2 → T3 → T4
补偿操作：C1 ← C2 ← C3 ← C4

执行流程：
┌─────────────────────────────────────────────────────────────┐
│  成功场景：                                                  │
│  T1(创建订单) → T2(扣库存) → T3(扣款) → T4(发货) → 完成     │
│                                                             │
│  失败场景（T3 扣款失败）：                                    │
│  T1(创建订单) → T2(扣库存) → T3(扣款失败)                    │
│                                  ↓                          │
│  C2(恢复库存) ← C1(取消订单) ← 触发补偿                      │
└─────────────────────────────────────────────────────────────┘
```

### 编排式 Saga

```go
// 使用消息队列编排
type SagaOrchestrator struct {
    orderService     OrderService
    inventoryService InventoryService
    paymentService   PaymentService
    messageBus       MessageBus
}

func (s *SagaOrchestrator) CreateOrder(ctx context.Context, req *CreateOrderRequest) error {
    sagaID := uuid.New().String()

    // Step 1: 创建订单
    order, err := s.orderService.Create(ctx, req)
    if err != nil {
        return err
    }
    s.saveSagaLog(sagaID, "ORDER_CREATED", order.ID)

    // Step 2: 扣减库存
    err = s.inventoryService.Deduct(ctx, order.Items)
    if err != nil {
        // 补偿：取消订单
        s.orderService.Cancel(ctx, order.ID)
        return err
    }
    s.saveSagaLog(sagaID, "INVENTORY_DEDUCTED", order.ID)

    // Step 3: 扣款
    err = s.paymentService.Pay(ctx, order.UserID, order.Amount)
    if err != nil {
        // 补偿：恢复库存、取消订单
        s.inventoryService.Restore(ctx, order.Items)
        s.orderService.Cancel(ctx, order.ID)
        return err
    }
    s.saveSagaLog(sagaID, "PAYMENT_COMPLETED", order.ID)

    // Step 4: 完成订单
    s.orderService.Complete(ctx, order.ID)
    s.saveSagaLog(sagaID, "SAGA_COMPLETED", order.ID)

    return nil
}
```

### 协同式 Saga（事件驱动）

```go
// 每个服务监听事件，自主执行和发布

// 订单服务
func (s *OrderService) HandleOrderCreated(event OrderCreatedEvent) {
    // 发布事件，触发库存扣减
    s.eventBus.Publish(ReserveInventoryCommand{
        OrderID: event.OrderID,
        Items:   event.Items,
    })
}

// 库存服务
func (s *InventoryService) HandleReserveInventory(cmd ReserveInventoryCommand) {
    err := s.deduct(cmd.Items)
    if err != nil {
        // 发布失败事件
        s.eventBus.Publish(InventoryReservationFailed{
            OrderID: cmd.OrderID,
            Reason:  err.Error(),
        })
        return
    }

    // 发布成功事件
    s.eventBus.Publish(InventoryReserved{
        OrderID: cmd.OrderID,
    })
}

// 订单服务监听失败事件
func (s *OrderService) HandleInventoryReservationFailed(event InventoryReservationFailed) {
    // 补偿：取消订单
    s.cancelOrder(event.OrderID)
}
```

## 六、本地消息表

```
原理：
1. 业务操作和消息写入在同一个本地事务
2. 定时任务扫描消息表，发送到 MQ
3. 消费者消费消息，执行下游操作

┌──────────────────────────────────────────────────────────────┐
│                      本地消息表方案                           │
│                                                              │
│  ┌─────────────────┐                    ┌─────────────────┐  │
│  │    订单服务      │                    │    库存服务      │  │
│  │                 │                    │                 │  │
│  │  BEGIN TX       │                    │                 │  │
│  │  ├─ 创建订单    │                    │                 │  │
│  │  └─ 写消息表    │                    │                 │  │
│  │  COMMIT         │                    │                 │  │
│  │                 │      Kafka         │                 │  │
│  │  定时任务 ──────┼──────────────────▶│  消费消息       │  │
│  │  扫描消息表     │                    │  扣减库存       │  │
│  │  发送到 MQ      │                    │  ACK            │  │
│  │  标记已发送     │◀──────────────────┼──回调确认       │  │
│  │                 │                    │                 │  │
│  └─────────────────┘                    └─────────────────┘  │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────┐                │
│  │  local_message_table                    │                │
│  │  ┌───────┬────────┬────────┬────────┐  │                │
│  │  │  id   │ content│ status │ retry  │  │                │
│  │  ├───────┼────────┼────────┼────────┤  │                │
│  │  │  1    │ {...}  │ SENT   │   0    │  │                │
│  │  │  2    │ {...}  │ PENDING│   0    │  │                │
│  │  └───────┴────────┴────────┴────────┘  │                │
│  └─────────────────────────────────────────┘                │
└──────────────────────────────────────────────────────────────┘
```

```python
# 本地消息表实现
class LocalMessageService:

    def create_order_with_message(self, order: Order):
        """创建订单并写入本地消息"""
        with self.db.transaction():
            # 1. 创建订单
            self.order_repo.save(order)

            # 2. 写入本地消息表
            message = LocalMessage(
                message_id=str(uuid.uuid4()),
                topic="inventory.deduct",
                content=json.dumps({
                    "order_id": order.id,
                    "items": order.items
                }),
                status="PENDING"
            )
            self.message_repo.save(message)

    def scan_and_send(self):
        """定时任务：扫描并发送消息"""
        messages = self.message_repo.find_pending(limit=100)

        for msg in messages:
            try:
                self.kafka.send(msg.topic, msg.content)
                self.message_repo.update_status(msg.id, "SENT")
            except Exception as e:
                self.message_repo.increment_retry(msg.id)
                if msg.retry_count >= 3:
                    self.message_repo.update_status(msg.id, "FAILED")
                    self.alert(f"消息发送失败: {msg.id}")
```

## 七、事务消息（RocketMQ）

```
RocketMQ 事务消息流程：

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Producer                    RocketMQ                      │
│      │                           │                          │
│      │ ① 发送半消息（Half）       │                          │
│      │ ─────────────────────────▶│                          │
│      │                           │ 消息暂不投递              │
│      │                           │                          │
│      │ ② 执行本地事务             │                          │
│      │                           │                          │
│      │ ③ 提交/回滚               │                          │
│      │ ─────────────────────────▶│                          │
│      │                           │                          │
│      │    如果超时未收到确认：    │                          │
│      │ ④ 回查本地事务状态         │                          │
│      │ ◀─────────────────────────│                          │
│      │                           │                          │
│      │ ⑤ 返回事务状态             │                          │
│      │ ─────────────────────────▶│                          │
│      │                           │ 投递或丢弃                │
│                                  │                          │
│                                  │            Consumer      │
│                                  │ ──────────────────▶│    │
│                                  │                    │    │
└─────────────────────────────────────────────────────────────┘
```

```java
// RocketMQ 事务消息示例
public class TransactionMessageProducer {

    private TransactionMQProducer producer;

    public void sendTransactionMessage(Order order) throws Exception {
        Message msg = new Message("ORDER_TOPIC", JSON.toJSONBytes(order));

        // 发送事务消息
        producer.sendMessageInTransaction(msg, new LocalTransactionExecuter() {
            @Override
            public LocalTransactionState executeLocalTransactionBranch(Message msg, Object arg) {
                try {
                    // 执行本地事务
                    orderService.createOrder(order);
                    return LocalTransactionState.COMMIT_MESSAGE;
                } catch (Exception e) {
                    return LocalTransactionState.ROLLBACK_MESSAGE;
                }
            }

            @Override
            public LocalTransactionState checkLocalTransactionState(MessageExt msg) {
                // 回查本地事务状态
                Order order = orderService.getOrder(orderId);
                if (order != null && order.getStatus() == OrderStatus.CREATED) {
                    return LocalTransactionState.COMMIT_MESSAGE;
                }
                return LocalTransactionState.ROLLBACK_MESSAGE;
            }
        }, null);
    }
}
```

## 八、方案选择指南

```
┌─────────────────────────────────────────────────────────────┐
│                    分布式事务选择决策树                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  需要强一致吗？                                              │
│       │                                                     │
│       ├── 是 ──▶ 性能要求高吗？                             │
│       │              │                                      │
│       │              ├── 是 ──▶ TCC                         │
│       │              └── 否 ──▶ 2PC / XA                    │
│       │                                                     │
│       └── 否 ──▶ 需要同步返回结果吗？                        │
│                      │                                      │
│                      ├── 是 ──▶ Saga                        │
│                      │                                      │
│                      └── 否 ──▶ 有 MQ 吗？                   │
│                                    │                        │
│                                    ├── 是 ──▶ 事务消息       │
│                                    └── 否 ──▶ 本地消息表     │
│                                                             │
└─────────────────────────────────────────────────────────────┘

实际场景推荐：
- 跨数据库事务：XA（性能要求不高）或 Saga
- 微服务间事务：Saga 或 TCC
- 订单支付：TCC 或 事务消息
- 异步操作：本地消息表 / 事务消息
- 跨系统通知：最大努力通知
```
