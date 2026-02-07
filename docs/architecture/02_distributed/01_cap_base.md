# CAP 与 BASE 理论

## 一、CAP 定理

### 核心概念

```
CAP 定理：分布式系统不可能同时满足以下三个特性，最多只能满足其中两个。

┌─────────────────────────────────────────────────────────────┐
│                         CAP                                  │
│                                                             │
│                     Consistency                              │
│                    (一致性)                                   │
│                        /\                                    │
│                       /  \                                   │
│                      /    \                                  │
│                     /  CA  \     CP: 牺牲可用性              │
│                    /________\    CA: 单机系统（无分区）       │
│                   /    AP    \   AP: 牺牲一致性              │
│                  /____________\                              │
│       Availability          Partition Tolerance              │
│        (可用性)                (分区容错)                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 三个特性详解

#### 1. 一致性 (Consistency)

```
所有节点在同一时刻看到相同的数据

场景：用户修改密码

┌────────────────────────────────────────────────────────────┐
│                    强一致性                                 │
│                                                            │
│  Client ──写密码──▶ Node A                                  │
│                      │                                      │
│                      ▼ 同步复制                             │
│                    Node B                                   │
│                      │                                      │
│                      ▼ 同步复制                             │
│                    Node C                                   │
│                      │                                      │
│                      ▼                                      │
│  Client ◀──响应成功──                                       │
│                                                            │
│  此时任意节点读取都是新密码                                  │
└────────────────────────────────────────────────────────────┘
```

#### 2. 可用性 (Availability)

```
每个请求都能收到响应（不保证数据最新）

┌────────────────────────────────────────────────────────────┐
│                    高可用性                                 │
│                                                            │
│  Client ──请求──▶ Node A (正常) ──▶ 立即响应                │
│                                                            │
│  Client ──请求──▶ Node B (正常) ──▶ 立即响应                │
│                                                            │
│  Client ──请求──▶ Node C (故障) ──▶ 路由到其他节点          │
│                                                            │
│  保证：只要有节点存活，就能响应请求                           │
└────────────────────────────────────────────────────────────┘
```

#### 3. 分区容错 (Partition Tolerance)

```
网络分区发生时系统仍能运作

┌────────────────────────────────────────────────────────────┐
│                    网络分区                                 │
│                                                            │
│      ┌─────────┐           ┌─────────┐                     │
│      │ Node A  │   ╳ ╳ ╳   │ Node B  │                     │
│      │ Node C  │   网络中断  │ Node D  │                     │
│      └─────────┘           └─────────┘                     │
│        分区 1                分区 2                         │
│                                                            │
│  分区容错：两个分区都能继续服务（但数据可能不一致）           │
└────────────────────────────────────────────────────────────┘
```

### CAP 选择

```
┌─────────────────────────────────────────────────────────────┐
│              为什么必须选择 P（分区容错）？                    │
│                                                             │
│  分布式系统中，网络分区是不可避免的现实：                      │
│  • 网络延迟/抖动                                             │
│  • 路由器故障                                                │
│  • 机房断连                                                  │
│                                                             │
│  因此实际选择是：CP 或 AP                                    │
└─────────────────────────────────────────────────────────────┘

CP 系统（牺牲可用性）：
┌─────────────────────────────────────────────┐
│ 场景：银行转账、库存扣减                      │
│ 特点：分区时拒绝服务，保证数据一致            │
│ 实现：ZooKeeper, etcd, HBase                 │
└─────────────────────────────────────────────┘

AP 系统（牺牲一致性）：
┌─────────────────────────────────────────────┐
│ 场景：社交动态、商品详情                      │
│ 特点：分区时继续服务，最终一致                │
│ 实现：Cassandra, DynamoDB, Eureka            │
└─────────────────────────────────────────────┘
```

### 实际系统的 CAP 选择

| 系统 | 选择 | 说明 |
|------|------|------|
| MySQL 主从 | CA | 单机强一致，主从最终一致 |
| MySQL Cluster | CP | NDB 存储引擎，同步复制 |
| Redis Sentinel | AP | 故障转移时可能数据丢失 |
| Redis Cluster | AP | 异步复制，可能数据丢失 |
| ZooKeeper | CP | 强一致，分区时少数派不可用 |
| Consul | CP | 默认强一致，可配置 AP |
| Eureka | AP | 可用性优先，最终一致 |
| MongoDB | CP/AP | 可配置 |
| Cassandra | AP | 可调节一致性级别 |

---

## 二、BASE 理论

### 核心概念

```
BASE 是对 CAP 中 AP 的延伸，牺牲强一致性换取可用性

BA - Basically Available（基本可用）
S  - Soft State（软状态）
E  - Eventually Consistent（最终一致）

┌─────────────────────────────────────────────────────────────┐
│                    ACID vs BASE                              │
├─────────────────────────────────────────────────────────────┤
│         ACID                    │           BASE             │
├─────────────────────────────────┼───────────────────────────┤
│  Atomicity（原子性）            │  Basically Available       │
│  Consistency（一致性）          │  Soft State                │
│  Isolation（隔离性）            │  Eventually Consistent     │
│  Durability（持久性）           │                           │
├─────────────────────────────────┼───────────────────────────┤
│  强一致性                       │  最终一致性                │
│  悲观锁                         │  乐观锁                    │
│  实时响应                       │  允许延迟                  │
│  单机/强同步                    │  分布式/异步               │
└─────────────────────────────────┴───────────────────────────┘
```

### 1. 基本可用 (Basically Available)

```python
# 示例：电商大促时的降级策略

class ProductService:
    def get_product_detail(self, product_id: str) -> ProductDetail:
        try:
            # 正常情况：完整数据
            product = self.db.get_product(product_id)
            reviews = self.review_service.get_reviews(product_id)
            recommendations = self.recommend_service.get(product_id)

            return ProductDetail(
                product=product,
                reviews=reviews,
                recommendations=recommendations
            )
        except TimeoutError:
            # 降级：返回基本信息
            product = self.cache.get_product(product_id)
            return ProductDetail(
                product=product,
                reviews=[],  # 暂不展示评论
                recommendations=[]  # 暂不展示推荐
            )

# 基本可用的体现：
# 1. 响应时间变长（但仍能响应）
# 2. 功能降级（核心功能可用）
# 3. 流量削峰（排队、限流）
```

### 2. 软状态 (Soft State)

```
软状态：允许系统中的数据存在中间状态

┌─────────────────────────────────────────────────────────────┐
│                    订单状态流转                              │
│                                                             │
│  [待支付] ──支付成功──▶ [支付中] ──确认到账──▶ [已支付]      │
│              │              │                               │
│              │              │ 软状态（中间状态）              │
│              │              │ • 支付请求已发送               │
│              │              │ • 等待支付网关回调             │
│              │              │ • 可能持续几秒到几分钟         │
│              │                                              │
│              └──支付失败──▶ [支付失败]                       │
└─────────────────────────────────────────────────────────────┘
```

```python
# 软状态示例：订单支付流程
class OrderPaymentService:
    def pay(self, order_id: str, amount: Decimal):
        # 1. 更新为软状态
        self.order_repo.update_status(order_id, OrderStatus.PAYING)

        # 2. 异步调用支付网关
        payment_result = self.payment_gateway.pay_async(order_id, amount)

        # 3. 等待回调更新最终状态
        # 回调处理在另一个方法

    def handle_payment_callback(self, callback: PaymentCallback):
        if callback.success:
            self.order_repo.update_status(callback.order_id, OrderStatus.PAID)
            self.event_bus.publish(OrderPaidEvent(callback.order_id))
        else:
            self.order_repo.update_status(callback.order_id, OrderStatus.PAYMENT_FAILED)
```

### 3. 最终一致性 (Eventually Consistent)

```
最终一致性：经过一段时间后，所有副本最终达到一致状态

一致性强度（从弱到强）：

┌─────────────────────────────────────────────────────────────┐
│  最终一致性         因果一致性         顺序一致性        强一致性  │
│     (Eventual)       (Causal)        (Sequential)     (Strong) │
│        │               │                 │               │     │
│        ▼               ▼                 ▼               ▼     │
│    [最弱]  ────────────────────────────────────────  [最强]    │
│                                                             │
│  实现复杂度：低 ──────────────────────────────────▶ 高        │
│  性能/可用性：高 ──────────────────────────────────▶ 低        │
└─────────────────────────────────────────────────────────────┘
```

#### 最终一致性的实现模式

```python
# 1. 读己之写（Read Your Writes）
# 用户写入后能立即读到自己的写入

class UserProfileService:
    def update_profile(self, user_id: str, data: dict):
        # 写入主库
        self.master_db.update(user_id, data)
        # 写入本地缓存，保证用户能读到
        self.local_cache.set(f"profile:{user_id}", data, ttl=60)

    def get_profile(self, user_id: str):
        # 先查本地缓存
        cached = self.local_cache.get(f"profile:{user_id}")
        if cached:
            return cached
        # 缓存未命中，查从库
        return self.slave_db.get(user_id)


# 2. 单调读（Monotonic Reads）
# 用户不会读到比之前更旧的数据

class MonotonicReadService:
    def read(self, user_id: str, key: str):
        # 记录用户上次读取的版本
        last_version = self.get_user_read_version(user_id, key)

        # 选择版本不低于 last_version 的副本读取
        replicas = self.get_replicas_with_min_version(key, last_version)
        data = replicas[0].read(key)

        # 更新用户读取版本
        self.set_user_read_version(user_id, key, data.version)
        return data


# 3. 因果一致性（Causal Consistency）
# 有因果关系的操作按顺序执行

class CausalConsistencyService:
    def post_comment(self, post_id: str, comment: str, reply_to: str = None):
        # 确保被回复的评论已同步
        if reply_to:
            self.wait_for_sync(reply_to)

        # 发布评论
        comment_id = self.comment_repo.create(post_id, comment, reply_to)

        # 返回因果依赖信息
        return CausalContext(comment_id, dependencies=[reply_to] if reply_to else [])
```

---

## 三、一致性模型选择

### 场景分析

```
┌─────────────────────────────────────────────────────────────┐
│                    一致性选择指南                            │
├─────────────────────┬───────────────────────────────────────┤
│ 场景                │ 建议                                   │
├─────────────────────┼───────────────────────────────────────┤
│ 金融交易            │ 强一致性（ACID）                       │
│ 库存扣减            │ 强一致性 或 最终一致 + 补偿             │
│ 用户余额            │ 强一致性                               │
├─────────────────────┼───────────────────────────────────────┤
│ 社交动态            │ 最终一致性                             │
│ 商品浏览数          │ 最终一致性                             │
│ 日志收集            │ 最终一致性                             │
├─────────────────────┼───────────────────────────────────────┤
│ 购物车              │ 最终一致性（读己之写）                  │
│ 用户资料            │ 最终一致性（读己之写）                  │
│ 评论系统            │ 因果一致性                             │
└─────────────────────┴───────────────────────────────────────┘
```

### 混合策略实战

```python
class OrderService:
    """
    订单服务：根据不同操作选择不同一致性
    """

    def create_order(self, order: Order) -> str:
        """创建订单：强一致性"""
        with self.db.transaction():
            # 1. 扣减库存（强一致）
            self.inventory_service.deduct(order.items)

            # 2. 创建订单
            order_id = self.order_repo.create(order)

            return order_id

    def get_order_list(self, user_id: str) -> List[Order]:
        """订单列表：最终一致性，可读从库"""
        return self.order_repo.find_by_user(user_id, use_slave=True)

    def get_order_detail(self, order_id: str, user_id: str) -> Order:
        """订单详情：读己之写"""
        # 如果刚创建，强制读主库
        if self.is_recently_modified(order_id, user_id):
            return self.order_repo.find_by_id(order_id, use_master=True)

        return self.order_repo.find_by_id(order_id, use_slave=True)

    def update_order_status(self, order_id: str, status: str):
        """更新状态：最终一致 + 事件驱动"""
        # 更新主库
        self.order_repo.update_status(order_id, status)

        # 发布事件（异步同步到其他系统）
        self.event_bus.publish(OrderStatusChangedEvent(order_id, status))

        # 标记为最近修改
        self.mark_recently_modified(order_id)
```

---

## 四、避坑指南

### 坑1：误解 CAP

```
❌ 错误理解：CAP 说的是三选二

✅ 正确理解：
1. P（分区容错）在分布式系统中是必须的
2. 实际选择是 CP 还是 AP
3. 在没有网络分区时，可以同时满足 C 和 A
4. CAP 是针对单个数据的，不是整个系统
```

### 坑2：一致性过度设计

```
❌ 所有操作都用强一致性
   → 性能差、可用性低

✅ 根据业务场景选择：
   • 核心数据：强一致
   • 辅助数据：最终一致
   • 统计数据：弱一致
```

### 坑3：最终一致性的时间窗口

```python
# ❌ 忽视一致性窗口
def transfer(from_account, to_account, amount):
    deduct(from_account, amount)  # 异步
    add(to_account, amount)       # 异步
    # 中间可能出现总金额不一致！

# ✅ 控制一致性窗口
def transfer(from_account, to_account, amount):
    # 使用事务消息保证最终一致
    tx_id = start_transaction()
    try:
        deduct(from_account, amount, tx_id)
        add(to_account, amount, tx_id)
        commit_transaction(tx_id)
    except Exception:
        rollback_transaction(tx_id)
```

### 坑4：脑裂问题

```
网络分区导致多个 Master：

┌─────────────┐         ┌─────────────┐
│  Master A   │   ╳╳╳   │  Master B   │
│  (原 Master) │  网络断  │ (被选为新M) │
└─────────────┘         └─────────────┘
     写入 X=1                写入 X=2
            ↘             ↙
              网络恢复
                 ↓
              冲突！

解决方案：
1. Quorum 机制：多数派写入才算成功
2. Fencing Token：防止旧 Master 写入
3. Lease 机制：Master 需要续约
```
