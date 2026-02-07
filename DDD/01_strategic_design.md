# DDD战略设计：限界上下文与上下文映射

## 1. 战略设计核心概念

### 1.1 什么是战略设计

战略设计是DDD的高层设计方法，关注业务领域的划分和团队组织结构。

```
战略设计层次：
┌─────────────────────────────────────────────────────────┐
│                      业务领域                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  子域1       │  │  子域2       │  │  子域3       │  │
│  │ (核心域)     │  │ (支撑域)     │  │ (通用域)     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
         ↓                  ↓                  ↓
┌─────────────────────────────────────────────────────────┐
│                   限界上下文层                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │订单上下文    │  │库存上下文    │  │支付上下文    │  │
│  │(独立模型)    │  │(独立模型)    │  │(独立模型)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 1.2 子域分类

**核心域 (Core Domain)**
- 业务的核心竞争力
- 投入最多资源
- 需要最优秀的团队

**支撑域 (Supporting Subdomain)**
- 支持核心业务
- 可以定制开发
- 不是竞争优势

**通用域 (Generic Subdomain)**
- 通用解决方案
- 优先采购或使用开源
- 不应投入过多资源

## 2. 限界上下文 (Bounded Context)

### 2.1 限界上下文定义

限界上下文是一个明确的边界，在边界内，所有术语和概念都有明确的含义。

```
电商系统的限界上下文划分：

┌──────────────────────────────────────────────────────────┐
│                    电商业务域                             │
│                                                           │
│  ┌─────────────┐      ┌─────────────┐                   │
│  │  商品上下文  │      │  营销上下文  │                   │
│  │             │      │             │                   │
│  │  - Product  │      │  - Campaign │                   │
│  │  - Category │      │  - Coupon   │                   │
│  │  - SKU      │      │  - Activity │                   │
│  └─────────────┘      └─────────────┘                   │
│                                                           │
│  ┌─────────────┐      ┌─────────────┐                   │
│  │  订单上下文  │◄────►│  库存上下文  │                   │
│  │             │      │             │                   │
│  │  - Order    │      │  - Stock    │                   │
│  │  - OrderItem│      │  - Warehouse│                   │
│  │  - Payment  │      │  - Location │                   │
│  └─────────────┘      └─────────────┘                   │
│         │                                                 │
│         ▼                                                 │
│  ┌─────────────┐      ┌─────────────┐                   │
│  │  物流上下文  │      │  用户上下文  │                   │
│  │             │      │             │                   │
│  │  - Shipment │      │  - Customer │                   │
│  │  - Tracking │      │  - Address  │                   │
│  │  - Carrier  │      │  - Profile  │                   │
│  └─────────────┘      └─────────────┘                   │
└──────────────────────────────────────────────────────────┘
```

### 2.2 同一概念在不同上下文中的含义

**示例：Product在不同上下文中的含义**

```
商品上下文中的Product：
┌──────────────────────┐
│ Product              │
├──────────────────────┤
│ + id                 │
│ + name               │
│ + description        │
│ + specifications     │
│ + images[]           │
│ + categoryId         │
│ + brandId            │
│ + attributes         │
└──────────────────────┘

订单上下文中的Product：
┌──────────────────────┐
│ OrderProduct         │
├──────────────────────┤
│ + productId          │
│ + name               │
│ + price              │
│ + quantity           │
│ + snapshot           │ ← 快照，防止商品信息变更
└──────────────────────┘

库存上下文中的Product：
┌──────────────────────┐
│ StockItem            │
├──────────────────────┤
│ + skuId              │
│ + quantity           │
│ + warehouseId        │
│ + location           │
│ + reservedQuantity   │
└──────────────────────┘
```

## 3. 上下文映射 (Context Mapping)

### 3.1 上下文集成模式

#### 3.1.1 合作关系 (Partnership)

两个上下文团队紧密合作，共同成功或失败。

```
┌──────────────┐          ┌──────────────┐
│  订单上下文   │ ◄──────► │  支付上下文   │
│              │  共同开发  │              │
│  Order API   │          │  Payment API │
└──────────────┘          └──────────────┘
```

#### 3.1.2 共享内核 (Shared Kernel)

两个上下文共享一部分领域模型和代码。

```
┌──────────────────────────────────────┐
│        Shared Kernel                 │
│    ┌──────────────────────┐          │
│    │  CommonTypes         │          │
│    │  - Money             │          │
│    │  - Address           │          │
│    │  - ContactInfo       │          │
│    └──────────────────────┘          │
└──────────────────────────────────────┘
         ↑                ↑
         │                │
┌────────┴────────┐  ┌────┴──────────┐
│  订单上下文      │  │  用户上下文    │
└─────────────────┘  └───────────────┘
```

#### 3.1.3 客户-供应商 (Customer-Supplier)

下游(客户)依赖上游(供应商)，上游需要考虑下游需求。

```
┌──────────────┐          ┌──────────────┐
│  商品上下文   │──供应商──►│  订单上下文   │
│  (Supplier)  │          │  (Customer)  │
│              │          │              │
│  Product API │          │  依赖商品信息 │
└──────────────┘          └──────────────┘
```

#### 3.1.4 遵奉者 (Conformist)

下游完全遵循上游的模型，没有议价权。

```
┌──────────────┐          ┌──────────────┐
│  第三方支付   │──────────►│  订单上下文   │
│  (上游)      │  完全遵循  │  (下游)      │
│              │          │              │
│  微信/支付宝  │          │  适配上游模型 │
└──────────────┘          └──────────────┘
```

#### 3.1.5 防腐层 (Anti-Corruption Layer)

通过适配器隔离外部系统，保护领域模型。

```
┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   外部系统    │───►│   防腐层     │───►│   订单系统    │
│   (遗留系统)  │    │   (ACL)     │    │   (核心域)    │
│              │    │             │    │              │
│  旧模型/协议  │    │  Adapter    │    │  纯净模型     │
│              │    │  Translator │    │              │
└──────────────┘    └─────────────┘    └──────────────┘
```

#### 3.1.6 开放主机服务 (Open Host Service)

提供标准化的API供多个下游使用。

```
                    ┌──────────────┐
          ┌────────►│  订单系统     │
          │         └──────────────┘
          │
┌─────────┴─────┐   ┌──────────────┐
│  商品服务      │──►│  报表系统     │
│  (OHS)        │   └──────────────┘
│  RESTful API  │
│  GraphQL API  │   ┌──────────────┐
└───────────────┘──►│  推荐系统     │
          │         └──────────────┘
          │
          │         ┌──────────────┐
          └────────►│  搜索系统     │
                    └──────────────┘
```

#### 3.1.7 发布语言 (Published Language)

使用文档化的共享语言进行通信。

```
┌─────────────────────────────────────┐
│      Published Language             │
│  ┌───────────────────────────────┐  │
│  │  JSON Schema / Protocol Buffer│  │
│  │  OpenAPI Specification        │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
         ↑                ↑
         │                │
┌────────┴────────┐  ┌────┴──────────┐
│  服务A          │  │  服务B         │
└─────────────────┘  └───────────────┘
```

### 3.2 上下文映射实战案例

#### 电商系统完整上下文映射图

```
┌────────────────────────────────────────────────────────────┐
│                     电商系统上下文映射                        │
│                                                             │
│  ┌──────────────┐                    ┌──────────────┐      │
│  │  用户上下文   │                    │  商品上下文   │      │
│  │  User BC     │                    │  Product BC  │      │
│  └──────┬───────┘                    └──────┬───────┘      │
│         │                                   │              │
│         │ OHS                               │ OHS          │
│         ↓                                   ↓              │
│  ┌──────────────────────────────────────────────────┐      │
│  │              订单上下文 (核心域)                   │      │
│  │              Order BC                            │      │
│  │  ┌────────────────────────────────────────────┐  │      │
│  │  │  ACL: 适配用户、商品、库存、支付服务        │  │      │
│  │  └────────────────────────────────────────────┘  │      │
│  └───────┬─────────────────┬────────────────────────┘      │
│          │                 │                               │
│  Customer│           Customer│                             │
│          ↓                 ↓                               │
│  ┌──────────────┐   ┌──────────────┐                      │
│  │  库存上下文   │   │  支付上下文   │                      │
│  │  Inventory BC│   │  Payment BC  │                      │
│  │  (Supplier)  │   │  Partnership │                      │
│  └──────────────┘   └──────┬───────┘                      │
│                             │                               │
│                             │ Conformist                    │
│                             ↓                               │
│                     ┌──────────────┐                        │
│                     │  第三方支付   │                        │
│                     │  (微信/支付宝)│                        │
│                     └──────────────┘                        │
│                                                             │
│  ┌──────────────┐          ┌──────────────┐                │
│  │  物流上下文   │          │  营销上下文   │                │
│  │  Logistics BC│          │  Marketing BC│                │
│  │  (Supplier)  │          │  (Supporting)│                │
│  └──────────────┘          └──────────────┘                │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

## 4. 战略设计实施步骤

### 4.1 识别子域

```
步骤1: 业务访谈和领域专家研讨
┌────────────────────────────────┐
│  收集业务需求                   │
│  - 业务流程                     │
│  - 核心功能                     │
│  - 业务规则                     │
└────────────────────────────────┘
              ↓
步骤2: 子域分类
┌────────────────────────────────┐
│  核心域：订单管理               │
│  支撑域：库存、物流             │
│  通用域：用户认证、支付         │
└────────────────────────────────┘
              ↓
步骤3: 优先级排序
┌────────────────────────────────┐
│  1. 订单管理 (核心域)           │
│  2. 库存管理 (支撑域)           │
│  3. 用户认证 (通用域-采购)      │
└────────────────────────────────┘
```

### 4.2 划分限界上下文

**划分原则：**

1. **业务能力对齐** - 每个上下文对应一个业务能力
2. **语言一致性** - 上下文内使用统一的术语
3. **高内聚低耦合** - 上下文内部紧密相关，上下文间松耦合
4. **团队对齐** - 一个上下文对应一个团队

**示例：订单上下文的划分**

```java
// 订单上下文的边界
package com.ecommerce.order.domain;

/**
 * 订单聚合根
 * 在订单上下文中，Order是核心概念
 */
public class Order {
    private OrderId orderId;
    private CustomerId customerId;  // 用户上下文的引用
    private List<OrderItem> items;
    private OrderStatus status;
    private Money totalAmount;
    private Address shippingAddress;
    private PaymentInfo paymentInfo;

    /**
     * 创建订单 - 领域行为
     */
    public static Order create(CustomerId customerId,
                               List<OrderItem> items,
                               Address shippingAddress) {
        // 验证订单项
        if (items == null || items.isEmpty()) {
            throw new IllegalArgumentException("订单必须包含商品");
        }

        // 计算总价
        Money total = items.stream()
            .map(OrderItem::getSubtotal)
            .reduce(Money.ZERO, Money::add);

        Order order = new Order();
        order.orderId = OrderId.generate();
        order.customerId = customerId;
        order.items = new ArrayList<>(items);
        order.status = OrderStatus.PENDING;
        order.totalAmount = total;
        order.shippingAddress = shippingAddress;

        // 发布领域事件
        order.addDomainEvent(new OrderCreatedEvent(order.orderId));

        return order;
    }

    /**
     * 支付订单
     */
    public void pay(PaymentInfo paymentInfo) {
        if (this.status != OrderStatus.PENDING) {
            throw new IllegalStateException("只能支付待支付订单");
        }

        this.paymentInfo = paymentInfo;
        this.status = OrderStatus.PAID;

        this.addDomainEvent(new OrderPaidEvent(this.orderId, this.totalAmount));
    }

    /**
     * 发货
     */
    public void ship(ShipmentInfo shipmentInfo) {
        if (this.status != OrderStatus.PAID) {
            throw new IllegalStateException("只能发货已支付订单");
        }

        this.status = OrderStatus.SHIPPED;

        this.addDomainEvent(new OrderShippedEvent(
            this.orderId,
            shipmentInfo
        ));
    }
}

/**
 * 订单项 - 值对象
 */
public class OrderItem {
    private ProductId productId;  // 商品上下文的引用
    private String productName;   // 商品快照
    private Money unitPrice;
    private int quantity;

    public Money getSubtotal() {
        return unitPrice.multiply(quantity);
    }
}

/**
 * 订单状态
 */
public enum OrderStatus {
    PENDING,    // 待支付
    PAID,       // 已支付
    SHIPPED,    // 已发货
    COMPLETED,  // 已完成
    CANCELLED   // 已取消
}
```

### 4.3 定义上下文集成策略

**订单上下文与其他上下文的集成**

```java
// 防腐层：订单上下文访问商品上下文
package com.ecommerce.order.infrastructure.acl;

/**
 * 商品服务适配器 - 防腐层
 */
@Component
public class ProductServiceAdapter {

    @Autowired
    private ProductServiceClient productServiceClient;

    /**
     * 获取商品信息（转换为订单上下文的模型）
     */
    public OrderProduct getProductForOrder(ProductId productId) {
        // 调用商品服务
        ProductDTO productDTO = productServiceClient.getProduct(
            productId.getValue()
        );

        // 转换为订单上下文的模型
        return new OrderProduct(
            productId,
            productDTO.getName(),
            new Money(productDTO.getPrice()),
            productDTO.getStockQuantity()
        );
    }

    /**
     * 批量获取商品信息
     */
    public List<OrderProduct> getProductsForOrder(List<ProductId> productIds) {
        List<String> ids = productIds.stream()
            .map(ProductId::getValue)
            .collect(Collectors.toList());

        List<ProductDTO> products = productServiceClient.batchGetProducts(ids);

        return products.stream()
            .map(p -> new OrderProduct(
                new ProductId(p.getId()),
                p.getName(),
                new Money(p.getPrice()),
                p.getStockQuantity()
            ))
            .collect(Collectors.toList());
    }
}

/**
 * 订单上下文中的商品模型（简化版）
 */
public class OrderProduct {
    private final ProductId productId;
    private final String name;
    private final Money price;
    private final int availableStock;

    // 只包含订单创建所需的信息
    // 不包含商品详情、图片等无关信息
}
```

**开放主机服务：订单服务对外API**

```java
// 订单服务的开放API
package com.ecommerce.order.interfaces.rest;

/**
 * 订单REST API - 开放主机服务
 */
@RestController
@RequestMapping("/api/orders")
public class OrderController {

    @Autowired
    private OrderApplicationService orderApplicationService;

    /**
     * 创建订单
     */
    @PostMapping
    public ResponseEntity<OrderDTO> createOrder(
            @RequestBody CreateOrderRequest request) {

        CreateOrderCommand command = new CreateOrderCommand(
            new CustomerId(request.getCustomerId()),
            request.getItems().stream()
                .map(item -> new OrderItemCommand(
                    new ProductId(item.getProductId()),
                    item.getQuantity()
                ))
                .collect(Collectors.toList()),
            new Address(
                request.getAddress().getProvince(),
                request.getAddress().getCity(),
                request.getAddress().getDetail()
            )
        );

        OrderId orderId = orderApplicationService.createOrder(command);
        OrderDTO orderDTO = orderApplicationService.getOrder(orderId);

        return ResponseEntity.ok(orderDTO);
    }

    /**
     * 查询订单
     */
    @GetMapping("/{orderId}")
    public ResponseEntity<OrderDTO> getOrder(@PathVariable String orderId) {
        OrderDTO order = orderApplicationService.getOrder(
            new OrderId(orderId)
        );
        return ResponseEntity.ok(order);
    }

    /**
     * 支付订单
     */
    @PostMapping("/{orderId}/pay")
    public ResponseEntity<Void> payOrder(
            @PathVariable String orderId,
            @RequestBody PayOrderRequest request) {

        PayOrderCommand command = new PayOrderCommand(
            new OrderId(orderId),
            new PaymentInfo(
                request.getPaymentMethod(),
                request.getTransactionId()
            )
        );

        orderApplicationService.payOrder(command);
        return ResponseEntity.ok().build();
    }
}
```

## 5. 电商系统战略设计完整案例

### 5.1 业务场景

构建一个中型电商平台，包含以下核心功能：
- 商品管理
- 订单管理
- 库存管理
- 支付处理
- 物流配送
- 营销活动

### 5.2 子域划分

```
核心域 (Core Domain):
┌─────────────────────┐
│  订单管理            │
│  - 订单创建          │
│  - 订单支付          │
│  - 订单状态管理      │
│  - 退款处理          │
└─────────────────────┘

支撑域 (Supporting):
┌─────────────────────┐  ┌─────────────────────┐
│  商品管理            │  │  库存管理            │
│  - 商品信息          │  │  - 库存查询          │
│  - 类目管理          │  │  - 库存预占          │
│  - SPU/SKU          │  │  - 库存释放          │
└─────────────────────┘  └─────────────────────┘

┌─────────────────────┐  ┌─────────────────────┐
│  营销管理            │  │  物流管理            │
│  - 优惠券            │  │  - 发货             │
│  - 促销活动          │  │  - 物流跟踪          │
│  - 会员体系          │  │  - 签收             │
└─────────────────────┘  └─────────────────────┘

通用域 (Generic):
┌─────────────────────┐  ┌─────────────────────┐
│  用户认证            │  │  支付处理            │
│  - 登录/注册         │  │  - 第三方支付接入    │
│  - 权限管理          │  │  - 支付回调          │
│  - SSO              │  │  - 退款             │
└─────────────────────┘  └─────────────────────┘
```

### 5.3 限界上下文实现

**Go语言实现示例**

```go
// domain/order/order.go
package order

import (
    "errors"
    "time"
)

// Order 订单聚合根
type Order struct {
    id              OrderID
    customerID      CustomerID
    items           []OrderItem
    status          OrderStatus
    totalAmount     Money
    shippingAddress Address
    paymentInfo     *PaymentInfo
    createdAt       time.Time
    updatedAt       time.Time

    // 领域事件
    domainEvents    []DomainEvent
}

// OrderID 订单ID值对象
type OrderID struct {
    value string
}

func NewOrderID(value string) OrderID {
    return OrderID{value: value}
}

func (id OrderID) String() string {
    return id.value
}

// Money 金额值对象
type Money struct {
    amount   int64  // 分为单位
    currency string
}

func NewMoney(amount int64, currency string) Money {
    return Money{
        amount:   amount,
        currency: currency,
    }
}

func (m Money) Add(other Money) (Money, error) {
    if m.currency != other.currency {
        return Money{}, errors.New("货币类型不匹配")
    }
    return Money{
        amount:   m.amount + other.amount,
        currency: m.currency,
    }, nil
}

func (m Money) Multiply(quantity int) Money {
    return Money{
        amount:   m.amount * int64(quantity),
        currency: m.currency,
    }
}

// OrderStatus 订单状态
type OrderStatus int

const (
    OrderStatusPending OrderStatus = iota
    OrderStatusPaid
    OrderStatusShipped
    OrderStatusCompleted
    OrderStatusCancelled
)

// CreateOrder 创建订单工厂方法
func CreateOrder(
    customerID CustomerID,
    items []OrderItem,
    shippingAddress Address,
) (*Order, error) {
    if len(items) == 0 {
        return nil, errors.New("订单必须包含商品")
    }

    // 计算总价
    totalAmount := Money{amount: 0, currency: "CNY"}
    for _, item := range items {
        subtotal := item.UnitPrice.Multiply(item.Quantity)
        var err error
        totalAmount, err = totalAmount.Add(subtotal)
        if err != nil {
            return nil, err
        }
    }

    order := &Order{
        id:              NewOrderID(generateID()),
        customerID:      customerID,
        items:           items,
        status:          OrderStatusPending,
        totalAmount:     totalAmount,
        shippingAddress: shippingAddress,
        createdAt:       time.Now(),
        updatedAt:       time.Now(),
        domainEvents:    make([]DomainEvent, 0),
    }

    // 发布领域事件
    order.addDomainEvent(OrderCreatedEvent{
        OrderID:   order.id,
        CreatedAt: order.createdAt,
    })

    return order, nil
}

// Pay 支付订单
func (o *Order) Pay(paymentInfo PaymentInfo) error {
    if o.status != OrderStatusPending {
        return errors.New("只能支付待支付订单")
    }

    o.paymentInfo = &paymentInfo
    o.status = OrderStatusPaid
    o.updatedAt = time.Now()

    o.addDomainEvent(OrderPaidEvent{
        OrderID:     o.id,
        TotalAmount: o.totalAmount,
        PaidAt:      o.updatedAt,
    })

    return nil
}

// Ship 发货
func (o *Order) Ship(shipmentInfo ShipmentInfo) error {
    if o.status != OrderStatusPaid {
        return errors.New("只能发货已支付订单")
    }

    o.status = OrderStatusShipped
    o.updatedAt = time.Now()

    o.addDomainEvent(OrderShippedEvent{
        OrderID:      o.id,
        ShipmentInfo: shipmentInfo,
        ShippedAt:    o.updatedAt,
    })

    return nil
}

// Cancel 取消订单
func (o *Order) Cancel(reason string) error {
    if o.status == OrderStatusCompleted {
        return errors.New("已完成订单不能取消")
    }

    if o.status == OrderStatusShipped {
        return errors.New("已发货订单不能直接取消，请使用退货流程")
    }

    o.status = OrderStatusCancelled
    o.updatedAt = time.Now()

    o.addDomainEvent(OrderCancelledEvent{
        OrderID:     o.id,
        Reason:      reason,
        CancelledAt: o.updatedAt,
    })

    return nil
}

func (o *Order) addDomainEvent(event DomainEvent) {
    o.domainEvents = append(o.domainEvents, event)
}

func (o *Order) GetDomainEvents() []DomainEvent {
    return o.domainEvents
}

func (o *Order) ClearDomainEvents() {
    o.domainEvents = make([]DomainEvent, 0)
}

// 辅助函数
func generateID() string {
    // 实际应该使用更复杂的ID生成策略
    return fmt.Sprintf("ORD%d", time.Now().UnixNano())
}
```

### 5.4 上下文集成实现

```go
// infrastructure/acl/product_service_adapter.go
package acl

import (
    "context"
    "github.com/ecommerce/order/domain/order"
)

// ProductServiceAdapter 商品服务适配器（防腐层）
type ProductServiceAdapter struct {
    productClient ProductServiceClient
}

func NewProductServiceAdapter(client ProductServiceClient) *ProductServiceAdapter {
    return &ProductServiceAdapter{
        productClient: client,
    }
}

// GetProductForOrder 获取商品信息用于订单
func (a *ProductServiceAdapter) GetProductForOrder(
    ctx context.Context,
    productID string,
) (*order.OrderProduct, error) {
    // 调用商品服务
    productDTO, err := a.productClient.GetProduct(ctx, productID)
    if err != nil {
        return nil, err
    }

    // 转换为订单上下文的模型
    return &order.OrderProduct{
        ProductID:      order.NewProductID(productDTO.ID),
        Name:           productDTO.Name,
        Price:          order.NewMoney(productDTO.Price, "CNY"),
        AvailableStock: productDTO.StockQuantity,
    }, nil
}

// ProductServiceClient 商品服务客户端接口
type ProductServiceClient interface {
    GetProduct(ctx context.Context, productID string) (*ProductDTO, error)
    BatchGetProducts(ctx context.Context, productIDs []string) ([]*ProductDTO, error)
}

// ProductDTO 商品服务的DTO
type ProductDTO struct {
    ID            string
    Name          string
    Price         int64
    StockQuantity int
    // 其他字段...
}
```

## 6. 战略设计最佳实践

### 6.1 上下文划分原则

1. **按业务能力划分，不按技术层次**
2. **保持上下文的独立演化能力**
3. **避免过度细分，也避免过度集中**
4. **考虑团队结构和沟通成本**

### 6.2 常见反模式

```
反模式1: 共享数据库
┌──────────┐    ┌──────────┐
│ 服务A    │    │ 服务B    │
└────┬─────┘    └────┬─────┘
     │               │
     └───────┬───────┘
           ┌─┴─┐
           │ DB│
           └───┘
问题：紧耦合，无法独立演化

正确做法: 每个上下文独立数据库
┌──────────┐    ┌──────────┐
│ 服务A    │    │ 服务B    │
└────┬─────┘    └────┬─────┘
   ┌─┴─┐          ┌─┴─┐
   │DBA│          │DBB│
   └───┘          └───┘
```

### 6.3 度量指标

- **上下文数量**: 通常5-15个为宜
- **团队对齐度**: 一个上下文一个团队
- **API调用链路长度**: 不超过3跳
- **共享代码比例**: 少于10%

## 7. 总结

战略设计的核心价值：
1. **明确边界** - 清晰的上下文边界减少理解成本
2. **独立演化** - 每个上下文可以独立迭代
3. **团队自治** - 团队在上下文内有完全自主权
4. **技术多样性** - 不同上下文可选择不同技术栈

战略设计是DDD的起点，决定了整个系统的架构格局。投入时间做好战略设计，可以大幅降低后续的维护成本。
