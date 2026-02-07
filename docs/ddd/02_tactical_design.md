# DDD战术设计：实体、值对象与聚合

## 1. 战术设计概述

### 1.1 战术设计构建块

```
DDD战术设计层次结构：

┌─────────────────────────────────────────────────────┐
│                    应用层                            │
│  ┌──────────────┐  ┌──────────────┐                │
│  │应用服务      │  │事件处理器    │                │
│  └──────────────┘  └──────────────┘                │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│                    领域层                            │
│  ┌──────────────────────────────────────────────┐  │
│  │              聚合 (Aggregate)                 │  │
│  │  ┌────────────────────────────────────────┐  │  │
│  │  │     聚合根 (Aggregate Root)            │  │  │
│  │  │     - 实体 (Entity)                    │  │  │
│  │  │     - 值对象 (Value Object)            │  │  │
│  │  └────────────────────────────────────────┘  │  │
│  │                                               │  │
│  │  ┌──────────────┐     ┌──────────────┐      │  │
│  │  │领域服务      │     │领域事件      │      │  │
│  │  └──────────────┘     └──────────────┘      │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  ┌──────────────┐                                  │
│  │仓储接口      │                                  │
│  └──────────────┘                                  │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│                    基础设施层                        │
│  ┌──────────────┐  ┌──────────────┐                │
│  │仓储实现      │  │消息队列      │                │
│  └──────────────┘  └──────────────┘                │
└─────────────────────────────────────────────────────┘
```

## 2. 实体 (Entity)

### 2.1 实体的特征

实体是具有唯一标识的对象，即使属性相同，只要标识不同就是不同的实体。

**实体vs值对象对比：**

```
实体示例：用户
┌──────────────────┐     ┌──────────────────┐
│  User #1001      │     │  User #1002      │
├──────────────────┤     ├──────────────────┤
│  name: "张三"    │     │  name: "张三"    │
│  email: "a@.com" │     │  email: "a@.com" │
└──────────────────┘     └──────────────────┘
       不相等                  不相等
   (ID不同，是两个不同的用户)

值对象示例：地址
┌──────────────────┐     ┌──────────────────┐
│  Address         │     │  Address         │
├──────────────────┤     ├──────────────────┤
│  city: "北京"    │     │  city: "北京"    │
│  street: "XXX"   │     │  street: "XXX"   │
└──────────────────┘     └──────────────────┘
       相等                    相等
   (属性相同，认为是相同的地址)
```

### 2.2 实体实现示例

#### Java实现

```java
package com.ecommerce.order.domain;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.UUID;

/**
 * 订单实体 - 聚合根
 */
public class Order {
    // 唯一标识
    private final OrderId id;

    // 属性（可变）
    private CustomerId customerId;
    private List<OrderItem> items;
    private OrderStatus status;
    private Money totalAmount;
    private Address shippingAddress;
    private PaymentInfo paymentInfo;

    // 时间戳
    private final LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    // 领域事件
    private List<DomainEvent> domainEvents = new ArrayList<>();

    // 构造函数私有化，使用工厂方法创建
    private Order(OrderId id, CustomerId customerId) {
        this.id = Objects.requireNonNull(id, "订单ID不能为空");
        this.customerId = Objects.requireNonNull(customerId, "客户ID不能为空");
        this.items = new ArrayList<>();
        this.status = OrderStatus.PENDING;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = this.createdAt;
    }

    /**
     * 工厂方法：创建订单
     */
    public static Order create(
            CustomerId customerId,
            List<OrderItem> items,
            Address shippingAddress) {

        // 前置条件验证
        if (items == null || items.isEmpty()) {
            throw new IllegalArgumentException("订单必须包含至少一个商品");
        }

        Order order = new Order(OrderId.generate(), customerId);
        order.items = new ArrayList<>(items);
        order.shippingAddress = shippingAddress;

        // 计算总价
        order.calculateTotalAmount();

        // 发布领域事件
        order.addDomainEvent(new OrderCreatedEvent(
            order.id,
            order.customerId,
            order.totalAmount,
            LocalDateTime.now()
        ));

        return order;
    }

    /**
     * 添加订单项
     */
    public void addItem(OrderItem item) {
        if (this.status != OrderStatus.PENDING) {
            throw new IllegalStateException("只能向待支付订单添加商品");
        }

        this.items.add(item);
        this.calculateTotalAmount();
        this.updatedAt = LocalDateTime.now();
    }

    /**
     * 移除订单项
     */
    public void removeItem(ProductId productId) {
        if (this.status != OrderStatus.PENDING) {
            throw new IllegalStateException("只能从待支付订单移除商品");
        }

        this.items.removeIf(item -> item.getProductId().equals(productId));
        this.calculateTotalAmount();
        this.updatedAt = LocalDateTime.now();
    }

    /**
     * 支付订单
     */
    public void pay(PaymentInfo paymentInfo) {
        if (this.status != OrderStatus.PENDING) {
            throw new IllegalStateException(
                "只能支付待支付订单，当前状态：" + this.status
            );
        }

        if (this.items.isEmpty()) {
            throw new IllegalStateException("订单没有商品，无法支付");
        }

        this.paymentInfo = paymentInfo;
        this.status = OrderStatus.PAID;
        this.updatedAt = LocalDateTime.now();

        // 发布领域事件
        this.addDomainEvent(new OrderPaidEvent(
            this.id,
            this.totalAmount,
            paymentInfo,
            LocalDateTime.now()
        ));
    }

    /**
     * 发货
     */
    public void ship(ShipmentInfo shipmentInfo) {
        if (this.status != OrderStatus.PAID) {
            throw new IllegalStateException("只能发货已支付订单");
        }

        this.status = OrderStatus.SHIPPED;
        this.updatedAt = LocalDateTime.now();

        this.addDomainEvent(new OrderShippedEvent(
            this.id,
            shipmentInfo,
            LocalDateTime.now()
        ));
    }

    /**
     * 完成订单
     */
    public void complete() {
        if (this.status != OrderStatus.SHIPPED) {
            throw new IllegalStateException("只能完成已发货订单");
        }

        this.status = OrderStatus.COMPLETED;
        this.updatedAt = LocalDateTime.now();

        this.addDomainEvent(new OrderCompletedEvent(
            this.id,
            LocalDateTime.now()
        ));
    }

    /**
     * 取消订单
     */
    public void cancel(String reason) {
        if (this.status == OrderStatus.COMPLETED) {
            throw new IllegalStateException("已完成订单不能取消");
        }

        if (this.status == OrderStatus.SHIPPED) {
            throw new IllegalStateException("已发货订单不能直接取消，请使用退货流程");
        }

        this.status = OrderStatus.CANCELLED;
        this.updatedAt = LocalDateTime.now();

        this.addDomainEvent(new OrderCancelledEvent(
            this.id,
            reason,
            LocalDateTime.now()
        ));
    }

    /**
     * 计算总价（私有方法，内部一致性）
     */
    private void calculateTotalAmount() {
        this.totalAmount = this.items.stream()
            .map(OrderItem::getSubtotal)
            .reduce(Money.ZERO, Money::add);
    }

    /**
     * 添加领域事件
     */
    private void addDomainEvent(DomainEvent event) {
        this.domainEvents.add(event);
    }

    // Getters
    public OrderId getId() {
        return id;
    }

    public CustomerId getCustomerId() {
        return customerId;
    }

    public List<OrderItem> getItems() {
        return new ArrayList<>(items); // 防御性复制
    }

    public OrderStatus getStatus() {
        return status;
    }

    public Money getTotalAmount() {
        return totalAmount;
    }

    public List<DomainEvent> getDomainEvents() {
        return new ArrayList<>(domainEvents);
    }

    public void clearDomainEvents() {
        this.domainEvents.clear();
    }

    /**
     * 实体相等性：基于ID
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Order order = (Order) o;
        return Objects.equals(id, order.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}

/**
 * 订单ID - 值对象
 */
public class OrderId {
    private final String value;

    private OrderId(String value) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("订单ID不能为空");
        }
        this.value = value;
    }

    public static OrderId of(String value) {
        return new OrderId(value);
    }

    public static OrderId generate() {
        return new OrderId("ORD-" + UUID.randomUUID().toString());
    }

    public String getValue() {
        return value;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        OrderId orderId = (OrderId) o;
        return Objects.equals(value, orderId.value);
    }

    @Override
    public int hashCode() {
        return Objects.hash(value);
    }

    @Override
    public String toString() {
        return value;
    }
}

/**
 * 订单状态枚举
 */
public enum OrderStatus {
    PENDING("待支付"),
    PAID("已支付"),
    SHIPPED("已发货"),
    COMPLETED("已完成"),
    CANCELLED("已取消");

    private final String description;

    OrderStatus(String description) {
        this.description = description;
    }

    public String getDescription() {
        return description;
    }
}
```

#### Go实现

```go
package order

import (
    "errors"
    "time"
    "github.com/google/uuid"
)

// Order 订单实体
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
    domainEvents    []DomainEvent
}

// OrderID 订单ID值对象
type OrderID struct {
    value string
}

// NewOrderID 创建订单ID
func NewOrderID(value string) (OrderID, error) {
    if value == "" {
        return OrderID{}, errors.New("订单ID不能为空")
    }
    return OrderID{value: value}, nil
}

// GenerateOrderID 生成新的订单ID
func GenerateOrderID() OrderID {
    return OrderID{value: "ORD-" + uuid.New().String()}
}

func (id OrderID) Value() string {
    return id.value
}

func (id OrderID) Equals(other OrderID) bool {
    return id.value == other.value
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

func (s OrderStatus) String() string {
    switch s {
    case OrderStatusPending:
        return "待支付"
    case OrderStatusPaid:
        return "已支付"
    case OrderStatusShipped:
        return "已发货"
    case OrderStatusCompleted:
        return "已完成"
    case OrderStatusCancelled:
        return "已取消"
    default:
        return "未知"
    }
}

// CreateOrder 工厂方法：创建订单
func CreateOrder(
    customerID CustomerID,
    items []OrderItem,
    shippingAddress Address,
) (*Order, error) {
    // 前置条件验证
    if len(items) == 0 {
        return nil, errors.New("订单必须包含至少一个商品")
    }

    now := time.Now()
    order := &Order{
        id:              GenerateOrderID(),
        customerID:      customerID,
        items:           make([]OrderItem, len(items)),
        status:          OrderStatusPending,
        shippingAddress: shippingAddress,
        createdAt:       now,
        updatedAt:       now,
        domainEvents:    make([]DomainEvent, 0),
    }

    copy(order.items, items)

    // 计算总价
    if err := order.calculateTotalAmount(); err != nil {
        return nil, err
    }

    // 发布领域事件
    order.addDomainEvent(OrderCreatedEvent{
        OrderID:    order.id,
        CustomerID: order.customerID,
        Amount:     order.totalAmount,
        OccurredAt: now,
    })

    return order, nil
}

// AddItem 添加订单项
func (o *Order) AddItem(item OrderItem) error {
    if o.status != OrderStatusPending {
        return errors.New("只能向待支付订单添加商品")
    }

    o.items = append(o.items, item)
    if err := o.calculateTotalAmount(); err != nil {
        return err
    }
    o.updatedAt = time.Now()

    return nil
}

// RemoveItem 移除订单项
func (o *Order) RemoveItem(productID ProductID) error {
    if o.status != OrderStatusPending {
        return errors.New("只能从待支付订单移除商品")
    }

    newItems := make([]OrderItem, 0, len(o.items))
    for _, item := range o.items {
        if !item.ProductID.Equals(productID) {
            newItems = append(newItems, item)
        }
    }

    o.items = newItems
    if err := o.calculateTotalAmount(); err != nil {
        return err
    }
    o.updatedAt = time.Now()

    return nil
}

// Pay 支付订单
func (o *Order) Pay(paymentInfo PaymentInfo) error {
    if o.status != OrderStatusPending {
        return errors.New("只能支付待支付订单，当前状态：" + o.status.String())
    }

    if len(o.items) == 0 {
        return errors.New("订单没有商品，无法支付")
    }

    o.paymentInfo = &paymentInfo
    o.status = OrderStatusPaid
    o.updatedAt = time.Now()

    o.addDomainEvent(OrderPaidEvent{
        OrderID:     o.id,
        Amount:      o.totalAmount,
        PaymentInfo: paymentInfo,
        OccurredAt:  o.updatedAt,
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
        OccurredAt:   o.updatedAt,
    })

    return nil
}

// Complete 完成订单
func (o *Order) Complete() error {
    if o.status != OrderStatusShipped {
        return errors.New("只能完成已发货订单")
    }

    o.status = OrderStatusCompleted
    o.updatedAt = time.Now()

    o.addDomainEvent(OrderCompletedEvent{
        OrderID:    o.id,
        OccurredAt: o.updatedAt,
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
        OrderID:    o.id,
        Reason:     reason,
        OccurredAt: o.updatedAt,
    })

    return nil
}

// calculateTotalAmount 计算总价（私有方法）
func (o *Order) calculateTotalAmount() error {
    total := NewMoney(0, "CNY")

    for _, item := range o.items {
        subtotal := item.Subtotal()
        var err error
        total, err = total.Add(subtotal)
        if err != nil {
            return err
        }
    }

    o.totalAmount = total
    return nil
}

// addDomainEvent 添加领域事件
func (o *Order) addDomainEvent(event DomainEvent) {
    o.domainEvents = append(o.domainEvents, event)
}

// Getters
func (o *Order) ID() OrderID {
    return o.id
}

func (o *Order) CustomerID() CustomerID {
    return o.customerID
}

func (o *Order) Items() []OrderItem {
    // 防御性复制
    items := make([]OrderItem, len(o.items))
    copy(items, o.items)
    return items
}

func (o *Order) Status() OrderStatus {
    return o.status
}

func (o *Order) TotalAmount() Money {
    return o.totalAmount
}

func (o *Order) DomainEvents() []DomainEvent {
    events := make([]DomainEvent, len(o.domainEvents))
    copy(events, o.domainEvents)
    return events
}

func (o *Order) ClearDomainEvents() {
    o.domainEvents = make([]DomainEvent, 0)
}

// Equals 实体相等性：基于ID
func (o *Order) Equals(other *Order) bool {
    if other == nil {
        return false
    }
    return o.id.Equals(other.id)
}
```

## 3. 值对象 (Value Object)

### 3.1 值对象的特征

1. **不可变性** - 创建后不能修改
2. **无标识** - 通过属性判断相等性
3. **可替换性** - 可以用另一个相同属性的值对象替换
4. **自验证** - 在构造时验证不变性

### 3.2 值对象实现示例

#### Money值对象（Java）

```java
package com.ecommerce.common.domain;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Currency;
import java.util.Objects;

/**
 * 金额值对象
 * 不可变、自验证
 */
public final class Money {
    public static final Money ZERO = new Money(BigDecimal.ZERO, Currency.getInstance("CNY"));

    private final BigDecimal amount;
    private final Currency currency;

    private Money(BigDecimal amount, Currency currency) {
        this.amount = Objects.requireNonNull(amount, "金额不能为空")
            .setScale(2, RoundingMode.HALF_UP);
        this.currency = Objects.requireNonNull(currency, "货币类型不能为空");

        // 自验证
        if (amount.compareTo(BigDecimal.ZERO) < 0) {
            throw new IllegalArgumentException("金额不能为负数");
        }
    }

    /**
     * 静态工厂方法
     */
    public static Money of(BigDecimal amount, String currencyCode) {
        return new Money(amount, Currency.getInstance(currencyCode));
    }

    public static Money of(long cents, String currencyCode) {
        return new Money(
            BigDecimal.valueOf(cents).divide(BigDecimal.valueOf(100)),
            Currency.getInstance(currencyCode)
        );
    }

    public static Money yuan(BigDecimal yuan) {
        return new Money(yuan, Currency.getInstance("CNY"));
    }

    /**
     * 加法
     */
    public Money add(Money other) {
        if (!this.currency.equals(other.currency)) {
            throw new IllegalArgumentException("货币类型不匹配");
        }
        return new Money(this.amount.add(other.amount), this.currency);
    }

    /**
     * 减法
     */
    public Money subtract(Money other) {
        if (!this.currency.equals(other.currency)) {
            throw new IllegalArgumentException("货币类型不匹配");
        }
        BigDecimal result = this.amount.subtract(other.amount);
        return new Money(result, this.currency);
    }

    /**
     * 乘法
     */
    public Money multiply(int multiplier) {
        return new Money(
            this.amount.multiply(BigDecimal.valueOf(multiplier)),
            this.currency
        );
    }

    public Money multiply(BigDecimal multiplier) {
        return new Money(
            this.amount.multiply(multiplier),
            this.currency
        );
    }

    /**
     * 比较
     */
    public boolean greaterThan(Money other) {
        if (!this.currency.equals(other.currency)) {
            throw new IllegalArgumentException("货币类型不匹配");
        }
        return this.amount.compareTo(other.amount) > 0;
    }

    public boolean lessThan(Money other) {
        if (!this.currency.equals(other.currency)) {
            throw new IllegalArgumentException("货币类型不匹配");
        }
        return this.amount.compareTo(other.amount) < 0;
    }

    /**
     * 转换为分
     */
    public long toCents() {
        return this.amount.multiply(BigDecimal.valueOf(100)).longValue();
    }

    // Getters
    public BigDecimal getAmount() {
        return amount;
    }

    public Currency getCurrency() {
        return currency;
    }

    /**
     * 值对象相等性：基于所有属性
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Money money = (Money) o;
        return amount.compareTo(money.amount) == 0 &&
               Objects.equals(currency, money.currency);
    }

    @Override
    public int hashCode() {
        return Objects.hash(amount, currency);
    }

    @Override
    public String toString() {
        return currency.getSymbol() + amount.toString();
    }
}
```

#### Address值对象（Go）

```go
package valueobject

import (
    "errors"
    "fmt"
    "strings"
)

// Address 地址值对象
type Address struct {
    province string
    city     string
    district string
    street   string
    zipCode  string
}

// NewAddress 创建地址值对象（自验证）
func NewAddress(
    province, city, district, street, zipCode string,
) (Address, error) {
    // 验证
    if strings.TrimSpace(province) == "" {
        return Address{}, errors.New("省份不能为空")
    }
    if strings.TrimSpace(city) == "" {
        return Address{}, errors.New("城市不能为空")
    }
    if strings.TrimSpace(street) == "" {
        return Address{}, errors.New("街道地址不能为空")
    }

    return Address{
        province: strings.TrimSpace(province),
        city:     strings.TrimSpace(city),
        district: strings.TrimSpace(district),
        street:   strings.TrimSpace(street),
        zipCode:  strings.TrimSpace(zipCode),
    }, nil
}

// FullAddress 返回完整地址
func (a Address) FullAddress() string {
    parts := []string{a.province, a.city}
    if a.district != "" {
        parts = append(parts, a.district)
    }
    parts = append(parts, a.street)
    return strings.Join(parts, "")
}

// WithStreet 创建新的地址（不可变性）
func (a Address) WithStreet(newStreet string) (Address, error) {
    if strings.TrimSpace(newStreet) == "" {
        return Address{}, errors.New("街道地址不能为空")
    }

    return Address{
        province: a.province,
        city:     a.city,
        district: a.district,
        street:   strings.TrimSpace(newStreet),
        zipCode:  a.zipCode,
    }, nil
}

// Getters
func (a Address) Province() string {
    return a.province
}

func (a Address) City() string {
    return a.city
}

func (a Address) District() string {
    return a.district
}

func (a Address) Street() string {
    return a.street
}

func (a Address) ZipCode() string {
    return a.zipCode
}

// Equals 值对象相等性：基于所有属性
func (a Address) Equals(other Address) bool {
    return a.province == other.province &&
        a.city == other.city &&
        a.district == other.district &&
        a.street == other.street &&
        a.zipCode == other.zipCode
}

func (a Address) String() string {
    return fmt.Sprintf("%s %s",
        a.FullAddress(),
        a.zipCode,
    )
}
```

## 4. 聚合 (Aggregate)

### 4.1 聚合的概念

聚合是一组相关对象的集合，作为数据修改的单元。

```
聚合边界：
┌───────────────────────────────────────┐
│         订单聚合                       │
│  ┌─────────────────────────────┐     │
│  │   Order (聚合根)             │     │
│  │   - orderId                 │     │
│  │   - customerId              │     │
│  │   - status                  │     │
│  │   - totalAmount             │     │
│  └─────────────────────────────┘     │
│         │                             │
│         │ 包含                        │
│         ↓                             │
│  ┌─────────────────┐                 │
│  │  OrderItem      │ (实体)          │
│  │  - productId    │                 │
│  │  - quantity     │                 │
│  │  - price        │                 │
│  └─────────────────┘                 │
│         │                             │
│         │ 包含                        │
│         ↓                             │
│  ┌─────────────────┐                 │
│  │  Money          │ (值对象)        │
│  └─────────────────┘                 │
└───────────────────────────────────────┘

外部只能通过Order聚合根访问OrderItem
```

### 4.2 聚合设计原则

1. **在边界内保证不变性** - 聚合内部保持一致性
2. **小聚合设计** - 聚合不应过大
3. **通过ID引用其他聚合** - 不直接持有其他聚合对象
4. **一个事务修改一个聚合** - 最终一致性

### 4.3 聚合完整实现

```java
/**
 * 订单项实体（聚合内部实体）
 */
public class OrderItem {
    private OrderItemId id;
    private ProductId productId;  // 通过ID引用商品聚合
    private String productName;   // 商品快照
    private Money unitPrice;
    private int quantity;

    private OrderItem() {
        // JPA需要
    }

    public static OrderItem create(
            ProductId productId,
            String productName,
            Money unitPrice,
            int quantity) {

        if (quantity <= 0) {
            throw new IllegalArgumentException("数量必须大于0");
        }

        OrderItem item = new OrderItem();
        item.id = OrderItemId.generate();
        item.productId = productId;
        item.productName = productName;
        item.unitPrice = unitPrice;
        item.quantity = quantity;

        return item;
    }

    /**
     * 修改数量
     */
    public void changeQuantity(int newQuantity) {
        if (newQuantity <= 0) {
            throw new IllegalArgumentException("数量必须大于0");
        }
        this.quantity = newQuantity;
    }

    /**
     * 计算小计
     */
    public Money getSubtotal() {
        return unitPrice.multiply(quantity);
    }

    // Getters
    public OrderItemId getId() {
        return id;
    }

    public ProductId getProductId() {
        return productId;
    }

    public String getProductName() {
        return productName;
    }

    public Money getUnitPrice() {
        return unitPrice;
    }

    public int getQuantity() {
        return quantity;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        OrderItem orderItem = (OrderItem) o;
        return Objects.equals(id, orderItem.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}

/**
 * 仓储接口
 */
public interface OrderRepository {
    /**
     * 保存聚合
     */
    void save(Order order);

    /**
     * 通过ID查找
     */
    Optional<Order> findById(OrderId orderId);

    /**
     * 查询客户的订单
     */
    List<Order> findByCustomerId(CustomerId customerId);

    /**
     * 删除（物理删除，通常不推荐）
     */
    void delete(Order order);

    /**
     * 生成下一个ID
     */
    OrderId nextId();
}
```

## 5. 领域服务 (Domain Service)

### 5.1 何时使用领域服务

当一个操作：
1. 不自然属于某个实体或值对象
2. 涉及多个聚合
3. 是无状态的

```java
/**
 * 订单定价领域服务
 * 涉及订单、营销、会员等多个聚合
 */
public class OrderPricingService {

    /**
     * 计算订单最终价格
     * 应用优惠券、会员折扣等
     */
    public Money calculateFinalPrice(
            Order order,
            List<Coupon> coupons,
            MemberLevel memberLevel) {

        Money originalPrice = order.getTotalAmount();

        // 应用会员折扣
        Money afterMemberDiscount = applyMemberDiscount(
            originalPrice,
            memberLevel
        );

        // 应用优惠券
        Money afterCouponDiscount = applyCoupons(
            afterMemberDiscount,
            coupons
        );

        return afterCouponDiscount;
    }

    private Money applyMemberDiscount(Money price, MemberLevel level) {
        BigDecimal discount = level.getDiscountRate();
        return price.multiply(discount);
    }

    private Money applyCoupons(Money price, List<Coupon> coupons) {
        Money result = price;

        for (Coupon coupon : coupons) {
            if (coupon.isApplicable(result)) {
                result = coupon.apply(result);
            }
        }

        return result;
    }
}
```

## 6. 领域事件 (Domain Event)

### 6.1 领域事件设计

```java
/**
 * 领域事件基类
 */
public interface DomainEvent {
    LocalDateTime occurredAt();
}

/**
 * 订单已支付事件
 */
public class OrderPaidEvent implements DomainEvent {
    private final OrderId orderId;
    private final Money amount;
    private final PaymentInfo paymentInfo;
    private final LocalDateTime occurredAt;

    public OrderPaidEvent(
            OrderId orderId,
            Money amount,
            PaymentInfo paymentInfo,
            LocalDateTime occurredAt) {
        this.orderId = orderId;
        this.amount = amount;
        this.paymentInfo = paymentInfo;
        this.occurredAt = occurredAt;
    }

    // Getters
    public OrderId getOrderId() {
        return orderId;
    }

    public Money getAmount() {
        return amount;
    }

    public PaymentInfo getPaymentInfo() {
        return paymentInfo;
    }

    @Override
    public LocalDateTime occurredAt() {
        return occurredAt;
    }
}

/**
 * 事件发布器
 */
@Component
public class DomainEventPublisher {

    @Autowired
    private ApplicationEventPublisher eventPublisher;

    public void publish(DomainEvent event) {
        eventPublisher.publishEvent(event);
    }

    public void publishAll(List<DomainEvent> events) {
        events.forEach(this::publish);
    }
}

/**
 * 事件处理器
 */
@Component
public class OrderEventHandler {

    @Autowired
    private InventoryService inventoryService;

    @Autowired
    private NotificationService notificationService;

    /**
     * 处理订单已支付事件
     */
    @EventListener
    @Transactional
    public void handleOrderPaid(OrderPaidEvent event) {
        // 扣减库存
        inventoryService.decreaseStock(event.getOrderId());

        // 发送通知
        notificationService.sendOrderPaidNotification(event.getOrderId());
    }
}
```

## 7. 战术设计最佳实践

### 7.1 聚合大小

```
小聚合原则：
✓ 好的设计
┌──────────────┐
│  Order       │
│  - items[ID] │ ← 只保存ID引用
└──────────────┘

✗ 不好的设计
┌────────────────────────┐
│  Order                 │
│  - items[Item对象]     │ ← 聚合过大
│  - customer[Customer]  │
│  - inventory[Stock]    │
└────────────────────────┘
```

### 7.2 不变性保护

```java
// 防御性编程
public List<OrderItem> getItems() {
    // 返回副本，防止外部修改
    return new ArrayList<>(items);
}

// 不可变值对象
public final class Money {
    private final BigDecimal amount;
    private final Currency currency;

    // 没有setter方法
    // 所有修改操作返回新对象
    public Money add(Money other) {
        return new Money(
            this.amount.add(other.amount),
            this.currency
        );
    }
}
```

### 7.3 常见错误

1. **贫血模型** - 实体只有getter/setter，没有行为
2. **过大的聚合** - 一个聚合包含太多对象
3. **直接引用其他聚合** - 应该通过ID引用
4. **跨聚合事务** - 应该使用最终一致性

## 8. 总结

战术设计的核心价值：
1. **业务逻辑集中在领域层** - 不散落在各个服务层
2. **保护业务不变性** - 通过聚合边界
3. **表达业务规则** - 通过领域模型
4. **可测试性** - 领域对象是POJO，易于单元测试

DDD不是银弹，但对复杂业务逻辑的项目非常有效。
