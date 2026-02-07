# DDD实战：订单系统完整案例

## 1. 项目背景

### 1.1 业务场景

构建一个中型电商平台的订单系统，包含以下核心功能：

```
核心业务流程：
┌─────────────────────────────────────────────────┐
│  1. 用户浏览商品，加入购物车                     │
│  2. 创建订单（检查库存）                         │
│  3. 支付订单（对接支付网关）                     │
│  4. 扣减库存、生成物流单                         │
│  5. 发货、物流跟踪                               │
│  6. 确认收货、订单完成                           │
│  7. 售后：退款、退货                             │
└─────────────────────────────────────────────────┘

非功能需求：
- QPS: 5000+ (促销期间)
- 并发订单: 10000+
- 库存一致性要求高
- 订单数据不可丢失
```

### 1.2 技术栈选择

```
后端框架：
┌─────────────────────────────────┐
│  Spring Boot 3.2               │
│  Spring Data JPA               │
│  Spring Cloud                  │
└─────────────────────────────────┘

数据库：
┌─────────────────────────────────┐
│  MySQL 8.0 (订单主库)          │
│  Redis (缓存 + 分布式锁)       │
│  MongoDB (订单归档)            │
└─────────────────────────────────┘

消息队列：
┌─────────────────────────────────┐
│  RabbitMQ (领域事件)           │
└─────────────────────────────────┘

监控：
┌─────────────────────────────────┐
│  Prometheus + Grafana          │
│  ELK Stack                     │
└─────────────────────────────────┘
```

## 2. 领域建模

### 2.1 限界上下文划分

```
上下文映射图：
┌──────────────────────────────────────────────┐
│                                               │
│  ┌─────────────┐          ┌─────────────┐   │
│  │  商品上下文  │─────OHS─►│  订单上下文  │   │
│  │  Product BC │          │  Order BC   │   │
│  └─────────────┘          └──────┬──────┘   │
│                                   │           │
│                              Customer│        │
│                                   ↓           │
│  ┌─────────────┐          ┌─────────────┐   │
│  │  库存上下文  │◄────ACL──│  支付上下文  │   │
│  │Inventory BC │          │ Payment BC  │   │
│  └─────────────┘          └─────────────┘   │
│         │                                     │
│    Supplier│                                  │
│         ↓                                     │
│  ┌─────────────┐                             │
│  │  物流上下文  │                             │
│  │Logistics BC │                             │
│  └─────────────┘                             │
└──────────────────────────────────────────────┘
```

### 2.2 订单上下文领域模型

```
订单聚合：
┌────────────────────────────────────────────┐
│          Order (订单聚合根)                 │
├────────────────────────────────────────────┤
│  - OrderId id                              │
│  - CustomerId customerId                   │
│  - List<OrderItem> items                   │
│  - OrderStatus status                      │
│  - Money totalAmount                       │
│  - Address shippingAddress                 │
│  - PaymentInfo paymentInfo                 │
├────────────────────────────────────────────┤
│  + create()                                │
│  + pay()                                   │
│  + ship()                                  │
│  + complete()                              │
│  + cancel()                                │
│  + refund()                                │
└────────────────────────────────────────────┘
         │ contains
         ↓
┌────────────────────────────────────────────┐
│        OrderItem (订单项实体)               │
├────────────────────────────────────────────┤
│  - OrderItemId id                          │
│  - ProductId productId                     │
│  - String productSnapshot                  │
│  - Money unitPrice                         │
│  - int quantity                            │
├────────────────────────────────────────────┤
│  + changeQuantity()                        │
│  + getSubtotal()                           │
└────────────────────────────────────────────┘
```

## 3. 项目结构

### 3.1 模块划分

```
order-service/
├── order-domain/              # 领域层
│   ├── model/
│   │   ├── aggregate/         # 聚合
│   │   │   ├── Order.java
│   │   │   └── OrderItem.java
│   │   ├── entity/            # 实体
│   │   ├── valueobject/       # 值对象
│   │   │   ├── OrderId.java
│   │   │   ├── Money.java
│   │   │   ├── Address.java
│   │   │   └── OrderStatus.java
│   │   └── event/             # 领域事件
│   │       ├── OrderCreatedEvent.java
│   │       ├── OrderPaidEvent.java
│   │       └── OrderShippedEvent.java
│   ├── repository/            # 仓储接口
│   │   └── OrderRepository.java
│   └── service/               # 领域服务
│       └── OrderPricingService.java
│
├── order-application/         # 应用层
│   ├── service/
│   │   └── OrderApplicationService.java
│   ├── command/               # 命令
│   │   ├── CreateOrderCommand.java
│   │   └── PayOrderCommand.java
│   └── dto/                   # DTO
│       └── OrderDTO.java
│
├── order-infrastructure/      # 基础设施层
│   ├── persistence/
│   │   ├── OrderRepositoryImpl.java
│   │   └── entity/
│   │       └── OrderPO.java   # 持久化对象
│   ├── messaging/
│   │   └── RabbitMQEventPublisher.java
│   └── acl/                   # 防腐层
│       ├── ProductServiceAdapter.java
│       └── InventoryServiceAdapter.java
│
├── order-interfaces/          # 接口层
│   ├── rest/
│   │   └── OrderController.java
│   ├── dto/
│   │   └── CreateOrderRequest.java
│   └── facade/
│       └── OrderFacade.java
│
└── order-starter/             # 启动模块
    └── OrderServiceApplication.java
```

### 3.2 依赖关系

```
依赖方向（单向依赖）：

order-interfaces
    ↓
order-application
    ↓
order-domain  ←  order-infrastructure
    ↑
(领域层不依赖任何其他层)
```

## 4. 核心代码实现

### 4.1 领域层 - Order聚合

```java
// order-domain/model/aggregate/Order.java
package com.ecommerce.order.domain.model.aggregate;

import com.ecommerce.order.domain.model.entity.OrderItem;
import com.ecommerce.order.domain.model.event.*;
import com.ecommerce.order.domain.model.valueobject.*;
import lombok.Getter;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Getter
public class Order extends AggregateRoot<OrderId> {

    private OrderId id;
    private CustomerId customerId;
    private List<OrderItem> items;
    private OrderStatus status;
    private Money totalAmount;
    private Address shippingAddress;
    private PaymentInfo paymentInfo;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    // 构造函数私有化
    private Order() {
        super();
        this.items = new ArrayList<>();
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
            throw new OrderDomainException("订单必须包含至少一个商品");
        }

        Order order = new Order();
        order.id = OrderId.generate();
        order.customerId = customerId;
        order.items = new ArrayList<>(items);
        order.shippingAddress = shippingAddress;
        order.status = OrderStatus.PENDING;
        order.createdAt = LocalDateTime.now();
        order.updatedAt = order.createdAt;

        // 计算总价
        order.calculateTotalAmount();

        // 应用不变性规则
        order.validate();

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
     * 支付订单
     */
    public void pay(PaymentInfo paymentInfo) {
        // 验证状态
        if (this.status != OrderStatus.PENDING) {
            throw new OrderDomainException(
                "只能支付待支付订单，当前状态：" + this.status
            );
        }

        // 验证金额
        if (!paymentInfo.getAmount().equals(this.totalAmount)) {
            throw new OrderDomainException(
                "支付金额与订单金额不一致"
            );
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
            throw new OrderDomainException("只能发货已支付订单");
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
     * 确认收货
     */
    public void complete() {
        if (this.status != OrderStatus.SHIPPED) {
            throw new OrderDomainException("只能完成已发货订单");
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
        // 业务规则：已完成订单不能取消
        if (this.status == OrderStatus.COMPLETED) {
            throw new OrderDomainException("已完成订单不能取消");
        }

        // 业务规则：已发货订单不能直接取消
        if (this.status == OrderStatus.SHIPPED) {
            throw new OrderDomainException(
                "已发货订单不能直接取消，请使用退货流程"
            );
        }

        OrderStatus previousStatus = this.status;
        this.status = OrderStatus.CANCELLED;
        this.updatedAt = LocalDateTime.now();

        this.addDomainEvent(new OrderCancelledEvent(
            this.id,
            previousStatus,
            reason,
            LocalDateTime.now()
        ));
    }

    /**
     * 申请退款
     */
    public RefundRequest requestRefund(String reason, Money amount) {
        // 只有已支付或已发货的订单可以退款
        if (this.status != OrderStatus.PAID &&
            this.status != OrderStatus.SHIPPED &&
            this.status != OrderStatus.COMPLETED) {
            throw new OrderDomainException("订单状态不允许退款");
        }

        // 退款金额不能超过订单金额
        if (amount.greaterThan(this.totalAmount)) {
            throw new OrderDomainException("退款金额不能超过订单金额");
        }

        RefundRequest refundRequest = RefundRequest.create(
            this.id,
            reason,
            amount
        );

        this.addDomainEvent(new RefundRequestedEvent(
            this.id,
            refundRequest.getId(),
            amount,
            LocalDateTime.now()
        ));

        return refundRequest;
    }

    /**
     * 添加订单项（待支付状态）
     */
    public void addItem(OrderItem item) {
        if (this.status != OrderStatus.PENDING) {
            throw new OrderDomainException("只能向待支付订单添加商品");
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
            throw new OrderDomainException("只能从待支付订单移除商品");
        }

        this.items.removeIf(item ->
            item.getProductId().equals(productId)
        );
        this.calculateTotalAmount();
        this.updatedAt = LocalDateTime.now();
    }

    /**
     * 计算总价（私有方法，保证内部一致性）
     */
    private void calculateTotalAmount() {
        this.totalAmount = this.items.stream()
            .map(OrderItem::getSubtotal)
            .reduce(Money.ZERO, Money::add);
    }

    /**
     * 验证不变性规则
     */
    private void validate() {
        if (this.items.isEmpty()) {
            throw new OrderDomainException("订单必须包含商品");
        }

        if (this.totalAmount == null ||
            this.totalAmount.lessThanOrEqual(Money.ZERO)) {
            throw new OrderDomainException("订单金额必须大于0");
        }

        if (this.shippingAddress == null) {
            throw new OrderDomainException("订单必须有收货地址");
        }
    }

    /**
     * 防御性复制
     */
    public List<OrderItem> getItems() {
        return new ArrayList<>(items);
    }
}

/**
 * 聚合根基类
 */
public abstract class AggregateRoot<ID> {
    private List<DomainEvent> domainEvents = new ArrayList<>();

    protected void addDomainEvent(DomainEvent event) {
        this.domainEvents.add(event);
    }

    public List<DomainEvent> getDomainEvents() {
        return new ArrayList<>(domainEvents);
    }

    public void clearDomainEvents() {
        this.domainEvents.clear();
    }
}

/**
 * 订单异常
 */
public class OrderDomainException extends RuntimeException {
    public OrderDomainException(String message) {
        super(message);
    }
}
```

### 4.2 值对象实现

```java
// order-domain/model/valueobject/Money.java
package com.ecommerce.order.domain.model.valueobject;

import lombok.EqualsAndHashCode;
import lombok.Getter;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * 金额值对象 - 不可变
 */
@Getter
@EqualsAndHashCode
public final class Money {
    public static final Money ZERO = new Money(BigDecimal.ZERO);

    private final BigDecimal amount;
    private final String currency;

    private Money(BigDecimal amount) {
        this(amount, "CNY");
    }

    private Money(BigDecimal amount, String currency) {
        if (amount == null) {
            throw new IllegalArgumentException("金额不能为null");
        }
        if (amount.compareTo(BigDecimal.ZERO) < 0) {
            throw new IllegalArgumentException("金额不能为负数");
        }

        this.amount = amount.setScale(2, RoundingMode.HALF_UP);
        this.currency = currency;
    }

    public static Money of(BigDecimal amount) {
        return new Money(amount);
    }

    public static Money yuan(double yuan) {
        return new Money(BigDecimal.valueOf(yuan));
    }

    public static Money cents(long cents) {
        return new Money(
            BigDecimal.valueOf(cents).divide(
                BigDecimal.valueOf(100),
                2,
                RoundingMode.HALF_UP
            )
        );
    }

    /**
     * 加法 - 返回新对象
     */
    public Money add(Money other) {
        checkCurrency(other);
        return new Money(this.amount.add(other.amount), this.currency);
    }

    /**
     * 减法
     */
    public Money subtract(Money other) {
        checkCurrency(other);
        BigDecimal result = this.amount.subtract(other.amount);
        if (result.compareTo(BigDecimal.ZERO) < 0) {
            throw new IllegalArgumentException("结果金额不能为负数");
        }
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
        checkCurrency(other);
        return this.amount.compareTo(other.amount) > 0;
    }

    public boolean lessThan(Money other) {
        checkCurrency(other);
        return this.amount.compareTo(other.amount) < 0;
    }

    public boolean lessThanOrEqual(Money other) {
        checkCurrency(other);
        return this.amount.compareTo(other.amount) <= 0;
    }

    /**
     * 转换为分
     */
    public long toCents() {
        return this.amount
            .multiply(BigDecimal.valueOf(100))
            .longValue();
    }

    private void checkCurrency(Money other) {
        if (!this.currency.equals(other.currency)) {
            throw new IllegalArgumentException(
                "货币类型不匹配: " + this.currency + " vs " + other.currency
            );
        }
    }

    @Override
    public String toString() {
        return "¥" + amount.toString();
    }
}

// order-domain/model/valueobject/Address.java
package com.ecommerce.order.domain.model.valueobject;

import lombok.EqualsAndHashCode;
import lombok.Getter;

/**
 * 地址值对象 - 不可变
 */
@Getter
@EqualsAndHashCode
public final class Address {
    private final String province;
    private final String city;
    private final String district;
    private final String street;
    private final String zipCode;
    private final String receiverName;
    private final String receiverPhone;

    private Address(Builder builder) {
        this.province = builder.province;
        this.city = builder.city;
        this.district = builder.district;
        this.street = builder.street;
        this.zipCode = builder.zipCode;
        this.receiverName = builder.receiverName;
        this.receiverPhone = builder.receiverPhone;

        validate();
    }

    private void validate() {
        if (province == null || province.isBlank()) {
            throw new IllegalArgumentException("省份不能为空");
        }
        if (city == null || city.isBlank()) {
            throw new IllegalArgumentException("城市不能为空");
        }
        if (street == null || street.isBlank()) {
            throw new IllegalArgumentException("街道地址不能为空");
        }
        if (receiverName == null || receiverName.isBlank()) {
            throw new IllegalArgumentException("收货人姓名不能为空");
        }
        if (receiverPhone == null || receiverPhone.isBlank()) {
            throw new IllegalArgumentException("收货人电话不能为空");
        }
    }

    public String getFullAddress() {
        return province + city +
               (district != null ? district : "") +
               street;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String province;
        private String city;
        private String district;
        private String street;
        private String zipCode;
        private String receiverName;
        private String receiverPhone;

        public Builder province(String province) {
            this.province = province;
            return this;
        }

        public Builder city(String city) {
            this.city = city;
            return this;
        }

        public Builder district(String district) {
            this.district = district;
            return this;
        }

        public Builder street(String street) {
            this.street = street;
            return this;
        }

        public Builder zipCode(String zipCode) {
            this.zipCode = zipCode;
            return this;
        }

        public Builder receiverName(String receiverName) {
            this.receiverName = receiverName;
            return this;
        }

        public Builder receiverPhone(String receiverPhone) {
            this.receiverPhone = receiverPhone;
            return this;
        }

        public Address build() {
            return new Address(this);
        }
    }

    @Override
    public String toString() {
        return String.format("%s %s (%s)",
            getFullAddress(),
            receiverName,
            receiverPhone
        );
    }
}
```

### 4.3 应用层服务

```java
// order-application/service/OrderApplicationService.java
package com.ecommerce.order.application.service;

import com.ecommerce.order.application.command.*;
import com.ecommerce.order.application.dto.OrderDTO;
import com.ecommerce.order.domain.model.aggregate.Order;
import com.ecommerce.order.domain.model.entity.OrderItem;
import com.ecommerce.order.domain.model.valueobject.*;
import com.ecommerce.order.domain.repository.OrderRepository;
import com.ecommerce.order.infrastructure.acl.*;
import com.ecommerce.order.infrastructure.messaging.DomainEventPublisher;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

/**
 * 订单应用服务
 * 编排领域对象，不包含业务逻辑
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class OrderApplicationService {

    private final OrderRepository orderRepository;
    private final ProductServiceAdapter productServiceAdapter;
    private final InventoryServiceAdapter inventoryServiceAdapter;
    private final DomainEventPublisher eventPublisher;

    /**
     * 创建订单
     */
    @Transactional
    public OrderId createOrder(CreateOrderCommand command) {
        log.info("创建订单: customerId={}", command.getCustomerId());

        // 1. 获取商品信息（通过防腐层）
        List<OrderItem> orderItems = command.getItems().stream()
            .map(itemCmd -> {
                // 调用商品服务获取商品信息
                ProductInfo product = productServiceAdapter
                    .getProduct(itemCmd.getProductId());

                return OrderItem.create(
                    itemCmd.getProductId(),
                    product.getName(),
                    product.getPrice(),
                    itemCmd.getQuantity()
                );
            })
            .collect(Collectors.toList());

        // 2. 检查库存（通过防腐层）
        boolean stockAvailable = inventoryServiceAdapter
            .checkStock(orderItems);

        if (!stockAvailable) {
            throw new OrderApplicationException("库存不足");
        }

        // 3. 创建订单聚合（领域逻辑）
        Order order = Order.create(
            command.getCustomerId(),
            orderItems,
            command.getShippingAddress()
        );

        // 4. 预占库存
        inventoryServiceAdapter.reserveStock(order.getId(), orderItems);

        // 5. 保存订单
        orderRepository.save(order);

        // 6. 发布领域事件
        eventPublisher.publishAll(order.getDomainEvents());
        order.clearDomainEvents();

        log.info("订单创建成功: orderId={}", order.getId());
        return order.getId();
    }

    /**
     * 支付订单
     */
    @Transactional
    public void payOrder(PayOrderCommand command) {
        log.info("支付订单: orderId={}", command.getOrderId());

        // 1. 加载订单聚合
        Order order = orderRepository.findById(command.getOrderId())
            .orElseThrow(() ->
                new OrderApplicationException("订单不存在")
            );

        // 2. 执行支付（领域逻辑）
        order.pay(command.getPaymentInfo());

        // 3. 保存订单
        orderRepository.save(order);

        // 4. 发布领域事件
        eventPublisher.publishAll(order.getDomainEvents());
        order.clearDomainEvents();

        log.info("订单支付成功: orderId={}", order.getId());
    }

    /**
     * 发货
     */
    @Transactional
    public void shipOrder(ShipOrderCommand command) {
        log.info("订单发货: orderId={}", command.getOrderId());

        // 1. 加载订单
        Order order = orderRepository.findById(command.getOrderId())
            .orElseThrow(() ->
                new OrderApplicationException("订单不存在")
            );

        // 2. 执行发货（领域逻辑）
        order.ship(command.getShipmentInfo());

        // 3. 保存订单
        orderRepository.save(order);

        // 4. 发布领域事件
        eventPublisher.publishAll(order.getDomainEvents());
        order.clearDomainEvents();

        log.info("订单发货成功: orderId={}", order.getId());
    }

    /**
     * 查询订单（读操作）
     */
    @Transactional(readOnly = true)
    public OrderDTO getOrder(OrderId orderId) {
        Order order = orderRepository.findById(orderId)
            .orElseThrow(() ->
                new OrderApplicationException("订单不存在")
            );

        return OrderDTO.fromDomain(order);
    }

    /**
     * 查询用户订单列表
     */
    @Transactional(readOnly = true)
    public List<OrderDTO> getCustomerOrders(CustomerId customerId) {
        List<Order> orders = orderRepository
            .findByCustomerId(customerId);

        return orders.stream()
            .map(OrderDTO::fromDomain)
            .collect(Collectors.toList());
    }
}
```

### 4.4 防腐层实现

```java
// order-infrastructure/acl/InventoryServiceAdapter.java
package com.ecommerce.order.infrastructure.acl;

import com.ecommerce.inventory.api.InventoryServiceClient;
import com.ecommerce.inventory.api.dto.ReserveStockRequest;
import com.ecommerce.order.domain.model.entity.OrderItem;
import com.ecommerce.order.domain.model.valueobject.OrderId;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.stream.Collectors;

/**
 * 库存服务适配器 - 防腐层
 * 隔离订单上下文和库存上下文
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class InventoryServiceAdapter {

    private final InventoryServiceClient inventoryServiceClient;

    /**
     * 检查库存
     */
    public boolean checkStock(List<OrderItem> items) {
        try {
            List<String> skuIds = items.stream()
                .map(item -> item.getProductId().getValue())
                .collect(Collectors.toList());

            // 调用库存服务
            return inventoryServiceClient.checkStock(skuIds);

        } catch (Exception e) {
            log.error("检查库存失败", e);
            throw new InventoryServiceException("库存服务不可用", e);
        }
    }

    /**
     * 预占库存
     */
    public void reserveStock(OrderId orderId, List<OrderItem> items) {
        try {
            ReserveStockRequest request = new ReserveStockRequest();
            request.setOrderId(orderId.getValue());
            request.setItems(items.stream()
                .map(item -> {
                    ReserveStockRequest.Item reqItem =
                        new ReserveStockRequest.Item();
                    reqItem.setSkuId(item.getProductId().getValue());
                    reqItem.setQuantity(item.getQuantity());
                    return reqItem;
                })
                .collect(Collectors.toList())
            );

            inventoryServiceClient.reserveStock(request);

        } catch (Exception e) {
            log.error("预占库存失败", e);
            throw new InventoryServiceException("预占库存失败", e);
        }
    }

    /**
     * 扣减库存
     */
    public void deductStock(OrderId orderId) {
        try {
            inventoryServiceClient.deductStock(orderId.getValue());
        } catch (Exception e) {
            log.error("扣减库存失败", e);
            throw new InventoryServiceException("扣减库存失败", e);
        }
    }

    /**
     * 释放库存
     */
    public void releaseStock(OrderId orderId) {
        try {
            inventoryServiceClient.releaseStock(orderId.getValue());
        } catch (Exception e) {
            log.error("释放库存失败", e);
            // 释放库存失败不抛异常，记录日志，后续补偿
            log.warn("库存释放失败，需要人工介入: orderId={}",
                orderId.getValue());
        }
    }
}
```

### 4.5 事件处理

```java
// order-application/eventhandler/OrderEventHandler.java
package com.ecommerce.order.application.eventhandler;

import com.ecommerce.order.domain.model.event.*;
import com.ecommerce.order.infrastructure.acl.InventoryServiceAdapter;
import com.ecommerce.order.infrastructure.acl.NotificationServiceAdapter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;
import org.springframework.transaction.event.TransactionPhase;
import org.springframework.transaction.event.TransactionalEventListener;

/**
 * 订单事件处理器
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class OrderEventHandler {

    private final InventoryServiceAdapter inventoryServiceAdapter;
    private final NotificationServiceAdapter notificationServiceAdapter;

    /**
     * 处理订单已支付事件
     * 事务提交后异步处理
     */
    @Async
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    public void handleOrderPaid(OrderPaidEvent event) {
        log.info("处理订单已支付事件: orderId={}", event.getOrderId());

        try {
            // 扣减库存
            inventoryServiceAdapter.deductStock(event.getOrderId());

            // 发送支付成功通知
            notificationServiceAdapter.sendOrderPaidNotification(
                event.getOrderId()
            );

        } catch (Exception e) {
            log.error("处理订单已支付事件失败", e);
            // 发送告警，人工介入
        }
    }

    /**
     * 处理订单已取消事件
     */
    @Async
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    public void handleOrderCancelled(OrderCancelledEvent event) {
        log.info("处理订单已取消事件: orderId={}", event.getOrderId());

        try {
            // 释放预占库存
            if (event.getPreviousStatus() == OrderStatus.PENDING) {
                inventoryServiceAdapter.releaseStock(event.getOrderId());
            }

            // 发送取消通知
            notificationServiceAdapter.sendOrderCancelledNotification(
                event.getOrderId(),
                event.getReason()
            );

        } catch (Exception e) {
            log.error("处理订单已取消事件失败", e);
        }
    }
}
```

## 5. 性能优化

### 5.1 缓存策略

```java
/**
 * 订单缓存
 */
@Service
@RequiredArgsConstructor
public class OrderCacheService {

    private final RedisTemplate<String, OrderDTO> redisTemplate;
    private final OrderRepository orderRepository;

    private static final String ORDER_CACHE_PREFIX = "order:";
    private static final Duration CACHE_TTL = Duration.ofMinutes(30);

    /**
     * 获取订单（先查缓存）
     */
    public OrderDTO getOrder(OrderId orderId) {
        String cacheKey = ORDER_CACHE_PREFIX + orderId.getValue();

        // 1. 查询缓存
        OrderDTO cached = redisTemplate.opsForValue().get(cacheKey);
        if (cached != null) {
            return cached;
        }

        // 2. 查询数据库
        Order order = orderRepository.findById(orderId)
            .orElseThrow(() -> new OrderNotFoundException());

        OrderDTO dto = OrderDTO.fromDomain(order);

        // 3. 写入缓存
        redisTemplate.opsForValue().set(cacheKey, dto, CACHE_TTL);

        return dto;
    }

    /**
     * 删除缓存
     */
    public void evictOrder(OrderId orderId) {
        String cacheKey = ORDER_CACHE_PREFIX + orderId.getValue();
        redisTemplate.delete(cacheKey);
    }
}
```

### 5.2 分库分表

```yaml
# ShardingSphere配置
spring:
  shardingsphere:
    datasource:
      names: ds0,ds1
      ds0:
        type: com.zaxxer.hikari.HikariDataSource
        driver-class-name: com.mysql.cj.jdbc.Driver
        jdbc-url: jdbc:mysql://localhost:3306/order_0
      ds1:
        type: com.zaxxer.hikari.HikariDataSource
        driver-class-name: com.mysql.cj.jdbc.Driver
        jdbc-url: jdbc:mysql://localhost:3306/order_1

    rules:
      sharding:
        tables:
          t_order:
            actual-data-nodes: ds$->{0..1}.t_order_$->{0..3}
            database-strategy:
              standard:
                sharding-column: customer_id
                sharding-algorithm-name: customer-db-inline
            table-strategy:
              standard:
                sharding-column: order_id
                sharding-algorithm-name: order-table-inline

        sharding-algorithms:
          customer-db-inline:
            type: INLINE
            props:
              algorithm-expression: ds$->{customer_id % 2}
          order-table-inline:
            type: INLINE
            props:
              algorithm-expression: t_order_$->{order_id.hashCode() % 4}
```

## 6. 测试策略

### 6.1 单元测试

```java
/**
 * Order聚合单元测试
 */
class OrderTest {

    @Test
    void should_create_order_successfully() {
        // Given
        CustomerId customerId = CustomerId.of("CUST001");
        List<OrderItem> items = List.of(
            createTestOrderItem("SKU001", 2),
            createTestOrderItem("SKU002", 1)
        );
        Address address = createTestAddress();

        // When
        Order order = Order.create(customerId, items, address);

        // Then
        assertThat(order.getId()).isNotNull();
        assertThat(order.getStatus()).isEqualTo(OrderStatus.PENDING);
        assertThat(order.getItems()).hasSize(2);
        assertThat(order.getTotalAmount())
            .isEqualTo(Money.yuan(300)); // 200*2 + 100*1
        assertThat(order.getDomainEvents()).hasSize(1);
        assertThat(order.getDomainEvents().get(0))
            .isInstanceOf(OrderCreatedEvent.class);
    }

    @Test
    void should_throw_exception_when_pay_non_pending_order() {
        // Given
        Order order = createTestOrder();
        order.pay(createTestPaymentInfo());

        // When & Then
        assertThatThrownBy(() ->
            order.pay(createTestPaymentInfo())
        )
        .isInstanceOf(OrderDomainException.class)
        .hasMessageContaining("只能支付待支付订单");
    }
}
```

## 7. 总结

DDD实战的关键点：
1. **业务逻辑在领域层** - 不要泄漏到应用层
2. **聚合保护不变性** - 通过聚合根控制访问
3. **防腐层隔离上下文** - 保护领域模型纯净
4. **领域事件解耦** - 异步处理副作用
5. **单元测试覆盖领域逻辑** - 快速验证业务规则

DDD不是银弹，但对复杂业务系统非常有效。关键是找到核心域，投入资源做好领域建模。
