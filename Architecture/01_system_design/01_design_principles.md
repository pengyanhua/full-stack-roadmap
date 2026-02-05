# 设计原则

## 一、SOLID 原则

### 1. 单一职责原则 (SRP)

> 一个类/模块只应该有一个引起它变化的原因

**反例**：
```python
# ❌ 一个类做了太多事情
class UserService:
    def register(self, user):
        # 验证用户
        # 保存到数据库
        # 发送欢迎邮件
        # 记录日志
        # 同步到搜索引擎
        pass
```

**正例**：
```python
# ✅ 职责分离
class UserRepository:
    def save(self, user): pass

class EmailService:
    def send_welcome_email(self, user): pass

class UserSearchIndexer:
    def index(self, user): pass

class UserService:
    def __init__(self, repo, email, indexer):
        self.repo = repo
        self.email = email
        self.indexer = indexer

    def register(self, user):
        self.repo.save(user)
        self.email.send_welcome_email(user)  # 可以异步
        self.indexer.index(user)  # 可以异步
```

**架构层面应用**：
```
┌─────────────────────────────────────────────────────────┐
│                       API Gateway                        │
│         职责：路由、认证、限流（不做业务逻辑）            │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  用户服务      │   │  订单服务      │   │  支付服务      │
│ 只管用户相关   │   │ 只管订单相关   │   │ 只管支付相关   │
└───────────────┘   └───────────────┘   └───────────────┘
```

**坑点**：
- 过度拆分导致服务调用链过长
- 建议：按业务领域而非技术功能拆分

---

### 2. 开闭原则 (OCP)

> 对扩展开放，对修改关闭

**反例**：
```go
// ❌ 每次新增支付方式都要修改这个函数
func Pay(method string, amount float64) error {
    switch method {
    case "alipay":
        return payByAlipay(amount)
    case "wechat":
        return payByWechat(amount)
    case "bank":
        return payByBank(amount)
    // 新增支付方式需要改这里...
    }
    return errors.New("unknown method")
}
```

**正例**：
```go
// ✅ 通过接口扩展
type PaymentStrategy interface {
    Pay(amount float64) error
}

type AlipayStrategy struct{}
func (a *AlipayStrategy) Pay(amount float64) error { /* ... */ }

type WechatStrategy struct{}
func (w *WechatStrategy) Pay(amount float64) error { /* ... */ }

// 工厂/注册表模式
var strategies = map[string]PaymentStrategy{}

func RegisterStrategy(name string, s PaymentStrategy) {
    strategies[name] = s
}

func Pay(method string, amount float64) error {
    s, ok := strategies[method]
    if !ok {
        return errors.New("unknown method")
    }
    return s.Pay(amount)
}
```

**架构层面应用**：插件化架构
```
┌─────────────────────────────────────────┐
│              核心系统                    │
│  ┌─────────────────────────────────┐   │
│  │        插件接口定义              │   │
│  └─────────────────────────────────┘   │
│                  │                      │
│      ┌───────────┼───────────┐         │
│      ▼           ▼           ▼         │
│  ┌───────┐   ┌───────┐   ┌───────┐    │
│  │插件 A │   │插件 B │   │插件 C │    │
│  └───────┘   └───────┘   └───────┘    │
└─────────────────────────────────────────┘
```

---

### 3. 里氏替换原则 (LSP)

> 子类必须能够替换其父类

**反例**：
```java
// ❌ 违反 LSP
class Rectangle {
    protected int width, height;

    public void setWidth(int w) { this.width = w; }
    public void setHeight(int h) { this.height = h; }
    public int getArea() { return width * height; }
}

class Square extends Rectangle {
    // 正方形强制宽高相等，破坏了父类的契约
    public void setWidth(int w) {
        this.width = w;
        this.height = w;  // 违反预期！
    }
}

// 客户端代码会出问题
Rectangle r = new Square();
r.setWidth(5);
r.setHeight(10);
assert r.getArea() == 50;  // 失败！实际是 100
```

**架构层面应用**：
- 微服务版本升级必须向后兼容
- API 契约一旦发布不能随意变更

---

### 4. 接口隔离原则 (ISP)

> 客户端不应该被迫依赖它不使用的接口

**反例**：
```typescript
// ❌ 接口太胖
interface UserService {
    register(user: User): void;
    login(email: string, password: string): Token;
    updateProfile(userId: string, profile: Profile): void;
    deleteUser(userId: string): void;
    listUsers(): User[];           // 管理员功能
    banUser(userId: string): void; // 管理员功能
}
```

**正例**：
```typescript
// ✅ 接口拆分
interface UserAuthService {
    register(user: User): void;
    login(email: string, password: string): Token;
}

interface UserProfileService {
    updateProfile(userId: string, profile: Profile): void;
}

interface UserAdminService {
    listUsers(): User[];
    banUser(userId: string): void;
    deleteUser(userId: string): void;
}
```

**架构层面应用**：API 按角色/场景拆分
```
/api/v1/user/       # 普通用户 API
/api/v1/admin/      # 管理员 API
/api/v1/internal/   # 内部服务 API
```

---

### 5. 依赖倒置原则 (DIP)

> 高层模块不应依赖低层模块，两者都应依赖抽象

**反例**：
```python
# ❌ 高层直接依赖具体实现
class OrderService:
    def __init__(self):
        self.db = MySQLDatabase()  # 直接依赖 MySQL
        self.cache = RedisCache()  # 直接依赖 Redis

    def create_order(self, order):
        self.db.save(order)
        self.cache.delete(f"user:{order.user_id}:orders")
```

**正例**：
```python
# ✅ 依赖抽象
from abc import ABC, abstractmethod

class Database(ABC):
    @abstractmethod
    def save(self, entity): pass

class Cache(ABC):
    @abstractmethod
    def delete(self, key): pass

class OrderService:
    def __init__(self, db: Database, cache: Cache):
        self.db = db
        self.cache = cache

    def create_order(self, order):
        self.db.save(order)
        self.cache.delete(f"user:{order.user_id}:orders")

# 依赖注入
service = OrderService(
    db=MySQLDatabase(),
    cache=RedisCache()
)
```

**架构层面应用**：
```
┌─────────────────────────────────────────────────────┐
│                   业务逻辑层                         │
│               （依赖抽象接口）                        │
└─────────────────────────────────────────────────────┘
                        │
                        ▼ 依赖接口
┌─────────────────────────────────────────────────────┐
│                    接口层                            │
│     DatabasePort    CachePort    MessagePort        │
└─────────────────────────────────────────────────────┘
                        │
                        ▼ 实现接口
┌─────────────────────────────────────────────────────┐
│                   基础设施层                         │
│   MySQLAdapter   RedisAdapter   KafkaAdapter        │
└─────────────────────────────────────────────────────┘
```

---

## 二、其他重要原则

### DRY (Don't Repeat Yourself)

> 避免重复代码

**注意**：
- 代码重复 ≠ 逻辑重复
- 两段代码相似但服务不同业务 → 可以保持"重复"
- 过度 DRY 会导致不恰当的耦合

```python
# ❌ 过度 DRY - 强行抽象
def calculate_discount(type, amount):
    if type == "vip":
        return amount * 0.8
    elif type == "normal":
        return amount * 0.95
    elif type == "employee":  # 员工折扣和 VIP 一样
        return amount * 0.8

# ✅ 保持适度"重复" - 未来可能独立演进
def calculate_vip_discount(amount):
    return amount * 0.8

def calculate_employee_discount(amount):
    return amount * 0.8  # 现在相同，但可能分别调整
```

---

### KISS (Keep It Simple, Stupid)

> 保持简单

**反例**：
```java
// ❌ 过度设计
interface PaymentProcessor {
    void process(PaymentContext context);
}

interface PaymentContext {
    PaymentRequest getRequest();
    PaymentResponse getResponse();
    PaymentMetadata getMetadata();
}

abstract class AbstractPaymentProcessor implements PaymentProcessor {
    protected abstract void validate(PaymentContext context);
    protected abstract void preProcess(PaymentContext context);
    protected abstract void doProcess(PaymentContext context);
    protected abstract void postProcess(PaymentContext context);

    @Override
    public void process(PaymentContext context) {
        validate(context);
        preProcess(context);
        doProcess(context);
        postProcess(context);
    }
}
// 还需要 Factory, Builder, Strategy, Observer...
```

**正例**：
```java
// ✅ 简单直接
class PaymentService {
    public PaymentResult pay(PaymentRequest request) {
        validateRequest(request);
        return gateway.process(request);
    }
}
// 等真正需要时再重构
```

**架构层面**：
- 初创公司用单体架构，而非一开始就微服务
- 日活不到 100 万，不需要分布式数据库
- 没有高并发场景，不需要消息队列

---

### YAGNI (You Ain't Gonna Need It)

> 不要实现你现在不需要的功能

**反例**：
```
"以后可能需要支持多语言，先把 i18n 框架搭好"
"以后可能有 100 万用户，先做分库分表"
"以后可能要上云，先把 K8s 搞定"
```

**正确做法**：
1. 解决当前问题
2. 留好扩展点（接口抽象）
3. 真正需要时再实现

---

## 三、架构原则

### 1. 关注点分离

```
┌────────────────────────────────────────────────────────┐
│                    表示层 (Presentation)               │
│                  UI / API / CLI                        │
├────────────────────────────────────────────────────────┤
│                    应用层 (Application)                │
│              用例编排、事务管理                         │
├────────────────────────────────────────────────────────┤
│                    领域层 (Domain)                     │
│              业务逻辑、领域模型                         │
├────────────────────────────────────────────────────────┤
│                  基础设施层 (Infrastructure)           │
│              数据库、缓存、消息队列、外部服务            │
└────────────────────────────────────────────────────────┘
```

### 2. 单一数据源

> 每种数据应该有唯一的权威来源

```
❌ 订单金额在 订单服务 和 支付服务 各存一份
✅ 订单金额只在 订单服务 存储，支付服务从订单服务获取
```

### 3. 失败设计 (Design for Failure)

```python
# ✅ 假设一切都会失败
class OrderService:
    def create_order(self, order):
        try:
            # 1. 保存订单（本地事务）
            self.order_repo.save(order)

            # 2. 扣减库存（可能失败）
            try:
                self.inventory_client.deduct(order.items)
            except Exception as e:
                # 库存服务失败，标记订单待处理
                self.order_repo.mark_pending(order.id)
                self.alert("库存扣减失败", order.id, e)
                return

            # 3. 发送通知（允许失败）
            try:
                self.notification.send(order)
            except Exception:
                pass  # 通知失败不影响主流程

        except Exception as e:
            self.logger.error(f"订单创建失败: {e}")
            raise
```

### 4. 幂等设计

> 同一操作执行多次的效果与执行一次相同

```python
# ✅ 幂等的支付接口
class PaymentService:
    def pay(self, order_id: str, amount: float, idempotency_key: str):
        # 检查是否已处理
        existing = self.cache.get(f"payment:{idempotency_key}")
        if existing:
            return existing  # 返回之前的结果

        # 使用数据库唯一约束防止重复
        try:
            result = self._do_pay(order_id, amount)
            self.cache.set(f"payment:{idempotency_key}", result, ttl=86400)
            return result
        except DuplicateError:
            return self.get_payment_by_order(order_id)
```

### 5. 无状态设计

> 服务实例不保存请求相关状态

```
❌ Session 存在应用服务器内存
✅ Session 存在 Redis

❌ 上传文件到本地磁盘
✅ 上传文件到对象存储（S3/OSS）

❌ 定时任务在单机执行
✅ 定时任务用分布式调度（XXL-Job）
```

---

## 四、实战检查清单

### 代码设计检查
- [ ] 类的职责是否单一？
- [ ] 新增功能是否需要修改已有代码？
- [ ] 是否依赖抽象而非具体实现？
- [ ] 是否有不必要的复杂性？
- [ ] 是否有过早优化？

### 架构设计检查
- [ ] 服务边界是否清晰？
- [ ] 是否考虑了失败场景？
- [ ] 接口是否幂等？
- [ ] 服务是否无状态？
- [ ] 数据源是否唯一？
