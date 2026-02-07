# 云设计模式

## 目录
- [设计模式概览](#设计模式概览)
- [可靠性模式](#可靠性模式)
- [性能模式](#性能模式)
- [可扩展性模式](#可扩展性模式)
- [数据管理模式](#数据管理模式)
- [安全模式](#安全模式)

---

## 设计模式概览

### 云设计模式分类

```
┌────────────────────────────────────────────────────────┐
│              云设计模式全景图                          │
├────────────────────────────────────────────────────────┤
│                                                        │
│  🛡️  可靠性模式 (Reliability)                         │
│     ├─ Retry (重试)                                   │
│     ├─ Circuit Breaker (断路器)                       │
│     ├─ Bulkhead (舱壁隔离)                            │
│     └─ Health Endpoint Monitoring (健康检查)          │
│                                                        │
│  ⚡ 性能模式 (Performance)                            │
│     ├─ Cache-Aside (旁路缓存)                         │
│     ├─ CQRS (读写分离)                                │
│     ├─ Static Content Hosting (静态内容托管)          │
│     └─ Throttling (限流)                              │
│                                                        │
│  📈 可扩展性模式 (Scalability)                        │
│     ├─ Queue-Based Load Leveling (队列削峰)          │
│     ├─ Auto-Scaling (自动扩展)                        │
│     ├─ Competing Consumers (竞争消费者)               │
│     └─ Sharding (分片)                                │
│                                                        │
│  💾 数据管理模式 (Data Management)                    │
│     ├─ Event Sourcing (事件溯源)                      │
│     ├─ Saga (长事务)                                  │
│     ├─ Materialized View (物化视图)                  │
│     └─ Database per Service (每服务一个数据库)        │
│                                                        │
│  🔒 安全模式 (Security)                               │
│     ├─ Federated Identity (联合身份)                 │
│     ├─ Gatekeeper (守门员)                            │
│     ├─ Valet Key (代客钥匙)                           │
│     └─ Token-Based Authentication (令牌认证)          │
└────────────────────────────────────────────────────────┘
```

---

## 可靠性模式

### 1. Retry Pattern (重试模式)

**问题**：网络抖动或临时故障导致请求失败

**解决方案**：自动重试失败的操作

```
┌────────────────────────────────────────┐
│          重试策略对比                  │
├────────────────────────────────────────┤
│                                        │
│  1. 立即重试 (Immediate Retry)        │
│     请求 ─X─▶ 立即重试 ─X─▶ 立即重试 │
│     适用: 瞬时故障                     │
│                                        │
│  2. 固定间隔 (Fixed Interval)         │
│     请求 ─X─▶ 等1秒 ─▶ 重试          │
│     适用: 一般场景                     │
│                                        │
│  3. 指数退避 (Exponential Backoff)    │
│     请求 ─X─▶ 等1秒 ─X─▶ 等2秒       │
│           ─X─▶ 等4秒 ─X─▶ 等8秒      │
│     适用: 防止服务雪崩 ⭐              │
│                                        │
│  4. 随机抖动 (Jitter)                 │
│     在退避时间基础上增加随机值         │
│     避免重试风暴                       │
└────────────────────────────────────────┘
```

**Python 实现**：
```python
# retry_pattern.py
import time
import random
import logging
from functools import wraps
from typing import Type, Tuple

logger = logging.getLogger(__name__)

def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    指数退避重试装饰器

    参数:
        max_retries: 最大重试次数
        base_delay: 基础延迟（秒）
        max_delay: 最大延迟（秒）
        exponential_base: 指数基数
        jitter: 是否添加随机抖动
        exceptions: 需要重试的异常类型
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0

            while True:
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    retries += 1

                    if retries > max_retries:
                        logger.error(
                            f"❌ {func.__name__} 失败 (已重试 {max_retries} 次): {e}"
                        )
                        raise

                    # 计算延迟时间
                    delay = min(
                        base_delay * (exponential_base ** (retries - 1)),
                        max_delay
                    )

                    # 添加随机抖动（避免重试风暴）
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"⚠️  {func.__name__} 失败，{delay:.2f}秒后第{retries}次重试: {e}"
                    )

                    time.sleep(delay)

        return wrapper
    return decorator

# 使用示例
import requests
from requests.exceptions import RequestException

@retry_with_exponential_backoff(
    max_retries=5,
    base_delay=1.0,
    exceptions=(RequestException,)
)
def call_external_api(url: str):
    """调用外部 API"""
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.json()

# 测试
try:
    data = call_external_api("https://api.example.com/data")
    print("✅ 成功:", data)
except RequestException as e:
    print("❌ 最终失败:", e)
```

### 2. Circuit Breaker Pattern (断路器模式)

**问题**：下游服务故障导致级联失败

**解决方案**：快速失败，避免无谓重试

```
┌────────────────────────────────────────────────────┐
│              断路器状态机                          │
├────────────────────────────────────────────────────┤
│                                                    │
│           成功次数达到阈值                         │
│      ┌──────────────────────┐                    │
│      │                      │                     │
│      │         ┌──────────┐ │                    │
│      └────────▶│  Closed  │◀┘                    │
│      超时复位   │  (关闭)  │                      │
│                └────┬─────┘                       │
│                     │ 失败率超过阈值               │
│                     ▼                              │
│                ┌──────────┐                       │
│           ┌───▶│  Open    │                       │
│  失败     │    │  (打开)  │                       │
│           │    └────┬─────┘                       │
│           │         │ 超时后                      │
│           │         ▼                              │
│           │    ┌──────────┐                       │
│           └────│Half-Open │                       │
│                │ (半开)   │                       │
│                └──────────┘                       │
│                                                    │
│  Closed:   正常转发请求                           │
│  Open:     直接返回错误（快速失败）               │
│  Half-Open: 尝试性转发，测试服务是否恢复          │
└────────────────────────────────────────────────────┘
```

**Python 实现**：
```python
# circuit_breaker.py
import time
import threading
from enum import Enum
from functools import wraps
from collections import deque

class CircuitState(Enum):
    CLOSED = "closed"       # 正常状态
    OPEN = "open"           # 断开状态
    HALF_OPEN = "half_open" # 半开状态

class CircuitBreaker:
    """断路器实现"""

    def __init__(
        self,
        failure_threshold: int = 5,      # 失败阈值
        recovery_timeout: float = 60.0,  # 恢复超时（秒）
        expected_exception: Type[Exception] = Exception,
        name: str = None
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name or "CircuitBreaker"

        # 状态
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0

        # 线程锁
        self._lock = threading.Lock()

        # 失败时间窗口（用于统计失败率）
        self.failure_window = deque(maxlen=100)

    def call(self, func, *args, **kwargs):
        """执行函数调用"""
        with self._lock:
            # 检查是否需要从 OPEN 切换到 HALF_OPEN
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    print(f"🔄 {self.name}: OPEN -> HALF_OPEN")
                else:
                    raise CircuitBreakerError(
                        f"断路器打开，拒绝请求（{self.recovery_timeout}秒后自动恢复）"
                    )

        # 尝试执行函数
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """成功回调"""
        with self._lock:
            self.failure_count = 0

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                # 连续成功3次，关闭断路器
                if self.success_count >= 3:
                    self.state = CircuitState.CLOSED
                    print(f"✅ {self.name}: HALF_OPEN -> CLOSED (服务恢复)")

    def _on_failure(self):
        """失败回调"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.failure_window.append(time.time())

            # HALF_OPEN 状态下失败，立即打开断路器
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                print(f"❌ {self.name}: HALF_OPEN -> OPEN (服务仍未恢复)")

            # CLOSED 状态下失败次数超过阈值
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    print(f"🚨 {self.name}: CLOSED -> OPEN (失败次数: {self.failure_count})")

    def __call__(self, func):
        """装饰器用法"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper

class CircuitBreakerError(Exception):
    """断路器错误"""
    pass

# 使用示例
@CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=10.0,
    expected_exception=RequestException,
    name="ExternalAPI"
)
def call_flaky_service():
    """调用不稳定的服务"""
    response = requests.get("https://flaky-api.com/data", timeout=2)
    response.raise_for_status()
    return response.json()

# 测试断路器
for i in range(10):
    try:
        result = call_flaky_service()
        print(f"调用 {i+1}: 成功")
    except CircuitBreakerError as e:
        print(f"调用 {i+1}: 断路器阻止 - {e}")
    except RequestException as e:
        print(f"调用 {i+1}: 服务失败 - {e}")

    time.sleep(1)
```

### 3. Bulkhead Pattern (舱壁隔离模式)

**问题**：单个资源耗尽影响整个系统

**解决方案**：资源隔离，防止级联失败

```
┌────────────────────────────────────────────────────┐
│              舱壁隔离架构                          │
├────────────────────────────────────────────────────┤
│                                                    │
│  无隔离（泰坦尼克号）      有隔离（现代邮轮）     │
│  ┌──────────────────┐      ┌──────────────────┐  │
│  │                  │      │ 🚢 服务A (20线程)│  │
│  │  共享线程池      │      ├──────────────────┤  │
│  │  (100线程)       │      │ 🚢 服务B (30线程)│  │
│  │                  │      ├──────────────────┤  │
│  │  一个服务阻塞    │      │ 🚢 服务C (50线程)│  │
│  │  ↓               │      └──────────────────┘  │
│  │  全部线程耗尽 💀 │      单个服务故障不影响其他 │
│  └──────────────────┘                            │
└────────────────────────────────────────────────────┘
```

**Python 实现**：
```python
# bulkhead_pattern.py
import concurrent.futures
from functools import wraps

class Bulkhead:
    """舱壁隔离：限制并发执行数"""

    def __init__(self, max_concurrent: int, name: str = "Bulkhead"):
        self.max_concurrent = max_concurrent
        self.name = name
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix=name
        )

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                future = self.executor.submit(func, *args, **kwargs)
                return future.result()
            except concurrent.futures.ThreadPoolExecutor as e:
                raise BulkheadFullError(
                    f"{self.name} 资源池已满 (max={self.max_concurrent})"
                )
        return wrapper

class BulkheadFullError(Exception):
    pass

# 使用示例：为不同服务分配独立资源池
payment_service_pool = Bulkhead(max_concurrent=10, name="PaymentService")
email_service_pool = Bulkhead(max_concurrent=5, name="EmailService")
analytics_pool = Bulkhead(max_concurrent=20, name="Analytics")

@payment_service_pool
def process_payment(amount):
    """支付服务（关键服务，独立池）"""
    time.sleep(2)  # 模拟慢操作
    return f"Payment ${amount} processed"

@email_service_pool
def send_email(to, subject):
    """邮件服务（非关键，小池）"""
    time.sleep(5)  # 模拟很慢的操作
    return f"Email sent to {to}"

@analytics_pool
def log_analytics(event):
    """分析服务（大量并发，大池）"""
    time.sleep(0.1)
    return f"Event logged: {event}"

# 即使邮件服务阻塞（5秒），支付服务仍可正常工作
```

---

## 性能模式

### 4. Cache-Aside Pattern (旁路缓存)

```
┌────────────────────────────────────────────────────┐
│              Cache-Aside 流程                      │
├────────────────────────────────────────────────────┤
│                                                    │
│  读取流程:                                         │
│  ┌──────┐  1. 查询缓存   ┌───────┐               │
│  │ 应用 │──────────────▶│ Cache │               │
│  └───┬──┘                └───┬───┘               │
│      │                       │                    │
│      │ 2. 缓存未命中 (miss)  │                    │
│      │                       │                    │
│      │  3. 查询数据库  ┌─────▼────┐              │
│      └────────────────▶│ Database │              │
│      │                 └─────┬────┘              │
│      │ 4. 返回数据           │                    │
│      ◀───────────────────────┘                    │
│      │                                            │
│      │  5. 写入缓存    ┌───────┐                 │
│      └────────────────▶│ Cache │                 │
│                        └───────┘                  │
│                                                    │
│  写入流程:                                         │
│  ┌──────┐  1. 写数据库  ┌──────────┐             │
│  │ 应用 │──────────────▶│ Database │             │
│  └───┬──┘                └──────────┘             │
│      │                                            │
│      │  2. 删除缓存    ┌───────┐                 │
│      └────────────────▶│ Cache │ (失效)          │
│                        └───────┘                  │
└────────────────────────────────────────────────────┘
```

**Python 实现**：
```python
# cache_aside.py
import redis
import json
import hashlib
from functools import wraps
from typing import Optional, Callable

class CacheAside:
    """旁路缓存装饰器"""

    def __init__(
        self,
        redis_client: redis.Redis,
        ttl: int = 3600,
        key_prefix: str = "cache"
    ):
        self.redis = redis_client
        self.ttl = ttl
        self.key_prefix = key_prefix

    def _generate_cache_key(self, func_name: str, args, kwargs) -> str:
        """生成缓存键"""
        # 将参数序列化为字符串
        params_str = json.dumps({
            'args': args,
            'kwargs': kwargs
        }, sort_keys=True)

        # MD5 哈希
        hash_value = hashlib.md5(params_str.encode()).hexdigest()

        return f"{self.key_prefix}:{func_name}:{hash_value}"

    def __call__(self, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 1. 生成缓存键
            cache_key = self._generate_cache_key(func.__name__, args, kwargs)

            # 2. 尝试从缓存读取
            cached_value = self.redis.get(cache_key)
            if cached_value:
                print(f"✅ 缓存命中: {cache_key}")
                return json.loads(cached_value)

            print(f"❌ 缓存未命中: {cache_key}")

            # 3. 缓存未命中，执行函数
            result = func(*args, **kwargs)

            # 4. 写入缓存
            self.redis.setex(
                cache_key,
                self.ttl,
                json.dumps(result, default=str)
            )

            return result

        return wrapper

# 使用示例
redis_client = redis.Redis(host='localhost', port=6379, db=0)
cache = CacheAside(redis_client, ttl=600)

@cache
def get_user_profile(user_id: int):
    """从数据库获取用户资料（慢操作）"""
    print(f"  📊 查询数据库: user_id={user_id}")
    time.sleep(2)  # 模拟数据库查询
    return {
        'user_id': user_id,
        'name': f'User {user_id}',
        'email': f'user{user_id}@example.com'
    }

def update_user_profile(user_id: int, data: dict):
    """更新用户资料"""
    # 1. 更新数据库
    print(f"  💾 更新数据库: user_id={user_id}")
    # db.users.update({'user_id': user_id}, data)

    # 2. 删除缓存（Cache Invalidation）
    cache_key = f"cache:get_user_profile:{user_id}"
    redis_client.delete(cache_key)
    print(f"  🗑️  删除缓存: {cache_key}")

# 测试
print("第1次调用（缓存未命中）:")
profile = get_user_profile(123)  # 2秒

print("\n第2次调用（缓存命中）:")
profile = get_user_profile(123)  # <1ms

print("\n更新资料:")
update_user_profile(123, {'name': 'New Name'})

print("\n第3次调用（缓存已失效）:")
profile = get_user_profile(123)  # 2秒
```

### 5. CQRS Pattern (命令查询职责分离)

```
┌────────────────────────────────────────────────────┐
│              CQRS 架构                             │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌─────────┐                                      │
│  │  客户端 │                                      │
│  └────┬────┘                                      │
│       │                                            │
│  ─────┼────────────────────────────────           │
│       │                                            │
│  ┌────▼─────────┐          ┌──────────────┐      │
│  │   命令 API   │          │   查询 API    │      │
│  │  (写操作)    │          │  (读操作)     │      │
│  └────┬─────────┘          └──────┬───────┘      │
│       │                            │               │
│  ┌────▼─────────┐          ┌──────▼───────┐      │
│  │  写数据库    │          │  读数据库     │      │
│  │  (PostgreSQL)│─事件同步─▶│  (ES/MongoDB)│      │
│  └──────────────┘          └──────────────┘      │
│   标准化存储                   优化查询             │
└────────────────────────────────────────────────────┘
```

**实现示例** (见下页)

---

## 总结

### 模式选择矩阵

```
┌──────────────┬───────────┬────────────┬──────────┐
│   场景       │  推荐模式 │  复杂度    │  收益    │
├──────────────┼───────────┼────────────┼──────────┤
│ API 调用失败 │  Retry    │  低 ⭐     │  高      │
│ 服务雪崩     │  Circuit  │  中 ⭐⭐   │  极高    │
│ 资源隔离     │  Bulkhead │  中 ⭐⭐   │  高      │
│ 数据库压力   │  Cache    │  低 ⭐     │  极高    │
│ 读写分离     │  CQRS     │  高 ⭐⭐⭐ │  中      │
└──────────────┴───────────┴────────────┴──────────┘
```

### 下一步学习

- [04_multi_cloud.md](04_multi_cloud.md) - 多云与混合云
- [05_cost_optimization.md](05_cost_optimization.md) - 成本优化
