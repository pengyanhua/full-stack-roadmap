"""
============================================================
                Redis 实战示例（Python）
============================================================
使用 redis-py 库实现常见的 Redis 应用场景
pip install redis
============================================================
"""

import redis
import json
import time
import hashlib
import random
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


# ============================================================
#                    连接配置
# ============================================================

def get_redis_client() -> redis.Redis:
    """获取 Redis 连接"""
    return redis.Redis(
        host='localhost',
        port=6379,
        password=None,  # 如果有密码
        db=0,
        decode_responses=True,  # 自动解码为字符串
        socket_timeout=5,
        socket_connect_timeout=5,
        retry_on_timeout=True,
    )


def get_redis_pool() -> redis.ConnectionPool:
    """获取连接池"""
    return redis.ConnectionPool(
        host='localhost',
        port=6379,
        db=0,
        max_connections=100,
        decode_responses=True,
    )


# ============================================================
#                    一、分布式锁
# ============================================================

class DistributedLock:
    """
    分布式锁实现

    特点：
    1. 互斥性：同一时刻只有一个客户端持有锁
    2. 防死锁：设置超时时间
    3. 解铃还须系铃人：只能释放自己的锁
    4. 可重入（可选）
    """

    def __init__(self, client: redis.Redis, name: str, timeout: int = 30):
        self.client = client
        self.name = f"lock:{name}"
        self.timeout = timeout
        self.identifier = None

    def acquire(self, blocking: bool = True, blocking_timeout: int = None) -> bool:
        """
        获取锁

        Args:
            blocking: 是否阻塞等待
            blocking_timeout: 阻塞超时时间

        Returns:
            是否成功获取锁
        """
        # 生成唯一标识
        self.identifier = f"{time.time()}:{random.random()}"

        end_time = time.time() + (blocking_timeout or self.timeout)

        while True:
            # 尝试获取锁
            if self.client.set(self.name, self.identifier, nx=True, ex=self.timeout):
                return True

            if not blocking:
                return False

            if time.time() >= end_time:
                return False

            time.sleep(0.1)

    def release(self) -> bool:
        """释放锁"""
        if not self.identifier:
            return False

        # 使用 Lua 脚本保证原子性
        script = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('DEL', KEYS[1])
        end
        return 0
        """
        result = self.client.eval(script, 1, self.name, self.identifier)
        self.identifier = None
        return result == 1

    def extend(self, additional_time: int = None) -> bool:
        """续期"""
        if not self.identifier:
            return False

        script = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('PEXPIRE', KEYS[1], ARGV[2])
        end
        return 0
        """
        timeout = additional_time or self.timeout
        return self.client.eval(script, 1, self.name, self.identifier, timeout * 1000) == 1

    @contextmanager
    def __call__(self, blocking: bool = True):
        """上下文管理器"""
        acquired = self.acquire(blocking=blocking)
        try:
            yield acquired
        finally:
            if acquired:
                self.release()


# 使用示例
def distributed_lock_example():
    """分布式锁示例"""
    client = get_redis_client()
    lock = DistributedLock(client, "order:1001")

    # 方式一：手动获取释放
    if lock.acquire():
        try:
            print("获取锁成功，执行业务逻辑...")
        finally:
            lock.release()

    # 方式二：上下文管理器
    with lock() as acquired:
        if acquired:
            print("获取锁成功，执行业务逻辑...")


# ============================================================
#                    二、限流器
# ============================================================

class RateLimiter:
    """
    限流器实现

    支持多种算法：
    1. 固定窗口
    2. 滑动窗口
    3. 令牌桶
    4. 漏桶
    """

    def __init__(self, client: redis.Redis, name: str):
        self.client = client
        self.name = name

    def fixed_window(self, limit: int, window: int = 60) -> bool:
        """
        固定窗口限流

        Args:
            limit: 窗口内最大请求数
            window: 窗口大小（秒）
        """
        key = f"ratelimit:fixed:{self.name}:{int(time.time() / window)}"

        current = self.client.incr(key)
        if current == 1:
            self.client.expire(key, window)

        return current <= limit

    def sliding_window(self, limit: int, window: int = 60) -> bool:
        """
        滑动窗口限流

        使用 Sorted Set 实现
        """
        key = f"ratelimit:sliding:{self.name}"
        now = time.time()
        window_start = now - window

        # Lua 脚本保证原子性
        script = """
        local key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local window_start = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local member = ARGV[4]

        -- 移除窗口外的记录
        redis.call('ZREMRANGEBYSCORE', key, 0, window_start)

        -- 获取当前窗口的请求数
        local count = redis.call('ZCARD', key)

        if count < limit then
            redis.call('ZADD', key, now, member)
            redis.call('EXPIRE', key, ARGV[5])
            return 1
        end
        return 0
        """

        member = f"{now}:{random.random()}"
        result = self.client.eval(script, 1, key, limit, window_start, now, member, window)
        return result == 1

    def token_bucket(self, rate: float, capacity: int) -> bool:
        """
        令牌桶限流

        Args:
            rate: 令牌生成速率（个/秒）
            capacity: 桶容量
        """
        key = f"ratelimit:token:{self.name}"
        now = time.time()

        script = """
        local key = KEYS[1]
        local rate = tonumber(ARGV[1])
        local capacity = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])

        local data = redis.call('HMGET', key, 'tokens', 'last_time')
        local tokens = tonumber(data[1]) or capacity
        local last_time = tonumber(data[2]) or now

        -- 计算生成的令牌数
        local elapsed = now - last_time
        local new_tokens = math.min(capacity, tokens + elapsed * rate)

        if new_tokens >= 1 then
            redis.call('HMSET', key, 'tokens', new_tokens - 1, 'last_time', now)
            redis.call('EXPIRE', key, 3600)
            return 1
        else
            redis.call('HMSET', key, 'tokens', new_tokens, 'last_time', now)
            redis.call('EXPIRE', key, 3600)
            return 0
        end
        """

        result = self.client.eval(script, 1, key, rate, capacity, now)
        return result == 1


# 使用示例
def rate_limiter_example():
    """限流器示例"""
    client = get_redis_client()
    limiter = RateLimiter(client, "api:user:1001")

    # 滑动窗口：60秒内最多100次请求
    if limiter.sliding_window(limit=100, window=60):
        print("请求允许")
    else:
        print("请求被限流")


# ============================================================
#                    三、缓存封装
# ============================================================

class Cache:
    """
    缓存封装

    特点：
    1. 防止缓存穿透（空值缓存）
    2. 防止缓存击穿（互斥锁）
    3. 防止缓存雪崩（随机过期时间）
    """

    def __init__(self, client: redis.Redis, prefix: str = "cache"):
        self.client = client
        self.prefix = prefix

    def _key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        value = self.client.get(self._key(key))
        if value is None:
            return None
        if value == "__NULL__":  # 空值标记
            return None
        return json.loads(value)

    def set(self, key: str, value: Any, ttl: int = 3600, add_random: bool = True):
        """
        设置缓存

        Args:
            ttl: 过期时间（秒）
            add_random: 是否添加随机时间（防止雪崩）
        """
        if add_random:
            ttl = ttl + random.randint(0, 300)

        self.client.setex(
            self._key(key),
            ttl,
            json.dumps(value, ensure_ascii=False)
        )

    def set_null(self, key: str, ttl: int = 300):
        """设置空值缓存（防止穿透）"""
        self.client.setex(self._key(key), ttl, "__NULL__")

    def delete(self, key: str):
        """删除缓存"""
        self.client.delete(self._key(key))

    def get_or_set(
        self,
        key: str,
        func,
        ttl: int = 3600,
        lock_timeout: int = 10
    ) -> Optional[Any]:
        """
        获取缓存，不存在则加载（防止击穿）

        Args:
            func: 数据加载函数
            lock_timeout: 锁超时时间
        """
        # 尝试从缓存获取
        value = self.get(key)
        if value is not None:
            return value

        # 获取锁
        lock_key = f"lock:{self._key(key)}"
        lock = DistributedLock(self.client, lock_key, timeout=lock_timeout)

        if lock.acquire(blocking=True, blocking_timeout=lock_timeout):
            try:
                # 双重检查
                value = self.get(key)
                if value is not None:
                    return value

                # 加载数据
                value = func()
                if value is not None:
                    self.set(key, value, ttl)
                else:
                    self.set_null(key)  # 缓存空值

                return value
            finally:
                lock.release()
        else:
            # 获取锁失败，等待后重试
            time.sleep(0.1)
            return self.get(key)


# 使用示例
def cache_example():
    """缓存示例"""
    client = get_redis_client()
    cache = Cache(client, "user")

    def load_user(user_id: int):
        # 模拟从数据库加载
        return {"id": user_id, "name": "Alice"}

    # 获取用户（自动处理缓存）
    user = cache.get_or_set("user:1001", lambda: load_user(1001))
    print(user)


# ============================================================
#                    四、排行榜
# ============================================================

class Leaderboard:
    """
    排行榜实现

    使用 Sorted Set
    """

    def __init__(self, client: redis.Redis, name: str):
        self.client = client
        self.key = f"leaderboard:{name}"

    def add_score(self, member: str, score: float):
        """添加/更新分数"""
        self.client.zadd(self.key, {member: score})

    def incr_score(self, member: str, increment: float) -> float:
        """增加分数"""
        return self.client.zincrby(self.key, increment, member)

    def get_score(self, member: str) -> Optional[float]:
        """获取分数"""
        return self.client.zscore(self.key, member)

    def get_rank(self, member: str, reverse: bool = True) -> Optional[int]:
        """
        获取排名

        Args:
            reverse: True 为降序排名（分数高的排前面）
        """
        if reverse:
            rank = self.client.zrevrank(self.key, member)
        else:
            rank = self.client.zrank(self.key, member)
        return rank + 1 if rank is not None else None

    def get_top(self, n: int = 10, with_scores: bool = True) -> List:
        """获取前 N 名"""
        return self.client.zrevrange(self.key, 0, n - 1, withscores=with_scores)

    def get_around(self, member: str, n: int = 5) -> List:
        """获取某成员附近的排名"""
        rank = self.client.zrevrank(self.key, member)
        if rank is None:
            return []

        start = max(0, rank - n)
        end = rank + n
        return self.client.zrevrange(self.key, start, end, withscores=True)

    def get_page(self, page: int, page_size: int = 10) -> List:
        """分页获取"""
        start = (page - 1) * page_size
        end = start + page_size - 1
        return self.client.zrevrange(self.key, start, end, withscores=True)

    def count(self) -> int:
        """获取总人数"""
        return self.client.zcard(self.key)


# 使用示例
def leaderboard_example():
    """排行榜示例"""
    client = get_redis_client()
    lb = Leaderboard(client, "game:daily")

    # 添加分数
    lb.add_score("player:1001", 1000)
    lb.add_score("player:1002", 950)
    lb.add_score("player:1003", 1100)

    # 增加分数
    lb.incr_score("player:1001", 50)

    # 获取前 10 名
    top10 = lb.get_top(10)
    print("Top 10:", top10)

    # 获取排名
    rank = lb.get_rank("player:1001")
    print(f"Player 1001 排名: {rank}")


# ============================================================
#                    五、消息队列
# ============================================================

class MessageQueue:
    """
    消息队列实现

    基于 List 的简单队列
    """

    def __init__(self, client: redis.Redis, name: str):
        self.client = client
        self.key = f"mq:{name}"

    def push(self, message: Any):
        """发送消息"""
        self.client.rpush(self.key, json.dumps(message, ensure_ascii=False))

    def pop(self, timeout: int = 0) -> Optional[Any]:
        """
        接收消息

        Args:
            timeout: 超时时间，0 表示永久等待
        """
        if timeout == 0:
            result = self.client.blpop(self.key)
        else:
            result = self.client.blpop(self.key, timeout=timeout)

        if result:
            return json.loads(result[1])
        return None

    def length(self) -> int:
        """队列长度"""
        return self.client.llen(self.key)


class StreamQueue:
    """
    基于 Stream 的消息队列

    支持消费者组
    """

    def __init__(self, client: redis.Redis, name: str):
        self.client = client
        self.stream = f"stream:{name}"

    def publish(self, data: Dict) -> str:
        """发布消息，返回消息 ID"""
        return self.client.xadd(self.stream, data, maxlen=10000)

    def create_group(self, group: str, start_id: str = "$"):
        """创建消费者组"""
        try:
            self.client.xgroup_create(self.stream, group, id=start_id, mkstream=True)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def consume(
        self,
        group: str,
        consumer: str,
        count: int = 10,
        block: int = 5000
    ) -> List:
        """消费消息"""
        messages = self.client.xreadgroup(
            group, consumer,
            {self.stream: ">"},
            count=count,
            block=block
        )
        return messages[0][1] if messages else []

    def ack(self, group: str, message_id: str):
        """确认消息"""
        self.client.xack(self.stream, group, message_id)

    def pending(self, group: str) -> Dict:
        """查看待处理消息"""
        return self.client.xpending(self.stream, group)


# ============================================================
#                    六、延迟队列
# ============================================================

class DelayQueue:
    """
    延迟队列实现

    使用 Sorted Set，score 为执行时间
    """

    def __init__(self, client: redis.Redis, name: str):
        self.client = client
        self.key = f"delay:{name}"

    def add(self, task_id: str, data: Any, delay: int):
        """
        添加延迟任务

        Args:
            delay: 延迟时间（秒）
        """
        execute_time = time.time() + delay
        task = json.dumps({"id": task_id, "data": data}, ensure_ascii=False)
        self.client.zadd(self.key, {task: execute_time})

    def poll(self) -> Optional[Dict]:
        """获取到期的任务"""
        now = time.time()

        # Lua 脚本保证原子性
        script = """
        local tasks = redis.call('ZRANGEBYSCORE', KEYS[1], 0, ARGV[1], 'LIMIT', 0, 1)
        if #tasks > 0 then
            redis.call('ZREM', KEYS[1], tasks[1])
            return tasks[1]
        end
        return nil
        """

        result = self.client.eval(script, 1, self.key, now)
        if result:
            return json.loads(result)
        return None

    def poll_batch(self, batch_size: int = 10) -> List[Dict]:
        """批量获取到期任务"""
        now = time.time()

        script = """
        local tasks = redis.call('ZRANGEBYSCORE', KEYS[1], 0, ARGV[1], 'LIMIT', 0, ARGV[2])
        if #tasks > 0 then
            redis.call('ZREM', KEYS[1], unpack(tasks))
        end
        return tasks
        """

        results = self.client.eval(script, 1, self.key, now, batch_size)
        return [json.loads(r) for r in results] if results else []

    def size(self) -> int:
        """队列大小"""
        return self.client.zcard(self.key)


# 使用示例
def delay_queue_example():
    """延迟队列示例"""
    client = get_redis_client()
    queue = DelayQueue(client, "order:timeout")

    # 添加订单超时任务（30分钟后执行）
    queue.add("order:1001", {"action": "cancel"}, delay=1800)

    # 消费者轮询
    while True:
        task = queue.poll()
        if task:
            print(f"处理任务: {task}")
        time.sleep(1)


# ============================================================
#                    七、会话管理
# ============================================================

class SessionManager:
    """
    分布式会话管理
    """

    def __init__(self, client: redis.Redis, ttl: int = 3600):
        self.client = client
        self.ttl = ttl

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}"

    def create(self, user_id: str, data: Dict = None) -> str:
        """创建会话"""
        session_id = hashlib.sha256(
            f"{user_id}:{time.time()}:{random.random()}".encode()
        ).hexdigest()

        session_data = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            **(data or {})
        }

        self.client.hset(self._key(session_id), mapping=session_data)
        self.client.expire(self._key(session_id), self.ttl)

        return session_id

    def get(self, session_id: str) -> Optional[Dict]:
        """获取会话"""
        data = self.client.hgetall(self._key(session_id))
        if data:
            self.client.expire(self._key(session_id), self.ttl)  # 续期
            return data
        return None

    def update(self, session_id: str, data: Dict):
        """更新会话"""
        key = self._key(session_id)
        if self.client.exists(key):
            self.client.hset(key, mapping=data)
            self.client.expire(key, self.ttl)

    def destroy(self, session_id: str):
        """销毁会话"""
        self.client.delete(self._key(session_id))


# ============================================================
#                    八、布隆过滤器
# ============================================================

class BloomFilter:
    """
    布隆过滤器（纯 Redis 实现）

    注意：生产环境建议使用 RedisBloom 模块
    """

    def __init__(
        self,
        client: redis.Redis,
        name: str,
        capacity: int = 1000000,
        error_rate: float = 0.01
    ):
        self.client = client
        self.key = f"bloom:{name}"

        # 计算最优参数
        import math
        self.size = int(-capacity * math.log(error_rate) / (math.log(2) ** 2))
        self.hash_count = int((self.size / capacity) * math.log(2))

    def _hashes(self, item: str) -> List[int]:
        """生成多个哈希值"""
        hashes = []
        for i in range(self.hash_count):
            h = hashlib.md5(f"{item}:{i}".encode()).hexdigest()
            hashes.append(int(h, 16) % self.size)
        return hashes

    def add(self, item: str):
        """添加元素"""
        pipe = self.client.pipeline()
        for pos in self._hashes(item):
            pipe.setbit(self.key, pos, 1)
        pipe.execute()

    def contains(self, item: str) -> bool:
        """检查元素是否可能存在"""
        pipe = self.client.pipeline()
        for pos in self._hashes(item):
            pipe.getbit(self.key, pos)
        results = pipe.execute()
        return all(results)


# ============================================================
#                    九、计数器
# ============================================================

class Counter:
    """
    计数器集合
    """

    def __init__(self, client: redis.Redis, prefix: str):
        self.client = client
        self.prefix = prefix

    def _key(self, name: str) -> str:
        return f"{self.prefix}:{name}"

    def incr(self, name: str, amount: int = 1) -> int:
        """增加计数"""
        return self.client.incrby(self._key(name), amount)

    def decr(self, name: str, amount: int = 1) -> int:
        """减少计数"""
        return self.client.decrby(self._key(name), amount)

    def get(self, name: str) -> int:
        """获取计数"""
        value = self.client.get(self._key(name))
        return int(value) if value else 0

    def reset(self, name: str):
        """重置计数"""
        self.client.delete(self._key(name))


class HyperLogLogCounter:
    """
    基数统计（去重计数）

    适用于统计独立访客等场景
    """

    def __init__(self, client: redis.Redis, name: str):
        self.client = client
        self.key = f"hll:{name}"

    def add(self, *items: str):
        """添加元素"""
        self.client.pfadd(self.key, *items)

    def count(self) -> int:
        """获取基数（去重后的数量）"""
        return self.client.pfcount(self.key)

    def merge(self, *others: 'HyperLogLogCounter'):
        """合并多个 HyperLogLog"""
        keys = [o.key for o in others]
        self.client.pfmerge(self.key, *keys)


# ============================================================
#                    十、地理位置
# ============================================================

class GeoService:
    """
    地理位置服务
    """

    def __init__(self, client: redis.Redis, name: str):
        self.client = client
        self.key = f"geo:{name}"

    def add(self, name: str, longitude: float, latitude: float):
        """添加位置"""
        self.client.geoadd(self.key, (longitude, latitude, name))

    def get_position(self, name: str) -> Optional[tuple]:
        """获取位置"""
        result = self.client.geopos(self.key, name)
        return result[0] if result and result[0] else None

    def distance(self, name1: str, name2: str, unit: str = "km") -> Optional[float]:
        """计算距离"""
        return self.client.geodist(self.key, name1, name2, unit=unit)

    def search_nearby(
        self,
        longitude: float,
        latitude: float,
        radius: float,
        unit: str = "km",
        count: int = 10
    ) -> List:
        """搜索附近"""
        return self.client.georadius(
            self.key,
            longitude, latitude,
            radius, unit=unit,
            withdist=True,
            withcoord=True,
            count=count,
            sort="ASC"
        )


# ============================================================
#                    主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Redis 实战示例")
    print("=" * 60)

    client = get_redis_client()

    # 测试连接
    try:
        client.ping()
        print("Redis 连接成功！\n")
    except redis.ConnectionError:
        print("Redis 连接失败，请确保 Redis 服务已启动")
        exit(1)

    # 示例：排行榜
    print("--- 排行榜示例 ---")
    lb = Leaderboard(client, "demo:game")
    lb.add_score("player:A", 1000)
    lb.add_score("player:B", 1200)
    lb.add_score("player:C", 800)
    lb.incr_score("player:A", 100)

    print(f"Top 3: {lb.get_top(3)}")
    print(f"Player A 排名: {lb.get_rank('player:A')}")

    # 示例：限流
    print("\n--- 限流示例 ---")
    limiter = RateLimiter(client, "demo:api")
    for i in range(5):
        allowed = limiter.sliding_window(limit=3, window=10)
        print(f"请求 {i+1}: {'允许' if allowed else '限流'}")

    # 示例：缓存
    print("\n--- 缓存示例 ---")
    cache = Cache(client, "demo")
    cache.set("user:1", {"name": "Alice", "age": 25})
    user = cache.get("user:1")
    print(f"缓存的用户: {user}")

    # 示例：计数器
    print("\n--- 计数器示例 ---")
    counter = Counter(client, "demo:views")
    counter.incr("article:1001", 1)
    counter.incr("article:1001", 1)
    print(f"文章浏览量: {counter.get('article:1001')}")

    # 清理测试数据
    for key in client.keys("demo:*"):
        client.delete(key)
    for key in client.keys("leaderboard:demo:*"):
        client.delete(key)
    for key in client.keys("cache:demo:*"):
        client.delete(key)
    for key in client.keys("ratelimit:*:demo:*"):
        client.delete(key)

    print("\n测试完成！")
