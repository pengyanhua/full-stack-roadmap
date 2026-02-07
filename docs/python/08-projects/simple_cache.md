# simple cache.py

::: info 文件信息
- 📄 原文件：`03_simple_cache.py`
- 🔤 语言：python
:::

项目3：线程安全的 LRU 缓存
一个带过期时间的 LRU（最近最少使用）缓存实现。

功能：
- 基本的 get/set 操作
- LRU 淘汰策略
- 过期时间支持
- 线程安全
- 缓存统计

知识点：
- 数据结构（OrderedDict）
- 多线程和锁
- 装饰器
- 泛型
- 时间处理

## 完整代码

```python
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

K = TypeVar('K')
V = TypeVar('V')


@dataclass
class CacheEntry(Generic[V]):
    """缓存条目"""
    value: V
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    def is_expired(self, ttl: Optional[float]) -> bool:
        """检查是否过期"""
        if ttl is None:
            return False
        return time.time() - self.created_at > ttl

    def access(self) -> V:
        """访问条目"""
        self.last_accessed = time.time()
        self.access_count += 1
        return self.value


@dataclass
class CacheStats:
    """缓存统计"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def __str__(self) -> str:
        return (
            f"缓存统计:\n"
            f"  命中次数: {self.hits}\n"
            f"  未命中次数: {self.misses}\n"
            f"  命中率: {self.hit_rate:.2%}\n"
            f"  淘汰次数: {self.evictions}\n"
            f"  过期次数: {self.expirations}"
        )


class LRUCache(Generic[K, V]):
    """
    线程安全的 LRU 缓存

    Args:
        max_size: 最大容量
        ttl: 过期时间（秒），None 表示不过期
    """

    def __init__(self, max_size: int = 100, ttl: Optional[float] = None):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()

    def get(self, key: K, default: V = None) -> Optional[V]:
        """获取缓存值"""
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return default

            # 检查是否过期
            if entry.is_expired(self.ttl):
                self._delete(key)
                self._stats.expirations += 1
                self._stats.misses += 1
                return default

            # 移到末尾（最近访问）
            self._cache.move_to_end(key)
            self._stats.hits += 1
            return entry.access()

    def set(self, key: K, value: V) -> None:
        """设置缓存值"""
        with self._lock:
            if key in self._cache:
                # 更新现有条目
                self._cache[key] = CacheEntry(value)
                self._cache.move_to_end(key)
            else:
                # 添加新条目
                if len(self._cache) >= self.max_size:
                    self._evict()
                self._cache[key] = CacheEntry(value)

    def delete(self, key: K) -> bool:
        """删除缓存条目"""
        with self._lock:
            return self._delete(key)

    def _delete(self, key: K) -> bool:
        """内部删除方法（不加锁）"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def _evict(self) -> None:
        """淘汰最老的条目"""
        if self._cache:
            self._cache.popitem(last=False)
            self._stats.evictions += 1

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self) -> int:
        """清理所有过期条目"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired(self.ttl)
            ]
            for key in expired_keys:
                self._delete(key)
                self._stats.expirations += 1
            return len(expired_keys)

    @property
    def size(self) -> int:
        """当前缓存大小"""
        return len(self._cache)

    @property
    def stats(self) -> CacheStats:
        """获取统计信息"""
        return self._stats

    def items(self) -> Dict[K, V]:
        """获取所有缓存项"""
        with self._lock:
            return {k: v.value for k, v in self._cache.items()}

    def __contains__(self, key: K) -> bool:
        """检查键是否存在"""
        with self._lock:
            if key not in self._cache:
                return False
            entry = self._cache[key]
            if entry.is_expired(self.ttl):
                self._delete(key)
                return False
            return True

    def __len__(self) -> int:
        return self.size


def cached(max_size: int = 100, ttl: Optional[float] = None):
    """
    缓存装饰器

    Args:
        max_size: 最大缓存数量
        ttl: 过期时间（秒）

    Examples:
        @cached(max_size=100, ttl=60)
        def expensive_function(x, y):
            return x + y
    """
    cache = LRUCache(max_size=max_size, ttl=ttl)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 创建缓存键
            key = (args, tuple(sorted(kwargs.items())))

            result = cache.get(key)
            if result is not None:
                return result

            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = lambda: cache.stats

        return wrapper

    return decorator


def demo():
    """演示"""
    print("=" * 50)
    print("         LRU 缓存演示")
    print("=" * 50)

    # 基本使用
    print("\n【基本使用】")
    cache = LRUCache[str, int](max_size=3)

    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    print(f"缓存内容: {cache.items()}")

    # 访问会更新 LRU 顺序
    cache.get("a")
    cache.set("d", 4)  # 这会淘汰 "b"（最久未访问）
    print(f"添加 'd' 后: {cache.items()}")

    # 统计
    print(f"\n{cache.stats}")

    # 装饰器使用
    print("\n【装饰器使用】")

    @cached(max_size=10, ttl=5)
    def fibonacci(n: int) -> int:
        if n < 2:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    start = time.time()
    result = fibonacci(30)
    elapsed = time.time() - start
    print(f"fibonacci(30) = {result}，耗时: {elapsed:.4f}秒")

    start = time.time()
    result = fibonacci(30)
    elapsed = time.time() - start
    print(f"再次调用，耗时: {elapsed:.6f}秒（使用缓存）")

    print(f"\n缓存统计:")
    print(fibonacci.cache_stats())

    # 带过期时间的缓存
    print("\n【过期时间】")
    cache = LRUCache[str, str](max_size=10, ttl=2)

    cache.set("key", "value")
    print(f"设置后: {cache.get('key')}")

    time.sleep(2.1)
    print(f"2秒后: {cache.get('key')}")  # 过期，返回 None

    # 线程安全测试
    print("\n【线程安全测试】")
    cache = LRUCache[int, int](max_size=100)
    errors = []

    def worker(start: int, count: int):
        try:
            for i in range(start, start + count):
                cache.set(i, i * 2)
                cache.get(i)
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=worker, args=(i * 100, 100))
        for i in range(10)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"线程数: 10")
    print(f"操作数: 1000")
    print(f"错误数: {len(errors)}")
    print(f"缓存大小: {cache.size}")


def main():
    """交互模式"""
    print("=" * 50)
    print("      LRU 缓存交互演示")
    print("=" * 50)
    print("""
命令:
  set <key> <value>   设置缓存
  get <key>           获取缓存
  del <key>           删除缓存
  list                列出所有缓存
  stats               显示统计
  clear               清空缓存
  demo                运行演示
  quit                退出
""")

    cache = LRUCache[str, str](max_size=10, ttl=60)

    while True:
        try:
            user_input = input(">>> ").strip()
            if not user_input:
                continue

            parts = user_input.split()
            command = parts[0].lower()

            if command == "quit":
                break

            elif command == "demo":
                demo()

            elif command == "set":
                if len(parts) < 3:
                    print("用法: set <key> <value>")
                else:
                    key, value = parts[1], ' '.join(parts[2:])
                    cache.set(key, value)
                    print(f"已设置: {key} = {value}")

            elif command == "get":
                if len(parts) < 2:
                    print("用法: get <key>")
                else:
                    value = cache.get(parts[1])
                    if value is not None:
                        print(f"{parts[1]} = {value}")
                    else:
                        print(f"键 '{parts[1]}' 不存在或已过期")

            elif command == "del":
                if len(parts) < 2:
                    print("用法: del <key>")
                else:
                    if cache.delete(parts[1]):
                        print(f"已删除: {parts[1]}")
                    else:
                        print(f"键 '{parts[1]}' 不存在")

            elif command == "list":
                items = cache.items()
                if not items:
                    print("缓存为空")
                else:
                    print("缓存内容:")
                    for k, v in items.items():
                        print(f"  {k}: {v}")

            elif command == "stats":
                print(cache.stats)
                print(f"当前大小: {cache.size}/{cache.max_size}")

            elif command == "clear":
                cache.clear()
                print("缓存已清空")

            else:
                print(f"未知命令: {command}")

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    main()
```
