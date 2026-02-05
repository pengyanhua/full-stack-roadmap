import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.locks.*;
import java.time.*;

/**
 * ============================================================
 *                    简单缓存实现
 * ============================================================
 * 一个线程安全的 LRU 缓存实现，支持过期时间。
 *
 * 功能：
 * - 基于 LRU (Least Recently Used) 策略
 * - 支持最大容量限制
 * - 支持过期时间
 * - 线程安全
 * ============================================================
 */
public class SimpleCache {

    public static void main(String[] args) throws InterruptedException {
        basicCacheDemo();
        expirationDemo();
        concurrencyDemo();
        statisticsDemo();
    }

    /**
     * 基础缓存演示
     */
    public static void basicCacheDemo() {
        System.out.println("=".repeat(60));
        System.out.println("1. 基础缓存操作");
        System.out.println("=".repeat(60));

        Cache<String, String> cache = new Cache<>(3);

        // 添加数据
        cache.put("a", "Apple");
        cache.put("b", "Banana");
        cache.put("c", "Cherry");

        System.out.println("初始缓存:");
        cache.printAll();

        // 访问元素（更新 LRU 顺序）
        System.out.println("\n访问 'a': " + cache.get("a"));

        // 添加新元素（触发 LRU 淘汰）
        cache.put("d", "Date");
        System.out.println("\n添加 'd' 后（容量为3，最少使用的 'b' 被淘汰）:");
        cache.printAll();

        // 检查元素
        System.out.println("\n包含 'b': " + cache.contains("b"));  // false
        System.out.println("包含 'a': " + cache.contains("a"));    // true
    }

    /**
     * 过期时间演示
     */
    public static void expirationDemo() throws InterruptedException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. 过期时间");
        System.out.println("=".repeat(60));

        Cache<String, String> cache = new Cache<>(10, Duration.ofSeconds(1));

        cache.put("key1", "value1");
        System.out.println("添加 key1: " + cache.get("key1"));

        System.out.println("等待 500ms...");
        Thread.sleep(500);
        System.out.println("key1 仍有效: " + cache.get("key1"));

        System.out.println("等待 600ms...");
        Thread.sleep(600);
        System.out.println("key1 已过期: " + cache.get("key1"));  // null
    }

    /**
     * 并发演示
     */
    public static void concurrencyDemo() throws InterruptedException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("3. 并发访问");
        System.out.println("=".repeat(60));

        Cache<Integer, Integer> cache = new Cache<>(100);
        int threadCount = 10;
        int operationsPerThread = 1000;

        CountDownLatch latch = new CountDownLatch(threadCount);
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);

        long start = System.currentTimeMillis();

        for (int t = 0; t < threadCount; t++) {
            final int threadId = t;
            executor.submit(() -> {
                try {
                    for (int i = 0; i < operationsPerThread; i++) {
                        int key = threadId * 100 + (i % 50);
                        cache.put(key, i);
                        cache.get(key);
                    }
                } finally {
                    latch.countDown();
                }
            });
        }

        latch.await();
        executor.shutdown();

        long elapsed = System.currentTimeMillis() - start;
        int totalOps = threadCount * operationsPerThread * 2;

        System.out.println("并发测试完成:");
        System.out.println("  线程数: " + threadCount);
        System.out.println("  总操作数: " + totalOps);
        System.out.println("  耗时: " + elapsed + "ms");
        System.out.println("  吞吐量: " + (totalOps * 1000 / elapsed) + " ops/s");
        System.out.println("  缓存大小: " + cache.size());
    }

    /**
     * 统计信息演示
     */
    public static void statisticsDemo() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("4. 缓存统计");
        System.out.println("=".repeat(60));

        Cache<String, Integer> cache = new Cache<>(5);

        // 添加一些数据
        for (int i = 0; i < 10; i++) {
            cache.put("key" + i, i);
        }

        // 命中和未命中
        for (int i = 0; i < 20; i++) {
            cache.get("key" + (i % 10));
        }

        System.out.println("缓存统计:");
        System.out.println("  当前大小: " + cache.size());
        System.out.println("  命中次数: " + cache.getHitCount());
        System.out.println("  未命中次数: " + cache.getMissCount());
        System.out.println("  命中率: " + String.format("%.2f%%", cache.getHitRate() * 100));
        System.out.println("  淘汰次数: " + cache.getEvictionCount());
    }
}

/**
 * LRU 缓存实现
 */
class Cache<K, V> {

    private final int maxSize;
    private final Duration defaultTtl;
    private final Map<K, CacheEntry<V>> cache;
    private final ReadWriteLock lock = new ReentrantReadWriteLock();

    // 统计信息
    private long hitCount = 0;
    private long missCount = 0;
    private long evictionCount = 0;

    /**
     * 创建无过期时间的缓存
     */
    public Cache(int maxSize) {
        this(maxSize, null);
    }

    /**
     * 创建带默认过期时间的缓存
     */
    public Cache(int maxSize, Duration defaultTtl) {
        this.maxSize = maxSize;
        this.defaultTtl = defaultTtl;

        // 使用 LinkedHashMap 实现 LRU
        this.cache = new LinkedHashMap<>(maxSize, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<K, CacheEntry<V>> eldest) {
                if (size() > Cache.this.maxSize) {
                    evictionCount++;
                    return true;
                }
                return false;
            }
        };
    }

    /**
     * 存入缓存
     */
    public void put(K key, V value) {
        put(key, value, defaultTtl);
    }

    /**
     * 存入缓存（指定过期时间）
     */
    public void put(K key, V value, Duration ttl) {
        lock.writeLock().lock();
        try {
            Instant expiresAt = (ttl != null) ? Instant.now().plus(ttl) : null;
            cache.put(key, new CacheEntry<>(value, expiresAt));
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * 获取缓存值
     */
    public V get(K key) {
        lock.readLock().lock();
        try {
            CacheEntry<V> entry = cache.get(key);

            if (entry == null) {
                missCount++;
                return null;
            }

            if (entry.isExpired()) {
                // 需要升级到写锁来删除
                lock.readLock().unlock();
                lock.writeLock().lock();
                try {
                    // 双重检查
                    entry = cache.get(key);
                    if (entry != null && entry.isExpired()) {
                        cache.remove(key);
                        missCount++;
                        return null;
                    }
                } finally {
                    lock.readLock().lock();
                    lock.writeLock().unlock();
                }
            }

            hitCount++;
            return entry.value();
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * 检查键是否存在
     */
    public boolean contains(K key) {
        return get(key) != null;
    }

    /**
     * 删除缓存项
     */
    public void remove(K key) {
        lock.writeLock().lock();
        try {
            cache.remove(key);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * 清空缓存
     */
    public void clear() {
        lock.writeLock().lock();
        try {
            cache.clear();
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * 当前缓存大小
     */
    public int size() {
        lock.readLock().lock();
        try {
            return cache.size();
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * 打印所有缓存项
     */
    public void printAll() {
        lock.readLock().lock();
        try {
            for (Map.Entry<K, CacheEntry<V>> entry : cache.entrySet()) {
                System.out.println("  " + entry.getKey() + " -> " + entry.getValue().value());
            }
        } finally {
            lock.readLock().unlock();
        }
    }

    // 统计方法
    public long getHitCount() { return hitCount; }
    public long getMissCount() { return missCount; }
    public long getEvictionCount() { return evictionCount; }

    public double getHitRate() {
        long total = hitCount + missCount;
        return total == 0 ? 0 : (double) hitCount / total;
    }
}

/**
 * 缓存条目（带过期时间）
 */
record CacheEntry<V>(V value, Instant expiresAt) {
    boolean isExpired() {
        return expiresAt != null && Instant.now().isAfter(expiresAt);
    }
}
