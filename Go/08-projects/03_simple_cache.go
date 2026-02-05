package main

import (
	"fmt"
	"sync"
	"time"
)

// ============================================================
//                      小项目3：简单缓存
// ============================================================
// 功能：内存缓存，支持过期时间
// 知识点：map、互斥锁、goroutine、time

// CacheItem 缓存项
type CacheItem struct {
	Value      interface{}
	Expiration int64 // Unix 时间戳，0 表示永不过期
}

// IsExpired 是否已过期
func (item CacheItem) IsExpired() bool {
	if item.Expiration == 0 {
		return false
	}
	return time.Now().UnixNano() > item.Expiration
}

// Cache 缓存
type Cache struct {
	items map[string]CacheItem
	mu    sync.RWMutex
}

// NewCache 创建缓存
func NewCache() *Cache {
	c := &Cache{
		items: make(map[string]CacheItem),
	}

	// 启动清理 goroutine
	go c.cleanupLoop()

	return c
}

// Set 设置缓存（带过期时间）
func (c *Cache) Set(key string, value interface{}, ttl time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()

	var expiration int64
	if ttl > 0 {
		expiration = time.Now().Add(ttl).UnixNano()
	}

	c.items[key] = CacheItem{
		Value:      value,
		Expiration: expiration,
	}
}

// Get 获取缓存
func (c *Cache) Get(key string) (interface{}, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	item, found := c.items[key]
	if !found {
		return nil, false
	}

	if item.IsExpired() {
		return nil, false
	}

	return item.Value, true
}

// Delete 删除缓存
func (c *Cache) Delete(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	delete(c.items, key)
}

// Clear 清空缓存
func (c *Cache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.items = make(map[string]CacheItem)
}

// Count 缓存数量
func (c *Cache) Count() int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return len(c.items)
}

// Keys 所有键
func (c *Cache) Keys() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	keys := make([]string, 0, len(c.items))
	for k := range c.items {
		keys = append(keys, k)
	}
	return keys
}

// cleanupLoop 定期清理过期项
func (c *Cache) cleanupLoop() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		c.deleteExpired()
	}
}

// deleteExpired 删除过期项
func (c *Cache) deleteExpired() {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now().UnixNano()
	for k, v := range c.items {
		if v.Expiration > 0 && now > v.Expiration {
			delete(c.items, k)
		}
	}
}

// runSimpleCache 运行缓存示例
func runSimpleCache() {
	fmt.Println("\n=== 小项目3: 简单缓存 ===")

	cache := NewCache()

	// 设置缓存
	cache.Set("name", "张三", 0)                    // 永不过期
	cache.Set("token", "abc123", 2*time.Second)     // 2秒后过期
	cache.Set("data", map[string]int{"a": 1}, 0)

	fmt.Println("--- 设置缓存 ---")
	fmt.Println("缓存数量:", cache.Count())
	fmt.Println("所有键:", cache.Keys())

	// 获取缓存
	fmt.Println("\n--- 获取缓存 ---")
	if name, ok := cache.Get("name"); ok {
		fmt.Println("name:", name)
	}

	if token, ok := cache.Get("token"); ok {
		fmt.Println("token:", token)
	}

	if data, ok := cache.Get("data"); ok {
		fmt.Println("data:", data)
	}

	// 等待过期
	fmt.Println("\n--- 等待 3 秒... ---")
	time.Sleep(3 * time.Second)

	fmt.Println("--- 过期后 ---")
	if _, ok := cache.Get("name"); ok {
		fmt.Println("name: 仍存在")
	}

	if _, ok := cache.Get("token"); ok {
		fmt.Println("token: 仍存在")
	} else {
		fmt.Println("token: 已过期")
	}

	// 删除缓存
	cache.Delete("name")
	fmt.Println("\n删除 'name' 后，缓存数量:", cache.Count())

	// 清空缓存
	cache.Clear()
	fmt.Println("清空后，缓存数量:", cache.Count())
}

// ============================================================
//                      小项目3b：并发安全计数器
// ============================================================

// Counter 并发安全计数器
type Counter struct {
	mu    sync.Mutex
	value int
}

// Increment 增加
func (c *Counter) Increment() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.value++
}

// Decrement 减少
func (c *Counter) Decrement() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.value--
}

// Value 获取值
func (c *Counter) Value() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.value
}

func runConcurrentCounter() {
	fmt.Println("\n=== 并发安全计数器 ===")

	counter := &Counter{}
	var wg sync.WaitGroup

	// 启动 100 个 goroutine 增加计数
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				counter.Increment()
			}
		}()
	}

	wg.Wait()
	fmt.Println("100 个 goroutine 各增加 100 次，最终值:", counter.Value())
}
