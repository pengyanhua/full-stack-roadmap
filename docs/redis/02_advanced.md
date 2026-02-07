# advanced

::: info 文件信息
- 📄 原文件：`02_advanced.redis`
- 🔤 类型：Redis Commands
:::

## Redis 命令

```redis
-- ============================================================
--                    Redis 高级特性
-- ============================================================
-- 包含：事务、Lua脚本、发布订阅、管道、分布式锁等
-- ============================================================

-- ============================================================
--                    一、事务（Transaction）
-- ============================================================

-- Redis 事务通过 MULTI/EXEC 实现，保证原子性执行

-- 基本事务
MULTI                           -- 开始事务
SET account:A 1000
SET account:B 2000
DECRBY account:A 100
INCRBY account:B 100
EXEC                            -- 执行事务

-- 放弃事务
MULTI
SET key1 "value1"
SET key2 "value2"
DISCARD                         -- 取消事务

-- 监视键（乐观锁）
WATCH account:A                 -- 监视键
GET account:A                   -- 读取当前值
MULTI
DECRBY account:A 100
EXEC                            -- 如果 account:A 被其他客户端修改，返回 nil

-- 取消监视
UNWATCH

-- 注意事项：
-- 1. Redis 事务不支持回滚
-- 2. 如果命令语法错误，整个事务不执行
-- 3. 如果命令执行时出错，其他命令继续执行

-- 示例：语法错误（整个事务不执行）
MULTI
SET key1 "value1"
INCR key1 wrong_arg             -- 语法错误
SET key2 "value2"
EXEC                            -- 返回错误，所有命令都不执行

-- 示例：执行时错误（其他命令继续）
SET mystring "hello"
MULTI
INCR mystring                   -- 对字符串执行 INCR 会失败
SET key1 "value1"
EXEC                            -- INCR 失败，但 SET 成功

-- ============================================================
--                    二、Lua 脚本
-- ============================================================

-- Lua 脚本在 Redis 中原子执行，适合复杂的原子操作

-- 执行 Lua 脚本
-- EVAL script numkeys key [key ...] arg [arg ...]
EVAL "return 'Hello, Lua!'" 0

-- 访问键和参数
-- KEYS[n]: 第 n 个键（1-based）
-- ARGV[n]: 第 n 个参数
EVAL "return redis.call('GET', KEYS[1])" 1 mykey
EVAL "return redis.call('SET', KEYS[1], ARGV[1])" 1 mykey "myvalue"

-- 完整示例：原子自增并返回
EVAL "
local current = redis.call('GET', KEYS[1])
if not current then
    current = 0
end
local new = current + ARGV[1]
redis.call('SET', KEYS[1], new)
return new
" 1 counter 10

-- 加载脚本（返回 SHA1）
SCRIPT LOAD "return redis.call('GET', KEYS[1])"
-- 返回类似: "a42059b356c875f0717db19a51f6aaa9161e77a2"

-- 通过 SHA1 执行脚本
EVALSHA a42059b356c875f0717db19a51f6aaa9161e77a2 1 mykey

-- 检查脚本是否存在
SCRIPT EXISTS a42059b356c875f0717db19a51f6aaa9161e77a2

-- 清空脚本缓存
SCRIPT FLUSH

-- 终止正在执行的脚本
SCRIPT KILL

-- ============================================================
-- Lua 脚本实战示例
-- ============================================================

-- 示例1：分布式锁（带超时）
-- 加锁脚本
EVAL "
if redis.call('SETNX', KEYS[1], ARGV[1]) == 1 then
    redis.call('PEXPIRE', KEYS[1], ARGV[2])
    return 1
end
return 0
" 1 lock:order:1001 "client:uuid" 30000

-- 解锁脚本（只能解除自己的锁）
EVAL "
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
" 1 lock:order:1001 "client:uuid"

-- 示例2：限流（滑动窗口）
EVAL "
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

-- 移除窗口外的记录
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

-- 获取当前窗口内的请求数
local count = redis.call('ZCARD', key)

if count < limit then
    -- 添加当前请求
    redis.call('ZADD', key, now, now .. math.random())
    redis.call('PEXPIRE', key, window)
    return 1
end
return 0
" 1 ratelimit:user:1001 10 60000 1704067200000

-- 示例3：库存扣减
EVAL "
local stock = tonumber(redis.call('GET', KEYS[1]))
local quantity = tonumber(ARGV[1])

if stock == nil then
    return -1  -- 商品不存在
end

if stock < quantity then
    return 0   -- 库存不足
end

redis.call('DECRBY', KEYS[1], quantity)
return 1       -- 扣减成功
" 1 stock:product:2001 2

-- 示例4：抢红包
EVAL "
local remain_count = tonumber(redis.call('HGET', KEYS[1], 'remain_count'))
local remain_amount = tonumber(redis.call('HGET', KEYS[1], 'remain_amount'))
local user_id = ARGV[1]

-- 检查是否已抢过
if redis.call('SISMEMBER', KEYS[2], user_id) == 1 then
    return -1  -- 已抢过
end

if remain_count <= 0 then
    return 0   -- 红包已抢完
end

-- 计算红包金额（简单随机）
local amount
if remain_count == 1 then
    amount = remain_amount
else
    amount = math.random(1, remain_amount - remain_count + 1)
end

-- 更新剩余
redis.call('HSET', KEYS[1], 'remain_count', remain_count - 1)
redis.call('HSET', KEYS[1], 'remain_amount', remain_amount - amount)
redis.call('SADD', KEYS[2], user_id)

return amount
" 2 redpacket:1001 redpacket:1001:users user:2001

-- ============================================================
--                    三、发布订阅（Pub/Sub）
-- ============================================================

-- 订阅频道（在另一个客户端执行）
SUBSCRIBE channel1 channel2

-- 发布消息
PUBLISH channel1 "Hello, subscribers!"

-- 按模式订阅
PSUBSCRIBE news.*               -- 订阅所有 news.* 频道
PSUBSCRIBE user:*:notification

-- 取消订阅
UNSUBSCRIBE channel1
PUNSUBSCRIBE news.*

-- 查看订阅信息
PUBSUB CHANNELS                 -- 查看活跃频道
PUBSUB CHANNELS news.*          -- 按模式查看
PUBSUB NUMSUB channel1          -- 查看订阅者数量
PUBSUB NUMPAT                   -- 查看模式订阅数量

-- 注意事项：
-- 1. 消息不会持久化
-- 2. 没有订阅者时消息会丢失
-- 3. 客户端断开重连会丢失期间的消息

-- 应用场景：实时通知、聊天室、配置更新广播

-- ============================================================
--                    四、管道（Pipeline）
-- ============================================================

-- Pipeline 将多个命令一次性发送，减少网络往返
-- 需要在客户端实现，以下是伪代码示例

-- 不使用 Pipeline（每个命令单独往返）
-- SET key1 value1  -> 等待响应
-- SET key2 value2  -> 等待响应
-- SET key3 value3  -> 等待响应
-- 总共 3 次网络往返

-- 使用 Pipeline（批量发送）
-- pipeline.SET key1 value1
-- pipeline.SET key2 value2
-- pipeline.SET key3 value3
-- pipeline.execute()
-- 只需 1 次网络往返

-- Python 示例
/*
import redis
r = redis.Redis()

# 使用 Pipeline
pipe = r.pipeline()
pipe.set('key1', 'value1')
pipe.set('key2', 'value2')
pipe.set('key3', 'value3')
pipe.incr('counter')
results = pipe.execute()

# 事务 Pipeline
pipe = r.pipeline(transaction=True)
pipe.set('key1', 'value1')
pipe.incr('counter')
results = pipe.execute()
*/

-- ============================================================
--                    五、分布式锁
-- ============================================================

-- 方法1：SETNX + EXPIRE（不推荐，非原子）
SETNX lock:resource "owner"
EXPIRE lock:resource 30

-- 方法2：SET NX EX（推荐，原子操作）
SET lock:resource "client:uuid:12345" NX EX 30

-- 解锁（使用 Lua 保证原子性）
EVAL "
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
" 1 lock:resource "client:uuid:12345"

-- 方法3：Redlock 算法（多节点）
-- 1. 获取当前时间
-- 2. 依次向 N 个节点请求锁
-- 3. 如果大多数节点（N/2+1）获取成功，且耗时小于锁超时时间，则获取成功
-- 4. 否则释放所有节点的锁

-- 锁续期（看门狗机制）
-- 定期检查锁是否还在，如果在则续期
EVAL "
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('PEXPIRE', KEYS[1], ARGV[2])
end
return 0
" 1 lock:resource "client:uuid:12345" 30000

-- ============================================================
--                    六、缓存策略
-- ============================================================

-- 1. 缓存穿透（查询不存在的数据）
-- 解决方案：布隆过滤器 + 空值缓存
--
-- 布隆过滤器（Redis 4.0+ RedisBloom 模块）
-- BF.ADD bloom_filter item
-- BF.EXISTS bloom_filter item
--
-- 空值缓存
SET user:nonexistent "" EX 300    -- 缓存空值 5 分钟

-- 2. 缓存击穿（热点 key 过期）
-- 解决方案：互斥锁 + 永不过期

-- 互斥锁重建缓存
EVAL "
local value = redis.call('GET', KEYS[1])
if value then
    return value
end

-- 尝试获取锁
if redis.call('SET', KEYS[2], '1', 'NX', 'EX', 10) == 1 then
    return nil  -- 获取锁成功，通知客户端重建缓存
end

-- 获取锁失败，等待重试
return nil
" 2 cache:hot_data lock:cache:hot_data

-- 3. 缓存雪崩（大量 key 同时过期）
-- 解决方案：过期时间加随机数 + 多级缓存

-- 设置随机过期时间
-- base_ttl + random(0, 300)

-- ============================================================
--                    七、内存优化
-- ============================================================

-- 查看内存使用
MEMORY USAGE mykey              -- 查看键的内存占用
MEMORY STATS                    -- 内存统计信息
MEMORY DOCTOR                   -- 内存诊断

-- 1. 使用合适的数据结构
-- Hash 比 String 更省内存（存储对象时）
--
-- 不推荐：
-- SET user:1001:name "Alice"
-- SET user:1001:age "25"
-- SET user:1001:city "Beijing"
--
-- 推荐：
-- HSET user:1001 name "Alice" age "25" city "Beijing"

-- 2. 使用整数 ID 而非长字符串

-- 3. 压缩列表（ziplist）优化
-- Hash/List/ZSet 在元素少时使用压缩列表
-- 配置参数：
-- hash-max-ziplist-entries 512
-- hash-max-ziplist-value 64
-- list-max-ziplist-size -2
-- zset-max-ziplist-entries 128
-- zset-max-ziplist-value 64

-- 4. 使用 OBJECT ENCODING 查看编码方式
OBJECT ENCODING mykey

-- 5. 淘汰策略配置
-- maxmemory 4gb
-- maxmemory-policy allkeys-lru
--
-- 淘汰策略选项：
-- noeviction: 不淘汰，内存满时写入报错
-- allkeys-lru: 所有键中淘汰最近最少使用的
-- volatile-lru: 设置了过期时间的键中淘汰 LRU
-- allkeys-random: 所有键中随机淘汰
-- volatile-random: 设置了过期时间的键中随机淘汰
-- volatile-ttl: 设置了过期时间的键中淘汰 TTL 最小的
-- allkeys-lfu: 所有键中淘汰最不经常使用的（Redis 4.0+）
-- volatile-lfu: 设置了过期时间的键中淘汰 LFU

-- ============================================================
--                    八、慢查询分析
-- ============================================================

-- 配置慢查询
-- slowlog-log-slower-than 10000  -- 超过 10ms 记录
-- slowlog-max-len 128            -- 最多保存 128 条

-- 查看慢查询日志
SLOWLOG GET 10                  -- 获取最近 10 条
SLOWLOG LEN                     -- 慢查询数量
SLOWLOG RESET                   -- 清空慢查询日志

-- 慢查询日志格式：
-- 1) 日志 ID
-- 2) 发生时间戳
-- 3) 耗时（微秒）
-- 4) 命令及参数
-- 5) 客户端地址
-- 6) 客户端名称

-- ============================================================
--                    九、客户端管理
-- ============================================================

-- 查看客户端连接
CLIENT LIST

-- 设置客户端名称
CLIENT SETNAME myclient

-- 获取客户端名称
CLIENT GETNAME

-- 关闭客户端连接
CLIENT KILL ID 1234
CLIENT KILL ADDR 127.0.0.1:6379

-- 暂停客户端
CLIENT PAUSE 5000               -- 暂停 5 秒

-- 客户端输出缓冲区配置
-- client-output-buffer-limit normal 0 0 0
-- client-output-buffer-limit replica 256mb 64mb 60
-- client-output-buffer-limit pubsub 32mb 8mb 60

-- ============================================================
--                    十、调试命令
-- ============================================================

-- 模拟延迟（调试用）
DEBUG SLEEP 1                   -- 睡眠 1 秒

-- 查看对象信息
OBJECT ENCODING mykey           -- 编码方式
OBJECT REFCOUNT mykey           -- 引用计数
OBJECT IDLETIME mykey           -- 空闲时间
OBJECT FREQ mykey               -- 访问频率（LFU 策略）

-- 查看键的序列化长度
DEBUG DIGEST mykey

-- 触发 RDB 保存
BGSAVE

-- 触发 AOF 重写
BGREWRITEAOF

-- 查看服务器时间
TIME

```
