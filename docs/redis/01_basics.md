# basics

::: info 文件信息
- 📄 原文件：`01_basics.redis`
- 🔤 类型：Redis Commands
:::

## Redis 命令

```redis
-- ============================================================
--                    Redis 基础教程
-- ============================================================
-- Redis: Remote Dictionary Server
-- 高性能的键值存储数据库，支持多种数据结构
-- ============================================================

-- ============================================================
--                    一、连接与基本操作
-- ============================================================

-- 连接 Redis（命令行）
-- redis-cli -h localhost -p 6379 -a password

-- 选择数据库（默认 0-15，共 16 个库）
SELECT 0

-- 查看当前数据库的 key 数量
DBSIZE

-- 清空当前数据库
FLUSHDB

-- 清空所有数据库
FLUSHALL

-- 查看服务器信息
INFO
INFO memory
INFO replication

-- ============================================================
--                    二、Key 操作
-- ============================================================

-- 设置键值
SET name "Redis"

-- 获取键值
GET name

-- 检查键是否存在
EXISTS name

-- 删除键
DEL name

-- 批量删除（使用通配符需要借助 SCAN）
-- 不推荐使用 KEYS *（会阻塞）

-- 设置过期时间（秒）
SET session "abc123"
EXPIRE session 3600

-- 设置过期时间（毫秒）
PEXPIRE session 3600000

-- 设置键值同时设置过期时间
SET token "xyz789" EX 3600
-- 或使用毫秒
SET token "xyz789" PX 3600000

-- 在指定时间点过期（Unix 时间戳）
EXPIREAT session 1735689600

-- 查看剩余生存时间（秒）
TTL session

-- 查看剩余生存时间（毫秒）
PTTL session

-- 移除过期时间（持久化）
PERSIST session

-- 重命名键
RENAME name newname
-- 仅当新键不存在时重命名
RENAMENX name newname

-- 查看键的类型
TYPE name

-- 序列化键值
DUMP name

-- 反序列化并恢复
RESTORE newkey 0 "\x00\x05Redis\t\x00..."

-- 移动键到其他数据库
MOVE name 1

-- 查找键（生产环境避免使用 KEYS *）
KEYS user:*
KEYS *name*

-- 使用 SCAN 迭代查找（推荐）
SCAN 0 MATCH user:* COUNT 100

-- 随机返回一个键
RANDOMKEY

-- ============================================================
--                    三、String（字符串）
-- ============================================================

-- String 是最基本的数据类型，可以存储字符串、整数、浮点数

-- 基本操作
SET greeting "Hello, Redis!"
GET greeting

-- 仅当键不存在时设置（常用于分布式锁）
SETNX lock:order "locked"

-- 仅当键存在时设置
SET name "NewValue" XX

-- 设置并返回旧值
GETSET counter 0

-- 同时设置多个键值
MSET user:1:name "Alice" user:1:age "25" user:1:city "Beijing"

-- 同时获取多个键值
MGET user:1:name user:1:age user:1:city

-- 仅当所有键都不存在时设置（原子操作）
MSETNX key1 "value1" key2 "value2"

-- 追加字符串
APPEND greeting " Welcome!"

-- 获取字符串长度
STRLEN greeting

-- 获取子字符串（0-based，包含两端）
GETRANGE greeting 0 4

-- 覆盖部分字符串
SETRANGE greeting 7 "World"

-- ============================================================
--                    四、数值操作
-- ============================================================

-- 设置数值
SET counter 100

-- 自增 1
INCR counter

-- 自减 1
DECR counter

-- 增加指定整数
INCRBY counter 10

-- 减少指定整数
DECRBY counter 5

-- 增加浮点数
SET price 9.99
INCRBYFLOAT price 0.01

-- 应用场景：计数器、限流、分布式 ID 生成

-- 示例：文章阅读计数
SET article:1001:views 0
INCR article:1001:views

-- 示例：限流（每秒最多 10 次请求）
-- 使用 INCR + EXPIRE 实现滑动窗口

-- ============================================================
--                    五、位操作（Bitmap）
-- ============================================================

-- Bitmap 使用 String 存储，每个 bit 可以是 0 或 1

-- 设置位
SETBIT user:login:20240101 1001 1    -- 用户 1001 在 20240101 登录
SETBIT user:login:20240101 1002 1    -- 用户 1002 在 20240101 登录
SETBIT user:login:20240102 1001 1    -- 用户 1001 在 20240102 登录

-- 获取位
GETBIT user:login:20240101 1001      -- 返回 1
GETBIT user:login:20240101 1003      -- 返回 0

-- 统计位为 1 的数量
BITCOUNT user:login:20240101         -- 统计 20240101 登录用户数

-- 位运算
-- AND: 两天都登录的用户
BITOP AND both_days user:login:20240101 user:login:20240102
-- OR: 任一天登录的用户
BITOP OR any_day user:login:20240101 user:login:20240102
-- XOR: 只有一天登录的用户
BITOP XOR only_one user:login:20240101 user:login:20240102
-- NOT: 取反
BITOP NOT not_login user:login:20240101

-- 查找第一个为 0 或 1 的位
BITPOS user:login:20240101 1         -- 第一个登录用户的 ID
BITPOS user:login:20240101 0         -- 第一个未登录用户的 ID

-- 应用场景：用户签到、在线状态、特征标记

-- ============================================================
--                    六、List（列表）
-- ============================================================

-- List 是双向链表，支持两端操作，适合队列、栈等场景

-- 左侧插入（头部）
LPUSH tasks "task1" "task2" "task3"

-- 右侧插入（尾部）
RPUSH tasks "task4" "task5"

-- 在指定元素前/后插入
LINSERT tasks BEFORE "task3" "task2.5"
LINSERT tasks AFTER "task3" "task3.5"

-- 获取列表长度
LLEN tasks

-- 获取指定范围的元素（0-based，-1 表示最后一个）
LRANGE tasks 0 -1       -- 获取所有
LRANGE tasks 0 2        -- 获取前 3 个

-- 获取指定位置的元素
LINDEX tasks 0          -- 第一个
LINDEX tasks -1         -- 最后一个

-- 设置指定位置的值
LSET tasks 0 "new_task1"

-- 左侧弹出
LPOP tasks

-- 右侧弹出
RPOP tasks

-- 弹出多个元素（Redis 6.2+）
LPOP tasks 2
RPOP tasks 2

-- 阻塞弹出（常用于消息队列）
BLPOP tasks 10          -- 等待 10 秒，0 表示永久等待
BRPOP tasks 10

-- 从一个列表弹出并推入另一个列表
RPOPLPUSH source dest
-- 阻塞版本
BRPOPLPUSH source dest 10

-- 移动元素（Redis 6.2+）
LMOVE source dest LEFT RIGHT

-- 删除元素
-- count > 0: 从头开始删除 count 个
-- count < 0: 从尾开始删除 |count| 个
-- count = 0: 删除所有匹配的
LREM tasks 1 "task1"

-- 保留指定范围（裁剪）
LTRIM tasks 0 99        -- 只保留前 100 个

-- 应用场景：消息队列、最近浏览、时间线

-- 示例：最近浏览的商品（保留最近 10 个）
LPUSH user:1001:recent_view "product:2001"
LTRIM user:1001:recent_view 0 9

-- ============================================================
--                    七、Set（集合）
-- ============================================================

-- Set 是无序、不重复的字符串集合

-- 添加元素
SADD tags "redis" "database" "nosql" "cache"

-- 获取所有元素
SMEMBERS tags

-- 获取元素数量
SCARD tags

-- 检查元素是否存在
SISMEMBER tags "redis"

-- 随机获取元素
SRANDMEMBER tags           -- 随机获取 1 个
SRANDMEMBER tags 3         -- 随机获取 3 个（可能重复）
SRANDMEMBER tags -3        -- 随机获取 3 个（可能重复，负数允许重复）

-- 随机弹出元素
SPOP tags                  -- 随机弹出 1 个
SPOP tags 2                -- 随机弹出 2 个

-- 移除元素
SREM tags "cache"

-- 移动元素到另一个集合
SMOVE tags new_tags "redis"

-- 集合运算
SADD set1 "a" "b" "c" "d"
SADD set2 "c" "d" "e" "f"

-- 交集
SINTER set1 set2           -- 返回 "c" "d"

-- 并集
SUNION set1 set2           -- 返回 "a" "b" "c" "d" "e" "f"

-- 差集
SDIFF set1 set2            -- 返回 "a" "b"（在 set1 但不在 set2）

-- 将结果存储到新集合
SINTERSTORE result set1 set2
SUNIONSTORE result set1 set2
SDIFFSTORE result set1 set2

-- 迭代遍历
SSCAN tags 0 MATCH r* COUNT 100

-- 应用场景：标签系统、共同好友、抽奖

-- 示例：共同关注
SADD user:1001:following "user:1002" "user:1003" "user:1004"
SADD user:1005:following "user:1002" "user:1004" "user:1006"
SINTER user:1001:following user:1005:following    -- 共同关注

-- 示例：抽奖
SADD lottery:2024 "user:1001" "user:1002" "user:1003"
SRANDMEMBER lottery:2024 3     -- 抽取 3 名幸运用户
SPOP lottery:2024              -- 抽取并移除（不能重复中奖）

-- ============================================================
--                    八、Sorted Set（有序集合）
-- ============================================================

-- Sorted Set 是有序、不重复的集合，每个元素有一个分数（score）

-- 添加元素
ZADD leaderboard 100 "player:1001"
ZADD leaderboard 95 "player:1002" 88 "player:1003" 120 "player:1004"

-- 批量添加（带选项，Redis 3.0.2+）
-- NX: 只添加新元素
-- XX: 只更新已存在的元素
-- GT: 只在新分数大于当前分数时更新
-- LT: 只在新分数小于当前分数时更新
-- CH: 返回修改的元素数量（包括新增和更新）
ZADD leaderboard NX 50 "player:1005"
ZADD leaderboard XX GT 150 "player:1001"

-- 获取元素数量
ZCARD leaderboard

-- 获取指定分数范围的元素数量
ZCOUNT leaderboard 90 100

-- 获取元素的分数
ZSCORE leaderboard "player:1001"

-- 获取元素的排名（从 0 开始）
ZRANK leaderboard "player:1001"       -- 升序排名
ZREVRANK leaderboard "player:1001"    -- 降序排名

-- 增加元素的分数
ZINCRBY leaderboard 10 "player:1002"

-- 按排名范围获取（升序）
ZRANGE leaderboard 0 -1               -- 所有元素
ZRANGE leaderboard 0 2                -- 前 3 名（分数最低的）
ZRANGE leaderboard 0 2 WITHSCORES     -- 包含分数

-- 按排名范围获取（降序）
ZREVRANGE leaderboard 0 2             -- 前 3 名（分数最高的）
ZREVRANGE leaderboard 0 2 WITHSCORES

-- 按分数范围获取
ZRANGEBYSCORE leaderboard 80 100
ZRANGEBYSCORE leaderboard 80 100 WITHSCORES
ZRANGEBYSCORE leaderboard 80 100 LIMIT 0 10    -- 分页

-- 按分数范围获取（降序）
ZREVRANGEBYSCORE leaderboard 100 80

-- 特殊语法
ZRANGEBYSCORE leaderboard -inf +inf   -- 所有
ZRANGEBYSCORE leaderboard (80 100     -- 不包含 80
ZRANGEBYSCORE leaderboard 80 (100     -- 不包含 100

-- 删除元素
ZREM leaderboard "player:1003"

-- 按排名范围删除
ZREMRANGEBYRANK leaderboard 0 1       -- 删除排名 0-1 的元素

-- 按分数范围删除
ZREMRANGEBYSCORE leaderboard 0 60     -- 删除分数 0-60 的元素

-- 集合运算
ZADD zset1 1 "a" 2 "b" 3 "c"
ZADD zset2 2 "b" 3 "c" 4 "d"

-- 并集（默认分数相加）
ZUNIONSTORE result 2 zset1 zset2
-- 指定权重
ZUNIONSTORE result 2 zset1 zset2 WEIGHTS 1 2
-- 指定聚合方式（SUM/MIN/MAX）
ZUNIONSTORE result 2 zset1 zset2 AGGREGATE MAX

-- 交集
ZINTERSTORE result 2 zset1 zset2

-- 迭代遍历
ZSCAN leaderboard 0 MATCH player:* COUNT 100

-- 应用场景：排行榜、延迟队列、时间线

-- 示例：排行榜
ZADD game:leaderboard 1000 "player:1"
ZINCRBY game:leaderboard 50 "player:1"    -- 加分
ZREVRANGE game:leaderboard 0 9 WITHSCORES -- 前 10 名

-- 示例：延迟队列（使用时间戳作为分数）
ZADD delay:queue 1704067200 "task:1001"   -- 在指定时间执行
-- 获取到期的任务
ZRANGEBYSCORE delay:queue 0 1704067200

-- ============================================================
--                    九、Hash（哈希）
-- ============================================================

-- Hash 是字段-值对的集合，适合存储对象

-- 设置字段
HSET user:1001 name "Alice" age 25 city "Beijing"

-- 获取字段
HGET user:1001 name

-- 获取多个字段
HMGET user:1001 name age city

-- 获取所有字段和值
HGETALL user:1001

-- 获取所有字段名
HKEYS user:1001

-- 获取所有值
HVALS user:1001

-- 获取字段数量
HLEN user:1001

-- 检查字段是否存在
HEXISTS user:1001 email

-- 仅当字段不存在时设置
HSETNX user:1001 email "alice@example.com"

-- 删除字段
HDEL user:1001 city

-- 字段值自增
HINCRBY user:1001 age 1
HINCRBYFLOAT user:1001 balance 10.5

-- 获取字段值的长度
HSTRLEN user:1001 name

-- 迭代遍历
HSCAN user:1001 0 MATCH * COUNT 100

-- 应用场景：存储对象、购物车、配置信息

-- 示例：购物车
HSET cart:user:1001 "product:2001" 2
HSET cart:user:1001 "product:2002" 1
HINCRBY cart:user:1001 "product:2001" 1   -- 增加数量
HDEL cart:user:1001 "product:2002"        -- 删除商品
HGETALL cart:user:1001                    -- 获取购物车

-- 示例：用户信息缓存
HSET user:1001 name "Alice" age 25 email "alice@example.com"
-- 对比 String 存储 JSON 的优势：
-- 1. 可以单独获取/修改某个字段
-- 2. 无需序列化/反序列化整个对象

-- ============================================================
--                    十、HyperLogLog
-- ============================================================

-- HyperLogLog 用于基数统计（去重计数），误差率约 0.81%
-- 优势：无论元素多少，只占用 12KB 内存

-- 添加元素
PFADD visitors:20240101 "user:1001" "user:1002" "user:1003"
PFADD visitors:20240101 "user:1001" "user:1004"    -- 重复的不会计数

-- 获取基数（去重后的数量）
PFCOUNT visitors:20240101

-- 合并多个 HyperLogLog
PFMERGE visitors:total visitors:20240101 visitors:20240102

-- 应用场景：UV 统计、独立访客数

-- 示例：网站 UV 统计
PFADD page:home:uv "192.168.1.1" "192.168.1.2"
PFCOUNT page:home:uv

-- ============================================================
--                    十一、Geo（地理位置）
-- ============================================================

-- Geo 用于存储地理位置信息，底层使用 Sorted Set

-- 添加位置（经度 纬度 名称）
GEOADD locations 116.404 39.915 "beijing"
GEOADD locations 121.473 31.230 "shanghai"
GEOADD locations 113.264 23.129 "guangzhou"
GEOADD locations 114.057 22.543 "shenzhen"

-- 获取位置坐标
GEOPOS locations "beijing" "shanghai"

-- 计算两点距离
GEODIST locations "beijing" "shanghai" km    -- 千米
GEODIST locations "beijing" "shanghai" m     -- 米

-- 获取位置的 geohash
GEOHASH locations "beijing"

-- 搜索指定范围内的位置
-- 以坐标为中心
GEORADIUS locations 116.404 39.915 500 km
GEORADIUS locations 116.404 39.915 500 km WITHCOORD WITHDIST COUNT 10 ASC

-- 以成员为中心
GEORADIUSBYMEMBER locations "beijing" 1500 km WITHDIST

-- Redis 6.2+ 新语法
GEOSEARCH locations FROMMEMBER "beijing" BYRADIUS 1500 km WITHDIST
GEOSEARCH locations FROMLONLAT 116.404 39.915 BYBOX 1000 1000 km WITHDIST

-- 应用场景：附近的人、附近的商家、配送范围

-- 示例：查找附近的商家
GEOADD shops 116.410 39.920 "shop:1001"
GEOADD shops 116.405 39.918 "shop:1002"
GEORADIUSBYMEMBER shops "shop:1001" 1 km WITHDIST

-- ============================================================
--                    十二、Stream（消息流）
-- ============================================================

-- Stream 是 Redis 5.0 引入的消息队列数据结构

-- 添加消息
XADD mystream * field1 value1 field2 value2
-- * 表示自动生成 ID，格式为 时间戳-序号

-- 指定 ID 添加
XADD mystream 1704067200000-0 field1 value1

-- 限制 Stream 长度
XADD mystream MAXLEN 1000 * field1 value1
XADD mystream MAXLEN ~ 1000 * field1 value1    -- 近似限制，性能更好

-- 获取消息数量
XLEN mystream

-- 读取消息（范围查询）
XRANGE mystream - +                            -- 所有消息
XRANGE mystream - + COUNT 10                   -- 最多 10 条
XRANGE mystream 1704067200000-0 +              -- 从指定 ID 开始

-- 反向读取
XREVRANGE mystream + - COUNT 10

-- 读取新消息（阻塞）
XREAD COUNT 10 BLOCK 5000 STREAMS mystream $   -- $ 表示最新消息
XREAD COUNT 10 BLOCK 5000 STREAMS mystream 0   -- 0 表示从头开始

-- 消费者组
-- 创建消费者组
XGROUP CREATE mystream mygroup $ MKSTREAM      -- $ 从最新开始
XGROUP CREATE mystream mygroup 0               -- 0 从头开始

-- 读取消息（消费者组）
XREADGROUP GROUP mygroup consumer1 COUNT 10 BLOCK 5000 STREAMS mystream >

-- 确认消息
XACK mystream mygroup 1704067200000-0

-- 查看待处理消息
XPENDING mystream mygroup

-- 转移待处理消息（消费者宕机时）
XCLAIM mystream mygroup consumer2 3600000 1704067200000-0

-- 删除消息
XDEL mystream 1704067200000-0

-- 裁剪 Stream
XTRIM mystream MAXLEN 1000

-- 查看 Stream 信息
XINFO STREAM mystream
XINFO GROUPS mystream
XINFO CONSUMERS mystream mygroup

-- 应用场景：消息队列、事件溯源、日志收集

```
