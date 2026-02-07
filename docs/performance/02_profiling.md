# 性能分析：Profiling工具与实践

## 1. Profiling概述

### 1.1 什么是Profiling

性能分析(Profiling)是通过采样或插桩技术,收集程序运行时的性能数据,帮助定位性能瓶颈。

```
性能分析分类:

┌──────────────────────────────────────────┐
│  CPU Profiling                           │
│  - 方法调用耗时                           │
│  - 热点函数识别                           │
│  - 调用栈分析                             │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│  Memory Profiling                        │
│  - 堆内存分配                             │
│  - 内存泄漏检测                           │
│  - GC分析                                │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│  IO Profiling                            │
│  - 文件IO                                │
│  - 网络IO                                │
│  - 数据库查询                             │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│  Lock Profiling                          │
│  - 线程竞争                               │
│  - 锁等待时间                             │
│  - 死锁检测                               │
└──────────────────────────────────────────┘
```

### 1.2 Profiling方法论

```
分析流程:

1. 建立基准
   ↓
2. 运行压测
   ↓
3. 收集性能数据
   ↓
4. 分析热点
   ↓
5. 优化代码
   ↓
6. 验证效果
   ↓
7. 重复迭代
```

## 2. Go语言 - pprof

### 2.1 启用pprof

```go
// main.go
package main

import (
    "net/http"
    _ "net/http/pprof"  // 导入pprof
    "runtime"
)

func main() {
    // 设置CPU数量
    runtime.GOMAXPROCS(runtime.NumCPU())

    // 启动pprof HTTP服务
    go func() {
        http.ListenAndServe("localhost:6060", nil)
    }()

    // 你的应用代码
    startServer()
}
```

### 2.2 CPU Profiling

```go
// cpu_profiling.go
package main

import (
    "os"
    "runtime/pprof"
    "time"
)

func cpuProfilingExample() {
    // 创建CPU profile文件
    f, err := os.Create("cpu.prof")
    if err != nil {
        panic(err)
    }
    defer f.Close()

    // 开始CPU profiling
    pprof.StartCPUProfile(f)
    defer pprof.StopCPUProfile()

    // 运行你要分析的代码
    heavyComputation()
}

func heavyComputation() {
    // 模拟CPU密集型操作
    for i := 0; i < 10000000; i++ {
        _ = fibonacci(20)
    }
}

func fibonacci(n int) int {
    if n <= 1 {
        return n
    }
    return fibonacci(n-1) + fibonacci(n-2)
}

// 命令行分析
// go run cpu_profiling.go
// go tool pprof cpu.prof
// (pprof) top10
// (pprof) list heavyComputation
// (pprof) web  # 生成可视化图表
```

**pprof交互式命令:**

```bash
# 1. 启动pprof
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=30

# 进入交互模式后:

# 查看top函数
(pprof) top10
Showing nodes accounting for 8.50s, 85.00% of 10.00s total
      flat  flat%   sum%        cum   cum%
     3.20s 32.00% 32.00%      3.20s 32.00%  main.fibonacci
     2.10s 21.00% 53.00%      5.30s 53.00%  main.heavyComputation
     1.50s 15.00% 68.00%      1.50s 15.00%  runtime.mallocgc
     0.90s  9.00% 77.00%      0.90s  9.00%  runtime.scanobject
     0.80s  8.00% 85.00%      0.80s  8.00%  runtime.greyobject

# 查看函数详细信息
(pprof) list heavyComputation
Total: 10s
ROUTINE ======================== main.heavyComputation
     2.10s      5.30s (flat, cum) 53.00% of Total
         .          .      8:func heavyComputation() {
         .          .      9:    for i := 0; i < 10000000; i++ {
     2.10s      5.30s     10:        _ = fibonacci(20)
         .          .     11:    }
         .          .     12:}

# 生成火焰图
(pprof) web

# 生成PDF报告
(pprof) pdf > cpu_profile.pdf

# 查看调用栈
(pprof) traces fibonacci

# 按累积时间排序
(pprof) top -cum
```

### 2.3 Memory Profiling

```go
// memory_profiling.go
package main

import (
    "os"
    "runtime"
    "runtime/pprof"
)

func memoryProfilingExample() {
    // 模拟内存分配
    allocateMemory()

    // 触发GC,获取准确的堆快照
    runtime.GC()

    // 创建heap profile文件
    f, err := os.Create("mem.prof")
    if err != nil {
        panic(err)
    }
    defer f.Close()

    // 写入heap profile
    pprof.WriteHeapProfile(f)
}

func allocateMemory() {
    // 模拟内存泄漏
    var leakySlice [][]byte
    for i := 0; i < 1000; i++ {
        data := make([]byte, 1024*1024) // 每次分配1MB
        leakySlice = append(leakySlice, data)
    }
}

// 分析内存
// go run memory_profiling.go
// go tool pprof mem.prof
// (pprof) top
// (pprof) list allocateMemory
```

**实时内存分析:**

```bash
# 查看heap分配
go tool pprof http://localhost:6060/debug/pprof/heap

# 进入交互模式
(pprof) top
Showing nodes accounting for 1024MB, 95.00% of 1078MB total
      flat  flat%   sum%        cum   cum%
   512.00MB 47.50% 47.50%   512.00MB 47.50%  main.allocateMemory
   256.00MB 23.75% 71.25%   256.00MB 23.75%  runtime.allocm
   128.00MB 11.88% 83.13%   128.00MB 11.88%  net/http.(*conn).serve

# 查看内存分配详情
(pprof) list allocateMemory

# 查看对象数量(inuse_objects)
go tool pprof -inuse_objects http://localhost:6060/debug/pprof/heap

# 查看已分配但未释放的内存
go tool pprof -alloc_space http://localhost:6060/debug/pprof/heap
```

### 2.4 Goroutine Profiling

```go
// goroutine_profiling.go
package main

import (
    "os"
    "runtime/pprof"
    "sync"
    "time"
)

func goroutineProfilingExample() {
    var wg sync.WaitGroup

    // 创建大量goroutine
    for i := 0; i < 10000; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            time.Sleep(10 * time.Second)
        }(i)
    }

    // 保存goroutine profile
    f, _ := os.Create("goroutine.prof")
    defer f.Close()
    pprof.Lookup("goroutine").WriteTo(f, 0)

    wg.Wait()
}

// 查看goroutine堆栈
// go tool pprof goroutine.prof
// (pprof) top
// (pprof) traces
```

### 2.5 完整示例 - HTTP服务性能分析

```go
// server.go
package main

import (
    "encoding/json"
    "math/rand"
    "net/http"
    _ "net/http/pprof"
    "time"
)

type Response struct {
    Data   string `json:"data"`
    Status string `json:"status"`
}

// 模拟CPU密集型操作
func computeHandler(w http.ResponseWriter, r *http.Request) {
    result := heavyCompute()

    resp := Response{
        Data:   result,
        Status: "success",
    }

    json.NewEncoder(w).Encode(resp)
}

func heavyCompute() string {
    // 模拟复杂计算
    sum := 0
    for i := 0; i < 1000000; i++ {
        sum += rand.Intn(100)
    }
    return "computed"
}

// 模拟内存分配
func allocHandler(w http.ResponseWriter, r *http.Request) {
    // 分配大量内存
    data := make([]byte, 1024*1024*10) // 10MB
    for i := range data {
        data[i] = byte(rand.Intn(256))
    }

    w.Write([]byte("allocated"))
}

// 模拟阻塞IO
func slowHandler(w http.ResponseWriter, r *http.Request) {
    time.Sleep(100 * time.Millisecond)
    w.Write([]byte("slow response"))
}

func main() {
    http.HandleFunc("/compute", computeHandler)
    http.HandleFunc("/alloc", allocHandler)
    http.HandleFunc("/slow", slowHandler)

    println("Server started at :8080")
    println("Pprof available at :6060")

    go http.ListenAndServe(":6060", nil)
    http.ListenAndServe(":8080", nil)
}
```

**性能分析步骤:**

```bash
# 1. 启动服务
go run server.go

# 2. 压测
wrk -t4 -c100 -d30s http://localhost:8080/compute

# 3. 同时采集CPU profile
curl http://localhost:6060/debug/pprof/profile?seconds=30 > cpu.prof

# 4. 分析
go tool pprof cpu.prof
(pprof) top
(pprof) web

# 5. 内存分析
curl http://localhost:6060/debug/pprof/heap > heap.prof
go tool pprof heap.prof
(pprof) top

# 6. 查看goroutine
curl http://localhost:6060/debug/pprof/goroutine > goroutine.prof
go tool pprof goroutine.prof
```

## 3. Python - cProfile与line_profiler

### 3.1 cProfile (标准库)

```python
# cpu_profiling.py
import cProfile
import pstats
import io
from pstats import SortKey

def fibonacci(n):
    """递归计算斐波那契数"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def heavy_computation():
    """CPU密集型函数"""
    result = []
    for i in range(100):
        result.append(fibonacci(20))
    return result

def main():
    # 使用cProfile
    pr = cProfile.Profile()
    pr.enable()

    heavy_computation()

    pr.disable()

    # 输出到文件
    pr.dump_stats('output.prof')

    # 打印结果
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(20)
    print(s.getvalue())

if __name__ == '__main__':
    main()

# 运行
# python cpu_profiling.py

# 使用命令行分析
# python -m cProfile -o output.prof cpu_profiling.py
# python -m pstats output.prof
# % sort cumtime
# % stats 10
```

**cProfile输出解读:**

```
         21892 function calls (122 primitive calls) in 0.028 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.028    0.028 cpu_profiling.py:1(<module>)
        1    0.000    0.000    0.028    0.028 cpu_profiling.py:11(main)
        1    0.000    0.000    0.028    0.028 cpu_profiling.py:6(heavy_computation)
  21891/1    0.028    0.000    0.028    0.028 cpu_profiling.py:1(fibonacci)

字段说明:
ncalls: 调用次数 (递归调用显示为 总调用/原始调用)
tottime: 函数内部总耗时 (不含子函数)
percall: tottime / ncalls
cumtime: 累计耗时 (含子函数)
percall: cumtime / primitive calls
```

### 3.2 line_profiler (逐行分析)

```python
# line_profiling.py
# 安装: pip install line_profiler

from line_profiler import LineProfiler

def slow_function():
    """需要优化的函数"""
    total = 0

    # 操作1: 列表推导
    squares = [x**2 for x in range(10000)]

    # 操作2: 循环求和
    for i in range(10000):
        total += i

    # 操作3: 字符串拼接
    text = ""
    for i in range(1000):
        text += str(i)

    return total

def main():
    lp = LineProfiler()
    lp.add_function(slow_function)
    lp_wrapper = lp(slow_function)
    lp_wrapper()
    lp.print_stats()

if __name__ == '__main__':
    main()

# 或使用装饰器
# @profile  # kernprof会识别这个装饰器
# def slow_function():
#     ...

# 运行:
# kernprof -l -v line_profiling.py
```

**line_profiler输出:**

```
Timer unit: 1e-06 s

Total time: 0.003451 s
File: line_profiling.py
Function: slow_function at line 4

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     4                                           def slow_function():
     5         1          1.0      1.0      0.0      total = 0
     6
     7         1        458.0    458.0     13.3      squares = [x**2 for x in range(10000)]
     8
     9     10001       1203.0      0.1     34.9      for i in range(10000):
    10     10000       1156.0      0.1     33.5          total += i
    11
    12         1          1.0      1.0      0.0      text = ""
    13      1001        401.0      0.4     11.6      for i in range(1000):
    14      1000        231.0      0.2      6.7          text += str(i)
    15
    16         1          0.0      0.0      0.0      return total
```

### 3.3 memory_profiler

```python
# memory_profiling.py
# 安装: pip install memory_profiler

from memory_profiler import profile

@profile
def memory_hog():
    """内存密集型函数"""
    # 创建大列表
    big_list = [i for i in range(1000000)]

    # 创建大字典
    big_dict = {i: i**2 for i in range(100000)}

    # 字符串拼接
    text = ""
    for i in range(10000):
        text += str(i) * 100

    return len(big_list) + len(big_dict)

if __name__ == '__main__':
    memory_hog()

# 运行:
# python -m memory_profiler memory_profiling.py

# 或使用mprof绘图:
# mprof run memory_profiling.py
# mprof plot
```

**memory_profiler输出:**

```
Line #    Mem usage    Increment  Occurences   Line Contents
============================================================
     4     38.7 MiB     38.7 MiB           1   @profile
     5                                         def memory_hog():
     6     46.4 MiB      7.7 MiB           1       big_list = [i for i in range(1000000)]
     7
     8     54.1 MiB      7.7 MiB           1       big_dict = {i: i**2 for i in range(100000)}
     9
    10     54.1 MiB      0.0 MiB           1       text = ""
    11     62.3 MiB      8.2 MiB       10001       for i in range(10000):
    12     62.3 MiB      0.0 MiB       10000           text += str(i) * 100
```

### 3.4 py-spy (采样profiler)

```bash
# 安装
pip install py-spy

# 对运行中的进程采样
py-spy top --pid 12345

# 生成火焰图
py-spy record -o profile.svg --pid 12345

# 记录30秒
py-spy record -o profile.svg --duration 30 -- python app.py

# 实时查看
py-spy top -- python app.py
```

## 4. Java - JProfiler与Async Profiler

### 4.1 JVM内置工具

```bash
# jps - 查看Java进程
jps -lvm

# jstat - 查看GC统计
jstat -gc <pid> 1000 10  # 每秒采样,共10次
jstat -gcutil <pid>      # GC利用率

# jmap - 查看堆内存
jmap -heap <pid>                    # 堆配置和使用情况
jmap -histo <pid> | head -20        # 对象直方图
jmap -dump:format=b,file=heap.bin <pid>  # dump堆快照

# jstack - 查看线程堆栈
jstack <pid>                        # 线程dump
jstack <pid> | grep "java.lang.Thread.State" | wc -l  # 统计线程数

# jcmd - 多功能诊断工具
jcmd <pid> VM.uptime               # 运行时长
jcmd <pid> GC.heap_info            # 堆信息
jcmd <pid> Thread.print            # 线程dump
jcmd <pid> VM.flags                # JVM参数
```

### 4.2 Async Profiler

```bash
# 下载
wget https://github.com/jvm-profiling-tools/async-profiler/releases/download/v2.9/async-profiler-2.9-linux-x64.tar.gz
tar -xzf async-profiler-2.9-linux-x64.tar.gz

# CPU profiling (60秒)
./profiler.sh -d 60 -f cpu.html <pid>

# 内存分配 profiling
./profiler.sh -d 60 -e alloc -f alloc.html <pid>

# 锁竞争 profiling
./profiler.sh -d 60 -e lock -f lock.html <pid>

# 生成火焰图
./profiler.sh -d 60 -o flamegraph -f cpu.svg <pid>
```

### 4.3 Java Flight Recorder (JFR)

```java
// 启用JFR (JVM参数)
// -XX:+UnlockCommercialFeatures
// -XX:+FlightRecorder
// -XX:StartFlightRecording=duration=60s,filename=recording.jfr

// 或使用jcmd启动
jcmd <pid> JFR.start duration=60s filename=recording.jfr

// 停止记录
jcmd <pid> JFR.stop name=1

// 使用JMC分析
jmc -open recording.jfr
```

**JFR事件类型:**

```
CPU Events:
- Method Profiling Samples
- Execution Samples

Memory Events:
- Object Allocation in TLAB
- Object Allocation outside TLAB
- Garbage Collections

Lock Events:
- Monitor Enter
- Monitor Wait
- Thread Park

IO Events:
- Socket Read/Write
- File Read/Write
```

### 4.4 VisualVM

```bash
# 启动VisualVM
jvisualvm

# 功能:
# 1. 监控 - CPU, 内存, 线程, 类
# 2. 采样器 - CPU和内存采样
# 3. Profiler - 精确profiling
# 4. 堆dump分析
# 5. 线程dump分析
# 6. 插件 - Visual GC等
```

## 5. 实战案例

### 5.1 案例1: Go服务CPU占用高

**问题描述:** 服务CPU使用率持续90%+

**分析步骤:**

```bash
# 1. 采集CPU profile
curl http://localhost:6060/debug/pprof/profile?seconds=30 > cpu.prof

# 2. 分析
go tool pprof cpu.prof
(pprof) top10
```

**发现问题:**

```
(pprof) top
      flat  flat%   sum%        cum   cum%
     8.50s 85.00% 85.00%      8.50s 85.00%  encoding/json.Marshal
     1.20s 12.00% 97.00%      9.70s 97.00%  api.serializeResponse
     0.30s  3.00% 100.00%     10.00s 100.00%  api.handleRequest
```

**优化方案:**

```go
// 优化前: 每次请求都序列化
func serializeResponse(data interface{}) []byte {
    bytes, _ := json.Marshal(data)
    return bytes
}

// 优化后: 使用对象池
var responsePool = sync.Pool{
    New: func() interface{} {
        return &bytes.Buffer{}
    },
}

func serializeResponse(data interface{}) []byte {
    buf := responsePool.Get().(*bytes.Buffer)
    defer func() {
        buf.Reset()
        responsePool.Put(buf)
    }()

    json.NewEncoder(buf).Encode(data)
    return buf.Bytes()
}

// 或使用更快的序列化库
import "github.com/bytedance/sonic"

func serializeResponse(data interface{}) []byte {
    bytes, _ := sonic.Marshal(data)
    return bytes
}
```

**效果:**

```
优化前: CPU 90%, QPS 1000
优化后: CPU 45%, QPS 2500 (提升2.5倍)
```

### 5.2 案例2: Python内存泄漏

**问题描述:** 服务运行几小时后内存持续增长,最终OOM

**分析步骤:**

```python
# 1. 使用memory_profiler
@profile
def process_data():
    global cache
    data = fetch_data()
    cache[data.id] = data  # 可疑点
    return process(data)

# 2. 使用tracemalloc
import tracemalloc

tracemalloc.start()

# ... 运行一段时间 ...

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)
```

**发现问题:**

```python
# 全局cache无限增长
cache = {}  # 永不清理!

def process_data():
    data = fetch_data()
    cache[data.id] = data
    return process(data)
```

**优化方案:**

```python
from functools import lru_cache
from cachetools import TTLCache

# 方案1: 使用LRU缓存(限制大小)
cache = TTLCache(maxsize=1000, ttl=3600)

# 方案2: 使用装饰器
@lru_cache(maxsize=128)
def fetch_data(data_id):
    return expensive_db_query(data_id)

# 方案3: 定期清理
import time
import threading

class ManagedCache:
    def __init__(self, ttl=3600):
        self.cache = {}
        self.ttl = ttl
        self.timestamps = {}
        self._start_cleanup_thread()

    def set(self, key, value):
        self.cache[key] = value
        self.timestamps[key] = time.time()

    def get(self, key):
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None

    def _cleanup(self):
        while True:
            time.sleep(60)
            current_time = time.time()
            expired_keys = [
                k for k, t in self.timestamps.items()
                if current_time - t > self.ttl
            ]
            for key in expired_keys:
                del self.cache[key]
                del self.timestamps[key]

    def _start_cleanup_thread(self):
        t = threading.Thread(target=self._cleanup, daemon=True)
        t.start()
```

## 6. Profiling最佳实践

### 6.1 Do's and Don'ts

```
✓ Do:
- 先建立性能基准
- 在生产环境或生产级压力下profiling
- 关注top 10热点
- 多次采样确认问题
- 优化后验证效果
- 记录优化过程

✗ Don't:
- 不要过早优化
- 不要在开发环境profiling
- 不要只看一次采样结果
- 不要优化不重要的代码
- 不要盲目优化
```

### 6.2 Profiling工具选择

```
语言选择:
- Go: pprof (内置, 零依赖)
- Python: cProfile + line_profiler + py-spy
- Java: Async Profiler + JFR + VisualVM
- Node.js: --prof + clinic.js
- Rust: perf + flamegraph

场景选择:
- 开发环境: VisualVM, PyCharm Profiler
- 生产环境: pprof (HTTP), py-spy (无侵入)
- CI/CD: 基准测试 + 性能回归检测
```

### 6.3 性能优化优先级

```
优化优先级(从高到低):

1. 算法优化
   O(n²) → O(n log n) → O(n)
   效果: 10x - 100x

2. 减少IO
   - 批量操作
   - 缓存
   - 异步IO
   效果: 5x - 20x

3. 并发优化
   - 线程池
   - 协程
   - 并行处理
   效果: 2x - 8x

4. 代码优化
   - 避免重复计算
   - 对象池
   - 内联函数
   效果: 1.2x - 3x
```

## 7. 总结

性能分析的关键:
1. **建立基准** - 优化前知道现状
2. **数据驱动** - 用profiling数据指导优化
3. **聚焦热点** - 80/20原则
4. **持续监控** - 防止性能回归
5. **文档记录** - 分享优化经验

推荐学习资源:
- Go: https://go.dev/blog/pprof
- Python: https://docs.python.org/3/library/profile.html
- Java: https://openjdk.org/projects/jmc/
- 通用: Brendan Gregg's Performance Tools
