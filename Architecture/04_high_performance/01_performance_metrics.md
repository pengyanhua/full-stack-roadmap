# 性能指标与分析

## 一、核心性能指标

### 指标体系

```
┌─────────────────────────────────────────────────────────────────┐
│                      性能指标金字塔                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                          业务指标                                │
│                         /        \                              │
│                    转化率        成交额                          │
│                   /                  \                          │
│               用户体验指标              系统指标                  │
│              /          \            /        \                 │
│          页面加载     首屏时间     吞吐量      资源利用率          │
│             |            |          |            |              │
│        ┌────┴────┐  ┌────┴────┐ ┌───┴───┐  ┌────┴────┐        │
│        │  延迟   │  │ 响应时间 │ │  QPS  │  │  CPU    │        │
│        │ Latency │  │   RT    │ │  TPS  │  │ Memory  │        │
│        └─────────┘  └─────────┘ └───────┘  └─────────┘        │
│                                                                 │
│   核心公式:                                                      │
│   吞吐量 = 并发数 / 平均响应时间                                  │
│   QPS = 1000 / 平均响应时间(ms) × 并发数                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1. 延迟 (Latency)

```
┌─────────────────────────────────────────────────────────────────┐
│                       延迟分位数解析                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   请求延迟分布 (示例):                                           │
│                                                                 │
│   延迟(ms)                                                      │
│       │                                                         │
│   500 │                                              *          │
│       │                                          *   *          │
│   200 │                                      *  *    *          │
│       │                                  *  *   *    *          │
│   100 │                          * *  * *   *   *    *          │
│       │                    * * * * *  * *   *   *    *          │
│    50 │            * * * * * * * * *  * *   *   *    *          │
│       │        * * * * * * * * * * *  * *   *   *    *          │
│    20 │ * * * * * * * * * * * * * * * * * * * * * * * *         │
│       └────────────────────────────────────────────────▶ 请求   │
│              P50    P90   P95  P99  P99.9                       │
│              20ms   50ms  100ms 200ms 500ms                     │
│                                                                 │
│   为什么关注 P99 而不是平均值？                                   │
│   - 平均值掩盖了长尾延迟                                         │
│   - 1% 的慢请求影响用户体验                                      │
│   - 高并发下长尾效应更明显                                       │
│                                                                 │
│   计算公式 (近似):                                               │
│   P99 系统延迟 = 1 - (1 - P99单服务)^N                          │
│   10个服务串行，每个 P99=100ms → 系统 P99 ≈ 1000ms              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 吞吐量 (Throughput)

```go
// QPS (Queries Per Second) - 每秒查询数
// TPS (Transactions Per Second) - 每秒事务数
// RPS (Requests Per Second) - 每秒请求数

// 计算公式
type PerformanceMetrics struct {
    TotalRequests    int64         // 总请求数
    Duration         time.Duration // 测试时长
    SuccessRequests  int64         // 成功请求数
    TotalLatency     time.Duration // 总延迟
}

func (m *PerformanceMetrics) QPS() float64 {
    return float64(m.TotalRequests) / m.Duration.Seconds()
}

func (m *PerformanceMetrics) SuccessRate() float64 {
    return float64(m.SuccessRequests) / float64(m.TotalRequests) * 100
}

func (m *PerformanceMetrics) AvgLatency() time.Duration {
    return time.Duration(int64(m.TotalLatency) / m.TotalRequests)
}

// 理论最大 QPS
// QPS_max = 并发连接数 / 平均响应时间
// 例: 1000 并发，平均 10ms → QPS_max = 100,000
```

### 3. 资源利用率

```
┌─────────────────────────────────────────────────────────────────┐
│                     资源利用率指标                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   CPU 使用率                                                     │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │ 用户态 (user)   │ 系统态 (sys) │ IO等待 │ 空闲 (idle)    │ │
│   │     40%         │    10%       │  20%   │    30%         │ │
│   └──────────────────────────────────────────────────────────┘ │
│   健康指标: user + sys < 70%, iowait < 10%                      │
│                                                                 │
│   内存使用                                                       │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │ 已用内存 │ 缓存 (cache) │ 缓冲 (buffer) │ 可用内存        │ │
│   │   50%    │     20%      │      5%       │     25%        │ │
│   └──────────────────────────────────────────────────────────┘ │
│   健康指标: 可用内存 > 20%, 无频繁 swap                          │
│                                                                 │
│   磁盘 I/O                                                       │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │ IOPS: 10000     │ 吞吐量: 500 MB/s  │ 利用率: 60%        │ │
│   │ 读延迟: 0.5ms   │ 写延迟: 1ms       │ 队列深度: 4        │ │
│   └──────────────────────────────────────────────────────────┘ │
│   健康指标: 利用率 < 80%, 延迟 < 10ms                            │
│                                                                 │
│   网络                                                           │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │ 带宽利用率: 40%  │ 连接数: 50000  │ 丢包率: 0.01%        │ │
│   │ 入流量: 400Mbps  │ 出流量: 800Mbps │ TCP重传: 0.1%       │ │
│   └──────────────────────────────────────────────────────────┘ │
│   健康指标: 带宽 < 70%, 丢包 < 0.1%, 重传 < 1%                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、性能分析方法

### 1. USE 方法

```
┌─────────────────────────────────────────────────────────────────┐
│              USE (Utilization, Saturation, Errors)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   对每个资源检查:                                                │
│                                                                 │
│   ┌─────────────────┬───────────────────────────────────────┐  │
│   │     资源         │   利用率(U)   饱和度(S)   错误(E)     │  │
│   ├─────────────────┼───────────────────────────────────────┤  │
│   │ CPU             │   使用率      运行队列    硬件错误    │  │
│   │ 内存            │   使用率      Swap/OOM   ECC错误     │  │
│   │ 磁盘            │   利用率      等待队列    IO错误     │  │
│   │ 网络            │   带宽使用    积压队列    丢包/重传   │  │
│   │ 文件描述符      │   使用数量    等待打开    打开失败    │  │
│   │ 连接池          │   使用率      等待连接    连接错误    │  │
│   └─────────────────┴───────────────────────────────────────┘  │
│                                                                 │
│   分析流程:                                                      │
│   1. 利用率高 → 资源瓶颈，需要扩容或优化                         │
│   2. 饱和度高 → 请求排队，需要限流或扩容                         │
│   3. 错误率高 → 系统异常，需要排查修复                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 火焰图分析

```bash
# CPU 火焰图生成 (Linux perf)
perf record -F 99 -p <PID> -g -- sleep 30
perf script | stackcollapse-perf.pl | flamegraph.pl > cpu.svg

# Go pprof 火焰图
go tool pprof -http=:8080 http://localhost:6060/debug/pprof/profile?seconds=30

# 内存火焰图
go tool pprof -http=:8080 http://localhost:6060/debug/pprof/heap

# 阻塞分析
go tool pprof -http=:8080 http://localhost:6060/debug/pprof/block
```

```go
// 在 Go 应用中开启 pprof
import (
    "net/http"
    _ "net/http/pprof"
)

func main() {
    // 开启 pprof 端口
    go func() {
        http.ListenAndServe(":6060", nil)
    }()

    // 业务代码...
}
```

### 3. 链路分析

```
┌─────────────────────────────────────────────────────────────────┐
│                     请求链路时间分解                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   总耗时: 150ms                                                  │
│                                                                 │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │ DNS    │ TCP │ TLS │ 等待 │     服务端处理     │ 下载    │ │
│   │ 10ms   │ 5ms │ 20ms│ 15ms │       80ms        │  20ms  │ │
│   └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│   服务端处理分解 (80ms):                                         │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │ 框架解析 │ 鉴权  │ 业务逻辑 │ 数据库  │ 缓存   │ 序列化  │ │
│   │   5ms   │ 10ms │   15ms   │  30ms   │ 10ms  │  10ms   │ │
│   └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│   优化优先级: 数据库(30ms) > TLS(20ms) > 业务逻辑(15ms)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```go
// 链路追踪埋点
type SpanTimer struct {
    spans []Span
    start time.Time
}

type Span struct {
    Name     string
    Duration time.Duration
}

func (t *SpanTimer) Start(name string) func() {
    start := time.Now()
    return func() {
        t.spans = append(t.spans, Span{
            Name:     name,
            Duration: time.Since(start),
        })
    }
}

// 使用示例
func ProcessRequest(ctx context.Context) {
    timer := &SpanTimer{start: time.Now()}

    // 鉴权
    done := timer.Start("auth")
    authenticate(ctx)
    done()

    // 查询数据库
    done = timer.Start("db_query")
    queryDatabase(ctx)
    done()

    // 业务逻辑
    done = timer.Start("business_logic")
    processLogic(ctx)
    done()

    // 记录追踪信息
    for _, span := range timer.spans {
        log.Printf("%s: %v", span.Name, span.Duration)
    }
}
```

---

## 三、性能基准测试

### 1. 压测工具对比

| 工具 | 语言 | 特点 | 适用场景 |
|------|------|------|----------|
| wrk | C | 高性能、简单 | HTTP 接口压测 |
| hey | Go | 易用、跨平台 | 快速 HTTP 压测 |
| k6 | Go/JS | 脚本化、可视化 | 复杂场景压测 |
| JMeter | Java | 功能全面、GUI | 企业级压测 |
| Locust | Python | 分布式、编程式 | 大规模压测 |
| Gatling | Scala | 高性能、报告美观 | 持续集成压测 |

### 2. wrk 压测示例

```bash
# 基础压测
wrk -t4 -c100 -d30s http://localhost:8080/api/users

# 带 Lua 脚本的复杂压测
wrk -t4 -c100 -d30s -s post.lua http://localhost:8080/api/orders

# post.lua
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"user_id": 1, "product_id": 100}'

request = function()
    return wrk.format(nil, "/api/orders?id=" .. math.random(1, 10000))
end

response = function(status, headers, body)
    if status ~= 200 then
        io.write("Error: " .. status .. "\n")
    end
end
```

### 3. k6 压测示例

```javascript
// stress_test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
    stages: [
        { duration: '1m', target: 100 },  // 1分钟内增加到100用户
        { duration: '3m', target: 100 },  // 保持100用户3分钟
        { duration: '1m', target: 200 },  // 1分钟内增加到200用户
        { duration: '3m', target: 200 },  // 保持200用户3分钟
        { duration: '1m', target: 0 },    // 1分钟内降到0
    ],
    thresholds: {
        http_req_duration: ['p(95)<200'],  // 95%请求延迟<200ms
        http_req_failed: ['rate<0.01'],    // 错误率<1%
    },
};

export default function() {
    let res = http.get('http://localhost:8080/api/users');

    check(res, {
        'status is 200': (r) => r.status === 200,
        'response time < 200ms': (r) => r.timings.duration < 200,
    });

    sleep(1);
}
```

```bash
# 运行压测
k6 run stress_test.js

# 输出到 InfluxDB + Grafana 可视化
k6 run --out influxdb=http://localhost:8086/k6 stress_test.js
```

### 4. Go 基准测试

```go
// benchmark_test.go
package main

import (
    "testing"
)

// 基准测试
func BenchmarkSerialize(b *testing.B) {
    data := createTestData()
    b.ResetTimer()

    for i := 0; i < b.N; i++ {
        _ = serialize(data)
    }
}

// 并行基准测试
func BenchmarkSerializeParallel(b *testing.B) {
    data := createTestData()
    b.ResetTimer()

    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            _ = serialize(data)
        }
    })
}

// 不同输入大小的基准测试
func BenchmarkSerializeSize(b *testing.B) {
    sizes := []int{100, 1000, 10000}

    for _, size := range sizes {
        b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
            data := createDataWithSize(size)
            b.ResetTimer()

            for i := 0; i < b.N; i++ {
                _ = serialize(data)
            }
        })
    }
}

// 内存分配测试
func BenchmarkAllocations(b *testing.B) {
    b.ReportAllocs()

    for i := 0; i < b.N; i++ {
        _ = processWithAllocations()
    }
}
```

```bash
# 运行基准测试
go test -bench=. -benchmem

# 输出示例:
# BenchmarkSerialize-8         1000000    1050 ns/op    256 B/op    4 allocs/op
# BenchmarkSerializeParallel-8 4000000     312 ns/op    256 B/op    4 allocs/op

# 比较两次基准测试结果
go test -bench=. -count=10 > old.txt
# 修改代码后
go test -bench=. -count=10 > new.txt
benchstat old.txt new.txt
```

---

## 四、性能监控体系

### 1. 应用性能监控 (APM)

```go
// 自定义指标收集
import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
    // 请求延迟直方图
    httpRequestDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "http_request_duration_seconds",
            Help:    "HTTP request duration in seconds",
            Buckets: []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10},
        },
        []string{"method", "path", "status"},
    )

    // 请求计数器
    httpRequestsTotal = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
            Help: "Total number of HTTP requests",
        },
        []string{"method", "path", "status"},
    )

    // 活跃连接数
    activeConnections = prometheus.NewGauge(prometheus.GaugeOpts{
        Name: "active_connections",
        Help: "Number of active connections",
    })
)

func init() {
    prometheus.MustRegister(httpRequestDuration)
    prometheus.MustRegister(httpRequestsTotal)
    prometheus.MustRegister(activeConnections)
}

// 中间件
func MetricsMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()

        activeConnections.Inc()
        defer activeConnections.Dec()

        // 包装 ResponseWriter 以获取状态码
        wrapped := &responseWriter{ResponseWriter: w, status: 200}
        next.ServeHTTP(wrapped, r)

        duration := time.Since(start).Seconds()
        status := strconv.Itoa(wrapped.status)

        httpRequestDuration.WithLabelValues(r.Method, r.URL.Path, status).Observe(duration)
        httpRequestsTotal.WithLabelValues(r.Method, r.URL.Path, status).Inc()
    })
}
```

### 2. Grafana Dashboard 配置

```yaml
# 关键性能指标 Dashboard
panels:
  - title: "QPS"
    query: |
      sum(rate(http_requests_total[1m]))

  - title: "P99 延迟"
    query: |
      histogram_quantile(0.99,
        sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
      )

  - title: "错误率"
    query: |
      sum(rate(http_requests_total{status=~"5.."}[5m]))
      /
      sum(rate(http_requests_total[5m])) * 100

  - title: "慢请求"
    query: |
      sum(rate(http_request_duration_seconds_bucket{le="1"}[5m]))
      /
      sum(rate(http_request_duration_seconds_count[5m])) * 100

  - title: "资源使用"
    queries:
      - label: "CPU"
        query: process_cpu_seconds_total
      - label: "内存"
        query: process_resident_memory_bytes
      - label: "Goroutines"
        query: go_goroutines
```

### 3. 告警规则

```yaml
# prometheus-alerts.yml
groups:
  - name: performance
    rules:
      # 高延迟告警
      - alert: HighLatency
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
          ) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P99 延迟超过 1 秒"

      # QPS 下降告警
      - alert: QPSDrop
        expr: |
          sum(rate(http_requests_total[5m]))
          < sum(rate(http_requests_total[5m] offset 1h)) * 0.5
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "QPS 下降超过 50%"

      # 高错误率告警
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m]))
          / sum(rate(http_requests_total[5m])) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "错误率超过 1%"

      # 资源耗尽预警
      - alert: HighMemoryUsage
        expr: |
          process_resident_memory_bytes / 1024 / 1024 / 1024 > 4
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "内存使用超过 4GB"
```

---

## 五、性能问题诊断

### 常见性能问题清单

```
┌─────────────────────────────────────────────────────────────────┐
│                    常见性能问题诊断                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   问题现象          │    可能原因           │    诊断方法       │
│ ──────────────────────────────────────────────────────────────│
│   CPU 使用率高       │ 计算密集/锁竞争       │ CPU 火焰图        │
│   内存使用率高       │ 内存泄漏/缓存过大     │ Heap Profile     │
│   响应时间长        │ I/O阻塞/依赖慢        │ Trace/链路追踪    │
│   吞吐量上不去      │ 连接池满/GC频繁       │ GC日志/连接池指标 │
│   偶发超时          │ GC Stop-the-world     │ GC Pause 监控    │
│   连接数暴增        │ 连接泄漏/慢查询       │ netstat/连接追踪  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 诊断脚本

```bash
#!/bin/bash
# performance_diagnosis.sh

PID=$1

echo "=== CPU 使用情况 ==="
top -b -n 1 -p $PID

echo "=== 内存使用情况 ==="
pmap -x $PID | tail -1

echo "=== 线程数 ==="
ls /proc/$PID/task | wc -l

echo "=== 文件描述符 ==="
ls /proc/$PID/fd | wc -l
cat /proc/$PID/limits | grep "open files"

echo "=== 网络连接 ==="
ss -tnp | grep $PID | wc -l
ss -tnp | grep $PID | awk '{print $4}' | cut -d: -f1 | sort | uniq -c | sort -rn | head

echo "=== 系统调用 (5秒采样) ==="
strace -c -p $PID -f -e trace=all 2>&1 &
sleep 5
kill %1 2>/dev/null

echo "=== GC 统计 (Go应用) ==="
curl -s http://localhost:6060/debug/pprof/heap?debug=1 | grep -A 20 "runtime.MemStats"
```

---

## 六、检查清单

### 性能指标检查

- [ ] 是否定义了明确的性能目标 (SLO)?
- [ ] 是否监控了 P99 延迟而不仅是平均延迟?
- [ ] 是否有吞吐量的历史趋势数据?
- [ ] 是否监控了所有关键资源的 USE 指标?

### 性能测试检查

- [ ] 是否进行了基准测试?
- [ ] 压测是否覆盖了峰值场景?
- [ ] 是否进行了持续性压测 (发现内存泄漏)?
- [ ] 压测环境是否接近生产环境?

### 监控告警检查

- [ ] 是否配置了延迟告警?
- [ ] 是否配置了错误率告警?
- [ ] 是否配置了资源告警?
- [ ] 告警阈值是否合理?
