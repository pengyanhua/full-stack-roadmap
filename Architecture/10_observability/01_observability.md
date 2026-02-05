# 可观测性

## 一、可观测性三支柱

### 概述

```
┌─────────────────────────────────────────────────────────────────┐
│                   可观测性三支柱                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                    Observability                         │  │
│   └─────────────────────────────────────────────────────────┘  │
│              │                │                │                │
│              ▼                ▼                ▼                │
│   ┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐     │
│   │     Metrics     │ │    Logs     │ │    Traces       │     │
│   │      指标       │ │    日志     │ │    链路追踪      │     │
│   ├─────────────────┤ ├─────────────┤ ├─────────────────┤     │
│   │ 聚合数据        │ │ 离散事件    │ │ 请求路径        │     │
│   │ 趋势分析        │ │ 详细上下文  │ │ 延迟分析        │     │
│   │ 告警触发        │ │ 问题排查    │ │ 依赖关系        │     │
│   ├─────────────────┤ ├─────────────┤ ├─────────────────┤     │
│   │ Prometheus      │ │ ELK Stack   │ │ Jaeger          │     │
│   │ Grafana         │ │ Loki        │ │ Zipkin          │     │
│   │                 │ │             │ │ SkyWalking      │     │
│   └─────────────────┘ └─────────────┘ └─────────────────┘     │
│                                                                 │
│   三者关系:                                                      │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                                                          │  │
│   │   告警 (Metrics) → 定位 (Logs) → 分析 (Traces)           │  │
│   │                                                          │  │
│   │   "指标告诉你有问题"                                      │  │
│   │   "日志告诉你发生了什么"                                  │  │
│   │   "链路告诉你问题在哪里"                                  │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、指标 (Metrics)

### 1. 指标类型

```
┌─────────────────────────────────────────────────────────────────┐
│                    Prometheus 指标类型                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Counter (计数器)                                              │
│   ─────────────────────────────────────────────────────────    │
│   只增不减，用于累计值                                          │
│   示例: 请求总数、错误总数                                      │
│   http_requests_total{method="GET", path="/api/users"} 1234    │
│                                                                 │
│   Gauge (仪表盘)                                                │
│   ─────────────────────────────────────────────────────────    │
│   可增可减，用于瞬时值                                          │
│   示例: 当前连接数、内存使用量                                  │
│   active_connections 42                                        │
│                                                                 │
│   Histogram (直方图)                                            │
│   ─────────────────────────────────────────────────────────    │
│   分布统计，自动计算分位数                                      │
│   示例: 请求延迟分布                                            │
│   http_request_duration_seconds_bucket{le="0.1"} 100           │
│   http_request_duration_seconds_bucket{le="0.5"} 150           │
│   http_request_duration_seconds_bucket{le="1"} 200             │
│                                                                 │
│   Summary (摘要)                                                │
│   ─────────────────────────────────────────────────────────    │
│   客户端计算分位数                                              │
│   http_request_duration_seconds{quantile="0.99"} 0.23          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Go 指标埋点

```go
import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
    // Counter
    httpRequestsTotal = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
            Help: "Total number of HTTP requests",
        },
        []string{"method", "path", "status"},
    )

    // Gauge
    activeConnections = prometheus.NewGauge(prometheus.GaugeOpts{
        Name: "active_connections",
        Help: "Number of active connections",
    })

    // Histogram
    httpRequestDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "http_request_duration_seconds",
            Help:    "HTTP request duration in seconds",
            Buckets: []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10},
        },
        []string{"method", "path"},
    )
)

func init() {
    prometheus.MustRegister(httpRequestsTotal)
    prometheus.MustRegister(activeConnections)
    prometheus.MustRegister(httpRequestDuration)
}

// 中间件
func MetricsMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        start := time.Now()

        activeConnections.Inc()
        defer activeConnections.Dec()

        c.Next()

        duration := time.Since(start).Seconds()
        status := strconv.Itoa(c.Writer.Status())

        httpRequestsTotal.WithLabelValues(c.Request.Method, c.FullPath(), status).Inc()
        httpRequestDuration.WithLabelValues(c.Request.Method, c.FullPath()).Observe(duration)
    }
}

// 暴露 metrics 端点
func main() {
    r := gin.New()
    r.Use(MetricsMiddleware())
    r.GET("/metrics", gin.WrapH(promhttp.Handler()))
    r.Run(":8080")
}
```

### 3. 关键业务指标

```yaml
# 四个黄金信号 (Google SRE)
# 1. Latency (延迟)
- record: http_request_latency_p99
  expr: histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))

# 2. Traffic (流量)
- record: http_requests_per_second
  expr: sum(rate(http_requests_total[5m]))

# 3. Errors (错误)
- record: http_error_rate
  expr: sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))

# 4. Saturation (饱和度)
- record: cpu_saturation
  expr: avg(1 - rate(node_cpu_seconds_total{mode="idle"}[5m]))
```

---

## 三、日志 (Logs)

### 1. 结构化日志

```go
import "go.uber.org/zap"

func InitLogger() *zap.Logger {
    config := zap.Config{
        Level:       zap.NewAtomicLevelAt(zap.InfoLevel),
        Development: false,
        Encoding:    "json",  // 结构化 JSON 格式
        EncoderConfig: zapcore.EncoderConfig{
            TimeKey:        "timestamp",
            LevelKey:       "level",
            NameKey:        "logger",
            CallerKey:      "caller",
            MessageKey:     "message",
            StacktraceKey:  "stacktrace",
            LineEnding:     zapcore.DefaultLineEnding,
            EncodeLevel:    zapcore.LowercaseLevelEncoder,
            EncodeTime:     zapcore.ISO8601TimeEncoder,
            EncodeDuration: zapcore.SecondsDurationEncoder,
            EncodeCaller:   zapcore.ShortCallerEncoder,
        },
        OutputPaths:      []string{"stdout"},
        ErrorOutputPaths: []string{"stderr"},
    }

    logger, _ := config.Build()
    return logger
}

// 使用
func main() {
    logger := InitLogger()
    defer logger.Sync()

    // 结构化日志
    logger.Info("user login",
        zap.String("user_id", "123"),
        zap.String("ip", "192.168.1.1"),
        zap.Duration("latency", 100*time.Millisecond),
    )

    // 输出:
    // {"timestamp":"2024-01-15T10:30:00Z","level":"info","message":"user login","user_id":"123","ip":"192.168.1.1","latency":0.1}
}

// 请求日志中间件
func LoggingMiddleware(logger *zap.Logger) gin.HandlerFunc {
    return func(c *gin.Context) {
        start := time.Now()
        requestID := c.GetHeader("X-Request-ID")
        if requestID == "" {
            requestID = uuid.New().String()
        }

        c.Set("request_id", requestID)
        c.Next()

        logger.Info("request",
            zap.String("request_id", requestID),
            zap.String("method", c.Request.Method),
            zap.String("path", c.Request.URL.Path),
            zap.Int("status", c.Writer.Status()),
            zap.Duration("latency", time.Since(start)),
            zap.String("client_ip", c.ClientIP()),
            zap.String("user_agent", c.Request.UserAgent()),
        )
    }
}
```

### 2. 日志级别规范

```
┌─────────────────────────────────────────────────────────────────┐
│                    日志级别使用规范                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   DEBUG - 调试信息                                               │
│   ─────────────────────────────────────────────────────────    │
│   • 开发调试用，生产环境关闭                                    │
│   • 函数入参、中间状态                                          │
│                                                                 │
│   INFO - 重要业务事件                                           │
│   ─────────────────────────────────────────────────────────    │
│   • 服务启动/关闭                                               │
│   • 用户登录/登出                                               │
│   • 订单创建/完成                                               │
│                                                                 │
│   WARN - 预期内的异常                                           │
│   ─────────────────────────────────────────────────────────    │
│   • 参数校验失败                                                │
│   • 重试成功                                                    │
│   • 性能下降预警                                                │
│                                                                 │
│   ERROR - 需要关注的错误                                        │
│   ─────────────────────────────────────────────────────────    │
│   • 业务处理失败                                                │
│   • 外部服务调用失败                                            │
│   • 数据库操作失败                                              │
│                                                                 │
│   FATAL - 导致服务不可用                                        │
│   ─────────────────────────────────────────────────────────    │
│   • 配置错误无法启动                                            │
│   • 关键依赖不可用                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 四、链路追踪 (Traces)

### OpenTelemetry 集成

```go
import (
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/exporters/jaeger"
    "go.opentelemetry.io/otel/sdk/resource"
    "go.opentelemetry.io/otel/sdk/trace"
    semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
)

func InitTracer(serviceName string) (*trace.TracerProvider, error) {
    // Jaeger exporter
    exporter, err := jaeger.New(jaeger.WithCollectorEndpoint(
        jaeger.WithEndpoint("http://jaeger:14268/api/traces"),
    ))
    if err != nil {
        return nil, err
    }

    // 创建 TracerProvider
    tp := trace.NewTracerProvider(
        trace.WithBatcher(exporter),
        trace.WithResource(resource.NewWithAttributes(
            semconv.SchemaURL,
            semconv.ServiceNameKey.String(serviceName),
            semconv.ServiceVersionKey.String("1.0.0"),
        )),
        trace.WithSampler(trace.AlwaysSample()),  // 采样策略
    )

    otel.SetTracerProvider(tp)
    return tp, nil
}

// HTTP 服务端中间件
func TracingMiddleware() gin.HandlerFunc {
    tracer := otel.Tracer("http-server")

    return func(c *gin.Context) {
        ctx, span := tracer.Start(c.Request.Context(), c.FullPath())
        defer span.End()

        // 设置属性
        span.SetAttributes(
            attribute.String("http.method", c.Request.Method),
            attribute.String("http.url", c.Request.URL.String()),
            attribute.String("http.user_agent", c.Request.UserAgent()),
        )

        c.Request = c.Request.WithContext(ctx)
        c.Next()

        // 记录响应状态
        span.SetAttributes(
            attribute.Int("http.status_code", c.Writer.Status()),
        )

        if c.Writer.Status() >= 400 {
            span.SetStatus(codes.Error, "request failed")
        }
    }
}

// HTTP 客户端
func CallExternalService(ctx context.Context, url string) error {
    tracer := otel.Tracer("http-client")
    ctx, span := tracer.Start(ctx, "call-external-service")
    defer span.End()

    req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)

    // 注入 trace context
    otel.GetTextMapPropagator().Inject(ctx, propagation.HeaderCarrier(req.Header))

    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        span.RecordError(err)
        span.SetStatus(codes.Error, err.Error())
        return err
    }
    defer resp.Body.Close()

    span.SetAttributes(
        attribute.Int("http.status_code", resp.StatusCode),
    )

    return nil
}
```

---

## 五、告警策略

### Prometheus 告警规则

```yaml
# alerting_rules.yml
groups:
  - name: availability
    rules:
      # 服务不可用
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "服务 {{ $labels.instance }} 不可用"

      # 高错误率
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)
          / sum(rate(http_requests_total[5m])) by (service)
          > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "{{ $labels.service }} 错误率超过 1%"

  - name: latency
    rules:
      # 高延迟
      - alert: HighLatency
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service)
          ) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "{{ $labels.service }} P99 延迟超过 1 秒"

  - name: resources
    rules:
      # CPU 使用率高
      - alert: HighCPUUsage
        expr: |
          100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "{{ $labels.instance }} CPU 使用率超过 80%"

      # 内存使用率高
      - alert: HighMemoryUsage
        expr: |
          (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes)
          / node_memory_MemTotal_bytes * 100 > 85
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "{{ $labels.instance }} 内存使用率超过 85%"
```

---

## 六、检查清单

### 指标检查

- [ ] 是否覆盖四个黄金信号？
- [ ] 是否有业务指标？
- [ ] 是否配置了告警规则？
- [ ] Dashboard 是否完善？

### 日志检查

- [ ] 是否使用结构化日志？
- [ ] 日志级别是否规范？
- [ ] 是否包含请求 ID？
- [ ] 日志是否有采集和检索？

### 链路追踪检查

- [ ] 是否接入了链路追踪？
- [ ] 跨服务调用是否传播 context？
- [ ] 采样策略是否合理？
- [ ] 是否能快速定位问题？
