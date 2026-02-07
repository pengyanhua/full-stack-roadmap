# Apache Flink流处理实战

## 1. Flink架构与核心概念

### 1.1 Flink整体架构

```
Flink集群架构
┌─────────────────────────────────────────────────────────┐
│                   Client (提交作业)                      │
└─────────────────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  JobManager (Master)                    │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ ResourceManager│  │  Dispatcher  │  │ JobMaster    │ │
│  │               │  │              │  │              │ │
│  │ - 资源分配    │  │ - 接收作业   │  │ - 调度Task   │ │
│  │ - Slot管理    │  │ - 提交到JM   │  │ - Checkpoint │ │
│  └───────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            ↓             ↓             ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ TaskManager1 │  │ TaskManager2 │  │ TaskManager3 │
│ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │
│ │ Task Slot│ │  │ │ Task Slot│ │  │ │ Task Slot│ │
│ │  ┌────┐  │ │  │ │  ┌────┐  │ │  │ │  ┌────┐  │ │
│ │  │Task│  │ │  │ │  │Task│  │ │  │ │  │Task│  │ │
│ │  └────┘  │ │  │ │  └────┘  │ │  │ │  └────┘  │ │
│ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │
│ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │
│ │ Task Slot│ │  │ │ Task Slot│ │  │ │ Task Slot│ │
│ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │
└──────────────┘  └──────────────┘  └──────────────┘
```

### 1.2 核心概念

**DataStream（数据流）**：
```
数据流模型
Time ──────────────────────────────────────────────────────→
      │    │    │    │    │    │    │    │    │    │
      e1   e2   e3   e4   e5   e6   e7   e8   e9   e10

每个事件包含：
- Data: 业务数据
- Timestamp: 事件时间
- Watermark: 水位线（处理乱序）
```

**Watermark（水位线）**：
```
Watermark机制
Event Time: 1  3  2  5  4  7  6  9  8  10
            │  │  │  │  │  │  │  │  │  │
            ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
Watermark:  1  3  3  5  5  7  7  9  9  10
                 ↑        ↑        ↑
            乱序等待      乱序等待   乱序等待

Watermark(t) = MaxEventTime - AllowedLateness
```

**State（状态）**：
```
状态类型
├── Keyed State (键控状态)
│   ├── ValueState<T>        单个值
│   ├── ListState<T>         列表
│   ├── MapState<K,V>        Map结构
│   ├── ReducingState<T>     聚合值
│   └── AggregatingState<T>  聚合结果
│
└── Operator State (算子状态)
    ├── ListState<T>         均匀分布
    ├── UnionListState<T>    全量复制
    └── BroadcastState<K,V>  广播状态
```

### 1.3 时间语义

```
三种时间语义对比
┌──────────────────────────────────────────────────────┐
│ Event Time (事件时间)                                │
│ ┌────────────────────────────────────────────────┐  │
│ │ 数据产生的时间（最准确，支持乱序）              │  │
│ │ 示例：日志中的timestamp字段                    │  │
│ └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ Ingestion Time (摄入时间)                            │
│ ┌────────────────────────────────────────────────┐  │
│ │ 数据进入Flink的时间（折中方案）                │  │
│ └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ Processing Time (处理时间)                           │
│ ┌────────────────────────────────────────────────┐  │
│ │ 算子处理数据的机器时间（最简单，不支持乱序）   │  │
│ └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

## 2. DataStream API完整示例

### 2.1 基础流处理

```java
import org.apache.flink.api.common.eventtime.*;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer;

import java.time.Duration;
import java.util.Properties;

/**
 * Flink流处理基础示例
 */
public class BasicStreamProcessing {

    public static void main(String[] args) throws Exception {
        // 1. 创建执行环境
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // 设置并行度
        env.setParallelism(4);

        // 启用Checkpointing（每5秒一次）
        env.enableCheckpointing(5000);

        // 2. 配置Kafka Source
        Properties kafkaProps = new Properties();
        kafkaProps.setProperty("bootstrap.servers", "localhost:9092");
        kafkaProps.setProperty("group.id", "flink-consumer");

        FlinkKafkaConsumer<String> kafkaSource = new FlinkKafkaConsumer<>(
            "events",
            new SimpleStringSchema(),
            kafkaProps
        );

        // 设置从最早的offset开始消费
        kafkaSource.setStartFromEarliest();

        // 3. 读取数据流
        DataStream<String> rawStream = env.addSource(kafkaSource);

        // 4. 数据转换
        DataStream<Event> eventStream = rawStream
            .map(new MapFunction<String, Event>() {
                @Override
                public Event map(String value) throws Exception {
                    // 解析JSON
                    String[] parts = value.split(",");
                    return new Event(
                        parts[0],                           // userId
                        parts[1],                           // eventType
                        Long.parseLong(parts[2])            // timestamp
                    );
                }
            })
            // 使用Lambda简化
            // .map(value -> Event.fromJson(value))
            .assignTimestampsAndWatermarks(
                WatermarkStrategy
                    .<Event>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                    .withTimestampAssigner((event, timestamp) -> event.timestamp)
            );

        // 5. 业务处理
        DataStream<UserBehavior> result = eventStream
            .filter(event -> event.eventType.equals("click"))
            .keyBy(event -> event.userId)
            .map(event -> new UserBehavior(event.userId, 1));

        // 6. 输出结果
        result.print();

        // 7. 执行
        env.execute("Basic Stream Processing");
    }

    // 事件POJO
    public static class Event {
        public String userId;
        public String eventType;
        public long timestamp;

        public Event() {}

        public Event(String userId, String eventType, long timestamp) {
            this.userId = userId;
            this.eventType = eventType;
            this.timestamp = timestamp;
        }
    }

    // 结果POJO
    public static class UserBehavior {
        public String userId;
        public int count;

        public UserBehavior() {}

        public UserBehavior(String userId, int count) {
            this.userId = userId;
            this.count = count;
        }

        @Override
        public String toString() {
            return "UserBehavior{userId='" + userId + "', count=" + count + "}";
        }
    }
}
```

### 2.2 复杂事件处理（CEP）

```java
import org.apache.flink.cep.CEP;
import org.apache.flink.cep.PatternStream;
import org.apache.flink.cep.pattern.Pattern;
import org.apache.flink.cep.pattern.conditions.SimpleCondition;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.time.Time;

import java.util.List;
import java.util.Map;

/**
 * 复杂事件处理：检测连续3次登录失败
 */
public class LoginFailureDetection {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // 模拟登录事件流
        DataStream<LoginEvent> loginStream = env.fromElements(
            new LoginEvent("user1", "192.168.1.1", "fail", 1000L),
            new LoginEvent("user1", "192.168.1.1", "fail", 2000L),
            new LoginEvent("user1", "192.168.1.1", "fail", 3000L),
            new LoginEvent("user1", "192.168.1.1", "success", 4000L)
        ).assignTimestampsAndWatermarks(
            WatermarkStrategy
                .<LoginEvent>forMonotonousTimestamps()
                .withTimestampAssigner((event, timestamp) -> event.timestamp)
        );

        // 定义模式：2秒内连续3次失败
        Pattern<LoginEvent, ?> loginFailPattern = Pattern
            .<LoginEvent>begin("firstFail")
            .where(new SimpleCondition<LoginEvent>() {
                @Override
                public boolean filter(LoginEvent event) {
                    return "fail".equals(event.status);
                }
            })
            .next("secondFail")
            .where(new SimpleCondition<LoginEvent>() {
                @Override
                public boolean filter(LoginEvent event) {
                    return "fail".equals(event.status);
                }
            })
            .next("thirdFail")
            .where(new SimpleCondition<LoginEvent>() {
                @Override
                public boolean filter(LoginEvent event) {
                    return "fail".equals(event.status);
                }
            })
            .within(Time.seconds(2));

        // 应用模式
        PatternStream<LoginEvent> patternStream = CEP.pattern(
            loginStream.keyBy(event -> event.userId),
            loginFailPattern
        );

        // 提取匹配结果
        DataStream<Warning> warnings = patternStream.select(
            (Map<String, List<LoginEvent>> pattern) -> {
                LoginEvent first = pattern.get("firstFail").get(0);
                LoginEvent third = pattern.get("thirdFail").get(0);
                return new Warning(
                    first.userId,
                    first.ip,
                    "连续3次登录失败",
                    third.timestamp
                );
            }
        );

        warnings.print();

        env.execute("Login Failure Detection");
    }

    public static class LoginEvent {
        public String userId;
        public String ip;
        public String status;
        public long timestamp;

        public LoginEvent() {}

        public LoginEvent(String userId, String ip, String status, long timestamp) {
            this.userId = userId;
            this.ip = ip;
            this.status = status;
            this.timestamp = timestamp;
        }
    }

    public static class Warning {
        public String userId;
        public String ip;
        public String message;
        public long timestamp;

        public Warning() {}

        public Warning(String userId, String ip, String message, long timestamp) {
            this.userId = userId;
            this.ip = ip;
            this.message = message;
            this.timestamp = timestamp;
        }

        @Override
        public String toString() {
            return "Warning{" +
                "userId='" + userId + '\'' +
                ", ip='" + ip + '\'' +
                ", message='" + message + '\'' +
                ", timestamp=" + timestamp +
                '}';
        }
    }
}
```

## 3. 窗口函数详解

### 3.1 三种窗口类型

```java
import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.assigners.*;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;

/**
 * 三种窗口类型演示
 */
public class WindowTypes {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<SensorReading> sensorStream = env.addSource(new SensorSource());

        // 1. Tumbling Window (滚动窗口) - 无重叠
        // [0-5) [5-10) [10-15) ...
        DataStream<SensorReading> tumblingResult = sensorStream
            .keyBy(r -> r.sensorId)
            .window(TumblingEventTimeWindows.of(Time.seconds(5)))
            .reduce(new ReduceFunction<SensorReading>() {
                @Override
                public SensorReading reduce(SensorReading r1, SensorReading r2) {
                    return new SensorReading(
                        r1.sensorId,
                        Math.max(r1.timestamp, r2.timestamp),
                        Math.max(r1.temperature, r2.temperature)
                    );
                }
            });

        // 2. Sliding Window (滑动窗口) - 有重叠
        // [0-10) [5-15) [10-20) ...
        DataStream<SensorReading> slidingResult = sensorStream
            .keyBy(r -> r.sensorId)
            .window(SlidingEventTimeWindows.of(
                Time.seconds(10),  // 窗口大小
                Time.seconds(5)    // 滑动步长
            ))
            .reduce((r1, r2) -> new SensorReading(
                r1.sensorId,
                r2.timestamp,
                (r1.temperature + r2.temperature) / 2
            ));

        // 3. Session Window (会话窗口) - 基于间隔
        // 30秒内无活动则关闭窗口
        DataStream<SensorReading> sessionResult = sensorStream
            .keyBy(r -> r.sensorId)
            .window(EventTimeSessionWindows.withGap(Time.seconds(30)))
            .aggregate(new AverageAggregate());

        tumblingResult.print("Tumbling");
        slidingResult.print("Sliding");
        sessionResult.print("Session");

        env.execute("Window Types Demo");
    }

    // 自定义聚合函数
    public static class AverageAggregate
            implements AggregateFunction<SensorReading, Tuple2<Double, Integer>, Double> {

        @Override
        public Tuple2<Double, Integer> createAccumulator() {
            return new Tuple2<>(0.0, 0);
        }

        @Override
        public Tuple2<Double, Integer> add(SensorReading value, Tuple2<Double, Integer> acc) {
            return new Tuple2<>(acc.f0 + value.temperature, acc.f1 + 1);
        }

        @Override
        public Double getResult(Tuple2<Double, Integer> acc) {
            return acc.f0 / acc.f1;
        }

        @Override
        public Tuple2<Double, Integer> merge(Tuple2<Double, Integer> a, Tuple2<Double, Integer> b) {
            return new Tuple2<>(a.f0 + b.f0, a.f1 + b.f1);
        }
    }

    public static class SensorReading {
        public String sensorId;
        public long timestamp;
        public double temperature;

        public SensorReading() {}

        public SensorReading(String sensorId, long timestamp, double temperature) {
            this.sensorId = sensorId;
            this.timestamp = timestamp;
            this.temperature = temperature;
        }
    }

    // 模拟数据源
    public static class SensorSource implements SourceFunction<SensorReading> {
        private volatile boolean running = true;

        @Override
        public void run(SourceContext<SensorReading> ctx) throws Exception {
            // 实现省略
        }

        @Override
        public void cancel() {
            running = false;
        }
    }

    public static class Tuple2<T0, T1> {
        public T0 f0;
        public T1 f1;

        public Tuple2(T0 f0, T1 f1) {
            this.f0 = f0;
            this.f1 = f1;
        }
    }
}
```

### 3.2 窗口函数对比

```java
import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.streaming.api.functions.windowing.ProcessWindowFunction;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.util.Collector;

/**
 * 窗口函数性能对比
 */
public class WindowFunctions {

    // 1. ReduceFunction - 增量聚合（性能最好）
    // 每来一条数据就计算一次，内存占用小
    public static class MaxTempReduce implements ReduceFunction<SensorReading> {
        @Override
        public SensorReading reduce(SensorReading r1, SensorReading r2) {
            return r1.temperature > r2.temperature ? r1 : r2;
        }
    }

    // 2. AggregateFunction - 增量聚合（灵活）
    // 可以改变输出类型
    public static class AvgTempAggregate
            implements AggregateFunction<SensorReading, Tuple2<Double, Integer>, Double> {

        @Override
        public Tuple2<Double, Integer> createAccumulator() {
            return new Tuple2<>(0.0, 0);
        }

        @Override
        public Tuple2<Double, Integer> add(SensorReading value, Tuple2<Double, Integer> acc) {
            return new Tuple2<>(acc.f0 + value.temperature, acc.f1 + 1);
        }

        @Override
        public Double getResult(Tuple2<Double, Integer> acc) {
            return acc.f0 / acc.f1;
        }

        @Override
        public Tuple2<Double, Integer> merge(Tuple2<Double, Integer> a, Tuple2<Double, Integer> b) {
            return new Tuple2<>(a.f0 + b.f0, a.f1 + b.f1);
        }
    }

    // 3. ProcessWindowFunction - 全量聚合（功能最强）
    // 可以访问窗口元数据，但需要缓存所有数据
    public static class WindowTempProcess
            extends ProcessWindowFunction<SensorReading, String, String, TimeWindow> {

        @Override
        public void process(String key,
                          Context context,
                          Iterable<SensorReading> elements,
                          Collector<String> out) {

            // 可以访问窗口信息
            long windowStart = context.window().getStart();
            long windowEnd = context.window().getEnd();

            // 计算统计信息
            int count = 0;
            double sum = 0;
            double max = Double.MIN_VALUE;

            for (SensorReading r : elements) {
                count++;
                sum += r.temperature;
                max = Math.max(max, r.temperature);
            }

            out.collect(String.format(
                "传感器: %s, 窗口: [%d-%d), 平均温度: %.2f, 最高温度: %.2f, 数据量: %d",
                key, windowStart, windowEnd, sum/count, max, count
            ));
        }
    }

    // 4. 组合使用：增量聚合 + ProcessWindowFunction
    // 兼顾性能和功能
    public static void combinedWindowFunction(DataStream<SensorReading> stream) {
        stream
            .keyBy(r -> r.sensorId)
            .window(TumblingEventTimeWindows.of(Time.minutes(1)))
            .aggregate(
                new AvgTempAggregate(),
                new ProcessWindowFunction<Double, String, String, TimeWindow>() {
                    @Override
                    public void process(String key,
                                      Context context,
                                      Iterable<Double> elements,
                                      Collector<String> out) {
                        Double avgTemp = elements.iterator().next();
                        long windowEnd = context.window().getEnd();
                        out.collect(String.format(
                            "传感器: %s, 窗口结束: %d, 平均温度: %.2f",
                            key, windowEnd, avgTemp
                        ));
                    }
                }
            );
    }
}
```

## 4. Watermark与迟到数据

### 4.1 Watermark生成策略

```java
import org.apache.flink.api.common.eventtime.*;

/**
 * Watermark生成策略
 */
public class WatermarkStrategies {

    // 策略1：周期性Watermark（常用）
    public static class PeriodicWatermarkGenerator implements WatermarkGenerator<Event> {
        private long maxTimestamp = Long.MIN_VALUE;
        private final long maxOutOfOrderness = 5000; // 5秒

        @Override
        public void onEvent(Event event, long eventTimestamp, WatermarkOutput output) {
            maxTimestamp = Math.max(maxTimestamp, event.timestamp);
        }

        @Override
        public void onPeriodicEmit(WatermarkOutput output) {
            // 每200ms发射一次（默认配置）
            output.emitWatermark(new Watermark(maxTimestamp - maxOutOfOrderness));
        }
    }

    // 策略2：间断性Watermark
    public static class PunctuatedWatermarkGenerator implements WatermarkGenerator<Event> {
        @Override
        public void onEvent(Event event, long eventTimestamp, WatermarkOutput output) {
            // 遇到特殊标记时发射Watermark
            if (event.hasWatermarkMarker()) {
                output.emitWatermark(new Watermark(event.timestamp));
            }
        }

        @Override
        public void onPeriodicEmit(WatermarkOutput output) {
            // 不需要周期性发射
        }
    }

    // 内置策略1：单调递增（无乱序）
    public static WatermarkStrategy<Event> monotonicStrategy() {
        return WatermarkStrategy
            .<Event>forMonotonousTimestamps()
            .withTimestampAssigner((event, timestamp) -> event.timestamp);
    }

    // 内置策略2：固定延迟（有乱序）
    public static WatermarkStrategy<Event> boundedStrategy() {
        return WatermarkStrategy
            .<Event>forBoundedOutOfOrderness(Duration.ofSeconds(5))
            .withTimestampAssigner((event, timestamp) -> event.timestamp)
            .withIdleness(Duration.ofMinutes(1));  // 处理空闲数据源
    }

    // 自定义策略
    public static WatermarkStrategy<Event> customStrategy() {
        return WatermarkStrategy
            .forGenerator(ctx -> new PeriodicWatermarkGenerator())
            .withTimestampAssigner((event, timestamp) -> event.timestamp);
    }

    public static class Event {
        public long timestamp;

        public boolean hasWatermarkMarker() {
            // 实现逻辑
            return false;
        }
    }
}
```

### 4.2 处理迟到数据

```java
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.streaming.api.functions.windowing.ProcessWindowFunction;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;

/**
 * 迟到数据处理方案
 */
public class LateDataHandling {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // 定义侧输出流
        final OutputTag<Event> lateOutputTag = new OutputTag<Event>("late-data"){};

        DataStream<Event> stream = env.addSource(new EventSource())
            .assignTimestampsAndWatermarks(
                WatermarkStrategy
                    .<Event>forBoundedOutOfOrderness(Duration.ofSeconds(2))
                    .withTimestampAssigner((event, ts) -> event.timestamp)
            );

        // 方案1：设置允许迟到时间 + 侧输出流
        SingleOutputStreamOperator<String> result = stream
            .keyBy(e -> e.key)
            .window(TumblingEventTimeWindows.of(Time.seconds(10)))
            .allowedLateness(Time.seconds(5))  // 允许迟到5秒
            .sideOutputLateData(lateOutputTag)  // 超过5秒的发送到侧输出
            .process(new ProcessWindowFunction<Event, String, String, TimeWindow>() {
                @Override
                public void process(String key,
                                  Context ctx,
                                  Iterable<Event> elements,
                                  Collector<String> out) {
                    int count = 0;
                    for (Event e : elements) count++;
                    out.collect(String.format(
                        "窗口[%d-%d): key=%s, count=%d",
                        ctx.window().getStart(),
                        ctx.window().getEnd(),
                        key,
                        count
                    ));
                }
            });

        // 获取迟到数据
        DataStream<Event> lateStream = result.getSideOutput(lateOutputTag);

        // 处理迟到数据（例如：写入HBase进行修正）
        lateStream.addSink(new MyHBaseSink());

        result.print("正常数据");
        lateStream.print("迟到数据");

        env.execute("Late Data Handling");
    }

    // 方案2：使用Process Function手动管理状态
    public static class LateDataProcessor
            extends KeyedProcessFunction<String, Event, String> {

        private ValueState<Long> windowEndState;
        private ValueState<Integer> countState;

        @Override
        public void open(Configuration parameters) {
            windowEndState = getRuntimeContext().getState(
                new ValueStateDescriptor<>("window-end", Long.class)
            );
            countState = getRuntimeContext().getState(
                new ValueStateDescriptor<>("count", Integer.class)
            );
        }

        @Override
        public void processElement(Event event,
                                  Context ctx,
                                  Collector<String> out) throws Exception {

            long windowSize = 10000; // 10秒窗口
            long windowEnd = (event.timestamp / windowSize + 1) * windowSize;

            // 注册定时器
            ctx.timerService().registerEventTimeTimer(windowEnd);

            // 更新状态
            if (windowEndState.value() == null) {
                windowEndState.update(windowEnd);
                countState.update(1);
            } else {
                countState.update(countState.value() + 1);
            }
        }

        @Override
        public void onTimer(long timestamp,
                          OnTimerContext ctx,
                          Collector<String> out) throws Exception {

            // 窗口触发
            out.collect(String.format(
                "窗口结束: %d, count: %d",
                timestamp,
                countState.value()
            ));

            // 清理状态
            windowEndState.clear();
            countState.clear();
        }
    }

    public static class Event {
        public String key;
        public long timestamp;

        public Event() {}

        public Event(String key, long timestamp) {
            this.key = key;
            this.timestamp = timestamp;
        }
    }

    public static class EventSource implements SourceFunction<Event> {
        private volatile boolean running = true;

        @Override
        public void run(SourceContext<Event> ctx) throws Exception {
            // 实现省略
        }

        @Override
        public void cancel() {
            running = false;
        }
    }

    public static class MyHBaseSink implements SinkFunction<Event> {
        @Override
        public void invoke(Event value, Context context) {
            // 写入HBase逻辑
        }
    }
}
```

## 5. Checkpoint机制与状态后端

### 5.1 Checkpoint原理

```
Checkpoint机制（Chandy-Lamport算法）
┌──────────────────────────────────────────────────────┐
│  1. JobManager触发Checkpoint                         │
│     发送Barrier到Source                              │
└──────────────────────────────────────────────────────┘
                    │
                    ↓
┌──────────────────────────────────────────────────────┐
│  Source                                              │
│  ┌────────────────────────────────────────────────┐ │
│  │ 1. 接收Barrier                                 │ │
│  │ 2. 保存自身状态（offset等）                    │ │
│  │ 3. 广播Barrier到下游                           │ │
│  └────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
┌─────────────────┐  ┌─────────────────┐
│  Operator 1     │  │  Operator 2     │
│  等待所有上游   │  │  等待所有上游   │
│  Barrier对齐    │  │  Barrier对齐    │
│  ↓              │  │  ↓              │
│  保存状态       │  │  保存状态       │
│  ↓              │  │  ↓              │
│  广播Barrier    │  │  广播Barrier    │
└─────────────────┘  └─────────────────┘
          │                   │
          └─────────┬─────────┘
                    ↓
┌──────────────────────────────────────────────────────┐
│  Sink                                                │
│  1. 接收Barrier                                      │
│  2. 保存状态                                         │
│  3. 通知JobManager完成                               │
└──────────────────────────────────────────────────────┘
```

### 5.2 Checkpoint配置

```java
import org.apache.flink.contrib.streaming.state.RocksDBStateBackend;
import org.apache.flink.runtime.state.StateBackend;
import org.apache.flink.runtime.state.filesystem.FsStateBackend;
import org.apache.flink.runtime.state.memory.MemoryStateBackend;
import org.apache.flink.streaming.api.CheckpointingMode;
import org.apache.flink.streaming.api.environment.CheckpointConfig;

/**
 * Checkpoint完整配置
 */
public class CheckpointConfiguration {

    public static void configureCheckpoint(StreamExecutionEnvironment env) {
        // 1. 启用Checkpoint（间隔5秒）
        env.enableCheckpointing(5000);

        // 2. Checkpoint配置
        CheckpointConfig config = env.getCheckpointConfig();

        // Checkpoint模式
        config.setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);
        // config.setCheckpointingMode(CheckpointingMode.AT_LEAST_ONCE);

        // Checkpoint超时时间
        config.setCheckpointTimeout(60000);  // 60秒

        // 最小间隔（避免频繁checkpoint）
        config.setMinPauseBetweenCheckpoints(500);

        // 最大并发checkpoint数
        config.setMaxConcurrentCheckpoints(1);

        // Job取消后保留Checkpoint
        config.enableExternalizedCheckpoints(
            CheckpointConfig.ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION
        );

        // 容忍checkpoint失败次数
        config.setTolerableCheckpointFailureNumber(3);

        // 3. 状态后端配置
        // 选项1：MemoryStateBackend（开发测试用）
        // env.setStateBackend(new MemoryStateBackend());

        // 选项2：FsStateBackend（小状态生产环境）
        // env.setStateBackend(new FsStateBackend(
        //     "hdfs://namenode:9000/flink/checkpoints"
        // ));

        // 选项3：RocksDBStateBackend（大状态生产环境，推荐）
        env.setStateBackend(new RocksDBStateBackend(
            "hdfs://namenode:9000/flink/checkpoints",
            true  // 启用增量checkpoint
        ));

        // 4. 重启策略
        env.setRestartStrategy(RestartStrategies.fixedDelayRestart(
            3,      // 重启次数
            10000   // 重启间隔(ms)
        ));

        // 或使用失败率重启策略
        env.setRestartStrategy(RestartStrategies.failureRateRestart(
            3,                                    // 每个时间段内最大失败次数
            Time.of(5, TimeUnit.MINUTES),        // 时间段
            Time.of(10, TimeUnit.SECONDS)        // 重启延迟
        ));
    }
}
```

### 5.3 状态后端对比

```
三种状态后端对比
┌────────────────────────────────────────────────────────────┐
│ MemoryStateBackend                                         │
├────────────────────────────────────────────────────────────┤
│ 存储位置: JVM Heap                                         │
│ Checkpoint: JobManager内存                                 │
│ 优点: 快速                                                 │
│ 缺点: 受限于内存大小，Job失败丢失数据                     │
│ 适用: 开发测试、小状态                                     │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ FsStateBackend                                             │
├────────────────────────────────────────────────────────────┤
│ 存储位置: JVM Heap                                         │
│ Checkpoint: 文件系统(HDFS/S3)                              │
│ 优点: 可靠持久化                                           │
│ 缺点: 受限于内存，状态大小有限                             │
│ 适用: 中等状态(GB级别)                                     │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ RocksDBStateBackend (推荐)                                 │
├────────────────────────────────────────────────────────────┤
│ 存储位置: RocksDB(磁盘)                                    │
│ Checkpoint: 文件系统(HDFS/S3)                              │
│ 优点: 支持超大状态(TB级别)，增量checkpoint                 │
│ 缺点: 访问速度比内存慢                                     │
│ 适用: 大状态生产环境                                       │
│ 特性:                                                      │
│   - 增量checkpoint（只保存变更）                           │
│   - 异步snapshot（不阻塞处理）                             │
│   - 状态可超过内存                                         │
└────────────────────────────────────────────────────────────┘
```

## 6. 实战案例：实时用户行为分析

```java
import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.state.*;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.streaming.api.functions.source.SourceFunction;
import org.apache.flink.streaming.api.functions.windowing.ProcessWindowFunction;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.util.Collector;

import java.util.*;

/**
 * 实时用户行为分析系统
 * 功能：
 * 1. 实时计算PV、UV
 * 2. 热门商品Top-N
 * 3. 用户会话分析
 */
public class RealtimeUserBehaviorAnalysis {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(4);
        env.enableCheckpointing(10000);

        // 读取用户行为日志
        DataStream<UserBehavior> behaviorStream = env
            .addSource(new UserBehaviorSource())
            .assignTimestampsAndWatermarks(
                WatermarkStrategy
                    .<UserBehavior>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                    .withTimestampAssigner((event, ts) -> event.timestamp * 1000)
            );

        // 任务1：每小时PV、UV统计
        DataStream<PageViewStats> pvuvStream = behaviorStream
            .filter(behavior -> "pv".equals(behavior.behavior))
            .keyBy(b -> "all")  // 全局聚合
            .window(TumblingEventTimeWindows.of(Time.hours(1)))
            .aggregate(
                new PvUvAggregateFunction(),
                new PvUvWindowFunction()
            );

        pvuvStream.print("PV/UV");

        // 任务2：热门商品Top-N（每5分钟）
        DataStream<String> topNStream = behaviorStream
            .filter(behavior -> "pv".equals(behavior.behavior))
            .keyBy(UserBehavior::getItemId)
            .window(TumblingEventTimeWindows.of(Time.minutes(5)))
            .aggregate(
                new CountAggregateFunction(),
                new ItemViewWindowFunction()
            )
            .keyBy(ItemViewCount::getWindowEnd)
            .process(new TopNHotItems(5));

        topNStream.print("Top-N");

        // 任务3：用户会话分析（30分钟无活动则关闭会话）
        DataStream<UserSession> sessionStream = behaviorStream
            .keyBy(UserBehavior::getUserId)
            .process(new UserSessionProcessor());

        sessionStream.print("Session");

        env.execute("Realtime User Behavior Analysis");
    }

    // ============ 数据结构 ============

    public static class UserBehavior {
        public long userId;
        public long itemId;
        public String behavior;  // pv, cart, fav, buy
        public long timestamp;

        public UserBehavior() {}

        public UserBehavior(long userId, long itemId, String behavior, long timestamp) {
            this.userId = userId;
            this.itemId = itemId;
            this.behavior = behavior;
            this.timestamp = timestamp;
        }

        public long getUserId() { return userId; }
        public long getItemId() { return itemId; }
    }

    public static class PageViewStats {
        public long windowEnd;
        public long pv;
        public long uv;

        public PageViewStats(long windowEnd, long pv, long uv) {
            this.windowEnd = windowEnd;
            this.pv = pv;
            this.uv = uv;
        }

        @Override
        public String toString() {
            return String.format("窗口结束: %d, PV: %d, UV: %d", windowEnd, pv, uv);
        }
    }

    public static class ItemViewCount {
        public long itemId;
        public long windowEnd;
        public long count;

        public ItemViewCount(long itemId, long windowEnd, long count) {
            this.itemId = itemId;
            this.windowEnd = windowEnd;
            this.count = count;
        }

        public long getWindowEnd() { return windowEnd; }
    }

    public static class UserSession {
        public long userId;
        public long sessionStart;
        public long sessionEnd;
        public int eventCount;

        @Override
        public String toString() {
            return String.format("用户: %d, 会话时长: %d秒, 事件数: %d",
                userId, (sessionEnd - sessionStart) / 1000, eventCount);
        }
    }

    // ============ 聚合函数 ============

    // PV/UV聚合
    public static class PvUvAggregateFunction
            implements AggregateFunction<UserBehavior, Tuple2<Long, Set<Long>>, Tuple2<Long, Long>> {

        @Override
        public Tuple2<Long, Set<Long>> createAccumulator() {
            return new Tuple2<>(0L, new HashSet<>());
        }

        @Override
        public Tuple2<Long, Set<Long>> add(UserBehavior value, Tuple2<Long, Set<Long>> acc) {
            acc.f0 += 1;  // PV
            acc.f1.add(value.userId);  // UV
            return acc;
        }

        @Override
        public Tuple2<Long, Long> getResult(Tuple2<Long, Set<Long>> acc) {
            return new Tuple2<>(acc.f0, (long) acc.f1.size());
        }

        @Override
        public Tuple2<Long, Set<Long>> merge(Tuple2<Long, Set<Long>> a, Tuple2<Long, Set<Long>> b) {
            a.f0 += b.f0;
            a.f1.addAll(b.f1);
            return a;
        }
    }

    public static class PvUvWindowFunction
            extends ProcessWindowFunction<Tuple2<Long, Long>, PageViewStats, String, TimeWindow> {

        @Override
        public void process(String key,
                          Context ctx,
                          Iterable<Tuple2<Long, Long>> elements,
                          Collector<PageViewStats> out) {
            Tuple2<Long, Long> result = elements.iterator().next();
            out.collect(new PageViewStats(ctx.window().getEnd(), result.f0, result.f1));
        }
    }

    // 商品点击计数
    public static class CountAggregateFunction
            implements AggregateFunction<UserBehavior, Long, Long> {

        @Override
        public Long createAccumulator() { return 0L; }

        @Override
        public Long add(UserBehavior value, Long acc) { return acc + 1; }

        @Override
        public Long getResult(Long acc) { return acc; }

        @Override
        public Long merge(Long a, Long b) { return a + b; }
    }

    public static class ItemViewWindowFunction
            extends ProcessWindowFunction<Long, ItemViewCount, Long, TimeWindow> {

        @Override
        public void process(Long itemId,
                          Context ctx,
                          Iterable<Long> elements,
                          Collector<ItemViewCount> out) {
            long count = elements.iterator().next();
            out.collect(new ItemViewCount(itemId, ctx.window().getEnd(), count));
        }
    }

    // Top-N热门商品
    public static class TopNHotItems extends KeyedProcessFunction<Long, ItemViewCount, String> {
        private final int topN;
        private ListState<ItemViewCount> itemState;

        public TopNHotItems(int topN) {
            this.topN = topN;
        }

        @Override
        public void open(Configuration parameters) {
            itemState = getRuntimeContext().getListState(
                new ListStateDescriptor<>("item-state", ItemViewCount.class)
            );
        }

        @Override
        public void processElement(ItemViewCount value,
                                  Context ctx,
                                  Collector<String> out) throws Exception {
            itemState.add(value);
            ctx.timerService().registerEventTimeTimer(value.windowEnd + 1);
        }

        @Override
        public void onTimer(long timestamp,
                          OnTimerContext ctx,
                          Collector<String> out) throws Exception {

            List<ItemViewCount> allItems = new ArrayList<>();
            for (ItemViewCount item : itemState.get()) {
                allItems.add(item);
            }
            itemState.clear();

            allItems.sort((a, b) -> Long.compare(b.count, a.count));

            StringBuilder result = new StringBuilder();
            result.append("========== 窗口结束: ").append(timestamp - 1).append(" ==========\n");
            for (int i = 0; i < Math.min(topN, allItems.size()); i++) {
                ItemViewCount item = allItems.get(i);
                result.append("TOP").append(i + 1).append(": ")
                      .append("商品ID=").append(item.itemId)
                      .append(", 浏览量=").append(item.count)
                      .append("\n");
            }

            out.collect(result.toString());
        }
    }

    // 用户会话分析
    public static class UserSessionProcessor
            extends KeyedProcessFunction<Long, UserBehavior, UserSession> {

        private ValueState<Long> sessionStartState;
        private ValueState<Long> lastEventTimeState;
        private ValueState<Integer> eventCountState;

        private final long sessionTimeout = 30 * 60 * 1000;  // 30分钟

        @Override
        public void open(Configuration parameters) {
            sessionStartState = getRuntimeContext().getState(
                new ValueStateDescriptor<>("session-start", Long.class)
            );
            lastEventTimeState = getRuntimeContext().getState(
                new ValueStateDescriptor<>("last-event-time", Long.class)
            );
            eventCountState = getRuntimeContext().getState(
                new ValueStateDescriptor<>("event-count", Integer.class)
            );
        }

        @Override
        public void processElement(UserBehavior value,
                                  Context ctx,
                                  Collector<UserSession> out) throws Exception {

            long currentTime = value.timestamp * 1000;

            if (sessionStartState.value() == null) {
                // 新会话
                sessionStartState.update(currentTime);
                lastEventTimeState.update(currentTime);
                eventCountState.update(1);
            } else {
                long lastEventTime = lastEventTimeState.value();

                if (currentTime - lastEventTime > sessionTimeout) {
                    // 会话超时，输出旧会话
                    UserSession session = new UserSession();
                    session.userId = value.userId;
                    session.sessionStart = sessionStartState.value();
                    session.sessionEnd = lastEventTime;
                    session.eventCount = eventCountState.value();
                    out.collect(session);

                    // 开始新会话
                    sessionStartState.update(currentTime);
                    eventCountState.update(1);
                } else {
                    // 继续当前会话
                    eventCountState.update(eventCountState.value() + 1);
                }

                lastEventTimeState.update(currentTime);
            }

            // 注册超时定时器
            ctx.timerService().registerEventTimeTimer(currentTime + sessionTimeout);
        }

        @Override
        public void onTimer(long timestamp,
                          OnTimerContext ctx,
                          Collector<UserSession> out) throws Exception {

            if (lastEventTimeState.value() != null &&
                timestamp == lastEventTimeState.value() + sessionTimeout) {
                // 会话超时
                UserSession session = new UserSession();
                session.userId = ctx.getCurrentKey();
                session.sessionStart = sessionStartState.value();
                session.sessionEnd = lastEventTimeState.value();
                session.eventCount = eventCountState.value();
                out.collect(session);

                // 清理状态
                sessionStartState.clear();
                lastEventTimeState.clear();
                eventCountState.clear();
            }
        }
    }

    // 模拟数据源
    public static class UserBehaviorSource implements SourceFunction<UserBehavior> {
        private volatile boolean running = true;
        private Random random = new Random();

        @Override
        public void run(SourceContext<UserBehavior> ctx) throws Exception {
            String[] behaviors = {"pv", "cart", "fav", "buy"};

            while (running) {
                UserBehavior behavior = new UserBehavior(
                    random.nextInt(1000),           // userId
                    random.nextInt(100),            // itemId
                    behaviors[random.nextInt(4)],   // behavior
                    System.currentTimeMillis() / 1000  // timestamp
                );

                ctx.collect(behavior);
                Thread.sleep(100);
            }
        }

        @Override
        public void cancel() {
            running = false;
        }
    }

    public static class Tuple2<T0, T1> {
        public T0 f0;
        public T1 f1;

        public Tuple2(T0 f0, T1 f1) {
            this.f0 = f0;
            this.f1 = f1;
        }
    }
}
```

## 7. 性能调优

### 7.1 调优检查清单

```
Flink性能优化清单
├── 资源配置
│   ├── TaskManager内存合理分配
│   ├── 并行度设置（source/map/sink）
│   └── Slot共享组配置
│
├── Checkpoint优化
│   ├── 增量Checkpoint（RocksDB）
│   ├── 异步Checkpoint
│   └── 本地恢复
│
├── 状态优化
│   ├── 使用RocksDB状态后端
│   ├── 状态TTL配置
│   └── 避免状态无限增长
│
├── 网络优化
│   ├── 网络缓冲区调整
│   ├── 压缩配置
│   └── Rebalance vs Rescale
│
└── 代码优化
    ├── 避免热点key
    ├── 合理使用缓存
    └── 减少序列化开销
```

**关键配置参数**：

```yaml
# flink-conf.yaml
taskmanager.memory.process.size: 4096m
taskmanager.numberOfTaskSlots: 4

# Checkpoint配置
state.backend: rocksdb
state.checkpoints.dir: hdfs:///flink/checkpoints
state.backend.incremental: true
state.backend.local-recovery: true

# 网络缓冲区
taskmanager.network.memory.fraction: 0.1
taskmanager.network.memory.min: 64mb
taskmanager.network.memory.max: 1gb

# RocksDB调优
state.backend.rocksdb.predefined-options: SPINNING_DISK_OPTIMIZED
state.backend.rocksdb.block.cache-size: 256m
```

Flink流处理完整教程完成！
