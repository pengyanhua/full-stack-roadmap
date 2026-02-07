# 性能压测：JMeter/Gatling/K6完整方案

## 1. 压测概述

### 1.1 压测类型

```
性能测试类型金字塔：

                  /\
                 /  \
                /生产  \          - 真实流量
               /  验证  \         - Chaos Engineering
              /──────────\
             /            \
            /  压力测试    \      - 找到系统极限
           /   (Stress)    \     - 超出预期负载
          /────────────────\
         /                  \
        /    负载测试        \    - 预期负载
       /   (Load Testing)    \   - 持续时间长
      /──────────────────────\
     /                        \
    /        基准测试          \  - 单接口性能
   /    (Benchmark Testing)    \ - 确定baseline
  /──────────────────────────────\
```

### 1.2 关键指标

```
核心性能指标：

┌─────────────────────────────────────┐
│  吞吐量 (Throughput)                │
│  - RPS (Requests Per Second)       │
│  - TPS (Transactions Per Second)   │
│  目标: 5000+ RPS                    │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  响应时间 (Response Time)           │
│  - P50 (中位数)                     │
│  - P95 (95%用户体验)                │
│  - P99 (99%用户体验)                │
│  目标: P95 < 500ms, P99 < 1s       │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  错误率 (Error Rate)                │
│  - 4xx客户端错误                    │
│  - 5xx服务端错误                    │
│  - 超时                             │
│  目标: < 0.1%                       │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  并发用户数 (Concurrent Users)      │
│  - 同时在线                         │
│  - 同时请求                         │
│  目标: 10000+ CCU                   │
└─────────────────────────────────────┘
```

## 2. JMeter压测方案

### 2.1 JMeter安装配置

```bash
# 下载JMeter
wget https://dlcdn.apache.org//jmeter/binaries/apache-jmeter-5.6.3.tgz
tar -xzf apache-jmeter-5.6.3.tgz
cd apache-jmeter-5.6.3

# 配置JVM参数（bin/jmeter）
export HEAP="-Xms1g -Xmx4g -XX:MaxMetaspaceSize=512m"

# 启动GUI模式（脚本开发）
./bin/jmeter

# 启动CLI模式（实际压测）
./bin/jmeter -n -t test.jmx -l results.jtl -j jmeter.log
```

### 2.2 完整压测脚本

**电商订单接口压测.jmx**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<jmeterTestPlan version="1.2" properties="5.0">
  <hashTree>
    <!-- 测试计划 -->
    <TestPlan guiclass="TestPlanGui" testclass="TestPlan" testname="订单API压测">
      <elementProp name="TestPlan.user_defined_variables"
                   elementType="Arguments">
        <collectionProp name="Arguments.arguments">
          <!-- 变量定义 -->
          <elementProp name="BASE_URL" elementType="Argument">
            <stringProp name="Argument.name">BASE_URL</stringProp>
            <stringProp name="Argument.value">http://api.example.com</stringProp>
          </elementProp>
          <elementProp name="THREAD_COUNT" elementType="Argument">
            <stringProp name="Argument.name">THREAD_COUNT</stringProp>
            <stringProp name="Argument.value">100</stringProp>
          </elementProp>
          <elementProp name="RAMP_TIME" elementType="Argument">
            <stringProp name="Argument.name">RAMP_TIME</stringProp>
            <stringProp name="Argument.value">60</stringProp>
          </elementProp>
          <elementProp name="DURATION" elementType="Argument">
            <stringProp name="Argument.name">DURATION</stringProp>
            <stringProp name="Argument.value">300</stringProp>
          </elementProp>
        </collectionProp>
      </elementProp>
    </TestPlan>
    <hashTree>
      <!-- 线程组 -->
      <ThreadGroup guiclass="ThreadGroupGui"
                   testclass="ThreadGroup"
                   testname="订单用户组">
        <stringProp name="ThreadGroup.num_threads">${THREAD_COUNT}</stringProp>
        <stringProp name="ThreadGroup.ramp_time">${RAMP_TIME}</stringProp>
        <stringProp name="ThreadGroup.duration">${DURATION}</stringProp>
        <boolProp name="ThreadGroup.scheduler">true</boolProp>
        <stringProp name="ThreadGroup.on_sample_error">continue</stringProp>
      </ThreadGroup>
      <hashTree>
        <!-- HTTP请求默认值 -->
        <ConfigTestElement guiclass="HttpDefaultsGui"
                          testclass="ConfigTestElement"
                          testname="HTTP默认配置">
          <elementProp name="HTTPsampler.Arguments"
                       elementType="Arguments">
            <collectionProp name="Arguments.arguments"/>
          </elementProp>
          <stringProp name="HTTPSampler.domain">${BASE_URL}</stringProp>
          <stringProp name="HTTPSampler.protocol">http</stringProp>
          <stringProp name="HTTPSampler.port">80</stringProp>
        </ConfigTestElement>
        <hashTree/>

        <!-- HTTP Header Manager -->
        <HeaderManager guiclass="HeaderPanel"
                      testclass="HeaderManager"
                      testname="HTTP Header">
          <collectionProp name="HeaderManager.headers">
            <elementProp name="" elementType="Header">
              <stringProp name="Header.name">Content-Type</stringProp>
              <stringProp name="Header.value">application/json</stringProp>
            </elementProp>
            <elementProp name="" elementType="Header">
              <stringProp name="Header.name">Authorization</stringProp>
              <stringProp name="Header.value">Bearer ${token}</stringProp>
            </elementProp>
          </collectionProp>
        </HeaderManager>
        <hashTree/>

        <!-- CSV数据文件 -->
        <CSVDataSet guiclass="TestBeanGUI"
                    testclass="CSVDataSet"
                    testname="用户数据">
          <stringProp name="filename">users.csv</stringProp>
          <stringProp name="fileEncoding">UTF-8</stringProp>
          <stringProp name="variableNames">userId,token</stringProp>
          <boolProp name="recycle">true</boolProp>
          <boolProp name="stopThread">false</boolProp>
          <stringProp name="shareMode">shareMode.all</stringProp>
        </CSVDataSet>
        <hashTree/>

        <!-- 场景1: 创建订单 -->
        <HTTPSamplerProxy guiclass="HttpTestSampleGui"
                         testclass="HTTPSamplerProxy"
                         testname="创建订单">
          <stringProp name="HTTPSampler.domain"></stringProp>
          <stringProp name="HTTPSampler.port"></stringProp>
          <stringProp name="HTTPSampler.protocol"></stringProp>
          <stringProp name="HTTPSampler.path">/api/orders</stringProp>
          <stringProp name="HTTPSampler.method">POST</stringProp>
          <boolProp name="HTTPSampler.follow_redirects">true</boolProp>
          <boolProp name="HTTPSampler.use_keepalive">true</boolProp>
          <elementProp name="HTTPsampler.Arguments"
                       elementType="Arguments">
            <collectionProp name="Arguments.arguments">
              <elementProp name="" elementType="HTTPArgument">
                <boolProp name="HTTPArgument.always_encode">false</boolProp>
                <stringProp name="Argument.value"><![CDATA[{
  "customerId": "${userId}",
  "items": [
    {
      "productId": "SKU${__Random(1,100)}",
      "quantity": ${__Random(1,5)}
    }
  ],
  "shippingAddress": {
    "province": "北京",
    "city": "北京市",
    "district": "朝阳区",
    "street": "测试街道${__Random(1,100)}号",
    "receiverName": "测试用户",
    "receiverPhone": "13800138000"
  }
}]]></stringProp>
                <stringProp name="Argument.metadata">=</stringProp>
              </elementProp>
            </collectionProp>
          </elementProp>
        </HTTPSamplerProxy>
        <hashTree>
          <!-- JSON断言 -->
          <JSONPathAssertion guiclass="JSONPathAssertionGui"
                            testclass="JSONPathAssertion"
                            testname="验证订单ID">
            <stringProp name="JSON_PATH">$.orderId</stringProp>
            <stringProp name="EXPECTED_VALUE"></stringProp>
            <boolProp name="JSONVALIDATION">true</boolProp>
            <boolProp name="EXPECT_NULL">false</boolProp>
            <boolProp name="INVERT">false</boolProp>
          </JSONPathAssertion>
          <hashTree/>

          <!-- 提取订单ID -->
          <JSONPostProcessor guiclass="JSONPostProcessorGui"
                            testclass="JSONPostProcessor"
                            testname="提取订单ID">
            <stringProp name="JSONPostProcessor.referenceNames">orderId</stringProp>
            <stringProp name="JSONPostProcessor.jsonPathExprs">$.orderId</stringProp>
            <stringProp name="JSONPostProcessor.match_numbers">1</stringProp>
          </JSONPostProcessor>
          <hashTree/>
        </hashTree>

        <!-- 场景2: 查询订单 -->
        <HTTPSamplerProxy guiclass="HttpTestSampleGui"
                         testclass="HTTPSamplerProxy"
                         testname="查询订单">
          <stringProp name="HTTPSampler.path">/api/orders/${orderId}</stringProp>
          <stringProp name="HTTPSampler.method">GET</stringProp>
        </HTTPSamplerProxy>
        <hashTree>
          <!-- 响应断言 -->
          <ResponseAssertion guiclass="AssertionGui"
                            testclass="ResponseAssertion"
                            testname="响应状态码200">
            <collectionProp name="Asserion.test_strings">
              <stringProp name="49586">200</stringProp>
            </collectionProp>
            <stringProp name="Assertion.test_field">Assertion.response_code</stringProp>
            <boolProp name="Assertion.assume_success">false</boolProp>
            <intProp name="Assertion.test_type">8</intProp>
          </ResponseAssertion>
          <hashTree/>
        </hashTree>

        <!-- 思考时间 -->
        <UniformRandomTimer guiclass="UniformRandomTimerGui"
                           testclass="UniformRandomTimer"
                           testname="随机等待">
          <stringProp name="ConstantTimer.delay">1000</stringProp>
          <stringProp name="RandomTimer.range">2000</stringProp>
        </UniformRandomTimer>
        <hashTree/>

        <!-- 监听器 -->
        <BackendListener guiclass="BackendListenerGui"
                        testclass="BackendListener"
                        testname="InfluxDB后端监听">
          <elementProp name="arguments" elementType="Arguments">
            <collectionProp name="Arguments.arguments">
              <elementProp name="influxdbUrl" elementType="Argument">
                <stringProp name="Argument.name">influxdbUrl</stringProp>
                <stringProp name="Argument.value">http://localhost:8086/write?db=jmeter</stringProp>
              </elementProp>
            </collectionProp>
          </elementProp>
          <stringProp name="classname">org.apache.jmeter.visualizers.backend.influxdb.InfluxdbBackendListenerClient</stringProp>
        </BackendListener>
        <hashTree/>
      </hashTree>
    </hashTree>
  </hashTree>
</jmeterTestPlan>
```

### 2.3 JMeter命令行执行

```bash
#!/bin/bash
# run_jmeter.sh

# 配置变量
TEST_PLAN="order_load_test.jmx"
RESULT_DIR="results/$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULT_DIR

# 执行压测
jmeter -n \
  -t $TEST_PLAN \
  -l $RESULT_DIR/results.jtl \
  -j $RESULT_DIR/jmeter.log \
  -e -o $RESULT_DIR/html \
  -JTHREAD_COUNT=500 \
  -JRAMP_TIME=300 \
  -JDURATION=3600

# 生成报告
echo "压测完成，报告路径: $RESULT_DIR/html/index.html"
```

### 2.4 分布式压测

```
主从架构：

┌─────────────┐
│  Controller │ ← JMeter GUI/CLI
│  (主节点)   │
└──────┬──────┘
       │
       ├──────────┬──────────┬──────────┐
       ↓          ↓          ↓          ↓
 ┌──────────┐┌──────────┐┌──────────┐┌──────────┐
 │ Agent 1  ││ Agent 2  ││ Agent 3  ││ Agent 4  │
 │(压测机1) ││(压测机2) ││(压测机3) ││(压测机4) │
 └─────┬────┘└─────┬────┘└─────┬────┘└─────┬────┘
       │           │           │           │
       └───────────┴───────────┴───────────┘
                      ↓
            ┌────────────────┐
            │  目标服务器    │
            └────────────────┘
```

**配置步骤：**

```bash
# 1. 所有机器安装JMeter（相同版本）

# 2. 配置Agent（从节点）
# 编辑 jmeter-server 启动脚本
vi bin/jmeter-server
# 设置 RMI_HOST_DEF=-Djava.rmi.server.hostname=<Agent_IP>

# 启动Agent
./bin/jmeter-server

# 3. 配置Controller（主节点）
# 编辑 jmeter.properties
vi bin/jmeter.properties
# 添加Agent地址
remote_hosts=192.168.1.101:1099,192.168.1.102:1099,192.168.1.103:1099

# 4. 启动分布式压测
jmeter -n -t test.jmx -r -l results.jtl
# -r 参数：启动所有远程Agent
```

## 3. Gatling压测方案

### 3.1 Gatling安装

```bash
# 使用Maven/Gradle
# pom.xml
<dependency>
    <groupId>io.gatling.highcharts</groupId>
    <artifactId>gatling-charts-highcharts</artifactId>
    <version>3.10.3</version>
    <scope>test</scope>
</dependency>

# 或下载独立版本
wget https://repo1.maven.org/maven2/io/gatling/highcharts/gatling-charts-highcharts-bundle/3.10.3/gatling-charts-highcharts-bundle-3.10.3.zip
unzip gatling-charts-highcharts-bundle-3.10.3.zip
cd gatling-charts-highcharts-bundle-3.10.3
```

### 3.2 Gatling脚本（Scala）

```scala
// OrderSimulation.scala
package simulations

import io.gatling.core.Predef._
import io.gatling.http.Predef._
import scala.concurrent.duration._

class OrderSimulation extends Simulation {

  // HTTP配置
  val httpProtocol = http
    .baseUrl("http://api.example.com")
    .acceptHeader("application/json")
    .contentTypeHeader("application/json")
    .userAgentHeader("Gatling")

  // 场景定义
  object OrderScenario {

    // 用户数据Feeder
    val userFeeder = csv("users.csv").random

    // 商品ID Feeder
    val productFeeder = Iterator.continually(Map(
      "productId" -> s"SKU${scala.util.Random.nextInt(100)}",
      "quantity" -> (scala.util.Random.nextInt(5) + 1)
    ))

    // 创建订单
    val createOrder = exec(
      http("创建订单")
        .post("/api/orders")
        .header("Authorization", "Bearer ${token}")
        .body(StringBody("""{
          "customerId": "${userId}",
          "items": [
            {
              "productId": "${productId}",
              "quantity": ${quantity}
            }
          ],
          "shippingAddress": {
            "province": "北京",
            "city": "北京市",
            "district": "朝阳区",
            "street": "测试街道",
            "receiverName": "测试用户",
            "receiverPhone": "13800138000"
          }
        }""")).asJson
        .check(status.is(200))
        .check(jsonPath("$.orderId").saveAs("orderId"))
    )
    .pause(1, 3) // 思考时间1-3秒

    // 查询订单
    val getOrder = exec(
      http("查询订单")
        .get("/api/orders/${orderId}")
        .header("Authorization", "Bearer ${token}")
        .check(status.is(200))
        .check(jsonPath("$.status").exists)
    )
    .pause(2, 5)

    // 支付订单
    val payOrder = exec(
      http("支付订单")
        .post("/api/orders/${orderId}/pay")
        .header("Authorization", "Bearer ${token}")
        .body(StringBody("""{
          "paymentMethod": "WECHAT_PAY",
          "amount": 10000
        }""")).asJson
        .check(status.is(200))
    )
  }

  // 场景组合
  val normalUserScenario = scenario("正常用户购物流程")
    .feed(OrderScenario.userFeeder)
    .feed(OrderScenario.productFeeder)
    .exec(OrderScenario.createOrder)
    .exec(OrderScenario.getOrder)
    .exec(OrderScenario.payOrder)

  val heavyUserScenario = scenario("重度用户购物流程")
    .feed(OrderScenario.userFeeder)
    .repeat(5) {
      feed(OrderScenario.productFeeder)
        .exec(OrderScenario.createOrder)
    }
    .exec(OrderScenario.getOrder)

  // 负载模型
  setUp(
    // 逐步增加负载
    normalUserScenario.inject(
      nothingFor(10 seconds), // 预热
      rampUsers(100) during (1 minute), // 1分钟内增加到100用户
      constantUsersPerSec(50) during (5 minutes), // 保持50 RPS持续5分钟
      rampUsersPerSec(50) to 200 during (2 minutes), // 2分钟内从50 RPS增加到200 RPS
      constantUsersPerSec(200) during (10 minutes) // 保持200 RPS持续10分钟
    ),

    heavyUserScenario.inject(
      nothingFor(2 minutes),
      rampUsers(20) during (1 minute)
    )
  ).protocols(httpProtocol)
   .assertions(
     global.responseTime.max.lt(5000), // 最大响应时间 < 5s
     global.responseTime.percentile3.lt(1000), // P95 < 1s
     global.successfulRequests.percent.gt(99) // 成功率 > 99%
   )
}
```

### 3.3 执行Gatling测试

```bash
# 使用Gatling独立版
./bin/gatling.sh -s simulations.OrderSimulation

# 使用Maven
mvn gatling:test -Dgatling.simulationClass=simulations.OrderSimulation

# 使用Gradle
gradle gatlingRun
```

## 4. K6压测方案

### 4.1 K6安装

```bash
# macOS
brew install k6

# Linux
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6

# Docker
docker pull grafana/k6
```

### 4.2 K6脚本（JavaScript）

```javascript
// order_load_test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { SharedArray } from 'k6/data';

// 自定义指标
const errorRate = new Rate('errors');
const orderCreationTime = new Trend('order_creation_time');
const orderCounter = new Counter('orders_created');

// 测试数据
const users = new SharedArray('users', function () {
  return JSON.parse(open('./users.json'));
});

// 负载配置
export const options = {
  stages: [
    { duration: '1m', target: 100 },   // 1分钟增加到100用户
    { duration: '5m', target: 100 },   // 保持100用户5分钟
    { duration: '2m', target: 500 },   // 2分钟增加到500用户
    { duration: '10m', target: 500 },  // 保持500用户10分钟
    { duration: '2m', target: 1000 },  // 2分钟增加到1000用户
    { duration: '5m', target: 1000 },  // 保持1000用户5分钟（压力测试）
    { duration: '3m', target: 0 },     // 3分钟逐步减少到0
  ],

  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'], // P95<500ms, P99<1s
    http_req_failed: ['rate<0.01'],  // 错误率 < 1%
    errors: ['rate<0.1'],
    order_creation_time: ['p(95)<800'],
  },

  ext: {
    loadimpact: {
      projectID: 3499952,
      name: '订单API压测'
    }
  }
};

// 基础URL
const BASE_URL = 'http://api.example.com';

// 主测试函数
export default function () {
  // 随机选择用户
  const user = users[Math.floor(Math.random() * users.length)];

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${user.token}`,
    },
  };

  // 1. 创建订单
  const createOrderPayload = JSON.stringify({
    customerId: user.userId,
    items: [
      {
        productId: `SKU${Math.floor(Math.random() * 100)}`,
        quantity: Math.floor(Math.random() * 5) + 1,
      },
    ],
    shippingAddress: {
      province: '北京',
      city: '北京市',
      district: '朝阳区',
      street: `测试街道${Math.floor(Math.random() * 100)}号`,
      receiverName: '测试用户',
      receiverPhone: '13800138000',
    },
  });

  const createOrderStart = Date.now();
  const createOrderRes = http.post(
    `${BASE_URL}/api/orders`,
    createOrderPayload,
    params
  );

  const createOrderDuration = Date.now() - createOrderStart;
  orderCreationTime.add(createOrderDuration);

  // 验证响应
  const createOrderSuccess = check(createOrderRes, {
    '订单创建成功': (r) => r.status === 200,
    '返回订单ID': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.orderId !== undefined;
      } catch (e) {
        return false;
      }
    },
  });

  if (!createOrderSuccess) {
    errorRate.add(1);
  } else {
    orderCounter.add(1);

    const orderId = JSON.parse(createOrderRes.body).orderId;

    // 2. 查询订单
    sleep(Math.random() * 2 + 1); // 思考时间1-3秒

    const getOrderRes = http.get(
      `${BASE_URL}/api/orders/${orderId}`,
      params
    );

    check(getOrderRes, {
      '订单查询成功': (r) => r.status === 200,
      '订单状态正确': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.status === 'PENDING';
        } catch (e) {
          return false;
        }
      },
    });

    // 3. 支付订单（30%概率）
    if (Math.random() < 0.3) {
      sleep(Math.random() * 3 + 2); // 思考时间2-5秒

      const payOrderPayload = JSON.stringify({
        paymentMethod: 'WECHAT_PAY',
        amount: 10000,
      });

      const payOrderRes = http.post(
        `${BASE_URL}/api/orders/${orderId}/pay`,
        payOrderPayload,
        params
      );

      check(payOrderRes, {
        '订单支付成功': (r) => r.status === 200,
      });
    }
  }

  sleep(1);
}

// 设置阶段（Setup）
export function setup() {
  console.log('压测开始准备...');
  // 可以在这里创建测试数据
  return { startTime: Date.now() };
}

// 拆卸阶段（Teardown）
export function teardown(data) {
  console.log('压测完成！');
  console.log(`总耗时: ${(Date.now() - data.startTime) / 1000}秒`);
}
```

### 4.3 执行K6测试

```bash
# 本地执行
k6 run order_load_test.js

# 指定虚拟用户数和持续时间
k6 run --vus 100 --duration 30s order_load_test.js

# 输出结果到JSON
k6 run --out json=results.json order_load_test.js

# 输出到InfluxDB + Grafana可视化
k6 run --out influxdb=http://localhost:8086/k6 order_load_test.js

# 使用K6 Cloud
k6 cloud order_load_test.js
```

### 4.4 K6结果分析

```
K6输出示例：

          /\      |‾‾| /‾‾/   /‾‾/
     /\  /  \     |  |/  /   /  /
    /  \/    \    |     (   /   ‾‾\
   /          \   |  |\  \ |  (‾)  |
  / __________ \  |__| \__\ \_____/ .io

  execution: local
     script: order_load_test.js
     output: -

  scenarios: (100.00%) 1 scenario, 1000 max VUs, 28m30s max duration
           * default: Up to 1000 looping VUs for 28m0s over 7 stages

     ✓ 订单创建成功
     ✓ 返回订单ID
     ✓ 订单查询成功
     ✓ 订单状态正确

     checks.........................: 99.85% ✓ 589215    ✗ 885
     data_received..................: 156 MB 93 kB/s
     data_sent......................: 98 MB  59 kB/s
     errors.........................: 0.15%  ✓ 885       ✗ 588330
     http_req_blocked...............: avg=12.45µs  min=1.2µs   med=4.8µs    max=456.23ms p(90)=8.9µs   p(95)=11.2µs
     http_req_connecting............: avg=4.67µs   min=0s      med=0s       max=234.12ms p(90)=0s      p(95)=0s
     http_req_duration..............: avg=234.56ms min=45.23ms med=198.45ms max=4.98s    p(90)=412.3ms p(95)=567.8ms
       { expected_response:true }...: avg=232.12ms min=45.23ms med=197.23ms max=3.45s    p(90)=408.9ms p(95)=562.1ms
     http_req_failed................: 0.15%  ✓ 885       ✗ 588330
     http_req_receiving.............: avg=123.45µs min=23.4µs  med=98.2µs   max=234.56ms p(90)=198.7µs p(95)=267.3µs
     http_req_sending...............: avg=45.23µs  min=8.9µs   med=34.5µs   max=123.45ms p(90)=67.8µs  p(95)=89.2µs
     http_req_tls_handshaking.......: avg=0s       min=0s      med=0s       max=0s       p(90)=0s      p(95)=0s
     http_req_waiting...............: avg=234.39ms min=45.12ms med=198.28ms max=4.97s    p(90)=412.1ms p(95)=567.5ms
     http_reqs......................: 589215 353.129/s
     iteration_duration.............: avg=2.8s     min=1.23s   med=2.56s    max=12.34s   p(90)=4.12s   p(95)=5.23s
     iterations.....................: 196405 117.709/s
     order_creation_time............: avg=245.67ms min=47.12ms med=205.34ms max=5.01s    p(90)=425.6ms p(95)=582.3ms
     orders_created.................: 195520 117.178/s
     vus............................: 4      min=4       max=1000
     vus_max........................: 1000   min=1000    max=1000

running (28m00.0s), 0000/1000 VUs, 196405 complete and 0 interrupted iterations
default ✓ [======================================] 0000/1000 VUs  28m0s
```

## 5. 压测最佳实践

### 5.1 压测准备清单

```
环境准备：
□ 独立压测环境（与生产环境隔离）
□ 压测环境配置与生产一致
□ 数据库初始化（足够测试数据）
□ 第三方依赖Mock（避免影响真实系统）
□ 监控系统就绪（Prometheus+Grafana）

脚本准备：
□ 场景覆盖核心业务流程
□ 参数化（避免缓存命中率过高）
□ 合理的思考时间
□ 断言验证
□ 错误处理

执行准备：
□ 通知相关团队
□ 预留时间窗口
□ 准备回滚方案
□ 设置告警阈值
```

### 5.2 逐步增压策略

```
负载增长曲线：

并发数
 │
1000│                    ┌────────┐
    │                   /│        │
 500│            ┌─────┘ │        │\
    │           /│       │        │ \
 100│    ┌─────┘ │       │        │  \
    │   /│       │       │        │   └────
  50│──┘ │       │       │        │
    │    │       │       │        │
   0└────┴───────┴───────┴────────┴──────────► 时间
      预热  基准   峰值   极限    稳定性  降压

阶段说明：
1. 预热 (5-10min): 50用户，检查系统状态
2. 基准测试 (10min): 100用户，建立baseline
3. 峰值测试 (15min): 500用户，模拟业务高峰
4. 极限测试 (10min): 1000用户，找到系统上限
5. 稳定性测试 (30min): 峰值负载长时间运行
6. 降压 (5min): 逐步减少负载
```

### 5.3 监控指标

```
应用层监控：
- QPS/TPS
- 响应时间（P50/P90/P95/P99/Max）
- 错误率
- 线程池状态
- JVM堆内存使用率
- GC频率和时间

系统层监控：
- CPU使用率
- 内存使用率
- 磁盘IO
- 网络IO
- TCP连接数

数据库监控：
- 慢查询
- 连接池使用率
- QPS
- 锁等待

中间件监控：
- Redis命中率
- 消息队列堆积
- 缓存内存使用
```

## 6. 结果分析与优化

### 6.1 性能瓶颈识别

```
瓶颈类型矩阵：

CPU密集型：
症状: CPU 90%+, 响应时间慢
原因: 复杂计算、序列化、加解密
优化: 算法优化、缓存、异步

内存密集型：
症状: 内存占用高、频繁GC
原因: 大对象、内存泄漏
优化: 对象池、流式处理

IO密集型：
症状: CPU低、IO wait高
原因: 数据库查询、文件读写
优化: 索引、连接池、异步IO

网络密集型：
症状: 网络带宽打满
原因: 大量数据传输
优化: 压缩、CDN、协议优化
```

### 6.2 报告模板

```markdown
# 订单API压测报告

## 1. 测试环境
- 服务器: 4C8G × 3台
- 数据库: MySQL 8.0, 16C32G
- Redis: 8C16G
- 网络: 1Gbps

## 2. 测试场景
- 并发用户: 1000
- 测试时长: 30分钟
- 业务场景: 创建订单 + 查询订单

## 3. 测试结果
| 指标 | 目标 | 实际 | 是否达标 |
|------|------|------|---------|
| P95响应时间 | <500ms | 423ms | ✓ |
| P99响应时间 | <1s | 876ms | ✓ |
| 错误率 | <0.1% | 0.08% | ✓ |
| QPS | >5000 | 5234 | ✓ |

## 4. 瓶颈分析
- 数据库连接池在高峰期接近上限
- Redis缓存命中率较低(65%)

## 5. 优化建议
1. 增加数据库连接池大小: 50 → 100
2. 优化缓存策略，提高命中率
3. 添加熔断机制，防止雪崩

## 6. 附件
- 详细监控截图
- JMeter测试报告
- 数据库慢查询日志
```

## 7. 总结

压测工具选择建议：
- **JMeter**: 功能全面，生态成熟，适合传统企业
- **Gatling**: 代码即测试，性能好，适合DevOps团队
- **K6**: 轻量级，云原生，适合微服务架构

压测成功的关键：
1. 充分准备，环境隔离
2. 逐步增压，找到极限
3. 全面监控，快速定位
4. 持续优化，建立baseline
