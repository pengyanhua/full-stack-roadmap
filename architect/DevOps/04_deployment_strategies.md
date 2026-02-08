# 部署策略

## 目录
- [部署策略概览](#部署策略概览)
- [蓝绿部署](#蓝绿部署)
- [金丝雀发布](#金丝雀发布)
- [滚动更新](#滚动更新)
- [A/B测试](#ab测试)
- [特性开关](#特性开关)
- [实战案例](#实战案例)

---

## 部署策略概览

### 部署策略对比

```
┌──────────────────────────────────────────────────────────────┐
│                    部署策略对比矩阵                          │
├────────────┬──────────┬──────────┬──────────┬───────────────┤
│ 策略       │ 回滚速度 │ 资源成本 │ 风险     │  复杂度       │
├────────────┼──────────┼──────────┼──────────┼───────────────┤
│ 蓝绿部署   │ ⚡️ 秒级  │ 💰💰    │ ⭐       │  ⭐⭐        │
│ 金丝雀     │ ⚡️ 分钟  │ 💰       │ ⭐⭐     │  ⭐⭐⭐      │
│ 滚动更新   │ 🐢 分钟  │ 💰       │ ⭐⭐⭐   │  ⭐          │
│ A/B测试    │ ⚡️ 秒级  │ 💰💰    │ ⭐       │  ⭐⭐⭐⭐    │
│ 重建       │ 🐢 长    │ 💰       │ ⭐⭐⭐⭐ │  ⭐          │
└────────────┴──────────┴──────────┴──────────┴───────────────┘

图例:
⚡️ = 快速   🐢 = 慢速
💰 = 低成本  💰💰 = 高成本
⭐ = 低风险  ⭐⭐⭐⭐ = 高风险
```

### 部署流程可视化

```
1. 蓝绿部署 (Blue-Green)
┌─────────────────────────────────────────┐
│  步骤1: 蓝色(当前版本) + 绿色(新版本)   │
│  [蓝 v1.0] ◀── 100% 流量               │
│  [绿 v1.1] ◀── 0% 流量                 │
│                                         │
│  步骤2: 切换流量到绿色                  │
│  [蓝 v1.0] ◀── 0% 流量                 │
│  [绿 v1.1] ◀── 100% 流量               │
│                                         │
│  步骤3: 移除蓝色环境                    │
│  [绿 v1.1] ◀── 100% 流量               │
└─────────────────────────────────────────┘

2. 金丝雀发布 (Canary)
┌─────────────────────────────────────────┐
│  [v1.0] ◀── 90% 流量                    │
│  [v1.1] ◀── 10% 流量 (金丝雀)          │
│            ↓ 观察指标                    │
│  [v1.0] ◀── 70% 流量                    │
│  [v1.1] ◀── 30% 流量                    │
│            ↓ 继续观察                    │
│  [v1.0] ◀── 0% 流量                     │
│  [v1.1] ◀── 100% 流量                   │
└─────────────────────────────────────────┘

3. 滚动更新 (Rolling)
┌─────────────────────────────────────────┐
│  [v1.0] [v1.0] [v1.0] [v1.0]           │
│    ↓                                    │
│  [v1.1] [v1.0] [v1.0] [v1.0]           │
│    ↓                                    │
│  [v1.1] [v1.1] [v1.0] [v1.0]           │
│    ↓                                    │
│  [v1.1] [v1.1] [v1.1] [v1.0]           │
│    ↓                                    │
│  [v1.1] [v1.1] [v1.1] [v1.1]           │
└─────────────────────────────────────────┘
```

---

## 蓝绿部署

### Kubernetes 蓝绿部署

```yaml
# blue-deployment.yaml (当前版本)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-blue
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
      version: blue
  template:
    metadata:
      labels:
        app: myapp
        version: blue
    spec:
      containers:
      - name: myapp
        image: myapp:v1.0.0
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi

---
# green-deployment.yaml (新版本)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-green
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
      version: green
  template:
    metadata:
      labels:
        app: myapp
        version: green
    spec:
      containers:
      - name: myapp
        image: myapp:v1.1.0
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: myapp
  namespace: production
spec:
  type: LoadBalancer
  selector:
    app: myapp
    version: blue  # 切换到 green 实现蓝绿切换
  ports:
  - port: 80
    targetPort: 8080
```

### 蓝绿部署脚本

```bash
#!/bin/bash
# blue-green-deploy.sh

set -euo pipefail

NAMESPACE="production"
APP_NAME="myapp"
NEW_VERSION="v1.1.0"
CURRENT_COLOR=$(kubectl get svc ${APP_NAME} -n ${NAMESPACE} -o jsonpath='{.spec.selector.version}')
NEW_COLOR=$([ "$CURRENT_COLOR" = "blue" ] && echo "green" || echo "blue")

echo "🚀 Starting Blue-Green Deployment"
echo "Current version: ${CURRENT_COLOR}"
echo "New version: ${NEW_COLOR}"

# 1. 部署新版本
echo "📦 Deploying ${NEW_COLOR} version..."
kubectl apply -f ${NEW_COLOR}-deployment.yaml

# 2. 等待新版本就绪
echo "⏳ Waiting for ${NEW_COLOR} deployment to be ready..."
kubectl rollout status deployment/${APP_NAME}-${NEW_COLOR} -n ${NAMESPACE} --timeout=300s

# 3. 健康检查
echo "🏥 Running health checks..."
REPLICAS=$(kubectl get deployment ${APP_NAME}-${NEW_COLOR} -n ${NAMESPACE} -o jsonpath='{.status.readyReplicas}')
if [ "$REPLICAS" -lt 3 ]; then
  echo "❌ Health check failed: only $REPLICAS replicas ready"
  exit 1
fi

# 4. 烟雾测试
echo "💨 Running smoke tests..."
POD_IP=$(kubectl get pod -n ${NAMESPACE} -l version=${NEW_COLOR} -o jsonpath='{.items[0].status.podIP}')
HEALTH_STATUS=$(kubectl run --rm -i --restart=Never curl-test --image=curlimages/curl:latest -- \
  curl -s -o /dev/null -w "%{http_code}" http://${POD_IP}:8080/health)

if [ "$HEALTH_STATUS" != "200" ]; then
  echo "❌ Smoke test failed: HTTP $HEALTH_STATUS"
  exit 1
fi

# 5. 切换流量
echo "🔄 Switching traffic to ${NEW_COLOR}..."
kubectl patch service ${APP_NAME} -n ${NAMESPACE} -p "{\"spec\":{\"selector\":{\"version\":\"${NEW_COLOR}\"}}}"

# 6. 监控新版本
echo "📊 Monitoring new version for 5 minutes..."
sleep 300

# 7. 验证成功
ERROR_RATE=$(kubectl run --rm -i --restart=Never metrics-check --image=curlimages/curl:latest -- \
  curl -s http://prometheus:9090/api/v1/query --data-urlencode 'query=rate(http_requests_total{status=~"5.."}[5m])' \
  | jq -r '.data.result[0].value[1]' || echo "0")

if (( $(echo "$ERROR_RATE > 0.05" | bc -l) )); then
  echo "⚠️ High error rate detected, rolling back..."
  kubectl patch service ${APP_NAME} -n ${NAMESPACE} -p "{\"spec\":{\"selector\":{\"version\":\"${CURRENT_COLOR}\"}}}"
  exit 1
fi

# 8. 清理旧版本
echo "🧹 Cleaning up ${CURRENT_COLOR} deployment..."
kubectl scale deployment/${APP_NAME}-${CURRENT_COLOR} -n ${NAMESPACE} --replicas=0

echo "✅ Blue-Green deployment completed successfully!"
```

---

## 金丝雀发布

### Argo Rollouts 金丝雀配置

```yaml
# rollout.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: myapp
  namespace: production
spec:
  replicas: 10
  revisionHistoryLimit: 3

  selector:
    matchLabels:
      app: myapp

  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:v1.1.0
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: 100m
            memory: 128Mi

  strategy:
    canary:
      # 金丝雀步骤
      steps:
      - setWeight: 10        # 10% 流量到金丝雀
      - pause: {duration: 5m}  # 暂停5分钟观察

      - setWeight: 30        # 增加到30%
      - pause: {duration: 10m}

      - setWeight: 50        # 增加到50%
      - pause: {duration: 10m}

      - setWeight: 80        # 增加到80%
      - pause: {}            # 手动审批

      # 分析模板
      analysis:
        templates:
        - templateName: success-rate
        - templateName: latency
        startingStep: 2      # 从第2步开始分析
        args:
        - name: service-name
          value: myapp

      # 流量路由
      trafficRouting:
        nginx:
          stableIngress: myapp-stable
          annotationPrefix: nginx.ingress.kubernetes.io
          additionalIngressAnnotations:
            canary-by-header: X-Canary
            canary-by-header-value: "true"

        # 或使用 Istio
        istio:
          virtualService:
            name: myapp-vsvc
            routes:
            - primary

      # 自动提升
      autoPromotionEnabled: false
      autoPromotionSeconds: 0

      # 反亲和性
      antiAffinity:
        requiredDuringSchedulingIgnoredDuringExecution: {}

---
# analysis-template.yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
spec:
  args:
  - name: service-name

  metrics:
  - name: success-rate
    interval: 1m
    count: 5
    successCondition: result >= 0.95
    failureLimit: 3
    provider:
      prometheus:
        address: http://prometheus:9090
        query: |
          sum(rate(
            http_requests_total{
              service="{{args.service-name}}",
              status!~"5.."
            }[5m]
          ))
          /
          sum(rate(
            http_requests_total{
              service="{{args.service-name}}"
            }[5m]
          ))

---
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: latency
spec:
  args:
  - name: service-name

  metrics:
  - name: p95-latency
    interval: 1m
    count: 5
    successCondition: result <= 500
    failureLimit: 3
    provider:
      prometheus:
        address: http://prometheus:9090
        query: |
          histogram_quantile(0.95,
            sum(rate(
              http_request_duration_seconds_bucket{
                service="{{args.service-name}}"
              }[5m]
            )) by (le)
          ) * 1000
```

### Flagger 自动化金丝雀

```yaml
# flagger-canary.yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: myapp
  namespace: production
spec:
  # 目标 Deployment
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: myapp

  # 服务配置
  service:
    port: 80
    targetPort: 8080
    gateways:
    - myapp-gateway
    hosts:
    - myapp.example.com
    trafficPolicy:
      tls:
        mode: DISABLE

  # 分析配置
  analysis:
    interval: 1m
    threshold: 5
    maxWeight: 50
    stepWeight: 10

    # 指标
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
      interval: 1m

    - name: request-duration
      thresholdRange:
        max: 500
      interval: 1m

    # Webhook 测试
    webhooks:
    - name: load-test
      url: http://flagger-loadtester/
      timeout: 5s
      metadata:
        type: cmd
        cmd: "hey -z 1m -q 10 -c 2 http://myapp-canary:80/"

    - name: acceptance-test
      url: http://flagger-loadtester/
      timeout: 10s
      metadata:
        type: bash
        cmd: |
          curl -s http://myapp-canary:80/health | grep -q "healthy"
```

---

## 滚动更新

### Kubernetes Deployment 滚动更新

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  namespace: production
spec:
  replicas: 10

  # 滚动更新策略
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2         # 最多多2个Pod
      maxUnavailable: 1   # 最多不可用1个Pod

  minReadySeconds: 30     # Pod就绪后等待30秒
  revisionHistoryLimit: 10
  progressDeadlineSeconds: 600

  selector:
    matchLabels:
      app: myapp

  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:v1.1.0

        # 探针配置
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3

        # 优雅关闭
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]

      # Pod反亲和性
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - myapp
              topologyKey: kubernetes.io/hostname

      terminationGracePeriodSeconds: 30
```

### 滚动更新脚本

```bash
#!/bin/bash
# rolling-update.sh

set -euo pipefail

NAMESPACE="production"
DEPLOYMENT="myapp"
NEW_IMAGE="myapp:v1.1.0"

echo "🚀 Starting Rolling Update"

# 1. 更新镜像
echo "📦 Updating image to ${NEW_IMAGE}..."
kubectl set image deployment/${DEPLOYMENT} \
  myapp=${NEW_IMAGE} \
  -n ${NAMESPACE} \
  --record

# 2. 监控更新进度
echo "⏳ Monitoring rollout progress..."
kubectl rollout status deployment/${DEPLOYMENT} -n ${NAMESPACE} --timeout=600s

# 3. 验证更新
READY_REPLICAS=$(kubectl get deployment ${DEPLOYMENT} -n ${NAMESPACE} \
  -o jsonpath='{.status.readyReplicas}')
DESIRED_REPLICAS=$(kubectl get deployment ${DEPLOYMENT} -n ${NAMESPACE} \
  -o jsonpath='{.spec.replicas}')

if [ "$READY_REPLICAS" != "$DESIRED_REPLICAS" ]; then
  echo "❌ Rollout failed: ${READY_REPLICAS}/${DESIRED_REPLICAS} replicas ready"

  echo "🔙 Rolling back..."
  kubectl rollout undo deployment/${DEPLOYMENT} -n ${NAMESPACE}
  kubectl rollout status deployment/${DEPLOYMENT} -n ${NAMESPACE}
  exit 1
fi

echo "✅ Rolling update completed successfully!"

# 4. 查看历史
kubectl rollout history deployment/${DEPLOYMENT} -n ${NAMESPACE}
```

---

## A/B测试

### Istio VirtualService A/B测试

```yaml
# virtualservice.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: myapp
  namespace: production
spec:
  hosts:
  - myapp.example.com

  gateways:
  - myapp-gateway

  http:
  # A/B测试规则
  - match:
    - headers:
        user-agent:
          regex: ".*Mobile.*"
    route:
    - destination:
        host: myapp
        subset: version-b
      weight: 100

  # 基于Cookie
  - match:
    - headers:
        cookie:
          regex: "^(.*;)?experiment=b(;.*)?$"
    route:
    - destination:
        host: myapp
        subset: version-b
      weight: 100

  # 基于用户ID(Header)
  - match:
    - headers:
        x-user-id:
          regex: "^[0-9]*[02468]$"  # 偶数用户ID
    route:
    - destination:
        host: myapp
        subset: version-b
      weight: 100

  # 默认路由
  - route:
    - destination:
        host: myapp
        subset: version-a
      weight: 90
    - destination:
        host: myapp
        subset: version-b
      weight: 10

---
# destinationrule.yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: myapp
  namespace: production
spec:
  host: myapp

  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        http2MaxRequests: 100
        maxRequestsPerConnection: 2

    loadBalancer:
      simple: LEAST_REQUEST

    outlierDetection:
      consecutive5xxErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50

  subsets:
  - name: version-a
    labels:
      version: v1.0.0

  - name: version-b
    labels:
      version: v1.1.0
```

---

## 特性开关

### 特性开关实现

```python
# feature_flags.py
from typing import Dict, Any
import redis
import json

class FeatureFlags:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.cache = {}

    def is_enabled(self, feature: str, context: Dict[str, Any] = None) -> bool:
        """检查特性是否启用"""
        # 尝试从缓存获取
        if feature in self.cache:
            config = self.cache[feature]
        else:
            # 从Redis获取
            config_json = self.redis.get(f"feature:{feature}")
            if not config_json:
                return False

            config = json.loads(config_json)
            self.cache[feature] = config

        # 检查全局开关
        if not config.get('enabled', False):
            return False

        # 如果没有上下文,返回全局状态
        if not context:
            return config.get('percentage', 100) == 100

        # 检查用户白名单
        if context.get('user_id') in config.get('whitelist', []):
            return True

        # 检查用户黑名单
        if context.get('user_id') in config.get('blacklist', []):
            return False

        # 检查环境
        if config.get('environments') and \
           context.get('environment') not in config.get('environments', []):
            return False

        # 百分比灰度
        percentage = config.get('percentage', 100)
        if percentage < 100:
            user_id = context.get('user_id', '')
            hash_value = hash(f"{feature}:{user_id}") % 100
            return hash_value < percentage

        return True

    def get_variant(self, feature: str, context: Dict[str, Any] = None) -> str:
        """获取特性变体"""
        if not self.is_enabled(feature, context):
            return 'control'

        config = self.cache.get(feature, {})
        variants = config.get('variants', {})

        if not variants:
            return 'treatment'

        # 基于用户ID的一致性哈希
        user_id = context.get('user_id', '') if context else ''
        hash_value = hash(f"{feature}:{user_id}") % 100

        cumulative = 0
        for variant, weight in variants.items():
            cumulative += weight
            if hash_value < cumulative:
                return variant

        return 'control'

# 使用示例
flags = FeatureFlags('redis://localhost:6379')

# 应用代码中
def process_payment(user_id: str, amount: float):
    context = {
        'user_id': user_id,
        'environment': 'production'
    }

    if flags.is_enabled('new_payment_flow', context):
        # 新支付流程
        return process_payment_v2(user_id, amount)
    else:
        # 旧支付流程
        return process_payment_v1(user_id, amount)

def show_ui():
    variant = flags.get_variant('checkout_redesign', {'user_id': user_id})

    if variant == 'variant_a':
        return render_template('checkout_v1.html')
    elif variant == 'variant_b':
        return render_template('checkout_v2.html')
    else:
        return render_template('checkout_default.html')
```

### 特性开关配置

```json
{
  "new_payment_flow": {
    "enabled": true,
    "percentage": 20,
    "environments": ["production"],
    "whitelist": ["user123", "user456"],
    "blacklist": [],
    "description": "新支付流程灰度发布"
  },
  "checkout_redesign": {
    "enabled": true,
    "percentage": 50,
    "variants": {
      "variant_a": 25,
      "variant_b": 25,
      "control": 50
    },
    "description": "结算页面A/B测试"
  }
}
```

---

## 实战案例

### 综合部署策略

```yaml
# 生产环境部署策略组合
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: myapp-production
spec:
  replicas: 20

  strategy:
    canary:
      # 第一阶段: 金丝雀(5%)
      steps:
      - setWeight: 5
      - pause: {duration: 10m}
      - analysis:
          templates:
          - templateName: success-rate
          - templateName: latency

      # 第二阶段: 扩大到20%
      - setWeight: 20
      - pause: {duration: 20m}
      - analysis:
          templates:
          - templateName: success-rate
          - templateName: latency

      # 第三阶段: 50% (手动审批)
      - setWeight: 50
      - pause: {}  # 手动审批

      # 第四阶段: 完全切换
      - setWeight: 100

      # 蓝绿切换
      trafficRouting:
        nginx:
          stableIngress: myapp

      # 反亲和性确保分散部署
      antiAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
          labelSelector:
            matchLabels:
              app: myapp
          topologyKey: kubernetes.io/hostname
```

---

## 总结

### 部署策略选择指南

```
┌────────────────────────────────────────────────┐
│          部署策略选择决策树                    │
├────────────────────────────────────────────────┤
│                                                │
│  关键系统?                                     │
│    ├─ 是 → 蓝绿部署 (零停机)                  │
│    └─ 否                                       │
│         │                                      │
│         └─ 需要渐进式验证?                     │
│              ├─ 是 → 金丝雀发布               │
│              └─ 否 → 滚动更新                 │
│                                                │
│  需要A/B测试?                                  │
│    └─ 是 → Istio + 特性开关                   │
│                                                │
│  资源受限?                                     │
│    └─ 是 → 滚动更新                           │
└────────────────────────────────────────────────┘
```

### 关键要点

1. **蓝绿部署**: 快速回滚,适合关键系统
2. **金丝雀**: 渐进式验证,降低风险
3. **滚动更新**: 节省资源,适合日常发布
4. **A/B测试**: 业务验证,需要流量控制
5. **特性开关**: 代码级控制,最灵活

### 下一步学习
- [05_release_management.md](05_release_management.md) - 版本管理
