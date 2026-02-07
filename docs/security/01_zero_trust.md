# 零信任安全架构

## 目录
- [概述](#概述)
- [BeyondCorp零信任模型](#beyondcorp零信任模型)
- [微分段网络架构](#微分段网络架构)
- [身份驱动访问控制](#身份驱动访问控制)
- [实施路线图](#实施路线图)
- [实战案例](#实战案例)

## 概述

### 传统边界安全 vs 零信任

```
传统边界安全模型 (Castle-and-Moat)
┌─────────────────────────────────────────────────────────────┐
│                       外部威胁                               │
│  ┌───────────────────────────────────────────────────┐     │
│  │              防火墙 (Perimeter)                    │     │
│  │  ┌─────────────────────────────────────────────┐ │     │
│  │  │         可信内网 (Trusted Zone)             │ │     │
│  │  │  ┌─────┬─────┬─────┬─────┬─────┐           │ │     │
│  │  │  │用户  │服务器│数据库│应用 │API  │           │ │     │
│  │  │  └─────┴─────┴─────┴─────┴─────┘           │ │     │
│  │  │  所有内部流量默认信任 ✗                     │ │     │
│  │  └─────────────────────────────────────────────┘ │     │
│  └───────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
问题:
- 内部威胁难以防范
- 横向移动容易
- VPN成为单点故障


零信任安全模型 (Zero Trust)
┌─────────────────────────────────────────────────────────────┐
│               "永不信任,始终验证"                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │            身份验证层 (Identity)                   │     │
│  │  ┌─────────────────────────────────────────┐     │     │
│  │  │  MFA  │  Device  │ Location │  Behavior │     │     │
│  │  └───┬───┴────┬─────┴────┬─────┴────┬──────┘     │     │
│  └──────┼────────┼──────────┼──────────┼────────────┘     │
│         │        │          │          │                   │
│  ┌──────▼────────▼──────────▼──────────▼──────────┐       │
│  │          策略引擎 (Policy Engine)              │       │
│  │   - 动态访问决策                               │       │
│  │   - 最小权限原则                               │       │
│  │   - 上下文感知                                 │       │
│  └──────┬─────────┬─────────┬─────────┬───────────┘       │
│         │         │         │         │                   │
│  ┌──────▼─────────▼─────────▼─────────▼──────────┐       │
│  │         微分段 (Micro-segmentation)           │       │
│  │  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐     │       │
│  │  │App1│──│App2│  │App3│  │DB1 │  │API │     │       │
│  │  └────┘  └────┘  └────┘  └────┘  └────┘     │       │
│  │  每个连接都需要验证和授权 ✓                  │       │
│  └───────────────────────────────────────────────┘       │
│                                                            │
│  ┌───────────────────────────────────────────────────┐   │
│  │        持续监控与分析 (Monitoring)                │   │
│  │   - 异常检测                                      │   │
│  │   - 威胁情报                                      │   │
│  │   - 自适应响应                                    │   │
│  └───────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 零信任核心原则

```
1. 验证身份 (Verify Identity)
   ┌────────────────────────────────┐
   │ - 多因素认证 (MFA)              │
   │ - 设备信任状态                  │
   │ - 持续身份验证                  │
   └────────────────────────────────┘

2. 最小权限 (Least Privilege)
   ┌────────────────────────────────┐
   │ - Just-in-Time访问              │
   │ - Just-Enough-Access            │
   │ - 定期权限审查                  │
   └────────────────────────────────┘

3. 假设失陷 (Assume Breach)
   ┌────────────────────────────────┐
   │ - 微分段隔离                    │
   │ - 横向移动检测                  │
   │ - 数据加密                      │
   └────────────────────────────────┘

4. 显式验证 (Explicit Verification)
   ┌────────────────────────────────┐
   │ - 上下文感知                    │
   │ - 风险评分                      │
   │ - 动态策略                      │
   └────────────────────────────────┘

5. 持续监控 (Continuous Monitoring)
   ┌────────────────────────────────┐
   │ - 实时日志分析                  │
   │ - 异常行为检测                  │
   │ - 自动化响应                    │
   └────────────────────────────────┘
```

## BeyondCorp零信任模型

### BeyondCorp架构

```
┌─────────────────────────────────────────────────────────────┐
│                  BeyondCorp Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  终端用户                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  任何设备 (Laptop/Phone/Tablet)                     │   │
│  │  任何位置 (Office/Home/Café)                        │   │
│  │  任何网络 (Corporate/Public/Home)                   │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │ HTTPS                             │
│  ┌──────────────────────▼──────────────────────────────┐   │
│  │         Access Proxy (访问代理)                     │   │
│  │  ┌──────────────────────────────────────────────┐  │   │
│  │  │  - TLS终止                                    │  │   │
│  │  │  - 设备识别                                   │  │   │
│  │  │  - 用户认证                                   │  │   │
│  │  └──────────────────┬───────────────────────────┘  │   │
│  └─────────────────────┼──────────────────────────────┘   │
│                        │                                   │
│  ┌─────────────────────▼──────────────────────────────┐   │
│  │         Trust Inference (信任推断)                 │   │
│  │  ┌──────────────────────────────────────────────┐ │   │
│  │  │  Device Inventory                            │ │   │
│  │  │  ┌────────────────────────────────────────┐  │ │   │
│  │  │  │ - 设备ID                               │  │ │   │
│  │  │  │ - OS版本                               │  │ │   │
│  │  │  │ - 安全状态 (防病毒/防火墙/加密)        │  │ │   │
│  │  │  │ - 合规状态                             │  │ │   │
│  │  │  └────────────────────────────────────────┘  │ │   │
│  │  │                                              │ │   │
│  │  │  User & Group Database                       │ │   │
│  │  │  ┌────────────────────────────────────────┐  │ │   │
│  │  │  │ - 用户ID                               │  │ │   │
│  │  │  │ - 角色/职位                            │  │ │   │
│  │  │  │ - 部门/团队                            │  │ │   │
│  │  │  │ - 访问历史                             │  │ │   │
│  │  │  └────────────────────────────────────────┘  │ │   │
│  │  │                                              │ │   │
│  │  │  Security Policies                           │ │   │
│  │  │  ┌────────────────────────────────────────┐  │ │   │
│  │  │  │ - 资源敏感级别                         │  │ │   │
│  │  │  │ - 访问控制规则                         │  │ │   │
│  │  │  │ - 风险评分模型                         │  │ │   │
│  │  │  └────────────────────────────────────────┘  │ │   │
│  │  └──────────────────┬───────────────────────────┘ │   │
│  └─────────────────────┼─────────────────────────────┘   │
│                        │                                   │
│                        ▼                                   │
│              信任等级计算 (Trust Score)                     │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Trust = f(User, Device, Location, Time, Behavior)  │  │
│  │                                                      │  │
│  │  Score: 0-100                                        │  │
│  │  ├─ 0-30:  Deny                                     │  │
│  │  ├─ 31-60: Challenge (二次验证)                     │  │
│  │  └─ 61-100: Allow                                   │  │
│  └──────────────────────┬──────────────────────────────┘  │
│                         │                                  │
│  ┌──────────────────────▼──────────────────────────────┐  │
│  │      Access Control Engine (访问控制引擎)           │  │
│  │  - 动态访问决策                                      │  │
│  │  - 策略执行                                          │  │
│  │  - 审计日志                                          │  │
│  └──────────────────────┬──────────────────────────────┘  │
│                         │                                  │
│  ┌──────────────────────▼──────────────────────────────┐  │
│  │          Protected Resources                         │  │
│  │  ┌────────┬────────┬────────┬────────┐             │  │
│  │  │ App 1  │ App 2  │ DB     │ API    │             │  │
│  │  └────────┴────────┴────────┴────────┘             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### BeyondCorp实施示例

#### 1. 设备信任评估

```python
"""
设备信任评估系统
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

class TrustLevel(Enum):
    UNKNOWN = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3

@dataclass
class DeviceInfo:
    device_id: str
    os_type: str
    os_version: str
    manufacturer: str
    model: str
    last_seen: datetime
    encryption_enabled: bool
    antivirus_enabled: bool
    firewall_enabled: bool
    os_auto_update: bool
    compliant: bool
    managed_by_mdm: bool

class DeviceTrustEvaluator:
    """设备信任评估器"""

    def evaluate(self, device: DeviceInfo) -> tuple[TrustLevel, int]:
        """
        评估设备信任等级
        返回: (信任等级, 信任分数 0-100)
        """
        score = 0

        # 1. 设备管理 (20分)
        if device.managed_by_mdm:
            score += 20
        elif device.compliant:
            score += 10

        # 2. 安全软件 (30分)
        if device.encryption_enabled:
            score += 10
        if device.antivirus_enabled:
            score += 10
        if device.firewall_enabled:
            score += 10

        # 3. OS更新 (20分)
        if self._is_os_updated(device):
            score += 20
        elif device.os_auto_update:
            score += 10

        # 4. 活跃度 (15分)
        if self._is_recently_active(device):
            score += 15
        elif self._is_moderately_active(device):
            score += 8

        # 5. 合规性 (15分)
        if device.compliant:
            score += 15

        # 确定信任等级
        if score >= 80:
            level = TrustLevel.HIGH
        elif score >= 60:
            level = TrustLevel.MEDIUM
        elif score >= 40:
            level = TrustLevel.LOW
        else:
            level = TrustLevel.UNKNOWN

        return level, score

    def _is_os_updated(self, device: DeviceInfo) -> bool:
        """检查OS是否为最新版本"""
        known_versions = {
            'Windows': '11',
            'macOS': '14',
            'iOS': '17',
            'Android': '14'
        }
        return device.os_version >= known_versions.get(device.os_type, '0')

    def _is_recently_active(self, device: DeviceInfo) -> bool:
        """检查设备是否最近活跃 (24小时内)"""
        return datetime.now() - device.last_seen < timedelta(days=1)

    def _is_moderately_active(self, device: DeviceInfo) -> bool:
        """检查设备是否中度活跃 (7天内)"""
        return datetime.now() - device.last_seen < timedelta(days=7)

# 使用示例
evaluator = DeviceTrustEvaluator()

device = DeviceInfo(
    device_id="device-12345",
    os_type="macOS",
    os_version="14.2",
    manufacturer="Apple",
    model="MacBook Pro",
    last_seen=datetime.now() - timedelta(hours=2),
    encryption_enabled=True,
    antivirus_enabled=True,
    firewall_enabled=True,
    os_auto_update=True,
    compliant=True,
    managed_by_mdm=True
)

level, score = evaluator.evaluate(device)
print(f"Trust Level: {level.name}, Score: {score}")  # HIGH, 100
```

#### 2. 动态访问决策引擎

```python
"""
动态访问决策引擎
"""
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime, time
import ipaddress

@dataclass
class AccessContext:
    """访问上下文"""
    user_id: str
    device_trust_score: int
    user_role: str
    source_ip: str
    location: str
    time: datetime
    resource: str
    action: str
    mfa_verified: bool
    risk_score: int  # 0-100

class AccessDecisionEngine:
    """访问决策引擎"""

    def __init__(self):
        self.policies = self._load_policies()

    def decide(self, context: AccessContext) -> tuple[bool, str]:
        """
        做出访问决策
        返回: (是否允许, 原因)
        """
        # 1. 检查黑名单
        if self._is_blacklisted(context):
            return False, "User or IP is blacklisted"

        # 2. 检查工作时间
        if not self._is_business_hours(context):
            if context.resource in ["production-db", "financial-data"]:
                return False, "Access denied outside business hours"

        # 3. 检查地理位置
        if not self._is_allowed_location(context):
            return False, f"Access from {context.location} not allowed"

        # 4. 检查设备信任
        if context.device_trust_score < 60:
            return False, "Device trust score too low"

        # 5. 检查MFA
        if self._requires_mfa(context) and not context.mfa_verified:
            return False, "MFA verification required"

        # 6. 检查风险评分
        if context.risk_score > 70:
            return False, f"Risk score too high: {context.risk_score}"

        # 7. 检查资源权限
        if not self._has_permission(context):
            return False, "Insufficient permissions"

        # 8. 检查异常行为
        if self._is_anomalous_behavior(context):
            return False, "Anomalous behavior detected"

        return True, "Access granted"

    def _load_policies(self) -> Dict:
        """加载访问策略"""
        return {
            'mfa_required_resources': [
                'production-db',
                'financial-data',
                'customer-pii',
                'admin-panel'
            ],
            'allowed_locations': [
                'China',
                'Singapore',
                'United States'
            ],
            'business_hours': {
                'start': time(9, 0),
                'end': time(18, 0)
            },
            'role_permissions': {
                'admin': ['*'],
                'developer': ['read', 'write', 'deploy'],
                'analyst': ['read'],
                'guest': ['read_public']
            }
        }

    def _is_blacklisted(self, context: AccessContext) -> bool:
        """检查是否在黑名单中"""
        blacklist_ips = ['192.168.1.100', '10.0.0.50']
        blacklist_users = ['suspended_user']
        return (context.source_ip in blacklist_ips or
                context.user_id in blacklist_users)

    def _is_business_hours(self, context: AccessContext) -> bool:
        """检查是否工作时间"""
        current_time = context.time.time()
        business_hours = self.policies['business_hours']
        return business_hours['start'] <= current_time <= business_hours['end']

    def _is_allowed_location(self, context: AccessContext) -> bool:
        """检查地理位置是否允许"""
        return context.location in self.policies['allowed_locations']

    def _requires_mfa(self, context: AccessContext) -> bool:
        """检查是否需要MFA"""
        return context.resource in self.policies['mfa_required_resources']

    def _has_permission(self, context: AccessContext) -> bool:
        """检查用户是否有权限"""
        allowed_actions = self.policies['role_permissions'].get(
            context.user_role, []
        )
        return '*' in allowed_actions or context.action in allowed_actions

    def _is_anomalous_behavior(self, context: AccessContext) -> bool:
        """检查是否异常行为"""
        # 实际实现需要机器学习模型
        # 这里简化处理
        if context.risk_score > 80:
            return True

        # 检查异常IP
        try:
            ip = ipaddress.ip_address(context.source_ip)
            if ip.is_private:
                return False
            # 检查是否来自异常国家
            # (实际需要IP地理位置数据库)
        except ValueError:
            return True

        return False

# 使用示例
engine = AccessDecisionEngine()

# 正常访问
context1 = AccessContext(
    user_id="alice",
    device_trust_score=85,
    user_role="developer",
    source_ip="192.168.1.10",
    location="China",
    time=datetime.now().replace(hour=14),
    resource="api-server",
    action="deploy",
    mfa_verified=True,
    risk_score=20
)

allowed, reason = engine.decide(context1)
print(f"Access: {allowed}, Reason: {reason}")

# 高风险访问
context2 = AccessContext(
    user_id="bob",
    device_trust_score=45,
    user_role="developer",
    source_ip="1.2.3.4",
    location="Unknown",
    time=datetime.now().replace(hour=2),  # 凌晨2点
    resource="production-db",
    action="read",
    mfa_verified=False,
    risk_score=85
)

allowed, reason = engine.decide(context2)
print(f"Access: {allowed}, Reason: {reason}")
```

## 微分段网络架构

### 微分段概念

```
传统VLAN分段
┌─────────────────────────────────────────────────────────────┐
│  VLAN 10 (Web Tier)                                         │
│  ┌──────┬──────┬──────┬──────┬──────┐                      │
│  │Web-1 │Web-2 │Web-3 │Web-4 │Web-5 │                      │
│  └──────┴──────┴──────┴──────┴──────┘                      │
│  互相可通信 ✗                                                │
└─────────────────────────────────────────────────────────────┘

微分段 (Micro-segmentation)
┌─────────────────────────────────────────────────────────────┐
│  每个工作负载独立隔离                                        │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐         │
│  │Web-1 │  │Web-2 │  │Web-3 │  │Web-4 │  │Web-5 │         │
│  └───┬──┘  └───┬──┘  └───┬──┘  └───┬──┘  └───┬──┘         │
│      │         │         │         │         │             │
│      └─────────┴─────────┴─────────┴─────────┘             │
│                       │                                     │
│              ┌────────▼────────┐                            │
│              │  Policy Engine  │                            │
│              │  - App-1 -> DB  │                            │
│              │  - App-2 -> API │                            │
│              └─────────────────┘                            │
│  仅允许必要通信 ✓                                            │
└─────────────────────────────────────────────────────────────┘
```

### Kubernetes网络策略实现微分段

```yaml
# 1. 默认拒绝所有入站流量
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Ingress

---
# 2. 允许前端访问后端API
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-backend
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend-api
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080

---
# 3. 允许后端访问数据库
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-backend-to-database
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: postgresql
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: backend-api
    ports:
    - protocol: TCP
      port: 5432

---
# 4. 允许特定命名空间访问
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-monitoring-namespace
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 9090  # Prometheus

---
# 5. 允许外部流量到Ingress
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-external-to-ingress
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: nginx-ingress
  policyTypes:
  - Ingress
  ingress:
  - from:
    - ipBlock:
        cidr: 0.0.0.0/0
        except:
        - 169.254.0.0/16  # 排除元数据服务
    ports:
    - protocol: TCP
      port: 80
    - protocol: TCP
      port: 443

---
# 6. 限制出站流量 (Egress)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: restrict-egress
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend-api
  policyTypes:
  - Egress
  egress:
  # 允许DNS查询
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53
  # 允许访问数据库
  - to:
    - podSelector:
        matchLabels:
          app: postgresql
    ports:
    - protocol: TCP
      port: 5432
  # 允许访问外部API (特定IP)
  - to:
    - ipBlock:
        cidr: 203.0.113.0/24
    ports:
    - protocol: TCP
      port: 443
```

### Cilium实现高级微分段

```yaml
# Cilium Layer 7 (HTTP) 网络策略
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: l7-http-policy
  namespace: production
spec:
  endpointSelector:
    matchLabels:
      app: backend-api
  ingress:
  - fromEndpoints:
    - matchLabels:
        app: frontend
    toPorts:
    - ports:
      - port: "8080"
        protocol: TCP
      rules:
        http:
        # 只允许GET和POST方法
        - method: "GET"
          path: "/api/v1/.*"
        - method: "POST"
          path: "/api/v1/orders"
          headers:
          - "Content-Type: application/json"

---
# DNS策略 - 限制外部访问
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: dns-egress-policy
  namespace: production
spec:
  endpointSelector:
    matchLabels:
      app: backend-api
  egress:
  - toEndpoints:
    - matchLabels:
        k8s:io.kubernetes.pod.namespace: kube-system
        k8s-app: kube-dns
    toPorts:
    - ports:
      - port: "53"
        protocol: UDP
      rules:
        dns:
        # 只允许访问特定域名
        - matchPattern: "*.company.com"
        - matchPattern: "api.stripe.com"
        - matchPattern: "*.amazonaws.com"

---
# Service Mesh级别的策略
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: service-mesh-policy
  namespace: production
spec:
  endpointSelector:
    matchLabels:
      app: payment-service
  ingress:
  - fromEndpoints:
    - matchLabels:
        app: order-service
    toPorts:
    - ports:
      - port: "9000"
        protocol: TCP
      rules:
        http:
        - method: "POST"
          path: "/v1/payments/charge"
          headers:
          - "Authorization: Bearer .*"
          - "X-Request-ID: .*"
        # 限流
        - method: "GET"
          path: "/v1/payments/status/.*"
        # L7限流需要配合Rate Limiting
```

## 身份驱动访问控制

### OAuth 2.0 + OIDC实现

```python
"""
基于OAuth 2.0和OpenID Connect的身份驱动访问控制
"""
from flask import Flask, request, jsonify, redirect
from functools import wraps
import jwt
import requests
from datetime import datetime, timedelta

app = Flask(__name__)

# 配置
OIDC_PROVIDER = "https://auth.company.com"
CLIENT_ID = "your-client-id"
CLIENT_SECRET = "your-client-secret"
JWT_SECRET = "your-jwt-secret"

class OIDCClient:
    """OpenID Connect客户端"""

    def __init__(self, provider_url, client_id, client_secret):
        self.provider_url = provider_url
        self.client_id = client_id
        self.client_secret = client_secret
        self._discovery_doc = None

    @property
    def discovery_document(self):
        """获取OIDC发现文档"""
        if not self._discovery_doc:
            url = f"{self.provider_url}/.well-known/openid-configuration"
            response = requests.get(url)
            self._discovery_doc = response.json()
        return self._discovery_doc

    def get_authorization_url(self, redirect_uri, state, scope="openid profile email"):
        """获取授权URL"""
        auth_endpoint = self.discovery_document['authorization_endpoint']
        params = {
            'client_id': self.client_id,
            'response_type': 'code',
            'scope': scope,
            'redirect_uri': redirect_uri,
            'state': state
        }
        query = '&'.join(f"{k}={v}" for k, v in params.items())
        return f"{auth_endpoint}?{query}"

    def exchange_code_for_token(self, code, redirect_uri):
        """使用授权码交换令牌"""
        token_endpoint = self.discovery_document['token_endpoint']
        data = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': redirect_uri,
            'client_id': self.client_id,
            'client_secret': self.client_secret
        }
        response = requests.post(token_endpoint, data=data)
        return response.json()

    def get_userinfo(self, access_token):
        """获取用户信息"""
        userinfo_endpoint = self.discovery_document['userinfo_endpoint']
        headers = {'Authorization': f'Bearer {access_token}'}
        response = requests.get(userinfo_endpoint, headers=headers)
        return response.json()

    def verify_token(self, id_token):
        """验证ID Token"""
        jwks_uri = self.discovery_document['jwks_uri']
        # 获取公钥
        jwks = requests.get(jwks_uri).json()
        # 验证签名
        try:
            decoded = jwt.decode(
                id_token,
                jwks,
                algorithms=['RS256'],
                audience=self.client_id,
                issuer=self.provider_url
            )
            return decoded
        except jwt.InvalidTokenError as e:
            return None

oidc_client = OIDCClient(OIDC_PROVIDER, CLIENT_ID, CLIENT_SECRET)

def require_auth(required_scopes=None):
    """认证装饰器"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # 1. 获取Token
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({'error': 'Missing authorization header'}), 401

            token = auth_header.split(' ')[1]

            # 2. 验证Token
            try:
                payload = jwt.decode(
                    token,
                    JWT_SECRET,
                    algorithms=['HS256']
                )
            except jwt.ExpiredSignatureError:
                return jsonify({'error': 'Token expired'}), 401
            except jwt.InvalidTokenError:
                return jsonify({'error': 'Invalid token'}), 401

            # 3. 检查权限
            if required_scopes:
                user_scopes = payload.get('scopes', [])
                if not any(scope in user_scopes for scope in required_scopes):
                    return jsonify({'error': 'Insufficient permissions'}), 403

            # 4. 将用户信息注入到请求上下文
            request.user = payload

            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/login')
def login():
    """登录端点"""
    redirect_uri = "https://app.company.com/callback"
    state = "random_state_string"  # 实际应该生成随机值并存储
    auth_url = oidc_client.get_authorization_url(redirect_uri, state)
    return redirect(auth_url)

@app.route('/callback')
def callback():
    """OAuth回调端点"""
    code = request.args.get('code')
    state = request.args.get('state')

    # 验证state防止CSRF

    # 交换令牌
    redirect_uri = "https://app.company.com/callback"
    tokens = oidc_client.exchange_code_for_token(code, redirect_uri)

    # 验证ID Token
    id_token = tokens.get('id_token')
    user_info = oidc_client.verify_token(id_token)

    if not user_info:
        return jsonify({'error': 'Invalid ID token'}), 401

    # 获取详细用户信息
    access_token = tokens.get('access_token')
    userinfo = oidc_client.get_userinfo(access_token)

    # 生成应用JWT
    app_token = jwt.encode({
        'user_id': userinfo['sub'],
        'email': userinfo['email'],
        'name': userinfo['name'],
        'scopes': userinfo.get('scopes', []),
        'exp': datetime.utcnow() + timedelta(hours=1)
    }, JWT_SECRET, algorithm='HS256')

    return jsonify({
        'token': app_token,
        'user': userinfo
    })

@app.route('/api/public')
def public_endpoint():
    """公开端点"""
    return jsonify({'message': 'This is a public endpoint'})

@app.route('/api/protected')
@require_auth()
def protected_endpoint():
    """受保护端点"""
    return jsonify({
        'message': 'This is a protected endpoint',
        'user': request.user
    })

@app.route('/api/admin')
@require_auth(required_scopes=['admin'])
def admin_endpoint():
    """管理员端点"""
    return jsonify({
        'message': 'This is an admin endpoint',
        'user': request.user
    })

if __name__ == '__main__':
    app.run(debug=True)
```

## 实施路线图

### 零信任实施阶段

```
阶段1: 评估与规划 (1-2个月)
┌────────────────────────────────────────┐
│ 1. 现状评估                            │
│    - 资产清单                          │
│    - 数据流分析                        │
│    - 风险评估                          │
│                                        │
│ 2. 目标设定                            │
│    - 定义零信任策略                    │
│    - 确定优先级                        │
│    - 制定路线图                        │
│                                        │
│ 3. 技术选型                            │
│    - 身份提供商 (Okta/Auth0)          │
│    - 网络安全 (Cilium/Calico)         │
│    - SIEM (Splunk/ELK)                │
└────────────────────────────────────────┘

阶段2: 身份与访问管理 (2-3个月)
┌────────────────────────────────────────┐
│ 1. 部署身份提供商                      │
│    - SSO集成                          │
│    - MFA强制                          │
│    - 用户目录同步                      │
│                                        │
│ 2. 设备管理                            │
│    - MDM部署                          │
│    - 设备清单                          │
│    - 合规检查                          │
│                                        │
│ 3. 访问策略                            │
│    - RBAC规则                         │
│    - 动态策略                          │
│    - JIT访问                          │
└────────────────────────────────────────┘

阶段3: 网络微分段 (3-4个月)
┌────────────────────────────────────────┐
│ 1. 应用分类                            │
│    - 业务关键性                        │
│    - 数据敏感性                        │
│    - 依赖关系                          │
│                                        │
│ 2. 微分段实施                          │
│    - 网络策略                          │
│    - Service Mesh                     │
│    - 防火墙规则                        │
│                                        │
│ 3. 流量监控                            │
│    - 网络可观测性                      │
│    - 异常检测                          │
│    - 自动化响应                        │
└────────────────────────────────────────┘

阶段4: 持续监控与优化 (持续)
┌────────────────────────────────────────┐
│ 1. 安全监控                            │
│    - SIEM集成                         │
│    - 实时告警                          │
│    - 威胁情报                          │
│                                        │
│ 2. 策略优化                            │
│    - 定期审查                          │
│    - 策略调优                          │
│    - 自动化改进                        │
│                                        │
│ 3. 合规审计                            │
│    - 访问审计                          │
│    - 合规报告                          │
│    - 定期演练                          │
└────────────────────────────────────────┘
```

### 实施检查清单

```markdown
## 零信任实施检查清单

### 身份与访问
- [ ] 部署企业级身份提供商
- [ ] 启用MFA for all用户
- [ ] 实施SSO for所有应用
- [ ] 配置RBAC策略
- [ ] 实施JIT访问
- [ ] 定期访问审查

### 设备管理
- [ ] 部署MDM/EMM解决方案
- [ ] 设备清单和分类
- [ ] 设备健康检查
- [ ] 设备合规策略
- [ ] 远程擦除能力
- [ ] BYOD策略

### 网络安全
- [ ] 实施微分段
- [ ] 部署网络策略
- [ ] 配置Service Mesh
- [ ] 启用mTLS
- [ ] 网络流量加密
- [ ] DNS安全

### 数据保护
- [ ] 数据分类
- [ ] 静态数据加密
- [ ] 传输加密
- [ ] DLP策略
- [ ] 备份与恢复
- [ ] 数据访问审计

### 监控与响应
- [ ] 部署SIEM
- [ ] 配置日志聚合
- [ ] 实时告警
- [ ] 异常检测
- [ ] 事件响应计划
- [ ] 定期演练

### 合规与审计
- [ ] 合规要求映射
- [ ] 审计日志保留
- [ ] 定期安全评估
- [ ] 渗透测试
- [ ] 合规报告
- [ ] 策略文档
```

## 实战案例

### 案例:金融公司零信任改造

```
背景:
- 员工: 5000人
- 应用: 200+ 内部系统
- 数据中心: 混合云 (自有+AWS)
- 合规要求: SOC2, ISO27001

挑战:
- VPN单点故障
- 内部网络全信任
- 远程办公安全风险
- 第三方供应商访问管理

解决方案:
┌─────────────────────────────────────────────────────────────┐
│                     零信任架构                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 身份层                                                    │
│     - Okta: 统一身份认证                                      │
│     - MFA: 强制多因素认证                                     │
│     - SCIM: 自动账号生命周期管理                              │
│                                                              │
│  2. 访问层                                                    │
│     - Cloudflare Access: 应用访问代理                         │
│     - Dynamic Access Policies: 基于上下文的访问控制           │
│     - JIT: Just-in-Time特权访问                              │
│                                                              │
│  3. 网络层                                                    │
│     - Cilium: Kubernetes网络策略                             │
│     - AWS Security Groups: 云资源隔离                        │
│     - mTLS: Service-to-Service加密                          │
│                                                              │
│  4. 监控层                                                    │
│     - Splunk: SIEM和日志分析                                 │
│     - CrowdStrike: 端点检测与响应                            │
│     - Datadog: 基础设施监控                                  │
└─────────────────────────────────────────────────────────────┘

实施结果:
✓ VPN使用率降低 90%
✓ 安全事件响应时间减少 70%
✓ 远程访问体验提升 50%
✓ 第三方访问审计100%可追踪
✓ 通过SOC2 Type II审计
```

## 总结

零信任安全的核心要点:

1. **核心原则**
   - 永不信任,始终验证
   - 最小权限访问
   - 假设网络已失陷

2. **关键技术**
   - 强身份认证 (MFA, SSO)
   - 微分段网络
   - 加密通信 (mTLS)
   - 持续监控

3. **实施建议**
   - 分阶段推进
   - 先易后难
   - 持续优化
   - 培训员工

4. **常见挑战**
   - 遗留系统改造
   - 用户体验平衡
   - 策略复杂度
   - 运维成本

零信任不是产品,而是一种安全理念和架构方法论。
