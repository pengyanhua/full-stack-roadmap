# 多云与混合云策略

## 目录
- [多云策略概述](#多云策略概述)
- [多云架构模式](#多云架构模式)
- [云迁移策略](#云迁移策略)
- [多云管理](#多云管理)
- [实战案例](#实战案例)

---

## 多云策略概述

### 为什么选择多云

```
┌────────────────────────────────────────────────────┐
│          采用多云策略的主要原因                    │
├────────────────────────────────────────────────────┤
│                                                    │
│  1. 🔒 避免供应商锁定 (Vendor Lock-in)            │
│     降低对单一云厂商的依赖                         │
│                                                    │
│  2. 🌍 地理覆盖优化                                │
│     利用不同云商的全球数据中心                     │
│                                                    │
│  3. 💰 成本优化                                    │
│     选择性价比最高的服务                           │
│                                                    │
│  4. 🛡️  灾难恢复                                   │
│     跨云备份，提高可用性                           │
│                                                    │
│  5. ⚡ 服务优势互补                                │
│     AWS ML + GCP 大数据 + Azure 企业集成          │
│                                                    │
│  6. 📜 合规要求                                    │
│     满足数据主权和合规性要求                       │
└────────────────────────────────────────────────────┘
```

### 多云 vs 混合云

```
┌──────────────────────────────────────────────────────┐
│              多云 vs 混合云对比                      │
├──────────────┬───────────────┬───────────────────────┤
│   特性       │    多云       │      混合云           │
├──────────────┼───────────────┼───────────────────────┤
│ 定义         │ 多个公有云    │ 公有云 + 私有云       │
│ 部署位置     │ 全部云端      │ 云端 + 本地           │
│ 主要目标     │ 避免锁定      │ 渐进式迁移            │
│ 数据位置     │ 分散在云端    │ 敏感数据本地          │
│ 复杂度       │ 高 ⭐⭐⭐    │ 极高 ⭐⭐⭐⭐         │
│ 成本         │ 较高          │ 最高                  │
│ 适用行业     │ 互联网公司    │ 金融、政府            │
│ 典型架构     │ 主备          │ 分层                  │
└──────────────┴───────────────┴───────────────────────┘
```

---

## 多云架构模式

### 1. 多云备份模式 (Redundancy)

**场景**：业务关键系统，需要最高可用性

```
┌────────────────────────────────────────────────────┐
│           主备多云架构（Active-Passive）           │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌─────────────┐         流量分发                 │
│  │  用户请求   │              │                    │
│  └──────┬──────┘              │                    │
│         │                     │                    │
│    ┌────▼────────────────────▼──────┐            │
│    │    Global Load Balancer        │            │
│    │    (Route 53 / Cloud DNS)      │            │
│    └────┬────────────────────┬──────┘            │
│         │ 100%流量           │ 0%流量(备用)      │
│         │                    │                    │
│  ┌──────▼──────┐      ┌──────▼──────┐           │
│  │   AWS       │      │   GCP       │           │
│  │  (主区域)   │      │  (备用)     │           │
│  ├─────────────┤      ├─────────────┤           │
│  │ • EC2       │      │ • GCE       │           │
│  │ • RDS       │◀同步─│ • Cloud SQL │           │
│  │ • S3        │      │ • GCS       │           │
│  └─────────────┘      └─────────────┘           │
│                                                    │
│  RTO: 5-15分钟   RPO: <5分钟                     │
└────────────────────────────────────────────────────┘
```

**Terraform 实现**:
```hcl
# multi_cloud_backup.tf

# ========== AWS 主区域 ==========
provider "aws" {
  region = "us-east-1"
  alias  = "primary"
}

resource "aws_instance" "primary_app" {
  provider      = aws.primary
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.medium"

  tags = {
    Name        = "primary-app"
    Environment = "production"
    Cloud       = "AWS"
  }
}

resource "aws_db_instance" "primary_db" {
  provider             = aws.primary
  identifier           = "primary-db"
  engine               = "postgres"
  instance_class       = "db.t3.medium"
  allocated_storage    = 100
  backup_retention_period = 7

  # 跨区域备份
  backup_target = "region"
}

# ========== GCP 备用区域 ==========
provider "google" {
  project = "my-project"
  region  = "us-central1"
  alias   = "backup"
}

resource "google_compute_instance" "backup_app" {
  provider     = google.backup
  name         = "backup-app"
  machine_type = "n1-standard-2"
  zone         = "us-central1-a"

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
    }
  }

  tags = ["backup", "standby"]
}

resource "google_sql_database_instance" "backup_db" {
  provider         = google.backup
  name             = "backup-db"
  database_version = "POSTGRES_14"

  settings {
    tier = "db-n1-standard-2"

    backup_configuration {
      enabled            = true
      start_time         = "03:00"
      point_in_time_recovery_enabled = true
    }
  }
}

# ========== Route 53 故障转移 ==========
resource "aws_route53_health_check" "primary" {
  provider          = aws.primary
  fqdn              = aws_instance.primary_app.public_dns
  port              = 80
  type              = "HTTP"
  resource_path     = "/health"
  failure_threshold = 3
  request_interval  = 30
}

resource "aws_route53_record" "primary" {
  provider = aws.primary
  zone_id  = aws_route53_zone.main.zone_id
  name     = "app.example.com"
  type     = "A"
  ttl      = 60

  failover_routing_policy {
    type = "PRIMARY"
  }

  set_identifier  = "primary"
  health_check_id = aws_route53_health_check.primary.id
  records         = [aws_instance.primary_app.public_ip]
}

resource "aws_route53_record" "backup" {
  provider = aws.primary
  zone_id  = aws_route53_zone.main.zone_id
  name     = "app.example.com"
  type     = "A"
  ttl      = 60

  failover_routing_policy {
    type = "SECONDARY"
  }

  set_identifier = "backup"
  records        = [google_compute_instance.backup_app.network_interface[0].access_config[0].nat_ip]
}
```

### 2. 多云分布式模式 (Active-Active)

**场景**：全球用户，需要最低延迟

```
┌────────────────────────────────────────────────────┐
│         主主多云架构（Active-Active）              │
├────────────────────────────────────────────────────┤
│                                                    │
│               Global Traffic Manager               │
│     (基于地理位置和负载的智能路由)                 │
│               ┌─────┴─────┐                        │
│               │           │                        │
│       欧洲用户 50%    美国用户 50%                 │
│               │           │                        │
│      ┌────────▼──┐   ┌───▼────────┐              │
│      │  AWS EU   │   │  GCP US    │              │
│      │ (法兰克福)│   │ (爱荷华)   │              │
│      ├───────────┤   ├────────────┤              │
│      │ • EKS     │   │ • GKE      │              │
│      │ • Aurora  │◀──│ • Spanner  │  双向同步    │
│      │ • S3      │──▶│ • GCS      │              │
│      └───────────┘   └────────────┘              │
│                                                    │
│      亚洲用户                                      │
│           │                                        │
│      ┌────▼────────┐                              │
│      │ Azure Asia  │                              │
│      │  (新加坡)   │                              │
│      ├─────────────┤                              │
│      │ • AKS       │                              │
│      │ • Cosmos DB │◀─┐                          │
│      │ • Blob      │  │ 多向同步                  │
│      └─────────────┘  │                          │
│            ▲──────────┘                          │
│                                                    │
│  优势: 低延迟、高可用   劣势: 数据一致性复杂      │
└────────────────────────────────────────────────────┘
```

### 3. 多云服务聚合模式 (Best-of-Breed)

**场景**：利用各云平台的最佳服务

```
┌────────────────────────────────────────────────────┐
│         多云服务聚合架构                           │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌─────────────────────────────────────────┐     │
│  │         AWS (计算与存储)                │     │
│  │  ┌────────┐  ┌────────┐  ┌─────────┐   │     │
│  │  │  EKS   │  │   S3   │  │ Lambda  │   │     │
│  │  └───┬────┘  └────────┘  └─────────┘   │     │
│  └──────┼───────────────────────────────────┘     │
│         │                                          │
│         ├───────────────────┐                     │
│         │                   │                      │
│  ┌──────▼──────────┐  ┌─────▼──────────────┐    │
│  │  GCP (大数据)   │  │  Azure (企业集成)  │    │
│  │  ┌───────────┐  │  │  ┌──────────────┐  │    │
│  │  │  BigQuery │  │  │  │  AD / Entra  │  │    │
│  │  ├───────────┤  │  │  ├──────────────┤  │    │
│  │  │  Dataflow │  │  │  │  Logic Apps  │  │    │
│  │  ├───────────┤  │  │  ├──────────────┤  │    │
│  │  │ Vertex AI │  │  │  │  Power BI    │  │    │
│  │  └───────────┘  │  │  └──────────────┘  │    │
│  └─────────────────┘  └─────────────────────┘    │
│                                                    │
│  示例: Spotify 使用 GCP 数据分析 + AWS 服务       │
└────────────────────────────────────────────────────┘
```

---

## 云迁移策略

### 迁移波次规划

```
┌────────────────────────────────────────────────────┐
│          云迁移波次示例（6个月计划）               │
├────────────────────────────────────────────────────┤
│                                                    │
│  Phase 1: 基础设施准备 (Week 1-4)                 │
│  ├─ 建立云账户和网络                              │
│  ├─ 配置 VPN/专线                                 │
│  ├─ 设置 IAM 和安全策略                           │
│  └─ 迁移非关键开发环境                            │
│                                                    │
│  Phase 2: 第一波应用迁移 (Week 5-8)               │
│  ├─ 内部工具系统 (低风险)                         │
│  ├─ 测试环境                                      │
│  └─ 建立监控和日志系统                            │
│                                                    │
│  Phase 3: 第二波应用迁移 (Week 9-16)              │
│  ├─ 边缘业务系统                                  │
│  ├─ 数据仓库 / BI 系统                            │
│  └─ 异步处理任务                                  │
│                                                    │
│  Phase 4: 核心系统迁移 (Week 17-24)               │
│  ├─ 核心业务 API                                  │
│  ├─ 主数据库（读写分离优先）                      │
│  ├─ 实时交易系统                                  │
│  └─ 完整切流验证                                  │
│                                                    │
│  Phase 5: 优化与下线 (Week 25-26)                 │
│  ├─ 性能调优                                      │
│  ├─ 成本优化                                      │
│  └─ 下线本地机房                                  │
└────────────────────────────────────────────────────┘
```

### 迁移风险矩阵

```
┌──────────────────────────────────────────────────┐
│           应用迁移风险评估矩阵                   │
├────────────┬───────────┬────────────┬───────────┤
│ 应用       │  业务重要性│  技术复杂度│ 迁移策略  │
├────────────┼───────────┼────────────┼───────────┤
│ 官网       │  低       │  低        │ Rehost    │
│ 内部工具   │  低       │  中        │ Replatform│
│ 数据分析   │  中       │  低        │ Refactor  │
│ 支付系统   │  极高     │  高        │ 保留本地  │
│ 订单系统   │  高       │  中        │ Replatform│
│ CRM        │  中       │  低        │ Repurchase│
└────────────┴───────────┴────────────┴───────────┘
```

---

## 多云管理

### 多云管理平台

```
┌────────────────────────────────────────────────────┐
│          多云管理工具对比                          │
├──────────────┬─────────────────────────────────────┤
│ 工具         │  特点                               │
├──────────────┼─────────────────────────────────────┤
│ Terraform    │ • 开源 IaC 工具                     │
│              │ • 支持 100+ 云服务商                │
│              │ • 声明式配置                        │
│              │                                     │
│ Pulumi       │ • 使用编程语言编写                  │
│              │ • 强类型检查                        │
│              │ • 适合开发者                        │
│              │                                     │
│ Crossplane   │ • Kubernetes 原生                   │
│              │ • 声明式 API                        │
│              │ • 适合云原生团队                    │
│              │                                     │
│ Cloudify     │ • 企业级编排                        │
│              │ • 工作流引擎                        │
│              │ • 可视化设计                        │
│              │                                     │
│ VMware Aria  │ • 混合云管理                        │
│              │ • vSphere 集成                      │
│              │ • 成本优化                          │
└──────────────┴─────────────────────────────────────┘
```

### 统一监控方案

```python
# multi_cloud_monitoring.py
import boto3
from google.cloud import monitoring_v3
from azure.monitor.query import MetricsQueryClient

class MultiCloudMonitor:
    """多云统一监控"""

    def __init__(self):
        # AWS CloudWatch
        self.cloudwatch = boto3.client('cloudwatch')

        # GCP Cloud Monitoring
        self.gcp_client = monitoring_v3.MetricServiceClient()

        # Azure Monitor
        self.azure_client = MetricsQueryClient(credential)

    def get_all_cpu_metrics(self):
        """获取所有云平台的 CPU 指标"""
        metrics = {}

        # AWS
        metrics['aws'] = self._get_aws_cpu()

        # GCP
        metrics['gcp'] = self._get_gcp_cpu()

        # Azure
        metrics['azure'] = self._get_azure_cpu()

        return metrics

    def _get_aws_cpu(self):
        """获取 AWS EC2 CPU 使用率"""
        response = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName='CPUUtilization',
            Dimensions=[{
                'Name': 'InstanceId',
                'Value': 'i-1234567890abcdef0'
            }],
            StartTime=datetime.utcnow() - timedelta(minutes=5),
            EndTime=datetime.utcnow(),
            Period=300,
            Statistics=['Average']
        )

        return {
            'provider': 'AWS',
            'value': response['Datapoints'][0]['Average'],
            'unit': 'Percent'
        }

    def _get_gcp_cpu(self):
        """获取 GCP GCE CPU 使用率"""
        project_name = f"projects/{PROJECT_ID}"

        interval = monitoring_v3.TimeInterval({
            'end_time': {'seconds': int(time.time())},
            'start_time': {'seconds': int(time.time()) - 300}
        })

        results = self.gcp_client.list_time_series(
            request={
                'name': project_name,
                'filter': 'metric.type="compute.googleapis.com/instance/cpu/utilization"',
                'interval': interval
            }
        )

        for result in results:
            return {
                'provider': 'GCP',
                'value': result.points[0].value.double_value * 100,
                'unit': 'Percent'
            }

    def _get_azure_cpu(self):
        """获取 Azure VM CPU 使用率"""
        response = self.azure_client.query_resource(
            resource_id=AZURE_VM_ID,
            metric_names=["Percentage CPU"],
            timespan=timedelta(minutes=5)
        )

        return {
            'provider': 'Azure',
            'value': response.metrics[0].timeseries[0].data[0].average,
            'unit': 'Percent'
        }

# 使用示例
monitor = MultiCloudMonitor()
all_metrics = monitor.get_all_cpu_metrics()

for cloud, metric in all_metrics.items():
    print(f"{cloud.upper()}: CPU {metric['value']:.2f}%")
```

---

## 实战案例

### 案例: 金融企业混合云架构

```
┌────────────────────────────────────────────────────┐
│         某银行混合云架构实战                       │
├────────────────────────────────────────────────────┤
│                                                    │
│  互联网区域（公有云 - 阿里云）                     │
│  ┌──────────────────────────────────────┐         │
│  │  ┌────────┐  ┌─────────┐  ┌───────┐ │         │
│  │  │ WAF    │─▶│ SLB     │─▶│ ECS   │ │         │
│  │  └────────┘  └─────────┘  │ Web   │ │         │
│  │                            └───┬───┘ │         │
│  │  ┌────────────────────────────┘     │         │
│  │  │                                   │         │
│  │  ▼                                   │         │
│  │  API Gateway                         │         │
│  └──┬──────────────────────────────────┘         │
│     │ 专线 (100Mbps)                             │
│  ───┼──────────────────────────────────          │
│     │                                             │
│  核心区域（私有云 - 本地机房）                    │
│  ┌──▼──────────────────────────────────┐         │
│  │  ┌──────────┐  ┌──────────────────┐│         │
│  │  │ 防火墙   │─▶│  应用服务器       ││         │
│  │  └──────────┘  │  (核心业务逻辑)   ││         │
│  │                └───────┬───────────┘│         │
│  │                        │             │         │
│  │  ┌─────────────────────┼─────────┐  │         │
│  │  │     数据库集群       │         │  │         │
│  │  │  ┌────────┐  ┌──────▼─────┐  │  │         │
│  │  │  │ Oracle │  │  MySQL     │  │  │         │
│  │  │  │ RAC    │  │  主从      │  │  │         │
│  │  │  └────────┘  └────────────┘  │  │         │
│  │  └──────────────────────────────┘  │         │
│  └─────────────────────────────────────┘         │
│                                                    │
│  数据同步: 单向（核心→云端）                      │
│  安全: 双因素认证 + 堡垒机 + 加密传输             │
│  合规: 通过等保三级                               │
└────────────────────────────────────────────────────┘
```

### 迁移收益

```
┌────────────────────────────────────────────────────┐
│          某企业多云迁移ROI分析                     │
├────────────────────────────────────────────────────┤
│                                                    │
│  迁移前（单一数据中心）                            │
│  • 服务器成本: $500K/年                           │
│  • 人力运维: $200K/年                             │
│  • 电力冷却: $100K/年                             │
│  • 总计: $800K/年                                 │
│                                                    │
│  迁移后（AWS + GCP多云）                          │
│  • 云服务成本: $400K/年                           │
│  • 人力成本: $150K/年                             │
│  • 多云管理工具: $20K/年                          │
│  • 总计: $570K/年                                 │
│                                                    │
│  节省: $230K/年 (29%)                             │
│                                                    │
│  非财务收益:                                       │
│  ✅ 部署速度提升 10倍                             │
│  ✅ 可用性从 99.5% → 99.95%                       │
│  ✅ 全球延迟降低 60%                              │
└────────────────────────────────────────────────────┘
```

---

## 总结

### 多云决策树

```
    是否需要多云？
         │
    ┌────┴────┐
    │         │
   是         否
    │      (单云更简单)
    ▼
 主要目标？
    │
  ┌─┴──────┬─────────┐
  │        │         │
避免锁定  灾备    最佳服务
  │        │         │
  ▼        ▼         ▼
主备模式  双活    服务聚合
```

### 关键建议

1. **谨慎选择**：多云复杂度高，需权衡收益与成本
2. **抽象层**：使用 Terraform/K8s 等工具减少供应商锁定
3. **数据策略**：明确数据主权和同步策略
4. **团队技能**：确保团队具备多云管理能力
5. **成本控制**：建立 FinOps 流程，持续优化

### 下一步学习

- [05_cost_optimization.md](05_cost_optimization.md) - 深度成本优化
