# 云成本优化

## 目录
- [成本优化概述](#成本优化概述)
- [成本分析](#成本分析)
- [优化策略](#优化策略)
- [FinOps实践](#finops实践)
- [自动化工具](#自动化工具)

---

## 成本优化概述

### 云成本浪费的主要原因

```
┌────────────────────────────────────────────────────┐
│          云成本浪费TOP 10                          │
├────────────────────────────────────────────────────┤
│                                                    │
│  1. 💤 闲置资源 (35%)                             │
│     未关闭的开发/测试环境                          │
│                                                    │
│  2. 📊 过度配置 (30%)                             │
│     实例规格远超实际需求                           │
│                                                    │
│  3. 🗄️  低效存储 (15%)                            │
│     未使用存储分层、快照未清理                     │
│                                                    │
│  4. 🌐 数据传输费用 (8%)                          │
│     跨区域/跨云传输                                │
│                                                    │
│  5. 🔄 未使用预留实例 (5%)                        │
│     长期运行资源未购买RI                           │
│                                                    │
│  6. 📡 孤立资源 (3%)                              │
│     EBS卷、EIP、负载均衡器                        │
│                                                    │
│  7. 🔍 缺乏监控 (2%)                              │
│     无成本告警机制                                 │
│                                                    │
│  8. 🏗️  架构设计不当 (1%)                         │
│     未使用Serverless等低成本方案                   │
│                                                    │
│  9. 🔑 权限管理混乱 (0.5%)                        │
│     开发人员随意创建资源                           │
│                                                    │
│ 10. 📅 未定期审查 (0.5%)                          │
│     成本优化非持续性工作                           │
└────────────────────────────────────────────────────┘
```

### 成本优化金字塔

```
┌────────────────────────────────────────────────────┐
│              云成本优化金字塔                      │
├────────────────────────────────────────────────────┤
│                                                    │
│                    ╱╲                              │
│                   ╱  ╲                             │
│                  ╱ 文化 ╲    (5%)                  │
│                 ╱  变革  ╲   FinOps团队            │
│                ╱──────────╲                        │
│               ╱            ╲                       │
│              ╱   架构优化   ╲  (15%)               │
│             ╱  Serverless   ╲ 重构应用             │
│            ╱────────────────╲                      │
│           ╱                  ╲                     │
│          ╱    资源优化        ╲ (30%)              │
│         ╱   Right Sizing      ╲ 预留实例           │
│        ╱──────────────────────╲                    │
│       ╱                        ╲                   │
│      ╱     清理浪费资源         ╲ (50%)            │
│     ╱   关闭闲置/删除孤立资源    ╲ 低垂的果实      │
│    ╱──────────────────────────────╲               │
│                                                    │
│  建议: 从底层开始，快速见效！                      │
└────────────────────────────────────────────────────┘
```

---

## 成本分析

### AWS成本分析脚本

```python
# aws_cost_analyzer.py
import boto3
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd

class AWSCostAnalyzer:
    """AWS 成本分析器"""

    def __init__(self, region='us-east-1'):
        self.ce = boto3.client('ce', region_name=region)
        self.ec2 = boto3.client('ec2', region_name=region)
        self.rds = boto3.client('rds', region_name=region)
        self.s3 = boto3.client('s3')

    def get_monthly_cost_by_service(self, months=3):
        """获取最近N个月各服务成本"""
        end = datetime.now()
        start = end - timedelta(days=months * 30)

        response = self.ce.get_cost_and_usage(
            TimePeriod={
                'Start': start.strftime('%Y-%m-%d'),
                'End': end.strftime('%Y-%m-%d')
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
        )

        # 解析数据
        data = []
        for result in response['ResultsByTime']:
            month = result['TimePeriod']['Start']
            for group in result['Groups']:
                service = group['Keys'][0]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                if cost > 0:
                    data.append({
                        'Month': month,
                        'Service': service,
                        'Cost': cost
                    })

        df = pd.DataFrame(data)

        # 透视表
        pivot = df.pivot_table(
            index='Service',
            columns='Month',
            values='Cost',
            aggfunc='sum',
            fill_value=0
        )

        # 计算增长率
        if len(pivot.columns) >= 2:
            pivot['Growth'] = ((pivot.iloc[:, -1] - pivot.iloc[:, -2]) /
                               pivot.iloc[:, -2] * 100)

        return pivot.sort_values(by=pivot.columns[-1], ascending=False)

    def find_idle_ec2_instances(self):
        """查找闲置EC2实例（CPU<5%）"""
        cloudwatch = boto3.client('cloudwatch')
        idle_instances = []

        # 获取所有运行中的实例
        response = self.ec2.describe_instances(
            Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
        )

        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instance_id = instance['InstanceId']
                instance_type = instance['InstanceType']

                # 获取过去7天的CPU使用率
                metrics = cloudwatch.get_metric_statistics(
                    Namespace='AWS/EC2',
                    MetricName='CPUUtilization',
                    Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                    StartTime=datetime.utcnow() - timedelta(days=7),
                    EndTime=datetime.utcnow(),
                    Period=3600,  # 1小时
                    Statistics=['Average']
                )

                if metrics['Datapoints']:
                    avg_cpu = sum(p['Average'] for p in metrics['Datapoints']) / len(metrics['Datapoints'])

                    if avg_cpu < 5:  # CPU < 5%
                        # 计算每月成本
                        monthly_cost = self._estimate_instance_cost(instance_type)

                        idle_instances.append({
                            'InstanceId': instance_id,
                            'InstanceType': instance_type,
                            'AvgCPU': f"{avg_cpu:.2f}%",
                            'MonthlyCost': f"${monthly_cost:.2f}",
                            'Tags': {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                        })

        return pd.DataFrame(idle_instances)

    def _estimate_instance_cost(self, instance_type):
        """估算EC2实例月成本（简化版）"""
        # 简化的价格表（实际应使用AWS Pricing API）
        prices = {
            't2.micro': 8.47,
            't2.small': 16.79,
            't2.medium': 33.58,
            't3.medium': 30.37,
            't3.large': 60.74,
            'm5.large': 69.35,
            'm5.xlarge': 138.70,
            'c5.large': 61.37,
            'c5.xlarge': 122.74
        }
        return prices.get(instance_type, 100)  # 默认$100

    def find_unattached_ebs_volumes(self):
        """查找未挂载的EBS卷"""
        response = self.ec2.describe_volumes(
            Filters=[{'Name': 'status', 'Values': ['available']}]
        )

        unattached = []
        for volume in response['Volumes']:
            size_gb = volume['Size']
            volume_type = volume['VolumeType']

            # 估算月成本
            cost_per_gb = {
                'gp2': 0.10,
                'gp3': 0.08,
                'io1': 0.125,
                'io2': 0.125,
                'st1': 0.045,
                'sc1': 0.015
            }.get(volume_type, 0.10)

            monthly_cost = size_gb * cost_per_gb

            unattached.append({
                'VolumeId': volume['VolumeId'],
                'Size': f"{size_gb} GB",
                'Type': volume_type,
                'CreateTime': volume['CreateTime'].strftime('%Y-%m-%d'),
                'MonthlyCost': f"${monthly_cost:.2f}"
            })

        return pd.DataFrame(unattached)

    def find_unassociated_eips(self):
        """查找未关联的弹性IP"""
        response = self.ec2.describe_addresses()

        unassociated = []
        for address in response['Addresses']:
            if 'InstanceId' not in address:  # 未关联
                unassociated.append({
                    'AllocationId': address['AllocationId'],
                    'PublicIp': address['PublicIp'],
                    'MonthlyCost': '$3.60'  # AWS固定费用
                })

        return pd.DataFrame(unassociated)

    def analyze_s3_storage_class(self):
        """分析S3存储类别优化机会"""
        opportunities = []

        for bucket in self.s3.list_buckets()['Buckets']:
            bucket_name = bucket['Name']

            try:
                # 获取对象列表
                paginator = self.s3.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket_name)

                total_size = 0
                old_objects = 0  # 90天未访问

                for page in pages:
                    if 'Contents' not in page:
                        continue

                    for obj in page['Contents']:
                        size = obj['Size']
                        last_modified = obj['LastModified']
                        storage_class = obj.get('StorageClass', 'STANDARD')

                        total_size += size

                        # 90天未修改的对象
                        age_days = (datetime.now(last_modified.tzinfo) - last_modified).days
                        if age_days > 90 and storage_class == 'STANDARD':
                            old_objects += 1

                if old_objects > 0:
                    # 计算节省成本
                    size_gb = total_size / (1024**3)
                    current_cost = size_gb * 0.023  # STANDARD
                    glacier_cost = size_gb * 0.004  # GLACIER
                    savings = (current_cost - glacier_cost) * (old_objects / (total_size / size_gb))

                    opportunities.append({
                        'Bucket': bucket_name,
                        'TotalSize': f"{size_gb:.2f} GB",
                        'OldObjects': old_objects,
                        'PotentialSavings': f"${savings:.2f}/month"
                    })

            except Exception as e:
                print(f"跳过桶 {bucket_name}: {e}")

        return pd.DataFrame(opportunities)

    def generate_report(self):
        """生成完整的成本优化报告"""
        print("=" * 60)
        print("AWS 成本优化报告")
        print("=" * 60)

        # 1. 月度成本趋势
        print("\n📊 月度成本（按服务）")
        print(self.get_monthly_cost_by_service())

        # 2. 闲置EC2实例
        print("\n💤 闲置 EC2 实例（CPU < 5%）")
        idle_ec2 = self.find_idle_ec2_instances()
        if not idle_ec2.empty:
            print(idle_ec2)
            print(f"\n潜在节省: ${idle_ec2['MonthlyCost'].str.replace('$', '').astype(float).sum():.2f}/月")
        else:
            print("✅ 未发现闲置实例")

        # 3. 未挂载EBS卷
        print("\n🗄️  未挂载的 EBS 卷")
        unattached_ebs = self.find_unattached_ebs_volumes()
        if not unattached_ebs.empty:
            print(unattached_ebs)
            print(f"\n潜在节省: ${unattached_ebs['MonthlyCost'].str.replace('$', '').astype(float).sum():.2f}/月")
        else:
            print("✅ 未发现未挂载卷")

        # 4. 未关联EIP
        print("\n📡 未关联的弹性 IP")
        unassociated_eips = self.find_unassociated_eips()
        if not unassociated_eips.empty:
            print(unassociated_eips)
            print(f"\n潜在节省: ${len(unassociated_eips) * 3.60:.2f}/月")
        else:
            print("✅ 未发现未关联 EIP")

        # 5. S3存储优化
        print("\n🗂️  S3 存储类别优化机会")
        s3_opportunities = self.analyze_s3_storage_class()
        if not s3_opportunities.empty:
            print(s3_opportunities)
        else:
            print("✅ 未发现优化机会")

# 使用示例
if __name__ == '__main__':
    analyzer = AWSCostAnalyzer()
    analyzer.generate_report()
```

---

## 优化策略

### 1. 实例规格优化（Right Sizing）

```python
# right_sizing.py
import boto3
from datetime import datetime, timedelta

class RightSizingRecommender:
    """EC2 实例规格推荐"""

    def __init__(self):
        self.ec2 = boto3.client('ec2')
        self.cloudwatch = boto3.client('cloudwatch')
        self.compute_optimizer = boto3.client('compute-optimizer')

    def get_recommendations(self):
        """获取 AWS Compute Optimizer 推荐"""
        try:
            response = self.compute_optimizer.get_ec2_instance_recommendations()

            recommendations = []
            for rec in response['instanceRecommendations']:
                instance_arn = rec['instanceArn']
                current_type = rec['currentInstanceType']

                # 推荐选项
                options = rec['recommendationOptions']
                if options:
                    best_option = options[0]  # 第一个通常是最佳推荐

                    recommendations.append({
                        'InstanceId': instance_arn.split('/')[-1],
                        'Current': current_type,
                        'Recommended': best_option['instanceType'],
                        'CurrentCost': rec.get('currentPerformanceRisk', 'N/A'),
                        'EstimatedSavings': f"${best_option.get('estimatedMonthlySavings', {}).get('value', 0):.2f}",
                        'Reason': self._format_reason(best_option)
                    })

            return recommendations

        except Exception as e:
            print(f"错误: {e}")
            return []

    def _format_reason(self, option):
        """格式化推荐原因"""
        utilization = option.get('projectedUtilizationMetrics', [])
        reasons = []

        for metric in utilization:
            name = metric['name']
            value = metric['statistic']

            if name == 'CPU' and float(value) < 40:
                reasons.append(f"CPU低({value}%)")
            elif name == 'MEMORY' and float(value) < 40:
                reasons.append(f"内存低({value}%)")

        return ', '.join(reasons) if reasons else '正常'

# 使用示例
recommender = RightSizingRecommender()
recs = recommender.get_recommendations()

for rec in recs:
    print(f"{rec['InstanceId']}: {rec['Current']} → {rec['Recommended']}")
    print(f"  节省: {rec['EstimatedSavings']}, 原因: {rec['Reason']}\n")
```

### 2. 自动化启停策略

```python
# auto_start_stop.py
import boto3
from datetime import datetime

def lambda_handler(event, context):
    """
    Lambda 函数：自动启停 EC2 实例
    配合 EventBridge 定时触发

    标签规则:
    - AutoStop: true (启用自动停止)
    - Schedule: weekdays-9to18 (工作日9点到18点)
    """

    ec2 = boto3.client('ec2')

    # 获取当前时间
    now = datetime.now()
    hour = now.hour
    weekday = now.weekday()  # 0=周一, 6=周日

    print(f"当前时间: {now}, 星期{weekday+1}, {hour}点")

    # 获取所有带自动停止标签的实例
    response = ec2.describe_instances(
        Filters=[
            {'Name': 'tag:AutoStop', 'Values': ['true']},
            {'Name': 'instance-state-name', 'Values': ['running', 'stopped']}
        ]
    )

    instances_to_stop = []
    instances_to_start = []

    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            instance_id = instance['InstanceId']
            state = instance['State']['Name']
            tags = {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}

            schedule = tags.get('Schedule', 'weekdays-9to18')

            # 解析时间表
            should_run = parse_schedule(schedule, weekday, hour)

            if should_run and state == 'stopped':
                instances_to_start.append(instance_id)
            elif not should_run and state == 'running':
                instances_to_stop.append(instance_id)

    # 执行启停操作
    if instances_to_stop:
        print(f"停止实例: {instances_to_stop}")
        ec2.stop_instances(InstanceIds=instances_to_stop)

    if instances_to_start:
        print(f"启动实例: {instances_to_start}")
        ec2.start_instances(InstanceIds=instances_to_start)

    return {
        'stopped': len(instances_to_stop),
        'started': len(instances_to_start)
    }

def parse_schedule(schedule, weekday, hour):
    """
    解析时间表
    示例:
    - weekdays-9to18: 工作日9-18点
    - weekends-0to24: 周末全天
    - always: 始终运行
    """

    if schedule == 'always':
        return True

    parts = schedule.split('-')
    days_part = parts[0]
    hours_part = parts[1] if len(parts) > 1 else '0to24'

    # 检查日期
    if days_part == 'weekdays' and weekday >= 5:  # 周末
        return False
    elif days_part == 'weekends' and weekday < 5:  # 工作日
        return False

    # 检查小时
    start_hour, end_hour = map(int, hours_part.replace('to', ' ').split())
    return start_hour <= hour < end_hour

# EventBridge 规则示例
"""
{
  "name": "AutoStopEC2",
  "scheduleExpression": "cron(0 * * * ? *)",  # 每小时执行
  "state": "ENABLED",
  "targets": [{
    "arn": "arn:aws:lambda:us-east-1:123456:function:AutoStartStop",
    "id": "1"
  }]
}
"""
```

### 3. 预留实例分析器

```python
# reserved_instance_analyzer.py
import boto3
from datetime import datetime, timedelta

class ReservedInstanceAnalyzer:
    """预留实例推荐分析"""

    def __init__(self):
        self.ce = boto3.client('ce')
        self.ec2 = boto3.client('ec2')

    def analyze_ri_opportunities(self):
        """分析预留实例购买机会"""

        # 获取过去30天的实例使用情况
        end = datetime.now()
        start = end - timedelta(days=30)

        response = self.ce.get_reservation_purchase_recommendation(
            Service='Amazon Elastic Compute Cloud - Compute',
            LookbackPeriodInDays='THIRTY_DAYS',
            TermInYears='ONE_YEAR',
            PaymentOption='NO_UPFRONT'
        )

        recommendations = []

        for rec in response['Recommendations']:
            details = rec['RecommendationDetails']

            for detail in details:
                instance_details = detail['InstanceDetails']['EC2InstanceDetails']

                recommendations.append({
                    'InstanceType': instance_details['InstanceType'],
                    'Region': instance_details['Region'],
                    'Quantity': detail['RecommendedNumberOfInstancesToPurchase'],
                    'MonthlySavings': f"${detail['EstimatedMonthlySavingsAmount']:.2f}",
                    'UpfrontCost': f"${detail['UpfrontCost']:.2f}",
                    'BreakEvenMonths': detail['EstimatedBreakEvenInMonths']
                })

        return recommendations

    def calculate_ri_coverage(self):
        """计算当前RI覆盖率"""
        end = datetime.now()
        start = end - timedelta(days=7)

        response = self.ce.get_reservation_coverage(
            TimePeriod={
                'Start': start.strftime('%Y-%m-%d'),
                'End': end.strftime('%Y-%m-%d')
            },
            Granularity='DAILY',
            GroupBy=[{'Type': 'DIMENSION', 'Key': 'INSTANCE_TYPE'}]
        )

        coverage_data = []

        for result in response['CoveragesByTime']:
            date = result['TimePeriod']['Start']

            for group in result['Groups']:
                instance_type = group['Attributes']['instanceType']
                coverage = group['Coverage']

                coverage_pct = float(coverage.get('CoverageHours', {}).get('CoverageHoursPercentage', 0))

                if coverage_pct < 80:  # 覆盖率 < 80%
                    coverage_data.append({
                        'Date': date,
                        'InstanceType': instance_type,
                        'Coverage': f"{coverage_pct:.1f}%",
                        'OnDemandHours': coverage['CoverageHours']['OnDemandHours'],
                        'ReservedHours': coverage['CoverageHours']['ReservedHours']
                    })

        return coverage_data

# 使用示例
analyzer = ReservedInstanceAnalyzer()

print("💰 预留实例购买建议:")
recommendations = analyzer.analyze_ri_opportunities()
for rec in recommendations:
    print(f"  {rec['InstanceType']} x{rec['Quantity']}")
    print(f"    节省: {rec['MonthlySavings']}/月")
    print(f"    回本周期: {rec['BreakEvenMonths']}个月\n")

print("📊 当前 RI 覆盖率:")
coverage = analyzer.calculate_ri_coverage()
for item in coverage:
    print(f"  {item['InstanceType']}: {item['Coverage']}")
```

---

## FinOps实践

### FinOps 组织结构

```
┌────────────────────────────────────────────────────┐
│            FinOps 团队结构                         │
├────────────────────────────────────────────────────┤
│                                                    │
│         ┌──────────────────┐                      │
│         │   FinOps Lead    │                      │
│         │  (CFO/CTO直属)   │                      │
│         └────────┬─────────┘                      │
│                  │                                 │
│      ┌───────────┼───────────┐                    │
│      │           │           │                     │
│  ┌───▼──┐   ┌───▼──┐   ┌───▼──┐                 │
│  │财务  │   │工程  │   │业务  │                 │
│  │分析师│   │负责人│   │负责人│                 │
│  └──────┘   └──────┘   └──────┘                 │
│                                                    │
│  职责:                                             │
│  • 成本建模与预测                                  │
│  • 制定优化策略                                    │
│  • 跨部门协调                                      │
│  • 推动文化变革                                    │
└────────────────────────────────────────────────────┘
```

### 成本分摊模型

```python
# cost_allocation.py
import boto3
import pandas as pd

class CostAllocator:
    """成本分摊计算"""

    def __init__(self):
        self.ce = boto3.client('ce')

    def allocate_by_tags(self, start_date, end_date):
        """按标签分摊成本"""

        response = self.ce.get_cost_and_usage(
            TimePeriod={'Start': start_date, 'End': end_date},
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            GroupBy=[
                {'Type': 'TAG', 'Key': 'Project'},
                {'Type': 'TAG', 'Key': 'Environment'}
            ]
        )

        # 解析数据
        allocations = []

        for result in response['ResultsByTime']:
            month = result['TimePeriod']['Start']

            for group in result['Groups']:
                tags = group['Keys']
                project = tags[0].split('$')[-1] if len(tags) > 0 else 'Untagged'
                env = tags[1].split('$')[-1] if len(tags) > 1 else 'Unknown'

                cost = float(group['Metrics']['UnblendedCost']['Amount'])

                allocations.append({
                    'Month': month,
                    'Project': project,
                    'Environment': env,
                    'Cost': cost
                })

        df = pd.DataFrame(allocations)

        # 透视表 - 按项目汇总
        pivot = df.pivot_table(
            index='Project',
            columns='Environment',
            values='Cost',
            aggfunc='sum',
            fill_value=0,
            margins=True  # 添加总计行
        )

        return pivot

# 使用示例
allocator = CostAllocator()
allocation = allocator.allocate_by_tags('2026-01-01', '2026-02-01')

print("💼 成本分摊（按项目和环境）:")
print(allocation)

# 输出示例:
"""
Environment    dev     prod   staging    All
Project
ProjectA     125.50  1250.30    50.20  1426.00
ProjectB      80.00   950.00    30.00  1060.00
ProjectC     200.00  2000.00   100.00  2300.00
Untagged      50.00   300.00    20.00   370.00
All          455.50  4500.30   200.20  5156.00
"""
```

---

## 总结

### 成本优化检查清单

```
☐ 清理浪费资源
  ☐ 停止闲置 EC2 实例
  ☐ 删除未挂载 EBS 卷
  ☐ 释放未关联弹性 IP
  ☐ 清理旧快照

☐ 实例规格优化
  ☐ Right Sizing 评估
  ☐ 使用 Graviton 实例（ARM）
  ☐ 使用 Spot 实例（非关键）

☐ 预留实例与节省计划
  ☐ 购买 1年/3年 RI
  ☐ 使用 Savings Plans

☐ 存储优化
  ☐ S3 生命周期策略
  ☐ EBS 卷类型优化（gp3）
  ☐ 删除未使用快照

☐ 自动化与调度
  ☐ 开发环境自动启停
  ☐ Auto Scaling 配置

☐ 监控与治理
  ☐ 成本告警配置
  ☐ 预算设置
  ☐ 标签策略执行

☐ 架构优化
  ☐ 考虑 Serverless
  ☐ 使用托管服务
  ☐ CDN 加速

☐ FinOps 文化
  ☐ 成本可见性
  ☐ 团队培训
  ☐ 定期 Review
```

### 下一步

成本优化是持续性工作，建议：
1. 每周查看成本趋势
2. 每月执行优化 Review
3. 每季度更新预留实例
4. 每年评估架构设计

**云成本 = 云资源 × 单价 × 时间**

优化任一因素都能降低成本！
