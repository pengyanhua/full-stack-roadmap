# 架构评审流程与实践

## 目录
- [架构评审概述](#架构评审概述)
- [评审流程设计](#评审流程设计)
- [评审检查清单](#评审检查清单)
- [评审会议实践](#评审会议实践)
- [评审模板](#评审模板)
- [工具与自动化](#工具与自动化)
- [实战案例](#实战案例)
- [最佳实践](#最佳实践)

## 架构评审概述

### 什么是架构评审

架构评审是对系统架构设计进行系统性检查和评估的过程，确保架构满足业务需求、技术要求和质量属性。

### 评审目标

```
┌─────────────────────────────────────────────────────────┐
│                    架构评审目标                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌──────────────┐                  │
│  │  风险识别    │───▶│  早期发现    │                  │
│  │  Risk ID     │    │  问题缺陷    │                  │
│  └──────────────┘    └──────────────┘                  │
│         │                    │                          │
│         ▼                    ▼                          │
│  ┌──────────────┐    ┌──────────────┐                  │
│  │  质量保证    │    │  知识共享    │                  │
│  │  QA          │───▶│  团队成长    │                  │
│  └──────────────┘    └──────────────┘                  │
│         │                    │                          │
│         ▼                    ▼                          │
│  ┌──────────────┐    ┌──────────────┐                  │
│  │  决策支持    │    │  标准执行    │                  │
│  │  Decision    │───▶│  Compliance  │                  │
│  └──────────────┘    └──────────────┘                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 评审类型

| 类型 | 时机 | 参与者 | 目标 | 周期 |
|------|------|--------|------|------|
| 概念评审 | 需求阶段 | 架构师、产品经理 | 验证架构可行性 | 一次性 |
| 设计评审 | 设计阶段 | 架构师、技术负责人 | 评估设计方案 | 1-2次 |
| 实施评审 | 开发阶段 | 架构师、开发团队 | 检查实施质量 | 迭代中 |
| 运维评审 | 上线前 | 架构师、运维团队 | 评估运维就绪度 | 上线前 |
| 持续评审 | 运行阶段 | 所有相关方 | 优化改进 | 季度/半年 |

## 评审流程设计

### 标准评审流程

```
┌────────────────────────────────────────────────────────────┐
│                     架构评审流程                            │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   1. 评审准备   │
                    │   - 材料准备    │
                    │   - 角色分配    │
                    │   - 会议安排    │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ 2. 材料预审     │
                    │   - 文档审查    │
                    │   - 问题收集    │
                    │   - 初步评估    │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ 3. 评审会议     │
                    │   - 方案讲解    │
                    │   - 问题讨论    │
                    │   - 决策记录    │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ 4. 问题跟踪     │
                    │   - 问题分类    │
                    │   - 责任分配    │
                    │   - 期限设定    │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ 5. 结果验证     │
                    │   - 改进确认    │
                    │   - 文档更新    │
                    │   - 评审关闭    │
                    └─────────────────┘
```

### 评审流程实施代码

```python
# architecture_review_system.py
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json

class ReviewType(Enum):
    """评审类型"""
    CONCEPT = "概念评审"
    DESIGN = "设计评审"
    IMPLEMENTATION = "实施评审"
    OPERATION = "运维评审"
    CONTINUOUS = "持续评审"

class Severity(Enum):
    """问题严重程度"""
    CRITICAL = "严重"  # 必须解决
    HIGH = "高"        # 建议解决
    MEDIUM = "中"      # 可选解决
    LOW = "低"         # 后续优化

class IssueStatus(Enum):
    """问题状态"""
    OPEN = "待解决"
    IN_PROGRESS = "处理中"
    RESOLVED = "已解决"
    CLOSED = "已关闭"
    WONT_FIX = "不修复"

@dataclass
class ReviewIssue:
    """评审问题"""
    id: str
    title: str
    description: str
    severity: Severity
    category: str  # 如：性能、安全、可维护性
    status: IssueStatus = IssueStatus.OPEN
    assignee: Optional[str] = None
    due_date: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    resolution: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'category': self.category,
            'status': self.status.value,
            'assignee': self.assignee,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'created_at': self.created_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolution': self.resolution
        }

@dataclass
class ReviewParticipant:
    """评审参与者"""
    name: str
    role: str  # 如：主架构师、评审员、观察员
    department: str
    email: str

@dataclass
class ArchitectureReview:
    """架构评审"""
    id: str
    project_name: str
    review_type: ReviewType
    description: str
    scheduled_date: datetime
    participants: List[ReviewParticipant]
    issues: List[ReviewIssue] = field(default_factory=list)
    decision: Optional[str] = None  # 批准/有条件批准/拒绝
    notes: str = ""
    documents: List[str] = field(default_factory=list)

    def add_issue(self, issue: ReviewIssue):
        """添加问题"""
        self.issues.append(issue)

    def get_critical_issues(self) -> List[ReviewIssue]:
        """获取严重问题"""
        return [i for i in self.issues if i.severity == Severity.CRITICAL]

    def get_open_issues(self) -> List[ReviewIssue]:
        """获取未解决问题"""
        return [i for i in self.issues
                if i.status in [IssueStatus.OPEN, IssueStatus.IN_PROGRESS]]

    def calculate_completion_rate(self) -> float:
        """计算问题解决率"""
        if not self.issues:
            return 100.0
        resolved = len([i for i in self.issues
                       if i.status in [IssueStatus.RESOLVED, IssueStatus.CLOSED]])
        return (resolved / len(self.issues)) * 100

    def can_approve(self) -> bool:
        """判断是否可以批准"""
        # 所有严重问题必须解决
        critical = self.get_critical_issues()
        return all(i.status in [IssueStatus.RESOLVED, IssueStatus.CLOSED]
                  for i in critical)

    def generate_report(self) -> str:
        """生成评审报告"""
        report = f"""
架构评审报告
================

项目名称: {self.project_name}
评审类型: {self.review_type.value}
评审日期: {self.scheduled_date.strftime('%Y-%m-%d')}
评审决策: {self.decision or '待定'}

参与人员:
{self._format_participants()}

问题统计:
- 总问题数: {len(self.issues)}
- 严重: {len([i for i in self.issues if i.severity == Severity.CRITICAL])}
- 高: {len([i for i in self.issues if i.severity == Severity.HIGH])}
- 中: {len([i for i in self.issues if i.severity == Severity.MEDIUM])}
- 低: {len([i for i in self.issues if i.severity == Severity.LOW])}

解决率: {self.calculate_completion_rate():.1f}%

问题详情:
{self._format_issues()}

评审备注:
{self.notes}
"""
        return report

    def _format_participants(self) -> str:
        lines = []
        for p in self.participants:
            lines.append(f"- {p.name} ({p.role}) - {p.department}")
        return '\n'.join(lines)

    def _format_issues(self) -> str:
        if not self.issues:
            return "无问题"

        lines = []
        for issue in self.issues:
            lines.append(f"""
[{issue.id}] {issue.title}
严重程度: {issue.severity.value} | 分类: {issue.category} | 状态: {issue.status.value}
描述: {issue.description}
责任人: {issue.assignee or '未分配'} | 截止日期: {issue.due_date.strftime('%Y-%m-%d') if issue.due_date else '未设定'}
""")
        return '\n'.join(lines)

class ReviewChecklist:
    """评审检查清单"""

    @staticmethod
    def get_design_checklist() -> Dict[str, List[str]]:
        """设计评审检查清单"""
        return {
            "业务需求": [
                "业务目标是否清晰明确？",
                "功能需求是否完整？",
                "非功能需求是否定义？",
                "约束条件是否识别？"
            ],
            "架构设计": [
                "架构风格是否合适？",
                "层次划分是否清晰？",
                "模块职责是否单一？",
                "接口定义是否合理？",
                "依赖关系是否最小化？"
            ],
            "技术选型": [
                "技术栈是否成熟稳定？",
                "团队是否具备相关技能？",
                "是否符合公司技术规范？",
                "第三方依赖是否可控？"
            ],
            "性能": [
                "是否有性能目标定义？",
                "关键路径是否优化？",
                "缓存策略是否合理？",
                "数据库设计是否支持高并发？",
                "是否考虑水平扩展？"
            ],
            "可用性": [
                "是否有冗余设计？",
                "故障转移机制是否完善？",
                "是否有降级策略？",
                "监控告警是否覆盖？"
            ],
            "安全性": [
                "认证授权机制是否完善？",
                "数据传输是否加密？",
                "敏感数据是否脱敏？",
                "是否有安全审计？",
                "是否考虑常见攻击防护？"
            ],
            "可维护性": [
                "代码结构是否清晰？",
                "是否遵循编码规范？",
                "日志是否完善？",
                "配置管理是否规范？",
                "文档是否完整？"
            ],
            "可测试性": [
                "是否支持单元测试？",
                "是否支持集成测试？",
                "测试环境是否就绪？",
                "测试覆盖率目标是否定义？"
            ],
            "运维": [
                "部署方案是否可行？",
                "监控指标是否定义？",
                "日志采集是否配置？",
                "备份恢复方案是否完善？",
                "应急预案是否制定？"
            ]
        }

    @staticmethod
    def evaluate_checklist(review: ArchitectureReview,
                          responses: Dict[str, Dict[str, bool]]) -> float:
        """评估检查清单完成度"""
        total = 0
        passed = 0
        for category, items in responses.items():
            for question, answer in items.items():
                total += 1
                if answer:
                    passed += 1
        return (passed / total * 100) if total > 0 else 0

# 使用示例
def example_usage():
    # 创建评审
    review = ArchitectureReview(
        id="ARCH-2024-001",
        project_name="电商平台重构",
        review_type=ReviewType.DESIGN,
        description="将单体应用拆分为微服务架构",
        scheduled_date=datetime.now() + timedelta(days=7),
        participants=[
            ReviewParticipant("张三", "主架构师", "技术部", "zhangsan@company.com"),
            ReviewParticipant("李四", "安全架构师", "安全部", "lisi@company.com"),
            ReviewParticipant("王五", "DBA", "运维部", "wangwu@company.com"),
        ],
        documents=[
            "架构设计文档 v1.0",
            "API接口规范",
            "数据库设计文档"
        ]
    )

    # 添加问题
    review.add_issue(ReviewIssue(
        id="ISSUE-001",
        title="缺少分布式事务处理方案",
        description="跨服务的订单流程缺少事务一致性保证，需要引入Saga或TCC模式",
        severity=Severity.CRITICAL,
        category="架构设计",
        assignee="张三",
        due_date=datetime.now() + timedelta(days=3)
    ))

    review.add_issue(ReviewIssue(
        id="ISSUE-002",
        title="API网关缺少限流配置",
        description="需要添加基于令牌桶的限流策略，防止服务过载",
        severity=Severity.HIGH,
        category="性能",
        assignee="李四",
        due_date=datetime.now() + timedelta(days=5)
    ))

    review.add_issue(ReviewIssue(
        id="ISSUE-003",
        title="日志格式不统一",
        description="建议统一使用JSON格式的结构化日志",
        severity=Severity.MEDIUM,
        category="可维护性",
        assignee="王五",
        due_date=datetime.now() + timedelta(days=7)
    ))

    # 检查是否可以批准
    print(f"可以批准: {review.can_approve()}")
    print(f"未解决问题数: {len(review.get_open_issues())}")

    # 生成报告
    print(review.generate_report())

    # 评估检查清单
    checklist = ReviewChecklist.get_design_checklist()
    responses = {
        "业务需求": {
            "业务目标是否清晰明确？": True,
            "功能需求是否完整？": True,
            "非功能需求是否定义？": False,
            "约束条件是否识别？": True
        },
        # ... 其他类别
    }

    score = ReviewChecklist.evaluate_checklist(review, responses)
    print(f"检查清单完成度: {score:.1f}%")

if __name__ == "__main__":
    example_usage()
```

## 评审检查清单

### 架构设计检查清单

#### 1. 业务对齐检查

```markdown
## 业务需求检查
- [ ] 业务目标清晰定义
- [ ] 关键业务流程识别
- [ ] 业务优先级排序
- [ ] 业务约束条件明确
- [ ] 业务增长预期评估

## 功能需求检查
- [ ] 功能列表完整
- [ ] 用户故事清晰
- [ ] 验收标准明确
- [ ] 边界条件考虑
- [ ] 异常场景覆盖

## 非功能需求检查
- [ ] 性能指标量化（QPS、RT、TPS）
- [ ] 可用性目标定义（SLA）
- [ ] 扩展性要求明确
- [ ] 安全性标准设定
- [ ] 兼容性范围界定
```

#### 2. 技术架构检查

```markdown
## 架构风格
- [ ] 架构风格选择合理（单体/微服务/Serverless）
- [ ] 架构模式适用（分层/事件驱动/CQRS）
- [ ] 服务划分清晰
- [ ] 边界上下文明确（DDD）

## 技术选型
- [ ] 编程语言选择合理
- [ ] 框架版本稳定且活跃
- [ ] 数据库类型匹配需求
- [ ] 中间件选型成熟
- [ ] 第三方依赖可控
- [ ] 许可证合规

## 接口设计
- [ ] API规范统一（RESTful/gRPC）
- [ ] 接口版本管理
- [ ] 参数校验完整
- [ ] 错误码定义清晰
- [ ] 接口文档完善

## 数据设计
- [ ] 数据模型合理
- [ ] 索引设计优化
- [ ] 分库分表策略
- [ ] 数据一致性保证
- [ ] 数据归档方案
```

#### 3. 质量属性检查

```markdown
## 性能 (Performance)
- [ ] 响应时间目标：P99 < 200ms
- [ ] 吞吐量目标：QPS > 10000
- [ ] 并发用户数：> 100000
- [ ] 缓存命中率：> 90%
- [ ] 数据库连接池配置
- [ ] 慢查询优化
- [ ] CDN加速配置

## 可用性 (Availability)
- [ ] SLA目标：99.95%
- [ ] 单点故障消除
- [ ] 故障自动转移
- [ ] 降级策略定义
- [ ] 熔断机制配置
- [ ] 限流规则设置
- [ ] 重试策略合理

## 可扩展性 (Scalability)
- [ ] 水平扩展支持
- [ ] 无状态设计
- [ ] 弹性伸缩配置
- [ ] 数据库分片方案
- [ ] 存储扩展方案

## 安全性 (Security)
- [ ] 认证机制（OAuth2/JWT）
- [ ] 授权控制（RBAC/ABAC）
- [ ] 数据加密（传输/存储）
- [ ] 敏感信息脱敏
- [ ] SQL注入防护
- [ ] XSS攻击防护
- [ ] CSRF防护
- [ ] 安全审计日志

## 可维护性 (Maintainability)
- [ ] 代码结构清晰
- [ ] 模块耦合度低
- [ ] 命名规范统一
- [ ] 注释文档完善
- [ ] 配置外部化
- [ ] 日志级别合理
- [ ] 监控指标完整

## 可测试性 (Testability)
- [ ] 单元测试覆盖率 > 80%
- [ ] 集成测试用例完整
- [ ] 测试数据准备方案
- [ ] Mock/Stub机制
- [ ] 性能测试脚本
```

### 评审检查清单自动化工具

```python
# review_checklist_tool.py
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class CheckResult(Enum):
    """检查结果"""
    PASS = "通过"
    FAIL = "失败"
    WARNING = "警告"
    NA = "不适用"

@dataclass
class CheckItem:
    """检查项"""
    id: str
    category: str
    question: str
    weight: int  # 权重1-5
    result: CheckResult = CheckResult.NA
    comment: str = ""
    evidence: str = ""  # 证据/参考链接

class ChecklistTemplate:
    """检查清单模板"""

    @staticmethod
    def get_microservice_checklist() -> List[CheckItem]:
        """微服务架构检查清单"""
        return [
            # 服务设计
            CheckItem("MS-001", "服务设计", "服务是否按照业务能力划分？", 5),
            CheckItem("MS-002", "服务设计", "服务间是否避免了循环依赖？", 5),
            CheckItem("MS-003", "服务设计", "是否定义了服务API规范？", 4),
            CheckItem("MS-004", "服务设计", "是否实现了服务注册与发现？", 5),

            # 数据管理
            CheckItem("MS-005", "数据管理", "每个服务是否有独立数据库？", 4),
            CheckItem("MS-006", "数据管理", "是否定义了数据一致性策略？", 5),
            CheckItem("MS-007", "数据管理", "是否有分布式事务处理方案？", 5),

            # 通信机制
            CheckItem("MS-008", "通信", "服务间通信是否异步化？", 3),
            CheckItem("MS-009", "通信", "是否实现了API网关？", 5),
            CheckItem("MS-010", "通信", "是否有服务调用超时控制？", 4),
            CheckItem("MS-011", "通信", "是否实现了熔断器模式？", 5),

            # 可观测性
            CheckItem("MS-012", "可观测性", "是否有分布式追踪？", 5),
            CheckItem("MS-013", "可观测性", "是否有集中式日志？", 5),
            CheckItem("MS-014", "可观测性", "是否有服务监控指标？", 5),
            CheckItem("MS-015", "可观测性", "是否有告警规则？", 4),

            # 部署运维
            CheckItem("MS-016", "部署", "是否容器化部署？", 4),
            CheckItem("MS-017", "部署", "是否支持蓝绿/金丝雀发布？", 3),
            CheckItem("MS-018", "部署", "是否有自动化CI/CD？", 5),

            # 安全
            CheckItem("MS-019", "安全", "是否有统一的认证授权？", 5),
            CheckItem("MS-020", "安全", "服务间通信是否加密？", 4),
        ]

    @staticmethod
    def get_database_checklist() -> List[CheckItem]:
        """数据库设计检查清单"""
        return [
            CheckItem("DB-001", "模型设计", "是否符合第三范式？", 4),
            CheckItem("DB-002", "模型设计", "是否有合理的反范式优化？", 3),
            CheckItem("DB-003", "索引", "主键和外键是否有索引？", 5),
            CheckItem("DB-004", "索引", "查询字段是否有覆盖索引？", 4),
            CheckItem("DB-005", "性能", "是否有分库分表策略？", 4),
            CheckItem("DB-006", "性能", "是否有读写分离？", 3),
            CheckItem("DB-007", "安全", "敏感字段是否加密？", 5),
            CheckItem("DB-008", "备份", "是否有定时备份策略？", 5),
        ]

class ChecklistEvaluator:
    """检查清单评估器"""

    @staticmethod
    def calculate_score(items: List[CheckItem]) -> Dict:
        """计算得分"""
        total_weight = sum(item.weight for item in items)
        passed_weight = sum(
            item.weight for item in items
            if item.result == CheckResult.PASS
        )
        failed_weight = sum(
            item.weight for item in items
            if item.result == CheckResult.FAIL
        )

        score = (passed_weight / total_weight * 100) if total_weight > 0 else 0

        return {
            'total_items': len(items),
            'passed': len([i for i in items if i.result == CheckResult.PASS]),
            'failed': len([i for i in items if i.result == CheckResult.FAIL]),
            'warning': len([i for i in items if i.result == CheckResult.WARNING]),
            'na': len([i for i in items if i.result == CheckResult.NA]),
            'score': round(score, 2),
            'weighted_score': round(passed_weight / total_weight * 100, 2) if total_weight > 0 else 0
        }

    @staticmethod
    def generate_report(items: List[CheckItem]) -> str:
        """生成评估报告"""
        stats = ChecklistEvaluator.calculate_score(items)

        # 按类别分组
        by_category = {}
        for item in items:
            if item.category not in by_category:
                by_category[item.category] = []
            by_category[item.category].append(item)

        report = f"""
检查清单评估报告
==================

总体统计:
- 总检查项: {stats['total_items']}
- 通过: {stats['passed']} ({stats['passed']/stats['total_items']*100:.1f}%)
- 失败: {stats['failed']} ({stats['failed']/stats['total_items']*100:.1f}%)
- 警告: {stats['warning']}
- 不适用: {stats['na']}
- 综合得分: {stats['weighted_score']}/100

分类详情:
"""

        for category, cat_items in by_category.items():
            cat_stats = ChecklistEvaluator.calculate_score(cat_items)
            report += f"\n{category} (得分: {cat_stats['weighted_score']}/100):\n"

            for item in cat_items:
                status_icon = {
                    CheckResult.PASS: "✓",
                    CheckResult.FAIL: "✗",
                    CheckResult.WARNING: "⚠",
                    CheckResult.NA: "-"
                }[item.result]

                report += f"  {status_icon} [{item.id}] {item.question} (权重:{item.weight})\n"
                if item.comment:
                    report += f"      备注: {item.comment}\n"

        # 失败项汇总
        failed_items = [i for i in items if i.result == CheckResult.FAIL]
        if failed_items:
            report += "\n需要改进的项目:\n"
            for item in failed_items:
                report += f"- [{item.id}] {item.question}\n"
                if item.comment:
                    report += f"  原因: {item.comment}\n"

        return report

# 使用示例
def checklist_example():
    # 获取微服务检查清单
    checklist = ChecklistTemplate.get_microservice_checklist()

    # 填写检查结果
    checklist[0].result = CheckResult.PASS
    checklist[0].comment = "服务按照用户、订单、支付等业务能力划分"

    checklist[1].result = CheckResult.PASS
    checklist[1].comment = "通过依赖分析工具确认无循环依赖"

    checklist[2].result = CheckResult.WARNING
    checklist[2].comment = "API规范已定义，但部分服务未严格遵循"

    checklist[3].result = CheckResult.PASS
    checklist[3].comment = "使用Consul实现服务注册与发现"

    checklist[6].result = CheckResult.FAIL
    checklist[6].comment = "缺少Saga模式实现，跨服务事务一致性无保证"

    # 生成报告
    report = ChecklistEvaluator.generate_report(checklist)
    print(report)

    # 计算得分
    stats = ChecklistEvaluator.calculate_score(checklist)
    print(f"\n加权得分: {stats['weighted_score']}/100")

if __name__ == "__main__":
    checklist_example()
```

## 评审会议实践

### 会议流程

```
评审会议流程 (建议时长: 2小时)
====================================

00:00 - 00:10  会议开场 (10分钟)
               - 介绍参会人员
               - 说明评审目标
               - 回顾评审规则

00:10 - 00:40  方案讲解 (30分钟)
               - 业务背景和目标
               - 架构设计方案
               - 技术选型理由
               - 风险和挑战

00:40 - 01:20  问题讨论 (40分钟)
               - 提出疑问和建议
               - 深入技术细节
               - 替代方案探讨
               - 风险评估

01:20 - 01:50  决策制定 (30分钟)
               - 问题优先级排序
               - 责任人分配
               - 期限确定
               - 评审结论

01:50 - 02:00  总结行动 (10分钟)
               - 回顾行动项
               - 确认下一步
               - 会议纪要分发
```

### 会议角色

| 角色 | 职责 | 人数 |
|------|------|------|
| 主持人 | 控制会议节奏，引导讨论 | 1人 |
| 方案讲解人 | 介绍架构方案，回答问题 | 1-2人 |
| 评审专家 | 提出问题，给出建议 | 3-5人 |
| 记录员 | 记录问题和决策 | 1人 |
| 观察员 | 学习参考，不参与决策 | 若干 |

## 评审模板

### 评审申请模板

```markdown
# 架构评审申请表

## 基本信息
- 项目名称:
- 申请人:
- 申请日期:
- 期望评审日期:
- 评审类型: [ ] 概念评审 [ ] 设计评审 [ ] 实施评审 [ ] 运维评审

## 项目背景
(简要说明项目背景、业务价值、预期收益)

## 评审范围
(明确本次评审的范围和重点)

## 材料清单
- [ ] 架构设计文档
- [ ] 接口设计文档
- [ ] 数据库设计文档
- [ ] 部署方案
- [ ] 测试方案
- [ ] 其他: _____________

## 期望参与的评审专家
1.
2.
3.

## 关注的重点问题
1.
2.
3.
```

### 评审记录模板

```markdown
# 架构评审会议记录

## 会议信息
- 项目名称:
- 评审编号:
- 评审日期:
- 会议时长:
- 主持人:
- 记录人:

## 参会人员
| 姓名 | 角色 | 部门 |
|------|------|------|
|      |      |      |

## 评审材料
1.
2.

## 方案概述
(简要记录方案讲解的核心内容)

## 讨论要点
1.
2.
3.

## 问题列表

### 严重问题 (必须解决)
| 编号 | 问题描述 | 责任人 | 截止日期 |
|------|----------|--------|----------|
| P0-1 |          |        |          |

### 高优先级 (建议解决)
| 编号 | 问题描述 | 责任人 | 截止日期 |
|------|----------|--------|----------|
| P1-1 |          |        |          |

### 中/低优先级 (可选)
| 编号 | 问题描述 | 责任人 | 截止日期 |
|------|----------|--------|----------|
| P2-1 |          |        |          |

## 评审决策
[ ] 批准 - 可以进入下一阶段
[ ] 有条件批准 - 解决P0问题后可进入下一阶段
[ ] 拒绝 - 需要重大调整后重新评审

## 行动计划
1.
2.
3.

## 下次评审计划
- 日期:
- 重点:
```

## 工具与自动化

### 架构符合性检查工具

```python
# architecture_compliance_checker.py
import ast
import os
from typing import List, Dict, Set
from pathlib import Path

class ArchitectureRule:
    """架构规则"""

    def __init__(self, rule_id: str, description: str):
        self.rule_id = rule_id
        self.description = description
        self.violations = []

    def check(self, codebase_path: str) -> bool:
        """检查规则，返回是否通过"""
        raise NotImplementedError

class LayerDependencyRule(ArchitectureRule):
    """分层依赖规则"""

    def __init__(self, layers: List[str]):
        super().__init__(
            "LAYER-001",
            "检查分层架构的依赖关系是否符合规范"
        )
        self.layers = layers  # 从上到下：如 ['web', 'service', 'dao']

    def check(self, codebase_path: str) -> bool:
        """检查每一层是否只依赖下层"""
        violations = []

        for i, layer in enumerate(self.layers):
            layer_path = os.path.join(codebase_path, layer)
            if not os.path.exists(layer_path):
                continue

            # 扫描该层的所有Python文件
            for root, dirs, files in os.walk(layer_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        imports = self._extract_imports(file_path)

                        # 检查是否导入了上层模块
                        for imp in imports:
                            for upper_layer in self.layers[:i]:
                                if upper_layer in imp:
                                    violations.append(
                                        f"{file_path} 违规导入上层 {upper_layer}: {imp}"
                                    )

        self.violations = violations
        return len(violations) == 0

    def _extract_imports(self, file_path: str) -> List[str]:
        """提取文件中的import语句"""
        imports = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except:
            pass

        return imports

class CircularDependencyRule(ArchitectureRule):
    """循环依赖检查"""

    def __init__(self):
        super().__init__(
            "CIRC-001",
            "检查模块间是否存在循环依赖"
        )

    def check(self, codebase_path: str) -> bool:
        """使用DFS检测循环依赖"""
        dependency_graph = self._build_dependency_graph(codebase_path)
        cycles = self._find_cycles(dependency_graph)

        if cycles:
            self.violations = [f"发现循环依赖: {' -> '.join(cycle)}"
                             for cycle in cycles]
            return False
        return True

    def _build_dependency_graph(self, codebase_path: str) -> Dict[str, Set[str]]:
        """构建模块依赖图"""
        graph = {}

        for root, dirs, files in os.walk(codebase_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    module_name = self._get_module_name(codebase_path, file_path)

                    if module_name not in graph:
                        graph[module_name] = set()

                    # 获取该模块的所有导入
                    imports = self._extract_local_imports(file_path, codebase_path)
                    graph[module_name].update(imports)

        return graph

    def _get_module_name(self, base_path: str, file_path: str) -> str:
        """获取模块名"""
        rel_path = os.path.relpath(file_path, base_path)
        return rel_path.replace(os.sep, '.').replace('.py', '')

    def _extract_local_imports(self, file_path: str, base_path: str) -> Set[str]:
        """提取本地模块的导入"""
        imports = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # 简化版：实际需要更复杂的逻辑判断是否本地模块
                    if isinstance(node, ast.ImportFrom) and node.module:
                        if not node.module.startswith('.'): # 相对导入
                            imports.add(node.module)
        except:
            pass

        return imports

    def _find_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """使用DFS查找所有环"""
        cycles = []
        visited = set()
        rec_stack = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # 找到环
                    cycle_start = rec_stack.index(neighbor)
                    cycle = rec_stack[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                    return True

            rec_stack.pop()
            return False

        for node in graph:
            if node not in visited:
                dfs(node)

        return cycles

class ComplianceChecker:
    """架构符合性检查器"""

    def __init__(self):
        self.rules: List[ArchitectureRule] = []

    def add_rule(self, rule: ArchitectureRule):
        """添加检查规则"""
        self.rules.append(rule)

    def check_all(self, codebase_path: str) -> Dict:
        """执行所有检查"""
        results = {
            'passed': [],
            'failed': [],
            'total': len(self.rules)
        }

        for rule in self.rules:
            print(f"检查规则: {rule.rule_id} - {rule.description}")

            if rule.check(codebase_path):
                results['passed'].append(rule.rule_id)
                print(f"  ✓ 通过")
            else:
                results['failed'].append({
                    'rule_id': rule.rule_id,
                    'description': rule.description,
                    'violations': rule.violations
                })
                print(f"  ✗ 失败")
                for violation in rule.violations:
                    print(f"    - {violation}")

        return results

    def generate_report(self, results: Dict) -> str:
        """生成检查报告"""
        report = f"""
架构符合性检查报告
===================

总计: {results['total']} 条规则
通过: {len(results['passed'])} 条
失败: {len(results['failed'])} 条

"""

        if results['failed']:
            report += "失败规则详情:\n"
            for failed in results['failed']:
                report += f"\n[{failed['rule_id']}] {failed['description']}\n"
                for violation in failed['violations']:
                    report += f"  - {violation}\n"
        else:
            report += "所有规则检查通过！\n"

        return report

# 使用示例
def compliance_check_example():
    checker = ComplianceChecker()

    # 添加检查规则
    checker.add_rule(LayerDependencyRule(['web', 'service', 'dao']))
    checker.add_rule(CircularDependencyRule())

    # 执行检查
    results = checker.check_all('/path/to/your/project')

    # 生成报告
    report = checker.generate_report(results)
    print(report)

    # 保存报告
    with open('architecture_compliance_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

if __name__ == "__main__":
    compliance_check_example()
```

## 实战案例

### 案例：电商平台微服务化评审

#### 背景
某电商平台单体应用面临性能瓶颈，计划拆分为微服务架构。

#### 评审准备

**提交材料：**
1. 服务拆分方案（10个微服务）
2. API网关设计
3. 数据库拆分方案
4. 分布式事务方案（Saga模式）
5. 监控方案

#### 评审过程

**发现的主要问题：**

```python
# 评审问题记录
problems = [
    {
        'id': 'P0-1',
        'severity': 'CRITICAL',
        'category': '数据一致性',
        'description': '订单服务和库存服务之间的Saga补偿逻辑不完整',
        'impact': '可能导致超卖或库存不准',
        'recommendation': '补充所有失败场景的补偿操作，增加幂等性保证',
        'assignee': '张三',
        'due_date': '2024-01-15'
    },
    {
        'id': 'P0-2',
        'severity': 'CRITICAL',
        'category': '性能',
        'description': 'API网关没有限流配置',
        'impact': '可能被恶意请求打垮',
        'recommendation': '实施基于令牌桶的限流，每个API设置QPS上限',
        'assignee': '李四',
        'due_date': '2024-01-10'
    },
    {
        'id': 'P1-1',
        'severity': 'HIGH',
        'category': '监控',
        'description': '缺少分布式链路追踪',
        'impact': '难以排查跨服务问题',
        'recommendation': '引入Jaeger或Zipkin实现全链路追踪',
        'assignee': '王五',
        'due_date': '2024-01-20'
    },
    {
        'id': 'P1-2',
        'severity': 'HIGH',
        'category': '安全',
        'description': '服务间通信未加密',
        'impact': '内网嗅探风险',
        'recommendation': '启用mTLS实现服务间双向认证',
        'assignee': '赵六',
        'due_date': '2024-01-25'
    },
    {
        'id': 'P2-1',
        'severity': 'MEDIUM',
        'category': '可维护性',
        'description': 'API文档不完整',
        'impact': '团队协作效率低',
        'recommendation': '使用Swagger/OpenAPI自动生成文档',
        'assignee': '孙七',
        'due_date': '2024-02-01'
    }
]
```

#### 评审结论

**决策：有条件批准**

**前置条件：**
- 必须解决所有P0级别问题
- P1问题在上线前解决
- P2问题在下个迭代解决

**后续行动：**
1. 一周后复审P0问题解决情况
2. 进行POC验证Saga方案可行性
3. 完善监控和告警配置

## 最佳实践

### 评审前

1. **提前准备**
   - 至少提前3天分发评审材料
   - 评审材料要完整、清晰
   - 准备演示Demo或原型

2. **明确目标**
   - 清楚本次评审的重点
   - 准备好需要决策的问题
   - 识别最大的风险点

### 评审中

1. **高效沟通**
   - 控制会议时间（不超过2小时）
   - 聚焦核心问题
   - 避免过度纠结细节

2. **建设性讨论**
   - 对事不对人
   - 提出问题的同时给出建议
   - 记录所有重要决策

3. **决策明确**
   - 问题优先级清晰
   - 责任人和期限明确
   - 评审结论清楚

### 评审后

1. **及时跟进**
   - 24小时内分发会议纪要
   - 问题录入跟踪系统
   - 定期检查问题解决进度

2. **持续改进**
   - 收集评审反馈
   - 优化评审流程
   - 完善检查清单

### 常见陷阱

1. **过度设计** - 为不存在的需求做过多设计
2. **忽视非功能需求** - 只关注功能，忽略性能、安全等
3. **技术崇拜** - 盲目追求新技术，忽视团队能力
4. **缺少量化指标** - 目标模糊，无法衡量
5. **评审流于形式** - 走过场，没有实质性讨论

### 评审指标

```python
# 评审效果指标
metrics = {
    '问题发现率': '评审发现的问题数 / 实际存在的问题数',
    '问题解决率': '已解决问题数 / 发现问题总数',
    '按时完成率': '按期解决问题数 / 承诺解决问题数',
    '评审覆盖率': '评审项目数 / 总项目数',
    '返工率': '因架构问题返工次数 / 总项目数',
    '线上故障率': '架构相关故障数 / 总故障数',
}
```

### 评审文化

- **鼓励提问** - 没有愚蠢的问题
- **开放心态** - 欢迎不同意见
- **持续学习** - 评审也是学习的过程
- **注重实效** - 评审是为了让系统更好

## 总结

架构评审是保证系统架构质量的重要手段。通过系统化的评审流程、完善的检查清单、高效的会议组织和自动化工具支持，可以在早期发现和解决架构问题，降低项目风险，提高交付质量。

**关键要点：**
1. 评审要趁早 - 问题发现越早，修复成本越低
2. 评审要系统 - 使用检查清单确保全面性
3. 评审要高效 - 控制时间，聚焦核心问题
4. 评审要跟踪 - 问题要闭环管理
5. 评审要改进 - 持续优化评审流程

**进阶阅读：**
- 《软件架构评估》
- 《ATAM架构权衡分析方法》
- 《架构整洁之道》

