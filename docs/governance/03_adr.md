# 架构决策记录 ADR (Architecture Decision Records)

## 目录
- [ADR概述](#adr概述)
- [ADR模板](#adr模板)
- [决策流程](#决策流程)
- [真实案例](#真实案例)
- [工具支持](#工具支持)
- [最佳实践](#最佳实践)

## ADR概述

### 什么是ADR

架构决策记录(Architecture Decision Record, ADR)是一种轻量级的文档形式，用于记录架构设计过程中的重要决策及其背景、原因和影响。

### ADR的价值

```
┌──────────────────────────────────────────────────────────┐
│                     ADR的价值                             │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────┐        ┌─────────────┐                 │
│  │  知识保留   │───────▶│  决策透明   │                 │
│  │  记录为什么 │        │  团队共识   │                 │
│  └─────────────┘        └─────────────┘                 │
│         │                      │                         │
│         │                      │                         │
│         ▼                      ▼                         │
│  ┌─────────────┐        ┌─────────────┐                 │
│  │  避免重复   │        │  新人上手   │                 │
│  │  决策失忆   │───────▶│  快速理解   │                 │
│  └─────────────┘        └─────────────┘                 │
│         │                      │                         │
│         │                      │                         │
│         ▼                      ▼                         │
│  ┌─────────────┐        ┌─────────────┐                 │
│  │  技术债务   │        │  决策审计   │                 │
│  │  可追溯     │───────▶│  合规证明   │                 │
│  └─────────────┘        └─────────────┘                 │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### 何时创建ADR

- 选择技术栈或框架时
- 架构模式选择时
- 重要的设计决策时
- 技术债务决策时
- 安全或性能权衡时

## ADR模板

### 标准ADR模板

```markdown
# ADR-XXXX: [简短标题]

## 状态
[提议中 | 已接受 | 已拒绝 | 已废弃 | 已替代]

如果已替代，链接到替代的ADR: [ADR-YYYY](#)

## 背景 (Context)
描述需要做决策的背景和问题：
- 业务背景是什么？
- 技术现状如何？
- 面临什么问题或挑战？
- 有什么约束条件？

## 决策 (Decision)
我们将要采取的决策是什么？
- 用简洁明确的语言描述决策
- 使用主动语态："我们将使用X技术"

## 理由 (Rationale)
为什么做这个决策？
- 解释选择的原因
- 说明考虑的权衡因素
- 引用相关数据或研究

## 后果 (Consequences)
这个决策会带来什么影响？

### 正面影响
- 好处1
- 好处2

### 负面影响
- 代价1
- 代价2

### 风险和缓解措施
- 风险1 → 缓解方案
- 风险2 → 缓解方案

## 替代方案 (Alternatives)
考虑过哪些其他方案？为什么没选它们？

### 方案A: [名称]
- 优点:
- 缺点:
- 为什么没选:

### 方案B: [名称]
- 优点:
- 缺点:
- 为什么没选:

## 参考资料 (References)
- [相关文档]
- [技术文章]
- [POC代码]

## 元数据
- 作者: [姓名]
- 日期: YYYY-MM-DD
- 评审者: [姓名列表]
- 相关ADR: [链接]
- 相关Issue: [链接]
```

### ADR编号规范

```
ADR-[YYYY]-[NNN]

YYYY: 年份
NNN:  序号（从001开始）

示例:
- ADR-2024-001
- ADR-2024-002
```

## 决策流程

### 决策过程

```
┌────────────────────────────────────────────────────┐
│              架构决策流程                           │
└────────────────────────────────────────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  1. 识别问题    │
              │  - 问题定义     │
              │  - 目标设定     │
              │  - 约束识别     │
              └─────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  2. 收集信息    │
              │  - 技术调研     │
              │  - 竞品分析     │
              │  - 团队讨论     │
              └─────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  3. 提出方案    │
              │  - 候选方案     │
              │  - 优劣分析     │
              │  - POC验证      │
              └─────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  4. 评估权衡    │
              │  - 多维度评分   │
              │  - 风险评估     │
              │  - 成本分析     │
              └─────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  5. 做出决策    │
              │  - 选择方案     │
              │  - 记录ADR      │
              │  - 团队评审     │
              └─────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  6. 执行跟踪    │
              │  - 实施方案     │
              │  - 效果验证     │
              │  - 持续改进     │
              └─────────────────┘
```

### 决策权衡矩阵

```python
# decision_matrix.py
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class CriteriaWeight(Enum):
    """评估维度权重"""
    CRITICAL = 5    # 关键
    HIGH = 4        # 高
    MEDIUM = 3      # 中
    LOW = 2         # 低
    MINIMAL = 1     # 最低

@dataclass
class Criteria:
    """评估维度"""
    name: str
    weight: CriteriaWeight
    description: str

@dataclass
class Alternative:
    """候选方案"""
    name: str
    description: str
    scores: Dict[str, int]  # 维度名称 -> 得分(1-10)
    pros: List[str]
    cons: List[str]
    estimated_cost: float  # 估算成本
    estimated_time: int    # 估算时间(天)

class DecisionMatrix:
    """决策矩阵"""

    def __init__(self, title: str):
        self.title = title
        self.criteria: List[Criteria] = []
        self.alternatives: List[Alternative] = []

    def add_criteria(self, criteria: Criteria):
        """添加评估维度"""
        self.criteria.append(criteria)

    def add_alternative(self, alternative: Alternative):
        """添加候选方案"""
        self.alternatives.append(alternative)

    def calculate_weighted_score(self, alternative: Alternative) -> float:
        """计算加权得分"""
        total_score = 0
        total_weight = 0

        for criteria in self.criteria:
            score = alternative.scores.get(criteria.name, 0)
            weight = criteria.weight.value
            total_score += score * weight
            total_weight += weight * 10  # 满分10

        return (total_score / total_weight * 100) if total_weight > 0 else 0

    def rank_alternatives(self) -> List[tuple]:
        """对方案进行排序"""
        ranked = []
        for alt in self.alternatives:
            score = self.calculate_weighted_score(alt)
            ranked.append((alt, score))

        return sorted(ranked, key=lambda x: x[1], reverse=True)

    def generate_report(self) -> str:
        """生成决策报告"""
        report = f"""
决策矩阵分析报告: {self.title}
{'='*60}

评估维度:
"""
        for c in self.criteria:
            report += f"  - {c.name} (权重: {c.weight.value}): {c.description}\n"

        report += f"\n方案排名:\n"
        ranked = self.rank_alternatives()
        for i, (alt, score) in enumerate(ranked, 1):
            report += f"\n{i}. {alt.name} (得分: {score:.1f}/100)\n"
            report += f"   描述: {alt.description}\n"
            report += f"   成本: ${alt.estimated_cost:,.0f} | 时间: {alt.estimated_time}天\n"

            # 详细得分
            report += "   各维度得分:\n"
            for criteria in self.criteria:
                raw_score = alt.scores.get(criteria.name, 0)
                weighted = raw_score * criteria.weight.value
                report += f"     - {criteria.name}: {raw_score}/10 (加权: {weighted})\n"

            report += f"   优点:\n"
            for pro in alt.pros:
                report += f"     + {pro}\n"

            report += f"   缺点:\n"
            for con in alt.cons:
                report += f"     - {con}\n"

        # 推荐
        if ranked:
            best = ranked[0][0]
            report += f"\n推荐方案: {best.name}\n"
            report += f"推荐理由: 综合评分最高，在关键维度表现优秀\n"

        return report

    def export_markdown_table(self) -> str:
        """导出为Markdown表格"""
        # 表头
        header = "| 方案 | " + " | ".join(c.name for c in self.criteria) + " | 总分 | 成本 | 时间 |\n"
        separator = "|------|" + "|".join(["------"] * len(self.criteria)) + "|------|------|------|\n"

        # 数据行
        rows = ""
        ranked = self.rank_alternatives()
        for alt, total_score in ranked:
            row = f"| {alt.name} |"
            for criteria in self.criteria:
                score = alt.scores.get(criteria.name, 0)
                row += f" {score}/10 |"
            row += f" {total_score:.1f} | ${alt.estimated_cost:,.0f} | {alt.estimated_time}d |\n"
            rows += row

        return header + separator + rows

# 使用示例
def decision_matrix_example():
    # 创建决策矩阵：选择消息队列
    matrix = DecisionMatrix("消息队列技术选型")

    # 添加评估维度
    matrix.add_criteria(Criteria(
        "性能", CriteriaWeight.CRITICAL,
        "吞吐量和延迟表现"
    ))
    matrix.add_criteria(Criteria(
        "可靠性", CriteriaWeight.CRITICAL,
        "消息不丢失，高可用"
    ))
    matrix.add_criteria(Criteria(
        "运维成本", CriteriaWeight.HIGH,
        "部署和维护的复杂度"
    ))
    matrix.add_criteria(Criteria(
        "生态", CriteriaWeight.MEDIUM,
        "社区活跃度和工具支持"
    ))
    matrix.add_criteria(Criteria(
        "团队熟悉度", CriteriaWeight.HIGH,
        "团队对技术的掌握程度"
    ))

    # 添加候选方案
    matrix.add_alternative(Alternative(
        name="Kafka",
        description="分布式流处理平台，高吞吐量",
        scores={
            "性能": 10,
            "可靠性": 9,
            "运维成本": 6,
            "生态": 10,
            "团队熟悉度": 8
        },
        pros=[
            "超高吞吐量（百万级QPS）",
            "持久化存储，支持消息回溯",
            "生态完善，工具丰富",
            "适合大数据场景"
        ],
        cons=[
            "运维复杂，需要ZooKeeper",
            "学习曲线陡峭",
            "延迟相对较高（毫秒级）"
        ],
        estimated_cost=50000,
        estimated_time=30
    ))

    matrix.add_alternative(Alternative(
        name="RabbitMQ",
        description="传统消息队列，支持多种协议",
        scores={
            "性能": 7,
            "可靠性": 9,
            "运维成本": 8,
            "生态": 9,
            "团队熟悉度": 9
        },
        pros=[
            "成熟稳定，企业级支持",
            "支持多种消息模式",
            "管理界面友好",
            "延迟低（微秒级）"
        ],
        cons=[
            "吞吐量相对较低",
            "集群模式复杂",
            "Erlang技术栈小众"
        ],
        estimated_cost=30000,
        estimated_time=15
    ))

    matrix.add_alternative(Alternative(
        name="RocketMQ",
        description="阿里开源的分布式消息中间件",
        scores={
            "性能": 9,
            "可靠性": 9,
            "运维成本": 7,
            "生态": 7,
            "团队熟悉度": 5
        },
        pros=[
            "高吞吐量，低延迟",
            "可靠性高，事务消息支持",
            "国内社区活跃",
            "Java技术栈，易集成"
        ],
        cons=[
            "国际化程度不如Kafka",
            "文档相对较少",
            "团队不熟悉"
        ],
        estimated_cost=40000,
        estimated_time=25
    ))

    # 生成报告
    print(matrix.generate_report())
    print("\n" + "="*60 + "\n")
    print("Markdown表格:\n")
    print(matrix.export_markdown_table())

if __name__ == "__main__":
    decision_matrix_example()
```

## 真实案例

### 案例1: 选择API网关

```markdown
# ADR-2024-001: 选择API网关技术

## 状态
已接受 (2024-01-15)

## 背景

我们的微服务架构包含30+个服务，目前存在以下问题：

1. **认证分散**: 每个服务独立实现认证，代码重复
2. **调用混乱**: 前端直接调用各个服务，域名管理困难
3. **限流缺失**: 没有统一的流量控制，容易被打垮
4. **监控盲区**: 无法统一监控API调用情况
5. **协议不统一**: 部分用HTTP，部分用gRPC，前端适配困难

**约束条件**:
- 团队主要使用Go和Java
- 要求99.9%的可用性
- 峰值QPS预计5万+
- 预算有限，倾向开源方案

## 决策

我们将使用 **Kong** 作为API网关，采用数据库模式部署（PostgreSQL）。

具体实施:
- 使用Kong作为统一入口
- 通过Kong插件实现认证、限流、日志
- 保留内部服务间gRPC调用
- 使用Prometheus插件实现监控

## 理由

1. **功能完善**: Kong提供了我们需要的所有功能（认证、限流、监控、协议转换）

2. **性能优异**: 基于OpenResty/Nginx，单实例可支持数万QPS
   - 压测结果：单实例2万QPS，P99延迟 < 10ms

3. **插件生态**: 丰富的官方和社区插件，可快速扩展功能
   - 已有100+插件
   - 支持Lua/Go自定义插件

4. **团队能力**: 团队有Nginx使用经验，学习成本可控

5. **商业支持**: 有企业版可选，提供技术支持和额外功能

## 后果

### 正面影响

1. **统一管理**: 所有外部API通过Kong统一管理
2. **简化开发**: 服务无需实现认证等横切关注点
3. **提升性能**: 通过缓存、限流等策略保护后端
4. **可观测性**: 统一的日志和监控入口

### 负面影响

1. **新的单点**: Kong成为关键路径，需要高可用部署
   - 缓解：部署至少3个实例，使用负载均衡器

2. **运维复杂度**: 需要维护Kong和PostgreSQL
   - 缓解：使用Kubernetes部署，自动化运维

3. **学习成本**: 团队需要学习Kong配置和插件开发
   - 缓解：安排培训，准备文档

4. **数据库依赖**: 使用数据库模式，增加了组件依赖
   - 缓解：PostgreSQL使用主备部署，定期备份

### 风险和缓解措施

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| Kong故障导致全站不可用 | 高 | 中 | 多实例部署+健康检查+快速回滚 |
| 性能不足 | 中 | 低 | 提前压测，预留扩容空间 |
| 插件bug | 中 | 中 | 充分测试，准备回退方案 |
| 数据库故障 | 高 | 低 | 主备+备份+监控告警 |

## 替代方案

### 方案A: Nginx + Lua脚本

**优点**:
- 性能最优，无额外抽象层
- 团队熟悉Nginx
- 部署简单，无数据库依赖

**缺点**:
- 需要大量自研开发
- 配置管理困难
- 动态更新支持弱

**为什么没选**:
开发成本太高，需要从零实现认证、限流等功能，预计需要3个月时间。Kong已经提供了这些能力。

### 方案B: Spring Cloud Gateway

**优点**:
- 与现有Spring Boot服务集成好
- Java技术栈，团队熟悉
- 功能完善

**缺点**:
- 性能不如Nginx based方案
- 需要JVM，资源消耗较大
- Go服务集成需要额外工作

**为什么没选**:
我们服务不全是Java，有较多Go服务。且压测显示Spring Cloud Gateway在高并发下性能不足。

### 方案C: Traefik

**优点**:
- Cloud Native设计，K8s集成好
- 配置简单，自动服务发现
- 支持多种协议

**缺点**:
- 插件生态不如Kong
- 在大规模场景下的实践案例较少
- 限流等高级功能需要企业版

**为什么没选**:
我们需要的高级功能（精细化限流、自定义认证）在开源版本不完善，而企业版价格较高。

### 方案D: 自研网关

**优点**:
- 完全定制化
- 无License限制
- 性能可优化到极致

**缺点**:
- 开发周期长（6个月+）
- 维护成本高
- 缺乏生态和社区

**为什么没选**:
时间和资源不允许，且自研网关不是公司核心竞争力。

## 实施计划

### 第一阶段（2周）
- [x] 搭建Kong开发环境
- [x] POC验证核心功能
- [x] 性能压测
- [x] 编写部署文档

### 第二阶段（3周）
- [ ] 生产环境部署（3实例+PostgreSQL主备）
- [ ] 配置JWT认证插件
- [ ] 配置限流插件
- [ ] 配置Prometheus监控

### 第三阶段（4周）
- [ ] 迁移用户服务API
- [ ] 迁移订单服务API
- [ ] 迁移支付服务API
- [ ] 灰度发布，监控指标

### 第四阶段（持续）
- [ ] 陆续迁移其他服务
- [ ] 优化性能配置
- [ ] 开发自定义插件

## 成功指标

- Kong可用性 ≥ 99.9%
- P99延迟 < 50ms（含网关处理时间）
- 支持峰值5万QPS
- 零安全事故
- API统一管理覆盖率 100%

## 参考资料

- [Kong官方文档](https://docs.konghq.com/)
- [Kong vs 其他网关对比](https://konghq.com/learning-center/api-gateway/comparison)
- [内部POC报告](./poc-reports/kong-poc-2024-01.md)
- [性能测试报告](./test-reports/kong-performance-test.md)

## 元数据

- **作者**: 张三 (zhangsan@company.com)
- **日期**: 2024-01-15
- **评审者**: 李四(架构师), 王五(安全专家), 赵六(运维负责人)
- **状态**: 已接受
- **相关Issue**: ARCH-123, ARCH-124
- **实施负责人**: 张三
- **预计完成时间**: 2024-03-15
```

### 案例2: 数据库分库分表方案

```markdown
# ADR-2024-005: 订单表分库分表方案

## 状态
已接受 (2024-02-20)

## 背景

订单表(orders)已经达到5000万行，查询性能急剧下降：

**当前问题**:
1. 查询慢：普通查询3-5秒，分页查询10秒+
2. 写入慢：插入订单耗时500ms+
3. 备份困难：全量备份需要6小时
4. 容量瓶颈：单表即将达到MySQL推荐上限

**业务特征**:
- 订单每日新增50万
- 预计3年内达到5亿订单
- 主要查询：
  - 按order_id查询（90%）
  - 按user_id查询（8%）
  - 按时间范围查询（2%）

**约束条件**:
- 不能停服
- 历史数据需要保留
- 需要支持跨分片查询

## 决策

采用 **按user_id分库 + 按时间分表** 的方案：

- **分库策略**: 8个数据库，user_id % 8
- **分表策略**: 每月一张表，orders_202401, orders_202402...
- **中间件**: 使用ShardingSphere-JDBC
- **冷热分离**: 6个月以上数据归档到历史库

```
┌─────────────────────────────────────────────┐
│              分库分表架构                    │
├─────────────────────────────────────────────┤
│                                              │
│    ┌──────────┐                             │
│    │   App    │                             │
│    └────┬─────┘                             │
│         │                                    │
│    ┌────▼────────────┐                      │
│    │ ShardingSphere  │                      │
│    └─────┬───────────┘                      │
│          │                                   │
│    ┌─────┴─────┬─────────┬────────┐        │
│    │           │         │         │        │
│ ┌──▼───┐  ┌───▼──┐  ┌───▼──┐  ┌──▼───┐   │
│ │ DB0  │  │ DB1  │  │ DB2  │  │ DB7  │   │
│ │ ┌──┐ │  │ ┌──┐ │  │ ┌──┐ │  │ ┌──┐ │   │
│ │ │01│ │  │ │01│ │  │ │01│ │  │ │01│ │   │
│ │ ├──┤ │  │ ├──┤ │  │ ├──┤ │  │ ├──┤ │   │
│ │ │02│ │  │ │02│ │  │ │02│ │  │ │02│ │   │
│ │ ├──┤ │  │ ├──┤ │  │ ├──┤ │  │ ├──┤ │   │
│ │ │12│ │  │ │12│ │  │ │12│ │  │ │12│ │   │
│ │ └──┘ │  │ └──┘ │  │ └──┘ │  │ └──┘ │   │
│ └──────┘  └──────┘  └──────┘  └──────┘   │
│                                             │
└─────────────────────────────────────────────┘
```

## 理由

1. **user_id分库**: 90%的查询都包含user_id，数据亲和性好

2. **时间分表**: 订单有明显的时间属性，老订单查询少

3. **性能提升**:
   - 单表行数：5000万 → 600万（假设月均）
   - 查询时间：5秒 → 100ms（预估）
   - 写入时间：500ms → 50ms（预估）

4. **扩展性**:
   - 垂直：单库可升级配置
   - 水平：可增加到16库、32库

5. **运维友好**:
   - 备份：分库并行备份，1小时完成
   - 归档：按月清理老表
   - 监控：分库独立监控

## 后果

### 正面影响

1. 查询性能大幅提升
2. 数据增长可持续支撑
3. 备份恢复时间大幅缩短
4. 单点故障影响范围缩小

### 负面影响

1. **跨库事务困难**
   - 缓解：避免跨库事务，使用最终一致性

2. **跨库查询性能差**
   - 缓解：禁止不带user_id的查询，使用ES做全局搜索

3. **运维复杂度上升**
   - 缓解：自动化脚本创建分表，监控告警

4. **数据迁移风险**
   - 缓解：双写+校验+分批切换

### 风险和缓解措施

- **风险1**: 数据迁移过程中数据不一致
  - 缓解：双写验证，自动对账脚本

- **风险2**: user_id分布不均匀导致数据倾斜
  - 缓解：分析user_id分布，必要时调整分片算法

- **风险3**: 全局ID生成冲突
  - 缓解：使用雪花算法，数据中心ID区分

## 替代方案

### 方案A: 只分表不分库

**优点**: 实施简单，无需多数据源
**缺点**: 单库瓶颈仍存在，扩展性有限
**为什么没选**: 无法解决单库连接数和IO瓶颈

### 方案B: 按order_id分库

**优点**: 数据分布均匀
**缺点**: 按user_id查询需要扫描所有库
**为什么没选**: 不符合查询模式

### 方案C: 迁移到NoSQL（MongoDB）

**优点**: 天然支持分片，扩展性好
**缺点**:
- 需要重写所有SQL
- 事务支持弱
- 团队不熟悉

**为什么没选**: 改造成本太高，风险大

## 实施计划

### 阶段1: 准备（2周）
- [ ] 搭建ShardingSphere开发环境
- [ ] 编写分库分表配置
- [ ] 开发双写逻辑
- [ ] 编写数据校验脚本

### 阶段2: 验证（2周）
- [ ] 在测试环境完整迁移
- [ ] 压力测试
- [ ] 数据一致性验证

### 阶段3: 迁移（4周）
- [ ] 开启双写（新数据写入分片）
- [ ] 历史数据分批迁移
- [ ] 实时对账验证
- [ ] 灰度切换读流量

### 阶段4: 收尾（1周）
- [ ] 全量切换
- [ ] 关闭双写
- [ ] 清理老表
- [ ] 优化监控

## 成功指标

- 查询P99 < 100ms
- 写入P99 < 50ms
- 数据一致性 100%
- 迁移过程零故障

## 参考资料

- [ShardingSphere文档](https://shardingsphere.apache.org/)
- [分库分表实践案例](internal-doc/sharding-best-practice.md)
- [数据迁移方案](internal-doc/data-migration-plan.md)

## 元数据

- **作者**: 李四
- **日期**: 2024-02-20
- **评审者**: 张三, 王五, DBA团队
- **相关ADR**: ADR-2024-003 (分布式ID生成)
```

### 案例3: 缓存策略选择

```markdown
# ADR-2024-008: 商品详情页缓存策略

## 状态
已接受 (2024-03-10)

## 背景

商品详情页是流量最大的页面，日均PV 1000万，数据库压力巨大。

**现状**:
- 直接查询MySQL，P99响应时间800ms
- 高峰期数据库连接数打满
- 数据库主从延迟导致数据不一致

**需求**:
- 响应时间 < 100ms
- 支持百万级QPS
- 数据实时性要求：1分钟内

## 决策

采用 **三级缓存架构**：

```
┌────────────────────────────────────────────┐
│          三级缓存架构                       │
├────────────────────────────────────────────┤
│                                             │
│  请求 ──▶ CDN (L1)                         │
│            │ Miss                           │
│            ▼                                │
│         Nginx本地缓存 (L2)                 │
│            │ Miss                           │
│            ▼                                │
│         Redis (L3)                         │
│            │ Miss                           │
│            ▼                                │
│         MySQL                              │
│                                             │
└────────────────────────────────────────────┘
```

**策略**:
1. **L1 - CDN**: 静态资源+页面HTML，TTL 5分钟
2. **L2 - Nginx**: 热点商品，TTL 1分钟，LRU淘汰
3. **L3 - Redis**: 全量商品，TTL 10分钟
4. **更新**: 商品修改时主动清除缓存

## 理由

1. **性能**: 三级缓存确保99%请求不打DB
2. **成本**: CDN承载静态资源，节省带宽
3. **实时性**: 1分钟TTL满足业务需求
4. **高可用**: 缓存失效时可降级查DB

## 后果

### 正面影响
- DB压力下降95%
- 响应时间从800ms降至50ms
- 支持10倍流量增长

### 负面影响
- 缓存击穿风险 → 使用Singleflight防止
- 缓存不一致 → 主动清除+TTL兜底
- 运维复杂度 → 统一配置管理

## 元数据
- **作者**: 王五
- **日期**: 2024-03-10
- **状态**: 已实施
```

### 案例4: 技术债务决策

```markdown
# ADR-2024-012: 暂不重构遗留支付模块

## 状态
已接受 (2024-04-05)

## 背景

支付模块是5年前开发的遗留代码，存在以下问题：
- 代码混乱，耦合严重
- 测试覆盖率不足30%
- 使用过时的框架
- 每次修改都可能引入bug

技术团队强烈要求重构，预计需要3个月。

## 决策

**暂不重构**，而是：
1. 冻结功能，只修bug
2. 增加集成测试
3. 文档化关键逻辑
4. 列入下半年规划

## 理由

1. **时间紧迫**: Q2有重要业务目标，无法抽调人力
2. **风险高**: 支付是核心模块，重构风险极大
3. **可控**: 当前虽不优雅但运行稳定
4. **ROI低**: 重构3个月vs每月1-2个bug修复

## 后果

### 正面影响
- 资源集中于业务目标
- 避免重构引入新bug的风险

### 负面影响
- 技术债务继续累积
- 团队士气可能受影响

### 缓解措施
- 明确重构时间(Q3)，给团队希望
- 暂时增加代码审查强度
- 记录所有workaround，方便重构时参考

## 替代方案

### 方案A: 立即重构
**为什么没选**: 时间不允许，风险太高

### 方案B: 渐进式重构
**为什么没选**: 支付模块耦合太严重，难以渐进

### 方案C: 微服务拆分
**为什么没选**: 比重构更复杂，时间更长

## 重新评估条件

如果出现以下情况，重新评估决策：
- 支付bug频率超过5个/月
- 出现严重生产事故
- Q2业务目标提前完成

## 元数据
- **作者**: 张三
- **日期**: 2024-04-05
- **复审日期**: 2024-07-01
- **状态**: 已接受（附条件）
```

### 案例5: 安全架构决策

```markdown
# ADR-2024-015: 零信任网络架构

## 状态
已接受 (2024-05-01)

## 背景

公司发生了一次内网渗透事件，暴露了传统"城堡-护城河"安全模型的缺陷：
- 内网默认信任，一旦突破边界即可横向移动
- 缺少细粒度访问控制
- 无法识别异常内部流量

**新的安全要求**:
- 最小权限原则
- 持续验证
- 微隔离

## 决策

实施 **零信任网络架构 (Zero Trust)**:

```
┌──────────────────────────────────────────────┐
│           零信任架构                          │
├──────────────────────────────────────────────┤
│                                               │
│  每个请求都需要：                             │
│  1. 身份认证 (mTLS)                          │
│  2. 设备认证 (Device Trust)                  │
│  3. 上下文验证 (IP/Time/Behavior)            │
│  4. 权限检查 (RBAC/ABAC)                     │
│  5. 持续监控 (Audit Log)                     │
│                                               │
│  ┌─────┐   verify   ┌──────────┐            │
│  │User │───────────▶│  Policy  │            │
│  └─────┘            │  Engine  │            │
│                     └──────────┘             │
│                           │                  │
│                           ▼                  │
│  ┌──────────────────────────────┐           │
│  │  Service A │ Service B │ DB  │           │
│  └──────────────────────────────┘           │
│                                               │
└──────────────────────────────────────────────┘
```

**实施**:
1. 服务间通信启用mTLS (Istio)
2. 实施RBAC策略 (OPA)
3. 部署网络策略 (NetworkPolicy)
4. 集中审计日志 (ELK)

## 理由

1. **安全性**: 即使突破边界也无法横向移动
2. **合规**: 满足等保2.0要求
3. **可见性**: 所有访问可审计
4. **灵活性**: 细粒度权限控制

## 后果

### 正面影响
- 安全性大幅提升
- 满足合规要求
- 异常行为可快速发现

### 负面影响
- 性能开销：mTLS增加5-10ms延迟
- 复杂度：证书管理、策略配置
- 学习成本：团队需要培训

### 实施计划

**阶段1 (2个月)**:
- 部署Service Mesh (Istio)
- 启用mTLS

**阶段2 (1个月)**:
- 配置RBAC策略
- 集成身份提供商

**阶段3 (1个月)**:
- 部署审计日志系统
- 配置告警规则

## 元数据
- **作者**: 赵六 (安全团队)
- **日期**: 2024-05-01
- **评审者**: 架构组、安全组、运维组
- **预算**: $150,000
- **预计完成**: 2024-09-01
```

## 工具支持

### ADR命令行工具

```python
# adr_cli.py
import os
import re
from datetime import datetime
from pathlib import Path

class ADRManager:
    """ADR管理工具"""

    def __init__(self, adr_dir: str = "./docs/adr"):
        self.adr_dir = Path(adr_dir)
        self.adr_dir.mkdir(parents=True, exist_ok=True)

    def create_adr(self, title: str, status: str = "提议中") -> str:
        """创建新ADR"""
        # 生成ADR编号
        adr_number = self._get_next_number()
        filename = f"ADR-{adr_number:04d}-{self._slugify(title)}.md"
        filepath = self.adr_dir / filename

        # 生成内容
        content = self._generate_template(adr_number, title, status)

        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"创建ADR: {filepath}")
        return str(filepath)

    def list_adrs(self, status_filter: str = None) -> list:
        """列出所有ADR"""
        adrs = []
        for file in sorted(self.adr_dir.glob("ADR-*.md")):
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 提取元数据
            number_match = re.search(r'ADR-(\d+)', file.name)
            title_match = re.search(r'# ADR-\d+: (.+)', content)
            status_match = re.search(r'## 状态\s+(.+)', content)

            if all([number_match, title_match, status_match]):
                status = status_match.group(1).strip()

                if status_filter and status != status_filter:
                    continue

                adrs.append({
                    'number': int(number_match.group(1)),
                    'title': title_match.group(1).strip(),
                    'status': status,
                    'file': str(file)
                })

        return adrs

    def supersede_adr(self, old_number: int, new_title: str):
        """替代旧ADR"""
        # 创建新ADR
        new_file = self.create_adr(new_title, "已接受")

        # 更新旧ADR状态
        old_files = list(self.adr_dir.glob(f"ADR-{old_number:04d}-*.md"))
        if old_files:
            old_file = old_files[0]
            with open(old_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 提取新ADR编号
            new_number = self._get_current_number()

            # 更新状态
            content = re.sub(
                r'## 状态\s+(.+)',
                f'## 状态\n已替代\n\n被 [ADR-{new_number:04d}]({Path(new_file).name}) 替代',
                content
            )

            with open(old_file, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"ADR-{old_number:04d} 已标记为已替代")

    def generate_index(self) -> str:
        """生成ADR索引"""
        adrs = self.list_adrs()

        index = "# 架构决策记录索引\n\n"

        # 按状态分组
        by_status = {}
        for adr in adrs:
            status = adr['status']
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(adr)

        # 生成索引
        for status in ['已接受', '提议中', '已废弃', '已拒绝', '已替代']:
            if status in by_status:
                index += f"\n## {status}\n\n"
                for adr in by_status[status]:
                    filename = Path(adr['file']).name
                    index += f"- [ADR-{adr['number']:04d}]({filename}): {adr['title']}\n"

        # 写入索引文件
        index_file = self.adr_dir / "README.md"
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(index)

        print(f"生成索引: {index_file}")
        return index

    def _get_next_number(self) -> int:
        """获取下一个ADR编号"""
        current = self._get_current_number()
        return current + 1

    def _get_current_number(self) -> int:
        """获取当前最大ADR编号"""
        max_number = 0
        for file in self.adr_dir.glob("ADR-*.md"):
            match = re.search(r'ADR-(\d+)', file.name)
            if match:
                number = int(match.group(1))
                max_number = max(max_number, number)
        return max_number

    def _slugify(self, text: str) -> str:
        """转换为文件名友好格式"""
        # 简化版，实际应该更完善
        text = text.lower()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '-', text)
        return text[:50]  # 限制长度

    def _generate_template(self, number: int, title: str, status: str) -> str:
        """生成ADR模板"""
        today = datetime.now().strftime('%Y-%m-%d')

        return f"""# ADR-{number:04d}: {title}

## 状态
{status}

## 背景 (Context)

描述需要做决策的背景和问题。

## 决策 (Decision)

我们将要采取的决策。

## 理由 (Rationale)

为什么做这个决策。

## 后果 (Consequences)

### 正面影响
-

### 负面影响
-

### 风险和缓解措施
-

## 替代方案 (Alternatives)

### 方案A:
- 优点:
- 缺点:
- 为什么没选:

## 参考资料 (References)
-

## 元数据
- 作者:
- 日期: {today}
- 评审者:
- 相关ADR:
- 相关Issue:
"""

# CLI接口
if __name__ == "__main__":
    import sys

    manager = ADRManager()

    if len(sys.argv) < 2:
        print("用法:")
        print("  python adr_cli.py new <标题>         # 创建新ADR")
        print("  python adr_cli.py list [状态]        # 列出ADR")
        print("  python adr_cli.py supersede <编号> <新标题>  # 替代旧ADR")
        print("  python adr_cli.py index              # 生成索引")
        sys.exit(1)

    command = sys.argv[1]

    if command == "new" and len(sys.argv) >= 3:
        title = " ".join(sys.argv[2:])
        manager.create_adr(title)

    elif command == "list":
        status_filter = sys.argv[2] if len(sys.argv) >= 3 else None
        adrs = manager.list_adrs(status_filter)
        for adr in adrs:
            print(f"ADR-{adr['number']:04d} [{adr['status']}] {adr['title']}")

    elif command == "supersede" and len(sys.argv) >= 4:
        old_number = int(sys.argv[2])
        new_title = " ".join(sys.argv[3:])
        manager.supersede_adr(old_number, new_title)

    elif command == "index":
        manager.generate_index()

    else:
        print("未知命令")
        sys.exit(1)
```

### 使用示例

```bash
# 创建新ADR
python adr_cli.py new "选择消息队列技术"

# 列出所有ADR
python adr_cli.py list

# 列出特定状态的ADR
python adr_cli.py list "已接受"

# 替代旧ADR
python adr_cli.py supersede 5 "改用RocketMQ作为消息队列"

# 生成索引
python adr_cli.py index
```

## 最佳实践

### ADR编写原则

1. **简洁明了** - 不要写长篇大论，关键信息突出
2. **决策为主** - 重点记录"是什么"和"为什么"，而非"怎么做"
3. **及时记录** - 决策做出时立即记录，不要事后补充
4. **保持不变** - ADR一旦接受就不应修改，新决策创建新ADR
5. **团队共识** - ADR应该经过团队讨论和评审

### ADR管理建议

1. **集中存储** - 所有ADR放在代码仓库的docs/adr目录
2. **版本控制** - ADR作为代码一部分，使用Git管理
3. **定期评审** - 季度回顾ADR，更新状态
4. **索引目录** - 维护README索引，方便查找
5. **模板规范** - 统一使用标准模板

### 常见错误

1. **记录实现细节** - ADR关注决策，不是实现文档
2. **追求完美** - 不需要把所有细节都写清楚
3. **害怕犯错** - 决策可以被替代，记录错误决策也有价值
4. **堆积不写** - 决策后立即记录，不要拖延
5. **只有架构师写** - 任何有决策的人都可以写ADR

## 总结

ADR是一种轻量级但非常有价值的架构文档实践。通过记录架构决策的背景、理由和后果，可以：

- 保留架构知识，避免团队失忆
- 让新人快速理解系统演进历史
- 为技术债务提供可追溯的证据
- 促进团队对架构的共同理解

**关键要点**:
1. 简洁高效 - 模板简单，快速记录
2. 决策导向 - 关注"为什么"而非"怎么做"
3. 持续维护 - 纳入日常工作流程
4. 工具支持 - 使用脚本或工具简化管理
5. 团队文化 - 让写ADR成为团队习惯

**进阶阅读**:
- [adr.github.io](https://adr.github.io/)
- [Architecture Decision Records by Michael Nygard](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR工具集合](https://github.com/npryce/adr-tools)
