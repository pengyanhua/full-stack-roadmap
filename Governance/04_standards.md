# 技术标准与规范

## 目录
- [技术选型标准](#技术选型标准)
- [编码规范](#编码规范)
- [数据库规范](#数据库规范)
- [API设计规范](#api设计规范)
- [安全规范](#安全规范)
- [自动化检查](#自动化检查)
- [规范治理](#规范治理)

## 技术选型标准

### 选型决策框架

```
┌────────────────────────────────────────────────────────┐
│              技术选型决策矩阵                           │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐          │
│  │  业务匹配 │──▶│  技术成熟 │──▶│  团队能力 │          │
│  │  30%     │   │  25%     │   │  20%     │          │
│  └──────────┘   └──────────┘   └──────────┘          │
│       │              │              │                  │
│       └──────────────┼──────────────┘                  │
│                      ▼                                 │
│              ┌──────────────┐                          │
│              │  综合评分    │                          │
│              │  80分以上    │                          │
│              └──────────────┘                          │
│                      │                                 │
│       ┌──────────────┼──────────────┐                  │
│       ▼              ▼              ▼                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐          │
│  │  成本控制 │   │  风险评估 │   │  生态支持 │          │
│  │  15%     │   │  5%      │   │  5%      │          │
│  └──────────┘   └──────────┘   └──────────┘          │
│                                                         │
└────────────────────────────────────────────────────────┘
```

### 技术选型清单

#### 编程语言选型

```python
# technology_selection_criteria.py
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class MaturityLevel(Enum):
    """成熟度等级"""
    EXPERIMENTAL = 1  # 实验性
    EMERGING = 2      # 新兴
    MAINSTREAM = 3    # 主流
    MATURE = 4        # 成熟
    LEGACY = 5        # 遗留

@dataclass
class TechnologyCriteria:
    """技术选型标准"""
    # 业务匹配度 (30%)
    business_fit: int  # 1-10分
    performance_req: int  # 性能要求匹配
    scalability_req: int  # 扩展性要求匹配

    # 技术成熟度 (25%)
    maturity_level: MaturityLevel
    community_activity: int  # 社区活跃度
    production_cases: int    # 生产案例数

    # 团队能力 (20%)
    team_experience: int     # 团队经验
    learning_curve: int      # 学习曲线(低分=易学)
    available_talent: int    # 可招聘人才

    # 成本 (15%)
    licensing_cost: float    # 许可成本
    infrastructure_cost: float  # 基础设施成本
    maintenance_cost: float     # 维护成本

    # 风险 (5%)
    vendor_lock_in: int      # 供应商锁定风险(低分=低风险)
    tech_debt_risk: int      # 技术债务风险

    # 生态 (5%)
    tool_support: int        # 工具支持
    library_ecosystem: int   # 库生态系统

    def calculate_score(self) -> float:
        """计算综合得分"""
        # 业务匹配 (30%)
        business_score = (
            self.business_fit * 0.4 +
            self.performance_req * 0.3 +
            self.scalability_req * 0.3
        ) * 0.3

        # 技术成熟度 (25%)
        maturity_score = (
            self.maturity_level.value * 2 +  # 归一化到10分
            self.community_activity * 0.4 +
            min(self.production_cases / 10, 10) * 0.4
        ) * 0.25

        # 团队能力 (20%)
        team_score = (
            self.team_experience * 0.5 +
            (10 - self.learning_curve) * 0.25 +  # 反转学习曲线
            self.available_talent * 0.25
        ) * 0.2

        # 成本 (15%) - 归一化
        total_cost = self.licensing_cost + self.infrastructure_cost + self.maintenance_cost
        cost_score = max(0, 10 - total_cost / 10000) * 0.15

        # 风险 (5%) - 反转风险
        risk_score = (
            (10 - self.vendor_lock_in) * 0.6 +
            (10 - self.tech_debt_risk) * 0.4
        ) * 0.05

        # 生态 (5%)
        ecosystem_score = (
            self.tool_support * 0.5 +
            self.library_ecosystem * 0.5
        ) * 0.05

        total = business_score + maturity_score + team_score + cost_score + risk_score + ecosystem_score
        return round(total, 2)

# 使用示例
def evaluate_language_options():
    """评估编程语言选项"""

    # Go语言
    golang = TechnologyCriteria(
        business_fit=9,
        performance_req=10,
        scalability_req=10,
        maturity_level=MaturityLevel.MAINSTREAM,
        community_activity=9,
        production_cases=50,
        team_experience=6,
        learning_curve=4,  # 相对容易
        available_talent=7,
        licensing_cost=0,
        infrastructure_cost=5000,
        maintenance_cost=10000,
        vendor_lock_in=2,
        tech_debt_risk=3,
        tool_support=8,
        library_ecosystem=8
    )

    # Java
    java = TechnologyCriteria(
        business_fit=8,
        performance_req=8,
        scalability_req=9,
        maturity_level=MaturityLevel.MATURE,
        community_activity=10,
        production_cases=100,
        team_experience=9,
        learning_curve=6,  # 较复杂
        available_talent=10,
        licensing_cost=0,
        infrastructure_cost=8000,
        maintenance_cost=15000,
        vendor_lock_in=3,
        tech_debt_risk=4,
        tool_support=10,
        library_ecosystem=10
    )

    # Python
    python = TechnologyCriteria(
        business_fit=7,
        performance_req=6,
        scalability_req=7,
        maturity_level=MaturityLevel.MATURE,
        community_activity=10,
        production_cases=80,
        team_experience=7,
        learning_curve=2,  # 很容易
        available_talent=9,
        licensing_cost=0,
        infrastructure_cost=6000,
        maintenance_cost=12000,
        vendor_lock_in=2,
        tech_debt_risk=5,
        tool_support=9,
        library_ecosystem=10
    )

    options = {
        "Go": golang,
        "Java": java,
        "Python": python
    }

    print("编程语言选型评估\n" + "="*50)
    for name, criteria in sorted(options.items(), key=lambda x: x[1].calculate_score(), reverse=True):
        score = criteria.calculate_score()
        print(f"{name}: {score}/10 分")
        print(f"  - 业务匹配: {criteria.business_fit}/10")
        print(f"  - 成熟度: {criteria.maturity_level.name}")
        print(f"  - 团队经验: {criteria.team_experience}/10")
        print(f"  - 总成本: ${criteria.licensing_cost + criteria.infrastructure_cost + criteria.maintenance_cost:,.0f}")
        print()

if __name__ == "__main__":
    evaluate_language_options()
```

### 框架选型标准

| 维度 | 权重 | 评分标准 | 及格线 |
|------|------|----------|--------|
| License | 必要 | Apache 2.0/MIT优先 | 必须开源 |
| 成熟度 | 20% | 发布3年+，10K+ stars | 7分 |
| 性能 | 25% | 基准测试达标 | 8分 |
| 文档 | 15% | 中文文档完整 | 7分 |
| 社区 | 15% | 活跃提交，快速响应 | 7分 |
| 团队 | 25% | 至少1人熟悉 | 必须 |

### 数据库选型标准

```
数据库选型决策树
=================

START
  │
  ▼
数据结构化？
  ├── 是 ──▶ OLTP还是OLAP？
  │         ├── OLTP ──▶ 事务要求强？
  │         │           ├── 是 ──▶ MySQL/PostgreSQL
  │         │           └── 否 ──▶ MongoDB/Cassandra
  │         └── OLAP ──▶ ClickHouse/Doris
  │
  └── 否 ──▶ 数据类型？
            ├── 文档 ──▶ MongoDB/ES
            ├── 时序 ──▶ InfluxDB/TimescaleDB
            ├── 图   ──▶ Neo4j/JanusGraph
            └── KV   ──▶ Redis/RocksDB
```

## 编码规范

### Java编码规范

#### 命名规范

```java
// NamingStandards.java
public class NamingStandards {

    // 类名：大驼峰，名词
    public class UserService { }
    public class OrderController { }

    // 接口名：大驼峰，形容词或名词
    public interface Serializable { }
    public interface UserRepository { }

    // 方法名：小驼峰，动词开头
    public void calculateTotal() { }
    public boolean isValid() { }
    public User getUserById(Long id) { }

    // 变量名：小驼峰，名词
    private String userName;
    private int orderCount;

    // 常量名：全大写，下划线分隔
    public static final int MAX_SIZE = 100;
    public static final String DEFAULT_ENCODING = "UTF-8";

    // 包名：全小写，单数
    // com.company.project.module.layer
    // 正确: com.example.shop.user.service
    // 错误: com.example.shop.users.services

    // 枚举类：大驼峰，枚举值全大写
    public enum OrderStatus {
        PENDING,
        PAID,
        SHIPPED,
        DELIVERED,
        CANCELLED
    }

    // 泛型：单个大写字母
    public <T> List<T> convert(List<T> source) {
        return source;
    }

    // 集合命名：类型+名词复数
    private List<User> userList;
    private Map<String, Order> orderMap;
    private Set<Long> userIdSet;

    // 布尔变量：is/has/can/should开头
    private boolean isActive;
    private boolean hasPermission;
    private boolean canDelete;
}
```

#### 代码格式

```java
// CodeFormatting.java
public class CodeFormatting {

    // 1. 缩进：4个空格，禁止Tab
    public void example() {
        if (condition) {
            // 4个空格缩进
            doSomething();
        }
    }

    // 2. 行长度：不超过120字符
    public void longLine() {
        // 正确：换行并对齐
        String result = doSomething(param1, param2)
                .thenDoAnother(param3)
                .finallyReturn();

        // 方法参数过多时换行
        public void manyParameters(
                String param1,
                Integer param2,
                Boolean param3) {
            // ...
        }
    }

    // 3. 空行：逻辑块之间空一行
    public void emptyLines() {
        // 变量声明
        int a = 1;
        int b = 2;

        // 逻辑块1
        if (a > b) {
            doSomething();
        }

        // 逻辑块2
        for (int i = 0; i < 10; i++) {
            process(i);
        }

        // 返回
        return;
    }

    // 4. 空格：操作符两边加空格
    public void spaces() {
        // 正确
        int sum = a + b;
        boolean result = (a > b) && (c < d);

        // 方法调用不加空格
        doSomething(param1, param2);

        // if/for/while 关键字后加空格
        if (condition) { }
        for (int i = 0; i < 10; i++) { }
    }

    // 5. 大括号：K&R风格（同行开始）
    public void braces() {
        // 正确
        if (condition) {
            doSomething();
        } else {
            doOther();
        }

        // 单行也要加大括号
        if (condition) {
            return;
        }

        // 错误：不加大括号
        // if (condition)
        //     return;
    }
}
```

#### 代码质量规范

```java
// CodeQuality.java
public class CodeQuality {

    // 1. 方法长度：不超过50行
    public void shortMethod() {
        // 如果超过50行，拆分为多个方法
    }

    // 2. 参数数量：不超过5个
    // 正确：使用对象封装参数
    public void createUser(UserCreateRequest request) {
        // request包含所有参数
    }

    // 错误：参数过多
    // public void createUser(String name, String email, int age, String phone, String address) { }

    // 3. 循环复杂度：不超过10
    public void lowComplexity() {
        // 避免深层嵌套
        if (condition1) {
            return;  // 提前返回
        }

        if (condition2) {
            return;
        }

        // 主逻辑
        doSomething();
    }

    // 4. 魔法数字：使用常量
    private static final int MAX_RETRY_COUNT = 3;
    private static final long TIMEOUT_MILLISECONDS = 5000L;

    public void noMagicNumbers() {
        // 正确
        for (int i = 0; i < MAX_RETRY_COUNT; i++) {
            retry();
        }

        // 错误
        // for (int i = 0; i < 3; i++) { }
    }

    // 5. 异常处理：不吞异常
    public void exceptionHandling() {
        try {
            riskyOperation();
        } catch (Exception e) {
            // 正确：记录日志并处理
            log.error("Operation failed", e);
            throw new BusinessException("Failed to process", e);
        }

        // 错误：吞异常
        // try {
        //     riskyOperation();
        // } catch (Exception e) {
        //     // 什么都不做
        // }
    }

    // 6. 资源关闭：使用try-with-resources
    public void resourceManagement() {
        // 正确
        try (InputStream is = new FileInputStream("file.txt");
             BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
            String line = reader.readLine();
        } catch (IOException e) {
            log.error("Failed to read file", e);
        }
    }

    // 7. 空值检查
    public void nullCheck(String input) {
        // 正确：参数校验
        if (input == null || input.isEmpty()) {
            throw new IllegalArgumentException("Input cannot be null or empty");
        }

        // 使用Optional避免空指针
        Optional<User> userOpt = findUser(id);
        userOpt.ifPresent(user -> process(user));

        // 返回空集合而非null
        public List<User> getUsers() {
            return Collections.emptyList();  // 而非 return null;
        }
    }

    // 8. 日志规范
    public void loggingStandards() {
        // 使用正确的日志级别
        log.trace("Entering method with params: {}", params);  // 追踪
        log.debug("Processing item: {}", item);                // 调试
        log.info("User {} logged in", userId);                 // 信息
        log.warn("Retry attempt {} failed", attemptCount);     // 警告
        log.error("Failed to process order {}", orderId, e);   // 错误

        // 使用占位符而非字符串拼接
        // 正确
        log.info("User {} ordered product {}", userId, productId);
        // 错误
        // log.info("User " + userId + " ordered product " + productId);

        // 敏感信息脱敏
        log.info("User phone: {}****", phone.substring(0, 3));
    }

    // 9. 注释规范
    /**
     * 计算订单总金额（含税）
     *
     * @param order 订单对象，不能为null
     * @return 订单总金额，单位：元
     * @throws IllegalArgumentException 如果订单为null或无效
     */
    public BigDecimal calculateOrderTotal(Order order) {
        // 实现...
        return total;
    }

    // 10. TODO/FIXME标记
    public void todoMarkers() {
        // TODO: 实现缓存逻辑 (张三 2024-01-15)
        // FIXME: 修复并发问题 (李四 2024-01-16)
        // XXX: 临时方案，需要重构 (王五 2024-01-17)
    }
}
```

### Checkstyle配置

```xml
<!-- checkstyle.xml -->
<?xml version="1.0"?>
<!DOCTYPE module PUBLIC
        "-//Checkstyle//DTD Checkstyle Configuration 1.3//EN"
        "https://checkstyle.org/dtds/configuration_1_3.dtd">

<module name="Checker">
    <!-- 文件编码 -->
    <property name="charset" value="UTF-8"/>

    <!-- 文件扩展名 -->
    <property name="fileExtensions" value="java, properties, xml"/>

    <!-- 排除文件 -->
    <module name="BeforeExecutionExclusionFileFilter">
        <property name="fileNamePattern" value="module\-info\.java$"/>
    </module>

    <!-- 抑制警告过滤器 -->
    <module name="SuppressionFilter">
        <property name="file" value="${config_loc}/suppressions.xml"/>
        <property name="optional" value="false"/>
    </module>

    <!-- 行长度检查 -->
    <module name="LineLength">
        <property name="max" value="120"/>
        <property name="ignorePattern" value="^package.*|^import.*|a href|href|http://|https://|ftp://"/>
    </module>

    <!-- TreeWalker模块 -->
    <module name="TreeWalker">
        <!-- 命名检查 -->
        <module name="PackageName">
            <property name="format" value="^[a-z]+(\.[a-z][a-z0-9]*)*$"/>
        </module>

        <module name="TypeName">
            <property name="format" value="^[A-Z][a-zA-Z0-9]*$"/>
        </module>

        <module name="MethodName">
            <property name="format" value="^[a-z][a-zA-Z0-9]*$"/>
        </module>

        <module name="ConstantName">
            <property name="format" value="^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$"/>
        </module>

        <module name="LocalVariableName">
            <property name="format" value="^[a-z][a-zA-Z0-9]*$"/>
        </module>

        <module name="MemberName">
            <property name="format" value="^[a-z][a-zA-Z0-9]*$"/>
        </module>

        <module name="ParameterName">
            <property name="format" value="^[a-z][a-zA-Z0-9]*$"/>
        </module>

        <!-- 代码格式检查 -->
        <module name="Indentation">
            <property name="basicOffset" value="4"/>
            <property name="braceAdjustment" value="0"/>
            <property name="caseIndent" value="4"/>
            <property name="throwsIndent" value="4"/>
            <property name="lineWrappingIndentation" value="4"/>
            <property name="arrayInitIndent" value="4"/>
        </module>

        <module name="WhitespaceAround">
            <property name="allowEmptyConstructors" value="true"/>
            <property name="allowEmptyMethods" value="true"/>
            <property name="allowEmptyTypes" value="true"/>
            <property name="allowEmptyLoops" value="true"/>
        </module>

        <module name="LeftCurly">
            <property name="option" value="eol"/>
        </module>

        <module name="RightCurly">
            <property name="option" value="same"/>
        </module>

        <!-- 代码质量检查 -->
        <module name="MethodLength">
            <property name="tokens" value="METHOD_DEF"/>
            <property name="max" value="50"/>
        </module>

        <module name="ParameterNumber">
            <property name="max" value="5"/>
            <property name="tokens" value="METHOD_DEF"/>
        </module>

        <module name="CyclomaticComplexity">
            <property name="max" value="10"/>
        </module>

        <!-- 导入检查 -->
        <module name="AvoidStarImport"/>
        <module name="RedundantImport"/>
        <module name="UnusedImports"/>

        <!-- 注释检查 -->
        <module name="JavadocMethod">
            <property name="scope" value="public"/>
            <property name="allowMissingParamTags" value="false"/>
            <property name="allowMissingReturnTag" value="false"/>
        </module>

        <module name="JavadocType">
            <property name="scope" value="public"/>
        </module>

        <!-- 其他检查 -->
        <module name="EmptyBlock"/>
        <module name="NeedBraces"/>
        <module name="SimplifyBooleanExpression"/>
        <module name="SimplifyBooleanReturn"/>
        <module name="MagicNumber">
            <property name="ignoreNumbers" value="-1, 0, 1, 2"/>
            <property name="ignoreHashCodeMethod" value="true"/>
            <property name="ignoreAnnotation" value="true"/>
        </module>

        <module name="TodoComment">
            <property name="format" value="(TODO)|(FIXME)"/>
        </module>
    </module>
</module>
```

### PMD配置

```xml
<!-- pmd-ruleset.xml -->
<?xml version="1.0"?>
<ruleset name="Custom Rules"
         xmlns="http://pmd.sourceforge.net/ruleset/2.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://pmd.sourceforge.net/ruleset/2.0.0
         https://pmd.sourceforge.io/ruleset_2_0_0.xsd">

    <description>PMD代码检查规则</description>

    <!-- 基础规则集 -->
    <rule ref="category/java/bestpractices.xml">
        <exclude name="JUnitTestsShouldIncludeAssert"/>
    </rule>

    <rule ref="category/java/codestyle.xml">
        <exclude name="OnlyOneReturn"/>
        <exclude name="AtLeastOneConstructor"/>
        <exclude name="CommentDefaultAccessModifier"/>
    </rule>

    <rule ref="category/java/design.xml">
        <exclude name="LawOfDemeter"/>
        <exclude name="LoosePackageCoupling"/>
    </rule>

    <rule ref="category/java/performance.xml"/>
    <rule ref="category/java/security.xml"/>

    <!-- 自定义规则 -->
    <rule ref="category/java/errorprone.xml/AvoidCatchingNPE"/>
    <rule ref="category/java/errorprone.xml/EmptyCatchBlock"/>

    <!-- 命名规则 -->
    <rule ref="category/java/codestyle.xml/ClassNamingConventions"/>
    <rule ref="category/java/codestyle.xml/MethodNamingConventions"/>
    <rule ref="category/java/codestyle.xml/VariableNamingConventions"/>

    <!-- 复杂度规则 -->
    <rule ref="category/java/design.xml/CyclomaticComplexity">
        <properties>
            <property name="classReportLevel" value="80"/>
            <property name="methodReportLevel" value="10"/>
        </properties>
    </rule>

    <rule ref="category/java/design.xml/NPathComplexity">
        <properties>
            <property name="reportLevel" value="200"/>
        </properties>
    </rule>

    <!-- 大小规则 -->
    <rule ref="category/java/design.xml/ExcessiveMethodLength">
        <properties>
            <property name="minimum" value="50"/>
        </properties>
    </rule>

    <rule ref="category/java/design.xml/ExcessiveParameterList">
        <properties>
            <property name="minimum" value="5"/>
        </properties>
    </rule>

    <rule ref="category/java/design.xml/TooManyFields">
        <properties>
            <property name="maxfields" value="15"/>
        </properties>
    </rule>

    <rule ref="category/java/design.xml/TooManyMethods">
        <properties>
            <property name="maxmethods" value="20"/>
        </properties>
    </rule>
</ruleset>
```

## 数据库规范

### 表设计规范

```sql
-- database_standards.sql

-- ============================================
-- 1. 命名规范
-- ============================================

-- 表名：小写，下划线分隔，复数形式
CREATE TABLE users (
    id BIGINT PRIMARY KEY,
    name VARCHAR(100)
);

CREATE TABLE order_items (
    id BIGINT PRIMARY KEY,
    order_id BIGINT
);

-- 字段名：小写，下划线分隔，避免保留字
CREATE TABLE products (
    id BIGINT PRIMARY KEY,
    product_name VARCHAR(200),  -- 而非 name
    product_code VARCHAR(50),
    created_at DATETIME,
    updated_at DATETIME
);

-- 索引名：idx_表名_字段名
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_orders_user_id_created_at ON orders(user_id, created_at);

-- 外键名：fk_从表_主表
ALTER TABLE orders ADD CONSTRAINT fk_orders_users
    FOREIGN KEY (user_id) REFERENCES users(id);

-- ============================================
-- 2. 字段规范
-- ============================================

CREATE TABLE standard_table (
    -- 主键：id，BIGINT UNSIGNED，自增
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,

    -- 外键：表名_id
    user_id BIGINT UNSIGNED NOT NULL,
    product_id BIGINT UNSIGNED NOT NULL,

    -- 字符串：VARCHAR，明确长度
    name VARCHAR(100) NOT NULL COMMENT '姓名',
    email VARCHAR(255) NOT NULL COMMENT '邮箱',
    description VARCHAR(500) DEFAULT '' COMMENT '描述',

    -- 长文本：TEXT（谨慎使用）
    content TEXT COMMENT '内容',

    -- 数字：根据范围选择类型
    age TINYINT UNSIGNED COMMENT '年龄(0-255)',
    quantity INT UNSIGNED DEFAULT 0 COMMENT '数量',
    price DECIMAL(10, 2) NOT NULL COMMENT '价格（元）',

    -- 布尔：TINYINT(1)
    is_active TINYINT(1) DEFAULT 1 COMMENT '是否激活(0否 1是)',
    is_deleted TINYINT(1) DEFAULT 0 COMMENT '是否删除(0否 1是)',

    -- 枚举：TINYINT + 注释说明
    status TINYINT NOT NULL DEFAULT 0 COMMENT '状态(0待处理 1处理中 2已完成 3已取消)',

    -- 时间：DATETIME
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    deleted_at DATETIME DEFAULT NULL COMMENT '删除时间',

    -- JSON（MySQL 5.7+）
    extra_data JSON COMMENT '扩展数据',

    -- 索引
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at),
    INDEX idx_status_created_at (status, created_at),
    UNIQUE KEY uk_email (email)

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='标准表示例';

-- ============================================
-- 3. 索引规范
-- ============================================

-- 主键索引：每个表必须有主键
-- 唯一索引：业务唯一键
CREATE UNIQUE INDEX uk_users_username ON users(username);
CREATE UNIQUE INDEX uk_users_phone ON users(phone);

-- 普通索引：高频查询字段
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status);

-- 联合索引：遵循最左前缀原则
CREATE INDEX idx_orders_user_status_time ON orders(user_id, status, created_at);
-- 可以匹配：(user_id), (user_id, status), (user_id, status, created_at)
-- 不能匹配：(status), (created_at)

-- 覆盖索引：索引包含所有查询字段
CREATE INDEX idx_users_email_name ON users(email, name);
-- SELECT name FROM users WHERE email = ?  不需要回表

-- 前缀索引：长字符串字段
CREATE INDEX idx_users_address ON users(address(20));

-- ============================================
-- 4. 表结构最佳实践
-- ============================================

-- 订单表示例
CREATE TABLE orders (
    -- 主键
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY COMMENT '订单ID',

    -- 业务字段
    order_no VARCHAR(64) NOT NULL COMMENT '订单号',
    user_id BIGINT UNSIGNED NOT NULL COMMENT '用户ID',
    total_amount DECIMAL(10, 2) NOT NULL COMMENT '订单金额（元）',
    status TINYINT NOT NULL DEFAULT 0 COMMENT '订单状态(0待付款 1已付款 2已发货 3已完成 4已取消)',
    payment_method TINYINT COMMENT '支付方式(1支付宝 2微信 3银行卡)',
    payment_time DATETIME COMMENT '支付时间',
    delivery_time DATETIME COMMENT '发货时间',
    delivery_address VARCHAR(500) NOT NULL COMMENT '收货地址',
    remark VARCHAR(500) DEFAULT '' COMMENT '备注',

    -- 审计字段（所有表必须有）
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    created_by BIGINT UNSIGNED COMMENT '创建人ID',
    updated_by BIGINT UNSIGNED COMMENT '更新人ID',

    -- 软删除（可选）
    is_deleted TINYINT(1) DEFAULT 0 COMMENT '是否删除(0否 1是)',
    deleted_at DATETIME DEFAULT NULL COMMENT '删除时间',

    -- 索引
    UNIQUE KEY uk_order_no (order_no),
    INDEX idx_user_id (user_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at),
    INDEX idx_user_status_time (user_id, status, created_at)

) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='订单表';

-- ============================================
-- 5. 分表策略
-- ============================================

-- 按时间分表（月表）
CREATE TABLE orders_202401 LIKE orders;
CREATE TABLE orders_202402 LIKE orders;
CREATE TABLE orders_202403 LIKE orders;

-- 按范围分表（用户ID分表）
CREATE TABLE users_0 LIKE users;  -- id % 10 = 0
CREATE TABLE users_1 LIKE users;  -- id % 10 = 1
-- ...
CREATE TABLE users_9 LIKE users;  -- id % 10 = 9

-- ============================================
-- 6. 分区策略（MySQL 8.0+）
-- ============================================

-- 按时间范围分区
CREATE TABLE logs (
    id BIGINT PRIMARY KEY,
    message TEXT,
    created_at DATETIME NOT NULL
)
PARTITION BY RANGE (YEAR(created_at)) (
    PARTITION p2023 VALUES LESS THAN (2024),
    PARTITION p2024 VALUES LESS THAN (2025),
    PARTITION p2025 VALUES LESS THAN (2026),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);

-- 按哈希分区
CREATE TABLE user_logs (
    id BIGINT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    action VARCHAR(100)
)
PARTITION BY HASH(user_id)
PARTITIONS 10;
```

### 查询优化规范

```sql
-- ============================================
-- 查询优化规范
-- ============================================

-- 1. 避免SELECT *
-- 错误
SELECT * FROM users WHERE id = 1;

-- 正确：明确列出需要的字段
SELECT id, username, email FROM users WHERE id = 1;


-- 2. 使用LIMIT
-- 错误：可能返回大量数据
SELECT * FROM orders WHERE status = 1;

-- 正确
SELECT id, order_no FROM orders WHERE status = 1 LIMIT 100;


-- 3. 避免在WHERE中使用函数
-- 错误：无法使用索引
SELECT * FROM orders WHERE DATE(created_at) = '2024-01-15';

-- 正确
SELECT * FROM orders
WHERE created_at >= '2024-01-15 00:00:00'
  AND created_at < '2024-01-16 00:00:00';


-- 4. 避免隐式类型转换
-- 错误：phone是VARCHAR，'123'会导致全表扫描
SELECT * FROM users WHERE phone = 123456;

-- 正确
SELECT * FROM users WHERE phone = '123456';


-- 5. 使用UNION ALL代替UNION
-- UNION会去重，性能较差
SELECT id FROM users WHERE status = 1
UNION ALL  -- 不去重，性能更好
SELECT id FROM users WHERE status = 2;


-- 6. JOIN优化
-- 小表驱动大表
-- 错误：大表在前
SELECT o.* FROM orders o
LEFT JOIN users u ON o.user_id = u.id
WHERE u.is_vip = 1;

-- 正确：小表在前（假设VIP用户少）
SELECT o.* FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.is_vip = 1;


-- 7. 分页优化
-- 错误：深分页性能差
SELECT * FROM orders ORDER BY id LIMIT 1000000, 20;

-- 正确：使用上次查询的最大ID
SELECT * FROM orders WHERE id > 1000000 ORDER BY id LIMIT 20;


-- 8. COUNT优化
-- 错误：COUNT(*)全表扫描
SELECT COUNT(*) FROM orders;

-- 正确：使用估算
SELECT table_rows FROM information_schema.tables
WHERE table_schema = 'mydb' AND table_name = 'orders';

-- 或使用缓存存储总数


-- 9. 避免N+1查询
-- 错误：循环查询
-- for order in orders:
--     user = SELECT * FROM users WHERE id = order.user_id

-- 正确：JOIN或IN
SELECT o.*, u.username
FROM orders o
LEFT JOIN users u ON o.user_id = u.id;

-- 或
SELECT * FROM users WHERE id IN (1, 2, 3, ...);


-- 10. 使用EXPLAIN分析
EXPLAIN SELECT * FROM orders WHERE user_id = 1 AND status = 1;
-- 检查：
-- - type: 至少是ref，最好是const
-- - key: 使用了索引
-- - rows: 扫描行数少
-- - Extra: 没有Using filesort, Using temporary
```

## API设计规范

### RESTful API规范

```
API设计规范
============

1. URL设计
----------
- 使用名词，不使用动词
- 使用复数形式
- 使用小写，单词用连字符分隔

正确:
GET    /api/v1/users              # 获取用户列表
GET    /api/v1/users/123          # 获取单个用户
POST   /api/v1/users              # 创建用户
PUT    /api/v1/users/123          # 更新用户(全量)
PATCH  /api/v1/users/123          # 更新用户(部分)
DELETE /api/v1/users/123          # 删除用户
GET    /api/v1/users/123/orders   # 获取用户的订单

错误:
GET    /api/v1/getUsers
POST   /api/v1/createUser
GET    /api/v1/user/123
GET    /api/v1/User/123


2. HTTP方法
-----------
GET    - 查询（幂等、安全）
POST   - 创建（非幂等）
PUT    - 更新全部字段（幂等）
PATCH  - 更新部分字段（幂等）
DELETE - 删除（幂等）


3. HTTP状态码
-------------
200 OK                  - 请求成功
201 Created             - 创建成功
204 No Content          - 删除成功
400 Bad Request         - 请求参数错误
401 Unauthorized        - 未认证
403 Forbidden           - 无权限
404 Not Found           - 资源不存在
409 Conflict            - 资源冲突
422 Unprocessable Entity - 业务逻辑错误
429 Too Many Requests   - 请求过于频繁
500 Internal Server Error - 服务器错误
503 Service Unavailable  - 服务不可用


4. 响应格式
-----------
{
  "code": 0,            // 业务码，0表示成功
  "message": "success", // 消息
  "data": {},           // 数据
  "timestamp": 1705320000000,  // 时间戳
  "trace_id": "abc123"  // 链路追踪ID
}

列表响应:
{
  "code": 0,
  "data": {
    "items": [...],
    "total": 100,
    "page": 1,
    "page_size": 20
  }
}

错误响应:
{
  "code": 40001,
  "message": "用户名已存在",
  "errors": [
    {
      "field": "username",
      "message": "username already exists"
    }
  ]
}


5. 版本控制
-----------
- URL路径: /api/v1/users (推荐)
- 请求头: Accept: application/vnd.company.v1+json
- 查询参数: /api/users?version=1 (不推荐)


6. 过滤、排序、分页
------------------
# 过滤
GET /api/v1/users?status=active&role=admin

# 排序
GET /api/v1/users?sort=created_at&order=desc

# 分页
GET /api/v1/users?page=1&page_size=20

# 组合
GET /api/v1/users?status=active&sort=created_at&order=desc&page=1&page_size=20


7. 字段选择
-----------
GET /api/v1/users?fields=id,name,email


8. 搜索
-------
GET /api/v1/users/search?q=john


9. 批量操作
-----------
POST   /api/v1/users/batch        # 批量创建
PATCH  /api/v1/users/batch        # 批量更新
DELETE /api/v1/users/batch?ids=1,2,3  # 批量删除


10. 认证
--------
Authorization: Bearer <JWT_TOKEN>
```

### API实现示例

```go
// user_api.go
package api

import (
    "net/http"
    "strconv"
    "github.com/gin-gonic/gin"
)

// Response 统一响应结构
type Response struct {
    Code      int         `json:"code"`
    Message   string      `json:"message"`
    Data      interface{} `json:"data,omitempty"`
    Timestamp int64       `json:"timestamp"`
    TraceID   string      `json:"trace_id"`
}

// ListResponse 列表响应
type ListResponse struct {
    Items    interface{} `json:"items"`
    Total    int64       `json:"total"`
    Page     int         `json:"page"`
    PageSize int         `json:"page_size"`
}

// ErrorDetail 错误详情
type ErrorDetail struct {
    Field   string `json:"field"`
    Message string `json:"message"`
}

// UserAPI 用户API
type UserAPI struct {
    userService *UserService
}

// ListUsers 获取用户列表
// @Summary 获取用户列表
// @Tags users
// @Accept json
// @Produce json
// @Param status query string false "状态过滤"
// @Param role query string false "角色过滤"
// @Param page query int false "页码" default(1)
// @Param page_size query int false "每页数量" default(20)
// @Param sort query string false "排序字段" default(created_at)
// @Param order query string false "排序方向" Enums(asc, desc) default(desc)
// @Success 200 {object} Response
// @Router /api/v1/users [get]
func (api *UserAPI) ListUsers(c *gin.Context) {
    // 解析查询参数
    status := c.Query("status")
    role := c.Query("role")
    page, _ := strconv.Atoi(c.DefaultQuery("page", "1"))
    pageSize, _ := strconv.Atoi(c.DefaultQuery("page_size", "20"))
    sort := c.DefaultQuery("sort", "created_at")
    order := c.DefaultQuery("order", "desc")

    // 参数校验
    if page < 1 {
        c.JSON(http.StatusBadRequest, Response{
            Code:    400,
            Message: "page must be greater than 0",
        })
        return
    }

    if pageSize < 1 || pageSize > 100 {
        c.JSON(http.StatusBadRequest, Response{
            Code:    400,
            Message: "page_size must be between 1 and 100",
        })
        return
    }

    // 调用服务
    users, total, err := api.userService.ListUsers(ListUsersRequest{
        Status:   status,
        Role:     role,
        Page:     page,
        PageSize: pageSize,
        Sort:     sort,
        Order:    order,
    })

    if err != nil {
        c.JSON(http.StatusInternalServerError, Response{
            Code:    500,
            Message: "failed to list users",
        })
        return
    }

    // 返回响应
    c.JSON(http.StatusOK, Response{
        Code:    0,
        Message: "success",
        Data: ListResponse{
            Items:    users,
            Total:    total,
            Page:     page,
            PageSize: pageSize,
        },
        Timestamp: time.Now().Unix(),
        TraceID:   c.GetString("trace_id"),
    })
}

// GetUser 获取单个用户
// @Summary 获取用户详情
// @Tags users
// @Param id path int true "用户ID"
// @Success 200 {object} Response
// @Router /api/v1/users/{id} [get]
func (api *UserAPI) GetUser(c *gin.Context) {
    id, err := strconv.ParseInt(c.Param("id"), 10, 64)
    if err != nil {
        c.JSON(http.StatusBadRequest, Response{
            Code:    400,
            Message: "invalid user id",
        })
        return
    }

    user, err := api.userService.GetUser(id)
    if err != nil {
        if errors.Is(err, ErrUserNotFound) {
            c.JSON(http.StatusNotFound, Response{
                Code:    404,
                Message: "user not found",
            })
            return
        }

        c.JSON(http.StatusInternalServerError, Response{
            Code:    500,
            Message: "failed to get user",
        })
        return
    }

    c.JSON(http.StatusOK, Response{
        Code:      0,
        Message:   "success",
        Data:      user,
        Timestamp: time.Now().Unix(),
        TraceID:   c.GetString("trace_id"),
    })
}

// CreateUser 创建用户
// @Summary 创建用户
// @Tags users
// @Accept json
// @Param body body CreateUserRequest true "用户信息"
// @Success 201 {object} Response
// @Router /api/v1/users [post]
func (api *UserAPI) CreateUser(c *gin.Context) {
    var req CreateUserRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, Response{
            Code:    400,
            Message: "invalid request body",
            Errors:  parseValidationErrors(err),
        })
        return
    }

    user, err := api.userService.CreateUser(req)
    if err != nil {
        if errors.Is(err, ErrUserExists) {
            c.JSON(http.StatusConflict, Response{
                Code:    409,
                Message: "user already exists",
            })
            return
        }

        c.JSON(http.StatusInternalServerError, Response{
            Code:    500,
            Message: "failed to create user",
        })
        return
    }

    c.JSON(http.StatusCreated, Response{
        Code:      0,
        Message:   "success",
        Data:      user,
        Timestamp: time.Now().Unix(),
        TraceID:   c.GetString("trace_id"),
    })
}

// UpdateUser 更新用户
// @Summary 更新用户
// @Tags users
// @Param id path int true "用户ID"
// @Param body body UpdateUserRequest true "用户信息"
// @Success 200 {object} Response
// @Router /api/v1/users/{id} [put]
func (api *UserAPI) UpdateUser(c *gin.Context) {
    id, _ := strconv.ParseInt(c.Param("id"), 10, 64)

    var req UpdateUserRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        c.JSON(http.StatusBadRequest, Response{
            Code:    400,
            Message: "invalid request body",
        })
        return
    }

    user, err := api.userService.UpdateUser(id, req)
    if err != nil {
        if errors.Is(err, ErrUserNotFound) {
            c.JSON(http.StatusNotFound, Response{
                Code:    404,
                Message: "user not found",
            })
            return
        }

        c.JSON(http.StatusInternalServerError, Response{
            Code:    500,
            Message: "failed to update user",
        })
        return
    }

    c.JSON(http.StatusOK, Response{
        Code:      0,
        Message:   "success",
        Data:      user,
        Timestamp: time.Now().Unix(),
        TraceID:   c.GetString("trace_id"),
    })
}

// DeleteUser 删除用户
// @Summary 删除用户
// @Tags users
// @Param id path int true "用户ID"
// @Success 204
// @Router /api/v1/users/{id} [delete]
func (api *UserAPI) DeleteUser(c *gin.Context) {
    id, _ := strconv.ParseInt(c.Param("id"), 10, 64)

    err := api.userService.DeleteUser(id)
    if err != nil {
        if errors.Is(err, ErrUserNotFound) {
            c.JSON(http.StatusNotFound, Response{
                Code:    404,
                Message: "user not found",
            })
            return
        }

        c.JSON(http.StatusInternalServerError, Response{
            Code:    500,
            Message: "failed to delete user",
        })
        return
    }

    c.Status(http.StatusNoContent)
}
```

## 安全规范

### 安全检查清单

```markdown
## 应用安全检查清单

### 认证授权
- [ ] 使用强密码策略（8位+，包含大小写字母、数字、特殊字符）
- [ ] 密码使用bcrypt/PBKDF2等安全哈希
- [ ] 实施账号锁定机制（5次失败锁定30分钟）
- [ ] 使用JWT或OAuth2进行API认证
- [ ] Token设置合理过期时间（Access Token 15分钟，Refresh Token 7天）
- [ ] 实施RBAC/ABAC权限控制
- [ ] 最小权限原则
- [ ] 敏感操作二次验证（MFA）

### 输入验证
- [ ] 所有输入进行白名单验证
- [ ] 防止SQL注入（使用参数化查询）
- [ ] 防止XSS攻击（输出转义）
- [ ] 防止CSRF攻击（CSRF Token）
- [ ] 文件上传类型和大小限制
- [ ] 文件名安全检查
- [ ] JSON/XML输入大小限制

### 数据保护
- [ ] 敏感数据加密存储（AES-256）
- [ ] 传输使用HTTPS (TLS 1.2+)
- [ ] 密钥安全管理（Vault/KMS）
- [ ] 敏感日志脱敏
- [ ] PII数据匿名化
- [ ] 数据库连接加密
- [ ] 定期备份加密

### API安全
- [ ] API限流（令牌桶/滑动窗口）
- [ ] IP白名单/黑名单
- [ ] API Key轮换
- [ ] 请求签名验证
- [ ] 防重放攻击（nonce+timestamp）
- [ ] 响应不泄露敏感信息
- [ ] CORS正确配置

### 依赖安全
- [ ] 定期扫描依赖漏洞（npm audit/Snyk）
- [ ] 及时更新依赖版本
- [ ] 使用可信赖的依赖源
- [ ] Lock文件版本锁定
- [ ] 禁用不必要的依赖

### 配置安全
- [ ] 生产环境关闭Debug模式
- [ ] 错误信息不泄露敏感信息
- [ ] 移除默认账号密码
- [ ] 禁用不必要的HTTP方法
- [ ] 安全的CORS配置
- [ ] 设置Security Headers

### 日志审计
- [ ] 记录所有敏感操作
- [ ] 登录/登出日志
- [ ] 权限变更日志
- [ ] 数据修改日志
- [ ] 日志集中存储
- [ ] 日志不可篡改
- [ ] 定期审计日志

### 基础设施
- [ ] 最小化攻击面（关闭不用的端口）
- [ ] 防火墙配置
- [ ] IDS/IPS部署
- [ ] WAF配置
- [ ] DDoS防护
- [ ] 容器镜像扫描
- [ ] 定期安全扫描
```

## 自动化检查

### CI/CD集成

```yaml
# .github/workflows/code-quality.yml
name: Code Quality Check

on:
  pull_request:
    branches: [ main, develop ]
  push:
    branches: [ main, develop ]

jobs:
  checkstyle:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up JDK 11
        uses: actions/setup-java@v3
        with:
          java-version: '11'
          distribution: 'adopt'

      - name: Run Checkstyle
        run: mvn checkstyle:check

      - name: Upload Checkstyle Report
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: checkstyle-report
          path: target/checkstyle-result.xml

  pmd:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up JDK 11
        uses: actions/setup-java@v3
        with:
          java-version: '11'

      - name: Run PMD
        run: mvn pmd:check

  sonarqube:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0  # 全量克隆用于分析

      - name: SonarQube Scan
        uses: sonarsource/sonarqube-scan-action@master
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
          SONAR_HOST_URL: ${{ secrets.SONAR_HOST_URL }}

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run Snyk Security Scan
        uses: snyk/actions/maven@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
```

## 规范治理

### 规范推行策略

```
规范推行路线图
==============

阶段1: 试点（1个月）
- 选择1-2个项目试点
- 收集反馈，调整规范
- 培训核心团队

阶段2: 推广（2个月）
- 全员培训
- 工具配置到所有项目
- CI/CD集成

阶段3: 强制（持续）
- PR必须通过质量检查
- 定期审计
- 持续改进

关键成功因素:
1. 高层支持
2. 工具自动化
3. 持续培训
4. 激励机制
5. 定期回顾
```

## 总结

技术标准和规范是保证代码质量和团队协作的基础。通过建立清晰的标准、自动化检查工具和持续的规范治理，可以显著提升代码质量和开发效率。

**关键要点**:
1. 标准要明确、可执行
2. 工具自动化检查
3. 持续培训和改进
4. 团队共识和执行
5. 定期回顾和更新
