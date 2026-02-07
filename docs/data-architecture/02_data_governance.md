# 数据治理与数据质量管理

## 目录
- [数据治理概述](#数据治理概述)
- [数据质量管理DQM](#数据质量管理dqm)
- [元数据管理](#元数据管理)
- [数据血缘Atlas](#数据血缘atlas)
- [主数据管理](#主数据管理)
- [实战案例](#实战案例)

## 数据治理概述

### 数据治理框架

```
┌─────────────────────────────────────────────────────────┐
│                 数据治理全景图                           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐   │
│  │  数据战略  │───▶│  组织架构  │───▶│  制度流程  │   │
│  │  Strategy  │    │  Organization│   │  Process   │   │
│  └────────────┘    └────────────┘    └────────────┘   │
│         │                 │                  │          │
│         └─────────────────┼──────────────────┘          │
│                           ▼                             │
│              ┌────────────────────────┐                 │
│              │      数据治理平台       │                 │
│              └────────────────────────┘                 │
│                           │                             │
│      ┌────────────────────┼────────────────────┐       │
│      ▼                    ▼                     ▼       │
│  ┌────────┐         ┌──────────┐         ┌─────────┐  │
│  │元数据  │         │ 数据质量  │         │数据血缘 │  │
│  │管理    │         │ 管理      │         │分析     │  │
│  └────────┘         └──────────┘         └─────────┘  │
│      │                    │                     │       │
│      ▼                    ▼                     ▼       │
│  ┌────────┐         ┌──────────┐         ┌─────────┐  │
│  │主数据  │         │ 数据安全  │         │数据资产 │  │
│  │管理    │         │ 管理      │         │目录     │  │
│  └────────┘         └──────────┘         └─────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 数据治理组织

| 角色 | 职责 | 人员 |
|------|------|------|
| 数据治理委员会 | 战略决策、重大问题裁决 | CTO、业务VP |
| 数据治理办公室 | 日常管理、协调推进 | 数据治理经理 |
| 数据所有者 | 业务数据负责人 | 业务部门负责人 |
| 数据管家 | 数据标准、质量监控 | 数据分析师 |
| 数据开发者 | 数据开发、模型设计 | 数据工程师 |
| 数据使用者 | 数据消费、反馈问题 | 业务用户、分析师 |

## 数据质量管理DQM

### 数据质量维度

```python
# data_quality_dimensions.py
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

class QualityDimension(Enum):
    """数据质量维度"""
    COMPLETENESS = "完整性"      # 数据是否完整
    ACCURACY = "准确性"           # 数据是否准确
    CONSISTENCY = "一致性"        # 数据是否一致
    TIMELINESS = "及时性"         # 数据是否及时
    VALIDITY = "有效性"           # 数据是否符合规则
    UNIQUENESS = "唯一性"         # 数据是否重复

@dataclass
class QualityRule:
    """质量规则"""
    rule_id: str
    rule_name: str
    dimension: QualityDimension
    table_name: str
    column_name: Optional[str]
    rule_type: str  # SQL/PYTHON/REGEX
    rule_expression: str
    threshold: float  # 阈值（如：完整性 >= 95%）
    severity: str  # CRITICAL/HIGH/MEDIUM/LOW
    enabled: bool = True

class DataQualityChecker:
    """数据质量检查器"""

    def __init__(self, db_connection):
        self.conn = db_connection
        self.cursor = db_connection.cursor()
        self.rules: List[QualityRule] = []

    def add_rule(self, rule: QualityRule):
        """添加质量规则"""
        self.rules.append(rule)

    def check_completeness(self, table: str, column: str) -> Dict:
        """检查完整性（非空率）"""
        sql = f"""
        SELECT
            COUNT(*) AS total_count,
            COUNT({column}) AS non_null_count,
            COUNT(*) - COUNT({column}) AS null_count,
            COUNT({column}) * 100.0 / COUNT(*) AS completeness_rate
        FROM {table}
        """
        self.cursor.execute(sql)
        result = self.cursor.fetchone()

        return {
            'dimension': QualityDimension.COMPLETENESS.value,
            'table': table,
            'column': column,
            'total_count': result[0],
            'non_null_count': result[1],
            'null_count': result[2],
            'completeness_rate': round(result[3], 2),
            'passed': result[3] >= 95.0  # 阈值95%
        }

    def check_accuracy(self, table: str, column: str, expected_pattern: str) -> Dict:
        """检查准确性（格式正确率）"""
        sql = f"""
        SELECT
            COUNT(*) AS total_count,
            SUM(CASE WHEN {column} REGEXP '{expected_pattern}' THEN 1 ELSE 0 END) AS valid_count,
            SUM(CASE WHEN {column} REGEXP '{expected_pattern}' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS accuracy_rate
        FROM {table}
        WHERE {column} IS NOT NULL
        """
        self.cursor.execute(sql)
        result = self.cursor.fetchone()

        return {
            'dimension': QualityDimension.ACCURACY.value,
            'table': table,
            'column': column,
            'total_count': result[0],
            'valid_count': result[1],
            'accuracy_rate': round(result[2], 2),
            'passed': result[2] >= 99.0  # 阈值99%
        }

    def check_uniqueness(self, table: str, columns: List[str]) -> Dict:
        """检查唯一性（重复率）"""
        columns_str = ', '.join(columns)
        sql = f"""
        SELECT
            COUNT(*) AS total_count,
            COUNT(DISTINCT {columns_str}) AS unique_count,
            COUNT(*) - COUNT(DISTINCT {columns_str}) AS duplicate_count,
            COUNT(DISTINCT {columns_str}) * 100.0 / COUNT(*) AS uniqueness_rate
        FROM {table}
        """
        self.cursor.execute(sql)
        result = self.cursor.fetchone()

        return {
            'dimension': QualityDimension.UNIQUENESS.value,
            'table': table,
            'columns': columns,
            'total_count': result[0],
            'unique_count': result[1],
            'duplicate_count': result[2],
            'uniqueness_rate': round(result[3], 2),
            'passed': result[2] == 0  # 无重复
        }

    def check_consistency(self, table1: str, table2: str, join_key: str) -> Dict:
        """检查一致性（跨表数据一致性）"""
        sql = f"""
        SELECT
            (SELECT COUNT(*) FROM {table1}) AS table1_count,
            (SELECT COUNT(*) FROM {table2}) AS table2_count,
            COUNT(*) AS matched_count
        FROM {table1} t1
        INNER JOIN {table2} t2 ON t1.{join_key} = t2.{join_key}
        """
        self.cursor.execute(sql)
        result = self.cursor.fetchone()

        consistency_rate = result[2] * 100.0 / result[0] if result[0] > 0 else 0

        return {
            'dimension': QualityDimension.CONSISTENCY.value,
            'table1': table1,
            'table2': table2,
            'table1_count': result[0],
            'table2_count': result[1],
            'matched_count': result[2],
            'consistency_rate': round(consistency_rate, 2),
            'passed': consistency_rate >= 95.0
        }

    def check_timeliness(self, table: str, timestamp_column: str, max_delay_hours: int = 24) -> Dict:
        """检查及时性（数据延迟）"""
        sql = f"""
        SELECT
            COUNT(*) AS total_count,
            MAX({timestamp_column}) AS latest_timestamp,
            TIMESTAMPDIFF(HOUR, MAX({timestamp_column}), NOW()) AS delay_hours
        FROM {table}
        """
        self.cursor.execute(sql)
        result = self.cursor.fetchone()

        delay_hours = result[2] if result[2] else 0

        return {
            'dimension': QualityDimension.TIMELINESS.value,
            'table': table,
            'latest_timestamp': result[1],
            'delay_hours': delay_hours,
            'max_delay_hours': max_delay_hours,
            'passed': delay_hours <= max_delay_hours
        }

    def check_validity(self, table: str, column: str, valid_values: List) -> Dict:
        """检查有效性（值域检查）"""
        valid_values_str = ', '.join([f"'{v}'" for v in valid_values])

        sql = f"""
        SELECT
            COUNT(*) AS total_count,
            SUM(CASE WHEN {column} IN ({valid_values_str}) THEN 1 ELSE 0 END) AS valid_count,
            SUM(CASE WHEN {column} IN ({valid_values_str}) THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS validity_rate
        FROM {table}
        WHERE {column} IS NOT NULL
        """
        self.cursor.execute(sql)
        result = self.cursor.fetchone()

        return {
            'dimension': QualityDimension.VALIDITY.value,
            'table': table,
            'column': column,
            'valid_values': valid_values,
            'total_count': result[0],
            'valid_count': result[1],
            'validity_rate': round(result[2], 2),
            'passed': result[2] == 100.0
        }

    def run_all_checks(self) -> List[Dict]:
        """运行所有质量检查"""
        results = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            try:
                if rule.dimension == QualityDimension.COMPLETENESS:
                    result = self.check_completeness(rule.table_name, rule.column_name)
                elif rule.dimension == QualityDimension.ACCURACY:
                    # 从rule_expression中提取正则表达式
                    result = self.check_accuracy(rule.table_name, rule.column_name, rule.rule_expression)
                elif rule.dimension == QualityDimension.UNIQUENESS:
                    columns = rule.rule_expression.split(',')
                    result = self.check_uniqueness(rule.table_name, columns)
                # ... 其他维度

                result['rule_id'] = rule.rule_id
                result['rule_name'] = rule.rule_name
                result['severity'] = rule.severity
                result['check_time'] = datetime.now()

                results.append(result)

            except Exception as e:
                results.append({
                    'rule_id': rule.rule_id,
                    'rule_name': rule.rule_name,
                    'dimension': rule.dimension.value,
                    'error': str(e),
                    'passed': False
                })

        return results

    def generate_report(self, results: List[Dict]) -> str:
        """生成质量报告"""
        total = len(results)
        passed = len([r for r in results if r.get('passed')])
        failed = total - passed

        report = f"""
数据质量检查报告
================
检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
检查规则数: {total}
通过: {passed} ({passed/total*100:.1f}%)
失败: {failed} ({failed/total*100:.1f}%)

失败详情:
"""

        for result in results:
            if not result.get('passed'):
                report += f"\n[{result.get('severity')}] {result.get('rule_name')}\n"
                report += f"  维度: {result.get('dimension')}\n"
                report += f"  表: {result.get('table')}\n"

                if 'completeness_rate' in result:
                    report += f"  完整率: {result['completeness_rate']}% (要求 >= 95%)\n"
                elif 'accuracy_rate' in result:
                    report += f"  准确率: {result['accuracy_rate']}% (要求 >= 99%)\n"
                elif 'duplicate_count' in result:
                    report += f"  重复数: {result['duplicate_count']} (要求 0)\n"
                elif 'delay_hours' in result:
                    report += f"  延迟: {result['delay_hours']}小时 (要求 <= {result['max_delay_hours']})\n"

                report += "\n"

        return report

# 使用示例
def quality_check_example():
    import pymysql

    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='password',
        database='ecommerce'
    )

    checker = DataQualityChecker(conn)

    # 添加规则
    checker.add_rule(QualityRule(
        rule_id='R001',
        rule_name='用户表邮箱完整性检查',
        dimension=QualityDimension.COMPLETENESS,
        table_name='users',
        column_name='email',
        rule_type='SQL',
        rule_expression='',
        threshold=95.0,
        severity='HIGH'
    ))

    checker.add_rule(QualityRule(
        rule_id='R002',
        rule_name='用户表手机号格式检查',
        dimension=QualityDimension.ACCURACY,
        table_name='users',
        column_name='phone',
        rule_type='REGEX',
        rule_expression='^1[3-9][0-9]{9}$',
        threshold=99.0,
        severity='HIGH'
    ))

    checker.add_rule(QualityRule(
        rule_id='R003',
        rule_name='订单表订单号唯一性检查',
        dimension=QualityDimension.UNIQUENESS,
        table_name='orders',
        column_name=None,
        rule_type='SQL',
        rule_expression='order_no',
        threshold=100.0,
        severity='CRITICAL'
    ))

    # 运行检查
    results = checker.run_all_checks()

    # 生成报告
    report = checker.generate_report(results)
    print(report)

    conn.close()

if __name__ == "__main__":
    quality_check_example()
```

### 数据质量规则库

```sql
-- ============================================
-- 数据质量规则表
-- ============================================

CREATE TABLE dq_rules (
    rule_id VARCHAR(50) PRIMARY KEY,
    rule_name VARCHAR(200) NOT NULL,
    dimension VARCHAR(50) NOT NULL COMMENT '完整性/准确性/一致性/及时性/有效性/唯一性',
    category VARCHAR(50) COMMENT '规则分类',

    -- 检查对象
    table_name VARCHAR(100) NOT NULL,
    column_name VARCHAR(100),

    -- 规则定义
    rule_type VARCHAR(20) NOT NULL COMMENT 'SQL/PYTHON/REGEX',
    rule_expression TEXT NOT NULL COMMENT '规则表达式',
    threshold DECIMAL(5,2) COMMENT '阈值（百分比）',

    -- 严重程度
    severity VARCHAR(20) DEFAULT 'MEDIUM' COMMENT 'CRITICAL/HIGH/MEDIUM/LOW',

    -- 执行配置
    enabled TINYINT DEFAULT 1,
    schedule VARCHAR(50) COMMENT '执行频率: DAILY/HOURLY/REALTIME',

    -- 通知配置
    alert_enabled TINYINT DEFAULT 0,
    alert_emails VARCHAR(500),

    -- 审计
    created_by VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_table (table_name),
    INDEX idx_dimension (dimension),
    INDEX idx_enabled (enabled)
) COMMENT='数据质量规则表';

-- 质量检查结果表
CREATE TABLE dq_check_results (
    result_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    rule_id VARCHAR(50) NOT NULL,

    check_time DATETIME NOT NULL,
    check_date DATE NOT NULL,

    -- 检查结果
    total_count BIGINT,
    valid_count BIGINT,
    invalid_count BIGINT,
    quality_score DECIMAL(5,2) COMMENT '质量分数',

    passed TINYINT NOT NULL COMMENT '是否通过',

    -- 详情
    result_detail JSON COMMENT '详细结果',
    error_message TEXT,

    -- 执行信息
    execution_time_ms INT COMMENT '执行耗时(毫秒)',

    INDEX idx_rule_id (rule_id),
    INDEX idx_check_date (check_date),
    INDEX idx_passed (passed)
) COMMENT='数据质量检查结果表';

-- 质量问题表
CREATE TABLE dq_issues (
    issue_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    result_id BIGINT NOT NULL,
    rule_id VARCHAR(50) NOT NULL,

    -- 问题描述
    issue_type VARCHAR(50) NOT NULL,
    issue_description TEXT NOT NULL,

    -- 问题数据
    table_name VARCHAR(100),
    record_id VARCHAR(100) COMMENT '问题记录ID',
    column_name VARCHAR(100),
    current_value TEXT,
    expected_value TEXT,

    -- 状态
    status VARCHAR(20) DEFAULT 'OPEN' COMMENT 'OPEN/IN_PROGRESS/RESOLVED/CLOSED',
    assigned_to VARCHAR(50),
    resolved_at DATETIME,
    resolution_note TEXT,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_result_id (result_id),
    INDEX idx_status (status),
    INDEX idx_table (table_name)
) COMMENT='数据质量问题表';
```

## 元数据管理

### 元数据分类

```
元数据分类
==========

1. 技术元数据 (Technical Metadata)
   ├─ 数据库元数据: 表、列、索引、约束
   ├─ ETL元数据: 数据流、转换规则
   ├─ 数据血缘: 上下游依赖关系
   └─ 运行元数据: 执行日志、性能指标

2. 业务元数据 (Business Metadata)
   ├─ 业务术语: 业务概念、定义
   ├─ 数据字典: 字段含义、取值说明
   ├─ 业务规则: 计算逻辑、校验规则
   └─ 数据分类: 敏感级别、安全等级

3. 操作元数据 (Operational Metadata)
   ├─ 访问日志: 谁在何时访问了什么数据
   ├─ 数据质量: 质量检查结果
   ├─ 数据使用: 使用频率、热度
   └─ 变更历史: 模式变更记录
```

### 元数据采集

```python
# metadata_collector.py
import pymysql
from typing import List, Dict
import json

class MetadataCollector:
    """元数据采集器"""

    def __init__(self, db_config: Dict):
        self.conn = pymysql.connect(**db_config)
        self.cursor = self.conn.cursor(pymysql.cursors.DictCursor)

    def collect_table_metadata(self, database: str) -> List[Dict]:
        """采集表元数据"""
        sql = """
        SELECT
            TABLE_SCHEMA AS database_name,
            TABLE_NAME AS table_name,
            TABLE_TYPE AS table_type,
            ENGINE AS storage_engine,
            TABLE_ROWS AS row_count,
            AVG_ROW_LENGTH AS avg_row_length,
            DATA_LENGTH AS data_length,
            INDEX_LENGTH AS index_length,
            CREATE_TIME AS create_time,
            UPDATE_TIME AS update_time,
            TABLE_COLLATION AS collation,
            TABLE_COMMENT AS table_comment
        FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = %s
        ORDER BY TABLE_NAME
        """
        self.cursor.execute(sql, (database,))
        return self.cursor.fetchall()

    def collect_column_metadata(self, database: str, table: str) -> List[Dict]:
        """采集列元数据"""
        sql = """
        SELECT
            COLUMN_NAME AS column_name,
            ORDINAL_POSITION AS position,
            COLUMN_DEFAULT AS default_value,
            IS_NULLABLE AS is_nullable,
            DATA_TYPE AS data_type,
            CHARACTER_MAXIMUM_LENGTH AS max_length,
            NUMERIC_PRECISION AS numeric_precision,
            NUMERIC_SCALE AS numeric_scale,
            COLUMN_TYPE AS column_type,
            COLUMN_KEY AS column_key,
            EXTRA AS extra,
            COLUMN_COMMENT AS column_comment
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        ORDER BY ORDINAL_POSITION
        """
        self.cursor.execute(sql, (database, table))
        return self.cursor.fetchall()

    def collect_index_metadata(self, database: str, table: str) -> List[Dict]:
        """采集索引元数据"""
        sql = """
        SELECT
            INDEX_NAME AS index_name,
            NON_UNIQUE AS non_unique,
            COLUMN_NAME AS column_name,
            SEQ_IN_INDEX AS seq_in_index,
            COLLATION AS collation,
            CARDINALITY AS cardinality,
            INDEX_TYPE AS index_type,
            COMMENT AS comment
        FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        ORDER BY INDEX_NAME, SEQ_IN_INDEX
        """
        self.cursor.execute(sql, (database, table))
        return self.cursor.fetchall()

    def collect_foreign_key_metadata(self, database: str, table: str) -> List[Dict]:
        """采集外键元数据"""
        sql = """
        SELECT
            CONSTRAINT_NAME AS fk_name,
            COLUMN_NAME AS column_name,
            REFERENCED_TABLE_SCHEMA AS ref_database,
            REFERENCED_TABLE_NAME AS ref_table,
            REFERENCED_COLUMN_NAME AS ref_column
        FROM information_schema.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = %s
          AND TABLE_NAME = %s
          AND REFERENCED_TABLE_NAME IS NOT NULL
        ORDER BY ORDINAL_POSITION
        """
        self.cursor.execute(sql, (database, table))
        return self.cursor.fetchall()

    def collect_full_metadata(self, database: str) -> Dict:
        """采集完整元数据"""
        metadata = {
            'database': database,
            'tables': []
        }

        # 获取所有表
        tables = self.collect_table_metadata(database)

        for table in tables:
            table_name = table['table_name']

            table_meta = {
                **table,
                'columns': self.collect_column_metadata(database, table_name),
                'indexes': self.collect_index_metadata(database, table_name),
                'foreign_keys': self.collect_foreign_key_metadata(database, table_name)
            }

            metadata['tables'].append(table_meta)

        return metadata

    def export_to_json(self, metadata: Dict, output_file: str):
        """导出为JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    def close(self):
        self.cursor.close()
        self.conn.close()

# 使用示例
if __name__ == "__main__":
    collector = MetadataCollector({
        'host': 'localhost',
        'user': 'root',
        'password': 'password',
        'database': 'information_schema'
    })

    # 采集元数据
    metadata = collector.collect_full_metadata('ecommerce')

    # 导出JSON
    collector.export_to_json(metadata, 'metadata_ecommerce.json')

    # 打印摘要
    print(f"数据库: {metadata['database']}")
    print(f"表数量: {len(metadata['tables'])}")
    for table in metadata['tables']:
        print(f"  - {table['table_name']}: {table['row_count']} 行, {len(table['columns'])} 列")

    collector.close()
```

## 数据血缘Atlas

### 血缘关系模型

```
数据血缘示例
============

┌─────────────┐
│ source_db   │
│ orders      │
└──────┬──────┘
       │
       │ ETL Job: extract_orders
       ▼
┌─────────────┐
│ ods_orders  │
└──────┬──────┘
       │
       │ ETL Job: clean_orders
       ▼
┌─────────────┐
│ dwd_orders  │
└──────┬──────┘
       │
       │ ETL Job: aggregate_daily
       ▼
┌─────────────┐      ┌──────────────┐
│dws_daily    │─────▶│ Dashboard    │
│_orders      │      │ 日销售报表    │
└─────────────┘      └──────────────┘

血缘类型:
- 表级血缘: 表与表之间的依赖关系
- 字段级血缘: 字段与字段的转换关系
- 任务级血缘: ETL任务的依赖关系
```

### Apache Atlas集成

```python
# atlas_lineage.py
from typing import List, Dict
import requests
import json

class AtlasClient:
    """Apache Atlas客户端"""

    def __init__(self, atlas_url: str, username: str, password: str):
        self.atlas_url = atlas_url.rstrip('/')
        self.auth = (username, password)
        self.headers = {'Content-Type': 'application/json'}

    def create_table_entity(self, database: str, table: str, columns: List[Dict]) -> str:
        """创建表实体"""
        entity = {
            "entity": {
                "typeName": "hive_table",
                "attributes": {
                    "qualifiedName": f"{database}.{table}@cluster",
                    "name": table,
                    "db": {
                        "typeName": "hive_db",
                        "uniqueAttributes": {
                            "qualifiedName": f"{database}@cluster"
                        }
                    },
                    "columns": [
                        {
                            "typeName": "hive_column",
                            "attributes": {
                                "qualifiedName": f"{database}.{table}.{col['name']}@cluster",
                                "name": col['name'],
                                "type": col['type'],
                                "comment": col.get('comment', '')
                            }
                        }
                        for col in columns
                    ]
                }
            }
        }

        response = requests.post(
            f"{self.atlas_url}/api/atlas/v2/entity",
            json=entity,
            auth=self.auth,
            headers=self.headers
        )

        if response.status_code == 200:
            return response.json()['guidAssignments'][0]
        else:
            raise Exception(f"Failed to create entity: {response.text}")

    def create_process_entity(self, name: str, inputs: List[str], outputs: List[str], script: str) -> str:
        """创建ETL流程实体（表示数据血缘）"""
        entity = {
            "entity": {
                "typeName": "Process",
                "attributes": {
                    "qualifiedName": f"{name}@cluster",
                    "name": name,
                    "inputs": [
                        {"guid": guid} for guid in inputs
                    ],
                    "outputs": [
                        {"guid": guid} for guid in outputs
                    ],
                    "processScript": script
                }
            }
        }

        response = requests.post(
            f"{self.atlas_url}/api/atlas/v2/entity",
            json=entity,
            auth=self.auth,
            headers=self.headers
        )

        if response.status_code == 200:
            return response.json()['guidAssignments'][0]
        else:
            raise Exception(f"Failed to create process: {response.text}")

    def get_lineage(self, guid: str, direction: str = "BOTH", depth: int = 3) -> Dict:
        """获取血缘关系"""
        response = requests.get(
            f"{self.atlas_url}/api/atlas/v2/lineage/{guid}",
            params={'direction': direction, 'depth': depth},
            auth=self.auth
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get lineage: {response.text}")

    def search_entities(self, query: str, type_name: str = None) -> List[Dict]:
        """搜索实体"""
        params = {'query': query}
        if type_name:
            params['typeName'] = type_name

        response = requests.get(
            f"{self.atlas_url}/api/atlas/v2/search/basic",
            params=params,
            auth=self.auth
        )

        if response.status_code == 200:
            return response.json().get('entities', [])
        else:
            raise Exception(f"Failed to search: {response.text}")

# 使用示例
def atlas_lineage_example():
    client = AtlasClient(
        atlas_url='http://localhost:21000',
        username='admin',
        password='admin'
    )

    # 创建源表
    source_guid = client.create_table_entity(
        database='source_db',
        table='orders',
        columns=[
            {'name': 'id', 'type': 'bigint'},
            {'name': 'user_id', 'type': 'bigint'},
            {'name': 'amount', 'type': 'decimal(10,2)'},
            {'name': 'created_at', 'type': 'datetime'}
        ]
    )

    # 创建ODS表
    ods_guid = client.create_table_entity(
        database='ods',
        table='ods_orders',
        columns=[
            {'name': 'id', 'type': 'bigint'},
            {'name': 'user_id', 'type': 'bigint'},
            {'name': 'amount', 'type': 'decimal(10,2)'},
            {'name': 'created_at', 'type': 'datetime'},
            {'name': 'etl_date', 'type': 'date'}
        ]
    )

    # 创建DWD表
    dwd_guid = client.create_table_entity(
        database='dwd',
        table='dwd_orders',
        columns=[
            {'name': 'order_id', 'type': 'bigint'},
            {'name': 'user_id', 'type': 'bigint'},
            {'name': 'order_amount', 'type': 'decimal(10,2)'},
            {'name': 'order_date_key', 'type': 'int'}
        ]
    )

    # 创建ETL流程1: source -> ODS
    process1_guid = client.create_process_entity(
        name='extract_orders',
        inputs=[source_guid],
        outputs=[ods_guid],
        script='SELECT * FROM source_db.orders'
    )

    # 创建ETL流程2: ODS -> DWD
    process2_guid = client.create_process_entity(
        name='clean_orders',
        inputs=[ods_guid],
        outputs=[dwd_guid],
        script='''
        SELECT
            id AS order_id,
            user_id,
            amount AS order_amount,
            DATE_FORMAT(created_at, '%Y%m%d') AS order_date_key
        FROM ods.ods_orders
        WHERE amount > 0
        '''
    )

    # 获取DWD表的血缘
    lineage = client.get_lineage(dwd_guid, direction='INPUT', depth=5)
    print(json.dumps(lineage, indent=2))

if __name__ == "__main__":
    atlas_lineage_example()
```

### 自建血缘系统

```sql
-- ============================================
-- 数据血缘表设计
-- ============================================

-- 数据集表（表、视图、文件等）
CREATE TABLE lineage_dataset (
    dataset_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    dataset_type VARCHAR(50) NOT NULL COMMENT 'TABLE/VIEW/FILE/API',
    dataset_name VARCHAR(200) NOT NULL COMMENT '数据集名称',
    database_name VARCHAR(100),
    schema_name VARCHAR(100),
    qualified_name VARCHAR(500) NOT NULL COMMENT '完全限定名',

    -- 元数据
    columns JSON COMMENT '列信息',
    row_count BIGINT,
    size_bytes BIGINT,

    -- 分类标签
    tags JSON COMMENT '标签',
    business_owner VARCHAR(100) COMMENT '业务负责人',
    technical_owner VARCHAR(100) COMMENT '技术负责人',

    -- 审计
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY uk_qualified_name (qualified_name),
    INDEX idx_dataset_type (dataset_type),
    INDEX idx_database (database_name)
) COMMENT='数据集表';

-- 数据转换表（ETL任务、SQL查询等）
CREATE TABLE lineage_process (
    process_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    process_type VARCHAR(50) NOT NULL COMMENT 'ETL/SQL/SCRIPT',
    process_name VARCHAR(200) NOT NULL,

    -- 转换逻辑
    process_script TEXT COMMENT '处理脚本',
    description TEXT COMMENT '描述',

    -- 调度信息
    schedule_type VARCHAR(50) COMMENT 'DAILY/HOURLY/REALTIME',
    last_run_time DATETIME,
    next_run_time DATETIME,

    -- 状态
    status VARCHAR(20) DEFAULT 'ACTIVE' COMMENT 'ACTIVE/INACTIVE/DEPRECATED',

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_process_type (process_type),
    INDEX idx_status (status)
) COMMENT='数据转换表';

-- 血缘关系表
CREATE TABLE lineage_relation (
    relation_id BIGINT AUTO_INCREMENT PRIMARY KEY,

    -- 源和目标
    source_dataset_id BIGINT NOT NULL COMMENT '源数据集ID',
    target_dataset_id BIGINT NOT NULL COMMENT '目标数据集ID',
    process_id BIGINT COMMENT '转换流程ID',

    -- 关系类型
    relation_type VARCHAR(50) NOT NULL COMMENT 'DERIVE/COPY/TRANSFORM',

    -- 字段级血缘
    column_mapping JSON COMMENT '字段映射关系',

    -- 依赖强度
    dependency_level VARCHAR(20) COMMENT 'STRONG/WEAK',

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_source (source_dataset_id),
    INDEX idx_target (target_dataset_id),
    INDEX idx_process (process_id),
    FOREIGN KEY (source_dataset_id) REFERENCES lineage_dataset(dataset_id),
    FOREIGN KEY (target_dataset_id) REFERENCES lineage_dataset(dataset_id),
    FOREIGN KEY (process_id) REFERENCES lineage_process(process_id)
) COMMENT='血缘关系表';

-- 血缘查询函数
DELIMITER $$

CREATE FUNCTION get_upstream_datasets(
    p_dataset_id BIGINT,
    p_depth INT
)
RETURNS JSON
BEGIN
    DECLARE result JSON;

    -- 递归查询上游数据集
    WITH RECURSIVE upstream AS (
        SELECT
            source_dataset_id AS dataset_id,
            1 AS depth
        FROM lineage_relation
        WHERE target_dataset_id = p_dataset_id

        UNION ALL

        SELECT
            lr.source_dataset_id,
            u.depth + 1
        FROM lineage_relation lr
        INNER JOIN upstream u ON lr.target_dataset_id = u.dataset_id
        WHERE u.depth < p_depth
    )
    SELECT JSON_ARRAYAGG(
        JSON_OBJECT(
            'dataset_id', d.dataset_id,
            'dataset_name', d.dataset_name,
            'depth', u.depth
        )
    ) INTO result
    FROM upstream u
    JOIN lineage_dataset d ON u.dataset_id = d.dataset_id;

    RETURN result;
END$$

DELIMITER ;
```

## 主数据管理

### 主数据概念

```
主数据管理 (Master Data Management, MDM)
==========================================

主数据: 企业核心业务实体的权威数据源

常见主数据:
1. 客户主数据 (Customer)
   - 个人客户、企业客户
   - 统一客户视图

2. 产品主数据 (Product)
   - 产品信息、SKU
   - 产品层级结构

3. 员工主数据 (Employee)
   - 员工基本信息
   - 组织架构

4. 供应商主数据 (Supplier)
5. 物料主数据 (Material)
6. 地址主数据 (Location)
```

### 主数据实现

```sql
-- ============================================
-- 客户主数据表
-- ============================================

-- 客户主表（黄金记录）
CREATE TABLE mdm_customer (
    customer_mdm_id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '主数据ID',
    customer_global_id VARCHAR(50) UNIQUE NOT NULL COMMENT '全局唯一ID',

    -- 基本信息
    customer_name VARCHAR(200) NOT NULL,
    customer_type VARCHAR(50) COMMENT '个人/企业',
    id_type VARCHAR(50) COMMENT '证件类型',
    id_number VARCHAR(100) COMMENT '证件号码',

    -- 联系方式（主）
    primary_phone VARCHAR(20),
    primary_email VARCHAR(100),
    primary_address VARCHAR(500),

    -- 数据质量
    completeness_score DECIMAL(5,2) COMMENT '完整性评分',
    accuracy_score DECIMAL(5,2) COMMENT '准确性评分',
    data_quality_level VARCHAR(20) COMMENT '数据质量等级: A/B/C',

    -- 数据来源
    source_systems JSON COMMENT '来源系统列表',
    master_source VARCHAR(100) COMMENT '主来源系统',

    -- 状态
    status VARCHAR(20) DEFAULT 'ACTIVE' COMMENT 'ACTIVE/INACTIVE/MERGED',
    merged_to_id BIGINT COMMENT '合并到的主数据ID',

    -- 审计
    created_by VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(50),
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_customer_name (customer_name),
    INDEX idx_phone (primary_phone),
    INDEX idx_email (primary_email),
    INDEX idx_id_number (id_number)
) COMMENT='客户主数据表';

-- 客户来源数据表（保留各系统的原始数据）
CREATE TABLE mdm_customer_source (
    source_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    customer_mdm_id BIGINT NOT NULL COMMENT '关联主数据ID',

    source_system VARCHAR(100) NOT NULL COMMENT '来源系统',
    source_system_id VARCHAR(100) NOT NULL COMMENT '来源系统中的ID',

    -- 来源系统的数据（JSON存储）
    source_data JSON NOT NULL,

    -- 数据质量
    data_quality_score DECIMAL(5,2),
    is_authoritative TINYINT DEFAULT 0 COMMENT '是否权威数据',

    -- 同步信息
    last_sync_time DATETIME,
    sync_status VARCHAR(20),

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY uk_source_system_id (source_system, source_system_id),
    INDEX idx_mdm_id (customer_mdm_id),
    FOREIGN KEY (customer_mdm_id) REFERENCES mdm_customer(customer_mdm_id)
) COMMENT='客户来源数据表';

-- 客户匹配规则表
CREATE TABLE mdm_match_rules (
    rule_id VARCHAR(50) PRIMARY KEY,
    rule_name VARCHAR(200) NOT NULL,
    entity_type VARCHAR(50) NOT NULL COMMENT 'CUSTOMER/PRODUCT/...',

    -- 匹配字段和权重
    match_fields JSON NOT NULL COMMENT '匹配字段配置',
    /*
    示例:
    [
        {"field": "id_number", "weight": 100, "match_type": "EXACT"},
        {"field": "phone", "weight": 80, "match_type": "EXACT"},
        {"field": "email", "weight": 80, "match_type": "EXACT"},
        {"field": "name", "weight": 50, "match_type": "FUZZY", "threshold": 0.8}
    ]
    */

    match_threshold DECIMAL(5,2) DEFAULT 80 COMMENT '匹配阈值',

    enabled TINYINT DEFAULT 1,
    priority INT DEFAULT 0 COMMENT '优先级',

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) COMMENT='主数据匹配规则表';

-- 客户合并历史表
CREATE TABLE mdm_customer_merge_history (
    merge_id BIGINT AUTO_INCREMENT PRIMARY KEY,

    source_customer_id BIGINT NOT NULL COMMENT '被合并的客户ID',
    target_customer_id BIGINT NOT NULL COMMENT '合并到的客户ID',

    merge_reason TEXT COMMENT '合并原因',
    merge_algorithm VARCHAR(100) COMMENT '合并算法',
    confidence_score DECIMAL(5,2) COMMENT '置信度',

    merged_by VARCHAR(50),
    merged_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- 回滚信息
    is_rollback TINYINT DEFAULT 0,
    rollback_at DATETIME,
    rollback_by VARCHAR(50),

    INDEX idx_source (source_customer_id),
    INDEX idx_target (target_customer_id)
) COMMENT='客户合并历史表';
```

### 主数据匹配算法

```python
# mdm_matcher.py
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
import Levenshtein  # pip install python-Levenshtein

class MDMMatcher:
    """主数据匹配器"""

    def __init__(self, match_threshold: float = 80.0):
        self.match_threshold = match_threshold

    def exact_match(self, value1: str, value2: str) -> float:
        """精确匹配"""
        if not value1 or not value2:
            return 0.0
        return 100.0 if value1.strip().lower() == value2.strip().lower() else 0.0

    def fuzzy_match(self, value1: str, value2: str) -> float:
        """模糊匹配（Levenshtein距离）"""
        if not value1 or not value2:
            return 0.0

        # 计算相似度
        similarity = Levenshtein.ratio(value1.lower(), value2.lower())
        return similarity * 100

    def phone_match(self, phone1: str, phone2: str) -> float:
        """电话号码匹配"""
        # 提取数字
        p1 = ''.join(filter(str.isdigit, phone1 or ''))
        p2 = ''.join(filter(str.isdigit, phone2 or ''))

        if not p1 or not p2:
            return 0.0

        # 后11位匹配
        if len(p1) >= 11 and len(p2) >= 11:
            if p1[-11:] == p2[-11:]:
                return 100.0

        return 0.0

    def email_match(self, email1: str, email2: str) -> float:
        """邮箱匹配"""
        return self.exact_match(email1, email2)

    def calculate_match_score(self, record1: Dict, record2: Dict, rules: List[Dict]) -> Tuple[float, Dict]:
        """计算匹配分数"""
        total_weight = 0
        weighted_score = 0
        field_scores = {}

        for rule in rules:
            field = rule['field']
            weight = rule['weight']
            match_type = rule['match_type']

            value1 = record1.get(field)
            value2 = record2.get(field)

            # 根据匹配类型计算分数
            if match_type == 'EXACT':
                score = self.exact_match(str(value1) if value1 else '', str(value2) if value2 else '')
            elif match_type == 'FUZZY':
                score = self.fuzzy_match(str(value1) if value1 else '', str(value2) if value2 else '')
                threshold = rule.get('threshold', 0.8) * 100
                if score < threshold:
                    score = 0
            elif match_type == 'PHONE':
                score = self.phone_match(str(value1) if value1 else '', str(value2) if value2 else '')
            elif match_type == 'EMAIL':
                score = self.email_match(str(value1) if value1 else '', str(value2) if value2 else '')
            else:
                score = 0

            field_scores[field] = score
            weighted_score += score * weight
            total_weight += weight * 100  # 满分100

        final_score = (weighted_score / total_weight * 100) if total_weight > 0 else 0

        return final_score, field_scores

    def find_matches(self, new_record: Dict, existing_records: List[Dict], rules: List[Dict]) -> List[Dict]:
        """查找匹配的记录"""
        matches = []

        for existing in existing_records:
            score, field_scores = self.calculate_match_score(new_record, existing, rules)

            if score >= self.match_threshold:
                matches.append({
                    'record': existing,
                    'score': round(score, 2),
                    'field_scores': field_scores
                })

        # 按分数降序排序
        matches.sort(key=lambda x: x['score'], reverse=True)

        return matches

# 使用示例
def mdm_match_example():
    matcher = MDMMatcher(match_threshold=80.0)

    # 匹配规则
    rules = [
        {'field': 'id_number', 'weight': 100, 'match_type': 'EXACT'},
        {'field': 'phone', 'weight': 80, 'match_type': 'PHONE'},
        {'field': 'email', 'weight': 80, 'match_type': 'EMAIL'},
        {'field': 'name', 'weight': 50, 'match_type': 'FUZZY', 'threshold': 0.8}
    ]

    # 新记录
    new_record = {
        'name': '张三',
        'id_number': '110101199001011234',
        'phone': '13800138000',
        'email': 'zhangsan@example.com'
    }

    # 现有记录
    existing_records = [
        {
            'id': 1,
            'name': '张三',
            'id_number': '110101199001011234',
            'phone': '13800138000',
            'email': 'zhangsan@example.com'
        },
        {
            'id': 2,
            'name': '张三',
            'id_number': None,
            'phone': '+86-138-0013-8000',  # 格式不同但是同一个号码
            'email': 'zhangsan@example.com'
        },
        {
            'id': 3,
            'name': '张珊',  # 名字相似
            'id_number': None,
            'phone': '13900139000',  # 不同号码
            'email': 'zhangshan@example.com'
        }
    ]

    # 查找匹配
    matches = matcher.find_matches(new_record, existing_records, rules)

    print("匹配结果:")
    for match in matches:
        print(f"\n记录ID: {match['record']['id']}, 匹配分数: {match['score']}")
        print(f"  字段得分: {match['field_scores']}")

if __name__ == "__main__":
    mdm_match_example()
```

## 实战案例

### 案例：构建数据治理平台

```python
# data_governance_platform.py
"""
数据治理平台核心功能示例
整合元数据、质量、血缘等功能
"""

from typing import Dict, List
from datetime import datetime
import json

class DataGovernancePlatform:
    """数据治理平台"""

    def __init__(self, db_connection):
        self.conn = db_connection
        self.metadata_collector = MetadataCollector(db_connection)
        self.quality_checker = DataQualityChecker(db_connection)
        self.mdm_matcher = MDMMatcher()

    def register_dataset(self, dataset_info: Dict) -> int:
        """注册数据集"""
        # 采集元数据
        metadata = self.metadata_collector.collect_table_metadata(
            dataset_info['database'],
            dataset_info['table']
        )

        # 注册到数据目录
        cursor = self.conn.cursor()
        sql = """
        INSERT INTO data_catalog (
            dataset_name, database_name, table_name,
            dataset_type, description, owner,
            metadata, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        """
        cursor.execute(sql, (
            dataset_info['name'],
            dataset_info['database'],
            dataset_info['table'],
            dataset_info.get('type', 'TABLE'),
            dataset_info.get('description', ''),
            dataset_info.get('owner', ''),
            json.dumps(metadata)
        ))

        dataset_id = cursor.lastrowid
        self.conn.commit()

        return dataset_id

    def run_quality_check(self, dataset_id: int) -> Dict:
        """运行数据质量检查"""
        # 获取数据集信息
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM data_catalog WHERE dataset_id = %s", (dataset_id,))
        dataset = cursor.fetchone()

        # 运行质量检查
        results = self.quality_checker.run_all_checks()

        # 保存检查结果
        for result in results:
            sql = """
            INSERT INTO dq_check_results (
                dataset_id, rule_id, check_time, quality_score, passed, result_detail
            ) VALUES (%s, %s, NOW(), %s, %s, %s)
            """
            cursor.execute(sql, (
                dataset_id,
                result.get('rule_id'),
                result.get('quality_score'),
                result.get('passed'),
                json.dumps(result)
            ))

        self.conn.commit()

        return {
            'dataset_id': dataset_id,
            'check_time': datetime.now(),
            'total_rules': len(results),
            'passed': len([r for r in results if r.get('passed')]),
            'results': results
        }

    def track_lineage(self, source_id: int, target_id: int, process_info: Dict):
        """跟踪数据血缘"""
        cursor = self.conn.cursor()

        # 记录血缘关系
        sql = """
        INSERT INTO lineage_relation (
            source_dataset_id, target_dataset_id,
            process_type, process_script, column_mapping,
            created_at
        ) VALUES (%s, %s, %s, %s, %s, NOW())
        """
        cursor.execute(sql, (
            source_id,
            target_id,
            process_info.get('type', 'ETL'),
            process_info.get('script', ''),
            json.dumps(process_info.get('column_mapping', {}))
        ))

        self.conn.commit()

    def get_data_health_dashboard(self) -> Dict:
        """获取数据健康仪表盘"""
        cursor = self.conn.cursor()

        # 数据集统计
        cursor.execute("""
            SELECT
                COUNT(*) AS total_datasets,
                COUNT(DISTINCT database_name) AS total_databases,
                SUM(CASE WHEN status = 'ACTIVE' THEN 1 ELSE 0 END) AS active_datasets
            FROM data_catalog
        """)
        dataset_stats = cursor.fetchone()

        # 质量统计
        cursor.execute("""
            SELECT
                COUNT(*) AS total_checks,
                AVG(quality_score) AS avg_quality_score,
                SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pass_rate
            FROM dq_check_results
            WHERE check_date = CURDATE()
        """)
        quality_stats = cursor.fetchone()

        # 问题统计
        cursor.execute("""
            SELECT
                COUNT(*) AS total_issues,
                SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END) AS open_issues,
                SUM(CASE WHEN severity = 'CRITICAL' THEN 1 ELSE 0 END) AS critical_issues
            FROM dq_issues
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        """)
        issue_stats = cursor.fetchone()

        return {
            'dataset_statistics': {
                'total_datasets': dataset_stats[0],
                'total_databases': dataset_stats[1],
                'active_datasets': dataset_stats[2]
            },
            'quality_statistics': {
                'total_checks_today': quality_stats[0],
                'avg_quality_score': round(quality_stats[1], 2) if quality_stats[1] else 0,
                'pass_rate': round(quality_stats[2], 2) if quality_stats[2] else 0
            },
            'issue_statistics': {
                'total_issues_week': issue_stats[0],
                'open_issues': issue_stats[1],
                'critical_issues': issue_stats[2]
            }
        }

# 使用示例
if __name__ == "__main__":
    platform = DataGovernancePlatform(db_connection)

    # 注册数据集
    dataset_id = platform.register_dataset({
        'name': '用户订单表',
        'database': 'ecommerce',
        'table': 'orders',
        'type': 'TABLE',
        'description': '电商订单明细数据',
        'owner': '张三'
    })

    # 运行质量检查
    quality_report = platform.run_quality_check(dataset_id)
    print(json.dumps(quality_report, indent=2, default=str))

    # 获取健康仪表盘
    dashboard = platform.get_data_health_dashboard()
    print(json.dumps(dashboard, indent=2))
```

## 总结

数据治理是确保数据资产价值的系统性工程，涉及组织、流程、技术多个层面。

**关键要点**:
1. 建立数据治理组织和流程
2. 实施数据质量管理（6个维度）
3. 完善元数据管理
4. 构建数据血缘体系
5. 推行主数据管理
6. 持续监控和改进

**进阶阅读**:
- 《DAMA数据管理知识体系指南》
- Apache Atlas官方文档
- Great Expectations数据质量框架
- Amundsen数据发现平台
