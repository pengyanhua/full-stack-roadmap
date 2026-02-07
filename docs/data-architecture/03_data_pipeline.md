# 数据管道架构设计

## 目录
- [概述](#概述)
- [Airflow完整示例](#airflow完整示例)
- [CDC实时数据捕获](#cdc实时数据捕获)
- [Flink流处理](#flink流处理)
- [数据质量检查](#数据质量检查)
- [监控与告警](#监控与告警)
- [实战案例](#实战案例)

## 概述

### 数据管道架构演进

```
传统批处理          混合架构           实时流处理
┌──────────┐       ┌──────────┐      ┌──────────┐
│  定时任务  │       │  Airflow  │      │  Flink   │
│  Cron    │  -->  │  +       │ -->  │  Kafka   │
│  ETL     │       │  CDC     │      │  实时处理 │
└──────────┘       └──────────┘      └──────────┘
  T+1延迟           小时级延迟         秒级延迟
```

### 现代数据管道架构

```
┌─────────────────────────────────────────────────────────────┐
│                      数据源层                                 │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│  MySQL   │PostgreSQL│ MongoDB  │   API    │   日志文件        │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴─────────┬───────┘
     │          │          │          │               │
     v          v          v          v               v
┌─────────────────────────────────────────────────────────────┐
│                    数据采集层                                 │
├─────────┬──────────┬──────────┬──────────┬──────────────────┤
│ Debezium│ Maxwell  │ Sqoop    │ Flume    │  Filebeat        │
│  CDC    │   CDC    │  批量     │  日志     │   日志           │
└────┬────┴────┬─────┴────┬─────┴────┬─────┴────┬─────────────┘
     │         │          │          │          │
     └─────────┴──────────┴──────────┴──────────┘
                          │
                          v
┌─────────────────────────────────────────────────────────────┐
│                    消息队列层                                 │
│              Kafka (3 Brokers, RF=3)                        │
│   ┌──────────┬──────────┬──────────┬──────────┐            │
│   │  Topic1  │  Topic2  │  Topic3  │  Topic4  │            │
│   │ order    │  user    │  product │   log    │            │
│   └────┬─────┴────┬─────┴────┬─────┴────┬─────┘            │
└────────┼──────────┼──────────┼──────────┼──────────────────┘
         │          │          │          │
         v          v          v          v
┌─────────────────────────────────────────────────────────────┐
│                    流处理层                                   │
│         Flink Cluster (JobManager + TaskManager)            │
│  ┌──────────┬──────────┬──────────┬──────────────────┐     │
│  │ 清洗转换  │  聚合计算 │ 关联Join  │  窗口分析         │     │
│  └────┬─────┴────┬─────┴────┬─────┴────┬────────────┘     │
└───────┼──────────┼──────────┼──────────┼───────────────────┘
        │          │          │          │
        v          v          v          v
┌─────────────────────────────────────────────────────────────┐
│                    存储层                                     │
├─────────┬──────────┬──────────┬──────────┬──────────────────┤
│ Hive    │ ClickHouse│  Doris  │  Redis   │  Elasticsearch   │
│ 数据湖   │   OLAP   │  实时仓  │  缓存     │   搜索引擎        │
└─────────┴──────────┴──────────┴──────────┴──────────────────┘
```

## Airflow完整示例

### Airflow架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Airflow Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐         ┌──────────┐        ┌──────────┐    │
│  │   Web    │◄────────┤ Metadata │────────►│ Scheduler│    │
│  │  Server  │         │ Database │        │          │    │
│  └──────────┘         │(Postgres)│        └────┬─────┘    │
│       ▲               └──────────┘             │           │
│       │                                        │           │
│       │                                        v           │
│       │               ┌─────────────────────────────┐      │
│       └───────────────┤      Task Queue          │      │
│                       │      (Redis/RabbitMQ)      │      │
│                       └────────────┬───────────────┘      │
│                                    │                       │
│                                    v                       │
│               ┌────────────────────────────────────┐       │
│               │         Executor Pool              │       │
│               ├────────┬────────┬────────┬────────┤       │
│               │Worker 1│Worker 2│Worker 3│Worker N│       │
│               └────────┴────────┴────────┴────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 完整DAG示例：电商订单ETL

```python
"""
电商订单数据ETL管道
功能：
1. 从MySQL提取订单数据
2. 数据清洗与转换
3. 关联用户和产品信息
4. 写入数据仓库
5. 数据质量检查
6. 告警通知
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.mysql.operators.mysql import MySqlOperator
from airflow.providers.amazon.aws.transfers.mysql_to_s3 import MySQLToS3Operator
from airflow.providers.amazon.aws.transfers.s3_to_redshift import S3ToRedshiftOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import logging
import pandas as pd
import great_expectations as ge

# 默认参数配置
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['data-alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
    'execution_timeout': timedelta(hours=2),
    'sla': timedelta(hours=3),
}

# 创建DAG
dag = DAG(
    'ecommerce_order_etl',
    default_args=default_args,
    description='电商订单数据ETL管道',
    schedule_interval='0 2 * * *',  # 每天凌晨2点运行
    catchup=False,
    max_active_runs=1,
    tags=['etl', 'orders', 'production'],
)

# ============================================================================
# 任务1：检查上游依赖
# ============================================================================
check_upstream = ExternalTaskSensor(
    task_id='check_upstream_dag',
    external_dag_id='user_data_etl',
    external_task_id='load_to_warehouse',
    allowed_states=['success'],
    failed_states=['failed', 'skipped'],
    mode='reschedule',
    timeout=3600,
    poke_interval=300,
    dag=dag,
)

# ============================================================================
# 任务2：数据提取
# ============================================================================
def extract_orders(**context):
    """从MySQL提取订单数据"""
    from airflow.providers.mysql.hooks.mysql import MySqlHook

    execution_date = context['execution_date']
    prev_date = execution_date - timedelta(days=1)

    mysql_hook = MySqlHook(mysql_conn_id='mysql_orders_db')

    # 增量提取SQL
    sql = f"""
    SELECT
        order_id,
        user_id,
        product_id,
        quantity,
        price,
        total_amount,
        order_status,
        payment_method,
        shipping_address,
        created_at,
        updated_at
    FROM orders
    WHERE DATE(created_at) = '{prev_date.strftime('%Y-%m-%d')}'
        AND is_deleted = 0
    """

    # 执行查询
    df = mysql_hook.get_pandas_df(sql)

    # 保存到临时位置
    output_path = f'/tmp/orders_{prev_date.strftime("%Y%m%d")}.parquet'
    df.to_parquet(output_path, compression='snappy', index=False)

    logging.info(f"提取了 {len(df)} 条订单记录")

    # 推送元数据到XCom
    context['task_instance'].xcom_push(
        key='order_count',
        value=len(df)
    )
    context['task_instance'].xcom_push(
        key='output_path',
        value=output_path
    )

    return output_path

extract_orders_task = PythonOperator(
    task_id='extract_orders',
    python_callable=extract_orders,
    provide_context=True,
    dag=dag,
)

# ============================================================================
# 任务3：数据清洗与转换
# ============================================================================
def clean_and_transform(**context):
    """数据清洗与转换"""
    import numpy as np

    # 从XCom获取上游数据
    ti = context['task_instance']
    input_path = ti.xcom_pull(task_ids='extract_orders', key='output_path')

    # 读取数据
    df = pd.read_parquet(input_path)

    logging.info(f"原始数据: {len(df)} 行")

    # 1. 删除重复数据
    df = df.drop_duplicates(subset=['order_id'], keep='last')

    # 2. 处理缺失值
    df['shipping_address'] = df['shipping_address'].fillna('Unknown')
    df['payment_method'] = df['payment_method'].fillna('Unknown')

    # 3. 数据类型转换
    df['order_id'] = df['order_id'].astype('int64')
    df['user_id'] = df['user_id'].astype('int64')
    df['product_id'] = df['product_id'].astype('int64')
    df['quantity'] = df['quantity'].astype('int32')
    df['price'] = df['price'].astype('float64')
    df['total_amount'] = df['total_amount'].astype('float64')

    # 4. 数据验证
    # 验证金额计算
    df['calculated_amount'] = df['quantity'] * df['price']
    df['amount_diff'] = np.abs(df['total_amount'] - df['calculated_amount'])

    # 标记异常数据
    df['is_amount_valid'] = df['amount_diff'] < 0.01
    invalid_count = (~df['is_amount_valid']).sum()

    if invalid_count > 0:
        logging.warning(f"发现 {invalid_count} 条金额异常数据")

    # 5. 添加派生字段
    df['order_date'] = pd.to_datetime(df['created_at']).dt.date
    df['order_hour'] = pd.to_datetime(df['created_at']).dt.hour
    df['order_day_of_week'] = pd.to_datetime(df['created_at']).dt.dayofweek

    # 6. 数据分类
    df['price_category'] = pd.cut(
        df['price'],
        bins=[0, 50, 100, 500, float('inf')],
        labels=['low', 'medium', 'high', 'premium']
    )

    # 保存清洗后的数据
    output_path = input_path.replace('.parquet', '_cleaned.parquet')
    df.to_parquet(output_path, compression='snappy', index=False)

    logging.info(f"清洗后数据: {len(df)} 行")

    # 推送统计信息
    stats = {
        'total_orders': len(df),
        'invalid_amount_count': int(invalid_count),
        'total_revenue': float(df['total_amount'].sum()),
        'avg_order_value': float(df['total_amount'].mean()),
    }

    ti.xcom_push(key='cleaned_path', value=output_path)
    ti.xcom_push(key='stats', value=stats)

    return output_path

clean_transform_task = PythonOperator(
    task_id='clean_and_transform',
    python_callable=clean_and_transform,
    provide_context=True,
    dag=dag,
)

# ============================================================================
# 任务4：关联维度数据
# ============================================================================
def join_dimensions(**context):
    """关联用户和产品维度信息"""
    from airflow.providers.postgres.hooks.postgres import PostgresHook

    ti = context['task_instance']
    input_path = ti.xcom_pull(task_ids='clean_and_transform', key='cleaned_path')

    # 读取订单数据
    orders_df = pd.read_parquet(input_path)

    # 获取维度数据
    pg_hook = PostgresHook(postgres_conn_id='dwh_postgres')

    # 获取用户维度
    users_df = pg_hook.get_pandas_df("""
        SELECT user_id, user_name, user_level, register_date, city
        FROM dim_users
        WHERE is_active = true
    """)

    # 获取产品维度
    products_df = pg_hook.get_pandas_df("""
        SELECT product_id, product_name, category, brand, cost_price
        FROM dim_products
        WHERE is_active = true
    """)

    # 关联数据
    result_df = orders_df.merge(
        users_df,
        on='user_id',
        how='left'
    ).merge(
        products_df,
        on='product_id',
        how='left'
    )

    # 计算利润
    result_df['profit'] = (result_df['price'] - result_df['cost_price']) * result_df['quantity']
    result_df['profit_margin'] = result_df['profit'] / result_df['total_amount'] * 100

    # 保存结果
    output_path = input_path.replace('_cleaned.parquet', '_enriched.parquet')
    result_df.to_parquet(output_path, compression='snappy', index=False)

    logging.info(f"关联后数据: {len(result_df)} 行, {result_df.shape[1]} 列")

    ti.xcom_push(key='enriched_path', value=output_path)

    return output_path

join_dimensions_task = PythonOperator(
    task_id='join_dimensions',
    python_callable=join_dimensions,
    provide_context=True,
    dag=dag,
)

# ============================================================================
# 任务5：数据质量检查
# ============================================================================
def validate_data_quality(**context):
    """使用Great Expectations进行数据质量检查"""
    ti = context['task_instance']
    input_path = ti.xcom_pull(task_ids='join_dimensions', key='enriched_path')

    # 读取数据
    df = pd.read_parquet(input_path)

    # 转换为GE DataFrame
    ge_df = ge.from_pandas(df)

    # 定义数据质量规则
    validation_results = []

    # 1. 检查主键唯一性
    result1 = ge_df.expect_column_values_to_be_unique('order_id')
    validation_results.append(('唯一性检查', result1['success']))

    # 2. 检查必填字段
    for col in ['user_id', 'product_id', 'quantity', 'price']:
        result = ge_df.expect_column_values_to_not_be_null(col)
        validation_results.append((f'{col}非空检查', result['success']))

    # 3. 检查数值范围
    result3 = ge_df.expect_column_values_to_be_between('quantity', min_value=1, max_value=1000)
    validation_results.append(('数量范围检查', result3['success']))

    result4 = ge_df.expect_column_values_to_be_between('price', min_value=0, max_value=1000000)
    validation_results.append(('价格范围检查', result4['success']))

    # 4. 检查枚举值
    valid_statuses = ['pending', 'paid', 'shipped', 'delivered', 'cancelled']
    result5 = ge_df.expect_column_values_to_be_in_set('order_status', valid_statuses)
    validation_results.append(('订单状态检查', result5['success']))

    # 5. 检查数据完整性
    result6 = ge_df.expect_column_values_to_not_be_null('user_name')
    validation_results.append(('用户信息完整性', result6['success']))

    result7 = ge_df.expect_column_values_to_not_be_null('product_name')
    validation_results.append(('产品信息完整性', result7['success']))

    # 统计结果
    total_checks = len(validation_results)
    passed_checks = sum(1 for _, success in validation_results if success)

    logging.info(f"数据质量检查: {passed_checks}/{total_checks} 通过")

    for check_name, success in validation_results:
        status = "✓ 通过" if success else "✗ 失败"
        logging.info(f"  {check_name}: {status}")

    # 如果关键检查失败，抛出异常
    critical_checks = validation_results[:5]  # 前5个是关键检查
    if not all(success for _, success in critical_checks):
        raise ValueError("关键数据质量检查失败")

    ti.xcom_push(key='quality_check_passed', value=passed_checks)
    ti.xcom_push(key='quality_check_total', value=total_checks)

    return passed_checks

validate_quality_task = PythonOperator(
    task_id='validate_data_quality',
    python_callable=validate_data_quality,
    provide_context=True,
    dag=dag,
)

# ============================================================================
# 任务6：加载到数据仓库
# ============================================================================
def load_to_warehouse(**context):
    """加载数据到数据仓库"""
    from airflow.providers.postgres.hooks.postgres import PostgresHook
    import psycopg2.extras as extras

    ti = context['task_instance']
    input_path = ti.xcom_pull(task_ids='join_dimensions', key='enriched_path')
    execution_date = context['execution_date']

    # 读取数据
    df = pd.read_parquet(input_path)

    # 添加ETL元数据
    df['etl_batch_id'] = execution_date.strftime('%Y%m%d%H%M%S')
    df['etl_load_time'] = datetime.now()

    # 获取数据库连接
    pg_hook = PostgresHook(postgres_conn_id='dwh_postgres')
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    try:
        # 1. 创建临时表
        cursor.execute("""
            CREATE TEMP TABLE temp_orders (LIKE fact_orders INCLUDING ALL)
        """)

        # 2. 批量插入临时表
        tuples = [tuple(x) for x in df.to_numpy()]
        cols = ','.join(list(df.columns))

        query = f"INSERT INTO temp_orders({cols}) VALUES %s"
        extras.execute_values(cursor, query, tuples)

        logging.info(f"已插入 {len(df)} 条记录到临时表")

        # 3. Merge到目标表 (UPDATE + INSERT)
        cursor.execute("""
            -- 更新已存在的记录
            UPDATE fact_orders fo
            SET
                order_status = t.order_status,
                updated_at = t.updated_at,
                etl_batch_id = t.etl_batch_id,
                etl_load_time = t.etl_load_time
            FROM temp_orders t
            WHERE fo.order_id = t.order_id;

            -- 插入新记录
            INSERT INTO fact_orders
            SELECT * FROM temp_orders
            WHERE order_id NOT IN (SELECT order_id FROM fact_orders);
        """)

        # 4. 记录加载统计
        cursor.execute("""
            INSERT INTO etl_load_log (
                dag_id, task_id, execution_date,
                table_name, rows_inserted, rows_updated, load_time
            ) VALUES (
                %s, %s, %s, %s,
                (SELECT COUNT(*) FROM temp_orders WHERE order_id NOT IN (SELECT order_id FROM fact_orders)),
                (SELECT COUNT(*) FROM temp_orders WHERE order_id IN (SELECT order_id FROM fact_orders)),
                %s
            )
        """, (
            dag.dag_id,
            'load_to_warehouse',
            execution_date,
            'fact_orders',
            datetime.now()
        ))

        conn.commit()
        logging.info("数据成功加载到数据仓库")

    except Exception as e:
        conn.rollback()
        logging.error(f"加载失败: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

    return len(df)

load_warehouse_task = PythonOperator(
    task_id='load_to_warehouse',
    python_callable=load_to_warehouse,
    provide_context=True,
    dag=dag,
)

# ============================================================================
# 任务7：更新统计表
# ============================================================================
update_stats = MySqlOperator(
    task_id='update_statistics',
    mysql_conn_id='dwh_postgres',
    sql="""
        -- 更新订单统计表
        INSERT INTO order_daily_stats (
            stat_date,
            total_orders,
            total_revenue,
            avg_order_value,
            updated_at
        )
        SELECT
            DATE(created_at) as stat_date,
            COUNT(*) as total_orders,
            SUM(total_amount) as total_revenue,
            AVG(total_amount) as avg_order_value,
            NOW() as updated_at
        FROM fact_orders
        WHERE DATE(created_at) = '{{ ds }}'
        GROUP BY DATE(created_at)
        ON CONFLICT (stat_date) DO UPDATE SET
            total_orders = EXCLUDED.total_orders,
            total_revenue = EXCLUDED.total_revenue,
            avg_order_value = EXCLUDED.avg_order_value,
            updated_at = EXCLUDED.updated_at;
    """,
    dag=dag,
)

# ============================================================================
# 任务8：成功通知
# ============================================================================
def send_success_notification(**context):
    """发送成功通知"""
    ti = context['task_instance']
    stats = ti.xcom_pull(task_ids='clean_and_transform', key='stats')
    quality_passed = ti.xcom_pull(task_ids='validate_data_quality', key='quality_check_passed')
    quality_total = ti.xcom_pull(task_ids='validate_data_quality', key='quality_check_total')

    message = f"""
    ✅ 订单ETL管道执行成功

    执行日期: {context['ds']}
    执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    📊 数据统计:
    - 订单数量: {stats['total_orders']:,}
    - 总收入: ¥{stats['total_revenue']:,.2f}
    - 平均订单金额: ¥{stats['avg_order_value']:,.2f}
    - 异常数据: {stats['invalid_amount_count']}

    ✔️ 质量检查: {quality_passed}/{quality_total} 通过

    🔗 DAG链接: {context['task_instance'].log_url}
    """

    logging.info(message)
    # 这里可以集成Slack/钉钉/企业微信通知

    return message

success_notification = PythonOperator(
    task_id='send_success_notification',
    python_callable=send_success_notification,
    provide_context=True,
    trigger_rule='all_success',
    dag=dag,
)

# ============================================================================
# 任务9：失败告警
# ============================================================================
def send_failure_alert(**context):
    """发送失败告警"""
    ti = context['task_instance']
    exception = context.get('exception')

    message = f"""
    ❌ 订单ETL管道执行失败

    DAG ID: {context['dag'].dag_id}
    Task ID: {ti.task_id}
    执行日期: {context['ds']}
    执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    错误信息:
    {str(exception)}

    🔗 日志链接: {ti.log_url}

    请及时处理！
    """

    logging.error(message)
    # 这里可以集成告警系统

    return message

failure_alert = PythonOperator(
    task_id='send_failure_alert',
    python_callable=send_failure_alert,
    provide_context=True,
    trigger_rule='one_failed',
    dag=dag,
)

# ============================================================================
# 定义任务依赖关系
# ============================================================================
check_upstream >> extract_orders_task >> clean_transform_task
clean_transform_task >> join_dimensions_task >> validate_quality_task
validate_quality_task >> load_warehouse_task >> update_stats
update_stats >> success_notification

# 任何任务失败都触发告警
[extract_orders_task, clean_transform_task, join_dimensions_task,
 validate_quality_task, load_warehouse_task, update_stats] >> failure_alert
```

### 任务组管理

```python
# 使用TaskGroup组织复杂任务
with TaskGroup('data_preparation', tooltip='数据准备阶段') as prep_group:
    extract = PythonOperator(task_id='extract', ...)
    clean = PythonOperator(task_id='clean', ...)
    validate = PythonOperator(task_id='validate', ...)

    extract >> clean >> validate

with TaskGroup('data_loading', tooltip='数据加载阶段') as load_group:
    load_staging = PythonOperator(task_id='load_staging', ...)
    load_prod = PythonOperator(task_id='load_prod', ...)
    update_metadata = PythonOperator(task_id='update_metadata', ...)

    load_staging >> load_prod >> update_metadata

prep_group >> load_group
```

## CDC实时数据捕获

### Debezium架构

```
┌─────────────────────────────────────────────────────────────┐
│                  Debezium CDC Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Source Database (MySQL)               │    │
│  │  ┌──────────┬──────────┬──────────┬──────────┐    │    │
│  │  │ orders   │  users   │ products │   ...    │    │    │
│  │  └──────────┴──────────┴──────────┴──────────┘    │    │
│  │              binlog (Row Format)                   │    │
│  └──────────────────────┬─────────────────────────────┘    │
│                         │ Read binlog events               │
│                         v                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │         Debezium MySQL Connector                   │    │
│  │  ┌──────────────────────────────────────────┐     │    │
│  │  │  - Binlog Reader                         │     │    │
│  │  │  - Event Parser                          │     │    │
│  │  │  - Schema Registry Client                │     │    │
│  │  │  - Offset Storage                        │     │    │
│  │  └──────────────────────────────────────────┘     │    │
│  └──────────────────────┬─────────────────────────────┘    │
│                         │ Publish CDC events               │
│                         v                                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Kafka Cluster                         │    │
│  │  Topic: dbserver1.inventory.orders                 │    │
│  │  ┌───────┬───────┬───────┬───────┬───────┐        │    │
│  │  │ Part0 │ Part1 │ Part2 │ Part3 │ Part4 │        │    │
│  │  └───────┴───────┴───────┴───────┴───────┘        │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Debezium Connector配置

```json
{
  "name": "mysql-orders-connector",
  "config": {
    "connector.class": "io.debezium.connector.mysql.MySqlConnector",
    "tasks.max": "1",

    "database.hostname": "mysql.prod.company.com",
    "database.port": "3306",
    "database.user": "debezium_user",
    "database.password": "${file:/secrets/db-password.txt:password}",
    "database.server.id": "184054",
    "database.server.name": "dbserver1",

    "database.include.list": "inventory",
    "table.include.list": "inventory.orders,inventory.order_items",

    "database.history.kafka.bootstrap.servers": "kafka1:9092,kafka2:9092,kafka3:9092",
    "database.history.kafka.topic": "dbhistory.inventory",

    "include.schema.changes": "true",
    "snapshot.mode": "when_needed",
    "snapshot.locking.mode": "minimal",

    "decimal.handling.mode": "precise",
    "time.precision.mode": "adaptive_time_microseconds",
    "bigint.unsigned.handling.mode": "precise",

    "event.processing.failure.handling.mode": "warn",
    "inconsistent.schema.handling.mode": "warn",

    "key.converter": "org.apache.kafka.connect.json.JsonConverter",
    "key.converter.schemas.enable": "false",
    "value.converter": "org.apache.kafka.connect.json.JsonConverter",
    "value.converter.schemas.enable": "true",

    "transforms": "unwrap,route",
    "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
    "transforms.unwrap.drop.tombstones": "false",
    "transforms.unwrap.delete.handling.mode": "rewrite",
    "transforms.unwrap.add.fields": "op,source.ts_ms,source.db,source.table",

    "transforms.route.type": "org.apache.kafka.connect.transforms.RegexRouter",
    "transforms.route.regex": "([^.]+)\\.([^.]+)\\.([^.]+)",
    "transforms.route.replacement": "cdc.$3",

    "heartbeat.interval.ms": "10000",
    "heartbeat.topics.prefix": "__debezium-heartbeat",

    "skipped.operations": "t",

    "tombstones.on.delete": "true",

    "min.row.count.to.stream.results": "1000",
    "max.batch.size": "2048",
    "max.queue.size": "8192",
    "poll.interval.ms": "1000"
  }
}
```

### 部署Debezium

```bash
# 1. 启动Kafka Connect
docker run -d \
  --name kafka-connect \
  --network=kafka-network \
  -p 8083:8083 \
  -e BOOTSTRAP_SERVERS=kafka1:9092,kafka2:9092,kafka3:9092 \
  -e GROUP_ID=debezium-cluster \
  -e CONFIG_STORAGE_TOPIC=debezium_connect_configs \
  -e OFFSET_STORAGE_TOPIC=debezium_connect_offsets \
  -e STATUS_STORAGE_TOPIC=debezium_connect_statuses \
  -e KEY_CONVERTER=org.apache.kafka.connect.json.JsonConverter \
  -e VALUE_CONVERTER=org.apache.kafka.connect.json.JsonConverter \
  -e CONNECT_KEY_CONVERTER_SCHEMAS_ENABLE=false \
  -e CONNECT_VALUE_CONVERTER_SCHEMAS_ENABLE=true \
  debezium/connect:2.5

# 2. 创建Connector
curl -X POST http://localhost:8083/connectors \
  -H "Content-Type: application/json" \
  -d @mysql-orders-connector.json

# 3. 查看Connector状态
curl http://localhost:8083/connectors/mysql-orders-connector/status | jq

# 4. 消费CDC事件
kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic cdc.orders \
  --from-beginning \
  --property print.key=true
```

### CDC事件格式

```json
{
  "before": null,
  "after": {
    "order_id": 1001,
    "user_id": 5001,
    "product_id": 3001,
    "quantity": 2,
    "price": 99.99,
    "total_amount": 199.98,
    "order_status": "paid",
    "payment_method": "credit_card",
    "created_at": "2026-02-07T10:30:00Z",
    "updated_at": "2026-02-07T10:30:00Z"
  },
  "source": {
    "version": "2.5.0.Final",
    "connector": "mysql",
    "name": "dbserver1",
    "ts_ms": 1707303000000,
    "snapshot": "false",
    "db": "inventory",
    "table": "orders",
    "server_id": 184054,
    "gtid": null,
    "file": "mysql-bin.000123",
    "pos": 456789,
    "row": 0,
    "thread": 7,
    "query": null
  },
  "op": "c",
  "ts_ms": 1707303000123,
  "transaction": null
}
```

## Flink流处理

### Flink SQL实时订单分析

```sql
-- 1. 创建Kafka源表 (订单CDC流)
CREATE TABLE orders_cdc (
    order_id BIGINT,
    user_id BIGINT,
    product_id BIGINT,
    quantity INT,
    price DECIMAL(10, 2),
    total_amount DECIMAL(10, 2),
    order_status STRING,
    payment_method STRING,
    created_at TIMESTAMP(3),
    updated_at TIMESTAMP(3),
    op STRING,
    ts_ms BIGINT,
    proc_time AS PROCTIME(),
    event_time AS TO_TIMESTAMP(FROM_UNIXTIME(ts_ms / 1000)),
    WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
) WITH (
    'connector' = 'kafka',
    'topic' = 'cdc.orders',
    'properties.bootstrap.servers' = 'kafka1:9092,kafka2:9092,kafka3:9092',
    'properties.group.id' = 'flink-orders-consumer',
    'scan.startup.mode' = 'latest-offset',
    'format' = 'json',
    'json.fail-on-missing-field' = 'false',
    'json.ignore-parse-errors' = 'true'
);

-- 2. 创建用户维度表
CREATE TABLE dim_users (
    user_id BIGINT,
    user_name STRING,
    user_level STRING,
    register_date DATE,
    city STRING,
    PRIMARY KEY (user_id) NOT ENFORCED
) WITH (
    'connector' = 'jdbc',
    'url' = 'jdbc:mysql://mysql:3306/warehouse',
    'table-name' = 'dim_users',
    'username' = 'flink',
    'password' = 'password',
    'lookup.cache.max-rows' = '10000',
    'lookup.cache.ttl' = '1 hour'
);

-- 3. 创建产品维度表
CREATE TABLE dim_products (
    product_id BIGINT,
    product_name STRING,
    category STRING,
    brand STRING,
    cost_price DECIMAL(10, 2),
    PRIMARY KEY (product_id) NOT ENFORCED
) WITH (
    'connector' = 'jdbc',
    'url' = 'jdbc:mysql://mysql:3306/warehouse',
    'table-name' = 'dim_products',
    'username' = 'flink',
    'password' = 'password',
    'lookup.cache.max-rows' = '50000',
    'lookup.cache.ttl' = '2 hour'
);

-- 4. 实时订单宽表 (关联维度)
CREATE VIEW enriched_orders AS
SELECT
    o.order_id,
    o.user_id,
    u.user_name,
    u.user_level,
    u.city,
    o.product_id,
    p.product_name,
    p.category,
    p.brand,
    o.quantity,
    o.price,
    p.cost_price,
    o.total_amount,
    (o.price - p.cost_price) * o.quantity AS profit,
    o.order_status,
    o.payment_method,
    o.created_at,
    o.event_time
FROM orders_cdc o
LEFT JOIN dim_users FOR SYSTEM_TIME AS OF o.proc_time AS u
    ON o.user_id = u.user_id
LEFT JOIN dim_products FOR SYSTEM_TIME AS OF o.proc_time AS p
    ON o.product_id = p.product_id
WHERE o.op IN ('c', 'r');  -- 只处理INSERT和READ事件

-- 5. 实时GMV统计 (滚动窗口 - 每分钟)
CREATE TABLE gmv_per_minute (
    window_start TIMESTAMP(3),
    window_end TIMESTAMP(3),
    order_count BIGINT,
    total_gmv DECIMAL(18, 2),
    avg_order_value DECIMAL(10, 2),
    PRIMARY KEY (window_start, window_end) NOT ENFORCED
) WITH (
    'connector' = 'jdbc',
    'url' = 'jdbc:mysql://mysql:3306/warehouse',
    'table-name' = 'rt_gmv_per_minute',
    'username' = 'flink',
    'password' = 'password',
    'sink.buffer-flush.max-rows' = '100',
    'sink.buffer-flush.interval' = '10s'
);

INSERT INTO gmv_per_minute
SELECT
    TUMBLE_START(event_time, INTERVAL '1' MINUTE) AS window_start,
    TUMBLE_END(event_time, INTERVAL '1' MINUTE) AS window_end,
    COUNT(*) AS order_count,
    SUM(total_amount) AS total_gmv,
    AVG(total_amount) AS avg_order_value
FROM enriched_orders
GROUP BY TUMBLE(event_time, INTERVAL '1' MINUTE);

-- 6. 实时品类销售排行 (滑动窗口 - 5分钟窗口, 1分钟滑动)
CREATE TABLE category_top_sales (
    window_start TIMESTAMP(3),
    window_end TIMESTAMP(3),
    category STRING,
    order_count BIGINT,
    total_sales DECIMAL(18, 2),
    total_profit DECIMAL(18, 2),
    ranking INT,
    PRIMARY KEY (window_start, window_end, category) NOT ENFORCED
) WITH (
    'connector' = 'jdbc',
    'url' = 'jdbc:mysql://mysql:3306/warehouse',
    'table-name' = 'rt_category_top_sales',
    'username' = 'flink',
    'password' = 'password'
);

INSERT INTO category_top_sales
SELECT
    window_start,
    window_end,
    category,
    order_count,
    total_sales,
    total_profit,
    ROW_NUMBER() OVER (
        PARTITION BY window_start, window_end
        ORDER BY total_sales DESC
    ) AS ranking
FROM (
    SELECT
        HOP_START(event_time, INTERVAL '1' MINUTE, INTERVAL '5' MINUTE) AS window_start,
        HOP_END(event_time, INTERVAL '1' MINUTE, INTERVAL '5' MINUTE) AS window_end,
        category,
        COUNT(*) AS order_count,
        SUM(total_amount) AS total_sales,
        SUM(profit) AS total_profit
    FROM enriched_orders
    WHERE category IS NOT NULL
    GROUP BY
        HOP(event_time, INTERVAL '1' MINUTE, INTERVAL '5' MINUTE),
        category
)
WHERE ranking <= 10;

-- 7. 实时用户行为分析 (会话窗口)
CREATE TABLE user_session_analysis (
    user_id BIGINT,
    session_start TIMESTAMP(3),
    session_end TIMESTAMP(3),
    session_duration_minutes BIGINT,
    order_count BIGINT,
    total_spending DECIMAL(18, 2),
    PRIMARY KEY (user_id, session_start) NOT ENFORCED
) WITH (
    'connector' = 'jdbc',
    'url' = 'jdbc:mysql://mysql:3306/warehouse',
    'table-name' = 'rt_user_sessions',
    'username' = 'flink',
    'password' = 'password'
);

INSERT INTO user_session_analysis
SELECT
    user_id,
    SESSION_START(event_time, INTERVAL '30' MINUTE) AS session_start,
    SESSION_END(event_time, INTERVAL '30' MINUTE) AS session_end,
    TIMESTAMPDIFF(
        MINUTE,
        SESSION_START(event_time, INTERVAL '30' MINUTE),
        SESSION_END(event_time, INTERVAL '30' MINUTE)
    ) AS session_duration_minutes,
    COUNT(*) AS order_count,
    SUM(total_amount) AS total_spending
FROM enriched_orders
GROUP BY
    user_id,
    SESSION(event_time, INTERVAL '30' MINUTE);

-- 8. 实时异常检测 (金额异常)
CREATE TABLE suspicious_orders (
    order_id BIGINT,
    user_id BIGINT,
    user_name STRING,
    product_id BIGINT,
    product_name STRING,
    quantity INT,
    price DECIMAL(10, 2),
    total_amount DECIMAL(18, 2),
    user_avg_spending DECIMAL(10, 2),
    deviation_ratio DECIMAL(10, 2),
    alert_time TIMESTAMP(3),
    alert_reason STRING,
    PRIMARY KEY (order_id) NOT ENFORCED
) WITH (
    'connector' = 'jdbc',
    'url' = 'jdbc:mysql://mysql:3306/warehouse',
    'table-name' = 'rt_suspicious_orders',
    'username' = 'flink',
    'password' = 'password'
);

-- 计算用户历史平均消费
CREATE VIEW user_avg_spending AS
SELECT
    user_id,
    AVG(total_amount) OVER (
        PARTITION BY user_id
        ORDER BY event_time
        ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING
    ) AS avg_spending
FROM enriched_orders;

-- 检测异常订单
INSERT INTO suspicious_orders
SELECT
    o.order_id,
    o.user_id,
    o.user_name,
    o.product_id,
    o.product_name,
    o.quantity,
    o.price,
    o.total_amount,
    u.avg_spending AS user_avg_spending,
    CASE
        WHEN u.avg_spending > 0
        THEN (o.total_amount - u.avg_spending) / u.avg_spending * 100
        ELSE 0
    END AS deviation_ratio,
    CURRENT_TIMESTAMP AS alert_time,
    CASE
        WHEN o.total_amount > u.avg_spending * 5 THEN '订单金额异常高'
        WHEN o.quantity > 50 THEN '购买数量异常'
        ELSE '其他异常'
    END AS alert_reason
FROM enriched_orders o
JOIN user_avg_spending u ON o.user_id = u.user_id
WHERE
    o.total_amount > u.avg_spending * 5  -- 金额超过平均值5倍
    OR o.quantity > 50;  -- 数量超过50
```

### Flink DataStream API示例

```java
package com.company.flink;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.util.Collector;

import java.time.Duration;

/**
 * 实时订单GMV统计
 */
public class OrderGMVAnalysis {

    public static void main(String[] args) throws Exception {
        // 1. 创建执行环境
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(3);
        env.enableCheckpointing(60000); // 1分钟checkpoint

        // 2. 配置Kafka Source
        KafkaSource<String> source = KafkaSource.<String>builder()
            .setBootstrapServers("kafka1:9092,kafka2:9092,kafka3:9092")
            .setTopics("cdc.orders")
            .setGroupId("flink-gmv-consumer")
            .setStartingOffsets(OffsetsInitializer.latest())
            .setValueOnlyDeserializer(new SimpleStringSchema())
            .build();

        // 3. 创建数据流
        DataStream<String> orderStream = env
            .fromSource(source, WatermarkStrategy.noWatermarks(), "Kafka Source");

        // 4. 解析JSON并过滤
        DataStream<OrderEvent> parsedStream = orderStream
            .map(json -> parseOrderEvent(json))
            .filter(order -> order != null && order.getOp().equals("c"));

        // 5. 设置Watermark
        DataStream<OrderEvent> watermarkedStream = parsedStream
            .assignTimestampsAndWatermarks(
                WatermarkStrategy
                    .<OrderEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                    .withTimestampAssigner((event, timestamp) -> event.getTimestamp())
            );

        // 6. 按分钟统计GMV
        DataStream<GMVResult> gmvStream = watermarkedStream
            .keyBy(order -> "global")
            .window(TumblingEventTimeWindows.of(Time.minutes(1)))
            .aggregate(new GMVAggregateFunction());

        // 7. 输出结果
        gmvStream.print();

        // 8. 写入MySQL
        gmvStream.addSink(new JdbcSink<>(/* JDBC配置 */));

        // 9. 执行任务
        env.execute("Order GMV Analysis");
    }

    // GMV聚合函数
    public static class GMVAggregateFunction
            implements AggregateFunction<OrderEvent, GMVAccumulator, GMVResult> {

        @Override
        public GMVAccumulator createAccumulator() {
            return new GMVAccumulator();
        }

        @Override
        public GMVAccumulator add(OrderEvent order, GMVAccumulator acc) {
            acc.count++;
            acc.totalAmount += order.getTotalAmount();
            return acc;
        }

        @Override
        public GMVResult getResult(GMVAccumulator acc) {
            return new GMVResult(
                acc.count,
                acc.totalAmount,
                acc.totalAmount / acc.count
            );
        }

        @Override
        public GMVAccumulator merge(GMVAccumulator a, GMVAccumulator b) {
            a.count += b.count;
            a.totalAmount += b.totalAmount;
            return a;
        }
    }

    // 累加器
    public static class GMVAccumulator {
        long count = 0;
        double totalAmount = 0.0;
    }
}
```

## 数据质量检查

### Great Expectations配置

```python
import great_expectations as ge
from great_expectations.core import ExpectationConfiguration
from great_expectations.data_context import DataContext

# 创建数据上下文
context = DataContext()

# 创建Expectation Suite
suite_name = "orders_quality_suite"
suite = context.create_expectation_suite(
    expectation_suite_name=suite_name,
    overwrite_existing=True
)

# 定义数据质量规则
expectations = [
    # 1. 主键唯一性
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_unique",
        kwargs={"column": "order_id"}
    ),

    # 2. 非空检查
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_not_be_null",
        kwargs={"column": "user_id"}
    ),

    # 3. 数值范围
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column": "quantity",
            "min_value": 1,
            "max_value": 1000
        }
    ),

    # 4. 枚举值检查
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_in_set",
        kwargs={
            "column": "order_status",
            "value_set": ["pending", "paid", "shipped", "delivered", "cancelled"]
        }
    ),

    # 5. 正则表达式
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_match_regex",
        kwargs={
            "column": "email",
            "regex": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        }
    ),

    # 6. 列存在性
    ExpectationConfiguration(
        expectation_type="expect_table_columns_to_match_ordered_list",
        kwargs={
            "column_list": ["order_id", "user_id", "product_id", "quantity", "price"]
        }
    ),

    # 7. 行数检查
    ExpectationConfiguration(
        expectation_type="expect_table_row_count_to_be_between",
        kwargs={
            "min_value": 100,
            "max_value": 1000000
        }
    ),
]

# 添加到Suite
for exp in expectations:
    suite.add_expectation(expectation_configuration=exp)

# 保存Suite
context.save_expectation_suite(suite, suite_name)
```

## 监控与告警

### Prometheus监控指标

```yaml
# airflow_exporter配置
---
apiVersion: v1
kind: Service
metadata:
  name: airflow-exporter
  labels:
    app: airflow
spec:
  ports:
  - port: 9112
    name: metrics
  selector:
    app: airflow
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: airflow-metrics
spec:
  selector:
    matchLabels:
      app: airflow
  endpoints:
  - port: metrics
    interval: 30s
```

### Grafana仪表板

```json
{
  "dashboard": {
    "title": "数据管道监控",
    "panels": [
      {
        "title": "DAG执行成功率",
        "targets": [{
          "expr": "rate(airflow_dag_run_success_total[5m]) / rate(airflow_dag_run_total[5m]) * 100"
        }]
      },
      {
        "title": "任务执行时长",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(airflow_task_duration_seconds_bucket[5m]))"
        }]
      },
      {
        "title": "Kafka消费延迟",
        "targets": [{
          "expr": "kafka_consumer_lag"
        }]
      },
      {
        "title": "Flink Checkpoint成功率",
        "targets": [{
          "expr": "flink_jobmanager_job_lastCheckpointDuration / 1000"
        }]
      }
    ]
  }
}
```

## 实战案例

### 完整数据管道案例：用户行为分析

```
数据源 --> CDC --> Kafka --> Flink --> 数据仓库 --> BI报表

1. 数据采集
   - 用户行为日志 (Nginx)
   - 业务数据库 (MySQL CDC)
   - 第三方API数据

2. 实时处理
   - 日志解析与清洗
   - 会话识别与关联
   - 实时特征计算

3. 数据存储
   - 原始数据 --> HDFS
   - 处理数据 --> ClickHouse
   - 维度数据 --> MySQL

4. 数据应用
   - 实时大屏
   - 推荐系统
   - 用户画像
```

## 总结

数据管道设计的关键要素：

1. **可靠性**：重试机制、数据校验、监控告警
2. **可扩展性**：分布式架构、并行处理、弹性伸缩
3. **实时性**：CDC捕获、流式处理、秒级延迟
4. **数据质量**：自动化检查、异常检测、数据治理
5. **可维护性**：清晰架构、完善文档、标准化流程

核心工具链：
- **调度编排**：Airflow、DolphinScheduler
- **数据采集**：Debezium、Maxwell、Canal
- **消息队列**：Kafka、Pulsar、RocketMQ
- **流处理**：Flink、Spark Streaming、Kafka Streams
- **数据质量**：Great Expectations、Deequ、Griffin
