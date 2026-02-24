# Apache Airflow工作流调度实战

## 1. Airflow架构与核心概念

### 1.1 整体架构

```
Airflow架构
┌─────────────────────────────────────────────────────────────┐
│                        用户                                  │
│              (浏览器 / CLI / API)                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    Web Server                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ DAG视图  │  │ 任务日志 │  │ 触发DAG  │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   Metadata DB                                │
│              (PostgreSQL / MySQL)                            │
│  DAG定义、任务状态、调度历史、连接信息、变量                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    Scheduler                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ DAG解析      │  │ 调度决策     │  │ 任务分发     │      │
│  │ (DagFileProc)│  │ (SchedulerJob)│ │ (→Executor)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    Executor                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ LocalExecutor    │ CeleryExecutor  │ KubernetesExecutor│  │
│  │ (单机多进程)     │ (分布式Worker)  │ (K8s Pod)         │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────┼──────────┐
              ↓          ↓          ↓
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Worker-1 │ │ Worker-2 │ │ Worker-N │
        │ 执行Task │ │ 执行Task │ │ 执行Task │
        └──────────┘ └──────────┘ └──────────┘
```

**Executor类型对比**：

| Executor | 适用场景 | 扩展性 | 隔离性 | 复杂度 |
|----------|---------|--------|--------|--------|
| **SequentialExecutor** | 本地开发测试 | 无（串行） | 无 | 极低 |
| **LocalExecutor** | 小规模（单机） | 低（受单机限制） | 进程级 | 低 |
| **CeleryExecutor** | 中大规模生产 | 高（水平扩展） | 进程级 | 中 |
| **KubernetesExecutor** | 云原生/弹性 | 极高（Pod级） | 容器级 | 高 |

### 1.2 核心概念

| 概念 | 说明 | 示例 |
|------|------|------|
| **DAG** | 有向无环图，定义工作流 | `etl_daily_pipeline` |
| **Task** | DAG中的一个执行单元 | `extract_orders` |
| **Operator** | Task的具体执行逻辑 | `BashOperator`, `PythonOperator` |
| **Sensor** | 等待外部条件满足的特殊Operator | `FileSensor`, `HttpSensor` |
| **Hook** | 与外部系统交互的接口 | `HiveHook`, `MySqlHook` |
| **Connection** | 外部系统连接配置 | `mysql_conn_id='prod_db'` |
| **Variable** | 全局键值对配置 | `Variable.get('env')` |
| **XCom** | Task间传递小数据 | `ti.xcom_push(key, value)` |

```
Task生命周期
┌────────┐    ┌───────────┐    ┌────────┐    ┌─────────┐
│  none  │───→│ scheduled │───→│ queued │───→│ running │
└────────┘    └───────────┘    └────────┘    └────┬────┘
                                                   │
                                    ┌──────────────┼──────────────┐
                                    ↓              ↓              ↓
                              ┌──────────┐  ┌────────────┐  ┌─────────┐
                              │ success  │  │   failed   │  │ up_for  │
                              └──────────┘  └────────────┘  │ _retry  │
                                                            └────┬────┘
                                                                 │ 重试
                                                                 ↓
                                                            ┌────────┐
                                                            │ queued │
                                                            └────────┘
```

### 1.3 调度机制

```python
from datetime import datetime, timedelta
from airflow import DAG

# schedule_interval支持三种格式:

# 1. Cron表达式
dag = DAG('etl_daily', schedule_interval='0 2 * * *')      # 每天凌晨2点
dag = DAG('etl_hourly', schedule_interval='30 * * * *')     # 每小时30分
dag = DAG('etl_weekly', schedule_interval='0 0 * * 1')      # 每周一零点

# 2. timedelta
dag = DAG('check_every_10min', schedule_interval=timedelta(minutes=10))

# 3. 预设值
dag = DAG('daily', schedule_interval='@daily')     # 等价于 '0 0 * * *'
dag = DAG('hourly', schedule_interval='@hourly')   # 等价于 '0 * * * *'
dag = DAG('weekly', schedule_interval='@weekly')   # 等价于 '0 0 * * 0'

# catchup机制（重要！）
dag = DAG(
    'etl_daily',
    start_date=datetime(2026, 1, 1),
    schedule_interval='@daily',
    catchup=True,     # ✅ 补跑历史：从start_date到now逐日执行
    # catchup=False,  # ❌ 不补跑：只执行最新一次
)

# execution_date / logical_date 理解
# execution_date表示数据区间的开始时间，不是执行时间
# 例如：schedule_interval='@daily', execution_date='2026-02-23'
# 实际在 2026-02-24 00:00 执行，处理 2026-02-23 的数据
```

## 2. DAG编写实战

### 2.1 基础DAG

```python
"""
基础DAG示例：每日数据处理流水线
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# DAG默认参数
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,          # 不依赖上一次执行结果
    'email': ['data-team@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,                      # 失败重试3次
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,  # 指数退避
    'max_retry_delay': timedelta(minutes=30),
}

with DAG(
    dag_id='etl_daily_pipeline',
    default_args=default_args,
    description='每日ETL数据处理流水线',
    schedule_interval='0 2 * * *',     # 每天凌晨2点
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['etl', 'daily'],
    max_active_runs=1,                 # 同时只允许1个DAG运行
) as dag:

    # 任务1：检查数据源
    check_source = BashOperator(
        task_id='check_data_source',
        bash_command='hdfs dfs -test -d /data/raw/{{ ds }} && echo "OK" || exit 1',
    )

    # 任务2：数据清洗（Python）
    def clean_data(**context):
        execution_date = context['ds']           # 格式: 2026-02-23
        print(f"清洗 {execution_date} 的数据...")
        # 实际清洗逻辑
        record_count = 1000000
        context['ti'].xcom_push(key='record_count', value=record_count)  # 传递给下游

    clean = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
    )

    # 任务3：数据加载
    load = BashOperator(
        task_id='load_to_warehouse',
        bash_command='hive -f /scripts/load_dwd.hql --hivevar dt={{ ds }}',
    )

    # 任务4：数据质量检查
    def quality_check(**context):
        count = context['ti'].xcom_pull(task_ids='clean_data', key='record_count')
        if count < 100000:
            raise ValueError(f"数据量异常: {count} < 100000")
        print(f"数据质量检查通过，共 {count} 条记录")

    check_quality = PythonOperator(
        task_id='quality_check',
        python_callable=quality_check,
    )

    # 任务5：通知
    notify = BashOperator(
        task_id='send_notification',
        bash_command='echo "ETL完成: {{ ds }}" | mail -s "ETL Success" team@company.com',
        trigger_rule='all_success',
    )

    # 定义依赖关系
    check_source >> clean >> load >> check_quality >> notify
```

### 2.2 TaskFlow API (Airflow 2.x)

```python
"""
TaskFlow API - 更Pythonic的DAG定义方式
"""
from datetime import datetime
from airflow.decorators import dag, task

@dag(
    dag_id='taskflow_etl',
    schedule_interval='@daily',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['taskflow'],
)
def taskflow_etl():

    @task()
    def extract():
        """从数据源提取数据"""
        import json
        data = {"orders": 1500, "users": 300, "revenue": 50000}
        return data                # 自动XCom push

    @task()
    def transform(raw_data: dict):
        """数据转换"""
        transformed = {
            "total_orders": raw_data["orders"],
            "avg_order_value": raw_data["revenue"] / raw_data["orders"],
            "new_users": raw_data["users"],
        }
        return transformed         # 自动XCom push

    @task()
    def load(transformed_data: dict):
        """加载到目标系统"""
        print(f"写入数仓: {transformed_data}")

    # 自动推断依赖关系
    raw = extract()
    transformed = transform(raw)
    load(transformed)

    # 动态任务映射 (Airflow 2.3+)
    @task()
    def get_table_list():
        return ["orders", "users", "products", "events"]

    @task()
    def process_table(table_name: str):
        print(f"处理表: {table_name}")

    tables = get_table_list()
    process_table.expand(table_name=tables)   # 动态生成4个并行任务

# 实例化DAG
taskflow_etl()
```

### 2.3 常用Operator

| Operator | Provider包 | 用途 |
|----------|-----------|------|
| `BashOperator` | 内置 | 执行Bash命令 |
| `PythonOperator` | 内置 | 执行Python函数 |
| `SparkSubmitOperator` | `apache-airflow-providers-apache-spark` | 提交Spark任务 |
| `HiveOperator` | `apache-airflow-providers-apache-hive` | 执行HQL |
| `MySqlOperator` | `apache-airflow-providers-mysql` | 执行SQL |
| `HttpOperator` | `apache-airflow-providers-http` | HTTP请求 |
| `EmailOperator` | 内置 | 发送邮件 |
| `S3ToHiveOperator` | `apache-airflow-providers-amazon` | S3数据导入Hive |
| `SlackWebhookOperator` | `apache-airflow-providers-slack` | Slack通知 |

### 2.4 Sensor与触发

```python
from airflow.sensors.filesystem import FileSensor
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.sensors.http_sensor import HttpSensor

# 文件Sensor：等待文件出现
wait_for_file = FileSensor(
    task_id='wait_for_data_file',
    filepath='/data/raw/{{ ds }}/orders.csv',
    poke_interval=60,            # 每60秒检查一次
    timeout=3600,                # 最长等待1小时
    mode='reschedule',           # ✅ reschedule模式：释放Worker槽位
    # mode='poke',               # ❌ poke模式：持续占用Worker
)

# 外部任务Sensor：等待另一个DAG的任务完成
wait_for_upstream = ExternalTaskSensor(
    task_id='wait_for_ods_dag',
    external_dag_id='ods_import_daily',
    external_task_id='import_done',
    execution_delta=timedelta(hours=0),  # 同一execution_date
    timeout=7200,
    mode='reschedule',
)

# HTTP Sensor：等待API就绪
wait_for_api = HttpSensor(
    task_id='wait_for_api_ready',
    http_conn_id='data_api',
    endpoint='/health',
    response_check=lambda response: response.json()['status'] == 'ok',
    poke_interval=30,
    timeout=600,
)
```

**Trigger Rule（触发规则）**：

| 规则 | 说明 | 场景 |
|------|------|------|
| `all_success` | 所有上游成功（默认） | 正常依赖 |
| `all_failed` | 所有上游失败 | 错误处理分支 |
| `all_done` | 所有上游完成（无论成败） | 清理任务 |
| `one_success` | 至少一个上游成功 | 并行分支汇聚 |
| `one_failed` | 至少一个上游失败 | 告警任务 |
| `none_failed` | 没有上游失败（允许skip） | 条件分支后 |

## 3. 大数据任务调度

### 3.1 Spark任务

```python
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

# 提交Spark任务到YARN
spark_etl = SparkSubmitOperator(
    task_id='spark_etl_dwd',
    application='/opt/spark-jobs/etl_dwd.py',
    conn_id='spark_default',
    conf={
        'spark.executor.memory': '8g',
        'spark.executor.cores': '4',
        'spark.executor.instances': '10',
        'spark.driver.memory': '4g',
        'spark.sql.shuffle.partitions': '200',
        'spark.sql.adaptive.enabled': 'true',
        'spark.dynamicAllocation.enabled': 'true',
    },
    application_args=['--date', '{{ ds }}', '--env', 'prod'],
    name='etl_dwd_{{ ds }}',
    deploy_mode='cluster',
    verbose=True,
)
```

### 3.2 Hive任务

```python
from airflow.providers.apache.hive.operators.hive import HiveOperator

# 执行HQL脚本
hive_dws = HiveOperator(
    task_id='hive_build_dws',
    hql="""
        SET hive.exec.dynamic.partition=true;
        SET hive.exec.dynamic.partition.mode=nonstrict;

        INSERT OVERWRITE TABLE dws.user_daily_stats PARTITION(dt='{{ ds }}')
        SELECT
            user_id,
            COUNT(CASE WHEN action='view' THEN 1 END) AS view_cnt,
            COUNT(CASE WHEN action='cart' THEN 1 END) AS cart_cnt,
            COUNT(CASE WHEN action='buy' THEN 1 END)  AS buy_cnt,
            SUM(CASE WHEN action='buy' THEN amount ELSE 0 END) AS buy_amount
        FROM dwd.user_behavior
        WHERE dt = '{{ ds }}'
        GROUP BY user_id;
    """,
    hive_cli_conn_id='hive_default',
    schema='dws',
)
```

### 3.3 Sqoop任务

```python
# 使用BashOperator包装Sqoop命令
sqoop_import = BashOperator(
    task_id='sqoop_import_orders',
    bash_command="""
        sqoop import \
            --connect jdbc:mysql://mysql-prod:3306/ecommerce \
            --username {{ var.value.mysql_user }} \
            --password {{ var.value.mysql_pass }} \
            --table orders \
            --where "order_date='{{ ds }}'" \
            --target-dir /data/ods/orders/dt={{ ds }} \
            --delete-target-dir \
            --fields-terminated-by '\t' \
            --compress \
            --compression-codec snappy \
            --num-mappers 8
    """,
)
```

### 3.4 数据质量检查

```python
from airflow.providers.common.sql.operators.sql import (
    SQLCheckOperator, SQLValueCheckOperator
)

# 行数检查
check_row_count = SQLValueCheckOperator(
    task_id='check_row_count',
    conn_id='hive_default',
    sql="SELECT COUNT(*) FROM dwd.user_behavior WHERE dt='{{ ds }}'",
    pass_value=100000,          # 期望至少10万条
    tolerance=0.1,              # 允许10%误差
)

# 空值检查
check_nulls = SQLCheckOperator(
    task_id='check_null_ratio',
    conn_id='hive_default',
    sql="""
        SELECT
            CASE WHEN null_ratio < 0.05 THEN 1 ELSE 0 END
        FROM (
            SELECT COUNT(CASE WHEN user_id IS NULL THEN 1 END) * 1.0
                   / COUNT(*) AS null_ratio
            FROM dwd.user_behavior
            WHERE dt='{{ ds }}'
        ) t
    """,
)

# 自定义数据新鲜度检查
def check_data_freshness(**context):
    from airflow.providers.apache.hive.hooks.hive import HiveServer2Hook
    hook = HiveServer2Hook(hiveserver2_conn_id='hive_default')
    result = hook.get_first(f"""
        SELECT MAX(event_time) FROM dwd.user_behavior WHERE dt='{context['ds']}'
    """)
    max_time = result[0]
    if max_time is None:
        raise ValueError("数据为空！")
    print(f"最新数据时间: {max_time}")

freshness_check = PythonOperator(
    task_id='check_freshness',
    python_callable=check_data_freshness,
)
```

## 4. 高级特性

### 4.1 动态DAG生成

```python
"""
从YAML配置文件动态生成DAG
"""
import yaml
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# 配置文件: /opt/airflow/config/tables.yaml
# tables:
#   - name: orders
#     schedule: "0 2 * * *"
#     hql: /scripts/etl_orders.hql
#   - name: users
#     schedule: "0 3 * * *"
#     hql: /scripts/etl_users.hql
#   - name: products
#     schedule: "0 4 * * *"
#     hql: /scripts/etl_products.hql

with open('/opt/airflow/config/tables.yaml') as f:
    config = yaml.safe_load(f)

for table_config in config['tables']:
    dag_id = f"etl_{table_config['name']}"

    dag = DAG(
        dag_id=dag_id,
        schedule_interval=table_config['schedule'],
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=['auto-generated', 'etl'],
    )

    extract = BashOperator(
        task_id='extract',
        bash_command=f"sqoop import --table {table_config['name']} ...",
        dag=dag,
    )

    transform = BashOperator(
        task_id='transform',
        bash_command=f"hive -f {table_config['hql']} --hivevar dt={{{{ ds }}}}",
        dag=dag,
    )

    extract >> transform

    # 必须注册到全局命名空间
    globals()[dag_id] = dag
```

### 4.2 分支与条件

```python
from airflow.operators.python import BranchPythonOperator, ShortCircuitOperator

# 分支操作：根据条件选择不同路径
def choose_branch(**context):
    ds = context['ds']
    day_of_week = datetime.strptime(ds, '%Y-%m-%d').weekday()
    if day_of_week == 0:  # 周一
        return 'run_weekly_report'      # 返回task_id
    else:
        return 'run_daily_report'

branch = BranchPythonOperator(
    task_id='branch_decision',
    python_callable=choose_branch,
)

daily_report = BashOperator(task_id='run_daily_report', bash_command='echo daily')
weekly_report = BashOperator(task_id='run_weekly_report', bash_command='echo weekly')

# 汇合节点必须设置trigger_rule='none_failed_min_one_success'
merge = BashOperator(
    task_id='merge_results',
    bash_command='echo done',
    trigger_rule='none_failed_min_one_success',
)

branch >> [daily_report, weekly_report] >> merge

# 短路操作：条件不满足则跳过下游所有任务
def check_is_holiday(**context):
    holidays = ['2026-01-01', '2026-02-17', '2026-02-18']
    return context['ds'] not in holidays  # True继续，False跳过

skip_if_holiday = ShortCircuitOperator(
    task_id='skip_if_holiday',
    python_callable=check_is_holiday,
)
```

### 4.3 TaskGroup

```python
from airflow.utils.task_group import TaskGroup

with DAG('grouped_etl', schedule_interval='@daily',
         start_date=datetime(2026, 1, 1), catchup=False) as dag:

    # 抽取阶段
    with TaskGroup('extract') as extract_group:
        extract_orders = BashOperator(task_id='orders', bash_command='echo extract orders')
        extract_users = BashOperator(task_id='users', bash_command='echo extract users')
        extract_products = BashOperator(task_id='products', bash_command='echo extract products')

    # 转换阶段
    with TaskGroup('transform') as transform_group:
        build_dwd = BashOperator(task_id='build_dwd', bash_command='echo build dwd')
        build_dws = BashOperator(task_id='build_dws', bash_command='echo build dws')
        build_dwd >> build_dws

    # 加载阶段
    with TaskGroup('load') as load_group:
        export_to_mysql = BashOperator(task_id='to_mysql', bash_command='echo export mysql')
        push_to_redis = BashOperator(task_id='to_redis', bash_command='echo push redis')

    # 组间依赖
    extract_group >> transform_group >> load_group
```

### 4.4 SLA与告警

```python
import requests
from airflow.operators.python import PythonOperator

# SLA配置：任务超时告警
task_with_sla = PythonOperator(
    task_id='critical_etl',
    python_callable=etl_func,
    sla=timedelta(hours=2),     # 必须在2小时内完成
)

# SLA miss回调
def sla_miss_callback(dag, task_list, blocking_task_list, slas, blocking_tis):
    msg = f"SLA告警！DAG: {dag.dag_id}, 超时任务: {[t.task_id for t in task_list]}"
    # 发送钉钉告警
    requests.post('https://oapi.dingtalk.com/robot/send?access_token=xxx', json={
        "msgtype": "text",
        "text": {"content": msg}
    })

dag = DAG('etl_with_sla', sla_miss_callback=sla_miss_callback, ...)

# 失败回调
def on_failure(context):
    task_id = context['task_instance'].task_id
    dag_id = context['dag'].dag_id
    log_url = context['task_instance'].log_url
    exception = context.get('exception', 'Unknown')

    msg = f"任务失败！DAG: {dag_id}, Task: {task_id}\n异常: {exception}\n日志: {log_url}"

    # 发送Slack告警
    from airflow.providers.slack.hooks.slack_webhook import SlackWebhookHook
    hook = SlackWebhookHook(slack_webhook_conn_id='slack_alerts')
    hook.send(text=msg)

# 在DAG级别设置回调
default_args = {
    'on_failure_callback': on_failure,
    'on_retry_callback': lambda ctx: print(f"重试: {ctx['task_instance'].task_id}"),
}
```

## 5. 部署与运维

### 5.1 部署模式

```
CeleryExecutor架构 (生产推荐)
┌────────────┐    ┌──────────────┐    ┌────────────┐
│ Web Server │    │  Scheduler   │    │  Flower    │
│  (Gunicorn)│    │  (调度进程)  │    │ (监控面板) │
└─────┬──────┘    └──────┬───────┘    └────────────┘
      │                  │
      │     ┌────────────┼────────────┐
      │     │            │            │
      ↓     ↓            ↓            ↓
┌──────────────┐  ┌──────────────┐
│  Metadata DB │  │ Message Broker│
│ (PostgreSQL) │  │ (Redis/RabbitMQ)│
└──────────────┘  └──────┬───────┘
                         │
              ┌──────────┼──────────┐
              ↓          ↓          ↓
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Worker-1 │ │ Worker-2 │ │ Worker-3 │
        │ (Celery) │ │ (Celery) │ │ (Celery) │
        └──────────┘ └──────────┘ └──────────┘
```

### 5.2 Docker/K8s部署

```yaml
# docker-compose.yaml (Airflow + CeleryExecutor)
version: '3.8'
x-airflow-common: &airflow-common
  image: apache/airflow:2.8.1
  environment:
    AIRFLOW__CORE__EXECUTOR: CeleryExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://airflow:airflow@postgres/airflow
    AIRFLOW__CELERY__BROKER_URL: redis://:@redis:6379/0
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
    AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    AIRFLOW__SCHEDULER__DAG_DIR_LIST_INTERVAL: 30
  volumes:
    - ./dags:/opt/airflow/dags
    - ./logs:/opt/airflow/logs
    - ./plugins:/opt/airflow/plugins
    - ./config:/opt/airflow/config

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-data:/var/lib/postgresql/data

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  airflow-webserver:
    <<: *airflow-common
    command: webserver
    ports:
      - "8080:8080"
    depends_on:
      - postgres
      - redis

  airflow-scheduler:
    <<: *airflow-common
    command: scheduler
    depends_on:
      - postgres
      - redis

  airflow-worker:
    <<: *airflow-common
    command: celery worker
    depends_on:
      - airflow-scheduler
    deploy:
      replicas: 3                  # 3个Worker

  flower:
    <<: *airflow-common
    command: celery flower
    ports:
      - "5555:5555"

volumes:
  postgres-data:
```

### 5.3 配置管理

**关键配置参数**：

| 配置项 | 推荐值 | 说明 |
|--------|--------|------|
| `core.parallelism` | 32-128 | 最大同时运行的Task数 |
| `core.dag_concurrency` | 16-32 | 单个DAG最大并行Task数 |
| `core.max_active_runs_per_dag` | 1-3 | 单个DAG最大同时运行实例 |
| `scheduler.parsing_processes` | 2-4 | DAG解析进程数 |
| `scheduler.dag_dir_list_interval` | 30-300 | DAG目录扫描间隔(秒) |
| `celery.worker_concurrency` | 16-32 | 每个Worker并行Task数 |
| `webserver.worker_refresh_interval` | 30 | Web Worker刷新间隔 |

```bash
# Connection管理
# 方式1：Web UI -> Admin -> Connections
# 方式2：CLI
airflow connections add 'hive_default' \
    --conn-type 'hive_cli' \
    --conn-host 'hiveserver2-host' \
    --conn-port 10000 \
    --conn-schema 'default'

# 方式3：环境变量
export AIRFLOW_CONN_MYSQL_PROD='mysql://user:pass@host:3306/db'

# Variable管理
airflow variables set 'env' 'production'
airflow variables set 'alert_email' 'team@company.com'
```

## 6. 实战案例：数仓ETL调度平台

### 6.1 需求分析

```
每日数仓ETL调度流程
┌─────────────────────────────────────────────────────────────┐
│  T+1 00:30        T+1 01:00        T+1 02:00    T+1 03:00  │
│                                                              │
│  ┌─────────┐    ┌─────────────┐    ┌────────┐    ┌───────┐ │
│  │   ODS   │───→│    DWD      │───→│  DWS   │───→│  ADS  │ │
│  │  数据导入│    │  数据清洗   │    │ 汇总层 │    │ 应用层│ │
│  └────┬────┘    └──────┬──────┘    └───┬────┘    └───┬───┘ │
│       │                │               │             │      │
│  Sqoop Import    Spark ETL        Hive SQL      Export      │
│  (MySQL→HDFS)    (清洗+转换)      (聚合计算)    (→MySQL/ES) │
│       │                │               │             │      │
│       ↓                ↓               ↓             ↓      │
│  ┌─────────┐    ┌──────────┐    ┌────────┐    ┌───────┐   │
│  │质量检查 │    │质量检查  │    │质量检查│    │BI推送 │   │
│  └─────────┘    └──────────┘    └────────┘    └───────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 DAG实现

```python
"""
数仓ETL全链路DAG
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.apache.hive.operators.hive import HiveOperator
from airflow.providers.common.sql.operators.sql import SQLCheckOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.task_group import TaskGroup

default_args = {
    'owner': 'data-warehouse',
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
    'on_failure_callback': alert_on_failure,     # 前面定义的告警函数
}

with DAG(
    dag_id='dw_etl_daily',
    default_args=default_args,
    schedule_interval='30 0 * * *',              # 每天 00:30
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['data-warehouse', 'production'],
) as dag:

    # ========== ODS层：数据导入 ==========
    with TaskGroup('ods_import') as ods_group:
        import_orders = BashOperator(
            task_id='import_orders',
            bash_command="""
                sqoop import --connect jdbc:mysql://mysql:3306/ecommerce \
                    --table orders --where "dt='{{ ds }}'" \
                    --target-dir /warehouse/ods/orders/dt={{ ds }} \
                    --delete-target-dir -m 8
            """,
        )
        import_users = BashOperator(
            task_id='import_users',
            bash_command="""
                sqoop import --connect jdbc:mysql://mysql:3306/ecommerce \
                    --table users --where "updated_at>='{{ ds }}'" \
                    --target-dir /warehouse/ods/users/dt={{ ds }} \
                    --delete-target-dir -m 4
            """,
        )

        # ODS质量检查
        ods_check = PythonOperator(
            task_id='ods_quality_check',
            python_callable=check_ods_data,
        )

        [import_orders, import_users] >> ods_check

    # ========== DWD层：Spark ETL ==========
    with TaskGroup('dwd_etl') as dwd_group:
        dwd_orders = SparkSubmitOperator(
            task_id='dwd_orders',
            application='/opt/spark-jobs/dwd_orders.py',
            application_args=['--date', '{{ ds }}'],
            conf={'spark.executor.instances': '10', 'spark.executor.memory': '8g'},
            deploy_mode='cluster',
        )
        dwd_users = SparkSubmitOperator(
            task_id='dwd_users',
            application='/opt/spark-jobs/dwd_users.py',
            application_args=['--date', '{{ ds }}'],
            conf={'spark.executor.instances': '5', 'spark.executor.memory': '4g'},
            deploy_mode='cluster',
        )

        dwd_check = SQLCheckOperator(
            task_id='dwd_quality_check',
            conn_id='hive_default',
            sql="SELECT CASE WHEN COUNT(*)>0 THEN 1 ELSE 0 END FROM dwd.orders WHERE dt='{{ ds }}'",
        )

        [dwd_orders, dwd_users] >> dwd_check

    # ========== DWS层：Hive聚合 ==========
    with TaskGroup('dws_aggregate') as dws_group:
        dws_user_daily = HiveOperator(
            task_id='dws_user_daily',
            hql="/scripts/dws_user_daily.hql",
            hiveconf_dict={'dt': '{{ ds }}'},
        )
        dws_product_daily = HiveOperator(
            task_id='dws_product_daily',
            hql="/scripts/dws_product_daily.hql",
            hiveconf_dict={'dt': '{{ ds }}'},
        )

    # ========== ADS层：应用指标 ==========
    with TaskGroup('ads_metrics') as ads_group:
        ads_gmv = HiveOperator(
            task_id='ads_gmv',
            hql="/scripts/ads_gmv_daily.hql",
            hiveconf_dict={'dt': '{{ ds }}'},
        )
        ads_retention = HiveOperator(
            task_id='ads_retention',
            hql="/scripts/ads_retention.hql",
            hiveconf_dict={'dt': '{{ ds }}'},
        )

        # 导出到MySQL（供BI查询）
        export_to_mysql = BashOperator(
            task_id='export_to_mysql',
            bash_command="""
                sqoop export --connect jdbc:mysql://mysql:3306/bi_report \
                    --table daily_gmv --export-dir /warehouse/ads/gmv/dt={{ ds }} \
                    --update-mode allowinsert --update-key dt -m 4
            """,
        )

        [ads_gmv, ads_retention] >> export_to_mysql

    # ========== 通知 ==========
    notify_success = PythonOperator(
        task_id='notify_success',
        python_callable=send_success_notification,
        trigger_rule='all_success',
    )

    # 全链路依赖
    ods_group >> dwd_group >> dws_group >> ads_group >> notify_success
```

### 6.3 监控与告警

```python
# 自定义监控回调
def alert_on_failure(context):
    """统一失败告警"""
    ti = context['task_instance']
    dag_id = context['dag'].dag_id
    task_id = ti.task_id
    execution_date = context['ds']
    log_url = ti.log_url
    exception = str(context.get('exception', ''))[:500]

    # 钉钉告警
    import requests
    requests.post(
        'https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN',
        json={
            "msgtype": "markdown",
            "markdown": {
                "title": f"Airflow告警: {dag_id}",
                "text": f"### Airflow任务失败告警\n\n"
                        f"- **DAG**: {dag_id}\n"
                        f"- **Task**: {task_id}\n"
                        f"- **日期**: {execution_date}\n"
                        f"- **异常**: {exception}\n"
                        f"- [查看日志]({log_url})"
            }
        }
    )

def send_success_notification(**context):
    """每日ETL完成通知"""
    import requests
    requests.post(
        'https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN',
        json={
            "msgtype": "text",
            "text": {
                "content": f"数仓ETL完成 | 日期: {context['ds']} | 耗时: 正常"
            }
        }
    )
```

## 7. Airflow vs 其他调度工具

### 7.1 工具对比

| 维度 | Airflow | Oozie | DolphinScheduler | Azkaban | Prefect |
|------|---------|-------|-----------------|---------|---------|
| **语言** | Python | Java/XML | Java | Java | Python |
| **DAG定义** | Python代码 | XML | Web UI拖拽 | Properties | Python代码 |
| **UI** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **扩展性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **社区** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **学习曲线** | 中等 | 高 | 低 | 低 | 中等 |
| **大数据支持** | Provider丰富 | Hadoop原生 | 内置支持 | 有限 | 需扩展 |
| **云原生** | K8s Executor | 弱 | K8s部署 | 弱 | 原生 |
| **适用场景** | 通用ETL | Hadoop生态 | 国内企业 | 小规模 | 现代数据栈 |

### 7.2 最佳实践总结

```
✅ DAG设计
  ├── 幂等性：重复执行不产生副作用（OVERWRITE而非APPEND）
  ├── 原子性：每个Task是独立的执行单元
  ├── DAG文件中不放重逻辑（解析速度影响调度）
  ├── 使用模板变量 {{ ds }} 而非 datetime.now()
  └── 合理设置 max_active_runs 防止并行冲突

✅ 重试策略
  ├── retries=2-3（大数据任务可能瞬时失败）
  ├── retry_delay=timedelta(minutes=5)
  ├── retry_exponential_backoff=True
  └── max_retry_delay=timedelta(minutes=60)

✅ 监控与告警
  ├── 配置 on_failure_callback
  ├── 关键任务设置 SLA
  ├── 使用 Flower 监控 Celery Worker
  └── 接入 Prometheus + Grafana

❌ 常见反模式
  ├── ❌ 在DAG文件中连接数据库或执行查询
  ├── ❌ 使用 datetime.now() 代替 {{ ds }}
  ├── ❌ XCom传递大数据（应传路径而非数据）
  ├── ❌ 单个Task做太多事（应拆分）
  └── ❌ catchup=True 但未处理幂等性
```
