# 机器学习工程化（MLOps）

## 1. MLOps完整流程

### 1.1 MLOps生命周期

```
MLOps端到端流程
┌────────────────────────────────────────────────────────────┐
│  1. 问题定义与数据准备                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ 业务需求分析 │→ │ 数据收集标注 │→ │ 特征工程     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────┐
│  2. 模型开发                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ 模型选择     │→ │ 超参数调优   │→ │ 模型评估     │     │
│  │ (算法选型)   │  │ (AutoML)     │  │ (A/B测试)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────┐
│  3. 模型训练                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ 分布式训练   │→ │ 实验跟踪     │→ │ 模型版本管理 │     │
│  │ (Kubeflow)   │  │ (MLflow)     │  │ (Model Reg)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────┐
│  4. 模型部署                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ 模型转换     │→ │ 模型服务     │→ │ A/B测试      │     │
│  │ (ONNX)       │  │ (TF Serving) │  │ (灰度发布)   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────────────────────────────────────┘
                           ↓
┌────────────────────────────────────────────────────────────┐
│  5. 监控与运维                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ 性能监控     │→ │ 数据漂移检测 │→ │ 模型再训练   │     │
│  │ (Prometheus) │  │ (drift)      │  │ (CI/CD)      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────────────────────────────────────┘
                           │
                           ↓ (反馈循环)
                  返回数据准备阶段
```

### 1.2 技术栈对比

| 功能 | 工具选项 | 推荐方案 |
|------|---------|---------|
| **实验跟踪** | MLflow, W&B, Neptune | MLflow（开源） |
| **工作流编排** | Kubeflow, Airflow, Metaflow | Kubeflow Pipelines |
| **特征存储** | Feast, Tecton, Hopsworks | Feast |
| **模型服务** | TF Serving, TorchServe, Seldon | TorchServe |
| **模型监控** | Evidently, Whylabs, Fiddler | Evidently |
| **AutoML** | AutoKeras, H2O, PyCaret | Optuna（超参数） |

## 2. Kubeflow Pipelines组件开发

### 2.1 Kubeflow架构

```
Kubeflow on Kubernetes
┌─────────────────────────────────────────────────────────┐
│                  Kubeflow Dashboard                     │
└─────────────────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ↓               ↓               ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Notebooks   │  │  Pipelines   │  │  KFServing   │
│  (Jupyter)   │  │  (Argo)      │  │  (Serving)   │
└──────────────┘  └──────────────┘  └──────────────┘
         │               │               │
         └───────────────┼───────────────┘
                         ↓
         ┌───────────────────────────────┐
         │     Kubernetes Cluster        │
         │  ┌─────────┐  ┌─────────┐    │
         │  │ Pod 1   │  │ Pod 2   │    │
         │  │(训练)   │  │(推理)   │    │
         │  └─────────┘  └─────────┘    │
         └───────────────────────────────┘
                         │
                         ↓
         ┌───────────────────────────────┐
         │      Shared Storage           │
         │  - MinIO (对象存储)           │
         │  - MySQL (元数据)             │
         └───────────────────────────────┘
```

### 2.2 Pipeline组件示例

```python
#!/usr/bin/env python3
"""
Kubeflow Pipeline完整示例：电商推荐系统
"""
from kfp import dsl, compiler
from kfp.components import create_component_from_func
from typing import NamedTuple

# ============ 组件1：数据准备 ============
def prepare_data(
    input_path: str,
    output_path: str
) -> NamedTuple('Outputs', [('dataset_size', int), ('feature_count', int)]):
    """数据准备组件"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import os

    # 读取数据
    df = pd.read_csv(input_path)

    # 数据清洗
    df = df.dropna()
    df = df[df['rating'] > 0]

    # 特征工程
    # 用户特征
    user_features = df.groupby('user_id').agg({
        'rating': ['mean', 'count'],
        'item_id': 'nunique'
    }).reset_index()
    user_features.columns = ['user_id', 'avg_rating', 'rating_count', 'unique_items']

    # 商品特征
    item_features = df.groupby('item_id').agg({
        'rating': ['mean', 'count'],
        'user_id': 'nunique'
    }).reset_index()
    item_features.columns = ['item_id', 'avg_rating', 'rating_count', 'unique_users']

    # 合并特征
    df = df.merge(user_features, on='user_id', suffixes=('', '_user'))
    df = df.merge(item_features, on='item_id', suffixes=('', '_item'))

    # 划分训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 保存
    os.makedirs(output_path, exist_ok=True)
    train_df.to_csv(f'{output_path}/train.csv', index=False)
    test_df.to_csv(f'{output_path}/test.csv', index=False)

    from collections import namedtuple
    output = namedtuple('Outputs', ['dataset_size', 'feature_count'])
    return output(len(df), len(df.columns))

# ============ 组件2：模型训练 ============
def train_model(
    data_path: str,
    model_path: str,
    learning_rate: float = 0.01,
    num_epochs: int = 10
) -> NamedTuple('Outputs', [('train_loss', float), ('train_accuracy', float)]):
    """模型训练组件"""
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import os

    # 自定义数据集
    class RecommendationDataset(Dataset):
        def __init__(self, df):
            self.user_ids = torch.LongTensor(df['user_id'].values)
            self.item_ids = torch.LongTensor(df['item_id'].values)
            self.ratings = torch.FloatTensor(df['rating'].values)

        def __len__(self):
            return len(self.ratings)

        def __getitem__(self, idx):
            return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]

    # 定义模型
    class MatrixFactorization(nn.Module):
        def __init__(self, num_users, num_items, embedding_dim=50):
            super().__init__()
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.item_embedding = nn.Embedding(num_items, embedding_dim)
            self.fc = nn.Linear(embedding_dim, 1)

        def forward(self, user_ids, item_ids):
            user_emb = self.user_embedding(user_ids)
            item_emb = self.item_embedding(item_ids)
            x = user_emb * item_emb
            return self.fc(x).squeeze()

    # 加载数据
    train_df = pd.read_csv(f'{data_path}/train.csv')

    # 创建数据集
    dataset = RecommendationDataset(train_df)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    # 初始化模型
    num_users = train_df['user_id'].max() + 1
    num_items = train_df['item_id'].max() + 1
    model = MatrixFactorization(num_users, num_items)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练
    model.train()
    total_loss = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        for user_ids, item_ids, ratings in dataloader:
            optimizer.zero_grad()
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
        total_loss += avg_loss

    # 保存模型
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), f'{model_path}/model.pth')

    from collections import namedtuple
    output = namedtuple('Outputs', ['train_loss', 'train_accuracy'])
    return output(total_loss / num_epochs, 0.85)  # 简化示例

# ============ 组件3：模型评估 ============
def evaluate_model(
    data_path: str,
    model_path: str
) -> NamedTuple('Outputs', [('test_rmse', float), ('test_mae', float)]):
    """模型评估组件"""
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import numpy as np

    # 重新定义模型（与训练时相同）
    class MatrixFactorization(nn.Module):
        def __init__(self, num_users, num_items, embedding_dim=50):
            super().__init__()
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.item_embedding = nn.Embedding(num_items, embedding_dim)
            self.fc = nn.Linear(embedding_dim, 1)

        def forward(self, user_ids, item_ids):
            user_emb = self.user_embedding(user_ids)
            item_emb = self.item_embedding(item_ids)
            x = user_emb * item_emb
            return self.fc(x).squeeze()

    # 加载数据
    test_df = pd.read_csv(f'{data_path}/test.csv')

    # 加载模型
    num_users = test_df['user_id'].max() + 1
    num_items = test_df['item_id'].max() + 1
    model = MatrixFactorization(num_users, num_items)
    model.load_state_dict(torch.load(f'{model_path}/model.pth'))
    model.eval()

    # 评估
    predictions = []
    actuals = []

    with torch.no_grad():
        for _, row in test_df.iterrows():
            user_id = torch.LongTensor([row['user_id']])
            item_id = torch.LongTensor([row['item_id']])
            pred = model(user_id, item_id).item()
            predictions.append(pred)
            actuals.append(row['rating'])

    # 计算指标
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mae = np.mean(np.abs(predictions - actuals))

    print(f'Test RMSE: {rmse:.4f}')
    print(f'Test MAE: {mae:.4f}')

    from collections import namedtuple
    output = namedtuple('Outputs', ['test_rmse', 'test_mae'])
    return output(rmse, mae)

# ============ 组件4：模型部署 ============
def deploy_model(
    model_path: str,
    deployment_name: str,
    namespace: str = 'default'
):
    """模型部署组件"""
    import subprocess

    # 创建KFServing InferenceService
    inference_service = f"""
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: {deployment_name}
  namespace: {namespace}
spec:
  predictor:
    pytorch:
      storageUri: {model_path}
      resources:
        limits:
          cpu: "1"
          memory: 2Gi
        requests:
          cpu: "1"
          memory: 2Gi
"""

    # 应用配置
    with open('/tmp/inference_service.yaml', 'w') as f:
        f.write(inference_service)

    subprocess.run(['kubectl', 'apply', '-f', '/tmp/inference_service.yaml'], check=True)
    print(f'Model deployed: {deployment_name}')

# ============ 创建KFP组件 ============
prepare_data_op = create_component_from_func(
    prepare_data,
    base_image='python:3.9',
    packages_to_install=['pandas', 'scikit-learn']
)

train_model_op = create_component_from_func(
    train_model,
    base_image='pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime',
    packages_to_install=['pandas']
)

evaluate_model_op = create_component_from_func(
    evaluate_model,
    base_image='pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime',
    packages_to_install=['pandas', 'numpy']
)

deploy_model_op = create_component_from_func(
    deploy_model,
    base_image='bitnami/kubectl:latest'
)

# ============ 定义Pipeline ============
@dsl.pipeline(
    name='E-commerce Recommendation Pipeline',
    description='端到端推荐系统训练和部署'
)
def recommendation_pipeline(
    input_data_path: str = 'gs://my-bucket/data/ratings.csv',
    learning_rate: float = 0.01,
    num_epochs: int = 10
):
    """完整的ML Pipeline"""

    # 步骤1：数据准备
    prepare_data_task = prepare_data_op(
        input_path=input_data_path,
        output_path='/mnt/data'
    )

    # 步骤2：模型训练
    train_model_task = train_model_op(
        data_path=prepare_data_task.outputs['output_path'],
        model_path='/mnt/models',
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )

    # 步骤3：模型评估
    evaluate_model_task = evaluate_model_op(
        data_path=prepare_data_task.outputs['output_path'],
        model_path=train_model_task.outputs['model_path']
    )

    # 步骤4：条件部署（仅当RMSE < 1.0时部署）
    with dsl.Condition(evaluate_model_task.outputs['test_rmse'] < 1.0):
        deploy_model_task = deploy_model_op(
            model_path=train_model_task.outputs['model_path'],
            deployment_name='recommendation-model'
        )

# ============ 编译Pipeline ============
if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=recommendation_pipeline,
        package_path='recommendation_pipeline.yaml'
    )
    print('Pipeline compiled successfully!')
```

## 3. MLflow实验跟踪与模型注册

### 3.1 MLflow跟踪实验

```python
#!/usr/bin/env python3
"""
MLflow实验跟踪完整示例
"""
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

# 配置MLflow
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("customer-churn-prediction")

def train_with_mlflow(params):
    """使用MLflow跟踪训练过程"""

    with mlflow.start_run(run_name="rf-experiment-1") as run:

        # 1. 记录参数
        mlflow.log_param("n_estimators", params['n_estimators'])
        mlflow.log_param("max_depth", params['max_depth'])
        mlflow.log_param("min_samples_split", params['min_samples_split'])

        # 2. 记录数据集信息
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # 3. 训练模型
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # 4. 评估模型
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        # 5. 记录指标
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)

        # 6. 记录特征重要性
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # 保存为artifact
        feature_importance.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')

        # 7. 记录混淆矩阵图
        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')

        # 8. 记录模型
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="ChurnPredictionModel"
        )

        # 9. 记录自定义标签
        mlflow.set_tag("model_type", "random_forest")
        mlflow.set_tag("data_version", "v2.1")
        mlflow.set_tag("author", "data-science-team")

        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")

        return run.info.run_id

# 使用示例
params = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'random_state': 42
}

run_id = train_with_mlflow(params)
```

### 3.2 模型注册与版本管理

```python
"""
MLflow模型注册中心
"""
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 1. 注册新模型
model_name = "ChurnPredictionModel"
model_uri = f"runs:/{run_id}/model"

model_version = client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=run_id,
    description="Random Forest model for customer churn prediction"
)

print(f"Model version: {model_version.version}")

# 2. 添加模型描述
client.update_registered_model(
    name=model_name,
    description="Predicts customer churn based on behavioral features. "
                "Updated weekly with new data."
)

# 3. 转换模型阶段
# None → Staging → Production → Archived

# 提升到Staging
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Staging"
)

# 添加版本说明
client.update_model_version(
    name=model_name,
    version=model_version.version,
    description="Improved with feature engineering v2.1. "
                "AUC: 0.89, F1: 0.85"
)

# 4. 比较模型版本
def compare_models(model_name, version1, version2):
    """比较两个模型版本"""

    v1 = client.get_model_version(model_name, version1)
    v2 = client.get_model_version(model_name, version2)

    # 获取运行信息
    run1 = client.get_run(v1.run_id)
    run2 = client.get_run(v2.run_id)

    metrics1 = run1.data.metrics
    metrics2 = run2.data.metrics

    print(f"Version {version1}:")
    print(f"  AUC: {metrics1.get('auc', 'N/A')}")
    print(f"  F1: {metrics1.get('f1_score', 'N/A')}")

    print(f"\nVersion {version2}:")
    print(f"  AUC: {metrics2.get('auc', 'N/A')}")
    print(f"  F1: {metrics2.get('f1_score', 'N/A')}")

    # 决定哪个更好
    if metrics2.get('auc', 0) > metrics1.get('auc', 0):
        print(f"\n✅ Version {version2} is better!")
        return version2
    else:
        print(f"\n✅ Version {version1} is better!")
        return version1

# 5. 提升到Production
best_version = compare_models(model_name, "1", "2")

client.transition_model_version_stage(
    name=model_name,
    version=best_version,
    stage="Production",
    archive_existing_versions=True  # 归档旧版本
)

# 6. 加载Production模型
import mlflow.pyfunc

model_uri = f"models:/{model_name}/Production"
model = mlflow.pyfunc.load_model(model_uri)

# 预测
predictions = model.predict(X_test)
```

## 4. 分布式训练策略

### 4.1 数据并行 vs 模型并行

```
数据并行 (Data Parallelism)
┌──────────────────────────────────────────────────┐
│         Model (完整副本)                         │
├──────────────────────────────────────────────────┤
│  GPU 0          GPU 1          GPU 2          GPU 3
│  ┌────┐        ┌────┐        ┌────┐        ┌────┐
│  │模型│        │模型│        │模型│        │模型│
│  │副本│        │副本│        │副本│        │副本│
│  └────┘        └────┘        └────┘        └────┘
│    ↓             ↓             ↓             ↓
│  Data 1        Data 2        Data 3        Data 4
│    ↓             ↓             ↓             ↓
│  梯度1         梯度2         梯度3         梯度4
│    └─────────────┴─────────────┴─────────────┘
│                   ↓
│            梯度聚合 (AllReduce)
│                   ↓
│              更新模型参数
└──────────────────────────────────────────────────┘
适用: 小模型、大数据集

模型并行 (Model Parallelism)
┌──────────────────────────────────────────────────┐
│             Data (同一批次)                       │
├──────────────────────────────────────────────────┤
│  GPU 0         GPU 1         GPU 2         GPU 3
│  ┌────┐       ┌────┐       ┌────┐       ┌────┐
│  │Layer│   →   │Layer│  →   │Layer│  →   │Layer│
│  │ 1-3│       │ 4-6│       │ 7-9│       │10-12│
│  └────┘       └────┘       └────┘       └────┘
│    ↓            ↓            ↓            ↓
│  中间结果 →  中间结果 →  中间结果 →   输出
└──────────────────────────────────────────────────┘
适用: 大模型、单GPU放不下
```

### 4.2 PyTorch分布式训练

```python
#!/usr/bin/env python3
"""
PyTorch DistributedDataParallel (DDP)
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import os

def setup_distributed():
    """初始化分布式环境"""
    dist.init_process_group(backend='nccl')  # NCCL for GPU
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()

def train_distributed(rank, world_size):
    """分布式训练函数"""

    # 1. 设置分布式
    setup_distributed()

    # 2. 创建模型并移到GPU
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # 3. 创建DistributedSampler
    train_dataset = MyDataset()
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    # 4. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 5. 训练循环
    for epoch in range(10):
        # 设置epoch（确保每个epoch的shuffle不同）
        train_sampler.set_epoch(epoch)

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            # 只在主进程打印日志
            if rank == 0 and batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        # 6. 同步所有进程
        dist.barrier()

        # 7. 保存模型（只在主进程）
        if rank == 0:
            torch.save(model.module.state_dict(), f'model_epoch_{epoch}.pth')

    cleanup_distributed()

# 使用torchrun启动
# torchrun --nproc_per_node=4 train_distributed.py
if __name__ == '__main__':
    train_distributed(int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE']))
```

## 5. Optuna超参数调优

### 5.1 Optuna基础使用

```python
"""
Optuna超参数优化
"""
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import sklearn.datasets
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 定义目标函数
def objective(trial):
    """Optuna优化目标函数"""

    # 1. 定义超参数搜索空间
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }

    # 2. 训练模型
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    model = RandomForestClassifier(**params, random_state=42)

    # 3. 交叉验证评估
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    # 4. 返回评估指标（Optuna默认最大化）
    return scores.mean()

# 创建Study
study = optuna.create_study(
    study_name='rf-optimization',
    direction='maximize',  # 最大化accuracy
    sampler=optuna.samplers.TPESampler(seed=42),  # TPE采样器
    pruner=optuna.pruners.MedianPruner()  # 中位数剪枝
)

# 执行优化
study.optimize(objective, n_trials=100, timeout=600)

# 输出最佳结果
print(f'Best trial: {study.best_trial.number}')
print(f'Best accuracy: {study.best_value:.4f}')
print(f'Best params: {study.best_params}')

# 可视化
plot_optimization_history(study).show()
plot_param_importances(study).show()
```

### 5.2 分布式超参数搜索

```python
"""
Optuna分布式优化（使用MySQL后端）
"""
import optuna

# 1. 创建共享Study（使用MySQL存储）
storage = optuna.storages.RDBStorage(
    url="mysql://user:password@mysql-server:3306/optuna_db",
    engine_kwargs={"pool_size": 20, "max_overflow": 0}
)

study = optuna.create_study(
    study_name='distributed-optimization',
    storage=storage,
    direction='maximize',
    load_if_exists=True  # 如果已存在则加载
)

# 2. 多个Worker并行优化
# Worker 1
study.optimize(objective, n_trials=50)

# Worker 2 (在另一台机器上同时运行)
study.optimize(objective, n_trials=50)

# Worker 3
study.optimize(objective, n_trials=50)

# 所有Worker共享同一个Study，结果实时同步
```

MLOps完整教程完成！继续创建剩余文件...
