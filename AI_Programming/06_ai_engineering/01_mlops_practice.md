# MLOps实践完整教程

## 目录
1. [MLOps概述](#mlops概述)
2. [实验跟踪与MLflow](#实验跟踪与mlflow)
3. [模型注册与版本管理](#模型注册与版本管理)
4. [CI/CD for ML](#cicd-for-ml)
5. [数据版本控制](#数据版本控制)
6. [Feature Store](#feature-store)
7. [完整ML Pipeline](#完整ml-pipeline)
8. [生产环境最佳实践](#生产环境最佳实践)

---

## MLOps概述

### 什么是MLOps

MLOps（Machine Learning Operations）是将DevOps原则应用于机器学习系统的实践，
旨在自动化和标准化ML生命周期中的各个环节，包括数据准备、模型训练、部署、监控等。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MLOps 成熟度模型                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Level 0: 手动流程                                                          │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐         │
│  │ 数据准备  │───>│ 模型训练  │───>│ 手动部署  │───>│ 无监控    │         │
│  │  (手动)   │    │  (Jupyter)│    │ (脚本复制)│    │           │         │
│  └───────────┘    └───────────┘    └───────────┘    └───────────┘         │
│  问题: 无法复现, 无版本控制, 部署周期长                                      │
│                                                                             │
│  Level 1: ML Pipeline自动化                                                 │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐         │
│  │ 数据验证  │───>│ 自动训练  │───>│ 模型验证  │───>│ 自动部署  │         │
│  │ (Schema)  │    │(Pipeline) │    │ (测试集)  │    │ (CI/CD)   │         │
│  └─────┬─────┘    └─────┬─────┘    └─────┬─────┘    └─────┬─────┘         │
│        │                │                │                │               │
│        └────────────────┴────────────────┴────────────────┘               │
│                              │                                              │
│                    ┌─────────▼─────────┐                                   │
│                    │   实验跟踪系统     │                                   │
│                    │ (MLflow/W&B)       │                                   │
│                    └───────────────────┘                                   │
│                                                                             │
│  Level 2: CI/CD + 自动化再训练                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      持续监控 & 触发器                               │   │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐     │   │
│  │  │数据  │─>│特征  │─>│训练  │─>│验证  │─>│部署  │─>│监控  │──┐  │   │
│  │  │采集  │  │工程  │  │Pipeline│ │测试  │  │发布  │  │告警  │  │  │   │
│  │  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──┬───┘  │  │   │
│  │                                                         │      │  │   │
│  │  ┌───────────────────────────────────────────────────────┘      │  │   │
│  │  │  数据漂移/模型衰退检测 ──> 自动触发再训练                     │  │   │
│  │  └─────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  关键组件:                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ 版本控制 │ │ 实验跟踪 │ │ 模型注册 │ │ CI/CD    │ │ 监控告警 │       │
│  │ Git/DVC  │ │ MLflow   │ │ Registry │ │ GitHub   │ │Prometheus│       │
│  │          │ │ W&B      │ │          │ │ Actions  │ │ Grafana  │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### MLOps核心原则

```python
"""
MLOps 核心原则与组件概览
"""

# MLOps 核心原则定义
MLOPS_PRINCIPLES = {
    "可复现性": {
        "描述": "任何实验都能被完整复现",
        "工具": ["MLflow", "DVC", "Git"],
        "实践": [
            "代码版本控制 (Git)",
            "数据版本控制 (DVC)",
            "环境版本控制 (Docker)",
            "超参数记录 (MLflow)",
        ]
    },
    "自动化": {
        "描述": "从数据处理到部署的全流程自动化",
        "工具": ["Airflow", "Kubeflow", "GitHub Actions"],
        "实践": [
            "自动化训练Pipeline",
            "自动化测试与验证",
            "自动化部署 (CI/CD)",
            "自动化监控与告警",
        ]
    },
    "持续监控": {
        "描述": "部署后持续跟踪模型表现",
        "工具": ["Prometheus", "Grafana", "Evidently"],
        "实践": [
            "模型性能监控",
            "数据漂移检测",
            "资源使用监控",
            "业务指标追踪",
        ]
    },
    "协作治理": {
        "描述": "团队间高效协作与模型治理",
        "工具": ["MLflow Registry", "Model Cards"],
        "实践": [
            "模型审批流程",
            "A/B测试机制",
            "模型文档化",
            "权限与审计",
        ]
    },
}


def print_mlops_overview():
    """打印MLOps核心原则概览"""
    print("=" * 60)
    print("MLOps 核心原则概览")
    print("=" * 60)
    for principle, details in MLOPS_PRINCIPLES.items():
        print(f"\n{'='*40}")
        print(f"原则: {principle}")
        print(f"描述: {details['描述']}")
        print(f"工具: {', '.join(details['工具'])}")
        print("实践:")
        for practice in details['实践']:
            print(f"  - {practice}")


if __name__ == "__main__":
    print_mlops_overview()
```

---

## 实验跟踪与MLflow

### 实验跟踪架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MLflow 实验跟踪架构                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  数据科学家工作站                        MLflow Tracking Server              │
│  ┌──────────────────────┐               ┌────────────────────────┐         │
│  │  训练脚本            │               │  REST API              │         │
│  │  ┌────────────────┐  │   HTTP/REST   │  ┌──────────────────┐  │         │
│  │  │ mlflow.log_*() │──┼──────────────>│  │  参数/指标/工件  │  │         │
│  │  │ mlflow.start() │  │               │  │  存储与查询      │  │         │
│  │  │ mlflow.end()   │  │               │  └────────┬─────────┘  │         │
│  │  └────────────────┘  │               │           │            │         │
│  │                      │               │  ┌────────▼─────────┐  │         │
│  │  Jupyter Notebook    │               │  │  Backend Store   │  │         │
│  │  ┌────────────────┐  │               │  │ (PostgreSQL/     │  │         │
│  │  │ 交互式实验     │──┼──────────────>│  │  MySQL/SQLite)   │  │         │
│  │  └────────────────┘  │               │  └────────┬─────────┘  │         │
│  └──────────────────────┘               │           │            │         │
│                                         │  ┌────────▼─────────┐  │         │
│  CI/CD Pipeline                         │  │ Artifact Store   │  │         │
│  ┌──────────────────────┐               │  │ (S3/GCS/Azure/   │  │         │
│  │ 自动化训练任务       │               │  │  本地文件系统)    │  │         │
│  │  ┌────────────────┐  │               │  └──────────────────┘  │         │
│  │  │ mlflow run     │──┼──────────────>│                        │         │
│  │  └────────────────┘  │               │  ┌──────────────────┐  │         │
│  └──────────────────────┘               │  │  MLflow UI       │  │         │
│                                         │  │  http://host:5000│  │         │
│                                         │  │  实验对比/可视化  │  │         │
│                                         │  └──────────────────┘  │         │
│                                         └────────────────────────┘         │
│                                                                             │
│  核心概念:                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐          │
│  │ Experiment │  │    Run     │  │ Parameter  │  │  Metric    │          │
│  │ 实验(项目) │  │ 单次运行   │  │ 超参数     │  │ 评估指标   │          │
│  │ 包含多个Run│  │ 一次训练   │  │ lr, batch  │  │ loss, acc  │          │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                          │
│  │  Artifact  │  │    Tag     │  │   Model    │                          │
│  │ 模型/数据  │  │ 标签分类   │  │ 模型版本   │                          │
│  │ 图表/配置  │  │ 环境/用途  │  │ 注册管理   │                          │
│  └────────────┘  └────────────┘  └────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### MLflow实验跟踪完整示例

```python
"""
MLflow 实验跟踪完整示例
包含: 参数记录、指标跟踪、模型保存、实验对比
"""

import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional


# ============================================================
# 第一部分: 轻量级实验跟踪器 (不依赖MLflow, 可独立运行)
# ============================================================

@dataclass
class ExperimentRun:
    """单次实验运行记录"""
    run_id: str = ""
    experiment_name: str = ""
    start_time: str = ""
    end_time: str = ""
    status: str = "RUNNING"
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)


class ExperimentTracker:
    """
    轻量级实验跟踪器
    模拟MLflow核心功能, 可独立运行用于学习和小型项目
    """

    def __init__(self, tracking_dir: str = "./mlruns"):
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.current_run: Optional[ExperimentRun] = None
        self.experiments: Dict[str, List[ExperimentRun]] = {}

    def create_experiment(self, name: str) -> str:
        """创建新实验"""
        if name not in self.experiments:
            self.experiments[name] = []
            exp_dir = self.tracking_dir / name
            exp_dir.mkdir(parents=True, exist_ok=True)
            print(f"[实验创建] 实验 '{name}' 已创建")
        return name

    def start_run(self, experiment_name: str, run_name: str = None,
                  tags: Dict[str, str] = None) -> ExperimentRun:
        """开始一次实验运行"""
        if experiment_name not in self.experiments:
            self.create_experiment(experiment_name)

        run_id = hashlib.md5(
            f"{experiment_name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        self.current_run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            start_time=datetime.now().isoformat(),
            tags=tags or {},
        )
        if run_name:
            self.current_run.tags["run_name"] = run_name

        print(f"[运行开始] Run ID: {run_id}")
        return self.current_run

    def log_param(self, key: str, value: Any):
        """记录超参数"""
        if self.current_run is None:
            raise RuntimeError("没有活跃的运行, 请先调用 start_run()")
        self.current_run.params[key] = value
        print(f"  [参数] {key} = {value}")

    def log_params(self, params: Dict[str, Any]):
        """批量记录超参数"""
        for key, value in params.items():
            self.log_param(key, value)

    def log_metric(self, key: str, value: float, step: int = None):
        """记录指标值"""
        if self.current_run is None:
            raise RuntimeError("没有活跃的运行, 请先调用 start_run()")
        if key not in self.current_run.metrics:
            self.current_run.metrics[key] = []
        self.current_run.metrics[key].append(value)
        step_info = f" (step={step})" if step is not None else ""
        print(f"  [指标] {key} = {value:.4f}{step_info}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """批量记录指标"""
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_artifact(self, artifact_path: str):
        """记录工件(模型文件、图表等)"""
        if self.current_run is None:
            raise RuntimeError("没有活跃的运行")
        self.current_run.artifacts.append(artifact_path)
        print(f"  [工件] 已记录: {artifact_path}")

    def set_tag(self, key: str, value: str):
        """设置标签"""
        if self.current_run is None:
            raise RuntimeError("没有活跃的运行")
        self.current_run.tags[key] = value

    def end_run(self, status: str = "COMPLETED"):
        """结束当前运行"""
        if self.current_run is None:
            return

        self.current_run.end_time = datetime.now().isoformat()
        self.current_run.status = status

        # 保存运行记录
        exp_name = self.current_run.experiment_name
        self.experiments[exp_name].append(self.current_run)

        run_dir = (self.tracking_dir / exp_name
                   / self.current_run.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        # 保存为JSON
        run_data = asdict(self.current_run)
        with open(run_dir / "run_info.json", "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2, ensure_ascii=False)

        print(f"[运行结束] Run ID: {self.current_run.run_id}, "
              f"状态: {status}")
        self.current_run = None

    def compare_runs(self, experiment_name: str) -> None:
        """对比实验中的所有运行"""
        runs = self.experiments.get(experiment_name, [])
        if not runs:
            print("没有找到运行记录")
            return

        print(f"\n{'=' * 70}")
        print(f"实验对比: {experiment_name}")
        print(f"{'=' * 70}")

        # 收集所有参数和指标名称
        all_params = set()
        all_metrics = set()
        for run in runs:
            all_params.update(run.params.keys())
            all_metrics.update(run.metrics.keys())

        # 打印表头
        run_names = [r.tags.get("run_name", r.run_id[:8]) for r in runs]
        header = f"{'指标/参数':<20}" + "".join(
            f"{name:>15}" for name in run_names
        )
        print(header)
        print("-" * len(header))

        # 打印参数
        print("[超参数]")
        for param in sorted(all_params):
            row = f"  {param:<18}"
            for run in runs:
                val = run.params.get(param, "N/A")
                row += f"{str(val):>15}"
            print(row)

        # 打印指标(取最后一个值)
        print("[指标]")
        for metric in sorted(all_metrics):
            row = f"  {metric:<18}"
            for run in runs:
                vals = run.metrics.get(metric, [])
                val = f"{vals[-1]:.4f}" if vals else "N/A"
                row += f"{val:>15}"
            print(row)

        # 找出最优运行
        if "accuracy" in all_metrics:
            best_run = max(
                runs,
                key=lambda r: r.metrics.get("accuracy", [0])[-1]
            )
            best_name = best_run.tags.get("run_name", best_run.run_id[:8])
            best_acc = best_run.metrics["accuracy"][-1]
            print(f"\n最优运行: {best_name} (accuracy={best_acc:.4f})")


# ============================================================
# 第二部分: 模拟训练过程与实验跟踪
# ============================================================

def simulate_model_training(params: Dict[str, Any]) -> Dict[str, float]:
    """
    模拟模型训练过程
    根据超参数返回模拟的训练指标
    """
    import random
    random.seed(hash(str(params)) % 2**32)

    lr = params.get("learning_rate", 0.001)
    epochs = params.get("epochs", 10)
    batch_size = params.get("batch_size", 32)

    # 模拟训练过程: 不同超参数组合产生不同结果
    base_acc = 0.7 + 0.1 * (1 - abs(lr - 0.001) / 0.01)
    base_acc += 0.05 * (batch_size / 64)
    base_acc = min(base_acc, 0.98)

    metrics_history = []
    for epoch in range(epochs):
        progress = (epoch + 1) / epochs
        noise = random.uniform(-0.02, 0.02)
        train_loss = 2.0 * (1 - progress) + 0.1 + noise
        train_acc = base_acc * progress + noise
        val_loss = train_loss + random.uniform(0, 0.3)
        val_acc = train_acc - random.uniform(0, 0.05)

        metrics_history.append({
            "train_loss": max(0.01, train_loss),
            "train_accuracy": min(0.99, max(0, train_acc)),
            "val_loss": max(0.01, val_loss),
            "val_accuracy": min(0.99, max(0, val_acc)),
        })

    return metrics_history


def run_experiment_tracking_demo():
    """运行实验跟踪演示"""
    tracker = ExperimentTracker("./mlruns_demo")

    # 定义多组超参数进行实验
    param_configs = [
        {
            "run_name": "baseline",
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 5,
            "optimizer": "SGD",
            "model_type": "ResNet18",
        },
        {
            "run_name": "lr_tuned",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 5,
            "optimizer": "Adam",
            "model_type": "ResNet18",
        },
        {
            "run_name": "large_batch",
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 5,
            "optimizer": "Adam",
            "model_type": "ResNet50",
        },
    ]

    experiment_name = "image_classification_v1"
    tracker.create_experiment(experiment_name)

    for config in param_configs:
        run_name = config.pop("run_name")

        # 开始运行
        run = tracker.start_run(
            experiment_name=experiment_name,
            run_name=run_name,
            tags={"model_type": config["model_type"], "env": "dev"},
        )

        # 记录超参数
        tracker.log_params(config)

        # 模拟训练并记录指标
        metrics_history = simulate_model_training(config)
        for step, metrics in enumerate(metrics_history):
            tracker.log_metrics(metrics, step=step)

        # 记录最终指标
        final_metrics = metrics_history[-1]
        tracker.log_metric("accuracy", final_metrics["val_accuracy"])

        # 记录模型工件
        tracker.log_artifact(f"models/{run_name}_model.pth")
        tracker.log_artifact(f"plots/{run_name}_learning_curve.png")

        tracker.end_run()
        config["run_name"] = run_name  # 恢复run_name

    # 对比所有运行
    tracker.compare_runs(experiment_name)


# ============================================================
# 第三部分: MLflow 标准用法 (需要安装 mlflow)
# ============================================================

MLFLOW_EXAMPLE = """
# ---- 以下为MLflow标准用法, 需安装: pip install mlflow ----

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# 设置跟踪服务器 (本地开发可省略, 默认使用 ./mlruns)
# mlflow.set_tracking_uri("http://localhost:5000")

# 创建/选择实验
mlflow.set_experiment("sklearn_classification")

# 生成示例数据
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 使用 autolog 自动记录 (最简方式)
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="random_forest_baseline"):
    # 训练模型
    rf = RandomForestClassifier(n_estimators=100, max_depth=10)
    rf.fit(X_train, y_train)

    # 预测与评估
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 手动记录额外指标
    mlflow.log_metric("custom_accuracy", accuracy)
    mlflow.log_metric("custom_f1", f1)

    # 手动记录额外参数
    mlflow.log_param("data_version", "v1.0")
    mlflow.log_param("feature_count", X.shape[1])

    # 设置标签
    mlflow.set_tag("developer", "team_a")
    mlflow.set_tag("model_purpose", "spam_detection")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

# 查看实验结果: mlflow ui  (在浏览器中打开 http://localhost:5000)
"""


if __name__ == "__main__":
    print("=" * 60)
    print("MLflow 实验跟踪演示")
    print("=" * 60)
    run_experiment_tracking_demo()

    print("\n\n" + "=" * 60)
    print("MLflow 标准用法参考代码:")
    print("=" * 60)
    print(MLFLOW_EXAMPLE)
```

---

## 模型注册与版本管理

### 模型注册流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      模型注册与版本管理流程                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  实验阶段                   注册阶段                   部署阶段              │
│  ┌──────────┐              ┌──────────────┐           ┌──────────┐         │
│  │ 训练运行 │              │ Model        │           │ 生产环境 │         │
│  │ Run #1   │──┐           │ Registry     │           │          │         │
│  └──────────┘  │           │              │           │ Staging  │         │
│  ┌──────────┐  │  注册     │ ┌──────────┐ │  审批     │ ┌──────┐ │         │
│  │ 训练运行 │──┼─────────> │ │ Model v1 │ │─────────> │ │测试  │ │         │
│  │ Run #2   │  │  最优模型 │ │ (Staging)│ │  通过     │ │验证  │ │         │
│  └──────────┘  │           │ └──────────┘ │           │ └──┬───┘ │         │
│  ┌──────────┐  │           │ ┌──────────┐ │           │    │     │         │
│  │ 训练运行 │──┘           │ │ Model v2 │ │           │    ▼     │         │
│  │ Run #3   │              │ │(Productn)│ │  晋升     │ ┌──────┐ │         │
│  └──────────┘              │ └──────────┘ │─────────> │ │生产  │ │         │
│                            │ ┌──────────┐ │           │ │服务  │ │         │
│                            │ │ Model v3 │ │           │ └──────┘ │         │
│                            │ │(Archived)│ │           │          │         │
│                            │ └──────────┘ │           └──────────┘         │
│                            └──────────────┘                                │
│                                                                             │
│  模型生命周期状态:                                                           │
│                                                                             │
│  ┌────────┐   审批    ┌─────────┐   晋升    ┌────────────┐   归档          │
│  │  None  │ ────────> │ Staging │ ────────> │ Production │ ────────>       │
│  │ (新建) │           │ (测试)  │           │  (生产)    │                 │
│  └────────┘           └─────────┘           └────────────┘                 │
│       │                    │                      │          ┌──────────┐  │
│       │                    │                      └────────> │ Archived │  │
│       │                    │                                 │  (归档)  │  │
│       └────────────────────┴────────────────────────────────>└──────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 模型注册代码实现

```python
"""
模型注册与版本管理系统
支持: 模型注册、版本管理、阶段晋升、元数据记录
"""

import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from enum import Enum


class ModelStage(Enum):
    """模型阶段"""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


@dataclass
class ModelVersion:
    """模型版本"""
    version: int
    run_id: str
    model_path: str
    stage: str = ModelStage.NONE.value
    description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class RegisteredModel:
    """注册模型"""
    name: str
    description: str = ""
    versions: List[ModelVersion] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    latest_version: int = 0


class ModelRegistry:
    """
    模型注册中心
    管理模型版本、阶段转换和元数据
    """

    def __init__(self, registry_dir: str = "./model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, RegisteredModel] = {}
        self._load_registry()

    def _load_registry(self):
        """从磁盘加载注册表"""
        index_file = self.registry_dir / "registry_index.json"
        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for name, model_data in data.items():
                    versions = [
                        ModelVersion(**v) for v in model_data.get("versions", [])
                    ]
                    model_data["versions"] = versions
                    self.models[name] = RegisteredModel(**model_data)

    def _save_registry(self):
        """保存注册表到磁盘"""
        data = {}
        for name, model in self.models.items():
            model_dict = asdict(model)
            data[name] = model_dict

        index_file = self.registry_dir / "registry_index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def register_model(self, name: str, description: str = "",
                       tags: Dict[str, str] = None) -> RegisteredModel:
        """注册新模型"""
        if name not in self.models:
            self.models[name] = RegisteredModel(
                name=name,
                description=description,
                tags=tags or {},
                created_at=datetime.now().isoformat(),
            )
            print(f"[注册] 模型 '{name}' 已注册")
        return self.models[name]

    def create_model_version(
        self, model_name: str, run_id: str, model_path: str,
        metrics: Dict[str, float] = None,
        description: str = "",
    ) -> ModelVersion:
        """创建模型新版本"""
        if model_name not in self.models:
            self.register_model(model_name)

        model = self.models[model_name]
        model.latest_version += 1

        version = ModelVersion(
            version=model.latest_version,
            run_id=run_id,
            model_path=model_path,
            description=description,
            metrics=metrics or {},
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        model.versions.append(version)
        self._save_registry()

        print(f"[版本] 模型 '{model_name}' v{version.version} 已创建")
        print(f"  指标: {metrics}")
        return version

    def transition_model_stage(
        self, model_name: str, version: int, stage: str,
        archive_existing: bool = True,
    ):
        """转换模型阶段"""
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"模型 '{model_name}' 不存在")

        target_version = None
        for v in model.versions:
            if v.version == version:
                target_version = v
                break

        if target_version is None:
            raise ValueError(f"版本 {version} 不存在")

        # 如果是晋升到生产, 先归档现有生产版本
        if stage == ModelStage.PRODUCTION.value and archive_existing:
            for v in model.versions:
                if v.stage == ModelStage.PRODUCTION.value:
                    v.stage = ModelStage.ARCHIVED.value
                    v.updated_at = datetime.now().isoformat()
                    print(f"  [归档] v{v.version} 已从 Production 归档")

        old_stage = target_version.stage
        target_version.stage = stage
        target_version.updated_at = datetime.now().isoformat()
        self._save_registry()

        print(f"[阶段转换] '{model_name}' v{version}: "
              f"{old_stage} -> {stage}")

    def get_latest_version(self, model_name: str,
                           stage: str = None) -> Optional[ModelVersion]:
        """获取最新版本"""
        model = self.models.get(model_name)
        if model is None:
            return None

        if stage:
            versions = [v for v in model.versions if v.stage == stage]
        else:
            versions = model.versions

        if not versions:
            return None
        return max(versions, key=lambda v: v.version)

    def list_models(self):
        """列出所有注册的模型"""
        print(f"\n{'=' * 60}")
        print("模型注册中心")
        print(f"{'=' * 60}")
        for name, model in self.models.items():
            print(f"\n模型: {name}")
            print(f"  描述: {model.description}")
            print(f"  版本数: {len(model.versions)}")
            for v in model.versions:
                metrics_str = ", ".join(
                    f"{k}={val:.4f}" for k, val in v.metrics.items()
                )
                print(f"  v{v.version} [{v.stage}] "
                      f"run={v.run_id[:8]} | {metrics_str}")

    def compare_versions(self, model_name: str):
        """对比模型各版本"""
        model = self.models.get(model_name)
        if model is None:
            print(f"模型 '{model_name}' 不存在")
            return

        print(f"\n{'=' * 60}")
        print(f"模型版本对比: {model_name}")
        print(f"{'=' * 60}")

        all_metrics = set()
        for v in model.versions:
            all_metrics.update(v.metrics.keys())

        header = f"{'版本':<8}{'阶段':<15}"
        header += "".join(f"{m:>12}" for m in sorted(all_metrics))
        print(header)
        print("-" * len(header))

        for v in model.versions:
            row = f"v{v.version:<7}{v.stage:<15}"
            for m in sorted(all_metrics):
                val = v.metrics.get(m, 0)
                row += f"{val:>12.4f}"
            print(row)


def demo_model_registry():
    """模型注册演示"""
    registry = ModelRegistry("./registry_demo")

    # 注册模型
    registry.register_model(
        "text_classifier",
        description="文本分类模型 - 新闻分类",
        tags={"team": "nlp", "framework": "pytorch"},
    )

    # 创建多个版本
    registry.create_model_version(
        "text_classifier", run_id="abc123def456",
        model_path="models/v1/model.pth",
        metrics={"accuracy": 0.85, "f1": 0.83, "latency_ms": 45},
        description="Baseline BERT模型",
    )

    registry.create_model_version(
        "text_classifier", run_id="xyz789ghi012",
        model_path="models/v2/model.pth",
        metrics={"accuracy": 0.91, "f1": 0.89, "latency_ms": 42},
        description="增加数据增强和学习率调度",
    )

    registry.create_model_version(
        "text_classifier", run_id="mno345pqr678",
        model_path="models/v3/model.pth",
        metrics={"accuracy": 0.93, "f1": 0.92, "latency_ms": 38},
        description="知识蒸馏优化版本",
    )

    # 阶段转换
    registry.transition_model_stage(
        "text_classifier", version=1, stage=ModelStage.ARCHIVED.value
    )
    registry.transition_model_stage(
        "text_classifier", version=2, stage=ModelStage.PRODUCTION.value
    )
    registry.transition_model_stage(
        "text_classifier", version=3, stage=ModelStage.STAGING.value
    )

    # 将v3晋升到生产 (v2将自动归档)
    print("\n--- 晋升v3到生产 ---")
    registry.transition_model_stage(
        "text_classifier", version=3, stage=ModelStage.PRODUCTION.value
    )

    # 查看所有模型
    registry.list_models()

    # 对比版本
    registry.compare_versions("text_classifier")

    # 获取当前生产版本
    prod_version = registry.get_latest_version(
        "text_classifier", stage=ModelStage.PRODUCTION.value
    )
    if prod_version:
        print(f"\n当前生产版本: v{prod_version.version}")
        print(f"  路径: {prod_version.model_path}")
        print(f"  指标: {prod_version.metrics}")


if __name__ == "__main__":
    demo_model_registry()
```

### MLflow模型注册标准用法

```python
"""
MLflow Model Registry 标准用法参考
需安装: pip install mlflow
"""

MLFLOW_REGISTRY_CODE = """
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 1. 从实验运行注册模型
run_id = "your_run_id_here"
model_uri = f"runs:/{run_id}/model"
result = mlflow.register_model(model_uri, "text_classifier")
print(f"注册版本: {result.version}")

# 2. 转换模型阶段
client.transition_model_version_stage(
    name="text_classifier",
    version=result.version,
    stage="Staging",
    archive_existing_versions=False,
)

# 3. 添加版本描述
client.update_model_version(
    name="text_classifier",
    version=result.version,
    description="增加了数据增强, accuracy提升3%",
)

# 4. 加载特定阶段的模型
model = mlflow.pyfunc.load_model("models:/text_classifier/Production")
predictions = model.predict(test_data)

# 5. 使用标签搜索模型
results = client.search_model_versions("name='text_classifier'")
for mv in results:
    print(f"  v{mv.version} [{mv.current_stage}] run={mv.run_id[:8]}")
"""

print(MLFLOW_REGISTRY_CODE)
```

---

## CI/CD for ML

### ML CI/CD 流水线架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ML CI/CD 流水线架构                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  代码提交触发                                                               │
│  ┌──────┐                                                                  │
│  │ Git  │                                                                  │
│  │ Push │                                                                  │
│  └──┬───┘                                                                  │
│     │                                                                      │
│     ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CI Pipeline (持续集成)                            │   │
│  │                                                                     │   │
│  │  Stage 1          Stage 2          Stage 3          Stage 4         │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │   │
│  │  │ 代码检查 │───>│ 单元测试 │───>│ 数据验证 │───>│ 模型训练 │     │   │
│  │  │          │    │          │    │          │    │          │     │   │
│  │  │ - Lint   │    │ - pytest │    │ - Schema │    │ - Train  │     │   │
│  │  │ - Type   │    │ - 覆盖率 │    │ - 分布   │    │ - 小数据 │     │   │
│  │  │ - Style  │    │ - Mock   │    │ - 完整性 │    │ - 快速版 │     │   │
│  │  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘     │   │
│  │                                                       │           │   │
│  │  Stage 5          Stage 6          Stage 7            │           │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐        │           │   │
│  │  │ 模型评估 │<───│ 完整训练 │<───┘            │       │           │   │
│  │  │          │    │          │    │ 模型注册 │        │           │   │
│  │  │ - 准确率 │    │ - Full   │    │          │        │           │   │
│  │  │ - 延迟   │───>│ - GPU    │───>│ - 版本   │        │           │   │
│  │  │ - 回归   │    │ - 分布式 │    │ - 元数据 │        │           │   │
│  │  └──────────┘    └──────────┘    └──────────┘        │           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│     │                                                                      │
│     ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CD Pipeline (持续部署)                            │   │
│  │                                                                     │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │   │
│  │  │ 构建镜像 │───>│ 部署到   │───>│ 集成测试 │───>│ 金丝雀   │     │   │
│  │  │          │    │ Staging  │    │          │    │ 发布     │     │   │
│  │  │ - Docker │    │          │    │ - API测试│    │          │     │   │
│  │  │ - 依赖   │    │ - K8s    │    │ - E2E    │    │ - 5%流量 │     │   │
│  │  │ - 模型   │    │ - 测试集 │    │ - 性能   │    │ - 监控   │     │   │
│  │  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘     │   │
│  │                                                       │           │   │
│  │                                   ┌──────────┐    ┌───▼──────┐    │   │
│  │                                   │ 回滚策略 │<───│ 全量发布 │    │   │
│  │                                   │          │    │          │    │   │
│  │                                   │ - 自动   │    │ - 100%   │    │   │
│  │                                   │ - 手动   │    │ - 流量   │    │   │
│  │                                   └──────────┘    └──────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### GitHub Actions ML Pipeline

```python
"""
ML CI/CD Pipeline 配置与实现
包含: GitHub Actions配置生成、模型测试框架、部署脚本
"""

import json
from typing import Dict, List, Any
from dataclasses import dataclass


# ============================================================
# 第一部分: GitHub Actions工作流生成器
# ============================================================

class MLPipelineGenerator:
    """生成ML CI/CD Pipeline的YAML配置"""

    @staticmethod
    def generate_ci_workflow() -> str:
        """生成CI工作流配置"""
        workflow = """
# .github/workflows/ml-ci.yml
# ML持续集成工作流

name: ML CI Pipeline

on:
  push:
    branches: [main, develop]
    paths:
      - 'src/**'
      - 'models/**'
      - 'tests/**'
      - 'data/schemas/**'
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: '3.10'
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}

jobs:
  # ---- 阶段1: 代码质量检查 ----
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: |
          pip install ruff mypy pytest pytest-cov
          pip install -r requirements.txt
      - name: Lint with ruff
        run: ruff check src/ --output-format=github
      - name: Type check with mypy
        run: mypy src/ --ignore-missing-imports

  # ---- 阶段2: 单元测试 ----
  unit-tests:
    needs: code-quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  # ---- 阶段3: 数据验证 ----
  data-validation:
    needs: code-quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Validate data schema
        run: python scripts/validate_data.py
      - name: Check data distribution
        run: python scripts/check_distribution.py

  # ---- 阶段4: 快速训练测试 ----
  quick-train:
    needs: [unit-tests, data-validation]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Quick training test (small data)
        run: |
          python src/train.py \\
            --data-sample 0.01 \\
            --epochs 2 \\
            --quick-test
      - name: Model smoke test
        run: python tests/smoke/test_model_inference.py

  # ---- 阶段5: 完整训练 (仅main分支) ----
  full-training:
    if: github.ref == 'refs/heads/main'
    needs: quick-train
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - name: Full model training
        run: |
          python src/train.py \\
            --config configs/production.yaml \\
            --experiment-name "ci_training"
      - name: Evaluate model
        run: python src/evaluate.py --threshold 0.90
      - name: Register model
        run: python src/register_model.py
"""
        return workflow.strip()

    @staticmethod
    def generate_cd_workflow() -> str:
        """生成CD工作流配置"""
        workflow = """
# .github/workflows/ml-cd.yml
# ML持续部署工作流

name: ML CD Pipeline

on:
  workflow_run:
    workflows: ["ML CI Pipeline"]
    types: [completed]
    branches: [main]

jobs:
  deploy-staging:
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: |
          docker build -t ml-model:${{ github.sha }} .
          docker push registry.example.com/ml-model:${{ github.sha }}
      - name: Deploy to staging
        run: |
          kubectl set image deployment/ml-model \\
            ml-model=registry.example.com/ml-model:${{ github.sha }} \\
            --namespace=staging
      - name: Run integration tests
        run: python tests/integration/test_api.py --env staging
      - name: Run performance tests
        run: python tests/performance/test_latency.py --env staging

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4
      - name: Canary deployment (5%)
        run: |
          python scripts/canary_deploy.py \\
            --image registry.example.com/ml-model:${{ github.sha }} \\
            --percentage 5
      - name: Monitor canary (10min)
        run: python scripts/monitor_canary.py --duration 600
      - name: Full rollout
        run: |
          kubectl set image deployment/ml-model \\
            ml-model=registry.example.com/ml-model:${{ github.sha }} \\
            --namespace=production
"""
        return workflow.strip()


# ============================================================
# 第二部分: ML测试框架
# ============================================================

@dataclass
class ModelTestResult:
    """模型测试结果"""
    test_name: str
    passed: bool
    metric_name: str
    actual_value: float
    threshold: float
    message: str = ""


class MLTestFramework:
    """
    ML模型测试框架
    包含: 数据验证、模型性能、推理延迟、回归测试
    """

    def __init__(self):
        self.results: List[ModelTestResult] = []

    def test_data_schema(self, data: List[Dict],
                         required_fields: List[str]) -> ModelTestResult:
        """测试数据Schema是否符合预期"""
        missing = []
        for i, row in enumerate(data[:100]):
            for field_name in required_fields:
                if field_name not in row:
                    missing.append(f"行{i}缺少字段'{field_name}'")

        passed = len(missing) == 0
        result = ModelTestResult(
            test_name="数据Schema验证",
            passed=passed,
            metric_name="missing_fields",
            actual_value=len(missing),
            threshold=0,
            message=f"缺失字段数: {len(missing)}" + (
                f", 示例: {missing[:3]}" if missing else ""
            ),
        )
        self.results.append(result)
        return result

    def test_model_accuracy(self, y_true: List, y_pred: List,
                            threshold: float = 0.85) -> ModelTestResult:
        """测试模型准确率是否达标"""
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / len(y_true) if y_true else 0

        result = ModelTestResult(
            test_name="模型准确率测试",
            passed=accuracy >= threshold,
            metric_name="accuracy",
            actual_value=accuracy,
            threshold=threshold,
            message=f"准确率: {accuracy:.4f}, 阈值: {threshold}",
        )
        self.results.append(result)
        return result

    def test_inference_latency(self, latencies_ms: List[float],
                               p95_threshold: float = 100.0
                               ) -> ModelTestResult:
        """测试推理延迟P95是否达标"""
        sorted_lat = sorted(latencies_ms)
        p95_idx = int(len(sorted_lat) * 0.95)
        p95 = sorted_lat[p95_idx] if sorted_lat else 0

        result = ModelTestResult(
            test_name="推理延迟P95测试",
            passed=p95 <= p95_threshold,
            metric_name="latency_p95_ms",
            actual_value=p95,
            threshold=p95_threshold,
            message=f"P95延迟: {p95:.1f}ms, 阈值: {p95_threshold}ms",
        )
        self.results.append(result)
        return result

    def test_model_regression(
        self, current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        tolerance: float = 0.02,
    ) -> ModelTestResult:
        """回归测试: 新模型不应比基线差太多"""
        regressions = []
        for metric, baseline_val in baseline_metrics.items():
            current_val = current_metrics.get(metric, 0)
            if current_val < baseline_val - tolerance:
                regressions.append(
                    f"{metric}: {current_val:.4f} < {baseline_val:.4f}"
                )

        passed = len(regressions) == 0
        result = ModelTestResult(
            test_name="模型回归测试",
            passed=passed,
            metric_name="regression_count",
            actual_value=len(regressions),
            threshold=0,
            message=(
                "无回归" if passed
                else f"回归项: {', '.join(regressions)}"
            ),
        )
        self.results.append(result)
        return result

    def generate_report(self) -> str:
        """生成测试报告"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        report = []
        report.append("=" * 60)
        report.append("ML模型测试报告")
        report.append("=" * 60)
        report.append(f"总计: {total} | 通过: {passed} | 失败: {failed}")
        report.append("-" * 60)

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            icon = "[+]" if r.passed else "[-]"
            report.append(f"{icon} [{status}] {r.test_name}")
            report.append(f"      {r.message}")

        report.append("=" * 60)
        all_passed = failed == 0
        report.append(f"结果: {'ALL PASSED' if all_passed else 'HAS FAILURES'}")
        return "\n".join(report)


def demo_cicd_pipeline():
    """CI/CD Pipeline 演示"""

    # 1. 生成Pipeline配置
    generator = MLPipelineGenerator()
    print("=" * 60)
    print("GitHub Actions CI 工作流配置:")
    print("=" * 60)
    print(generator.generate_ci_workflow()[:500] + "\n... (截断)")

    # 2. 运行ML测试
    print("\n\n" + "=" * 60)
    print("ML测试框架演示")
    print("=" * 60)

    tester = MLTestFramework()

    # 数据Schema测试
    sample_data = [
        {"text": "示例1", "label": 1, "id": 1},
        {"text": "示例2", "label": 0, "id": 2},
        {"text": "示例3", "id": 3},  # 缺少label
    ]
    tester.test_data_schema(sample_data, ["text", "label", "id"])

    # 模型准确率测试
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 1]
    tester.test_model_accuracy(y_true, y_pred, threshold=0.75)

    # 推理延迟测试
    import random
    random.seed(42)
    latencies = [random.uniform(10, 80) for _ in range(100)]
    tester.test_inference_latency(latencies, p95_threshold=100.0)

    # 回归测试
    baseline = {"accuracy": 0.88, "f1": 0.86}
    current = {"accuracy": 0.91, "f1": 0.85}
    tester.test_model_regression(current, baseline, tolerance=0.02)

    # 生成报告
    print(tester.generate_report())


if __name__ == "__main__":
    demo_cicd_pipeline()
```

---

## 数据版本控制

### DVC数据版本控制架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      数据版本控制 (DVC) 架构                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  工作目录                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  project/                                                            │  │
│  │  ├── .git/              <-- Git: 代码版本控制                        │  │
│  │  ├── .dvc/              <-- DVC: 配置与缓存                          │  │
│  │  │   ├── config         <-- 远程存储配置                              │  │
│  │  │   └── cache/         <-- 本地数据缓存                              │  │
│  │  ├── data/                                                           │  │
│  │  │   ├── raw/           <-- 原始数据 (DVC跟踪)                       │  │
│  │  │   ├── processed/     <-- 处理后数据 (DVC跟踪)                     │  │
│  │  │   └── features/      <-- 特征数据 (DVC跟踪)                       │  │
│  │  ├── data.dvc           <-- DVC元文件 (Git跟踪, 记录数据哈希)        │  │
│  │  ├── models/                                                         │  │
│  │  │   └── model.pkl      <-- 模型文件 (DVC跟踪)                       │  │
│  │  ├── dvc.yaml           <-- Pipeline定义                              │  │
│  │  ├── dvc.lock           <-- Pipeline锁文件 (依赖哈希)                 │  │
│  │  └── src/               <-- 代码 (Git跟踪)                           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Git vs DVC 分工:                                                           │
│  ┌────────────────────────┐    ┌────────────────────────┐                  │
│  │       Git 跟踪         │    │       DVC 跟踪         │                  │
│  │                        │    │                        │                  │
│  │  - 源代码 (.py)       │    │  - 大数据集            │                  │
│  │  - 配置文件 (.yaml)   │    │  - 模型文件            │                  │
│  │  - DVC元文件 (.dvc)   │    │  - 特征文件            │                  │
│  │  - Pipeline定义        │    │  - 中间产物            │                  │
│  │  - 小文件/文档         │    │  - 大型二进制文件      │                  │
│  └────────────────────────┘    └────────────────────────┘                  │
│         │                               │                                  │
│         ▼                               ▼                                  │
│  ┌────────────────┐            ┌────────────────────────┐                  │
│  │ GitHub/GitLab  │            │ 远程存储               │                  │
│  │ 代码仓库       │            │ S3/GCS/Azure/NAS       │                  │
│  └────────────────┘            └────────────────────────┘                  │
│                                                                             │
│  DVC Pipeline 执行流程:                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐              │
│  │ 数据获取 │──>│ 数据预处 │──>│ 特征工程 │──>│ 模型训练 │              │
│  │ dvc pull │   │ 理清洗   │   │ 特征提取 │   │ 评估验证 │              │
│  │          │   │          │   │          │   │          │              │
│  │ deps:    │   │ deps:    │   │ deps:    │   │ deps:    │              │
│  │  remote  │   │  raw/    │   │  clean/  │   │  feats/  │              │
│  │ outs:    │   │ outs:    │   │ outs:    │   │ outs:    │              │
│  │  raw/    │   │  clean/  │   │  feats/  │   │  model/  │              │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 数据版本控制代码实现

```python
"""
数据版本控制系统 (轻量级DVC模拟)
支持: 数据哈希、版本快照、差异比对
"""

import hashlib
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class DataVersionControl:
    """
    轻量级数据版本控制
    模拟DVC核心功能: 数据哈希、版本快照、差异检测
    """

    def __init__(self, project_dir: str = "./dvc_project"):
        self.project_dir = Path(project_dir)
        self.cache_dir = self.project_dir / ".dvc_cache"
        self.versions_file = self.project_dir / ".dvc_versions.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.versions: Dict[str, List[Dict]] = self._load_versions()

    def _load_versions(self) -> Dict:
        if self.versions_file.exists():
            with open(self.versions_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_versions(self):
        with open(self.versions_file, "w", encoding="utf-8") as f:
            json.dump(self.versions, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _compute_hash(content: str) -> str:
        """计算内容的MD5哈希"""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def track(self, name: str, data: str, message: str = "") -> Dict:
        """
        跟踪数据版本
        name: 数据集名称
        data: 数据内容(实际项目中为文件路径)
        message: 版本说明
        """
        data_hash = self._compute_hash(data)

        # 检查是否有变化
        if name in self.versions:
            last = self.versions[name][-1]
            if last["hash"] == data_hash:
                print(f"[DVC] '{name}' 未检测到变化, 跳过")
                return last

        # 保存到缓存
        cache_path = self.cache_dir / data_hash
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(data)

        # 记录版本
        if name not in self.versions:
            self.versions[name] = []

        version_info = {
            "version": len(self.versions[name]) + 1,
            "hash": data_hash,
            "size": len(data),
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        self.versions[name].append(version_info)
        self._save_versions()

        print(f"[DVC] '{name}' v{version_info['version']} "
              f"已保存 (hash={data_hash[:8]}..., size={len(data)})")
        return version_info

    def checkout(self, name: str, version: int = None) -> Optional[str]:
        """检出特定版本的数据"""
        if name not in self.versions:
            print(f"[DVC] 数据集 '{name}' 不存在")
            return None

        versions = self.versions[name]
        if version is None:
            target = versions[-1]
        else:
            target = next((v for v in versions if v["version"] == version), None)
            if target is None:
                print(f"[DVC] 版本 {version} 不存在")
                return None

        cache_path = self.cache_dir / target["hash"]
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                data = f.read()
            print(f"[DVC] 检出 '{name}' v{target['version']}")
            return data
        print(f"[DVC] 缓存丢失: {target['hash']}")
        return None

    def diff(self, name: str, v1: int, v2: int) -> Dict:
        """比较两个版本的差异"""
        data1 = self.checkout(name, v1)
        data2 = self.checkout(name, v2)

        if data1 is None or data2 is None:
            return {"error": "版本不存在"}

        lines1 = set(data1.strip().split("\n"))
        lines2 = set(data2.strip().split("\n"))

        added = lines2 - lines1
        removed = lines1 - lines2
        unchanged = lines1 & lines2

        return {
            "v1": v1, "v2": v2,
            "added": len(added),
            "removed": len(removed),
            "unchanged": len(unchanged),
            "added_samples": list(added)[:3],
            "removed_samples": list(removed)[:3],
        }

    def log(self, name: str):
        """显示数据版本历史"""
        if name not in self.versions:
            print(f"数据集 '{name}' 不存在")
            return

        print(f"\n版本历史: {name}")
        print("-" * 50)
        for v in self.versions[name]:
            print(f"  v{v['version']} | {v['hash'][:12]}... | "
                  f"size={v['size']:>6} | {v['message']}")


def demo_data_versioning():
    """数据版本控制演示"""
    dvc = DataVersionControl("./dvc_demo")

    # 模拟数据变更
    v1_data = "id,text,label\n1,好评,1\n2,差评,0\n3,一般,1"
    dvc.track("train_data", v1_data, "初始训练数据")

    v2_data = v1_data + "\n4,非常好,1\n5,很差,0\n6,还行,1"
    dvc.track("train_data", v2_data, "新增3条标注数据")

    v3_data = v2_data.replace("3,一般,1", "3,一般,0")  # 修正标注
    dvc.track("train_data", v3_data, "修正第3条标注")

    # 查看版本历史
    dvc.log("train_data")

    # 检出特定版本
    old_data = dvc.checkout("train_data", version=1)
    print(f"\nv1数据:\n{old_data}")

    # 版本差异
    diff = dvc.diff("train_data", 1, 3)
    print(f"\nv1 vs v3 差异:")
    print(f"  新增行: {diff['added']}")
    print(f"  删除行: {diff['removed']}")
    print(f"  不变行: {diff['unchanged']}")


if __name__ == "__main__":
    demo_data_versioning()
```

---

## Feature Store

### Feature Store架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Feature Store 架构                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  数据源                    特征工程                    Feature Store         │
│  ┌──────────┐             ┌──────────┐              ┌──────────────────┐   │
│  │ 数据库   │────────────>│ 批量特征 │────────────> │  离线存储        │   │
│  │ 数据湖   │             │ 计算     │              │  (历史特征)      │   │
│  │ CSV/JSON │             │ (Spark)  │              │  Parquet/Hive    │   │
│  └──────────┘             └──────────┘              └────────┬─────────┘   │
│                                                              │             │
│  ┌──────────┐             ┌──────────┐              ┌────────▼─────────┐   │
│  │ 事件流   │────────────>│ 实时特征 │────────────> │  在线存储        │   │
│  │ Kafka    │             │ 计算     │              │  (实时服务)      │   │
│  │ 日志     │             │ (Flink)  │              │  Redis/DynamoDB  │   │
│  └──────────┘             └──────────┘              └────────┬─────────┘   │
│                                                              │             │
│                                                              ▼             │
│  消费者:                                                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  模型训练    │    │  模型推理    │    │  数据分析    │                  │
│  │  (离线特征)  │    │  (在线特征)  │    │  (特征探索)  │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Feature Store代码实现

```python
"""
轻量级 Feature Store 实现
支持: 特征注册、存储、版本管理、在线/离线读取
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class FeatureDefinition:
    """特征定义"""
    name: str
    dtype: str
    description: str = ""
    source: str = ""
    transform: str = ""
    version: int = 1


@dataclass
class FeatureGroup:
    """特征组"""
    name: str
    entity_key: str  # 实体主键, 如 user_id
    features: List[FeatureDefinition] = field(default_factory=list)
    description: str = ""
    created_at: str = ""
    ttl_seconds: int = 3600  # 在线存储TTL


class SimpleFeatureStore:
    """
    简易Feature Store
    同时支持离线(字典存储)和在线(带TTL的缓存)
    """

    def __init__(self):
        self.feature_groups: Dict[str, FeatureGroup] = {}
        self.offline_store: Dict[str, List[Dict]] = {}  # 离线存储
        self.online_store: Dict[str, Dict] = {}          # 在线存储
        self.online_timestamps: Dict[str, float] = {}    # TTL时间戳

    def register_feature_group(self, group: FeatureGroup):
        """注册特征组"""
        group.created_at = datetime.now().isoformat()
        self.feature_groups[group.name] = group
        self.offline_store[group.name] = []
        print(f"[FeatureStore] 注册特征组: {group.name}")
        for f in group.features:
            print(f"  - {f.name} ({f.dtype}): {f.description}")

    def ingest_batch(self, group_name: str, records: List[Dict]):
        """批量写入离线特征"""
        if group_name not in self.feature_groups:
            raise ValueError(f"特征组 '{group_name}' 不存在")

        self.offline_store[group_name].extend(records)

        # 同步到在线存储
        group = self.feature_groups[group_name]
        for record in records:
            entity_val = record.get(group.entity_key)
            if entity_val is not None:
                key = f"{group_name}:{entity_val}"
                self.online_store[key] = record
                self.online_timestamps[key] = time.time()

        print(f"[FeatureStore] 写入 {len(records)} 条到 '{group_name}'")

    def get_online_features(self, group_name: str,
                            entity_values: List[Any]) -> List[Dict]:
        """在线特征查询(低延迟)"""
        group = self.feature_groups.get(group_name)
        if group is None:
            return []

        results = []
        for val in entity_values:
            key = f"{group_name}:{val}"
            record = self.online_store.get(key)

            if record is not None:
                # 检查TTL
                ts = self.online_timestamps.get(key, 0)
                if time.time() - ts > group.ttl_seconds:
                    record = None  # 已过期

            results.append(record or {group.entity_key: val})
        return results

    def get_offline_features(self, group_name: str,
                             entity_values: List[Any] = None
                             ) -> List[Dict]:
        """离线特征查询(用于训练)"""
        records = self.offline_store.get(group_name, [])
        if entity_values is None:
            return records

        group = self.feature_groups[group_name]
        return [
            r for r in records
            if r.get(group.entity_key) in entity_values
        ]

    def list_feature_groups(self):
        """列出所有特征组"""
        print(f"\n{'=' * 50}")
        print("Feature Store 特征组列表")
        print(f"{'=' * 50}")
        for name, group in self.feature_groups.items():
            n_records = len(self.offline_store.get(name, []))
            print(f"\n  {name} ({n_records} records)")
            print(f"    实体键: {group.entity_key}")
            for f in group.features:
                print(f"    - {f.name}: {f.description}")


def demo_feature_store():
    """Feature Store 演示"""
    store = SimpleFeatureStore()

    # 注册用户特征组
    user_features = FeatureGroup(
        name="user_features",
        entity_key="user_id",
        description="用户画像特征",
        features=[
            FeatureDefinition("age", "int", "用户年龄"),
            FeatureDefinition("total_orders", "int", "历史订单数"),
            FeatureDefinition("avg_order_amount", "float", "平均订单金额"),
            FeatureDefinition("last_login_days", "int", "距上次登录天数"),
        ],
        ttl_seconds=3600,
    )
    store.register_feature_group(user_features)

    # 批量写入特征
    user_records = [
        {"user_id": "u001", "age": 28, "total_orders": 15,
         "avg_order_amount": 120.5, "last_login_days": 1},
        {"user_id": "u002", "age": 35, "total_orders": 42,
         "avg_order_amount": 230.0, "last_login_days": 0},
        {"user_id": "u003", "age": 22, "total_orders": 3,
         "avg_order_amount": 55.0, "last_login_days": 7},
    ]
    store.ingest_batch("user_features", user_records)

    # 在线查询(推理时使用)
    print("\n在线特征查询:")
    results = store.get_online_features(
        "user_features", ["u001", "u002", "u999"]
    )
    for r in results:
        print(f"  {r}")

    # 离线查询(训练时使用)
    print("\n离线特征查询:")
    offline = store.get_offline_features("user_features")
    for r in offline:
        print(f"  {r}")

    store.list_feature_groups()


if __name__ == "__main__":
    demo_feature_store()
```

---

## 完整ML Pipeline

### 端到端Pipeline架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    完整ML Pipeline 端到端架构                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Pipeline Orchestrator                        │   │
│  │                   (Airflow / Kubeflow / Prefect)                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │          │           │           │           │          │          │
│       ▼          ▼           ▼           ▼           ▼          ▼          │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐     │
│  │Step 1  │ │Step 2  │ │Step 3  │ │Step 4  │ │Step 5  │ │Step 6  │     │
│  │数据采集│>│数据验证│>│特征工程│>│模型训练│>│模型评估│>│模型部署│     │
│  │        │ │        │ │        │ │        │ │        │ │        │     │
│  │- API   │ │- Schema│ │- 清洗  │ │- 超参  │ │- 准确率│ │- 打包  │     │
│  │- DB    │ │- 分布  │ │- 编码  │ │- 训练  │ │- 延迟  │ │- 部署  │     │
│  │- 文件  │ │- 质量  │ │- 缩放  │ │- 验证  │ │- 回归  │ │- 监控  │     │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘     │
│       │          │           │           │           │          │          │
│       ▼          ▼           ▼           ▼           ▼          ▼          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       共享基础设施层                                  │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │   │
│  │  │实验跟踪  │ │模型注册  │ │Feature   │ │数据版本  │              │   │
│  │  │ MLflow   │ │ Registry │ │ Store    │ │ DVC     │              │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘              │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │   │
│  │  │监控告警  │ │日志收集  │ │配置管理  │ │密钥管理  │              │   │
│  │  │Prometheus│ │  ELK     │ │  Consul  │ │  Vault   │              │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Pipeline编排代码实现

```python
"""
完整ML Pipeline编排系统
支持: 步骤定义、依赖管理、执行调度、状态跟踪、错误处理
"""

import time
import traceback
from datetime import datetime
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStep:
    """Pipeline步骤定义"""
    name: str
    func: Callable
    depends_on: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: str = ""
    start_time: float = 0
    end_time: float = 0
    retry_count: int = 0
    max_retries: int = 2


class MLPipeline:
    """
    ML Pipeline编排器
    支持: DAG依赖管理、重试机制、状态追踪
    """

    def __init__(self, name: str):
        self.name = name
        self.steps: Dict[str, PipelineStep] = {}
        self.context: Dict[str, Any] = {}  # 步骤间共享数据
        self.start_time = 0
        self.end_time = 0

    def add_step(self, name: str, func: Callable,
                 depends_on: List[str] = None,
                 params: Dict[str, Any] = None,
                 max_retries: int = 2) -> "MLPipeline":
        """添加Pipeline步骤"""
        self.steps[name] = PipelineStep(
            name=name,
            func=func,
            depends_on=depends_on or [],
            params=params or {},
            max_retries=max_retries,
        )
        return self

    def _can_run(self, step: PipelineStep) -> bool:
        """检查步骤依赖是否满足"""
        for dep in step.depends_on:
            if dep not in self.steps:
                return False
            if self.steps[dep].status != StepStatus.COMPLETED:
                return False
        return True

    def _run_step(self, step: PipelineStep):
        """执行单个步骤(带重试)"""
        step.status = StepStatus.RUNNING
        step.start_time = time.time()
        print(f"\n{'='*50}")
        print(f"[Step] 正在执行: {step.name}")
        print(f"{'='*50}")

        while step.retry_count <= step.max_retries:
            try:
                result = step.func(self.context, **step.params)
                step.result = result
                step.status = StepStatus.COMPLETED
                step.end_time = time.time()
                elapsed = step.end_time - step.start_time
                print(f"[Step] {step.name} 完成 "
                      f"(耗时: {elapsed:.2f}s)")
                return
            except Exception as e:
                step.retry_count += 1
                step.error = str(e)
                if step.retry_count <= step.max_retries:
                    print(f"[Step] {step.name} 失败, "
                          f"第{step.retry_count}次重试... "
                          f"错误: {e}")
                    time.sleep(0.5)

        step.status = StepStatus.FAILED
        step.end_time = time.time()
        print(f"[Step] {step.name} 最终失败: {step.error}")

    def run(self) -> bool:
        """执行Pipeline"""
        self.start_time = time.time()
        print(f"\n{'#'*60}")
        print(f"# Pipeline: {self.name}")
        print(f"# 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"# 步骤数: {len(self.steps)}")
        print(f"{'#'*60}")

        completed = set()
        failed = False

        while len(completed) < len(self.steps) and not failed:
            progress = False
            for name, step in self.steps.items():
                if step.status != StepStatus.PENDING:
                    continue
                if self._can_run(step):
                    self._run_step(step)
                    completed.add(name)
                    progress = True
                    if step.status == StepStatus.FAILED:
                        # 跳过依赖此步骤的后续步骤
                        for s in self.steps.values():
                            if name in s.depends_on:
                                s.status = StepStatus.SKIPPED
                                completed.add(s.name)
                        failed = True
                        break

            if not progress and not failed:
                print("[Pipeline] 死锁检测: 无法继续执行")
                break

        self.end_time = time.time()
        self._print_summary()
        return not failed

    def _print_summary(self):
        """打印Pipeline执行总结"""
        total_time = self.end_time - self.start_time
        print(f"\n{'#'*60}")
        print(f"# Pipeline 执行总结: {self.name}")
        print(f"{'#'*60}")
        print(f"总耗时: {total_time:.2f}s")
        print(f"\n{'步骤':<20} {'状态':<12} {'耗时':<10}")
        print("-" * 42)
        for step in self.steps.values():
            elapsed = step.end_time - step.start_time if step.end_time else 0
            status_str = step.status.value.upper()
            print(f"{step.name:<20} {status_str:<12} {elapsed:.2f}s")
            if step.error:
                print(f"  错误: {step.error}")

        statuses = [s.status for s in self.steps.values()]
        if all(s == StepStatus.COMPLETED for s in statuses):
            print(f"\n结果: ALL STEPS PASSED")
        else:
            failed = sum(1 for s in statuses if s == StepStatus.FAILED)
            skipped = sum(1 for s in statuses if s == StepStatus.SKIPPED)
            print(f"\n结果: {failed} FAILED, {skipped} SKIPPED")


# ============================================================
# Pipeline步骤实现
# ============================================================

def step_data_ingestion(context: Dict, source: str = "csv") -> Dict:
    """步骤1: 数据采集"""
    print(f"  从 {source} 采集数据...")
    data = [
        {"id": 1, "text": "这个产品很好", "label": 1},
        {"id": 2, "text": "质量太差了", "label": 0},
        {"id": 3, "text": "还可以吧", "label": 1},
        {"id": 4, "text": "非常满意", "label": 1},
        {"id": 5, "text": "不推荐购买", "label": 0},
    ]
    context["raw_data"] = data
    print(f"  采集到 {len(data)} 条数据")
    return {"count": len(data)}


def step_data_validation(context: Dict, min_samples: int = 3) -> Dict:
    """步骤2: 数据验证"""
    data = context.get("raw_data", [])
    issues = []

    if len(data) < min_samples:
        issues.append(f"数据量不足: {len(data)} < {min_samples}")

    # 检查必要字段
    required = {"id", "text", "label"}
    for i, row in enumerate(data):
        missing = required - set(row.keys())
        if missing:
            issues.append(f"行{i}缺少字段: {missing}")

    # 检查标签分布
    labels = [r.get("label") for r in data if "label" in r]
    if labels:
        pos_ratio = sum(labels) / len(labels)
        print(f"  标签分布: 正类={pos_ratio:.1%}, 负类={1-pos_ratio:.1%}")
        if pos_ratio > 0.9 or pos_ratio < 0.1:
            issues.append(f"标签分布严重不均衡: {pos_ratio:.1%}")

    context["validation_issues"] = issues
    if issues:
        print(f"  警告: {len(issues)} 个问题")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  数据验证通过")
    return {"issues": len(issues)}


def step_feature_engineering(context: Dict) -> Dict:
    """步骤3: 特征工程"""
    data = context.get("raw_data", [])
    features = []
    for row in data:
        text = row.get("text", "")
        feat = {
            "id": row["id"],
            "text_length": len(text),
            "word_count": len(text.split()),
            "has_positive_word": any(
                w in text for w in ["好", "满意", "推荐"]
            ),
            "has_negative_word": any(
                w in text for w in ["差", "不"]
            ),
            "label": row.get("label", 0),
        }
        features.append(feat)

    context["features"] = features
    print(f"  生成 {len(features)} 条特征, "
          f"特征维度: {len(features[0]) - 2}")
    return {"feature_count": len(features)}


def step_model_training(context: Dict, model_type: str = "simple") -> Dict:
    """步骤4: 模型训练"""
    features = context.get("features", [])
    print(f"  使用 {model_type} 模型训练...")
    print(f"  训练数据: {len(features)} 条")

    # 简单规则模型(演示)
    def predict(feat):
        score = 0
        if feat["has_positive_word"]:
            score += 1
        if feat["has_negative_word"]:
            score -= 1
        return 1 if score >= 0 else 0

    # 评估
    correct = sum(
        1 for f in features if predict(f) == f["label"]
    )
    accuracy = correct / len(features) if features else 0

    context["model"] = predict
    context["train_accuracy"] = accuracy
    print(f"  训练准确率: {accuracy:.2%}")
    return {"accuracy": accuracy}


def step_model_evaluation(context: Dict,
                          accuracy_threshold: float = 0.6) -> Dict:
    """步骤5: 模型评估"""
    accuracy = context.get("train_accuracy", 0)
    print(f"  模型准确率: {accuracy:.2%}")
    print(f"  准确率阈值: {accuracy_threshold:.2%}")

    passed = accuracy >= accuracy_threshold
    if not passed:
        raise ValueError(
            f"模型准确率 {accuracy:.2%} 低于阈值 {accuracy_threshold:.2%}"
        )

    print("  模型评估通过")
    return {"passed": True, "accuracy": accuracy}


def step_model_deployment(context: Dict,
                          target: str = "staging") -> Dict:
    """步骤6: 模型部署"""
    print(f"  部署目标: {target}")
    print(f"  模型准确率: {context.get('train_accuracy', 0):.2%}")
    print(f"  模拟部署完成")

    deploy_info = {
        "target": target,
        "timestamp": datetime.now().isoformat(),
        "accuracy": context.get("train_accuracy", 0),
    }
    context["deployment"] = deploy_info
    return deploy_info


def demo_ml_pipeline():
    """完整ML Pipeline演示"""
    pipeline = MLPipeline("text_classification_pipeline")

    # 构建DAG
    pipeline.add_step(
        "data_ingestion", step_data_ingestion,
        params={"source": "database"},
    )
    pipeline.add_step(
        "data_validation", step_data_validation,
        depends_on=["data_ingestion"],
        params={"min_samples": 3},
    )
    pipeline.add_step(
        "feature_engineering", step_feature_engineering,
        depends_on=["data_validation"],
    )
    pipeline.add_step(
        "model_training", step_model_training,
        depends_on=["feature_engineering"],
        params={"model_type": "rule_based"},
    )
    pipeline.add_step(
        "model_evaluation", step_model_evaluation,
        depends_on=["model_training"],
        params={"accuracy_threshold": 0.6},
    )
    pipeline.add_step(
        "model_deployment", step_model_deployment,
        depends_on=["model_evaluation"],
        params={"target": "staging"},
    )

    # 执行
    success = pipeline.run()
    return success


if __name__ == "__main__":
    demo_ml_pipeline()
```

---

## 生产环境最佳实践

### 最佳实践总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MLOps 生产环境最佳实践                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 可复现性                           2. 自动化                            │
│  ┌──────────────────────────────┐     ┌──────────────────────────────┐     │
│  │ - 固定随机种子               │     │ - Pipeline全流程自动化       │     │
│  │ - 锁定依赖版本               │     │ - 数据漂移自动检测           │     │
│  │   (pip freeze/poetry.lock)   │     │ - 模型再训练自动触发         │     │
│  │ - 数据版本控制 (DVC)         │     │ - 部署金丝雀自动化           │     │
│  │ - Docker环境一致性           │     │ - 回滚策略自动执行           │     │
│  │ - 实验参数完整记录           │     │ - 告警自动通知               │     │
│  └──────────────────────────────┘     └──────────────────────────────┘     │
│                                                                             │
│  3. 测试策略                           4. 监控体系                          │
│  ┌──────────────────────────────┐     ┌──────────────────────────────┐     │
│  │ - 数据Schema测试             │     │ - 模型准确率/延迟实时监控    │     │
│  │ - 数据分布测试               │     │ - 数据漂移检测               │     │
│  │ - 模型性能回归测试           │     │ - 资源使用率 (GPU/内存)      │     │
│  │ - 推理延迟压力测试           │     │ - 业务指标关联分析           │     │
│  │ - A/B测试统计显著性          │     │ - 定期模型审计               │     │
│  │ - 集成测试 (API端到端)       │     │ - 公平性与偏见检测           │     │
│  └──────────────────────────────┘     └──────────────────────────────┘     │
│                                                                             │
│  5. 安全与治理                         6. 团队协作                          │
│  ┌──────────────────────────────┐     ┌──────────────────────────────┐     │
│  │ - API密钥安全管理            │     │ - 模型审批流程               │     │
│  │ - 数据脱敏处理               │     │ - 文档化 (Model Cards)       │     │
│  │ - 访问控制 (RBAC)            │     │ - 代码Review流程             │     │
│  │ - 审计日志                   │     │ - 知识共享与培训             │     │
│  │ - 合规检查                   │     │ - 故障复盘与改进             │     │
│  └──────────────────────────────┘     └──────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 生产环境配置模板

```python
"""
MLOps生产环境配置与检查清单
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MLOpsChecklist:
    """MLOps生产就绪检查清单"""

    项目名称: str = ""
    检查日期: str = ""

    # 各项检查
    checks: Dict[str, List[Dict]] = field(default_factory=lambda: {
        "代码与版本控制": [
            {"项": "所有代码在Git版本控制中", "状态": False},
            {"项": "有分支保护策略(main/develop)", "状态": False},
            {"项": "PR需要Code Review", "状态": False},
            {"项": "有完整的.gitignore", "状态": False},
        ],
        "数据管理": [
            {"项": "训练数据有版本控制(DVC)", "状态": False},
            {"项": "数据Schema有文档化", "状态": False},
            {"项": "有数据质量验证流程", "状态": False},
            {"项": "敏感数据已脱敏处理", "状态": False},
        ],
        "实验跟踪": [
            {"项": "使用MLflow/W&B跟踪实验", "状态": False},
            {"项": "超参数完整记录", "状态": False},
            {"项": "训练指标有可视化", "状态": False},
            {"项": "模型工件自动保存", "状态": False},
        ],
        "模型管理": [
            {"项": "模型在Registry中注册", "状态": False},
            {"项": "有Staging/Production阶段", "状态": False},
            {"项": "模型有描述文档", "状态": False},
            {"项": "有模型回滚策略", "状态": False},
        ],
        "CI/CD": [
            {"项": "有自动化单元测试", "状态": False},
            {"项": "有模型性能回归测试", "状态": False},
            {"项": "有自动化部署流水线", "状态": False},
            {"项": "有金丝雀发布机制", "状态": False},
        ],
        "监控与告警": [
            {"项": "模型性能指标监控", "状态": False},
            {"项": "数据漂移检测", "状态": False},
            {"项": "资源使用率监控", "状态": False},
            {"项": "告警通知已配置", "状态": False},
        ],
    })


def run_production_checklist():
    """运行生产就绪检查"""
    checklist = MLOpsChecklist(
        项目名称="文本分类服务",
        检查日期="2024-01-15",
    )

    # 模拟检查结果
    results = {
        "代码与版本控制": [True, True, True, True],
        "数据管理": [True, True, False, True],
        "实验跟踪": [True, True, True, True],
        "模型管理": [True, True, False, True],
        "CI/CD": [True, True, True, False],
        "监控与告警": [True, False, True, True],
    }

    print(f"{'=' * 60}")
    print(f"MLOps 生产就绪检查报告")
    print(f"项目: {checklist.项目名称}")
    print(f"日期: {checklist.检查日期}")
    print(f"{'=' * 60}")

    total_pass = 0
    total_items = 0

    for category, checks in checklist.checks.items():
        category_results = results.get(category, [])
        pass_count = sum(category_results)
        total = len(checks)
        total_pass += pass_count
        total_items += total

        status = "PASS" if pass_count == total else "WARN"
        print(f"\n[{status}] {category} ({pass_count}/{total})")
        for i, check in enumerate(checks):
            r = category_results[i] if i < len(category_results) else False
            icon = "[+]" if r else "[-]"
            print(f"  {icon} {check['项']}")

    print(f"\n{'=' * 60}")
    score = total_pass / total_items * 100 if total_items > 0 else 0
    print(f"综合得分: {total_pass}/{total_items} ({score:.0f}%)")
    if score >= 90:
        print("评级: 生产就绪")
    elif score >= 70:
        print("评级: 基本就绪, 建议完善上述未通过项")
    else:
        print("评级: 未就绪, 需要完善多项基础能力")


if __name__ == "__main__":
    run_production_checklist()
```

---

## 总结

本教程涵盖了MLOps实践的核心内容:

1. **MLOps概述**: MLOps成熟度模型(Level 0-2)与核心原则(可复现/自动化/监控/治理)
2. **实验跟踪**: 基于MLflow的实验管理, 参数/指标/工件记录, 实验对比分析
3. **模型注册**: 版本管理、阶段晋升(None->Staging->Production->Archived)
4. **CI/CD for ML**: GitHub Actions工作流, ML测试框架(数据验证/性能测试/回归测试)
5. **数据版本控制**: DVC原理与实践, 数据哈希/版本快照/差异比对
6. **Feature Store**: 在线/离线特征管理, 特征注册与查询
7. **完整Pipeline**: DAG编排, 步骤依赖/重试/状态跟踪
8. **生产最佳实践**: 可复现性/自动化/测试/监控/安全/协作检查清单

## 参考资源

- [MLflow官方文档](https://mlflow.org/docs/latest/index.html)
- [DVC官方文档](https://dvc.org/doc)
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)
- [Google MLOps白皮书](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Feast Feature Store](https://feast.dev/)
- [Evidently AI (数据漂移检测)](https://www.evidentlyai.com/)

---

**创建时间**: 2024-01-15
**最后更新**: 2024-01-15
