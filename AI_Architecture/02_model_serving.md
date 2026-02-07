# 模型服务化部署

## 1. 模型服务架构对比

### 1.1 三大主流方案

```
┌────────────────────────────────────────────────────────┐
│ TensorFlow Serving (Google)                           │
├────────────────────────────────────────────────────────┤
│ 优势: 高性能、成熟稳定、支持版本管理                   │
│ 劣势: 仅支持TensorFlow/Keras                           │
│ 适用: TensorFlow模型生产部署                           │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ TorchServe (AWS + Facebook)                            │
├────────────────────────────────────────────────────────┤
│ 优势: PyTorch官方、灵活、支持自定义Handler              │
│ 劣势: 社区较小、文档较少                               │
│ 适用: PyTorch模型部署                                  │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ ONNX Runtime (Microsoft)                               │
├────────────────────────────────────────────────────────┤
│ 优势: 跨框架、高性能、硬件加速                         │
│ 劣势: 模型转换可能有兼容性问题                         │
│ 适用: 跨框架模型部署、边缘设备                         │
└────────────────────────────────────────────────────────┘
```

### 1.2 性能对比

| 指标 | TF Serving | TorchServe | ONNX Runtime |
|------|------------|------------|--------------|
| **吞吐量** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **延迟** | <5ms | <10ms | <3ms |
| **内存占用** | 中 | 高 | 低 |
| **GPU加速** | ✅ | ✅ | ✅ (CUDA/TensorRT) |
| **批处理** | ✅ | ✅ | ✅ |
| **热更新** | ✅ | ✅ | ❌ |

## 2. TensorFlow Serving部署

### 2.1 模型导出

```python
"""
TensorFlow模型导出为SavedModel格式
"""
import tensorflow as tf
from tensorflow import keras

# 训练模型
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练（省略）
# model.fit(X_train, y_train, epochs=5)

# 导出SavedModel
export_path = './saved_models/mnist/1'  # 版本号为1

tf.saved_model.save(model, export_path)

# 查看SavedModel
!saved_model_cli show --dir {export_path} --all
```

### 2.2 Docker部署

```bash
# 1. 拉取TensorFlow Serving镜像
docker pull tensorflow/serving:latest

# 2. 启动服务
docker run -d \
  --name tf-serving \
  -p 8501:8501 \
  -p 8500:8500 \
  -v $(pwd)/saved_models:/models \
  -e MODEL_NAME=mnist \
  tensorflow/serving

# 8501: REST API
# 8500: gRPC API
```

### 2.3 客户端调用

```python
"""
TensorFlow Serving客户端
"""
import requests
import numpy as np
import json

# REST API调用
def predict_rest(image):
    """使用REST API预测"""
    url = 'http://localhost:8501/v1/models/mnist:predict'

    # 准备数据
    data = {
        "instances": image.tolist()
    }

    # 发送请求
    response = requests.post(url, json=data)
    predictions = response.json()['predictions']

    return np.array(predictions)

# gRPC调用（性能更好）
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

def predict_grpc(image):
    """使用gRPC预测（推荐）"""
    channel = grpc.insecure_channel('localhost:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # 创建请求
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mnist'
    request.model_spec.signature_name = 'serving_default'

    # 设置输入
    request.inputs['dense_input'].CopyFrom(
        tf.make_tensor_proto(image, shape=[1, 784])
    )

    # 调用
    result = stub.Predict(request, 10.0)  # 10秒超时

    # 解析结果
    output = tf.make_ndarray(result.outputs['dense'])
    return output

# 使用示例
test_image = np.random.rand(1, 784).astype(np.float32)

# REST
pred_rest = predict_rest(test_image)
print(f'REST Prediction: {pred_rest}')

# gRPC
pred_grpc = predict_grpc(test_image)
print(f'gRPC Prediction: {pred_grpc}')
```

### 2.4 版本管理与A/B测试

```bash
# 目录结构
saved_models/
└── mnist/
    ├── 1/  # 版本1
    │   ├── saved_model.pb
    │   └── variables/
    ├── 2/  # 版本2
    │   ├── saved_model.pb
    │   └── variables/
    └── 3/  # 版本3
        ├── saved_model.pb
        └── variables/

# 配置文件：model_config.config
model_config_list {
  config {
    name: "mnist"
    base_path: "/models/mnist"
    model_platform: "tensorflow"
    model_version_policy {
      specific {
        versions: 2
        versions: 3
      }
    }
  }
}

# 启动时指定配置
docker run -d \
  --name tf-serving \
  -p 8501:8501 \
  -v $(pwd)/saved_models:/models \
  -v $(pwd)/model_config.config:/models/model_config.config \
  tensorflow/serving \
  --model_config_file=/models/model_config.config \
  --model_config_file_poll_wait_seconds=60  # 每60秒检查更新
```

```python
"""
A/B测试：流量分配到不同版本
"""
import random

def predict_with_ab_test(image, version_weights={2: 0.9, 3: 0.1}):
    """
    A/B测试预测
    version_weights: {版本号: 流量占比}
    """
    # 随机选择版本
    versions = list(version_weights.keys())
    weights = list(version_weights.values())
    selected_version = random.choices(versions, weights=weights)[0]

    # 调用指定版本
    url = f'http://localhost:8501/v1/models/mnist/versions/{selected_version}:predict'

    data = {"instances": image.tolist()}
    response = requests.post(url, json=data)

    return {
        'version': selected_version,
        'prediction': response.json()['predictions']
    }

# 使用示例
result = predict_with_ab_test(test_image)
print(f"Version: {result['version']}, Prediction: {result['prediction']}")
```

## 3. TorchServe Handler开发

### 3.1 自定义Handler

```python
"""
TorchServe自定义Handler
文件名: custom_handler.py
"""
import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler
import json
import logging

logger = logging.getLogger(__name__)

class CustomImageClassifier(BaseHandler):
    """
    自定义图像分类Handler
    """

    def __init__(self):
        super(CustomImageClassifier, self).__init__()
        self.initialized = False

    def initialize(self, context):
        """
        初始化Handler
        Args:
            context: TorchServe上下文
        """
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # 加载模型
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = f"{model_dir}/{serialized_file}"

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )

        # 加载模型结构
        self.model = torch.jit.load(model_pt_path, map_location=self.device)
        self.model.eval()

        # 加载类别标签
        mapping_file_path = f"{model_dir}/index_to_name.json"
        with open(mapping_file_path) as f:
            self.mapping = json.load(f)

        self.initialized = True
        logger.info("Model initialized successfully")

    def preprocess(self, data):
        """
        数据预处理
        Args:
            data: 输入数据列表
        Returns:
            Tensor
        """
        from PIL import Image
        from torchvision import transforms
        import io

        # 定义预处理
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        images = []
        for row in data:
            # 读取图像
            image = row.get("data") or row.get("body")

            if isinstance(image, str):
                # Base64编码的图像
                image = base64.b64decode(image)

            # 转换为PIL Image
            image = Image.open(io.BytesIO(image))

            # 预处理
            image = preprocess(image)
            images.append(image)

        # 批处理
        batch = torch.stack(images).to(self.device)
        return batch

    def inference(self, data):
        """
        模型推理
        Args:
            data: 预处理后的Tensor
        Returns:
            推理结果
        """
        with torch.no_grad():
            outputs = self.model(data)
            # Softmax归一化
            probabilities = F.softmax(outputs, dim=1)

        return probabilities

    def postprocess(self, inference_output):
        """
        后处理
        Args:
            inference_output: 推理结果
        Returns:
            JSON响应
        """
        # Top-5预测
        top5_prob, top5_class = torch.topk(inference_output, 5, dim=1)

        results = []
        for i in range(len(top5_class)):
            result = {}
            for j in range(5):
                class_id = top5_class[i][j].item()
                prob = top5_prob[i][j].item()
                result[self.mapping[str(class_id)]] = prob

            results.append(result)

        return results

    def handle(self, data, context):
        """
        完整处理流程
        """
        # 预处理
        preprocessed_data = self.preprocess(data)

        # 推理
        inference_output = self.inference(preprocessed_data)

        # 后处理
        response = self.postprocess(inference_output)

        return response
```

### 3.2 模型打包与部署

```bash
# 1. 导出TorchScript模型
python <<EOF
import torch
from torchvision import models

model = models.resnet50(pretrained=True)
model.eval()

# 转换为TorchScript
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("resnet50.pt")
EOF

# 2. 创建类别映射文件
cat > index_to_name.json <<EOF
{
  "0": "tench",
  "1": "goldfish",
  "2": "great_white_shark"
}
EOF

# 3. 打包模型
torch-model-archiver \
  --model-name resnet50 \
  --version 1.0 \
  --serialized-file resnet50.pt \
  --handler custom_handler.py \
  --extra-files index_to_name.json \
  --export-path model_store

# 4. 启动TorchServe
torchserve \
  --start \
  --ncs \
  --model-store model_store \
  --models resnet50=resnet50.mar \
  --ts-config config.properties

# config.properties内容
# inference_address=http://0.0.0.0:8080
# management_address=http://0.0.0.0:8081
# metrics_address=http://0.0.0.0:8082
# number_of_netty_threads=32
# job_queue_size=1000
# enable_envvars_config=true
# install_py_dep_per_model=true
# default_workers_per_model=4
```

### 3.3 动态批处理

```python
"""
TorchServe动态批处理配置
"""
# config.properties添加
batch_size=8             # 批大小
max_batch_delay=100      # 最大等待时间(ms)
```

```python
"""
Handler中处理批次
"""
class BatchedHandler(BaseHandler):

    def handle(self, data, context):
        """处理批次请求"""
        # data是一个列表，包含多个请求
        batch_size = len(data)
        logger.info(f"Processing batch of size: {batch_size}")

        # 批量预处理
        preprocessed_data = self.preprocess(data)

        # 批量推理
        inference_output = self.inference(preprocessed_data)

        # 批量后处理
        response = self.postprocess(inference_output)

        return response
```

## 4. ONNX Runtime优化

### 4.1 模型转换

```python
"""
PyTorch → ONNX转换
"""
import torch
import torch.onnx

# 加载PyTorch模型
model = torch.load('model.pth')
model.eval()

# 准备示例输入
dummy_input = torch.randn(1, 3, 224, 224)

# 导出ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# 验证ONNX模型
import onnx

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")
```

### 4.2 ONNX Runtime推理

```python
"""
ONNX Runtime高性能推理
"""
import onnxruntime as ort
import numpy as np

# 创建推理会话
session = ort.InferenceSession(
    "model.onnx",
    providers=[
        'CUDAExecutionProvider',  # GPU加速
        'CPUExecutionProvider'
    ]
)

# 查看输入输出信息
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"Input name: {input_name}")
print(f"Output name: {output_name}")

# 推理
def predict(image):
    """ONNX Runtime推理"""
    # 准备输入
    input_data = {input_name: image.astype(np.float32)}

    # 执行推理
    outputs = session.run([output_name], input_data)

    return outputs[0]

# 性能测试
import time

image = np.random.rand(1, 3, 224, 224).astype(np.float32)

# 预热
for _ in range(10):
    predict(image)

# 测试
start_time = time.time()
iterations = 1000

for _ in range(iterations):
    predict(image)

elapsed_time = time.time() - start_time
print(f"Average inference time: {elapsed_time/iterations*1000:.2f}ms")
```

### 4.3 量化优化

```python
"""
ONNX模型量化（减小模型大小，加速推理）
"""
from onnxruntime.quantization import quantize_dynamic, QuantType

# 动态量化
quantize_dynamic(
    model_input="model.onnx",
    model_output="model_quantized.onnx",
    weight_type=QuantType.QUInt8  # 权重量化为8位整数
)

# 对比模型大小
import os

original_size = os.path.getsize("model.onnx") / (1024 * 1024)
quantized_size = os.path.getsize("model_quantized.onnx") / (1024 * 1024)

print(f"Original model: {original_size:.2f} MB")
print(f"Quantized model: {quantized_size:.2f} MB")
print(f"Compression ratio: {original_size / quantized_size:.2f}x")

# 性能对比
session_quantized = ort.InferenceSession("model_quantized.onnx")

# 测试量化模型
start_time = time.time()
for _ in range(1000):
    session_quantized.run(None, {input_name: image})

quantized_time = time.time() - start_time
print(f"Quantized model time: {quantized_time:.2f}s")
```

## 5. 模型版本管理与A/B测试

### 5.1 Kubernetes部署多版本

```yaml
# deployment-v1.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-v1
  labels:
    app: model
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model
      version: v1
  template:
    metadata:
      labels:
        app: model
        version: v1
    spec:
      containers:
      - name: model
        image: my-model:v1
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
---
# deployment-v2.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-v2
  labels:
    app: model
    version: v2
spec:
  replicas: 1  # 初始只部署少量副本
  selector:
    matchLabels:
      app: model
      version: v2
  template:
    metadata:
      labels:
        app: model
        version: v2
    spec:
      containers:
      - name: model
        image: my-model:v2
        ports:
        - containerPort: 8080
---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: model-service
spec:
  selector:
    app: model
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

### 5.2 Istio流量分配

```yaml
# virtual-service.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: model-vs
spec:
  hosts:
  - model-service
  http:
  - match:
    - headers:
        x-user-type:
          exact: "beta"
    route:
    - destination:
        host: model-service
        subset: v2
      weight: 100
  - route:
    - destination:
        host: model-service
        subset: v1
      weight: 90  # 90%流量到v1
    - destination:
        host: model-service
        subset: v2
      weight: 10  # 10%流量到v2
---
# destination-rule.yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: model-dr
spec:
  host: model-service
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
```

### 5.3 A/B测试指标收集

```python
"""
模型性能监控与A/B测试分析
"""
from prometheus_client import Counter, Histogram, Gauge
import time
import random

# 定义指标
prediction_requests = Counter(
    'model_prediction_requests_total',
    'Total prediction requests',
    ['model_version', 'status']
)

prediction_latency = Histogram(
    'model_prediction_latency_seconds',
    'Prediction latency',
    ['model_version']
)

prediction_accuracy = Gauge(
    'model_prediction_accuracy',
    'Model accuracy',
    ['model_version']
)

def predict_with_metrics(image, model_version):
    """带监控的预测"""
    start_time = time.time()

    try:
        # 调用模型
        result = model_predict(image, model_version)

        # 记录成功
        prediction_requests.labels(
            model_version=model_version,
            status='success'
        ).inc()

        # 记录延迟
        latency = time.time() - start_time
        prediction_latency.labels(model_version=model_version).observe(latency)

        return result

    except Exception as e:
        # 记录失败
        prediction_requests.labels(
            model_version=model_version,
            status='error'
        ).inc()
        raise e

# A/B测试分析
def analyze_ab_test():
    """分析A/B测试结果"""
    from prometheus_api_client import PrometheusConnect

    prom = PrometheusConnect(url="http://prometheus:9090")

    # 查询v1和v2的成功率
    v1_success_rate = prom.custom_query(
        'rate(model_prediction_requests_total{model_version="v1",status="success"}[5m])'
    )
    v2_success_rate = prom.custom_query(
        'rate(model_prediction_requests_total{model_version="v2",status="success"}[5m])'
    )

    # 查询平均延迟
    v1_latency = prom.custom_query(
        'histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket{model_version="v1"}[5m]))'
    )
    v2_latency = prom.custom_query(
        'histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket{model_version="v2"}[5m]))'
    )

    print(f"V1 Success Rate: {v1_success_rate}")
    print(f"V2 Success Rate: {v2_success_rate}")
    print(f"V1 P95 Latency: {v1_latency}")
    print(f"V2 P95 Latency: {v2_latency}")

    # 决策：如果v2指标更好，增加流量
    if v2_success_rate > v1_success_rate and v2_latency < v1_latency:
        print("✅ V2性能更好，建议增加流量")
        return "promote_v2"
    else:
        print("⚠️ V2性能未达预期，保持当前流量分配")
        return "keep_current"
```

## 6. 推理性能优化

### 6.1 批处理优化

```python
"""
动态批处理实现
"""
import asyncio
from collections import deque
import time

class DynamicBatcher:
    """动态批处理器"""

    def __init__(self, max_batch_size=32, max_wait_time=0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time  # 100ms
        self.queue = deque()
        self.batch_ready = asyncio.Event()

    async def add_request(self, request):
        """添加请求到批次"""
        future = asyncio.Future()
        self.queue.append((request, future))

        # 如果达到批大小，立即处理
        if len(self.queue) >= self.max_batch_size:
            self.batch_ready.set()

        # 等待结果
        return await future

    async def process_batches(self):
        """处理批次"""
        while True:
            # 等待批次就绪或超时
            try:
                await asyncio.wait_for(
                    self.batch_ready.wait(),
                    timeout=self.max_wait_time
                )
            except asyncio.TimeoutError:
                pass

            # 如果队列为空，继续等待
            if not self.queue:
                continue

            # 取出批次
            batch_size = min(len(self.queue), self.max_batch_size)
            batch = [self.queue.popleft() for _ in range(batch_size)]

            # 批量推理
            requests = [req for req, _ in batch]
            futures = [fut for _, fut in batch]

            results = await self.batch_inference(requests)

            # 返回结果
            for future, result in zip(futures, results):
                future.set_result(result)

            self.batch_ready.clear()

    async def batch_inference(self, requests):
        """批量推理（实际调用模型）"""
        import numpy as np

        # 合并输入
        batch_input = np.stack([req['data'] for req in requests])

        # 批量推理
        batch_output = model_predict(batch_input)

        # 拆分输出
        return [output for output in batch_output]

# 使用示例
batcher = DynamicBatcher(max_batch_size=32, max_wait_time=0.1)

# 启动批处理器
asyncio.create_task(batcher.process_batches())

# 处理请求
async def handle_request(request):
    result = await batcher.add_request(request)
    return result
```

### 6.2 模型并发优化

```python
"""
多线程/多进程推理
"""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np

class ConcurrentInferenceServer:
    """并发推理服务器"""

    def __init__(self, num_workers=4):
        # 使用线程池（适合IO密集型）
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        # 或使用进程池（适合CPU密集型）
        # self.executor = ProcessPoolExecutor(max_workers=num_workers)

        # 加载模型（每个worker一个实例）
        self.models = [load_model() for _ in range(num_workers)]

    def predict(self, image):
        """异步预测"""
        future = self.executor.submit(self._predict, image)
        return future.result()

    def _predict(self, image):
        """实际预测逻辑"""
        import threading

        # 获取当前线程的模型实例
        thread_id = threading.get_ident() % len(self.models)
        model = self.models[thread_id]

        return model.predict(image)

# 使用示例
server = ConcurrentInferenceServer(num_workers=4)

# 并发处理多个请求
from concurrent.futures import as_completed

requests = [np.random.rand(224, 224, 3) for _ in range(100)]

futures = [server.executor.submit(server._predict, req) for req in requests]

for future in as_completed(futures):
    result = future.result()
    print(f"Prediction: {result}")
```

模型服务化完整教程完成！继续创建剩余5个文件...
