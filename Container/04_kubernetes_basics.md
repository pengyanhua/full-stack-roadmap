# Kubernetes (K8s) 基础教程

## 一、Kubernetes 概述

### 什么是 Kubernetes

Kubernetes 是一个开源的容器编排平台，用于自动化部署、扩展和管理容器化应用。

```
Kubernetes 架构：

┌─────────────────────────────────────────────────────────────────────────┐
│                           Control Plane (Master)                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │  API Server  │ │   Scheduler  │ │  Controller  │ │     etcd     │   │
│  │              │ │              │ │   Manager    │ │   (数据库)    │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ 管理
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Worker Nodes                                   │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Node 1                                                         │    │
│  │  ┌─────────┐  ┌────────────────────────────────────────────┐   │    │
│  │  │ kubelet │  │  Pods                                       │   │    │
│  │  └─────────┘  │  ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │    │
│  │  ┌─────────┐  │  │Container│ │Container│ │Container│       │   │    │
│  │  │kube-proxy│  │  └─────────┘ └─────────┘ └─────────┘       │   │    │
│  │  └─────────┘  └────────────────────────────────────────────┘   │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Node 2                                                         │    │
│  │  ...                                                            │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 核心概念

| 概念 | 说明 |
|------|------|
| **Cluster** | 集群，由 Master 和多个 Node 组成 |
| **Node** | 工作节点，运行容器的机器 |
| **Pod** | 最小部署单元，包含一个或多个容器 |
| **Deployment** | 声明式管理 Pod 和 ReplicaSet |
| **Service** | 为 Pod 提供稳定的网络端点 |
| **Namespace** | 资源隔离的虚拟集群 |
| **ConfigMap** | 配置数据存储 |
| **Secret** | 敏感数据存储 |
| **Volume** | 持久化存储 |
| **Ingress** | HTTP/HTTPS 路由 |

### 安装和配置

```bash
# 安装 kubectl（命令行工具）
# macOS
brew install kubectl

# Linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# 验证安装
kubectl version --client

# 本地开发环境
# 1. Minikube
minikube start
minikube status
minikube dashboard  # 打开 Web UI

# 2. Kind (Kubernetes in Docker)
kind create cluster --name my-cluster

# 3. Docker Desktop 自带 Kubernetes
# 在 Docker Desktop 设置中启用

# 配置 kubeconfig
export KUBECONFIG=~/.kube/config

# 查看集群信息
kubectl cluster-info
kubectl get nodes
```

## 二、kubectl 常用命令

### 基础命令

```bash
# 查看资源
kubectl get pods                     # 查看 Pod
kubectl get pods -o wide             # 详细信息
kubectl get pods -A                  # 所有命名空间
kubectl get deployments              # 查看 Deployment
kubectl get services                 # 查看 Service
kubectl get all                      # 查看所有资源

# 描述资源
kubectl describe pod <pod-name>
kubectl describe deployment <name>
kubectl describe node <node-name>

# 创建资源
kubectl create -f manifest.yaml
kubectl apply -f manifest.yaml       # 推荐：声明式

# 删除资源
kubectl delete -f manifest.yaml
kubectl delete pod <pod-name>
kubectl delete deployment <name>

# 编辑资源
kubectl edit deployment <name>

# 查看日志
kubectl logs <pod-name>
kubectl logs -f <pod-name>           # 实时跟踪
kubectl logs <pod-name> -c <container>  # 指定容器
kubectl logs --tail=100 <pod-name>   # 最后100行

# 进入容器
kubectl exec -it <pod-name> -- /bin/bash
kubectl exec -it <pod-name> -c <container> -- /bin/sh

# 端口转发
kubectl port-forward <pod-name> 8080:80
kubectl port-forward svc/<service-name> 8080:80

# 复制文件
kubectl cp <pod-name>:/path/file ./local-file
kubectl cp ./local-file <pod-name>:/path/file
```

### 调试命令

```bash
# 查看 Pod 事件
kubectl get events --sort-by=.metadata.creationTimestamp

# 查看资源使用
kubectl top nodes
kubectl top pods

# 运行临时 Pod 调试
kubectl run debug --image=busybox -it --rm -- /bin/sh
kubectl run debug --image=nicolaka/netshoot -it --rm -- /bin/bash

# 查看 API 资源
kubectl api-resources
kubectl explain pod
kubectl explain pod.spec.containers
```

### 命名空间操作

```bash
# 查看命名空间
kubectl get namespaces
kubectl get ns

# 创建命名空间
kubectl create namespace dev
kubectl create ns prod

# 设置默认命名空间
kubectl config set-context --current --namespace=dev

# 在指定命名空间操作
kubectl get pods -n dev
kubectl apply -f manifest.yaml -n dev

# 删除命名空间（及其所有资源）
kubectl delete namespace dev
```

## 三、核心资源详解

### Pod

Pod 是 K8s 最小的部署单元。

```yaml
# pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
  labels:
    app: my-app
    environment: dev
spec:
  # 容器列表
  containers:
    - name: main-container
      image: nginx:alpine
      ports:
        - containerPort: 80

      # 资源限制
      resources:
        requests:
          memory: "64Mi"
          cpu: "250m"
        limits:
          memory: "128Mi"
          cpu: "500m"

      # 环境变量
      env:
        - name: NODE_ENV
          value: "production"
        - name: DB_HOST
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: db_host
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: db_password

      # 存活探针
      livenessProbe:
        httpGet:
          path: /health
          port: 80
        initialDelaySeconds: 10
        periodSeconds: 10
        failureThreshold: 3

      # 就绪探针
      readinessProbe:
        httpGet:
          path: /ready
          port: 80
        initialDelaySeconds: 5
        periodSeconds: 5

      # 启动探针
      startupProbe:
        httpGet:
          path: /health
          port: 80
        failureThreshold: 30
        periodSeconds: 10

      # 卷挂载
      volumeMounts:
        - name: config-volume
          mountPath: /etc/config
        - name: data-volume
          mountPath: /data

  # 卷定义
  volumes:
    - name: config-volume
      configMap:
        name: app-config
    - name: data-volume
      persistentVolumeClaim:
        claimName: my-pvc

  # 重启策略
  restartPolicy: Always  # Always, OnFailure, Never

  # 节点选择器
  nodeSelector:
    disk: ssd
```

### Deployment

Deployment 用于声明式管理 Pod。

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  labels:
    app: my-app
spec:
  # 副本数
  replicas: 3

  # 选择器（必须与 template.labels 匹配）
  selector:
    matchLabels:
      app: my-app

  # 更新策略
  strategy:
    type: RollingUpdate  # RollingUpdate 或 Recreate
    rollingUpdate:
      maxSurge: 1        # 最多超出期望副本数
      maxUnavailable: 0  # 最多不可用副本数

  # Pod 模板
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
        - name: app
          image: my-app:v1.0.0
          ports:
            - containerPort: 3000
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /health
              port: 3000
            initialDelaySeconds: 15
            periodSeconds: 20
          readinessProbe:
            httpGet:
              path: /ready
              port: 3000
            initialDelaySeconds: 5
            periodSeconds: 10
```

```bash
# Deployment 操作
kubectl apply -f deployment.yaml

# 查看部署状态
kubectl rollout status deployment/my-app

# 查看历史版本
kubectl rollout history deployment/my-app

# 回滚到上一版本
kubectl rollout undo deployment/my-app

# 回滚到指定版本
kubectl rollout undo deployment/my-app --to-revision=2

# 扩缩容
kubectl scale deployment/my-app --replicas=5

# 暂停/恢复部署
kubectl rollout pause deployment/my-app
kubectl rollout resume deployment/my-app

# 更新镜像
kubectl set image deployment/my-app app=my-app:v2.0.0
```

### Service

Service 为 Pod 提供稳定的网络访问。

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
spec:
  # 类型
  type: ClusterIP  # ClusterIP, NodePort, LoadBalancer, ExternalName

  # 选择器
  selector:
    app: my-app

  # 端口映射
  ports:
    - name: http
      port: 80         # Service 端口
      targetPort: 3000 # Pod 端口
      protocol: TCP
```

```yaml
# NodePort Service（对外暴露）
apiVersion: v1
kind: Service
metadata:
  name: my-app-nodeport
spec:
  type: NodePort
  selector:
    app: my-app
  ports:
    - port: 80
      targetPort: 3000
      nodePort: 30080  # 范围 30000-32767
```

```yaml
# LoadBalancer Service（云环境）
apiVersion: v1
kind: Service
metadata:
  name: my-app-lb
spec:
  type: LoadBalancer
  selector:
    app: my-app
  ports:
    - port: 80
      targetPort: 3000
```

```yaml
# Headless Service（用于 StatefulSet）
apiVersion: v1
kind: Service
metadata:
  name: my-app-headless
spec:
  clusterIP: None
  selector:
    app: my-app
  ports:
    - port: 3000
```

### ConfigMap

存储非敏感配置数据。

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  # 键值对
  db_host: "postgres"
  db_port: "5432"

  # 多行配置文件
  nginx.conf: |
    server {
        listen 80;
        server_name localhost;
        location / {
            proxy_pass http://backend:3000;
        }
    }

  # JSON 配置
  app.json: |
    {
      "debug": false,
      "logLevel": "info"
    }
```

```bash
# 从命令行创建
kubectl create configmap app-config \
    --from-literal=db_host=postgres \
    --from-literal=db_port=5432

# 从文件创建
kubectl create configmap app-config --from-file=config.json
kubectl create configmap app-config --from-file=./configs/

# 使用 ConfigMap
# 1. 环境变量
env:
  - name: DB_HOST
    valueFrom:
      configMapKeyRef:
        name: app-config
        key: db_host

# 2. 全部导入为环境变量
envFrom:
  - configMapRef:
      name: app-config

# 3. 挂载为文件
volumes:
  - name: config
    configMap:
      name: app-config
volumeMounts:
  - name: config
    mountPath: /etc/config
```

### Secret

存储敏感数据（Base64 编码）。

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
data:
  # 值需要 Base64 编码
  # echo -n "password123" | base64
  db_password: cGFzc3dvcmQxMjM=
  api_key: bXktYXBpLWtleQ==

---
# 使用 stringData 可以直接写明文
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
stringData:
  db_password: password123
  api_key: my-api-key
```

```bash
# 从命令行创建
kubectl create secret generic app-secrets \
    --from-literal=db_password=password123 \
    --from-literal=api_key=my-api-key

# 创建 TLS Secret
kubectl create secret tls tls-secret \
    --cert=path/to/cert.pem \
    --key=path/to/key.pem

# 创建 Docker Registry Secret
kubectl create secret docker-registry regcred \
    --docker-server=<registry-server> \
    --docker-username=<username> \
    --docker-password=<password> \
    --docker-email=<email>
```

### Ingress

HTTP/HTTPS 路由。

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx

  # TLS 配置
  tls:
    - hosts:
        - example.com
      secretName: tls-secret

  # 路由规则
  rules:
    - host: example.com
      http:
        paths:
          - path: /api
            pathType: Prefix
            backend:
              service:
                name: api-service
                port:
                  number: 80

          - path: /
            pathType: Prefix
            backend:
              service:
                name: web-service
                port:
                  number: 80

    # 另一个域名
    - host: admin.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: admin-service
                port:
                  number: 80
```

### PersistentVolume & PersistentVolumeClaim

持久化存储。

```yaml
# pv.yaml - 管理员创建
apiVersion: v1
kind: PersistentVolume
metadata:
  name: my-pv
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce  # RWO, ROX, RWX
  persistentVolumeReclaimPolicy: Retain  # Retain, Delete, Recycle
  storageClassName: standard
  hostPath:
    path: /data/pv
```

```yaml
# pvc.yaml - 用户申请
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: standard
```

```yaml
# 使用 PVC
volumes:
  - name: data
    persistentVolumeClaim:
      claimName: my-pvc
```

## 四、常用命令速查表

```bash
# ===== 集群信息 =====
kubectl cluster-info
kubectl get nodes
kubectl describe node <node>
kubectl top nodes

# ===== Pod 操作 =====
kubectl get pods [-o wide] [-A] [-n namespace]
kubectl describe pod <pod>
kubectl logs <pod> [-f] [-c container]
kubectl exec -it <pod> -- /bin/bash
kubectl delete pod <pod>
kubectl top pods

# ===== Deployment 操作 =====
kubectl get deployments
kubectl describe deployment <name>
kubectl scale deployment <name> --replicas=N
kubectl rollout status deployment/<name>
kubectl rollout history deployment/<name>
kubectl rollout undo deployment/<name>
kubectl set image deployment/<name> container=image:tag

# ===== Service 操作 =====
kubectl get services
kubectl describe service <name>
kubectl expose deployment <name> --port=80 --type=NodePort

# ===== 配置管理 =====
kubectl get configmaps
kubectl get secrets
kubectl create configmap <name> --from-literal=key=value
kubectl create secret generic <name> --from-literal=key=value

# ===== 调试 =====
kubectl get events --sort-by=.metadata.creationTimestamp
kubectl run debug --image=busybox -it --rm -- /bin/sh
kubectl port-forward <pod> local:remote

# ===== 资源应用 =====
kubectl apply -f manifest.yaml
kubectl delete -f manifest.yaml
kubectl diff -f manifest.yaml
```
