# Docker & Kubernetes 命令速查表

> 常用命令快速参考，适合日常开发和运维使用

---

## 目录

1. [Docker 命令](#docker-命令)
2. [Docker Compose 命令](#docker-compose-命令)
3. [Kubernetes 命令](#kubernetes-命令)
4. [常见问题排查](#常见问题排查)

---

## Docker 命令

### 镜像管理

```bash
# 查看本地镜像
docker images
docker image ls

# 搜索镜像
docker search nginx

# 拉取镜像
docker pull nginx:latest
docker pull nginx:1.24-alpine

# 构建镜像
docker build -t myapp:v1 .
docker build -t myapp:v1 -f Dockerfile.prod .
docker build --no-cache -t myapp:v1 .          # 不使用缓存

# 给镜像打标签
docker tag myapp:v1 registry.example.com/myapp:v1

# 推送镜像
docker push registry.example.com/myapp:v1

# 删除镜像
docker rmi nginx:latest
docker rmi $(docker images -q)                 # 删除所有镜像
docker image prune                              # 删除悬空镜像
docker image prune -a                           # 删除所有未使用镜像

# 导出/导入镜像
docker save -o myapp.tar myapp:v1              # 导出
docker load -i myapp.tar                        # 导入

# 查看镜像历史
docker history myapp:v1
```

### 容器管理

```bash
# 运行容器
docker run nginx                                # 前台运行
docker run -d nginx                             # 后台运行
docker run -d --name web nginx                  # 指定名称
docker run -d -p 8080:80 nginx                  # 端口映射
docker run -d -P nginx                          # 随机端口映射
docker run -d -v /data:/app/data nginx          # 卷挂载
docker run -d -e ENV=prod nginx                 # 设置环境变量
docker run -d --restart=always nginx            # 自动重启
docker run -it ubuntu /bin/bash                 # 交互模式

# 查看容器
docker ps                                       # 运行中的容器
docker ps -a                                    # 所有容器
docker ps -q                                    # 只显示容器ID

# 启动/停止/重启
docker start <container>
docker stop <container>
docker restart <container>
docker kill <container>                         # 强制停止

# 进入容器
docker exec -it <container> /bin/bash
docker exec -it <container> sh
docker attach <container>                       # 附加到主进程

# 查看容器信息
docker inspect <container>
docker logs <container>
docker logs -f <container>                      # 实时日志
docker logs --tail 100 <container>              # 最后100行
docker logs --since 1h <container>              # 最近1小时

# 复制文件
docker cp <container>:/path/file ./local        # 从容器复制
docker cp ./local <container>:/path/            # 复制到容器

# 资源使用
docker stats                                    # 实时资源统计
docker stats <container>
docker top <container>                          # 容器进程

# 删除容器
docker rm <container>
docker rm -f <container>                        # 强制删除运行中的
docker container prune                          # 删除所有停止的容器
docker rm $(docker ps -aq)                      # 删除所有容器
```

### 网络管理

```bash
# 查看网络
docker network ls
docker network inspect bridge

# 创建网络
docker network create mynet
docker network create --driver bridge mynet
docker network create --subnet=172.20.0.0/16 mynet

# 连接/断开网络
docker network connect mynet <container>
docker network disconnect mynet <container>

# 运行时指定网络
docker run -d --network mynet nginx

# 删除网络
docker network rm mynet
docker network prune                            # 删除未使用的网络
```

### 卷管理

```bash
# 查看卷
docker volume ls
docker volume inspect myvolume

# 创建卷
docker volume create myvolume

# 使用卷
docker run -d -v myvolume:/app/data nginx       # 命名卷
docker run -d -v /host/path:/container/path nginx  # 绑定挂载
docker run -d -v /app/data nginx                # 匿名卷

# 删除卷
docker volume rm myvolume
docker volume prune                             # 删除未使用的卷
```

### 系统维护

```bash
# 系统信息
docker info
docker version

# 磁盘使用
docker system df
docker system df -v                             # 详细信息

# 清理
docker system prune                             # 清理未使用资源
docker system prune -a                          # 清理所有未使用资源
docker system prune -a --volumes                # 包括卷
```

---

## Docker Compose 命令

```bash
# 启动服务
docker compose up                               # 前台运行
docker compose up -d                            # 后台运行
docker compose up -d --build                    # 重新构建
docker compose up -d --force-recreate           # 强制重建容器

# 停止服务
docker compose stop                             # 停止
docker compose down                             # 停止并删除容器
docker compose down -v                          # 同时删除卷
docker compose down --rmi all                   # 同时删除镜像

# 查看状态
docker compose ps
docker compose ps -a                            # 所有容器

# 查看日志
docker compose logs
docker compose logs -f                          # 实时日志
docker compose logs <service>                   # 指定服务

# 执行命令
docker compose exec <service> sh
docker compose exec <service> npm run migrate

# 扩缩容
docker compose up -d --scale web=3

# 重启服务
docker compose restart
docker compose restart <service>

# 构建
docker compose build
docker compose build --no-cache

# 配置验证
docker compose config                           # 验证并显示配置

# 查看镜像
docker compose images

# 指定文件
docker compose -f docker-compose.prod.yml up -d
```

---

## Kubernetes 命令

### 集群信息

```bash
# 集群信息
kubectl cluster-info
kubectl version

# 查看节点
kubectl get nodes
kubectl get nodes -o wide
kubectl describe node <node-name>

# 查看 API 资源
kubectl api-resources
kubectl api-versions
```

### 命名空间

```bash
# 查看命名空间
kubectl get namespaces
kubectl get ns

# 创建命名空间
kubectl create namespace myapp

# 设置默认命名空间
kubectl config set-context --current --namespace=myapp

# 查看当前命名空间
kubectl config view --minify | grep namespace
```

### Pod 操作

```bash
# 查看 Pod
kubectl get pods
kubectl get pods -o wide
kubectl get pods -A                             # 所有命名空间
kubectl get pods -n <namespace>
kubectl get pods -l app=nginx                   # 按标签筛选
kubectl get pods --show-labels
kubectl get pods -w                             # 监视变化

# Pod 详情
kubectl describe pod <pod-name>

# 创建 Pod
kubectl run nginx --image=nginx
kubectl run nginx --image=nginx --port=80

# 删除 Pod
kubectl delete pod <pod-name>
kubectl delete pods --all                       # 删除所有
kubectl delete pods -l app=nginx                # 按标签删除

# 进入 Pod
kubectl exec -it <pod-name> -- /bin/bash
kubectl exec -it <pod-name> -c <container> -- sh

# 查看日志
kubectl logs <pod-name>
kubectl logs -f <pod-name>                      # 实时日志
kubectl logs <pod-name> -c <container>          # 指定容器
kubectl logs <pod-name> --previous              # 上一个容器的日志
kubectl logs -l app=nginx                       # 按标签查看

# 复制文件
kubectl cp <pod-name>:/path/file ./local
kubectl cp ./local <pod-name>:/path/

# 端口转发
kubectl port-forward <pod-name> 8080:80
kubectl port-forward svc/<service-name> 8080:80
```

### Deployment 操作

```bash
# 查看 Deployment
kubectl get deployments
kubectl get deploy
kubectl describe deployment <name>

# 创建 Deployment
kubectl create deployment nginx --image=nginx
kubectl create deployment nginx --image=nginx --replicas=3

# 扩缩容
kubectl scale deployment nginx --replicas=5

# 更新镜像
kubectl set image deployment/nginx nginx=nginx:1.19

# 滚动更新状态
kubectl rollout status deployment/nginx
kubectl rollout history deployment/nginx
kubectl rollout undo deployment/nginx           # 回滚
kubectl rollout undo deployment/nginx --to-revision=2

# 暂停/恢复更新
kubectl rollout pause deployment/nginx
kubectl rollout resume deployment/nginx

# 重启 Deployment
kubectl rollout restart deployment/nginx

# 删除
kubectl delete deployment nginx
```

### Service 操作

```bash
# 查看 Service
kubectl get services
kubectl get svc
kubectl describe svc <name>

# 创建 Service
kubectl expose deployment nginx --port=80 --type=ClusterIP
kubectl expose deployment nginx --port=80 --type=NodePort
kubectl expose deployment nginx --port=80 --type=LoadBalancer

# 创建临时 Service（测试）
kubectl expose pod nginx --port=80 --target-port=80

# 查看 Endpoints
kubectl get endpoints

# 删除
kubectl delete svc nginx
```

### ConfigMap 和 Secret

```bash
# ConfigMap
kubectl get configmaps
kubectl describe configmap <name>
kubectl create configmap myconfig --from-literal=key1=value1
kubectl create configmap myconfig --from-file=config.txt
kubectl create configmap myconfig --from-env-file=.env

# Secret
kubectl get secrets
kubectl describe secret <name>
kubectl create secret generic mysecret --from-literal=password=123456
kubectl create secret docker-registry regcred \
  --docker-server=registry.example.com \
  --docker-username=user \
  --docker-password=pass

# 查看 Secret 内容（base64 解码）
kubectl get secret mysecret -o jsonpath='{.data.password}' | base64 -d
```

### 声明式管理

```bash
# 应用配置
kubectl apply -f deployment.yaml
kubectl apply -f ./manifests/                   # 目录下所有文件
kubectl apply -f ./manifests/ -R                # 递归
kubectl apply -f https://example.com/manifest.yaml

# 验证配置
kubectl apply -f deployment.yaml --dry-run=client
kubectl diff -f deployment.yaml                 # 查看差异

# 删除资源
kubectl delete -f deployment.yaml
kubectl delete -f ./manifests/

# 编辑资源
kubectl edit deployment nginx

# 导出资源配置
kubectl get deployment nginx -o yaml > deployment.yaml
```

### 调试命令

```bash
# 运行调试容器
kubectl run debug --rm -it --image=busybox -- sh
kubectl run debug --rm -it --image=nicolaka/netshoot -- bash

# 调试 Pod
kubectl debug <pod-name> -it --image=busybox

# 查看事件
kubectl get events
kubectl get events --sort-by='.lastTimestamp'
kubectl get events -w                           # 实时监视

# 资源使用（需要 metrics-server）
kubectl top nodes
kubectl top pods
kubectl top pods --containers

# 查看资源配额
kubectl describe quota
kubectl describe limitrange
```

### 常用 JSONPath

```bash
# 获取所有 Pod 名称
kubectl get pods -o jsonpath='{.items[*].metadata.name}'

# 获取所有 Pod IP
kubectl get pods -o jsonpath='{.items[*].status.podIP}'

# 获取所有节点 IP
kubectl get nodes -o jsonpath='{.items[*].status.addresses[0].address}'

# 获取特定标签的 Pod
kubectl get pods -o jsonpath='{.items[?(@.metadata.labels.app=="nginx")].metadata.name}'
```

---

## 常见问题排查

### Docker 问题

```bash
# 容器无法启动
docker logs <container>                         # 查看日志
docker inspect <container>                      # 检查配置

# 磁盘空间不足
docker system df                                # 查看使用情况
docker system prune -a --volumes                # 清理

# 网络问题
docker network inspect bridge
docker exec -it <container> ping <other-container>

# 镜像拉取失败
docker login registry.example.com               # 重新登录
docker pull --disable-content-trust nginx       # 禁用内容信任
```

### Kubernetes 问题

```bash
# Pod 一直 Pending
kubectl describe pod <pod-name>                 # 查看事件
kubectl get events                              # 查看集群事件
kubectl describe nodes                          # 检查节点资源

# Pod 一直 CrashLoopBackOff
kubectl logs <pod-name>                         # 查看日志
kubectl logs <pod-name> --previous              # 上一次的日志
kubectl describe pod <pod-name>                 # 检查探针配置

# Pod 一直 ImagePullBackOff
kubectl describe pod <pod-name>                 # 查看错误详情
kubectl get secret                              # 检查镜像拉取凭证

# Service 无法访问
kubectl get endpoints                           # 检查端点
kubectl describe svc <service-name>             # 检查选择器

# 调试网络
kubectl run debug --rm -it --image=nicolaka/netshoot -- bash
# 在容器内
nslookup <service-name>
curl <service-name>:<port>

# 检查 DNS
kubectl run debug --rm -it --image=busybox -- nslookup kubernetes

# 检查 RBAC 权限
kubectl auth can-i get pods
kubectl auth can-i get pods --as=system:serviceaccount:default:myapp
```

### 性能问题

```bash
# 查看资源使用
kubectl top nodes
kubectl top pods --sort-by=memory
kubectl top pods --sort-by=cpu

# 查看 HPA 状态
kubectl get hpa
kubectl describe hpa <name>

# 检查资源限制
kubectl describe pod <pod-name> | grep -A5 Limits
kubectl describe pod <pod-name> | grep -A5 Requests
```

---

## 快捷别名配置

将以下内容添加到 `~/.bashrc` 或 `~/.zshrc`:

```bash
# Docker 别名
alias d='docker'
alias dc='docker compose'
alias dps='docker ps'
alias dpsa='docker ps -a'
alias di='docker images'
alias dex='docker exec -it'
alias dlog='docker logs -f'
alias dprune='docker system prune -a'

# Kubernetes 别名
alias k='kubectl'
alias kgp='kubectl get pods'
alias kgd='kubectl get deployments'
alias kgs='kubectl get services'
alias kgn='kubectl get nodes'
alias kd='kubectl describe'
alias kdp='kubectl describe pod'
alias kl='kubectl logs -f'
alias kex='kubectl exec -it'
alias kaf='kubectl apply -f'
alias kdf='kubectl delete -f'
alias kctx='kubectl config current-context'
alias kns='kubectl config set-context --current --namespace'

# 快捷函数
klog() { kubectl logs -f $(kubectl get pods -l app=$1 -o jsonpath='{.items[0].metadata.name}'); }
kbash() { kubectl exec -it $(kubectl get pods -l app=$1 -o jsonpath='{.items[0].metadata.name}') -- /bin/bash; }
```

---

## 参考资源

- Docker 官方文档: https://docs.docker.com/
- Kubernetes 官方文档: https://kubernetes.io/docs/
- kubectl 速查表: https://kubernetes.io/docs/reference/kubectl/cheatsheet/
