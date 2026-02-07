# GitOps 实践

## 目录
- [GitOps概述](#gitops概述)
- [ArgoCD实战](#argocd实战)
- [Flux CD实战](#flux-cd实战)
- [最佳实践](#最佳实践)

---

## GitOps概述

### GitOps核心原则

```
┌────────────────────────────────────────────────────┐
│              GitOps 工作流程                       │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌──────────┐                                     │
│  │   开发   │                                     │
│  └────┬─────┘                                     │
│       │ git push                                   │
│  ┌────▼─────────┐                                 │
│  │  Git Repo    │  (Single Source of Truth)      │
│  │  (YAML配置)  │                                 │
│  └────┬─────────┘                                 │
│       │ webhook/poll                               │
│  ┌────▼─────────┐                                 │
│  │  GitOps      │                                 │
│  │  Controller  │  (ArgoCD/Flux)                  │
│  │  (K8s内运行) │                                 │
│  └────┬─────────┘                                 │
│       │ kubectl apply                              │
│  ┌────▼─────────┐                                 │
│  │  Kubernetes  │                                 │
│  │  Cluster     │                                 │
│  └──────────────┘                                 │
│                                                    │
│  自动同步、自动修复、版本可追溯                   │
└────────────────────────────────────────────────────┘
```

## ArgoCD实战

### 安装ArgoCD

```bash
# 创建命名空间
kubectl create namespace argocd

# 安装ArgoCD
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# 暴露ArgoCD服务
kubectl patch svc argocd-server -n argocd -p '{"spec": {"type": "LoadBalancer"}}'

# 获取初始密码
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
```

### ArgoCD Application配置

```yaml
# application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/myorg/myapp-config
    targetRevision: HEAD
    path: k8s/production
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true      # 自动删除
      selfHeal: true   # 自动修复
      allowEmpty: false
    syncOptions:
    - CreateNamespace=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

### 多环境配置

```yaml
# apps/dev.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp-dev
spec:
  source:
    path: overlays/dev
    helm:
      values: |
        replicas: 1
        resources:
          requests:
            cpu: 100m
            memory: 128Mi

# apps/prod.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp-prod
spec:
  source:
    path: overlays/prod
    helm:
      values: |
        replicas: 3
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
```

## Flux CD实战

### 安装Flux

```bash
# 安装Flux CLI
curl -s https://fluxcd.io/install.sh | sudo bash

# Bootstrap Flux
flux bootstrap github \
  --owner=myorg \
  --repository=fleet-infra \
  --branch=main \
  --path=clusters/production \
  --personal

# 查看状态
flux get all
```

### Flux配置

```yaml
# apps/kustomization.yaml
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: myapp
  namespace: flux-system
spec:
  interval: 10m
  path: ./k8s/production
  prune: true
  sourceRef:
    kind: GitRepository
    name: myapp
  healthChecks:
    - apiVersion: apps/v1
      kind: Deployment
      name: myapp
      namespace: production
```

## 总结

GitOps优势：
- 声明式配置
- 版本控制
- 自动同步
- 易于回滚
