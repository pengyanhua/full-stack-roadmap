# GitOps 实践

## 目录
- [GitOps 概述](#gitops-概述)
- [ArgoCD 实战](#argocd-实战)
- [Flux CD](#flux-cd)
- [GitOps 工作流](#gitops-工作流)
- [多集群管理](#多集群管理)
- [最佳实践](#最佳实践)
- [实战案例](#实战案例)

---

## GitOps 概述

### 什么是 GitOps

```
传统运维模式                          GitOps 模式
┌────────────────┐                   ┌────────────────┐
│  手动运维      │                   │  声明式配置    │
│  ├─ 人工执行   │                   │  ├─ Git 仓库   │
│  ├─ 文档记录   │                   │  ├─ 自动同步   │
│  ├─ 难以追溯   │     ────────▶     │  ├─ 自动对齐   │
│  └─ 配置漂移   │                   │  └─ 可审计     │
└────────────────┘                   └────────────────┘
```

### GitOps 四大原则

```
┌──────────────────────────────────────────────────────┐
│              GitOps 四大核心原则                     │
├──────────────────────────────────────────────────────┤
│                                                      │
│ 1️⃣ 声明式 (Declarative)                             │
│    系统状态用声明式配置描述（YAML/JSON）             │
│                                                      │
│    ✅ Kubernetes YAML                                │
│    ✅ Terraform HCL                                  │
│    ✅ Helm Charts                                    │
│                                                      │
│ 2️⃣ 版本控制 (Versioned & Immutable)                 │
│    所有配置存储在 Git 仓库                           │
│                                                      │
│    ✅ 完整历史记录                                   │
│    ✅ 可回滚到任意版本                               │
│    ✅ 代码审查流程                                   │
│                                                      │
│ 3️⃣ 自动拉取 (Pulled Automatically)                  │
│    Agent 主动从 Git 拉取并应用配置                   │
│                                                      │
│    ✅ 无需集群外部访问权限                           │
│    ✅ 自动检测配置变更                               │
│    ✅ 自动同步到目标状态                             │
│                                                      │
│ 4️⃣ 持续对齐 (Continuously Reconciled)               │
│    自动检测并修复配置漂移                            │
│                                                      │
│    ✅ 实时监控实际状态                               │
│    ✅ 自动修复差异                                   │
│    ✅ 告警通知                                       │
└──────────────────────────────────────────────────────┘
```

### GitOps 工作流程

```
┌──────────────────────────────────────────────────────┐
│                 GitOps 工作流                        │
├──────────────────────────────────────────────────────┤
│                                                      │
│  开发者                                              │
│     │                                                │
│     ├─ 1. 修改配置文件                               │
│     │   (deployment.yaml)                           │
│     │                                                │
│     ├─ 2. 提交到 Git 仓库                           │
│     │   (git push)                                  │
│     │                                                │
│     ▼                                                │
│  Git 仓库 (Source of Truth)                         │
│     │                                                │
│     ├─ manifests/                                   │
│     ├── apps/                                       │
│     └── infra/                                      │
│     │                                                │
│     │ 3. Webhook 通知                                │
│     ▼                                                │
│  GitOps Operator                                    │
│  (ArgoCD / Flux)                                    │
│     │                                                │
│     ├─ 4. 检测配置变更                               │
│     │                                                │
│     ├─ 5. 拉取最新配置                               │
│     │                                                │
│     ├─ 6. 对比期望状态与实际状态                     │
│     │                                                │
│     ▼                                                │
│  Kubernetes Cluster                                 │
│     │                                                │
│     ├─ 7. 应用配置变更                               │
│     │   (kubectl apply)                             │
│     │                                                │
│     ├─ 8. 验证部署状态                               │
│     │                                                │
│     └─ 9. 持续监控与对齐                             │
└──────────────────────────────────────────────────────┘
```

---

## ArgoCD 实战

### ArgoCD 安装

```bash
# 创建命名空间
kubectl create namespace argocd

# 安装 ArgoCD
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# 暴露 ArgoCD Server (LoadBalancer)
kubectl patch svc argocd-server -n argocd -p '{"spec": {"type": "LoadBalancer"}}'

# 或使用 Port Forward
kubectl port-forward svc/argocd-server -n argocd 8080:443

# 获取初始密码
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d

# 安装 ArgoCD CLI
curl -sSL -o argocd https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-amd64
chmod +x argocd
sudo mv argocd /usr/local/bin/

# 登录 ArgoCD
argocd login localhost:8080 --insecure
argocd account update-password
```

### ArgoCD Application 配置

```yaml
# application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  # 项目
  project: default

  # Git 仓库配置
  source:
    repoURL: https://github.com/myorg/myapp-config.git
    targetRevision: main
    path: manifests/production

    # Helm 配置 (可选)
    helm:
      releaseName: myapp
      values: |
        replicaCount: 3
        image:
          repository: myapp
          tag: v1.2.3
      valueFiles:
        - values-production.yaml

    # Kustomize 配置 (可选)
    kustomize:
      namePrefix: prod-
      commonLabels:
        env: production
      images:
        - myapp=myapp:v1.2.3

  # 目标集群
  destination:
    server: https://kubernetes.default.svc
    namespace: production

  # 同步策略
  syncPolicy:
    automated:
      prune: true        # 自动删除不在 Git 中的资源
      selfHeal: true     # 自动修复配置漂移
      allowEmpty: false  # 不允许空目录

    syncOptions:
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
      - PruneLast=true

    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m

  # 健康检查
  health:
    timeout: 300

  # 忽略差异
  ignoreDifferences:
    - group: apps
      kind: Deployment
      jsonPointers:
        - /spec/replicas  # 忽略副本数差异(HPA 控制)

  # 通知
  notifications:
    - on-sync-succeeded
    - on-sync-failed
    - on-health-degraded
```

### AppProject 多租户配置

```yaml
# appproject.yaml
apiVersion: argoproj.io/v1alpha1
kind: AppProject
metadata:
  name: team-a
  namespace: argocd
spec:
  description: Team A Project

  # 允许的 Git 仓库
  sourceRepos:
    - https://github.com/myorg/team-a-*

  # 允许的目标集群
  destinations:
    - namespace: team-a-*
      server: https://kubernetes.default.svc
    - namespace: team-a-shared
      server: https://prod-cluster.example.com

  # 允许的资源类型
  clusterResourceWhitelist:
    - group: ''
      kind: Namespace
    - group: 'rbac.authorization.k8s.io'
      kind: RoleBinding

  namespaceResourceWhitelist:
    - group: '*'
      kind: '*'

  # 拒绝的资源类型
  namespaceResourceBlacklist:
    - group: ''
      kind: ResourceQuota
    - group: ''
      kind: LimitRange

  # RBAC 角色
  roles:
    - name: developer
      description: Developer role
      policies:
        - p, proj:team-a:developer, applications, get, team-a/*, allow
        - p, proj:team-a:developer, applications, sync, team-a/*, allow
      groups:
        - team-a-developers

    - name: admin
      description: Admin role
      policies:
        - p, proj:team-a:admin, applications, *, team-a/*, allow
      groups:
        - team-a-admins
```

### ArgoCD CLI 常用命令

```bash
# 创建应用
argocd app create myapp \
  --repo https://github.com/myorg/myapp-config.git \
  --path manifests/production \
  --dest-server https://kubernetes.default.svc \
  --dest-namespace production \
  --sync-policy automated \
  --auto-prune \
  --self-heal

# 查看应用状态
argocd app list
argocd app get myapp
argocd app diff myapp

# 同步应用
argocd app sync myapp
argocd app sync myapp --force
argocd app sync myapp --dry-run

# 回滚应用
argocd app rollback myapp
argocd app history myapp

# 查看日志
argocd app logs myapp
argocd app logs myapp --follow

# 删除应用
argocd app delete myapp
argocd app delete myapp --cascade=false  # 只删除应用定义,保留资源
```

---

## Flux CD

### Flux 安装

```bash
# 安装 Flux CLI
curl -s https://fluxcd.io/install.sh | sudo bash

# 检查集群兼容性
flux check --pre

# Bootstrap Flux
flux bootstrap github \
  --owner=myorg \
  --repository=fleet-infra \
  --branch=main \
  --path=clusters/production \
  --personal \
  --token-auth

# 验证安装
flux check
kubectl get pods -n flux-system
```

### GitRepository 配置

```yaml
# gitrepository.yaml
apiVersion: source.toolkit.fluxcd.io/v1
kind: GitRepository
metadata:
  name: myapp
  namespace: flux-system
spec:
  interval: 1m
  url: https://github.com/myorg/myapp-config
  ref:
    branch: main
  secretRef:
    name: github-credentials

  # Git 验证
  verify:
    mode: head
    secretRef:
      name: git-pgp-key

  # 忽略路径
  ignore: |
    /.github/
    /docs/
```

### Kustomization 配置

```yaml
# kustomization.yaml
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: myapp
  namespace: flux-system
spec:
  interval: 10m
  retryInterval: 1m
  timeout: 5m

  sourceRef:
    kind: GitRepository
    name: myapp

  path: ./manifests/production

  prune: true
  wait: true
  force: false

  # 健康检查
  healthChecks:
    - apiVersion: apps/v1
      kind: Deployment
      name: myapp
      namespace: production

  # 依赖关系
  dependsOn:
    - name: infrastructure
    - name: databases

  # 变量替换
  postBuild:
    substitute:
      CLUSTER_NAME: production
      REGION: us-east-1
    substituteFrom:
      - kind: ConfigMap
        name: cluster-settings

  # 解密
  decryption:
    provider: sops
    secretRef:
      name: sops-gpg
```

### HelmRelease 配置

```yaml
# helmrelease.yaml
apiVersion: helm.toolkit.fluxcd.io/v2beta1
kind: HelmRelease
metadata:
  name: nginx-ingress
  namespace: flux-system
spec:
  interval: 10m
  timeout: 5m

  chart:
    spec:
      chart: nginx-ingress
      version: 4.x.x
      sourceRef:
        kind: HelmRepository
        name: nginx-stable
        namespace: flux-system
      interval: 1m

  install:
    remediation:
      retries: 3

  upgrade:
    remediation:
      retries: 3
      remediateLastFailure: true

  rollback:
    recreate: true
    force: true

  values:
    controller:
      replicaCount: 3
      service:
        type: LoadBalancer
      metrics:
        enabled: true

  valuesFrom:
    - kind: ConfigMap
      name: nginx-values
      valuesKey: values.yaml
```

---

## GitOps 工作流

### 多环境 Git 分支策略

```
┌────────────────────────────────────────────────┐
│           GitOps 分支策略对比                  │
├────────────────────────────────────────────────┤
│                                                │
│ 策略 1: 环境分支 (Environment Branches)       │
│                                                │
│  main                                          │
│   ├── dev                                      │
│   ├── staging                                  │
│   └── production                               │
│                                                │
│  优点: 环境隔离清晰                            │
│  缺点: 跨环境合并复杂                          │
│                                                │
│ ────────────────────────────────────────────  │
│                                                │
│ 策略 2: 目录分离 (Directory Structure)        │
│                                                │
│  main/                                         │
│   ├── environments/                            │
│   │   ├── dev/                                 │
│   │   ├── staging/                             │
│   │   └── production/                          │
│   └── base/                                    │
│                                                │
│  优点: 单一分支,易管理                         │
│  缺点: 所有环境在同一分支                      │
│                                                │
│ ────────────────────────────────────────────  │
│                                                │
│ 策略 3: 仓库分离 (Repository Per Environment) │
│                                                │
│  myapp-dev          (repo)                    │
│  myapp-staging      (repo)                    │
│  myapp-production   (repo)                    │
│                                                │
│  优点: 最高隔离度,安全                         │
│  缺点: 管理复杂                                │
└────────────────────────────────────────────────┘
```

### 推荐目录结构

```
gitops-repo/
├── apps/                      # 应用配置
│   ├── base/                  # 基础配置
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── kustomization.yaml
│   ├── overlays/              # 环境差异
│   │   ├── dev/
│   │   │   ├── kustomization.yaml
│   │   │   └── patch.yaml
│   │   ├── staging/
│   │   │   └── kustomization.yaml
│   │   └── production/
│   │       ├── kustomization.yaml
│   │       └── hpa.yaml
│   └── argocd/                # ArgoCD 应用定义
│       ├── dev.yaml
│       ├── staging.yaml
│       └── production.yaml
│
├── infrastructure/            # 基础设施
│   ├── networking/
│   │   ├── ingress-nginx.yaml
│   │   └── cert-manager.yaml
│   ├── monitoring/
│   │   ├── prometheus.yaml
│   │   └── grafana.yaml
│   └── security/
│       └── vault.yaml
│
└── clusters/                  # 集群配置
    ├── dev/
    │   ├── flux-system/
    │   └── apps.yaml
    ├── staging/
    └── production/
```

---

## 多集群管理

### ArgoCD 多集群配置

```bash
# 添加外部集群
argocd cluster add prod-cluster \
  --kubeconfig ~/.kube/config \
  --name production

# 查看集群列表
argocd cluster list

# 多集群应用部署
cat <<EOF | kubectl apply -f -
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: myapp-multicluster
  namespace: argocd
spec:
  generators:
    - list:
        elements:
          - cluster: dev
            url: https://dev.k8s.local
            namespace: myapp-dev
          - cluster: staging
            url: https://staging.k8s.local
            namespace: myapp-staging
          - cluster: production
            url: https://prod.k8s.local
            namespace: myapp-prod

  template:
    metadata:
      name: 'myapp-{{cluster}}'
    spec:
      project: default
      source:
        repoURL: https://github.com/myorg/myapp-config.git
        targetRevision: main
        path: 'environments/{{cluster}}'
      destination:
        server: '{{url}}'
        namespace: '{{namespace}}'
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
EOF
```

### Flux 多集群管理

```yaml
# clusters/production/infrastructure.yaml
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: infrastructure
  namespace: flux-system
spec:
  interval: 10m
  sourceRef:
    kind: GitRepository
    name: fleet-infra
  path: ./infrastructure/production
  prune: true

---
# clusters/production/apps.yaml
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: apps
  namespace: flux-system
spec:
  interval: 5m
  dependsOn:
    - name: infrastructure
  sourceRef:
    kind: GitRepository
    name: fleet-infra
  path: ./apps/production
  prune: true
```

---

## 最佳实践

### 密钥管理

```yaml
# 使用 Sealed Secrets
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: db-credentials
  namespace: production
spec:
  encryptedData:
    password: AgBx5QvZ8... # 加密后的密钥
    username: AgCq3Km...

# 或使用 SOPS + Age
# secrets.enc.yaml
apiVersion: v1
kind: Secret
metadata:
    name: db-credentials
type: Opaque
data:
    password: ENC[AES256_GCM,data:xxx,iv:yyy,tag:zzz,type:str]
    username: ENC[AES256_GCM,data:aaa,iv:bbb,tag:ccc,type:str]
sops:
    age:
        - recipient: age1xxx...
          enc: |
            -----BEGIN AGE ENCRYPTED FILE-----
            xxx
            -----END AGE ENCRYPTED FILE-----
```

### PR 预览环境

```yaml
# applicationset-pr-preview.yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: pr-preview
  namespace: argocd
spec:
  generators:
    - pullRequest:
        github:
          owner: myorg
          repo: myapp
          tokenRef:
            secretName: github-token
            key: token
        requeueAfterSeconds: 60

  template:
    metadata:
      name: 'pr-{{number}}'
    spec:
      project: default
      source:
        repoURL: https://github.com/myorg/myapp.git
        targetRevision: '{{head_sha}}'
        path: manifests
        helm:
          values: |
            ingress:
              host: pr-{{number}}.preview.example.com
      destination:
        server: https://kubernetes.default.svc
        namespace: 'pr-{{number}}'
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
        syncOptions:
          - CreateNamespace=true
```

---

## 实战案例

### 案例 1: 灰度发布

```yaml
# rollout.yaml - Argo Rollouts
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: myapp
spec:
  replicas: 10
  strategy:
    canary:
      steps:
        - setWeight: 10
        - pause: {duration: 5m}
        - setWeight: 30
        - pause: {duration: 10m}
        - setWeight: 50
        - pause: {duration: 10m}
      analysis:
        templates:
          - templateName: success-rate
        startingStep: 2
      trafficRouting:
        nginx:
          stableIngress: myapp
          additionalIngressAnnotations:
            canary-by-header: X-Canary

  revisionHistoryLimit: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
        - name: myapp
          image: myapp:latest

---
# analysis-template.yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
spec:
  metrics:
    - name: success-rate
      interval: 1m
      successCondition: result >= 0.95
      provider:
        prometheus:
          address: http://prometheus:9090
          query: |
            sum(rate(http_requests_total{status!~"5.."}[5m]))
            /
            sum(rate(http_requests_total[5m]))
```

### 案例 2: 自动化镜像更新

```yaml
# image-update-automation.yaml (Flux)
apiVersion: image.toolkit.fluxcd.io/v1beta1
kind: ImageRepository
metadata:
  name: myapp
  namespace: flux-system
spec:
  image: ghcr.io/myorg/myapp
  interval: 1m

---
apiVersion: image.toolkit.fluxcd.io/v1beta1
kind: ImagePolicy
metadata:
  name: myapp
  namespace: flux-system
spec:
  imageRepositoryRef:
    name: myapp
  policy:
    semver:
      range: 1.x.x

---
apiVersion: image.toolkit.fluxcd.io/v1beta1
kind: ImageUpdateAutomation
metadata:
  name: myapp
  namespace: flux-system
spec:
  interval: 1m
  sourceRef:
    kind: GitRepository
    name: fleet-infra
  git:
    checkout:
      ref:
        branch: main
    commit:
      author:
        email: fluxcd@example.com
        name: Flux Bot
      messageTemplate: |
        Auto-update image {{range .Updated.Images}}{{println .}}{{end}}
    push:
      branch: main
  update:
    path: ./apps/production
    strategy: Setters
```

---

## 总结

### GitOps vs 传统 CD

```
┌────────────────────────────────────────────────┐
│          GitOps vs Traditional CD              │
├──────────────┬──────────────┬─────────────────┤
│   特性       │  传统 CD     │    GitOps       │
├──────────────┼──────────────┼─────────────────┤
│ 部署触发     │  Push 模式   │   Pull 模式     │
│ 配置存储     │  CI 变量     │   Git 仓库      │
│ 权限管理     │  集群凭证    │   只读 Git      │
│ 配置漂移     │  难以发现    │   自动修复      │
│ 审计追溯     │  日志        │   Git 历史      │
│ 回滚         │  重新部署    │   Git Revert    │
│ 多集群       │  复杂        │   声明式        │
│ 安全性       │  中等        │   高            │
└──────────────┴──────────────┴─────────────────┘
```

### 关键要点

1. **声明式配置**: 一切皆代码,存储在 Git
2. **自动化**: 自动检测、同步、修复
3. **可审计**: 完整的变更历史
4. **安全**: 减少集群访问权限

### 下一步学习

- [03_infrastructure_as_code.md](03_infrastructure_as_code.md) - 基础设施即代码
- [04_deployment_strategies.md](04_deployment_strategies.md) - 部署策略
- [05_release_management.md](05_release_management.md) - 版本管理
