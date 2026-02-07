# DevOps 实践指南

## 目录

```
DevOps/
├── 01_cicd_pipeline.md          # CI/CD 流水线设计
├── 02_gitops.md                 # GitOps 实践（ArgoCD/Flux）
├── 03_infrastructure_as_code.md # IaC（Terraform/Ansible）
├── 04_deployment_strategies.md  # 部署策略（蓝绿/金丝雀/滚动）
└── 05_release_management.md     # 发布管理与回滚
```

## DevOps 核心理念

```
┌────────────────────────────────────────────────────┐
│              DevOps 无限循环                       │
├────────────────────────────────────────────────────┤
│                                                    │
│         Plan → Code → Build → Test                │
│           ↑                         ↓              │
│        Monitor                    Release          │
│           ↑                         ↓              │
│        Operate ← Deploy ← Configure                │
│                                                    │
│  核心价值:                                         │
│  • 缩短交付周期                                    │
│  • 提高部署频率                                    │
│  • 降低变更失败率                                  │
│  • 缩短故障恢复时间                                │
└────────────────────────────────────────────────────┘
```

## 学习路径

1. **CI/CD 基础** - 理解持续集成和持续部署
2. **GitOps** - 声明式运维
3. **基础设施即代码** - Terraform/Ansible
4. **部署策略** - 零停机部署
5. **发布管理** - 版本控制与回滚

## 工具链

| 阶段 | 工具 |
|------|------|
| 代码管理 | Git, GitHub/GitLab |
| CI/CD | Jenkins, GitLab CI, GitHub Actions |
| 容器化 | Docker, Kubernetes |
| IaC | Terraform, Ansible, Pulumi |
| 监控 | Prometheus, Grafana, ELK |
| 安全扫描 | SonarQube, Snyk, Trivy |

开始学习 → [01_cicd_pipeline.md](01_cicd_pipeline.md)
