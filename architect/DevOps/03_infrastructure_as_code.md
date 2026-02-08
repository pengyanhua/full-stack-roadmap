# 基础设施即代码 (Infrastructure as Code)

## 目录
- [IaC 概述](#iac-概述)
- [Terraform 实战](#terraform-实战)
- [Ansible 自动化](#ansible-自动化)
- [Pulumi 编程式 IaC](#pulumi-编程式-iac)
- [状态管理](#状态管理)
- [模块化设计](#模块化设计)
- [实战案例](#实战案例)

---

## IaC 概述

### 什么是基础设施即代码

```
传统基础设施管理                    IaC 管理
┌────────────────┐                 ┌────────────────┐
│  手动配置      │                 │  代码定义      │
│  ├─ 控制台操作 │                 │  ├─ 版本控制   │
│  ├─ 文档记录   │    ────────▶    │  ├─ 自动化     │
│  ├─ 难以复现   │                 │  ├─ 可复现     │
│  └─ 配置漂移   │                 │  └─ 一致性     │
└────────────────┘                 └────────────────┘
```

### IaC 工具对比

```
┌──────────────────────────────────────────────────────────────┐
│                    IaC 工具对比矩阵                          │
├──────────┬─────────┬──────────┬──────────┬──────────────────┤
│ 工具     │Terraform│ Ansible  │ Pulumi   │  CloudFormation  │
├──────────┼─────────┼──────────┼──────────┼──────────────────┤
│ 类型     │ 声明式  │ 过程式   │ 编程式   │  声明式          │
│ 语言     │ HCL     │ YAML     │ 多语言   │  JSON/YAML       │
│ 状态管理 │ ✅      │ ❌       │ ✅       │  ✅              │
│ 云无关   │ ✅      │ ✅       │ ✅       │  ❌ (AWS Only)   │
│ 易用性   │ ⭐⭐⭐ │ ⭐⭐⭐⭐ │ ⭐⭐     │  ⭐⭐⭐          │
│ 生态     │ ⭐⭐⭐⭐│ ⭐⭐⭐  │ ⭐⭐     │  ⭐⭐⭐          │
│ 配置管理 │ ❌      │ ✅       │ ❌       │  ❌              │
│ 适用场景 │ 基础设施│ 配置管理 │ 复杂逻辑 │  AWS 原生        │
└──────────┴─────────┴──────────┴──────────┴──────────────────┘
```

---

## Terraform 实战

### Terraform 基础结构

```hcl
# main.tf
terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
  }

  # 远程状态存储
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}

# Provider 配置
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = var.environment
      ManagedBy   = "Terraform"
      Project     = var.project_name
    }
  }
}

# 数据源
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}
```

### VPC 完整配置

```hcl
# vpc.tf
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.project_name}-${var.environment}-vpc"
  cidr = var.vpc_cidr

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = [for i in range(3) : cidrsubnet(var.vpc_cidr, 4, i)]
  public_subnets  = [for i in range(3) : cidrsubnet(var.vpc_cidr, 4, i + 3)]
  database_subnets = [for i in range(3) : cidrsubnet(var.vpc_cidr, 4, i + 6)]

  enable_nat_gateway     = true
  single_nat_gateway     = var.environment != "production"
  enable_dns_hostnames   = true
  enable_dns_support     = true

  # VPC Flow Logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_iam_role  = true
  create_flow_log_cloudwatch_log_group = true

  # 标签
  tags = {
    Name = "${var.project_name}-${var.environment}-vpc"
  }

  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
  }
}

# 安全组
resource "aws_security_group" "app" {
  name_prefix = "${var.project_name}-app-"
  description = "Security group for application tier"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = module.vpc.private_subnets_cidr_blocks
    description = "Allow HTTP from private subnets"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound traffic"
  }

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "${var.project_name}-app-sg"
  }
}
```

### EKS 集群配置

```hcl
# eks.tf
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "${var.project_name}-${var.environment}"
  cluster_version = "1.28"

  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  # 加密配置
  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.eks.arn
    resources        = ["secrets"]
  }

  # VPC 配置
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # 集群附加组件
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  # 节点组
  eks_managed_node_groups = {
    general = {
      min_size     = 2
      max_size     = 10
      desired_size = 3

      instance_types = ["t3.large"]
      capacity_type  = "ON_DEMAND"

      labels = {
        role = "general"
      }

      taints = []

      update_config = {
        max_unavailable_percentage = 50
      }
    }

    spot = {
      min_size     = 0
      max_size     = 10
      desired_size = 2

      instance_types = ["t3.large", "t3a.large"]
      capacity_type  = "SPOT"

      labels = {
        role = "spot"
      }

      taints = [{
        key    = "spot"
        value  = "true"
        effect = "NoSchedule"
      }]
    }
  }

  # IRSA (IAM Roles for Service Accounts)
  enable_irsa = true

  # Cluster security group rules
  cluster_security_group_additional_rules = {
    egress_nodes_ephemeral_ports_tcp = {
      description                = "To node 1025-65535"
      protocol                   = "tcp"
      from_port                  = 1025
      to_port                    = 65535
      type                       = "egress"
      source_node_security_group = true
    }
  }

  # aws-auth configmap
  manage_aws_auth_configmap = true

  aws_auth_roles = [
    {
      rolearn  = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/Admin"
      username = "admin"
      groups   = ["system:masters"]
    }
  ]

  tags = local.common_tags
}

# KMS Key for EKS
resource "aws_kms_key" "eks" {
  description             = "EKS Secret Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = {
    Name = "${var.project_name}-eks-key"
  }
}

resource "aws_kms_alias" "eks" {
  name          = "alias/${var.project_name}-eks"
  target_key_id = aws_kms_key.eks.key_id
}
```

### RDS 数据库配置

```hcl
# rds.tf
module "db" {
  source  = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  identifier = "${var.project_name}-${var.environment}-db"

  engine               = "postgres"
  engine_version       = "15.3"
  family               = "postgres15"
  major_engine_version = "15"
  instance_class       = var.environment == "production" ? "db.r6g.xlarge" : "db.t3.medium"

  allocated_storage     = 100
  max_allocated_storage = 500
  storage_encrypted     = true
  kms_key_id            = aws_kms_key.rds.arn

  db_name  = var.db_name
  username = var.db_username
  port     = 5432

  # 密码从 Secrets Manager 获取
  manage_master_user_password = true

  multi_az               = var.environment == "production"
  db_subnet_group_name   = module.vpc.database_subnet_group_name
  vpc_security_group_ids = [aws_security_group.rds.id]

  # 备份配置
  backup_retention_period = var.environment == "production" ? 30 : 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"

  # 性能洞察
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  performance_insights_enabled    = true
  performance_insights_retention_period = 7

  # 参数组
  parameters = [
    {
      name  = "log_connections"
      value = "1"
    },
    {
      name  = "log_disconnections"
      value = "1"
    },
    {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements"
    }
  ]

  # 只读副本
  create_db_instance        = true
  create_db_parameter_group = true

  tags = local.common_tags
}

# 只读副本
resource "aws_db_instance" "read_replica" {
  count = var.environment == "production" ? 2 : 0

  identifier             = "${var.project_name}-${var.environment}-db-replica-${count.index + 1}"
  replicate_source_db    = module.db.db_instance_id
  instance_class         = "db.r6g.large"
  publicly_accessible    = false
  skip_final_snapshot    = true
  performance_insights_enabled = true

  tags = merge(
    local.common_tags,
    {
      Name = "${var.project_name}-read-replica-${count.index + 1}"
    }
  )
}

# RDS 安全组
resource "aws_security_group" "rds" {
  name_prefix = "${var.project_name}-rds-"
  description = "Security group for RDS"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
    description     = "Allow PostgreSQL from application"
  }

  tags = {
    Name = "${var.project_name}-rds-sg"
  }
}

# KMS Key for RDS
resource "aws_kms_key" "rds" {
  description             = "RDS Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true
}
```

### 变量和输出

```hcl
# variables.tf
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "project_name" {
  description = "Project name"
  type        = string
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "myapp"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "dbadmin"
  sensitive   = true
}

# outputs.tf
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
  sensitive   = true
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = module.db.db_instance_endpoint
  sensitive   = true
}

output "rds_replica_endpoints" {
  description = "RDS read replica endpoints"
  value       = aws_db_instance.read_replica[*].endpoint
  sensitive   = true
}

# locals.tf
locals {
  common_tags = {
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "Terraform"
    Owner       = "DevOps Team"
  }

  azs = slice(data.aws_availability_zones.available.names, 0, 3)
}
```

### Terraform 最佳实践

```bash
# terraform.sh - Terraform wrapper script
#!/bin/bash

set -euo pipefail

ENV=${1:-dev}
ACTION=${2:-plan}

# 初始化
terraform init \
  -backend-config="key=${ENV}/terraform.tfstate" \
  -reconfigure

# 选择或创建 workspace
terraform workspace select ${ENV} || terraform workspace new ${ENV}

# 验证配置
terraform validate

# 格式化检查
terraform fmt -check -recursive

# 安全扫描
tfsec .

# 成本估算
infracost breakdown --path .

case ${ACTION} in
  plan)
    terraform plan \
      -var-file="environments/${ENV}.tfvars" \
      -out="${ENV}.tfplan"
    ;;

  apply)
    terraform apply \
      -var-file="environments/${ENV}.tfvars" \
      -auto-approve \
      "${ENV}.tfplan"
    ;;

  destroy)
    echo "Are you sure you want to destroy ${ENV}? (yes/no)"
    read -r confirm
    if [ "$confirm" = "yes" ]; then
      terraform destroy \
        -var-file="environments/${ENV}.tfvars" \
        -auto-approve
    fi
    ;;

  *)
    echo "Unknown action: ${ACTION}"
    exit 1
    ;;
esac
```

---

## Ansible 自动化

### Ansible Inventory

```ini
# inventory/production.ini
[web]
web1.example.com ansible_host=10.0.1.10
web2.example.com ansible_host=10.0.1.11
web3.example.com ansible_host=10.0.1.12

[db]
db1.example.com ansible_host=10.0.2.10
db2.example.com ansible_host=10.0.2.11

[web:vars]
ansible_user=ubuntu
ansible_ssh_private_key_file=~/.ssh/production.pem

[db:vars]
ansible_user=ubuntu
ansible_ssh_private_key_file=~/.ssh/production.pem

[production:children]
web
db

[production:vars]
environment=production
```

### Ansible Playbook

```yaml
# playbook.yml
---
- name: Setup Web Servers
  hosts: web
  become: yes
  vars:
    app_name: myapp
    app_version: "1.2.3"
    app_port: 8080

  pre_tasks:
    - name: Update apt cache
      apt:
        update_cache: yes
        cache_valid_time: 3600

  roles:
    - common
    - nginx
    - nodejs
    - monitoring

  tasks:
    - name: Create application directory
      file:
        path: "/opt/{{ app_name }}"
        state: directory
        owner: www-data
        group: www-data
        mode: '0755'

    - name: Deploy application
      copy:
        src: "dist/{{ app_name }}-{{ app_version }}.tar.gz"
        dest: "/opt/{{ app_name }}/"
      notify: restart application

    - name: Extract application
      unarchive:
        src: "/opt/{{ app_name }}/{{ app_name }}-{{ app_version }}.tar.gz"
        dest: "/opt/{{ app_name }}/"
        remote_src: yes

    - name: Install dependencies
      npm:
        path: "/opt/{{ app_name }}"
        production: yes

    - name: Configure systemd service
      template:
        src: templates/myapp.service.j2
        dest: /etc/systemd/system/{{ app_name }}.service
      notify:
        - reload systemd
        - restart application

    - name: Enable and start application
      systemd:
        name: "{{ app_name }}"
        enabled: yes
        state: started

  handlers:
    - name: reload systemd
      systemd:
        daemon_reload: yes

    - name: restart application
      systemd:
        name: "{{ app_name }}"
        state: restarted

---
- name: Setup Database Servers
  hosts: db
  become: yes

  roles:
    - common
    - postgresql
    - backup

  tasks:
    - name: Create database
      postgresql_db:
        name: "{{ db_name }}"
        encoding: UTF-8
        lc_collate: en_US.UTF-8
        lc_ctype: en_US.UTF-8

    - name: Create database user
      postgresql_user:
        name: "{{ db_user }}"
        password: "{{ db_password }}"
        db: "{{ db_name }}"
        priv: ALL

    - name: Configure PostgreSQL
      template:
        src: templates/postgresql.conf.j2
        dest: /etc/postgresql/15/main/postgresql.conf
      notify: restart postgresql

  handlers:
    - name: restart postgresql
      systemd:
        name: postgresql
        state: restarted
```

### Ansible Role 示例

```yaml
# roles/nginx/tasks/main.yml
---
- name: Install Nginx
  apt:
    name: nginx
    state: present

- name: Create SSL directory
  file:
    path: /etc/nginx/ssl
    state: directory
    mode: '0700'

- name: Copy SSL certificate
  copy:
    src: "{{ ssl_cert_file }}"
    dest: /etc/nginx/ssl/cert.pem
    mode: '0600'
  when: ssl_enabled | default(false)

- name: Configure Nginx
  template:
    src: nginx.conf.j2
    dest: /etc/nginx/nginx.conf
    validate: 'nginx -t -c %s'
  notify: reload nginx

- name: Configure site
  template:
    src: site.conf.j2
    dest: "/etc/nginx/sites-available/{{ app_name }}"
  notify: reload nginx

- name: Enable site
  file:
    src: "/etc/nginx/sites-available/{{ app_name }}"
    dest: "/etc/nginx/sites-enabled/{{ app_name }}"
    state: link
  notify: reload nginx

- name: Start Nginx
  systemd:
    name: nginx
    enabled: yes
    state: started

# roles/nginx/handlers/main.yml
---
- name: reload nginx
  systemd:
    name: nginx
    state: reloaded

# roles/nginx/templates/site.conf.j2
upstream {{ app_name }} {
    {% for host in groups['web'] %}
    server {{ hostvars[host]['ansible_host'] }}:{{ app_port }} max_fails=3 fail_timeout=30s;
    {% endfor %}
}

server {
    listen 80;
    server_name {{ domain_name }};

    {% if ssl_enabled | default(false) %}
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name {{ domain_name }};

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    {% endif %}

    location / {
        proxy_pass http://{{ app_name }};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /health {
        access_log off;
        return 200 "healthy\n";
    }
}
```

---

## Pulumi 编程式 IaC

### Pulumi Python 示例

```python
# __main__.py
import pulumi
import pulumi_aws as aws
import pulumi_kubernetes as k8s

# 配置
config = pulumi.Config()
environment = pulumi.get_stack()
project_name = pulumi.get_project()

# VPC
vpc = aws.ec2.Vpc(
    f"{project_name}-vpc",
    cidr_block="10.0.0.0/16",
    enable_dns_hostnames=True,
    enable_dns_support=True,
    tags={
        "Name": f"{project_name}-{environment}-vpc",
        "Environment": environment,
    }
)

# 子网
azs = aws.get_availability_zones(state="available")
public_subnets = []
private_subnets = []

for i, az in enumerate(azs.names[:3]):
    # 公有子网
    public_subnet = aws.ec2.Subnet(
        f"public-subnet-{i}",
        vpc_id=vpc.id,
        cidr_block=f"10.0.{i}.0/24",
        availability_zone=az,
        map_public_ip_on_launch=True,
        tags={
            "Name": f"{project_name}-public-{i}",
            "kubernetes.io/role/elb": "1",
        }
    )
    public_subnets.append(public_subnet)

    # 私有子网
    private_subnet = aws.ec2.Subnet(
        f"private-subnet-{i}",
        vpc_id=vpc.id,
        cidr_block=f"10.0.{i+10}.0/24",
        availability_zone=az,
        tags={
            "Name": f"{project_name}-private-{i}",
            "kubernetes.io/role/internal-elb": "1",
        }
    )
    private_subnets.append(private_subnet)

# Internet Gateway
igw = aws.ec2.InternetGateway(
    f"{project_name}-igw",
    vpc_id=vpc.id,
    tags={"Name": f"{project_name}-igw"}
)

# EKS 集群
eks_role = aws.iam.Role(
    f"{project_name}-eks-role",
    assume_role_policy="""{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "eks.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }"""
)

aws.iam.RolePolicyAttachment(
    f"{project_name}-eks-cluster-policy",
    role=eks_role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
)

eks_cluster = aws.eks.Cluster(
    f"{project_name}-cluster",
    role_arn=eks_role.arn,
    vpc_config=aws.eks.ClusterVpcConfigArgs(
        subnet_ids=[s.id for s in private_subnets],
    ),
    enabled_cluster_log_types=[
        "api",
        "audit",
        "authenticator",
        "controllerManager",
        "scheduler",
    ],
)

# RDS 实例
db_subnet_group = aws.rds.SubnetGroup(
    f"{project_name}-db-subnet",
    subnet_ids=[s.id for s in private_subnets],
    tags={"Name": f"{project_name}-db-subnet"}
)

db_instance = aws.rds.Instance(
    f"{project_name}-db",
    engine="postgres",
    engine_version="15.3",
    instance_class="db.t3.medium" if environment != "production" else "db.r6g.xlarge",
    allocated_storage=100,
    storage_encrypted=True,
    db_name=config.require("db_name"),
    username=config.require("db_username"),
    password=config.require_secret("db_password"),
    db_subnet_group_name=db_subnet_group.name,
    multi_az=environment == "production",
    skip_final_snapshot=environment != "production",
    backup_retention_period=30 if environment == "production" else 7,
    tags={
        "Name": f"{project_name}-{environment}-db",
        "Environment": environment,
    }
)

# Kubernetes Provider
k8s_provider = k8s.Provider(
    "k8s-provider",
    kubeconfig=eks_cluster.kubeconfig_json,
)

# 部署应用
app_namespace = k8s.core.v1.Namespace(
    "app",
    metadata=k8s.meta.v1.ObjectMetaArgs(
        name="app"
    ),
    opts=pulumi.ResourceOptions(provider=k8s_provider)
)

app_deployment = k8s.apps.v1.Deployment(
    "app-deployment",
    metadata=k8s.meta.v1.ObjectMetaArgs(
        namespace=app_namespace.metadata["name"],
        labels={"app": "myapp"},
    ),
    spec=k8s.apps.v1.DeploymentSpecArgs(
        replicas=3 if environment == "production" else 1,
        selector=k8s.meta.v1.LabelSelectorArgs(
            match_labels={"app": "myapp"}
        ),
        template=k8s.core.v1.PodTemplateSpecArgs(
            metadata=k8s.meta.v1.ObjectMetaArgs(
                labels={"app": "myapp"}
            ),
            spec=k8s.core.v1.PodSpecArgs(
                containers=[k8s.core.v1.ContainerArgs(
                    name="app",
                    image=config.require("app_image"),
                    ports=[k8s.core.v1.ContainerPortArgs(
                        container_port=8080
                    )],
                    env=[
                        k8s.core.v1.EnvVarArgs(
                            name="DB_HOST",
                            value=db_instance.endpoint,
                        ),
                        k8s.core.v1.EnvVarArgs(
                            name="DB_NAME",
                            value=db_instance.db_name,
                        ),
                    ],
                    resources=k8s.core.v1.ResourceRequirementsArgs(
                        requests={
                            "cpu": "100m",
                            "memory": "128Mi",
                        },
                        limits={
                            "cpu": "500m",
                            "memory": "512Mi",
                        },
                    ),
                )],
            ),
        ),
    ),
    opts=pulumi.ResourceOptions(provider=k8s_provider)
)

# 导出
pulumi.export("vpc_id", vpc.id)
pulumi.export("eks_cluster_name", eks_cluster.name)
pulumi.export("eks_cluster_endpoint", eks_cluster.endpoint)
pulumi.export("db_endpoint", db_instance.endpoint)
```

---

## 总结

### IaC 成熟度模型

```
┌────────────────────────────────────────────────┐
│          IaC 成熟度五级模型                    │
├────────────────────────────────────────────────┤
│ Level 5: 完全自动化                            │
│  ├─ 自助服务                                   │
│  ├─ 策略即代码                                 │
│  └─ 持续优化                                   │
│                                                │
│ Level 4: 测试与验证                            │
│  ├─ 自动化测试                                 │
│  ├─ 合规检查                                   │
│  └─ 成本优化                                   │
│                                                │
│ Level 3: 模块化与复用                          │
│  ├─ 模块库                                     │
│  ├─ 多环境                                     │
│  └─ 标准化                                     │
│                                                │
│ Level 2: 版本控制                              │
│  ├─ Git 管理                                   │
│  ├─ CI/CD                                      │
│  └─ 状态管理                                   │
│                                                │
│ Level 1: 脚本化                                │
│  ├─ 基础脚本                                   │
│  └─ 手动执行                                   │
└────────────────────────────────────────────────┘
```

### 下一步学习

- [04_deployment_strategies.md](04_deployment_strategies.md) - 部署策略
- [05_release_management.md](05_release_management.md) - 版本管理
