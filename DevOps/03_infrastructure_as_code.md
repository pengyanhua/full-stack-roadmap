# 基础设施即代码 (IaC)

## 目录
- [IaC概述](#iac概述)
- [Terraform](#terraform)
- [Ansible](#ansible)
- [最佳实践](#最佳实践)

---

## IaC概述

```
┌────────────────────────────────────────────────────┐
│          IaC vs 手动配置                           │
├────────────────────────────────────────────────────┤
│                                                    │
│  手动配置                 IaC                       │
│  ❌ 不可重复             ✅ 完全可重复              │
│  ❌ 难以追踪变更         ✅ 版本控制                │
│  ❌ 人为错误             ✅ 自动化测试              │
│  ❌ 难以扩展             ✅ 轻松扩展                │
│  ❌ 文档过时             ✅ 代码即文档              │
└────────────────────────────────────────────────────┘
```

## Terraform

### 完整示例

```hcl
# main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    bucket = "terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.region
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true

  tags = {
    Name = "${var.project}-vpc"
  }
}

# 子网
resource "aws_subnet" "public" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "${var.project}-public-${count.index}"
  }
}

# EC2实例
resource "aws_instance" "app" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type
  subnet_id     = aws_subnet.public[0].id

  user_data = <<-EOF
              #!/bin/bash
              apt-get update
              apt-get install -y docker.io
              docker run -d -p 80:80 nginx
              EOF

  tags = {
    Name = "${var.project}-app"
  }
}

# 输出
output "instance_ip" {
  value = aws_instance.app.public_ip
}
```

### Modules

```hcl
# modules/vpc/main.tf
resource "aws_vpc" "this" {
  cidr_block = var.cidr_block

  tags = merge(
    var.tags,
    {
      Name = var.name
    }
  )
}

# 使用Module
module "vpc" {
  source = "./modules/vpc"

  name       = "production-vpc"
  cidr_block = "10.0.0.0/16"
  tags = {
    Environment = "production"
  }
}
```

## Ansible

### Playbook示例

```yaml
# deploy.yml
---
- name: Deploy Application
  hosts: webservers
  become: yes

  vars:
    app_version: "1.2.3"
    app_port: 8080

  tasks:
    - name: Install Docker
      apt:
        name: docker.io
        state: present
        update_cache: yes

    - name: Start Docker service
      service:
        name: docker
        state: started
        enabled: yes

    - name: Pull application image
      docker_image:
        name: "myapp:{{ app_version }}"
        source: pull

    - name: Run application container
      docker_container:
        name: myapp
        image: "myapp:{{ app_version }}"
        state: started
        restart_policy: always
        ports:
          - "{{ app_port }}:8080"
        env:
          APP_ENV: production
          DB_HOST: "{{ db_host }}"

    - name: Wait for application to start
      wait_for:
        port: "{{ app_port }}"
        delay: 10
        timeout: 60

    - name: Health check
      uri:
        url: "http://localhost:{{ app_port }}/health"
        status_code: 200
      register: result
      until: result.status == 200
      retries: 5
      delay: 10
```

### Ansible Roles

```
roles/
└── webserver/
    ├── tasks/
    │   └── main.yml
    ├── handlers/
    │   └── main.yml
    ├── templates/
    │   └── nginx.conf.j2
    ├── files/
    └── defaults/
        └── main.yml
```

```yaml
# roles/webserver/tasks/main.yml
---
- name: Install Nginx
  apt:
    name: nginx
    state: present

- name: Configure Nginx
  template:
    src: nginx.conf.j2
    dest: /etc/nginx/nginx.conf
  notify: restart nginx

- name: Start Nginx
  service:
    name: nginx
    state: started
```

## 总结

IaC关键原则：
1. 版本控制所有配置
2. 模块化和复用
3. 不可变基础设施
4. 自动化测试
