# CI/CD 流水线设计

## 目录
- [CI/CD概述](#cicd概述)
- [Jenkins流水线](#jenkins流水线)
- [GitLab CI](#gitlab-ci)
- [GitHub Actions](#github-actions)
- [最佳实践](#最佳实践)

---

## CI/CD概述

### CI/CD流程图

```
┌────────────────────────────────────────────────────┐
│              CI/CD 完整流程                        │
├────────────────────────────────────────────────────┤
│                                                    │
│  开发 ─▶ 提交代码 ─▶ 自动触发                     │
│              │                                     │
│         ┌────▼────┐                               │
│         │  CI     │                               │
│         ├─────────┤                               │
│         │ ✓ 代码检出│                              │
│         │ ✓ 依赖安装│                              │
│         │ ✓ 单元测试│                              │
│         │ ✓ 代码扫描│                              │
│         │ ✓ 构建镜像│                              │
│         │ ✓ 推送仓库│                              │
│         └────┬────┘                               │
│              │                                     │
│         ┌────▼────┐                               │
│         │  CD     │                               │
│         ├─────────┤                               │
│         │ ✓ 部署测试│                              │
│         │ ✓ 集成测试│                              │
│         │ ✓ 部署预发│                              │
│         │ ✓ 冒烟测试│                              │
│         │ ✓ 部署生产│                              │
│         │ ✓ 健康检查│                              │
│         └─────────┘                               │
└────────────────────────────────────────────────────┘
```

## Jenkins Pipeline完整示例

```groovy
// Jenkinsfile
pipeline {
    agent {
        kubernetes {
            yaml '''
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: docker
    image: docker:latest
    command: ['cat']
    tty: true
    volumeMounts:
    - name: docker-sock
      mountPath: /var/run/docker.sock
  volumes:
  - name: docker-sock
    hostPath:
      path: /var/run/docker.sock
'''
        }
    }

    environment {
        DOCKER_REGISTRY = 'registry.example.com'
        IMAGE_NAME = 'myapp'
        GIT_COMMIT_SHORT = sh(
            script: "git rev-parse --short HEAD",
            returnStdout: true
        ).trim()
    }

    stages {
        stage('Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh 'pytest tests/unit --cov=src'
                    }
                }
                stage('Lint') {
                    steps {
                        sh 'pylint src/'
                    }
                }
            }
        }

        stage('Build') {
            steps {
                container('docker') {
                    sh 'docker build -t ${DOCKER_REGISTRY}/${IMAGE_NAME}:${GIT_COMMIT_SHORT} .'
                }
            }
        }

        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            input {
                message "Deploy to production?"
                ok "Deploy"
            }
            steps {
                sh 'kubectl set image deployment/myapp myapp=${DOCKER_REGISTRY}/${IMAGE_NAME}:${GIT_COMMIT_SHORT} -n production'
            }
        }
    }
}
```

## GitHub Actions示例

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: pytest tests/

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t myapp:${{ github.sha }} .
      - name: Push to registry
        run: docker push myapp:${{ github.sha }}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to K8s
        run: kubectl set image deployment/myapp myapp=myapp:${{ github.sha }}
```

## 总结

成功的CI/CD需要：自动化、快速反馈、频繁发布。
