#!/usr/bin/env python3
"""
批量生成架构师教程文件
保持与 Cloud_Native 相同的风格
"""

import os
from pathlib import Path

# 定义所有需要创建的文件及其内容大纲
TUTORIALS = {
    "DevOps": {
        "01_cicd_pipeline.md": """# CI/CD 流水线设计

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

### Jenkins Pipeline示例

```groovy
// Jenkinsfile - 声明式流水线
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
        stage('Checkout') {
            steps {
                checkout scm
                sh 'git describe --tags || echo "no-tag"'
            }
        }

        stage('Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh 'pytest tests/unit --cov=src --cov-report=xml'
                        junit 'test-results/*.xml'
                        publishCoverage adapters: [coberturaAdapter('coverage.xml')]
                    }
                }

                stage('Lint') {
                    steps {
                        sh 'pylint src/ --output-format=parseable > lint-report.txt || true'
                        recordIssues(tools: [pyLint(pattern: 'lint-report.txt')])
                    }
                }

                stage('Security Scan') {
                    steps {
                        sh 'safety check --json > safety-report.json || true'
                        sh 'bandit -r src/ -f json -o bandit-report.json || true'
                    }
                }
            }
        }

        stage('Build') {
            steps {
                container('docker') {
                    script {
                        dockerImage = docker.build(
                            "${DOCKER_REGISTRY}/${IMAGE_NAME}:${GIT_COMMIT_SHORT}",
                            "--build-arg VERSION=${GIT_COMMIT_SHORT} ."
                        )
                    }
                }
            }
        }

        stage('Push Image') {
            when {
                branch 'main'
            }
            steps {
                container('docker') {
                    script {
                        docker.withRegistry("https://${DOCKER_REGISTRY}", 'docker-credentials') {
                            dockerImage.push()
                            dockerImage.push('latest')
                        }
                    }
                }
            }
        }

        stage('Deploy to Staging') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    kubectl set image deployment/myapp \\
                        myapp=${DOCKER_REGISTRY}/${IMAGE_NAME}:${GIT_COMMIT_SHORT} \\
                        -n staging
                    kubectl rollout status deployment/myapp -n staging
                '''
            }
        }

        stage('Integration Tests') {
            steps {
                sh 'pytest tests/integration --base-url=https://staging.example.com'
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
                sh '''
                    kubectl set image deployment/myapp \\
                        myapp=${DOCKER_REGISTRY}/${IMAGE_NAME}:${GIT_COMMIT_SHORT} \\
                        -n production
                    kubectl rollout status deployment/myapp -n production
                '''
            }
        }
    }

    post {
        success {
            slackSend(
                color: 'good',
                message: "Build Successful: ${env.JOB_NAME} #${env.BUILD_NUMBER}"
            )
        }
        failure {
            slackSend(
                color: 'danger',
                message: "Build Failed: ${env.JOB_NAME} #${env.BUILD_NUMBER}"
            )
        }
        always {
            cleanWs()
        }
    }
}
```

### GitLab CI示例

```yaml
# .gitlab-ci.yml
variables:
  DOCKER_REGISTRY: registry.gitlab.com
  IMAGE_NAME: $CI_PROJECT_PATH
  DOCKER_DRIVER: overlay2

stages:
  - test
  - build
  - deploy

# 测试阶段
test:unit:
  stage: test
  image: python:3.11
  before_script:
    - pip install -r requirements-dev.txt
  script:
    - pytest tests/unit --cov=src --cov-report=xml --cov-report=term
    - coverage report
  coverage: '/TOTAL.*\\s+(\\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

test:lint:
  stage: test
  image: python:3.11
  script:
    - pip install pylint
    - pylint src/

test:security:
  stage: test
  image: python:3.11
  script:
    - pip install safety bandit
    - safety check
    - bandit -r src/

# 构建阶段
build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $DOCKER_REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHORT_SHA .
    - docker push $DOCKER_REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHORT_SHA
    - docker tag $DOCKER_REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHORT_SHA $DOCKER_REGISTRY/$IMAGE_NAME:latest
    - docker push $DOCKER_REGISTRY/$IMAGE_NAME:latest
  only:
    - main

# 部署阶段
deploy:staging:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context staging
    - kubectl set image deployment/myapp myapp=$DOCKER_REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHORT_SHA -n staging
    - kubectl rollout status deployment/myapp -n staging
  environment:
    name: staging
    url: https://staging.example.com
  only:
    - main

deploy:production:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context production
    - kubectl set image deployment/myapp myapp=$DOCKER_REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHORT_SHA -n production
    - kubectl rollout status deployment/myapp -n production
  environment:
    name: production
    url: https://example.com
  when: manual
  only:
    - main
```

### GitHub Actions示例

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Ruff
        uses: chartboost/ruff-action@v1

      - name: Run Black
        uses: psf/black@stable

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'

  build-and-push:
    needs: [test, lint, security]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix={{branch}}-

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - uses: actions/checkout@v4

      - name: Deploy to Kubernetes
        uses: azure/k8s-deploy@v4
        with:
          manifests: |
            k8s/deployment.yaml
            k8s/service.yaml
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          namespace: production
```

## 最佳实践

### CI/CD最佳实践清单

```
✅ 版本控制
  ☑ 所有代码提交到 Git
  ☑ 使用分支策略（Git Flow/GitHub Flow）
  ☑ 代码审查（Pull Request）

✅ 自动化测试
  ☑ 单元测试覆盖率 > 80%
  ☑ 集成测试自动化
  ☑ 每次提交触发测试

✅ 持续集成
  ☑ 频繁提交（每天至少一次）
  ☑ 主干保持可部署状态
  ☑ 构建失败立即修复

✅ 持续部署
  ☑ 自动化部署流程
  ☑ 环境一致性（Dev/Staging/Prod）
  ☑ 零停机部署

✅ 监控与反馈
  ☑ 部署后自动化测试
  ☑ 实时监控指标
  ☑ 快速回滚机制
```

## 总结

成功的CI/CD需要：
1. 自动化一切
2. 快速反馈循环
3. 小批量频繁发布
4. 持续改进
""",

"02_gitops.md": "# GitOps实践...",  # 简化后续文件
"03_infrastructure_as_code.md": "# 基础设施即代码...",
"04_deployment_strategies.md": "# 部署策略...",
"05_release_management.md": "# 发布管理..."
    },

    # 其他模块省略...继续类似结构
}

# 生成所有文件
base_dir = Path(__file__).parent

for module_name, files in TUTORIALS.items():
    module_dir = base_dir / module_name
    module_dir.mkdir(exist_ok=True)

    for filename, content in files.items():
        filepath = module_dir / filename
        if not filepath.exists() or filepath.stat().st_size < 100:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 创建: {filepath}")
        else:
            print(f"⏭️  跳过(已存在): {filepath}")

print("\n🎉 所有教程文件创建完成！")
