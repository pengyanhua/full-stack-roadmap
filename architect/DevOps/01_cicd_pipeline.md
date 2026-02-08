# CI/CD 流水线设计

## 目录
- [CI/CD 概述](#cicd-概述)
- [Jenkins 流水线](#jenkins-流水线)
- [GitLab CI](#gitlab-ci)
- [GitHub Actions](#github-actions)
- [流水线最佳实践](#流水线最佳实践)
- [多环境部署](#多环境部署)
- [实战案例](#实战案例)

---

## CI/CD 概述

### 什么是 CI/CD

```
传统开发流程                          CI/CD 流程
┌────────────────┐                   ┌────────────────┐
│  开发          │                   │  持续集成      │
│  ├─ 手动构建   │                   │  ├─ 自动构建   │
│  ├─ 人工测试   │                   │  ├─ 自动测试   │
│  ├─ 周期长     │     ────────▶     │  ├─ 快速反馈   │
│  └─ 风险高     │                   │  └─ 持续部署   │
└────────────────┘                   └────────────────┘

发布周期: 数周/数月                   发布周期: 数小时/数天
```

### CI/CD 完整流程

```
┌──────────────────────────────────────────────────────────┐
│                   CI/CD Pipeline 全流程                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1️⃣ Source (代码提交)                                    │
│     │                                                    │
│     ├─ Git Push                                         │
│     ├─ Pull Request                                     │
│     └─ Webhook 触发                                     │
│     │                                                    │
│  2️⃣ Build (构建)                                         │
│     │                                                    │
│     ├─ 代码检出                                          │
│     ├─ 依赖安装                                          │
│     ├─ 编译构建                                          │
│     └─ 打包 Docker 镜像                                  │
│     │                                                    │
│  3️⃣ Test (测试)                                          │
│     │                                                    │
│     ├─ 单元测试                                          │
│     ├─ 集成测试                                          │
│     ├─ 代码质量扫描 (SonarQube)                         │
│     └─ 安全扫描 (Trivy)                                 │
│     │                                                    │
│  4️⃣ Release (发布)                                       │
│     │                                                    │
│     ├─ 推送镜像到 Registry                              │
│     ├─ 生成版本标签                                      │
│     └─ 发布说明                                          │
│     │                                                    │
│  5️⃣ Deploy (部署)                                        │
│     │                                                    │
│     ├─ Dev 环境自动部署                                  │
│     ├─ Staging 手动审批                                 │
│     ├─ Production 蓝绿/金丝雀                           │
│     └─ 健康检查                                          │
│     │                                                    │
│  6️⃣ Monitor (监控)                                       │
│     │                                                    │
│     ├─ 日志聚合                                          │
│     ├─ 指标监控                                          │
│     ├─ 告警通知                                          │
│     └─ 自动回滚                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Jenkins 流水线

### Jenkinsfile 示例

```groovy
// Jenkinsfile - 声明式流水线
pipeline {
    agent {
        kubernetes {
            yaml """
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: docker
    image: docker:24-dind
    command:
    - cat
    tty: true
    volumeMounts:
    - name: docker-sock
      mountPath: /var/run/docker.sock
  - name: kubectl
    image: bitnami/kubectl:latest
    command:
    - cat
    tty: true
  volumes:
  - name: docker-sock
    hostPath:
      path: /var/run/docker.sock
"""
        }
    }

    environment {
        DOCKER_REGISTRY = 'harbor.example.com'
        IMAGE_NAME = 'myapp'
        GIT_COMMIT_SHORT = sh(
            script: "git rev-parse --short HEAD",
            returnStdout: true
        ).trim()
        VERSION = "${env.BUILD_NUMBER}-${GIT_COMMIT_SHORT}"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.GIT_AUTHOR = sh(
                        script: "git log -1 --pretty=format:'%an'",
                        returnStdout: true
                    ).trim()
                }
            }
        }

        stage('Build') {
            steps {
                container('docker') {
                    sh """
                        docker build \
                          --build-arg VERSION=${VERSION} \
                          --tag ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION} \
                          --tag ${DOCKER_REGISTRY}/${IMAGE_NAME}:latest \
                          .
                    """
                }
            }
        }

        stage('Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        container('docker') {
                            sh """
                                docker run --rm \
                                  ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION} \
                                  pytest tests/unit --junitxml=reports/unit.xml
                            """
                        }
                    }
                }

                stage('Integration Tests') {
                    steps {
                        container('docker') {
                            sh """
                                docker-compose -f docker-compose.test.yml up -d
                                docker-compose -f docker-compose.test.yml run test
                                docker-compose -f docker-compose.test.yml down
                            """
                        }
                    }
                }

                stage('Code Quality') {
                    steps {
                        script {
                            def scannerHome = tool 'SonarQube Scanner'
                            withSonarQubeEnv('SonarQube') {
                                sh """
                                    ${scannerHome}/bin/sonar-scanner \
                                      -Dsonar.projectKey=myapp \
                                      -Dsonar.sources=. \
                                      -Dsonar.host.url=${SONAR_HOST_URL} \
                                      -Dsonar.login=${SONAR_AUTH_TOKEN}
                                """
                            }
                        }
                    }
                }

                stage('Security Scan') {
                    steps {
                        container('docker') {
                            sh """
                                docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
                                  aquasec/trivy image \
                                  --severity HIGH,CRITICAL \
                                  --exit-code 1 \
                                  ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}
                            """
                        }
                    }
                }
            }
        }

        stage('Quality Gate') {
            steps {
                timeout(time: 5, unit: 'MINUTES') {
                    waitForQualityGate abortPipeline: true
                }
            }
        }

        stage('Push Image') {
            when {
                branch 'main'
            }
            steps {
                container('docker') {
                    withCredentials([
                        usernamePassword(
                            credentialsId: 'harbor-credentials',
                            usernameVariable: 'REGISTRY_USER',
                            passwordVariable: 'REGISTRY_PASS'
                        )
                    ]) {
                        sh """
                            echo \$REGISTRY_PASS | docker login ${DOCKER_REGISTRY} \
                              -u \$REGISTRY_USER --password-stdin
                            docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}
                            docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:latest
                        """
                    }
                }
            }
        }

        stage('Deploy to Dev') {
            when {
                branch 'main'
            }
            steps {
                container('kubectl') {
                    sh """
                        kubectl set image deployment/myapp \
                          myapp=${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION} \
                          -n dev
                        kubectl rollout status deployment/myapp -n dev
                    """
                }
            }
        }

        stage('Deploy to Staging') {
            when {
                branch 'main'
            }
            steps {
                input message: 'Deploy to Staging?', ok: 'Deploy'
                container('kubectl') {
                    sh """
                        kubectl set image deployment/myapp \
                          myapp=${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION} \
                          -n staging
                        kubectl rollout status deployment/myapp -n staging
                    """
                }
            }
        }

        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                input message: 'Deploy to Production?', ok: 'Deploy', submitter: 'admin'
                container('kubectl') {
                    sh """
                        # 蓝绿部署
                        kubectl apply -f k8s/production/deployment-green.yaml
                        kubectl wait --for=condition=available deployment/myapp-green -n prod --timeout=300s

                        # 切换流量
                        kubectl patch service myapp -n prod -p '{"spec":{"selector":{"version":"green"}}}'

                        # 清理旧版本
                        kubectl delete deployment myapp-blue -n prod || true
                    """
                }
            }
        }
    }

    post {
        success {
            script {
                def message = """
                ✅ 构建成功
                项目: ${env.JOB_NAME}
                版本: ${VERSION}
                提交者: ${env.GIT_AUTHOR}
                构建时间: ${currentBuild.durationString}
                """

                // 发送钉钉通知
                sh """
                    curl -X POST 'https://oapi.dingtalk.com/robot/send?access_token=xxx' \
                      -H 'Content-Type: application/json' \
                      -d '{
                        "msgtype": "text",
                        "text": {"content": "${message}"}
                      }'
                """
            }
        }

        failure {
            script {
                def message = """
                ❌ 构建失败
                项目: ${env.JOB_NAME}
                版本: ${VERSION}
                失败阶段: ${env.STAGE_NAME}
                查看日志: ${env.BUILD_URL}
                """

                sh """
                    curl -X POST 'https://oapi.dingtalk.com/robot/send?access_token=xxx' \
                      -H 'Content-Type: application/json' \
                      -d '{
                        "msgtype": "text",
                        "text": {"content": "${message}"}
                      }'
                """
            }
        }

        always {
            junit 'reports/*.xml'
            archiveArtifacts artifacts: 'reports/**', allowEmptyArchive: true
            cleanWs()
        }
    }
}
```

---

## GitLab CI

### .gitlab-ci.yml 完整示例

```yaml
# .gitlab-ci.yml
variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: ""
  IMAGE_TAG: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA
  LATEST_TAG: $CI_REGISTRY_IMAGE:latest

stages:
  - build
  - test
  - release
  - deploy

# 构建阶段
build:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker build --cache-from $LATEST_TAG -t $IMAGE_TAG -t $LATEST_TAG .
    - docker push $IMAGE_TAG
    - docker push $LATEST_TAG
  only:
    - branches
    - tags

# 测试阶段
unit-test:
  stage: test
  image: python:3.11
  before_script:
    - pip install -r requirements.txt
    - pip install pytest pytest-cov
  script:
    - pytest tests/unit --cov=app --cov-report=xml --cov-report=html
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    paths:
      - htmlcov/
    expire_in: 1 week

integration-test:
  stage: test
  image: docker:24
  services:
    - docker:24-dind
  script:
    - docker-compose -f docker-compose.test.yml up -d
    - docker-compose -f docker-compose.test.yml run --rm test
  after_script:
    - docker-compose -f docker-compose.test.yml down
  only:
    - main
    - merge_requests

code-quality:
  stage: test
  image: sonarsource/sonar-scanner-cli:latest
  variables:
    SONAR_USER_HOME: "${CI_PROJECT_DIR}/.sonar"
    GIT_DEPTH: "0"
  cache:
    key: "${CI_JOB_NAME}"
    paths:
      - .sonar/cache
  script:
    - sonar-scanner
      -Dsonar.qualitygate.wait=true
      -Dsonar.projectKey=$CI_PROJECT_PATH_SLUG
      -Dsonar.sources=.
      -Dsonar.host.url=$SONAR_HOST_URL
      -Dsonar.login=$SONAR_TOKEN
  allow_failure: false
  only:
    - main
    - merge_requests

security-scan:
  stage: test
  image: aquasec/trivy:latest
  script:
    - trivy image --severity HIGH,CRITICAL --exit-code 1 $IMAGE_TAG
  allow_failure: false

# 发布阶段
release:
  stage: release
  image: registry.gitlab.com/gitlab-org/release-cli:latest
  script:
    - echo "Creating release $CI_COMMIT_TAG"
  release:
    tag_name: '$CI_COMMIT_TAG'
    description: 'Release $CI_COMMIT_TAG'
  only:
    - tags

# 部署阶段
deploy-dev:
  stage: deploy
  image: bitnami/kubectl:latest
  environment:
    name: development
    url: https://dev.example.com
  script:
    - kubectl config use-context $KUBE_CONTEXT
    - kubectl set image deployment/myapp myapp=$IMAGE_TAG -n dev
    - kubectl rollout status deployment/myapp -n dev
  only:
    - main

deploy-staging:
  stage: deploy
  image: bitnami/kubectl:latest
  environment:
    name: staging
    url: https://staging.example.com
  script:
    - kubectl config use-context $KUBE_CONTEXT
    - kubectl set image deployment/myapp myapp=$IMAGE_TAG -n staging
    - kubectl rollout status deployment/myapp -n staging
  when: manual
  only:
    - main

deploy-production:
  stage: deploy
  image: bitnami/kubectl:latest
  environment:
    name: production
    url: https://example.com
  script:
    - kubectl config use-context $KUBE_CONTEXT
    - |
      # 金丝雀部署
      kubectl apply -f k8s/canary-deployment.yaml
      kubectl set image deployment/myapp-canary myapp=$IMAGE_TAG -n prod

      # 等待金丝雀健康
      kubectl rollout status deployment/myapp-canary -n prod

      # 监控 5 分钟
      sleep 300

      # 全量发布
      kubectl set image deployment/myapp myapp=$IMAGE_TAG -n prod
      kubectl rollout status deployment/myapp -n prod

      # 清理金丝雀
      kubectl delete deployment myapp-canary -n prod
  when: manual
  only:
    - tags
  allow_failure: false
```

---

## GitHub Actions

### 完整 Workflow 示例

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
    tags:
      - 'v*'
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # 构建和测试
  build-and-test:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov flake8

      - name: Lint with flake8
        run: |
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
          flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

      - name: Run unit tests
        run: |
          pytest tests/unit -v --cov=app --cov-report=xml --cov-report=html

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          flags: unittests
          name: codecov-umbrella

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

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

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # 安全扫描
  security-scan:
    runs-on: ubuntu-latest
    needs: build-and-test
    if: github.event_name != 'pull_request'

    steps:
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'

  # 部署到开发环境
  deploy-dev:
    runs-on: ubuntu-latest
    needs: [build-and-test, security-scan]
    if: github.ref == 'refs/heads/main'
    environment:
      name: development
      url: https://dev.example.com

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Kubectl
        uses: azure/setup-kubectl@v3

      - name: Configure Kubernetes
        run: |
          echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig

      - name: Deploy to Dev
        run: |
          export KUBECONFIG=kubeconfig
          kubectl set image deployment/myapp \
            myapp=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            -n dev
          kubectl rollout status deployment/myapp -n dev --timeout=300s

  # 部署到生产环境
  deploy-prod:
    runs-on: ubuntu-latest
    needs: build-and-test
    if: startsWith(github.ref, 'refs/tags/v')
    environment:
      name: production
      url: https://example.com

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Kubectl
        uses: azure/setup-kubectl@v3

      - name: Configure Kubernetes
        run: |
          echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig

      - name: Deploy to Production
        run: |
          export KUBECONFIG=kubeconfig

          # 蓝绿部署
          kubectl apply -f k8s/production/deployment-green.yaml
          kubectl set image deployment/myapp-green \
            myapp=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.ref_name }} \
            -n prod

          kubectl rollout status deployment/myapp-green -n prod --timeout=600s

          # 切换流量
          kubectl patch service myapp -n prod \
            -p '{"spec":{"selector":{"version":"green"}}}'

          # 清理旧版本
          kubectl delete deployment myapp-blue -n prod --ignore-not-found=true

      - name: Create GitHub Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref_name }}
          release_name: Release ${{ github.ref_name }}
          draft: false
          prerelease: false
```

---

## 流水线最佳实践

### 流水线优化策略

```
┌────────────────────────────────────────────────────────┐
│              CI/CD 流水线优化八大原则                  │
├────────────────────────────────────────────────────────┤
│                                                        │
│ 1. 快速反馈 (Fast Feedback)                          │
│    ├─ 并行执行测试                                     │
│    ├─ 缓存依赖                                         │
│    └─ 增量构建                                         │
│                                                        │
│ 2. 失败快速 (Fail Fast)                              │
│    ├─ 代码检查最先执行                                 │
│    ├─ 快速测试优先                                     │
│    └─ 及时中止失败流水线                               │
│                                                        │
│ 3. 可复现性 (Reproducibility)                        │
│    ├─ 固定依赖版本                                     │
│    ├─ 容器化构建环境                                   │
│    └─ 幂等性部署                                       │
│                                                        │
│ 4. 安全第一 (Security First)                         │
│    ├─ 密钥管理                                         │
│    ├─ 镜像扫描                                         │
│    └─ 依赖审计                                         │
│                                                        │
│ 5. 可观测性 (Observability)                          │
│    ├─ 详细日志                                         │
│    ├─ 指标监控                                         │
│    └─ 告警通知                                         │
│                                                        │
│ 6. 自动化一切 (Automate Everything)                  │
│    ├─ 代码即配置                                       │
│    ├─ 自动化测试                                       │
│    └─ 自动化部署                                       │
│                                                        │
│ 7. 版本控制 (Version Control)                        │
│    ├─ 流水线配置版本化                                 │
│    ├─ 配置文件版本化                                   │
│    └─ 基础设施版本化                                   │
│                                                        │
│ 8. 持续改进 (Continuous Improvement)                 │
│    ├─ 定期回顾                                         │
│    ├─ 指标分析                                         │
│    └─ 流程优化                                         │
└────────────────────────────────────────────────────────┘
```

### 缓存策略优化

```yaml
# GitHub Actions 缓存示例
- name: Cache dependencies
  uses: actions/cache@v3
  with:
    path: |
      ~/.cache/pip
      ~/.npm
      ~/.m2/repository
      **/node_modules
    key: ${{ runner.os }}-deps-${{ hashFiles('**/requirements.txt', '**/package-lock.json', '**/pom.xml') }}
    restore-keys: |
      ${{ runner.os }}-deps-

# Docker Layer 缓存
- name: Build with cache
  uses: docker/build-push-action@v5
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

---

## 多环境部署

### 环境配置管理

```
┌────────────────────────────────────────────────┐
│           多环境部署策略                       │
├────────┬────────┬────────┬──────────┬─────────┤
│ 环境   │  Dev   │ Staging│   Prod   │ 说明    │
├────────┼────────┼────────┼──────────┼─────────┤
│ 触发   │ 自动   │  手动  │   手动   │         │
│ 审批   │ 无     │  可选  │   必须   │         │
│ 实例数 │ 1      │  2     │   5+     │         │
│ 数据库 │ 共享   │  独立  │   独立   │         │
│ 监控   │ 基础   │  完整  │   完整   │         │
│ 日志   │ 7天    │  30天  │   90天   │         │
└────────┴────────┴────────┴──────────┴─────────┘
```

### Kustomize 多环境配置

```yaml
# base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - deployment.yaml
  - service.yaml
  - configmap.yaml

commonLabels:
  app: myapp

# overlays/dev/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

bases:
  - ../../base

namespace: dev

replicas:
  - name: myapp
    count: 1

images:
  - name: myapp
    newTag: dev-latest

configMapGenerator:
  - name: app-config
    behavior: merge
    literals:
      - ENV=development
      - DEBUG=true
      - LOG_LEVEL=debug

# overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

bases:
  - ../../base

namespace: prod

replicas:
  - name: myapp
    count: 5

images:
  - name: myapp
    newTag: v1.2.3

configMapGenerator:
  - name: app-config
    behavior: merge
    literals:
      - ENV=production
      - DEBUG=false
      - LOG_LEVEL=info

resources:
  - hpa.yaml
  - pdb.yaml
```

---

## 实战案例

### 案例 1: 微服务 Monorepo CI/CD

```yaml
# .github/workflows/monorepo-ci.yml
name: Monorepo CI/CD

on:
  push:
    branches: [main]

jobs:
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      user-service: ${{ steps.filter.outputs.user-service }}
      order-service: ${{ steps.filter.outputs.order-service }}
      payment-service: ${{ steps.filter.outputs.payment-service }}
    steps:
      - uses: actions/checkout@v4
      - uses: dorny/paths-filter@v2
        id: filter
        with:
          filters: |
            user-service:
              - 'services/user/**'
            order-service:
              - 'services/order/**'
            payment-service:
              - 'services/payment/**'

  build-user-service:
    needs: detect-changes
    if: needs.detect-changes.outputs.user-service == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build User Service
        run: |
          cd services/user
          docker build -t user-service:${{ github.sha }} .
          docker push ghcr.io/${{ github.repository }}/user-service:${{ github.sha }}

  build-order-service:
    needs: detect-changes
    if: needs.detect-changes.outputs.order-service == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Order Service
        run: |
          cd services/order
          docker build -t order-service:${{ github.sha }} .
          docker push ghcr.io/${{ github.repository }}/order-service:${{ github.sha }}

  build-payment-service:
    needs: detect-changes
    if: needs.detect-changes.outputs.payment-service == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Payment Service
        run: |
          cd services/payment
          docker build -t payment-service:${{ github.sha }} .
          docker push ghcr.io/${{ github.repository }}/payment-service:${{ github.sha }}
```

### 案例 2: 自动回滚流水线

```groovy
// Jenkinsfile - 带自动回滚
pipeline {
    agent any

    environment {
        ROLLBACK_ENABLED = 'true'
        HEALTH_CHECK_RETRIES = '5'
    }

    stages {
        stage('Deploy') {
            steps {
                script {
                    // 保存当前版本
                    env.PREVIOUS_VERSION = sh(
                        script: "kubectl get deployment myapp -n prod -o jsonpath='{.spec.template.spec.containers[0].image}'",
                        returnStdout: true
                    ).trim()

                    echo "Previous version: ${env.PREVIOUS_VERSION}"

                    // 部署新版本
                    sh """
                        kubectl set image deployment/myapp \
                          myapp=${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION} \
                          -n prod
                        kubectl rollout status deployment/myapp -n prod --timeout=300s
                    """
                }
            }
        }

        stage('Health Check') {
            steps {
                script {
                    def healthy = false

                    for (int i = 0; i < HEALTH_CHECK_RETRIES.toInteger(); i++) {
                        sleep(10)

                        def response = sh(
                            script: "curl -s -o /dev/null -w '%{http_code}' https://example.com/health",
                            returnStdout: true
                        ).trim()

                        if (response == '200') {
                            healthy = true
                            break
                        }

                        echo "Health check failed (${i+1}/${HEALTH_CHECK_RETRIES}): HTTP ${response}"
                    }

                    if (!healthy) {
                        error("Health check failed after ${HEALTH_CHECK_RETRIES} retries")
                    }
                }
            }
        }

        stage('Smoke Tests') {
            steps {
                sh """
                    docker run --rm \
                      -e API_URL=https://example.com \
                      smoke-tests:latest
                """
            }
        }
    }

    post {
        failure {
            script {
                if (env.ROLLBACK_ENABLED == 'true' && env.PREVIOUS_VERSION) {
                    echo "🔄 Rolling back to ${env.PREVIOUS_VERSION}"

                    sh """
                        kubectl set image deployment/myapp \
                          myapp=${env.PREVIOUS_VERSION} \
                          -n prod
                        kubectl rollout status deployment/myapp -n prod --timeout=300s
                    """

                    // 通知团队
                    sh """
                        curl -X POST 'https://hooks.slack.com/services/xxx' \
                          -H 'Content-Type: application/json' \
                          -d '{
                            "text": "⚠️ Deployment failed and rolled back to ${env.PREVIOUS_VERSION}"
                          }'
                    """
                }
            }
        }
    }
}
```

---

## 总结

### CI/CD 成熟度模型

```
┌────────────────────────────────────────────────┐
│          CI/CD 成熟度五级模型                  │
├────────────────────────────────────────────────┤
│                                                │
│ Level 5: 优化 (Optimizing)                    │
│  ├─ 全自动部署                                 │
│  ├─ A/B 测试                                   │
│  ├─ 特性开关                                   │
│  └─ 持续优化                                   │
│                                                │
│ Level 4: 度量 (Measured)                      │
│  ├─ 完整监控                                   │
│  ├─ 自动回滚                                   │
│  └─ 部署指标分析                               │
│                                                │
│ Level 3: 自动化 (Automated)                   │
│  ├─ 自动化测试                                 │
│  ├─ 自动化部署                                 │
│  └─ 多环境管理                                 │
│                                                │
│ Level 2: 可重复 (Repeatable)                  │
│  ├─ 版本控制                                   │
│  ├─ 构建自动化                                 │
│  └─ 基础测试                                   │
│                                                │
│ Level 1: 初始 (Initial)                       │
│  ├─ 手动构建                                   │
│  ├─ 手动测试                                   │
│  └─ 手动部署                                   │
└────────────────────────────────────────────────┘
```

### 关键指标

- **部署频率**: 每天多次 vs 每月一次
- **变更前置时间**: < 1小时 vs > 1周
- **平均恢复时间**: < 1小时 vs > 1天
- **变更失败率**: < 15% vs > 50%

### 下一步学习

- [02_gitops.md](02_gitops.md) - GitOps 实践
- [03_infrastructure_as_code.md](03_infrastructure_as_code.md) - 基础设施即代码
- [04_deployment_strategies.md](04_deployment_strategies.md) - 部署策略
