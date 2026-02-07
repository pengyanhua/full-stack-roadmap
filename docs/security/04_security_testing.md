# 安全测试与漏洞扫描

## 1. 安全测试概述

### 1.1 安全测试分类

```
安全测试体系
│
├── 静态应用安全测试 (SAST)
│   ├── 源代码分析
│   ├── 编译后代码分析
│   └── 配置文件审计
│
├── 动态应用安全测试 (DAST)
│   ├── 黑盒渗透测试
│   ├── 模糊测试
│   └── 运行时监控
│
├── 交互式应用安全测试 (IAST)
│   ├── 灰盒测试
│   └── 插桩技术
│
├── 依赖安全扫描 (SCA)
│   ├── 已知漏洞检测
│   ├── 许可证合规
│   └── 依赖关系分析
│
└── 基础设施安全测试
    ├── 容器镜像扫描
    ├── 云配置审计
    └── 网络安全测试
```

### 1.2 安全测试生命周期

```
开发阶段                     测试内容
┌────────────────────┐
│  需求分析           │ → 威胁建模
│                    │   安全需求定义
├────────────────────┤
│  设计阶段           │ → 架构安全审查
│                    │   数据流分析
├────────────────────┤
│  编码阶段           │ → IDE安全插件
│                    │   代码审查
├────────────────────┤
│  构建阶段           │ → SAST扫描
│                    │   SCA依赖检查
├────────────────────┤
│  测试阶段           │ → DAST渗透测试
│                    │   API安全测试
├────────────────────┤
│  部署阶段           │ → 容器扫描
│                    │   配置审计
└────────────────────┘
│  运行阶段           │ → RASP防护
                     │   WAF监控
                     │   日志分析
```

## 2. SonarQube代码质量与安全分析

### 2.1 SonarQube架构

```
┌─────────────────────────────────────────┐
│           SonarQube Web UI              │
│  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │ 仪表板   │  │ 问题管理 │  │ 规则库 ││
│  └──────────┘  └──────────┘  └────────┘│
├─────────────────────────────────────────┤
│         SonarQube Server                │
│  ┌──────────────────┐  ┌───────────────┐│
│  │ 计算引擎         │  │ Web服务       ││
│  │ (Compute Engine) │  │ (REST API)    ││
│  └──────────────────┘  └───────────────┘│
│  ┌──────────────────────────────────────┤
│  │ Elasticsearch (搜索引擎)            ││
│  └──────────────────────────────────────┤
│  │ PostgreSQL/Oracle (数据库)          ││
│  └──────────────────────────────────────┘│
├─────────────────────────────────────────┤
│        SonarScanner (客户端)            │
│  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │ Maven插件│  │ Gradle插件│  │ CLI   ││
│  └──────────┘  └──────────┘  └────────┘│
└─────────────────────────────────────────┘
```

### 2.2 Docker部署SonarQube

```yaml
# docker-compose.yml
version: "3.8"

services:
  sonarqube:
    image: sonarqube:10.3-community
    container_name: sonarqube
    depends_on:
      - db
    environment:
      SONAR_JDBC_URL: jdbc:postgresql://db:5432/sonar
      SONAR_JDBC_USERNAME: sonar
      SONAR_JDBC_PASSWORD: sonar_password
      SONAR_ES_BOOTSTRAP_CHECKS_DISABLE: "true"
    volumes:
      - sonarqube_data:/opt/sonarqube/data
      - sonarqube_extensions:/opt/sonarqube/extensions
      - sonarqube_logs:/opt/sonarqube/logs
    ports:
      - "9000:9000"
    networks:
      - sonarnet
    ulimits:
      nofile:
        soft: 65536
        hard: 65536
    mem_limit: 4g

  db:
    image: postgres:15-alpine
    container_name: sonarqube-db
    environment:
      POSTGRES_USER: sonar
      POSTGRES_PASSWORD: sonar_password
      POSTGRES_DB: sonar
    volumes:
      - postgresql_data:/var/lib/postgresql/data
    networks:
      - sonarnet

volumes:
  sonarqube_data:
  sonarqube_extensions:
  sonarqube_logs:
  postgresql_data:

networks:
  sonarnet:
    driver: bridge
```

### 2.3 Quality Gates配置

```javascript
// sonar-project.js - 项目配置
module.exports = {
  sonar: {
    projectKey: 'my-project',
    projectName: 'My Project',
    projectVersion: '1.0.0',
    sources: 'src',
    tests: 'tests',
    exclusions: '**/node_modules/**,**/*.test.js',
    javascript: {
      lcov: {
        reportPaths: 'coverage/lcov.info'
      }
    }
  }
};
```

**Quality Gate规则定制**：

```java
// 通过API创建自定义Quality Gate
import org.sonarqube.ws.client.HttpConnector;
import org.sonarqube.ws.client.WsClient;
import org.sonarqube.ws.client.WsClientFactories;
import org.sonarqube.ws.client.qualitygates.*;

public class CustomQualityGate {

    public static void createQualityGate() {
        WsClient wsClient = WsClientFactories.getDefault().newClient(
            HttpConnector.newBuilder()
                .url("http://localhost:9000")
                .token("squ_your_token_here")
                .build()
        );

        QualitygatesService qgService = wsClient.qualitygates();

        // 创建Quality Gate
        CreateResponse createResponse = qgService.create(
            new CreateRequest().setName("Enterprise Security Gate")
        );
        long gateId = createResponse.getId();

        // 添加条件：代码覆盖率 >= 80%
        qgService.createCondition(new CreateConditionRequest()
            .setGateId(gateId)
            .setMetric("coverage")
            .setOp("LT")
            .setError("80")
        );

        // 添加条件：安全热点审查率 = 100%
        qgService.createCondition(new CreateConditionRequest()
            .setGateId(gateId)
            .setMetric("security_hotspots_reviewed")
            .setOp("LT")
            .setError("100")
        );

        // 添加条件：阻断性问题 = 0
        qgService.createCondition(new CreateConditionRequest()
            .setGateId(gateId)
            .setMetric("blocker_violations")
            .setOp("GT")
            .setError("0")
        );

        // 添加条件：严重问题 = 0
        qgService.createCondition(new CreateConditionRequest()
            .setGateId(gateId)
            .setMetric("critical_violations")
            .setOp("GT")
            .setError("0")
        );

        // 添加条件：技术债务比率 <= 5%
        qgService.createCondition(new CreateConditionRequest()
            .setGateId(gateId)
            .setMetric("sqale_debt_ratio")
            .setOp("GT")
            .setError("5")
        );

        System.out.println("Quality Gate created: " + gateId);
    }
}
```

### 2.4 CI/CD集成

**Maven项目**：

```xml
<!-- pom.xml -->
<properties>
    <sonar.host.url>http://localhost:9000</sonar.host.url>
    <sonar.login>squ_your_token</sonar.login>
    <sonar.coverage.jacoco.xmlReportPaths>
        target/site/jacoco/jacoco.xml
    </sonar.coverage.jacoco.xmlReportPaths>
</properties>

<build>
    <plugins>
        <plugin>
            <groupId>org.jacoco</groupId>
            <artifactId>jacoco-maven-plugin</artifactId>
            <version>0.8.10</version>
            <executions>
                <execution>
                    <goals>
                        <goal>prepare-agent</goal>
                    </goals>
                </execution>
                <execution>
                    <id>report</id>
                    <phase>test</phase>
                    <goals>
                        <goal>report</goal>
                    </goals>
                </execution>
            </executions>
        </plugin>
    </plugins>
</build>
```

**Jenkins Pipeline**：

```groovy
// Jenkinsfile
pipeline {
    agent any

    environment {
        SONAR_TOKEN = credentials('sonarqube-token')
        SONAR_HOST = 'http://sonarqube:9000'
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/example/project.git'
            }
        }

        stage('Build & Test') {
            steps {
                sh 'mvn clean test'
            }
        }

        stage('SonarQube Analysis') {
            steps {
                script {
                    def scannerHome = tool 'SonarScanner'
                    withSonarQubeEnv('SonarQube') {
                        sh """
                            ${scannerHome}/bin/sonar-scanner \
                            -Dsonar.projectKey=my-project \
                            -Dsonar.sources=src \
                            -Dsonar.java.binaries=target/classes \
                            -Dsonar.coverage.jacoco.xmlReportPaths=target/site/jacoco/jacoco.xml
                        """
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
    }

    post {
        failure {
            emailext(
                subject: "SonarQube Quality Gate Failed: ${env.JOB_NAME}",
                body: "Check console output at ${env.BUILD_URL}",
                to: "dev-team@example.com"
            )
        }
    }
}
```

## 3. Snyk依赖安全扫描

### 3.1 Snyk CLI使用

```bash
# 安装Snyk CLI
npm install -g snyk

# 认证
snyk auth

# 扫描依赖漏洞
snyk test

# 监控项目（持续监控）
snyk monitor

# 扫描Docker镜像
snyk container test nginx:latest

# 扫描Kubernetes配置
snyk iac test deployment.yaml
```

### 3.2 Snyk配置文件

```yaml
# .snyk
version: v1.22.0

# 忽略特定漏洞
ignore:
  SNYK-JS-LODASH-590103:
    - '*':
        reason: 'Fix requires major version upgrade'
        expires: '2026-06-01T00:00:00.000Z'

# 补丁配置
patch:
  'npm:moment:20170905':
    - moment:
        patched: '2026-01-15T10:00:00.000Z'

# 排除路径
exclude:
  global:
    - test/**
    - docs/**
```

### 3.3 GitHub Actions集成

```yaml
# .github/workflows/security.yml
name: Security Scan

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # 每天2AM执行

jobs:
  security:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Run Snyk to check for vulnerabilities
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high --fail-on=all

      - name: Upload Snyk report
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: snyk.sarif

      - name: Run Snyk on Docker image
        uses: snyk/actions/docker@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          image: myapp:latest
          args: --file=Dockerfile
```

### 3.4 GitLab CI/CD集成

```yaml
# .gitlab-ci.yml
stages:
  - test
  - security

snyk_test:
  stage: security
  image: snyk/snyk:node
  script:
    - snyk auth $SNYK_TOKEN
    - snyk test --json > snyk-report.json
    - snyk monitor --project-name=${CI_PROJECT_NAME}
  artifacts:
    reports:
      dependency_scanning: snyk-report.json
    expire_in: 1 week
  only:
    - branches
  allow_failure: false

snyk_container:
  stage: security
  image: snyk/snyk:docker
  services:
    - docker:dind
  script:
    - snyk auth $SNYK_TOKEN
    - snyk container test $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main
```

## 4. OWASP ZAP自动化测试

### 4.1 ZAP架构

```
┌────────────────────────────────────────┐
│         OWASP ZAP Architecture         │
├────────────────────────────────────────┤
│  ┌──────────────────────────────────┐  │
│  │       ZAP Desktop GUI/CLI        │  │
│  └──────────────────────────────────┘  │
│  ┌──────────────────────────────────┐  │
│  │         API Server               │  │
│  │  (REST/JSON/XML endpoints)       │  │
│  └──────────────────────────────────┘  │
│  ┌──────────────────────────────────┐  │
│  │         Core Engine              │  │
│  │  ┌────────┐  ┌────────┐  ┌─────┐│  │
│  │  │Spider  │  │Scanner │  │Fuzzer││  │
│  │  └────────┘  └────────┘  └─────┘│  │
│  └──────────────────────────────────┘  │
│  ┌──────────────────────────────────┐  │
│  │      Proxy (Interceptor)         │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
         ↓                    ↑
    [Requests]           [Responses]
         ↓                    ↑
┌────────────────────────────────────────┐
│        Target Application              │
└────────────────────────────────────────┘
```

### 4.2 Python自动化脚本

```python
#!/usr/bin/env python3
"""
OWASP ZAP自动化扫描脚本
"""
import time
import json
from zapv2 import ZAPv2

class ZAPScanner:
    def __init__(self, target_url, api_key='changeme', zap_proxy='http://127.0.0.1:8080'):
        self.target = target_url
        self.zap = ZAPv2(apikey=api_key, proxies={'http': zap_proxy, 'https': zap_proxy})

    def spider_scan(self):
        """爬虫扫描"""
        print(f'[*] Spider scanning: {self.target}')
        scan_id = self.zap.spider.scan(self.target)

        # 等待爬虫完成
        while int(self.zap.spider.status(scan_id)) < 100:
            progress = self.zap.spider.status(scan_id)
            print(f'[*] Spider progress: {progress}%')
            time.sleep(5)

        print('[+] Spider scan completed')
        urls = self.zap.spider.results(scan_id)
        print(f'[+] Found {len(urls)} URLs')
        return urls

    def ajax_spider_scan(self):
        """AJAX爬虫扫描（用于SPA应用）"""
        print(f'[*] AJAX Spider scanning: {self.target}')
        self.zap.ajaxSpider.scan(self.target)

        # 等待AJAX爬虫完成
        while self.zap.ajaxSpider.status == 'running':
            print(f'[*] AJAX Spider running...')
            time.sleep(5)

        print('[+] AJAX Spider scan completed')
        urls = self.zap.ajaxSpider.results()
        return urls

    def active_scan(self):
        """主动扫描"""
        print(f'[*] Active scanning: {self.target}')
        scan_id = self.zap.ascan.scan(self.target)

        # 等待主动扫描完成
        while int(self.zap.ascan.status(scan_id)) < 100:
            progress = self.zap.ascan.status(scan_id)
            print(f'[*] Active scan progress: {progress}%')
            time.sleep(10)

        print('[+] Active scan completed')

    def get_alerts(self, risk_level='High'):
        """获取告警"""
        alerts = self.zap.core.alerts(baseurl=self.target)

        # 按风险级别过滤
        risk_levels = {'High': 3, 'Medium': 2, 'Low': 1, 'Informational': 0}
        min_risk = risk_levels.get(risk_level, 0)

        filtered_alerts = [
            alert for alert in alerts
            if risk_levels.get(alert['risk'], 0) >= min_risk
        ]

        return filtered_alerts

    def generate_report(self, output_file='zap_report.html'):
        """生成报告"""
        print(f'[*] Generating report: {output_file}')

        # HTML报告
        html_report = self.zap.core.htmlreport()
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_report)

        # JSON报告
        json_file = output_file.replace('.html', '.json')
        json_report = self.zap.core.jsonreport()
        with open(json_file, 'w', encoding='utf-8') as f:
            f.write(json_report)

        print(f'[+] Report saved to {output_file}')

    def full_scan(self):
        """完整扫描流程"""
        print('[*] Starting full security scan')

        # 1. 访问目标
        print(f'[*] Accessing target: {self.target}')
        self.zap.urlopen(self.target)
        time.sleep(2)

        # 2. 爬虫扫描
        self.spider_scan()

        # 3. AJAX爬虫（如果是SPA应用）
        # self.ajax_spider_scan()

        # 4. 主动扫描
        self.active_scan()

        # 5. 获取结果
        alerts = self.get_alerts(risk_level='Low')

        # 6. 输出摘要
        print('\n[+] Scan Summary:')
        risk_summary = {}
        for alert in alerts:
            risk = alert['risk']
            risk_summary[risk] = risk_summary.get(risk, 0) + 1

        for risk, count in sorted(risk_summary.items(), reverse=True):
            print(f'  {risk}: {count}')

        # 7. 生成报告
        self.generate_report()

        return alerts

def main():
    # 配置
    target_url = 'http://testphp.vulnweb.com'
    api_key = 'changeme'

    # 执行扫描
    scanner = ZAPScanner(target_url, api_key)
    alerts = scanner.full_scan()

    # 输出高危漏洞
    print('\n[!] High Risk Vulnerabilities:')
    for alert in alerts:
        if alert['risk'] == 'High':
            print(f"  - {alert['alert']}")
            print(f"    URL: {alert['url']}")
            print(f"    Description: {alert['description'][:100]}...")
            print()

if __name__ == '__main__':
    main()
```

### 4.3 Docker部署ZAP

```bash
# 启动ZAP容器（守护模式）
docker run -d \
  --name zap \
  -p 8080:8080 \
  -v $(pwd)/zap:/zap/wrk:rw \
  owasp/zap2docker-stable zap.sh -daemon \
  -host 0.0.0.0 -port 8080 \
  -config api.key=changeme \
  -config api.addrs.addr.name=.* \
  -config api.addrs.addr.regex=true

# 执行基线扫描
docker run --rm \
  -v $(pwd)/reports:/zap/wrk:rw \
  owasp/zap2docker-stable zap-baseline.py \
  -t https://example.com \
  -r baseline_report.html

# 执行完整扫描
docker run --rm \
  -v $(pwd)/reports:/zap/wrk:rw \
  owasp/zap2docker-stable zap-full-scan.py \
  -t https://example.com \
  -r full_report.html
```

## 5. Burp Suite渗透测试

### 5.1 Burp Suite扩展开发

```python
"""
Burp Suite扩展：自定义漏洞扫描器
"""
from burp import IBurpExtender, IScannerCheck, IScanIssue
from java.net import URL
import re

class BurpExtender(IBurpExtender, IScannerCheck):

    def registerExtenderCallbacks(self, callbacks):
        self._callbacks = callbacks
        self._helpers = callbacks.getHelpers()

        callbacks.setExtensionName("Custom Security Scanner")
        callbacks.registerScannerCheck(self)

        print("Custom Security Scanner loaded")

    def doPassiveScan(self, baseRequestResponse):
        """被动扫描"""
        issues = []

        # 获取响应
        response = baseRequestResponse.getResponse()
        if response is None:
            return None

        response_str = self._helpers.bytesToString(response)

        # 检查敏感信息泄露
        issues.extend(self._check_sensitive_data(baseRequestResponse, response_str))

        # 检查安全头缺失
        issues.extend(self._check_security_headers(baseRequestResponse, response_str))

        return issues if issues else None

    def doActiveScan(self, baseRequestResponse, insertionPoint):
        """主动扫描"""
        issues = []

        # SQL注入测试
        sql_payloads = ["'", "1' OR '1'='1", "1; DROP TABLE users--"]
        for payload in sql_payloads:
            check_request = insertionPoint.buildRequest(payload)
            check_response = self._callbacks.makeHttpRequest(
                baseRequestResponse.getHttpService(),
                check_request
            )

            if self._is_sql_injection(check_response):
                issues.append(CustomScanIssue(
                    baseRequestResponse.getHttpService(),
                    self._helpers.analyzeRequest(baseRequestResponse).getUrl(),
                    [check_response],
                    "SQL Injection",
                    "Possible SQL injection vulnerability",
                    "High"
                ))
                break

        return issues if issues else None

    def _check_sensitive_data(self, baseRequestResponse, response_str):
        """检查敏感数据泄露"""
        issues = []

        # 检查API密钥
        api_key_pattern = r'(api[_-]?key|apikey)["\']?\s*[:=]\s*["\']([a-zA-Z0-9]{32,})'
        matches = re.findall(api_key_pattern, response_str, re.IGNORECASE)

        if matches:
            issues.append(CustomScanIssue(
                baseRequestResponse.getHttpService(),
                self._helpers.analyzeRequest(baseRequestResponse).getUrl(),
                [baseRequestResponse],
                "API Key Exposure",
                "API key found in response",
                "High"
            ))

        return issues

    def _check_security_headers(self, baseRequestResponse, response_str):
        """检查安全响应头"""
        issues = []

        required_headers = {
            'X-Frame-Options': 'Clickjacking protection missing',
            'X-Content-Type-Options': 'MIME sniffing protection missing',
            'Content-Security-Policy': 'CSP header missing',
            'Strict-Transport-Security': 'HSTS header missing'
        }

        response_lower = response_str.lower()

        for header, description in required_headers.items():
            if header.lower() not in response_lower:
                issues.append(CustomScanIssue(
                    baseRequestResponse.getHttpService(),
                    self._helpers.analyzeRequest(baseRequestResponse).getUrl(),
                    [baseRequestResponse],
                    f"Missing Security Header: {header}",
                    description,
                    "Low"
                ))

        return issues

    def _is_sql_injection(self, response):
        """检测SQL注入特征"""
        response_str = self._helpers.bytesToString(response.getResponse())

        sql_errors = [
            "SQL syntax",
            "mysql_fetch",
            "ORA-01",
            "PostgreSQL.*ERROR",
            "SQLite/JDBCDriver"
        ]

        for error in sql_errors:
            if re.search(error, response_str, re.IGNORECASE):
                return True

        return False

    def consolidateDuplicateIssues(self, existingIssue, newIssue):
        if existingIssue.getIssueName() == newIssue.getIssueName():
            return -1  # 保留已有问题
        return 0

class CustomScanIssue(IScanIssue):
    def __init__(self, httpService, url, httpMessages, name, detail, severity):
        self._httpService = httpService
        self._url = url
        self._httpMessages = httpMessages
        self._name = name
        self._detail = detail
        self._severity = severity

    def getUrl(self):
        return self._url

    def getIssueName(self):
        return self._name

    def getIssueType(self):
        return 0

    def getSeverity(self):
        return self._severity

    def getConfidence(self):
        return "Certain"

    def getIssueBackground(self):
        return None

    def getRemediationBackground(self):
        return None

    def getIssueDetail(self):
        return self._detail

    def getRemediationDetail(self):
        return None

    def getHttpMessages(self):
        return self._httpMessages

    def getHttpService(self):
        return self._httpService
```

## 6. 完整安全测试流程

### 6.1 企业级安全测试Pipeline

```yaml
# .gitlab-ci.yml - 完整安全测试流程
stages:
  - build
  - sast
  - sca
  - dast
  - report

variables:
  SECURE_LOG_LEVEL: "info"

build:
  stage: build
  script:
    - mvn clean package
  artifacts:
    paths:
      - target/*.jar

# 静态代码分析
sonarqube_scan:
  stage: sast
  image: sonarsource/sonar-scanner-cli:latest
  script:
    - sonar-scanner
      -Dsonar.projectKey=$CI_PROJECT_NAME
      -Dsonar.sources=src
      -Dsonar.host.url=$SONAR_HOST_URL
      -Dsonar.login=$SONAR_TOKEN
  allow_failure: false

# 依赖扫描
snyk_scan:
  stage: sca
  image: snyk/snyk:maven
  script:
    - snyk auth $SNYK_TOKEN
    - snyk test --severity-threshold=high
    - snyk monitor
  artifacts:
    reports:
      dependency_scanning: snyk-report.json

# 容器镜像扫描
trivy_scan:
  stage: sca
  image: aquasec/trivy:latest
  script:
    - trivy image --severity HIGH,CRITICAL
      --format json --output trivy-report.json
      $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  artifacts:
    reports:
      container_scanning: trivy-report.json

# 动态应用安全测试
zap_scan:
  stage: dast
  image: owasp/zap2docker-stable
  script:
    - mkdir -p /zap/wrk
    - zap-baseline.py
      -t $TARGET_URL
      -r zap-report.html
      -J zap-report.json
  artifacts:
    paths:
      - zap-report.html
      - zap-report.json
    reports:
      dast: zap-report.json
  only:
    - main

# 生成综合报告
security_report:
  stage: report
  image: python:3.11
  script:
    - pip install jinja2
    - python scripts/generate_security_report.py
  artifacts:
    paths:
      - security-report.html
    expire_in: 30 days
```

### 6.2 安全测试报告生成

```python
#!/usr/bin/env python3
"""
综合安全测试报告生成器
"""
import json
from datetime import datetime
from jinja2 import Template

def generate_security_report():
    # 读取各工具报告
    sonar_data = load_sonar_report()
    snyk_data = load_snyk_report()
    zap_data = load_zap_report()

    # 汇总数据
    summary = {
        'scan_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_issues': 0,
        'critical': 0,
        'high': 0,
        'medium': 0,
        'low': 0,
        'tools': {
            'sonarqube': sonar_data,
            'snyk': snyk_data,
            'zap': zap_data
        }
    }

    # 统计漏洞
    for tool_data in summary['tools'].values():
        for severity, count in tool_data['severity_counts'].items():
            summary['total_issues'] += count
            if severity in summary:
                summary[severity] += count

    # 生成HTML报告
    template = Template('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Security Test Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .summary { background: #f5f5f5; padding: 20px; border-radius: 5px; }
            .critical { color: #d32f2f; }
            .high { color: #f57c00; }
            .medium { color: #fbc02d; }
            .low { color: #388e3c; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #1976d2; color: white; }
        </style>
    </head>
    <body>
        <h1>Security Test Report</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>Scan Date: {{ scan_date }}</p>
            <p>Total Issues: {{ total_issues }}</p>
            <p class="critical">Critical: {{ critical }}</p>
            <p class="high">High: {{ high }}</p>
            <p class="medium">Medium: {{ medium }}</p>
            <p class="low">Low: {{ low }}</p>
        </div>

        <h2>SonarQube Analysis</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Code Coverage</td><td>{{ tools.sonarqube.coverage }}%</td></tr>
            <tr><td>Security Hotspots</td><td>{{ tools.sonarqube.security_hotspots }}</td></tr>
            <tr><td>Vulnerabilities</td><td>{{ tools.sonarqube.vulnerabilities }}</td></tr>
        </table>

        <h2>Snyk Dependency Scan</h2>
        <table>
            <tr><th>Vulnerability</th><th>Severity</th><th>Package</th></tr>
            {% for vuln in tools.snyk.vulnerabilities %}
            <tr>
                <td>{{ vuln.title }}</td>
                <td class="{{ vuln.severity }}">{{ vuln.severity }}</td>
                <td>{{ vuln.package }}</td>
            </tr>
            {% endfor %}
        </table>

        <h2>OWASP ZAP DAST</h2>
        <table>
            <tr><th>Alert</th><th>Risk</th><th>URL</th></tr>
            {% for alert in tools.zap.alerts %}
            <tr>
                <td>{{ alert.name }}</td>
                <td class="{{ alert.risk }}">{{ alert.risk }}</td>
                <td>{{ alert.url }}</td>
            </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    ''')

    html_report = template.render(**summary)

    with open('security-report.html', 'w', encoding='utf-8') as f:
        f.write(html_report)

    print('[+] Security report generated: security-report.html')

    # 检查是否通过
    if summary['critical'] > 0 or summary['high'] > 5:
        print('[!] Security gate FAILED')
        exit(1)
    else:
        print('[+] Security gate PASSED')

def load_sonar_report():
    # 实际实现需要调用SonarQube API
    return {
        'coverage': 85.2,
        'security_hotspots': 3,
        'vulnerabilities': 1,
        'severity_counts': {'high': 1, 'medium': 2}
    }

def load_snyk_report():
    try:
        with open('snyk-report.json') as f:
            data = json.load(f)
        return {
            'vulnerabilities': data.get('vulnerabilities', []),
            'severity_counts': {
                'critical': len([v for v in data['vulnerabilities'] if v['severity'] == 'critical']),
                'high': len([v for v in data['vulnerabilities'] if v['severity'] == 'high'])
            }
        }
    except FileNotFoundError:
        return {'vulnerabilities': [], 'severity_counts': {}}

def load_zap_report():
    try:
        with open('zap-report.json') as f:
            data = json.load(f)
        return {
            'alerts': data.get('site', [{}])[0].get('alerts', []),
            'severity_counts': {
                'high': len([a for a in data['site'][0]['alerts'] if a['riskcode'] == '3']),
                'medium': len([a for a in data['site'][0]['alerts'] if a['riskcode'] == '2'])
            }
        }
    except (FileNotFoundError, IndexError):
        return {'alerts': [], 'severity_counts': {}}

if __name__ == '__main__':
    generate_security_report()
```

## 7. 最佳实践总结

### 7.1 安全测试成熟度模型

```
Level 1: 初始级
└── 手动渗透测试，无自动化

Level 2: 可重复级
├── CI/CD集成基础扫描
└── 定期依赖更新

Level 3: 已定义级
├── 多工具组合扫描
├── Security Champions计划
└── 安全编码培训

Level 4: 管理级
├── 实时监控与告警
├── 漏洞SLA管理
└── 安全度量指标

Level 5: 优化级
├── AI辅助漏洞分析
├── 持续优化与改进
└── 行业领先实践
```

### 7.2 关键指标

- **漏洞修复时间**: Critical < 24h, High < 7d
- **代码覆盖率**: >= 80%
- **依赖更新频率**: 每月检查
- **安全培训**: 每季度一次
- **渗透测试**: 每半年一次
