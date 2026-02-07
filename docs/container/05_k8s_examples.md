# 05_k8s_examples

```yaml
# ============================================================
#           Kubernetes 实战配置示例大全
# ============================================================
# 本文件包含生产级别的 K8s 配置示例
# 所有配置都有详细的中文注释
#
# 目录：
# 1. 完整的 Web 应用部署示例
# 2. 数据库有状态应用部署
# 3. 微服务架构示例
# 4. 定时任务和批处理
# 5. 自动扩缩容配置
# 6. 网络策略和安全
# 7. 监控和日志收集
# ============================================================


# ============================================================
#     第一部分：完整的 Web 应用部署示例
# ============================================================
# 这是一个完整的前后端分离应用部署示例
# 包含：Nginx 前端 + Node.js 后端 + PostgreSQL 数据库


# ------------------------------------------------------------
#                    1.1 命名空间
# ------------------------------------------------------------
# 为应用创建独立的命名空间，实现资源隔离
apiVersion: v1
kind: Namespace
metadata:
  name: myapp
  labels:
    # 标签用于组织和选择资源
    name: myapp
    environment: production
    team: backend

---
# ------------------------------------------------------------
#                    1.2 ConfigMap - 应用配置
# ------------------------------------------------------------
# ConfigMap 存储非敏感的配置数据
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: myapp
  labels:
    app: myapp
data:
  # 简单的键值对配置
  APP_ENV: "production"
  LOG_LEVEL: "info"
  API_VERSION: "v1"

  # 完整的配置文件（多行字符串）
  # 使用 | 表示保留换行符的多行字符串
  nginx.conf: |
    # Nginx 配置文件
    server {
        listen 80;
        server_name localhost;

        # 静态文件
        location / {
            root /usr/share/nginx/html;
            index index.html;
            # SPA 应用的 history 模式
            try_files $uri $uri/ /index.html;
        }

        # API 代理
        location /api/ {
            # 转发到后端服务
            # 使用 K8s 服务名称作为主机名
            proxy_pass http://api-service:3000/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_cache_bypass $http_upgrade;
        }

        # 健康检查端点
        location /health {
            return 200 'OK';
            add_header Content-Type text/plain;
        }
    }

  # 数据库初始化脚本
  init.sql: |
    -- 创建数据库表
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS posts (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        title VARCHAR(200) NOT NULL,
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- 创建索引
    CREATE INDEX idx_posts_user_id ON posts(user_id);

---
# ------------------------------------------------------------
#                    1.3 Secret - 敏感信息
# ------------------------------------------------------------
# Secret 存储敏感数据（密码、密钥等）
# 注意：生产环境建议使用外部密钥管理系统
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: myapp
  labels:
    app: myapp
# Opaque 是最常见的 Secret 类型
type: Opaque
# data 中的值必须是 base64 编码
# 生成方式：echo -n 'your-value' | base64
data:
  # 数据库密码：mypassword123
  DB_PASSWORD: bXlwYXNzd29yZDEyMw==
  # JWT 密钥：my-super-secret-jwt-key
  JWT_SECRET: bXktc3VwZXItc2VjcmV0LWp3dC1rZXk=
  # API 密钥：api-key-12345
  API_KEY: YXBpLWtleS0xMjM0NQ==

# 也可以使用 stringData（明文，K8s 会自动编码）
# stringData:
#   DB_PASSWORD: mypassword123

---
# ------------------------------------------------------------
#              1.4 PersistentVolumeClaim - 持久存储
# ------------------------------------------------------------
# 为数据库请求持久存储
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: myapp
  labels:
    app: postgres
spec:
  # 访问模式
  # ReadWriteOnce: 单节点读写
  # ReadOnlyMany: 多节点只读
  # ReadWriteMany: 多节点读写（需要特定存储类支持）
  accessModes:
    - ReadWriteOnce

  # 存储类名称
  # 不同云平台有不同的存储类
  # AWS: gp2, gp3, io1
  # GCP: standard, ssd
  # Azure: managed-premium, managed-standard
  storageClassName: standard

  resources:
    requests:
      # 请求 10GB 存储空间
      storage: 10Gi

---
# ------------------------------------------------------------
#              1.5 PostgreSQL 数据库 Deployment
# ------------------------------------------------------------
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: myapp
  labels:
    app: postgres
    tier: database
spec:
  # 数据库通常只需要 1 个副本
  # 如果需要高可用，考虑使用 StatefulSet 或数据库 Operator
  replicas: 1

  selector:
    matchLabels:
      app: postgres

  # 更新策略
  strategy:
    # 数据库使用 Recreate 策略
    # 确保同时只有一个实例访问存储
    type: Recreate

  template:
    metadata:
      labels:
        app: postgres
        tier: database
    spec:
      containers:
        - name: postgres
          image: postgres:15-alpine

          ports:
            - containerPort: 5432
              name: postgres

          # 环境变量配置
          env:
            # 从 Secret 获取密码
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: app-secrets
                  key: DB_PASSWORD

            # 直接设置的环境变量
            - name: POSTGRES_USER
              value: "appuser"
            - name: POSTGRES_DB
              value: "appdb"

            # 指定数据目录
            - name: PGDATA
              value: /var/lib/postgresql/data/pgdata

          # 资源限制
          resources:
            requests:
              # 请求的最小资源
              memory: "256Mi"
              cpu: "250m"
            limits:
              # 最大可用资源
              memory: "512Mi"
              cpu: "500m"

          # 存储卷挂载
          volumeMounts:
            # 数据目录
            - name: postgres-data
              mountPath: /var/lib/postgresql/data

            # 初始化脚本
            - name: init-script
              mountPath: /docker-entrypoint-initdb.d
              readOnly: true

          # 存活探针
          # 检测容器是否还在运行
          livenessProbe:
            exec:
              command:
                - pg_isready
                - -U
                - appuser
                - -d
                - appdb
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3

          # 就绪探针
          # 检测容器是否准备好接收流量
          readinessProbe:
            exec:
              command:
                - pg_isready
                - -U
                - appuser
                - -d
                - appdb
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3

      # 卷定义
      volumes:
        # 持久化数据卷
        - name: postgres-data
          persistentVolumeClaim:
            claimName: postgres-pvc

        # 配置文件卷
        - name: init-script
          configMap:
            name: app-config
            items:
              - key: init.sql
                path: init.sql

---
# ------------------------------------------------------------
#              1.6 PostgreSQL Service
# ------------------------------------------------------------
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: myapp
  labels:
    app: postgres
spec:
  # ClusterIP: 只在集群内部访问
  type: ClusterIP

  selector:
    app: postgres

  ports:
    - port: 5432
      targetPort: 5432
      name: postgres

---
# ------------------------------------------------------------
#              1.7 后端 API Deployment
# ------------------------------------------------------------
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  namespace: myapp
  labels:
    app: api
    tier: backend
spec:
  # 后端服务 3 个副本，实现负载均衡
  replicas: 3

  selector:
    matchLabels:
      app: api

  # 滚动更新策略
  strategy:
    type: RollingUpdate
    rollingUpdate:
      # 更新时最多额外创建的 Pod 数量
      maxSurge: 1
      # 更新时最多不可用的 Pod 数量
      maxUnavailable: 0

  template:
    metadata:
      labels:
        app: api
        tier: backend
      # 添加 annotation 触发滚动更新
      annotations:
        # 当配置更改时，修改这个值触发重新部署
        rollout-version: "1"
    spec:
      # 初始化容器
      # 在主容器启动前执行，常用于等待依赖服务
      initContainers:
        - name: wait-for-postgres
          image: busybox:1.36
          command:
            - sh
            - -c
            # 等待 PostgreSQL 服务可用
            - |
              until nc -z postgres-service 5432; do
                echo "等待 PostgreSQL 启动..."
                sleep 2
              done
              echo "PostgreSQL 已就绪！"

      # 主容器
      containers:
        - name: api
          image: node:18-alpine

          # 工作目录
          workingDir: /app

          # 启动命令
          command: ["npm", "run", "start:prod"]

          ports:
            - containerPort: 3000
              name: http

          # 环境变量
          env:
            # 从 ConfigMap 获取
            - name: NODE_ENV
              valueFrom:
                configMapKeyRef:
                  name: app-config
                  key: APP_ENV

            - name: LOG_LEVEL
              valueFrom:
                configMapKeyRef:
                  name: app-config
                  key: LOG_LEVEL

            # 数据库连接信息
            - name: DB_HOST
              value: "postgres-service"
            - name: DB_PORT
              value: "5432"
            - name: DB_USER
              value: "appuser"
            - name: DB_NAME
              value: "appdb"

            # 从 Secret 获取敏感信息
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: app-secrets
                  key: DB_PASSWORD

            - name: JWT_SECRET
              valueFrom:
                secretKeyRef:
                  name: app-secrets
                  key: JWT_SECRET

          # 资源配置
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "500m"

          # 挂载应用代码
          volumeMounts:
            - name: app-code
              mountPath: /app

          # 健康检查
          livenessProbe:
            httpGet:
              path: /health
              port: 3000
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3

          readinessProbe:
            httpGet:
              path: /ready
              port: 3000
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3

          # 启动探针（适用于启动慢的应用）
          startupProbe:
            httpGet:
              path: /health
              port: 3000
            # 最多等待 300 秒（10 * 30）
            failureThreshold: 30
            periodSeconds: 10

      # 反亲和性：尽量将 Pod 分散到不同节点
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app: api
                topologyKey: kubernetes.io/hostname

      # 卷定义
      volumes:
        - name: app-code
          emptyDir: {}

---
# ------------------------------------------------------------
#              1.8 后端 API Service
# ------------------------------------------------------------
apiVersion: v1
kind: Service
metadata:
  name: api-service
  namespace: myapp
  labels:
    app: api
spec:
  type: ClusterIP

  selector:
    app: api

  ports:
    - port: 3000
      targetPort: 3000
      name: http

---
# ------------------------------------------------------------
#              1.9 前端 Nginx Deployment
# ------------------------------------------------------------
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: myapp
  labels:
    app: frontend
    tier: frontend
spec:
  replicas: 2

  selector:
    matchLabels:
      app: frontend

  template:
    metadata:
      labels:
        app: frontend
        tier: frontend
    spec:
      containers:
        - name: nginx
          image: nginx:alpine

          ports:
            - containerPort: 80
              name: http

          # 挂载 Nginx 配置
          volumeMounts:
            - name: nginx-config
              mountPath: /etc/nginx/conf.d/default.conf
              subPath: nginx.conf
              readOnly: true

            # 挂载静态文件（实际项目中通常使用持久卷或 ConfigMap）
            - name: static-files
              mountPath: /usr/share/nginx/html

          resources:
            requests:
              memory: "64Mi"
              cpu: "50m"
            limits:
              memory: "128Mi"
              cpu: "100m"

          livenessProbe:
            httpGet:
              path: /health
              port: 80
            initialDelaySeconds: 10
            periodSeconds: 10

          readinessProbe:
            httpGet:
              path: /health
              port: 80
            initialDelaySeconds: 5
            periodSeconds: 5

      volumes:
        - name: nginx-config
          configMap:
            name: app-config

        - name: static-files
          emptyDir: {}

---
# ------------------------------------------------------------
#              1.10 前端 Service
# ------------------------------------------------------------
apiVersion: v1
kind: Service
metadata:
  name: frontend-service
  namespace: myapp
  labels:
    app: frontend
spec:
  type: ClusterIP

  selector:
    app: frontend

  ports:
    - port: 80
      targetPort: 80
      name: http

---
# ------------------------------------------------------------
#              1.11 Ingress - 外部访问入口
# ------------------------------------------------------------
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
  namespace: myapp
  labels:
    app: myapp
  annotations:
    # Nginx Ingress Controller 注解
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "60"

    # SSL 重定向
    nginx.ingress.kubernetes.io/ssl-redirect: "true"

    # 启用 CORS
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "*"

    # 速率限制
    nginx.ingress.kubernetes.io/limit-rps: "100"

    # 自动申请 Let's Encrypt 证书（需要 cert-manager）
    # cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  # 指定 Ingress 类
  ingressClassName: nginx

  # TLS 配置
  tls:
    - hosts:
        - myapp.example.com
        - api.myapp.example.com
      # 引用包含 TLS 证书的 Secret
      secretName: myapp-tls-secret

  rules:
    # 主域名 - 前端
    - host: myapp.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: frontend-service
                port:
                  number: 80

          # API 路径转发到后端
          - path: /api
            pathType: Prefix
            backend:
              service:
                name: api-service
                port:
                  number: 3000

    # API 子域名
    - host: api.myapp.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: api-service
                port:
                  number: 3000


---
# ============================================================
#     第二部分：StatefulSet - 有状态应用部署
# ============================================================
# StatefulSet 用于部署有状态应用，如数据库集群
# 特点：
# - 稳定的网络标识（Pod 名称固定）
# - 稳定的存储（每个 Pod 有自己的 PVC）
# - 有序部署和删除


# ------------------------------------------------------------
#              2.1 Headless Service
# ------------------------------------------------------------
# StatefulSet 需要 Headless Service 来实现稳定的网络标识
apiVersion: v1
kind: Service
metadata:
  name: redis-headless
  namespace: myapp
  labels:
    app: redis
spec:
  # ClusterIP 设为 None，创建 Headless Service
  clusterIP: None

  selector:
    app: redis

  ports:
    - port: 6379
      name: redis

---
# ------------------------------------------------------------
#              2.2 Redis Cluster StatefulSet
# ------------------------------------------------------------
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
  namespace: myapp
  labels:
    app: redis
spec:
  # 关联的 Headless Service
  serviceName: redis-headless

  # 3 个副本组成集群
  replicas: 3

  selector:
    matchLabels:
      app: redis

  # Pod 管理策略
  # OrderedReady: 按顺序创建和删除（默认）
  # Parallel: 并行创建和删除
  podManagementPolicy: OrderedReady

  # 更新策略
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      # 一次更新一个 Pod
      partition: 0

  template:
    metadata:
      labels:
        app: redis
    spec:
      # 终止宽限期
      terminationGracePeriodSeconds: 30

      containers:
        - name: redis
          image: redis:7-alpine

          ports:
            - containerPort: 6379
              name: redis

          # 启动命令配置 Redis 集群
          command:
            - redis-server
          args:
            - --appendonly
            - "yes"
            - --appendfsync
            - everysec
            # 集群模式配置
            # - --cluster-enabled
            # - "yes"

          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "200m"

          volumeMounts:
            - name: redis-data
              mountPath: /data

          livenessProbe:
            exec:
              command:
                - redis-cli
                - ping
            initialDelaySeconds: 30
            periodSeconds: 10

          readinessProbe:
            exec:
              command:
                - redis-cli
                - ping
            initialDelaySeconds: 5
            periodSeconds: 5

  # 卷声明模板
  # StatefulSet 会为每个 Pod 创建独立的 PVC
  volumeClaimTemplates:
    - metadata:
        name: redis-data
        labels:
          app: redis
      spec:
        accessModes:
          - ReadWriteOnce
        storageClassName: standard
        resources:
          requests:
            storage: 5Gi


---
# ============================================================
#     第三部分：DaemonSet - 每个节点运行一个 Pod
# ============================================================
# DaemonSet 确保每个（或指定的）节点运行一个 Pod
# 常用于：日志收集、监控代理、网络插件


# ------------------------------------------------------------
#              3.1 日志收集 DaemonSet
# ------------------------------------------------------------
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
  namespace: kube-system
  labels:
    app: fluentd
    component: logging
spec:
  selector:
    matchLabels:
      app: fluentd

  # 更新策略
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      # 最大不可用节点数
      maxUnavailable: 1

  template:
    metadata:
      labels:
        app: fluentd
        component: logging
    spec:
      # 容忍 master 节点的污点
      tolerations:
        - key: node-role.kubernetes.io/master
          effect: NoSchedule
        - key: node-role.kubernetes.io/control-plane
          effect: NoSchedule

      # 使用主机网络
      hostNetwork: true

      # 使用主机 PID 命名空间
      hostPID: true

      containers:
        - name: fluentd
          image: fluent/fluentd-kubernetes-daemonset:v1.16-debian-elasticsearch8

          env:
            # Elasticsearch 配置
            - name: FLUENT_ELASTICSEARCH_HOST
              value: "elasticsearch.logging.svc.cluster.local"
            - name: FLUENT_ELASTICSEARCH_PORT
              value: "9200"
            - name: FLUENT_ELASTICSEARCH_SCHEME
              value: "http"

          resources:
            requests:
              memory: "200Mi"
              cpu: "100m"
            limits:
              memory: "500Mi"
              cpu: "500m"

          volumeMounts:
            # 挂载容器日志目录
            - name: varlog
              mountPath: /var/log
              readOnly: true

            # 挂载 Docker 容器日志
            - name: containers
              mountPath: /var/lib/docker/containers
              readOnly: true

            # Fluentd 配置
            - name: config
              mountPath: /fluentd/etc

      volumes:
        - name: varlog
          hostPath:
            path: /var/log

        - name: containers
          hostPath:
            path: /var/lib/docker/containers

        - name: config
          configMap:
            name: fluentd-config


---
# ============================================================
#     第四部分：Job 和 CronJob - 任务调度
# ============================================================


# ------------------------------------------------------------
#              4.1 一次性 Job
# ------------------------------------------------------------
# Job 用于运行一次性任务
apiVersion: batch/v1
kind: Job
metadata:
  name: database-migration
  namespace: myapp
  labels:
    app: migration
spec:
  # 任务完成后保留多少秒（用于查看日志）
  ttlSecondsAfterFinished: 3600

  # 并行度
  parallelism: 1

  # 完成数（需要成功完成的 Pod 数量）
  completions: 1

  # 重试次数
  backoffLimit: 3

  # 活动截止时间（秒）
  activeDeadlineSeconds: 600

  template:
    metadata:
      labels:
        app: migration
    spec:
      # Job 的 Pod 不重启
      restartPolicy: Never

      # 等待数据库就绪
      initContainers:
        - name: wait-for-db
          image: busybox:1.36
          command:
            - sh
            - -c
            - until nc -z postgres-service 5432; do sleep 2; done

      containers:
        - name: migration
          image: myapp/migration:latest

          command: ["npm", "run", "migrate"]

          env:
            - name: DB_HOST
              value: "postgres-service"
            - name: DB_PORT
              value: "5432"
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: app-secrets
                  key: DB_PASSWORD

          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "200m"

---
# ------------------------------------------------------------
#              4.2 定时任务 CronJob
# ------------------------------------------------------------
# CronJob 用于周期性执行任务
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-backup
  namespace: myapp
  labels:
    app: backup
spec:
  # Cron 表达式
  # 格式：分 时 日 月 周
  # 每天凌晨 2 点执行
  schedule: "0 2 * * *"

  # 时区（K8s 1.24+）
  # timeZone: "Asia/Shanghai"

  # 并发策略
  # Allow: 允许并发
  # Forbid: 禁止并发
  # Replace: 替换正在运行的
  concurrencyPolicy: Forbid

  # 如果错过调度，在多少秒内仍然执行
  startingDeadlineSeconds: 3600

  # 保留的成功历史数量
  successfulJobsHistoryLimit: 3

  # 保留的失败历史数量
  failedJobsHistoryLimit: 3

  # 是否暂停调度
  suspend: false

  jobTemplate:
    spec:
      template:
        metadata:
          labels:
            app: backup
        spec:
          restartPolicy: OnFailure

          containers:
            - name: backup
              image: postgres:15-alpine

              command:
                - /bin/sh
                - -c
                - |
                  # 生成备份文件名
                  BACKUP_FILE="/backup/db_$(date +%Y%m%d_%H%M%S).sql"

                  # 执行备份
                  pg_dump -h postgres-service -U appuser -d appdb > $BACKUP_FILE

                  # 压缩备份
                  gzip $BACKUP_FILE

                  # 删除 7 天前的备份
                  find /backup -name "*.sql.gz" -mtime +7 -delete

                  echo "备份完成: ${BACKUP_FILE}.gz"

              env:
                - name: PGPASSWORD
                  valueFrom:
                    secretKeyRef:
                      name: app-secrets
                      key: DB_PASSWORD

              volumeMounts:
                - name: backup-storage
                  mountPath: /backup

              resources:
                requests:
                  memory: "128Mi"
                  cpu: "100m"
                limits:
                  memory: "256Mi"
                  cpu: "200m"

          volumes:
            - name: backup-storage
              persistentVolumeClaim:
                claimName: backup-pvc


---
# ============================================================
#     第五部分：HorizontalPodAutoscaler - 自动扩缩容
# ============================================================


# ------------------------------------------------------------
#              5.1 基于 CPU 的自动扩缩容
# ------------------------------------------------------------
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
  namespace: myapp
  labels:
    app: api
spec:
  # 目标 Deployment
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api

  # 副本数范围
  minReplicas: 2
  maxReplicas: 10

  # 扩缩容指标
  metrics:
    # CPU 使用率
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          # CPU 使用率超过 70% 时扩容
          averageUtilization: 70

    # 内存使用率
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          # 内存使用率超过 80% 时扩容
          averageUtilization: 80

    # 自定义指标（需要 Prometheus Adapter）
    # - type: Pods
    #   pods:
    #     metric:
    #       name: http_requests_per_second
    #     target:
    #       type: AverageValue
    #       averageValue: "1000"

  # 扩缩容行为（K8s 1.23+）
  behavior:
    # 扩容行为
    scaleUp:
      # 稳定窗口（秒）
      stabilizationWindowSeconds: 0
      policies:
        # 每 60 秒最多增加 4 个 Pod
        - type: Pods
          value: 4
          periodSeconds: 60
        # 每 60 秒最多增加 100%
        - type: Percent
          value: 100
          periodSeconds: 60
      # 使用最大的扩容值
      selectPolicy: Max

    # 缩容行为
    scaleDown:
      # 稳定窗口，防止频繁缩容
      stabilizationWindowSeconds: 300
      policies:
        # 每 60 秒最多减少 1 个 Pod
        - type: Pods
          value: 1
          periodSeconds: 60
      selectPolicy: Min


---
# ============================================================
#     第六部分：NetworkPolicy - 网络策略
# ============================================================
# NetworkPolicy 用于控制 Pod 之间的网络流量
# 需要支持 NetworkPolicy 的 CNI 插件（如 Calico、Cilium）


# ------------------------------------------------------------
#              6.1 默认拒绝所有入站流量
# ------------------------------------------------------------
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
  namespace: myapp
spec:
  # 选择所有 Pod
  podSelector: {}

  # 策略类型
  policyTypes:
    - Ingress

---
# ------------------------------------------------------------
#              6.2 允许特定流量
# ------------------------------------------------------------
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-network-policy
  namespace: myapp
spec:
  # 应用到 API Pod
  podSelector:
    matchLabels:
      app: api

  policyTypes:
    - Ingress
    - Egress

  # 入站规则
  ingress:
    # 允许来自前端的流量
    - from:
        - podSelector:
            matchLabels:
              app: frontend
      ports:
        - protocol: TCP
          port: 3000

    # 允许来自 Ingress Controller 的流量
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 3000

  # 出站规则
  egress:
    # 允许访问数据库
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - protocol: TCP
          port: 5432

    # 允许访问 Redis
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - protocol: TCP
          port: 6379

    # 允许 DNS 查询
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53

---
# ------------------------------------------------------------
#              6.3 数据库网络策略
# ------------------------------------------------------------
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: postgres-network-policy
  namespace: myapp
spec:
  podSelector:
    matchLabels:
      app: postgres

  policyTypes:
    - Ingress
    - Egress

  ingress:
    # 只允许 API Pod 访问
    - from:
        - podSelector:
            matchLabels:
              app: api
      ports:
        - protocol: TCP
          port: 5432

  # 数据库通常不需要出站流量
  egress: []


---
# ============================================================
#     第七部分：PodDisruptionBudget - Pod 中断预算
# ============================================================
# PDB 确保在自愿中断（如升级、维护）期间
# 保持最少数量的 Pod 可用


apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: api-pdb
  namespace: myapp
spec:
  selector:
    matchLabels:
      app: api

  # 最少可用数量（二选一）
  minAvailable: 2

  # 最多不可用数量
  # maxUnavailable: 1


---
# ============================================================
#     第八部分：ResourceQuota 和 LimitRange
# ============================================================
# 资源配额和限制范围，用于多租户集群


# ------------------------------------------------------------
#              8.1 ResourceQuota - 命名空间资源配额
# ------------------------------------------------------------
apiVersion: v1
kind: ResourceQuota
metadata:
  name: myapp-quota
  namespace: myapp
spec:
  hard:
    # CPU 和内存限制
    requests.cpu: "10"
    requests.memory: "20Gi"
    limits.cpu: "20"
    limits.memory: "40Gi"

    # Pod 数量限制
    pods: "50"

    # 服务数量限制
    services: "20"
    services.loadbalancers: "2"
    services.nodeports: "5"

    # 存储限制
    persistentvolumeclaims: "10"
    requests.storage: "100Gi"

    # ConfigMap 和 Secret 限制
    configmaps: "20"
    secrets: "20"

---
# ------------------------------------------------------------
#              8.2 LimitRange - 默认资源限制
# ------------------------------------------------------------
apiVersion: v1
kind: LimitRange
metadata:
  name: myapp-limits
  namespace: myapp
spec:
  limits:
    # 容器级别限制
    - type: Container
      # 默认限制（如果 Pod 没有指定）
      default:
        cpu: "500m"
        memory: "512Mi"
      # 默认请求
      defaultRequest:
        cpu: "100m"
        memory: "128Mi"
      # 最大限制
      max:
        cpu: "2"
        memory: "4Gi"
      # 最小限制
      min:
        cpu: "50m"
        memory: "64Mi"

    # Pod 级别限制
    - type: Pod
      max:
        cpu: "4"
        memory: "8Gi"

    # PVC 级别限制
    - type: PersistentVolumeClaim
      max:
        storage: "50Gi"
      min:
        storage: "1Gi"


---
# ============================================================
#     第九部分：ServiceAccount 和 RBAC
# ============================================================
# 服务账户和基于角色的访问控制


# ------------------------------------------------------------
#              9.1 ServiceAccount
# ------------------------------------------------------------
apiVersion: v1
kind: ServiceAccount
metadata:
  name: api-service-account
  namespace: myapp
  labels:
    app: api
# 自动挂载 API 凭证
automountServiceAccountToken: true

---
# ------------------------------------------------------------
#              9.2 Role - 命名空间级别角色
# ------------------------------------------------------------
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: api-role
  namespace: myapp
rules:
  # 允许读取 ConfigMap 和 Secret
  - apiGroups: [""]
    resources: ["configmaps", "secrets"]
    verbs: ["get", "list", "watch"]

  # 允许读取 Pod 信息
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "watch"]

  # 允许创建事件
  - apiGroups: [""]
    resources: ["events"]
    verbs: ["create"]

---
# ------------------------------------------------------------
#              9.3 RoleBinding - 绑定角色到服务账户
# ------------------------------------------------------------
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: api-role-binding
  namespace: myapp
subjects:
  - kind: ServiceAccount
    name: api-service-account
    namespace: myapp
roleRef:
  kind: Role
  name: api-role
  apiGroup: rbac.authorization.k8s.io


---
# ============================================================
#     第十部分：常用工具部署示例
# ============================================================


# ------------------------------------------------------------
#              10.1 Prometheus 监控
# ------------------------------------------------------------
# 简化版 Prometheus 部署（生产环境建议使用 Helm Chart）
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: monitoring
  labels:
    app: prometheus
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      serviceAccountName: prometheus
      containers:
        - name: prometheus
          image: prom/prometheus:v2.47.0

          ports:
            - containerPort: 9090

          args:
            - "--config.file=/etc/prometheus/prometheus.yml"
            - "--storage.tsdb.path=/prometheus"
            - "--storage.tsdb.retention.time=15d"
            - "--web.enable-lifecycle"

          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"

          volumeMounts:
            - name: config
              mountPath: /etc/prometheus
            - name: data
              mountPath: /prometheus

          livenessProbe:
            httpGet:
              path: /-/healthy
              port: 9090
            initialDelaySeconds: 30
            periodSeconds: 15

          readinessProbe:
            httpGet:
              path: /-/ready
              port: 9090
            initialDelaySeconds: 5
            periodSeconds: 5

      volumes:
        - name: config
          configMap:
            name: prometheus-config
        - name: data
          persistentVolumeClaim:
            claimName: prometheus-pvc

---
# ------------------------------------------------------------
#              10.2 Grafana 可视化
# ------------------------------------------------------------
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: monitoring
  labels:
    app: grafana
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
        - name: grafana
          image: grafana/grafana:10.1.0

          ports:
            - containerPort: 3000

          env:
            - name: GF_SECURITY_ADMIN_USER
              value: "admin"
            - name: GF_SECURITY_ADMIN_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: grafana-secrets
                  key: admin-password

            # 匿名访问（可选）
            - name: GF_AUTH_ANONYMOUS_ENABLED
              value: "false"

            # 数据源配置
            - name: GF_INSTALL_PLUGINS
              value: "grafana-piechart-panel,grafana-clock-panel"

          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "200m"

          volumeMounts:
            - name: data
              mountPath: /var/lib/grafana
            - name: datasources
              mountPath: /etc/grafana/provisioning/datasources
            - name: dashboards
              mountPath: /etc/grafana/provisioning/dashboards

          livenessProbe:
            httpGet:
              path: /api/health
              port: 3000
            initialDelaySeconds: 60
            periodSeconds: 10

          readinessProbe:
            httpGet:
              path: /api/health
              port: 3000
            initialDelaySeconds: 5
            periodSeconds: 5

      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: grafana-pvc
        - name: datasources
          configMap:
            name: grafana-datasources
        - name: dashboards
          configMap:
            name: grafana-dashboards


---
# ============================================================
#           附录：常用 kubectl 命令
# ============================================================
#
# 应用配置：
#   kubectl apply -f 05_k8s_examples.yaml
#   kubectl apply -f . --recursive
#
# 查看资源：
#   kubectl get all -n myapp
#   kubectl get pods -n myapp -o wide
#   kubectl describe pod <pod-name> -n myapp
#
# 查看日志：
#   kubectl logs <pod-name> -n myapp
#   kubectl logs -f <pod-name> -n myapp  # 实时日志
#   kubectl logs <pod-name> -c <container-name> -n myapp  # 指定容器
#
# 进入容器：
#   kubectl exec -it <pod-name> -n myapp -- /bin/sh
#
# 端口转发：
#   kubectl port-forward svc/api-service 3000:3000 -n myapp
#
# 扩缩容：
#   kubectl scale deployment api --replicas=5 -n myapp
#
# 滚动更新：
#   kubectl set image deployment/api api=myapp/api:v2 -n myapp
#   kubectl rollout status deployment/api -n myapp
#   kubectl rollout undo deployment/api -n myapp
#
# 调试：
#   kubectl run debug --rm -it --image=busybox -- /bin/sh
#   kubectl debug <pod-name> -it --image=busybox -n myapp
#
# 资源使用：
#   kubectl top nodes
#   kubectl top pods -n myapp
#
# ============================================================

```
