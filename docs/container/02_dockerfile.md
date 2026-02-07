# Dockerfile 详解

## 一、Dockerfile 概述

Dockerfile 是一个文本文件，包含构建 Docker 镜像的所有指令。

```
Dockerfile 构建流程：

Dockerfile ──build──> Image ──run──> Container

┌─────────────────────────────────────────────────────────────┐
│  FROM node:18-alpine          # 基础镜像层                  │
├─────────────────────────────────────────────────────────────┤
│  WORKDIR /app                 # 工作目录层                  │
├─────────────────────────────────────────────────────────────┤
│  COPY package*.json ./        # 复制依赖文件层              │
├─────────────────────────────────────────────────────────────┤
│  RUN npm install              # 安装依赖层                  │
├─────────────────────────────────────────────────────────────┤
│  COPY . .                     # 复制源码层                  │
├─────────────────────────────────────────────────────────────┤
│  CMD ["npm", "start"]         # 启动命令（不产生新层）      │
└─────────────────────────────────────────────────────────────┘
```

### 构建镜像

```bash
# 基本构建
docker build -t myapp:v1.0 .

# 指定 Dockerfile
docker build -f Dockerfile.prod -t myapp:v1.0 .

# 构建时传递参数
docker build --build-arg VERSION=1.0 -t myapp:v1.0 .

# 不使用缓存
docker build --no-cache -t myapp:v1.0 .

# 查看构建过程
docker build --progress=plain -t myapp:v1.0 .

# 多平台构建
docker buildx build --platform linux/amd64,linux/arm64 -t myapp:v1.0 .
```

## 二、Dockerfile 指令详解

### FROM - 基础镜像

```dockerfile
# 基本用法
FROM ubuntu:22.04

# 使用特定版本
FROM node:18.17.0

# 使用轻量级镜像（推荐）
FROM node:18-alpine
FROM python:3.11-slim

# 使用 scratch（空镜像，用于静态编译的程序）
FROM scratch

# 多阶段构建时命名
FROM node:18 AS builder
FROM nginx:alpine AS production

# 使用变量
ARG BASE_IMAGE=node:18
FROM ${BASE_IMAGE}
```

### WORKDIR - 工作目录

```dockerfile
# 设置工作目录（不存在会自动创建）
WORKDIR /app

# 后续的 RUN、CMD、COPY 等指令都在此目录下执行
WORKDIR /app
COPY . .          # 复制到 /app
RUN npm install   # 在 /app 目录执行

# 可以多次使用，支持相对路径
WORKDIR /app
WORKDIR src       # 现在是 /app/src
WORKDIR ../build  # 现在是 /app/build
```

### COPY - 复制文件

```dockerfile
# 复制单个文件
COPY package.json /app/

# 复制多个文件
COPY package.json package-lock.json /app/

# 使用通配符
COPY package*.json /app/
COPY *.txt /app/

# 复制目录
COPY src/ /app/src/

# 复制当前目录所有内容
COPY . /app/

# 改变所有者
COPY --chown=node:node . /app/

# 从其他阶段复制（多阶段构建）
COPY --from=builder /app/dist /usr/share/nginx/html
```

### ADD - 添加文件（高级版 COPY）

```dockerfile
# 与 COPY 类似
ADD package.json /app/

# 自动解压 tar 文件
ADD app.tar.gz /app/

# 支持 URL（不推荐，因为不能利用缓存）
ADD https://example.com/file.tar.gz /app/

# 建议：除非需要自动解压，否则使用 COPY
```

### RUN - 执行命令

```dockerfile
# Shell 形式（使用 /bin/sh -c）
RUN apt-get update && apt-get install -y curl

# Exec 形式（推荐，更可控）
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "curl"]

# 多行命令（使用 \ 换行）
RUN apt-get update && \
    apt-get install -y \
        curl \
        vim \
        git && \
    rm -rf /var/lib/apt/lists/*

# 每个 RUN 指令创建一个新层，建议合并相关命令
# ❌ 不推荐：创建多个层
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y vim

# ✅ 推荐：合并为一个层
RUN apt-get update && \
    apt-get install -y curl vim && \
    rm -rf /var/lib/apt/lists/*
```

### CMD - 容器启动命令

```dockerfile
# Exec 形式（推荐）
CMD ["node", "app.js"]
CMD ["npm", "start"]

# Shell 形式
CMD node app.js

# 作为 ENTRYPOINT 的默认参数
ENTRYPOINT ["python"]
CMD ["app.py"]
# 运行时执行：python app.py

# 注意：
# - 只有最后一个 CMD 生效
# - 可以被 docker run 的参数覆盖
# docker run myapp node other.js  # 覆盖 CMD
```

### ENTRYPOINT - 入口点

```dockerfile
# Exec 形式（推荐）
ENTRYPOINT ["python", "app.py"]

# Shell 形式
ENTRYPOINT python app.py

# 与 CMD 配合使用
ENTRYPOINT ["python"]
CMD ["app.py"]

# 运行时：
# docker run myapp           # 执行 python app.py
# docker run myapp test.py   # 执行 python test.py（CMD 被覆盖）

# 注意：
# - ENTRYPOINT 不易被覆盖（需要 --entrypoint 参数）
# - 常用于固定执行程序，CMD 提供默认参数
```

### ENV - 环境变量

```dockerfile
# 设置环境变量
ENV NODE_ENV=production

# 设置多个
ENV NODE_ENV=production \
    PORT=3000 \
    HOST=0.0.0.0

# 旧语法（不推荐）
ENV NODE_ENV production

# 在后续指令中使用
ENV APP_HOME=/app
WORKDIR ${APP_HOME}
COPY . ${APP_HOME}

# 运行时可以被覆盖
# docker run -e NODE_ENV=development myapp
```

### ARG - 构建参数

```dockerfile
# 定义构建参数
ARG VERSION=latest
ARG NODE_VERSION=18

# 使用构建参数
FROM node:${NODE_VERSION}

ARG APP_VERSION=1.0.0
ENV VERSION=${APP_VERSION}

# 构建时传递参数
# docker build --build-arg VERSION=2.0.0 -t myapp .

# 注意：
# - ARG 只在构建时有效，不会保留在镜像中
# - 如果需要在运行时使用，需要用 ENV 保存
ARG BUILD_VERSION
ENV APP_VERSION=${BUILD_VERSION}
```

### EXPOSE - 声明端口

```dockerfile
# 声明容器监听的端口
EXPOSE 3000

# 多个端口
EXPOSE 80 443

# 指定协议
EXPOSE 80/tcp
EXPOSE 53/udp

# 注意：
# - EXPOSE 只是声明，不会自动发布端口
# - 运行时需要 -p 参数映射端口
# docker run -p 8080:3000 myapp
```

### VOLUME - 声明数据卷

```dockerfile
# 声明数据卷挂载点
VOLUME /data

# 多个挂载点
VOLUME ["/data", "/logs", "/config"]

# 注意：
# - 声明的目录会被自动挂载为匿名卷
# - 运行时可以用 -v 指定具体路径
# docker run -v /host/data:/data myapp
```

### USER - 指定用户

```dockerfile
# 创建用户并切换
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
USER appuser

# 使用 UID:GID
USER 1000:1000

# 推荐：不使用 root 用户运行应用
FROM node:18-alpine
RUN addgroup -S app && adduser -S app -G app
WORKDIR /app
COPY --chown=app:app . .
USER app
CMD ["node", "app.js"]
```

### LABEL - 元数据

```dockerfile
# 添加元数据
LABEL version="1.0"
LABEL description="My application"
LABEL maintainer="user@example.com"

# 多个标签
LABEL version="1.0" \
      description="My application" \
      maintainer="user@example.com"

# 使用 OCI 标准标签
LABEL org.opencontainers.image.title="My App"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/user/repo"
```

### HEALTHCHECK - 健康检查

```dockerfile
# 基本健康检查
HEALTHCHECK CMD curl -f http://localhost:3000/health || exit 1

# 带选项的健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# 参数说明：
# --interval：检查间隔（默认 30s）
# --timeout：超时时间（默认 30s）
# --start-period：启动等待时间（默认 0s）
# --retries：连续失败次数（默认 3）

# 禁用健康检查
HEALTHCHECK NONE

# 返回值：
# 0: healthy
# 1: unhealthy
```

### 其他指令

```dockerfile
# SHELL - 指定 shell
SHELL ["/bin/bash", "-c"]
RUN echo "Using bash"

# STOPSIGNAL - 停止信号
STOPSIGNAL SIGTERM

# ONBUILD - 触发器（在子镜像构建时执行）
ONBUILD COPY . /app
ONBUILD RUN npm install
```

## 三、多阶段构建

多阶段构建可以大幅减小最终镜像的大小。

### Node.js 应用示例

```dockerfile
# ============================================================
# 阶段 1：构建阶段
# ============================================================
FROM node:18-alpine AS builder

WORKDIR /app

# 复制依赖文件
COPY package*.json ./

# 安装依赖（包括开发依赖）
RUN npm ci

# 复制源码
COPY . .

# 构建应用
RUN npm run build

# ============================================================
# 阶段 2：生产阶段
# ============================================================
FROM node:18-alpine AS production

WORKDIR /app

# 只复制生产依赖
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

# 从构建阶段复制构建产物
COPY --from=builder /app/dist ./dist

# 创建非 root 用户
RUN addgroup -S app && adduser -S app -G app
USER app

# 设置环境变量
ENV NODE_ENV=production
ENV PORT=3000

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=3s \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

CMD ["node", "dist/main.js"]
```

### Go 应用示例

```dockerfile
# ============================================================
# 阶段 1：构建阶段
# ============================================================
FROM golang:1.21-alpine AS builder

# 安装构建依赖
RUN apk add --no-cache git

WORKDIR /app

# 复制 go mod 文件
COPY go.mod go.sum ./

# 下载依赖
RUN go mod download

# 复制源码
COPY . .

# 构建静态二进制文件
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main .

# ============================================================
# 阶段 2：运行阶段
# ============================================================
FROM scratch

# 复制 SSL 证书（用于 HTTPS 请求）
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# 复制二进制文件
COPY --from=builder /app/main /main

EXPOSE 8080

ENTRYPOINT ["/main"]
```

### 前端应用示例（React/Vue）

```dockerfile
# ============================================================
# 阶段 1：构建阶段
# ============================================================
FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

# 构建时传入环境变量
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}

RUN npm run build

# ============================================================
# 阶段 2：Nginx 运行阶段
# ============================================================
FROM nginx:alpine AS production

# 复制 Nginx 配置
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 复制构建产物
COPY --from=builder /app/dist /usr/share/nginx/html

# 创建非 root 用户
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chown -R nginx:nginx /var/cache/nginx && \
    chown -R nginx:nginx /var/log/nginx && \
    touch /var/run/nginx.pid && \
    chown -R nginx:nginx /var/run/nginx.pid

USER nginx

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

### Python 应用示例

```dockerfile
# ============================================================
# 阶段 1：构建阶段
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 创建虚拟环境
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# 阶段 2：运行阶段
# ============================================================
FROM python:3.11-slim AS production

WORKDIR /app

# 复制虚拟环境
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 复制应用代码
COPY . .

# 创建非 root 用户
RUN useradd --create-home appuser
USER appuser

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]
```

## 四、最佳实践

### 1. 选择合适的基础镜像

```dockerfile
# ✅ 推荐：使用官方镜像
FROM node:18-alpine

# ✅ 推荐：使用特定版本标签
FROM python:3.11.4-slim

# ❌ 避免：使用 latest 标签
FROM python:latest

# ❌ 避免：使用过大的镜像
FROM ubuntu:22.04  # 77MB
# ✅ 改用
FROM ubuntu:22.04-minimal  # 更小
FROM alpine:3.18           # 5MB
```

### 2. 减小镜像层数和大小

```dockerfile
# ✅ 推荐：合并 RUN 命令
RUN apt-get update && \
    apt-get install -y \
        curl \
        git \
        vim && \
    rm -rf /var/lib/apt/lists/*

# ❌ 避免：多个 RUN 命令
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y git

# ✅ 推荐：清理缓存和临时文件
RUN pip install --no-cache-dir -r requirements.txt
RUN npm ci && npm cache clean --force
```

### 3. 利用构建缓存

```dockerfile
# ✅ 推荐：先复制依赖文件，再复制源码
COPY package*.json ./
RUN npm install
COPY . .

# ❌ 避免：先复制所有文件
COPY . .
RUN npm install
# 任何文件变化都会导致重新 npm install
```

### 4. 使用 .dockerignore

```dockerignore
# .dockerignore 文件

# 依赖目录
node_modules
__pycache__
*.pyc
venv
.venv

# 构建产物
dist
build
*.egg-info

# 版本控制
.git
.gitignore
.svn

# IDE
.idea
.vscode
*.swp
*.swo

# 测试和文档
tests
test
*.md
docs
coverage

# 敏感文件
.env
.env.local
*.pem
*.key
secrets

# Docker 相关
Dockerfile*
docker-compose*
.dockerignore

# 日志
*.log
logs
```

### 5. 安全最佳实践

```dockerfile
# 1. 不使用 root 用户
RUN addgroup -S app && adduser -S app -G app
USER app

# 2. 不在镜像中存储敏感信息
# ❌ 错误
ENV DB_PASSWORD=secret123

# ✅ 正确：运行时传入
# docker run -e DB_PASSWORD=xxx myapp

# 3. 使用特定版本的基础镜像
FROM node:18.17.0-alpine3.18

# 4. 扫描镜像漏洞
# docker scout quickview myapp
# trivy image myapp

# 5. 使用只读文件系统
# docker run --read-only myapp
```

### 6. 开发环境 vs 生产环境

```dockerfile
# Dockerfile.dev - 开发环境
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
```

```dockerfile
# Dockerfile.prod - 生产环境
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
USER node
EXPOSE 3000
CMD ["node", "dist/main.js"]
```

```bash
# 使用不同的 Dockerfile
docker build -f Dockerfile.dev -t myapp:dev .
docker build -f Dockerfile.prod -t myapp:prod .
```
