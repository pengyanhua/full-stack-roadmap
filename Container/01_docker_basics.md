# Docker 基础教程

## 一、Docker 概述

### 什么是 Docker

Docker 是一个开源的容器化平台，用于开发、部署和运行应用程序。

```
传统部署 vs 容器化部署：

传统部署：
┌─────────────────────────────────────┐
│            Application              │
├─────────────────────────────────────┤
│     Binaries/Libraries              │
├─────────────────────────────────────┤
│         Guest OS                    │
├─────────────────────────────────────┤
│         Hypervisor                  │
├─────────────────────────────────────┤
│         Host OS                     │
├─────────────────────────────────────┤
│         Hardware                    │
└─────────────────────────────────────┘

容器化部署：
┌───────┐ ┌───────┐ ┌───────┐
│ App A │ │ App B │ │ App C │
├───────┤ ├───────┤ ├───────┤
│Bins/  │ │Bins/  │ │Bins/  │
│Libs   │ │Libs   │ │Libs   │
└───────┴─┴───────┴─┴───────┘
┌─────────────────────────────────────┐
│          Docker Engine              │
├─────────────────────────────────────┤
│            Host OS                  │
├─────────────────────────────────────┤
│           Hardware                  │
└─────────────────────────────────────┘
```

### 核心概念

| 概念 | 说明 |
|------|------|
| **镜像 (Image)** | 只读模板，包含运行应用所需的一切（代码、运行时、库、配置） |
| **容器 (Container)** | 镜像的运行实例，相互隔离的进程 |
| **仓库 (Registry)** | 存储和分发镜像的服务（如 Docker Hub） |
| **Dockerfile** | 构建镜像的脚本文件 |
| **Docker Compose** | 定义和运行多容器应用的工具 |

### 安装 Docker

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER  # 将当前用户加入 docker 组

# CentOS/RHEL
sudo yum install -y docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo systemctl enable docker

# macOS / Windows
# 下载 Docker Desktop: https://www.docker.com/products/docker-desktop

# 验证安装
docker --version
docker run hello-world
```

## 二、镜像操作

### 搜索镜像

```bash
# 在 Docker Hub 搜索镜像
docker search nginx

# 输出示例：
# NAME                  DESCRIPTION                    STARS   OFFICIAL
# nginx                 Official build of Nginx        18000   [OK]
# bitnami/nginx         Bitnami nginx Docker Image     150

# 搜索并过滤
docker search --filter=stars=100 nginx
docker search --filter=is-official=true nginx
```

### 拉取镜像

```bash
# 拉取最新版本
docker pull nginx

# 拉取指定版本（标签）
docker pull nginx:1.24
docker pull nginx:alpine     # 轻量级 Alpine 版本

# 拉取指定仓库的镜像
docker pull registry.cn-hangzhou.aliyuncs.com/library/nginx

# 从私有仓库拉取
docker login myregistry.com
docker pull myregistry.com/myapp:v1.0
```

### 查看镜像

```bash
# 列出本地镜像
docker images
docker image ls

# 输出示例：
# REPOSITORY   TAG       IMAGE ID       CREATED        SIZE
# nginx        latest    605c77e624dd   2 weeks ago    141MB
# nginx        alpine    513f9a9d8748   2 weeks ago    23.4MB
# node         18        b4cec2ee88ef   3 weeks ago    991MB

# 只显示镜像 ID
docker images -q

# 查看镜像详细信息
docker inspect nginx

# 查看镜像历史（构建层）
docker history nginx

# 查看镜像大小
docker images --format "{{.Repository}}:{{.Tag}} {{.Size}}"

# 过滤镜像
docker images --filter "dangling=true"  # 悬空镜像
docker images --filter "reference=nginx*"
```

### 删除镜像

```bash
# 删除指定镜像
docker rmi nginx
docker image rm nginx:alpine

# 强制删除（即使有容器在使用）
docker rmi -f nginx

# 删除多个镜像
docker rmi nginx redis mysql

# 删除所有悬空镜像（未被任何容器使用的）
docker image prune

# 删除所有未使用的镜像
docker image prune -a

# 删除所有镜像
docker rmi $(docker images -q)
```

### 镜像标签和推送

```bash
# 给镜像打标签
docker tag nginx:latest myregistry.com/nginx:v1.0
docker tag nginx:latest myregistry.com/nginx:latest

# 推送到仓库
docker login myregistry.com
docker push myregistry.com/nginx:v1.0

# 推送到 Docker Hub
docker login
docker tag myapp:latest username/myapp:v1.0
docker push username/myapp:v1.0
```

### 镜像导入导出

```bash
# 导出镜像为文件
docker save nginx:latest > nginx.tar
docker save -o nginx.tar nginx:latest

# 导出多个镜像
docker save nginx redis mysql > images.tar

# 导入镜像
docker load < nginx.tar
docker load -i nginx.tar

# 从容器创建镜像
docker commit <container_id> myimage:v1.0
```

## 三、容器操作

### 创建和运行容器

```bash
# 运行容器（前台）
docker run nginx

# 运行容器（后台）
docker run -d nginx

# 运行并指定名称
docker run -d --name my-nginx nginx

# 运行并映射端口
# 格式：-p <宿主机端口>:<容器端口>
docker run -d -p 8080:80 nginx
docker run -d -p 80:80 -p 443:443 nginx

# 运行并挂载数据卷
# 格式：-v <宿主机路径>:<容器路径>
docker run -d -v /data/html:/usr/share/nginx/html nginx
docker run -d -v $(pwd)/config:/etc/nginx/conf.d nginx

# 运行并设置环境变量
docker run -d -e MYSQL_ROOT_PASSWORD=123456 mysql
docker run -d --env-file .env myapp

# 运行并限制资源
docker run -d --memory=512m --cpus=1 nginx

# 运行并设置重启策略
docker run -d --restart=always nginx          # 总是重启
docker run -d --restart=unless-stopped nginx  # 除非手动停止
docker run -d --restart=on-failure:3 nginx    # 失败时最多重启3次

# 交互式运行（进入容器）
docker run -it ubuntu /bin/bash
docker run -it --rm ubuntu /bin/bash  # 退出后自动删除

# 完整示例
docker run -d \
    --name web-server \
    -p 80:80 \
    -p 443:443 \
    -v /data/nginx/html:/usr/share/nginx/html \
    -v /data/nginx/conf:/etc/nginx/conf.d \
    -v /data/nginx/logs:/var/log/nginx \
    -e TZ=Asia/Shanghai \
    --restart=unless-stopped \
    nginx:alpine
```

### 查看容器

```bash
# 查看运行中的容器
docker ps

# 查看所有容器（包括已停止的）
docker ps -a

# 输出示例：
# CONTAINER ID   IMAGE   COMMAND                  STATUS          PORTS                  NAMES
# a1b2c3d4e5f6   nginx   "/docker-entrypoint.…"   Up 2 hours      0.0.0.0:80->80/tcp     web-server

# 只显示容器 ID
docker ps -q

# 查看容器详细信息
docker inspect <container_id>

# 查看容器日志
docker logs <container_id>
docker logs -f <container_id>         # 实时跟踪
docker logs --tail 100 <container_id> # 最后100行
docker logs --since 1h <container_id> # 最近1小时

# 查看容器进程
docker top <container_id>

# 查看容器资源使用
docker stats
docker stats <container_id>

# 查看容器端口映射
docker port <container_id>

# 查看容器文件系统变化
docker diff <container_id>
```

### 容器生命周期管理

```bash
# 启动已停止的容器
docker start <container_id>

# 停止容器（优雅停止，发送 SIGTERM）
docker stop <container_id>
docker stop -t 30 <container_id>  # 等待30秒后强制停止

# 强制停止容器（发送 SIGKILL）
docker kill <container_id>

# 重启容器
docker restart <container_id>

# 暂停容器
docker pause <container_id>
docker unpause <container_id>

# 删除容器
docker rm <container_id>
docker rm -f <container_id>  # 强制删除运行中的容器

# 删除所有已停止的容器
docker container prune

# 删除所有容器
docker rm -f $(docker ps -aq)
```

### 进入容器

```bash
# 方式1：exec（推荐）
docker exec -it <container_id> /bin/bash
docker exec -it <container_id> /bin/sh   # Alpine 镜像

# 执行单条命令
docker exec <container_id> ls -la
docker exec <container_id> cat /etc/nginx/nginx.conf

# 以特定用户执行
docker exec -u root -it <container_id> /bin/bash

# 方式2：attach（不推荐，退出会停止容器）
docker attach <container_id>
# Ctrl+P Ctrl+Q 可以不停止容器退出
```

### 容器与宿主机文件传输

```bash
# 从容器复制文件到宿主机
docker cp <container_id>:/path/to/file /local/path

# 从宿主机复制文件到容器
docker cp /local/path <container_id>:/path/to/file

# 示例
docker cp web-server:/etc/nginx/nginx.conf ./nginx.conf
docker cp ./index.html web-server:/usr/share/nginx/html/
```

## 四、数据管理

### 数据卷（Volumes）

```bash
# 创建数据卷
docker volume create my-volume

# 查看所有数据卷
docker volume ls

# 查看数据卷详情
docker volume inspect my-volume

# 使用数据卷
docker run -d -v my-volume:/data nginx

# 删除数据卷
docker volume rm my-volume

# 删除所有未使用的数据卷
docker volume prune
```

### 绑定挂载（Bind Mounts）

```bash
# 绑定挂载目录
docker run -d -v /host/path:/container/path nginx

# 只读挂载
docker run -d -v /host/path:/container/path:ro nginx

# 使用 --mount 语法（更清晰）
docker run -d \
    --mount type=bind,source=/host/path,target=/container/path \
    nginx

# 只读挂载
docker run -d \
    --mount type=bind,source=/host/path,target=/container/path,readonly \
    nginx
```

### 临时文件系统（tmpfs）

```bash
# tmpfs 挂载（数据存在内存中）
docker run -d --tmpfs /tmp nginx

docker run -d \
    --mount type=tmpfs,target=/tmp,tmpfs-size=100m \
    nginx
```

## 五、网络管理

### 网络类型

```
Docker 网络类型：

1. bridge（默认）
   - 容器通过虚拟网桥连接
   - 容器间可以通过 IP 通信
   - 需要端口映射才能从外部访问

2. host
   - 容器直接使用宿主机网络
   - 性能更好，但端口可能冲突

3. none
   - 容器没有网络
   - 完全隔离

4. overlay
   - 跨主机网络
   - 用于 Docker Swarm 或 Kubernetes

5. macvlan
   - 容器有独立 MAC 地址
   - 直接连接物理网络
```

### 网络操作

```bash
# 查看网络
docker network ls

# 创建网络
docker network create my-network
docker network create --driver bridge --subnet 172.20.0.0/16 my-network

# 查看网络详情
docker network inspect my-network

# 连接容器到网络
docker network connect my-network <container_id>

# 断开容器网络
docker network disconnect my-network <container_id>

# 删除网络
docker network rm my-network

# 删除所有未使用的网络
docker network prune
```

### 容器网络通信

```bash
# 创建自定义网络
docker network create app-network

# 在同一网络中运行容器（可以通过容器名通信）
docker run -d --name db --network app-network mysql
docker run -d --name web --network app-network nginx

# 在 web 容器中可以通过 db 名称访问数据库
# ping db
# mysql -h db -u root -p

# 使用 host 网络
docker run -d --network host nginx

# 禁用网络
docker run -d --network none alpine
```

## 六、常用命令速查

### 系统命令

```bash
# 查看 Docker 信息
docker info

# 查看 Docker 版本
docker version

# 查看磁盘使用
docker system df

# 清理系统
docker system prune        # 清理悬空镜像、停止的容器、未使用的网络
docker system prune -a     # 清理所有未使用的资源
docker system prune --volumes  # 同时清理数据卷

# 查看实时事件
docker events
```

### 格式化输出

```bash
# 自定义输出格式
docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"

docker images --format "{{.Repository}}:{{.Tag}} - {{.Size}}"

docker inspect --format '{{.NetworkSettings.IPAddress}}' <container_id>

docker inspect --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <container_id>
```

### 资源限制

```bash
# 限制内存
docker run -d --memory=512m nginx
docker run -d --memory=512m --memory-swap=1g nginx

# 限制 CPU
docker run -d --cpus=1.5 nginx           # 使用 1.5 个 CPU
docker run -d --cpu-shares=512 nginx     # CPU 权重
docker run -d --cpuset-cpus="0,1" nginx  # 绑定到 CPU 0 和 1

# 更新运行中容器的资源限制
docker update --memory=1g --cpus=2 <container_id>
```

## 七、常见应用部署示例

### Nginx

```bash
docker run -d \
    --name nginx \
    -p 80:80 \
    -p 443:443 \
    -v /data/nginx/html:/usr/share/nginx/html \
    -v /data/nginx/conf.d:/etc/nginx/conf.d \
    -v /data/nginx/logs:/var/log/nginx \
    --restart=unless-stopped \
    nginx:alpine
```

### MySQL

```bash
docker run -d \
    --name mysql \
    -p 3306:3306 \
    -e MYSQL_ROOT_PASSWORD=your_password \
    -e MYSQL_DATABASE=mydb \
    -v /data/mysql:/var/lib/mysql \
    --restart=unless-stopped \
    mysql:8.0
```

### Redis

```bash
docker run -d \
    --name redis \
    -p 6379:6379 \
    -v /data/redis:/data \
    --restart=unless-stopped \
    redis:alpine redis-server --appendonly yes
```

### PostgreSQL

```bash
docker run -d \
    --name postgres \
    -p 5432:5432 \
    -e POSTGRES_PASSWORD=your_password \
    -e POSTGRES_DB=mydb \
    -v /data/postgres:/var/lib/postgresql/data \
    --restart=unless-stopped \
    postgres:15-alpine
```

### MongoDB

```bash
docker run -d \
    --name mongo \
    -p 27017:27017 \
    -e MONGO_INITDB_ROOT_USERNAME=root \
    -e MONGO_INITDB_ROOT_PASSWORD=your_password \
    -v /data/mongo:/data/db \
    --restart=unless-stopped \
    mongo:6
```
