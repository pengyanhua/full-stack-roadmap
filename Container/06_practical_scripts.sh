#!/bin/bash
# ============================================================
#           Docker & Kubernetes 实战脚本集合
# ============================================================
# 本文件包含常用的 Docker 和 Kubernetes 操作脚本
# 可以直接使用或作为参考
#
# 使用前请确保：
# 1. 已安装 Docker 和 kubectl
# 2. 已配置好 Kubernetes 集群访问权限
# 3. 给脚本执行权限: chmod +x 06_practical_scripts.sh
# ============================================================


# ============================================================
#     第一部分：Docker 常用操作脚本
# ============================================================


# ------------------------------------------------------------
#              1.1 Docker 环境清理脚本
# ------------------------------------------------------------
# 清理未使用的 Docker 资源，释放磁盘空间

docker_cleanup() {
    echo "=========================================="
    echo "       Docker 环境清理脚本"
    echo "=========================================="

    # 显示清理前的磁盘使用情况
    echo ""
    echo "📊 清理前磁盘使用情况："
    docker system df

    echo ""
    echo "🧹 开始清理..."

    # 1. 停止所有运行中的容器
    echo ""
    echo "➡️  停止所有运行中的容器..."
    running_containers=$(docker ps -q)
    if [ -n "$running_containers" ]; then
        docker stop $running_containers
        echo "   已停止 $(echo $running_containers | wc -w) 个容器"
    else
        echo "   没有运行中的容器"
    fi

    # 2. 删除所有已停止的容器
    echo ""
    echo "➡️  删除所有已停止的容器..."
    docker container prune -f

    # 3. 删除未使用的镜像
    echo ""
    echo "➡️  删除悬空镜像（dangling images）..."
    docker image prune -f

    # 4. 删除未使用的卷
    echo ""
    echo "➡️  删除未使用的卷..."
    docker volume prune -f

    # 5. 删除未使用的网络
    echo ""
    echo "➡️  删除未使用的网络..."
    docker network prune -f

    # 6. 综合清理（可选，会删除所有未使用资源）
    # docker system prune -a -f --volumes

    # 显示清理后的磁盘使用情况
    echo ""
    echo "📊 清理后磁盘使用情况："
    docker system df

    echo ""
    echo "✅ 清理完成！"
}


# ------------------------------------------------------------
#              1.2 Docker 镜像构建脚本
# ------------------------------------------------------------
# 构建并推送 Docker 镜像

build_and_push_image() {
    # 参数检查
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "用法: build_and_push_image <镜像名称> <版本标签>"
        echo "示例: build_and_push_image myapp v1.0.0"
        return 1
    fi

    local IMAGE_NAME=$1
    local TAG=$2
    local REGISTRY=${3:-"docker.io"}  # 默认使用 Docker Hub

    local FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

    echo "=========================================="
    echo "       Docker 镜像构建脚本"
    echo "=========================================="
    echo "镜像名称: ${FULL_IMAGE}"
    echo ""

    # 检查 Dockerfile 是否存在
    if [ ! -f "Dockerfile" ]; then
        echo "❌ 错误: 当前目录没有找到 Dockerfile"
        return 1
    fi

    # 构建镜像
    echo "🔨 开始构建镜像..."
    docker build \
        --tag "${FULL_IMAGE}" \
        --tag "${REGISTRY}/${IMAGE_NAME}:latest" \
        --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
        --build-arg VERSION=${TAG} \
        --file Dockerfile \
        .

    # 检查构建结果
    if [ $? -ne 0 ]; then
        echo "❌ 镜像构建失败"
        return 1
    fi

    echo ""
    echo "✅ 镜像构建成功"

    # 显示镜像信息
    echo ""
    echo "📦 镜像信息："
    docker images | grep ${IMAGE_NAME}

    # 询问是否推送
    echo ""
    read -p "是否推送镜像到仓库？(y/n): " push_confirm
    if [ "$push_confirm" = "y" ] || [ "$push_confirm" = "Y" ]; then
        echo ""
        echo "📤 推送镜像..."
        docker push "${FULL_IMAGE}"
        docker push "${REGISTRY}/${IMAGE_NAME}:latest"

        if [ $? -eq 0 ]; then
            echo "✅ 镜像推送成功"
        else
            echo "❌ 镜像推送失败，请检查登录状态"
        fi
    fi
}


# ------------------------------------------------------------
#              1.3 容器日志查看脚本
# ------------------------------------------------------------
# 查看和导出容器日志

view_container_logs() {
    echo "=========================================="
    echo "       容器日志查看工具"
    echo "=========================================="

    # 列出所有容器
    echo ""
    echo "📋 当前容器列表："
    docker ps -a --format "table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}"

    echo ""
    read -p "请输入容器名称或ID: " container_name

    if [ -z "$container_name" ]; then
        echo "❌ 容器名称不能为空"
        return 1
    fi

    echo ""
    echo "选择操作："
    echo "1. 查看最近 100 行日志"
    echo "2. 实时查看日志（Ctrl+C 退出）"
    echo "3. 导出全部日志到文件"
    echo "4. 查看指定时间范围的日志"
    read -p "请选择 (1-4): " choice

    case $choice in
        1)
            echo ""
            echo "📜 最近 100 行日志："
            docker logs --tail 100 "$container_name"
            ;;
        2)
            echo ""
            echo "📜 实时日志（按 Ctrl+C 退出）："
            docker logs -f "$container_name"
            ;;
        3)
            local log_file="${container_name}_$(date +%Y%m%d_%H%M%S).log"
            docker logs "$container_name" > "$log_file" 2>&1
            echo ""
            echo "✅ 日志已导出到: $log_file"
            echo "   文件大小: $(ls -lh $log_file | awk '{print $5}')"
            ;;
        4)
            read -p "请输入开始时间 (如 2024-01-01T00:00:00): " since_time
            read -p "请输入结束时间 (如 2024-01-02T00:00:00): " until_time
            echo ""
            docker logs --since "$since_time" --until "$until_time" "$container_name"
            ;;
        *)
            echo "❌ 无效选择"
            ;;
    esac
}


# ------------------------------------------------------------
#              1.4 Docker Compose 项目管理
# ------------------------------------------------------------
# Docker Compose 项目的启动、停止、重启脚本

compose_manager() {
    local compose_file=${1:-"docker-compose.yaml"}

    # 检查 compose 文件
    if [ ! -f "$compose_file" ]; then
        echo "❌ 找不到 compose 文件: $compose_file"
        return 1
    fi

    echo "=========================================="
    echo "     Docker Compose 项目管理"
    echo "=========================================="
    echo "Compose 文件: $compose_file"
    echo ""
    echo "选择操作："
    echo "1. 启动服务（后台运行）"
    echo "2. 停止服务"
    echo "3. 重启服务"
    echo "4. 查看服务状态"
    echo "5. 查看服务日志"
    echo "6. 重新构建并启动"
    echo "7. 停止并删除所有资源"
    echo "8. 扩缩容服务"
    read -p "请选择 (1-8): " choice

    case $choice in
        1)
            echo ""
            echo "🚀 启动服务..."
            docker compose -f "$compose_file" up -d
            echo ""
            echo "📊 服务状态："
            docker compose -f "$compose_file" ps
            ;;
        2)
            echo ""
            echo "⏹️  停止服务..."
            docker compose -f "$compose_file" stop
            ;;
        3)
            echo ""
            echo "🔄 重启服务..."
            docker compose -f "$compose_file" restart
            ;;
        4)
            echo ""
            echo "📊 服务状态："
            docker compose -f "$compose_file" ps
            ;;
        5)
            echo ""
            echo "📜 服务日志（按 Ctrl+C 退出）："
            docker compose -f "$compose_file" logs -f
            ;;
        6)
            echo ""
            echo "🔨 重新构建并启动..."
            docker compose -f "$compose_file" up -d --build
            ;;
        7)
            echo ""
            read -p "⚠️  确定要停止并删除所有资源吗？(y/n): " confirm
            if [ "$confirm" = "y" ]; then
                docker compose -f "$compose_file" down -v --rmi local
                echo "✅ 已停止并删除所有资源"
            fi
            ;;
        8)
            read -p "请输入服务名称: " service_name
            read -p "请输入副本数量: " replicas
            docker compose -f "$compose_file" up -d --scale ${service_name}=${replicas}
            ;;
        *)
            echo "❌ 无效选择"
            ;;
    esac
}


# ============================================================
#     第二部分：Kubernetes 常用操作脚本
# ============================================================


# ------------------------------------------------------------
#              2.1 K8s 资源查看脚本
# ------------------------------------------------------------
# 综合查看 Kubernetes 集群资源状态

k8s_status() {
    local namespace=${1:-"default"}

    echo "=========================================="
    echo "    Kubernetes 集群状态概览"
    echo "=========================================="
    echo "命名空间: $namespace"
    echo "时间: $(date)"
    echo ""

    # 节点状态
    echo "🖥️  节点状态："
    echo "----------------------------------------"
    kubectl get nodes -o wide
    echo ""

    # Pod 状态
    echo "📦 Pod 状态："
    echo "----------------------------------------"
    kubectl get pods -n "$namespace" -o wide
    echo ""

    # 服务状态
    echo "🌐 Service 状态："
    echo "----------------------------------------"
    kubectl get svc -n "$namespace"
    echo ""

    # Deployment 状态
    echo "🚀 Deployment 状态："
    echo "----------------------------------------"
    kubectl get deployments -n "$namespace"
    echo ""

    # 资源使用情况（需要 metrics-server）
    echo "📊 资源使用情况："
    echo "----------------------------------------"
    kubectl top pods -n "$namespace" 2>/dev/null || echo "   (需要安装 metrics-server)"
    echo ""

    # 最近事件
    echo "📋 最近事件："
    echo "----------------------------------------"
    kubectl get events -n "$namespace" --sort-by='.lastTimestamp' | tail -10
}


# ------------------------------------------------------------
#              2.2 K8s 应用部署脚本
# ------------------------------------------------------------
# 部署应用到 Kubernetes 集群

k8s_deploy() {
    local manifest_file=$1
    local namespace=${2:-"default"}

    if [ -z "$manifest_file" ]; then
        echo "用法: k8s_deploy <manifest文件> [命名空间]"
        echo "示例: k8s_deploy deployment.yaml production"
        return 1
    fi

    if [ ! -f "$manifest_file" ]; then
        echo "❌ 找不到文件: $manifest_file"
        return 1
    fi

    echo "=========================================="
    echo "    Kubernetes 应用部署"
    echo "=========================================="
    echo "配置文件: $manifest_file"
    echo "命名空间: $namespace"
    echo ""

    # 验证配置文件
    echo "🔍 验证配置文件..."
    kubectl apply --dry-run=client -f "$manifest_file" -n "$namespace"
    if [ $? -ne 0 ]; then
        echo "❌ 配置文件验证失败"
        return 1
    fi
    echo "✅ 配置文件验证通过"
    echo ""

    # 显示将要部署的资源
    echo "📋 将要部署的资源："
    kubectl apply --dry-run=client -f "$manifest_file" -n "$namespace" -o name
    echo ""

    # 确认部署
    read -p "确认部署？(y/n): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "取消部署"
        return 0
    fi

    # 执行部署
    echo ""
    echo "🚀 开始部署..."
    kubectl apply -f "$manifest_file" -n "$namespace"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 部署成功"
        echo ""
        echo "📊 部署状态："
        kubectl get all -n "$namespace" | grep -E "NAME|$(basename $manifest_file .yaml)"
    else
        echo ""
        echo "❌ 部署失败"
        return 1
    fi
}


# ------------------------------------------------------------
#              2.3 K8s 滚动更新脚本
# ------------------------------------------------------------
# 执行 Deployment 的滚动更新

k8s_rolling_update() {
    local deployment=$1
    local image=$2
    local namespace=${3:-"default"}

    if [ -z "$deployment" ] || [ -z "$image" ]; then
        echo "用法: k8s_rolling_update <deployment名称> <新镜像> [命名空间]"
        echo "示例: k8s_rolling_update api-deployment myapp/api:v2 production"
        return 1
    fi

    echo "=========================================="
    echo "    Kubernetes 滚动更新"
    echo "=========================================="
    echo "Deployment: $deployment"
    echo "新镜像: $image"
    echo "命名空间: $namespace"
    echo ""

    # 检查 Deployment 是否存在
    if ! kubectl get deployment "$deployment" -n "$namespace" &>/dev/null; then
        echo "❌ Deployment 不存在: $deployment"
        return 1
    fi

    # 显示当前状态
    echo "📊 当前状态："
    kubectl get deployment "$deployment" -n "$namespace" -o wide
    echo ""

    # 获取容器名称（假设第一个容器）
    local container=$(kubectl get deployment "$deployment" -n "$namespace" \
        -o jsonpath='{.spec.template.spec.containers[0].name}')

    echo "容器名称: $container"
    echo ""

    # 确认更新
    read -p "确认开始滚动更新？(y/n): " confirm
    if [ "$confirm" != "y" ]; then
        echo "取消更新"
        return 0
    fi

    # 执行更新
    echo ""
    echo "🔄 开始滚动更新..."
    kubectl set image deployment/"$deployment" "${container}=${image}" -n "$namespace"

    # 监控更新状态
    echo ""
    echo "📊 监控更新状态（按 Ctrl+C 退出监控，更新会继续）..."
    kubectl rollout status deployment/"$deployment" -n "$namespace"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 滚动更新完成"
        echo ""
        echo "📊 最新状态："
        kubectl get deployment "$deployment" -n "$namespace" -o wide
    else
        echo ""
        echo "⚠️  更新可能未完成，检查状态..."
        kubectl get pods -n "$namespace" -l app="$deployment"

        read -p "是否回滚到上一版本？(y/n): " rollback
        if [ "$rollback" = "y" ]; then
            echo "🔙 执行回滚..."
            kubectl rollout undo deployment/"$deployment" -n "$namespace"
        fi
    fi
}


# ------------------------------------------------------------
#              2.4 K8s Pod 调试脚本
# ------------------------------------------------------------
# 调试 Kubernetes Pod

k8s_debug_pod() {
    local namespace=${1:-"default"}

    echo "=========================================="
    echo "    Kubernetes Pod 调试工具"
    echo "=========================================="
    echo "命名空间: $namespace"
    echo ""

    # 列出 Pod
    echo "📋 Pod 列表："
    kubectl get pods -n "$namespace" --show-labels
    echo ""

    read -p "请输入要调试的 Pod 名称: " pod_name

    if [ -z "$pod_name" ]; then
        echo "❌ Pod 名称不能为空"
        return 1
    fi

    # 检查 Pod 是否存在
    if ! kubectl get pod "$pod_name" -n "$namespace" &>/dev/null; then
        echo "❌ Pod 不存在: $pod_name"
        return 1
    fi

    echo ""
    echo "选择调试操作："
    echo "1. 查看 Pod 详细信息"
    echo "2. 查看 Pod 日志"
    echo "3. 进入 Pod 容器"
    echo "4. 查看 Pod 事件"
    echo "5. 查看 Pod 资源使用"
    echo "6. 端口转发"
    echo "7. 复制文件到/从 Pod"
    read -p "请选择 (1-7): " choice

    case $choice in
        1)
            echo ""
            echo "📝 Pod 详细信息："
            kubectl describe pod "$pod_name" -n "$namespace"
            ;;
        2)
            # 获取容器列表
            containers=$(kubectl get pod "$pod_name" -n "$namespace" \
                -o jsonpath='{.spec.containers[*].name}')
            echo ""
            echo "可用容器: $containers"
            read -p "请输入容器名称（直接回车选择第一个）: " container

            if [ -z "$container" ]; then
                container=$(echo $containers | awk '{print $1}')
            fi

            echo ""
            echo "📜 Pod 日志 (容器: $container)："
            kubectl logs "$pod_name" -c "$container" -n "$namespace" --tail=100
            ;;
        3)
            containers=$(kubectl get pod "$pod_name" -n "$namespace" \
                -o jsonpath='{.spec.containers[*].name}')
            echo ""
            echo "可用容器: $containers"
            read -p "请输入容器名称（直接回车选择第一个）: " container

            if [ -z "$container" ]; then
                container=$(echo $containers | awk '{print $1}')
            fi

            echo ""
            echo "🔗 进入容器 (输入 exit 退出)..."
            kubectl exec -it "$pod_name" -c "$container" -n "$namespace" -- /bin/sh
            ;;
        4)
            echo ""
            echo "📋 Pod 相关事件："
            kubectl get events -n "$namespace" --field-selector involvedObject.name="$pod_name"
            ;;
        5)
            echo ""
            echo "📊 Pod 资源使用："
            kubectl top pod "$pod_name" -n "$namespace" --containers 2>/dev/null || \
                echo "   (需要安装 metrics-server)"
            ;;
        6)
            read -p "请输入本地端口: " local_port
            read -p "请输入 Pod 端口: " pod_port
            echo ""
            echo "🔗 端口转发 localhost:$local_port -> pod:$pod_port"
            echo "   按 Ctrl+C 停止"
            kubectl port-forward "$pod_name" "${local_port}:${pod_port}" -n "$namespace"
            ;;
        7)
            echo ""
            echo "选择复制方向："
            echo "1. 从 Pod 复制到本地"
            echo "2. 从本地复制到 Pod"
            read -p "请选择 (1-2): " copy_dir

            if [ "$copy_dir" = "1" ]; then
                read -p "请输入 Pod 中的路径: " pod_path
                read -p "请输入本地保存路径: " local_path
                kubectl cp "${namespace}/${pod_name}:${pod_path}" "$local_path"
                echo "✅ 复制完成"
            else
                read -p "请输入本地文件路径: " local_path
                read -p "请输入 Pod 中的目标路径: " pod_path
                kubectl cp "$local_path" "${namespace}/${pod_name}:${pod_path}"
                echo "✅ 复制完成"
            fi
            ;;
        *)
            echo "❌ 无效选择"
            ;;
    esac
}


# ------------------------------------------------------------
#              2.5 K8s 资源清理脚本
# ------------------------------------------------------------
# 清理 Kubernetes 命名空间中的资源

k8s_cleanup() {
    local namespace=$1

    if [ -z "$namespace" ]; then
        echo "用法: k8s_cleanup <命名空间>"
        return 1
    fi

    echo "=========================================="
    echo "    Kubernetes 资源清理"
    echo "=========================================="
    echo "命名空间: $namespace"
    echo ""

    # 显示将要删除的资源
    echo "📋 将要删除的资源："
    echo ""
    echo "Deployments:"
    kubectl get deployments -n "$namespace" -o name
    echo ""
    echo "Services:"
    kubectl get svc -n "$namespace" -o name
    echo ""
    echo "ConfigMaps:"
    kubectl get configmaps -n "$namespace" -o name
    echo ""
    echo "Secrets:"
    kubectl get secrets -n "$namespace" -o name
    echo ""
    echo "PVCs:"
    kubectl get pvc -n "$namespace" -o name
    echo ""

    # 确认删除
    echo "⚠️  警告: 此操作将删除命名空间 $namespace 中的所有资源！"
    read -p "确认删除？请输入命名空间名称以确认: " confirm

    if [ "$confirm" != "$namespace" ]; then
        echo "取消操作"
        return 0
    fi

    echo ""
    echo "🗑️  开始删除资源..."

    # 删除 Deployments
    kubectl delete deployments --all -n "$namespace"

    # 删除 Services
    kubectl delete svc --all -n "$namespace"

    # 删除 ConfigMaps（排除系统的）
    kubectl delete configmaps --all -n "$namespace"

    # 删除 Secrets（排除系统的）
    kubectl delete secrets --all -n "$namespace"

    # 删除 PVCs
    kubectl delete pvc --all -n "$namespace"

    # 删除 Jobs
    kubectl delete jobs --all -n "$namespace"

    echo ""
    echo "✅ 资源清理完成"
    echo ""
    echo "📊 剩余资源："
    kubectl get all -n "$namespace"
}


# ------------------------------------------------------------
#              2.6 K8s 快速扩缩容脚本
# ------------------------------------------------------------
# 快速扩缩容 Deployment

k8s_scale() {
    local deployment=$1
    local replicas=$2
    local namespace=${3:-"default"}

    if [ -z "$deployment" ] || [ -z "$replicas" ]; then
        echo "用法: k8s_scale <deployment名称> <副本数> [命名空间]"
        echo "示例: k8s_scale api 5 production"
        return 1
    fi

    echo "=========================================="
    echo "    Kubernetes 扩缩容"
    echo "=========================================="
    echo "Deployment: $deployment"
    echo "目标副本数: $replicas"
    echo "命名空间: $namespace"
    echo ""

    # 显示当前状态
    echo "📊 当前状态："
    kubectl get deployment "$deployment" -n "$namespace"
    echo ""

    # 执行扩缩容
    echo "🔄 执行扩缩容..."
    kubectl scale deployment "$deployment" --replicas="$replicas" -n "$namespace"

    # 等待完成
    echo ""
    echo "⏳ 等待扩缩容完成..."
    kubectl rollout status deployment/"$deployment" -n "$namespace" --timeout=120s

    echo ""
    echo "📊 最新状态："
    kubectl get deployment "$deployment" -n "$namespace"
    echo ""
    kubectl get pods -n "$namespace" -l app="$deployment"
}


# ============================================================
#     第三部分：综合运维脚本
# ============================================================


# ------------------------------------------------------------
#              3.1 健康检查脚本
# ------------------------------------------------------------
# 检查整个应用栈的健康状态

health_check() {
    local namespace=${1:-"default"}

    echo "=========================================="
    echo "    应用健康检查"
    echo "=========================================="
    echo "命名空间: $namespace"
    echo "检查时间: $(date)"
    echo ""

    local has_error=0

    # 检查 Pod 状态
    echo "🔍 检查 Pod 状态..."
    not_running=$(kubectl get pods -n "$namespace" --no-headers | grep -v Running | grep -v Completed)
    if [ -n "$not_running" ]; then
        echo "❌ 发现异常 Pod："
        echo "$not_running"
        has_error=1
    else
        echo "✅ 所有 Pod 运行正常"
    fi
    echo ""

    # 检查 Pod 重启次数
    echo "🔍 检查 Pod 重启情况..."
    high_restarts=$(kubectl get pods -n "$namespace" -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.containerStatuses[0].restartCount}{"\n"}{end}' | awk '$2 > 5 {print}')
    if [ -n "$high_restarts" ]; then
        echo "⚠️  发现高重启次数的 Pod："
        echo "$high_restarts"
        has_error=1
    else
        echo "✅ 没有频繁重启的 Pod"
    fi
    echo ""

    # 检查 Service 端点
    echo "🔍 检查 Service 端点..."
    for svc in $(kubectl get svc -n "$namespace" -o name); do
        endpoints=$(kubectl get endpoints -n "$namespace" $(echo $svc | cut -d'/' -f2) -o jsonpath='{.subsets[*].addresses[*].ip}')
        if [ -z "$endpoints" ]; then
            echo "⚠️  Service $svc 没有可用端点"
            has_error=1
        fi
    done
    if [ $has_error -eq 0 ]; then
        echo "✅ 所有 Service 端点正常"
    fi
    echo ""

    # 检查 PVC 状态
    echo "🔍 检查 PVC 状态..."
    pending_pvc=$(kubectl get pvc -n "$namespace" --no-headers | grep -v Bound)
    if [ -n "$pending_pvc" ]; then
        echo "⚠️  发现未绑定的 PVC："
        echo "$pending_pvc"
        has_error=1
    else
        echo "✅ 所有 PVC 已绑定"
    fi
    echo ""

    # 检查最近错误事件
    echo "🔍 检查最近错误事件..."
    error_events=$(kubectl get events -n "$namespace" --field-selector type=Warning --sort-by='.lastTimestamp' | tail -5)
    if [ -n "$error_events" ]; then
        echo "⚠️  最近的警告事件："
        echo "$error_events"
    else
        echo "✅ 没有警告事件"
    fi
    echo ""

    # 总结
    echo "=========================================="
    if [ $has_error -eq 0 ]; then
        echo "✅ 健康检查通过"
    else
        echo "⚠️  健康检查发现问题，请处理"
    fi
    echo "=========================================="

    return $has_error
}


# ------------------------------------------------------------
#              3.2 备份脚本
# ------------------------------------------------------------
# 备份 Kubernetes 配置和数据

k8s_backup() {
    local backup_dir=${1:-"./k8s-backup-$(date +%Y%m%d_%H%M%S)"}

    echo "=========================================="
    echo "    Kubernetes 配置备份"
    echo "=========================================="
    echo "备份目录: $backup_dir"
    echo ""

    mkdir -p "$backup_dir"

    # 备份所有命名空间的资源
    echo "📦 开始备份..."

    for ns in $(kubectl get namespaces -o jsonpath='{.items[*].metadata.name}'); do
        # 跳过系统命名空间
        if [[ "$ns" == "kube-system" || "$ns" == "kube-public" || "$ns" == "kube-node-lease" ]]; then
            continue
        fi

        echo "  备份命名空间: $ns"
        mkdir -p "$backup_dir/$ns"

        # 备份各类资源
        for resource in deployments services configmaps secrets ingresses pvc; do
            kubectl get $resource -n "$ns" -o yaml > "$backup_dir/$ns/${resource}.yaml" 2>/dev/null
        done
    done

    # 备份集群级别资源
    echo "  备份集群级别资源..."
    mkdir -p "$backup_dir/cluster"
    kubectl get clusterroles -o yaml > "$backup_dir/cluster/clusterroles.yaml" 2>/dev/null
    kubectl get clusterrolebindings -o yaml > "$backup_dir/cluster/clusterrolebindings.yaml" 2>/dev/null
    kubectl get storageclasses -o yaml > "$backup_dir/cluster/storageclasses.yaml" 2>/dev/null

    # 压缩备份
    echo ""
    echo "📦 压缩备份文件..."
    tar -czf "${backup_dir}.tar.gz" "$backup_dir"
    rm -rf "$backup_dir"

    echo ""
    echo "✅ 备份完成: ${backup_dir}.tar.gz"
    echo "   文件大小: $(ls -lh ${backup_dir}.tar.gz | awk '{print $5}')"
}


# ============================================================
#     第四部分：使用说明
# ============================================================

show_help() {
    echo "=========================================="
    echo "    Docker & K8s 运维脚本帮助"
    echo "=========================================="
    echo ""
    echo "Docker 相关函数："
    echo "  docker_cleanup              - 清理 Docker 环境"
    echo "  build_and_push_image        - 构建并推送镜像"
    echo "  view_container_logs         - 查看容器日志"
    echo "  compose_manager             - Docker Compose 管理"
    echo ""
    echo "Kubernetes 相关函数："
    echo "  k8s_status [namespace]      - 查看集群状态"
    echo "  k8s_deploy <file> [ns]      - 部署应用"
    echo "  k8s_rolling_update          - 滚动更新"
    echo "  k8s_debug_pod [namespace]   - 调试 Pod"
    echo "  k8s_cleanup <namespace>     - 清理资源"
    echo "  k8s_scale <deploy> <n> [ns] - 扩缩容"
    echo ""
    echo "综合运维函数："
    echo "  health_check [namespace]    - 健康检查"
    echo "  k8s_backup [backup_dir]     - 备份配置"
    echo ""
    echo "使用方法："
    echo "  1. source 06_practical_scripts.sh"
    echo "  2. 调用上述函数，如: k8s_status default"
    echo ""
}

# 如果直接运行脚本，显示帮助
if [ "$0" = "$BASH_SOURCE" ]; then
    show_help
fi
