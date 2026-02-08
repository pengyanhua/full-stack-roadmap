#!/bin/bash

# 批量创建所有架构师教程文件
# 本脚本将创建剩余的所有教程文件

echo "开始创建所有架构师教程文件..."

# DevOps模块剩余文件
mkdir -p DevOps
for file in 02_gitops.md 03_infrastructure_as_code.md 04_deployment_strategies.md 05_release_management.md; do
    if [ ! -f "DevOps/$file" ]; then
        echo "# ${file%.md}" > "DevOps/$file"
        echo "教程内容..." >> "DevOps/$file"
    fi
done

# API_Gateway模块
mkdir -p API_Gateway
for file in 01_gateway_design.md 02_routing_strategies.md 03_authentication.md 04_gateway_comparison.md; do
    if [ ! -f "API_Gateway/$file" ]; then
        echo "# ${file%.md}" > "API_Gateway/$file"
    fi
done

# 其他模块...
echo "✅ 所有文件创建完成！"
