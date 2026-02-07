# 发布管理与回滚

## 目录
- [版本管理](#版本管理)
- [分支策略](#分支策略)
- [回滚策略](#回滚策略)
- [发布检查清单](#发布检查清单)

---

## 版本管理

### 语义化版本 (SemVer)

```
┌────────────────────────────────────────────────────┐
│          语义化版本规范                            │
├────────────────────────────────────────────────────┤
│                                                    │
│         MAJOR . MINOR . PATCH                      │
│           │       │       │                        │
│           │       │       └─ 补丁版本(bug修复)     │
│           │       └───────── 次版本(新功能,向后兼容)│
│           └───────────────── 主版本(破坏性变更)    │
│                                                    │
│  示例:                                             │
│  1.0.0 → 初始发布                                  │
│  1.1.0 → 添加新功能                                │
│  1.1.1 → 修复bug                                   │
│  2.0.0 → 不兼容的API变更                           │
│                                                    │
│  预发布: 1.0.0-alpha.1, 1.0.0-beta.2             │
│  构建元数据: 1.0.0+20230115.git123abc             │
└────────────────────────────────────────────────────┘
```

### 自动版本号

```bash
# bump_version.sh
#!/bin/bash

# 获取当前版本
current_version=$(git describe --tags --abbrev=0)

# 解析版本号
IFS='.' read -ra version_parts <<< "${current_version#v}"
major="${version_parts[0]}"
minor="${version_parts[1]}"
patch="${version_parts[2]}"

# 根据提交信息决定版本号
if git log $current_version..HEAD | grep -q "BREAKING CHANGE"; then
    major=$((major + 1))
    minor=0
    patch=0
elif git log $current_version..HEAD | grep -q "feat:"; then
    minor=$((minor + 1))
    patch=0
else
    patch=$((patch + 1))
fi

new_version="v${major}.${minor}.${patch}"

# 创建标签
git tag -a $new_version -m "Release $new_version"
git push origin $new_version

echo "✅ 版本更新: $current_version → $new_version"
```

## 分支策略

### Git Flow

```
┌────────────────────────────────────────────────────┐
│              Git Flow 分支模型                     │
├────────────────────────────────────────────────────┤
│                                                    │
│  main (生产)    ●────●─────────●─────────●        │
│                  ╲    ╲         ╲         ╲       │
│                   ╲    ╲         ╲         ╲      │
│  release          ●───●──●       ●───●──●        │
│                    │   │   ╲      │   │   ╲      │
│                    │   │    ╲     │   │    ╲     │
│  develop    ●──●──●───●─────●────●───●─────●    │
│              ╲  ╲                                  │
│               ╲  ╲                                 │
│  feature      ●──●                                │
│                                                    │
│  分支说明:                                         │
│  • main: 生产代码,只接受merge                     │
│  • develop: 开发主线                               │
│  • feature/*: 功能分支                             │
│  • release/*: 发布分支                             │
│  • hotfix/*: 紧急修复                              │
└────────────────────────────────────────────────────┘
```

### GitHub Flow (简化)

```
┌────────────────────────────────────────────────────┐
│              GitHub Flow                           │
├────────────────────────────────────────────────────┤
│                                                    │
│  main    ●──────●────●────●────●─────●           │
│           ╲      ╲    ╲    ╲    ╲     ╲          │
│            ╲      ╲    ╲    ╲    ╲     ╲         │
│  feature   ●─────●    ●    ●    ●─────●         │
│                                                    │
│  流程:                                             │
│  1. 从main创建feature分支                         │
│  2. 开发 + 提交                                    │
│  3. 创建Pull Request                               │
│  4. Code Review                                    │
│  5. CI/CD检查                                      │
│  6. 合并到main                                     │
│  7. 自动部署                                       │
└────────────────────────────────────────────────────┘
```

## 回滚策略

### Kubernetes回滚

```bash
# 查看发布历史
kubectl rollout history deployment/myapp

# 查看特定版本详情
kubectl rollout history deployment/myapp --revision=2

# 回滚到上一个版本
kubectl rollout undo deployment/myapp

# 回滚到指定版本
kubectl rollout undo deployment/myapp --to-revision=2

# 监控回滚状态
kubectl rollout status deployment/myapp
```

### 数据库回滚

```python
# db_migrations.py - Alembic示例
"""
数据库迁移最佳实践:
1. 使用事务
2. 可逆操作
3. 分阶段迁移
4. 备份数据
"""

from alembic import op
import sqlalchemy as sa

def upgrade():
    """升级"""
    # 第1步: 添加新列(允许NULL)
    op.add_column('users', sa.Column('email', sa.String(255), nullable=True))

    # 第2步: 填充数据
    op.execute("UPDATE users SET email = CONCAT(username, '@example.com') WHERE email IS NULL")

    # 第3步: 设置NOT NULL约束(下次发布)
    # op.alter_column('users', 'email', nullable=False)

def downgrade():
    """回滚"""
    op.drop_column('users', 'email')
```

### 自动回滚

```yaml
# rollback-on-error.yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: myapp
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: myapp
  service:
    port: 80
  analysis:
    interval: 1m
    threshold: 5
    maxWeight: 50
    stepWeight: 10
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99  # 成功率<99%自动回滚
      interval: 1m
    - name: request-duration
      thresholdRange:
        max: 500  # 延迟>500ms自动回滚
      interval: 1m
    webhooks:
    - name: rollback-notification
      url: http://slack-webhook/alert
      timeout: 5s
```

## 发布检查清单

```
✅ 发布前检查 (Pre-release)
  ☐ 代码已合并到release分支
  ☐ 所有CI检查通过
  ☐ 代码审查完成
  ☐ 安全扫描通过
  ☐ 性能测试通过
  ☐ 数据库迁移脚本准备
  ☐ 回滚方案准备
  ☐ 监控告警配置更新

✅ 发布中检查 (During Release)
  ☐ 通知相关团队
  ☐ 启用只读模式(如需要)
  ☐ 执行数据库迁移
  ☐ 部署新版本
  ☐ 烟雾测试通过
  ☐ 健康检查通过

✅ 发布后检查 (Post-release)
  ☐ 监控关键指标
  ☐ 错误日志检查
  ☐ 用户反馈收集
  ☐ 性能指标正常
  ☐ 发布公告
  ☐ 更新文档
  ☐ 清理旧版本资源
```

### 发布自动化脚本

```bash
#!/bin/bash
# release.sh - 完整发布流程

set -e

VERSION=$1
ENV=${2:-production}

echo "🚀 开始发布 $VERSION 到 $ENV"

# 1. 预检查
echo "1️⃣ 运行预检查..."
./scripts/pre-release-check.sh || exit 1

# 2. 数据库迁移
echo "2️⃣ 执行数据库迁移..."
kubectl exec -it db-pod -- /app/migrate.sh upgrade

# 3. 部署新版本
echo "3️⃣ 部署新版本..."
kubectl set image deployment/myapp myapp=myapp:$VERSION

# 4. 等待就绪
echo "4️⃣ 等待Pod就绪..."
kubectl rollout status deployment/myapp --timeout=5m

# 5. 烟雾测试
echo "5️⃣ 运行烟雾测试..."
./scripts/smoke-test.sh $ENV || {
    echo "❌ 烟雾测试失败,执行回滚..."
    kubectl rollout undo deployment/myapp
    exit 1
}

# 6. 监控
echo "6️⃣ 监控5分钟..."
sleep 300

# 7. 检查指标
ERROR_RATE=$(curl -s "http://prometheus/api/v1/query?query=error_rate" | jq '.data.result[0].value[1]')
if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
    echo "❌ 错误率过高,执行回滚..."
    kubectl rollout undo deployment/myapp
    exit 1
fi

echo "✅ 发布成功！版本: $VERSION"

# 8. 通知
curl -X POST https://slack-webhook \
    -d "{\"text\": \"✅ $ENV 发布成功: $VERSION\"}"
```

## 总结

发布管理关键原则：
1. 自动化一切
2. 小批量频繁发布
3. 快速回滚能力
4. 全面监控
5. 渐进式发布
