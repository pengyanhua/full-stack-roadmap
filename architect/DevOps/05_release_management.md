# 版本管理与发布策略

## 目录
- [版本管理概述](#版本管理概述)
- [语义化版本](#语义化版本)
- [分支策略](#分支策略)
- [发布流程](#发布流程)
- [回滚策略](#回滚策略)
- [变更日志](#变更日志)
- [实战案例](#实战案例)

---

## 版本管理概述

### 版本号规范

```
┌──────────────────────────────────────────────────────┐
│              语义化版本 (SemVer)                     │
├──────────────────────────────────────────────────────┤
│                                                      │
│         MAJOR . MINOR . PATCH - PRERELEASE + BUILD  │
│           ↓       ↓       ↓         ↓          ↓    │
│           1   .   2   .   3    -  alpha.1  + 001    │
│                                                      │
│  MAJOR: 不兼容的API变更                              │
│  MINOR: 向后兼容的功能新增                           │
│  PATCH: 向后兼容的问题修复                           │
│  PRERELEASE: 预发布版本                              │
│  BUILD: 构建元数据                                   │
│                                                      │
│  示例:                                               │
│    1.0.0         - 首个正式版本                      │
│    1.1.0         - 新增功能                          │
│    1.1.1         - 修复Bug                           │
│    2.0.0         - 破坏性变更                        │
│    2.0.0-beta.1  - Beta测试版                        │
│    2.0.0-rc.1    - Release Candidate                │
└──────────────────────────────────────────────────────┘
```

### 版本生命周期

```
┌────────────────────────────────────────────────┐
│           版本发布生命周期                     │
├────────────────────────────────────────────────┤
│                                                │
│  Alpha (α)                                     │
│   ├─ 内部测试版本                              │
│   ├─ 功能不完整                                │
│   └─ 可能有严重Bug                             │
│       │                                        │
│       ▼                                        │
│  Beta (β)                                      │
│   ├─ 功能基本完整                              │
│   ├─ 公开测试                                  │
│   └─ 可能有已知问题                            │
│       │                                        │
│       ▼                                        │
│  RC (Release Candidate)                        │
│   ├─ 功能冻结                                  │
│   ├─ 仅修复严重Bug                             │
│   └─ 最终测试                                  │
│       │                                        │
│       ▼                                        │
│  GA (General Availability)                     │
│   ├─ 正式发布                                  │
│   ├─ 生产可用                                  │
│   └─ 完整文档                                  │
│       │                                        │
│       ▼                                        │
│  LTS (Long Term Support)                       │
│   ├─ 长期支持                                  │
│   ├─ 仅安全更新                                │
│   └─ 稳定维护                                  │
│       │                                        │
│       ▼                                        │
│  EOL (End of Life)                             │
│   └─ 停止支持                                  │
└────────────────────────────────────────────────┘
```

---

## 语义化版本

### 自动版本管理

```bash
# version-bump.sh
#!/bin/bash

set -euo pipefail

# 当前版本
CURRENT_VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
CURRENT_VERSION=${CURRENT_VERSION#v}

IFS='.' read -r -a VERSION_PARTS <<< "$CURRENT_VERSION"
MAJOR=${VERSION_PARTS[0]}
MINOR=${VERSION_PARTS[1]}
PATCH=${VERSION_PARTS[2]}

# 根据提交信息确定版本类型
BUMP_TYPE=${1:-auto}

if [ "$BUMP_TYPE" = "auto" ]; then
  # 分析 git log
  LOGS=$(git log --pretty=format:"%s" "${CURRENT_VERSION}..HEAD")

  if echo "$LOGS" | grep -qE "^BREAKING CHANGE:|^feat!:|^fix!:"; then
    BUMP_TYPE="major"
  elif echo "$LOGS" | grep -qE "^feat:"; then
    BUMP_TYPE="minor"
  else
    BUMP_TYPE="patch"
  fi
fi

# 计算新版本
case $BUMP_TYPE in
  major)
    MAJOR=$((MAJOR + 1))
    MINOR=0
    PATCH=0
    ;;
  minor)
    MINOR=$((MINOR + 1))
    PATCH=0
    ;;
  patch)
    PATCH=$((PATCH + 1))
    ;;
  *)
    echo "Unknown bump type: $BUMP_TYPE"
    exit 1
    ;;
esac

NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}"
echo "📦 Bumping version from ${CURRENT_VERSION} to ${NEW_VERSION}"

# 更新版本文件
echo "${NEW_VERSION}" > VERSION
git add VERSION

# 创建 Git tag
git tag -a "v${NEW_VERSION}" -m "Release v${NEW_VERSION}"

echo "✅ Version bumped to v${NEW_VERSION}"
```

### Conventional Commits

```bash
# 提交信息规范
# <type>(<scope>): <subject>
#
# type: feat, fix, docs, style, refactor, test, chore, perf
# scope: 影响范围
# subject: 简短描述

# 示例
git commit -m "feat(api): add user authentication endpoint"
git commit -m "fix(database): resolve connection pool leak"
git commit -m "docs: update API documentation"
git commit -m "chore(deps): upgrade dependencies"

# 破坏性变更
git commit -m "feat(api)!: change authentication flow

BREAKING CHANGE: OAuth2 is now required for all endpoints"
```

---

## 分支策略

### Git Flow

```
┌────────────────────────────────────────────────┐
│              Git Flow 分支模型                 │
├────────────────────────────────────────────────┤
│                                                │
│  main (生产分支)                               │
│   │                                            │
│   ├──────────────────────────▶ v1.0.0         │
│   │                           │                │
│   │                           │                │
│  develop (开发分支)           │                │
│   │                           │                │
│   ├─ feature/login ──┐        │                │
│   │                  │        │                │
│   ├─ feature/payment─┤        │                │
│   │                  │        │                │
│   ◀──────────────────┘        │                │
│   │                           │                │
│   ├─ release/1.0 ────────────▶│                │
│   │      │                    │                │
│   ◀──────┤ (bug fixes)        │                │
│          │                    │                │
│          └───────────────────▶│                │
│                                │                │
│  hotfix/critical-bug ─────────┤                │
│   │                           │                │
│   └──────────────────────────▶│                │
└────────────────────────────────────────────────┘
```

### GitHub Flow (简化版)

```
┌────────────────────────────────────────────────┐
│            GitHub Flow 分支模型                │
├────────────────────────────────────────────────┤
│                                                │
│  main (主分支)                                 │
│   │                                            │
│   ├─ feature-1 ─┐                             │
│   │              │                             │
│   │              ├─ PR ─▶ Merge               │
│   │              │                             │
│   ◀──────────────┘                             │
│   │                                            │
│   ├─ feature-2 ─┐                             │
│   │              │                             │
│   │              ├─ PR ─▶ Merge               │
│   │              │                             │
│   ◀──────────────┘                             │
│   │                                            │
│   └─ 每次合并自动部署                          │
└────────────────────────────────────────────────┘
```

### Trunk-Based Development

```
┌────────────────────────────────────────────────┐
│         Trunk-Based Development                │
├────────────────────────────────────────────────┤
│                                                │
│  main (主干)                                   │
│   │                                            │
│   ├─ 短期分支 (1-2天) ─┐                      │
│   │                     │                      │
│   ◀────────────────────┘                       │
│   │                                            │
│   ├─ 短期分支 ─┐                              │
│   │            │                               │
│   ◀───────────┘                                │
│   │                                            │
│   └─ 特性开关控制功能发布                      │
└────────────────────────────────────────────────┘
```

---

## 发布流程

### 完整发布 Workflow

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    branches:
      - main
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to release (e.g., 1.2.3)'
        required: true

jobs:
  prepare:
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.version.outputs.version }}
      changelog: ${{ steps.changelog.outputs.changelog }}

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Determine version
        id: version
        run: |
          if [ -n "${{ github.event.inputs.version }}" ]; then
            VERSION="${{ github.event.inputs.version }}"
          else
            # 自动计算版本
            CURRENT=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
            CURRENT=${CURRENT#v}

            # 基于提交判断版本类型
            if git log ${CURRENT}..HEAD --pretty=format:"%s" | grep -qE "^BREAKING CHANGE:|^feat!:|^fix!:"; then
              BUMP="major"
            elif git log ${CURRENT}..HEAD --pretty=format:"%s" | grep -qE "^feat:"; then
              BUMP="minor"
            else
              BUMP="patch"
            fi

            # 计算新版本
            IFS='.' read -r -a parts <<< "$CURRENT"
            case $BUMP in
              major) VERSION="$((parts[0]+1)).0.0" ;;
              minor) VERSION="${parts[0]}.$((parts[1]+1)).0" ;;
              patch) VERSION="${parts[0]}.${parts[1]}.$((parts[2]+1))" ;;
            esac
          fi

          echo "version=${VERSION}" >> $GITHUB_OUTPUT
          echo "📦 Version: ${VERSION}"

      - name: Generate changelog
        id: changelog
        run: |
          CURRENT=$(git describe --tags --abbrev=0 2>/dev/null || echo "")

          if [ -n "$CURRENT" ]; then
            CHANGELOG=$(git log ${CURRENT}..HEAD --pretty=format:"- %s (%h)" --no-merges)
          else
            CHANGELOG=$(git log --pretty=format:"- %s (%h)" --no-merges)
          fi

          # 分类提交
          FEATURES=$(echo "$CHANGELOG" | grep "^- feat:" || echo "")
          FIXES=$(echo "$CHANGELOG" | grep "^- fix:" || echo "")
          OTHERS=$(echo "$CHANGELOG" | grep -v "^- feat:" | grep -v "^- fix:" || echo "")

          FORMATTED_CHANGELOG="## What's Changed

### Features
$FEATURES

### Bug Fixes
$FIXES

### Other Changes
$OTHERS
"
          echo "changelog<<EOF" >> $GITHUB_OUTPUT
          echo "$FORMATTED_CHANGELOG" >> $GITHUB_OUTPUT
          echo "EOF" >> $GITHUB_OUTPUT

  build:
    needs: prepare
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:${{ needs.prepare.outputs.version }}
            ghcr.io/${{ github.repository }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  test:
    needs: build
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Run integration tests
        run: |
          docker-compose -f docker-compose.test.yml up -d
          docker-compose -f docker-compose.test.yml run tests
          docker-compose -f docker-compose.test.yml down

  deploy-staging:
    needs: test
    runs-on: ubuntu-latest
    environment:
      name: staging
      url: https://staging.example.com

    steps:
      - name: Deploy to staging
        run: |
          kubectl set image deployment/myapp \
            myapp=ghcr.io/${{ github.repository }}:${{ needs.prepare.outputs.version }} \
            -n staging

  release:
    needs: [prepare, deploy-staging]
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Create Git tag
        run: |
          git config user.name github-actions
          git config user.email github-actions@github.com
          git tag -a v${{ needs.prepare.outputs.version }} \
            -m "Release v${{ needs.prepare.outputs.version }}"
          git push origin v${{ needs.prepare.outputs.version }}

      - name: Create GitHub Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: v${{ needs.prepare.outputs.version }}
          release_name: Release v${{ needs.prepare.outputs.version }}
          body: ${{ needs.prepare.outputs.changelog }}
          draft: false
          prerelease: false

  deploy-production:
    needs: release
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://example.com

    steps:
      - name: Deploy to production
        run: |
          kubectl set image deployment/myapp \
            myapp=ghcr.io/${{ github.repository }}:${{ needs.prepare.outputs.version }} \
            -n production
```

---

## 回滚策略

### 快速回滚脚本

```bash
#!/bin/bash
# rollback.sh

set -euo pipefail

NAMESPACE=${1:-production}
DEPLOYMENT=${2:-myapp}

echo "🔙 Starting rollback for ${DEPLOYMENT} in ${NAMESPACE}"

# 1. 查看历史版本
echo "📜 Deployment history:"
kubectl rollout history deployment/${DEPLOYMENT} -n ${NAMESPACE}

# 2. 获取当前版本
CURRENT_REVISION=$(kubectl get deployment ${DEPLOYMENT} -n ${NAMESPACE} \
  -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}')

echo "Current revision: ${CURRENT_REVISION}"

# 3. 确认回滚
read -p "Rollback to previous revision? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
  echo "❌ Rollback cancelled"
  exit 0
fi

# 4. 执行回滚
echo "⏪ Rolling back..."
kubectl rollout undo deployment/${DEPLOYMENT} -n ${NAMESPACE}

# 5. 等待回滚完成
echo "⏳ Waiting for rollback to complete..."
kubectl rollout status deployment/${DEPLOYMENT} -n ${NAMESPACE} --timeout=300s

# 6. 验证回滚
NEW_REVISION=$(kubectl get deployment ${DEPLOYMENT} -n ${NAMESPACE} \
  -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}')

READY_REPLICAS=$(kubectl get deployment ${DEPLOYMENT} -n ${NAMESPACE} \
  -o jsonpath='{.status.readyReplicas}')

echo "✅ Rollback completed"
echo "   Previous revision: ${CURRENT_REVISION}"
echo "   Current revision: ${NEW_REVISION}"
echo "   Ready replicas: ${READY_REPLICAS}"

# 7. 发送通知
curl -X POST 'https://hooks.slack.com/services/xxx' \
  -H 'Content-Type: application/json' \
  -d "{
    \"text\": \"🔙 Rollback completed for ${DEPLOYMENT} in ${NAMESPACE}\\nRevision: ${CURRENT_REVISION} → ${NEW_REVISION}\"
  }"
```

### 数据库回滚策略

```sql
-- migration-rollback.sql
-- 使用事务确保原子性

BEGIN;

-- 1. 备份当前数据
CREATE TABLE users_backup_20260207 AS
SELECT * FROM users;

-- 2. 执行回滚
ALTER TABLE users DROP COLUMN new_feature_column;

-- 3. 验证数据完整性
DO $$
DECLARE
  user_count INTEGER;
BEGIN
  SELECT COUNT(*) INTO user_count FROM users;

  IF user_count < 1000 THEN
    RAISE EXCEPTION 'Data integrity check failed: too few users';
  END IF;
END $$;

-- 4. 提交事务
COMMIT;

-- 如果失败会自动回滚
```

---

## 变更日志

### 自动生成 CHANGELOG

```bash
#!/bin/bash
# generate-changelog.sh

set -euo pipefail

CURRENT_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
OUTPUT_FILE="CHANGELOG.md"

# 生成 CHANGELOG
{
  echo "# Changelog"
  echo ""
  echo "All notable changes to this project will be documented in this file."
  echo ""

  # 遍历所有标签
  TAGS=$(git tag --sort=-version:refname)

  PREV_TAG=""
  for TAG in $TAGS; do
    echo "## [${TAG}] - $(git log -1 --format=%ai ${TAG} | cut -d' ' -f1)"
    echo ""

    # 确定范围
    if [ -z "$PREV_TAG" ]; then
      RANGE="${TAG}"
    else
      RANGE="${TAG}..${PREV_TAG}"
    fi

    # 分类提交
    echo "### Features"
    git log ${RANGE} --pretty=format:"- %s ([%h](https://github.com/myorg/myrepo/commit/%H))" \
      --grep="^feat" --no-merges || echo ""
    echo ""

    echo "### Bug Fixes"
    git log ${RANGE} --pretty=format:"- %s ([%h](https://github.com/myorg/myrepo/commit/%H))" \
      --grep="^fix" --no-merges || echo ""
    echo ""

    echo "### Documentation"
    git log ${RANGE} --pretty=format:"- %s ([%h](https://github.com/myorg/myrepo/commit/%H))" \
      --grep="^docs" --no-merges || echo ""
    echo ""

    PREV_TAG=$TAG
  done
} > ${OUTPUT_FILE}

echo "✅ Changelog generated: ${OUTPUT_FILE}"
```

### CHANGELOG 模板

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New feature X
- Support for Y

### Changed
- Improved performance of Z

### Deprecated
- Feature A will be removed in v2.0.0

### Removed
- Unused configuration option B

### Fixed
- Bug in component C
- Security vulnerability in dependency D

### Security
- Fixed CVE-2024-XXXXX

## [1.2.0] - 2026-02-07

### Added
- User authentication with OAuth2
- Rate limiting for API endpoints

### Fixed
- Memory leak in background worker
- Race condition in cache invalidation

## [1.1.0] - 2026-01-15

### Added
- Email notification system
- Export to CSV feature

### Changed
- Updated UI design
- Improved database query performance

## [1.0.0] - 2026-01-01

### Added
- Initial release
- Core features implemented
```

---

## 实战案例

### 案例: 大规模服务发布流程

```yaml
# release-workflow.yaml
# 完整的生产发布流程

stages:
  # 1. 准备阶段
  prepare:
    - version_bump
    - changelog_generation
    - dependency_audit
    - security_scan

  # 2. 构建阶段
  build:
    - compile_code
    - run_unit_tests
    - build_containers
    - push_to_registry

  # 3. 测试阶段
  test:
    - integration_tests
    - e2e_tests
    - performance_tests
    - security_tests

  # 4. 预发布
  pre_release:
    - deploy_to_staging
    - smoke_tests
    - manual_verification

  # 5. 发布
  release:
    - create_release_tag
    - generate_release_notes
    - publish_artifacts

  # 6. 部署
  deploy:
    - canary_deployment:
        percentage: 10
        duration: 30m
    - expand_deployment:
        percentage: 50
        duration: 1h
    - full_deployment:
        percentage: 100

  # 7. 验证
  verify:
    - health_checks
    - metric_validation
    - error_rate_check

  # 8. 清理
  cleanup:
    - remove_old_versions
    - cleanup_artifacts
    - update_documentation
```

---

## 总结

### 版本管理最佳实践

```
┌────────────────────────────────────────────────┐
│         版本管理十大最佳实践                   │
├────────────────────────────────────────────────┤
│                                                │
│ 1. 使用语义化版本 (SemVer)                    │
│ 2. 自动化版本号生成                            │
│ 3. 保持详细的变更日志                          │
│ 4. 使用 Git 标签标记版本                       │
│ 5. 实施代码审查流程                            │
│ 6. 自动化测试覆盖                              │
│ 7. 分阶段发布(金丝雀/蓝绿)                     │
│ 8. 准备回滚预案                                │
│ 9. 监控发布指标                                │
│ 10. 文档化发布流程                             │
└────────────────────────────────────────────────┘
```

### 发布检查清单

- [ ] 代码审查通过
- [ ] 所有测试通过
- [ ] 安全扫描无高危漏洞
- [ ] 性能测试达标
- [ ] 文档已更新
- [ ] 变更日志已生成
- [ ] 回滚方案已准备
- [ ] 监控告警已配置
- [ ] 相关团队已通知
- [ ] 发布时间窗口已确认

### 下一步学习
- [../API_Gateway/01_gateway_design.md](../API_Gateway/01_gateway_design.md) - API网关设计
