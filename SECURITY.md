# 安全政策 / Security Policy

## 🔒 项目安全状态

本项目是**教育性质的学习资源**，不包含生产环境的敏感信息。

### ✅ 已验证安全的内容

1. **Giscus 配置**
   - `repo-id` 和 `category-id` 是公开的仓库标识符
   - 这些 ID 本身不是密钥，公开是安全的
   - 用户需要通过 GitHub 登录才能评论

2. **GitHub Actions**
   - 使用 GitHub Secrets 管理 API tokens
   - 不包含硬编码的密钥

3. **示例代码**
   - 教程中的 password、token 等都是示例
   - 不是真实的凭证

## 🚨 报告安全问题

如果你发现安全漏洞，请：

1. **不要**在公开 Issue 中报告
2. 发送邮件至仓库维护者（见 GitHub profile）
3. 或通过 GitHub Security Advisories 报告

## 🛡️ 安全最佳实践

### 对于贡献者

如果你要贡献代码，请确保：

- ✅ 不要提交 `.env` 文件
- ✅ 不要提交任何真实的 API keys 或密码
- ✅ 示例代码中使用占位符（如 `YOUR_API_KEY`）
- ✅ 检查提交历史，确保没有意外包含敏感信息

### 文件类型检查清单

以下文件类型**永远不应该**提交到仓库：

```
❌ .env, .env.local, .env.production
❌ *.pem, *.key, *.cert, *.crt
❌ credentials.json, serviceAccount.json
❌ config/secrets.yml
❌ *.db, *.sqlite (包含真实数据的数据库)
```

### Git 历史清理

如果意外提交了敏感信息：

1. **立即撤销**
   ```bash
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch PATH/TO/FILE" \
     --prune-empty --tag-name-filter cat -- --all
   ```

2. **强制推送**（⚠️ 慎重）
   ```bash
   git push origin --force --all
   ```

3. **更新所有凭证**
   - 立即废除泄露的密钥
   - 生成新的凭证
   - 更新相关服务

## 📋 安全检查清单

在推送代码前，请检查：

- [ ] 没有包含 `.env` 文件
- [ ] 没有包含真实的 API keys
- [ ] 没有包含密码或私钥
- [ ] 已更新 `.gitignore` 排除敏感文件
- [ ] 示例代码使用占位符

## 🔍 自动安全扫描

### 本地检查

运行以下命令检查敏感信息：

```bash
# 检查是否包含可能的密钥
git secrets --scan-history

# 使用 gitleaks 扫描
gitleaks detect --source . --verbose

# 简单的 grep 检查
grep -r "password\|api_key\|secret" --include="*.js" --include="*.ts" .
```

### GitHub 安全功能

项目启用了以下安全功能：

- ✅ Dependabot 安全更新
- ✅ Code scanning (如果适用)
- ✅ Secret scanning

## 📚 相关资源

- [GitHub 密钥扫描](https://docs.github.com/en/code-security/secret-scanning)
- [移除敏感数据](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [Git Secrets](https://github.com/awslabs/git-secrets)

## 📞 联系方式

安全相关问题请通过以下方式联系：

- GitHub Security Advisories
- 仓库 Issues（非敏感问题）

---

**最后更新**: 2026-02-07
