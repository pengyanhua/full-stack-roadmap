---
name: publish
description: 提交代码、构建并部署站点到 Cloudflare Pages。当用户说"发布"、"部署"、"上线"、"publish"、"deploy"时使用。
argument-hint: "commit|build|deploy|all"
disable-model-invocation: true
allowed-tools: Bash, Read, Glob, Grep
---

# 发布部署

提交代码推送到 GitHub，构建 VitePress 站点，并手动部署到 Cloudflare Pages。

## 参数

- `$ARGUMENTS` — 可选，指定操作：
  - `commit` — 仅提交推送代码
  - `build` — 仅构建 VitePress
  - `deploy` — 仅部署到 Cloudflare（需先构建）
  - `all`（默认） — 提交 + 构建 + 部署
  - 空 — 等同于 `all`

## 执行流程

### 1. commit — 提交推送代码

```bash
git add -A
git commit -m "描述本次改动"
git push
```

提交前先用 `git status` 和 `git diff` 查看变更，编写准确的提交信息。

### 2. build — 构建站点

```bash
NODE_OPTIONS="--max-old-space-size=8192" npm run docs:build
```

注意：项目有 370+ 个 markdown 文件，必须设置 8GB 堆内存，否则会 OOM。

构建完成后检查是否有错误输出。

### 3. deploy — 部署到 Cloudflare Pages

```bash
npx wrangler pages deploy docs/.vitepress/dist --project-name=full-stack-roadmap
```

需要已通过 `npx wrangler login` 登录 Cloudflare 账号。

部署完成后会输出预览 URL，生产站点为 https://t.tecfav.com

## 执行步骤

1. 根据 `$ARGUMENTS` 决定执行哪些操作（默认执行全部）
2. **commit**: 查看变更 → 编写提交信息 → `git add` → `git commit` → `git push`
3. **build**: 执行 `NODE_OPTIONS="--max-old-space-size=8192" npm run docs:build`，检查构建错误
4. **deploy**: 执行 `npx wrangler pages deploy`，确认部署成功
5. 汇总报告结果

## 快捷命令

一键全流程（等同于 `npm run deploy`，但包含提交推送）：

```bash
git add -A && git commit -m "消息" && git push
NODE_OPTIONS="--max-old-space-size=8192" npm run docs:build
npx wrangler pages deploy docs/.vitepress/dist --project-name=full-stack-roadmap
```

## 常见问题

- 如果构建 OOM，确认 `NODE_OPTIONS="--max-old-space-size=8192"` 已设置
- 如果构建报 "Invalid end tag"，检查 `docs/index.md` 中 HTML 标签是否被 `fix-markdown.js` 错误转义
- 如果 wrangler 部署报错未登录，先执行 `npx wrangler login`
- 如果需要只构建不部署，用 `npm run docs:build`
- 死链接已在 VitePress 配置中设置为忽略 (`ignoreDeadLinks: true`)
