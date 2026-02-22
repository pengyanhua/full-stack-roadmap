---
name: publish
description: 将源码转换为文档并构建 VitePress 站点。当用户说"发布"、"更新文档"、"构建文档"、"生成文档"时使用。
argument-hint: [convert|build|preview|all]
disable-model-invocation: true
allowed-tools: Bash, Read, Glob, Grep
---

# 发布文档

将源码文件转换为 Markdown 文档，并构建 VitePress 站点。

## 参数

- `$ARGUMENTS` — 可选，指定操作：
  - `convert` — 仅转换代码为 Markdown
  - `build` — 仅构建 VitePress
  - `preview` — 构建并启动预览服务器
  - `all`（默认） — 转换 + 构建
  - 空 — 等同于 `all`

## 执行流程

### 1. convert — 代码转 Markdown

```bash
npm run convert
```

这会扫描 Python, Go, Java, JavaScript, React, Vue, DataStructures 目录下的代码文件，转换为 `docs/` 下对应的 Markdown 文件。

转换规则：
- `Python/02-functions/02_closure.py` → `docs/python/02-functions/closure.md`
- 文件名数字前缀被去除
- 目录名转为小写

### 2. build — 构建站点

```bash
npm run docs:build
```

构建完成后检查是否有错误输出。

### 3. preview — 预览站点

```bash
npm run docs:preview
```

启动本地预览服务器。

## 执行步骤

1. 根据 `$ARGUMENTS` 决定执行哪些操作（默认执行全部）
2. 执行 `npm run convert`，报告转换结果（成功/失败文件数）
3. 执行 `npm run docs:build`，检查构建错误
4. 如果用户指定 `preview`，启动预览服务器
5. 汇总报告结果

## 常见问题

- 如果 convert 报告某些文件失败，检查文件编码是否为 UTF-8
- 如果 build 失败，检查 Markdown 语法错误（常见：未闭合的代码块、无效链接）
- 死链接已在 VitePress 配置中设置为忽略 (`ignoreDeadLinks: true`)
