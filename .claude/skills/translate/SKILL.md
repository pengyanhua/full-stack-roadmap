---
name: translate
description: 翻译项目内容。用于中英文互译代码注释或文档。当用户说"翻译"、"translate"、"转成英文"、"转成中文"时使用。
argument-hint: <文件路径> [to-en|to-zh]
disable-model-invocation: true
allowed-tools: Read, Write, Edit, Glob
---

# 翻译内容

翻译项目中的代码注释或文档内容。

## 参数

- `$ARGUMENTS` — 格式为 `<文件路径> [方向]`，例如：
  - `Go/04-concurrency/01_goroutines.go to-en` — 将中文注释翻译为英文
  - `README.md to-zh` — 翻译为中文
  - `Go/04-concurrency/` — 翻译整个目录（默认中→英）

方向默认值：
- `.go/.py/.java/.js` 代码文件 → `to-en`（代码注释中→英）
- `.md` 文档文件 → 需要明确指定方向

## 翻译规则

### 代码文件翻译

1. **只翻译注释**，不修改任何代码逻辑
2. 保持分隔符格式不变（`====` 和 `----`）
3. 保持注释的位置和缩进不变
4. 字符串中的中文内容（如 `fmt.Println("你好")`）保持不变，除非是纯演示文本
5. 翻译要自然流畅，使用技术领域标准术语

### 文档文件翻译

1. 保持 Markdown 格式不变
2. 保持 VitePress 特有语法不变（如 `::: info`、`::: tip`）
3. 代码块内容不翻译
4. 链接文本翻译，URL 不变

### 术语对照

保持以下术语一致：
- Goroutine → goroutine（不翻译）
- 闭包 → closure
- 并发 → concurrency
- 协程 → coroutine
- 通道 → channel
- 接口 → interface
- 切片 → slice
- 映射 → map
- 指针 → pointer
- 结构体 → struct
- 包 → package
- 泛型 → generics

## 执行步骤

1. 读取目标文件
2. 判断翻译方向
3. 执行翻译，严格遵循翻译规则
4. 将翻译后的内容写回原文件（覆盖）或创建新文件（如 `_en` 后缀）
5. 询问用户：是覆盖原文件还是创建新文件

## 注意事项

- 翻译前先读取文件，确认内容
- 对于大文件（>200行），分段翻译以确保质量
- 翻译完成后提醒用户检查专业术语是否准确
