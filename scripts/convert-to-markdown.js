#!/usr/bin/env node

/**
 * 代码文件转 Markdown 脚本
 *
 * 功能：
 * 1. 扫描所有代码文件（.py, .go, .java, .js）
 * 2. 解析代码中的注释和分节
 * 3. 转换为格式化的 Markdown 文档
 * 4. 保留原有的详细注释风格
 */

import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const ROOT_DIR = path.resolve(__dirname, '..')
const DOCS_DIR = path.join(ROOT_DIR, 'docs')

// 语言配置
const LANG_CONFIG = {
  '.py': { name: 'python', comment: '#' },
  '.go': { name: 'go', comment: '//' },
  '.java': { name: 'java', comment: '//' },
  '.js': { name: 'javascript', comment: '//' },
  '.ts': { name: 'typescript', comment: '//' }
}

// 需要转换的目录
const CONVERT_DIRS = ['Python', 'Go', 'Java', 'JavaScript']

/**
 * 解析代码文件，提取结构化内容
 */
function parseCodeFile(content, ext) {
  const config = LANG_CONFIG[ext]
  if (!config) return null

  const lines = content.split('\n')
  const sections = []
  let currentSection = null
  let codeBuffer = []
  let inSeparator = false

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trim()

    // 检测分隔符（如 ============）
    if (trimmed.match(/^[=#-]{20,}$/)) {
      if (!inSeparator) {
        // 开始新的分隔符区域
        inSeparator = true

        // 保存之前的代码缓冲
        if (currentSection && codeBuffer.length > 0) {
          currentSection.code = codeBuffer.join('\n')
          codeBuffer = []
        }
      } else {
        // 结束分隔符区域
        inSeparator = false
      }
      continue
    }

    // 在分隔符中间，提取标题
    if (inSeparator) {
      const titleMatch = trimmed.match(/^[#/\*\s]*(.+?)[\*\s]*$/)
      if (titleMatch && titleMatch[1]) {
        const title = titleMatch[1].trim()

        // 保存上一个 section
        if (currentSection && codeBuffer.length > 0) {
          currentSection.code = codeBuffer.join('\n')
        }

        // 创建新的 section
        currentSection = {
          title,
          level: getHeaderLevel(title),
          code: '',
          lineNumber: i + 1
        }
        sections.push(currentSection)
        codeBuffer = []
      }
      continue
    }

    // 普通代码行
    codeBuffer.push(line)
  }

  // 保存最后的缓冲
  if (currentSection && codeBuffer.length > 0) {
    currentSection.code = codeBuffer.join('\n')
  }

  return sections
}

/**
 * 判断标题级别
 */
function getHeaderLevel(title) {
  // 如果包含数字编号（如 "1. "），是二级标题
  if (/^\d+[\.\、]/.test(title)) return 2
  // 如果是中文主标题，是一级标题
  return 1
}

/**
 * 转换为 Markdown
 */
function convertToMarkdown(sections, filename, ext) {
  const config = LANG_CONFIG[ext]
  let markdown = ''

  // 生成文档标题（从文件名提取）
  const titleMatch = filename.match(/\d+_(.+)/)
  const docTitle = titleMatch ? titleMatch[1].replace(/_/g, ' ') : filename
  markdown += `# ${docTitle}\n\n`

  // 添加文件信息
  markdown += `::: info 文件信息\n`
  markdown += `- 📄 原文件：\`${filename}\`\n`
  markdown += `- 🔤 语言：${config.name}\n`
  markdown += `:::\n\n`

  // 转换各个 section
  for (const section of sections) {
    // 添加标题
    markdown += `${'#'.repeat(section.level + 1)} ${section.title}\n\n`

    // 处理代码块
    if (section.code.trim()) {
      // 清理代码（移除文件头的多余空行）
      const cleanCode = cleanCodeBlock(section.code, config.comment)

      markdown += `\`\`\`${config.name}\n`
      markdown += cleanCode
      markdown += `\n\`\`\`\n\n`
    }
  }

  return markdown
}

/**
 * 清理代码块
 */
function cleanCodeBlock(code, commentChar) {
  const lines = code.split('\n')
  let cleanedLines = []

  for (let line of lines) {
    const trimmed = line.trim()

    // 跳过文件头的编码声明（如果还有的话）
    if (trimmed.startsWith('#!') || trimmed.match(/^#.*coding[:=]/)) {
      continue
    }

    cleanedLines.push(line)
  }

  // 移除首尾空行
  while (cleanedLines.length > 0 && cleanedLines[0].trim() === '') {
    cleanedLines.shift()
  }
  while (cleanedLines.length > 0 && cleanedLines[cleanedLines.length - 1].trim() === '') {
    cleanedLines.pop()
  }

  return cleanedLines.join('\n')
}

/**
 * 生成简单的 Markdown（用于没有分节的文件）
 */
function generateSimpleMarkdown(content, filename, ext) {
  const config = LANG_CONFIG[ext]
  const titleMatch = filename.match(/\d+_(.+)/)
  const docTitle = titleMatch ? titleMatch[1].replace(/_/g, ' ') : filename.replace(ext, '')

  let markdown = `# ${docTitle}\n\n`
  markdown += `::: info 文件信息\n`
  markdown += `- 📄 原文件：\`${filename}\`\n`
  markdown += `- 🔤 语言：${config.name}\n`
  markdown += `:::\n\n`

  // 提取 Python 文件顶部的文档字符串（特殊处理）
  let codeContent = content
  if (ext === '.py') {
    // 先移除 shebang 和 encoding
    let lines = content.split('\n')
    let startIdx = 0

    // 跳过 shebang 和 encoding 行
    while (startIdx < lines.length) {
      const line = lines[startIdx].trim()
      if (line.startsWith('#!') || line.match(/^#.*coding[:=]/)) {
        startIdx++
      } else {
        break
      }
    }

    // 重新组合内容
    const remainingContent = lines.slice(startIdx).join('\n')

    // 匹配 Python docstring
    const docstringMatch = remainingContent.match(/^"""([\s\S]*?)"""/) ||
                           remainingContent.match(/^'''([\s\S]*?)'''/)

    if (docstringMatch) {
      const docstring = docstringMatch[1].trim()

      // 提取标题和描述（去掉分隔符）
      const cleanDoc = docstring
        .replace(/^[=\-]{20,}\s*/gm, '')  // 移除分隔线
        .replace(/\s*[=\-]{20,}$/gm, '')
        .replace(/<([a-zA-Z\u4e00-\u9fa5]+)>/g, '`<$1>`')  // 转义占位符为代码
        .trim()

      if (cleanDoc) {
        // 添加描述信息
        markdown += `${cleanDoc}\n\n`
      }

      // 从代码中移除 docstring
      codeContent = remainingContent.replace(/^"""[\s\S]*?"""\s*/, '')
                                   .replace(/^'''[\s\S]*?'''\s*/, '')
    } else {
      codeContent = remainingContent
    }
  } else {
    // 其他语言的文档字符串提取（Go, Java, JS）
    const docstringMatch = content.match(/^\/\*\*([\s\S]*?)\*\//m)

    if (docstringMatch) {
      const docstring = docstringMatch[1].trim()
        .replace(/^\s*\*\s?/gm, '')  // 移除每行开头的 *
        .replace(/^[=\-]{20,}\s*/gm, '')  // 移除分隔线
        .replace(/\s*[=\-]{20,}$/gm, '')
        .replace(/<([a-zA-Z\u4e00-\u9fa5]+)>/g, '`<$1>`')  // 转义占位符为代码
        .trim()

      if (docstring) {
        markdown += `${docstring}\n\n`
      }
    }
  }

  markdown += `## 完整代码\n\n`
  markdown += `\`\`\`${config.name}\n`
  markdown += cleanCodeBlock(codeContent, config.comment)
  markdown += `\n\`\`\`\n`

  return markdown
}

/**
 * 处理单个代码文件
 */
function processCodeFile(filePath, relativePath) {
  const ext = path.extname(filePath)
  const filename = path.basename(filePath)

  console.log(`  转换: ${relativePath}`)

  try {
    const content = fs.readFileSync(filePath, 'utf-8')

    // 直接使用简单格式，展示完整代码
    const markdown = generateSimpleMarkdown(content, filename, ext)

    // 生成输出路径
    const outputPath = getOutputPath(relativePath)
    const outputDir = path.dirname(outputPath)

    // 确保输出目录存在
    fs.mkdirSync(outputDir, { recursive: true })

    // 写入文件
    fs.writeFileSync(outputPath, markdown, 'utf-8')

    return true
  } catch (error) {
    console.error(`    ❌ 转换失败: ${error.message}`)
    return false
  }
}

/**
 * 获取输出路径
 */
function getOutputPath(relativePath) {
  // Python/02-functions/02_closure.py -> docs/python/02-functions/closure.md
  const parts = relativePath.split(path.sep)
  const lang = parts[0].toLowerCase() // Python -> python

  // 移除编号前缀，生成文件名
  const filename = path.basename(relativePath, path.extname(relativePath))
  const cleanFilename = filename.replace(/^\d+_/, '') + '.md'

  // 构建路径
  const pathParts = [DOCS_DIR, lang, ...parts.slice(1, -1), cleanFilename]

  return path.join(...pathParts)
}

/**
 * 扫描目录
 */
function scanDirectory(dir, baseDir = dir) {
  const files = []

  try {
    const entries = fs.readdirSync(dir, { withFileTypes: true })

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name)

      if (entry.isDirectory()) {
        // 递归扫描子目录
        files.push(...scanDirectory(fullPath, baseDir))
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name)
        if (LANG_CONFIG[ext]) {
          const relativePath = path.relative(baseDir, fullPath)
          files.push({ fullPath, relativePath })
        }
      }
    }
  } catch (error) {
    console.error(`扫描目录失败 ${dir}: ${error.message}`)
  }

  return files
}

/**
 * 主函数
 */
function main() {
  console.log('🚀 开始转换代码文件为 Markdown...\n')

  let totalFiles = 0
  let successFiles = 0

  for (const dir of CONVERT_DIRS) {
    const dirPath = path.join(ROOT_DIR, dir)

    if (!fs.existsSync(dirPath)) {
      console.log(`⚠️  目录不存在: ${dir}`)
      continue
    }

    console.log(`📁 处理目录: ${dir}`)

    const files = scanDirectory(dirPath, ROOT_DIR)

    for (const { fullPath, relativePath } of files) {
      totalFiles++
      if (processCodeFile(fullPath, relativePath)) {
        successFiles++
      }
    }

    console.log()
  }

  console.log('=' .repeat(60))
  console.log(`✅ 转换完成！`)
  console.log(`   总文件数: ${totalFiles}`)
  console.log(`   成功: ${successFiles}`)
  console.log(`   失败: ${totalFiles - successFiles}`)
  console.log('=' .repeat(60))
}

// 运行
main()
