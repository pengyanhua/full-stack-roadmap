#!/usr/bin/env node

/**
 * 修复 Markdown 文件中的 HTML 标签问题
 */

import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const DOCS_DIR = path.resolve(__dirname, '../docs')

/**
 * 修复文件内容
 */
function fixMarkdownContent(content, filePath) {
  let fixed = content
  let changes = []

  // 常见 HTML 标签白名单，不应被转义
  const htmlTags = new Set([
    'div', 'span', 'p', 'a', 'br', 'hr', 'img', 'ul', 'ol', 'li',
    'table', 'tr', 'td', 'th', 'thead', 'tbody', 'tfoot',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'pre', 'code', 'blockquote', 'em', 'strong', 'b', 'i', 'u', 's',
    'script', 'style', 'link', 'meta', 'head', 'body', 'html',
    'form', 'input', 'button', 'select', 'option', 'textarea', 'label',
    'section', 'article', 'nav', 'aside', 'header', 'footer', 'main',
    'details', 'summary', 'ins', 'del', 'sup', 'sub', 'small', 'mark',
    'video', 'audio', 'source', 'iframe', 'canvas', 'svg', 'path',
  ])

  // 1. 修复代码块外的 <id> 和类似标签
  // 但不修改代码块内的内容
  const lines = content.split('\n')
  let inCodeBlock = false
  let fixedLines = []

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i]

    // 检测代码块
    if (line.trim().startsWith('```')) {
      inCodeBlock = !inCodeBlock
      fixedLines.push(line)
      continue
    }

    // 如果不在代码块中，转义尖括号
    if (!inCodeBlock) {
      // 检查是否包含类似 <id>, <命令> 等占位符
      const placeholderPattern = /<([a-zA-Z\u4e00-\u9fa5]+)>/g
      if (placeholderPattern.test(line)) {
        const originalLine = line
        // 只转义非 HTML 标签的尖括号（跳过已知 HTML 标签）
        line = line.replace(/<([a-zA-Z\u4e00-\u9fa5]+)>/g, (match, tag) => {
          if (htmlTags.has(tag.toLowerCase())) return match
          return `&lt;${tag}&gt;`
        })
        if (line !== originalLine) {
          changes.push(`Line ${i + 1}: 转义占位符`)
        }
      }
    }

    fixedLines.push(line)
  }

  fixed = fixedLines.join('\n')

  if (changes.length > 0) {
    console.log(`  ✓ ${path.relative(DOCS_DIR, filePath)}`)
    changes.forEach(change => console.log(`    - ${change}`))
  }

  return { content: fixed, hasChanges: changes.length > 0 }
}

/**
 * 处理目录
 */
function processDirectory(dir) {
  let totalFiles = 0
  let fixedFiles = 0

  const entries = fs.readdirSync(dir, { withFileTypes: true })

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name)

    if (entry.isDirectory()) {
      // 跳过隐藏目录
      if (entry.name.startsWith('.')) continue

      const result = processDirectory(fullPath)
      totalFiles += result.total
      fixedFiles += result.fixed
    } else if (entry.isFile() && entry.name.endsWith('.md')) {
      totalFiles++

      try {
        const content = fs.readFileSync(fullPath, 'utf-8')
        const { content: fixedContent, hasChanges } = fixMarkdownContent(content, fullPath)

        if (hasChanges) {
          fs.writeFileSync(fullPath, fixedContent, 'utf-8')
          fixedFiles++
        }
      } catch (error) {
        console.error(`  ✗ 处理失败: ${path.relative(DOCS_DIR, fullPath)} - ${error.message}`)
      }
    }
  }

  return { total: totalFiles, fixed: fixedFiles }
}

/**
 * 主函数
 */
function main() {
  console.log('🔧 修复 Markdown 文件...\n')

  const result = processDirectory(DOCS_DIR)

  console.log('\n' + '='.repeat(60))
  console.log('✅ 修复完成！')
  console.log(`   总文件数: ${result.total}`)
  console.log(`   修复文件: ${result.fixed}`)
  console.log(`   未修改: ${result.total - result.fixed}`)
  console.log('='.repeat(60))
}

main()
