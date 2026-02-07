#!/usr/bin/env node

/**
 * 安全检查脚本 - 扫描可能的敏感信息
 */

import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const ROOT_DIR = path.resolve(__dirname, '..')

// 敏感关键词模式
const SENSITIVE_PATTERNS = [
  { pattern: /password\s*=\s*['"][^'"]{8,}['"]/gi, name: 'Hard-coded password' },
  { pattern: /api[_-]?key\s*=\s*['"][^'"]{20,}['"]/gi, name: 'API key' },
  { pattern: /secret\s*=\s*['"][^'"]{20,}['"]/gi, name: 'Secret key' },
  { pattern: /token\s*=\s*['"][^'"]{20,}['"]/gi, name: 'Token' },
  { pattern: /Bearer\s+[A-Za-z0-9\-._~+/]+=*/g, name: 'Bearer token' },
  { pattern: /-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----/g, name: 'Private key' },
  { pattern: /sk_live_[0-9a-zA-Z]{24,}/g, name: 'Stripe secret key' },
  { pattern: /AWS[0-9A-Z]{16,}/g, name: 'AWS key' }
]

// 需要检查的文件扩展名
const CHECK_EXTENSIONS = ['.js', '.ts', '.json', '.yml', '.yaml', '.env', '.md']

// 排除的目录
const EXCLUDE_DIRS = ['node_modules', '.git', 'dist', 'build', 'docs/.vitepress/dist']

// 排除的文件（已知的教程示例）
const EXCLUDE_FILES = [
  'Architecture/01_system_design/01_design_principles.md',
  'Architecture/05_microservices/02_api_design.md',
  'SECURITY.md'
]

let foundIssues = []

/**
 * 检查文件内容
 */
function checkFile(filePath, relativePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf-8')
    const lines = content.split('\n')

    for (const { pattern, name } of SENSITIVE_PATTERNS) {
      pattern.lastIndex = 0 // Reset regex

      let match
      while ((match = pattern.exec(content)) !== null) {
        // 找到匹配的行号
        let lineNum = 1
        let pos = 0
        for (let i = 0; i < lines.length; i++) {
          pos += lines[i].length + 1
          if (pos > match.index) {
            lineNum = i + 1
            break
          }
        }

        foundIssues.push({
          file: relativePath,
          line: lineNum,
          type: name,
          preview: match[0].substring(0, 50) + '...'
        })
      }
    }
  } catch (error) {
    // Ignore read errors
  }
}

/**
 * 扫描目录
 */
function scanDirectory(dir, baseDir = dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true })

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name)
    const relativePath = path.relative(ROOT_DIR, fullPath).replace(/\\/g, '/')

    // 排除目录
    if (entry.isDirectory()) {
      if (!EXCLUDE_DIRS.some(excl => relativePath.includes(excl))) {
        scanDirectory(fullPath, baseDir)
      }
    } else if (entry.isFile()) {
      // 检查文件扩展名
      const ext = path.extname(entry.name)
      if (CHECK_EXTENSIONS.includes(ext)) {
        // 排除已知的教程文件
        if (!EXCLUDE_FILES.some(excl => relativePath === excl)) {
          checkFile(fullPath, relativePath)
        }
      }
    }
  }
}

/**
 * 主函数
 */
function main() {
  console.log('🔍 开始安全扫描...\n')

  scanDirectory(ROOT_DIR)

  console.log('='.repeat(60))

  if (foundIssues.length === 0) {
    console.log('✅ 未发现可疑的敏感信息！')
    console.log('\n扫描了以下文件类型:', CHECK_EXTENSIONS.join(', '))
    console.log('排除了教程示例文件')
  } else {
    console.log(`⚠️  发现 ${foundIssues.length} 个可能的问题：\n`)

    for (const issue of foundIssues) {
      console.log(`📄 ${issue.file}:${issue.line}`)
      console.log(`   类型: ${issue.type}`)
      console.log(`   内容: ${issue.preview}`)
      console.log()
    }

    console.log('⚠️  请检查以上内容是否为真实的敏感信息')
    console.log('   如果是示例代码，可以添加到 EXCLUDE_FILES 中')
  }

  console.log('='.repeat(60))

  process.exit(foundIssues.length > 0 ? 1 : 0)
}

main()
