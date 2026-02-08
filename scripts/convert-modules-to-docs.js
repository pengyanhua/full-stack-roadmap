#!/usr/bin/env node

/**
 * 将模块内容转换为文档
 */

import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const ROOT_DIR = path.resolve(__dirname, '..')

// 模块映射配置
const MODULES = {
  'AI_Programming': {
    source: 'AI_Programming',
    dest: 'docs/ai-programming',
    type: 'markdown'  // AI 编程教程
  },
  'Architecture': {
    source: 'Architecture',
    dest: 'docs/architecture',
    type: 'markdown'  // 系统架构文档
  },
  'Computer_Hardware': {
    source: 'Computer_Hardware',
    dest: 'docs/computer-hardware',
    type: 'markdown'
  },
  'Operating_Systems': {
    source: 'Operating_Systems',
    dest: 'docs/operating-systems',
    type: 'markdown'
  },
  'Cloud_Native': {
    source: 'Cloud_Native',
    dest: 'docs/cloud-native',
    type: 'markdown'
  },
  'DevOps': {
    source: 'DevOps',
    dest: 'docs/devops',
    type: 'markdown'
  },
  'API_Gateway': {
    source: 'API_Gateway',
    dest: 'docs/api-gateway',
    type: 'markdown'
  },
  'DDD': {
    source: 'DDD',
    dest: 'docs/ddd',
    type: 'markdown'
  },
  'Performance': {
    source: 'Performance',
    dest: 'docs/performance',
    type: 'markdown'
  },
  'Governance': {
    source: 'Governance',
    dest: 'docs/governance',
    type: 'markdown'
  },
  'Data_Architecture': {
    source: 'Data_Architecture',
    dest: 'docs/data-architecture',
    type: 'markdown'
  },
  'Security_Advanced': {
    source: 'Security_Advanced',
    dest: 'docs/security',
    type: 'markdown'
  },
  'BigData': {
    source: 'BigData',
    dest: 'docs/bigdata',
    type: 'markdown'
  },
  'AI_Architecture': {
    source: 'AI_Architecture',
    dest: 'docs/ai-architecture',
    type: 'markdown'
  },
  'Soft_Skills': {
    source: 'Soft_Skills',
    dest: 'docs/soft-skills',
    type: 'markdown'
  },
  'Container': {
    source: 'Container',
    dest: 'docs/container',
    type: 'markdown'
  },
  'Elasticsearch': {
    source: 'Elasticsearch',
    dest: 'docs/elasticsearch',
    type: 'markdown'
  },
  'Kafka': {
    source: 'Kafka',
    dest: 'docs/kafka',
    type: 'markdown'
  },
  'MySQL': {
    source: 'MySQL',
    dest: 'docs/mysql',
    type: 'sql'
  },
  'PostgreSQL': {
    source: 'PostgreSQL',
    dest: 'docs/postgresql',
    type: 'sql'
  },
  'Redis': {
    source: 'Redis',
    dest: 'docs/redis',
    type: 'redis'
  }
}

/**
 * 转换 SQL 文件为 Markdown
 */
function convertSqlToMarkdown(filePath, filename) {
  const content = fs.readFileSync(filePath, 'utf-8')
  const title = filename.replace(/^\d+_/, '').replace(/\.sql$/, '').replace(/_/g, ' ')

  let markdown = `# ${title}\n\n`
  markdown += `::: info 文件信息\n`
  markdown += `- 📄 原文件：\`${filename}\`\n`
  markdown += `- 🔤 语言：SQL\n`
  markdown += `:::\n\n`
  markdown += `## SQL 脚本\n\n`
  markdown += `\`\`\`sql\n`
  markdown += content
  markdown += `\n\`\`\`\n`

  return markdown
}

/**
 * 转换 Redis 命令文件为 Markdown
 */
function convertRedisToMarkdown(filePath, filename) {
  const content = fs.readFileSync(filePath, 'utf-8')
  const title = filename.replace(/^\d+_/, '').replace(/\.redis$/, '').replace(/_/g, ' ')

  let markdown = `# ${title}\n\n`
  markdown += `::: info 文件信息\n`
  markdown += `- 📄 原文件：\`${filename}\`\n`
  markdown += `- 🔤 类型：Redis Commands\n`
  markdown += `:::\n\n`
  markdown += `## Redis 命令\n\n`
  markdown += `\`\`\`redis\n`
  markdown += content
  markdown += `\n\`\`\`\n`

  return markdown
}

/**
 * 处理单个模块
 */
function processModule(moduleName, config) {
  const sourcePath = path.join(ROOT_DIR, config.source)
  const destPath = path.join(ROOT_DIR, config.dest)

  console.log(`\n📁 处理模块: ${moduleName}`)
  console.log(`   源目录: ${config.source}`)
  console.log(`   目标目录: ${config.dest}`)

  // 确保目标目录存在
  if (!fs.existsSync(destPath)) {
    fs.mkdirSync(destPath, { recursive: true })
  }

  // 读取源目录中的文件
  const files = fs.readdirSync(sourcePath)
  let count = 0

  for (const file of files) {
    const filePath = path.join(sourcePath, file)
    const stat = fs.statSync(filePath)

    if (stat.isDirectory()) {
      // 递归处理子目录
      const subDestPath = path.join(destPath, file)
      if (!fs.existsSync(subDestPath)) {
        fs.mkdirSync(subDestPath, { recursive: true })
      }

      const subFiles = fs.readdirSync(filePath)
      for (const subFile of subFiles) {
        const subFilePath = path.join(filePath, subFile)
        const subStat = fs.statSync(subFilePath)

        if (subStat.isFile()) {
          let destFile, content

          if (config.type === 'markdown' && subFile.endsWith('.md')) {
            destFile = path.join(subDestPath, subFile)
            content = fs.readFileSync(subFilePath, 'utf-8')
          }

          if (destFile && content) {
            fs.writeFileSync(destFile, content, 'utf-8')
            console.log(`   ✓ ${file}/${subFile}`)
            count++
          }
        }
      }
    } else if (stat.isFile()) {
      let destFile, content

      if (config.type === 'markdown') {
        // Markdown 文件直接复制
        if (file.endsWith('.md')) {
          destFile = path.join(destPath, file)
          content = fs.readFileSync(filePath, 'utf-8')
        } else if (file.endsWith('.yaml') || file.endsWith('.yml')) {
          // YAML 文件转换为 Markdown
          const baseName = file.replace(/\.(yaml|yml)$/, '')
          destFile = path.join(destPath, baseName + '.md')
          const yamlContent = fs.readFileSync(filePath, 'utf-8')
          content = `# ${baseName}\n\n\`\`\`yaml\n${yamlContent}\n\`\`\`\n`
        } else if (file.endsWith('.sh')) {
          // Shell 脚本转换为 Markdown
          const baseName = file.replace(/\.sh$/, '')
          destFile = path.join(destPath, baseName + '.md')
          const shContent = fs.readFileSync(filePath, 'utf-8')
          content = `# ${baseName}\n\n\`\`\`bash\n${shContent}\n\`\`\`\n`
        } else {
          continue
        }
      } else if (config.type === 'sql') {
        if (!file.endsWith('.sql')) continue
        destFile = path.join(destPath, file.replace('.sql', '.md'))
        content = convertSqlToMarkdown(filePath, file)
      } else if (config.type === 'redis') {
        if (!file.endsWith('.redis')) continue
        destFile = path.join(destPath, file.replace('.redis', '.md'))
        content = convertRedisToMarkdown(filePath, file)
      }

      if (destFile && content) {
        fs.writeFileSync(destFile, content, 'utf-8')
        console.log(`   ✓ ${file} -> ${path.basename(destFile)}`)
        count++
      }
    }
  }

  console.log(`   完成！共转换 ${count} 个文件`)
}

/**
 * 主函数
 */
function main() {
  console.log('🚀 开始转换模块内容为文档...\n')
  console.log('='.repeat(60))

  for (const [moduleName, config] of Object.entries(MODULES)) {
    try {
      processModule(moduleName, config)
    } catch (error) {
      console.error(`   ✗ 处理失败: ${error.message}`)
    }
  }

  console.log('\n' + '='.repeat(60))
  console.log('✅ 所有模块转换完成！')
}

main()
