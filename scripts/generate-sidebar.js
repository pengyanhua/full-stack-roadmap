#!/usr/bin/env node

/**
 * 自动生成 VitePress 侧边栏配置
 */

import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const DOCS_DIR = path.resolve(__dirname, '../docs')

// 模块名称映射
const MODULE_NAMES = {
  'python': 'Python',
  'go': 'Go',
  'java': 'Java',
  'javascript': 'JavaScript'
}

// 章节名称映射
const SECTION_NAMES = {
  '01-basics': '01 - 基础',
  '02-functions': '02 - 函数',
  '03-classes': '03 - 类与对象',
  '03-structs': '03 - 结构体',
  '02-oop': '02 - 面向对象',
  '03-collections': '03 - 集合框架',
  '04-concurrency': '04 - 并发编程',
  '04-async': '04 - 异步编程',
  '05-io': '05 - I/O 操作',
  '05-modules': '05 - 模块与包',
  '05-packages': '05 - 包管理',
  '05-typescript': '05 - TypeScript',
  '06-functional': '06 - 函数式编程',
  '06-testing': '06 - 测试',
  '07-modern': '07 - 现代特性',
  '07-stdlib': '07 - 标准库',
  '07-node': '07 - Node.js',
  '08-projects': '08 - 项目实战'
}

/**
 * 扫描目录获取所有 md 文件
 */
function scanDirectory(dir) {
  const result = {}

  try {
    const entries = fs.readdirSync(dir, { withFileTypes: true })

    for (const entry of entries) {
      if (entry.isDirectory() && !entry.name.startsWith('.')) {
        const sectionPath = path.join(dir, entry.name)
        const files = fs.readdirSync(sectionPath)
          .filter(f => f.endsWith('.md') && f !== 'index.md')
          .sort()

        if (files.length > 0) {
          result[entry.name] = files.map(f => ({
            name: f.replace('.md', ''),
            file: f
          }))
        }
      }
    }
  } catch (error) {
    console.error(`扫描目录失败 ${dir}: ${error.message}`)
  }

  return result
}

/**
 * 生成文件标题
 */
function getFileTitle(filename) {
  // 移除编号前缀和文件扩展名
  let title = filename.replace(/^\d+[-_]/, '').replace('.md', '')

  // 转换下划线和连字符为空格，首字母大写
  title = title
    .replace(/[-_]/g, ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')

  return title
}

/**
 * 生成侧边栏配置
 */
function generateSidebar() {
  const sidebar = {}

  for (const [module, moduleName] of Object.entries(MODULE_NAMES)) {
    const moduleDir = path.join(DOCS_DIR, module)

    if (!fs.existsSync(moduleDir)) continue

    const sections = scanDirectory(moduleDir)

    const items = [
      { text: '简介', link: `/${module}/` }
    ]

    // 按章节编号排序
    const sortedSections = Object.keys(sections).sort()

    for (const sectionName of sortedSections) {
      const files = sections[sectionName]

      const sectionItem = {
        text: SECTION_NAMES[sectionName] || sectionName,
        collapsed: true,
        items: files.map(f => ({
          text: getFileTitle(f.name),
          link: `/${module}/${sectionName}/${f.name}`
        }))
      }

      items.push(sectionItem)
    }

    sidebar[`/${module}/`] = [
      {
        text: `${moduleName} 学习路径`,
        items
      }
    ]
  }

  return sidebar
}

/**
 * 主函数
 */
function main() {
  console.log('🚀 生成侧边栏配置...\n')

  const sidebar = generateSidebar()

  const output = JSON.stringify(sidebar, null, 2)

  console.log('✅ 侧边栏配置：\n')
  console.log(output)
  console.log('\n📝 请将以上配置复制到 docs/.vitepress/config.ts 的 sidebar 字段中')
}

main()
