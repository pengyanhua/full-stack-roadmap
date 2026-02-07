import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "全栈开发学习路线",
  description: "从基础到进阶的全栈学习资源，涵盖编程语言、框架、数据库、系统架构和数据结构",
  lang: 'zh-CN',

  // 忽略死链接（用于还未创建的页面）
  ignoreDeadLinks: true,

  // 主题配置
  themeConfig: {
    // Logo
    logo: '/logo.svg',

    // 导航栏
    nav: [
      { text: '首页', link: '/' },
      {
        text: '编程语言',
        items: [
          { text: 'Python', link: '/python/' },
          { text: 'Go', link: '/go/' },
          { text: 'Java', link: '/java/' },
          { text: 'JavaScript', link: '/javascript/' },
        ]
      },
      {
        text: '前端框架',
        items: [
          { text: 'React', link: '/react/' },
          { text: 'Vue', link: '/vue/' },
        ]
      },
      {
        text: '后端技术',
        items: [
          { text: 'MySQL', link: '/mysql/' },
          { text: 'PostgreSQL', link: '/postgresql/' },
          { text: 'Redis', link: '/redis/' },
          { text: 'Elasticsearch', link: '/elasticsearch/' },
          { text: 'Kafka', link: '/kafka/' },
        ]
      },
      {
        text: '系统架构',
        link: '/architecture/'
      },
      {
        text: '数据结构',
        link: '/datastructures/'
      },
      {
        text: '容器化',
        link: '/container/'
      },
      {
        text: 'GitHub',
        link: 'https://github.com/pengyanhua/full-stack-roadmap'
      }
    ],

    // 侧边栏
    sidebar: {
      '/python/': [
        {
          text: 'Python 学习路径',
          items: [
            { text: '简介', link: '/python/' },
            { text: '01 - 基础', link: '/python/01-basics/variables' },
            { text: '02 - 函数', link: '/python/02-functions/basics' },
            { text: '03 - 类与对象', link: '/python/03-classes/class_basics' },
            { text: '04 - 异步编程', link: '/python/04-async/threading' },
            { text: '05 - 模块与包', link: '/python/05-modules/main' },
            { text: '06 - 测试', link: '/python/06-testing/calculator' },
            { text: '07 - 标准库', link: '/python/07-stdlib/os_sys' },
            { text: '08 - 项目实战', link: '/python/08-projects/todo_cli' }
          ]
        }
      ],

      '/go/': [
        {
          text: 'Go 学习路径',
          items: [
            { text: '简介', link: '/go/' },
            { text: '01 - 基础', link: '/go/01-basics/variables' },
            { text: '02 - 函数', link: '/go/02-functions/basics' },
            { text: '03 - 结构体', link: '/go/03-structs/struct_basics' },
            { text: '04 - 并发编程', link: '/go/04-concurrency/goroutines' },
            { text: '05 - 包管理', link: '/go/05-packages/main' },
            { text: '06 - 测试', link: '/go/06-testing/calculator' },
            { text: '07 - 标准库', link: '/go/07-stdlib/strings_fmt' },
            { text: '08 - 项目实战', link: '/go/08-projects/todo_cli' }
          ]
        }
      ],

      '/java/': [
        {
          text: 'Java 学习路径',
          items: [
            { text: '简介', link: '/java/' },
            { text: '01 - 基础', link: '/java/01-basics/Variables' },
            { text: '02 - 面向对象', link: '/java/02-oop/ClassBasics' },
            { text: '03 - 集合框架', link: '/java/03-collections/ListDemo' },
            { text: '04 - 并发编程', link: '/java/04-concurrency/ThreadBasics' },
            { text: '05 - I/O 操作', link: '/java/05-io/FileIO' },
            { text: '06 - 函数式编程', link: '/java/06-functional/Lambda' },
            { text: '07 - 现代特性', link: '/java/07-modern/Records' },
            { text: '08 - 项目实战', link: '/java/08-projects/TodoApp' }
          ]
        }
      ],

      '/javascript/': [
        {
          text: 'JavaScript 学习路径',
          items: [
            { text: '简介', link: '/javascript/' },
            { text: '01 - 基础', link: '/javascript/01-basics/variables' },
            { text: '02 - 函数', link: '/javascript/02-functions/basics' },
            { text: '03 - 异步编程', link: '/javascript/03-async/promises' },
            { text: '04 - 面向对象', link: '/javascript/04-oop/classes' },
            { text: '05 - TypeScript', link: '/javascript/05-typescript/types' },
            { text: '06 - 函数式编程', link: '/javascript/06-functional/functional_basics' },
            { text: '07 - Node.js', link: '/javascript/07-node/filesystem' },
            { text: '08 - 项目实战', link: '/javascript/08-projects/todo_cli' }
          ]
        }
      ],

      '/architecture/': [
        {
          text: '系统架构',
          items: [
            { text: '简介', link: '/architecture/' }
          ]
        }
      ],

      '/guide/': [
        {
          text: '指南',
          items: [
            { text: '快速开始', link: '/guide/getting-started' }
          ]
        }
      ]
    },

    // 社交链接
    socialLinks: [
      { icon: 'github', link: 'https://github.com/pengyanhua/full-stack-roadmap' }
    ],

    // 页脚
    footer: {
      message: '基于 MIT 许可发布',
      copyright: 'Copyright © 2025-present'
    },

    // 搜索
    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档'
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换'
            }
          }
        }
      }
    },

    // 大纲配置
    outline: {
      level: [2, 3],
      label: '页面导航'
    },

    // 文档页脚
    docFooter: {
      prev: '上一页',
      next: '下一页'
    },

    // 最后更新时间文本
    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'short'
      }
    },

    // 返回顶部
    returnToTopLabel: '返回顶部',

    // 侧边栏菜单标签
    sidebarMenuLabel: '菜单',

    // 深色模式切换
    darkModeSwitchLabel: '主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式'
  },

  // Markdown 配置
  markdown: {
    lineNumbers: true, // 显示行号
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    },
    // 代码块配置
    container: {
      tipLabel: '提示',
      warningLabel: '警告',
      dangerLabel: '危险',
      infoLabel: '信息',
      detailsLabel: '详细信息'
    }
  },

  // 最后更新时间
  lastUpdated: true,

  // 站点地图
  sitemap: {
    hostname: 'https://pengyanhua.github.io/full-stack-roadmap'
  }
})
