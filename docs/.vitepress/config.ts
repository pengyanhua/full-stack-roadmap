import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "全栈开发学习路线",
  description: "从基础到进阶的全栈学习资源，涵盖编程语言、框架、数据库、系统架构和数据结构",
  lang: 'zh-CN',

  // Head 配置
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }],
    ['link', { rel: 'icon', type: 'image/png', href: '/logo.svg' }],
    ['meta', { name: 'theme-color', content: '#4F46E5' }],
    ['meta', { name: 'og:type', content: 'website' }],
    ['meta', { name: 'og:locale', content: 'zh_CN' }],
    ['meta', { name: 'og:site_name', content: '全栈开发学习路线' }],
    ['meta', { name: 'og:image', content: 'https://t.tecfav.com/logo.svg' }]
  ],

  // 忽略死链接（用于还未创建的页面）
  ignoreDeadLinks: true,

  // 排除项目元文档
  srcExclude: ['../.meta/**'],

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
        items: [
          { text: '架构设计', link: '/architecture/' },
          { text: 'DDD', link: '/ddd/' },
          { text: 'API 网关', link: '/api-gateway/' },
          { text: '性能优化', link: '/performance/' },
          { text: '技术治理', link: '/governance/' },
        ]
      },
      {
        text: '云原生&DevOps',
        items: [
          { text: '云原生', link: '/cloud-native/' },
          { text: 'DevOps', link: '/devops/' },
          { text: '容器化', link: '/container/' },
        ]
      },
      {
        text: '数据&AI',
        items: [
          { text: '数据架构', link: '/data-architecture/' },
          { text: '大数据', link: '/bigdata/' },
          { text: 'AI 架构', link: '/ai-architecture/' },
        ]
      },
      {
        text: '其他',
        items: [
          { text: '数据结构', link: '/datastructures/' },
          { text: '安全', link: '/security/' },
          { text: '软技能', link: '/soft-skills/' },
        ]
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

      '/react/': [
        {
          text: 'React 学习路径',
          items: [
            { text: '简介', link: '/react/' },
            { text: 'JSX 语法', link: '/react/01-basics/jsx' },
            { text: '组件', link: '/react/01-basics/components' },
            { text: 'useState Hook', link: '/react/02-hooks/useState' },
            { text: 'useEffect Hook', link: '/react/02-hooks/useEffect' },
            { text: 'Context API', link: '/react/03-advanced/context' }
          ]
        }
      ],

      '/vue/': [
        {
          text: 'Vue 学习路径',
          items: [
            { text: '简介', link: '/vue/' },
            { text: '模板语法', link: '/vue/01-basics/template' },
            { text: '组件基础', link: '/vue/01-basics/components' },
            { text: '响应式', link: '/vue/02-composition/reactivity' },
            { text: 'Composables', link: '/vue/02-composition/composables' },
            { text: 'Router', link: '/vue/03-advanced/router' },
            { text: 'Pinia', link: '/vue/03-advanced/pinia' }
          ]
        }
      ],

      '/mysql/': [
        {
          text: 'MySQL 学习路径',
          items: [
            { text: '简介', link: '/mysql/' },
            { text: 'SQL 基础', link: '/mysql/01_basics' },
            { text: '连接与子查询', link: '/mysql/02_joins_subqueries' },
            { text: '索引优化', link: '/mysql/03_indexes_optimization' },
            { text: '事务与约束', link: '/mysql/04_transactions_constraints' },
            { text: '存储过程', link: '/mysql/05_procedures_functions' },
            { text: '实战案例', link: '/mysql/06_practical_examples' }
          ]
        }
      ],

      '/postgresql/': [
        {
          text: 'PostgreSQL 学习路径',
          items: [
            { text: '简介', link: '/postgresql/' },
            { text: 'SQL 基础', link: '/postgresql/01_basics' },
            { text: '连接与子查询', link: '/postgresql/02_joins_subqueries' },
            { text: '高级特性', link: '/postgresql/03_advanced_features' },
            { text: '事务与索引', link: '/postgresql/04_transactions_indexes' },
            { text: '管理运维', link: '/postgresql/05_administration' },
            { text: '实战案例', link: '/postgresql/06_practical_examples' }
          ]
        }
      ],

      '/redis/': [
        {
          text: 'Redis 学习路径',
          items: [
            { text: '简介', link: '/redis/' },
            { text: 'Redis 基础', link: '/redis/01_basics' },
            { text: '高级特性', link: '/redis/02_advanced' },
            { text: '持久化与集群', link: '/redis/03_persistence_cluster' }
          ]
        }
      ],

      '/elasticsearch/': [
        {
          text: 'Elasticsearch 学习路径',
          items: [
            { text: '简介', link: '/elasticsearch/' },
            { text: 'ES 基础', link: '/elasticsearch/01_basics' },
            { text: '查询 DSL', link: '/elasticsearch/02_query_dsl' },
            { text: '聚合分析', link: '/elasticsearch/03_aggregations' }
          ]
        }
      ],

      '/kafka/': [
        {
          text: 'Kafka 学习路径',
          items: [
            { text: '简介', link: '/kafka/' },
            { text: 'Kafka 基础', link: '/kafka/01_basics' },
            { text: '常用命令', link: '/kafka/02_commands' },
            { text: '高级特性', link: '/kafka/03_advanced' }
          ]
        }
      ],

      '/datastructures/': [
        {
          text: '数据结构学习路径',
          items: [
            { text: '简介', link: '/datastructures/' },
            { text: '数组', link: '/datastructures/01_array/implementation' },
            { text: '链表', link: '/datastructures/02_linked_list/implementation' },
            { text: '栈和队列', link: '/datastructures/03_stack_queue/implementation' },
            { text: '哈希表', link: '/datastructures/04_hash_table/implementation' },
            { text: '树', link: '/datastructures/05_tree/implementation' },
            { text: '堆', link: '/datastructures/06_heap/implementation' },
            { text: '图', link: '/datastructures/07_graph/implementation' },
            { text: '高级数据结构', link: '/datastructures/08_advanced/implementation' }
          ]
        }
      ],

      '/container/': [
        {
          text: '容器化学习路径',
          items: [
            { text: '简介', link: '/container/' },
            { text: 'Docker 基础', link: '/container/01_docker_basics' },
            { text: 'Dockerfile', link: '/container/02_dockerfile' },
            { text: 'Docker Compose', link: '/container/03_docker_compose' },
            { text: 'Kubernetes 基础', link: '/container/04_kubernetes_basics' },
            { text: 'K8s 示例', link: '/container/05_k8s_examples' },
            { text: '实战脚本', link: '/container/06_practical_scripts' },
            { text: '速查表', link: '/container/07_cheatsheet' }
          ]
        }
      ],

      '/architecture/': [
        {
          text: '系统架构',
          items: [
            { text: '简介', link: '/architecture/' }
          ]
        },
        {
          text: '系统设计',
          items: [
            { text: '设计原则', link: '/architecture/01_system_design/01_design_principles' },
            { text: '架构模式', link: '/architecture/01_system_design/02_architecture_patterns' },
            { text: '容量规划', link: '/architecture/01_system_design/03_capacity_planning' }
          ]
        },
        {
          text: '分布式系统',
          items: [
            { text: 'CAP/BASE 理论', link: '/architecture/02_distributed/01_cap_base' },
            { text: '分布式锁', link: '/architecture/02_distributed/02_distributed_lock' },
            { text: '分布式事务', link: '/architecture/02_distributed/03_distributed_transaction' }
          ]
        },
        {
          text: '高可用',
          items: [
            { text: '高可用原则', link: '/architecture/03_high_availability/01_ha_principles' },
            { text: '限流', link: '/architecture/03_high_availability/02_rate_limiting' },
            { text: '故障转移', link: '/architecture/03_high_availability/03_failover' },
            { text: '容灾', link: '/architecture/03_high_availability/04_disaster_recovery' }
          ]
        },
        {
          text: '高性能',
          items: [
            { text: '性能指标', link: '/architecture/04_high_performance/01_performance_metrics' },
            { text: '并发', link: '/architecture/04_high_performance/02_concurrency' },
            { text: 'I/O 优化', link: '/architecture/04_high_performance/03_io_optimization' },
            { text: '池化模式', link: '/architecture/04_high_performance/04_pool_pattern' }
          ]
        },
        {
          text: '微服务',
          items: [
            { text: '服务拆分', link: '/architecture/05_microservices/01_service_splitting' },
            { text: 'API 设计', link: '/architecture/05_microservices/02_api_design' },
            { text: '服务治理', link: '/architecture/05_microservices/03_service_governance' },
            { text: 'Service Mesh', link: '/architecture/05_microservices/04_service_mesh' }
          ]
        },
        {
          text: '数据库架构',
          items: [
            { text: 'MySQL 优化', link: '/architecture/06_database_architecture/01_mysql_optimization' },
            { text: '分库分表', link: '/architecture/06_database_architecture/02_sharding' },
            { text: '读写分离', link: '/architecture/06_database_architecture/03_read_write_splitting' }
          ]
        },
        {
          text: '缓存架构',
          items: [
            { text: '缓存模式', link: '/architecture/07_cache_architecture/01_cache_patterns' }
          ]
        },
        {
          text: '消息队列',
          items: [
            { text: 'MQ 模式', link: '/architecture/08_message_queue/01_mq_patterns' }
          ]
        },
        {
          text: '安全',
          items: [
            { text: '安全基础', link: '/architecture/09_security/01_security_fundamentals' }
          ]
        },
        {
          text: '可观测性',
          items: [
            { text: '可观测性', link: '/architecture/10_observability/01_observability' }
          ]
        }
      ],

      '/cloud-native/': [
        {
          text: '云原生',
          items: [
            { text: '简介', link: '/cloud-native/' },
            { text: '云计算基础', link: '/cloud-native/01_cloud_fundamentals' },
            { text: 'Serverless', link: '/cloud-native/02_serverless' },
            { text: '云原生模式', link: '/cloud-native/03_cloud_patterns' },
            { text: '多云架构', link: '/cloud-native/04_multi_cloud' },
            { text: '成本优化', link: '/cloud-native/05_cost_optimization' }
          ]
        }
      ],

      '/devops/': [
        {
          text: 'DevOps',
          items: [
            { text: '简介', link: '/devops/' },
            { text: 'CI/CD 流水线', link: '/devops/01_cicd_pipeline' },
            { text: 'GitOps', link: '/devops/02_gitops' },
            { text: '基础设施即代码', link: '/devops/03_infrastructure_as_code' },
            { text: '部署策略', link: '/devops/04_deployment_strategies' },
            { text: '发布管理', link: '/devops/05_release_management' }
          ]
        }
      ],

      '/api-gateway/': [
        {
          text: 'API 网关',
          items: [
            { text: '简介', link: '/api-gateway/' },
            { text: '网关设计', link: '/api-gateway/01_gateway_design' },
            { text: '路由策略', link: '/api-gateway/02_routing_strategies' },
            { text: '认证授权', link: '/api-gateway/03_authentication' },
            { text: '网关对比', link: '/api-gateway/04_gateway_comparison' }
          ]
        }
      ],

      '/ddd/': [
        {
          text: '领域驱动设计',
          items: [
            { text: '简介', link: '/ddd/' },
            { text: '战略设计', link: '/ddd/01_strategic_design' },
            { text: '战术设计', link: '/ddd/02_tactical_design' },
            { text: '事件风暴', link: '/ddd/03_event_storming' },
            { text: 'DDD 实践', link: '/ddd/04_ddd_in_practice' }
          ]
        }
      ],

      '/performance/': [
        {
          text: '性能优化',
          items: [
            { text: '简介', link: '/performance/' },
            { text: '压力测试', link: '/performance/01_load_testing' },
            { text: '性能分析', link: '/performance/02_profiling' },
            { text: '瓶颈分析', link: '/performance/03_bottleneck_analysis' },
            { text: '优化案例', link: '/performance/04_optimization_cases' }
          ]
        }
      ],

      '/governance/': [
        {
          text: '技术治理',
          items: [
            { text: '简介', link: '/governance/' },
            { text: '技术债务', link: '/governance/01_technical_debt' },
            { text: '架构评审', link: '/governance/02_architecture_review' },
            { text: 'ADR', link: '/governance/03_adr' },
            { text: '技术标准', link: '/governance/04_standards' }
          ]
        }
      ],

      '/data-architecture/': [
        {
          text: '数据架构',
          items: [
            { text: '简介', link: '/data-architecture/' },
            { text: '数据建模', link: '/data-architecture/01_data_modeling' },
            { text: '数据治理', link: '/data-architecture/02_data_governance' },
            { text: '数据管道', link: '/data-architecture/03_data_pipeline' },
            { text: '数据湖', link: '/data-architecture/04_data_lake' }
          ]
        }
      ],

      '/security/': [
        {
          text: '安全',
          items: [
            { text: '简介', link: '/security/' },
            { text: '零信任架构', link: '/security/01_zero_trust' },
            { text: '密钥管理', link: '/security/02_secret_management' },
            { text: '合规', link: '/security/03_compliance' },
            { text: '安全测试', link: '/security/04_security_testing' }
          ]
        }
      ],

      '/bigdata/': [
        {
          text: '大数据',
          items: [
            { text: '简介', link: '/bigdata/' },
            { text: '批处理 (Spark)', link: '/bigdata/01_batch_processing' },
            { text: '流处理 (Flink)', link: '/bigdata/02_stream_processing' },
            { text: '实时数仓', link: '/bigdata/03_realtime_warehouse' },
            { text: 'OLAP', link: '/bigdata/04_olap' }
          ]
        }
      ],

      '/ai-architecture/': [
        {
          text: 'AI 架构',
          items: [
            { text: '简介', link: '/ai-architecture/' },
            { text: 'ML 流水线', link: '/ai-architecture/01_ml_pipeline' },
            { text: '模型服务', link: '/ai-architecture/02_model_serving' },
            { text: '特征平台', link: '/ai-architecture/03_feature_store' }
          ]
        }
      ],

      '/soft-skills/': [
        {
          text: '软技能',
          items: [
            { text: '简介', link: '/soft-skills/' },
            { text: '技术决策', link: '/soft-skills/01_technical_decisions' },
            { text: '架构文档', link: '/soft-skills/02_architecture_documentation' },
            { text: '沟通协作', link: '/soft-skills/03_communication' }
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
    hostname: 'https://t.tecfav.com'
  }
})
