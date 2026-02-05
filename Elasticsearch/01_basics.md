# Elasticsearch 基础教程

## 一、概述

Elasticsearch 是一个基于 Lucene 的分布式搜索和分析引擎，主要特点：

- **分布式**：自动分片、副本、负载均衡
- **实时搜索**：近实时（NRT）索引和搜索
- **RESTful API**：通过 HTTP/JSON 进行交互
- **Schema Free**：动态映射，无需预定义结构
- **全文搜索**：强大的文本分析和搜索能力

### 核心概念

| 概念 | 说明 | 类比 RDBMS |
|------|------|-----------|
| Index | 索引，文档的集合 | Database |
| Type | 类型（7.x 已废弃） | Table |
| Document | 文档，JSON 格式的数据 | Row |
| Field | 字段 | Column |
| Mapping | 映射，定义字段类型 | Schema |
| Shard | 分片，数据的水平分割 | Partition |
| Replica | 副本，分片的复制 | Replication |

### 架构图

```
┌─────────────────────────────────────────────────────────┐
│                    Elasticsearch Cluster                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Node 1    │  │   Node 2    │  │   Node 3    │     │
│  │  (Master)   │  │   (Data)    │  │   (Data)    │     │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │     │
│  │ │ Shard 0 │ │  │ │ Shard 1 │ │  │ │ Shard 2 │ │     │
│  │ │(Primary)│ │  │ │(Primary)│ │  │ │(Primary)│ │     │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │     │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │     │
│  │ │ Shard 1 │ │  │ │ Shard 2 │ │  │ │ Shard 0 │ │     │
│  │ │(Replica)│ │  │ │(Replica)│ │  │ │(Replica)│ │     │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## 二、安装与配置

### Docker 快速启动

```bash
# 单节点
docker run -d --name elasticsearch \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  elasticsearch:8.11.0

# 验证
curl http://localhost:9200
```

### 主要配置（elasticsearch.yml）

```yaml
# 集群名称
cluster.name: my-cluster

# 节点名称
node.name: node-1

# 数据和日志路径
path.data: /var/lib/elasticsearch
path.logs: /var/log/elasticsearch

# 网络
network.host: 0.0.0.0
http.port: 9200
transport.port: 9300

# 集群发现
discovery.seed_hosts: ["host1", "host2", "host3"]
cluster.initial_master_nodes: ["node-1", "node-2", "node-3"]

# 内存锁定（生产环境建议开启）
bootstrap.memory_lock: true

# 跨域
http.cors.enabled: true
http.cors.allow-origin: "*"
```

## 三、索引操作

### 创建索引

```json
PUT /my_index
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "refresh_interval": "1s",
    "analysis": {
      "analyzer": {
        "my_analyzer": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": ["lowercase", "stop"]
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "title": {
        "type": "text",
        "analyzer": "my_analyzer"
      },
      "content": {
        "type": "text"
      },
      "author": {
        "type": "keyword"
      },
      "publish_date": {
        "type": "date",
        "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"
      },
      "views": {
        "type": "integer"
      },
      "tags": {
        "type": "keyword"
      }
    }
  }
}
```

### 查看索引

```json
// 查看所有索引
GET /_cat/indices?v

// 查看索引详情
GET /my_index

// 查看映射
GET /my_index/_mapping

// 查看设置
GET /my_index/_settings
```

### 修改索引

```json
// 修改设置（部分设置不可动态修改）
PUT /my_index/_settings
{
  "number_of_replicas": 2,
  "refresh_interval": "30s"
}

// 添加字段映射
PUT /my_index/_mapping
{
  "properties": {
    "new_field": {
      "type": "text"
    }
  }
}

// 注意：已有字段的类型不能修改，需要重建索引
```

### 删除索引

```json
DELETE /my_index

// 删除多个
DELETE /index1,index2

// 通配符删除（谨慎使用）
DELETE /log-*
```

### 索引别名

```json
// 创建别名
POST /_aliases
{
  "actions": [
    { "add": { "index": "my_index_v1", "alias": "my_index" } }
  ]
}

// 切换别名（零停机切换）
POST /_aliases
{
  "actions": [
    { "remove": { "index": "my_index_v1", "alias": "my_index" } },
    { "add": { "index": "my_index_v2", "alias": "my_index" } }
  ]
}

// 查看别名
GET /_alias/my_index
GET /my_index_v1/_alias
```

## 四、文档操作

### 创建/更新文档

```json
// 指定 ID 创建
PUT /my_index/_doc/1
{
  "title": "Elasticsearch 入门",
  "content": "这是一篇关于 ES 的教程...",
  "author": "张三",
  "publish_date": "2024-01-01",
  "views": 100,
  "tags": ["elasticsearch", "搜索", "教程"]
}

// 自动生成 ID
POST /my_index/_doc
{
  "title": "另一篇文章",
  "content": "..."
}

// 仅创建（如果已存在则失败）
PUT /my_index/_doc/1?op_type=create
{
  "title": "..."
}
// 或
PUT /my_index/_create/1
{
  "title": "..."
}
```

### 获取文档

```json
// 获取单个文档
GET /my_index/_doc/1

// 只获取 _source
GET /my_index/_source/1

// 获取部分字段
GET /my_index/_doc/1?_source=title,author

// 检查文档是否存在
HEAD /my_index/_doc/1

// 批量获取
GET /my_index/_mget
{
  "ids": ["1", "2", "3"]
}
// 或
POST /_mget
{
  "docs": [
    { "_index": "my_index", "_id": "1" },
    { "_index": "my_index", "_id": "2", "_source": ["title"] }
  ]
}
```

### 更新文档

```json
// 部分更新
POST /my_index/_update/1
{
  "doc": {
    "views": 150,
    "tags": ["elasticsearch", "搜索", "教程", "更新"]
  }
}

// 使用脚本更新
POST /my_index/_update/1
{
  "script": {
    "source": "ctx._source.views += params.increment",
    "params": {
      "increment": 10
    }
  }
}

// upsert（不存在则创建）
POST /my_index/_update/1
{
  "doc": {
    "views": 100
  },
  "doc_as_upsert": true
}

// 或使用脚本 upsert
POST /my_index/_update/1
{
  "script": {
    "source": "ctx._source.views += 1"
  },
  "upsert": {
    "title": "新文档",
    "views": 1
  }
}
```

### 删除文档

```json
// 删除单个
DELETE /my_index/_doc/1

// 条件删除
POST /my_index/_delete_by_query
{
  "query": {
    "match": {
      "author": "张三"
    }
  }
}
```

### 批量操作

```json
POST /_bulk
{ "index": { "_index": "my_index", "_id": "1" } }
{ "title": "文档1", "content": "内容1" }
{ "index": { "_index": "my_index", "_id": "2" } }
{ "title": "文档2", "content": "内容2" }
{ "update": { "_index": "my_index", "_id": "1" } }
{ "doc": { "views": 100 } }
{ "delete": { "_index": "my_index", "_id": "3" } }

// 注意：每行必须以换行符结尾，包括最后一行
```

## 五、字段类型

### 核心类型

```json
{
  "mappings": {
    "properties": {
      // 字符串
      "title": { "type": "text" },           // 全文搜索
      "status": { "type": "keyword" },       // 精确匹配

      // 数值
      "age": { "type": "integer" },
      "price": { "type": "float" },
      "count": { "type": "long" },
      "score": { "type": "double" },
      "rank": { "type": "short" },
      "flag": { "type": "byte" },
      "ratio": { "type": "half_float" },
      "amount": { "type": "scaled_float", "scaling_factor": 100 },

      // 日期
      "created_at": {
        "type": "date",
        "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"
      },

      // 布尔
      "is_active": { "type": "boolean" },

      // 二进制
      "file": { "type": "binary" },

      // 范围
      "age_range": { "type": "integer_range" },
      "date_range": { "type": "date_range" }
    }
  }
}
```

### 复杂类型

```json
{
  "mappings": {
    "properties": {
      // 对象
      "author": {
        "type": "object",
        "properties": {
          "name": { "type": "text" },
          "email": { "type": "keyword" }
        }
      },

      // 嵌套（保持数组内对象的独立性）
      "comments": {
        "type": "nested",
        "properties": {
          "user": { "type": "keyword" },
          "content": { "type": "text" },
          "date": { "type": "date" }
        }
      },

      // 地理位置
      "location": { "type": "geo_point" },
      "area": { "type": "geo_shape" },

      // IP 地址
      "ip": { "type": "ip" },

      // 自动补全
      "suggest": { "type": "completion" },

      // 向量（用于语义搜索）
      "embedding": {
        "type": "dense_vector",
        "dims": 768
      }
    }
  }
}
```

### text vs keyword

```json
// text: 会被分词，用于全文搜索
// keyword: 不分词，用于精确匹配、聚合、排序

{
  "properties": {
    "title": {
      "type": "text",
      "fields": {
        "keyword": {           // 多字段
          "type": "keyword",
          "ignore_above": 256  // 超过 256 字符不索引
        }
      }
    }
  }
}

// 使用
// 全文搜索: title
// 精确匹配: title.keyword
// 聚合/排序: title.keyword
```

## 六、分析器

### 分析器组成

```
文本 → Character Filters → Tokenizer → Token Filters → 词条
        (字符过滤器)        (分词器)      (词条过滤器)
```

### 测试分析器

```json
// 测试内置分析器
POST /_analyze
{
  "analyzer": "standard",
  "text": "Hello World, 你好世界!"
}

// 测试自定义分析过程
POST /_analyze
{
  "tokenizer": "standard",
  "filter": ["lowercase", "stop"],
  "text": "The Quick Brown Fox"
}

// 测试索引的分析器
POST /my_index/_analyze
{
  "field": "title",
  "text": "测试文本"
}
```

### 内置分析器

| 分析器 | 说明 |
|--------|------|
| standard | 默认，按词切分，转小写 |
| simple | 非字母切分，转小写 |
| whitespace | 空白符切分，不转小写 |
| stop | standard + 去停用词 |
| keyword | 不分词，整体作为一个词条 |
| pattern | 正则切分 |
| language | 特定语言分析器（english, chinese...） |

### 自定义分析器

```json
PUT /my_index
{
  "settings": {
    "analysis": {
      "char_filter": {
        "my_char_filter": {
          "type": "mapping",
          "mappings": ["& => and", "| => or"]
        }
      },
      "tokenizer": {
        "my_tokenizer": {
          "type": "pattern",
          "pattern": "[\\s,;]+"
        }
      },
      "filter": {
        "my_stopwords": {
          "type": "stop",
          "stopwords": ["the", "a", "an", "is"]
        }
      },
      "analyzer": {
        "my_analyzer": {
          "type": "custom",
          "char_filter": ["my_char_filter"],
          "tokenizer": "my_tokenizer",
          "filter": ["lowercase", "my_stopwords"]
        }
      }
    }
  }
}
```

### 中文分析器（IK）

```json
// 安装 IK 分词器后
PUT /chinese_index
{
  "mappings": {
    "properties": {
      "content": {
        "type": "text",
        "analyzer": "ik_max_word",        // 索引时最细粒度分词
        "search_analyzer": "ik_smart"     // 搜索时智能分词
      }
    }
  }
}

// 测试
POST /_analyze
{
  "analyzer": "ik_max_word",
  "text": "中华人民共和国国歌"
}
// 结果: 中华人民共和国, 中华人民, 中华, 华人, 人民共和国, 人民, 共和国, 共和, 国歌
```

## 七、集群管理

### 集群健康

```json
// 集群健康状态
GET /_cluster/health
// green: 所有分片正常
// yellow: 主分片正常，副本分片异常
// red: 有主分片异常

// 详细信息
GET /_cluster/health?level=indices
GET /_cluster/health?level=shards

// 等待状态变化
GET /_cluster/health?wait_for_status=green&timeout=50s
```

### 节点信息

```json
// 节点列表
GET /_cat/nodes?v

// 节点详情
GET /_nodes
GET /_nodes/stats
GET /_nodes/hot_threads
```

### 分片管理

```json
// 分片分布
GET /_cat/shards?v
GET /_cat/shards/my_index?v

// 分片分配解释
GET /_cluster/allocation/explain
{
  "index": "my_index",
  "shard": 0,
  "primary": true
}

// 手动移动分片
POST /_cluster/reroute
{
  "commands": [
    {
      "move": {
        "index": "my_index",
        "shard": 0,
        "from_node": "node1",
        "to_node": "node2"
      }
    }
  ]
}
```

### 索引模板

```json
// 创建索引模板
PUT /_index_template/logs_template
{
  "index_patterns": ["logs-*"],
  "priority": 100,
  "template": {
    "settings": {
      "number_of_shards": 3,
      "number_of_replicas": 1
    },
    "mappings": {
      "properties": {
        "timestamp": { "type": "date" },
        "message": { "type": "text" },
        "level": { "type": "keyword" }
      }
    }
  }
}

// 查看模板
GET /_index_template/logs_template

// 删除模板
DELETE /_index_template/logs_template
```
