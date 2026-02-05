# Elasticsearch 查询 DSL

## 一、查询基础

### 查询结构

```json
GET /my_index/_search
{
  "query": { ... },           // 查询条件
  "from": 0,                  // 起始位置
  "size": 10,                 // 返回数量
  "sort": [ ... ],            // 排序
  "_source": [ ... ],         // 返回字段
  "highlight": { ... },       // 高亮
  "aggs": { ... }             // 聚合
}
```

### 查询上下文 vs 过滤上下文

```json
// 查询上下文 (Query Context)
// - 计算相关性评分 (_score)
// - 不缓存结果
// - 适用于全文搜索

// 过滤上下文 (Filter Context)
// - 不计算评分
// - 结果可缓存
// - 适用于精确匹配、范围过滤

// 组合使用
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": "elasticsearch" } }     // 查询上下文
      ],
      "filter": [
        { "term": { "status": "published" } },        // 过滤上下文
        { "range": { "date": { "gte": "2024-01-01" } } }
      ]
    }
  }
}
```

## 二、全文查询

### match 查询

```json
// 基本 match（会分词）
{
  "query": {
    "match": {
      "title": "elasticsearch 入门"
    }
  }
}

// 带参数的 match
{
  "query": {
    "match": {
      "title": {
        "query": "elasticsearch 入门",
        "operator": "and",              // 默认 or
        "minimum_should_match": "75%",  // 最少匹配比例
        "fuzziness": "AUTO",            // 模糊匹配
        "prefix_length": 2,             // 模糊匹配前缀长度
        "analyzer": "ik_smart"          // 指定分析器
      }
    }
  }
}
```

### match_phrase 查询

```json
// 短语匹配（词序和位置必须一致）
{
  "query": {
    "match_phrase": {
      "content": {
        "query": "quick brown fox",
        "slop": 2    // 允许的词间距
      }
    }
  }
}

// match_phrase_prefix（前缀短语匹配，用于自动补全）
{
  "query": {
    "match_phrase_prefix": {
      "title": {
        "query": "elastic sea",
        "max_expansions": 50    // 最大扩展数
      }
    }
  }
}
```

### multi_match 查询

```json
// 多字段匹配
{
  "query": {
    "multi_match": {
      "query": "elasticsearch guide",
      "fields": ["title^3", "content", "tags"],    // title 权重 3 倍
      "type": "best_fields"    // 类型
    }
  }
}

// type 选项：
// best_fields: 最佳字段匹配（默认）
// most_fields: 多字段匹配，分数相加
// cross_fields: 跨字段匹配，词条可分布在不同字段
// phrase: 短语匹配
// phrase_prefix: 短语前缀匹配
```

### query_string 查询

```json
// 支持 Lucene 查询语法
{
  "query": {
    "query_string": {
      "query": "(title:elasticsearch OR content:search) AND status:published",
      "default_field": "content",
      "default_operator": "AND"
    }
  }
}

// simple_query_string（更安全，不会因语法错误而失败）
{
  "query": {
    "simple_query_string": {
      "query": "elasticsearch +guide -beginner",
      "fields": ["title", "content"],
      "default_operator": "and"
    }
  }
}
// 语法：+ 必须包含，- 必须不包含，| 或，"" 短语
```

## 三、精确查询

### term 查询

```json
// 精确匹配（不分词）
{
  "query": {
    "term": {
      "status": {
        "value": "published",
        "boost": 2.0    // 权重
      }
    }
  }
}

// 多值匹配
{
  "query": {
    "terms": {
      "status": ["published", "draft"]
    }
  }
}

// 注意：对 text 字段使用 term 通常不会得到预期结果
// 应该使用 keyword 子字段
{
  "query": {
    "term": {
      "title.keyword": "Elasticsearch Guide"
    }
  }
}
```

### range 查询

```json
{
  "query": {
    "range": {
      "publish_date": {
        "gte": "2024-01-01",
        "lte": "2024-12-31",
        "format": "yyyy-MM-dd",
        "time_zone": "+08:00"
      }
    }
  }
}

// 数值范围
{
  "query": {
    "range": {
      "price": {
        "gte": 100,
        "lt": 500
      }
    }
  }
}

// 相对日期
{
  "query": {
    "range": {
      "publish_date": {
        "gte": "now-1M/d",     // 一个月前，向下取整到天
        "lte": "now/d"         // 现在，向下取整到天
      }
    }
  }
}
```

### exists 查询

```json
// 字段存在且非空
{
  "query": {
    "exists": {
      "field": "email"
    }
  }
}

// 字段不存在
{
  "query": {
    "bool": {
      "must_not": {
        "exists": {
          "field": "email"
        }
      }
    }
  }
}
```

### prefix / wildcard / regexp 查询

```json
// 前缀匹配
{
  "query": {
    "prefix": {
      "title.keyword": {
        "value": "Elastic"
      }
    }
  }
}

// 通配符匹配
{
  "query": {
    "wildcard": {
      "title.keyword": {
        "value": "Elastic*ch"    // * 多个字符，? 单个字符
      }
    }
  }
}

// 正则匹配
{
  "query": {
    "regexp": {
      "title.keyword": {
        "value": "elastic.*",
        "flags": "ALL"
      }
    }
  }
}

// 注意：这些查询性能较差，避免以通配符开头
```

### fuzzy 查询

```json
// 模糊匹配（容错）
{
  "query": {
    "fuzzy": {
      "title": {
        "value": "elasticsearh",    // 拼写错误
        "fuzziness": "AUTO",        // 自动根据长度决定
        "prefix_length": 2,         // 前 2 个字符必须精确匹配
        "max_expansions": 50
      }
    }
  }
}

// fuzziness 选项：
// 0, 1, 2: 允许的编辑距离
// AUTO: 根据词长自动决定 (0-2: 0, 3-5: 1, >5: 2)
```

### ids 查询

```json
{
  "query": {
    "ids": {
      "values": ["1", "2", "3"]
    }
  }
}
```

## 四、复合查询

### bool 查询

```json
{
  "query": {
    "bool": {
      "must": [                          // AND，计算评分
        { "match": { "title": "elasticsearch" } }
      ],
      "must_not": [                      // NOT，不计算评分
        { "term": { "status": "deleted" } }
      ],
      "should": [                        // OR，计算评分
        { "match": { "content": "guide" } },
        { "match": { "content": "tutorial" } }
      ],
      "filter": [                        // AND，不计算评分（可缓存）
        { "term": { "status": "published" } },
        { "range": { "date": { "gte": "2024-01-01" } } }
      ],
      "minimum_should_match": 1          // should 至少匹配数量
    }
  }
}
```

### boosting 查询

```json
// 降低某些文档的评分而不排除它们
{
  "query": {
    "boosting": {
      "positive": {
        "match": { "title": "elasticsearch" }
      },
      "negative": {
        "term": { "category": "outdated" }
      },
      "negative_boost": 0.5    // 负面匹配的文档评分乘以此值
    }
  }
}
```

### constant_score 查询

```json
// 所有匹配文档获得相同评分
{
  "query": {
    "constant_score": {
      "filter": {
        "term": { "status": "published" }
      },
      "boost": 1.2
    }
  }
}
```

### dis_max 查询

```json
// 取最高评分（而非相加）
{
  "query": {
    "dis_max": {
      "queries": [
        { "match": { "title": "elasticsearch" } },
        { "match": { "content": "elasticsearch" } }
      ],
      "tie_breaker": 0.3    // 其他查询评分的权重
    }
  }
}
```

### function_score 查询

```json
// 自定义评分函数
{
  "query": {
    "function_score": {
      "query": {
        "match": { "title": "elasticsearch" }
      },
      "functions": [
        {
          "filter": { "term": { "featured": true } },
          "weight": 2
        },
        {
          "field_value_factor": {
            "field": "views",
            "factor": 1.2,
            "modifier": "log1p",    // log, log1p, log2p, ln, ln1p, ln2p, sqrt, none
            "missing": 1
          }
        },
        {
          "gauss": {               // 衰减函数
            "publish_date": {
              "origin": "now",
              "scale": "30d",
              "decay": 0.5
            }
          }
        },
        {
          "script_score": {
            "script": {
              "source": "Math.log(2 + doc['likes'].value)"
            }
          }
        },
        {
          "random_score": {
            "seed": 12345,
            "field": "_seq_no"
          }
        }
      ],
      "score_mode": "multiply",    // multiply, sum, avg, first, max, min
      "boost_mode": "multiply",    // multiply, replace, sum, avg, max, min
      "max_boost": 10
    }
  }
}
```

## 五、嵌套查询

### nested 查询

```json
// 索引定义
PUT /orders
{
  "mappings": {
    "properties": {
      "items": {
        "type": "nested",
        "properties": {
          "product": { "type": "keyword" },
          "quantity": { "type": "integer" },
          "price": { "type": "float" }
        }
      }
    }
  }
}

// 嵌套查询
{
  "query": {
    "nested": {
      "path": "items",
      "query": {
        "bool": {
          "must": [
            { "term": { "items.product": "iPhone" } },
            { "range": { "items.quantity": { "gte": 2 } } }
          ]
        }
      },
      "score_mode": "avg",    // avg, sum, min, max, none
      "inner_hits": {         // 返回匹配的嵌套文档
        "size": 3,
        "highlight": {
          "fields": {
            "items.product": {}
          }
        }
      }
    }
  }
}
```

### has_child / has_parent 查询

```json
// 父子关系定义
PUT /my_index
{
  "mappings": {
    "properties": {
      "my_join_field": {
        "type": "join",
        "relations": {
          "question": "answer"    // 问题是父，答案是子
        }
      }
    }
  }
}

// has_child: 查询有符合条件子文档的父文档
{
  "query": {
    "has_child": {
      "type": "answer",
      "query": {
        "match": { "content": "elasticsearch" }
      },
      "min_children": 1,
      "max_children": 10,
      "score_mode": "avg"
    }
  }
}

// has_parent: 查询有符合条件父文档的子文档
{
  "query": {
    "has_parent": {
      "parent_type": "question",
      "query": {
        "match": { "title": "elasticsearch" }
      },
      "score": true
    }
  }
}
```

## 六、地理位置查询

### geo_bounding_box

```json
// 矩形范围查询
{
  "query": {
    "geo_bounding_box": {
      "location": {
        "top_left": {
          "lat": 40.73,
          "lon": -74.1
        },
        "bottom_right": {
          "lat": 40.01,
          "lon": -71.12
        }
      }
    }
  }
}
```

### geo_distance

```json
// 距离范围查询
{
  "query": {
    "geo_distance": {
      "distance": "5km",
      "location": {
        "lat": 40.715,
        "lon": -73.988
      }
    }
  }
}

// 带排序
{
  "query": {
    "geo_distance": {
      "distance": "10km",
      "location": [116.404, 39.915]
    }
  },
  "sort": [
    {
      "_geo_distance": {
        "location": [116.404, 39.915],
        "order": "asc",
        "unit": "km"
      }
    }
  ]
}
```

### geo_polygon

```json
// 多边形范围查询
{
  "query": {
    "geo_polygon": {
      "location": {
        "points": [
          { "lat": 40.73, "lon": -74.1 },
          { "lat": 40.01, "lon": -71.12 },
          { "lat": 50.56, "lon": -90.58 }
        ]
      }
    }
  }
}
```

## 七、排序与分页

### 排序

```json
{
  "query": { "match_all": {} },
  "sort": [
    { "date": { "order": "desc" } },
    { "title.keyword": { "order": "asc" } },
    { "_score": { "order": "desc" } },        // 按评分
    {
      "price": {
        "order": "asc",
        "missing": "_last",                   // 缺失值排最后
        "unmapped_type": "float"              // 字段不存在时的类型
      }
    }
  ]
}

// 嵌套字段排序
{
  "sort": [
    {
      "items.price": {
        "order": "asc",
        "nested": {
          "path": "items",
          "filter": {
            "term": { "items.category": "electronics" }
          }
        },
        "mode": "avg"    // min, max, sum, avg, median
      }
    }
  ]
}
```

### 分页

```json
// from + size（浅分页，不建议超过 10000）
{
  "from": 0,
  "size": 10,
  "query": { "match_all": {} }
}

// search_after（深度分页，推荐）
// 第一次查询
{
  "size": 10,
  "query": { "match_all": {} },
  "sort": [
    { "date": "desc" },
    { "_id": "asc" }
  ]
}

// 后续查询
{
  "size": 10,
  "query": { "match_all": {} },
  "sort": [
    { "date": "desc" },
    { "_id": "asc" }
  ],
  "search_after": ["2024-01-01T00:00:00", "abc123"]    // 上一页最后一条的排序值
}

// scroll API（大量数据导出）
POST /my_index/_search?scroll=1m
{
  "size": 1000,
  "query": { "match_all": {} }
}

// 获取下一批
POST /_search/scroll
{
  "scroll": "1m",
  "scroll_id": "DXF1ZXJ5QW5..."
}
```

## 八、高亮显示

```json
{
  "query": {
    "match": { "content": "elasticsearch" }
  },
  "highlight": {
    "pre_tags": ["<em>"],
    "post_tags": ["</em>"],
    "fields": {
      "content": {
        "type": "unified",              // unified, plain, fvh
        "fragment_size": 150,           // 片段大小
        "number_of_fragments": 3,       // 片段数量
        "no_match_size": 150,           // 无匹配时返回的字符数
        "fragmenter": "span"            // span, simple
      },
      "title": {
        "number_of_fragments": 0        // 0 表示返回整个字段
      }
    },
    "encoder": "html",                  // default, html
    "require_field_match": false        // 是否要求字段匹配
  }
}
```

## 九、Source 过滤

```json
// 禁用 _source
{
  "_source": false,
  "query": { "match_all": {} }
}

// 包含字段
{
  "_source": ["title", "author"],
  "query": { "match_all": {} }
}

// 包含/排除
{
  "_source": {
    "includes": ["title", "author.*"],
    "excludes": ["*.secret"]
  },
  "query": { "match_all": {} }
}
```

## 十、脚本查询

```json
// 脚本过滤
{
  "query": {
    "bool": {
      "filter": {
        "script": {
          "script": {
            "source": "doc['views'].value > params.min_views",
            "params": {
              "min_views": 100
            }
          }
        }
      }
    }
  }
}

// 脚本字段
{
  "query": { "match_all": {} },
  "script_fields": {
    "total_amount": {
      "script": {
        "source": "doc['price'].value * doc['quantity'].value"
      }
    }
  }
}

// 脚本排序
{
  "query": { "match_all": {} },
  "sort": {
    "_script": {
      "type": "number",
      "script": {
        "source": "doc['likes'].value + doc['shares'].value * 2"
      },
      "order": "desc"
    }
  }
}
```
