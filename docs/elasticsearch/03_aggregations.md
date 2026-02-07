# Elasticsearch 聚合分析

## 一、聚合概述

聚合（Aggregation）用于数据分析和统计，类似 SQL 中的 GROUP BY。

### 聚合类型

| 类型 | 说明 | 示例 |
|------|------|------|
| Bucket | 桶聚合，分组 | terms, range, histogram |
| Metric | 指标聚合，计算 | avg, sum, min, max, stats |
| Pipeline | 管道聚合，二次聚合 | derivative, moving_avg |
| Matrix | 矩阵聚合，多字段统计 | matrix_stats |

### 聚合结构

```json
{
  "size": 0,                    // 不需要文档，只要聚合结果
  "query": { ... },             // 可选，先过滤再聚合
  "aggs": {
    "my_agg_name": {            // 自定义聚合名称
      "agg_type": {             // 聚合类型
        "field": "field_name",
        // 其他参数
      },
      "aggs": {                 // 子聚合
        "sub_agg_name": { ... }
      }
    }
  }
}
```

## 二、指标聚合（Metric）

### 基本统计

```json
// 平均值
{
  "size": 0,
  "aggs": {
    "avg_price": {
      "avg": {
        "field": "price",
        "missing": 0         // 缺失值处理
      }
    }
  }
}

// 求和
{
  "aggs": {
    "total_sales": {
      "sum": { "field": "amount" }
    }
  }
}

// 最大/最小值
{
  "aggs": {
    "max_price": { "max": { "field": "price" } },
    "min_price": { "min": { "field": "price" } }
  }
}

// 计数（非空值）
{
  "aggs": {
    "price_count": {
      "value_count": { "field": "price" }
    }
  }
}

// 去重计数
{
  "aggs": {
    "unique_users": {
      "cardinality": {
        "field": "user_id",
        "precision_threshold": 3000    // 精度阈值
      }
    }
  }
}
```

### 综合统计

```json
// stats: 返回 count, min, max, avg, sum
{
  "aggs": {
    "price_stats": {
      "stats": { "field": "price" }
    }
  }
}

// extended_stats: 额外返回方差、标准差等
{
  "aggs": {
    "price_extended_stats": {
      "extended_stats": { "field": "price" }
    }
  }
}
// 返回: count, min, max, avg, sum,
//       sum_of_squares, variance, std_deviation,
//       std_deviation_bounds.upper/lower
```

### 百分位数

```json
// percentiles: 计算百分位数
{
  "aggs": {
    "price_percentiles": {
      "percentiles": {
        "field": "price",
        "percents": [25, 50, 75, 95, 99]
      }
    }
  }
}

// percentile_ranks: 计算值对应的百分位排名
{
  "aggs": {
    "price_ranks": {
      "percentile_ranks": {
        "field": "price",
        "values": [100, 200, 500]
      }
    }
  }
}
```

### Top Hits

```json
// 获取每个桶的热门文档
{
  "aggs": {
    "by_category": {
      "terms": { "field": "category" },
      "aggs": {
        "top_products": {
          "top_hits": {
            "size": 3,
            "sort": [{ "sales": "desc" }],
            "_source": ["name", "price", "sales"]
          }
        }
      }
    }
  }
}
```

### 脚本聚合

```json
{
  "aggs": {
    "total_revenue": {
      "sum": {
        "script": {
          "source": "doc['price'].value * doc['quantity'].value"
        }
      }
    }
  }
}
```

## 三、桶聚合（Bucket）

### Terms 聚合

```json
// 按字段值分组
{
  "size": 0,
  "aggs": {
    "by_category": {
      "terms": {
        "field": "category",
        "size": 10,                     // 返回桶数量
        "min_doc_count": 1,             // 最小文档数
        "order": { "_count": "desc" },  // 排序
        "missing": "未分类"              // 缺失值处理
      }
    }
  }
}

// 多级分组
{
  "aggs": {
    "by_category": {
      "terms": { "field": "category" },
      "aggs": {
        "by_brand": {
          "terms": { "field": "brand" },
          "aggs": {
            "avg_price": {
              "avg": { "field": "price" }
            }
          }
        }
      }
    }
  }
}

// 排序方式
{
  "aggs": {
    "by_category": {
      "terms": {
        "field": "category",
        "order": [
          { "avg_price": "desc" },     // 按子聚合排序
          { "_count": "desc" }          // 按文档数排序
        ]
      },
      "aggs": {
        "avg_price": {
          "avg": { "field": "price" }
        }
      }
    }
  }
}
```

### Range 聚合

```json
// 数值范围
{
  "aggs": {
    "price_ranges": {
      "range": {
        "field": "price",
        "ranges": [
          { "to": 100 },
          { "from": 100, "to": 500 },
          { "from": 500, "to": 1000 },
          { "from": 1000 }
        ]
      }
    }
  }
}

// 带 key 的范围
{
  "aggs": {
    "price_ranges": {
      "range": {
        "field": "price",
        "keyed": true,
        "ranges": [
          { "key": "cheap", "to": 100 },
          { "key": "medium", "from": 100, "to": 500 },
          { "key": "expensive", "from": 500 }
        ]
      }
    }
  }
}

// 日期范围
{
  "aggs": {
    "date_ranges": {
      "date_range": {
        "field": "date",
        "format": "yyyy-MM-dd",
        "ranges": [
          { "to": "2024-01-01" },
          { "from": "2024-01-01", "to": "2024-06-01" },
          { "from": "2024-06-01" }
        ]
      }
    }
  }
}

// 相对日期
{
  "aggs": {
    "recent": {
      "date_range": {
        "field": "date",
        "ranges": [
          { "key": "last_week", "from": "now-1w/d", "to": "now/d" },
          { "key": "last_month", "from": "now-1M/d", "to": "now/d" }
        ]
      }
    }
  }
}
```

### Histogram 聚合

```json
// 数值直方图
{
  "aggs": {
    "price_histogram": {
      "histogram": {
        "field": "price",
        "interval": 100,
        "min_doc_count": 0,            // 显示空桶
        "extended_bounds": {           // 扩展边界
          "min": 0,
          "max": 1000
        }
      }
    }
  }
}

// 日期直方图
{
  "aggs": {
    "sales_over_time": {
      "date_histogram": {
        "field": "date",
        "calendar_interval": "month",  // year, quarter, month, week, day, hour, minute
        // 或
        "fixed_interval": "30d",       // 固定间隔
        "format": "yyyy-MM",
        "time_zone": "+08:00",
        "min_doc_count": 0,
        "extended_bounds": {
          "min": "2024-01-01",
          "max": "2024-12-31"
        }
      }
    }
  }
}

// 自动间隔
{
  "aggs": {
    "auto_histogram": {
      "auto_date_histogram": {
        "field": "date",
        "buckets": 10                  // 目标桶数
      }
    }
  }
}
```

### Filter 聚合

```json
// 单个过滤器
{
  "aggs": {
    "expensive_products": {
      "filter": {
        "range": { "price": { "gte": 1000 } }
      },
      "aggs": {
        "avg_price": {
          "avg": { "field": "price" }
        }
      }
    }
  }
}

// 多个过滤器
{
  "aggs": {
    "categories": {
      "filters": {
        "filters": {
          "electronics": { "term": { "category": "electronics" } },
          "clothing": { "term": { "category": "clothing" } },
          "other": {
            "bool": {
              "must_not": [
                { "term": { "category": "electronics" } },
                { "term": { "category": "clothing" } }
              ]
            }
          }
        }
      }
    }
  }
}
```

### 嵌套聚合

```json
// nested 聚合
{
  "aggs": {
    "items": {
      "nested": {
        "path": "items"
      },
      "aggs": {
        "by_product": {
          "terms": { "field": "items.product" },
          "aggs": {
            "avg_quantity": {
              "avg": { "field": "items.quantity" }
            }
          }
        }
      }
    }
  }
}

// reverse_nested: 从嵌套回到父文档
{
  "aggs": {
    "items": {
      "nested": { "path": "items" },
      "aggs": {
        "by_product": {
          "terms": { "field": "items.product" },
          "aggs": {
            "back_to_root": {
              "reverse_nested": {},
              "aggs": {
                "unique_orders": {
                  "cardinality": { "field": "order_id" }
                }
              }
            }
          }
        }
      }
    }
  }
}
```

### 特殊桶聚合

```json
// global: 忽略查询范围
{
  "query": {
    "match": { "category": "electronics" }
  },
  "aggs": {
    "filtered_avg": {
      "avg": { "field": "price" }
    },
    "all_products": {
      "global": {},
      "aggs": {
        "global_avg": {
          "avg": { "field": "price" }
        }
      }
    }
  }
}

// missing: 缺失值桶
{
  "aggs": {
    "products_without_category": {
      "missing": { "field": "category" }
    }
  }
}

// significant_terms: 显著项（异常检测）
{
  "query": {
    "match": { "content": "elasticsearch" }
  },
  "aggs": {
    "significant_tags": {
      "significant_terms": {
        "field": "tags",
        "size": 10
      }
    }
  }
}
```

## 四、管道聚合（Pipeline）

### 父级管道聚合

```json
// 累计求和
{
  "aggs": {
    "sales_per_month": {
      "date_histogram": {
        "field": "date",
        "calendar_interval": "month"
      },
      "aggs": {
        "sales": {
          "sum": { "field": "amount" }
        },
        "cumulative_sales": {
          "cumulative_sum": {
            "buckets_path": "sales"
          }
        }
      }
    }
  }
}

// 移动平均
{
  "aggs": {
    "sales_per_month": {
      "date_histogram": {
        "field": "date",
        "calendar_interval": "month"
      },
      "aggs": {
        "sales": {
          "sum": { "field": "amount" }
        },
        "moving_avg_sales": {
          "moving_fn": {
            "buckets_path": "sales",
            "window": 3,
            "script": "MovingFunctions.unweightedAvg(values)"
          }
        }
      }
    }
  }
}

// 导数（变化率）
{
  "aggs": {
    "sales_per_month": {
      "date_histogram": {
        "field": "date",
        "calendar_interval": "month"
      },
      "aggs": {
        "sales": {
          "sum": { "field": "amount" }
        },
        "sales_derivative": {
          "derivative": {
            "buckets_path": "sales"
          }
        }
      }
    }
  }
}
```

### 兄弟管道聚合

```json
// 统计所有桶的值
{
  "aggs": {
    "sales_per_category": {
      "terms": { "field": "category" },
      "aggs": {
        "total_sales": {
          "sum": { "field": "amount" }
        }
      }
    },
    "avg_category_sales": {
      "avg_bucket": {
        "buckets_path": "sales_per_category>total_sales"
      }
    },
    "max_category_sales": {
      "max_bucket": {
        "buckets_path": "sales_per_category>total_sales"
      }
    },
    "stats_category_sales": {
      "stats_bucket": {
        "buckets_path": "sales_per_category>total_sales"
      }
    }
  }
}

// bucket_sort: 对桶排序
{
  "aggs": {
    "sales_per_category": {
      "terms": {
        "field": "category",
        "size": 100
      },
      "aggs": {
        "total_sales": {
          "sum": { "field": "amount" }
        },
        "sales_bucket_sort": {
          "bucket_sort": {
            "sort": [{ "total_sales": { "order": "desc" } }],
            "from": 0,
            "size": 10
          }
        }
      }
    }
  }
}

// bucket_selector: 过滤桶
{
  "aggs": {
    "sales_per_category": {
      "terms": { "field": "category" },
      "aggs": {
        "total_sales": {
          "sum": { "field": "amount" }
        },
        "sales_filter": {
          "bucket_selector": {
            "buckets_path": {
              "sales": "total_sales"
            },
            "script": "params.sales > 10000"
          }
        }
      }
    }
  }
}
```

## 五、复杂聚合示例

### 多维度分析

```json
// 按类别和时间分析销售额
{
  "size": 0,
  "aggs": {
    "by_category": {
      "terms": { "field": "category", "size": 10 },
      "aggs": {
        "by_month": {
          "date_histogram": {
            "field": "date",
            "calendar_interval": "month"
          },
          "aggs": {
            "total_sales": { "sum": { "field": "amount" } },
            "avg_order_value": { "avg": { "field": "amount" } },
            "order_count": { "value_count": { "field": "order_id" } }
          }
        },
        "category_total": {
          "sum": { "field": "amount" }
        }
      }
    }
  }
}
```

### 环比同比分析

```json
// 计算月度环比
{
  "size": 0,
  "aggs": {
    "monthly_sales": {
      "date_histogram": {
        "field": "date",
        "calendar_interval": "month"
      },
      "aggs": {
        "sales": {
          "sum": { "field": "amount" }
        },
        "sales_diff": {
          "serial_diff": {
            "buckets_path": "sales",
            "lag": 1
          }
        },
        "mom_growth": {
          "bucket_script": {
            "buckets_path": {
              "current": "sales",
              "diff": "sales_diff"
            },
            "script": "params.current > 0 && params.diff != null ? params.diff / (params.current - params.diff) * 100 : null"
          }
        }
      }
    }
  }
}
```

### 漏斗分析

```json
// 用户行为漏斗
{
  "size": 0,
  "aggs": {
    "funnel": {
      "filters": {
        "filters": {
          "view": { "term": { "event": "view" } },
          "add_to_cart": { "term": { "event": "add_to_cart" } },
          "checkout": { "term": { "event": "checkout" } },
          "purchase": { "term": { "event": "purchase" } }
        }
      },
      "aggs": {
        "unique_users": {
          "cardinality": { "field": "user_id" }
        }
      }
    }
  }
}
```

### 地理位置聚合

```json
// 按距离范围统计
{
  "aggs": {
    "distance_rings": {
      "geo_distance": {
        "field": "location",
        "origin": { "lat": 39.915, "lon": 116.404 },
        "unit": "km",
        "ranges": [
          { "to": 1 },
          { "from": 1, "to": 5 },
          { "from": 5, "to": 10 },
          { "from": 10 }
        ]
      }
    }
  }
}

// 地理网格聚合
{
  "aggs": {
    "grid": {
      "geohash_grid": {
        "field": "location",
        "precision": 5
      },
      "aggs": {
        "center": {
          "geo_centroid": { "field": "location" }
        }
      }
    }
  }
}
```

## 六、性能优化

### 聚合缓存

```json
// 使用 filter 上下文以利用缓存
{
  "query": {
    "bool": {
      "filter": [
        { "term": { "status": "published" } }
      ]
    }
  },
  "aggs": {
    "by_category": {
      "terms": { "field": "category" }
    }
  }
}
```

### 采样聚合

```json
// 使用采样减少计算量
{
  "aggs": {
    "sample": {
      "sampler": {
        "shard_size": 1000
      },
      "aggs": {
        "by_category": {
          "terms": { "field": "category" }
        }
      }
    }
  }
}

// 多样性采样
{
  "aggs": {
    "diverse_sample": {
      "diversified_sampler": {
        "shard_size": 1000,
        "field": "category"
      },
      "aggs": {
        "keywords": {
          "significant_terms": { "field": "tags" }
        }
      }
    }
  }
}
```

### 复合聚合（大数据量分页）

```json
// composite 聚合：支持分页的聚合
{
  "size": 0,
  "aggs": {
    "my_composite": {
      "composite": {
        "size": 1000,
        "sources": [
          { "category": { "terms": { "field": "category" } } },
          { "date": { "date_histogram": { "field": "date", "calendar_interval": "day" } } }
        ]
      },
      "aggs": {
        "total_sales": { "sum": { "field": "amount" } }
      }
    }
  }
}

// 下一页
{
  "size": 0,
  "aggs": {
    "my_composite": {
      "composite": {
        "size": 1000,
        "sources": [
          { "category": { "terms": { "field": "category" } } },
          { "date": { "date_histogram": { "field": "date", "calendar_interval": "day" } } }
        ],
        "after": { "category": "electronics", "date": 1704067200000 }
      }
    }
  }
}
```
