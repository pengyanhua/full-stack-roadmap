"""
============================================================
            Elasticsearch 实战示例（Python）
============================================================
使用 elasticsearch-py 库实现常见的 ES 应用场景
pip install elasticsearch
============================================================
"""

from elasticsearch import Elasticsearch, helpers
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json


# ============================================================
#                    连接配置
# ============================================================

def get_es_client() -> Elasticsearch:
    """获取 ES 连接"""
    return Elasticsearch(
        hosts=["http://localhost:9200"],
        # 如果有认证
        # basic_auth=("username", "password"),
        # 如果是 HTTPS
        # ca_certs="/path/to/ca.crt",
        request_timeout=30,
        retry_on_timeout=True,
        max_retries=3,
    )


# ============================================================
#                    一、索引管理
# ============================================================

class IndexManager:
    """索引管理工具"""

    def __init__(self, client: Elasticsearch):
        self.client = client

    def create_index(self, index: str, mappings: Dict, settings: Dict = None):
        """创建索引"""
        body = {"mappings": mappings}
        if settings:
            body["settings"] = settings

        if not self.client.indices.exists(index=index):
            self.client.indices.create(index=index, body=body)
            print(f"索引 {index} 创建成功")
        else:
            print(f"索引 {index} 已存在")

    def delete_index(self, index: str):
        """删除索引"""
        if self.client.indices.exists(index=index):
            self.client.indices.delete(index=index)
            print(f"索引 {index} 删除成功")

    def reindex(self, source_index: str, dest_index: str, query: Dict = None):
        """重建索引"""
        body = {
            "source": {"index": source_index},
            "dest": {"index": dest_index}
        }
        if query:
            body["source"]["query"] = query

        result = self.client.reindex(body=body, wait_for_completion=True)
        print(f"重建索引完成: {result}")

    def update_alias(self, index: str, alias: str, remove_from: str = None):
        """更新别名"""
        actions = [{"add": {"index": index, "alias": alias}}]
        if remove_from:
            actions.insert(0, {"remove": {"index": remove_from, "alias": alias}})

        self.client.indices.update_aliases(body={"actions": actions})
        print(f"别名 {alias} 已更新")


# ============================================================
#                    二、商品搜索
# ============================================================

class ProductSearch:
    """商品搜索服务"""

    INDEX = "products"

    # 索引映射
    MAPPINGS = {
        "properties": {
            "id": {"type": "keyword"},
            "name": {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },
            "description": {"type": "text"},
            "category": {"type": "keyword"},
            "brand": {"type": "keyword"},
            "price": {"type": "float"},
            "original_price": {"type": "float"},
            "sales": {"type": "integer"},
            "rating": {"type": "float"},
            "review_count": {"type": "integer"},
            "tags": {"type": "keyword"},
            "status": {"type": "keyword"},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
            "location": {"type": "geo_point"},
            "suggest": {
                "type": "completion",
                "analyzer": "standard"
            }
        }
    }

    def __init__(self, client: Elasticsearch):
        self.client = client

    def init_index(self):
        """初始化索引"""
        settings = {
            "number_of_shards": 3,
            "number_of_replicas": 1,
            "refresh_interval": "1s"
        }
        manager = IndexManager(self.client)
        manager.create_index(self.INDEX, self.MAPPINGS, settings)

    def index_product(self, product: Dict):
        """索引单个商品"""
        # 添加自动补全字段
        product["suggest"] = {
            "input": [product["name"]] + product.get("tags", []),
            "weight": product.get("sales", 1)
        }

        self.client.index(
            index=self.INDEX,
            id=product["id"],
            document=product,
            refresh=True
        )

    def bulk_index(self, products: List[Dict]):
        """批量索引"""
        actions = []
        for product in products:
            product["suggest"] = {
                "input": [product["name"]] + product.get("tags", []),
                "weight": product.get("sales", 1)
            }
            actions.append({
                "_index": self.INDEX,
                "_id": product["id"],
                "_source": product
            })

        success, failed = helpers.bulk(self.client, actions, refresh=True)
        print(f"批量索引完成: 成功 {success}, 失败 {len(failed)}")

    def search(
        self,
        keyword: str = None,
        category: str = None,
        brand: str = None,
        price_min: float = None,
        price_max: float = None,
        tags: List[str] = None,
        sort_by: str = "relevance",
        page: int = 1,
        page_size: int = 20
    ) -> Dict:
        """
        商品搜索

        Args:
            keyword: 搜索关键词
            category: 分类
            brand: 品牌
            price_min: 最低价格
            price_max: 最高价格
            tags: 标签列表
            sort_by: 排序方式 (relevance, price_asc, price_desc, sales, rating)
            page: 页码
            page_size: 每页数量
        """
        query = {"bool": {"must": [], "filter": []}}

        # 关键词搜索
        if keyword:
            query["bool"]["must"].append({
                "multi_match": {
                    "query": keyword,
                    "fields": ["name^3", "description", "tags^2"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            })
        else:
            query["bool"]["must"].append({"match_all": {}})

        # 过滤条件
        query["bool"]["filter"].append({"term": {"status": "active"}})

        if category:
            query["bool"]["filter"].append({"term": {"category": category}})

        if brand:
            query["bool"]["filter"].append({"term": {"brand": brand}})

        if price_min is not None or price_max is not None:
            price_range = {}
            if price_min is not None:
                price_range["gte"] = price_min
            if price_max is not None:
                price_range["lte"] = price_max
            query["bool"]["filter"].append({"range": {"price": price_range}})

        if tags:
            query["bool"]["filter"].append({"terms": {"tags": tags}})

        # 排序
        sort = []
        if sort_by == "price_asc":
            sort = [{"price": "asc"}]
        elif sort_by == "price_desc":
            sort = [{"price": "desc"}]
        elif sort_by == "sales":
            sort = [{"sales": "desc"}]
        elif sort_by == "rating":
            sort = [{"rating": "desc"}, {"review_count": "desc"}]
        else:  # relevance
            sort = [{"_score": "desc"}, {"sales": "desc"}]

        # 执行搜索
        body = {
            "query": query,
            "sort": sort,
            "from": (page - 1) * page_size,
            "size": page_size,
            "highlight": {
                "fields": {
                    "name": {"number_of_fragments": 0},
                    "description": {"fragment_size": 100, "number_of_fragments": 3}
                },
                "pre_tags": ["<em>"],
                "post_tags": ["</em>"]
            },
            "aggs": {
                "categories": {"terms": {"field": "category", "size": 20}},
                "brands": {"terms": {"field": "brand", "size": 20}},
                "price_ranges": {
                    "range": {
                        "field": "price",
                        "ranges": [
                            {"key": "0-100", "to": 100},
                            {"key": "100-500", "from": 100, "to": 500},
                            {"key": "500-1000", "from": 500, "to": 1000},
                            {"key": "1000+", "from": 1000}
                        ]
                    }
                },
                "price_stats": {"stats": {"field": "price"}}
            }
        }

        result = self.client.search(index=self.INDEX, body=body)

        return {
            "total": result["hits"]["total"]["value"],
            "products": [
                {
                    **hit["_source"],
                    "score": hit["_score"],
                    "highlight": hit.get("highlight", {})
                }
                for hit in result["hits"]["hits"]
            ],
            "aggregations": {
                "categories": [
                    {"key": b["key"], "count": b["doc_count"]}
                    for b in result["aggregations"]["categories"]["buckets"]
                ],
                "brands": [
                    {"key": b["key"], "count": b["doc_count"]}
                    for b in result["aggregations"]["brands"]["buckets"]
                ],
                "price_ranges": [
                    {"key": b["key"], "count": b["doc_count"]}
                    for b in result["aggregations"]["price_ranges"]["buckets"]
                ],
                "price_stats": result["aggregations"]["price_stats"]
            }
        }

    def suggest(self, prefix: str, size: int = 10) -> List[str]:
        """自动补全建议"""
        body = {
            "suggest": {
                "product_suggest": {
                    "prefix": prefix,
                    "completion": {
                        "field": "suggest",
                        "size": size,
                        "skip_duplicates": True
                    }
                }
            }
        }

        result = self.client.search(index=self.INDEX, body=body)
        suggestions = result["suggest"]["product_suggest"][0]["options"]

        return [s["text"] for s in suggestions]

    def get_similar(self, product_id: str, size: int = 10) -> List[Dict]:
        """相似商品推荐（基于 More Like This）"""
        body = {
            "query": {
                "more_like_this": {
                    "fields": ["name", "description", "tags"],
                    "like": [{"_index": self.INDEX, "_id": product_id}],
                    "min_term_freq": 1,
                    "min_doc_freq": 1,
                    "max_query_terms": 25
                }
            },
            "size": size
        }

        result = self.client.search(index=self.INDEX, body=body)
        return [hit["_source"] for hit in result["hits"]["hits"]]


# ============================================================
#                    三、日志分析
# ============================================================

class LogAnalyzer:
    """日志分析服务"""

    INDEX_PATTERN = "logs-*"

    MAPPINGS = {
        "properties": {
            "timestamp": {"type": "date"},
            "level": {"type": "keyword"},
            "service": {"type": "keyword"},
            "host": {"type": "keyword"},
            "message": {"type": "text"},
            "trace_id": {"type": "keyword"},
            "user_id": {"type": "keyword"},
            "request_path": {"type": "keyword"},
            "response_time": {"type": "integer"},
            "status_code": {"type": "integer"},
            "error_type": {"type": "keyword"},
            "stack_trace": {"type": "text"}
        }
    }

    def __init__(self, client: Elasticsearch):
        self.client = client

    def get_index_name(self, date: datetime = None) -> str:
        """获取日期索引名"""
        date = date or datetime.now()
        return f"logs-{date.strftime('%Y.%m.%d')}"

    def index_log(self, log: Dict):
        """索引日志"""
        index = self.get_index_name(
            datetime.fromisoformat(log["timestamp"]) if isinstance(log["timestamp"], str)
            else log["timestamp"]
        )
        self.client.index(index=index, document=log)

    def search_logs(
        self,
        keyword: str = None,
        level: str = None,
        service: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        page: int = 1,
        page_size: int = 50
    ) -> Dict:
        """搜索日志"""
        query = {"bool": {"must": [], "filter": []}}

        if keyword:
            query["bool"]["must"].append({
                "multi_match": {
                    "query": keyword,
                    "fields": ["message", "stack_trace"]
                }
            })
        else:
            query["bool"]["must"].append({"match_all": {}})

        if level:
            query["bool"]["filter"].append({"term": {"level": level}})

        if service:
            query["bool"]["filter"].append({"term": {"service": service}})

        if start_time or end_time:
            time_range = {}
            if start_time:
                time_range["gte"] = start_time.isoformat()
            if end_time:
                time_range["lte"] = end_time.isoformat()
            query["bool"]["filter"].append({"range": {"timestamp": time_range}})

        body = {
            "query": query,
            "sort": [{"timestamp": "desc"}],
            "from": (page - 1) * page_size,
            "size": page_size
        }

        result = self.client.search(index=self.INDEX_PATTERN, body=body)

        return {
            "total": result["hits"]["total"]["value"],
            "logs": [hit["_source"] for hit in result["hits"]["hits"]]
        }

    def get_error_stats(
        self,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1h"
    ) -> Dict:
        """错误统计"""
        body = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"level": "ERROR"}},
                        {"range": {"timestamp": {
                            "gte": start_time.isoformat(),
                            "lte": end_time.isoformat()
                        }}}
                    ]
                }
            },
            "size": 0,
            "aggs": {
                "errors_over_time": {
                    "date_histogram": {
                        "field": "timestamp",
                        "fixed_interval": interval,
                        "min_doc_count": 0
                    }
                },
                "by_service": {
                    "terms": {"field": "service", "size": 20},
                    "aggs": {
                        "by_error_type": {
                            "terms": {"field": "error_type", "size": 10}
                        }
                    }
                },
                "top_errors": {
                    "terms": {"field": "error_type", "size": 10}
                }
            }
        }

        result = self.client.search(index=self.INDEX_PATTERN, body=body)

        return {
            "total_errors": result["hits"]["total"]["value"],
            "errors_over_time": [
                {"time": b["key_as_string"], "count": b["doc_count"]}
                for b in result["aggregations"]["errors_over_time"]["buckets"]
            ],
            "by_service": [
                {
                    "service": b["key"],
                    "count": b["doc_count"],
                    "error_types": [
                        {"type": e["key"], "count": e["doc_count"]}
                        for e in b["by_error_type"]["buckets"]
                    ]
                }
                for b in result["aggregations"]["by_service"]["buckets"]
            ],
            "top_errors": [
                {"type": b["key"], "count": b["doc_count"]}
                for b in result["aggregations"]["top_errors"]["buckets"]
            ]
        }

    def get_performance_stats(
        self,
        service: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict:
        """性能统计"""
        body = {
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"service": service}},
                        {"exists": {"field": "response_time"}},
                        {"range": {"timestamp": {
                            "gte": start_time.isoformat(),
                            "lte": end_time.isoformat()
                        }}}
                    ]
                }
            },
            "size": 0,
            "aggs": {
                "response_time_stats": {
                    "extended_stats": {"field": "response_time"}
                },
                "response_time_percentiles": {
                    "percentiles": {
                        "field": "response_time",
                        "percents": [50, 75, 90, 95, 99]
                    }
                },
                "by_path": {
                    "terms": {"field": "request_path", "size": 20},
                    "aggs": {
                        "avg_time": {"avg": {"field": "response_time"}},
                        "p95_time": {
                            "percentiles": {
                                "field": "response_time",
                                "percents": [95]
                            }
                        }
                    }
                },
                "status_codes": {
                    "terms": {"field": "status_code"}
                }
            }
        }

        result = self.client.search(index=self.INDEX_PATTERN, body=body)
        aggs = result["aggregations"]

        return {
            "total_requests": result["hits"]["total"]["value"],
            "response_time": {
                "avg": aggs["response_time_stats"]["avg"],
                "min": aggs["response_time_stats"]["min"],
                "max": aggs["response_time_stats"]["max"],
                "std_dev": aggs["response_time_stats"]["std_deviation"],
                "percentiles": aggs["response_time_percentiles"]["values"]
            },
            "by_path": [
                {
                    "path": b["key"],
                    "count": b["doc_count"],
                    "avg_time": b["avg_time"]["value"],
                    "p95_time": b["p95_time"]["values"]["95.0"]
                }
                for b in aggs["by_path"]["buckets"]
            ],
            "status_codes": [
                {"code": b["key"], "count": b["doc_count"]}
                for b in aggs["status_codes"]["buckets"]
            ]
        }


# ============================================================
#                    四、全文搜索引擎
# ============================================================

class ArticleSearch:
    """文章搜索引擎"""

    INDEX = "articles"

    MAPPINGS = {
        "properties": {
            "id": {"type": "keyword"},
            "title": {
                "type": "text",
                "analyzer": "standard",
                "fields": {"keyword": {"type": "keyword"}}
            },
            "content": {"type": "text", "analyzer": "standard"},
            "summary": {"type": "text"},
            "author": {
                "type": "object",
                "properties": {
                    "id": {"type": "keyword"},
                    "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}}
                }
            },
            "category": {"type": "keyword"},
            "tags": {"type": "keyword"},
            "publish_date": {"type": "date"},
            "views": {"type": "integer"},
            "likes": {"type": "integer"},
            "status": {"type": "keyword"}
        }
    }

    def __init__(self, client: Elasticsearch):
        self.client = client

    def search(
        self,
        query: str,
        category: str = None,
        author_id: str = None,
        tags: List[str] = None,
        date_from: datetime = None,
        date_to: datetime = None,
        page: int = 1,
        page_size: int = 10
    ) -> Dict:
        """
        文章搜索

        支持高亮、聚合、评分优化
        """
        bool_query = {"must": [], "filter": [], "should": []}

        # 主查询
        if query:
            bool_query["must"].append({
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "content", "summary^2", "tags^2"],
                    "type": "best_fields",
                    "minimum_should_match": "75%"
                }
            })

            # 短语匹配加分
            bool_query["should"].append({
                "match_phrase": {
                    "title": {"query": query, "boost": 2}
                }
            })
        else:
            bool_query["must"].append({"match_all": {}})

        # 过滤条件
        bool_query["filter"].append({"term": {"status": "published"}})

        if category:
            bool_query["filter"].append({"term": {"category": category}})

        if author_id:
            bool_query["filter"].append({"term": {"author.id": author_id}})

        if tags:
            bool_query["filter"].append({"terms": {"tags": tags}})

        if date_from or date_to:
            date_range = {}
            if date_from:
                date_range["gte"] = date_from.isoformat()
            if date_to:
                date_range["lte"] = date_to.isoformat()
            bool_query["filter"].append({"range": {"publish_date": date_range}})

        # 构建最终查询（带评分优化）
        body = {
            "query": {
                "function_score": {
                    "query": {"bool": bool_query},
                    "functions": [
                        # 新文章加分
                        {
                            "gauss": {
                                "publish_date": {
                                    "origin": "now",
                                    "scale": "30d",
                                    "decay": 0.5
                                }
                            },
                            "weight": 1.5
                        },
                        # 热门文章加分
                        {
                            "field_value_factor": {
                                "field": "views",
                                "factor": 1.2,
                                "modifier": "log1p",
                                "missing": 1
                            },
                            "weight": 1
                        },
                        # 高赞文章加分
                        {
                            "field_value_factor": {
                                "field": "likes",
                                "factor": 1.5,
                                "modifier": "log1p",
                                "missing": 1
                            },
                            "weight": 1
                        }
                    ],
                    "score_mode": "sum",
                    "boost_mode": "multiply"
                }
            },
            "from": (page - 1) * page_size,
            "size": page_size,
            "highlight": {
                "pre_tags": ["<mark>"],
                "post_tags": ["</mark>"],
                "fields": {
                    "title": {"number_of_fragments": 0},
                    "content": {"fragment_size": 200, "number_of_fragments": 3},
                    "summary": {"number_of_fragments": 0}
                }
            },
            "aggs": {
                "categories": {"terms": {"field": "category", "size": 20}},
                "tags": {"terms": {"field": "tags", "size": 30}},
                "by_month": {
                    "date_histogram": {
                        "field": "publish_date",
                        "calendar_interval": "month",
                        "min_doc_count": 0
                    }
                }
            },
            "_source": {
                "excludes": ["content"]  # 不返回全文内容
            }
        }

        result = self.client.search(index=self.INDEX, body=body)

        return {
            "total": result["hits"]["total"]["value"],
            "articles": [
                {
                    **hit["_source"],
                    "score": hit["_score"],
                    "highlight": hit.get("highlight", {})
                }
                for hit in result["hits"]["hits"]
            ],
            "aggregations": {
                "categories": result["aggregations"]["categories"]["buckets"],
                "tags": result["aggregations"]["tags"]["buckets"],
                "by_month": result["aggregations"]["by_month"]["buckets"]
            }
        }


# ============================================================
#                    主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Elasticsearch 实战示例")
    print("=" * 60)

    # 获取客户端
    es = get_es_client()

    # 测试连接
    try:
        info = es.info()
        print(f"连接成功！ES 版本: {info['version']['number']}\n")
    except Exception as e:
        print(f"连接失败: {e}")
        exit(1)

    # 示例：商品搜索
    print("--- 商品搜索示例 ---")
    product_search = ProductSearch(es)

    # 初始化索引
    product_search.init_index()

    # 索引测试数据
    test_products = [
        {
            "id": "1",
            "name": "iPhone 15 Pro",
            "description": "Apple 最新款智能手机，搭载 A17 Pro 芯片",
            "category": "手机",
            "brand": "Apple",
            "price": 8999,
            "original_price": 9999,
            "sales": 10000,
            "rating": 4.8,
            "review_count": 5000,
            "tags": ["智能手机", "5G", "iOS"],
            "status": "active",
            "created_at": datetime.now().isoformat()
        },
        {
            "id": "2",
            "name": "MacBook Pro 14",
            "description": "专业级笔记本电脑，M3 Pro 芯片",
            "category": "电脑",
            "brand": "Apple",
            "price": 16999,
            "sales": 5000,
            "rating": 4.9,
            "review_count": 2000,
            "tags": ["笔记本", "专业", "macOS"],
            "status": "active",
            "created_at": datetime.now().isoformat()
        },
        {
            "id": "3",
            "name": "Samsung Galaxy S24",
            "description": "三星旗舰智能手机，AI 功能强大",
            "category": "手机",
            "brand": "Samsung",
            "price": 6999,
            "sales": 8000,
            "rating": 4.7,
            "review_count": 3000,
            "tags": ["智能手机", "5G", "Android", "AI"],
            "status": "active",
            "created_at": datetime.now().isoformat()
        }
    ]

    product_search.bulk_index(test_products)

    # 搜索测试
    print("\n搜索 '智能手机':")
    results = product_search.search(keyword="智能手机", page=1, page_size=10)
    print(f"找到 {results['total']} 个商品")
    for p in results["products"]:
        print(f"  - {p['name']} (¥{p['price']}, 评分: {p.get('score', 'N/A'):.2f})")

    print("\n分类聚合:")
    for cat in results["aggregations"]["categories"]:
        print(f"  - {cat['key']}: {cat['count']} 个商品")

    # 自动补全
    print("\n自动补全 'iph':")
    suggestions = product_search.suggest("iph")
    print(f"  建议: {suggestions}")

    # 清理
    # es.indices.delete(index=ProductSearch.INDEX, ignore=[404])

    print("\n测试完成！")
