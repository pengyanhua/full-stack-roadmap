# 向量数据库完整教程

## 目录
1. [向量数据库原理](#1-向量数据库原理)
2. [ANN 近似最近邻算法](#2-ann-近似最近邻算法)
3. [Milvus 完整教程](#3-milvus-完整教程)
4. [Qdrant 使用指南](#4-qdrant-使用指南)
5. [Chroma 轻量级方案](#5-chroma-轻量级方案)
6. [Pinecone 云服务](#6-pinecone-云服务)
7. [性能对比与选型](#7-性能对比与选型)
8. [LangChain 集成实战](#8-langchain-集成实战)
9. [最佳实践](#9-最佳实践)

---

## 1. 向量数据库原理

### 1.1 什么是向量数据库

向量数据库是一种专门设计用于存储、索引和检索高维向量数据的数据库系统。在 AI 应用中，文本、图像、音频等非结构化数据通过 Embedding 模型转换为高维向量后，需要高效的存储和相似度检索能力。

```
┌──────────────────────────────────────────────────────────────────┐
│                   向量数据库核心原理                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  非结构化数据                向量表示              向量数据库        │
│  ┌──────────┐        ┌──────────────┐      ┌──────────────┐    │
│  │ "Hello"  │──┐     │ [0.1, 0.3,   │      │  存储向量     │    │
│  │ "World"  │  │     │  0.8, -0.2,  │─────>│  构建索引     │    │
│  │  (文本)   │  │     │  ..., 0.5]   │      │  相似度搜索   │    │
│  └──────────┘  │     │  (1536维)    │      └──────────────┘    │
│                │     └──────────────┘                           │
│  ┌──────────┐  │                                                │
│  │  (图片)   │──┼──> Embedding Model                           │
│  └──────────┘  │                                                │
│                │     查询向量                                    │
│  ┌──────────┐  │     ┌──────────────┐      ┌──────────────┐    │
│  │  (音频)   │──┘     │ [0.2, 0.4,   │─────>│  Top-K 最近邻 │    │
│  └──────────┘        │  0.7, ...]   │      │  结果返回     │    │
│                      └──────────────┘      └──────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 向量数据库 vs 传统数据库

| 特性 | 传统数据库（MySQL/PostgreSQL） | 向量数据库（Milvus/Qdrant） |
|------|-------------------------------|----------------------------|
| 数据类型 | 结构化数据（行/列） | 高维向量 + 元数据 |
| 查询方式 | SQL 精确查询 | 相似度搜索（ANN） |
| 索引类型 | B-Tree、Hash | HNSW、IVF、PQ |
| 查询结果 | 精确匹配 | 近似最近邻 |
| 适用场景 | 事务处理、精确查询 | 语义搜索、推荐系统 |
| 扩展性 | 垂直扩展为主 | 水平扩展（分布式） |

### 1.3 相似度度量方式

```python
import numpy as np
from typing import List


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    余弦相似度 - 最常用
    衡量两个向量方向的相似性，值域 [-1, 1]
    1 表示完全相同方向，0 表示正交，-1 表示完全相反
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def euclidean_distance(vec_a: List[float], vec_b: List[float]) -> float:
    """
    欧氏距离 (L2)
    衡量两个向量在空间中的直线距离，值越小越相似
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    return np.linalg.norm(a - b)


def inner_product(vec_a: List[float], vec_b: List[float]) -> float:
    """
    内积 (IP)
    当向量已归一化时等同于余弦相似度
    适用于已归一化的 Embedding 向量
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    return np.dot(a, b)


# 使用示例
if __name__ == "__main__":
    # 模拟两个Embedding向量
    vec1 = [0.1, 0.3, 0.5, 0.7, 0.2]
    vec2 = [0.2, 0.4, 0.6, 0.6, 0.3]
    vec3 = [-0.1, -0.3, -0.5, -0.7, -0.2]  # 与vec1方向相反

    print(f"vec1 与 vec2 余弦相似度: {cosine_similarity(vec1, vec2):.4f}")
    print(f"vec1 与 vec3 余弦相似度: {cosine_similarity(vec1, vec3):.4f}")
    print(f"vec1 与 vec2 欧氏距离: {euclidean_distance(vec1, vec2):.4f}")
    print(f"vec1 与 vec2 内积: {inner_product(vec1, vec2):.4f}")
```

---

## 2. ANN 近似最近邻算法

### 2.1 为什么需要 ANN

精确的最近邻搜索（KNN）在高维空间中计算量极大，时间复杂度为 O(n*d)，其中 n 是向量数量，d 是维度。对于百万级向量来说，精确搜索耗时不可接受。ANN 通过牺牲少量精度换取数量级的速度提升。

```
┌──────────────────────────────────────────────────────────────────┐
│                    ANN 算法对比                                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─── HNSW (Hierarchical Navigable Small World) ──────────────┐ │
│  │                                                              │ │
│  │  Layer 2:    A ─────────────────── D                        │ │
│  │              │                     │                        │ │
│  │  Layer 1:    A ──── B ──── C ──── D                        │ │
│  │              │      │      │      │                        │ │
│  │  Layer 0:    A ─ E ─ B ─ F ─ C ─ G ─ D ─ H                │ │
│  │                                                              │ │
│  │  原理: 多层图结构，从顶层快速定位到底层精确搜索                │ │
│  │  优点: 查询速度快，召回率高                                  │ │
│  │  缺点: 内存开销大，构建索引慢                                │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─── IVF (Inverted File Index) ──────────────────────────────┐ │
│  │                                                              │ │
│  │  聚类中心:  C1      C2      C3      C4                     │ │
│  │             │       │       │       │                      │ │
│  │  倒排表:  [v1,v5] [v2,v6] [v3,v7] [v4,v8]                 │ │
│  │                                                              │ │
│  │  原理: 先用K-Means聚类，查询时只搜索最近的几个聚类             │ │
│  │  优点: 内存效率高，适合大规模数据                             │ │
│  │  缺点: 需要训练聚类，nprobe参数影响精度                       │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─── PQ (Product Quantization) ─────────────────────────────┐  │
│  │                                                              │ │
│  │  原始向量:  [0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8]     │ │
│  │              └─ 子空间1 ─┘ └─ 子空间2 ─┘                     │ │
│  │  量化编码:  [codebook_id_1, codebook_id_2]                   │ │
│  │                                                              │ │
│  │  原理: 将高维向量分段量化，大幅压缩存储空间                    │ │
│  │  优点: 极低内存占用                                          │ │
│  │  缺点: 精度损失较大                                          │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 ANN 算法对比

| 算法 | 时间复杂度（查询） | 空间复杂度 | 召回率 | 构建速度 | 适用场景 |
|------|-------------------|-----------|--------|---------|----------|
| HNSW | O(log n) | O(n * M) | 95-99% | 慢 | 中小规模高精度 |
| IVF_FLAT | O(n/k * nprobe) | O(n * d) | 90-95% | 中等 | 大规模数据 |
| IVF_PQ | O(n/k * nprobe) | O(n * m) | 85-92% | 中等 | 超大规模低内存 |
| FLAT（暴力搜索） | O(n * d) | O(n * d) | 100% | 无需构建 | 小规模精确搜索 |

### 2.3 索引参数对性能的影响

```python
"""
不同索引参数对搜索性能的影响实验
"""

# HNSW 参数说明
hnsw_params = {
    "M": 16,           # 每个节点的最大连接数
                        # 增大M -> 更高召回率，更多内存，更慢的构建
                        # 推荐值: 8-64, 默认16

    "ef_construction": 200,  # 构建索引时的搜索宽度
                             # 增大 -> 更高质量的图，更慢的构建
                             # 推荐值: 100-500

    "ef": 100,          # 查询时的搜索宽度
                        # 增大ef -> 更高召回率，更慢的查询
                        # 必须 >= k (返回结果数)
                        # 推荐值: 50-500
}

# IVF 参数说明
ivf_params = {
    "nlist": 1024,      # 聚类中心数量
                        # 推荐值: sqrt(n) 到 4*sqrt(n)
                        # n=100万时，推荐1024-4096

    "nprobe": 16,       # 查询时搜索的聚类数量
                        # 增大nprobe -> 更高召回率，更慢的查询
                        # 推荐值: nlist的1%-10%
}

# 不同场景的推荐配置
configurations = {
    "高精度低延迟": {
        "index": "HNSW",
        "M": 32,
        "ef_construction": 400,
        "ef": 200,
        "use_case": "实时问答系统"
    },
    "大规模均衡": {
        "index": "IVF_FLAT",
        "nlist": 4096,
        "nprobe": 64,
        "use_case": "企业知识库 (百万级文档)"
    },
    "超大规模低内存": {
        "index": "IVF_PQ",
        "nlist": 4096,
        "nprobe": 32,
        "m": 16,  # PQ子空间数
        "use_case": "推荐系统 (亿级向量)"
    },
    "小规模精确": {
        "index": "FLAT",
        "use_case": "原型开发 (<10万向量)"
    }
}
```

---

## 3. Milvus 完整教程

### 3.1 Milvus 架构概览

```
┌──────────────────────────────────────────────────────────────────┐
│                    Milvus 系统架构                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                   接入层 (Proxy)                       │       │
│  │  - 请求路由和负载均衡                                   │       │
│  │  - 认证和鉴权                                          │       │
│  └──────────────────────┬───────────────────────────────┘       │
│                          │                                       │
│  ┌──────────────┐  ┌────v───────┐  ┌──────────────┐           │
│  │  协调服务      │  │  工作节点   │  │  消息队列     │           │
│  │  (Coordinator) │  │  (Worker)  │  │  (Pulsar/    │           │
│  │  - Root Coord  │  │  - Query   │  │   Kafka)     │           │
│  │  - Data Coord  │  │  - Data    │  │              │           │
│  │  - Query Coord │  │  - Index   │  │              │           │
│  └──────────────┘  └────────────┘  └──────────────┘           │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                  存储层                                 │       │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐       │       │
│  │  │  etcd    │  │  MinIO   │  │  RocksMQ     │       │       │
│  │  │ (元数据)  │  │ (对象存储)│  │ (日志存储)    │       │       │
│  │  └──────────┘  └──────────┘  └──────────────┘       │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Docker 安装 Milvus

```bash
# 下载 docker-compose 配置
wget https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 或者使用最小配置
cat > docker-compose.yml << 'EOF'
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - etcd_data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - minio_data:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "standalone"]
    security_opt:
      - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus_data:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

volumes:
  etcd_data:
  minio_data:
  milvus_data:
EOF

# 启动 Milvus
docker-compose up -d

# 检查状态
docker-compose ps
```

### 3.3 Milvus Python SDK 完整使用

```python
"""
Milvus 完整使用教程
安装: pip install pymilvus
"""

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
import numpy as np
from typing import List, Dict, Any


class MilvusManager:
    """Milvus 向量数据库管理器"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        db_name: str = "default"
    ):
        self.host = host
        self.port = port
        self.db_name = db_name
        self.collection = None

    def connect(self) -> None:
        """连接到 Milvus 服务器"""
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port,
            db_name=self.db_name
        )
        print(f"已连接到 Milvus: {self.host}:{self.port}")

    def create_collection(
        self,
        collection_name: str,
        dim: int = 1536,
        description: str = ""
    ) -> Collection:
        """
        创建集合（类似关系数据库的表）

        Args:
            collection_name: 集合名称
            dim: 向量维度
            description: 集合描述
        """
        # 检查是否已存在
        if utility.has_collection(collection_name):
            print(f"集合 {collection_name} 已存在，直接加载")
            self.collection = Collection(collection_name)
            return self.collection

        # 定义字段
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
                description="主键ID"
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="原始文本内容"
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=dim,
                description="文本向量"
            ),
            FieldSchema(
                name="source",
                dtype=DataType.VARCHAR,
                max_length=512,
                description="文档来源"
            ),
            FieldSchema(
                name="page_number",
                dtype=DataType.INT64,
                description="页码"
            ),
        ]

        # 创建Schema
        schema = CollectionSchema(
            fields=fields,
            description=description or f"RAG知识库: {collection_name}"
        )

        # 创建集合
        self.collection = Collection(
            name=collection_name,
            schema=schema
        )

        print(f"集合 {collection_name} 创建成功, 维度: {dim}")
        return self.collection

    def create_index(
        self,
        index_type: str = "HNSW",
        metric_type: str = "COSINE",
        params: Dict = None
    ) -> None:
        """
        创建向量索引

        Args:
            index_type: 索引类型 (HNSW, IVF_FLAT, IVF_PQ, FLAT)
            metric_type: 距离度量 (COSINE, L2, IP)
            params: 索引参数
        """
        if params is None:
            if index_type == "HNSW":
                params = {"M": 16, "efConstruction": 256}
            elif index_type == "IVF_FLAT":
                params = {"nlist": 1024}
            elif index_type == "IVF_PQ":
                params = {"nlist": 1024, "m": 16, "nbits": 8}
            else:
                params = {}

        index_params = {
            "index_type": index_type,
            "metric_type": metric_type,
            "params": params
        }

        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        print(f"索引创建完成: {index_type}, 度量: {metric_type}")

    def insert_data(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        sources: List[str],
        page_numbers: List[int]
    ) -> List[int]:
        """
        插入数据

        Args:
            texts: 文本列表
            embeddings: 向量列表
            sources: 来源列表
            page_numbers: 页码列表
        Returns:
            插入的ID列表
        """
        data = [
            texts,
            embeddings,
            sources,
            page_numbers
        ]

        insert_result = self.collection.insert(data)
        self.collection.flush()

        print(f"插入 {len(texts)} 条数据, IDs: {insert_result.primary_keys[:5]}...")
        return insert_result.primary_keys

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        output_fields: List[str] = None,
        search_params: Dict = None
    ) -> List[Dict[str, Any]]:
        """
        向量搜索

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            output_fields: 需要返回的字段
            search_params: 搜索参数
        """
        # 加载集合到内存
        self.collection.load()

        if output_fields is None:
            output_fields = ["text", "source", "page_number"]

        if search_params is None:
            search_params = {"metric_type": "COSINE", "params": {"ef": 100}}

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=output_fields
        )

        # 格式化结果
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.entity.get("text", ""),
                    "source": hit.entity.get("source", ""),
                    "page_number": hit.entity.get("page_number", 0)
                })

        return formatted_results

    def hybrid_search(
        self,
        query_embedding: List[float],
        filter_expr: str = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        带过滤条件的混合搜索

        Args:
            query_embedding: 查询向量
            filter_expr: 过滤表达式，如 'source == "manual.pdf"'
            top_k: 返回数量
        """
        self.collection.load()

        search_params = {"metric_type": "COSINE", "params": {"ef": 100}}

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["text", "source", "page_number"]
        )

        formatted = []
        for hits in results:
            for hit in hits:
                formatted.append({
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.entity.get("text", ""),
                    "source": hit.entity.get("source", ""),
                    "page_number": hit.entity.get("page_number", 0)
                })

        return formatted

    def delete_by_expr(self, expr: str) -> None:
        """按条件删除数据"""
        self.collection.delete(expr)
        print(f"已删除满足条件的数据: {expr}")

    def get_stats(self) -> Dict:
        """获取集合统计信息"""
        self.collection.flush()
        return {
            "name": self.collection.name,
            "num_entities": self.collection.num_entities,
            "schema": str(self.collection.schema)
        }

    def disconnect(self) -> None:
        """断开连接"""
        connections.disconnect("default")
        print("已断开 Milvus 连接")


# ============================================================
#  完整使用示例
# ============================================================

if __name__ == "__main__":
    # 1. 初始化并连接
    manager = MilvusManager(host="localhost", port=19530)
    manager.connect()

    # 2. 创建集合
    manager.create_collection(
        collection_name="rag_knowledge_base",
        dim=1536,
        description="RAG知识库向量存储"
    )

    # 3. 创建索引
    manager.create_index(
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256}
    )

    # 4. 模拟插入数据
    np.random.seed(42)
    num_docs = 100
    dim = 1536

    texts = [f"这是第{i}条测试文档的内容" for i in range(num_docs)]
    embeddings = np.random.randn(num_docs, dim).tolist()
    sources = [f"doc_{i // 10}.pdf" for i in range(num_docs)]
    pages = [i % 20 + 1 for i in range(num_docs)]

    ids = manager.insert_data(texts, embeddings, sources, pages)

    # 5. 搜索
    query_vec = np.random.randn(dim).tolist()
    results = manager.search(query_vec, top_k=5)

    print("\n搜索结果:")
    for r in results:
        print(f"  ID: {r['id']}, Score: {r['score']:.4f}, "
              f"Source: {r['source']}, Text: {r['text'][:50]}")

    # 6. 带过滤的搜索
    filtered_results = manager.hybrid_search(
        query_vec,
        filter_expr='source == "doc_0.pdf"',
        top_k=3
    )

    print("\n过滤搜索结果 (source='doc_0.pdf'):")
    for r in filtered_results:
        print(f"  ID: {r['id']}, Score: {r['score']:.4f}, Source: {r['source']}")

    # 7. 统计信息
    stats = manager.get_stats()
    print(f"\n集合统计: {stats}")

    # 8. 断开连接
    manager.disconnect()
```

---

## 4. Qdrant 使用指南

### 4.1 Qdrant 安装与启动

```bash
# Docker 安装
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant

# Python SDK 安装
pip install qdrant-client
```

### 4.2 Qdrant 架构特点

```
┌──────────────────────────────────────────────────────────────────┐
│                    Qdrant 核心特性                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  Collection (集合)                                    │        │
│  │  ┌─────────────────────────────────────────────┐    │        │
│  │  │  Point (数据点)                               │    │        │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │    │        │
│  │  │  │  ID       │  │  Vector  │  │  Payload │  │    │        │
│  │  │  │  (唯一标识)│  │  (向量)   │  │  (元数据) │  │    │        │
│  │  │  └──────────┘  └──────────┘  └──────────┘  │    │        │
│  │  └─────────────────────────────────────────────┘    │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
│  核心优势:                                                        │
│  - Payload 过滤: 支持丰富的条件过滤（数值、文本、地理位置）          │
│  - 多向量: 一个Point可存储多个向量（稀疏+稠密）                    │
│  - 量化: 内置标量量化和乘积量化                                    │
│  - Rust 实现: 高性能、内存安全                                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 4.3 Qdrant Python SDK 完整使用

```python
"""
Qdrant 完整使用教程
安装: pip install qdrant-client
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    SearchRequest,
    PayloadSchemaType,
)
import numpy as np
from typing import List, Dict, Any
import uuid


class QdrantManager:
    """Qdrant 向量数据库管理器"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: str = None
    ):
        """
        初始化Qdrant客户端

        Args:
            host: 服务器地址
            port: 端口号
            api_key: API密钥（Qdrant Cloud使用）
        """
        if api_key:
            self.client = QdrantClient(
                url=f"https://{host}",
                api_key=api_key
            )
        else:
            self.client = QdrantClient(host=host, port=port)

        print(f"已连接到 Qdrant: {host}:{port}")

    def create_collection(
        self,
        collection_name: str,
        dim: int = 1536,
        distance: str = "Cosine"
    ) -> None:
        """创建集合"""
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT
        }

        # 检查是否存在
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]

        if collection_name in existing:
            print(f"集合 {collection_name} 已存在")
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dim,
                distance=distance_map.get(distance, Distance.COSINE)
            )
        )

        # 创建Payload索引以加速过滤
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD
        )

        print(f"集合 {collection_name} 创建成功, 维度: {dim}, 度量: {distance}")

    def upsert_points(
        self,
        collection_name: str,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict] = None
    ) -> None:
        """
        插入或更新数据点

        Args:
            collection_name: 集合名称
            texts: 文本列表
            embeddings: 向量列表
            metadata: 元数据列表
        """
        points = []
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            payload = {"text": text}
            if metadata and i < len(metadata):
                payload.update(metadata[i])

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload=payload
                )
            )

        # 批量插入
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch
            )

        print(f"已插入 {len(points)} 个数据点到 {collection_name}")

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """基础向量搜索"""
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold
        )

        formatted = []
        for hit in results:
            formatted.append({
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
            })

        return formatted

    def search_with_filter(
        self,
        collection_name: str,
        query_vector: List[float],
        filter_conditions: Dict,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        带过滤条件的搜索

        Args:
            filter_conditions: 过滤条件
                示例: {"source": "manual.pdf", "page_min": 1, "page_max": 10}
        """
        must_conditions = []

        if "source" in filter_conditions:
            must_conditions.append(
                FieldCondition(
                    key="source",
                    match=MatchValue(value=filter_conditions["source"])
                )
            )

        if "page_min" in filter_conditions or "page_max" in filter_conditions:
            range_filter = Range(
                gte=filter_conditions.get("page_min"),
                lte=filter_conditions.get("page_max")
            )
            must_conditions.append(
                FieldCondition(key="page_number", range=range_filter)
            )

        search_filter = Filter(must=must_conditions) if must_conditions else None

        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=top_k
        )

        formatted = []
        for hit in results:
            formatted.append({
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
            })

        return formatted

    def delete_collection(self, collection_name: str) -> None:
        """删除集合"""
        self.client.delete_collection(collection_name)
        print(f"集合 {collection_name} 已删除")

    def get_collection_info(self, collection_name: str) -> Dict:
        """获取集合信息"""
        info = self.client.get_collection(collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
            "config": str(info.config)
        }


# 使用示例
if __name__ == "__main__":
    manager = QdrantManager(host="localhost", port=6333)

    # 创建集合
    manager.create_collection("rag_docs", dim=1536, distance="Cosine")

    # 插入数据
    np.random.seed(42)
    texts = [
        "Python是一种编程语言",
        "机器学习模型训练需要大量数据",
        "向量数据库用于存储和检索向量",
        "RAG结合了检索和生成技术",
        "Docker是一种容器化技术"
    ]
    embeddings = np.random.randn(5, 1536).tolist()
    metadata = [
        {"source": "python.pdf", "page_number": 1, "category": "programming"},
        {"source": "ml.pdf", "page_number": 5, "category": "ai"},
        {"source": "vectordb.pdf", "page_number": 3, "category": "database"},
        {"source": "rag.pdf", "page_number": 1, "category": "ai"},
        {"source": "docker.pdf", "page_number": 2, "category": "devops"},
    ]

    manager.upsert_points("rag_docs", texts, embeddings, metadata)

    # 搜索
    query_vec = np.random.randn(1536).tolist()
    results = manager.search("rag_docs", query_vec, top_k=3)
    print("\n搜索结果:")
    for r in results:
        print(f"  Score: {r['score']:.4f}, Text: {r['text']}")

    # 带过滤搜索
    filtered = manager.search_with_filter(
        "rag_docs", query_vec,
        filter_conditions={"source": "ml.pdf"},
        top_k=3
    )
    print("\n过滤搜索 (source=ml.pdf):")
    for r in filtered:
        print(f"  Score: {r['score']:.4f}, Text: {r['text']}")

    # 集合信息
    info = manager.get_collection_info("rag_docs")
    print(f"\n集合信息: {info}")
```

---

## 5. Chroma 轻量级方案

### 5.1 Chroma 简介

Chroma 是一个轻量级的开源向量数据库，专为 AI 应用设计，特别适合：
- 本地开发和原型验证
- 中小规模数据集（< 100万向量）
- 与 LangChain 无缝集成

### 5.2 Chroma 完整使用

```python
"""
Chroma 完整使用教程
安装: pip install chromadb
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import uuid


class ChromaManager:
    """ChromaDB 管理器"""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        mode: str = "persistent"
    ):
        """
        初始化ChromaDB

        Args:
            persist_directory: 持久化存储目录
            mode: "persistent"(持久化) 或 "memory"(内存模式)
        """
        if mode == "persistent":
            self.client = chromadb.PersistentClient(
                path=persist_directory
            )
        elif mode == "memory":
            self.client = chromadb.Client()
        else:
            # 连接到远程Chroma服务器
            self.client = chromadb.HttpClient(
                host="localhost",
                port=8000
            )

        self.collection = None
        print(f"ChromaDB 初始化完成 (模式: {mode})")

    def create_or_get_collection(
        self,
        name: str,
        distance_fn: str = "cosine"
    ):
        """
        创建或获取集合

        Args:
            name: 集合名称
            distance_fn: 距离函数 ("cosine", "l2", "ip")
        """
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": distance_fn}
        )

        count = self.collection.count()
        print(f"集合 '{name}' 就绪, 当前 {count} 条记录")
        return self.collection

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]] = None,
        metadata: List[Dict] = None,
        ids: List[str] = None
    ) -> None:
        """
        添加文档

        Args:
            texts: 文本列表
            embeddings: 向量列表（可选，Chroma可自动生成）
            metadata: 元数据列表
            ids: 文档ID列表
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]

        kwargs = {
            "documents": texts,
            "ids": ids
        }

        if embeddings:
            kwargs["embeddings"] = embeddings
        if metadata:
            kwargs["metadatas"] = metadata

        self.collection.add(**kwargs)
        print(f"已添加 {len(texts)} 条文档")

    def query(
        self,
        query_texts: List[str] = None,
        query_embeddings: List[List[float]] = None,
        n_results: int = 5,
        where: Dict = None,
        where_document: Dict = None
    ) -> Dict[str, Any]:
        """
        查询文档

        Args:
            query_texts: 查询文本（Chroma会自动向量化）
            query_embeddings: 查询向量
            n_results: 返回数量
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
        """
        kwargs = {
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }

        if query_texts:
            kwargs["query_texts"] = query_texts
        elif query_embeddings:
            kwargs["query_embeddings"] = query_embeddings

        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        results = self.collection.query(**kwargs)
        return results

    def update_documents(
        self,
        ids: List[str],
        texts: List[str] = None,
        embeddings: List[List[float]] = None,
        metadata: List[Dict] = None
    ) -> None:
        """更新文档"""
        kwargs = {"ids": ids}
        if texts:
            kwargs["documents"] = texts
        if embeddings:
            kwargs["embeddings"] = embeddings
        if metadata:
            kwargs["metadatas"] = metadata

        self.collection.update(**kwargs)
        print(f"已更新 {len(ids)} 条文档")

    def delete_documents(self, ids: List[str] = None, where: Dict = None) -> None:
        """删除文档"""
        kwargs = {}
        if ids:
            kwargs["ids"] = ids
        if where:
            kwargs["where"] = where

        self.collection.delete(**kwargs)
        print("文档已删除")

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "collection_name": self.collection.name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata
        }


# 使用示例
if __name__ == "__main__":
    # 初始化
    manager = ChromaManager(persist_directory="./my_chroma_db")

    # 创建集合
    manager.create_or_get_collection("documents", distance_fn="cosine")

    # 添加文档（Chroma会自动使用内置Embedding模型）
    manager.add_documents(
        texts=[
            "Python是一种通用编程语言",
            "机器学习需要大量标注数据",
            "向量数据库专门用于语义搜索",
            "Docker简化了应用部署流程",
            "RAG技术增强了LLM的回答质量"
        ],
        metadata=[
            {"source": "python.pdf", "category": "programming"},
            {"source": "ml.pdf", "category": "ai"},
            {"source": "vectordb.pdf", "category": "database"},
            {"source": "docker.pdf", "category": "devops"},
            {"source": "rag.pdf", "category": "ai"},
        ]
    )

    # 文本查询（自动向量化）
    results = manager.query(
        query_texts=["什么是向量数据库"],
        n_results=3
    )

    print("\n查询结果:")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"  [{i+1}] 距离: {dist:.4f} | {doc} | 来源: {meta['source']}")

    # 带过滤的查询
    filtered = manager.query(
        query_texts=["AI技术"],
        n_results=3,
        where={"category": "ai"}
    )

    print("\n过滤查询 (category=ai):")
    for doc, meta in zip(filtered["documents"][0], filtered["metadatas"][0]):
        print(f"  {doc} | 来源: {meta['source']}")
```

### 5.3 Chroma + LangChain 集成

```python
"""
Chroma 与 LangChain 的深度集成
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List


def create_langchain_chroma(
    documents: List[Document],
    persist_directory: str = "./langchain_chroma",
    collection_name: str = "langchain_docs",
    embedding_model: str = "text-embedding-3-small"
) -> Chroma:
    """使用LangChain创建Chroma向量存储"""
    embeddings = OpenAIEmbeddings(model=embedding_model)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    print(f"LangChain Chroma 创建完成: {len(documents)} 个文档")
    return vectorstore


def search_with_langchain(
    vectorstore: Chroma,
    query: str,
    k: int = 4,
    search_type: str = "similarity"
) -> List[Document]:
    """
    使用LangChain进行搜索

    search_type: "similarity" | "mmr" | "similarity_score_threshold"
    """
    if search_type == "similarity":
        results = vectorstore.similarity_search(query, k=k)
    elif search_type == "mmr":
        # MMR (Maximal Marginal Relevance) - 兼顾相关性和多样性
        results = vectorstore.max_marginal_relevance_search(
            query, k=k, fetch_k=20
        )
    elif search_type == "similarity_score_threshold":
        results_with_scores = vectorstore.similarity_search_with_relevance_scores(
            query, k=k, score_threshold=0.5
        )
        results = [doc for doc, score in results_with_scores]
    else:
        results = vectorstore.similarity_search(query, k=k)

    return results


# 使用示例
if __name__ == "__main__":
    # 准备文档
    docs = [
        Document(
            page_content="Python 3.12引入了新的类型提示语法",
            metadata={"source": "python_news.md", "category": "programming"}
        ),
        Document(
            page_content="LangChain v0.2 采用了全新的LCEL表达式语言",
            metadata={"source": "langchain_docs.md", "category": "ai"}
        ),
        Document(
            page_content="ChromaDB 是最流行的开源向量数据库之一",
            metadata={"source": "vectordb_guide.md", "category": "database"}
        ),
    ]

    # 创建向量存储
    vectorstore = create_langchain_chroma(docs)

    # 各种搜索方式
    print("--- 相似度搜索 ---")
    for doc in search_with_langchain(vectorstore, "LangChain怎么用"):
        print(f"  {doc.page_content[:60]}...")

    print("\n--- MMR搜索（多样性） ---")
    for doc in search_with_langchain(vectorstore, "AI开发", search_type="mmr"):
        print(f"  {doc.page_content[:60]}...")
```

---

## 6. Pinecone 云服务

### 6.1 Pinecone 概览

Pinecone 是一个全托管的向量数据库云服务，用户无需管理基础设施。

```bash
# 安装
pip install pinecone-client
```

### 6.2 Pinecone 完整使用

```python
"""
Pinecone 云向量数据库教程
安装: pip install pinecone-client
"""

from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
import os
import numpy as np


class PineconeManager:
    """Pinecone 向量数据库管理器"""

    def __init__(self, api_key: str = None):
        """初始化 Pinecone 客户端"""
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None
        print("Pinecone 客户端初始化完成")

    def create_index(
        self,
        index_name: str,
        dimension: int = 1536,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1"
    ) -> None:
        """
        创建 Serverless 索引

        Args:
            index_name: 索引名称
            dimension: 向量维度
            metric: 度量方式 ("cosine", "euclidean", "dotproduct")
            cloud: 云提供商
            region: 区域
        """
        existing = [idx.name for idx in self.pc.list_indexes()]

        if index_name not in existing:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
            print(f"索引 {index_name} 创建中...")
        else:
            print(f"索引 {index_name} 已存在")

        self.index = self.pc.Index(index_name)

    def upsert_vectors(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict] = None,
        namespace: str = ""
    ) -> None:
        """
        插入或更新向量

        Args:
            ids: ID列表
            vectors: 向量列表
            metadata: 元数据列表
            namespace: 命名空间（用于数据隔离）
        """
        records = []
        for i, (id_, vec) in enumerate(zip(ids, vectors)):
            record = {"id": id_, "values": vec}
            if metadata and i < len(metadata):
                record["metadata"] = metadata[i]
            records.append(record)

        # 批量upsert
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            self.index.upsert(
                vectors=batch,
                namespace=namespace
            )

        print(f"已upsert {len(records)} 条向量")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: str = "",
        filter_dict: Dict = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        搜索最相似的向量

        Args:
            query_vector: 查询向量
            top_k: 返回数量
            namespace: 命名空间
            filter_dict: 过滤条件
            include_metadata: 是否返回元数据
        """
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter_dict,
            include_metadata=include_metadata
        )

        formatted = []
        for match in results.matches:
            formatted.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata if include_metadata else {}
            })

        return formatted

    def delete_vectors(
        self,
        ids: List[str] = None,
        namespace: str = "",
        delete_all: bool = False,
        filter_dict: Dict = None
    ) -> None:
        """删除向量"""
        if delete_all:
            self.index.delete(delete_all=True, namespace=namespace)
        elif filter_dict:
            self.index.delete(filter=filter_dict, namespace=namespace)
        elif ids:
            self.index.delete(ids=ids, namespace=namespace)

        print("向量已删除")

    def get_stats(self) -> Dict:
        """获取索引统计信息"""
        stats = self.index.describe_index_stats()
        return {
            "total_vector_count": stats.total_vector_count,
            "dimension": stats.dimension,
            "namespaces": dict(stats.namespaces)
        }


# 使用示例
if __name__ == "__main__":
    manager = PineconeManager()

    # 创建索引
    manager.create_index(
        index_name="rag-knowledge-base",
        dimension=1536,
        metric="cosine"
    )

    # 插入数据
    np.random.seed(42)
    ids = [f"doc_{i}" for i in range(10)]
    vectors = np.random.randn(10, 1536).tolist()
    metadata = [
        {"text": f"文档{i}的内容", "source": f"file_{i}.pdf", "page": i}
        for i in range(10)
    ]

    manager.upsert_vectors(ids, vectors, metadata)

    # 搜索
    query = np.random.randn(1536).tolist()
    results = manager.search(query, top_k=3)

    print("\n搜索结果:")
    for r in results:
        print(f"  ID: {r['id']}, Score: {r['score']:.4f}, "
              f"Metadata: {r['metadata']}")

    # 带过滤搜索
    filtered = manager.search(
        query, top_k=3,
        filter_dict={"page": {"$lt": 5}}
    )

    print("\n过滤搜索 (page < 5):")
    for r in filtered:
        print(f"  ID: {r['id']}, Score: {r['score']:.4f}")

    # 统计
    stats = manager.get_stats()
    print(f"\n索引统计: {stats}")
```

---

## 7. 性能对比与选型

### 7.1 四大向量数据库性能对比

```
┌──────────────────────────────────────────────────────────────────────┐
│                    向量数据库性能对比                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  QPS (每秒查询数) - 100万向量, 1536维, Top-10                        │
│                                                                      │
│  Milvus   ████████████████████████████████████  ~3000 QPS            │
│  Qdrant   ██████████████████████████████████    ~2800 QPS            │
│  Pinecone ████████████████████████████          ~2200 QPS            │
│  Chroma   ██████████████                        ~1200 QPS            │
│                                                                      │
│  P99延迟 (毫秒)                                                      │
│  Milvus   ████  ~5ms                                                 │
│  Qdrant   █████  ~6ms                                                │
│  Pinecone ████████████  ~15ms (含网络)                               │
│  Chroma   ████████  ~10ms                                            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.2 详细对比表

| 特性 | Milvus | Qdrant | Chroma | Pinecone |
|------|--------|--------|--------|----------|
| **部署方式** | 自部署/云 | 自部署/云 | 嵌入式/自部署 | 全托管云 |
| **开源协议** | Apache 2.0 | Apache 2.0 | Apache 2.0 | 闭源 |
| **编程语言** | Go/C++ | Rust | Python | N/A |
| **最大向量数** | 数十亿 | 数亿 | 数百万 | 数十亿 |
| **分布式** | 原生支持 | 支持 | 不支持 | 自动 |
| **GPU加速** | 支持 | 不支持 | 不支持 | N/A |
| **过滤搜索** | 支持 | 优秀 | 基础 | 支持 |
| **多向量** | 支持 | 支持 | 不支持 | 支持 |
| **混合搜索** | Dense+Sparse | Dense+Sparse | 仅Dense | Dense+Sparse |
| **QPS (100万)** | ~3000 | ~2800 | ~1200 | ~2200 |
| **P99延迟** | ~5ms | ~6ms | ~10ms | ~15ms |
| **内存占用/百万** | ~2GB | ~1.5GB | ~3GB | N/A |
| **LangChain集成** | 完善 | 完善 | 最佳 | 完善 |
| **学习曲线** | 中等 | 低 | 最低 | 低 |
| **社区活跃度** | 高 | 高 | 高 | 中 |
| **免费额度** | 开源免费 | 开源免费 | 开源免费 | 有限免费 |

### 7.3 选型决策树

```
┌──────────────────────────────────────────────────────────────────┐
│                    向量数据库选型决策树                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    需要向量数据库?                                 │
│                         │                                        │
│                         v                                        │
│                  数据规模多大?                                     │
│                  /      |       \                                 │
│                 v       v        v                                │
│            < 10万    10万-1000万   > 1000万                       │
│              │         │            │                             │
│              v         v            v                             │
│          Chroma    是否需要       是否需要                         │
│        (开发/原型)  云托管?       分布式?                          │
│                   /      \      /      \                         │
│                  v        v    v        v                         │
│               Pinecone  Qdrant  Milvus  Pinecone                │
│              (零运维)  (自部署)  (分布式) (托管)                   │
│                                                                  │
│  快速决策:                                                        │
│  - 个人项目/原型 ────────> Chroma                                │
│  - 中小企业/自部署 ──────> Qdrant                                │
│  - 大规模/分布式 ────────> Milvus                                │
│  - 零运维/快速上线 ──────> Pinecone                              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 7.4 不同场景推荐

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| 个人学习/原型验证 | Chroma | 零配置、嵌入式、LangChain原生支持 |
| 初创公司MVP | Qdrant | 轻量级但功能完整、Docker一键部署 |
| 企业知识库 (< 500万) | Qdrant | 过滤能力强、性能好、运维简单 |
| 企业级大规模 (千万+) | Milvus | 原生分布式、GPU加速、高吞吐 |
| 不想运维基础设施 | Pinecone | 全托管、自动扩缩容、免运维 |
| 多模态搜索 | Milvus | 支持多种向量类型和混合搜索 |

---

## 8. LangChain 集成实战

### 8.1 统一的向量存储接口

```python
"""
LangChain 统一向量存储接口
支持在不同向量数据库间无缝切换
"""

from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from typing import List, Optional
import os


class UnifiedVectorStore:
    """统一向量存储接口，支持多种后端"""

    def __init__(
        self,
        backend: str = "chroma",
        embedding_model: str = "text-embedding-3-small",
        **kwargs
    ):
        """
        初始化统一向量存储

        Args:
            backend: 后端选择 ("chroma", "milvus", "qdrant", "pinecone")
            embedding_model: Embedding模型
            **kwargs: 各后端特定参数
        """
        self.backend = backend
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vectorstore = None
        self.kwargs = kwargs

    def create_from_documents(
        self,
        documents: List[Document],
        collection_name: str = "default"
    ):
        """从文档创建向量存储"""
        if self.backend == "chroma":
            from langchain_community.vectorstores import Chroma
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.kwargs.get("persist_directory", "./chroma_db"),
                collection_name=collection_name
            )

        elif self.backend == "milvus":
            from langchain_community.vectorstores import Milvus
            self.vectorstore = Milvus.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name,
                connection_args={
                    "host": self.kwargs.get("host", "localhost"),
                    "port": self.kwargs.get("port", 19530)
                }
            )

        elif self.backend == "qdrant":
            from langchain_community.vectorstores import Qdrant
            self.vectorstore = Qdrant.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name,
                url=self.kwargs.get("url", "http://localhost:6333")
            )

        elif self.backend == "pinecone":
            from langchain_pinecone import PineconeVectorStore
            self.vectorstore = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                index_name=collection_name
            )

        print(f"向量存储创建完成 (后端: {self.backend}, 文档数: {len(documents)})")
        return self.vectorstore

    def as_retriever(self, search_type: str = "similarity", k: int = 4):
        """获取检索器"""
        if self.vectorstore is None:
            raise ValueError("请先调用 create_from_documents()")

        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """相似度搜索"""
        return self.vectorstore.similarity_search(query, k=k)

    def mmr_search(self, query: str, k: int = 4, fetch_k: int = 20) -> List[Document]:
        """MMR多样性搜索"""
        return self.vectorstore.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k
        )


# 使用示例: 一键切换后端
if __name__ == "__main__":
    docs = [
        Document(page_content="RAG是检索增强生成技术", metadata={"source": "rag.md"}),
        Document(page_content="向量数据库用于存储嵌入向量", metadata={"source": "db.md"}),
        Document(page_content="LangChain是LLM应用开发框架", metadata={"source": "lc.md"}),
    ]

    # 使用Chroma后端
    store = UnifiedVectorStore(
        backend="chroma",
        persist_directory="./unified_chroma"
    )
    store.create_from_documents(docs, "test_collection")

    # 搜索
    results = store.similarity_search("什么是RAG?", k=2)
    for doc in results:
        print(f"[Chroma] {doc.page_content}")

    # 切换到Qdrant只需改变backend参数
    # store = UnifiedVectorStore(
    #     backend="qdrant",
    #     url="http://localhost:6333"
    # )
```

---

## 9. 最佳实践

### 9.1 向量数据库使用建议

```
┌──────────────────────────────────────────────────────────────────┐
│                   向量数据库最佳实践清单                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  数据准备:                                                        │
│  [x] 选择合适的Embedding模型（中文推荐BGE系列）                     │
│  [x] 文本分块大小通常500-1000字符                                  │
│  [x] 添加丰富的元数据（来源、时间、类别）                           │
│  [x] 向量归一化（如使用IP距离度量）                                │
│                                                                  │
│  索引配置:                                                        │
│  [x] < 10万向量: FLAT 或 HNSW (低M值)                            │
│  [x] 10万-1000万: HNSW (M=16, ef=200)                           │
│  [x] > 1000万: IVF_FLAT 或 IVF_PQ                               │
│  [x] 定期重建索引以保持性能                                       │
│                                                                  │
│  查询优化:                                                        │
│  [x] 合理设置 top_k（通常4-10）                                   │
│  [x] 使用过滤条件缩小搜索范围                                     │
│  [x] 结合Reranker提高精度                                        │
│  [x] 考虑MMR保证结果多样性                                       │
│                                                                  │
│  运维监控:                                                        │
│  [x] 监控查询延迟和QPS                                           │
│  [x] 定期备份数据                                                 │
│  [x] 监控内存使用情况                                             │
│  [x] 设置合理的集合数量限制                                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 9.2 常见陷阱

| 陷阱 | 说明 | 解决方案 |
|------|------|----------|
| 向量维度不匹配 | 查询和文档使用不同的Embedding模型 | 统一使用同一个模型 |
| 分块过大或过小 | 影响检索精度和召回率 | 实验确定最佳大小 |
| 忽略元数据过滤 | 搜索全部数据导致噪声多 | 利用元数据缩小范围 |
| 未使用批量操作 | 逐条插入效率极低 | 批量插入（batch 100-1000） |
| 索引参数未调优 | 默认参数不适合所有场景 | 根据数据规模调整参数 |

---

## 总结

本教程完整介绍了向量数据库的核心知识：

1. **原理基础**：向量表示、相似度度量、ANN 算法（HNSW/IVF/PQ）
2. **Milvus**：企业级分布式向量数据库，适合大规模场景
3. **Qdrant**：高性能 Rust 实现，过滤能力强
4. **Chroma**：轻量级嵌入式方案，开发友好
5. **Pinecone**：全托管云服务，零运维
6. **性能对比**：QPS、延迟、内存等维度详细对比
7. **选型指南**：根据规模、预算、运维能力选择合适方案

## 参考资源

- [Milvus 官方文档](https://milvus.io/docs)
- [Qdrant 官方文档](https://qdrant.tech/documentation/)
- [Chroma 官方文档](https://docs.trychroma.com/)
- [Pinecone 官方文档](https://docs.pinecone.io/)
- [LangChain Vector Stores](https://python.langchain.com/docs/integrations/vectorstores/)
- [ANN Benchmarks](https://ann-benchmarks.com/)

---

**创建时间**: 2024-01
**最后更新**: 2024-01
