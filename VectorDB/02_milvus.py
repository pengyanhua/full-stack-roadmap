"""
============================================================
                Milvus 向量数据库实战
============================================================
Milvus 是一个开源的向量数据库，支持大规模向量相似度搜索

安装：
pip install pymilvus

启动 Milvus（Docker）：
docker-compose up -d

或使用 Milvus Lite（嵌入式）：
pip install milvus
============================================================
"""

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    MilvusClient
)
import numpy as np
from typing import List, Dict, Any, Optional
import json


# ============================================================
#                    一、基础连接
# ============================================================

def connect_milvus(host: str = "localhost", port: int = 19530):
    """连接 Milvus 服务器"""
    connections.connect(
        alias="default",
        host=host,
        port=port
    )
    print(f"已连接到 Milvus {host}:{port}")


def disconnect_milvus():
    """断开连接"""
    connections.disconnect("default")
    print("已断开 Milvus 连接")


# ============================================================
#                    二、集合管理
# ============================================================

class MilvusManager:
    """Milvus 集合管理器"""

    def __init__(self, collection_name: str, dim: int = 768):
        self.collection_name = collection_name
        self.dim = dim
        self.collection = None

    def create_collection(
        self,
        description: str = "",
        index_type: str = "HNSW",
        metric_type: str = "COSINE"
    ):
        """
        创建集合

        Args:
            description: 集合描述
            index_type: 索引类型 (FLAT, IVF_FLAT, IVF_SQ8, IVF_PQ, HNSW)
            metric_type: 距离度量 (L2, IP, COSINE)
        """
        # 检查是否存在
        if utility.has_collection(self.collection_name):
            print(f"集合 {self.collection_name} 已存在")
            self.collection = Collection(self.collection_name)
            return

        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="created_at", dtype=DataType.INT64),
        ]

        # 创建 Schema
        schema = CollectionSchema(
            fields=fields,
            description=description,
            enable_dynamic_field=True  # 允许动态字段
        )

        # 创建集合
        self.collection = Collection(
            name=self.collection_name,
            schema=schema
        )

        # 创建索引
        index_params = self._get_index_params(index_type, metric_type)
        self.collection.create_index(
            field_name="vector",
            index_params=index_params
        )

        print(f"集合 {self.collection_name} 创建成功")

    def _get_index_params(self, index_type: str, metric_type: str) -> Dict:
        """获取索引参数"""
        params_map = {
            "FLAT": {},
            "IVF_FLAT": {"nlist": 1024},
            "IVF_SQ8": {"nlist": 1024},
            "IVF_PQ": {"nlist": 1024, "m": 8, "nbits": 8},
            "HNSW": {"M": 16, "efConstruction": 200},
        }

        return {
            "index_type": index_type,
            "metric_type": metric_type,
            "params": params_map.get(index_type, {})
        }

    def load_collection(self):
        """加载集合到内存"""
        if self.collection is None:
            self.collection = Collection(self.collection_name)
        self.collection.load()
        print(f"集合 {self.collection_name} 已加载到内存")

    def release_collection(self):
        """释放集合"""
        if self.collection:
            self.collection.release()
            print(f"集合 {self.collection_name} 已释放")

    def drop_collection(self):
        """删除集合"""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            self.collection = None
            print(f"集合 {self.collection_name} 已删除")

    def get_stats(self) -> Dict:
        """获取集合统计信息"""
        if self.collection is None:
            self.collection = Collection(self.collection_name)

        self.collection.flush()
        return {
            "name": self.collection_name,
            "num_entities": self.collection.num_entities,
            "schema": str(self.collection.schema)
        }


# ============================================================
#                    三、数据操作
# ============================================================

class MilvusVectorStore:
    """Milvus 向量存储"""

    def __init__(self, collection_name: str, dim: int = 768):
        self.collection_name = collection_name
        self.dim = dim
        self.manager = MilvusManager(collection_name, dim)

    def init(self, index_type: str = "HNSW", metric_type: str = "COSINE"):
        """初始化"""
        self.manager.create_collection(index_type=index_type, metric_type=metric_type)
        self.manager.load_collection()

    def insert(
        self,
        ids: List[str],
        texts: List[str],
        vectors: List[List[float]],
        metadata: List[Dict] = None,
        timestamps: List[int] = None
    ) -> int:
        """
        插入数据

        Args:
            ids: ID 列表
            texts: 文本列表
            vectors: 向量列表
            metadata: 元数据列表
            timestamps: 时间戳列表

        Returns:
            插入数量
        """
        import time

        if metadata is None:
            metadata = [{}] * len(ids)
        if timestamps is None:
            timestamps = [int(time.time())] * len(ids)

        data = [
            ids,
            texts,
            vectors,
            metadata,
            timestamps
        ]

        self.manager.collection.insert(data)
        self.manager.collection.flush()

        return len(ids)

    def search(
        self,
        query_vectors: List[List[float]],
        top_k: int = 10,
        filter_expr: str = None,
        output_fields: List[str] = None
    ) -> List[List[Dict]]:
        """
        向量搜索

        Args:
            query_vectors: 查询向量列表
            top_k: 返回数量
            filter_expr: 过滤表达式
            output_fields: 返回字段

        Returns:
            搜索结果列表
        """
        if output_fields is None:
            output_fields = ["id", "text", "metadata"]

        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 100}  # HNSW 搜索参数
        }

        results = self.manager.collection.search(
            data=query_vectors,
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=output_fields
        )

        # 格式化结果
        formatted_results = []
        for hits in results:
            hits_list = []
            for hit in hits:
                hits_list.append({
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.entity.get("text"),
                    "metadata": hit.entity.get("metadata")
                })
            formatted_results.append(hits_list)

        return formatted_results

    def delete(self, ids: List[str]) -> int:
        """删除数据"""
        expr = f'id in {json.dumps(ids)}'
        result = self.manager.collection.delete(expr)
        self.manager.collection.flush()
        return result.delete_count

    def query(
        self,
        filter_expr: str,
        output_fields: List[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        条件查询（非向量）

        Args:
            filter_expr: 过滤表达式
            output_fields: 返回字段
            limit: 返回数量
        """
        if output_fields is None:
            output_fields = ["id", "text", "metadata"]

        results = self.manager.collection.query(
            expr=filter_expr,
            output_fields=output_fields,
            limit=limit
        )

        return results


# ============================================================
#                    四、RAG 应用示例
# ============================================================

class MilvusRAG:
    """
    基于 Milvus 的 RAG 系统

    集成向量存储和检索增强生成
    """

    def __init__(
        self,
        collection_name: str,
        embedding_model: Any = None,
        llm: Any = None,
        dim: int = 768
    ):
        self.store = MilvusVectorStore(collection_name, dim)
        self.embedding_model = embedding_model
        self.llm = llm
        self.dim = dim

    def init(self):
        """初始化"""
        self.store.init(index_type="HNSW", metric_type="COSINE")

    def add_documents(
        self,
        documents: List[str],
        ids: List[str] = None,
        metadata: List[Dict] = None
    ):
        """添加文档"""
        import uuid

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        # 生成嵌入向量
        if self.embedding_model:
            vectors = self.embedding_model.encode(documents).tolist()
        else:
            # 模拟向量（实际使用时需要真实的嵌入模型）
            vectors = [np.random.rand(self.dim).tolist() for _ in documents]

        self.store.insert(ids, documents, vectors, metadata)
        print(f"已添加 {len(documents)} 个文档")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_expr: str = None
    ) -> List[Dict]:
        """搜索相关文档"""
        # 生成查询向量
        if self.embedding_model:
            query_vector = self.embedding_model.encode([query]).tolist()
        else:
            query_vector = [np.random.rand(self.dim).tolist()]

        results = self.store.search(query_vector, top_k, filter_expr)
        return results[0] if results else []

    def query(
        self,
        question: str,
        top_k: int = 5,
        system_prompt: str = None
    ) -> str:
        """
        RAG 查询

        1. 检索相关文档
        2. 构造提示词
        3. 调用 LLM 生成回答
        """
        # 检索
        docs = self.search(question, top_k)

        # 构造上下文
        context = "\n\n".join([
            f"[文档{i+1}] (相似度: {doc['score']:.4f})\n{doc['text']}"
            for i, doc in enumerate(docs)
        ])

        # 构造提示词
        if system_prompt is None:
            system_prompt = "你是一个有帮助的助手。请根据给定的参考资料回答问题。如果资料中没有相关信息，请说明。"

        prompt = f"""{system_prompt}

参考资料：
{context}

问题：{question}

回答："""

        # 调用 LLM
        if self.llm:
            answer = self.llm.generate(prompt)
        else:
            answer = f"[模拟回答] 根据检索到的 {len(docs)} 个相关文档，问题是关于 '{question}' 的。"

        return answer


# ============================================================
#                    五、高级功能
# ============================================================

class MilvusHybridSearch:
    """混合搜索（向量 + 标量）"""

    def __init__(self, collection_name: str, dim: int = 768):
        self.store = MilvusVectorStore(collection_name, dim)

    def hybrid_search(
        self,
        query_vector: List[float],
        filters: Dict[str, Any],
        top_k: int = 10
    ) -> List[Dict]:
        """
        混合搜索

        Args:
            query_vector: 查询向量
            filters: 过滤条件
            top_k: 返回数量

        示例：
            filters = {
                "category": "tech",
                "date_range": ("2024-01-01", "2024-12-31"),
                "tags": ["AI", "ML"]
            }
        """
        # 构建过滤表达式
        expr_parts = []

        for key, value in filters.items():
            if key.endswith("_range") and isinstance(value, tuple):
                field = key.replace("_range", "")
                expr_parts.append(f'{field} >= "{value[0]}" && {field} <= "{value[1]}"')
            elif isinstance(value, list):
                # 数组包含
                values_str = ", ".join([f'"{v}"' for v in value])
                expr_parts.append(f'{key} in [{values_str}]')
            elif isinstance(value, str):
                expr_parts.append(f'{key} == "{value}"')
            else:
                expr_parts.append(f'{key} == {value}')

        filter_expr = " && ".join(expr_parts) if expr_parts else None

        return self.store.search([query_vector], top_k, filter_expr)[0]


class MilvusBatchProcessor:
    """批量处理器"""

    def __init__(self, store: MilvusVectorStore, batch_size: int = 1000):
        self.store = store
        self.batch_size = batch_size
        self.buffer = {"ids": [], "texts": [], "vectors": [], "metadata": []}

    def add(self, id: str, text: str, vector: List[float], metadata: Dict = None):
        """添加到缓冲区"""
        self.buffer["ids"].append(id)
        self.buffer["texts"].append(text)
        self.buffer["vectors"].append(vector)
        self.buffer["metadata"].append(metadata or {})

        if len(self.buffer["ids"]) >= self.batch_size:
            self.flush()

    def flush(self):
        """刷新缓冲区"""
        if not self.buffer["ids"]:
            return

        self.store.insert(
            self.buffer["ids"],
            self.buffer["texts"],
            self.buffer["vectors"],
            self.buffer["metadata"]
        )

        # 清空缓冲区
        self.buffer = {"ids": [], "texts": [], "vectors": [], "metadata": []}
        print(f"批量插入完成")


# ============================================================
#                    六、使用 MilvusClient（简化 API）
# ============================================================

class SimpleMilvus:
    """
    使用 MilvusClient 的简化封装

    MilvusClient 是 Milvus 2.x 提供的简化 API
    """

    def __init__(self, uri: str = "http://localhost:19530"):
        """
        Args:
            uri: Milvus 连接地址
                 - 本地文件: "./milvus.db"
                 - 服务器: "http://localhost:19530"
        """
        self.client = MilvusClient(uri=uri)

    def create_collection(self, name: str, dim: int = 768):
        """创建集合"""
        if self.client.has_collection(name):
            print(f"集合 {name} 已存在")
            return

        self.client.create_collection(
            collection_name=name,
            dimension=dim,
            metric_type="COSINE"
        )
        print(f"集合 {name} 创建成功")

    def insert(self, collection_name: str, data: List[Dict]):
        """
        插入数据

        data 格式:
        [
            {"id": 1, "vector": [...], "text": "...", "subject": "..."},
            ...
        ]
        """
        result = self.client.insert(
            collection_name=collection_name,
            data=data
        )
        return result

    def search(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        top_k: int = 10,
        filter_expr: str = None,
        output_fields: List[str] = None
    ):
        """搜索"""
        results = self.client.search(
            collection_name=collection_name,
            data=query_vectors,
            limit=top_k,
            filter=filter_expr,
            output_fields=output_fields or ["text"]
        )
        return results

    def query(
        self,
        collection_name: str,
        filter_expr: str,
        output_fields: List[str] = None
    ):
        """条件查询"""
        return self.client.query(
            collection_name=collection_name,
            filter=filter_expr,
            output_fields=output_fields
        )

    def delete(self, collection_name: str, ids: List):
        """删除"""
        return self.client.delete(
            collection_name=collection_name,
            ids=ids
        )


# ============================================================
#                    主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Milvus 向量数据库示例")
    print("=" * 60)

    # 使用 MilvusClient（简化 API）
    # 本地文件模式，无需启动服务
    print("\n--- 使用 MilvusClient ---")

    try:
        # 创建客户端（使用本地文件）
        client = SimpleMilvus(uri="./milvus_demo.db")

        # 创建集合
        collection_name = "demo_collection"
        dim = 384

        client.create_collection(collection_name, dim)

        # 准备数据
        np.random.seed(42)
        data = [
            {
                "id": i,
                "vector": np.random.rand(dim).tolist(),
                "text": f"这是第 {i} 个文档的内容",
                "category": "tech" if i % 2 == 0 else "news"
            }
            for i in range(100)
        ]

        # 插入数据
        client.insert(collection_name, data)
        print(f"插入 {len(data)} 条数据")

        # 搜索
        query_vector = np.random.rand(dim).tolist()
        results = client.search(
            collection_name,
            [query_vector],
            top_k=5,
            output_fields=["text", "category"]
        )

        print("\n搜索结果:")
        for hits in results:
            for hit in hits:
                print(f"  ID: {hit['id']}, 距离: {hit['distance']:.4f}")

        # 条件查询
        filtered = client.query(
            collection_name,
            filter_expr='category == "tech"',
            output_fields=["id", "text"]
        )
        print(f"\n筛选 category='tech': 共 {len(filtered)} 条")

        print("\n示例完成！")

    except Exception as e:
        print(f"错误: {e}")
        print("请确保已安装 pymilvus: pip install pymilvus")
