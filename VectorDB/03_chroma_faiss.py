"""
============================================================
            Chroma & Faiss 向量数据库实战
============================================================
Chroma: 轻量级嵌入式向量数据库，Python 友好
Faiss: Facebook 的高性能向量索引库

安装：
pip install chromadb
pip install faiss-cpu  # 或 faiss-gpu
============================================================
"""

import numpy as np
from typing import List, Dict, Any, Optional
import json


# ============================================================
#                    一、Chroma 基础
# ============================================================

"""
Chroma 是一个开源的嵌入式向量数据库，特点：
- 极简 API
- 内置持久化
- 支持元数据过滤
- Python 原生
"""

# pip install chromadb

class ChromaVectorStore:
    """Chroma 向量存储封装"""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str = None
    ):
        """
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录，None 表示内存模式
        """
        import chromadb
        from chromadb.config import Settings

        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.collection_name = collection_name
        self.collection = None

    def create_collection(
        self,
        embedding_function: Any = None,
        distance_fn: str = "cosine"
    ):
        """
        创建或获取集合

        Args:
            embedding_function: 嵌入函数（可选，Chroma 可自动嵌入）
            distance_fn: 距离函数 (cosine, l2, ip)
        """
        metadata = {"hnsw:space": distance_fn}

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embedding_function,
            metadata=metadata
        )

        print(f"集合 {self.collection_name} 已就绪")

    def add(
        self,
        ids: List[str],
        documents: List[str] = None,
        embeddings: List[List[float]] = None,
        metadatas: List[Dict] = None
    ):
        """
        添加数据

        可以只提供 documents（Chroma 自动生成嵌入）
        或直接提供 embeddings
        """
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"已添加 {len(ids)} 条数据")

    def query(
        self,
        query_texts: List[str] = None,
        query_embeddings: List[List[float]] = None,
        n_results: int = 10,
        where: Dict = None,
        where_document: Dict = None
    ) -> Dict:
        """
        查询

        Args:
            query_texts: 查询文本（使用集合的嵌入函数）
            query_embeddings: 查询向量
            n_results: 返回数量
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
        """
        results = self.collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"]
        )

        return results

    def update(
        self,
        ids: List[str],
        documents: List[str] = None,
        embeddings: List[List[float]] = None,
        metadatas: List[Dict] = None
    ):
        """更新数据"""
        self.collection.update(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def delete(self, ids: List[str] = None, where: Dict = None):
        """删除数据"""
        self.collection.delete(ids=ids, where=where)

    def count(self) -> int:
        """获取数据数量"""
        return self.collection.count()

    def get(
        self,
        ids: List[str] = None,
        where: Dict = None,
        limit: int = None
    ) -> Dict:
        """获取数据"""
        return self.collection.get(
            ids=ids,
            where=where,
            limit=limit,
            include=["documents", "metadatas", "embeddings"]
        )


# Chroma 过滤语法示例
CHROMA_FILTER_EXAMPLES = """
# where 条件（元数据过滤）
{
    "category": "tech"                    # 等于
}
{
    "category": {"$ne": "tech"}           # 不等于
}
{
    "price": {"$gt": 100}                 # 大于
}
{
    "price": {"$gte": 100}                # 大于等于
}
{
    "price": {"$lt": 100}                 # 小于
}
{
    "price": {"$lte": 100}                # 小于等于
}
{
    "tags": {"$in": ["AI", "ML"]}         # 包含于
}
{
    "tags": {"$nin": ["spam"]}            # 不包含于
}
{
    "$and": [                              # 与
        {"category": "tech"},
        {"price": {"$gt": 100}}
    ]
}
{
    "$or": [                               # 或
        {"category": "tech"},
        {"category": "news"}
    ]
}

# where_document 条件（文档内容过滤）
{
    "$contains": "机器学习"                # 包含文本
}
{
    "$not_contains": "广告"               # 不包含文本
}
"""


# ============================================================
#                    二、Chroma RAG 示例
# ============================================================

class ChromaRAG:
    """基于 Chroma 的 RAG 系统"""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str = None
    ):
        self.store = ChromaVectorStore(collection_name, persist_directory)

    def init(self):
        """初始化（使用 Chroma 内置嵌入）"""
        self.store.create_collection()

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict] = None,
        ids: List[str] = None
    ):
        """添加文档（Chroma 自动生成嵌入）"""
        import uuid

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        self.store.add(ids=ids, documents=documents, metadatas=metadatas)

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Dict = None
    ) -> List[Dict]:
        """搜索相关文档"""
        results = self.store.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )

        # 格式化结果
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else None,
                "distance": results["distances"][0][i]
            })

        return formatted


# ============================================================
#                    三、Faiss 基础
# ============================================================

"""
Faiss 是 Facebook 开发的高性能向量索引库

特点：
- 极致性能
- 丰富的索引类型
- GPU 加速支持
- 不是完整数据库（需要自行管理元数据）
"""

# pip install faiss-cpu 或 pip install faiss-gpu

class FaissVectorStore:
    """Faiss 向量存储封装"""

    def __init__(self, dim: int, index_type: str = "Flat"):
        """
        Args:
            dim: 向量维度
            index_type: 索引类型
                - Flat: 暴力搜索，精确
                - IVF: 倒排索引
                - HNSW: 层次小世界图
                - PQ: 乘积量化
        """
        import faiss

        self.dim = dim
        self.index_type = index_type
        self.index = None
        self.id_to_metadata = {}  # 自行管理元数据
        self.next_id = 0

        self._create_index(index_type)

    def _create_index(self, index_type: str):
        """创建索引"""
        import faiss

        if index_type == "Flat":
            # 精确搜索（小数据量）
            self.index = faiss.IndexFlatIP(self.dim)  # 内积
            # 或 faiss.IndexFlatL2(self.dim)  # L2 距离

        elif index_type == "IVF":
            # IVF 索引（中等数据量）
            nlist = 100  # 聚类数
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
            self.needs_training = True

        elif index_type == "HNSW":
            # HNSW 索引（推荐）
            M = 32  # 连接数
            self.index = faiss.IndexHNSWFlat(self.dim, M)
            self.index.hnsw.efConstruction = 200

        elif index_type == "IVF_PQ":
            # IVF + PQ（大数据量，节省内存）
            nlist = 100
            m = 8  # 子向量数
            quantizer = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, 8)
            self.needs_training = True

        print(f"创建 Faiss {index_type} 索引，维度: {self.dim}")

    def train(self, vectors: np.ndarray):
        """训练索引（IVF 类型需要）"""
        if hasattr(self, 'needs_training') and self.needs_training:
            if not self.index.is_trained:
                print("训练索引...")
                self.index.train(vectors.astype('float32'))
                print("训练完成")

    def add(
        self,
        vectors: np.ndarray,
        metadatas: List[Dict] = None
    ) -> List[int]:
        """
        添加向量

        Args:
            vectors: 向量数组 (n, dim)
            metadatas: 元数据列表

        Returns:
            分配的 ID 列表
        """
        vectors = np.array(vectors).astype('float32')

        # 归一化（如果使用内积）
        if self.index_type in ["Flat", "IVF", "HNSW"]:
            import faiss
            faiss.normalize_L2(vectors)

        # 训练（如果需要）
        self.train(vectors)

        # 分配 ID
        ids = list(range(self.next_id, self.next_id + len(vectors)))
        self.next_id += len(vectors)

        # 存储元数据
        if metadatas:
            for i, metadata in zip(ids, metadatas):
                self.id_to_metadata[i] = metadata

        # 添加到索引
        self.index.add(vectors)

        return ids

    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 10
    ) -> List[List[Dict]]:
        """
        搜索

        Args:
            query_vectors: 查询向量 (n_queries, dim)
            top_k: 返回数量

        Returns:
            搜索结果列表
        """
        import faiss

        query_vectors = np.array(query_vectors).astype('float32')

        # 归一化
        if self.index_type in ["Flat", "IVF", "HNSW"]:
            faiss.normalize_L2(query_vectors)

        # 设置搜索参数
        if self.index_type == "IVF" or self.index_type == "IVF_PQ":
            self.index.nprobe = 10  # 搜索的聚类数

        if self.index_type == "HNSW":
            self.index.hnsw.efSearch = 100

        # 搜索
        distances, indices = self.index.search(query_vectors, top_k)

        # 格式化结果
        results = []
        for dist_row, idx_row in zip(distances, indices):
            hits = []
            for dist, idx in zip(dist_row, idx_row):
                if idx >= 0:  # -1 表示无结果
                    hits.append({
                        "id": int(idx),
                        "score": float(dist),
                        "metadata": self.id_to_metadata.get(int(idx))
                    })
            results.append(hits)

        return results

    def save(self, path: str):
        """保存索引"""
        import faiss
        faiss.write_index(self.index, f"{path}.index")

        # 保存元数据
        with open(f"{path}.meta", 'w') as f:
            json.dump({
                "id_to_metadata": {str(k): v for k, v in self.id_to_metadata.items()},
                "next_id": self.next_id,
                "dim": self.dim,
                "index_type": self.index_type
            }, f)

        print(f"索引已保存到 {path}")

    @classmethod
    def load(cls, path: str) -> 'FaissVectorStore':
        """加载索引"""
        import faiss

        # 加载元数据
        with open(f"{path}.meta", 'r') as f:
            meta = json.load(f)

        # 创建实例
        instance = cls(meta["dim"], meta["index_type"])
        instance.index = faiss.read_index(f"{path}.index")
        instance.id_to_metadata = {int(k): v for k, v in meta["id_to_metadata"].items()}
        instance.next_id = meta["next_id"]

        print(f"索引已从 {path} 加载")
        return instance

    @property
    def count(self) -> int:
        """向量数量"""
        return self.index.ntotal


# ============================================================
#                    四、Faiss 高级功能
# ============================================================

class FaissIndexFactory:
    """Faiss 索引工厂"""

    @staticmethod
    def create_index(dim: int, n_vectors: int, memory_limit_mb: int = None):
        """
        根据数据量自动选择索引

        Args:
            dim: 向量维度
            n_vectors: 预期向量数量
            memory_limit_mb: 内存限制（MB）
        """
        import faiss

        if n_vectors < 10000:
            # 小数据量：精确搜索
            return faiss.IndexFlatIP(dim)

        elif n_vectors < 1000000:
            # 中等数据量：HNSW
            index = faiss.IndexHNSWFlat(dim, 32)
            index.hnsw.efConstruction = 200
            return index

        else:
            # 大数据量：IVF + PQ
            nlist = int(np.sqrt(n_vectors))
            m = min(dim // 4, 64)  # 子向量数

            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
            return index

    @staticmethod
    def create_gpu_index(cpu_index, gpu_id: int = 0):
        """将索引转移到 GPU"""
        import faiss

        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
        return gpu_index


class FaissClusterIndex:
    """支持按类别过滤的索引"""

    def __init__(self, dim: int):
        self.dim = dim
        self.category_indices = {}  # 每个类别一个索引
        self.category_metadata = {}

    def add(self, vectors: np.ndarray, categories: List[str], metadatas: List[Dict] = None):
        """按类别添加向量"""
        import faiss

        vectors = np.array(vectors).astype('float32')
        faiss.normalize_L2(vectors)

        # 按类别分组
        category_data = {}
        for i, cat in enumerate(categories):
            if cat not in category_data:
                category_data[cat] = {"vectors": [], "metadatas": []}
            category_data[cat]["vectors"].append(vectors[i])
            if metadatas:
                category_data[cat]["metadatas"].append(metadatas[i])

        # 添加到各类别索引
        for cat, data in category_data.items():
            if cat not in self.category_indices:
                self.category_indices[cat] = faiss.IndexFlatIP(self.dim)
                self.category_metadata[cat] = []

            self.category_indices[cat].add(np.array(data["vectors"]))
            self.category_metadata[cat].extend(data["metadatas"])

    def search(
        self,
        query_vector: np.ndarray,
        category: str = None,
        top_k: int = 10
    ) -> List[Dict]:
        """搜索（可按类别过滤）"""
        import faiss

        query_vector = np.array([query_vector]).astype('float32')
        faiss.normalize_L2(query_vector)

        if category:
            # 在指定类别中搜索
            if category not in self.category_indices:
                return []
            index = self.category_indices[category]
            metadata_list = self.category_metadata[category]

            distances, indices = index.search(query_vector, top_k)

            return [
                {"id": int(idx), "score": float(dist), "metadata": metadata_list[idx]}
                for dist, idx in zip(distances[0], indices[0]) if idx >= 0
            ]
        else:
            # 在所有类别中搜索，合并结果
            all_results = []
            for cat in self.category_indices:
                results = self.search(query_vector[0], cat, top_k)
                for r in results:
                    r["category"] = cat
                all_results.extend(results)

            # 按分数排序
            all_results.sort(key=lambda x: x["score"], reverse=True)
            return all_results[:top_k]


# ============================================================
#                    五、对比与选择
# ============================================================

COMPARISON = """
┌─────────────┬───────────────────┬─────────────────┬─────────────────┐
│   特性       │      Chroma       │      Faiss      │     Milvus      │
├─────────────┼───────────────────┼─────────────────┼─────────────────┤
│ 类型        │ 嵌入式数据库      │ 索引库          │ 分布式数据库    │
│ 部署        │ 内嵌/独立         │ 库引入          │ 独立服务        │
│ 学习曲线    │ 简单              │ 中等            │ 中等            │
│ 元数据      │ 内置支持          │ 需自行管理      │ 内置支持        │
│ 过滤        │ 支持              │ 需自行实现      │ 支持            │
│ 持久化      │ 内置              │ 需手动          │ 自动            │
│ 扩展性      │ 单机              │ 单机            │ 分布式          │
│ 性能        │ 一般              │ 极高            │ 高              │
│ GPU         │ 不支持            │ 支持            │ 支持            │
│ 适用场景    │ 原型/小项目       │ 性能关键/大规模 │ 生产环境        │
└─────────────┴───────────────────┴─────────────────┴─────────────────┘

选择建议：
- 快速原型/学习: Chroma
- 最高性能/大数据: Faiss
- 生产环境/需要元数据过滤: Milvus
- 已有 PostgreSQL: pgvector
- 全托管服务: Pinecone
"""


# ============================================================
#                    主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Chroma & Faiss 向量数据库示例")
    print("=" * 60)

    # ---- Chroma 示例 ----
    print("\n--- Chroma 示例 ---")

    try:
        chroma_store = ChromaVectorStore("demo_collection")
        chroma_store.create_collection()

        # 添加数据
        documents = [
            "机器学习是人工智能的一个分支",
            "深度学习使用神经网络进行学习",
            "自然语言处理用于理解人类语言",
            "计算机视觉让机器能够看见",
            "Python 是最流行的编程语言之一"
        ]

        metadatas = [
            {"category": "AI", "topic": "ML"},
            {"category": "AI", "topic": "DL"},
            {"category": "AI", "topic": "NLP"},
            {"category": "AI", "topic": "CV"},
            {"category": "Programming", "topic": "Python"}
        ]

        chroma_store.add(
            ids=[f"doc_{i}" for i in range(len(documents))],
            documents=documents,
            metadatas=metadatas
        )

        # 搜索
        results = chroma_store.query(
            query_texts=["什么是深度学习？"],
            n_results=3
        )

        print("搜索 '什么是深度学习？':")
        for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
            print(f"  {i+1}. (距离: {dist:.4f}) {doc}")

        # 带过滤的搜索
        results_filtered = chroma_store.query(
            query_texts=["编程"],
            n_results=3,
            where={"category": "AI"}
        )

        print("\n搜索 '编程' (过滤 category=AI):")
        for doc in results_filtered["documents"][0]:
            print(f"  - {doc}")

    except ImportError:
        print("请安装 chromadb: pip install chromadb")

    # ---- Faiss 示例 ----
    print("\n--- Faiss 示例 ---")

    try:
        import faiss

        dim = 128
        faiss_store = FaissVectorStore(dim, index_type="HNSW")

        # 生成随机向量
        np.random.seed(42)
        vectors = np.random.rand(1000, dim).astype('float32')
        metadatas = [{"id": i, "label": f"item_{i}"} for i in range(1000)]

        # 添加
        faiss_store.add(vectors, metadatas)
        print(f"添加 {faiss_store.count} 个向量")

        # 搜索
        query = np.random.rand(1, dim).astype('float32')
        results = faiss_store.search(query, top_k=5)

        print("\n搜索结果:")
        for hit in results[0]:
            print(f"  ID: {hit['id']}, Score: {hit['score']:.4f}")

    except ImportError:
        print("请安装 faiss: pip install faiss-cpu")

    print("\n" + "=" * 60)
    print("选择建议")
    print("=" * 60)
    print(COMPARISON)
