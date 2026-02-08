# Embedding与向量表示完整教程

## 目录
1. [Embedding原理](#embedding原理)
2. [OpenAI Embeddings API](#openai-embeddings-api)
3. [Sentence Transformers](#sentence-transformers)
4. [向量相似度计算](#向量相似度计算)
5. [语义搜索实现](#语义搜索实现)
6. [完整代码示例](#完整代码示例)

---

## Embedding原理

### 什么是Embedding

Embedding是将文本转换为高维向量的技术，使得语义相似的文本在向量空间中距离更近。

```
┌────────────────────────────────────────────────────┐
│            文本 → 向量转换过程                       │
├────────────────────────────────────────────────────┤
│                                                    │
│  原始文本                                           │
│  "人工智能改变世界"                                  │
│         │                                          │
│         ▼                                          │
│  Tokenization (分词)                               │
│  ["人工智能", "改变", "世界"]                        │
│         │                                          │
│         ▼                                          │
│  Embedding Model                                   │
│  (深度学习模型)                                      │
│         │                                          │
│         ▼                                          │
│  向量表示 (1536维)                                  │
│  [0.023, -0.015, 0.048, ..., 0.012]               │
│                                                    │
│  特性:                                              │
│  • 相似语义 → 相似向量                               │
│  • 捕获语义关系                                      │
│  • 支持数学运算                                      │
│                                                    │
└────────────────────────────────────────────────────┘
```

### 向量空间可视化

```
        维度2
         │
         │    • "机器学习"
         │  •    • "深度学习"
         │    "AI"
         │
         │
─────────┼─────────────────── 维度1
         │
         │         • "足球"
         │       • "篮球"
         │
         │
```

相似概念在向量空间中聚集在一起。

---

## OpenAI Embeddings API

### 基础使用

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    """获取文本的Embedding"""
    text = text.replace("\n", " ")  # 替换换行符

    response = client.embeddings.create(
        input=text,
        model=model
    )

    return response.data[0].embedding

# 使用示例
text = "人工智能正在改变世界"
embedding = get_embedding(text)

print(f"文本: {text}")
print(f"向量维度: {len(embedding)}")
print(f"前10个值: {embedding[:10]}")
```

### 模型对比

```python
class EmbeddingComparison:
    """Embedding模型对比"""

    MODELS = {
        "text-embedding-3-small": {
            "dimensions": 1536,
            "price_per_1m": 0.02,  # USD
            "performance": "高性价比"
        },
        "text-embedding-3-large": {
            "dimensions": 3072,
            "price_per_1m": 0.13,
            "performance": "最高精度"
        },
        "text-embedding-ada-002": {
            "dimensions": 1536,
            "price_per_1m": 0.10,
            "performance": "旧版模型"
        }
    }

    @staticmethod
    def compare_models(text):
        """对比不同模型"""
        client = OpenAI()

        results = {}

        for model, info in EmbeddingComparison.MODELS.items():
            print(f"\n测试模型: {model}")

            response = client.embeddings.create(
                input=text,
                model=model
            )

            embedding = response.data[0].embedding

            results[model] = {
                "dimensions": len(embedding),
                "sample_values": embedding[:5],
                "info": info
            }

        return results

# 使用示例
text = "深度学习是机器学习的一个分支"
results = EmbeddingComparison.compare_models(text)

for model, data in results.items():
    print(f"\n{model}:")
    print(f"  维度: {data['dimensions']}")
    print(f"  性能: {data['info']['performance']}")
    print(f"  价格: ${data['info']['price_per_1m']}/1M tokens")
```

### 批量处理

```python
def batch_embed(texts, batch_size=100, model="text-embedding-3-small"):
    """批量生成Embeddings"""
    client = OpenAI()
    embeddings = []

    # 分批处理
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # 清理文本
        batch = [text.replace("\n", " ") for text in batch]

        # 调用API
        response = client.embeddings.create(
            input=batch,
            model=model
        )

        # 提取embeddings
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)

        print(f"处理进度: {min(i + batch_size, len(texts))}/{len(texts)}")

    return embeddings

# 使用示例
texts = [
    "人工智能",
    "机器学习",
    "深度学习",
    "神经网络",
    "自然语言处理"
]

embeddings = batch_embed(texts)
print(f"\n生成了{len(embeddings)}个向量")
```

### 成本计算

```python
import tiktoken

class EmbeddingCostCalculator:
    """Embedding成本计算器"""

    # 价格 (USD per 1M tokens)
    PRICING = {
        "text-embedding-3-small": 0.02,
        "text-embedding-3-large": 0.13,
        "text-embedding-ada-002": 0.10
    }

    @staticmethod
    def count_tokens(texts):
        """计算token数量"""
        encoding = tiktoken.get_encoding("cl100k_base")

        total_tokens = 0
        for text in texts:
            tokens = encoding.encode(text)
            total_tokens += len(tokens)

        return total_tokens

    @staticmethod
    def calculate_cost(texts, model="text-embedding-3-small"):
        """计算成本"""
        tokens = EmbeddingCostCalculator.count_tokens(texts)
        price_per_token = EmbeddingCostCalculator.PRICING[model] / 1_000_000

        cost = tokens * price_per_token

        return {
            "texts_count": len(texts),
            "total_tokens": tokens,
            "avg_tokens": tokens / len(texts),
            "model": model,
            "cost_usd": cost,
            "cost_per_1000": (tokens / 1000) * (EmbeddingCostCalculator.PRICING[model] / 1000)
        }

# 使用示例
texts = ["这是一段测试文本" * 50 for _ in range(100)]

cost_info = EmbeddingCostCalculator.calculate_cost(
    texts,
    model="text-embedding-3-small"
)

print(f"文本数量: {cost_info['texts_count']}")
print(f"总Token数: {cost_info['total_tokens']}")
print(f"平均Token数: {cost_info['avg_tokens']:.1f}")
print(f"总成本: ${cost_info['cost_usd']:.4f}")
```

---

## Sentence Transformers

### 安装与配置

```bash
pip install sentence-transformers
```

### 基础使用

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 加载模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 生成Embedding
texts = [
    "人工智能正在改变世界",
    "机器学习是AI的核心技术",
    "今天天气很好"
]

embeddings = model.encode(texts)

print(f"生成了{len(embeddings)}个向量")
print(f"每个向量的维度: {embeddings[0].shape}")
```

### 推荐模型

```python
class SentenceTransformerModels:
    """Sentence-Transformers推荐模型"""

    MODELS = {
        # 中文模型
        "chinese": [
            {
                "name": "moka-ai/m3e-base",
                "dimensions": 768,
                "language": "中文",
                "use_case": "通用中文embedding"
            },
            {
                "name": "shibing624/text2vec-base-chinese",
                "dimensions": 768,
                "language": "中文",
                "use_case": "中文语义匹配"
            }
        ],

        # 英文模型
        "english": [
            {
                "name": "all-MiniLM-L6-v2",
                "dimensions": 384,
                "language": "英文",
                "use_case": "轻量级，速度快"
            },
            {
                "name": "all-mpnet-base-v2",
                "dimensions": 768,
                "language": "英文",
                "use_case": "高性能，精度高"
            }
        ],

        # 多语言模型
        "multilingual": [
            {
                "name": "paraphrase-multilingual-MiniLM-L12-v2",
                "dimensions": 384,
                "language": "多语言",
                "use_case": "支持50+语言"
            },
            {
                "name": "paraphrase-multilingual-mpnet-base-v2",
                "dimensions": 768,
                "language": "多语言",
                "use_case": "多语言高精度"
            }
        ],

        # 代码模型
        "code": [
            {
                "name": "microsoft/codebert-base",
                "dimensions": 768,
                "language": "代码",
                "use_case": "代码语义理解"
            }
        ]
    }

    @staticmethod
    def list_models(category=None):
        """列出推荐模型"""
        if category:
            return SentenceTransformerModels.MODELS.get(category, [])
        return SentenceTransformerModels.MODELS

# 使用示例
print("中文模型推荐:")
for model in SentenceTransformerModels.list_models("chinese"):
    print(f"  {model['name']}")
    print(f"  - 维度: {model['dimensions']}")
    print(f"  - 用途: {model['use_case']}\n")
```

### 自定义模型训练

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

class CustomEmbeddingTrainer:
    """自定义Embedding模型训练"""

    def __init__(self, base_model="paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(base_model)

    def prepare_training_data(self, data):
        """准备训练数据

        data格式: [
            {"text1": "句子1", "text2": "句子2", "label": 1.0},
            ...
        ]
        """
        examples = []

        for item in data:
            example = InputExample(
                texts=[item["text1"], item["text2"]],
                label=float(item["label"])
            )
            examples.append(example)

        return examples

    def train(self, train_data, epochs=1, batch_size=16):
        """训练模型"""
        # 准备数据
        examples = self.prepare_training_data(train_data)
        dataloader = DataLoader(examples, batch_size=batch_size, shuffle=True)

        # 定义损失函数
        loss = losses.CosineSimilarityLoss(self.model)

        # 训练
        self.model.fit(
            train_objectives=[(dataloader, loss)],
            epochs=epochs,
            warmup_steps=100
        )

    def save(self, path):
        """保存模型"""
        self.model.save(path)

# 使用示例
training_data = [
    {"text1": "人工智能", "text2": "机器学习", "label": 0.8},
    {"text1": "人工智能", "text2": "足球", "label": 0.1},
    {"text1": "深度学习", "text2": "神经网络", "label": 0.9},
]

trainer = CustomEmbeddingTrainer()
trainer.train(training_data, epochs=1)
trainer.save("./my_embedding_model")
```

---

## 向量相似度计算

### 余弦相似度

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """计算余弦相似度

    公式: cos(θ) = (A·B) / (||A|| * ||B||)
    范围: [-1, 1]，越接近1越相似
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    return dot_product / (norm1 * norm2)

# 使用示例
vec1 = np.array([1, 2, 3])
vec2 = np.array([2, 3, 4])
vec3 = np.array([-1, -2, -3])

sim_12 = cosine_similarity(vec1, vec2)
sim_13 = cosine_similarity(vec1, vec3)

print(f"vec1 vs vec2: {sim_12:.4f}")  # 相似
print(f"vec1 vs vec3: {sim_13:.4f}")  # 相反
```

### 欧几里得距离

```python
def euclidean_distance(vec1, vec2):
    """计算欧几里得距离

    公式: d = sqrt(Σ(xi - yi)²)
    范围: [0, ∞]，越小越相似
    """
    return np.linalg.norm(vec1 - vec2)

# 使用示例
vec1 = np.array([1, 2, 3])
vec2 = np.array([2, 3, 4])
vec3 = np.array([10, 20, 30])

dist_12 = euclidean_distance(vec1, vec2)
dist_13 = euclidean_distance(vec1, vec3)

print(f"vec1 vs vec2: {dist_12:.4f}")  # 相近
print(f"vec1 vs vec3: {dist_13:.4f}")  # 距离远
```

### 点积

```python
def dot_product(vec1, vec2):
    """计算点积

    公式: A·B = Σ(ai * bi)
    """
    return np.dot(vec1, vec2)

# 使用示例
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])

dp = dot_product(vec1, vec2)
print(f"点积: {dp}")  # 32
```

### 相似度计算工具类

```python
class SimilarityCalculator:
    """向量相似度计算工具"""

    @staticmethod
    def cosine(vec1, vec2):
        """余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    @staticmethod
    def euclidean(vec1, vec2):
        """欧几里得距离"""
        return np.linalg.norm(vec1 - vec2)

    @staticmethod
    def manhattan(vec1, vec2):
        """曼哈顿距离"""
        return np.sum(np.abs(vec1 - vec2))

    @staticmethod
    def dot_product(vec1, vec2):
        """点积"""
        return np.dot(vec1, vec2)

    @staticmethod
    def batch_cosine_similarity(query_vec, vectors):
        """批量计算余弦相似度"""
        # 归一化
        query_norm = query_vec / np.linalg.norm(query_vec)
        vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        # 批量点积
        similarities = np.dot(vectors_norm, query_norm)

        return similarities

    @staticmethod
    def find_most_similar(query_vec, vectors, top_k=5, metric="cosine"):
        """找到最相似的向量"""
        if metric == "cosine":
            similarities = SimilarityCalculator.batch_cosine_similarity(
                query_vec, vectors
            )
            # 余弦相似度越大越相似
            top_indices = np.argsort(similarities)[::-1][:top_k]
            top_scores = similarities[top_indices]

        elif metric == "euclidean":
            distances = np.array([
                np.linalg.norm(query_vec - vec) for vec in vectors
            ])
            # 欧氏距离越小越相似
            top_indices = np.argsort(distances)[:top_k]
            top_scores = distances[top_indices]

        else:
            raise ValueError(f"不支持的度量: {metric}")

        return top_indices, top_scores

# 使用示例
from openai import OpenAI

client = OpenAI()

# 生成测试向量
texts = [
    "人工智能正在改变世界",
    "机器学习是AI的重要分支",
    "深度学习使用神经网络",
    "今天天气很好",
    "我喜欢吃苹果"
]

embeddings = [
    client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding
    for text in texts
]

vectors = np.array(embeddings)

# 查询
query = "什么是人工智能？"
query_vec = np.array(
    client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
)

# 找到最相似的
indices, scores = SimilarityCalculator.find_most_similar(
    query_vec, vectors, top_k=3, metric="cosine"
)

print(f"查询: {query}\n")
print("最相似的文本:")
for i, (idx, score) in enumerate(zip(indices, scores), 1):
    print(f"{i}. {texts[idx]} (相似度: {score:.4f})")
```

---

## 语义搜索实现

### 简单语义搜索

```python
class SimpleSemanticSearch:
    """简单的语义搜索引擎"""

    def __init__(self, model="text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model
        self.documents = []
        self.embeddings = []

    def add_documents(self, documents):
        """添加文档"""
        self.documents.extend(documents)

        # 生成embeddings
        new_embeddings = []
        for doc in documents:
            embedding = self.client.embeddings.create(
                input=doc,
                model=self.model
            ).data[0].embedding

            new_embeddings.append(embedding)

        self.embeddings.extend(new_embeddings)

        print(f"已添加{len(documents)}个文档，总共{len(self.documents)}个")

    def search(self, query, top_k=5):
        """搜索"""
        if not self.documents:
            return []

        # 生成查询向量
        query_embedding = self.client.embeddings.create(
            input=query,
            model=self.model
        ).data[0].embedding

        query_vec = np.array(query_embedding)
        vectors = np.array(self.embeddings)

        # 计算相似度
        indices, scores = SimilarityCalculator.find_most_similar(
            query_vec, vectors, top_k=top_k
        )

        # 返回结果
        results = []
        for idx, score in zip(indices, scores):
            results.append({
                "document": self.documents[idx],
                "score": float(score)
            })

        return results

# 使用示例
search_engine = SimpleSemanticSearch()

# 添加文档
documents = [
    "Python是一种高级编程语言，广泛用于Web开发、数据分析和人工智能。",
    "JavaScript是Web开发的核心语言，运行在浏览器中。",
    "机器学习是人工智能的一个分支，让计算机从数据中学习。",
    "深度学习使用多层神经网络处理复杂模式。",
    "自然语言处理让计算机理解和生成人类语言。",
    "计算机视觉使机器能够理解图像和视频。",
    "数据库用于存储和管理大量结构化数据。",
    "云计算提供按需的计算资源和服务。"
]

search_engine.add_documents(documents)

# 搜索
results = search_engine.search("AI技术有哪些？", top_k=3)

print("\n搜索结果:")
for i, result in enumerate(results, 1):
    print(f"\n{i}. {result['document']}")
    print(f"   相似度: {result['score']:.4f}")
```

### 高级语义搜索

```python
from datetime import datetime
import json

class AdvancedSemanticSearch:
    """高级语义搜索引擎"""

    def __init__(self, model="text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model
        self.documents = []
        self.embeddings = []
        self.metadata = []

    def add_documents(self, documents, metadata=None):
        """添加文档及元数据

        documents: 文档列表
        metadata: 元数据列表 [{"title": "...", "category": "...", ...}, ...]
        """
        if metadata and len(documents) != len(metadata):
            raise ValueError("文档和元数据数量不匹配")

        # 生成embeddings
        for i, doc in enumerate(documents):
            embedding = self.client.embeddings.create(
                input=doc,
                model=self.model
            ).data[0].embedding

            self.documents.append(doc)
            self.embeddings.append(embedding)

            # 添加元数据
            doc_metadata = metadata[i] if metadata else {}
            doc_metadata["added_at"] = datetime.now().isoformat()
            doc_metadata["index"] = len(self.documents) - 1
            self.metadata.append(doc_metadata)

    def search(self, query, top_k=5, filter_fn=None, rerank=False):
        """搜索

        query: 查询文本
        top_k: 返回结果数量
        filter_fn: 过滤函数 lambda metadata: bool
        rerank: 是否重排序
        """
        if not self.documents:
            return []

        # 生成查询向量
        query_embedding = self.client.embeddings.create(
            input=query,
            model=self.model
        ).data[0].embedding

        query_vec = np.array(query_embedding)

        # 应用过滤
        if filter_fn:
            valid_indices = [
                i for i, meta in enumerate(self.metadata)
                if filter_fn(meta)
            ]
            vectors = np.array([self.embeddings[i] for i in valid_indices])
        else:
            valid_indices = list(range(len(self.documents)))
            vectors = np.array(self.embeddings)

        if len(vectors) == 0:
            return []

        # 计算相似度
        similarities = SimilarityCalculator.batch_cosine_similarity(
            query_vec, vectors
        )

        # 获取top_k结果
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # 获取2倍数量用于重排

        results = []
        for idx in top_indices[:top_k]:
            original_idx = valid_indices[idx]
            results.append({
                "document": self.documents[original_idx],
                "score": float(similarities[idx]),
                "metadata": self.metadata[original_idx]
            })

        # 重排序 (可以基于其他因素)
        if rerank:
            results = self._rerank(results, query)

        return results[:top_k]

    def _rerank(self, results, query):
        """重排序结果"""
        # 简单的重排策略: 结合相似度和文档长度
        for result in results:
            doc_length = len(result["document"])
            length_score = min(doc_length / 200, 1.0)  # 标准化长度分数
            result["final_score"] = result["score"] * 0.7 + length_score * 0.3

        return sorted(results, key=lambda x: x["final_score"], reverse=True)

    def save(self, filepath):
        """保存索引"""
        data = {
            "documents": self.documents,
            "embeddings": [emb for emb in self.embeddings],
            "metadata": self.metadata,
            "model": self.model
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath):
        """加载索引"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.documents = data["documents"]
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.model = data["model"]

# 使用示例
search = AdvancedSemanticSearch()

# 添加文档和元数据
documents = [
    "Python是一种高级编程语言",
    "机器学习是AI的重要分支",
    "深度学习使用神经网络"
]

metadata = [
    {"title": "Python简介", "category": "编程"},
    {"title": "机器学习", "category": "AI"},
    {"title": "深度学习", "category": "AI"}
]

search.add_documents(documents, metadata)

# 普通搜索
results = search.search("人工智能技术", top_k=2)
print("普通搜索:")
for r in results:
    print(f"  {r['metadata']['title']}: {r['document']}")

# 过滤搜索 (只搜索AI类别)
results = search.search(
    "学习技术",
    top_k=2,
    filter_fn=lambda meta: meta["category"] == "AI"
)
print("\n过滤搜索(只看AI类别):")
for r in results:
    print(f"  {r['metadata']['title']}: {r['document']}")

# 保存和加载
search.save("search_index.json")
```

---

## 完整代码示例

### 构建知识库问答系统

```python
from typing import List, Dict
import numpy as np
from openai import OpenAI

class KnowledgeBase:
    """知识库系统"""

    def __init__(self, model="text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model
        self.chunks = []
        self.embeddings = []
        self.sources = []

    def load_documents(self, documents: List[Dict[str, str]]):
        """加载文档

        documents: [
            {"content": "文档内容", "source": "来源", "metadata": {...}},
            ...
        ]
        """
        for doc in documents:
            # 分块
            chunks = self._chunk_text(doc["content"])

            # 为每个块生成embedding
            for chunk in chunks:
                embedding = self.client.embeddings.create(
                    input=chunk,
                    model=self.model
                ).data[0].embedding

                self.chunks.append(chunk)
                self.embeddings.append(embedding)
                self.sources.append({
                    "source": doc.get("source", "unknown"),
                    "metadata": doc.get("metadata", {})
                })

        print(f"已加载{len(documents)}个文档，生成{len(self.chunks)}个chunks")

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """文本分块"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索相关文档"""
        if not self.chunks:
            return []

        # 生成查询向量
        query_embedding = self.client.embeddings.create(
            input=query,
            model=self.model
        ).data[0].embedding

        query_vec = np.array(query_embedding)
        vectors = np.array(self.embeddings)

        # 计算相似度
        indices, scores = SimilarityCalculator.find_most_similar(
            query_vec, vectors, top_k=top_k
        )

        # 返回结果
        results = []
        for idx, score in zip(indices, scores):
            results.append({
                "content": self.chunks[idx],
                "score": float(score),
                "source": self.sources[idx]
            })

        return results

    def answer_question(self, question: str, top_k: int = 3) -> str:
        """回答问题"""
        # 检索相关文档
        relevant_docs = self.retrieve(question, top_k=top_k)

        if not relevant_docs:
            return "抱歉，我没有找到相关信息。"

        # 构建上下文
        context = "\n\n".join([
            f"[来源: {doc['source']['source']}]\n{doc['content']}"
            for doc in relevant_docs
        ])

        # 生成答案
        prompt = f"""
基于以下上下文回答问题。如果上下文中没有相关信息，请说"我不知道"。

上下文:
{context}

问题: {question}

答案:
"""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": [doc["source"] for doc in relevant_docs]
        }

# 使用示例
kb = KnowledgeBase()

# 加载文档
documents = [
    {
        "content": "Python是一种解释型、面向对象、动态数据类型的高级程序设计语言。Python由Guido van Rossum于1989年底发明。Python的设计哲学强调代码的可读性和简洁的语法。",
        "source": "Python文档",
        "metadata": {"category": "编程语言"}
    },
    {
        "content": "机器学习是人工智能的一个分支。它是一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、算法复杂度理论等多门学科。机器学习的核心是使用算法解析数据，从中学习，然后对真实世界中的事件做出决策和预测。",
        "source": "AI百科",
        "metadata": {"category": "人工智能"}
    },
    {
        "content": "深度学习是机器学习的一个子集，它使用多层神经网络来学习数据的表示。深度学习在图像识别、语音识别、自然语言处理等领域取得了突破性进展。",
        "source": "深度学习指南",
        "metadata": {"category": "人工智能"}
    }
]

kb.load_documents(documents)

# 问答
result = kb.answer_question("什么是Python？")
print(f"问题: 什么是Python？")
print(f"答案: {result['answer']}")
print(f"\n参考来源:")
for source in result['sources']:
    print(f"  - {source['source']}")
```

---

## 总结

本教程涵盖了Embedding和向量表示的核心知识:

1. **Embedding原理**: 文本到向量的转换
2. **OpenAI API**: 使用最先进的embedding模型
3. **Sentence Transformers**: 开源替代方案
4. **相似度计算**: 余弦、欧氏、点积等度量
5. **语义搜索**: 从简单到高级的实现
6. **实战应用**: 构建知识库问答系统

## 关键要点

- Embedding捕获文本的语义信息
- 相似文本的向量在空间中距离近
- 余弦相似度是最常用的度量
- 批量处理可以提高效率
- 元数据和过滤增强搜索能力

## 下一步

- 学习向量数据库 (Milvus, Pinecone)
- 探索RAG (检索增强生成)
- 优化分块策略
- 实现混合搜索 (关键词+语义)

## 参考资源

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Sentence Transformers](https://www.sbert.net/)
- [Hugging Face Models](https://huggingface.co/models)
