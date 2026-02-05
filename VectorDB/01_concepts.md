# 向量数据库基础概念

## 一、什么是向量数据库

向量数据库是专门用于存储和检索高维向量数据的数据库系统，主要用于支持语义搜索和 AI 应用。

### 核心概念

```
传统数据库 vs 向量数据库

传统数据库（精确匹配）：
┌─────────────────────────────────┐
│  SELECT * FROM products         │
│  WHERE name = 'iPhone'          │
└─────────────────────────────────┘
结果：只匹配 "iPhone"，不匹配 "苹果手机"

向量数据库（语义相似）：
┌─────────────────────────────────┐
│  "iPhone" → [0.1, 0.8, 0.3...]  │  嵌入向量
│  "苹果手机" → [0.12, 0.79, 0.31...]
│  相似度: 0.98                    │  语义相似！
└─────────────────────────────────┘
结果：找到语义相似的结果
```

### 向量（Embedding）

```
文本 → 嵌入模型 → 向量

"今天天气真好"
      ↓
   Embedding Model (如 OpenAI text-embedding-ada-002)
      ↓
[0.023, -0.045, 0.078, 0.012, ..., 0.056]  (1536 维)

特点：
- 语义相近的文本，向量也相近
- 可以表示文本、图像、音频等任何数据
- 维度由模型决定（常见：384, 768, 1536）
```

### 相似度度量

```
1. 余弦相似度（Cosine Similarity）
   - 测量向量夹角
   - 范围 [-1, 1]，越大越相似
   - 适合：文本语义搜索

   cos(θ) = (A · B) / (||A|| × ||B||)

2. 欧氏距离（Euclidean Distance / L2）
   - 测量向量间直线距离
   - 范围 [0, ∞)，越小越相似
   - 适合：图像相似度

   d = √Σ(ai - bi)²

3. 内积（Inner Product / Dot Product）
   - 向量点积
   - 越大越相似
   - 适合：已归一化的向量

   A · B = Σ(ai × bi)
```

## 二、向量索引算法

### 为什么需要索引？

```
暴力搜索（Brute Force）：
- 对每个向量计算相似度
- 时间复杂度 O(n × d)
- 100万向量 × 1536维 ≈ 计算量巨大

索引加速：
- 预处理构建索引结构
- 查询时间大幅降低
- 牺牲少量精度换取速度
```

### 1. IVF（Inverted File Index）

```
核心思想：将向量空间划分为多个聚类

构建索引：
1. 使用 K-Means 将向量聚成 nlist 个簇
2. 每个向量分配到最近的簇

      ●  ●        ○  ○
    ●  ●[C1]●      ○[C2]○  ○
      ●  ●          ○  ○

查询：
1. 找到查询向量最近的 nprobe 个簇
2. 只在这些簇中搜索

参数：
- nlist: 聚类数量（通常 √n 到 4√n）
- nprobe: 查询时搜索的簇数（越大越精确，越慢）
```

### 2. HNSW（Hierarchical Navigable Small World）

```
核心思想：多层图结构，类似跳表

结构：
Layer 2:  [A] ─────────────────── [G]
           │                       │
Layer 1:  [A] ───── [C] ───── [E] ─ [G]
           │         │         │    │
Layer 0:  [A] ─ [B] ─ [C] ─ [D] ─ [E] ─ [F] ─ [G]

查询：
1. 从最高层开始
2. 贪心移动到离目标最近的节点
3. 下降到下一层继续

优点：
- 查询速度快 O(log n)
- 召回率高
- 支持动态插入

参数：
- M: 每层连接数
- efConstruction: 构建时的搜索范围
- efSearch: 查询时的搜索范围
```

### 3. PQ（Product Quantization）

```
核心思想：向量压缩，减少内存

过程：
原始向量 (128维)
[v1, v2, v3, v4, ..., v128]
    ↓ 分成 M 个子向量
[v1-v32] [v33-v64] [v65-v96] [v97-v128]
    ↓ 每个子向量量化为一个码字 ID
   [23]    [156]     [89]      [201]

存储：
- 原始：128 × 4 = 512 字节
- 量化后：4 × 1 = 4 字节

优点：大幅减少内存
缺点：有精度损失
```

### 4. 常见组合

```
IVF + PQ (IVF-PQ)
- 先用 IVF 缩小搜索范围
- 再用 PQ 压缩向量节省内存

IVF + HNSW
- 用 HNSW 加速 IVF 的聚类中心搜索

实际选择：
- 小数据量（<100万）：HNSW
- 中等数据量：IVF-HNSW
- 大数据量（>1亿）：IVF-PQ
- 内存受限：PQ 系列
```

## 三、主流向量数据库对比

| 数据库 | 类型 | 特点 | 适用场景 |
|--------|------|------|----------|
| **Milvus** | 开源 | 功能全面、分布式、性能强 | 大规模生产环境 |
| **Pinecone** | 云服务 | 全托管、易用、开箱即用 | 快速开发、无运维需求 |
| **Chroma** | 开源 | 轻量、Python 友好 | 原型开发、小项目 |
| **Weaviate** | 开源 | 内置向量化、GraphQL | 需要自动向量化 |
| **Qdrant** | 开源 | Rust 编写、高性能 | 性能敏感场景 |
| **Faiss** | 库 | Facebook 出品、纯索引 | 需要自行管理存储 |
| **pgvector** | 扩展 | PostgreSQL 扩展 | 已有 PG、数据量小 |

### 详细对比

```
Milvus
├── 优点
│   ├── 分布式架构，支持水平扩展
│   ├── 多种索引类型（IVF, HNSW, PQ...）
│   ├── 支持混合搜索（向量 + 标量过滤）
│   └── 活跃的社区和文档
└── 缺点
    ├── 部署相对复杂
    └── 资源消耗较大

Pinecone
├── 优点
│   ├── 全托管，零运维
│   ├── 简单的 API
│   ├── 自动扩缩容
│   └── 高可用保证
└── 缺点
    ├── 闭源，厂商锁定
    ├── 成本较高
    └── 自定义能力有限

Chroma
├── 优点
│   ├── 极简 API
│   ├── Python 原生
│   ├── 内嵌模式无需部署
│   └── 适合原型和学习
└── 缺点
    ├── 功能相对基础
    ├── 大规模性能一般
    └── 生态不如 Milvus

Faiss (Facebook AI Similarity Search)
├── 优点
│   ├── 极致性能
│   ├── 索引算法全面
│   ├── GPU 加速支持
│   └── 工业级验证
└── 缺点
    ├── 只是库，非数据库
    ├── 需要自行管理持久化
    └── 无元数据存储
```

## 四、核心功能

### 1. 向量插入

```python
# 伪代码
collection.insert([
    {
        "id": "doc1",
        "vector": [0.1, 0.2, 0.3, ...],
        "metadata": {"title": "文档1", "category": "tech"}
    },
    {
        "id": "doc2",
        "vector": [0.2, 0.3, 0.4, ...],
        "metadata": {"title": "文档2", "category": "news"}
    }
])
```

### 2. 相似度搜索

```python
# 基本搜索
results = collection.search(
    query_vector=[0.15, 0.25, 0.35, ...],
    top_k=10  # 返回最相似的 10 个
)

# 带过滤条件
results = collection.search(
    query_vector=[0.15, 0.25, 0.35, ...],
    top_k=10,
    filter={"category": "tech"}  # 只在 tech 类别中搜索
)
```

### 3. 混合搜索

```python
# 向量相似度 + 关键词匹配 + 属性过滤
results = collection.hybrid_search(
    query_vector=[0.15, 0.25, 0.35, ...],
    keywords="机器学习",
    filter={
        "category": {"$in": ["tech", "ai"]},
        "date": {"$gte": "2024-01-01"}
    },
    top_k=10
)
```

### 4. 元数据过滤

```python
# 支持的操作符（以 Milvus 为例）
filter_expr = """
    category == "tech"
    AND price > 100
    AND tags ARRAY_CONTAINS "AI"
    AND status IN ["active", "pending"]
"""
```

## 五、RAG 应用架构

### RAG（Retrieval-Augmented Generation）

```
用户问题: "什么是向量数据库？"
           │
           ↓
    ┌──────────────┐
    │   Embedding   │  将问题转为向量
    │    Model      │
    └──────────────┘
           │
           ↓ 问题向量
    ┌──────────────┐
    │   向量数据库   │  检索相关文档
    │   Vector DB   │
    └──────────────┘
           │
           ↓ Top K 相关文档
    ┌──────────────┐
    │   Prompt      │  构造提示词
    │  Construction │
    └──────────────┘
           │
           ↓
    ┌──────────────┐
    │     LLM      │  生成回答
    │   (GPT-4)    │
    └──────────────┘
           │
           ↓
    回答: "向量数据库是一种专门用于..."
```

### 典型流程

```python
# 1. 文档预处理
documents = load_documents("./docs/")
chunks = split_into_chunks(documents, chunk_size=500)

# 2. 生成嵌入向量
embeddings = embedding_model.encode(chunks)

# 3. 存入向量数据库
vector_db.insert(chunks, embeddings)

# 4. 查询
def query(question: str):
    # 4.1 问题向量化
    query_embedding = embedding_model.encode(question)

    # 4.2 检索相关文档
    results = vector_db.search(query_embedding, top_k=5)

    # 4.3 构造提示词
    context = "\n".join([r.text for r in results])
    prompt = f"""根据以下资料回答问题：

资料：
{context}

问题：{question}

回答："""

    # 4.4 调用 LLM
    answer = llm.generate(prompt)
    return answer
```

## 六、最佳实践

### 1. 向量维度选择

```
维度越高：
+ 表达能力越强
- 存储和计算成本越高

推荐：
- 小模型：384 维（all-MiniLM-L6）
- 中等模型：768 维（BERT base）
- 大模型：1536 维（OpenAI ada-002）
- 多模态：512-1024 维
```

### 2. 分块策略

```
文档分块考虑因素：
- 块大小：200-1000 tokens（常用 500）
- 重叠：10-20%（保持上下文连贯）
- 分割方式：段落、句子、语义

示例：
┌────────────────────────────────┐
│ Chunk 1: This is the first... │
│ ~~~~~~~~~~~overlap~~~~~~~~~~~~ │
│ Chunk 2: ...first part and... │
│ ~~~~~~~~~~~overlap~~~~~~~~~~~~ │
│ Chunk 3: ...and this is the.. │
└────────────────────────────────┘
```

### 3. 索引参数调优

```python
# HNSW 参数
{
    "M": 16,              # 连接数，越大精度越高，内存越大
    "efConstruction": 200,  # 构建时搜索范围
    "efSearch": 100        # 查询时搜索范围
}

# IVF 参数
{
    "nlist": 1024,         # 聚类数
    "nprobe": 16           # 查询时搜索的聚类数
}

调优建议：
- 先用小参数快速验证
- 逐步增大直到满足精度要求
- 监控查询延迟和内存使用
```

### 4. 批量操作

```python
# 不推荐：逐条插入
for doc in documents:
    collection.insert(doc)

# 推荐：批量插入
batch_size = 1000
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    collection.insert(batch)
```

### 5. 性能优化

```
存储优化：
- 使用 PQ 压缩大规模数据
- 只存储必要的元数据
- 定期清理过期数据

查询优化：
- 合理设置 top_k
- 使用过滤条件减少搜索范围
- 预热常用查询

架构优化：
- 分片提高并发
- 副本提高可用性
- 读写分离
```

## 七、常见应用场景

| 场景 | 描述 | 技术要点 |
|------|------|----------|
| 语义搜索 | 理解用户意图的搜索 | 文本嵌入 + 余弦相似度 |
| 问答系统 | RAG 增强的 QA | 检索 + LLM 生成 |
| 推荐系统 | 相似内容/用户推荐 | 协同过滤 + 向量检索 |
| 图像搜索 | 以图搜图 | 图像嵌入（CLIP） |
| 去重/聚类 | 检测重复内容 | 相似度阈值判断 |
| 异常检测 | 发现异常模式 | 距离最远的点 |
