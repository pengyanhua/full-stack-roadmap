"""
============================================================
                RAG 系统完整实战示例
============================================================
实现一个完整的 RAG（检索增强生成）系统

包含：
1. 文档加载与分块
2. 向量嵌入
3. 向量存储
4. 检索与重排序
5. 生成回答
============================================================
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
import re
from abc import ABC, abstractmethod


# ============================================================
#                    一、数据结构
# ============================================================

@dataclass
class Document:
    """文档"""
    content: str
    metadata: Dict[str, Any] = None
    id: str = None

    def __post_init__(self):
        if self.id is None:
            self.id = hashlib.md5(self.content.encode()).hexdigest()[:16]
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Chunk:
    """文档块"""
    content: str
    metadata: Dict[str, Any]
    doc_id: str
    chunk_index: int
    embedding: List[float] = None

    @property
    def id(self) -> str:
        return f"{self.doc_id}_{self.chunk_index}"


@dataclass
class SearchResult:
    """搜索结果"""
    chunk: Chunk
    score: float
    rank: int = 0


# ============================================================
#                    二、文档处理
# ============================================================

class TextSplitter:
    """文本分块器"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        """
        Args:
            chunk_size: 块大小（字符数）
            chunk_overlap: 块重叠大小
            separators: 分隔符列表（按优先级）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", ".", " ", ""]

    def split(self, text: str) -> List[str]:
        """分割文本"""
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """递归分割"""
        final_chunks = []

        # 找到有效的分隔符
        separator = separators[-1]
        for sep in separators:
            if sep in text:
                separator = sep
                break

        # 分割
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # 合并小块
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = len(split)

            if current_length + split_length > self.chunk_size:
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    final_chunks.append(chunk_text)

                    # 保留重叠部分
                    overlap_text = chunk_text[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                    current_chunk = [overlap_text] if overlap_text else []
                    current_length = len(overlap_text)

            current_chunk.append(split)
            current_length += split_length + len(separator)

        # 处理剩余
        if current_chunk:
            final_chunks.append(separator.join(current_chunk))

        return final_chunks

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """分割文档列表"""
        chunks = []

        for doc in documents:
            text_chunks = self.split(doc.content)

            for i, text in enumerate(text_chunks):
                chunk = Chunk(
                    content=text,
                    metadata={**doc.metadata, "source": doc.id},
                    doc_id=doc.id,
                    chunk_index=i
                )
                chunks.append(chunk)

        return chunks


class DocumentLoader:
    """文档加载器"""

    @staticmethod
    def load_text(file_path: str) -> Document:
        """加载文本文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return Document(
            content=content,
            metadata={"source": file_path, "type": "text"}
        )

    @staticmethod
    def load_markdown(file_path: str) -> List[Document]:
        """加载 Markdown 文件，按标题分割"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 按一级标题分割
        sections = re.split(r'\n#\s+', content)
        documents = []

        for i, section in enumerate(sections):
            if not section.strip():
                continue

            lines = section.split('\n', 1)
            title = lines[0].strip()
            body = lines[1] if len(lines) > 1 else ""

            documents.append(Document(
                content=f"# {title}\n{body}" if i > 0 else section,
                metadata={
                    "source": file_path,
                    "type": "markdown",
                    "title": title if i > 0 else "Introduction"
                }
            ))

        return documents


# ============================================================
#                    三、嵌入模型
# ============================================================

class EmbeddingModel(ABC):
    """嵌入模型基类"""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """生成嵌入向量"""
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量维度"""
        pass


class MockEmbeddingModel(EmbeddingModel):
    """模拟嵌入模型（用于测试）"""

    def __init__(self, dim: int = 384):
        self._dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        import numpy as np
        np.random.seed(42)

        embeddings = []
        for text in texts:
            # 使用文本哈希生成伪随机向量
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            np.random.seed(seed)
            embedding = np.random.rand(self._dim).tolist()
            embeddings.append(embedding)

        return embeddings

    @property
    def dim(self) -> int:
        return self._dim


class SentenceTransformerEmbedding(EmbeddingModel):
    """
    SentenceTransformer 嵌入模型

    pip install sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._dim = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    @property
    def dim(self) -> int:
        return self._dim


class OpenAIEmbedding(EmbeddingModel):
    """
    OpenAI 嵌入模型

    pip install openai
    """

    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        try:
            import openai
            openai.api_key = api_key
            self.model = model
            self._dim = 1536 if "ada" in model else 3072
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

    def embed(self, texts: List[str]) -> List[List[float]]:
        import openai

        response = openai.embeddings.create(
            model=self.model,
            input=texts
        )

        return [item.embedding for item in response.data]

    @property
    def dim(self) -> int:
        return self._dim


# ============================================================
#                    四、向量存储
# ============================================================

class VectorStore(ABC):
    """向量存储基类"""

    @abstractmethod
    def add(self, chunks: List[Chunk]) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int) -> List[SearchResult]:
        pass


class InMemoryVectorStore(VectorStore):
    """内存向量存储（简单实现）"""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.embeddings: List[List[float]] = []

    def add(self, chunks: List[Chunk]) -> None:
        for chunk in chunks:
            self.chunks.append(chunk)
            self.embeddings.append(chunk.embedding)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_fn=None
    ) -> List[SearchResult]:
        """
        搜索

        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            filter_fn: 过滤函数
        """
        import numpy as np

        query = np.array(query_embedding)

        # 计算余弦相似度
        scores = []
        for i, (chunk, embedding) in enumerate(zip(self.chunks, self.embeddings)):
            if filter_fn and not filter_fn(chunk):
                continue

            emb = np.array(embedding)
            similarity = np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb))
            scores.append((chunk, similarity))

        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 返回 top_k
        results = []
        for rank, (chunk, score) in enumerate(scores[:top_k]):
            results.append(SearchResult(chunk=chunk, score=score, rank=rank))

        return results


# ============================================================
#                    五、检索器
# ============================================================

class Retriever:
    """检索器"""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        top_k: int = 5
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_fn=None
    ) -> List[SearchResult]:
        """检索相关文档"""
        top_k = top_k or self.top_k

        # 生成查询向量
        query_embedding = self.embedding_model.embed([query])[0]

        # 搜索
        results = self.vector_store.search(query_embedding, top_k)

        return results


class HybridRetriever:
    """混合检索器（向量 + 关键词）"""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        top_k: int = 5,
        alpha: float = 0.7
    ):
        """
        Args:
            alpha: 向量检索权重，1-alpha 为关键词检索权重
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.alpha = alpha

    def _keyword_search(
        self,
        query: str,
        chunks: List[Chunk],
        top_k: int
    ) -> List[Tuple[Chunk, float]]:
        """简单关键词搜索"""
        query_terms = set(query.lower().split())

        scores = []
        for chunk in chunks:
            content_terms = set(chunk.content.lower().split())
            # 计算 Jaccard 相似度
            intersection = len(query_terms & content_terms)
            union = len(query_terms | content_terms)
            score = intersection / union if union > 0 else 0
            scores.append((chunk, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def retrieve(self, query: str, top_k: int = None) -> List[SearchResult]:
        """混合检索"""
        top_k = top_k or self.top_k

        # 向量检索
        query_embedding = self.embedding_model.embed([query])[0]
        vector_results = self.vector_store.search(query_embedding, top_k * 2)

        # 关键词检索
        chunks = [r.chunk for r in vector_results]
        keyword_results = self._keyword_search(query, chunks, top_k * 2)

        # 合并分数（RRF 或加权平均）
        chunk_scores = {}

        for r in vector_results:
            chunk_scores[r.chunk.id] = self.alpha * r.score

        for chunk, score in keyword_results:
            if chunk.id in chunk_scores:
                chunk_scores[chunk.id] += (1 - self.alpha) * score
            else:
                chunk_scores[chunk.id] = (1 - self.alpha) * score

        # 排序
        chunk_map = {r.chunk.id: r.chunk for r in vector_results}
        sorted_results = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (chunk_id, score) in enumerate(sorted_results[:top_k]):
            if chunk_id in chunk_map:
                results.append(SearchResult(
                    chunk=chunk_map[chunk_id],
                    score=score,
                    rank=rank
                ))

        return results


# ============================================================
#                    六、RAG 系统
# ============================================================

class RAGSystem:
    """RAG 系统"""

    def __init__(
        self,
        embedding_model: EmbeddingModel = None,
        vector_store: VectorStore = None,
        llm=None,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.embedding_model = embedding_model or MockEmbeddingModel()
        self.vector_store = vector_store or InMemoryVectorStore()
        self.llm = llm
        self.splitter = TextSplitter(chunk_size, chunk_overlap)
        self.retriever = Retriever(
            self.vector_store,
            self.embedding_model
        )

    def add_documents(self, documents: List[Document]) -> int:
        """添加文档"""
        # 分块
        chunks = self.splitter.split_documents(documents)

        # 生成嵌入
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.embed(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        # 存储
        self.vector_store.add(chunks)

        print(f"已添加 {len(documents)} 个文档，共 {len(chunks)} 个块")
        return len(chunks)

    def add_texts(self, texts: List[str], metadatas: List[Dict] = None) -> int:
        """添加文本"""
        if metadatas is None:
            metadatas = [{}] * len(texts)

        documents = [
            Document(content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]

        return self.add_documents(documents)

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """检索"""
        return self.retriever.retrieve(query, top_k)

    def generate_prompt(
        self,
        query: str,
        context_results: List[SearchResult],
        system_prompt: str = None
    ) -> str:
        """生成提示词"""
        if system_prompt is None:
            system_prompt = """你是一个有帮助的助手。请根据给定的参考资料回答用户的问题。
如果资料中没有相关信息，请诚实地说明。
回答时请引用相关资料的来源。"""

        # 构建上下文
        context_parts = []
        for r in context_results:
            source = r.chunk.metadata.get("source", "未知来源")
            context_parts.append(f"[来源: {source}]\n{r.chunk.content}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""{system_prompt}

参考资料：
{context}

问题：{query}

回答："""

        return prompt

    def query(
        self,
        question: str,
        top_k: int = 5,
        return_context: bool = False
    ) -> Dict:
        """
        查询

        Args:
            question: 问题
            top_k: 检索数量
            return_context: 是否返回上下文

        Returns:
            包含答案和相关信息的字典
        """
        # 检索
        results = self.retrieve(question, top_k)

        # 生成提示词
        prompt = self.generate_prompt(question, results)

        # 生成回答
        if self.llm:
            answer = self.llm.generate(prompt)
        else:
            # 模拟回答
            answer = f"[模拟回答] 根据检索到的 {len(results)} 个相关文档片段，这是关于 '{question}' 的问题。"
            answer += f"\n\n最相关的内容来自: {results[0].chunk.metadata.get('source', '未知') if results else '无'}"

        response = {
            "answer": answer,
            "sources": [
                {
                    "content": r.chunk.content[:200] + "...",
                    "metadata": r.chunk.metadata,
                    "score": r.score
                }
                for r in results
            ]
        }

        if return_context:
            response["context"] = [r.chunk.content for r in results]
            response["prompt"] = prompt

        return response


# ============================================================
#                    七、评估
# ============================================================

class RAGEvaluator:
    """RAG 系统评估器"""

    @staticmethod
    def calculate_recall(
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """计算召回率"""
        if not relevant:
            return 0.0
        retrieved_set = set(retrieved)
        relevant_set = set(relevant)
        return len(retrieved_set & relevant_set) / len(relevant_set)

    @staticmethod
    def calculate_precision(
        retrieved: List[str],
        relevant: List[str]
    ) -> float:
        """计算精确率"""
        if not retrieved:
            return 0.0
        retrieved_set = set(retrieved)
        relevant_set = set(relevant)
        return len(retrieved_set & relevant_set) / len(retrieved_set)

    @staticmethod
    def calculate_mrr(
        retrieved_list: List[List[str]],
        relevant_list: List[List[str]]
    ) -> float:
        """计算 MRR（Mean Reciprocal Rank）"""
        mrr_sum = 0.0

        for retrieved, relevant in zip(retrieved_list, relevant_list):
            relevant_set = set(relevant)
            for rank, item in enumerate(retrieved, 1):
                if item in relevant_set:
                    mrr_sum += 1.0 / rank
                    break

        return mrr_sum / len(retrieved_list) if retrieved_list else 0.0


# ============================================================
#                    主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RAG 系统完整示例")
    print("=" * 60)

    # 创建 RAG 系统
    rag = RAGSystem(chunk_size=200, chunk_overlap=20)

    # 准备示例文档
    sample_documents = [
        Document(
            content="""
            向量数据库是一种专门用于存储和检索高维向量数据的数据库系统。
            它的主要用途是支持语义搜索和人工智能应用。
            与传统数据库不同，向量数据库使用相似度搜索而非精确匹配。
            常见的向量数据库包括 Milvus、Pinecone、Chroma 等。
            向量数据库的核心是向量索引，常用的索引算法有 HNSW、IVF、PQ 等。
            """,
            metadata={"source": "vector_db_intro.md", "topic": "vector_database"}
        ),
        Document(
            content="""
            RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。
            它首先从知识库中检索相关文档，然后将这些文档作为上下文提供给语言模型。
            RAG 可以有效减少语言模型的幻觉问题，提高回答的准确性。
            RAG 系统的核心组件包括：文档处理、向量嵌入、向量存储、检索器和生成器。
            """,
            metadata={"source": "rag_intro.md", "topic": "rag"}
        ),
        Document(
            content="""
            Embedding（嵌入）是将文本、图像等数据转换为数值向量的过程。
            嵌入模型通过深度学习训练，能够捕捉数据的语义信息。
            相似的内容会得到相近的向量表示，这是语义搜索的基础。
            常用的文本嵌入模型包括 BERT、Sentence-BERT、OpenAI 的 text-embedding 等。
            嵌入向量的维度通常在 384 到 1536 之间。
            """,
            metadata={"source": "embedding_intro.md", "topic": "embedding"}
        )
    ]

    # 添加文档
    rag.add_documents(sample_documents)

    # 测试查询
    test_questions = [
        "什么是向量数据库？",
        "RAG 技术有什么优势？",
        "Embedding 是什么？",
        "常见的向量索引算法有哪些？"
    ]

    print("\n--- 测试查询 ---")
    for question in test_questions:
        print(f"\n问题: {question}")

        result = rag.query(question, top_k=2, return_context=True)

        print(f"回答: {result['answer']}")
        print(f"来源:")
        for source in result['sources']:
            print(f"  - {source['metadata'].get('source', '未知')} (相似度: {source['score']:.4f})")

    # 评估示例
    print("\n--- 评估示例 ---")
    evaluator = RAGEvaluator()

    # 模拟评估数据
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = ["doc1", "doc3", "doc5"]

    recall = evaluator.calculate_recall(retrieved, relevant)
    precision = evaluator.calculate_precision(retrieved, relevant)

    print(f"召回率: {recall:.2%}")
    print(f"精确率: {precision:.2%}")

    print("\n示例完成！")
