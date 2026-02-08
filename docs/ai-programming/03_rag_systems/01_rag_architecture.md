# RAG 完整架构设计

## 目录
1. [RAG 简介与核心概念](#1-rag-简介与核心概念)
2. [三代 RAG 架构演进](#2-三代-rag-架构演进)
3. [RAG 完整架构图](#3-rag-完整架构图)
4. [文档加载器](#4-文档加载器)
5. [文档处理 Pipeline](#5-文档处理-pipeline)
6. [检索策略详解](#6-检索策略详解)
7. [生成优化](#7-生成优化)
8. [端到端 RAG 系统实现](#8-端到端-rag-系统实现)
9. [RAG 评估体系](#9-rag-评估体系)
10. [最佳实践与常见问题](#10-最佳实践与常见问题)

---

## 1. RAG 简介与核心概念

### 1.1 什么是 RAG

RAG（Retrieval-Augmented Generation，检索增强生成）是一种将**外部知识检索**与**大语言模型生成**相结合的技术架构。它解决了 LLM 的三大核心痛点：

- **知识截止**：LLM 训练数据有时间限制，无法获取最新信息
- **幻觉问题**：LLM 可能生成看似合理但事实错误的内容
- **领域知识缺乏**：通用 LLM 缺少特定企业或行业的专有知识

### 1.2 RAG 核心工作原理

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG 核心工作流程                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐             │
│   │  用户提问  │───>│  检索相关文档  │───>│  构建增强提示  │            │
│   └──────────┘    └──────────────┘    └──────┬───────┘             │
│                          │                    │                      │
│                          v                    v                      │
│                   ┌──────────────┐    ┌──────────────┐             │
│                   │  向量数据库   │    │   LLM 生成    │             │
│                   │  知识库检索   │    │   最终回答    │             │
│                   └──────────────┘    └──────────────┘             │
│                                                                     │
│   公式: Answer = LLM(Query + Retrieved_Context)                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 RAG vs Fine-tuning vs Prompt Engineering

| 特性 | RAG | Fine-tuning | Prompt Engineering |
|------|-----|-------------|-------------------|
| 知识更新 | 实时（更新知识库即可） | 需要重新训练 | 受限于上下文窗口 |
| 成本 | 中等（向量DB + 检索） | 高（GPU训练） | 低 |
| 幻觉控制 | 好（有引用来源） | 中等 | 差 |
| 领域适应 | 好 | 最好 | 有限 |
| 实现复杂度 | 中等 | 高 | 低 |
| 数据隐私 | 好（数据不离开本地） | 需要上传训练数据 | 需要在提示中包含 |

---

## 2. 三代 RAG 架构演进

### 2.1 Naive RAG（第一代）

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Naive RAG 架构                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   索引阶段:                                                         │
│   ┌────────┐    ┌────────┐    ┌────────┐    ┌────────────┐        │
│   │ 文档加载 │───>│ 文本分块 │───>│ 向量化  │───>│ 存入向量DB  │       │
│   └────────┘    └────────┘    └────────┘    └────────────┘        │
│                                                                     │
│   查询阶段:                                                         │
│   ┌────────┐    ┌────────┐    ┌──────────┐   ┌──────────┐        │
│   │ 用户查询 │───>│ 查询向量化│───>│ 相似度检索 │──>│ LLM 生成  │       │
│   └────────┘    └────────┘    └──────────┘   └──────────┘        │
│                                                                     │
│   问题:                                                             │
│   - 分块质量差导致检索不准                                            │
│   - 检索结果与查询不匹配                                              │
│   - 生成时无法有效利用上下文                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Naive RAG 的典型问题**：
- **检索精度低**：简单的向量相似度匹配容易返回不相关文档
- **冗余信息**：检索到的多个片段可能包含重复内容
- **上下文丢失**：分块导致语义断裂，关键信息可能被切分
- **生成质量差**：模型可能忽略检索内容，继续产生幻觉

### 2.2 Advanced RAG（第二代）

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Advanced RAG 架构                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   预检索优化:                                                        │
│   ┌────────────┐   ┌────────────┐   ┌──────────────┐              │
│   │ 查询重写     │──>│ 查询扩展    │──>│ HyDE假设文档  │             │
│   └────────────┘   └────────────┘   └──────────────┘              │
│         │                                    │                      │
│         v                                    v                      │
│   ┌────────────────────────────────────────────────┐               │
│   │              混合检索 (Dense + Sparse)           │               │
│   └────────────────────────────────────────────────┘               │
│         │                                                           │
│         v                                                           │
│   后检索优化:                                                        │
│   ┌────────────┐   ┌────────────┐   ┌──────────────┐              │
│   │ Reranking   │──>│ 上下文压缩   │──>│ 多样性筛选    │             │
│   └────────────┘   └────────────┘   └──────────────┘              │
│         │                                                           │
│         v                                                           │
│   ┌────────────────────────────────────────────────┐               │
│   │           优化的 Prompt + LLM 生成              │               │
│   └────────────────────────────────────────────────┘               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Advanced RAG 的核心改进**：

1. **预检索优化（Pre-retrieval）**
   - 查询重写：将用户的自然语言查询转换为更适合检索的形式
   - 查询扩展：生成多个相关查询，提高检索覆盖率
   - HyDE：先让 LLM 生成假设性文档，再用该文档做检索

2. **检索优化（Retrieval）**
   - 混合检索：结合稠密向量检索和稀疏词频检索
   - 多路召回：从多个索引同时检索

3. **后检索优化（Post-retrieval）**
   - Reranking：使用交叉编码器对检索结果重排序
   - 上下文压缩：去除无关信息，保留核心内容
   - 去重与多样性：确保检索结果的多样性

### 2.3 Modular RAG（第三代）

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Modular RAG 架构                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│   │ 路由模块  │  │ 检索模块  │  │ 重排模块  │  │ 生成模块  │           │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│        │            │            │            │                     │
│   ┌────v────────────v────────────v────────────v────┐              │
│   │              编排引擎 (Orchestrator)              │              │
│   │                                                  │              │
│   │   支持的编排模式:                                  │              │
│   │   - 线性链式 (Sequential)                         │              │
│   │   - 条件分支 (Conditional)                        │              │
│   │   - 循环迭代 (Iterative)                          │              │
│   │   - 自适应 (Adaptive)                             │              │
│   └──────────────────────────────────────────────────┘              │
│        │            │            │            │                     │
│   ┌────v────┐  ┌────v────┐  ┌────v────┐  ┌────v────┐            │
│   │ 缓存模块  │  │ 评估模块  │  │ 反馈模块  │  │ 记忆模块  │           │
│   └─────────┘  └─────────┘  └─────────┘  └─────────┘            │
│                                                                     │
│   新增能力:                                                         │
│   - 自适应检索: 判断是否需要检索                                      │
│   - 迭代检索: 多轮检索逐步深入                                        │
│   - 检索+推理交替: 类似 ReAct 模式                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.4 三代架构对比

| 特性 | Naive RAG | Advanced RAG | Modular RAG |
|------|-----------|-------------|-------------|
| 检索方式 | 单一向量检索 | 混合检索 + Rerank | 自适应多策略检索 |
| 查询处理 | 直接使用原始查询 | 查询重写/扩展 | 动态路由 + 查询变换 |
| 分块策略 | 固定大小分块 | 语义分块 | 多粒度分块 |
| 生成策略 | 简单拼接上下文 | 上下文压缩 + 优化提示 | 迭代生成 + 自我反思 |
| 架构灵活性 | 固定流程 | 部分可配置 | 完全模块化 |
| 适用场景 | 简单问答 | 企业级应用 | 复杂推理任务 |
| 实现复杂度 | 低 | 中 | 高 |

---

## 3. RAG 完整架构图

### 3.1 生产级 RAG 系统架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        生产级 RAG 系统完整架构                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────── 数据摄入层 ────────────────────────────┐   │
│  │                                                                    │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐        │   │
│  │  │ PDF  │ │ Word │ │  MD  │ │ HTML │ │  DB  │ │ API  │        │   │
│  │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘        │   │
│  │     └────────┴────────┴────────┴────────┴────────┘              │   │
│  │                        │                                         │   │
│  │                   ┌────v────┐                                    │   │
│  │                   │ 文档解析 │                                    │   │
│  │                   └────┬────┘                                    │   │
│  │                        │                                         │   │
│  │  ┌─────────┐   ┌──────v──────┐   ┌───────────┐                │   │
│  │  │ 元数据提取│<──│  智能分块     │──>│  质量过滤   │               │   │
│  │  └─────────┘   └──────┬──────┘   └───────────┘                │   │
│  │                        │                                         │   │
│  └────────────────────────┼─────────────────────────────────────────┘   │
│                           │                                              │
│  ┌────────────────────────v─────────────────────────────────────────┐   │
│  │                       向量化与存储层                                │   │
│  │                                                                    │   │
│  │  ┌──────────────┐        ┌──────────────────────┐                │   │
│  │  │ Embedding模型 │        │    向量数据库          │                │   │
│  │  │ (OpenAI /     │───────>│  (Chroma / Milvus /  │                │   │
│  │  │  BGE / etc.)  │        │   Qdrant / Pinecone) │                │   │
│  │  └──────────────┘        └──────────────────────┘                │   │
│  │                                                                    │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌───────────────────── 查询与检索层 ──────────────────────────────┐   │
│  │                                                                    │   │
│  │  ┌────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐      │   │
│  │  │用户查询 │──>│ 查询理解   │──>│ 查询重写   │──>│ 多路检索   │     │   │
│  │  └────────┘   └──────────┘   └──────────┘   └────┬─────┘      │   │
│  │                                                    │             │   │
│  │       ┌──────────┐    ┌──────────┐    ┌──────────v─┐           │   │
│  │       │ BM25稀疏  │    │Dense稠密  │    │  结果融合    │          │   │
│  │       │  检索     │───>│  检索     │───>│  + Rerank   │          │   │
│  │       └──────────┘    └──────────┘    └────────────┘           │   │
│  │                                                                    │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌───────────────────── 生成与输出层 ──────────────────────────────┐   │
│  │                                                                    │   │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │   │
│  │  │上下文压缩 │──>│Prompt构建 │──>│ LLM生成   │──>│ 引用标注  │   │   │
│  │  └──────────┘   └──────────┘   └──────────┘   └──────────┘   │   │
│  │                                                                    │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌───────────────────── 评估与监控层 ──────────────────────────────┐   │
│  │                                                                    │   │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │   │
│  │  │检索质量   │   │生成质量   │   │端到端评估  │   │ 用户反馈  │   │   │
│  │  │ 监控     │   │ 监控     │   │(RAGAS)   │   │ 收集     │   │   │
│  │  └──────────┘   └──────────┘   └──────────┘   └──────────┘   │   │
│  │                                                                    │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 文档加载器

### 4.1 多格式文档加载

RAG 系统需要处理企业中常见的多种文档格式。LangChain 提供了丰富的文档加载器。

#### 安装依赖

```bash
pip install langchain langchain-community langchain-openai
pip install pypdf python-docx unstructured markdown beautifulsoup4
pip install chromadb tiktoken
```

#### PDF 文档加载

```python
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.schema import Document
from typing import List

def load_pdf(file_path: str) -> List[Document]:
    """
    加载单个PDF文件，按页拆分

    Args:
        file_path: PDF文件路径
    Returns:
        Document列表，每页一个Document
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 添加自定义元数据
    for i, doc in enumerate(documents):
        doc.metadata.update({
            "source_type": "pdf",
            "file_name": file_path.split("/")[-1],
            "page_number": i + 1,
            "total_pages": len(documents)
        })

    print(f"从 {file_path} 加载了 {len(documents)} 页")
    return documents


def load_pdf_directory(dir_path: str) -> List[Document]:
    """
    批量加载目录下所有PDF文件

    Args:
        dir_path: 包含PDF文件的目录路径
    Returns:
        所有PDF页面的Document列表
    """
    loader = PyPDFDirectoryLoader(dir_path)
    documents = loader.load()
    print(f"从 {dir_path} 共加载了 {len(documents)} 个文档页面")
    return documents


# 使用示例
if __name__ == "__main__":
    # 加载单个PDF
    docs = load_pdf("./data/technical_report.pdf")
    print(f"第1页内容预览: {docs[0].page_content[:200]}")
    print(f"元数据: {docs[0].metadata}")
```

#### Word 文档加载

```python
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader

def load_word(file_path: str, method: str = "docx2txt") -> List[Document]:
    """
    加载Word文档

    Args:
        file_path: Word文件路径
        method: 解析方法 - "docx2txt" 或 "unstructured"
    Returns:
        Document列表
    """
    if method == "docx2txt":
        loader = Docx2txtLoader(file_path)
    elif method == "unstructured":
        loader = UnstructuredWordDocumentLoader(
            file_path,
            mode="elements"  # 按元素拆分（段落、表格等）
        )
    else:
        raise ValueError(f"不支持的方法: {method}")

    documents = loader.load()

    for doc in documents:
        doc.metadata.update({
            "source_type": "word",
            "file_name": file_path.split("/")[-1]
        })

    return documents
```

#### Markdown 文档加载

```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

def load_markdown(file_path: str) -> List[Document]:
    """
    加载Markdown文件，按标题层级拆分

    Args:
        file_path: Markdown文件路径
    Returns:
        按标题层级拆分的Document列表
    """
    # 方法1: 使用UnstructuredMarkdownLoader
    loader = UnstructuredMarkdownLoader(file_path, mode="elements")
    documents = loader.load()

    return documents


def load_markdown_by_headers(file_path: str) -> List[Document]:
    """
    按Markdown标题层级拆分文档
    """
    with open(file_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # 定义标题层级
    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    documents = splitter.split_text(markdown_text)

    for doc in documents:
        doc.metadata["source_type"] = "markdown"
        doc.metadata["file_name"] = file_path.split("/")[-1]

    return documents
```

#### HTML 文档加载

```python
from langchain_community.document_loaders import BSHTMLLoader, WebBaseLoader

def load_html_file(file_path: str) -> List[Document]:
    """加载本地HTML文件"""
    loader = BSHTMLLoader(file_path, open_encoding="utf-8")
    documents = loader.load()

    for doc in documents:
        doc.metadata["source_type"] = "html"

    return documents


def load_web_page(url: str) -> List[Document]:
    """加载网页内容"""
    loader = WebBaseLoader(
        web_paths=[url],
        bs_kwargs={
            "parse_only": None  # 可以指定BeautifulSoup解析参数
        }
    )
    documents = loader.load()

    for doc in documents:
        doc.metadata["source_type"] = "web"
        doc.metadata["url"] = url

    return documents
```

### 4.2 统一文档加载管理器

```python
import os
from pathlib import Path
from typing import List, Optional
from langchain.schema import Document


class UnifiedDocumentLoader:
    """
    统一文档加载管理器
    支持 PDF、Word、Markdown、HTML、TXT 等格式
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf": "pdf",
        ".docx": "word",
        ".doc": "word",
        ".md": "markdown",
        ".html": "html",
        ".htm": "html",
        ".txt": "text",
    }

    def __init__(self):
        self.loaded_documents: List[Document] = []
        self.load_stats = {"success": 0, "failed": 0, "skipped": 0}

    def load_file(self, file_path: str) -> List[Document]:
        """加载单个文件"""
        ext = Path(file_path).suffix.lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            print(f"跳过不支持的格式: {ext} ({file_path})")
            self.load_stats["skipped"] += 1
            return []

        file_type = self.SUPPORTED_EXTENSIONS[ext]

        try:
            if file_type == "pdf":
                docs = load_pdf(file_path)
            elif file_type == "word":
                docs = load_word(file_path)
            elif file_type == "markdown":
                docs = load_markdown_by_headers(file_path)
            elif file_type == "html":
                docs = load_html_file(file_path)
            elif file_type == "text":
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
            else:
                docs = []

            self.loaded_documents.extend(docs)
            self.load_stats["success"] += 1
            print(f"成功加载: {file_path} ({len(docs)} 个片段)")
            return docs

        except Exception as e:
            print(f"加载失败: {file_path} - {str(e)}")
            self.load_stats["failed"] += 1
            return []

    def load_directory(self, dir_path: str, recursive: bool = True) -> List[Document]:
        """
        批量加载目录下所有支持的文档

        Args:
            dir_path: 目录路径
            recursive: 是否递归加载子目录
        """
        all_docs = []

        if recursive:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    docs = self.load_file(file_path)
                    all_docs.extend(docs)
        else:
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if os.path.isfile(file_path):
                    docs = self.load_file(file_path)
                    all_docs.extend(docs)

        print(f"\n加载统计: 成功 {self.load_stats['success']}, "
              f"失败 {self.load_stats['failed']}, "
              f"跳过 {self.load_stats['skipped']}")
        print(f"总文档片段数: {len(all_docs)}")

        return all_docs

    def get_stats(self) -> dict:
        """返回加载统计信息"""
        return {
            **self.load_stats,
            "total_documents": len(self.loaded_documents),
            "total_chars": sum(
                len(doc.page_content) for doc in self.loaded_documents
            )
        }


# 使用示例
if __name__ == "__main__":
    loader = UnifiedDocumentLoader()

    # 批量加载目录
    documents = loader.load_directory("./knowledge_base/")

    # 查看统计
    stats = loader.get_stats()
    print(f"加载统计: {stats}")
```

---

## 5. 文档处理 Pipeline

### 5.1 完整处理流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    文档处理 Pipeline 完整流程                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐ │
│  │ 1.加载  │──>│ 2.清洗  │──>│ 3.分块  │──>│ 4.向量化│──>│ 5.存储  │ │
│  └────────┘   └────────┘   └────────┘   └────────┘   └────────┘ │
│                                                                     │
│  加载: 多格式文档 ──> Document对象                                    │
│  清洗: 去除噪声、格式化、去重                                         │
│  分块: 按语义切分为合适大小的片段                                      │
│  向量化: 使用Embedding模型转换为向量                                   │
│  存储: 写入向量数据库（带元数据）                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 文本清洗

```python
import re
from typing import List
from langchain.schema import Document


class TextCleaner:
    """文本清洗器"""

    @staticmethod
    def clean_text(text: str) -> str:
        """基础文本清洗"""
        # 去除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 去除特殊控制字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # 标准化引号
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        # 去除页眉页脚常见模式
        text = re.sub(r'第\s*\d+\s*页.*?共\s*\d+\s*页', '', text)
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)

        return text.strip()

    @staticmethod
    def remove_duplicates(documents: List[Document],
                          similarity_threshold: float = 0.95) -> List[Document]:
        """
        基于内容相似度去重
        使用简单的Jaccard相似度
        """
        unique_docs = []
        seen_contents = []

        for doc in documents:
            content = doc.page_content.strip()

            if not content or len(content) < 10:
                continue

            is_duplicate = False
            content_words = set(content.split())

            for seen in seen_contents:
                seen_words = set(seen.split())
                # Jaccard相似度
                intersection = len(content_words & seen_words)
                union = len(content_words | seen_words)

                if union > 0 and intersection / union > similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_docs.append(doc)
                seen_contents.append(content)

        removed = len(documents) - len(unique_docs)
        print(f"去重: 原始 {len(documents)} 篇, 去除 {removed} 篇, 保留 {len(unique_docs)} 篇")
        return unique_docs

    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """清洗Document列表"""
        cleaned = []
        for doc in documents:
            cleaned_content = self.clean_text(doc.page_content)
            if len(cleaned_content) > 10:  # 过滤过短内容
                doc.page_content = cleaned_content
                cleaned.append(doc)

        return self.remove_duplicates(cleaned)
```

### 5.3 文本分块

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List


def create_chunks(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: List[str] = None
) -> List[Document]:
    """
    使用RecursiveCharacterTextSplitter进行文本分块

    Args:
        documents: 输入文档列表
        chunk_size: 每个分块的最大字符数
        chunk_overlap: 相邻分块的重叠字符数
        separators: 分隔符优先级列表
    Returns:
        分块后的Document列表
    """
    if separators is None:
        # 中文优化的分隔符
        separators = [
            "\n\n",     # 段落分隔
            "\n",       # 换行分隔
            "。",       # 中文句号
            "！",       # 中文感叹号
            "？",       # 中文问号
            "；",       # 中文分号
            "，",       # 中文逗号
            ". ",       # 英文句号
            "! ",       # 英文感叹号
            "? ",       # 英文问号
            " ",        # 空格
            ""          # 字符级
        ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)

    # 为每个分块添加序号元数据
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    print(f"分块完成: {len(documents)} 个文档 -> {len(chunks)} 个分块")
    print(f"平均分块大小: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} 字符")

    return chunks
```

### 5.4 向量化与存储

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List
import os


def create_vectorstore(
    documents: List[Document],
    collection_name: str = "knowledge_base",
    persist_directory: str = "./chroma_db",
    embedding_model: str = "text-embedding-3-small"
) -> Chroma:
    """
    将文档向量化并存入ChromaDB

    Args:
        documents: 分块后的文档列表
        collection_name: 集合名称
        persist_directory: 持久化存储目录
        embedding_model: OpenAI Embedding模型名称
    Returns:
        ChromaDB向量存储实例
    """
    # 初始化Embedding模型
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # 创建向量存储
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    print(f"向量存储创建完成:")
    print(f"  - 集合名称: {collection_name}")
    print(f"  - 文档数量: {len(documents)}")
    print(f"  - 存储路径: {persist_directory}")

    return vectorstore


def load_existing_vectorstore(
    persist_directory: str = "./chroma_db",
    collection_name: str = "knowledge_base",
    embedding_model: str = "text-embedding-3-small"
) -> Chroma:
    """加载已有的向量存储"""
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    vectorstore = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    count = vectorstore._collection.count()
    print(f"加载向量存储: {collection_name}, 包含 {count} 个向量")

    return vectorstore
```

### 5.5 完整 Pipeline 组装

```python
from typing import List, Optional
from langchain.schema import Document


class RAGPipeline:
    """
    完整的RAG文档处理Pipeline
    加载 -> 清洗 -> 分块 -> 向量化 -> 存储
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-3-small",
        persist_directory: str = "./chroma_db",
        collection_name: str = "knowledge_base"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.loader = UnifiedDocumentLoader()
        self.cleaner = TextCleaner()
        self.vectorstore = None

    def process_directory(self, dir_path: str) -> None:
        """处理整个目录的文档"""
        print("=" * 60)
        print("RAG Pipeline 开始处理")
        print("=" * 60)

        # Step 1: 加载文档
        print("\n[1/4] 加载文档...")
        raw_documents = self.loader.load_directory(dir_path)

        # Step 2: 清洗文档
        print("\n[2/4] 清洗文档...")
        cleaned_documents = self.cleaner.clean_documents(raw_documents)

        # Step 3: 文本分块
        print("\n[3/4] 文本分块...")
        chunks = create_chunks(
            cleaned_documents,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        # Step 4: 向量化与存储
        print("\n[4/4] 向量化与存储...")
        self.vectorstore = create_vectorstore(
            documents=chunks,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_model=self.embedding_model
        )

        print("\n" + "=" * 60)
        print("Pipeline 处理完成!")
        print(f"总文档数: {len(raw_documents)}")
        print(f"清洗后: {len(cleaned_documents)}")
        print(f"分块数: {len(chunks)}")
        print("=" * 60)

    def query(self, question: str, k: int = 4) -> List[Document]:
        """检索相关文档"""
        if self.vectorstore is None:
            self.vectorstore = load_existing_vectorstore(
                self.persist_directory,
                self.collection_name,
                self.embedding_model
            )

        results = self.vectorstore.similarity_search_with_score(question, k=k)

        print(f"\n查询: {question}")
        print(f"检索到 {len(results)} 个相关片段:")
        for i, (doc, score) in enumerate(results):
            print(f"  [{i+1}] 相似度: {score:.4f} | "
                  f"来源: {doc.metadata.get('file_name', 'unknown')}")

        return [doc for doc, _ in results]


# 使用示例
if __name__ == "__main__":
    pipeline = RAGPipeline(
        chunk_size=800,
        chunk_overlap=150,
        embedding_model="text-embedding-3-small",
        persist_directory="./my_knowledge_db",
        collection_name="company_docs"
    )

    # 处理文档目录
    pipeline.process_directory("./documents/")

    # 查询测试
    results = pipeline.query("公司的年假政策是什么？")
```

---

## 6. 检索策略详解

### 6.1 检索策略对比

```
┌──────────────────────────────────────────────────────────────────┐
│                      检索策略对比                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Dense Retrieval (稠密检索)                                      │
│  ┌──────────────────────────────────────────┐                   │
│  │  Query ──> [Embedding] ──> [0.1, 0.3, ..]│                   │
│  │  Doc   ──> [Embedding] ──> [0.2, 0.4, ..]│                   │
│  │  Score = cosine_similarity(q_vec, d_vec)  │                   │
│  │  优点: 语义理解强  缺点: 精确匹配弱        │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
│  Sparse Retrieval (稀疏检索)                                     │
│  ┌──────────────────────────────────────────┐                   │
│  │  Query ──> [BM25/TF-IDF] ──> 词频向量    │                   │
│  │  Score = BM25(query_terms, doc_terms)     │                   │
│  │  优点: 精确关键词匹配  缺点: 语义理解弱    │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
│  Hybrid Search (混合检索)                                        │
│  ┌──────────────────────────────────────────┐                   │
│  │  Score = α × Dense + (1-α) × Sparse      │                   │
│  │  结合两者优势，通常 α = 0.5 ~ 0.7         │                   │
│  │  优点: 兼顾语义和精确匹配                  │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 检索策略详细对比表

| 策略 | 原理 | 优势 | 劣势 | 适用场景 |
|------|------|------|------|----------|
| Dense Retrieval | 使用语义Embedding计算向量相似度 | 语义理解强，支持跨语言 | 对精确关键词匹配弱 | 通用问答、语义搜索 |
| Sparse Retrieval (BM25) | 基于词频和逆文档频率计算相关性 | 精确匹配强，无需GPU | 无法理解同义词 | 关键词搜索、技术文档 |
| Hybrid Search | 加权融合Dense和Sparse结果 | 兼顾语义和精确匹配 | 需要调优权重 | 企业级应用 |
| Multi-vector | 为同一文档生成多种表示 | 更全面的文档表示 | 存储开销大 | 复杂文档 |

### 6.3 混合检索实现

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List


def create_hybrid_retriever(
    documents: List[Document],
    persist_directory: str = "./chroma_db",
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    k: int = 4
) -> EnsembleRetriever:
    """
    创建混合检索器（Dense + Sparse）

    Args:
        documents: 文档列表
        persist_directory: 向量数据库存储路径
        dense_weight: 稠密检索权重
        sparse_weight: 稀疏检索权重
        k: 返回结果数量
    Returns:
        EnsembleRetriever混合检索器
    """
    # 创建稠密检索器（基于向量）
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    dense_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    # 创建稀疏检索器（基于BM25）
    sparse_retriever = BM25Retriever.from_documents(
        documents,
        k=k
    )

    # 组合为混合检索器
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[dense_weight, sparse_weight]
    )

    print(f"混合检索器创建完成:")
    print(f"  Dense权重: {dense_weight}, Sparse权重: {sparse_weight}")

    return hybrid_retriever


# 使用示例
if __name__ == "__main__":
    # 准备测试文档
    test_docs = [
        Document(page_content="Python是一种解释型编程语言，广泛用于数据科学和人工智能领域。"),
        Document(page_content="机器学习是人工智能的一个分支，通过数据训练模型来做出预测。"),
        Document(page_content="深度学习使用多层神经网络来学习数据的复杂表示。"),
        Document(page_content="自然语言处理(NLP)是AI处理人类语言的技术分支。"),
        Document(page_content="RAG技术将检索和生成结合起来，增强大语言模型的回答质量。"),
    ]

    retriever = create_hybrid_retriever(test_docs)

    # 测试检索
    results = retriever.invoke("什么是RAG技术？")
    for i, doc in enumerate(results):
        print(f"[{i+1}] {doc.page_content[:80]}...")
```

---

## 7. 生成优化

### 7.1 Context Compression（上下文压缩）

检索到的文档片段中往往包含很多与查询不直接相关的内容。上下文压缩可以提取出最相关的部分，减少噪声。

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def create_compressed_retriever(
    vectorstore: Chroma,
    model_name: str = "gpt-4o-mini",
    k: int = 6
):
    """
    创建带上下文压缩的检索器
    先检索较多文档，再用LLM压缩提取关键内容

    Args:
        vectorstore: 向量存储
        model_name: 用于压缩的LLM模型
        k: 初始检索数量（压缩前）
    """
    # 基础检索器
    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    # LLM压缩器
    llm = ChatOpenAI(model=model_name, temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)

    # 组合为压缩检索器
    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    return compressed_retriever


# 使用示例
if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    retriever = create_compressed_retriever(vectorstore)

    # 压缩检索
    results = retriever.invoke("公司的报销流程是什么？")
    for doc in results:
        print(f"压缩后内容: {doc.page_content[:200]}")
```

### 7.2 Lost in the Middle 问题与解决

研究表明，LLM 在处理长上下文时，对中间位置的信息关注度较低，更倾向于关注开头和结尾的内容。这就是 "Lost in the Middle" 问题。

```
┌──────────────────────────────────────────────────────────────────┐
│              Lost in the Middle 问题示意图                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  LLM 对上下文不同位置的关注度:                                     │
│                                                                  │
│  关注度                                                          │
│  ▲                                                               │
│  │ ████                                              ████        │
│  │ ████                                              ████        │
│  │ ████  ████                                  ████  ████        │
│  │ ████  ████                                  ████  ████        │
│  │ ████  ████  ████                      ████  ████  ████        │
│  │ ████  ████  ████  ████          ████  ████  ████  ████        │
│  │ ████  ████  ████  ████  ████   ████  ████  ████  ████        │
│  └──────────────────────────────────────────────────────>        │
│    开头                   中间                     结尾           │
│                                                                  │
│  解决策略:                                                        │
│  1. 将最相关文档放在开头和结尾                                      │
│  2. 减少上下文长度（压缩）                                         │
│  3. 使用 Map-Reduce 分别处理每个文档                               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

```python
from langchain.schema import Document
from typing import List


def reorder_documents_for_llm(documents: List[Document]) -> List[Document]:
    """
    重排文档顺序以缓解 Lost in the Middle 问题
    将最相关的文档放在开头和结尾，次相关的放中间

    假设输入文档已按相关性降序排列:
    原始: [1st, 2nd, 3rd, 4th, 5th]
    重排: [1st, 3rd, 5th, 4th, 2nd]

    Args:
        documents: 按相关性降序排列的文档列表
    Returns:
        重新排列的文档列表
    """
    if len(documents) <= 2:
        return documents

    reordered = []
    # 奇数位置放开头（正序）
    for i in range(0, len(documents), 2):
        reordered.append(documents[i])
    # 偶数位置放结尾（倒序）
    even_positions = [documents[i] for i in range(1, len(documents), 2)]
    reordered.extend(reversed(even_positions))

    return reordered


def format_context_with_sources(
    documents: List[Document],
    max_context_length: int = 4000
) -> str:
    """
    格式化检索结果为LLM上下文，附带来源标注

    Args:
        documents: 检索到的文档列表
        max_context_length: 最大上下文字符长度
    """
    # 重排文档顺序
    reordered_docs = reorder_documents_for_llm(documents)

    context_parts = []
    current_length = 0

    for i, doc in enumerate(reordered_docs):
        source = doc.metadata.get("file_name", "未知来源")
        page = doc.metadata.get("page_number", "")
        source_info = f"[来源: {source}"
        if page:
            source_info += f", 第{page}页"
        source_info += "]"

        section = f"---\n参考文档 {i+1} {source_info}:\n{doc.page_content}\n"

        if current_length + len(section) > max_context_length:
            break

        context_parts.append(section)
        current_length += len(section)

    return "\n".join(context_parts)
```

### 7.3 生成优化 Prompt 模板

```python
from langchain_core.prompts import ChatPromptTemplate


# RAG 问答 Prompt 模板
RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的问答助手。请根据提供的参考文档回答用户的问题。

回答要求：
1. 仅基于参考文档中的信息回答，不要编造信息
2. 如果参考文档中没有相关信息，请明确说明"根据现有资料，无法回答该问题"
3. 在回答中引用具体来源，格式为 [来源: 文件名]
4. 回答要准确、简洁、结构化
5. 如有多个角度或观点，请分点列出

参考文档：
{context}
"""),
    ("human", "{question}")
])


# 带思考链的 RAG Prompt
RAG_COT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的问答助手。请根据提供的参考文档回答用户的问题。

回答步骤：
1. 首先分析用户问题的核心意图
2. 检查参考文档中哪些内容与问题相关
3. 综合相关信息给出准确回答
4. 标注信息来源

如果文档信息不足以回答问题，请说明缺少哪方面的信息。

参考文档：
{context}
"""),
    ("human", "{question}")
])
```

---

## 8. 端到端 RAG 系统实现

### 8.1 完整 RAG 系统代码

```python
"""
完整的端到端 RAG 系统实现
技术栈: LangChain v0.2+ / ChromaDB / OpenAI
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)


@dataclass
class RAGConfig:
    """RAG系统配置"""
    # Embedding配置
    embedding_model: str = "text-embedding-3-small"

    # LLM配置
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 2000

    # 分块配置
    chunk_size: int = 800
    chunk_overlap: int = 150

    # 检索配置
    search_k: int = 4
    search_type: str = "similarity"  # similarity, mmr

    # 存储配置
    persist_directory: str = "./rag_chroma_db"
    collection_name: str = "rag_collection"


class EndToEndRAGSystem:
    """
    端到端RAG系统

    功能:
    - 多格式文档加载
    - 智能文本分块
    - 向量化存储
    - 语义检索
    - LLM增强生成
    - 引用来源标注
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()

        # 初始化Embedding模型
        self.embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model
        )

        # 初始化LLM
        self.llm = ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
        )

        # 向量存储
        self.vectorstore: Optional[Chroma] = None

        # Prompt模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的知识库问答助手。请严格根据以下参考文档回答问题。

要求：
1. 仅基于参考文档回答，不编造信息
2. 回答末尾标注引用来源 [来源: 文件名]
3. 信息不足时明确说明
4. 回答简洁、结构化

参考文档：
{context}"""),
            ("human", "{question}")
        ])

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """加载多种格式的文档"""
        all_docs = []

        for path in file_paths:
            try:
                ext = os.path.splitext(path)[1].lower()

                if ext == ".pdf":
                    loader = PyPDFLoader(path)
                elif ext in [".docx", ".doc"]:
                    loader = Docx2txtLoader(path)
                elif ext == ".txt":
                    loader = TextLoader(path, encoding="utf-8")
                else:
                    print(f"跳过不支持的格式: {path}")
                    continue

                docs = loader.load()

                # 添加元数据
                for doc in docs:
                    doc.metadata["file_name"] = os.path.basename(path)
                    doc.metadata["file_type"] = ext

                all_docs.extend(docs)
                print(f"已加载: {path} ({len(docs)} 个片段)")

            except Exception as e:
                print(f"加载失败 {path}: {e}")

        return all_docs

    def index_documents(self, documents: List[Document]) -> None:
        """将文档分块并索引到向量数据库"""
        # 分块
        chunks = self.text_splitter.split_documents(documents)
        print(f"文档分块完成: {len(documents)} 文档 -> {len(chunks)} 分块")

        # 添加分块元数据
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i

        # 创建向量存储
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.config.persist_directory,
            collection_name=self.config.collection_name,
        )

        print(f"索引完成: {len(chunks)} 个向量已存储")

    def load_index(self) -> None:
        """加载已有索引"""
        self.vectorstore = Chroma(
            persist_directory=self.config.persist_directory,
            collection_name=self.config.collection_name,
            embedding_function=self.embeddings,
        )
        count = self.vectorstore._collection.count()
        print(f"已加载索引: {count} 个向量")

    def _format_docs(self, docs: List[Document]) -> str:
        """格式化检索结果"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("file_name", "未知")
            page = doc.metadata.get("page", "")
            header = f"--- 参考文档 {i} [来源: {source}"
            if page:
                header += f", 第{page}页"
            header += "] ---"
            formatted.append(f"{header}\n{doc.page_content}")
        return "\n\n".join(formatted)

    def build_chain(self):
        """构建LCEL RAG链"""
        if self.vectorstore is None:
            raise ValueError("请先调用 index_documents() 或 load_index()")

        retriever = self.vectorstore.as_retriever(
            search_type=self.config.search_type,
            search_kwargs={"k": self.config.search_k}
        )

        # 使用LCEL构建链
        rag_chain = (
            RunnableParallel(
                context=retriever | self._format_docs,
                question=RunnablePassthrough()
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain

    def query(self, question: str) -> Dict[str, Any]:
        """
        查询RAG系统

        Args:
            question: 用户问题
        Returns:
            包含答案和来源的字典
        """
        if self.vectorstore is None:
            self.load_index()

        # 检索相关文档
        retriever = self.vectorstore.as_retriever(
            search_type=self.config.search_type,
            search_kwargs={"k": self.config.search_k}
        )
        relevant_docs = retriever.invoke(question)

        # 构建上下文
        context = self._format_docs(relevant_docs)

        # 生成回答
        chain = self.build_chain()
        answer = chain.invoke(question)

        # 提取来源信息
        sources = []
        for doc in relevant_docs:
            source_info = {
                "file_name": doc.metadata.get("file_name", "未知"),
                "page": doc.metadata.get("page", None),
                "chunk_id": doc.metadata.get("chunk_id", None),
                "content_preview": doc.page_content[:100]
            }
            sources.append(source_info)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources)
        }

    def query_with_history(
        self,
        question: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        带对话历史的查询（多轮对话）

        Args:
            question: 当前问题
            chat_history: 对话历史 [{"role": "user", "content": "..."}, ...]
        """
        if chat_history:
            # 将对话历史融入查询
            history_text = "\n".join(
                f"{msg['role']}: {msg['content']}"
                for msg in chat_history[-4:]  # 保留最近4轮
            )
            enhanced_question = (
                f"对话历史:\n{history_text}\n\n当前问题: {question}"
            )
        else:
            enhanced_question = question

        return self.query(enhanced_question)


# ============================================================
#  完整使用示例
# ============================================================

if __name__ == "__main__":
    # 1. 初始化系统
    config = RAGConfig(
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini",
        chunk_size=800,
        chunk_overlap=150,
        search_k=4,
        persist_directory="./my_rag_db",
        collection_name="demo_collection"
    )

    rag = EndToEndRAGSystem(config)

    # 2. 加载并索引文档
    documents = rag.load_documents([
        "./docs/company_policy.pdf",
        "./docs/product_manual.docx",
        "./docs/faq.txt"
    ])

    rag.index_documents(documents)

    # 3. 查询
    result = rag.query("公司的远程办公政策是什么？")

    print(f"\n问题: {result['question']}")
    print(f"\n回答: {result['answer']}")
    print(f"\n引用来源 ({result['num_sources']} 个):")
    for src in result['sources']:
        print(f"  - {src['file_name']} | 预览: {src['content_preview'][:50]}...")

    # 4. 多轮对话
    history = [
        {"role": "user", "content": "公司的远程办公政策是什么？"},
        {"role": "assistant", "content": "根据公司政策..."},
    ]

    result2 = rag.query_with_history(
        "那需要提前多久申请？",
        chat_history=history
    )
    print(f"\n追问: {result2['answer']}")
```

---

## 9. RAG 评估体系

### 9.1 RAG 评估指标概览

```
┌──────────────────────────────────────────────────────────────────┐
│                    RAG 评估指标体系                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────┐           │
│  │               检索质量评估                         │           │
│  │  ┌──────────────┐    ┌──────────────┐            │           │
│  │  │ Context       │    │ Context       │            │           │
│  │  │ Precision     │    │ Recall        │            │           │
│  │  │ (上下文精确率) │    │ (上下文召回率) │            │           │
│  │  └──────────────┘    └──────────────┘            │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
│  ┌──────────────────────────────────────────────────┐           │
│  │               生成质量评估                         │           │
│  │  ┌──────────────┐    ┌──────────────┐            │           │
│  │  │ Faithfulness  │    │ Answer        │            │           │
│  │  │ (忠实度)      │    │ Relevancy     │            │           │
│  │  │ 回答是否基于   │    │ (答案相关性)   │            │           │
│  │  │ 检索到的上下文 │    │ 回答是否切题   │            │           │
│  │  └──────────────┘    └──────────────┘            │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
│  ┌──────────────────────────────────────────────────┐           │
│  │               端到端评估                           │           │
│  │  ┌──────────────┐    ┌──────────────┐            │           │
│  │  │ Answer        │    │ Answer        │            │           │
│  │  │ Correctness   │    │ Similarity    │            │           │
│  │  │ (答案正确性)   │    │ (答案相似度)   │            │           │
│  │  └──────────────┘    └──────────────┘            │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 9.2 核心评估指标详解

| 指标 | 含义 | 计算方式 | 理想值 |
|------|------|----------|--------|
| Faithfulness（忠实度） | 回答是否基于检索到的上下文 | 将回答拆分为陈述，检查每个陈述是否可从上下文推导 | 接近1.0 |
| Answer Relevancy（答案相关性） | 回答是否与问题相关 | 从答案生成多个问题，计算与原始问题的相似度 | 接近1.0 |
| Context Precision（上下文精确率） | 检索结果中相关文档的比例 | 相关检索结果数 / 总检索结果数 | 越高越好 |
| Context Recall（上下文召回率） | 所有相关信息是否都被检索到 | 被检索到的相关信息 / 所有相关信息 | 越高越好 |

### 9.3 使用 RAGAS 框架评估

```python
"""
使用 RAGAS 框架评估 RAG 系统
安装: pip install ragas
"""

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset


def evaluate_rag_system(
    questions: list,
    answers: list,
    contexts: list,
    ground_truths: list
) -> dict:
    """
    使用RAGAS评估RAG系统

    Args:
        questions: 测试问题列表
        answers: RAG系统生成的回答列表
        contexts: 检索到的上下文列表（每个问题对应一个上下文列表）
        ground_truths: 标准答案列表

    Returns:
        评估结果字典
    """
    # 构建评估数据集
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # 执行评估
    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
    )

    return result


def create_test_dataset() -> dict:
    """创建测试数据集示例"""
    return {
        "questions": [
            "公司的年假政策是什么？",
            "如何申请报销？",
            "新员工入职流程包括哪些步骤？"
        ],
        "answers": [
            "根据公司政策，员工入职满一年后可享受5天年假，满3年为10天，满5年为15天。",
            "报销流程为：填写报销单 -> 部门经理审批 -> 财务审核 -> 打款到工资账户。",
            "新员工入职流程包括：1.签署劳动合同 2.领取工牌 3.IT设备配置 4.部门介绍 5.导师分配。"
        ],
        "contexts": [
            ["公司年假政策：入职满1年享有5天年假，满3年10天，满5年15天，满10年20天。"],
            ["报销流程：员工填写报销单并附发票 -> 直属经理审批 -> 财务部审核 -> 5个工作日内打款。"],
            ["新员工入职checklist：签合同、领工牌、IT配置、部门介绍、导师分配、安全培训。"]
        ],
        "ground_truths": [
            "入职满1年5天年假，满3年10天，满5年15天，满10年20天。",
            "填写报销单附发票，经理审批，财务审核，5个工作日内打款。",
            "签合同、领工牌、IT配置、部门介绍、导师分配、安全培训。"
        ]
    }


# 使用示例
if __name__ == "__main__":
    # 准备测试数据
    test_data = create_test_dataset()

    # 运行评估
    results = evaluate_rag_system(
        questions=test_data["questions"],
        answers=test_data["answers"],
        contexts=test_data["contexts"],
        ground_truths=test_data["ground_truths"]
    )

    # 输出结果
    print("=" * 50)
    print("RAG 系统评估结果")
    print("=" * 50)
    print(f"Faithfulness (忠实度):        {results['faithfulness']:.4f}")
    print(f"Answer Relevancy (答案相关性): {results['answer_relevancy']:.4f}")
    print(f"Context Precision (上下文精确率): {results['context_precision']:.4f}")
    print(f"Context Recall (上下文召回率):   {results['context_recall']:.4f}")
```

### 9.4 自定义评估Pipeline

```python
"""
自定义RAG评估Pipeline
当不使用RAGAS时的替代方案
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json
from typing import List, Dict


class RAGEvaluator:
    """
    自定义RAG评估器
    使用LLM作为评判者（LLM-as-Judge）
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0)

    def evaluate_faithfulness(
        self, question: str, answer: str, context: str
    ) -> Dict:
        """评估回答的忠实度"""
        prompt = ChatPromptTemplate.from_template("""
请评估以下回答是否忠实于给定的上下文。

问题: {question}
上下文: {context}
回答: {answer}

评估标准：
- 回答中的每个事实性陈述是否都能在上下文中找到支持
- 回答是否包含了上下文中不存在的信息（幻觉）

请以JSON格式返回：
{{
    "score": <0.0到1.0的分数>,
    "reasoning": "<评估理由>",
    "hallucinated_claims": ["<幻觉内容列表>"]
}}
""")
        chain = prompt | self.llm
        result = chain.invoke({
            "question": question,
            "answer": answer,
            "context": context
        })

        try:
            return json.loads(result.content)
        except json.JSONDecodeError:
            return {"score": 0.0, "reasoning": result.content, "hallucinated_claims": []}

    def evaluate_relevancy(
        self, question: str, answer: str
    ) -> Dict:
        """评估回答与问题的相关性"""
        prompt = ChatPromptTemplate.from_template("""
请评估以下回答与问题的相关性。

问题: {question}
回答: {answer}

评估标准：
- 回答是否直接针对问题
- 回答是否完整覆盖问题的各个方面
- 回答是否包含过多与问题无关的信息

请以JSON格式返回：
{{
    "score": <0.0到1.0的分数>,
    "reasoning": "<评估理由>"
}}
""")
        chain = prompt | self.llm
        result = chain.invoke({"question": question, "answer": answer})

        try:
            return json.loads(result.content)
        except json.JSONDecodeError:
            return {"score": 0.0, "reasoning": result.content}

    def run_evaluation(
        self,
        test_cases: List[Dict[str, str]]
    ) -> Dict:
        """
        运行完整评估

        Args:
            test_cases: [{"question": ..., "answer": ..., "context": ..., "ground_truth": ...}]
        """
        total_faithfulness = 0
        total_relevancy = 0

        detailed_results = []

        for i, case in enumerate(test_cases):
            print(f"评估 {i+1}/{len(test_cases)}: {case['question'][:30]}...")

            faith_result = self.evaluate_faithfulness(
                case["question"], case["answer"], case["context"]
            )
            rel_result = self.evaluate_relevancy(
                case["question"], case["answer"]
            )

            total_faithfulness += faith_result.get("score", 0)
            total_relevancy += rel_result.get("score", 0)

            detailed_results.append({
                "question": case["question"],
                "faithfulness": faith_result,
                "relevancy": rel_result
            })

        n = len(test_cases)
        return {
            "avg_faithfulness": total_faithfulness / n if n > 0 else 0,
            "avg_relevancy": total_relevancy / n if n > 0 else 0,
            "num_test_cases": n,
            "detailed_results": detailed_results
        }


# 使用示例
if __name__ == "__main__":
    evaluator = RAGEvaluator(model="gpt-4o-mini")

    test_cases = [
        {
            "question": "Python的GIL是什么？",
            "answer": "GIL是全局解释器锁，它确保同一时刻只有一个线程执行Python字节码。",
            "context": "Python的GIL(Global Interpreter Lock)是一个互斥锁，确保同一时刻只有一个线程执行Python字节码。这限制了Python多线程的并行性。",
            "ground_truth": "GIL是全局解释器锁，限制了Python多线程的真正并行。"
        }
    ]

    results = evaluator.run_evaluation(test_cases)

    print(f"\n评估结果:")
    print(f"  平均忠实度: {results['avg_faithfulness']:.2f}")
    print(f"  平均相关性: {results['avg_relevancy']:.2f}")
```

---

## 10. 最佳实践与常见问题

### 10.1 RAG 系统优化清单

```
┌──────────────────────────────────────────────────────────────────┐
│                   RAG 优化清单                                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  数据层优化:                                                      │
│  [x] 选择合适的分块大小（推荐500-1000字符）                         │
│  [x] 使用重叠分块（overlap 10-20%）                               │
│  [x] 保留文档结构信息（标题、段落层级）                              │
│  [x] 清洗噪声数据（页眉页脚、重复内容）                             │
│  [x] 添加丰富的元数据（来源、日期、类别）                           │
│                                                                  │
│  检索层优化:                                                      │
│  [x] 使用混合检索（Dense + Sparse）                               │
│  [x] 添加Reranker重排序                                          │
│  [x] 实现查询重写/扩展                                            │
│  [x] 考虑MMR多样性检索                                           │
│  [x] 调优检索数量k（通常4-8个）                                   │
│                                                                  │
│  生成层优化:                                                      │
│  [x] 设计精确的Prompt模板                                        │
│  [x] 实现上下文压缩                                              │
│  [x] 处理Lost in the Middle问题                                 │
│  [x] 添加引用来源标注                                            │
│  [x] 使用低temperature（0-0.3）                                  │
│                                                                  │
│  评估与监控:                                                      │
│  [x] 建立评估数据集                                              │
│  [x] 使用RAGAS框架定期评估                                       │
│  [x] 监控检索质量和生成质量                                       │
│  [x] 收集用户反馈持续迭代                                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 10.2 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 回答包含幻觉 | Prompt约束不足或检索内容不相关 | 强化Prompt中的"仅基于上下文回答"指令，加入Reranker |
| 检索结果不相关 | 分块策略不当或Embedding不匹配 | 优化分块大小，尝试不同Embedding模型 |
| 回答不完整 | 相关信息分散在多个分块中 | 增大k值，使用Parent Document Retriever |
| 延迟过高 | 向量搜索慢或LLM响应慢 | 优化索引（HNSW参数），使用流式输出 |
| 无法回答 | 知识库缺少相关信息 | 扩充知识库，提供回退机制 |

### 10.3 Embedding 模型选择

| 模型 | 维度 | 中文支持 | 价格 | 推荐场景 |
|------|------|----------|------|----------|
| text-embedding-3-small | 1536 | 好 | $0.02/1M tokens | 通用场景，性价比高 |
| text-embedding-3-large | 3072 | 好 | $0.13/1M tokens | 高精度场景 |
| BGE-large-zh | 1024 | 优秀 | 免费（本地） | 中文场景，私有部署 |
| BGE-M3 | 1024 | 优秀 | 免费（本地） | 多语言混合场景 |
| Cohere embed-v3 | 1024 | 好 | $0.1/1M tokens | 多语言检索 |

### 10.4 分块大小选择指南

| 文档类型 | 推荐分块大小 | 重叠大小 | 说明 |
|----------|------------|----------|------|
| 技术文档 | 800-1200 | 150-200 | 保持代码块和段落完整 |
| 法律合同 | 500-800 | 100-150 | 条款通常较短，需精确匹配 |
| 新闻文章 | 600-1000 | 100-200 | 按段落分块效果好 |
| FAQ | 300-500 | 50-100 | 每个QA对作为独立块 |
| 学术论文 | 1000-1500 | 200-300 | 保持论证连贯性 |

---

## 总结

本教程完整介绍了 RAG 系统的架构设计，涵盖以下核心内容：

1. **RAG 核心概念**：理解 RAG 的工作原理及其相比 Fine-tuning 的优势
2. **三代架构演进**：从 Naive RAG 到 Advanced RAG 再到 Modular RAG 的发展路径
3. **文档加载器**：支持 PDF、Word、Markdown、HTML 等多格式加载
4. **文档处理 Pipeline**：加载、清洗、分块、向量化、存储的完整流程
5. **检索策略**：Dense、Sparse、Hybrid 三种检索策略及其适用场景
6. **生成优化**：上下文压缩、Lost in the Middle 处理、Prompt 工程
7. **端到端实现**：基于 LangChain + ChromaDB + OpenAI 的完整代码
8. **评估体系**：RAGAS 框架与自定义评估方案

## 参考资源

- [LangChain 官方文档](https://python.langchain.com/)
- [RAGAS 评估框架](https://docs.ragas.io/)
- [ChromaDB 文档](https://docs.trychroma.com/)
- [论文: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [论文: Lost in the Middle](https://arxiv.org/abs/2307.03172)
- [论文: Self-RAG](https://arxiv.org/abs/2310.11511)

---

**创建时间**: 2024-01
**最后更新**: 2024-01
