# RAG 实战项目：企业知识库问答系统

## 目录
1. [项目概述与需求分析](#1-项目概述与需求分析)
2. [系统架构设计](#2-系统架构设计)
3. [环境搭建与项目结构](#3-环境搭建与项目结构)
4. [核心模块：文档处理引擎](#4-核心模块文档处理引擎)
5. [核心模块：向量存储服务](#5-核心模块向量存储服务)
6. [核心模块：RAG 检索与生成](#6-核心模块rag-检索与生成)
7. [对话历史管理](#7-对话历史管理)
8. [引用溯源与来源标注](#8-引用溯源与来源标注)
9. [FastAPI 后端完整实现](#9-fastapi-后端完整实现)
10. [Streamlit 前端界面](#10-streamlit-前端界面)
11. [Docker 部署方案](#11-docker-部署方案)
12. [测试与评估](#12-测试与评估)
13. [生产环境优化](#13-生产环境优化)
14. [总结与最佳实践](#14-总结与最佳实践)

---

## 1. 项目概述与需求分析

### 1.1 项目背景

企业内部积累了大量的技术文档、产品手册、FAQ、规章制度等知识资料。传统的关键词搜索无法理解用户的语义意图，而直接使用 LLM 又无法获取企业的私有知识。本项目构建一个基于 RAG 的企业知识库问答系统，让员工能够通过自然语言对话快速获取准确的知识信息。

### 1.2 功能需求

```
┌──────────────────────────────────────────────────────────────────────┐
│                    企业知识库问答系统 - 功能需求                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  核心功能:                                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  1. 文档管理                                                  │    │
│  │     - 支持 PDF/Word/Markdown/TXT 文档上传                    │    │
│  │     - 自动解析、分块、向量化                                  │    │
│  │     - 文档列表查看与删除                                      │    │
│  │                                                               │    │
│  │  2. 智能问答                                                  │    │
│  │     - 自然语言提问                                            │    │
│  │     - 基于知识库的准确回答                                    │    │
│  │     - 回答附带来源引用                                        │    │
│  │                                                               │    │
│  │  3. 对话管理                                                  │    │
│  │     - 多轮对话支持                                            │    │
│  │     - 对话历史保存                                            │    │
│  │     - 上下文理解                                              │    │
│  │                                                               │    │
│  │  4. 引用溯源                                                  │    │
│  │     - 标注回答来源文档                                        │    │
│  │     - 显示相关原文片段                                        │    │
│  │     - 置信度评分                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  非功能需求:                                                          │
│  - 响应时间 < 5秒                                                    │
│  - 支持并发 50 用户                                                   │
│  - 知识库容量 > 10000 文档                                            │
│  - Docker 一键部署                                                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.3 技术选型

| 组件 | 技术方案 | 选型理由 |
|------|----------|----------|
| 后端框架 | FastAPI | 高性能异步框架，自动生成 API 文档 |
| 前端界面 | Streamlit | 快速构建数据应用 UI，Python 原生 |
| LLM 框架 | LangChain | RAG 工具链完善，社区活跃 |
| 向量数据库 | ChromaDB / Milvus | Chroma 适合开发，Milvus 适合生产 |
| Embedding | OpenAI text-embedding-3-small | 性价比高，中文支持好 |
| LLM | GPT-4o-mini / GPT-4o | 生成质量高，支持长上下文 |
| 部署 | Docker Compose | 一键部署，环境隔离 |

---

## 2. 系统架构设计

### 2.1 整体架构图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    企业知识库问答系统 - 系统架构                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────── 前端层 ─────────────────────────────────┐    │
│  │                                                                  │    │
│  │  ┌──────────────────────────────────────────────────────┐      │    │
│  │  │              Streamlit Web 界面                        │      │    │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │      │    │
│  │  │  │ 文档上传  │  │ 对话问答  │  │ 知识库管理        │  │      │    │
│  │  │  │ 页面     │  │ 页面     │  │ 页面             │  │      │    │
│  │  │  └──────────┘  └──────────┘  └──────────────────┘  │      │    │
│  │  └──────────────────────┬───────────────────────────────┘      │    │
│  │                          │ HTTP/REST API                        │    │
│  └──────────────────────────┼─────────────────────────────────────┘    │
│                              │                                          │
│  ┌──────────────────────────v───────── API 层 ────────────────────┐    │
│  │                                                                  │    │
│  │  ┌──────────────────── FastAPI ────────────────────────────┐  │    │
│  │  │                                                          │  │    │
│  │  │  POST /api/documents/upload   文档上传                   │  │    │
│  │  │  GET  /api/documents/         文档列表                   │  │    │
│  │  │  DELETE /api/documents/{id}   删除文档                   │  │    │
│  │  │  POST /api/chat               对话问答                   │  │    │
│  │  │  GET  /api/chat/history/{id}  对话历史                   │  │    │
│  │  │  GET  /api/health             健康检查                   │  │    │
│  │  │                                                          │  │    │
│  │  └──────────────────────┬───────────────────────────────────┘  │    │
│  │                          │                                      │    │
│  └──────────────────────────┼──────────────────────────────────────┘    │
│                              │                                          │
│  ┌──────────────────────────v───────── 业务层 ────────────────────┐    │
│  │                                                                  │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │    │
│  │  │ 文档处理引擎  │  │ RAG 检索引擎  │  │ 对话管理器        │    │    │
│  │  │              │  │              │  │                  │    │    │
│  │  │ - 文档解析   │  │ - 查询重写   │  │ - 历史存储       │    │    │
│  │  │ - 文本清洗   │  │ - 向量检索   │  │ - 上下文构建     │    │    │
│  │  │ - 智能分块   │  │ - 混合检索   │  │ - 会话管理       │    │    │
│  │  │ - 元数据提取 │  │ - 重排序     │  │                  │    │    │
│  │  └──────┬───────┘  │ - LLM生成   │  └──────────────────┘    │    │
│  │          │          │ - 引用标注   │                          │    │
│  │          │          └──────┬───────┘                          │    │
│  │          │                 │                                   │    │
│  └──────────┼─────────────────┼───────────────────────────────────┘    │
│              │                 │                                        │
│  ┌──────────v─────────────────v───────── 数据层 ──────────────────┐    │
│  │                                                                  │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │    │
│  │  │ ChromaDB /   │  │ SQLite       │  │ 文件系统          │    │    │
│  │  │ Milvus       │  │ (对话历史/   │  │ (上传文档存储)    │    │    │
│  │  │ (向量存储)   │  │  文档元数据)  │  │                  │    │    │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘    │    │
│  │                                                                  │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌───────────────────── 外部服务 ─────────────────────────────────┐    │
│  │  ┌──────────────┐  ┌──────────────┐                            │    │
│  │  │ OpenAI API   │  │ Cohere API   │                            │    │
│  │  │ (Embedding   │  │ (Reranker)   │                            │    │
│  │  │  + LLM)      │  │  [可选]      │                            │    │
│  │  └──────────────┘  └──────────────┘                            │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流图

```
┌──────────────────────────────────────────────────────────────────────┐
│                         数据流示意图                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  文档摄入流程:                                                        │
│                                                                      │
│  用户上传 ──> 文档解析 ──> 文本清洗 ──> 智能分块 ──> Embedding ──>   │
│              (PDF/Word/    (去噪/     (800字符/   (OpenAI)          │
│               MD/TXT)      格式化)    150重叠)                       │
│                                                          │           │
│                                                          v           │
│                                              ┌──────────────────┐   │
│                                              │  向量数据库存储    │   │
│                                              │  (ChromaDB/Milvus)│   │
│                                              └──────────────────┘   │
│                                                                      │
│  问答流程:                                                            │
│                                                                      │
│  用户提问 ──> 查询理解 ──> 向量检索 ──> 结果重排 ──> Prompt构建 ──> │
│              (上下文     (Top-10    (Reranker   (上下文+        │
│               融合)       候选)      Top-5)      问题+历史)      │
│                                                          │           │
│                                                          v           │
│                                              ┌──────────────────┐   │
│                                              │  LLM生成回答      │   │
│                                              │  + 引用来源标注   │   │
│                                              └──────────────────┘   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. 环境搭建与项目结构

### 3.1 项目目录结构

```
enterprise-rag-system/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 主入口
│   ├── config.py               # 配置管理
│   ├── models.py               # Pydantic 数据模型
│   ├── api/
│   │   ├── __init__.py
│   │   ├── documents.py        # 文档管理 API
│   │   └── chat.py             # 对话问答 API
│   ├── core/
│   │   ├── __init__.py
│   │   ├── document_processor.py   # 文档处理引擎
│   │   ├── vector_store.py         # 向量存储服务
│   │   ├── rag_engine.py           # RAG 检索与生成
│   │   ├── chat_history.py         # 对话历史管理
│   │   └── citation.py             # 引用溯源
│   └── utils/
│       ├── __init__.py
│       └── logger.py               # 日志工具
├── frontend/
│   └── streamlit_app.py        # Streamlit 前端
├── data/
│   ├── uploads/                # 上传文档存储
│   ├── chroma_db/              # ChromaDB 持久化
│   └── chat_history.db         # 对话历史数据库
├── tests/
│   ├── test_document_processor.py
│   ├── test_rag_engine.py
│   └── test_api.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

### 3.2 依赖安装

```bash
# requirements.txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6
pydantic==2.5.3
pydantic-settings==2.1.0

# LangChain
langchain==0.2.16
langchain-openai==0.1.25
langchain-community==0.2.16
langchain-text-splitters==0.2.4

# 文档处理
pypdf==4.0.1
python-docx==1.1.0
unstructured==0.12.0
markdown==3.5.2

# 向量数据库
chromadb==0.4.24
# pymilvus==2.4.0  # 如果使用 Milvus

# 其他
tiktoken==0.5.2
python-dotenv==1.0.0
aiosqlite==0.19.0
httpx==0.26.0

# 前端
streamlit==1.31.0
```

```bash
# 安装依赖
pip install -r requirements.txt
```

### 3.3 配置管理

```python
"""
app/config.py - 配置管理
使用 pydantic-settings 管理环境变量
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """系统配置"""

    # 应用配置
    APP_NAME: str = "企业知识库问答系统"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # OpenAI 配置
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: Optional[str] = None
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2000

    # 向量数据库配置
    VECTOR_DB_TYPE: str = "chroma"  # "chroma" 或 "milvus"
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION: str = "enterprise_kb"
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # 文档处理配置
    UPLOAD_DIR: str = "./data/uploads"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    MAX_FILE_SIZE_MB: int = 50

    # 检索配置
    SEARCH_K: int = 6
    RERANK_ENABLED: bool = False
    RERANK_TOP_K: int = 4

    # 对话配置
    MAX_HISTORY_TURNS: int = 5
    CHAT_DB_PATH: str = "./data/chat_history.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例
settings = Settings()


# 确保必要目录存在
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.CHAT_DB_PATH), exist_ok=True)
```

### 3.4 数据模型定义

```python
"""
app/models.py - Pydantic 数据模型
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentStatus(str, Enum):
    """文档处理状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentInfo(BaseModel):
    """文档信息"""
    id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int = 0
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = None


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    total: int
    documents: List[DocumentInfo]


class ChatRequest(BaseModel):
    """对话请求"""
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    use_history: bool = True


class SourceDocument(BaseModel):
    """来源文档信息"""
    filename: str
    page_number: Optional[int] = None
    chunk_index: int
    content_preview: str
    relevance_score: float


class ChatResponse(BaseModel):
    """对话响应"""
    answer: str
    conversation_id: str
    sources: List[SourceDocument]
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time_ms: float


class ChatHistoryItem(BaseModel):
    """对话历史条目"""
    role: str  # "user" 或 "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    sources: Optional[List[SourceDocument]] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    vector_db_status: str
    document_count: int
```

---

## 4. 核心模块：文档处理引擎

### 4.1 文档处理流程

```
┌──────────────────────────────────────────────────────────────────────┐
│                    文档处理引擎工作流程                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │ 1. 文档   │──>│ 2. 文本   │──>│ 3. 智能   │──>│ 4. 元数据 │       │
│  │    解析   │   │    清洗   │   │    分块   │   │    丰富   │       │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘        │
│                                                                      │
│  支持格式:      去除噪声:      策略:          附加信息:              │
│  - PDF         - 页眉页脚     - 递归字符     - 文件名               │
│  - Word        - 控制字符     - 中文优化     - 页码                 │
│  - Markdown    - 多余空白     - 800字符      - 分块序号             │
│  - TXT         - 去重         - 150重叠      - 文档类型             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 完整代码实现

```python
"""
app/core/document_processor.py - 文档处理引擎
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    文档处理引擎
    负责文档的解析、清洗、分块和元数据丰富
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf": "pdf",
        ".docx": "word",
        ".doc": "word",
        ".md": "markdown",
        ".txt": "text",
    }

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        # 中文优化的分隔符列表
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n", "\n", "。", "！", "？", "；",
                "，", ". ", "! ", "? ", "; ", ", ", " ", ""
            ],
            length_function=len,
        )

    def process_file(self, file_path: str) -> Tuple[List[Document], dict]:
        """
        处理单个文件的完整流程

        Args:
            file_path: 文件路径

        Returns:
            (分块后的Document列表, 处理统计信息)
        """
        stats = {
            "filename": os.path.basename(file_path),
            "raw_pages": 0,
            "cleaned_pages": 0,
            "chunks": 0,
            "total_chars": 0,
        }

        # Step 1: 加载文档
        logger.info(f"加载文档: {file_path}")
        raw_documents = self._load_document(file_path)
        stats["raw_pages"] = len(raw_documents)

        if not raw_documents:
            raise ValueError(f"文档加载失败或内容为空: {file_path}")

        # Step 2: 清洗文本
        logger.info("清洗文本...")
        cleaned_documents = self._clean_documents(raw_documents)
        stats["cleaned_pages"] = len(cleaned_documents)

        # Step 3: 分块
        logger.info("文本分块...")
        chunks = self.text_splitter.split_documents(cleaned_documents)

        # Step 4: 丰富元数据
        logger.info("丰富元数据...")
        filename = os.path.basename(file_path)
        file_id = self._generate_file_id(file_path)

        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "file_id": file_id,
                "filename": filename,
                "file_type": Path(file_path).suffix.lower(),
                "chunk_index": i,
                "chunk_size": len(chunk.page_content),
                "total_chunks": len(chunks),
            })

        stats["chunks"] = len(chunks)
        stats["total_chars"] = sum(len(c.page_content) for c in chunks)

        logger.info(
            f"文档处理完成: {filename} | "
            f"{stats['raw_pages']} 页 -> {stats['chunks']} 分块"
        )

        return chunks, stats

    def _load_document(self, file_path: str) -> List[Document]:
        """根据文件类型选择合适的加载器"""
        ext = Path(file_path).suffix.lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件格式: {ext}，"
                f"支持: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext in (".docx", ".doc"):
                loader = Docx2txtLoader(file_path)
            elif ext == ".md":
                loader = UnstructuredMarkdownLoader(file_path)
            elif ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                raise ValueError(f"未实现的加载器: {ext}")

            return loader.load()

        except Exception as e:
            logger.error(f"文档加载失败 {file_path}: {e}")
            raise

    def _clean_documents(self, documents: List[Document]) -> List[Document]:
        """清洗文档内容"""
        cleaned = []

        for doc in documents:
            text = doc.page_content

            # 去除多余空白行
            text = re.sub(r'\n{3,}', '\n\n', text)
            # 去除行内多余空白
            text = re.sub(r'[ \t]+', ' ', text)
            # 去除控制字符
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
            # 去除页眉页脚模式
            text = re.sub(r'第\s*\d+\s*页.*?共\s*\d+\s*页', '', text)
            text = re.sub(
                r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE
            )
            # 标准化引号
            text = text.replace('\u201c', '"').replace('\u201d', '"')
            text = text.replace('\u2018', "'").replace('\u2019', "'")

            text = text.strip()

            # 过滤过短内容
            if len(text) >= 20:
                doc.page_content = text
                cleaned.append(doc)

        return cleaned

    @staticmethod
    def _generate_file_id(file_path: str) -> str:
        """基于文件内容生成唯一ID"""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]

    @staticmethod
    def get_supported_extensions() -> List[str]:
        """返回支持的文件扩展名列表"""
        return list(DocumentProcessor.SUPPORTED_EXTENSIONS.keys())
```

---

## 5. 核心模块：向量存储服务

### 5.1 向量存储抽象层

```python
"""
app/core/vector_store.py - 向量存储服务
支持 ChromaDB 和 Milvus 两种后端
"""

import os
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """向量存储抽象基类"""

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> int:
        """添加文档，返回添加数量"""
        pass

    @abstractmethod
    def search(
        self, query: str, k: int = 5, filter_dict: Dict = None
    ) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        pass

    @abstractmethod
    def delete_by_file_id(self, file_id: str) -> int:
        """按文件ID删除文档，返回删除数量"""
        pass

    @abstractmethod
    def get_document_count(self) -> int:
        """获取文档总数"""
        pass


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB 向量存储实现"""

    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = None,
        embedding_model: str = None,
    ):
        self.persist_directory = (
            persist_directory or settings.CHROMA_PERSIST_DIR
        )
        self.collection_name = (
            collection_name or settings.CHROMA_COLLECTION
        )

        # 初始化 Embedding 模型
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model or settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )

        # 初始化 ChromaDB
        from langchain_community.vectorstores import Chroma

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

        count = self.get_document_count()
        logger.info(
            f"ChromaDB 初始化完成 | 集合: {self.collection_name} | "
            f"文档数: {count}"
        )

    def add_documents(self, documents: List[Document]) -> int:
        """添加文档到 ChromaDB"""
        if not documents:
            return 0

        self.vectorstore.add_documents(documents)
        logger.info(f"已添加 {len(documents)} 个文档分块到 ChromaDB")
        return len(documents)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Dict = None,
    ) -> List[Dict[str, Any]]:
        """
        搜索相似文档

        Returns:
            [{"document": Document, "score": float}, ...]
        """
        kwargs = {"k": k}
        if filter_dict:
            kwargs["filter"] = filter_dict

        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, **kwargs
        )

        formatted = []
        for doc, score in results:
            formatted.append({
                "document": doc,
                "score": score,
                "content": doc.page_content,
                "metadata": doc.metadata,
            })

        return formatted

    def delete_by_file_id(self, file_id: str) -> int:
        """按文件ID删除所有相关分块"""
        collection = self.vectorstore._collection

        # 查找匹配的文档
        results = collection.get(
            where={"file_id": file_id},
            include=["metadatas"]
        )

        if results and results["ids"]:
            ids_to_delete = results["ids"]
            collection.delete(ids=ids_to_delete)
            logger.info(f"已删除 file_id={file_id} 的 {len(ids_to_delete)} 个分块")
            return len(ids_to_delete)

        return 0

    def get_document_count(self) -> int:
        """获取文档总数"""
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0

    def get_file_list(self) -> List[Dict[str, Any]]:
        """获取已索引的文件列表"""
        collection = self.vectorstore._collection
        results = collection.get(include=["metadatas"])

        files = {}
        if results and results["metadatas"]:
            for meta in results["metadatas"]:
                file_id = meta.get("file_id", "unknown")
                if file_id not in files:
                    files[file_id] = {
                        "file_id": file_id,
                        "filename": meta.get("filename", "unknown"),
                        "file_type": meta.get("file_type", ""),
                        "chunk_count": 0,
                    }
                files[file_id]["chunk_count"] += 1

        return list(files.values())


class MilvusVectorStore(BaseVectorStore):
    """Milvus 向量存储实现（生产环境）"""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection_name: str = None,
        embedding_model: str = None,
    ):
        from langchain_community.vectorstores import Milvus

        self.host = host or settings.MILVUS_HOST
        self.port = port or settings.MILVUS_PORT
        self.collection_name = collection_name or settings.CHROMA_COLLECTION

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model or settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )

        self.vectorstore = Milvus(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_args={
                "host": self.host,
                "port": self.port,
            },
        )

        logger.info(
            f"Milvus 初始化完成 | {self.host}:{self.port} | "
            f"集合: {self.collection_name}"
        )

    def add_documents(self, documents: List[Document]) -> int:
        if not documents:
            return 0
        self.vectorstore.add_documents(documents)
        return len(documents)

    def search(
        self, query: str, k: int = 5, filter_dict: Dict = None
    ) -> List[Dict[str, Any]]:
        kwargs = {"k": k}
        results = self.vectorstore.similarity_search_with_score(
            query, **kwargs
        )
        formatted = []
        for doc, score in results:
            formatted.append({
                "document": doc,
                "score": float(score),
                "content": doc.page_content,
                "metadata": doc.metadata,
            })
        return formatted

    def delete_by_file_id(self, file_id: str) -> int:
        expr = f'file_id == "{file_id}"'
        self.vectorstore.col.delete(expr)
        return -1  # Milvus 不方便返回准确删除数

    def get_document_count(self) -> int:
        try:
            return self.vectorstore.col.num_entities
        except Exception:
            return 0


def create_vector_store() -> BaseVectorStore:
    """
    工厂函数：根据配置创建向量存储实例

    Returns:
        BaseVectorStore 实例
    """
    if settings.VECTOR_DB_TYPE == "chroma":
        return ChromaVectorStore()
    elif settings.VECTOR_DB_TYPE == "milvus":
        return MilvusVectorStore()
    else:
        raise ValueError(
            f"不支持的向量数据库类型: {settings.VECTOR_DB_TYPE}"
        )
```

---

## 6. 核心模块：RAG 检索与生成

### 6.1 RAG 引擎架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                      RAG 引擎处理流程                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  用户问题 + 对话历史                                                  │
│       │                                                              │
│       v                                                              │
│  ┌──────────────────┐                                                │
│  │ 1. 查询预处理     │  - 融合对话历史上下文                          │
│  │                   │  - 生成独立完整的查询                          │
│  └────────┬─────────┘                                                │
│           │                                                          │
│           v                                                          │
│  ┌──────────────────┐                                                │
│  │ 2. 向量检索       │  - 从向量数据库检索 Top-K 候选文档            │
│  │    (Top-10)      │  - 返回文档内容 + 相关性分数                  │
│  └────────┬─────────┘                                                │
│           │                                                          │
│           v                                                          │
│  ┌──────────────────┐                                                │
│  │ 3. 结果过滤       │  - 过滤低相关性结果 (score < 阈值)           │
│  │    + 重排序      │  - [可选] Reranker 重排序                     │
│  │    (Top-5)       │  - 截取最终 Top-K                             │
│  └────────┬─────────┘                                                │
│           │                                                          │
│           v                                                          │
│  ┌──────────────────┐                                                │
│  │ 4. Prompt 构建    │  - 将检索结果格式化为上下文                   │
│  │                   │  - 注入系统指令和约束                         │
│  │                   │  - 添加引用标注要求                           │
│  └────────┬─────────┘                                                │
│           │                                                          │
│           v                                                          │
│  ┌──────────────────┐                                                │
│  │ 5. LLM 生成       │  - 调用 GPT-4o-mini 生成回答                │
│  │    + 引用解析     │  - 解析回答中的引用标注                       │
│  └──────────────────┘                                                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.2 完整代码实现

```python
"""
app/core/rag_engine.py - RAG 检索与生成引擎
"""

import time
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from app.config import settings
from app.core.vector_store import BaseVectorStore
from app.models import SourceDocument
import logging

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    RAG 检索与生成引擎

    功能:
    - 查询预处理（融合对话历史）
    - 向量检索 + 过滤
    - LLM 生成（含引用标注）
    """

    # 系统 Prompt 模板
    SYSTEM_PROMPT = """你是一个专业的企业知识库问答助手。请严格根据以下参考文档回答用户的问题。

回答要求:
1. 仅基于参考文档中的信息回答，不要编造或推测
2. 如果参考文档中没有相关信息，请明确说明"根据现有知识库资料，暂无法回答该问题"
3. 在回答中引用来源，格式为 [来源: 文件名]
4. 回答要准确、简洁、结构化
5. 如有多个要点，请分点列出
6. 使用中文回答

参考文档:
{context}"""

    # 查询改写 Prompt（融合对话历史）
    QUERY_REWRITE_PROMPT = """根据对话历史和最新问题，生成一个独立完整的搜索查询。
这个查询应该不依赖对话上下文就能被理解。只返回改写后的查询，不要其他内容。

对话历史:
{history}

最新问题: {question}

改写后的查询:"""

    def __init__(self, vector_store: BaseVectorStore):
        self.vector_store = vector_store

        # 初始化 LLM
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            openai_api_key=settings.OPENAI_API_KEY,
        )

        # 用于查询改写的轻量 LLM
        self.rewrite_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=200,
            openai_api_key=settings.OPENAI_API_KEY,
        )

        # 生成 Prompt
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        # 查询改写 Prompt
        self.rewrite_prompt = ChatPromptTemplate.from_template(
            self.QUERY_REWRITE_PROMPT
        )

        logger.info("RAG 引擎初始化完成")

    def query(
        self,
        question: str,
        chat_history: List[Dict[str, str]] = None,
        search_k: int = None,
        score_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """
        执行 RAG 查询

        Args:
            question: 用户问题
            chat_history: 对话历史
            search_k: 检索数量
            score_threshold: 最低相关性阈值

        Returns:
            {
                "answer": str,
                "sources": [SourceDocument],
                "confidence": float,
                "processing_time_ms": float,
            }
        """
        start_time = time.time()
        search_k = search_k or settings.SEARCH_K

        # Step 1: 查询预处理
        search_query = self._preprocess_query(question, chat_history)
        logger.info(f"搜索查询: {search_query}")

        # Step 2: 向量检索
        search_results = self.vector_store.search(
            query=search_query,
            k=search_k,
        )
        logger.info(f"检索到 {len(search_results)} 个候选文档")

        # Step 3: 过滤低相关性结果
        filtered_results = [
            r for r in search_results
            if r["score"] >= score_threshold
        ]

        if not filtered_results:
            # 如果全部被过滤，保留得分最高的结果
            filtered_results = search_results[:2] if search_results else []

        logger.info(f"过滤后保留 {len(filtered_results)} 个文档")

        # Step 4: 构建上下文
        context = self._build_context(filtered_results)
        sources = self._extract_sources(filtered_results)

        # Step 5: LLM 生成
        chain = self.qa_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": question,
        })

        # 计算置信度（基于检索结果的平均分数）
        if filtered_results:
            confidence = sum(
                r["score"] for r in filtered_results
            ) / len(filtered_results)
            confidence = min(max(confidence, 0.0), 1.0)
        else:
            confidence = 0.0

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"RAG 查询完成 | 耗时: {processing_time:.0f}ms | "
            f"置信度: {confidence:.2f}"
        )

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "processing_time_ms": processing_time,
        }

    def _preprocess_query(
        self,
        question: str,
        chat_history: List[Dict[str, str]] = None,
    ) -> str:
        """
        查询预处理：融合对话历史，生成独立的搜索查询
        """
        if not chat_history:
            return question

        # 构建历史文本
        history_text = ""
        for msg in chat_history[-settings.MAX_HISTORY_TURNS:]:
            role = "用户" if msg["role"] == "user" else "助手"
            history_text += f"{role}: {msg['content']}\n"

        # 使用 LLM 改写查询
        chain = self.rewrite_prompt | self.rewrite_llm | StrOutputParser()
        rewritten = chain.invoke({
            "history": history_text,
            "question": question,
        })

        return rewritten.strip()

    def _build_context(
        self, results: List[Dict[str, Any]]
    ) -> str:
        """构建 LLM 上下文"""
        context_parts = []

        for i, result in enumerate(results, 1):
            metadata = result["metadata"]
            filename = metadata.get("filename", "未知来源")
            page = metadata.get("page", "")
            score = result["score"]

            header = f"--- 参考文档 {i} [来源: {filename}"
            if page:
                header += f", 第{page}页"
            header += f", 相关性: {score:.2f}] ---"

            context_parts.append(f"{header}\n{result['content']}")

        return "\n\n".join(context_parts)

    def _extract_sources(
        self, results: List[Dict[str, Any]]
    ) -> List[SourceDocument]:
        """提取来源文档信息"""
        sources = []

        for result in results:
            metadata = result["metadata"]
            sources.append(SourceDocument(
                filename=metadata.get("filename", "unknown"),
                page_number=metadata.get("page", None),
                chunk_index=metadata.get("chunk_index", 0),
                content_preview=result["content"][:150],
                relevance_score=round(result["score"], 4),
            ))

        return sources
```

---

## 7. 对话历史管理

### 7.1 基于 SQLite 的对话历史存储

```python
"""
app/core/chat_history.py - 对话历史管理
使用 SQLite 持久化存储对话记录
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from contextlib import contextmanager
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class ChatHistoryManager:
    """
    对话历史管理器
    支持多会话、持久化存储、历史查询
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.CHAT_DB_PATH
        self._init_db()
        logger.info(f"对话历史管理器初始化完成: {self.db_path}")

    def _init_db(self):
        """初始化数据库表"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(id)
                        ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conv
                ON messages(conversation_id)
            """)
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def create_conversation(self, title: str = None) -> str:
        """创建新对话会话"""
        conv_id = str(uuid.uuid4())
        title = title or f"对话 {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title) VALUES (?, ?)",
                (conv_id, title)
            )
            conn.commit()

        logger.info(f"创建新对话: {conv_id}")
        return conv_id

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: List[Dict] = None,
    ):
        """添加消息到对话"""
        sources_json = json.dumps(sources, ensure_ascii=False) if sources else None

        with self._get_connection() as conn:
            # 确保对话存在
            existing = conn.execute(
                "SELECT id FROM conversations WHERE id = ?",
                (conversation_id,)
            ).fetchone()

            if not existing:
                conn.execute(
                    "INSERT INTO conversations (id, title) VALUES (?, ?)",
                    (conversation_id, f"对话 {datetime.now().strftime('%m-%d %H:%M')}")
                )

            conn.execute(
                """INSERT INTO messages
                   (conversation_id, role, content, sources)
                   VALUES (?, ?, ?, ?)""",
                (conversation_id, role, content, sources_json)
            )
            conn.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (conversation_id,)
            )
            conn.commit()

    def get_history(
        self,
        conversation_id: str,
        max_turns: int = None,
    ) -> List[Dict[str, str]]:
        """
        获取对话历史

        Args:
            conversation_id: 对话ID
            max_turns: 最大返回轮数

        Returns:
            [{"role": "user", "content": "..."}, ...]
        """
        max_turns = max_turns or settings.MAX_HISTORY_TURNS

        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT role, content, sources, created_at
                   FROM messages
                   WHERE conversation_id = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (conversation_id, max_turns * 2)
            ).fetchall()

        # 反转为时间正序
        messages = []
        for row in reversed(rows):
            msg = {
                "role": row["role"],
                "content": row["content"],
            }
            if row["sources"]:
                msg["sources"] = json.loads(row["sources"])
            messages.append(msg)

        return messages

    def get_conversations(self, limit: int = 20) -> List[Dict]:
        """获取对话列表"""
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT c.id, c.title, c.created_at, c.updated_at,
                          COUNT(m.id) as message_count
                   FROM conversations c
                   LEFT JOIN messages m ON c.id = m.conversation_id
                   GROUP BY c.id
                   ORDER BY c.updated_at DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()

        return [dict(row) for row in rows]

    def delete_conversation(self, conversation_id: str):
        """删除对话"""
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,)
            )
            conn.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            conn.commit()

        logger.info(f"已删除对话: {conversation_id}")
```

---

## 8. 引用溯源与来源标注

### 8.1 引用溯源模块

```python
"""
app/core/citation.py - 引用溯源模块
解析回答中的引用标注，关联到来源文档
"""

import re
from typing import List, Dict, Any
from app.models import SourceDocument
import logging

logger = logging.getLogger(__name__)


class CitationParser:
    """
    引用溯源解析器

    功能:
    - 从 LLM 回答中解析引用标注 [来源: xxx]
    - 将引用关联到具体的来源文档片段
    - 生成引用摘要
    """

    # 匹配引用模式: [来源: 文件名] 或 [来源: 文件名, 第X页]
    CITATION_PATTERN = re.compile(
        r'\[来源:\s*([^\]]+)\]'
    )

    @staticmethod
    def parse_citations(
        answer: str,
        sources: List[SourceDocument],
    ) -> Dict[str, Any]:
        """
        解析回答中的引用并关联来源

        Args:
            answer: LLM 生成的回答
            sources: 检索到的来源文档列表

        Returns:
            {
                "answer": str,
                "cited_sources": [SourceDocument],
                "uncited_sources": [SourceDocument],
                "citation_count": int,
            }
        """
        # 提取引用中提到的文件名
        cited_filenames = set()
        matches = CitationParser.CITATION_PATTERN.findall(answer)

        for match in matches:
            # 处理可能包含页码的情况: "文件名, 第X页"
            filename = match.split(",")[0].strip()
            cited_filenames.add(filename)

        # 分类来源
        cited_sources = []
        uncited_sources = []

        for source in sources:
            if source.filename in cited_filenames:
                cited_sources.append(source)
            else:
                uncited_sources.append(source)

        return {
            "answer": answer,
            "cited_sources": cited_sources,
            "uncited_sources": uncited_sources,
            "citation_count": len(cited_filenames),
        }

    @staticmethod
    def format_sources_for_display(
        sources: List[SourceDocument],
    ) -> str:
        """格式化来源信息供前端展示"""
        if not sources:
            return "无引用来源"

        lines = ["**引用来源:**"]
        for i, src in enumerate(sources, 1):
            line = f"{i}. **{src.filename}**"
            if src.page_number:
                line += f" (第{src.page_number}页)"
            line += f" | 相关性: {src.relevance_score:.0%}"
            line += f"\n   > {src.content_preview}..."
            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def calculate_answer_confidence(
        sources: List[SourceDocument],
        citation_count: int,
    ) -> float:
        """
        计算回答置信度

        综合考虑:
        - 来源数量
        - 来源相关性分数
        - 引用数量
        """
        if not sources:
            return 0.0

        # 平均相关性分数
        avg_score = sum(s.relevance_score for s in sources) / len(sources)

        # 引用覆盖率
        citation_ratio = min(citation_count / max(len(sources), 1), 1.0)

        # 来源数量因子 (1-4个来源最佳)
        source_factor = min(len(sources) / 3, 1.0)

        confidence = (
            avg_score * 0.5 +
            citation_ratio * 0.3 +
            source_factor * 0.2
        )

        return min(max(confidence, 0.0), 1.0)
```

---

## 9. FastAPI 后端完整实现

### 9.1 主入口

```python
"""
app/main.py - FastAPI 主入口
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api.documents import router as documents_router
from app.api.chat import router as chat_router
from app.core.vector_store import create_vector_store
from app.core.rag_engine import RAGEngine
from app.core.chat_history import ChatHistoryManager
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="基于 RAG 的企业知识库问答系统 API",
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局服务实例
vector_store = None
rag_engine = None
chat_manager = None


@app.on_event("startup")
async def startup():
    """应用启动时初始化服务"""
    global vector_store, rag_engine, chat_manager

    logger.info("正在初始化服务...")
    vector_store = create_vector_store()
    rag_engine = RAGEngine(vector_store)
    chat_manager = ChatHistoryManager()
    logger.info("所有服务初始化完成")


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "vector_db_type": settings.VECTOR_DB_TYPE,
        "document_count": vector_store.get_document_count() if vector_store else 0,
    }


# 注册路由
app.include_router(documents_router, prefix="/api/documents", tags=["文档管理"])
app.include_router(chat_router, prefix="/api/chat", tags=["对话问答"])
```

### 9.2 文档管理 API

```python
"""
app/api/documents.py - 文档管理 API
"""

import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from app.config import settings
from app.core.document_processor import DocumentProcessor
from app.models import DocumentInfo, DocumentListResponse, DocumentStatus
import logging

logger = logging.getLogger(__name__)
router = APIRouter()
processor = DocumentProcessor()


@router.post("/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档并自动处理

    支持格式: PDF, Word(.docx), Markdown, TXT
    文件大小限制: 50MB
    """
    # 验证文件类型
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in processor.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，"
                   f"支持: {list(processor.SUPPORTED_EXTENSIONS.keys())}"
        )

    # 验证文件大小
    content = await file.read()
    file_size = len(content)
    if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小超过限制: {settings.MAX_FILE_SIZE_MB}MB"
        )

    # 保存文件
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        # 处理文档
        from app.main import vector_store
        chunks, stats = processor.process_file(file_path)

        # 存入向量数据库
        added_count = vector_store.add_documents(chunks)

        doc_info = DocumentInfo(
            id=stats.get("file_id", chunks[0].metadata.get("file_id", "")),
            filename=file.filename,
            file_type=ext,
            file_size=file_size,
            chunk_count=added_count,
            status=DocumentStatus.COMPLETED,
        )

        logger.info(
            f"文档上传成功: {file.filename} | "
            f"{added_count} 个分块已索引"
        )
        return doc_info

    except Exception as e:
        logger.error(f"文档处理失败: {e}")
        # 清理已上传的文件
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """获取已上传的文档列表"""
    from app.main import vector_store

    files = vector_store.get_file_list()

    documents = []
    for f in files:
        documents.append(DocumentInfo(
            id=f["file_id"],
            filename=f["filename"],
            file_type=f.get("file_type", ""),
            file_size=0,
            chunk_count=f["chunk_count"],
            status=DocumentStatus.COMPLETED,
        ))

    return DocumentListResponse(
        total=len(documents),
        documents=documents,
    )


@router.delete("/{file_id}")
async def delete_document(file_id: str):
    """删除指定文档及其所有分块"""
    from app.main import vector_store

    deleted_count = vector_store.delete_by_file_id(file_id)

    if deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"未找到文档: {file_id}"
        )

    return {
        "message": f"已删除文档 {file_id}",
        "deleted_chunks": deleted_count,
    }
```

### 9.3 对话问答 API

```python
"""
app/api/chat.py - 对话问答 API
"""

import uuid
from fastapi import APIRouter, HTTPException
from app.models import ChatRequest, ChatResponse
from app.core.citation import CitationParser
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    对话问答接口

    支持:
    - 单轮问答
    - 多轮对话（传入 conversation_id）
    - 引用来源标注
    """
    from app.main import rag_engine, chat_manager

    # 获取或创建对话ID
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # 获取对话历史
    chat_history = None
    if request.use_history and request.conversation_id:
        chat_history = chat_manager.get_history(conversation_id)

    try:
        # 执行 RAG 查询
        result = rag_engine.query(
            question=request.question,
            chat_history=chat_history,
        )

        # 解析引用
        citation_result = CitationParser.parse_citations(
            answer=result["answer"],
            sources=result["sources"],
        )

        # 计算置信度
        confidence = CitationParser.calculate_answer_confidence(
            sources=result["sources"],
            citation_count=citation_result["citation_count"],
        )

        # 保存对话历史
        chat_manager.add_message(
            conversation_id=conversation_id,
            role="user",
            content=request.question,
        )
        chat_manager.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=result["answer"],
            sources=[s.model_dump() for s in result["sources"]],
        )

        return ChatResponse(
            answer=result["answer"],
            conversation_id=conversation_id,
            sources=result["sources"],
            confidence=confidence,
            processing_time_ms=result["processing_time_ms"],
        )

    except Exception as e:
        logger.error(f"对话查询失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"查询处理失败: {str(e)}"
        )


@router.get("/history/{conversation_id}")
async def get_chat_history(conversation_id: str):
    """获取对话历史"""
    from app.main import chat_manager

    history = chat_manager.get_history(conversation_id, max_turns=50)

    if not history:
        raise HTTPException(
            status_code=404,
            detail=f"未找到对话: {conversation_id}"
        )

    return {
        "conversation_id": conversation_id,
        "messages": history,
    }


@router.get("/conversations")
async def list_conversations():
    """获取对话列表"""
    from app.main import chat_manager

    conversations = chat_manager.get_conversations(limit=20)
    return {"conversations": conversations}
```

---

## 10. Streamlit 前端界面

### 10.1 前端架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Streamlit 前端界面布局                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────── 侧边栏 ──────────────────────────────┐    │
│  │                                                              │    │
│  │  [企业知识库问答系统]                                        │    │
│  │                                                              │    │
│  │  导航:                                                       │    │
│  │  * 对话问答                                                  │    │
│  │  * 文档管理                                                  │    │
│  │                                                              │    │
│  │  ─────────────                                               │    │
│  │  对话历史:                                                    │    │
│  │  > 对话1 (2024-01-15)                                       │    │
│  │  > 对话2 (2024-01-14)                                       │    │
│  │  [新建对话]                                                  │    │
│  │                                                              │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌────────────────────── 主内容区 ─────────────────────────────┐    │
│  │                                                              │    │
│  │  对话问答页:                                                  │    │
│  │  ┌──────────────────────────────────────────────┐           │    │
│  │  │ [用户] 公司的年假政策是什么？                   │           │    │
│  │  │                                                │           │    │
│  │  │ [助手] 根据公司员工手册，年假政策如下：          │           │    │
│  │  │ 1. 入职满1年：5天年假                          │           │    │
│  │  │ 2. 入职满3年：10天年假                         │           │    │
│  │  │ [来源: 员工手册.pdf]                           │           │    │
│  │  │                                                │           │    │
│  │  │ > 引用来源 (展开/收起)                          │           │    │
│  │  └──────────────────────────────────────────────┘           │    │
│  │                                                              │    │
│  │  ┌──────────────────────────────────────┐ [发送]           │    │
│  │  │ 请输入您的问题...                     │                  │    │
│  │  └────────────────���─────────────────────┘                  │    │
│  │                                                              │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 10.2 完整前端代码

```python
"""
frontend/streamlit_app.py - Streamlit 前端界面
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import Optional

# ============================================================
# 配置
# ============================================================

API_BASE_URL = "http://localhost:8000/api"

st.set_page_config(
    page_title="企业知识库问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# API 调用函数
# ============================================================

def api_chat(question: str, conversation_id: Optional[str] = None) -> dict:
    """调用对话 API"""
    payload = {
        "question": question,
        "conversation_id": conversation_id,
        "use_history": True,
    }
    try:
        resp = requests.post(f"{API_BASE_URL}/chat/", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"error": str(e)}


def api_upload_document(file) -> dict:
    """调用文档上传 API"""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        resp = requests.post(
            f"{API_BASE_URL}/documents/upload",
            files=files,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"error": str(e)}


def api_list_documents() -> dict:
    """获取文档列表"""
    try:
        resp = requests.get(f"{API_BASE_URL}/documents/", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"error": str(e), "total": 0, "documents": []}


def api_delete_document(file_id: str) -> dict:
    """删除文档"""
    try:
        resp = requests.delete(
            f"{API_BASE_URL}/documents/{file_id}", timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"error": str(e)}


def api_health() -> dict:
    """健康检查"""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return {"status": "unavailable"}


# ============================================================
# Session State 初始化
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None


# ============================================================
# 侧边栏
# ============================================================

with st.sidebar:
    st.title("企业知识库问答系统")

    # 系统状态
    health = api_health()
    if health.get("status") == "healthy":
        st.success(
            f"系统运行中 | 文档数: {health.get('document_count', 0)}"
        )
    else:
        st.error("后端服务不可用，请检查是否已启动")

    st.divider()

    # 导航
    page = st.radio("导航", ["对话问答", "文档管理"], label_visibility="collapsed")

    st.divider()

    # 新建对话按钮
    if page == "对话问答":
        if st.button("新建对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()


# ============================================================
# 对话问答页面
# ============================================================

if page == "对话问答":
    st.header("智能问答")

    # 显示对话历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # 显示来源引用
            if message.get("sources"):
                with st.expander("查看引用来源", expanded=False):
                    for i, src in enumerate(message["sources"], 1):
                        st.markdown(
                            f"**{i}. {src['filename']}** "
                            f"(相关性: {src['relevance_score']:.0%})"
                        )
                        st.caption(f"> {src['content_preview']}")

    # 用户输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 显示用户消息
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用 API 获取回答
        with st.chat_message("assistant"):
            with st.spinner("正在检索知识库并生成回答..."):
                result = api_chat(
                    question=prompt,
                    conversation_id=st.session_state.conversation_id,
                )

            if "error" in result:
                st.error(f"查询失败: {result['error']}")
            else:
                # 显示回答
                st.markdown(result["answer"])

                # 更新对话ID
                st.session_state.conversation_id = result.get(
                    "conversation_id"
                )

                # 显示元信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(
                        f"置信度: {result.get('confidence', 0):.0%}"
                    )
                with col2:
                    st.caption(
                        f"耗时: {result.get('processing_time_ms', 0):.0f}ms"
                    )
                with col3:
                    st.caption(
                        f"引用: {len(result.get('sources', []))} 个来源"
                    )

                # 显示来源
                sources = result.get("sources", [])
                if sources:
                    with st.expander("查看引用来源", expanded=False):
                        for i, src in enumerate(sources, 1):
                            st.markdown(
                                f"**{i}. {src['filename']}** "
                                f"(相关性: {src['relevance_score']:.0%})"
                            )
                            st.caption(f"> {src['content_preview']}")

                # 保存到 session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": sources,
                })


# ============================================================
# 文档管理页面
# ============================================================

elif page == "文档管理":
    st.header("文档管理")

    # 文档上传
    st.subheader("上传文档")

    uploaded_file = st.file_uploader(
        "选择文档文件",
        type=["pdf", "docx", "md", "txt"],
        help="支持 PDF、Word、Markdown、TXT 格式，最大 50MB",
    )

    if uploaded_file is not None:
        if st.button("开始上传并处理", type="primary"):
            with st.spinner(f"正在处理文档: {uploaded_file.name}..."):
                result = api_upload_document(uploaded_file)

            if "error" in result:
                st.error(f"上传失败: {result['error']}")
            else:
                st.success(
                    f"文档处理成功: {result['filename']} | "
                    f"生成 {result['chunk_count']} 个知识分块"
                )
                st.rerun()

    st.divider()

    # 文档列表
    st.subheader("已上传文档")

    doc_list = api_list_documents()

    if doc_list.get("total", 0) == 0:
        st.info("暂无已上传的文档，请先上传文档到知识库")
    else:
        st.write(f"共 {doc_list['total']} 个文档")

        for doc in doc_list.get("documents", []):
            col1, col2, col3 = st.columns([4, 2, 1])
            with col1:
                st.write(f"**{doc['filename']}**")
            with col2:
                st.caption(f"{doc['chunk_count']} 个分块")
            with col3:
                if st.button("删除", key=f"del_{doc['id']}"):
                    result = api_delete_document(doc["id"])
                    if "error" not in result:
                        st.success(f"已删除: {doc['filename']}")
                        st.rerun()
                    else:
                        st.error(f"删除失败: {result['error']}")
```

---

## 11. Docker 部署方案

### 11.1 部署架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Docker Compose 部署架构                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  docker-compose.yml                                                  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  network: rag-network                                       │     │
│  │                                                             │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │     │
│  │  │  rag-backend  │  │ rag-frontend │  │ milvus       │    │     │
│  │  │  (FastAPI)    │  │ (Streamlit)  │  │ (可选)       │    │     │
│  │  │  Port: 8000   │  │ Port: 8501   │  │ Port: 19530  │    │     │
│  │  │               │  │              │  │              │    │     │
│  │  │  volumes:     │  │              │  │  volumes:    │    │     │
│  │  │  - ./data     │  │              │  │  - milvus_db │    │     │
│  │  └──────────────┘  └──────────────┘  └──────────────┘    │     │
│  │                                                             │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  volumes:                                                            │
│  - rag_data: 文档存储 + ChromaDB + SQLite                           │
│  - milvus_data: Milvus数据 (可选)                                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 11.2 Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app/ ./app/
COPY frontend/ ./frontend/

# 创建数据目录
RUN mkdir -p /app/data/uploads /app/data/chroma_db

# 暴露端口
EXPOSE 8000 8501

# 默认启动后端
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 11.3 Docker Compose 配置

```yaml
# docker-compose.yml
version: "3.8"

services:
  # FastAPI 后端
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rag-backend
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - VECTOR_DB_TYPE=chroma
      - CHROMA_PERSIST_DIR=/app/data/chroma_db
      - CHAT_DB_PATH=/app/data/chat_history.db
      - UPLOAD_DIR=/app/data/uploads
    volumes:
      - rag_data:/app/data
    command: >
      uvicorn app.main:app
      --host 0.0.0.0
      --port 8000
      --workers 2
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Streamlit 前端
  frontend:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rag-frontend
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://backend:8000/api
    command: >
      streamlit run frontend/streamlit_app.py
      --server.port 8501
      --server.address 0.0.0.0
      --server.headless true
    depends_on:
      backend:
        condition: service_healthy
    restart: unless-stopped

  # Milvus（可选，生产环境使用）
  # 取消注释以下部分启用 Milvus
  #
  # etcd:
  #   container_name: milvus-etcd
  #   image: quay.io/coreos/etcd:v3.5.5
  #   environment:
  #     - ETCD_AUTO_COMPACTION_MODE=revision
  #     - ETCD_AUTO_COMPACTION_RETENTION=1000
  #     - ETCD_QUOTA_BACKEND_BYTES=4294967296
  #   volumes:
  #     - etcd_data:/etcd
  #   command: >
  #     etcd
  #     -advertise-client-urls=http://127.0.0.1:2379
  #     -listen-client-urls http://0.0.0.0:2379
  #     --data-dir /etcd
  #
  # minio:
  #   container_name: milvus-minio
  #   image: minio/minio:RELEASE.2023-03-20T20-16-18Z
  #   environment:
  #     MINIO_ACCESS_KEY: minioadmin
  #     MINIO_SECRET_KEY: minioadmin
  #   volumes:
  #     - minio_data:/minio_data
  #   command: minio server /minio_data --console-address ":9001"
  #
  # milvus:
  #   container_name: milvus-standalone
  #   image: milvusdb/milvus:v2.4.0
  #   command: ["milvus", "run", "standalone"]
  #   environment:
  #     ETCD_ENDPOINTS: etcd:2379
  #     MINIO_ADDRESS: minio:9000
  #   ports:
  #     - "19530:19530"
  #   depends_on:
  #     - etcd
  #     - minio
  #   volumes:
  #     - milvus_data:/var/lib/milvus

volumes:
  rag_data:
  # etcd_data:
  # minio_data:
  # milvus_data:
```

### 11.4 环境变量配置

```bash
# .env.example
# 复制为 .env 并填写实际值

# OpenAI API 配置
OPENAI_API_KEY=sk-your-api-key-here

# 向量数据库配置 (chroma 或 milvus)
VECTOR_DB_TYPE=chroma

# 应用配置
DEBUG=false
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### 11.5 启动命令

```bash
# 1. 准备环境变量
cp .env.example .env
# 编辑 .env 填写 OPENAI_API_KEY

# 2. 构建并启动
docker-compose up -d --build

# 3. 查看日志
docker-compose logs -f backend
docker-compose logs -f frontend

# 4. 访问服务
# 前端: http://localhost:8501
# API文档: http://localhost:8000/docs
# 健康检查: http://localhost:8000/api/health

# 5. 停止服务
docker-compose down

# 6. 停止并清除数据
docker-compose down -v
```

---

## 12. 测试与评估

### 12.1 单元测试

```python
"""
tests/test_document_processor.py - 文档处理器测试
"""

import os
import tempfile
import pytest
from app.core.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """文档处理器测试"""

    def setup_method(self):
        self.processor = DocumentProcessor(
            chunk_size=200,
            chunk_overlap=50,
        )

    def test_supported_extensions(self):
        """测试支持的文件格式"""
        exts = DocumentProcessor.get_supported_extensions()
        assert ".pdf" in exts
        assert ".docx" in exts
        assert ".md" in exts
        assert ".txt" in exts

    def test_process_txt_file(self):
        """测试 TXT 文件处理"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("这是第一段测试内容。" * 50 + "\n\n")
            f.write("这是第二段测试内容。" * 50)
            temp_path = f.name

        try:
            chunks, stats = self.processor.process_file(temp_path)

            assert len(chunks) > 0
            assert stats["chunks"] > 0
            assert stats["raw_pages"] >= 1

            # 验证元数据
            for chunk in chunks:
                assert "filename" in chunk.metadata
                assert "chunk_index" in chunk.metadata
                assert "file_type" in chunk.metadata
                assert chunk.metadata["file_type"] == ".txt"

        finally:
            os.unlink(temp_path)

    def test_unsupported_format(self):
        """测试不支持的文件格式"""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="不支持的文件格式"):
                self.processor.process_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_chunk_size_compliance(self):
        """测试分块大小是否符合配置"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("测试文本内容。" * 500)
            temp_path = f.name

        try:
            chunks, _ = self.processor.process_file(temp_path)

            for chunk in chunks:
                # 允许少量超出（分隔符导致）
                assert len(chunk.page_content) <= self.processor.chunk_size * 1.2

        finally:
            os.unlink(temp_path)
```

### 12.2 API 集成测试

```python
"""
tests/test_api.py - API 集成测试
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
import tempfile
import os


client = TestClient(app)


class TestHealthAPI:
    """健康检查 API 测试"""

    def test_health_check(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "document_count" in data


class TestDocumentAPI:
    """文档管理 API 测试"""

    def test_upload_txt_document(self):
        """测试上传 TXT 文档"""
        content = "这是测试文档的内容。" * 100
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                resp = client.post(
                    "/api/documents/upload",
                    files={"file": ("test.txt", f, "text/plain")},
                )

            assert resp.status_code == 200
            data = resp.json()
            assert data["filename"] == "test.txt"
            assert data["chunk_count"] > 0
            assert data["status"] == "completed"

        finally:
            os.unlink(temp_path)

    def test_upload_unsupported_format(self):
        """测试上传不支持的格式"""
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("test.xyz", b"content", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_list_documents(self):
        """测试获取文档列表"""
        resp = client.get("/api/documents/")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "documents" in data


class TestChatAPI:
    """对话问答 API 测试"""

    def test_chat_basic(self):
        """测试基本问答"""
        resp = client.post(
            "/api/chat/",
            json={
                "question": "这是一个测试问题",
                "use_history": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "conversation_id" in data
        assert "sources" in data
        assert "confidence" in data

    def test_chat_with_history(self):
        """测试多轮对话"""
        # 第一轮
        resp1 = client.post(
            "/api/chat/",
            json={"question": "第一个问题"},
        )
        data1 = resp1.json()
        conv_id = data1["conversation_id"]

        # 第二轮（带上对话ID）
        resp2 = client.post(
            "/api/chat/",
            json={
                "question": "追问上一个问题",
                "conversation_id": conv_id,
                "use_history": True,
            },
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["conversation_id"] == conv_id
```

### 12.3 RAG 效果评估

```python
"""
tests/test_rag_engine.py - RAG 引擎效果评估
"""

from typing import List, Dict
from app.core.rag_engine import RAGEngine
from app.core.vector_store import ChromaVectorStore
from langchain.schema import Document
import json


class RAGEvaluator:
    """RAG 系统效果评估器"""

    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine
        self.results = []

    def evaluate(
        self,
        test_cases: List[Dict[str, str]],
    ) -> Dict[str, float]:
        """
        评估 RAG 系统

        Args:
            test_cases: 测试用例列表
                [{"question": "...", "expected_keywords": ["..."], "expected_source": "..."}]

        Returns:
            评估指标字典
        """
        hit_count = 0
        keyword_match_count = 0
        total_confidence = 0.0

        for case in test_cases:
            result = self.rag_engine.query(question=case["question"])

            # 检查来源是否命中
            source_hit = any(
                case.get("expected_source", "") in s.filename
                for s in result["sources"]
            )
            if source_hit:
                hit_count += 1

            # 检查关键词是否出现在回答中
            keywords = case.get("expected_keywords", [])
            if keywords:
                matched = sum(
                    1 for kw in keywords
                    if kw in result["answer"]
                )
                if matched >= len(keywords) * 0.5:
                    keyword_match_count += 1

            total_confidence += result["confidence"]

            self.results.append({
                "question": case["question"],
                "answer_preview": result["answer"][:100],
                "source_hit": source_hit,
                "confidence": result["confidence"],
                "num_sources": len(result["sources"]),
            })

        n = len(test_cases)
        return {
            "source_hit_rate": hit_count / n if n > 0 else 0,
            "keyword_accuracy": keyword_match_count / n if n > 0 else 0,
            "avg_confidence": total_confidence / n if n > 0 else 0,
            "total_cases": n,
        }

    def print_report(self, metrics: Dict[str, float]):
        """打印评估报告"""
        print("\n" + "=" * 60)
        print("RAG 系统评估报告")
        print("=" * 60)
        print(f"测试用例数:     {metrics['total_cases']}")
        print(f"来源命中率:     {metrics['source_hit_rate']:.2%}")
        print(f"关键词准确率:   {metrics['keyword_accuracy']:.2%}")
        print(f"平均置信度:     {metrics['avg_confidence']:.2%}")
        print("=" * 60)


# 使用示例
if __name__ == "__main__":
    # 准备测试数据
    test_cases = [
        {
            "question": "公司的年假政策是什么？",
            "expected_keywords": ["年假", "天"],
            "expected_source": "员工手册",
        },
        {
            "question": "如何申请报销？",
            "expected_keywords": ["报销", "审批"],
            "expected_source": "财务制度",
        },
        {
            "question": "新员工入职需要准备什么？",
            "expected_keywords": ["入职", "合同"],
            "expected_source": "入职指南",
        },
    ]

    # 初始化
    vector_store = ChromaVectorStore()
    rag_engine = RAGEngine(vector_store)
    evaluator = RAGEvaluator(rag_engine)

    # 运行评估
    metrics = evaluator.evaluate(test_cases)
    evaluator.print_report(metrics)
```

---

## 13. 生产环境优化

### 13.1 性能优化清单

```
┌──────────────────────────────────────────────────────────────────────┐
│                    生产环境优化清单                                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  检索优化:                                                            │
│  [x] 使用 HNSW 索引替代 FLAT (Milvus 场景)                          │
│  [x] 开启混合检索 (Dense + BM25)                                    │
│  [x] 添加 Reranker 重排序                                           │
│  [x] 调优 Top-K 参数 (推荐 6-10)                                    │
│  [x] 使用 MMR 保证结果多样性                                        │
│                                                                      │
│  生成优化:                                                            │
│  [x] 使用流式输出 (Streaming) 降低首字延迟                           │
│  [x] 设置合理的 temperature (0.1-0.3)                                │
│  [x] 限制 max_tokens 避免过长回答                                    │
│  [x] 优化 Prompt 模板减少 token 消耗                                 │
│                                                                      │
│  基础设施:                                                            │
│  [x] 生产环境使用 Milvus 替代 ChromaDB                              │
│  [x] 使用 Redis 缓存热门查询结果                                    │
│  [x] 配置 uvicorn workers (CPU核数 * 2 + 1)                        │
│  [x] 开启 API 速率限制                                              │
│  [x] 添加请求日志和监控                                              │
│                                                                      │
│  安全:                                                                │
│  [x] API Key 认证                                                    │
│  [x] 文件上传大小限制和类型校验                                      │
│  [x] 输入内容安全过滤                                                │
│  [x] HTTPS 加密传输                                                  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 13.2 流式输出实现

```python
"""
流式输出优化 - 降低用户感知延迟
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.config import settings
import json
import asyncio


streaming_router = APIRouter()


async def generate_streaming_response(question: str, context: str):
    """
    流式生成回答

    使用 Server-Sent Events (SSE) 格式
    """
    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        streaming=True,
        openai_api_key=settings.OPENAI_API_KEY,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是企业知识库问答助手。根据参考文档回答问题。
在回答中引用来源 [来源: 文件名]。

参考文档:
{context}"""),
        ("human", "{question}"),
    ])

    chain = prompt | llm

    async for chunk in chain.astream({
        "context": context,
        "question": question,
    }):
        if hasattr(chunk, "content") and chunk.content:
            # SSE 格式
            data = json.dumps(
                {"type": "token", "content": chunk.content},
                ensure_ascii=False,
            )
            yield f"data: {data}\n\n"

    # 发送结束信号
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@streaming_router.post("/stream")
async def chat_stream(request: dict):
    """流式对话接口"""
    from app.main import rag_engine

    question = request.get("question", "")

    # 检索上下文
    results = rag_engine.vector_store.search(query=question, k=settings.SEARCH_K)
    context = rag_engine._build_context(results)

    return StreamingResponse(
        generate_streaming_response(question, context),
        media_type="text/event-stream",
    )
```

### 13.3 缓存策略

```python
"""
查询结果缓存 - 减少重复查询的 API 调用
"""

import hashlib
import json
import time
from typing import Optional, Dict, Any
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class QueryCache:
    """
    查询结果缓存
    基于内存的简单 LRU 缓存，适合单实例部署
    生产环境建议替换为 Redis
    """

    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        """
        Args:
            max_size: 最大缓存条目数
            ttl_seconds: 缓存过期时间（秒）
        """
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _make_key(self, question: str) -> str:
        """生成缓存 key"""
        normalized = question.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, question: str) -> Optional[Dict[str, Any]]:
        """获取缓存结果"""
        key = self._make_key(question)
        entry = self._cache.get(key)

        if entry is None:
            return None

        # 检查是否过期
        if time.time() - entry["timestamp"] > self.ttl:
            del self._cache[key]
            return None

        logger.info(f"缓存命中: {question[:30]}...")
        return entry["data"]

    def set(self, question: str, result: Dict[str, Any]):
        """设置缓存"""
        # 如果缓存已满，删除最旧的条目
        if len(self._cache) >= self.max_size:
            oldest_key = min(
                self._cache,
                key=lambda k: self._cache[k]["timestamp"],
            )
            del self._cache[oldest_key]

        key = self._make_key(question)
        self._cache[key] = {
            "data": result,
            "timestamp": time.time(),
        }

    def clear(self):
        """清空缓存"""
        self._cache.clear()

    def stats(self) -> Dict[str, int]:
        """缓存统计"""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
        }
```

---

## 14. 总结与最佳实践

### 14.1 项目回顾

本教程完整实现了一个基于 RAG 的企业知识库问答系统，涵盖以下核心模块：

1. **文档处理引擎**：支持 PDF/Word/Markdown/TXT 多格式文档的自动解析、清洗和智能分块
2. **向量存储服务**：支持 ChromaDB（开发）和 Milvus（生产）两种后端，通过抽象层实现无缝切换
3. **RAG 检索与生成**：查询预处理、向量检索、结果过滤、LLM 生成的完整 Pipeline
4. **对话历史管理**：基于 SQLite 的多会话持久化存储
5. **引用溯源**：自动解析引用标注，关联来源文档，计算置信度
6. **FastAPI 后端**：完整的 RESTful API，包含文档管理和对话问答接口
7. **Streamlit 前端**：直观的 Web 界面，支持文档上传和对话问答
8. **Docker 部署**：一键部署方案，支持生产环境扩展

### 14.2 最佳实践总结

```
┌──────────────────────────────────────────────────────────────────────┐
│                    RAG 实战项目最佳实践                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  架构设计:                                                            │
│  1. 使用抽象层隔离向量数据库，便于后续切换                            │
│  2. 配置管理集中化，使用环境变量控制不同环境                          │
│  3. 模块化设计，各组件职责清晰，便于测试和维护                        │
│                                                                      │
│  文档处理:                                                            │
│  1. 中文文档使用 800 字符分块，150 字符重叠                          │
│  2. 使用中文优化的分隔符列表                                         │
│  3. 保留丰富的元数据（文件名、页码、分块序号）                        │
│  4. 清洗阶段去除页眉页脚等噪声                                      │
│                                                                      │
│  检索优化:                                                            │
│  1. 多轮对话场景下进行查询改写                                       │
│  2. 设置相关性阈值过滤低质量结果                                     │
│  3. 检索数量建议 6-10，过多会引入噪声                                │
│  4. 生产环境添加 Reranker 可显著提升精度                             │
│                                                                      │
│  生成质量:                                                            │
│  1. Prompt 明确要求"仅基于参考文档回答"                              │
│  2. 要求标注引用来源，便于用户验证                                   │
│  3. temperature 设为 0.1，减少生成随机性                             │
│  4. 信息不足时明确告知用户                                           │
│                                                                      │
│  部署运维:                                                            │
│  1. 使用 Docker Compose 统一管理服务                                 │
│  2. 开发环境用 ChromaDB，生产环境用 Milvus                           │
│  3. 添加健康检查和日志监控                                           │
│  4. 定期评估 RAG 效果，持续迭代优化                                  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 14.3 快速启动指南

```bash
# 1. 克隆项目
git clone https://github.com/your-org/enterprise-rag-system.git
cd enterprise-rag-system

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填写 OPENAI_API_KEY

# 3. Docker 一键启动
docker-compose up -d --build

# 4. 访问服务
# 前端界面: http://localhost:8501
# API 文档: http://localhost:8000/docs

# 5. 上传文档并开始提问
# 通过前端界面上传企业文档，然后在对话页面提问
```

## 参考资源

- [LangChain 官方文档](https://python.langchain.com/)
- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [Streamlit 官方文档](https://docs.streamlit.io/)
- [ChromaDB 文档](https://docs.trychroma.com/)
- [Milvus 文档](https://milvus.io/docs)
- [Docker Compose 文档](https://docs.docker.com/compose/)
- [论文: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

---

**创建时间**: 2024-01
**最后更新**: 2024-01
