# 智能客服系统 - RAG + Agent 完整实战

## 目录
1. [项目概述](#项目概述)
2. [系统架构](#系统架构)
3. [RAG知识库实现](#rag知识库实现)
4. [多轮对话管理](#多轮对话管理)
5. [Agent工具调用](#agent工具调用)
6. [FastAPI后端服务](#fastapi后端服务)
7. [前端对话界面](#前端对话界面)
8. [部署与监控](#部署与监控)

---

## 项目概述

### 项目背景

智能客服系统是AI落地最广泛的场景之一。传统客服系统基于关键词匹配和规则引擎，
无法理解用户的真实意图。结合大语言模型(LLM)、检索增强生成(RAG)和智能体(Agent)
技术，可以构建一个真正理解用户问题、能查阅知识库、能调用业务系统的智能客服。

### 系统整体架构

```
+===========================================================================+
|                        智能客服系统 - 总体架构                              |
+===========================================================================+
|                                                                           |
|   +-------------------+     +-----------------+     +-----------------+   |
|   |                   |     |                 |     |                 |   |
|   |    Web 前端       |     |  微信小程序      |     |   API 调用方    |   |
|   |  (React/Vue)      |     |                 |     |  (第三方系统)   |   |
|   +--------+----------+     +--------+--------+     +--------+--------+   |
|            |                         |                       |            |
|            +------------+------------+-----------+-----------+            |
|                         |                        |                        |
|                         v                        v                        |
|            +------------+------------------------+-----------+            |
|            |                  API Gateway                    |            |
|            |            (Nginx / Kong)                       |            |
|            +-------------------------+-----------------------+            |
|                                      |                                    |
|                                      v                                    |
|            +-------------------------+-----------------------+            |
|            |              FastAPI 后端服务                    |            |
|            |  +---------------+  +----------------+          |            |
|            |  | 对话管理器    |  | 会话状态管理    |          |            |
|            |  | (Dialogue     |  | (Session       |          |            |
|            |  |  Manager)     |  |  Manager)      |          |            |
|            |  +-------+-------+  +-------+--------+          |            |
|            |          |                  |                    |            |
|            |          v                  v                    |            |
|            |  +-------+------------------+--------+          |            |
|            |  |         意图路由器                  |          |            |
|            |  |     (Intent Router)                |          |            |
|            |  +--+--------+--------+--------+-----+          |            |
|            |     |        |        |        |                |            |
|            +-----+--------+--------+--------+----------------+            |
|                  |        |        |        |                             |
|         +--------+--+ +---+----+ +-+------+ +---+--------+               |
|         |           | |        | |        | |            |               |
|         | RAG知识库 | | Agent  | | 闲聊   | | 人工转接   |               |
|         | 问答      | | 工具   | | 模块   | | 模块       |               |
|         |           | | 调用   | |        | |            |               |
|         +-----+-----+ +---+---+ +---+----+ +---+--------+               |
|               |            |         |          |                         |
|         +-----+-----+ +---+---+ +---+----+ +---+--------+               |
|         | ChromaDB  | | 订单  | |  LLM   | | 消息队列   |               |
|         | 向量数据库 | | 系统  | | 直接   | | (Redis)    |               |
|         |           | | CRM   | | 生成   | |            |               |
|         +-----------+ +-------+ +--------+ +------------+               |
|                                                                           |
+===========================================================================+
```

### 核心功能清单

| 功能模块 | 说明 | 技术方案 |
|---------|------|---------|
| RAG问答 | 基于企业知识库回答用户问题 | ChromaDB + OpenAI Embedding |
| 多轮对话 | 保持对话上下文，理解指代关系 | Redis会话存储 + 滑动窗口 |
| Agent工具 | 查订单、退款、修改信息等操作 | Function Calling + 工具链 |
| 意图识别 | 自动判断用户意图并路由 | LLM分类 + 规则兜底 |
| 人工转接 | 复杂问题无缝转接人工客服 | WebSocket + Redis队列 |
| 对话评估 | 自动评估回答质量 | LLM打分 + 用户反馈 |

### 技术栈总览

```
+------------------------------------------------------------------+
|                        技术栈                                     |
+------------------------------------------------------------------+
|  前端:  React 18 + TypeScript + Tailwind CSS + WebSocket         |
|  后端:  FastAPI + Pydantic + SQLAlchemy + Celery                 |
|  AI:    OpenAI GPT-4 + text-embedding-3-small                   |
|  向量库: ChromaDB (开发) / Milvus (生产)                          |
|  缓存:  Redis (会话 + 消息队列)                                   |
|  数据库: PostgreSQL (业务数据) + MongoDB (对话日志)               |
|  部署:  Docker Compose + Nginx                                   |
+------------------------------------------------------------------+
```

### 环境准备

```python
# requirements.txt - 项目依赖
"""
fastapi==0.104.1
uvicorn==0.24.0
openai==1.6.1
chromadb==0.4.22
redis==5.0.1
pydantic==2.5.3
sqlalchemy==2.0.23
python-multipart==0.0.6
websockets==12.0
httpx==0.25.2
tenacity==8.2.3
python-jose==3.3.0
passlib==1.7.4
celery==5.3.6
jinja2==3.1.2
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AppConfig:
    """应用配置类 - 统一管理所有配置项"""

    # OpenAI 配置
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # ChromaDB 配置
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "customer_service_kb"

    # Redis 配置
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = 0

    # 对话配置
    max_history_turns: int = 10          # 最多保留的对话轮数
    session_ttl_seconds: int = 3600      # 会话过期时间(秒)
    max_rag_results: int = 5             # RAG最多返回结果数
    similarity_threshold: float = 0.75   # 相似度阈值

    # 系统提示词
    system_prompt: str = """你是一个专业的智能客服助手。你的职责是:
1. 准确回答用户关于产品和服务的问题
2. 帮助用户查询订单、处理退款等操作
3. 当无法回答时，主动转接人工客服
4. 保持友好、专业、耐心的态度

重要规则:
- 只基于知识库内容回答，不编造信息
- 涉及金额操作需要二次确认
- 敏感操作(退款/注销)需要验证身份"""


# 创建全局配置实例
config = AppConfig()
print(f"[配置加载完成] 模型: {config.chat_model}, 向量库: {config.chroma_persist_dir}")
```

---

## 系统架构

### 分层架构设计

```
+=======================================================================+
|                     分层架构详细设计                                    |
+=======================================================================+
|                                                                       |
|  +--- 接入层 (Access Layer) ----------------------------------------+ |
|  |                                                                   | |
|  |  HTTP REST API    WebSocket长连接    Webhook回调                  | |
|  |  (问答请求)        (实时对话)         (第三方通知)                 | |
|  +-------------------------------------------------------------------+ |
|                              |                                        |
|  +--- 业务层 (Business Layer) --------------------------------------+ |
|  |                                                                   | |
|  |  +-----------+  +------------+  +------------+  +-----------+    | |
|  |  | 意图识别  |  | 对话编排   |  | 会话管理   |  | 权限校验  |    | |
|  |  | Intent    |  | Dialogue   |  | Session    |  | Auth      |    | |
|  |  | Classify  |  | Orchestra  |  | Manager    |  | Guard     |    | |
|  |  +-----------+  +------------+  +------------+  +-----------+    | |
|  +-------------------------------------------------------------------+ |
|                              |                                        |
|  +--- 能力层 (Capability Layer) ------------------------------------+ |
|  |                                                                   | |
|  |  +-----------+  +------------+  +------------+  +-----------+    | |
|  |  | RAG检索   |  | LLM生成   |  | Agent执行  |  | 对话评估  |    | |
|  |  | Retrieval |  | Generation |  | Tool Call  |  | Evaluate  |    | |
|  |  +-----------+  +------------+  +------------+  +-----------+    | |
|  +-------------------------------------------------------------------+ |
|                              |                                        |
|  +--- 数据层 (Data Layer) ------------------------------------------+ |
|  |                                                                   | |
|  |  ChromaDB     Redis       PostgreSQL     MongoDB                  | |
|  |  (向量)       (缓存)      (业务)         (日志)                   | |
|  +-------------------------------------------------------------------+ |
|                                                                       |
+=======================================================================+
```

### 请求处理流程

```
用户发送消息
       |
       v
+------+-------+
| 接收消息      |
| 提取session_id|
+------+-------+
       |
       v
+------+-------+
| 加载会话上下文|  <--- Redis: 获取历史消息
| 检查会话状态  |
+------+-------+
       |
       v
+------+--------+
| 意图识别       |  <--- LLM: 判断用户意图
| - FAQ问答      |
| - 业务操作     |
| - 闲聊寒暄     |
| - 投诉/转人工  |
+------+--------+
       |
       +------------------+------------------+------------------+
       |                  |                  |                  |
       v                  v                  v                  v
  +----+----+       +-----+-----+      +----+----+      +-----+-----+
  |  RAG    |       |  Agent    |      | 闲聊    |      | 人工转接  |
  |  知识库 |       |  工具调用 |      | LLM直接 |      | WebSocket |
  |  检索   |       |  执行操作 |      | 生成    |      | 通知      |
  +----+----+       +-----+-----+      +----+----+      +-----+-----+
       |                  |                  |                  |
       +------------------+------------------+------------------+
       |
       v
+------+-------+
| 生成回复      |  <--- LLM: 组合上下文生成最终答案
| 格式化输出    |
+------+-------+
       |
       v
+------+-------+
| 保存对话记录  |  ---> Redis: 更新会话历史
| 异步写日志    |  ---> MongoDB: 对话日志
+------+-------+
       |
       v
  返回给用户
```

### 数据模型定义

```python
"""
数据模型定义 - 使用 Pydantic 定义所有数据结构
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid


class MessageRole(str, Enum):
    """消息角色"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class IntentType(str, Enum):
    """意图类型"""
    FAQ = "faq"                      # 知识库问答
    ORDER_QUERY = "order_query"      # 查询订单
    REFUND = "refund"                # 退款申请
    COMPLAINT = "complaint"          # 投诉建议
    CHITCHAT = "chitchat"            # 闲聊寒暄
    HUMAN_TRANSFER = "human_transfer"  # 转人工
    UNKNOWN = "unknown"              # 未知意图


class ChatMessage(BaseModel):
    """单条对话消息"""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """用户请求模型"""
    session_id: Optional[str] = Field(default=None, description="会话ID，为空则新建会话")
    message: str = Field(..., min_length=1, max_length=2000, description="用户消息")
    user_id: Optional[str] = Field(default=None, description="用户ID，用于身份识别")


class SourceReference(BaseModel):
    """RAG引用来源"""
    doc_title: str = Field(description="文档标题")
    chunk_text: str = Field(description="引用的原文片段")
    similarity_score: float = Field(description="相似度分数")
    page_number: Optional[int] = Field(default=None, description="页码")


class ToolCallResult(BaseModel):
    """工具调用结果"""
    tool_name: str = Field(description="工具名称")
    arguments: Dict[str, Any] = Field(description="调用参数")
    result: Any = Field(description="执行结果")
    success: bool = Field(default=True, description="是否成功")


class ChatResponse(BaseModel):
    """系统回复模型"""
    session_id: str = Field(description="会话ID")
    reply: str = Field(description="回复内容")
    intent: IntentType = Field(description="识别的意图")
    sources: List[SourceReference] = Field(default_factory=list, description="引用来源")
    tool_calls: List[ToolCallResult] = Field(default_factory=list, description="工具调用")
    confidence: float = Field(default=0.0, description="置信度")
    need_human: bool = Field(default=False, description="是否需要转人工")
    timestamp: datetime = Field(default_factory=datetime.now)


class SessionState(BaseModel):
    """会话状态"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    messages: List[ChatMessage] = Field(default_factory=list)
    intent_history: List[IntentType] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    is_transferred: bool = False      # 是否已转人工


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    # 创建一个请求
    request = ChatRequest(
        session_id="test-session-001",
        message="我想查一下我的订单状态",
        user_id="user-123"
    )
    print(f"[请求] session={request.session_id}, message={request.message}")

    # 创建一个回复
    response = ChatResponse(
        session_id="test-session-001",
        reply="好的，请提供您的订单号，我帮您查询。",
        intent=IntentType.ORDER_QUERY,
        confidence=0.95,
    )
    print(f"[回复] intent={response.intent}, reply={response.reply}")

    # 创建会话状态
    session = SessionState(user_id="user-123")
    session.messages.append(ChatMessage(role=MessageRole.USER, content="你好"))
    session.messages.append(ChatMessage(role=MessageRole.ASSISTANT, content="您好！有什么可以帮您？"))
    print(f"[会话] id={session.session_id}, 消息数={len(session.messages)}")
```

---

## RAG知识库实现

### RAG 工作原理

```
+=======================================================================+
|                    RAG (检索增强生成) 工作原理                          |
+=======================================================================+
|                                                                       |
|  【离线阶段 - 知识库构建】                                             |
|                                                                       |
|  原始文档             文档分块              向量化             存储    |
|  +----------+     +----------+        +----------+      +---------+  |
|  | FAQ.md   | --> | chunk_1  |  --->  | [0.12,   | ---> |         |  |
|  | 产品手册 | --> | chunk_2  |  --->  |  0.45,   | ---> | ChromaDB|  |
|  | 退款政策 | --> | chunk_3  |  --->  |  -0.23,  | ---> |  向量   |  |
|  | ...      | --> | ...      |  --->  |  ...]    | ---> |  数据库 |  |
|  +----------+     +----------+        +----------+      +---------+  |
|       |                |                    |                         |
|    读取文件        RecursiveText        OpenAI                       |
|                    Splitter             Embedding                     |
|                                                                       |
|  【在线阶段 - 检索问答】                                               |
|                                                                       |
|  用户问题 --> Embedding --> 向量检索 --> Top-K文档 --> LLM生成回答    |
|                                |                         |            |
|                                v                         v            |
|                           相似度计算               Prompt + Context   |
|                          (余弦相似度)             组合生成最终答案     |
|                                                                       |
+=======================================================================+
```

### 文档加载与分块

```python
"""
文档加载器 - 支持多种格式的文档加载和智能分块
"""
import os
import re
import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    """文档分块数据结构"""
    chunk_id: str              # 分块唯一ID
    content: str               # 分块文本内容
    doc_title: str             # 来源文档标题
    doc_path: str              # 来源文档路径
    chunk_index: int           # 在文档中的序号
    page_number: Optional[int] = None  # 页码(PDF)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.chunk_id:
            # 自动生成chunk_id
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            self.chunk_id = f"{self.doc_title}_{self.chunk_index}_{content_hash}"


class TextSplitter:
    """
    递归文本分块器

    策略: 按优先级尝试不同分隔符，确保语义完整性
    分隔符优先级: 段落 > 句子 > 逗号 > 空格 > 字符
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """将文本分成多个块"""
        final_chunks: List[str] = []
        # 找到合适的分隔符
        separator = self.separators[-1]
        for sep in self.separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        # 按分隔符切分
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # 合并小块
        current_chunk = ""
        for split_text in splits:
            piece = split_text + separator if separator else split_text
            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                if current_chunk.strip():
                    final_chunks.append(current_chunk.strip())
                # 保留overlap
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + piece
                else:
                    current_chunk = piece

        if current_chunk.strip():
            final_chunks.append(current_chunk.strip())

        return final_chunks


class DocumentLoader:
    """
    多格式文档加载器

    支持格式: .txt, .md, .csv, .json
    """

    def __init__(self, splitter: Optional[TextSplitter] = None):
        self.splitter = splitter or TextSplitter()

    def load_file(self, file_path: str) -> List[DocumentChunk]:
        """加载单个文件并返回分块列表"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        doc_title = os.path.basename(file_path)

        # 根据文件类型读取内容
        if ext in (".txt", ".md"):
            content = self._load_text(file_path)
        elif ext == ".csv":
            content = self._load_csv(file_path)
        elif ext == ".json":
            content = self._load_json(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

        # 分块
        chunks_text = self.splitter.split_text(content)

        # 构建DocumentChunk对象
        chunks = []
        for i, text in enumerate(chunks_text):
            chunk = DocumentChunk(
                chunk_id="",
                content=text,
                doc_title=doc_title,
                doc_path=file_path,
                chunk_index=i,
                metadata={"file_type": ext, "total_chunks": len(chunks_text)},
            )
            chunks.append(chunk)

        return chunks

    def load_directory(self, dir_path: str, extensions: Optional[List[str]] = None) -> List[DocumentChunk]:
        """批量加载目录下的所有文档"""
        extensions = extensions or [".txt", ".md", ".csv", ".json"]
        all_chunks: List[DocumentChunk] = []

        for root, _dirs, files in os.walk(dir_path):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext in extensions:
                    fpath = os.path.join(root, fname)
                    try:
                        chunks = self.load_file(fpath)
                        all_chunks.extend(chunks)
                        print(f"  [加载成功] {fname} -> {len(chunks)} 个分块")
                    except Exception as e:
                        print(f"  [加载失败] {fname}: {e}")

        print(f"[文档加载完成] 共 {len(all_chunks)} 个分块")
        return all_chunks

    def _load_text(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_csv(self, path: str) -> str:
        import csv
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(" | ".join(row))
        return "\n".join(rows)

    def _load_json(self, path: str) -> str:
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return "\n\n".join(
                json.dumps(item, ensure_ascii=False, indent=2) if isinstance(item, dict)
                else str(item)
                for item in data
            )
        return json.dumps(data, ensure_ascii=False, indent=2)


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    # 创建分块器
    splitter = TextSplitter(chunk_size=300, chunk_overlap=30)

    # 测试文本分块
    sample_text = """
    退款政策说明

    1. 7天无理由退款：购买后7天内可申请无理由退款，商品需保持完好。
    2. 质量问题退款：收到商品存在质量问题，15天内可申请退款退货。
    3. 退款流程：提交退款申请 -> 客服审核 -> 退回商品 -> 确认收货 -> 退款到账。
    4. 退款时效：审核通过后，退款将在3-5个工作日内原路返回。
    5. 不支持退款的情况：定制商品、已使用的虚拟商品、超过退款期限。
    """
    chunks = splitter.split_text(sample_text.strip())
    for i, chunk in enumerate(chunks):
        print(f"--- 分块 {i+1} (长度: {len(chunk)}) ---")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
```

### 向量存储与检索

```python
"""
向量知识库 - 基于 ChromaDB 的向量存储和语义检索
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
import os

# ChromaDB 和 OpenAI 客户端
try:
    import chromadb
    from chromadb.config import Settings
    from openai import OpenAI
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("[警告] 请安装依赖: pip install chromadb openai")


@dataclass
class RetrievalResult:
    """检索结果"""
    content: str
    doc_title: str
    chunk_id: str
    similarity_score: float
    metadata: Dict


class VectorKnowledgeBase:
    """
    向量知识库

    功能:
    1. 文档向量化并存入ChromaDB
    2. 语义检索 - 根据查询返回最相关的文档分块
    3. 支持增量更新和文档删除

    使用流程:
        kb = VectorKnowledgeBase()
        kb.add_documents(chunks)              # 添加文档
        results = kb.search("退款政策是什么")   # 语义检索
    """

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "customer_service_kb",
        embedding_model: str = "text-embedding-3-small",
    ):
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        if not HAS_DEPS:
            raise RuntimeError("缺少依赖，请安装: pip install chromadb openai")

        # 初始化 OpenAI 客户端
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

        # 初始化 ChromaDB
        os.makedirs(persist_directory, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
        )
        print(f"[向量知识库] 集合: {collection_name}, 已有 {self.collection.count()} 条记录")

    def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        """调用 OpenAI Embedding API 获取向量"""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def add_documents(self, chunks: list, batch_size: int = 50) -> int:
        """
        批量添加文档到向量库

        参数:
            chunks: DocumentChunk 列表
            batch_size: 每批处理数量(避免API超时)
        返回:
            成功添加的文档数量
        """
        added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [
                {
                    "doc_title": c.doc_title,
                    "doc_path": c.doc_path,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number or -1,
                }
                for c in batch
            ]

            # 获取向量
            embeddings = self._get_embedding(texts)

            # 写入ChromaDB
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            added += len(batch)
            print(f"  [写入] {added}/{len(chunks)} 条")

        print(f"[文档入库完成] 共 {added} 条，向量库总量: {self.collection.count()}")
        return added

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.70,
        filter_metadata: Optional[Dict] = None,
    ) -> List[RetrievalResult]:
        """
        语义检索

        参数:
            query: 用户查询文本
            top_k: 返回最多结果数
            score_threshold: 最低相似度阈值
            filter_metadata: 元数据过滤条件
        """
        # 查询向量化
        query_embedding = self._get_embedding([query])[0]

        # ChromaDB检索
        search_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_metadata:
            search_params["where"] = filter_metadata

        results = self.collection.query(**search_params)

        # 解析结果
        retrieval_results = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB cosine distance: 0=完全相同, 2=完全不同
                # 转换为相似度: similarity = 1 - distance/2
                similarity = 1 - dist / 2

                if similarity >= score_threshold:
                    retrieval_results.append(
                        RetrievalResult(
                            content=doc,
                            doc_title=meta.get("doc_title", ""),
                            chunk_id=meta.get("chunk_id", ""),
                            similarity_score=round(similarity, 4),
                            metadata=meta,
                        )
                    )

        # 按相似度降序排列
        retrieval_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return retrieval_results

    def delete_document(self, doc_title: str) -> int:
        """删除指定文档的所有分块"""
        # 查找该文档的所有chunk
        results = self.collection.get(where={"doc_title": doc_title})
        if results and results["ids"]:
            self.collection.delete(ids=results["ids"])
            print(f"[删除文档] {doc_title}: 删除 {len(results['ids'])} 个分块")
            return len(results["ids"])
        return 0

    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "embedding_model": self.embedding_model,
        }


# ============================================================
# RAG问答链
# ============================================================
class RAGChain:
    """
    RAG问答链 - 组合检索和生成

    流程: 用户问题 -> 向量检索 -> 构建Prompt -> LLM生成 -> 返回带引用的答案
    """

    def __init__(self, knowledge_base: VectorKnowledgeBase):
        self.kb = knowledge_base
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

    def build_rag_prompt(self, query: str, contexts: List[RetrievalResult]) -> str:
        """构建包含检索上下文的Prompt"""
        context_text = ""
        for i, ctx in enumerate(contexts, 1):
            context_text += f"\n【参考资料 {i}】(来源: {ctx.doc_title}, 相似度: {ctx.similarity_score})\n"
            context_text += f"{ctx.content}\n"

        prompt = f"""请基于以下参考资料回答用户问题。

要求:
1. 只使用参考资料中的信息回答，不要编造内容
2. 如果参考资料不足以回答问题，请诚实说明
3. 回答要简洁、准确、有条理
4. 在回答末尾标注引用了哪些参考资料

===== 参考资料 =====
{context_text}
===== 参考资料结束 =====

用户问题: {query}

请回答:"""
        return prompt

    def answer(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None,
        top_k: int = 5,
    ) -> Tuple[str, List[RetrievalResult]]:
        """
        RAG问答

        返回: (回答文本, 引用来源列表)
        """
        # 1. 检索相关文档
        results = self.kb.search(query, top_k=top_k)

        if not results:
            return "抱歉，我在知识库中没有找到相关信息。请问您能换个方式描述问题吗？", []

        # 2. 构建Prompt
        rag_prompt = self.build_rag_prompt(query, results)

        # 3. 构建消息列表
        messages = [
            {"role": "system", "content": "你是一个专业的客服助手，基于提供的参考资料回答问题。"},
        ]
        # 添加历史对话(提供上下文)
        if chat_history:
            for msg in chat_history[-6:]:  # 最多保留最近3轮
                messages.append(msg)

        messages.append({"role": "user", "content": rag_prompt})

        # 4. 调用LLM生成回答
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,       # 低温度，更准确
            max_tokens=1000,
        )

        answer_text = response.choices[0].message.content
        return answer_text, results


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    print("=== RAG知识库 使用示例 ===")
    print()
    print("# 1. 初始化知识库")
    print("kb = VectorKnowledgeBase(persist_directory='./data/chroma_db')")
    print()
    print("# 2. 加载文档")
    print("loader = DocumentLoader()")
    print("chunks = loader.load_directory('./knowledge_base/')")
    print("kb.add_documents(chunks)")
    print()
    print("# 3. 检索测试")
    print("results = kb.search('退款需要多长时间')")
    print("for r in results:")
    print("    print(f'  [{r.similarity_score}] {r.doc_title}: {r.content[:80]}...')")
    print()
    print("# 4. RAG问答")
    print("rag = RAGChain(kb)")
    print("answer, sources = rag.answer('退款需要多长时间？')")
    print(f"# answer = '退款审核通过后3-5个工作日原路返回...'")
```

---

## 多轮对话管理

### 对话管理架构

```
+=======================================================================+
|                      多轮对话管理架构                                   |
+=======================================================================+
|                                                                       |
|  +--- 会话存储 (Redis) -------------------------------------------+  |
|  |                                                                 |  |
|  |  session:abc123 = {                                             |  |
|  |    "messages": [                                                |  |
|  |      {"role":"user",      "content":"我想查订单"},               |  |
|  |      {"role":"assistant", "content":"请提供订单号"},             |  |
|  |      {"role":"user",      "content":"A12345"},                  |  |
|  |      {"role":"assistant", "content":"订单A12345状态..."}        |  |
|  |    ],                                                           |  |
|  |    "context": {"order_id":"A12345", "verified": true},          |  |
|  |    "intent_history": ["order_query", "order_query"]             |  |
|  |  }                                                              |  |
|  |  TTL: 3600秒                                                    |  |
|  +-----------------------------------------------------------------+  |
|                                                                       |
|  +--- 上下文窗口策略 ---------------------------------------------+  |
|  |                                                                 |  |
|  |  滑动窗口: 保留最近 N 轮对话                                    |  |
|  |                                                                 |  |
|  |  [轮1] [轮2] [轮3] [轮4] [轮5] [轮6] [轮7] [轮8]              |  |
|  |                          |<--- 保留最近5轮 --->|               |  |
|  |  |<-- 被截断(摘要化) -->|                                      |  |
|  |                                                                 |  |
|  |  摘要策略: 对早期对话生成摘要，作为系统消息注入                  |  |
|  +-----------------------------------------------------------------+  |
|                                                                       |
+=======================================================================+
```

### 会话管理器

```python
"""
会话管理器 - 基于Redis的多轮对话状态管理
"""
import json
import time
import uuid
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

# 尝试导入Redis，提供Mock实现用于本地测试
try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


class InMemoryStore:
    """
    内存存储 - Redis的本地替代，用于开发和测试
    接口与Redis兼容
    """

    def __init__(self):
        self._store: Dict[str, str] = {}
        self._ttl: Dict[str, float] = {}

    def get(self, key: str) -> Optional[str]:
        # 检查过期
        if key in self._ttl and time.time() > self._ttl[key]:
            del self._store[key]
            del self._ttl[key]
            return None
        return self._store.get(key)

    def set(self, key: str, value: str, ex: Optional[int] = None):
        self._store[key] = value
        if ex:
            self._ttl[key] = time.time() + ex

    def delete(self, key: str):
        self._store.pop(key, None)
        self._ttl.pop(key, None)

    def exists(self, key: str) -> bool:
        if key in self._ttl and time.time() > self._ttl[key]:
            del self._store[key]
            del self._ttl[key]
            return False
        return key in self._store

    def keys(self, pattern: str = "*") -> List[str]:
        import fnmatch
        return [k for k in self._store.keys() if fnmatch.fnmatch(k, pattern)]


@dataclass
class Message:
    """对话消息"""
    role: str        # "user" | "assistant" | "system"
    content: str
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class Session:
    """会话对象"""
    session_id: str = ""
    user_id: str = ""
    messages: List[Message] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    intent_history: List[str] = field(default_factory=list)
    created_at: str = ""
    last_active: str = ""
    is_transferred: bool = False

    def __post_init__(self):
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.last_active:
            self.last_active = now


class SessionManager:
    """
    会话管理器

    负责:
    1. 会话的创建、读取、更新、删除
    2. 对话历史的管理(滑动窗口 + 历史摘要)
    3. 上下文变量的维护
    4. 会话过期自动清理
    """

    KEY_PREFIX = "chatbot:session:"

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        session_ttl: int = 3600,
        max_history_turns: int = 10,
        use_memory: bool = False,
    ):
        self.session_ttl = session_ttl
        self.max_history_turns = max_history_turns

        # 选择存储后端
        if use_memory or not HAS_REDIS:
            self.store = InMemoryStore()
            print("[会话管理器] 使用内存存储(开发模式)")
        else:
            self.store = redis.Redis(
                host=redis_host, port=redis_port, db=redis_db, decode_responses=True
            )
            print(f"[会话管理器] 连接Redis: {redis_host}:{redis_port}")

    def _key(self, session_id: str) -> str:
        return f"{self.KEY_PREFIX}{session_id}"

    def create_session(self, user_id: str = "") -> Session:
        """创建新会话"""
        session = Session(user_id=user_id)
        self._save(session)
        print(f"[新会话] id={session.session_id}, user={user_id}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话，不存在则返回None"""
        data = self.store.get(self._key(session_id))
        if data is None:
            return None
        return self._deserialize(data)

    def get_or_create(self, session_id: Optional[str] = None, user_id: str = "") -> Session:
        """获取或创建会话"""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        return self.create_session(user_id)

    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> Session:
        """添加消息到会话"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"会话不存在: {session_id}")

        msg = Message(role=role, content=content, metadata=metadata or {})
        session.messages.append(msg)
        session.last_active = datetime.now().isoformat()

        # 如果超过最大轮数，截断早期消息
        max_messages = self.max_history_turns * 2  # 每轮 user+assistant
        if len(session.messages) > max_messages:
            # 保留system消息 + 最近的N条
            system_msgs = [m for m in session.messages if m.role == "system"]
            recent_msgs = session.messages[-max_messages:]
            session.messages = system_msgs + recent_msgs

        self._save(session)
        return session

    def update_context(self, session_id: str, key: str, value: Any):
        """更新会话上下文变量"""
        session = self.get_session(session_id)
        if session:
            session.context[key] = value
            self._save(session)

    def add_intent(self, session_id: str, intent: str):
        """记录意图历史"""
        session = self.get_session(session_id)
        if session:
            session.intent_history.append(intent)
            self._save(session)

    def get_chat_history(self, session_id: str, max_turns: Optional[int] = None) -> List[Dict]:
        """获取对话历史(OpenAI格式)"""
        session = self.get_session(session_id)
        if not session:
            return []

        messages = session.messages
        if max_turns:
            # 只取最近N轮
            messages = messages[-(max_turns * 2):]

        return [{"role": m.role, "content": m.content} for m in messages]

    def delete_session(self, session_id: str):
        """删除会话"""
        self.store.delete(self._key(session_id))
        print(f"[删除会话] {session_id}")

    def list_active_sessions(self) -> List[str]:
        """列出所有活跃会话"""
        keys = self.store.keys(f"{self.KEY_PREFIX}*")
        return [k.replace(self.KEY_PREFIX, "") for k in keys]

    def _save(self, session: Session):
        """序列化并保存会话"""
        data = json.dumps(asdict(session), ensure_ascii=False)
        self.store.set(self._key(session.session_id), data, ex=self.session_ttl)

    def _deserialize(self, data: str) -> Session:
        """反序列化会话"""
        d = json.loads(data)
        d["messages"] = [Message(**m) for m in d.get("messages", [])]
        return Session(**d)


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    # 使用内存存储进行测试
    manager = SessionManager(use_memory=True, max_history_turns=5)

    # 创建会话
    session = manager.create_session(user_id="user-001")
    sid = session.session_id
    print(f"会话ID: {sid}")

    # 模拟多轮对话
    conversations = [
        ("user", "你好，我想查一下订单"),
        ("assistant", "您好！请提供您的订单号，我帮您查询。"),
        ("user", "订单号是 ORD-20240101-001"),
        ("assistant", "查询到订单 ORD-20240101-001：\n- 商品: iPhone 15\n- 状态: 已发货\n- 预计到达: 1月5日"),
        ("user", "物流到哪了？"),
        ("assistant", "您的包裹目前在北京转运中心，预计明天送达。"),
        ("user", "好的，谢谢"),
        ("assistant", "不客气！还有其他问题随时问我。"),
    ]

    for role, content in conversations:
        manager.add_message(sid, role, content)

    # 查看对话历史
    history = manager.get_chat_history(sid)
    print(f"\n对话历史 ({len(history)} 条):")
    for msg in history:
        prefix = "  用户: " if msg["role"] == "user" else "  助手: "
        print(f"{prefix}{msg['content'][:50]}")

    # 更新上下文
    manager.update_context(sid, "order_id", "ORD-20240101-001")
    manager.update_context(sid, "verified", True)

    # 查看会话状态
    session = manager.get_session(sid)
    print(f"\n会话上下文: {session.context}")
    print(f"活跃会话数: {len(manager.list_active_sessions())}")
```

---

## Agent工具调用

### Agent 架构

```
+=======================================================================+
|                      Agent 工具调用架构                                 |
+=======================================================================+
|                                                                       |
|  用户消息: "帮我退款，订单号A12345"                                    |
|       |                                                               |
|       v                                                               |
|  +----+--------------------------------------------+                  |
|  |              LLM (function calling)             |                  |
|  |                                                  |                  |
|  |  分析用户意图，决定需要调用哪些工具              |                  |
|  |  输出: tool_name + arguments                     |                  |
|  +----+--------------------------------------------+                  |
|       |                                                               |
|       v                                                               |
|  +----+--------------------------------------------+                  |
|  |              工具路由器 (Tool Router)             |                  |
|  +----+--------+--------+--------+---------+-------+                  |
|       |        |        |        |         |                          |
|       v        v        v        v         v                          |
|  +--------+ +------+ +------+ +-------+ +--------+                   |
|  | 查订单 | | 退款 | | 查物流| | 查余额 | | 修改  |                   |
|  | query  | | refund| | track | | balance| | 信息  |                   |
|  | _order | |       | |       | |        | | update|                   |
|  +----+---+ +---+--+ +---+--+ +----+---+ +---+---+                   |
|       |         |        |         |          |                       |
|       v         v        v         v          v                       |
|  +----+---------+--------+---------+----------+---+                   |
|  |            业务数据库 / 外部API                  |                   |
|  +------------------------------------------------+                   |
|       |                                                               |
|       v                                                               |
|  +----+--------------------------------------------+                  |
|  |    LLM 组合工具结果，生成自然语言回复            |                  |
|  +----+--------------------------------------------+                  |
|       |                                                               |
|       v                                                               |
|  回复用户: "您的订单A12345退款已提交，预计3个工作日到账"              |
|                                                                       |
+=======================================================================+
```

### 工具定义与执行

```python
"""
Agent 工具系统 - 定义和执行客服场景下的各种工具
"""
import json
import random
import string
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


# ============================================================
# 工具定义 (OpenAI Function Calling 格式)
# ============================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "根据订单号查询订单详情，包括商品信息、金额、状态、物流等",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号，格式如 ORD-YYYYMMDD-NNN",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_refund",
            "description": "提交退款申请。需要订单号和退款原因。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "要退款的订单号",
                    },
                    "reason": {
                        "type": "string",
                        "enum": ["quality_issue", "wrong_item", "no_longer_needed", "not_received", "other"],
                        "description": "退款原因",
                    },
                    "description": {
                        "type": "string",
                        "description": "详细描述(可选)",
                    },
                },
                "required": ["order_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "track_shipping",
            "description": "查询物流配送状态",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "订单号",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_balance",
            "description": "查询用户账户余额和积分",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "用户ID",
                    }
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_user_info",
            "description": "修改用户信息（收货地址、手机号等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "用户ID"},
                    "field": {
                        "type": "string",
                        "enum": ["address", "phone", "email", "name"],
                        "description": "要修改的字段",
                    },
                    "new_value": {"type": "string", "description": "新值"},
                },
                "required": ["user_id", "field", "new_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "搜索知识库获取产品信息、政策说明等",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["query"],
            },
        },
    },
]


# ============================================================
# 工具实现 (模拟业务逻辑)
# ============================================================

# 模拟数据库
MOCK_ORDERS = {
    "ORD-20240101-001": {
        "order_id": "ORD-20240101-001",
        "user_id": "user-001",
        "product": "iPhone 15 128GB 蓝色",
        "amount": 5999.00,
        "status": "shipped",
        "status_text": "已发货",
        "created_at": "2024-01-01 10:30:00",
        "shipping_company": "顺丰速运",
        "tracking_number": "SF1234567890",
    },
    "ORD-20240102-002": {
        "order_id": "ORD-20240102-002",
        "user_id": "user-001",
        "product": "AirPods Pro 2",
        "amount": 1799.00,
        "status": "delivered",
        "status_text": "已签收",
        "created_at": "2024-01-02 14:20:00",
        "shipping_company": "京东物流",
        "tracking_number": "JD9876543210",
    },
}

MOCK_USERS = {
    "user-001": {
        "user_id": "user-001",
        "name": "张三",
        "phone": "138****1234",
        "balance": 200.50,
        "points": 3580,
        "address": "北京市朝阳区xxx街道xxx号",
    },
}


def query_order(order_id: str) -> Dict[str, Any]:
    """查询订单"""
    order = MOCK_ORDERS.get(order_id)
    if not order:
        return {"success": False, "error": f"未找到订单: {order_id}"}
    return {"success": True, "data": order}


def submit_refund(order_id: str, reason: str, description: str = "") -> Dict[str, Any]:
    """提交退款"""
    order = MOCK_ORDERS.get(order_id)
    if not order:
        return {"success": False, "error": f"未找到订单: {order_id}"}

    if order["status"] == "refunded":
        return {"success": False, "error": "该订单已退款，请勿重复申请"}

    refund_id = "REF-" + "".join(random.choices(string.digits, k=8))
    return {
        "success": True,
        "data": {
            "refund_id": refund_id,
            "order_id": order_id,
            "amount": order["amount"],
            "reason": reason,
            "description": description,
            "status": "pending",
            "estimated_days": "3-5个工作日",
        },
    }


def track_shipping(order_id: str) -> Dict[str, Any]:
    """查询物流"""
    order = MOCK_ORDERS.get(order_id)
    if not order:
        return {"success": False, "error": f"未找到订单: {order_id}"}

    tracking_info = [
        {"time": "2024-01-03 08:00", "status": "包裹已揽收", "location": "深圳"},
        {"time": "2024-01-03 15:30", "status": "已到达转运中心", "location": "广州"},
        {"time": "2024-01-04 06:00", "status": "运输中", "location": "武汉"},
        {"time": "2024-01-04 18:00", "status": "已到达目的城市", "location": "北京"},
        {"time": "2024-01-05 09:30", "status": "派送中", "location": "北京朝阳区"},
    ]
    return {
        "success": True,
        "data": {
            "order_id": order_id,
            "company": order.get("shipping_company", "未知"),
            "tracking_number": order.get("tracking_number", ""),
            "latest_status": tracking_info[-1]["status"],
            "tracking_details": tracking_info,
        },
    }


def check_balance(user_id: str) -> Dict[str, Any]:
    """查询余额"""
    user = MOCK_USERS.get(user_id)
    if not user:
        return {"success": False, "error": f"未找到用户: {user_id}"}
    return {
        "success": True,
        "data": {
            "user_id": user_id,
            "balance": user["balance"],
            "points": user["points"],
            "points_value": round(user["points"] / 100, 2),
        },
    }


def update_user_info(user_id: str, field_name: str, new_value: str) -> Dict[str, Any]:
    """修改用户信息"""
    user = MOCK_USERS.get(user_id)
    if not user:
        return {"success": False, "error": f"未找到用户: {user_id}"}

    old_value = user.get(field_name, "")
    # 模拟更新
    return {
        "success": True,
        "data": {
            "field": field_name,
            "old_value": old_value,
            "new_value": new_value,
            "message": f"已将{field_name}修改为: {new_value}",
        },
    }


def search_knowledge_base(query: str) -> Dict[str, Any]:
    """搜索知识库(模拟)"""
    # 实际项目中调用 VectorKnowledgeBase.search()
    mock_results = {
        "退款": "7天无理由退款，审核通过后3-5个工作日到账。",
        "发货": "订单支付后24小时内发货，偏远地区48小时。",
        "保修": "电子产品享受一年保修服务，人为损坏不在保修范围内。",
    }
    for keyword, answer in mock_results.items():
        if keyword in query:
            return {"success": True, "data": {"query": query, "results": [answer]}}
    return {"success": True, "data": {"query": query, "results": ["暂未找到相关信息"]}}


# ============================================================
# 工具注册表
# ============================================================

TOOL_REGISTRY: Dict[str, Callable] = {
    "query_order": query_order,
    "submit_refund": submit_refund,
    "track_shipping": track_shipping,
    "check_balance": check_balance,
    "update_user_info": update_user_info,
    "search_knowledge_base": search_knowledge_base,
}


class ToolExecutor:
    """
    工具执行器 - 安全地执行工具调用

    功能:
    1. 参数校验
    2. 权限检查
    3. 执行工具
    4. 记录日志
    """

    def __init__(self, registry: Optional[Dict[str, Callable]] = None):
        self.registry = registry or TOOL_REGISTRY
        # 需要二次确认的危险操作
        self.dangerous_tools = {"submit_refund", "update_user_info"}

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具调用"""
        if tool_name not in self.registry:
            return {"success": False, "error": f"未知工具: {tool_name}"}

        try:
            func = self.registry[tool_name]
            result = func(**arguments)
            print(f"  [工具调用] {tool_name}({arguments}) -> success={result.get('success')}")
            return result
        except Exception as e:
            print(f"  [工具错误] {tool_name}: {e}")
            return {"success": False, "error": str(e)}

    def needs_confirmation(self, tool_name: str) -> bool:
        """是否需要用户二次确认"""
        return tool_name in self.dangerous_tools

    def list_tools(self) -> List[str]:
        """列出所有可用工具"""
        return list(self.registry.keys())


# ============================================================
# Agent 主循环
# ============================================================

class CustomerServiceAgent:
    """
    客服Agent - 结合LLM和工具调用的智能代理

    工作流程:
    1. 接收用户消息
    2. LLM分析意图，决定是否调用工具
    3. 如需工具 -> 执行工具 -> 将结果返回LLM
    4. LLM生成最终回复
    """

    def __init__(self):
        self.tool_executor = ToolExecutor()
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", "your-key"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )
            self.available = True
        except ImportError:
            self.available = False

    def run(
        self,
        user_message: str,
        chat_history: Optional[List[Dict]] = None,
        user_id: str = "",
    ) -> Dict[str, Any]:
        """
        Agent主入口

        返回: {"reply": str, "tool_calls": list, "intent": str}
        """
        if not self.available:
            return self._mock_run(user_message, user_id)

        messages = [
            {
                "role": "system",
                "content": f"""你是一个智能客服助手。当前用户ID: {user_id}

你可以使用工具来帮助用户完成操作。规则:
1. 查询类操作直接执行
2. 退款、修改信息等操作需要先确认
3. 无法处理的问题建议转人工
4. 保持友好专业的语气""",
            }
        ]
        if chat_history:
            messages.extend(chat_history[-10:])
        messages.append({"role": "user", "content": user_message})

        # 第一次调用: LLM决定是否使用工具
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=0.3,
        )

        assistant_msg = response.choices[0].message
        tool_calls_result = []

        # 如果LLM决定调用工具
        if assistant_msg.tool_calls:
            messages.append(assistant_msg)

            for tool_call in assistant_msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                # 执行工具
                result = self.tool_executor.execute(func_name, func_args)
                tool_calls_result.append({
                    "tool": func_name,
                    "args": func_args,
                    "result": result,
                })

                # 将工具结果添加到消息
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

            # 第二次调用: LLM根据工具结果生成回复
            final_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.5,
            )
            reply = final_response.choices[0].message.content
        else:
            reply = assistant_msg.content

        return {
            "reply": reply,
            "tool_calls": tool_calls_result,
            "intent": self._classify_intent(tool_calls_result),
        }

    def _mock_run(self, user_message: str, user_id: str) -> Dict[str, Any]:
        """无API时的模拟运行"""
        tool_calls_result = []

        # 简单规则匹配
        if "订单" in user_message or "查询" in user_message:
            # 尝试提取订单号
            import re
            match = re.search(r"ORD-\d{8}-\d{3}", user_message)
            if match:
                oid = match.group()
                result = self.tool_executor.execute("query_order", {"order_id": oid})
                tool_calls_result.append({"tool": "query_order", "args": {"order_id": oid}, "result": result})
                if result["success"]:
                    data = result["data"]
                    reply = (
                        f"查询到您的订单信息:\n"
                        f"- 订单号: {data['order_id']}\n"
                        f"- 商品: {data['product']}\n"
                        f"- 金额: {data['amount']}元\n"
                        f"- 状态: {data['status_text']}"
                    )
                else:
                    reply = f"抱歉，{result['error']}"
            else:
                reply = "请提供您的订单号（格式: ORD-XXXXXXXX-XXX），我帮您查询。"

        elif "退款" in user_message:
            reply = "好的，请提供订单号和退款原因，我帮您提交退款申请。"

        elif "物流" in user_message or "快递" in user_message:
            reply = "请提供订单号，我帮您查询物流信息。"

        else:
            reply = "您好！我是智能客服助手，可以帮您查询订单、申请退款、查物流等。请问有什么可以帮您的？"

        return {
            "reply": reply,
            "tool_calls": tool_calls_result,
            "intent": self._classify_intent(tool_calls_result),
        }

    def _classify_intent(self, tool_calls: List) -> str:
        """根据工具调用推断意图"""
        if not tool_calls:
            return "chitchat"
        tool_names = [tc["tool"] for tc in tool_calls]
        if "query_order" in tool_names:
            return "order_query"
        if "submit_refund" in tool_names:
            return "refund"
        if "track_shipping" in tool_names:
            return "order_query"
        return "faq"


import os

# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    print("=== Agent 工具调用测试 ===\n")

    agent = CustomerServiceAgent()

    # 测试场景1: 查询订单
    print("[场景1] 查询订单")
    result = agent.run("帮我查一下订单 ORD-20240101-001 的状态", user_id="user-001")
    print(f"  回复: {result['reply']}")
    print(f"  意图: {result['intent']}")
    print(f"  工具调用: {len(result['tool_calls'])} 次")
    print()

    # 测试场景2: 退款
    print("[场景2] 退款咨询")
    result = agent.run("我想退款", user_id="user-001")
    print(f"  回复: {result['reply']}")
    print()

    # 测试场景3: 闲聊
    print("[场景3] 闲聊")
    result = agent.run("你好呀", user_id="user-001")
    print(f"  回复: {result['reply']}")
    print(f"  意图: {result['intent']}")
```

---

## FastAPI后端服务

### API架构

```
+=======================================================================+
|                       FastAPI 后端架构                                  |
+=======================================================================+
|                                                                       |
|  +--- API路由 ---------------------------------------------------+   |
|  |                                                                |   |
|  |  POST /api/v1/chat           主对话接口                        |   |
|  |  POST /api/v1/chat/stream    流式对话接口(SSE)                 |   |
|  |  GET  /api/v1/session/{id}   获取会话信息                      |   |
|  |  DELETE /api/v1/session/{id} 删除会话                          |   |
|  |  POST /api/v1/feedback       用户反馈                          |   |
|  |  GET  /api/v1/health         健康检查                          |   |
|  |  POST /api/v1/kb/upload      上传知识库文档                    |   |
|  |  WS   /ws/chat               WebSocket实时对话                 |   |
|  +----------------------------------------------------------------+   |
|                                                                       |
|  +--- 中间件 ----------------------------------------------------+   |
|  |                                                                |   |
|  |  CORS中间件     请求日志     限流中间件     认证中间件          |   |
|  +----------------------------------------------------------------+   |
|                                                                       |
+=======================================================================+
```

### 完整后端代码

```python
"""
FastAPI 后端服务 - 智能客服系统完整API
"""
import os
import json
import uuid
import time
import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime
from contextlib import asynccontextmanager

# FastAPI 相关
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field


# ============================================================
# 请求/响应模型
# ============================================================
class ChatRequestModel(BaseModel):
    session_id: Optional[str] = Field(default=None, description="会话ID")
    message: str = Field(..., min_length=1, max_length=2000, description="用户消息")
    user_id: Optional[str] = Field(default=None, description="用户ID")
    stream: bool = Field(default=False, description="是否流式返回")


class ChatResponseModel(BaseModel):
    session_id: str
    reply: str
    intent: str = "unknown"
    sources: List[Dict] = []
    tool_calls: List[Dict] = []
    confidence: float = 0.0
    need_human: bool = False
    timestamp: str = ""


class FeedbackModel(BaseModel):
    session_id: str
    message_index: int = -1
    rating: int = Field(..., ge=1, le=5, description="1-5分评价")
    comment: Optional[str] = None


class SessionInfoModel(BaseModel):
    session_id: str
    user_id: str
    message_count: int
    created_at: str
    last_active: str
    is_transferred: bool


# ============================================================
# 核心服务类 (简化版，整合前面的模块)
# ============================================================
class ChatService:
    """聊天服务 - 整合所有AI能力"""

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}      # 内存会话存储
        self.feedback_log: List[Dict] = []         # 反馈日志
        print("[ChatService] 初始化完成")

    def get_or_create_session(self, session_id: Optional[str], user_id: str = "") -> str:
        """获取或创建会话"""
        if session_id and session_id in self.sessions:
            self.sessions[session_id]["last_active"] = datetime.now().isoformat()
            return session_id

        new_id = session_id or str(uuid.uuid4())
        self.sessions[new_id] = {
            "session_id": new_id,
            "user_id": user_id,
            "messages": [],
            "context": {},
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "is_transferred": False,
        }
        return new_id

    def process_message(self, session_id: str, message: str, user_id: str = "") -> Dict:
        """处理用户消息并返回回复"""
        session = self.sessions.get(session_id, {})
        messages = session.get("messages", [])

        # 添加用户消息
        messages.append({"role": "user", "content": message, "time": datetime.now().isoformat()})

        # === 意图识别 (简化版规则) ===
        intent = self._classify_intent(message)

        # === 生成回复 ===
        reply, sources, tool_calls = self._generate_reply(message, intent, messages, user_id)

        # 添加助手回复
        messages.append({"role": "assistant", "content": reply, "time": datetime.now().isoformat()})
        session["messages"] = messages
        self.sessions[session_id] = session

        return {
            "reply": reply,
            "intent": intent,
            "sources": sources,
            "tool_calls": tool_calls,
            "confidence": 0.9 if intent != "unknown" else 0.5,
            "need_human": intent == "complaint",
        }

    def _classify_intent(self, message: str) -> str:
        """简化意图分类"""
        keywords_map = {
            "order_query": ["订单", "查询", "查一下", "物流", "快递", "发货"],
            "refund": ["退款", "退货", "退钱", "退回"],
            "complaint": ["投诉", "不满", "差评", "垃圾"],
            "faq": ["怎么", "如何", "什么是", "政策", "规则", "保修"],
            "chitchat": ["你好", "谢谢", "再见", "哈哈"],
        }
        for intent, keywords in keywords_map.items():
            for kw in keywords:
                if kw in message:
                    return intent
        return "unknown"

    def _generate_reply(
        self, message: str, intent: str, history: list, user_id: str
    ) -> tuple:
        """根据意图生成回复"""
        sources = []
        tool_calls = []

        reply_map = {
            "order_query": "请提供您的订单号，我帮您查询订单状态和物流信息。",
            "refund": "好的，请提供订单号和退款原因。我们支持7天无理由退款，审核通过后3-5个工作日到账。",
            "complaint": "非常抱歉给您带来不好的体验。我已记录您的反馈，马上为您转接人工客服处理。",
            "faq": "这是一个好问题！让我在知识库中为您查找相关信息...",
            "chitchat": "您好！我是智能客服助手，随时为您服务。您可以问我订单查询、退款、产品信息等问题。",
            "unknown": "感谢您的咨询！请问您是想查询订单、申请退款，还是有其他问题需要帮助？",
        }

        reply = reply_map.get(intent, reply_map["unknown"])

        # 模拟RAG引用来源
        if intent == "faq":
            sources = [
                {"doc_title": "产品FAQ.md", "chunk_text": "相关知识库内容...", "score": 0.92},
            ]

        return reply, sources, tool_calls

    async def process_message_stream(
        self, session_id: str, message: str, user_id: str = ""
    ) -> AsyncGenerator[str, None]:
        """流式处理消息 (SSE)"""
        result = self.process_message(session_id, message, user_id)
        reply = result["reply"]

        # 模拟逐字输出
        for i, char in enumerate(reply):
            chunk_data = {"type": "token", "content": char, "index": i}
            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.03)  # 模拟生成延迟

        # 发送完成信号
        done_data = {
            "type": "done",
            "session_id": session_id,
            "intent": result["intent"],
            "sources": result["sources"],
        }
        yield f"data: {json.dumps(done_data, ensure_ascii=False)}\n\n"


# ============================================================
# FastAPI 应用
# ============================================================

# 全局服务实例
chat_service = ChatService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("[启动] 智能客服系统后端")
    yield
    print("[关闭] 清理资源...")


app = FastAPI(
    title="智能客服系统 API",
    description="基于 RAG + Agent 的智能客服后端服务",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 健康检查 ---
@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(chat_service.sessions),
    }


# --- 主对话接口 ---
@app.post("/api/v1/chat", response_model=ChatResponseModel)
async def chat(request: ChatRequestModel):
    """
    主对话接口

    接收用户消息，返回AI回复。
    支持多轮对话，自动管理会话状态。
    """
    try:
        # 获取或创建会话
        session_id = chat_service.get_or_create_session(
            request.session_id, request.user_id or ""
        )

        # 处理消息
        result = chat_service.process_message(
            session_id, request.message, request.user_id or ""
        )

        return ChatResponseModel(
            session_id=session_id,
            reply=result["reply"],
            intent=result["intent"],
            sources=result["sources"],
            tool_calls=result["tool_calls"],
            confidence=result["confidence"],
            need_human=result["need_human"],
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理消息失败: {str(e)}")


# --- 流式对话接口 (SSE) ---
@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequestModel):
    """流式对话接口 - Server-Sent Events"""
    session_id = chat_service.get_or_create_session(
        request.session_id, request.user_id or ""
    )
    return StreamingResponse(
        chat_service.process_message_stream(session_id, request.message, request.user_id or ""),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- 获取会话信息 ---
@app.get("/api/v1/session/{session_id}", response_model=SessionInfoModel)
async def get_session(session_id: str):
    """获取会话详细信息"""
    session = chat_service.sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    return SessionInfoModel(
        session_id=session["session_id"],
        user_id=session.get("user_id", ""),
        message_count=len(session.get("messages", [])),
        created_at=session.get("created_at", ""),
        last_active=session.get("last_active", ""),
        is_transferred=session.get("is_transferred", False),
    )


# --- 删除会话 ---
@app.delete("/api/v1/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in chat_service.sessions:
        del chat_service.sessions[session_id]
        return {"message": f"会话 {session_id} 已删除"}
    raise HTTPException(status_code=404, detail="会话不存在")


# --- 用户反馈 ---
@app.post("/api/v1/feedback")
async def submit_feedback(feedback: FeedbackModel):
    """提交对话反馈评价"""
    chat_service.feedback_log.append({
        "session_id": feedback.session_id,
        "rating": feedback.rating,
        "comment": feedback.comment,
        "timestamp": datetime.now().isoformat(),
    })
    return {"message": "感谢您的反馈！", "feedback_id": str(uuid.uuid4())}


# --- WebSocket 实时对话 ---
class ConnectionManager:
    """WebSocket连接管理器"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)

    async def send_message(self, session_id: str, message: Dict):
        ws = self.active_connections.get(session_id)
        if ws:
            await ws.send_json(message)


ws_manager = ConnectionManager()


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket实时对话端点"""
    session_id = str(uuid.uuid4())
    await ws_manager.connect(session_id, websocket)
    chat_service.get_or_create_session(session_id)

    # 发送欢迎消息
    await ws_manager.send_message(session_id, {
        "type": "connected",
        "session_id": session_id,
        "message": "连接成功！我是智能客服助手，请问有什么可以帮您？",
    })

    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "")

            if not user_message:
                continue

            # 处理消息
            result = chat_service.process_message(session_id, user_message)

            # 返回回复
            await ws_manager.send_message(session_id, {
                "type": "reply",
                "session_id": session_id,
                "reply": result["reply"],
                "intent": result["intent"],
                "sources": result["sources"],
                "need_human": result["need_human"],
            })

    except WebSocketDisconnect:
        ws_manager.disconnect(session_id)
        print(f"[WebSocket断开] session={session_id}")


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  智能客服系统 - FastAPI 后端")
    print("  API文档: http://localhost:8000/docs")
    print("  健康检查: http://localhost:8000/api/v1/health")
    print("=" * 60)

    uvicorn.run(
        "01_chatbot:app",  # 或使用 app 直接传入
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
```

---

## 前端对话界面

### 前端架构

```
+=======================================================================+
|                        前端界面架构                                     |
+=======================================================================+
|                                                                       |
|  +--------------------------------------------------------------+    |
|  |                      对话界面布局                              |    |
|  +--------------------------------------------------------------+    |
|  |  +------------------+  +----------------------------------+  |    |
|  |  |                  |  |          对话区域                 |  |    |
|  |  |  侧边栏          |  |  +----------------------------+  |  |    |
|  |  |  +------------+  |  |  | [系统] 您好，有什么帮您？   |  |  |    |
|  |  |  | 会话列表   |  |  |  +----------------------------+  |  |    |
|  |  |  |            |  |  |  +----------------------------+  |  |    |
|  |  |  | > 会话1    |  |  |  | [用户] 查一下我的订单       |  |  |    |
|  |  |  |   会话2    |  |  |  +----------------------------+  |  |    |
|  |  |  |   会话3    |  |  |  +----------------------------+  |  |    |
|  |  |  +------------+  |  |  | [助手] 好的，订单信息...    |  |  |    |
|  |  |  +------------+  |  |  |  [来源: FAQ.md]             |  |  |    |
|  |  |  | 新建对话   |  |  |  +----------------------------+  |  |    |
|  |  |  +------------+  |  |                                  |  |    |
|  |  +------------------+  |  +----------------------------+  |  |    |
|  |                        |  | 输入框        [发送] [语音] |  |  |    |
|  |                        |  +----------------------------+  |  |    |
|  |                        +----------------------------------+  |    |
|  +--------------------------------------------------------------+    |
|                                                                       |
+=======================================================================+
```

### HTML + JavaScript 前端

```python
"""
前端对话界面 - 使用 FastAPI 内置模板引擎提供简单的聊天界面
实际生产中建议使用 React/Vue 独立前端项目
"""

# 以下为嵌入式HTML模板，通过FastAPI的Jinja2渲染
CHAT_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能客服系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .chat-container {
            width: 480px; height: 680px;
            background: white; border-radius: 16px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.1);
            display: flex; flex-direction: column;
            overflow: hidden;
        }
        .chat-header {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white; padding: 16px 20px;
            font-size: 16px; font-weight: 600;
        }
        .chat-header small { opacity: 0.8; font-weight: 400; }
        .chat-messages {
            flex: 1; overflow-y: auto; padding: 16px;
            display: flex; flex-direction: column; gap: 12px;
        }
        .message { max-width: 80%; padding: 10px 14px; border-radius: 12px;
                    line-height: 1.6; font-size: 14px; word-break: break-word; }
        .message.user { align-self: flex-end; background: #667eea; color: white; }
        .message.assistant { align-self: flex-start; background: #f0f0f0; color: #333; }
        .message .source { font-size: 12px; color: #888; margin-top: 6px;
                           border-top: 1px solid #ddd; padding-top: 4px; }
        .chat-input {
            padding: 12px 16px; border-top: 1px solid #eee;
            display: flex; gap: 8px;
        }
        .chat-input input {
            flex: 1; padding: 10px 14px; border: 1px solid #ddd;
            border-radius: 8px; font-size: 14px; outline: none;
        }
        .chat-input input:focus { border-color: #667eea; }
        .chat-input button {
            padding: 10px 20px; background: #667eea; color: white;
            border: none; border-radius: 8px; cursor: pointer; font-size: 14px;
        }
        .chat-input button:hover { background: #5a6fd6; }
        .typing { color: #999; font-style: italic; font-size: 13px; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            智能客服助手
            <br><small>在线 | 基于RAG+Agent技术</small>
        </div>
        <div class="chat-messages" id="messages">
            <div class="message assistant">
                您好！我是智能客服助手，可以帮您：<br>
                - 查询订单状态<br>
                - 申请退款退货<br>
                - 解答产品问题<br>
                请问有什么可以帮您？
            </div>
        </div>
        <div class="chat-input">
            <input type="text" id="userInput" placeholder="请输入您的问题..."
                   onkeypress="if(event.key==='Enter') sendMessage()">
            <button onclick="sendMessage()">发送</button>
        </div>
    </div>

    <script>
        let sessionId = null;
        const API_BASE = '/api/v1';

        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message) return;

            // 显示用户消息
            appendMessage('user', message);
            input.value = '';

            // 显示"正在输入"
            const typingEl = showTyping();

            try {
                const response = await fetch(`${API_BASE}/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        message: message,
                        user_id: 'web-user'
                    })
                });
                const data = await response.json();
                sessionId = data.session_id;

                // 移除"正在输入"
                typingEl.remove();

                // 显示回复
                let replyHtml = data.reply;
                if (data.sources && data.sources.length > 0) {
                    replyHtml += '<div class="source">来源: ' +
                        data.sources.map(s => s.doc_title).join(', ') + '</div>';
                }
                appendMessage('assistant', replyHtml, true);

                if (data.need_human) {
                    appendMessage('assistant',
                        '正在为您转接人工客服，请稍候...', true);
                }
            } catch (error) {
                typingEl.remove();
                appendMessage('assistant', '抱歉，网络连接异常，请稍后重试。', true);
            }
        }

        function appendMessage(role, content, isHtml = false) {
            const div = document.createElement('div');
            div.className = `message ${role}`;
            if (isHtml) div.innerHTML = content;
            else div.textContent = content;
            document.getElementById('messages').appendChild(div);
            div.scrollIntoView({ behavior: 'smooth' });
        }

        function showTyping() {
            const div = document.createElement('div');
            div.className = 'typing';
            div.textContent = '客服助手正在输入...';
            document.getElementById('messages').appendChild(div);
            div.scrollIntoView({ behavior: 'smooth' });
            return div;
        }
    </script>
</body>
</html>
"""


def setup_frontend_route(app):
    """在FastAPI应用上注册前端路由"""
    from fastapi.responses import HTMLResponse

    @app.get("/", response_class=HTMLResponse)
    async def chat_page():
        return CHAT_HTML_TEMPLATE

    print("[前端] 已注册聊天页面路由: GET /")


# 使用示例：
# setup_frontend_route(app)   # 在 FastAPI app 上注册
```

---

## 部署与监控

### 部署架构

```
+=======================================================================+
|                         生产部署架构                                    |
+=======================================================================+
|                                                                       |
|                          互联网用户                                    |
|                              |                                        |
|                              v                                        |
|                     +--------+--------+                               |
|                     |   CDN / 域名    |                               |
|                     +--------+--------+                               |
|                              |                                        |
|                              v                                        |
|                     +--------+--------+                               |
|                     |     Nginx       |   SSL终止                     |
|                     |  反向代理+负载均衡|   静态文件                    |
|                     +--------+--------+                               |
|                              |                                        |
|              +---------------+---------------+                        |
|              |                               |                        |
|              v                               v                        |
|     +--------+--------+            +--------+--------+               |
|     |  FastAPI 实例1   |            |  FastAPI 实例2  |               |
|     |  (uvicorn:8001)  |            |  (uvicorn:8002) |               |
|     +--------+--------+            +--------+--------+               |
|              |                               |                        |
|              +---------------+---------------+                        |
|                              |                                        |
|         +--------------------+--------------------+                   |
|         |                    |                    |                   |
|         v                    v                    v                   |
|  +------+------+    +-------+------+    +--------+-----+            |
|  |   Redis     |    | PostgreSQL   |    |  ChromaDB    |            |
|  | (会话/缓存) |    | (业务数据)   |    | (向量存储)   |            |
|  +-------------+    +--------------+    +--------------+            |
|                                                                       |
+=======================================================================+
```

### Docker Compose 配置

```python
"""
Docker Compose 配置生成器
生成用于部署智能客服系统的 docker-compose.yml
"""

DOCKER_COMPOSE_YAML = """
# docker-compose.yml - 智能客服系统
version: '3.8'

services:
  # FastAPI 后端服务
  chatbot-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://api.openai.com/v1}
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - DB_URL=postgresql://chatbot:password@postgres:5432/chatbot_db
    depends_on:
      - redis
      - postgres
    volumes:
      - chroma_data:/app/data/chroma_db
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Redis - 会话存储和缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  # PostgreSQL - 业务数据
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: chatbot_db
      POSTGRES_USER: chatbot
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  # Nginx - 反向代理
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      - chatbot-api
    restart: unless-stopped

volumes:
  chroma_data:
  redis_data:
  postgres_data:
"""

DOCKERFILE_CONTENT = """
# Dockerfile - 智能客服系统
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl gcc && \\
    rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 创建数据目录
RUN mkdir -p /app/data/chroma_db /app/logs

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
"""

NGINX_CONF = """
# nginx.conf
upstream chatbot_api {
    server chatbot-api:8000;
}

server {
    listen 80;
    server_name localhost;

    # API 代理
    location /api/ {
        proxy_pass http://chatbot_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # WebSocket 代理
    location /ws/ {
        proxy_pass http://chatbot_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    # 前端页面
    location / {
        proxy_pass http://chatbot_api;
    }
}
"""


def generate_deployment_files(output_dir: str = "."):
    """生成所有部署配置文件"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    files = {
        "docker-compose.yml": DOCKER_COMPOSE_YAML.strip(),
        "Dockerfile": DOCKERFILE_CONTENT.strip(),
        "nginx.conf": NGINX_CONF.strip(),
    }

    for filename, content in files.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  [生成] {filepath}")

    print(f"\n[部署文件就绪] 运行命令: docker compose up -d")


if __name__ == "__main__":
    print("=== 生成部署配置 ===")
    generate_deployment_files("./deploy")
    print()
    print("=== 部署步骤 ===")
    print("1. 设置环境变量: export OPENAI_API_KEY=your-key")
    print("2. 构建并启动:   docker compose up -d --build")
    print("3. 查看日志:     docker compose logs -f chatbot-api")
    print("4. 健康检查:     curl http://localhost/api/v1/health")
    print("5. 打开界面:     浏览器访问 http://localhost")
```

### 监控与日志

```python
"""
监控与日志系统 - 对话质量评估和系统指标监控
"""
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict


@dataclass
class MetricPoint:
    """指标数据点"""
    name: str
    value: float
    timestamp: str = ""
    labels: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ChatbotMetrics:
    """
    客服系统监控指标收集器

    收集的指标:
    - 请求量/QPS
    - 响应延迟
    - 意图分布
    - 工具调用成功率
    - 用户满意度
    - 转人工率
    """

    def __init__(self):
        self.request_count = 0
        self.total_latency = 0.0
        self.intent_counter: Dict[str, int] = defaultdict(int)
        self.tool_call_counter: Dict[str, int] = defaultdict(int)
        self.tool_error_counter: Dict[str, int] = defaultdict(int)
        self.feedback_scores: List[int] = []
        self.human_transfer_count = 0
        self.start_time = time.time()

    def record_request(self, latency_ms: float, intent: str, tool_calls: List[str] = None, need_human: bool = False):
        """记录一次请求"""
        self.request_count += 1
        self.total_latency += latency_ms
        self.intent_counter[intent] += 1
        if need_human:
            self.human_transfer_count += 1
        if tool_calls:
            for tool in tool_calls:
                self.tool_call_counter[tool] += 1

    def record_tool_error(self, tool_name: str):
        """记录工具调用错误"""
        self.tool_error_counter[tool_name] += 1

    def record_feedback(self, score: int):
        """记录用户评分"""
        self.feedback_scores.append(score)

    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        uptime = time.time() - self.start_time
        avg_latency = (self.total_latency / self.request_count) if self.request_count > 0 else 0
        avg_score = (sum(self.feedback_scores) / len(self.feedback_scores)) if self.feedback_scores else 0
        qps = self.request_count / uptime if uptime > 0 else 0

        return {
            "uptime_seconds": round(uptime, 1),
            "total_requests": self.request_count,
            "qps": round(qps, 2),
            "avg_latency_ms": round(avg_latency, 1),
            "intent_distribution": dict(self.intent_counter),
            "tool_calls": dict(self.tool_call_counter),
            "tool_errors": dict(self.tool_error_counter),
            "human_transfer_rate": round(
                self.human_transfer_count / max(self.request_count, 1) * 100, 1
            ),
            "avg_feedback_score": round(avg_score, 2),
            "total_feedbacks": len(self.feedback_scores),
        }

    def print_dashboard(self):
        """打印监控面板"""
        s = self.get_summary()
        print()
        print("+" + "=" * 58 + "+")
        print("|{:^58s}|".format("智能客服系统 - 监控面板"))
        print("+" + "=" * 58 + "+")
        print(f"| {'运行时间':<16s} | {s['uptime_seconds']:.0f} 秒{'':<27s}|")
        print(f"| {'总请求数':<16s} | {s['total_requests']:<33d}|")
        print(f"| {'QPS':<18s} | {s['qps']:<33.2f}|")
        print(f"| {'平均延迟':<16s} | {s['avg_latency_ms']:.1f} ms{'':<26s}|")
        print(f"| {'转人工率':<16s} | {s['human_transfer_rate']:.1f}%{'':<27s}|")
        print(f"| {'平均满意度':<14s} | {s['avg_feedback_score']:.2f} / 5.00{'':<20s}|")
        print("+" + "-" * 58 + "+")
        print(f"| {'意图分布':<16s} |{'':<34s}|")
        for intent, count in s["intent_distribution"].items():
            pct = count / max(s["total_requests"], 1) * 100
            bar = "#" * int(pct / 5)
            print(f"|   {intent:<14s} | {count:>4d} ({pct:>5.1f}%) {bar:<18s}|")
        print("+" + "=" * 58 + "+")


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    metrics = ChatbotMetrics()

    # 模拟一些请求数据
    import random
    intents = ["faq", "order_query", "refund", "chitchat", "complaint"]
    for i in range(100):
        intent = random.choice(intents)
        latency = random.uniform(100, 800)
        tools = ["query_order"] if intent == "order_query" else []
        need_human = intent == "complaint" and random.random() > 0.5
        metrics.record_request(latency, intent, tools, need_human)

    # 模拟反馈
    for _ in range(30):
        metrics.record_feedback(random.choice([3, 4, 4, 5, 5, 5]))

    # 打印面板
    metrics.print_dashboard()
```

---

## 总结

本教程完整实现了一个基于 RAG + Agent 的智能客服系统，涵盖以下核心内容:

1. **项目概述**: 系统整体架构设计、技术栈选择、核心功能规划
2. **系统架构**: 分层架构(接入层/业务层/能力层/数据层)、请求处理流程、数据模型定义
3. **RAG知识库**: 文档加载/分块/向量化、ChromaDB语义检索、RAG问答链
4. **多轮对话**: Redis会话管理、滑动窗口策略、上下文维护
5. **Agent工具**: OpenAI Function Calling、工具定义/注册/执行、安全确认机制
6. **FastAPI后端**: REST API + SSE流式 + WebSocket实时对话
7. **前端界面**: 嵌入式HTML/CSS/JS聊天界面
8. **部署监控**: Docker Compose部署、Nginx反向代理、指标收集与监控面板

## 最佳实践

1. **知识库管理**: 定期更新知识库内容，监控检索命中率，对低质量回答进行人工标注
2. **Prompt工程**: 系统提示词要明确角色、规则和限制，用few-shot示例提升回答质量
3. **安全防护**: 敏感操作二次确认，用户身份验证，输入内容过滤，防止Prompt注入
4. **性能优化**: 向量检索结果缓存，LLM响应流式输出，异步处理非关键任务
5. **持续改进**: 收集用户反馈，分析Bad Case，定期优化Prompt和检索策略

## 参考资源

- [OpenAI Function Calling 文档](https://platform.openai.com/docs/guides/function-calling)
- [ChromaDB 官方文档](https://docs.trychroma.com/)
- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [LangChain RAG 教程](https://python.langchain.com/docs/tutorials/rag/)

---

**创建时间**: 2024-01-01
**最后更新**: 2024-01-01
