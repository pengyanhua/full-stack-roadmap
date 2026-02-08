# 企业文档问答系统 - 完整实战

## 目录
1. [项目概述](#项目概述)
2. [文档处理引擎](#文档处理引擎)
3. [向量检索系统](#向量检索系统)
4. [引用溯源机制](#引用溯源机制)
5. [权限与安全](#权限与安全)
6. [FastAPI完整服务](#fastapi完整服务)
7. [前端问答界面](#前端问答界面)
8. [部署与优化](#部署与优化)

---

## 项目概述

### 项目背景

企业内部积累了大量文档资料(产品手册、技术文档、制度规范、会议纪要等)，
员工查找信息效率低下。企业文档问答系统通过 RAG 技术，让员工用自然语言提问，
系统自动从海量文档中检索相关内容并生成精准回答，同时提供引用来源方便核实。

### 系统整体架构

```
+============================================================================+
|                      企业文档问答系统 - 总体架构                             |
+============================================================================+
|                                                                            |
|   用户层                                                                   |
|   +------------------+  +------------------+  +------------------+         |
|   |  Web问答界面     |  |  飞书/钉钉机器人 |  |  API集成         |         |
|   |  (React前端)     |  |  (Webhook)       |  |  (REST/gRPC)    |         |
|   +--------+---------+  +--------+---------+  +--------+---------+         |
|            |                     |                     |                   |
|            +----------+----------+----------+----------+                   |
|                       |                     |                              |
|                       v                     v                              |
|   服务层                                                                   |
|   +----------------------------------------------------------------+      |
|   |                    FastAPI 后端服务                              |      |
|   |                                                                 |      |
|   |  +-------------+  +-------------+  +-------------+             |      |
|   |  | 问答引擎    |  | 文档管理    |  | 用户权限    |             |      |
|   |  | QA Engine   |  | Doc Manager |  | Auth Guard  |             |      |
|   |  +------+------+  +------+------+  +------+------+             |      |
|   |         |                |                |                     |      |
|   |  +------+------+  +-----+-------+  +-----+-------+            |      |
|   |  | RAG Pipeline|  | 文档解析    |  | JWT认证     |            |      |
|   |  | 检索+生成   |  | PDF/Word/MD |  | RBAC角色    |            |      |
|   |  +------+------+  +------+------+  +-------------+            |      |
|   +---------|-----------------|--------------------------------+    |      |
|             |                 |                                     |      |
|   数据层    |                 |                                     |      |
|   +---------+-----------------+-----------------------------------+|      |
|   |         v                 v                                    ||      |
|   |  +------+------+  +------+------+  +--------------+           ||      |
|   |  |  ChromaDB   |  | PostgreSQL  |  |    Redis     |           ||      |
|   |  |  向量索引   |  | 文档元数据  |  | 查询缓存    |           ||      |
|   |  |  语义检索   |  | 用户权限    |  | 会话状态    |           ||      |
|   |  +-------------+  +-------------+  +--------------+           ||      |
|   +----------------------------------------------------------------+|      |
|                                                                     |      |
+============================================================================+
```

### 核心功能

| 功能模块 | 说明 | 关键技术 |
|---------|------|---------|
| 多格式解析 | 支持PDF、Word、Markdown、TXT、CSV | PyMuPDF、python-docx |
| 智能分块 | 按语义结构分块，保留章节层次 | 递归分块 + 标题感知 |
| 向量检索 | 语义相似度搜索 + 关键词混合检索 | ChromaDB + BM25 |
| 引用溯源 | 回答标注来源文档、页码、原文 | 元数据追踪 |
| 权限控制 | 不同部门只能访问授权文档 | RBAC + 文档标签 |
| 查询缓存 | 相似问题复用答案，降低API调用 | Redis语义缓存 |

### 环境配置

```python
"""
项目配置与依赖管理
"""
import os
from dataclasses import dataclass


@dataclass
class DocQAConfig:
    """文档问答系统配置"""

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # 文档处理
    chunk_size: int = 500
    chunk_overlap: int = 50
    supported_formats: tuple = (".pdf", ".docx", ".doc", ".md", ".txt", ".csv", ".json")
    max_file_size_mb: int = 50
    upload_dir: str = "./data/uploads"

    # 向量检索
    chroma_dir: str = "./data/chroma_db"
    collection_name: str = "enterprise_docs"
    search_top_k: int = 8
    similarity_threshold: float = 0.72
    use_hybrid_search: bool = True       # 启用混合检索(向量+BM25)

    # 缓存
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = 6379
    cache_ttl: int = 3600
    enable_semantic_cache: bool = True

    # 安全
    jwt_secret: str = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24


config = DocQAConfig()

# requirements.txt
REQUIREMENTS = """
fastapi==0.104.1
uvicorn==0.24.0
openai==1.6.1
chromadb==0.4.22
redis==5.0.1
pydantic==2.5.3
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
PyMuPDF==1.23.8
python-docx==1.1.0
pandas==2.1.4
rank-bm25==0.2.2
httpx==0.25.2
"""

if __name__ == "__main__":
    print(f"[配置] 模型: {config.chat_model}")
    print(f"[配置] 向量维度: {config.embedding_dim}")
    print(f"[配置] 支持格式: {config.supported_formats}")
    print(f"[配置] 分块大小: {config.chunk_size}, 重叠: {config.chunk_overlap}")
```

---

## 文档处理引擎

### 文档处理流程

```
+=========================================================================+
|                       文档处理流水线                                      |
+=========================================================================+
|                                                                         |
|   上传文件                                                              |
|      |                                                                  |
|      v                                                                  |
|   +--+-------------+                                                    |
|   | 格式检测       |   <-- 检查文件类型、大小、安全性                   |
|   +--+-------------+                                                    |
|      |                                                                  |
|      +----------+----------+----------+----------+                      |
|      |          |          |          |          |                      |
|      v          v          v          v          v                      |
|   +------+  +------+  +------+  +------+  +------+                    |
|   | PDF  |  | Word |  |  MD  |  | TXT  |  | CSV  |                    |
|   | 解析 |  | 解析 |  | 解析 |  | 解析 |  | 解析 |                    |
|   +--+---+  +--+---+  +--+---+  +--+---+  +--+---+                    |
|      |         |         |         |         |                          |
|      +----+----+----+----+----+----+----+----+                          |
|           |              |              |                               |
|           v              v              v                               |
|      +----+----+    +----+----+    +----+----+                          |
|      | 元数据  |    | 文本    |    | 结构    |                          |
|      | 提取    |    | 清洗    |    | 识别    |                          |
|      | (标题   |    | (去噪   |    | (章节   |                          |
|      |  作者   |    |  规范化)|    |  段落   |                          |
|      |  页码)  |    |         |    |  列表)  |                          |
|      +----+----+    +----+----+    +----+----+                          |
|           |              |              |                               |
|           +-------+------+------+-------+                               |
|                   |             |                                       |
|                   v             v                                       |
|              +----+----+  +----+----+                                   |
|              | 智能分块 |  | 标题层次|                                   |
|              | 递归切分 |  | 感知    |                                   |
|              +----+----+  +----+----+                                   |
|                   |             |                                       |
|                   v             v                                       |
|              +----+-------------+----+                                  |
|              | DocumentChunk 列表    |                                  |
|              | (含元数据+页码+来源)   |                                  |
|              +-----------------------+                                  |
|                                                                         |
+=========================================================================+
```

### 多格式文档解析器

```python
"""
多格式文档解析器 - 支持 PDF、Word、Markdown、TXT、CSV
"""
import os
import re
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class DocumentMeta:
    """文档元信息"""
    file_name: str
    file_path: str
    file_type: str
    file_size: int              # 字节
    title: str = ""
    author: str = ""
    total_pages: int = 0
    created_at: str = ""
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(
                f"{self.file_name}:{self.file_size}".encode()
            ).hexdigest()[:12]
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class DocumentChunk:
    """文档分块"""
    chunk_id: str = ""
    content: str = ""
    doc_id: str = ""
    doc_title: str = ""
    file_name: str = ""
    chunk_index: int = 0
    page_number: int = -1
    section_title: str = ""          # 所属章节标题
    total_chunks: int = 0
    char_count: int = 0
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.char_count = len(self.content)
        if not self.chunk_id:
            h = hashlib.md5(f"{self.doc_id}:{self.chunk_index}:{self.content[:50]}".encode()).hexdigest()[:8]
            self.chunk_id = f"{self.doc_id}_chunk_{self.chunk_index}_{h}"


class TextCleaner:
    """文本清洗工具"""

    @staticmethod
    def clean(text: str) -> str:
        """清洗文本：去除多余空白、特殊字符"""
        # 去除连续空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 去除行首尾空白
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        # 去除控制字符(保留换行和制表符)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        return text.strip()

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """规范化空白字符"""
        text = text.replace('\t', '    ')
        text = re.sub(r' {4,}', '  ', text)
        return text


class PDFParser:
    """
    PDF 文档解析器

    使用 PyMuPDF (fitz) 提取文本和元信息
    支持按页提取、保留页码信息
    """

    def parse(self, file_path: str) -> Tuple[DocumentMeta, List[Dict]]:
        """
        解析PDF文件

        返回: (文档元信息, 按页的文本列表)
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("请安装PyMuPDF: pip install PyMuPDF")

        doc = fitz.open(file_path)
        meta = DocumentMeta(
            file_name=os.path.basename(file_path),
            file_path=file_path,
            file_type="pdf",
            file_size=os.path.getsize(file_path),
            title=doc.metadata.get("title", "") or os.path.basename(file_path),
            author=doc.metadata.get("author", ""),
            total_pages=len(doc),
        )

        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            cleaned = TextCleaner.clean(text)
            if cleaned:
                pages.append({
                    "page_number": page_num + 1,
                    "text": cleaned,
                    "char_count": len(cleaned),
                })

        doc.close()
        print(f"  [PDF] {meta.file_name}: {meta.total_pages} 页, {len(pages)} 页有内容")
        return meta, pages


class WordParser:
    """
    Word 文档解析器

    使用 python-docx 提取段落、表格
    """

    def parse(self, file_path: str) -> Tuple[DocumentMeta, List[Dict]]:
        try:
            from docx import Document
        except ImportError:
            raise ImportError("请安装python-docx: pip install python-docx")

        doc = Document(file_path)
        meta = DocumentMeta(
            file_name=os.path.basename(file_path),
            file_path=file_path,
            file_type="docx",
            file_size=os.path.getsize(file_path),
            title=doc.core_properties.title or os.path.basename(file_path),
            author=doc.core_properties.author or "",
        )

        paragraphs = []
        current_section = ""

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            # 识别标题(Heading样式)
            if para.style and para.style.name.startswith("Heading"):
                current_section = text

            paragraphs.append({
                "text": text,
                "section": current_section,
                "style": para.style.name if para.style else "Normal",
            })

        # 提取表格内容
        for table in doc.tables:
            rows_text = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows_text.append(" | ".join(cells))
            if rows_text:
                table_text = "\n".join(rows_text)
                paragraphs.append({
                    "text": table_text,
                    "section": current_section,
                    "style": "Table",
                })

        print(f"  [Word] {meta.file_name}: {len(paragraphs)} 个段落/表格")
        return meta, paragraphs


class MarkdownParser:
    """
    Markdown 文档解析器

    按标题层次切分，保留章节结构
    """

    def parse(self, file_path: str) -> Tuple[DocumentMeta, List[Dict]]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        meta = DocumentMeta(
            file_name=os.path.basename(file_path),
            file_path=file_path,
            file_type="md",
            file_size=os.path.getsize(file_path),
        )

        # 按标题切分
        sections = []
        current_title = ""
        current_level = 0
        current_text_lines = []

        for line in content.split("\n"):
            heading_match = re.match(r'^(#{1,6})\s+(.+)', line)
            if heading_match:
                # 保存上一个section
                if current_text_lines:
                    sections.append({
                        "title": current_title,
                        "level": current_level,
                        "text": "\n".join(current_text_lines).strip(),
                    })
                current_title = heading_match.group(2)
                current_level = len(heading_match.group(1))
                current_text_lines = [line]
            else:
                current_text_lines.append(line)

        # 最后一个section
        if current_text_lines:
            sections.append({
                "title": current_title,
                "level": current_level,
                "text": "\n".join(current_text_lines).strip(),
            })

        # 提取文档标题
        if sections and sections[0].get("level") == 1:
            meta.title = sections[0]["title"]
        else:
            meta.title = os.path.basename(file_path)

        print(f"  [Markdown] {meta.file_name}: {len(sections)} 个章节")
        return meta, sections


class PlainTextParser:
    """纯文本解析器"""

    def parse(self, file_path: str) -> Tuple[DocumentMeta, List[Dict]]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        meta = DocumentMeta(
            file_name=os.path.basename(file_path),
            file_path=file_path,
            file_type="txt",
            file_size=os.path.getsize(file_path),
            title=os.path.basename(file_path),
        )

        cleaned = TextCleaner.clean(content)
        paragraphs = [
            {"text": p.strip(), "section": ""}
            for p in cleaned.split("\n\n")
            if p.strip()
        ]

        print(f"  [TXT] {meta.file_name}: {len(paragraphs)} 个段落")
        return meta, paragraphs


class CSVParser:
    """CSV文件解析器 - 按行转为文本"""

    def parse(self, file_path: str) -> Tuple[DocumentMeta, List[Dict]]:
        import csv

        meta = DocumentMeta(
            file_name=os.path.basename(file_path),
            file_path=file_path,
            file_type="csv",
            file_size=os.path.getsize(file_path),
            title=os.path.basename(file_path),
        )

        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            for row in reader:
                row_text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
                rows.append({"text": row_text, "section": "", "headers": headers})

        print(f"  [CSV] {meta.file_name}: {len(rows)} 行数据")
        return meta, rows


# ============================================================
# 智能文本分块器
# ============================================================

class SmartChunker:
    """
    智能文本分块器

    特点:
    1. 按语义边界分块(段落 > 句子 > 词)
    2. 保留章节标题信息
    3. 支持重叠，确保上下文连续
    4. 自动跳过过短的分块
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, min_chunk_size: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.separators = ["\n\n", "\n", "。", "；", "！", "？", ".", "!", "?", " "]

    def chunk_text(self, text: str, section_title: str = "") -> List[str]:
        """将文本分成多个块"""
        if len(text) <= self.chunk_size:
            return [text] if len(text) >= self.min_chunk_size else []

        chunks = []
        # 找到合适的分隔符
        separator = ""
        for sep in self.separators:
            if sep in text:
                separator = sep
                break

        # 按分隔符切分
        parts = text.split(separator) if separator else [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        current = ""
        for part in parts:
            piece = part + separator if separator else part
            if len(current) + len(piece) <= self.chunk_size:
                current += piece
            else:
                if len(current.strip()) >= self.min_chunk_size:
                    # 添加章节标题前缀
                    if section_title:
                        chunk_text = f"[{section_title}] {current.strip()}"
                    else:
                        chunk_text = current.strip()
                    chunks.append(chunk_text)

                # 保留overlap
                if self.chunk_overlap > 0 and len(current) > self.chunk_overlap:
                    current = current[-self.chunk_overlap:] + piece
                else:
                    current = piece

        # 处理最后一块
        if len(current.strip()) >= self.min_chunk_size:
            if section_title:
                chunks.append(f"[{section_title}] {current.strip()}")
            else:
                chunks.append(current.strip())

        return chunks


# ============================================================
# 统一文档处理器
# ============================================================

class DocumentProcessor:
    """
    统一文档处理器

    整合多格式解析 + 智能分块，输出标准化的 DocumentChunk 列表
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunker = SmartChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.parsers = {
            ".pdf": PDFParser(),
            ".docx": WordParser(),
            ".doc": WordParser(),
            ".md": MarkdownParser(),
            ".txt": PlainTextParser(),
            ".csv": CSVParser(),
        }
        self.supported_formats = set(self.parsers.keys())

    def process_file(self, file_path: str, department: str = "") -> Tuple[DocumentMeta, List[DocumentChunk]]:
        """
        处理单个文件

        返回: (文档元信息, 分块列表)
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_formats:
            raise ValueError(f"不支持的文件格式: {ext}，支持: {self.supported_formats}")

        parser = self.parsers[ext]
        meta, raw_sections = parser.parse(file_path)

        # 合并所有文本并分块
        chunks: List[DocumentChunk] = []
        chunk_index = 0

        for section in raw_sections:
            text = section.get("text", "")
            section_title = section.get("title", section.get("section", ""))
            page_num = section.get("page_number", -1)

            # 分块
            chunk_texts = self.chunker.chunk_text(text, section_title)

            for ct in chunk_texts:
                chunk = DocumentChunk(
                    content=ct,
                    doc_id=meta.doc_id,
                    doc_title=meta.title,
                    file_name=meta.file_name,
                    chunk_index=chunk_index,
                    page_number=page_num,
                    section_title=section_title,
                    metadata={
                        "file_type": meta.file_type,
                        "department": department,
                        "author": meta.author,
                    },
                )
                chunks.append(chunk)
                chunk_index += 1

        # 回填total_chunks
        for c in chunks:
            c.total_chunks = len(chunks)

        print(f"[处理完成] {meta.file_name}: {len(chunks)} 个分块, 平均 {sum(c.char_count for c in chunks)//max(len(chunks),1)} 字/块")
        return meta, chunks

    def process_directory(self, dir_path: str, department: str = "") -> Tuple[List[DocumentMeta], List[DocumentChunk]]:
        """批量处理目录下所有文档"""
        all_metas: List[DocumentMeta] = []
        all_chunks: List[DocumentChunk] = []

        for root, _dirs, files in os.walk(dir_path):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self.supported_formats:
                    continue
                fpath = os.path.join(root, fname)
                try:
                    meta, chunks = self.process_file(fpath, department)
                    all_metas.append(meta)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"  [错误] {fname}: {e}")

        print(f"\n[批量处理完成] {len(all_metas)} 个文档, {len(all_chunks)} 个分块")
        return all_metas, all_chunks


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    # 测试智能分块
    chunker = SmartChunker(chunk_size=200, chunk_overlap=30)
    sample = """
    第一章 产品概述

    本产品是一款企业级文档管理系统，支持多种文件格式的上传、检索和管理。
    系统采用微服务架构，前端使用React，后端使用FastAPI，数据库使用PostgreSQL。

    第二章 安装部署

    1. 环境要求：Python 3.10+，Node.js 18+，PostgreSQL 15+
    2. 安装步骤：克隆代码，安装依赖，初始化数据库，启动服务
    3. 配置说明：修改config.yaml中的数据库连接、API密钥等配置项
    """

    chunks = chunker.chunk_text(sample.strip(), "产品手册")
    for i, c in enumerate(chunks):
        print(f"--- 分块 {i+1} ({len(c)} 字) ---")
        print(c[:120] + "..." if len(c) > 120 else c)
        print()

    # 测试文档处理器
    print("=== 文档处理器 ===")
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=30)
    print(f"支持格式: {processor.supported_formats}")
```

---

## 向量检索系统

### 检索架构

```
+=========================================================================+
|                     混合检索系统架构                                      |
+=========================================================================+
|                                                                         |
|  用户查询: "员工入职需要准备什么材料？"                                 |
|       |                                                                 |
|       v                                                                 |
|  +----+-----------------------------+                                   |
|  |        查询预处理                 |                                   |
|  |  - 纠错/同义词扩展               |                                   |
|  |  - 关键词提取                     |                                   |
|  |  - 查询改写(Query Rewrite)       |                                   |
|  +----+----------+------------------+                                   |
|       |          |                                                      |
|       v          v                                                      |
|  +----+----+ +---+------+                                               |
|  | 向量检索 | | BM25检索 |     <-- 双路检索                              |
|  | (语义)   | | (关键词) |                                               |
|  +----+----+ +---+------+                                               |
|       |          |                                                      |
|       v          v                                                      |
|  +----+----------+-------+                                              |
|  |     结果融合            |    <-- RRF (Reciprocal Rank Fusion)         |
|  |  rank_score =           |                                             |
|  |    a * semantic_score + |                                             |
|  |    b * bm25_score       |                                             |
|  +----+-------------------+                                              |
|       |                                                                 |
|       v                                                                 |
|  +----+-------------------+                                              |
|  |     权限过滤            |    <-- 按用户部门过滤文档                   |
|  +----+-------------------+                                              |
|       |                                                                 |
|       v                                                                 |
|  +----+-------------------+                                              |
|  |     重排序 (Reranker)   |    <-- 对Top-K结果精排                      |
|  +----+-------------------+                                              |
|       |                                                                 |
|       v                                                                 |
|  返回 Top-N 最相关的文档分块                                            |
|                                                                         |
+=========================================================================+
```

### 混合检索引擎

```python
"""
混合检索引擎 - 向量语义检索 + BM25关键词检索
"""
import os
import math
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import chromadb
    from openai import OpenAI
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@dataclass
class SearchResult:
    """检索结果"""
    chunk_id: str
    content: str
    doc_title: str
    file_name: str
    page_number: int = -1
    section_title: str = ""
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    final_score: float = 0.0
    metadata: Dict = field(default_factory=dict)


class BM25Index:
    """
    BM25 关键词检索索引

    BM25是信息检索中经典的排序算法，基于词频(TF)和逆文档频率(IDF)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_doc_len = 0
        self.doc_lengths: Dict[str, int] = {}
        self.doc_term_freqs: Dict[str, Dict[str, int]] = {}
        self.idf_cache: Dict[str, float] = {}
        self.term_doc_count: Dict[str, int] = defaultdict(int)
        self.documents: Dict[str, str] = {}

    def _tokenize(self, text: str) -> List[str]:
        """简单分词(中文按字，英文按空格)"""
        import re
        # 提取中文字符和英文单词
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text.lower())
        return tokens

    def add_documents(self, doc_id_text_pairs: List[Tuple[str, str]]):
        """批量添加文档到索引"""
        for doc_id, text in doc_id_text_pairs:
            tokens = self._tokenize(text)
            self.doc_lengths[doc_id] = len(tokens)
            self.documents[doc_id] = text

            # 统计词频
            term_freq: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            self.doc_term_freqs[doc_id] = dict(term_freq)

            # 统计包含某词的文档数
            for term in set(tokens):
                self.term_doc_count[term] += 1

        self.doc_count = len(self.documents)
        self.avg_doc_len = sum(self.doc_lengths.values()) / max(self.doc_count, 1)

        # 计算IDF
        for term, df in self.term_doc_count.items():
            self.idf_cache[term] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

        print(f"[BM25] 索引构建完成: {self.doc_count} 篇文档, {len(self.term_doc_count)} 个词项")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """BM25检索"""
        query_tokens = self._tokenize(query)
        scores: Dict[str, float] = defaultdict(float)

        for token in query_tokens:
            if token not in self.idf_cache:
                continue
            idf = self.idf_cache[token]

            for doc_id, term_freqs in self.doc_term_freqs.items():
                if token not in term_freqs:
                    continue
                tf = term_freqs[token]
                doc_len = self.doc_lengths[doc_id]
                # BM25公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                scores[doc_id] += idf * numerator / denominator

        # 排序取Top-K
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


class HybridSearchEngine:
    """
    混合检索引擎

    结合向量语义检索和BM25关键词检索，提供更全面的检索结果

    融合策略: RRF (Reciprocal Rank Fusion)
    score = alpha * semantic_rank_score + (1-alpha) * bm25_rank_score
    """

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "enterprise_docs",
        embedding_model: str = "text-embedding-3-small",
        alpha: float = 0.6,             # 向量检索权重
    ):
        self.alpha = alpha
        self.bm25_index = BM25Index()
        self.chunk_store: Dict[str, Dict] = {}  # chunk_id -> chunk_data

        if HAS_DEPS:
            self.openai_client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", "your-key"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )
            os.makedirs(persist_dir, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"[混合检索引擎] ChromaDB集合: {collection_name}")
        else:
            self.openai_client = None
            self.collection = None
            print("[混合检索引擎] 仅BM25模式(缺少chromadb/openai)")

        self.embedding_model = embedding_model

    def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        """获取文本向量"""
        if not self.openai_client:
            return [[0.0] * 1536 for _ in texts]
        resp = self.openai_client.embeddings.create(model=self.embedding_model, input=texts)
        return [item.embedding for item in resp.data]

    def index_chunks(self, chunks: list, batch_size: int = 50):
        """将文档分块添加到索引"""
        # 添加到BM25
        bm25_pairs = [(c.chunk_id, c.content) for c in chunks]
        self.bm25_index.add_documents(bm25_pairs)

        # 存储chunk数据
        for c in chunks:
            self.chunk_store[c.chunk_id] = {
                "content": c.content,
                "doc_title": c.doc_title,
                "file_name": c.file_name,
                "page_number": c.page_number,
                "section_title": c.section_title,
                "metadata": c.metadata,
            }

        # 添加到向量库
        if self.collection is not None:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                texts = [c.content for c in batch]
                ids = [c.chunk_id for c in batch]
                metadatas = [
                    {
                        "doc_title": c.doc_title,
                        "file_name": c.file_name,
                        "page_number": c.page_number,
                        "section_title": c.section_title,
                        "department": c.metadata.get("department", ""),
                    }
                    for c in batch
                ]
                embeddings = self._get_embedding(texts)
                self.collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

        print(f"[索引完成] {len(chunks)} 个分块已索引")

    def search(
        self,
        query: str,
        top_k: int = 8,
        department_filter: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """
        混合检索

        参数:
            query: 用户查询
            top_k: 返回结果数
            department_filter: 部门过滤
            score_threshold: 最低分数阈值
        """
        results_map: Dict[str, SearchResult] = {}

        # 1. BM25检索
        bm25_results = self.bm25_index.search(query, top_k=top_k * 2)
        for rank, (chunk_id, score) in enumerate(bm25_results):
            rrf_score = 1.0 / (60 + rank + 1)
            data = self.chunk_store.get(chunk_id, {})
            results_map[chunk_id] = SearchResult(
                chunk_id=chunk_id,
                content=data.get("content", ""),
                doc_title=data.get("doc_title", ""),
                file_name=data.get("file_name", ""),
                page_number=data.get("page_number", -1),
                section_title=data.get("section_title", ""),
                bm25_score=rrf_score,
                metadata=data.get("metadata", {}),
            )

        # 2. 向量检索
        if self.collection is not None:
            query_emb = self._get_embedding([query])[0]
            search_kwargs = {
                "query_embeddings": [query_emb],
                "n_results": top_k * 2,
                "include": ["documents", "metadatas", "distances"],
            }
            if department_filter:
                search_kwargs["where"] = {"department": department_filter}

            chroma_results = self.collection.query(**search_kwargs)

            if chroma_results and chroma_results["ids"][0]:
                for rank, (cid, doc, meta, dist) in enumerate(zip(
                    chroma_results["ids"][0],
                    chroma_results["documents"][0],
                    chroma_results["metadatas"][0],
                    chroma_results["distances"][0],
                )):
                    similarity = 1 - dist / 2
                    rrf_score = 1.0 / (60 + rank + 1)

                    if cid in results_map:
                        results_map[cid].semantic_score = rrf_score
                    else:
                        results_map[cid] = SearchResult(
                            chunk_id=cid,
                            content=doc,
                            doc_title=meta.get("doc_title", ""),
                            file_name=meta.get("file_name", ""),
                            page_number=meta.get("page_number", -1),
                            section_title=meta.get("section_title", ""),
                            semantic_score=rrf_score,
                            metadata=meta,
                        )

        # 3. 融合分数
        for r in results_map.values():
            r.final_score = self.alpha * r.semantic_score + (1 - self.alpha) * r.bm25_score

        # 4. 部门权限过滤(BM25结果也要过滤)
        if department_filter:
            results_map = {
                k: v for k, v in results_map.items()
                if v.metadata.get("department", "") in ("", department_filter)
            }

        # 5. 排序取Top-K
        sorted_results = sorted(results_map.values(), key=lambda x: x.final_score, reverse=True)
        final_results = [r for r in sorted_results[:top_k] if r.final_score >= score_threshold]

        return final_results


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    print("=== 混合检索引擎 测试 ===\n")

    # 创建引擎(无API模式)
    engine = HybridSearchEngine()

    # 模拟文档分块
    class MockChunk:
        def __init__(self, chunk_id, content, doc_title, file_name, page_number=-1, section_title="", metadata=None):
            self.chunk_id = chunk_id
            self.content = content
            self.doc_title = doc_title
            self.file_name = file_name
            self.page_number = page_number
            self.section_title = section_title
            self.metadata = metadata or {}

    mock_chunks = [
        MockChunk("c1", "新员工入职需要准备身份证、学历证明、体检报告、离职证明等材料。", "入职指南", "onboarding.md"),
        MockChunk("c2", "试用期为3个月，转正需要完成部门考核和HR面谈。", "入职指南", "onboarding.md"),
        MockChunk("c3", "年假制度：工作满1年享5天，满10年享10天，满20年享15天。", "假期制度", "leave_policy.md"),
        MockChunk("c4", "报销流程：填写报销单 -> 部门审批 -> 财务审核 -> 打款。", "报销制度", "expense.md"),
        MockChunk("c5", "公司提供五险一金，另有补充医疗保险和商业意外险。", "福利制度", "benefits.md"),
    ]

    engine.index_chunks(mock_chunks)

    # 测试检索
    print("\n--- 检索测试 ---")
    queries = ["入职要带什么", "年假有几天", "怎么报销"]
    for q in queries:
        results = engine.search(q, top_k=3)
        print(f"\n查询: {q}")
        for r in results:
            print(f"  [{r.final_score:.4f}] {r.doc_title}: {r.content[:60]}...")
```

---

## 引用溯源机制

### 引用溯源架构

```
+=========================================================================+
|                        引用溯源系统                                      |
+=========================================================================+
|                                                                         |
|  用户问题: "报销流程是什么？"                                           |
|       |                                                                 |
|       v                                                                 |
|  +----+------+                                                          |
|  | 混合检索  | ---> 返回 Top-K 分块(含元数据)                           |
|  +----+------+                                                          |
|       |                                                                 |
|       v                                                                 |
|  +----+----------------------------------------------+                  |
|  |              LLM 生成回答                          |                  |
|  |                                                    |                  |
|  |  Prompt:                                           |                  |
|  |  "基于以下参考资料回答问题。                        |                  |
|  |   请在回答中用 [来源X] 标注引用。                   |                  |
|  |                                                    |                  |
|  |   [来源1] 报销制度.md (第2页)                       |                  |
|  |   报销流程：填写报销单 -> 部门审批 -> ...           |                  |
|  |                                                    |                  |
|  |   [来源2] 财务规范.md (第5页)                       |                  |
|  |   报销金额超过5000元需要总监审批..."                 |                  |
|  +----+----------------------------------------------+                  |
|       |                                                                 |
|       v                                                                 |
|  +----+----------------------------------------------+                  |
|  |  AI回答:                                           |                  |
|  |  "报销流程如下:                                     |                  |
|  |   1. 填写报销单 [来源1]                             |                  |
|  |   2. 部门审批 [来源1]                               |                  |
|  |   3. 超过5000元需总监审批 [来源2]                   |                  |
|  |   4. 财务审核 -> 打款 [来源1]"                      |                  |
|  +----+----------------------------------------------+                  |
|       |                                                                 |
|       v                                                                 |
|  +----+----------------------------------------------+                  |
|  |  引用解析器                                        |                  |
|  |  提取 [来源X] 标记 -> 映射到原始文档信息            |                  |
|  |  输出: [{doc, page, text, score}, ...]             |                  |
|  +---------------------------------------------------+                  |
|                                                                         |
+=========================================================================+
```

### 引用溯源实现

```python
"""
引用溯源系统 - 生成带引用标注的回答，并解析引用来源
"""
import re
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class Citation:
    """引用信息"""
    citation_id: int               # 引用编号 [1], [2], ...
    doc_title: str                 # 文档标题
    file_name: str                 # 文件名
    page_number: int = -1          # 页码
    section_title: str = ""        # 章节
    original_text: str = ""        # 原文片段
    similarity_score: float = 0.0  # 相关度


@dataclass
class AnswerWithCitations:
    """带引用的回答"""
    answer: str                    # 回答文本(含 [来源X] 标记)
    citations: List[Citation] = field(default_factory=list)  # 引用列表
    confidence: float = 0.0        # 置信度
    query: str = ""                # 原始问题


class CitationRAGEngine:
    """
    引用溯源 RAG 引擎

    核心功能:
    1. 构建包含来源编号的Prompt
    2. 要求LLM在回答中标注 [来源X]
    3. 解析回答中的引用标记
    4. 映射到原始文档信息
    """

    SYSTEM_PROMPT = """你是一个企业知识库问答助手。请基于提供的参考资料准确回答问题。

重要规则:
1. 只使用参考资料中的信息，不要编造
2. 在回答中用 [来源X] 标注你引用了哪个参考资料(X是编号)
3. 如果参考资料不足以回答，诚实说明"未找到相关信息"
4. 回答要简洁、有条理、使用中文
5. 如有多个来源涉及，分别标注"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        if HAS_OPENAI:
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", "your-key"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )
        else:
            self.client = None

    def build_prompt_with_sources(self, query: str, search_results: list) -> Tuple[str, Dict[int, Dict]]:
        """
        构建带来源编号的Prompt

        返回: (prompt文本, 来源映射 {编号: 文档信息})
        """
        source_map: Dict[int, Dict] = {}
        sources_text = ""

        for i, result in enumerate(search_results, 1):
            source_info = {
                "doc_title": getattr(result, "doc_title", "未知"),
                "file_name": getattr(result, "file_name", ""),
                "page_number": getattr(result, "page_number", -1),
                "section_title": getattr(result, "section_title", ""),
                "content": getattr(result, "content", ""),
                "score": getattr(result, "final_score", 0.0),
            }
            source_map[i] = source_info

            page_info = f" (第{source_info['page_number']}页)" if source_info['page_number'] > 0 else ""
            section_info = f" [{source_info['section_title']}]" if source_info['section_title'] else ""

            sources_text += f"\n[来源{i}] {source_info['doc_title']}{page_info}{section_info}\n"
            sources_text += f"{source_info['content']}\n"

        prompt = f"""请基于以下参考资料回答用户问题。请在回答中用 [来源X] 标注引用。

===== 参考资料 =====
{sources_text}
===== 参考资料结束 =====

用户问题: {query}

请回答:"""

        return prompt, source_map

    def answer_with_citations(
        self,
        query: str,
        search_results: list,
        chat_history: Optional[List[Dict]] = None,
    ) -> AnswerWithCitations:
        """
        生成带引用的回答

        参数:
            query: 用户问题
            search_results: 检索结果列表
            chat_history: 对话历史
        """
        if not search_results:
            return AnswerWithCitations(
                answer="抱歉，在知识库中未找到与您问题相关的信息。请换个方式描述或联系管理员。",
                query=query,
                confidence=0.0,
            )

        # 1. 构建Prompt
        prompt, source_map = self.build_prompt_with_sources(query, search_results)

        # 2. 调用LLM (或使用模拟)
        if self.client:
            messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
            if chat_history:
                messages.extend(chat_history[-6:])
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=1500,
            )
            answer_text = response.choices[0].message.content
        else:
            # 模拟回答
            answer_text = self._mock_answer(query, source_map)

        # 3. 解析引用标记
        citations = self._parse_citations(answer_text, source_map)

        # 4. 计算置信度
        confidence = self._calculate_confidence(search_results, citations)

        return AnswerWithCitations(
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            query=query,
        )

    def _parse_citations(self, answer: str, source_map: Dict[int, Dict]) -> List[Citation]:
        """解析回答中的 [来源X] 引用标记"""
        citation_ids = set()
        # 匹配 [来源1], [来源2] 等
        matches = re.findall(r'\[来源(\d+)\]', answer)
        for m in matches:
            citation_ids.add(int(m))

        citations = []
        for cid in sorted(citation_ids):
            if cid in source_map:
                info = source_map[cid]
                citations.append(Citation(
                    citation_id=cid,
                    doc_title=info["doc_title"],
                    file_name=info["file_name"],
                    page_number=info["page_number"],
                    section_title=info["section_title"],
                    original_text=info["content"][:200],
                    similarity_score=info["score"],
                ))

        return citations

    def _calculate_confidence(self, search_results: list, citations: list) -> float:
        """计算回答置信度"""
        if not search_results:
            return 0.0

        # 基于检索分数和引用数量
        max_score = max(getattr(r, "final_score", 0.0) for r in search_results)
        citation_ratio = len(citations) / max(len(search_results), 1)

        confidence = 0.6 * min(max_score * 5, 1.0) + 0.4 * min(citation_ratio, 1.0)
        return round(min(confidence, 1.0), 3)

    def _mock_answer(self, query: str, source_map: Dict) -> str:
        """无API时的模拟回答"""
        if not source_map:
            return "未找到相关信息。"

        answer_parts = [f"关于您的问题「{query}」，以下是相关信息:\n"]
        for sid, info in list(source_map.items())[:3]:
            content = info["content"][:100]
            answer_parts.append(f"- {content} [来源{sid}]")

        return "\n".join(answer_parts)

    def format_answer_display(self, result: AnswerWithCitations) -> str:
        """格式化输出带引用的回答(用于终端展示)"""
        lines = []
        lines.append(f"问题: {result.query}")
        lines.append(f"置信度: {result.confidence:.1%}")
        lines.append("-" * 50)
        lines.append(f"回答:\n{result.answer}")
        lines.append("-" * 50)

        if result.citations:
            lines.append("引用来源:")
            for c in result.citations:
                page = f" 第{c.page_number}页" if c.page_number > 0 else ""
                lines.append(f"  [{c.citation_id}] {c.doc_title}{page}")
                lines.append(f"      原文: {c.original_text[:80]}...")
        else:
            lines.append("(无引用来源)")

        return "\n".join(lines)


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    print("=== 引用溯源 RAG 测试 ===\n")

    engine = CitationRAGEngine()

    # 模拟检索结果
    class MockResult:
        def __init__(self, content, doc_title, file_name, page_number=-1, section_title="", final_score=0.85):
            self.content = content
            self.doc_title = doc_title
            self.file_name = file_name
            self.page_number = page_number
            self.section_title = section_title
            self.final_score = final_score

    results = [
        MockResult("报销流程：1.填写报销单 2.部门审批 3.财务审核 4.打款到账", "报销制度", "expense.md", 2, "报销流程"),
        MockResult("报销金额超过5000元需要总监审批，超过10000元需要VP审批", "报销制度", "expense.md", 3, "审批权限"),
        MockResult("差旅报销需附上行程单、发票原件、出差审批单", "差旅管理", "travel.md", 1, "报销材料"),
    ]

    answer_result = engine.answer_with_citations("公司报销流程是什么？", results)
    print(engine.format_answer_display(answer_result))
```

---

## 权限与安全

### 权限模型

```
+=========================================================================+
|                         权限控制模型                                      |
+=========================================================================+
|                                                                         |
|  用户 (User)          角色 (Role)          文档标签 (DocTag)            |
|  +----------+         +----------+         +----------+                 |
|  | 张三     | ------> | 技术部   | ------> | tech     |                 |
|  | user-001 |    M:N  | 员工     |    N:M  | public   |                 |
|  +----------+         +----------+         +----------+                 |
|  | 李四     | ------> | HR部     | ------> | hr       |                 |
|  | user-002 |         | 管理员   |         | policy   |                 |
|  +----------+         +----------+         | finance  |                 |
|  | 王五     | ------> | 财务部   | ------> +----------+                 |
|  | user-003 |         | 员工     |                                      |
|  +----------+         +----------+                                      |
|                                                                         |
|  权限检查流程:                                                          |
|  1. 用户登录 -> JWT令牌(含user_id, roles)                               |
|  2. 查询请求 -> 提取用户角色 -> 获取可访问的文档标签                    |
|  3. 向量检索时 -> 按文档标签过滤 -> 只返回授权文档的结果               |
|                                                                         |
+=========================================================================+
```

### 认证与授权实现

```python
"""
权限管理系统 - JWT认证 + RBAC角色权限
"""
import os
import hashlib
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta

try:
    from jose import jwt, JWTError
    HAS_JOSE = True
except ImportError:
    HAS_JOSE = False
    print("[警告] 请安装: pip install python-jose[cryptography]")


JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-for-doc-qa")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24


@dataclass
class User:
    """用户信息"""
    user_id: str
    username: str
    password_hash: str
    roles: List[str] = field(default_factory=list)         # 角色列表
    department: str = ""
    is_active: bool = True


@dataclass
class Role:
    """角色定义"""
    role_name: str
    doc_tags: List[str] = field(default_factory=list)       # 可访问的文档标签
    description: str = ""


# ============================================================
# 模拟用户和角色数据库
# ============================================================
ROLES_DB: Dict[str, Role] = {
    "admin": Role("admin", ["public", "tech", "hr", "finance", "confidential"], "管理员-全部权限"),
    "tech_staff": Role("tech_staff", ["public", "tech"], "技术部员工"),
    "hr_staff": Role("hr_staff", ["public", "hr", "policy"], "HR部员工"),
    "finance_staff": Role("finance_staff", ["public", "finance"], "财务部员工"),
    "general": Role("general", ["public"], "普通员工-仅公开文档"),
}

USERS_DB: Dict[str, User] = {
    "admin": User("u-admin", "admin", hashlib.sha256(b"admin123").hexdigest(), ["admin"], "管理部"),
    "zhangsan": User("u-001", "zhangsan", hashlib.sha256(b"pass123").hexdigest(), ["tech_staff"], "技术部"),
    "lisi": User("u-002", "lisi", hashlib.sha256(b"pass123").hexdigest(), ["hr_staff"], "HR部"),
    "wangwu": User("u-003", "wangwu", hashlib.sha256(b"pass123").hexdigest(), ["general"], "市场部"),
}


class AuthService:
    """认证服务"""

    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        return hashlib.sha256(password.encode()).hexdigest() == hashed

    @staticmethod
    def create_token(user: User) -> str:
        """创建JWT令牌"""
        if not HAS_JOSE:
            return f"mock-token-{user.user_id}"

        payload = {
            "sub": user.user_id,
            "username": user.username,
            "roles": user.roles,
            "department": user.department,
            "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS),
            "iat": datetime.utcnow(),
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    @staticmethod
    def verify_token(token: str) -> Optional[Dict]:
        """验证JWT令牌并返回payload"""
        if not HAS_JOSE:
            return {"sub": "mock-user", "roles": ["general"]}

        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return payload
        except JWTError:
            return None

    @staticmethod
    def login(username: str, password: str) -> Optional[str]:
        """登录并返回JWT令牌"""
        user = USERS_DB.get(username)
        if not user:
            return None
        if not AuthService.verify_password(password, user.password_hash):
            return None
        if not user.is_active:
            return None

        token = AuthService.create_token(user)
        print(f"[登录成功] {username}, 角色: {user.roles}")
        return token


class PermissionService:
    """权限服务"""

    @staticmethod
    def get_accessible_tags(roles: List[str]) -> Set[str]:
        """根据角色列表获取所有可访问的文档标签"""
        tags: Set[str] = set()
        for role_name in roles:
            role = ROLES_DB.get(role_name)
            if role:
                tags.update(role.doc_tags)
        return tags

    @staticmethod
    def can_access_document(user_roles: List[str], doc_tag: str) -> bool:
        """检查用户是否有权限访问某个文档标签"""
        accessible = PermissionService.get_accessible_tags(user_roles)
        return doc_tag in accessible or "admin" in user_roles

    @staticmethod
    def filter_search_results(results: list, user_roles: List[str]) -> list:
        """过滤搜索结果，只保留用户有权限的文档"""
        accessible_tags = PermissionService.get_accessible_tags(user_roles)

        filtered = []
        for r in results:
            doc_tag = getattr(r, "metadata", {}).get("department", "public")
            if doc_tag in accessible_tags or not doc_tag:
                filtered.append(r)

        return filtered


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    print("=== 权限系统测试 ===\n")

    # 1. 用户登录
    token = AuthService.login("zhangsan", "pass123")
    print(f"JWT令牌: {token[:50]}...")

    # 2. 验证令牌
    payload = AuthService.verify_token(token)
    print(f"令牌载荷: user={payload.get('username')}, roles={payload.get('roles')}")

    # 3. 查看可访问标签
    roles = payload.get("roles", [])
    tags = PermissionService.get_accessible_tags(roles)
    print(f"可访问文档标签: {tags}")

    # 4. 权限检查
    checks = [
        ("tech", "技术文档"),
        ("hr", "HR文档"),
        ("finance", "财务文档"),
        ("public", "公开文档"),
    ]
    print("\n权限检查:")
    for tag, desc in checks:
        ok = PermissionService.can_access_document(roles, tag)
        status = "允许" if ok else "拒绝"
        print(f"  {desc}({tag}): {status}")
```

---

## FastAPI完整服务

### API 接口设计

```
+=========================================================================+
|                       文档问答 API 设计                                  |
+=========================================================================+
|                                                                         |
|  认证接口                                                               |
|  POST /api/v1/auth/login          用户登录，返回JWT                     |
|  GET  /api/v1/auth/me             获取当前用户信息                       |
|                                                                         |
|  文档管理                                                               |
|  POST /api/v1/docs/upload         上传文档                              |
|  GET  /api/v1/docs                列出所有文档                          |
|  GET  /api/v1/docs/{doc_id}       获取文档详情                          |
|  DELETE /api/v1/docs/{doc_id}     删除文档                              |
|                                                                         |
|  问答接口                                                               |
|  POST /api/v1/qa/ask              提问(带引用溯源)                      |
|  POST /api/v1/qa/ask/stream       流式提问(SSE)                         |
|  POST /api/v1/qa/feedback         反馈评价                              |
|                                                                         |
|  管理接口                                                               |
|  GET  /api/v1/admin/stats         系统统计                              |
|  POST /api/v1/admin/reindex       重建索引                              |
|                                                                         |
+=========================================================================+
```

### 完整FastAPI代码

```python
"""
企业文档问答系统 - FastAPI 后端服务
"""
import os
import uuid
import time
from typing import Optional, List, Dict
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# 请求/响应模型
# ============================================================
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    token: str
    user_id: str
    username: str
    roles: List[str]

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: Optional[str] = None

class CitationItem(BaseModel):
    citation_id: int
    doc_title: str
    file_name: str
    page_number: int = -1
    section_title: str = ""
    original_text: str = ""

class AskResponse(BaseModel):
    answer: str
    citations: List[CitationItem] = []
    confidence: float = 0.0
    session_id: str = ""
    processing_time_ms: float = 0.0

class DocInfo(BaseModel):
    doc_id: str
    title: str
    file_name: str
    file_type: str
    chunk_count: int
    uploaded_at: str
    department: str = ""

class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


# ============================================================
# 应用服务层
# ============================================================
class DocQAService:
    """文档问答服务 - 整合所有能力"""

    def __init__(self):
        self.documents: Dict[str, DocInfo] = {}
        self.qa_cache: Dict[str, AskResponse] = {}
        self.feedback_log: List[Dict] = []
        self._init_sample_docs()

    def _init_sample_docs(self):
        """初始化示例文档"""
        sample_docs = [
            DocInfo(doc_id="d001", title="员工手册", file_name="handbook.pdf", file_type="pdf", chunk_count=45, uploaded_at="2024-01-01", department="hr"),
            DocInfo(doc_id="d002", title="产品技术文档", file_name="tech_doc.md", file_type="md", chunk_count=32, uploaded_at="2024-01-05", department="tech"),
            DocInfo(doc_id="d003", title="报销制度", file_name="expense.docx", file_type="docx", chunk_count=15, uploaded_at="2024-01-10", department="finance"),
        ]
        for d in sample_docs:
            self.documents[d.doc_id] = d

    def ask(self, question: str, user_roles: List[str], top_k: int = 5) -> AskResponse:
        """处理问答请求"""
        start_time = time.time()

        # 模拟RAG流程
        answer, citations = self._mock_rag(question, user_roles)

        elapsed = (time.time() - start_time) * 1000

        return AskResponse(
            answer=answer,
            citations=citations,
            confidence=0.88,
            session_id=str(uuid.uuid4()),
            processing_time_ms=round(elapsed, 1),
        )

    def _mock_rag(self, question: str, roles: List[str]):
        """模拟RAG问答"""
        qa_map = {
            "入职": ("新员工入职需要准备以下材料:\n1. 身份证原件及复印件\n2. 学历证明\n3. 体检报告\n4. 离职证明 [来源1]\n\n入职当天请到HR部门报到 [来源1]",
                     [CitationItem(citation_id=1, doc_title="员工手册", file_name="handbook.pdf", page_number=5, section_title="入职指南", original_text="新员工入职需要准备身份证、学历证明...")]),
            "报销": ("报销流程如下:\n1. 填写电子报销单\n2. 上传发票附件\n3. 部门经理审批\n4. 财务审核\n5. 打款到工资卡 [来源1]\n\n注意：超过5000元需总监审批 [来源2]",
                     [CitationItem(citation_id=1, doc_title="报销制度", file_name="expense.docx", page_number=2, section_title="报销流程"),
                      CitationItem(citation_id=2, doc_title="报销制度", file_name="expense.docx", page_number=3, section_title="审批权限")]),
            "年假": ("年假制度:\n- 工作满1年: 5天\n- 工作满10年: 10天\n- 工作满20年: 15天 [来源1]",
                     [CitationItem(citation_id=1, doc_title="员工手册", file_name="handbook.pdf", page_number=12, section_title="假期制度")]),
        }

        for keyword, (answer, citations) in qa_map.items():
            if keyword in question:
                return answer, citations

        return f"关于您的问题「{question}」，我在知识库中查找了相关信息，但未找到完全匹配的内容。建议您联系相关部门了解详情。", []


# ============================================================
# FastAPI 应用
# ============================================================
service = DocQAService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[启动] 企业文档问答系统")
    yield
    print("[关闭] 清理资源")


app = FastAPI(
    title="企业文档问答系统 API",
    description="基于RAG的企业知识库问答服务，支持多格式文档、引用溯源、权限控制",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_current_user(authorization: Optional[str] = Header(None)) -> Dict:
    """从请求头提取用户信息(简化版)"""
    # 生产环境应验证JWT
    return {"user_id": "u-001", "username": "zhangsan", "roles": ["tech_staff"], "department": "技术部"}


@app.get("/api/v1/health")
async def health():
    return {"status": "healthy", "docs_count": len(service.documents), "timestamp": datetime.now().isoformat()}


@app.post("/api/v1/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    """用户登录"""
    # 简化演示
    if req.username in ("admin", "zhangsan", "lisi"):
        return LoginResponse(
            token=f"jwt-token-{req.username}",
            user_id=f"u-{req.username}",
            username=req.username,
            roles=["tech_staff"],
        )
    raise HTTPException(status_code=401, detail="用户名或密码错误")


@app.post("/api/v1/qa/ask", response_model=AskResponse)
async def ask_question(req: AskRequest, user: Dict = Depends(get_current_user)):
    """提问接口 - 核心问答功能"""
    try:
        result = service.ask(req.question, user.get("roles", []), req.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/docs", response_model=List[DocInfo])
async def list_documents(user: Dict = Depends(get_current_user)):
    """列出文档"""
    return list(service.documents.values())


@app.post("/api/v1/docs/upload")
async def upload_document(file: UploadFile = File(...), department: str = "public"):
    """上传文档"""
    doc_id = f"d-{uuid.uuid4().hex[:8]}"
    ext = os.path.splitext(file.filename or "")[1]
    doc = DocInfo(
        doc_id=doc_id,
        title=file.filename or "untitled",
        file_name=file.filename or "untitled",
        file_type=ext.lstrip("."),
        chunk_count=0,
        uploaded_at=datetime.now().isoformat(),
        department=department,
    )
    service.documents[doc_id] = doc
    return {"message": f"文档 {file.filename} 上传成功", "doc_id": doc_id}


@app.post("/api/v1/qa/feedback")
async def submit_feedback(req: FeedbackRequest):
    """提交反馈"""
    service.feedback_log.append({
        "session_id": req.session_id,
        "question": req.question,
        "rating": req.rating,
        "comment": req.comment,
        "timestamp": datetime.now().isoformat(),
    })
    return {"message": "感谢反馈"}


@app.get("/api/v1/admin/stats")
async def get_stats():
    """系统统计"""
    return {
        "total_documents": len(service.documents),
        "total_feedbacks": len(service.feedback_log),
        "avg_rating": sum(f["rating"] for f in service.feedback_log) / max(len(service.feedback_log), 1),
    }


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  企业文档问答系统")
    print("  API文档: http://localhost:8001/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
```

---

## 前端问答界面

### 界面设计

```
+=========================================================================+
|                       文档问答界面                                       |
+=========================================================================+
|                                                                         |
|  +---------------------------------------------------------------+     |
|  |  企业知识库问答                             [张三] [退出]      |     |
|  +---------------------------------------------------------------+     |
|  |                                                                |     |
|  |  +----------------------------------------------------------+ |     |
|  |  |                                                          | |     |
|  |  |  Q: 报销流程是什么？                                     | |     |
|  |  |                                                          | |     |
|  |  |  A: 报销流程如下:                                        | |     |
|  |  |     1. 填写电子报销单                                    | |     |
|  |  |     2. 上传发票附件                                      | |     |
|  |  |     3. 部门经理审批                                      | |     |
|  |  |     4. 财务审核 -> 打款                                  | |     |
|  |  |                                                          | |     |
|  |  |  引用来源:                                               | |     |
|  |  |  +----------------------------------------------------+ | |     |
|  |  |  | [1] 报销制度.docx - 第2页 "报销流程"                | | |     |
|  |  |  | [2] 报销制度.docx - 第3页 "审批权限"                | | |     |
|  |  |  +----------------------------------------------------+ | |     |
|  |  |                                                          | |     |
|  |  |  置信度: 88%    处理耗时: 1.2s   [有用] [无用]          | |     |
|  |  +----------------------------------------------------------+ |     |
|  |                                                                |     |
|  |  +----------------------------------------------------------+ |     |
|  |  | 请输入您的问题...                           [发送]       | |     |
|  |  +----------------------------------------------------------+ |     |
|  +---------------------------------------------------------------+     |
|                                                                         |
+=========================================================================+
```

### 前端代码

```python
"""
前端界面 - 嵌入式HTML，用于快速原型演示
"""

DOC_QA_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>企业知识库问答</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family: -apple-system, sans-serif; background:#f0f2f5; }
        .container { max-width:800px; margin:20px auto; }
        .header { background:linear-gradient(135deg,#1a73e8,#0d47a1);
                   color:white; padding:20px; border-radius:12px 12px 0 0; }
        .header h1 { font-size:20px; }
        .qa-area { background:white; min-height:400px; padding:20px;
                    max-height:500px; overflow-y:auto; }
        .qa-item { margin-bottom:20px; border-bottom:1px solid #eee; padding-bottom:16px; }
        .question { color:#1a73e8; font-weight:600; margin-bottom:8px; }
        .answer { color:#333; line-height:1.8; white-space:pre-wrap; }
        .citations { background:#f8f9fa; border-radius:8px; padding:10px; margin-top:8px; font-size:13px; }
        .citation-item { color:#666; padding:2px 0; }
        .meta { font-size:12px; color:#999; margin-top:8px; }
        .input-area { display:flex; gap:8px; padding:16px; background:white;
                       border-radius:0 0 12px 12px; border-top:1px solid #eee; }
        .input-area input { flex:1; padding:12px; border:1px solid #ddd;
                             border-radius:8px; font-size:14px; }
        .input-area button { padding:12px 24px; background:#1a73e8; color:white;
                              border:none; border-radius:8px; cursor:pointer; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>企业知识库问答系统</h1>
            <p style="font-size:13px;opacity:0.8;margin-top:4px;">
                基于RAG技术 | 支持引用溯源 | 权限控制
            </p>
        </div>
        <div class="qa-area" id="qaArea">
            <div style="text-align:center;color:#999;padding:40px;">
                请在下方输入您的问题，例如"报销流程是什么？"
            </div>
        </div>
        <div class="input-area">
            <input id="questionInput" placeholder="请输入您的问题..."
                   onkeypress="if(event.key==='Enter')askQuestion()">
            <button onclick="askQuestion()">提问</button>
        </div>
    </div>
    <script>
        async function askQuestion() {
            const input = document.getElementById('questionInput');
            const q = input.value.trim();
            if (!q) return;
            input.value = '';
            const area = document.getElementById('qaArea');
            if (area.querySelector('div[style]')) area.innerHTML = '';
            area.innerHTML += '<div class="qa-item"><div class="question">Q: '+q+'</div><div class="answer" style="color:#999">正在查询知识库...</div></div>';
            try {
                const resp = await fetch('/api/v1/qa/ask', {
                    method:'POST', headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({question:q, top_k:5})
                });
                const data = await resp.json();
                let html = '<div class="qa-item"><div class="question">Q: '+q+'</div>';
                html += '<div class="answer">'+data.answer+'</div>';
                if (data.citations && data.citations.length > 0) {
                    html += '<div class="citations"><strong>引用来源:</strong>';
                    data.citations.forEach(c => {
                        let page = c.page_number > 0 ? ' 第'+c.page_number+'页' : '';
                        html += '<div class="citation-item">['+c.citation_id+'] '+c.doc_title+' - '+c.file_name+page+'</div>';
                    });
                    html += '</div>';
                }
                html += '<div class="meta">置信度: '+(data.confidence*100).toFixed(0)+'% | 耗时: '+data.processing_time_ms+'ms</div>';
                html += '</div>';
                area.lastElementChild.outerHTML = html;
                area.scrollTop = area.scrollHeight;
            } catch(e) {
                area.lastElementChild.querySelector('.answer').textContent = '查询失败: '+e.message;
            }
        }
    </script>
</body>
</html>
"""


def setup_doc_qa_frontend(app):
    """注册前端路由"""
    from fastapi.responses import HTMLResponse

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return DOC_QA_HTML

    print("[前端] 文档问答界面已注册: GET /")
```

---

## 部署与优化

### 性能优化策略

```
+=========================================================================+
|                       性能优化全景图                                      |
+=========================================================================+
|                                                                         |
|  +--- 查询优化 ---------------------------------------------------+    |
|  |                                                                 |    |
|  |  1. 语义缓存: 相似查询复用结果(余弦相似度 > 0.95)              |    |
|  |  2. 查询改写: 消除歧义，补充上下文                              |    |
|  |  3. 预计算热门问题的答案                                        |    |
|  +-----------------------------------------------------------------+    |
|                                                                         |
|  +--- 检索优化 ---------------------------------------------------+    |
|  |                                                                 |    |
|  |  1. 混合检索: 向量 + BM25 互补                                  |    |
|  |  2. 重排序: Cross-Encoder精排Top-K                              |    |
|  |  3. 分级检索: 先粗筛(快)，再精排(准)                            |    |
|  +-----------------------------------------------------------------+    |
|                                                                         |
|  +--- 生成优化 ---------------------------------------------------+    |
|  |                                                                 |    |
|  |  1. 流式输出: SSE逐字返回，减少等待感                           |    |
|  |  2. Prompt压缩: 只传最相关的上下文                              |    |
|  |  3. 模型选择: 简单问题用小模型，复杂问题用大模型                |    |
|  +-----------------------------------------------------------------+    |
|                                                                         |
+=========================================================================+
```

### 语义缓存实现

```python
"""
语义缓存 - 相似问题复用已有答案，减少API调用
"""
import time
import math
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """缓存条目"""
    query: str
    query_vector: List[float]
    answer: str
    citations: list
    created_at: float
    hit_count: int = 0


class SemanticCache:
    """
    语义缓存

    原理: 将查询向量化，如果新查询与缓存查询的余弦相似度 > 阈值，
    则直接返回缓存的答案，避免重复调用LLM。

    适用场景: 高频重复问题(FAQ)
    """

    def __init__(self, similarity_threshold: float = 0.95, max_size: int = 1000, ttl: int = 3600):
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.ttl = ttl
        self.cache: List[CacheEntry] = []

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get(self, query_vector: List[float]) -> Optional[CacheEntry]:
        """查询缓存"""
        now = time.time()
        best_entry = None
        best_sim = 0.0

        for entry in self.cache:
            # 检查过期
            if now - entry.created_at > self.ttl:
                continue

            sim = self._cosine_similarity(query_vector, entry.query_vector)
            if sim > self.threshold and sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry:
            best_entry.hit_count += 1
            print(f"  [缓存命中] 相似度={best_sim:.4f}, 命中次数={best_entry.hit_count}")

        return best_entry

    def put(self, query: str, query_vector: List[float], answer: str, citations: list):
        """添加到缓存"""
        # 容量检查
        if len(self.cache) >= self.max_size:
            # 删除最老的条目
            self.cache.sort(key=lambda e: e.created_at)
            self.cache = self.cache[self.max_size // 4:]

        self.cache.append(CacheEntry(
            query=query,
            query_vector=query_vector,
            answer=answer,
            citations=citations,
            created_at=time.time(),
        ))

    def get_stats(self) -> Dict:
        """缓存统计"""
        total_hits = sum(e.hit_count for e in self.cache)
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "ttl": self.ttl,
        }


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    import random

    cache = SemanticCache(similarity_threshold=0.90, max_size=100)

    # 模拟向量(实际应调用 embedding API)
    def mock_vector():
        return [random.uniform(-1, 1) for _ in range(10)]

    # 添加缓存
    v1 = mock_vector()
    cache.put("报销流程是什么", v1, "报销流程：填单->审批->打款", [])
    print(f"缓存大小: {cache.get_stats()['cache_size']}")

    # 命中测试(用相同向量)
    result = cache.get(v1)
    if result:
        print(f"命中: {result.answer}")

    # 未命中测试
    v2 = mock_vector()
    result = cache.get(v2)
    print(f"查询结果: {'命中' if result else '未命中'}")
```

---

## 总结

本教程完整实现了一个企业级文档问答系统，涵盖以下核心模块:

1. **文档处理引擎**: 支持PDF/Word/Markdown/TXT/CSV多格式解析、智能分块、元数据提取
2. **向量检索系统**: 混合检索(向量语义 + BM25关键词)、RRF结果融合、权限过滤
3. **引用溯源机制**: Prompt内嵌来源编号、LLM回答中标注引用、自动解析映射
4. **权限与安全**: JWT认证、RBAC角色权限、文档标签过滤
5. **FastAPI后端**: 完整REST API(登录/上传/问答/反馈/统计)
6. **前端界面**: 嵌入式问答界面，展示引用来源和置信度
7. **性能优化**: 语义缓存、混合检索、流式输出

## 最佳实践

1. **分块策略**: 按语义边界分块，保留章节上下文，分块大小300-600字为宜
2. **检索质量**: 混合检索优于单一向量检索，BM25处理精确匹配更佳
3. **引用标注**: 在Prompt中明确要求标注来源，提高可信度和可审计性
4. **权限设计**: 最小权限原则，文档打标签分级，检索时过滤
5. **缓存策略**: 高频问题使用语义缓存，降低API成本50%以上

## 参考资源

- [ChromaDB 文档](https://docs.trychroma.com/)
- [OpenAI Embeddings 指南](https://platform.openai.com/docs/guides/embeddings)
- [PyMuPDF 文档](https://pymupdf.readthedocs.io/)
- [BM25 算法详解](https://en.wikipedia.org/wiki/Okapi_BM25)

---

**创建时间**: 2024-01-01
**最后更新**: 2024-01-01
