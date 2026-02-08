# LlamaIndex完整教程

## 目录
1. [LlamaIndex简介](#llamaindex简介)
2. [核心架构](#核心架构)
3. [索引构建](#索引构建)
4. [查询引擎](#查询引擎)
5. [与LangChain对比](#与langchain对比)
6. [完整文档问答示例](#完整文档问答示例)

---

## LlamaIndex简介

### 什么是LlamaIndex

LlamaIndex (原GPT Index) 是一个专注于数据索引和检索的框架，特别适合构建RAG应用。

### 安装

```bash
pip install llama-index
pip install llama-index-llms-openai
pip install llama-index-embeddings-openai
```

### 核心概念

```
┌─────────────────────────────────────────────────┐
│         LlamaIndex数据处理流程                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  原始文档                                        │
│     │                                           │
│     ▼                                           │
│  Document Loading (文档加载)                    │
│     │                                           │
│     ▼                                           │
│  Parsing & Chunking (解析与分块)                │
│     │                                           │
│     ▼                                           │
│  Indexing (索引构建)                            │
│     │                                           │
│     ├─ VectorStoreIndex (向量索引)              │
│     ├─ ListIndex (列表索引)                     │
│     ├─ TreeIndex (树索引)                       │
│     └─ KeywordTableIndex (关键词索引)           │
│     │                                           │
│     ▼                                           │
│  Query Engine (查询引擎)                        │
│     │                                           │
│     ▼                                           │
│  Response Synthesis (响应合成)                  │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 核心架构

### 快速开始

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 配置
llm = OpenAI(model="gpt-4", temperature=0.1)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 构建索引
index = VectorStoreIndex.from_documents(
    documents,
    llm=llm,
    embed_model=embed_model
)

# 查询
query_engine = index.as_query_engine()
response = query_engine.query("这些文档讲了什么？")
print(response)
```

### Document和Node

```python
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser

# 创建Document
doc1 = Document(
    text="LlamaIndex是一个数据框架",
    metadata={"source": "intro.txt"}
)

doc2 = Document(
    text="它专注于索引和检索",
    metadata={"source": "intro.txt"}
)

# 解析成Nodes
parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=20)
nodes = parser.get_nodes_from_documents([doc1, doc2])

print(f"生成了{len(nodes)}个nodes")
for node in nodes:
    print(f"Node: {node.text[:50]}...")
```

---

## 索引构建

### VectorStoreIndex (向量索引)

```python
from llama_index.core import VectorStoreIndex, Document

documents = [
    Document(text="Python是一种编程语言"),
    Document(text="机器学习是AI的分支"),
    Document(text="深度学习使用神经网络")
]

# 构建向量索引
index = VectorStoreIndex.from_documents(documents)

# 查询
query_engine = index.as_query_engine(similarity_top_k=2)
response = query_engine.query("什么是Python？")
print(response)
```

### ListIndex (列表索引)

```python
from llama_index.core import ListIndex

# 列表索引: 顺序扫描所有节点
index = ListIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("总结这些文档")
print(response)
```

### TreeIndex (树索引)

```python
from llama_index.core import TreeIndex

# 树索引: 构建层次结构
index = TreeIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("主要内容是什么？")
print(response)
```

### KeywordTableIndex (关键词索引)

```python
from llama_index.core import KeywordTableIndex

# 关键词索引: 基于关键词匹配
index = KeywordTableIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("Python")
print(response)
```

### 持久化索引

```python
from llama_index.core import StorageContext, load_index_from_storage

# 保存索引
index.storage_context.persist(persist_dir="./storage")

# 加载索引
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

---

## 查询引擎

### 基础查询

```python
# 创建查询引擎
query_engine = index.as_query_engine(
    similarity_top_k=3,  # 返回top3相似结果
    response_mode="compact"  # 响应模式
)

response = query_engine.query("什么是机器学习？")
print(response)

# 查看源节点
for node in response.source_nodes:
    print(f"Score: {node.score:.3f}")
    print(f"Text: {node.text[:100]}...")
```

### 高级查询配置

```python
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# 自定义Retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5
)

# 自定义Response Synthesizer
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize"
)

# 组合查询引擎
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer
)

response = query_engine.query("总结主要内容")
print(response)
```

### Response Modes

```python
# compact: 紧凑模式，组合多个chunks
query_engine = index.as_query_engine(response_mode="compact")

# refine: 迭代优化答案
query_engine = index.as_query_engine(response_mode="refine")

# tree_summarize: 树状总结
query_engine = index.as_query_engine(response_mode="tree_summarize")

# simple_summarize: 简单总结
query_engine = index.as_query_engine(response_mode="simple_summarize")
```

---

## 与LangChain对比

### 功能对比

```
┌──────────────────────────────────────────────────────┐
│          LlamaIndex vs LangChain                      │
├──────────────────────────────────────────────────────┤
│                                                      │
│  特性            LlamaIndex        LangChain         │
│  ────────────────────────────────────────────────   │
│  核心定位        数据索引检索      通用LLM应用框架   │
│  索引类型        多种专门索引      通用向量存储      │
│  查询优化        内置优化          需自己实现        │
│  文档处理        强大              基础              │
│  Agent支持       基础              强大              │
│  学习曲线        简单              中等              │
│  最适合          RAG应用           复杂Agent应用     │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 集成使用

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# LlamaIndex构建索引
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 转换为LangChain Retriever
retriever = index.as_retriever()

# 在LangChain中使用
llm = ChatOpenAI(model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

result = qa_chain.invoke({"query": "什么是机器学习？"})
print(result)
```

---

## 完整文档问答示例

```python
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from pathlib import Path
import os

class DocumentQA:
    """文档问答系统"""

    def __init__(self, data_dir="./data", storage_dir="./storage"):
        self.data_dir = data_dir
        self.storage_dir = storage_dir
        self.index = None
        self.query_engine = None

        # 配置
        self.llm = OpenAI(model="gpt-4", temperature=0)

    def load_documents(self):
        """加载文档"""
        print("正在加载文档...")
        documents = SimpleDirectoryReader(self.data_dir).load_data()
        print(f"加载了{len(documents)}个文档")
        return documents

    def build_index(self, force_rebuild=False):
        """构建或加载索引"""
        if not force_rebuild and os.path.exists(self.storage_dir):
            print("加载已有索引...")
            storage_context = StorageContext.from_defaults(
                persist_dir=self.storage_dir
            )
            self.index = load_index_from_storage(storage_context)
        else:
            print("构建新索引...")
            documents = self.load_documents()

            # 分块
            parser = SimpleNodeParser.from_defaults(
                chunk_size=1024,
                chunk_overlap=20
            )
            nodes = parser.get_nodes_from_documents(documents)

            # 构建索引
            self.index = VectorStoreIndex(nodes, llm=self.llm)

            # 保存
            self.index.storage_context.persist(persist_dir=self.storage_dir)

        # 创建查询引擎
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=3,
            response_mode="compact"
        )

        print("索引准备完成！")

    def query(self, question: str, show_sources=True):
        """查询"""
        if not self.query_engine:
            raise ValueError("请先构建索引！")

        print(f"\n问题: {question}")
        response = self.query_engine.query(question)

        print(f"\n答案: {response}")

        if show_sources:
            print("\n参考来源:")
            for i, node in enumerate(response.source_nodes, 1):
                print(f"\n[{i}] 相似度: {node.score:.3f}")
                print(f"内容: {node.text[:200]}...")
                if node.metadata:
                    print(f"元数据: {node.metadata}")

        return response

    def chat(self):
        """交互式问答"""
        if not self.query_engine:
            self.build_index()

        print("\n文档问答系统已启动! (输入'quit'退出)")

        while True:
            question = input("\n请输入问题: ").strip()

            if question.lower() == 'quit':
                break

            if not question:
                continue

            try:
                self.query(question)
            except Exception as e:
                print(f"错误: {e}")

# 使用示例
if __name__ == "__main__":
    # 创建测试文档
    os.makedirs("./data", exist_ok=True)

    with open("./data/python.txt", "w", encoding="utf-8") as f:
        f.write("""
Python是一种高级编程语言，由Guido van Rossum在1989年底发明。
Python的设计哲学强调代码的可读性和简洁的语法。
Python支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。
Python有一个庞大的标准库，被称为"自带电池"。
        """)

    with open("./data/ml.txt", "w", encoding="utf-8") as f:
        f.write("""
机器学习是人工智能的一个分支。
它使用算法解析数据，从中学习，然后对真实世界中的事件做出决策和预测。
机器学习算法包括监督学习、无监督学习和强化学习。
常见的机器学习库包括scikit-learn、TensorFlow和PyTorch。
        """)

    # 初始化系统
    qa_system = DocumentQA(data_dir="./data", storage_dir="./storage")

    # 构建索引
    qa_system.build_index()

    # 查询示例
    qa_system.query("什么是Python？")
    qa_system.query("机器学习有哪些类型？")

    # 交互式问答
    # qa_system.chat()
```

---

## 总结

LlamaIndex是构建RAG应用的强大工具:

1. **专注数据**: 提供多种索引类型
2. **易于使用**: API简洁直观
3. **高性能**: 内置查询优化
4. **灵活扩展**: 支持自定义组件
5. **良好集成**: 可与LangChain配合使用

## 最佳实践

1. 根据数据特点选择索引类型
2. 合理设置chunk_size和overlap
3. 使用持久化避免重复构建
4. 调整similarity_top_k优化结果
5. 利用元数据过滤提高精度

## 参考资源

- [LlamaIndex官方文档](https://docs.llamaindex.ai/)
- [LlamaIndex GitHub](https://github.com/run-llama/llama_index)
