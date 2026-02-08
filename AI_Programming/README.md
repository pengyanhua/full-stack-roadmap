# AI编程完整知识体系

从基础到实战，掌握AI应用开发、深度学习、AI工程和AI辅助编程的完整技能栈。

## 📚 学习路径

```
┌─────────────────────────────────────────────────────────┐
│              AI编程学习路线图                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  初学者 ────▶ 基础篇 ────▶ 框架篇 ────▶ 实战项目      │
│              (LLM/Prompt)  (LangChain)    (Chatbot)     │
│                                                         │
│  进阶者 ────▶ RAG系统 ───▶ Agent ─────▶ 深度学习      │
│              (检索增强)    (智能体)     (微调/优化)     │
│                                                         │
│  工程师 ────▶ AI工程 ────▶ AI辅助编程                  │
│              (MLOps)       (Copilot/Cursor)             │
└─────────────────────────────────────────────────────────┘
```

## 📖 目录结构

### 01. 基础篇 (3个教程)
- [LLM基础与API](01_fundamentals/01_llm_basics.md) - OpenAI/Claude/Gemini API
- [Prompt工程](01_fundamentals/02_prompt_engineering.md) - Few-shot/CoT/ReAct
- [Embedding与向量](01_fundamentals/03_embedding_vectors.md) - 语义理解基础

### 02. 开发框架篇 (4个教程)
- [LangChain完整教程](02_development_frameworks/01_langchain.md) - LCEL/Chains/Memory
- [LlamaIndex](02_development_frameworks/02_llamaindex.md) - 索引与查询引擎
- [Semantic Kernel](02_development_frameworks/03_semantic_kernel.md) - 微软AI框架
- [AutoGen多Agent](02_development_frameworks/04_autogen.md) - 多智能体对话

### 03. RAG系统篇 (5个教程) ⭐
- [RAG架构设计](03_rag_systems/01_rag_architecture.md) - 端到端架构
- [向量数据库](03_rag_systems/02_vector_databases.md) - Milvus/Qdrant/Pinecone
- [文档分块策略](03_rag_systems/03_chunking_strategies.md) - Semantic Chunking
- [检索优化](03_rag_systems/04_retrieval_optimization.md) - Rerank/HyDE
- [RAG实战项目](03_rag_systems/05_rag_practice.md) - 企业知识库

### 04. Agent系统篇 (5个教程) ⭐
- [Agent基础](04_agent_systems/01_agent_basics.md) - 规划、工具、记忆
- [ReAct模式](04_agent_systems/02_react_pattern.md) - 推理与行动
- [Function Calling](04_agent_systems/03_tool_calling.md) - 工具调用
- [LangGraph](04_agent_systems/04_langgraph.md) - 状态图Agent
- [多Agent协作](04_agent_systems/05_multi_agent.md) - 协作模式

### 05. 深度学习篇 (4个教程)
- [PyTorch基础](05_deep_learning/01_pytorch_basics.md) - Tensor/Autograd/Module
- [Transformer实现](05_deep_learning/02_transformer.md) - 从零实现
- [模型微调](05_deep_learning/03_fine_tuning.md) - LoRA/QLoRA/PEFT
- [模型优化](05_deep_learning/04_model_optimization.md) - 量化/剪枝/蒸馏

### 06. AI工程篇 (4个教程)
- [MLOps实践](06_ai_engineering/01_mlops_practice.md) - 完整流程
- [模型服务化](06_ai_engineering/02_model_serving.md) - vLLM/TGI
- [AI系统监控](06_ai_engineering/03_monitoring.md) - Prometheus/Grafana
- [成本优化](06_ai_engineering/04_cost_optimization.md) - Token/GPU优化

### 07. AI辅助编程篇 (4个教程)
- [GitHub Copilot](07_ai_assisted_coding/01_github_copilot.md) - 最佳实践
- [Cursor编辑器](07_ai_assisted_coding/02_cursor_editor.md) - AI-first IDE
- [AI代码审查](07_ai_assisted_coding/03_ai_code_review.md) - 自动化Review
- [效率提升](07_ai_assisted_coding/04_productivity_boost.md) - 10x工程师

### 08. 实战项目篇 (4个教程)
- [智能客服](08_practical_projects/01_chatbot.md) - RAG+Agent聊天机器人
- [文档问答](08_practical_projects/02_document_qa.md) - 企业文档助手
- [代码助手](08_practical_projects/03_code_assistant.md) - AI编程助手
- [数据分析](08_practical_projects/04_data_analysis.md) - AI数据分析师

## 🎯 快速开始

### 环境准备

```bash
# 1. 安装Python 3.10+
python --version

# 2. 创建虚拟环境
python -m venv ai_env
source ai_env/bin/activate  # Windows: ai_env\Scripts\activate

# 3. 安装核心依赖
pip install openai langchain langchain-community chromadb
pip install sentence-transformers torch transformers
```

### Hello World

```python
# hello_ai.py - 第一个AI程序
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "用一句话解释什么是AI编程"}
    ]
)

print(response.choices[0].message.content)
```

## 🛠️ 技术栈

### LLM平台
- **OpenAI**: GPT-4/GPT-3.5
- **Anthropic**: Claude 3.5 Sonnet
- **Google**: Gemini Pro
- **开源**: Llama 3/Mistral/Qwen

### 开发框架
- **LangChain**: 最流行的LLM应用框架
- **LlamaIndex**: 专注于索引与检索
- **Semantic Kernel**: 微软AI编排框架
- **AutoGen**: 多Agent对话框架

### 向量数据库
- **Milvus**: 开源高性能
- **Qdrant**: Rust编写，速度快
- **Pinecone**: 托管服务
- **Chroma**: 轻量级本地

### 深度学习框架
- **PyTorch**: 最流行
- **TensorFlow**: Google出品
- **JAX**: 高性能计算

## 📊 学习建议

### 初学者（0-3个月）
1. 从基础篇开始，理解LLM和Prompt
2. 学习LangChain，构建简单应用
3. 完成一个实战项目（Chatbot）

### 进阶者（3-6个月）
1. 深入学习RAG系统设计
2. 掌握Agent开发
3. 完成企业级项目

### 高级开发者（6个月+）
1. 学习模型微调和优化
2. 掌握AI工程化部署
3. 研究多Agent系统

## 🔗 相关资源

### 官方文档
- [LangChain](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs)
- [Hugging Face](https://huggingface.co/docs)

### 推荐课程
- [DeepLearning.AI - LangChain](https://www.deeplearning.ai/short-courses/)
- [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/)

### 社区
- [LangChain Discord](https://discord.gg/langchain)
- [Hugging Face Forums](https://discuss.huggingface.co/)

## 💡 最佳实践

1. **从小模型开始**: 先用GPT-3.5调试，再用GPT-4
2. **管理成本**: 使用缓存、限制token数
3. **监控质量**: 记录输入输出，持续优化
4. **安全第一**: 防止Prompt注入攻击
5. **用户反馈**: 收集反馈，迭代改进

## 🚀 开始学习

选择适合你的起点：
- **零基础**: 从 [01_fundamentals/01_llm_basics.md](01_fundamentals/01_llm_basics.md) 开始
- **有Python经验**: 直接学习 [02_development_frameworks/01_langchain.md](02_development_frameworks/01_langchain.md)
- **想快速实战**: 跳转到 [08_practical_projects/](08_practical_projects/)

祝您学习愉快！🎉
