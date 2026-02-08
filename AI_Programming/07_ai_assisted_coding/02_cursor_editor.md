# Cursor 编辑器完整教程

## 目录
1. [Cursor简介与核心优势](#cursor简介与核心优势)
2. [安装与配置](#安装与配置)
3. [Cmd+K 内联编辑](#cmdk-内联编辑)
4. [Chat 智能对话](#chat-智能对话)
5. [Composer 多文件编辑](#composer-多文件编辑)
6. [与GitHub Copilot对比](#与github-copilot对比)
7. [高级技巧与工作流](#高级技巧与工作流)
8. [实战项目案例](#实战项目案例)

---

## Cursor简介与核心优势

### 什么是 Cursor

Cursor 是一款基于 VS Code 深度分叉（fork）的 AI 原生代码编辑器。它不是简单地在
编辑器上叠加 AI 插件,而是将 AI 能力深度融入编辑器的每一个交互环节。Cursor 支持
GPT-4、Claude 等多种大模型,提供了代码补全、内联编辑（Cmd+K）、智能对话（Chat）、
多文件编辑（Composer）等核心功能。

### 核心架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cursor 编辑器架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                     用户界面层                              │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │  │
│  │  │ 代码编辑  │  │ Cmd+K    │  │ Chat     │  │ Composer │  │  │
│  │  │ (补全)    │  │ (内联)   │  │ (对话)   │  │ (多文件) │  │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │  │
│  └───────┼──────────────┼──────────────┼──────────────┼───────┘  │
│          v              v              v              v          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   AI 引擎层                                │  │
│  │  ┌──────────────────────────────────────────────────────┐ │  │
│  │  │  上下文引擎                                          │ │  │
│  │  │  - 代码库索引 (全项目语义搜索)                        │ │  │
│  │  │  - 打开文件分析                                      │ │  │
│  │  │  - Git 历史追踪                                      │ │  │
│  │  │  - 符号/引用解析                                     │ │  │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  │  ┌──────────────────────────────────────────────────────┐ │  │
│  │  │  模型路由                                            │ │  │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │ │  │
│  │  │  │ GPT-4o   │  │ Claude   │  │ cursor-small     │  │ │  │
│  │  │  │ (对话)   │  │ (编辑)   │  │ (快速补全)       │  │ │  │
│  │  │  └──────────┘  └──────────┘  └──────────────────┘  │ │  │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   VS Code 基础层                           │  │
│  │  扩展系统 | 文件系统 | 终端 | 调试器 | Git 集成             │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Cursor 三大核心功能

```
┌────────────────────────────────────────────────────────────────┐
│                 Cursor 三大核心功能对比                          │
├──────────────┬──────────────┬──────────────┬──────────────────┤
│  维度         │  Cmd+K       │  Chat        │  Composer        │
├──────────────┼──────────────┼──────────────┼──────────────────┤
│  触发方式     │  Ctrl+K      │  Ctrl+L      │  Ctrl+Shift+I   │
│  操作范围     │  选中代码/    │  整个项目     │  多个文件        │
│              │  当前位置     │              │                  │
│  交互模式     │  单次指令     │  多轮对话     │  任务式编排      │
│  适用场景     │  局部修改     │  问答/分析    │  跨文件重构      │
│  上下文       │  当前文件     │  @引用       │  自动检测        │
│  输出方式     │  原地替换     │  侧边面板     │  diff 预览       │
├──────────────┴──────────────┴──────────────┴──────────────────┤
│  使用频率建议:                                                 │
│  Cmd+K (60%) > Chat (25%) > Composer (15%)                    │
│  日常小修改用Cmd+K, 问问题用Chat, 大改动用Composer              │
└────────────────────────────────────────────────────────────────┘
```

---

## 安装与配置

### 安装流程

```
┌──────────────────────────────────────────────────────────────┐
│                  Cursor 安装配置流程                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: 下载安装                                            │
│  ┌────────────────────────────────────┐                      │
│  │  访问 cursor.com                   │                      │
│  │  下载对应平台安装包                 │                      │
│  │  (Windows / macOS / Linux)         │                      │
│  └──────────────┬─────────────────────┘                      │
│                 v                                             │
│  Step 2: 迁移 VS Code 配置                                   │
│  ┌────────────────────────────────────┐                      │
│  │  首次启动时会提示:                  │                      │
│  │  - 导入 VS Code 扩展               │                      │
│  │  - 导入 VS Code 设置               │                      │
│  │  - 导入 VS Code 快捷键             │                      │
│  │  建议: 全部导入, 无缝切换           │                      │
│  └──────────────┬─────────────────────┘                      │
│                 v                                             │
│  Step 3: 登录 Cursor 账号                                    │
│  ┌────────────────────────────────────┐                      │
│  │  支持: GitHub / Google / Email     │                      │
│  │  免费版: 每月 2000 次补全           │                      │
│  │  Pro版:  $20/月, 无限补全+GPT-4    │                      │
│  └──────────────┬─────────────────────┘                      │
│                 v                                             │
│  Step 4: 配置 AI 模型偏好                                    │
│  ┌────────────────────────────────────┐                      │
│  │  Settings > Models                 │                      │
│  │  选择默认模型 (GPT-4o / Claude)    │                      │
│  │  可以按功能分别设置模型             │                      │
│  └────────────────────────────────────┘                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 核心设置推荐

```json
// Cursor Settings (settings.json)
{
    // ===== AI 功能配置 =====

    // 开启代码库索引(让AI理解整个项目)
    "cursor.codebaseIndexing.enabled": true,

    // Tab 补全行为
    "cursor.autocomplete.enabled": true,
    "cursor.autocomplete.alwaysSuggest": true,

    // 自动导入建议
    "cursor.autocomplete.autoImport": true,

    // ===== 编辑器优化 =====

    // 内联建议样式
    "editor.inlineSuggest.enabled": true,
    "editor.suggest.preview": true,

    // 保存时自动格式化
    "editor.formatOnSave": true,

    // ===== 模型选择 =====
    // 可在 Cursor Settings > Models 中配置
    // 推荐:
    //   Chat: claude-3.5-sonnet (推理更强)
    //   Edit: gpt-4o (编辑更快)
    //   Autocomplete: cursor-small (速度优先)

    // ===== 隐私设置 =====
    // 隐私模式: 代码不会被用于训练
    "cursor.privacy.mode": "privacy"
}
```

### 快捷键体系

```
┌────────────────────────────────────────────────────────────────┐
│              Cursor 快捷键完整速查表                             │
├────────────────────┬───────────────────────────────────────────┤
│  功能               │  快捷键 (Win/Linux)                       │
├────────────────────┼───────────────────────────────────────────┤
│                    │  === AI 核心功能 ===                       │
│  Cmd+K 内联编辑     │  Ctrl + K                                 │
│  Chat 对话面板      │  Ctrl + L                                 │
│  Composer 多文件    │  Ctrl + Shift + I                         │
│  接受 AI 建议       │  Tab                                      │
│  拒绝 AI 建议       │  Esc                                      │
│  查看下一个建议      │  Alt + ]                                  │
│                    │                                            │
│                    │  === Chat 操作 ===                         │
│  新建 Chat          │  Ctrl + L (无选中代码时)                   │
│  添加代码到 Chat     │  选中代码后 Ctrl + L                      │
│  添加文件到 Chat     │  在Chat中输入 @file                       │
│  添加文档到 Chat     │  在Chat中输入 @doc                        │
│  添加代码库到 Chat   │  在Chat中输入 @codebase                   │
│                    │                                            │
│                    │  === 编辑器标准 ===                        │
│  命令面板           │  Ctrl + Shift + P                          │
│  文件搜索           │  Ctrl + P                                  │
│  全局搜索           │  Ctrl + Shift + F                          │
│  终端切换           │  Ctrl + `                                  │
├────────────────────┴───────────────────────────────────────────┤
│  提示: macOS 用户将 Ctrl 替换为 Cmd                             │
└────────────────────────────────────────────────────────────────┘
```

---

## Cmd+K 内联编辑

### 工作原理

Cmd+K（Windows 上为 Ctrl+K）是 Cursor 最高频使用的功能。它允许你在编辑器内直接
用自然语言描述你想要的修改,AI 会在原位生成或替换代码。

```
┌────────────────────────────────────────────────────────────────┐
│                  Cmd+K 工作流程                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  模式A: 无选中代码 (生成新代码)                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. 将光标放在想插入代码的位置                            │  │
│  │  2. 按 Ctrl+K                                            │  │
│  │  3. 输入指令: "创建一个用户注册的Express路由"              │  │
│  │  4. AI 在光标位置生成代码                                 │  │
│  │  5. 按 Tab 接受 / Esc 拒绝                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  模式B: 有选中代码 (修改现有代码)                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. 选中要修改的代码块                                    │  │
│  │  2. 按 Ctrl+K                                            │  │
│  │  3. 输入指令: "添加错误处理和输入验证"                     │  │
│  │  4. AI 显示 diff 对比(红色删除/绿色新增)                  │  │
│  │  5. 按 Tab 接受 / Esc 拒绝                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  模式C: 终端中使用 (生成命令)                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. 在终端中按 Ctrl+K                                    │  │
│  │  2. 输入: "查找所有大于100MB的日志文件并删除"              │  │
│  │  3. AI 生成对应的 shell 命令                              │  │
│  │  4. 按 Enter 执行                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Cmd+K 实战示例

```python
# ============================================================
# 示例1: 生成新代码
# 光标放在这里, 按 Ctrl+K, 输入:
# "创建一个FastAPI路由, 实现用户CRUD, 包含分页和搜索"
# ============================================================

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

app = FastAPI(title="用户管理API")

# 用户数据模型
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    created_at: datetime
    is_active: bool

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None

# 模拟数据库
users_db: dict = {}
next_id = 1


# 创建用户
@app.post("/users", response_model=UserResponse, status_code=201)
async def create_user(user: UserCreate):
    global next_id

    # 检查用户名是否已存在
    for existing in users_db.values():
        if existing["username"] == user.username:
            raise HTTPException(status_code=409, detail="用户名已存在")

    user_data = {
        "id": next_id,
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "created_at": datetime.now(),
        "is_active": True
    }
    users_db[next_id] = user_data
    next_id += 1
    return user_data


# 获取用户列表（支持分页和搜索）
@app.get("/users", response_model=List[UserResponse])
async def list_users(
    page: int = Query(1, ge=1, description="页码"),
    size: int = Query(10, ge=1, le=100, description="每页数量"),
    search: Optional[str] = Query(None, description="搜索关键词")
):
    all_users = list(users_db.values())

    # 搜索过滤
    if search:
        all_users = [
            u for u in all_users
            if search.lower() in u["username"].lower()
            or search.lower() in (u.get("full_name") or "").lower()
        ]

    # 分页
    start = (page - 1) * size
    end = start + size
    return all_users[start:end]


# 获取单个用户
@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="用户不存在")
    return users_db[user_id]


# 更新用户
@app.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: int, user_update: UserUpdate):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="用户不存在")

    user_data = users_db[user_id]
    update_dict = user_update.dict(exclude_unset=True)
    user_data.update(update_dict)
    return user_data


# 删除用户
@app.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: int):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="用户不存在")
    del users_db[user_id]
```

```python
# ============================================================
# 示例2: 选中代码后用 Cmd+K 重构
# 选中下面的函数, 按 Ctrl+K, 输入:
# "重构为async, 添加超时控制和重试逻辑"
# ============================================================

import asyncio
import aiohttp
from typing import Optional

# 重构前的简单版本:
def fetch_data_simple(url: str) -> dict:
    import requests
    response = requests.get(url)
    return response.json()


# Cmd+K 重构后的完整版本:
async def fetch_data(
    url: str,
    timeout: int = 30,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Optional[dict]:
    """
    异步获取远程数据,支持超时和重试。

    Args:
        url: 请求URL
        timeout: 超时时间(秒)
        max_retries: 最大重试次数
        retry_delay: 重试间隔(秒), 每次翻倍

    Returns:
        JSON响应数据, 失败返回None
    """
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status >= 500:
                        # 服务端错误,可以重试
                        raise aiohttp.ClientError(f"服务端错误: {response.status}")
                    else:
                        # 客户端错误,不重试
                        print(f"请求失败: HTTP {response.status}")
                        return None

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"第{attempt + 1}次请求失败, {wait_time}秒后重试: {e}")
                await asyncio.sleep(wait_time)
            else:
                print(f"所有{max_retries}次请求均失败: {e}")
                return None

    return None
```

---

## Chat 智能对话

### Chat 功能详解

```
┌────────────────────────────────────────────────────────────────┐
│                  Cursor Chat 功能架构                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  上下文引用系统                            │  │
│  │                                                          │  │
│  │  @file        引用指定文件                                │  │
│  │  @folder      引用整个文件夹                              │  │
│  │  @codebase    搜索整个代码库 (语义搜索)                   │  │
│  │  @doc         引用外部文档 URL                            │  │
│  │  @git         引用 Git 变更记录                           │  │
│  │  @web         联网搜索最新信息                             │  │
│  │  @definitions 引用符号定义                                │  │
│  │  选中代码      自动作为上下文                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  Chat 使用场景                             │  │
│  │                                                          │  │
│  │  [1] 代码理解    "解释这个函数的作用和潜在问题"            │  │
│  │  [2] Bug 排查    "这段代码为什么输出和预期不一致?"         │  │
│  │  [3] 方案设计    "@codebase 如何给这个项目添加缓存层?"    │  │
│  │  [4] 学习提问    "这里用到了什么设计模式? 为什么?"        │  │
│  │  [5] 代码审查    "审查这个PR的代码质量和安全性"           │  │
│  │  [6] 性能分析    "@file:api.ts 这个文件有什么性能瓶颈?"  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  Apply 按钮                               │  │
│  │                                                          │  │
│  │  Chat 回复中的代码块旁边有 "Apply" 按钮                  │  │
│  │  点击后自动将代码应用到对应文件,显示 diff 预览            │  │
│  │  流程: Chat建议 -> Apply -> diff预览 -> Accept/Reject    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Chat 对话示例

```python
# ============================================================
# 场景: 在 Chat 中输入以下内容进行项目分析
# ============================================================

# 用户输入: "@codebase 帮我分析这个项目的数据库访问模式,
#           是否存在 N+1 查询问题?"

# Cursor Chat 会:
# 1. 扫描整个代码库的数据库相关代码
# 2. 识别 ORM 查询模式
# 3. 指出潜在的 N+1 问题
# 4. 给出修复建议

# 示例: Chat 发现的 N+1 问题
# ---- 有问题的代码 ----
class OrderService:
    def get_orders_with_items(self, user_id: int):
        """获取用户的所有订单及商品 - 存在 N+1 问题"""
        orders = Order.query.filter_by(user_id=user_id).all()  # 1次查询

        result = []
        for order in orders:
            items = OrderItem.query.filter_by(order_id=order.id).all()  # N次查询!
            result.append({
                "order": order,
                "items": items
            })
        return result  # 总共 N+1 次数据库查询


# ---- Chat 建议的修复 ----
class OrderServiceOptimized:
    def get_orders_with_items(self, user_id: int):
        """获取用户的所有订单及商品 - 使用 eager loading 优化"""
        # 使用 joinedload 一次性加载关联数据, 只需1次查询
        orders = (
            Order.query
            .options(joinedload(Order.items))
            .filter_by(user_id=user_id)
            .all()
        )

        return [
            {"order": order, "items": order.items}
            for order in orders
        ]  # 总共只需 1 次数据库查询
```

### @doc 引用外部文档

```typescript
// ============================================================
// 场景: 引用第三方库文档来生成代码
// Chat 输入: "@doc https://docs.stripe.com/api/charges
//            帮我用 Node.js 实现 Stripe 支付流程"
// ============================================================

// Cursor 会阅读 Stripe 文档并生成符合最新 API 的代码:

import Stripe from 'stripe';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
    apiVersion: '2023-10-16'
});

interface PaymentParams {
    amount: number;       // 金额(分)
    currency: string;     // 货币类型
    customerId: string;   // 客户ID
    description?: string; // 描述
}

// 创建支付意图 (Payment Intent)
async function createPayment(params: PaymentParams) {
    try {
        const paymentIntent = await stripe.paymentIntents.create({
            amount: params.amount,
            currency: params.currency,
            customer: params.customerId,
            description: params.description,
            automatic_payment_methods: {
                enabled: true,  // 自动启用可用的支付方式
            },
        });

        return {
            success: true,
            clientSecret: paymentIntent.client_secret,
            paymentIntentId: paymentIntent.id,
            status: paymentIntent.status
        };
    } catch (error) {
        if (error instanceof Stripe.errors.StripeCardError) {
            return { success: false, error: '卡片被拒绝', code: error.code };
        }
        throw error;
    }
}

// 确认支付
async function confirmPayment(paymentIntentId: string) {
    const paymentIntent = await stripe.paymentIntents.retrieve(paymentIntentId);

    if (paymentIntent.status === 'succeeded') {
        return { success: true, message: '支付成功' };
    } else if (paymentIntent.status === 'requires_action') {
        return { success: false, message: '需要额外验证', requiresAction: true };
    } else {
        return { success: false, message: `支付状态: ${paymentIntent.status}` };
    }
}

// Webhook 处理支付事件
async function handleWebhook(payload: string, signature: string) {
    const event = stripe.webhooks.constructEvent(
        payload,
        signature,
        process.env.STRIPE_WEBHOOK_SECRET!
    );

    switch (event.type) {
        case 'payment_intent.succeeded':
            console.log('支付成功:', event.data.object);
            break;
        case 'payment_intent.payment_failed':
            console.log('支付失败:', event.data.object);
            break;
        default:
            console.log('未处理的事件类型:', event.type);
    }
}
```

---

## Composer 多文件编辑

### Composer 工作原理

```
┌────────────────────────────────────────────────────────────────┐
│                Composer 多文件编辑流程                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Step 1: 打开 Composer (Ctrl+Shift+I)                   │  │
│  │  ┌──────────────────────────────────────────────┐       │  │
│  │  │  描述你的需求:                                │       │  │
│  │  │  "给这个Express项目添加JWT认证中间件,          │       │  │
│  │  │   需要修改路由文件、创建中间件文件、             │       │  │
│  │  │   更新配置文件和添加相关测试"                   │       │  │
│  │  └──────────────────────────────────────────────┘       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                     │
│                          v                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Step 2: Composer 分析项目结构                           │  │
│  │  ┌────────────────┐  ┌────────────────┐                 │  │
│  │  │ 扫描文件结构    │  │ 理解依赖关系    │                 │  │
│  │  │ src/            │  │ routes -> auth  │                 │  │
│  │  │ ├── routes/     │  │ auth -> config  │                 │  │
│  │  │ ├── middleware/ │  │ test -> all     │                 │  │
│  │  │ ├── config/     │  │                 │                 │  │
│  │  │ └── tests/      │  │                 │                 │  │
│  │  └────────────────┘  └────────────────┘                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                     │
│                          v                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Step 3: 生成多文件变更计划                              │  │
│  │                                                          │  │
│  │  [新建] src/middleware/auth.js      认证中间件            │  │
│  │  [修改] src/routes/api.js           添加认证保护          │  │
│  │  [修改] src/config/index.js         添加JWT配置           │  │
│  │  [修改] package.json                添加jsonwebtoken依赖  │  │
│  │  [新建] src/tests/auth.test.js      认证测试              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │                                     │
│                          v                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Step 4: 逐文件预览和确认                                │  │
│  │                                                          │  │
│  │  每个文件显示 diff 对比:                                  │  │
│  │  ┌────────────────────────────────────────────┐          │  │
│  │  │  - const express = require('express');      │          │  │
│  │  │  + const express = require('express');      │          │  │
│  │  │  + const { authenticate } = require(        │          │  │
│  │  │  +     '../middleware/auth'                  │          │  │
│  │  │  + );                                       │          │  │
│  │  └────────────────────────────────────────────┘          │  │
│  │                                                          │  │
│  │  [Accept All] [Accept File] [Reject] [Edit]              │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Composer 实战: 添加认证系统

```javascript
// ============================================================
// Composer 生成的文件1: src/middleware/auth.js
// ============================================================

const jwt = require('jsonwebtoken');
const { JWT_SECRET, JWT_EXPIRES_IN } = require('../config');

/**
 * JWT 认证中间件
 * 验证请求头中的 Bearer Token
 */
function authenticate(req, res, next) {
    // 从 Authorization 头获取 token
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return res.status(401).json({
            error: '未提供认证令牌',
            code: 'AUTH_TOKEN_MISSING'
        });
    }

    const token = authHeader.split(' ')[1];

    try {
        // 验证 token
        const decoded = jwt.verify(token, JWT_SECRET);
        req.user = decoded;  // 将用户信息附加到请求对象
        next();
    } catch (error) {
        if (error.name === 'TokenExpiredError') {
            return res.status(401).json({
                error: '令牌已过期',
                code: 'AUTH_TOKEN_EXPIRED'
            });
        }
        return res.status(403).json({
            error: '无效的认证令牌',
            code: 'AUTH_TOKEN_INVALID'
        });
    }
}

/**
 * 角色授权中间件
 * 检查用户是否拥有指定角色
 */
function authorize(...roles) {
    return (req, res, next) => {
        if (!req.user) {
            return res.status(401).json({ error: '未认证' });
        }
        if (!roles.includes(req.user.role)) {
            return res.status(403).json({
                error: '权限不足',
                required: roles,
                current: req.user.role
            });
        }
        next();
    };
}

/**
 * 生成 JWT 令牌
 */
function generateToken(payload) {
    return jwt.sign(payload, JWT_SECRET, { expiresIn: JWT_EXPIRES_IN });
}

module.exports = { authenticate, authorize, generateToken };
```

```javascript
// ============================================================
// Composer 生成的文件2: src/routes/auth.js
// ============================================================

const express = require('express');
const bcrypt = require('bcryptjs');
const { generateToken, authenticate } = require('../middleware/auth');
const router = express.Router();

// 模拟用户数据库
const users = new Map();

// POST /auth/register - 用户注册
router.post('/register', async (req, res) => {
    try {
        const { username, email, password } = req.body;

        // 参数验证
        if (!username || !email || !password) {
            return res.status(400).json({ error: '缺少必填字段' });
        }

        // 检查用户是否已存在
        if (users.has(email)) {
            return res.status(409).json({ error: '邮箱已注册' });
        }

        // 加密密码
        const salt = await bcrypt.genSalt(12);
        const hashedPassword = await bcrypt.hash(password, salt);

        // 存储用户
        const user = {
            id: users.size + 1,
            username,
            email,
            password: hashedPassword,
            role: 'user',
            createdAt: new Date()
        };
        users.set(email, user);

        // 生成令牌
        const token = generateToken({
            userId: user.id,
            email: user.email,
            role: user.role
        });

        res.status(201).json({
            message: '注册成功',
            token,
            user: { id: user.id, username, email, role: user.role }
        });
    } catch (error) {
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// POST /auth/login - 用户登录
router.post('/login', async (req, res) => {
    try {
        const { email, password } = req.body;

        const user = users.get(email);
        if (!user) {
            return res.status(401).json({ error: '邮箱或密码错误' });
        }

        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            return res.status(401).json({ error: '邮箱或密码错误' });
        }

        const token = generateToken({
            userId: user.id,
            email: user.email,
            role: user.role
        });

        res.json({
            message: '登录成功',
            token,
            user: { id: user.id, username: user.username, email, role: user.role }
        });
    } catch (error) {
        res.status(500).json({ error: '服务器内部错误' });
    }
});

// GET /auth/me - 获取当前用户信息(需要认证)
router.get('/me', authenticate, (req, res) => {
    const user = users.get(req.user.email);
    if (!user) {
        return res.status(404).json({ error: '用户不存在' });
    }
    res.json({
        id: user.id,
        username: user.username,
        email: user.email,
        role: user.role
    });
});

module.exports = router;
```

---

## 与GitHub Copilot对比

### 功能对比矩阵

```
┌────────────────────────────────────────────────────────────────────┐
│              Cursor vs GitHub Copilot 深度对比                      │
├────────────────┬────────────────────┬──────────────────────────────┤
│  对比维度       │  Cursor             │  GitHub Copilot              │
├────────────────┼────────────────────┼──────────────────────────────┤
│  产品形态       │  独立编辑器(VS Code  │  IDE 插件                    │
│                │  分叉)              │                              │
│  AI 模型       │  GPT-4o / Claude /  │  OpenAI Codex (专用)         │
│                │  可选模型           │                              │
│  代码补全       │  Tab 补全           │  Tab 补全 (更成熟)           │
│  内联编辑       │  Cmd+K (核心优势)   │  无                          │
│  Chat 功能     │  Ctrl+L (强)        │  Copilot Chat (强)           │
│  多文件编辑     │  Composer (核心优势)│  无                          │
│  代码库理解     │  @codebase 索引     │  @workspace (较弱)           │
│  外部文档       │  @doc URL引用       │  无                          │
│  联网搜索       │  @web 支持          │  无                          │
│  终端集成       │  Ctrl+K in Terminal │  @terminal                   │
│  隐私模式       │  支持               │  企业版支持                   │
│  价格           │  $20/月 (Pro)       │  $10/月 (Individual)         │
│  学习曲线       │  需要适应新编辑器    │  在现有IDE中使用             │
├────────────────┴────────────────────┴──────────────────────────────┤
│                                                                    │
│  选择建议:                                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  选 Cursor 如果:                                             │  │
│  │  - 需要频繁的代码重构和多文件修改                              │  │
│  │  - 想要更强的代码库级别理解                                    │  │
│  │  - 愿意尝试 AI 原生的编辑器体验                               │  │
│  │  - 需要引用外部文档生成代码                                    │  │
│  │                                                              │  │
│  │  选 Copilot 如果:                                            │  │
│  │  - 不想离开熟悉的 IDE (特别是JetBrains用户)                   │  │
│  │  - 主要需求是代码补全                                         │  │
│  │  - 已有 GitHub 企业版订阅                                     │  │
│  │  - 团队统一使用 Copilot 且有组织策略管控需求                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 高级技巧与工作流

### .cursorrules 项目规则

```
┌────────────────────────────────────────────────────────────────┐
│             .cursorrules 项目级AI规则配置                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  在项目根目录创建 .cursorrules 文件                             │
│  Cursor 会在每次 AI 请求时自动注入这些规则                      │
│  相当于给 AI 一份"项目编码规范手册"                             │
│                                                                │
│  作用:                                                         │
│  - 统一团队的AI输出风格                                        │
│  - 减少重复的Prompt说明                                        │
│  - 确保生成代码符合项目架构                                    │
│  - 强制安全和最佳实践                                          │
└────────────────────────────────────────────────────────────────┘
```

```markdown
<!-- 示例 .cursorrules 文件内容 -->

# 项目规范

## 技术栈
- 后端: Python 3.11 + FastAPI + SQLAlchemy 2.0
- 前端: React 18 + TypeScript + TailwindCSS
- 数据库: PostgreSQL 15
- 缓存: Redis

## 编码规范
- Python 使用 Google 风格的 docstring
- 所有函数必须有类型提示
- 使用 async/await 而非同步代码
- 错误处理使用自定义异常类, 不要用通用 Exception
- 变量命名使用 snake_case, 类名使用 PascalCase

## 项目结构
- API 路由放在 src/routes/ 目录
- 业务逻辑放在 src/services/ 目录
- 数据模型放在 src/models/ 目录
- 工具函数放在 src/utils/ 目录

## 安全要求
- 所有用户输入必须验证和清洗
- 数据库查询必须使用参数化查询
- 敏感配置必须从环境变量读取
- API 响应不得泄露内部实现细节

## 测试要求
- 每个 Service 函数都要有对应的单元测试
- 测试文件命名: test_[模块名].py
- 使用 pytest + pytest-asyncio
```

### 高效工作流模式

```
┌────────────────────────────────────────────────────────────────┐
│            Cursor 日常开发工作流                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  阶段1: 需求理解 (Chat)                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  "@codebase 当前项目的架构是什么?                         │  │
│  │   我需要添加一个通知系统, 应该如何设计?"                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          |                                     │
│                          v                                     │
│  阶段2: 代码生成 (Composer)                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  "根据刚才的设计方案, 创建通知服务相关的所有文件:           │  │
│  │   模型、服务层、路由、测试"                                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          |                                     │
│                          v                                     │
│  阶段3: 细节调整 (Cmd+K)                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  选中生成的代码, Cmd+K:                                   │  │
│  │  "添加邮件和短信两种通知渠道"                              │  │
│  │  "给这个函数添加限流逻辑, 每用户每分钟最多10条"            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          |                                     │
│                          v                                     │
│  阶段4: 审查优化 (Chat)                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  选中所有新增代码, Ctrl+L:                                │  │
│  │  "审查这段代码的安全性、性能和可维护性"                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 实战项目案例

### 用 Composer 搭建 REST API 脚手架

```python
# ============================================================
# Composer 指令:
# "创建一个完整的 FastAPI 项目结构, 包含:
#  1. 用户认证 (JWT)
#  2. 数据库模型 (SQLAlchemy)
#  3. CRUD 路由
#  4. 请求验证 (Pydantic)
#  5. 错误处理中间件
#  6. 日志配置"
#
# Composer 会一次性生成以下所有文件:
# ============================================================

# --- 文件: src/models/user.py ---
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from src.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


# --- 文件: src/schemas/user.py ---
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# --- 文件: src/middleware/error_handler.py ---
from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

class AppException(Exception):
    """自定义应用异常基类"""
    def __init__(self, status_code: int, detail: str, error_code: str = None):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code

class NotFoundError(AppException):
    def __init__(self, resource: str):
        super().__init__(404, f"{resource}不存在", "NOT_FOUND")

class ConflictError(AppException):
    def __init__(self, message: str):
        super().__init__(409, message, "CONFLICT")

class UnauthorizedError(AppException):
    def __init__(self, message: str = "未授权"):
        super().__init__(401, message, "UNAUTHORIZED")

async def app_exception_handler(request: Request, exc: AppException):
    logger.warning(f"AppException: {exc.detail} | Path: {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "code": exc.error_code,
            "path": str(request.url.path)
        }
    )
```

---

## 总结

本教程涵盖了 Cursor 编辑器的核心内容:

1. **核心优势**: AI 原生编辑器,三大核心功能（Cmd+K / Chat / Composer）深度融合
2. **安装配置**: 从 VS Code 无缝迁移,模型选择和隐私设置
3. **Cmd+K**: 最高频功能,无选中生成代码/选中修改代码/终端命令生成
4. **Chat**: @上下文引用系统（@codebase/@file/@doc/@web），Apply一键应用
5. **Composer**: 多文件编辑的核心武器,适合大规模代码变更和项目脚手架搭建
6. **与Copilot对比**: Cursor 在多文件编辑和代码库理解方面优势明显
7. **高级技巧**: .cursorrules 项目规则、四阶段工作流、上下文管理

## 参考资源

- [Cursor 官网](https://cursor.com)
- [Cursor 文档](https://docs.cursor.com)
- [Cursor Changelog](https://cursor.com/changelog)
- [.cursorrules 社区规则库](https://github.com/PatrickJS/awesome-cursorrules)
- [Cursor vs Copilot 对比分析](https://cursor.com/blog)

---

**创建时间**: 2024-01-15
**最后更新**: 2024-01-15
