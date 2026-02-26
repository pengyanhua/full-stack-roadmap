/**
 * ============================================================
 *              Next.js Server Actions
 * ============================================================
 * 本文件介绍 Next.js 中 Server Actions 的使用方式。
 *
 * Server Actions 允许在服务端直接执行函数，无需手动创建 API 路由。
 * 它们是处理表单提交、数据变更的推荐方式。
 *
 * 核心概念：
 * - 'use server' 指令标记服务端函数
 * - 可直接作为 form action 使用
 * - 与 React 19 的 useActionState、useOptimistic 深度集成
 * ============================================================
 */

import { revalidatePath, revalidateTag } from 'next/cache';
import { redirect } from 'next/navigation';

// ============================================================
//               1. Server Actions 基础
// ============================================================

/**
 * 【Server Actions — 服务端操作】
 *
 * 通过 'use server' 指令标记：
 * - 文件顶部 'use server'：所有导出函数都是 Server Action
 * - 函数体内 'use server'：单个函数标记为 Server Action
 *
 * 工作原理：
 * 1. 编译时为每个 Server Action 生成唯一端点
 * 2. 客户端调用时自动发送 POST 请求
 * 3. 服务端执行函数，返回结果
 */

// --- 方式一：在服务端组件中内联定义 ---
async function InlineActionPage() {
    async function createItem(formData: FormData) {
        'use server';
        const name = formData.get('name') as string;
        // await db.item.create({ data: { name } });
        revalidatePath('/items');
    }

    return (
        <form action={createItem}>
            <input name="name" placeholder="名称" />
            <button type="submit">创建</button>
        </form>
    );
}

// --- 方式二：独立文件定义（推荐用于客户端组件）---
// app/actions.ts
// 'use server';   // 文件顶部标记，所有导出都是 Server Action

export async function createUser(formData: FormData) {
    'use server';
    const name = formData.get('name') as string;
    const email = formData.get('email') as string;
    // await db.user.create({ data: { name, email } });
    revalidatePath('/users');
}

// --- 可传递的参数类型 ---
// 支持：string, number, boolean, Date, FormData, null, undefined, 普通对象/数组
// 不支持：函数、类实例、Symbol、DOM 节点


// ============================================================
//               2. 表单处理
// ============================================================

/**
 * 【表单处理 — Form Handling】
 *
 * Server Actions 最自然的使用场景是表单处理：
 * - form action 直接绑定 Server Action
 * - FormData 自动传入
 * - 支持渐进增强：即使 JS 未加载，表单也能提交
 * - 无需 preventDefault / fetch / axios
 */

async function BasicFormPage() {
    async function handleSubmit(formData: FormData) {
        'use server';
        const title = formData.get('title') as string;
        const content = formData.get('content') as string;
        // await db.post.create({ data: { title, content } });
        revalidatePath('/posts');
        redirect('/posts');    // 提交后重定向
    }

    return (
        <form action={handleSubmit}>
            <input type="text" name="title" required />
            <select name="category">
                <option value="tech">技术</option>
                <option value="life">生活</option>
            </select>
            <textarea name="content" rows={5} required />
            <button type="submit">发布文章</button>
        </form>
    );
}

// --- 使用 bind 传递额外参数 ---
async function EditFormPage({ postId }: { postId: string }) {
    async function updatePost(id: string, formData: FormData) {
        'use server';
        const title = formData.get('title') as string;
        // await db.post.update({ where: { id }, data: { title } });
        revalidatePath(`/posts/${id}`);
    }

    // bind 将 postId 绑定到第一个参数
    const updatePostWithId = updatePost.bind(null, postId);

    return (
        <form action={updatePostWithId}>
            <input name="title" defaultValue="原始标题" />
            <button type="submit">更新</button>
        </form>
    );
}


// ============================================================
//               3. useActionState
// ============================================================

/**
 * 【useActionState — 表单状态管理】
 *
 * React 19 中的 useActionState（原 useFormState，已更名）：
 * - 跟踪 Server Action 的返回状态
 * - 管理 pending 状态
 *
 * 签名：
 *   const [state, formAction, isPending] = useActionState(action, initialState);
 *
 *   action       — Server Action（第一个参数是 prevState）
 *   state        — 当前状态（Action 的返回值）
 *   formAction   — 绑定到 form action 的函数
 *   isPending    — 是否正在提交
 */

type FormState = {
    success: boolean;
    message: string;
    errors?: Record<string, string[]>;
};

// --- 带状态返回的 Server Action ---
async function createAccount(prevState: FormState, formData: FormData): Promise<FormState> {
    'use server';
    const username = formData.get('username') as string;

    if (!username || username.length < 3) {
        return {
            success: false,
            message: '验证失败',
            errors: { username: ['用户名至少 3 个字符'] },
        };
    }

    try {
        // await db.account.create({ data: { username } });
        revalidatePath('/accounts');
        return { success: true, message: '账户创建成功！' };
    } catch {
        return { success: false, message: '创建失败，请稍后重试' };
    }
}

// --- 客户端组件中使用 useActionState ---
// 'use client';
// import { useActionState } from 'react';

function CreateAccountForm() {
    // const [state, formAction, isPending] = useActionState(createAccount, {
    //     success: false, message: '',
    // });
    // return (
    //     <form action={formAction}>
    //         {state.message && (
    //             <div className={state.success ? 'success' : 'error'}>
    //                 {state.message}
    //             </div>
    //         )}
    //         <input name="username" placeholder="用户名" />
    //         {state.errors?.username && (
    //             <span className="error">{state.errors.username[0]}</span>
    //         )}
    //         <button type="submit" disabled={isPending}>
    //             {isPending ? '创建中...' : '创建账户'}
    //         </button>
    //     </form>
    // );
    return null;  // 占位：实际需在 'use client' 文件中使用
}


// ============================================================
//               4. useFormStatus
// ============================================================

/**
 * 【useFormStatus — 提交状态】
 *
 * 获取父级 form 的提交状态，必须在 form 的子组件中使用。
 *
 * 签名：
 *   const { pending, data, method, action } = useFormStatus();
 *
 * 典型用法：创建可复用的提交按钮组件
 */

// 'use client';
// import { useFormStatus } from 'react-dom';

// --- 通用提交按钮 ---
function SubmitButton({ children }: { children: React.ReactNode }) {
    // const { pending } = useFormStatus();
    // return (
    //     <button type="submit" disabled={pending}>
    //         {pending ? '提交中...' : children}
    //     </button>
    // );
    return null;
}

// --- 显示提交详情 ---
function FormStatusDisplay() {
    // const { pending, data, method } = useFormStatus();
    // if (!pending) return null;
    // return (
    //     <div className="status">
    //         <p>正在通过 {method} 方法提交...</p>
    //         <p>提交的邮箱：{data?.get('email')}</p>
    //     </div>
    // );
    return null;
}


// ============================================================
//               5. 数据验证
// ============================================================

/**
 * 【数据验证 — Zod Schema Validation】
 *
 * Server Actions 中应始终验证输入数据：
 * - 永远不要信任客户端传来的数据
 * - 推荐使用 Zod 进行类型安全的验证
 * - 验证失败时返回结构化错误信息
 */

// import { z } from 'zod';
// const CreateProductSchema = z.object({
//     name: z.string().min(2, '商品名至少 2 字').max(100),
//     price: z.coerce.number().positive('价格必须为正数'),
//     category: z.enum(['electronics', 'clothing', 'food']),
// });

async function createProduct(prevState: FormState, formData: FormData): Promise<FormState> {
    'use server';

    const rawData = {
        name: formData.get('name'),
        price: formData.get('price'),
        category: formData.get('category'),
    };

    // const result = CreateProductSchema.safeParse(rawData);
    // if (!result.success) {
    //     return { success: false, message: '验证失败',
    //              errors: result.error.flatten().fieldErrors };
    // }
    // await db.product.create({ data: result.data });

    revalidatePath('/products');
    return { success: true, message: '商品创建成功！' };
}


// ============================================================
//               6. 错误处理
// ============================================================

/**
 * 【错误处理 — Error Handling】
 *
 * 错误处理策略：
 * - 预期错误（验证失败）→ 返回错误状态
 * - 意外错误（数据库故障）→ try/catch 捕获
 * - 安全原则：不要将内部错误详情暴露给客户端
 */

type ActionResult = {
    success: boolean;
    message: string;
    errors?: Record<string, string[]>;
};

async function updateProfile(
    prevState: ActionResult,
    formData: FormData
): Promise<ActionResult> {
    'use server';

    try {
        // 1. 认证检查
        // const session = await getSession();
        // if (!session) return { success: false, message: '请先登录' };

        // 2. 数据验证
        const name = formData.get('name') as string;
        if (!name?.trim()) {
            return { success: false, message: '验证失败', errors: { name: ['姓名不能为空'] } };
        }

        // 3. 执行更新
        // await db.user.update({ where: { id: session.user.id }, data: { name } });
        revalidatePath('/profile');
        return { success: true, message: '个人资料已更新' };

    } catch (error) {
        console.error('更新资料失败:', error);
        // 不要将原始错误信息返回给客户端
        return { success: false, message: '服务器内部错误，请稍后重试' };
    }
}

// --- redirect 必须放在 try/catch 之外 ---
async function actionWithRedirect(formData: FormData) {
    'use server';

    try {
        // await db.item.create({ ... });
    } catch (error) {
        return { success: false, message: '操作失败' };
    }

    // redirect 通过抛出异常实现，不能在 try 块内调用
    redirect('/success');
}


// ============================================================
//               7. 乐观更新
// ============================================================

/**
 * 【乐观更新 — Optimistic Updates】
 *
 * useOptimistic 在 Server Action 完成前乐观地更新 UI：
 * - 用户操作后立即显示预期结果
 * - 失败时自动回滚
 *
 * 签名：
 *   const [optimisticState, addOptimistic] = useOptimistic(
 *       currentState,
 *       (current, optimisticValue) => newState
 *   );
 */

// 'use client';
// import { useOptimistic } from 'react';

type Todo = { id: string; text: string; completed: boolean; pending?: boolean };

// --- 乐观添加待办 ---
function OptimisticTodoList({ todos }: { todos: Todo[] }) {
    // const [optimisticTodos, addOptimistic] = useOptimistic(
    //     todos,
    //     (current: Todo[], newTodo: Todo) => [...current, { ...newTodo, pending: true }]
    // );
    // async function handleAdd(formData: FormData) {
    //     addOptimistic({ id: crypto.randomUUID(), text: formData.get('text'), completed: false });
    //     await addTodoAction(formData.get('text') as string);
    // }
    // return (
    //     <form action={handleAdd}>
    //         <input name="text" />
    //         <button>添加</button>
    //         <ul>{optimisticTodos.map(t => (
    //             <li key={t.id} style={{ opacity: t.pending ? 0.5 : 1 }}>{t.text}</li>
    //         ))}</ul>
    //     </form>
    // );
    return null;
}

// --- 乐观点赞 ---
function OptimisticLikeButton({ likes, postId }: { likes: number; postId: string }) {
    // const [optimisticLikes, addLike] = useOptimistic(likes, (c, i: number) => c + i);
    // return <button onClick={() => { addLike(1); likePostAction(postId); }}>
    //     点赞 ({optimisticLikes})
    // </button>;
    return null;
}

async function likePostAction(postId: string) {
    'use server';
    // await db.post.update({ where: { id: postId }, data: { likes: { increment: 1 } } });
    revalidateTag('posts');
}


// ============================================================
//               8. 最佳实践
// ============================================================

/**
 * 【Server Actions 最佳实践】
 *
 * ✅ 推荐做法：
 * - 将 Server Actions 放在独立 actions.ts 文件中统一管理
 * - 始终验证输入数据，推荐 Zod 做 schema 验证
 * - 使用 useActionState 管理表单状态和错误反馈
 * - 操作完成后调用 revalidatePath/revalidateTag 刷新缓存
 * - 使用 try/catch 处理意外错误
 * - 敏感操作前检查用户认证和权限
 * - 使用 useOptimistic 提升交互体验
 *
 * ❌ 避免做法：
 * - 避免返回敏感信息（密码哈希、内部错误详情）
 * - 避免将 redirect 写在 try 块内
 * - 避免直接信任 FormData，始终做服务端验证
 * - 避免在服务端组件中使用 useActionState（它是客户端 hook）
 * - 避免执行耗时过长的任务（考虑队列）
 *
 * 【Server Actions vs Route Handlers】
 *
 *   表单提交/数据变更       → Server Actions
 *   RESTful API / Webhook  → Route Handlers
 *   自定义 HTTP 响应头      → Route Handlers
 *   大文件上传/流式响应      → Route Handlers
 *
 * 【文件组织推荐】
 *
 *   app/
 *   ├── actions/           # 集中管理 Server Actions
 *   │   ├── auth.ts
 *   │   ├── posts.ts
 *   │   └── users.ts
 *   ├── lib/
 *   │   └── validations.ts # Zod schema 定义
 *   └── (routes)/
 *       └── page.tsx        # 页面组件引用 actions
 */
