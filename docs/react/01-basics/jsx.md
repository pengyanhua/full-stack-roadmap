# jsx.jsx

::: info 文件信息
- 📄 原文件：`01_jsx.jsx`
- 🔤 语言：jsx
:::

React JSX 基础
JSX 是 JavaScript 的语法扩展，用于描述 UI 结构。
它看起来像 HTML，但实际上会被编译成 JavaScript。

## 完整代码

```jsx
/**
 * ============================================================
 *                    React JSX 基础
 * ============================================================
 * JSX 是 JavaScript 的语法扩展，用于描述 UI 结构。
 * 它看起来像 HTML，但实际上会被编译成 JavaScript。
 * ============================================================
 */

import React from 'react';

// ============================================================
//                    1. JSX 基础语法
// ============================================================

/**
 * 【什么是 JSX】
 *
 * JSX = JavaScript + XML
 * - 不是模板语言，而是 JavaScript 的语法扩展
 * - 会被 Babel 编译成 React.createElement() 调用
 * - 可以在 JSX 中使用任意 JavaScript 表达式
 *
 * 【编译示例】
 * JSX:
 *   <div className="hello">Hello</div>
 *
 * 编译后:
 *   React.createElement('div', { className: 'hello' }, 'Hello')
 */

// --- 基本元素 ---
function BasicElement() {
    // 单个元素
    return <h1>Hello, React!</h1>;
}

// --- 嵌套元素 ---
function NestedElements() {
    // 多个元素必须有一个父元素包裹
    return (
        <div>
            <h1>标题</h1>
            <p>段落内容</p>
        </div>
    );
}

// --- Fragment 片段 ---
function FragmentExample() {
    // 使用 Fragment 避免额外的 DOM 节点
    return (
        <>
            <h1>标题</h1>
            <p>段落内容</p>
        </>
    );

    // 或者使用完整写法（可以添加 key）
    // return (
    //     <React.Fragment key="unique">
    //         <h1>标题</h1>
    //         <p>段落内容</p>
    //     </React.Fragment>
    // );
}


// ============================================================
//                    2. JSX 中使用 JavaScript
// ============================================================

/**
 * 【在 JSX 中嵌入表达式】
 *
 * 使用花括号 {} 可以在 JSX 中嵌入任意 JavaScript 表达式
 * - 变量
 * - 函数调用
 * - 算术运算
 * - 三元表达式
 * - 数组方法
 */

// --- 变量和表达式 ---
function VariablesAndExpressions() {
    const name = 'Alice';
    const age = 25;
    const isAdult = age >= 18;

    return (
        <div>
            {/* 变量 */}
            <p>姓名: {name}</p>

            {/* 表达式 */}
            <p>年龄: {age}</p>
            <p>明年年龄: {age + 1}</p>

            {/* 函数调用 */}
            <p>大写姓名: {name.toUpperCase()}</p>

            {/* 三元表达式 */}
            <p>状态: {isAdult ? '成年' : '未成年'}</p>

            {/* 模板字符串 */}
            <p>{`${name} 今年 ${age} 岁`}</p>
        </div>
    );
}

// --- 条件渲染 ---
function ConditionalRendering() {
    const isLoggedIn = true;
    const user = { name: 'Alice', role: 'admin' };
    const messages = ['消息1', '消息2', '消息3'];

    return (
        <div>
            {/* 三元表达式 */}
            {isLoggedIn ? <p>欢迎回来!</p> : <p>请登录</p>}

            {/* && 短路运算 - 只渲染真值 */}
            {isLoggedIn && <p>你已登录</p>}

            {/* 多条件判断 */}
            {user.role === 'admin' && <button>管理面板</button>}
            {user.role === 'user' && <button>用户中心</button>}

            {/* 显示消息数量 */}
            {messages.length > 0 && (
                <p>你有 {messages.length} 条新消息</p>
            )}
        </div>
    );
}

// --- 复杂条件渲染 ---
function ComplexConditional() {
    const status = 'loading'; // 'loading' | 'success' | 'error'

    // 使用函数处理复杂条件
    const renderContent = () => {
        switch (status) {
            case 'loading':
                return <p>加载中...</p>;
            case 'success':
                return <p>加载成功!</p>;
            case 'error':
                return <p>加载失败</p>;
            default:
                return null;
        }
    };

    return (
        <div>
            {renderContent()}
        </div>
    );
}


// ============================================================
//                    3. 列表渲染
// ============================================================

/**
 * 【列表渲染】
 *
 * 使用数组的 map() 方法将数据数组转换为 JSX 元素数组
 *
 * 【key 的作用】
 * - 帮助 React 识别哪些元素改变了
 * - key 应该是稳定、唯一的标识符
 * - 不推荐使用数组索引作为 key（除非列表不会重新排序）
 */

// --- 基本列表渲染 ---
function BasicList() {
    const fruits = ['苹果', '香蕉', '橙子', '葡萄'];

    return (
        <ul>
            {fruits.map((fruit, index) => (
                // 简单列表可以用索引作为 key
                <li key={index}>{fruit}</li>
            ))}
        </ul>
    );
}

// --- 对象列表渲染 ---
function ObjectList() {
    const users = [
        { id: 1, name: 'Alice', email: 'alice@example.com' },
        { id: 2, name: 'Bob', email: 'bob@example.com' },
        { id: 3, name: 'Charlie', email: 'charlie@example.com' },
    ];

    return (
        <div>
            <h2>用户列表</h2>
            <ul>
                {users.map(user => (
                    // 使用唯一 id 作为 key
                    <li key={user.id}>
                        <strong>{user.name}</strong>
                        <span> - {user.email}</span>
                    </li>
                ))}
            </ul>
        </div>
    );
}

// --- 嵌套列表渲染 ---
function NestedList() {
    const categories = [
        {
            id: 1,
            name: '电子产品',
            items: ['手机', '电脑', '平板'],
        },
        {
            id: 2,
            name: '服装',
            items: ['T恤', '牛仔裤', '外套'],
        },
    ];

    return (
        <div>
            {categories.map(category => (
                <div key={category.id}>
                    <h3>{category.name}</h3>
                    <ul>
                        {category.items.map((item, index) => (
                            <li key={`${category.id}-${index}`}>{item}</li>
                        ))}
                    </ul>
                </div>
            ))}
        </div>
    );
}

// --- 过滤和排序列表 ---
function FilteredList() {
    const products = [
        { id: 1, name: 'iPhone', price: 999, inStock: true },
        { id: 2, name: 'MacBook', price: 1999, inStock: false },
        { id: 3, name: 'iPad', price: 799, inStock: true },
        { id: 4, name: 'AirPods', price: 199, inStock: true },
    ];

    // 过滤有库存的商品
    const inStockProducts = products.filter(p => p.inStock);

    // 按价格排序
    const sortedProducts = [...products].sort((a, b) => a.price - b.price);

    return (
        <div>
            <h3>有库存商品</h3>
            <ul>
                {inStockProducts.map(product => (
                    <li key={product.id}>
                        {product.name} - ${product.price}
                    </li>
                ))}
            </ul>

            <h3>按价格排序</h3>
            <ul>
                {sortedProducts.map(product => (
                    <li key={product.id}>
                        {product.name} - ${product.price}
                    </li>
                ))}
            </ul>
        </div>
    );
}


// ============================================================
//                    4. JSX 属性
// ============================================================

/**
 * 【JSX 属性规则】
 *
 * 1. 使用 camelCase 命名
 *    - class → className
 *    - for → htmlFor
 *    - tabindex → tabIndex
 *
 * 2. 布尔属性
 *    - disabled={true} 可简写为 disabled
 *    - 显式传 false: disabled={false}
 *
 * 3. 展开属性
 *    - {...props} 将对象的所有属性展开
 */

// --- className 和 style ---
function StylingExample() {
    const isActive = true;
    const customStyle = {
        color: 'blue',
        fontSize: '18px',      // 注意: camelCase
        backgroundColor: '#f0f0f0',
        padding: '10px',
    };

    return (
        <div>
            {/* className */}
            <p className="text-bold">粗体文字</p>

            {/* 动态 className */}
            <p className={isActive ? 'active' : 'inactive'}>
                动态类名
            </p>

            {/* 多个类名 */}
            <p className={`base-class ${isActive ? 'active' : ''}`}>
                多个类名
            </p>

            {/* 内联样式（对象形式） */}
            <p style={customStyle}>内联样式</p>

            {/* 直接写内联样式 */}
            <p style={{ color: 'red', fontWeight: 'bold' }}>
                直接内联样式
            </p>
        </div>
    );
}

// --- 属性展开 ---
function PropsSpread() {
    const buttonProps = {
        type: 'submit',
        className: 'btn btn-primary',
        disabled: false,
        onClick: () => console.log('clicked'),
    };

    const inputProps = {
        type: 'text',
        placeholder: '请输入...',
        maxLength: 100,
    };

    return (
        <div>
            {/* 展开所有属性 */}
            <button {...buttonProps}>提交</button>

            {/* 展开并覆盖某些属性 */}
            <button {...buttonProps} disabled={true}>
                禁用按钮
            </button>

            {/* 输入框 */}
            <input {...inputProps} />
        </div>
    );
}

// --- 特殊属性 ---
function SpecialAttributes() {
    return (
        <div>
            {/* htmlFor 代替 for */}
            <label htmlFor="username">用户名:</label>
            <input id="username" type="text" />

            {/* tabIndex */}
            <button tabIndex={1}>第一个</button>
            <button tabIndex={2}>第二个</button>

            {/* data-* 属性 */}
            <div data-testid="test-element" data-user-id="123">
                自定义数据属性
            </div>

            {/* aria-* 无障碍属性 */}
            <button aria-label="关闭" aria-pressed="false">
                ×
            </button>
        </div>
    );
}


// ============================================================
//                    5. JSX 中的注释
// ============================================================

function CommentsExample() {
    return (
        <div>
            {/* 这是 JSX 中的注释 */}

            {/*
                多行注释
                也是这样写
            */}

            <p>内容</p>

            {/* 注释掉元素
            <p>这个元素被注释了</p>
            */}
        </div>
    );
}


// ============================================================
//                    6. JSX 防止注入攻击
// ============================================================

/**
 * 【XSS 防护】
 *
 * React DOM 在渲染前会对所有值进行转义
 * 这可以防止 XSS（跨站脚本）攻击
 *
 * 如果确实需要渲染 HTML，使用 dangerouslySetInnerHTML
 * 但要确保内容是安全的
 */

function XSSPrevention() {
    // 恶意输入会被转义，不会执行
    const userInput = '<script>alert("xss")</script>';

    // 安全的 HTML 内容（来自可信源）
    const safeHTML = '<strong>加粗文字</strong>';

    return (
        <div>
            {/* 自动转义，安全 */}
            <p>{userInput}</p>

            {/*
                危险！只在确保内容安全时使用
                比如：内容来自 CMS 的富文本编辑器
            */}
            <div dangerouslySetInnerHTML={{ __html: safeHTML }} />
        </div>
    );
}


// ============================================================
//                    7. JSX 最佳实践
// ============================================================

/**
 * 【最佳实践】
 *
 * 1. 保持 JSX 简洁
 *    - 复杂逻辑提取到函数或变量
 *    - 避免过深的嵌套
 *
 * 2. 组件拆分
 *    - 一个组件只做一件事
 *    - 可复用的部分提取为独立组件
 *
 * 3. 条件渲染
 *    - 简单条件用 && 或三元表达式
 *    - 复杂条件用函数或提前返回
 *
 * 4. key 的使用
 *    - 使用稳定唯一的标识符
 *    - 避免使用数组索引（除非静态列表）
 */

// --- 好的实践 ---
function GoodPractice() {
    const items = [
        { id: 1, name: 'Item 1', completed: false },
        { id: 2, name: 'Item 2', completed: true },
    ];

    // 复杂逻辑提取到函数
    const renderItem = (item) => (
        <li key={item.id} className={item.completed ? 'completed' : ''}>
            {item.name}
        </li>
    );

    // 条件提前计算
    const completedCount = items.filter(i => i.completed).length;
    const hasCompleted = completedCount > 0;

    return (
        <div>
            <h2>待办列表</h2>
            {hasCompleted && <p>已完成: {completedCount}</p>}
            <ul>{items.map(renderItem)}</ul>
        </div>
    );
}

// --- 避免的写法 ---
function AvoidThis() {
    const data = [1, 2, 3];

    return (
        <div>
            {/* ❌ 避免: 过于复杂的内联逻辑 */}
            {data.length > 0 ? (
                data.filter(x => x > 1).map(x => (
                    <span key={x}>{x * 2}</span>
                ))
            ) : (
                <p>无数据</p>
            )}

            {/* ✅ 推荐: 提取逻辑 */}
            {/* 见上面 GoodPractice 的写法 */}
        </div>
    );
}


// ============================================================
//                    导出示例组件
// ============================================================

export {
    BasicElement,
    NestedElements,
    FragmentExample,
    VariablesAndExpressions,
    ConditionalRendering,
    ComplexConditional,
    BasicList,
    ObjectList,
    NestedList,
    FilteredList,
    StylingExample,
    PropsSpread,
    SpecialAttributes,
    CommentsExample,
    XSSPrevention,
    GoodPractice,
};

export default function JSXTutorial() {
    return (
        <div className="tutorial">
            <h1>JSX 基础教程</h1>
            <BasicElement />
            <VariablesAndExpressions />
            <ConditionalRendering />
            <ObjectList />
            <StylingExample />
        </div>
    );
}
```
