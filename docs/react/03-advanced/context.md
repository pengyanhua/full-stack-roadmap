# context.jsx

::: info 文件信息
- 📄 原文件：`01_context.jsx`
- 🔤 语言：jsx
:::

React Context 上下文
Context 提供了一种在组件树中共享数据的方式，无需手动逐层传递 props。
适用于全局状态如主题、用户信息、语言偏好等。

## 完整代码

```jsx
/**
 * ============================================================
 *                    React Context 上下文
 * ============================================================
 * Context 提供了一种在组件树中共享数据的方式，无需手动逐层传递 props。
 * 适用于全局状态如主题、用户信息、语言偏好等。
 * ============================================================
 */

import React, { createContext, useContext, useState, useReducer } from 'react';

// ============================================================
//                    1. Context 基础
// ============================================================

/**
 * 【Context 解决什么问题】
 *
 * Props Drilling（属性逐层传递）问题：
 * - 当深层组件需要数据时，需要通过中间组件逐层传递
 * - 中间组件可能并不需要这些数据
 * - 代码冗余且难以维护
 *
 * Context 提供了一种"广播"机制：
 * - 创建一个 Context 对象
 * - 使用 Provider 包裹组件树
 * - 任意深层组件都可以直接访问
 */

// --- 创建 Context ---
// createContext 接收一个默认值，当组件不在 Provider 内时使用
const ThemeContext = createContext('light');

// --- Provider 提供数据 ---
function ThemeProvider({ children }) {
    const [theme, setTheme] = useState('light');

    // 切换主题
    const toggleTheme = () => {
        setTheme(prev => prev === 'light' ? 'dark' : 'light');
    };

    // value 属性提供给所有子组件
    return (
        <ThemeContext.Provider value={{ theme, toggleTheme }}>
            {children}
        </ThemeContext.Provider>
    );
}

// --- Consumer 消费数据 ---
// 方式1: useContext Hook（推荐）
function ThemedButton() {
    // 使用 useContext 获取 Context 值
    const { theme, toggleTheme } = useContext(ThemeContext);

    return (
        <button
            onClick={toggleTheme}
            style={{
                background: theme === 'light' ? '#fff' : '#333',
                color: theme === 'light' ? '#333' : '#fff',
                padding: '10px 20px',
                border: '1px solid #ccc',
            }}
        >
            当前主题: {theme} (点击切换)
        </button>
    );
}

// 方式2: Consumer 组件（类组件或需要渲染函数时使用）
function ThemedText() {
    return (
        <ThemeContext.Consumer>
            {({ theme }) => (
                <p style={{
                    color: theme === 'light' ? '#333' : '#fff',
                    background: theme === 'light' ? '#fff' : '#333',
                }}>
                    这是 {theme} 主题的文字
                </p>
            )}
        </ThemeContext.Consumer>
    );
}


// ============================================================
//                    2. 完整的 Context 示例
// ============================================================

/**
 * 【用户认证 Context 示例】
 *
 * 一个完整的认证系统包括：
 * - 用户状态
 * - 登录/登出方法
 * - 加载状态
 */

// 创建 AuthContext
const AuthContext = createContext(null);

// 自定义 Hook - 简化使用
function useAuth() {
    const context = useContext(AuthContext);

    // 检查是否在 Provider 内使用
    if (context === null) {
        throw new Error('useAuth 必须在 AuthProvider 内使用');
    }

    return context;
}

// AuthProvider 组件
function AuthProvider({ children }) {
    // 用户状态
    const [user, setUser] = useState(null);
    // 加载状态
    const [loading, setLoading] = useState(true);

    // 模拟检查登录状态
    React.useEffect(() => {
        const checkAuth = async () => {
            // 模拟 API 请求
            await new Promise(r => setTimeout(r, 500));

            // 检查本地存储
            const savedUser = localStorage.getItem('user');
            if (savedUser) {
                setUser(JSON.parse(savedUser));
            }

            setLoading(false);
        };

        checkAuth();
    }, []);

    // 登录方法
    const login = async (username, password) => {
        setLoading(true);

        // 模拟登录 API
        await new Promise(r => setTimeout(r, 1000));

        // 模拟验证
        if (username === 'admin' && password === '123456') {
            const userData = {
                id: 1,
                username,
                role: 'admin',
            };

            setUser(userData);
            localStorage.setItem('user', JSON.stringify(userData));
            setLoading(false);
            return { success: true };
        }

        setLoading(false);
        return { success: false, error: '用户名或密码错误' };
    };

    // 登出方法
    const logout = () => {
        setUser(null);
        localStorage.removeItem('user');
    };

    // 提供的值
    const value = {
        user,
        loading,
        login,
        logout,
        isAuthenticated: !!user,
    };

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
}

// --- 使用 AuthContext 的组件 ---
function LoginForm() {
    const { login, loading } = useAuth();
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');

        const result = await login(username, password);
        if (!result.success) {
            setError(result.error);
        }
    };

    return (
        <form onSubmit={handleSubmit}>
            <h2>登录</h2>
            {error && <p className="error">{error}</p>}
            <div>
                <input
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder="用户名 (admin)"
                    disabled={loading}
                />
            </div>
            <div>
                <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="密码 (123456)"
                    disabled={loading}
                />
            </div>
            <button type="submit" disabled={loading}>
                {loading ? '登录中...' : '登录'}
            </button>
        </form>
    );
}

function UserProfile() {
    const { user, logout, isAuthenticated } = useAuth();

    if (!isAuthenticated) {
        return <p>请先登录</p>;
    }

    return (
        <div>
            <h2>用户资料</h2>
            <p>用户名: {user.username}</p>
            <p>角色: {user.role}</p>
            <button onClick={logout}>退出登录</button>
        </div>
    );
}


// ============================================================
//                    3. 多个 Context 组合
// ============================================================

/**
 * 【组合多个 Context】
 *
 * 实际应用中可能需要多个 Context：
 * - 主题 Context
 * - 用户 Context
 * - 语言 Context
 * - 等等
 *
 * 可以通过嵌套 Provider 来组合
 */

// 语言 Context
const LanguageContext = createContext({
    language: 'zh',
    setLanguage: () => {},
});

function useLanguage() {
    return useContext(LanguageContext);
}

function LanguageProvider({ children }) {
    const [language, setLanguage] = useState('zh');

    const translations = {
        zh: {
            hello: '你好',
            welcome: '欢迎',
            logout: '退出',
        },
        en: {
            hello: 'Hello',
            welcome: 'Welcome',
            logout: 'Logout',
        },
    };

    const t = (key) => translations[language][key] || key;

    return (
        <LanguageContext.Provider value={{ language, setLanguage, t }}>
            {children}
        </LanguageContext.Provider>
    );
}

// --- 组合 Provider ---
function AppProviders({ children }) {
    return (
        <ThemeProvider>
            <LanguageProvider>
                <AuthProvider>
                    {children}
                </AuthProvider>
            </LanguageProvider>
        </ThemeProvider>
    );
}

// --- 使用多个 Context ---
function Header() {
    const { theme, toggleTheme } = useContext(ThemeContext);
    const { language, setLanguage, t } = useLanguage();
    const { user, logout, isAuthenticated } = useAuth();

    return (
        <header style={{
            background: theme === 'light' ? '#fff' : '#333',
            color: theme === 'light' ? '#333' : '#fff',
            padding: '10px',
        }}>
            <span>{t('welcome')}!</span>

            <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
            >
                <option value="zh">中文</option>
                <option value="en">English</option>
            </select>

            <button onClick={toggleTheme}>
                切换主题
            </button>

            {isAuthenticated && (
                <>
                    <span>{user.username}</span>
                    <button onClick={logout}>{t('logout')}</button>
                </>
            )}
        </header>
    );
}


// ============================================================
//                    4. Context 与 useReducer 结合
// ============================================================

/**
 * 【Context + useReducer】
 *
 * 对于复杂状态管理，可以结合 useReducer：
 * - useReducer 管理状态逻辑
 * - Context 提供全局访问
 *
 * 这是一个轻量级的 Redux 替代方案
 */

// 定义 actions
const CART_ACTIONS = {
    ADD_ITEM: 'ADD_ITEM',
    REMOVE_ITEM: 'REMOVE_ITEM',
    UPDATE_QUANTITY: 'UPDATE_QUANTITY',
    CLEAR_CART: 'CLEAR_CART',
};

// 定义 reducer
function cartReducer(state, action) {
    switch (action.type) {
        case CART_ACTIONS.ADD_ITEM: {
            const existingItem = state.items.find(
                item => item.id === action.payload.id
            );

            if (existingItem) {
                return {
                    ...state,
                    items: state.items.map(item =>
                        item.id === action.payload.id
                            ? { ...item, quantity: item.quantity + 1 }
                            : item
                    ),
                };
            }

            return {
                ...state,
                items: [...state.items, { ...action.payload, quantity: 1 }],
            };
        }

        case CART_ACTIONS.REMOVE_ITEM:
            return {
                ...state,
                items: state.items.filter(item => item.id !== action.payload),
            };

        case CART_ACTIONS.UPDATE_QUANTITY:
            return {
                ...state,
                items: state.items.map(item =>
                    item.id === action.payload.id
                        ? { ...item, quantity: action.payload.quantity }
                        : item
                ).filter(item => item.quantity > 0),
            };

        case CART_ACTIONS.CLEAR_CART:
            return { ...state, items: [] };

        default:
            return state;
    }
}

// 创建 Context
const CartContext = createContext(null);

// 自定义 Hook
function useCart() {
    const context = useContext(CartContext);
    if (!context) {
        throw new Error('useCart 必须在 CartProvider 内使用');
    }
    return context;
}

// CartProvider
function CartProvider({ children }) {
    const [state, dispatch] = useReducer(cartReducer, { items: [] });

    // 封装操作方法
    const addItem = (item) => {
        dispatch({ type: CART_ACTIONS.ADD_ITEM, payload: item });
    };

    const removeItem = (id) => {
        dispatch({ type: CART_ACTIONS.REMOVE_ITEM, payload: id });
    };

    const updateQuantity = (id, quantity) => {
        dispatch({
            type: CART_ACTIONS.UPDATE_QUANTITY,
            payload: { id, quantity },
        });
    };

    const clearCart = () => {
        dispatch({ type: CART_ACTIONS.CLEAR_CART });
    };

    // 计算属性
    const totalItems = state.items.reduce(
        (sum, item) => sum + item.quantity,
        0
    );

    const totalPrice = state.items.reduce(
        (sum, item) => sum + item.price * item.quantity,
        0
    );

    const value = {
        items: state.items,
        totalItems,
        totalPrice,
        addItem,
        removeItem,
        updateQuantity,
        clearCart,
    };

    return (
        <CartContext.Provider value={value}>
            {children}
        </CartContext.Provider>
    );
}

// --- 使用 Cart Context 的组件 ---
function ProductCard({ product }) {
    const { addItem } = useCart();

    return (
        <div className="product-card">
            <h3>{product.name}</h3>
            <p>${product.price}</p>
            <button onClick={() => addItem(product)}>
                加入购物车
            </button>
        </div>
    );
}

function ShoppingCart() {
    const { items, totalItems, totalPrice, updateQuantity, removeItem, clearCart } = useCart();

    if (items.length === 0) {
        return <p>购物车是空的</p>;
    }

    return (
        <div className="cart">
            <h2>购物车 ({totalItems})</h2>

            {items.map(item => (
                <div key={item.id} className="cart-item">
                    <span>{item.name}</span>
                    <span>${item.price}</span>
                    <button onClick={() => updateQuantity(item.id, item.quantity - 1)}>-</button>
                    <span>{item.quantity}</span>
                    <button onClick={() => updateQuantity(item.id, item.quantity + 1)}>+</button>
                    <button onClick={() => removeItem(item.id)}>删除</button>
                </div>
            ))}

            <div className="cart-total">
                <strong>总计: ${totalPrice.toFixed(2)}</strong>
            </div>

            <button onClick={clearCart}>清空购物车</button>
        </div>
    );
}


// ============================================================
//                    5. Context 性能优化
// ============================================================

/**
 * 【Context 性能问题】
 *
 * 当 Context value 变化时，所有使用该 Context 的组件都会重新渲染。
 *
 * 【优化策略】
 * 1. 拆分 Context：将频繁变化和不常变化的数据分开
 * 2. 使用 memo：防止不必要的子组件渲染
 * 3. 分离 state 和 dispatch
 */

// --- 拆分 Context ---
// 状态 Context（可能频繁变化）
const CountStateContext = createContext(null);
// 操作 Context（稳定的函数引用）
const CountDispatchContext = createContext(null);

function CountProvider({ children }) {
    const [count, setCount] = useState(0);

    // 使用 useCallback 保持函数引用稳定
    const increment = React.useCallback(() => setCount(c => c + 1), []);
    const decrement = React.useCallback(() => setCount(c => c - 1), []);

    return (
        <CountStateContext.Provider value={count}>
            <CountDispatchContext.Provider value={{ increment, decrement }}>
                {children}
            </CountDispatchContext.Provider>
        </CountStateContext.Provider>
    );
}

// 只使用 count 的组件 - count 变化时重新渲染
function CountDisplay() {
    const count = useContext(CountStateContext);
    console.log('CountDisplay 渲染');
    return <p>Count: {count}</p>;
}

// 只使用 dispatch 的组件 - 不会因为 count 变化而重新渲染
function CountButtons() {
    const { increment, decrement } = useContext(CountDispatchContext);
    console.log('CountButtons 渲染');

    return (
        <div>
            <button onClick={decrement}>-</button>
            <button onClick={increment}>+</button>
        </div>
    );
}


// ============================================================
//                    6. Context 最佳实践
// ============================================================

/**
 * 【最佳实践总结】
 *
 * 1. 何时使用 Context
 *    - 全局数据：主题、用户、语言
 *    - 跨多层组件共享的数据
 *    - 避免 props drilling
 *
 * 2. 何时不使用 Context
 *    - 只传递一两层的数据
 *    - 频繁变化的数据（考虑性能）
 *    - 可以用 props 解决的场景
 *
 * 3. 结构建议
 *    - 每个 Context 一个文件
 *    - 提供自定义 Hook
 *    - 添加错误检查
 */

// --- 推荐的 Context 文件结构 ---
/*
// contexts/ThemeContext.js

import { createContext, useContext, useState } from 'react';

// 1. 创建 Context
const ThemeContext = createContext(null);

// 2. 自定义 Hook（包含错误检查）
export function useTheme() {
    const context = useContext(ThemeContext);
    if (context === null) {
        throw new Error('useTheme must be used within ThemeProvider');
    }
    return context;
}

// 3. Provider 组件
export function ThemeProvider({ children }) {
    const [theme, setTheme] = useState('light');

    const toggleTheme = () => {
        setTheme(t => t === 'light' ? 'dark' : 'light');
    };

    return (
        <ThemeContext.Provider value={{ theme, toggleTheme }}>
            {children}
        </ThemeContext.Provider>
    );
}

// 4. 可选：导出 Context（用于特殊情况）
export { ThemeContext };
*/


// ============================================================
//                    导出
// ============================================================

export {
    // Theme Context
    ThemeContext,
    ThemeProvider,
    ThemedButton,
    ThemedText,

    // Auth Context
    AuthContext,
    AuthProvider,
    useAuth,
    LoginForm,
    UserProfile,

    // Language Context
    LanguageContext,
    LanguageProvider,
    useLanguage,

    // Combined Providers
    AppProviders,
    Header,

    // Cart Context (with useReducer)
    CartContext,
    CartProvider,
    useCart,
    ProductCard,
    ShoppingCart,

    // Optimized Context
    CountProvider,
    CountDisplay,
    CountButtons,
};

export default function ContextTutorial() {
    return (
        <AppProviders>
            <div className="tutorial">
                <h1>React Context 教程</h1>
                <Header />
                <ThemedButton />
                <ThemedText />
            </div>
        </AppProviders>
    );
}
```
