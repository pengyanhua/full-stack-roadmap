# useEffect.jsx

::: info 文件信息
- 📄 原文件：`02_useEffect.jsx`
- 🔤 语言：jsx
:::

React useEffect Hook
useEffect 用于处理副作用（Side Effects），如数据获取、订阅、DOM 操作等。
它是函数组件中处理生命周期逻辑的主要方式。

## 完整代码

```jsx
/**
 * ============================================================
 *                    React useEffect Hook
 * ============================================================
 * useEffect 用于处理副作用（Side Effects），如数据获取、订阅、DOM 操作等。
 * 它是函数组件中处理生命周期逻辑的主要方式。
 * ============================================================
 */

import React, { useState, useEffect } from 'react';

// ============================================================
//                    1. useEffect 基础
// ============================================================

/**
 * 【什么是副作用】
 *
 * 副作用是指组件渲染之外的操作：
 * - 数据获取（API 请求）
 * - 订阅（WebSocket、事件监听）
 * - 手动 DOM 操作
 * - 定时器
 * - 日志记录
 *
 * 【useEffect 语法】
 * useEffect(effectFunction, dependencyArray)
 *
 * - effectFunction: 副作用函数，在组件渲染后执行
 * - dependencyArray: 依赖数组，控制副作用何时重新执行
 */

// --- 基本用法 ---
function BasicEffect() {
    const [count, setCount] = useState(0);

    // 每次渲染后都执行
    useEffect(() => {
        console.log('组件渲染了，count =', count);
        // 更新文档标题
        document.title = `点击了 ${count} 次`;
    });

    return (
        <div>
            <p>点击次数: {count}</p>
            <button onClick={() => setCount(count + 1)}>点击</button>
        </div>
    );
}


// ============================================================
//                    2. 依赖数组
// ============================================================

/**
 * 【依赖数组的作用】
 *
 * 依赖数组决定 useEffect 何时重新执行：
 *
 * 1. 不传依赖数组：每次渲染后都执行
 *    useEffect(() => { ... })
 *
 * 2. 空数组 []：只在挂载时执行一次
 *    useEffect(() => { ... }, [])
 *
 * 3. 有依赖 [a, b]：依赖变化时执行
 *    useEffect(() => { ... }, [a, b])
 */

// --- 不传依赖数组：每次渲染都执行 ---
function EveryRender() {
    const [count, setCount] = useState(0);
    const [text, setText] = useState('');

    // 任何状态变化导致的重新渲染都会执行
    useEffect(() => {
        console.log('EveryRender: 组件渲染了');
        console.log('count:', count, 'text:', text);
    });

    return (
        <div>
            <p>Count: {count}</p>
            <button onClick={() => setCount(c => c + 1)}>增加</button>
            <input
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="输入文字"
            />
        </div>
    );
}

// --- 空依赖数组：只在挂载时执行一次 ---
function MountOnly() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);

    // 只在组件挂载时执行一次
    // 适合：初始数据获取、一次性设置
    useEffect(() => {
        console.log('MountOnly: 组件挂载');

        // 模拟 API 请求
        const fetchData = async () => {
            // 模拟延迟
            await new Promise(resolve => setTimeout(resolve, 1000));
            setData({ message: '数据加载成功!' });
            setLoading(false);
        };

        fetchData();
    }, []);  // 空数组 = 只执行一次

    if (loading) return <p>加载中...</p>;
    return <p>{data?.message}</p>;
}

// --- 有依赖数组：依赖变化时执行 ---
function WithDependencies() {
    const [userId, setUserId] = useState(1);
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(false);

    // 当 userId 变化时重新获取用户数据
    useEffect(() => {
        console.log('WithDependencies: userId 变化，重新获取数据');

        const fetchUser = async () => {
            setLoading(true);
            // 模拟 API 请求
            await new Promise(resolve => setTimeout(resolve, 500));
            setUser({
                id: userId,
                name: `用户 ${userId}`,
                email: `user${userId}@example.com`,
            });
            setLoading(false);
        };

        fetchUser();
    }, [userId]);  // 依赖 userId

    return (
        <div>
            <div>
                <button onClick={() => setUserId(1)}>用户 1</button>
                <button onClick={() => setUserId(2)}>用户 2</button>
                <button onClick={() => setUserId(3)}>用户 3</button>
            </div>
            {loading ? (
                <p>加载中...</p>
            ) : (
                user && (
                    <div>
                        <p>ID: {user.id}</p>
                        <p>姓名: {user.name}</p>
                        <p>邮箱: {user.email}</p>
                    </div>
                )
            )}
        </div>
    );
}


// ============================================================
//                    3. 清理函数
// ============================================================

/**
 * 【清理函数】
 *
 * useEffect 可以返回一个清理函数：
 * - 在组件卸载时执行
 * - 在下一次 effect 执行前执行
 *
 * 【使用场景】
 * - 取消订阅
 * - 清除定时器
 * - 取消网络请求
 * - 移除事件监听
 */

// --- 定时器清理 ---
function TimerExample() {
    const [seconds, setSeconds] = useState(0);
    const [isRunning, setIsRunning] = useState(false);

    useEffect(() => {
        // 如果没有运行，不设置定时器
        if (!isRunning) return;

        console.log('设置定时器');

        // 设置定时器
        const intervalId = setInterval(() => {
            setSeconds(s => s + 1);
        }, 1000);

        // 清理函数：清除定时器
        return () => {
            console.log('清除定时器');
            clearInterval(intervalId);
        };
    }, [isRunning]);  // 依赖 isRunning

    return (
        <div>
            <p>秒数: {seconds}</p>
            <button onClick={() => setIsRunning(!isRunning)}>
                {isRunning ? '暂停' : '开始'}
            </button>
            <button onClick={() => setSeconds(0)}>重置</button>
        </div>
    );
}

// --- 事件监听清理 ---
function WindowSizeTracker() {
    const [windowSize, setWindowSize] = useState({
        width: window.innerWidth,
        height: window.innerHeight,
    });

    useEffect(() => {
        // 事件处理函数
        const handleResize = () => {
            setWindowSize({
                width: window.innerWidth,
                height: window.innerHeight,
            });
        };

        console.log('添加 resize 监听器');

        // 添加事件监听
        window.addEventListener('resize', handleResize);

        // 清理函数：移除事件监听
        return () => {
            console.log('移除 resize 监听器');
            window.removeEventListener('resize', handleResize);
        };
    }, []);  // 空数组，只在挂载/卸载时执行

    return (
        <div>
            <p>窗口宽度: {windowSize.width}px</p>
            <p>窗口高度: {windowSize.height}px</p>
        </div>
    );
}

// --- 订阅清理 ---
function ChatRoom({ roomId }) {
    const [messages, setMessages] = useState([]);

    useEffect(() => {
        console.log(`连接到聊天室: ${roomId}`);

        // 模拟 WebSocket 连接
        const connection = createConnection(roomId);

        // 订阅消息
        connection.on('message', (message) => {
            setMessages(prev => [...prev, message]);
        });

        // 连接
        connection.connect();

        // 清理函数：断开连接
        return () => {
            console.log(`断开聊天室: ${roomId}`);
            connection.disconnect();
        };
    }, [roomId]);  // roomId 变化时重新连接

    return (
        <div>
            <h3>聊天室: {roomId}</h3>
            <ul>
                {messages.map((msg, i) => (
                    <li key={i}>{msg}</li>
                ))}
            </ul>
        </div>
    );
}

// 模拟 WebSocket 连接
function createConnection(roomId) {
    return {
        on: (event, callback) => {
            console.log(`监听事件: ${event}`);
        },
        connect: () => {
            console.log(`已连接到 ${roomId}`);
        },
        disconnect: () => {
            console.log(`已断开 ${roomId}`);
        },
    };
}


// ============================================================
//                    4. 数据获取模式
// ============================================================

/**
 * 【数据获取最佳实践】
 *
 * 1. 处理加载状态
 * 2. 处理错误状态
 * 3. 处理竞态条件（Race Condition）
 * 4. 支持取消请求
 */

// --- 完整的数据获取示例 ---
function DataFetchingExample() {
    const [query, setQuery] = useState('react');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        // 标记是否已取消（用于处理竞态条件）
        let cancelled = false;

        const fetchData = async () => {
            // 空查询不请求
            if (!query.trim()) {
                setResults([]);
                return;
            }

            setLoading(true);
            setError(null);

            try {
                // 模拟 API 请求
                await new Promise(resolve => setTimeout(resolve, 500));

                // 模拟返回数据
                const data = [
                    { id: 1, title: `${query} 结果 1` },
                    { id: 2, title: `${query} 结果 2` },
                    { id: 3, title: `${query} 结果 3` },
                ];

                // 检查是否已取消（防止竞态条件）
                if (!cancelled) {
                    setResults(data);
                }
            } catch (err) {
                if (!cancelled) {
                    setError('获取数据失败');
                }
            } finally {
                if (!cancelled) {
                    setLoading(false);
                }
            }
        };

        fetchData();

        // 清理函数：标记为已取消
        return () => {
            cancelled = true;
        };
    }, [query]);

    return (
        <div>
            <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="搜索..."
            />

            {loading && <p>搜索中...</p>}
            {error && <p className="error">{error}</p>}

            <ul>
                {results.map(item => (
                    <li key={item.id}>{item.title}</li>
                ))}
            </ul>
        </div>
    );
}

// --- 使用 AbortController 取消请求 ---
function FetchWithAbort() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        // 创建 AbortController
        const controller = new AbortController();
        const signal = controller.signal;

        const fetchData = async () => {
            setLoading(true);

            try {
                const response = await fetch('/api/data', { signal });
                const json = await response.json();
                setData(json);
            } catch (err) {
                // 忽略取消错误
                if (err.name !== 'AbortError') {
                    console.error('Fetch error:', err);
                }
            } finally {
                setLoading(false);
            }
        };

        fetchData();

        // 清理：取消请求
        return () => {
            controller.abort();
        };
    }, []);

    return (
        <div>
            {loading ? <p>加载中...</p> : <pre>{JSON.stringify(data)}</pre>}
        </div>
    );
}


// ============================================================
//                    5. 多个 useEffect
// ============================================================

/**
 * 【多个 useEffect】
 *
 * 可以使用多个 useEffect 将不相关的逻辑分开：
 * - 更好的代码组织
 * - 更清晰的依赖关系
 * - 更容易理解和维护
 */

function MultipleEffects({ userId }) {
    const [user, setUser] = useState(null);
    const [posts, setPosts] = useState([]);

    // Effect 1: 获取用户信息
    useEffect(() => {
        console.log('获取用户信息');

        const fetchUser = async () => {
            // 模拟请求
            await new Promise(r => setTimeout(r, 300));
            setUser({ id: userId, name: `用户 ${userId}` });
        };

        fetchUser();
    }, [userId]);

    // Effect 2: 获取用户文章（独立的逻辑）
    useEffect(() => {
        console.log('获取用户文章');

        const fetchPosts = async () => {
            await new Promise(r => setTimeout(r, 500));
            setPosts([
                { id: 1, title: `文章 1 by ${userId}` },
                { id: 2, title: `文章 2 by ${userId}` },
            ]);
        };

        fetchPosts();
    }, [userId]);

    // Effect 3: 更新文档标题
    useEffect(() => {
        if (user) {
            document.title = `${user.name} 的主页`;
        }

        return () => {
            document.title = 'React App';
        };
    }, [user]);

    return (
        <div>
            {user && <h2>{user.name}</h2>}
            <ul>
                {posts.map(post => (
                    <li key={post.id}>{post.title}</li>
                ))}
            </ul>
        </div>
    );
}


// ============================================================
//                    6. 常见陷阱和解决方案
// ============================================================

/**
 * 【常见陷阱】
 *
 * 1. 无限循环
 * 2. 过时的闭包
 * 3. 缺少依赖
 * 4. 对象/数组依赖
 */

// --- 陷阱1: 无限循环 ---
function InfiniteLoopProblem() {
    const [count, setCount] = useState(0);

    // ❌ 错误：每次渲染都会触发，导致无限循环
    // useEffect(() => {
    //     setCount(count + 1);  // 更新状态 -> 触发渲染 -> 再次执行 effect
    // });

    // ✅ 正确：添加条件或正确的依赖
    useEffect(() => {
        if (count < 10) {
            setCount(count + 1);
        }
    }, [count]);

    return <p>Count: {count}</p>;
}

// --- 陷阱2: 过时的闭包 ---
function StaleClosureProblem() {
    const [count, setCount] = useState(0);

    useEffect(() => {
        const timer = setInterval(() => {
            // ❌ 问题：count 总是初始值 0（闭包陷阱）
            // console.log('count:', count);

            // ✅ 解决方案1：使用函数式更新
            setCount(c => c + 1);
        }, 1000);

        return () => clearInterval(timer);
    }, []);  // 空依赖，effect 只执行一次

    // ✅ 解决方案2：将 count 加入依赖（但会重新创建定时器）
    // useEffect(() => {
    //     const timer = setInterval(() => {
    //         console.log('count:', count);
    //         setCount(count + 1);
    //     }, 1000);
    //     return () => clearInterval(timer);
    // }, [count]);

    return <p>Count: {count}</p>;
}

// --- 陷阱3: 对象/数组依赖 ---
function ObjectDependencyProblem() {
    const [user, setUser] = useState({ name: 'Alice' });

    // ❌ 问题：每次渲染都创建新对象，导致 effect 每次都执行
    // const options = { userId: user.name };
    // useEffect(() => {
    //     console.log('options changed');
    // }, [options]);

    // ✅ 解决方案1：直接依赖原始值
    useEffect(() => {
        console.log('user.name changed');
    }, [user.name]);

    // ✅ 解决方案2：使用 useMemo 缓存对象（见后续章节）

    return (
        <div>
            <p>User: {user.name}</p>
            <button onClick={() => setUser({ name: 'Bob' })}>
                改名
            </button>
        </div>
    );
}


// ============================================================
//                    7. useEffect vs 事件处理
// ============================================================

/**
 * 【何时用 useEffect，何时用事件处理】
 *
 * useEffect 用于：
 * - 同步外部系统（数据获取、订阅）
 * - 响应状态/props 变化
 * - 组件生命周期相关操作
 *
 * 事件处理用于：
 * - 响应用户交互
 * - 不需要在渲染时执行的操作
 */

// --- 正确使用事件处理 ---
function EventVsEffect() {
    const [items, setItems] = useState([]);

    // ❌ 不要用 effect 来响应事件
    // const [shouldAdd, setShouldAdd] = useState(false);
    // useEffect(() => {
    //     if (shouldAdd) {
    //         setItems([...items, 'new item']);
    //         setShouldAdd(false);
    //     }
    // }, [shouldAdd]);

    // ✅ 直接在事件处理中更新状态
    const handleAdd = () => {
        setItems([...items, `Item ${items.length + 1}`]);
    };

    // ✅ useEffect 用于同步外部系统
    useEffect(() => {
        // 将 items 同步到服务器
        console.log('同步 items 到服务器:', items);
    }, [items]);

    return (
        <div>
            <button onClick={handleAdd}>添加</button>
            <ul>
                {items.map((item, i) => (
                    <li key={i}>{item}</li>
                ))}
            </ul>
        </div>
    );
}


// ============================================================
//                    8. 自定义 Hook 抽取逻辑
// ============================================================

/**
 * 【自定义 Hook】
 *
 * 当 useEffect 逻辑可复用时，可以抽取为自定义 Hook
 */

// --- 自定义 Hook: useDocumentTitle ---
function useDocumentTitle(title) {
    useEffect(() => {
        document.title = title;
        return () => {
            document.title = 'React App';
        };
    }, [title]);
}

// --- 自定义 Hook: useWindowSize ---
function useWindowSize() {
    const [size, setSize] = useState({
        width: window.innerWidth,
        height: window.innerHeight,
    });

    useEffect(() => {
        const handleResize = () => {
            setSize({
                width: window.innerWidth,
                height: window.innerHeight,
            });
        };

        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    return size;
}

// --- 自定义 Hook: useFetch ---
function useFetch(url) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        let cancelled = false;

        const fetchData = async () => {
            setLoading(true);
            setError(null);

            try {
                const response = await fetch(url);
                const json = await response.json();

                if (!cancelled) {
                    setData(json);
                }
            } catch (err) {
                if (!cancelled) {
                    setError(err.message);
                }
            } finally {
                if (!cancelled) {
                    setLoading(false);
                }
            }
        };

        fetchData();

        return () => {
            cancelled = true;
        };
    }, [url]);

    return { data, loading, error };
}

// --- 使用自定义 Hook ---
function CustomHookDemo() {
    // 使用自定义 Hook
    useDocumentTitle('自定义 Hook 示例');
    const { width, height } = useWindowSize();
    // const { data, loading, error } = useFetch('/api/users');

    return (
        <div>
            <p>窗口尺寸: {width} x {height}</p>
        </div>
    );
}


// ============================================================
//                    导出
// ============================================================

export {
    BasicEffect,
    EveryRender,
    MountOnly,
    WithDependencies,
    TimerExample,
    WindowSizeTracker,
    DataFetchingExample,
    MultipleEffects,
    CustomHookDemo,
    useDocumentTitle,
    useWindowSize,
    useFetch,
};

export default function UseEffectTutorial() {
    return (
        <div className="tutorial">
            <h1>useEffect Hook 教程</h1>
            <BasicEffect />
            <TimerExample />
            <WindowSizeTracker />
            <DataFetchingExample />
        </div>
    );
}
```
