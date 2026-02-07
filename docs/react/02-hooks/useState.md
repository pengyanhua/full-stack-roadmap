# useState.jsx

::: info 文件信息
- 📄 原文件：`01_useState.jsx`
- 🔤 语言：jsx
:::

React useState Hook
useState 是 React 最基础的 Hook，用于在函数组件中添加状态。
状态变化会触发组件重新渲染。

## 完整代码

```jsx
/**
 * ============================================================
 *                    React useState Hook
 * ============================================================
 * useState 是 React 最基础的 Hook，用于在函数组件中添加状态。
 * 状态变化会触发组件重新渲染。
 * ============================================================
 */

import React, { useState } from 'react';

// ============================================================
//                    1. useState 基础
// ============================================================

/**
 * 【useState 是什么】
 *
 * useState 让函数组件拥有状态管理能力：
 * - 返回一个数组：[当前状态值, 更新状态的函数]
 * - 状态更新会触发组件重新渲染
 * - 每次渲染都有自己的状态快照
 *
 * 【语法】
 * const [state, setState] = useState(initialValue);
 *
 * - state: 当前状态值
 * - setState: 更新状态的函数
 * - initialValue: 初始值（只在首次渲染时使用）
 */

// --- 基本计数器 ---
function Counter() {
    // 声明状态变量 count，初始值为 0
    // setCount 是更新 count 的函数
    const [count, setCount] = useState(0);

    // 点击处理函数
    const handleIncrement = () => {
        setCount(count + 1);  // 更新状态
    };

    const handleDecrement = () => {
        setCount(count - 1);
    };

    const handleReset = () => {
        setCount(0);
    };

    return (
        <div className="counter">
            <h2>计数器: {count}</h2>
            <button onClick={handleDecrement}>-1</button>
            <button onClick={handleReset}>重置</button>
            <button onClick={handleIncrement}>+1</button>
        </div>
    );
}


// ============================================================
//                    2. 不同类型的状态
// ============================================================

/**
 * 【状态可以是任意类型】
 *
 * - 基本类型：number, string, boolean
 * - 引用类型：object, array
 *
 * 【注意】
 * 对于对象和数组，需要创建新的引用才能触发更新
 * 不能直接修改原对象/数组
 */

// --- 字符串状态 ---
function TextInput() {
    const [text, setText] = useState('');

    const handleChange = (event) => {
        setText(event.target.value);
    };

    return (
        <div>
            <input
                type="text"
                value={text}
                onChange={handleChange}
                placeholder="输入文字..."
            />
            <p>你输入了: {text}</p>
            <p>字符数: {text.length}</p>
        </div>
    );
}

// --- 布尔状态 ---
function Toggle() {
    const [isOn, setIsOn] = useState(false);

    // 切换状态
    const toggle = () => {
        setIsOn(!isOn);
    };

    return (
        <div>
            <button onClick={toggle}>
                {isOn ? '关闭' : '开启'}
            </button>
            <p>状态: {isOn ? 'ON' : 'OFF'}</p>
        </div>
    );
}

// --- 对象状态 ---
function UserForm() {
    // 对象作为状态
    const [user, setUser] = useState({
        name: '',
        email: '',
        age: 0,
    });

    // 更新对象的某个属性
    // 必须创建新对象，不能直接修改
    const handleNameChange = (event) => {
        setUser({
            ...user,           // 展开旧对象
            name: event.target.value,  // 覆盖 name
        });
    };

    const handleEmailChange = (event) => {
        setUser({
            ...user,
            email: event.target.value,
        });
    };

    // 通用的字段更新函数
    const handleFieldChange = (field) => (event) => {
        setUser({
            ...user,
            [field]: event.target.value,
        });
    };

    return (
        <div>
            <h3>用户表单</h3>
            <input
                type="text"
                value={user.name}
                onChange={handleNameChange}
                placeholder="姓名"
            />
            <input
                type="email"
                value={user.email}
                onChange={handleEmailChange}
                placeholder="邮箱"
            />
            <input
                type="number"
                value={user.age}
                onChange={handleFieldChange('age')}
                placeholder="年龄"
            />
            <pre>{JSON.stringify(user, null, 2)}</pre>
        </div>
    );
}

// --- 数组状态 ---
function TodoList() {
    const [todos, setTodos] = useState([
        { id: 1, text: '学习 React', completed: false },
        { id: 2, text: '写代码', completed: true },
    ]);
    const [inputText, setInputText] = useState('');

    // 添加项目（创建新数组）
    const addTodo = () => {
        if (!inputText.trim()) return;

        const newTodo = {
            id: Date.now(),  // 简单的唯一 ID
            text: inputText,
            completed: false,
        };

        // 方式1: 展开运算符
        setTodos([...todos, newTodo]);

        // 方式2: concat
        // setTodos(todos.concat(newTodo));

        setInputText('');  // 清空输入
    };

    // 删除项目（过滤创建新数组）
    const deleteTodo = (id) => {
        setTodos(todos.filter(todo => todo.id !== id));
    };

    // 切换完成状态（map 创建新数组）
    const toggleTodo = (id) => {
        setTodos(todos.map(todo =>
            todo.id === id
                ? { ...todo, completed: !todo.completed }
                : todo
        ));
    };

    return (
        <div>
            <h3>待办列表</h3>
            <div>
                <input
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="添加待办..."
                    onKeyDown={(e) => e.key === 'Enter' && addTodo()}
                />
                <button onClick={addTodo}>添加</button>
            </div>
            <ul>
                {todos.map(todo => (
                    <li key={todo.id}>
                        <input
                            type="checkbox"
                            checked={todo.completed}
                            onChange={() => toggleTodo(todo.id)}
                        />
                        <span style={{
                            textDecoration: todo.completed ? 'line-through' : 'none'
                        }}>
                            {todo.text}
                        </span>
                        <button onClick={() => deleteTodo(todo.id)}>删除</button>
                    </li>
                ))}
            </ul>
            <p>总数: {todos.length}, 已完成: {todos.filter(t => t.completed).length}</p>
        </div>
    );
}


// ============================================================
//                    3. 函数式更新
// ============================================================

/**
 * 【函数式更新】
 *
 * 当新状态依赖于旧状态时，应该使用函数式更新：
 * setState(prevState => newState)
 *
 * 【为什么需要函数式更新】
 * - 状态更新是异步的
 * - 多次调用 setState 可能会被合并
 * - 函数式更新确保获取最新的状态值
 */

// --- 问题示例 ---
function ProblemCounter() {
    const [count, setCount] = useState(0);

    // ❌ 错误：多次调用只会增加 1
    const incrementThreeTimes = () => {
        // 这三次调用都基于同一个 count 快照
        setCount(count + 1);  // count = 0, 设置为 1
        setCount(count + 1);  // count = 0, 设置为 1
        setCount(count + 1);  // count = 0, 设置为 1
        // 结果：count = 1（不是 3）
    };

    return (
        <div>
            <p>Count: {count}</p>
            <button onClick={incrementThreeTimes}>
                增加三次（错误方式）
            </button>
        </div>
    );
}

// --- 正确示例 ---
function CorrectCounter() {
    const [count, setCount] = useState(0);

    // ✅ 正确：使用函数式更新
    const incrementThreeTimes = () => {
        // 每次都基于最新的 prevCount
        setCount(prevCount => prevCount + 1);  // 0 -> 1
        setCount(prevCount => prevCount + 1);  // 1 -> 2
        setCount(prevCount => prevCount + 1);  // 2 -> 3
        // 结果：count = 3
    };

    return (
        <div>
            <p>Count: {count}</p>
            <button onClick={incrementThreeTimes}>
                增加三次（正确方式）
            </button>
        </div>
    );
}

// --- 函数式更新用于对象和数组 ---
function ShoppingCart() {
    const [cart, setCart] = useState([]);

    const addItem = (item) => {
        // 函数式更新确保获取最新的 cart
        setCart(prevCart => {
            // 检查是否已存在
            const existingItem = prevCart.find(i => i.id === item.id);

            if (existingItem) {
                // 已存在，增加数量
                return prevCart.map(i =>
                    i.id === item.id
                        ? { ...i, quantity: i.quantity + 1 }
                        : i
                );
            }

            // 不存在，添加新项
            return [...prevCart, { ...item, quantity: 1 }];
        });
    };

    const removeItem = (id) => {
        setCart(prevCart => prevCart.filter(item => item.id !== id));
    };

    const updateQuantity = (id, quantity) => {
        setCart(prevCart =>
            prevCart.map(item =>
                item.id === id
                    ? { ...item, quantity: Math.max(0, quantity) }
                    : item
            ).filter(item => item.quantity > 0)  // 移除数量为 0 的
        );
    };

    return (
        <div>
            <h3>购物车</h3>
            {cart.map(item => (
                <div key={item.id}>
                    <span>{item.name} x {item.quantity}</span>
                    <button onClick={() => updateQuantity(item.id, item.quantity - 1)}>-</button>
                    <button onClick={() => updateQuantity(item.id, item.quantity + 1)}>+</button>
                    <button onClick={() => removeItem(item.id)}>删除</button>
                </div>
            ))}
            <button onClick={() => addItem({ id: 1, name: '商品A', price: 10 })}>
                添加商品A
            </button>
            <button onClick={() => addItem({ id: 2, name: '商品B', price: 20 })}>
                添加商品B
            </button>
        </div>
    );
}


// ============================================================
//                    4. 惰性初始化
// ============================================================

/**
 * 【惰性初始化】
 *
 * 如果初始状态需要复杂计算，可以传递函数：
 * useState(() => computeExpensiveValue())
 *
 * 【好处】
 * - 函数只在首次渲染时执行
 * - 避免每次渲染都进行不必要的计算
 */

// --- 惰性初始化示例 ---
function LazyInitExample() {
    // ❌ 不好：每次渲染都会计算
    // const [value, setValue] = useState(expensiveComputation());

    // ✅ 好：只在首次渲染时计算
    const [value, setValue] = useState(() => {
        console.log('惰性初始化执行');
        // 模拟复杂计算
        return heavyComputation();
    });

    return (
        <div>
            <p>值: {value}</p>
            <button onClick={() => setValue(v => v + 1)}>增加</button>
        </div>
    );
}

function heavyComputation() {
    // 模拟耗时计算
    let result = 0;
    for (let i = 0; i < 1000; i++) {
        result += i;
    }
    return result;
}

// --- 从 localStorage 读取初始值 ---
function PersistentCounter() {
    // 惰性初始化：从 localStorage 读取
    const [count, setCount] = useState(() => {
        const saved = localStorage.getItem('counter');
        return saved ? parseInt(saved, 10) : 0;
    });

    // 状态变化时保存到 localStorage
    const updateCount = (newCount) => {
        setCount(newCount);
        localStorage.setItem('counter', newCount.toString());
    };

    return (
        <div>
            <p>持久化计数: {count}</p>
            <button onClick={() => updateCount(count + 1)}>增加</button>
            <button onClick={() => updateCount(0)}>重置</button>
        </div>
    );
}


// ============================================================
//                    5. 多个状态
// ============================================================

/**
 * 【多个状态】
 *
 * 可以在一个组件中使用多个 useState：
 * - 每个状态独立管理
 * - 按逻辑分组状态
 *
 * 【何时使用多个 useState vs 一个对象】
 * - 相关的数据放在一起（用对象）
 * - 不相关的数据分开（用多个 useState）
 */

// --- 多个独立状态 ---
function MultipleStates() {
    // 不相关的状态分开管理
    const [name, setName] = useState('');
    const [age, setAge] = useState(0);
    const [isSubscribed, setIsSubscribed] = useState(false);

    return (
        <div>
            <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="姓名"
            />
            <input
                type="number"
                value={age}
                onChange={(e) => setAge(parseInt(e.target.value) || 0)}
                placeholder="年龄"
            />
            <label>
                <input
                    type="checkbox"
                    checked={isSubscribed}
                    onChange={(e) => setIsSubscribed(e.target.checked)}
                />
                订阅通知
            </label>
            <p>
                {name}, {age}岁, {isSubscribed ? '已订阅' : '未订阅'}
            </p>
        </div>
    );
}

// --- 相关状态用对象 ---
function FormWithObject() {
    // 相关的表单数据用一个对象
    const [formData, setFormData] = useState({
        username: '',
        password: '',
        confirmPassword: '',
    });

    // UI 状态单独管理
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [errors, setErrors] = useState({});

    const handleChange = (field) => (event) => {
        setFormData(prev => ({
            ...prev,
            [field]: event.target.value,
        }));
    };

    const handleSubmit = async (event) => {
        event.preventDefault();

        // 验证
        const newErrors = {};
        if (!formData.username) {
            newErrors.username = '用户名不能为空';
        }
        if (formData.password !== formData.confirmPassword) {
            newErrors.confirmPassword = '密码不匹配';
        }

        if (Object.keys(newErrors).length > 0) {
            setErrors(newErrors);
            return;
        }

        setIsSubmitting(true);
        // 模拟提交
        await new Promise(resolve => setTimeout(resolve, 1000));
        setIsSubmitting(false);

        console.log('提交成功', formData);
    };

    return (
        <form onSubmit={handleSubmit}>
            <div>
                <input
                    value={formData.username}
                    onChange={handleChange('username')}
                    placeholder="用户名"
                />
                {errors.username && <span className="error">{errors.username}</span>}
            </div>
            <div>
                <input
                    type="password"
                    value={formData.password}
                    onChange={handleChange('password')}
                    placeholder="密码"
                />
            </div>
            <div>
                <input
                    type="password"
                    value={formData.confirmPassword}
                    onChange={handleChange('confirmPassword')}
                    placeholder="确认密码"
                />
                {errors.confirmPassword && (
                    <span className="error">{errors.confirmPassword}</span>
                )}
            </div>
            <button type="submit" disabled={isSubmitting}>
                {isSubmitting ? '提交中...' : '提交'}
            </button>
        </form>
    );
}


// ============================================================
//                    6. 状态提升
// ============================================================

/**
 * 【状态提升】
 *
 * 当多个组件需要共享状态时：
 * 1. 将状态提升到最近的共同父组件
 * 2. 通过 props 传递状态和更新函数
 *
 * 这是 React 数据流的核心模式
 */

// 子组件：温度输入
function TemperatureInput({ scale, temperature, onTemperatureChange }) {
    const scaleNames = {
        c: '摄氏度',
        f: '华氏度',
    };

    return (
        <div>
            <label>输入{scaleNames[scale]}:</label>
            <input
                type="number"
                value={temperature}
                onChange={(e) => onTemperatureChange(e.target.value)}
            />
        </div>
    );
}

// 父组件：管理共享状态
function TemperatureCalculator() {
    // 状态提升到父组件
    const [temperature, setTemperature] = useState('');
    const [scale, setScale] = useState('c');

    // 转换函数
    const toCelsius = (fahrenheit) => {
        return ((fahrenheit - 32) * 5 / 9).toFixed(2);
    };

    const toFahrenheit = (celsius) => {
        return ((celsius * 9 / 5) + 32).toFixed(2);
    };

    // 处理摄氏度变化
    const handleCelsiusChange = (temp) => {
        setScale('c');
        setTemperature(temp);
    };

    // 处理华氏度变化
    const handleFahrenheitChange = (temp) => {
        setScale('f');
        setTemperature(temp);
    };

    // 计算显示值
    const celsius = scale === 'f' ? toCelsius(parseFloat(temperature)) : temperature;
    const fahrenheit = scale === 'c' ? toFahrenheit(parseFloat(temperature)) : temperature;

    return (
        <div>
            <h3>温度转换器</h3>
            <TemperatureInput
                scale="c"
                temperature={celsius}
                onTemperatureChange={handleCelsiusChange}
            />
            <TemperatureInput
                scale="f"
                temperature={fahrenheit}
                onTemperatureChange={handleFahrenheitChange}
            />
            {temperature && (
                <p>
                    {celsius}°C = {fahrenheit}°F
                </p>
            )}
        </div>
    );
}


// ============================================================
//                    7. useState 最佳实践
// ============================================================

/**
 * 【最佳实践总结】
 *
 * 1. 最小化状态
 *    - 只存储必要的数据
 *    - 派生数据用计算代替存储
 *
 * 2. 状态结构
 *    - 相关数据放在一起
 *    - 避免深层嵌套
 *
 * 3. 更新方式
 *    - 依赖旧值时用函数式更新
 *    - 对象/数组必须创建新引用
 *
 * 4. 性能考虑
 *    - 复杂初始值用惰性初始化
 *    - 频繁更新考虑使用 useReducer
 */

// --- 派生状态示例 ---
function ProductFilter() {
    const [products] = useState([
        { id: 1, name: 'iPhone', category: '电子', price: 999 },
        { id: 2, name: 'MacBook', category: '电子', price: 1999 },
        { id: 3, name: 'T恤', category: '服装', price: 29 },
        { id: 4, name: '牛仔裤', category: '服装', price: 59 },
    ]);

    const [searchTerm, setSearchTerm] = useState('');
    const [selectedCategory, setSelectedCategory] = useState('全部');

    // ✅ 派生数据：通过计算得到，不需要额外的状态
    const filteredProducts = products.filter(product => {
        const matchesSearch = product.name
            .toLowerCase()
            .includes(searchTerm.toLowerCase());
        const matchesCategory =
            selectedCategory === '全部' ||
            product.category === selectedCategory;

        return matchesSearch && matchesCategory;
    });

    const categories = ['全部', ...new Set(products.map(p => p.category))];

    return (
        <div>
            <input
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="搜索商品..."
            />
            <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
            >
                {categories.map(cat => (
                    <option key={cat} value={cat}>{cat}</option>
                ))}
            </select>

            <ul>
                {filteredProducts.map(product => (
                    <li key={product.id}>
                        {product.name} - ${product.price}
                    </li>
                ))}
            </ul>
            <p>显示 {filteredProducts.length} / {products.length} 个商品</p>
        </div>
    );
}


// ============================================================
//                    导出
// ============================================================

export {
    Counter,
    TextInput,
    Toggle,
    UserForm,
    TodoList,
    ProblemCounter,
    CorrectCounter,
    ShoppingCart,
    LazyInitExample,
    PersistentCounter,
    MultipleStates,
    FormWithObject,
    TemperatureCalculator,
    ProductFilter,
};

export default function UseStateTutorial() {
    return (
        <div className="tutorial">
            <h1>useState Hook 教程</h1>
            <Counter />
            <Toggle />
            <TextInput />
            <TodoList />
            <TemperatureCalculator />
        </div>
    );
}
```
