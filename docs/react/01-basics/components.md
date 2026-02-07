# components.jsx

::: info 文件信息
- 📄 原文件：`02_components.jsx`
- 🔤 语言：jsx
:::

React 组件基础
React 组件是构建 UI 的基本单元。
组件可以是函数组件或类组件，现代 React 推荐使用函数组件。

## 完整代码

```jsx
/**
 * ============================================================
 *                    React 组件基础
 * ============================================================
 * React 组件是构建 UI 的基本单元。
 * 组件可以是函数组件或类组件，现代 React 推荐使用函数组件。
 * ============================================================
 */

import React, { Component } from 'react';

// ============================================================
//                    1. 函数组件
// ============================================================

/**
 * 【函数组件】
 *
 * 函数组件是最简单的组件形式：
 * - 接收 props 作为参数
 * - 返回 JSX（描述 UI）
 * - 没有 this 关键字
 * - 配合 Hooks 可以使用状态和生命周期
 *
 * 【命名规则】
 * - 组件名必须以大写字母开头
 * - 使用 PascalCase 命名（如 UserProfile）
 * - 文件名通常与组件名一致
 */

// --- 最简单的函数组件 ---
// 不接收任何参数，返回固定的 JSX
function Welcome() {
    return <h1>欢迎来到 React 世界!</h1>;
}

// --- 箭头函数组件 ---
// 使用箭头函数语法，更简洁
const Greeting = () => {
    return <p>你好!</p>;
};

// --- 带隐式返回的箭头函数 ---
// 当只有一个表达式时，可以省略 return 和花括号
const SimpleButton = () => <button>点击我</button>;


// ============================================================
//                    2. Props（属性）
// ============================================================

/**
 * 【Props 是什么】
 *
 * Props（properties）是组件的输入：
 * - 从父组件传递给子组件
 * - 在组件内部是只读的（不能修改）
 * - 可以是任意类型：字符串、数字、对象、数组、函数等
 *
 * 【Props 的特点】
 * - 单向数据流：数据从父组件流向子组件
 * - 不可变：组件不能修改自己的 props
 * - 声明式：通过 props 声明组件需要什么数据
 */

// --- 接收 Props ---
// props 是一个对象，包含所有传递给组件的属性
function UserGreeting(props) {
    // props = { name: "Alice", age: 25 }
    return (
        <div>
            <h2>你好, {props.name}!</h2>
            <p>你今年 {props.age} 岁</p>
        </div>
    );
}

// 使用方式:
// <UserGreeting name="Alice" age={25} />

// --- 解构 Props ---
// 推荐使用解构语法，代码更清晰
function UserCard({ name, email, avatar }) {
    return (
        <div className="user-card">
            <img src={avatar} alt={name} />
            <h3>{name}</h3>
            <p>{email}</p>
        </div>
    );
}

// 使用方式:
// <UserCard
//     name="Alice"
//     email="alice@example.com"
//     avatar="/images/alice.jpg"
// />

// --- 默认 Props ---
// 为 props 设置默认值，当父组件没有传递时使用
function Button({ text = '按钮', type = 'primary', disabled = false }) {
    return (
        <button
            className={`btn btn-${type}`}
            disabled={disabled}
        >
            {text}
        </button>
    );
}

// 使用方式:
// <Button />                        // 使用全部默认值
// <Button text="提交" />            // 只覆盖 text
// <Button text="删除" type="danger" /> // 覆盖多个

// --- 展开 Props ---
// 使用展开运算符传递所有属性
function Input(props) {
    // 接收所有属性并传递给 input 元素
    return <input className="custom-input" {...props} />;
}

// 使用方式:
// <Input type="text" placeholder="请输入" maxLength={100} />


// ============================================================
//                    3. children Props
// ============================================================

/**
 * 【children 是什么】
 *
 * children 是一个特殊的 prop：
 * - 表示组件的子元素
 * - 可以是文本、元素、组件或数组
 * - 用于创建容器组件或布局组件
 */

// --- 基本 children 用法 ---
function Card({ title, children }) {
    return (
        <div className="card">
            <div className="card-header">
                <h3>{title}</h3>
            </div>
            <div className="card-body">
                {/* children 会渲染在这里 */}
                {children}
            </div>
        </div>
    );
}

// 使用方式:
// <Card title="用户信息">
//     <p>姓名: Alice</p>
//     <p>邮箱: alice@example.com</p>
// </Card>

// --- 容器组件 ---
// 用于包装其他组件，提供通用功能
function Container({ maxWidth = '1200px', children }) {
    const style = {
        maxWidth,
        margin: '0 auto',
        padding: '0 20px',
    };

    return <div style={style}>{children}</div>;
}

// --- 布局组件 ---
// 用于页面布局
function Layout({ header, sidebar, children, footer }) {
    return (
        <div className="layout">
            <header className="layout-header">{header}</header>
            <div className="layout-content">
                <aside className="layout-sidebar">{sidebar}</aside>
                <main className="layout-main">{children}</main>
            </div>
            <footer className="layout-footer">{footer}</footer>
        </div>
    );
}

// 使用方式:
// <Layout
//     header={<Navbar />}
//     sidebar={<Menu />}
//     footer={<Footer />}
// >
//     <ArticleContent />
// </Layout>


// ============================================================
//                    4. Props 类型和验证
// ============================================================

/**
 * 【PropTypes】
 *
 * PropTypes 用于在开发环境中验证 props 类型：
 * - 帮助捕获错误
 * - 作为组件文档
 * - 只在开发环境运行，不影响生产性能
 *
 * 【TypeScript 替代】
 * 在 TypeScript 项目中，通常使用接口定义 props 类型
 */

import PropTypes from 'prop-types';

// --- PropTypes 验证 ---
function UserProfile({ name, age, email, isAdmin, tags, onUpdate }) {
    return (
        <div>
            <h2>{name}</h2>
            <p>年龄: {age}</p>
            <p>邮箱: {email}</p>
            {isAdmin && <span className="badge">管理员</span>}
            <div>
                标签: {tags.join(', ')}
            </div>
            <button onClick={onUpdate}>更新</button>
        </div>
    );
}

// 定义 props 类型
UserProfile.propTypes = {
    // 必需的字符串
    name: PropTypes.string.isRequired,

    // 可选的数字
    age: PropTypes.number,

    // 可选的字符串
    email: PropTypes.string,

    // 可选的布尔值
    isAdmin: PropTypes.bool,

    // 字符串数组
    tags: PropTypes.arrayOf(PropTypes.string),

    // 必需的函数
    onUpdate: PropTypes.func.isRequired,
};

// 定义默认值
UserProfile.defaultProps = {
    age: 0,
    email: '',
    isAdmin: false,
    tags: [],
};

// --- TypeScript 方式（推荐） ---
/*
interface UserProfileProps {
    name: string;
    age?: number;
    email?: string;
    isAdmin?: boolean;
    tags?: string[];
    onUpdate: () => void;
}

function UserProfile({
    name,
    age = 0,
    email = '',
    isAdmin = false,
    tags = [],
    onUpdate,
}: UserProfileProps) {
    // ...
}
*/


// ============================================================
//                    5. 组件组合
// ============================================================

/**
 * 【组件组合】
 *
 * React 推荐使用组合而非继承：
 * - 小组件组合成大组件
 * - 通过 props 和 children 实现灵活性
 * - 遵循单一职责原则
 */

// --- 小组件 ---
function Avatar({ src, alt, size = 'medium' }) {
    const sizes = {
        small: '32px',
        medium: '48px',
        large: '64px',
    };

    return (
        <img
            src={src}
            alt={alt}
            style={{
                width: sizes[size],
                height: sizes[size],
                borderRadius: '50%',
            }}
        />
    );
}

function UserName({ name, isOnline }) {
    return (
        <span className="username">
            {name}
            {isOnline && <span className="online-dot">●</span>}
        </span>
    );
}

// --- 组合成复杂组件 ---
function UserInfo({ user }) {
    return (
        <div className="user-info">
            <Avatar
                src={user.avatar}
                alt={user.name}
                size="medium"
            />
            <div className="user-details">
                <UserName
                    name={user.name}
                    isOnline={user.isOnline}
                />
                <span className="user-email">{user.email}</span>
            </div>
        </div>
    );
}

// --- 特化组件 ---
// 通过预设 props 创建特化版本
function PrimaryButton(props) {
    return <Button {...props} type="primary" />;
}

function DangerButton(props) {
    return <Button {...props} type="danger" />;
}

function IconButton({ icon, children, ...props }) {
    return (
        <Button {...props}>
            <span className="icon">{icon}</span>
            {children}
        </Button>
    );
}


// ============================================================
//                    6. 条件渲染组件
// ============================================================

/**
 * 【条件渲染模式】
 *
 * 根据条件决定渲染哪个组件或元素
 */

// --- 登录状态组件 ---
function LoginButton({ onLogin }) {
    return <button onClick={onLogin}>登录</button>;
}

function LogoutButton({ onLogout }) {
    return <button onClick={onLogout}>退出</button>;
}

function AuthButton({ isLoggedIn, onLogin, onLogout }) {
    // 根据登录状态返回不同组件
    if (isLoggedIn) {
        return <LogoutButton onLogout={onLogout} />;
    }
    return <LoginButton onLogin={onLogin} />;
}

// --- 加载状态组件 ---
function LoadingSpinner() {
    return <div className="spinner">加载中...</div>;
}

function ErrorMessage({ message }) {
    return <div className="error">{message}</div>;
}

function DataDisplay({ data }) {
    return <div className="data">{JSON.stringify(data)}</div>;
}

function AsyncContent({ loading, error, data }) {
    // 优先级：错误 > 加载 > 数据
    if (error) {
        return <ErrorMessage message={error} />;
    }

    if (loading) {
        return <LoadingSpinner />;
    }

    if (data) {
        return <DataDisplay data={data} />;
    }

    return null;
}


// ============================================================
//                    7. 事件处理
// ============================================================

/**
 * 【事件处理】
 *
 * React 事件处理的特点：
 * - 事件名使用 camelCase（onClick 而非 onclick）
 * - 传递函数而非字符串
 * - 事件对象是合成事件（SyntheticEvent）
 * - 需要显式调用 preventDefault() 阻止默认行为
 */

// --- 基本事件处理 ---
function ClickCounter() {
    // 事件处理函数
    const handleClick = () => {
        console.log('按钮被点击');
    };

    // 带事件对象的处理函数
    const handleClickWithEvent = (event) => {
        console.log('点击位置:', event.clientX, event.clientY);
        console.log('目标元素:', event.target);
    };

    return (
        <div>
            {/* 绑定事件处理函数 */}
            <button onClick={handleClick}>点击我</button>

            {/* 带事件对象 */}
            <button onClick={handleClickWithEvent}>查看事件</button>

            {/* 内联函数（简单逻辑时使用） */}
            <button onClick={() => console.log('内联点击')}>
                内联处理
            </button>
        </div>
    );
}

// --- 传递参数给事件处理函数 ---
function ItemList() {
    const items = ['苹果', '香蕉', '橙子'];

    // 方式1: 使用箭头函数包装
    const handleItemClick = (item, index) => {
        console.log(`点击了第 ${index + 1} 个: ${item}`);
    };

    // 方式2: 使用柯里化
    const createClickHandler = (item) => () => {
        console.log(`点击了: ${item}`);
    };

    return (
        <ul>
            {items.map((item, index) => (
                <li key={index}>
                    {/* 方式1: 箭头函数 */}
                    <button onClick={() => handleItemClick(item, index)}>
                        {item}
                    </button>

                    {/* 方式2: 柯里化 */}
                    <button onClick={createClickHandler(item)}>
                        {item} (柯里化)
                    </button>
                </li>
            ))}
        </ul>
    );
}

// --- 表单事件 ---
function FormExample() {
    const handleSubmit = (event) => {
        // 阻止表单默认提交行为
        event.preventDefault();
        console.log('表单提交');
    };

    const handleInputChange = (event) => {
        // 获取输入值
        const value = event.target.value;
        console.log('输入值:', value);
    };

    const handleKeyDown = (event) => {
        // 检测按键
        if (event.key === 'Enter') {
            console.log('按下回车键');
        }
    };

    return (
        <form onSubmit={handleSubmit}>
            <input
                type="text"
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                placeholder="输入内容..."
            />
            <button type="submit">提交</button>
        </form>
    );
}


// ============================================================
//                    8. 类组件（了解即可）
// ============================================================

/**
 * 【类组件】
 *
 * 类组件是 React 早期的组件写法：
 * - 继承 React.Component
 * - 使用 this.props 访问属性
 * - 使用 this.state 管理状态
 * - 有生命周期方法
 *
 * 【现代 React】
 * - 推荐使用函数组件 + Hooks
 * - 类组件仍然可用，但不推荐新代码使用
 * - 了解类组件有助于维护旧代码
 */

class ClassComponent extends Component {
    // 构造函数
    constructor(props) {
        super(props);  // 必须调用 super(props)

        // 初始化状态
        this.state = {
            count: 0,
        };

        // 绑定方法（或使用箭头函数）
        this.handleClick = this.handleClick.bind(this);
    }

    // 事件处理方法
    handleClick() {
        // 使用 this.setState 更新状态
        this.setState({ count: this.state.count + 1 });
    }

    // 箭头函数方法（自动绑定 this）
    handleReset = () => {
        this.setState({ count: 0 });
    };

    // 生命周期方法
    componentDidMount() {
        console.log('组件已挂载');
    }

    componentDidUpdate(prevProps, prevState) {
        if (prevState.count !== this.state.count) {
            console.log('count 已更新:', this.state.count);
        }
    }

    componentWillUnmount() {
        console.log('组件将卸载');
    }

    // 渲染方法（必需）
    render() {
        return (
            <div>
                <h2>{this.props.title}</h2>
                <p>计数: {this.state.count}</p>
                <button onClick={this.handleClick}>增加</button>
                <button onClick={this.handleReset}>重置</button>
            </div>
        );
    }
}


// ============================================================
//                    9. 组件设计最佳实践
// ============================================================

/**
 * 【组件设计原则】
 *
 * 1. 单一职责
 *    - 一个组件只做一件事
 *    - 如果组件太复杂，拆分成多个小组件
 *
 * 2. 可复用性
 *    - 通过 props 控制行为和外观
 *    - 避免硬编码
 *
 * 3. 可组合性
 *    - 小组件组合成大组件
 *    - 使用 children 实现灵活布局
 *
 * 4. 可测试性
 *    - 纯组件（相同 props 产生相同输出）
 *    - 逻辑和 UI 分离
 */

// --- 好的组件设计示例 ---

// 通用列表组件
function List({ items, renderItem, emptyMessage = '暂无数据' }) {
    if (items.length === 0) {
        return <p className="empty">{emptyMessage}</p>;
    }

    return (
        <ul className="list">
            {items.map((item, index) => (
                <li key={item.id || index} className="list-item">
                    {renderItem(item)}
                </li>
            ))}
        </ul>
    );
}

// 使用方式:
// <List
//     items={users}
//     renderItem={(user) => <UserCard user={user} />}
//     emptyMessage="没有用户"
// />

// 通用模态框组件
function Modal({ isOpen, onClose, title, children, footer }) {
    if (!isOpen) return null;

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div
                className="modal-content"
                onClick={(e) => e.stopPropagation()}
            >
                <div className="modal-header">
                    <h2>{title}</h2>
                    <button
                        className="modal-close"
                        onClick={onClose}
                        aria-label="关闭"
                    >
                        ×
                    </button>
                </div>
                <div className="modal-body">{children}</div>
                {footer && (
                    <div className="modal-footer">{footer}</div>
                )}
            </div>
        </div>
    );
}

// 使用方式:
// <Modal
//     isOpen={showModal}
//     onClose={() => setShowModal(false)}
//     title="确认删除"
//     footer={
//         <>
//             <Button onClick={handleCancel}>取消</Button>
//             <Button type="danger" onClick={handleDelete}>删除</Button>
//         </>
//     }
// >
//     <p>确定要删除这条记录吗？</p>
// </Modal>


// ============================================================
//                    导出示例组件
// ============================================================

export {
    Welcome,
    Greeting,
    SimpleButton,
    UserGreeting,
    UserCard,
    Button,
    Card,
    Container,
    Layout,
    UserProfile,
    Avatar,
    UserName,
    UserInfo,
    AuthButton,
    AsyncContent,
    ClickCounter,
    ItemList,
    FormExample,
    ClassComponent,
    List,
    Modal,
};

export default function ComponentsTutorial() {
    const sampleUser = {
        name: 'Alice',
        email: 'alice@example.com',
        avatar: '/default-avatar.png',
        isOnline: true,
    };

    return (
        <div className="tutorial">
            <h1>React 组件教程</h1>
            <Welcome />
            <UserInfo user={sampleUser} />
            <Card title="卡片示例">
                <p>这是卡片内容</p>
            </Card>
            <ClickCounter />
        </div>
    );
}
```
