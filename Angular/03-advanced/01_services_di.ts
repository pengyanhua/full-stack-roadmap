/**
 * ============================================================
 *                    Angular 服务与依赖注入
 * ============================================================
 * 服务 (Service) 是 Angular 中封装业务逻辑的核心机制。
 * 依赖注入 (DI) 是 Angular 最强大的特性之一。
 * ============================================================
 */

import { Component, Injectable, inject, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

// ============================================================
//                    1. 创建服务
// ============================================================

/**
 * 【什么是服务】
 * - 独立于组件的可复用业务逻辑
 * - 使用 @Injectable 装饰器
 * - providedIn: 'root' 表示全局单例
 *
 * 【服务用途】
 * - 数据获取（HTTP 请求）
 * - 状态管理
 * - 业务逻辑封装
 * - 跨组件通信
 * - 日志记录
 */

// --- 日志服务 ---
@Injectable({
    providedIn: 'root', // 全局单例，自动注册
})
export class LoggerService {
    private logs: string[] = [];

    log(message: string) {
        const timestamp = new Date().toLocaleTimeString();
        const entry = `[${timestamp}] ${message}`;
        this.logs.push(entry);
        console.log(entry);
    }

    getLogs(): string[] {
        return [...this.logs];
    }

    clear() {
        this.logs = [];
    }
}

// --- Todo 服务 ---
export interface Todo {
    id: number;
    title: string;
    completed: boolean;
}

@Injectable({
    providedIn: 'root',
})
export class TodoService {
    private todos: Todo[] = [
        { id: 1, title: '学习 Angular 基础', completed: true },
        { id: 2, title: '学习组件通信', completed: true },
        { id: 3, title: '学习服务与 DI', completed: false },
        { id: 4, title: '学习路由', completed: false },
    ];
    private nextId = 5;

    constructor(private logger: LoggerService) {
        this.logger.log('TodoService 初始化');
    }

    getAll(): Todo[] {
        return [...this.todos];
    }

    add(title: string): Todo {
        const todo: Todo = {
            id: this.nextId++,
            title,
            completed: false,
        };
        this.todos.push(todo);
        this.logger.log(`添加待办: ${title}`);
        return todo;
    }

    toggle(id: number): void {
        const todo = this.todos.find(t => t.id === id);
        if (todo) {
            todo.completed = !todo.completed;
            this.logger.log(`切换待办 #${id}: ${todo.completed ? '完成' : '未完成'}`);
        }
    }

    remove(id: number): void {
        const index = this.todos.findIndex(t => t.id === id);
        if (index > -1) {
            const removed = this.todos.splice(index, 1)[0];
            this.logger.log(`删除待办: ${removed.title}`);
        }
    }

    getCompletedCount(): number {
        return this.todos.filter(t => t.completed).length;
    }
}


// ============================================================
//                    2. 注入服务
// ============================================================

/**
 * 【注入方式】
 *
 * 方式一：构造函数注入（传统）
 *   constructor(private todoService: TodoService) {}
 *
 * 方式二：inject() 函数（Angular 14+ 推荐）
 *   todoService = inject(TodoService);
 *
 * 【inject() 优势】
 * - 不需要构造函数
 * - 可以在字段初始化器中使用
 * - 更好的 tree-shaking
 * - 与函数式守卫/拦截器配合更好
 */

@Component({
    selector: 'app-todo-list',
    standalone: true,
    imports: [CommonModule, FormsModule],
    template: `
        <div class="todo-app">
            <h3>待办列表</h3>

            <!-- 添加待办 -->
            <div class="add-form">
                <input
                    [(ngModel)]="newTitle"
                    (keyup.enter)="addTodo()"
                    placeholder="输入待办事项..."
                >
                <button (click)="addTodo()">添加</button>
            </div>

            <!-- 待办列表 -->
            <ul class="todo-list">
                @for (todo of todos; track todo.id) {
                    <li [class.completed]="todo.completed">
                        <input
                            type="checkbox"
                            [checked]="todo.completed"
                            (change)="toggleTodo(todo.id)"
                        >
                        <span>{{ todo.title }}</span>
                        <button (click)="removeTodo(todo.id)">删除</button>
                    </li>
                } @empty {
                    <li class="empty">暂无待办事项</li>
                }
            </ul>

            <!-- 统计 -->
            <p class="stats">
                完成: {{ completedCount }} / {{ todos.length }}
            </p>
        </div>
    `,
    styles: [`
        .todo-app { max-width: 400px; }
        .add-form { display: flex; gap: 8px; margin-bottom: 16px; }
        .add-form input { flex: 1; padding: 8px; }
        .todo-list { list-style: none; padding: 0; }
        .todo-list li { padding: 8px; display: flex; align-items: center; gap: 8px; }
        .completed span { text-decoration: line-through; color: #999; }
        .empty { color: #999; text-align: center; }
        .stats { color: #666; font-size: 14px; }
    `]
})
export class TodoListComponent implements OnInit {
    // Angular 14+ inject() 方式注入
    private todoService = inject(TodoService);

    todos: Todo[] = [];
    newTitle = '';
    completedCount = 0;

    ngOnInit() {
        this.loadTodos();
    }

    loadTodos() {
        this.todos = this.todoService.getAll();
        this.completedCount = this.todoService.getCompletedCount();
    }

    addTodo() {
        if (this.newTitle.trim()) {
            this.todoService.add(this.newTitle.trim());
            this.newTitle = '';
            this.loadTodos();
        }
    }

    toggleTodo(id: number) {
        this.todoService.toggle(id);
        this.loadTodos();
    }

    removeTodo(id: number) {
        this.todoService.remove(id);
        this.loadTodos();
    }
}


// ============================================================
//                    3. 提供者层级
// ============================================================

/**
 * 【DI 提供者层级】
 *
 * 1. 根级别 (providedIn: 'root')
 *    - 全局单例
 *    - 所有组件共享同一实例
 *
 * 2. 模块级别 (NgModule providers)
 *    - 模块内单例
 *    - 懒加载模块有独立实例
 *
 * 3. 组件级别 (Component providers)
 *    - 每个组件实例一个服务实例
 *    - 子组件可以继承父组件的服务
 *
 * 【何时使用组件级别】
 * - 每个组件需要独立的状态
 * - 组件销毁时需要清理服务资源
 */

// --- 计数器服务（非全局） ---
@Injectable() // 注意没有 providedIn
export class CounterService {
    count = 0;

    increment() { this.count++; }
    decrement() { this.count--; }
    reset() { this.count = 0; }
}

// --- 每个组件实例有独立的 CounterService ---
@Component({
    selector: 'app-counter-widget',
    standalone: true,
    providers: [CounterService], // 组件级别提供 → 每个实例独立
    template: `
        <div class="counter-widget">
            <h4>{{ label }}</h4>
            <button (click)="counter.decrement()">-</button>
            <span>{{ counter.count }}</span>
            <button (click)="counter.increment()">+</button>
            <button (click)="counter.reset()">重置</button>
        </div>
    `,
    styles: [`
        .counter-widget { display: flex; align-items: center; gap: 8px; padding: 8px; border: 1px solid #ddd; margin: 4px 0; border-radius: 4px; }
        span { min-width: 30px; text-align: center; font-weight: bold; }
    `]
})
export class CounterWidgetComponent {
    @Input() label = '计数器';

    // 构造函数注入（传统方式）
    constructor(public counter: CounterService) {}
}


// ============================================================
//                    4. 使用接口和令牌
// ============================================================

/**
 * 【InjectionToken】
 * - 用于注入非类类型的值（字符串、对象、函数等）
 * - 解决接口无法作为 DI 令牌的问题
 * - 适合注入配置对象
 */

import { InjectionToken } from '@angular/core';

// 定义配置接口
export interface AppConfig {
    apiUrl: string;
    appName: string;
    debug: boolean;
}

// 创建注入令牌
export const APP_CONFIG = new InjectionToken<AppConfig>('app.config');

// 配置值
export const appConfig: AppConfig = {
    apiUrl: 'https://api.example.com',
    appName: 'Angular 学习',
    debug: true,
};

// --- 使用配置的组件 ---
@Component({
    selector: 'app-config-demo',
    standalone: true,
    providers: [
        { provide: APP_CONFIG, useValue: appConfig },
    ],
    template: `
        <div>
            <h4>应用配置</h4>
            <p>应用名称: {{ config.appName }}</p>
            <p>API 地址: {{ config.apiUrl }}</p>
            <p>调试模式: {{ config.debug ? '开启' : '关闭' }}</p>
        </div>
    `,
})
export class ConfigDemoComponent {
    config = inject(APP_CONFIG);
}


// ============================================================
//                    5. 服务间通信
// ============================================================

/**
 * 【服务间依赖】
 * - 服务可以注入其他服务
 * - 构成一个依赖关系图
 * - Angular DI 自动解析依赖链
 *
 * 【适用场景】
 * - 日志服务被多个服务依赖
 * - 缓存服务被数据服务依赖
 * - 认证服务被 API 服务依赖
 */

@Injectable({ providedIn: 'root' })
export class NotificationService {
    private logger = inject(LoggerService);
    messages: string[] = [];

    success(msg: string) {
        this.messages.push(`✅ ${msg}`);
        this.logger.log(`通知: ${msg}`);
    }

    error(msg: string) {
        this.messages.push(`❌ ${msg}`);
        this.logger.log(`错误通知: ${msg}`);
    }

    clear() {
        this.messages = [];
    }
}


// ============================================================
//                    6. 父组件示例
// ============================================================

@Component({
    selector: 'app-di-demo',
    standalone: true,
    imports: [CommonModule, TodoListComponent, CounterWidgetComponent, ConfigDemoComponent],
    template: `
        <h2>服务与依赖注入演示</h2>

        <!-- Todo 列表（使用全局服务） -->
        <h3>1. 全局服务</h3>
        <app-todo-list />

        <!-- 独立计数器（组件级服务） -->
        <h3>2. 组件级服务（每个实例独立）</h3>
        <app-counter-widget label="计数器 A" />
        <app-counter-widget label="计数器 B" />
        <p style="color: #666; font-size: 14px">
            ↑ 两个计数器互不影响，各自有独立的 CounterService 实例
        </p>

        <!-- InjectionToken 配置 -->
        <h3>3. InjectionToken</h3>
        <app-config-demo />

        <!-- 日志查看 -->
        <h3>4. 日志服务</h3>
        <button (click)="showLogs()">查看日志</button>
        <button (click)="clearLogs()">清空日志</button>
        @if (logs.length > 0) {
            <ul class="log-list">
                @for (log of logs; track log) {
                    <li>{{ log }}</li>
                }
            </ul>
        }
    `,
    styles: [`
        .log-list { background: #263238; color: #a5d6a7; padding: 12px; border-radius: 4px; font-family: monospace; font-size: 13px; list-style: none; }
        .log-list li { padding: 2px 0; }
    `]
})
export class DiDemoComponent {
    private logger = inject(LoggerService);
    logs: string[] = [];

    showLogs() {
        this.logs = this.logger.getLogs();
    }

    clearLogs() {
        this.logger.clear();
        this.logs = [];
    }
}


// ============================================================
//                    7. 最佳实践
// ============================================================

/**
 * 【DI 最佳实践】
 *
 * ✅ 推荐做法：
 * 1. 使用 inject() 函数替代构造函数注入
 * 2. 全局服务用 providedIn: 'root'
 * 3. 组件专用状态用组件级 providers
 * 4. 配置对象用 InjectionToken
 * 5. 服务保持单一职责
 *
 * ❌ 避免做法：
 * 1. 在服务中引用组件 → 服务应该是纯逻辑
 * 2. 服务之间循环依赖 → 重构拆分
 * 3. 到处使用 new 创建实例 → 让 DI 管理
 * 4. 在组件中放置过多业务逻辑 → 抽取到服务
 */
