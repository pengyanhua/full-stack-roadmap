# services_di.ts

::: info 文件信息
- 📄 原文件：`01_services_di.ts`
- 🔤 语言：TypeScript (Angular)
:::

Angular 服务与依赖注入
服务 (Service) 是 Angular 中封装业务逻辑的核心机制。依赖注入 (DI) 是 Angular 最强大的特性之一。

## 完整代码

```typescript
/**
 * ============================================================
 *                    Angular 服务与依赖注入
 * ============================================================
 * 服务 (Service) 是 Angular 中封装业务逻辑的核心机制。
 * 依赖注入 (DI) 是 Angular 最强大的特性之一。
 * ============================================================
 */

import { Component, Injectable, inject, OnInit, InjectionToken } from '@angular/core';
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
 */

// --- 日志服务 ---
@Injectable({ providedIn: 'root' })
export class LoggerService {
    private logs: string[] = [];

    log(message: string) {
        const timestamp = new Date().toLocaleTimeString();
        const entry = `[${timestamp}] ${message}`;
        this.logs.push(entry);
        console.log(entry);
    }

    getLogs(): string[] { return [...this.logs]; }
    clear() { this.logs = []; }
}

// --- Todo 服务 ---
export interface Todo {
    id: number;
    title: string;
    completed: boolean;
}

@Injectable({ providedIn: 'root' })
export class TodoService {
    private todos: Todo[] = [
        { id: 1, title: '学习 Angular 基础', completed: true },
        { id: 2, title: '学习组件通信', completed: true },
        { id: 3, title: '学习服务与 DI', completed: false },
    ];
    private nextId = 4;

    constructor(private logger: LoggerService) {
        this.logger.log('TodoService 初始化');
    }

    getAll(): Todo[] { return [...this.todos]; }

    add(title: string): Todo {
        const todo: Todo = { id: this.nextId++, title, completed: false };
        this.todos.push(todo);
        this.logger.log(`添加待办: ${title}`);
        return todo;
    }

    toggle(id: number): void {
        const todo = this.todos.find(t => t.id === id);
        if (todo) {
            todo.completed = !todo.completed;
            this.logger.log(`切换待办 #${id}`);
        }
    }

    remove(id: number): void {
        const index = this.todos.findIndex(t => t.id === id);
        if (index > -1) {
            this.todos.splice(index, 1);
        }
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
 */

@Component({
    selector: 'app-todo-list',
    standalone: true,
    imports: [CommonModule, FormsModule],
    template: `
        <div class="todo-app">
            <h3>待办列表</h3>
            <div class="add-form">
                <input [(ngModel)]="newTitle" (keyup.enter)="addTodo()" placeholder="输入待办事项...">
                <button (click)="addTodo()">添加</button>
            </div>
            <ul>
                @for (todo of todos; track todo.id) {
                    <li [class.completed]="todo.completed">
                        <input type="checkbox" [checked]="todo.completed" (change)="toggleTodo(todo.id)">
                        <span>{{ todo.title }}</span>
                        <button (click)="removeTodo(todo.id)">删除</button>
                    </li>
                } @empty {
                    <li>暂无待办事项</li>
                }
            </ul>
        </div>
    `,
})
export class TodoListComponent implements OnInit {
    private todoService = inject(TodoService);
    todos: Todo[] = [];
    newTitle = '';

    ngOnInit() { this.loadTodos(); }

    loadTodos() { this.todos = this.todoService.getAll(); }
    addTodo() {
        if (this.newTitle.trim()) {
            this.todoService.add(this.newTitle.trim());
            this.newTitle = '';
            this.loadTodos();
        }
    }
    toggleTodo(id: number) { this.todoService.toggle(id); this.loadTodos(); }
    removeTodo(id: number) { this.todoService.remove(id); this.loadTodos(); }
}


// ============================================================
//                    3. 提供者层级
// ============================================================

/**
 * 【DI 提供者层级】
 *
 * 1. 根级别 (providedIn: 'root') - 全局单例
 * 2. 模块级别 (NgModule providers) - 模块内单例
 * 3. 组件级别 (Component providers) - 每个组件实例独立
 */

@Injectable() // 注意没有 providedIn
export class CounterService {
    count = 0;
    increment() { this.count++; }
    decrement() { this.count--; }
    reset() { this.count = 0; }
}

@Component({
    selector: 'app-counter-widget',
    standalone: true,
    providers: [CounterService], // 组件级别 → 每个实例独立
    template: `
        <div class="counter-widget">
            <h4>{{ label }}</h4>
            <button (click)="counter.decrement()">-</button>
            <span>{{ counter.count }}</span>
            <button (click)="counter.increment()">+</button>
        </div>
    `,
})
export class CounterWidgetComponent {
    @Input() label = '计数器';
    constructor(public counter: CounterService) {}
}


// ============================================================
//                    4. InjectionToken
// ============================================================

/**
 * 【InjectionToken】
 * - 用于注入非类类型的值（配置对象等）
 */

export interface AppConfig {
    apiUrl: string;
    appName: string;
    debug: boolean;
}

export const APP_CONFIG = new InjectionToken<AppConfig>('app.config');

@Component({
    selector: 'app-config-demo',
    standalone: true,
    providers: [
        { provide: APP_CONFIG, useValue: { apiUrl: 'https://api.example.com', appName: 'Angular 学习', debug: true } },
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
//                    5. 最佳实践
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
 */
```
