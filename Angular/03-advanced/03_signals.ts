/**
 * ============================================================
 *                    Angular Signals 响应式
 * ============================================================
 * Signals 是 Angular 16+ 引入的全新响应式原语。
 * 提供更细粒度的变更检测和更好的性能。
 * ============================================================
 */

import { Component, signal, computed, effect, untracked, Injectable, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

// ============================================================
//                    1. Signal 基础
// ============================================================

/**
 * 【什么是 Signal】
 * - 一个包含值的响应式包装器
 * - 读取值: signal() （函数调用）
 * - 修改值: signal.set() / signal.update()
 * - 类似 Vue 的 ref()，但不需要 .value
 *
 * 【与 RxJS 的区别】
 * - Signal: 同步、始终有当前值、更简单
 * - Observable: 异步流、可能没有值、更强大但更复杂
 * - 两者可以互转: toSignal() / toObservable()
 */

@Component({
    selector: 'app-signal-basics',
    standalone: true,
    template: `
        <h3>Signal 基础</h3>

        <!-- 读取 Signal 值 -->
        <p>计数: {{ count() }}</p>
        <p>名字: {{ name() }}</p>

        <!-- 修改 Signal -->
        <button (click)="increment()">+1</button>
        <button (click)="decrement()">-1</button>
        <button (click)="reset()">重置</button>
        <button (click)="setName('Angular')">设为 Angular</button>
        <button (click)="setName('TypeScript')">设为 TypeScript</button>
    `,
})
export class SignalBasicsComponent {
    // 创建 Signal
    count = signal(0);
    name = signal('Angular');

    increment() {
        // update: 基于当前值更新
        this.count.update(value => value + 1);
    }

    decrement() {
        this.count.update(value => value - 1);
    }

    reset() {
        // set: 直接设置新值
        this.count.set(0);
    }

    setName(newName: string) {
        this.name.set(newName);
    }
}


// ============================================================
//                    2. Computed Signal
// ============================================================

/**
 * 【computed()】
 * - 基于其他 Signal 自动计算的派生值
 * - 类似 Vue 的 computed
 * - 惰性求值: 只在被读取时计算
 * - 自动缓存: 依赖不变时不重新计算
 * - 只读: 不能调用 set/update
 */

@Component({
    selector: 'app-computed-demo',
    standalone: true,
    imports: [CommonModule, FormsModule],
    template: `
        <h3>Computed Signal</h3>

        <!-- 购物车示例 -->
        <div class="cart">
            @for (item of items(); track item.name) {
                <div class="cart-item">
                    <span>{{ item.name }}</span>
                    <span>¥{{ item.price }} × {{ item.quantity }}</span>
                    <button (click)="addQuantity(item.name)">+</button>
                    <button (click)="removeQuantity(item.name)">-</button>
                </div>
            }
        </div>

        <!-- Computed 值自动更新 -->
        <div class="summary">
            <p>商品数量: {{ totalItems() }} 件</p>
            <p>总价: ¥{{ totalPrice() }}</p>
            <p>折扣 (满100减10): ¥{{ discount() }}</p>
            <p><strong>实付: ¥{{ finalPrice() }}</strong></p>
        </div>
    `,
    styles: [`
        .cart { margin: 8px 0; }
        .cart-item { display: flex; align-items: center; gap: 12px; padding: 4px 0; }
        .summary { background: #f5f5f5; padding: 12px; border-radius: 4px; margin-top: 8px; }
    `]
})
export class ComputedDemoComponent {
    // 原始 Signal
    items = signal([
        { name: 'Angular 实战', price: 59, quantity: 1 },
        { name: 'TypeScript 入门', price: 39, quantity: 2 },
        { name: 'RxJS 精通', price: 49, quantity: 1 },
    ]);

    // Computed Signal - 自动追踪依赖
    totalItems = computed(() =>
        this.items().reduce((sum, item) => sum + item.quantity, 0)
    );

    totalPrice = computed(() =>
        this.items().reduce((sum, item) => sum + item.price * item.quantity, 0)
    );

    discount = computed(() =>
        this.totalPrice() >= 100 ? 10 : 0
    );

    // computed 可以依赖其他 computed
    finalPrice = computed(() =>
        this.totalPrice() - this.discount()
    );

    addQuantity(name: string) {
        this.items.update(items =>
            items.map(item =>
                item.name === name ? { ...item, quantity: item.quantity + 1 } : item
            )
        );
    }

    removeQuantity(name: string) {
        this.items.update(items =>
            items.map(item =>
                item.name === name && item.quantity > 0
                    ? { ...item, quantity: item.quantity - 1 }
                    : item
            )
        );
    }
}


// ============================================================
//                    3. Effect
// ============================================================

/**
 * 【effect()】
 * - 当依赖的 Signal 变化时自动执行的副作用
 * - 类似 Vue 的 watchEffect
 * - 自动追踪依赖（不需要显式指定）
 * - 在组件销毁时自动清理
 *
 * 【使用场景】
 * - 日志记录
 * - 同步到 localStorage
 * - 调用外部 API
 * - DOM 操作
 *
 * 【untracked()】
 * - 在 effect 中读取 Signal 但不追踪它
 * - 防止不必要的 effect 重新执行
 */

@Component({
    selector: 'app-effect-demo',
    standalone: true,
    imports: [FormsModule],
    template: `
        <h3>Effect 副作用</h3>

        <div>
            <label>主题: </label>
            <select [value]="theme()" (change)="onThemeChange($event)">
                <option value="light">浅色</option>
                <option value="dark">深色</option>
                <option value="auto">自动</option>
            </select>
        </div>

        <div>
            <label>字体大小: </label>
            <input
                type="range" min="12" max="24"
                [value]="fontSize()"
                (input)="onFontSizeChange($event)"
            >
            <span>{{ fontSize() }}px</span>
        </div>

        <div [style.font-size.px]="fontSize()"
             [style.background]="theme() === 'dark' ? '#333' : '#fff'"
             [style.color]="theme() === 'dark' ? '#fff' : '#333'"
             style="padding: 16px; margin: 8px 0; border-radius: 4px;">
            预览效果：Hello Angular Signals!
        </div>

        <div class="log">
            <h4>Effect 日志:</h4>
            @for (log of effectLogs; track log) {
                <p>{{ log }}</p>
            }
        </div>
    `,
    styles: [`
        .log { background: #f5f5f5; padding: 8px; font-size: 12px; max-height: 120px; overflow-y: auto; }
    `]
})
export class EffectDemoComponent {
    theme = signal<'light' | 'dark' | 'auto'>('light');
    fontSize = signal(16);
    effectLogs: string[] = [];

    constructor() {
        // Effect: 主题变化时保存到 localStorage
        effect(() => {
            const currentTheme = this.theme();
            this.effectLogs.push(`主题变更为: ${currentTheme}`);
            // localStorage.setItem('theme', currentTheme);
        });

        // Effect: 字体大小变化时记录
        effect(() => {
            const size = this.fontSize();
            // 使用 untracked 读取 theme 但不追踪它
            const currentTheme = untracked(() => this.theme());
            this.effectLogs.push(`字体: ${size}px (主题: ${currentTheme}，不触发此 effect)`);
        });
    }

    onThemeChange(event: Event) {
        this.theme.set((event.target as HTMLSelectElement).value as any);
    }

    onFontSizeChange(event: Event) {
        this.fontSize.set(Number((event.target as HTMLInputElement).value));
    }
}


// ============================================================
//                    4. Signal 在服务中的使用
// ============================================================

/**
 * 【Signal Store 模式】
 * - 用 Signal 在服务中管理全局状态
 * - 替代简单的 RxJS BehaviorSubject
 * - 组件自动响应状态变化
 */

interface User {
    id: number;
    name: string;
    email: string;
}

@Injectable({ providedIn: 'root' })
export class UserStore {
    // 状态 Signal
    private _users = signal<User[]>([
        { id: 1, name: '小明', email: 'ming@example.com' },
        { id: 2, name: '小红', email: 'hong@example.com' },
    ]);
    private _selectedId = signal<number | null>(null);
    private _loading = signal(false);

    // 只读 Signal（对外暴露）
    readonly users = this._users.asReadonly();
    readonly loading = this._loading.asReadonly();

    // Computed
    readonly selectedUser = computed(() => {
        const id = this._selectedId();
        return id ? this._users().find(u => u.id === id) ?? null : null;
    });

    readonly userCount = computed(() => this._users().length);

    // Actions
    select(id: number) {
        this._selectedId.set(id);
    }

    add(user: Omit<User, 'id'>) {
        const newId = Math.max(...this._users().map(u => u.id), 0) + 1;
        this._users.update(users => [...users, { ...user, id: newId }]);
    }

    remove(id: number) {
        this._users.update(users => users.filter(u => u.id !== id));
        if (this._selectedId() === id) {
            this._selectedId.set(null);
        }
    }
}

@Component({
    selector: 'app-user-manager',
    standalone: true,
    imports: [CommonModule, FormsModule],
    template: `
        <h3>用户管理（Signal Store）</h3>

        <p>用户数量: {{ store.userCount() }}</p>

        <ul>
            @for (user of store.users(); track user.id) {
                <li
                    (click)="store.select(user.id)"
                    [class.selected]="store.selectedUser()?.id === user.id"
                >
                    {{ user.name }} ({{ user.email }})
                    <button (click)="store.remove(user.id); $event.stopPropagation()">删除</button>
                </li>
            }
        </ul>

        @if (store.selectedUser(); as user) {
            <div class="detail">
                <h4>选中: {{ user.name }}</h4>
                <p>邮箱: {{ user.email }}</p>
            </div>
        }

        <div class="add-form">
            <input [(ngModel)]="newName" placeholder="姓名">
            <input [(ngModel)]="newEmail" placeholder="邮箱">
            <button (click)="addUser()">添加用户</button>
        </div>
    `,
    styles: [`
        li { cursor: pointer; padding: 4px 8px; }
        li.selected { background: #e3f2fd; border-radius: 4px; }
        .detail { background: #f5f5f5; padding: 12px; border-radius: 4px; margin: 8px 0; }
        .add-form { display: flex; gap: 8px; margin-top: 8px; }
    `]
})
export class UserManagerComponent {
    store = inject(UserStore);
    newName = '';
    newEmail = '';

    addUser() {
        if (this.newName && this.newEmail) {
            this.store.add({ name: this.newName, email: this.newEmail });
            this.newName = '';
            this.newEmail = '';
        }
    }
}


// ============================================================
//                    5. 最佳实践
// ============================================================

/**
 * 【Signal 最佳实践】
 *
 * ✅ 推荐做法：
 * 1. 简单状态管理优先使用 Signal（替代 BehaviorSubject）
 * 2. 派生数据用 computed()（自动缓存）
 * 3. 副作用用 effect()（自动清理）
 * 4. 服务中暴露 readonly Signal（防止外部修改）
 * 5. 使用 untracked() 避免不必要的依赖追踪
 *
 * ❌ 避免做法：
 * 1. 在 effect 中修改其他 Signal → 可能导致循环
 * 2. 忽略 computed 的缓存能力 → 不要用 effect 模拟 computed
 * 3. 所有场景都用 Signal → 异步流仍然用 RxJS
 * 4. 在模板中频繁调用方法 → 用 computed 替代
 *
 * 【Signal vs RxJS 选择指南】
 * - 同步状态、UI 绑定 → Signal
 * - HTTP 请求、事件流、复杂异步操作 → RxJS
 * - 两者可以用 toSignal()/toObservable() 互转
 */
