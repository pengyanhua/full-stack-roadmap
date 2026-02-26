# signals.ts

::: info 文件信息
- 📄 原文件：`03_signals.ts`
- 🔤 语言：TypeScript (Angular)
:::

Angular Signals 响应式
Signals 是 Angular 16+ 引入的全新响应式原语。提供更细粒度的变更检测和更好的性能。

## 完整代码

```typescript
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
 * - Observable: 异步流、更强大但更复杂
 * - 两者可以互转: toSignal() / toObservable()
 */

@Component({
    selector: 'app-signal-basics',
    standalone: true,
    template: `
        <h3>Signal 基础</h3>
        <p>计数: {{ count() }}</p>
        <p>名字: {{ name() }}</p>
        <button (click)="increment()">+1</button>
        <button (click)="decrement()">-1</button>
        <button (click)="reset()">重置</button>
    `,
})
export class SignalBasicsComponent {
    count = signal(0);
    name = signal('Angular');

    increment() { this.count.update(v => v + 1); }
    decrement() { this.count.update(v => v - 1); }
    reset() { this.count.set(0); }
}


// ============================================================
//                    2. Computed Signal
// ============================================================

/**
 * 【computed()】
 * - 基于其他 Signal 自动计算的派生值
 * - 惰性求值: 只在被读取时计算
 * - 自动缓存: 依赖不变时不重新计算
 * - 只读: 不能调用 set/update
 */

@Component({
    selector: 'app-computed-demo',
    standalone: true,
    imports: [CommonModule],
    template: `
        <h3>购物车（Computed Signal）</h3>
        <div>
            @for (item of items(); track item.name) {
                <div>
                    {{ item.name }} - ¥{{ item.price }} × {{ item.quantity }}
                    <button (click)="addQuantity(item.name)">+</button>
                    <button (click)="removeQuantity(item.name)">-</button>
                </div>
            }
        </div>
        <div>
            <p>商品数量: {{ totalItems() }} 件</p>
            <p>总价: ¥{{ totalPrice() }}</p>
            <p>折扣 (满100减10): ¥{{ discount() }}</p>
            <p><strong>实付: ¥{{ finalPrice() }}</strong></p>
        </div>
    `,
})
export class ComputedDemoComponent {
    items = signal([
        { name: 'Angular 实战', price: 59, quantity: 1 },
        { name: 'TypeScript 入门', price: 39, quantity: 2 },
        { name: 'RxJS 精通', price: 49, quantity: 1 },
    ]);

    totalItems = computed(() =>
        this.items().reduce((sum, item) => sum + item.quantity, 0)
    );
    totalPrice = computed(() =>
        this.items().reduce((sum, item) => sum + item.price * item.quantity, 0)
    );
    discount = computed(() => this.totalPrice() >= 100 ? 10 : 0);
    finalPrice = computed(() => this.totalPrice() - this.discount());

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
                    ? { ...item, quantity: item.quantity - 1 } : item
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
 * - 自动追踪依赖
 * - 组件销毁时自动清理
 *
 * 【untracked()】
 * - 在 effect 中读取 Signal 但不追踪它
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
            </select>
        </div>
        <div>
            <label>字体大小: </label>
            <input type="range" min="12" max="24" [value]="fontSize()" (input)="onFontSizeChange($event)">
            <span>{{ fontSize() }}px</span>
        </div>
        <div [style.font-size.px]="fontSize()"
             [style.background]="theme() === 'dark' ? '#333' : '#fff'"
             [style.color]="theme() === 'dark' ? '#fff' : '#333'"
             style="padding: 16px; margin: 8px 0; border-radius: 4px;">
            预览效果：Hello Angular Signals!
        </div>
    `,
})
export class EffectDemoComponent {
    theme = signal<'light' | 'dark'>('light');
    fontSize = signal(16);

    constructor() {
        effect(() => {
            const currentTheme = this.theme();
            console.log(`主题变更为: ${currentTheme}`);
        });

        effect(() => {
            const size = this.fontSize();
            const currentTheme = untracked(() => this.theme());
            console.log(`字体: ${size}px (主题: ${currentTheme}，不触发此 effect)`);
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
 */

interface User {
    id: number;
    name: string;
    email: string;
}

@Injectable({ providedIn: 'root' })
export class UserStore {
    private _users = signal<User[]>([
        { id: 1, name: '小明', email: 'ming@example.com' },
        { id: 2, name: '小红', email: 'hong@example.com' },
    ]);
    private _selectedId = signal<number | null>(null);

    readonly users = this._users.asReadonly();

    readonly selectedUser = computed(() => {
        const id = this._selectedId();
        return id ? this._users().find(u => u.id === id) ?? null : null;
    });

    readonly userCount = computed(() => this._users().length);

    select(id: number) { this._selectedId.set(id); }

    add(user: Omit<User, 'id'>) {
        const newId = Math.max(...this._users().map(u => u.id), 0) + 1;
        this._users.update(users => [...users, { ...user, id: newId }]);
    }

    remove(id: number) {
        this._users.update(users => users.filter(u => u.id !== id));
        if (this._selectedId() === id) this._selectedId.set(null);
    }
}


// ============================================================
//                    5. 最佳实践
// ============================================================

/**
 * 【Signal 最佳实践】
 *
 * ✅ 推荐做法：
 * 1. 简单状态管理优先使用 Signal
 * 2. 派生数据用 computed()（自动缓存）
 * 3. 副作用用 effect()（自动清理）
 * 4. 服务中暴露 readonly Signal
 * 5. 使用 untracked() 避免不必要的依赖追踪
 *
 * ❌ 避免做法：
 * 1. 在 effect 中修改其他 Signal → 可能导致循环
 * 2. 忽略 computed 的缓存能力 → 不要用 effect 模拟
 * 3. 所有场景都用 Signal → 异步流仍然用 RxJS
 *
 * 【Signal vs RxJS】
 * - 同步状态、UI 绑定 → Signal
 * - HTTP 请求、事件流、复杂异步操作 → RxJS
 * - 两者可以用 toSignal()/toObservable() 互转
 */
```
