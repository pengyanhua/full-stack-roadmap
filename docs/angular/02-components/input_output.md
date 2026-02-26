# input_output.ts

::: info 文件信息
- 📄 原文件：`01_input_output.ts`
- 🔤 语言：TypeScript (Angular)
:::

Angular 组件通信
Angular 组件间的数据传递通过 @Input 和 @Output 实现。父传子用 @Input，子传父用 @Output + EventEmitter。

## 完整代码

```typescript
/**
 * ============================================================
 *                    Angular 组件通信
 * ============================================================
 * Angular 组件间的数据传递通过 @Input 和 @Output 实现。
 * 父传子用 @Input，子传父用 @Output + EventEmitter。
 * ============================================================
 */

import { Component, Input, Output, EventEmitter, input, output, model } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

// ============================================================
//                    1. @Input - 父传子
// ============================================================

/**
 * 【@Input 装饰器】
 * - 让组件接收来自父组件的数据
 * - 支持默认值
 * - 支持 required 标记（Angular 16+）
 *
 * 【Signal Input (Angular 17+)】
 * - input() 函数替代 @Input 装饰器
 * - input.required<Type>() 必填输入
 * - 返回 Signal，更好的响应式支持
 */

// --- 子组件: 用户卡片 ---
@Component({
    selector: 'app-user-card',
    standalone: true,
    imports: [CommonModule],
    template: `
        <div class="card" [class.vip]="isVip">
            <h3>{{ name }}</h3>
            <p>年龄: {{ age }}</p>
            <p>角色: {{ role }}</p>
            @if (isVip) {
                <span class="badge">VIP</span>
            }
        </div>
    `,
})
export class UserCardComponent {
    @Input() name = '未知用户';
    @Input() age = 0;
    @Input({ required: true }) role!: string;
    @Input() isVip = false;
}

// --- Signal Input (Angular 17+) ---
@Component({
    selector: 'app-product-card',
    standalone: true,
    template: `
        <div class="product">
            <h4>{{ title() }}</h4>
            <p>价格: ¥{{ price() }}</p>
            @if (discount()) {
                <p class="discount">折扣: {{ discount() }}%</p>
            }
        </div>
    `,
})
export class ProductCardComponent {
    title = input.required<string>();
    price = input(0);
    discount = input<number | null>(null);
}


// ============================================================
//                    2. @Output - 子传父
// ============================================================

/**
 * 【@Output 装饰器 + EventEmitter】
 * - 子组件通过事件向父组件发送数据
 * - EventEmitter<T> 定义事件类型
 * - emit() 方法触发事件
 *
 * 【output() 函数 (Angular 17+)】
 * - 替代 @Output 装饰器
 */

// --- 子组件: 计数器 ---
@Component({
    selector: 'app-counter',
    standalone: true,
    template: `
        <div class="counter">
            <button (click)="decrement()">-</button>
            <span>{{ count }}</span>
            <button (click)="increment()">+</button>
            <button (click)="reset()">重置</button>
        </div>
    `,
})
export class CounterComponent {
    @Input() count = 0;
    @Output() countChange = new EventEmitter<number>();
    @Output() onReset = new EventEmitter<void>();

    increment() {
        this.count++;
        this.countChange.emit(this.count);
    }

    decrement() {
        this.count--;
        this.countChange.emit(this.count);
    }

    reset() {
        this.count = 0;
        this.countChange.emit(this.count);
        this.onReset.emit();
    }
}

// --- Angular 17+ output() 函数 ---
@Component({
    selector: 'app-search-box',
    standalone: true,
    imports: [FormsModule],
    template: `
        <div class="search-box">
            <input
                [(ngModel)]="keyword"
                (keyup.enter)="doSearch()"
                placeholder="搜索..."
            >
            <button (click)="doSearch()">搜索</button>
        </div>
    `,
})
export class SearchBoxComponent {
    keyword = '';
    search = output<string>();
    clear = output<void>();

    doSearch() {
        if (this.keyword.trim()) {
            this.search.emit(this.keyword);
        }
    }
}


// ============================================================
//                    3. 双向绑定 - model()
// ============================================================

/**
 * 【组件双向绑定】
 * - 传统方式: @Input() + @Output() xxxChange
 * - 父组件使用 [(xxx)]="value" 语法
 *
 * 【model() 函数 (Angular 17+)】
 * - 替代 @Input + @Output 组合
 * - model<Type>() 创建可写 Signal
 */

// --- 评分组件（传统双向绑定） ---
@Component({
    selector: 'app-rating',
    standalone: true,
    imports: [CommonModule],
    template: `
        <div class="rating">
            @for (star of stars; track star) {
                <span (click)="setRating(star)" [class.filled]="star <= value">
                    {{ star <= value ? '★' : '☆' }}
                </span>
            }
            <span class="label">{{ value }} 分</span>
        </div>
    `,
})
export class RatingComponent {
    @Input() value = 0;
    @Output() valueChange = new EventEmitter<number>();
    stars = [1, 2, 3, 4, 5];

    setRating(star: number) {
        this.value = star;
        this.valueChange.emit(this.value);
    }
}

// --- Angular 17+ model() ---
@Component({
    selector: 'app-toggle',
    standalone: true,
    template: `
        <button
            (click)="checked.set(!checked())"
            [class.on]="checked()"
        >
            {{ checked() ? label() + ': 开' : label() + ': 关' }}
        </button>
    `,
})
export class ToggleComponent {
    checked = model(false);
    label = input('开关');
}


// ============================================================
//                    4. 父组件示例
// ============================================================

@Component({
    selector: 'app-communication-demo',
    standalone: true,
    imports: [
        CommonModule, UserCardComponent, ProductCardComponent,
        CounterComponent, SearchBoxComponent, RatingComponent, ToggleComponent,
    ],
    template: `
        <h2>组件通信演示</h2>

        <h3>1. 父传子 (@Input)</h3>
        <app-user-card name="小明" [age]="25" role="admin" [isVip]="true" />
        <app-user-card name="小红" [age]="22" role="user" />

        <h3>2. 子传父 (@Output)</h3>
        <app-counter
            [count]="myCount"
            (countChange)="onCountChange($event)"
        />
        <p>父组件中的计数: {{ myCount }}</p>

        <h3>3. 双向绑定</h3>
        <app-rating [(value)]="rating" />
        <p>评分: {{ rating }}</p>

        <app-toggle [(checked)]="darkMode" label="深色模式" />
        <p>深色模式: {{ darkMode }}</p>
    `,
})
export class CommunicationDemoComponent {
    myCount = 10;
    rating = 3;
    darkMode = false;

    onCountChange(value: number) {
        this.myCount = value;
    }
}


// ============================================================
//                    5. 最佳实践
// ============================================================

/**
 * 【组件通信最佳实践】
 *
 * ✅ 推荐做法：
 * 1. 父子通信用 @Input/@Output，简单直接
 * 2. Angular 17+ 优先使用 input()/output()/model()
 * 3. 复杂场景用 Service + RxJS 进行跨组件通信
 * 4. @Input 使用不可变数据，避免子组件直接修改
 *
 * ❌ 避免做法：
 * 1. 子组件直接修改 @Input 引用类型的数据
 * 2. 过度嵌套的 @Input/@Output 传递（超过 3 层考虑用 Service）
 */
```
