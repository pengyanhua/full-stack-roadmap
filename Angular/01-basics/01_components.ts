/**
 * ============================================================
 *                    Angular 组件基础
 * ============================================================
 * 组件是 Angular 应用的基本构建块。
 * 每个组件由模板(HTML)、样式(CSS)和逻辑(TypeScript)组成。
 * ============================================================
 */

import { Component } from '@angular/core';

// ============================================================
//                    1. 组件基础结构
// ============================================================

/**
 * 【什么是 Angular 组件】
 *
 * Angular 组件 = TypeScript 类 + 装饰器 @Component
 * - selector: 组件在 HTML 中的标签名
 * - template / templateUrl: 组件的 HTML 模板
 * - styles / styleUrls: 组件的样式
 * - standalone: 是否为独立组件（Angular 14+ 推荐）
 *
 * 【组件生命周期】
 * constructor → ngOnChanges → ngOnInit → ngDoCheck
 *   → ngAfterContentInit → ngAfterContentChecked
 *   → ngAfterViewInit → ngAfterViewChecked → ngOnDestroy
 */

// --- 最简单的组件 ---
@Component({
    selector: 'app-hello',
    standalone: true,
    template: `<h1>Hello, Angular!</h1>`,
})
export class HelloComponent {}

// --- 带数据绑定的组件 ---
@Component({
    selector: 'app-greeting',
    standalone: true,
    template: `
        <h2>{{ title }}</h2>
        <p>欢迎来到 {{ framework }} 世界！</p>
    `,
})
export class GreetingComponent {
    title = 'Angular 入门';
    framework = 'Angular';
}


// ============================================================
//                    2. 模板语法 - 插值与属性绑定
// ============================================================

/**
 * 【插值表达式 {{ }}】
 * - 将组件属性渲染到模板中
 * - 支持 JavaScript 表达式
 * - 自动转义 HTML（防止 XSS）
 *
 * 【属性绑定 [property]】
 * - 单向绑定：组件 → 视图
 * - 绑定 DOM 属性而不是 HTML 属性
 * - 用方括号 [] 包裹属性名
 */

@Component({
    selector: 'app-binding-demo',
    standalone: true,
    template: `
        <!-- 插值表达式 -->
        <h2>{{ title }}</h2>
        <p>计算结果: {{ 1 + 2 + 3 }}</p>
        <p>字符串方法: {{ name.toUpperCase() }}</p>
        <p>三元表达式: {{ isActive ? '激活' : '未激活' }}</p>

        <!-- 属性绑定 -->
        <img [src]="imageUrl" [alt]="imageAlt">
        <button [disabled]="isDisabled">提交</button>
        <div [class.active]="isActive">动态 class</div>
        <div [style.color]="textColor">动态 style</div>

        <!-- Attribute 绑定 -->
        <td [attr.colspan]="colSpan">合并列</td>
    `,
})
export class BindingDemoComponent {
    title = '数据绑定演示';
    name = 'Angular';
    isActive = true;
    isDisabled = false;
    imageUrl = 'https://angular.io/assets/images/logos/angular/angular.svg';
    imageAlt = 'Angular Logo';
    textColor = 'blue';
    colSpan = 2;
}


// ============================================================
//                    3. 事件绑定
// ============================================================

/**
 * 【事件绑定 (event)】
 * - 用圆括号 () 包裹事件名
 * - $event 可以访问原始 DOM 事件对象
 * - 支持所有标准 DOM 事件
 *
 * 【常用事件】
 * (click)     - 点击事件
 * (input)     - 输入事件
 * (keyup)     - 按键抬起
 * (submit)    - 表单提交
 * (mouseover) - 鼠标悬停
 */

@Component({
    selector: 'app-event-demo',
    standalone: true,
    template: `
        <!-- 点击事件 -->
        <button (click)="onClick()">点击我</button>
        <button (click)="count = count + 1">计数: {{ count }}</button>

        <!-- 传递 $event -->
        <input (input)="onInput($event)" placeholder="输入内容">
        <p>输入的内容: {{ inputValue }}</p>

        <!-- 键盘事件过滤 -->
        <input (keyup.enter)="onEnter()" placeholder="按 Enter 提交">

        <!-- 鼠标事件 -->
        <div
            (mouseenter)="isHovered = true"
            (mouseleave)="isHovered = false"
            [class.hovered]="isHovered"
        >
            {{ isHovered ? '鼠标在这里' : '鼠标移到这里' }}
        </div>
    `,
})
export class EventDemoComponent {
    count = 0;
    inputValue = '';
    isHovered = false;

    onClick() {
        console.log('按钮被点击了！');
    }

    onInput(event: Event) {
        const target = event.target as HTMLInputElement;
        this.inputValue = target.value;
    }

    onEnter() {
        console.log('Enter 被按下！');
    }
}


// ============================================================
//                    4. 双向绑定
// ============================================================

/**
 * 【双向绑定 [(ngModel)]】
 * - 结合属性绑定 [] 和事件绑定 ()
 * - "香蕉在盒子里" 语法: [( )]
 * - 需要导入 FormsModule
 * - 常用于表单控件
 *
 * 【自定义双向绑定】
 * - 属性名: value
 * - 事件名: valueChange
 * - 使用 [(value)]="data" 即可
 */

import { FormsModule } from '@angular/forms';

@Component({
    selector: 'app-two-way-demo',
    standalone: true,
    imports: [FormsModule],
    template: `
        <!-- 基本双向绑定 -->
        <input [(ngModel)]="username" placeholder="输入用户名">
        <p>你好, {{ username }}!</p>

        <!-- 复选框 -->
        <label>
            <input type="checkbox" [(ngModel)]="agreed">
            我同意协议
        </label>
        <p>是否同意: {{ agreed }}</p>

        <!-- 下拉框 -->
        <select [(ngModel)]="selectedCity">
            <option value="">请选择城市</option>
            <option value="beijing">北京</option>
            <option value="shanghai">上海</option>
            <option value="shenzhen">深圳</option>
        </select>
        <p>选中的城市: {{ selectedCity }}</p>
    `,
})
export class TwoWayDemoComponent {
    username = '';
    agreed = false;
    selectedCity = '';
}


// ============================================================
//                    5. 条件渲染
// ============================================================

/**
 * 【@if (Angular 17+ 新语法)】
 * - 替代 *ngIf 指令
 * - 更直观、更接近 JavaScript 语法
 * - 支持 @else if 和 @else
 *
 * 【*ngIf (传统语法)】
 * - 为 true 时渲染元素
 * - 可配合 else 模板使用
 * - 需要导入 NgIf 或 CommonModule
 */

import { NgIf } from '@angular/common';

@Component({
    selector: 'app-conditional-demo',
    standalone: true,
    imports: [NgIf],
    template: `
        <!-- Angular 17+ 新语法 @if -->
        @if (score >= 90) {
            <p class="excellent">优秀! 🎉</p>
        } @else if (score >= 60) {
            <p class="pass">及格 ✅</p>
        } @else {
            <p class="fail">不及格 ❌</p>
        }

        <!-- 传统 *ngIf 指令 -->
        <div *ngIf="isLoggedIn; else loginTemplate">
            <p>欢迎回来, {{ username }}!</p>
            <button (click)="logout()">登出</button>
        </div>
        <ng-template #loginTemplate>
            <button (click)="login()">登录</button>
        </ng-template>

        <button (click)="changeScore()">改变分数</button>
        <p>当前分数: {{ score }}</p>
    `,
})
export class ConditionalDemoComponent {
    score = 85;
    isLoggedIn = false;
    username = '小明';

    login() {
        this.isLoggedIn = true;
    }

    logout() {
        this.isLoggedIn = false;
    }

    changeScore() {
        this.score = Math.floor(Math.random() * 100);
    }
}


// ============================================================
//                    6. 列表渲染
// ============================================================

/**
 * 【@for (Angular 17+ 新语法)】
 * - 替代 *ngFor 指令
 * - 必须指定 track 表达式（用于优化）
 * - 支持 @empty 块处理空列表
 *
 * 【*ngFor (传统语法)】
 * - 遍历数组生成 DOM 元素
 * - 提供 index, first, last, even, odd 等上下文变量
 * - trackBy 优化性能
 */

import { NgFor } from '@angular/common';

@Component({
    selector: 'app-list-demo',
    standalone: true,
    imports: [NgFor],
    template: `
        <h3>水果列表</h3>

        <!-- Angular 17+ 新语法 @for -->
        <ul>
            @for (fruit of fruits; track fruit.id) {
                <li>{{ fruit.name }} - ¥{{ fruit.price }}</li>
            } @empty {
                <li>暂无水果</li>
            }
        </ul>

        <!-- 传统 *ngFor 指令 -->
        <ul>
            <li *ngFor="let item of items; let i = index; let isFirst = first; let isLast = last">
                {{ i + 1 }}. {{ item }}
                <span *ngIf="isFirst"> (第一个)</span>
                <span *ngIf="isLast"> (最后一个)</span>
            </li>
        </ul>

        <!-- 嵌套循环 -->
        <div *ngFor="let category of categories">
            <h4>{{ category.name }}</h4>
            <ul>
                <li *ngFor="let product of category.products">
                    {{ product }}
                </li>
            </ul>
        </div>
    `,
})
export class ListDemoComponent {
    fruits = [
        { id: 1, name: '苹果', price: 8 },
        { id: 2, name: '香蕉', price: 5 },
        { id: 3, name: '橙子', price: 6 },
    ];

    items = ['HTML', 'CSS', 'TypeScript', 'Angular'];

    categories = [
        { name: '前端', products: ['Angular', 'React', 'Vue'] },
        { name: '后端', products: ['Node.js', 'Java', 'Go'] },
    ];
}


// ============================================================
//                    7. 管道 (Pipe)
// ============================================================

/**
 * 【内置管道】
 * - {{ value | uppercase }}    大写
 * - {{ value | lowercase }}    小写
 * - {{ value | date:'yyyy-MM-dd' }} 日期格式
 * - {{ value | currency:'CNY' }} 货币格式
 * - {{ value | number:'1.2-2' }} 数字格式
 * - {{ value | json }}          JSON 序列化
 * - {{ value | slice:0:10 }}    切片
 * - {{ value | async }}         异步管道
 *
 * 【管道链】
 * 可以连续使用: {{ value | slice:0:10 | uppercase }}
 */

import { CommonModule } from '@angular/common';

@Component({
    selector: 'app-pipe-demo',
    standalone: true,
    imports: [CommonModule],
    template: `
        <h3>管道演示</h3>

        <!-- 文本转换 -->
        <p>大写: {{ 'hello angular' | uppercase }}</p>
        <p>小写: {{ 'HELLO ANGULAR' | lowercase }}</p>
        <p>首字母大写: {{ 'hello' | titlecase }}</p>

        <!-- 日期格式化 -->
        <p>默认: {{ today | date }}</p>
        <p>完整: {{ today | date:'full' }}</p>
        <p>自定义: {{ today | date:'yyyy年MM月dd日 HH:mm' }}</p>

        <!-- 数字格式 -->
        <p>小数: {{ 3.14159 | number:'1.2-3' }}</p>
        <p>百分比: {{ 0.85 | percent }}</p>
        <p>货币: {{ 99.9 | currency:'CNY':'symbol' }}</p>

        <!-- JSON 调试 -->
        <pre>{{ user | json }}</pre>

        <!-- 管道链 -->
        <p>{{ 'hello world' | uppercase | slice:0:5 }}</p>
    `,
})
export class PipeDemoComponent {
    today = new Date();
    user = { name: '小明', age: 25, city: '北京' };
}


// ============================================================
//                    8. 模板引用变量
// ============================================================

/**
 * 【模板引用变量 #variableName】
 * - 获取 DOM 元素或组件的引用
 * - 可以在模板中直接使用
 * - 配合 @ViewChild 在 TypeScript 中使用
 *
 * 【用途】
 * - 获取输入框的值
 * - 调用子组件的方法
 * - 操作原生 DOM 元素
 */

@Component({
    selector: 'app-template-ref-demo',
    standalone: true,
    template: `
        <!-- 模板引用变量获取 input 值 -->
        <input #nameInput placeholder="输入名字">
        <button (click)="greet(nameInput.value)">
            打招呼
        </button>
        <p>{{ greeting }}</p>

        <!-- 获取元素尺寸 -->
        <div #myDiv style="width: 200px; height: 100px; background: #eee;">
            宽: {{ myDiv.offsetWidth }}px,
            高: {{ myDiv.offsetHeight }}px
        </div>
    `,
})
export class TemplateRefDemoComponent {
    greeting = '';

    greet(name: string) {
        this.greeting = name ? `你好, ${name}!` : '请输入名字';
    }
}


// ============================================================
//                    9. 最佳实践
// ============================================================

/**
 * 【Angular 组件最佳实践】
 *
 * ✅ 推荐做法：
 * 1. 使用 standalone 组件（Angular 14+）
 * 2. 优先使用 @if/@for 新语法（Angular 17+）
 * 3. 保持组件单一职责
 * 4. 使用 OnPush 变更检测策略提升性能
 * 5. 合理使用管道处理数据展示
 * 6. 使用模板引用变量代替直接 DOM 操作
 *
 * ❌ 避免做法：
 * 1. 在模板中使用复杂逻辑 → 移到组件类中
 * 2. 在模板中调用函数（会重复执行） → 使用管道或计算属性
 * 3. 组件逻辑过于臃肿 → 拆分为更小的组件
 * 4. 直接操作 DOM → 使用 Angular 提供的 API
 */
