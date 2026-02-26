/**
 * ============================================================
 *                    Angular 生命周期
 * ============================================================
 * Angular 组件从创建到销毁有一系列生命周期钩子。
 * 合理使用生命周期钩子是编写高质量 Angular 组件的关键。
 * ============================================================
 */

import {
    Component, Input, OnInit, OnChanges, OnDestroy,
    DoCheck, AfterContentInit, AfterContentChecked,
    AfterViewInit, AfterViewChecked, SimpleChanges,
    ViewChild, ElementRef, ContentChild
} from '@angular/core';
import { CommonModule } from '@angular/common';

// ============================================================
//                    1. 生命周期钩子概览
// ============================================================

/**
 * 【生命周期执行顺序】
 *
 * 1. constructor          - 构造函数（依赖注入）
 * 2. ngOnChanges          - @Input 属性变化时（首次也会触发）
 * 3. ngOnInit             - 组件初始化完成（只执行一次）
 * 4. ngDoCheck            - 每次变更检测时
 * 5. ngAfterContentInit   - 内容投影初始化完成（只一次）
 * 6. ngAfterContentChecked - 内容投影检查完成
 * 7. ngAfterViewInit      - 视图初始化完成（只一次）
 * 8. ngAfterViewChecked   - 视图检查完成
 * 9. ngOnDestroy          - 组件销毁前（清理资源）
 *
 * 【最常用的钩子】
 * - ngOnInit: 初始化逻辑（API 调用、初始化数据）
 * - ngOnChanges: 响应 @Input 变化
 * - ngOnDestroy: 清理资源（取消订阅、清除定时器）
 * - ngAfterViewInit: 需要访问 DOM 时
 */


// ============================================================
//                    2. ngOnInit & ngOnDestroy
// ============================================================

/**
 * 【ngOnInit】
 * - 组件初始化时执行一次
 * - 此时 @Input 属性已经有值
 * - 适合：发起 HTTP 请求、初始化复杂数据
 *
 * 【ngOnDestroy】
 * - 组件销毁前执行一次
 * - 适合：取消订阅、清除定时器、释放资源
 */

@Component({
    selector: 'app-timer',
    standalone: true,
    template: `
        <div class="timer">
            <h4>计时器组件</h4>
            <p>已运行: {{ seconds }} 秒</p>
        </div>
    `,
})
export class TimerComponent implements OnInit, OnDestroy {
    seconds = 0;
    private intervalId: any = null;

    ngOnInit() {
        console.log('TimerComponent 初始化');
        // 启动定时器
        this.intervalId = setInterval(() => {
            this.seconds++;
        }, 1000);
    }

    ngOnDestroy() {
        console.log('TimerComponent 销毁，清理定时器');
        // 必须清理！否则会内存泄漏
        if (this.intervalId) {
            clearInterval(this.intervalId);
        }
    }
}


// ============================================================
//                    3. ngOnChanges
// ============================================================

/**
 * 【ngOnChanges】
 * - 当 @Input 属性值发生变化时触发
 * - 接收 SimpleChanges 对象
 * - 可以获取前一个值和当前值
 * - 首次绑定时 isFirstChange() 返回 true
 */

@Component({
    selector: 'app-profile',
    standalone: true,
    imports: [CommonModule],
    template: `
        <div class="profile">
            <h4>{{ name }} 的信息</h4>
            <p>年龄: {{ age }}</p>
            <div class="log">
                <h5>变更日志:</h5>
                @for (log of changeLogs; track log) {
                    <p>{{ log }}</p>
                }
            </div>
        </div>
    `,
    styles: [`
        .log { background: #f5f5f5; padding: 8px; font-size: 12px; max-height: 150px; overflow-y: auto; }
    `]
})
export class ProfileComponent implements OnChanges {
    @Input() name = '';
    @Input() age = 0;

    changeLogs: string[] = [];

    ngOnChanges(changes: SimpleChanges) {
        for (const propName in changes) {
            const change = changes[propName];
            const prev = JSON.stringify(change.previousValue);
            const curr = JSON.stringify(change.currentValue);

            if (change.isFirstChange()) {
                this.changeLogs.push(
                    `[首次] ${propName}: ${curr}`
                );
            } else {
                this.changeLogs.push(
                    `[变更] ${propName}: ${prev} → ${curr}`
                );
            }
        }
    }
}


// ============================================================
//                    4. ngAfterViewInit
// ============================================================

/**
 * 【ngAfterViewInit】
 * - 组件视图（含子组件）初始化完成后
 * - 可以安全地访问 @ViewChild 引用
 * - 注意：不能在此钩子中同步修改绑定数据
 *
 * 【@ViewChild】
 * - 获取模板中的 DOM 元素或子组件引用
 * - 在 ngAfterViewInit 之后才能使用
 */

@Component({
    selector: 'app-canvas-demo',
    standalone: true,
    template: `
        <div>
            <h4>Canvas 演示</h4>
            <canvas #myCanvas width="300" height="150"></canvas>
            <p>Canvas 宽度: {{ canvasWidth }}px</p>
        </div>
    `,
})
export class CanvasDemoComponent implements AfterViewInit {
    @ViewChild('myCanvas') canvasRef!: ElementRef<HTMLCanvasElement>;
    canvasWidth = 0;

    ngAfterViewInit() {
        // 此时可以安全访问 ViewChild
        const canvas = this.canvasRef.nativeElement;
        this.canvasWidth = canvas.width;

        const ctx = canvas.getContext('2d');
        if (ctx) {
            // 绘制简单图形
            ctx.fillStyle = '#4CAF50';
            ctx.fillRect(10, 10, 130, 130);

            ctx.fillStyle = 'white';
            ctx.font = '16px Arial';
            ctx.fillText('Angular Canvas', 20, 80);
        }
    }
}


// ============================================================
//                    5. 完整生命周期演示
// ============================================================

@Component({
    selector: 'app-lifecycle-child',
    standalone: true,
    template: `
        <div class="lifecycle-child">
            <h4>子组件: {{ label }}</h4>
            <ng-content></ng-content>
        </div>
    `,
    styles: [`
        .lifecycle-child { border: 1px dashed #aaa; padding: 8px; margin: 4px 0; }
    `]
})
export class LifecycleChildComponent implements
    OnChanges, OnInit, DoCheck,
    AfterContentInit, AfterContentChecked,
    AfterViewInit, AfterViewChecked,
    OnDestroy {

    @Input() label = '';
    @ContentChild('projected') projectedContent: any;

    constructor() {
        console.log(`[${this.label}] 1. constructor`);
    }

    ngOnChanges(changes: SimpleChanges) {
        console.log(`[${this.label}] 2. ngOnChanges`, changes);
    }

    ngOnInit() {
        console.log(`[${this.label}] 3. ngOnInit`);
    }

    ngDoCheck() {
        console.log(`[${this.label}] 4. ngDoCheck`);
    }

    ngAfterContentInit() {
        console.log(`[${this.label}] 5. ngAfterContentInit`);
    }

    ngAfterContentChecked() {
        console.log(`[${this.label}] 6. ngAfterContentChecked`);
    }

    ngAfterViewInit() {
        console.log(`[${this.label}] 7. ngAfterViewInit`);
    }

    ngAfterViewChecked() {
        console.log(`[${this.label}] 8. ngAfterViewChecked`);
    }

    ngOnDestroy() {
        console.log(`[${this.label}] 9. ngOnDestroy`);
    }
}

// --- 父组件 ---
@Component({
    selector: 'app-lifecycle-demo',
    standalone: true,
    imports: [CommonModule, TimerComponent, ProfileComponent, CanvasDemoComponent, LifecycleChildComponent],
    template: `
        <h2>生命周期演示</h2>

        <!-- 计时器（展示 ngOnInit / ngOnDestroy） -->
        <h3>1. OnInit & OnDestroy</h3>
        <button (click)="showTimer = !showTimer">
            {{ showTimer ? '销毁计时器' : '创建计时器' }}
        </button>
        @if (showTimer) {
            <app-timer />
        }

        <!-- Profile（展示 ngOnChanges） -->
        <h3>2. OnChanges</h3>
        <app-profile [name]="profileName" [age]="profileAge" />
        <button (click)="profileAge = profileAge + 1">年龄 +1</button>
        <button (click)="profileName = profileName === '小明' ? '小红' : '小明'">切换姓名</button>

        <!-- Canvas（展示 ngAfterViewInit） -->
        <h3>3. AfterViewInit</h3>
        <app-canvas-demo />

        <!-- 完整生命周期日志 -->
        <h3>4. 完整生命周期（打开控制台查看）</h3>
        <app-lifecycle-child [label]="'Demo'">
            <p #projected>投影内容</p>
        </app-lifecycle-child>
    `,
})
export class LifecycleDemoComponent {
    showTimer = true;
    profileName = '小明';
    profileAge = 25;
}


// ============================================================
//                    6. 最佳实践
// ============================================================

/**
 * 【生命周期最佳实践】
 *
 * ✅ 推荐做法：
 * 1. ngOnInit 中初始化数据和发起请求（而非 constructor）
 * 2. ngOnDestroy 中取消所有订阅和清理资源
 * 3. ngOnChanges 中响应 @Input 变化
 * 4. ngAfterViewInit 中访问 DOM 和 @ViewChild
 *
 * ❌ 避免做法：
 * 1. constructor 中做复杂逻辑 → 用 ngOnInit
 * 2. 忘记在 ngOnDestroy 中清理 → 导致内存泄漏
 * 3. ngAfterViewInit 中同步修改绑定数据 → 会触发 ExpressionChangedAfterChecked 错误
 * 4. 过度使用 ngDoCheck → 性能开销大
 */
