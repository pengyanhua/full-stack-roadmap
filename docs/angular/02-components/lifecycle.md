# lifecycle.ts

::: info 文件信息
- 📄 原文件：`02_lifecycle.ts`
- 🔤 语言：TypeScript (Angular)
:::

Angular 生命周期
Angular 组件从创建到销毁有一系列生命周期钩子。合理使用生命周期钩子是编写高质量 Angular 组件的关键。

## 完整代码

```typescript
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
                this.changeLogs.push(`[首次] ${propName}: ${curr}`);
            } else {
                this.changeLogs.push(`[变更] ${propName}: ${prev} → ${curr}`);
            }
        }
    }
}


// ============================================================
//                    4. ngAfterViewInit
// ============================================================

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
        const canvas = this.canvasRef.nativeElement;
        this.canvasWidth = canvas.width;

        const ctx = canvas.getContext('2d');
        if (ctx) {
            ctx.fillStyle = '#4CAF50';
            ctx.fillRect(10, 10, 130, 130);
            ctx.fillStyle = 'white';
            ctx.font = '16px Arial';
            ctx.fillText('Angular Canvas', 20, 80);
        }
    }
}


// ============================================================
//                    5. 最佳实践
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
 * 3. ngAfterViewInit 中同步修改绑定数据 → 触发 ExpressionChangedAfterChecked 错误
 * 4. 过度使用 ngDoCheck → 性能开销大
 */
```
