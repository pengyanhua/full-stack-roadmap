# content_projection.ts

::: info 文件信息
- 📄 原文件：`03_content_projection.ts`
- 🔤 语言：TypeScript (Angular)
:::

Angular 内容投影
内容投影 (Content Projection) 是 Angular 实现组件复用的核心机制。类似于 Vue 的 Slots 和 React 的 children。

## 完整代码

```typescript
/**
 * ============================================================
 *                    Angular 内容投影
 * ============================================================
 * 内容投影 (Content Projection) 是 Angular 实现组件复用的核心机制。
 * 类似于 Vue 的 Slots 和 React 的 children。
 * ============================================================
 */

import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

// ============================================================
//                    1. 单插槽投影
// ============================================================

/**
 * 【单插槽 ng-content】
 * - 最简单的内容投影方式
 * - 父组件传入的内容会替换 <ng-content> 位置
 * - 类似 Vue 的默认插槽
 */

@Component({
    selector: 'app-card',
    standalone: true,
    template: `
        <div class="card">
            <div class="card-header">
                <h3>{{ title }}</h3>
            </div>
            <div class="card-body">
                <ng-content></ng-content>
            </div>
        </div>
    `,
})
export class CardComponent {
    @Input() title = '';
}


// ============================================================
//                    2. 多插槽投影 (select)
// ============================================================

/**
 * 【多插槽投影】
 * - 使用 select 属性指定投影内容
 * - select 支持 CSS 选择器
 *   - 标签名: select="header"
 *   - 属性: select="[card-footer]"
 *   - CSS 类: select=".card-actions"
 * - 没有 select 的 ng-content 接收剩余内容
 */

@Component({
    selector: 'app-fancy-card',
    standalone: true,
    template: `
        <div class="fancy-card">
            <div class="header">
                <ng-content select="[card-header]"></ng-content>
            </div>
            <div class="body">
                <ng-content></ng-content>
            </div>
            <div class="footer">
                <ng-content select="[card-footer]"></ng-content>
            </div>
        </div>
    `,
})
export class FancyCardComponent {}


// ============================================================
//                    3. 条件投影 (ngProjectAs)
// ============================================================

/**
 * 【ngProjectAs】
 * - 改变组件在投影时的"身份"
 * - 让包裹元素匹配 select 选择器
 *
 * 【ng-container】
 * - 逻辑分组容器，不会渲染到 DOM 中
 */

@Component({
    selector: 'app-dialog',
    standalone: true,
    template: `
        <div class="dialog">
            <div class="dialog-title">
                <ng-content select="[dialog-title]"></ng-content>
            </div>
            <div class="dialog-content">
                <ng-content select="[dialog-content]"></ng-content>
            </div>
            <div class="dialog-actions">
                <ng-content select="[dialog-actions]"></ng-content>
            </div>
        </div>
    `,
})
export class DialogComponent {}


// ============================================================
//                    4. Alert 组件
// ============================================================

@Component({
    selector: 'app-alert',
    standalone: true,
    imports: [CommonModule],
    template: `
        <div class="alert" [class]="'alert-' + type">
            <strong>{{ iconMap[type] }} {{ titleMap[type] }}</strong>
            <div class="alert-body">
                <ng-content></ng-content>
            </div>
        </div>
    `,
})
export class AlertComponent {
    @Input() type: 'info' | 'success' | 'warning' | 'error' = 'info';

    iconMap: Record<string, string> = {
        info: 'ℹ️', success: '✅', warning: '⚠️', error: '❌',
    };
    titleMap: Record<string, string> = {
        info: '提示', success: '成功', warning: '警告', error: '错误',
    };
}


// ============================================================
//                    5. 使用示例
// ============================================================

@Component({
    selector: 'app-projection-demo',
    standalone: true,
    imports: [CommonModule, CardComponent, FancyCardComponent, DialogComponent, AlertComponent],
    template: `
        <h2>内容投影演示</h2>

        <!-- 单插槽 -->
        <app-card title="简介">
            <p>这是投影到卡片内部的内容。</p>
        </app-card>

        <!-- 多插槽 -->
        <app-fancy-card>
            <div card-header><h4>标题</h4></div>
            <p>主体内容</p>
            <div card-footer><button>确定</button></div>
        </app-fancy-card>

        <!-- 对话框 -->
        <app-dialog>
            <span dialog-title>确认删除</span>
            <div dialog-content><p>此操作不可撤销。</p></div>
            <ng-container ngProjectAs="[dialog-actions]">
                <button>取消</button>
                <button style="color: red">删除</button>
            </ng-container>
        </app-dialog>

        <!-- Alert -->
        <app-alert type="success">操作成功！</app-alert>
        <app-alert type="error">网络连接失败</app-alert>
    `,
})
export class ProjectionDemoComponent {}


// ============================================================
//                    6. 最佳实践
// ============================================================

/**
 * 【内容投影最佳实践】
 *
 * ✅ 推荐做法：
 * 1. 使用 ng-content 创建可复用的布局组件
 * 2. 多插槽投影用属性选择器 [name]，语义更清晰
 * 3. 用 ng-container 避免多余的 DOM 元素
 *
 * ❌ 避免做法：
 * 1. 投影内容过于复杂 → 考虑拆分为独立组件
 * 2. 依赖投影内容的顺序 → 使用 select 明确指定
 */
```
