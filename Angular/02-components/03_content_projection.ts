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
                <!-- 父组件内容投影到此处 -->
                <ng-content></ng-content>
            </div>
        </div>
    `,
    styles: [`
        .card { border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; margin: 12px 0; }
        .card-header { background: #f5f5f5; padding: 12px 16px; border-bottom: 1px solid #e0e0e0; }
        .card-body { padding: 16px; }
    `]
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
            <!-- 头部插槽 -->
            <div class="header">
                <ng-content select="[card-header]"></ng-content>
            </div>

            <!-- 默认插槽（接收未匹配的内容） -->
            <div class="body">
                <ng-content></ng-content>
            </div>

            <!-- 底部插槽 -->
            <div class="footer">
                <ng-content select="[card-footer]"></ng-content>
            </div>
        </div>
    `,
    styles: [`
        .fancy-card { border: 2px solid #1976d2; border-radius: 12px; overflow: hidden; margin: 12px 0; }
        .header { background: #1976d2; color: white; padding: 12px 16px; }
        .body { padding: 16px; }
        .footer { background: #f5f5f5; padding: 12px 16px; border-top: 1px solid #e0e0e0; }
    `]
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
 * - 常配合 ngProjectAs 使用
 */

@Component({
    selector: 'app-dialog',
    standalone: true,
    template: `
        <div class="dialog-backdrop">
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
        </div>
    `,
    styles: [`
        .dialog { border: 1px solid #ccc; border-radius: 8px; max-width: 400px; margin: 16px auto; }
        .dialog-title { padding: 16px; font-size: 18px; font-weight: bold; border-bottom: 1px solid #eee; }
        .dialog-content { padding: 16px; }
        .dialog-actions { padding: 12px 16px; text-align: right; border-top: 1px solid #eee; }
    `]
})
export class DialogComponent {}


// ============================================================
//                    4. 带默认内容的投影
// ============================================================

/**
 * 【默认内容】
 * ng-content 没有直接的默认内容机制，
 * 但可以通过 @ContentChild 检查是否有内容投影，
 * 然后决定是否显示默认内容。
 */

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
    styles: [`
        .alert { padding: 12px 16px; border-radius: 6px; margin: 8px 0; }
        .alert-info { background: #e3f2fd; color: #1565c0; }
        .alert-success { background: #e8f5e9; color: #2e7d32; }
        .alert-warning { background: #fff3e0; color: #e65100; }
        .alert-error { background: #ffebee; color: #c62828; }
    `]
})
export class AlertComponent {
    @Input() type: 'info' | 'success' | 'warning' | 'error' = 'info';

    iconMap: Record<string, string> = {
        info: 'ℹ️',
        success: '✅',
        warning: '⚠️',
        error: '❌',
    };

    titleMap: Record<string, string> = {
        info: '提示',
        success: '成功',
        warning: '警告',
        error: '错误',
    };
}


// ============================================================
//                    5. 父组件使用示例
// ============================================================

@Component({
    selector: 'app-projection-demo',
    standalone: true,
    imports: [CommonModule, CardComponent, FancyCardComponent, DialogComponent, AlertComponent],
    template: `
        <h2>内容投影演示</h2>

        <!-- 1. 单插槽投影 -->
        <h3>1. 单插槽投影</h3>
        <app-card title="简介">
            <p>这是投影到卡片内部的内容。</p>
            <p>可以包含任意 HTML 和 Angular 组件。</p>
        </app-card>

        <!-- 2. 多插槽投影 -->
        <h3>2. 多插槽投影</h3>
        <app-fancy-card>
            <div card-header>
                <h4>多插槽卡片标题</h4>
            </div>

            <!-- 默认插槽 -->
            <p>这里是主体内容，没有特别的属性标记。</p>

            <div card-footer>
                <button>取消</button>
                <button>确定</button>
            </div>
        </app-fancy-card>

        <!-- 3. 对话框 -->
        <h3>3. 对话框投影</h3>
        <app-dialog>
            <span dialog-title>确认删除</span>

            <div dialog-content>
                <p>确定要删除这条记录吗？此操作不可撤销。</p>
            </div>

            <!-- 使用 ng-container + ngProjectAs -->
            <ng-container ngProjectAs="[dialog-actions]">
                <button>取消</button>
                <button style="color: red">删除</button>
            </ng-container>
        </app-dialog>

        <!-- 4. Alert 组件 -->
        <h3>4. Alert 组件</h3>
        <app-alert type="info">这是一条提示信息</app-alert>
        <app-alert type="success">操作成功！</app-alert>
        <app-alert type="warning">请注意数据安全</app-alert>
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
 * 4. 适当使用 @ContentChild/@ContentChildren 访问投影内容
 *
 * ❌ 避免做法：
 * 1. 投影内容过于复杂 → 考虑拆分为独立组件
 * 2. 依赖投影内容的顺序 → 使用 select 明确指定
 * 3. 在投影内容中使用过多逻辑 → 保持投影内容简洁
 */
