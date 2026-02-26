# directives.ts

::: info 文件信息
- 📄 原文件：`02_directives.ts`
- 🔤 语言：TypeScript (Angular)
:::

Angular 指令 (Directives)
指令是 Angular 中用来扩展 HTML 元素行为的机制。分为属性指令、结构指令和自定义指令。

## 完整代码

```typescript
/**
 * ============================================================
 *                    Angular 指令 (Directives)
 * ============================================================
 * 指令是 Angular 中用来扩展 HTML 元素行为的机制。
 * 分为属性指令、结构指令和自定义指令。
 * ============================================================
 */

import { Component, Directive, ElementRef, HostListener, Input, TemplateRef, ViewContainerRef } from '@angular/core';
import { CommonModule } from '@angular/common';

// ============================================================
//                    1. 内置属性指令
// ============================================================

/**
 * 【ngClass】
 * - 动态设置 CSS 类
 * - 接受字符串、数组或对象
 *
 * 【ngStyle】
 * - 动态设置内联样式
 * - 接受对象 { styleName: value }
 */

@Component({
    selector: 'app-attribute-directives',
    standalone: true,
    imports: [CommonModule],
    template: `
        <!-- ngClass - 对象语法 -->
        <div [ngClass]="{
            'active': isActive,
            'disabled': isDisabled,
            'highlight': isHighlighted
        }">
            动态 Class 绑定
        </div>

        <!-- ngClass - 数组语法 -->
        <div [ngClass]="['base-class', currentTheme]">
            数组方式绑定 Class
        </div>

        <!-- ngStyle - 对象语法 -->
        <div [ngStyle]="{
            'color': textColor,
            'font-size': fontSize + 'px',
            'background-color': isActive ? '#e8f5e9' : '#ffebee'
        }">
            动态样式绑定
        </div>

        <button (click)="toggleActive()">切换状态</button>
    `,
    styles: [`
        .active { border: 2px solid green; }
        .disabled { opacity: 0.5; }
        .highlight { background-color: yellow; }
    `]
})
export class AttributeDirectivesComponent {
    isActive = true;
    isDisabled = false;
    isHighlighted = false;
    textColor = '#333';
    fontSize = 16;
    currentTheme = 'light-theme';

    toggleActive() {
        this.isActive = !this.isActive;
    }
}


// ============================================================
//                    2. 内置结构指令
// ============================================================

/**
 * 【ngSwitch】
 * - 类似 JavaScript 的 switch 语句
 * - [ngSwitch] + *ngSwitchCase + *ngSwitchDefault
 *
 * 【Angular 17+ @switch 新语法】
 * - 更直观的控制流语法
 */

@Component({
    selector: 'app-structural-directives',
    standalone: true,
    imports: [CommonModule],
    template: `
        <!-- ngSwitch -->
        <div [ngSwitch]="currentTab">
            <div *ngSwitchCase="'home'">🏠 首页内容</div>
            <div *ngSwitchCase="'about'">ℹ️ 关于我们</div>
            <div *ngSwitchCase="'contact'">📞 联系方式</div>
            <div *ngSwitchDefault>404 页面不存在</div>
        </div>

        <!-- Angular 17+ @switch -->
        @switch (status) {
            @case ('loading') { <p>加载中...</p> }
            @case ('success') { <p>加载成功！</p> }
            @case ('error') { <p>加载失败！</p> }
            @default { <p>未知状态</p> }
        }
    `,
})
export class StructuralDirectivesComponent {
    currentTab = 'home';
    status = 'success';
}


// ============================================================
//                    3. 自定义属性指令
// ============================================================

/**
 * 【自定义指令】
 * - @Directive 装饰器
 * - ElementRef 访问宿主 DOM 元素
 * - @HostListener 监听宿主事件
 * - @Input 接收绑定参数
 */

// --- 高亮指令 ---
@Directive({
    selector: '[appHighlight]',
    standalone: true,
})
export class HighlightDirective {
    @Input() appHighlight = 'yellow';
    @Input() defaultColor = '';

    constructor(private el: ElementRef) {}

    @HostListener('mouseenter')
    onMouseEnter() {
        this.highlight(this.appHighlight || 'yellow');
    }

    @HostListener('mouseleave')
    onMouseLeave() {
        this.highlight(this.defaultColor);
    }

    private highlight(color: string) {
        this.el.nativeElement.style.backgroundColor = color;
    }
}

// --- 自动聚焦指令 ---
@Directive({
    selector: '[appAutoFocus]',
    standalone: true,
})
export class AutoFocusDirective {
    constructor(private el: ElementRef) {}

    ngAfterViewInit() {
        this.el.nativeElement.focus();
    }
}


// ============================================================
//                    4. 自定义结构指令
// ============================================================

/**
 * 【自定义结构指令】
 * - 使用 TemplateRef 和 ViewContainerRef
 * - TemplateRef: 获取宿主模板
 * - ViewContainerRef: 操作 DOM 视图容器
 */

// --- 权限控制指令 ---
@Directive({
    selector: '[appHasRole]',
    standalone: true,
})
export class HasRoleDirective {
    private currentRole = 'admin';

    constructor(
        private templateRef: TemplateRef<any>,
        private viewContainer: ViewContainerRef,
    ) {}

    @Input() set appHasRole(role: string) {
        if (this.currentRole === role) {
            this.viewContainer.createEmbeddedView(this.templateRef);
        } else {
            this.viewContainer.clear();
        }
    }
}

// --- 重复渲染指令 ---
@Directive({
    selector: '[appRepeat]',
    standalone: true,
})
export class RepeatDirective {
    constructor(
        private templateRef: TemplateRef<any>,
        private viewContainer: ViewContainerRef,
    ) {}

    @Input() set appRepeat(count: number) {
        this.viewContainer.clear();
        for (let i = 0; i < count; i++) {
            this.viewContainer.createEmbeddedView(this.templateRef, {
                $implicit: i,
                index: i,
            });
        }
    }
}


// ============================================================
//                    5. 最佳实践
// ============================================================

/**
 * 【指令最佳实践】
 *
 * ✅ 推荐做法：
 * 1. 属性指令用于改变元素外观/行为
 * 2. 结构指令用于改变 DOM 结构
 * 3. 指令应保持单一职责
 * 4. 使用 @HostListener 代替手动添加事件监听
 * 5. Angular 17+ 优先使用 @if/@for/@switch 新语法
 *
 * ❌ 避免做法：
 * 1. 在指令中直接操作过多 DOM → 使用 Renderer2
 * 2. 指令逻辑过于复杂 → 考虑拆分为组件
 * 3. 忽略清理工作 → 在 ngOnDestroy 中清理资源
 */
```
