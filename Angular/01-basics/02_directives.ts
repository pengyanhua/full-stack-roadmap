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
 *
 * 【ngModel】
 * - 双向数据绑定（表单中使用）
 * - 需要导入 FormsModule
 */

@Component({
    selector: 'app-attribute-directives',
    standalone: true,
    imports: [CommonModule],
    template: `
        <h3>属性指令演示</h3>

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
        <button (click)="changeFontSize()">改变字号</button>
    `,
    styles: [`
        .active { border: 2px solid green; }
        .disabled { opacity: 0.5; }
        .highlight { background-color: yellow; }
        .base-class { padding: 10px; margin: 5px 0; }
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

    changeFontSize() {
        this.fontSize = this.fontSize >= 24 ? 14 : this.fontSize + 2;
    }
}


// ============================================================
//                    2. 内置结构指令
// ============================================================

/**
 * 【结构指令】
 * - 改变 DOM 的结构（添加/删除元素）
 * - 以 * 号开头（语法糖）
 *
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
        <h3>结构指令演示</h3>

        <!-- ngSwitch -->
        <div [ngSwitch]="currentTab">
            <div *ngSwitchCase="'home'">🏠 首页内容</div>
            <div *ngSwitchCase="'about'">ℹ️ 关于我们</div>
            <div *ngSwitchCase="'contact'">📞 联系方式</div>
            <div *ngSwitchDefault>404 页面不存在</div>
        </div>

        <button (click)="currentTab = 'home'">首页</button>
        <button (click)="currentTab = 'about'">关于</button>
        <button (click)="currentTab = 'contact'">联系</button>

        <!-- Angular 17+ @switch -->
        @switch (status) {
            @case ('loading') {
                <p>加载中...</p>
            }
            @case ('success') {
                <p>加载成功！</p>
            }
            @case ('error') {
                <p>加载失败！</p>
            }
            @default {
                <p>未知状态</p>
            }
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
 *
 * 【使用场景】
 * - 高亮效果
 * - 权限控制（显示/隐藏）
 * - 自动聚焦
 * - 防抖点击
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

    // 鼠标进入时高亮
    @HostListener('mouseenter')
    onMouseEnter() {
        this.highlight(this.appHighlight || 'yellow');
    }

    // 鼠标离开时恢复
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

// --- 使用自定义指令的组件 ---
@Component({
    selector: 'app-custom-directive-demo',
    standalone: true,
    imports: [HighlightDirective, AutoFocusDirective],
    template: `
        <h3>自定义指令演示</h3>

        <!-- 高亮指令 -->
        <p appHighlight>默认黄色高亮（鼠标悬停）</p>
        <p [appHighlight]="'lightblue'">蓝色高亮</p>
        <p [appHighlight]="'lightgreen'" defaultColor="white">绿色高亮</p>

        <!-- 自动聚焦指令 -->
        <input appAutoFocus placeholder="自动获得焦点">
    `,
})
export class CustomDirectiveDemoComponent {}


// ============================================================
//                    4. 自定义结构指令
// ============================================================

/**
 * 【自定义结构指令】
 * - 使用 TemplateRef 和 ViewContainerRef
 * - TemplateRef: 获取宿主模板
 * - ViewContainerRef: 操作 DOM 视图容器
 *
 * 【应用场景】
 * - 权限控制: *appHasRole="'admin'"
 * - 延迟加载: *appDefer
 * - 重复渲染: *appRepeat="3"
 */

// --- 权限控制指令 ---
@Directive({
    selector: '[appHasRole]',
    standalone: true,
})
export class HasRoleDirective {
    private currentRole = 'admin'; // 模拟当前用户角色

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

// --- 使用自定义结构指令 ---
@Component({
    selector: 'app-structural-directive-demo',
    standalone: true,
    imports: [HasRoleDirective, RepeatDirective],
    template: `
        <h3>自定义结构指令演示</h3>

        <!-- 权限控制 -->
        <div *appHasRole="'admin'">
            🔐 管理员才能看到的内容
        </div>
        <div *appHasRole="'user'">
            👤 普通用户才能看到的内容（当前角色是 admin，所以不显示）
        </div>

        <!-- 重复渲染 -->
        <p *appRepeat="3; let i">
            第 {{ i + 1 }} 次渲染
        </p>
    `,
})
export class StructuralDirectiveDemoComponent {}


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
