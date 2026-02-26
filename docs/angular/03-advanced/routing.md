# routing.ts

::: info 文件信息
- 📄 原文件：`02_routing.ts`
- 🔤 语言：TypeScript (Angular)
:::

Angular 路由
Angular Router 提供了强大的导航和路由功能。支持路径匹配、参数传递、守卫、懒加载等。

## 完整代码

```typescript
/**
 * ============================================================
 *                    Angular 路由
 * ============================================================
 * Angular Router 提供了强大的导航和路由功能。
 * 支持路径匹配、参数传递、守卫、懒加载等。
 * ============================================================
 */

import { Component, inject, OnInit, Injectable } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
    Routes, RouterOutlet, RouterLink, RouterLinkActive,
    Router, ActivatedRoute, CanActivateFn, ResolveFn
} from '@angular/router';

// ============================================================
//                    1. 基本路由配置
// ============================================================

/**
 * 【路由基础概念】
 * - Routes: 路由配置数组
 * - RouterOutlet: 路由出口（组件渲染位置）
 * - RouterLink: 导航链接指令
 * - RouterLinkActive: 激活链接样式
 * - Router: 编程式导航服务
 * - ActivatedRoute: 获取路由参数
 */

@Component({
    selector: 'app-home-page',
    standalone: true,
    template: `<div><h2>🏠 首页</h2><p>欢迎来到 Angular 路由示例！</p></div>`,
})
export class HomePageComponent {}

@Component({
    selector: 'app-about-page',
    standalone: true,
    template: `<div><h2>ℹ️ 关于</h2><p>这是一个 Angular 路由学习项目。</p></div>`,
})
export class AboutPageComponent {}

@Component({
    selector: 'app-not-found',
    standalone: true,
    imports: [RouterLink],
    template: `<div><h2>404 - 页面未找到</h2><a routerLink="/">返回首页</a></div>`,
})
export class NotFoundPageComponent {}


// ============================================================
//                    2. 路由参数
// ============================================================

/**
 * 【路由参数类型】
 * 1. 路径参数: /user/:id → route.paramMap
 * 2. 查询参数: /search?q=angular → route.queryParamMap
 * 3. Fragment: /page#section → route.fragment
 */

@Component({
    selector: 'app-user-detail',
    standalone: true,
    imports: [CommonModule, RouterLink],
    template: `
        <div>
            <h2>用户详情</h2>
            <p>用户 ID: {{ userId }}</p>
            <nav>
                <a [routerLink]="['/user', 1]">用户 1</a> |
                <a [routerLink]="['/user', 2]">用户 2</a> |
                <a [routerLink]="['/user', 3]">用户 3</a>
            </nav>
        </div>
    `,
})
export class UserDetailComponent implements OnInit {
    userId = '';
    private route = inject(ActivatedRoute);

    ngOnInit() {
        // 订阅方式（响应参数变化，推荐）
        this.route.paramMap.subscribe(params => {
            this.userId = params.get('id') || '';
        });
    }
}


// ============================================================
//                    3. 编程式导航
// ============================================================

/**
 * 【Router 服务】
 * - navigate(): 导航到指定路径
 * - navigateByUrl(): 通过完整 URL 导航
 */

@Component({
    selector: 'app-nav-demo',
    standalone: true,
    imports: [FormsModule],
    template: `
        <div>
            <input [(ngModel)]="userId" placeholder="输入用户ID">
            <button (click)="goToUser()">查看用户</button>
            <button (click)="goHome()">返回首页</button>
        </div>
    `,
})
export class NavDemoComponent {
    userId = '';
    private router = inject(Router);

    goToUser() { this.router.navigate(['/user', this.userId]); }
    goHome() { this.router.navigateByUrl('/'); }
}


// ============================================================
//                    4. 路由守卫
// ============================================================

/**
 * 【函数式路由守卫 (Angular 15+ 推荐)】
 * - CanActivateFn: 是否允许访问
 * - CanDeactivateFn: 是否允许离开
 * - ResolveFn: 预加载数据
 */

@Injectable({ providedIn: 'root' })
export class AuthService {
    private _isLoggedIn = false;
    get isLoggedIn() { return this._isLoggedIn; }
    login() { this._isLoggedIn = true; }
    logout() { this._isLoggedIn = false; }
}

export const authGuard: CanActivateFn = (route, state) => {
    const authService = inject(AuthService);
    const router = inject(Router);
    if (authService.isLoggedIn) return true;
    return router.createUrlTree(['/']);
};


// ============================================================
//                    5. 嵌套路由
// ============================================================

/**
 * 【嵌套路由 (children)】
 * - 父路由组件中放置 <router-outlet>
 * - children 数组定义子路由
 */

@Component({
    selector: 'app-dashboard',
    standalone: true,
    imports: [RouterOutlet, RouterLink, RouterLinkActive],
    template: `
        <div class="dashboard">
            <nav class="sidebar">
                <a routerLink="overview" routerLinkActive="active">概览</a>
                <a routerLink="settings" routerLinkActive="active">设置</a>
            </nav>
            <main>
                <router-outlet></router-outlet>
            </main>
        </div>
    `,
})
export class DashboardComponent {}


// ============================================================
//                    6. 路由配置汇总
// ============================================================

export const routes: Routes = [
    { path: '', component: HomePageComponent, title: '首页' },
    { path: 'about', component: AboutPageComponent, title: '关于' },
    {
        path: 'user/:id',
        component: UserDetailComponent,
        canActivate: [authGuard],
        title: '用户详情',
    },
    {
        path: 'dashboard',
        component: DashboardComponent,
        canActivate: [authGuard],
        children: [
            { path: '', redirectTo: 'overview', pathMatch: 'full' },
            // { path: 'overview', component: DashboardOverviewComponent },
            // { path: 'settings', component: DashboardSettingsComponent },
        ],
    },
    // 懒加载示例
    // {
    //     path: 'admin',
    //     loadComponent: () => import('./admin.component').then(m => m.AdminComponent),
    // },
    { path: '**', component: NotFoundPageComponent, title: '404' },
];


// ============================================================
//                    7. 最佳实践
// ============================================================

/**
 * 【路由最佳实践】
 *
 * ✅ 推荐做法：
 * 1. 使用函数式守卫（CanActivateFn）替代类守卫
 * 2. 懒加载 loadComponent 减小首屏体积
 * 3. 使用 title 属性设置页面标题
 * 4. 通配符路由放在配置最后
 *
 * ❌ 避免做法：
 * 1. 路由嵌套过深 → 扁平化路由结构
 * 2. 在守卫中做复杂逻辑 → 委托给服务
 * 3. 路由路径硬编码 → 使用常量
 */
```
