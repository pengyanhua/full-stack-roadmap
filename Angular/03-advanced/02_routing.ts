/**
 * ============================================================
 *                    Angular 路由
 * ============================================================
 * Angular Router 提供了强大的导航和路由功能。
 * 支持路径匹配、参数传递、守卫、懒加载等。
 * ============================================================
 */

import { Component, inject, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import {
    Routes, RouterModule, RouterOutlet, RouterLink, RouterLinkActive,
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
 *
 * 【路由配置项】
 * - path: 路径（不以 / 开头）
 * - component: 对应组件
 * - redirectTo: 重定向目标
 * - pathMatch: 匹配策略 ('full' | 'prefix')
 * - children: 子路由
 * - canActivate: 路由守卫
 * - loadComponent: 懒加载组件
 * - title: 页面标题
 */

// --- 页面组件 ---
@Component({
    selector: 'app-home-page',
    standalone: true,
    template: `
        <div class="page">
            <h2>🏠 首页</h2>
            <p>欢迎来到 Angular 路由示例！</p>
        </div>
    `,
})
export class HomePageComponent {}

@Component({
    selector: 'app-about-page',
    standalone: true,
    template: `
        <div class="page">
            <h2>ℹ️ 关于</h2>
            <p>这是一个 Angular 路由学习项目。</p>
        </div>
    `,
})
export class AboutPageComponent {}

@Component({
    selector: 'app-not-found-page',
    standalone: true,
    template: `
        <div class="page">
            <h2>404 - 页面未找到</h2>
            <p>您访问的页面不存在。</p>
            <a routerLink="/">返回首页</a>
        </div>
    `,
    imports: [RouterLink],
})
export class NotFoundPageComponent {}


// ============================================================
//                    2. 路由参数
// ============================================================

/**
 * 【路由参数类型】
 *
 * 1. 路径参数: /user/:id
 *    route.params 或 route.paramMap
 *
 * 2. 查询参数: /search?q=angular
 *    route.queryParams 或 route.queryParamMap
 *
 * 3. Fragment: /page#section
 *    route.fragment
 *
 * 【获取参数方式】
 * - 快照: route.snapshot.paramMap.get('id') （一次性读取）
 * - 订阅: route.paramMap.subscribe(...) （响应参数变化）
 *
 * 【withComponentInputBinding (Angular 16+)】
 * - 路由参数自动绑定到 @Input
 * - 无需手动读取 ActivatedRoute
 */

@Component({
    selector: 'app-user-detail',
    standalone: true,
    imports: [CommonModule, RouterLink],
    template: `
        <div class="page">
            <h2>用户详情</h2>
            <p>用户 ID: {{ userId }}</p>

            <!-- 导航到其他用户 -->
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
        // 方式一: 快照（不响应变化）
        this.userId = this.route.snapshot.paramMap.get('id') || '';

        // 方式二: 订阅（响应参数变化，推荐）
        this.route.paramMap.subscribe(params => {
            this.userId = params.get('id') || '';
            console.log('用户 ID 变化:', this.userId);
        });
    }
}

// --- 搜索结果页（查询参数） ---
@Component({
    selector: 'app-search-results',
    standalone: true,
    imports: [CommonModule],
    template: `
        <div class="page">
            <h2>搜索结果</h2>
            <p>关键词: {{ keyword }}</p>
            <p>页码: {{ page }}</p>
        </div>
    `,
})
export class SearchResultsComponent implements OnInit {
    keyword = '';
    page = 1;
    private route = inject(ActivatedRoute);

    ngOnInit() {
        this.route.queryParamMap.subscribe(params => {
            this.keyword = params.get('q') || '';
            this.page = Number(params.get('page')) || 1;
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
 *
 * 【导航选项】
 * - queryParams: 查询参数
 * - fragment: 片段标识
 * - relativeTo: 相对导航
 * - replaceUrl: 替换浏览器历史记录
 */

@Component({
    selector: 'app-nav-demo',
    standalone: true,
    imports: [CommonModule, FormsModule],
    template: `
        <div class="nav-demo">
            <h3>编程式导航</h3>

            <!-- 导航到用户页面 -->
            <input [(ngModel)]="userId" placeholder="输入用户ID">
            <button (click)="goToUser()">查看用户</button>

            <!-- 带查询参数的导航 -->
            <input [(ngModel)]="searchQuery" placeholder="搜索关键词">
            <button (click)="search()">搜索</button>

            <!-- 返回 -->
            <button (click)="goHome()">返回首页</button>
        </div>
    `,
})
export class NavDemoComponent {
    userId = '';
    searchQuery = '';
    private router = inject(Router);

    goToUser() {
        // 数组方式
        this.router.navigate(['/user', this.userId]);
    }

    search() {
        // 带查询参数
        this.router.navigate(['/search'], {
            queryParams: { q: this.searchQuery, page: 1 },
        });
    }

    goHome() {
        this.router.navigateByUrl('/');
    }
}

import { FormsModule } from '@angular/forms';


// ============================================================
//                    4. 路由守卫
// ============================================================

/**
 * 【函数式路由守卫 (Angular 15+ 推荐)】
 * - CanActivateFn: 是否允许访问
 * - CanDeactivateFn: 是否允许离开
 * - ResolveFn: 预加载数据
 * - CanMatchFn: 是否匹配路由
 *
 * 【守卫返回值】
 * - true: 允许导航
 * - false: 阻止导航
 * - UrlTree: 重定向到其他路由
 * - Observable/Promise: 异步判断
 */

// --- 认证服务（模拟） ---
import { Injectable } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class AuthService {
    private _isLoggedIn = false;

    get isLoggedIn() {
        return this._isLoggedIn;
    }

    login() {
        this._isLoggedIn = true;
    }

    logout() {
        this._isLoggedIn = false;
    }
}

// --- 函数式守卫 ---
export const authGuard: CanActivateFn = (route, state) => {
    const authService = inject(AuthService);
    const router = inject(Router);

    if (authService.isLoggedIn) {
        return true;
    }

    // 未登录则重定向到登录页
    console.log('未登录，重定向到首页');
    return router.createUrlTree(['/']);
};

// --- 数据预加载 ---
export const userResolver: ResolveFn<any> = (route, state) => {
    const userId = route.paramMap.get('id');
    // 模拟 API 请求
    return { id: userId, name: `用户${userId}`, email: `user${userId}@example.com` };
};


// ============================================================
//                    5. 嵌套路由
// ============================================================

/**
 * 【嵌套路由 (children)】
 * - 父路由组件中放置 <router-outlet>
 * - children 数组定义子路由
 * - 子路由路径是相对于父路由的
 *
 * 【适用场景】
 * - 管理后台侧边栏 + 主内容区
 * - Tab 页面切换
 * - 多步骤表单
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
                <a routerLink="profile" routerLinkActive="active">个人资料</a>
            </nav>
            <main class="content">
                <!-- 子路由渲染在这里 -->
                <router-outlet></router-outlet>
            </main>
        </div>
    `,
    styles: [`
        .dashboard { display: flex; gap: 16px; }
        .sidebar { display: flex; flex-direction: column; gap: 8px; min-width: 120px; }
        .sidebar a { padding: 8px 12px; text-decoration: none; color: #333; border-radius: 4px; }
        .sidebar a.active { background: #1976d2; color: white; }
        .content { flex: 1; padding: 16px; border: 1px solid #e0e0e0; border-radius: 8px; }
    `]
})
export class DashboardComponent {}

@Component({
    standalone: true,
    template: '<h3>📊 概览页面</h3><p>这里是仪表盘概览。</p>',
})
export class DashboardOverviewComponent {}

@Component({
    standalone: true,
    template: '<h3>⚙️ 设置页面</h3><p>这里是系统设置。</p>',
})
export class DashboardSettingsComponent {}

@Component({
    standalone: true,
    template: '<h3>👤 个人资料</h3><p>这里是个人资料设置。</p>',
})
export class DashboardProfileComponent {}


// ============================================================
//                    6. 路由配置汇总
// ============================================================

export const routes: Routes = [
    // 首页
    { path: '', component: HomePageComponent, title: '首页' },

    // 关于页
    { path: 'about', component: AboutPageComponent, title: '关于' },

    // 用户详情（带路径参数和守卫）
    {
        path: 'user/:id',
        component: UserDetailComponent,
        canActivate: [authGuard],
        resolve: { user: userResolver },
        title: '用户详情',
    },

    // 搜索结果（查询参数）
    { path: 'search', component: SearchResultsComponent, title: '搜索' },

    // 嵌套路由
    {
        path: 'dashboard',
        component: DashboardComponent,
        canActivate: [authGuard],
        children: [
            { path: '', redirectTo: 'overview', pathMatch: 'full' },
            { path: 'overview', component: DashboardOverviewComponent },
            { path: 'settings', component: DashboardSettingsComponent },
            { path: 'profile', component: DashboardProfileComponent },
        ],
    },

    // 懒加载示例
    // {
    //     path: 'admin',
    //     loadComponent: () => import('./admin.component').then(m => m.AdminComponent),
    // },

    // 通配符路由（必须放在最后）
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
 * 3. 路由参数订阅时注意取消订阅
 * 4. 使用 title 属性设置页面标题
 * 5. 通配符路由放在路由配置最后
 * 6. 使用 withComponentInputBinding() 自动绑定路由参数
 *
 * ❌ 避免做法：
 * 1. 路由嵌套过深 → 扁平化路由结构
 * 2. 在守卫中做复杂逻辑 → 委托给服务
 * 3. 忘记通配符路由 → 用户可能看到空白页
 * 4. 路由路径硬编码在组件中 → 使用常量或枚举
 */
