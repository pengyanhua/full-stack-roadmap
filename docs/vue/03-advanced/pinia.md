# pinia.js

::: info 文件信息
- 📄 原文件：`02_pinia.js`
- 🔤 语言：javascript
:::

Pinia 状态管理
Pinia 是 Vue 的官方状态管理库（Vue 3 推荐）。
替代 Vuex，提供更简单的 API 和更好的 TypeScript 支持。
安装：npm install pinia

## 完整代码

```javascript
/**
 * ============================================================
 *                    Pinia 状态管理
 * ============================================================
 * Pinia 是 Vue 的官方状态管理库（Vue 3 推荐）。
 * 替代 Vuex，提供更简单的 API 和更好的 TypeScript 支持。
 *
 * 安装：npm install pinia
 * ============================================================
 */

import { defineStore, storeToRefs } from 'pinia';
import { ref, computed } from 'vue';

// ============================================================
//                    1. 创建 Pinia 实例
// ============================================================

/**
 * 在 main.js 中：
 *
 * ```js
 * import { createApp } from 'vue';
 * import { createPinia } from 'pinia';
 * import App from './App.vue';
 *
 * const app = createApp(App);
 * const pinia = createPinia();
 *
 * app.use(pinia);
 * app.mount('#app');
 * ```
 */


// ============================================================
//                    2. 定义 Store（选项式）
// ============================================================

/**
 * 【选项式 Store】
 *
 * 类似 Vue 2 的选项式 API：
 * - state: 状态（类似 data）
 * - getters: 计算属性（类似 computed）
 * - actions: 方法（类似 methods，可以是异步的）
 */

export const useCounterStore = defineStore('counter', {
    /**
     * state：定义状态
     * 必须是一个函数，返回初始状态对象
     */
    state: () => ({
        count: 0,
        name: 'Counter Store',
        items: [],
    }),

    /**
     * getters：计算属性
     * 可以访问 this 获取状态
     * 也可以访问其他 getter
     */
    getters: {
        // 基本 getter
        doubleCount: (state) => state.count * 2,

        // 使用 this 访问其他 getter
        quadrupleCount() {
            return this.doubleCount * 2;
        },

        // 带参数的 getter（返回函数）
        getItemById: (state) => {
            return (id) => state.items.find(item => item.id === id);
        },

        // 访问其他 store
        // combinedData() {
        //     const userStore = useUserStore();
        //     return `${this.name} by ${userStore.username}`;
        // },
    },

    /**
     * actions：方法
     * 可以是同步或异步
     * 使用 this 访问状态
     */
    actions: {
        // 同步 action
        increment() {
            this.count++;
        },

        decrement() {
            this.count--;
        },

        // 带参数的 action
        incrementBy(amount) {
            this.count += amount;
        },

        // 异步 action
        async fetchItems() {
            try {
                const response = await fetch('/api/items');
                this.items = await response.json();
            } catch (error) {
                console.error('获取数据失败:', error);
                throw error;
            }
        },

        // 调用其他 action
        reset() {
            this.count = 0;
            this.items = [];
        },
    },
});


// ============================================================
//                    3. 定义 Store（组合式）
// ============================================================

/**
 * 【组合式 Store】
 *
 * 使用 Composition API 风格：
 * - ref/reactive 定义状态
 * - computed 定义 getters
 * - function 定义 actions
 *
 * 更灵活，可以使用 composables
 */

export const useUserStore = defineStore('user', () => {
    // ===== State =====
    // 使用 ref 定义响应式状态
    const user = ref(null);
    const isLoading = ref(false);
    const error = ref(null);
    const token = ref(localStorage.getItem('token') || null);

    // ===== Getters =====
    // 使用 computed 定义计算属性
    const isAuthenticated = computed(() => !!token.value);

    const username = computed(() => user.value?.name || 'Guest');

    const userRole = computed(() => user.value?.role || 'visitor');

    // ===== Actions =====
    // 普通函数作为 actions

    /**
     * 登录
     * @param {string} email - 邮箱
     * @param {string} password - 密码
     */
    async function login(email, password) {
        isLoading.value = true;
        error.value = null;

        try {
            // 模拟 API 请求
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });

            if (!response.ok) {
                throw new Error('登录失败');
            }

            const data = await response.json();

            // 保存用户信息和 token
            user.value = data.user;
            token.value = data.token;
            localStorage.setItem('token', data.token);

            return data;
        } catch (err) {
            error.value = err.message;
            throw err;
        } finally {
            isLoading.value = false;
        }
    }

    /**
     * 登出
     */
    function logout() {
        user.value = null;
        token.value = null;
        localStorage.removeItem('token');
    }

    /**
     * 获取当前用户信息
     */
    async function fetchUser() {
        if (!token.value) return;

        isLoading.value = true;

        try {
            const response = await fetch('/api/user', {
                headers: { Authorization: `Bearer ${token.value}` },
            });

            if (!response.ok) {
                throw new Error('获取用户信息失败');
            }

            user.value = await response.json();
        } catch (err) {
            error.value = err.message;
            // Token 无效，清除登录状态
            logout();
        } finally {
            isLoading.value = false;
        }
    }

    /**
     * 更新用户信息
     */
    async function updateProfile(profileData) {
        isLoading.value = true;

        try {
            const response = await fetch('/api/user', {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${token.value}`,
                },
                body: JSON.stringify(profileData),
            });

            user.value = await response.json();
        } catch (err) {
            error.value = err.message;
            throw err;
        } finally {
            isLoading.value = false;
        }
    }

    // 必须返回所有要暴露的状态和方法
    return {
        // State
        user,
        isLoading,
        error,
        token,

        // Getters
        isAuthenticated,
        username,
        userRole,

        // Actions
        login,
        logout,
        fetchUser,
        updateProfile,
    };
});


// ============================================================
//                    4. 购物车 Store 示例
// ============================================================

/**
 * 完整的购物车 Store 示例
 * 展示复杂状态管理
 */

export const useCartStore = defineStore('cart', () => {
    // ===== State =====
    const items = ref([]);
    const couponCode = ref('');
    const discount = ref(0);

    // ===== Getters =====

    // 购物车商品数量
    const itemCount = computed(() => {
        return items.value.reduce((sum, item) => sum + item.quantity, 0);
    });

    // 小计（不含折扣）
    const subtotal = computed(() => {
        return items.value.reduce((sum, item) => {
            return sum + item.price * item.quantity;
        }, 0);
    });

    // 折扣金额
    const discountAmount = computed(() => {
        return subtotal.value * discount.value;
    });

    // 总计
    const total = computed(() => {
        return subtotal.value - discountAmount.value;
    });

    // 购物车是否为空
    const isEmpty = computed(() => items.value.length === 0);

    // 检查商品是否在购物车中
    const isInCart = computed(() => {
        return (productId) => items.value.some(item => item.id === productId);
    });

    // ===== Actions =====

    /**
     * 添加商品到购物车
     */
    function addItem(product) {
        const existingItem = items.value.find(item => item.id === product.id);

        if (existingItem) {
            existingItem.quantity++;
        } else {
            items.value.push({
                id: product.id,
                name: product.name,
                price: product.price,
                image: product.image,
                quantity: 1,
            });
        }

        // 保存到本地存储
        saveToStorage();
    }

    /**
     * 从购物车移除商品
     */
    function removeItem(productId) {
        const index = items.value.findIndex(item => item.id === productId);
        if (index > -1) {
            items.value.splice(index, 1);
            saveToStorage();
        }
    }

    /**
     * 更新商品数量
     */
    function updateQuantity(productId, quantity) {
        const item = items.value.find(item => item.id === productId);
        if (item) {
            if (quantity <= 0) {
                removeItem(productId);
            } else {
                item.quantity = quantity;
                saveToStorage();
            }
        }
    }

    /**
     * 清空购物车
     */
    function clearCart() {
        items.value = [];
        couponCode.value = '';
        discount.value = 0;
        saveToStorage();
    }

    /**
     * 应用优惠券
     */
    async function applyCoupon(code) {
        try {
            // 模拟 API 验证优惠券
            const response = await fetch(`/api/coupons/${code}`);
            const coupon = await response.json();

            if (coupon.valid) {
                couponCode.value = code;
                discount.value = coupon.discount;  // 如 0.1 表示 10% 折扣
                return { success: true, message: `优惠券已应用，折扣 ${coupon.discount * 100}%` };
            } else {
                return { success: false, message: '优惠券无效' };
            }
        } catch (error) {
            return { success: false, message: '验证优惠券失败' };
        }
    }

    /**
     * 保存到本地存储
     */
    function saveToStorage() {
        localStorage.setItem('cart', JSON.stringify({
            items: items.value,
            couponCode: couponCode.value,
            discount: discount.value,
        }));
    }

    /**
     * 从本地存储恢复
     */
    function loadFromStorage() {
        const saved = localStorage.getItem('cart');
        if (saved) {
            const data = JSON.parse(saved);
            items.value = data.items || [];
            couponCode.value = data.couponCode || '';
            discount.value = data.discount || 0;
        }
    }

    /**
     * 结算
     */
    async function checkout() {
        const userStore = useUserStore();

        if (!userStore.isAuthenticated) {
            throw new Error('请先登录');
        }

        if (isEmpty.value) {
            throw new Error('购物车为空');
        }

        try {
            const response = await fetch('/api/orders', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${userStore.token}`,
                },
                body: JSON.stringify({
                    items: items.value,
                    couponCode: couponCode.value,
                    total: total.value,
                }),
            });

            const order = await response.json();

            // 清空购物车
            clearCart();

            return order;
        } catch (error) {
            throw new Error('结算失败');
        }
    }

    // 初始化时从存储恢复
    loadFromStorage();

    return {
        // State
        items,
        couponCode,
        discount,

        // Getters
        itemCount,
        subtotal,
        discountAmount,
        total,
        isEmpty,
        isInCart,

        // Actions
        addItem,
        removeItem,
        updateQuantity,
        clearCart,
        applyCoupon,
        checkout,
    };
});


// ============================================================
//                    5. 在组件中使用 Store
// ============================================================

/**
 * 组件中使用示例：
 *
 * ```vue
 * <template>
 *   <div>
 *     <h1>计数器: {{ counter.count }}</h1>
 *     <p>双倍: {{ counter.doubleCount }}</p>
 *     <button @click="counter.increment()">+1</button>
 *     <button @click="counter.decrement()">-1</button>
 *
 *     <h2>用户: {{ user.username }}</h2>
 *     <p v-if="user.isAuthenticated">已登录</p>
 *
 *     <h3>购物车 ({{ cart.itemCount }})</h3>
 *     <ul>
 *       <li v-for="item in cart.items" :key="item.id">
 *         {{ item.name }} x {{ item.quantity }}
 *         <button @click="cart.removeItem(item.id)">删除</button>
 *       </li>
 *     </ul>
 *     <p>总计: ¥{{ cart.total.toFixed(2) }}</p>
 *   </div>
 * </template>
 *
 * <script setup>
 * import { storeToRefs } from 'pinia';
 * import { useCounterStore, useUserStore, useCartStore } from '@/stores';
 *
 * // 获取 store 实例
 * const counter = useCounterStore();
 * const user = useUserStore();
 * const cart = useCartStore();
 *
 * // 直接使用 store 的状态和方法
 * console.log(counter.count);
 * counter.increment();
 *
 * // ===== 解构响应式状态 =====
 * // ❌ 错误：直接解构会丢失响应式
 * // const { count, doubleCount } = counter;
 *
 * // ✅ 正确：使用 storeToRefs
 * const { count, doubleCount } = storeToRefs(counter);
 *
 * // ✅ actions 可以直接解构（它们只是函数）
 * const { increment, decrement } = counter;
 *
 * // ===== 监听状态变化 =====
 * import { watch } from 'vue';
 *
 * watch(
 *   () => counter.count,
 *   (newCount) => {
 *     console.log('count 变化:', newCount);
 *   }
 * );
 *
 * // ===== 订阅状态变化 =====
 * counter.$subscribe((mutation, state) => {
 *   console.log('状态变化:', mutation.type);
 *   console.log('新状态:', state);
 *
 *   // 保存到本地存储
 *   localStorage.setItem('counter', JSON.stringify(state));
 * });
 *
 * // ===== 订阅 action =====
 * counter.$onAction(({ name, args, after, onError }) => {
 *   console.log(`调用 action: ${name}`);
 *   console.log('参数:', args);
 *
 *   after((result) => {
 *     console.log('action 完成，返回值:', result);
 *   });
 *
 *   onError((error) => {
 *     console.error('action 出错:', error);
 *   });
 * });
 *
 * // ===== 重置 store =====
 * counter.$reset();  // 重置到初始状态
 *
 * // ===== 批量更新状态 =====
 * counter.$patch({
 *   count: 10,
 *   name: 'New Name',
 * });
 *
 * // 或使用函数
 * counter.$patch((state) => {
 *   state.count++;
 *   state.items.push({ id: 1, name: 'Item' });
 * });
 * </script>
 * ```
 */


// ============================================================
//                    6. Store 之间互相访问
// ============================================================

/**
 * Store 可以访问其他 store
 */

export const useOrderStore = defineStore('order', () => {
    const userStore = useUserStore();
    const cartStore = useCartStore();

    const orders = ref([]);
    const currentOrder = ref(null);

    async function createOrder() {
        if (!userStore.isAuthenticated) {
            throw new Error('请先登录');
        }

        const order = await cartStore.checkout();
        orders.value.push(order);
        currentOrder.value = order;

        return order;
    }

    return {
        orders,
        currentOrder,
        createOrder,
    };
});


// ============================================================
//                    7. 插件
// ============================================================

/**
 * Pinia 插件示例
 *
 * ```js
 * // plugins/piniaLogger.js
 * export function piniaLogger({ store }) {
 *   // 每个 store 创建时调用
 *
 *   // 添加自定义属性
 *   store.customProperty = 'hello';
 *
 *   // 监听状态变化
 *   store.$subscribe((mutation) => {
 *     console.log(`[${store.$id}] ${mutation.type}`);
 *   });
 *
 *   // 监听 action
 *   store.$onAction(({ name, after, onError }) => {
 *     const startTime = Date.now();
 *
 *     after(() => {
 *       console.log(`[${store.$id}] ${name} 耗时 ${Date.now() - startTime}ms`);
 *     });
 *
 *     onError((error) => {
 *       console.error(`[${store.$id}] ${name} 失败:`, error);
 *     });
 *   });
 * }
 *
 * // main.js
 * const pinia = createPinia();
 * pinia.use(piniaLogger);
 * ```
 */


// ============================================================
//                    8. 持久化插件
// ============================================================

/**
 * 持久化插件示例
 *
 * ```js
 * // plugins/piniaPersist.js
 * export function piniaPersist({ store }) {
 *   // 从存储恢复状态
 *   const saved = localStorage.getItem(store.$id);
 *   if (saved) {
 *     store.$patch(JSON.parse(saved));
 *   }
 *
 *   // 监听变化并保存
 *   store.$subscribe((mutation, state) => {
 *     localStorage.setItem(store.$id, JSON.stringify(state));
 *   });
 * }
 * ```
 *
 * 或者使用现成的插件：pinia-plugin-persistedstate
 * npm install pinia-plugin-persistedstate
 */


export default {
    useCounterStore,
    useUserStore,
    useCartStore,
    useOrderStore,
};
```
