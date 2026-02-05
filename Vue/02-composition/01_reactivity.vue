<!--
============================================================
                Vue 3 Composition API - 响应式
============================================================
Composition API 是 Vue 3 的核心特性，提供了更灵活的代码组织方式。
本文件介绍响应式系统的核心 API。
============================================================
-->

<template>
    <div class="tutorial">
        <h1>Vue 3 响应式系统</h1>

        <!-- ref 示例 -->
        <section>
            <h2>1. ref - 基本类型响应式</h2>
            <p>计数: {{ count }}</p>
            <p>双倍: {{ doubleCount }}</p>
            <button @click="increment">增加</button>
            <button @click="decrement">减少</button>
        </section>

        <!-- reactive 示例 -->
        <section>
            <h2>2. reactive - 对象响应式</h2>
            <p>姓名: {{ user.name }}</p>
            <p>年龄: {{ user.age }}</p>
            <p>邮箱: {{ user.email }}</p>
            <button @click="updateUser">更新用户</button>
        </section>

        <!-- 计算属性示例 -->
        <section>
            <h2>3. computed - 计算属性</h2>
            <input v-model="firstName" placeholder="名" />
            <input v-model="lastName" placeholder="姓" />
            <p>全名: {{ fullName }}</p>
            <p>全名（可写）: {{ fullNameWritable }}</p>
            <button @click="fullNameWritable = '李 四'">设置为李四</button>
        </section>

        <!-- watch 示例 -->
        <section>
            <h2>4. watch - 侦听器</h2>
            <input v-model="searchQuery" placeholder="搜索..." />
            <p>搜索结果: {{ searchResults }}</p>
        </section>

        <!-- 生命周期 -->
        <section>
            <h2>5. 生命周期</h2>
            <p>组件已挂载: {{ mounted ? '是' : '否' }}</p>
            <p>更新次数: {{ updateCount }}</p>
        </section>
    </div>
</template>

<script setup>
/**
 * ============================================================
 *                    1. ref - 基本类型响应式
 * ============================================================
 *
 * ref() 用于创建响应式的基本类型值：
 * - 返回一个 Ref 对象
 * - 在 JS 中通过 .value 访问/修改
 * - 在模板中自动解包（不需要 .value）
 * - 也可以包装对象（但推荐用 reactive）
 */

import {
    ref,
    reactive,
    computed,
    watch,
    watchEffect,
    onMounted,
    onUpdated,
    onUnmounted,
    toRef,
    toRefs,
    isRef,
    unref,
    shallowRef,
    triggerRef,
} from 'vue';

// --- 创建 ref ---
const count = ref(0);  // 初始值为 0

// 在 setup 中访问需要 .value
console.log('初始 count:', count.value);

// 修改值
function increment() {
    count.value++;  // 必须用 .value
}

function decrement() {
    count.value--;
}

// --- ref 的类型推断 ---
const message = ref('Hello');  // 推断为 Ref<string>
const items = ref<string[]>([]); // 显式指定类型

// --- ref 也可以包装对象 ---
const userRef = ref({
    name: 'Alice',
    age: 25,
});
// 访问：userRef.value.name


/**
 * ============================================================
 *                    2. reactive - 对象响应式
 * ============================================================
 *
 * reactive() 用于创建响应式对象：
 * - 深层响应式（嵌套对象也是响应式的）
 * - 不需要 .value
 * - 不能用于基本类型
 * - 解构会丢失响应式（需要用 toRefs）
 */

// --- 创建 reactive ---
const user = reactive({
    name: 'Alice',
    age: 25,
    email: 'alice@example.com',
    address: {
        city: 'Beijing',
        street: 'Main St',
    },
});

// 直接访问和修改（不需要 .value）
console.log('用户名:', user.name);

function updateUser() {
    user.name = 'Bob';
    user.age++;
    // 嵌套对象也是响应式的
    user.address.city = 'Shanghai';
}

// --- 解构问题 ---
// ❌ 错误：解构会丢失响应式
// const { name, age } = user;

// ✅ 正确：使用 toRefs
const { name: userName, age: userAge } = toRefs(user);

// 或者使用 toRef 获取单个属性
const userEmail = toRef(user, 'email');


/**
 * ============================================================
 *                    3. computed - 计算属性
 * ============================================================
 *
 * computed() 创建计算属性：
 * - 自动追踪依赖
 * - 缓存结果（依赖不变不会重新计算）
 * - 默认只读，可以配置 getter/setter
 */

const firstName = ref('张');
const lastName = ref('三');

// --- 只读计算属性 ---
const fullName = computed(() => {
    console.log('计算 fullName');  // 只有依赖变化时才打印
    return `${firstName.value} ${lastName.value}`;
});

// --- 可写计算属性 ---
const fullNameWritable = computed({
    get() {
        return `${firstName.value} ${lastName.value}`;
    },
    set(newValue) {
        // 解析设置的值
        const parts = newValue.split(' ');
        firstName.value = parts[0] || '';
        lastName.value = parts[1] || '';
    },
});

// --- 基于 count 的计算属性 ---
const doubleCount = computed(() => count.value * 2);


/**
 * ============================================================
 *                    4. watch - 侦听器
 * ============================================================
 *
 * watch() 用于侦听响应式数据的变化：
 * - 可以执行副作用（如 API 请求）
 * - 支持侦听多个源
 * - 可以访问新值和旧值
 */

const searchQuery = ref('');
const searchResults = ref([]);

// --- 侦听单个 ref ---
watch(searchQuery, (newValue, oldValue) => {
    console.log(`搜索词从 "${oldValue}" 变为 "${newValue}"`);

    // 模拟搜索
    if (newValue.trim()) {
        // 模拟异步搜索
        searchResults.value = [
            `结果 1: ${newValue}`,
            `结果 2: ${newValue}`,
        ];
    } else {
        searchResults.value = [];
    }
});

// --- 侦听 reactive 对象的属性 ---
// 需要使用 getter 函数
watch(
    () => user.name,
    (newName) => {
        console.log('用户名变化:', newName);
    }
);

// --- 侦听整个 reactive 对象 ---
watch(
    user,  // 直接传 reactive 对象
    (newUser) => {
        console.log('用户对象变化:', newUser);
    },
    { deep: true }  // 深度侦听
);

// --- 侦听多个源 ---
watch(
    [firstName, lastName],
    ([newFirst, newLast], [oldFirst, oldLast]) => {
        console.log(`姓名从 "${oldFirst} ${oldLast}" 变为 "${newFirst} ${newLast}"`);
    }
);

// --- watch 选项 ---
watch(
    count,
    (newCount) => {
        console.log('count 变化:', newCount);
    },
    {
        immediate: true,  // 立即执行一次
        deep: true,       // 深度侦听（对于对象）
        flush: 'post',    // 回调时机：'pre' | 'post' | 'sync'
        // once: true,    // Vue 3.4+: 只触发一次
    }
);


/**
 * ============================================================
 *                    5. watchEffect - 自动收集依赖
 * ============================================================
 *
 * watchEffect() 自动追踪回调中使用的响应式数据：
 * - 立即执行一次
 * - 自动收集依赖
 * - 不需要指定侦听源
 */

// 自动追踪 count 和 searchQuery
const stopWatchEffect = watchEffect(() => {
    console.log('watchEffect 执行');
    console.log('当前 count:', count.value);
    console.log('当前搜索词:', searchQuery.value);
});

// 停止侦听
// stopWatchEffect();

// --- watchEffect 清理副作用 ---
watchEffect((onCleanup) => {
    // 模拟订阅
    const unsubscribe = subscribeToSomething();

    // 清理函数：在下次执行前或组件卸载时调用
    onCleanup(() => {
        unsubscribe();
    });
});

function subscribeToSomething() {
    console.log('订阅');
    return () => console.log('取消订阅');
}


/**
 * ============================================================
 *                    6. 生命周期钩子
 * ============================================================
 *
 * Composition API 中的生命周期钩子：
 * - onBeforeMount
 * - onMounted
 * - onBeforeUpdate
 * - onUpdated
 * - onBeforeUnmount
 * - onUnmounted
 * - onErrorCaptured
 * - onActivated / onDeactivated (keep-alive)
 */

const mounted = ref(false);
const updateCount = ref(0);

// 组件挂载后
onMounted(() => {
    console.log('组件已挂载');
    mounted.value = true;

    // 可以访问 DOM
    // const el = document.querySelector('.tutorial');
});

// 组件更新后
onUpdated(() => {
    console.log('组件已更新');
    updateCount.value++;
});

// 组件卸载前
onUnmounted(() => {
    console.log('组件将卸载');
    // 清理工作：取消订阅、清除定时器等
});


/**
 * ============================================================
 *                    7. 响应式工具函数
 * ============================================================
 */

// --- isRef: 检查是否为 ref ---
console.log('count 是 ref:', isRef(count));  // true
console.log('user 是 ref:', isRef(user));    // false

// --- unref: 解包 ref ---
// 如果是 ref 返回 .value，否则返回原值
const unwrapped = unref(count);  // 等同于 count.value

// --- toRef: 为 reactive 的属性创建 ref ---
const nameRef = toRef(user, 'name');
// nameRef.value 和 user.name 保持同步

// --- toRefs: 将 reactive 的所有属性转为 ref ---
const userRefs = toRefs(user);
// userRefs.name, userRefs.age 等都是 ref

// --- shallowRef: 浅层 ref ---
// 只有 .value 的替换是响应式的，内部变化不触发更新
const shallowState = shallowRef({ count: 0 });

// 不会触发更新
shallowState.value.count = 1;

// 触发更新
shallowState.value = { count: 1 };

// 强制触发更新
// triggerRef(shallowState);


/**
 * ============================================================
 *                    8. 响应式原理简介
 * ============================================================
 *
 * Vue 3 使用 Proxy 实现响应式：
 *
 * 1. reactive() 使用 Proxy 包装对象
 *    - 拦截 get: 收集依赖
 *    - 拦截 set: 触发更新
 *
 * 2. ref() 也是基于 Proxy（RefImpl 类）
 *    - .value 的 get/set 触发响应式
 *
 * 3. effect() 是底层 API
 *    - watch/watchEffect/computed 都基于它
 *    - 建立数据与副作用的关联
 */
</script>

<style scoped>
.tutorial {
    padding: 20px;
    max-width: 800px;
    margin: 0 auto;
}

section {
    margin-bottom: 30px;
    padding: 15px;
    border: 1px solid #eee;
    border-radius: 8px;
}

h2 {
    color: #42b883;
    margin-top: 0;
}

input {
    padding: 8px;
    margin: 5px;
    border: 1px solid #ddd;
    border-radius: 4px;
}

button {
    padding: 8px 16px;
    margin: 5px;
    background: #42b883;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

button:hover {
    background: #3aa876;
}
</style>
