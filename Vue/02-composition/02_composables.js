/**
 * ============================================================
 *                Vue 3 Composables（组合式函数）
 * ============================================================
 * Composables 是使用 Composition API 封装和复用有状态逻辑的函数。
 * 类似于 React 的自定义 Hooks。
 *
 * 命名约定：以 "use" 开头，如 useCounter、useFetch
 * ============================================================
 */

import {
    ref,
    reactive,
    computed,
    watch,
    watchEffect,
    onMounted,
    onUnmounted,
    toValue,
    readonly,
} from 'vue';


// ============================================================
//                    1. useCounter - 计数器
// ============================================================

/**
 * 计数器 Composable
 *
 * 封装计数器的状态和逻辑
 *
 * @param {number} initialValue - 初始值，默认为 0
 * @returns {Object} 计数器状态和方法
 *
 * @example
 * ```vue
 * <script setup>
 * const { count, increment, decrement, reset } = useCounter(10);
 * </script>
 *
 * <template>
 *   <p>{{ count }}</p>
 *   <button @click="increment">+</button>
 *   <button @click="decrement">-</button>
 *   <button @click="reset">重置</button>
 * </template>
 * ```
 */
export function useCounter(initialValue = 0) {
    // 响应式状态
    const count = ref(initialValue);

    // 方法
    function increment() {
        count.value++;
    }

    function decrement() {
        count.value--;
    }

    function reset() {
        count.value = initialValue;
    }

    function set(value) {
        count.value = value;
    }

    // 计算属性
    const doubleCount = computed(() => count.value * 2);
    const isPositive = computed(() => count.value > 0);

    // 返回状态和方法
    return {
        count: readonly(count),  // 只读，防止外部直接修改
        doubleCount,
        isPositive,
        increment,
        decrement,
        reset,
        set,
    };
}


// ============================================================
//                    2. useFetch - 数据获取
// ============================================================

/**
 * 数据获取 Composable
 *
 * 封装异步数据获取逻辑，包括加载状态和错误处理
 *
 * @param {string | Ref<string> | () => string} url - 请求 URL
 * @param {Object} options - 请求选项
 * @returns {Object} 数据、加载状态、错误信息和方法
 *
 * @example
 * ```vue
 * <script setup>
 * const { data, loading, error, execute } = useFetch('/api/users');
 * </script>
 *
 * <template>
 *   <div v-if="loading">加载中...</div>
 *   <div v-else-if="error">{{ error }}</div>
 *   <div v-else>{{ data }}</div>
 * </template>
 * ```
 */
export function useFetch(url, options = {}) {
    const {
        immediate = true,      // 是否立即执行
        refetch = false,       // URL 变化时是否重新获取
        initialData = null,    // 初始数据
    } = options;

    // 状态
    const data = ref(initialData);
    const loading = ref(false);
    const error = ref(null);

    // 获取数据
    async function execute() {
        // 解析 URL（支持 ref 和函数）
        const resolvedUrl = toValue(url);

        if (!resolvedUrl) return;

        loading.value = true;
        error.value = null;

        try {
            const response = await fetch(resolvedUrl);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            data.value = await response.json();
        } catch (err) {
            error.value = err.message || '请求失败';
        } finally {
            loading.value = false;
        }
    }

    // 立即执行
    if (immediate) {
        execute();
    }

    // URL 变化时重新获取
    if (refetch) {
        watch(
            () => toValue(url),
            () => execute(),
            { immediate: false }
        );
    }

    return {
        data,
        loading,
        error,
        execute,  // 手动重新获取
    };
}


// ============================================================
//                    3. useLocalStorage - 本地存储
// ============================================================

/**
 * 本地存储 Composable
 *
 * 将 ref 的值同步到 localStorage
 *
 * @param {string} key - 存储的键名
 * @param {any} defaultValue - 默认值
 * @returns {Ref} 响应式的值
 *
 * @example
 * ```vue
 * <script setup>
 * const theme = useLocalStorage('theme', 'light');
 * </script>
 *
 * <template>
 *   <select v-model="theme">
 *     <option value="light">浅色</option>
 *     <option value="dark">深色</option>
 *   </select>
 * </template>
 * ```
 */
export function useLocalStorage(key, defaultValue) {
    // 从 localStorage 读取初始值
    const storedValue = localStorage.getItem(key);
    const initialValue = storedValue ? JSON.parse(storedValue) : defaultValue;

    // 创建响应式引用
    const data = ref(initialValue);

    // 监听变化，同步到 localStorage
    watch(
        data,
        (newValue) => {
            if (newValue === null || newValue === undefined) {
                localStorage.removeItem(key);
            } else {
                localStorage.setItem(key, JSON.stringify(newValue));
            }
        },
        { deep: true }
    );

    return data;
}


// ============================================================
//                    4. useEventListener - 事件监听
// ============================================================

/**
 * 事件监听 Composable
 *
 * 自动在组件卸载时移除事件监听
 *
 * @param {EventTarget} target - 目标元素
 * @param {string} event - 事件名
 * @param {Function} handler - 事件处理函数
 * @param {Object} options - addEventListener 选项
 *
 * @example
 * ```vue
 * <script setup>
 * useEventListener(window, 'resize', () => {
 *   console.log('窗口大小变化');
 * });
 * </script>
 * ```
 */
export function useEventListener(target, event, handler, options = {}) {
    // 支持 ref 作为 target
    const targetElement = toValue(target);

    onMounted(() => {
        targetElement?.addEventListener(event, handler, options);
    });

    onUnmounted(() => {
        targetElement?.removeEventListener(event, handler, options);
    });
}


// ============================================================
//                    5. useWindowSize - 窗口尺寸
// ============================================================

/**
 * 窗口尺寸 Composable
 *
 * 响应式获取窗口尺寸
 *
 * @returns {Object} 窗口宽度和高度
 *
 * @example
 * ```vue
 * <script setup>
 * const { width, height } = useWindowSize();
 * </script>
 *
 * <template>
 *   <p>窗口: {{ width }} x {{ height }}</p>
 * </template>
 * ```
 */
export function useWindowSize() {
    const width = ref(window.innerWidth);
    const height = ref(window.innerHeight);

    function update() {
        width.value = window.innerWidth;
        height.value = window.innerHeight;
    }

    onMounted(() => {
        window.addEventListener('resize', update);
    });

    onUnmounted(() => {
        window.removeEventListener('resize', update);
    });

    return {
        width: readonly(width),
        height: readonly(height),
    };
}


// ============================================================
//                    6. useMouse - 鼠标位置
// ============================================================

/**
 * 鼠标位置 Composable
 *
 * 响应式追踪鼠标位置
 *
 * @returns {Object} 鼠标 x, y 坐标
 *
 * @example
 * ```vue
 * <script setup>
 * const { x, y } = useMouse();
 * </script>
 *
 * <template>
 *   <p>鼠标位置: ({{ x }}, {{ y }})</p>
 * </template>
 * ```
 */
export function useMouse() {
    const x = ref(0);
    const y = ref(0);

    function update(event) {
        x.value = event.clientX;
        y.value = event.clientY;
    }

    onMounted(() => {
        window.addEventListener('mousemove', update);
    });

    onUnmounted(() => {
        window.removeEventListener('mousemove', update);
    });

    return { x: readonly(x), y: readonly(y) };
}


// ============================================================
//                    7. useDebounce - 防抖
// ============================================================

/**
 * 防抖 Composable
 *
 * 创建一个防抖的 ref
 *
 * @param {any} value - 初始值
 * @param {number} delay - 延迟时间（毫秒）
 * @returns {Object} 原始值和防抖后的值
 *
 * @example
 * ```vue
 * <script setup>
 * const { value, debouncedValue } = useDebounce('', 300);
 * </script>
 *
 * <template>
 *   <input v-model="value" />
 *   <p>防抖值: {{ debouncedValue }}</p>
 * </template>
 * ```
 */
export function useDebounce(initialValue, delay = 300) {
    const value = ref(initialValue);
    const debouncedValue = ref(initialValue);

    let timeoutId = null;

    watch(value, (newValue) => {
        // 清除之前的定时器
        if (timeoutId) {
            clearTimeout(timeoutId);
        }

        // 设置新的定时器
        timeoutId = setTimeout(() => {
            debouncedValue.value = newValue;
        }, delay);
    });

    // 清理定时器
    onUnmounted(() => {
        if (timeoutId) {
            clearTimeout(timeoutId);
        }
    });

    return {
        value,
        debouncedValue: readonly(debouncedValue),
    };
}


// ============================================================
//                    8. useToggle - 切换状态
// ============================================================

/**
 * 切换状态 Composable
 *
 * @param {boolean} initialValue - 初始值
 * @returns {Array} [状态, 切换函数, 设置函数]
 *
 * @example
 * ```vue
 * <script setup>
 * const [isOpen, toggle, setOpen] = useToggle(false);
 * </script>
 *
 * <template>
 *   <button @click="toggle">切换</button>
 *   <div v-if="isOpen">内容</div>
 * </template>
 * ```
 */
export function useToggle(initialValue = false) {
    const state = ref(initialValue);

    function toggle() {
        state.value = !state.value;
    }

    function set(value) {
        state.value = value;
    }

    return [readonly(state), toggle, set];
}


// ============================================================
//                    9. useAsync - 异步状态
// ============================================================

/**
 * 异步状态 Composable
 *
 * 管理异步操作的状态
 *
 * @param {Function} asyncFn - 异步函数
 * @param {Object} options - 选项
 * @returns {Object} 状态和执行函数
 *
 * @example
 * ```vue
 * <script setup>
 * const { execute, loading, error, data } = useAsync(async () => {
 *   const response = await fetch('/api/data');
 *   return response.json();
 * });
 * </script>
 * ```
 */
export function useAsync(asyncFn, options = {}) {
    const { immediate = false, initialData = null } = options;

    const data = ref(initialData);
    const loading = ref(false);
    const error = ref(null);

    async function execute(...args) {
        loading.value = true;
        error.value = null;

        try {
            data.value = await asyncFn(...args);
            return data.value;
        } catch (err) {
            error.value = err;
            throw err;
        } finally {
            loading.value = false;
        }
    }

    if (immediate) {
        execute();
    }

    return {
        data,
        loading,
        error,
        execute,
    };
}


// ============================================================
//                    10. useForm - 表单处理
// ============================================================

/**
 * 表单处理 Composable
 *
 * @param {Object} initialValues - 初始表单值
 * @param {Object} validationRules - 验证规则
 * @returns {Object} 表单状态和方法
 *
 * @example
 * ```vue
 * <script setup>
 * const { values, errors, handleSubmit, resetForm } = useForm(
 *   { email: '', password: '' },
 *   {
 *     email: (v) => /.+@.+/.test(v) || '邮箱格式不正确',
 *     password: (v) => v.length >= 6 || '密码至少6位',
 *   }
 * );
 *
 * const onSubmit = handleSubmit((values) => {
 *   console.log('提交:', values);
 * });
 * </script>
 * ```
 */
export function useForm(initialValues, validationRules = {}) {
    // 表单值
    const values = reactive({ ...initialValues });

    // 错误信息
    const errors = reactive({});

    // 是否被修改
    const isDirty = ref(false);

    // 是否正在提交
    const isSubmitting = ref(false);

    // 验证单个字段
    function validateField(field) {
        const rule = validationRules[field];
        if (!rule) return true;

        const result = rule(values[field]);
        if (result === true) {
            delete errors[field];
            return true;
        } else {
            errors[field] = result;
            return false;
        }
    }

    // 验证所有字段
    function validate() {
        let isValid = true;
        for (const field in validationRules) {
            if (!validateField(field)) {
                isValid = false;
            }
        }
        return isValid;
    }

    // 处理提交
    function handleSubmit(onSubmit) {
        return async (event) => {
            event?.preventDefault();

            if (!validate()) return;

            isSubmitting.value = true;
            try {
                await onSubmit(values);
            } finally {
                isSubmitting.value = false;
            }
        };
    }

    // 重置表单
    function resetForm() {
        Object.assign(values, initialValues);
        Object.keys(errors).forEach((key) => delete errors[key]);
        isDirty.value = false;
    }

    // 设置字段值
    function setFieldValue(field, value) {
        values[field] = value;
        isDirty.value = true;
    }

    // 监听值变化，标记为已修改
    watch(
        () => values,
        () => {
            isDirty.value = true;
        },
        { deep: true }
    );

    return {
        values,
        errors,
        isDirty,
        isSubmitting,
        validateField,
        validate,
        handleSubmit,
        resetForm,
        setFieldValue,
    };
}


// ============================================================
//                    导出所有 Composables
// ============================================================

export default {
    useCounter,
    useFetch,
    useLocalStorage,
    useEventListener,
    useWindowSize,
    useMouse,
    useDebounce,
    useToggle,
    useAsync,
    useForm,
};
