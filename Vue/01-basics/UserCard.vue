<!--
============================================================
                    UserCard 子组件
============================================================
演示 Vue 3 组件的核心概念：
- Props（属性）
- Emits（事件）
- Slots（插槽）
============================================================
-->

<template>
    <div class="user-card" :class="{ 'user-card--highlighted': highlighted }">
        <!-- 头像 -->
        <div class="user-card__avatar">
            <img :src="user.avatar || defaultAvatar" :alt="user.name" />
        </div>

        <!-- 信息 -->
        <div class="user-card__info">
            <h3 class="user-card__name">{{ user.name }}</h3>

            <!-- 条件显示邮箱 -->
            <p v-if="showEmail" class="user-card__email">
                {{ user.email }}
            </p>

            <!-- 显示额外信息 -->
            <p v-if="user.bio" class="user-card__bio">
                {{ user.bio }}
            </p>
        </div>

        <!-- 操作按钮 -->
        <div class="user-card__actions">
            <!-- 默认插槽 -->
            <slot name="actions">
                <!-- 默认内容（当没有提供插槽内容时显示） -->
                <button @click="handleEdit">编辑</button>
                <button @click="handleDelete" class="btn-danger">删除</button>
            </slot>
        </div>

        <!-- 底部插槽 -->
        <div v-if="$slots.footer" class="user-card__footer">
            <slot name="footer"></slot>
        </div>
    </div>
</template>

<script setup>
/**
 * ============================================================
 *                    1. Props（属性）
 * ============================================================
 *
 * Props 是组件接收外部数据的方式：
 * - 单向数据流：父 -> 子
 * - 子组件不应该修改 props
 * - 支持类型验证和默认值
 */

import { computed, useSlots } from 'vue';

// --- defineProps: 定义组件接收的属性 ---
const props = defineProps({
    /**
     * 用户对象（必需）
     * @type {{ id: number, name: string, email: string, avatar?: string, bio?: string }}
     */
    user: {
        type: Object,
        required: true,  // 必需属性
        // 对象/数组默认值必须从函数返回
        default: () => ({
            id: 0,
            name: 'Unknown',
            email: '',
        }),
        // 自定义验证器
        validator(value) {
            return value.id !== undefined && value.name !== undefined;
        },
    },

    /**
     * 是否显示邮箱
     */
    showEmail: {
        type: Boolean,
        default: true,
    },

    /**
     * 是否高亮显示
     */
    highlighted: {
        type: Boolean,
        default: false,
    },

    /**
     * 卡片大小
     * @type {'small' | 'medium' | 'large'}
     */
    size: {
        type: String,
        default: 'medium',
        validator(value) {
            return ['small', 'medium', 'large'].includes(value);
        },
    },
});

// --- 在模板中直接使用 props ---
// 在 <script setup> 中使用 props.xxx
// 在 template 中直接使用 xxx（无需 props. 前缀）

// 默认头像
const defaultAvatar = '/images/default-avatar.png';


/**
 * ============================================================
 *                    2. Emits（事件）
 * ============================================================
 *
 * 子组件通过 emit 向父组件发送事件：
 * - 用于子 -> 父通信
 * - 可以携带数据
 * - 支持验证
 */

// --- defineEmits: 定义组件发出的事件 ---
const emit = defineEmits({
    /**
     * 更新事件
     * @param {Object} user - 更新的用户对象
     * @returns {boolean} 验证是否通过
     */
    update: (user) => {
        // 验证事件参数
        if (!user || !user.id) {
            console.warn('update 事件需要用户对象');
            return false;
        }
        return true;
    },

    /**
     * 删除事件
     * @param {number} userId - 用户 ID
     */
    delete: (userId) => {
        return typeof userId === 'number';
    },

    /**
     * 简单声明（不需要验证时）
     */
    click: null,
});

// --- 发出事件 ---
function handleEdit() {
    // 发出 update 事件，携带用户数据
    emit('update', props.user);
}

function handleDelete() {
    // 发出 delete 事件，携带用户 ID
    emit('delete', props.user.id);
}


/**
 * ============================================================
 *                    3. Slots（插槽）
 * ============================================================
 *
 * 插槽允许父组件向子组件传递模板内容：
 * - 默认插槽：<slot></slot>
 * - 具名插槽：<slot name="xxx"></slot>
 * - 作用域插槽：<slot :data="data"></slot>
 */

// 访问插槽
const slots = useSlots();

// 检查是否有某个插槽
const hasFooter = computed(() => !!slots.footer);


/**
 * ============================================================
 *                    4. 组件内部逻辑
 * ============================================================
 */

// 计算属性
const cardClass = computed(() => {
    return [
        'user-card',
        `user-card--${props.size}`,
        { 'user-card--highlighted': props.highlighted },
    ];
});

// 内部状态（不暴露给父组件）
import { ref } from 'vue';
const isExpanded = ref(false);

function toggleExpand() {
    isExpanded.value = !isExpanded.value;
}


/**
 * ============================================================
 *                    5. defineExpose（暴露给父组件）
 * ============================================================
 *
 * 默认情况下，<script setup> 中的绑定不会暴露给父组件
 * 使用 defineExpose 可以显式暴露
 */

defineExpose({
    // 暴露方法给父组件调用
    toggleExpand,
    // 暴露状态
    isExpanded,
});
</script>

<style scoped>
/**
 * ============================================================
 *                    组件样式
 * ============================================================
 *
 * scoped 属性：
 * - 样式只作用于当前组件
 * - 通过添加 data 属性实现隔离
 * - 不影响子组件的根元素
 */

.user-card {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    padding: 16px;
    margin: 10px 0;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    background: #fff;
    transition: all 0.3s ease;
}

.user-card:hover {
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.user-card--highlighted {
    border-color: #42b883;
    background: #f0fff4;
}

/* 尺寸变体 */
.user-card--small {
    padding: 8px;
}

.user-card--small .user-card__avatar img {
    width: 32px;
    height: 32px;
}

.user-card--large {
    padding: 24px;
}

.user-card--large .user-card__avatar img {
    width: 80px;
    height: 80px;
}

.user-card__avatar {
    margin-right: 16px;
}

.user-card__avatar img {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    object-fit: cover;
}

.user-card__info {
    flex: 1;
}

.user-card__name {
    margin: 0 0 4px;
    font-size: 16px;
    font-weight: 600;
}

.user-card__email {
    margin: 0;
    color: #666;
    font-size: 14px;
}

.user-card__bio {
    margin: 8px 0 0;
    color: #888;
    font-size: 13px;
}

.user-card__actions {
    display: flex;
    gap: 8px;
}

.user-card__actions button {
    padding: 6px 12px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: #fff;
    cursor: pointer;
    transition: all 0.2s;
}

.user-card__actions button:hover {
    background: #f5f5f5;
}

.user-card__actions .btn-danger {
    color: #ff4d4f;
    border-color: #ff4d4f;
}

.user-card__actions .btn-danger:hover {
    background: #fff1f0;
}

.user-card__footer {
    width: 100%;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #eee;
}

/**
 * 【深度选择器】
 * 当需要影响子组件样式时使用
 */
.user-card :deep(.child-component) {
    /* 影响子组件的样式 */
}

/**
 * 【插槽选择器】
 * 选择插槽内容
 */
.user-card :slotted(.custom-action) {
    font-weight: bold;
}

/**
 * 【全局选择器】
 * 在 scoped 中创建全局样式
 */
:global(.global-class) {
    /* 全局样式 */
}
</style>
