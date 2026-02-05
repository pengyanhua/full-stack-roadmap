<!--
============================================================
                    Vue 3 模板语法基础
============================================================
Vue 使用基于 HTML 的模板语法，可以声明式地将 DOM 绑定到组件实例的数据。
所有 Vue 模板都是语法有效的 HTML。
============================================================
-->

<template>
    <div class="tutorial">
        <h1>Vue 3 模板语法教程</h1>

        <!-- ============================================================
                              1. 文本插值
             ============================================================ -->

        <!--
        【文本插值】
        使用双花括号 {{ }} 进行文本插值
        - 会自动转义 HTML（防止 XSS）
        - 支持 JavaScript 表达式
        - 响应式：数据变化时自动更新
        -->

        <section>
            <h2>1. 文本插值</h2>

            <!-- 基本用法 -->
            <p>消息: {{ message }}</p>

            <!-- 表达式 -->
            <p>计算: {{ count + 1 }}</p>
            <p>三元: {{ isActive ? '激活' : '未激活' }}</p>
            <p>方法: {{ message.toUpperCase() }}</p>
            <p>模板: {{ `你好, ${name}!` }}</p>

            <!-- 注意：只能是单个表达式，不能是语句 -->
            <!-- ❌ 错误：{{ let x = 1 }} -->
            <!-- ❌ 错误：{{ if (ok) { return 'yes' } }} -->
        </section>


        <!-- ============================================================
                              2. 原始 HTML
             ============================================================ -->

        <!--
        【v-html 指令】
        当需要输出真正的 HTML 时使用 v-html
        ⚠️ 警告：只对可信内容使用，避免 XSS 攻击
        -->

        <section>
            <h2>2. 原始 HTML</h2>

            <!-- 自动转义 -->
            <p>转义输出: {{ rawHtml }}</p>

            <!-- 原始 HTML -->
            <p>HTML 输出: <span v-html="rawHtml"></span></p>
        </section>


        <!-- ============================================================
                              3. 属性绑定
             ============================================================ -->

        <!--
        【v-bind 指令】
        用于动态绑定 HTML 属性
        - 完整语法：v-bind:attr="value"
        - 简写语法：:attr="value"
        -->

        <section>
            <h2>3. 属性绑定</h2>

            <!-- 动态 id -->
            <div v-bind:id="dynamicId">动态 ID</div>

            <!-- 简写形式（推荐） -->
            <div :id="dynamicId">简写形式</div>

            <!-- 动态 class -->
            <div :class="className">动态类名</div>

            <!-- 动态 src -->
            <img :src="imageUrl" :alt="imageAlt" />

            <!-- 动态 disabled -->
            <button :disabled="isDisabled">
                {{ isDisabled ? '禁用' : '可用' }}
            </button>

            <!-- 布尔属性：值为 falsy 时，属性会被移除 -->
            <input :disabled="false" />  <!-- 不会有 disabled 属性 -->

            <!-- 动态绑定多个属性 -->
            <div v-bind="objectOfAttrs">多属性绑定</div>
        </section>


        <!-- ============================================================
                              4. Class 绑定
             ============================================================ -->

        <!--
        【Class 绑定】
        Vue 对 class 绑定做了增强，支持：
        - 对象语法
        - 数组语法
        - 混合使用
        -->

        <section>
            <h2>4. Class 绑定</h2>

            <!-- 对象语法：根据条件添加类 -->
            <div :class="{ active: isActive, 'text-danger': hasError }">
                对象语法
            </div>

            <!-- 绑定计算属性 -->
            <div :class="classObject">计算属性</div>

            <!-- 数组语法：绑定多个类 -->
            <div :class="[activeClass, errorClass]">数组语法</div>

            <!-- 数组中使用三元表达式 -->
            <div :class="[isActive ? 'active' : '', errorClass]">
                数组 + 三元
            </div>

            <!-- 数组中使用对象 -->
            <div :class="[{ active: isActive }, errorClass]">
                数组 + 对象
            </div>

            <!-- 与普通 class 共存 -->
            <div class="static-class" :class="{ active: isActive }">
                静态 + 动态
            </div>
        </section>


        <!-- ============================================================
                              5. Style 绑定
             ============================================================ -->

        <!--
        【Style 绑定】
        支持对象语法和数组语法
        CSS 属性名使用 camelCase 或 kebab-case（需要引号）
        -->

        <section>
            <h2>5. Style 绑定</h2>

            <!-- 对象语法 -->
            <div :style="{ color: activeColor, fontSize: fontSize + 'px' }">
                对象语法
            </div>

            <!-- kebab-case 属性名 -->
            <div :style="{ 'font-size': fontSize + 'px' }">
                kebab-case
            </div>

            <!-- 绑定样式对象 -->
            <div :style="styleObject">样式对象</div>

            <!-- 数组语法：合并多个样式对象 -->
            <div :style="[baseStyles, overridingStyles]">
                数组语法
            </div>

            <!-- 自动前缀 -->
            <!-- Vue 会自动添加浏览器前缀 -->
            <div :style="{ display: 'flex' }">自动前缀</div>
        </section>


        <!-- ============================================================
                              6. 条件渲染
             ============================================================ -->

        <!--
        【v-if / v-else-if / v-else】
        条件渲染：根据条件决定是否渲染元素
        - 真正的条件渲染：切换时会销毁和重建元素
        - 适合不经常切换的场景
        -->

        <section>
            <h2>6. 条件渲染</h2>

            <!-- v-if -->
            <p v-if="type === 'A'">类型 A</p>
            <p v-else-if="type === 'B'">类型 B</p>
            <p v-else>其他类型</p>

            <!-- 在 template 上使用（不会渲染额外元素） -->
            <template v-if="showGroup">
                <h3>标题</h3>
                <p>段落 1</p>
                <p>段落 2</p>
            </template>

            <!--
            【v-show】
            通过 CSS display 控制显示/隐藏
            - 元素始终存在于 DOM 中
            - 适合频繁切换的场景
            -->
            <p v-show="isVisible">v-show 元素</p>

            <!--
            【v-if vs v-show】
            v-if：更高的切换开销（销毁/创建）
            v-show：更高的初始渲染开销（始终渲染）

            频繁切换用 v-show，运行时条件很少改变用 v-if
            -->
        </section>


        <!-- ============================================================
                              7. 列表渲染
             ============================================================ -->

        <!--
        【v-for 指令】
        用于遍历数组或对象渲染列表
        - 语法：v-for="item in items"
        - 必须提供 key 属性
        -->

        <section>
            <h2>7. 列表渲染</h2>

            <!-- 遍历数组 -->
            <ul>
                <li v-for="item in items" :key="item.id">
                    {{ item.name }}
                </li>
            </ul>

            <!-- 带索引 -->
            <ul>
                <li v-for="(item, index) in items" :key="item.id">
                    {{ index + 1 }}. {{ item.name }}
                </li>
            </ul>

            <!-- 遍历对象 -->
            <ul>
                <li v-for="(value, key) in userInfo" :key="key">
                    {{ key }}: {{ value }}
                </li>
            </ul>

            <!-- 遍历对象（带索引） -->
            <ul>
                <li v-for="(value, key, index) in userInfo" :key="key">
                    {{ index }}. {{ key }}: {{ value }}
                </li>
            </ul>

            <!-- 遍历数字范围 -->
            <span v-for="n in 5" :key="n">{{ n }} </span>

            <!-- 在 template 上使用 -->
            <ul>
                <template v-for="item in items" :key="item.id">
                    <li>{{ item.name }}</li>
                    <li class="divider"></li>
                </template>
            </ul>

            <!--
            【key 的作用】
            - 帮助 Vue 识别节点
            - 用于 diff 算法优化
            - 必须是唯一的字符串或数字
            - 不推荐用 index 作为 key（除非列表不会变）
            -->
        </section>


        <!-- ============================================================
                              8. 事件处理
             ============================================================ -->

        <!--
        【v-on 指令】
        用于监听 DOM 事件
        - 完整语法：v-on:event="handler"
        - 简写语法：@event="handler"
        -->

        <section>
            <h2>8. 事件处理</h2>

            <!-- 内联语句 -->
            <button @click="count++">点击次数: {{ count }}</button>

            <!-- 方法处理器 -->
            <button @click="handleClick">方法处理</button>

            <!-- 传递参数 -->
            <button @click="handleClickWithArg('hello')">传递参数</button>

            <!-- 访问原始事件对象 -->
            <button @click="handleClickWithEvent($event)">获取事件</button>

            <!-- 多事件处理器 -->
            <button @click="handleA(), handleB()">多处理器</button>

            <!--
            【事件修饰符】
            -->
            <!-- .stop - 阻止冒泡 -->
            <div @click="handleOuter">
                <button @click.stop="handleInner">阻止冒泡</button>
            </div>

            <!-- .prevent - 阻止默认行为 -->
            <form @submit.prevent="handleSubmit">
                <button type="submit">提交</button>
            </form>

            <!-- .self - 只有事件源是自身时触发 -->
            <div @click.self="handleSelf">
                <button>点击按钮不触发</button>
            </div>

            <!-- .once - 只触发一次 -->
            <button @click.once="handleOnce">只能点一次</button>

            <!-- .capture - 使用捕获模式 -->
            <div @click.capture="handleCapture">捕获</div>

            <!-- .passive - 不阻止默认行为（提升滚动性能） -->
            <div @scroll.passive="handleScroll">滚动</div>

            <!-- 修饰符链 -->
            <a @click.stop.prevent="handleLink">链接</a>

            <!--
            【按键修饰符】
            -->
            <!-- 监听特定按键 -->
            <input @keyup.enter="submitForm" placeholder="按回车提交" />
            <input @keyup.esc="clearInput" placeholder="按 ESC 清空" />

            <!-- 组合键 -->
            <input @keyup.ctrl.enter="send" placeholder="Ctrl+Enter 发送" />
            <input @keyup.alt.s="save" placeholder="Alt+S 保存" />

            <!-- 精确修饰符：只有指定的键按下时触发 -->
            <input @keyup.ctrl.exact="onlyCtrl" placeholder="只按 Ctrl" />

            <!--
            【鼠标按键修饰符】
            -->
            <button @click.left="handleLeft">左键</button>
            <button @click.right="handleRight">右键</button>
            <button @click.middle="handleMiddle">中键</button>
        </section>


        <!-- ============================================================
                              9. 表单输入绑定
             ============================================================ -->

        <!--
        【v-model 指令】
        实现表单元素的双向数据绑定
        本质是 :value + @input 的语法糖
        -->

        <section>
            <h2>9. 表单绑定</h2>

            <!-- 文本输入 -->
            <div>
                <input v-model="textInput" placeholder="文本输入" />
                <p>输入值: {{ textInput }}</p>
            </div>

            <!-- 多行文本 -->
            <div>
                <textarea v-model="textareaInput" placeholder="多行文本"></textarea>
                <p>内容: {{ textareaInput }}</p>
            </div>

            <!-- 复选框 - 单个（布尔值） -->
            <div>
                <input type="checkbox" v-model="checked" id="checkbox" />
                <label for="checkbox">同意条款: {{ checked }}</label>
            </div>

            <!-- 复选框 - 多个（数组） -->
            <div>
                <input type="checkbox" v-model="checkedNames" value="Jack" id="jack" />
                <label for="jack">Jack</label>
                <input type="checkbox" v-model="checkedNames" value="John" id="john" />
                <label for="john">John</label>
                <input type="checkbox" v-model="checkedNames" value="Mike" id="mike" />
                <label for="mike">Mike</label>
                <p>选中: {{ checkedNames }}</p>
            </div>

            <!-- 单选按钮 -->
            <div>
                <input type="radio" v-model="picked" value="One" id="one" />
                <label for="one">One</label>
                <input type="radio" v-model="picked" value="Two" id="two" />
                <label for="two">Two</label>
                <p>选中: {{ picked }}</p>
            </div>

            <!-- 下拉选择 -->
            <div>
                <select v-model="selected">
                    <option disabled value="">请选择</option>
                    <option value="A">选项 A</option>
                    <option value="B">选项 B</option>
                    <option value="C">选项 C</option>
                </select>
                <p>选中: {{ selected }}</p>
            </div>

            <!-- 多选下拉 -->
            <div>
                <select v-model="multiSelected" multiple>
                    <option value="A">选项 A</option>
                    <option value="B">选项 B</option>
                    <option value="C">选项 C</option>
                </select>
                <p>选中: {{ multiSelected }}</p>
            </div>

            <!--
            【v-model 修饰符】
            -->
            <!-- .lazy - 在 change 事件后更新（而非 input） -->
            <input v-model.lazy="lazyValue" placeholder=".lazy" />

            <!-- .number - 自动转为数字 -->
            <input v-model.number="age" type="number" placeholder=".number" />

            <!-- .trim - 自动去除首尾空格 -->
            <input v-model.trim="trimValue" placeholder=".trim" />
        </section>
    </div>
</template>

<script setup>
/**
 * ============================================================
 *                    Script Setup 语法
 * ============================================================
 * <script setup> 是 Vue 3 的语法糖：
 * - 更简洁的代码
 * - 更好的运行时性能
 * - 更好的 IDE 类型支持
 * ============================================================
 */

import { ref, reactive, computed } from 'vue';

// ============================================================
//                    响应式数据
// ============================================================

// --- ref: 用于基本类型 ---
const message = ref('Hello Vue!');
const count = ref(0);
const name = ref('Alice');
const isActive = ref(true);
const hasError = ref(false);
const isDisabled = ref(false);
const isVisible = ref(true);
const showGroup = ref(true);
const type = ref('A');

// --- reactive: 用于对象 ---
const userInfo = reactive({
    name: 'Alice',
    age: 25,
    email: 'alice@example.com',
});

// --- 数组 ---
const items = ref([
    { id: 1, name: '苹果' },
    { id: 2, name: '香蕉' },
    { id: 3, name: '橙子' },
]);

// --- 属性绑定相关 ---
const dynamicId = ref('my-element');
const className = ref('container');
const imageUrl = ref('/images/logo.png');
const imageAlt = ref('Logo');

// 绑定多个属性
const objectOfAttrs = reactive({
    id: 'container',
    class: 'wrapper',
    'data-test': 'value',
});

// --- Class 绑定相关 ---
const activeClass = ref('active');
const errorClass = ref('text-danger');

// 计算属性返回 class 对象
const classObject = computed(() => ({
    active: isActive.value,
    'text-danger': hasError.value,
}));

// --- Style 绑定相关 ---
const activeColor = ref('red');
const fontSize = ref(16);

const styleObject = reactive({
    color: 'blue',
    fontSize: '18px',
    fontWeight: 'bold',
});

const baseStyles = reactive({
    color: 'green',
});

const overridingStyles = reactive({
    fontSize: '20px',
});

// --- HTML 相关 ---
const rawHtml = ref('<span style="color: red;">红色文字</span>');

// --- 表单相关 ---
const textInput = ref('');
const textareaInput = ref('');
const checked = ref(false);
const checkedNames = ref([]);
const picked = ref('');
const selected = ref('');
const multiSelected = ref([]);
const lazyValue = ref('');
const age = ref(18);
const trimValue = ref('');


// ============================================================
//                    事件处理函数
// ============================================================

/**
 * 基本点击处理
 */
function handleClick() {
    console.log('按钮被点击');
    message.value = '按钮被点击了!';
}

/**
 * 带参数的点击处理
 * @param {string} msg - 传入的消息
 */
function handleClickWithArg(msg) {
    console.log('收到参数:', msg);
}

/**
 * 带事件对象的处理
 * @param {Event} event - 原始事件对象
 */
function handleClickWithEvent(event) {
    console.log('事件类型:', event.type);
    console.log('目标元素:', event.target);
}

// 多处理器
function handleA() {
    console.log('处理器 A');
}

function handleB() {
    console.log('处理器 B');
}

// 事件修饰符相关
function handleOuter() {
    console.log('外层点击');
}

function handleInner() {
    console.log('内层点击');
}

function handleSubmit() {
    console.log('表单提交');
}

function handleSelf() {
    console.log('只有点击自身才触发');
}

function handleOnce() {
    console.log('只触发一次');
}

function handleCapture() {
    console.log('捕获阶段触发');
}

function handleScroll() {
    console.log('滚动');
}

function handleLink() {
    console.log('链接点击');
}

// 按键相关
function submitForm() {
    console.log('提交表单');
}

function clearInput() {
    textInput.value = '';
}

function send() {
    console.log('发送');
}

function save() {
    console.log('保存');
}

function onlyCtrl() {
    console.log('只按了 Ctrl');
}

// 鼠标按键
function handleLeft() {
    console.log('左键点击');
}

function handleRight() {
    console.log('右键点击');
}

function handleMiddle() {
    console.log('中键点击');
}
</script>

<style scoped>
/*
【scoped 样式】
scoped 属性使样式只作用于当前组件
*/

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
    border-bottom: 2px solid #42b883;
    padding-bottom: 10px;
}

/* Class 绑定示例样式 */
.active {
    color: green;
    font-weight: bold;
}

.text-danger {
    color: red;
}

.static-class {
    padding: 10px;
    background: #f5f5f5;
}

/* 其他样式 */
button {
    margin: 5px;
    padding: 8px 16px;
    cursor: pointer;
}

input, select, textarea {
    margin: 5px;
    padding: 8px;
}

.divider {
    height: 1px;
    background: #eee;
    list-style: none;
}
</style>
