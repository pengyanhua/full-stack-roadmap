# closures this.js

::: info 文件信息
- 📄 原文件：`02_closures_this.js`
- 🔤 语言：javascript
:::

JavaScript 闭包与 this
本文件介绍 JavaScript 中的闭包和 this 绑定机制。

## 完整代码

```javascript
/**
 * ============================================================
 *              JavaScript 闭包与 this
 * ============================================================
 * 本文件介绍 JavaScript 中的闭包和 this 绑定机制。
 * ============================================================
 */

console.log("=".repeat(60));
console.log("1. 闭包 (Closure)");
console.log("=".repeat(60));

// ============================================================
//                    1. 闭包
// ============================================================

/**
 * 闭包 = 函数 + 词法环境
 * 当函数可以记住并访问其词法作用域时，就产生了闭包
 */

// --- 基本闭包 ---
console.log("\n--- 基本闭包 ---");
function outer() {
    const message = "Hello from outer";  // 自由变量

    function inner() {
        console.log(message);  // 访问外层变量
    }

    return inner;
}

const closure = outer();
closure();  // 即使 outer 已执行完，仍能访问 message

// --- 计数器 ---
console.log("\n--- 计数器闭包 ---");
function createCounter() {
    let count = 0;  // 私有变量

    return {
        increment() { return ++count; },
        decrement() { return --count; },
        getCount() { return count; }
    };
}

const counter = createCounter();
console.log("increment:", counter.increment());
console.log("increment:", counter.increment());
console.log("decrement:", counter.decrement());
console.log("getCount:", counter.getCount());

// count 是私有的，无法直接访问
// console.log(counter.count);  // undefined

// --- 工厂函数 ---
console.log("\n--- 工厂函数 ---");
function createMultiplier(factor) {
    return function(number) {
        return number * factor;
    };
}

const double = createMultiplier(2);
const triple = createMultiplier(3);
console.log("double(5):", double(5));
console.log("triple(5):", triple(5));

// --- 循环中的闭包（经典问题）---
console.log("\n--- 循环中的闭包 ---");

// 问题代码（var）
console.log("使用 var（问题）:");
for (var i = 0; i < 3; i++) {
    setTimeout(() => console.log("  var i:", i), 10);
}
// 输出：3, 3, 3（因为 var 是函数作用域）

// 解决方案1：使用 let
console.log("使用 let（解决）:");
for (let j = 0; j < 3; j++) {
    setTimeout(() => console.log("  let j:", j), 20);
}

// 解决方案2：使用 IIFE
console.log("使用 IIFE（解决）:");
for (var k = 0; k < 3; k++) {
    (function(index) {
        setTimeout(() => console.log("  IIFE k:", index), 30);
    })(k);
}

// --- 模块模式 ---
console.log("\n--- 模块模式 ---");
const module = (function() {
    // 私有变量
    let privateData = 0;

    // 私有函数
    function privateMethod() {
        return "private";
    }

    // 公开接口
    return {
        publicMethod() {
            return privateMethod() + " -> public";
        },
        increment() {
            return ++privateData;
        },
        getData() {
            return privateData;
        }
    };
})();

console.log("publicMethod:", module.publicMethod());
console.log("increment:", module.increment());
console.log("getData:", module.getData());

// --- 闭包的内存考虑 ---
console.log("\n--- 闭包与内存 ---");
console.log(`
【注意】闭包会保持对外部变量的引用

潜在问题：
1. 内存泄漏：大对象被闭包引用无法回收
2. 意外保留：只需要对象的一个属性，却保留了整个对象

解决方案：
1. 只保留需要的值
2. 不再需要时解除引用
3. 使用 WeakMap/WeakSet
`);


console.log("\n" + "=".repeat(60));
console.log("2. this 绑定");
console.log("=".repeat(60));

// ============================================================
//                    2. this 绑定
// ============================================================

/**
 * this 的值取决于函数的调用方式，而不是定义位置
 *
 * 绑定规则（优先级从高到低）：
 * 1. new 绑定
 * 2. 显式绑定（call, apply, bind）
 * 3. 隐式绑定（作为对象方法调用）
 * 4. 默认绑定（独立调用）
 */

// --- 默认绑定 ---
console.log("\n--- 默认绑定 ---");
function showThis() {
    // 严格模式下是 undefined，非严格模式下是全局对象
    console.log("this:", this === globalThis ? "globalThis" : this);
}
showThis();

// --- 隐式绑定 ---
console.log("\n--- 隐式绑定 ---");
const obj = {
    name: "Object",
    greet() {
        console.log(`Hello from ${this.name}`);
    }
};
obj.greet();  // this -> obj

// 隐式绑定丢失
const greetFn = obj.greet;
// greetFn();  // this -> undefined 或 globalThis

// --- 显式绑定 ---
console.log("\n--- 显式绑定 ---");
function introduce(greeting, punctuation) {
    console.log(`${greeting}, I'm ${this.name}${punctuation}`);
}

const person = { name: "Alice" };

// call：立即调用，参数逐个传递
introduce.call(person, "Hello", "!");

// apply：立即调用，参数以数组传递
introduce.apply(person, ["Hi", "."]);

// bind：返回新函数，不立即调用
const boundIntroduce = introduce.bind(person, "Hey");
boundIntroduce("?");

// --- new 绑定 ---
console.log("\n--- new 绑定 ---");
function Person(name) {
    this.name = name;
    console.log("构造函数中的 this:", this);
}

const alice = new Person("Alice");
console.log("实例:", alice);

// --- 箭头函数的 this ---
console.log("\n--- 箭头函数的 this ---");
const arrowObj = {
    name: "Arrow Object",

    // 普通方法
    regularMethod() {
        console.log("普通方法 this.name:", this.name);

        // 内部函数问题
        // function inner() {
        //     console.log(this.name);  // undefined
        // }
        // inner();

        // 箭头函数解决方案
        const inner = () => {
            console.log("箭头函数 this.name:", this.name);
        };
        inner();
    },

    // 箭头函数方法（不推荐）
    arrowMethod: () => {
        console.log("箭头方法 this:", this);  // 不是对象！
    }
};

arrowObj.regularMethod();
arrowObj.arrowMethod();

// --- this 在回调中 ---
console.log("\n--- 回调中的 this ---");
const timer = {
    seconds: 0,

    // 问题：setTimeout 中的 this
    startBroken() {
        setTimeout(function() {
            // this.seconds++;  // undefined.seconds++
        }, 100);
    },

    // 解决方案1：保存 this
    startWithThat() {
        const that = this;
        setTimeout(function() {
            that.seconds++;
            console.log("方案1 - seconds:", that.seconds);
        }, 100);
    },

    // 解决方案2：箭头函数（推荐）
    startWithArrow() {
        setTimeout(() => {
            this.seconds++;
            console.log("方案2 - seconds:", this.seconds);
        }, 200);
    },

    // 解决方案3：bind
    startWithBind() {
        setTimeout(function() {
            this.seconds++;
            console.log("方案3 - seconds:", this.seconds);
        }.bind(this), 300);
    }
};

timer.startWithThat();
timer.startWithArrow();
timer.startWithBind();


console.log("\n" + "=".repeat(60));
console.log("3. 类中的 this");
console.log("=".repeat(60));

// ============================================================
//                    3. 类中的 this
// ============================================================

class Button {
    constructor(label) {
        this.label = label;

        // 方案1：在构造函数中绑定
        this.handleClickBound = this.handleClick.bind(this);
    }

    // 普通方法：this 可能丢失
    handleClick() {
        console.log(`${this.label} clicked`);
    }

    // 方案2：箭头函数属性（推荐）
    handleClickArrow = () => {
        console.log(`${this.label} clicked (arrow)`);
    }

    // 模拟事件监听
    attachTo(element) {
        // 问题：直接传递方法
        // element.addEventListener('click', this.handleClick);

        // 解决方案
        // element.addEventListener('click', this.handleClickBound);
        // element.addEventListener('click', this.handleClickArrow);
        // element.addEventListener('click', () => this.handleClick());
    }
}

const btn = new Button("Submit");
btn.handleClick();  // 正常
btn.handleClickArrow();  // 正常

const detached = btn.handleClick;
// detached();  // 错误：this 丢失

const detachedArrow = btn.handleClickArrow;
detachedArrow();  // 正常：箭头函数保持 this


console.log("\n" + "=".repeat(60));
console.log("4. 实际应用场景");
console.log("=".repeat(60));

// ============================================================
//                    4. 实际应用场景
// ============================================================

// --- 缓存函数 ---
console.log("\n--- 缓存函数 ---");
function createCache() {
    const cache = new Map();

    return {
        get(key) {
            return cache.get(key);
        },
        set(key, value, ttl) {
            cache.set(key, { value, expires: Date.now() + ttl });

            // 自动清理（闭包保持对 cache 的引用）
            setTimeout(() => cache.delete(key), ttl);

            return value;
        },
        has(key) {
            const entry = cache.get(key);
            if (!entry) return false;
            if (Date.now() > entry.expires) {
                cache.delete(key);
                return false;
            }
            return true;
        }
    };
}

const cache = createCache();
cache.set("user", { name: "Alice" }, 1000);
console.log("缓存命中:", cache.has("user"));

// --- 事件发射器 ---
console.log("\n--- 事件发射器 ---");
function createEventEmitter() {
    const events = {};

    return {
        on(event, handler) {
            if (!events[event]) {
                events[event] = [];
            }
            events[event].push(handler);

            // 返回取消订阅函数（闭包）
            return () => {
                events[event] = events[event].filter(h => h !== handler);
            };
        },

        emit(event, data) {
            if (events[event]) {
                events[event].forEach(handler => handler(data));
            }
        }
    };
}

const emitter = createEventEmitter();
const unsubscribe = emitter.on("message", data => {
    console.log("收到消息:", data);
});

emitter.emit("message", "Hello!");
unsubscribe();  // 取消订阅
emitter.emit("message", "这条不会收到");

// --- 函数节流（带 this）---
console.log("\n--- 函数节流 ---");
function throttle(fn, limit) {
    let lastCall = 0;

    return function(...args) {
        const now = Date.now();
        if (now - lastCall >= limit) {
            lastCall = now;
            return fn.apply(this, args);  // 保持 this
        }
    };
}

const api = {
    name: "API",
    fetch: throttle(function() {
        console.log(`${this.name}: 发送请求`);
    }, 100)
};

api.fetch();
api.fetch();  // 被节流
setTimeout(() => api.fetch(), 150);  // 正常执行


console.log("\n【总结】");
console.log(`
闭包：
- 函数可以记住创建时的词法作用域
- 用于创建私有变量、工厂函数、模块模式
- 注意循环中的闭包问题（使用 let 或 IIFE）
- 注意内存泄漏

this 绑定：
- 默认绑定：独立调用 -> undefined/globalThis
- 隐式绑定：obj.method() -> obj
- 显式绑定：call/apply/bind
- new 绑定：new Constructor() -> 新对象

箭头函数：
- 没有自己的 this，继承外层
- 不能用 new 调用
- 适合回调函数和需要保持 this 的场景
`);


// 等待异步输出
setTimeout(() => {
    console.log("\n--- 异步输出结束 ---");
}, 500);
```
