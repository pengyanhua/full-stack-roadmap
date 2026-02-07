# closures.js

::: info 文件信息
- 📄 原文件：`02_closures.js`
- 🔤 语言：javascript
:::

JavaScript 闭包
本文件介绍 JavaScript 中的闭包概念和应用。

## 完整代码

```javascript
/**
 * ============================================================
 *                JavaScript 闭包
 * ============================================================
 * 本文件介绍 JavaScript 中的闭包概念和应用。
 * ============================================================
 */

console.log("=".repeat(60));
console.log("1. 闭包基础");
console.log("=".repeat(60));

// ============================================================
//                    1. 闭包基础
// ============================================================

/**
 * 【什么是闭包】
 *
 * 闭包是指函数能够访问其定义时所在的词法作用域中的变量，
 * 即使该函数在其词法作用域之外执行。
 *
 * 简单说：函数 + 其能访问的外部变量 = 闭包
 */

console.log("\n--- 基本闭包 ---");
function outer() {
    const message = "Hello from outer";

    function inner() {
        console.log(message);  // 访问外部变量
    }

    return inner;
}

const fn = outer();
fn();  // 即使 outer 已执行完，inner 仍能访问 message

// --- 闭包保持引用 ---
console.log("\n--- 闭包保持引用 ---");
function createCounter() {
    let count = 0;  // 私有变量

    return {
        increment() {
            count++;
            return count;
        },
        decrement() {
            count--;
            return count;
        },
        getCount() {
            return count;
        }
    };
}

const counter = createCounter();
console.log("increment:", counter.increment());
console.log("increment:", counter.increment());
console.log("decrement:", counter.decrement());
console.log("getCount:", counter.getCount());
// console.log(count);  // 错误：count 是私有的

// 每次调用创建新的闭包
const counter2 = createCounter();
console.log("counter2:", counter2.getCount());  // 0（独立的计数器）


console.log("\n" + "=".repeat(60));
console.log("2. 闭包的应用");
console.log("=".repeat(60));

// ============================================================
//                    2. 闭包的应用
// ============================================================

// --- 数据封装（模块模式）---
console.log("\n--- 数据封装 ---");
const bankAccount = (function() {
    let balance = 0;  // 私有变量

    return {
        deposit(amount) {
            if (amount > 0) {
                balance += amount;
                return `存入 ${amount}，余额 ${balance}`;
            }
        },
        withdraw(amount) {
            if (amount > 0 && amount <= balance) {
                balance -= amount;
                return `取出 ${amount}，余额 ${balance}`;
            }
            return "余额不足";
        },
        getBalance() {
            return balance;
        }
    };
})();

console.log(bankAccount.deposit(100));
console.log(bankAccount.withdraw(30));
console.log("余额:", bankAccount.getBalance());
// bankAccount.balance = 1000000;  // 无法直接修改

// --- 函数工厂 ---
console.log("\n--- 函数工厂 ---");
function createMultiplier(factor) {
    return function(x) {
        return x * factor;
    };
}

const double = createMultiplier(2);
const triple = createMultiplier(3);
const quadruple = createMultiplier(4);

console.log("double(5):", double(5));
console.log("triple(5):", triple(5));
console.log("quadruple(5):", quadruple(5));

// --- 缓存/记忆化 ---
console.log("\n--- 记忆化 ---");
function memoize(fn) {
    const cache = new Map();

    return function(...args) {
        const key = JSON.stringify(args);

        if (cache.has(key)) {
            console.log(`  缓存命中: ${key}`);
            return cache.get(key);
        }

        console.log(`  计算: ${key}`);
        const result = fn.apply(this, args);
        cache.set(key, result);
        return result;
    };
}

const expensiveCalc = memoize(function(n) {
    // 模拟耗时计算
    let result = 0;
    for (let i = 0; i < n * 1000000; i++) {
        result += i;
    }
    return result;
});

console.log("第一次:", expensiveCalc(10));
console.log("第二次:", expensiveCalc(10));  // 从缓存获取
console.log("不同参数:", expensiveCalc(5));

// --- 偏函数应用 ---
console.log("\n--- 偏函数 ---");
function partial(fn, ...presetArgs) {
    return function(...laterArgs) {
        return fn(...presetArgs, ...laterArgs);
    };
}

function greet(greeting, name, punctuation) {
    return `${greeting}, ${name}${punctuation}`;
}

const sayHello = partial(greet, "Hello");
const sayHelloToAlice = partial(greet, "Hello", "Alice");

console.log(sayHello("Bob", "!"));
console.log(sayHelloToAlice("?"));

// --- 柯里化 ---
console.log("\n--- 柯里化 ---");
function curry(fn) {
    return function curried(...args) {
        if (args.length >= fn.length) {
            return fn.apply(this, args);
        }
        return function(...moreArgs) {
            return curried.apply(this, args.concat(moreArgs));
        };
    };
}

function add3(a, b, c) {
    return a + b + c;
}

const curriedAdd = curry(add3);
console.log("一次传所有:", curriedAdd(1, 2, 3));
console.log("逐个传:", curriedAdd(1)(2)(3));
console.log("分批传:", curriedAdd(1, 2)(3));


console.log("\n" + "=".repeat(60));
console.log("3. 闭包陷阱");
console.log("=".repeat(60));

// ============================================================
//                    3. 闭包陷阱
// ============================================================

// --- 循环中的闭包问题 ---
console.log("\n--- 循环闭包问题 ---");

// 问题示例
console.log("使用 var（有问题）:");
const functionsWithVar = [];
for (var i = 0; i < 3; i++) {
    functionsWithVar.push(function() {
        return i;
    });
}
// 所有函数都返回 3，因为它们共享同一个 i
console.log("结果:", functionsWithVar.map(f => f()));

// 解决方案 1：使用 let
console.log("\n使用 let（正确）:");
const functionsWithLet = [];
for (let j = 0; j < 3; j++) {
    functionsWithLet.push(function() {
        return j;
    });
}
console.log("结果:", functionsWithLet.map(f => f()));

// 解决方案 2：使用 IIFE
console.log("\n使用 IIFE（正确）:");
const functionsWithIIFE = [];
for (var k = 0; k < 3; k++) {
    functionsWithIIFE.push((function(index) {
        return function() {
            return index;
        };
    })(k));
}
console.log("结果:", functionsWithIIFE.map(f => f()));

// --- 内存泄漏 ---
console.log("\n--- 避免内存泄漏 ---");
console.log(`
闭包可能导致内存泄漏：
- 闭包持有外部变量的引用
- 如果闭包长期存在，外部变量无法被垃圾回收

最佳实践：
- 不再需要时，将闭包设为 null
- 避免在闭包中保存大量数据
- 使用 WeakMap/WeakSet 存储对象引用
`);

// 示例：正确清理
function createHeavyObject() {
    const largeData = new Array(1000000).fill("data");

    return {
        getData() {
            return largeData.length;
        },
        cleanup() {
            // 如果可能，提供清理方法
            // largeData = null;  // 但闭包变量不能重新赋值
        }
    };
}


console.log("\n" + "=".repeat(60));
console.log("4. this 与闭包");
console.log("=".repeat(60));

// ============================================================
//                    4. this 与闭包
// ============================================================

console.log("\n--- this 问题 ---");
const obj = {
    name: "Object",
    regularMethod() {
        console.log("普通方法 this:", this.name);

        // 问题：嵌套函数的 this 不是 obj
        function inner() {
            console.log("嵌套函数 this:", this?.name);  // undefined
        }
        inner();
    }
};
obj.regularMethod();

console.log("\n--- 解决方案 ---");

// 方案 1：保存 this
const obj1 = {
    name: "Object1",
    method() {
        const self = this;
        function inner() {
            console.log("使用 self:", self.name);
        }
        inner();
    }
};
obj1.method();

// 方案 2：使用箭头函数
const obj2 = {
    name: "Object2",
    method() {
        const inner = () => {
            console.log("箭头函数:", this.name);
        };
        inner();
    }
};
obj2.method();

// 方案 3：使用 bind
const obj3 = {
    name: "Object3",
    method() {
        const inner = function() {
            console.log("使用 bind:", this.name);
        }.bind(this);
        inner();
    }
};
obj3.method();


console.log("\n" + "=".repeat(60));
console.log("5. 实际应用案例");
console.log("=".repeat(60));

// ============================================================
//                    5. 实际应用案例
// ============================================================

// --- 事件处理 ---
console.log("\n--- 事件处理（模拟）---");
function createButtonHandler(buttonId) {
    let clickCount = 0;

    return function handleClick() {
        clickCount++;
        console.log(`按钮 ${buttonId} 被点击了 ${clickCount} 次`);
    };
}

const button1Handler = createButtonHandler("btn1");
const button2Handler = createButtonHandler("btn2");

button1Handler();
button1Handler();
button2Handler();

// --- 定时器 ---
console.log("\n--- 定时器（模拟）---");
function createCountdown(from, callback) {
    let count = from;

    function tick() {
        console.log(`倒计时: ${count}`);
        count--;

        if (count >= 0) {
            setTimeout(tick, 100);  // 使用较短时间演示
        } else {
            callback();
        }
    }

    return tick;
}

// 模拟而不实际执行
console.log("（倒计时函数已创建）");

// --- 状态机 ---
console.log("\n--- 状态机 ---");
function createStateMachine(initialState, transitions) {
    let currentState = initialState;

    return {
        getState() {
            return currentState;
        },
        transition(action) {
            const nextState = transitions[currentState]?.[action];
            if (nextState) {
                console.log(`${currentState} -> ${action} -> ${nextState}`);
                currentState = nextState;
                return true;
            }
            console.log(`无效转换: ${currentState} -> ${action}`);
            return false;
        }
    };
}

const trafficLight = createStateMachine("red", {
    red: { next: "green" },
    green: { next: "yellow" },
    yellow: { next: "red" }
});

console.log("初始状态:", trafficLight.getState());
trafficLight.transition("next");
trafficLight.transition("next");
trafficLight.transition("next");

// --- 函数组合 ---
console.log("\n--- 函数组合 ---");
function compose(...fns) {
    return function(x) {
        return fns.reduceRight((acc, fn) => fn(acc), x);
    };
}

const addOne = x => x + 1;
const doubleIt = x => x * 2;
const square = x => x * x;

const composed = compose(addOne, doubleIt, square);
console.log("compose(addOne, double, square)(3):", composed(3));
// 等同于 addOne(doubleIt(square(3))) = addOne(doubleIt(9)) = addOne(18) = 19


console.log("\n【总结】");
console.log(`
闭包核心概念：
- 函数 + 词法作用域 = 闭包
- 内部函数可以访问外部函数的变量
- 变量的生命周期被延长

常见应用：
- 数据封装/私有变量
- 函数工厂
- 记忆化/缓存
- 偏函数和柯里化
- 模块模式

注意事项：
- 循环中使用 let 代替 var
- 注意内存泄漏
- 箭头函数没有自己的 this
`);
```
