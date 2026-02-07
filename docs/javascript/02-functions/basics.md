# basics.js

::: info 文件信息
- 📄 原文件：`01_basics.js`
- 🔤 语言：javascript
:::

============================================================
               JavaScript 函数基础
============================================================
本文件介绍 JavaScript 中的函数定义和调用。
============================================================

## 完整代码

```javascript
/**
 * ============================================================
 *                JavaScript 函数基础
 * ============================================================
 * 本文件介绍 JavaScript 中的函数定义和调用。
 * ============================================================
 */

console.log("=".repeat(60));
console.log("1. 函数定义方式");
console.log("=".repeat(60));

// ============================================================
//                    1. 函数定义方式
// ============================================================

// --- 函数声明 ---
console.log("\n--- 函数声明 ---");
function greet(name) {
    return `Hello, ${name}!`;
}
console.log(greet("Alice"));

// 函数声明会提升
console.log("提升:", hoistedFn());
function hoistedFn() {
    return "我被提升了";
}

// --- 函数表达式 ---
console.log("\n--- 函数表达式 ---");
const add = function(a, b) {
    return a + b;
};
console.log("add(3, 5):", add(3, 5));

// 命名函数表达式（用于递归和调试）
const factorial = function fact(n) {
    return n <= 1 ? 1 : n * fact(n - 1);
};
console.log("factorial(5):", factorial(5));

// --- 箭头函数 ---
console.log("\n--- 箭头函数 ---");

// 完整写法
const multiply = (a, b) => {
    return a * b;
};

// 单表达式简写
const double = x => x * 2;

// 返回对象需要括号
const createPerson = (name, age) => ({ name, age });

console.log("multiply(3, 4):", multiply(3, 4));
console.log("double(5):", double(5));
console.log("createPerson:", createPerson("Alice", 25));

// --- 箭头函数与普通函数的区别 ---
console.log("\n--- 箭头函数特点 ---");
console.log(`
1. 没有自己的 this（继承外层）
2. 没有 arguments 对象
3. 不能用作构造函数（不能 new）
4. 没有 prototype 属性
5. 不能用作生成器（不能使用 yield）
`);

// --- IIFE（立即执行函数）---
console.log("--- IIFE ---");
const result = (function(x) {
    return x * x;
})(5);
console.log("IIFE 结果:", result);

// 箭头函数 IIFE
const result2 = ((x) => x * x)(6);
console.log("箭头 IIFE:", result2);


console.log("\n" + "=".repeat(60));
console.log("2. 参数");
console.log("=".repeat(60));

// ============================================================
//                    2. 参数
// ============================================================

// --- 默认参数 ---
console.log("\n--- 默认参数 ---");
function greetWithDefault(name = "Guest", greeting = "Hello") {
    return `${greeting}, ${name}!`;
}
console.log(greetWithDefault());
console.log(greetWithDefault("Alice"));
console.log(greetWithDefault("Bob", "Hi"));

// 默认参数可以使用前面的参数
function createRect(width, height = width) {
    return { width, height };
}
console.log("正方形:", createRect(5));
console.log("矩形:", createRect(4, 3));

// --- 剩余参数 ---
console.log("\n--- 剩余参数 ---");
function sum(...numbers) {
    return numbers.reduce((acc, n) => acc + n, 0);
}
console.log("sum(1, 2, 3, 4, 5):", sum(1, 2, 3, 4, 5));

function firstAndRest(first, ...rest) {
    console.log("first:", first);
    console.log("rest:", rest);
}
firstAndRest(1, 2, 3, 4, 5);

// --- arguments 对象（仅普通函数）---
console.log("\n--- arguments 对象 ---");
function showArgs() {
    console.log("arguments:", Array.from(arguments));
    console.log("长度:", arguments.length);
}
showArgs(1, 2, 3);

// --- 参数解构 ---
console.log("\n--- 参数解构 ---");
function printUser({ name, age, city = "Unknown" }) {
    console.log(`${name}, ${age}岁, 来自${city}`);
}
printUser({ name: "Alice", age: 25 });
printUser({ name: "Bob", age: 30, city: "Beijing" });

// 数组参数解构
function printCoords([x, y]) {
    console.log(`坐标: (${x}, ${y})`);
}
printCoords([10, 20]);


console.log("\n" + "=".repeat(60));
console.log("3. 返回值");
console.log("=".repeat(60));

// ============================================================
//                    3. 返回值
// ============================================================

// --- 基本返回 ---
console.log("\n--- 基本返回 ---");
function getMax(a, b) {
    return a > b ? a : b;
}
console.log("getMax(3, 7):", getMax(3, 7));

// 没有 return 返回 undefined
function noReturn() {
    console.log("执行了");
}
console.log("无返回值:", noReturn());

// --- 返回多个值 ---
console.log("\n--- 返回多个值 ---");

// 返回数组
function minMax(arr) {
    return [Math.min(...arr), Math.max(...arr)];
}
const [min, max] = minMax([3, 1, 4, 1, 5, 9]);
console.log("min:", min, "max:", max);

// 返回对象
function createUser(name, age) {
    return {
        name,
        age,
        createdAt: new Date()
    };
}
console.log("createUser:", createUser("Alice", 25));

// --- 返回函数 ---
console.log("\n--- 返回函数 ---");
function createMultiplier(factor) {
    return function(x) {
        return x * factor;
    };
}
const triple = createMultiplier(3);
console.log("triple(5):", triple(5));


console.log("\n" + "=".repeat(60));
console.log("4. 函数作为值");
console.log("=".repeat(60));

// ============================================================
//                    4. 函数作为值
// ============================================================

// --- 函数是一等公民 ---
console.log("\n--- 函数是一等公民 ---");

// 赋值给变量
const fn = function() { return "Hello"; };

// 存储在数组中
const operations = [
    x => x + 1,
    x => x * 2,
    x => x ** 2
];

let value = 5;
for (const op of operations) {
    value = op(value);
}
console.log("链式操作结果:", value);

// 存储在对象中
const calculator = {
    add: (a, b) => a + b,
    subtract: (a, b) => a - b,
    multiply: (a, b) => a * b,
    divide: (a, b) => a / b
};
console.log("calculator.add(10, 5):", calculator.add(10, 5));

// --- 作为参数传递 ---
console.log("\n--- 回调函数 ---");
function processNumbers(arr, callback) {
    return arr.map(callback);
}
console.log("回调:", processNumbers([1, 2, 3], x => x * 2));

// --- 高阶函数 ---
console.log("\n--- 高阶函数 ---");
function withLogging(fn) {
    return function(...args) {
        console.log(`调用 ${fn.name}，参数:`, args);
        const result = fn(...args);
        console.log(`返回:`, result);
        return result;
    };
}

function addNumbers(a, b) {
    return a + b;
}

const loggedAdd = withLogging(addNumbers);
loggedAdd(3, 5);


console.log("\n" + "=".repeat(60));
console.log("5. 函数属性和方法");
console.log("=".repeat(60));

// ============================================================
//                    5. 函数属性和方法
// ============================================================

// --- 函数属性 ---
console.log("\n--- 函数属性 ---");
function exampleFn(a, b, c) {
    return a + b + c;
}
console.log("name:", exampleFn.name);
console.log("length:", exampleFn.length);  // 形参个数

// 自定义属性
exampleFn.description = "这是一个示例函数";
console.log("自定义属性:", exampleFn.description);

// --- call, apply, bind ---
console.log("\n--- call, apply, bind ---");

const person = {
    name: "Alice"
};

function introduce(greeting, punctuation) {
    return `${greeting}, I'm ${this.name}${punctuation}`;
}

// call：逐个传参
console.log("call:", introduce.call(person, "Hello", "!"));

// apply：数组传参
console.log("apply:", introduce.apply(person, ["Hi", "?"]));

// bind：返回绑定 this 的新函数
const boundIntroduce = introduce.bind(person);
console.log("bind:", boundIntroduce("Hey", "."));

// bind 可以预设参数（柯里化）
const sayHello = introduce.bind(person, "Hello");
console.log("预设参数:", sayHello("!!!"));


console.log("\n" + "=".repeat(60));
console.log("6. 递归");
console.log("=".repeat(60));

// ============================================================
//                    6. 递归
// ============================================================

// --- 基本递归 ---
console.log("\n--- 基本递归 ---");
function countdown(n) {
    if (n <= 0) {
        console.log("发射！");
        return;
    }
    console.log(n);
    countdown(n - 1);
}
countdown(3);

// --- 递归计算 ---
console.log("\n--- 递归计算 ---");

// 阶乘
function factorialRec(n) {
    if (n <= 1) return 1;
    return n * factorialRec(n - 1);
}
console.log("5! =", factorialRec(5));

// 斐波那契（效率低）
function fibRec(n) {
    if (n <= 1) return n;
    return fibRec(n - 1) + fibRec(n - 2);
}
console.log("fib(10) =", fibRec(10));

// --- 尾递归优化 ---
console.log("\n--- 尾递归 ---");
function factorialTail(n, acc = 1) {
    if (n <= 1) return acc;
    return factorialTail(n - 1, n * acc);
}
console.log("尾递归阶乘:", factorialTail(5));

// --- 遍历树结构 ---
console.log("\n--- 遍历树结构 ---");
const tree = {
    value: 1,
    children: [
        { value: 2, children: [] },
        {
            value: 3,
            children: [
                { value: 4, children: [] },
                { value: 5, children: [] }
            ]
        }
    ]
};

function sumTree(node) {
    let total = node.value;
    for (const child of node.children) {
        total += sumTree(child);
    }
    return total;
}
console.log("树节点之和:", sumTree(tree));


console.log("\n【总结】");
console.log(`
函数定义：
- 函数声明：会提升
- 函数表达式：不会提升
- 箭头函数：没有自己的 this

参数：
- 默认参数：function(a = 1) {}
- 剩余参数：function(...args) {}
- 解构参数：function({ name }) {}

this 绑定：
- call(thisArg, arg1, arg2)
- apply(thisArg, [args])
- bind(thisArg) 返回新函数

函数是一等公民：
- 可以赋值给变量
- 可以作为参数传递
- 可以作为返回值
`);
```
