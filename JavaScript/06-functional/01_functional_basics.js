/**
 * ============================================================
 *                JavaScript 函数式编程基础
 * ============================================================
 * 本文件介绍 JavaScript 中的函数式编程概念和技术。
 * ============================================================
 */

console.log("=".repeat(60));
console.log("1. 纯函数");
console.log("=".repeat(60));

// ============================================================
//                    1. 纯函数
// ============================================================

/**
 * 纯函数特点：
 * 1. 相同输入总是返回相同输出
 * 2. 没有副作用（不修改外部状态）
 */

// --- 纯函数示例 ---
console.log("\n--- 纯函数 ---");

// 纯函数
function add(a, b) {
    return a + b;
}

function multiply(a, b) {
    return a * b;
}

console.log("add(2, 3):", add(2, 3));
console.log("multiply(4, 5):", multiply(4, 5));

// 非纯函数（有副作用）
let total = 0;
function addToTotal(value) {
    total += value;  // 修改外部状态
    return total;
}

console.log("\n--- 非纯函数 ---");
console.log("addToTotal(5):", addToTotal(5));
console.log("addToTotal(5):", addToTotal(5));  // 相同输入，不同输出

// --- 避免副作用 ---
console.log("\n--- 避免副作用 ---");

// 不好：修改原数组
function addItemBad(arr, item) {
    arr.push(item);
    return arr;
}

// 好：返回新数组
function addItemGood(arr, item) {
    return [...arr, item];
}

const original = [1, 2, 3];
const newArr = addItemGood(original, 4);
console.log("原数组:", original);
console.log("新数组:", newArr);


console.log("\n" + "=".repeat(60));
console.log("2. 不可变性");
console.log("=".repeat(60));

// ============================================================
//                    2. 不可变性
// ============================================================

// --- 对象不可变 ---
console.log("\n--- 对象不可变操作 ---");

const person = {
    name: "Alice",
    age: 25,
    address: {
        city: "Beijing",
        street: "Main St"
    }
};

// 浅层更新
const updatedPerson = {
    ...person,
    age: 26
};

console.log("原对象:", person);
console.log("新对象:", updatedPerson);

// 深层更新
const movedPerson = {
    ...person,
    address: {
        ...person.address,
        city: "Shanghai"
    }
};

console.log("搬家后:", movedPerson);
console.log("原对象地址:", person.address);

// --- 数组不可变 ---
console.log("\n--- 数组不可变操作 ---");

const numbers = [1, 2, 3, 4, 5];

// 添加元素
const added = [...numbers, 6];
console.log("添加:", added);

// 删除元素
const removed = numbers.filter((_, i) => i !== 2);
console.log("删除索引2:", removed);

// 更新元素
const updated = numbers.map((n, i) => i === 2 ? 10 : n);
console.log("更新索引2:", updated);

// 插入元素
const inserted = [...numbers.slice(0, 2), 99, ...numbers.slice(2)];
console.log("插入到索引2:", inserted);

// --- Object.freeze ---
console.log("\n--- Object.freeze ---");

const frozen = Object.freeze({
    name: "Frozen",
    nested: { value: 1 }
});

// frozen.name = "Changed";  // 静默失败（严格模式下报错）
frozen.nested.value = 2;  // 嵌套对象可以修改！

console.log("frozen:", frozen);
console.log("注意：freeze 是浅冻结");

// 深冻结函数
function deepFreeze(obj) {
    Object.keys(obj).forEach(key => {
        if (typeof obj[key] === "object" && obj[key] !== null) {
            deepFreeze(obj[key]);
        }
    });
    return Object.freeze(obj);
}


console.log("\n" + "=".repeat(60));
console.log("3. 高阶函数");
console.log("=".repeat(60));

// ============================================================
//                    3. 高阶函数
// ============================================================

/**
 * 高阶函数：接收函数作为参数或返回函数的函数
 */

// --- map, filter, reduce ---
console.log("\n--- map, filter, reduce ---");

const nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// map：转换每个元素
const doubled = nums.map(n => n * 2);
console.log("doubled:", doubled);

// filter：筛选元素
const evens = nums.filter(n => n % 2 === 0);
console.log("evens:", evens);

// reduce：归约为单个值
const sum = nums.reduce((acc, n) => acc + n, 0);
console.log("sum:", sum);

// 链式调用
const result = nums
    .filter(n => n % 2 === 0)  // 偶数
    .map(n => n * n)            // 平方
    .reduce((acc, n) => acc + n, 0);  // 求和
console.log("偶数平方和:", result);

// --- 自定义高阶函数 ---
console.log("\n--- 自定义高阶函数 ---");

// 函数作为参数
function applyOperation(arr, operation) {
    return arr.map(operation);
}

console.log("applyOperation:", applyOperation([1, 2, 3], x => x * 10));

// 返回函数
function createMultiplier(factor) {
    return function(number) {
        return number * factor;
    };
}

const triple = createMultiplier(3);
console.log("triple(5):", triple(5));

// 函数作为参数和返回值
function compose(f, g) {
    return function(x) {
        return f(g(x));
    };
}

const addOne = x => x + 1;
const square = x => x * x;
const addOneThenSquare = compose(square, addOne);
console.log("addOneThenSquare(4):", addOneThenSquare(4));  // (4+1)^2 = 25


console.log("\n" + "=".repeat(60));
console.log("4. 函数组合");
console.log("=".repeat(60));

// ============================================================
//                    4. 函数组合
// ============================================================

// --- compose 和 pipe ---
console.log("\n--- compose 和 pipe ---");

// compose：从右到左执行
function composeMany(...fns) {
    return function(x) {
        return fns.reduceRight((acc, fn) => fn(acc), x);
    };
}

// pipe：从左到右执行
function pipe(...fns) {
    return function(x) {
        return fns.reduce((acc, fn) => fn(acc), x);
    };
}

const addTwo = x => x + 2;
const multiplyThree = x => x * 3;
const subtractOne = x => x - 1;

const composed = composeMany(subtractOne, multiplyThree, addTwo);
console.log("compose (5+2)*3-1 =", composed(5));

const piped = pipe(addTwo, multiplyThree, subtractOne);
console.log("pipe (5+2)*3-1 =", piped(5));

// --- 实际应用 ---
console.log("\n--- 实际应用 ---");

// 数据处理管道
const users = [
    { name: "Alice", age: 25, active: true },
    { name: "Bob", age: 30, active: false },
    { name: "Charlie", age: 35, active: true },
    { name: "Diana", age: 28, active: true }
];

const getActiveUsers = users => users.filter(u => u.active);
const getNames = users => users.map(u => u.name);
const sortNames = names => [...names].sort();
const formatList = names => names.join(", ");

const processUsers = pipe(
    getActiveUsers,
    getNames,
    sortNames,
    formatList
);

console.log("活跃用户:", processUsers(users));


console.log("\n" + "=".repeat(60));
console.log("5. 柯里化");
console.log("=".repeat(60));

// ============================================================
//                    5. 柯里化
// ============================================================

/**
 * 柯里化：将多参数函数转换为一系列单参数函数
 */

// --- 手动柯里化 ---
console.log("\n--- 手动柯里化 ---");

// 普通函数
function addThree(a, b, c) {
    return a + b + c;
}

// 柯里化版本
function curriedAddThree(a) {
    return function(b) {
        return function(c) {
            return a + b + c;
        };
    };
}

console.log("addThree(1, 2, 3):", addThree(1, 2, 3));
console.log("curriedAddThree(1)(2)(3):", curriedAddThree(1)(2)(3));

// 箭头函数写法
const curriedAdd = a => b => c => a + b + c;
console.log("curriedAdd(1)(2)(3):", curriedAdd(1)(2)(3));

// --- 自动柯里化函数 ---
console.log("\n--- 自动柯里化 ---");

function curry(fn) {
    return function curried(...args) {
        if (args.length >= fn.length) {
            return fn.apply(this, args);
        }
        return function(...moreArgs) {
            return curried.apply(this, [...args, ...moreArgs]);
        };
    };
}

function greet(greeting, name, punctuation) {
    return `${greeting}, ${name}${punctuation}`;
}

const curriedGreet = curry(greet);

console.log(curriedGreet("Hello", "Alice", "!"));
console.log(curriedGreet("Hello")("Bob")("?"));
console.log(curriedGreet("Hi", "Charlie")("~"));

// --- 柯里化的实际应用 ---
console.log("\n--- 柯里化应用 ---");

// 创建专用函数
const multiply = curry((a, b) => a * b);
const double2 = multiply(2);
const triple2 = multiply(3);

console.log("double2(5):", double2(5));
console.log("triple2(5):", triple2(5));

// 配置函数
const log = curry((level, message) => {
    console.log(`[${level}] ${message}`);
});

const info = log("INFO");
const error = log("ERROR");

info("程序启动");
error("发生错误");


console.log("\n" + "=".repeat(60));
console.log("6. 偏应用");
console.log("=".repeat(60));

// ============================================================
//                    6. 偏应用
// ============================================================

/**
 * 偏应用：固定函数的部分参数，返回新函数
 */

// --- 使用 bind ---
console.log("\n--- 使用 bind ---");

function fullGreet(greeting, title, name) {
    return `${greeting}, ${title} ${name}`;
}

const sayHello = fullGreet.bind(null, "Hello");
const sayHelloToMr = fullGreet.bind(null, "Hello", "Mr.");

console.log(sayHello("Ms.", "Smith"));
console.log(sayHelloToMr("Johnson"));

// --- 自定义偏应用 ---
console.log("\n--- 自定义偏应用 ---");

function partial(fn, ...presetArgs) {
    return function(...laterArgs) {
        return fn(...presetArgs, ...laterArgs);
    };
}

const addFive = partial(add, 5);
console.log("addFive(10):", addFive(10));

// 带占位符的偏应用
const _ = Symbol("placeholder");

function partialWithPlaceholder(fn, ...presetArgs) {
    return function(...laterArgs) {
        const args = presetArgs.map(arg =>
            arg === _ ? laterArgs.shift() : arg
        );
        return fn(...args, ...laterArgs);
    };
}

function subtract(a, b) {
    return a - b;
}

const subtractFrom10 = partialWithPlaceholder(subtract, 10, _);
const subtract5 = partialWithPlaceholder(subtract, _, 5);

console.log("subtractFrom10(3):", subtractFrom10(3));  // 10 - 3 = 7
console.log("subtract5(10):", subtract5(10));          // 10 - 5 = 5


console.log("\n" + "=".repeat(60));
console.log("7. 函数式工具函数");
console.log("=".repeat(60));

// ============================================================
//                    7. 函数式工具函数
// ============================================================

// --- identity 和 constant ---
console.log("\n--- identity 和 constant ---");

const identity = x => x;
const constant = x => () => x;

console.log("identity(5):", identity(5));
console.log("constant(5)():", constant(5)());

// 用途：作为默认函数
const data = [1, 2, 3];
const mapped = data.map(identity);
console.log("mapped:", mapped);

// --- tap（用于调试）---
console.log("\n--- tap（调试工具）---");

function tap(fn) {
    return function(x) {
        fn(x);
        return x;
    };
}

const debugResult = [1, 2, 3, 4, 5]
    .map(x => x * 2)
    .filter(tap(x => console.log("  中间值:", x)))
    .filter(x => x > 5);

console.log("结果:", debugResult);

// --- memoize ---
console.log("\n--- memoize ---");

function memoize(fn) {
    const cache = new Map();
    return function(...args) {
        const key = JSON.stringify(args);
        if (cache.has(key)) {
            console.log("  (缓存命中)");
            return cache.get(key);
        }
        const result = fn.apply(this, args);
        cache.set(key, result);
        return result;
    };
}

function expensiveCalculation(n) {
    console.log("  计算中...");
    return n * n;
}

const memoizedCalc = memoize(expensiveCalculation);

console.log("第一次调用:", memoizedCalc(5));
console.log("第二次调用:", memoizedCalc(5));
console.log("不同参数:", memoizedCalc(6));

// --- once ---
console.log("\n--- once ---");

function once(fn) {
    let called = false;
    let result;
    return function(...args) {
        if (!called) {
            called = true;
            result = fn.apply(this, args);
        }
        return result;
    };
}

const initialize = once(() => {
    console.log("  初始化执行");
    return { initialized: true };
});

console.log("第一次:", initialize());
console.log("第二次:", initialize());

// --- debounce ---
console.log("\n--- debounce ---");

function debounce(fn, delay) {
    let timeoutId;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn.apply(this, args), delay);
    };
}

// --- throttle ---
console.log("\n--- throttle ---");

function throttle(fn, limit) {
    let inThrottle = false;
    return function(...args) {
        if (!inThrottle) {
            fn.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}


console.log("\n" + "=".repeat(60));
console.log("8. 函子和 Monad 概念");
console.log("=".repeat(60));

// ============================================================
//                    8. 函子和 Monad 概念
// ============================================================

// --- 简单函子 ---
console.log("\n--- 简单函子（Box）---");

class Box {
    constructor(value) {
        this._value = value;
    }

    static of(value) {
        return new Box(value);
    }

    map(fn) {
        return Box.of(fn(this._value));
    }

    fold(fn) {
        return fn(this._value);
    }

    toString() {
        return `Box(${this._value})`;
    }
}

const boxResult = Box.of(5)
    .map(x => x + 1)
    .map(x => x * 2)
    .fold(x => x);

console.log("Box 计算结果:", boxResult);

// --- Maybe 函子（处理 null）---
console.log("\n--- Maybe 函子 ---");

class Maybe {
    constructor(value) {
        this._value = value;
    }

    static of(value) {
        return new Maybe(value);
    }

    static nothing() {
        return new Maybe(null);
    }

    isNothing() {
        return this._value === null || this._value === undefined;
    }

    map(fn) {
        return this.isNothing() ? Maybe.nothing() : Maybe.of(fn(this._value));
    }

    getOrElse(defaultValue) {
        return this.isNothing() ? defaultValue : this._value;
    }

    toString() {
        return this.isNothing() ? "Nothing" : `Just(${this._value})`;
    }
}

// 安全地访问嵌套属性
const user1 = { name: "Alice", address: { city: "Beijing" } };
const user2 = { name: "Bob" };

function getCity(user) {
    return Maybe.of(user)
        .map(u => u.address)
        .map(a => a.city)
        .getOrElse("Unknown");
}

console.log("user1 city:", getCity(user1));
console.log("user2 city:", getCity(user2));

// --- Either 函子（错误处理）---
console.log("\n--- Either 函子 ---");

class Left {
    constructor(value) {
        this._value = value;
    }

    map(fn) {
        return this;  // 不执行 map
    }

    fold(leftFn, rightFn) {
        return leftFn(this._value);
    }
}

class Right {
    constructor(value) {
        this._value = value;
    }

    map(fn) {
        return new Right(fn(this._value));
    }

    fold(leftFn, rightFn) {
        return rightFn(this._value);
    }
}

function divide2(a, b) {
    return b === 0
        ? new Left("除数不能为零")
        : new Right(a / b);
}

const divResult1 = divide2(10, 2)
    .map(x => x * 2)
    .fold(
        error => `错误: ${error}`,
        value => `结果: ${value}`
    );

const divResult2 = divide2(10, 0)
    .map(x => x * 2)
    .fold(
        error => `错误: ${error}`,
        value => `结果: ${value}`
    );

console.log(divResult1);
console.log(divResult2);


console.log("\n【总结】");
console.log(`
函数式编程核心概念：

纯函数：
- 相同输入 -> 相同输出
- 无副作用
- 易于测试和推理

不可变性：
- 不修改原数据
- 使用展开运算符创建新对象/数组
- Object.freeze 冻结对象

高阶函数：
- map, filter, reduce
- 函数作为参数
- 函数作为返回值

函数组合：
- compose（从右到左）
- pipe（从左到右）
- 创建数据处理管道

柯里化和偏应用：
- 柯里化：多参数 -> 单参数函数链
- 偏应用：固定部分参数

工具函数：
- memoize：缓存结果
- once：只执行一次
- debounce/throttle：控制执行频率

函子概念：
- Box：简单包装器
- Maybe：处理 null/undefined
- Either：错误处理
`);
