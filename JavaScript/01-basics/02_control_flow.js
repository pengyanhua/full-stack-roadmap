/**
 * ============================================================
 *                JavaScript 控制流
 * ============================================================
 * 本文件介绍 JavaScript 中的条件语句和循环。
 * ============================================================
 */

console.log("=".repeat(60));
console.log("1. 条件语句");
console.log("=".repeat(60));

// ============================================================
//                    1. 条件语句
// ============================================================

// --- if...else ---
console.log("\n--- if...else ---");
const score = 85;

if (score >= 90) {
    console.log("等级: A");
} else if (score >= 80) {
    console.log("等级: B");
} else if (score >= 70) {
    console.log("等级: C");
} else {
    console.log("等级: D");
}

// --- 三元运算符 ---
console.log("\n--- 三元运算符 ---");
const age = 20;
const status = age >= 18 ? "成年" : "未成年";
console.log(`${age}岁 -> ${status}`);

// 嵌套三元
const grade = score >= 90 ? "A" : score >= 80 ? "B" : score >= 70 ? "C" : "D";
console.log("嵌套三元:", grade);

// --- switch ---
console.log("\n--- switch ---");
const day = 3;

switch (day) {
    case 1:
        console.log("周一");
        break;
    case 2:
        console.log("周二");
        break;
    case 3:
        console.log("周三");
        break;
    case 4:
    case 5:  // 多个 case 共享代码
        console.log("周四或周五");
        break;
    case 6:
    case 7:
        console.log("周末");
        break;
    default:
        console.log("无效的日期");
}

// switch 的严格比较
console.log("\n--- switch 使用严格比较 ---");
const value = "1";
switch (value) {
    case 1:
        console.log("数字 1");
        break;
    case "1":
        console.log("字符串 '1'");  // 这个会执行
        break;
}

// --- 逻辑运算符的短路求值 ---
console.log("\n--- 短路求值 ---");

// && 短路：第一个假值或最后一个值
console.log("true && 'hello':", true && "hello");  // "hello"
console.log("false && 'hello':", false && "hello");  // false
console.log("'a' && 'b' && 'c':", "a" && "b" && "c");  // "c"

// || 短路：第一个真值或最后一个值
console.log("'' || 'default':", "" || "default");  // "default"
console.log("'value' || 'default':", "value" || "default");  // "value"

// ?? 空值合并（只对 null/undefined）
console.log("0 ?? 'default':", 0 ?? "default");  // 0
console.log("'' ?? 'default':", "" ?? "default");  // ""
console.log("null ?? 'default':", null ?? "default");  // "default"
console.log("undefined ?? 'default':", undefined ?? "default");  // "default"

// 实际应用
const config = {
    timeout: 0,
    retries: null
};
const timeout = config.timeout ?? 5000;  // 0（不是 5000）
const retries = config.retries ?? 3;  // 3
console.log("timeout:", timeout);
console.log("retries:", retries);


console.log("\n" + "=".repeat(60));
console.log("2. 循环语句");
console.log("=".repeat(60));

// ============================================================
//                    2. 循环语句
// ============================================================

// --- for 循环 ---
console.log("\n--- for 循环 ---");
for (let i = 0; i < 5; i++) {
    process.stdout.write(`${i} `);
}
console.log();

// --- while 循环 ---
console.log("\n--- while 循环 ---");
let count = 0;
while (count < 5) {
    process.stdout.write(`${count} `);
    count++;
}
console.log();

// --- do...while 循环 ---
console.log("\n--- do...while 循环 ---");
let num = 0;
do {
    process.stdout.write(`${num} `);
    num++;
} while (num < 5);
console.log();

// --- for...of（迭代值）---
console.log("\n--- for...of ---");
const fruits = ["apple", "banana", "cherry"];
for (const fruit of fruits) {
    console.log("  fruit:", fruit);
}

// 字符串迭代
console.log("\n字符串迭代:");
for (const char of "Hello") {
    process.stdout.write(`${char} `);
}
console.log();

// 带索引的 for...of
console.log("\n带索引:");
for (const [index, fruit] of fruits.entries()) {
    console.log(`  ${index}: ${fruit}`);
}

// --- for...in（迭代键）---
console.log("\n--- for...in ---");
const person = { name: "Alice", age: 25, city: "Beijing" };
for (const key in person) {
    console.log(`  ${key}: ${person[key]}`);
}

// 【警告】for...in 会遍历原型链上的属性
console.log("\n【注意】for...in 和数组");
const arr = [10, 20, 30];
arr.customProp = "custom";
for (const key in arr) {
    console.log(`  key: ${key}, value: ${arr[key]}`);
}
// 推荐使用 for...of 遍历数组

// --- break 和 continue ---
console.log("\n--- break 和 continue ---");

// break
console.log("break 示例:");
for (let i = 0; i < 10; i++) {
    if (i === 5) break;
    process.stdout.write(`${i} `);
}
console.log();

// continue
console.log("continue 示例:");
for (let i = 0; i < 10; i++) {
    if (i % 2 === 0) continue;
    process.stdout.write(`${i} `);
}
console.log();

// --- 标签语句 ---
console.log("\n--- 标签语句 ---");
outer: for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
        if (i === 1 && j === 1) {
            console.log("跳出外层循环");
            break outer;
        }
        console.log(`  i=${i}, j=${j}`);
    }
}


console.log("\n" + "=".repeat(60));
console.log("3. 数组方法（迭代）");
console.log("=".repeat(60));

// ============================================================
//                    3. 数组方法
// ============================================================

const numbers = [1, 2, 3, 4, 5];

// --- forEach ---
console.log("\n--- forEach ---");
numbers.forEach((num, index) => {
    console.log(`  [${index}] = ${num}`);
});

// --- map ---
console.log("\n--- map ---");
const doubled = numbers.map(n => n * 2);
console.log("doubled:", doubled);

// --- filter ---
console.log("\n--- filter ---");
const evens = numbers.filter(n => n % 2 === 0);
console.log("evens:", evens);

// --- reduce ---
console.log("\n--- reduce ---");
const sum = numbers.reduce((acc, n) => acc + n, 0);
console.log("sum:", sum);

const max = numbers.reduce((a, b) => a > b ? a : b);
console.log("max:", max);

// --- find / findIndex ---
console.log("\n--- find / findIndex ---");
const found = numbers.find(n => n > 3);
const foundIndex = numbers.findIndex(n => n > 3);
console.log("find(n > 3):", found);
console.log("findIndex(n > 3):", foundIndex);

// --- some / every ---
console.log("\n--- some / every ---");
console.log("some(n > 3):", numbers.some(n => n > 3));
console.log("every(n > 0):", numbers.every(n => n > 0));

// --- flat / flatMap ---
console.log("\n--- flat / flatMap ---");
const nested = [[1, 2], [3, 4], [5]];
console.log("flat:", nested.flat());

const words = ["hello world", "foo bar"];
console.log("flatMap:", words.flatMap(w => w.split(" ")));

// --- 链式调用 ---
console.log("\n--- 链式调用 ---");
const result = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    .filter(n => n % 2 === 0)  // 偶数
    .map(n => n * n)           // 平方
    .reduce((a, b) => a + b);  // 求和
console.log("偶数的平方和:", result);


console.log("\n" + "=".repeat(60));
console.log("4. 错误处理");
console.log("=".repeat(60));

// ============================================================
//                    4. 错误处理
// ============================================================

// --- try...catch ---
console.log("\n--- try...catch ---");
try {
    const obj = JSON.parse("invalid json");
} catch (error) {
    console.log("捕获错误:", error.message);
}

// --- try...catch...finally ---
console.log("\n--- try...catch...finally ---");
function divide(a, b) {
    try {
        if (b === 0) {
            throw new Error("除数不能为零");
        }
        return a / b;
    } catch (error) {
        console.log("错误:", error.message);
        return null;
    } finally {
        console.log("finally 总是执行");
    }
}
console.log("结果:", divide(10, 0));

// --- 自定义错误 ---
console.log("\n--- 自定义错误 ---");
class ValidationError extends Error {
    constructor(message) {
        super(message);
        this.name = "ValidationError";
    }
}

function validateAge(age) {
    if (typeof age !== "number") {
        throw new TypeError("年龄必须是数字");
    }
    if (age < 0 || age > 150) {
        throw new ValidationError("年龄必须在 0-150 之间");
    }
    return true;
}

try {
    validateAge(-5);
} catch (error) {
    if (error instanceof ValidationError) {
        console.log("验证错误:", error.message);
    } else if (error instanceof TypeError) {
        console.log("类型错误:", error.message);
    } else {
        throw error;  // 重新抛出未知错误
    }
}

// --- 错误类型 ---
console.log("\n--- 内置错误类型 ---");
console.log(`
  Error         - 通用错误
  SyntaxError   - 语法错误
  ReferenceError - 引用未定义变量
  TypeError     - 类型错误
  RangeError    - 数值超出范围
  URIError      - URI 处理错误
  EvalError     - eval() 错误（已废弃）
`);

// --- 可选链和错误预防 ---
console.log("--- 可选链 (?.) ---");
const user = {
    profile: {
        name: "Alice"
    }
};

// 传统方式
const city1 = user && user.address && user.address.city;
console.log("传统方式:", city1);

// 可选链
const city2 = user?.address?.city;
console.log("可选链:", city2);

// 可选链调用方法
const result2 = user?.getName?.();
console.log("可选链调用:", result2);

// 可选链访问数组
const arr2 = null;
console.log("可选链数组:", arr2?.[0]);


console.log("\n" + "=".repeat(60));
console.log("5. 迭代器和生成器");
console.log("=".repeat(60));

// ============================================================
//                    5. 迭代器和生成器
// ============================================================

// --- 可迭代对象 ---
console.log("\n--- 可迭代对象 ---");
const iterable = [1, 2, 3];
const iterator = iterable[Symbol.iterator]();

console.log("next():", iterator.next());
console.log("next():", iterator.next());
console.log("next():", iterator.next());
console.log("next():", iterator.next());

// --- 自定义可迭代对象 ---
console.log("\n--- 自定义可迭代对象 ---");
const range = {
    start: 1,
    end: 5,
    [Symbol.iterator]() {
        let current = this.start;
        const end = this.end;
        return {
            next() {
                if (current <= end) {
                    return { value: current++, done: false };
                }
                return { done: true };
            }
        };
    }
};

console.log("自定义 range:");
for (const n of range) {
    process.stdout.write(`${n} `);
}
console.log();

// --- 生成器函数 ---
console.log("\n--- 生成器函数 ---");
function* numberGenerator() {
    yield 1;
    yield 2;
    yield 3;
}

const gen = numberGenerator();
console.log("生成器:", [...gen]);

// 生成器实现 range
function* rangeGen(start, end) {
    for (let i = start; i <= end; i++) {
        yield i;
    }
}

console.log("rangeGen(1, 5):", [...rangeGen(1, 5)]);

// 无限序列
function* fibonacci() {
    let [a, b] = [0, 1];
    while (true) {
        yield a;
        [a, b] = [b, a + b];
    }
}

console.log("前 10 个斐波那契数:");
const fib = fibonacci();
for (let i = 0; i < 10; i++) {
    process.stdout.write(`${fib.next().value} `);
}
console.log();


console.log("\n【总结】");
console.log(`
- 优先使用 === 进行比较
- 使用 ?? 处理 null/undefined，使用 || 处理所有假值
- 使用 for...of 遍历数组，for...in 遍历对象键
- 善用数组方法：map, filter, reduce, find 等
- 使用 try...catch 处理错误
- 使用可选链 (?.) 安全访问嵌套属性
`);
