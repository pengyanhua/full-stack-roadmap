/**
 * ============================================================
 *                JavaScript 变量与数据类型
 * ============================================================
 * 本文件介绍 JavaScript 中的变量声明和基本数据类型。
 * ============================================================
 */

console.log("=".repeat(60));
console.log("1. 变量声明");
console.log("=".repeat(60));

// ============================================================
//                    1. 变量声明
// ============================================================

/**
 * 【var、let、const 的区别】
 *
 * var   - 函数作用域，可重复声明，有变量提升
 * let   - 块级作用域，不可重复声明，暂时性死区
 * const - 块级作用域，不可重复声明，必须初始化，不可重新赋值
 *
 * 【最佳实践】优先使用 const，需要重新赋值时使用 let，避免使用 var
 */

// --- var（不推荐）---
var x = 10;
var x = 20;  // 可以重复声明
console.log("var x:", x);

// --- let ---
let y = 10;
// let y = 20;  // 错误：不能重复声明
y = 20;  // 可以重新赋值
console.log("let y:", y);

// --- const ---
const z = 10;
// z = 20;  // 错误：不能重新赋值
console.log("const z:", z);

// 【注意】const 对象的属性可以修改
const person = { name: "Alice" };
person.name = "Bob";  // 可以修改属性
// person = {};  // 错误：不能重新赋值
console.log("const 对象:", person);

// 【变量提升】
console.log("\n--- 变量提升 ---");
console.log("var 提升:", typeof hoistedVar);  // undefined
// console.log(hoistedLet);  // ReferenceError: 暂时性死区
var hoistedVar = "I'm hoisted";

// 【块级作用域】
console.log("\n--- 块级作用域 ---");
{
    let blockLet = "block scoped";
    var blockVar = "function scoped";
}
// console.log(blockLet);  // 错误：blockLet 不可访问
console.log("var 可访问:", blockVar);


console.log("\n" + "=".repeat(60));
console.log("2. 基本数据类型");
console.log("=".repeat(60));

// ============================================================
//                    2. 基本数据类型
// ============================================================

/**
 * JavaScript 有 7 种原始类型 + 1 种引用类型：
 *
 * 原始类型（Primitive）：
 * - number    数字（整数和浮点数）
 * - string    字符串
 * - boolean   布尔值
 * - undefined 未定义
 * - null      空值
 * - symbol    符号（ES6+）
 * - bigint    大整数（ES2020+）
 *
 * 引用类型：
 * - object    对象（包括数组、函数等）
 */

// --- Number ---
console.log("\n--- Number ---");
const integer = 42;
const float = 3.14;
const negative = -10;
const infinity = Infinity;
const notANumber = NaN;

console.log("整数:", integer);
console.log("浮点数:", float);
console.log("Infinity:", infinity);
console.log("NaN:", notANumber);
console.log("NaN === NaN:", NaN === NaN);  // false
console.log("Number.isNaN(NaN):", Number.isNaN(NaN));  // true

// 数字方法
console.log("toFixed(2):", (3.14159).toFixed(2));
console.log("parseInt:", parseInt("42px"));
console.log("parseFloat:", parseFloat("3.14abc"));

// --- String ---
console.log("\n--- String ---");
const single = 'single quotes';
const double = "double quotes";
const template = `template literal`;

// 模板字符串
const name = "Alice";
const age = 25;
console.log(`${name} is ${age} years old`);

// 多行字符串
const multiline = `
  Line 1
  Line 2
  Line 3
`;
console.log("多行字符串:", multiline.trim());

// 字符串方法
const str = "Hello, World!";
console.log("length:", str.length);
console.log("toUpperCase:", str.toUpperCase());
console.log("toLowerCase:", str.toLowerCase());
console.log("indexOf:", str.indexOf("World"));
console.log("includes:", str.includes("World"));
console.log("startsWith:", str.startsWith("Hello"));
console.log("slice(0, 5):", str.slice(0, 5));
console.log("split(','):", str.split(","));
console.log("replace:", str.replace("World", "JavaScript"));
console.log("trim:", "  spaces  ".trim());
console.log("padStart:", "5".padStart(3, "0"));
console.log("repeat:", "ab".repeat(3));

// --- Boolean ---
console.log("\n--- Boolean ---");
const truthy = true;
const falsy = false;

// 假值（Falsy）
console.log("假值列表:");
console.log("  false:", Boolean(false));
console.log("  0:", Boolean(0));
console.log("  '':", Boolean(""));
console.log("  null:", Boolean(null));
console.log("  undefined:", Boolean(undefined));
console.log("  NaN:", Boolean(NaN));

// 真值（Truthy）
console.log("真值示例:");
console.log("  []:", Boolean([]));  // true！空数组是真值
console.log("  {}:", Boolean({}));  // true！空对象是真值
console.log("  '0':", Boolean("0"));  // true！非空字符串

// --- undefined 和 null ---
console.log("\n--- undefined 和 null ---");
let undefinedVar;
const nullVar = null;

console.log("undefined:", undefinedVar);
console.log("null:", nullVar);
console.log("typeof undefined:", typeof undefined);
console.log("typeof null:", typeof null);  // "object" - 历史遗留问题
console.log("null == undefined:", null == undefined);  // true
console.log("null === undefined:", null === undefined);  // false

// --- Symbol ---
console.log("\n--- Symbol ---");
const sym1 = Symbol("description");
const sym2 = Symbol("description");
console.log("sym1 === sym2:", sym1 === sym2);  // false，每个 Symbol 都是唯一的

// Symbol 作为对象属性
const obj = {
    [sym1]: "value1",
    normalKey: "value2"
};
console.log("Symbol 属性:", obj[sym1]);
console.log("Object.keys 不包含 Symbol:", Object.keys(obj));

// --- BigInt ---
console.log("\n--- BigInt ---");
const bigInt1 = 9007199254740991n;
const bigInt2 = BigInt("9007199254740992");
console.log("BigInt:", bigInt1);
console.log("BigInt + 1n:", bigInt1 + 1n);
// console.log(bigInt1 + 1);  // 错误：不能混合 BigInt 和 Number


console.log("\n" + "=".repeat(60));
console.log("3. 类型检查与转换");
console.log("=".repeat(60));

// ============================================================
//                    3. 类型检查与转换
// ============================================================

// --- typeof 运算符 ---
console.log("\n--- typeof ---");
console.log("typeof 42:", typeof 42);
console.log("typeof 'hello':", typeof "hello");
console.log("typeof true:", typeof true);
console.log("typeof undefined:", typeof undefined);
console.log("typeof null:", typeof null);  // "object" - 历史遗留
console.log("typeof Symbol():", typeof Symbol());
console.log("typeof {}:", typeof {});
console.log("typeof []:", typeof []);  // "object"
console.log("typeof function(){}:", typeof function(){});

// --- instanceof ---
console.log("\n--- instanceof ---");
console.log("[] instanceof Array:", [] instanceof Array);
console.log("{} instanceof Object:", {} instanceof Object);
console.log("new Date() instanceof Date:", new Date() instanceof Date);

// --- 类型转换 ---
console.log("\n--- 显式类型转换 ---");

// 转换为字符串
console.log("String(123):", String(123));
console.log("(123).toString():", (123).toString());
console.log("'' + 123:", "" + 123);

// 转换为数字
console.log("Number('123'):", Number("123"));
console.log("parseInt('123'):", parseInt("123"));
console.log("parseFloat('3.14'):", parseFloat("3.14"));
console.log("+'123':", +"123");

// 转换为布尔值
console.log("Boolean(1):", Boolean(1));
console.log("!!1:", !!1);

// --- 隐式类型转换 ---
console.log("\n--- 隐式类型转换 ---");
console.log("'5' + 3:", "5" + 3);  // "53"
console.log("'5' - 3:", "5" - 3);  // 2
console.log("'5' * '2':", "5" * "2");  // 10
console.log("true + true:", true + true);  // 2
console.log("[] + []:", [] + []);  // ""
console.log("[] + {}:", [] + {});  // "[object Object]"


console.log("\n" + "=".repeat(60));
console.log("4. 相等性比较");
console.log("=".repeat(60));

// ============================================================
//                    4. 相等性比较
// ============================================================

/**
 * == （宽松相等）：会进行类型转换
 * === （严格相等）：不进行类型转换，推荐使用
 * Object.is()：与 === 类似，但处理 NaN 和 ±0 不同
 */

console.log("\n--- == vs === ---");
console.log("5 == '5':", 5 == "5");    // true
console.log("5 === '5':", 5 === "5");  // false

console.log("null == undefined:", null == undefined);    // true
console.log("null === undefined:", null === undefined);  // false

console.log("0 == false:", 0 == false);    // true
console.log("0 === false:", 0 === false);  // false

// --- Object.is ---
console.log("\n--- Object.is ---");
console.log("Object.is(NaN, NaN):", Object.is(NaN, NaN));  // true
console.log("Object.is(0, -0):", Object.is(0, -0));  // false
console.log("0 === -0:", 0 === -0);  // true


console.log("\n" + "=".repeat(60));
console.log("5. 解构赋值");
console.log("=".repeat(60));

// ============================================================
//                    5. 解构赋值
// ============================================================

// --- 数组解构 ---
console.log("\n--- 数组解构 ---");
const [a, b, c] = [1, 2, 3];
console.log("a, b, c:", a, b, c);

// 跳过元素
const [first, , third] = [1, 2, 3];
console.log("first, third:", first, third);

// 剩余元素
const [head, ...tail] = [1, 2, 3, 4, 5];
console.log("head:", head);
console.log("tail:", tail);

// 默认值
const [val1, val2 = 10] = [1];
console.log("val1, val2:", val1, val2);

// 交换变量
let swap1 = 1, swap2 = 2;
[swap1, swap2] = [swap2, swap1];
console.log("交换后:", swap1, swap2);

// --- 对象解构 ---
console.log("\n--- 对象解构 ---");
const { name: userName, age: userAge } = { name: "Alice", age: 25 };
console.log("userName, userAge:", userName, userAge);

// 同名简写
const { name: n, age: a2 } = { name: "Bob", age: 30 };
console.log("n, a2:", n, a2);

// 嵌套解构
const user = {
    id: 1,
    profile: {
        email: "alice@example.com",
        avatar: "avatar.png"
    }
};
const { profile: { email } } = user;
console.log("email:", email);

// 默认值
const { missing = "default" } = {};
console.log("missing:", missing);

// 函数参数解构
function greet({ name, greeting = "Hello" }) {
    console.log(`${greeting}, ${name}!`);
}
greet({ name: "Alice" });


console.log("\n" + "=".repeat(60));
console.log("6. 展开运算符");
console.log("=".repeat(60));

// ============================================================
//                    6. 展开运算符 (Spread Operator)
// ============================================================

// --- 数组展开 ---
console.log("\n--- 数组展开 ---");
const arr1 = [1, 2, 3];
const arr2 = [4, 5, 6];
const merged = [...arr1, ...arr2];
console.log("合并数组:", merged);

// 复制数组
const copy = [...arr1];
console.log("复制数组:", copy);

// 函数调用
console.log("Math.max:", Math.max(...arr1));

// --- 对象展开 ---
console.log("\n--- 对象展开 ---");
const obj1 = { a: 1, b: 2 };
const obj2 = { c: 3, d: 4 };
const mergedObj = { ...obj1, ...obj2 };
console.log("合并对象:", mergedObj);

// 覆盖属性
const updated = { ...obj1, b: 10, e: 5 };
console.log("覆盖属性:", updated);

// 浅拷贝
const original = { nested: { value: 1 } };
const shallowCopy = { ...original };
shallowCopy.nested.value = 2;
console.log("浅拷贝影响原对象:", original.nested.value);  // 2


console.log("\n【总结】");
console.log(`
- 优先使用 const，需要重新赋值时使用 let
- 使用 === 而不是 ==
- 了解假值：false, 0, '', null, undefined, NaN
- 善用解构赋值和展开运算符
- typeof 检查基本类型，instanceof 检查对象类型
`);
