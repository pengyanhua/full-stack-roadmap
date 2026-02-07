# objects arrays.js

::: info 文件信息
- 📄 原文件：`03_objects_arrays.js`
- 🔤 语言：javascript
:::

============================================================
             JavaScript 对象和数组
============================================================
本文件介绍 JavaScript 中的对象和数组操作。
============================================================

## 完整代码

```javascript
/**
 * ============================================================
 *              JavaScript 对象和数组
 * ============================================================
 * 本文件介绍 JavaScript 中的对象和数组操作。
 * ============================================================
 */

console.log("=".repeat(60));
console.log("1. 对象基础");
console.log("=".repeat(60));

// ============================================================
//                    1. 对象基础
// ============================================================

// --- 创建对象 ---
console.log("\n--- 创建对象 ---");

// 字面量
const person = {
    name: "Alice",
    age: 25,
    isStudent: false
};
console.log("字面量:", person);

// 构造函数
const obj = new Object();
obj.key = "value";
console.log("构造函数:", obj);

// Object.create
const proto = { greet() { return "Hello!"; } };
const child = Object.create(proto);
console.log("Object.create:", child.greet());

// --- 属性访问 ---
console.log("\n--- 属性访问 ---");
console.log("点语法:", person.name);
console.log("括号语法:", person["name"]);

// 动态属性名
const key = "age";
console.log("动态属性:", person[key]);

// --- 属性简写 ---
console.log("\n--- 属性简写 ---");
const name = "Bob";
const age = 30;
const shorthand = { name, age };  // 等同于 { name: name, age: age }
console.log("简写:", shorthand);

// 方法简写
const calculator = {
    add(a, b) { return a + b; },  // 等同于 add: function(a, b) { ... }
    subtract(a, b) { return a - b; }
};
console.log("方法简写:", calculator.add(5, 3));

// --- 计算属性名 ---
console.log("\n--- 计算属性名 ---");
const propName = "dynamicKey";
const dynamicObj = {
    [propName]: "value",
    ["key" + 1]: "value1",
    [`key${2}`]: "value2"
};
console.log("计算属性名:", dynamicObj);

// --- 属性操作 ---
console.log("\n--- 属性操作 ---");
const user = { name: "Alice" };

// 添加属性
user.email = "alice@example.com";
user["phone"] = "123456";
console.log("添加后:", user);

// 删除属性
delete user.phone;
console.log("删除后:", user);

// 检查属性
console.log("'name' in user:", "name" in user);
console.log("hasOwnProperty:", user.hasOwnProperty("name"));

// --- 对象方法 ---
console.log("\n--- Object 静态方法 ---");
const source = { a: 1, b: 2, c: 3 };

console.log("Object.keys:", Object.keys(source));
console.log("Object.values:", Object.values(source));
console.log("Object.entries:", Object.entries(source));
console.log("Object.fromEntries:", Object.fromEntries([["a", 1], ["b", 2]]));

// Object.assign（浅拷贝/合并）
const target = { x: 1 };
const merged = Object.assign(target, { y: 2 }, { z: 3 });
console.log("Object.assign:", merged);

// Object.freeze / Object.seal
const frozen = Object.freeze({ value: 1 });
// frozen.value = 2;  // 静默失败（严格模式下报错）
console.log("Object.isFrozen:", Object.isFrozen(frozen));

// --- 对象遍历 ---
console.log("\n--- 对象遍历 ---");
const data = { a: 1, b: 2, c: 3 };

// for...in
console.log("for...in:");
for (const key in data) {
    console.log(`  ${key}: ${data[key]}`);
}

// Object.entries + for...of
console.log("Object.entries + for...of:");
for (const [key, value] of Object.entries(data)) {
    console.log(`  ${key}: ${value}`);
}


console.log("\n" + "=".repeat(60));
console.log("2. 数组基础");
console.log("=".repeat(60));

// ============================================================
//                    2. 数组基础
// ============================================================

// --- 创建数组 ---
console.log("\n--- 创建数组 ---");
const arr1 = [1, 2, 3];
const arr2 = new Array(5);  // 长度为 5 的空数组
const arr3 = Array.of(1, 2, 3);
const arr4 = Array.from("hello");  // 从可迭代对象创建
const arr5 = Array.from({ length: 5 }, (_, i) => i * 2);

console.log("字面量:", arr1);
console.log("new Array(5):", arr2);
console.log("Array.of:", arr3);
console.log("Array.from('hello'):", arr4);
console.log("Array.from with map:", arr5);

// --- 访问和修改 ---
console.log("\n--- 访问和修改 ---");
const fruits = ["apple", "banana", "cherry"];
console.log("fruits[0]:", fruits[0]);
console.log("fruits.at(-1):", fruits.at(-1));  // ES2022

fruits[1] = "blueberry";
console.log("修改后:", fruits);

// --- 长度 ---
console.log("\n--- 数组长度 ---");
const nums = [1, 2, 3, 4, 5];
console.log("length:", nums.length);

nums.length = 3;  // 截断数组
console.log("截断后:", nums);

nums.length = 5;  // 扩展数组（填充 undefined）
console.log("扩展后:", nums);


console.log("\n" + "=".repeat(60));
console.log("3. 数组方法 - 增删改");
console.log("=".repeat(60));

// ============================================================
//                    3. 数组方法 - 增删改
// ============================================================

let array = [1, 2, 3, 4, 5];

// --- 添加/删除元素 ---
console.log("\n--- push / pop（末尾）---");
array.push(6);
console.log("push(6):", array);
const popped = array.pop();
console.log("pop():", popped, "->", array);

console.log("\n--- unshift / shift（开头）---");
array.unshift(0);
console.log("unshift(0):", array);
const shifted = array.shift();
console.log("shift():", shifted, "->", array);

// --- splice（任意位置增删改）---
console.log("\n--- splice ---");
let arr = [1, 2, 3, 4, 5];

// 删除
const deleted = arr.splice(2, 1);  // 从索引 2 开始删除 1 个
console.log("删除:", deleted, "->", arr);

// 插入
arr.splice(2, 0, "a", "b");  // 在索引 2 插入
console.log("插入:", arr);

// 替换
arr.splice(2, 2, 3);  // 替换 2 个元素
console.log("替换:", arr);

// --- concat（合并）---
console.log("\n--- concat ---");
const a = [1, 2];
const b = [3, 4];
const c = a.concat(b, [5, 6]);
console.log("concat:", c);

// --- slice（切片，不修改原数组）---
console.log("\n--- slice ---");
const original = [1, 2, 3, 4, 5];
console.log("slice(1, 4):", original.slice(1, 4));
console.log("slice(-2):", original.slice(-2));
console.log("原数组不变:", original);

// --- 填充 ---
console.log("\n--- fill ---");
const fillArr = new Array(5).fill(0);
console.log("fill(0):", fillArr);

const fillPartial = [1, 2, 3, 4, 5];
fillPartial.fill(0, 1, 4);  // 从索引 1 到 4
console.log("fill(0, 1, 4):", fillPartial);


console.log("\n" + "=".repeat(60));
console.log("4. 数组方法 - 查找和排序");
console.log("=".repeat(60));

// ============================================================
//                    4. 数组方法 - 查找和排序
// ============================================================

const numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3];

// --- 查找 ---
console.log("\n--- 查找方法 ---");
console.log("indexOf(5):", numbers.indexOf(5));
console.log("lastIndexOf(5):", numbers.lastIndexOf(5));
console.log("includes(5):", numbers.includes(5));

console.log("find(n > 5):", numbers.find(n => n > 5));
console.log("findIndex(n > 5):", numbers.findIndex(n => n > 5));
console.log("findLast(n > 5):", numbers.findLast(n => n > 5));  // ES2023
console.log("findLastIndex(n > 5):", numbers.findLastIndex(n => n > 5));  // ES2023

// --- 排序 ---
console.log("\n--- 排序方法 ---");
const unsorted = [3, 1, 4, 1, 5, 9, 2, 6];

// 【注意】sort 会修改原数组，默认按字符串排序
const sorted = [...unsorted].sort();
console.log("默认 sort:", sorted);  // 按字符串排序

// 数字排序需要比较函数
const numSorted = [...unsorted].sort((a, b) => a - b);
console.log("数字升序:", numSorted);

const descSorted = [...unsorted].sort((a, b) => b - a);
console.log("数字降序:", descSorted);

// toSorted（ES2023，不修改原数组）
// const newSorted = unsorted.toSorted((a, b) => a - b);

// 对象排序
const users = [
    { name: "Charlie", age: 30 },
    { name: "Alice", age: 25 },
    { name: "Bob", age: 28 }
];
users.sort((a, b) => a.age - b.age);
console.log("按年龄排序:", users.map(u => u.name));

// --- 反转 ---
console.log("\n--- reverse ---");
const toReverse = [1, 2, 3, 4, 5];
console.log("reverse:", [...toReverse].reverse());

// toReversed（ES2023，不修改原数组）
// console.log("toReversed:", toReverse.toReversed());


console.log("\n" + "=".repeat(60));
console.log("5. 数组方法 - 高阶函数");
console.log("=".repeat(60));

// ============================================================
//                    5. 数组方法 - 高阶函数
// ============================================================

const nums2 = [1, 2, 3, 4, 5];

// --- map ---
console.log("\n--- map ---");
const squares = nums2.map(n => n * n);
console.log("平方:", squares);

const objects = nums2.map((n, i) => ({ value: n, index: i }));
console.log("转对象:", objects);

// --- filter ---
console.log("\n--- filter ---");
const evens = nums2.filter(n => n % 2 === 0);
console.log("偶数:", evens);

// --- reduce ---
console.log("\n--- reduce ---");
const sum = nums2.reduce((acc, n) => acc + n, 0);
console.log("求和:", sum);

const product = nums2.reduce((acc, n) => acc * n, 1);
console.log("求积:", product);

// reduce 实现 map
const mapped = nums2.reduce((acc, n) => [...acc, n * 2], []);
console.log("reduce 实现 map:", mapped);

// reduce 分组
const items = [
    { type: "fruit", name: "apple" },
    { type: "vegetable", name: "carrot" },
    { type: "fruit", name: "banana" }
];
const grouped = items.reduce((acc, item) => {
    (acc[item.type] = acc[item.type] || []).push(item.name);
    return acc;
}, {});
console.log("分组:", grouped);

// --- reduceRight ---
console.log("\n--- reduceRight ---");
const concat = ["a", "b", "c"].reduceRight((acc, s) => acc + s, "");
console.log("reduceRight:", concat);

// --- forEach ---
console.log("\n--- forEach ---");
nums2.forEach((n, i) => console.log(`  [${i}] = ${n}`));

// --- some / every ---
console.log("\n--- some / every ---");
console.log("some(n > 4):", nums2.some(n => n > 4));
console.log("every(n > 0):", nums2.every(n => n > 0));

// --- flat / flatMap ---
console.log("\n--- flat / flatMap ---");
const nested = [[1, 2], [3, [4, 5]]];
console.log("flat():", nested.flat());
console.log("flat(2):", nested.flat(2));
console.log("flat(Infinity):", nested.flat(Infinity));

const sentences = ["Hello World", "Foo Bar"];
const wordsArr = sentences.flatMap(s => s.split(" "));
console.log("flatMap:", wordsArr);


console.log("\n" + "=".repeat(60));
console.log("6. 对象和数组的深拷贝");
console.log("=".repeat(60));

// ============================================================
//                    6. 深拷贝
// ============================================================

// --- 浅拷贝 ---
console.log("\n--- 浅拷贝 ---");
const orig = { a: 1, nested: { b: 2 } };
const shallow = { ...orig };
shallow.nested.b = 999;
console.log("原对象受影响:", orig.nested.b);  // 999

// --- 深拷贝方法 ---
console.log("\n--- 深拷贝方法 ---");

// 方法1：JSON（有限制）
const obj1 = { a: 1, b: { c: 2 } };
const deep1 = JSON.parse(JSON.stringify(obj1));
console.log("JSON 方法:", deep1);
console.log("【限制】不支持 undefined, Symbol, 函数, 循环引用");

// 方法2：structuredClone（现代浏览器和 Node.js 17+）
const obj2 = { a: 1, b: { c: 2 }, date: new Date() };
const deep2 = structuredClone(obj2);
deep2.b.c = 999;
console.log("structuredClone:", obj2.b.c);  // 2（原对象不受影响）

// 方法3：递归实现
function deepClone(obj, seen = new WeakMap()) {
    // 处理基本类型
    if (obj === null || typeof obj !== "object") {
        return obj;
    }

    // 处理循环引用
    if (seen.has(obj)) {
        return seen.get(obj);
    }

    // 处理特殊对象
    if (obj instanceof Date) return new Date(obj);
    if (obj instanceof RegExp) return new RegExp(obj);
    if (obj instanceof Map) return new Map([...obj].map(([k, v]) => [k, deepClone(v, seen)]));
    if (obj instanceof Set) return new Set([...obj].map(v => deepClone(v, seen)));

    // 处理数组和普通对象
    const clone = Array.isArray(obj) ? [] : {};
    seen.set(obj, clone);

    for (const key of Reflect.ownKeys(obj)) {
        clone[key] = deepClone(obj[key], seen);
    }

    return clone;
}

const complex = { a: 1, b: [2, 3], c: { d: 4 } };
const cloned = deepClone(complex);
cloned.c.d = 999;
console.log("递归深拷贝:", complex.c.d);  // 4


console.log("\n" + "=".repeat(60));
console.log("7. 实用技巧");
console.log("=".repeat(60));

// ============================================================
//                    7. 实用技巧
// ============================================================

// --- 数组去重 ---
console.log("\n--- 数组去重 ---");
const duplicates = [1, 2, 2, 3, 3, 3, 4];
const unique = [...new Set(duplicates)];
console.log("Set 去重:", unique);

// --- 数组交集/并集/差集 ---
console.log("\n--- 集合运算 ---");
const setA = [1, 2, 3, 4];
const setB = [3, 4, 5, 6];

const union = [...new Set([...setA, ...setB])];
console.log("并集:", union);

const intersection = setA.filter(x => setB.includes(x));
console.log("交集:", intersection);

const difference = setA.filter(x => !setB.includes(x));
console.log("差集:", difference);

// --- 对象过滤 ---
console.log("\n--- 对象过滤 ---");
const objToFilter = { a: 1, b: 2, c: 3, d: 4 };
const filtered = Object.fromEntries(
    Object.entries(objToFilter).filter(([k, v]) => v > 2)
);
console.log("过滤:", filtered);

// --- 对象映射 ---
console.log("\n--- 对象映射 ---");
const objToMap = { a: 1, b: 2, c: 3 };
const mappedObj = Object.fromEntries(
    Object.entries(objToMap).map(([k, v]) => [k, v * 2])
);
console.log("映射:", mappedObj);

// --- 安全访问嵌套属性 ---
console.log("\n--- 安全访问 ---");
const deepObj = { level1: { level2: { value: 42 } } };
const value = deepObj?.level1?.level2?.value ?? "default";
console.log("安全访问:", value);


console.log("\n【总结】");
console.log(`
对象：
- 使用字面量 {} 创建对象
- 解构赋值简化属性访问
- Object.keys/values/entries 获取键值对
- Object.assign 或展开运算符合并对象
- structuredClone 深拷贝

数组：
- 使用 [] 或 Array.from 创建数组
- push/pop 操作末尾，unshift/shift 操作开头
- splice 任意位置增删改
- map/filter/reduce 是最常用的高阶函数
- sort 需要比较函数才能正确排序数字
- Set 用于去重
`);
```
