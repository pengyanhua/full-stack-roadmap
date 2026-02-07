# types.ts

::: info 文件信息
- 📄 原文件：`01_types.ts`
- 🔤 语言：typescript
:::

============================================================
               TypeScript 类型系统
============================================================
本文件介绍 TypeScript 的基本类型和类型注解。
运行方式：
npx ts-node 01_types.ts
或
tsc 01_types.ts && node 01_types.js
============================================================

## 完整代码

```typescript
/**
 * ============================================================
 *                TypeScript 类型系统
 * ============================================================
 * 本文件介绍 TypeScript 的基本类型和类型注解。
 *
 * 运行方式：
 * npx ts-node 01_types.ts
 * 或
 * tsc 01_types.ts && node 01_types.js
 * ============================================================
 */

console.log("=".repeat(60));
console.log("1. 基本类型");
console.log("=".repeat(60));

// ============================================================
//                    1. 基本类型
// ============================================================

// --- 类型注解 ---
let message: string = "Hello, TypeScript!";
let count: number = 42;
let isActive: boolean = true;

console.log("\n--- 基本类型 ---");
console.log("string:", message);
console.log("number:", count);
console.log("boolean:", isActive);

// --- 类型推断 ---
let inferred = "TypeScript 会推断这是 string";
// inferred = 123;  // 错误：不能将 number 分配给 string

// --- 特殊类型 ---
console.log("\n--- 特殊类型 ---");

// null 和 undefined
let nullValue: null = null;
let undefinedValue: undefined = undefined;

// any - 任意类型（谨慎使用）
let anyValue: any = "可以是任何值";
anyValue = 123;
anyValue = { key: "value" };
console.log("any:", anyValue);

// unknown - 安全的 any
let unknownValue: unknown = "需要类型检查才能使用";
// unknownValue.length;  // 错误：必须先检查类型
if (typeof unknownValue === "string") {
    console.log("unknown string length:", unknownValue.length);
}

// void - 无返回值
function logMessage(msg: string): void {
    console.log("void 函数:", msg);
}
logMessage("Hello");

// never - 永不返回
function throwError(msg: string): never {
    throw new Error(msg);
}

// --- 数组 ---
console.log("\n--- 数组类型 ---");
const numbers: number[] = [1, 2, 3, 4, 5];
const strings: Array<string> = ["a", "b", "c"];

console.log("number[]:", numbers);
console.log("Array<string>:", strings);

// 只读数组
const readonlyArr: readonly number[] = [1, 2, 3];
// readonlyArr.push(4);  // 错误：readonly

// --- 元组 ---
console.log("\n--- 元组 ---");
const tuple: [string, number, boolean] = ["Alice", 25, true];
console.log("tuple:", tuple);
console.log("tuple[0]:", tuple[0]);  // string
console.log("tuple[1]:", tuple[1]);  // number

// 具名元组
const namedTuple: [name: string, age: number] = ["Bob", 30];


console.log("\n" + "=".repeat(60));
console.log("2. 对象类型");
console.log("=".repeat(60));

// ============================================================
//                    2. 对象类型
// ============================================================

// --- 内联对象类型 ---
console.log("\n--- 内联对象类型 ---");
const person: {
    name: string;
    age: number;
    email?: string;  // 可选属性
} = {
    name: "Alice",
    age: 25
};
console.log("person:", person);

// --- 接口 ---
console.log("\n--- 接口 ---");

interface User {
    id: number;
    name: string;
    email?: string;
    readonly createdAt: Date;
}

const user: User = {
    id: 1,
    name: "Alice",
    createdAt: new Date()
};
console.log("user:", user);
// user.createdAt = new Date();  // 错误：readonly

// 接口继承
interface Admin extends User {
    role: string;
    permissions: string[];
}

const admin: Admin = {
    id: 2,
    name: "Bob",
    createdAt: new Date(),
    role: "super_admin",
    permissions: ["read", "write", "delete"]
};
console.log("admin:", admin);

// --- 类型别名 ---
console.log("\n--- 类型别名 ---");

type Point = {
    x: number;
    y: number;
};

type ID = string | number;

const point: Point = { x: 10, y: 20 };
const userId: ID = "abc123";
const postId: ID = 42;

console.log("point:", point);
console.log("userId:", userId);
console.log("postId:", postId);

// 类型别名 vs 接口
console.log(`
类型别名 vs 接口：
- 接口可以 extends，类型别名用 &
- 接口可以声明合并
- 类型别名可以表示联合类型、元组等
- 推荐：对象用接口，其他用类型别名
`);


console.log("\n" + "=".repeat(60));
console.log("3. 联合类型与交叉类型");
console.log("=".repeat(60));

// ============================================================
//                    3. 联合类型与交叉类型
// ============================================================

// --- 联合类型 ---
console.log("\n--- 联合类型 (|) ---");

type StringOrNumber = string | number;

function printId(id: StringOrNumber) {
    if (typeof id === "string") {
        console.log("  字符串 ID:", id.toUpperCase());
    } else {
        console.log("  数字 ID:", id);
    }
}

printId("abc");
printId(123);

// 字面量联合类型
type Direction = "up" | "down" | "left" | "right";
type HttpStatus = 200 | 201 | 400 | 404 | 500;

function move(direction: Direction) {
    console.log(`  移动: ${direction}`);
}
move("up");

// --- 交叉类型 ---
console.log("\n--- 交叉类型 (&) ---");

type HasName = { name: string };
type HasAge = { age: number };
type PersonType = HasName & HasAge;

const personWithBoth: PersonType = {
    name: "Alice",
    age: 25
};
console.log("交叉类型:", personWithBoth);

// --- 类型收窄 ---
console.log("\n--- 类型收窄 ---");

function processValue(value: string | number | null) {
    // typeof 收窄
    if (typeof value === "string") {
        return value.toUpperCase();
    }

    // 真值收窄
    if (value) {
        return value * 2;
    }

    return "null value";
}

console.log("processValue('hello'):", processValue("hello"));
console.log("processValue(21):", processValue(21));
console.log("processValue(null):", processValue(null));


console.log("\n" + "=".repeat(60));
console.log("4. 函数类型");
console.log("=".repeat(60));

// ============================================================
//                    4. 函数类型
// ============================================================

// --- 函数类型注解 ---
console.log("\n--- 函数类型 ---");

function add(a: number, b: number): number {
    return a + b;
}

const multiply: (a: number, b: number) => number = (a, b) => a * b;

console.log("add(3, 5):", add(3, 5));
console.log("multiply(3, 5):", multiply(3, 5));

// --- 可选参数和默认参数 ---
console.log("\n--- 可选和默认参数 ---");

function greet(name: string, greeting: string = "Hello", punctuation?: string): string {
    return `${greeting}, ${name}${punctuation || "!"}`;
}

console.log(greet("Alice"));
console.log(greet("Bob", "Hi"));
console.log(greet("Charlie", "Hey", "?"));

// --- 剩余参数 ---
function sum(...numbers: number[]): number {
    return numbers.reduce((a, b) => a + b, 0);
}
console.log("sum(1,2,3,4,5):", sum(1, 2, 3, 4, 5));

// --- 函数重载 ---
console.log("\n--- 函数重载 ---");

function format(value: string): string;
function format(value: number): string;
function format(value: string | number): string {
    if (typeof value === "string") {
        return value.toUpperCase();
    }
    return value.toFixed(2);
}

console.log("format('hello'):", format("hello"));
console.log("format(3.14159):", format(3.14159));

// --- 泛型函数 ---
console.log("\n--- 泛型函数 ---");

function identity<T>(value: T): T {
    return value;
}

console.log("identity<string>:", identity<string>("hello"));
console.log("identity<number>:", identity<number>(42));

// 泛型约束
function getLength<T extends { length: number }>(value: T): number {
    return value.length;
}

console.log("getLength([1,2,3]):", getLength([1, 2, 3]));
console.log("getLength('hello'):", getLength("hello"));


console.log("\n" + "=".repeat(60));
console.log("5. 类型断言与类型守卫");
console.log("=".repeat(60));

// ============================================================
//                    5. 类型断言与类型守卫
// ============================================================

// --- 类型断言 ---
console.log("\n--- 类型断言 ---");

const someValue: unknown = "hello";

// as 语法
const strLength1: number = (someValue as string).length;

// 尖括号语法（JSX 中不可用）
const strLength2: number = (<string>someValue).length;

console.log("strLength:", strLength1);

// --- 类型守卫 ---
console.log("\n--- 类型守卫 ---");

interface Cat {
    meow(): void;
}

interface Dog {
    bark(): void;
}

// 自定义类型守卫
function isCat(animal: Cat | Dog): animal is Cat {
    return "meow" in animal;
}

function makeSound(animal: Cat | Dog) {
    if (isCat(animal)) {
        animal.meow();
    } else {
        animal.bark();
    }
}

const cat: Cat = { meow: () => console.log("  喵~") };
const dog: Dog = { bark: () => console.log("  汪!") };

makeSound(cat);
makeSound(dog);

// --- 可辨识联合 ---
console.log("\n--- 可辨识联合 ---");

type Shape =
    | { kind: "circle"; radius: number }
    | { kind: "rectangle"; width: number; height: number }
    | { kind: "triangle"; base: number; height: number };

function calculateArea(shape: Shape): number {
    switch (shape.kind) {
        case "circle":
            return Math.PI * shape.radius ** 2;
        case "rectangle":
            return shape.width * shape.height;
        case "triangle":
            return (shape.base * shape.height) / 2;
    }
}

console.log("圆面积:", calculateArea({ kind: "circle", radius: 5 }));
console.log("矩形面积:", calculateArea({ kind: "rectangle", width: 4, height: 5 }));


console.log("\n【总结】");
console.log(`
基本类型：
- string, number, boolean
- null, undefined
- any（避免使用）, unknown（安全版 any）
- void, never

复合类型：
- 数组: number[] 或 Array<number>
- 元组: [string, number]
- 对象: { name: string }

接口与类型别名：
- interface 用于对象
- type 用于联合、元组等

高级类型：
- 联合类型: A | B
- 交叉类型: A & B
- 字面量类型: "up" | "down"

类型安全：
- 类型守卫
- 类型断言
- 可辨识联合
`);
```
