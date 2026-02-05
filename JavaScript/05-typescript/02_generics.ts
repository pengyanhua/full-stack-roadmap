/**
 * ============================================================
 *                TypeScript 泛型
 * ============================================================
 * 本文件介绍 TypeScript 中的泛型编程。
 * ============================================================
 */

console.log("=".repeat(60));
console.log("1. 泛型基础");
console.log("=".repeat(60));

// ============================================================
//                    1. 泛型基础
// ============================================================

// --- 泛型函数 ---
console.log("\n--- 泛型函数 ---");

// 没有泛型 - 类型信息丢失
function identityAny(value: any): any {
    return value;
}

// 使用泛型 - 保留类型信息
function identity<T>(value: T): T {
    return value;
}

// 显式指定类型
const str = identity<string>("hello");
console.log("identity<string>:", str);

// 类型推断
const num = identity(42);
console.log("identity(42):", num);

// --- 多个类型参数 ---
console.log("\n--- 多个类型参数 ---");

function pair<T, U>(first: T, second: U): [T, U] {
    return [first, second];
}

const result = pair<string, number>("age", 25);
console.log("pair:", result);

function swap<T, U>(tuple: [T, U]): [U, T] {
    return [tuple[1], tuple[0]];
}

console.log("swap:", swap([1, "one"]));

// --- 泛型约束 ---
console.log("\n--- 泛型约束 ---");

interface Lengthwise {
    length: number;
}

function logLength<T extends Lengthwise>(value: T): T {
    console.log(`  长度: ${value.length}`);
    return value;
}

logLength("hello");
logLength([1, 2, 3]);
// logLength(123);  // 错误：number 没有 length 属性

// 使用 keyof
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
    return obj[key];
}

const person = { name: "Alice", age: 25 };
console.log("getProperty:", getProperty(person, "name"));
// getProperty(person, "email");  // 错误：email 不是 person 的键


console.log("\n" + "=".repeat(60));
console.log("2. 泛型接口和类型");
console.log("=".repeat(60));

// ============================================================
//                    2. 泛型接口和类型
// ============================================================

// --- 泛型接口 ---
console.log("\n--- 泛型接口 ---");

interface Container<T> {
    value: T;
    getValue(): T;
    setValue(value: T): void;
}

class Box<T> implements Container<T> {
    constructor(public value: T) {}

    getValue(): T {
        return this.value;
    }

    setValue(value: T): void {
        this.value = value;
    }
}

const stringBox = new Box<string>("hello");
console.log("stringBox:", stringBox.getValue());

const numberBox = new Box(42);  // 类型推断
console.log("numberBox:", numberBox.getValue());

// --- 泛型类型别名 ---
console.log("\n--- 泛型类型别名 ---");

type Result<T, E = Error> = {
    success: true;
    data: T;
} | {
    success: false;
    error: E;
};

function divide(a: number, b: number): Result<number> {
    if (b === 0) {
        return { success: false, error: new Error("除数不能为零") };
    }
    return { success: true, data: a / b };
}

const divResult = divide(10, 2);
if (divResult.success) {
    console.log("结果:", divResult.data);
} else {
    console.log("错误:", divResult.error.message);
}

// --- 泛型默认类型 ---
console.log("\n--- 泛型默认类型 ---");

interface ApiResponse<T = any> {
    code: number;
    message: string;
    data: T;
}

const response1: ApiResponse<string[]> = {
    code: 200,
    message: "success",
    data: ["a", "b", "c"]
};

const response2: ApiResponse = {  // 使用默认类型 any
    code: 200,
    message: "success",
    data: { key: "value" }
};

console.log("response1:", response1);
console.log("response2:", response2);


console.log("\n" + "=".repeat(60));
console.log("3. 泛型类");
console.log("=".repeat(60));

// ============================================================
//                    3. 泛型类
// ============================================================

// --- 泛型栈 ---
console.log("\n--- 泛型栈 ---");

class Stack<T> {
    private items: T[] = [];

    push(item: T): void {
        this.items.push(item);
    }

    pop(): T | undefined {
        return this.items.pop();
    }

    peek(): T | undefined {
        return this.items[this.items.length - 1];
    }

    isEmpty(): boolean {
        return this.items.length === 0;
    }

    size(): number {
        return this.items.length;
    }
}

const numberStack = new Stack<number>();
numberStack.push(1);
numberStack.push(2);
numberStack.push(3);

console.log("peek:", numberStack.peek());
console.log("pop:", numberStack.pop());
console.log("size:", numberStack.size());

// --- 泛型队列 ---
console.log("\n--- 泛型队列 ---");

class Queue<T> {
    private items: T[] = [];

    enqueue(item: T): void {
        this.items.push(item);
    }

    dequeue(): T | undefined {
        return this.items.shift();
    }

    front(): T | undefined {
        return this.items[0];
    }

    isEmpty(): boolean {
        return this.items.length === 0;
    }
}

const taskQueue = new Queue<string>();
taskQueue.enqueue("任务1");
taskQueue.enqueue("任务2");
taskQueue.enqueue("任务3");

console.log("front:", taskQueue.front());
console.log("dequeue:", taskQueue.dequeue());
console.log("front after dequeue:", taskQueue.front());


console.log("\n" + "=".repeat(60));
console.log("4. 工具类型");
console.log("=".repeat(60));

// ============================================================
//                    4. 工具类型
// ============================================================

interface User {
    id: number;
    name: string;
    email: string;
    age: number;
}

// --- Partial<T> ---
console.log("\n--- Partial<T> ---");
type PartialUser = Partial<User>;

function updateUser(id: number, updates: Partial<User>): void {
    console.log(`更新用户 ${id}:`, updates);
}

updateUser(1, { name: "Bob" });  // 只更新部分字段

// --- Required<T> ---
console.log("\n--- Required<T> ---");
interface Config {
    host?: string;
    port?: number;
}

type RequiredConfig = Required<Config>;
const config: RequiredConfig = {
    host: "localhost",
    port: 3000  // 必须提供
};
console.log("config:", config);

// --- Readonly<T> ---
console.log("\n--- Readonly<T> ---");
type ReadonlyUser = Readonly<User>;

const user: ReadonlyUser = {
    id: 1,
    name: "Alice",
    email: "alice@example.com",
    age: 25
};
// user.name = "Bob";  // 错误：只读

// --- Pick<T, K> ---
console.log("\n--- Pick<T, K> ---");
type UserBasic = Pick<User, "id" | "name">;

const basic: UserBasic = {
    id: 1,
    name: "Alice"
};
console.log("picked:", basic);

// --- Omit<T, K> ---
console.log("\n--- Omit<T, K> ---");
type UserWithoutEmail = Omit<User, "email">;

const noEmail: UserWithoutEmail = {
    id: 1,
    name: "Alice",
    age: 25
};
console.log("omitted:", noEmail);

// --- Record<K, T> ---
console.log("\n--- Record<K, T> ---");
type Status = "pending" | "active" | "completed";
type StatusCount = Record<Status, number>;

const counts: StatusCount = {
    pending: 5,
    active: 10,
    completed: 20
};
console.log("record:", counts);

// --- Exclude<T, U> 和 Extract<T, U> ---
console.log("\n--- Exclude 和 Extract ---");
type AllTypes = string | number | boolean | null | undefined;

type NonNullable2 = Exclude<AllTypes, null | undefined>;
// 结果: string | number | boolean

type OnlyPrimitives = Extract<AllTypes, string | number>;
// 结果: string | number

// --- ReturnType<T> ---
console.log("\n--- ReturnType<T> ---");
function createUser2(): { id: number; name: string } {
    return { id: 1, name: "Alice" };
}

type CreatedUser = ReturnType<typeof createUser2>;
console.log("ReturnType:", {} as CreatedUser);

// --- Parameters<T> ---
console.log("\n--- Parameters<T> ---");
function greet2(name: string, age: number): string {
    return `Hello, ${name}! You are ${age}.`;
}

type GreetParams = Parameters<typeof greet2>;
// 结果: [string, number]


console.log("\n" + "=".repeat(60));
console.log("5. 条件类型");
console.log("=".repeat(60));

// ============================================================
//                    5. 条件类型
// ============================================================

// --- 基本条件类型 ---
console.log("\n--- 条件类型 ---");

type IsString<T> = T extends string ? true : false;

type A = IsString<string>;  // true
type B = IsString<number>;  // false

// --- infer 关键字 ---
type GetReturnType<T> = T extends (...args: any[]) => infer R ? R : never;

function foo(): string { return "hello"; }
function bar(): number { return 42; }

type FooReturn = GetReturnType<typeof foo>;  // string
type BarReturn = GetReturnType<typeof bar>;  // number

// 获取数组元素类型
type ArrayElement<T> = T extends (infer E)[] ? E : T;

type NumElement = ArrayElement<number[]>;  // number
type StrElement = ArrayElement<string[]>;  // string

// --- 分布式条件类型 ---
type ToArray<T> = T extends any ? T[] : never;

type StrOrNumArray = ToArray<string | number>;
// 结果: string[] | number[]


console.log("\n【总结】");
console.log(`
泛型基础：
- <T> 声明类型参数
- 可以有多个类型参数 <T, U>
- extends 添加约束
- keyof 获取键类型

泛型使用场景：
- 函数：function fn<T>(arg: T): T
- 接口：interface Box<T> { value: T }
- 类：class Stack<T> {}
- 类型别名：type Result<T> = ...

内置工具类型：
- Partial<T> - 所有属性可选
- Required<T> - 所有属性必需
- Readonly<T> - 所有属性只读
- Pick<T, K> - 选取部分属性
- Omit<T, K> - 排除部分属性
- Record<K, T> - 键值对类型
- Exclude<T, U> - 从联合类型排除
- Extract<T, U> - 从联合类型提取
- ReturnType<T> - 函数返回类型
- Parameters<T> - 函数参数类型

条件类型：
- T extends U ? X : Y
- infer 推断类型
`);
