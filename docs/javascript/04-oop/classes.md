# classes.js

::: info 文件信息
- 📄 原文件：`01_classes.js`
- 🔤 语言：javascript
:::

============================================================
               JavaScript 类与面向对象
============================================================
本文件介绍 JavaScript 中的类和面向对象编程。
============================================================

## 完整代码

```javascript
/**
 * ============================================================
 *                JavaScript 类与面向对象
 * ============================================================
 * 本文件介绍 JavaScript 中的类和面向对象编程。
 * ============================================================
 */

console.log("=".repeat(60));
console.log("1. 类基础");
console.log("=".repeat(60));

// ============================================================
//                    1. 类基础
// ============================================================

// --- 类声明 ---
console.log("\n--- 类声明 ---");

class Person {
    // 构造函数
    constructor(name, age) {
        this.name = name;
        this.age = age;
    }

    // 实例方法
    introduce() {
        return `我是 ${this.name}，${this.age} 岁`;
    }

    // getter
    get info() {
        return `${this.name} (${this.age})`;
    }

    // setter
    set info(value) {
        const [name, age] = value.split(",");
        this.name = name;
        this.age = parseInt(age);
    }
}

const alice = new Person("Alice", 25);
console.log("实例:", alice);
console.log("方法:", alice.introduce());
console.log("getter:", alice.info);

alice.info = "Bob,30";
console.log("setter 后:", alice.info);

// --- 类表达式 ---
console.log("\n--- 类表达式 ---");

const Animal = class {
    constructor(name) {
        this.name = name;
    }

    speak() {
        console.log(`${this.name} 发出声音`);
    }
};

const dog = new Animal("Dog");
dog.speak();


console.log("\n" + "=".repeat(60));
console.log("2. 静态成员");
console.log("=".repeat(60));

// ============================================================
//                    2. 静态成员
// ============================================================

class MathUtils {
    // 静态属性
    static PI = 3.14159;

    // 静态方法
    static add(a, b) {
        return a + b;
    }

    static multiply(a, b) {
        return a * b;
    }

    // 静态工厂方法
    static createCalculator() {
        return {
            add: this.add,
            multiply: this.multiply
        };
    }
}

console.log("\n--- 静态成员 ---");
console.log("MathUtils.PI:", MathUtils.PI);
console.log("MathUtils.add(3, 5):", MathUtils.add(3, 5));

// 实例不能访问静态成员
// const math = new MathUtils();
// console.log(math.PI);  // undefined


console.log("\n" + "=".repeat(60));
console.log("3. 继承");
console.log("=".repeat(60));

// ============================================================
//                    3. 继承
// ============================================================

class Animal2 {
    constructor(name) {
        this.name = name;
    }

    speak() {
        console.log(`${this.name} 发出声音`);
    }

    move(distance) {
        console.log(`${this.name} 移动了 ${distance}m`);
    }
}

class Dog2 extends Animal2 {
    constructor(name, breed) {
        super(name);  // 调用父类构造函数
        this.breed = breed;
    }

    // 重写方法
    speak() {
        console.log(`${this.name} 汪汪叫`);
    }

    // 调用父类方法
    speakLoud() {
        super.speak();  // 调用父类的 speak
        console.log("（很大声）");
    }

    // 新方法
    fetch() {
        console.log(`${this.name} 去捡球`);
    }
}

console.log("\n--- 继承 ---");
const buddy = new Dog2("Buddy", "Golden Retriever");
console.log("实例:", buddy);
buddy.speak();
buddy.move(10);
buddy.fetch();
buddy.speakLoud();

// instanceof 检查
console.log("\n--- instanceof ---");
console.log("buddy instanceof Dog2:", buddy instanceof Dog2);
console.log("buddy instanceof Animal2:", buddy instanceof Animal2);
console.log("buddy instanceof Object:", buddy instanceof Object);


console.log("\n" + "=".repeat(60));
console.log("4. 私有成员");
console.log("=".repeat(60));

// ============================================================
//                    4. 私有成员
// ============================================================

class BankAccount {
    // 私有字段（# 开头）
    #balance = 0;
    #transactions = [];

    constructor(owner, initialBalance = 0) {
        this.owner = owner;
        this.#balance = initialBalance;
        this.#log("账户创建");
    }

    // 私有方法
    #log(message) {
        this.#transactions.push({
            message,
            time: new Date().toISOString()
        });
    }

    deposit(amount) {
        if (amount > 0) {
            this.#balance += amount;
            this.#log(`存入 ${amount}`);
            return true;
        }
        return false;
    }

    withdraw(amount) {
        if (amount > 0 && amount <= this.#balance) {
            this.#balance -= amount;
            this.#log(`取出 ${amount}`);
            return true;
        }
        return false;
    }

    get balance() {
        return this.#balance;
    }

    getHistory() {
        return [...this.#transactions];  // 返回副本
    }
}

console.log("\n--- 私有成员 ---");
const account = new BankAccount("Alice", 1000);
console.log("初始余额:", account.balance);

account.deposit(500);
console.log("存入后:", account.balance);

account.withdraw(200);
console.log("取出后:", account.balance);

// 无法访问私有成员
// console.log(account.#balance);  // 语法错误
// account.#log("hack");  // 语法错误

console.log("交易记录:", account.getHistory());


console.log("\n" + "=".repeat(60));
console.log("5. 抽象模式");
console.log("=".repeat(60));

// ============================================================
//                    5. 抽象模式
// ============================================================

/**
 * JavaScript 没有原生抽象类，但可以模拟
 */

class Shape {
    constructor() {
        if (new.target === Shape) {
            throw new Error("Shape 是抽象类，不能直接实例化");
        }
    }

    // 抽象方法
    area() {
        throw new Error("子类必须实现 area 方法");
    }

    perimeter() {
        throw new Error("子类必须实现 perimeter 方法");
    }

    // 具体方法
    describe() {
        return `面积: ${this.area()}, 周长: ${this.perimeter()}`;
    }
}

class Rectangle extends Shape {
    constructor(width, height) {
        super();
        this.width = width;
        this.height = height;
    }

    area() {
        return this.width * this.height;
    }

    perimeter() {
        return 2 * (this.width + this.height);
    }
}

class Circle extends Shape {
    constructor(radius) {
        super();
        this.radius = radius;
    }

    area() {
        return Math.PI * this.radius ** 2;
    }

    perimeter() {
        return 2 * Math.PI * this.radius;
    }
}

console.log("\n--- 抽象模式 ---");

// const shape = new Shape();  // 错误：不能实例化

const rect = new Rectangle(4, 5);
const circle = new Circle(3);

console.log("矩形:", rect.describe());
console.log("圆形:", circle.describe());


console.log("\n" + "=".repeat(60));
console.log("6. 混入 (Mixin)");
console.log("=".repeat(60));

// ============================================================
//                    6. 混入
// ============================================================

/**
 * JavaScript 不支持多继承，但可以使用混入模式
 */

// 定义混入
const Flyable = {
    fly() {
        console.log(`${this.name} 正在飞行`);
    },
    land() {
        console.log(`${this.name} 已着陆`);
    }
};

const Swimmable = {
    swim() {
        console.log(`${this.name} 正在游泳`);
    },
    dive() {
        console.log(`${this.name} 正在潜水`);
    }
};

// 使用混入
class Duck {
    constructor(name) {
        this.name = name;
    }

    quack() {
        console.log(`${this.name}: 嘎嘎！`);
    }
}

// 混入方法
Object.assign(Duck.prototype, Flyable, Swimmable);

console.log("\n--- 混入 ---");
const duck = new Duck("Donald");
duck.quack();
duck.fly();
duck.swim();

// 函数式混入
function withTimestamp(target) {
    target.prototype.getTimestamp = function() {
        return new Date().toISOString();
    };
    return target;
}

@withTimestamp  // 需要装饰器支持
class Event {
    constructor(name) {
        this.name = name;
    }
}

// 手动应用
withTimestamp(Duck);
console.log("时间戳:", duck.getTimestamp());


console.log("\n" + "=".repeat(60));
console.log("7. 原型链");
console.log("=".repeat(60));

// ============================================================
//                    7. 原型链
// ============================================================

console.log("\n--- 类的原型链 ---");
console.log("Dog2.prototype:", Dog2.prototype);
console.log("buddy.__proto__ === Dog2.prototype:", buddy.__proto__ === Dog2.prototype);
console.log("Dog2.prototype.__proto__ === Animal2.prototype:",
    Dog2.prototype.__proto__ === Animal2.prototype);

// 方法在原型上
console.log("\n--- 方法位置 ---");
console.log("hasOwnProperty 'name':", buddy.hasOwnProperty("name"));  // true，在实例上
console.log("hasOwnProperty 'speak':", buddy.hasOwnProperty("speak"));  // false，在原型上

// 检查原型链
console.log("\n--- 原型链检查 ---");
console.log("Object.getPrototypeOf(buddy):", Object.getPrototypeOf(buddy) === Dog2.prototype);


console.log("\n【总结】");
console.log(`
类基础：
- class 声明定义类
- constructor 定义构造函数
- 实例方法、getter、setter

静态成员：
- static 关键字
- 属于类本身，不属于实例

继承：
- extends 关键字
- super 调用父类
- 可以重写方法

私有成员：
- # 开头的字段和方法
- 类外部无法访问

设计模式：
- 抽象类模式
- 混入模式
- 工厂模式

原型链：
- 类是原型继承的语法糖
- 方法在 prototype 上
- 实例属性在实例上
`);
```
