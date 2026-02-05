/**
 * ============================================================
 *                Node.js 模块与事件
 * ============================================================
 * 本文件介绍 Node.js 中的模块系统和事件机制。
 * ============================================================
 */

const EventEmitter = require("events");
const path = require("path");

console.log("=".repeat(60));
console.log("1. CommonJS 模块系统");
console.log("=".repeat(60));

// ============================================================
//                    1. CommonJS 模块系统
// ============================================================

/**
 * CommonJS 是 Node.js 的默认模块系统
 * - require() 导入模块
 * - module.exports 导出模块
 * - exports 是 module.exports 的别名
 */

console.log("\n--- 模块导出方式 ---");

// 方式1：导出单个值
// module.exports = function() { ... };

// 方式2：导出对象
// module.exports = { fn1, fn2 };

// 方式3：使用 exports 快捷方式
// exports.fn1 = function() { ... };
// exports.fn2 = function() { ... };

// --- 模块查找规则 ---
console.log("\n--- 模块查找规则 ---");
console.log(`
1. 核心模块（fs, path, http 等）
   - 直接加载，优先级最高

2. 文件模块（以 ./ 或 ../ 开头）
   - 精确匹配：./module.js
   - 补全扩展名：./module -> ./module.js, ./module.json, ./module.node
   - 目录模块：./dir -> ./dir/package.json(main) 或 ./dir/index.js

3. node_modules 模块
   - 从当前目录向上查找 node_modules 文件夹
`);

// --- 模块缓存 ---
console.log("\n--- 模块缓存 ---");
console.log("模块在首次 require 时被缓存");
console.log("后续 require 返回缓存的模块");
console.log("缓存存储在 require.cache 中");

// 查看缓存（示例）
// console.log(Object.keys(require.cache));

// 清除缓存（热重载时使用）
// delete require.cache[require.resolve('./myModule')];

// --- 模块信息 ---
console.log("\n--- 模块信息 ---");
console.log("__filename:", __filename);
console.log("__dirname:", __dirname);
console.log("module.id:", module.id);


console.log("\n" + "=".repeat(60));
console.log("2. ES 模块（ESM）");
console.log("=".repeat(60));

// ============================================================
//                    2. ES 模块
// ============================================================

/**
 * ES 模块是 JavaScript 的标准模块系统
 *
 * 启用方式：
 * 1. 文件使用 .mjs 扩展名
 * 2. package.json 中设置 "type": "module"
 *
 * 语法：
 * - import { fn } from './module.js'
 * - import * as module from './module.js'
 * - import defaultExport from './module.js'
 * - export const fn = () => {}
 * - export default function() {}
 */

console.log("\n--- ESM vs CommonJS ---");
console.log(`
CommonJS:
- require() 同步加载
- 动态导入（运行时解析）
- this 指向 module.exports
- 可以条件导入

ES Modules:
- import 异步加载
- 静态导入（编译时解析）
- this 是 undefined
- 顶层 await 支持
- 更好的 tree shaking
`);

// 在 CommonJS 中使用 ESM
console.log("\n--- 动态导入 ESM ---");
async function loadESM() {
    // 使用动态 import() 加载 ESM
    // const module = await import('./esm-module.mjs');
    console.log("动态导入使用: const mod = await import('./module.mjs')");
}
loadESM();


console.log("\n" + "=".repeat(60));
console.log("3. EventEmitter 事件发射器");
console.log("=".repeat(60));

// ============================================================
//                    3. EventEmitter
// ============================================================

/**
 * EventEmitter 是 Node.js 事件驱动架构的核心
 */

console.log("\n--- 基本用法 ---");

const emitter = new EventEmitter();

// 监听事件
emitter.on("message", (data) => {
    console.log("收到消息:", data);
});

// 一次性监听
emitter.once("connect", () => {
    console.log("连接事件（只触发一次）");
});

// 触发事件
emitter.emit("message", "Hello!");
emitter.emit("connect");
emitter.emit("connect");  // 不会触发

// --- 事件参数 ---
console.log("\n--- 多参数事件 ---");

emitter.on("data", (a, b, c) => {
    console.log(`data 事件: a=${a}, b=${b}, c=${c}`);
});

emitter.emit("data", 1, 2, 3);

// --- 移除监听器 ---
console.log("\n--- 移除监听器 ---");

const handler = (msg) => console.log("handler:", msg);

emitter.on("test", handler);
emitter.emit("test", "第一次");

emitter.off("test", handler);  // 或 removeListener
emitter.emit("test", "第二次");  // 不会输出

// --- 监听器数量 ---
console.log("\n--- 监听器管理 ---");

emitter.on("event1", () => {});
emitter.on("event1", () => {});
emitter.on("event2", () => {});

console.log("event1 监听器数量:", emitter.listenerCount("event1"));
console.log("所有事件名:", emitter.eventNames());

// --- 最大监听器警告 ---
console.log("\n--- 最大监听器数量 ---");

// 默认每个事件最多 10 个监听器
emitter.setMaxListeners(20);
console.log("最大监听器数:", emitter.getMaxListeners());


console.log("\n" + "=".repeat(60));
console.log("4. 自定义事件发射器");
console.log("=".repeat(60));

// ============================================================
//                    4. 自定义事件发射器
// ============================================================

/**
 * 数据库连接类（继承 EventEmitter）
 */
class Database extends EventEmitter {
    constructor(name) {
        super();
        this.name = name;
        this.connected = false;
    }

    connect() {
        console.log(`\n连接数据库: ${this.name}...`);

        // 模拟异步连接
        setTimeout(() => {
            this.connected = true;
            this.emit("connect", { database: this.name });
        }, 100);

        return this;
    }

    query(sql) {
        if (!this.connected) {
            this.emit("error", new Error("未连接数据库"));
            return this;
        }

        console.log(`执行查询: ${sql}`);

        // 模拟查询
        setTimeout(() => {
            const results = [
                { id: 1, name: "Alice" },
                { id: 2, name: "Bob" }
            ];
            this.emit("data", results);
            this.emit("end");
        }, 50);

        return this;
    }

    disconnect() {
        console.log(`断开数据库: ${this.name}`);
        this.connected = false;
        this.emit("disconnect");
        return this;
    }
}

// 使用自定义事件发射器
const db = new Database("mydb");

db.on("connect", (info) => {
    console.log("已连接:", info);
    db.query("SELECT * FROM users");
});

db.on("data", (results) => {
    console.log("查询结果:", results);
});

db.on("end", () => {
    console.log("查询完成");
    db.disconnect();
});

db.on("disconnect", () => {
    console.log("已断开连接");
});

db.on("error", (err) => {
    console.error("数据库错误:", err.message);
});

db.connect();


console.log("\n" + "=".repeat(60));
console.log("5. 事件模式与最佳实践");
console.log("=".repeat(60));

// ============================================================
//                    5. 事件模式与最佳实践
// ============================================================

/**
 * 发布-订阅模式实现
 */
class PubSub {
    constructor() {
        this.emitter = new EventEmitter();
        this.emitter.setMaxListeners(0);  // 无限制
    }

    subscribe(channel, callback) {
        this.emitter.on(channel, callback);
        return () => this.emitter.off(channel, callback);
    }

    publish(channel, data) {
        this.emitter.emit(channel, data);
    }

    subscribeOnce(channel, callback) {
        this.emitter.once(channel, callback);
    }
}

console.log("\n--- 发布-订阅模式 ---");

const pubsub = new PubSub();

const unsubscribe = pubsub.subscribe("news", (article) => {
    console.log("新闻:", article.title);
});

pubsub.publish("news", { title: "今日头条" });
pubsub.publish("news", { title: "热点新闻" });

unsubscribe();
pubsub.publish("news", { title: "这条不会收到" });

/**
 * 异步事件处理
 */
console.log("\n--- 异步事件处理 ---");

class AsyncEmitter extends EventEmitter {
    async emitAsync(event, ...args) {
        const listeners = this.listeners(event);
        for (const listener of listeners) {
            await listener(...args);
        }
    }
}

const asyncEmitter = new AsyncEmitter();

asyncEmitter.on("task", async (data) => {
    console.log("开始处理:", data);
    await new Promise(r => setTimeout(r, 100));
    console.log("处理完成:", data);
});

// 等待异步事件完成
(async () => {
    await asyncEmitter.emitAsync("task", "任务1");
    console.log("所有异步处理完成");
})();

/**
 * 错误处理最佳实践
 */
console.log("\n--- 错误处理 ---");

const safeEmitter = new EventEmitter();

// 必须监听 error 事件，否则会抛出异常
safeEmitter.on("error", (err) => {
    console.error("捕获错误:", err.message);
});

// 使用 process 捕获未处理的事件错误
// process.on('uncaughtException', (err) => {
//     console.error('未捕获异常:', err);
// });

safeEmitter.emit("error", new Error("测试错误"));

/**
 * 事件包装器（确保事件只触发一次并带超时）
 */
function waitForEvent(emitter, event, timeout = 5000) {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
            reject(new Error(`等待 ${event} 事件超时`));
        }, timeout);

        emitter.once(event, (data) => {
            clearTimeout(timer);
            resolve(data);
        });

        emitter.once("error", (err) => {
            clearTimeout(timer);
            reject(err);
        });
    });
}


console.log("\n" + "=".repeat(60));
console.log("6. 进程事件");
console.log("=".repeat(60));

// ============================================================
//                    6. 进程事件
// ============================================================

console.log("\n--- 进程事件 ---");
console.log(`
常用进程事件：

process.on('exit', (code) => {})
  - 进程即将退出时触发
  - 同步操作，不能执行异步代码

process.on('uncaughtException', (err) => {})
  - 未捕获的异常
  - 应该记录日志后退出进程

process.on('unhandledRejection', (reason, promise) => {})
  - 未处理的 Promise 拒绝

process.on('SIGINT', () => {})
  - 收到 Ctrl+C 信号

process.on('SIGTERM', () => {})
  - 收到终止信号（如 kill 命令）
`);

// 示例：优雅关闭
process.on("SIGINT", () => {
    console.log("\n收到 SIGINT，准备退出...");
    // 清理资源
    // server.close();
    // db.disconnect();
    process.exit(0);
});


console.log("\n【总结】");
console.log(`
模块系统：
- CommonJS: require/module.exports（Node.js 默认）
- ES Modules: import/export（现代标准）
- 模块有缓存机制
- 动态导入: await import()

EventEmitter：
- on(event, handler): 监听事件
- once(event, handler): 一次性监听
- emit(event, ...args): 触发事件
- off(event, handler): 移除监听
- 必须处理 error 事件

最佳实践：
- 继承 EventEmitter 创建自定义类
- 使用发布-订阅模式解耦
- 正确处理异步事件
- 处理 error 事件避免崩溃
- 设置合理的最大监听器数量
`);

// 等待异步操作完成
setTimeout(() => {
    console.log("\n--- 示例运行完成 ---");
}, 500);
