# event emitter.js

::: info 文件信息
- 📄 原文件：`02_event_emitter.js`
- 🔤 语言：javascript
:::

============================================================
               自定义事件发射器
============================================================
一个功能完整的事件发射器实现，支持多种高级特性。
功能：
- 事件监听/触发
- 一次性监听
- 优先级监听
- 通配符事件
- 异步事件
- 事件命名空间
============================================================

## 完整代码

```javascript
/**
 * ============================================================
 *                自定义事件发射器
 * ============================================================
 * 一个功能完整的事件发射器实现，支持多种高级特性。
 *
 * 功能：
 * - 事件监听/触发
 * - 一次性监听
 * - 优先级监听
 * - 通配符事件
 * - 异步事件
 * - 事件命名空间
 * ============================================================
 */

/**
 * 高级事件发射器
 */
class EventEmitter {
    constructor(options = {}) {
        this._events = new Map();
        this._maxListeners = options.maxListeners || 10;
        this._wildcardCache = new Map();
    }

    // ========================================================
    //                    基本方法
    // ========================================================

    /**
     * 监听事件
     * @param {string} event - 事件名
     * @param {Function} listener - 监听函数
     * @param {Object} options - 选项
     * @returns {Function} 取消监听的函数
     */
    on(event, listener, options = {}) {
        if (typeof listener !== "function") {
            throw new TypeError("Listener must be a function");
        }

        const { priority = 0, once = false } = options;

        if (!this._events.has(event)) {
            this._events.set(event, []);
        }

        const listeners = this._events.get(event);

        // 检查最大监听器数量
        if (listeners.length >= this._maxListeners) {
            console.warn(
                `Warning: Event '${event}' has more than ${this._maxListeners} listeners.`
            );
        }

        const wrapper = {
            listener,
            priority,
            once
        };

        // 按优先级插入
        const index = listeners.findIndex(l => l.priority < priority);
        if (index === -1) {
            listeners.push(wrapper);
        } else {
            listeners.splice(index, 0, wrapper);
        }

        // 清除通配符缓存
        this._wildcardCache.clear();

        // 返回取消函数
        return () => this.off(event, listener);
    }

    /**
     * 一次性监听
     */
    once(event, listener, options = {}) {
        return this.on(event, listener, { ...options, once: true });
    }

    /**
     * 移除监听器
     */
    off(event, listener) {
        if (!this._events.has(event)) return this;

        if (listener) {
            const listeners = this._events.get(event);
            const index = listeners.findIndex(l => l.listener === listener);
            if (index !== -1) {
                listeners.splice(index, 1);
            }
            if (listeners.length === 0) {
                this._events.delete(event);
            }
        } else {
            // 移除该事件的所有监听器
            this._events.delete(event);
        }

        this._wildcardCache.clear();
        return this;
    }

    /**
     * 触发事件
     */
    emit(event, ...args) {
        let listeners = [];

        // 精确匹配
        if (this._events.has(event)) {
            listeners = [...this._events.get(event)];
        }

        // 通配符匹配
        for (const [pattern, patternListeners] of this._events) {
            if (pattern.includes("*") && this._matchWildcard(pattern, event)) {
                listeners.push(...patternListeners);
            }
        }

        if (listeners.length === 0) {
            // error 事件没有监听器时抛出
            if (event === "error" && args[0] instanceof Error) {
                throw args[0];
            }
            return false;
        }

        // 按优先级排序
        listeners.sort((a, b) => b.priority - a.priority);

        // 执行监听器
        for (const wrapper of listeners) {
            try {
                wrapper.listener.apply(this, args);

                // 移除一次性监听器
                if (wrapper.once) {
                    this.off(event, wrapper.listener);
                }
            } catch (err) {
                // 错误处理
                if (event !== "error") {
                    this.emit("error", err);
                } else {
                    throw err;
                }
            }
        }

        return true;
    }

    /**
     * 异步触发事件
     */
    async emitAsync(event, ...args) {
        let listeners = [];

        if (this._events.has(event)) {
            listeners = [...this._events.get(event)];
        }

        for (const [pattern, patternListeners] of this._events) {
            if (pattern.includes("*") && this._matchWildcard(pattern, event)) {
                listeners.push(...patternListeners);
            }
        }

        if (listeners.length === 0) {
            return false;
        }

        listeners.sort((a, b) => b.priority - a.priority);

        for (const wrapper of listeners) {
            try {
                await wrapper.listener.apply(this, args);

                if (wrapper.once) {
                    this.off(event, wrapper.listener);
                }
            } catch (err) {
                if (event !== "error") {
                    await this.emitAsync("error", err);
                } else {
                    throw err;
                }
            }
        }

        return true;
    }

    // ========================================================
    //                    实用方法
    // ========================================================

    /**
     * 获取监听器数量
     */
    listenerCount(event) {
        return this._events.has(event) ? this._events.get(event).length : 0;
    }

    /**
     * 获取所有事件名
     */
    eventNames() {
        return Array.from(this._events.keys());
    }

    /**
     * 获取监听器列表
     */
    listeners(event) {
        if (!this._events.has(event)) return [];
        return this._events.get(event).map(w => w.listener);
    }

    /**
     * 设置最大监听器数量
     */
    setMaxListeners(n) {
        this._maxListeners = n;
        return this;
    }

    /**
     * 移除所有监听器
     */
    removeAllListeners(event) {
        if (event) {
            this._events.delete(event);
        } else {
            this._events.clear();
        }
        this._wildcardCache.clear();
        return this;
    }

    /**
     * 等待事件
     */
    waitFor(event, timeout = 0) {
        return new Promise((resolve, reject) => {
            let timeoutId;

            const cleanup = this.once(event, (...args) => {
                if (timeoutId) clearTimeout(timeoutId);
                resolve(args.length === 1 ? args[0] : args);
            });

            if (timeout > 0) {
                timeoutId = setTimeout(() => {
                    cleanup();
                    reject(new Error(`Timeout waiting for event: ${event}`));
                }, timeout);
            }
        });
    }

    // ========================================================
    //                    命名空间支持
    // ========================================================

    /**
     * 创建命名空间
     */
    namespace(ns) {
        const self = this;
        return {
            on: (event, listener, options) =>
                self.on(`${ns}:${event}`, listener, options),
            once: (event, listener, options) =>
                self.once(`${ns}:${event}`, listener, options),
            off: (event, listener) =>
                self.off(`${ns}:${event}`, listener),
            emit: (event, ...args) =>
                self.emit(`${ns}:${event}`, ...args),
            emitAsync: (event, ...args) =>
                self.emitAsync(`${ns}:${event}`, ...args)
        };
    }

    // ========================================================
    //                    私有方法
    // ========================================================

    /**
     * 通配符匹配
     */
    _matchWildcard(pattern, event) {
        const cacheKey = `${pattern}:${event}`;
        if (this._wildcardCache.has(cacheKey)) {
            return this._wildcardCache.get(cacheKey);
        }

        const regex = new RegExp(
            "^" + pattern.replace(/\*/g, "[^:]*") + "$"
        );
        const result = regex.test(event);
        this._wildcardCache.set(cacheKey, result);
        return result;
    }
}

// ============================================================
//                    示例与测试
// ============================================================

console.log("=".repeat(60));
console.log("自定义事件发射器示例");
console.log("=".repeat(60));

// --- 基本用法 ---
console.log("\n--- 基本用法 ---");

const emitter = new EventEmitter();

emitter.on("message", (msg) => {
    console.log("收到消息:", msg);
});

emitter.emit("message", "Hello, World!");

// --- 一次性监听 ---
console.log("\n--- 一次性监听 ---");

emitter.once("connect", () => {
    console.log("连接成功（只触发一次）");
});

emitter.emit("connect");
emitter.emit("connect");  // 不会触发

// --- 优先级 ---
console.log("\n--- 优先级监听 ---");

emitter.on("task", () => console.log("  普通优先级"), { priority: 0 });
emitter.on("task", () => console.log("  高优先级"), { priority: 10 });
emitter.on("task", () => console.log("  低优先级"), { priority: -10 });

emitter.emit("task");

// --- 取消监听 ---
console.log("\n--- 取消监听 ---");

const unsubscribe = emitter.on("data", (data) => {
    console.log("数据:", data);
});

emitter.emit("data", "第一条");
unsubscribe();
emitter.emit("data", "第二条（不会显示）");

// --- 通配符 ---
console.log("\n--- 通配符事件 ---");

emitter.on("user:*", (data) => {
    console.log("用户事件:", data);
});

emitter.emit("user:login", { username: "alice" });
emitter.emit("user:logout", { username: "alice" });

// --- 命名空间 ---
console.log("\n--- 命名空间 ---");

const userEvents = emitter.namespace("user");
const orderEvents = emitter.namespace("order");

userEvents.on("created", (user) => {
    console.log("用户创建:", user.name);
});

orderEvents.on("created", (order) => {
    console.log("订单创建:", order.id);
});

userEvents.emit("created", { name: "Bob" });
orderEvents.emit("created", { id: "ORD001" });

// --- 异步事件 ---
console.log("\n--- 异步事件 ---");

(async () => {
    const asyncEmitter = new EventEmitter();

    asyncEmitter.on("fetch", async (url) => {
        console.log(`开始获取: ${url}`);
        await new Promise(r => setTimeout(r, 100));
        console.log(`获取完成: ${url}`);
    });

    await asyncEmitter.emitAsync("fetch", "https://api.example.com");
    console.log("所有异步操作完成");

    // --- 等待事件 ---
    console.log("\n--- 等待事件 ---");

    setTimeout(() => {
        asyncEmitter.emit("ready", { status: "ok" });
    }, 100);

    const result = await asyncEmitter.waitFor("ready", 1000);
    console.log("收到 ready 事件:", result);

    // --- 错误处理 ---
    console.log("\n--- 错误处理 ---");

    asyncEmitter.on("error", (err) => {
        console.error("捕获错误:", err.message);
    });

    asyncEmitter.on("danger", () => {
        throw new Error("出错了！");
    });

    asyncEmitter.emit("danger");

    // --- 监听器信息 ---
    console.log("\n--- 监听器信息 ---");
    console.log("事件列表:", emitter.eventNames());
    console.log("message 监听器数:", emitter.listenerCount("message"));

    console.log("\n【示例完成】");
})();

// 导出
module.exports = EventEmitter;
```
