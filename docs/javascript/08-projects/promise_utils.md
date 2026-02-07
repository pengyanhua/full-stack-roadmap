# promise utils.js

::: info 文件信息
- 📄 原文件：`03_promise_utils.js`
- 🔤 语言：javascript
:::

Promise 工具库
一组实用的 Promise 工具函数。
功能：
- 延迟/超时
- 重试机制
- 并发控制
- 队列执行
- 缓存

## 完整代码

```javascript
/**
 * ============================================================
 *                Promise 工具库
 * ============================================================
 * 一组实用的 Promise 工具函数。
 *
 * 功能：
 * - 延迟/超时
 * - 重试机制
 * - 并发控制
 * - 队列执行
 * - 缓存
 * ============================================================
 */

// ============================================================
//                    基础工具
// ============================================================

/**
 * 延迟执行
 * @param {number} ms - 延迟毫秒数
 * @returns {Promise<void>}
 */
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * 带超时的 Promise
 * @param {Promise} promise - 原始 Promise
 * @param {number} ms - 超时时间
 * @param {string} message - 超时错误消息
 * @returns {Promise}
 */
function timeout(promise, ms, message = "Operation timed out") {
    const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error(message)), ms);
    });
    return Promise.race([promise, timeoutPromise]);
}

/**
 * 可取消的 Promise
 * @param {Function} executor - 执行函数
 * @returns {{ promise: Promise, cancel: Function }}
 */
function cancellable(executor) {
    let cancelled = false;
    let rejectFn;

    const promise = new Promise((resolve, reject) => {
        rejectFn = reject;

        executor(
            (value) => !cancelled && resolve(value),
            (error) => !cancelled && reject(error),
            () => cancelled
        );
    });

    const cancel = (reason = "Cancelled") => {
        cancelled = true;
        rejectFn(new Error(reason));
    };

    return { promise, cancel };
}

// ============================================================
//                    重试机制
// ============================================================

/**
 * 带重试的异步操作
 * @param {Function} fn - 异步函数
 * @param {Object} options - 选项
 * @returns {Promise}
 */
async function retry(fn, options = {}) {
    const {
        retries = 3,
        delay: retryDelay = 1000,
        backoff = 1,        // 退避系数
        maxDelay = 30000,
        onRetry = null,
        shouldRetry = () => true
    } = options;

    let lastError;
    let currentDelay = retryDelay;

    for (let attempt = 0; attempt <= retries; attempt++) {
        try {
            return await fn(attempt);
        } catch (error) {
            lastError = error;

            if (attempt >= retries || !shouldRetry(error, attempt)) {
                throw error;
            }

            if (onRetry) {
                onRetry(error, attempt + 1);
            }

            await delay(currentDelay);
            currentDelay = Math.min(currentDelay * backoff, maxDelay);
        }
    }

    throw lastError;
}

/**
 * 指数退避重试
 * @param {Function} fn - 异步函数
 * @param {Object} options - 选项
 * @returns {Promise}
 */
function retryWithExponentialBackoff(fn, options = {}) {
    return retry(fn, {
        ...options,
        backoff: options.backoff || 2
    });
}

// ============================================================
//                    并发控制
// ============================================================

/**
 * 并发限制执行
 * @param {Array} items - 数据项
 * @param {Function} fn - 处理函数
 * @param {number} concurrency - 并发数
 * @returns {Promise<Array>}
 */
async function mapLimit(items, fn, concurrency = 5) {
    const results = new Array(items.length);
    let index = 0;

    async function worker() {
        while (index < items.length) {
            const currentIndex = index++;
            results[currentIndex] = await fn(items[currentIndex], currentIndex);
        }
    }

    const workers = Array(Math.min(concurrency, items.length))
        .fill(null)
        .map(() => worker());

    await Promise.all(workers);
    return results;
}

/**
 * 并发池
 */
class ConcurrencyPool {
    constructor(concurrency = 5) {
        this.concurrency = concurrency;
        this.running = 0;
        this.queue = [];
    }

    async add(fn) {
        return new Promise((resolve, reject) => {
            this.queue.push({ fn, resolve, reject });
            this._process();
        });
    }

    async _process() {
        if (this.running >= this.concurrency || this.queue.length === 0) {
            return;
        }

        this.running++;
        const { fn, resolve, reject } = this.queue.shift();

        try {
            const result = await fn();
            resolve(result);
        } catch (error) {
            reject(error);
        } finally {
            this.running--;
            this._process();
        }
    }

    get pending() {
        return this.queue.length;
    }

    get active() {
        return this.running;
    }
}

/**
 * 限流执行
 * @param {Function} fn - 异步函数
 * @param {number} limit - 每秒最大调用次数
 * @returns {Function}
 */
function rateLimit(fn, limit) {
    const queue = [];
    let lastCall = 0;
    const interval = 1000 / limit;

    return async function (...args) {
        return new Promise((resolve, reject) => {
            queue.push({ args, resolve, reject });
            processQueue();
        });

        async function processQueue() {
            if (queue.length === 0) return;

            const now = Date.now();
            const waitTime = Math.max(0, lastCall + interval - now);

            if (waitTime > 0) {
                setTimeout(processQueue, waitTime);
                return;
            }

            lastCall = now;
            const { args, resolve, reject } = queue.shift();

            try {
                const result = await fn.apply(this, args);
                resolve(result);
            } catch (error) {
                reject(error);
            }

            if (queue.length > 0) {
                setTimeout(processQueue, interval);
            }
        }
    };
}

// ============================================================
//                    队列执行
// ============================================================

/**
 * 串行执行队列
 * @param {Array<Function>} tasks - 任务数组
 * @returns {Promise<Array>}
 */
async function series(tasks) {
    const results = [];
    for (const task of tasks) {
        results.push(await task());
    }
    return results;
}

/**
 * 瀑布流执行（前一个结果传给后一个）
 * @param {Array<Function>} tasks - 任务数组
 * @param {any} initial - 初始值
 * @returns {Promise}
 */
async function waterfall(tasks, initial) {
    let result = initial;
    for (const task of tasks) {
        result = await task(result);
    }
    return result;
}

/**
 * 异步任务队列
 */
class AsyncQueue {
    constructor(options = {}) {
        this.concurrency = options.concurrency || 1;
        this.autoStart = options.autoStart !== false;
        this.tasks = [];
        this.running = 0;
        this.paused = false;
        this.onEmpty = options.onEmpty || (() => {});
        this.onDrain = options.onDrain || (() => {});
    }

    push(task, priority = 0) {
        return new Promise((resolve, reject) => {
            this.tasks.push({ task, priority, resolve, reject });
            this.tasks.sort((a, b) => b.priority - a.priority);

            if (this.autoStart && !this.paused) {
                this._process();
            }
        });
    }

    async _process() {
        if (this.paused || this.running >= this.concurrency) {
            return;
        }

        if (this.tasks.length === 0) {
            if (this.running === 0) {
                this.onEmpty();
            }
            return;
        }

        this.running++;
        const { task, resolve, reject } = this.tasks.shift();

        try {
            const result = await task();
            resolve(result);
        } catch (error) {
            reject(error);
        } finally {
            this.running--;

            if (this.tasks.length === 0 && this.running === 0) {
                this.onDrain();
            }

            this._process();
        }
    }

    start() {
        this.paused = false;
        this._process();
    }

    pause() {
        this.paused = true;
    }

    clear() {
        const rejected = this.tasks.map(t =>
            t.reject(new Error("Queue cleared"))
        );
        this.tasks = [];
        return rejected;
    }

    get length() {
        return this.tasks.length;
    }

    get idle() {
        return this.running === 0 && this.tasks.length === 0;
    }
}

// ============================================================
//                    缓存
// ============================================================

/**
 * Promise 缓存
 * @param {Function} fn - 异步函数
 * @param {Object} options - 选项
 * @returns {Function}
 */
function memoizeAsync(fn, options = {}) {
    const {
        keyGenerator = (...args) => JSON.stringify(args),
        ttl = 0,
        maxSize = 100
    } = options;

    const cache = new Map();

    return async function (...args) {
        const key = keyGenerator(...args);

        if (cache.has(key)) {
            const entry = cache.get(key);
            if (!entry.expiresAt || entry.expiresAt > Date.now()) {
                return entry.value;
            }
            cache.delete(key);
        }

        const value = await fn.apply(this, args);

        // 检查缓存大小
        if (cache.size >= maxSize) {
            const firstKey = cache.keys().next().value;
            cache.delete(firstKey);
        }

        cache.set(key, {
            value,
            expiresAt: ttl ? Date.now() + ttl : null
        });

        return value;
    };
}

/**
 * 去重请求（相同请求共享结果）
 * @param {Function} fn - 异步函数
 * @param {Function} keyGenerator - 键生成函数
 * @returns {Function}
 */
function dedupe(fn, keyGenerator = (...args) => JSON.stringify(args)) {
    const pending = new Map();

    return async function (...args) {
        const key = keyGenerator(...args);

        if (pending.has(key)) {
            return pending.get(key);
        }

        const promise = fn.apply(this, args).finally(() => {
            pending.delete(key);
        });

        pending.set(key, promise);
        return promise;
    };
}

// ============================================================
//                    组合工具
// ============================================================

/**
 * Promise.all 的对象版本
 * @param {Object} obj - 包含 Promise 的对象
 * @returns {Promise<Object>}
 */
async function allObject(obj) {
    const keys = Object.keys(obj);
    const values = await Promise.all(Object.values(obj));
    return keys.reduce((result, key, i) => {
        result[key] = values[i];
        return result;
    }, {});
}

/**
 * 并行执行但保留所有结果（成功和失败）
 * @param {Array<Promise>} promises
 * @returns {Promise<Array<{status, value?, reason?}>>}
 */
async function allSettledWithDetails(promises) {
    return Promise.all(
        promises.map(async (promise) => {
            try {
                const value = await promise;
                return { status: "fulfilled", value };
            } catch (reason) {
                return { status: "rejected", reason };
            }
        })
    );
}

/**
 * 过滤异步
 * @param {Array} items - 数据项
 * @param {Function} predicate - 异步谓词函数
 * @param {number} concurrency - 并发数
 * @returns {Promise<Array>}
 */
async function filterAsync(items, predicate, concurrency = 5) {
    const results = await mapLimit(
        items,
        async (item, index) => ({
            item,
            keep: await predicate(item, index)
        }),
        concurrency
    );
    return results.filter(r => r.keep).map(r => r.item);
}

// ============================================================
//                    示例与测试
// ============================================================

async function main() {
    console.log("=".repeat(60));
    console.log("Promise 工具库示例");
    console.log("=".repeat(60));

    // --- 延迟和超时 ---
    console.log("\n--- 延迟和超时 ---");

    await delay(100);
    console.log("延迟 100ms 完成");

    try {
        await timeout(delay(200), 100, "超时了！");
    } catch (err) {
        console.log("捕获超时:", err.message);
    }

    // --- 可取消的 Promise ---
    console.log("\n--- 可取消的 Promise ---");

    const { promise, cancel } = cancellable((resolve, reject, isCancelled) => {
        setTimeout(() => {
            if (!isCancelled()) resolve("完成");
        }, 100);
    });

    setTimeout(() => cancel("用户取消"), 50);

    try {
        await promise;
    } catch (err) {
        console.log("操作被取消:", err.message);
    }

    // --- 重试 ---
    console.log("\n--- 重试机制 ---");

    let attempts = 0;
    try {
        await retry(
            async (attempt) => {
                attempts++;
                console.log(`  尝试 ${attempt + 1}`);
                if (attempt < 2) throw new Error("暂时失败");
                return "成功";
            },
            {
                retries: 3,
                delay: 100,
                onRetry: (err, n) => console.log(`  重试 ${n}: ${err.message}`)
            }
        );
        console.log(`  总共尝试: ${attempts} 次`);
    } catch (err) {
        console.log("重试失败:", err.message);
    }

    // --- 并发控制 ---
    console.log("\n--- 并发控制 ---");

    const urls = [1, 2, 3, 4, 5, 6, 7, 8];
    const startTime = Date.now();

    const results = await mapLimit(
        urls,
        async (n) => {
            await delay(100);
            return n * 2;
        },
        3
    );

    console.log(`  结果: ${results.join(", ")}`);
    console.log(`  耗时: ${Date.now() - startTime}ms（并发=3）`);

    // --- 并发池 ---
    console.log("\n--- 并发池 ---");

    const pool = new ConcurrencyPool(2);

    const poolResults = await Promise.all([
        pool.add(async () => { await delay(100); return 1; }),
        pool.add(async () => { await delay(100); return 2; }),
        pool.add(async () => { await delay(100); return 3; }),
        pool.add(async () => { await delay(100); return 4; })
    ]);

    console.log("  池结果:", poolResults);

    // --- 异步队列 ---
    console.log("\n--- 异步队列 ---");

    const queue = new AsyncQueue({
        concurrency: 1,
        onDrain: () => console.log("  队列已空")
    });

    queue.push(async () => { await delay(50); console.log("  任务1"); }, 0);
    queue.push(async () => { await delay(50); console.log("  任务2（高优先级）"); }, 10);
    queue.push(async () => { await delay(50); console.log("  任务3"); }, 0);

    await delay(200);

    // --- 缓存 ---
    console.log("\n--- 异步缓存 ---");

    let fetchCount = 0;
    const fetchData = memoizeAsync(
        async (id) => {
            fetchCount++;
            await delay(50);
            return { id, data: `数据${id}` };
        },
        { ttl: 1000 }
    );

    console.log("  第一次调用:", await fetchData(1));
    console.log("  第二次调用（缓存）:", await fetchData(1));
    console.log(`  实际请求次数: ${fetchCount}`);

    // --- 去重请求 ---
    console.log("\n--- 去重请求 ---");

    let requestCount = 0;
    const dedupedFetch = dedupe(async (id) => {
        requestCount++;
        await delay(100);
        return `结果${id}`;
    });

    const [r1, r2, r3] = await Promise.all([
        dedupedFetch(1),
        dedupedFetch(1),
        dedupedFetch(1)
    ]);

    console.log(`  三次调用结果: ${r1}, ${r2}, ${r3}`);
    console.log(`  实际请求次数: ${requestCount}`);

    // --- 对象并行 ---
    console.log("\n--- 对象并行 ---");

    const data = await allObject({
        user: Promise.resolve({ name: "Alice" }),
        posts: Promise.resolve([{ id: 1 }, { id: 2 }]),
        settings: Promise.resolve({ theme: "dark" })
    });

    console.log("  并行结果:", data);

    // --- 瀑布流 ---
    console.log("\n--- 瀑布流执行 ---");

    const waterfallResult = await waterfall([
        async (x) => { console.log(`  步骤1: ${x}`); return x + 1; },
        async (x) => { console.log(`  步骤2: ${x}`); return x * 2; },
        async (x) => { console.log(`  步骤3: ${x}`); return x + 10; }
    ], 0);

    console.log(`  最终结果: ${waterfallResult}`);

    console.log("\n【示例完成】");
}

main().catch(console.error);

// 导出
module.exports = {
    delay,
    timeout,
    cancellable,
    retry,
    retryWithExponentialBackoff,
    mapLimit,
    ConcurrencyPool,
    rateLimit,
    series,
    waterfall,
    AsyncQueue,
    memoizeAsync,
    dedupe,
    allObject,
    allSettledWithDetails,
    filterAsync
};
```
