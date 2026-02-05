/**
 * ============================================================
 *                JavaScript async/await
 * ============================================================
 * 本文件介绍 JavaScript 中的 async/await 语法。
 * ============================================================
 */

console.log("=".repeat(60));
console.log("1. async/await 基础");
console.log("=".repeat(60));

// ============================================================
//                    1. async/await 基础
// ============================================================

/**
 * 【async/await 是 Promise 的语法糖】
 *
 * async 函数总是返回 Promise
 * await 等待 Promise 解决，暂停函数执行
 */

// --- async 函数 ---
console.log("\n--- async 函数 ---");

async function greet() {
    return "Hello, async!";  // 自动包装为 Promise
}

greet().then(msg => console.log("async 返回:", msg));

// 等同于
function greetPromise() {
    return Promise.resolve("Hello, Promise!");
}

// --- await ---
console.log("\n--- await 基础 ---");

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function fetchData() {
    console.log("开始获取数据...");
    await delay(100);  // 等待 100ms
    console.log("数据获取完成");
    return { id: 1, name: "Data" };
}

fetchData().then(data => console.log("获取到:", data));


// --- 顺序执行 ---
setTimeout(async () => {
    console.log("\n" + "=".repeat(60));
    console.log("2. 顺序执行与并行执行");
    console.log("=".repeat(60));

    // ============================================================
    //                    2. 顺序与并行
    // ============================================================

    const mockFetch = (name, time) => new Promise(resolve => {
        setTimeout(() => resolve(`${name} 完成`), time);
    });

    // --- 顺序执行 ---
    console.log("\n--- 顺序执行 ---");
    async function sequential() {
        console.log("顺序开始");
        const start = Date.now();

        const result1 = await mockFetch("请求1", 100);
        console.log("  ", result1);

        const result2 = await mockFetch("请求2", 100);
        console.log("  ", result2);

        const result3 = await mockFetch("请求3", 100);
        console.log("  ", result3);

        console.log(`顺序完成，耗时 ${Date.now() - start}ms`);
    }

    await sequential();

    // --- 并行执行 ---
    console.log("\n--- 并行执行 ---");
    async function parallel() {
        console.log("并行开始");
        const start = Date.now();

        // 同时发起所有请求
        const [result1, result2, result3] = await Promise.all([
            mockFetch("请求1", 100),
            mockFetch("请求2", 100),
            mockFetch("请求3", 100)
        ]);

        console.log("  ", result1, result2, result3);
        console.log(`并行完成，耗时 ${Date.now() - start}ms`);
    }

    await parallel();

    // --- 并发控制 ---
    console.log("\n--- 并发控制 ---");

    async function withConcurrencyLimit(tasks, limit) {
        const results = [];
        const executing = [];

        for (const task of tasks) {
            const p = Promise.resolve().then(task);
            results.push(p);

            if (limit <= tasks.length) {
                const e = p.then(() => executing.splice(executing.indexOf(e), 1));
                executing.push(e);

                if (executing.length >= limit) {
                    await Promise.race(executing);
                }
            }
        }

        return Promise.all(results);
    }

    const tasks = Array.from({ length: 5 }, (_, i) =>
        () => mockFetch(`任务${i + 1}`, 50)
    );

    const results = await withConcurrencyLimit(tasks, 2);
    console.log("并发控制结果:", results);

    runPart3();
}, 300);


async function runPart3() {
    console.log("\n" + "=".repeat(60));
    console.log("3. 错误处理");
    console.log("=".repeat(60));

    // ============================================================
    //                    3. 错误处理
    // ============================================================

    // --- try/catch ---
    console.log("\n--- try/catch ---");

    async function mightFail(shouldFail) {
        await new Promise(resolve => setTimeout(resolve, 50));
        if (shouldFail) {
            throw new Error("操作失败");
        }
        return "操作成功";
    }

    async function handleError() {
        try {
            const result = await mightFail(true);
            console.log(result);
        } catch (error) {
            console.log("捕获错误:", error.message);
        } finally {
            console.log("清理资源");
        }
    }

    await handleError();

    // --- 多个 await 的错误处理 ---
    console.log("\n--- 多个 await 错误处理 ---");

    async function multipleAwaits() {
        try {
            const a = await mightFail(false);
            console.log("步骤1:", a);

            const b = await mightFail(true);  // 这里会失败
            console.log("步骤2:", b);

            const c = await mightFail(false);  // 不会执行
            console.log("步骤3:", c);
        } catch (error) {
            console.log("捕获:", error.message);
        }
    }

    await multipleAwaits();

    // --- Promise.all 的错误处理 ---
    console.log("\n--- Promise.all 错误处理 ---");

    async function parallelWithError() {
        try {
            const results = await Promise.all([
                mightFail(false),
                mightFail(true),  // 失败
                mightFail(false)
            ]);
            console.log(results);
        } catch (error) {
            console.log("并行错误:", error.message);
        }
    }

    await parallelWithError();

    // --- 使用 allSettled 避免快速失败 ---
    console.log("\n--- Promise.allSettled ---");

    async function parallelNoFail() {
        const results = await Promise.allSettled([
            mightFail(false),
            mightFail(true),
            mightFail(false)
        ]);

        results.forEach((result, i) => {
            if (result.status === "fulfilled") {
                console.log(`  任务${i + 1}: ${result.value}`);
            } else {
                console.log(`  任务${i + 1}: 失败 - ${result.reason.message}`);
            }
        });
    }

    await parallelNoFail();

    runPart4();
}


async function runPart4() {
    console.log("\n" + "=".repeat(60));
    console.log("4. 实际应用模式");
    console.log("=".repeat(60));

    // ============================================================
    //                    4. 实际应用模式
    // ============================================================

    // --- 重试模式 ---
    console.log("\n--- 重试模式 ---");

    async function retry(fn, times, delayMs = 100) {
        for (let i = 0; i < times; i++) {
            try {
                return await fn();
            } catch (error) {
                console.log(`  尝试 ${i + 1}/${times} 失败`);
                if (i === times - 1) throw error;
                await new Promise(r => setTimeout(r, delayMs));
            }
        }
    }

    let attemptCount = 0;
    const unreliable = async () => {
        attemptCount++;
        if (attemptCount < 3) throw new Error("暂时失败");
        return "成功！";
    };

    try {
        const result = await retry(unreliable, 5);
        console.log("重试结果:", result);
    } catch (error) {
        console.log("重试失败:", error.message);
    }

    // --- 超时模式 ---
    console.log("\n--- 超时模式 ---");

    async function withTimeout(promise, ms) {
        const timeout = new Promise((_, reject) =>
            setTimeout(() => reject(new Error("超时")), ms)
        );
        return Promise.race([promise, timeout]);
    }

    const slowOp = new Promise(r => setTimeout(() => r("完成"), 500));

    try {
        const result = await withTimeout(slowOp, 100);
        console.log("超时测试:", result);
    } catch (error) {
        console.log("超时测试:", error.message);
    }

    // --- 队列模式 ---
    console.log("\n--- 队列模式 ---");

    class AsyncQueue {
        constructor() {
            this.queue = [];
            this.processing = false;
        }

        async add(task) {
            return new Promise((resolve, reject) => {
                this.queue.push({ task, resolve, reject });
                this.process();
            });
        }

        async process() {
            if (this.processing) return;
            this.processing = true;

            while (this.queue.length > 0) {
                const { task, resolve, reject } = this.queue.shift();
                try {
                    const result = await task();
                    resolve(result);
                } catch (error) {
                    reject(error);
                }
            }

            this.processing = false;
        }
    }

    const queue = new AsyncQueue();
    const delay = ms => new Promise(r => setTimeout(r, ms));

    queue.add(async () => { await delay(50); console.log("  任务1"); return 1; });
    queue.add(async () => { await delay(30); console.log("  任务2"); return 2; });
    queue.add(async () => { await delay(40); console.log("  任务3"); return 3; });

    await delay(200);

    // --- 取消模式（AbortController）---
    console.log("\n--- 取消模式 ---");

    async function cancellableOperation(signal) {
        for (let i = 0; i < 5; i++) {
            if (signal?.aborted) {
                throw new Error("操作已取消");
            }
            console.log(`  步骤 ${i + 1}`);
            await new Promise(r => setTimeout(r, 50));
        }
        return "完成";
    }

    const controller = new AbortController();

    // 100ms 后取消
    setTimeout(() => controller.abort(), 100);

    try {
        const result = await cancellableOperation(controller.signal);
        console.log(result);
    } catch (error) {
        console.log("取消:", error.message);
    }

    console.log("\n【总结】");
    console.log(`
async/await 核心：
- async 函数总是返回 Promise
- await 暂停执行直到 Promise 解决
- 使用 try/catch 处理错误

并行 vs 顺序：
- 顺序：逐个 await
- 并行：Promise.all([...])
- 部分失败：Promise.allSettled([...])

实用模式：
- 重试：循环 + try/catch
- 超时：Promise.race + setTimeout
- 取消：AbortController
- 并发限制：自定义控制逻辑

最佳实践：
- 总是处理错误（try/catch 或 .catch）
- 能并行就并行，提高性能
- 避免在循环中无意义的顺序 await
`);
}
