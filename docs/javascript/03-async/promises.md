# promises.js

::: info 文件信息
- 📄 原文件：`01_promises.js`
- 🔤 语言：javascript
:::

============================================================
               JavaScript Promise
============================================================
本文件介绍 JavaScript 中的 Promise 异步编程。
============================================================

## 完整代码

```javascript
/**
 * ============================================================
 *                JavaScript Promise
 * ============================================================
 * 本文件介绍 JavaScript 中的 Promise 异步编程。
 * ============================================================
 */

console.log("=".repeat(60));
console.log("1. Promise 基础");
console.log("=".repeat(60));

// ============================================================
//                    1. Promise 基础
// ============================================================

/**
 * 【Promise 状态】
 * - pending：初始状态，等待中
 * - fulfilled：操作成功完成
 * - rejected：操作失败
 *
 * 状态一旦改变就不可逆
 */

// --- 创建 Promise ---
console.log("\n--- 创建 Promise ---");

const promise = new Promise((resolve, reject) => {
    // 模拟异步操作
    const success = true;

    setTimeout(() => {
        if (success) {
            resolve("操作成功");
        } else {
            reject(new Error("操作失败"));
        }
    }, 100);
});

// --- then / catch / finally ---
console.log("\n--- then / catch / finally ---");

promise
    .then(result => {
        console.log("成功:", result);
        return "处理后的结果";
    })
    .then(processed => {
        console.log("链式处理:", processed);
    })
    .catch(error => {
        console.log("错误:", error.message);
    })
    .finally(() => {
        console.log("无论成功失败都执行");
    });

// --- 快捷方法 ---
console.log("\n--- Promise.resolve / reject ---");

// 立即解决
Promise.resolve("立即解决的值")
    .then(value => console.log("resolve:", value));

// 立即拒绝
Promise.reject(new Error("立即拒绝"))
    .catch(err => console.log("reject:", err.message));


// 等待初始 Promise 完成
setTimeout(() => {
    console.log("\n" + "=".repeat(60));
    console.log("2. Promise 链式调用");
    console.log("=".repeat(60));

    // ============================================================
    //                    2. Promise 链式调用
    // ============================================================

    // --- 链式调用 ---
    console.log("\n--- 链式调用 ---");

    function step1(value) {
        return new Promise(resolve => {
            setTimeout(() => {
                console.log("Step 1 完成");
                resolve(value + 1);
            }, 100);
        });
    }

    function step2(value) {
        return new Promise(resolve => {
            setTimeout(() => {
                console.log("Step 2 完成");
                resolve(value * 2);
            }, 100);
        });
    }

    function step3(value) {
        return new Promise(resolve => {
            setTimeout(() => {
                console.log("Step 3 完成");
                resolve(value + 10);
            }, 100);
        });
    }

    step1(1)
        .then(step2)
        .then(step3)
        .then(result => {
            console.log("最终结果:", result);  // ((1+1)*2)+10 = 14

            // 继续执行后续代码
            runPart3();
        });

}, 500);


function runPart3() {
    console.log("\n" + "=".repeat(60));
    console.log("3. Promise 静态方法");
    console.log("=".repeat(60));

    // ============================================================
    //                    3. Promise 静态方法
    // ============================================================

    // 辅助函数
    const delay = (ms, value) =>
        new Promise(resolve => setTimeout(() => resolve(value), ms));

    const delayReject = (ms, error) =>
        new Promise((_, reject) => setTimeout(() => reject(new Error(error)), ms));

    // --- Promise.all ---
    console.log("\n--- Promise.all（全部成功）---");

    Promise.all([
        delay(100, "结果1"),
        delay(200, "结果2"),
        delay(150, "结果3")
    ])
        .then(results => {
            console.log("all 成功:", results);
        })
        .catch(error => {
            console.log("all 失败:", error.message);
        });

    // --- Promise.allSettled ---
    console.log("\n--- Promise.allSettled（不管成功失败）---");

    Promise.allSettled([
        delay(100, "成功1"),
        delayReject(150, "失败1"),
        delay(200, "成功2")
    ])
        .then(results => {
            console.log("allSettled 结果:");
            results.forEach((result, i) => {
                if (result.status === "fulfilled") {
                    console.log(`  ${i}: 成功 - ${result.value}`);
                } else {
                    console.log(`  ${i}: 失败 - ${result.reason.message}`);
                }
            });

            runPart4();
        });
}


function runPart4() {
    const delay = (ms, value) =>
        new Promise(resolve => setTimeout(() => resolve(value), ms));

    // --- Promise.race ---
    console.log("\n--- Promise.race（第一个完成）---");

    Promise.race([
        delay(300, "慢"),
        delay(100, "快"),
        delay(200, "中")
    ])
        .then(result => {
            console.log("race 胜出:", result);
        });

    // --- Promise.any ---
    console.log("\n--- Promise.any（第一个成功）---");

    const delayReject = (ms, error) =>
        new Promise((_, reject) => setTimeout(() => reject(new Error(error)), ms));

    Promise.any([
        delayReject(100, "快速失败"),
        delay(200, "成功"),
        delayReject(150, "另一个失败")
    ])
        .then(result => {
            console.log("any 成功:", result);
            runPart5();
        })
        .catch(error => {
            console.log("any 全部失败:", error);
        });
}


function runPart5() {
    console.log("\n" + "=".repeat(60));
    console.log("4. 错误处理");
    console.log("=".repeat(60));

    // ============================================================
    //                    4. 错误处理
    // ============================================================

    // --- catch 位置 ---
    console.log("\n--- 错误传播 ---");

    Promise.resolve()
        .then(() => {
            throw new Error("第一步出错");
        })
        .then(() => {
            console.log("这不会执行");
        })
        .then(() => {
            console.log("这也不会执行");
        })
        .catch(error => {
            console.log("捕获错误:", error.message);
            return "恢复正常";  // 可以恢复
        })
        .then(value => {
            console.log("恢复后:", value);
        });

    // --- 未处理的拒绝 ---
    console.log("\n--- 未处理的拒绝 ---");
    console.log(`
    建议始终添加 .catch() 处理错误
    Node.js 可以监听 unhandledRejection 事件
    浏览器可以监听 unhandledrejection 事件
    `);

    setTimeout(() => {
        runPart6();
    }, 200);
}


function runPart6() {
    console.log("\n" + "=".repeat(60));
    console.log("5. 实际应用");
    console.log("=".repeat(60));

    // ============================================================
    //                    5. 实际应用
    // ============================================================

    // --- 模拟 fetch ---
    console.log("\n--- 模拟 API 请求 ---");

    function mockFetch(url) {
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                if (url.includes("error")) {
                    reject(new Error("Network error"));
                } else {
                    resolve({
                        ok: true,
                        json: () => Promise.resolve({
                            url,
                            data: "模拟数据"
                        })
                    });
                }
            }, 100);
        });
    }

    mockFetch("/api/users")
        .then(response => response.json())
        .then(data => {
            console.log("请求成功:", data);
        })
        .catch(error => {
            console.log("请求失败:", error.message);
        });

    // --- 并行请求 ---
    console.log("\n--- 并行请求 ---");

    Promise.all([
        mockFetch("/api/users"),
        mockFetch("/api/posts"),
        mockFetch("/api/comments")
    ])
        .then(responses => Promise.all(responses.map(r => r.json())))
        .then(([users, posts, comments]) => {
            console.log("并行请求完成:");
            console.log("  users:", users.url);
            console.log("  posts:", posts.url);
            console.log("  comments:", comments.url);
        });

    // --- 超时处理 ---
    console.log("\n--- 超时处理 ---");

    function withTimeout(promise, ms) {
        const timeout = new Promise((_, reject) =>
            setTimeout(() => reject(new Error("Timeout")), ms)
        );
        return Promise.race([promise, timeout]);
    }

    const slowRequest = new Promise(resolve =>
        setTimeout(() => resolve("慢请求完成"), 500)
    );

    withTimeout(slowRequest, 200)
        .then(result => console.log("超时测试成功:", result))
        .catch(error => console.log("超时测试失败:", error.message));

    // --- 重试逻辑 ---
    console.log("\n--- 重试逻辑 ---");

    function retry(fn, times, delay = 100) {
        return new Promise((resolve, reject) => {
            function attempt(n) {
                fn()
                    .then(resolve)
                    .catch(error => {
                        if (n === 0) {
                            reject(error);
                        } else {
                            console.log(`  重试剩余 ${n} 次...`);
                            setTimeout(() => attempt(n - 1), delay);
                        }
                    });
            }
            attempt(times);
        });
    }

    let attemptCount = 0;
    const unreliableOperation = () => {
        attemptCount++;
        return new Promise((resolve, reject) => {
            if (attemptCount < 3) {
                reject(new Error("暂时失败"));
            } else {
                resolve("终于成功了");
            }
        });
    };

    retry(unreliableOperation, 5, 50)
        .then(result => console.log("重试结果:", result))
        .catch(error => console.log("重试失败:", error.message));

    setTimeout(() => {
        console.log("\n【总结】");
        console.log(`
Promise 基础：
- new Promise((resolve, reject) => {})
- .then() 处理成功
- .catch() 处理失败
- .finally() 总是执行

静态方法：
- Promise.all() - 全部成功才成功
- Promise.allSettled() - 等待全部完成
- Promise.race() - 第一个完成
- Promise.any() - 第一个成功
- Promise.resolve() / reject() - 快速创建

最佳实践：
- 始终添加 .catch() 处理错误
- 使用 Promise.all 并行请求
- 添加超时和重试机制
`);
    }, 1000);
}
```
