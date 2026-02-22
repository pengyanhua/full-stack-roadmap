// ============================================================
//                      异步编程（Async/Await）
// ============================================================
// Rust 的异步编程基于 Future trait
// async fn 返回一个实现了 Future 的值
// await 等待 Future 完成
//
// 【重要】Rust 标准库只提供 Future trait，不提供运行时
// 需要第三方运行时：tokio（最流行）、async-std、smol
//
// 【注意】本文件演示核心概念，实际运行需要 tokio 等运行时
// 添加依赖: cargo add tokio --features full

// 模拟异步运行时的简单实现（用于演示）
// 实际项目中使用 #[tokio::main] 或 #[async_std::main]

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

fn main() {
    println!("=== 异步编程 ===");

    // ----------------------------------------------------------
    // 1. 基本概念
    // ----------------------------------------------------------
    // async fn 定义异步函数，返回 impl Future<Output = T>
    // .await 暂停当前任务，等待 Future 完成
    //
    // 【与线程的区别】
    // 线程：由操作系统调度，每个线程有独立的栈（MB 级别）
    // 异步：由运行时调度，任务非常轻量（KB 级别）
    //
    // 【适用场景】
    // 线程：CPU 密集型任务
    // 异步：IO 密集型任务（网络请求、文件读写、数据库查询）

    println!("注意: 以下代码展示语法和概念");
    println!("实际运行需要 tokio 等异步运行时\n");

    // ----------------------------------------------------------
    // 2. async fn 语法
    // ----------------------------------------------------------
    // async fn 实际上是语法糖：
    // async fn foo() -> i32 { 42 }
    // 等价于：
    // fn foo() -> impl Future<Output = i32> { async { 42 } }

    println!("=== async fn 语法演示 ===");
    println!("async fn hello() -> String");
    println!("  等价于 fn hello() -> impl Future<Output = String>");

    // ----------------------------------------------------------
    // 3. Future trait
    // ----------------------------------------------------------
    // trait Future {
    //     type Output;
    //     fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output>;
    // }
    //
    // Poll::Ready(value) — Future 已完成，返回值
    // Poll::Pending      — Future 未完成，稍后再试
    //
    // 【重要】Future 是惰性的，不 poll 就不会执行
    // 这与 JavaScript 的 Promise 不同（Promise 创建即开始执行）

    // 手动实现一个简单的 Future
    let countdown = Countdown { remaining: 3 };
    println!("创建了一个 Countdown Future（需要运行时来驱动）");
    println!("Countdown 类型: {:?}", countdown);

    // ----------------------------------------------------------
    // 4. 实际 tokio 代码示例（伪代码展示）
    // ----------------------------------------------------------
    println!("\n=== Tokio 代码示例 ===");

    // 以下是在 tokio 运行时中的实际写法：
    println!(r#"
// Cargo.toml:
// [dependencies]
// tokio = {{ version = "1", features = ["full"] }}
// reqwest = {{ version = "0.11" }}

#[tokio::main]
async fn main() {{
    // 基本异步函数调用
    let result = fetch_data("https://api.example.com").await;
    println!("结果: {{}}", result);

    // 并发执行多个任务
    let (r1, r2, r3) = tokio::join!(
        fetch_data("url1"),
        fetch_data("url2"),
        fetch_data("url3"),
    );

    // 竞争：返回最先完成的
    let fastest = tokio::select! {{
        val = fetch_data("fast_url") => val,
        val = fetch_data("slow_url") => val,
    }};

    // 生成并发任务
    let handle = tokio::spawn(async {{
        // 在独立任务中运行
        heavy_computation().await
    }});
    let result = handle.await.unwrap();
}}

async fn fetch_data(url: &str) -> String {{
    // 模拟网络请求
    tokio::time::sleep(Duration::from_millis(100)).await;
    format!("来自 {{}} 的数据", url)
}}
"#);

    // ----------------------------------------------------------
    // 5. 异步模式
    // ----------------------------------------------------------
    println!("=== 常用异步模式 ===\n");

    // 模式1: join!（并发执行所有任务）
    println!("模式1: tokio::join!(task1, task2, task3)");
    println!("  所有任务并发执行，全部完成后返回\n");

    // 模式2: select!（竞争，取最快的）
    println!("模式2: tokio::select! {{ val = task1 => ..., val = task2 => ... }}");
    println!("  返回最先完成的任务结果\n");

    // 模式3: spawn（后台任务）
    println!("模式3: tokio::spawn(async {{ ... }})");
    println!("  在后台运行，返回 JoinHandle\n");

    // 模式4: Stream（异步迭代器）
    println!("模式4: while let Some(item) = stream.next().await {{ ... }}");
    println!("  异步遍历数据流\n");

    // 模式5: 超时
    println!("模式5: tokio::time::timeout(Duration, future).await");
    println!("  给异步操作设置超时\n");

    // ----------------------------------------------------------
    // 6. 异步错误处理
    // ----------------------------------------------------------
    println!("=== 异步错误处理 ===\n");

    println!(r#"
async fn process() -> Result<String, Box<dyn std::error::Error>> {{
    // ? 操作符在 async 中正常工作
    let data = fetch("url").await?;
    let parsed = parse(&data)?;
    Ok(parsed)
}}

// 重试模式
async fn fetch_with_retry(url: &str, max_retries: u32) -> Result<String, String> {{
    for attempt in 1..=max_retries {{
        match fetch(url).await {{
            Ok(data) => return Ok(data),
            Err(e) if attempt < max_retries => {{
                println!("尝试 {{}} 失败: {{}}, 重试...", attempt, e);
                tokio::time::sleep(Duration::from_secs(attempt as u64)).await;
            }}
            Err(e) => return Err(e),
        }}
    }}
    unreachable!()
}}
"#);

    // ----------------------------------------------------------
    // 7. Send + 'static 约束
    // ----------------------------------------------------------
    println!("=== Send + 'static ===\n");

    println!("tokio::spawn 要求 Future 是 Send + 'static");
    println!("这意味着:");
    println!("  - Future 中引用的数据必须是 Send 的（可跨线程）");
    println!("  - Future 不能包含非 'static 的引用");
    println!("");
    println!("常见错误:");
    println!("  - 在 .await 点持有 MutexGuard（不是 Send 的）");
    println!("  - 在 async 块中使用局部引用");
    println!("");
    println!("解决方法:");
    println!("  - 使用 tokio::sync::Mutex 代替 std::sync::Mutex");
    println!("  - 在 .await 之前释放锁");
    println!("  - 使用 Arc 共享数据");

    // ----------------------------------------------------------
    // 8. 同步 vs 异步的选择
    // ----------------------------------------------------------
    println!("\n=== 何时使用异步 ===\n");

    println!("✅ 使用异步:");
    println!("  - 大量并发 IO（Web 服务器、爬虫、微服务）");
    println!("  - 需要处理数千个连接");
    println!("  - 长时间等待的 IO 操作");
    println!("");
    println!("❌ 不使用异步:");
    println!("  - CPU 密集型计算（用线程 + rayon）");
    println!("  - 简单脚本和 CLI 工具");
    println!("  - 并发量很低的应用");

    println!("\n=== 异步编程结束 ===");
}

// ----------------------------------------------------------
// 手动实现 Future（了解底层原理）
// ----------------------------------------------------------
#[derive(Debug)]
struct Countdown {
    remaining: u32,
}

impl Future for Countdown {
    type Output = String;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.remaining == 0 {
            Poll::Ready("倒计时完成！".to_string())
        } else {
            println!("Countdown: 还剩 {}", self.remaining);
            self.remaining -= 1;
            cx.waker().wake_by_ref();  // 通知运行时再次 poll
            Poll::Pending
        }
    }
}
