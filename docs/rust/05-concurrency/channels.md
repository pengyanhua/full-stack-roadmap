# channels.rs

::: info 文件信息
- 📄 原文件：`02_channels.rs`
- 🔤 语言：rust
:::

## 完整代码

```rust
// ============================================================
//                      通道（Channels）
// ============================================================
// 通道是线程间通信的主要方式（CSP 模型）
// 【口号】"不要通过共享内存来通信，要通过通信来共享内存"
//
// Rust 标准库提供 mpsc（Multiple Producer, Single Consumer）通道
// mpsc::channel() — 无界通道
// mpsc::sync_channel(n) — 有界通道（缓冲区大小为 n）

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    println!("=== 通道 ===");

    // ----------------------------------------------------------
    // 1. 基本通道使用
    // ----------------------------------------------------------
    // tx = transmitter（发送端）
    // rx = receiver（接收端）

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let msg = String::from("你好，来自子线程");
        tx.send(msg).unwrap();
        // println!("{}", msg);  // 错误！msg 的所有权已通过 send 转移
    });

    // recv() 阻塞等待消息
    let received = rx.recv().unwrap();
    println!("收到: {}", received);

    // ----------------------------------------------------------
    // 2. 发送多条消息
    // ----------------------------------------------------------
    println!("\n=== 多条消息 ===");

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let messages = vec![
            String::from("消息1: 你好"),
            String::from("消息2: 世界"),
            String::from("消息3: Rust"),
            String::from("消息4: 并发"),
        ];

        for msg in messages {
            tx.send(msg).unwrap();
            thread::sleep(Duration::from_millis(50));
        }
        // tx 在此被丢弃，通道关闭
    });

    // 【技巧】将 rx 当作迭代器使用
    // 通道关闭时迭代自动结束
    for received in rx {
        println!("收到: {}", received);
    }

    // ----------------------------------------------------------
    // 3. 多个生产者（Multiple Producers）
    // ----------------------------------------------------------
    println!("\n=== 多个生产者 ===");

    let (tx, rx) = mpsc::channel();

    for i in 0..3 {
        let tx_clone = tx.clone();  // 克隆发送端
        thread::spawn(move || {
            let msg = format!("线程 {} 的消息", i);
            tx_clone.send(msg).unwrap();
        });
    }
    drop(tx);  // 【重要】必须丢弃原始 tx，否则通道不会关闭

    for received in rx {
        println!("收到: {}", received);
    }

    // ----------------------------------------------------------
    // 4. 有界通道（Sync Channel）
    // ----------------------------------------------------------
    // sync_channel(n) 创建容量为 n 的有界通道
    // 当缓冲区满时，send 会阻塞
    // 【用途】控制生产速度，防止内存溢出（背压机制）

    println!("\n=== 有界通道 ===");

    let (tx, rx) = mpsc::sync_channel(2);  // 缓冲区大小为 2

    thread::spawn(move || {
        for i in 1..=5 {
            println!("发送: {}", i);
            tx.send(i).unwrap();
            println!("已发送: {}", i);
        }
    });

    thread::sleep(Duration::from_millis(100));  // 让发送方先发几个

    for val in rx {
        println!("接收: {}", val);
        thread::sleep(Duration::from_millis(50));
    }

    // ----------------------------------------------------------
    // 5. try_recv（非阻塞接收）
    // ----------------------------------------------------------
    println!("\n=== 非阻塞接收 ===");

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        thread::sleep(Duration::from_millis(100));
        tx.send("延迟消息").unwrap();
    });

    // try_recv 不会阻塞
    loop {
        match rx.try_recv() {
            Ok(msg) => {
                println!("收到: {}", msg);
                break;
            }
            Err(mpsc::TryRecvError::Empty) => {
                println!("还没有消息，继续做其他事...");
                thread::sleep(Duration::from_millis(30));
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                println!("通道已关闭");
                break;
            }
        }
    }

    // ----------------------------------------------------------
    // 6. recv_timeout（带超时的接收）
    // ----------------------------------------------------------
    println!("\n=== 超时接收 ===");

    let (tx, rx) = mpsc::channel::<String>();

    // 不发送任何消息，直接丢弃 tx
    drop(tx);

    match rx.recv_timeout(Duration::from_millis(100)) {
        Ok(msg) => println!("收到: {}", msg),
        Err(mpsc::RecvTimeoutError::Timeout) => println!("超时！"),
        Err(mpsc::RecvTimeoutError::Disconnected) => println!("通道已断开"),
    }

    // ----------------------------------------------------------
    // 7. 实用模式：工作池（Worker Pool）
    // ----------------------------------------------------------
    println!("\n=== 工作池 ===");

    let (task_tx, task_rx) = mpsc::channel::<u64>();
    let (result_tx, result_rx) = mpsc::channel::<(u64, u64)>();

    let task_rx = std::sync::Arc::new(std::sync::Mutex::new(task_rx));

    // 启动 4 个工作线程
    let num_workers = 4;
    for id in 0..num_workers {
        let task_rx = std::sync::Arc::clone(&task_rx);
        let result_tx = result_tx.clone();

        thread::spawn(move || {
            loop {
                let task = {
                    let rx = task_rx.lock().unwrap();
                    rx.recv()
                };

                match task {
                    Ok(n) => {
                        // 模拟耗时计算（计算阶乘）
                        let result = (1..=n).product::<u64>();
                        println!("工作者 {}: {}! = {}", id, n, result);
                        result_tx.send((n, result)).unwrap();
                    }
                    Err(_) => break,  // 通道关闭，退出
                }
            }
        });
    }
    drop(result_tx);  // 丢弃原始发送端

    // 分发任务
    for n in 1..=10 {
        task_tx.send(n).unwrap();
    }
    drop(task_tx);  // 关闭任务通道

    // 收集结果
    let mut results: Vec<(u64, u64)> = result_rx.into_iter().collect();
    results.sort_by_key(|&(n, _)| n);

    println!("\n结果汇总:");
    for (n, result) in &results {
        println!("  {}! = {}", n, result);
    }

    // ----------------------------------------------------------
    // 8. 管道模式（Pipeline）
    // ----------------------------------------------------------
    println!("\n=== 管道模式 ===");

    // 阶段1: 生成数据
    let (tx1, rx1) = mpsc::channel();
    thread::spawn(move || {
        for i in 1..=5 {
            tx1.send(i).unwrap();
        }
    });

    // 阶段2: 平方
    let (tx2, rx2) = mpsc::channel();
    thread::spawn(move || {
        for val in rx1 {
            tx2.send(val * val).unwrap();
        }
    });

    // 阶段3: 过滤（只保留 > 10 的）
    let (tx3, rx3) = mpsc::channel();
    thread::spawn(move || {
        for val in rx2 {
            if val > 10 {
                tx3.send(val).unwrap();
            }
        }
    });

    // 收集最终结果
    let results: Vec<i32> = rx3.into_iter().collect();
    println!("管道结果 (1..=5 的平方中 > 10 的): {:?}", results);

    println!("\n=== 通道结束 ===");
}
```
