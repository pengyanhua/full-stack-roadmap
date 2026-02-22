// ============================================================
//                      线程（Threads）
// ============================================================
// Rust 的并发模型基于操作系统线程（1:1 模型）
// 所有权和类型系统在编译时就能防止数据竞争
// 【口号】"无畏并发"（Fearless Concurrency）
//
// 关键概念：
// - Send trait: 类型可以安全地在线程间转移所有权
// - Sync trait: 类型可以安全地在线程间共享引用
// 大部分类型都自动实现了这两个 trait

use std::thread;
use std::time::Duration;
use std::sync::{Arc, Mutex, RwLock};

fn main() {
    println!("=== 线程 ===");

    // ----------------------------------------------------------
    // 1. 创建线程
    // ----------------------------------------------------------
    // thread::spawn 创建新线程，接受一个闭包
    // 返回 JoinHandle，可以等待线程结束

    let handle = thread::spawn(|| {
        for i in 1..=5 {
            println!("子线程: {}", i);
            thread::sleep(Duration::from_millis(10));
        }
    });

    for i in 1..=3 {
        println!("主线程: {}", i);
        thread::sleep(Duration::from_millis(10));
    }

    // join() 等待线程完成
    // 【重要】如果不 join，主线程结束时子线程会被终止
    handle.join().unwrap();
    println!("所有线程完成\n");

    // ----------------------------------------------------------
    // 2. move 闭包（转移所有权到线程）
    // ----------------------------------------------------------
    // 线程可能比创建它的作用域活得更久
    // 所以必须用 move 获取数据的所有权

    let data = vec![1, 2, 3];
    let handle = thread::spawn(move || {
        // data 的所有权被移入线程
        println!("线程中的数据: {:?}", data);
    });
    // println!("{:?}", data);  // 错误！data 已被移动
    handle.join().unwrap();

    // ----------------------------------------------------------
    // 3. 返回值
    // ----------------------------------------------------------
    let handle = thread::spawn(|| {
        let mut sum = 0;
        for i in 1..=100 {
            sum += i;
        }
        sum  // 返回值
    });

    let result = handle.join().unwrap();
    println!("1 到 100 的和: {}\n", result);

    // ----------------------------------------------------------
    // 4. Mutex<T>（互斥锁）
    // ----------------------------------------------------------
    // Mutex 提供内部可变性，同一时刻只有一个线程能访问数据
    // lock() 返回 MutexGuard，离开作用域时自动解锁
    // 【注意】Mutex 在单线程中用 lock().unwrap() 即可
    //        在多线程中需要配合 Arc 使用

    println!("=== Mutex ===");

    let counter = Mutex::new(0);
    {
        let mut num = counter.lock().unwrap();
        *num += 1;
    } // MutexGuard 在此释放，锁被自动解开
    println!("单线程 Mutex: {}", *counter.lock().unwrap());

    // ----------------------------------------------------------
    // 5. Arc<T> + Mutex<T>（多线程共享数据）
    // ----------------------------------------------------------
    // Arc = Atomic Reference Counting（原子引用计数）
    // 【为什么不用 Rc】Rc 不是线程安全的，Arc 是
    // 【模式】Arc<Mutex<T>> 是多线程共享可变数据的标准模式

    println!("\n=== Arc + Mutex ===");

    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);  // 克隆 Arc（增加引用计数）
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("最终计数: {}", *counter.lock().unwrap()); // 10

    // ----------------------------------------------------------
    // 6. RwLock<T>（读写锁）
    // ----------------------------------------------------------
    // 允许多个读者或一个写者（与借用规则类似）
    // 【适用场景】读多写少的情况，比 Mutex 性能更好

    println!("\n=== RwLock ===");

    let data = Arc::new(RwLock::new(vec![1, 2, 3]));
    let mut handles = vec![];

    // 多个读者
    for i in 0..3 {
        let data = Arc::clone(&data);
        let handle = thread::spawn(move || {
            let read_guard = data.read().unwrap();
            println!("读者 {}: {:?}", i, *read_guard);
        });
        handles.push(handle);
    }

    // 一个写者
    {
        let data = Arc::clone(&data);
        let handle = thread::spawn(move || {
            let mut write_guard = data.write().unwrap();
            write_guard.push(4);
            println!("写者: 添加了 4");
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("最终数据: {:?}", *data.read().unwrap());

    // ----------------------------------------------------------
    // 7. 线程局部存储
    // ----------------------------------------------------------
    println!("\n=== 线程局部存储 ===");

    thread_local! {
        static COUNTER: std::cell::RefCell<u32> = std::cell::RefCell::new(0);
    }

    let mut handles = vec![];
    for id in 0..3 {
        let handle = thread::spawn(move || {
            COUNTER.with(|c| {
                *c.borrow_mut() += 1;
                println!("线程 {}: 局部计数 = {}", id, c.borrow());
            });
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // 主线程的计数器是独立的
    COUNTER.with(|c| {
        println!("主线程: 局部计数 = {}", c.borrow()); // 0
    });

    // ----------------------------------------------------------
    // 8. 实用示例：并行计算
    // ----------------------------------------------------------
    println!("\n=== 并行计算 ===");

    let data: Vec<u64> = (1..=1_000_000).collect();

    // 分块并行求和
    let num_threads = 4;
    let chunk_size = data.len() / num_threads;
    let data = Arc::new(data);
    let mut handles = vec![];

    for i in 0..num_threads {
        let data = Arc::clone(&data);
        let handle = thread::spawn(move || {
            let start = i * chunk_size;
            let end = if i == num_threads - 1 {
                data.len()
            } else {
                (i + 1) * chunk_size
            };
            let partial_sum: u64 = data[start..end].iter().sum();
            println!("线程 {}: 部分和 = {}", i, partial_sum);
            partial_sum
        });
        handles.push(handle);
    }

    let total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
    println!("总和: {}", total);
    println!("验证: {}", (1u64..=1_000_000).sum::<u64>());

    println!("\n=== 线程结束 ===");
}
