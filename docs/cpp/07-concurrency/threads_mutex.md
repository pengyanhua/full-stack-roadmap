# threads mutex.cpp

::: info 文件信息
- 📄 原文件：`01_threads_mutex.cpp`
- 🔤 语言：cpp
:::

## 完整代码

```cpp
// ============================================================
//                      并发编程
// ============================================================
// C++11 引入标准线程库，无需依赖 pthread 或 Win32 API
// thread：OS 线程包装
// mutex / lock_guard / unique_lock：互斥锁
// condition_variable：条件变量（生产者-消费者）
// future / promise / async：异步任务
// atomic：无锁原子操作

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>
#include <vector>
#include <queue>
#include <string>
#include <chrono>
#include <functional>
#include <numeric>

using namespace std::chrono_literals;

// ============================================================
//                      基本线程
// ============================================================

void thread_function(int id, int count) {
    for (int i = 0; i < count; i++) {
        std::cout << "线程 " << id << ": " << i << std::endl;
        std::this_thread::sleep_for(10ms);
    }
}

void demo_thread() {
    std::cout << "=== 基本线程 ===" << std::endl;

    // 创建线程
    std::thread t1(thread_function, 1, 3);
    std::thread t2(thread_function, 2, 3);

    // lambda 线程
    std::thread t3([]{
        std::cout << "Lambda 线程 ID: " << std::this_thread::get_id() << std::endl;
    });

    // join：等待线程完成
    t1.join();
    t2.join();
    t3.join();

    std::cout << "主线程 ID: " << std::this_thread::get_id() << std::endl;
    std::cout << "硬件并发数: " << std::thread::hardware_concurrency() << std::endl;
}

// ============================================================
//                      互斥锁
// ============================================================

std::mutex g_mutex;
int g_counter = 0;

void increment(int n, const std::string& label) {
    for (int i = 0; i < n; i++) {
        // lock_guard：RAII 锁，超出作用域自动解锁
        std::lock_guard<std::mutex> lock(g_mutex);
        ++g_counter;
    }
}

void demo_mutex() {
    std::cout << "\n=== mutex + lock_guard ===" << std::endl;

    g_counter = 0;
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; i++) {
        threads.emplace_back(increment, 1000, "T" + std::to_string(i));
    }
    for (auto& t : threads) t.join();

    std::cout << "期望 10000，实际: " << g_counter << std::endl;

    // unique_lock：比 lock_guard 更灵活（可手动 lock/unlock、延迟锁）
    std::mutex m;
    std::unique_lock<std::mutex> ul(m, std::defer_lock);
    ul.lock();
    std::cout << "unique_lock 已加锁" << std::endl;
    ul.unlock();
    std::cout << "unique_lock 已解锁" << std::endl;

    // 避免死锁：std::lock 同时锁多个 mutex
    std::mutex m1, m2;
    std::lock(m1, m2);  // 原子地同时锁两个
    std::lock_guard<std::mutex> lg1(m1, std::adopt_lock);
    std::lock_guard<std::mutex> lg2(m2, std::adopt_lock);
    std::cout << "两个 mutex 同时锁定（无死锁）" << std::endl;
}

// ============================================================
//                      条件变量（生产者-消费者）
// ============================================================

class BoundedQueue {
    std::queue<int>         queue_;
    std::mutex              mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    const size_t            max_size_;
    bool                    done_ = false;

public:
    explicit BoundedQueue(size_t max) : max_size_(max) {}

    void push(int val) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] { return queue_.size() < max_size_ || done_; });
        if (!done_) {
            queue_.push(val);
            not_empty_.notify_one();
        }
    }

    bool pop(int& val) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return !queue_.empty() || done_; });
        if (queue_.empty()) return false;
        val = queue_.front();
        queue_.pop();
        not_full_.notify_one();
        return true;
    }

    void done() {
        std::unique_lock<std::mutex> lock(mutex_);
        done_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    size_t size() const { return queue_.size(); }
};

void demo_condition_variable() {
    std::cout << "\n=== 条件变量（生产者-消费者）===" << std::endl;

    BoundedQueue q(5);
    std::vector<int> consumed;
    std::mutex print_mutex;

    // 生产者
    auto producer = std::thread([&] {
        for (int i = 1; i <= 10; i++) {
            q.push(i);
            std::lock_guard<std::mutex> lk(print_mutex);
            std::cout << "  生产: " << i << std::endl;
            std::this_thread::sleep_for(20ms);
        }
        q.done();
    });

    // 消费者
    auto consumer = std::thread([&] {
        int val;
        while (q.pop(val)) {
            std::lock_guard<std::mutex> lk(print_mutex);
            std::cout << "  消费: " << val << std::endl;
            consumed.push_back(val);
            std::this_thread::sleep_for(40ms);
        }
    });

    producer.join();
    consumer.join();
    std::cout << "共消费 " << consumed.size() << " 个元素" << std::endl;
}

// ============================================================
//                      future / promise / async
// ============================================================

// 耗时计算
long long heavy_compute(int n) {
    std::this_thread::sleep_for(50ms);
    long long sum = 0;
    for (int i = 0; i <= n; i++) sum += i;
    return sum;
}

void demo_future() {
    std::cout << "\n=== future / async ===" << std::endl;

    // async：异步执行函数，返回 future
    auto f1 = std::async(std::launch::async, heavy_compute, 1000000);
    auto f2 = std::async(std::launch::async, heavy_compute, 2000000);
    auto f3 = std::async(std::launch::async, heavy_compute, 3000000);

    std::cout << "异步任务已启动，继续其他工作..." << std::endl;

    // 等待并获取结果（阻塞到任务完成）
    auto r1 = f1.get();
    auto r2 = f2.get();
    auto r3 = f3.get();

    std::cout << "结果: " << r1 << ", " << r2 << ", " << r3 << std::endl;

    // promise：手动设置 future 的值
    std::promise<std::string> prom;
    auto fut = prom.get_future();

    std::thread([&prom] {
        std::this_thread::sleep_for(30ms);
        prom.set_value("来自另一个线程的消息");
    }).detach();

    std::cout << "等待 promise 的值..." << std::endl;
    std::cout << "收到: " << fut.get() << std::endl;

    // 异常传播
    auto failing_future = std::async(std::launch::async, [] {
        throw std::runtime_error("异步任务出错！");
        return 0;
    });

    try {
        failing_future.get();
    } catch (const std::exception& e) {
        std::cout << "捕获异步异常: " << e.what() << std::endl;
    }
}

// ============================================================
//                      atomic（无锁原子操作）
// ============================================================

void demo_atomic() {
    std::cout << "\n=== atomic ===" << std::endl;

    std::atomic<int> counter{0};
    std::atomic<bool> flag{false};

    std::vector<std::thread> threads;
    for (int i = 0; i < 10; i++) {
        threads.emplace_back([&counter] {
            for (int j = 0; j < 1000; j++)
                counter.fetch_add(1, std::memory_order_relaxed);
        });
    }
    for (auto& t : threads) t.join();

    std::cout << "原子计数器（期望10000）: " << counter.load() << std::endl;

    // CAS（Compare And Swap）
    int expected = 10000;
    bool swapped = counter.compare_exchange_strong(expected, 0);
    std::cout << "CAS 成功: " << swapped << "，新值: " << counter.load() << std::endl;

    // 无锁标志
    auto worker = std::thread([&flag] {
        std::this_thread::sleep_for(20ms);
        flag.store(true, std::memory_order_release);
    });

    while (!flag.load(std::memory_order_acquire)) {
        std::this_thread::yield();  // 让出 CPU
    }
    std::cout << "flag 已设置，工作完成" << std::endl;
    worker.join();
}

// ============================================================
//                      线程池（简单实现）
// ============================================================

class ThreadPool {
    std::vector<std::thread>          workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex                        mutex_;
    std::condition_variable           cv_;
    bool                              stop_ = false;

public:
    explicit ThreadPool(size_t n) {
        for (size_t i = 0; i < n; i++) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    template <typename F>
    auto submit(F&& f) -> std::future<decltype(f())> {
        auto task = std::make_shared<std::packaged_task<decltype(f())()>>(std::forward<F>(f));
        auto fut = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.emplace([task] { (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }
};

void demo_thread_pool() {
    std::cout << "\n=== 线程池 ===" << std::endl;

    ThreadPool pool(4);

    std::vector<std::future<int>> futures;
    for (int i = 1; i <= 8; i++) {
        futures.push_back(pool.submit([i] {
            std::this_thread::sleep_for(20ms);
            return i * i;
        }));
    }

    std::cout << "平方结果: ";
    for (auto& f : futures)
        std::cout << f.get() << " ";
    std::cout << std::endl;
}

// ============================================================
//                      主函数
// ============================================================
int main() {
    demo_thread();
    demo_mutex();
    demo_condition_variable();
    demo_future();
    demo_atomic();
    demo_thread_pool();

    std::cout << "\n=== 并发编程演示完成 ===" << std::endl;
    return 0;
}
```
