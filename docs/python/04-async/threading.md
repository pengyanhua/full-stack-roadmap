# threading.py

::: info 文件信息
- 📄 原文件：`01_threading.py`
- 🔤 语言：python
:::

Python 多线程编程
本文件介绍 Python 中的多线程编程基础。

【重要】Python 有 GIL（全局解释器锁），
限制了多线程在 CPU 密集型任务中的性能。
但对于 I/O 密集型任务，多线程仍然有效。

## 完整代码

```python
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue


def main01_thread_basics():
    """
    ============================================================
                    1. 线程基础
    ============================================================
    """
    print("=" * 60)
    print("1. 线程基础")
    print("=" * 60)

    # 【创建线程方式1：传入函数】
    def worker(name, delay):
        print(f"  线程 {name} 开始")
        time.sleep(delay)
        print(f"  线程 {name} 结束")

    print("方式1：传入函数")
    t1 = threading.Thread(target=worker, args=("A", 0.1))
    t2 = threading.Thread(target=worker, args=("B", 0.15))

    t1.start()
    t2.start()

    t1.join()  # 等待线程结束
    t2.join()
    print("所有线程完成")

    # 【创建线程方式2：继承 Thread 类】
    print(f"\n方式2：继承 Thread 类")

    class MyThread(threading.Thread):
        def __init__(self, name, delay):
            super().__init__()
            self.name = name
            self.delay = delay

        def run(self):
            print(f"  线程 {self.name} 开始")
            time.sleep(self.delay)
            print(f"  线程 {self.name} 结束")

    threads = [MyThread(f"T{i}", 0.05) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 【线程属性】
    print(f"\n--- 线程属性 ---")
    t = threading.Thread(target=lambda: None, name="MyThread")
    print(f"线程名: {t.name}")
    print(f"是否存活: {t.is_alive()}")
    print(f"是否守护线程: {t.daemon}")
    print(f"当前线程: {threading.current_thread().name}")
    print(f"活跃线程数: {threading.active_count()}")


def main02_daemon_thread():
    """
    ============================================================
                    2. 守护线程
    ============================================================
    守护线程在主线程结束时自动终止
    """
    print("\n" + "=" * 60)
    print("2. 守护线程")
    print("=" * 60)

    def background_task():
        while True:
            print("  后台任务运行中...")
            time.sleep(0.1)

    # 【守护线程】
    daemon = threading.Thread(target=background_task, daemon=True)
    daemon.start()

    print("主线程睡眠 0.3 秒")
    time.sleep(0.3)
    print("主线程结束，守护线程也会终止")

    # 【注意】daemon=True 必须在 start() 之前设置


def main03_lock():
    """
    ============================================================
                    3. 线程同步 - 锁
    ============================================================
    """
    print("\n" + "=" * 60)
    print("3. 线程同步 - 锁")
    print("=" * 60)

    # 【没有锁的问题：竞态条件】
    print("--- 竞态条件演示 ---")

    counter_unsafe = 0

    def increment_unsafe():
        global counter_unsafe
        for _ in range(100000):
            counter_unsafe += 1  # 不是原子操作！

    threads = [threading.Thread(target=increment_unsafe) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"无锁结果（应该是200000）: {counter_unsafe}")

    # 【使用锁解决】
    print(f"\n--- 使用锁 ---")

    counter_safe = 0
    lock = threading.Lock()

    def increment_safe():
        global counter_safe
        for _ in range(100000):
            with lock:  # 推荐用 with 语句
                counter_safe += 1

    threads = [threading.Thread(target=increment_safe) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"有锁结果: {counter_safe}")

    # 【RLock 可重入锁】
    print(f"\n--- RLock 可重入锁 ---")

    rlock = threading.RLock()

    def recursive_func(n):
        with rlock:  # RLock 允许同一线程多次获取
            if n > 0:
                print(f"  递归层级: {n}")
                recursive_func(n - 1)

    recursive_func(3)


def main04_condition_event():
    """
    ============================================================
                4. 条件变量和事件
    ============================================================
    """
    print("\n" + "=" * 60)
    print("4. 条件变量和事件")
    print("=" * 60)

    # 【Condition 条件变量】
    print("--- Condition 条件变量 ---")

    queue = []
    condition = threading.Condition()

    def producer():
        for i in range(5):
            time.sleep(0.05)
            with condition:
                queue.append(i)
                print(f"  生产者: 生产了 {i}")
                condition.notify()  # 通知消费者

    def consumer():
        for _ in range(5):
            with condition:
                while not queue:
                    condition.wait()  # 等待通知
                item = queue.pop(0)
                print(f"  消费者: 消费了 {item}")

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)
    t2.start()
    t1.start()
    t1.join()
    t2.join()

    # 【Event 事件】
    print(f"\n--- Event 事件 ---")

    event = threading.Event()

    def waiter(name):
        print(f"  {name} 等待事件...")
        event.wait()  # 阻塞直到事件被设置
        print(f"  {name} 收到事件!")

    def setter():
        time.sleep(0.1)
        print("  设置事件!")
        event.set()

    threads = [threading.Thread(target=waiter, args=(f"Waiter-{i}",)) for i in range(3)]
    for t in threads:
        t.start()

    threading.Thread(target=setter).start()

    for t in threads:
        t.join()


def main05_semaphore_barrier():
    """
    ============================================================
                5. 信号量和屏障
    ============================================================
    """
    print("\n" + "=" * 60)
    print("5. 信号量和屏障")
    print("=" * 60)

    # 【Semaphore 信号量】控制并发数量
    print("--- Semaphore 信号量 ---")

    semaphore = threading.Semaphore(3)  # 最多3个并发

    def limited_task(n):
        with semaphore:
            print(f"  任务 {n} 开始（最多3个并发）")
            time.sleep(0.1)
            print(f"  任务 {n} 结束")

    threads = [threading.Thread(target=limited_task, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 【Barrier 屏障】等待多个线程到达同一点
    print(f"\n--- Barrier 屏障 ---")

    barrier = threading.Barrier(3)  # 3个线程同步

    def synchronized_task(n):
        print(f"  线程 {n} 准备就绪")
        barrier.wait()  # 等待所有线程到达
        print(f"  线程 {n} 同时开始执行!")

    threads = [threading.Thread(target=synchronized_task, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main06_thread_local():
    """
    ============================================================
                6. 线程局部数据
    ============================================================
    """
    print("\n" + "=" * 60)
    print("6. 线程局部数据")
    print("=" * 60)

    # 【ThreadLocal】每个线程独立的数据
    local_data = threading.local()

    def worker(name):
        local_data.name = name  # 每个线程有自己的 name
        time.sleep(0.01)
        print(f"  线程 {threading.current_thread().name}: local_data.name = {local_data.name}")

    threads = [threading.Thread(target=worker, args=(f"Worker-{i}",)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main07_queue():
    """
    ============================================================
                7. 线程安全队列
    ============================================================
    """
    print("\n" + "=" * 60)
    print("7. 线程安全队列")
    print("=" * 60)

    # 【Queue】线程安全的 FIFO 队列
    q = Queue(maxsize=5)

    def producer():
        for i in range(5):
            q.put(i)  # 阻塞直到有空间
            print(f"  生产: {i}")
            time.sleep(0.02)
        q.put(None)  # 结束信号

    def consumer():
        while True:
            item = q.get()  # 阻塞直到有数据
            if item is None:
                break
            print(f"  消费: {item}")
            q.task_done()  # 标记任务完成

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    # 【其他队列类型】
    print(f"\n--- 其他队列类型 ---")
    print("queue.Queue: FIFO 队列")
    print("queue.LifoQueue: LIFO 栈")
    print("queue.PriorityQueue: 优先级队列")


def main08_thread_pool():
    """
    ============================================================
                8. 线程池
    ============================================================
    """
    print("\n" + "=" * 60)
    print("8. 线程池")
    print("=" * 60)

    # 【ThreadPoolExecutor】
    def task(n):
        time.sleep(0.05)
        return n * n

    # 【方式1：使用 map】
    print("方式1：使用 map")
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(task, range(5))
        print(f"  结果: {list(results)}")

    # 【方式2：使用 submit + as_completed】
    print(f"\n方式2：使用 submit + as_completed")
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(task, i): i for i in range(5)}
        for future in as_completed(futures):
            n = futures[future]
            result = future.result()
            print(f"  task({n}) = {result}")

    # 【获取异常】
    print(f"\n--- 处理异常 ---")

    def risky_task(n):
        if n == 2:
            raise ValueError(f"错误：n={n}")
        return n

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(risky_task, i) for i in range(5)]
        for i, future in enumerate(futures):
            try:
                result = future.result()
                print(f"  task({i}) = {result}")
            except Exception as e:
                print(f"  task({i}) 异常: {e}")


def main09_gil():
    """
    ============================================================
                9. GIL（全局解释器锁）
    ============================================================
    """
    print("\n" + "=" * 60)
    print("9. GIL（全局解释器锁）")
    print("=" * 60)

    print("""
    【GIL 是什么】
    - Global Interpreter Lock（全局解释器锁）
    - 确保同一时刻只有一个线程执行 Python 字节码
    - 是 CPython 实现的特性，不是 Python 语言规范

    【GIL 的影响】
    - CPU 密集型：多线程无法利用多核，性能可能更差
    - I/O 密集型：GIL 在 I/O 等待时释放，多线程仍有效

    【解决方案】
    - CPU 密集型：使用 multiprocessing 多进程
    - I/O 密集型：多线程仍然有效
    - 使用 C 扩展或 Cython
    - 使用其他 Python 实现（Jython, IronPython）
    """)

    # 【演示：I/O 密集型适合多线程】
    print("--- I/O 密集型任务（模拟网络请求）---")

    def io_task(n):
        time.sleep(0.1)  # 模拟 I/O
        return n

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(io_task, range(8)))
    elapsed = time.perf_counter() - start
    print(f"  多线程耗时: {elapsed:.2f}秒（并行 I/O 有效）")

    # 串行执行对比
    start = time.perf_counter()
    results = [io_task(i) for i in range(8)]
    elapsed = time.perf_counter() - start
    print(f"  串行耗时: {elapsed:.2f}秒")


if __name__ == "__main__":
    main01_thread_basics()
    main02_daemon_thread()
    main03_lock()
    main04_condition_event()
    main05_semaphore_barrier()
    main06_thread_local()
    main07_queue()
    main08_thread_pool()
    main09_gil()
```
