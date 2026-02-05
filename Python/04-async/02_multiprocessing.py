#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
                Python 多进程编程
============================================================
本文件介绍 Python 中的多进程编程。

多进程可以绑过 GIL 限制，充分利用多核 CPU。
适合 CPU 密集型任务。
============================================================
"""
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def cpu_bound_task(n):
    """CPU 密集型任务"""
    total = 0
    for i in range(n):
        total += i * i
    return total


def worker_with_info(name):
    """带进程信息的工作函数"""
    info = f"进程 {name}: PID={os.getpid()}, 父PID={os.getppid()}"
    time.sleep(0.1)
    return info


def main01_process_basics():
    """
    ============================================================
                    1. 进程基础
    ============================================================
    """
    print("=" * 60)
    print("1. 进程基础")
    print("=" * 60)

    print(f"主进程 PID: {os.getpid()}")

    # 【创建进程】
    def worker(name):
        print(f"  子进程 {name} 开始，PID={os.getpid()}")
        time.sleep(0.1)
        print(f"  子进程 {name} 结束")

    print("\n创建进程:")
    p1 = mp.Process(target=worker, args=("A",))
    p2 = mp.Process(target=worker, args=("B",))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print("所有进程完成")

    # 【进程属性】
    print(f"\n--- 进程属性 ---")
    p = mp.Process(target=lambda: None, name="MyProcess")
    print(f"进程名: {p.name}")
    print(f"PID: {p.pid}")  # 启动前为 None
    print(f"是否存活: {p.is_alive()}")
    print(f"是否守护进程: {p.daemon}")


def main02_process_communication():
    """
    ============================================================
                2. 进程间通信
    ============================================================
    """
    print("\n" + "=" * 60)
    print("2. 进程间通信")
    print("=" * 60)

    # 【Queue 队列】
    print("--- Queue 队列 ---")

    def producer(q):
        for i in range(5):
            q.put(i)
            print(f"  生产: {i}")
        q.put(None)  # 结束信号

    def consumer(q):
        while True:
            item = q.get()
            if item is None:
                break
            print(f"  消费: {item}")

    q = mp.Queue()
    p1 = mp.Process(target=producer, args=(q,))
    p2 = mp.Process(target=consumer, args=(q,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    # 【Pipe 管道】
    print(f"\n--- Pipe 管道 ---")

    def sender(conn):
        for i in range(3):
            conn.send(f"消息 {i}")
            time.sleep(0.05)
        conn.close()

    def receiver(conn):
        while True:
            try:
                msg = conn.recv()
                print(f"  收到: {msg}")
            except EOFError:
                break

    parent_conn, child_conn = mp.Pipe()
    p1 = mp.Process(target=sender, args=(child_conn,))
    p2 = mp.Process(target=receiver, args=(parent_conn,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()


def main03_shared_memory():
    """
    ============================================================
                3. 共享内存
    ============================================================
    """
    print("\n" + "=" * 60)
    print("3. 共享内存")
    print("=" * 60)

    # 【Value 和 Array】
    print("--- Value 和 Array ---")

    def increment_shared(val, arr):
        for _ in range(100):
            val.value += 1
        for i in range(len(arr)):
            arr[i] += 1

    shared_value = mp.Value('i', 0)  # 'i' 表示整数
    shared_array = mp.Array('d', [0.0, 0.0, 0.0])  # 'd' 表示双精度浮点

    processes = [
        mp.Process(target=increment_shared, args=(shared_value, shared_array))
        for _ in range(2)
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print(f"共享值（可能有竞态）: {shared_value.value}")
    print(f"共享数组: {list(shared_array)}")

    # 【使用锁保护共享数据】
    print(f"\n--- 使用锁保护 ---")

    def safe_increment(val, lock):
        for _ in range(100):
            with lock:
                val.value += 1

    shared_value = mp.Value('i', 0)
    lock = mp.Lock()

    processes = [
        mp.Process(target=safe_increment, args=(shared_value, lock))
        for _ in range(2)
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print(f"使用锁后的值: {shared_value.value}")

    # 【Manager 共享复杂对象】
    print(f"\n--- Manager 共享对象 ---")

    def modify_shared_dict(d, l):
        d['count'] = d.get('count', 0) + 1
        l.append(os.getpid())

    with mp.Manager() as manager:
        shared_dict = manager.dict()
        shared_list = manager.list()

        processes = [
            mp.Process(target=modify_shared_dict, args=(shared_dict, shared_list))
            for _ in range(3)
        ]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        print(f"共享字典: {dict(shared_dict)}")
        print(f"共享列表: {list(shared_list)}")


def main04_process_pool():
    """
    ============================================================
                4. 进程池
    ============================================================
    """
    print("\n" + "=" * 60)
    print("4. 进程池")
    print("=" * 60)

    # 【Pool】
    print("--- multiprocessing.Pool ---")

    def square(x):
        return x * x

    with mp.Pool(processes=4) as pool:
        # map: 同步执行
        results = pool.map(square, range(10))
        print(f"map 结果: {results}")

        # apply: 单个任务
        result = pool.apply(square, (5,))
        print(f"apply 结果: {result}")

        # map_async: 异步执行
        async_result = pool.map_async(square, range(5))
        print(f"map_async 结果: {async_result.get()}")

    # 【ProcessPoolExecutor】推荐使用
    print(f"\n--- ProcessPoolExecutor ---")

    with ProcessPoolExecutor(max_workers=4) as executor:
        # map
        results = list(executor.map(square, range(10)))
        print(f"map 结果: {results}")

        # submit + as_completed
        futures = [executor.submit(square, i) for i in range(5)]
        for future in as_completed(futures):
            print(f"  完成: {future.result()}")


def main05_cpu_bound_comparison():
    """
    ============================================================
            5. CPU 密集型任务对比
    ============================================================
    """
    print("\n" + "=" * 60)
    print("5. CPU 密集型任务对比")
    print("=" * 60)

    n = 1000000
    tasks = [n] * 4

    # 【串行执行】
    print("串行执行:")
    start = time.perf_counter()
    results = [cpu_bound_task(t) for t in tasks]
    serial_time = time.perf_counter() - start
    print(f"  耗时: {serial_time:.2f}秒")

    # 【多进程执行】
    print(f"\n多进程执行:")
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(cpu_bound_task, tasks))
    parallel_time = time.perf_counter() - start
    print(f"  耗时: {parallel_time:.2f}秒")
    print(f"  加速比: {serial_time / parallel_time:.2f}x")


def main06_process_context():
    """
    ============================================================
                6. 进程启动方式
    ============================================================
    """
    print("\n" + "=" * 60)
    print("6. 进程启动方式")
    print("=" * 60)

    print("""
    【进程启动方式】

    1. spawn（默认在 Windows 和 macOS）
       - 启动新的 Python 解释器
       - 只继承必要的资源
       - 更安全，但启动较慢

    2. fork（默认在 Unix）
       - 使用 os.fork()
       - 继承父进程的所有资源
       - 启动快，但可能有问题（如多线程程序）

    3. forkserver
       - 启动服务器进程，由服务器 fork 新进程
       - 结合了 spawn 和 fork 的优点
    """)

    # 【设置启动方式】
    # mp.set_start_method('spawn')  # 必须在主模块的 if __name__ == '__main__' 块中

    print(f"当前启动方式: {mp.get_start_method()}")

    # 【使用 get_context 创建特定启动方式的进程】
    ctx = mp.get_context('spawn')
    print(f"使用 spawn 上下文")


def main07_practical_example():
    """
    ============================================================
                7. 实际应用示例
    ============================================================
    """
    print("\n" + "=" * 60)
    print("7. 实际应用示例：并行图像处理（模拟）")
    print("=" * 60)

    def process_image(image_path):
        """模拟图像处理"""
        time.sleep(0.1)  # 模拟处理时间
        return f"处理完成: {image_path}"

    images = [f"image_{i}.jpg" for i in range(8)]

    print("串行处理:")
    start = time.perf_counter()
    results = [process_image(img) for img in images]
    print(f"  耗时: {time.perf_counter() - start:.2f}秒")

    print(f"\n并行处理:")
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_image, images))
    print(f"  耗时: {time.perf_counter() - start:.2f}秒")

    for r in results[:3]:
        print(f"  {r}")


if __name__ == "__main__":
    # 【重要】Windows 上必须在 if __name__ == '__main__' 块中运行多进程代码
    main01_process_basics()
    main02_process_communication()
    main03_shared_memory()
    main04_process_pool()
    main05_cpu_bound_comparison()
    main06_process_context()
    main07_practical_example()
