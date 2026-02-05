#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
                Python asyncio 异步编程
============================================================
本文件介绍 Python 中的 asyncio 异步编程。

asyncio 是 Python 的异步 I/O 框架，使用协程实现并发。
适合 I/O 密集型任务，如网络请求、文件操作等。
============================================================
"""
import asyncio
import time
from typing import List


async def main01_coroutine_basics():
    """
    ============================================================
                    1. 协程基础
    ============================================================
    """
    print("=" * 60)
    print("1. 协程基础")
    print("=" * 60)

    # 【定义协程】使用 async def
    async def hello():
        print("  Hello")
        await asyncio.sleep(0.1)  # 异步等待
        print("  World")
        return "完成"

    # 【运行协程】
    print("运行协程:")
    result = await hello()
    print(f"返回值: {result}")

    # 【协程对象】
    print(f"\n--- 协程对象 ---")
    coro = hello()  # 创建协程对象，不执行
    print(f"协程对象: {coro}")
    await coro  # 执行协程

    # 【await 关键字】
    print(f"\n--- await 关键字 ---")
    print("await 用于等待:")
    print("  - 协程 (coroutine)")
    print("  - Task 对象")
    print("  - Future 对象")
    print("  - 任何实现 __await__ 的对象")


async def main02_tasks():
    """
    ============================================================
                    2. 任务 (Task)
    ============================================================
    """
    print("\n" + "=" * 60)
    print("2. 任务 (Task)")
    print("=" * 60)

    async def fetch_data(name, delay):
        print(f"  开始获取 {name}")
        await asyncio.sleep(delay)
        print(f"  完成获取 {name}")
        return f"{name} 的数据"

    # 【创建任务】asyncio.create_task()
    print("创建任务并发执行:")
    start = time.perf_counter()

    task1 = asyncio.create_task(fetch_data("A", 0.2))
    task2 = asyncio.create_task(fetch_data("B", 0.1))
    task3 = asyncio.create_task(fetch_data("C", 0.15))

    # 等待所有任务完成
    result1 = await task1
    result2 = await task2
    result3 = await task3

    elapsed = time.perf_counter() - start
    print(f"总耗时: {elapsed:.2f}秒（并发执行）")
    print(f"结果: {result1}, {result2}, {result3}")

    # 【asyncio.gather】同时等待多个协程
    print(f"\n--- asyncio.gather ---")
    start = time.perf_counter()

    results = await asyncio.gather(
        fetch_data("X", 0.1),
        fetch_data("Y", 0.15),
        fetch_data("Z", 0.2),
    )

    elapsed = time.perf_counter() - start
    print(f"gather 耗时: {elapsed:.2f}秒")
    print(f"结果: {results}")

    # 【gather 处理异常】
    print(f"\n--- gather 处理异常 ---")

    async def maybe_fail(n):
        if n == 2:
            raise ValueError(f"任务 {n} 失败")
        await asyncio.sleep(0.05)
        return n

    # return_exceptions=True 返回异常而不是抛出
    results = await asyncio.gather(
        maybe_fail(1),
        maybe_fail(2),
        maybe_fail(3),
        return_exceptions=True
    )
    print(f"结果（含异常）: {results}")


async def main03_wait_and_timeout():
    """
    ============================================================
                3. 等待和超时
    ============================================================
    """
    print("\n" + "=" * 60)
    print("3. 等待和超时")
    print("=" * 60)

    async def slow_task(n, delay):
        await asyncio.sleep(delay)
        return f"任务{n}完成"

    # 【asyncio.wait】更细粒度的控制
    print("--- asyncio.wait ---")

    tasks = [
        asyncio.create_task(slow_task(i, 0.1 * i))
        for i in range(1, 4)
    ]

    # 等待第一个完成
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )
    print(f"第一个完成: {[t.result() for t in done]}")
    print(f"待完成数: {len(pending)}")

    # 等待剩余任务
    if pending:
        done, _ = await asyncio.wait(pending)
        print(f"剩余完成: {[t.result() for t in done]}")

    # 【asyncio.wait_for】超时控制
    print(f"\n--- asyncio.wait_for 超时 ---")

    async def long_task():
        await asyncio.sleep(1)
        return "完成"

    try:
        result = await asyncio.wait_for(long_task(), timeout=0.1)
    except asyncio.TimeoutError:
        print("任务超时!")

    # 【asyncio.timeout】Python 3.11+ 上下文管理器
    print(f"\n--- asyncio.timeout (Python 3.11+) ---")
    try:
        async with asyncio.timeout(0.1):
            await asyncio.sleep(1)
    except asyncio.TimeoutError:
        print("上下文超时!")


async def main04_semaphore_lock():
    """
    ============================================================
                4. 异步同步原语
    ============================================================
    """
    print("\n" + "=" * 60)
    print("4. 异步同步原语")
    print("=" * 60)

    # 【asyncio.Lock】异步锁
    print("--- asyncio.Lock ---")

    lock = asyncio.Lock()
    shared_resource = []

    async def safe_append(item):
        async with lock:
            print(f"  获取锁，添加 {item}")
            shared_resource.append(item)
            await asyncio.sleep(0.05)
            print(f"  释放锁")

    await asyncio.gather(*[safe_append(i) for i in range(3)])
    print(f"结果: {shared_resource}")

    # 【asyncio.Semaphore】限制并发数
    print(f"\n--- asyncio.Semaphore ---")

    semaphore = asyncio.Semaphore(2)  # 最多2个并发

    async def limited_task(n):
        async with semaphore:
            print(f"  任务 {n} 开始")
            await asyncio.sleep(0.1)
            print(f"  任务 {n} 结束")

    await asyncio.gather(*[limited_task(i) for i in range(5)])

    # 【asyncio.Event】事件
    print(f"\n--- asyncio.Event ---")

    event = asyncio.Event()

    async def waiter(name):
        print(f"  {name} 等待事件")
        await event.wait()
        print(f"  {name} 收到事件!")

    async def setter():
        await asyncio.sleep(0.1)
        print("  设置事件!")
        event.set()

    await asyncio.gather(
        waiter("A"),
        waiter("B"),
        setter()
    )

    # 【asyncio.Queue】异步队列
    print(f"\n--- asyncio.Queue ---")

    queue: asyncio.Queue = asyncio.Queue()

    async def producer():
        for i in range(3):
            await queue.put(i)
            print(f"  生产: {i}")
            await asyncio.sleep(0.05)

    async def consumer():
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.2)
                print(f"  消费: {item}")
                queue.task_done()
            except asyncio.TimeoutError:
                break

    await asyncio.gather(producer(), consumer())


async def main05_async_generators():
    """
    ============================================================
                5. 异步生成器和迭代器
    ============================================================
    """
    print("\n" + "=" * 60)
    print("5. 异步生成器和迭代器")
    print("=" * 60)

    # 【异步生成器】async def + yield
    async def async_range(start, stop):
        for i in range(start, stop):
            await asyncio.sleep(0.05)
            yield i

    print("异步生成器:")
    async for num in async_range(0, 5):
        print(f"  {num}")

    # 【异步列表推导】
    print(f"\n异步列表推导:")
    result = [x async for x in async_range(0, 3)]
    print(f"结果: {result}")

    # 【异步上下文管理器】
    print(f"\n--- 异步上下文管理器 ---")

    class AsyncResource:
        async def __aenter__(self):
            print("  获取资源")
            await asyncio.sleep(0.05)
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            print("  释放资源")
            await asyncio.sleep(0.05)
            return False

        async def do_something(self):
            print("  使用资源")

    async with AsyncResource() as resource:
        await resource.do_something()


async def main06_practical_examples():
    """
    ============================================================
                6. 实际应用示例
    ============================================================
    """
    print("\n" + "=" * 60)
    print("6. 实际应用示例")
    print("=" * 60)

    # 【模拟并发 HTTP 请求】
    print("--- 模拟并发 HTTP 请求 ---")

    async def fetch_url(url: str) -> dict:
        """模拟 HTTP 请求"""
        delay = 0.1 + (hash(url) % 10) / 100
        await asyncio.sleep(delay)
        return {"url": url, "status": 200}

    urls = [
        "https://api.example.com/users",
        "https://api.example.com/posts",
        "https://api.example.com/comments",
        "https://api.example.com/albums",
    ]

    start = time.perf_counter()
    results = await asyncio.gather(*[fetch_url(url) for url in urls])
    elapsed = time.perf_counter() - start

    print(f"并发请求 {len(urls)} 个 URL，耗时: {elapsed:.2f}秒")
    for r in results:
        print(f"  {r['url']}: {r['status']}")

    # 【限速请求】
    print(f"\n--- 限速请求（使用 Semaphore）---")

    semaphore = asyncio.Semaphore(2)  # 最多2个并发

    async def rate_limited_fetch(url: str) -> dict:
        async with semaphore:
            return await fetch_url(url)

    start = time.perf_counter()
    results = await asyncio.gather(*[rate_limited_fetch(url) for url in urls])
    elapsed = time.perf_counter() - start

    print(f"限速请求耗时: {elapsed:.2f}秒")

    # 【生产者消费者模式】
    print(f"\n--- 生产者消费者模式 ---")

    async def producer(queue: asyncio.Queue, items: List[int]):
        for item in items:
            await queue.put(item)
            print(f"  生产: {item}")
            await asyncio.sleep(0.02)
        await queue.put(None)  # 结束信号

    async def consumer(queue: asyncio.Queue, name: str):
        while True:
            item = await queue.get()
            if item is None:
                await queue.put(None)  # 传递结束信号
                break
            print(f"  {name} 消费: {item}")
            await asyncio.sleep(0.05)

    queue: asyncio.Queue = asyncio.Queue()
    await asyncio.gather(
        producer(queue, list(range(5))),
        consumer(queue, "Consumer-1"),
        consumer(queue, "Consumer-2"),
    )


async def main07_error_handling():
    """
    ============================================================
                7. 错误处理
    ============================================================
    """
    print("\n" + "=" * 60)
    print("7. 错误处理")
    print("=" * 60)

    # 【单个任务异常】
    print("--- 单个任务异常 ---")

    async def risky_task(n):
        if n == 2:
            raise ValueError(f"任务 {n} 出错")
        await asyncio.sleep(0.05)
        return f"任务 {n} 成功"

    try:
        result = await risky_task(2)
    except ValueError as e:
        print(f"捕获异常: {e}")

    # 【gather 中的异常】
    print(f"\n--- gather 中的异常 ---")

    results = await asyncio.gather(
        risky_task(1),
        risky_task(2),
        risky_task(3),
        return_exceptions=True
    )

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  任务 {i+1} 失败: {result}")
        else:
            print(f"  任务 {i+1} 成功: {result}")

    # 【TaskGroup (Python 3.11+)】
    print(f"\n--- TaskGroup (Python 3.11+) ---")

    async def safe_task(n):
        await asyncio.sleep(0.05)
        return f"任务 {n} 完成"

    try:
        async with asyncio.TaskGroup() as tg:
            task1 = tg.create_task(safe_task(1))
            task2 = tg.create_task(safe_task(2))
        print(f"TaskGroup 结果: {task1.result()}, {task2.result()}")
    except* ValueError as eg:
        print(f"TaskGroup 异常组: {eg.exceptions}")


async def main08_patterns():
    """
    ============================================================
                8. 常用模式
    ============================================================
    """
    print("\n" + "=" * 60)
    print("8. 常用模式")
    print("=" * 60)

    # 【批量处理】
    print("--- 批量处理 ---")

    async def process_item(item):
        await asyncio.sleep(0.02)
        return item * 2

    async def batch_process(items, batch_size=3):
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = await asyncio.gather(*[process_item(x) for x in batch])
            results.extend(batch_results)
            print(f"  处理批次 {i // batch_size + 1}: {batch_results}")
        return results

    items = list(range(10))
    results = await batch_process(items, batch_size=3)
    print(f"最终结果: {results}")

    # 【重试模式】
    print(f"\n--- 重试模式 ---")

    attempt_count = 0

    async def unreliable_task():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError(f"尝试 {attempt_count} 失败")
        return "成功!"

    async def retry(coro_func, max_retries=3, delay=0.1):
        for attempt in range(max_retries):
            try:
                return await coro_func()
            except Exception as e:
                print(f"  尝试 {attempt + 1} 失败: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
        raise Exception("重试次数用尽")

    result = await retry(unreliable_task, max_retries=5)
    print(f"最终结果: {result}")

    # 【超时重试】
    print(f"\n--- 超时重试 ---")

    async def slow_or_fast():
        import random
        delay = random.uniform(0.05, 0.2)
        await asyncio.sleep(delay)
        return f"耗时 {delay:.2f}秒"

    async def with_timeout_retry(coro_func, timeout=0.1, max_retries=3):
        for attempt in range(max_retries):
            try:
                return await asyncio.wait_for(coro_func(), timeout=timeout)
            except asyncio.TimeoutError:
                print(f"  尝试 {attempt + 1} 超时")
        raise asyncio.TimeoutError("重试超时")

    try:
        result = await with_timeout_retry(slow_or_fast, timeout=0.15, max_retries=5)
        print(f"结果: {result}")
    except asyncio.TimeoutError:
        print("所有尝试都超时")


async def main():
    """主函数"""
    await main01_coroutine_basics()
    await main02_tasks()
    await main03_wait_and_timeout()
    await main04_semaphore_lock()
    await main05_async_generators()
    await main06_practical_examples()
    await main07_error_handling()
    await main08_patterns()


if __name__ == "__main__":
    asyncio.run(main())
