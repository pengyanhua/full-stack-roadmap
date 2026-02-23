// ============================================================
//                      Task 与线程
// ============================================================
// Thread：OS 线程，精细控制但开销大
// Task：线程池任务，推荐用于 CPU 密集型并行计算
// Parallel：数据并行，处理集合的高效方式
// lock / Monitor / SemaphoreSlim：线程同步机制
// Channel<T>：高性能的生产者-消费者通道（C# 8.0+）

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

class TasksAndThreads
{
    // ----------------------------------------------------------
    // 共享状态（演示线程安全问题）
    // ----------------------------------------------------------
    private static int _counter = 0;
    private static readonly object _lock = new();

    // ----------------------------------------------------------
    // 1. Thread 基础（低级，不推荐直接使用）
    // ----------------------------------------------------------
    static void ThreadBasics()
    {
        Console.WriteLine("\n=== Thread 基础 ===");

        // 创建线程
        var t1 = new Thread(() =>
        {
            for (int i = 0; i < 3; i++)
            {
                Console.WriteLine($"  线程1: {i} (ID={Thread.CurrentThread.ManagedThreadId})");
                Thread.Sleep(50);
            }
        });

        var t2 = new Thread(name =>
        {
            for (int i = 0; i < 3; i++)
            {
                Console.WriteLine($"  {name}: {i}");
                Thread.Sleep(50);
            }
        });

        t1.Name = "Worker-1";
        t1.IsBackground = true;  // 后台线程不阻止程序退出

        t1.Start();
        t2.Start("Worker-2");

        t1.Join();  // 等待 t1 完成
        t2.Join();

        Console.WriteLine($"主线程 ID: {Thread.CurrentThread.ManagedThreadId}");
    }

    // ----------------------------------------------------------
    // 2. Task 与线程池（推荐）
    // ----------------------------------------------------------
    static async Task TaskDemo()
    {
        Console.WriteLine("\n=== Task 并发 ===");

        // Task.Run：在线程池上运行 CPU 密集型任务
        int result = await Task.Run(() =>
        {
            // 模拟 CPU 密集计算
            int sum = 0;
            for (int i = 0; i < 1_000_000; i++) sum += i;
            return sum;
        });
        Console.WriteLine($"CPU 计算结果: {result}");

        // 并行执行多个 CPU 任务
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var tasks = Enumerable.Range(1, 4).Select(i => Task.Run(() =>
        {
            Thread.Sleep(100);  // 模拟 CPU 工作
            return i * i;
        })).ToArray();

        int[] results = await Task.WhenAll(tasks);
        sw.Stop();
        Console.WriteLine($"并行结果: [{string.Join(", ", results)}]，耗时: {sw.ElapsedMilliseconds}ms");
    }

    // ----------------------------------------------------------
    // 3. 线程安全（Race Condition 与解决方案）
    // ----------------------------------------------------------
    static void ThreadSafetyDemo()
    {
        Console.WriteLine("\n=== 线程安全 ===");

        // 问题：不安全的计数器（Race Condition）
        int unsafeCounter = 0;
        var t1 = new Thread(() => { for (int i = 0; i < 10000; i++) unsafeCounter++; });
        var t2 = new Thread(() => { for (int i = 0; i < 10000; i++) unsafeCounter++; });
        t1.Start(); t2.Start();
        t1.Join(); t2.Join();
        Console.WriteLine($"不安全计数器（期望 20000）: {unsafeCounter}");

        // 解决方案1：Interlocked（原子操作，性能最好）
        int atomicCounter = 0;
        t1 = new Thread(() => { for (int i = 0; i < 10000; i++) Interlocked.Increment(ref atomicCounter); });
        t2 = new Thread(() => { for (int i = 0; i < 10000; i++) Interlocked.Increment(ref atomicCounter); });
        t1.Start(); t2.Start();
        t1.Join(); t2.Join();
        Console.WriteLine($"Interlocked 计数器: {atomicCounter}");

        // 解决方案2：lock（互斥锁）
        int lockedCounter = 0;
        var lockObj = new object();
        t1 = new Thread(() => { for (int i = 0; i < 10000; i++) { lock (lockObj) lockedCounter++; } });
        t2 = new Thread(() => { for (int i = 0; i < 10000; i++) { lock (lockObj) lockedCounter++; } });
        t1.Start(); t2.Start();
        t1.Join(); t2.Join();
        Console.WriteLine($"lock 计数器: {lockedCounter}");

        // 解决方案3：ConcurrentDictionary（线程安全集合）
        var dict = new ConcurrentDictionary<string, int>();
        var tasks = Enumerable.Range(0, 10).Select(i => Task.Run(() =>
        {
            dict.AddOrUpdate("key", 1, (k, v) => v + 1);
        })).ToArray();
        Task.WhenAll(tasks).Wait();
        Console.WriteLine($"ConcurrentDictionary 值: {dict["key"]}（期望 10）");
    }

    // ----------------------------------------------------------
    // 4. Parallel 类（数据并行）
    // ----------------------------------------------------------
    static void ParallelDemo()
    {
        Console.WriteLine("\n=== Parallel 并行 ===");

        // Parallel.For — 并行 for 循环
        var results = new int[10];
        Parallel.For(0, 10, i =>
        {
            results[i] = i * i;
            Thread.Sleep(10);  // 模拟工作
        });
        Console.WriteLine($"Parallel.For: [{string.Join(", ", results)}]");

        // Parallel.ForEach — 并行处理集合
        var names = new[] { "张三", "李四", "王五", "赵六", "钱七" };
        var processed = new ConcurrentBag<string>();
        Parallel.ForEach(names, name =>
        {
            processed.Add(name.ToUpper() + "！");
        });
        Console.WriteLine($"Parallel.ForEach: [{string.Join(", ", processed.OrderBy(s => s))}]");

        // 控制并行度
        var options = new ParallelOptions { MaxDegreeOfParallelism = 2 };
        Parallel.For(0, 6, options, i =>
        {
            Console.Write($"[{i}T{Thread.CurrentThread.ManagedThreadId}] ");
            Thread.Sleep(20);
        });
        Console.WriteLine("(最大2线程并行)");

        // PLINQ（Parallel LINQ）
        var numbers = Enumerable.Range(1, 20).ToList();
        var evenSquares = numbers
            .AsParallel()               // 启用并行
            .WithDegreeOfParallelism(4) // 设置并行度
            .Where(n => n % 2 == 0)
            .Select(n => n * n)
            .OrderBy(n => n)
            .ToList();
        Console.WriteLine($"\nPLINQ: [{string.Join(", ", evenSquares)}]");
    }

    // ----------------------------------------------------------
    // 5. SemaphoreSlim（限制并发数量）
    // ----------------------------------------------------------
    static async Task SemaphoreDemo()
    {
        Console.WriteLine("\n=== SemaphoreSlim（限流） ===");

        // 最多同时运行 3 个任务
        using var semaphore = new SemaphoreSlim(3, 3);

        async Task WorkAsync(int id)
        {
            await semaphore.WaitAsync();
            try
            {
                Console.WriteLine($"  任务 {id} 开始（{DateTime.Now:HH:mm:ss.fff}）");
                await Task.Delay(150);
                Console.WriteLine($"  任务 {id} 完成");
            }
            finally
            {
                semaphore.Release();
            }
        }

        var tasks = Enumerable.Range(1, 6).Select(WorkAsync).ToArray();
        await Task.WhenAll(tasks);
    }

    // ----------------------------------------------------------
    // 6. Channel<T>（高性能生产者-消费者）
    // ----------------------------------------------------------
    static async Task ChannelDemo()
    {
        Console.WriteLine("\n=== Channel 生产者-消费者 ===");

        var channel = Channel.CreateBounded<int>(new BoundedChannelOptions(5)
        {
            FullMode = BoundedChannelFullMode.Wait
        });

        // 生产者
        async Task ProduceAsync()
        {
            for (int i = 1; i <= 10; i++)
            {
                await channel.Writer.WriteAsync(i);
                Console.Write($"生产:{i} ");
                await Task.Delay(30);
            }
            channel.Writer.Complete();
        }

        // 消费者
        async Task ConsumeAsync()
        {
            await foreach (int item in channel.Reader.ReadAllAsync())
            {
                Console.Write($"消费:{item} ");
                await Task.Delay(60);
            }
        }

        await Task.WhenAll(ProduceAsync(), ConsumeAsync());
        Console.WriteLine("\nChannel 完成！");
    }

    static async Task Main()
    {
        ThreadBasics();
        await TaskDemo();
        ThreadSafetyDemo();
        ParallelDemo();
        await SemaphoreDemo();
        await ChannelDemo();

        Console.WriteLine("\n=== 并发编程演示完成 ===");
    }
}
