# async await.cs

::: info 文件信息
- 📄 原文件：`01_async_await.cs`
- 🔤 语言：csharp
:::

## 完整代码

```csharp
// ============================================================
//                      异步编程（async/await）
// ============================================================
// C# 的 async/await 是基于 Task 的异步模型（TAP）
// async 方法在遇到 await 时会释放线程，不阻塞调用线程
// 【核心原则】IO 密集型用 async/await，CPU 密集型用 Task.Run

using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

class AsyncAwaitDemo
{
    // ----------------------------------------------------------
    // 1. 基本 async/await
    // ----------------------------------------------------------
    // async 方法返回 Task（无返回值）或 Task<T>（有返回值）
    // 在最顶层（Main）用 await 等待任务完成

    static async Task<string> FetchDataAsync(string url)
    {
        // 模拟网络请求延迟
        await Task.Delay(100);  // 非阻塞等待
        return $"来自 {url} 的数据";
    }

    static async Task<int> ComputeAsync(int n)
    {
        await Task.Delay(50);  // 模拟计算延迟
        return n * n;
    }

    // ----------------------------------------------------------
    // 2. 异常处理
    // ----------------------------------------------------------
    static async Task<string> FetchWithErrorAsync(bool shouldFail)
    {
        await Task.Delay(50);
        if (shouldFail)
            throw new InvalidOperationException("模拟的网络错误");
        return "成功获取数据";
    }

    // ----------------------------------------------------------
    // 3. 取消令牌（CancellationToken）
    // ----------------------------------------------------------
    // 【最佳实践】长时间运行的异步操作应支持取消

    static async Task<string> LongRunningTaskAsync(CancellationToken cancellationToken)
    {
        Console.WriteLine("长任务开始...");
        for (int i = 0; i < 10; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            await Task.Delay(100, cancellationToken);
            Console.Write($"{i + 1} ");
        }
        Console.WriteLine("\n长任务完成！");
        return "完成";
    }

    // ----------------------------------------------------------
    // 4. 进度报告
    // ----------------------------------------------------------
    static async Task ProcessWithProgressAsync(IProgress<int> progress)
    {
        for (int i = 0; i <= 100; i += 20)
        {
            await Task.Delay(50);
            progress.Report(i);
        }
    }

    // ----------------------------------------------------------
    // 5. 异步流（IAsyncEnumerable，C# 8.0+）
    // ----------------------------------------------------------
    // 【用途】流式处理大量数据，分批返回而不是一次性加载全部

    static async IAsyncEnumerable<int> GenerateNumbersAsync(int count)
    {
        for (int i = 1; i <= count; i++)
        {
            await Task.Delay(50);  // 模拟逐个生成数据
            yield return i;
        }
    }

    static async Task Main()
    {
        Console.WriteLine("=== async/await 基础 ===");

        // 基本异步调用
        string data = await FetchDataAsync("https://api.example.com");
        Console.WriteLine(data);

        int result = await ComputeAsync(7);
        Console.WriteLine($"7² = {result}");

        // ----------------------------------------------------------
        // 并发执行（同时启动多个任务）
        // ----------------------------------------------------------
        Console.WriteLine("\n=== 并发执行 ===");

        // 错误方式（顺序执行，耗时叠加）
        var sw = System.Diagnostics.Stopwatch.StartNew();
        // var r1 = await FetchDataAsync("url1");  // 等待完成再执行下一个
        // var r2 = await FetchDataAsync("url2");

        // 正确方式（并发执行，任务同时开始）
        var task1 = FetchDataAsync("url1");
        var task2 = FetchDataAsync("url2");
        var task3 = ComputeAsync(10);

        await Task.WhenAll(task1, task2, task3);
        sw.Stop();
        Console.WriteLine($"并发完成：{task1.Result}");
        Console.WriteLine($"并发完成：{task2.Result}");
        Console.WriteLine($"计算结果：{task3.Result}");
        Console.WriteLine($"耗时约 {sw.ElapsedMilliseconds}ms（并发，非顺序叠加）");

        // WhenAny：等待任意一个完成
        Console.WriteLine("\n=== WhenAny ===");
        var tasks = new[]
        {
            Task.Delay(300).ContinueWith(_ => "慢任务"),
            Task.Delay(100).ContinueWith(_ => "快任务"),
            Task.Delay(200).ContinueWith(_ => "中等任务"),
        };
        var first = await Task.WhenAny(tasks);
        Console.WriteLine($"最先完成: {await first}");

        // ----------------------------------------------------------
        // 异常处理
        // ----------------------------------------------------------
        Console.WriteLine("\n=== 异常处理 ===");

        try
        {
            string ok = await FetchWithErrorAsync(false);
            Console.WriteLine($"成功: {ok}");

            string fail = await FetchWithErrorAsync(true);
        }
        catch (InvalidOperationException ex)
        {
            Console.WriteLine($"捕获异常: {ex.Message}");
        }

        // 批量任务中的异常处理
        var failingTasks = new[]
        {
            FetchWithErrorAsync(false),
            FetchWithErrorAsync(true),
            FetchWithErrorAsync(false),
        };

        try
        {
            await Task.WhenAll(failingTasks);
        }
        catch (Exception)
        {
            // WhenAll 失败时，所有任务都完成了（包括失败的）
            foreach (var t in failingTasks)
            {
                if (t.IsFaulted)
                    Console.WriteLine($"任务失败: {t.Exception?.InnerException?.Message}");
                else
                    Console.WriteLine($"任务成功: {t.Result}");
            }
        }

        // ----------------------------------------------------------
        // 取消令牌
        // ----------------------------------------------------------
        Console.WriteLine("\n=== 取消令牌 ===");

        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(350));
        try
        {
            await LongRunningTaskAsync(cts.Token);
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine("\n任务被取消！");
        }

        // ----------------------------------------------------------
        // 进度报告
        // ----------------------------------------------------------
        Console.WriteLine("\n=== 进度报告 ===");

        var progress = new Progress<int>(percent =>
            Console.Write($"\r进度: {percent}%   "));
        await ProcessWithProgressAsync(progress);
        Console.WriteLine("\n处理完成！");

        // ----------------------------------------------------------
        // 异步流
        // ----------------------------------------------------------
        Console.WriteLine("\n=== 异步流 ===");

        await foreach (int num in GenerateNumbersAsync(5))
        {
            Console.Write($"{num} ");
        }
        Console.WriteLine("(异步流完成)");

        // ----------------------------------------------------------
        // ValueTask（轻量级异步）
        // ----------------------------------------------------------
        Console.WriteLine("\n=== ValueTask ===");
        // 【用途】经常同步完成的方法，避免 Task 的堆分配开销

        static async ValueTask<int> GetCachedValueAsync(bool useCache)
        {
            if (useCache)
                return 42;  // 同步返回，不分配 Task 对象

            await Task.Delay(50);
            return 100;
        }

        int cached = await GetCachedValueAsync(true);
        int computed = await GetCachedValueAsync(false);
        Console.WriteLine($"缓存值: {cached}, 计算值: {computed}");

        Console.WriteLine("\n=== async/await 演示完成 ===");
    }
}
```
