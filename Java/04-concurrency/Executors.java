import java.util.concurrent.*;
import java.util.*;

/**
 * ============================================================
 *                    Java 线程池与 Executor
 * ============================================================
 * 本文件介绍 Java 中的线程池框架。
 * ============================================================
 */
public class Executors {

    public static void main(String[] args) throws Exception {
        executorBasics();
        threadPoolTypes();
        callableAndFuture();
        completableFutureDemo();
        scheduledExecutor();
    }

    /**
     * ============================================================
     *                    1. Executor 基础
     * ============================================================
     */
    public static void executorBasics() throws InterruptedException {
        System.out.println("=".repeat(60));
        System.out.println("1. Executor 基础");
        System.out.println("=".repeat(60));

        System.out.println("""
            Executor 框架层次：
            Executor          - 执行任务的接口
            └── ExecutorService  - 管理生命周期
                └── ScheduledExecutorService  - 定时任务

            【优点】
            - 线程复用，减少创建销毁开销
            - 控制并发数量
            - 提供任务队列
            - 支持定时和周期任务
            """);

        // 【创建线程池】
        ExecutorService executor = java.util.concurrent.Executors.newFixedThreadPool(3);

        // 【提交任务】
        System.out.println("--- 提交任务 ---");
        for (int i = 1; i <= 5; i++) {
            final int taskId = i;
            executor.submit(() -> {
                System.out.println("任务 " + taskId + " 由 " +
                    Thread.currentThread().getName() + " 执行");
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
        }

        // 【关闭线程池】
        System.out.println("\n--- 关闭线程池 ---");
        executor.shutdown();  // 不再接受新任务，等待已提交任务完成
        boolean terminated = executor.awaitTermination(5, TimeUnit.SECONDS);
        System.out.println("线程池已关闭: " + terminated);

        // 【shutdown vs shutdownNow】
        System.out.println("""

            shutdown()     - 平滑关闭，等待任务完成
            shutdownNow()  - 立即关闭，尝试中断正在执行的任务
            awaitTermination() - 等待关闭完成
            """);
    }

    /**
     * ============================================================
     *                    2. 线程池类型
     * ============================================================
     */
    public static void threadPoolTypes() throws InterruptedException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. 线程池类型");
        System.out.println("=".repeat(60));

        // 【FixedThreadPool】固定大小
        System.out.println("--- FixedThreadPool ---");
        System.out.println("固定数量的线程，任务队列无界");
        ExecutorService fixed = java.util.concurrent.Executors.newFixedThreadPool(2);
        submitTasks(fixed, 3);
        fixed.shutdown();
        fixed.awaitTermination(3, TimeUnit.SECONDS);

        // 【CachedThreadPool】缓存线程池
        System.out.println("\n--- CachedThreadPool ---");
        System.out.println("按需创建线程，空闲 60 秒回收");
        ExecutorService cached = java.util.concurrent.Executors.newCachedThreadPool();
        submitTasks(cached, 3);
        cached.shutdown();
        cached.awaitTermination(3, TimeUnit.SECONDS);

        // 【SingleThreadExecutor】单线程
        System.out.println("\n--- SingleThreadExecutor ---");
        System.out.println("单个工作线程，保证任务顺序执行");
        ExecutorService single = java.util.concurrent.Executors.newSingleThreadExecutor();
        submitTasks(single, 3);
        single.shutdown();
        single.awaitTermination(3, TimeUnit.SECONDS);

        // 【VirtualThreadPerTaskExecutor】虚拟线程（Java 21+）
        System.out.println("\n--- VirtualThreadPerTaskExecutor (Java 21+) ---");
        System.out.println("每个任务一个虚拟线程，适合大量 I/O 密集任务");

        // 【ThreadPoolExecutor 自定义】
        System.out.println("\n--- 自定义 ThreadPoolExecutor ---");
        ThreadPoolExecutor custom = new ThreadPoolExecutor(
            2,                      // 核心线程数
            4,                      // 最大线程数
            60L,                    // 空闲时间
            TimeUnit.SECONDS,       // 时间单位
            new ArrayBlockingQueue<>(10),  // 任务队列
            new ThreadPoolExecutor.CallerRunsPolicy()  // 拒绝策略
        );

        System.out.println("""
            ThreadPoolExecutor 参数：
            - corePoolSize: 核心线程数
            - maximumPoolSize: 最大线程数
            - keepAliveTime: 空闲线程存活时间
            - workQueue: 任务队列
            - threadFactory: 线程工厂
            - handler: 拒绝策略

            拒绝策略：
            - AbortPolicy: 抛出异常（默认）
            - CallerRunsPolicy: 调用者线程执行
            - DiscardPolicy: 静默丢弃
            - DiscardOldestPolicy: 丢弃最旧的任务
            """);

        custom.shutdown();
    }

    private static void submitTasks(ExecutorService executor, int count) {
        for (int i = 1; i <= count; i++) {
            final int id = i;
            executor.submit(() -> {
                System.out.println("  任务 " + id + " -> " +
                    Thread.currentThread().getName());
            });
        }
    }

    /**
     * ============================================================
     *                    3. Callable 和 Future
     * ============================================================
     */
    public static void callableAndFuture() throws Exception {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("3. Callable 和 Future");
        System.out.println("=".repeat(60));

        System.out.println("""
            Runnable vs Callable：
            - Runnable.run() 无返回值，不能抛出检查异常
            - Callable.call() 有返回值，可以抛出异常

            Future：表示异步计算的结果
            """);

        ExecutorService executor = java.util.concurrent.Executors.newFixedThreadPool(2);

        // 【Callable 返回结果】
        System.out.println("--- Callable ---");
        Callable<Integer> task = () -> {
            Thread.sleep(1000);
            return 42;
        };

        Future<Integer> future = executor.submit(task);
        System.out.println("任务已提交，是否完成: " + future.isDone());

        // 阻塞获取结果
        Integer result = future.get();
        System.out.println("结果: " + result);
        System.out.println("是否完成: " + future.isDone());

        // 【带超时的 get】
        System.out.println("\n--- 带超时的 get ---");
        Future<String> slowFuture = executor.submit(() -> {
            Thread.sleep(3000);
            return "完成";
        });

        try {
            String r = slowFuture.get(1, TimeUnit.SECONDS);
            System.out.println("结果: " + r);
        } catch (TimeoutException e) {
            System.out.println("超时！取消任务: " + slowFuture.cancel(true));
        }

        // 【invokeAll】等待所有任务完成
        System.out.println("\n--- invokeAll ---");
        List<Callable<Integer>> tasks = List.of(
            () -> { Thread.sleep(500); return 1; },
            () -> { Thread.sleep(300); return 2; },
            () -> { Thread.sleep(100); return 3; }
        );

        List<Future<Integer>> futures = executor.invokeAll(tasks);
        System.out.print("所有结果: ");
        for (Future<Integer> f : futures) {
            System.out.print(f.get() + " ");
        }
        System.out.println();

        // 【invokeAny】返回第一个完成的
        System.out.println("\n--- invokeAny ---");
        Integer first = executor.invokeAny(tasks);
        System.out.println("最快完成的结果: " + first);

        executor.shutdown();
    }

    /**
     * ============================================================
     *                    4. CompletableFuture
     * ============================================================
     */
    public static void completableFutureDemo() throws Exception {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("4. CompletableFuture");
        System.out.println("=".repeat(60));

        System.out.println("""
            CompletableFuture 优势：
            - 链式调用
            - 组合多个异步操作
            - 异常处理
            - 不阻塞主线程
            """);

        // 【创建 CompletableFuture】
        System.out.println("--- 创建 ---");

        // 异步执行，无返回值
        CompletableFuture<Void> cf1 = CompletableFuture.runAsync(() -> {
            System.out.println("runAsync 执行");
        });

        // 异步执行，有返回值
        CompletableFuture<String> cf2 = CompletableFuture.supplyAsync(() -> {
            return "Hello";
        });

        System.out.println("supplyAsync 结果: " + cf2.get());

        // 【链式操作】
        System.out.println("\n--- 链式操作 ---");

        CompletableFuture<String> chain = CompletableFuture
            .supplyAsync(() -> "Hello")
            .thenApply(s -> s + " World")      // 转换
            .thenApply(String::toUpperCase);   // 再转换

        System.out.println("链式结果: " + chain.get());

        // 【thenAccept / thenRun】
        System.out.println("\n--- thenAccept / thenRun ---");

        CompletableFuture.supplyAsync(() -> "数据")
            .thenAccept(s -> System.out.println("thenAccept: " + s))  // 消费结果
            .thenRun(() -> System.out.println("thenRun: 完成"));      // 无参数

        Thread.sleep(100);

        // 【组合多个 Future】
        System.out.println("\n--- 组合操作 ---");

        CompletableFuture<String> future1 = CompletableFuture.supplyAsync(() -> "Hello");
        CompletableFuture<String> future2 = CompletableFuture.supplyAsync(() -> "World");

        // thenCombine: 两个都完成后合并
        CompletableFuture<String> combined = future1.thenCombine(future2,
            (s1, s2) -> s1 + " " + s2);
        System.out.println("thenCombine: " + combined.get());

        // allOf: 等待所有完成
        CompletableFuture<Void> all = CompletableFuture.allOf(future1, future2);
        all.join();
        System.out.println("allOf 完成");

        // anyOf: 任一完成
        CompletableFuture<Object> any = CompletableFuture.anyOf(
            CompletableFuture.supplyAsync(() -> { sleep(100); return "A"; }),
            CompletableFuture.supplyAsync(() -> { sleep(200); return "B"; })
        );
        System.out.println("anyOf: " + any.get());

        // 【异常处理】
        System.out.println("\n--- 异常处理 ---");

        CompletableFuture<Integer> errorFuture = CompletableFuture
            .supplyAsync(() -> {
                if (true) throw new RuntimeException("模拟错误");
                return 1;
            })
            .exceptionally(ex -> {
                System.out.println("捕获异常: " + ex.getMessage());
                return -1;  // 返回默认值
            });

        System.out.println("异常处理结果: " + errorFuture.get());

        // handle: 同时处理正常和异常
        CompletableFuture<String> handled = CompletableFuture
            .supplyAsync(() -> "OK")
            .handle((result, ex) -> {
                if (ex != null) return "Error: " + ex.getMessage();
                return "Success: " + result;
            });

        System.out.println("handle: " + handled.get());
    }

    private static void sleep(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * ============================================================
     *                    5. ScheduledExecutorService
     * ============================================================
     */
    public static void scheduledExecutor() throws InterruptedException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("5. ScheduledExecutorService");
        System.out.println("=".repeat(60));

        ScheduledExecutorService scheduler =
            java.util.concurrent.Executors.newScheduledThreadPool(2);

        // 【延迟执行】
        System.out.println("--- 延迟执行 ---");
        scheduler.schedule(() -> {
            System.out.println("延迟 1 秒后执行");
        }, 1, TimeUnit.SECONDS);

        // 【固定速率】
        System.out.println("\n--- 固定速率执行 ---");
        ScheduledFuture<?> fixedRate = scheduler.scheduleAtFixedRate(() -> {
            System.out.println("固定速率: " + System.currentTimeMillis() % 10000);
        }, 0, 500, TimeUnit.MILLISECONDS);

        Thread.sleep(2000);
        fixedRate.cancel(false);

        // 【固定延迟】
        System.out.println("\n--- 固定延迟执行 ---");
        System.out.println("""
            scheduleAtFixedRate: 按固定速率，不管任务耗时
            scheduleWithFixedDelay: 上个任务结束后等待固定时间
            """);

        scheduler.shutdown();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
    }
}
