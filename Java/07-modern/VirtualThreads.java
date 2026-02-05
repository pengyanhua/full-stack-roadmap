import java.time.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * ============================================================
 *                    Java 虚拟线程
 * ============================================================
 * 本文件介绍 Java 21+ 引入的虚拟线程特性。
 * ============================================================
 */
public class VirtualThreads {

    public static void main(String[] args) throws Exception {
        virtualThreadBasics();
        virtualThreadExecutor();
        structuredConcurrency();
        comparison();
    }

    /**
     * ============================================================
     *                    1. 虚拟线程基础
     * ============================================================
     */
    public static void virtualThreadBasics() throws InterruptedException {
        System.out.println("=".repeat(60));
        System.out.println("1. 虚拟线程基础");
        System.out.println("=".repeat(60));

        System.out.println("""
            虚拟线程（Java 21+）特点：
            - 轻量级线程，由 JVM 管理
            - 可以创建百万级虚拟线程
            - 适合 I/O 密集型任务
            - 不适合 CPU 密集型任务
            - 与平台线程（OS 线程）的区别
            """);

        // 【创建虚拟线程】
        System.out.println("--- 创建虚拟线程 ---");

        // 方式1：Thread.startVirtualThread
        Thread vt1 = Thread.startVirtualThread(() -> {
            System.out.println("虚拟线程1: " + Thread.currentThread());
        });
        vt1.join();

        // 方式2：Thread.ofVirtual().start()
        Thread vt2 = Thread.ofVirtual()
            .name("my-virtual-thread")
            .start(() -> {
                System.out.println("虚拟线程2: " + Thread.currentThread().getName());
            });
        vt2.join();

        // 方式3：Thread.Builder
        Thread.Builder builder = Thread.ofVirtual().name("vt-", 1);
        Thread vt3 = builder.start(() -> System.out.println("虚拟线程3"));
        Thread vt4 = builder.start(() -> System.out.println("虚拟线程4"));
        vt3.join();
        vt4.join();

        // 【检查是否虚拟线程】
        System.out.println("\n--- 检查线程类型 ---");
        Thread current = Thread.currentThread();
        Thread virtual = Thread.startVirtualThread(() -> {});
        virtual.join();

        System.out.println("主线程是虚拟线程: " + current.isVirtual());
        System.out.println("创建的是虚拟线程: " + virtual.isVirtual());

        // 【创建大量虚拟线程】
        System.out.println("\n--- 创建大量虚拟线程 ---");
        long start = System.currentTimeMillis();

        List<Thread> threads = new ArrayList<>();
        for (int i = 0; i < 10000; i++) {
            Thread t = Thread.startVirtualThread(() -> {
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
            threads.add(t);
        }

        for (Thread t : threads) {
            t.join();
        }

        long elapsed = System.currentTimeMillis() - start;
        System.out.println("创建并运行 10000 个虚拟线程耗时: " + elapsed + "ms");
    }

    /**
     * ============================================================
     *                    2. 虚拟线程执行器
     * ============================================================
     */
    public static void virtualThreadExecutor() throws Exception {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. 虚拟线程执行器");
        System.out.println("=".repeat(60));

        // 【newVirtualThreadPerTaskExecutor】
        System.out.println("--- newVirtualThreadPerTaskExecutor ---");

        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<String>> futures = new ArrayList<>();

            for (int i = 0; i < 5; i++) {
                int taskId = i;
                Future<String> future = executor.submit(() -> {
                    Thread.sleep(100);
                    return "任务 " + taskId + " 完成，线程: " + Thread.currentThread();
                });
                futures.add(future);
            }

            for (Future<String> future : futures) {
                System.out.println("  " + future.get());
            }
        }

        // 【模拟 I/O 密集型任务】
        System.out.println("\n--- I/O 密集型任务示例 ---");
        simulateHttpRequests();
    }

    private static void simulateHttpRequests() throws Exception {
        int requestCount = 1000;

        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            long start = System.currentTimeMillis();

            List<Future<String>> futures = new ArrayList<>();
            for (int i = 0; i < requestCount; i++) {
                final int id = i;
                futures.add(executor.submit(() -> simulateHttpRequest(id)));
            }

            int successCount = 0;
            for (Future<String> f : futures) {
                try {
                    f.get();
                    successCount++;
                } catch (Exception e) {
                    // ignore
                }
            }

            long elapsed = System.currentTimeMillis() - start;
            System.out.println("完成 " + successCount + "/" + requestCount +
                " 个请求，耗时: " + elapsed + "ms");
        }
    }

    private static String simulateHttpRequest(int id) throws InterruptedException {
        // 模拟网络延迟
        Thread.sleep(50);
        return "Response " + id;
    }

    /**
     * ============================================================
     *                    3. 结构化并发
     * ============================================================
     */
    public static void structuredConcurrency() throws Exception {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("3. 结构化并发");
        System.out.println("=".repeat(60));

        System.out.println("""
            结构化并发（Java 21+ 预览特性）：
            - StructuredTaskScope 管理多个并发任务
            - 子任务的生命周期受父任务控制
            - 更容易处理错误和取消
            - 提供 ShutdownOnSuccess 和 ShutdownOnFailure 策略

            【注意】需要启用预览特性：
            --enable-preview
            """);

        // 【模拟结构化并发概念】
        System.out.println("--- 结构化并发概念 ---");

        // 使用传统方式模拟
        CompletableFuture<String> userFuture = CompletableFuture.supplyAsync(() -> {
            sleep(100);
            return "User: Alice";
        });

        CompletableFuture<String> orderFuture = CompletableFuture.supplyAsync(() -> {
            sleep(150);
            return "Orders: [Order1, Order2]";
        });

        CompletableFuture<String> combined = userFuture.thenCombine(orderFuture,
            (user, orders) -> user + ", " + orders);

        System.out.println("结果: " + combined.get());

        // 【取消和超时】
        System.out.println("\n--- 取消和超时 ---");

        CompletableFuture<String> slowTask = CompletableFuture.supplyAsync(() -> {
            sleep(5000);
            return "完成";
        });

        try {
            String result = slowTask.get(1, TimeUnit.SECONDS);
            System.out.println("结果: " + result);
        } catch (TimeoutException e) {
            slowTask.cancel(true);
            System.out.println("任务超时，已取消");
        }
    }

    /**
     * ============================================================
     *                    4. 虚拟线程 vs 平台线程
     * ============================================================
     */
    public static void comparison() throws Exception {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("4. 虚拟线程 vs 平台线程");
        System.out.println("=".repeat(60));

        System.out.println("""
            【虚拟线程优势】
            - 创建成本极低（几 KB 内存）
            - 可以创建百万级线程
            - 阻塞操作不会阻塞 OS 线程
            - 简化异步编程

            【虚拟线程限制】
            - 不适合 CPU 密集型任务
            - synchronized 块可能导致线程固定
            - 使用 ReentrantLock 替代 synchronized

            【最佳实践】
            - I/O 密集型任务使用虚拟线程
            - CPU 密集型任务使用平台线程池
            - 避免在虚拟线程中使用 ThreadLocal（内存开销大）
            - 使用 try-with-resources 管理执行器
            """);

        // 【比较：创建时间】
        System.out.println("--- 创建时间比较 ---");

        // 虚拟线程
        long vtStart = System.currentTimeMillis();
        List<Thread> virtualThreads = new ArrayList<>();
        for (int i = 0; i < 10000; i++) {
            virtualThreads.add(Thread.ofVirtual().unstarted(() -> {}));
        }
        long vtElapsed = System.currentTimeMillis() - vtStart;

        // 平台线程（数量有限）
        long ptStart = System.currentTimeMillis();
        List<Thread> platformThreads = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {  // 只创建1000个
            platformThreads.add(Thread.ofPlatform().unstarted(() -> {}));
        }
        long ptElapsed = System.currentTimeMillis() - ptStart;

        System.out.println("创建 10000 个虚拟线程: " + vtElapsed + "ms");
        System.out.println("创建 1000 个平台线程: " + ptElapsed + "ms");

        // 【使用建议】
        System.out.println("\n--- 使用场景建议 ---");
        System.out.println("""
            使用虚拟线程：
            - Web 服务器处理请求
            - 数据库查询
            - 网络 I/O
            - 文件 I/O

            使用平台线程：
            - 数学计算
            - 图像处理
            - 加密解密
            - 科学计算

            混合使用：
            - CPU 密集部分用 ForkJoinPool
            - I/O 密集部分用虚拟线程
            """);
    }

    private static void sleep(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
