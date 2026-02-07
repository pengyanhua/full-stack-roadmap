# ThreadBasics

::: info 文件信息
- 📄 原文件：`ThreadBasics.java`
- 🔤 语言：java
:::

============================================================
                   Java 线程基础
============================================================
本文件介绍 Java 中的线程创建、生命周期和基本操作。
============================================================

## 完整代码

```java
/**
 * ============================================================
 *                    Java 线程基础
 * ============================================================
 * 本文件介绍 Java 中的线程创建、生命周期和基本操作。
 * ============================================================
 */
public class ThreadBasics {

    public static void main(String[] args) throws InterruptedException {
        threadCreation();
        threadMethods();
        threadStates();
        daemonThreads();
        threadPriority();
    }

    /**
     * ============================================================
     *                    1. 创建线程
     * ============================================================
     */
    public static void threadCreation() throws InterruptedException {
        System.out.println("=".repeat(60));
        System.out.println("1. 创建线程");
        System.out.println("=".repeat(60));

        // 【方式1：继承 Thread 类】
        System.out.println("--- 继承 Thread 类 ---");
        MyThread thread1 = new MyThread("Thread-1");
        thread1.start();  // 启动线程

        // 【方式2：实现 Runnable 接口】推荐
        System.out.println("\n--- 实现 Runnable 接口 ---");
        MyRunnable runnable = new MyRunnable();
        Thread thread2 = new Thread(runnable, "Thread-2");
        thread2.start();

        // 【方式3：Lambda 表达式】最简洁
        System.out.println("\n--- Lambda 表达式 ---");
        Thread thread3 = new Thread(() -> {
            System.out.println(Thread.currentThread().getName() + " 正在运行");
        }, "Thread-3");
        thread3.start();

        // 【方式4：匿名内部类】
        System.out.println("\n--- 匿名内部类 ---");
        Thread thread4 = new Thread() {
            @Override
            public void run() {
                System.out.println(getName() + " 正在运行");
            }
        };
        thread4.setName("Thread-4");
        thread4.start();

        // 等待所有线程完成
        thread1.join();
        thread2.join();
        thread3.join();
        thread4.join();

        System.out.println("\n所有线程执行完毕");
    }

    /**
     * ============================================================
     *                    2. 线程方法
     * ============================================================
     */
    public static void threadMethods() throws InterruptedException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. 线程方法");
        System.out.println("=".repeat(60));

        // 【当前线程】
        System.out.println("--- 当前线程信息 ---");
        Thread current = Thread.currentThread();
        System.out.println("名称: " + current.getName());
        System.out.println("ID: " + current.getId());
        System.out.println("优先级: " + current.getPriority());
        System.out.println("状态: " + current.getState());

        // 【sleep】线程休眠
        System.out.println("\n--- sleep 方法 ---");
        Thread sleeper = new Thread(() -> {
            try {
                System.out.println("开始休眠...");
                Thread.sleep(1000);  // 休眠 1 秒
                System.out.println("休眠结束");
            } catch (InterruptedException e) {
                System.out.println("休眠被中断");
            }
        });
        sleeper.start();
        sleeper.join();

        // 【yield】让出 CPU
        System.out.println("\n--- yield 方法 ---");
        System.out.println("yield() 提示调度器让出 CPU，但不保证效果");
        Thread.yield();

        // 【join】等待线程完成
        System.out.println("\n--- join 方法 ---");
        Thread worker = new Thread(() -> {
            for (int i = 1; i <= 3; i++) {
                System.out.println("Worker 工作中: " + i);
                try {
                    Thread.sleep(300);
                } catch (InterruptedException e) {
                    break;
                }
            }
        });
        worker.start();
        System.out.println("主线程等待 worker 完成...");
        worker.join();  // 阻塞直到 worker 完成
        System.out.println("worker 已完成");

        // 【带超时的 join】
        System.out.println("\n--- 带超时的 join ---");
        Thread longTask = new Thread(() -> {
            try {
                Thread.sleep(5000);
            } catch (InterruptedException e) {
                System.out.println("长任务被中断");
            }
        });
        longTask.start();
        longTask.join(500);  // 最多等待 500 毫秒
        System.out.println("等待超时，longTask 是否还活着: " + longTask.isAlive());
        longTask.interrupt();  // 中断线程

        // 【interrupt】中断线程
        System.out.println("\n--- interrupt 方法 ---");
        Thread interruptable = new Thread(() -> {
            while (!Thread.currentThread().isInterrupted()) {
                System.out.println("线程运行中...");
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    System.out.println("捕获中断异常，退出线程");
                    Thread.currentThread().interrupt();  // 重新设置中断标志
                    break;
                }
            }
            System.out.println("线程已退出");
        });
        interruptable.start();
        Thread.sleep(1200);
        interruptable.interrupt();  // 发送中断信号
        interruptable.join();
    }

    /**
     * ============================================================
     *                    3. 线程状态
     * ============================================================
     */
    public static void threadStates() throws InterruptedException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("3. 线程状态");
        System.out.println("=".repeat(60));

        System.out.println("""
            线程状态（Thread.State）：

            NEW          - 新建，尚未启动
            RUNNABLE     - 可运行，正在 JVM 中执行
            BLOCKED      - 阻塞，等待监视器锁
            WAITING      - 等待，无限期等待另一线程
            TIMED_WAITING - 计时等待，有时限的等待
            TERMINATED   - 终止，已完成执行
            """);

        // 【演示状态转换】
        Object lock = new Object();

        Thread t = new Thread(() -> {
            synchronized (lock) {
                try {
                    lock.wait();  // WAITING
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        });

        System.out.println("创建后: " + t.getState());  // NEW

        t.start();
        Thread.sleep(100);
        System.out.println("启动后: " + t.getState());  // WAITING

        synchronized (lock) {
            lock.notify();
        }
        t.join();
        System.out.println("终止后: " + t.getState());  // TERMINATED
    }

    /**
     * ============================================================
     *                    4. 守护线程
     * ============================================================
     */
    public static void daemonThreads() throws InterruptedException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("4. 守护线程");
        System.out.println("=".repeat(60));

        System.out.println("""
            守护线程（Daemon Thread）特点：
            - 为其他线程提供服务
            - JVM 会在所有非守护线程结束后退出，不等待守护线程
            - 典型例子：垃圾回收器
            """);

        Thread daemon = new Thread(() -> {
            while (true) {
                System.out.println("守护线程运行中...");
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    break;
                }
            }
        });

        daemon.setDaemon(true);  // 设置为守护线程（必须在 start 前设置）
        System.out.println("是否守护线程: " + daemon.isDaemon());

        daemon.start();
        Thread.sleep(1500);
        System.out.println("主线程结束，守护线程将自动终止");
        // 不需要显式停止守护线程
    }

    /**
     * ============================================================
     *                    5. 线程优先级
     * ============================================================
     */
    public static void threadPriority() throws InterruptedException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("5. 线程优先级");
        System.out.println("=".repeat(60));

        System.out.println("""
            线程优先级范围：1-10
            - MIN_PRIORITY = 1
            - NORM_PRIORITY = 5（默认）
            - MAX_PRIORITY = 10

            【注意】优先级只是建议，不保证执行顺序
            """);

        Thread low = new Thread(() -> {
            int count = 0;
            for (int i = 0; i < 1000000; i++) {
                count++;
            }
            System.out.println("低优先级完成: " + count);
        });

        Thread high = new Thread(() -> {
            int count = 0;
            for (int i = 0; i < 1000000; i++) {
                count++;
            }
            System.out.println("高优先级完成: " + count);
        });

        low.setPriority(Thread.MIN_PRIORITY);
        high.setPriority(Thread.MAX_PRIORITY);

        System.out.println("低优先级: " + low.getPriority());
        System.out.println("高优先级: " + high.getPriority());

        low.start();
        high.start();

        low.join();
        high.join();

        System.out.println("\n【警告】不要依赖线程优先级来保证执行顺序");
    }
}

/**
 * 方式1：继承 Thread 类
 */
class MyThread extends Thread {
    public MyThread(String name) {
        super(name);
    }

    @Override
    public void run() {
        System.out.println(getName() + " 正在运行（继承 Thread）");
    }
}

/**
 * 方式2：实现 Runnable 接口
 */
class MyRunnable implements Runnable {
    @Override
    public void run() {
        System.out.println(Thread.currentThread().getName() + " 正在运行（实现 Runnable）");
    }
}
```
