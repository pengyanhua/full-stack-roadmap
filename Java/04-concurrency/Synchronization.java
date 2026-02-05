import java.util.concurrent.locks.*;
import java.util.concurrent.atomic.*;

/**
 * ============================================================
 *                    Java 线程同步
 * ============================================================
 * 本文件介绍 Java 中的线程同步机制。
 * ============================================================
 */
public class Synchronization {

    public static void main(String[] args) throws InterruptedException {
        raceCondition();
        synchronizedDemo();
        lockDemo();
        atomicDemo();
        volatileDemo();
    }

    /**
     * ============================================================
     *                    1. 竞态条件问题
     * ============================================================
     */
    public static void raceCondition() throws InterruptedException {
        System.out.println("=".repeat(60));
        System.out.println("1. 竞态条件问题");
        System.out.println("=".repeat(60));

        // 【不安全的计数器】
        UnsafeCounter unsafeCounter = new UnsafeCounter();

        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 10000; i++) {
                unsafeCounter.increment();
            }
        });

        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 10000; i++) {
                unsafeCounter.increment();
            }
        });

        t1.start();
        t2.start();
        t1.join();
        t2.join();

        System.out.println("不安全计数器结果: " + unsafeCounter.getCount());
        System.out.println("期望值: 20000");
        System.out.println("【问题】由于竞态条件，结果可能小于 20000");
    }

    /**
     * ============================================================
     *                    2. synchronized 关键字
     * ============================================================
     */
    public static void synchronizedDemo() throws InterruptedException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. synchronized 关键字");
        System.out.println("=".repeat(60));

        // 【同步方法】
        System.out.println("--- 同步方法 ---");
        SafeCounter safeCounter = new SafeCounter();

        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 10000; i++) {
                safeCounter.increment();
            }
        });

        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 10000; i++) {
                safeCounter.increment();
            }
        });

        t1.start();
        t2.start();
        t1.join();
        t2.join();

        System.out.println("安全计数器结果: " + safeCounter.getCount());
        System.out.println("期望值: 20000");

        // 【同步块】
        System.out.println("\n--- 同步块 ---");
        BlockCounter blockCounter = new BlockCounter();

        Thread t3 = new Thread(() -> {
            for (int i = 0; i < 10000; i++) {
                blockCounter.increment();
            }
        });

        Thread t4 = new Thread(() -> {
            for (int i = 0; i < 10000; i++) {
                blockCounter.increment();
            }
        });

        t3.start();
        t4.start();
        t3.join();
        t4.join();

        System.out.println("同步块计数器结果: " + blockCounter.getCount());

        // 【静态同步方法】
        System.out.println("\n--- 静态同步方法 ---");
        System.out.println("""
            同步方法锁对象：
            - 实例方法: this
            - 静态方法: 类的 Class 对象

            synchronized(this) { }      // 锁当前实例
            synchronized(MyClass.class) { }  // 锁类对象
            synchronized(lockObject) { }     // 锁指定对象
            """);
    }

    /**
     * ============================================================
     *                    3. Lock 接口
     * ============================================================
     */
    public static void lockDemo() throws InterruptedException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("3. Lock 接口");
        System.out.println("=".repeat(60));

        System.out.println("""
            Lock vs synchronized：
            - Lock 可以尝试获取锁（tryLock）
            - Lock 可以被中断（lockInterruptibly）
            - Lock 可以超时获取（tryLock(timeout)）
            - Lock 需要手动释放（finally 中 unlock）
            - Lock 可以实现公平锁
            """);

        // 【ReentrantLock 基本用法】
        System.out.println("--- ReentrantLock ---");
        LockCounter lockCounter = new LockCounter();

        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 10000; i++) {
                lockCounter.increment();
            }
        });

        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 10000; i++) {
                lockCounter.increment();
            }
        });

        t1.start();
        t2.start();
        t1.join();
        t2.join();

        System.out.println("Lock 计数器结果: " + lockCounter.getCount());

        // 【tryLock】
        System.out.println("\n--- tryLock ---");
        ReentrantLock lock = new ReentrantLock();

        Thread holder = new Thread(() -> {
            lock.lock();
            try {
                System.out.println("持有锁的线程开始工作");
                Thread.sleep(2000);
                System.out.println("持有锁的线程完成");
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } finally {
                lock.unlock();
            }
        });

        Thread tryLocker = new Thread(() -> {
            System.out.println("尝试获取锁...");
            if (lock.tryLock()) {
                try {
                    System.out.println("成功获取锁");
                } finally {
                    lock.unlock();
                }
            } else {
                System.out.println("获取锁失败，执行其他操作");
            }
        });

        holder.start();
        Thread.sleep(100);  // 确保 holder 先获取锁
        tryLocker.start();

        holder.join();
        tryLocker.join();

        // 【ReadWriteLock】
        System.out.println("\n--- ReadWriteLock ---");
        System.out.println("""
            读写锁特点：
            - 读锁可以被多个读线程同时持有
            - 写锁是独占的
            - 读写互斥
            - 适合读多写少的场景
            """);
    }

    /**
     * ============================================================
     *                    4. 原子类
     * ============================================================
     */
    public static void atomicDemo() throws InterruptedException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("4. 原子类");
        System.out.println("=".repeat(60));

        System.out.println("""
            常用原子类：
            - AtomicInteger, AtomicLong, AtomicBoolean
            - AtomicReference<T>
            - AtomicIntegerArray, AtomicLongArray
            - LongAdder（高并发场景更高效）
            """);

        // 【AtomicInteger】
        System.out.println("--- AtomicInteger ---");
        AtomicInteger atomicInt = new AtomicInteger(0);

        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 10000; i++) {
                atomicInt.incrementAndGet();
            }
        });

        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 10000; i++) {
                atomicInt.incrementAndGet();
            }
        });

        t1.start();
        t2.start();
        t1.join();
        t2.join();

        System.out.println("AtomicInteger 结果: " + atomicInt.get());

        // 【AtomicInteger 方法】
        System.out.println("\n--- AtomicInteger 方法 ---");
        AtomicInteger ai = new AtomicInteger(10);
        System.out.println("初始值: " + ai.get());
        System.out.println("getAndIncrement(): " + ai.getAndIncrement());  // 返回旧值
        System.out.println("incrementAndGet(): " + ai.incrementAndGet());  // 返回新值
        System.out.println("getAndAdd(5): " + ai.getAndAdd(5));
        System.out.println("addAndGet(5): " + ai.addAndGet(5));
        System.out.println("当前值: " + ai.get());

        // CAS 操作
        System.out.println("\n--- CAS 操作 ---");
        AtomicInteger cas = new AtomicInteger(100);
        boolean success = cas.compareAndSet(100, 200);  // 期望100，设置200
        System.out.println("CAS 成功: " + success + ", 值: " + cas.get());

        success = cas.compareAndSet(100, 300);  // 期望100（但当前是200）
        System.out.println("CAS 成功: " + success + ", 值: " + cas.get());

        // 【AtomicReference】
        System.out.println("\n--- AtomicReference ---");
        AtomicReference<String> ref = new AtomicReference<>("Hello");
        ref.set("World");
        System.out.println("AtomicReference: " + ref.get());

        // 【LongAdder】高并发场景
        System.out.println("\n--- LongAdder（高并发推荐）---");
        var adder = new java.util.concurrent.atomic.LongAdder();
        for (int i = 0; i < 100; i++) {
            adder.increment();
        }
        System.out.println("LongAdder 结果: " + adder.sum());
    }

    /**
     * ============================================================
     *                    5. volatile 关键字
     * ============================================================
     */
    public static void volatileDemo() throws InterruptedException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("5. volatile 关键字");
        System.out.println("=".repeat(60));

        System.out.println("""
            volatile 特性：
            1. 可见性：一个线程修改后，其他线程立即可见
            2. 禁止指令重排序

            【注意】volatile 不保证原子性！
            - count++ 不是原子操作
            - 需要原子性请使用 synchronized 或 Atomic 类

            适用场景：
            - 状态标志位
            - 单次写入的配置项
            - 双重检查锁定（DCL）单例
            """);

        // 【volatile 可见性演示】
        System.out.println("--- volatile 可见性 ---");
        VolatileFlag flag = new VolatileFlag();

        Thread writer = new Thread(() -> {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            flag.stop();
            System.out.println("Writer 设置 running = false");
        });

        Thread reader = new Thread(() -> {
            int count = 0;
            while (flag.isRunning()) {
                count++;
            }
            System.out.println("Reader 检测到停止，循环次数: " + count);
        });

        reader.start();
        writer.start();

        reader.join();
        writer.join();

        // 【双重检查锁定单例】
        System.out.println("\n--- DCL 单例模式 ---");
        System.out.println("""
            public class Singleton {
                private static volatile Singleton instance;

                private Singleton() {}

                public static Singleton getInstance() {
                    if (instance == null) {
                        synchronized (Singleton.class) {
                            if (instance == null) {
                                instance = new Singleton();
                            }
                        }
                    }
                    return instance;
                }
            }

            volatile 防止指令重排序导致的问题
            """);
    }
}

/**
 * 不安全的计数器
 */
class UnsafeCounter {
    private int count = 0;

    public void increment() {
        count++;  // 非原子操作：读取、增加、写入
    }

    public int getCount() {
        return count;
    }
}

/**
 * 使用 synchronized 方法的安全计数器
 */
class SafeCounter {
    private int count = 0;

    public synchronized void increment() {
        count++;
    }

    public synchronized int getCount() {
        return count;
    }
}

/**
 * 使用 synchronized 块的计数器
 */
class BlockCounter {
    private int count = 0;
    private final Object lock = new Object();

    public void increment() {
        synchronized (lock) {
            count++;
        }
    }

    public int getCount() {
        synchronized (lock) {
            return count;
        }
    }
}

/**
 * 使用 Lock 的计数器
 */
class LockCounter {
    private int count = 0;
    private final Lock lock = new ReentrantLock();

    public void increment() {
        lock.lock();
        try {
            count++;
        } finally {
            lock.unlock();  // 必须在 finally 中释放
        }
    }

    public int getCount() {
        lock.lock();
        try {
            return count;
        } finally {
            lock.unlock();
        }
    }
}

/**
 * volatile 标志类
 */
class VolatileFlag {
    private volatile boolean running = true;

    public boolean isRunning() {
        return running;
    }

    public void stop() {
        running = false;
    }
}
