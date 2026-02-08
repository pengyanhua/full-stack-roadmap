# 并发控制与同步机制

## 课程概述

本教程深入讲解操作系统的并发控制机制,从临界区问题到各种同步原语,从死锁处理到无锁编程,帮助你全面掌握并发编程的核心技术和最佳实践。

**学习目标**:
- 理解临界区与竞态条件问题
- 掌握互斥锁、信号量、条件变量等同步原语
- 深入了解死锁的产生、检测与预防
- 学习无锁编程与原子操作
- 掌握RCU(Read-Copy-Update)机制
- 理解内存模型与内存屏障

---

## 1. 并发基础

### 1.1 临界区问题

```
┌─────────────────────────────────────────────────────────────┐
│              临界区(Critical Section)问题                     │
└─────────────────────────────────────────────────────────────┘

问题场景: 多线程访问共享资源

进程A                     共享变量              进程B
┌────────┐               counter = 0           ┌────────┐
│        │                                     │        │
│ 读取   │ ─────────▶    读: 0                │ 读取   │
│counter │               │                     │counter │
│ = 0    │               │                     │ = 0    │
│        │               │                     │        │
│ +1     │               │                     │ +1     │
│ = 1    │               │                     │ = 1    │
│        │               │                     │        │
│ 写回   │ ─────────▶    写: 1    ◀─────────  │ 写回   │
│counter │                                     │counter │
│        │                                     │        │
└────────┘               期望: 2               └────────┘
                        实际: 1  ❌  竞态条件!

竞态条件(Race Condition):
多个执行流对共享资源的访问顺序会影响最终结果

临界区的三个要求:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  1. 互斥访问(Mutual Exclusion)                           │
│     ┌───────────────────────────────────────────────┐   │
│     │ 同一时刻只有一个进程在临界区                    │   │
│     └───────────────────────────────────────────────┘   │
│                                                         │
│  2. 空闲让进(Progress)                                   │
│     ┌───────────────────────────────────────────────┐   │
│     │ 如果没有进程在临界区,想进入的进程应该能进入      │   │
│     │ (不能无限期等待)                                │   │
│     └───────────────────────────────────────────────┘   │
│                                                         │
│  3. 有限等待(Bounded Waiting)                            │
│     ┌───────────────────────────────────────────────┐   │
│     │ 进程请求进入临界区后,等待时间应该有限            │   │
│     │ (防止饥饿)                                      │   │
│     └───────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

经典同步问题:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│ 1. 生产者-消费者问题(Producer-Consumer)                      │
│    ┌────────┐    ┌──────────┐    ┌────────┐               │
│    │生产者  │───▶│ 有界缓冲区│───▶│消费者  │               │
│    │        │    │ (n个槽位) │    │        │               │
│    └────────┘    └──────────┘    └────────┘               │
│    挑战: 缓冲区满时生产者等待,空时消费者等待                 │
│                                                             │
│ 2. 读者-写者问题(Readers-Writers)                            │
│    ┌────────┐                                               │
│    │ 读者1  │──┐                                            │
│    └────────┘  │                                            │
│    ┌────────┐  │    ┌──────────┐                           │
│    │ 读者2  │──┼───▶│ 共享数据 │                           │
│    └────────┘  │    └──────────┘                           │
│    ┌────────┐  │         ▲                                 │
│    │ 写者   │──┘         │ (互斥)                           │
│    └────────┘            │                                 │
│    挑战: 多读者可同时读,写者独占,读者优先或写者优先?         │
│                                                             │
│ 3. 哲学家就餐问题(Dining Philosophers)                       │
│                 ┌────────┐                                  │
│                 │哲学家1 │                                  │
│            ┌────┴────┬───┴────┐                            │
│       叉子1│         │        │叉子2                        │
│    ┌───────┴─┐     桌子     ┌─┴───────┐                    │
│    │哲学家5  │             │哲学家2  │                    │
│    └─────┬───┘             └───┬─────┘                    │
│      叉子5│                     │叉子3                      │
│    ┌─────┴───┐             ┌───┴─────┐                    │
│    │哲学家4  │             │哲学家3  │                    │
│    └─────────┘             └─────────┘                    │
│          └───────叉子4───────┘                             │
│    挑战: 每个哲学家需要两把叉子,如何避免死锁和饥饿?           │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Peterson算法(软件解决方案)

```c
/*
 * Peterson算法 - 两进程互斥的软件解决方案
 */
#include <stdio.h>
#include <pthread.h>
#include <stdbool.h>

#define NUM_ITERATIONS 1000000

/* 共享变量 */
volatile bool flag[2] = {false, false};
volatile int turn = 0;
volatile int counter = 0;

/* Peterson进入临界区 */
void peterson_enter(int id)
{
    int other = 1 - id;

    flag[id] = true;     /* 表明自己想进入临界区 */
    turn = other;        /* 礼让:让对方先进 */

    /* 自旋等待 */
    while (flag[other] && turn == other) {
        /* 等待:对方想进且轮到对方 */
    }
}

/* Peterson离开临界区 */
void peterson_exit(int id)
{
    flag[id] = false;    /* 表明自己离开临界区 */
}

/* 线程函数 */
void *thread_func(void *arg)
{
    int id = *(int *)arg;

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        /* 进入临界区 */
        peterson_enter(id);

        /* 临界区代码 */
        counter++;

        /* 离开临界区 */
        peterson_exit(id);
    }

    return NULL;
}

int main(void)
{
    pthread_t t1, t2;
    int id1 = 0, id2 = 1;

    printf("Peterson Algorithm Demo\n");
    printf("Expected counter: %d\n", 2 * NUM_ITERATIONS);

    pthread_create(&t1, NULL, thread_func, &id1);
    pthread_create(&t2, NULL, thread_func, &id2);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Actual counter:   %d\n", counter);

    if (counter == 2 * NUM_ITERATIONS) {
        printf("✓ Mutual exclusion works!\n");
    } else {
        printf("✗ Race condition detected!\n");
    }

    return 0;
}
```

---

## 2. 互斥锁(Mutex)

### 2.1 互斥锁原理

```
┌─────────────────────────────────────────────────────────────┐
│              互斥锁(Mutex)工作原理                            │
└─────────────────────────────────────────────────────────────┘

互斥锁状态:
┌──────────────┐          ┌──────────────┐
│              │  lock()  │              │
│    解锁状态   │ ───────▶ │    加锁状态   │
│  (Unlocked)  │          │   (Locked)   │
│              │ ◀─────── │              │
│              │ unlock() │              │
└──────────────┘          └──────────────┘
   任何线程可获取              持有者独占

线程竞争互斥锁:
         线程A          线程B          线程C
           │               │               │
           │ lock()        │               │
           ├───────────────┤               │
           │ ✓获得锁       │               │
           │               │ lock()        │
           │               ├───────────────┤
           │               │ ✗阻塞等待     │
           │               │               │ lock()
           │               │               ├─────────────
           │               │               │ ✗阻塞等待
  临界区   │               │               │
  代码执行 │               │               │
           │               │               │
           │ unlock()      │               │
           ├───────────────┤               │
           │               │ ✓被唤醒       │
           │               │ ✓获得锁       │
           │               │               │
           │               │  临界区       │
           │               │  代码执行     │
           │               │               │
           │               │ unlock()      │
           │               ├───────────────┤
           │               │               │ ✓被唤醒
           │               │               │ ✓获得锁

自旋锁 vs 睡眠锁:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│ 自旋锁(Spinlock)                                            │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ while (test_and_set(&lock)) {                        │    │
│ │     /* 忙等待,持续检查锁 */                           │    │
│ │ }                                                    │    │
│ │                                                      │    │
│ │ 优点: 无上下文切换开销                                │    │
│ │ 缺点: 浪费CPU,适合短临界区                            │    │
│ │ 适用: 内核中断处理,多核系统                           │    │
│ └─────────────────────────────────────────────────────┘    │
│                                                             │
│ 睡眠锁(Sleeping Mutex)                                      │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ if (lock is held) {                                  │    │
│ │     /* 进入睡眠,让出CPU */                            │    │
│ │     sleep_and_wait();                                │    │
│ │     /* 被唤醒后继续 */                                │    │
│ │ }                                                    │    │
│ │                                                      │    │
│ │ 优点: 不浪费CPU                                       │    │
│ │ 缺点: 有上下文切换开销                                │    │
│ │ 适用: 用户空间,长临界区                               │    │
│ └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

Futex (Fast Userspace Mutex):
┌─────────────────────────────────────────────────────────────┐
│  用户空间                    内核空间                        │
│  ┌────────────────┐        ┌────────────────┐              │
│  │                │        │                │              │
│  │ 1. 原子操作    │        │                │              │
│  │    尝试加锁    │        │                │              │
│  │    (CAS)       │        │                │              │
│  │                │        │                │              │
│  │ 2. 成功? ──────┼────Yes─▶ 直接返回        │              │
│  │                │   (快速路径)            │              │
│  │    失败? ──────┼────No───▶               │              │
│  │                │        │ 3. 系统调用    │              │
│  │                │        │    futex()     │              │
│  │                │        │                │              │
│  │                │        │ 4. 加入等待队列 │              │
│  │                │        │                │              │
│  │ 5. 睡眠 ◀──────┼────────│ 5. 挂起线程    │              │
│  │                │        │                │              │
│  │                │        │ 6. 被唤醒      │              │
│  │ 7. 醒来 ◀──────┼────────│    (unlock时)  │              │
│  │                │        │                │              │
│  └────────────────┘        └────────────────┘              │
│                                                             │
│  优点: 无竞争时在用户空间完成,有竞争时才进入内核              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 互斥锁实战

```c
/*
 * pthread互斥锁示例
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define NUM_THREADS 10
#define NUM_INCREMENTS 100000

typedef struct {
    int counter;
    pthread_mutex_t mutex;
} shared_data_t;

shared_data_t shared_data = {
    .counter = 0,
    .mutex = PTHREAD_MUTEX_INITIALIZER
};

/* 不使用互斥锁 - 错误示范 */
void *unsafe_increment(void *arg)
{
    for (int i = 0; i < NUM_INCREMENTS; i++) {
        shared_data.counter++;  /* 竞态条件! */
    }
    return NULL;
}

/* 使用互斥锁 - 正确做法 */
void *safe_increment(void *arg)
{
    for (int i = 0; i < NUM_INCREMENTS; i++) {
        pthread_mutex_lock(&shared_data.mutex);
        shared_data.counter++;
        pthread_mutex_unlock(&shared_data.mutex);
    }
    return NULL;
}

/* 使用互斥锁 - 批量操作优化 */
void *optimized_increment(void *arg)
{
    int local_counter = 0;

    /* 先在本地累加 */
    for (int i = 0; i < NUM_INCREMENTS; i++) {
        local_counter++;
    }

    /* 一次性更新共享变量 */
    pthread_mutex_lock(&shared_data.mutex);
    shared_data.counter += local_counter;
    pthread_mutex_unlock(&shared_data.mutex);

    return NULL;
}

void test_concurrent_access(void *(*func)(void *), const char *name)
{
    pthread_t threads[NUM_THREADS];
    struct timespec start, end;

    shared_data.counter = 0;

    clock_gettime(CLOCK_MONOTONIC, &start);

    /* 创建线程 */
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, func, NULL);
    }

    /* 等待线程结束 */
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                    (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("%-20s Counter: %10d (Expected: %10d) Time: %.3fs\n",
           name, shared_data.counter,
           NUM_THREADS * NUM_INCREMENTS, elapsed);
}

int main(void)
{
    printf("Testing concurrent counter with %d threads, "
           "%d increments each\n\n", NUM_THREADS, NUM_INCREMENTS);

    test_concurrent_access(unsafe_increment, "Unsafe:");
    test_concurrent_access(safe_increment, "Safe (mutex):");
    test_concurrent_access(optimized_increment, "Optimized:");

    pthread_mutex_destroy(&shared_data.mutex);

    return 0;
}
```

```c
/*
 * 死锁示例与解决方案
 */
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

pthread_mutex_t lock1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t lock2 = PTHREAD_MUTEX_INITIALIZER;

/* 死锁示例 */
void *thread1_deadlock(void *arg)
{
    printf("Thread 1: Locking lock1\n");
    pthread_mutex_lock(&lock1);

    sleep(1);  /* 模拟一些工作 */

    printf("Thread 1: Waiting for lock2\n");
    pthread_mutex_lock(&lock2);  /* 死锁! */

    printf("Thread 1: Got both locks\n");

    pthread_mutex_unlock(&lock2);
    pthread_mutex_unlock(&lock1);

    return NULL;
}

void *thread2_deadlock(void *arg)
{
    printf("Thread 2: Locking lock2\n");
    pthread_mutex_lock(&lock2);

    sleep(1);  /* 模拟一些工作 */

    printf("Thread 2: Waiting for lock1\n");
    pthread_mutex_lock(&lock1);  /* 死锁! */

    printf("Thread 2: Got both locks\n");

    pthread_mutex_unlock(&lock1);
    pthread_mutex_unlock(&lock2);

    return NULL;
}

/* 解决方案1: 统一加锁顺序 */
void *thread1_ordered(void *arg)
{
    printf("Thread 1: Locking lock1 then lock2\n");
    pthread_mutex_lock(&lock1);
    pthread_mutex_lock(&lock2);

    printf("Thread 1: Got both locks\n");
    sleep(1);

    pthread_mutex_unlock(&lock2);
    pthread_mutex_unlock(&lock1);

    return NULL;
}

void *thread2_ordered(void *arg)
{
    printf("Thread 2: Locking lock1 then lock2\n");
    pthread_mutex_lock(&lock1);  /* 同样的顺序 */
    pthread_mutex_lock(&lock2);

    printf("Thread 2: Got both locks\n");
    sleep(1);

    pthread_mutex_unlock(&lock2);
    pthread_mutex_unlock(&lock1);

    return NULL;
}

/* 解决方案2: 尝试加锁(trylock) */
void *thread_trylock(void *arg)
{
    int id = *(int *)arg;

    while (1) {
        pthread_mutex_lock(&lock1);

        if (pthread_mutex_trylock(&lock2) == 0) {
            /* 成功获得两个锁 */
            printf("Thread %d: Got both locks\n", id);
            sleep(1);

            pthread_mutex_unlock(&lock2);
            pthread_mutex_unlock(&lock1);
            break;
        } else {
            /* 获取lock2失败,释放lock1并重试 */
            pthread_mutex_unlock(&lock1);
            printf("Thread %d: Failed to get lock2, retry\n", id);
            usleep(rand() % 1000);  /* 随机退避 */
        }
    }

    return NULL;
}

int main(void)
{
    pthread_t t1, t2;
    int id1 = 1, id2 = 2;

    printf("=== Demo 1: Deadlock ===\n");
    printf("(Will hang forever, press Ctrl+C to stop)\n\n");

    // pthread_create(&t1, NULL, thread1_deadlock, NULL);
    // pthread_create(&t2, NULL, thread2_deadlock, NULL);

    printf("=== Demo 2: Fixed with lock ordering ===\n");
    pthread_create(&t1, NULL, thread1_ordered, NULL);
    pthread_create(&t2, NULL, thread2_ordered, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("\n=== Demo 3: Fixed with trylock ===\n");
    pthread_create(&t1, NULL, thread_trylock, &id1);
    pthread_create(&t2, NULL, thread_trylock, &id2);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    return 0;
}
```

---

## 3. 信号量(Semaphore)

### 3.1 信号量原理

```
┌─────────────────────────────────────────────────────────────┐
│              信号量(Semaphore)原理                            │
└─────────────────────────────────────────────────────────────┘

信号量 = 整数计数器 + 等待队列

┌────────────────────────────────────────────────┐
│ struct semaphore {                              │
│     int value;        /* 计数器 */              │
│     queue wait_queue; /* 等待队列 */            │
│ };                                              │
└────────────────────────────────────────────────┘

两个原子操作:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│ P操作 (wait / down / acquire)                               │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ P(S):                                                │    │
│ │     S.value--;                                       │    │
│ │     if (S.value < 0) {                               │    │
│ │         /* 资源不足,进入等待队列 */                   │    │
│ │         add_to_wait_queue(S.wait_queue);             │    │
│ │         block();  /* 阻塞当前进程 */                  │    │
│ │     }                                                │    │
│ └─────────────────────────────────────────────────────┘    │
│                                                             │
│ V操作 (signal / up / release)                               │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ V(S):                                                │    │
│ │     S.value++;                                       │    │
│ │     if (S.value <= 0) {                              │    │
│ │         /* 有进程在等待,唤醒一个 */                   │    │
│ │         remove_from_wait_queue(S.wait_queue);        │    │
│ │         wakeup(process);                             │    │
│ │     }                                                │    │
│ └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

信号量类型:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│ 1. 二值信号量(Binary Semaphore)                             │
│    初始值: 0或1                                             │
│    用途: 互斥锁的替代                                       │
│                                                             │
│    sem_t mutex;                                             │
│    sem_init(&mutex, 0, 1);  /* 初始值=1 */                  │
│                                                             │
│    sem_wait(&mutex);   /* P操作,获取锁 */                   │
│    /* 临界区 */                                             │
│    sem_post(&mutex);   /* V操作,释放锁 */                   │
│                                                             │
│ 2. 计数信号量(Counting Semaphore)                           │
│    初始值: N (资源数量)                                     │
│    用途: 资源池管理,限流                                    │
│                                                             │
│    sem_t resources;                                         │
│    sem_init(&resources, 0, 5);  /* 5个资源 */               │
│                                                             │
│    sem_wait(&resources);  /* 获取一个资源 */                │
│    /* 使用资源 */                                           │
│    sem_post(&resources);  /* 归还资源 */                    │
└─────────────────────────────────────────────────────────────┘

生产者-消费者问题:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  缓冲区: [slot0][slot1][slot2]...[slotN-1]                 │
│                                                             │
│  信号量:                                                    │
│  • empty: N    (空槽位数量,初始=N)                          │
│  • full:  0    (满槽位数量,初始=0)                          │
│  • mutex: 1    (互斥访问缓冲区)                             │
│                                                             │
│  生产者:                    消费者:                         │
│  ┌───────────────────┐    ┌───────────────────┐           │
│  │ while (1) {       │    │ while (1) {       │           │
│  │   produce_item(); │    │   P(full);        │           │
│  │                   │    │   P(mutex);       │           │
│  │   P(empty);       │    │   item=remove();  │           │
│  │   P(mutex);       │    │   V(mutex);       │           │
│  │   insert(item);   │    │   V(empty);       │           │
│  │   V(mutex);       │    │                   │           │
│  │   V(full);        │    │   consume(item);  │           │
│  │ }                 │    │ }                 │           │
│  └───────────────────┘    └───────────────────┘           │
│                                                             │
│  关键点:                                                    │
│  • empty/full控制同步                                       │
│  • mutex控制互斥                                            │
│  • 顺序很重要: 先P(empty/full),再P(mutex)                   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 信号量实战

```c
/*
 * 生产者-消费者问题(有界缓冲区)
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define BUFFER_SIZE 5
#define NUM_PRODUCERS 2
#define NUM_CONSUMERS 3
#define NUM_ITEMS 20

/* 环形缓冲区 */
typedef struct {
    int buffer[BUFFER_SIZE];
    int in;   /* 生产者写入位置 */
    int out;  /* 消费者读取位置 */
    sem_t empty;  /* 空槽位信号量 */
    sem_t full;   /* 满槽位信号量 */
    sem_t mutex;  /* 互斥信号量 */
} bounded_buffer_t;

bounded_buffer_t bb;

/* 初始化缓冲区 */
void buffer_init(void)
{
    bb.in = 0;
    bb.out = 0;
    sem_init(&bb.empty, 0, BUFFER_SIZE);  /* 初始全空 */
    sem_init(&bb.full, 0, 0);              /* 初始无数据 */
    sem_init(&bb.mutex, 0, 1);             /* 互斥锁 */
}

/* 生产者插入数据 */
void buffer_insert(int item)
{
    bb.buffer[bb.in] = item;
    bb.in = (bb.in + 1) % BUFFER_SIZE;
}

/* 消费者取出数据 */
int buffer_remove(void)
{
    int item = bb.buffer[bb.out];
    bb.out = (bb.out + 1) % BUFFER_SIZE;
    return item;
}

/* 生产者线程 */
void *producer(void *arg)
{
    int id = *(int *)arg;

    for (int i = 0; i < NUM_ITEMS / NUM_PRODUCERS; i++) {
        int item = id * 100 + i;

        /* 生产数据 */
        usleep(rand() % 100000);  /* 模拟生产时间 */

        /* 等待空槽位 */
        sem_wait(&bb.empty);

        /* 互斥访问缓冲区 */
        sem_wait(&bb.mutex);

        buffer_insert(item);
        printf("Producer %d: produced %d\n", id, item);

        sem_post(&bb.mutex);

        /* 增加满槽位计数 */
        sem_post(&bb.full);
    }

    printf("Producer %d: finished\n", id);
    return NULL;
}

/* 消费者线程 */
void *consumer(void *arg)
{
    int id = *(int *)arg;

    for (int i = 0; i < NUM_ITEMS / NUM_CONSUMERS; i++) {
        /* 等待满槽位 */
        sem_wait(&bb.full);

        /* 互斥访问缓冲区 */
        sem_wait(&bb.mutex);

        int item = buffer_remove();
        printf("Consumer %d: consumed %d\n", id, item);

        sem_post(&bb.mutex);

        /* 增加空槽位计数 */
        sem_post(&bb.empty);

        /* 消费数据 */
        usleep(rand() % 150000);  /* 模拟消费时间 */
    }

    printf("Consumer %d: finished\n", id);
    return NULL;
}

int main(void)
{
    pthread_t producers[NUM_PRODUCERS];
    pthread_t consumers[NUM_CONSUMERS];
    int producer_ids[NUM_PRODUCERS];
    int consumer_ids[NUM_CONSUMERS];

    srand(time(NULL));
    buffer_init();

    printf("Producer-Consumer with bounded buffer (size=%d)\n\n",
           BUFFER_SIZE);

    /* 创建生产者 */
    for (int i = 0; i < NUM_PRODUCERS; i++) {
        producer_ids[i] = i;
        pthread_create(&producers[i], NULL, producer, &producer_ids[i]);
    }

    /* 创建消费者 */
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        consumer_ids[i] = i;
        pthread_create(&consumers[i], NULL, consumer, &consumer_ids[i]);
    }

    /* 等待所有线程完成 */
    for (int i = 0; i < NUM_PRODUCERS; i++) {
        pthread_join(producers[i], NULL);
    }
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        pthread_join(consumers[i], NULL);
    }

    /* 清理 */
    sem_destroy(&bb.empty);
    sem_destroy(&bb.full);
    sem_destroy(&bb.mutex);

    printf("\nAll done!\n");

    return 0;
}
```

---

## 4. 读写锁(Reader-Writer Lock)

### 4.1 读写锁原理

```
┌─────────────────────────────────────────────────────────────┐
│              读写锁(RWLock)原理                               │
└─────────────────────────────────────────────────────────────┘

锁状态:
┌────────────────────────────────────────────────────────┐
│                                                        │
│  未加锁 ───────┬───────┐                               │
│               │       │                               │
│            读锁│       │写锁                           │
│               │       │                               │
│               ▼       ▼                               │
│         ┌─────────┐ ┌─────────┐                       │
│         │ 共享读   │ │ 独占写   │                       │
│         │ (多个)  │ │ (单个)  │                       │
│         └─────────┘ └─────────┘                       │
│               │       │                               │
│               └───────┘                               │
│                  解锁                                  │
└────────────────────────────────────────────────────────┘

访问模式:
┌─────────────────────────────────────────────────────────────┐
│ 当前状态      │  读请求   │  写请求                          │
├──────────────┼──────────┼──────────────────────────────────┤
│ 未加锁        │  ✓ 允许  │  ✓ 允许                          │
│ 读锁(N个读者) │  ✓ 允许  │  ✗ 阻塞(等待所有读者完成)        │
│ 写锁(1个写者) │  ✗ 阻塞  │  ✗ 阻塞                          │
└─────────────────────────────────────────────────────────────┘

读者优先 vs 写者优先:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│ 读者优先(Reader-Preference)                                 │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ • 只要有读者持有锁,新到的读者可以直接获取            │    │
│ │ • 写者必须等待所有读者完成                           │    │
│ │ • 问题: 连续的读者会导致写者饥饿                     │    │
│ │                                                     │    │
│ │ 场景:                                                │    │
│ │ R1获取读锁 → R2获取读锁 → W1等待 → R3获取读锁       │    │
│ │ → R1释放 → R4获取读锁 → R2释放 → ...                │    │
│ │ W1一直等待! (写者饥饿)                               │    │
│ └─────────────────────────────────────────────────────┘    │
│                                                             │
│ 写者优先(Writer-Preference)                                 │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ • 有写者等待时,新到的读者必须等待                    │    │
│ │ • 写者优先获得锁                                     │    │
│ │ • 问题: 连续的写者会导致读者饥饿                     │    │
│ │                                                     │    │
│ │ 场景:                                                │    │
│ │ R1获取读锁 → W1等待 → R2等待(因为W1在等待)          │    │
│ │ → R1释放 → W1获取写锁 → W2等待 → W1释放 → W2获取... │    │
│ │ R2一直等待! (读者饥饿)                               │    │
│ └─────────────────────────────────────────────────────┘    │
│                                                             │
│ 公平锁(Fair RWLock)                                         │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ • 按照请求顺序排队,FIFO                              │    │
│ │ • 避免饥饿                                           │    │
│ │ • 可能吞吐量略低                                     │    │
│ └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

实现原理(简化版):
┌────────────────────────────────────────────────┐
│ struct rwlock {                                 │
│     int readers;      /* 当前读者数 */          │
│     int writers;      /* 当前写者数(0或1) */    │
│     int waiting_writers; /* 等待的写者数 */     │
│     mutex_t lock;     /* 保护上述变量 */        │
│     cond_t read_ok;   /* 读者条件变量 */        │
│     cond_t write_ok;  /* 写者条件变量 */        │
│ };                                              │
└────────────────────────────────────────────────┘
```

### 4.2 读写锁实战

```c
/*
 * pthread读写锁示例
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define NUM_READERS 5
#define NUM_WRITERS 2

typedef struct {
    int data;
    pthread_rwlock_t rwlock;
} shared_resource_t;

shared_resource_t resource = {
    .data = 0,
    .rwlock = PTHREAD_RWLOCK_INITIALIZER
};

/* 读者线程 */
void *reader_thread(void *arg)
{
    int id = *(int *)arg;

    for (int i = 0; i < 3; i++) {
        /* 获取读锁 */
        pthread_rwlock_rdlock(&resource.rwlock);

        printf("[Reader %d] Reading data: %d\n", id, resource.data);
        usleep(100000);  /* 模拟读取时间 */

        pthread_rwlock_unlock(&resource.rwlock);

        usleep(rand() % 200000);
    }

    printf("[Reader %d] Finished\n", id);
    return NULL;
}

/* 写者线程 */
void *writer_thread(void *arg)
{
    int id = *(int *)arg;

    for (int i = 0; i < 3; i++) {
        /* 获取写锁 */
        pthread_rwlock_wrlock(&resource.rwlock);

        resource.data++;
        printf("[Writer %d] Writing data: %d\n", id, resource.data);
        usleep(200000);  /* 模拟写入时间 */

        pthread_rwlock_unlock(&resource.rwlock);

        usleep(rand() % 300000);
    }

    printf("[Writer %d] Finished\n", id);
    return NULL;
}

int main(void)
{
    pthread_t readers[NUM_READERS];
    pthread_t writers[NUM_WRITERS];
    int reader_ids[NUM_READERS];
    int writer_ids[NUM_WRITERS];

    srand(time(NULL));

    printf("Reader-Writer Lock Demo\n\n");

    /* 创建读者线程 */
    for (int i = 0; i < NUM_READERS; i++) {
        reader_ids[i] = i;
        pthread_create(&readers[i], NULL, reader_thread, &reader_ids[i]);
    }

    /* 创建写者线程 */
    for (int i = 0; i < NUM_WRITERS; i++) {
        writer_ids[i] = i;
        pthread_create(&writers[i], NULL, writer_thread, &writer_ids[i]);
    }

    /* 等待所有线程 */
    for (int i = 0; i < NUM_READERS; i++) {
        pthread_join(readers[i], NULL);
    }
    for (int i = 0; i < NUM_WRITERS; i++) {
        pthread_join(writers[i], NULL);
    }

    pthread_rwlock_destroy(&resource.rwlock);

    printf("\nAll done! Final data: %d\n", resource.data);

    return 0;
}
```

---

## 5. 死锁

### 5.1 死锁四个必要条件

```
┌─────────────────────────────────────────────────────────────┐
│              死锁的四个必要条件(Coffman条件)                  │
└─────────────────────────────────────────────────────────────┘

1. 互斥(Mutual Exclusion)
   ┌─────────────────────────────────────────────────────┐
   │ 资源同一时刻只能被一个进程占用                       │
   │                                                     │
   │  进程A ◀─── 资源R1 (独占)                           │
   │  进程B      ✗ 无法同时访问                          │
   └─────────────────────────────────────────────────────┘

2. 占有并等待(Hold and Wait)
   ┌─────────────────────────────────────────────────────┐
   │ 进程持有至少一个资源,同时等待获取其他资源            │
   │                                                     │
   │  进程A: 持有R1,等待R2                               │
   │  进程B: 持有R2,等待R1                               │
   └─────────────────────────────────────────────────────┘

3. 不可剥夺(No Preemption)
   ┌─────────────────────────────────────────────────────┐
   │ 资源不能被强制从进程中抢占,只能主动释放              │
   │                                                     │
   │  进程A持有R1 → 系统不能强制收回                     │
   └─────────────────────────────────────────────────────┘

4. 循环等待(Circular Wait)
   ┌─────────────────────────────────────────────────────┐
   │ 存在进程-资源的循环等待链                            │
   │                                                     │
   │  P1 ───等待─▶ R1                                    │
   │   ▲           │                                     │
   │   │          持有                                   │
   │   │           ▼                                     │
   │  持有         P2 ───等待─▶ R2                       │
   │   │           ▲           │                        │
   │   │          持有         持有                      │
   │   │           │           ▼                        │
   │   └───等待──  R3  ◀───持有── P3                     │
   │                                                     │
   │  形成环路!                                           │
   └─────────────────────────────────────────────────────┘

资源分配图:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  无死锁情况:                  死锁情况:                      │
│                                                             │
│  P1 ──▶ R1                   P1 ──▶ R2                     │
│  │                            │      ▲                     │
│  │                            │      │                     │
│  ▼                            ▼      │                     │
│  R2 ──▶ P2                   R1 ──▶ P2                     │
│                                                             │
│  (无环,无死锁)                (有环,死锁!)                   │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 死锁处理策略

```
┌─────────────────────────────────────────────────────────────┐
│              死锁处理的四种策略                               │
└─────────────────────────────────────────────────────────────┘

1. 死锁预防(Deadlock Prevention)
   破坏四个必要条件之一
   ┌─────────────────────────────────────────────────────┐
   │ 破坏互斥: 不可行(有些资源必须互斥)                   │
   │                                                     │
   │ 破坏占有并等待:                                      │
   │ • 一次性申请所有资源                                 │
   │ • 申请新资源前释放已有资源                           │
   │   缺点: 资源利用率低,可能饥饿                        │
   │                                                     │
   │ 破坏不可剥夺:                                        │
   │ • 允许抢占资源                                       │
   │   缺点: 不适用于打印机等资源                         │
   │                                                     │
   │ 破坏循环等待:                                        │
   │ • 资源排序,按序申请                                  │
   │   例: Lock1 < Lock2,总是先申请Lock1                 │
   │   缺点: 灵活性差                                     │
   └─────────────────────────────────────────────────────┘

2. 死锁避免(Deadlock Avoidance)
   动态检查资源分配状态,确保系统处于安全状态
   ┌─────────────────────────────────────────────────────┐
   │ 银行家算法(Banker's Algorithm)                       │
   │                                                     │
   │ 系统资源: [10, 5, 7]                                │
   │                                                     │
   │        已分配  最大需求  还需要  可用                │
   │ P0    [0,1,0]  [7,5,3]  [7,4,3]  [3,3,2]           │
   │ P1    [2,0,0]  [3,2,2]  [1,2,2]                    │
   │ P2    [3,0,2]  [9,0,2]  [6,0,0]                    │
   │ P3    [2,1,1]  [2,2,2]  [0,1,1]                    │
   │ P4    [0,0,2]  [4,3,3]  [4,3,1]                    │
   │                                                     │
   │ 安全序列检查: <P1, P3, P4, P2, P0>                  │
   │                                                     │
   │ 优点: 可避免死锁                                     │
   │ 缺点: 需要预知最大需求,开销大                        │
   └─────────────────────────────────────────────────────┘

3. 死锁检测与恢复(Detection & Recovery)
   允许死锁发生,定期检测并恢复
   ┌─────────────────────────────────────────────────────┐
   │ 检测算法:                                            │
   │ • 维护资源分配图                                     │
   │ • 周期性检测环路                                     │
   │                                                     │
   │ 恢复策略:                                            │
   │ 1. 进程终止                                          │
   │    • 终止所有死锁进程(简单但代价大)                  │
   │    • 逐个终止直到解除死锁(需要回滚)                  │
   │                                                     │
   │ 2. 资源抢占                                          │
   │    • 选择牺牲品进程                                  │
   │    • 回滚到安全状态                                  │
   │    • 避免饥饿(不总是选同一进程)                      │
   │                                                     │
   │ 适用: 死锁罕见的系统                                 │
   └─────────────────────────────────────────────────────┘

4. 鸵鸟策略(Ostrich Algorithm)
   忽略死锁问题
   ┌─────────────────────────────────────────────────────┐
   │ • 假装死锁不存在                                     │
   │ • 依赖用户重启系统                                   │
   │ • 成本效益权衡                                       │
   │                                                     │
   │ 适用: Unix/Linux(死锁极少发生)                      │
   │ 理由: 死锁处理成本 > 死锁造成的损失                  │
   └─────────────────────────────────────────────────────┘
```

---

## 6. 无锁编程

### 6.1 原子操作

```c
/*
 * 原子操作与CAS示例
 */
#include <stdio.h>
#include <stdatomic.h>
#include <pthread.h>

#define NUM_THREADS 10
#define NUM_INCREMENTS 100000

/* 使用原子变量 */
atomic_int atomic_counter = 0;

/* 普通变量(对比用) */
int normal_counter = 0;

/* 使用原子操作 */
void *atomic_increment(void *arg)
{
    for (int i = 0; i < NUM_INCREMENTS; i++) {
        atomic_fetch_add(&atomic_counter, 1);
        /* 等价于: atomic_counter++ (原子的) */
    }
    return NULL;
}

/* 不使用原子操作 */
void *normal_increment(void *arg)
{
    for (int i = 0; i < NUM_INCREMENTS; i++) {
        normal_counter++;  /* 非原子! */
    }
    return NULL;
}

/* CAS (Compare-And-Swap) 示例 */
void *cas_increment(void *arg)
{
    for (int i = 0; i < NUM_INCREMENTS; i++) {
        int expected, desired;

        do {
            expected = atomic_load(&atomic_counter);
            desired = expected + 1;
        } while (!atomic_compare_exchange_weak(&atomic_counter,
                                              &expected, desired));
        /*
         * CAS伪代码:
         * if (atomic_counter == expected) {
         *     atomic_counter = desired;
         *     return true;
         * } else {
         *     expected = atomic_counter;  // 更新expected
         *     return false;
         * }
         */
    }
    return NULL;
}

int main(void)
{
    pthread_t threads[NUM_THREADS];

    printf("=== Test 1: Normal counter (race condition) ===\n");
    normal_counter = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, normal_increment, NULL);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    printf("Expected: %d, Actual: %d\n",
           NUM_THREADS * NUM_INCREMENTS, normal_counter);

    printf("\n=== Test 2: Atomic counter ===\n");
    atomic_counter = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, atomic_increment, NULL);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    printf("Expected: %d, Actual: %d\n",
           NUM_THREADS * NUM_INCREMENTS, atomic_counter);

    printf("\n=== Test 3: CAS ===\n");
    atomic_counter = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, cas_increment, NULL);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    printf("Expected: %d, Actual: %d\n",
           NUM_THREADS * NUM_INCREMENTS, atomic_counter);

    return 0;
}
```

```
┌─────────────────────────────────────────────────────────────┐
│              原子操作与内存序(Memory Order)                   │
└─────────────────────────────────────────────────────────────┘

C11原子操作内存序:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│ 1. memory_order_relaxed (最弱)                              │
│    • 无同步保证                                             │
│    • 只保证原子性                                           │
│    • 性能最高                                               │
│    atomic_fetch_add_explicit(&counter, 1,                   │
│                             memory_order_relaxed);          │
│                                                             │
│ 2. memory_order_acquire (读操作)                            │
│    • 后续读写不会被重排到此操作之前                          │
│    • 用于获取锁                                             │
│                                                             │
│ 3. memory_order_release (写操作)                            │
│    • 之前的读写不会被重排到此操作之后                        │
│    • 用于释放锁                                             │
│                                                             │
│ 4. memory_order_acq_rel                                     │
│    • acquire + release                                      │
│    • 用于read-modify-write操作                              │
│                                                             │
│ 5. memory_order_seq_cst (最强,默认)                         │
│    • 全局顺序一致性                                         │
│    • 所有线程看到相同的操作顺序                              │
│    • 性能最低                                               │
└─────────────────────────────────────────────────────────────┘

内存屏障(Memory Barrier):
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  无屏障(可能被重排):          有屏障(顺序保证):              │
│                                                             │
│  CPU看到的顺序:               强制顺序:                      │
│  ┌──────────┐                ┌──────────┐                  │
│  │ 写 X = 1 │                │ 写 X = 1 │                  │
│  ├──────────┤  可能重排       ├──────────┤                  │
│  │ 写 Y = 1 │   ───▶         │ BARRIER  │                  │
│  ├──────────┤                ├──────────┤                  │
│  │ 读 Y     │                │ 写 Y = 1 │                  │
│  ├──────────┤                ├──────────┤                  │
│  │ 读 X     │                │ 读 Y     │                  │
│  └──────────┘                ├──────────┤                  │
│                              │ 读 X     │                  │
│                              └──────────┘                  │
│                                                             │
│  Linux内核屏障:                                              │
│  • smp_mb()   : 完全内存屏障                                │
│  • smp_rmb()  : 读屏障                                      │
│  • smp_wmb()  : 写屏障                                      │
│  • barrier()  : 编译器屏障(禁止重排,无硬件屏障)             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 RCU(Read-Copy-Update)

```
┌─────────────────────────────────────────────────────────────┐
│              RCU(Read-Copy-Update)机制                        │
└─────────────────────────────────────────────────────────────┘

RCU的核心思想:
• 读者无锁,零开销
• 写者复制-修改-发布
• 延迟释放旧数据

RCU更新过程:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│ 初始状态:                                                    │
│ ┌────────┐                                                  │
│ │ 全局指针│──▶ [old data: version 1]                        │
│ └────────┘                                                  │
│     │                                                       │
│     │ 读者1,读者2,读者3 (直接访问,无锁)                      │
│                                                             │
│ 更新步骤1: 复制(Copy)                                        │
│ ┌────────┐                                                  │
│ │ 全局指针│──▶ [old data: version 1]                        │
│ └────────┘          ▲                                       │
│                     │ 读者仍在访问                           │
│                     │                                       │
│          [new data: version 2] (写者创建副本并修改)          │
│                                                             │
│ 更新步骤2: 发布(Update)                                      │
│ ┌────────┐                                                  │
│ │ 全局指针│──┐                                              │
│ └────────┘  │                                              │
│             │                                              │
│             ├─▶ [old data: version 1]                      │
│             │       ▲ 旧读者仍在访问                        │
│             │                                              │
│             └─▶ [new data: version 2]                      │
│                     ▲ 新读者访问新版本                      │
│                                                             │
│ 更新步骤3: 等待宽限期(Grace Period)                          │
│ • 等待所有旧读者完成(不阻塞,被动等待)                        │
│ • 新读者都访问新数据                                         │
│                                                             │
│ 更新步骤4: 回收(Reclaim)                                     │
│ ┌────────┐                                                  │
│ │ 全局指针│─────────▶ [new data: version 2]                 │
│ └────────┘                                                  │
│                                                             │
│ [old data: version 1] (已释放)                              │
└─────────────────────────────────────────────────────────────┘

RCU API:
┌─────────────────────────────────────────────────────────────┐
│ 读者侧:                                                      │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ rcu_read_lock();      /* 标记读临界区开始 */         │    │
│ │                                                     │    │
│ │ ptr = rcu_dereference(global_ptr);  /* 读取指针 */  │    │
│ │ /* 使用ptr访问数据 */                                │    │
│ │                                                     │    │
│ │ rcu_read_unlock();    /* 标记读临界区结束 */         │    │
│ └─────────────────────────────────────────────────────┘    │
│                                                             │
│ 写者侧:                                                      │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ new = kmalloc(...);                  /* 分配新结构 */ │    │
│ │ *new = *old;                         /* 复制 */      │    │
│ │ new->field = new_value;              /* 修改 */      │    │
│ │                                                     │    │
│ │ rcu_assign_pointer(global_ptr, new); /* 发布 */     │    │
│ │                                                     │    │
│ │ synchronize_rcu();                   /* 等待宽限期 */ │    │
│ │ kfree(old);                          /* 释放旧数据 */ │    │
│ └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

RCU优缺点:
┌─────────────────────────────────────────────────────────────┐
│ 优点:                                                        │
│ • 读者完全无锁,零开销                                        │
│ • 读者不会阻塞写者                                           │
│ • 可扩展性好(读密集场景)                                     │
│                                                             │
│ 缺点:                                                        │
│ • 写者开销大(复制+等待)                                      │
│ • 内存占用增加(存在多版本)                                   │
│ • 不适合写密集场景                                           │
│                                                             │
│ 适用场景:                                                    │
│ • 读多写少                                                   │
│ • 数据结构遍历(链表、树)                                     │
│ • 路由表、配置信息                                           │
│ • Linux内核网络、VFS                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. 并发编程最佳实践

```bash
#!/bin/bash
# 并发问题检测工具

echo "=== 1. ThreadSanitizer (TSan) ==="
echo "检测数据竞争"
cat > race.c << 'EOF'
#include <pthread.h>
int g = 0;
void *thread_func(void *arg) {
    g++;  // 竞态条件
    return NULL;
}
int main() {
    pthread_t t1, t2;
    pthread_create(&t1, NULL, thread_func, NULL);
    pthread_create(&t2, NULL, thread_func, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    return 0;
}
EOF
gcc -fsanitize=thread -g race.c -o race -lpthread
./race

echo -e "\n=== 2. Helgrind (Valgrind工具) ==="
echo "检测锁顺序问题和数据竞争"
valgrind --tool=helgrind ./race

echo -e "\n=== 3. 查看死锁 ==="
# 检测进程死锁
cat /proc/*/status | grep State

# 使用gdb检测死锁
# gdb -p <pid>
# (gdb) thread apply all bt

echo -e "\n=== 4. 性能分析 ==="
perf record -g ./your_program
perf report

echo "=== 5. 锁竞争统计 (Linux) ==="
cat /proc/lock_stat
```

```python
#!/usr/bin/env python3
"""
并发编程最佳实践示例
"""
import threading
import queue
import time
from contextlib import contextmanager

# 最佳实践1: 使用上下文管理器自动释放锁
class SafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    @contextmanager
    def _acquire(self):
        """上下文管理器,确保锁被释放"""
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()

    def increment(self):
        with self._acquire():
            self._value += 1

    def get(self):
        with self._acquire():
            return self._value

# 最佳实践2: 避免嵌套锁
class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance
        self.lock = threading.Lock()

    def transfer_to(self, target, amount):
        """错误: 嵌套锁可能死锁"""
        # with self.lock:
        #     with target.lock:  # 危险!
        #         self.balance -= amount
        #         target.balance += amount

        """正确: 使用锁排序"""
        first, second = (self, target) if id(self) < id(target) else (target, self)
        with first.lock:
            with second.lock:
                self.balance -= amount
                target.balance += amount

# 最佳实践3: 使用Queue避免手动同步
def producer_consumer_pattern():
    task_queue = queue.Queue(maxsize=10)

    def producer():
        for i in range(20):
            task_queue.put(i)  # 自动阻塞
            print(f"Produced: {i}")

    def consumer():
        while True:
            item = task_queue.get()  # 自动阻塞
            if item is None:
                break
            print(f"Consumed: {item}")
            task_queue.task_done()

    # 启动线程
    threads = []
    threads.append(threading.Thread(target=producer))
    threads.append(threading.Thread(target=consumer))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

# 最佳实践4: 使用线程局部存储
thread_local = threading.local()

def thread_specific_data():
    # 每个线程有自己的副本
    thread_local.value = threading.current_thread().name
    time.sleep(0.1)
    print(f"Thread {thread_local.value}")

# 最佳实践5: 使用concurrent.futures简化并发
from concurrent.futures import ThreadPoolExecutor, as_completed

def parallel_processing():
    def task(n):
        return n * n

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(task, i) for i in range(10)]
        for future in as_completed(futures):
            print(f"Result: {future.result()}")

if __name__ == '__main__':
    print("Concurrent programming best practices\n")
    producer_consumer_pattern()
```

---

## 8. 总结

本教程深入讲解了操作系统的并发控制机制:

**核心知识点**:
1. 临界区问题: 竞态条件、Peterson算法
2. 互斥锁: 自旋锁、睡眠锁、Futex
3. 信号量: 生产者-消费者、读者-写者
4. 读写锁: 读者优先、写者优先、公平锁
5. 死锁: 四个必要条件、预防/避免/检测
6. 无锁编程: 原子操作、CAS、RCU

**实战技能**:
- 正确使用各种同步原语
- 识别和解决死锁问题
- 应用无锁编程技术
- 并发程序调试与性能优化

**最佳实践**:
1. 减小临界区范围
2. 避免嵌套锁
3. 统一锁顺序
4. 读多写少用RCU
5. 使用高级抽象(Queue等)
6. 工具辅助(TSan, Helgrind)

掌握并发控制是编写高性能多线程程序的基础!
