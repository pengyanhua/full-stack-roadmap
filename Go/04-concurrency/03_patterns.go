package main

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ============================================================
//                      常见并发模式
// ============================================================
// Go 的并发模式基于 goroutine 和 channel
// 这些模式可以组合使用来解决复杂的并发问题

func main03() {
	fmt.Println("\n==================== 03_patterns ====================")

	// ----------------------------------------------------------
	// 模式1: Worker Pool（工作池）
	// ----------------------------------------------------------
	fmt.Println("=== Worker Pool ===")

	jobs := make(chan int, 10)
	results := make(chan int, 10)

	// 启动 3 个 worker
	for w := 1; w <= 3; w++ {
		go worker(w, jobs, results)
	}

	// 发送 5 个任务
	for j := 1; j <= 5; j++ {
		jobs <- j
	}
	close(jobs)

	// 收集结果
	for r := 1; r <= 5; r++ {
		result := <-results
		fmt.Println("结果:", result)
	}

	// ----------------------------------------------------------
	// 模式2: Fan-out/Fan-in（扇出/扇入）
	// ----------------------------------------------------------
	fmt.Println("\n=== Fan-out/Fan-in ===")

	input := generateNumbers(1, 5)

	// Fan-out: 多个 goroutine 从同一个 channel 读取
	c1 := squareWorker(input)
	c2 := squareWorker(input)

	// Fan-in: 合并多个 channel 到一个
	merged := fanIn(c1, c2)

	// 收集结果
	for i := 0; i < 5; i++ {
		fmt.Println("平方结果:", <-merged)
	}

	// ----------------------------------------------------------
	// 模式3: Pipeline（管道）
	// ----------------------------------------------------------
	fmt.Println("\n=== Pipeline ===")

	// 创建管道: 生成 -> 翻倍 -> 加10
	nums := generate(1, 2, 3, 4, 5)
	doubled := double(nums)
	final := addTen(doubled)

	for n := range final {
		fmt.Println("管道结果:", n)
	}

	// ----------------------------------------------------------
	// 模式4: Done Channel（退出信号）
	// ----------------------------------------------------------
	fmt.Println("\n=== Done Channel ===")

	done := make(chan struct{})

	go func() {
		for {
			select {
			case <-done:
				fmt.Println("收到退出信号，goroutine 退出")
				return
			default:
				fmt.Println("工作中...")
				time.Sleep(50 * time.Millisecond)
			}
		}
	}()

	time.Sleep(120 * time.Millisecond)
	close(done) // 发送退出信号
	time.Sleep(50 * time.Millisecond)

	// ----------------------------------------------------------
	// 模式5: Context（上下文控制）
	// ----------------------------------------------------------
	fmt.Println("\n=== Context ===")

	// 带超时的 context
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()

	go longRunningTask(ctx)

	time.Sleep(200 * time.Millisecond)

	// 手动取消
	ctx2, cancel2 := context.WithCancel(context.Background())

	go func() {
		for {
			select {
			case <-ctx2.Done():
				fmt.Println("Context 被取消:", ctx2.Err())
				return
			default:
				time.Sleep(30 * time.Millisecond)
			}
		}
	}()

	time.Sleep(80 * time.Millisecond)
	cancel2() // 手动取消
	time.Sleep(50 * time.Millisecond)

	// ----------------------------------------------------------
	// 模式6: Semaphore（信号量，限制并发数）
	// ----------------------------------------------------------
	fmt.Println("\n=== Semaphore ===")

	// 使用带缓冲的 channel 作为信号量
	sem := make(chan struct{}, 2) // 最多 2 个并发

	var wg sync.WaitGroup
	for i := 1; i <= 5; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			sem <- struct{}{}        // 获取信号量
			defer func() { <-sem }() // 释放信号量

			fmt.Printf("任务 %d 开始（并发限制）\n", id)
			time.Sleep(50 * time.Millisecond)
			fmt.Printf("任务 %d 完成\n", id)
		}(i)
	}
	wg.Wait()

	// ----------------------------------------------------------
	// 模式7: Once（只执行一次）
	// ----------------------------------------------------------
	fmt.Println("\n=== sync.Once ===")

	var once sync.Once
	var wg2 sync.WaitGroup

	initFunc := func() {
		fmt.Println("初始化只执行一次")
	}

	for i := 0; i < 5; i++ {
		wg2.Add(1)
		go func(n int) {
			defer wg2.Done()
			fmt.Printf("Goroutine %d 尝试初始化\n", n)
			once.Do(initFunc) // 只有第一个执行
		}(i)
	}
	wg2.Wait()

	// ----------------------------------------------------------
	// 模式8: 超时与重试
	// ----------------------------------------------------------
	fmt.Println("\n=== 超时与重试 ===")

	result, err := doWithRetry(3, 50*time.Millisecond, func() error {
		// 模拟可能失败的操作
		return fmt.Errorf("操作失败")
	})

	if err != nil {
		fmt.Println("最终失败:", err)
	} else {
		fmt.Println("成功:", result)
	}

	// ----------------------------------------------------------
	// 模式9: Rate Limiting（限流）
	// ----------------------------------------------------------
	fmt.Println("\n=== Rate Limiting ===")

	// 简单限流：每 50ms 处理一个请求
	limiter := time.Tick(50 * time.Millisecond)

	start := time.Now()
	for i := 1; i <= 3; i++ {
		<-limiter // 等待令牌
		fmt.Printf("请求 %d 处理于 %v\n", i, time.Since(start).Round(time.Millisecond))
	}
}

// ============================================================
//                      辅助函数
// ============================================================

// Worker Pool worker
func worker(id int, jobs <-chan int, results chan<- int) {
	for job := range jobs {
		fmt.Printf("Worker %d 处理任务 %d\n", id, job)
		time.Sleep(20 * time.Millisecond)
		results <- job * 2
	}
}

// 生成数字
func generateNumbers(start, count int) <-chan int {
	out := make(chan int)
	go func() {
		for i := 0; i < count; i++ {
			out <- start + i
		}
		close(out)
	}()
	return out
}

// 计算平方
func squareWorker(in <-chan int) <-chan int {
	out := make(chan int)
	go func() {
		for n := range in {
			out <- n * n
		}
		close(out)
	}()
	return out
}

// Fan-in: 合并多个 channel
func fanIn(channels ...<-chan int) <-chan int {
	out := make(chan int)
	var wg sync.WaitGroup

	for _, ch := range channels {
		wg.Add(1)
		go func(c <-chan int) {
			defer wg.Done()
			for n := range c {
				out <- n
			}
		}(ch)
	}

	go func() {
		wg.Wait()
		close(out)
	}()

	return out
}

// Pipeline: 生成
func generate(nums ...int) <-chan int {
	out := make(chan int)
	go func() {
		for _, n := range nums {
			out <- n
		}
		close(out)
	}()
	return out
}

// Pipeline: 翻倍
func double(in <-chan int) <-chan int {
	out := make(chan int)
	go func() {
		for n := range in {
			out <- n * 2
		}
		close(out)
	}()
	return out
}

// Pipeline: 加10
func addTen(in <-chan int) <-chan int {
	out := make(chan int)
	go func() {
		for n := range in {
			out <- n + 10
		}
		close(out)
	}()
	return out
}

// 长时间运行的任务
func longRunningTask(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			fmt.Println("任务超时:", ctx.Err())
			return
		default:
			fmt.Println("长任务执行中...")
			time.Sleep(50 * time.Millisecond)
		}
	}
}

// 带重试的操作
func doWithRetry(maxRetries int, delay time.Duration, fn func() error) (string, error) {
	var err error
	for i := 0; i < maxRetries; i++ {
		fmt.Printf("尝试 %d/%d\n", i+1, maxRetries)
		err = fn()
		if err == nil {
			return "成功", nil
		}
		time.Sleep(delay)
	}
	return "", fmt.Errorf("重试 %d 次后失败: %w", maxRetries, err)
}

// ============================================================
//                      重要注意事项
// ============================================================
//
// 1. 【Worker Pool】
//    - 控制并发数量
//    - 复用 goroutine，减少创建开销
//
// 2. 【Fan-out/Fan-in】
//    - Fan-out: 多个 goroutine 读取同一 channel
//    - Fan-in: 多个 channel 合并到一个
//
// 3. 【Pipeline】
//    - 数据流经多个处理阶段
//    - 每个阶段是独立的 goroutine
//    - 使用 channel 连接
//
// 4. 【Context】
//    - 传递取消信号
//    - 设置超时和截止时间
//    - 传递请求范围的值
//    - 始终调用 cancel 释放资源
//
// 5. 【信号量】
//    - 带缓冲 channel 可作为信号量
//    - 容量 = 最大并发数
//
// 6. 【sync.Once】
//    - 确保初始化只执行一次
//    - 线程安全
//
// 7. 【goroutine 泄漏】
//    - 确保所有 goroutine 有退出路径
//    - 使用 done channel 或 context
