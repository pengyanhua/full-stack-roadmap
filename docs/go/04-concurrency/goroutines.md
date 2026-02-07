# goroutines.go

::: info 文件信息
- 📄 原文件：`01_goroutines.go`
- 🔤 语言：go
:::

## 完整代码

```go
package main

import (
	"fmt"
	"runtime"
	"sync"
	"time"
)

// ============================================================
//                      Goroutine 基础
// ============================================================
// Goroutine 是 Go 的轻量级线程，由 Go 运行时管理
// 比操作系统线程更轻量：初始栈只有 2KB，可动态增长
// 一个程序可以轻松创建数十万个 goroutine

func main() {
	fmt.Println("=== Goroutine 基础 ===")

	// ----------------------------------------------------------
	// 启动 Goroutine
	// ----------------------------------------------------------
	// 语法: go 函数调用
	// 【注意】main 函数退出时，所有 goroutine 都会被终止

	fmt.Println("主 goroutine 开始")

	// 启动一个新 goroutine
	go sayHello("世界")

	// 启动匿名函数 goroutine
	go func() {
		fmt.Println("匿名 goroutine")
	}()

	// 带参数的匿名 goroutine
	message := "你好"
	go func(msg string) {
		fmt.Println("带参数:", msg)
	}(message) // 【重要】参数在此传入，避免闭包陷阱

	// 等待一下让 goroutine 执行（生产代码不应该这样做）
	time.Sleep(100 * time.Millisecond)

	fmt.Println("主 goroutine 结束")

	// ----------------------------------------------------------
	// 多个 Goroutine
	// ----------------------------------------------------------
	fmt.Println("\n=== 多个 Goroutine ===")

	for i := 1; i <= 5; i++ {
		go func(n int) {
			fmt.Printf("Goroutine %d 开始\n", n)
			time.Sleep(time.Duration(n*10) * time.Millisecond)
			fmt.Printf("Goroutine %d 结束\n", n)
		}(i)
	}

	time.Sleep(200 * time.Millisecond)

	// ----------------------------------------------------------
	// WaitGroup：等待多个 goroutine 完成
	// ----------------------------------------------------------
	fmt.Println("\n=== WaitGroup ===")

	var wg sync.WaitGroup

	for i := 1; i <= 3; i++ {
		wg.Add(1) // 增加计数器

		go func(n int) {
			defer wg.Done() // 完成时减少计数器

			fmt.Printf("Worker %d 开始工作\n", n)
			time.Sleep(time.Duration(n*50) * time.Millisecond)
			fmt.Printf("Worker %d 完成工作\n", n)
		}(i)
	}

	wg.Wait() // 阻塞直到计数器为 0
	fmt.Println("所有 Worker 完成")

	// ----------------------------------------------------------
	// Goroutine 数量
	// ----------------------------------------------------------
	fmt.Println("\n=== Goroutine 信息 ===")

	fmt.Println("当前 Goroutine 数量:", runtime.NumGoroutine())
	fmt.Println("CPU 核心数:", runtime.NumCPU())
	fmt.Println("GOMAXPROCS:", runtime.GOMAXPROCS(0))

	// ----------------------------------------------------------
	// 闭包陷阱（重要！）
	// ----------------------------------------------------------
	fmt.Println("\n=== 闭包陷阱 ===")

	// 【错误示例】循环变量被共享
	fmt.Println("错误方式:")
	var wg2 sync.WaitGroup
	for i := range 3 {
		wg2.Go(func() {
			fmt.Println("  i =", i) // 可能都打印 3
		})
	}
	wg2.Wait()

	// 【正确示例】通过参数传递
	fmt.Println("正确方式:")
	var wg3 sync.WaitGroup
	for i := range 3 {
		wg3.Add(1)
		go func(n int) {
			defer wg3.Done()
			fmt.Println("  n =", n)
		}(i) // 传递当前值
	}
	wg3.Wait()

	// ----------------------------------------------------------
	// 并发安全问题
	// ----------------------------------------------------------
	fmt.Println("\n=== 并发安全问题 ===")

	// 【错误示例】数据竞争
	counter := 0
	var wg4 sync.WaitGroup

	for range 1000 {
		wg4.Go(func() {
			counter++ // 数据竞争！
		})
	}
	wg4.Wait()
	fmt.Println("不安全计数器:", counter, "(可能小于1000)")

	// 【正确示例】使用 Mutex
	counter2 := 0
	var mu sync.Mutex
	var wg5 sync.WaitGroup

	for range 1000 {
		wg5.Go(func() {
			mu.Lock()
			counter2++
			mu.Unlock()
		})
	}
	wg5.Wait()
	fmt.Println("安全计数器:", counter2)

	main02()
	main03()
}

// ----------------------------------------------------------
// 辅助函数
// ----------------------------------------------------------

func sayHello(name string) {
	fmt.Println("Hello,", name)
}

// ============================================================
//                      重要注意事项
// ============================================================
//
// 1. 【启动语法】
//    go 函数名(参数)
//    go func() { ... }()
//
// 2. 【生命周期】
//    - main 退出时所有 goroutine 终止
//    - 没有父子关系，无法直接"杀死"goroutine
//    - 使用 channel 或 context 来协调
//
// 3. 【WaitGroup 使用】
//    - Add() 在启动 goroutine 前调用
//    - Done() 用 defer 确保调用
//    - Wait() 阻塞等待完成
//
// 4. 【闭包陷阱】
//    循环中启动 goroutine 时，通过参数传递循环变量
//
// 5. 【数据竞争】
//    - 多个 goroutine 访问共享数据需要同步
//    - 使用 Mutex、Channel 或 atomic
//    - go run -race 检测数据竞争
//
// 6. 【goroutine 泄漏】
//    - 确保 goroutine 有退出条件
//    - 阻塞的 goroutine 不会被 GC
//
// 7. 【调度】
//    - Go 使用 M:N 调度（M 个 goroutine 对应 N 个 OS 线程）
//    - GOMAXPROCS 控制并行度（默认等于 CPU 核心数）
```
