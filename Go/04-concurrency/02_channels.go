package main

import (
	"fmt"
	"time"
)

// ============================================================
//                      Channel 通道
// ============================================================
// Channel 是 goroutine 之间通信的管道
// Go 的并发哲学：不要通过共享内存来通信，而要通过通信来共享内存

func main02() {
	fmt.Println("\n==================== 02_channels ====================")
	fmt.Println("=== Channel 基础 ===")

	// ----------------------------------------------------------
	// 创建 Channel
	// ----------------------------------------------------------
	// 无缓冲 channel：发送和接收必须同步
	ch1 := make(chan int)

	// 有缓冲 channel：缓冲区满时才阻塞
	ch2 := make(chan string, 3)

	fmt.Printf("无缓冲 channel: %T\n", ch1)
	fmt.Printf("有缓冲 channel: %T, 容量=%d\n", ch2, cap(ch2))

	// ----------------------------------------------------------
	// 发送和接收
	// ----------------------------------------------------------
	fmt.Println("\n=== 发送和接收 ===")

	// 必须在不同 goroutine 中操作无缓冲 channel
	go func() {
		ch1 <- 42 // 发送
	}()

	value := <-ch1 // 接收
	fmt.Println("收到:", value)

	// 有缓冲 channel 可以在同一个 goroutine 中操作
	ch2 <- "A"
	ch2 <- "B"
	fmt.Println("缓冲 channel 收到:", <-ch2, <-ch2)

	// ----------------------------------------------------------
	// Channel 方向
	// ----------------------------------------------------------
	fmt.Println("\n=== Channel 方向 ===")

	ch := make(chan int, 1)

	// 只写 channel
	go sendOnly(ch)

	// 只读 channel
	go receiveOnly(ch)

	time.Sleep(50 * time.Millisecond)

	// ----------------------------------------------------------
	// 关闭 Channel
	// ----------------------------------------------------------
	fmt.Println("\n=== 关闭 Channel ===")

	dataCh := make(chan int, 5)

	// 发送数据后关闭
	go func() {
		for i := 1; i <= 5; i++ {
			dataCh <- i
		}
		close(dataCh) // 关闭 channel
	}()

	// 接收数据
	for {
		v, ok := <-dataCh
		if !ok {
			fmt.Println("Channel 已关闭")
			break
		}
		fmt.Println("收到:", v)
	}

	// ----------------------------------------------------------
	// range 遍历 Channel
	// ----------------------------------------------------------
	fmt.Println("\n=== range 遍历 Channel ===")

	numCh := make(chan int, 3)

	go func() {
		for i := 10; i <= 30; i += 10 {
			numCh <- i
		}
		close(numCh)
	}()

	// range 会自动检测 channel 关闭
	for num := range numCh {
		fmt.Println("range 收到:", num)
	}

	// ----------------------------------------------------------
	// Select 多路复用
	// ----------------------------------------------------------
	fmt.Println("\n=== Select 多路复用 ===")

	ch1 = make(chan int)
	ch2New := make(chan string)

	go func() {
		time.Sleep(30 * time.Millisecond)
		ch1 <- 100
	}()

	go func() {
		time.Sleep(20 * time.Millisecond)
		ch2New <- "hello"
	}()

	// select 等待多个 channel
	for i := 0; i < 2; i++ {
		select {
		case v := <-ch1:
			fmt.Println("从 ch1 收到:", v)
		case v := <-ch2New:
			fmt.Println("从 ch2 收到:", v)
		}
	}

	// ----------------------------------------------------------
	// Select 超时处理
	// ----------------------------------------------------------
	fmt.Println("\n=== Select 超时 ===")

	slowCh := make(chan int)

	go func() {
		time.Sleep(200 * time.Millisecond)
		slowCh <- 999
	}()

	select {
	case v := <-slowCh:
		fmt.Println("收到:", v)
	case <-time.After(100 * time.Millisecond):
		fmt.Println("超时!")
	}

	// ----------------------------------------------------------
	// Select default（非阻塞）
	// ----------------------------------------------------------
	fmt.Println("\n=== Select default ===")

	emptyCh := make(chan int)

	select {
	case v := <-emptyCh:
		fmt.Println("收到:", v)
	default:
		fmt.Println("没有数据可读，继续执行")
	}

	// 非阻塞发送
	fullCh := make(chan int, 1)
	fullCh <- 1 // 缓冲区满

	select {
	case fullCh <- 2:
		fmt.Println("发送成功")
	default:
		fmt.Println("缓冲区满，发送失败")
	}

	// ----------------------------------------------------------
	// 单向 Channel 转换
	// ----------------------------------------------------------
	fmt.Println("\n=== 单向 Channel ===")

	biCh := make(chan int, 1)

	// 双向可以赋值给单向
	var sendCh chan<- int = biCh // 只写
	var recvCh <-chan int = biCh // 只读

	sendCh <- 42
	fmt.Println("单向接收:", <-recvCh)

	// ----------------------------------------------------------
	// nil Channel
	// ----------------------------------------------------------
	fmt.Println("\n=== nil Channel ===")

	var nilCh chan int

	// nil channel 永远阻塞
	// <-nilCh   // 永远阻塞
	// nilCh <- 1 // 永远阻塞

	// 在 select 中可以用 nil 禁用某个 case
	select {
	case <-nilCh:
		fmt.Println("不会执行")
	default:
		fmt.Println("nil channel 被跳过")
	}
}

// ----------------------------------------------------------
// 单向 Channel 函数参数
// ----------------------------------------------------------

// 只能发送
func sendOnly(ch chan<- int) {
	ch <- 100
	fmt.Println("sendOnly: 发送完成")
}

// 只能接收
func receiveOnly(ch <-chan int) {
	v := <-ch
	fmt.Println("receiveOnly: 收到", v)
}

// ============================================================
//                      重要注意事项
// ============================================================
//
// 1. 【创建 Channel】
//    make(chan Type)      // 无缓冲
//    make(chan Type, n)   // 有缓冲，容量 n
//
// 2. 【无缓冲 vs 有缓冲】
//    无缓冲：同步通信，发送阻塞直到有人接收
//    有缓冲：异步通信，缓冲区满时才阻塞
//
// 3. 【关闭 Channel】
//    - 只有发送方应该关闭
//    - 关闭后不能再发送（panic）
//    - 关闭后可以继续接收，返回零值
//    - 重复关闭会 panic
//
// 4. 【检测关闭】
//    v, ok := <-ch  // ok 为 false 表示已关闭
//    for v := range ch  // 自动检测关闭
//
// 5. 【select 规则】
//    - 多个 case 就绪时，随机选择一个
//    - 没有 case 就绪且有 default，执行 default
//    - 没有 case 就绪且无 default，阻塞
//
// 6. 【常见模式】
//    - 超时: select + time.After
//    - 退出: select + done channel
//    - 非阻塞: select + default
//
// 7. 【死锁注意】
//    - 无缓冲 channel 在同一 goroutine 中发送和接收会死锁
//    - 所有 goroutine 都阻塞时程序会 panic
