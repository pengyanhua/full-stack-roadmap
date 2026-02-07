# control flow.go

::: info 文件信息
- 📄 原文件：`02_control_flow.go`
- 🔤 语言：go
:::

## 完整代码

```go
package main

import "fmt"

func main02() {

	fmt.Println("\n ====================  02_control_flow ====================")
	// ==================== if 语句 ====================
	// 【注意】Go 的 if 条件不需要括号 ()，但必须有花括号 {}
	// 【注意】左花括号 { 必须和 if 在同一行，否则编译错误！
	fmt.Println("=== if 语句 ===")

	score := 85

	// 基本 if-else
	if score >= 90 {
		fmt.Println("优秀")
	} else if score >= 60 {
		fmt.Println("及格")
	} else {
		fmt.Println("不及格")
	}

	// 【重要】if 带初始化语句
	// 格式: if 初始化语句; 条件 { }
	// 变量 num 的作用域仅在 if-else 块内，外部无法访问
	// 常用于错误处理: if err := doSomething(); err != nil { }
	if num := 10; num%2 == 0 {
		fmt.Println(num, "是偶数")
	}
	// fmt.Println(num)  // 错误！num 在这里不可见

	// ==================== for 循环 ====================
	// 【重要】Go 只有 for 循环，没有 while 和 do-while
	// 但 for 可以模拟所有循环形式
	fmt.Println("\n=== for 循环 ===")

	// 形式1: 标准 for 循环（类似 C/Java）
	// for 初始化; 条件; 后置语句 { }
	fmt.Print("标准 for: ")
	for i := range 5 {
		fmt.Print(i, " ")
	}
	fmt.Println()

	// 形式2: while 风格（省略初始化和后置语句）
	// for 条件 { }
	fmt.Print("while 风格: ")
	j := 0
	for j < 5 {
		fmt.Print(j, " ")
		j++
	}
	fmt.Println()

	// 形式3: 无限循环（省略所有）
	// for { } 等价于 for true { }
	// 【注意】必须有 break、return 或 panic 退出，否则死循环
	fmt.Print("无限循环 + break: ")
	k := 0
	for {
		if k >= 5 {
			break
		}
		fmt.Print(k, " ")
		k++
	}
	fmt.Println()

	// 形式4: for-range 遍历（最常用）
	// 可遍历: 数组、切片、字符串、map、channel
	fmt.Print("range 遍历切片: ")
	nums := []int{10, 20, 30, 40, 50}
	for index, value := range nums {
		fmt.Printf("[%d]=%d ", index, value)
	}
	fmt.Println()

	// 【注意】只需要索引时，可省略 value
	// for index := range nums { }

	// 【注意】只需要值时，用 _ 忽略索引
	// for _, value := range nums { }

	// 【重要】range 遍历字符串返回的是 rune（Unicode码点），不是字节
	// 索引是字节位置，所以中文字符索引会跳跃（UTF-8 中文占3字节）
	fmt.Print("range 遍历字符串: ")
	for i, char := range "Go语言" {
		fmt.Printf("[%d]=%c ", i, char)
	}
	fmt.Println()
	// 输出: [0]=G [1]=o [2]=语 [5]=言
	// 注意索引从 2 跳到 5，因为 "语" 占 3 个字节

	// break 和 continue
	// break: 跳出整个循环
	// continue: 跳过本次迭代，进入下一次
	fmt.Print("continue 跳过偶数: ")
	for i := range 10 {
		if i%2 == 0 {
			continue // 跳过偶数
		}
		fmt.Print(i, " ")
	}
	fmt.Println()

	// 【进阶】带标签的 break/continue（用于嵌套循环）
	fmt.Print("带标签 break: ")
outer:
	for i := range 3 {
		for j := range 3 {
			if i == 1 && j == 1 {
				break outer // 直接跳出外层循环
			}
			fmt.Printf("(%d,%d) ", i, j)
		}
	}
	fmt.Println()

	// ==================== switch 语句 ====================
	// 【重要】Go 的 switch 默认自动 break，不会穿透！
	// 这与 C/Java 不同，更安全
	fmt.Println("\n=== switch 语句 ===")

	day := 3

	// 基本 switch
	switch day {
	case 1:
		fmt.Println("星期一")
	case 2:
		fmt.Println("星期二")
	case 3:
		fmt.Println("星期三")
	case 4, 5: // 【技巧】多个值用逗号分隔
		fmt.Println("星期四或五")
	default:
		fmt.Println("周末")
	}

	// 【重要】switch 也可以带初始化语句
	switch num := 15; {
	case num < 10:
		fmt.Println("小于10")
	case num < 20:
		fmt.Println("10-19之间")
	default:
		fmt.Println("20或更大")
	}

	// 【技巧】无表达式 switch（替代长 if-else 链）
	// switch { } 等价于 switch true { }
	// case 后面是布尔表达式
	hour := 14
	switch {
	case hour < 12:
		fmt.Println("上午好")
	case hour < 18:
		fmt.Println("下午好")
	default:
		fmt.Println("晚上好")
	}

	// 【特殊】fallthrough 强制穿透到下一个 case
	// 注意: fallthrough 会无条件执行下一个 case，不检查条件
	// 实际开发中很少使用
	fmt.Print("fallthrough 示例: ")
	n := 1
	switch n {
	case 1:
		fmt.Print("一 ")
		fallthrough // 穿透到 case 2
	case 2:
		fmt.Print("二 ") // 即使 n != 2 也会执行
	case 3:
		fmt.Print("三")
	}
	fmt.Println()

	// ==================== defer 延迟执行 ====================
	// 【重要】defer 会在函数返回前执行，常用于资源清理
	// 典型场景: 关闭文件、解锁、关闭连接
	fmt.Println("\n=== defer 延迟执行 ===")

	fmt.Println("开始")
	defer fmt.Println("延迟1: 最后执行")
	defer fmt.Println("延迟2: 倒数第二")
	fmt.Println("结束")

	// 【重要】多个 defer 按 LIFO（后进先出/栈）顺序执行
	// 输出顺序: 开始 -> 结束 -> 延迟2 -> 延迟1

	// 【注意】defer 的参数在声明时就已求值，不是执行时
	// 例如:
	// x := 10
	// defer fmt.Println(x)  // 打印 10
	// x = 20                // 改变 x 不影响 defer

	// 【典型用法】
	// file, err := os.Open("file.txt")
	// if err != nil { return err }
	// defer file.Close()  // 确保函数结束时关闭文件
}
```
