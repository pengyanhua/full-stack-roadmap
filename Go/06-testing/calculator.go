// Package main 包含计算器功能及其测试示例
package main

import (
	"errors"
	"fmt"
)

// ============================================================
//                      被测试的代码
// ============================================================

// Calculator 计算器结构体
type Calculator struct {
	precision int
}

// NewCalculator 创建计算器
func NewCalculator() *Calculator {
	return &Calculator{precision: 2}
}

// Add 加法
func (c *Calculator) Add(a, b float64) float64 {
	return a + b
}

// Subtract 减法
func (c *Calculator) Subtract(a, b float64) float64 {
	return a - b
}

// Multiply 乘法
func (c *Calculator) Multiply(a, b float64) float64 {
	return a * b
}

// Divide 除法，除数为零时返回错误
func (c *Calculator) Divide(a, b float64) (float64, error) {
	if b == 0 {
		return 0, errors.New("division by zero")
	}
	return a / b, nil
}

// ----------------------------------------------------------
// 独立函数
// ----------------------------------------------------------

// Abs 返回绝对值
func Abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}

// Max 返回最大值
func Max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Fibonacci 计算斐波那契数
func Fibonacci(n int) int {
	if n <= 1 {
		return n
	}
	return Fibonacci(n-1) + Fibonacci(n-2)
}

// IsPrime 判断是否为素数
func IsPrime(n int) bool {
	if n < 2 {
		return false
	}
	for i := 2; i*i <= n; i++ {
		if n%i == 0 {
			return false
		}
	}
	return true
}

// Reverse 反转字符串
func Reverse(s string) string {
	runes := []rune(s)
	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]
	}
	return string(runes)
}

// ----------------------------------------------------------
// 用于 Benchmark 的函数
// ----------------------------------------------------------

// FibonacciIterative 迭代方式计算斐波那契
func FibonacciIterative(n int) int {
	if n <= 1 {
		return n
	}
	a, b := 0, 1
	for i := 2; i <= n; i++ {
		a, b = b, a+b
	}
	return b
}

// ----------------------------------------------------------
// 用于 Example 的函数
// ----------------------------------------------------------

// Greet 问候语
func Greet(name string) string {
	return fmt.Sprintf("Hello, %s!", name)
}

// Sum 求和
func Sum(nums ...int) int {
	total := 0
	for _, n := range nums {
		total += n
	}
	return total
}

func main() {
	fmt.Println("=== 测试示例代码 ===")
	fmt.Println("运行测试: go test -v")
	fmt.Println("运行基准测试: go test -bench=.")
	fmt.Println("查看覆盖率: go test -cover")
	fmt.Println("生成覆盖率报告: go test -coverprofile=coverage.out && go tool cover -html=coverage.out")
}
