// Package mymath 提供基础数学运算函数
// 这是包的文档注释，会被 go doc 工具提取
package mymath

import "math"

// ============================================================
//                      包的可见性规则
// ============================================================
// 【核心规则】大写开头 = 可导出（public），小写开头 = 私有（private）
// 这适用于：函数、类型、变量、常量、结构体字段、方法

// ----------------------------------------------------------
// 可导出的常量（大写开头）
// ----------------------------------------------------------

// Pi 圆周率常量
const Pi = 3.14159265358979323846

// E 自然对数的底数
const E = 2.71828182845904523536

// ----------------------------------------------------------
// 私有常量（小写开头，包外不可访问）
// ----------------------------------------------------------

const precision = 1e-10

// ----------------------------------------------------------
// 可导出的变量
// ----------------------------------------------------------

// DefaultPrecision 默认精度
var DefaultPrecision = 6

// ----------------------------------------------------------
// 私有变量
// ----------------------------------------------------------

var cache = make(map[int]int)

// ----------------------------------------------------------
// 可导出的函数
// ----------------------------------------------------------

// Add 返回两个整数的和
// 函数文档注释应该以函数名开头
func Add(a, b int) int {
	return a + b
}

// Subtract 返回两个整数的差
func Subtract(a, b int) int {
	return a - b
}

// Multiply 返回两个整数的积
func Multiply(a, b int) int {
	return a * b
}

// Divide 返回两个整数的商
// 如果除数为零，返回 0
func Divide(a, b int) int {
	if b == 0 {
		return 0
	}
	return a / b
}

// Sqrt 返回平方根
func Sqrt(x float64) float64 {
	return math.Sqrt(x)
}

// Power 返回 x 的 n 次方
func Power(x float64, n int) float64 {
	return math.Pow(x, float64(n))
}

// Abs 返回绝对值
func Abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// Max 返回两个数中的较大值
func Max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Min 返回两个数中的较小值
func Min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Fibonacci 返回第 n 个斐波那契数（带缓存）
func Fibonacci(n int) int {
	if n <= 1 {
		return n
	}

	// 检查缓存
	if v, ok := cache[n]; ok {
		return v
	}

	// 计算并缓存
	result := Fibonacci(n-1) + Fibonacci(n-2)
	cache[n] = result
	return result
}

// ----------------------------------------------------------
// 私有函数（包外不可调用）
// ----------------------------------------------------------

// helper 是一个私有辅助函数
func helper(x int) int {
	return x * x
}

// validate 验证输入
func validate(x int) bool {
	return x >= 0
}

// ----------------------------------------------------------
// 可导出的类型
// ----------------------------------------------------------

// Point 表示二维平面上的点
type Point struct {
	X, Y float64 // 大写，可导出字段
}

// NewPoint 创建新的点（构造函数）
func NewPoint(x, y float64) *Point {
	return &Point{X: x, Y: y}
}

// Distance 计算到另一个点的距离
func (p Point) Distance(other Point) float64 {
	dx := p.X - other.X
	dy := p.Y - other.Y
	return math.Sqrt(dx*dx + dy*dy)
}

// ----------------------------------------------------------
// 带私有字段的类型
// ----------------------------------------------------------

// Calculator 计算器类型
type Calculator struct {
	Result    float64 // 可导出
	precision int     // 私有，包外无法直接访问
}

// NewCalculator 创建计算器
func NewCalculator() *Calculator {
	return &Calculator{
		Result:    0,
		precision: DefaultPrecision,
	}
}

// SetPrecision 设置精度（通过方法访问私有字段）
func (c *Calculator) SetPrecision(p int) {
	c.precision = p
}

// GetPrecision 获取精度
func (c *Calculator) GetPrecision() int {
	return c.precision
}
