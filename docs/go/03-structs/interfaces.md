# interfaces.go

::: info 文件信息
- 📄 原文件：`02_interfaces.go`
- 🔤 语言：go
:::

## 完整代码

```go
package main

import (
	"fmt"
	"math"
)

// ============================================================
//                      接口（Interfaces）
// ============================================================
// 接口是 Go 实现多态的核心机制
// 接口定义行为（方法签名），不定义实现
//
// 【核心特点】隐式实现
// - 不需要 implements 关键字
// - 只要类型实现了接口的所有方法，就自动实现了该接口
// - 这是 Go 的"鸭子类型"：如果它走起来像鸭子，叫起来像鸭子，那它就是鸭子

func main02() {
	fmt.Println("\n==================== 02_interfaces ====================")
	fmt.Println("=== 接口基础 ===")

	// ----------------------------------------------------------
	// 接口变量
	// ----------------------------------------------------------
	// 接口变量可以存储任何实现了该接口的值

	var s Shape // 接口变量，零值为 nil

	// 赋值为 Circle
	s = Circle{Radius: 5}
	fmt.Printf("Circle: 面积=%.2f, 周长=%.2f\n", s.Area(), s.Perimeter())

	// 赋值为 Rect
	s = Rect{Width: 4, Height: 3}
	fmt.Printf("Rect: 面积=%.2f, 周长=%.2f\n", s.Area(), s.Perimeter())

	// ----------------------------------------------------------
	// 接口的多态
	// ----------------------------------------------------------
	fmt.Println("\n=== 接口多态 ===")

	shapes := []Shape{
		Circle{Radius: 3},
		Rect{Width: 4, Height: 5},
		Circle{Radius: 2},
		Rect{Width: 6, Height: 3},
	}

	// 统一处理不同类型
	for i, shape := range shapes {
		fmt.Printf("形状%d: 面积=%.2f\n", i+1, shape.Area())
	}

	// 计算总面积
	total := TotalArea(shapes)
	fmt.Printf("总面积: %.2f\n", total)

	// ----------------------------------------------------------
	// 类型断言
	// ----------------------------------------------------------
	fmt.Println("\n=== 类型断言 ===")

	var shape Shape = Circle{Radius: 5}

	// 方式1: 直接断言（可能 panic）
	circle := shape.(Circle)
	fmt.Printf("断言成功: %+v\n", circle)

	// 方式2: 安全断言（推荐）
	if c, ok := shape.(Circle); ok {
		fmt.Printf("是 Circle: 半径=%v\n", c.Radius)
	}

	if r, ok := shape.(Rect); ok {
		fmt.Printf("是 Rect: %+v\n", r)
	} else {
		fmt.Println("不是 Rect")
	}

	// ----------------------------------------------------------
	// 类型选择（Type Switch）
	// ----------------------------------------------------------
	fmt.Println("\n=== 类型选择 ===")

	PrintShapeInfo(Circle{Radius: 5})
	PrintShapeInfo(Rect{Width: 4, Height: 3})
	PrintShapeInfo("hello") // 非 Shape 类型

	// ----------------------------------------------------------
	// 空接口
	// ----------------------------------------------------------
	fmt.Println("\n=== 空接口 ===")

	// any 是 interface{} 的别名
	var anything any

	anything = 42
	fmt.Printf("int: %v (类型: %T)\n", anything, anything)

	anything = "hello"
	fmt.Printf("string: %v (类型: %T)\n", anything, anything)

	anything = Circle{Radius: 3}
	fmt.Printf("Circle: %v (类型: %T)\n", anything, anything)

	// 使用空接口的函数
	PrintAny(123)
	PrintAny("world")
	PrintAny([]int{1, 2, 3})

	// ----------------------------------------------------------
	// 接口组合
	// ----------------------------------------------------------
	fmt.Println("\n=== 接口组合 ===")

	var rw ReadWriter = &Buffer{data: []byte("Hello")}

	// 可以调用 Reader 和 Writer 的方法
	data := make([]byte, 10)
	n, _ := rw.Read(data)
	fmt.Printf("读取: %s (%d bytes)\n", data[:n], n)

	rw.Write([]byte(" World"))
	n, _ = rw.Read(data)
	fmt.Printf("再读: %s (%d bytes)\n", data[:n], n)

	// 接口可以赋值给其组成部分
	var r Reader = rw
	var w Writer = rw
	fmt.Printf("Reader: %T, Writer: %T\n", r, w)

	// ----------------------------------------------------------
	// 接口值的内部结构
	// ----------------------------------------------------------
	fmt.Println("\n=== 接口值内部 ===")

	var s2 Shape
	fmt.Printf("nil 接口: value=%v, type=%T\n", s2, s2)

	s2 = Circle{Radius: 5}
	fmt.Printf("赋值后: value=%v, type=%T\n", s2, s2)

	// 【重要】nil 接口 vs 持有 nil 值的接口
	var c *Circle = nil
	s2 = c // 接口持有 nil 指针
	fmt.Printf("持有nil: value=%v, type=%T, isNil=%t\n", s2, s2, s2 == nil)
	// 注意: s2 != nil，因为接口有类型信息

	// ----------------------------------------------------------
	// 常见标准库接口
	// ----------------------------------------------------------
	fmt.Println("\n=== 常见标准库接口 ===")

	// Stringer 接口（类似 Java 的 toString）
	p := PersonWithStringer{Name: "张三", Age: 25}
	fmt.Println("Stringer:", p) // 自动调用 String() 方法

	// error 接口
	err := &MyError{Code: 404, Message: "Not Found"}
	fmt.Println("Error:", err.Error())
}

// ============================================================
//                      接口定义
// ============================================================

// ----------------------------------------------------------
// 基本接口
// ----------------------------------------------------------
// 语法: type 接口名 interface { 方法签名... }

// Shape 形状接口
type Shape interface {
	Area() float64      // 面积
	Perimeter() float64 // 周长
}

// ----------------------------------------------------------
// 实现接口的类型
// ----------------------------------------------------------

// Circle 圆形
type Circle struct {
	Radius float64
}

// 实现 Shape 接口（隐式实现）
func (c Circle) Area() float64 {
	return math.Pi * c.Radius * c.Radius
}

func (c Circle) Perimeter() float64 {
	return 2 * math.Pi * c.Radius
}

// Rect 矩形
type Rect struct {
	Width, Height float64
}

// 实现 Shape 接口
func (r Rect) Area() float64 {
	return r.Width * r.Height
}

func (r Rect) Perimeter() float64 {
	return 2 * (r.Width + r.Height)
}

// ----------------------------------------------------------
// 使用接口的函数
// ----------------------------------------------------------

// TotalArea 计算多个形状的总面积
func TotalArea(shapes []Shape) float64 {
	total := 0.0
	for _, s := range shapes {
		total += s.Area()
	}
	return total
}

// PrintShapeInfo 打印形状信息（使用类型选择）
func PrintShapeInfo(v any) {
	switch s := v.(type) {
	case Circle:
		fmt.Printf("圆形: 半径=%.2f, 面积=%.2f\n", s.Radius, s.Area())
	case Rect:
		fmt.Printf("矩形: %.2fx%.2f, 面积=%.2f\n", s.Width, s.Height, s.Area())
	default:
		fmt.Printf("未知类型: %T\n", v)
	}
}

// ----------------------------------------------------------
// 空接口
// ----------------------------------------------------------
// interface{} 或 any 可以接受任何类型

// PrintAny 打印任意类型
func PrintAny(v any) {
	fmt.Printf("PrintAny: %v (类型: %T)\n", v, v)
}

// ----------------------------------------------------------
// 接口组合
// ----------------------------------------------------------
// 接口可以嵌入其他接口，形成更大的接口

// Reader 读取接口
type Reader interface {
	Read(p []byte) (n int, err error)
}

// Writer 写入接口
type Writer interface {
	Write(p []byte) (n int, err error)
}

// ReadWriter 组合接口
type ReadWriter interface {
	Reader // 嵌入 Reader
	Writer // 嵌入 Writer
}

// Buffer 实现 ReadWriter
type Buffer struct {
	data []byte
	pos  int
}

func (b *Buffer) Read(p []byte) (int, error) {
	if b.pos >= len(b.data) {
		return 0, fmt.Errorf("EOF")
	}
	n := copy(p, b.data[b.pos:])
	b.pos += n
	return n, nil
}

func (b *Buffer) Write(p []byte) (int, error) {
	b.data = append(b.data, p...)
	return len(p), nil
}

// ----------------------------------------------------------
// 常见标准库接口示例
// ----------------------------------------------------------

// fmt.Stringer 接口
type PersonWithStringer struct {
	Name string
	Age  int
}

func (p PersonWithStringer) String() string {
	return fmt.Sprintf("%s (%d岁)", p.Name, p.Age)
}

// error 接口
type MyError struct {
	Code    int
	Message string
}

func (e *MyError) Error() string {
	return fmt.Sprintf("错误 %d: %s", e.Code, e.Message)
}

// ============================================================
//                      重要注意事项
// ============================================================
//
// 1. 【隐式实现】
//    不需要 implements 关键字
//    实现所有方法 = 实现接口
//
// 2. 【接口命名惯例】
//    - 单方法接口：方法名 + er（Reader, Writer, Stringer）
//    - 多方法接口：描述性名词（Shape, Vehicle）
//
// 3. 【接口设计原则】
//    - 接口应该小而专注
//    - "接受接口，返回结构体"
//    - 在使用方定义接口，而非实现方
//
// 4. 【类型断言】
//    - v.(Type): 可能 panic
//    - v, ok := v.(Type): 安全方式
//
// 5. 【nil 接口陷阱】
//    var s Shape = nil      // s == nil ✓
//    var c *Circle = nil
//    s = c                  // s != nil ✗（有类型信息）
//
// 6. 【方法集规则】
//    - T 的方法集：接收者为 T 的方法
//    - *T 的方法集：接收者为 T 或 *T 的方法
//    这影响接口实现判断
//
// 7. 【空接口 any】
//    - 可以接受任何类型
//    - 使用前需要类型断言
//    - 会失去类型安全，谨慎使用
```
