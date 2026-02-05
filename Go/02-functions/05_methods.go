package main

import (
	"fmt"
	"math"
)

// ============================================================
//                      方法（Methods）
// ============================================================
// 方法是绑定到特定类型的函数
// 语法: func (接收者 类型) 方法名(参数) 返回类型 { }
//
// 【与函数的区别】
// - 函数: func 函数名(参数) 返回类型
// - 方法: func (接收者 类型) 方法名(参数) 返回类型
//
// 【接收者】类似于其他语言的 this 或 self

func main05() {
	fmt.Println("\n==================== 05_methods ====================")
	fmt.Println("=== 方法基础 ===")

	// ----------------------------------------------------------
	// 值接收者方法
	// ----------------------------------------------------------
	r := Rectangle{Width: 10, Height: 5}

	// 调用方法：实例.方法名()
	fmt.Printf("矩形: %+v\n", r)
	fmt.Println("面积:", r.Area())
	fmt.Println("周长:", r.Perimeter())

	// ----------------------------------------------------------
	// 指针接收者方法
	// ----------------------------------------------------------
	fmt.Println("\n=== 指针接收者 ===")

	fmt.Printf("缩放前: %+v\n", r)
	r.Scale(2) // 修改原始值
	fmt.Printf("缩放后: %+v\n", r)

	// 【重要】Go 自动处理指针/值转换
	// r.Scale(2) 等价于 (&r).Scale(2)

	// ----------------------------------------------------------
	// 值接收者 vs 指针接收者
	// ----------------------------------------------------------
	fmt.Println("\n=== 值 vs 指针接收者对比 ===")

	c := Counter{count: 0}

	c.IncrementValue() // 值接收者，不会修改原值
	fmt.Println("IncrementValue 后:", c.count) // 0

	c.IncrementPointer() // 指针接收者，会修改原值
	fmt.Println("IncrementPointer 后:", c.count) // 1

	// ----------------------------------------------------------
	// 方法链
	// ----------------------------------------------------------
	fmt.Println("\n=== 方法链 ===")

	builder := &StringBuilder{}
	result := builder.
		Append("Hello").
		Append(", ").
		Append("World").
		Append("!").
		String()

	fmt.Println("方法链结果:", result)

	// ----------------------------------------------------------
	// 自定义类型的方法
	// ----------------------------------------------------------
	fmt.Println("\n=== 自定义类型方法 ===")

	var m MyInt = 10
	fmt.Println("MyInt:", m)
	fmt.Println("Double:", m.Double())
	fmt.Println("IsPositive:", m.IsPositive())

	var m2 MyInt = -5
	fmt.Println("MyInt:", m2, "IsPositive:", m2.IsPositive())

	// ----------------------------------------------------------
	// 切片类型的方法
	// ----------------------------------------------------------
	fmt.Println("\n=== 切片类型方法 ===")

	nums := IntSlice{3, 1, 4, 1, 5, 9, 2, 6}
	fmt.Println("原切片:", nums)
	fmt.Println("Sum:", nums.Sum())
	fmt.Println("Max:", nums.Max())
	fmt.Println("Min:", nums.Min())

	// ----------------------------------------------------------
	// 嵌入类型与方法提升
	// ----------------------------------------------------------
	fmt.Println("\n=== 嵌入类型（方法提升）===")

	e := Employee{
		Person: Person{Name: "张三", Age: 30},
		Title:  "工程师",
		Salary: 10000,
	}

	// 可以直接调用嵌入类型的方法
	e.Greet()               // Person 的方法
	fmt.Println(e.Describe()) // Employee 的方法

	// 也可以访问嵌入类型的字段
	fmt.Println("姓名:", e.Name) // 等价于 e.Person.Name

	// ----------------------------------------------------------
	// 方法覆盖
	// ----------------------------------------------------------
	fmt.Println("\n=== 方法覆盖 ===")

	admin := Admin{
		Employee: Employee{
			Person: Person{Name: "李四", Age: 35},
			Title:  "管理员",
			Salary: 20000,
		},
		Level: 5,
	}

	admin.Greet() // Admin 覆盖了 Greet 方法

	// 仍可以调用被覆盖的方法
	admin.Person.Greet()   // 调用 Person 的 Greet
	admin.Employee.Greet() // 调用 Person 的 Greet（通过 Employee）

	// ----------------------------------------------------------
	// 方法值与方法表达式
	// ----------------------------------------------------------
	fmt.Println("\n=== 方法值与方法表达式 ===")

	rect := Rectangle{Width: 5, Height: 3}

	// 方法值：绑定到特定实例
	areaFunc := rect.Area
	fmt.Println("方法值 areaFunc():", areaFunc())

	// 方法表达式：需要传递接收者
	areaExpr := Rectangle.Area
	fmt.Println("方法表达式 areaExpr(rect):", areaExpr(rect))

	// 指针方法表达式
	scaleExpr := (*Rectangle).Scale
	scaleExpr(&rect, 2)
	fmt.Printf("方法表达式 Scale 后: %+v\n", rect)

	// ----------------------------------------------------------
	// nil 接收者
	// ----------------------------------------------------------
	fmt.Println("\n=== nil 接收者 ===")

	var list *IntList = nil
	fmt.Println("nil 列表长度:", list.Len()) // 可以安全调用

	list = &IntList{data: []int{1, 2, 3}}
	fmt.Println("实际列表长度:", list.Len())
}

// ============================================================
//                      类型与方法定义
// ============================================================

// ----------------------------------------------------------
// 基本结构体与方法
// ----------------------------------------------------------

// Rectangle 矩形结构体
type Rectangle struct {
	Width  float64
	Height float64
}

// Area 计算面积（值接收者）
// 【值接收者】方法内部是副本，不会修改原值
// 【适用场景】
// - 不需要修改接收者
// - 接收者是小型结构体（复制成本低）
// - 需要值语义（如 time.Time）
func (r Rectangle) Area() float64 {
	return r.Width * r.Height
}

// Perimeter 计算周长（值接收者）
func (r Rectangle) Perimeter() float64 {
	return 2 * (r.Width + r.Height)
}

// Scale 缩放矩形（指针接收者）
// 【指针接收者】方法内部操作原始值
// 【适用场景】
// - 需要修改接收者
// - 接收者是大型结构体（避免复制）
// - 需要一致性（如果有一个方法需要指针，全部用指针）
func (r *Rectangle) Scale(factor float64) {
	r.Width *= factor
	r.Height *= factor
}

// ----------------------------------------------------------
// 值接收者 vs 指针接收者对比
// ----------------------------------------------------------

type Counter struct {
	count int
}

// IncrementValue 值接收者 - 不会修改原值
func (c Counter) IncrementValue() {
	c.count++ // 修改的是副本
}

// IncrementPointer 指针接收者 - 会修改原值
func (c *Counter) IncrementPointer() {
	c.count++ // 修改的是原值
}

// ----------------------------------------------------------
// 方法链（Fluent Interface）
// ----------------------------------------------------------

type StringBuilder struct {
	data []byte
}

// Append 追加字符串，返回指针以支持链式调用
func (sb *StringBuilder) Append(s string) *StringBuilder {
	sb.data = append(sb.data, s...)
	return sb // 返回自身指针
}

// String 获取结果
func (sb *StringBuilder) String() string {
	return string(sb.data)
}

// ----------------------------------------------------------
// 自定义类型的方法
// ----------------------------------------------------------
// 【重要】只能为当前包内定义的类型添加方法
// 不能为内置类型（int, string）或其他包的类型添加方法

// MyInt 自定义整数类型
type MyInt int

// Double 返回两倍值
func (m MyInt) Double() MyInt {
	return m * 2
}

// IsPositive 是否为正数
func (m MyInt) IsPositive() bool {
	return m > 0
}

// ----------------------------------------------------------
// 切片类型的方法
// ----------------------------------------------------------

type IntSlice []int

// Sum 求和
func (s IntSlice) Sum() int {
	total := 0
	for _, v := range s {
		total += v
	}
	return total
}

// Max 最大值
func (s IntSlice) Max() int {
	if len(s) == 0 {
		return 0
	}
	max := s[0]
	for _, v := range s[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

// Min 最小值
func (s IntSlice) Min() int {
	if len(s) == 0 {
		return 0
	}
	min := s[0]
	for _, v := range s[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

// ----------------------------------------------------------
// 嵌入类型与方法提升
// ----------------------------------------------------------

// Person 人员基类
type Person struct {
	Name string
	Age  int
}

// Greet 打招呼
func (p Person) Greet() {
	fmt.Printf("你好，我是%s，今年%d岁\n", p.Name, p.Age)
}

// Employee 员工（嵌入 Person）
type Employee struct {
	Person        // 嵌入类型（匿名字段）
	Title  string
	Salary float64
}

// Describe 描述员工
func (e Employee) Describe() string {
	return fmt.Sprintf("%s - %s (%.0f元/月)", e.Name, e.Title, e.Salary)
}

// Admin 管理员（嵌入 Employee）
type Admin struct {
	Employee
	Level int
}

// Greet 覆盖 Person 的 Greet 方法
func (a Admin) Greet() {
	fmt.Printf("你好，我是%d级管理员%s\n", a.Level, a.Name)
}

// ----------------------------------------------------------
// 支持 nil 接收者的方法
// ----------------------------------------------------------

type IntList struct {
	data []int
}

// Len 返回列表长度（安全处理 nil）
// 【技巧】方法可以安全处理 nil 接收者
func (l *IntList) Len() int {
	if l == nil {
		return 0
	}
	return len(l.data)
}

// ----------------------------------------------------------
// 几何形状示例（用于展示多态，将在接口章节详细介绍）
// ----------------------------------------------------------

type Circle struct {
	Radius float64
}

func (c Circle) Area() float64 {
	return math.Pi * c.Radius * c.Radius
}

func (c Circle) Perimeter() float64 {
	return 2 * math.Pi * c.Radius
}

// ============================================================
//                      重要注意事项
// ============================================================
//
// 1. 【值接收者 vs 指针接收者选择】
//    使用指针接收者的情况：
//    - 需要修改接收者的值
//    - 接收者是大型结构体
//    - 一致性：类型的其他方法使用指针接收者
//
//    使用值接收者的情况：
//    - 不需要修改接收者
//    - 接收者是小型结构体或基本类型
//    - 需要值语义（不可变性）
//
// 2. 【自动转换】
//    Go 会自动在值和指针之间转换来调用方法
//    r.Scale(2)  // r 是值，自动转为 (&r).Scale(2)
//    p.Area()    // p 是指针，自动转为 (*p).Area()
//
// 3. 【只能为本包类型添加方法】
//    不能为 int、string 等内置类型添加方法
//    解决方案：定义新类型 type MyInt int
//
// 4. 【方法集规则】
//    - 类型 T 的方法集：所有接收者为 T 的方法
//    - 类型 *T 的方法集：所有接收者为 T 或 *T 的方法
//    这在实现接口时很重要
//
// 5. 【嵌入不是继承】
//    Go 的嵌入是组合，不是继承
//    - 没有 super 关键字
//    - 被嵌入类型不知道外部类型
//    - 方法提升是语法糖
//
// 6. 【命名冲突】
//    如果嵌入类型和外部类型有同名方法/字段
//    外部类型的优先级更高（遮蔽）
