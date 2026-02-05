package main

import (
	"encoding/json"
	"fmt"
)

// ============================================================
//                      结构体基础
// ============================================================
// 结构体是 Go 中最重要的复合类型
// 用于将多个相关的数据组织在一起
// Go 没有类（class），使用结构体 + 方法实现面向对象

func main() {
	fmt.Println("=== 结构体基础 ===")

	// ----------------------------------------------------------
	// 创建结构体实例
	// ----------------------------------------------------------

	// 方式1: 字段名赋值（推荐，顺序无关）
	p1 := Person{
		Name: "张三",
		Age:  25,
		City: "北京",
	}
	fmt.Printf("字段名赋值: %+v\n", p1)

	// 方式2: 按顺序赋值（不推荐，顺序敏感）
	p2 := Person{"李四", 30, "上海"}
	fmt.Printf("顺序赋值: %+v\n", p2)

	// 方式3: 零值初始化
	var p3 Person
	fmt.Printf("零值: %+v\n", p3) // 所有字段为零值

	// 方式4: 部分字段赋值（其余为零值）
	p4 := Person{Name: "王五"}
	fmt.Printf("部分赋值: %+v\n", p4)

	// 方式5: new 函数（返回指针）
	p5 := new(Person)
	p5.Name = "赵六"
	p5.Age = 35
	fmt.Printf("new 创建: %+v\n", *p5)

	// ----------------------------------------------------------
	// 访问和修改字段
	// ----------------------------------------------------------
	fmt.Println("\n=== 访问和修改字段 ===")

	fmt.Println("姓名:", p1.Name)
	fmt.Println("年龄:", p1.Age)

	p1.Age = 26 // 修改字段
	fmt.Println("修改后年龄:", p1.Age)

	// ----------------------------------------------------------
	// 结构体指针
	// ----------------------------------------------------------
	fmt.Println("\n=== 结构体指针 ===")

	pp := &Person{Name: "钱七", Age: 40, City: "广州"}

	// 【重要】Go 自动解引用，以下两种写法等价
	fmt.Println("指针访问:", pp.Name)   // 语法糖
	fmt.Println("显式解引用:", (*pp).Name) // 完整写法

	// 修改指针指向的值
	pp.Age = 41
	fmt.Printf("修改后: %+v\n", *pp)

	// ----------------------------------------------------------
	// 结构体比较
	// ----------------------------------------------------------
	fmt.Println("\n=== 结构体比较 ===")

	a := Person{Name: "Test", Age: 20, City: "Test"}
	b := Person{Name: "Test", Age: 20, City: "Test"}
	c := Person{Name: "Test", Age: 21, City: "Test"}

	fmt.Println("a == b:", a == b) // true
	fmt.Println("a == c:", a == c) // false

	// 【注意】包含切片、map、函数的结构体不能直接比较

	// ----------------------------------------------------------
	// 匿名结构体
	// ----------------------------------------------------------
	fmt.Println("\n=== 匿名结构体 ===")

	// 临时使用，无需定义类型
	point := struct {
		X, Y int
	}{10, 20}

	fmt.Printf("匿名结构体: %+v\n", point)

	// 常用于测试数据
	testCases := []struct {
		input    int
		expected int
	}{
		{1, 2},
		{2, 4},
		{3, 6},
	}
	fmt.Println("测试用例:", testCases)

	// ----------------------------------------------------------
	// 结构体嵌入（组合）
	// ----------------------------------------------------------
	fmt.Println("\n=== 结构体嵌入 ===")

	emp := Employee{
		Person:   Person{Name: "孙八", Age: 28, City: "深圳"},
		Title:    "工程师",
		Salary:   15000,
		Department: "技术部",
	}

	// 直接访问嵌入字段（字段提升）
	fmt.Println("姓名:", emp.Name)     // 等价于 emp.Person.Name
	fmt.Println("职位:", emp.Title)
	fmt.Printf("完整信息: %+v\n", emp)

	// ----------------------------------------------------------
	// 结构体标签（Tags）
	// ----------------------------------------------------------
	fmt.Println("\n=== 结构体标签 ===")

	user := User{
		ID:       1,
		Username: "johndoe",
		Email:    "john@example.com",
		Password: "secret123",
	}

	// JSON 序列化
	jsonData, _ := json.Marshal(user)
	fmt.Println("JSON:", string(jsonData))
	// 注意: password 字段被 json:"-" 排除

	// JSON 反序列化
	jsonStr := `{"id":2,"username":"janedoe","email":"jane@example.com"}`
	var user2 User
	json.Unmarshal([]byte(jsonStr), &user2)
	fmt.Printf("反序列化: %+v\n", user2)

	// ----------------------------------------------------------
	// 结构体方法
	// ----------------------------------------------------------
	fmt.Println("\n=== 结构体方法 ===")

	rect := Rectangle{Width: 10, Height: 5}
	fmt.Println("面积:", rect.Area())
	fmt.Println("周长:", rect.Perimeter())
	fmt.Println("信息:", rect.String())

	// 指针方法
	rect.Scale(2)
	fmt.Println("缩放后:", rect.String())

	// ----------------------------------------------------------
	// 构造函数模式
	// ----------------------------------------------------------
	fmt.Println("\n=== 构造函数模式 ===")

	// Go 没有构造函数，使用工厂函数代替
	config := NewConfig("localhost", 8080)
	fmt.Printf("默认配置: %+v\n", config)

	config2 := NewConfigWithOptions("0.0.0.0", 80, true, 30)
	fmt.Printf("自定义配置: %+v\n", config2)

	main02()
	main03()
}

// ============================================================
//                      结构体定义
// ============================================================

// ----------------------------------------------------------
// 基本结构体
// ----------------------------------------------------------
// 语法: type 结构体名 struct { 字段名 类型 }
// 【命名规则】
// - 大写开头：可导出（public）
// - 小写开头：包内私有（private）

// Person 人员结构体
type Person struct {
	Name string // 可导出字段
	Age  int
	City string
}

// ----------------------------------------------------------
// 嵌入结构体（组合）
// ----------------------------------------------------------
// Go 推崇组合而非继承
// 嵌入的字段和方法会被提升到外层

// Employee 员工结构体
type Employee struct {
	Person           // 嵌入 Person（匿名字段）
	Title      string
	Salary     float64
	Department string
}

// ----------------------------------------------------------
// 带标签的结构体
// ----------------------------------------------------------
// 标签是字符串，用反引号包裹
// 常用于 JSON、数据库映射、验证等

// User 用户结构体（带标签）
type User struct {
	ID       int    `json:"id"`
	Username string `json:"username"`
	Email    string `json:"email"`
	Password string `json:"-"` // - 表示 JSON 序列化时忽略
}

// ----------------------------------------------------------
// 带方法的结构体
// ----------------------------------------------------------

// Rectangle 矩形
type Rectangle struct {
	Width  float64
	Height float64
}

// Area 计算面积（值接收者）
func (r Rectangle) Area() float64 {
	return r.Width * r.Height
}

// Perimeter 计算周长
func (r Rectangle) Perimeter() float64 {
	return 2 * (r.Width + r.Height)
}

// String 字符串表示
func (r Rectangle) String() string {
	return fmt.Sprintf("Rectangle{%.1f x %.1f}", r.Width, r.Height)
}

// Scale 缩放（指针接收者，修改原值）
func (r *Rectangle) Scale(factor float64) {
	r.Width *= factor
	r.Height *= factor
}

// ----------------------------------------------------------
// 构造函数（工厂函数）
// ----------------------------------------------------------
// 【惯例】函数名为 New + 类型名
// 【惯例】返回指针（大型结构体）或值（小型结构体）

// Config 配置结构体
type Config struct {
	Host    string
	Port    int
	Debug   bool
	Timeout int
}

// NewConfig 创建默认配置
func NewConfig(host string, port int) *Config {
	return &Config{
		Host:    host,
		Port:    port,
		Debug:   false,  // 默认值
		Timeout: 10,     // 默认值
	}
}

// NewConfigWithOptions 创建自定义配置
func NewConfigWithOptions(host string, port int, debug bool, timeout int) *Config {
	return &Config{
		Host:    host,
		Port:    port,
		Debug:   debug,
		Timeout: timeout,
	}
}

// ============================================================
//                      重要注意事项
// ============================================================
//
// 1. 【字段命名】
//    - 大写开头：可导出（其他包可访问）
//    - 小写开头：包内私有
//
// 2. 【零值】
//    结构体的零值是所有字段的零值
//    数值=0, 字符串="", 布尔=false, 指针/切片/map=nil
//
// 3. 【值 vs 指针】
//    - 小结构体（几个字段）：传值
//    - 大结构体或需要修改：传指针
//    - 方法需要修改接收者：指针接收者
//
// 4. 【嵌入 vs 字段】
//    嵌入：Person（匿名，字段提升）
//    字段：person Person（命名，需要通过字段访问）
//
// 5. 【结构体比较】
//    - 所有字段可比较 → 结构体可比较
//    - 包含切片/map/函数 → 不可直接比较
//
// 6. 【标签使用】
//    - json: JSON 序列化
//    - db: 数据库映射
//    - validate: 验证规则
//    - 可通过 reflect 包读取
