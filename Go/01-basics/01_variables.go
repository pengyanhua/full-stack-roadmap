package main

import "fmt"

var Name string = "Go语言"

func main() {
	// ============================================================
	//                      变量声明
	// ============================================================
	// Go 是静态类型语言，变量必须有明确的类型
	// 声明变量有三种主要方式

	fmt.Println("=== 变量声明 ===")

	// ----------------------------------------------------------
	// 方式1: var 声明（显式指定类型）
	// ----------------------------------------------------------
	// 语法: var 变量名 类型 = 值
	// 【适用场景】需要明确类型时，或声明包级别变量时
	var age int = 15
	fmt.Println("var 显式类型:", Name, age)

	// ----------------------------------------------------------
	// 方式2: var 声明（类型推断）
	// ----------------------------------------------------------
	// 语法: var 变量名 = 值
	// 编译器根据右边的值自动推断类型
	// 【注意】整数默认推断为 int，浮点数默认推断为 float64
	var version = "1.21" // 推断为 string
	var count = 100      // 推断为 int（不是 int32 或 int64）
	var price = 9.99     // 推断为 float64（不是 float32）
	fmt.Println("var 类型推断:", version, count, price)

	// ----------------------------------------------------------
	// 方式3: 短声明 :=（最常用）
	// ----------------------------------------------------------
	// 语法: 变量名 := 值
	// 【重要】只能在函数内部使用，不能用于包级别变量
	// 【重要】:= 是声明+赋值，= 是纯赋值
	language := "Golang"
	score := 95.5
	fmt.Println("短声明 :=:", language, score)

	// 【易错点】:= 左边必须至少有一个新变量
	// language := "Python"  // 错误！language 已存在
	// language = "Python"   // 正确，这是赋值

	// 【技巧】:= 可以部分重新声明
	language, newVar := "Python", "new" // language 赋值，newVar 新声明
	fmt.Println("部分重新声明:", language, newVar)

	// ----------------------------------------------------------
	// 方式4: 多变量声明
	// ----------------------------------------------------------
	// 一行声明多个变量，用逗号分隔

	// 同类型多变量
	var a, b, c int = 1, 2, 3
	fmt.Println("同类型多变量:", a, b, c)

	// 不同类型多变量（短声明）
	x, y, z := 10, "hello", true
	fmt.Println("不同类型多变量:", x, y, z)

	// 【实用】交换变量值
	a, b = b, a
	fmt.Println("交换后 a, b:", a, b)

	// 【实用】函数多返回值
	// value, err := someFunction()

	// ----------------------------------------------------------
	// 方式5: 声明块（分组声明）
	// ----------------------------------------------------------
	// 用 () 将多个声明组织在一起，代码更整洁
	// 【适用场景】包级别变量、相关变量分组
	var (
		width  = 100
		height = 200
		title  = "窗口"
	)
	fmt.Println("声明块:", width, height, title)

	// ============================================================
	//                      基本数据类型
	// ============================================================
	fmt.Println("\n=== 基本类型 ===")

	// ----------------------------------------------------------
	// 整型（有符号）
	// ----------------------------------------------------------
	// int8:   -128 到 127
	// int16:  -32768 到 32767
	// int32:  -21亿 到 21亿
	// int64:  -922亿亿 到 922亿亿
	// int:    根据平台，32位系统是int32，64位系统是int64
	//
	// 【建议】一般情况用 int，除非有特殊需求（如节省内存、与C交互）

	var i8 int8 = 127          // 最大值 127
	var i16 int16 = 32767      // 最大值 32767
	var i32 int32 = 2147483647 // 约 21 亿
	var i64 int64 = 9223372036854775807
	var i int = 42 // 【推荐】通用整型

	fmt.Printf("int8=%d, int16=%d, int32=%d\n", i8, i16, i32)
	fmt.Printf("int64=%d, int=%d\n", i64, i)

	// 【注意】不同整型之间不能直接运算，需要显式转换
	// result := i32 + i64        // 错误！
	// result := int64(i32) + i64 // 正确

	// ----------------------------------------------------------
	// 整型（无符号）
	// ----------------------------------------------------------
	// uint8:  0 到 255（别名 byte）
	// uint16: 0 到 65535
	// uint32: 0 到 42亿
	// uint64: 0 到 1844亿亿
	// uint:   根据平台

	var ui uint = 42
	var ui8 uint8 = 255
	fmt.Printf("uint=%d, uint8=%d\n", ui, ui8)

	// ----------------------------------------------------------
	// 浮点型
	// ----------------------------------------------------------
	// float32: 单精度，约 7 位有效数字
	// float64: 双精度，约 15 位有效数字
	//
	// 【建议】默认使用 float64，精度更高
	// 【注意】浮点数有精度问题，不要用 == 比较

	var f32 float32 = 3.14
	var f64 float64 = 3.141592653589793

	fmt.Printf("float32: %.7f\n", f32)
	fmt.Printf("float64: %.15f\n", f64)

	// 【警告】浮点数精度问题
	fmt.Printf("0.1 + 0.2 = %.20f (不等于 0.3)\n", 0.1+0.2)

	// ----------------------------------------------------------
	// 布尔型
	// ----------------------------------------------------------
	// 只有 true 和 false 两个值
	// 【注意】不能用 0 和 1 代替，不能隐式转换
	// 【注意】不能用 !0 或 !!x 这种写法

	var isTrue bool = true
	var isFalse bool = false
	fmt.Printf("bool: %t, %t\n", isTrue, isFalse)

	// if 1 { }  // 错误！Go 不允许

	// ----------------------------------------------------------
	// 字符串
	// ----------------------------------------------------------
	// 字符串是不可变的字节序列
	// 使用双引号 "" 或反引号 ``（原始字符串）

	var str1 string = "Hello, 世界"       // 普通字符串，支持转义
	var str2 string = `C:\path\to\file` // 原始字符串，不转义
	var str3 string = `多行
字符串`

	fmt.Println("普通字符串:", str1)
	fmt.Println("原始字符串:", str2)
	fmt.Println("多行字符串:", str3)

	// 【重要】字符串操作
	fmt.Println("长度(字节数):", len(str1))     // 13（中文3字节）
	fmt.Println("字符数:", len([]rune(str1))) // 9
	fmt.Println("拼接:", "Hello"+" "+"World")
	fmt.Println("索引(字节):", str1[0]) // 72 ('H')

	// 【注意】字符串不可变
	// str1[0] = 'h'  // 错误！不能修改

	// ----------------------------------------------------------
	// byte 和 rune
	// ----------------------------------------------------------
	// byte: uint8 的别名，表示一个字节（ASCII字符）
	// rune: int32 的别名，表示一个 Unicode 码点（任意字符）
	//
	// 【重要】单引号 '' 表示字符，双引号 "" 表示字符串

	var b1 byte = 'A'   // ASCII 字符
	var b2 byte = 65    // 同上，'A' 的 ASCII 值
	var r1 rune = '中'   // Unicode 字符
	var r2 rune = 20013 // 同上，'中' 的 Unicode 码点

	fmt.Printf("byte: %c = %d\n", b1, b1)
	fmt.Printf("byte: %c = %d\n", b2, b2)
	fmt.Printf("rune: %c = %d (U+%04X)\n", r1, r1, r1)
	fmt.Printf("rune: %c = %d\n", r2, r2)

	// 【技巧】遍历字符串的正确方式
	fmt.Print("遍历字符: ")
	for _, char := range "Go语言" {
		fmt.Printf("%c ", char)
	}
	fmt.Println()

	// ============================================================
	//                      常量
	// ============================================================
	fmt.Println("\n=== 常量 ===")

	// ----------------------------------------------------------
	// 常量声明
	// ----------------------------------------------------------
	// 使用 const 关键字，值在编译时确定，不能修改
	// 【注意】常量不能用 := 声明
	// 【注意】常量可以是无类型的，有更大的灵活性

	const pi = 3.14159        // 无类型常量
	const e float64 = 2.71828 // 有类型常量

	fmt.Println("pi:", pi)
	fmt.Println("e:", e)

	// 常量块
	const (
		StatusOK       = 200
		StatusNotFound = 404
		StatusError    = 500
	)
	fmt.Println("HTTP状态码:", StatusOK, StatusNotFound, StatusError)

	// ----------------------------------------------------------
	// iota 枚举生成器
	// ----------------------------------------------------------
	// iota 在 const 块中从 0 开始，每行自动 +1
	// 【技巧】用于创建枚举值

	const (
		Sunday    = iota // 0
		Monday           // 1
		Tuesday          // 2
		Wednesday        // 3
		Thursday         // 4
		Friday           // 5
		Saturday         // 6
	)
	fmt.Println("星期:", Sunday, Monday, Tuesday)

	// 【技巧】iota 表达式
	const (
		_  = iota             // 0，用 _ 丢弃
		KB = 1 << (10 * iota) // 1 << 10 = 1024
		MB                    // 1 << 20 = 1048576
		GB                    // 1 << 30
		TB                    // 1 << 40
	)
	fmt.Printf("KB=%d, MB=%d, GB=%d\n", KB, MB, GB)

	// 【技巧】跳过值
	const (
		A = iota // 0
		B        // 1
		_        // 2（跳过）
		D        // 3
	)
	fmt.Println("跳过值:", A, B, D)

	// ============================================================
	//                      零值（Zero Value）
	// ============================================================
	fmt.Println("\n=== 零值 ===")

	// Go 的变量声明后如果没有初始化，会有默认的零值
	// 这是 Go 的重要特性，避免了未初始化变量的问题

	var zeroInt int         // 0
	var zeroFloat float64   // 0.0
	var zeroBool bool       // false
	var zeroString string   // ""（空字符串）
	var zeroPointer *int    // nil
	var zeroSlice []int     // nil
	var zeroMap map[int]int // nil
	var zeroFunc func()     // nil

	fmt.Printf("int 零值: %d\n", zeroInt)
	fmt.Printf("float64 零值: %g\n", zeroFloat)
	fmt.Printf("bool 零值: %t\n", zeroBool)
	fmt.Printf("string 零值: %q\n", zeroString)
	fmt.Printf("指针零值: %v\n", zeroPointer)
	fmt.Printf("切片零值: %v (nil=%t)\n", zeroSlice, zeroSlice == nil)
	fmt.Printf("map零值: %v (nil=%t)\n", zeroMap, zeroMap == nil)
	fmt.Printf("函数零值: %v\n", zeroFunc)

	// 【重要】零值表
	// 数值类型:  0
	// 布尔类型:  false
	// 字符串:    ""
	// 指针/切片/map/channel/接口/函数: nil

	// ============================================================
	//                      类型转换
	// ============================================================
	fmt.Println("\n=== 类型转换 ===")

	// Go 没有隐式类型转换，必须显式转换
	// 语法: 目标类型(值)

	var intVal int = 42
	var floatVal float64 = float64(intVal) // int -> float64
	var intVal2 int = int(floatVal)        // float64 -> int（截断小数）

	fmt.Printf("int->float64: %d -> %f\n", intVal, floatVal)
	fmt.Printf("float64->int: %f -> %d\n", floatVal, intVal2)

	// 【注意】浮点数转整数会截断小数，不是四舍五入
	var pi64 float64 = 3.9
	fmt.Printf("截断示例: %.1f -> %d\n", pi64, int(pi64)) // 3

	// 字符串转换
	var num int = 65
	var char string = string(num) // 转为 Unicode 字符 "A"
	fmt.Printf("int->string: %d -> %s\n", num, char)

	// 【注意】数字和字符串互转要用 strconv 包
	// strconv.Itoa(42)      -> "42"
	// strconv.Atoi("42")    -> 42, error

	// ============================================================
	//                      类型别名与自定义类型
	// ============================================================
	fmt.Println("\n=== 类型别名与自定义类型 ===")

	// 类型别名（Type Alias）: type 新名 = 原类型
	// 两者完全相同，可以互换使用
	type MyByte = byte
	var mb MyByte = 'X'
	fmt.Printf("类型别名: %c (类型完全相同)\n", mb)

	// 自定义类型（Type Definition）: type 新类型 原类型
	// 创建全新类型，不能直接与原类型互换
	type Age int
	var myAge Age = 25
	// var otherAge int = myAge  // 错误！类型不同
	var otherAge int = int(myAge) // 需要显式转换
	fmt.Printf("自定义类型: %d (需要转换: %d)\n", myAge, otherAge)

	main02()
	main03()
	main04()
}
