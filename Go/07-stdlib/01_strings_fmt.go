package main

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// ============================================================
//                      strings 和 fmt 包
// ============================================================

func main() {
	fmt.Println("=== strings 包 ===")

	// ----------------------------------------------------------
	// 字符串查找
	// ----------------------------------------------------------
	s := "Hello, World!"

	fmt.Println("Contains 'World':", strings.Contains(s, "World"))
	fmt.Println("HasPrefix 'Hello':", strings.HasPrefix(s, "Hello"))
	fmt.Println("HasSuffix '!':", strings.HasSuffix(s, "!"))
	fmt.Println("Index 'o':", strings.Index(s, "o"))
	fmt.Println("LastIndex 'o':", strings.LastIndex(s, "o"))
	fmt.Println("Count 'l':", strings.Count(s, "l"))

	// ----------------------------------------------------------
	// 字符串转换
	// ----------------------------------------------------------
	fmt.Println("\n=== 字符串转换 ===")

	fmt.Println("ToUpper:", strings.ToUpper(s))
	fmt.Println("ToLower:", strings.ToLower(s))
	fmt.Println("Title:", strings.Title("hello world"))
	fmt.Println("ToTitle:", strings.ToTitle("hello"))

	// ----------------------------------------------------------
	// 字符串修剪
	// ----------------------------------------------------------
	fmt.Println("\n=== 字符串修剪 ===")

	padded := "  Hello, World!  "
	fmt.Printf("原始: '%s'\n", padded)
	fmt.Printf("TrimSpace: '%s'\n", strings.TrimSpace(padded))
	fmt.Printf("TrimLeft: '%s'\n", strings.TrimLeft(padded, " "))
	fmt.Printf("TrimRight: '%s'\n", strings.TrimRight(padded, " "))
	fmt.Printf("TrimPrefix: '%s'\n", strings.TrimPrefix("Hello, World!", "Hello, "))
	fmt.Printf("TrimSuffix: '%s'\n", strings.TrimSuffix("Hello, World!", "!"))

	// ----------------------------------------------------------
	// 字符串分割与连接
	// ----------------------------------------------------------
	fmt.Println("\n=== 分割与连接 ===")

	csv := "apple,banana,cherry"
	parts := strings.Split(csv, ",")
	fmt.Println("Split:", parts)

	words := "  hello   world   go  "
	fmt.Println("Fields:", strings.Fields(words))

	joined := strings.Join(parts, " | ")
	fmt.Println("Join:", joined)

	// ----------------------------------------------------------
	// 字符串替换
	// ----------------------------------------------------------
	fmt.Println("\n=== 字符串替换 ===")

	fmt.Println("Replace:", strings.Replace("aaa", "a", "b", 2))      // "bba"
	fmt.Println("ReplaceAll:", strings.ReplaceAll("aaa", "a", "b"))   // "bbb"

	// 使用 Replacer（高效的多重替换）
	replacer := strings.NewReplacer("<", "&lt;", ">", "&gt;")
	html := replacer.Replace("<div>Hello</div>")
	fmt.Println("Replacer:", html)

	// ----------------------------------------------------------
	// 字符串构建（高效拼接）
	// ----------------------------------------------------------
	fmt.Println("\n=== strings.Builder ===")

	var builder strings.Builder
	builder.WriteString("Hello")
	builder.WriteString(", ")
	builder.WriteString("World")
	builder.WriteByte('!')
	fmt.Println("Builder:", builder.String())

	// ----------------------------------------------------------
	// 字符串重复
	// ----------------------------------------------------------
	fmt.Println("\n=== 其他操作 ===")

	fmt.Println("Repeat:", strings.Repeat("Go ", 3))
	fmt.Println("EqualFold:", strings.EqualFold("Go", "go")) // 不区分大小写比较

	// ----------------------------------------------------------
	// unicode/utf8 包
	// ----------------------------------------------------------
	fmt.Println("\n=== unicode/utf8 ===")

	str := "Hello, 世界"
	fmt.Println("字符串:", str)
	fmt.Println("len() 字节数:", len(str))
	fmt.Println("RuneCountInString 字符数:", utf8.RuneCountInString(str))

	// 遍历 UTF-8 字符串
	fmt.Print("逐字符: ")
	for i, r := range str {
		fmt.Printf("[%d]%c ", i, r)
	}
	fmt.Println()

	// ============================================================
	//                      fmt 格式化
	// ============================================================
	fmt.Println("\n=== fmt 格式化 ===")

	// ----------------------------------------------------------
	// 通用格式
	// ----------------------------------------------------------
	type Point struct{ X, Y int }
	p := Point{10, 20}

	fmt.Printf("%%v  默认格式: %v\n", p)
	fmt.Printf("%%+v 带字段名: %+v\n", p)
	fmt.Printf("%%#v Go 语法: %#v\n", p)
	fmt.Printf("%%T  类型: %T\n", p)

	// ----------------------------------------------------------
	// 整数格式
	// ----------------------------------------------------------
	fmt.Println("\n=== 整数格式 ===")
	n := 42

	fmt.Printf("%%d 十进制: %d\n", n)
	fmt.Printf("%%b 二进制: %b\n", n)
	fmt.Printf("%%o 八进制: %o\n", n)
	fmt.Printf("%%x 十六进制: %x\n", n)
	fmt.Printf("%%X 十六进制大写: %X\n", n)
	fmt.Printf("%%c 字符: %c\n", 65)

	// ----------------------------------------------------------
	// 浮点格式
	// ----------------------------------------------------------
	fmt.Println("\n=== 浮点格式 ===")
	f := 3.141592653589793

	fmt.Printf("%%f 默认: %f\n", f)
	fmt.Printf("%%.2f 2位小数: %.2f\n", f)
	fmt.Printf("%%e 科学计数: %e\n", f)
	fmt.Printf("%%g 紧凑格式: %g\n", f)

	// ----------------------------------------------------------
	// 字符串格式
	// ----------------------------------------------------------
	fmt.Println("\n=== 字符串格式 ===")
	str2 := "Hello"

	fmt.Printf("%%s 默认: %s\n", str2)
	fmt.Printf("%%q 带引号: %q\n", str2)
	fmt.Printf("%%10s 右对齐: '%10s'\n", str2)
	fmt.Printf("%%-10s 左对齐: '%-10s'\n", str2)

	// ----------------------------------------------------------
	// 宽度和精度
	// ----------------------------------------------------------
	fmt.Println("\n=== 宽度和精度 ===")

	fmt.Printf("%%5d 宽度5: '%5d'\n", 42)
	fmt.Printf("%%05d 补零: '%05d'\n", 42)
	fmt.Printf("%%-5d 左对齐: '%-5d'\n", 42)
	fmt.Printf("%%+d 带符号: '%+d'\n", 42)

	// ----------------------------------------------------------
	// Sprintf（返回字符串）
	// ----------------------------------------------------------
	fmt.Println("\n=== Sprintf ===")

	result := fmt.Sprintf("Name: %s, Age: %d", "Alice", 25)
	fmt.Println(result)

	// ----------------------------------------------------------
	// Scanf（输入）
	// ----------------------------------------------------------
	fmt.Println("\n=== 输入函数（注释中）===")
	/*
	var name string
	var age int
	fmt.Print("Enter name and age: ")
	fmt.Scanf("%s %d", &name, &age)

	// 或逐行读取
	fmt.Scanln(&name)
	*/

	main02()
	main03()
	main04()
}
