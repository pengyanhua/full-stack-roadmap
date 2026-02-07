# main

::: info 文件信息
- 📄 原文件：`main.go`
- 🔤 语言：go
:::

## 完整代码

```go
package main

import (
	"fmt"

	// 导入本地包（使用模块路径）
	"example/packages/models"
	"example/packages/mymath"
	"example/packages/utils"
)

// ============================================================
//                      包与模块
// ============================================================
// Go 的代码组织：
// - 包（Package）：代码组织的基本单位，一个目录 = 一个包
// - 模块（Module）：包的集合，由 go.mod 定义

func main() {
	fmt.Println("=== Go 包与模块示例 ===")

	// ----------------------------------------------------------
	// 使用 mymath 包
	// ----------------------------------------------------------
	fmt.Println("\n=== mymath 包 ===")

	// 使用导出的常量
	fmt.Println("Pi =", mymath.Pi)
	fmt.Println("E =", mymath.E)

	// 使用导出的函数
	fmt.Println("Add(3, 5) =", mymath.Add(3, 5))
	fmt.Println("Multiply(4, 6) =", mymath.Multiply(4, 6))
	fmt.Println("Sqrt(16) =", mymath.Sqrt(16))
	fmt.Println("Power(2, 10) =", mymath.Power(2, 10))
	fmt.Println("Fibonacci(10) =", mymath.Fibonacci(10))

	// 使用导出的类型
	p1 := mymath.NewPoint(0, 0)
	p2 := mymath.NewPoint(3, 4)
	fmt.Printf("Distance from %+v to %+v = %.2f\n", p1, p2, p1.Distance(*p2))

	// 使用 Calculator
	calc := mymath.NewCalculator()
	calc.SetPrecision(8)
	fmt.Println("Calculator precision:", calc.GetPrecision())

	// 【注意】以下代码会编译错误（访问私有成员）：
	// fmt.Println(mymath.precision)  // 私有常量
	// fmt.Println(mymath.helper(5))  // 私有函数
	// fmt.Println(calc.precision)    // 私有字段

	// ----------------------------------------------------------
	// 使用 utils 包
	// ----------------------------------------------------------
	fmt.Println("\n=== utils 包 ===")

	// 字符串工具
	fmt.Println("Reverse(\"Hello\") =", utils.Reverse("Hello"))
	fmt.Println("Reverse(\"你好世界\") =", utils.Reverse("你好世界"))
	fmt.Println("IsPalindrome(\"level\") =", utils.IsPalindrome("level"))
	fmt.Println("CountWords(\"Hello World Go\") =", utils.CountWords("Hello World Go"))
	fmt.Println("Capitalize(\"hello\") =", utils.Capitalize("hello"))
	fmt.Println("Truncate =", utils.TruncateString("This is a long string", 15))

	// 切片工具（泛型函数）
	nums := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

	fmt.Println("Contains(nums, 5) =", utils.Contains(nums, 5))
	fmt.Println("Contains(nums, 11) =", utils.Contains(nums, 11))

	evens := utils.Filter(nums, func(n int) bool {
		return n%2 == 0
	})
	fmt.Println("Filter evens:", evens)

	doubled := utils.Map(nums, func(n int) int {
		return n * 2
	})
	fmt.Println("Map doubled:", doubled)

	sum := utils.Reduce(nums, 0, func(acc, n int) int {
		return acc + n
	})
	fmt.Println("Reduce sum:", sum)

	dupes := []int{1, 2, 2, 3, 3, 3, 4}
	fmt.Println("Unique:", utils.Unique(dupes))

	// ----------------------------------------------------------
	// 使用 models 包
	// ----------------------------------------------------------
	fmt.Println("\n=== models 包 ===")

	// 创建用户
	user := models.NewUser("johndoe", "john@example.com", "secret123")
	fmt.Println("用户:", user)

	// 验证密码
	fmt.Println("密码验证 (secret123):", user.CheckPassword("secret123"))
	fmt.Println("密码验证 (wrong):", user.CheckPassword("wrong"))

	// 【注意】无法直接访问 password 字段
	// fmt.Println(user.password)  // 编译错误

	// 创建产品
	product := models.NewProduct("iPhone", 999.99)
	product.AddStock(10)
	fmt.Printf("产品: %s, 价格: $%.2f, 库存: %d, 可用: %t\n",
		product.Name, product.Price, product.Stock, product.IsAvailable())

	// ----------------------------------------------------------
	// 包的导入方式
	// ----------------------------------------------------------
	fmt.Println("\n=== 包的导入方式 ===")

	/*
	// 1. 标准导入
	import "fmt"
	// 使用: fmt.Println()

	// 2. 别名导入
	import f "fmt"
	// 使用: f.Println()

	// 3. 点导入（不推荐，可能命名冲突）
	import . "fmt"
	// 使用: Println()  // 直接使用，无需包名

	// 4. 空白导入（只执行 init，不使用包）
	import _ "github.com/lib/pq"
	// 用于注册数据库驱动等副作用

	// 5. 多包导入
	import (
		"fmt"
		"os"
		"strings"
	)
	*/

	fmt.Println("参见代码中的注释")

	// ----------------------------------------------------------
	// 包的初始化顺序
	// ----------------------------------------------------------
	fmt.Println("\n=== 包初始化 ===")

	/*
	初始化顺序：
	1. 导入的包先初始化（递归）
	2. 包级别变量初始化（按声明顺序）
	3. init() 函数执行（可以有多个）

	特点：
	- 每个包只初始化一次
	- init() 无参数无返回值
	- 不能手动调用 init()
	*/

	fmt.Println("参见代码中的注释")
}

// init 在 main 之前自动执行
func init() {
	fmt.Println("[init] main 包初始化")
}

// 可以有多个 init
func init() {
	fmt.Println("[init] main 包第二个初始化函数")
}

// ============================================================
//                      重要注意事项
// ============================================================
//
// 1. 【包命名规则】
//    - 小写，无下划线或混合大小写
//    - 简短且有意义
//    - 避免与标准库冲突
//
// 2. 【可见性规则】
//    - 大写开头：可导出（public）
//    - 小写开头：私有（private）
//    - 适用于：函数、类型、变量、常量、字段、方法
//
// 3. 【包路径】
//    - 标准库：直接使用包名（如 "fmt"）
//    - 本地包：模块路径 + 相对路径
//    - 远程包：完整导入路径（如 "github.com/user/repo/pkg"）
//
// 4. 【模块管理】
//    go mod init <module-path>  // 初始化模块
//    go mod tidy                // 整理依赖
//    go get <package>           // 添加依赖
//
// 5. 【internal 包】
//    internal/ 目录下的包只能被同一模块的包导入
//    用于隐藏内部实现
//
// 6. 【vendor 目录】
//    存放项目依赖的副本
//    go mod vendor 生成
//
// 7. 【工作区（Go 1.18+）】
//    go work init              // 创建工作区
//    用于同时开发多个模块
```
