# maps.go

::: info 文件信息
- 📄 原文件：`04_maps.go`
- 🔤 语言：go
:::

## 完整代码

```go
package main

import "fmt"

func main04() {
	fmt.Println("\n ====================  04_maps ====================")
	// ========== 创建 map ==========
	fmt.Println("=== 创建 map ===")

	// 使用 make 创建
	scores := make(map[string]int)
	scores["张三"] = 90
	scores["李四"] = 85
	scores["王五"] = 78
	fmt.Println("make 创建:", scores)

	// 字面量创建
	ages := map[string]int{
		"Alice": 25,
		"Bob":   30,
		"Carol": 28,
	}
	fmt.Println("字面量创建:", ages)

	// 空 map
	empty := map[string]int{}
	fmt.Println("空 map:", empty)

	// ========== 访问和修改 ==========
	fmt.Println("\n=== 访问和修改 ===")

	fmt.Println("张三的分数:", scores["张三"])

	// 修改
	scores["张三"] = 95
	fmt.Println("修改后:", scores["张三"])

	// 添加新键
	scores["赵六"] = 88
	fmt.Println("添加后:", scores)

	// ========== 检查键是否存在 ==========
	fmt.Println("\n=== 检查键是否存在 ===")

	// 不存在的键返回零值
	fmt.Println("不存在的键:", scores["不存在"])

	// 使用 ok 模式检查
	value, ok := scores["张三"]
	if ok {
		fmt.Println("张三存在，分数:", value)
	}

	if _, exists := scores["田七"]; !exists {
		fmt.Println("田七不存在")
	}

	// ========== 删除 ==========
	fmt.Println("\n=== 删除 ===")

	fmt.Println("删除前:", scores)
	delete(scores, "李四")
	fmt.Println("删除后:", scores)

	// 删除不存在的键不会报错
	delete(scores, "不存在的键")

	// ========== 遍历 ==========
	fmt.Println("\n=== 遍历 ===")

	fmt.Println("遍历 key-value:")
	for name, score := range scores {
		fmt.Printf("  %s: %d\n", name, score)
	}

	fmt.Println("只遍历 key:")
	for name := range scores {
		fmt.Println(" ", name)
	}

	// ========== 长度 ==========
	fmt.Println("\n=== 长度 ===")
	fmt.Println("map 长度:", len(scores))

	// ========== map 是引用类型 ==========
	fmt.Println("\n=== map 是引用类型 ===")

	original := map[string]int{"a": 1, "b": 2}
	reference := original
	reference["a"] = 100
	fmt.Println("原 map 也被修改:", original)

	// ========== nil map ==========
	fmt.Println("\n=== nil map ===")

	var nilMap map[string]int
	fmt.Println("nil map:", nilMap, "nil =", nilMap == nil)
	// nilMap["key"] = 1  // 这会 panic！

	// ========== 复杂类型的 map ==========
	fmt.Println("\n=== 复杂类型的 map ===")

	// map 的值是切片
	groups := map[string][]string{
		"水果": {"苹果", "香蕉", "橙子"},
		"蔬菜": {"白菜", "萝卜", "土豆"},
	}
	fmt.Println("值为切片的 map:")
	for category, items := range groups {
		fmt.Printf("  %s: %v\n", category, items)
	}

	// 嵌套 map
	students := map[string]map[string]int{
		"张三": {"语文": 90, "数学": 85},
		"李四": {"语文": 88, "数学": 92},
	}
	fmt.Println("嵌套 map:")
	for name, scores := range students {
		fmt.Printf("  %s: %v\n", name, scores)
	}

	// ========== 使用 struct 作为 key ==========
	fmt.Println("\n=== struct 作为 key ===")

	type Point struct {
		X, Y int
	}

	points := map[Point]string{
		{0, 0}: "原点",
		{1, 0}: "X轴上",
		{0, 1}: "Y轴上",
	}
	fmt.Println("struct 为 key:", points)
	fmt.Println("原点:", points[Point{0, 0}])
}
```
