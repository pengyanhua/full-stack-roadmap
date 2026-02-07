# slices.go

::: info 文件信息
- 📄 原文件：`03_slices.go`
- 🔤 语言：go
:::

## 完整代码

```go
package main

import "fmt"

func main03() {
	fmt.Println("\n ====================  03_slices ====================")
	// ========== 数组 ==========
	fmt.Println("=== 数组 ===")

	// 声明数组（固定长度）
	var arr1 [5]int
	fmt.Println("零值数组:", arr1)

	// 初始化数组
	arr2 := [5]int{1, 2, 3, 4, 5}
	fmt.Println("初始化数组:", arr2)

	// 部分初始化
	arr3 := [5]int{1, 2} // 其余为零值
	fmt.Println("部分初始化:", arr3)

	// 自动推断长度
	arr4 := []string{"Go", "Python", "Java"}
	fmt.Println("自动长度:", arr4, "长度:", len(arr4))

	// 指定索引初始化
	arr5 := [5]int{1: 10, 3: 30}
	fmt.Println("指定索引:", arr5)

	// 访问和修改
	arr2[0] = 100
	fmt.Println("修改后:", arr2)
	fmt.Println("第一个元素:", arr2[0])

	// ========== 切片 ==========
	fmt.Println("\n=== 切片 ===")

	// 从数组创建切片
	arr := [5]int{1, 2, 3, 4, 5}
	slice1 := arr[1:4] // [2, 3, 4]
	fmt.Println("从数组切片:", slice1)

	// 切片字面量
	slice2 := []int{10, 20, 30, 40, 50}
	fmt.Println("切片字面量:", slice2)

	// make 创建切片
	slice3 := make([]int, 3)     // 长度3，容量3
	slice4 := make([]int, 3, 10) // 长度3，容量10
	fmt.Printf("make 切片: %v, len=%d, cap=%d\n", slice3, len(slice3), cap(slice3))
	fmt.Printf("make 切片: %v, len=%d, cap=%d\n", slice4, len(slice4), cap(slice4))

	// 切片操作
	s := []int{0, 1, 2, 3, 4, 5}
	fmt.Println("原切片:", s)
	fmt.Println("s[2:4]:", s[2:4]) // [2, 3]
	fmt.Println("s[:3]:", s[:3])   // [0, 1, 2]
	fmt.Println("s[3:]:", s[3:])   // [3, 4, 5]
	fmt.Println("s[:]:", s[:])     // 全部

	// ========== append ==========
	fmt.Println("\n=== append ===")

	nums := []int{1, 2, 3}
	fmt.Println("原切片:", nums)

	// 追加元素
	nums = append(nums, 4)
	fmt.Println("追加一个:", nums)

	// 追加多个
	nums = append(nums, 5, 6, 7)
	fmt.Println("追加多个:", nums)

	// 追加另一个切片
	more := []int{8, 9}
	nums = append(nums, more...) // 注意 ... 展开
	fmt.Println("追加切片:", nums)

	// ========== copy ==========
	fmt.Println("\n=== copy ===")

	src := []int{1, 2, 3, 4, 5}
	dst := make([]int, 3)
	copied := copy(dst, src)
	fmt.Printf("复制了 %d 个元素: %v\n", copied, dst)

	// ========== 切片是引用类型 ==========
	fmt.Println("\n=== 切片是引用类型 ===")

	original := []int{1, 2, 3}
	reference := original
	reference[0] = 100
	fmt.Println("原切片也被修改:", original)

	// ========== 二维切片 ==========
	fmt.Println("\n=== 二维切片 ===")

	matrix := [][]int{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	fmt.Println("二维切片:")
	for i, row := range matrix {
		fmt.Printf("  行%d: %v\n", i, row)
	}

	// ========== nil 切片 vs 空切片 ==========
	fmt.Println("\n=== nil 切片 vs 空切片 ===")

	var nilSlice []int
	emptySlice := []int{}
	makeSlice := make([]int, 0)

	fmt.Printf("nil 切片: %v, len=%d, nil=%t\n", nilSlice, len(nilSlice), nilSlice == nil)
	fmt.Printf("空切片: %v, len=%d, nil=%t\n", emptySlice, len(emptySlice), emptySlice == nil)
	fmt.Printf("make 空切片: %v, len=%d, nil=%t\n", makeSlice, len(makeSlice), makeSlice == nil)
}
```
