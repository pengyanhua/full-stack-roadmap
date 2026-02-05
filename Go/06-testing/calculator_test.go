package main

import (
	"fmt"
	"testing"
)

// ============================================================
//                      Go 测试基础
// ============================================================
// 测试文件命名: xxx_test.go
// 测试函数命名: TestXxx（必须以 Test 开头，后面首字母大写）
// 运行测试: go test [-v] [-run=pattern]

// ----------------------------------------------------------
// 基本测试
// ----------------------------------------------------------

// TestAbs 测试 Abs 函数
func TestAbs(t *testing.T) {
	// 测试正数
	result := Abs(5)
	if result != 5 {
		t.Errorf("Abs(5) = %d; want 5", result)
	}

	// 测试负数
	result = Abs(-5)
	if result != 5 {
		t.Errorf("Abs(-5) = %d; want 5", result)
	}

	// 测试零
	result = Abs(0)
	if result != 0 {
		t.Errorf("Abs(0) = %d; want 0", result)
	}
}

// TestMax 测试 Max 函数
func TestMax(t *testing.T) {
	if Max(3, 5) != 5 {
		t.Error("Max(3, 5) should be 5")
	}

	if Max(5, 3) != 5 {
		t.Error("Max(5, 3) should be 5")
	}

	if Max(3, 3) != 3 {
		t.Error("Max(3, 3) should be 3")
	}
}

// ----------------------------------------------------------
// 表格驱动测试（推荐方式）
// ----------------------------------------------------------

// TestFibonacci 使用表格驱动测试
func TestFibonacci(t *testing.T) {
	// 定义测试用例
	tests := []struct {
		name     string // 测试用例名称
		input    int    // 输入
		expected int    // 期望输出
	}{
		{"fib(0)", 0, 0},
		{"fib(1)", 1, 1},
		{"fib(2)", 2, 1},
		{"fib(5)", 5, 5},
		{"fib(10)", 10, 55},
	}

	// 遍历测试用例
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Fibonacci(tt.input)
			if result != tt.expected {
				t.Errorf("Fibonacci(%d) = %d; want %d", tt.input, result, tt.expected)
			}
		})
	}
}

// TestIsPrime 表格驱动测试
func TestIsPrime(t *testing.T) {
	tests := []struct {
		n        int
		expected bool
	}{
		{-1, false},
		{0, false},
		{1, false},
		{2, true},
		{3, true},
		{4, false},
		{5, true},
		{10, false},
		{17, true},
		{100, false},
	}

	for _, tt := range tests {
		// 使用 %d 作为子测试名称
		t.Run(string(rune(tt.n)), func(t *testing.T) {
			result := IsPrime(tt.n)
			if result != tt.expected {
				t.Errorf("IsPrime(%d) = %v; want %v", tt.n, result, tt.expected)
			}
		})
	}
}

// ----------------------------------------------------------
// 测试结构体方法
// ----------------------------------------------------------

func TestCalculator_Add(t *testing.T) {
	calc := NewCalculator()

	tests := []struct {
		a, b     float64
		expected float64
	}{
		{1, 2, 3},
		{-1, 1, 0},
		{0, 0, 0},
		{1.5, 2.5, 4},
	}

	for _, tt := range tests {
		result := calc.Add(tt.a, tt.b)
		if result != tt.expected {
			t.Errorf("Add(%v, %v) = %v; want %v", tt.a, tt.b, result, tt.expected)
		}
	}
}

func TestCalculator_Divide(t *testing.T) {
	calc := NewCalculator()

	// 正常除法
	result, err := calc.Divide(10, 2)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if result != 5 {
		t.Errorf("Divide(10, 2) = %v; want 5", result)
	}

	// 除以零
	_, err = calc.Divide(10, 0)
	if err == nil {
		t.Error("expected error for division by zero")
	}
}

// ----------------------------------------------------------
// 子测试
// ----------------------------------------------------------

func TestReverse(t *testing.T) {
	t.Run("ASCII", func(t *testing.T) {
		result := Reverse("hello")
		if result != "olleh" {
			t.Errorf("got %s; want olleh", result)
		}
	})

	t.Run("Unicode", func(t *testing.T) {
		result := Reverse("你好")
		if result != "好你" {
			t.Errorf("got %s; want 好你", result)
		}
	})

	t.Run("Empty", func(t *testing.T) {
		result := Reverse("")
		if result != "" {
			t.Errorf("got %s; want empty", result)
		}
	})
}

// ----------------------------------------------------------
// 跳过测试
// ----------------------------------------------------------

func TestSkipExample(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	// 耗时的测试...
	t.Log("Running full test")
}

// ----------------------------------------------------------
// 并行测试
// ----------------------------------------------------------

func TestParallel(t *testing.T) {
	t.Parallel() // 标记为可并行

	tests := []struct {
		name     string
		input    int
		expected int
	}{
		{"abs positive", 5, 5},
		{"abs negative", -5, 5},
		{"abs zero", 0, 0},
	}

	for _, tt := range tests {
		tt := tt // 捕获循环变量
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel() // 子测试也可以并行
			result := Abs(tt.input)
			if result != tt.expected {
				t.Errorf("Abs(%d) = %d; want %d", tt.input, result, tt.expected)
			}
		})
	}
}

// ----------------------------------------------------------
// Setup 和 Teardown
// ----------------------------------------------------------

func TestWithSetup(t *testing.T) {
	// Setup
	calc := NewCalculator()
	t.Log("Setup: Calculator created")

	// Cleanup（defer 确保执行）
	t.Cleanup(func() {
		t.Log("Cleanup: Resources released")
	})

	// 测试
	result := calc.Add(1, 2)
	if result != 3 {
		t.Errorf("Add(1, 2) = %v; want 3", result)
	}
}

// TestMain 可以用于全局 setup/teardown
// func TestMain(m *testing.M) {
//     // 全局 setup
//     fmt.Println("Global Setup")
//
//     code := m.Run() // 运行所有测试
//
//     // 全局 teardown
//     fmt.Println("Global Teardown")
//
//     os.Exit(code)
// }

// ----------------------------------------------------------
// 辅助函数
// ----------------------------------------------------------

// assertEqual 辅助断言函数
func assertEqual(t *testing.T, got, want interface{}) {
	t.Helper() // 标记为辅助函数，错误时报告调用者位置
	if got != want {
		t.Errorf("got %v; want %v", got, want)
	}
}

func TestWithHelper(t *testing.T) {
	assertEqual(t, Abs(-5), 5)
	assertEqual(t, Max(3, 5), 5)
}

// ============================================================
//                      基准测试 (Benchmark)
// ============================================================
// 函数命名: BenchmarkXxx
// 运行: go test -bench=. [-benchmem] [-benchtime=3s]

func BenchmarkFibonacci(b *testing.B) {
	// b.N 由测试框架自动调整
	for i := 0; i < b.N; i++ {
		Fibonacci(20)
	}
}

func BenchmarkFibonacciIterative(b *testing.B) {
	for i := 0; i < b.N; i++ {
		FibonacciIterative(20)
	}
}

// 基准测试子测试
func BenchmarkFibonacciComparison(b *testing.B) {
	b.Run("Recursive", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			Fibonacci(15)
		}
	})

	b.Run("Iterative", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			FibonacciIterative(15)
		}
	})
}

// 基准测试不同输入
func BenchmarkIsPrime(b *testing.B) {
	inputs := []int{17, 997, 7919}

	for _, n := range inputs {
		b.Run(string(rune(n)), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				IsPrime(n)
			}
		})
	}
}

// ============================================================
//                      示例函数 (Example)
// ============================================================
// 函数命名: ExampleXxx
// 用于文档和可运行示例
// Output: 注释标记期望输出

func ExampleGreet() {
	fmt.Println(Greet("World"))
	// Output: Hello, World!
}

func ExampleSum() {
	fmt.Println(Sum(1, 2, 3))
	fmt.Println(Sum(10, 20))
	// Output:
	// 6
	// 30
}

func ExampleReverse() {
	fmt.Println(Reverse("hello"))
	fmt.Println(Reverse("你好"))
	// Output:
	// olleh
	// 好你
}

// ============================================================
//                      重要注意事项
// ============================================================
//
// 1. 【测试命令】
//    go test              # 运行当前包测试
//    go test -v           # 详细输出
//    go test ./...        # 运行所有包测试
//    go test -run=Pattern # 运行匹配的测试
//
// 2. 【基准测试命令】
//    go test -bench=.           # 运行所有基准测试
//    go test -bench=. -benchmem # 显示内存分配
//    go test -bench=. -count=5  # 运行多次
//
// 3. 【覆盖率】
//    go test -cover                    # 显示覆盖率
//    go test -coverprofile=c.out       # 生成覆盖率文件
//    go tool cover -html=c.out         # HTML 报告
//
// 4. 【表格驱动测试】
//    推荐方式，易于维护和扩展
//    使用 t.Run 创建子测试
//
// 5. 【t.Helper()】
//    标记辅助函数，错误报告时显示调用者位置
//
// 6. 【t.Parallel()】
//    标记测试可并行运行
//    注意循环变量捕获
//
// 7. 【testing.Short()】
//    配合 -short 标志跳过耗时测试
