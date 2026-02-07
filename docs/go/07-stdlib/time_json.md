# time json.go

::: info 文件信息
- 📄 原文件：`03_time_json.go`
- 🔤 语言：go
:::

## 完整代码

```go
package main

import (
	"encoding/json"
	"fmt"
	"time"
)

// ============================================================
//                      time 和 json 包
// ============================================================

func main03() {
	fmt.Println("\n==================== 03_time_json ====================")

	// ----------------------------------------------------------
	// 获取时间
	// ----------------------------------------------------------
	fmt.Println("=== time 获取时间 ===")

	now := time.Now()
	fmt.Println("当前时间:", now)
	fmt.Println("年:", now.Year())
	fmt.Println("月:", now.Month())
	fmt.Println("日:", now.Day())
	fmt.Println("时:", now.Hour())
	fmt.Println("分:", now.Minute())
	fmt.Println("秒:", now.Second())
	fmt.Println("纳秒:", now.Nanosecond())
	fmt.Println("星期:", now.Weekday())
	fmt.Println("年中第几天:", now.YearDay())

	// 时间戳
	fmt.Println("Unix 时间戳:", now.Unix())
	fmt.Println("UnixMilli:", now.UnixMilli())

	// ----------------------------------------------------------
	// 时间格式化
	// ----------------------------------------------------------
	fmt.Println("\n=== 时间格式化 ===")

	// 【重要】Go 使用参考时间 "Mon Jan 2 15:04:05 MST 2006"
	// 记忆：1 2 3 4 5 6 7（月 日 时 分 秒 年 时区）

	fmt.Println("默认:", now.Format(time.RFC3339))
	fmt.Println("日期:", now.Format("2006-01-02"))
	fmt.Println("时间:", now.Format("15:04:05"))
	fmt.Println("完整:", now.Format("2006-01-02 15:04:05"))
	fmt.Println("中文:", now.Format("2006年01月02日 15时04分05秒"))
	fmt.Println("12小时制:", now.Format("03:04:05 PM"))

	// 常用格式常量
	fmt.Println("RFC822:", now.Format(time.RFC822))
	fmt.Println("Kitchen:", now.Format(time.Kitchen))

	// ----------------------------------------------------------
	// 时间解析
	// ----------------------------------------------------------
	fmt.Println("\n=== 时间解析 ===")

	timeStr := "2024-12-25 10:30:00"
	parsed, err := time.Parse("2006-01-02 15:04:05", timeStr)
	if err != nil {
		fmt.Println("解析错误:", err)
	} else {
		fmt.Println("解析结果:", parsed)
	}

	// 带时区解析
	loc, _ := time.LoadLocation("Asia/Shanghai")
	parsedLocal, _ := time.ParseInLocation("2006-01-02 15:04:05", timeStr, loc)
	fmt.Println("上海时区:", parsedLocal)

	// ----------------------------------------------------------
	// 时间计算
	// ----------------------------------------------------------
	fmt.Println("\n=== 时间计算 ===")

	// 增加时间
	future := now.Add(24 * time.Hour)
	fmt.Println("明天:", future.Format("2006-01-02"))

	past := now.Add(-7 * 24 * time.Hour)
	fmt.Println("上周:", past.Format("2006-01-02"))

	// 时间差
	duration := future.Sub(now)
	fmt.Println("时间差:", duration)

	// 比较时间
	fmt.Println("now.Before(future):", now.Before(future))
	fmt.Println("now.After(past):", now.After(past))
	fmt.Println("now.Equal(now):", now.Equal(now))

	// ----------------------------------------------------------
	// Duration
	// ----------------------------------------------------------
	fmt.Println("\n=== Duration ===")

	d := 2*time.Hour + 30*time.Minute + 45*time.Second
	fmt.Println("Duration:", d)
	fmt.Println("小时:", d.Hours())
	fmt.Println("分钟:", d.Minutes())
	fmt.Println("秒:", d.Seconds())

	// 解析 Duration
	d2, _ := time.ParseDuration("1h30m")
	fmt.Println("解析 1h30m:", d2)

	// ----------------------------------------------------------
	// 定时器和 Ticker
	// ----------------------------------------------------------
	fmt.Println("\n=== Timer 和 Sleep ===")

	// Sleep
	fmt.Println("休眠 100ms...")
	start := time.Now()
	time.Sleep(100 * time.Millisecond)
	fmt.Println("实际休眠:", time.Since(start))

	// Timer（在 goroutine 中使用）
	// timer := time.NewTimer(1 * time.Second)
	// <-timer.C // 阻塞等待

	// Ticker（周期性）
	// ticker := time.NewTicker(100 * time.Millisecond)
	// defer ticker.Stop()

	// ============================================================
	//                      encoding/json 包
	// ============================================================
	fmt.Println("\n=== JSON 序列化 ===")

	// ----------------------------------------------------------
	// 结构体序列化
	// ----------------------------------------------------------
	type Person struct {
		Name    string   `json:"name"`
		Age     int      `json:"age"`
		Email   string   `json:"email,omitempty"` // 空值时省略
		Active  bool     `json:"active"`
		Tags    []string `json:"tags"`
		Secret  string   `json:"-"` // 忽略此字段
	}

	p := Person{
		Name:   "张三",
		Age:    25,
		Active: true,
		Tags:   []string{"go", "dev"},
		Secret: "hidden",
	}

	// 序列化
	jsonBytes, _ := json.Marshal(p)
	fmt.Println("JSON:", string(jsonBytes))

	// 格式化序列化
	jsonPretty, _ := json.MarshalIndent(p, "", "  ")
	fmt.Printf("格式化:\n%s\n", jsonPretty)

	// ----------------------------------------------------------
	// 反序列化
	// ----------------------------------------------------------
	fmt.Println("\n=== JSON 反序列化 ===")

	jsonStr := `{"name":"李四","age":30,"active":false,"tags":["python","java"]}`
	var p2 Person
	json.Unmarshal([]byte(jsonStr), &p2)
	fmt.Printf("反序列化: %+v\n", p2)

	// ----------------------------------------------------------
	// 动态 JSON
	// ----------------------------------------------------------
	fmt.Println("\n=== 动态 JSON ===")

	// 使用 map
	var m map[string]interface{}
	json.Unmarshal([]byte(jsonStr), &m)
	fmt.Println("Map:", m)
	fmt.Println("Name:", m["name"])
	fmt.Println("Age:", m["age"])

	// 类型断言获取值
	if name, ok := m["name"].(string); ok {
		fmt.Println("Name (string):", name)
	}

	// ----------------------------------------------------------
	// JSON 数组
	// ----------------------------------------------------------
	fmt.Println("\n=== JSON 数组 ===")

	jsonArray := `[{"name":"A","age":20},{"name":"B","age":25}]`
	var people []Person
	json.Unmarshal([]byte(jsonArray), &people)
	fmt.Println("People:", people)

	// ----------------------------------------------------------
	// 嵌套 JSON
	// ----------------------------------------------------------
	fmt.Println("\n=== 嵌套 JSON ===")

	type Address struct {
		City    string `json:"city"`
		Country string `json:"country"`
	}

	type Employee struct {
		Name    string  `json:"name"`
		Address Address `json:"address"`
	}

	emp := Employee{
		Name: "王五",
		Address: Address{
			City:    "北京",
			Country: "中国",
		},
	}

	empJSON, _ := json.MarshalIndent(emp, "", "  ")
	fmt.Printf("嵌套 JSON:\n%s\n", empJSON)
}
```
