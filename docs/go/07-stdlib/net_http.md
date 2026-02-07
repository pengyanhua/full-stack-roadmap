# net http.go

::: info 文件信息
- 📄 原文件：`04_net_http.go`
- 🔤 语言：go
:::

## 完整代码

```go
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// ============================================================
//                      net/http 包
// ============================================================

func main04() {
	fmt.Println("\n==================== 04_net_http ====================")

	// ----------------------------------------------------------
	// HTTP 客户端：GET 请求
	// ----------------------------------------------------------
	fmt.Println("=== HTTP GET 请求 ===")

	// 简单 GET
	resp, err := http.Get("https://httpbin.org/get")
	if err != nil {
		fmt.Println("请求错误:", err)
	} else {
		defer resp.Body.Close()
		fmt.Println("状态码:", resp.StatusCode)
		fmt.Println("状态:", resp.Status)
		fmt.Println("Content-Type:", resp.Header.Get("Content-Type"))

		// 读取响应体
		body, _ := io.ReadAll(resp.Body)
		// 只显示前200个字符
		if len(body) > 200 {
			fmt.Printf("响应体（前200字符）:\n%s...\n", body[:200])
		} else {
			fmt.Printf("响应体:\n%s\n", body)
		}
	}

	// ----------------------------------------------------------
	// HTTP 客户端：带超时
	// ----------------------------------------------------------
	fmt.Println("\n=== 带超时的客户端 ===")

	client := &http.Client{
		Timeout: 10 * time.Second,
	}

	resp, err = client.Get("https://httpbin.org/delay/1")
	if err != nil {
		fmt.Println("请求错误:", err)
	} else {
		defer resp.Body.Close()
		fmt.Println("带超时请求成功:", resp.StatusCode)
	}

	// ----------------------------------------------------------
	// 自定义请求
	// ----------------------------------------------------------
	fmt.Println("\n=== 自定义请求 ===")

	req, _ := http.NewRequest("GET", "https://httpbin.org/headers", nil)
	req.Header.Set("User-Agent", "Go-HTTP-Client/1.0")
	req.Header.Set("Accept", "application/json")

	resp, err = client.Do(req)
	if err != nil {
		fmt.Println("请求错误:", err)
	} else {
		defer resp.Body.Close()
		fmt.Println("自定义请求成功:", resp.StatusCode)
	}

	// ----------------------------------------------------------
	// POST 请求
	// ----------------------------------------------------------
	fmt.Println("\n=== POST 请求 ===")

	// JSON POST
	jsonData := `{"name":"Go","version":"1.21"}`
	resp, err = http.Post(
		"https://httpbin.org/post",
		"application/json",
		strings.NewReader(jsonData),
	)
	if err != nil {
		fmt.Println("POST 错误:", err)
	} else {
		defer resp.Body.Close()
		fmt.Println("POST 成功:", resp.StatusCode)
	}

	// Form POST
	formData := url.Values{}
	formData.Set("username", "testuser")
	formData.Set("password", "testpass")

	resp, err = http.PostForm("https://httpbin.org/post", formData)
	if err != nil {
		fmt.Println("PostForm 错误:", err)
	} else {
		defer resp.Body.Close()
		fmt.Println("PostForm 成功:", resp.StatusCode)
	}

	// ----------------------------------------------------------
	// URL 解析
	// ----------------------------------------------------------
	fmt.Println("\n=== URL 解析 ===")

	u, _ := url.Parse("https://example.com:8080/path/to/resource?name=go&version=1.21#section")
	fmt.Println("Scheme:", u.Scheme)
	fmt.Println("Host:", u.Host)
	fmt.Println("Path:", u.Path)
	fmt.Println("RawQuery:", u.RawQuery)
	fmt.Println("Fragment:", u.Fragment)

	// 解析查询参数
	params := u.Query()
	fmt.Println("name:", params.Get("name"))
	fmt.Println("version:", params.Get("version"))

	// 构建 URL
	u2 := &url.URL{
		Scheme: "https",
		Host:   "api.example.com",
		Path:   "/v1/users",
	}
	q := u2.Query()
	q.Set("page", "1")
	q.Set("limit", "10")
	u2.RawQuery = q.Encode()
	fmt.Println("构建的 URL:", u2.String())

	// ----------------------------------------------------------
	// HTTP 服务器（示例代码）
	// ----------------------------------------------------------
	fmt.Println("\n=== HTTP 服务器（代码示例）===")

	/*
	// 基本服务器
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "Hello, World!")
	})

	http.HandleFunc("/api/users", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case "GET":
			users := []map[string]string{
				{"name": "Alice"},
				{"name": "Bob"},
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(users)
		case "POST":
			var user map[string]string
			json.NewDecoder(r.Body).Decode(&user)
			w.WriteHeader(http.StatusCreated)
			json.NewEncoder(w).Encode(user)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})

	// 静态文件服务
	http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("./static"))))

	// 启动服务器
	log.Fatal(http.ListenAndServe(":8080", nil))
	*/

	fmt.Println("参见代码注释中的服务器示例")

	// ----------------------------------------------------------
	// JSON API 调用示例
	// ----------------------------------------------------------
	fmt.Println("\n=== JSON API 示例 ===")

	type IPInfo struct {
		Origin string `json:"origin"`
	}

	resp, err = http.Get("https://httpbin.org/ip")
	if err != nil {
		fmt.Println("请求错误:", err)
		return
	}
	defer resp.Body.Close()

	var info IPInfo
	json.NewDecoder(resp.Body).Decode(&info)
	fmt.Println("你的 IP:", info.Origin)
}

// ============================================================
//                      重要注意事项
// ============================================================
//
// 1. 【响应体关闭】
//    务必使用 defer resp.Body.Close()
//    否则会造成连接泄漏
//
// 2. 【超时设置】
//    默认 http.Client 无超时
//    生产环境务必设置超时
//
// 3. 【复用 Client】
//    http.Client 可以安全复用
//    不要每次请求都创建新的
//
// 4. 【连接池】
//    http.Client 内置连接池
//    可通过 Transport 配置
//
// 5. 【HTTPS】
//    Go 默认验证 TLS 证书
//    不要禁用证书验证（不安全）
//
// 6. 【并发安全】
//    http.Client 是并发安全的
//    http.Transport 也是并发安全的
```
