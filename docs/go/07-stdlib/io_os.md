# io os.go

::: info 文件信息
- 📄 原文件：`02_io_os.go`
- 🔤 语言：go
:::

## 完整代码

```go
package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// ============================================================
//                      io 和 os 包
// ============================================================

func main02() {
	fmt.Println("\n==================== 02_io_os ====================")

	// ----------------------------------------------------------
	// os 包：环境和命令行
	// ----------------------------------------------------------
	fmt.Println("=== 环境和命令行 ===")

	// 环境变量
	fmt.Println("PATH:", os.Getenv("PATH")[:50]+"...")
	os.Setenv("MY_VAR", "Hello")
	fmt.Println("MY_VAR:", os.Getenv("MY_VAR"))

	// 命令行参数
	fmt.Println("程序名:", os.Args[0])
	fmt.Println("参数数量:", len(os.Args))

	// 当前目录
	pwd, _ := os.Getwd()
	fmt.Println("当前目录:", pwd)

	// 主机名
	hostname, _ := os.Hostname()
	fmt.Println("主机名:", hostname)

	// ----------------------------------------------------------
	// 文件操作
	// ----------------------------------------------------------
	fmt.Println("\n=== 文件操作 ===")

	// 创建临时文件
	tempFile := filepath.Join(os.TempDir(), "go_test.txt")
	fmt.Println("临时文件:", tempFile)

	// 写入文件（简单方式）
	content := []byte("Hello, Go!\n第二行\n")
	err := os.WriteFile(tempFile, content, 0644)
	if err != nil {
		fmt.Println("写入错误:", err)
		return
	}
	fmt.Println("写入成功")

	// 读取文件（简单方式）
	data, err := os.ReadFile(tempFile)
	if err != nil {
		fmt.Println("读取错误:", err)
		return
	}
	fmt.Printf("读取内容:\n%s", data)

	// 文件信息
	info, _ := os.Stat(tempFile)
	fmt.Println("文件名:", info.Name())
	fmt.Println("文件大小:", info.Size(), "字节")
	fmt.Println("修改时间:", info.ModTime())
	fmt.Println("是否目录:", info.IsDir())

	// ----------------------------------------------------------
	// 高级文件操作
	// ----------------------------------------------------------
	fmt.Println("\n=== 高级文件操作 ===")

	// 打开文件
	file, err := os.OpenFile(tempFile, os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Println("打开错误:", err)
		return
	}

	// 追加内容
	file.WriteString("追加的内容\n")
	file.Close()

	// 验证追加
	data, _ = os.ReadFile(tempFile)
	fmt.Printf("追加后:\n%s", data)

	// ----------------------------------------------------------
	// bufio 缓冲 I/O
	// ----------------------------------------------------------
	fmt.Println("\n=== bufio 缓冲读取 ===")

	file, _ = os.Open(tempFile)
	defer file.Close()

	// 按行读取
	scanner := bufio.NewScanner(file)
	lineNum := 1
	for scanner.Scan() {
		fmt.Printf("行 %d: %s\n", lineNum, scanner.Text())
		lineNum++
	}

	// ----------------------------------------------------------
	// io 包：通用 I/O 操作
	// ----------------------------------------------------------
	fmt.Println("\n=== io 操作 ===")

	// 复制
	srcFile, _ := os.Open(tempFile)
	defer srcFile.Close()

	dstFile, _ := os.Create(tempFile + ".copy")
	defer dstFile.Close()

	written, _ := io.Copy(dstFile, srcFile)
	fmt.Println("复制字节数:", written)

	// 读取固定字节
	srcFile.Seek(0, 0) // 回到文件开头
	buf := make([]byte, 5)
	n, _ := io.ReadFull(srcFile, buf)
	fmt.Printf("读取 %d 字节: %s\n", n, buf)

	// ----------------------------------------------------------
	// 目录操作
	// ----------------------------------------------------------
	fmt.Println("\n=== 目录操作 ===")

	tempDir := filepath.Join(os.TempDir(), "go_test_dir")

	// 创建目录
	os.MkdirAll(tempDir, 0755)
	fmt.Println("创建目录:", tempDir)

	// 读取目录
	entries, _ := os.ReadDir(os.TempDir())
	fmt.Println("临时目录文件数:", len(entries))

	// 遍历目录
	fmt.Println("前5个文件:")
	for i, entry := range entries {
		if i >= 5 {
			break
		}
		fmt.Printf("  %s (目录: %v)\n", entry.Name(), entry.IsDir())
	}

	// filepath 操作
	fmt.Println("\n=== filepath 操作 ===")

	path := "/usr/local/bin/go"
	fmt.Println("Dir:", filepath.Dir(path))
	fmt.Println("Base:", filepath.Base(path))
	fmt.Println("Ext:", filepath.Ext("file.txt"))
	fmt.Println("Join:", filepath.Join("usr", "local", "bin"))

	// 清理临时文件
	os.Remove(tempFile)
	os.Remove(tempFile + ".copy")
	os.RemoveAll(tempDir)
	fmt.Println("\n临时文件已清理")
}
```
