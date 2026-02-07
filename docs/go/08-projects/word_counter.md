# word counter.go

::: info 文件信息
- 📄 原文件：`02_word_counter.go`
- 🔤 语言：go
:::

## 完整代码

```go
package main

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
	"unicode"
)

// ============================================================
//                      小项目2：单词计数器
// ============================================================
// 功能：统计文本中的单词频率
// 知识点：map、排序、字符串处理、正则表达式

// WordCounter 单词计数器
type WordCounter struct {
	words map[string]int
	total int
}

// NewWordCounter 创建单词计数器
func NewWordCounter() *WordCounter {
	return &WordCounter{
		words: make(map[string]int),
		total: 0,
	}
}

// AddText 添加文本
func (wc *WordCounter) AddText(text string) {
	// 转小写并提取单词
	text = strings.ToLower(text)

	// 使用正则提取单词（只保留字母）
	re := regexp.MustCompile(`[a-zA-Z]+`)
	matches := re.FindAllString(text, -1)

	for _, word := range matches {
		wc.words[word]++
		wc.total++
	}
}

// AddFile 从文件添加
func (wc *WordCounter) AddFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		wc.AddText(scanner.Text())
	}

	return scanner.Err()
}

// WordCount 单词及计数（用于排序）
type WordCount struct {
	Word  string
	Count int
}

// TopN 获取出现频率最高的 N 个单词
func (wc *WordCounter) TopN(n int) []WordCount {
	// 转换为切片
	counts := make([]WordCount, 0, len(wc.words))
	for word, count := range wc.words {
		counts = append(counts, WordCount{word, count})
	}

	// 按计数降序排序
	sort.Slice(counts, func(i, j int) bool {
		return counts[i].Count > counts[j].Count
	})

	// 返回前 N 个
	if n > len(counts) {
		n = len(counts)
	}
	return counts[:n]
}

// Stats 统计信息
func (wc *WordCounter) Stats() {
	fmt.Printf("总单词数: %d\n", wc.total)
	fmt.Printf("不同单词数: %d\n", len(wc.words))
}

// Search 搜索单词
func (wc *WordCounter) Search(word string) int {
	return wc.words[strings.ToLower(word)]
}

// runWordCounter 运行单词计数器
func runWordCounter() {
	fmt.Println("\n=== 小项目2: 单词计数器 ===")

	wc := NewWordCounter()

	// 示例文本
	sampleText := `
	Go is an open source programming language that makes it easy to build
	simple, reliable, and efficient software. Go was designed at Google
	in 2007 to improve programming productivity in an era of multicore,
	networked machines and large codebases. The language is often referred
	to as Golang because of its domain name, golang.org, but the proper
	name is Go. Go is expressive, concise, clean, and efficient. Its
	concurrency mechanisms make it easy to write programs that get the
	most out of multicore and networked machines.
	`

	wc.AddText(sampleText)

	fmt.Println("\n--- 统计信息 ---")
	wc.Stats()

	fmt.Println("\n--- 出现频率最高的 10 个单词 ---")
	topWords := wc.TopN(10)
	for i, wc := range topWords {
		fmt.Printf("%2d. %-15s %d 次\n", i+1, wc.Word, wc.Count)
	}

	fmt.Println("\n--- 搜索单词 ---")
	searchWords := []string{"go", "programming", "python"}
	for _, word := range searchWords {
		count := wc.Search(word)
		fmt.Printf("'%s' 出现 %d 次\n", word, count)
	}
}

// ============================================================
//                      小项目2b：中文字符统计
// ============================================================

// CharCounter 字符计数器
type CharCounter struct {
	chars map[rune]int
	total int
}

// NewCharCounter 创建字符计数器
func NewCharCounter() *CharCounter {
	return &CharCounter{
		chars: make(map[rune]int),
		total: 0,
	}
}

// AddText 添加文本
func (cc *CharCounter) AddText(text string) {
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			cc.chars[r]++
			cc.total++
		}
	}
}

// CharCount 字符计数
type CharCount struct {
	Char  rune
	Count int
}

// TopN 获取出现频率最高的 N 个字符
func (cc *CharCounter) TopN(n int) []CharCount {
	counts := make([]CharCount, 0, len(cc.chars))
	for char, count := range cc.chars {
		counts = append(counts, CharCount{char, count})
	}

	sort.Slice(counts, func(i, j int) bool {
		return counts[i].Count > counts[j].Count
	})

	if n > len(counts) {
		n = len(counts)
	}
	return counts[:n]
}

func runCharCounter() {
	fmt.Println("\n=== 中文字符统计 ===")

	cc := NewCharCounter()
	text := "Go语言是一种静态强类型、编译型、并发型，并具有垃圾回收功能的编程语言。Go语言于2009年发布。"
	cc.AddText(text)

	fmt.Println("原文:", text)
	fmt.Println("\n出现频率最高的 10 个字符:")
	for i, c := range cc.TopN(10) {
		fmt.Printf("%2d. '%c' - %d 次\n", i+1, c.Char, c.Count)
	}
}
```
