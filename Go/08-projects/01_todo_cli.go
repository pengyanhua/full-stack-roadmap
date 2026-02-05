package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// ============================================================
//                      小项目1：命令行 Todo 应用
// ============================================================
// 功能：添加、列出、完成、删除待办事项
// 知识点：结构体、切片、文件操作、JSON、用户输入

// Task 待办事项
type Task struct {
	ID        int       `json:"id"`
	Title     string    `json:"title"`
	Completed bool      `json:"completed"`
	CreatedAt time.Time `json:"created_at"`
}

// TodoList 待办列表
type TodoList struct {
	Tasks    []Task `json:"tasks"`
	NextID   int    `json:"next_id"`
	filename string
}

// NewTodoList 创建待办列表
func NewTodoList(filename string) *TodoList {
	tl := &TodoList{
		Tasks:    []Task{},
		NextID:   1,
		filename: filename,
	}
	tl.Load() // 尝试加载已有数据
	return tl
}

// Add 添加任务
func (tl *TodoList) Add(title string) {
	task := Task{
		ID:        tl.NextID,
		Title:     title,
		Completed: false,
		CreatedAt: time.Now(),
	}
	tl.Tasks = append(tl.Tasks, task)
	tl.NextID++
	tl.Save()
	fmt.Printf("✅ 添加任务: [%d] %s\n", task.ID, task.Title)
}

// List 列出所有任务
func (tl *TodoList) List() {
	if len(tl.Tasks) == 0 {
		fmt.Println("📭 没有待办事项")
		return
	}

	fmt.Println("\n📋 待办事项列表:")
	fmt.Println(strings.Repeat("-", 50))

	for _, task := range tl.Tasks {
		status := "[ ]"
		if task.Completed {
			status = "[✓]"
		}
		fmt.Printf("%s %d. %s\n", status, task.ID, task.Title)
	}
	fmt.Println(strings.Repeat("-", 50))
	fmt.Printf("共 %d 项，已完成 %d 项\n", len(tl.Tasks), tl.CompletedCount())
}

// Complete 完成任务
func (tl *TodoList) Complete(id int) {
	for i := range tl.Tasks {
		if tl.Tasks[i].ID == id {
			tl.Tasks[i].Completed = true
			tl.Save()
			fmt.Printf("✅ 完成任务: [%d] %s\n", id, tl.Tasks[i].Title)
			return
		}
	}
	fmt.Printf("❌ 未找到任务 ID: %d\n", id)
}

// Delete 删除任务
func (tl *TodoList) Delete(id int) {
	for i, task := range tl.Tasks {
		if task.ID == id {
			tl.Tasks = append(tl.Tasks[:i], tl.Tasks[i+1:]...)
			tl.Save()
			fmt.Printf("🗑️  删除任务: [%d] %s\n", id, task.Title)
			return
		}
	}
	fmt.Printf("❌ 未找到任务 ID: %d\n", id)
}

// CompletedCount 已完成数量
func (tl *TodoList) CompletedCount() int {
	count := 0
	for _, task := range tl.Tasks {
		if task.Completed {
			count++
		}
	}
	return count
}

// Save 保存到文件
func (tl *TodoList) Save() error {
	data, err := json.MarshalIndent(tl, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(tl.filename, data, 0644)
}

// Load 从文件加载
func (tl *TodoList) Load() error {
	data, err := os.ReadFile(tl.filename)
	if err != nil {
		return err // 文件不存在时返回错误，但不影响使用
	}
	return json.Unmarshal(data, tl)
}

// runTodoApp 运行 Todo 应用
func runTodoApp() {
	fmt.Println("=== 小项目1: 命令行 Todo 应用 ===")

	// 使用临时文件
	todoList := NewTodoList("todos.json")

	// 演示操作
	fmt.Println("\n--- 演示操作 ---")

	todoList.Add("学习 Go 语言基础")
	todoList.Add("完成 goroutine 练习")
	todoList.Add("阅读 Go 标准库文档")

	todoList.List()

	todoList.Complete(1)
	todoList.Delete(2)

	todoList.List()

	// 清理演示文件
	os.Remove("todos.json")

	// 交互式使用说明
	fmt.Println("\n--- 交互式命令 ---")
	fmt.Println("add <title>  - 添加任务")
	fmt.Println("list         - 列出任务")
	fmt.Println("done <id>    - 完成任务")
	fmt.Println("del <id>     - 删除任务")
	fmt.Println("quit         - 退出")
}

// interactiveTodo 交互式 Todo（可选运行）
func interactiveTodo() {
	todoList := NewTodoList("todos.json")
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		parts := strings.SplitN(input, " ", 2)
		cmd := parts[0]

		switch cmd {
		case "add":
			if len(parts) < 2 {
				fmt.Println("用法: add <title>")
				continue
			}
			todoList.Add(parts[1])

		case "list":
			todoList.List()

		case "done":
			if len(parts) < 2 {
				fmt.Println("用法: done <id>")
				continue
			}
			id, _ := strconv.Atoi(parts[1])
			todoList.Complete(id)

		case "del":
			if len(parts) < 2 {
				fmt.Println("用法: del <id>")
				continue
			}
			id, _ := strconv.Atoi(parts[1])
			todoList.Delete(id)

		case "quit", "exit", "q":
			fmt.Println("再见!")
			return

		default:
			fmt.Println("未知命令:", cmd)
		}
	}
}
