# Rust 学习路径

## 简介

Rust 是一门注重安全性、并发性和性能的系统编程语言。通过所有权系统在编译时消除数据竞争和内存错误，无需垃圾回收即可保证内存安全。Rust 连续多年被 Stack Overflow 评为"最受喜爱的编程语言"。

## 为什么学 Rust？

- **内存安全**：所有权系统在编译时防止空指针、悬垂引用、数据竞争
- **零成本抽象**：高级语言的表达力，C/C++ 级别的性能
- **无畏并发**：类型系统确保线程安全
- **现代工具链**：Cargo 包管理器、rustfmt 格式化、clippy 代码检查
- **应用广泛**：系统编程、WebAssembly、嵌入式、CLI 工具、Web 服务

## 学习目录

### 基础篇

- [变量与数据类型](/rust/01-basics/variables) — 变量声明、基本类型、复合类型、类型转换
- [流程控制](/rust/01-basics/control_flow) — if/loop/while/for、match 模式匹配、if let

### 函数篇

- [函数基础](/rust/02-functions/basics) — 函数定义、返回值、表达式 vs 语句、泛型函数
- [闭包](/rust/02-functions/closures) — 闭包语法、捕获环境、Fn/FnMut/FnOnce

### 所有权篇（Rust 核心）

- [所有权](/rust/03-ownership/ownership) — 所有权规则、移动与克隆、栈与堆
- [借用与引用](/rust/03-ownership/borrowing) — 不可变引用、可变引用、切片
- [生命周期](/rust/03-ownership/lifetimes) — 生命周期标注、省略规则、结构体中的生命周期

### 类型系统篇

- [结构体](/rust/04-structs/struct_basics) — 结构体定义、方法、构建者模式
- [枚举与模式匹配](/rust/04-structs/enums) — 枚举、Option、match、if let
- [Trait](/rust/04-structs/traits) — trait 定义与实现、泛型约束、常用标准库 trait

### 并发篇

- [线程](/rust/05-concurrency/threads) — 线程创建、Arc/Mutex、RwLock
- [通道](/rust/05-concurrency/channels) — mpsc 通道、工作池、管道模式
- [异步编程](/rust/05-concurrency/async_await) — async/await、Future、tokio

### 错误处理篇

- [Result 与 Option](/rust/06-error-handling/result_option) — Result/Option、? 操作符、自定义错误

### 集合篇

- [Vec 与 HashMap](/rust/07-collections/vec_hashmap) — 动态数组、哈希映射、迭代器方法

### 实战篇

- [Todo CLI](/rust/08-projects/todo_cli) — 命令行 Todo 应用，综合实战

## 运行示例

```bash
# 单文件编译运行
rustc Rust/01-basics/01_variables.rs && ./01_variables

# 推荐使用 cargo（在项目目录中）
cargo run
```

## 推荐资源

- [The Rust Programming Language](https://doc.rust-lang.org/book/) — 官方教程（"The Book"）
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/) — 通过示例学习
- [Rustlings](https://github.com/rust-lang/rustlings) — 交互式练习
