# C# 学习路径

## 简介

C# 是微软开发的现代、面向对象、类型安全的编程语言，运行在 .NET 平台上。从 Web 服务、桌面应用、游戏开发（Unity）到移动端（Xamarin/MAUI），C# 拥有极其广泛的应用场景。

## 为什么学 C#？

- **强类型安全**：编译时类型检查，减少运行时错误
- **现代语法**：async/await、LINQ、模式匹配、记录类型等现代特性
- **跨平台**：.NET 6+ 支持 Windows、Linux、macOS
- **生态丰富**：ASP.NET Core、Entity Framework、Unity、MAUI
- **企业级**：广泛用于金融、政务、游戏等企业软件开发

## 学习目录

### 基础篇

- [变量与数据类型](/csharp/01-basics/variables) — 值类型/引用类型、var 推断、可空类型、类型转换
- [流程控制](/csharp/01-basics/control_flow) — if/switch 表达式、循环、模式匹配、异常处理

### 函数篇

- [方法基础](/csharp/02-functions/basics) — 默认参数、ref/out、params、重载、泛型方法、本地函数
- [委托与 Lambda](/csharp/02-functions/delegates_lambda) — 委托、Func/Action、闭包、高阶函数、LINQ

### 类与接口篇

- [类与面向对象](/csharp/03-classes/class_basics) — 属性、构造函数、继承、多态、抽象类、Record 类型
- [接口与泛型](/csharp/03-classes/interfaces) — 接口、多接口实现、泛型约束、扩展方法

### 并发篇

- [async/await](/csharp/04-concurrency/async_await) — Task、并发执行、异常处理、取消令牌、异步流
- [Task 与线程](/csharp/04-concurrency/tasks_threads) — Thread、线程安全、Parallel、SemaphoreSlim、Channel

### 包管理篇

- [命名空间与 NuGet](/csharp/05-packages/namespaces_nuget) — 命名空间、JSON 序列化、文件操作、反射、NuGet 使用

### 测试篇

- [单元测试](/csharp/06-testing/unit_testing) — xUnit、参数化测试、Mock 测试替身、FluentAssertions

### 标准库篇

- [集合与 LINQ](/csharp/07-stdlib/collections_linq) — List/Dictionary/HashSet、LINQ 查询/聚合/分组、不可变集合

### 实战篇

- [Todo CLI](/csharp/08-projects/todo_cli) — 命令行 Todo 应用，综合运用类、接口、异步、JSON、文件 IO

## 运行示例

```bash
# 安装 .NET SDK
# https://dotnet.microsoft.com/download

# 创建并运行单文件程序（.NET 6+）
dotnet-script CSharp/01-basics/01_variables.cs

# 创建控制台项目
dotnet new console -n MyApp
cd MyApp
dotnet run

# 运行测试
dotnet test
```

## 推荐资源

- [C# 官方文档](https://docs.microsoft.com/zh-cn/dotnet/csharp/) — 微软官方教程
- [C# 8.0 in a Nutshell](https://www.oreilly.com/library/view/c-80-in/9781492051121/) — 权威参考书
- [.NET API 文档](https://docs.microsoft.com/zh-cn/dotnet/api/) — 标准库查阅
- [LeetCode C#](https://leetcode.cn/) — 算法练习
