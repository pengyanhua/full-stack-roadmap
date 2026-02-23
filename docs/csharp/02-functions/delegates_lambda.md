# delegates lambda.cs

::: info 文件信息
- 📄 原文件：`02_delegates_lambda.cs`
- 🔤 语言：csharp
:::

## 完整代码

```csharp
// ============================================================
//                      委托、Lambda 与函数式编程
// ============================================================
// 委托（Delegate）是 C# 的函数类型，可以引用方法
// Lambda 表达式是匿名函数的简洁写法
// Func<T> 和 Action<T> 是内置的泛型委托类型
// 事件（Event）是基于委托的发布-订阅模式

using System;
using System.Collections.Generic;
using System.Linq;

class DelegatesAndLambda
{
    // ----------------------------------------------------------
    // 1. 委托类型定义
    // ----------------------------------------------------------
    // 委托定义了方法的签名（参数和返回值类型）
    // 【类比】委托类似于函数指针（C/C++），但更安全

    delegate int Operation(int a, int b);
    delegate void Printer(string message);
    delegate bool Predicate<T>(T item);

    // ----------------------------------------------------------
    // 2. 事件声明（基于委托）
    // ----------------------------------------------------------
    // 事件是委托的封装，遵循发布-订阅模式
    // EventHandler<T> 是标准事件委托

    event EventHandler<string>? OnMessageReceived;

    // ----------------------------------------------------------
    // 具体方法（用于委托赋值）
    // ----------------------------------------------------------
    static int Add(int a, int b) => a + b;
    static int Subtract(int a, int b) => a - b;
    static int Multiply(int a, int b) => a * b;

    static void PrintUpper(string msg) => Console.WriteLine(msg.ToUpper());
    static void PrintLower(string msg) => Console.WriteLine(msg.ToLower());

    static void Main()
    {
        Console.WriteLine("=== 委托基础 ===");

        // ----------------------------------------------------------
        // 委托实例化和调用
        // ----------------------------------------------------------
        Operation op = Add;
        Console.WriteLine($"Add(3, 4) = {op(3, 4)}");

        op = Subtract;
        Console.WriteLine($"Subtract(10, 3) = {op(10, 3)}");

        // ----------------------------------------------------------
        // 多播委托（Multicast Delegate）
        // ----------------------------------------------------------
        // 【特性】一个委托可以引用多个方法，调用时依次执行
        Printer printer = PrintUpper;
        printer += PrintLower;  // 添加第二个方法

        Console.WriteLine("\n=== 多播委托 ===");
        printer("Hello World");  // 依次调用两个方法

        printer -= PrintUpper;   // 移除方法
        printer("Only Lower");

        // ============================================================
        //                      Lambda 表达式
        // ============================================================
        Console.WriteLine("\n=== Lambda 表达式 ===");

        // ----------------------------------------------------------
        // Lambda 语法：(参数) => 表达式 或 (参数) => { 语句; }
        // ----------------------------------------------------------

        // 内置泛型委托 Func（有返回值）
        // Func<TParam1, TParam2, ..., TReturn>
        Func<int, int, int> add = (a, b) => a + b;
        Func<string, string> shout = s => s.ToUpper() + "!";
        Func<int, bool> isEven = n => n % 2 == 0;
        Func<int, int, string> compare = (a, b) =>
        {
            if (a > b) return $"{a} > {b}";
            if (a < b) return $"{a} < {b}";
            return $"{a} == {b}";
        };

        Console.WriteLine($"add(5, 3) = {add(5, 3)}");
        Console.WriteLine($"shout(\"hello\") = {shout("hello")}");
        Console.WriteLine($"isEven(4) = {isEven(4)}");
        Console.WriteLine($"compare(3, 7) = {compare(3, 7)}");

        // 内置泛型委托 Action（无返回值）
        // Action<TParam1, TParam2, ...>
        Action<string> log = msg => Console.WriteLine($"[LOG] {msg}");
        Action<string, int> repeat = (s, n) =>
        {
            for (int i = 0; i < n; i++) Console.Write(s);
            Console.WriteLine();
        };

        log("这是一条日志");
        repeat("★", 5);

        // Predicate<T>（返回 bool 的委托）
        Predicate<int> isPositive = n => n > 0;
        Console.WriteLine($"isPositive(5) = {isPositive(5)}");
        Console.WriteLine($"isPositive(-3) = {isPositive(-3)}");

        // ============================================================
        //                      闭包（Closure）
        // ============================================================
        Console.WriteLine("\n=== 闭包 ===");

        // 闭包：Lambda 可以捕获外部变量
        // 【注意】捕获的是变量的引用，不是值的副本

        int multiplier = 3;
        Func<int, int> triple = x => x * multiplier;
        Console.WriteLine($"triple(7) = {triple(7)}");

        multiplier = 10;  // 改变外部变量
        Console.WriteLine($"triple(7) = {triple(7)} (multiplier 已变为 {multiplier})");

        // 创建计数器（闭包经典用法）
        Func<int> makeCounter()
        {
            int count = 0;
            return () => ++count;
        }
        var counter1 = makeCounter();
        var counter2 = makeCounter();
        Console.WriteLine($"counter1: {counter1()}, {counter1()}, {counter1()}");
        Console.WriteLine($"counter2: {counter2()}, {counter2()}");

        // ============================================================
        //                      高阶函数
        // ============================================================
        Console.WriteLine("\n=== 高阶函数 ===");

        // 接受委托作为参数
        int[] numbers = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        // 自定义 Filter
        static T[] Filter<T>(T[] arr, Func<T, bool> predicate)
        {
            var result = new List<T>();
            foreach (var item in arr)
                if (predicate(item)) result.Add(item);
            return result.ToArray();
        }

        // 自定义 Map
        static TResult[] Map<T, TResult>(T[] arr, Func<T, TResult> transform)
        {
            var result = new TResult[arr.Length];
            for (int i = 0; i < arr.Length; i++)
                result[i] = transform(arr[i]);
            return result;
        }

        // 自定义 Reduce
        static TResult Reduce<T, TResult>(T[] arr, TResult seed, Func<TResult, T, TResult> accumulator)
        {
            var result = seed;
            foreach (var item in arr)
                result = accumulator(result, item);
            return result;
        }

        var evens = Filter(numbers, n => n % 2 == 0);
        Console.WriteLine($"偶数: [{string.Join(", ", evens)}]");

        var squares = Map(numbers, n => n * n);
        Console.WriteLine($"平方: [{string.Join(", ", squares)}]");

        var sum = Reduce(numbers, 0, (acc, n) => acc + n);
        Console.WriteLine($"求和: {sum}");

        // 返回委托的方法（函数工厂）
        static Func<int, int> MakeAdder(int addend) => x => x + addend;
        var addFive = MakeAdder(5);
        var addTen = MakeAdder(10);
        Console.WriteLine($"addFive(3) = {addFive(3)}");
        Console.WriteLine($"addTen(3) = {addTen(3)}");

        // ============================================================
        //                      LINQ 与 Lambda
        // ============================================================
        Console.WriteLine("\n=== LINQ 查询 ===");

        var students = new List<(string Name, int Score, string Grade)>
        {
            ("张三", 90, "A"),
            ("李四", 75, "B"),
            ("王五", 88, "A"),
            ("赵六", 62, "C"),
            ("钱七", 95, "A"),
        };

        // 方法链式写法（推荐）
        var topStudents = students
            .Where(s => s.Score >= 80)
            .OrderByDescending(s => s.Score)
            .Select(s => $"{s.Name}: {s.Score}分");

        Console.WriteLine("80分以上（倒序）:");
        foreach (var s in topStudents)
            Console.WriteLine($"  {s}");

        // 统计
        double avg = students.Average(s => s.Score);
        int max = students.Max(s => s.Score);
        Console.WriteLine($"平均分: {avg:F1}, 最高分: {max}");

        // GroupBy 分组
        var byGrade = students.GroupBy(s => s.Grade);
        foreach (var group in byGrade.OrderBy(g => g.Key))
        {
            var names = string.Join(", ", group.Select(s => s.Name));
            Console.WriteLine($"  {group.Key} 等: {names}");
        }

        Console.WriteLine("\n=== 委托与 Lambda 演示完成 ===");
    }
}
```
