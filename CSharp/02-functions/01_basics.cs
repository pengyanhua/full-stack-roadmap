// ============================================================
//                      方法基础
// ============================================================
// C# 方法是类的成员，支持丰富的参数特性
// 包括：默认参数、命名参数、ref/out/in 参数、params 可变参数
// C# 9.0+ 支持顶级语句，10.0+ 支持全局 using

using System;
using System.Collections.Generic;

class MethodBasics
{
    // ----------------------------------------------------------
    // 1. 基本方法定义
    // ----------------------------------------------------------
    // 访问修饰符 返回类型 方法名(参数列表)
    // 【返回 void】表示不返回值

    static int Add(int a, int b)
    {
        return a + b;
    }

    // 表达式体方法（C# 6.0+，单行简写）
    // 【适用场景】简单的计算和转换
    static double Square(double x) => x * x;
    static string Greet(string name) => $"你好，{name}！";

    // ----------------------------------------------------------
    // 2. 默认参数值
    // ----------------------------------------------------------
    // 调用时可以省略有默认值的参数
    // 【注意】有默认值的参数必须放在无默认值参数之后

    static string CreateEmail(string name, string domain = "example.com", bool isAdmin = false)
    {
        string prefix = isAdmin ? "admin." : "";
        return $"{prefix}{name.ToLower()}@{domain}";
    }

    // ----------------------------------------------------------
    // 3. ref / out / in 参数
    // ----------------------------------------------------------
    // ref: 引用传递，调用前必须初始化，双向传递
    // out: 输出参数，方法内必须赋值，适合返回多个值
    // in: 只读引用，不能修改（C# 7.2+），适合大型结构体的性能优化

    static void Swap(ref int a, ref int b)
    {
        int temp = a;
        a = b;
        b = temp;
    }

    static bool TryDivide(int numerator, int denominator, out double result)
    {
        if (denominator == 0)
        {
            result = 0;
            return false;
        }
        result = (double)numerator / denominator;
        return true;
    }

    // ----------------------------------------------------------
    // 4. params 可变参数
    // ----------------------------------------------------------
    // 允许传入任意数量的参数（必须是最后一个参数）

    static int Sum(params int[] numbers)
    {
        int total = 0;
        foreach (int n in numbers)
            total += n;
        return total;
    }

    static double Average(string label, params double[] values)
    {
        if (values.Length == 0) return 0;
        double sum = 0;
        foreach (double v in values) sum += v;
        double avg = sum / values.Length;
        Console.WriteLine($"{label}: 共 {values.Length} 个值，平均 {avg:F2}");
        return avg;
    }

    // ----------------------------------------------------------
    // 5. 方法重载（Overloading）
    // ----------------------------------------------------------
    // 相同方法名，不同参数（类型或数量）

    static string Format(int value) => $"整数: {value}";
    static string Format(double value) => $"浮点: {value:F2}";
    static string Format(string value) => $"字符串: \"{value}\"";
    static string Format(int value, int width) => value.ToString().PadLeft(width, '0');

    // ----------------------------------------------------------
    // 6. 递归方法
    // ----------------------------------------------------------
    static long Factorial(int n)
    {
        if (n <= 1) return 1;
        return n * Factorial(n - 1);
    }

    static int Fibonacci(int n)
    {
        if (n <= 1) return n;
        return Fibonacci(n - 1) + Fibonacci(n - 2);
    }

    // ----------------------------------------------------------
    // 7. 泛型方法
    // ----------------------------------------------------------
    // 【优势】一个方法处理多种类型，避免代码重复
    // T 是类型参数，调用时由编译器推断或手动指定

    static T Max<T>(T a, T b) where T : IComparable<T>
    {
        return a.CompareTo(b) >= 0 ? a : b;
    }

    static List<T> Repeat<T>(T item, int times)
    {
        var list = new List<T>();
        for (int i = 0; i < times; i++)
            list.Add(item);
        return list;
    }

    static void Main()
    {
        Console.WriteLine("=== 方法基础 ===");

        // 基本调用
        Console.WriteLine($"加法: {Add(3, 5)}");
        Console.WriteLine($"平方: {Square(4.0)}");
        Console.WriteLine($"问候: {Greet("世界")}");

        // 默认参数
        Console.WriteLine("\n=== 默认参数 ===");
        Console.WriteLine(CreateEmail("zhangsan"));
        Console.WriteLine(CreateEmail("zhangsan", "company.com"));
        Console.WriteLine(CreateEmail("admin", isAdmin: true));  // 命名参数

        // ref 参数
        Console.WriteLine("\n=== ref/out 参数 ===");
        int x = 10, y = 20;
        Console.WriteLine($"交换前: x={x}, y={y}");
        Swap(ref x, ref y);
        Console.WriteLine($"交换后: x={x}, y={y}");

        // out 参数
        if (TryDivide(10, 3, out double divResult))
            Console.WriteLine($"10 / 3 = {divResult:F4}");
        if (!TryDivide(10, 0, out _))  // _ 丢弃不需要的 out 参数
            Console.WriteLine("除以零失败");

        // params
        Console.WriteLine("\n=== params 参数 ===");
        Console.WriteLine($"Sum(1,2,3): {Sum(1, 2, 3)}");
        Console.WriteLine($"Sum(1..5): {Sum(1, 2, 3, 4, 5)}");
        int[] arr = { 10, 20, 30 };
        Console.WriteLine($"Sum(arr): {Sum(arr)}");  // 也可以传数组
        Average("成绩", 90, 85, 92, 88, 76);

        // 重载
        Console.WriteLine("\n=== 方法重载 ===");
        Console.WriteLine(Format(42));
        Console.WriteLine(Format(3.14));
        Console.WriteLine(Format("Hello"));
        Console.WriteLine(Format(7, 4));  // 格式化为 "0007"

        // 递归
        Console.WriteLine("\n=== 递归 ===");
        for (int i = 0; i <= 10; i++)
            Console.Write($"{Factorial(i)} ");
        Console.WriteLine("(阶乘)");

        for (int i = 0; i <= 10; i++)
            Console.Write($"{Fibonacci(i)} ");
        Console.WriteLine("(斐波那契)");

        // 泛型方法
        Console.WriteLine("\n=== 泛型方法 ===");
        Console.WriteLine($"Max(3, 7): {Max(3, 7)}");
        Console.WriteLine($"Max(\"apple\", \"banana\"): {Max("apple", "banana")}");
        Console.WriteLine($"Max(3.14, 2.71): {Max(3.14, 2.71)}");
        var repeated = Repeat("★", 5);
        Console.WriteLine($"Repeat: {string.Join("", repeated)}");

        // ----------------------------------------------------------
        // 8. 本地函数（Local Functions，C# 7.0+）
        // ----------------------------------------------------------
        Console.WriteLine("\n=== 本地函数 ===");
        // 【用途】将辅助逻辑限制在当前方法内，避免污染类的接口

        int result = ComputeWithHelper(10);
        Console.WriteLine($"计算结果: {result}");

        static int ComputeWithHelper(int input)
        {
            // 本地辅助函数，只在 ComputeWithHelper 内可见
            static int Double(int n) => n * 2;
            static int AddTen(int n) => n + 10;

            return AddTen(Double(input));
        }

        // ----------------------------------------------------------
        // 9. 命名参数
        // ----------------------------------------------------------
        Console.WriteLine("\n=== 命名参数 ===");
        // 【优势】提高代码可读性，可以任意顺序传参（只要都命名）
        string email = CreateEmail(
            domain: "gmail.com",
            name: "wangwu",
            isAdmin: false
        );
        Console.WriteLine($"邮箱: {email}");
    }
}
