# variables.cs

::: info 文件信息
- 📄 原文件：`01_variables.cs`
- 🔤 语言：csharp
:::

## 完整代码

```csharp
// ============================================================
//                      变量与数据类型
// ============================================================
// C# 是强类型、面向对象的语言，运行在 .NET 平台上
// 支持值类型（栈分配）和引用类型（堆分配）两大类型系统
// 使用 var 关键字可以进行类型推断，编译器自动推导类型

using System;

class Variables
{
    static void Main()
    {
        Console.WriteLine("=== 变量声明 ===");

        // ----------------------------------------------------------
        // 1. 值类型变量
        // ----------------------------------------------------------
        // 值类型直接存储数据，分配在栈上（或内联到包含它的对象中）
        // 【常用值类型】int, long, double, float, bool, char, decimal, struct

        int age = 25;                   // 32位整数，-2^31 到 2^31-1
        long bigNum = 9_000_000_000L;   // 64位整数（L 后缀）
        double pi = 3.141592653589793;  // 64位浮点（默认浮点类型）
        float f = 3.14f;               // 32位浮点（f 后缀）
        decimal money = 99.99m;        // 128位精确小数，适合金融计算（m 后缀）
        bool isActive = true;
        char grade = 'A';

        Console.WriteLine($"int: {age}");
        Console.WriteLine($"long: {bigNum}");
        Console.WriteLine($"double: {pi}");
        Console.WriteLine($"float: {f}");
        Console.WriteLine($"decimal: {money}");
        Console.WriteLine($"bool: {isActive}");
        Console.WriteLine($"char: {grade}");

        // 【技巧】数字字面量可以用下划线增加可读性（C# 7.0+）
        int million = 1_000_000;
        int hex = 0xFF;          // 十六进制
        int binary = 0b1111_0000; // 二进制（C# 7.0+）
        Console.WriteLine($"百万: {million}, 十六进制: {hex}, 二进制: {binary}");

        // ----------------------------------------------------------
        // 2. 引用类型变量
        // ----------------------------------------------------------
        // 引用类型存储的是对象的内存地址，数据在堆上
        // 【常用引用类型】string, object, class, array, delegate

        string name = "张三";           // 字符串（不可变）
        string greeting = $"你好，{name}！"; // 字符串插值（C# 6.0+）
        object obj = 42;               // object 是所有类型的基类

        Console.WriteLine(greeting);
        Console.WriteLine($"对象类型: {obj.GetType()}");

        // ----------------------------------------------------------
        // 3. var 类型推断
        // ----------------------------------------------------------
        // 编译器根据右侧表达式自动推断类型
        // 【注意】var 不是动态类型，编译后类型确定，不能改变
        // 【适用场景】类型名称冗长时简化代码，匿名类型

        var autoInt = 100;           // 推断为 int
        var autoStr = "C# 编程";     // 推断为 string
        var autoList = new System.Collections.Generic.List<int>(); // 推断为 List<int>

        Console.WriteLine($"var int: {autoInt.GetType().Name}");
        Console.WriteLine($"var string: {autoStr.GetType().Name}");

        // ----------------------------------------------------------
        // 4. 常量（const 和 readonly）
        // ----------------------------------------------------------
        // const: 编译时常量，必须在声明时赋值，隐式为 static
        // readonly: 运行时常量，可在构造函数中赋值

        const double MaxScore = 100.0;
        const string AppName = "学习系统";

        Console.WriteLine($"常量: {AppName} - 满分 {MaxScore}");

        // ============================================================
        //                      数据类型详解
        // ============================================================
        Console.WriteLine("\n=== 数据类型详解 ===");

        // ----------------------------------------------------------
        // 整数类型范围
        // ----------------------------------------------------------
        Console.WriteLine($"byte:   0 到 {byte.MaxValue}");       // 0 到 255
        Console.WriteLine($"sbyte:  {sbyte.MinValue} 到 {sbyte.MaxValue}");
        Console.WriteLine($"short:  {short.MinValue} 到 {short.MaxValue}");
        Console.WriteLine($"ushort: 0 到 {ushort.MaxValue}");
        Console.WriteLine($"int:    {int.MinValue} 到 {int.MaxValue}");
        Console.WriteLine($"uint:   0 到 {uint.MaxValue}");
        Console.WriteLine($"long:   {long.MinValue} 到 {long.MaxValue}");

        // ----------------------------------------------------------
        // 字符串详解
        // ----------------------------------------------------------
        Console.WriteLine("\n=== 字符串操作 ===");

        string s1 = "Hello";
        string s2 = "World";

        // 字符串拼接
        string concat = s1 + " " + s2;
        Console.WriteLine($"拼接: {concat}");

        // 字符串方法
        Console.WriteLine($"长度: {s1.Length}");
        Console.WriteLine($"大写: {s1.ToUpper()}");
        Console.WriteLine($"小写: {s1.ToLower()}");
        Console.WriteLine($"包含: {concat.Contains("World")}");
        Console.WriteLine($"替换: {concat.Replace("World", "C#")}");
        Console.WriteLine($"分割: {string.Join(",", "a-b-c".Split('-'))}");

        // 逐字字符串（@ 前缀，不处理转义）
        string path = @"C:\Users\张三\Documents";
        Console.WriteLine($"路径: {path}");

        // 多行字符串插值
        string report = $"""
            姓名: {name}
            年龄: {age}
            活跃: {isActive}
            """;  // C# 11 原始字符串字面量
        Console.WriteLine(report);

        // ----------------------------------------------------------
        // 可空类型（Nullable Types）
        // ----------------------------------------------------------
        Console.WriteLine("\n=== 可空类型 ===");

        // 值类型默认不能为 null，加 ? 使其可空
        // 【用途】数据库字段可能为 null、表示"未设置"状态
        int? nullableInt = null;
        double? nullableDouble = 3.14;

        Console.WriteLine($"可空 int: {nullableInt ?? -1}");  // ?? 空合并运算符
        Console.WriteLine($"可空 double: {nullableDouble}");
        Console.WriteLine($"HasValue: {nullableDouble.HasValue}");
        Console.WriteLine($"Value: {nullableDouble.Value}");

        // 空合并赋值（C# 8.0+）
        string? nullableStr = null;
        nullableStr ??= "默认值";
        Console.WriteLine($"空合并赋值: {nullableStr}");

        // ============================================================
        //                      类型转换
        // ============================================================
        Console.WriteLine("\n=== 类型转换 ===");

        // 隐式转换（安全，不会丢失精度）
        int i = 100;
        long l = i;     // int -> long 隐式转换
        double d = i;   // int -> double 隐式转换
        Console.WriteLine($"隐式: int {i} -> long {l} -> double {d}");

        // 显式转换（可能丢失精度或抛出异常）
        double pi2 = 3.99;
        int truncated = (int)pi2;  // 截断小数部分
        Console.WriteLine($"显式: double {pi2} -> int {truncated}（截断）");

        // Convert 类（更安全的转换）
        string numStr = "42";
        int parsed = Convert.ToInt32(numStr);
        Console.WriteLine($"Convert: \"{numStr}\" -> {parsed}");

        // TryParse（最安全的字符串转数字）
        // 【推荐】当字符串来自用户输入时使用 TryParse
        string input = "123abc";
        if (int.TryParse(input, out int result))
        {
            Console.WriteLine($"解析成功: {result}");
        }
        else
        {
            Console.WriteLine($"\"{input}\" 不是有效整数");
        }

        // is 和 as 运算符（引用类型转换）
        object o = "Hello, C#!";
        if (o is string str)  // 模式匹配（C# 7.0+）
        {
            Console.WriteLine($"is 转换: {str.ToUpper()}");
        }

        string? asStr = o as string;  // as 失败返回 null，不抛异常
        Console.WriteLine($"as 转换: {asStr?.Length} 个字符");
    }
}
```
