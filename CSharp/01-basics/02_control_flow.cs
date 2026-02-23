// ============================================================
//                      流程控制
// ============================================================
// C# 提供完整的流程控制结构：条件判断、循环、跳转
// C# 8.0+ 引入的模式匹配让条件逻辑更加简洁强大
// switch 表达式（C# 8.0+）是传统 switch 的现代替代品

using System;
using System.Collections.Generic;

class ControlFlow
{
    static void Main()
    {
        Console.WriteLine("=== 条件语句 ===");

        // ----------------------------------------------------------
        // 1. if / else if / else
        // ----------------------------------------------------------
        int score = 85;

        if (score >= 90)
        {
            Console.WriteLine("优秀");
        }
        else if (score >= 80)
        {
            Console.WriteLine("良好");
        }
        else if (score >= 60)
        {
            Console.WriteLine("及格");
        }
        else
        {
            Console.WriteLine("不及格");
        }

        // 三元运算符
        string result = score >= 60 ? "通过" : "未通过";
        Console.WriteLine($"结果: {result}");

        // ----------------------------------------------------------
        // 2. switch 语句（传统）
        // ----------------------------------------------------------
        Console.WriteLine("\n=== switch 语句 ===");

        int dayOfWeek = 3;
        switch (dayOfWeek)
        {
            case 1:
                Console.WriteLine("周一");
                break;
            case 2:
                Console.WriteLine("周二");
                break;
            case 3:
            case 4:
            case 5:
                Console.WriteLine($"工作日（第 {dayOfWeek} 天）");
                break;
            case 6:
            case 7:
                Console.WriteLine("周末");
                break;
            default:
                Console.WriteLine("无效日期");
                break;
        }

        // ----------------------------------------------------------
        // 3. switch 表达式（C# 8.0+，推荐）
        // ----------------------------------------------------------
        // 【优点】更简洁，是表达式（可以赋值），支持模式匹配
        Console.WriteLine("\n=== switch 表达式 ===");

        string dayName = dayOfWeek switch
        {
            1 => "周一",
            2 => "周二",
            3 => "周三",
            4 => "周四",
            5 => "周五",
            6 or 7 => "周末",
            _ => "无效"  // _ 是 discard 模式，相当于 default
        };
        Console.WriteLine($"今天: {dayName}");

        // 属性模式匹配
        var person = new { Name = "李四", Age = 25, IsVip = true };
        string discount = person switch
        {
            { IsVip: true, Age: >= 60 } => "老年VIP 8折",
            { IsVip: true } => "VIP 9折",
            { Age: < 18 } => "学生 7折",
            _ => "普通 无折扣"
        };
        Console.WriteLine($"折扣: {discount}");

        // ============================================================
        //                      循环语句
        // ============================================================
        Console.WriteLine("\n=== for 循环 ===");

        // ----------------------------------------------------------
        // 4. for 循环
        // ----------------------------------------------------------
        for (int i = 0; i < 5; i++)
        {
            Console.Write($"{i} ");
        }
        Console.WriteLine();

        // 倒序
        for (int i = 5; i > 0; i--)
        {
            Console.Write($"{i} ");
        }
        Console.WriteLine();

        // ----------------------------------------------------------
        // 5. foreach 循环（推荐用于集合遍历）
        // ----------------------------------------------------------
        Console.WriteLine("\n=== foreach 循环 ===");

        string[] fruits = { "苹果", "香蕉", "橙子", "葡萄" };
        foreach (string fruit in fruits)
        {
            Console.Write($"{fruit} ");
        }
        Console.WriteLine();

        // 带索引的 foreach（C# 没有内置，用 LINQ 的 Select 或 for 循环）
        for (int i = 0; i < fruits.Length; i++)
        {
            Console.WriteLine($"  [{i}] {fruits[i]}");
        }

        // 遍历字典
        var scores = new Dictionary<string, int>
        {
            { "张三", 90 },
            { "李四", 85 },
            { "王五", 92 }
        };

        foreach (var (name, s) in scores)  // 解构 KeyValuePair
        {
            Console.WriteLine($"  {name}: {s}分");
        }

        // ----------------------------------------------------------
        // 6. while 和 do-while
        // ----------------------------------------------------------
        Console.WriteLine("\n=== while / do-while ===");

        // while：先判断再执行
        int count = 0;
        while (count < 3)
        {
            Console.Write($"while:{count} ");
            count++;
        }
        Console.WriteLine();

        // do-while：先执行再判断（至少执行一次）
        int n = 0;
        do
        {
            Console.Write($"do:{n} ");
            n++;
        } while (n < 3);
        Console.WriteLine();

        // ----------------------------------------------------------
        // 7. 循环控制：break / continue
        // ----------------------------------------------------------
        Console.WriteLine("\n=== break / continue ===");

        // break：跳出循环
        for (int i = 0; i < 10; i++)
        {
            if (i == 5) break;
            Console.Write($"{i} ");
        }
        Console.WriteLine("(break at 5)");

        // continue：跳过当前迭代
        for (int i = 0; i < 10; i++)
        {
            if (i % 2 == 0) continue;  // 跳过偶数
            Console.Write($"{i} ");
        }
        Console.WriteLine("(奇数)");

        // ----------------------------------------------------------
        // 8. goto（谨慎使用，破坏代码可读性）
        // ----------------------------------------------------------
        // C# 支持 goto，主要在 switch 中跳转 case

        // ============================================================
        //                      模式匹配（C# 7.0+）
        // ============================================================
        Console.WriteLine("\n=== 模式匹配 ===");

        // is 模式匹配
        object[] objects = { 42, "hello", 3.14, true, null };
        foreach (object o in objects)
        {
            string desc = o switch
            {
                int num when num > 0  => $"正整数 {num}",
                int num               => $"非正整数 {num}",
                string s when s.Length > 3 => $"长字符串 \"{s}\"",
                string s              => $"短字符串 \"{s}\"",
                double d              => $"浮点数 {d}",
                bool b                => $"布尔值 {b}",
                null                  => "空值",
                _                     => $"未知类型 {o.GetType().Name}"
            };
            Console.WriteLine($"  {desc}");
        }

        // 元组模式匹配
        Console.WriteLine("\n=== 元组模式 ===");
        int x = 1, y = -1;
        string quadrant = (x, y) switch
        {
            (> 0, > 0) => "第一象限",
            (< 0, > 0) => "第二象限",
            (< 0, < 0) => "第三象限",
            (> 0, < 0) => "第四象限",
            (0, _) or (_, 0) => "在坐标轴上",
            _ => "原点"
        };
        Console.WriteLine($"({x}, {y}) 在 {quadrant}");

        // ============================================================
        //                      异常处理
        // ============================================================
        Console.WriteLine("\n=== 异常处理 ===");

        // ----------------------------------------------------------
        // 9. try / catch / finally
        // ----------------------------------------------------------
        try
        {
            int[] arr = new int[5];
            arr[10] = 42;  // 会抛出 IndexOutOfRangeException
        }
        catch (IndexOutOfRangeException ex)
        {
            Console.WriteLine($"数组越界: {ex.Message}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"通用异常: {ex.Message}");
        }
        finally
        {
            // 无论是否发生异常都会执行
            // 【用途】释放资源（但更推荐使用 using 语句）
            Console.WriteLine("finally 总是执行");
        }

        // ----------------------------------------------------------
        // 10. using 语句（自动释放资源）
        // ----------------------------------------------------------
        // 推荐用于实现 IDisposable 的对象（文件、数据库连接等）
        // using var file = new System.IO.StreamWriter("test.txt");
        // 超出作用域自动调用 Dispose()

        Console.WriteLine("\n=== 流程控制演示完成 ===");
    }
}
