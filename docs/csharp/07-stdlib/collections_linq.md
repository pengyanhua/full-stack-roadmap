# collections linq.cs

::: info 文件信息
- 📄 原文件：`01_collections_linq.cs`
- 🔤 语言：csharp
:::

## 完整代码

```csharp
// ============================================================
//                      集合与 LINQ
// ============================================================
// .NET 集合框架提供了丰富的数据结构
// LINQ（Language Integrated Query）统一了集合、数据库、XML 的查询语法
// 推荐使用泛型集合（System.Collections.Generic）而非非泛型集合

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;

class CollectionsAndLinq
{
    static void Main()
    {
        // ============================================================
        //                      常用集合类型
        // ============================================================
        Console.WriteLine("=== List<T> ===");

        // ----------------------------------------------------------
        // 1. List<T> — 动态数组
        // ----------------------------------------------------------
        var list = new List<int> { 3, 1, 4, 1, 5, 9, 2, 6, 5 };

        list.Add(7);
        list.AddRange(new[] { 10, 11 });
        list.Insert(0, 0);         // 在索引 0 处插入
        list.Remove(1);            // 删除第一个值为 1 的元素
        list.RemoveAt(0);          // 删除索引 0 处的元素

        Console.WriteLine($"Count: {list.Count}");
        Console.WriteLine($"包含5: {list.Contains(5)}");
        Console.WriteLine($"索引of9: {list.IndexOf(9)}");

        list.Sort();
        Console.WriteLine($"排序: [{string.Join(", ", list)}]");

        list.Sort((a, b) => b.CompareTo(a));  // 降序
        Console.WriteLine($"降序: [{string.Join(", ", list)}]");

        // 切片（C# 8.0+ Ranges）
        var slice = list[1..4];  // 索引 1, 2, 3
        Console.WriteLine($"切片[1..4]: [{string.Join(", ", slice)}]");

        // ----------------------------------------------------------
        // 2. Dictionary<K, V> — 哈希映射
        // ----------------------------------------------------------
        Console.WriteLine("\n=== Dictionary<K,V> ===");

        var scores = new Dictionary<string, int>
        {
            ["张三"] = 90,
            ["李四"] = 85,
            ["王五"] = 92
        };

        // 添加 / 更新
        scores["赵六"] = 78;
        scores["张三"] = 95;  // 更新已有 key

        // 安全读取（避免 KeyNotFoundException）
        if (scores.TryGetValue("钱七", out int score))
            Console.WriteLine($"钱七: {score}");
        else
            Console.WriteLine("钱七不存在");

        int val = scores.GetValueOrDefault("不存在", -1);
        Console.WriteLine($"GetValueOrDefault: {val}");

        // 遍历
        foreach (var (name, s) in scores.OrderByDescending(kv => kv.Value))
            Console.WriteLine($"  {name}: {s}");

        // ----------------------------------------------------------
        // 3. HashSet<T> — 集合（不重复）
        // ----------------------------------------------------------
        Console.WriteLine("\n=== HashSet<T> ===");

        var set1 = new HashSet<int> { 1, 2, 3, 4, 5 };
        var set2 = new HashSet<int> { 4, 5, 6, 7, 8 };

        var union = new HashSet<int>(set1);
        union.UnionWith(set2);
        Console.WriteLine($"并集: {{{string.Join(", ", union.OrderBy(x => x))}}}");

        var intersect = new HashSet<int>(set1);
        intersect.IntersectWith(set2);
        Console.WriteLine($"交集: {{{string.Join(", ", intersect)}}}");

        var except = new HashSet<int>(set1);
        except.ExceptWith(set2);
        Console.WriteLine($"差集(set1-set2): {{{string.Join(", ", except)}}}");

        // 去重
        var withDups = new[] { 1, 2, 2, 3, 3, 3, 4 };
        var unique = new HashSet<int>(withDups);
        Console.WriteLine($"去重: [{string.Join(", ", unique)}]");

        // ----------------------------------------------------------
        // 4. Queue<T> 和 Stack<T>
        // ----------------------------------------------------------
        Console.WriteLine("\n=== Queue / Stack ===");

        var queue = new Queue<string>();
        queue.Enqueue("第一");
        queue.Enqueue("第二");
        queue.Enqueue("第三");

        Console.WriteLine($"队列头: {queue.Peek()}");
        while (queue.Count > 0)
            Console.Write($"{queue.Dequeue()} ");
        Console.WriteLine("(FIFO)");

        var stack = new Stack<int>();
        stack.Push(1); stack.Push(2); stack.Push(3);

        Console.WriteLine($"栈顶: {stack.Peek()}");
        while (stack.Count > 0)
            Console.Write($"{stack.Pop()} ");
        Console.WriteLine("(LIFO)");

        // ----------------------------------------------------------
        // 5. LinkedList<T>（双向链表）
        // ----------------------------------------------------------
        Console.WriteLine("\n=== LinkedList<T> ===");

        var linked = new LinkedList<int>();
        linked.AddLast(1);
        linked.AddLast(3);
        linked.AddLast(5);
        linked.AddFirst(0);  // 头部插入

        var node = linked.Find(3);
        if (node != null)
            linked.AddBefore(node, 2);  // 在 3 前插入 2

        Console.WriteLine($"链表: [{string.Join(", ", linked)}]");

        // ============================================================
        //                      LINQ 查询
        // ============================================================
        Console.WriteLine("\n=== LINQ 基础 ===");

        var people = new List<(string Name, int Age, string City, double Salary)>
        {
            ("张三", 28, "北京", 15000),
            ("李四", 35, "上海", 25000),
            ("王五", 22, "北京", 8000),
            ("赵六", 42, "广州", 30000),
            ("钱七", 31, "上海", 20000),
            ("周八", 25, "北京", 12000),
        };

        // ----------------------------------------------------------
        // 基本操作：Where, Select, OrderBy
        // ----------------------------------------------------------
        var beijingPeople = people
            .Where(p => p.City == "北京")
            .OrderBy(p => p.Age)
            .Select(p => $"{p.Name}({p.Age}岁,¥{p.Salary:N0})");

        Console.WriteLine("北京（按年龄排序）:");
        foreach (var s in beijingPeople)
            Console.WriteLine($"  {s}");

        // ----------------------------------------------------------
        // 聚合操作
        // ----------------------------------------------------------
        Console.WriteLine("\n=== LINQ 聚合 ===");

        double avgSalary = people.Average(p => p.Salary);
        double maxSalary = people.Max(p => p.Salary);
        double minSalary = people.Min(p => p.Salary);
        double totalSalary = people.Sum(p => p.Salary);
        int count = people.Count(p => p.Salary > 15000);

        Console.WriteLine($"平均薪资: {avgSalary:N0}");
        Console.WriteLine($"最高薪资: {maxSalary:N0}");
        Console.WriteLine($"最低薪资: {minSalary:N0}");
        Console.WriteLine($"薪资合计: {totalSalary:N0}");
        Console.WriteLine($"高薪人数(>15k): {count}");

        // ----------------------------------------------------------
        // 分组操作
        // ----------------------------------------------------------
        Console.WriteLine("\n=== LINQ 分组 ===");

        var cityGroups = people
            .GroupBy(p => p.City)
            .Select(g => new
            {
                City = g.Key,
                Count = g.Count(),
                AvgAge = g.Average(p => p.Age),
                AvgSalary = g.Average(p => p.Salary)
            })
            .OrderByDescending(x => x.AvgSalary);

        foreach (var g in cityGroups)
            Console.WriteLine($"  {g.City}: {g.Count}人, 平均年龄{g.AvgAge:F0}岁, 平均薪资{g.AvgSalary:N0}");

        // ----------------------------------------------------------
        // 集合操作
        // ----------------------------------------------------------
        Console.WriteLine("\n=== LINQ 集合操作 ===");

        var nums1 = new[] { 1, 2, 3, 4, 5 };
        var nums2 = new[] { 3, 4, 5, 6, 7 };

        Console.WriteLine($"Union: [{string.Join(", ", nums1.Union(nums2))}]");
        Console.WriteLine($"Intersect: [{string.Join(", ", nums1.Intersect(nums2))}]");
        Console.WriteLine($"Except: [{string.Join(", ", nums1.Except(nums2))}]");
        Console.WriteLine($"Concat: [{string.Join(", ", nums1.Concat(nums2))}]");

        // ----------------------------------------------------------
        // 分页
        // ----------------------------------------------------------
        Console.WriteLine("\n=== LINQ 分页 ===");
        int pageSize = 2, pageIndex = 1;
        var paged = people
            .OrderBy(p => p.Name)
            .Skip(pageIndex * pageSize)
            .Take(pageSize)
            .Select(p => p.Name);
        Console.WriteLine($"第{pageIndex + 1}页: [{string.Join(", ", paged)}]");

        // ----------------------------------------------------------
        // 查询语法（SQL 风格）
        // ----------------------------------------------------------
        Console.WriteLine("\n=== 查询语法 ===");

        var query = from p in people
                    where p.Salary > 15000
                    orderby p.Salary descending
                    select new { p.Name, p.City, p.Salary };

        foreach (var p in query)
            Console.WriteLine($"  {p.Name} ({p.City}): {p.Salary:N0}");

        // ----------------------------------------------------------
        // ImmutableList（不可变集合）
        // ----------------------------------------------------------
        Console.WriteLine("\n=== 不可变集合 ===");

        var immutableList = ImmutableList.Create(1, 2, 3, 4, 5);
        var added = immutableList.Add(6);         // 返回新列表，原列表不变
        var removed = immutableList.Remove(3);    // 返回新列表

        Console.WriteLine($"原始: [{string.Join(", ", immutableList)}]");
        Console.WriteLine($"添加后: [{string.Join(", ", added)}]");
        Console.WriteLine($"删除后: [{string.Join(", ", removed)}]");

        // ----------------------------------------------------------
        // StringBuilder（高效字符串拼接）
        // ----------------------------------------------------------
        Console.WriteLine("\n=== StringBuilder ===");

        var sb = new StringBuilder();
        sb.Append("Hello");
        sb.Append(", ");
        sb.AppendLine("World!");
        sb.AppendFormat("今天是 {0:yyyy-MM-dd}", DateTime.Now);
        sb.Insert(0, "[开始] ");

        Console.WriteLine(sb.ToString());
        Console.WriteLine($"长度: {sb.Length}");

        Console.WriteLine("\n=== 集合与 LINQ 演示完成 ===");
    }
}
```
