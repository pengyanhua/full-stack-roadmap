# interfaces.cs

::: info 文件信息
- 📄 原文件：`02_interfaces.cs`
- 🔤 语言：csharp
:::

## 完整代码

```csharp
// ============================================================
//                      接口与泛型
// ============================================================
// 接口（Interface）定义行为契约，实现类必须满足
// 泛型（Generics）让代码适用于多种类型，避免重复
// C# 支持接口默认实现（C# 8.0+）和静态接口成员（C# 11+）

using System;
using System.Collections.Generic;
using System.Linq;

// ============================================================
//                      接口定义
// ============================================================

// ----------------------------------------------------------
// 1. 基本接口
// ----------------------------------------------------------
// 接口只定义"能做什么"，不关心"怎么做"
// 【命名约定】接口以 I 开头（IDisposable, IEnumerable 等）

interface IAnimal
{
    string Name { get; }      // 属性契约
    string Sound { get; }     // 属性契约

    string MakeSound();       // 方法契约

    // 默认实现（C# 8.0+）
    string Describe() => $"{Name} 会发出 \"{Sound}\" 的声音";
}

interface IPet
{
    string OwnerName { get; set; }
    void Play();
}

// 接口继承接口
interface IDomesticAnimal : IAnimal, IPet
{
    bool IsVaccinated { get; set; }
}

// ----------------------------------------------------------
// 2. 接口实现
// ----------------------------------------------------------
// C# 支持多接口实现（弥补单继承的不足）

class Dog : IDomesticAnimal
{
    public string Name { get; }
    public string Sound => "汪汪";
    public string OwnerName { get; set; }
    public bool IsVaccinated { get; set; }

    public Dog(string name, string ownerName)
    {
        Name = name;
        OwnerName = ownerName;
    }

    public string MakeSound() => $"{Name}：{Sound}！";
    public void Play() => Console.WriteLine($"{Name} 正在玩耍！");
}

class Cat : IDomesticAnimal
{
    public string Name { get; }
    public string Sound => "喵喵";
    public string OwnerName { get; set; }
    public bool IsVaccinated { get; set; }

    public Cat(string name, string ownerName)
    {
        Name = name;
        OwnerName = ownerName;
    }

    public string MakeSound() => $"{Name}：{Sound}~";
    public void Play() => Console.WriteLine($"{Name} 优雅地观望...");
}

// ----------------------------------------------------------
// 3. 常用内置接口
// ----------------------------------------------------------

// IComparable<T>：支持排序比较
class Temperature : IComparable<Temperature>
{
    public double Celsius { get; }

    public Temperature(double celsius)
    {
        Celsius = celsius;
    }

    public int CompareTo(Temperature? other)
    {
        if (other is null) return 1;
        return Celsius.CompareTo(other.Celsius);
    }

    public override string ToString() => $"{Celsius}°C";
}

// IEnumerable<T>：支持 foreach 和 LINQ
class NumberRange : System.Collections.Generic.IEnumerable<int>
{
    private readonly int _start;
    private readonly int _end;
    private readonly int _step;

    public NumberRange(int start, int end, int step = 1)
    {
        _start = start;
        _end = end;
        _step = step;
    }

    public System.Collections.Generic.IEnumerator<int> GetEnumerator()
    {
        for (int i = _start; i <= _end; i += _step)
            yield return i;  // yield return 懒求值
    }

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        => GetEnumerator();
}

// ============================================================
//                      泛型
// ============================================================

// ----------------------------------------------------------
// 4. 泛型类
// ----------------------------------------------------------
// 【优势】类型安全 + 代码复用，避免装箱拆箱性能损耗

class Stack<T>
{
    private readonly List<T> _items = new();

    public int Count => _items.Count;
    public bool IsEmpty => _items.Count == 0;

    public void Push(T item)
    {
        _items.Add(item);
    }

    public T Pop()
    {
        if (IsEmpty) throw new InvalidOperationException("栈为空");
        T item = _items[^1];  // C# 8.0+ 索引运算符 ^1 表示最后一个
        _items.RemoveAt(_items.Count - 1);
        return item;
    }

    public T Peek()
    {
        if (IsEmpty) throw new InvalidOperationException("栈为空");
        return _items[^1];
    }
}

// ----------------------------------------------------------
// 5. 泛型约束（Constraints）
// ----------------------------------------------------------
// where T : 约束类型
// 常用约束：
//   class     — T 必须是引用类型
//   struct    — T 必须是值类型
//   new()     — T 必须有无参构造函数
//   接口名    — T 必须实现指定接口
//   基类名    — T 必须继承自基类
//   notnull   — T 不能为 null（C# 8.0+）

class Repository<T> where T : class, new()
{
    private readonly List<T> _store = new();

    public void Add(T item) => _store.Add(item);
    public IReadOnlyList<T> GetAll() => _store.AsReadOnly();
    public int Count => _store.Count;
}

// 多约束泛型方法
static class GenericUtils
{
    // T 必须实现 IComparable<T>
    public static T Clamp<T>(T value, T min, T max) where T : IComparable<T>
    {
        if (value.CompareTo(min) < 0) return min;
        if (value.CompareTo(max) > 0) return max;
        return value;
    }

    // 两个类型参数，各自有约束
    public static TResult Transform<TInput, TResult>(
        TInput input,
        Func<TInput, TResult> transformer)
    {
        return transformer(input);
    }
}

// ============================================================
//                      主程序
// ============================================================
class InterfacesAndGenerics
{
    static void Main()
    {
        Console.WriteLine("=== 接口 ===");

        var dog = new Dog("旺财", "张三") { IsVaccinated = true };
        var cat = new Cat("咪咪", "李四") { IsVaccinated = false };

        // 使用接口类型引用
        IAnimal[] animals = { dog, cat };
        foreach (IAnimal animal in animals)
        {
            Console.WriteLine(animal.MakeSound());
            Console.WriteLine(animal.Describe());  // 使用接口默认实现
        }

        // 多接口使用
        Console.WriteLine("\n=== 多接口 ===");
        IDomesticAnimal[] pets = { dog, cat };
        foreach (var pet in pets)
        {
            pet.Play();
            Console.WriteLine($"  主人: {pet.OwnerName}, 已接种: {pet.IsVaccinated}");
        }

        // IComparable - 排序
        Console.WriteLine("\n=== IComparable 排序 ===");
        var temps = new List<Temperature>
        {
            new(36.5), new(38.2), new(35.0), new(37.1)
        };
        temps.Sort();
        Console.WriteLine($"排序后: {string.Join(", ", temps)}");
        Console.WriteLine($"最高: {temps.Max()}, 最低: {temps.Min()}");

        // IEnumerable - 自定义可迭代类型
        Console.WriteLine("\n=== 自定义 IEnumerable ===");
        var range = new NumberRange(1, 20, 3);
        foreach (int n in range)
            Console.Write($"{n} ");
        Console.WriteLine();

        // LINQ 可以直接用于自定义 IEnumerable
        var bigNums = range.Where(n => n > 10).ToList();
        Console.WriteLine($"大于10: [{string.Join(", ", bigNums)}]");

        // ----------------------------------------------------------
        // 泛型类
        // ----------------------------------------------------------
        Console.WriteLine("\n=== 泛型栈 ===");

        var intStack = new Stack<int>();
        var strStack = new Stack<string>();

        intStack.Push(1);
        intStack.Push(2);
        intStack.Push(3);
        Console.WriteLine($"int 栈顶: {intStack.Peek()}, 数量: {intStack.Count}");
        Console.WriteLine($"弹出: {intStack.Pop()}, {intStack.Pop()}");

        strStack.Push("Hello");
        strStack.Push("World");
        Console.WriteLine($"string 栈顶: {strStack.Peek()}");

        // 泛型约束
        Console.WriteLine("\n=== 泛型约束 ===");
        Console.WriteLine($"Clamp(15, 1, 10) = {GenericUtils.Clamp(15, 1, 10)}");
        Console.WriteLine($"Clamp(5, 1, 10) = {GenericUtils.Clamp(5, 1, 10)}");
        Console.WriteLine($"Clamp(-3, 1, 10) = {GenericUtils.Clamp(-3, 1, 10)}");

        string upper = GenericUtils.Transform("hello", s => s.ToUpper());
        int length = GenericUtils.Transform("hello", s => s.Length);
        Console.WriteLine($"Transform: {upper}, {length}");

        // ----------------------------------------------------------
        // 扩展方法
        // ----------------------------------------------------------
        Console.WriteLine("\n=== 扩展方法 ===");
        // 为 string 类型添加自定义方法（无需修改原类）
        string sentence = "hello world, welcome to c#!";
        Console.WriteLine(sentence.ToTitleCase());
        Console.WriteLine("  是否回文: " + "racecar".IsPalindrome());
        Console.WriteLine("  是否回文: " + "hello".IsPalindrome());

        int[] arr = { 3, 1, 4, 1, 5, 9, 2, 6 };
        Console.WriteLine($"中位数: {arr.Median():F1}");
    }
}

// ----------------------------------------------------------
// 扩展方法（必须在静态类中定义）
// ----------------------------------------------------------
static class StringExtensions
{
    // 第一个参数用 this 关键字
    public static string ToTitleCase(this string s)
    {
        if (string.IsNullOrEmpty(s)) return s;
        var words = s.Split(' ');
        for (int i = 0; i < words.Length; i++)
        {
            if (words[i].Length > 0)
                words[i] = char.ToUpper(words[i][0]) + words[i][1..];
        }
        return string.Join(' ', words);
    }

    public static bool IsPalindrome(this string s)
    {
        s = s.ToLower();
        int left = 0, right = s.Length - 1;
        while (left < right)
        {
            if (s[left] != s[right]) return false;
            left++;
            right--;
        }
        return true;
    }
}

static class ArrayExtensions
{
    public static double Median(this int[] arr)
    {
        var sorted = arr.OrderBy(x => x).ToArray();
        int mid = sorted.Length / 2;
        return sorted.Length % 2 == 0
            ? (sorted[mid - 1] + sorted[mid]) / 2.0
            : sorted[mid];
    }
}
```
