# unit testing.cs

::: info 文件信息
- 📄 原文件：`01_unit_testing.cs`
- 🔤 语言：csharp
:::

## 完整代码

```csharp
// ============================================================
//                      单元测试
// ============================================================
// C# 主流测试框架：xUnit（推荐）、NUnit、MSTest
// xUnit：[Fact] 单个测试，[Theory] 参数化测试
// FluentAssertions：更可读的断言语法
// Moq：Mock 对象，隔离外部依赖
//
// 安装依赖：
//   dotnet add package xunit
//   dotnet add package xunit.runner.visualstudio
//   dotnet add package FluentAssertions
//   dotnet add package Moq
// 运行测试：dotnet test

using System;
using System.Collections.Generic;
using System.Linq;

// ============================================================
//                      被测试的生产代码
// ============================================================

// ----------------------------------------------------------
// 1. 简单计算器（用于演示基本单元测试）
// ----------------------------------------------------------
class Calculator
{
    public int Add(int a, int b) => a + b;
    public int Subtract(int a, int b) => a - b;
    public int Multiply(int a, int b) => a * b;

    public double Divide(double a, double b)
    {
        if (b == 0)
            throw new DivideByZeroException("除数不能为零");
        return a / b;
    }

    public double Power(double base_, double exponent) => Math.Pow(base_, exponent);
}

// ----------------------------------------------------------
// 2. 字符串工具（用于演示边界条件测试）
// ----------------------------------------------------------
class StringUtils
{
    public string Reverse(string s)
    {
        if (s is null) throw new ArgumentNullException(nameof(s));
        return new string(s.Reverse().ToArray());
    }

    public bool IsPalindrome(string s)
    {
        if (s is null) throw new ArgumentNullException(nameof(s));
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

    public string[] SplitWords(string sentence)
    {
        if (string.IsNullOrWhiteSpace(sentence))
            return Array.Empty<string>();
        return sentence.Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }
}

// ----------------------------------------------------------
// 3. 订单服务（演示依赖注入和 Mock）
// ----------------------------------------------------------
interface IProductRepository
{
    decimal GetPrice(string productId);
    bool Exists(string productId);
}

interface IOrderLogger
{
    void LogOrder(string orderId, decimal total);
}

class OrderService
{
    private readonly IProductRepository _repository;
    private readonly IOrderLogger _logger;

    public OrderService(IProductRepository repository, IOrderLogger logger)
    {
        _repository = repository;
        _logger = logger;
    }

    public decimal CalculateTotal(Dictionary<string, int> items)
    {
        decimal total = 0;
        foreach (var (productId, quantity) in items)
        {
            if (!_repository.Exists(productId))
                throw new ArgumentException($"商品 {productId} 不存在");
            total += _repository.GetPrice(productId) * quantity;
        }
        return total;
    }

    public string PlaceOrder(Dictionary<string, int> items)
    {
        decimal total = CalculateTotal(items);
        string orderId = $"ORD-{DateTime.UtcNow:yyyyMMddHHmmss}";
        _logger.LogOrder(orderId, total);
        return orderId;
    }
}

// ============================================================
//                      测试代码演示
// ============================================================
// 注意：以下是模拟 xUnit 测试框架的演示写法
// 实际测试需要独立的 Test 项目和 xUnit/NUnit 框架

class TestRunner
{
    // 简单的测试结果记录
    private static int _passed = 0;
    private static int _failed = 0;
    private static List<string> _failures = new();

    static void Assert(bool condition, string testName)
    {
        if (condition)
        {
            _passed++;
            Console.WriteLine($"  ✓ {testName}");
        }
        else
        {
            _failed++;
            _failures.Add(testName);
            Console.WriteLine($"  ✗ {testName}");
        }
    }

    static void AssertEqual<T>(T expected, T actual, string testName)
    {
        bool condition = EqualityComparer<T>.Default.Equals(expected, actual);
        if (condition)
        {
            _passed++;
            Console.WriteLine($"  ✓ {testName}");
        }
        else
        {
            _failed++;
            _failures.Add($"{testName} (期望: {expected}, 实际: {actual})");
            Console.WriteLine($"  ✗ {testName} (期望: {expected}, 实际: {actual})");
        }
    }

    static void AssertThrows<TException>(Action action, string testName) where TException : Exception
    {
        try
        {
            action();
            _failed++;
            _failures.Add($"{testName} (未抛出 {typeof(TException).Name})");
            Console.WriteLine($"  ✗ {testName} (未抛出预期异常)");
        }
        catch (TException)
        {
            _passed++;
            Console.WriteLine($"  ✓ {testName}");
        }
        catch (Exception ex)
        {
            _failed++;
            _failures.Add($"{testName} (抛出了错误类型 {ex.GetType().Name})");
            Console.WriteLine($"  ✗ {testName} (错误异常类型)");
        }
    }

    static void Main()
    {
        Console.WriteLine("=== 单元测试演示 ===\n");

        // ----------------------------------------------------------
        // Calculator 测试
        // ----------------------------------------------------------
        Console.WriteLine("Calculator 测试:");
        var calc = new Calculator();

        AssertEqual(8, calc.Add(3, 5), "Add(3,5) = 8");
        AssertEqual(0, calc.Add(0, 0), "Add(0,0) = 0");
        AssertEqual(-2, calc.Add(-5, 3), "Add(-5,3) = -2");
        AssertEqual(5, calc.Subtract(8, 3), "Subtract(8,3) = 5");
        AssertEqual(6, calc.Multiply(2, 3), "Multiply(2,3) = 6");
        AssertEqual(0, calc.Multiply(0, 100), "Multiply(0,100) = 0");

        double divResult = calc.Divide(10, 4);
        Assert(Math.Abs(divResult - 2.5) < 1e-10, "Divide(10,4) = 2.5");

        AssertThrows<DivideByZeroException>(() => calc.Divide(5, 0), "Divide by zero throws");

        // 参数化测试（等价于 [Theory] [InlineData]）
        Console.WriteLine("\n  参数化测试 - 加法:");
        var addTestCases = new[] { (1, 2, 3), (0, 0, 0), (-1, 1, 0), (100, -50, 50) };
        foreach (var (a, b, expected) in addTestCases)
        {
            AssertEqual(expected, calc.Add(a, b), $"Add({a},{b}) = {expected}");
        }

        // ----------------------------------------------------------
        // StringUtils 测试
        // ----------------------------------------------------------
        Console.WriteLine("\nStringUtils 测试:");
        var utils = new StringUtils();

        AssertEqual("olleH", utils.Reverse("Hello"), "Reverse('Hello')");
        AssertEqual("", utils.Reverse(""), "Reverse('')");
        AssertEqual("a", utils.Reverse("a"), "Reverse('a')");

        Assert(utils.IsPalindrome("racecar"), "IsPalindrome('racecar')");
        Assert(utils.IsPalindrome("A"), "IsPalindrome('A')");
        Assert(!utils.IsPalindrome("hello"), "!IsPalindrome('hello')");
        Assert(utils.IsPalindrome(""), "IsPalindrome('')");

        AssertThrows<ArgumentNullException>(() => utils.Reverse(null!), "Reverse(null) throws");

        // SplitWords
        string[] words = utils.SplitWords("Hello World C#");
        AssertEqual(3, words.Length, "SplitWords 返回3个词");
        AssertEqual("Hello", words[0], "SplitWords[0] = 'Hello'");

        string[] empty = utils.SplitWords("  ");
        AssertEqual(0, empty.Length, "SplitWords('  ') = 空数组");

        // ----------------------------------------------------------
        // 测试隔离（Mock 模式）
        // ----------------------------------------------------------
        Console.WriteLine("\nOrderService 测试（使用测试替身）:");

        // 用测试替身替代真实依赖
        var mockRepo = new MockProductRepository();
        var mockLogger = new MockOrderLogger();
        var orderService = new OrderService(mockRepo, mockLogger);

        // 场景1：正常下单
        var items = new Dictionary<string, int> { { "P001", 2 }, { "P002", 1 } };
        decimal total = orderService.CalculateTotal(items);
        AssertEqual(299.0m, total, "订单总价 = 299 元");

        string orderId = orderService.PlaceOrder(items);
        Assert(orderId.StartsWith("ORD-"), "订单ID格式正确");
        AssertEqual(1, mockLogger.LoggedOrders.Count, "日志记录了1条订单");
        AssertEqual(299.0m, mockLogger.LoggedOrders[0].Total, "日志金额正确");

        // 场景2：商品不存在
        var badItems = new Dictionary<string, int> { { "INVALID", 1 } };
        AssertThrows<ArgumentException>(
            () => orderService.CalculateTotal(badItems),
            "无效商品抛出 ArgumentException");

        // ----------------------------------------------------------
        // 测试结果汇总
        // ----------------------------------------------------------
        Console.WriteLine("\n" + new string('=', 50));
        Console.WriteLine($"测试结果: {_passed} 通过 / {_failed} 失败");
        if (_failures.Any())
        {
            Console.WriteLine("失败详情:");
            foreach (var f in _failures)
                Console.WriteLine($"  - {f}");
        }
        else
        {
            Console.WriteLine("所有测试通过！");
        }

        // ----------------------------------------------------------
        // xUnit 实际写法示例（注释形式展示）
        // ----------------------------------------------------------
        Console.WriteLine("\n=== xUnit 实际写法 ===");
        Console.WriteLine("""
            // 测试类
            public class CalculatorTests
            {
                private readonly Calculator _calc = new();

                [Fact]
                public void Add_TwoPositiveNumbers_ReturnsSum()
                {
                    var result = _calc.Add(3, 5);
                    result.Should().Be(8);  // FluentAssertions
                }

                [Theory]
                [InlineData(1, 2, 3)]
                [InlineData(-1, 1, 0)]
                [InlineData(0, 0, 0)]
                public void Add_VariousInputs_ReturnsCorrectSum(int a, int b, int expected)
                {
                    _calc.Add(a, b).Should().Be(expected);
                }

                [Fact]
                public void Divide_ByZero_ThrowsDivideByZeroException()
                {
                    Action act = () => _calc.Divide(10, 0);
                    act.Should().Throw<DivideByZeroException>();
                }
            }
            """);
    }
}

// ----------------------------------------------------------
// 测试替身（Test Doubles）
// ----------------------------------------------------------
class MockProductRepository : IProductRepository
{
    private readonly Dictionary<string, decimal> _prices = new()
    {
        { "P001", 99.50m },
        { "P002", 100.00m },
        { "P003", 49.99m },
    };

    public decimal GetPrice(string productId) =>
        _prices.TryGetValue(productId, out decimal price) ? price : 0;

    public bool Exists(string productId) => _prices.ContainsKey(productId);
}

class MockOrderLogger : IOrderLogger
{
    public List<(string OrderId, decimal Total)> LoggedOrders { get; } = new();

    public void LogOrder(string orderId, decimal total)
    {
        LoggedOrders.Add((orderId, total));
    }
}
```
