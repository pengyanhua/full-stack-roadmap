// ============================================================
//                      命名空间与包管理
// ============================================================
// C# 使用命名空间（Namespace）组织代码
// NuGet 是 .NET 的包管理系统（类似 npm、pip）
// 全局 using（C# 10+）和隐式 using 减少样板代码
// .csproj 项目文件管理依赖和配置

// 全局 using（C# 10+，通常放在 GlobalUsings.cs 中）
// global using System;
// global using System.Collections.Generic;

// 命名空间导入
using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

// ============================================================
//                      自定义命名空间
// ============================================================

namespace MyApp.Models
{
    // ----------------------------------------------------------
    // 1. 命名空间内定义类型
    // ----------------------------------------------------------
    public class User
    {
        public int Id { get; set; }
        public string Name { get; set; } = "";
        public string Email { get; set; } = "";
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        // JsonPropertyName 控制 JSON 序列化的字段名
        [JsonPropertyName("display_name")]
        public string DisplayName => $"{Name} <{Email}>";
    }

    public record Product(int Id, string Name, decimal Price, string Category);
}

namespace MyApp.Services
{
    using MyApp.Models;

    // ----------------------------------------------------------
    // 2. 服务类（依赖注入风格）
    // ----------------------------------------------------------
    public class UserService
    {
        private readonly List<User> _users = new();
        private int _nextId = 1;

        public User Create(string name, string email)
        {
            var user = new User { Id = _nextId++, Name = name, Email = email };
            _users.Add(user);
            return user;
        }

        public User? FindById(int id) => _users.FirstOrDefault(u => u.Id == id);
        public IReadOnlyList<User> GetAll() => _users.AsReadOnly();
        public int Count => _users.Count;
    }

    public class ProductService
    {
        private readonly List<Product> _products = new();
        private int _nextId = 1;

        public Product Add(string name, decimal price, string category)
        {
            var product = new Product(_nextId++, name, price, category);
            _products.Add(product);
            return product;
        }

        public IEnumerable<Product> GetByCategory(string category)
            => _products.Where(p => p.Category == category);

        public IEnumerable<Product> GetAll() => _products;
    }
}

// ============================================================
//                      主程序
// ============================================================

using MyApp.Models;
using MyApp.Services;

class NamespacesAndPackages
{
    static void Main()
    {
        Console.WriteLine("=== 命名空间 ===");

        var userService = new UserService();
        userService.Create("张三", "zhangsan@example.com");
        userService.Create("李四", "lisi@example.com");
        userService.Create("王五", "wangwu@example.com");

        Console.WriteLine($"用户数量: {userService.Count}");
        foreach (var user in userService.GetAll())
            Console.WriteLine($"  [{user.Id}] {user.DisplayName}");

        // ----------------------------------------------------------
        // JSON 序列化（System.Text.Json，内置 NuGet 包）
        // ----------------------------------------------------------
        Console.WriteLine("\n=== JSON 序列化 ===");

        var user = userService.FindById(1)!;

        // 序列化（对象 -> JSON）
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        };
        string json = JsonSerializer.Serialize(user, options);
        Console.WriteLine("序列化结果:");
        Console.WriteLine(json);

        // 反序列化（JSON -> 对象）
        string jsonStr = """
            {
                "id": 99,
                "name": "新用户",
                "email": "new@example.com",
                "createdAt": "2026-01-01T00:00:00Z"
            }
            """;
        User? deserializedUser = JsonSerializer.Deserialize<User>(jsonStr,
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        Console.WriteLine($"\n反序列化: {deserializedUser?.DisplayName}");

        // 序列化集合
        var products = new List<Product>
        {
            new(1, "MacBook Pro", 14999.00m, "电脑"),
            new(2, "iPhone 15", 7999.00m, "手机"),
            new(3, "iPad Air", 4999.00m, "平板"),
        };
        string productsJson = JsonSerializer.Serialize(products, options);
        Console.WriteLine($"\n产品列表 JSON（{products.Count} 项）:");
        Console.WriteLine(productsJson);

        // ----------------------------------------------------------
        // 文件操作（System.IO）
        // ----------------------------------------------------------
        Console.WriteLine("\n=== 文件操作 ===");

        string tempFile = Path.GetTempFileName();
        try
        {
            // 写文件
            File.WriteAllText(tempFile, json);
            Console.WriteLine($"已写入文件: {Path.GetFileName(tempFile)}");

            // 读文件
            string content = File.ReadAllText(tempFile);
            User? loaded = JsonSerializer.Deserialize<User>(content, options);
            Console.WriteLine($"从文件加载: {loaded?.Name}");

            // 读取所有行
            File.WriteAllLines(tempFile, new[] { "行1", "行2", "行3" });
            string[] lines = File.ReadAllLines(tempFile);
            Console.WriteLine($"文件行数: {lines.Length}");
            foreach (string line in lines)
                Console.WriteLine($"  {line}");
        }
        finally
        {
            File.Delete(tempFile);  // 清理临时文件
        }

        // Path 操作
        Console.WriteLine("\n=== Path 操作 ===");
        string filePath = @"C:\Projects\MyApp\src\Services\UserService.cs";
        Console.WriteLine($"目录: {Path.GetDirectoryName(filePath)}");
        Console.WriteLine($"文件名: {Path.GetFileName(filePath)}");
        Console.WriteLine($"扩展名: {Path.GetExtension(filePath)}");
        Console.WriteLine($"无扩展名: {Path.GetFileNameWithoutExtension(filePath)}");
        Console.WriteLine($"组合: {Path.Combine("C:", "Projects", "MyApp", "README.md")}");

        // ----------------------------------------------------------
        // 反射（System.Reflection）
        // ----------------------------------------------------------
        Console.WriteLine("\n=== 反射 ===");

        Type userType = typeof(User);
        Console.WriteLine($"类型名: {userType.Name}");
        Console.WriteLine($"命名空间: {userType.Namespace}");

        Console.WriteLine("属性列表:");
        foreach (PropertyInfo prop in userType.GetProperties())
        {
            Console.WriteLine($"  {prop.PropertyType.Name} {prop.Name}");
        }

        // 动态创建实例和调用方法
        object instance = Activator.CreateInstance(userType)!;
        userType.GetProperty("Name")!.SetValue(instance, "动态创建");
        userType.GetProperty("Email")!.SetValue(instance, "dynamic@example.com");

        string? name = (string?)userType.GetProperty("Name")!.GetValue(instance);
        Console.WriteLine($"动态设置 Name: {name}");

        // ----------------------------------------------------------
        // NuGet 包使用说明
        // ----------------------------------------------------------
        Console.WriteLine("\n=== NuGet 包管理 ===");
        Console.WriteLine("常用 NuGet 命令:");
        Console.WriteLine("  dotnet add package Newtonsoft.Json");
        Console.WriteLine("  dotnet add package Microsoft.EntityFrameworkCore");
        Console.WriteLine("  dotnet add package Serilog");
        Console.WriteLine("  dotnet remove package <包名>");
        Console.WriteLine("  dotnet restore（恢复依赖）");
        Console.WriteLine("  dotnet list package（列出已安装的包）");

        Console.WriteLine("\n.csproj 依赖示例:");
        Console.WriteLine("""
            <ItemGroup>
              <PackageReference Include="Newtonsoft.Json" Version="13.0.3" />
              <PackageReference Include="Serilog" Version="3.1.1" />
              <PackageReference Include="Microsoft.EntityFrameworkCore.SqlServer" Version="8.0.0" />
            </ItemGroup>
            """);
    }
}
