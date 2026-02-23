// ============================================================
//                      项目实战：命令行 Todo 应用
// ============================================================
// 综合运用 C# 特性：类、接口、LINQ、异步、JSON、文件 IO
// 功能：添加/完成/删除/列出/搜索 Todo 项
// 数据持久化：JSON 文件存储
// 运行：dotnet run -- add "学习 C#"
//       dotnet run -- list
//       dotnet run -- done 1
//       dotnet run -- delete 1
//       dotnet run -- search "C#"

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

// ============================================================
//                      数据模型
// ============================================================

enum Priority { Low, Medium, High }

class TodoItem
{
    public int Id { get; set; }
    public string Title { get; set; } = "";
    public bool IsCompleted { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.Now;
    public DateTime? CompletedAt { get; set; }
    public Priority Priority { get; set; } = Priority.Medium;

    [JsonIgnore]
    public string StatusIcon => IsCompleted ? "✓" : "○";

    [JsonIgnore]
    public string PriorityIcon => Priority switch
    {
        Priority.High => "🔴",
        Priority.Medium => "🟡",
        Priority.Low => "🟢",
        _ => "⚪"
    };

    public override string ToString()
    {
        string timeStr = IsCompleted && CompletedAt.HasValue
            ? $"完成于 {CompletedAt:MM-dd HH:mm}"
            : $"创建于 {CreatedAt:MM-dd HH:mm}";
        return $"[{Id:D3}] {StatusIcon} {PriorityIcon} {Title} ({timeStr})";
    }
}

// ============================================================
//                      存储层
// ============================================================

interface ITodoStorage
{
    Task<List<TodoItem>> LoadAsync();
    Task SaveAsync(List<TodoItem> todos);
}

class JsonFileStorage : ITodoStorage
{
    private readonly string _filePath;
    private readonly JsonSerializerOptions _options;

    public JsonFileStorage(string filePath)
    {
        _filePath = filePath;
        _options = new JsonSerializerOptions
        {
            WriteIndented = true,
            Converters = { new JsonStringEnumConverter() }
        };
    }

    public async Task<List<TodoItem>> LoadAsync()
    {
        if (!File.Exists(_filePath))
            return new List<TodoItem>();

        string json = await File.ReadAllTextAsync(_filePath);
        return JsonSerializer.Deserialize<List<TodoItem>>(json, _options)
               ?? new List<TodoItem>();
    }

    public async Task SaveAsync(List<TodoItem> todos)
    {
        string json = JsonSerializer.Serialize(todos, _options);
        await File.WriteAllTextAsync(_filePath, json);
    }
}

// ============================================================
//                      业务逻辑层
// ============================================================

class TodoService
{
    private readonly ITodoStorage _storage;
    private List<TodoItem> _todos = new();
    private int _nextId = 1;

    public TodoService(ITodoStorage storage)
    {
        _storage = storage;
    }

    public async Task InitializeAsync()
    {
        _todos = await _storage.LoadAsync();
        _nextId = _todos.Count > 0 ? _todos.Max(t => t.Id) + 1 : 1;
    }

    public async Task<TodoItem> AddAsync(string title, Priority priority = Priority.Medium)
    {
        var item = new TodoItem
        {
            Id = _nextId++,
            Title = title,
            Priority = priority
        };
        _todos.Add(item);
        await _storage.SaveAsync(_todos);
        return item;
    }

    public async Task<bool> CompleteAsync(int id)
    {
        var item = _todos.FirstOrDefault(t => t.Id == id);
        if (item is null || item.IsCompleted) return false;

        item.IsCompleted = true;
        item.CompletedAt = DateTime.Now;
        await _storage.SaveAsync(_todos);
        return true;
    }

    public async Task<bool> DeleteAsync(int id)
    {
        var item = _todos.FirstOrDefault(t => t.Id == id);
        if (item is null) return false;

        _todos.Remove(item);
        await _storage.SaveAsync(_todos);
        return true;
    }

    public IReadOnlyList<TodoItem> GetAll() => _todos.AsReadOnly();

    public IEnumerable<TodoItem> Search(string keyword)
        => _todos.Where(t => t.Title.Contains(keyword, StringComparison.OrdinalIgnoreCase));

    public IEnumerable<TodoItem> GetPending()
        => _todos.Where(t => !t.IsCompleted).OrderByDescending(t => t.Priority);

    public IEnumerable<TodoItem> GetCompleted()
        => _todos.Where(t => t.IsCompleted).OrderByDescending(t => t.CompletedAt);

    public (int Total, int Completed, int Pending) GetStats()
    {
        int total = _todos.Count;
        int completed = _todos.Count(t => t.IsCompleted);
        return (total, completed, total - completed);
    }
}

// ============================================================
//                      命令行界面层
// ============================================================

class TodoCli
{
    private readonly TodoService _service;

    public TodoCli(TodoService service)
    {
        _service = service;
    }

    public async Task RunAsync(string[] args)
    {
        if (args.Length == 0)
        {
            PrintHelp();
            return;
        }

        string command = args[0].ToLower();

        switch (command)
        {
            case "add":
                await HandleAddAsync(args[1..]);
                break;
            case "list" or "ls":
                HandleList(args[1..]);
                break;
            case "done" or "complete":
                await HandleCompleteAsync(args[1..]);
                break;
            case "delete" or "del" or "rm":
                await HandleDeleteAsync(args[1..]);
                break;
            case "search" or "find":
                HandleSearch(args[1..]);
                break;
            case "stats":
                HandleStats();
                break;
            case "help" or "--help" or "-h":
                PrintHelp();
                break;
            default:
                Console.WriteLine($"未知命令: {command}");
                PrintHelp();
                break;
        }
    }

    private async Task HandleAddAsync(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("错误: 请提供 Todo 标题");
            Console.WriteLine("用法: add <标题> [--priority high|medium|low]");
            return;
        }

        Priority priority = Priority.Medium;
        string title = args[0];

        // 解析选项
        for (int i = 1; i < args.Length; i++)
        {
            if ((args[i] == "--priority" || args[i] == "-p") && i + 1 < args.Length)
            {
                priority = args[i + 1].ToLower() switch
                {
                    "high" or "h" => Priority.High,
                    "low" or "l" => Priority.Low,
                    _ => Priority.Medium
                };
                i++;
            }
        }

        var item = await _service.AddAsync(title, priority);
        Console.WriteLine($"已添加: {item}");
    }

    private void HandleList(string[] args)
    {
        string filter = args.Length > 0 ? args[0].ToLower() : "all";

        IEnumerable<TodoItem> items = filter switch
        {
            "pending" or "p" => _service.GetPending(),
            "completed" or "done" or "c" => _service.GetCompleted(),
            _ => _service.GetAll().OrderBy(t => t.Id)
        };

        var list = items.ToList();
        if (!list.Any())
        {
            Console.WriteLine("暂无 Todo 项");
            return;
        }

        Console.WriteLine($"\n📋 Todo 列表（{filter}）:");
        Console.WriteLine(new string('-', 60));
        foreach (var item in list)
            Console.WriteLine(item);
        Console.WriteLine(new string('-', 60));

        var (total, completed, pending) = _service.GetStats();
        Console.WriteLine($"共 {list.Count} 项 | 总计 {total} | 已完成 {completed} | 待完成 {pending}");
    }

    private async Task HandleCompleteAsync(string[] args)
    {
        if (args.Length == 0 || !int.TryParse(args[0], out int id))
        {
            Console.WriteLine("用法: done <ID>");
            return;
        }

        bool success = await _service.CompleteAsync(id);
        Console.WriteLine(success ? $"✓ 已完成 #{id}" : $"未找到 #{id} 或已完成");
    }

    private async Task HandleDeleteAsync(string[] args)
    {
        if (args.Length == 0 || !int.TryParse(args[0], out int id))
        {
            Console.WriteLine("用法: delete <ID>");
            return;
        }

        bool success = await _service.DeleteAsync(id);
        Console.WriteLine(success ? $"✗ 已删除 #{id}" : $"未找到 #{id}");
    }

    private void HandleSearch(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("用法: search <关键词>");
            return;
        }

        string keyword = args[0];
        var results = _service.Search(keyword).ToList();

        Console.WriteLine($"\n🔍 搜索 \"{keyword}\" 结果:");
        if (!results.Any())
        {
            Console.WriteLine("  未找到匹配项");
            return;
        }

        foreach (var item in results)
            Console.WriteLine(item);
    }

    private void HandleStats()
    {
        var (total, completed, pending) = _service.GetStats();
        double completionRate = total > 0 ? (double)completed / total * 100 : 0;

        Console.WriteLine("\n📊 统计信息:");
        Console.WriteLine($"  总计:   {total}");
        Console.WriteLine($"  已完成: {completed}");
        Console.WriteLine($"  待完成: {pending}");
        Console.WriteLine($"  完成率: {completionRate:F1}%");

        // 按优先级统计
        foreach (Priority p in Enum.GetValues<Priority>())
        {
            int cnt = _service.GetAll().Count(t => t.Priority == p);
            if (cnt > 0) Console.WriteLine($"  {p}: {cnt} 项");
        }
    }

    private void PrintHelp()
    {
        Console.WriteLine("""
            Todo CLI — C# 命令行待办应用

            用法:
              todo add <标题> [--priority high|medium|low]   添加新任务
              todo list [all|pending|completed]               列出任务
              todo done <ID>                                  标记完成
              todo delete <ID>                                删除任务
              todo search <关键词>                            搜索任务
              todo stats                                      显示统计

            示例:
              todo add "学习 C# 泛型" --priority high
              todo add "读《CLR via C#》"
              todo list pending
              todo done 1
              todo search "C#"
            """);
    }
}

// ============================================================
//                      程序入口
// ============================================================

class Program
{
    static async Task Main(string[] args)
    {
        string dataFile = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".todo.json"
        );

        var storage = new JsonFileStorage(dataFile);
        var service = new TodoService(storage);
        await service.InitializeAsync();

        // 演示模式：无参数时运行示例操作
        if (args.Length == 0)
        {
            await RunDemoAsync(service);
            return;
        }

        var cli = new TodoCli(service);
        await cli.RunAsync(args);
    }

    static async Task RunDemoAsync(TodoService service)
    {
        Console.WriteLine("=== Todo CLI 演示 ===\n");
        var cli = new TodoCli(service);

        // 添加任务
        await cli.RunAsync(new[] { "add", "学习 C# 基础语法", "--priority", "high" });
        await cli.RunAsync(new[] { "add", "完成 LINQ 练习", "--priority", "high" });
        await cli.RunAsync(new[] { "add", "阅读《C# 8.0 in a Nutshell》" });
        await cli.RunAsync(new[] { "add", "写一个 ASP.NET Core API", "--priority", "low" });
        await cli.RunAsync(new[] { "add", "配置开发环境" });

        Console.WriteLine();

        // 列出所有任务
        await cli.RunAsync(new[] { "list" });
        Console.WriteLine();

        // 完成几个任务
        await cli.RunAsync(new[] { "done", "1" });
        await cli.RunAsync(new[] { "done", "5" });
        Console.WriteLine();

        // 列出待完成任务
        await cli.RunAsync(new[] { "list", "pending" });
        Console.WriteLine();

        // 搜索
        await cli.RunAsync(new[] { "search", "C#" });
        Console.WriteLine();

        // 统计
        await cli.RunAsync(new[] { "stats" });
        Console.WriteLine();

        // 删除一个任务
        await cli.RunAsync(new[] { "delete", "4" });
        await cli.RunAsync(new[] { "list" });
    }
}
