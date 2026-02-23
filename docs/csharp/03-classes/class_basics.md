# class basics.cs

::: info 文件信息
- 📄 原文件：`01_class_basics.cs`
- 🔤 语言：csharp
:::

## 完整代码

```csharp
// ============================================================
//                      类与面向对象编程
// ============================================================
// C# 是完全面向对象的语言，支持封装、继承、多态三大特性
// 类（class）是引用类型，结构体（struct）是值类型
// 属性（Property）封装字段访问，提供 get/set 访问器

using System;
using System.Collections.Generic;
using System.Text;

// ============================================================
//                      基础类定义
// ============================================================

// ----------------------------------------------------------
// 1. 类与属性
// ----------------------------------------------------------
class Person
{
    // 私有字段（以下划线开头是常见约定）
    private string _name;
    private int _age;

    // 自动属性（Auto Property）— 编译器自动创建私有字段
    // 【推荐】简单存储用自动属性，需要验证用完整属性
    public string? Email { get; set; }
    public DateTime CreatedAt { get; } = DateTime.Now;  // 只读自动属性

    // 完整属性（带验证逻辑）
    public string Name
    {
        get => _name;
        set
        {
            if (string.IsNullOrWhiteSpace(value))
                throw new ArgumentException("姓名不能为空");
            _name = value.Trim();
        }
    }

    public int Age
    {
        get => _age;
        set
        {
            if (value < 0 || value > 150)
                throw new ArgumentOutOfRangeException(nameof(value), "年龄必须在 0-150 之间");
            _age = value;
        }
    }

    // 只读计算属性（Computed Property）
    public bool IsAdult => _age >= 18;
    public string DisplayName => $"{_name} ({_age}岁)";

    // ----------------------------------------------------------
    // 构造函数
    // ----------------------------------------------------------
    // 默认构造函数
    public Person()
    {
        _name = "未知";
        _age = 0;
    }

    // 主构造函数
    public Person(string name, int age)
    {
        Name = name;  // 使用属性赋值，触发验证
        Age = age;
    }

    // 构造函数链（this() 调用其他构造函数）
    public Person(string name) : this(name, 0)
    {
    }

    // ----------------------------------------------------------
    // 方法
    // ----------------------------------------------------------
    public virtual string Introduce()
    {
        return $"我是 {_name}，今年 {_age} 岁。";
    }

    public override string ToString()
    {
        return $"Person {{ Name={_name}, Age={_age} }}";
    }
}

// ----------------------------------------------------------
// 2. 继承与多态
// ----------------------------------------------------------
// 【继承】使用 : 基类名，C# 只支持单继承（可以实现多个接口）
// 【多态】virtual/override 实现运行时多态
// 【密封】sealed 阻止继承或方法被重写

class Student : Person
{
    public string School { get; set; }
    public double GPA { get; set; }

    public Student(string name, int age, string school) : base(name, age)
    {
        School = school;
    }

    // override 重写父类的 virtual 方法
    public override string Introduce()
    {
        return $"{base.Introduce()} 我在 {School} 上学，GPA {GPA:F1}。";
    }

    public override string ToString()
    {
        return $"Student {{ {base.ToString()}, School={School} }}";
    }
}

class Employee : Person
{
    public string Company { get; set; }
    public decimal Salary { get; set; }

    public Employee(string name, int age, string company, decimal salary)
        : base(name, age)
    {
        Company = company;
        Salary = salary;
    }

    public override string Introduce()
    {
        return $"{base.Introduce()} 我在 {Company} 工作，月薪 {Salary:C}。";
    }
}

// ----------------------------------------------------------
// 3. 抽象类
// ----------------------------------------------------------
// 抽象类不能被实例化，必须由子类实现抽象方法
// 【区别】抽象类可以有实现，接口（C# 8 之前）不能

abstract class Shape
{
    public string Color { get; set; } = "黑色";

    // 抽象方法：子类必须实现
    public abstract double Area();
    public abstract double Perimeter();

    // 普通方法：子类可直接使用
    public virtual void Describe()
    {
        Console.WriteLine($"{GetType().Name}（{Color}）: 面积={Area():F2}, 周长={Perimeter():F2}");
    }
}

class Circle : Shape
{
    public double Radius { get; }

    public Circle(double radius, string color = "红色")
    {
        Radius = radius;
        Color = color;
    }

    public override double Area() => Math.PI * Radius * Radius;
    public override double Perimeter() => 2 * Math.PI * Radius;
}

class Rectangle : Shape
{
    public double Width { get; }
    public double Height { get; }

    public Rectangle(double width, double height)
    {
        Width = width;
        Height = height;
    }

    public override double Area() => Width * Height;
    public override double Perimeter() => 2 * (Width + Height);
}

// ============================================================
//                      主程序
// ============================================================
class ClassBasics
{
    static void Main()
    {
        Console.WriteLine("=== 类与对象 ===");

        // ----------------------------------------------------------
        // 创建实例
        // ----------------------------------------------------------
        var p1 = new Person("张三", 25);
        var p2 = new Person("李四");
        var p3 = new Person();  // 默认构造函数

        Console.WriteLine(p1.Introduce());
        Console.WriteLine(p1.IsAdult);
        Console.WriteLine(p1.DisplayName);
        Console.WriteLine(p1);  // 调用 ToString()

        // 对象初始化器（C# 3.0+）
        var p4 = new Person("王五", 30) { Email = "wangwu@example.com" };
        Console.WriteLine($"邮箱: {p4.Email}");

        // ----------------------------------------------------------
        // 继承与多态
        // ----------------------------------------------------------
        Console.WriteLine("\n=== 继承与多态 ===");

        var student = new Student("小明", 20, "清华大学") { GPA = 3.8 };
        var employee = new Employee("赵总", 35, "科技公司", 20000);

        Console.WriteLine(student.Introduce());
        Console.WriteLine(employee.Introduce());

        // 多态：基类引用指向子类对象
        Person[] people = { p1, student, employee };
        Console.WriteLine("\n多态调用 Introduce():");
        foreach (Person person in people)
        {
            Console.WriteLine($"  [{person.GetType().Name}] {person.Introduce()}");
        }

        // is / as 类型检测
        foreach (Person person in people)
        {
            if (person is Student s)
                Console.WriteLine($"  {s.Name} 就读于 {s.School}");
            else if (person is Employee e)
                Console.WriteLine($"  {e.Name} 供职于 {e.Company}");
        }

        // ----------------------------------------------------------
        // 抽象类
        // ----------------------------------------------------------
        Console.WriteLine("\n=== 抽象类（形状） ===");

        Shape[] shapes = {
            new Circle(5.0),
            new Circle(3.0, "蓝色"),
            new Rectangle(4.0, 6.0)
        };

        foreach (Shape shape in shapes)
        {
            shape.Describe();
        }

        // ----------------------------------------------------------
        // 静态成员
        // ----------------------------------------------------------
        Console.WriteLine("\n=== 静态成员 ===");
        Console.WriteLine($"已创建对象数: {Counter.Count}");

        var c1 = new Counter();
        var c2 = new Counter();
        var c3 = new Counter();
        Console.WriteLine($"已创建对象数: {Counter.Count}");

        c1.Increment(3);
        c2.Increment(7);
        Console.WriteLine($"c1={c1.Value}, c2={c2.Value}");
        Console.WriteLine($"总计: {Counter.Total}");

        // ----------------------------------------------------------
        // 记录类型（Record，C# 9.0+）
        // ----------------------------------------------------------
        Console.WriteLine("\n=== Record 类型 ===");

        // Record 是不可变的引用类型，自动生成 Equals、GetHashCode、ToString
        var point1 = new Point2D(3.0, 4.0);
        var point2 = new Point2D(3.0, 4.0);
        var point3 = point1 with { Y = 0.0 };  // with 表达式创建副本

        Console.WriteLine($"point1: {point1}");
        Console.WriteLine($"point2: {point2}");
        Console.WriteLine($"point1 == point2: {point1 == point2}");  // 值相等
        Console.WriteLine($"point3: {point3}");
        Console.WriteLine($"距离: {point1.DistanceTo(point3):F2}");
    }
}

// ----------------------------------------------------------
// 静态成员示例
// ----------------------------------------------------------
class Counter
{
    // 静态字段：所有实例共享
    private static int _count = 0;
    private static int _total = 0;

    public int Value { get; private set; }

    public static int Count => _count;
    public static int Total => _total;

    public Counter()
    {
        _count++;  // 每次创建实例，计数器加1
        Value = 0;
    }

    public void Increment(int amount)
    {
        Value += amount;
        _total += amount;
    }
}

// ----------------------------------------------------------
// Record 类型示例
// ----------------------------------------------------------
record Point2D(double X, double Y)
{
    // Record 可以添加方法
    public double DistanceTo(Point2D other)
    {
        double dx = X - other.X;
        double dy = Y - other.Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }
}
```
