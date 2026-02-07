# Records

::: info 文件信息
- 📄 原文件：`Records.java`
- 🔤 语言：java
:::

============================================================
                   Java Record 类型
============================================================
本文件介绍 Java 14+ 引入的 Record 类型。
============================================================

## 完整代码

```java
import java.util.*;

/**
 * ============================================================
 *                    Java Record 类型
 * ============================================================
 * 本文件介绍 Java 14+ 引入的 Record 类型。
 * ============================================================
 */
public class Records {

    public static void main(String[] args) {
        recordBasics();
        recordFeatures();
        recordPatterns();
        recordUseCases();
    }

    /**
     * ============================================================
     *                    1. Record 基础
     * ============================================================
     */
    public static void recordBasics() {
        System.out.println("=".repeat(60));
        System.out.println("1. Record 基础");
        System.out.println("=".repeat(60));

        System.out.println("""
            Record 是不可变数据类的简洁语法（Java 14+）
            - 自动生成构造器、getter、equals、hashCode、toString
            - 组件是 final 的
            - 类本身是 final 的
            """);

        // 【创建 Record 实例】
        System.out.println("--- 创建 Record ---");
        Point p1 = new Point(3, 4);
        Point p2 = new Point(3, 4);
        Point p3 = new Point(5, 6);

        System.out.println("p1 = " + p1);
        System.out.println("p2 = " + p2);

        // 【访问组件】使用组件名作为方法名（不是 getX）
        System.out.println("\n--- 访问组件 ---");
        System.out.println("p1.x() = " + p1.x());
        System.out.println("p1.y() = " + p1.y());

        // 【equals 和 hashCode】
        System.out.println("\n--- equals 和 hashCode ---");
        System.out.println("p1.equals(p2): " + p1.equals(p2));  // true
        System.out.println("p1.equals(p3): " + p1.equals(p3));  // false
        System.out.println("p1.hashCode() == p2.hashCode(): " + (p1.hashCode() == p2.hashCode()));

        // 【toString】
        System.out.println("\n--- toString ---");
        System.out.println("p1.toString(): " + p1.toString());

        // 【不可变性】
        System.out.println("\n--- 不可变性 ---");
        System.out.println("""
            Record 的组件是 final 的，不能修改
            // p1.x = 10;  // 编译错误

            要"修改"需要创建新实例：
            Point moved = new Point(p1.x() + 1, p1.y() + 1);
            """);
    }

    /**
     * ============================================================
     *                    2. Record 特性
     * ============================================================
     */
    public static void recordFeatures() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. Record 特性");
        System.out.println("=".repeat(60));

        // 【紧凑构造器】
        System.out.println("--- 紧凑构造器 ---");
        try {
            Range range = new Range(5, 10);
            System.out.println("有效范围: " + range);

            // Range invalid = new Range(10, 5);  // 抛出异常
        } catch (IllegalArgumentException e) {
            System.out.println("验证失败: " + e.getMessage());
        }

        // 【自定义构造器】
        System.out.println("\n--- 自定义构造器 ---");
        ColorRGB red = new ColorRGB(255, 0, 0);
        ColorRGB blue = ColorRGB.fromHex("#0000FF");
        System.out.println("red: " + red);
        System.out.println("blue: " + blue);

        // 【实例方法】
        System.out.println("\n--- 实例方法 ---");
        Point p = new Point(3, 4);
        System.out.println("distance from origin: " + p.distanceFromOrigin());
        System.out.println("translated: " + p.translate(1, 1));

        // 【静态方法和字段】
        System.out.println("\n--- 静态成员 ---");
        System.out.println("origin: " + Point.origin());

        // 【实现接口】
        System.out.println("\n--- 实现接口 ---");
        Person person = new Person("Alice", 25);
        System.out.println("greeting: " + person.greet());

        // 【嵌套 Record】
        System.out.println("\n--- 嵌套 Record ---");
        Rectangle rect = new Rectangle(new Point(0, 0), new Point(10, 5));
        System.out.println("rectangle: " + rect);
        System.out.println("width: " + rect.width());
        System.out.println("height: " + rect.height());
        System.out.println("area: " + rect.area());
    }

    /**
     * ============================================================
     *                    3. Record 与模式匹配
     * ============================================================
     */
    public static void recordPatterns() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("3. Record 与模式匹配");
        System.out.println("=".repeat(60));

        // 【instanceof 模式匹配】
        System.out.println("--- instanceof 模式匹配 ---");
        Object obj = new Point(3, 4);

        if (obj instanceof Point p) {
            System.out.println("Point: x=" + p.x() + ", y=" + p.y());
        }

        // 【Record 解构模式】（Java 21+）
        System.out.println("\n--- Record 解构模式 ---");
        if (obj instanceof Point(int x, int y)) {
            System.out.println("解构: x=" + x + ", y=" + y);
        }

        // 【switch 中的 Record 模式】
        System.out.println("\n--- switch Record 模式 ---");
        printShape(new Circle(5));
        printShape(new Rect(4, 3));
        printShape(new Triangle(3, 4, 5));

        // 【嵌套 Record 解构】
        System.out.println("\n--- 嵌套解构 ---");
        Rectangle rect = new Rectangle(new Point(0, 0), new Point(10, 5));
        describeRectangle(rect);
    }

    public static void printShape(Shape shape) {
        String desc = switch (shape) {
            case Circle(double r) -> "圆形，半径 " + r + "，面积 " + (Math.PI * r * r);
            case Rect(double w, double h) -> "矩形，" + w + "x" + h + "，面积 " + (w * h);
            case Triangle(double a, double b, double c) -> "三角形，边长 " + a + ", " + b + ", " + c;
        };
        System.out.println(desc);
    }

    public static void describeRectangle(Rectangle rect) {
        if (rect instanceof Rectangle(Point(int x1, int y1), Point(int x2, int y2))) {
            System.out.println("矩形从 (" + x1 + "," + y1 + ") 到 (" + x2 + "," + y2 + ")");
        }
    }

    /**
     * ============================================================
     *                    4. Record 使用场景
     * ============================================================
     */
    public static void recordUseCases() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("4. Record 使用场景");
        System.out.println("=".repeat(60));

        // 【DTO（数据传输对象）】
        System.out.println("--- DTO ---");
        UserDTO user = new UserDTO(1L, "alice", "alice@example.com");
        System.out.println("User DTO: " + user);

        // 【API 响应】
        System.out.println("\n--- API 响应 ---");
        ApiResponse<String> success = new ApiResponse<>(200, "OK", "Hello!");
        ApiResponse<String> error = new ApiResponse<>(404, "Not Found", null);
        System.out.println("Success: " + success);
        System.out.println("Error: " + error);

        // 【复合键】
        System.out.println("\n--- Map 复合键 ---");
        Map<CacheKey, String> cache = new HashMap<>();
        cache.put(new CacheKey("users", 1L), "Alice");
        cache.put(new CacheKey("users", 2L), "Bob");
        cache.put(new CacheKey("products", 1L), "Laptop");

        System.out.println("cache.get(users, 1): " + cache.get(new CacheKey("users", 1L)));

        // 【返回多个值】
        System.out.println("\n--- 返回多个值 ---");
        MinMax result = findMinMax(List.of(3, 1, 4, 1, 5, 9, 2, 6));
        System.out.println("min: " + result.min() + ", max: " + result.max());

        // 【不可变配置】
        System.out.println("\n--- 不可变配置 ---");
        DatabaseConfig config = new DatabaseConfig("localhost", 5432, "mydb", true);
        System.out.println("DB Config: " + config);
        System.out.println("Connection URL: " + config.toConnectionUrl());

        // 【事件】
        System.out.println("\n--- 事件 ---");
        UserEvent event = new UserEvent("login", "alice", System.currentTimeMillis());
        System.out.println("Event: " + event);

        // 【Record 与 Stream】
        System.out.println("\n--- Record 与 Stream ---");
        List<Product> products = List.of(
            new Product("Apple", 1.50, 100),
            new Product("Banana", 0.75, 200),
            new Product("Cherry", 3.00, 50)
        );

        double totalValue = products.stream()
            .mapToDouble(p -> p.price() * p.quantity())
            .sum();
        System.out.println("Total inventory value: $" + totalValue);

        Product mostValuable = products.stream()
            .max(Comparator.comparingDouble(p -> p.price() * p.quantity()))
            .orElseThrow();
        System.out.println("Most valuable: " + mostValuable.name());
    }

    public static MinMax findMinMax(List<Integer> numbers) {
        int min = numbers.stream().min(Integer::compareTo).orElse(0);
        int max = numbers.stream().max(Integer::compareTo).orElse(0);
        return new MinMax(min, max);
    }
}

// ============================================================
//                    Record 定义
// ============================================================

/**
 * 基本 Record
 */
record Point(int x, int y) {
    // 静态工厂方法
    public static Point origin() {
        return new Point(0, 0);
    }

    // 实例方法
    public double distanceFromOrigin() {
        return Math.sqrt(x * x + y * y);
    }

    public Point translate(int dx, int dy) {
        return new Point(x + dx, y + dy);
    }
}

/**
 * 带验证的 Record（紧凑构造器）
 */
record Range(int start, int end) {
    // 紧凑构造器：验证参数
    public Range {
        if (start > end) {
            throw new IllegalArgumentException("start must be <= end");
        }
    }
}

/**
 * 带多个构造器的 Record
 */
record ColorRGB(int red, int green, int blue) {
    // 紧凑构造器：验证范围
    public ColorRGB {
        if (red < 0 || red > 255 || green < 0 || green > 255 || blue < 0 || blue > 255) {
            throw new IllegalArgumentException("Color values must be 0-255");
        }
    }

    // 额外的工厂方法
    public static ColorRGB fromHex(String hex) {
        hex = hex.replace("#", "");
        int r = Integer.parseInt(hex.substring(0, 2), 16);
        int g = Integer.parseInt(hex.substring(2, 4), 16);
        int b = Integer.parseInt(hex.substring(4, 6), 16);
        return new ColorRGB(r, g, b);
    }

    public String toHex() {
        return String.format("#%02X%02X%02X", red, green, blue);
    }
}

/**
 * 实现接口的 Record
 */
interface Greeting {
    String greet();
}

record Person(String name, int age) implements Greeting {
    @Override
    public String greet() {
        return "Hello, I'm " + name + ", " + age + " years old.";
    }
}

/**
 * 嵌套 Record
 */
record Rectangle(Point topLeft, Point bottomRight) {
    public int width() {
        return Math.abs(bottomRight.x() - topLeft.x());
    }

    public int height() {
        return Math.abs(bottomRight.y() - topLeft.y());
    }

    public int area() {
        return width() * height();
    }
}

/**
 * 密封接口 + Record
 */
sealed interface Shape permits Circle, Rect, Triangle {}

record Circle(double radius) implements Shape {}
record Rect(double width, double height) implements Shape {}
record Triangle(double a, double b, double c) implements Shape {}

// 使用场景示例
record UserDTO(Long id, String username, String email) {}

record ApiResponse<T>(int code, String message, T data) {}

record CacheKey(String type, Long id) {}

record MinMax(int min, int max) {}

record DatabaseConfig(String host, int port, String database, boolean ssl) {
    public String toConnectionUrl() {
        String protocol = ssl ? "jdbc:postgresql" : "jdbc:postgresql";
        return protocol + "://" + host + ":" + port + "/" + database;
    }
}

record UserEvent(String type, String username, long timestamp) {}

record Product(String name, double price, int quantity) {}
```
