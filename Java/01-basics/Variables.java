/**
 * ============================================================
 *                    Java 变量与数据类型
 * ============================================================
 * 本文件介绍 Java 中的变量声明、基本数据类型、类型转换等核心概念。
 *
 * Java 是静态类型语言，变量必须先声明类型再使用。
 * ============================================================
 */
public class Variables {

    public static void main(String[] args) {
        primitiveTypes();
        variableDeclaration();
        typeConversion();
        operators();
        constants();
        stringBasics();
    }

    /**
     * ============================================================
     *                    1. 基本数据类型
     * ============================================================
     * Java 有 8 种基本数据类型（primitive types）
     */
    public static void primitiveTypes() {
        System.out.println("=".repeat(60));
        System.out.println("1. 基本数据类型");
        System.out.println("=".repeat(60));

        // 【整数类型】
        byte byteVar = 127;              // 8位，范围 -128 到 127
        short shortVar = 32767;          // 16位，范围 -32768 到 32767
        int intVar = 2147483647;         // 32位，最常用
        long longVar = 9223372036854775807L;  // 64位，注意要加 L

        System.out.println("--- 整数类型 ---");
        System.out.println("byte: " + byteVar + ", 范围: -128 ~ 127");
        System.out.println("short: " + shortVar + ", 范围: -32768 ~ 32767");
        System.out.println("int: " + intVar);
        System.out.println("long: " + longVar);

        // 不同进制表示
        int binary = 0b1010;     // 二进制，值为 10
        int octal = 017;         // 八进制，值为 15
        int hex = 0xFF;          // 十六进制，值为 255
        int withUnderscore = 1_000_000;  // 可以用下划线分隔（Java 7+）

        System.out.println("\n不同进制:");
        System.out.println("二进制 0b1010 = " + binary);
        System.out.println("八进制 017 = " + octal);
        System.out.println("十六进制 0xFF = " + hex);
        System.out.println("下划线分隔 1_000_000 = " + withUnderscore);

        // 【浮点类型】
        float floatVar = 3.14f;          // 32位，注意要加 f
        double doubleVar = 3.141592653589793;  // 64位，默认类型

        System.out.println("\n--- 浮点类型 ---");
        System.out.println("float: " + floatVar + " (精度约7位)");
        System.out.println("double: " + doubleVar + " (精度约15位)");

        // 科学计数法
        double scientific = 1.23e-4;  // 0.000123
        System.out.println("科学计数法 1.23e-4 = " + scientific);

        // 【警告】浮点数精度问题
        System.out.println("\n【警告】浮点数精度问题:");
        System.out.println("0.1 + 0.2 = " + (0.1 + 0.2));  // 不是精确的 0.3

        // 【字符类型】
        char charVar = 'A';              // 16位 Unicode 字符
        char unicodeChar = '\u4E2D';     // Unicode 编码（中）
        char intChar = 65;               // 也可以用整数

        System.out.println("\n--- 字符类型 ---");
        System.out.println("char: " + charVar);
        System.out.println("Unicode \\u4E2D: " + unicodeChar);
        System.out.println("整数 65 作为字符: " + intChar);

        // 【布尔类型】
        boolean boolTrue = true;
        boolean boolFalse = false;

        System.out.println("\n--- 布尔类型 ---");
        System.out.println("boolean true: " + boolTrue);
        System.out.println("boolean false: " + boolFalse);

        // 【各类型的默认值和大小】
        System.out.println("\n--- 类型信息 ---");
        System.out.println("Integer.MAX_VALUE = " + Integer.MAX_VALUE);
        System.out.println("Integer.MIN_VALUE = " + Integer.MIN_VALUE);
        System.out.println("Double.MAX_VALUE = " + Double.MAX_VALUE);
    }

    /**
     * ============================================================
     *                    2. 变量声明与赋值
     * ============================================================
     */
    public static void variableDeclaration() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. 变量声明与赋值");
        System.out.println("=".repeat(60));

        // 【声明并初始化】
        int age = 25;
        String name = "Alice";

        System.out.println("声明并初始化: age = " + age + ", name = " + name);

        // 【先声明后赋值】
        int score;
        score = 100;
        System.out.println("先声明后赋值: score = " + score);

        // 【多变量声明】
        int x = 1, y = 2, z = 3;
        System.out.println("多变量声明: x=" + x + ", y=" + y + ", z=" + z);

        // 【var 关键字（Java 10+）】局部变量类型推断
        var message = "Hello";  // 编译器推断为 String
        var number = 42;        // 编译器推断为 int
        var list = new java.util.ArrayList<String>();  // 推断为 ArrayList<String>

        System.out.println("\nvar 类型推断:");
        System.out.println("var message = \"Hello\" → " + message.getClass().getSimpleName());
        System.out.println("var number = 42 → " + ((Object) number).getClass().getSimpleName());
        System.out.println("var list = new ArrayList<String>() → " + list.getClass().getSimpleName());

        // 【注意】var 只能用于局部变量，不能用于：
        // - 类的成员变量
        // - 方法参数
        // - 返回类型
    }

    /**
     * ============================================================
     *                    3. 类型转换
     * ============================================================
     */
    public static void typeConversion() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("3. 类型转换");
        System.out.println("=".repeat(60));

        // 【自动类型转换（隐式）】小类型 → 大类型
        System.out.println("--- 自动类型转换 ---");
        int intVal = 100;
        long longVal = intVal;      // int → long
        float floatVal = longVal;   // long → float
        double doubleVal = floatVal; // float → double

        System.out.println("int → long → float → double: " + doubleVal);

        // 转换顺序: byte → short → int → long → float → double
        //                    char ↗

        // 【强制类型转换（显式）】大类型 → 小类型
        System.out.println("\n--- 强制类型转换 ---");
        double d = 3.99;
        int i = (int) d;  // 截断，不是四舍五入
        System.out.println("(int) 3.99 = " + i);

        long big = 1000000000000L;
        int small = (int) big;  // 可能溢出！
        System.out.println("【警告】(int) 1000000000000L = " + small + " (溢出!)");

        // 【字符串转换】
        System.out.println("\n--- 字符串转换 ---");

        // 基本类型 → 字符串
        String s1 = String.valueOf(123);
        String s2 = Integer.toString(456);
        String s3 = "" + 789;  // 简便写法
        System.out.println("int → String: " + s1 + ", " + s2 + ", " + s3);

        // 字符串 → 基本类型
        int parsed1 = Integer.parseInt("123");
        double parsed2 = Double.parseDouble("3.14");
        boolean parsed3 = Boolean.parseBoolean("true");
        System.out.println("String → int: " + parsed1);
        System.out.println("String → double: " + parsed2);
        System.out.println("String → boolean: " + parsed3);

        // 【包装类自动装箱/拆箱】
        System.out.println("\n--- 自动装箱/拆箱 ---");
        Integer boxed = 100;        // 自动装箱 int → Integer
        int unboxed = boxed;        // 自动拆箱 Integer → int
        System.out.println("自动装箱: " + boxed + ", 自动拆箱: " + unboxed);

        // 【警告】Integer 缓存
        Integer a = 127;
        Integer b = 127;
        Integer c = 128;
        Integer d2 = 128;
        System.out.println("\n【警告】Integer 缓存:");
        System.out.println("127 == 127: " + (a == b));   // true (缓存)
        System.out.println("128 == 128: " + (c == d2));  // false (不缓存)
        System.out.println("应该用 equals: " + c.equals(d2));  // true
    }

    /**
     * ============================================================
     *                    4. 运算符
     * ============================================================
     */
    public static void operators() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("4. 运算符");
        System.out.println("=".repeat(60));

        int a = 17, b = 5;

        // 【算术运算符】
        System.out.println("--- 算术运算符 ---");
        System.out.println("a = " + a + ", b = " + b);
        System.out.println("a + b = " + (a + b));
        System.out.println("a - b = " + (a - b));
        System.out.println("a * b = " + (a * b));
        System.out.println("a / b = " + (a / b) + " (整数除法)");
        System.out.println("a % b = " + (a % b) + " (取模)");

        // 【自增自减】
        System.out.println("\n--- 自增自减 ---");
        int x = 5;
        System.out.println("x = " + x);
        System.out.println("x++ = " + (x++) + ", 之后 x = " + x);
        System.out.println("++x = " + (++x) + ", 之后 x = " + x);

        // 【比较运算符】
        System.out.println("\n--- 比较运算符 ---");
        System.out.println("a == b: " + (a == b));
        System.out.println("a != b: " + (a != b));
        System.out.println("a > b: " + (a > b));
        System.out.println("a <= b: " + (a <= b));

        // 【逻辑运算符】
        System.out.println("\n--- 逻辑运算符 ---");
        boolean p = true, q = false;
        System.out.println("p = " + p + ", q = " + q);
        System.out.println("p && q: " + (p && q));  // 短路与
        System.out.println("p || q: " + (p || q));  // 短路或
        System.out.println("!p: " + (!p));

        // 【位运算符】
        System.out.println("\n--- 位运算符 ---");
        int m = 0b1010, n = 0b1100;
        System.out.println("m = 0b1010 (10), n = 0b1100 (12)");
        System.out.println("m & n = " + (m & n) + " (按位与)");
        System.out.println("m | n = " + (m | n) + " (按位或)");
        System.out.println("m ^ n = " + (m ^ n) + " (按位异或)");
        System.out.println("~m = " + (~m) + " (按位取反)");
        System.out.println("m << 2 = " + (m << 2) + " (左移)");
        System.out.println("m >> 1 = " + (m >> 1) + " (右移)");

        // 【三元运算符】
        System.out.println("\n--- 三元运算符 ---");
        int max = (a > b) ? a : b;
        System.out.println("max(a, b) = " + max);
    }

    /**
     * ============================================================
     *                    5. 常量
     * ============================================================
     */
    public static void constants() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("5. 常量");
        System.out.println("=".repeat(60));

        // 【final 关键字】定义常量
        final double PI = 3.14159;
        final int MAX_SIZE = 100;
        final String APP_NAME = "MyApp";

        System.out.println("PI = " + PI);
        System.out.println("MAX_SIZE = " + MAX_SIZE);
        System.out.println("APP_NAME = " + APP_NAME);

        // PI = 3.14;  // 编译错误！final 变量不能重新赋值

        // 【命名约定】常量全大写，下划线分隔
        System.out.println("\n命名约定: 常量全大写，如 MAX_SIZE, APP_NAME");

        // 【static final】类常量
        System.out.println("\nstatic final 用于定义类常量（见类定义）");
    }

    // 类常量示例
    public static final String VERSION = "1.0.0";
    public static final int DEFAULT_PORT = 8080;

    /**
     * ============================================================
     *                    6. 字符串基础
     * ============================================================
     */
    public static void stringBasics() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("6. 字符串基础");
        System.out.println("=".repeat(60));

        // 【创建字符串】
        String s1 = "Hello";           // 字符串字面量（推荐）
        String s2 = new String("Hello");  // 使用构造函数
        String s3 = "Hello";           // 与 s1 共享同一对象

        System.out.println("--- 字符串创建 ---");
        System.out.println("s1 = \"Hello\", s2 = new String(\"Hello\"), s3 = \"Hello\"");
        System.out.println("s1 == s3: " + (s1 == s3) + " (字符串池)");
        System.out.println("s1 == s2: " + (s1 == s2) + " (不同对象)");
        System.out.println("s1.equals(s2): " + s1.equals(s2) + " (内容相等)");

        // 【字符串方法】
        System.out.println("\n--- 常用方法 ---");
        String text = "  Hello, World!  ";
        System.out.println("原字符串: \"" + text + "\"");
        System.out.println("length(): " + text.length());
        System.out.println("trim(): \"" + text.trim() + "\"");
        System.out.println("toUpperCase(): " + text.trim().toUpperCase());
        System.out.println("toLowerCase(): " + text.trim().toLowerCase());
        System.out.println("charAt(0): " + text.trim().charAt(0));
        System.out.println("substring(0, 5): " + text.trim().substring(0, 5));
        System.out.println("indexOf(\"World\"): " + text.indexOf("World"));
        System.out.println("contains(\"World\"): " + text.contains("World"));
        System.out.println("startsWith(\"  H\"): " + text.startsWith("  H"));
        System.out.println("replace(\"World\", \"Java\"): " + text.replace("World", "Java"));

        // 【字符串拼接】
        System.out.println("\n--- 字符串拼接 ---");
        String name = "Alice";
        int age = 25;

        // + 运算符
        String msg1 = "Name: " + name + ", Age: " + age;
        System.out.println("+ 拼接: " + msg1);

        // String.format
        String msg2 = String.format("Name: %s, Age: %d", name, age);
        System.out.println("format: " + msg2);

        // StringBuilder（推荐用于循环拼接）
        StringBuilder sb = new StringBuilder();
        sb.append("Name: ").append(name).append(", Age: ").append(age);
        System.out.println("StringBuilder: " + sb.toString());

        // 【文本块（Java 15+）】
        String json = """
                {
                    "name": "Alice",
                    "age": 25
                }
                """;
        System.out.println("\n文本块 (Java 15+):");
        System.out.println(json);

        // 【字符串不可变性】
        System.out.println("【重要】String 是不可变的，所有修改都返回新对象");
    }
}
