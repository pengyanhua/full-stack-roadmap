/**
 * ============================================================
 *                    Java 类基础
 * ============================================================
 * 本文件介绍 Java 中类的定义、构造函数、成员变量和方法。
 * ============================================================
 */
public class ClassBasics {

    public static void main(String[] args) {
        classDefinition();
        constructors();
        accessModifiers();
        staticMembers();
        innerClasses();
    }

    /**
     * ============================================================
     *                    1. 类的定义
     * ============================================================
     */
    public static void classDefinition() {
        System.out.println("=".repeat(60));
        System.out.println("1. 类的定义");
        System.out.println("=".repeat(60));

        // 创建对象
        Person person = new Person();
        person.name = "Alice";
        person.age = 25;

        System.out.println("姓名: " + person.name);
        System.out.println("年龄: " + person.age);
        person.introduce();

        // 使用带参构造函数
        Person bob = new Person("Bob", 30);
        bob.introduce();

        // 调用方法
        System.out.println("\n调用方法:");
        System.out.println("isAdult(): " + bob.isAdult());
        bob.setAge(35);
        System.out.println("设置年龄后: " + bob.getAge());
    }

    /**
     * ============================================================
     *                    2. 构造函数
     * ============================================================
     */
    public static void constructors() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. 构造函数");
        System.out.println("=".repeat(60));

        // 无参构造
        Book book1 = new Book();
        System.out.println("无参构造: " + book1);

        // 部分参数构造
        Book book2 = new Book("Java 入门");
        System.out.println("部分参数: " + book2);

        // 全参数构造
        Book book3 = new Book("Java 进阶", "张三", 59.9);
        System.out.println("全参数: " + book3);

        // 拷贝构造
        Book book4 = new Book(book3);
        System.out.println("拷贝构造: " + book4);
    }

    /**
     * ============================================================
     *                    3. 访问修饰符
     * ============================================================
     */
    public static void accessModifiers() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("3. 访问修饰符");
        System.out.println("=".repeat(60));

        System.out.println("""
            访问修饰符权限：

            修饰符      类内  包内  子类  其他包
            ----------------------------------------
            public      ✓     ✓     ✓     ✓
            protected   ✓     ✓     ✓     ✗
            (default)   ✓     ✓     ✗     ✗
            private     ✓     ✗     ✗     ✗
            """);

        AccessDemo demo = new AccessDemo();
        System.out.println("public: " + demo.publicField);
        // System.out.println(demo.privateField);  // 编译错误
        System.out.println("通过 getter 访问: " + demo.getPrivateField());
    }

    /**
     * ============================================================
     *                    4. 静态成员
     * ============================================================
     */
    public static void staticMembers() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("4. 静态成员");
        System.out.println("=".repeat(60));

        // 静态变量
        System.out.println("--- 静态变量 ---");
        System.out.println("Counter.count = " + Counter.count);

        Counter c1 = new Counter();
        Counter c2 = new Counter();
        Counter c3 = new Counter();

        System.out.println("创建 3 个实例后: Counter.count = " + Counter.count);

        // 静态方法
        System.out.println("\n--- 静态方法 ---");
        int result = MathUtils.add(3, 5);
        System.out.println("MathUtils.add(3, 5) = " + result);

        // 静态块
        System.out.println("\n--- 静态块 ---");
        System.out.println("Config.appName = " + Config.appName);

        // 静态导入
        System.out.println("\n--- 静态导入 ---");
        // import static java.lang.Math.PI;
        // import static java.lang.Math.sqrt;
        System.out.println("Math.PI = " + Math.PI);
        System.out.println("Math.sqrt(16) = " + Math.sqrt(16));
    }

    /**
     * ============================================================
     *                    5. 内部类
     * ============================================================
     */
    public static void innerClasses() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("5. 内部类");
        System.out.println("=".repeat(60));

        // 【成员内部类】
        System.out.println("--- 成员内部类 ---");
        Outer outer = new Outer();
        Outer.Inner inner = outer.new Inner();
        inner.display();

        // 【静态内部类】
        System.out.println("\n--- 静态内部类 ---");
        Outer.StaticInner staticInner = new Outer.StaticInner();
        staticInner.display();

        // 【局部内部类】
        System.out.println("\n--- 局部内部类 ---");
        outer.methodWithLocalClass();

        // 【匿名内部类】
        System.out.println("\n--- 匿名内部类 ---");
        Greeting greeting = new Greeting() {
            @Override
            public void sayHello(String name) {
                System.out.println("匿名内部类: Hello, " + name + "!");
            }
        };
        greeting.sayHello("World");
    }
}

// ============================================================
//                    辅助类定义
// ============================================================

/**
 * 人员类 - 演示基本类结构
 */
class Person {
    // 成员变量（字段）
    String name;
    int age;

    // 无参构造函数
    public Person() {
        this.name = "Unknown";
        this.age = 0;
    }

    // 带参构造函数
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    // 成员方法
    public void introduce() {
        System.out.println("我是 " + name + "，今年 " + age + " 岁");
    }

    public boolean isAdult() {
        return age >= 18;
    }

    // Getter 和 Setter
    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        if (age >= 0) {
            this.age = age;
        }
    }
}

/**
 * 书籍类 - 演示构造函数重载
 */
class Book {
    private String title;
    private String author;
    private double price;

    // 无参构造
    public Book() {
        this("未知书名", "未知作者", 0.0);
    }

    // 部分参数构造
    public Book(String title) {
        this(title, "未知作者", 0.0);
    }

    // 全参数构造
    public Book(String title, String author, double price) {
        this.title = title;
        this.author = author;
        this.price = price;
    }

    // 拷贝构造
    public Book(Book other) {
        this(other.title, other.author, other.price);
    }

    @Override
    public String toString() {
        return String.format("Book{title='%s', author='%s', price=%.2f}",
                title, author, price);
    }
}

/**
 * 访问修饰符演示类
 */
class AccessDemo {
    public String publicField = "public 字段";
    protected String protectedField = "protected 字段";
    String defaultField = "default 字段";
    private String privateField = "private 字段";

    public String getPrivateField() {
        return privateField;
    }
}

/**
 * 计数器类 - 演示静态变量
 */
class Counter {
    // 静态变量：所有实例共享
    public static int count = 0;

    public Counter() {
        count++;  // 每创建一个实例，计数加1
    }
}

/**
 * 数学工具类 - 演示静态方法
 */
class MathUtils {
    // 私有构造函数，防止实例化
    private MathUtils() {}

    public static int add(int a, int b) {
        return a + b;
    }

    public static int multiply(int a, int b) {
        return a * b;
    }
}

/**
 * 配置类 - 演示静态块
 */
class Config {
    public static String appName;
    public static String version;

    // 静态块：类加载时执行一次
    static {
        System.out.println("  [Config 静态块执行]");
        appName = "MyApp";
        version = "1.0.0";
    }
}

/**
 * 外部类 - 演示内部类
 */
class Outer {
    private String outerField = "外部类字段";

    // 成员内部类
    public class Inner {
        public void display() {
            // 可以访问外部类的私有成员
            System.out.println("内部类访问: " + outerField);
        }
    }

    // 静态内部类
    public static class StaticInner {
        public void display() {
            System.out.println("静态内部类");
            // 不能访问外部类的非静态成员
        }
    }

    // 方法中的局部内部类
    public void methodWithLocalClass() {
        class LocalClass {
            public void display() {
                System.out.println("局部内部类，访问: " + outerField);
            }
        }
        LocalClass local = new LocalClass();
        local.display();
    }
}

/**
 * 接口 - 用于匿名内部类演示
 */
interface Greeting {
    void sayHello(String name);
}
