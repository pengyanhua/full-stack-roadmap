/**
 * ============================================================
 *                    Java 接口
 * ============================================================
 * 本文件介绍 Java 中的接口定义、实现、默认方法等概念。
 * ============================================================
 */
public class Interfaces {

    public static void main(String[] args) {
        interfaceBasics();
        multipleInterfaces();
        defaultMethods();
        staticMethods();
        functionalInterfaces();
        sealedInterfaces();
    }

    /**
     * ============================================================
     *                    1. 接口基础
     * ============================================================
     */
    public static void interfaceBasics() {
        System.out.println("=".repeat(60));
        System.out.println("1. 接口基础");
        System.out.println("=".repeat(60));

        // 接口不能实例化
        // Flyable flyable = new Flyable();  // 错误

        // 使用实现类
        Bird bird = new Bird();
        Airplane airplane = new Airplane();

        System.out.println("--- 接口实现 ---");
        bird.fly();
        airplane.fly();

        // 接口作为类型
        System.out.println("\n--- 接口作为类型 ---");
        Flyable[] flyables = {bird, airplane};
        for (Flyable f : flyables) {
            f.fly();
        }

        // 接口常量
        System.out.println("\n--- 接口常量 ---");
        System.out.println("MAX_HEIGHT = " + Flyable.MAX_HEIGHT);
    }

    /**
     * ============================================================
     *                2. 实现多个接口
     * ============================================================
     */
    public static void multipleInterfaces() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. 实现多个接口");
        System.out.println("=".repeat(60));

        // 一个类可以实现多个接口
        Duck duck = new Duck();
        duck.fly();
        duck.swim();

        // 多接口类型
        System.out.println("\n--- 多接口引用 ---");
        Flyable flyable = duck;
        Swimmable swimmable = duck;

        flyable.fly();
        swimmable.swim();
    }

    /**
     * ============================================================
     *                3. 默认方法（Java 8+）
     * ============================================================
     */
    public static void defaultMethods() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("3. 默认方法（Java 8+）");
        System.out.println("=".repeat(60));

        Vehicle car = new Car();
        car.start();
        car.stop();
        car.horn();  // 使用默认实现

        System.out.println();

        Vehicle bike = new Bike();
        bike.start();
        bike.stop();
        bike.horn();  // 重写了默认方法

        // 解决默认方法冲突
        System.out.println("\n--- 解决默认方法冲突 ---");
        MultiInherit obj = new MultiInherit();
        obj.commonMethod();
    }

    /**
     * ============================================================
     *                4. 静态方法（Java 8+）
     * ============================================================
     */
    public static void staticMethods() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("4. 静态方法（Java 8+）");
        System.out.println("=".repeat(60));

        // 接口的静态方法
        Validator.validateNotNull("Hello");

        try {
            Validator.validateNotNull(null);
        } catch (IllegalArgumentException e) {
            System.out.println("捕获异常: " + e.getMessage());
        }

        System.out.println("isEmpty(\"\"): " + Validator.isEmpty(""));
        System.out.println("isEmpty(\"Hi\"): " + Validator.isEmpty("Hi"));
    }

    /**
     * ============================================================
     *                5. 函数式接口
     * ============================================================
     */
    public static void functionalInterfaces() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("5. 函数式接口");
        System.out.println("=".repeat(60));

        // 函数式接口：只有一个抽象方法的接口
        // 可以使用 Lambda 表达式

        // 匿名内部类方式
        Calculator add1 = new Calculator() {
            @Override
            public int calculate(int a, int b) {
                return a + b;
            }
        };

        // Lambda 表达式方式
        Calculator add2 = (a, b) -> a + b;
        Calculator subtract = (a, b) -> a - b;
        Calculator multiply = (a, b) -> a * b;

        System.out.println("add: " + add2.calculate(10, 5));
        System.out.println("subtract: " + subtract.calculate(10, 5));
        System.out.println("multiply: " + multiply.calculate(10, 5));

        // 方法引用
        System.out.println("\n--- 方法引用 ---");
        Calculator max = Math::max;
        Calculator min = Math::min;
        System.out.println("max(10, 5): " + max.calculate(10, 5));
        System.out.println("min(10, 5): " + min.calculate(10, 5));

        // 常用函数式接口
        System.out.println("\n--- 常用函数式接口 ---");
        System.out.println("""
            java.util.function 包中的常用接口：

            Predicate<T>     T -> boolean    判断
            Function<T,R>    T -> R          转换
            Consumer<T>      T -> void       消费
            Supplier<T>      () -> T         提供
            BiFunction<T,U,R> (T,U) -> R     双参数转换
            """);
    }

    /**
     * ============================================================
     *                6. 密封接口（Java 17+）
     * ============================================================
     */
    public static void sealedInterfaces() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("6. 密封接口（Java 17+）");
        System.out.println("=".repeat(60));

        System.out.println("""
            密封接口限制哪些类可以实现它：

            public sealed interface Shape
                permits Circle, Rectangle, Triangle {
                double area();
            }

            只有 permits 列出的类可以实现该接口。
            实现类必须是 final、sealed 或 non-sealed。
            """);

        // 使用密封接口
        SealedShape circle = new SealedCircle(5);
        SealedShape rect = new SealedRectangle(4, 3);

        System.out.println("圆形面积: " + circle.area());
        System.out.println("矩形面积: " + rect.area());

        // 模式匹配与密封类型
        printShape(circle);
        printShape(rect);
    }

    public static void printShape(SealedShape shape) {
        // 密封类型的模式匹配可以确保穷尽性
        String desc = switch (shape) {
            case SealedCircle c -> "圆形，半径 " + c.getRadius();
            case SealedRectangle r -> "矩形，" + r.getWidth() + "x" + r.getHeight();
        };
        System.out.println(desc);
    }
}

// ============================================================
//                    接口定义
// ============================================================

/**
 * 可飞行接口
 */
interface Flyable {
    // 接口常量（默认 public static final）
    int MAX_HEIGHT = 10000;

    // 抽象方法（默认 public abstract）
    void fly();
}

/**
 * 可游泳接口
 */
interface Swimmable {
    void swim();
}

/**
 * 车辆接口 - 带默认方法
 */
interface Vehicle {
    void start();
    void stop();

    // 默认方法（Java 8+）
    default void horn() {
        System.out.println("默认喇叭：嘟嘟！");
    }
}

/**
 * 接口 A - 用于演示默认方法冲突
 */
interface InterfaceA {
    default void commonMethod() {
        System.out.println("InterfaceA 的默认方法");
    }
}

/**
 * 接口 B - 用于演示默认方法冲突
 */
interface InterfaceB {
    default void commonMethod() {
        System.out.println("InterfaceB 的默认方法");
    }
}

/**
 * 验证器接口 - 带静态方法
 */
interface Validator {
    // 静态方法（Java 8+）
    static void validateNotNull(Object obj) {
        if (obj == null) {
            throw new IllegalArgumentException("对象不能为空");
        }
        System.out.println("验证通过: " + obj);
    }

    static boolean isEmpty(String str) {
        return str == null || str.isEmpty();
    }
}

/**
 * 计算器函数式接口
 */
@FunctionalInterface
interface Calculator {
    int calculate(int a, int b);

    // 默认方法不影响函数式接口
    default void printResult(int a, int b) {
        System.out.println("结果: " + calculate(a, b));
    }
}

/**
 * 密封接口（Java 17+）
 */
sealed interface SealedShape permits SealedCircle, SealedRectangle {
    double area();
}

// ============================================================
//                    实现类
// ============================================================

/**
 * 鸟类 - 实现 Flyable
 */
class Bird implements Flyable {
    @Override
    public void fly() {
        System.out.println("鸟在天空飞翔");
    }
}

/**
 * 飞机类 - 实现 Flyable
 */
class Airplane implements Flyable {
    @Override
    public void fly() {
        System.out.println("飞机在云层穿梭");
    }
}

/**
 * 鸭子类 - 实现多个接口
 */
class Duck implements Flyable, Swimmable {
    @Override
    public void fly() {
        System.out.println("鸭子拍打翅膀飞行");
    }

    @Override
    public void swim() {
        System.out.println("鸭子在水中游泳");
    }
}

/**
 * 汽车类
 */
class Car implements Vehicle {
    @Override
    public void start() {
        System.out.println("汽车启动");
    }

    @Override
    public void stop() {
        System.out.println("汽车停止");
    }
    // 使用默认的 horn() 方法
}

/**
 * 自行车类
 */
class Bike implements Vehicle {
    @Override
    public void start() {
        System.out.println("自行车开始骑行");
    }

    @Override
    public void stop() {
        System.out.println("自行车停下");
    }

    @Override
    public void horn() {
        System.out.println("自行车铃声：叮叮！");
    }
}

/**
 * 实现多个有冲突默认方法的接口
 */
class MultiInherit implements InterfaceA, InterfaceB {
    // 必须重写冲突的默认方法
    @Override
    public void commonMethod() {
        // 可以选择调用某个接口的默认方法
        InterfaceA.super.commonMethod();
        // 或者提供自己的实现
        System.out.println("MultiInherit 的实现");
    }
}

/**
 * 密封接口的实现类 - 圆形
 */
final class SealedCircle implements SealedShape {
    private final double radius;

    public SealedCircle(double radius) {
        this.radius = radius;
    }

    public double getRadius() {
        return radius;
    }

    @Override
    public double area() {
        return Math.PI * radius * radius;
    }
}

/**
 * 密封接口的实现类 - 矩形
 */
final class SealedRectangle implements SealedShape {
    private final double width;
    private final double height;

    public SealedRectangle(double width, double height) {
        this.width = width;
        this.height = height;
    }

    public double getWidth() {
        return width;
    }

    public double getHeight() {
        return height;
    }

    @Override
    public double area() {
        return width * height;
    }
}
