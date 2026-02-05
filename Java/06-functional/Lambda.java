import java.util.*;
import java.util.function.*;

/**
 * ============================================================
 *                    Java Lambda 表达式
 * ============================================================
 * 本文件介绍 Java 中的 Lambda 表达式和函数式接口。
 * ============================================================
 */
public class Lambda {

    public static void main(String[] args) {
        lambdaBasics();
        functionalInterfaces();
        builtInFunctions();
        methodReferences();
        closures();
    }

    /**
     * ============================================================
     *                    1. Lambda 基础
     * ============================================================
     */
    public static void lambdaBasics() {
        System.out.println("=".repeat(60));
        System.out.println("1. Lambda 基础");
        System.out.println("=".repeat(60));

        System.out.println("""
            Lambda 表达式语法：
            (参数列表) -> { 方法体 }

            简化规则：
            - 单个参数可省略括号：x -> x * 2
            - 单条语句可省略大括号和 return
            - 参数类型可推断时可省略
            """);

        // 【完整语法】
        System.out.println("--- Lambda 语法形式 ---");

        // 无参数
        Runnable r1 = () -> System.out.println("无参数 Lambda");
        r1.run();

        // 单个参数
        Consumer<String> c1 = s -> System.out.println("单参数: " + s);
        c1.accept("Hello");

        // 多个参数
        BiFunction<Integer, Integer, Integer> add = (a, b) -> a + b;
        System.out.println("多参数: " + add.apply(3, 5));

        // 带类型声明
        BiFunction<String, String, String> concat = (String a, String b) -> a + b;
        System.out.println("带类型: " + concat.apply("Hello, ", "World"));

        // 多语句
        BiFunction<Integer, Integer, Integer> complex = (a, b) -> {
            int sum = a + b;
            int product = a * b;
            return sum + product;
        };
        System.out.println("多语句: " + complex.apply(3, 4));

        // 【替代匿名内部类】
        System.out.println("\n--- 替代匿名内部类 ---");

        // 传统方式
        Comparator<String> comp1 = new Comparator<String>() {
            @Override
            public int compare(String s1, String s2) {
                return s1.length() - s2.length();
            }
        };

        // Lambda 方式
        Comparator<String> comp2 = (s1, s2) -> s1.length() - s2.length();

        List<String> words = new ArrayList<>(List.of("apple", "pie", "banana"));
        words.sort(comp2);
        System.out.println("按长度排序: " + words);
    }

    /**
     * ============================================================
     *                    2. 函数式接口
     * ============================================================
     */
    public static void functionalInterfaces() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. 函数式接口");
        System.out.println("=".repeat(60));

        System.out.println("""
            函数式接口：只有一个抽象方法的接口
            - 可以有默认方法和静态方法
            - @FunctionalInterface 注解用于编译时检查
            """);

        // 【自定义函数式接口】
        System.out.println("--- 自定义函数式接口 ---");

        // 定义接口见文件末尾
        Calculator calc = (a, b) -> a + b;
        System.out.println("calc.calculate(10, 5) = " + calc.calculate(10, 5));

        // 使用默认方法
        calc.printResult(10, 5);

        // 不同实现
        Calculator sub = (a, b) -> a - b;
        Calculator mul = (a, b) -> a * b;
        Calculator div = (a, b) -> b != 0 ? a / b : 0;

        System.out.println("sub: " + sub.calculate(10, 5));
        System.out.println("mul: " + mul.calculate(10, 5));
        System.out.println("div: " + div.calculate(10, 5));

        // 【作为参数传递】
        System.out.println("\n--- Lambda 作为参数 ---");
        processNumbers(10, 5, (a, b) -> a + b, "加法");
        processNumbers(10, 5, (a, b) -> a * b, "乘法");
    }

    public static void processNumbers(int a, int b, Calculator calc, String operation) {
        System.out.println(operation + ": " + calc.calculate(a, b));
    }

    /**
     * ============================================================
     *                    3. 内置函数式接口
     * ============================================================
     */
    public static void builtInFunctions() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("3. 内置函数式接口");
        System.out.println("=".repeat(60));

        // 【Predicate<T>】T -> boolean
        System.out.println("--- Predicate<T> ---");
        Predicate<Integer> isPositive = n -> n > 0;
        Predicate<Integer> isEven = n -> n % 2 == 0;

        System.out.println("isPositive(5): " + isPositive.test(5));
        System.out.println("isEven(4): " + isEven.test(4));

        // 组合 Predicate
        Predicate<Integer> positiveAndEven = isPositive.and(isEven);
        Predicate<Integer> positiveOrEven = isPositive.or(isEven);
        Predicate<Integer> notPositive = isPositive.negate();

        System.out.println("positiveAndEven(4): " + positiveAndEven.test(4));
        System.out.println("positiveOrEven(-2): " + positiveOrEven.test(-2));
        System.out.println("notPositive(-1): " + notPositive.test(-1));

        // 【Function<T, R>】T -> R
        System.out.println("\n--- Function<T, R> ---");
        Function<String, Integer> length = String::length;
        Function<String, String> toUpper = String::toUpperCase;

        System.out.println("length(\"Hello\"): " + length.apply("Hello"));
        System.out.println("toUpper(\"hello\"): " + toUpper.apply("hello"));

        // 组合 Function
        Function<String, Integer> upperThenLength = toUpper.andThen(length);
        System.out.println("upperThenLength(\"hello\"): " + upperThenLength.apply("hello"));

        // 【Consumer<T>】T -> void
        System.out.println("\n--- Consumer<T> ---");
        Consumer<String> print = System.out::println;
        Consumer<String> printUpper = s -> System.out.println(s.toUpperCase());

        print.accept("hello consumer");
        print.andThen(printUpper).accept("chain");

        // 【Supplier<T>】() -> T
        System.out.println("\n--- Supplier<T> ---");
        Supplier<Double> random = Math::random;
        Supplier<String> greeting = () -> "Hello, Supplier!";

        System.out.println("random: " + random.get());
        System.out.println("greeting: " + greeting.get());

        // 【BiFunction<T, U, R>】(T, U) -> R
        System.out.println("\n--- BiFunction<T, U, R> ---");
        BiFunction<String, String, String> combine = (a, b) -> a + " " + b;
        System.out.println("combine: " + combine.apply("Hello", "World"));

        // 【UnaryOperator<T>】T -> T
        System.out.println("\n--- UnaryOperator<T> ---");
        UnaryOperator<Integer> square = n -> n * n;
        System.out.println("square(5): " + square.apply(5));

        // 【BinaryOperator<T>】(T, T) -> T
        System.out.println("\n--- BinaryOperator<T> ---");
        BinaryOperator<Integer> max = Integer::max;
        System.out.println("max(3, 7): " + max.apply(3, 7));

        // 【完整列表】
        System.out.println("\n--- 常用函数式接口汇总 ---");
        System.out.println("""
            Predicate<T>        T -> boolean      判断
            Function<T,R>       T -> R            转换
            Consumer<T>         T -> void         消费
            Supplier<T>         () -> T           提供
            BiPredicate<T,U>    (T,U) -> boolean  双参判断
            BiFunction<T,U,R>   (T,U) -> R        双参转换
            BiConsumer<T,U>     (T,U) -> void     双参消费
            UnaryOperator<T>    T -> T            一元操作
            BinaryOperator<T>   (T,T) -> T        二元操作

            基本类型特化版本：
            IntPredicate, LongPredicate, DoublePredicate
            IntFunction<R>, LongFunction<R>, DoubleFunction<R>
            IntConsumer, LongConsumer, DoubleConsumer
            IntSupplier, LongSupplier, DoubleSupplier
            """);
    }

    /**
     * ============================================================
     *                    4. 方法引用
     * ============================================================
     */
    public static void methodReferences() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("4. 方法引用");
        System.out.println("=".repeat(60));

        System.out.println("""
            方法引用是 Lambda 的简写形式：
            - 类名::静态方法
            - 对象::实例方法
            - 类名::实例方法
            - 类名::new（构造器引用）
            """);

        List<String> words = List.of("apple", "Banana", "cherry");

        // 【静态方法引用】类名::静态方法
        System.out.println("--- 静态方法引用 ---");
        // Lambda: s -> Integer.parseInt(s)
        Function<String, Integer> parse = Integer::parseInt;
        System.out.println("parse(\"42\"): " + parse.apply("42"));

        // 【实例方法引用 - 对象】对象::实例方法
        System.out.println("\n--- 对象的实例方法引用 ---");
        String prefix = "-> ";
        // Lambda: s -> prefix.concat(s)
        Function<String, String> addPrefix = prefix::concat;
        System.out.println("addPrefix(\"hello\"): " + addPrefix.apply("hello"));

        // 【实例方法引用 - 类】类名::实例方法
        System.out.println("\n--- 类的实例方法引用 ---");
        // Lambda: s -> s.toUpperCase()
        Function<String, String> toUpper = String::toUpperCase;
        System.out.println("toUpper(\"hello\"): " + toUpper.apply("hello"));

        // Lambda: (s1, s2) -> s1.compareTo(s2)
        Comparator<String> comparator = String::compareTo;
        System.out.println("compare: " + comparator.compare("a", "b"));

        // 【构造器引用】类名::new
        System.out.println("\n--- 构造器引用 ---");
        // Lambda: () -> new ArrayList<>()
        Supplier<List<String>> listFactory = ArrayList::new;
        List<String> newList = listFactory.get();
        System.out.println("newList: " + newList.getClass().getSimpleName());

        // 带参数的构造器
        Function<Integer, StringBuilder> sbFactory = StringBuilder::new;
        StringBuilder sb = sbFactory.apply(100);  // 指定容量
        System.out.println("StringBuilder capacity: " + sb.capacity());

        // 【数组构造器引用】
        System.out.println("\n--- 数组构造器引用 ---");
        // Lambda: size -> new String[size]
        IntFunction<String[]> arrayFactory = String[]::new;
        String[] array = arrayFactory.apply(5);
        System.out.println("array length: " + array.length);

        // 【实际应用】
        System.out.println("\n--- 实际应用 ---");
        List<String> names = new ArrayList<>(List.of("Alice", "Bob", "Charlie"));

        // forEach 使用方法引用
        names.forEach(System.out::println);

        // sort 使用方法引用
        names.sort(String::compareToIgnoreCase);
        System.out.println("Sorted: " + names);
    }

    /**
     * ============================================================
     *                    5. 闭包与变量捕获
     * ============================================================
     */
    public static void closures() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("5. 闭包与变量捕获");
        System.out.println("=".repeat(60));

        System.out.println("""
            Lambda 可以捕获外部变量：
            - 只能捕获 effectively final 的变量
            - 变量在 Lambda 中不能被修改
            - Lambda 执行时使用的是变量的副本
            """);

        // 【捕获局部变量】
        System.out.println("--- 捕获局部变量 ---");
        int multiplier = 10;  // effectively final
        Function<Integer, Integer> multiply = n -> n * multiplier;
        System.out.println("multiply(5): " + multiply.apply(5));

        // multiplier = 20;  // 编译错误！一旦被 Lambda 捕获就不能修改

        // 【使用数组或对象绕过限制】
        System.out.println("\n--- 使用容器绕过限制 ---");
        int[] counter = {0};  // 使用数组
        Runnable increment = () -> counter[0]++;
        increment.run();
        increment.run();
        increment.run();
        System.out.println("counter: " + counter[0]);

        // 【this 引用】
        System.out.println("\n--- Lambda 中的 this ---");
        System.out.println("""
            Lambda 中的 this:
            - Lambda 中的 this 指向包含它的类的实例
            - 匿名内部类中的 this 指向匿名类实例
            """);

        // 【作用域】
        System.out.println("\n--- Lambda 作用域 ---");
        String message = "Hello";
        Consumer<String> printer = (text) -> {
            // String message = "World";  // 编译错误！与外部变量同名
            System.out.println(message + " " + text);
        };
        printer.accept("Lambda");
    }
}

/**
 * 自定义函数式接口
 */
@FunctionalInterface
interface Calculator {
    int calculate(int a, int b);

    // 默认方法不影响函数式接口
    default void printResult(int a, int b) {
        System.out.println("Result: " + calculate(a, b));
    }

    // 静态方法也不影响
    static Calculator add() {
        return (a, b) -> a + b;
    }
}
