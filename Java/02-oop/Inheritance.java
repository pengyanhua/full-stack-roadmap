/**
 * ============================================================
 *                    Java 继承与多态
 * ============================================================
 * 本文件介绍 Java 中的继承、方法重写、多态等概念。
 * ============================================================
 */
public class Inheritance {

    public static void main(String[] args) {
        basicInheritance();
        methodOverriding();
        polymorphism();
        abstractClasses();
        finalKeyword();
        objectMethods();
    }

    /**
     * ============================================================
     *                    1. 基本继承
     * ============================================================
     */
    public static void basicInheritance() {
        System.out.println("=".repeat(60));
        System.out.println("1. 基本继承");
        System.out.println("=".repeat(60));

        // 创建子类对象
        Dog dog = new Dog("旺财", 3);
        dog.eat();      // 继承的方法
        dog.bark();     // 子类特有方法
        System.out.println();

        Cat cat = new Cat("咪咪", 2);
        cat.eat();
        cat.meow();

        // 访问继承的属性
        System.out.println("\n" + dog.name + " 是一只 " + dog.getAge() + " 岁的狗");
    }

    /**
     * ============================================================
     *                    2. 方法重写
     * ============================================================
     */
    public static void methodOverriding() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. 方法重写");
        System.out.println("=".repeat(60));

        Animal animal = new Animal("动物", 1);
        Dog dog = new Dog("旺财", 3);
        Cat cat = new Cat("咪咪", 2);

        // 调用重写的方法
        System.out.println("--- 调用 speak() 方法 ---");
        animal.speak();  // Animal 的实现
        dog.speak();     // Dog 重写的实现
        cat.speak();     // Cat 重写的实现

        // super 关键字
        System.out.println("\n--- super 关键字 ---");
        dog.eatWithSuper();
    }

    /**
     * ============================================================
     *                    3. 多态
     * ============================================================
     */
    public static void polymorphism() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("3. 多态");
        System.out.println("=".repeat(60));

        // 【向上转型】父类引用指向子类对象
        System.out.println("--- 向上转型 ---");
        Animal animal1 = new Dog("旺财", 3);
        Animal animal2 = new Cat("咪咪", 2);

        animal1.speak();  // 调用的是 Dog 的 speak()
        animal2.speak();  // 调用的是 Cat 的 speak()

        // 【多态数组】
        System.out.println("\n--- 多态数组 ---");
        Animal[] animals = {
            new Dog("小黑", 1),
            new Cat("小白", 2),
            new Dog("大黄", 4)
        };

        for (Animal animal : animals) {
            animal.speak();  // 运行时决定调用哪个方法
        }

        // 【向下转型】需要强制类型转换
        System.out.println("\n--- 向下转型 ---");
        Animal a = new Dog("旺财", 3);

        if (a instanceof Dog) {
            Dog d = (Dog) a;  // 向下转型
            d.bark();         // 可以调用 Dog 特有方法
        }

        // 【instanceof 模式匹配】（Java 16+）
        if (a instanceof Dog d2) {  // 直接声明变量
            d2.bark();
        }

        // 【错误的向下转型】
        Animal a2 = new Cat("咪咪", 2);
        // Dog d = (Dog) a2;  // ClassCastException!

        // 【多态参数】
        System.out.println("\n--- 多态参数 ---");
        feedAnimal(new Dog("小狗", 1));
        feedAnimal(new Cat("小猫", 1));
    }

    public static void feedAnimal(Animal animal) {
        System.out.println("喂食 " + animal.name);
        animal.eat();
    }

    /**
     * ============================================================
     *                    4. 抽象类
     * ============================================================
     */
    public static void abstractClasses() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("4. 抽象类");
        System.out.println("=".repeat(60));

        // Shape shape = new Shape();  // 错误！不能实例化抽象类

        Shape circle = new Circle(5);
        Shape rectangle = new Rectangle(4, 3);

        System.out.println("圆形面积: " + circle.area());
        System.out.println("圆形周长: " + circle.perimeter());
        System.out.println();
        System.out.println("矩形面积: " + rectangle.area());
        System.out.println("矩形周长: " + rectangle.perimeter());

        // 调用抽象类中的具体方法
        circle.describe();
    }

    /**
     * ============================================================
     *                    5. final 关键字
     * ============================================================
     */
    public static void finalKeyword() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("5. final 关键字");
        System.out.println("=".repeat(60));

        System.out.println("""
            final 用途：

            1. final 变量：常量，不能重新赋值
               final int MAX = 100;

            2. final 方法：不能被子类重写
               public final void method() { }

            3. final 类：不能被继承
               public final class String { }

            4. final 参数：方法内不能修改
               void method(final int x) { }
            """);

        // final 变量
        final int MAX_VALUE = 100;
        // MAX_VALUE = 200;  // 编译错误

        // final 引用类型
        final StringBuilder sb = new StringBuilder("Hello");
        sb.append(" World");  // 可以修改对象内容
        // sb = new StringBuilder();  // 编译错误，不能重新赋值
        System.out.println("final StringBuilder: " + sb);
    }

    /**
     * ============================================================
     *                    6. Object 类方法
     * ============================================================
     */
    public static void objectMethods() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("6. Object 类方法");
        System.out.println("=".repeat(60));

        // 所有类都继承自 Object

        Student s1 = new Student("Alice", 20);
        Student s2 = new Student("Alice", 20);
        Student s3 = s1;

        // 【toString()】
        System.out.println("--- toString() ---");
        System.out.println("s1.toString(): " + s1.toString());
        System.out.println("直接打印: " + s1);

        // 【equals()】
        System.out.println("\n--- equals() ---");
        System.out.println("s1 == s2: " + (s1 == s2));           // false
        System.out.println("s1.equals(s2): " + s1.equals(s2));   // true
        System.out.println("s1 == s3: " + (s1 == s3));           // true

        // 【hashCode()】
        System.out.println("\n--- hashCode() ---");
        System.out.println("s1.hashCode(): " + s1.hashCode());
        System.out.println("s2.hashCode(): " + s2.hashCode());
        System.out.println("相等对象的 hashCode 应该相同: " + (s1.hashCode() == s2.hashCode()));

        // 【getClass()】
        System.out.println("\n--- getClass() ---");
        System.out.println("s1.getClass(): " + s1.getClass());
        System.out.println("s1.getClass().getName(): " + s1.getClass().getName());
        System.out.println("s1.getClass().getSimpleName(): " + s1.getClass().getSimpleName());

        // 【clone()】需要实现 Cloneable 接口
        System.out.println("\n--- clone() ---");
        try {
            Student s4 = (Student) s1.clone();
            System.out.println("克隆后: " + s4);
            System.out.println("s1 == s4: " + (s1 == s4));
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
    }
}

// ============================================================
//                    辅助类定义
// ============================================================

/**
 * 动物基类
 */
class Animal {
    protected String name;
    private int age;

    public Animal(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public void eat() {
        System.out.println(name + " 正在吃东西");
    }

    public void speak() {
        System.out.println(name + " 发出声音");
    }

    public int getAge() {
        return age;
    }
}

/**
 * 狗类 - 继承 Animal
 */
class Dog extends Animal {

    public Dog(String name, int age) {
        super(name, age);  // 调用父类构造函数
    }

    // 子类特有方法
    public void bark() {
        System.out.println(name + ": 汪汪汪!");
    }

    // 重写父类方法
    @Override
    public void speak() {
        System.out.println(name + ": 汪汪!");
    }

    // 使用 super 调用父类方法
    public void eatWithSuper() {
        super.eat();  // 调用父类的 eat()
        System.out.println("  (狗狗吃得很香)");
    }
}

/**
 * 猫类 - 继承 Animal
 */
class Cat extends Animal {

    public Cat(String name, int age) {
        super(name, age);
    }

    public void meow() {
        System.out.println(name + ": 喵喵喵~");
    }

    @Override
    public void speak() {
        System.out.println(name + ": 喵~");
    }
}

/**
 * 形状抽象类
 */
abstract class Shape {
    protected String color = "white";

    // 抽象方法：子类必须实现
    public abstract double area();
    public abstract double perimeter();

    // 具体方法：子类可以继承
    public void describe() {
        System.out.println("这是一个 " + color + " 的形状");
        System.out.println("面积: " + area() + ", 周长: " + perimeter());
    }
}

/**
 * 圆形
 */
class Circle extends Shape {
    private double radius;

    public Circle(double radius) {
        this.radius = radius;
    }

    @Override
    public double area() {
        return Math.PI * radius * radius;
    }

    @Override
    public double perimeter() {
        return 2 * Math.PI * radius;
    }
}

/**
 * 矩形
 */
class Rectangle extends Shape {
    private double width;
    private double height;

    public Rectangle(double width, double height) {
        this.width = width;
        this.height = height;
    }

    @Override
    public double area() {
        return width * height;
    }

    @Override
    public double perimeter() {
        return 2 * (width + height);
    }
}

/**
 * 学生类 - 演示 Object 方法重写
 */
class Student implements Cloneable {
    private String name;
    private int age;

    public Student(String name, int age) {
        this.name = name;
        this.age = age;
    }

    @Override
    public String toString() {
        return "Student{name='" + name + "', age=" + age + "}";
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Student student = (Student) obj;
        return age == student.age && name.equals(student.name);
    }

    @Override
    public int hashCode() {
        return java.util.Objects.hash(name, age);
    }

    @Override
    protected Object clone() throws CloneNotSupportedException {
        return super.clone();
    }
}
