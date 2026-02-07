# ControlFlow

::: info 文件信息
- 📄 原文件：`ControlFlow.java`
- 🔤 语言：java
:::

Java 控制流
本文件介绍 Java 中的条件语句、循环、跳转语句等控制流结构。

## 完整代码

```java
/**
 * ============================================================
 *                    Java 控制流
 * ============================================================
 * 本文件介绍 Java 中的条件语句、循环、跳转语句等控制流结构。
 * ============================================================
 */
public class ControlFlow {

    public static void main(String[] args) {
        ifElseStatements();
        switchStatements();
        switchExpressions();
        forLoops();
        whileLoops();
        breakAndContinue();
        exceptionHandling();
    }

    /**
     * ============================================================
     *                    1. if-else 语句
     * ============================================================
     */
    public static void ifElseStatements() {
        System.out.println("=".repeat(60));
        System.out.println("1. if-else 语句");
        System.out.println("=".repeat(60));

        int score = 85;

        // 【基本 if】
        if (score >= 60) {
            System.out.println("及格了！");
        }

        // 【if-else】
        if (score >= 90) {
            System.out.println("优秀");
        } else {
            System.out.println("继续努力");
        }

        // 【if-else if-else】
        String grade;
        if (score >= 90) {
            grade = "A";
        } else if (score >= 80) {
            grade = "B";
        } else if (score >= 70) {
            grade = "C";
        } else if (score >= 60) {
            grade = "D";
        } else {
            grade = "F";
        }
        System.out.println("分数 " + score + " 的等级: " + grade);

        // 【三元运算符】
        String result = (score >= 60) ? "通过" : "不通过";
        System.out.println("三元运算符: " + result);

        // 【嵌套三元运算符】（不推荐，可读性差）
        String level = score >= 90 ? "优秀" :
                       score >= 60 ? "及格" : "不及格";
        System.out.println("嵌套三元: " + level);

        // 【逻辑运算符】
        int age = 25;
        boolean hasLicense = true;

        if (age >= 18 && hasLicense) {
            System.out.println("\n可以开车");
        }

        if (age < 18 || !hasLicense) {
            System.out.println("不能开车");
        }
    }

    /**
     * ============================================================
     *                    2. switch 语句（传统）
     * ============================================================
     */
    public static void switchStatements() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. switch 语句（传统）");
        System.out.println("=".repeat(60));

        // 【基本 switch】
        int day = 3;
        String dayName;

        switch (day) {
            case 1:
                dayName = "星期一";
                break;
            case 2:
                dayName = "星期二";
                break;
            case 3:
                dayName = "星期三";
                break;
            case 4:
                dayName = "星期四";
                break;
            case 5:
                dayName = "星期五";
                break;
            case 6:
            case 7:
                dayName = "周末";
                break;
            default:
                dayName = "无效";
        }
        System.out.println("day " + day + " = " + dayName);

        // 【字符串 switch】（Java 7+）
        String fruit = "apple";
        String color;

        switch (fruit) {
            case "apple":
                color = "红色";
                break;
            case "banana":
                color = "黄色";
                break;
            case "grape":
                color = "紫色";
                break;
            default:
                color = "未知";
        }
        System.out.println(fruit + " 是 " + color);

        // 【fall-through】（忘记 break 的危险）
        System.out.println("\n【警告】忘记 break 会导致 fall-through:");
        int month = 2;
        switch (month) {
            case 1:
                System.out.println("  一月");
                // 忘记 break，会继续执行下一个 case
            case 2:
                System.out.println("  二月");
                break;
            default:
                System.out.println("  其他");
        }
    }

    /**
     * ============================================================
     *                3. switch 表达式（Java 14+）
     * ============================================================
     */
    public static void switchExpressions() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("3. switch 表达式（Java 14+）");
        System.out.println("=".repeat(60));

        int day = 3;

        // 【箭头语法】不需要 break
        String dayType = switch (day) {
            case 1, 2, 3, 4, 5 -> "工作日";
            case 6, 7 -> "周末";
            default -> "无效";
        };
        System.out.println("day " + day + " 是 " + dayType);

        // 【yield 返回值】（多行代码时使用）
        String description = switch (day) {
            case 1 -> "周一，新的开始";
            case 5 -> "周五，快乐的一天";
            case 6, 7 -> {
                String msg = "周末";
                msg += "，休息时间";
                yield msg;  // 使用 yield 返回值
            }
            default -> "普通的一天";
        };
        System.out.println(description);

        // 【模式匹配】（Java 21+）
        Object obj = "Hello";
        String result = switch (obj) {
            case Integer i -> "整数: " + i;
            case String s -> "字符串: " + s;
            case Double d -> "浮点数: " + d;
            case null -> "空值";
            default -> "未知类型";
        };
        System.out.println("\n模式匹配: " + result);
    }

    /**
     * ============================================================
     *                    4. for 循环
     * ============================================================
     */
    public static void forLoops() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("4. for 循环");
        System.out.println("=".repeat(60));

        // 【基本 for 循环】
        System.out.println("--- 基本 for 循环 ---");
        for (int i = 0; i < 5; i++) {
            System.out.print(i + " ");
        }
        System.out.println();

        // 【递减循环】
        System.out.println("\n递减:");
        for (int i = 5; i > 0; i--) {
            System.out.print(i + " ");
        }
        System.out.println();

        // 【步长】
        System.out.println("\n步长为2:");
        for (int i = 0; i < 10; i += 2) {
            System.out.print(i + " ");
        }
        System.out.println();

        // 【增强 for 循环（for-each）】
        System.out.println("\n--- 增强 for 循环 ---");
        int[] numbers = {1, 2, 3, 4, 5};
        for (int num : numbers) {
            System.out.print(num + " ");
        }
        System.out.println();

        String[] fruits = {"apple", "banana", "cherry"};
        for (String fruit : fruits) {
            System.out.println("  " + fruit);
        }

        // 【嵌套循环】
        System.out.println("\n--- 嵌套循环（乘法表）---");
        for (int i = 1; i <= 3; i++) {
            for (int j = 1; j <= 3; j++) {
                System.out.print(i + "×" + j + "=" + (i * j) + "\t");
            }
            System.out.println();
        }

        // 【无限循环】
        // for (;;) { }  // 等同于 while (true)
    }

    /**
     * ============================================================
     *                    5. while 和 do-while 循环
     * ============================================================
     */
    public static void whileLoops() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("5. while 和 do-while 循环");
        System.out.println("=".repeat(60));

        // 【while 循环】
        System.out.println("--- while 循环 ---");
        int count = 0;
        while (count < 5) {
            System.out.print(count + " ");
            count++;
        }
        System.out.println();

        // 【do-while 循环】至少执行一次
        System.out.println("\n--- do-while 循环 ---");
        int n = 0;
        do {
            System.out.print(n + " ");
            n++;
        } while (n < 5);
        System.out.println();

        // 【区别】
        System.out.println("\n--- while vs do-while ---");
        int x = 10;

        // while: 条件为 false，不执行
        while (x < 5) {
            System.out.println("while: " + x);
        }

        // do-while: 条件为 false，也执行一次
        do {
            System.out.println("do-while: " + x + " (至少执行一次)");
        } while (x < 5);
    }

    /**
     * ============================================================
     *                    6. break 和 continue
     * ============================================================
     */
    public static void breakAndContinue() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("6. break 和 continue");
        System.out.println("=".repeat(60));

        // 【break】跳出循环
        System.out.println("--- break ---");
        for (int i = 0; i < 10; i++) {
            if (i == 5) {
                System.out.println("遇到 5，跳出循环");
                break;
            }
            System.out.print(i + " ");
        }
        System.out.println();

        // 【continue】跳过本次迭代
        System.out.println("\n--- continue ---");
        for (int i = 0; i < 10; i++) {
            if (i % 2 == 0) {
                continue;  // 跳过偶数
            }
            System.out.print(i + " ");
        }
        System.out.println();

        // 【带标签的 break/continue】用于嵌套循环
        System.out.println("\n--- 带标签的 break ---");
        outer:
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i == 1 && j == 1) {
                    System.out.println("跳出外层循环");
                    break outer;
                }
                System.out.println("  i=" + i + ", j=" + j);
            }
        }
    }

    /**
     * ============================================================
     *                    7. 异常处理
     * ============================================================
     */
    public static void exceptionHandling() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("7. 异常处理");
        System.out.println("=".repeat(60));

        // 【try-catch】
        System.out.println("--- try-catch ---");
        try {
            int result = 10 / 0;  // ArithmeticException
            System.out.println(result);
        } catch (ArithmeticException e) {
            System.out.println("捕获异常: " + e.getMessage());
        }

        // 【多个 catch】
        System.out.println("\n--- 多个 catch ---");
        try {
            String s = null;
            System.out.println(s.length());  // NullPointerException
        } catch (NullPointerException e) {
            System.out.println("空指针异常");
        } catch (Exception e) {
            System.out.println("其他异常: " + e);
        }

        // 【多异常捕获】（Java 7+）
        System.out.println("\n--- 多异常捕获 ---");
        try {
            int[] arr = {1, 2, 3};
            System.out.println(arr[10]);  // ArrayIndexOutOfBoundsException
        } catch (ArrayIndexOutOfBoundsException | NullPointerException e) {
            System.out.println("数组或空指针异常: " + e.getClass().getSimpleName());
        }

        // 【try-catch-finally】
        System.out.println("\n--- try-catch-finally ---");
        try {
            System.out.println("try 块");
            // int x = 1 / 0;
        } catch (Exception e) {
            System.out.println("catch 块");
        } finally {
            System.out.println("finally 块（总是执行）");
        }

        // 【try-with-resources】（Java 7+）
        System.out.println("\n--- try-with-resources ---");
        // 自动关闭资源
        try (var scanner = new java.util.Scanner("Hello World")) {
            System.out.println("读取: " + scanner.next());
        }  // 自动调用 scanner.close()

        // 【抛出异常】
        System.out.println("\n--- 抛出异常 ---");
        try {
            validateAge(-5);
        } catch (IllegalArgumentException e) {
            System.out.println("捕获: " + e.getMessage());
        }

        // 【自定义异常】
        System.out.println("\n--- 自定义异常 ---");
        try {
            throw new CustomException("自定义错误信息");
        } catch (CustomException e) {
            System.out.println("捕获自定义异常: " + e.getMessage());
        }
    }

    // 抛出异常的方法
    public static void validateAge(int age) {
        if (age < 0) {
            throw new IllegalArgumentException("年龄不能为负: " + age);
        }
    }

    // 自定义异常类
    static class CustomException extends Exception {
        public CustomException(String message) {
            super(message);
        }
    }
}
```
