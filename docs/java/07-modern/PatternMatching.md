# PatternMatching

::: info 文件信息
- 📄 原文件：`PatternMatching.java`
- 🔤 语言：java
:::

Java 模式匹配
本文件介绍 Java 中的模式匹配特性。

## 完整代码

```java
import java.util.*;

/**
 * ============================================================
 *                    Java 模式匹配
 * ============================================================
 * 本文件介绍 Java 中的模式匹配特性。
 * ============================================================
 */
public class PatternMatching {

    public static void main(String[] args) {
        instanceofPattern();
        switchExpressions();
        switchPatterns();
        guardedPatterns();
        sealedTypesPattern();
    }

    /**
     * ============================================================
     *                    1. instanceof 模式匹配
     * ============================================================
     */
    public static void instanceofPattern() {
        System.out.println("=".repeat(60));
        System.out.println("1. instanceof 模式匹配");
        System.out.println("=".repeat(60));

        System.out.println("""
            instanceof 模式匹配（Java 16+）
            - 类型检查和类型转换合二为一
            - 模式变量的作用域由编译器推断
            """);

        // 【传统方式 vs 模式匹配】
        System.out.println("--- 传统方式 vs 模式匹配 ---");
        Object obj = "Hello, Pattern Matching!";

        // 传统方式
        if (obj instanceof String) {
            String s = (String) obj;
            System.out.println("传统方式: " + s.toUpperCase());
        }

        // 模式匹配
        if (obj instanceof String s) {
            System.out.println("模式匹配: " + s.toUpperCase());
        }

        // 【作用域规则】
        System.out.println("\n--- 作用域规则 ---");
        Object value = 42;

        // 模式变量在 true 分支中可用
        if (value instanceof Integer i) {
            System.out.println("整数: " + (i * 2));
        }

        // 在 false 分支后也可用（因为流程确定）
        if (!(value instanceof Integer i)) {
            System.out.println("不是整数");
        } else {
            System.out.println("是整数: " + i);
        }

        // 【与 && 组合】
        System.out.println("\n--- 与 && 组合 ---");
        Object data = "Hello";

        if (data instanceof String str && str.length() > 3) {
            System.out.println("长度大于3的字符串: " + str);
        }

        // 【处理多种类型】
        System.out.println("\n--- 处理多种类型 ---");
        Object[] items = {42, "Hello", 3.14, true, List.of(1, 2, 3)};

        for (Object item : items) {
            describeObject(item);
        }
    }

    public static void describeObject(Object obj) {
        String desc;
        if (obj instanceof Integer i) {
            desc = "整数: " + i + ", 平方: " + (i * i);
        } else if (obj instanceof String s) {
            desc = "字符串: " + s + ", 长度: " + s.length();
        } else if (obj instanceof Double d) {
            desc = "浮点数: " + d;
        } else if (obj instanceof Boolean b) {
            desc = "布尔值: " + b;
        } else if (obj instanceof List<?> list) {
            desc = "列表: " + list + ", 大小: " + list.size();
        } else {
            desc = "未知类型: " + obj.getClass().getSimpleName();
        }
        System.out.println("  " + desc);
    }

    /**
     * ============================================================
     *                    2. switch 表达式
     * ============================================================
     */
    public static void switchExpressions() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. switch 表达式");
        System.out.println("=".repeat(60));

        System.out.println("""
            switch 表达式（Java 14+）
            - 可以作为表达式返回值
            - 使用 -> 箭头语法
            - 不需要 break
            - yield 用于代码块中返回值
            """);

        // 【传统 switch vs switch 表达式】
        System.out.println("--- 传统 vs 表达式 ---");
        int day = 3;

        // 传统方式
        String dayName1;
        switch (day) {
            case 1:
                dayName1 = "周一";
                break;
            case 2:
                dayName1 = "周二";
                break;
            case 3:
                dayName1 = "周三";
                break;
            default:
                dayName1 = "其他";
        }

        // switch 表达式
        String dayName2 = switch (day) {
            case 1 -> "周一";
            case 2 -> "周二";
            case 3 -> "周三";
            case 4 -> "周四";
            case 5 -> "周五";
            case 6, 7 -> "周末";  // 多个标签
            default -> "无效";
        };

        System.out.println("传统: " + dayName1);
        System.out.println("表达式: " + dayName2);

        // 【yield 关键字】
        System.out.println("\n--- yield 关键字 ---");
        int month = 2;
        int days = switch (month) {
            case 1, 3, 5, 7, 8, 10, 12 -> 31;
            case 4, 6, 9, 11 -> 30;
            case 2 -> {
                // 复杂逻辑用代码块
                System.out.println("  计算二月天数...");
                yield 28;  // 简化，不考虑闰年
            }
            default -> throw new IllegalArgumentException("无效月份");
        };
        System.out.println(month + "月有 " + days + " 天");

        // 【enum switch】
        System.out.println("\n--- enum switch ---");
        DayOfWeek dow = DayOfWeek.WEDNESDAY;

        String type = switch (dow) {
            case MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY -> "工作日";
            case SATURDAY, SUNDAY -> "周末";
        };
        System.out.println(dow + " 是 " + type);

        // 【switch 表达式必须完备】
        System.out.println("\n【注意】switch 表达式必须覆盖所有情况（穷尽性）");
    }

    enum DayOfWeek {
        MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY
    }

    /**
     * ============================================================
     *                    3. switch 模式匹配
     * ============================================================
     */
    public static void switchPatterns() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("3. switch 模式匹配");
        System.out.println("=".repeat(60));

        System.out.println("""
            switch 模式匹配（Java 21+）
            - 支持类型模式
            - 支持 Record 模式
            - 支持 null 处理
            """);

        // 【类型模式】
        System.out.println("--- 类型模式 ---");
        Object[] values = {42, "hello", 3.14, true, null};

        for (Object value : values) {
            String result = formatValue(value);
            System.out.println("  " + result);
        }

        // 【Record 模式】
        System.out.println("\n--- Record 模式 ---");
        ShapeP[] shapes = {
            new CircleP(5),
            new RectangleP(4, 3),
            new SquareP(4)
        };

        for (ShapeP shape : shapes) {
            System.out.println("  " + describeShape(shape));
        }

        // 【嵌套 Record 模式】
        System.out.println("\n--- 嵌套 Record 模式 ---");
        Object wrapped = new Box(new CircleP(10));
        String desc = switch (wrapped) {
            case Box(CircleP(double r)) -> "盒子里有圆形，半径 " + r;
            case Box(RectangleP(double w, double h)) -> "盒子里有矩形，" + w + "x" + h;
            case Box(SquareP(double s)) -> "盒子里有正方形，边长 " + s;
            default -> "未知内容";
        };
        System.out.println(desc);
    }

    public static String formatValue(Object value) {
        return switch (value) {
            case null -> "null 值";
            case Integer i -> "整数: " + i;
            case String s -> "字符串: \"" + s + "\"";
            case Double d -> "浮点数: " + d;
            case Boolean b -> "布尔: " + b;
            default -> "其他: " + value;
        };
    }

    public static String describeShape(ShapeP shape) {
        return switch (shape) {
            case CircleP(double r) -> String.format("圆形(r=%.1f), 面积=%.2f", r, Math.PI * r * r);
            case RectangleP(double w, double h) -> String.format("矩形(%.1fx%.1f), 面积=%.2f", w, h, w * h);
            case SquareP(double s) -> String.format("正方形(s=%.1f), 面积=%.2f", s, s * s);
        };
    }

    /**
     * ============================================================
     *                    4. 守卫模式
     * ============================================================
     */
    public static void guardedPatterns() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("4. 守卫模式 (when)");
        System.out.println("=".repeat(60));

        System.out.println("""
            守卫模式（Java 21+）
            - 使用 when 关键字添加条件
            - 可以进一步细化模式匹配
            """);

        // 【数值范围守卫】
        System.out.println("--- 数值范围守卫 ---");
        int[] scores = {95, 85, 75, 65, 45};

        for (int score : scores) {
            String grade = switch (score) {
                case Integer s when s >= 90 -> "A";
                case Integer s when s >= 80 -> "B";
                case Integer s when s >= 70 -> "C";
                case Integer s when s >= 60 -> "D";
                default -> "F";
            };
            System.out.println("  " + score + " -> " + grade);
        }

        // 【字符串守卫】
        System.out.println("\n--- 字符串守卫 ---");
        String[] inputs = {"hello", "WORLD", "", null};

        for (String input : inputs) {
            String desc = switch (input) {
                case null -> "空引用";
                case String s when s.isEmpty() -> "空字符串";
                case String s when s.equals(s.toUpperCase()) -> "全大写: " + s;
                case String s when s.equals(s.toLowerCase()) -> "全小写: " + s;
                case String s -> "混合大小写: " + s;
            };
            System.out.println("  " + desc);
        }

        // 【Record 守卫】
        System.out.println("\n--- Record 守卫 ---");
        ShapeP[] shapes = {
            new CircleP(5),
            new CircleP(0.5),
            new RectangleP(10, 2),
            new SquareP(3)
        };

        for (ShapeP shape : shapes) {
            String category = switch (shape) {
                case CircleP(double r) when r > 3 -> "大圆";
                case CircleP(double r) -> "小圆";
                case RectangleP(double w, double h) when w > h * 3 -> "细长矩形";
                case RectangleP r -> "普通矩形";
                case SquareP s -> "正方形";
            };
            System.out.println("  " + shape + " -> " + category);
        }
    }

    /**
     * ============================================================
     *                    5. 密封类型与模式匹配
     * ============================================================
     */
    public static void sealedTypesPattern() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("5. 密封类型与模式匹配");
        System.out.println("=".repeat(60));

        System.out.println("""
            密封类型（Java 17+）与模式匹配完美配合：
            - 编译器知道所有可能的子类型
            - switch 可以确保穷尽性
            - 无需 default 分支
            """);

        // 【密封接口 + switch】
        System.out.println("--- 密封类型的穷尽 switch ---");
        Expr[] expressions = {
            new Num(42),
            new Add(new Num(3), new Num(5)),
            new Mul(new Add(new Num(2), new Num(3)), new Num(4)),
            new Neg(new Num(10))
        };

        for (Expr expr : expressions) {
            int result = evaluate(expr);
            System.out.println("  " + format(expr) + " = " + result);
        }

        // 【类型安全的访问者模式替代】
        System.out.println("\n--- 替代访问者模式 ---");
        System.out.println("""
            传统的访问者模式复杂且难以扩展
            密封类型 + switch 提供了更简洁的替代方案
            """);
    }

    public static int evaluate(Expr expr) {
        return switch (expr) {
            case Num(int value) -> value;
            case Add(Expr left, Expr right) -> evaluate(left) + evaluate(right);
            case Mul(Expr left, Expr right) -> evaluate(left) * evaluate(right);
            case Neg(Expr operand) -> -evaluate(operand);
        };
    }

    public static String format(Expr expr) {
        return switch (expr) {
            case Num(int value) -> String.valueOf(value);
            case Add(Expr left, Expr right) -> "(" + format(left) + " + " + format(right) + ")";
            case Mul(Expr left, Expr right) -> "(" + format(left) + " * " + format(right) + ")";
            case Neg(Expr operand) -> "-" + format(operand);
        };
    }
}

// ============================================================
//                    辅助类型定义
// ============================================================

sealed interface ShapeP permits CircleP, RectangleP, SquareP {}
record CircleP(double radius) implements ShapeP {}
record RectangleP(double width, double height) implements ShapeP {}
record SquareP(double side) implements ShapeP {}

record Box(ShapeP content) {}

// 表达式树（密封类型）
sealed interface Expr permits Num, Add, Mul, Neg {}
record Num(int value) implements Expr {}
record Add(Expr left, Expr right) implements Expr {}
record Mul(Expr left, Expr right) implements Expr {}
record Neg(Expr operand) implements Expr {}
```
