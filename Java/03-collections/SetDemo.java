import java.util.*;

/**
 * ============================================================
 *                    Java Set 集合
 * ============================================================
 * 本文件介绍 Java 中的 Set 接口及其实现类。
 * ============================================================
 */
public class SetDemo {

    public static void main(String[] args) {
        hashSetDemo();
        linkedHashSetDemo();
        treeSetDemo();
        setOperations();
        enumSetDemo();
    }

    /**
     * ============================================================
     *                    1. HashSet
     * ============================================================
     */
    public static void hashSetDemo() {
        System.out.println("=".repeat(60));
        System.out.println("1. HashSet");
        System.out.println("=".repeat(60));

        // 【创建 HashSet】
        Set<String> set = new HashSet<>();

        // 【添加元素】
        System.out.println("--- 添加元素 ---");
        set.add("apple");
        set.add("banana");
        set.add("cherry");
        System.out.println("添加后: " + set);

        // 重复元素不会被添加
        boolean added = set.add("apple");
        System.out.println("再次添加 apple: " + added);  // false
        System.out.println("Set: " + set);  // 仍然只有一个 apple

        // 【批量添加】
        set.addAll(List.of("date", "elderberry"));
        System.out.println("addAll: " + set);

        // 【检查元素】
        System.out.println("\n--- 检查元素 ---");
        System.out.println("contains(\"apple\"): " + set.contains("apple"));
        System.out.println("contains(\"fig\"): " + set.contains("fig"));
        System.out.println("size(): " + set.size());
        System.out.println("isEmpty(): " + set.isEmpty());

        // 【删除元素】
        System.out.println("\n--- 删除元素 ---");
        boolean removed = set.remove("banana");
        System.out.println("remove(\"banana\"): " + removed);
        System.out.println("删除后: " + set);

        // 【HashSet 特性】
        System.out.println("\n【HashSet 特性】");
        System.out.println("  - 基于 HashMap 实现");
        System.out.println("  - 不允许重复元素");
        System.out.println("  - 允许 null 元素");
        System.out.println("  - 无序");
        System.out.println("  - O(1) 的增删查（平均）");
        System.out.println("  - 非线程安全");

        // 【null 元素】
        System.out.println("\n--- null 元素 ---");
        set.add(null);
        System.out.println("添加 null 后: " + set);
    }

    /**
     * ============================================================
     *                    2. LinkedHashSet
     * ============================================================
     */
    public static void linkedHashSetDemo() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. LinkedHashSet");
        System.out.println("=".repeat(60));

        // 【保持插入顺序】
        System.out.println("--- 保持插入顺序 ---");
        Set<String> linkedSet = new LinkedHashSet<>();
        linkedSet.add("c");
        linkedSet.add("a");
        linkedSet.add("b");
        System.out.println("LinkedHashSet: " + linkedSet);  // [c, a, b]

        // 对比 HashSet
        Set<String> hashSet = new HashSet<>();
        hashSet.add("c");
        hashSet.add("a");
        hashSet.add("b");
        System.out.println("HashSet: " + hashSet);  // 顺序不确定

        // 【遍历顺序一致】
        System.out.println("\n--- 遍历顺序 ---");
        System.out.print("遍历 LinkedHashSet: ");
        for (String s : linkedSet) {
            System.out.print(s + " ");
        }
        System.out.println();

        // 【LinkedHashSet 特性】
        System.out.println("\n【LinkedHashSet 特性】");
        System.out.println("  - 基于 LinkedHashMap 实现");
        System.out.println("  - 保持插入顺序");
        System.out.println("  - 性能略低于 HashSet");
        System.out.println("  - 适合需要保持顺序的场景");
    }

    /**
     * ============================================================
     *                    3. TreeSet
     * ============================================================
     */
    public static void treeSetDemo() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("3. TreeSet");
        System.out.println("=".repeat(60));

        // 【自然排序】
        System.out.println("--- 自然排序 ---");
        Set<Integer> treeSet = new TreeSet<>();
        treeSet.add(5);
        treeSet.add(1);
        treeSet.add(3);
        treeSet.add(2);
        treeSet.add(4);
        System.out.println("TreeSet: " + treeSet);  // [1, 2, 3, 4, 5]

        // 【自定义排序】
        System.out.println("\n--- 自定义排序（降序）---");
        Set<Integer> descSet = new TreeSet<>(Comparator.reverseOrder());
        descSet.addAll(List.of(5, 1, 3, 2, 4));
        System.out.println("降序: " + descSet);  // [5, 4, 3, 2, 1]

        // 【字符串排序】
        System.out.println("\n--- 字符串排序 ---");
        Set<String> strSet = new TreeSet<>();
        strSet.addAll(List.of("cherry", "apple", "banana"));
        System.out.println("字符串自然排序: " + strSet);

        // 按长度排序
        Set<String> byLength = new TreeSet<>(Comparator.comparingInt(String::length));
        byLength.addAll(List.of("cherry", "apple", "banana", "fig"));
        System.out.println("按长度排序: " + byLength);  // 注意：长度相同的会被去重

        // 【导航方法】
        System.out.println("\n--- 导航方法 ---");
        TreeSet<Integer> navSet = new TreeSet<>(List.of(1, 3, 5, 7, 9));
        System.out.println("TreeSet: " + navSet);

        System.out.println("first(): " + navSet.first());
        System.out.println("last(): " + navSet.last());
        System.out.println("lower(5): " + navSet.lower(5));     // < 5 的最大元素
        System.out.println("higher(5): " + navSet.higher(5));   // > 5 的最小元素
        System.out.println("floor(4): " + navSet.floor(4));     // <= 4 的最大元素
        System.out.println("ceiling(4): " + navSet.ceiling(4)); // >= 4 的最小元素

        // 【子集】
        System.out.println("\n--- 子集 ---");
        System.out.println("subSet(3, 7): " + navSet.subSet(3, 7));    // [3, 5)
        System.out.println("headSet(5): " + navSet.headSet(5));        // < 5
        System.out.println("tailSet(5): " + navSet.tailSet(5));        // >= 5

        // 【弹出元素】
        System.out.println("\n--- 弹出元素 ---");
        TreeSet<Integer> popSet = new TreeSet<>(List.of(1, 2, 3, 4, 5));
        System.out.println("pollFirst(): " + popSet.pollFirst());  // 弹出最小
        System.out.println("pollLast(): " + popSet.pollLast());    // 弹出最大
        System.out.println("剩余: " + popSet);

        // 【TreeSet 特性】
        System.out.println("\n【TreeSet 特性】");
        System.out.println("  - 基于 TreeMap（红黑树）实现");
        System.out.println("  - 元素有序（自然排序或自定义排序）");
        System.out.println("  - 不允许 null（使用自然排序时）");
        System.out.println("  - O(log n) 的增删查");
    }

    /**
     * ============================================================
     *                    4. Set 集合运算
     * ============================================================
     */
    public static void setOperations() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("4. Set 集合运算");
        System.out.println("=".repeat(60));

        Set<Integer> set1 = new HashSet<>(List.of(1, 2, 3, 4, 5));
        Set<Integer> set2 = new HashSet<>(List.of(4, 5, 6, 7, 8));

        System.out.println("set1: " + set1);
        System.out.println("set2: " + set2);

        // 【并集】
        System.out.println("\n--- 并集 (Union) ---");
        Set<Integer> union = new HashSet<>(set1);
        union.addAll(set2);
        System.out.println("set1 ∪ set2 = " + union);

        // 【交集】
        System.out.println("\n--- 交集 (Intersection) ---");
        Set<Integer> intersection = new HashSet<>(set1);
        intersection.retainAll(set2);
        System.out.println("set1 ∩ set2 = " + intersection);

        // 【差集】
        System.out.println("\n--- 差集 (Difference) ---");
        Set<Integer> difference = new HashSet<>(set1);
        difference.removeAll(set2);
        System.out.println("set1 - set2 = " + difference);

        // 【对称差集】
        System.out.println("\n--- 对称差集 (Symmetric Difference) ---");
        Set<Integer> symDiff = new HashSet<>(set1);
        symDiff.addAll(set2);
        Set<Integer> temp = new HashSet<>(set1);
        temp.retainAll(set2);
        symDiff.removeAll(temp);
        System.out.println("set1 △ set2 = " + symDiff);

        // 【子集检查】
        System.out.println("\n--- 子集检查 ---");
        Set<Integer> small = new HashSet<>(List.of(2, 3));
        System.out.println("small: " + small);
        System.out.println("set1.containsAll(small): " + set1.containsAll(small));

        // 【不可变 Set】
        System.out.println("\n--- 不可变 Set ---");
        Set<String> immutable = Set.of("a", "b", "c");
        System.out.println("Set.of(): " + immutable);
        // immutable.add("d");  // UnsupportedOperationException

        Set<String> copyOf = Set.copyOf(List.of("x", "y", "z", "x"));  // 去重
        System.out.println("Set.copyOf(): " + copyOf);

        // 【遍历 Set】
        System.out.println("\n--- 遍历 Set ---");
        Set<String> fruits = Set.of("apple", "banana", "cherry");

        // for-each
        System.out.print("for-each: ");
        for (String fruit : fruits) {
            System.out.print(fruit + " ");
        }
        System.out.println();

        // forEach
        System.out.print("forEach: ");
        fruits.forEach(f -> System.out.print(f + " "));
        System.out.println();

        // Stream
        System.out.print("Stream: ");
        fruits.stream()
              .filter(f -> f.length() > 5)
              .forEach(f -> System.out.print(f + " "));
        System.out.println();
    }

    /**
     * ============================================================
     *                    5. EnumSet
     * ============================================================
     */
    public static void enumSetDemo() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("5. EnumSet");
        System.out.println("=".repeat(60));

        // 【创建 EnumSet】
        System.out.println("--- 创建 EnumSet ---");

        // 包含所有枚举值
        Set<Day> allDays = EnumSet.allOf(Day.class);
        System.out.println("allOf: " + allDays);

        // 空集
        Set<Day> noDays = EnumSet.noneOf(Day.class);
        System.out.println("noneOf: " + noDays);

        // 指定元素
        Set<Day> weekend = EnumSet.of(Day.SATURDAY, Day.SUNDAY);
        System.out.println("of: " + weekend);

        // 范围
        Set<Day> weekdays = EnumSet.range(Day.MONDAY, Day.FRIDAY);
        System.out.println("range: " + weekdays);

        // 补集
        Set<Day> notWeekend = EnumSet.complementOf(EnumSet.of(Day.SATURDAY, Day.SUNDAY));
        System.out.println("complementOf: " + notWeekend);

        // 【EnumSet 操作】
        System.out.println("\n--- EnumSet 操作 ---");
        EnumSet<Day> myDays = EnumSet.noneOf(Day.class);
        myDays.add(Day.MONDAY);
        myDays.add(Day.WEDNESDAY);
        myDays.add(Day.FRIDAY);
        System.out.println("工作日: " + myDays);

        // 检查
        System.out.println("包含周一: " + myDays.contains(Day.MONDAY));
        System.out.println("是工作日子集: " + weekdays.containsAll(myDays));

        // 【EnumSet 特性】
        System.out.println("\n【EnumSet 特性】");
        System.out.println("  - 专门用于枚举类型的 Set");
        System.out.println("  - 内部使用位向量实现，极其高效");
        System.out.println("  - 元素按枚举声明顺序排列");
        System.out.println("  - 不允许 null");
        System.out.println("  - 比 HashSet 更快、更节省内存");
    }
}

/**
 * 星期枚举 - 用于 EnumSet 演示
 */
enum Day {
    MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY
}
