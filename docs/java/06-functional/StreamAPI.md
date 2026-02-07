# StreamAPI

::: info 文件信息
- 📄 原文件：`StreamAPI.java`
- 🔤 语言：java
:::

Java Stream API
本文件介绍 Java 中的 Stream API 函数式编程。

## 完整代码

```java
import java.util.*;
import java.util.stream.*;
import java.util.function.*;

/**
 * ============================================================
 *                    Java Stream API
 * ============================================================
 * 本文件介绍 Java 中的 Stream API 函数式编程。
 * ============================================================
 */
public class StreamAPI {

    public static void main(String[] args) {
        streamBasics();
        intermediateOperations();
        terminalOperations();
        collectorsDemo();
        parallelStreams();
        practicalExamples();
    }

    /**
     * ============================================================
     *                    1. Stream 基础
     * ============================================================
     */
    public static void streamBasics() {
        System.out.println("=".repeat(60));
        System.out.println("1. Stream 基础");
        System.out.println("=".repeat(60));

        System.out.println("""
            Stream 特点：
            - 不存储数据，只是数据的视图
            - 惰性求值，只有终端操作时才执行
            - 只能使用一次，使用后即关闭
            - 支持并行处理
            """);

        // 【创建 Stream】
        System.out.println("--- 创建 Stream ---");

        // 从集合创建
        List<Integer> list = List.of(1, 2, 3, 4, 5);
        Stream<Integer> s1 = list.stream();
        System.out.println("从 List 创建 Stream");

        // 从数组创建
        String[] array = {"a", "b", "c"};
        Stream<String> s2 = Arrays.stream(array);
        System.out.println("从数组创建 Stream");

        // 使用 Stream.of
        Stream<Integer> s3 = Stream.of(1, 2, 3);
        System.out.println("Stream.of 创建");

        // 使用 Stream.generate（无限流）
        Stream<Double> randoms = Stream.generate(Math::random).limit(5);
        System.out.println("generate: " + randoms.toList());

        // 使用 Stream.iterate
        Stream<Integer> naturals = Stream.iterate(1, n -> n + 1).limit(5);
        System.out.println("iterate: " + naturals.toList());

        // 带终止条件的 iterate（Java 9+）
        Stream<Integer> limited = Stream.iterate(1, n -> n < 10, n -> n + 2);
        System.out.println("iterate with predicate: " + limited.toList());

        // IntStream, LongStream, DoubleStream
        System.out.println("\n--- 基本类型 Stream ---");
        IntStream intStream = IntStream.range(1, 6);
        System.out.println("IntStream.range(1, 6): " + intStream.boxed().toList());

        IntStream closed = IntStream.rangeClosed(1, 5);
        System.out.println("IntStream.rangeClosed(1, 5): " + closed.boxed().toList());

        // 空 Stream
        Stream<String> empty = Stream.empty();
        System.out.println("empty stream count: " + empty.count());
    }

    /**
     * ============================================================
     *                    2. 中间操作
     * ============================================================
     */
    public static void intermediateOperations() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. 中间操作");
        System.out.println("=".repeat(60));

        List<String> words = List.of("apple", "banana", "cherry", "date", "elderberry");
        List<Integer> numbers = List.of(3, 1, 4, 1, 5, 9, 2, 6, 5, 3);

        // 【filter】过滤
        System.out.println("--- filter ---");
        List<String> longWords = words.stream()
            .filter(w -> w.length() > 5)
            .toList();
        System.out.println("长度 > 5: " + longWords);

        // 【map】转换
        System.out.println("\n--- map ---");
        List<Integer> lengths = words.stream()
            .map(String::length)
            .toList();
        System.out.println("单词长度: " + lengths);

        List<String> upperWords = words.stream()
            .map(String::toUpperCase)
            .toList();
        System.out.println("大写: " + upperWords);

        // 【flatMap】扁平化
        System.out.println("\n--- flatMap ---");
        List<List<Integer>> nested = List.of(
            List.of(1, 2, 3),
            List.of(4, 5),
            List.of(6, 7, 8, 9)
        );
        List<Integer> flat = nested.stream()
            .flatMap(Collection::stream)
            .toList();
        System.out.println("扁平化: " + flat);

        // 字符串拆分
        List<String> chars = words.stream()
            .flatMap(s -> Arrays.stream(s.split("")))
            .distinct()
            .sorted()
            .toList();
        System.out.println("所有字符: " + chars);

        // 【distinct】去重
        System.out.println("\n--- distinct ---");
        List<Integer> unique = numbers.stream()
            .distinct()
            .toList();
        System.out.println("去重: " + unique);

        // 【sorted】排序
        System.out.println("\n--- sorted ---");
        List<Integer> sorted = numbers.stream()
            .sorted()
            .toList();
        System.out.println("自然排序: " + sorted);

        List<Integer> descending = numbers.stream()
            .sorted(Comparator.reverseOrder())
            .toList();
        System.out.println("降序: " + descending);

        List<String> byLength = words.stream()
            .sorted(Comparator.comparingInt(String::length))
            .toList();
        System.out.println("按长度: " + byLength);

        // 【limit / skip】截取
        System.out.println("\n--- limit / skip ---");
        List<Integer> first3 = numbers.stream()
            .limit(3)
            .toList();
        System.out.println("前 3 个: " + first3);

        List<Integer> skip3 = numbers.stream()
            .skip(3)
            .toList();
        System.out.println("跳过 3 个: " + skip3);

        // 分页
        int page = 2, pageSize = 3;
        List<Integer> pageData = numbers.stream()
            .skip((long) (page - 1) * pageSize)
            .limit(pageSize)
            .toList();
        System.out.println("第 2 页: " + pageData);

        // 【peek】调试
        System.out.println("\n--- peek ---");
        List<String> result = words.stream()
            .filter(w -> w.length() > 4)
            .peek(w -> System.out.println("过滤后: " + w))
            .map(String::toUpperCase)
            .peek(w -> System.out.println("转换后: " + w))
            .limit(2)
            .toList();
        System.out.println("最终结果: " + result);

        // 【takeWhile / dropWhile】（Java 9+）
        System.out.println("\n--- takeWhile / dropWhile ---");
        List<Integer> sorted2 = List.of(1, 2, 3, 4, 5, 6, 7);

        List<Integer> taken = sorted2.stream()
            .takeWhile(n -> n < 5)
            .toList();
        System.out.println("takeWhile(n < 5): " + taken);

        List<Integer> dropped = sorted2.stream()
            .dropWhile(n -> n < 5)
            .toList();
        System.out.println("dropWhile(n < 5): " + dropped);
    }

    /**
     * ============================================================
     *                    3. 终端操作
     * ============================================================
     */
    public static void terminalOperations() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("3. 终端操作");
        System.out.println("=".repeat(60));

        List<Integer> numbers = List.of(3, 1, 4, 1, 5, 9, 2, 6);
        List<String> words = List.of("apple", "banana", "cherry");

        // 【forEach】遍历
        System.out.println("--- forEach ---");
        numbers.stream()
            .limit(5)
            .forEach(n -> System.out.print(n + " "));
        System.out.println();

        // 【count】计数
        System.out.println("\n--- count ---");
        long count = numbers.stream()
            .filter(n -> n > 3)
            .count();
        System.out.println("大于 3 的数量: " + count);

        // 【reduce】归约
        System.out.println("\n--- reduce ---");
        int sum = numbers.stream()
            .reduce(0, Integer::sum);
        System.out.println("求和: " + sum);

        Optional<Integer> max = numbers.stream()
            .reduce(Integer::max);
        System.out.println("最大值: " + max.orElse(0));

        String concat = words.stream()
            .reduce("", (a, b) -> a + b);
        System.out.println("连接: " + concat);

        // 【collect】收集
        System.out.println("\n--- collect ---");
        List<Integer> list = numbers.stream()
            .filter(n -> n > 2)
            .collect(Collectors.toList());
        System.out.println("toList: " + list);

        Set<Integer> set = numbers.stream()
            .collect(Collectors.toSet());
        System.out.println("toSet: " + set);

        // 【min / max】
        System.out.println("\n--- min / max ---");
        Optional<Integer> minOpt = numbers.stream().min(Integer::compareTo);
        Optional<Integer> maxOpt = numbers.stream().max(Integer::compareTo);
        System.out.println("min: " + minOpt.orElse(0));
        System.out.println("max: " + maxOpt.orElse(0));

        // 【findFirst / findAny】
        System.out.println("\n--- findFirst / findAny ---");
        Optional<Integer> first = numbers.stream()
            .filter(n -> n > 5)
            .findFirst();
        System.out.println("findFirst(> 5): " + first.orElse(-1));

        // 【anyMatch / allMatch / noneMatch】
        System.out.println("\n--- match 操作 ---");
        boolean anyGreater5 = numbers.stream().anyMatch(n -> n > 5);
        boolean allPositive = numbers.stream().allMatch(n -> n > 0);
        boolean noneNegative = numbers.stream().noneMatch(n -> n < 0);

        System.out.println("anyMatch(> 5): " + anyGreater5);
        System.out.println("allMatch(> 0): " + allPositive);
        System.out.println("noneMatch(< 0): " + noneNegative);

        // 【toArray】
        System.out.println("\n--- toArray ---");
        Integer[] arr = numbers.stream().toArray(Integer[]::new);
        System.out.println("toArray: " + Arrays.toString(arr));

        // 【IntStream 专用操作】
        System.out.println("\n--- IntStream 操作 ---");
        IntStream intStream = numbers.stream().mapToInt(Integer::intValue);
        IntSummaryStatistics stats = numbers.stream()
            .mapToInt(Integer::intValue)
            .summaryStatistics();

        System.out.println("sum: " + stats.getSum());
        System.out.println("average: " + stats.getAverage());
        System.out.println("min: " + stats.getMin());
        System.out.println("max: " + stats.getMax());
        System.out.println("count: " + stats.getCount());
    }

    /**
     * ============================================================
     *                    4. Collectors 详解
     * ============================================================
     */
    public static void collectorsDemo() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("4. Collectors 详解");
        System.out.println("=".repeat(60));

        List<Person2> people = List.of(
            new Person2("Alice", 25, "Engineering"),
            new Person2("Bob", 30, "Engineering"),
            new Person2("Charlie", 35, "Marketing"),
            new Person2("Diana", 28, "Marketing"),
            new Person2("Eve", 32, "Engineering")
        );

        // 【toList / toSet / toMap】
        System.out.println("--- toList / toSet / toMap ---");
        List<String> names = people.stream()
            .map(Person2::name)
            .collect(Collectors.toList());
        System.out.println("names: " + names);

        Map<String, Integer> nameToAge = people.stream()
            .collect(Collectors.toMap(Person2::name, Person2::age));
        System.out.println("nameToAge: " + nameToAge);

        // 【joining】连接字符串
        System.out.println("\n--- joining ---");
        String joined = people.stream()
            .map(Person2::name)
            .collect(Collectors.joining(", ", "[", "]"));
        System.out.println("joining: " + joined);

        // 【groupingBy】分组
        System.out.println("\n--- groupingBy ---");
        Map<String, List<Person2>> byDept = people.stream()
            .collect(Collectors.groupingBy(Person2::department));
        byDept.forEach((dept, ps) ->
            System.out.println(dept + ": " + ps.stream().map(Person2::name).toList()));

        // 分组后计数
        Map<String, Long> countByDept = people.stream()
            .collect(Collectors.groupingBy(Person2::department, Collectors.counting()));
        System.out.println("countByDept: " + countByDept);

        // 分组后求平均
        Map<String, Double> avgAgeByDept = people.stream()
            .collect(Collectors.groupingBy(
                Person2::department,
                Collectors.averagingInt(Person2::age)
            ));
        System.out.println("avgAgeByDept: " + avgAgeByDept);

        // 【partitioningBy】二分
        System.out.println("\n--- partitioningBy ---");
        Map<Boolean, List<Person2>> partition = people.stream()
            .collect(Collectors.partitioningBy(p -> p.age() >= 30));
        System.out.println(">=30: " + partition.get(true).stream().map(Person2::name).toList());
        System.out.println("<30: " + partition.get(false).stream().map(Person2::name).toList());

        // 【summarizing】统计
        System.out.println("\n--- summarizing ---");
        IntSummaryStatistics ageStats = people.stream()
            .collect(Collectors.summarizingInt(Person2::age));
        System.out.println("Age stats: " + ageStats);

        // 【reducing】
        System.out.println("\n--- reducing ---");
        Optional<Person2> oldest = people.stream()
            .collect(Collectors.maxBy(Comparator.comparingInt(Person2::age)));
        System.out.println("oldest: " + oldest.map(Person2::name).orElse("none"));

        // 【collectingAndThen】后处理
        System.out.println("\n--- collectingAndThen ---");
        List<String> unmodifiableNames = people.stream()
            .map(Person2::name)
            .collect(Collectors.collectingAndThen(
                Collectors.toList(),
                Collections::unmodifiableList
            ));
        System.out.println("unmodifiable: " + unmodifiableNames);
    }

    /**
     * ============================================================
     *                    5. 并行流
     * ============================================================
     */
    public static void parallelStreams() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("5. 并行流");
        System.out.println("=".repeat(60));

        System.out.println("""
            并行流注意事项：
            - 适合 CPU 密集型任务
            - 数据量要足够大
            - 避免共享可变状态
            - 某些操作（如 findFirst）在并行流中代价更高
            """);

        List<Integer> numbers = IntStream.rangeClosed(1, 100).boxed().toList();

        // 【创建并行流】
        System.out.println("--- 创建并行流 ---");

        // 方式1：parallel()
        long sum1 = numbers.stream()
            .parallel()
            .mapToLong(Integer::longValue)
            .sum();
        System.out.println("parallel() sum: " + sum1);

        // 方式2：parallelStream()
        long sum2 = numbers.parallelStream()
            .mapToLong(Integer::longValue)
            .sum();
        System.out.println("parallelStream() sum: " + sum2);

        // 【检查是否并行】
        System.out.println("\n--- 检查并行状态 ---");
        Stream<Integer> s1 = numbers.stream();
        System.out.println("stream isParallel: " + s1.isParallel());

        Stream<Integer> s2 = numbers.parallelStream();
        System.out.println("parallelStream isParallel: " + s2.isParallel());

        // 【顺序与并行切换】
        System.out.println("\n--- sequential() 切回顺序 ---");
        List<Integer> result = numbers.parallelStream()
            .filter(n -> n % 2 == 0)
            .sequential()  // 切回顺序流
            .sorted()
            .limit(10)
            .toList();
        System.out.println("前 10 个偶数: " + result);

        // 【forEachOrdered 保持顺序】
        System.out.println("\n--- forEachOrdered ---");
        System.out.print("并行但保持顺序: ");
        IntStream.range(1, 6)
            .parallel()
            .forEachOrdered(n -> System.out.print(n + " "));
        System.out.println();
    }

    /**
     * ============================================================
     *                    6. 实际应用示例
     * ============================================================
     */
    public static void practicalExamples() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("6. 实际应用示例");
        System.out.println("=".repeat(60));

        // 【词频统计】
        System.out.println("--- 词频统计 ---");
        String text = "apple banana apple cherry banana apple date";
        Map<String, Long> wordCount = Arrays.stream(text.split(" "))
            .collect(Collectors.groupingBy(
                Function.identity(),
                Collectors.counting()
            ));
        System.out.println("词频: " + wordCount);

        // 【找出重复元素】
        System.out.println("\n--- 找出重复元素 ---");
        List<Integer> nums = List.of(1, 2, 3, 2, 4, 3, 5);
        Set<Integer> seen = new HashSet<>();
        Set<Integer> duplicates = nums.stream()
            .filter(n -> !seen.add(n))
            .collect(Collectors.toSet());
        System.out.println("重复元素: " + duplicates);

        // 【扁平化嵌套结构】
        System.out.println("\n--- 扁平化订单商品 ---");
        record Order(String id, List<String> items) {}
        List<Order> orders = List.of(
            new Order("O1", List.of("Apple", "Banana")),
            new Order("O2", List.of("Cherry", "Date")),
            new Order("O3", List.of("Apple", "Elderberry"))
        );

        List<String> allItems = orders.stream()
            .flatMap(o -> o.items().stream())
            .distinct()
            .sorted()
            .toList();
        System.out.println("所有商品: " + allItems);

        // 【Top N】
        System.out.println("\n--- Top N ---");
        List<Integer> scores = List.of(85, 92, 78, 95, 88, 76, 91);
        List<Integer> top3 = scores.stream()
            .sorted(Comparator.reverseOrder())
            .limit(3)
            .toList();
        System.out.println("Top 3 分数: " + top3);

        // 【按条件分组计数】
        System.out.println("\n--- 按年龄段分组 ---");
        List<Person2> people = List.of(
            new Person2("Alice", 22, "IT"),
            new Person2("Bob", 35, "IT"),
            new Person2("Charlie", 28, "HR"),
            new Person2("Diana", 45, "HR")
        );

        Map<String, Long> ageGroups = people.stream()
            .collect(Collectors.groupingBy(
                p -> {
                    int age = p.age();
                    if (age < 25) return "20-24";
                    else if (age < 35) return "25-34";
                    else if (age < 45) return "35-44";
                    else return "45+";
                },
                Collectors.counting()
            ));
        System.out.println("年龄段分布: " + ageGroups);
    }
}

/**
 * Person 记录类
 */
record Person2(String name, int age, String department) {}
```
