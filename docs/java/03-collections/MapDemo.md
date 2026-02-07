# MapDemo

::: info 文件信息
- 📄 原文件：`MapDemo.java`
- 🔤 语言：java
:::

Java Map 集合
本文件介绍 Java 中的 Map 接口及其实现类。

## 完整代码

```java
import java.util.*;

/**
 * ============================================================
 *                    Java Map 集合
 * ============================================================
 * 本文件介绍 Java 中的 Map 接口及其实现类。
 * ============================================================
 */
public class MapDemo {

    public static void main(String[] args) {
        hashMapDemo();
        linkedHashMapDemo();
        treeMapDemo();
        mapOperations();
        mapIteration();
        computeMethods();
    }

    /**
     * ============================================================
     *                    1. HashMap
     * ============================================================
     */
    public static void hashMapDemo() {
        System.out.println("=".repeat(60));
        System.out.println("1. HashMap");
        System.out.println("=".repeat(60));

        // 【创建 HashMap】
        Map<String, Integer> map = new HashMap<>();

        // 【添加元素】
        System.out.println("--- 添加元素 ---");
        map.put("apple", 100);
        map.put("banana", 200);
        map.put("cherry", 300);
        System.out.println("put: " + map);

        // put 返回旧值
        Integer oldValue = map.put("apple", 150);
        System.out.println("更新 apple，旧值: " + oldValue);
        System.out.println("更新后: " + map);

        // putIfAbsent：键不存在时才添加
        map.putIfAbsent("apple", 999);   // 不会更新
        map.putIfAbsent("date", 400);    // 会添加
        System.out.println("putIfAbsent: " + map);

        // putAll：批量添加
        Map<String, Integer> more = Map.of("elderberry", 500, "fig", 600);
        map.putAll(more);
        System.out.println("putAll: " + map);

        // 【访问元素】
        System.out.println("\n--- 访问元素 ---");
        System.out.println("get(\"apple\"): " + map.get("apple"));
        System.out.println("get(\"grape\"): " + map.get("grape"));  // null
        System.out.println("getOrDefault(\"grape\", 0): " + map.getOrDefault("grape", 0));

        // 【检查键/值】
        System.out.println("\n--- 检查键/值 ---");
        System.out.println("containsKey(\"apple\"): " + map.containsKey("apple"));
        System.out.println("containsValue(100): " + map.containsValue(100));
        System.out.println("size(): " + map.size());
        System.out.println("isEmpty(): " + map.isEmpty());

        // 【删除元素】
        System.out.println("\n--- 删除元素 ---");
        Integer removed = map.remove("fig");
        System.out.println("remove(\"fig\"): " + removed);

        // 条件删除：只有值匹配时才删除
        boolean wasRemoved = map.remove("elderberry", 500);
        System.out.println("remove(\"elderberry\", 500): " + wasRemoved);
        System.out.println("删除后: " + map);

        // 【HashMap 特性】
        System.out.println("\n【HashMap 特性】");
        System.out.println("  - 基于哈希表实现");
        System.out.println("  - 允许 null 键和 null 值");
        System.out.println("  - 无序");
        System.out.println("  - O(1) 的增删改查（平均）");
        System.out.println("  - 非线程安全");
    }

    /**
     * ============================================================
     *                    2. LinkedHashMap
     * ============================================================
     */
    public static void linkedHashMapDemo() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. LinkedHashMap");
        System.out.println("=".repeat(60));

        // 【保持插入顺序】
        System.out.println("--- 插入顺序 ---");
        Map<String, Integer> map = new LinkedHashMap<>();
        map.put("c", 3);
        map.put("a", 1);
        map.put("b", 2);
        System.out.println("LinkedHashMap: " + map);  // {c=3, a=1, b=2}

        // 对比 HashMap
        Map<String, Integer> hashMap = new HashMap<>();
        hashMap.put("c", 3);
        hashMap.put("a", 1);
        hashMap.put("b", 2);
        System.out.println("HashMap: " + hashMap);  // 顺序不确定

        // 【访问顺序模式】
        System.out.println("\n--- 访问顺序模式 ---");
        Map<String, Integer> accessOrder = new LinkedHashMap<>(16, 0.75f, true);
        accessOrder.put("a", 1);
        accessOrder.put("b", 2);
        accessOrder.put("c", 3);
        System.out.println("初始: " + accessOrder);

        accessOrder.get("a");  // 访问 a
        System.out.println("访问 a 后: " + accessOrder);  // a 移到末尾

        // 【实现 LRU 缓存】
        System.out.println("\n--- LRU 缓存 ---");
        Map<String, String> lruCache = new LinkedHashMap<>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, String> eldest) {
                return size() > 3;  // 超过 3 个元素时删除最旧的
            }
        };

        lruCache.put("1", "one");
        lruCache.put("2", "two");
        lruCache.put("3", "three");
        System.out.println("添加 3 个: " + lruCache);

        lruCache.put("4", "four");  // 触发删除
        System.out.println("添加第 4 个: " + lruCache);
    }

    /**
     * ============================================================
     *                    3. TreeMap
     * ============================================================
     */
    public static void treeMapDemo() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("3. TreeMap");
        System.out.println("=".repeat(60));

        // 【自然排序】
        System.out.println("--- 自然排序 ---");
        Map<String, Integer> map = new TreeMap<>();
        map.put("cherry", 3);
        map.put("apple", 1);
        map.put("banana", 2);
        System.out.println("TreeMap: " + map);  // 按键排序

        // 【自定义排序】
        System.out.println("\n--- 自定义排序（降序）---");
        Map<String, Integer> descMap = new TreeMap<>(Comparator.reverseOrder());
        descMap.put("cherry", 3);
        descMap.put("apple", 1);
        descMap.put("banana", 2);
        System.out.println("降序: " + descMap);

        // 【导航方法】
        System.out.println("\n--- 导航方法 ---");
        TreeMap<Integer, String> treeMap = new TreeMap<>();
        treeMap.put(1, "one");
        treeMap.put(3, "three");
        treeMap.put(5, "five");
        treeMap.put(7, "seven");
        treeMap.put(9, "nine");
        System.out.println("TreeMap: " + treeMap);

        System.out.println("firstKey(): " + treeMap.firstKey());
        System.out.println("lastKey(): " + treeMap.lastKey());
        System.out.println("lowerKey(5): " + treeMap.lowerKey(5));   // < 5 的最大键
        System.out.println("higherKey(5): " + treeMap.higherKey(5)); // > 5 的最小键
        System.out.println("floorKey(4): " + treeMap.floorKey(4));   // <= 4 的最大键
        System.out.println("ceilingKey(4): " + treeMap.ceilingKey(4)); // >= 4 的最小键

        // 【子映射】
        System.out.println("\n--- 子映射 ---");
        System.out.println("subMap(3, 7): " + treeMap.subMap(3, 7));  // [3, 7)
        System.out.println("headMap(5): " + treeMap.headMap(5));      // < 5
        System.out.println("tailMap(5): " + treeMap.tailMap(5));      // >= 5

        // 【TreeMap 特性】
        System.out.println("\n【TreeMap 特性】");
        System.out.println("  - 基于红黑树实现");
        System.out.println("  - 按键排序");
        System.out.println("  - 不允许 null 键（允许 null 值）");
        System.out.println("  - O(log n) 的增删改查");
    }

    /**
     * ============================================================
     *                    4. Map 常用操作
     * ============================================================
     */
    public static void mapOperations() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("4. Map 常用操作");
        System.out.println("=".repeat(60));

        Map<String, Integer> map = new HashMap<>();
        map.put("a", 1);
        map.put("b", 2);
        map.put("c", 3);

        // 【获取所有键/值/条目】
        System.out.println("--- 键/值/条目 ---");
        System.out.println("keySet(): " + map.keySet());
        System.out.println("values(): " + map.values());
        System.out.println("entrySet(): " + map.entrySet());

        // 【replace】
        System.out.println("\n--- replace ---");
        map.replace("a", 100);
        System.out.println("replace(\"a\", 100): " + map);

        // 条件替换
        map.replace("b", 2, 200);
        System.out.println("replace(\"b\", 2, 200): " + map);

        // 【replaceAll】
        System.out.println("\n--- replaceAll ---");
        map.replaceAll((k, v) -> v * 10);
        System.out.println("replaceAll(v * 10): " + map);

        // 【merge】合并值
        System.out.println("\n--- merge ---");
        Map<String, Integer> scores = new HashMap<>();
        scores.put("Alice", 80);
        scores.put("Bob", 70);

        // 如果键存在，使用函数合并；不存在则直接放入
        scores.merge("Alice", 20, Integer::sum);  // 80 + 20 = 100
        scores.merge("Charlie", 90, Integer::sum);  // 新增
        System.out.println("merge: " + scores);

        // 【统计词频】
        System.out.println("\n--- 统计词频 ---");
        String text = "apple banana apple cherry banana apple";
        Map<String, Integer> wordCount = new HashMap<>();
        for (String word : text.split(" ")) {
            wordCount.merge(word, 1, Integer::sum);
        }
        System.out.println("词频: " + wordCount);
    }

    /**
     * ============================================================
     *                    5. Map 遍历
     * ============================================================
     */
    public static void mapIteration() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("5. Map 遍历");
        System.out.println("=".repeat(60));

        Map<String, Integer> map = Map.of("a", 1, "b", 2, "c", 3);

        // 【遍历键】
        System.out.println("--- 遍历键 ---");
        for (String key : map.keySet()) {
            System.out.println("  " + key + " -> " + map.get(key));
        }

        // 【遍历条目】推荐
        System.out.println("\n--- 遍历条目（推荐）---");
        for (Map.Entry<String, Integer> entry : map.entrySet()) {
            System.out.println("  " + entry.getKey() + " -> " + entry.getValue());
        }

        // 【forEach】
        System.out.println("\n--- forEach ---");
        map.forEach((k, v) -> System.out.println("  " + k + " -> " + v));

        // 【Stream】
        System.out.println("\n--- Stream ---");
        map.entrySet().stream()
           .filter(e -> e.getValue() > 1)
           .forEach(e -> System.out.println("  " + e.getKey() + " -> " + e.getValue()));
    }

    /**
     * ============================================================
     *                    6. compute 方法
     * ============================================================
     */
    public static void computeMethods() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("6. compute 方法");
        System.out.println("=".repeat(60));

        Map<String, Integer> map = new HashMap<>();
        map.put("a", 1);
        map.put("b", 2);

        // 【compute】计算新值
        System.out.println("--- compute ---");
        map.compute("a", (k, v) -> v == null ? 1 : v + 10);
        map.compute("c", (k, v) -> v == null ? 1 : v + 10);  // 新增
        System.out.println("compute: " + map);

        // 【computeIfPresent】键存在时计算
        System.out.println("\n--- computeIfPresent ---");
        map.computeIfPresent("a", (k, v) -> v * 2);
        map.computeIfPresent("x", (k, v) -> v * 2);  // 不存在，不执行
        System.out.println("computeIfPresent: " + map);

        // 【computeIfAbsent】键不存在时计算
        System.out.println("\n--- computeIfAbsent ---");
        map.computeIfAbsent("d", k -> 100);  // 新增
        map.computeIfAbsent("a", k -> 999);  // 存在，不执行
        System.out.println("computeIfAbsent: " + map);

        // 【实际应用：分组】
        System.out.println("\n--- 分组应用 ---");
        List<String> words = Arrays.asList("apple", "banana", "apricot", "blueberry", "avocado");
        Map<Character, List<String>> groups = new HashMap<>();

        for (String word : words) {
            char firstChar = word.charAt(0);
            groups.computeIfAbsent(firstChar, k -> new ArrayList<>()).add(word);
        }
        System.out.println("按首字母分组: " + groups);
    }
}
```
