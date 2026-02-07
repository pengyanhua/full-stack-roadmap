# ListDemo

::: info 文件信息
- 📄 原文件：`ListDemo.java`
- 🔤 语言：java
:::

Java List 集合
本文件介绍 Java 中的 List 接口及其实现类。

## 完整代码

```java
import java.util.*;

/**
 * ============================================================
 *                    Java List 集合
 * ============================================================
 * 本文件介绍 Java 中的 List 接口及其实现类。
 * ============================================================
 */
public class ListDemo {

    public static void main(String[] args) {
        arrayListDemo();
        linkedListDemo();
        listOperations();
        listIteration();
        listSorting();
        immutableLists();
    }

    /**
     * ============================================================
     *                    1. ArrayList
     * ============================================================
     */
    public static void arrayListDemo() {
        System.out.println("=".repeat(60));
        System.out.println("1. ArrayList");
        System.out.println("=".repeat(60));

        // 【创建 ArrayList】
        List<String> list1 = new ArrayList<>();           // 推荐
        ArrayList<String> list2 = new ArrayList<>();      // 可以
        List<String> list3 = new ArrayList<>(100);        // 指定初始容量
        List<String> list4 = new ArrayList<>(Arrays.asList("a", "b", "c"));

        System.out.println("创建方式:");
        System.out.println("  new ArrayList<>(): " + list1);
        System.out.println("  带初始值: " + list4);

        // 【添加元素】
        System.out.println("\n--- 添加元素 ---");
        list1.add("Apple");           // 末尾添加
        list1.add("Banana");
        list1.add(1, "Cherry");       // 指定位置插入
        list1.addAll(Arrays.asList("Date", "Elderberry"));  // 添加多个
        System.out.println("添加后: " + list1);

        // 【访问元素】
        System.out.println("\n--- 访问元素 ---");
        System.out.println("get(0): " + list1.get(0));
        System.out.println("get(2): " + list1.get(2));
        System.out.println("size(): " + list1.size());
        System.out.println("isEmpty(): " + list1.isEmpty());

        // 【修改元素】
        System.out.println("\n--- 修改元素 ---");
        list1.set(0, "Apricot");
        System.out.println("set(0, \"Apricot\"): " + list1);

        // 【删除元素】
        System.out.println("\n--- 删除元素 ---");
        list1.remove(0);              // 按索引删除
        System.out.println("remove(0): " + list1);
        list1.remove("Banana");       // 按对象删除
        System.out.println("remove(\"Banana\"): " + list1);

        // 【查找元素】
        System.out.println("\n--- 查找元素 ---");
        System.out.println("contains(\"Cherry\"): " + list1.contains("Cherry"));
        System.out.println("indexOf(\"Cherry\"): " + list1.indexOf("Cherry"));
        System.out.println("lastIndexOf(\"Date\"): " + list1.lastIndexOf("Date"));

        // 【ArrayList 特性】
        System.out.println("\n【ArrayList 特性】");
        System.out.println("  - 基于动态数组实现");
        System.out.println("  - 随机访问快 O(1)");
        System.out.println("  - 插入删除慢 O(n)（需要移动元素）");
        System.out.println("  - 非线程安全");
    }

    /**
     * ============================================================
     *                    2. LinkedList
     * ============================================================
     */
    public static void linkedListDemo() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. LinkedList");
        System.out.println("=".repeat(60));

        // LinkedList 实现了 List 和 Deque 接口
        LinkedList<String> list = new LinkedList<>();

        // 【List 操作】
        list.add("A");
        list.add("B");
        list.add("C");
        System.out.println("List 操作: " + list);

        // 【Deque 操作】双端队列
        System.out.println("\n--- 双端队列操作 ---");
        list.addFirst("First");       // 头部添加
        list.addLast("Last");         // 尾部添加
        System.out.println("addFirst/Last: " + list);

        System.out.println("getFirst(): " + list.getFirst());
        System.out.println("getLast(): " + list.getLast());

        list.removeFirst();
        list.removeLast();
        System.out.println("removeFirst/Last: " + list);

        // 【栈操作】
        System.out.println("\n--- 栈操作 ---");
        LinkedList<Integer> stack = new LinkedList<>();
        stack.push(1);  // 等同于 addFirst
        stack.push(2);
        stack.push(3);
        System.out.println("push 1,2,3: " + stack);
        System.out.println("pop(): " + stack.pop());
        System.out.println("peek(): " + stack.peek());
        System.out.println("栈内容: " + stack);

        // 【队列操作】
        System.out.println("\n--- 队列操作 ---");
        LinkedList<Integer> queue = new LinkedList<>();
        queue.offer(1);  // 入队
        queue.offer(2);
        queue.offer(3);
        System.out.println("offer 1,2,3: " + queue);
        System.out.println("poll(): " + queue.poll());  // 出队
        System.out.println("peek(): " + queue.peek());
        System.out.println("队列内容: " + queue);

        // 【LinkedList 特性】
        System.out.println("\n【LinkedList 特性】");
        System.out.println("  - 基于双向链表实现");
        System.out.println("  - 随机访问慢 O(n)");
        System.out.println("  - 插入删除快 O(1)（已知位置时）");
        System.out.println("  - 可作为栈、队列、双端队列使用");
    }

    /**
     * ============================================================
     *                    3. List 常用操作
     * ============================================================
     */
    public static void listOperations() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("3. List 常用操作");
        System.out.println("=".repeat(60));

        List<Integer> list = new ArrayList<>(Arrays.asList(1, 2, 3, 4, 5));

        // 【子列表】
        System.out.println("--- 子列表 ---");
        List<Integer> subList = list.subList(1, 4);
        System.out.println("原列表: " + list);
        System.out.println("subList(1, 4): " + subList);

        // 【注意】子列表是视图，修改会影响原列表
        subList.set(0, 100);
        System.out.println("修改子列表后原列表: " + list);

        // 【转换为数组】
        System.out.println("\n--- 转换为数组 ---");
        list = new ArrayList<>(Arrays.asList(1, 2, 3, 4, 5));
        Integer[] arr = list.toArray(new Integer[0]);
        System.out.println("toArray: " + Arrays.toString(arr));

        // 【批量操作】
        System.out.println("\n--- 批量操作 ---");
        List<Integer> list2 = new ArrayList<>(Arrays.asList(3, 4, 5, 6, 7));
        System.out.println("list: " + list);
        System.out.println("list2: " + list2);

        // 保留交集
        List<Integer> intersection = new ArrayList<>(list);
        intersection.retainAll(list2);
        System.out.println("retainAll (交集): " + intersection);

        // 删除交集
        List<Integer> difference = new ArrayList<>(list);
        difference.removeAll(list2);
        System.out.println("removeAll (差集): " + difference);

        // 【replaceAll】替换所有元素
        System.out.println("\n--- replaceAll ---");
        List<Integer> numbers = new ArrayList<>(Arrays.asList(1, 2, 3, 4, 5));
        numbers.replaceAll(n -> n * 2);
        System.out.println("replaceAll(n -> n * 2): " + numbers);

        // 【removeIf】条件删除
        System.out.println("\n--- removeIf ---");
        numbers.removeIf(n -> n > 6);
        System.out.println("removeIf(n -> n > 6): " + numbers);
    }

    /**
     * ============================================================
     *                    4. List 遍历
     * ============================================================
     */
    public static void listIteration() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("4. List 遍历");
        System.out.println("=".repeat(60));

        List<String> list = Arrays.asList("A", "B", "C", "D", "E");

        // 【for 循环】
        System.out.println("--- for 循环 ---");
        for (int i = 0; i < list.size(); i++) {
            System.out.print(list.get(i) + " ");
        }
        System.out.println();

        // 【增强 for 循环】
        System.out.println("\n--- 增强 for 循环 ---");
        for (String s : list) {
            System.out.print(s + " ");
        }
        System.out.println();

        // 【Iterator】
        System.out.println("\n--- Iterator ---");
        Iterator<String> it = list.iterator();
        while (it.hasNext()) {
            System.out.print(it.next() + " ");
        }
        System.out.println();

        // 【ListIterator】支持双向遍历
        System.out.println("\n--- ListIterator（反向）---");
        ListIterator<String> lit = list.listIterator(list.size());
        while (lit.hasPrevious()) {
            System.out.print(lit.previous() + " ");
        }
        System.out.println();

        // 【forEach】
        System.out.println("\n--- forEach ---");
        list.forEach(s -> System.out.print(s + " "));
        System.out.println();

        // 【Stream】
        System.out.println("\n--- Stream ---");
        list.stream().forEach(s -> System.out.print(s + " "));
        System.out.println();

        // 【遍历时删除】
        System.out.println("\n--- 遍历时删除（使用 Iterator）---");
        List<Integer> numbers = new ArrayList<>(Arrays.asList(1, 2, 3, 4, 5));
        Iterator<Integer> iter = numbers.iterator();
        while (iter.hasNext()) {
            if (iter.next() % 2 == 0) {
                iter.remove();  // 安全删除
            }
        }
        System.out.println("删除偶数后: " + numbers);
    }

    /**
     * ============================================================
     *                    5. List 排序
     * ============================================================
     */
    public static void listSorting() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("5. List 排序");
        System.out.println("=".repeat(60));

        // 【自然排序】
        System.out.println("--- 自然排序 ---");
        List<Integer> numbers = new ArrayList<>(Arrays.asList(5, 2, 8, 1, 9, 3));
        Collections.sort(numbers);
        System.out.println("升序: " + numbers);

        Collections.sort(numbers, Collections.reverseOrder());
        System.out.println("降序: " + numbers);

        // 【List.sort()】Java 8+
        System.out.println("\n--- List.sort() ---");
        numbers.sort(Comparator.naturalOrder());
        System.out.println("自然顺序: " + numbers);

        numbers.sort(Comparator.reverseOrder());
        System.out.println("反向顺序: " + numbers);

        // 【自定义排序】
        System.out.println("\n--- 自定义排序 ---");
        List<String> words = new ArrayList<>(Arrays.asList("banana", "Apple", "cherry", "date"));

        // 按长度排序
        words.sort(Comparator.comparingInt(String::length));
        System.out.println("按长度: " + words);

        // 忽略大小写排序
        words.sort(String.CASE_INSENSITIVE_ORDER);
        System.out.println("忽略大小写: " + words);

        // 【对象排序】
        System.out.println("\n--- 对象排序 ---");
        List<Person2> people = new ArrayList<>();
        people.add(new Person2("Alice", 30));
        people.add(new Person2("Bob", 25));
        people.add(new Person2("Charlie", 35));

        // 按年龄排序
        people.sort(Comparator.comparingInt(Person2::getAge));
        System.out.println("按年龄: " + people);

        // 按名字排序
        people.sort(Comparator.comparing(Person2::getName));
        System.out.println("按名字: " + people);

        // 多级排序
        people.sort(Comparator
                .comparingInt(Person2::getAge)
                .thenComparing(Person2::getName));
        System.out.println("先年龄后名字: " + people);

        // 【其他工具方法】
        System.out.println("\n--- 其他工具方法 ---");
        List<Integer> list = new ArrayList<>(Arrays.asList(1, 2, 3, 4, 5));
        Collections.shuffle(list);       // 随机打乱
        System.out.println("shuffle: " + list);

        Collections.reverse(list);       // 反转
        System.out.println("reverse: " + list);

        Collections.rotate(list, 2);     // 旋转
        System.out.println("rotate(2): " + list);

        Collections.swap(list, 0, 4);    // 交换
        System.out.println("swap(0, 4): " + list);
    }

    /**
     * ============================================================
     *                    6. 不可变 List
     * ============================================================
     */
    public static void immutableLists() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("6. 不可变 List");
        System.out.println("=".repeat(60));

        // 【Arrays.asList】固定大小
        System.out.println("--- Arrays.asList ---");
        List<String> list1 = Arrays.asList("A", "B", "C");
        System.out.println("Arrays.asList: " + list1);
        list1.set(0, "X");  // 可以修改
        System.out.println("set(0, \"X\"): " + list1);
        // list1.add("D");  // 不能添加！UnsupportedOperationException

        // 【List.of】完全不可变（Java 9+）
        System.out.println("\n--- List.of (Java 9+) ---");
        List<String> list2 = List.of("A", "B", "C");
        System.out.println("List.of: " + list2);
        // list2.set(0, "X");  // 不能修改！
        // list2.add("D");     // 不能添加！

        // 【Collections.unmodifiableList】包装为不可变
        System.out.println("\n--- Collections.unmodifiableList ---");
        List<String> mutable = new ArrayList<>(Arrays.asList("A", "B", "C"));
        List<String> immutable = Collections.unmodifiableList(mutable);
        // immutable.add("D");  // 不能修改！

        // 【注意】底层列表的修改会反映到不可变视图
        mutable.add("D");
        System.out.println("底层修改后: " + immutable);

        // 【List.copyOf】真正的副本（Java 10+）
        System.out.println("\n--- List.copyOf (Java 10+) ---");
        List<String> copy = List.copyOf(mutable);
        mutable.add("E");
        System.out.println("原列表: " + mutable);
        System.out.println("copyOf: " + copy);  // 不受影响
    }
}

// 辅助类
class Person2 {
    private String name;
    private int age;

    public Person2(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public String getName() { return name; }
    public int getAge() { return age; }

    @Override
    public String toString() {
        return name + "(" + age + ")";
    }
}
```
