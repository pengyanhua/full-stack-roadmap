# vec hashmap.rs

::: info 文件信息
- 📄 原文件：`01_vec_hashmap.rs`
- 🔤 语言：rust
:::

## 完整代码

```rust
// ============================================================
//                      集合类型
// ============================================================
// Rust 标准库提供了三种最常用的集合：
// - Vec<T>: 动态数组（最常用）
// - HashMap<K, V>: 哈希映射（键值对）
// - String: UTF-8 字符串（前面已介绍，这里补充集合操作）
//
// 所有集合数据存储在堆上，大小可以在运行时变化

use std::collections::HashMap;

fn main() {
    println!("=== 集合类型 ===");

    // ============================================================
    //                      Vec<T> 动态数组
    // ============================================================
    println!("\n=== Vec<T> ===");

    // ----------------------------------------------------------
    // 1. 创建 Vec
    // ----------------------------------------------------------
    let mut v1: Vec<i32> = Vec::new();       // 空 Vec
    let v2 = vec![1, 2, 3, 4, 5];            // vec! 宏创建
    let v3 = vec![0; 10];                     // 10 个 0
    let v4: Vec<i32> = (1..=5).collect();     // 从迭代器创建
    let v5 = Vec::with_capacity(100);         // 预分配容量

    println!("v1: {:?}", v1);
    println!("v2: {:?}", v2);
    println!("v3: {:?}", v3);
    println!("v4: {:?}", v4);
    println!("v5 长度={}, 容量={}", v5.len(), v5.capacity());

    // ----------------------------------------------------------
    // 2. 增删改查
    // ----------------------------------------------------------
    // 添加元素
    v1.push(10);
    v1.push(20);
    v1.push(30);
    println!("\npush 后: {:?}", v1);

    // 弹出最后一个元素
    let last = v1.pop();  // 返回 Option<T>
    println!("pop: {:?}, 剩余: {:?}", last, v1);

    // 插入
    v1.insert(1, 15);  // 在索引 1 处插入
    println!("insert: {:?}", v1);

    // 删除
    let removed = v1.remove(0);  // 删除索引 0
    println!("remove: {} → {:?}", removed, v1);

    // 保留满足条件的元素
    let mut nums = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    nums.retain(|&x| x % 2 == 0);  // 只保留偶数
    println!("retain 偶数: {:?}", nums);

    // ----------------------------------------------------------
    // 3. 访问元素
    // ----------------------------------------------------------
    let v = vec![10, 20, 30, 40, 50];

    // 方式1: 索引（越界会 panic）
    println!("\n索引访问: {}", v[2]);

    // 方式2: get()（返回 Option，不会 panic）
    match v.get(2) {
        Some(val) => println!("get(2): {}", val),
        None => println!("get(2): 不存在"),
    }
    println!("get(99): {:?}", v.get(99)); // None

    // 【建议】不确定索引是否有效时用 get()

    // 首尾元素
    println!("first: {:?}", v.first());
    println!("last: {:?}", v.last());

    // ----------------------------------------------------------
    // 4. 遍历
    // ----------------------------------------------------------
    println!("\n=== 遍历 ===");

    let v = vec!["苹果", "香蕉", "橙子"];

    // 不可变遍历
    for item in &v {
        print!("{} ", item);
    }
    println!();

    // 可变遍历
    let mut scores = vec![80, 90, 70, 85];
    for score in &mut scores {
        *score += 5;  // 每个加 5 分
    }
    println!("加分后: {:?}", scores);

    // 带索引遍历
    for (i, item) in v.iter().enumerate() {
        println!("  [{}] {}", i, item);
    }

    // ----------------------------------------------------------
    // 5. 迭代器方法（函数式编程）
    // ----------------------------------------------------------
    println!("\n=== 迭代器方法 ===");

    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // map
    let doubled: Vec<i32> = numbers.iter().map(|&x| x * 2).collect();
    println!("map (×2): {:?}", doubled);

    // filter
    let evens: Vec<&i32> = numbers.iter().filter(|&&x| x % 2 == 0).collect();
    println!("filter (偶数): {:?}", evens);

    // fold (reduce)
    let sum: i32 = numbers.iter().fold(0, |acc, &x| acc + x);
    println!("fold (求和): {}", sum);

    // any / all
    println!("any > 5: {}", numbers.iter().any(|&x| x > 5));
    println!("all > 0: {}", numbers.iter().all(|&x| x > 0));

    // find
    let first_even = numbers.iter().find(|&&x| x % 2 == 0);
    println!("find 偶数: {:?}", first_even);

    // position
    let pos = numbers.iter().position(|&x| x == 5);
    println!("position of 5: {:?}", pos);

    // zip
    let names = vec!["Alice", "Bob", "Charlie"];
    let ages = vec![30, 25, 35];
    let pairs: Vec<_> = names.iter().zip(ages.iter()).collect();
    println!("zip: {:?}", pairs);

    // chain
    let a = vec![1, 2];
    let b = vec![3, 4];
    let combined: Vec<&i32> = a.iter().chain(b.iter()).collect();
    println!("chain: {:?}", combined);

    // ----------------------------------------------------------
    // 6. 排序
    // ----------------------------------------------------------
    println!("\n=== 排序 ===");

    let mut v = vec![3, 1, 4, 1, 5, 9, 2, 6];
    v.sort();
    println!("sort: {:?}", v);

    v.sort_by(|a, b| b.cmp(a));  // 降序
    println!("降序: {:?}", v);

    // 浮点数排序（f64 没有实现 Ord，需要 partial_cmp）
    let mut floats = vec![3.14, 1.41, 2.72, 0.58];
    floats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("浮点排序: {:?}", floats);

    // 按某个键排序
    let mut words = vec!["banana", "apple", "cherry", "date"];
    words.sort_by_key(|w| w.len());
    println!("按长度排序: {:?}", words);

    // 去重（需要先排序）
    let mut v = vec![1, 3, 2, 1, 4, 3, 5, 2];
    v.sort();
    v.dedup();
    println!("去重: {:?}", v);

    // ============================================================
    //                      HashMap<K, V>
    // ============================================================
    println!("\n=== HashMap<K, V> ===");

    // ----------------------------------------------------------
    // 1. 创建 HashMap
    // ----------------------------------------------------------
    let mut scores: HashMap<String, i32> = HashMap::new();

    // 从元组数组创建
    let teams = vec![
        ("蓝队".to_string(), 10),
        ("红队".to_string(), 50),
    ];
    let scores2: HashMap<String, i32> = teams.into_iter().collect();
    println!("从 Vec 创建: {:?}", scores2);

    // ----------------------------------------------------------
    // 2. 增删改查
    // ----------------------------------------------------------
    // 插入
    scores.insert("张三".to_string(), 95);
    scores.insert("李四".to_string(), 87);
    scores.insert("王五".to_string(), 92);
    println!("插入后: {:?}", scores);

    // 访问
    let name = "张三".to_string();
    match scores.get(&name) {
        Some(score) => println!("{} 的分数: {}", name, score),
        None => println!("{} 不存在", name),
    }

    // 【技巧】get 返回 Option<&V>

    // 删除
    scores.remove("王五");
    println!("删除后: {:?}", scores);

    // 检查键是否存在
    println!("contains_key 张三: {}", scores.contains_key("张三"));

    // ----------------------------------------------------------
    // 3. 更新策略
    // ----------------------------------------------------------
    println!("\n=== 更新策略 ===");

    let mut map = HashMap::new();

    // 直接覆盖
    map.insert("key", 1);
    map.insert("key", 2);  // 覆盖
    println!("覆盖: {:?}", map);

    // 不存在才插入（entry API）
    map.entry("key").or_insert(999);     // key 已存在，不插入
    map.entry("new_key").or_insert(100); // 新键，插入
    println!("or_insert: {:?}", map);

    // 基于旧值更新
    let text = "hello world hello rust hello world";
    let mut word_count = HashMap::new();
    for word in text.split_whitespace() {
        let count = word_count.entry(word).or_insert(0);
        *count += 1;
    }
    println!("词频统计: {:?}", word_count);

    // or_insert_with（惰性插入）
    let mut cache: HashMap<&str, Vec<i32>> = HashMap::new();
    cache.entry("data").or_insert_with(|| {
        println!("  计算默认值...");
        vec![1, 2, 3]
    });
    println!("cache: {:?}", cache);

    // ----------------------------------------------------------
    // 4. 遍历
    // ----------------------------------------------------------
    println!("\n=== HashMap 遍历 ===");

    let mut capitals = HashMap::new();
    capitals.insert("中国", "北京");
    capitals.insert("日本", "东京");
    capitals.insert("韩国", "首尔");

    // 遍历键值对
    for (country, capital) in &capitals {
        println!("  {} → {}", country, capital);
    }

    // 只遍历键或值
    print!("所有国家: ");
    for country in capitals.keys() {
        print!("{} ", country);
    }
    println!();

    // ----------------------------------------------------------
    // 5. 实用示例
    // ----------------------------------------------------------
    println!("\n=== 实用示例 ===");

    // 分组
    let students = vec![
        ("张三", "数学"),
        ("李四", "语文"),
        ("王五", "数学"),
        ("赵六", "英语"),
        ("钱七", "语文"),
    ];

    let mut groups: HashMap<&str, Vec<&str>> = HashMap::new();
    for (name, subject) in &students {
        groups.entry(subject).or_insert_with(Vec::new).push(name);
    }
    println!("分组: {:#?}", groups);

    // 用 Vec 存储枚举实现异构集合
    #[derive(Debug)]
    enum Value {
        Int(i32),
        Float(f64),
        Text(String),
    }

    let row: Vec<Value> = vec![
        Value::Int(1),
        Value::Float(3.14),
        Value::Text("hello".to_string()),
    ];
    println!("异构集合: {:?}", row);

    // ----------------------------------------------------------
    // 6. 其他集合（简要提及）
    // ----------------------------------------------------------
    println!("\n=== 其他集合 ===");

    // HashSet（无重复元素的集合）
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(1);
    set.insert(2);
    set.insert(2);  // 重复，不会添加
    set.insert(3);
    println!("HashSet: {:?}", set);

    // 集合运算
    let a: HashSet<i32> = [1, 2, 3, 4].iter().cloned().collect();
    let b: HashSet<i32> = [3, 4, 5, 6].iter().cloned().collect();
    println!("并集: {:?}", a.union(&b).collect::<Vec<_>>());
    println!("交集: {:?}", a.intersection(&b).collect::<Vec<_>>());
    println!("差集 A-B: {:?}", a.difference(&b).collect::<Vec<_>>());

    // BTreeMap（有序 Map，按键排序）
    use std::collections::BTreeMap;
    let mut bt = BTreeMap::new();
    bt.insert(3, "三");
    bt.insert(1, "一");
    bt.insert(2, "二");
    println!("BTreeMap（有序）: {:?}", bt);

    // VecDeque（双端队列）
    use std::collections::VecDeque;
    let mut deque = VecDeque::new();
    deque.push_back(1);
    deque.push_back(2);
    deque.push_front(0);
    println!("VecDeque: {:?}", deque);
    println!("pop_front: {:?}", deque.pop_front());

    println!("\n=== 集合类型结束 ===");
}
```
