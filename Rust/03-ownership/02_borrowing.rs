// ============================================================
//                      借用与引用（Borrowing & References）
// ============================================================
// 借用是 Rust 中不转移所有权就能使用数据的机制
// 引用（&T）就像指针，但保证有效（不会悬垂）
//
// 借用规则（Rust 编译器强制执行）：
// 1. 在任意时刻，要么只能有一个可变引用（&mut T），
//    要么可以有多个不可变引用（&T）
// 2. 引用必须始终有效（不能悬垂）
//
// 【记忆口诀】"可以有很多读者，或者一个写者，但不能同时"

fn main() {
    println!("=== 借用与引用 ===");

    // ----------------------------------------------------------
    // 1. 不可变引用（&T）
    // ----------------------------------------------------------
    // 使用 & 创建引用，不获取所有权
    // 可以同时有多个不可变引用

    let s1 = String::from("hello");
    let len = calculate_length(&s1);  // 借用 s1，不移动
    println!("'{}' 的长度是 {}", s1, len);  // s1 仍然有效

    // 多个不可变引用 — OK
    let r1 = &s1;
    let r2 = &s1;
    println!("多个不可变引用: {}, {}", r1, r2);

    // ----------------------------------------------------------
    // 2. 可变引用（&mut T）
    // ----------------------------------------------------------
    // 使用 &mut 创建可变引用，可以修改借用的数据
    // 【限制】同一时刻只能有一个可变引用

    let mut s2 = String::from("hello");
    change(&mut s2);
    println!("修改后: {}", s2);

    // 同一作用域中只能有一个可变引用
    let mut s = String::from("hello");
    let r1 = &mut s;
    // let r2 = &mut s;  // 错误！不能同时有两个可变引用
    r1.push_str(" world");
    println!("可变引用: {}", r1);

    // 【原因】防止数据竞争（data race）
    // 数据竞争发生在以下三个条件同时满足时：
    // 1. 两个或更多指针同时访问同一数据
    // 2. 至少一个指针用于写入
    // 3. 没有同步机制

    // ----------------------------------------------------------
    // 3. 不可变引用和可变引用不能共存
    // ----------------------------------------------------------
    let mut s = String::from("hello");

    let r1 = &s;     // OK：不可变引用
    let r2 = &s;     // OK：可以有多个不可变引用
    println!("不可变引用: {}, {}", r1, r2);
    // r1 和 r2 在这之后不再使用（NLL: Non-Lexical Lifetimes）

    let r3 = &mut s;  // OK：因为 r1, r2 已经不再使用
    r3.push_str(" world");
    println!("可变引用: {}", r3);

    // 【NLL (Non-Lexical Lifetimes)】
    // Rust 2018 之后，引用的生命周期在最后一次使用时结束（不是作用域结束）
    // 这让借用规则更灵活，上面的代码因此可以编译

    // ----------------------------------------------------------
    // 4. 悬垂引用（Dangling Reference）
    // ----------------------------------------------------------
    // Rust 编译器保证引用永远不会悬垂
    // 如果你有一个引用，编译器确保数据在引用之前不会离开作用域

    // fn dangle() -> &String {  // 错误！
    //     let s = String::from("hello");
    //     &s  // s 在函数结束时被释放，引用将悬垂
    // }

    // 正确做法：返回所有权
    fn no_dangle() -> String {
        let s = String::from("hello");
        s  // 移动所有权给调用者
    }
    let s = no_dangle();
    println!("无悬垂: {}", s);

    // ----------------------------------------------------------
    // 5. 切片引用（Slices）
    // ----------------------------------------------------------
    // 切片是对集合的部分引用，没有所有权
    // 【类型】&str 是字符串切片，&[T] 是数组切片

    // 字符串切片
    let s = String::from("hello world");
    let hello = &s[0..5];   // 或 &s[..5]
    let world = &s[6..11];  // 或 &s[6..]
    let whole = &s[..];     // 整个字符串
    println!("切片: '{}' '{}' '{}'", hello, world, whole);

    // 【注意】字符串切片的索引是字节位置，不是字符位置
    // 对于 UTF-8 中文字符（3字节），必须在字符边界切割
    let chinese = String::from("你好世界");
    let ni_hao = &chinese[0..6];  // "你好"（每个中文3字节）
    println!("中文切片: {}", ni_hao);
    // let bad = &chinese[0..1];  // panic！不在字符边界上

    // 数组切片
    let arr = [1, 2, 3, 4, 5];
    let slice = &arr[1..3];  // [2, 3]
    println!("数组切片: {:?}", slice);

    // 【实用】first_word 示例
    let sentence = String::from("hello world");
    let word = first_word(&sentence);
    println!("第一个单词: {}", word);

    // 切片确保了引用的有效性
    // let mut s = String::from("hello world");
    // let word = first_word(&s);  // 不可变借用
    // s.clear();  // 错误！不能修改，因为 word 还在使用不可变引用
    // println!("{}", word);

    // ----------------------------------------------------------
    // 6. 借用的实际应用模式
    // ----------------------------------------------------------
    println!("\n=== 实际应用 ===");

    // 模式1: 只读访问用不可变引用
    let data = vec![1, 2, 3, 4, 5];
    let sum = sum_vec(&data);
    let max = max_vec(&data);
    println!("data={:?}, sum={}, max={}", data, sum, max);

    // 模式2: 需要修改用可变引用
    let mut scores = vec![85, 92, 78, 96, 88];
    add_bonus(&mut scores, 5);
    println!("加分后: {:?}", scores);

    // 模式3: 方法中的借用
    let text = String::from("Hello, Rust! Welcome to Rust programming.");
    let stats = TextStats::new(&text);
    println!("字符数={}, 单词数={}, 行数={}", stats.chars, stats.words, stats.lines);

    println!("\n=== 借用与引用结束 ===");
}

// ----------------------------------------------------------
// 不可变借用参数
// ----------------------------------------------------------
fn calculate_length(s: &String) -> usize {
    s.len()
} // s 只是引用，离开作用域不会释放数据

// ----------------------------------------------------------
// 可变借用参数
// ----------------------------------------------------------
fn change(s: &mut String) {
    s.push_str(", world");
}

// ----------------------------------------------------------
// 返回切片引用
// ----------------------------------------------------------
fn first_word(s: &str) -> &str {
    // 【技巧】参数类型用 &str 而不是 &String
    // 这样 String 和 &str 都能传入（更灵活）
    let bytes = s.as_bytes();
    for (i, &byte) in bytes.iter().enumerate() {
        if byte == b' ' {
            return &s[..i];
        }
    }
    s  // 没有空格，返回整个字符串
}

// ----------------------------------------------------------
// 实际应用函数
// ----------------------------------------------------------
fn sum_vec(v: &Vec<i32>) -> i32 {
    v.iter().sum()
}

fn max_vec(v: &Vec<i32>) -> i32 {
    *v.iter().max().unwrap()
}

fn add_bonus(scores: &mut Vec<i32>, bonus: i32) {
    for score in scores.iter_mut() {
        *score += bonus;
    }
}

// ----------------------------------------------------------
// 结构体中的借用模式
// ----------------------------------------------------------
struct TextStats {
    chars: usize,
    words: usize,
    lines: usize,
}

impl TextStats {
    fn new(text: &str) -> TextStats {
        TextStats {
            chars: text.chars().count(),
            words: text.split_whitespace().count(),
            lines: text.lines().count(),
        }
    }
}
