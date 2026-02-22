// ============================================================
//                      生命周期（Lifetimes）
// ============================================================
// 生命周期是 Rust 确保引用有效的机制
// 大多数情况下编译器可以自动推断（生命周期省略规则）
// 只有当编译器无法推断时才需要手动标注
//
// 【核心概念】生命周期标注不改变引用的实际生命周期，
//             它只是告诉编译器多个引用之间的关系

fn main() {
    println!("=== 生命周期 ===");

    // ----------------------------------------------------------
    // 1. 为什么需要生命周期
    // ----------------------------------------------------------
    // 当函数返回引用时，编译器需要知道返回的引用有效多久
    // 否则可能出现悬垂引用

    let string1 = String::from("长字符串");
    let result;
    {
        let string2 = String::from("短");
        result = longest(string1.as_str(), string2.as_str());
        println!("更长的: {}", result);
    }
    // 如果 result 引用了 string2，这里就会悬垂
    // 生命周期标注帮助编译器检查这种情况

    // ----------------------------------------------------------
    // 2. 生命周期标注语法
    // ----------------------------------------------------------
    // 使用 'a, 'b 等命名（以单引号开头）
    // 'a 读作"生命周期 a"
    // 标注在 & 后面：&'a str, &'a mut String

    let s1 = String::from("hello");
    let s2 = String::from("hi");
    let longer = longest(&s1, &s2);
    println!("较长: {}", longer);

    // ----------------------------------------------------------
    // 3. 生命周期省略规则（Elision Rules）
    // ----------------------------------------------------------
    // 编译器自动应用三条规则，能推断出就不需要手动标注：
    //
    // 规则1: 每个引用参数获得独立的生命周期
    //   fn foo(x: &str, y: &str) → fn foo<'a, 'b>(x: &'a str, y: &'b str)
    //
    // 规则2: 如果只有一个输入生命周期，输出生命周期等于它
    //   fn foo(x: &str) -> &str → fn foo<'a>(x: &'a str) -> &'a str
    //
    // 规则3: 如果有 &self 或 &mut self，输出生命周期等于 self
    //   fn method(&self, x: &str) -> &str → 输出生命周期 = self

    // 自动推断的例子（不需要手动标注）：
    let s = String::from("hello world");
    let first = first_word(&s);  // 编译器知道返回值生命周期 = 参数
    println!("第一个词: {}", first);

    // ----------------------------------------------------------
    // 4. 结构体中的生命周期
    // ----------------------------------------------------------
    // 如果结构体持有引用，必须标注生命周期
    // 含义：结构体实例不能比其引用的数据活得更久

    let novel = String::from("在很久很久以前. 一个...");
    let first_sentence;
    {
        let i = novel.find('.').unwrap_or(novel.len());
        first_sentence = ImportantExcerpt {
            part: &novel[..i],
        };
    }
    println!("摘录: {}", first_sentence.part);

    // 结构体方法中的生命周期（通常自动推断）
    println!("通告: {}", first_sentence.announce_and_return("重要！"));

    // ----------------------------------------------------------
    // 5. 静态生命周期（'static）
    // ----------------------------------------------------------
    // 'static 表示引用在整个程序运行期间都有效
    // 【常见】字符串字面量都是 'static
    // 【警告】不要随意使用 'static 来"解决"生命周期问题

    let s: &'static str = "我是静态生命周期";
    println!("{}", s);

    // 字符串字面量存储在程序的二进制文件中，所以是 'static
    // const 常量引用也是 'static

    // ----------------------------------------------------------
    // 6. 生命周期约束
    // ----------------------------------------------------------
    // 'a: 'b 表示 'a 至少和 'b 一样长

    let s1 = String::from("long");
    let result;
    {
        let s2 = String::from("hi");
        result = longest_with_announcement(&s1, &s2, "比较中...");
        println!("结果: {}", result);
    }

    // ----------------------------------------------------------
    // 7. 常见生命周期模式
    // ----------------------------------------------------------
    println!("\n=== 常见模式 ===");

    // 模式1: 返回输入的引用
    let data = vec![1, 2, 3, 4, 5];
    let first = first_element(&data);
    println!("第一个元素: {}", first);

    // 模式2: 多个生命周期参数
    let s1 = "hello";
    let s2 = "world";
    let pair = StringPair::new(s1, s2);
    println!("StringPair: {} + {}", pair.first, pair.second);

    // 模式3: 生命周期 + 泛型 + trait bound
    let items = vec![
        String::from("banana"),
        String::from("apple"),
        String::from("cherry"),
    ];
    let longest_item = longest_item(&items);
    println!("最长项: {}", longest_item);

    println!("\n=== 生命周期结束 ===");
}

// ----------------------------------------------------------
// 需要手动标注生命周期的函数
// ----------------------------------------------------------
// 'a 表示：返回值的生命周期 = 两个参数中较短的那个
// 编译器无法自动推断，因为有两个输入引用
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

// ----------------------------------------------------------
// 编译器可以自动推断的函数（不需要标注）
// ----------------------------------------------------------
fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();
    for (i, &byte) in bytes.iter().enumerate() {
        if byte == b' ' {
            return &s[..i];
        }
    }
    s
}

// ----------------------------------------------------------
// 结构体中的生命周期
// ----------------------------------------------------------
// 含义：ImportantExcerpt 实例不能比 part 引用的数据活得更久
struct ImportantExcerpt<'a> {
    part: &'a str,
}

impl<'a> ImportantExcerpt<'a> {
    // 规则3 自动推断：返回值生命周期 = &self
    fn announce_and_return(&self, announcement: &str) -> &str {
        println!("请注意: {}", announcement);
        self.part
    }
}

// ----------------------------------------------------------
// 生命周期 + 泛型 + trait bound
// ----------------------------------------------------------
fn longest_with_announcement<'a, T: std::fmt::Display>(
    x: &'a str,
    y: &'a str,
    ann: T,
) -> &'a str {
    println!("公告: {}", ann);
    if x.len() > y.len() { x } else { y }
}

// ----------------------------------------------------------
// 返回集合元素的引用
// ----------------------------------------------------------
fn first_element(v: &Vec<i32>) -> &i32 {
    &v[0]
}

// ----------------------------------------------------------
// 多个生命周期参数
// ----------------------------------------------------------
struct StringPair<'a, 'b> {
    first: &'a str,
    second: &'b str,
}

impl<'a, 'b> StringPair<'a, 'b> {
    fn new(first: &'a str, second: &'b str) -> StringPair<'a, 'b> {
        StringPair { first, second }
    }
}

// ----------------------------------------------------------
// 泛型 + 生命周期
// ----------------------------------------------------------
fn longest_item<'a>(items: &'a Vec<String>) -> &'a str {
    let mut longest = &items[0] as &str;
    for item in items {
        if item.len() > longest.len() {
            longest = item;
        }
    }
    longest
}
