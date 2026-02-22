# traits.rs

::: info 文件信息
- 📄 原文件：`03_traits.rs`
- 🔤 语言：rust
:::

## 完整代码

```rust
// ============================================================
//                      Trait（特征）
// ============================================================
// Trait 定义共享行为，类似其他语言的接口（interface）
// 但 Rust 的 trait 更强大：可以有默认实现、泛型约束、关联类型
//
// 【类比】
//   - Go 的 interface（但 Rust trait 需要显式实现）
//   - Java 的 interface（但可以有默认方法和关联类型）
//   - Haskell 的 typeclass

// ----------------------------------------------------------
// 1. 定义 Trait
// ----------------------------------------------------------
trait Summary {
    // 必须实现的方法（没有函数体）
    fn summarize(&self) -> String;

    // 有默认实现的方法（可以覆盖）
    fn preview(&self) -> String {
        format!("(预览: {}...)", &self.summarize()[..20.min(self.summarize().len())])
    }
}

// ----------------------------------------------------------
// 2. 为类型实现 Trait
// ----------------------------------------------------------
struct Article {
    title: String,
    author: String,
    content: String,
}

impl Summary for Article {
    fn summarize(&self) -> String {
        format!("{}, 作者: {}", self.title, self.author)
    }
    // preview 使用默认实现
}

struct Tweet {
    username: String,
    content: String,
}

impl Summary for Tweet {
    fn summarize(&self) -> String {
        format!("@{}: {}", self.username, self.content)
    }

    // 覆盖默认实现
    fn preview(&self) -> String {
        format!("推文来自 @{}", self.username)
    }
}

// ----------------------------------------------------------
// 3. 常用的标准库 Trait
// ----------------------------------------------------------

// Display: 控制 {} 格式化输出
use std::fmt;

struct Point {
    x: f64,
    y: f64,
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

fn main() {
    println!("=== Trait ===");

    // ----------------------------------------------------------
    // 基本使用
    // ----------------------------------------------------------
    let article = Article {
        title: String::from("Rust 学习指南"),
        author: String::from("张三"),
        content: String::from("Rust 是一门系统编程语言..."),
    };

    let tweet = Tweet {
        username: String::from("rustlang"),
        content: String::from("Rust 1.75 发布了！"),
    };

    println!("文章: {}", article.summarize());
    println!("文章预览: {}", article.preview());
    println!("推文: {}", tweet.summarize());
    println!("推文预览: {}", tweet.preview());

    // ----------------------------------------------------------
    // Trait 作为参数（静态分发 vs 动态分发）
    // ----------------------------------------------------------
    println!("\n=== Trait 作为参数 ===");

    // 方式1: impl Trait（静态分发，编译时确定类型，有内联优化）
    // 【推荐】大多数情况用这种
    notify_static(&article);
    notify_static(&tweet);

    // 方式2: &dyn Trait（动态分发，运行时通过 vtable 查找）
    // 【适用】需要在运行时存储不同类型时
    notify_dynamic(&article);

    // 动态分发的典型用法：存储不同类型的集合
    let items: Vec<Box<dyn Summary>> = vec![
        Box::new(Article {
            title: String::from("标题1"),
            author: String::from("作者1"),
            content: String::from("内容1"),
        }),
        Box::new(Tweet {
            username: String::from("user1"),
            content: String::from("推文内容"),
        }),
    ];

    println!("\n动态集合:");
    for item in &items {
        println!("  {}", item.summarize());
    }

    // ----------------------------------------------------------
    // Trait Bound（约束）
    // ----------------------------------------------------------
    println!("\n=== Trait Bound ===");

    // 多重约束
    let p = Point { x: 3.0, y: 4.0 };
    print_info(&p);

    // where 子句（复杂约束时更清晰）
    let a = String::from("hello");
    let b = String::from("world");
    println!("较长的: {}", longer_display(&a, &b));

    // ----------------------------------------------------------
    // 返回 impl Trait
    // ----------------------------------------------------------
    let item = create_summarizable();
    println!("\n返回 impl Trait: {}", item.summarize());

    // ----------------------------------------------------------
    // Trait 继承
    // ----------------------------------------------------------
    println!("\n=== Trait 继承 ===");

    let user = WebUser {
        id: 1,
        name: String::from("张三"),
        email: String::from("zhangsan@example.com"),
    };

    // Printable 继承了 Display，所以可以用 {} 格式化
    print_item(&user);
    println!("用户: {}", user); // Display
    user.print();               // Printable

    // ----------------------------------------------------------
    // 关联类型
    // ----------------------------------------------------------
    println!("\n=== 关联类型 ===");

    let mut counter = Counter::new(5);
    print!("计数器: ");
    while let Some(val) = counter.next() {
        print!("{} ", val);
    }
    println!();

    // ----------------------------------------------------------
    // 常用标准库 Trait 演示
    // ----------------------------------------------------------
    println!("\n=== 常用 Trait ===");

    // Clone + Copy
    let p1 = Point { x: 1.0, y: 2.0 };
    let _p2 = p1;  // Point 没有实现 Copy，所以这是移动
    // println!("{}", p1);  // 错误！p1 已移动

    // PartialEq + Eq
    let a = Temperature(36.6);
    let b = Temperature(36.6);
    let c = Temperature(37.5);
    println!("{} == {} ? {}", a.0, b.0, a == b);
    println!("{} < {} ? {}", a.0, c.0, a < c);

    // Default
    let config = Config::default();
    println!("默认配置: host={}, port={}, debug={}",
             config.host, config.port, config.debug);

    // From/Into
    let greeting: Greeting = "你好".into();
    println!("From/Into: {}", greeting.0);

    let greeting2 = Greeting::from("世界");
    println!("From: {}", greeting2.0);

    println!("\n=== Trait 结束 ===");
}

// ----------------------------------------------------------
// impl Trait 参数（静态分发）
// ----------------------------------------------------------
fn notify_static(item: &impl Summary) {
    println!("[静态] 通知: {}", item.summarize());
}

// ----------------------------------------------------------
// dyn Trait 参数（动态分发）
// ----------------------------------------------------------
fn notify_dynamic(item: &dyn Summary) {
    println!("[动态] 通知: {}", item.summarize());
}

// ----------------------------------------------------------
// 多重 Trait Bound
// ----------------------------------------------------------
fn print_info(item: &(impl fmt::Display + fmt::Debug)) {
    println!("Display: {}", item);
    println!("Debug: {:?}", item);
}

// 实现 Debug for Point
impl fmt::Debug for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Point{{ x: {}, y: {} }}", self.x, self.y)
    }
}

// ----------------------------------------------------------
// where 子句
// ----------------------------------------------------------
fn longer_display<T>(a: &T, b: &T) -> &T
where
    T: fmt::Display + PartialOrd,
{
    if a >= b { a } else { b }
}

// ----------------------------------------------------------
// 返回 impl Trait
// ----------------------------------------------------------
fn create_summarizable() -> impl Summary {
    Tweet {
        username: String::from("bot"),
        content: String::from("自动生成的内容"),
    }
}

// ----------------------------------------------------------
// Trait 继承
// ----------------------------------------------------------
trait Printable: fmt::Display {
    fn print(&self) {
        println!("打印: {}", self);
    }
}

struct WebUser {
    id: u64,
    name: String,
    email: String,
}

impl fmt::Display for WebUser {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}(#{})", self.name, self.id)
    }
}

impl Printable for WebUser {}  // 使用 print 的默认实现

fn print_item(item: &impl Printable) {
    item.print();
}

// ----------------------------------------------------------
// 关联类型（Iterator trait）
// ----------------------------------------------------------
struct Counter {
    count: u32,
    max: u32,
}

impl Counter {
    fn new(max: u32) -> Counter {
        Counter { count: 0, max }
    }
}

impl Iterator for Counter {
    type Item = u32;  // 关联类型

    fn next(&mut self) -> Option<Self::Item> {
        if self.count < self.max {
            self.count += 1;
            Some(self.count)
        } else {
            None
        }
    }
}

// ----------------------------------------------------------
// 常用标准库 Trait 实现
// ----------------------------------------------------------

// PartialEq + PartialOrd
#[derive(Debug, PartialEq, PartialOrd)]
struct Temperature(f64);

// Default
struct Config {
    host: String,
    port: u16,
    debug: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            host: String::from("localhost"),
            port: 8080,
            debug: false,
        }
    }
}

// From/Into
struct Greeting(String);

impl From<&str> for Greeting {
    fn from(s: &str) -> Self {
        Greeting(s.to_string())
    }
}
```
