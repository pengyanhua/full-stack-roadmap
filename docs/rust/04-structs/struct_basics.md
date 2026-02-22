# struct basics.rs

::: info 文件信息
- 📄 原文件：`01_struct_basics.rs`
- 🔤 语言：rust
:::

## 完整代码

```rust
// ============================================================
//                      结构体（Structs）
// ============================================================
// 结构体是自定义数据类型，将相关数据组合在一起
// Rust 有三种结构体：命名字段结构体、元组结构体、单元结构体
// 【命名规范】结构体名使用 PascalCase，字段名使用 snake_case

// ----------------------------------------------------------
// 1. 定义结构体
// ----------------------------------------------------------
#[derive(Debug)]  // 使结构体可以用 {:?} 打印
struct User {
    username: String,
    email: String,
    active: bool,
    sign_in_count: u64,
}

// 元组结构体（字段没有名字）
// 【适用场景】新类型模式（Newtype Pattern）、简单包装
#[derive(Debug)]
struct Color(u8, u8, u8);

#[derive(Debug)]
struct Point(f64, f64, f64);

// 单元结构体（没有任何字段）
// 【适用场景】实现 trait 但不需要存储数据
struct AlwaysEqual;

fn main() {
    println!("=== 结构体 ===");

    // ----------------------------------------------------------
    // 2. 创建实例
    // ----------------------------------------------------------
    let user1 = User {
        username: String::from("张三"),
        email: String::from("zhangsan@example.com"),
        active: true,
        sign_in_count: 1,
    };
    println!("用户: {} ({})", user1.username, user1.email);

    // 可变实例（整个实例可变，不能只让部分字段可变）
    let mut user2 = User {
        username: String::from("李四"),
        email: String::from("lisi@example.com"),
        active: true,
        sign_in_count: 0,
    };
    user2.sign_in_count += 1;
    println!("登录次数: {}", user2.sign_in_count);

    // 【技巧】字段初始化简写（Field Init Shorthand）
    // 当变量名和字段名相同时，可以省略
    let username = String::from("王五");
    let email = String::from("wangwu@example.com");
    let user3 = User {
        username,  // 等同于 username: username
        email,     // 等同于 email: email
        active: true,
        sign_in_count: 0,
    };
    println!("简写: {:?}", user3.username);

    // ----------------------------------------------------------
    // 3. 结构体更新语法
    // ----------------------------------------------------------
    // 用 ..other_instance 从已有实例复制剩余字段
    // 【注意】如果复制了 String 类型的字段，原实例的该字段会被移动

    let user4 = User {
        email: String::from("user4@example.com"),
        ..user3  // 从 user3 复制其他字段
    };
    println!("更新语法: {} ({})", user4.username, user4.email);
    // println!("{}", user3.username);  // 错误！username 已被移动到 user4
    println!("user3.active 仍可用: {}", user3.active);  // bool 是 Copy 的

    // ----------------------------------------------------------
    // 4. 元组结构体
    // ----------------------------------------------------------
    let red = Color(255, 0, 0);
    let origin = Point(0.0, 0.0, 0.0);
    println!("颜色: ({}, {}, {})", red.0, red.1, red.2);
    println!("原点: ({}, {}, {})", origin.0, origin.1, origin.2);

    // 【重要】即使字段类型相同，不同的元组结构体也是不同的类型
    // let p: Point = Color(1, 2, 3);  // 错误！Color != Point

    // ----------------------------------------------------------
    // 5. Debug 输出
    // ----------------------------------------------------------
    println!("\n=== Debug 输出 ===");
    println!("Debug: {:?}", red);
    println!("Pretty: {:#?}", user4);  // 美化格式

    // dbg! 宏（打印到 stderr，并返回值的所有权）
    let scale = 2;
    let debug_result = dbg!(scale * 10);  // 输出到 stderr: [文件:行号] scale * 10 = 20
    println!("dbg! 返回值: {}", debug_result);

    // ----------------------------------------------------------
    // 6. 方法（impl 块）
    // ----------------------------------------------------------
    println!("\n=== 方法 ===");

    let rect = Rectangle::new(30.0, 50.0);
    println!("矩形: {:?}", rect);
    println!("面积: {}", rect.area());
    println!("周长: {}", rect.perimeter());
    println!("是正方形: {}", rect.is_square());

    let rect2 = Rectangle::new(10.0, 40.0);
    println!("rect 能容纳 rect2: {}", rect.can_hold(&rect2));

    // 关联函数调用
    let square = Rectangle::square(25.0);
    println!("正方形: {:?}", square);

    // ----------------------------------------------------------
    // 7. 多个 impl 块
    // ----------------------------------------------------------
    // 一个结构体可以有多个 impl 块
    // 【适用场景】按功能分组方法、条件编译、泛型实现
    println!("显示: {}", rect.display_info());

    // ----------------------------------------------------------
    // 8. 构建者模式（Builder Pattern）
    // ----------------------------------------------------------
    println!("\n=== 构建者模式 ===");

    let server = ServerConfig::builder()
        .host("localhost")
        .port(8080)
        .max_connections(100)
        .build();
    println!("服务器: {}:{} (最大连接: {})",
             server.host, server.port, server.max_connections);

    println!("\n=== 结构体结束 ===");
}

// ----------------------------------------------------------
// 结构体 + impl 方法示例
// ----------------------------------------------------------
#[derive(Debug)]
struct Rectangle {
    width: f64,
    height: f64,
}

impl Rectangle {
    // 关联函数（构造器，无 self）
    fn new(width: f64, height: f64) -> Rectangle {
        Rectangle { width, height }
    }

    fn square(size: f64) -> Rectangle {
        Rectangle {
            width: size,
            height: size,
        }
    }

    // 方法（&self 不可变借用）
    fn area(&self) -> f64 {
        self.width * self.height
    }

    fn perimeter(&self) -> f64 {
        2.0 * (self.width + self.height)
    }

    fn is_square(&self) -> bool {
        (self.width - self.height).abs() < f64::EPSILON
    }

    // 方法（参数中借用其他实例）
    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.height > other.height
    }
}

// 多个 impl 块
impl Rectangle {
    fn display_info(&self) -> String {
        format!("矩形 {}x{} (面积={})", self.width, self.height, self.area())
    }
}

// ----------------------------------------------------------
// 构建者模式
// ----------------------------------------------------------
#[derive(Debug)]
struct ServerConfig {
    host: String,
    port: u16,
    max_connections: u32,
}

struct ServerConfigBuilder {
    host: String,
    port: u16,
    max_connections: u32,
}

impl ServerConfig {
    fn builder() -> ServerConfigBuilder {
        ServerConfigBuilder {
            host: String::from("0.0.0.0"),
            port: 3000,
            max_connections: 10,
        }
    }
}

impl ServerConfigBuilder {
    fn host(mut self, host: &str) -> Self {
        self.host = host.to_string();
        self
    }

    fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    fn max_connections(mut self, max: u32) -> Self {
        self.max_connections = max;
        self
    }

    fn build(self) -> ServerConfig {
        ServerConfig {
            host: self.host,
            port: self.port,
            max_connections: self.max_connections,
        }
    }
}
```
