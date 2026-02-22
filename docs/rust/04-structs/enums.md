# enums.rs

::: info 文件信息
- 📄 原文件：`02_enums.rs`
- 🔤 语言：rust
:::

## 完整代码

```rust
// ============================================================
//                      枚举与模式匹配
// ============================================================
// Rust 的枚举（enum）远比其他语言的枚举强大
// 每个变体可以携带不同类型和数量的数据
// 结合模式匹配（match），是 Rust 中最常用的模式之一
//
// 【类比】类似 TypeScript 的联合类型，但更安全

// ----------------------------------------------------------
// 1. 基本枚举
// ----------------------------------------------------------
#[derive(Debug)]
enum Direction {
    Up,
    Down,
    Left,
    Right,
}

// ----------------------------------------------------------
// 2. 携带数据的枚举
// ----------------------------------------------------------
// 每个变体可以有不同的数据类型
#[derive(Debug)]
enum Message {
    Quit,                       // 无数据
    Move { x: i32, y: i32 },   // 命名字段（类似结构体）
    Write(String),              // 单个 String
    ChangeColor(u8, u8, u8),    // 三个 u8（类似元组）
}

// ----------------------------------------------------------
// 3. 枚举也可以有方法
// ----------------------------------------------------------
impl Message {
    fn call(&self) {
        match self {
            Message::Quit => println!("退出"),
            Message::Move { x, y } => println!("移动到 ({}, {})", x, y),
            Message::Write(text) => println!("写入: {}", text),
            Message::ChangeColor(r, g, b) => println!("颜色: ({}, {}, {})", r, g, b),
        }
    }
}

// ----------------------------------------------------------
// 4. Option<T>（Rust 没有 null）
// ----------------------------------------------------------
// Rust 没有 null 值！用 Option<T> 表示"可能没有值"
// enum Option<T> {
//     Some(T),  // 有值
//     None,     // 无值
// }
// 【重要】Option 在标准库预导入，不需要 use
// 【优势】编译器强制你处理 None 的情况，避免空指针异常

// ----------------------------------------------------------
// 5. Result<T, E>（错误处理）
// ----------------------------------------------------------
// enum Result<T, E> {
//     Ok(T),   // 成功
//     Err(E),  // 失败
// }
// 详见 06-error-handling

fn main() {
    println!("=== 枚举与模式匹配 ===");

    // ----------------------------------------------------------
    // 基本枚举使用
    // ----------------------------------------------------------
    let dir = Direction::Up;
    println!("方向: {:?}", dir);

    // match 匹配枚举
    let description = match dir {
        Direction::Up => "上",
        Direction::Down => "下",
        Direction::Left => "左",
        Direction::Right => "右",
    };
    println!("方向描述: {}", description);

    // ----------------------------------------------------------
    // 携带数据的枚举
    // ----------------------------------------------------------
    let messages = vec![
        Message::Quit,
        Message::Move { x: 10, y: 20 },
        Message::Write(String::from("你好")),
        Message::ChangeColor(255, 128, 0),
    ];

    for msg in &messages {
        msg.call();
    }

    // ----------------------------------------------------------
    // Option<T> 使用
    // ----------------------------------------------------------
    println!("\n=== Option<T> ===");

    let some_number: Option<i32> = Some(42);
    let no_number: Option<i32> = None;

    println!("some_number: {:?}", some_number);
    println!("no_number: {:?}", no_number);

    // 【重要】不能直接使用 Option<T> 的值，必须先"解包"
    // let result = some_number + 1;  // 错误！不能对 Option<i32> 做运算

    // 方式1: match
    match some_number {
        Some(n) => println!("match: 值是 {}", n),
        None => println!("match: 没有值"),
    }

    // 方式2: if let（只关心 Some 的情况）
    if let Some(n) = some_number {
        println!("if let: 值是 {}", n);
    }

    // 方式3: unwrap（有值返回值，无值 panic）
    // 【警告】仅在确定有值时使用，或者在原型代码中
    println!("unwrap: {}", some_number.unwrap());

    // 方式4: unwrap_or（提供默认值）
    println!("unwrap_or: {}", no_number.unwrap_or(0));

    // 方式5: unwrap_or_else（惰性默认值）
    println!("unwrap_or_else: {}", no_number.unwrap_or_else(|| {
        // 复杂的默认值计算
        42
    }));

    // 方式6: map（转换 Some 中的值）
    let doubled = some_number.map(|n| n * 2);
    println!("map: {:?}", doubled); // Some(84)

    // 方式7: and_then（链式操作，类似 flatMap）
    let result = some_number
        .and_then(|n| if n > 0 { Some(n * 10) } else { None })
        .and_then(|n| Some(n + 1));
    println!("and_then: {:?}", result); // Some(421)

    // 方式8: ? 操作符（在返回 Option 的函数中使用）
    println!("查找: {:?}", find_in_array(&[10, 20, 30], 20));
    println!("查找: {:?}", find_in_array(&[10, 20, 30], 50));

    // ----------------------------------------------------------
    // 模式匹配进阶
    // ----------------------------------------------------------
    println!("\n=== 模式匹配进阶 ===");

    // 解构复杂数据
    let msg = Message::Move { x: 10, y: 20 };
    if let Message::Move { x, y } = msg {
        println!("解构: x={}, y={}", x, y);
    }

    // 匹配多种模式
    let x = 4;
    match x {
        1 | 2 => println!("一或二"),
        3..=5 => println!("三到五"),  // 范围匹配
        _ => println!("其他"),
    }

    // 匹配守卫（Match Guard）
    let num = Some(4);
    match num {
        Some(x) if x < 0 => println!("负数: {}", x),
        Some(0) => println!("零"),
        Some(x) if x > 0 => println!("正数: {}", x),
        Some(_) => unreachable!(),
        None => println!("无值"),
    }

    // @ 绑定（匹配并捕获值）
    let age = 25;
    match age {
        n @ 0..=12 => println!("儿童, 年龄 {}", n),
        n @ 13..=17 => println!("青少年, 年龄 {}", n),
        n @ 18..=64 => println!("成人, 年龄 {}", n),
        n => println!("老年, 年龄 {}", n),
    }

    // 解构嵌套
    let points = vec![(0, 0), (1, 5), (10, -3)];
    for &(x, y) in &points {
        match (x, y) {
            (0, 0) => println!("在原点"),
            (x, 0) => println!("在 x 轴: x={}", x),
            (0, y) => println!("在 y 轴: y={}", y),
            (x, y) => println!("在 ({}, {})", x, y),
        }
    }

    // ----------------------------------------------------------
    // 实用枚举示例
    // ----------------------------------------------------------
    println!("\n=== 实用示例 ===");

    // 用枚举表示 JSON 值
    let json_data = vec![
        JsonValue::Null,
        JsonValue::Bool(true),
        JsonValue::Number(42.0),
        JsonValue::Str(String::from("hello")),
        JsonValue::Array(vec![
            JsonValue::Number(1.0),
            JsonValue::Number(2.0),
        ]),
    ];

    for value in &json_data {
        println!("  {}", value.to_string());
    }

    // 用枚举表示状态机
    let mut state = TrafficLight::Red;
    for _ in 0..6 {
        println!("交通灯: {:?} (等待 {} 秒)", state, state.duration());
        state = state.next();
    }

    println!("\n=== 枚举与模式匹配结束 ===");
}

// ----------------------------------------------------------
// Option 与 ? 操作符
// ----------------------------------------------------------
fn find_in_array(arr: &[i32], target: i32) -> Option<usize> {
    for (i, &val) in arr.iter().enumerate() {
        if val == target {
            return Some(i);
        }
    }
    None
}

// ----------------------------------------------------------
// 实用枚举：JSON 值
// ----------------------------------------------------------
#[derive(Debug)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    Str(String),
    Array(Vec<JsonValue>),
}

impl JsonValue {
    fn to_string(&self) -> String {
        match self {
            JsonValue::Null => "null".to_string(),
            JsonValue::Bool(b) => b.to_string(),
            JsonValue::Number(n) => n.to_string(),
            JsonValue::Str(s) => format!("\"{}\"", s),
            JsonValue::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| v.to_string()).collect();
                format!("[{}]", items.join(", "))
            }
        }
    }
}

// ----------------------------------------------------------
// 实用枚举：状态机
// ----------------------------------------------------------
#[derive(Debug)]
enum TrafficLight {
    Red,
    Yellow,
    Green,
}

impl TrafficLight {
    fn duration(&self) -> u32 {
        match self {
            TrafficLight::Red => 60,
            TrafficLight::Yellow => 5,
            TrafficLight::Green => 45,
        }
    }

    fn next(self) -> TrafficLight {
        match self {
            TrafficLight::Red => TrafficLight::Green,
            TrafficLight::Green => TrafficLight::Yellow,
            TrafficLight::Yellow => TrafficLight::Red,
        }
    }
}
```
