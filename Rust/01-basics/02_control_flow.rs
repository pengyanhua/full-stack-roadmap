// ============================================================
//                      流程控制
// ============================================================
// Rust 的流程控制与其他语言类似，但有几个独特之处：
// 1. if 是表达式，可以返回值
// 2. loop 是无限循环，也是表达式
// 3. 模式匹配（match）非常强大
// 4. 条件不需要括号

fn main() {
    println!("=== 流程控制 ===");

    // ----------------------------------------------------------
    // 1. if 表达式
    // ----------------------------------------------------------
    // 【重要】if 的条件不需要括号（与 C/Java 不同）
    // 【重要】条件必须是 bool 类型（不能用数字代替）
    // 【特性】if 是表达式，可以有返回值

    let number = 7;

    if number > 5 {
        println!("{} 大于 5", number);
    } else if number > 0 {
        println!("{} 在 1-5 之间", number);
    } else {
        println!("{} 小于等于 0", number);
    }

    // 【技巧】if 作为表达式赋值（类似三元运算符）
    // Rust 没有三元运算符 ?:，用 if 表达式代替
    let condition = true;
    let value = if condition { 5 } else { 10 };
    println!("if 表达式: value = {}", value);

    // 【注意】两个分支的返回类型必须一致
    // let bad = if condition { 5 } else { "hello" };  // 错误！类型不同

    // ----------------------------------------------------------
    // 2. loop 无限循环
    // ----------------------------------------------------------
    // loop 创建无限循环，必须用 break 退出
    // 【特性】loop 是表达式，break 可以返回值
    // 【与 while true 的区别】loop 更明确表示"我知道这是无限循环"

    let mut counter = 0;
    let result = loop {
        counter += 1;
        if counter == 10 {
            break counter * 2;  // break 返回值
        }
    };
    println!("loop 返回值: {}", result); // 20

    // 【技巧】循环标签（处理嵌套循环）
    // 使用 'label 命名循环，break/continue 可以指定跳出哪层
    let mut count = 0;
    'outer: loop {
        let mut remaining = 10;
        loop {
            if remaining == 9 {
                break;  // 只退出内层循环
            }
            if count == 2 {
                break 'outer;  // 退出外层循环
            }
            remaining -= 1;
        }
        count += 1;
    }
    println!("嵌套循环: count = {}", count);

    // ----------------------------------------------------------
    // 3. while 条件循环
    // ----------------------------------------------------------
    let mut n = 5;
    while n > 0 {
        print!("{} ", n);
        n -= 1;
    }
    println!("发射！");

    // ----------------------------------------------------------
    // 4. for 循环（最常用）
    // ----------------------------------------------------------
    // for 遍历迭代器，是 Rust 中最常用的循环
    // 【优势】安全、简洁、不会越界

    // 遍历数组
    let fruits = ["苹果", "香蕉", "橙子", "葡萄"];
    for fruit in fruits.iter() {
        print!("{} ", fruit);
    }
    println!();

    // 遍历范围（Range）
    // 【语法】start..end（不包含 end）或 start..=end（包含 end）
    print!("1到5: ");
    for i in 1..=5 {
        print!("{} ", i);
    }
    println!();

    // 反向遍历
    print!("倒数: ");
    for i in (1..=5).rev() {
        print!("{} ", i);
    }
    println!();

    // 带索引遍历
    for (index, value) in fruits.iter().enumerate() {
        println!("  [{}] {}", index, value);
    }

    // 【技巧】用 for 代替 while 遍历
    // 差: while i < arr.len() { arr[i]; i += 1; }  // 每次访问有边界检查
    // 好: for item in &arr { ... }                   // 没有边界检查，更快

    // ----------------------------------------------------------
    // 5. match 模式匹配
    // ----------------------------------------------------------
    // match 是 Rust 最强大的控制流工具
    // 【重要】match 必须穷尽所有可能性（exhaustive）
    // 【类比】类似 switch，但功能强大得多

    let number = 3;
    match number {
        1 => println!("一"),
        2 => println!("二"),
        3 => println!("三"),
        4 | 5 => println!("四或五"),         // 多个值用 | 分隔
        6..=10 => println!("六到十"),        // 范围匹配
        _ => println!("其他"),              // _ 是通配符，匹配所有剩余
    }

    // match 也是表达式
    let text = match number {
        1 => "one",
        2 => "two",
        3 => "three",
        _ => "other",
    };
    println!("match 表达式: {}", text);

    // 匹配元组
    let point = (0, -2);
    match point {
        (0, 0) => println!("原点"),
        (x, 0) => println!("x 轴上, x={}", x),
        (0, y) => println!("y 轴上, y={}", y),
        (x, y) => println!("点 ({}, {})", x, y),
    }

    // 带条件的匹配（match guard）
    let num = Some(4);
    match num {
        Some(x) if x < 5 => println!("小于 5 的值: {}", x),
        Some(x) => println!("值: {}", x),
        None => println!("没有值"),
    }

    // ----------------------------------------------------------
    // 6. if let 简洁匹配
    // ----------------------------------------------------------
    // 当只关心一种模式时，用 if let 比 match 更简洁
    // 【适用场景】只需处理 Some/Ok 等单一情况

    let some_value: Option<i32> = Some(42);

    // 用 match（略繁琐）
    match some_value {
        Some(val) => println!("match: {}", val),
        None => {},
    }

    // 用 if let（更简洁）
    if let Some(val) = some_value {
        println!("if let: {}", val);
    }

    // if let 也可以有 else
    let no_value: Option<i32> = None;
    if let Some(val) = no_value {
        println!("有值: {}", val);
    } else {
        println!("if let else: 没有值");
    }

    // ----------------------------------------------------------
    // 7. while let 条件循环模式匹配
    // ----------------------------------------------------------
    // 与 if let 类似，在模式匹配成功时持续循环

    let mut stack = vec![1, 2, 3];
    print!("while let 弹出: ");
    while let Some(top) = stack.pop() {
        print!("{} ", top);
    }
    println!();

    // ----------------------------------------------------------
    // 8. let-else（Rust 1.65+）
    // ----------------------------------------------------------
    // 当模式匹配失败时执行 else 分支（必须发散：return, break, panic 等）
    // 【用途】减少嵌套，提前退出

    let config_value: Option<&str> = Some("42");

    // 用 let-else 简化
    let Some(val) = config_value else {
        println!("配置值缺失");
        return;
    };
    println!("let-else: 配置值 = {}", val);

    println!("\n=== 流程控制结束 ===");
}
