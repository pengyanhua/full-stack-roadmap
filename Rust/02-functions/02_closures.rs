// ============================================================
//                      闭包（Closures）
// ============================================================
// 闭包是可以捕获所在环境变量的匿名函数
// 语法: |参数| 表达式  或  |参数| { 代码块 }
// 【与函数的区别】
//   1. 闭包可以捕获外部变量
//   2. 闭包参数和返回值类型可以推断（函数不行）
//   3. 闭包有三种 trait：Fn、FnMut、FnOnce

fn main() {
    println!("=== 闭包 ===");

    // ----------------------------------------------------------
    // 1. 闭包基本语法
    // ----------------------------------------------------------
    // 最简语法（单表达式，自动推断类型）
    let add_one = |x| x + 1;
    println!("add_one(5) = {}", add_one(5));

    // 带类型标注
    let add = |a: i32, b: i32| -> i32 { a + b };
    println!("add(3, 4) = {}", add(3, 4));

    // 多行闭包
    let calculate = |x: i32, y: i32| {
        let sum = x + y;
        let product = x * y;
        (sum, product)  // 返回元组
    };
    let (sum, product) = calculate(3, 4);
    println!("sum={}, product={}", sum, product);

    // 无参数闭包
    let say_hi = || println!("嗨！");
    say_hi();

    // ----------------------------------------------------------
    // 2. 捕获环境变量
    // ----------------------------------------------------------
    // 闭包会自动捕获外部变量，有三种捕获方式：
    // 1. 不可变借用（&T）  → 实现 Fn
    // 2. 可变借用（&mut T）→ 实现 FnMut
    // 3. 获取所有权（T）   → 实现 FnOnce

    // 不可变借用（默认优先选择）
    let name = String::from("Rust");
    let greet = || println!("你好, {}!", name);  // 借用 name
    greet();
    println!("name 仍然可用: {}", name);  // name 还在

    // 可变借用
    let mut count = 0;
    let mut increment = || {
        count += 1;  // 可变借用 count
        println!("计数: {}", count);
    };
    increment();
    increment();
    increment();
    // 【注意】increment 持有 count 的可变借用，
    //        在 increment 不再使用后才能再访问 count
    println!("最终计数: {}", count);

    // 获取所有权（使用 move 关键字）
    let data = vec![1, 2, 3];
    let print_data = move || {
        println!("data: {:?}", data);  // data 的所有权被移入闭包
    };
    print_data();
    // println!("{:?}", data);  // 错误！data 已被移入闭包

    // 【适用场景】move 在多线程中非常常用
    // 因为闭包需要在另一个线程中执行，必须拥有数据

    // ----------------------------------------------------------
    // 3. 闭包作为函数参数
    // ----------------------------------------------------------
    // 使用泛型 + trait bound 或 impl trait

    // 方式1: 泛型（推荐，可以内联优化）
    let result = apply_to_5(|x| x * 3);
    println!("apply_to_5(|x| x * 3) = {}", result);

    // 方式2: 作为 trait 对象（运行时多态，有额外开销）
    let ops: Vec<Box<dyn Fn(i32) -> i32>> = vec![
        Box::new(|x| x + 1),
        Box::new(|x| x * 2),
        Box::new(|x| x * x),
    ];
    for (i, op) in ops.iter().enumerate() {
        println!("ops[{}](5) = {}", i, op(5));
    }

    // ----------------------------------------------------------
    // 4. 闭包作为返回值
    // ----------------------------------------------------------
    let adder = make_adder(10);
    println!("make_adder(10)(5) = {}", adder(5));

    let multiplier = make_multiplier(3);
    println!("make_multiplier(3)(7) = {}", multiplier(7));

    // ----------------------------------------------------------
    // 5. Fn, FnMut, FnOnce 的区别
    // ----------------------------------------------------------
    // Fn:     可以多次调用，不修改捕获的变量（&self）
    // FnMut:  可以多次调用，可以修改捕获的变量（&mut self）
    // FnOnce: 只能调用一次，可以消耗捕获的变量（self）
    //
    // 继承关系: Fn : FnMut : FnOnce
    // 所有 Fn 都是 FnMut，所有 FnMut 都是 FnOnce

    // Fn 示例：不修改捕获的变量
    let x = 10;
    let fn_closure = || println!("Fn: x = {}", x);
    call_fn(&fn_closure);
    call_fn(&fn_closure);  // 可以多次调用

    // FnMut 示例：修改捕获的变量
    let mut total = 0;
    let mut fn_mut_closure = || {
        total += 1;
        println!("FnMut: total = {}", total);
    };
    call_fn_mut(&mut fn_mut_closure);
    call_fn_mut(&mut fn_mut_closure);

    // FnOnce 示例：消耗捕获的变量
    let data = String::from("一次性数据");
    let fn_once_closure = || {
        let _moved = data;  // 获取所有权
        println!("FnOnce: 数据已消耗");
    };
    call_fn_once(fn_once_closure);
    // call_fn_once(fn_once_closure);  // 错误！只能调用一次

    // ----------------------------------------------------------
    // 6. 实用示例
    // ----------------------------------------------------------
    println!("\n=== 实用示例 ===");

    // 过滤
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let evens: Vec<&i32> = numbers.iter().filter(|&&x| x % 2 == 0).collect();
    println!("偶数: {:?}", evens);

    // 映射
    let doubled: Vec<i32> = numbers.iter().map(|&x| x * 2).collect();
    println!("翻倍: {:?}", doubled);

    // 折叠（类似 reduce）
    let sum: i32 = numbers.iter().fold(0, |acc, &x| acc + x);
    println!("总和: {}", sum);

    // 排序（自定义比较）
    let mut words = vec!["banana", "apple", "cherry", "date"];
    words.sort_by(|a, b| a.len().cmp(&b.len()));  // 按长度排序
    println!("按长度排序: {:?}", words);

    // 组合迭代器操作（链式调用）
    let result: Vec<String> = (1..=5)
        .map(|x| x * x)
        .filter(|&x| x > 5)
        .map(|x| format!("{}²", (x as f64).sqrt() as i32))
        .collect();
    println!("链式操作: {:?}", result);

    println!("\n=== 闭包结束 ===");
}

// ----------------------------------------------------------
// 接受 Fn 闭包作为参数
// ----------------------------------------------------------
fn apply_to_5<F: Fn(i32) -> i32>(f: F) -> i32 {
    f(5)
}

// ----------------------------------------------------------
// 返回闭包（必须使用 impl Fn 或 Box<dyn Fn>）
// ----------------------------------------------------------
fn make_adder(x: i32) -> impl Fn(i32) -> i32 {
    move |y| x + y  // move 获取 x 的所有权
}

fn make_multiplier(factor: i32) -> Box<dyn Fn(i32) -> i32> {
    Box::new(move |x| x * factor)
}

// ----------------------------------------------------------
// Fn / FnMut / FnOnce 参数示例
// ----------------------------------------------------------
fn call_fn(f: &dyn Fn()) {
    f();
}

fn call_fn_mut(f: &mut dyn FnMut()) {
    f();
}

fn call_fn_once<F: FnOnce()>(f: F) {
    f();
}
