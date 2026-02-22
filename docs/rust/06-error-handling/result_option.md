# result option.rs

::: info 文件信息
- 📄 原文件：`01_result_option.rs`
- 🔤 语言：rust
:::

## 完整代码

```rust
// ============================================================
//                      错误处理
// ============================================================
// Rust 没有异常（exception），使用 Result<T, E> 和 Option<T> 处理错误
// 这是 Rust 可靠性的重要基石——编译器强制你处理所有可能的错误
//
// 两种错误：
// - 可恢复错误: Result<T, E>（如文件未找到、网络超时）
// - 不可恢复错误: panic!（如数组越界、断言失败）

use std::fs;
use std::io;
use std::num::ParseIntError;
use std::fmt;

fn main() {
    println!("=== 错误处理 ===");

    // ----------------------------------------------------------
    // 1. panic!（不可恢复错误）
    // ----------------------------------------------------------
    // 程序立即终止，打印错误信息和调用栈
    // 【适用场景】程序进入不可能恢复的状态
    // 【环境变量】RUST_BACKTRACE=1 显示完整调用栈

    // panic!("崩溃了！");  // 取消注释会终止程序

    // 数组越界也会 panic
    // let v = vec![1, 2, 3];
    // v[99];  // panic: index out of bounds

    // ----------------------------------------------------------
    // 2. Result<T, E> 基础
    // ----------------------------------------------------------
    println!("\n=== Result<T, E> ===");

    // 读取文件（可能失败）
    let result = fs::read_to_string("不存在的文件.txt");
    match result {
        Ok(content) => println!("文件内容: {}", content),
        Err(error) => println!("读取失败: {}", error),
    }

    // ----------------------------------------------------------
    // 3. ? 操作符（错误传播）
    // ----------------------------------------------------------
    // ? 是处理 Result 的语法糖：
    // - Ok(val) → 解包得到 val
    // - Err(e) → 提前返回 Err(e)
    //
    // 【重要】只能在返回 Result 或 Option 的函数中使用

    println!("\n=== ? 操作符 ===");

    match read_config("config.txt") {
        Ok(config) => println!("配置: {}", config),
        Err(e) => println!("读取配置失败: {}", e),
    }

    match parse_number("42") {
        Ok(n) => println!("解析成功: {}", n),
        Err(e) => println!("解析失败: {}", e),
    }

    match parse_number("abc") {
        Ok(n) => println!("解析成功: {}", n),
        Err(e) => println!("解析失败: {}", e),
    }

    // ----------------------------------------------------------
    // 4. Result 的常用方法
    // ----------------------------------------------------------
    println!("\n=== Result 方法 ===");

    let ok_val: Result<i32, String> = Ok(42);
    let err_val: Result<i32, String> = Err("错误".to_string());

    // unwrap / expect（有值返回，无值 panic）
    println!("unwrap: {}", ok_val.unwrap());
    // err_val.unwrap();  // panic!
    // err_val.expect("自定义 panic 信息");  // panic with message

    // unwrap_or / unwrap_or_else
    println!("unwrap_or: {}", err_val.unwrap_or(0));
    println!("unwrap_or_else: {}", err_val.unwrap_or_else(|_| 100));

    // map（转换 Ok 的值）
    let doubled = ok_val.map(|x| x * 2);
    println!("map: {:?}", doubled); // Ok(84)

    // map_err（转换 Err 的值）
    let mapped_err = err_val.map_err(|e| format!("包装: {}", e));
    println!("map_err: {:?}", mapped_err);

    // and_then（链式操作）
    let result = ok_val
        .and_then(|x| if x > 0 { Ok(x * 10) } else { Err("非正数".to_string()) });
    println!("and_then: {:?}", result);

    // is_ok / is_err
    println!("is_ok: {}, is_err: {}", ok_val.is_ok(), ok_val.is_err());

    // ok() / err() — 转换为 Option
    println!("ok(): {:?}", ok_val.ok());   // Some(42)
    println!("err(): {:?}", err_val.err()); // Some("错误")

    // ----------------------------------------------------------
    // 5. 自定义错误类型
    // ----------------------------------------------------------
    println!("\n=== 自定义错误 ===");

    match process_user_input("  ") {
        Ok(n) => println!("结果: {}", n),
        Err(e) => println!("错误: {}", e),
    }

    match process_user_input("abc") {
        Ok(n) => println!("结果: {}", n),
        Err(e) => println!("错误: {}", e),
    }

    match process_user_input("-5") {
        Ok(n) => println!("结果: {}", n),
        Err(e) => println!("错误: {}", e),
    }

    match process_user_input("42") {
        Ok(n) => println!("结果: {}", n),
        Err(e) => println!("错误: {}", e),
    }

    // ----------------------------------------------------------
    // 6. Box<dyn Error>（通用错误类型）
    // ----------------------------------------------------------
    // 当函数可能返回多种错误类型时，用 Box<dyn Error> 统一
    // 【适用场景】应用层代码、快速原型
    // 【不适用】库代码（应该用自定义错误类型）

    println!("\n=== Box<dyn Error> ===");

    match read_and_parse("config.txt") {
        Ok(n) => println!("值: {}", n),
        Err(e) => println!("错误: {}", e),
    }

    // ----------------------------------------------------------
    // 7. 多种错误类型的转换（From trait）
    // ----------------------------------------------------------
    println!("\n=== From 转换 ===");

    match read_config_typed("settings.txt") {
        Ok(val) => println!("配置值: {}", val),
        Err(e) => {
            match e {
                ConfigError::Io(ref io_err) => println!("IO 错误: {}", io_err),
                ConfigError::Parse(ref parse_err) => println!("解析错误: {}", parse_err),
                ConfigError::Validation(ref msg) => println!("验证错误: {}", msg),
            }
        }
    }

    // ----------------------------------------------------------
    // 8. 实用模式
    // ----------------------------------------------------------
    println!("\n=== 实用模式 ===");

    // 模式1: 收集 Result 的迭代器
    let strings = vec!["1", "2", "abc", "4"];

    // 遇到第一个错误就停止
    let numbers: Result<Vec<i32>, _> = strings.iter().map(|s| s.parse::<i32>()).collect();
    println!("collect Result: {:?}", numbers); // Err

    // 分离成功和失败
    let (successes, failures): (Vec<_>, Vec<_>) = strings
        .iter()
        .map(|s| s.parse::<i32>())
        .partition(Result::is_ok);

    let successes: Vec<i32> = successes.into_iter().map(Result::unwrap).collect();
    let failures: Vec<_> = failures.into_iter().map(Result::unwrap_err).collect();
    println!("成功: {:?}", successes);
    println!("失败: {:?}", failures);

    // 模式2: 忽略错误（只关心成功的）
    let numbers: Vec<i32> = strings
        .iter()
        .filter_map(|s| s.parse::<i32>().ok())
        .collect();
    println!("filter_map: {:?}", numbers);

    println!("\n=== 错误处理结束 ===");
}

// ----------------------------------------------------------
// ? 操作符示例
// ----------------------------------------------------------
fn read_config(path: &str) -> Result<String, io::Error> {
    let content = fs::read_to_string(path)?;  // 失败则提前返回 Err
    Ok(content.trim().to_string())
}

fn parse_number(s: &str) -> Result<i32, ParseIntError> {
    let n = s.trim().parse::<i32>()?;
    Ok(n * 2)
}

// ----------------------------------------------------------
// 自定义错误类型
// ----------------------------------------------------------
#[derive(Debug)]
enum InputError {
    Empty,
    InvalidFormat(ParseIntError),
    OutOfRange(i32),
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InputError::Empty => write!(f, "输入为空"),
            InputError::InvalidFormat(e) => write!(f, "格式无效: {}", e),
            InputError::OutOfRange(n) => write!(f, "值 {} 超出范围 (0-100)", n),
        }
    }
}

// 实现 From，让 ? 自动转换错误类型
impl From<ParseIntError> for InputError {
    fn from(err: ParseIntError) -> InputError {
        InputError::InvalidFormat(err)
    }
}

fn process_user_input(input: &str) -> Result<i32, InputError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(InputError::Empty);
    }

    let n: i32 = trimmed.parse()?;  // ParseIntError 自动转为 InputError

    if n < 0 || n > 100 {
        return Err(InputError::OutOfRange(n));
    }

    Ok(n)
}

// ----------------------------------------------------------
// Box<dyn Error>
// ----------------------------------------------------------
fn read_and_parse(path: &str) -> Result<i32, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;  // io::Error
    let n = content.trim().parse::<i32>()?;   // ParseIntError
    Ok(n)
}

// ----------------------------------------------------------
// 完整的自定义错误类型（库级别）
// ----------------------------------------------------------
#[derive(Debug)]
enum ConfigError {
    Io(io::Error),
    Parse(ParseIntError),
    Validation(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ConfigError::Io(e) => write!(f, "IO 错误: {}", e),
            ConfigError::Parse(e) => write!(f, "解析错误: {}", e),
            ConfigError::Validation(msg) => write!(f, "验证错误: {}", msg),
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<io::Error> for ConfigError {
    fn from(err: io::Error) -> ConfigError {
        ConfigError::Io(err)
    }
}

impl From<ParseIntError> for ConfigError {
    fn from(err: ParseIntError) -> ConfigError {
        ConfigError::Parse(err)
    }
}

fn read_config_typed(path: &str) -> Result<i32, ConfigError> {
    let content = fs::read_to_string(path)?;
    let value: i32 = content.trim().parse()?;
    if value < 0 {
        return Err(ConfigError::Validation("值不能为负".to_string()));
    }
    Ok(value)
}
```
