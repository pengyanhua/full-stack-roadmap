// ============================================================
//                      项目实战：Todo CLI
// ============================================================
// 一个简单的命令行 Todo 应用，综合运用：
// - 结构体与方法
// - 枚举与模式匹配
// - Vec 和 HashMap
// - 错误处理
// - 文件 IO
// - 字符串处理
//
// 用法:
//   cargo run -- add "买牛奶"
//   cargo run -- list
//   cargo run -- done 1
//   cargo run -- remove 1

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io;

// ----------------------------------------------------------
// 数据模型
// ----------------------------------------------------------
#[derive(Debug, Clone)]
enum Status {
    Pending,
    Done,
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Status::Pending => write!(f, "[ ]"),
            Status::Done => write!(f, "[✓]"),
        }
    }
}

#[derive(Debug, Clone)]
struct Todo {
    id: u32,
    title: String,
    status: Status,
}

impl fmt::Display for Todo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "#{:03} {} {}", self.id, self.status, self.title)
    }
}

// ----------------------------------------------------------
// Todo 管理器
// ----------------------------------------------------------
struct TodoManager {
    todos: Vec<Todo>,
    next_id: u32,
    file_path: String,
}

impl TodoManager {
    // 创建新的管理器，从文件加载数据
    fn new(file_path: &str) -> TodoManager {
        let mut manager = TodoManager {
            todos: Vec::new(),
            next_id: 1,
            file_path: file_path.to_string(),
        };
        // 尝试从文件加载
        if let Ok(()) = manager.load() {
            // 更新 next_id
            if let Some(max_id) = manager.todos.iter().map(|t| t.id).max() {
                manager.next_id = max_id + 1;
            }
        }
        manager
    }

    // 添加任务
    fn add(&mut self, title: &str) -> &Todo {
        let todo = Todo {
            id: self.next_id,
            title: title.trim().to_string(),
            status: Status::Pending,
        };
        self.next_id += 1;
        self.todos.push(todo);
        self.todos.last().unwrap()
    }

    // 列出所有任务
    fn list(&self) -> &[Todo] {
        &self.todos
    }

    // 标记完成
    fn mark_done(&mut self, id: u32) -> Result<&Todo, String> {
        match self.todos.iter_mut().find(|t| t.id == id) {
            Some(todo) => {
                todo.status = Status::Done;
                Ok(todo)
            }
            None => Err(format!("任务 #{} 不存在", id)),
        }
    }

    // 删除任务
    fn remove(&mut self, id: u32) -> Result<Todo, String> {
        let pos = self
            .todos
            .iter()
            .position(|t| t.id == id)
            .ok_or_else(|| format!("任务 #{} 不存在", id))?;
        Ok(self.todos.remove(pos))
    }

    // 统计
    fn stats(&self) -> HashMap<&str, usize> {
        let mut stats = HashMap::new();
        let total = self.todos.len();
        let done = self
            .todos
            .iter()
            .filter(|t| matches!(t.status, Status::Done))
            .count();
        let pending = total - done;

        stats.insert("总计", total);
        stats.insert("已完成", done);
        stats.insert("待完成", pending);
        stats
    }

    // ----------------------------------------------------------
    // 持久化（简单的文本文件格式）
    // ----------------------------------------------------------
    // 格式: id|status|title
    // 例如: 1|pending|买牛奶

    fn save(&self) -> io::Result<()> {
        let content: String = self
            .todos
            .iter()
            .map(|t| {
                let status = match t.status {
                    Status::Pending => "pending",
                    Status::Done => "done",
                };
                format!("{}|{}|{}", t.id, status, t.title)
            })
            .collect::<Vec<_>>()
            .join("\n");

        fs::write(&self.file_path, content)
    }

    fn load(&mut self) -> Result<(), io::Error> {
        let content = fs::read_to_string(&self.file_path)?;

        self.todos = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .filter_map(|line| {
                let parts: Vec<&str> = line.splitn(3, '|').collect();
                if parts.len() == 3 {
                    let id = parts[0].parse().ok()?;
                    let status = match parts[1] {
                        "done" => Status::Done,
                        _ => Status::Pending,
                    };
                    let title = parts[2].to_string();
                    Some(Todo { id, title, status })
                } else {
                    None
                }
            })
            .collect();

        Ok(())
    }
}

// ----------------------------------------------------------
// 命令解析
// ----------------------------------------------------------
#[derive(Debug)]
enum Command {
    Add(String),
    List,
    Done(u32),
    Remove(u32),
    Stats,
    Help,
}

fn parse_command(args: &[String]) -> Result<Command, String> {
    if args.len() < 2 {
        return Ok(Command::Help);
    }

    match args[1].as_str() {
        "add" => {
            if args.len() < 3 {
                Err("用法: todo add \"任务内容\"".to_string())
            } else {
                Ok(Command::Add(args[2..].join(" ")))
            }
        }
        "list" | "ls" => Ok(Command::List),
        "done" => {
            if args.len() < 3 {
                Err("用法: todo done <id>".to_string())
            } else {
                let id = args[2]
                    .parse::<u32>()
                    .map_err(|_| "ID 必须是正整数".to_string())?;
                Ok(Command::Done(id))
            }
        }
        "remove" | "rm" => {
            if args.len() < 3 {
                Err("用法: todo remove <id>".to_string())
            } else {
                let id = args[2]
                    .parse::<u32>()
                    .map_err(|_| "ID 必须是正整数".to_string())?;
                Ok(Command::Remove(id))
            }
        }
        "stats" => Ok(Command::Stats),
        "help" | "--help" | "-h" => Ok(Command::Help),
        other => Err(format!("未知命令: {}", other)),
    }
}

fn print_help() {
    println!("Todo CLI - 命令行任务管理");
    println!();
    println!("用法:");
    println!("  todo add <任务>     添加新任务");
    println!("  todo list           列出所有任务");
    println!("  todo done <id>      标记任务完成");
    println!("  todo remove <id>    删除任务");
    println!("  todo stats          显示统计");
    println!("  todo help           显示帮助");
}

// ----------------------------------------------------------
// 主函数
// ----------------------------------------------------------
fn main() {
    let args: Vec<String> = std::env::args().collect();

    let command = match parse_command(&args) {
        Ok(cmd) => cmd,
        Err(e) => {
            eprintln!("错误: {}", e);
            eprintln!("使用 'todo help' 查看帮助");
            std::process::exit(1);
        }
    };

    let mut manager = TodoManager::new("todos.txt");

    match command {
        Command::Add(title) => {
            let todo = manager.add(&title);
            println!("已添加: {}", todo);
        }
        Command::List => {
            let todos = manager.list();
            if todos.is_empty() {
                println!("没有任务。使用 'todo add \"任务\"' 添加");
            } else {
                println!("--- 任务列表 ---");
                for todo in todos {
                    println!("  {}", todo);
                }
                println!("--- 共 {} 项 ---", todos.len());
            }
        }
        Command::Done(id) => match manager.mark_done(id) {
            Ok(todo) => println!("已完成: {}", todo),
            Err(e) => eprintln!("错误: {}", e),
        },
        Command::Remove(id) => match manager.remove(id) {
            Ok(todo) => println!("已删除: {}", todo),
            Err(e) => eprintln!("错误: {}", e),
        },
        Command::Stats => {
            let stats = manager.stats();
            println!("--- 统计 ---");
            println!("  总计: {}", stats.get("总计").unwrap_or(&0));
            println!("  已完成: {}", stats.get("已完成").unwrap_or(&0));
            println!("  待完成: {}", stats.get("待完成").unwrap_or(&0));
        }
        Command::Help => print_help(),
    }

    // 保存到文件
    if let Err(e) = manager.save() {
        eprintln!("保存失败: {}", e);
    }
}
