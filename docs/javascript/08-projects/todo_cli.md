# todo cli.js

::: info 文件信息
- 📄 原文件：`01_todo_cli.js`
- 🔤 语言：javascript
:::

Todo 命令行应用
一个简单的命令行 Todo 应用，演示 Node.js 开发。
用法：
  node 01_todo_cli.js add "任务内容"
  node 01_todo_cli.js list
  node 01_todo_cli.js done `&lt;id&gt;`
  node 01_todo_cli.js remove `&lt;id&gt;`
  node 01_todo_cli.js clear

## 完整代码

```javascript
/**
 * ============================================================
 *                Todo 命令行应用
 * ============================================================
 * 一个简单的命令行 Todo 应用，演示 Node.js 开发。
 *
 * 用法：
 *   node 01_todo_cli.js add "任务内容"
 *   node 01_todo_cli.js list
 *   node 01_todo_cli.js done <id>
 *   node 01_todo_cli.js remove <id>
 *   node 01_todo_cli.js clear
 * ============================================================
 */

const fs = require("fs");
const path = require("path");
const readline = require("readline");

// 数据文件路径
const DATA_FILE = path.join(__dirname, "todos.json");

// ============================================================
//                    数据管理
// ============================================================

/**
 * 加载 Todo 列表
 */
function loadTodos() {
    try {
        if (fs.existsSync(DATA_FILE)) {
            const data = fs.readFileSync(DATA_FILE, "utf8");
            return JSON.parse(data);
        }
    } catch (error) {
        console.error("加载数据失败:", error.message);
    }
    return [];
}

/**
 * 保存 Todo 列表
 */
function saveTodos(todos) {
    try {
        fs.writeFileSync(DATA_FILE, JSON.stringify(todos, null, 2));
    } catch (error) {
        console.error("保存数据失败:", error.message);
    }
}

/**
 * 生成唯一 ID
 */
function generateId(todos) {
    if (todos.length === 0) return 1;
    return Math.max(...todos.map(t => t.id)) + 1;
}

// ============================================================
//                    命令处理
// ============================================================

/**
 * 添加任务
 */
function addTodo(content) {
    if (!content || content.trim().length === 0) {
        console.log("❌ 任务内容不能为空");
        return;
    }

    const todos = loadTodos();
    const newTodo = {
        id: generateId(todos),
        content: content.trim(),
        completed: false,
        createdAt: new Date().toISOString()
    };

    todos.push(newTodo);
    saveTodos(todos);

    console.log(`✅ 已添加任务 [${newTodo.id}]: ${newTodo.content}`);
}

/**
 * 列出任务
 */
function listTodos() {
    const todos = loadTodos();

    if (todos.length === 0) {
        console.log("📋 暂无任务");
        return;
    }

    console.log("\n" + "=".repeat(50));
    console.log("📋 Todo 列表");
    console.log("=".repeat(50));

    const pending = todos.filter(t => !t.completed);
    const completed = todos.filter(t => t.completed);

    if (pending.length > 0) {
        console.log("\n【待办】");
        pending.forEach(todo => {
            console.log(`  [${todo.id}] ○ ${todo.content}`);
        });
    }

    if (completed.length > 0) {
        console.log("\n【已完成】");
        completed.forEach(todo => {
            console.log(`  [${todo.id}] ● ${todo.content}`);
        });
    }

    console.log("\n" + "-".repeat(50));
    console.log(`总计: ${todos.length} 项 | 待办: ${pending.length} | 已完成: ${completed.length}`);
    console.log();
}

/**
 * 完成任务
 */
function completeTodo(id) {
    const todos = loadTodos();
    const todo = todos.find(t => t.id === parseInt(id));

    if (!todo) {
        console.log(`❌ 未找到任务 ID: ${id}`);
        return;
    }

    if (todo.completed) {
        console.log(`⚠️ 任务已经完成: ${todo.content}`);
        return;
    }

    todo.completed = true;
    todo.completedAt = new Date().toISOString();
    saveTodos(todos);

    console.log(`✅ 已完成任务 [${todo.id}]: ${todo.content}`);
}

/**
 * 删除任务
 */
function removeTodo(id) {
    const todos = loadTodos();
    const index = todos.findIndex(t => t.id === parseInt(id));

    if (index === -1) {
        console.log(`❌ 未找到任务 ID: ${id}`);
        return;
    }

    const removed = todos.splice(index, 1)[0];
    saveTodos(todos);

    console.log(`🗑️ 已删除任务 [${removed.id}]: ${removed.content}`);
}

/**
 * 清空所有任务
 */
function clearTodos() {
    saveTodos([]);
    console.log("🗑️ 已清空所有任务");
}

/**
 * 显示帮助
 */
function showHelp() {
    console.log(`
📝 Todo CLI - 命令行任务管理工具

用法:
  node 01_todo_cli.js <命令> [参数]

命令:
  add <内容>     添加新任务
  list           列出所有任务
  done <id>      标记任务完成
  remove <id>    删除任务
  clear          清空所有任务
  help           显示帮助

示例:
  node 01_todo_cli.js add "学习 Node.js"
  node 01_todo_cli.js list
  node 01_todo_cli.js done 1
  node 01_todo_cli.js remove 1
`);
}

// ============================================================
//                    交互模式
// ============================================================

/**
 * 交互模式
 */
function interactiveMode() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    console.log("\n📝 Todo CLI 交互模式");
    console.log("输入 help 查看命令，exit 退出\n");

    const prompt = () => {
        rl.question("todo> ", (input) => {
            const parts = input.trim().split(" ");
            const command = parts[0].toLowerCase();
            const args = parts.slice(1).join(" ");

            switch (command) {
                case "add":
                    addTodo(args);
                    break;
                case "list":
                case "ls":
                    listTodos();
                    break;
                case "done":
                case "complete":
                    completeTodo(args);
                    break;
                case "remove":
                case "rm":
                case "delete":
                    removeTodo(args);
                    break;
                case "clear":
                    clearTodos();
                    break;
                case "help":
                    showHelp();
                    break;
                case "exit":
                case "quit":
                case "q":
                    console.log("再见！👋");
                    rl.close();
                    return;
                case "":
                    break;
                default:
                    console.log(`未知命令: ${command}，输入 help 查看帮助`);
            }

            prompt();
        });
    };

    prompt();
}

// ============================================================
//                    主程序
// ============================================================

function main() {
    const args = process.argv.slice(2);

    if (args.length === 0) {
        // 无参数时进入交互模式
        interactiveMode();
        return;
    }

    const command = args[0].toLowerCase();
    const param = args.slice(1).join(" ");

    switch (command) {
        case "add":
            addTodo(param);
            break;
        case "list":
        case "ls":
            listTodos();
            break;
        case "done":
        case "complete":
            completeTodo(param);
            break;
        case "remove":
        case "rm":
        case "delete":
            removeTodo(param);
            break;
        case "clear":
            clearTodos();
            break;
        case "help":
        case "--help":
        case "-h":
            showHelp();
            break;
        default:
            console.log(`未知命令: ${command}`);
            showHelp();
    }
}

main();
```
