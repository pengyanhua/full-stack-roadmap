# todo cli.cpp

::: info 文件信息
- 📄 原文件：`01_todo_cli.cpp`
- 🔤 语言：cpp
:::

## 完整代码

```cpp
// ============================================================
//                      项目实战：Todo CLI（C++ 现代风格）
// ============================================================
// 综合运用现代 C++ 特性：
//   类与 RAII、智能指针、optional、variant、ranges
//   JSON 序列化（手写）、文件 I/O、异常处理
// 编译：g++ -std=c++20 -Wall -O2 -o todo 01_todo_cli.cpp

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <algorithm>
#include <functional>
#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <memory>
#include <ranges>

// ============================================================
//                      数据模型
// ============================================================

enum class Priority { Low, Medium, High };

struct TodoItem {
    int         id;
    std::string title;
    bool        completed = false;
    Priority    priority = Priority::Medium;
    std::string created_at;

    // 优先级字符串转换
    static std::string priority_str(Priority p) {
        switch (p) {
            case Priority::High:   return "high";
            case Priority::Low:    return "low";
            default:               return "medium";
        }
    }
    static Priority priority_from_str(std::string_view s) {
        if (s == "high")   return Priority::High;
        if (s == "low")    return Priority::Low;
        return Priority::Medium;
    }

    std::string priority_icon() const {
        switch (priority) {
            case Priority::High:   return "[H]";
            case Priority::Low:    return "[L]";
            default:               return "[M]";
        }
    }

    std::string status_icon() const {
        return completed ? "[x]" : "[ ]";
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << std::setw(4) << id << " " << status_icon()
            << " " << priority_icon() << " " << title;
        if (!created_at.empty())
            oss << "  (" << created_at << ")";
        return oss.str();
    }
};

// ============================================================
//                      JSON 序列化（轻量手写实现）
// ============================================================

class JsonBuilder {
    std::ostringstream oss_;
    bool first_ = true;

    void sep() { if (!first_) oss_ << ",\n"; first_ = false; }
    static std::string escape(const std::string& s) {
        std::string r;
        for (char c : s) {
            if (c == '"') r += "\\\"";
            else if (c == '\\') r += "\\\\";
            else if (c == '\n') r += "\\n";
            else r += c;
        }
        return r;
    }
public:
    void begin_object() { oss_ << "{\n"; first_ = true; }
    void end_object()   { oss_ << "\n}"; }
    void begin_array()  { oss_ << "[\n"; first_ = true; }
    void end_array()    { oss_ << "\n]"; }

    void field(const std::string& key, const std::string& val) {
        sep(); oss_ << "  \"" << key << "\": \"" << escape(val) << "\"";
    }
    void field(const std::string& key, int val) {
        sep(); oss_ << "  \"" << key << "\": " << val;
    }
    void field(const std::string& key, bool val) {
        sep(); oss_ << "  \"" << key << "\": " << (val ? "true" : "false");
    }

    std::string str() const { return oss_.str(); }
};

// ============================================================
//                      存储层
// ============================================================

class TodoStorage {
    std::string filepath_;

    static std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        std::ostringstream oss;
        oss << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M");
        return oss.str();
    }

    static std::string extract(const std::string& line, const std::string& key) {
        auto pos = line.find("\"" + key + "\":");
        if (pos == std::string::npos) return {};
        pos = line.find(':', pos) + 1;
        while (pos < line.size() && std::isspace(line[pos])) pos++;
        if (line[pos] == '"') {
            auto end = line.find('"', pos + 1);
            return line.substr(pos + 1, end - pos - 1);
        }
        // number or bool
        auto end = line.find_first_of(",}\n", pos);
        auto val = line.substr(pos, end - pos);
        while (!val.empty() && std::isspace(val.back())) val.pop_back();
        return val;
    }

public:
    explicit TodoStorage(std::string path) : filepath_(std::move(path)) {}

    std::vector<TodoItem> load() const {
        std::ifstream f(filepath_);
        if (!f.is_open()) return {};

        std::vector<TodoItem> items;
        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());

        // 简单解析：按 "id" 分割对象
        size_t pos = 0;
        while ((pos = content.find("\"id\":", pos)) != std::string::npos) {
            auto start = content.rfind('{', pos);
            auto end   = content.find('}', pos);
            if (start == std::string::npos || end == std::string::npos) break;

            std::string obj = content.substr(start, end - start + 1);
            TodoItem item;
            item.id         = std::stoi(extract(obj, "id"));
            item.title      = extract(obj, "title");
            item.completed  = extract(obj, "completed") == "true";
            item.priority   = TodoItem::priority_from_str(extract(obj, "priority"));
            item.created_at = extract(obj, "created_at");
            items.push_back(std::move(item));
            pos = end + 1;
        }
        return items;
    }

    void save(const std::vector<TodoItem>& items) const {
        std::ofstream f(filepath_);
        if (!f.is_open()) throw std::runtime_error("无法写入文件: " + filepath_);

        f << "[\n";
        for (size_t i = 0; i < items.size(); i++) {
            const auto& it = items[i];
            JsonBuilder jb;
            jb.begin_object();
            jb.field("id", it.id);
            jb.field("title", it.title);
            jb.field("completed", it.completed);
            jb.field("priority", TodoItem::priority_str(it.priority));
            jb.field("created_at", it.created_at);
            jb.end_object();
            f << jb.str();
            if (i + 1 < items.size()) f << ",";
            f << "\n";
        }
        f << "]\n";
    }

    static std::string now() { return get_timestamp(); }
};

// ============================================================
//                      业务逻辑
// ============================================================

class TodoService {
    std::vector<TodoItem> items_;
    TodoStorage           storage_;
    int                   next_id_ = 1;

public:
    explicit TodoService(const std::string& path) : storage_(path) {
        items_ = storage_.load();
        if (!items_.empty()) {
            next_id_ = 1 + std::ranges::max(items_, {}, &TodoItem::id).id;
        }
    }

    TodoItem& add(const std::string& title, Priority priority = Priority::Medium) {
        items_.push_back({next_id_++, title, false, priority, TodoStorage::now()});
        storage_.save(items_);
        return items_.back();
    }

    bool complete(int id) {
        auto it = std::ranges::find_if(items_, [id](const auto& t) { return t.id == id; });
        if (it == items_.end() || it->completed) return false;
        it->completed = true;
        storage_.save(items_);
        return true;
    }

    bool remove(int id) {
        auto it = std::ranges::find_if(items_, [id](const auto& t) { return t.id == id; });
        if (it == items_.end()) return false;
        items_.erase(it);
        storage_.save(items_);
        return true;
    }

    std::optional<TodoItem*> find(int id) {
        auto it = std::ranges::find_if(items_, [id](const auto& t) { return t.id == id; });
        if (it == items_.end()) return std::nullopt;
        return &(*it);
    }

    std::vector<TodoItem*> search(std::string_view keyword) {
        std::vector<TodoItem*> results;
        for (auto& item : items_) {
            if (item.title.find(keyword) != std::string::npos)
                results.push_back(&item);
        }
        return results;
    }

    const std::vector<TodoItem>& all() const { return items_; }

    std::vector<const TodoItem*> pending() const {
        std::vector<const TodoItem*> r;
        for (const auto& t : items_)
            if (!t.completed) r.push_back(&t);
        std::ranges::sort(r, [](const auto* a, const auto* b) {
            return static_cast<int>(a->priority) > static_cast<int>(b->priority);
        });
        return r;
    }

    struct Stats { int total, done, pending; double pct; };
    Stats stats() const {
        int done = static_cast<int>(std::ranges::count_if(items_, &TodoItem::completed));
        int total = static_cast<int>(items_.size());
        return {total, done, total - done, total ? 100.0 * done / total : 0.0};
    }
};

// ============================================================
//                      命令行界面
// ============================================================

class Cli {
    TodoService& svc_;

    static void print_line() { std::cout << std::string(58, '-') << "\n"; }

    void cmd_add(const std::vector<std::string>& args) {
        if (args.empty()) { std::cout << "用法: add <标题> [--priority high|medium|low]\n"; return; }
        std::string title = args[0];
        Priority prio = Priority::Medium;
        for (size_t i = 1; i + 1 < args.size(); i++) {
            if (args[i] == "--priority" || args[i] == "-p")
                prio = TodoItem::priority_from_str(args[++i]);
        }
        auto& item = svc_.add(title, prio);
        std::cout << "已添加: " << item.to_string() << "\n";
    }

    void cmd_list(const std::vector<std::string>& args) {
        std::string filter = args.empty() ? "all" : args[0];
        std::cout << "\n📋 Todo 列表";

        if (filter == "pending" || filter == "p") {
            auto items = svc_.pending();
            std::cout << "（待完成）\n"; print_line();
            for (const auto* t : items) std::cout << t->to_string() << "\n";
            std::cout << "共 " << items.size() << " 项\n";
        } else if (filter == "done" || filter == "completed") {
            std::cout << "（已完成）\n"; print_line();
            int cnt = 0;
            for (const auto& t : svc_.all())
                if (t.completed) { std::cout << t.to_string() << "\n"; cnt++; }
            std::cout << "共 " << cnt << " 项\n";
        } else {
            std::cout << "\n"; print_line();
            for (const auto& t : svc_.all()) std::cout << t.to_string() << "\n";
            auto s = svc_.stats();
            print_line();
            std::cout << "共 " << s.total << " 项 | 已完成 " << s.done
                      << " | 待完成 " << s.pending << " | 完成率 "
                      << std::fixed << std::setprecision(1) << s.pct << "%\n";
        }
    }

    void cmd_done(const std::vector<std::string>& args) {
        if (args.empty()) { std::cout << "用法: done <ID>\n"; return; }
        int id = std::stoi(args[0]);
        std::cout << (svc_.complete(id) ? "✓ 已完成 #" + args[0] : "未找到 #" + args[0]) << "\n";
    }

    void cmd_delete(const std::vector<std::string>& args) {
        if (args.empty()) { std::cout << "用法: delete <ID>\n"; return; }
        int id = std::stoi(args[0]);
        std::cout << (svc_.remove(id) ? "✗ 已删除 #" + args[0] : "未找到 #" + args[0]) << "\n";
    }

    void cmd_search(const std::vector<std::string>& args) {
        if (args.empty()) { std::cout << "用法: search <关键词>\n"; return; }
        auto results = svc_.search(args[0]);
        std::cout << "搜索 \"" << args[0] << "\" 结果 (" << results.size() << " 条):\n";
        for (const auto* t : results) std::cout << "  " << t->to_string() << "\n";
    }

    void cmd_stats() {
        auto s = svc_.stats();
        std::cout << "\n📊 统计\n";
        std::cout << "  总计:   " << s.total  << "\n";
        std::cout << "  已完成: " << s.done   << "\n";
        std::cout << "  待完成: " << s.pending << "\n";
        std::cout << "  完成率: " << std::fixed << std::setprecision(1) << s.pct << "%\n";
    }

    void print_help() {
        std::cout << R"(
Todo CLI — 现代 C++ 待办应用

命令:
  add <标题> [--priority high|medium|low]  添加任务
  list [all|pending|done]                  列出任务
  done <ID>                                标记完成
  delete <ID>                              删除任务
  search <关键词>                          搜索
  stats                                    统计
  help                                     帮助
)";
    }

public:
    explicit Cli(TodoService& svc) : svc_(svc) {}

    void run(const std::vector<std::string>& args) {
        if (args.empty()) { print_help(); return; }

        std::string cmd = args[0];
        std::vector<std::string> rest(args.begin() + 1, args.end());

        if      (cmd == "add")                cmd_add(rest);
        else if (cmd == "list" || cmd == "ls") cmd_list(rest);
        else if (cmd == "done")               cmd_done(rest);
        else if (cmd == "delete" || cmd == "rm") cmd_delete(rest);
        else if (cmd == "search")             cmd_search(rest);
        else if (cmd == "stats")              cmd_stats();
        else if (cmd == "help")               print_help();
        else std::cout << "未知命令: " << cmd << "\n";
    }
};

// ============================================================
//                      主函数
// ============================================================

int main(int argc, char* argv[]) {
    const std::string data_file = ".todo_cpp.json";

    try {
        TodoService svc(data_file);
        Cli cli(svc);

        if (argc <= 1) {
            // 演示模式
            std::cout << "=== C++ Todo CLI 演示 ===\n\n";

            cli.run({"add", "掌握现代 C++20 特性", "--priority", "high"});
            cli.run({"add", "学习 STL 容器与算法", "--priority", "high"});
            cli.run({"add", "理解智能指针与 RAII"});
            cli.run({"add", "实现线程池", "--priority", "low"});
            cli.run({"add", "阅读《Effective Modern C++》"});

            std::cout << "\n";
            cli.run({"list"});
            std::cout << "\n";

            cli.run({"done", "1"});
            cli.run({"done", "3"});
            std::cout << "\n";

            cli.run({"list", "pending"});
            std::cout << "\n";

            cli.run({"search", "C++"});
            std::cout << "\n";

            cli.run({"stats"});
            std::cout << "\n";

            cli.run({"delete", "4"});
            cli.run({"list"});

            // 清理演示文件
            std::remove(data_file.c_str());
        } else {
            std::vector<std::string> args(argv + 1, argv + argc);
            cli.run(args);
        }
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
```
