// ============================================================
//                      现代 C++（C++11/14/17/20）
// ============================================================
// C++11 是 C++ 的重大革新："新语言"
// C++14/17 对 C++11 的完善和补充
// C++20 引入概念（Concepts）、协程、范围（Ranges）
// 推荐始终使用最新标准：-std=c++20

#include <iostream>
#include <string>
#include <vector>
#include <optional>      // C++17
#include <variant>       // C++17
#include <any>           // C++17
#include <string_view>   // C++17
#include <filesystem>    // C++17
#include <span>          // C++20
#include <ranges>        // C++20
#include <concepts>      // C++20
#include <format>        // C++20（需要 GCC 13 / Clang 16 / MSVC 2022）
#include <algorithm>
#include <numeric>
#include <map>
#include <tuple>
#include <chrono>

namespace fs = std::filesystem;

// ============================================================
//                      C++11 特性
// ============================================================

// ----------------------------------------------------------
// Lambda 表达式（C++11）
// ----------------------------------------------------------
void demo_lambda() {
    std::cout << "=== Lambda（C++11）===" << std::endl;

    // 基本 lambda
    auto greet = [](const std::string& name) {
        return "你好，" + name + "！";
    };
    std::cout << greet("世界") << std::endl;

    // 捕获列表
    int base = 10;
    std::vector<int> v = {1, 2, 3, 4, 5};

    // [=] 捕获所有局部变量（值）
    auto by_value = [=](int x) { return x + base; };

    // [&] 捕获所有局部变量（引用）
    auto by_ref = [&](int x) { base += x; return base; };

    // 指定捕获
    auto specific = [base, &v](int x) {
        v.push_back(x);
        return base + x;
    };

    std::cout << "捕获值: " << by_value(5) << std::endl;
    std::cout << "捕获引用: " << by_ref(5) << std::endl;  // base 变为 15
    specific(99);
    std::cout << "v.back() = " << v.back() << std::endl;

    // 泛型 lambda（C++14）
    auto add = [](auto a, auto b) { return a + b; };
    std::cout << "泛型 add(3, 4) = " << add(3, 4) << std::endl;
    std::cout << "泛型 add(1.5, 2.5) = " << add(1.5, 2.5) << std::endl;

    // 立即调用 lambda（IIFE）
    int result = [](int x, int y) { return x * y; }(6, 7);
    std::cout << "IIFE: " << result << std::endl;
}

// ----------------------------------------------------------
// 结构化绑定（C++17）
// ----------------------------------------------------------
void demo_structured_bindings() {
    std::cout << "\n=== 结构化绑定（C++17）===" << std::endl;

    // pair
    auto p = std::make_pair("张三", 90);
    auto [name, score] = p;
    std::cout << name << ": " << score << std::endl;

    // tuple
    auto t = std::make_tuple("李四", 25, 3.8);
    auto [n, age, gpa] = t;
    std::cout << n << " (" << age << "岁, GPA " << gpa << ")" << std::endl;

    // map 遍历
    std::map<std::string, int> scores = {{"Alice", 95}, {"Bob", 87}, {"Charlie", 92}};
    for (auto& [k, v] : scores) {
        std::cout << "  " << k << ": " << v << std::endl;
    }

    // 数组
    int arr[] = {10, 20, 30};
    auto [x, y, z] = arr;
    std::cout << "数组: " << x << ", " << y << ", " << z << std::endl;
}

// ============================================================
//                      C++17 特性
// ============================================================

// ----------------------------------------------------------
// optional（可能无值）
// ----------------------------------------------------------
std::optional<int> safe_divide(int a, int b) {
    if (b == 0) return std::nullopt;
    return a / b;
}

std::optional<std::string> find_user(int id) {
    std::map<int, std::string> users = {{1, "张三"}, {2, "李四"}, {3, "王五"}};
    auto it = users.find(id);
    if (it == users.end()) return {};
    return it->second;
}

void demo_optional() {
    std::cout << "\n=== optional（C++17）===" << std::endl;

    auto r1 = safe_divide(10, 3);
    auto r2 = safe_divide(10, 0);

    std::cout << "10/3: ";
    if (r1) std::cout << *r1 << std::endl;
    else    std::cout << "无值" << std::endl;

    std::cout << "10/0: " << r2.value_or(-1) << "（默认-1）" << std::endl;

    // monadic 操作（C++23，这里用 value_or 模拟）
    auto user = find_user(2).value_or("未知用户");
    std::cout << "用户2: " << user << std::endl;
    std::cout << "用户99: " << find_user(99).value_or("未找到") << std::endl;
}

// ----------------------------------------------------------
// variant（类型安全的 union）
// ----------------------------------------------------------
using JsonValue = std::variant<int, double, std::string, bool, std::nullptr_t>;

std::string json_to_string(const JsonValue& v) {
    return std::visit([](const auto& val) -> std::string {
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, int>)         return std::to_string(val);
        else if constexpr (std::is_same_v<T, double>)  return std::to_string(val);
        else if constexpr (std::is_same_v<T, std::string>) return "\"" + val + "\"";
        else if constexpr (std::is_same_v<T, bool>)    return val ? "true" : "false";
        else                                            return "null";
    }, v);
}

void demo_variant() {
    std::cout << "\n=== variant（C++17）===" << std::endl;

    JsonValue v1 = 42;
    JsonValue v2 = 3.14;
    JsonValue v3 = std::string("hello");
    JsonValue v4 = true;
    JsonValue v5 = nullptr;

    for (auto& v : {v1, v2, v3, v4, v5}) {
        std::cout << "  " << json_to_string(v) << std::endl;
    }

    // holds_alternative
    std::cout << "v1 是 int: " << std::holds_alternative<int>(v1) << std::endl;

    // get / get_if
    if (auto* s = std::get_if<std::string>(&v3)) {
        std::cout << "字符串长度: " << s->size() << std::endl;
    }
}

// ----------------------------------------------------------
// string_view（轻量字符串视图，零拷贝）
// ----------------------------------------------------------
void demo_string_view() {
    std::cout << "\n=== string_view（C++17）===" << std::endl;

    // string_view 不拥有数据，只是视图
    std::string_view sv1 = "Hello, World!";
    std::string s = "C++ Programming";
    std::string_view sv2 = s;

    // 子串（零拷贝！）
    std::string_view sub = sv1.substr(7, 5);
    std::cout << "substr: " << sub << std::endl;

    // 查找
    auto pos = sv1.find("World");
    std::cout << "find: " << pos << std::endl;

    // 函数接受 string_view（兼容 string 和字面量，零拷贝）
    auto count_vowels = [](std::string_view text) {
        return std::count_if(text.begin(), text.end(),
                             [](char c) { return std::string("aeiouAEIOU").find(c) != std::string::npos; });
    };
    std::cout << "元音数: " << count_vowels(sv1) << std::endl;
    std::cout << "长度: " << sv1.size() << std::endl;
}

// ============================================================
//                      C++20 特性
// ============================================================

// ----------------------------------------------------------
// Concepts（概念约束）
// ----------------------------------------------------------
template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

template <typename T>
concept Printable = requires(T t) {
    { std::cout << t } -> std::same_as<std::ostream&>;
};

template <Numeric T>
T safe_sqrt(T val) {
    if (val < 0) throw std::domain_error("负数不能开平方");
    return static_cast<T>(std::sqrt(static_cast<double>(val)));
}

template <Printable T>
void print_val(const T& val) {
    std::cout << "  值: " << val << std::endl;
}

void demo_concepts() {
    std::cout << "\n=== Concepts（C++20）===" << std::endl;

    std::cout << "safe_sqrt(16) = " << safe_sqrt(16) << std::endl;
    std::cout << "safe_sqrt(2.0) = " << safe_sqrt(2.0) << std::endl;
    // safe_sqrt("abc");  // 编译错误：不满足 Numeric 约束

    print_val(42);
    print_val(std::string("hello"));
    print_val(3.14);
}

// ----------------------------------------------------------
// Ranges（范围，C++20）
// ----------------------------------------------------------
void demo_ranges() {
    std::cout << "\n=== Ranges（C++20）===" << std::endl;

    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // 管道操作（类似 UNIX 管道）
    auto result = v
        | std::views::filter([](int x) { return x % 2 == 0; })
        | std::views::transform([](int x) { return x * x; })
        | std::views::take(3);

    std::cout << "偶数平方（前3）: ";
    for (int x : result) std::cout << x << " ";
    std::cout << std::endl;

    // iota view（生成序列）
    auto seq = std::views::iota(1, 11)
             | std::views::filter([](int x) { return x % 3 == 0; });
    std::cout << "1-10 中 3 的倍数: ";
    for (int x : seq) std::cout << x << " ";
    std::cout << std::endl;

    // 范围算法
    std::ranges::sort(v, std::greater<int>{});
    std::cout << "ranges::sort 降序: ";
    for (int x : v) std::cout << x << " ";
    std::cout << std::endl;
}

// ----------------------------------------------------------
// chrono（时间库，C++11/20）
// ----------------------------------------------------------
void demo_chrono() {
    std::cout << "\n=== chrono ===" << std::endl;

    using namespace std::chrono;
    using namespace std::chrono_literals;  // 1s, 100ms, etc.

    auto start = high_resolution_clock::now();

    // 模拟工作
    volatile long long sum = 0;
    for (int i = 0; i < 10000000; i++) sum += i;

    auto end = high_resolution_clock::now();
    auto elapsed = duration_cast<microseconds>(end - start);

    std::cout << "计算耗时: " << elapsed.count() << " μs" << std::endl;
    std::cout << "sum = " << sum << std::endl;

    // 时间字面量
    auto timeout = 500ms;
    std::cout << "超时设置: " << timeout.count() << " ms" << std::endl;

    // system_clock（获取当前时间）
    auto now = system_clock::now();
    auto now_t = system_clock::to_time_t(now);
    std::cout << "当前时间戳: " << now_t << std::endl;
}

// ============================================================
//                      主函数
// ============================================================
int main() {
    demo_lambda();
    demo_structured_bindings();
    demo_optional();
    demo_variant();
    demo_string_view();
    demo_concepts();
    demo_ranges();
    demo_chrono();

    // ----------------------------------------------------------
    // if constexpr（C++17，编译时分支）
    // ----------------------------------------------------------
    std::cout << "\n=== if constexpr（C++17）===" << std::endl;

    auto describe = []<typename T>(T val) {
        if constexpr (std::is_integral_v<T>)
            std::cout << "整数: " << val << std::endl;
        else if constexpr (std::is_floating_point_v<T>)
            std::cout << "浮点: " << val << std::endl;
        else
            std::cout << "其他: " << val << std::endl;
    };

    describe(42);
    describe(3.14);
    describe(std::string("hello"));

    // ----------------------------------------------------------
    // 初始化语句 in if/switch（C++17）
    // ----------------------------------------------------------
    std::cout << "\n=== if 初始化语句（C++17）===" << std::endl;

    std::map<std::string, int> m = {{"a", 1}, {"b", 2}};
    if (auto it = m.find("b"); it != m.end()) {
        std::cout << "找到: " << it->first << "=" << it->second << std::endl;
    }

    std::cout << "\n=== 现代 C++ 演示完成 ===" << std::endl;
    return 0;
}
