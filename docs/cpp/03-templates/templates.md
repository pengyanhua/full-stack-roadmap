# templates.cpp

::: info 文件信息
- 📄 原文件：`01_templates.cpp`
- 🔤 语言：cpp
:::

## 完整代码

```cpp
// ============================================================
//                      模板编程
// ============================================================
// 模板是 C++ 泛型编程的基础，实现"写一次，适用多种类型"
// 函数模板：通用函数，类型由调用时推断或指定
// 类模板：通用容器/数据结构（STL 就是模板库）
// 模板特化：为特定类型提供专门实现

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <type_traits>  // C++11 类型特征

// ============================================================
//                      函数模板
// ============================================================

// ----------------------------------------------------------
// 1. 基本函数模板
// ----------------------------------------------------------
template <typename T>
T max_val(T a, T b) {
    return (a > b) ? a : b;
}

// 多类型参数
template <typename T, typename U>
auto add(T a, U b) -> decltype(a + b) {
    return a + b;
}

// 非类型模板参数
template <int N>
void print_n_times(const std::string& msg) {
    for (int i = 0; i < N; i++)
        std::cout << msg << " ";
    std::cout << std::endl;
}

// 模板函数特化（全特化：为特定类型提供专门实现）
template <>
const char* max_val(const char* a, const char* b) {
    return (std::string(a) > std::string(b)) ? a : b;
}

// ----------------------------------------------------------
// 2. 可变参数模板（C++11，Variadic Templates）
// ----------------------------------------------------------
// 递归终止（基本情况）
void print_all() {
    std::cout << std::endl;
}

// 递归展开
template <typename T, typename... Args>
void print_all(T first, Args... rest) {
    std::cout << first;
    if constexpr (sizeof...(rest) > 0)  // C++17 if constexpr
        std::cout << ", ";
    print_all(rest...);
}

// 折叠表达式（C++17，更简洁）
template <typename... Args>
auto sum_all(Args... args) {
    return (args + ...);  // 折叠表达式
}

template <typename... Args>
void print_types() {
    ((std::cout << typeid(Args).name() << " "), ...);
    std::cout << std::endl;
}

// ============================================================
//                      类模板
// ============================================================

// ----------------------------------------------------------
// 3. Stack 类模板
// ----------------------------------------------------------
template <typename T, size_t MaxSize = 64>
class Stack {
    T    data_[MaxSize];
    size_t size_ = 0;

public:
    void push(const T& val) {
        if (size_ >= MaxSize) throw std::overflow_error("栈满");
        data_[size_++] = val;
    }

    T pop() {
        if (size_ == 0) throw std::underflow_error("栈空");
        return data_[--size_];
    }

    const T& top() const {
        if (size_ == 0) throw std::underflow_error("栈空");
        return data_[size_ - 1];
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    // 模板成员函数
    template <typename U>
    void push_converted(const U& val) {
        push(static_cast<T>(val));
    }
};

// ----------------------------------------------------------
// 4. Pair 类模板
// ----------------------------------------------------------
template <typename First, typename Second>
struct Pair {
    First  first;
    Second second;

    Pair(First f, Second s) : first(std::move(f)), second(std::move(s)) {}

    bool operator==(const Pair& other) const {
        return first == other.first && second == other.second;
    }

    friend std::ostream& operator<<(std::ostream& os, const Pair& p) {
        return os << "(" << p.first << ", " << p.second << ")";
    }
};

// 工厂函数（C++11 之前常用，C++17 有类模板参数推断）
template <typename F, typename S>
Pair<F, S> make_pair_custom(F f, S s) {
    return Pair<F, S>(std::move(f), std::move(s));
}

// ----------------------------------------------------------
// 5. 类型特征与 SFINAE（C++11）
// ----------------------------------------------------------
// std::enable_if：根据类型条件启用/禁用模板

// 只对整数类型有效
template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
square_if_integral(T val) {
    return val * val;
}

// C++17 简写
template <typename T>
std::enable_if_t<std::is_floating_point_v<T>, T>
square_if_float(T val) {
    return val * val;
}

// ----------------------------------------------------------
// 6. 模板元编程（编译时计算）
// ----------------------------------------------------------
// 斐波那契数列（编译时计算）
template <int N>
struct Fibonacci {
    static constexpr int value = Fibonacci<N-1>::value + Fibonacci<N-2>::value;
};

template <> struct Fibonacci<0> { static constexpr int value = 0; };
template <> struct Fibonacci<1> { static constexpr int value = 1; };

// 阶乘（编译时计算）
template <int N>
struct Factorial {
    static constexpr long long value = N * Factorial<N-1>::value;
};
template <> struct Factorial<0> { static constexpr long long value = 1; };

// ============================================================
//                      主函数
// ============================================================
int main() {
    std::cout << "=== 函数模板 ===" << std::endl;

    // 类型推断
    std::cout << "max_val(3, 5) = " << max_val(3, 5) << std::endl;
    std::cout << "max_val(3.14, 2.71) = " << max_val(3.14, 2.71) << std::endl;
    std::cout << "max_val(\"apple\", \"banana\") = " << max_val(std::string("apple"), std::string("banana")) << std::endl;

    // 显式指定类型
    std::cout << "max_val<double>(3, 5.5) = " << max_val<double>(3, 5.5) << std::endl;

    // const char* 特化
    std::cout << "max_val(char*): " << max_val("apple", "banana") << std::endl;

    // 多类型参数
    auto r = add(3, 4.5);
    std::cout << "add(3, 4.5) = " << r << " (type: double)" << std::endl;

    // 非类型参数
    print_n_times<3>("hello");

    // ----------------------------------------------------------
    // 可变参数模板
    // ----------------------------------------------------------
    std::cout << "\n=== 可变参数模板 ===" << std::endl;

    print_all(1, "hello", 3.14, 'x');
    std::cout << "sum_all(1,2,3,4,5) = " << sum_all(1, 2, 3, 4, 5) << std::endl;
    std::cout << "sum_all(0.1, 0.2, 0.7) = " << sum_all(0.1, 0.2, 0.7) << std::endl;

    // ----------------------------------------------------------
    // 类模板
    // ----------------------------------------------------------
    std::cout << "\n=== 类模板 Stack ===" << std::endl;

    Stack<int> int_stack;
    Stack<std::string, 8> str_stack;

    for (int i = 1; i <= 5; i++) int_stack.push(i * 10);
    std::cout << "栈顶: " << int_stack.top() << std::endl;
    std::cout << "出栈: ";
    while (!int_stack.empty())
        std::cout << int_stack.pop() << " ";
    std::cout << std::endl;

    str_stack.push("Hello");
    str_stack.push("World");
    str_stack.push_converted(42);  // int -> string（但这会编译错误，改用数字）

    std::cout << "\n=== Pair 类模板 ===" << std::endl;

    auto p1 = make_pair_custom(42, std::string("hello"));
    auto p2 = make_pair_custom(3.14, true);
    std::cout << "p1 = " << p1 << std::endl;
    std::cout << "p2 = " << p2 << std::endl;

    // ----------------------------------------------------------
    // 类型特征
    // ----------------------------------------------------------
    std::cout << "\n=== 类型特征 ===" << std::endl;

    std::cout << "is_integral<int>: " << std::is_integral<int>::value << std::endl;
    std::cout << "is_floating_point<double>: " << std::is_floating_point<double>::value << std::endl;
    std::cout << "is_same<int,int>: " << std::is_same<int,int>::value << std::endl;

    std::cout << "square_if_integral(7) = " << square_if_integral(7) << std::endl;
    std::cout << "square_if_float(3.14) = " << square_if_float(3.14) << std::endl;

    // ----------------------------------------------------------
    // 模板元编程（编译时计算）
    // ----------------------------------------------------------
    std::cout << "\n=== 模板元编程 ===" << std::endl;

    std::cout << "Fibonacci<10> = " << Fibonacci<10>::value << std::endl;
    std::cout << "Fibonacci<20> = " << Fibonacci<20>::value << std::endl;

    std::cout << "Factorial<10> = " << Factorial<10>::value << std::endl;
    std::cout << "Factorial<15> = " << Factorial<15>::value << std::endl;

    // 编译时常量
    constexpr int fib_12 = Fibonacci<12>::value;
    constexpr long long fact_12 = Factorial<12>::value;
    static_assert(fib_12 == 144, "Fibonacci<12> 应为 144");
    std::cout << "编译时验证通过: Fib(12)=" << fib_12 << ", 12!=" << fact_12 << std::endl;

    std::cout << "\n=== 模板编程演示完成 ===" << std::endl;
    return 0;
}
```
