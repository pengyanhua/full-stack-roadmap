// ============================================================
//                      引用、命名空间与基础特性
// ============================================================
// C++ 在 C 基础上增加了引用、命名空间、重载、内联等特性
// 引用（Reference）是变量的别名，比指针更安全易用
// 命名空间（Namespace）解决大型项目中的命名冲突

#include <iostream>
#include <string>

// ============================================================
//                      命名空间
// ============================================================

// 定义命名空间
namespace math {
    const double PI = 3.14159265358979;

    double circle_area(double r) {
        return PI * r * r;
    }

    namespace geometry {   // 嵌套命名空间
        double distance(double x1, double y1, double x2, double y2) {
            double dx = x1 - x2, dy = y1 - y2;
            return std::sqrt(dx*dx + dy*dy);
        }
    }
}

namespace utils {
    std::string repeat(const std::string& s, int n) {
        std::string result;
        for (int i = 0; i < n; i++) result += s;
        return result;
    }
}

int main() {
    // ============================================================
    //                      引用（Reference）
    // ============================================================
    std::cout << "=== 引用（Reference）===" << std::endl;

    int x = 42;
    int &ref = x;  // ref 是 x 的别名，必须在声明时初始化

    std::cout << "x = " << x << std::endl;
    std::cout << "ref = " << ref << "（同一变量）" << std::endl;
    std::cout << "&x = " << &x << "（相同地址）" << std::endl;
    std::cout << "&ref = " << &ref << std::endl;

    ref = 100;  // 修改 ref 等于修改 x
    std::cout << "ref=100 后，x = " << x << std::endl;

    // ----------------------------------------------------------
    // 引用 vs 指针
    // ----------------------------------------------------------
    std::cout << "\n--- 引用 vs 指针 ---" << std::endl;

    int a = 10, b = 20;

    // 引用：不能为 null，不能重新绑定，更安全
    int &ra = a;
    // ra = &b;   // 错误！引用不能重新绑定

    // 指针：可以为 null，可以重新指向，更灵活
    int *pa = &a;
    pa = &b;  // 可以：指针可以指向新地址
    *pa = 200;
    std::cout << "指针修改 b = " << b << std::endl;

    // ----------------------------------------------------------
    // 函数参数中的引用（避免拷贝）
    // ----------------------------------------------------------
    std::cout << "\n--- 函数引用参数 ---" << std::endl;

    // 值传递：拷贝，修改不影响原变量
    auto swap_by_value = [](int x, int y) {
        int tmp = x; x = y; y = tmp;
    };

    // 引用传递：别名，修改原变量
    auto swap_by_ref = [](int &x, int &y) {
        int tmp = x; x = y; y = tmp;
    };

    // const 引用：只读访问，避免拷贝（最常用的函数参数形式）
    auto print_str = [](const std::string &s) {
        std::cout << "  长字符串长度: " << s.size() << std::endl;
    };

    int m = 5, n = 10;
    swap_by_value(m, n);
    std::cout << "值传递后: m=" << m << ", n=" << n << "（不变）" << std::endl;
    swap_by_ref(m, n);
    std::cout << "引用传递后: m=" << m << ", n=" << n << "（已交换）" << std::endl;

    std::string long_str(1000, 'x');
    print_str(long_str);  // const& 避免拷贝 1000 字节

    // ----------------------------------------------------------
    // 右值引用（C++11）
    // ----------------------------------------------------------
    std::cout << "\n--- 右值引用（C++11）---" << std::endl;

    int &&rref = 42;       // 右值引用绑定临时值
    int &&rref2 = 3 + 5;   // 绑定表达式结果
    std::cout << "右值引用: " << rref << ", " << rref2 << std::endl;

    // 移动语义：std::move 将左值转为右值引用
    std::string s1 = "Hello";
    std::string s2 = std::move(s1);  // s1 资源转移到 s2，s1 变空
    std::cout << "move 后 s2=" << s2 << "，s1='" << s1 << "'（已清空）" << std::endl;

    // ============================================================
    //                      命名空间使用
    // ============================================================
    std::cout << "\n=== 命名空间 ===" << std::endl;

    // 使用全限定名
    std::cout << "math::PI = " << math::PI << std::endl;
    std::cout << "circle_area(5) = " << math::circle_area(5) << std::endl;
    std::cout << "distance = " << math::geometry::distance(0,0,3,4) << std::endl;

    // using 声明（引入单个名字）
    using math::PI;
    std::cout << "PI = " << PI << std::endl;

    // using namespace（引入所有名字，谨慎使用）
    using namespace utils;
    std::cout << repeat("★", 5) << std::endl;

    // ============================================================
    //                      C++ 基础特性
    // ============================================================
    std::cout << "\n=== C++ 基础增强 ===" << std::endl;

    // ----------------------------------------------------------
    // auto 类型推断（C++11）
    // ----------------------------------------------------------
    auto i = 42;              // int
    auto d = 3.14;            // double
    auto s = std::string("hello");  // std::string
    auto lambda = [](int x) { return x * 2; };

    std::cout << "auto int: " << i << std::endl;
    std::cout << "auto double: " << d << std::endl;
    std::cout << "auto lambda(5): " << lambda(5) << std::endl;

    // ----------------------------------------------------------
    // 范围 for 循环（C++11）
    // ----------------------------------------------------------
    std::cout << "\n--- 范围 for ---" << std::endl;

    int arr[] = {1, 2, 3, 4, 5};

    std::cout << "值遍历: ";
    for (int v : arr) std::cout << v << " ";
    std::cout << std::endl;

    // 引用遍历（避免拷贝，可修改）
    for (int &v : arr) v *= 2;
    std::cout << "×2 后: ";
    for (const int &v : arr) std::cout << v << " ";
    std::cout << std::endl;

    // ----------------------------------------------------------
    // nullptr（C++11，替代 NULL）
    // ----------------------------------------------------------
    int *ptr = nullptr;  // 类型安全的空指针
    std::cout << "\nnullptr == NULL: " << (ptr == NULL) << std::endl;

    // ----------------------------------------------------------
    // 统一初始化语法（C++11）
    // ----------------------------------------------------------
    std::cout << "\n--- 统一初始化 ---" << std::endl;

    int x1{42};          // 初始化器列表
    double d1{3.14};
    std::string str1{"Hello"};

    // 防止窄化转换
    // int narrow{3.14};  // 错误！双精度不能窄化到 int

    std::cout << "x1=" << x1 << ", d1=" << d1 << ", str1=" << str1 << std::endl;

    // ----------------------------------------------------------
    // decltype（C++11）
    // ----------------------------------------------------------
    std::cout << "\n--- decltype ---" << std::endl;

    int val = 10;
    decltype(val) val2 = 20;         // val2 类型同 val（int）
    decltype(val + 3.0) val3 = 5.5;  // val3 类型是 double
    std::cout << "decltype: " << val2 << ", " << val3 << std::endl;

    std::cout << "\n=== 基础特性演示完成 ===" << std::endl;
    return 0;
}
