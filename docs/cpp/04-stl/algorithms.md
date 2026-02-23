# algorithms.cpp

::: info 文件信息
- 📄 原文件：`02_algorithms.cpp`
- 🔤 语言：cpp
:::

## 完整代码

```cpp
// ============================================================
//                      STL 算法与迭代器
// ============================================================
// STL 算法在迭代器上工作，与容器解耦
// 分类：查找、排序、修改、数值、集合操作
// Lambda + 算法 = 强大的函数式编程能力

#include <iostream>
#include <vector>
#include <algorithm>  // 主要算法
#include <numeric>    // 数值算法
#include <functional> // std::greater, std::bind
#include <iterator>   // 迭代器适配器
#include <string>
#include <map>
#include <random>

template <typename C>
void print(const std::string& label, const C& c) {
    std::cout << label << ": [";
    bool first = true;
    for (const auto& v : c) {
        if (!first) std::cout << ", ";
        std::cout << v;
        first = false;
    }
    std::cout << "]" << std::endl;
}

int main() {
    std::vector<int> v = {5, 2, 8, 1, 9, 3, 7, 4, 6};

    // ============================================================
    //                      查找算法
    // ============================================================
    std::cout << "=== 查找算法 ===" << std::endl;

    // find：线性查找
    auto it = std::find(v.begin(), v.end(), 7);
    if (it != v.end())
        std::cout << "find(7): 位置=" << std::distance(v.begin(), it) << std::endl;

    // find_if：条件查找
    auto even = std::find_if(v.begin(), v.end(), [](int x) { return x % 2 == 0; });
    std::cout << "第一个偶数: " << *even << std::endl;

    // count / count_if
    std::cout << "count(1)=" << std::count(v.begin(), v.end(), 1) << std::endl;
    int odd_count = std::count_if(v.begin(), v.end(), [](int x) { return x % 2 != 0; });
    std::cout << "奇数个数=" << odd_count << std::endl;

    // all_of / any_of / none_of
    std::cout << "all_of(>0)=" << std::all_of(v.begin(), v.end(), [](int x) { return x > 0; }) << std::endl;
    std::cout << "any_of(>8)=" << std::any_of(v.begin(), v.end(), [](int x) { return x > 8; }) << std::endl;
    std::cout << "none_of(<0)=" << std::none_of(v.begin(), v.end(), [](int x) { return x < 0; }) << std::endl;

    // min_element / max_element
    auto [min_it, max_it] = std::minmax_element(v.begin(), v.end());
    std::cout << "min=" << *min_it << ", max=" << *max_it << std::endl;

    // ============================================================
    //                      排序算法
    // ============================================================
    std::cout << "\n=== 排序算法 ===" << std::endl;

    std::vector<int> sv = v;  // 副本用于排序

    // sort：不稳定排序（快速排序）
    std::sort(sv.begin(), sv.end());
    print("sort 升序", sv);

    std::sort(sv.begin(), sv.end(), std::greater<int>());
    print("sort 降序", sv);

    // 自定义比较器
    std::vector<std::pair<std::string, int>> students = {
        {"张三", 90}, {"李四", 85}, {"王五", 90}, {"赵六", 75}
    };
    std::sort(students.begin(), students.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;  // 分数降序
        return a.first < b.first;  // 同分按姓名升序
    });
    std::cout << "按分数排序:" << std::endl;
    for (const auto& [name, score] : students)
        std::cout << "  " << name << ": " << score << std::endl;

    // stable_sort：稳定排序（保持相等元素的原始顺序）
    std::stable_sort(students.begin(), students.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });

    // partial_sort：只排序前 k 个
    sv = v;
    std::partial_sort(sv.begin(), sv.begin() + 3, sv.end());
    print("partial_sort 最小3个", sv);

    // nth_element：将第 n 小的元素放到正确位置
    sv = v;
    std::nth_element(sv.begin(), sv.begin() + 4, sv.end());
    std::cout << "第5小的元素: " << sv[4] << std::endl;

    // 二分查找（需有序）
    sv = v; std::sort(sv.begin(), sv.end());
    std::cout << "binary_search(5): " << std::binary_search(sv.begin(), sv.end(), 5) << std::endl;
    auto lb = std::lower_bound(sv.begin(), sv.end(), 5);
    auto ub = std::upper_bound(sv.begin(), sv.end(), 5);
    std::cout << "lower_bound(5)=" << *lb << ", upper_bound(5)=" << *ub << std::endl;

    // ============================================================
    //                      修改算法
    // ============================================================
    std::cout << "\n=== 修改算法 ===" << std::endl;

    sv = v;

    // transform：对每个元素应用函数
    std::vector<int> squares(sv.size());
    std::transform(sv.begin(), sv.end(), squares.begin(), [](int x) { return x * x; });
    print("transform 平方", squares);

    // 两个区间的 transform
    std::vector<int> a = {1, 2, 3}, b = {4, 5, 6};
    std::vector<int> c(3);
    std::transform(a.begin(), a.end(), b.begin(), c.begin(), std::plus<int>());
    print("两向量相加", c);

    // for_each：对每个元素执行操作
    std::cout << "for_each: ";
    std::for_each(sv.begin(), sv.end(), [](int x) { std::cout << x*2 << " "; });
    std::cout << std::endl;

    // fill / fill_n
    std::vector<int> filled(5);
    std::fill(filled.begin(), filled.end(), 42);
    print("fill(42)", filled);

    // generate：用函数生成元素
    int gen_val = 0;
    std::generate(filled.begin(), filled.end(), [&]() { return gen_val += 10; });
    print("generate", filled);

    // replace / replace_if
    sv = v;
    std::replace(sv.begin(), sv.end(), 5, 50);
    print("replace(5->50)", sv);

    std::replace_if(sv.begin(), sv.end(), [](int x) { return x > 40; }, -1);
    print("replace_if(>40 -> -1)", sv);

    // remove / remove_if（返回新的逻辑末尾，不缩小容器）
    sv = v;
    auto new_end = std::remove_if(sv.begin(), sv.end(), [](int x) { return x % 2 == 0; });
    sv.erase(new_end, sv.end());  // erase-remove 惯用法
    print("删除偶数", sv);

    // unique：删除相邻重复（先 sort 再 unique 去全部重复）
    sv = {1, 1, 2, 2, 3, 1, 1};
    sv.erase(std::unique(sv.begin(), sv.end()), sv.end());
    print("unique（相邻去重）", sv);

    // reverse / rotate
    sv = v;
    std::reverse(sv.begin(), sv.end());
    print("reverse", sv);

    std::rotate(sv.begin(), sv.begin() + 3, sv.end());
    print("rotate(3)", sv);

    // shuffle：随机打乱
    sv = v;
    std::mt19937 rng(42);  // 固定种子，可重复
    std::shuffle(sv.begin(), sv.end(), rng);
    print("shuffle", sv);

    // ============================================================
    //                      数值算法（numeric）
    // ============================================================
    std::cout << "\n=== 数值算法 ===" << std::endl;

    sv = {1, 2, 3, 4, 5};

    // accumulate：累积（求和、求积等）
    int sum = std::accumulate(sv.begin(), sv.end(), 0);
    int product = std::accumulate(sv.begin(), sv.end(), 1, std::multiplies<int>());
    std::cout << "sum=" << sum << ", product=" << product << std::endl;

    // reduce（C++17，可并行）
    int sum2 = std::reduce(sv.begin(), sv.end(), 0);
    std::cout << "reduce sum=" << sum2 << std::endl;

    // inner_product：内积（点积）
    std::vector<int> weights = {2, 3, 1, 4, 5};
    int dot = std::inner_product(sv.begin(), sv.end(), weights.begin(), 0);
    std::cout << "inner_product=" << dot << std::endl;

    // iota：生成连续序列
    std::vector<int> iota_v(10);
    std::iota(iota_v.begin(), iota_v.end(), 1);  // 1, 2, ..., 10
    print("iota", iota_v);

    // partial_sum：前缀和
    std::vector<int> prefix(sv.size());
    std::partial_sum(sv.begin(), sv.end(), prefix.begin());
    print("partial_sum", prefix);

    // adjacent_difference：相邻差
    std::vector<int> diff(sv.size());
    std::adjacent_difference(sv.begin(), sv.end(), diff.begin());
    print("adjacent_difference", diff);

    std::cout << "\n=== STL 算法演示完成 ===" << std::endl;
    return 0;
}
```
