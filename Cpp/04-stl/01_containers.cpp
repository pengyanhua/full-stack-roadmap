// ============================================================
//                      STL 容器
// ============================================================
// STL（Standard Template Library）是 C++ 标准库的核心
// 序列容器：vector、deque、list、array
// 关联容器：map、set、multimap、multiset
// 无序容器：unordered_map、unordered_set（C++11）
// 容器适配器：stack、queue、priority_queue

#include <iostream>
#include <vector>
#include <deque>
#include <list>
#include <array>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <queue>
#include <string>
#include <algorithm>

// 打印容器（通用模板）
template <typename Container>
void print_container(const std::string& label, const Container& c) {
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
    // ============================================================
    //                      vector（动态数组）
    // ============================================================
    std::cout << "=== vector ===" << std::endl;

    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    v.push_back(7);          // 末尾插入 O(1)摊销
    v.insert(v.begin(), 0);  // 头部插入 O(n)
    v.erase(v.begin() + 3);  // 删除索引3 O(n)
    v.emplace_back(8);       // 原地构造（比 push_back 高效）

    print_container("vector", v);
    std::cout << "size=" << v.size() << ", capacity=" << v.capacity() << std::endl;
    std::cout << "front=" << v.front() << ", back=" << v.back() << std::endl;
    std::cout << "at(2)=" << v.at(2) << ", [2]=" << v[2] << std::endl;

    // 预分配容量（避免频繁扩容）
    v.reserve(100);
    std::cout << "reserve(100) 后 capacity=" << v.capacity() << std::endl;

    // 收缩到合适大小
    v.shrink_to_fit();
    std::cout << "shrink_to_fit 后 capacity=" << v.capacity() << std::endl;

    // 排序、查找
    std::sort(v.begin(), v.end());
    print_container("排序后", v);

    auto it = std::lower_bound(v.begin(), v.end(), 5);
    std::cout << "lower_bound(5): " << *it << std::endl;

    // 切片（使用迭代器）
    std::vector<int> slice(v.begin() + 2, v.begin() + 5);
    print_container("slice[2:5]", slice);

    // ============================================================
    //                      deque（双端队列）
    // ============================================================
    std::cout << "\n=== deque ===" << std::endl;

    std::deque<int> dq = {3, 4, 5};
    dq.push_front(2);   // 头部 O(1)
    dq.push_front(1);
    dq.push_back(6);    // 尾部 O(1)
    print_container("deque", dq);

    dq.pop_front();
    dq.pop_back();
    print_container("pop front/back", dq);

    // ============================================================
    //                      list（双向链表）
    // ============================================================
    std::cout << "\n=== list ===" << std::endl;

    std::list<int> lst = {1, 5, 3, 4, 2};
    lst.push_front(0);
    lst.push_back(6);

    // 任意位置 O(1) 插入（前提：已有迭代器）
    auto pos = std::find(lst.begin(), lst.end(), 3);
    lst.insert(pos, 99);  // 在 3 前插入 99

    print_container("list", lst);

    lst.sort();           // 链表的排序
    lst.unique();         // 删除相邻重复元素
    lst.reverse();        // 反转
    print_container("sort+unique+reverse", lst);

    // ============================================================
    //                      array（固定大小数组）
    // ============================================================
    std::cout << "\n=== array（固定大小）===" << std::endl;

    std::array<int, 5> arr = {5, 2, 8, 1, 9};
    print_container("array", arr);
    std::cout << "size=" << arr.size() << std::endl;

    std::sort(arr.begin(), arr.end());
    print_container("排序后", arr);

    // ============================================================
    //                      map（有序映射，红黑树）
    // ============================================================
    std::cout << "\n=== map ===" << std::endl;

    std::map<std::string, int> scores;
    scores["张三"] = 90;
    scores["李四"] = 85;
    scores["王五"] = 92;
    scores.emplace("赵六", 78);  // 原地构造

    // 遍历（自动按 key 排序）
    for (const auto& [name, score] : scores)  // C++17 结构化绑定
        std::cout << "  " << name << ": " << score << std::endl;

    // 查找
    auto it2 = scores.find("李四");
    if (it2 != scores.end())
        std::cout << "找到李四: " << it2->second << std::endl;

    // 安全访问（不存在时不插入）
    if (scores.count("钱七") == 0)
        std::cout << "钱七不存在" << std::endl;

    // at() vs []：at() 不存在时抛异常，[] 不存在时插入零值
    scores["新成员"];  // 插入默认值 0，小心！
    std::cout << "score.size()=" << scores.size() << std::endl;

    // ============================================================
    //                      unordered_map（哈希映射）
    // ============================================================
    std::cout << "\n=== unordered_map（O(1) 查找）===" << std::endl;

    std::unordered_map<std::string, int> freq;
    std::string words[] = {"apple", "banana", "apple", "cherry", "banana", "apple"};
    for (const auto& w : words) freq[w]++;

    for (const auto& [word, count] : freq)
        std::cout << "  " << word << ": " << count << std::endl;

    // ============================================================
    //                      set / unordered_set
    // ============================================================
    std::cout << "\n=== set ===" << std::endl;

    std::set<int> s = {5, 2, 8, 2, 1, 5, 9};  // 自动去重+排序
    print_container("set（去重排序）", s);

    s.insert(3);
    s.erase(2);
    print_container("insert(3), erase(2)", s);

    std::cout << "count(5)=" << s.count(5) << std::endl;

    // set 的集合运算
    std::set<int> s1 = {1, 2, 3, 4, 5};
    std::set<int> s2 = {3, 4, 5, 6, 7};
    std::vector<int> inter, diff;
    std::set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(), std::back_inserter(inter));
    std::set_difference(s1.begin(), s1.end(), s2.begin(), s2.end(), std::back_inserter(diff));
    print_container("交集", inter);
    print_container("差集(s1-s2)", diff);

    // ============================================================
    //                      容器适配器
    // ============================================================
    std::cout << "\n=== stack / queue / priority_queue ===" << std::endl;

    std::stack<int> stk;
    for (int i = 1; i <= 5; i++) stk.push(i);
    std::cout << "stack 出栈: ";
    while (!stk.empty()) { std::cout << stk.top() << " "; stk.pop(); }
    std::cout << "(LIFO)" << std::endl;

    std::queue<std::string> q;
    q.push("第一"); q.push("第二"); q.push("第三");
    std::cout << "queue 出队: ";
    while (!q.empty()) { std::cout << q.front() << " "; q.pop(); }
    std::cout << "(FIFO)" << std::endl;

    // priority_queue：最大堆
    std::priority_queue<int> pq;
    for (int x : {5, 2, 8, 1, 9, 3}) pq.push(x);
    std::cout << "priority_queue（最大堆）出队: ";
    while (!pq.empty()) { std::cout << pq.top() << " "; pq.pop(); }
    std::cout << std::endl;

    // 最小堆
    std::priority_queue<int, std::vector<int>, std::greater<int>> min_pq;
    for (int x : {5, 2, 8, 1, 9}) min_pq.push(x);
    std::cout << "priority_queue（最小堆）出队: ";
    while (!min_pq.empty()) { std::cout << min_pq.top() << " "; min_pq.pop(); }
    std::cout << std::endl;

    std::cout << "\n=== STL 容器演示完成 ===" << std::endl;
    return 0;
}
