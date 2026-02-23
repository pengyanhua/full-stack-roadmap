# smart pointers.cpp

::: info 文件信息
- 📄 原文件：`01_smart_pointers.cpp`
- 🔤 语言：cpp
:::

## 完整代码

```cpp
// ============================================================
//                      智能指针与 RAII
// ============================================================
// RAII（Resource Acquisition Is Initialization）是 C++ 核心惯用法
// 智能指针自动管理内存，避免泄漏和悬空指针
// unique_ptr：独占所有权（零开销）
// shared_ptr：共享所有权（引用计数）
// weak_ptr：弱引用，不影响引用计数（解决循环引用）

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <functional>

// ============================================================
//                      演示类
// ============================================================

class Resource {
    std::string name_;
public:
    explicit Resource(const std::string& name) : name_(name) {
        std::cout << "  [构造] " << name_ << std::endl;
    }
    ~Resource() {
        std::cout << "  [析构] " << name_ << std::endl;
    }
    void use() const { std::cout << "  使用 " << name_ << std::endl; }
    const std::string& name() const { return name_; }
};

// ============================================================
//                      RAII 演示
// ============================================================

class FileHandle {
    FILE* fp_;
    std::string name_;
public:
    explicit FileHandle(const std::string& name, const char* mode)
        : fp_(fopen(name.c_str(), mode)), name_(name)
    {
        if (!fp_) throw std::runtime_error("无法打开文件: " + name);
        std::cout << "打开文件: " << name_ << std::endl;
    }

    ~FileHandle() {
        if (fp_) {
            fclose(fp_);
            fp_ = nullptr;
            std::cout << "关闭文件: " << name_ << std::endl;
        }
    }

    // 禁止拷贝（防止双重关闭）
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    // 允许移动
    FileHandle(FileHandle&& other) noexcept
        : fp_(other.fp_), name_(std::move(other.name_)) {
        other.fp_ = nullptr;
    }

    void write(const std::string& text) {
        if (fp_) fputs(text.c_str(), fp_);
    }

    bool is_open() const { return fp_ != nullptr; }
};

// ============================================================
//                      主函数
// ============================================================
int main() {
    // ============================================================
    //                      unique_ptr（独占所有权）
    // ============================================================
    std::cout << "=== unique_ptr ===" << std::endl;

    {
        auto p1 = std::make_unique<Resource>("UniqueA");  // 推荐方式
        p1->use();

        // 不能拷贝（独占所有权）
        // auto p2 = p1;  // 编译错误！

        // 可以移动（转移所有权）
        auto p2 = std::move(p1);  // p1 变为 nullptr
        std::cout << "移动后 p1=" << (p1 ? "有值" : "nullptr") << std::endl;
        p2->use();

        // 显式释放（通常不需要，let it go out of scope）
        p2.reset();
        std::cout << "reset 后 p2=" << (p2 ? "有值" : "nullptr") << std::endl;
    }
    std::cout << "--- unique_ptr 离开作用域，自动析构 ---" << std::endl;

    // unique_ptr 数组
    std::cout << "\n--- unique_ptr 数组 ---" << std::endl;
    auto arr = std::make_unique<int[]>(5);
    for (int i = 0; i < 5; i++) arr[i] = i * 10;
    for (int i = 0; i < 5; i++) std::cout << arr[i] << " ";
    std::cout << std::endl;

    // 自定义删除器
    std::cout << "\n--- 自定义删除器 ---" << std::endl;
    auto custom_deleter = [](Resource* p) {
        std::cout << "  自定义删除: " << p->name() << std::endl;
        delete p;
    };
    std::unique_ptr<Resource, decltype(custom_deleter)>
        p_custom(new Resource("CustomDeleted"), custom_deleter);

    // ============================================================
    //                      shared_ptr（共享所有权）
    // ============================================================
    std::cout << "\n=== shared_ptr ===" << std::endl;

    {
        auto sp1 = std::make_shared<Resource>("SharedB");
        std::cout << "引用计数: " << sp1.use_count() << std::endl;  // 1

        {
            auto sp2 = sp1;  // 拷贝：计数+1
            auto sp3 = sp1;  // 拷贝：计数+1
            std::cout << "引用计数: " << sp1.use_count() << std::endl;  // 3

            sp2->use();
            sp3->use();
        }  // sp2, sp3 析构，计数-2

        std::cout << "引用计数: " << sp1.use_count() << std::endl;  // 1
    }  // sp1 析构，计数归零，Resource 被释放
    std::cout << "--- shared_ptr 全部释放 ---" << std::endl;

    // shared_ptr 容器
    std::cout << "\n--- shared_ptr 容器 ---" << std::endl;
    std::vector<std::shared_ptr<Resource>> resources;
    resources.push_back(std::make_shared<Resource>("R1"));
    resources.push_back(std::make_shared<Resource>("R2"));
    resources.push_back(std::make_shared<Resource>("R3"));

    for (const auto& r : resources) r->use();
    std::cout << "容器清空前 R1 引用计数: " << resources[0].use_count() << std::endl;

    // ============================================================
    //                      weak_ptr（弱引用）
    // ============================================================
    std::cout << "\n=== weak_ptr（解决循环引用）===" << std::endl;

    // 循环引用问题：A 持有 B，B 持有 A -> 永不释放
    struct Node {
        std::string name;
        std::shared_ptr<Node> next;    // 强引用 -> 循环引用！
        std::weak_ptr<Node> prev;      // 弱引用 -> 打破循环

        Node(const std::string& n) : name(n) {
            std::cout << "  [构造] Node(" << name << ")" << std::endl;
        }
        ~Node() {
            std::cout << "  [析构] Node(" << name << ")" << std::endl;
        }
    };

    {
        auto n1 = std::make_shared<Node>("A");
        auto n2 = std::make_shared<Node>("B");

        n1->next = n2;   // A -> B（强引用）
        n2->prev = n1;   // B -> A（弱引用，不增加计数）

        std::cout << "n1 引用计数: " << n1.use_count() << std::endl;  // 1
        std::cout << "n2 引用计数: " << n2.use_count() << std::endl;  // 2 (n1->next + n2)

        // 使用 weak_ptr 需要先 lock
        if (auto locked = n2->prev.lock()) {  // lock() 返回 shared_ptr 或 nullptr
            std::cout << "n2.prev = " << locked->name << std::endl;
        }
    }  // n1, n2 正常析构（没有循环引用）
    std::cout << "--- Node 正常析构 ---" << std::endl;

    // ============================================================
    //                      RAII 文件句柄
    // ============================================================
    std::cout << "\n=== RAII 文件句柄 ===" << std::endl;

    try {
        {
            FileHandle fh("test_raii.tmp", "w");
            fh.write("RAII 测试\n");
            // fh.write 期间发生异常也能正确关闭文件
        }  // 自动关闭

        // 读取验证
        FileHandle fh2("test_raii.tmp", "r");
        char buf[128] = {};
        FILE* raw = nullptr;  // 这里只为演示，实际用 fh2 的方法

        remove("test_raii.tmp");  // 清理
    } catch (const std::exception& e) {
        std::cout << "异常: " << e.what() << std::endl;
    }

    // ============================================================
    //                      移动语义（Move Semantics）
    // ============================================================
    std::cout << "\n=== 移动语义 ===" << std::endl;

    class BigData {
        std::vector<int> data_;
    public:
        BigData(int size) : data_(size, 0) {
            std::cout << "  构造 BigData(size=" << data_.size() << ")" << std::endl;
        }
        BigData(const BigData& other) : data_(other.data_) {
            std::cout << "  拷贝 BigData" << std::endl;
        }
        BigData(BigData&& other) noexcept : data_(std::move(other.data_)) {
            std::cout << "  移动 BigData" << std::endl;
        }
        BigData& operator=(BigData&& other) noexcept {
            data_ = std::move(other.data_);
            return *this;
        }
        size_t size() const { return data_.size(); }
    };

    BigData bd1(1000000);
    BigData bd2 = std::move(bd1);      // 移动构造（零拷贝）
    std::cout << "bd1.size()=" << bd1.size() << "（已清空）" << std::endl;
    std::cout << "bd2.size()=" << bd2.size() << "（已转移）" << std::endl;

    std::cout << "\n=== 智能指针演示完成 ===" << std::endl;
    return 0;
}
```
