// ============================================================
//                      类与面向对象
// ============================================================
// C++ 类：封装数据和行为，支持访问控制
// 构造/析构：对象生命周期管理（RAII 的基础）
// 运算符重载：让自定义类型像内置类型一样使用
// this 指针：指向当前对象的指针

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>

// ============================================================
//                      类定义示例
// ============================================================

// ----------------------------------------------------------
// 1. 基本类（封装）
// ----------------------------------------------------------
class BankAccount {
private:
    // 私有成员：只能通过 public 接口访问
    std::string owner_;
    double      balance_;
    int         transaction_count_;

public:
    // 构造函数
    BankAccount(const std::string& owner, double initial_balance = 0.0)
        : owner_(owner), balance_(initial_balance), transaction_count_(0)
    {
        if (initial_balance < 0)
            throw std::invalid_argument("初始余额不能为负");
        std::cout << "账户创建: " << owner_ << std::endl;
    }

    // 析构函数（对象销毁时调用）
    ~BankAccount() {
        std::cout << "账户关闭: " << owner_
                  << "（余额: " << balance_ << "）" << std::endl;
    }

    // 拷贝构造函数
    BankAccount(const BankAccount& other)
        : owner_(other.owner_ + "_copy")
        , balance_(other.balance_)
        , transaction_count_(0)
    {
        std::cout << "账户复制: " << owner_ << std::endl;
    }

    // 成员函数
    void deposit(double amount) {
        if (amount <= 0) throw std::invalid_argument("存款金额必须为正");
        balance_ += amount;
        transaction_count_++;
        std::cout << "存入 " << amount << "，余额: " << balance_ << std::endl;
    }

    bool withdraw(double amount) {
        if (amount <= 0) return false;
        if (amount > balance_) {
            std::cout << "余额不足（余额: " << balance_ << "）" << std::endl;
            return false;
        }
        balance_ -= amount;
        transaction_count_++;
        std::cout << "取出 " << amount << "，余额: " << balance_ << std::endl;
        return true;
    }

    // const 成员函数（不修改对象状态）
    double balance() const { return balance_; }
    const std::string& owner() const { return owner_; }
    int transaction_count() const { return transaction_count_; }

    // 友元函数（可访问私有成员）
    friend std::ostream& operator<<(std::ostream& os, const BankAccount& acc);
};

// 运算符重载（输出流）
std::ostream& operator<<(std::ostream& os, const BankAccount& acc) {
    return os << "账户[" << acc.owner_ << ": ¥" << acc.balance_ << "]";
}

// ----------------------------------------------------------
// 2. 静态成员
// ----------------------------------------------------------
class Counter {
private:
    static int total_count_;  // 静态成员：所有实例共享
    int id_;

public:
    Counter() : id_(++total_count_) {
        std::cout << "创建第 " << id_ << " 个 Counter\n";
    }
    ~Counter() { total_count_--; }

    static int total() { return total_count_; }
    int id() const { return id_; }
};

int Counter::total_count_ = 0;  // 静态成员在类外定义

// ----------------------------------------------------------
// 3. 运算符重载
// ----------------------------------------------------------
class Vector2D {
public:
    double x, y;

    Vector2D(double x = 0, double y = 0) : x(x), y(y) {}

    // 加法运算符
    Vector2D operator+(const Vector2D& other) const {
        return Vector2D(x + other.x, y + other.y);
    }

    // 减法运算符
    Vector2D operator-(const Vector2D& other) const {
        return Vector2D(x - other.x, y - other.y);
    }

    // 标量乘法
    Vector2D operator*(double scalar) const {
        return Vector2D(x * scalar, y * scalar);
    }

    // 复合赋值
    Vector2D& operator+=(const Vector2D& other) {
        x += other.x; y += other.y;
        return *this;  // 返回 *this 支持链式调用
    }

    // 相等比较
    bool operator==(const Vector2D& other) const {
        return x == other.x && y == other.y;
    }

    // 下标运算符
    double& operator[](int i) {
        if (i == 0) return x;
        if (i == 1) return y;
        throw std::out_of_range("索引越界");
    }

    // 方法
    double length() const { return std::sqrt(x*x + y*y); }
    Vector2D normalize() const {
        double len = length();
        return (len > 0) ? Vector2D(x/len, y/len) : Vector2D(0, 0);
    }

    // 输出流运算符（友元）
    friend std::ostream& operator<<(std::ostream& os, const Vector2D& v) {
        return os << "(" << v.x << ", " << v.y << ")";
    }
};

// 非成员标量乘法（支持 3.0 * v 顺序）
Vector2D operator*(double scalar, const Vector2D& v) {
    return v * scalar;
}

// ----------------------------------------------------------
// 4. 不可变类（const 设计）
// ----------------------------------------------------------
class ImmutablePoint {
    const double x_, y_;
public:
    ImmutablePoint(double x, double y) : x_(x), y_(y) {}
    double x() const { return x_; }
    double y() const { return y_; }
    double dist_to(const ImmutablePoint& other) const {
        double dx = x_ - other.x_, dy = y_ - other.y_;
        return std::sqrt(dx*dx + dy*dy);
    }
    friend std::ostream& operator<<(std::ostream& os, const ImmutablePoint& p) {
        return os << "(" << p.x_ << ", " << p.y_ << ")";
    }
};

// ============================================================
//                      主函数
// ============================================================
int main() {
    std::cout << "=== 类与封装 ===" << std::endl;

    // 基本类使用
    {
        BankAccount acc("张三", 1000.0);
        acc.deposit(500.0);
        acc.withdraw(200.0);
        acc.withdraw(2000.0);  // 余额不足

        std::cout << acc << std::endl;
        std::cout << "交易次数: " << acc.transaction_count() << std::endl;

        // 拷贝构造
        BankAccount acc2 = acc;
        acc2.deposit(100.0);
        std::cout << "原账户余额: " << acc.balance() << std::endl;
        std::cout << "副本余额:   " << acc2.balance() << std::endl;
    }  // acc, acc2 在此处析构

    // ----------------------------------------------------------
    // 静态成员
    // ----------------------------------------------------------
    std::cout << "\n=== 静态成员 ===" << std::endl;
    std::cout << "初始计数: " << Counter::total() << std::endl;
    {
        Counter c1, c2, c3;
        std::cout << "创建3个后: " << Counter::total() << std::endl;
    }
    std::cout << "销毁后: " << Counter::total() << std::endl;

    // ----------------------------------------------------------
    // 运算符重载
    // ----------------------------------------------------------
    std::cout << "\n=== 运算符重载 ===" << std::endl;

    Vector2D v1(3.0, 4.0);
    Vector2D v2(1.0, 2.0);

    std::cout << "v1 = " << v1 << std::endl;
    std::cout << "v2 = " << v2 << std::endl;
    std::cout << "v1 + v2 = " << (v1 + v2) << std::endl;
    std::cout << "v1 - v2 = " << (v1 - v2) << std::endl;
    std::cout << "v1 * 2.0 = " << (v1 * 2.0) << std::endl;
    std::cout << "3.0 * v2 = " << (3.0 * v2) << std::endl;
    std::cout << "|v1| = " << v1.length() << std::endl;
    std::cout << "v1 归一化 = " << v1.normalize() << std::endl;

    v1 += v2;
    std::cout << "v1 += v2 -> " << v1 << std::endl;

    std::cout << "v1[0]=" << v1[0] << ", v1[1]=" << v1[1] << std::endl;

    // 链式调用（每次 += 返回 *this）
    Vector2D v3;
    v3 += Vector2D(1,1);
    v3 += Vector2D(2,2);
    std::cout << "链式 += : " << v3 << std::endl;

    // ----------------------------------------------------------
    // 不可变类
    // ----------------------------------------------------------
    std::cout << "\n=== 不可变类 ===" << std::endl;

    ImmutablePoint p1(0, 0), p2(3, 4);
    std::cout << "p1 = " << p1 << std::endl;
    std::cout << "p2 = " << p2 << std::endl;
    std::cout << "距离 = " << p1.dist_to(p2) << std::endl;

    std::cout << "\n=== 类与面向对象演示完成 ===" << std::endl;
    return 0;
}
