// ============================================================
//                      继承与多态
// ============================================================
// C++ 继承：public/protected/private 三种方式
// 虚函数（virtual）实现运行时多态
// 纯虚函数（pure virtual）定义抽象接口
// 虚析构函数：多态基类必须有虚析构函数

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cmath>

// ============================================================
//                      抽象基类（接口）
// ============================================================

// ----------------------------------------------------------
// 1. 抽象类（含纯虚函数）
// ----------------------------------------------------------
class Shape {
protected:
    std::string color_;

public:
    explicit Shape(const std::string& color = "黑色") : color_(color) {}

    // 纯虚函数：子类必须实现（= 0）
    virtual double area() const = 0;
    virtual double perimeter() const = 0;

    // 虚函数：子类可以重写
    virtual std::string name() const { return "形状"; }
    virtual void describe() const {
        std::cout << name() << "（" << color_ << "）: "
                  << "面积=" << area() << ", 周长=" << perimeter() << std::endl;
    }

    // 【重要】基类必须有虚析构函数，否则通过基类指针删除子类时内存泄漏
    virtual ~Shape() = default;
};

// ----------------------------------------------------------
// 2. 具体子类
// ----------------------------------------------------------
class Circle : public Shape {
    double radius_;
public:
    Circle(double radius, const std::string& color = "红色")
        : Shape(color), radius_(radius) {}

    double area() const override {
        return M_PI * radius_ * radius_;
    }
    double perimeter() const override {
        return 2 * M_PI * radius_;
    }
    std::string name() const override { return "圆形"; }
    double radius() const { return radius_; }
};

class Rectangle : public Shape {
    double width_, height_;
public:
    Rectangle(double w, double h, const std::string& color = "蓝色")
        : Shape(color), width_(w), height_(h) {}

    double area() const override { return width_ * height_; }
    double perimeter() const override { return 2 * (width_ + height_); }
    std::string name() const override { return "矩形"; }
};

class Triangle : public Shape {
    double a_, b_, c_;  // 三边长
public:
    Triangle(double a, double b, double c, const std::string& color = "绿色")
        : Shape(color), a_(a), b_(b), c_(c) {}

    double perimeter() const override { return a_ + b_ + c_; }
    double area() const override {
        double s = perimeter() / 2;
        return std::sqrt(s * (s-a_) * (s-b_) * (s-c_));  // 海伦公式
    }
    std::string name() const override { return "三角形"; }
};

// ----------------------------------------------------------
// 3. 多层继承
// ----------------------------------------------------------
class Animal {
protected:
    std::string name_;
    int age_;

public:
    Animal(const std::string& name, int age) : name_(name), age_(age) {}

    virtual std::string speak() const = 0;
    virtual std::string type() const { return "动物"; }

    std::string name() const { return name_; }
    int age() const { return age_; }

    virtual void describe() const {
        std::cout << type() << " " << name_ << "（" << age_ << "岁）: "
                  << speak() << std::endl;
    }

    virtual ~Animal() = default;
};

class Pet : public Animal {
    std::string owner_;
public:
    Pet(const std::string& name, int age, const std::string& owner)
        : Animal(name, age), owner_(owner) {}

    std::string type() const override { return "宠物"; }
    const std::string& owner() const { return owner_; }
};

class Dog : public Pet {
public:
    Dog(const std::string& name, int age, const std::string& owner)
        : Pet(name, age, owner) {}

    std::string speak() const override { return "汪汪！"; }
    std::string type() const override { return "狗"; }
    void fetch() const { std::cout << name_ << " 正在捡球！" << std::endl; }
};

class Cat : public Pet {
public:
    Cat(const std::string& name, int age, const std::string& owner)
        : Pet(name, age, owner) {}

    std::string speak() const override { return "喵喵~"; }
    std::string type() const override { return "猫"; }
    void purr() const { std::cout << name_ << " 正在呼噜噜..." << std::endl; }
};

// ----------------------------------------------------------
// 4. 多重继承
// ----------------------------------------------------------
class Flyable {
public:
    virtual void fly() const = 0;
    virtual ~Flyable() = default;
};

class Swimmable {
public:
    virtual void swim() const = 0;
    virtual ~Swimmable() = default;
};

class Duck : public Animal, public Flyable, public Swimmable {
public:
    Duck(const std::string& name) : Animal(name, 2) {}

    std::string speak() const override { return "嘎嘎！"; }
    std::string type() const override { return "鸭子"; }
    void fly() const override { std::cout << name_ << " 在飞！" << std::endl; }
    void swim() const override { std::cout << name_ << " 在游泳！" << std::endl; }
};

// ============================================================
//                      主函数
// ============================================================
int main() {
    std::cout << "=== 多态（Shape）===" << std::endl;

    // 多态：基类指针/引用指向子类对象
    std::vector<std::unique_ptr<Shape>> shapes;
    shapes.push_back(std::make_unique<Circle>(5.0));
    shapes.push_back(std::make_unique<Rectangle>(4.0, 6.0));
    shapes.push_back(std::make_unique<Triangle>(3.0, 4.0, 5.0));
    shapes.push_back(std::make_unique<Circle>(3.0, "绿色"));

    // 运行时多态：根据实际类型调用正确的方法
    for (const auto& s : shapes) {
        s->describe();
    }

    // 计算总面积
    double total_area = 0;
    for (const auto& s : shapes) total_area += s->area();
    std::cout << "总面积: " << total_area << std::endl;

    // dynamic_cast：安全的向下转型
    std::cout << "\n=== dynamic_cast ===" << std::endl;
    for (const auto& s : shapes) {
        if (auto *circle = dynamic_cast<Circle*>(s.get())) {
            std::cout << "圆形半径: " << circle->radius() << std::endl;
        }
    }

    // ----------------------------------------------------------
    // 多层继承与多态
    // ----------------------------------------------------------
    std::cout << "\n=== 动物多态 ===" << std::endl;

    std::vector<std::unique_ptr<Animal>> animals;
    animals.push_back(std::make_unique<Dog>("旺财", 3, "张三"));
    animals.push_back(std::make_unique<Cat>("咪咪", 2, "李四"));
    animals.push_back(std::make_unique<Duck>("唐老鸭"));

    for (const auto& a : animals) {
        a->describe();
    }

    // 向下转型
    for (const auto& a : animals) {
        if (auto *dog = dynamic_cast<Dog*>(a.get())) {
            dog->fetch();
        } else if (auto *cat = dynamic_cast<Cat*>(a.get())) {
            cat->purr();
        }
    }

    // ----------------------------------------------------------
    // 多重继承
    // ----------------------------------------------------------
    std::cout << "\n=== 多重继承（Duck）===" << std::endl;

    Duck duck("Donald");
    duck.describe();
    duck.fly();
    duck.swim();

    // 通过接口指针使用
    Flyable* flyer = &duck;
    flyer->fly();

    Swimmable* swimmer = &duck;
    swimmer->swim();

    // ----------------------------------------------------------
    // override 和 final 关键字（C++11）
    // ----------------------------------------------------------
    std::cout << "\n=== override / final ===" << std::endl;
    std::cout << "override 防止拼写错误导致的意外函数隐藏" << std::endl;
    std::cout << "final 阻止类被继承或方法被重写" << std::endl;

    // class FinalClass final : public Shape { ... };  // 不能被继承
    // virtual double area() const final override;     // 不能被重写

    std::cout << "\n=== 继承与多态演示完成 ===" << std::endl;
    return 0;
}
