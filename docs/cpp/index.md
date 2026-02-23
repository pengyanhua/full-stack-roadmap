# C++

C++ 是在 C 语言基础上构建的多范式编程语言，兼具底层控制能力与高级抽象特性。它被广泛用于游戏引擎、高频交易、编译器、数据库和操作系统等对性能要求极高的领域。

## 为什么学 C++？

- **零开销抽象**：模板、内联、constexpr 让高层抽象无运行时代价
- **现代特性**：C++11/14/17/20 带来智能指针、lambda、ranges、concepts
- **工业级生态**：Qt、Boost、Eigen、TensorFlow 底层都是 C++
- **理解语言设计**：学完 C++ 再看 Rust / Go，会有豁然开朗的感觉

## 学习路径

| 章节 | 内容 | 核心知识点 |
|------|------|-----------|
| 01 - 基础 | 引用、命名空间、auto | 左值/右值引用、移动语义、结构化绑定 |
| 02 - 面向对象 | 类与继承 | 封装、多态、虚函数、运算符重载 |
| 03 - 模板 | 泛型编程 | 函数/类模板、可变参数、折叠表达式、SFINAE |
| 04 - STL | 容器与算法 | vector/map/set、sort/transform/ranges |
| 05 - 内存管理 | 智能指针与 RAII | unique_ptr、shared_ptr、weak_ptr |
| 06 - 现代特性 | C++11~C++20 | optional、variant、string_view、Concepts |
| 07 - 并发编程 | 线程与同步 | thread、mutex、future、atomic、线程池 |
| 08 - 项目实战 | Todo CLI（现代 C++20） | ranges、optional、自定义 JSON 序列化 |

## 开始学习

从 [01 - 基础](/cpp/01-basics/references_namespaces) 开始，体验现代 C++ 的优雅之处。

::: tip 学习建议
使用支持 C++20 的编译器：
```bash
g++ -std=c++20 -Wall -O2 -o demo demo.cpp && ./demo
# 或使用 clang++
clang++ -std=c++20 -Wall -O2 -o demo demo.cpp
```
:::
