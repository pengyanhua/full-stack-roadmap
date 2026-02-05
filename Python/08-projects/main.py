#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
                Python 实战项目集合
============================================================
本目录包含多个实战项目，用于综合练习 Python 编程技能。

项目列表：
1. 01_todo_cli.py      - 命令行 Todo 应用
2. 02_word_counter.py  - 文本分析器（词频统计）
3. 03_simple_cache.py  - 线程安全的 LRU 缓存

每个项目都可以独立运行，也可以通过本文件选择运行。
============================================================
"""


def print_menu():
    """打印菜单"""
    print("""
============================================================
             欢迎来到 Python 实战项目集合
============================================================

请选择要运行的项目：

  1. Todo 应用      - 命令行待办事项管理
  2. 文本分析器    - 词频统计和文本分析
  3. LRU 缓存      - 线程安全的缓存实现

  q. 退出

============================================================
""")


def run_todo():
    """运行 Todo 应用"""
    from _01_todo_cli import main
    main()


def run_word_counter():
    """运行文本分析器"""
    from _02_word_counter import main
    main()


def run_cache():
    """运行缓存演示"""
    from _03_simple_cache import main
    main()


def main():
    """主函数"""
    projects = {
        '1': ('Todo 应用', run_todo),
        '2': ('文本分析器', run_word_counter),
        '3': ('LRU 缓存', run_cache),
    }

    while True:
        print_menu()
        choice = input("请输入选项 (1-3, q 退出): ").strip().lower()

        if choice == 'q' or choice == 'quit':
            print("\n再见！祝学习愉快！")
            break

        if choice in projects:
            name, func = projects[choice]
            print(f"\n正在启动: {name}\n")
            print("-" * 50)
            try:
                func()
            except KeyboardInterrupt:
                print("\n返回主菜单...")
            print("-" * 50)
        else:
            print("无效选项，请重新选择。")


if __name__ == "__main__":
    print("""
============================================================
                Python 实战项目说明
============================================================

直接运行各项目文件:
  python 01_todo_cli.py      # Todo 应用
  python 02_word_counter.py  # 文本分析器
  python 03_simple_cache.py  # LRU 缓存

每个项目都展示了不同的 Python 技能:

【01_todo_cli.py - Todo 应用】
  - 数据类 (dataclass)
  - JSON 序列化/反序列化
  - 文件操作
  - 命令行交互

【02_word_counter.py - 文本分析器】
  - 正则表达式
  - collections.Counter
  - 文本处理
  - 数据统计和可视化

【03_simple_cache.py - LRU 缓存】
  - 泛型 (Generic)
  - 多线程和锁
  - 装饰器
  - OrderedDict 数据结构

============================================================
""")

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n再见！")
