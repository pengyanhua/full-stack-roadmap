#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
                Python 模块与包
============================================================
本文件介绍 Python 中的模块和包的概念与使用。

模块（Module）：一个 .py 文件就是一个模块
包（Package）：包含 __init__.py 的目录
============================================================
"""
import sys
import os

# 将当前目录添加到路径（用于演示导入本地包）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main01_import_basics():
    """
    ============================================================
                    1. 导入基础
    ============================================================
    """
    print("=" * 60)
    print("1. 导入基础")
    print("=" * 60)

    # 【import 语句】
    import math
    print(f"import math: math.pi = {math.pi}")

    # 【from ... import ...】
    from math import sqrt, pow
    print(f"from math import sqrt: sqrt(16) = {sqrt(16)}")

    # 【别名 as】
    import math as m
    from math import factorial as fact
    print(f"import math as m: m.e = {m.e}")
    print(f"factorial as fact: fact(5) = {fact(5)}")

    # 【导入所有（不推荐）】
    # from math import *  # 会污染命名空间

    # 【导入检查】
    print(f"\n--- 导入检查 ---")
    print(f"math 模块路径: {math.__file__}")
    print(f"math 模块名: {math.__name__}")


def main02_module_search():
    """
    ============================================================
                    2. 模块搜索路径
    ============================================================
    """
    print("\n" + "=" * 60)
    print("2. 模块搜索路径")
    print("=" * 60)

    # 【sys.path】模块搜索路径列表
    print("模块搜索路径 (sys.path):")
    for i, path in enumerate(sys.path[:5]):
        print(f"  {i}: {path}")
    print("  ...")

    # 【搜索顺序】
    print(f"\n搜索顺序:")
    print("  1. 当前目录")
    print("  2. PYTHONPATH 环境变量")
    print("  3. 标准库目录")
    print("  4. site-packages（第三方包）")

    # 【动态添加路径】
    # sys.path.append('/custom/path')
    # sys.path.insert(0, '/priority/path')


def main03_package_structure():
    """
    ============================================================
                    3. 包结构
    ============================================================
    """
    print("\n" + "=" * 60)
    print("3. 包结构")
    print("=" * 60)

    print("""
    【包目录结构示例】

    mypackage/
    ├── __init__.py          # 包初始化文件
    ├── module1.py           # 模块1
    ├── module2.py           # 模块2
    └── subpackage/          # 子包
        ├── __init__.py
        └── module3.py

    【__init__.py 的作用】
    - 标识目录为 Python 包
    - 包初始化代码
    - 控制 from package import * 的行为
    - 定义包的公共接口
    """)

    # 【导入本地包示例】
    print("导入本地包:")
    try:
        from mypackage import greet
        from mypackage.utils import helper
        from mypackage.models import User

        print(f"  greet('World') = {greet('World')}")
        print(f"  helper.add(3, 5) = {helper.add(3, 5)}")
        user = User("Alice", 25)
        print(f"  User: {user}")
    except ImportError as e:
        print(f"  导入失败: {e}")
        print("  (需要创建 mypackage 目录)")


def main04_init_file():
    """
    ============================================================
                    4. __init__.py 详解
    ============================================================
    """
    print("\n" + "=" * 60)
    print("4. __init__.py 详解")
    print("=" * 60)

    print("""
    【__init__.py 的常见用法】

    1. 空文件（最简单）
       - 仅用于标识包

    2. 导入子模块
       from .module1 import func1
       from .module2 import Class2

    3. 定义 __all__
       __all__ = ['func1', 'Class2']
       # 控制 from package import * 的行为

    4. 包级别的变量和函数
       __version__ = '1.0.0'
       def package_func(): ...

    5. 延迟导入（优化启动时间）
       def __getattr__(name):
           if name == 'heavy_module':
               from . import heavy_module
               return heavy_module
           raise AttributeError
    """)


def main05_relative_import():
    """
    ============================================================
                    5. 相对导入与绝对导入
    ============================================================
    """
    print("\n" + "=" * 60)
    print("5. 相对导入与绝对导入")
    print("=" * 60)

    print("""
    【绝对导入】
    from mypackage.module1 import func
    import mypackage.subpackage.module3

    【相对导入】（只能在包内使用）
    from . import module1           # 当前包
    from .module1 import func       # 当前包的模块
    from .. import module2          # 父包
    from ..sibling import func      # 兄弟包

    【推荐】
    - 对于外部包：使用绝对导入
    - 对于包内模块：使用相对导入（更清晰）

    【注意】
    - 相对导入不能在顶层脚本中使用
    - 运行脚本时使用 python -m package.module
    """)


def main06_module_attributes():
    """
    ============================================================
                    6. 模块特殊属性
    ============================================================
    """
    print("\n" + "=" * 60)
    print("6. 模块特殊属性")
    print("=" * 60)

    import math

    print("模块特殊属性:")
    print(f"  __name__: {math.__name__}")
    print(f"  __file__: {math.__file__}")
    print(f"  __doc__: {math.__doc__[:50]}...")
    print(f"  __package__: {math.__package__}")

    # 【__name__ 的特殊用法】
    print(f"\n--- __name__ 的用法 ---")
    print(f"当前模块的 __name__: {__name__}")
    print("""
    if __name__ == '__main__':
        # 只在直接运行时执行
        main()

    这个模式用于:
    1. 区分直接运行和被导入
    2. 编写可测试的模块
    3. 提供命令行入口
    """)


def main07_lazy_import():
    """
    ============================================================
                    7. 延迟导入与导入优化
    ============================================================
    """
    print("\n" + "=" * 60)
    print("7. 延迟导入与导入优化")
    print("=" * 60)

    # 【延迟导入】在需要时才导入
    print("--- 延迟导入 ---")

    def process_json(data):
        import json  # 只在调用时导入
        return json.loads(data)

    result = process_json('{"key": "value"}')
    print(f"延迟导入 json: {result}")

    # 【条件导入】
    print(f"\n--- 条件导入 ---")

    try:
        import ujson as json
        print("使用 ujson（更快）")
    except ImportError:
        import json
        print("使用标准 json")

    # 【importlib 动态导入】
    print(f"\n--- 动态导入 ---")
    import importlib

    module_name = "math"
    math_module = importlib.import_module(module_name)
    print(f"动态导入 {module_name}: pi = {math_module.pi}")

    # 重新加载模块
    # importlib.reload(math_module)


def main08_namespace_packages():
    """
    ============================================================
                8. 命名空间包
    ============================================================
    """
    print("\n" + "=" * 60)
    print("8. 命名空间包")
    print("=" * 60)

    print("""
    【命名空间包】Python 3.3+
    - 不需要 __init__.py
    - 可以跨多个目录分布
    - 用于大型项目或插件系统

    【目录结构示例】
    path1/
    └── myns/
        └── package1/
            └── __init__.py

    path2/
    └── myns/
        └── package2/
            └── __init__.py

    # myns 是命名空间包，可以同时导入:
    from myns import package1
    from myns import package2

    【适用场景】
    - 插件系统
    - 多团队协作的大型项目
    - 第三方扩展
    """)


def main09_best_practices():
    """
    ============================================================
                    9. 最佳实践
    ============================================================
    """
    print("\n" + "=" * 60)
    print("9. 最佳实践")
    print("=" * 60)

    print("""
    【导入顺序】（PEP 8）
    1. 标准库导入
    2. 第三方库导入
    3. 本地模块导入
    每组之间空一行

    【示例】
    import os
    import sys

    import numpy as np
    import pandas as pd

    from mypackage import mymodule

    【命名规范】
    - 模块名：小写，下划线分隔（my_module.py）
    - 包名：小写，尽量不用下划线（mypackage）
    - 避免与标准库同名

    【避免循环导入】
    - 重构代码结构
    - 使用延迟导入
    - 将导入放在函数内部

    【其他建议】
    - 优先使用绝对导入
    - 避免 from module import *
    - 使用 __all__ 控制公共接口
    - 编写 __init__.py 提供简洁的 API
    """)


if __name__ == "__main__":
    main01_import_basics()
    main02_module_search()
    main03_package_structure()
    main04_init_file()
    main05_relative_import()
    main06_module_attributes()
    main07_lazy_import()
    main08_namespace_packages()
    main09_best_practices()
