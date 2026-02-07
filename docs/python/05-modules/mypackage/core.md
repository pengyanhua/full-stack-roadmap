# core

::: info 文件信息
- 📄 原文件：`core.py`
- 🔤 语言：python
:::

mypackage 核心功能模块

## 完整代码

```python
def greet(name: str) -> str:
    """
    问候函数

    Args:
        name: 要问候的名字

    Returns:
        问候语字符串
    """
    return f"Hello, {name}!"


def calculate(a: int, b: int, operation: str = "add") -> int:
    """
    简单计算器

    Args:
        a: 第一个数
        b: 第二个数
        operation: 操作类型 (add, sub, mul, div)

    Returns:
        计算结果
    """
    operations = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: x / y if y != 0 else None,
    }

    func = operations.get(operation)
    if func is None:
        raise ValueError(f"未知操作: {operation}")

    return func(a, b)


# 模块级别的常量
PI = 3.14159
E = 2.71828


if __name__ == "__main__":
    # 模块测试代码
    print(greet("World"))
    print(calculate(10, 5, "add"))
    print(calculate(10, 5, "mul"))
```
