# helper

::: info 文件信息
- 📄 原文件：`helper.py`
- 🔤 语言：python
:::

通用辅助函数

## 完整代码

```python
from typing import TypeVar, List, Callable

T = TypeVar('T')


def add(a: int, b: int) -> int:
    """两数相加"""
    return a + b


def multiply(a: int, b: int) -> int:
    """两数相乘"""
    return a * b


def clamp(value: float, min_val: float, max_val: float) -> float:
    """将值限制在指定范围内"""
    return max(min_val, min(max_val, value))


def flatten(nested_list: List) -> List:
    """展平嵌套列表"""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def pipe(*functions: Callable) -> Callable:
    """
    函数管道，将多个函数组合成一个

    示例:
        add_one = lambda x: x + 1
        double = lambda x: x * 2
        pipeline = pipe(add_one, double)
        pipeline(5)  # (5 + 1) * 2 = 12
    """
    def inner(data):
        result = data
        for func in functions:
            result = func(result)
        return result
    return inner


if __name__ == "__main__":
    print(f"add(3, 5) = {add(3, 5)}")
    print(f"clamp(15, 0, 10) = {clamp(15, 0, 10)}")
    print(f"flatten([1, [2, 3], [4, [5, 6]]]) = {flatten([1, [2, 3], [4, [5, 6]]])}")

    add_one = lambda x: x + 1
    double = lambda x: x * 2
    pipeline = pipe(add_one, double)
    print(f"pipe(add_one, double)(5) = {pipeline(5)}")
```
