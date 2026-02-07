# calculator

::: info 文件信息
- 📄 原文件：`calculator.py`
- 🔤 语言：python
:::

计算器模块
这是被测试的代码，将在 calculator_test.py 中进行测试。

## 完整代码

```python
from typing import Union

Number = Union[int, float]


class Calculator:
    """简单计算器类"""

    def __init__(self):
        self.history = []

    def add(self, a: Number, b: Number) -> Number:
        """加法"""
        result = a + b
        self._record(f"{a} + {b} = {result}")
        return result

    def subtract(self, a: Number, b: Number) -> Number:
        """减法"""
        result = a - b
        self._record(f"{a} - {b} = {result}")
        return result

    def multiply(self, a: Number, b: Number) -> Number:
        """乘法"""
        result = a * b
        self._record(f"{a} * {b} = {result}")
        return result

    def divide(self, a: Number, b: Number) -> Number:
        """除法"""
        if b == 0:
            raise ValueError("除数不能为零")
        result = a / b
        self._record(f"{a} / {b} = {result}")
        return result

    def power(self, base: Number, exponent: Number) -> Number:
        """幂运算"""
        result = base ** exponent
        self._record(f"{base} ^ {exponent} = {result}")
        return result

    def _record(self, operation: str) -> None:
        """记录操作历史"""
        self.history.append(operation)

    def clear_history(self) -> None:
        """清空历史"""
        self.history.clear()

    def get_history(self) -> list:
        """获取历史"""
        return self.history.copy()


def factorial(n: int) -> int:
    """
    计算阶乘

    Args:
        n: 非负整数

    Returns:
        n 的阶乘

    Raises:
        ValueError: 当 n 为负数时
    """
    if n < 0:
        raise ValueError("n 必须是非负整数")
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def fibonacci(n: int) -> int:
    """
    计算斐波那契数列第 n 项

    Args:
        n: 非负整数

    Returns:
        第 n 个斐波那契数

    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(10)
        55
    """
    if n < 0:
        raise ValueError("n 必须是非负整数")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def is_prime(n: int) -> bool:
    """
    判断是否为素数

    Args:
        n: 正整数

    Returns:
        是否为素数

    Examples:
        >>> is_prime(2)
        True
        >>> is_prime(4)
        False
        >>> is_prime(17)
        True
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


if __name__ == "__main__":
    # 简单测试
    calc = Calculator()
    print(calc.add(10, 5))
    print(calc.divide(10, 3))
    print(calc.get_history())

    print(f"factorial(5) = {factorial(5)}")
    print(f"fibonacci(10) = {fibonacci(10)}")
    print(f"is_prime(17) = {is_prime(17)}")
```
