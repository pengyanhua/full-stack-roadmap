#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
                    Python 测试教程
============================================================
本文件演示 Python 中的各种测试方法和技术。

运行测试的方法：
- pytest test_calculator.py
- pytest test_calculator.py -v  # 详细输出
- pytest test_calculator.py::TestCalculator  # 指定类
- pytest test_calculator.py::test_add  # 指定函数
- python -m pytest  # 作为模块运行
============================================================
"""
import pytest
from calculator import Calculator, factorial, fibonacci, is_prime


# ============================================================
#                    1. 基础测试
# ============================================================

def test_add():
    """
    【基础测试】
    测试函数名必须以 test_ 开头
    使用 assert 语句进行断言
    """
    calc = Calculator()
    assert calc.add(2, 3) == 5
    assert calc.add(-1, 1) == 0
    assert calc.add(0, 0) == 0


def test_subtract():
    """测试减法"""
    calc = Calculator()
    assert calc.subtract(5, 3) == 2
    assert calc.subtract(3, 5) == -2


def test_multiply():
    """测试乘法"""
    calc = Calculator()
    assert calc.multiply(3, 4) == 12
    assert calc.multiply(-2, 3) == -6
    assert calc.multiply(0, 100) == 0


def test_divide():
    """测试除法"""
    calc = Calculator()
    assert calc.divide(10, 2) == 5
    assert calc.divide(7, 2) == 3.5


# ============================================================
#                    2. 测试异常
# ============================================================

def test_divide_by_zero():
    """
    【测试异常】
    使用 pytest.raises 捕获预期的异常
    """
    calc = Calculator()
    with pytest.raises(ValueError) as exc_info:
        calc.divide(10, 0)

    # 可以检查异常信息
    assert "除数不能为零" in str(exc_info.value)


def test_factorial_negative():
    """测试负数阶乘抛出异常"""
    with pytest.raises(ValueError):
        factorial(-1)


# ============================================================
#                    3. 测试类
# ============================================================

class TestCalculator:
    """
    【测试类】
    测试类名必须以 Test 开头
    方法名必须以 test_ 开头
    """

    def setup_method(self):
        """
        每个测试方法执行前运行
        用于设置测试环境
        """
        self.calc = Calculator()

    def teardown_method(self):
        """
        每个测试方法执行后运行
        用于清理测试环境
        """
        self.calc.clear_history()

    def test_add(self):
        assert self.calc.add(1, 2) == 3

    def test_history(self):
        """测试历史记录"""
        self.calc.add(1, 2)
        self.calc.multiply(3, 4)
        history = self.calc.get_history()
        assert len(history) == 2
        assert "1 + 2 = 3" in history[0]


# ============================================================
#                    4. 参数化测试
# ============================================================

@pytest.mark.parametrize("a, b, expected", [
    (2, 3, 5),
    (0, 0, 0),
    (-1, 1, 0),
    (100, 200, 300),
    (1.5, 2.5, 4.0),
])
def test_add_parametrized(a, b, expected):
    """
    【参数化测试】
    使用 @pytest.mark.parametrize 运行多组测试数据
    """
    calc = Calculator()
    assert calc.add(a, b) == expected


@pytest.mark.parametrize("n, expected", [
    (0, 1),
    (1, 1),
    (5, 120),
    (10, 3628800),
])
def test_factorial_parametrized(n, expected):
    """参数化测试阶乘"""
    assert factorial(n) == expected


@pytest.mark.parametrize("n, expected", [
    (0, 0),
    (1, 1),
    (2, 1),
    (10, 55),
    (20, 6765),
])
def test_fibonacci_parametrized(n, expected):
    """参数化测试斐波那契"""
    assert fibonacci(n) == expected


@pytest.mark.parametrize("n, expected", [
    (2, True),
    (3, True),
    (4, False),
    (17, True),
    (18, False),
    (97, True),
    (100, False),
])
def test_is_prime_parametrized(n, expected):
    """参数化测试素数判断"""
    assert is_prime(n) == expected


# ============================================================
#                    5. Fixture（测试夹具）
# ============================================================

@pytest.fixture
def calculator():
    """
    【Fixture】
    提供测试所需的对象或数据
    使用 @pytest.fixture 装饰器
    """
    return Calculator()


@pytest.fixture
def calculator_with_history():
    """带历史记录的计算器"""
    calc = Calculator()
    calc.add(1, 2)
    calc.multiply(3, 4)
    return calc


def test_with_fixture(calculator):
    """使用 fixture 的测试"""
    assert calculator.add(5, 3) == 8


def test_history_length(calculator_with_history):
    """测试历史记录长度"""
    assert len(calculator_with_history.get_history()) == 2


# Fixture 作用域
@pytest.fixture(scope="module")
def shared_calculator():
    """
    【Fixture 作用域】
    - function: 每个测试函数创建新实例（默认）
    - class: 每个测试类创建一次
    - module: 每个模块创建一次
    - session: 整个测试会话创建一次
    """
    print("\n创建共享计算器")
    return Calculator()


def test_shared_1(shared_calculator):
    shared_calculator.add(1, 1)


def test_shared_2(shared_calculator):
    # 注意：这里的历史包含上一个测试的记录
    assert len(shared_calculator.get_history()) >= 1


# ============================================================
#                    6. 跳过和标记测试
# ============================================================

@pytest.mark.skip(reason="功能未实现")
def test_not_implemented():
    """跳过测试"""
    pass


@pytest.mark.skipif(True, reason="条件跳过示例")
def test_skip_conditionally():
    """条件跳过"""
    pass


@pytest.mark.slow
def test_slow_operation():
    """
    【自定义标记】
    运行标记的测试：pytest -m slow
    排除标记的测试：pytest -m "not slow"
    """
    import time
    time.sleep(0.1)
    assert True


@pytest.mark.xfail(reason="已知问题")
def test_known_failure():
    """
    【预期失败】
    测试预期会失败，如果通过则显示 XPASS
    """
    assert False


# ============================================================
#                    7. 近似比较
# ============================================================

def test_float_comparison():
    """
    【浮点数比较】
    使用 pytest.approx 进行近似比较
    """
    calc = Calculator()
    result = calc.divide(10, 3)
    assert result == pytest.approx(3.333, rel=1e-2)  # 相对误差 1%
    assert result == pytest.approx(3.333, abs=0.01)  # 绝对误差 0.01


def test_list_approx():
    """列表近似比较"""
    expected = [0.1 + 0.2, 0.3 + 0.4]
    actual = [0.3, 0.7]
    assert actual == pytest.approx(expected)


# ============================================================
#                    8. Mock 模拟
# ============================================================

from unittest.mock import Mock, patch, MagicMock


def test_mock_basic():
    """
    【Mock 基础】
    创建模拟对象
    """
    mock_calc = Mock()
    mock_calc.add.return_value = 10

    result = mock_calc.add(3, 5)
    assert result == 10

    # 验证调用
    mock_calc.add.assert_called_once_with(3, 5)


def test_mock_side_effect():
    """Mock 副作用"""
    mock_calc = Mock()
    mock_calc.divide.side_effect = ValueError("除数为零")

    with pytest.raises(ValueError):
        mock_calc.divide(10, 0)


def test_patch_decorator():
    """
    【Patch 装饰器】
    临时替换模块中的对象
    """
    with patch('calculator.Calculator') as MockCalculator:
        instance = MockCalculator.return_value
        instance.add.return_value = 100

        calc = MockCalculator()
        result = calc.add(1, 2)

        assert result == 100


# ============================================================
#                    9. 测试覆盖率
# ============================================================
"""
【测试覆盖率】

安装: pip install pytest-cov

运行: pytest --cov=calculator test_calculator.py
详细: pytest --cov=calculator --cov-report=html

覆盖率报告会显示：
- 语句覆盖率
- 分支覆盖率
- 未覆盖的代码行
"""


# ============================================================
#                    10. conftest.py
# ============================================================
"""
【conftest.py】

conftest.py 是 pytest 的配置文件，用于：
- 定义共享的 fixture
- 定义 pytest 插件
- 配置测试行为

示例 conftest.py:

```python
import pytest

@pytest.fixture(scope="session")
def db_connection():
    # 创建数据库连接
    conn = create_connection()
    yield conn
    conn.close()

def pytest_configure(config):
    # 添加自定义标记
    config.addinivalue_line("markers", "slow: 慢速测试")
```
"""


# ============================================================
#                    11. pytest.ini 配置
# ============================================================
"""
【pytest.ini】

pytest.ini 或 pyproject.toml 配置示例：

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
python_classes = Test*
addopts = -v --tb=short
markers =
    slow: 慢速测试
    integration: 集成测试
```

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
markers = [
    "slow: 慢速测试",
    "integration: 集成测试",
]
```
"""


if __name__ == "__main__":
    # 直接运行此文件
    pytest.main([__file__, "-v"])
