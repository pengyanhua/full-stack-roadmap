# __init__

::: info 文件信息
- 📄 原文件：`__init__.py`
- 🔤 语言：python
:::

mypackage 包
这是一个示例包，展示 Python 包的结构和用法。

包的公共接口通过 __init__.py 暴露。

## 完整代码

```python
# 包版本
__version__ = "1.0.0"
__author__ = "Python 学习者"

# 从子模块导入，提供简洁的 API
from .core import greet, calculate

# 控制 from mypackage import * 的行为
__all__ = ['greet', 'calculate', 'utils', 'models']

# 包初始化代码
print(f"[mypackage] 包已加载，版本 {__version__}")
```
