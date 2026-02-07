# product

::: info 文件信息
- 📄 原文件：`product.py`
- 🔤 语言：python
:::

产品模型

## 完整代码

```python
from dataclasses import dataclass, field
from typing import Optional
from decimal import Decimal


@dataclass
class Product:
    """
    产品数据类

    Attributes:
        name: 产品名称
        price: 价格
        quantity: 库存数量
        category: 分类
        description: 描述
    """
    name: str
    price: Decimal
    quantity: int = 0
    category: str = "未分类"
    description: Optional[str] = None

    def __post_init__(self):
        """初始化后处理"""
        if isinstance(self.price, (int, float)):
            self.price = Decimal(str(self.price))
        if self.price < 0:
            raise ValueError("价格不能为负数")
        if self.quantity < 0:
            raise ValueError("库存不能为负数")

    @property
    def in_stock(self) -> bool:
        """是否有库存"""
        return self.quantity > 0

    @property
    def total_value(self) -> Decimal:
        """库存总价值"""
        return self.price * self.quantity

    def add_stock(self, amount: int) -> None:
        """增加库存"""
        if amount < 0:
            raise ValueError("增加数量不能为负数")
        self.quantity += amount

    def remove_stock(self, amount: int) -> bool:
        """减少库存，返回是否成功"""
        if amount < 0:
            raise ValueError("减少数量不能为负数")
        if amount > self.quantity:
            return False
        self.quantity -= amount
        return True

    def apply_discount(self, percent: float) -> None:
        """应用折扣"""
        if not 0 <= percent <= 100:
            raise ValueError("折扣必须在 0-100 之间")
        discount = Decimal(str(percent / 100))
        self.price = self.price * (1 - discount)

    def __str__(self) -> str:
        return f"{self.name} - ¥{self.price} ({self.quantity}件)"


if __name__ == "__main__":
    product = Product("Python 编程书", Decimal("59.99"), 100, "书籍")
    print(f"Product: {product}")
    print(f"In stock: {product.in_stock}")
    print(f"Total value: ¥{product.total_value}")

    product.apply_discount(10)
    print(f"After 10% discount: {product}")

    product.remove_stock(5)
    print(f"After removing 5: {product}")
```
