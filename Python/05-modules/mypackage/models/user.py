"""
用户模型
"""
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


@dataclass
class User:
    """
    用户数据类

    Attributes:
        name: 用户名
        age: 年龄
        email: 邮箱（可选）
        created_at: 创建时间
        tags: 标签列表
    """
    name: str
    age: int
    email: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """初始化后验证"""
        if self.age < 0:
            raise ValueError("年龄不能为负数")
        if self.age > 150:
            raise ValueError("年龄不合理")

    def is_adult(self) -> bool:
        """是否成年"""
        return self.age >= 18

    def add_tag(self, tag: str) -> None:
        """添加标签"""
        if tag not in self.tags:
            self.tags.append(tag)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "name": self.name,
            "age": self.age,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        """从字典创建用户"""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        return cls(
            name=data["name"],
            age=data["age"],
            email=data.get("email"),
            created_at=created_at or datetime.now(),
            tags=data.get("tags", []),
        )


if __name__ == "__main__":
    user = User("Alice", 25, "alice@example.com")
    print(f"User: {user}")
    print(f"Is adult: {user.is_adult()}")

    user.add_tag("python")
    user.add_tag("developer")
    print(f"Tags: {user.tags}")

    user_dict = user.to_dict()
    print(f"Dict: {user_dict}")

    user2 = User.from_dict({"name": "Bob", "age": 30})
    print(f"From dict: {user2}")
