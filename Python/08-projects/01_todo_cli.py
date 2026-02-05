#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
            项目1：命令行 Todo 应用
============================================================
一个简单的命令行待办事项管理应用。

功能：
- 添加待办事项
- 列出所有待办事项
- 标记完成
- 删除待办事项
- 数据持久化（JSON 文件）

知识点：
- 类和对象
- 文件操作
- JSON 序列化
- 命令行交互
- 异常处理
============================================================
"""
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class Todo:
    """待办事项"""
    id: int
    title: str
    completed: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def complete(self):
        """标记为完成"""
        self.completed = True
        self.completed_at = datetime.now().isoformat()

    def __str__(self):
        status = "✓" if self.completed else "○"
        return f"[{status}] {self.id}. {self.title}"


class TodoApp:
    """Todo 应用"""

    def __init__(self, data_file: str = "todos.json"):
        self.data_file = Path(data_file)
        self.todos: List[Todo] = []
        self.next_id = 1
        self.load()

    def load(self):
        """从文件加载数据"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.todos = [Todo(**item) for item in data.get('todos', [])]
                    self.next_id = data.get('next_id', 1)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"加载数据失败: {e}")
                self.todos = []
                self.next_id = 1

    def save(self):
        """保存数据到文件"""
        data = {
            'todos': [asdict(todo) for todo in self.todos],
            'next_id': self.next_id
        }
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add(self, title: str) -> Todo:
        """添加待办事项"""
        todo = Todo(id=self.next_id, title=title)
        self.todos.append(todo)
        self.next_id += 1
        self.save()
        return todo

    def list_all(self, show_completed: bool = True) -> List[Todo]:
        """列出所有待办事项"""
        if show_completed:
            return self.todos
        return [t for t in self.todos if not t.completed]

    def get(self, todo_id: int) -> Optional[Todo]:
        """获取指定 ID 的待办事项"""
        for todo in self.todos:
            if todo.id == todo_id:
                return todo
        return None

    def complete(self, todo_id: int) -> bool:
        """标记为完成"""
        todo = self.get(todo_id)
        if todo:
            todo.complete()
            self.save()
            return True
        return False

    def delete(self, todo_id: int) -> bool:
        """删除待办事项"""
        todo = self.get(todo_id)
        if todo:
            self.todos.remove(todo)
            self.save()
            return True
        return False

    def clear_completed(self) -> int:
        """清除所有已完成的待办事项"""
        original_count = len(self.todos)
        self.todos = [t for t in self.todos if not t.completed]
        self.save()
        return original_count - len(self.todos)

    def stats(self) -> dict:
        """获取统计信息"""
        total = len(self.todos)
        completed = sum(1 for t in self.todos if t.completed)
        return {
            'total': total,
            'completed': completed,
            'pending': total - completed
        }


def print_help():
    """打印帮助信息"""
    print("""
命令列表:
  add <title>     添加待办事项
  list            列出所有待办事项
  list pending    只列出未完成的
  done <id>       标记为完成
  del <id>        删除待办事项
  clear           清除所有已完成
  stats           显示统计信息
  help            显示帮助
  quit            退出程序
""")


def main():
    """主函数"""
    print("=" * 50)
    print("       欢迎使用 Python Todo 应用")
    print("=" * 50)
    print("输入 'help' 查看命令列表\n")

    app = TodoApp()

    while True:
        try:
            user_input = input(">>> ").strip()
            if not user_input:
                continue

            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if command == "quit" or command == "exit":
                print("再见！")
                break

            elif command == "help":
                print_help()

            elif command == "add":
                if not args:
                    print("请提供待办事项标题")
                else:
                    todo = app.add(args)
                    print(f"已添加: {todo}")

            elif command == "list":
                show_all = args.lower() != "pending"
                todos = app.list_all(show_completed=show_all)
                if not todos:
                    print("没有待办事项")
                else:
                    print("\n待办事项列表:")
                    print("-" * 40)
                    for todo in todos:
                        print(f"  {todo}")
                    print("-" * 40)
                    stats = app.stats()
                    print(f"共 {stats['total']} 项，"
                          f"已完成 {stats['completed']} 项，"
                          f"待完成 {stats['pending']} 项\n")

            elif command == "done":
                try:
                    todo_id = int(args)
                    if app.complete(todo_id):
                        print(f"已完成: {app.get(todo_id)}")
                    else:
                        print(f"未找到 ID 为 {todo_id} 的待办事项")
                except ValueError:
                    print("请提供有效的 ID")

            elif command == "del":
                try:
                    todo_id = int(args)
                    if app.delete(todo_id):
                        print(f"已删除 ID {todo_id}")
                    else:
                        print(f"未找到 ID 为 {todo_id} 的待办事项")
                except ValueError:
                    print("请提供有效的 ID")

            elif command == "clear":
                count = app.clear_completed()
                print(f"已清除 {count} 个已完成的待办事项")

            elif command == "stats":
                stats = app.stats()
                print(f"\n统计信息:")
                print(f"  总数: {stats['total']}")
                print(f"  已完成: {stats['completed']}")
                print(f"  待完成: {stats['pending']}")
                if stats['total'] > 0:
                    rate = stats['completed'] / stats['total'] * 100
                    print(f"  完成率: {rate:.1f}%\n")

            else:
                print(f"未知命令: {command}")
                print("输入 'help' 查看命令列表")

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    main()
