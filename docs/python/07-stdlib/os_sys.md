# os sys.py

::: info 文件信息
- 📄 原文件：`01_os_sys.py`
- 🔤 语言：python
:::

Python 标准库：os 和 sys 模块
本文件介绍 Python 中最常用的操作系统和系统模块。

## 完整代码

```python
import os
import sys
from pathlib import Path


def main01_os_basics():
    """
    ============================================================
                    1. os 模块基础
    ============================================================
    """
    print("=" * 60)
    print("1. os 模块基础")
    print("=" * 60)

    # 【环境信息】
    print("--- 环境信息 ---")
    print(f"操作系统: {os.name}")  # 'nt' (Windows) 或 'posix' (Linux/Mac)
    print(f"当前工作目录: {os.getcwd()}")
    print(f"用户主目录: {os.path.expanduser('~')}")

    # 【环境变量】
    print(f"\n--- 环境变量 ---")
    print(f"PATH: {os.environ.get('PATH', '')[:50]}...")
    print(f"HOME: {os.environ.get('HOME', os.environ.get('USERPROFILE', ''))}")

    # 设置环境变量
    os.environ['MY_VAR'] = 'my_value'
    print(f"MY_VAR: {os.environ.get('MY_VAR')}")


def main02_path_operations():
    """
    ============================================================
                    2. 路径操作
    ============================================================
    """
    print("\n" + "=" * 60)
    print("2. 路径操作")
    print("=" * 60)

    # 【os.path 模块】
    print("--- os.path 模块 ---")
    path = "/home/user/documents/file.txt"

    print(f"路径: {path}")
    print(f"目录名: {os.path.dirname(path)}")
    print(f"文件名: {os.path.basename(path)}")
    print(f"分割: {os.path.split(path)}")
    print(f"扩展名: {os.path.splitext(path)}")

    # 路径拼接
    joined = os.path.join("/home", "user", "documents")
    print(f"\n路径拼接: {joined}")

    # 路径判断
    current = os.getcwd()
    print(f"\n路径判断:")
    print(f"  exists('{current}'): {os.path.exists(current)}")
    print(f"  isdir('{current}'): {os.path.isdir(current)}")
    print(f"  isfile('{current}'): {os.path.isfile(current)}")
    print(f"  isabs('/home'): {os.path.isabs('/home')}")

    # 【pathlib 模块】（推荐使用）
    print(f"\n--- pathlib 模块（推荐）---")
    p = Path("/home/user/documents/file.txt")

    print(f"Path: {p}")
    print(f"parent: {p.parent}")
    print(f"name: {p.name}")
    print(f"stem: {p.stem}")
    print(f"suffix: {p.suffix}")
    print(f"parts: {p.parts}")

    # 路径操作
    new_path = Path.cwd() / "subdir" / "file.txt"
    print(f"\n路径拼接: {new_path}")

    # 路径方法
    print(f"\n当前目录:")
    cwd = Path.cwd()
    print(f"  cwd: {cwd}")
    print(f"  exists: {cwd.exists()}")
    print(f"  is_dir: {cwd.is_dir()}")


def main03_file_operations():
    """
    ============================================================
                    3. 文件操作
    ============================================================
    """
    print("\n" + "=" * 60)
    print("3. 文件操作")
    print("=" * 60)

    import tempfile

    # 使用临时目录进行演示
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 【创建目录】
        print("--- 创建目录 ---")
        new_dir = tmpdir / "new_directory"
        os.makedirs(new_dir, exist_ok=True)
        print(f"创建目录: {new_dir}")

        # 使用 pathlib
        another_dir = tmpdir / "another" / "nested"
        another_dir.mkdir(parents=True, exist_ok=True)
        print(f"创建嵌套目录: {another_dir}")

        # 【创建文件】
        print(f"\n--- 创建文件 ---")
        file_path = tmpdir / "test.txt"
        file_path.write_text("Hello, World!\n第二行", encoding="utf-8")
        print(f"创建文件: {file_path}")

        # 【读取文件】
        print(f"\n--- 读取文件 ---")
        content = file_path.read_text(encoding="utf-8")
        print(f"文件内容: {content}")

        # 【文件信息】
        print(f"\n--- 文件信息 ---")
        stat = file_path.stat()
        print(f"大小: {stat.st_size} bytes")
        print(f"修改时间: {stat.st_mtime}")

        # 【列出目录内容】
        print(f"\n--- 列出目录 ---")
        # 创建更多文件用于演示
        (tmpdir / "file1.py").touch()
        (tmpdir / "file2.py").touch()
        (tmpdir / "data.json").touch()

        print("os.listdir:")
        for item in os.listdir(tmpdir):
            print(f"  {item}")

        print("\nPath.iterdir:")
        for item in tmpdir.iterdir():
            print(f"  {item.name} ({'目录' if item.is_dir() else '文件'})")

        print("\nPath.glob (*.py):")
        for item in tmpdir.glob("*.py"):
            print(f"  {item.name}")

        # 【重命名和移动】
        print(f"\n--- 重命名 ---")
        old_path = tmpdir / "file1.py"
        new_path = tmpdir / "renamed.py"
        old_path.rename(new_path)
        print(f"重命名: {old_path.name} -> {new_path.name}")

        # 【删除】
        print(f"\n--- 删除 ---")
        new_path.unlink()  # 删除文件
        print(f"删除文件: {new_path.name}")

        # 删除空目录
        (tmpdir / "empty_dir").mkdir()
        (tmpdir / "empty_dir").rmdir()
        print("删除空目录: empty_dir")

        # 删除目录树
        import shutil
        shutil.rmtree(new_dir)
        print(f"删除目录树: {new_dir.name}")


def main04_sys_module():
    """
    ============================================================
                    4. sys 模块
    ============================================================
    """
    print("\n" + "=" * 60)
    print("4. sys 模块")
    print("=" * 60)

    # 【Python 信息】
    print("--- Python 信息 ---")
    print(f"Python 版本: {sys.version}")
    print(f"版本信息: {sys.version_info}")
    print(f"平台: {sys.platform}")
    print(f"可执行文件: {sys.executable}")

    # 【命令行参数】
    print(f"\n--- 命令行参数 ---")
    print(f"sys.argv: {sys.argv}")

    # 【模块搜索路径】
    print(f"\n--- 模块搜索路径 ---")
    print("sys.path:")
    for i, path in enumerate(sys.path[:5]):
        print(f"  {i}: {path}")
    print("  ...")

    # 【已导入模块】
    print(f"\n--- 已导入模块 ---")
    print(f"已导入模块数: {len(sys.modules)}")
    print("部分模块:", list(sys.modules.keys())[:10])

    # 【递归限制】
    print(f"\n--- 递归限制 ---")
    print(f"递归限制: {sys.getrecursionlimit()}")
    # sys.setrecursionlimit(2000)  # 可以修改

    # 【标准流】
    print(f"\n--- 标准流 ---")
    print(f"stdin: {sys.stdin}")
    print(f"stdout: {sys.stdout}")
    print(f"stderr: {sys.stderr}")

    # 【内存和引用】
    print(f"\n--- 内存和引用 ---")
    x = [1, 2, 3]
    print(f"sys.getsizeof([1,2,3]): {sys.getsizeof(x)} bytes")
    print(f"sys.getrefcount(x): {sys.getrefcount(x)}")


def main05_shutil_module():
    """
    ============================================================
                    5. shutil 模块（高级文件操作）
    ============================================================
    """
    print("\n" + "=" * 60)
    print("5. shutil 模块")
    print("=" * 60)

    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 创建测试文件和目录
        src_file = tmpdir / "source.txt"
        src_file.write_text("Hello, World!")

        src_dir = tmpdir / "source_dir"
        src_dir.mkdir()
        (src_dir / "file1.txt").write_text("File 1")
        (src_dir / "file2.txt").write_text("File 2")

        # 【复制文件】
        print("--- 复制文件 ---")
        dst_file = tmpdir / "copy.txt"
        shutil.copy(src_file, dst_file)  # 复制文件
        print(f"copy: {src_file.name} -> {dst_file.name}")

        shutil.copy2(src_file, tmpdir / "copy2.txt")  # 保留元数据
        print("copy2: 保留元数据")

        # 【复制目录】
        print(f"\n--- 复制目录 ---")
        dst_dir = tmpdir / "dest_dir"
        shutil.copytree(src_dir, dst_dir)
        print(f"copytree: {src_dir.name} -> {dst_dir.name}")

        # 【移动】
        print(f"\n--- 移动 ---")
        shutil.move(tmpdir / "copy.txt", tmpdir / "moved.txt")
        print("move: copy.txt -> moved.txt")

        # 【删除目录树】
        # shutil.rmtree(dst_dir)

        # 【磁盘使用情况】
        print(f"\n--- 磁盘使用 ---")
        usage = shutil.disk_usage("/")
        print(f"总空间: {usage.total / (1024**3):.2f} GB")
        print(f"已使用: {usage.used / (1024**3):.2f} GB")
        print(f"可用: {usage.free / (1024**3):.2f} GB")

        # 【查找可执行文件】
        print(f"\n--- 查找可执行文件 ---")
        python_path = shutil.which("python")
        print(f"python: {python_path}")


def main06_glob_module():
    """
    ============================================================
                    6. glob 模块（文件模式匹配）
    ============================================================
    """
    print("\n" + "=" * 60)
    print("6. glob 模块")
    print("=" * 60)

    import glob
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 创建测试文件
        (tmpdir / "file1.py").touch()
        (tmpdir / "file2.py").touch()
        (tmpdir / "test.txt").touch()
        (tmpdir / "data.json").touch()
        subdir = tmpdir / "subdir"
        subdir.mkdir()
        (subdir / "nested.py").touch()

        os.chdir(tmpdir)

        # 【基本模式】
        print("--- 基本模式 ---")
        print(f"*.py: {glob.glob('*.py')}")
        print(f"*.txt: {glob.glob('*.txt')}")
        print(f"file*.py: {glob.glob('file*.py')}")

        # 【递归匹配】
        print(f"\n--- 递归匹配 ---")
        print(f"**/*.py: {glob.glob('**/*.py', recursive=True)}")

        # 【使用 pathlib】
        print(f"\n--- pathlib.glob ---")
        print(f"Path.glob('*.py'): {list(tmpdir.glob('*.py'))}")
        print(f"Path.rglob('*.py'): {list(tmpdir.rglob('*.py'))}")


if __name__ == "__main__":
    main01_os_basics()
    main02_path_operations()
    main03_file_operations()
    main04_sys_module()
    main05_shutil_module()
    main06_glob_module()
```
