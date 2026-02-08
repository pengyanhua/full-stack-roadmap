# 文件系统

## 课程概述

本教程深入讲解文件系统的原理与实现，包括文件系统层次结构、inode机制、日志系统、索引结构等核心技术。

**学习目标**：
- 理解文件系统的层次结构
- 掌握inode与文件元数据
- 深入理解ext4文件系统
- 学习日志文件系统原理
- 了解B+树索引机制
- 掌握VFS虚拟文件系统

---

## 1. 文件系统层次结构

### 1.1 文件系统架构

```
┌─────────────────────────────────────────────────────────────┐
│              Linux文件系统层次                                │
└─────────────────────────────────────────────────────────────┘

应用程序层
    │
    │ open(), read(), write(), close()
    ▼
┌─────────────────────────────────────────────┐
│  VFS (Virtual File System) 虚拟文件系统层   │
│  - 统一接口                                  │
│  - 文件描述符管理                            │
│  - dentry缓存                               │
│  - inode缓存                                │
└────────────────┬────────────────────────────┘
                 │
    ┌────────────┼────────────┬───────────┐
    ▼            ▼            ▼           ▼
┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│  ext4  │  │  XFS   │  │  Btrfs │  │  NFS   │  具体文件系统
└────┬───┘  └────┬───┘  └────┬───┘  └────┬───┘
     │           │           │           │
     └───────────┴───────────┴───────────┘
                 │
                 ▼
         ┌──────────────┐
         │ 块设备层 (BIO)│
         │ - 块I/O调度   │
         │ - 页缓存      │
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │  设备驱动层   │
         │ (SCSI/NVMe)  │
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │  物理存储     │
         │ (HDD/SSD)    │
         └──────────────┘

系统调用示例:
open("/home/user/file.txt", O_RDONLY)
    ↓
VFS查找dentry缓存
    ↓
未命中，调用ext4_lookup()
    ↓
读取inode
    ↓
返回文件描述符
```

### 1.2 文件系统挂载

```bash
# 查看已挂载文件系统
mount | column -t

# 查看文件系统类型
df -Th

# 挂载文件系统
mount -t ext4 /dev/sdb1 /mnt/data

# 查看挂载选项
cat /proc/mounts

# 查看文件系统超级块信息
dumpe2fs /dev/sda1 | head -20
```

## 2. inode与文件元数据

### 2.1 inode结构

```
┌─────────────────────────────────────────────────────────────┐
│                    inode结构                                  │
└─────────────────────────────────────────────────────────────┘

inode = index node (索引节点)

每个文件/目录都有一个inode，包含:
┌──────────────────────────────────┐
│  inode 编号: 123456              │
├──────────────────────────────────┤
│  元数据:                          │
│  • 文件类型 (普通/目录/链接)      │
│  • 权限 (rwxr-xr-x)              │
│  • 所有者 (UID/GID)              │
│  • 大小 (字节)                   │
│  • 时间戳:                       │
│    - atime (访问时间)            │
│    - mtime (修改时间)            │
│    - ctime (inode变更时间)       │
│  • 硬链接计数                     │
│  • 块指针:                       │
│    - 12个直接指针                │
│    - 1个一级间接指针             │
│    - 1个二级间接指针             │
│    - 1个三级间接指针             │
└──────────────────────────────────┘

注意: inode不包含文件名!
文件名存储在目录项(dentry)中
```

### 2.2 inode寻址机制

```
┌─────────────────────────────────────────────────────────────┐
│              inode数据块寻址 (ext4)                           │
└─────────────────────────────────────────────────────────────┘

inode
┌────────────────┐
│ 直接指针 0     │──────────▶ 数据块 (4KB)
│ 直接指针 1     │──────────▶ 数据块 (4KB)
│ ...            │
│ 直接指针 11    │──────────▶ 数据块 (4KB)
├────────────────┤                        共12个块 = 48KB
│ 一级间接指针   │──────────▶ 间接块
│                │            ├──▶ 数据块
│                │            ├──▶ 数据块
│                │            └──▶ 数据块
│                │                 (1024个指针 = 4MB)
├────────────────┤
│ 二级间接指针   │──────────▶ 间接块
│                │            ├──▶ 间接块 ──▶ 数据块...
│                │            ├──▶ 间接块 ──▶ 数据块...
│                │            └──▶ 间接块 ──▶ 数据块...
│                │                 (1024×1024个指针 = 4GB)
├────────────────┤
│ 三级间接指针   │──────────▶ 间接块
│                │            └──▶ 间接块 ──▶ 间接块 ──▶ 数据块...
└────────────────┘                 (最大 4TB)

现代ext4使用Extent代替间接块:
Extent = (逻辑块号, 物理块号, 块数量)
优势: 连续块只需一个extent，减少元数据
```

### 2.3 查看inode信息

```bash
# 查看文件inode号
ls -i file.txt

# 查看inode详细信息
stat file.txt

# 查看文件系统inode使用情况
df -i

# 查看目录的inode内容
ls -lid /home/user/

# 使用debugfs查看inode
debugfs -R "stat <12345>" /dev/sda1
```

```c
// 读取inode信息示例
#include <stdio.h>
#include <sys/stat.h>
#include <time.h>

void print_file_info(const char *path) {
    struct stat st;

    if (stat(path, &st) == -1) {
        perror("stat");
        return;
    }

    printf("文件: %s\n", path);
    printf("inode号: %lu\n", st.st_ino);
    printf("文件类型: ");

    switch (st.st_mode & S_IFMT) {
        case S_IFREG:  printf("普通文件\n"); break;
        case S_IFDIR:  printf("目录\n"); break;
        case S_IFLNK:  printf("符号链接\n"); break;
        case S_IFBLK:  printf("块设备\n"); break;
        case S_IFCHR:  printf("字符设备\n"); break;
        default:       printf("其他\n"); break;
    }

    printf("权限: %o\n", st.st_mode & 0777);
    printf("硬链接数: %lu\n", st.st_nlink);
    printf("所有者UID: %u\n", st.st_uid);
    printf("所有者GID: %u\n", st.st_gid);
    printf("文件大小: %ld 字节\n", st.st_size);
    printf("块数: %ld (512字节块)\n", st.st_blocks);
    printf("块大小: %ld 字节\n", st.st_blksize);

    printf("访问时间: %s", ctime(&st.st_atime));
    printf("修改时间: %s", ctime(&st.st_mtime));
    printf("状态改变时间: %s", ctime(&st.st_ctime));
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "用法: %s <文件路径>\n", argv[0]);
        return 1;
    }

    print_file_info(argv[1]);
    return 0;
}
```

## 3. ext4文件系统详解

### 3.1 ext4布局

```
┌─────────────────────────────────────────────────────────────┐
│              ext4文件系统布局                                 │
└─────────────────────────────────────────────────────────────┘

┌──────────────┬──────────────────────────────────────────────┐
│   引导块     │           块组0                               │
│   (1KB)     │                                                │
├──────────────┼──────────────────────────────────────────────┤
│              │  超级块  │ 块组描述符 │ 数据块位图 │ inode位图│
│              │  (1KB)   │            │            │          │
│              ├──────────┴────────────┴────────────┴──────────┤
│              │  inode表                                       │
│              │  (大量inode)                                  │
│              ├───────────────────────────────────────────────┤
│              │  数据块区域                                    │
│              │  (存储实际文件数据)                            │
└──────────────┴───────────────────────────────────────────────┘
                │
                │ 块组1、块组2... 重复相同结构
                ▼

超级块 (Superblock) 关键信息:
• 块大小 (1KB/2KB/4KB)
• 总块数
• 总inode数
• 空闲块数
• 空闲inode数
• 文件系统UUID
• 挂载次数
• 魔数 (0xEF53)

块组大小 = 块大小 × 8 × 块大小
例如: 4KB块 → 4096 × 8 × 4096 = 128MB/块组
```

### 3.2 目录项

```
┌─────────────────────────────────────────────────────────────┐
│              目录项 (Directory Entry)                         │
└─────────────────────────────────────────────────────────────┘

目录也是一个文件，存储目录项列表:

/home/user/目录的inode数据块:
┌───────────┬────────┬──────┬─────────────┐
│ inode号   │ 记录长│类型  │ 文件名       │
├───────────┼────────┼──────┼─────────────┤
│ 123       │ 12     │ DIR  │ .           │  (当前目录)
│ 100       │ 12     │ DIR  │ ..          │  (父目录)
│ 456       │ 20     │ FILE │ document.txt│
│ 789       │ 16     │ DIR  │ photos      │
│ 890       │ 24     │ LNK  │ link.txt    │
└───────────┴────────┴──────┴─────────────┘

文件名 ──查找目录项──▶ inode号 ──查找inode表──▶ inode ──▶ 数据块

这就是为什么:
• 硬链接不增加磁盘空间 (只增加目录项)
• 重命名文件很快 (只修改目录项)
• 符号链接是独立文件 (有自己的inode)
```

## 4. 日志文件系统

### 4.1 日志原理

```
┌─────────────────────────────────────────────────────────────┐
│              Journaling File System                           │
└─────────────────────────────────────────────────────────────┘

问题: 文件系统操作非原子性
例如: 创建文件需要:
1. 分配inode
2. 写入inode表
3. 更新目录
4. 标记位图
如果crash在中间 → 文件系统不一致!

日志解决方案:
    ┌──────────────┐
    │    日志区    │
    └──────┬───────┘
           │
  ┌────────┴─────────┐
  │  1. 写日志       │  记录要执行的操作
  │  2. 提交日志     │  标记日志完整
  │  3. 执行操作     │  真正修改文件系统
  │  4. 删除日志     │  操作完成
  └──────────────────┘

日志模式:
1. Journal (完整日志)
   - 记录元数据 + 数据
   - 最安全，最慢

2. Ordered (有序模式) **默认**
   - 只记录元数据
   - 先写数据，再写元数据
   - 平衡性能和安全

3. Writeback (回写模式)
   - 只记录元数据
   - 异步写入
   - 最快，但可能数据损坏

恢复流程:
系统崩溃后启动:
1. 扫描日志
2. 重放已提交但未完成的事务
3. 丢弃未提交的事务
4. 文件系统恢复一致性
```

### 4.2 配置日志模式

```bash
# 查看当前日志模式
tune2fs -l /dev/sda1 | grep "Default mount options"

# 修改日志模式 (需要重新挂载)
mount -o remount,data=journal /
mount -o remount,data=ordered /
mount -o remount,data=writeback /

# 查看日志状态
dumpe2fs /dev/sda1 | grep -i journal
```

## 5. B+树索引

### 5.1 B+树原理

```
┌─────────────────────────────────────────────────────────────┐
│              B+树索引 (用于目录和数据库)                      │
└─────────────────────────────────────────────────────────────┘

线性目录 vs B+树目录:

线性目录 (小目录):
查找"file999.txt" → 遍历所有项 → O(n)

B+树目录 (大目录):
                      根节点
                    [100, 500]
                   /    |    \
                  /     |     \
         [1-100] [101-500] [501-999]
           / | \    / | \     / | \
          叶  叶  叶  叶 叶 叶  叶 叶 叶
          节  节  节  节 节 节  节 节 节
          点  点  点  点 点 点  点 点 点
         (目录项)

查找"file999.txt" → 3次磁盘I/O → O(log n)

B+树特点:
• 所有数据在叶子节点
• 叶子节点链接成链表
• 非叶节点只存索引
• 高效范围查询
```

## 6. VFS虚拟文件系统

### 6.1 VFS架构

```
┌─────────────────────────────────────────────────────────────┐
│              VFS统一接口                                      │
└─────────────────────────────────────────────────────────────┘

VFS数据结构:
    superblock (超级块)
        │
        ├──▶ inode (索引节点)
        │       │
        │       └──▶ file (文件对象)
        │               │
        │               └──▶ dentry (目录项缓存)
        │
        └──▶ file_operations (文件操作)

用户空间
    │
    │ read(fd, buf, size)
    ▼
┌────────────────────────────┐
│  VFS Layer                 │
│  • 查找文件对象             │
│  • 调用file->f_op->read    │
└────────┬───────────────────┘
         │
     ┌───┴────┬─────────┬─────────┐
     ▼        ▼         ▼         ▼
  ext4_read  xfs_read  nfs_read  proc_read
     │        │         │         │
     └────────┴─────────┴─────────┘
              │
              ▼
          块设备层/网络层
```

### 6.2 文件描述符

```c
// 文件描述符与VFS对象
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

int main() {
    // open() 系统调用
    int fd = open("/etc/passwd", O_RDONLY);
    if (fd == -1) {
        perror("open");
        return 1;
    }

    // 内核中:
    // 1. 分配file对象
    // 2. 填充file->f_path (dentry + vfsmount)
    // 3. 填充file->f_op (文件操作函数表)
    // 4. 在进程文件描述符表中分配fd
    // 5. 返回fd给用户空间

    // read() 系统调用
    char buf[256];
    ssize_t n = read(fd, buf, sizeof(buf));

    // 内核中:
    // 1. 根据fd找到file对象
    // 2. 调用file->f_op->read()
    // 3. 具体文件系统实现read (如ext4_file_read_iter)
    // 4. 从页缓存或磁盘读取数据
    // 5. 返回读取的字节数

    printf("读取了 %zd 字节\n", n);
    close(fd);

    return 0;
}
```

## 7. 文件系统对比

### 7.1 常见文件系统比较

```
┌─────────────────────────────────────────────────────────────┐
│              文件系统对比                                     │
└─────────────────────────────────────────────────────────────┘

│文件系统│ 最大文件 │ 最大卷  │ 日志 │ 特点                │
├────────┼──────────┼─────────┼──────┼─────────────────────┤
│ ext4   │ 16 TB    │ 1 EB    │ ✓    │ Linux默认，稳定     │
│ XFS    │ 8 EB     │ 8 EB    │ ✓    │ 大文件性能好        │
│ Btrfs  │ 16 EB    │ 16 EB   │ ✓    │ 快照、压缩、COW     │
│ ZFS    │ 16 EB    │ 256 ZB  │ ✓    │ 企业级、数据完整性  │
│ NTFS   │ 16 EB    │ 256 TB  │ ✓    │ Windows默认         │
│ APFS   │ 8 EB     │ ∞       │ ✓    │ macOS默认，SSD优化  │
│ F2FS   │ 3.9 TB   │ 16 TB   │ ✓    │ 闪存优化            │

性能对比 (相对):
    顺序读写    随机读写    元数据操作
ext4    ★★★★☆    ★★★☆☆      ★★★★☆
XFS     ★★★★★    ★★★★☆      ★★★★★
Btrfs   ★★★★☆    ★★★☆☆      ★★★☆☆
F2FS    ★★★★★    ★★★★★      ★★★☆☆ (SSD)
```

## 8. 实战: 文件系统操作

### 8.1 创建和管理文件系统

```bash
# 创建ext4文件系统
mkfs.ext4 -L mydata /dev/sdb1

# 查看文件系统信息
dumpe2fs /dev/sdb1

# 修改文件系统参数
tune2fs -c 30 -i 180d /dev/sdb1  # 每30次挂载或180天检查

# 检查并修复文件系统
fsck.ext4 -f /dev/sdb1

# 调整文件系统大小
resize2fs /dev/sdb1 50G

# 创建XFS文件系统
mkfs.xfs -f -L xfsdata /dev/sdc1

# XFS文件系统修复
xfs_repair /dev/sdc1
```

### 8.2 文件系统性能测试

```python
# 文件系统性能测试工具
import os
import time

def test_sequential_write(filename, size_mb=100):
    """测试顺序写入性能"""
    data = b'x' * (1024 * 1024)  # 1MB数据
    start = time.time()

    with open(filename, 'wb') as f:
        for _ in range(size_mb):
            f.write(data)
        f.flush()
        os.fsync(f.fileno())  # 强制刷盘

    elapsed = time.time() - start
    throughput = size_mb / elapsed
    print(f"顺序写入: {throughput:.2f} MB/s")

def test_random_read(filename, num_reads=1000):
    """测试随机读取性能"""
    file_size = os.path.getsize(filename)
    start = time.time()

    with open(filename, 'rb') as f:
        for _ in range(num_reads):
            offset = (hash(os.urandom(8)) % file_size) // 4096 * 4096
            f.seek(offset)
            f.read(4096)

    elapsed = time.time() - start
    iops = num_reads / elapsed
    print(f"随机读取: {iops:.2f} IOPS")

def test_metadata_ops(dirname, num_files=1000):
    """测试元数据操作性能"""
    os.makedirs(dirname, exist_ok=True)
    start = time.time()

    # 创建文件
    for i in range(num_files):
        with open(f"{dirname}/file_{i}.txt", 'w') as f:
            f.write("test")

    # 删除文件
    for i in range(num_files):
        os.remove(f"{dirname}/file_{i}.txt")

    elapsed = time.time() - start
    ops_per_sec = (num_files * 2) / elapsed
    print(f"元数据操作: {ops_per_sec:.2f} ops/s")

    os.rmdir(dirname)

if __name__ == "__main__":
    print("文件系统性能测试")
    print("=" * 50)

    test_sequential_write("/tmp/test_seq.dat")
    test_random_read("/tmp/test_seq.dat")
    test_metadata_ops("/tmp/test_meta")

    # 清理
    os.remove("/tmp/test_seq.dat")
```

## 总结

文件系统是操作系统的重要组成部分：

**核心概念**：
1. VFS - 统一接口
2. inode - 文件元数据
3. 目录项 - 文件名到inode映射
4. 数据块 - 实际数据存储
5. 日志 - 一致性保证
6. B+树 - 快速索引

**关键技术**：
- Extent - 连续块分配
- 日志 - Crash恢复
- 延迟分配 - 性能优化
- 多块分配 - 减少碎片
- 预读 - 提高顺序读性能

**性能优化**：
- 使用合适的块大小
- 启用日志模式
- 禁用atime更新 (noatime)
- SSD使用discard/TRIM
- 定期碎片整理 (ext4)

## 下一步

- 学习 [05_io_management.md](05_io_management.md) - I/O管理
- 实践: 编写简单文件系统
- 深入: 阅读ext4源码

## 延伸阅读

- 《深入理解Linux内核》第12章 - VFS
- 《Operating Systems: Three Easy Pieces》- File Systems
- Linux源码: fs/目录
- ext4文档: https://ext4.wiki.kernel.org/
