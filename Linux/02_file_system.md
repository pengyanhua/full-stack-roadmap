# Linux 文件系统

## 一、文件系统概述

### 常见文件系统类型

| 文件系统 | 特点 | 适用场景 |
|---------|------|----------|
| **ext4** | Linux 默认,成熟稳定 | 通用场景 |
| **XFS** | 高性能,大文件支持好 | 大文件存储、数据库 |
| **Btrfs** | 快照、校验、压缩 | 现代化存储需求 |
| **ZFS** | 企业级特性,数据完整性 | 高可靠性需求 |
| **NTFS** | Windows 文件系统 | 与 Windows 共享 |
| **FAT32/exFAT** | 跨平台兼容性好 | U盘、移动存储 |
| **NFS** | 网络文件系统 | 网络共享 |
| **tmpfs** | 内存文件系统 | 临时数据、高速缓存 |

### 文件系统结构

```
文件系统组成:
┌────────────────────────────────────────────────────────────┐
│                     文件系统 (ext4)                         │
├────────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ 超级块      │  │  块组描述符  │  │    数据块         │  │
│  │ Superblock │  │  Group Desc. │  │    Data Blocks    │  │
│  │            │  │              │  │                   │  │
│  │ - 文件系统  │  │ - inode表位置│  │ - inode 表        │  │
│  │   信息      │  │ - 数据块位置 │  │ - 文件数据        │  │
│  │ - 总块数    │  │ - 空闲块统计 │  │ - 目录数据        │  │
│  │ - 块大小    │  │              │  │                   │  │
│  └────────────┘  └──────────────┘  └───────────────────┘  │
└────────────────────────────────────────────────────────────┘

inode 结构:
┌────────────────────────────────────────────┐
│           inode (索引节点)                 │
├────────────────────────────────────────────┤
│  文件元数据:                               │
│  ├── inode 号 (文件唯一标识)              │
│  ├── 文件类型 (普通文件/目录/链接)        │
│  ├── 权限 (rwx)                           │
│  ├── 所有者 (UID/GID)                     │
│  ├── 大小 (字节)                          │
│  ├── 时间戳:                              │
│  │   ├── atime (访问时间)                │
│  │   ├── mtime (修改时间)                │
│  │   └── ctime (状态改变时间)            │
│  ├── 链接数                               │
│  └── 数据块指针 (指向实际数据)            │
└────────────────────────────────────────────┘

注意:
- 文件名存储在目录项中,不在 inode 中
- 目录是特殊的文件,内容是文件名到 inode 的映射
- 硬链接共享同一个 inode
- 符号链接有自己的 inode,内容是目标路径
```

## 二、磁盘管理

### 查看磁盘信息

```bash
# ============================================================
#                   磁盘和分区信息
# ============================================================

# 查看块设备
lsblk
# 输出示例:
# NAME   MAJ:MIN RM   SIZE RO TYPE MOUNTPOINT
# sda      8:0    0   500G  0 disk
# ├─sda1   8:1    0   512M  0 part /boot/efi
# ├─sda2   8:2    0    20G  0 part /
# └─sda3   8:3    0   479G  0 part /home
# sr0     11:0    1  1024M  0 rom

# 包含文件系统类型
lsblk -f
# NAME   FSTYPE LABEL    UUID                                 MOUNTPOINT
# sda
# ├─sda1 vfat   EFI      XXXX-XXXX                            /boot/efi
# ├─sda2 ext4            xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx /
# └─sda3 ext4            xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx /home

# 查看所有磁盘分区
sudo fdisk -l
sudo fdisk -l /dev/sda                # 查看指定磁盘

# 查看磁盘使用情况
df -h                                 # 人类可读格式
df -h /                               # 查看根分区
df -h --total                         # 显示总计
df -i                                 # 查看 inode 使用情况

# 输出示例:
# Filesystem      Size  Used Avail Use% Mounted on
# /dev/sda2        20G  8.5G   11G  45% /
# /dev/sda3       479G  120G  335G  27% /home
# tmpfs           7.8G  2.1M  7.8G   1% /run

# 查看磁盘空间使用详情
du -h /var/log                        # 目录大小
du -sh /var/log                       # 总大小
du -h --max-depth=1 /var              # 只看一级目录
du -h --max-depth=1 / | sort -hr      # 按大小排序

# 查找大文件
find / -type f -size +100M -exec ls -lh {} \; 2>/dev/null
find /var -type f -size +100M -printf "%s %p\n" 2>/dev/null | sort -rn | head -10

# ============================================================
#                   磁盘健康状态
# ============================================================

# 使用 smartctl (需要安装 smartmontools)
sudo apt install smartmontools        # Debian/Ubuntu
sudo yum install smartmontools        # CentOS

sudo smartctl -a /dev/sda             # 查看详细信息
sudo smartctl -H /dev/sda             # 健康状态
sudo smartctl -t short /dev/sda       # 运行短测试
sudo smartctl -t long /dev/sda        # 运行长测试

# 查看磁盘 I/O 统计
iostat -x 2                           # 每2秒刷新
iotop                                 # 实时 I/O 监控
```

### 磁盘分区

```bash
# ============================================================
#                   使用 fdisk 分区 (MBR)
# ============================================================

# 进入 fdisk
sudo fdisk /dev/sdb

# fdisk 交互命令:
m       # 查看帮助
p       # 打印分区表
n       # 创建新分区
d       # 删除分区
t       # 修改分区类型
w       # 保存并退出
q       # 不保存退出

# 创建分区步骤:
# 1. sudo fdisk /dev/sdb
# 2. n (新建分区)
# 3. p (主分区) 或 e (扩展分区)
# 4. 分区号 (1-4)
# 5. 起始扇区 (默认回车)
# 6. 结束扇区 (如 +10G 表示10GB)
# 7. w (保存)

# ============================================================
#                   使用 parted 分区 (GPT,推荐)
# ============================================================

# 交互模式
sudo parted /dev/sdb

# parted 命令:
(parted) print                        # 打印分区表
(parted) mklabel gpt                  # 创建 GPT 分区表
(parted) mklabel msdos                # 创建 MBR 分区表
(parted) mkpart primary ext4 0% 10GB  # 创建分区
(parted) rm 1                         # 删除分区1
(parted) quit                         # 退出

# 非交互模式 (脚本使用)
sudo parted /dev/sdb mklabel gpt
sudo parted /dev/sdb mkpart primary ext4 0% 10GB
sudo parted /dev/sdb mkpart primary ext4 10GB 20GB

# ============================================================
#                   格式化分区
# ============================================================

# ext4 格式化
sudo mkfs.ext4 /dev/sdb1
sudo mkfs.ext4 -L mydisk /dev/sdb1    # 指定卷标

# XFS 格式化
sudo mkfs.xfs /dev/sdb1
sudo mkfs.xfs -L mydisk /dev/sdb1

# 其他文件系统
sudo mkfs.vfat /dev/sdb1              # FAT32
sudo mkfs.exfat /dev/sdb1             # exFAT (需要安装 exfat-utils)
sudo mkfs.ntfs /dev/sdb1              # NTFS (需要安装 ntfs-3g)

# 创建交换分区
sudo mkswap /dev/sdb2
sudo swapon /dev/sdb2                 # 启用交换分区
sudo swapoff /dev/sdb2                # 禁用交换分区
swapon --show                         # 查看交换分区

# ============================================================
#                   重新读取分区表
# ============================================================

sudo partprobe /dev/sdb               # 通知内核重新读取分区表
sudo partprobe -s                     # 显示当前分区
```

### 挂载文件系统

```bash
# ============================================================
#                   临时挂载
# ============================================================

# 创建挂载点
sudo mkdir -p /mnt/mydisk

# 挂载文件系统
sudo mount /dev/sdb1 /mnt/mydisk
sudo mount -t ext4 /dev/sdb1 /mnt/mydisk     # 指定文件系统类型

# 挂载选项
sudo mount -o ro /dev/sdb1 /mnt/mydisk       # 只读挂载
sudo mount -o rw /dev/sdb1 /mnt/mydisk       # 读写挂载
sudo mount -o noexec /dev/sdb1 /mnt/mydisk   # 不允许执行
sudo mount -o nosuid /dev/sdb1 /mnt/mydisk   # 忽略 SUID/SGID
sudo mount -o loop file.iso /mnt/iso         # 挂载 ISO 文件

# 查看挂载信息
mount                                # 所有挂载点
mount | grep sdb                     # 查看特定设备
df -h                                # 挂载点和使用情况
findmnt                              # 树形显示挂载点
findmnt /mnt/mydisk                  # 查看特定挂载点

# 卸载文件系统
sudo umount /mnt/mydisk
sudo umount /dev/sdb1
sudo umount -l /mnt/mydisk           # 懒卸载 (立即卸载,等待使用完成)
sudo umount -f /mnt/mydisk           # 强制卸载 (不推荐)

# 如果提示 "target is busy"
lsof /mnt/mydisk                     # 查看哪些进程在使用
fuser -m /mnt/mydisk                 # 查看使用的进程
fuser -km /mnt/mydisk                # 杀死使用的进程

# ============================================================
#                   永久挂载 (/etc/fstab)
# ============================================================

# 查看设备 UUID (推荐使用 UUID 而不是设备名)
sudo blkid /dev/sdb1
# 输出: /dev/sdb1: UUID="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" TYPE="ext4"

lsblk -f                             # 另一种查看方式

# 编辑 /etc/fstab
sudo nano /etc/fstab

# fstab 格式:
# <设备>  <挂载点>  <文件系统>  <选项>  <dump>  <fsck>

# 示例:
UUID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx /mnt/mydisk ext4 defaults 0 2
/dev/sdb1                                 /mnt/mydisk ext4 defaults 0 2

# 常用选项:
# defaults    = rw,suid,dev,exec,auto,nouser,async
# ro          = 只读
# rw          = 读写
# noauto      = 不自动挂载
# noexec      = 不允许执行程序
# nosuid      = 忽略 SUID/SGID
# nodev       = 不解释设备文件
# user        = 允许普通用户挂载
# users       = 允许所有用户挂载
# nofail      = 挂载失败不阻止启动 (移动设备推荐)

# dump 字段 (备份):
# 0 = 不备份
# 1 = 每天备份

# fsck 字段 (启动时检查顺序):
# 0 = 不检查
# 1 = 根文件系统
# 2 = 其他文件系统

# 常用配置示例:
UUID=xxx /              ext4    defaults                0 1
UUID=xxx /home          ext4    defaults                0 2
UUID=xxx /mnt/data      ext4    defaults,nofail         0 2
UUID=xxx /mnt/backup    ext4    defaults,noauto         0 0
UUID=xxx none           swap    sw                      0 0
/dev/sr0 /media/cdrom   iso9660 ro,noauto,user          0 0

# 测试 fstab 配置
sudo mount -a                    # 挂载所有 fstab 中的文件系统
sudo findmnt --verify            # 验证 fstab 语法

# ============================================================
#                   挂载网络文件系统
# ============================================================

# NFS 挂载
sudo apt install nfs-common      # Debian/Ubuntu
sudo yum install nfs-utils       # CentOS

# 临时挂载 NFS
sudo mount -t nfs server:/path /mnt/nfs

# fstab 中配置 NFS
server:/path /mnt/nfs nfs defaults,_netdev 0 0

# CIFS/SMB 挂载 (Windows 共享)
sudo apt install cifs-utils      # Debian/Ubuntu
sudo yum install cifs-utils      # CentOS

# 临时挂载
sudo mount -t cifs //server/share /mnt/smb -o username=user,password=pass

# fstab 中配置 (推荐使用凭证文件)
# 创建凭证文件
sudo nano /root/.smbcredentials
# username=user
# password=pass
sudo chmod 600 /root/.smbcredentials

# fstab 配置
//server/share /mnt/smb cifs credentials=/root/.smbcredentials,uid=1000,gid=1000 0 0
```

## 三、LVM (逻辑卷管理)

### LVM 概念

```
LVM 架构:
┌────────────────────────────────────────────────────────────┐
│                      逻辑卷 (LV)                            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ /dev/vg0/  │  │ /dev/vg0/  │  │ /dev/vg0/  │            │
│  │   lv_root  │  │   lv_home  │  │   lv_data  │            │
│  │   (20GB)   │  │   (50GB)   │  │   (100GB)  │            │
│  └────────────┘  └────────────┘  └────────────┘            │
├────────────────────────────────────────────────────────────┤
│                      卷组 (VG)                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                vg0 (200GB)                            │  │
│  │  ┌────────┬────────┬────────┬────────┬────────┐      │  │
│  │  │   PE   │   PE   │   PE   │  ...   │   PE   │      │  │
│  │  │  (4MB) │  (4MB) │  (4MB) │        │  (4MB) │      │  │
│  │  └────────┴────────┴────────┴────────┴────────┘      │  │
│  └──────────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────────┤
│                    物理卷 (PV)                              │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐   │
│  │   /dev/sda3   │  │   /dev/sdb1   │  │  /dev/sdc1   │   │
│  │    (50GB)     │  │    (100GB)    │  │   (50GB)     │   │
│  └───────────────┘  └───────────────┘  └──────────────┘   │
├────────────────────────────────────────────────────────────┤
│                    物理磁盘                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐   │
│  │   /dev/sda    │  │   /dev/sdb    │  │  /dev/sdc    │   │
│  └───────────────┘  └───────────────┘  └──────────────┘   │
└────────────────────────────────────────────────────────────┘

概念说明:
PV (Physical Volume)     - 物理卷:物理磁盘或分区
VG (Volume Group)        - 卷组:一个或多个 PV 的集合
LV (Logical Volume)      - 逻辑卷:从 VG 中分配的逻辑分区
PE (Physical Extent)     - 物理扩展:VG 中的最小存储单元 (默认4MB)
LE (Logical Extent)      - 逻辑扩展:LV 中的最小存储单元

优点:
├── 灵活调整大小:可以在线扩展或缩减
├── 快照功能:支持创建快照备份
├── 跨磁盘:一个 LV 可以跨多个物理磁盘
└── 磁盘整合:多个小磁盘可以组成大容量 VG
```

### LVM 操作

```bash
# ============================================================
#                   创建 LVM
# ============================================================

# 1. 创建物理卷 (PV)
sudo pvcreate /dev/sdb1
sudo pvcreate /dev/sdc1
sudo pvcreate /dev/sdb1 /dev/sdc1     # 批量创建

# 查看 PV
sudo pvs                              # 简要信息
sudo pvdisplay                        # 详细信息
sudo pvdisplay /dev/sdb1              # 查看特定 PV

# 2. 创建卷组 (VG)
sudo vgcreate vg0 /dev/sdb1 /dev/sdc1
sudo vgcreate -s 8M vg0 /dev/sdb1     # 指定 PE 大小为 8MB

# 查看 VG
sudo vgs                              # 简要信息
sudo vgdisplay                        # 详细信息
sudo vgdisplay vg0                    # 查看特定 VG

# 3. 创建逻辑卷 (LV)
sudo lvcreate -L 20G -n lv_root vg0              # 创建 20GB 的 LV
sudo lvcreate -L 50G -n lv_home vg0
sudo lvcreate -l 100%FREE -n lv_data vg0         # 使用所有剩余空间
sudo lvcreate -l 50%VG -n lv_data vg0            # 使用 VG 50%的空间

# 查看 LV
sudo lvs                              # 简要信息
sudo lvdisplay                        # 详细信息
sudo lvdisplay /dev/vg0/lv_root       # 查看特定 LV

# 4. 格式化并挂载
sudo mkfs.ext4 /dev/vg0/lv_root
sudo mkdir /mnt/lv_root
sudo mount /dev/vg0/lv_root /mnt/lv_root

# ============================================================
#                   扩展 LVM
# ============================================================

# 扩展逻辑卷
sudo lvextend -L +10G /dev/vg0/lv_home           # 增加 10GB
sudo lvextend -L 60G /dev/vg0/lv_home            # 扩展到 60GB
sudo lvextend -l +100%FREE /dev/vg0/lv_home      # 使用所有剩余空间

# 扩展文件系统 (ext4)
sudo resize2fs /dev/vg0/lv_home                  # ext4 在线扩展

# XFS 文件系统
sudo xfs_growfs /mnt/lv_home                     # XFS 在线扩展

# 一步完成 (扩展 LV 并自动扩展文件系统)
sudo lvextend -L +10G -r /dev/vg0/lv_home

# 扩展卷组 (添加新磁盘)
sudo pvcreate /dev/sdd1
sudo vgextend vg0 /dev/sdd1                      # 将新 PV 加入 VG

# ============================================================
#                   缩减 LVM (危险操作,备份数据!)
# ============================================================

# 注意:
# - XFS 不支持缩减
# - ext4 缩减需要先卸载
# - 缩减前务必备份数据

# 1. 卸载文件系统
sudo umount /mnt/lv_home

# 2. 检查文件系统
sudo e2fsck -f /dev/vg0/lv_home

# 3. 缩减文件系统
sudo resize2fs /dev/vg0/lv_home 40G              # 缩减到 40GB

# 4. 缩减逻辑卷
sudo lvreduce -L 40G /dev/vg0/lv_home

# 5. 重新挂载
sudo mount /dev/vg0/lv_home /mnt/lv_home

# ============================================================
#                   删除 LVM
# ============================================================

# 1. 卸载文件系统
sudo umount /mnt/lv_home

# 2. 删除逻辑卷
sudo lvremove /dev/vg0/lv_home

# 3. 删除卷组
sudo vgremove vg0

# 4. 删除物理卷
sudo pvremove /dev/sdb1

# ============================================================
#                   LVM 快照
# ============================================================

# 创建快照 (需要预留空间)
sudo lvcreate -L 5G -s -n lv_home_snap /dev/vg0/lv_home

# 挂载快照 (只读)
sudo mkdir /mnt/snap
sudo mount -o ro /dev/vg0/lv_home_snap /mnt/snap

# 恢复快照
sudo umount /mnt/lv_home
sudo lvconvert --merge /dev/vg0/lv_home_snap     # 合并快照

# 删除快照
sudo umount /mnt/snap
sudo lvremove /dev/vg0/lv_home_snap
```

## 四、文件操作

### inode 操作

```bash
# 查看 inode 信息
ls -i file.txt                       # 查看 inode 号
stat file.txt                        # 详细的 inode 信息

# 输出示例:
#   File: file.txt
#   Size: 1234        Blocks: 8          IO Block: 4096   regular file
#   Device: 801h/2049d	Inode: 123456      Links: 1
#   Access: (0644/-rw-r--r--)  Uid: ( 1000/   user)   Gid: ( 1000/   user)
#   Access: 2024-01-01 10:00:00.000000000 +0800
#   Modify: 2024-01-01 09:00:00.000000000 +0800
#   Change: 2024-01-01 09:00:00.000000000 +0800

# 查看 inode 使用情况
df -i                                # 所有文件系统
df -i /                              # 根文件系统

# 按 inode 号查找文件
find / -inum 123456

# 删除乱码文件名的文件 (通过 inode)
find . -inum 123456 -delete
find . -inum 123456 -exec rm {} \;

# 查看目录占用的 inode 数量
find /var -type f | wc -l            # 文件数量
find /var -type d | wc -l            # 目录数量
```

### 硬链接与软链接

```bash
# ============================================================
#                   硬链接 (Hard Link)
# ============================================================

# 创建硬链接
ln source.txt hardlink.txt

# 特点:
# - 共享同一个 inode
# - 删除任意一个,数据不会丢失 (链接计数 > 0)
# - 不能跨文件系统
# - 不能链接目录
# - 无法区分哪个是原始文件

# 验证硬链接
ls -li source.txt hardlink.txt
# 输出:
# 123456 -rw-r--r-- 2 user user 1234 Jan  1 10:00 source.txt
# 123456 -rw-r--r-- 2 user user 1234 Jan  1 10:00 hardlink.txt
#   ↑                ↑
# inode号相同      链接计数为2

# ============================================================
#                   软链接/符号链接 (Symbolic Link)
# ============================================================

# 创建软链接
ln -s source.txt symlink.txt
ln -s /path/to/source.txt /path/to/symlink.txt   # 建议使用绝对路径

# 特点:
# - 有独立的 inode
# - 内容是目标路径
# - 可以跨文件系统
# - 可以链接目录
# - 删除原文件,软链接失效 (悬空链接)
# - 类似 Windows 的快捷方式

# 验证软链接
ls -li source.txt symlink.txt
# 输出:
# 123456 -rw-r--r-- 1 user user   1234 Jan  1 10:00 source.txt
# 789012 lrwxrwxrwx 1 user user     10 Jan  1 10:05 symlink.txt -> source.txt
#   ↑                                                   ↑
# inode号不同                                        指向原文件

# 查看链接目标
readlink symlink.txt                 # 显示链接目标
readlink -f symlink.txt              # 显示绝对路径

# 查找软链接
find /etc -type l                    # 查找所有软链接
find /etc -type l -ls                # 详细信息

# 查找悬空链接 (原文件已删除)
find /etc -type l -xtype l

# 修改软链接
ln -sf new_target.txt symlink.txt    # -f 强制覆盖

# ============================================================
#                   对比
# ============================================================

# 硬链接 vs 软链接:
┌──────────────┬─────────────────┬─────────────────┐
│     特性     │    硬链接       │    软链接       │
├──────────────┼─────────────────┼─────────────────┤
│ inode        │ 相同            │ 不同            │
│ 跨文件系统   │ 不可以          │ 可以            │
│ 链接目录     │ 不可以          │ 可以            │
│ 原文件删除   │ 数据仍存在      │ 链接失效        │
│ 相对路径     │ 不适用          │ 支持            │
│ 文件大小     │ 与原文件相同    │ 目标路径长度    │
└──────────────┴─────────────────┴─────────────────┘
```

### 文件属性

```bash
# ============================================================
#                   扩展属性
# ============================================================

# 查看扩展属性
lsattr file.txt

# 输出示例:
# ----i---------e----- file.txt
# 常见属性:
#   a: append only (只能追加)
#   i: immutable (不可修改、删除、重命名)
#   c: compressed (自动压缩)
#   s: secure deletion (删除时覆盖磁盘)
#   u: undeletable (删除时可恢复)
#   A: no atime update (不更新访问时间)

# 设置属性
sudo chattr +i file.txt              # 设置不可修改
sudo chattr -i file.txt              # 取消不可修改
sudo chattr +a file.txt              # 只允许追加

# 使用场景:
# +i: 保护重要配置文件不被误删
# +a: 保护日志文件不被清空,只允许追加

# ============================================================
#                   ACL (访问控制列表)
# ============================================================

# 查看 ACL
getfacl file.txt

# 输出示例:
# # file: file.txt
# # owner: user
# # group: user
# user::rw-
# user:john:rw-         # 用户 john 有读写权限
# group::r--
# group:developers:rw-  # 组 developers 有读写权限
# mask::rw-
# other::r--

# 设置 ACL
setfacl -m u:john:rw file.txt        # 给用户 john 读写权限
setfacl -m g:developers:rw file.txt  # 给组 developers 读写权限
setfacl -m o::r file.txt             # 其他人只读

# 删除 ACL
setfacl -x u:john file.txt           # 删除用户 john 的 ACL
setfacl -b file.txt                  # 删除所有 ACL

# 递归设置 ACL
setfacl -R -m u:john:rwx /project    # 递归设置目录

# 默认 ACL (新建文件继承)
setfacl -d -m u:john:rw /project     # 设置默认 ACL
setfacl -d -m g:developers:rw /project

# 复制 ACL
getfacl file1 | setfacl --set-file=- file2

# 备份和恢复 ACL
getfacl -R /project > acl_backup.txt
setfacl --restore=acl_backup.txt
```

### 文件查找

```bash
# ============================================================
#                   find 命令
# ============================================================

# 按名称查找
find /path -name "*.txt"             # 精确匹配
find /path -iname "*.TXT"            # 忽略大小写
find /path -name "file*"             # 通配符

# 按类型查找
find /path -type f                   # 普通文件
find /path -type d                   # 目录
find /path -type l                   # 软链接
find /path -type b                   # 块设备
find /path -type c                   # 字符设备

# 按大小查找
find /path -size +100M               # 大于 100MB
find /path -size -10k                # 小于 10KB
find /path -size 1G                  # 等于 1GB
find /var -size +100M -size -1G      # 100MB 到 1GB 之间

# 按时间查找
find /path -mtime -7                 # 7天内修改过
find /path -mtime +30                # 30天前修改
find /path -atime -1                 # 1天内访问过
find /path -ctime -7                 # 7天内状态改变

# 按权限查找
find /path -perm 644                 # 权限为 644
find /path -perm -644                # 至少有 644 权限
find /path -perm /u+w                # 所有者可写

# 按所有者查找
find /path -user john                # 所有者是 john
find /path -group developers         # 组是 developers
find /path -nouser                   # 找到无主文件
find /path -nogroup                  # 找到无主组文件

# 组合条件
find /path -name "*.txt" -size +1M   # 与 (AND)
find /path -name "*.txt" -o -name "*.log"  # 或 (OR)
find /path ! -name "*.txt"           # 非 (NOT)
find /path \( -name "*.txt" -o -name "*.log" \) -size +1M

# 深度限制
find /path -maxdepth 1 -name "*.txt" # 只搜索一层
find /path -mindepth 2 -name "*.txt" # 从第二层开始

# 执行操作
find /path -name "*.tmp" -delete     # 删除找到的文件
find /path -name "*.txt" -exec cat {} \;  # 对每个文件执行命令
find /path -name "*.txt" -exec rm {} +    # 批量执行 (更快)
find /path -type f -exec chmod 644 {} \;  # 修改权限

# 排除目录
find /path -path /path/exclude -prune -o -name "*.txt" -print

# 实用示例
# 查找并删除空目录
find /path -type d -empty -delete

# 查找最近修改的文件 (top 10)
find /var/log -type f -mtime -7 -ls | sort -k 8,9 -r | head -10

# 查找并压缩大文件
find /var/log -type f -size +100M -exec gzip {} \;

# 查找重复文件 (按 MD5)
find /path -type f -exec md5sum {} \; | sort | uniq -w32 -dD

# ============================================================
#                   locate 命令 (更快)
# ============================================================

# 安装
sudo apt install mlocate             # Debian/Ubuntu
sudo yum install mlocate             # CentOS

# 更新数据库
sudo updatedb                        # 每天自动运行 (cron)

# 搜索文件
locate filename                      # 搜索文件名
locate -i filename                   # 忽略大小写
locate -r "\.txt$"                   # 正则表达式
locate -e filename                   # 只显示存在的文件
locate -c filename                   # 统计数量

# 限制结果数量
locate -l 10 filename                # 只显示前10个

# 注意:
# - locate 使用预建的数据库,速度快
# - 数据库可能不是最新的
# - 找不到最近创建的文件时,运行 updatedb
```

## 五、文件系统维护

### 文件系统检查与修复

```bash
# ============================================================
#                   fsck (文件系统检查)
# ============================================================

# 注意: fsck 必须在卸载的文件系统上运行,或以只读方式挂载

# 检查文件系统
sudo fsck /dev/sdb1
sudo fsck -t ext4 /dev/sdb1          # 指定文件系统类型

# 自动修复
sudo fsck -y /dev/sdb1               # 自动回答 yes
sudo fsck -a /dev/sdb1               # 自动修复 (不询问)

# 强制检查 (即使文件系统标记为干净)
sudo fsck -f /dev/sdb1

# 检查根文件系统 (需要重启)
# 1. 创建强制检查标记
sudo touch /forcefsck
# 2. 重启
sudo reboot

# 或者在启动时按 'e' 进入 GRUB 编辑,在内核参数中添加 fsck.mode=force

# ============================================================
#                   tune2fs (ext 文件系统调优)
# ============================================================

# 查看文件系统信息
sudo tune2fs -l /dev/sdb1

# 设置卷标
sudo tune2fs -L mydisk /dev/sdb1

# 设置检查间隔
sudo tune2fs -i 30 /dev/sdb1         # 30天
sudo tune2fs -i 0 /dev/sdb1          # 禁用基于时间的检查

# 设置挂载次数检查
sudo tune2fs -c 50 /dev/sdb1         # 挂载50次后检查
sudo tune2fs -c 0 /dev/sdb1          # 禁用基于次数的检查

# 设置保留块比例 (root 用户保留)
sudo tune2fs -m 1 /dev/sdb1          # 保留 1% (默认 5%)

# 启用日志功能
sudo tune2fs -j /dev/sdb1            # ext2 升级为 ext3

# ============================================================
#                   xfs 维护
# ============================================================

# 检查 XFS 文件系统
sudo xfs_repair /dev/sdb1            # 修复
sudo xfs_repair -n /dev/sdb1         # 只检查不修复

# XFS 信息
sudo xfs_info /dev/sdb1
sudo xfs_admin -l /dev/sdb1          # 显示卷标
sudo xfs_admin -L newlabel /dev/sdb1 # 设置卷标

# XFS 磁盘配额
sudo xfs_quota -x -c 'report' /mnt/xfs
```

### 磁盘配额

```bash
# ============================================================
#                   用户磁盘配额 (quota)
# ============================================================

# 安装 (如果未安装)
sudo apt install quota               # Debian/Ubuntu
sudo yum install quota               # CentOS

# 1. 编辑 /etc/fstab,添加 quota 选项
# /dev/sdb1  /data  ext4  defaults,usrquota,grpquota  0  2

# 2. 重新挂载
sudo mount -o remount /data

# 3. 创建 quota 文件
sudo quotacheck -cug /data           # -c创建, -u用户, -g组
sudo quotacheck -avugm               # 所有文件系统

# 4. 启用 quota
sudo quotaon -v /data                # 启用
sudo quotaoff -v /data               # 禁用

# 5. 设置用户配额
sudo edquota -u john                 # 编辑用户 john 的配额
sudo edquota -g developers           # 编辑组配额

# quota 文件示例:
# Filesystem  blocks  soft    hard    inodes  soft  hard
# /dev/sdb1   1000    900000  1000000 100     0     0
#             ↓       ↓       ↓
#          当前使用 软限制  硬限制

# 软限制: 可以短期超过,会有警告
# 硬限制: 绝对不能超过

# 6. 设置宽限期
sudo edquota -t                      # 编辑宽限期

# 7. 查看配额
quota -u john                        # 查看用户配额
quota -g developers                  # 查看组配额
sudo repquota -a                     # 所有用户配额报告
sudo repquota /data                  # 特定文件系统

# 8. 复制配额
sudo edquota -p john -u jane         # 将 john 的配额复制给 jane

# ============================================================
#                   XFS 配额
# ============================================================

# XFS 使用内置配额系统

# 1. 挂载时启用
# /dev/sdb1  /data  xfs  defaults,uquota,gquota  0  2

# 2. 设置配额
sudo xfs_quota -x -c 'limit bsoft=900m bhard=1g john' /data
sudo xfs_quota -x -c 'limit isoft=9000 ihard=10000 john' /data

# 3. 查看配额
sudo xfs_quota -x -c 'report -h' /data
sudo xfs_quota -x -c 'quota -u john' /data

# 4. 删除配额
sudo xfs_quota -x -c 'limit bsoft=0 bhard=0 john' /data
```

这是 Linux 文件系统教程,涵盖了磁盘管理、LVM、文件操作、文件系统维护等内容。
