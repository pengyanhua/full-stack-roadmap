# Linux 速查手册

## 文件操作

```bash
# 导航
pwd                           # 显示当前目录
cd /path/to/dir              # 切换目录
cd ~                         # 切换到家目录
cd -                         # 切换到上一次目录
cd ..                        # 上级目录

# 列出文件
ls -la                       # 详细列表,包括隐藏文件
ls -lh                       # 人类可读大小
ls -lS                       # 按大小排序
ls -lt                       # 按时间排序
tree -L 2                    # 树形显示(2层)

# 创建/删除
mkdir -p dir/subdir          # 创建目录(含父目录)
touch file.txt               # 创建空文件
rm file.txt                  # 删除文件
rm -rf dir/                  # 递归强制删除目录
rmdir dir/                   # 删除空目录

# 复制/移动
cp source dest               # 复制文件
cp -r source_dir/ dest_dir/  # 递归复制目录
mv source dest               # 移动/重命名

# 查看文件
cat file.txt                 # 显示全部内容
head -n 20 file.txt          # 前20行
tail -n 20 file.txt          # 后20行
tail -f file.txt             # 实时跟踪
less file.txt                # 分页查看
more file.txt                # 分页查看(简单)

# 查找
find /path -name "*.txt"     # 按名称查找
find /path -type f -size +100M  # 查找大于100M的文件
find /path -mtime -7         # 7天内修改的文件
locate filename              # 快速查找(需updatedb)
which command                # 查找命令路径
whereis command              # 查找命令、源码、手册

# 搜索内容
grep "pattern" file          # 搜索文件内容
grep -r "pattern" /path      # 递归搜索
grep -i "pattern" file       # 忽略大小写
grep -v "pattern" file       # 反向匹配
grep -n "pattern" file       # 显示行号
grep -A 3 "pattern" file     # 显示匹配行及后3行
grep -B 3 "pattern" file     # 显示匹配行及前3行
```

## 权限管理

```bash
# 查看权限
ls -l file                   # 查看文件权限
stat file                    # 详细信息

# 修改权限
chmod 644 file               # rw-r--r--
chmod 755 file               # rwxr-xr-x
chmod 700 file               # rwx------
chmod u+x file               # 所有者添加执行权限
chmod g-w file               # 组移除写权限
chmod o=r file               # 其他人只读
chmod -R 755 dir/            # 递归修改目录

# 修改所有者
chown user file              # 修改所有者
chown user:group file        # 修改所有者和组
chown -R user:group dir/     # 递归修改

# 特殊权限
chmod u+s file               # SUID
chmod g+s dir/               # SGID
chmod +t dir/                # Sticky bit

# ACL
getfacl file                 # 查看ACL
setfacl -m u:user:rw file    # 设置用户ACL
setfacl -m g:group:rw file   # 设置组ACL
setfacl -x u:user file       # 删除ACL
setfacl -b file              # 删除所有ACL
```

## 用户管理

```bash
# 用户操作
useradd -m -s /bin/bash user # 创建用户
passwd user                  # 设置密码
usermod -aG group user       # 添加到组
userdel -r user              # 删除用户及家目录
id user                      # 查看用户信息
who                          # 当前登录用户
w                            # 详细登录信息
last                         # 登录历史

# 组操作
groupadd group               # 创建组
groupdel group               # 删除组
groups user                  # 查看用户所属组

# 切换用户
su - user                    # 切换用户
sudo command                 # 以root执行命令
sudo -u user command         # 以指定用户执行
sudo -i                      # 切换到root shell
```

## 进程管理

```bash
# 查看进程
ps aux                       # 所有进程(BSD风格)
ps -ef                       # 所有进程(System V风格)
ps -u user                   # 用户的进程
pstree -p                    # 进程树
top                          # 实时监控
htop                         # 更友好的top(需安装)

# 查找进程
pgrep nginx                  # 按名称查找PID
pidof nginx                  # 查找PID

# 杀死进程
kill PID                     # 发送SIGTERM
kill -9 PID                  # 强制终止
killall nginx                # 按名称杀死
pkill nginx                  # 按模式杀死

# 后台任务
command &                    # 后台运行
Ctrl+Z                       # 暂停当前进程
bg                           # 后台运行
fg                           # 前台运行
jobs                         # 查看后台任务
nohup command &              # 后台运行,忽略HUP信号

# 优先级
nice -n 10 command           # 以nice值10启动
renice 5 -p PID              # 修改优先级
```

## 磁盘管理

```bash
# 查看磁盘
df -h                        # 磁盘使用情况
df -i                        # inode使用情况
du -sh dir/                  # 目录大小
du -h --max-depth=1 /        # 各目录大小
lsblk                        # 块设备列表
lsblk -f                     # 包含文件系统信息
fdisk -l                     # 磁盘分区信息

# 挂载
mount /dev/sdb1 /mnt         # 挂载
umount /mnt                  # 卸载
mount -a                     # 挂载/etc/fstab中的所有

# 分区
fdisk /dev/sdb               # MBR分区
parted /dev/sdb              # GPT分区
mkfs.ext4 /dev/sdb1          # 格式化为ext4
mkfs.xfs /dev/sdb1           # 格式化为xfs

# LVM
pvcreate /dev/sdb1           # 创建物理卷
vgcreate vg0 /dev/sdb1       # 创建卷组
lvcreate -L 20G -n lv0 vg0   # 创建逻辑卷
lvextend -L +10G /dev/vg0/lv0  # 扩展
resize2fs /dev/vg0/lv0       # 扩展文件系统

# 检查修复
fsck /dev/sdb1               # 检查文件系统(需卸载)
tune2fs -l /dev/sdb1         # 查看ext文件系统信息
```

## 网络管理

```bash
# 网络配置
ip addr show                 # 查看IP地址
ip link show                 # 查看网卡
ip route show                # 查看路由表
ip addr add 192.168.1.10/24 dev eth0  # 配置IP
ip link set eth0 up          # 启用网卡
ip route add default via 192.168.1.1  # 添加默认网关

# 连通性测试
ping google.com              # ping测试
ping -c 4 google.com         # ping 4次
traceroute google.com        # 追踪路由
mtr google.com               # 实时追踪
telnet host 80               # 测试端口
nc -zv host 80               # 测试端口

# 端口查看
ss -tulpn                    # 监听端口(推荐)
netstat -tulpn               # 监听端口(旧)
lsof -i :80                  # 查看端口80

# DNS
dig google.com               # DNS查询
nslookup google.com          # DNS查询
host google.com              # DNS查询

# 下载
wget URL                     # 下载文件
wget -c URL                  # 断点续传
curl -O URL                  # 下载文件
curl -I URL                  # 只获取头部

# 防火墙
ufw status                   # UFW状态(Ubuntu)
ufw allow 80                 # 允许端口
ufw deny 80                  # 拒绝端口
firewall-cmd --list-all      # firewalld状态(CentOS)
```

## 系统信息

```bash
# 系统信息
uname -a                     # 所有系统信息
uname -r                     # 内核版本
cat /etc/os-release          # 发行版信息
hostnamectl                  # 主机信息
uptime                       # 运行时间和负载

# 硬件信息
lscpu                        # CPU信息
free -h                      # 内存信息
lsblk                        # 磁盘信息
lspci                        # PCI设备
lsusb                        # USB设备

# 性能监控
top                          # 进程监控
htop                         # 增强的top
vmstat 2                     # 虚拟内存统计
iostat -x 2                  # I/O统计
sar -u 2 5                   # CPU使用率

# 日志
journalctl                   # systemd日志
journalctl -f                # 实时跟踪
journalctl -u service        # 查看服务日志
journalctl --since today     # 今天的日志
tail -f /var/log/syslog      # 跟踪系统日志
```

## 包管理

```bash
# APT (Debian/Ubuntu)
apt update                   # 更新包列表
apt upgrade                  # 升级所有包
apt install package          # 安装包
apt remove package           # 删除包
apt purge package            # 删除包及配置
apt autoremove               # 删除不需要的依赖
apt search keyword           # 搜索包
apt show package             # 查看包信息

# YUM/DNF (CentOS/RHEL)
yum check-update             # 检查更新
yum update                   # 更新所有包
yum install package          # 安装包
yum remove package           # 删除包
yum search keyword           # 搜索包
yum info package             # 查看包信息

# DNF (Fedora/新版CentOS)
dnf update                   # 更新
dnf install package          # 安装
dnf remove package           # 删除
dnf search keyword           # 搜索

# Snap
snap list                    # 已安装的snap
snap find keyword            # 搜索
snap install package         # 安装
snap remove package          # 删除

# 编译安装
./configure                  # 配置
make                         # 编译
sudo make install            # 安装
```

## 服务管理

```bash
# systemctl
systemctl status service     # 查看状态
systemctl start service      # 启动服务
systemctl stop service       # 停止服务
systemctl restart service    # 重启服务
systemctl reload service     # 重新加载配置
systemctl enable service     # 开机自启
systemctl disable service    # 禁用自启
systemctl list-units         # 列出所有单元
systemctl --failed           # 失败的服务

# journalctl
journalctl -u service        # 查看服务日志
journalctl -f                # 实时跟踪日志
journalctl -b                # 本次启动日志
journalctl --since today     # 今天的日志
```

## 文本处理

```bash
# sed
sed 's/old/new/g' file       # 替换
sed -i 's/old/new/g' file    # 直接修改文件
sed -n '10,20p' file         # 打印10-20行
sed '10d' file               # 删除第10行

# awk
awk '{print $1}' file        # 打印第1列
awk -F: '{print $1}' /etc/passwd  # 指定分隔符
awk '$3 > 100' file          # 第3列>100
awk '{sum+=$1} END{print sum}' file  # 求和

# cut
cut -d: -f1 /etc/passwd      # 提取第1列
cut -c 1-10 file             # 提取1-10字符

# sort
sort file                    # 排序
sort -r file                 # 逆序
sort -n file                 # 数字排序
sort -u file                 # 去重排序
sort -k2 file                # 按第2列排序

# uniq
sort file | uniq             # 去重
sort file | uniq -c          # 统计重复
sort file | uniq -d          # 只显示重复行

# wc
wc -l file                   # 统计行数
wc -w file                   # 统计字数
wc -c file                   # 统计字节数

# tr
tr 'a-z' 'A-Z' < file        # 小写转大写
tr -d ' ' < file             # 删除空格
```

## 压缩解压

```bash
# tar
tar -czf archive.tar.gz dir/     # 压缩(gzip)
tar -xzf archive.tar.gz          # 解压(gzip)
tar -cjf archive.tar.bz2 dir/    # 压缩(bzip2)
tar -xjf archive.tar.bz2         # 解压(bzip2)
tar -tf archive.tar.gz           # 查看内容

# gzip/gunzip
gzip file                    # 压缩(删除原文件)
gzip -k file                 # 压缩(保留原文件)
gunzip file.gz               # 解压

# zip/unzip
zip -r archive.zip dir/      # 压缩目录
unzip archive.zip            # 解压
unzip -l archive.zip         # 查看内容

# 7z
7z a archive.7z dir/         # 压缩
7z x archive.7z              # 解压
7z l archive.7z              # 查看内容
```

## SSH

```bash
# 连接
ssh user@host                # SSH连接
ssh -p 2222 user@host        # 指定端口
ssh -i key.pem user@host     # 指定密钥

# 密钥
ssh-keygen -t ed25519        # 生成密钥对
ssh-copy-id user@host        # 复制公钥到服务器

# 文件传输
scp file user@host:/path     # 上传文件
scp user@host:/path/file .   # 下载文件
scp -r dir/ user@host:/path  # 上传目录

# rsync
rsync -av src/ dest/         # 同步目录
rsync -av --delete src/ dest/  # 同步并删除目标多余文件
rsync -av -e ssh src/ user@host:/path  # 通过SSH同步

# 端口转发
ssh -L 8080:localhost:80 user@host  # 本地端口转发
ssh -R 8080:localhost:80 user@host  # 远程端口转发
ssh -D 1080 user@host        # 动态端口转发(SOCKS代理)

# 保持连接
ssh -o ServerAliveInterval=60 user@host
```

## 快捷键

```bash
# 命令行编辑
Ctrl+A                       # 移到行首
Ctrl+E                       # 移到行尾
Ctrl+U                       # 删除光标前内容
Ctrl+K                       # 删除光标后内容
Ctrl+W                       # 删除光标前单词
Ctrl+L                       # 清屏
Ctrl+R                       # 搜索历史命令
Ctrl+C                       # 终止当前命令
Ctrl+D                       # 退出(EOF)
Ctrl+Z                       # 暂停当前程序

# 历史命令
!!                           # 执行上一条命令
!n                           # 执行历史第n条命令
!$                           # 上一条命令的最后参数
!*                           # 上一条命令的所有参数
history                      # 查看历史
```

## 环境变量

```bash
# 查看
env                          # 所有环境变量
echo $PATH                   # 查看PATH

# 设置
export VAR=value             # 临时设置
export PATH=$PATH:/new/path  # 添加到PATH

# 永久设置
~/.bashrc                    # 用户配置(交互shell)
~/.bash_profile              # 用户配置(登录shell)
/etc/profile                 # 系统全局配置
/etc/environment             # 系统环境变量

# 加载配置
source ~/.bashrc             # 重新加载
. ~/.bashrc                  # 同上
```

## 常用技巧

```bash
# 管道和重定向
command1 | command2          # 管道
command > file               # 输出重定向(覆盖)
command >> file              # 输出重定向(追加)
command < file               # 输入重定向
command 2> file              # 错误重定向
command &> file              # 所有输出重定向
command 2>&1                 # 错误重定向到标准输出

# 命令替换
$(command)                   # 命令替换
`command`                    # 命令替换(旧语法)

# 逻辑运算
command1 && command2         # command1成功才执行command2
command1 || command2         # command1失败才执行command2
command1 ; command2          # 依次执行

# 后台运行
command &                    # 后台运行
nohup command &              # 忽略HUP信号
(command &)                  # 子shell后台运行

# 别名
alias ll='ls -lah'           # 创建别名
unalias ll                   # 删除别名
alias                        # 查看所有别名

# 通配符
*                            # 匹配任意字符
?                            # 匹配单个字符
[abc]                        # 匹配a、b或c
[!abc]                       # 不匹配a、b、c
[a-z]                        # 匹配a到z

# xargs
find . -name "*.txt" | xargs rm  # 批量删除
find . -name "*.log" | xargs -I {} cp {} /backup  # 使用占位符
```

## 安全

```bash
# SSH安全
# 编辑 /etc/ssh/sshd_config
PermitRootLogin no           # 禁止root登录
PasswordAuthentication no    # 禁用密码登录
Port 2222                    # 修改端口

# 防火墙
ufw enable                   # 启用UFW
ufw allow 22                 # 允许SSH
ufw allow 80/tcp             # 允许HTTP
ufw deny 25                  # 拒绝SMTP

# fail2ban
sudo systemctl enable fail2ban
sudo fail2ban-client status  # 查看状态

# 更新
sudo apt update && sudo apt upgrade  # 更新系统

# 检查安全
last                         # 登录历史
lastb                        # 失败登录
sudo cat /var/log/auth.log   # 认证日志
```

## 故障排查

```bash
# 系统负载高
top                          # 查看进程
ps aux --sort=-pcpu | head   # CPU使用最多
ps aux --sort=-pmem | head   # 内存使用最多
iotop                        # I/O监控

# 磁盘满
df -h                        # 查看磁盘使用
du -sh /*                    # 各目录大小
du -h /var | sort -rh | head # 最大的目录
find /var -type f -size +100M  # 大文件

# 网络问题
ping gateway                 # 测试网关
traceroute google.com        # 追踪路由
ss -tulpn                    # 查看端口
tcpdump -i eth0              # 抓包

# 服务问题
systemctl status service     # 查看状态
journalctl -u service        # 查看日志
journalctl -xe               # 查看最近错误
dmesg                        # 查看内核日志

# 进程问题
ps aux | grep process        # 查找进程
lsof -p PID                  # 查看进程打开的文件
strace -p PID                # 跟踪系统调用
```

---

## 在线资源

- **man pages**: `man command` - 命令手册
- **help**: `command --help` - 命令帮助
- **tldr**: `tldr command` - 简化的命令示例(需安装)
- **explainshell.com** - 解释shell命令
- **cheat.sh** - 在线速查表

```bash
# 安装 tldr
npm install -g tldr
# 或
pip install tldr

# 使用
tldr tar
tldr grep
```
