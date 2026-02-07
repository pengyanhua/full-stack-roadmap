# Linux 常用命令详解

## 一、文件与目录操作

### 基础导航

```bash
# ============================================================
#                   目录操作
# ============================================================

# 显示当前路径
pwd                            # /home/user

# 切换目录
cd /path/to/directory          # 切换到指定目录
cd ~                           # 切换到家目录
cd                             # 同上
cd -                           # 切换到上一次所在目录
cd ..                          # 上级目录
cd ../..                       # 上两级目录
cd ./subdir                    # 当前目录下的子目录

# 列出文件
ls                             # 列出当前目录文件
ls -l                          # 长格式 (详细信息)
ls -la                         # 包括隐藏文件
ls -lh                         # 人类可读的文件大小
ls -lt                         # 按修改时间排序
ls -lS                         # 按文件大小排序
ls -lR                         # 递归列出所有文件
ls -i                          # 显示 inode 号
ls --color=auto                # 彩色输出

# ls -l 输出格式:
# -rw-r--r--  1 user group  1234 Jan 01 10:00 file.txt
#  ↓          ↓  ↓    ↓      ↓       ↓          ↓
# 权限      链接 所有者 组  大小   修改时间    文件名

# 创建目录
mkdir directory                # 创建目录
mkdir -p parent/child/subdir   # 递归创建 (-p 创建父目录)
mkdir -m 755 directory         # 指定权限创建

# 删除目录
rmdir directory                # 删除空目录
rm -r directory                # 递归删除目录及内容
rm -rf directory               # 强制递归删除 (危险!)

# 树形显示目录
tree                           # 需要安装: apt install tree
tree -L 2                      # 只显示2层
tree -d                        # 只显示目录
tree -a                        # 包括隐藏文件
```

### 文件操作

```bash
# ============================================================
#                   文件创建与查看
# ============================================================

# 创建空文件
touch file.txt                 # 创建文件或更新时间戳
touch file1.txt file2.txt      # 创建多个文件

# 查看文件内容
cat file.txt                   # 显示全部内容
cat file1.txt file2.txt        # 连接多个文件
cat -n file.txt                # 显示行号
cat -A file.txt                # 显示所有字符 (包括控制字符)

tac file.txt                   # 倒序显示

# 分页查看
less file.txt                  # 分页查看 (推荐)
more file.txt                  # 分页查看 (less 更强大)

# less 快捷键:
#   Space    - 下一页
#   b        - 上一页
#   /pattern - 搜索
#   n        - 下一个搜索结果
#   N        - 上一个搜索结果
#   q        - 退出
#   G        - 跳到文件末尾
#   g        - 跳到文件开头

# 查看文件头尾
head file.txt                  # 前10行
head -n 20 file.txt            # 前20行
head -c 100 file.txt           # 前100字节

tail file.txt                  # 后10行
tail -n 20 file.txt            # 后20行
tail -f file.txt               # 实时跟踪文件变化 (日志文件常用)
tail -F file.txt               # 跟踪文件,即使文件被重新创建
tail -f file.txt --pid=1234    # 进程结束后停止跟踪

# ============================================================
#                   文件复制、移动、删除
# ============================================================

# 复制文件
cp source.txt dest.txt         # 复制文件
cp file1 file2 dest_dir/       # 复制多个文件到目录
cp -r source_dir/ dest_dir/    # 递归复制目录
cp -p source dest              # 保留属性 (权限、时间戳)
cp -a source dest              # 归档复制 (等同于 -dpR)
cp -i source dest              # 交互式 (覆盖前询问)
cp -u source dest              # 只复制更新的文件
cp -v source dest              # 显示详细信息

# 移动/重命名
mv source.txt dest.txt         # 重命名
mv file1 file2 dest_dir/       # 移动多个文件
mv -i source dest              # 交互式
mv -f source dest              # 强制覆盖
mv -u source dest              # 只移动更新的文件
mv -v source dest              # 显示详细信息

# 删除文件
rm file.txt                    # 删除文件
rm file1 file2 file3           # 删除多个文件
rm -i file.txt                 # 交互式删除 (确认)
rm -f file.txt                 # 强制删除
rm -r directory/               # 递归删除目录
rm -rf directory/              # 强制递归删除 (危险!)
rm -v file.txt                 # 显示详细信息

# 安全删除 (部分发行版有 trash-cli)
sudo apt install trash-cli
trash file.txt                 # 移到回收站
trash-list                     # 查看回收站
trash-restore                  # 恢复文件
trash-empty                    # 清空回收站

# 批量操作
rm *.tmp                       # 删除所有 .tmp 文件
rm -i *.txt                    # 交互式删除所有 .txt 文件
```

### 文件搜索

```bash
# ============================================================
#                   文件内容搜索 (grep)
# ============================================================

# 基本搜索
grep "pattern" file.txt        # 在文件中搜索
grep "error" *.log             # 在多个文件中搜索
grep -r "pattern" /path        # 递归搜索目录
grep -R "pattern" /path        # 同上,但跟踪符号链接

# 常用选项
grep -i "pattern" file         # 忽略大小写
grep -v "pattern" file         # 反向匹配 (不包含)
grep -n "pattern" file         # 显示行号
grep -c "pattern" file         # 统计匹配行数
grep -l "pattern" *.txt        # 只显示文件名
grep -L "pattern" *.txt        # 显示不匹配的文件名
grep -w "word" file            # 全词匹配
grep -x "exact line" file      # 全行匹配

# 上下文显示
grep -A 3 "pattern" file       # 显示匹配行及后3行
grep -B 3 "pattern" file       # 显示匹配行及前3行
grep -C 3 "pattern" file       # 显示匹配行及前后3行

# 正则表达式
grep "^start" file             # 以 start 开头的行
grep "end$" file               # 以 end 结尾的行
grep "a.c" file                # . 匹配任意单个字符
grep "a*b" file                # * 匹配前一个字符0次或多次
grep "[abc]" file              # 匹配 a、b 或 c
grep "[^abc]" file             # 不匹配 a、b、c
grep "\<word\>" file           # 单词边界

# 扩展正则 (egrep 或 grep -E)
grep -E "pattern1|pattern2" file   # 或
grep -E "a+" file                  # + 匹配1次或多次
grep -E "a?" file                  # ? 匹配0次或1次
grep -E "a{3}" file                # 匹配3次
grep -E "a{2,5}" file              # 匹配2到5次

# 实用示例
# 查找 IP 地址
grep -E "[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}" file

# 查找邮箱
grep -E "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}" file

# 排除注释和空行
grep -v "^#" config.conf | grep -v "^$"

# 统计关键词出现次数
grep -o "keyword" file | wc -l

# 高亮显示
grep --color=auto "pattern" file

# 多文件搜索并显示文件名
grep -Hn "pattern" *.txt

# ============================================================
#                   更强大的搜索工具
# ============================================================

# ripgrep (rg) - 现代化搜索工具,速度极快
sudo apt install ripgrep
rg "pattern" /path              # 递归搜索
rg -i "pattern"                 # 忽略大小写
rg -l "pattern"                 # 只显示文件名
rg -t py "pattern"              # 只搜索 Python 文件
rg --no-ignore "pattern"        # 不忽略 .gitignore

# ack - 面向程序员的搜索工具
sudo apt install ack
ack "pattern"                   # 递归搜索
ack --python "pattern"          # 只搜索 Python 文件
ack -l "pattern"                # 只显示文件名
```

## 二、文本处理

### sed 流编辑器

```bash
# ============================================================
#                   sed 基础
# ============================================================

# 替换 (默认只输出,不修改文件)
sed 's/old/new/' file.txt              # 替换每行第一个匹配
sed 's/old/new/g' file.txt             # 替换所有匹配
sed 's/old/new/2' file.txt             # 替换每行第2个匹配
sed 's/old/new/gi' file.txt            # 忽略大小写替换

# 直接修改文件
sed -i 's/old/new/g' file.txt          # Linux
sed -i '' 's/old/new/g' file.txt       # macOS

# 备份并修改
sed -i.bak 's/old/new/g' file.txt      # 创建 .bak 备份

# 行操作
sed -n '10p' file.txt                  # 打印第10行
sed -n '10,20p' file.txt               # 打印第10-20行
sed -n '/pattern/p' file.txt           # 打印匹配行
sed '10d' file.txt                     # 删除第10行
sed '10,20d' file.txt                  # 删除第10-20行
sed '/pattern/d' file.txt              # 删除匹配行

# 插入和追加
sed '10i\new line' file.txt            # 在第10行前插入
sed '10a\new line' file.txt            # 在第10行后追加
sed '/pattern/a\new line' file.txt     # 在匹配行后追加

# 多个操作
sed -e 's/old1/new1/' -e 's/old2/new2/' file.txt
sed 's/old1/new1/; s/old2/new2/' file.txt

# 实用示例
# 删除空行
sed '/^$/d' file.txt

# 删除注释行
sed '/^#/d' file.txt

# 在每行末尾添加内容
sed 's/$/ END/' file.txt

# 在每行开头添加内容
sed 's/^/START /' file.txt

# 替换特定行范围
sed '10,20s/old/new/g' file.txt

# 从文件读取替换规则
sed -f script.sed file.txt

# 替换包含 / 的路径 (使用 | 作为分隔符)
sed 's|/old/path|/new/path|g' file.txt

# 删除 HTML 标签
sed 's/<[^>]*>//g' file.html
```

### awk 文本分析

```bash
# ============================================================
#                   awk 基础
# ============================================================

# awk 基本语法
awk 'pattern {action}' file.txt

# 打印整行
awk '{print}' file.txt
awk '{print $0}' file.txt              # $0 表示整行

# 打印特定列 (默认以空格分隔)
awk '{print $1}' file.txt              # 第一列
awk '{print $1, $3}' file.txt          # 第一列和第三列
awk '{print $NF}' file.txt             # 最后一列
awk '{print $(NF-1)}' file.txt         # 倒数第二列

# 指定分隔符
awk -F: '{print $1}' /etc/passwd       # 以 : 分隔
awk -F',' '{print $1, $3}' data.csv    # CSV 文件
awk 'BEGIN{FS=":"} {print $1}' file    # 在 BEGIN 中设置分隔符

# 输出分隔符
awk -F: '{print $1, $3}' /etc/passwd   # 默认空格分隔
awk -F: 'BEGIN{OFS=","} {print $1, $3}' /etc/passwd  # 逗号分隔

# 条件过滤
awk '$3 > 100' file.txt                # 第三列 > 100
awk '$1 == "root"' /etc/passwd         # 第一列等于 root
awk '$3 >= 1000 && $3 < 2000' file     # 组合条件
awk '/pattern/' file.txt               # 包含 pattern
awk '!/pattern/' file.txt              # 不包含 pattern
awk 'NR==10' file.txt                  # 第10行
awk 'NR>=10 && NR<=20' file.txt        # 第10-20行

# 内置变量
# NR  : 行号 (所有文件的累计行号)
# NF  : 列数
# FNR : 当前文件的行号
# FS  : 输入分隔符 (默认空格)
# OFS : 输出分隔符 (默认空格)
# RS  : 记录分隔符 (默认换行)
# ORS : 输出记录分隔符

# 使用内置变量
awk '{print NR, $0}' file.txt          # 打印行号和内容
awk '{print NF, $0}' file.txt          # 打印列数和内容
awk 'NR%2==0' file.txt                 # 打印偶数行
awk 'NF>5' file.txt                    # 列数大于5的行

# BEGIN 和 END
awk 'BEGIN{print "Start"} {print $1} END{print "End"}' file

# 统计
awk '{sum+=$1} END{print sum}' file    # 求和
awk '{sum+=$1; count++} END{print sum/count}' file  # 平均值
awk 'BEGIN{max=0} {if($1>max) max=$1} END{print max}' file  # 最大值

# 实用示例
# 打印 /etc/passwd 中的用户名和 UID
awk -F: '{print $1, $3}' /etc/passwd

# 找出 UID 大于 1000 的用户
awk -F: '$3 > 1000 {print $1}' /etc/passwd

# 统计文件行数、字数
awk '{lines++; words+=NF} END{print lines, words}' file

# 去重
awk '!seen[$0]++' file.txt

# 打印重复行
awk 'seen[$0]++' file.txt

# 列求和
awk '{sum+=$1} END{print sum}' numbers.txt

# 按列格式化输出
awk '{printf "%-10s %-10s\n", $1, $2}' file

# 多文件处理
awk '{print FILENAME, $0}' file1 file2

# 处理 CSV
awk -F',' 'NR>1 {sum+=$3} END{print sum}' data.csv  # 跳过表头,第三列求和

# 统计 HTTP 访问日志的状态码
awk '{print $9}' access.log | sort | uniq -c

# 计算目录大小总和
du -h /var/log | awk '{sum+=$1} END{print sum}'
```

### 其他文本工具

```bash
# ============================================================
#                   cut - 列提取
# ============================================================

# 按字符位置提取
cut -c 1-10 file.txt                   # 提取1-10字符
cut -c 1,3,5 file.txt                  # 提取第1、3、5个字符
cut -c 5- file.txt                     # 从第5个字符到行尾

# 按分隔符提取
cut -d: -f1 /etc/passwd                # 提取第一列 (: 分隔)
cut -d: -f1,3 /etc/passwd              # 提取第1和第3列
cut -d: -f1-3 /etc/passwd              # 提取第1到第3列
cut -d, -f2 data.csv                   # CSV 文件提取第二列

# 按字节提取
cut -b 1-10 file.txt

# ============================================================
#                   sort - 排序
# ============================================================

# 基本排序
sort file.txt                          # 字典序排序
sort -r file.txt                       # 逆序
sort -u file.txt                       # 去重排序
sort -n file.txt                       # 数字排序
sort -h file.txt                       # 人类可读大小排序 (1K, 1M)
sort -f file.txt                       # 忽略大小写

# 按列排序
sort -k 2 file.txt                     # 按第2列排序
sort -k 2,2 file.txt                   # 只按第2列排序
sort -t: -k3 -n /etc/passwd            # 指定分隔符,按第3列数字排序
sort -k1,1 -k2,2n file.txt             # 先按第1列,再按第2列数字排序

# 稳定排序
sort -s -k2 file.txt                   # 保持相同键的原始顺序

# 实用示例
# 按文件大小排序
ls -lh | sort -k5 -h

# 找出最大的10个文件
du -h /var | sort -rh | head -10

# ============================================================
#                   uniq - 去重
# ============================================================

# 基本去重 (需要先排序)
sort file.txt | uniq                   # 去重
sort file.txt | uniq -c                # 统计重复次数
sort file.txt | uniq -d                # 只显示重复行
sort file.txt | uniq -u                # 只显示不重复行
sort file.txt | uniq -i                # 忽略大小写

# 按列比较
sort file.txt | uniq -f 1              # 跳过第1列
sort file.txt | uniq -w 10             # 只比较前10个字符

# 实用示例
# 统计日志中 IP 出现次数
awk '{print $1}' access.log | sort | uniq -c | sort -rn

# 找出重复的行
sort file.txt | uniq -d

# ============================================================
#                   wc - 统计
# ============================================================

# 统计行数、字数、字节数
wc file.txt                            # 输出: 行数 字数 字节数 文件名
wc -l file.txt                         # 只统计行数
wc -w file.txt                         # 只统计字数
wc -c file.txt                         # 只统计字节数
wc -m file.txt                         # 只统计字符数
wc -L file.txt                         # 最长行的长度

# 统计多个文件
wc *.txt                               # 每个文件统计,最后总计

# 实用示例
# 统计代码行数
find . -name "*.py" -exec wc -l {} + | tail -1

# 统计目录下文件数量
ls -1 | wc -l

# ============================================================
#                   tr - 字符转换
# ============================================================

# 字符替换
tr 'a' 'A' < file.txt                  # 将 a 替换为 A
tr 'a-z' 'A-Z' < file.txt              # 小写转大写
tr 'A-Z' 'a-z' < file.txt              # 大写转小写

# 删除字符
tr -d ' ' < file.txt                   # 删除空格
tr -d '\n' < file.txt                  # 删除换行
tr -d '0-9' < file.txt                 # 删除数字

# 压缩连续字符
tr -s ' ' < file.txt                   # 压缩多个连续空格为一个
tr -s '\n' < file.txt                  # 压缩多个空行为一个

# 实用示例
# 将空格替换为换行
echo "a b c" | tr ' ' '\n'

# 删除 Windows 换行符
tr -d '\r' < dos.txt > unix.txt

# ROT13 加密
tr 'A-Za-z' 'N-ZA-Mn-za-m' < file.txt

# ============================================================
#                   paste - 合并文件
# ============================================================

# 横向合并
paste file1.txt file2.txt              # 并排显示
paste -d: file1.txt file2.txt          # 使用 : 分隔
paste -d'\t' file1.txt file2.txt       # 使用 Tab 分隔

# 将多行转为一行
paste -s file.txt                      # 所有行合并为一行
paste -s -d: file.txt                  # 用 : 分隔

# ============================================================
#                   join - 按列连接文件
# ============================================================

# 连接两个文件 (需要先排序)
sort file1.txt > sorted1.txt
sort file2.txt > sorted2.txt
join sorted1.txt sorted2.txt           # 按第一列连接

# 指定连接列
join -1 2 -2 1 file1 file2             # file1 第2列 = file2 第1列

# 指定分隔符
join -t: file1 file2

# ============================================================
#                   comm - 比较两个排序文件
# ============================================================

# 比较文件 (需要先排序)
comm file1.txt file2.txt
# 输出三列:
#   第1列: 只在 file1 中
#   第2列: 只在 file2 中
#   第3列: 两个文件都有

# 只显示特定列
comm -12 file1.txt file2.txt           # 只显示共同行
comm -23 file1.txt file2.txt           # 只在 file1 中
comm -13 file1.txt file2.txt           # 只在 file2 中
```

## 三、进程管理

```bash
# ============================================================
#                   进程查看
# ============================================================

# 查看进程
ps                             # 当前终端的进程
ps aux                         # 所有进程 (BSD 风格)
ps -ef                         # 所有进程 (System V 风格)
ps -u username                 # 特定用户的进程
ps -C nginx                    # 特定命令的进程

# ps aux 输出说明:
# USER   PID %CPU %MEM    VSZ   RSS TTY STAT START TIME COMMAND
# root     1  0.0  0.1 169416 11484 ?   Ss   10:00 0:01 /sbin/init
#  ↓      ↓   ↓    ↓     ↓     ↓   ↓    ↓      ↓    ↓      ↓
# 用户  进程ID CPU 内存 虚拟 物理 终端 状态 启动时间 CPU时间 命令

# STAT 状态码:
#   R: 运行中
#   S: 可中断睡眠
#   D: 不可中断睡眠
#   Z: 僵尸进程
#   T: 停止
#   s: 会话领导者
#   +: 前台进程
#   <: 高优先级
#   N: 低优先级

# 进程树
pstree                         # 树形显示进程
pstree -p                      # 显示 PID
pstree -u                      # 显示用户
pstree username                # 特定用户的进程树

# 实时监控
top                            # 实时进程监控
top -u username                # 特定用户的进程
htop                           # 更友好的界面 (需安装)

# top 快捷键:
#   q    - 退出
#   k    - 杀死进程
#   r    - 重新设置 nice 值
#   M    - 按内存排序
#   P    - 按 CPU 排序
#   1    - 显示每个 CPU
#   h    - 帮助

# 按 PID 查看进程
ps -p 1234
ps -p 1234,5678,9012           # 多个 PID

# 按内存使用排序
ps aux --sort=-%mem | head     # 内存使用最多的进程
ps aux --sort=-pcpu | head     # CPU 使用最多的进程

# 查看进程详细信息
cat /proc/1234/status          # 进程状态
cat /proc/1234/cmdline         # 启动命令
cat /proc/1234/environ         # 环境变量
ls -l /proc/1234/fd            # 打开的文件描述符

# 查看进程打开的文件
lsof -p 1234                   # 进程打开的文件
lsof /var/log/syslog           # 哪些进程打开了该文件
lsof -i :80                    # 哪些进程监听 80 端口
lsof -u username               # 用户打开的文件

# ============================================================
#                   进程控制
# ============================================================

# 启动进程
command &                      # 后台运行
nohup command &                # 后台运行,忽略 HUP 信号
nohup command > output.log 2>&1 &  # 重定向输出

# 前后台切换
Ctrl+Z                         # 暂停当前进程
bg                             # 将暂停的进程放到后台运行
bg %1                          # 将作业1放到后台
fg                             # 将后台进程调到前台
fg %1                          # 将作业1调到前台

# 查看后台作业
jobs                           # 列出后台作业
jobs -l                        # 显示 PID

# 杀死进程
kill PID                       # 发送 SIGTERM 信号 (15,正常终止)
kill -9 PID                    # 发送 SIGKILL 信号 (9,强制终止)
kill -15 PID                   # 同 kill PID
kill -HUP PID                  # 发送 SIGHUP 信号 (1,重新加载配置)
kill -USR1 PID                 # 用户自定义信号1

# 常用信号:
#   1  SIGHUP   挂起,通常用于重新加载配置
#   2  SIGINT   中断 (Ctrl+C)
#   3  SIGQUIT  退出 (Ctrl+\)
#   9  SIGKILL  强制终止,不能被捕获
#   15 SIGTERM  终止,可以被捕获
#   18 SIGCONT  继续运行
#   19 SIGSTOP  停止,不能被捕获
#   20 SIGTSTP  停止 (Ctrl+Z)

# 按名称杀死进程
killall nginx                  # 杀死所有 nginx 进程
killall -9 nginx               # 强制杀死
killall -u username            # 杀死用户的所有进程

pkill nginx                    # 按模式杀死进程
pkill -9 nginx
pkill -u username              # 杀死用户的进程

# 交互式杀死进程
kill -9 $(ps aux | grep nginx | grep -v grep | awk '{print $2}')

# 优先级调整
nice -n 10 command             # 以优先级10启动 (默认0,-20最高,19最低)
renice 5 -p 1234               # 修改进程优先级
renice 5 -u username           # 修改用户所有进程优先级

# ============================================================
#                   后台任务管理
# ============================================================

# screen - 终端复用
screen                         # 创建新会话
screen -S name                 # 创建命名会话
screen -ls                     # 列出会话
screen -r                      # 恢复会话
screen -r name                 # 恢复指定会话
screen -d -r name              # 强制恢复会话

# screen 快捷键 (Ctrl+A 然后按):
#   c    - 创建新窗口
#   n    - 下一个窗口
#   p    - 上一个窗口
#   d    - 分离会话
#   k    - 杀死当前窗口
#   [    - 进入复制模式

# tmux - 更现代的终端复用器
tmux                           # 创建新会话
tmux new -s name               # 创建命名会话
tmux ls                        # 列出会话
tmux attach -t name            # 附加到会话
tmux kill-session -t name      # 删除会话

# tmux 快捷键 (Ctrl+B 然后按):
#   c    - 创建新窗口
#   n    - 下一个窗口
#   p    - 上一个窗口
#   d    - 分离会话
#   %    - 垂直分割
#   "    - 水平分割
#   方向键 - 切换面板
```

这是 Linux 常用命令教程的第一部分,涵盖了文件操作、文本处理和进程管理的核心命令。
