# Shell 脚本编程

## 一、Shell 脚本基础

### 什么是 Shell 脚本?

Shell 脚本是包含一系列命令的文本文件,可以被 Shell 解释器执行。

```bash
# 第一个 Shell 脚本
#!/bin/bash
# 这是注释

echo "Hello, World!"
```

### Shebang (释伴行)

```bash
#!/bin/bash              # 使用 Bash
#!/bin/sh                # 使用 sh (POSIX shell)
#!/usr/bin/env bash      # 推荐:在 PATH 中查找 bash
#!/usr/bin/env python3   # Python 脚本
#!/usr/bin/env node      # Node.js 脚本

# 为什么使用 /usr/bin/env?
# - 可移植性更好
# - bash 可能在 /bin/bash 或 /usr/local/bin/bash
# - env 会在 PATH 中查找
```

### 创建并执行脚本

```bash
# 创建脚本文件
cat > hello.sh << 'EOF'
#!/bin/bash
echo "Hello, World!"
EOF

# 方法1: 添加执行权限
chmod +x hello.sh
./hello.sh

# 方法2: 使用 bash 执行
bash hello.sh

# 方法3: 使用 source 或 . 执行 (在当前 shell 中)
source hello.sh
. hello.sh

# 区别:
# - ./hello.sh     : 在子 shell 中执行
# - source hello.sh: 在当前 shell 中执行,可以修改当前环境变量
```

## 二、变量

### 变量定义与使用

```bash
#!/bin/bash

# ============================================================
#                   变量定义
# ============================================================

# 定义变量 (注意:= 两边不能有空格)
name="John"
age=30
is_admin=true

# 使用变量
echo $name
echo ${name}           # 推荐:更清晰
echo "My name is $name and I'm $age years old"

# 只读变量
readonly PI=3.14159
# PI=3.14              # 报错:只读变量不能修改

# 删除变量
unset name

# ============================================================
#                   变量类型
# ============================================================

# 字符串
str1="Hello"
str2='World'
str3="Hello, $name"    # 双引号:变量替换
str4='Hello, $name'    # 单引号:不替换

# 数字
num1=10
num2=20
sum=$((num1 + num2))   # 算术运算
echo $sum              # 30

# 数组
fruits=("apple" "banana" "orange")
echo ${fruits[0]}      # apple
echo ${fruits[@]}      # 所有元素
echo ${#fruits[@]}     # 数组长度

# 关联数组 (Bash 4.0+)
declare -A person
person[name]="John"
person[age]=30
echo ${person[name]}

# ============================================================
#                   特殊变量
# ============================================================

# $0    脚本名称
# $1-$9 位置参数 (第1到第9个参数)
# $#    参数个数
# $@    所有参数 (作为独立字符串)
# $*    所有参数 (作为单个字符串)
# $$    当前进程 PID
# $!    最后一个后台进程 PID
# $?    上一个命令的退出状态 (0=成功)
# $_    上一个命令的最后一个参数

# 示例:
echo "Script name: $0"
echo "First argument: $1"
echo "Number of arguments: $#"
echo "All arguments: $@"
echo "Process ID: $$"
echo "Last exit status: $?"

# ============================================================
#                   变量替换
# ============================================================

# 默认值
echo ${var:-default}   # 如果 var 未设置或为空,返回 default
echo ${var:=default}   # 同上,并将 default 赋值给 var
echo ${var:+value}     # 如果 var 已设置,返回 value

# 长度
str="Hello"
echo ${#str}           # 5

# 子字符串
str="Hello, World!"
echo ${str:7}          # World! (从索引7开始)
echo ${str:7:5}        # World (从索引7开始,长度5)
echo ${str: -6}        # World! (从后往前数6个字符)

# 替换
str="Hello, World!"
echo ${str/World/Bash} # Hello, Bash! (替换第一个)
echo ${str//o/O}       # HellO, WOrld! (替换所有)
echo ${str/#Hello/Hi}  # Hi, World! (替换开头)
echo ${str/%World!/Linux!}  # Hello, Linux! (替换结尾)

# 删除
str="Hello, World!"
echo ${str#Hello, }    # World! (删除开头最短匹配)
echo ${str##*/}        # 从路径中提取文件名
echo ${str%!}          # Hello, World (删除结尾最短匹配)
echo ${str%%,*}        # Hello (删除结尾最长匹配)

# 大小写转换
str="Hello, World!"
echo ${str^^}          # HELLO, WORLD! (全部大写)
echo ${str,,}          # hello, world! (全部小写)
echo ${str^}           # Hello, World! (首字母大写)
```

### 环境变量

```bash
# 查看所有环境变量
env
printenv

# 查看特定环境变量
echo $HOME
echo $PATH
echo $USER

# 设置环境变量
export MY_VAR="value"

# 临时修改 PATH
export PATH=$PATH:/new/path

# 永久设置 (添加到 ~/.bashrc)
echo 'export MY_VAR="value"' >> ~/.bashrc
source ~/.bashrc

# 常用环境变量
$HOME          # 用户家目录
$USER          # 当前用户名
$UID           # 用户 ID
$PATH          # 可执行文件搜索路径
$PWD           # 当前工作目录
$OLDPWD        # 之前的工作目录
$SHELL         # 当前 shell
$TERM          # 终端类型
$LANG          # 语言设置
$EDITOR        # 默认编辑器
$HOSTNAME      # 主机名
```

## 三、运算符

### 算术运算

```bash
#!/bin/bash

# ============================================================
#                   算术运算
# ============================================================

# 方法1: (( )) 算术扩展 (推荐)
a=10
b=20

c=$((a + b))        # 加法: 30
d=$((a - b))        # 减法: -10
e=$((a * b))        # 乘法: 200
f=$((b / a))        # 除法: 2
g=$((b % a))        # 取模: 0
h=$((a ** 2))       # 幂运算: 100

# 自增自减
((a++))             # a = 11
((a--))             # a = 10
((a += 5))          # a = 15
((a -= 3))          # a = 12

# 方法2: let 命令
let "result = a + b"
let "a++"

# 方法3: expr 命令 (旧式,不推荐)
result=$(expr $a + $b)
result=$(expr $a \* $b)    # 需要转义 *

# 方法4: bc 计算器 (支持浮点数)
result=$(echo "scale=2; 10 / 3" | bc)  # 3.33
result=$(echo "scale=2; sqrt(100)" | bc)  # 10.00

# ============================================================
#                   比较运算
# ============================================================

# 数字比较 (使用 (( )) 或 [ ])
# 在 (( )) 中:
if ((a > b)); then
    echo "a > b"
fi

# 在 [ ] 中:
if [ $a -gt $b ]; then
    echo "a > b"
fi

# 数字比较运算符 ([ ] 中使用):
# -eq   等于
# -ne   不等于
# -gt   大于
# -ge   大于等于
# -lt   小于
# -le   小于等于

# (( )) 中可以使用:
# >  <  >=  <=  ==  !=

# ============================================================
#                   字符串比较
# ============================================================

str1="abc"
str2="xyz"

# 相等
if [ "$str1" = "$str2" ]; then
    echo "Equal"
fi

# 不等
if [ "$str1" != "$str2" ]; then
    echo "Not equal"
fi

# 字符串长度为0
if [ -z "$str1" ]; then
    echo "Empty"
fi

# 字符串长度不为0
if [ -n "$str1" ]; then
    echo "Not empty"
fi

# 字典序比较
if [[ "$str1" < "$str2" ]]; then
    echo "str1 < str2"
fi

# ============================================================
#                   逻辑运算
# ============================================================

# 与 (AND)
if [ $a -gt 0 ] && [ $a -lt 100 ]; then
    echo "0 < a < 100"
fi

if [ $a -gt 0 -a $a -lt 100 ]; then
    echo "0 < a < 100"
fi

# 或 (OR)
if [ $a -lt 0 ] || [ $a -gt 100 ]; then
    echo "a < 0 or a > 100"
fi

if [ $a -lt 0 -o $a -gt 100 ]; then
    echo "a < 0 or a > 100"
fi

# 非 (NOT)
if [ ! -f file.txt ]; then
    echo "file.txt does not exist"
fi

# ============================================================
#                   文件测试
# ============================================================

# 文件存在性
[ -e file.txt ]     # 文件存在
[ -f file.txt ]     # 是普通文件
[ -d directory ]    # 是目录
[ -L symlink ]      # 是符号链接
[ -b /dev/sda ]     # 是块设备
[ -c /dev/tty ]     # 是字符设备
[ -p pipe ]         # 是管道
[ -S socket ]       # 是 socket

# 文件权限
[ -r file.txt ]     # 可读
[ -w file.txt ]     # 可写
[ -x file.txt ]     # 可执行
[ -u file.txt ]     # 有 SUID
[ -g file.txt ]     # 有 SGID
[ -k file.txt ]     # 有 Sticky bit

# 文件属性
[ -s file.txt ]     # 文件大小 > 0
[ -t 1 ]            # 文件描述符1(stdout)是终端

# 文件比较
[ file1 -nt file2 ] # file1 比 file2 新
[ file1 -ot file2 ] # file1 比 file2 旧
[ file1 -ef file2 ] # 是同一个文件 (硬链接)
```

## 四、控制结构

### if 语句

```bash
#!/bin/bash

# ============================================================
#                   if 语句
# ============================================================

# 基本 if
if [ condition ]; then
    echo "True"
fi

# if-else
if [ condition ]; then
    echo "True"
else
    echo "False"
fi

# if-elif-else
if [ condition1 ]; then
    echo "Condition 1"
elif [ condition2 ]; then
    echo "Condition 2"
else
    echo "Default"
fi

# 示例:判断文件是否存在
if [ -f "file.txt" ]; then
    echo "File exists"
else
    echo "File does not exist"
fi

# 示例:判断目录是否存在
if [ ! -d "backup" ]; then
    mkdir backup
fi

# 示例:判断字符串
read -p "Enter your name: " name
if [ -z "$name" ]; then
    echo "Name is empty"
elif [ "$name" = "admin" ]; then
    echo "Welcome, admin!"
else
    echo "Hello, $name"
fi

# 示例:判断数字
read -p "Enter a number: " num
if [ $num -gt 0 ]; then
    echo "Positive"
elif [ $num -lt 0 ]; then
    echo "Negative"
else
    echo "Zero"
fi

# ============================================================
#                   [ ] vs [[ ]]
# ============================================================

# [ ]  : POSIX 标准,可移植性好
# [[ ]]: Bash 扩展,功能更强大

# [[ ]] 的优势:
# 1. 支持正则表达式
if [[ "$str" =~ ^[0-9]+$ ]]; then
    echo "String is a number"
fi

# 2. 支持 && 和 ||
if [[ $a -gt 0 && $a -lt 100 ]]; then
    echo "0 < a < 100"
fi

# 3. 不需要引号保护变量
if [[ $str = "hello" ]]; then    # 不需要 "$str"
    echo "Hello"
fi

# 4. 支持模式匹配
if [[ "$file" == *.txt ]]; then
    echo "Text file"
fi
```

### case 语句

```bash
#!/bin/bash

# ============================================================
#                   case 语句
# ============================================================

# 基本语法
case $variable in
    pattern1)
        commands
        ;;
    pattern2)
        commands
        ;;
    *)
        default commands
        ;;
esac

# 示例:菜单选择
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo "You selected option 1"
        ;;
    2)
        echo "You selected option 2"
        ;;
    3)
        echo "You selected option 3"
        ;;
    *)
        echo "Invalid choice"
        ;;
esac

# 示例:文件类型判断
case $file in
    *.txt)
        echo "Text file"
        ;;
    *.jpg|*.png|*.gif)
        echo "Image file"
        ;;
    *.tar.gz|*.tgz)
        echo "Compressed archive"
        ;;
    *)
        echo "Unknown file type"
        ;;
esac

# 示例:脚本参数处理
case $1 in
    start)
        echo "Starting service..."
        ;;
    stop)
        echo "Stopping service..."
        ;;
    restart)
        echo "Restarting service..."
        ;;
    status)
        echo "Checking status..."
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac

# 示例:多条件匹配
read -p "Continue? (yes/no): " answer
case $answer in
    [Yy]|[Yy][Ee][Ss])
        echo "Continuing..."
        ;;
    [Nn]|[Nn][Oo])
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Please answer yes or no"
        ;;
esac

# 示例:范围匹配
read -p "Enter a grade (A-F): " grade
case $grade in
    [Aa])
        echo "Excellent!"
        ;;
    [Bb])
        echo "Good"
        ;;
    [Cc])
        echo "Average"
        ;;
    [Dd])
        echo "Poor"
        ;;
    [Ff])
        echo "Fail"
        ;;
    *)
        echo "Invalid grade"
        ;;
esac
```

### 循环

```bash
#!/bin/bash

# ============================================================
#                   for 循环
# ============================================================

# 遍历列表
for item in apple banana orange; do
    echo $item
done

# 遍历数组
fruits=("apple" "banana" "orange")
for fruit in "${fruits[@]}"; do
    echo $fruit
done

# 遍历文件
for file in *.txt; do
    echo "Processing $file"
done

# 遍历目录
for dir in */; do
    echo "Directory: $dir"
done

# C 风格 for 循环
for ((i=0; i<10; i++)); do
    echo $i
done

# 遍历命令输出
for user in $(cut -d: -f1 /etc/passwd); do
    echo "User: $user"
done

# 遍历数字序列
for i in {1..10}; do
    echo $i
done

for i in {1..10..2}; do    # 步长为2
    echo $i                # 1 3 5 7 9
done

# seq 命令
for i in $(seq 1 10); do
    echo $i
done

# ============================================================
#                   while 循环
# ============================================================

# 基本 while 循环
count=0
while [ $count -lt 5 ]; do
    echo $count
    ((count++))
done

# 无限循环
while true; do
    echo "Press Ctrl+C to stop"
    sleep 1
done

# 读取文件
while read line; do
    echo $line
done < file.txt

# 读取文件 (更好的方式)
while IFS= read -r line; do
    echo $line
done < file.txt

# 逐行处理命令输出
ps aux | while read line; do
    echo $line
done

# 菜单循环
while true; do
    echo "1. Option 1"
    echo "2. Option 2"
    echo "3. Exit"
    read -p "Enter choice: " choice

    case $choice in
        1) echo "Option 1 selected" ;;
        2) echo "Option 2 selected" ;;
        3) break ;;
        *) echo "Invalid choice" ;;
    esac
done

# ============================================================
#                   until 循环
# ============================================================

# until 循环 (条件为假时执行)
count=0
until [ $count -ge 5 ]; do
    echo $count
    ((count++))
done

# 等待文件出现
until [ -f /tmp/ready ]; do
    echo "Waiting for file..."
    sleep 1
done

# ============================================================
#                   循环控制
# ============================================================

# break: 退出循环
for i in {1..10}; do
    if [ $i -eq 5 ]; then
        break
    fi
    echo $i
done

# continue: 跳过本次循环
for i in {1..10}; do
    if [ $i -eq 5 ]; then
        continue
    fi
    echo $i
done

# break 跳出多层循环
for i in {1..5}; do
    for j in {1..5}; do
        if [ $i -eq 3 ] && [ $j -eq 3 ]; then
            break 2    # 跳出两层循环
        fi
        echo "$i-$j"
    done
done

# ============================================================
#                   select 循环
# ============================================================

# select 创建菜单
select option in "Option 1" "Option 2" "Option 3" "Quit"; do
    case $option in
        "Option 1")
            echo "You selected Option 1"
            ;;
        "Option 2")
            echo "You selected Option 2"
            ;;
        "Option 3")
            echo "You selected Option 3"
            ;;
        "Quit")
            break
            ;;
        *)
            echo "Invalid option"
            ;;
    esac
done
```

## 五、函数

```bash
#!/bin/bash

# ============================================================
#                   函数定义
# ============================================================

# 方法1
function my_function() {
    echo "Hello from function"
}

# 方法2 (推荐)
my_function() {
    echo "Hello from function"
}

# 调用函数
my_function

# ============================================================
#                   函数参数
# ============================================================

# 函数参数
greet() {
    echo "Hello, $1!"
    echo "You are $2 years old"
}

greet "John" 30

# 所有参数
print_args() {
    echo "Number of arguments: $#"
    echo "All arguments: $@"

    for arg in "$@"; do
        echo $arg
    done
}

print_args arg1 arg2 arg3

# ============================================================
#                   返回值
# ============================================================

# 返回数字 (0-255)
is_even() {
    local num=$1
    if [ $((num % 2)) -eq 0 ]; then
        return 0    # 成功/真
    else
        return 1    # 失败/假
    fi
}

is_even 10
if [ $? -eq 0 ]; then
    echo "Even"
else
    echo "Odd"
fi

# 返回字符串 (通过 echo)
get_name() {
    echo "John Doe"
}

name=$(get_name)
echo "Name: $name"

# ============================================================
#                   局部变量
# ============================================================

my_function() {
    local local_var="I'm local"
    global_var="I'm global"

    echo $local_var
    echo $global_var
}

my_function
# echo $local_var    # 错误:局部变量在函数外不可见
echo $global_var     # 正确:全局变量

# ============================================================
#                   函数示例
# ============================================================

# 检查文件是否存在
file_exists() {
    if [ -f "$1" ]; then
        return 0
    else
        return 1
    fi
}

if file_exists "/etc/passwd"; then
    echo "File exists"
fi

# 创建备份
backup_file() {
    local file=$1
    local backup="${file}.bak.$(date +%Y%m%d_%H%M%S)"

    if [ -f "$file" ]; then
        cp "$file" "$backup"
        echo "Backup created: $backup"
        return 0
    else
        echo "Error: File $file not found"
        return 1
    fi
}

backup_file "config.conf"

# 日志函数
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[$timestamp] [$level] $message" | tee -a /var/log/script.log
}

log "INFO" "Script started"
log "ERROR" "Something went wrong"

# 清理函数 (在脚本退出时执行)
cleanup() {
    echo "Cleaning up..."
    rm -f /tmp/script.$$
}

trap cleanup EXIT

# 进度条函数
progress_bar() {
    local duration=$1
    local width=50

    for ((i=0; i<=duration; i++)); do
        local percentage=$((i * 100 / duration))
        local filled=$((percentage * width / 100))
        local empty=$((width - filled))

        printf "\r["
        printf "%${filled}s" | tr ' ' '='
        printf "%${empty}s" | tr ' ' ' '
        printf "] %3d%%" $percentage

        sleep 0.1
    done
    echo ""
}

progress_bar 50
```

这是 Shell 脚本编程教程的一部分,涵盖了基础语法、变量、运算符、控制结构和函数。
