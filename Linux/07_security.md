# Linux 系统安全

## 一、用户与认证安全

### SSH 安全加固

```bash
# ============================================================
#                   SSH 服务器配置
# ============================================================

# 编辑 SSH 配置文件
sudo nano /etc/ssh/sshd_config

# 推荐配置:
# ────────────────────────────────────────────────────────────
# 基本设置
Port 22                        # 修改默认端口 (可改为其他端口)
PermitRootLogin no             # 禁止 root 直接登录 ✓
MaxAuthTries 3                 # 最多尝试 3 次
MaxSessions 2                  # 最多 2 个会话

# 认证方式
PubkeyAuthentication yes       # 启用公钥认证 ✓
PasswordAuthentication no      # 禁用密码认证 (配置好密钥后) ✓
PermitEmptyPasswords no        # 禁止空密码 ✓
ChallengeResponseAuthentication no

# 限制用户
AllowUsers user1 user2         # 只允许特定用户
# 或
DenyUsers baduser              # 禁止特定用户

# 限制组
AllowGroups sshusers

# 其他安全设置
X11Forwarding no               # 禁用 X11 转发 (除非需要)
PermitUserEnvironment no       # 禁止用户设置环境变量
UsePAM yes                     # 启用 PAM 认证
StrictModes yes                # 严格模式
LoginGraceTime 30              # 登录超时 30 秒
ClientAliveInterval 300        # 5分钟客户端保活
ClientAliveCountMax 2          # 最多 2 次保活失败

# 日志
SyslogFacility AUTH
LogLevel VERBOSE               # 详细日志

# 应用配置
sudo systemctl restart sshd
sudo systemctl restart ssh     # Debian/Ubuntu

# 测试配置 (不要关闭当前会话!)
sudo sshd -t                   # 测试配置文件语法

# ============================================================
#                   SSH 密钥认证
# ============================================================

# 生成 SSH 密钥对 (客户端)
ssh-keygen -t ed25519 -C "your_email@example.com"
# 或使用 RSA 4096 位
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# 生成的密钥位置:
# ~/.ssh/id_ed25519      # 私钥 (保密!)
# ~/.ssh/id_ed25519.pub  # 公钥

# 设置私钥权限
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub

# 复制公钥到服务器
ssh-copy-id user@server
# 或手动复制
cat ~/.ssh/id_ed25519.pub | ssh user@server "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"

# 服务器端设置权限
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys

# 测试密钥登录
ssh user@server

# 禁用密码登录 (确保密钥登录成功后)
sudo nano /etc/ssh/sshd_config
# PasswordAuthentication no
sudo systemctl restart sshd

# ============================================================
#                   SSH 客户端配置
# ============================================================

# 编辑 ~/.ssh/config
Host myserver
    HostName 192.168.1.10
    User myuser
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3

Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_github

# 使用配置
ssh myserver                   # 自动使用配置

# ============================================================
#                   fail2ban - 防止暴力破解
# ============================================================

# 安装 fail2ban
sudo apt install fail2ban      # Debian/Ubuntu
sudo yum install fail2ban      # CentOS

# 配置 fail2ban
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
sudo nano /etc/fail2ban/jail.local

# 基本配置:
[DEFAULT]
bantime  = 1h                  # 封禁时间
findtime = 10m                 # 查找时间窗口
maxretry = 5                   # 最大尝试次数
destemail = admin@example.com  # 通知邮箱
action = %(action_mwl)s        # 发送邮件

[sshd]
enabled  = true
port     = ssh
filter   = sshd
logpath  = /var/log/auth.log   # Debian/Ubuntu
# logpath  = /var/log/secure   # CentOS
maxretry = 3                   # SSH 最多 3 次

# 启动服务
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# 查看状态
sudo fail2ban-client status
sudo fail2ban-client status sshd

# 查看被封禁的 IP
sudo fail2ban-client get sshd banned

# 手动封禁/解封
sudo fail2ban-client set sshd banip 1.2.3.4
sudo fail2ban-client set sshd unbanip 1.2.3.4

# ============================================================
#                   双因素认证 (2FA)
# ============================================================

# 安装 Google Authenticator
sudo apt install libpam-google-authenticator

# 配置
google-authenticator           # 运行配置向导
# 使用手机 App 扫描二维码

# 配置 PAM
sudo nano /etc/pam.d/sshd
# 添加: auth required pam_google_authenticator.so

# 配置 SSHD
sudo nano /etc/ssh/sshd_config
# ChallengeResponseAuthentication yes
# UsePAM yes

sudo systemctl restart sshd
```

### sudo 权限管理

```bash
# ============================================================
#                   sudo 配置
# ============================================================

# 编辑 sudo 配置 (使用 visudo,会检查语法)
sudo visudo

# 基本语法:
# 用户  主机=(执行者) 命令
user1  ALL=(ALL:ALL) ALL       # user1 可以执行任何命令

# 免密码 sudo
user1  ALL=(ALL) NOPASSWD: ALL

# 限制命令
user1  ALL=(ALL) /usr/bin/systemctl restart nginx, /usr/bin/systemctl reload nginx

# 命令别名
Cmnd_Alias SERVICES = /usr/bin/systemctl restart *, /usr/bin/systemctl reload *
user1  ALL=(ALL) SERVICES

# 用户别名
User_Alias ADMINS = user1, user2, user3
ADMINS ALL=(ALL) ALL

# 组权限 (组名前加 %)
%sudo  ALL=(ALL:ALL) ALL       # sudo 组成员可以执行任何命令
%wheel ALL=(ALL:ALL) ALL       # CentOS 默认管理员组

# 添加用户到 sudo 组
sudo usermod -aG sudo username      # Debian/Ubuntu
sudo usermod -aG wheel username     # CentOS

# 日志记录
Defaults logfile="/var/log/sudo.log"
Defaults log_year, log_host, loglinelen=0

# 安全设置
Defaults env_reset                 # 重置环境变量
Defaults secure_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Defaults passwd_timeout=15         # 密码输入超时
Defaults timestamp_timeout=15      # sudo 缓存15分钟

# 查看 sudo 权限
sudo -l                        # 查看当前用户权限
sudo -l -U username            # 查看指定用户权限

# 查看 sudo 日志
sudo cat /var/log/sudo.log
sudo journalctl -u sudo

# ============================================================
#                   su 限制
# ============================================================

# 限制只有 wheel/sudo 组可以使用 su
sudo nano /etc/pam.d/su
# 取消注释: auth required pam_wheel.so use_uid
```

## 二、防火墙

### UFW (简单防火墙)

```bash
# ============================================================
#                   UFW 基础
# ============================================================

# 安装 (Ubuntu 默认已安装)
sudo apt install ufw

# 查看状态
sudo ufw status
sudo ufw status verbose
sudo ufw status numbered       # 显示规则编号

# 启用/禁用防火墙
sudo ufw enable
sudo ufw disable

# 默认策略
sudo ufw default deny incoming  # 拒绝所有入站 (推荐)
sudo ufw default allow outgoing # 允许所有出站 (推荐)

# ============================================================
#                   基本规则
# ============================================================

# 允许端口
sudo ufw allow 22              # SSH
sudo ufw allow 80              # HTTP
sudo ufw allow 443             # HTTPS
sudo ufw allow 3306            # MySQL

# 允许端口范围
sudo ufw allow 6000:6007/tcp

# 指定协议
sudo ufw allow 53/udp          # DNS
sudo ufw allow 80/tcp

# 拒绝端口
sudo ufw deny 25               # SMTP

# 删除规则
sudo ufw delete allow 80
sudo ufw delete 3              # 按编号删除

# 按来源 IP 允许
sudo ufw allow from 192.168.1.100
sudo ufw allow from 192.168.1.0/24
sudo ufw allow from 192.168.1.100 to any port 22

# 按接口
sudo ufw allow in on eth0 to any port 80

# ============================================================
#                   应用配置
# ============================================================

# 查看应用配置
sudo ufw app list

# 允许应用
sudo ufw allow 'Nginx Full'
sudo ufw allow 'OpenSSH'

# 查看应用详情
sudo ufw app info 'Nginx Full'

# 创建自定义应用配置
sudo nano /etc/ufw/applications.d/myapp
# [MyApp]
# title=My Application
# description=My custom application
# ports=8080/tcp

sudo ufw app update myapp
sudo ufw allow myapp

# ============================================================
#                   高级规则
# ============================================================

# 速率限制 (防止暴力破解)
sudo ufw limit ssh             # 限制 SSH 连接速度

# 日志
sudo ufw logging on            # 启用日志
sudo ufw logging low           # 低级别日志
sudo ufw logging medium        # 中级别日志
sudo ufw logging high          # 高级别日志

# 查看日志
sudo tail -f /var/log/ufw.log

# 重置所有规则
sudo ufw reset

# ============================================================
#                   常用配置示例
# ============================================================

# Web 服务器
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw enable

# 数据库服务器 (只允许应用服务器访问)
sudo ufw allow from 192.168.1.10 to any port 3306
sudo ufw allow ssh

# 完全锁定,只允许 SSH
sudo ufw default deny incoming
sudo ufw default deny outgoing
sudo ufw allow out 53          # DNS
sudo ufw allow out 80          # HTTP
sudo ufw allow out 443         # HTTPS
sudo ufw limit ssh
sudo ufw enable
```

### firewalld (CentOS/RHEL)

```bash
# ============================================================
#                   firewalld 基础
# ============================================================

# 安装
sudo yum install firewalld

# 启动服务
sudo systemctl start firewalld
sudo systemctl enable firewalld

# 查看状态
sudo firewall-cmd --state
sudo firewall-cmd --list-all

# ============================================================
#                   区域 (Zone)
# ============================================================

# 查看所有区域
sudo firewall-cmd --get-zones

# 查看默认区域
sudo firewall-cmd --get-default-zone

# 设置默认区域
sudo firewall-cmd --set-default-zone=public

# 查看活跃区域
sudo firewall-cmd --get-active-zones

# 查看区域配置
sudo firewall-cmd --zone=public --list-all

# ============================================================
#                   服务和端口
# ============================================================

# 查看可用服务
sudo firewall-cmd --get-services

# 允许服务 (临时)
sudo firewall-cmd --add-service=http
sudo firewall-cmd --add-service=https

# 允许服务 (永久)
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --reload

# 删除服务
sudo firewall-cmd --remove-service=http
sudo firewall-cmd --permanent --remove-service=http

# 允许端口
sudo firewall-cmd --add-port=8080/tcp
sudo firewall-cmd --permanent --add-port=8080/tcp

# 允许端口范围
sudo firewall-cmd --add-port=6000-6010/tcp

# 删除端口
sudo firewall-cmd --remove-port=8080/tcp

# ============================================================
#                   富规则
# ============================================================

# 允许特定 IP
sudo firewall-cmd --add-rich-rule='rule family="ipv4" source address="192.168.1.100" accept'

# 限制速率
sudo firewall-cmd --add-rich-rule='rule service name=ssh limit value=3/m accept'

# 端口转发
sudo firewall-cmd --add-forward-port=port=80:proto=tcp:toport=8080

# 查看富规则
sudo firewall-cmd --list-rich-rules

# ============================================================
#                   重载配置
# ============================================================

# 重载防火墙
sudo firewall-cmd --reload

# 完全重启
sudo systemctl restart firewalld
```

## 三、文件系统安全

### 文件完整性监控

```bash
# ============================================================
#                   AIDE (高级入侵检测环境)
# ============================================================

# 安装
sudo apt install aide           # Debian/Ubuntu
sudo yum install aide           # CentOS

# 配置 /etc/aide/aide.conf
# 重要目录监控:
/bin/        R+b+sha256
/sbin/       R+b+sha256
/usr/bin/    R+b+sha256
/usr/sbin/   R+b+sha256
/etc/        R+b+sha256

# 规则说明:
# R = p+i+n+u+g+s+m+c+md5+sha256
# p = 权限
# i = inode
# n = 链接数
# u = 用户
# g = 组
# s = 大小
# m = 修改时间
# c = 创建时间

# 初始化数据库
sudo aideinit
# 或
sudo aide --init
sudo mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db

# 检查
sudo aide --check

# 更新数据库
sudo aide --update
sudo mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db

# 定期检查 (cron)
sudo crontab -e
# 每天凌晨 3 点检查
0 3 * * * /usr/bin/aide --check | mail -s "AIDE Report" admin@example.com

# ============================================================
#                   Tripwire
# ============================================================

# 类似 AIDE,功能更强大但配置复杂

# 安装
sudo apt install tripwire

# 初始化
sudo tripwire --init

# 检查
sudo tripwire --check

# 更新策略
sudo tripwire --update-policy policy.txt

# ============================================================
#                   inotify-tools (实时监控)
# ============================================================

# 安装
sudo apt install inotify-tools

# 监控文件变化
inotifywait -m /etc             # 持续监控
inotifywait -r -m /etc          # 递归监控

# 监控特定事件
inotifywait -m -e modify,create,delete /var/www

# 示例脚本:监控并记录
#!/bin/bash
inotifywait -m -r -e modify,create,delete --format '%T %w%f %e' --timefmt '%Y-%m-%d %H:%M:%S' /etc | \
while read timestamp file event; do
    echo "$timestamp $file $event" >> /var/log/file-monitor.log
done
```

### 加密

```bash
# ============================================================
#                   GPG 加密
# ============================================================

# 生成密钥对
gpg --gen-key

# 列出密钥
gpg --list-keys
gpg --list-secret-keys

# 加密文件
gpg -e -r recipient@example.com file.txt
# 生成 file.txt.gpg

# 解密文件
gpg -d file.txt.gpg > file.txt

# 签名文件
gpg --sign file.txt

# 验证签名
gpg --verify file.txt.gpg

# 导出公钥
gpg --export -a recipient@example.com > public.key

# 导入公钥
gpg --import public.key

# ============================================================
#                   OpenSSL 加密
# ============================================================

# 加密文件
openssl enc -aes-256-cbc -salt -in file.txt -out file.txt.enc

# 解密文件
openssl enc -aes-256-cbc -d -in file.txt.enc -out file.txt

# 生成随机密码
openssl rand -base64 32

# ============================================================
#                   LUKS 磁盘加密
# ============================================================

# 加密分区
sudo cryptsetup luksFormat /dev/sdb1

# 打开加密分区
sudo cryptsetup luksOpen /dev/sdb1 encrypted

# 格式化
sudo mkfs.ext4 /dev/mapper/encrypted

# 挂载
sudo mount /dev/mapper/encrypted /mnt/encrypted

# 卸载并关闭
sudo umount /mnt/encrypted
sudo cryptsetup luksClose encrypted

# 自动挂载 (编辑 /etc/crypttab)
encrypted /dev/sdb1 none luks

# 然后在 /etc/fstab 中:
/dev/mapper/encrypted /mnt/encrypted ext4 defaults 0 2
```

## 四、日志与审计

### 日志管理

```bash
# ============================================================
#                   系统日志
# ============================================================

# 主要日志文件 (传统 syslog)
/var/log/syslog          # 系统日志 (Debian/Ubuntu)
/var/log/messages        # 系统日志 (CentOS/RHEL)
/var/log/auth.log        # 认证日志 (Debian/Ubuntu)
/var/log/secure          # 认证日志 (CentOS/RHEL)
/var/log/kern.log        # 内核日志
/var/log/dmesg           # 启动日志
/var/log/boot.log        # 启动日志

# 查看日志
tail -f /var/log/syslog          # 实时跟踪
tail -n 100 /var/log/auth.log    # 最后 100 行
grep "Failed password" /var/log/auth.log  # 搜索失败登录

# ============================================================
#                   journalctl (systemd 日志)
# ============================================================

# 查看所有日志
journalctl

# 实时跟踪
journalctl -f

# 查看特定服务
journalctl -u ssh
journalctl -u nginx

# 按时间过滤
journalctl --since "2024-01-01"
journalctl --since "2024-01-01" --until "2024-01-02"
journalctl --since today
journalctl --since yesterday
journalctl --since "10 minutes ago"

# 按优先级过滤
journalctl -p err           # 错误及以上
journalctl -p warning       # 警告及以上

# 按启动过滤
journalctl -b               # 本次启动
journalctl -b -1            # 上次启动

# 按用户过滤
journalctl _UID=1000

# 显示内核消息
journalctl -k

# 输出格式
journalctl -o json          # JSON 格式
journalctl -o verbose       # 详细格式

# 磁盘使用
journalctl --disk-usage

# 清理日志
sudo journalctl --vacuum-time=7d     # 保留 7 天
sudo journalctl --vacuum-size=500M   # 保留 500M

# ============================================================
#                   日志轮转
# ============================================================

# 日志轮转配置
/etc/logrotate.conf
/etc/logrotate.d/

# 示例配置 (/etc/logrotate.d/myapp)
/var/log/myapp/*.log {
    daily                    # 每天轮转
    missingok                # 文件缺失不报错
    rotate 14                # 保留 14 个备份
    compress                 # 压缩旧日志
    delaycompress            # 延迟压缩
    notifempty               # 空文件不轮转
    create 0640 www-data www-data  # 创建新文件的权限
    sharedscripts
    postrotate
        systemctl reload nginx > /dev/null
    endscript
}

# 手动执行轮转
sudo logrotate -f /etc/logrotate.conf

# 测试配置
sudo logrotate -d /etc/logrotate.conf

# ============================================================
#                   auditd (审计系统)
# ============================================================

# 安装
sudo apt install auditd         # Debian/Ubuntu
sudo yum install audit          # CentOS

# 启动
sudo systemctl enable auditd
sudo systemctl start auditd

# 添加审计规则
sudo auditctl -w /etc/passwd -p wa -k passwd_changes
sudo auditctl -w /etc/shadow -p wa -k shadow_changes
sudo auditctl -w /var/log/auth.log -p wa -k auth_log_changes

# 参数说明:
# -w: 监控路径
# -p: 权限 (r=读, w=写, x=执行, a=属性修改)
# -k: 关键字 (用于搜索)

# 查看规则
sudo auditctl -l

# 搜索审计日志
sudo ausearch -k passwd_changes
sudo ausearch -f /etc/passwd
sudo ausearch -ts today
sudo ausearch -ui 1000          # 用户 ID

# 审计报告
sudo aureport
sudo aureport -au               # 认证报告
sudo aureport -f                # 文件报告
sudo aureport -x                # 可执行文件报告

# 永久规则 (编辑 /etc/audit/rules.d/audit.rules)
-w /etc/passwd -p wa -k passwd_changes
-w /etc/shadow -p wa -k shadow_changes
-w /var/log/auth.log -p wa -k auth_log

# 重载规则
sudo service auditd reload
```

## 五、安全检查清单

```bash
# ============================================================
#                   快速安全检查
# ============================================================

# 1. 检查可疑进程
ps aux | grep -E "(nc|netcat|ncat)" | grep -v grep
ps aux | sort -rn -k 3 | head -10   # CPU 使用最多的进程

# 2. 检查网络连接
ss -tulpn | grep LISTEN             # 监听端口
ss -antp | grep ESTABLISHED         # 已建立连接
lsof -i                             # 所有网络连接

# 3. 检查最近登录
last                                # 登录历史
lastb                               # 失败登录 (需要 root)
who                                 # 当前登录用户
w                                   # 详细登录信息

# 4. 检查 sudo 使用
sudo cat /var/log/auth.log | grep sudo
sudo journalctl -u sudo

# 5. 检查 cron 任务
crontab -l                          # 用户 cron
sudo cat /etc/crontab               # 系统 cron
ls -la /etc/cron.* /var/spool/cron

# 6. 检查 SUID/SGID 文件
sudo find / -perm -4000 -type f 2>/dev/null   # SUID
sudo find / -perm -2000 -type f 2>/dev/null   # SGID

# 7. 检查世界可写文件
sudo find / -perm -002 -type f 2>/dev/null

# 8. 检查无主文件
sudo find / -nouser -o -nogroup 2>/dev/null

# 9. 检查 hosts 文件
cat /etc/hosts

# 10. 检查防火墙状态
sudo ufw status                     # Ubuntu
sudo firewall-cmd --list-all        # CentOS

# 11. 检查开机启动服务
systemctl list-unit-files --state=enabled

# 12. 检查已安装的包
dpkg -l | wc -l                     # 包数量
apt list --installed | wc -l

# 13. 检查系统更新
sudo apt update && apt list --upgradable    # Debian/Ubuntu
sudo yum check-update                       # CentOS

# 14. 检查文件完整性
sudo aide --check                   # 如果安装了 AIDE

# 15. 检查 SELinux/AppArmor 状态
sestatus                            # SELinux (CentOS)
sudo aa-status                      # AppArmor (Ubuntu)
```

这是 Linux 系统安全教程,涵盖了认证、防火墙、加密、审计等核心安全主题。
