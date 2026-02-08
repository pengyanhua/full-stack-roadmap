# 操作系统安全

## 课程概述

本教程深入讲解操作系统安全机制,从权限控制到SELinux,从沙箱技术到内核安全,帮助你全面理解OS安全防护体系和常见攻击防御方法。

**学习目标**:
- 理解Unix/Linux权限模型
- 掌握SELinux强制访问控制
- 深入了解沙箱隔离技术
- 学习内核安全机制(ASLR、DEP等)
- 理解常见漏洞攻击与防御
- 掌握安全审计与日志分析

---

## 1. 权限控制

### 1.1 Unix/Linux权限模型

```
┌─────────────────────────────────────────────────────────────┐
│              Unix权限模型(DAC - 自主访问控制)                 │
└─────────────────────────────────────────────────────────────┘

文件权限:
┌─────────────────────────────────────────────────────────────┐
│ -rwxr-xr--  1  root  staff  4096  Jan 1 12:00  file.txt    │
│  │││││││││  │   │     │      │      │                      │
│  │││││││││  │   │     │      │      └─ 修改时间            │
│  │││││││││  │   │     │      └──────── 大小                │
│  │││││││││  │   │     └───────────── 组                    │
│  │││││││││  │   └─────────────────── 用户                  │
│  │││││││││  └─────────────────────── 链接数                │
│  ││││││││└────────────────────────── 其他人(r--): 读       │
│  │││││││└─────────────────────────── 组(r-x): 读+执行      │
│  ││││││└──────────────────────────── 用户(rwx): 读+写+执行 │
│  │││││└───────────────────────────── 特殊权限              │
│  ││││└────────────────────────────── 特殊权限              │
│  │││└─────────────────────────────── 特殊权限              │
│  ││└──────────────────────────────── 文件类型:             │
│  │└───────────────────────────────── - 普通文件            │
│  └────────────────────────────────── d 目录                │
│                                      l 符号链接             │
│                                      b 块设备               │
│                                      c 字符设备             │
│                                      p 管道                 │
│                                      s socket              │
└─────────────────────────────────────────────────────────────┘

权限位详解:
┌────────┬──────┬────────────────────────────────────────┐
│ 权限位 │ 八进制│ 含义                                    │
├────────┼──────┼────────────────────────────────────────┤
│ r (读) │  4   │ 文件: 读取内容                          │
│        │      │ 目录: 列出文件(ls)                      │
├────────┼──────┼────────────────────────────────────────┤
│ w (写) │  2   │ 文件: 修改内容                          │
│        │      │ 目录: 创建/删除文件                     │
├────────┼──────┼────────────────────────────────────────┤
│ x (执行)│  1   │ 文件: 执行程序                          │
│        │      │ 目录: 进入目录(cd)                      │
└────────┴──────┴────────────────────────────────────────┘

特殊权限位:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│ 1. SUID (Set User ID) - 4000                                │
│    ┌─────────────────────────────────────────────────┐     │
│    │ -rwsr-xr-x  root  root  /usr/bin/passwd         │     │
│    │    ▲                                            │     │
│    │    └─ s表示SUID                                 │     │
│    │                                                │     │
│    │ 效果: 执行时以文件所有者权限运行                │     │
│    │ 用例: passwd命令需要修改/etc/shadow (root所有)  │     │
│    │                                                │     │
│    │ 安全隐患: 如果程序有漏洞,可能被提权             │     │
│    └─────────────────────────────────────────────────┘     │
│                                                             │
│ 2. SGID (Set Group ID) - 2000                               │
│    ┌─────────────────────────────────────────────────┐     │
│    │ -rwxr-sr-x  root  staff  /usr/bin/wall          │     │
│    │       ▲                                         │     │
│    │       └─ s表示SGID                              │     │
│    │                                                │     │
│    │ 文件: 执行时以文件所属组权限运行                │     │
│    │ 目录: 新文件继承目录的组                        │     │
│    └─────────────────────────────────────────────────┘     │
│                                                             │
│ 3. Sticky Bit - 1000                                        │
│    ┌─────────────────────────────────────────────────┐     │
│    │ drwxrwxrwt  root  root  /tmp                    │     │
│    │         ▲                                       │     │
│    │         └─ t表示Sticky Bit                      │     │
│    │                                                │     │
│    │ 效果: 目录中的文件只能被所有者删除              │     │
│    │ 用例: /tmp目录,防止用户删除他人文件             │     │
│    └─────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘

权限检查流程:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  用户访问文件                                               │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────┐                                           │
│  │ UID == 0?   │ ──Yes─▶ 允许(root绕过检查)                │
│  └──────┬──────┘                                           │
│         │ No                                                │
│         ▼                                                   │
│  ┌─────────────┐                                           │
│  │UID==Owner? │ ──Yes─▶ 检查User权限位                     │
│  └──────┬──────┘                                           │
│         │ No                                                │
│         ▼                                                   │
│  ┌─────────────┐                                           │
│  │GID==Group? │ ──Yes─▶ 检查Group权限位                    │
│  └──────┬──────┘                                           │
│         │ No                                                │
│         ▼                                                   │
│  ┌─────────────┐                                           │
│  │ 检查Other   │                                           │
│  │ 权限位      │                                           │
│  └─────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 访问控制列表(ACL)

```
┌─────────────────────────────────────────────────────────────┐
│              ACL (Access Control List) - 扩展权限            │
└─────────────────────────────────────────────────────────────┘

传统权限的局限性:
• 只能为一个用户和一个组设置权限
• 无法精细控制多个用户

ACL解决方案:
┌─────────────────────────────────────────────────────────────┐
│ 文件: project_file                                          │
│                                                             │
│ 传统权限:                                                    │
│ -rw-r-----  alice  developers  project_file                 │
│                                                             │
│ ACL扩展:                                                     │
│ user::rw-           # 所有者alice: 读写                     │
│ user:bob:r--        # 用户bob: 只读                         │
│ user:charlie:rw-    # 用户charlie: 读写                     │
│ group::r--          # 组developers: 只读                    │
│ group:testers:r--   # 组testers: 只读                       │
│ mask::rw-           # 最大权限掩码                          │
│ other::---          # 其他人: 无权限                        │
└─────────────────────────────────────────────────────────────┘

ACL命令:
┌────────────────────────────────────────────────┐
│ # 查看ACL                                       │
│ getfacl file.txt                                │
│                                                 │
│ # 设置ACL                                       │
│ setfacl -m u:bob:r file.txt                     │
│         └─┬─┘ └┬┘ └┬┘ └──┬──┘                  │
│           │    │   │     └─ 文件                │
│           │    │   └─ 权限                      │
│           │    └─ 用户名                        │
│           └─ modify                             │
│                                                 │
│ # 递归设置目录ACL                                │
│ setfacl -R -m u:bob:rx /project                 │
│                                                 │
│ # 设置默认ACL(新文件继承)                        │
│ setfacl -m d:u:bob:rx /project                  │
│                                                 │
│ # 删除ACL                                       │
│ setfacl -x u:bob file.txt                       │
│                                                 │
│ # 删除所有ACL                                   │
│ setfacl -b file.txt                             │
└────────────────────────────────────────────────┘
```

---

## 2. SELinux强制访问控制

### 2.1 SELinux架构

```
┌─────────────────────────────────────────────────────────────┐
│              SELinux (Security-Enhanced Linux)               │
└─────────────────────────────────────────────────────────────┘

DAC vs MAC:
┌─────────────────────────────────────────────────────────────┐
│ DAC (自主访问控制)              MAC (强制访问控制)          │
│ ┌─────────────────────┐        ┌─────────────────────┐     │
│ │ • 文件所有者决定权限 │        │ • 系统策略决定权限   │     │
│ │ • root绕过所有检查  │        │ • root也受限制       │     │
│ │ • 灵活但不安全      │        │ • 严格但安全         │     │
│ │ • 默认Unix/Linux    │        │ • SELinux/AppArmor  │     │
│ └─────────────────────┘        └─────────────────────┘     │
└─────────────────────────────────────────────────────────────┘

SELinux三种模式:
┌────────────────────────────────────────────────┐
│ Enforcing (强制)                                │
│ • 执行SELinux策略                               │
│ • 违反策略的操作被拒绝                          │
│ • 记录AVC拒绝日志                               │
│                                                 │
│ Permissive (宽容)                               │
│ • 不执行策略                                    │
│ • 违反策略的操作允许执行                        │
│ • 记录AVC拒绝日志(用于调试)                     │
│                                                 │
│ Disabled (禁用)                                 │
│ • 完全禁用SELinux                               │
│ • 无任何检查和日志                              │
└────────────────────────────────────────────────┘

SELinux上下文(Context):
┌─────────────────────────────────────────────────────────────┐
│ 每个进程和文件都有安全上下文:                                │
│                                                             │
│ user:role:type:level                                        │
│  │    │    │     │                                         │
│  │    │    │     └─ MLS/MCS级别(可选)                      │
│  │    │    └─────── 类型(最重要,Type Enforcement)          │
│  │    └──────────── 角色                                    │
│  └───────────────── SELinux用户                             │
│                                                             │
│ 示例:                                                        │
│ system_u:object_r:httpd_sys_content_t:s0                    │
│                                                             │
│ 查看文件上下文:                                              │
│ $ ls -Z /var/www/html/index.html                            │
│ -rw-r--r--. apache apache unconfined_u:object_r:            │
│                          httpd_sys_content_t:s0             │
│                          index.html                         │
│                                                             │
│ 查看进程上下文:                                              │
│ $ ps auxZ | grep httpd                                      │
│ system_u:system_r:httpd_t:s0 apache 1234 ...                │
└─────────────────────────────────────────────────────────────┘

Type Enforcement (类型强制):
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│ SELinux策略规则:                                             │
│                                                             │
│ allow httpd_t httpd_sys_content_t:file { read getattr };    │
│   │     │          │                │       │              │
│   │     │          │                │       └─ 权限         │
│   │     │          │                └─ 对象类(文件)         │
│   │     │          └─ 目标类型(网页文件)                    │
│   │     └─ 源类型(Apache进程)                               │
│   └─ allow规则                                              │
│                                                             │
│ 解读: 允许httpd_t类型的进程读取httpd_sys_content_t类型的文件│
│                                                             │
│ 场景示例:                                                    │
│ ┌────────────────┐              ┌────────────────┐         │
│ │ Apache进程     │              │ 网页文件       │         │
│ │ httpd_t        │──允许读取──▶ │httpd_sys_      │         │
│ │                │              │content_t       │         │
│ └────────────────┘              └────────────────┘         │
│         │                              ▲                    │
│         │ 拒绝访问                     │                    │
│         ▼                              │                    │
│ ┌────────────────┐              ┌────────────────┐         │
│ │ 用户家目录     │              │ /etc/shadow    │         │
│ │ user_home_t    │              │ shadow_t       │         │
│ └────────────────┘              └────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 SELinux实战

```bash
#!/bin/bash
# SELinux管理脚本

echo "=== SELinux状态 ==="
# 查看当前模式
getenforce

# 查看配置文件
cat /etc/selinux/config

echo ""
echo "=== 切换模式 ==="
# 临时切换(重启后失效)
setenforce 0  # Permissive
setenforce 1  # Enforcing

# 永久切换(修改配置文件)
# sed -i 's/SELINUX=enforcing/SELINUX=permissive/' /etc/selinux/config

echo ""
echo "=== 查看上下文 ==="
# 文件上下文
ls -Z /var/www/html

# 进程上下文
ps auxZ | grep httpd

# 端口上下文
semanage port -l | grep http

echo ""
echo "=== 常见问题排查 ==="

# 场景1: Apache无法访问网页
# 问题: 文件上下文不正确
chcon -t httpd_sys_content_t /var/www/html/index.html

# 或者恢复默认上下文
restorecon -v /var/www/html/index.html

# 场景2: Apache监听非标准端口
# 问题: 端口类型不匹配
semanage port -a -t http_port_t -p tcp 8080

# 场景3: 用户家目录共享
# 设置布尔值
setsebool -P httpd_enable_homedirs on

echo ""
echo "=== 查看AVC拒绝日志 ==="
# 查看最近的拒绝
ausearch -m avc -ts recent

# 查看SELinux日志
grep "SELinux" /var/log/audit/audit.log | tail -20

# 使用audit2why分析
ausearch -m avc -ts recent | audit2why

# 生成允许规则(谨慎使用!)
ausearch -m avc -ts recent | audit2allow -M mypolicy
# semodule -i mypolicy.pp

echo ""
echo "=== SELinux布尔值 ==="
# 查看所有布尔值
getsebool -a | grep httpd

# 常用布尔值:
# httpd_can_network_connect  - Apache连接网络
# httpd_can_sendmail         - Apache发送邮件
# httpd_enable_cgi           - 允许CGI脚本
# httpd_enable_homedirs      - 访问用户家目录
```

```python
#!/usr/bin/env python3
"""
SELinux策略分析工具
"""
import subprocess
import re

def get_selinux_status():
    """获取SELinux状态"""
    result = subprocess.run(['getenforce'],
                          capture_output=True, text=True)
    return result.stdout.strip()

def get_process_context(process_name):
    """获取进程上下文"""
    cmd = f"ps auxZ | grep {process_name} | grep -v grep"
    result = subprocess.run(cmd, shell=True,
                          capture_output=True, text=True)
    if result.stdout:
        lines = result.stdout.strip().split('\n')
        contexts = []
        for line in lines:
            match = re.search(r'(\S+:\S+:\S+:\S+)', line)
            if match:
                contexts.append(match.group(1))
        return contexts
    return []

def get_file_context(filepath):
    """获取文件上下文"""
    result = subprocess.run(['ls', '-Z', filepath],
                          capture_output=True, text=True)
    if result.returncode == 0:
        match = re.search(r'(\S+:\S+:\S+:\S+)', result.stdout)
        if match:
            return match.group(1)
    return None

def check_avc_denials():
    """检查AVC拒绝"""
    cmd = "ausearch -m avc -ts recent 2>/dev/null | grep 'type=AVC'"
    result = subprocess.run(cmd, shell=True,
                          capture_output=True, text=True)
    if result.stdout:
        denials = result.stdout.strip().split('\n')
        return len(denials), denials[:5]  # 返回数量和前5条
    return 0, []

def main():
    print("SELinux Security Analysis")
    print("=" * 50)

    # 检查状态
    status = get_selinux_status()
    print(f"SELinux Status: {status}")

    # 检查关键进程
    for process in ['httpd', 'sshd', 'mysqld']:
        contexts = get_process_context(process)
        if contexts:
            print(f"\n{process} contexts:")
            for ctx in contexts:
                print(f"  {ctx}")

    # 检查关键文件
    files = ['/var/www/html', '/etc/passwd', '/etc/shadow']
    print("\nFile contexts:")
    for file in files:
        ctx = get_file_context(file)
        if ctx:
            print(f"  {file}: {ctx}")

    # 检查最近的AVC拒绝
    count, denials = check_avc_denials()
    print(f"\nRecent AVC Denials: {count}")
    if denials:
        print("Latest denials:")
        for denial in denials:
            print(f"  {denial[:100]}...")

if __name__ == '__main__':
    main()
```

---

## 3. 沙箱技术

### 3.1 Seccomp沙箱

```
┌─────────────────────────────────────────────────────────────┐
│              Seccomp (Secure Computing Mode)                 │
└─────────────────────────────────────────────────────────────┘

Seccomp模式:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│ 1. Seccomp Strict Mode (严格模式)                           │
│    • 只允许: read, write, exit, sigreturn                   │
│    • 其他系统调用: 进程终止                                  │
│    • 用途: 极端安全场景                                      │
│                                                             │
│ 2. Seccomp Filter Mode (过滤模式)                           │
│    • 使用BPF规则过滤系统调用                                │
│    • 灵活定义允许/拒绝的系统调用                            │
│    • 可以根据参数过滤                                        │
│    • 用途: 现代沙箱(Docker、Chrome等)                       │
└─────────────────────────────────────────────────────────────┘

BPF过滤器示例:
┌─────────────────────────────────────────────────────────────┐
│ 系统调用 ──▶ BPF过滤器 ──▶ 动作                            │
│                                                             │
│ open()   ──▶ [规则检查] ──▶ ALLOW  (允许)                  │
│ execve() ──▶ [规则检查] ──▶ KILL   (杀死进程)              │
│ socket() ──▶ [规则检查] ──▶ ERRNO  (返回错误码)            │
│ read()   ──▶ [规则检查] ──▶ TRACE  (追踪)                  │
└─────────────────────────────────────────────────────────────┘
```

```c
/*
 * Seccomp沙箱示例
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/prctl.h>
#include <linux/seccomp.h>
#include <linux/filter.h>
#include <linux/audit.h>
#include <sys/syscall.h>

/* 安装Seccomp过滤器 */
void install_seccomp_filter(void)
{
    struct sock_filter filter[] = {
        /* 加载架构 */
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 (offsetof(struct seccomp_data, arch))),
        /* 检查是否为x86_64 */
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
        /* 架构不匹配,杀死进程 */
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL),

        /* 加载系统调用号 */
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 (offsetof(struct seccomp_data, nr))),

        /* 允许的系统调用 */
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_read, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_write, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_exit, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_exit_group, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

        /* 默认: 杀死进程 */
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL),
    };

    struct sock_fprog prog = {
        .len = sizeof(filter) / sizeof(filter[0]),
        .filter = filter,
    };

    /* 禁止获取新特权 */
    prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);

    /* 安装过滤器 */
    prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog);
}

int main(void)
{
    printf("Before seccomp: PID = %d\n", getpid());

    /* 安装沙箱 */
    install_seccomp_filter();

    printf("After seccomp: can still write!\n");

    /* 这个调用会被阻止,进程被杀死 */
    printf("Trying to call getpid()...\n");
    getpid();  /* 不允许的系统调用 */

    printf("You won't see this!\n");

    return 0;
}

/*
 * 编译运行:
 * gcc seccomp_demo.c -o seccomp_demo
 * ./seccomp_demo
 *
 * 输出:
 * Before seccomp: PID = 12345
 * After seccomp: can still write!
 * Trying to call getpid()...
 * (进程被杀死)
 */
```

### 3.2 Capability权限分割

```
┌─────────────────────────────────────────────────────────────┐
│              Capabilities - 细粒度特权                        │
└─────────────────────────────────────────────────────────────┘

传统模型的问题:
• root: 拥有所有权限 (危险)
• 普通用户: 权限不足

Capabilities解决方案:
将root权限分割成独立的capability

常用Capabilities:
┌──────────────────────┬────────────────────────────────────┐
│ Capability           │ 权限                                │
├──────────────────────┼────────────────────────────────────┤
│ CAP_CHOWN            │ 改变文件所有者                      │
│ CAP_DAC_OVERRIDE     │ 绕过DAC权限检查                     │
│ CAP_KILL             │ 发送信号给任意进程                  │
│ CAP_NET_ADMIN        │ 网络管理(路由、防火墙)              │
│ CAP_NET_BIND_SERVICE │ 绑定特权端口(<1024)                 │
│ CAP_NET_RAW          │ 使用RAW socket                      │
│ CAP_SETUID           │ 改变进程UID                         │
│ CAP_SETGID           │ 改变进程GID                         │
│ CAP_SYS_ADMIN        │ 系统管理(mount等)                   │
│ CAP_SYS_CHROOT       │ 使用chroot()                        │
│ CAP_SYS_PTRACE       │ 追踪任意进程                        │
│ CAP_SYS_TIME         │ 设置系统时间                        │
└──────────────────────┴────────────────────────────────────┘

应用场景:
┌─────────────────────────────────────────────────────────────┐
│ 示例: nginx监听80端口                                        │
│                                                             │
│ 传统方法:                                                    │
│ • 以root启动                                                │
│ • 绑定80端口后降权                                          │
│ • 但启动时有完整root权限(危险)                              │
│                                                             │
│ Capability方法:                                             │
│ • 只给CAP_NET_BIND_SERVICE权限                              │
│ • 无需root,直接以普通用户启动                               │
│ • 减少攻击面                                                │
│                                                             │
│ setcap cap_net_bind_service=+ep /usr/sbin/nginx            │
│        └────────┬────────┘   └┬┘ └─────┬─────┘             │
│                │              │        └─ 目标文件          │
│                │              └─ +e(Effective) +p(Permitted)│
│                └─ capability名称                            │
└─────────────────────────────────────────────────────────────┘
```

```bash
#!/bin/bash
# Capability管理脚本

echo "=== 查看进程Capabilities ==="
# 查看当前进程
grep Cap /proc/$$/status

# 解码capabilities
capsh --decode=0000003fffffffff

echo ""
echo "=== 文件Capabilities ==="
# 查看文件capabilities
getcap /usr/bin/ping
getcap /usr/sbin/nginx

# 设置capabilities
# setcap cap_net_bind_service=+ep /path/to/program

# 删除capabilities
# setcap -r /path/to/program

echo ""
echo "=== 容器Capabilities ==="
# Docker默认capabilities
docker run --rm alpine sh -c 'grep Cap /proc/1/status'

# 运行特权容器(危险!)
# docker run --privileged ...

# 添加特定capability
# docker run --cap-add=NET_ADMIN ...

# 删除所有capabilities并添加特定的
# docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE ...

echo ""
echo "=== 安全建议 ==="
echo "最小权限原则:"
echo "• 避免使用--privileged"
echo "• 使用--cap-drop=ALL删除所有权限"
echo "• 只添加必需的权限"
echo "• 定期审计容器权限"
```

---

## 4. 内核安全机制

### 4.1 内存保护

```
┌─────────────────────────────────────────────────────────────┐
│              内存保护机制                                     │
└─────────────────────────────────────────────────────────────┘

1. ASLR (Address Space Layout Randomization)
   ┌──────────────────────────────────────────────────────┐
   │ 无ASLR:              有ASLR:                          │
   │ ┌────────────┐      ┌────────────┐ (每次随机)        │
   │ │ Stack      │      │ Stack      │                   │
   │ │ 0x7fff0000 │      │ 0x7f8a1000 │ ← 随机地址        │
   │ ├────────────┤      ├────────────┤                   │
   │ │ Heap       │      │ Heap       │                   │
   │ │ 0x08000000 │      │ 0x55a21000 │ ← 随机地址        │
   │ ├────────────┤      ├────────────┤                   │
   │ │ libc.so    │      │ libc.so    │                   │
   │ │ 0xb7e00000 │      │ 0x7f921000 │ ← 随机地址        │
   │ ├────────────┤      ├────────────┤                   │
   │ │ Program    │      │ Program    │                   │
   │ │ 0x08048000 │      │ 0x5649a000 │ ← 随机地址(PIE)   │
   │ └────────────┘      └────────────┘                   │
   │                                                      │
   │ 攻击者可预测地址     攻击者无法预测地址              │
   └──────────────────────────────────────────────────────┘

   配置ASLR:
   ┌────────────────────────────────────────────────┐
   │ # 查看ASLR状态                                  │
   │ cat /proc/sys/kernel/randomize_va_space         │
   │                                                 │
   │ # 值的含义:                                     │
   │ 0 - 禁用ASLR                                    │
   │ 1 - 随机化mmap, stack, VDSO页                  │
   │ 2 - 随机化mmap, stack, VDSO页, heap (推荐)     │
   │                                                 │
   │ # 临时设置                                      │
   │ echo 2 > /proc/sys/kernel/randomize_va_space    │
   │                                                 │
   │ # 永久设置                                      │
   │ echo "kernel.randomize_va_space=2" >>           │
   │   /etc/sysctl.conf                              │
   └────────────────────────────────────────────────┘

2. DEP/NX (Data Execution Prevention / No-Execute)
   ┌──────────────────────────────────────────────────────┐
   │ 内存页权限:                                           │
   │ ┌────────┬────┬────┬────┬────┐                      │
   │ │ 地址   │ r  │ w  │ x  │ 用途│                      │
   │ ├────────┼────┼────┼────┼────┤                      │
   │ │ .text  │ ✓  │ ✗  │ ✓  │代码│  可读可执行          │
   │ │ .data  │ ✓  │ ✓  │ ✗  │数据│  可读可写不可执行    │
   │ │ .bss   │ ✓  │ ✓  │ ✗  │数据│  可读可写不可执行    │
   │ │ heap   │ ✓  │ ✓  │ ✗  │堆  │  可读可写不可执行    │
   │ │ stack  │ ✓  │ ✓  │ ✗  │栈  │  可读可写不可执行    │
   │ └────────┴────┴────┴────┴────┘                      │
   │                                                      │
   │ 效果: 防止在数据段执行恶意代码                        │
   │ (例如: 栈溢出注入的shellcode无法执行)                │
   │                                                      │
   │ 硬件支持:                                             │
   │ • x86: NX bit (AMD)                                  │
   │ • x86: XD bit (Intel)                                │
   │ • ARM: XN bit                                        │
   └──────────────────────────────────────────────────────┘

3. Stack Canary (栈金丝雀)
   ┌──────────────────────────────────────────────────────┐
   │ 函数调用栈:                                           │
   │                                                      │
   │ 高地址                                                │
   │ ┌────────────────┐                                   │
   │ │ 返回地址       │ ← 攻击目标                        │
   │ ├────────────────┤                                   │
   │ │ 旧EBP          │                                   │
   │ ├────────────────┤                                   │
   │ │ Canary值       │ ← 随机值,函数开始时设置           │
   │ ├────────────────┤                                   │
   │ │ 局部变量       │                                   │
   │ │ char buf[100]  │ ← 如果溢出,会覆盖Canary           │
   │ └────────────────┘                                   │
   │ 低地址                                                │
   │                                                      │
   │ 函数返回前检查:                                       │
   │ if (canary != original_canary) {                     │
   │     abort(); // 检测到栈溢出!                        │
   │ }                                                    │
   │                                                      │
   │ 编译选项:                                             │
   │ gcc -fstack-protector      (保护部分函数)            │
   │ gcc -fstack-protector-all  (保护所有函数)            │
   │ gcc -fno-stack-protector   (禁用,不推荐)             │
   └──────────────────────────────────────────────────────┘

4. RELRO (Relocation Read-Only)
   ┌──────────────────────────────────────────────────────┐
   │ GOT (Global Offset Table)保护:                       │
   │                                                      │
   │ Partial RELRO (部分只读):                            │
   │ • .init_array, .fini_array, .dynamic只读             │
   │ • GOT可写(存在劫持风险)                              │
   │                                                      │
   │ Full RELRO (完全只读):                               │
   │ • 程序启动时解析所有符号                             │
   │ • GOT标记为只读                                      │
   │ • 防止GOT覆写攻击                                    │
   │                                                      │
   │ 编译选项:                                             │
   │ gcc -Wl,-z,relro         (Partial RELRO)            │
   │ gcc -Wl,-z,relro,-z,now  (Full RELRO)               │
   └──────────────────────────────────────────────────────┘
```

### 4.2 检查程序安全特性

```bash
#!/bin/bash
# 检查二进制文件安全特性

check_security() {
    local binary=$1

    echo "Checking security features of: $binary"
    echo "========================================"

    # 1. PIE (Position Independent Executable)
    echo -n "PIE: "
    if readelf -h "$binary" 2>/dev/null | grep -q "Type:.*DYN"; then
        echo "✓ Enabled"
    else
        echo "✗ Disabled"
    fi

    # 2. NX (No-Execute)
    echo -n "NX:  "
    if readelf -l "$binary" 2>/dev/null | grep -q "GNU_STACK.*RWE"; then
        echo "✗ Disabled (stack executable!)"
    elif readelf -l "$binary" 2>/dev/null | grep -q "GNU_STACK.*RW"; then
        echo "✓ Enabled"
    else
        echo "? Unknown"
    fi

    # 3. Stack Canary
    echo -n "Canary: "
    if readelf -s "$binary" 2>/dev/null | grep -q "__stack_chk_fail"; then
        echo "✓ Enabled"
    else
        echo "✗ Disabled"
    fi

    # 4. RELRO
    echo -n "RELRO: "
    if readelf -l "$binary" 2>/dev/null | grep -q "GNU_RELRO"; then
        if readelf -d "$binary" 2>/dev/null | grep -q "BIND_NOW"; then
            echo "✓ Full RELRO"
        else
            echo "⚠ Partial RELRO"
        fi
    else
        echo "✗ No RELRO"
    fi

    # 5. FORTIFY
    echo -n "FORTIFY: "
    if readelf -s "$binary" 2>/dev/null | grep -q "_chk@"; then
        echo "✓ Enabled"
    else
        echo "✗ Disabled"
    fi

    echo ""
}

# 检查系统ASLR
echo "System ASLR: $(cat /proc/sys/kernel/randomize_va_space)"
echo ""

# 检查常见程序
for prog in /bin/bash /usr/bin/python3 /usr/sbin/sshd; do
    if [ -f "$prog" ]; then
        check_security "$prog"
    fi
done

# 或使用checksec工具(需要安装)
# checksec --file=/bin/bash
```

---

## 5. 常见漏洞与防御

### 5.1 缓冲区溢出

```c
/*
 * 缓冲区溢出示例与防御
 */
#include <stdio.h>
#include <string.h>

/* ============= 不安全的代码 ============= */

void vulnerable_function(char *input)
{
    char buffer[64];
    strcpy(buffer, input);  /* 危险! 无边界检查 */
    printf("Buffer: %s\n", buffer);
}

/* 利用:
 * ./program $(python -c 'print "A"*64 + "\xef\xbe\xad\xde"')
 * 覆盖返回地址为0xdeadbeef
 */

/* ============= 安全的代码 ============= */

void safe_function(char *input)
{
    char buffer[64];

    /* 方法1: 使用strncpy */
    strncpy(buffer, input, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0';  /* 确保null终止 */

    /* 方法2: 使用snprintf */
    snprintf(buffer, sizeof(buffer), "%s", input);

    /* 方法3: 检查长度 */
    if (strlen(input) >= sizeof(buffer)) {
        fprintf(stderr, "Input too long!\n");
        return;
    }
    strcpy(buffer, input);

    printf("Buffer: %s\n", buffer);
}

/* 编译选项防御:
 * gcc -fstack-protector-all -D_FORTIFY_SOURCE=2 \
 *     -Wl,-z,relro,-z,now -pie -fPIE
 */

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Usage: %s <string>\n", argv[0]);
        return 1;
    }

    // vulnerable_function(argv[1]);  /* 不安全 */
    safe_function(argv[1]);           /* 安全 */

    return 0;
}
```

### 5.2 权限提升

```bash
#!/bin/bash
# 权限提升防御检查

echo "=== 检查危险的SUID/SGID文件 ==="
# 查找所有SUID文件
find / -perm -4000 -type f 2>/dev/null

# 查找所有SGID文件
find / -perm -2000 -type f 2>/dev/null

# 查找可写的SUID文件(极度危险!)
find / -perm -4000 -writable -type f 2>/dev/null

echo ""
echo "=== 检查sudo配置 ==="
# 查看sudo配置
sudo -l

# 检查sudoers文件权限
ls -l /etc/sudoers

# 危险配置示例:
# user ALL=(ALL) NOPASSWD: ALL  # 无密码sudo,危险!

echo ""
echo "=== 检查世界可写目录 ==="
# 查找世界可写但无sticky bit的目录
find / -type d -perm -0002 ! -perm -1000 2>/dev/null

echo ""
echo "=== 检查定时任务 ==="
# 查看系统定时任务
ls -la /etc/cron.*

# 查看用户定时任务
for user in $(cut -d: -f1 /etc/passwd); do
    crontab -l -u $user 2>/dev/null | grep -v "^#"
done

echo ""
echo "=== 内核提权漏洞检查 ==="
# 检查内核版本
uname -r

# 使用linux-exploit-suggester检查已知漏洞
# https://github.com/mzet-/linux-exploit-suggester
```

---

## 6. 安全审计

```bash
#!/bin/bash
# 安全审计脚本

LOG_FILE="/var/log/security_audit.log"

audit_log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

audit_log "==== 开始安全审计 ===="

# 1. 检查登录日志
audit_log "=== 最近的登录 ==="
last -10

# 失败的登录尝试
audit_log "=== 失败的登录尝试 ==="
grep "Failed password" /var/log/auth.log | tail -20

# 2. 检查sudo使用
audit_log "=== sudo命令历史 ==="
grep "sudo:" /var/log/auth.log | tail -20

# 3. 检查打开的端口
audit_log "=== 监听端口 ==="
netstat -tulpn | grep LISTEN

# 4. 检查运行的服务
audit_log "=== 运行的服务 ==="
systemctl list-units --type=service --state=running

# 5. 检查最近修改的文件
audit_log "=== 最近修改的系统文件 ==="
find /etc -type f -mtime -7 -ls 2>/dev/null

# 6. 检查异常进程
audit_log "=== CPU占用高的进程 ==="
ps aux --sort=-%cpu | head -10

audit_log "=== 内存占用高的进程 ==="
ps aux --sort=-%mem | head -10

# 7. 检查SELinux AVC拒绝
if command -v ausearch &> /dev/null; then
    audit_log "=== SELinux AVC拒绝 ==="
    ausearch -m avc -ts today 2>/dev/null | wc -l
fi

# 8. 检查异常网络连接
audit_log "=== 建立的网络连接 ==="
netstat -nap | grep ESTABLISHED

audit_log "==== 审计完成 ===="
```

---

## 7. 总结

本教程深入讲解了操作系统安全机制:

**核心知识点**:
1. 权限控制: Unix权限、ACL、特殊权限位
2. SELinux: MAC强制访问控制、Type Enforcement
3. 沙箱技术: Seccomp、Capability权限分割
4. 内存保护: ASLR、DEP/NX、Stack Canary、RELRO
5. 漏洞防御: 缓冲区溢出、权限提升
6. 安全审计: 日志分析、入侵检测

**实战技能**:
- 正确配置文件权限和ACL
- 编写SELinux策略
- 使用Seccomp限制系统调用
- 检查二进制文件安全特性
- 进行安全审计和日志分析

**最佳实践**:
1. 最小权限原则
2. 启用所有内核安全特性(ASLR/DEP)
3. 使用SELinux/AppArmor
4. 定期更新系统补丁
5. 使用安全编译选项
6. 定期安全审计
7. 监控异常行为

安全是一个持续的过程,需要多层防御!
