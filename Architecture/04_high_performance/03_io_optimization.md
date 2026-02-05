# I/O 与网络优化

## 一、I/O 模型

### 五种 I/O 模型

```
┌─────────────────────────────────────────────────────────────────┐
│                       Linux I/O 模型                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. 阻塞 I/O (Blocking I/O)                                    │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │  应用     内核                                            │ │
│   │    │                                                      │ │
│   │    │ ──────read()──────▶│                                │ │
│   │    │    (阻塞等待)       │ 等待数据                       │ │
│   │    │                    │ 数据准备好                      │ │
│   │    │                    │ 复制到用户空间                   │ │
│   │    │ ◀─────返回──────── │                                │ │
│   │                                                           │ │
│   │  特点: 简单，但线程阻塞期间无法做其他事                     │ │
│   └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│   2. 非阻塞 I/O (Non-blocking I/O)                              │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │  应用     内核                                            │ │
│   │    │                                                      │ │
│   │    │ ──read()──▶│ EAGAIN                                 │ │
│   │    │ ◀──────────│                                        │ │
│   │    │ ──read()──▶│ EAGAIN                                 │ │
│   │    │ ◀──────────│         (轮询)                         │ │
│   │    │ ──read()──▶│ 数据准备好，返回数据                    │ │
│   │    │ ◀──────────│                                        │ │
│   │                                                           │ │
│   │  特点: CPU 空转轮询，浪费资源                              │ │
│   └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│   3. I/O 多路复用 (select/poll/epoll)                          │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │  应用     内核                                            │ │
│   │    │                                                      │ │
│   │    │ ─select(fds)─▶│                                     │ │
│   │    │   (阻塞)       │ 监控多个 fd                         │ │
│   │    │ ◀─返回就绪fd──│ 某个 fd 就绪                        │ │
│   │    │ ──read()────▶│ 读取数据                             │ │
│   │    │ ◀──返回数据──│                                      │ │
│   │                                                           │ │
│   │  特点: 单线程处理多连接，高并发场景主流方案                 │ │
│   └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│   4. 信号驱动 I/O (SIGIO)                                       │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │  应用注册信号处理器，数据就绪时收到信号                     │ │
│   │  特点: 实际使用较少                                        │ │
│   └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│   5. 异步 I/O (AIO)                                             │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │  应用     内核                                            │ │
│   │    │                                                      │ │
│   │    │ ─aio_read()──▶│ 立即返回                            │ │
│   │    │ ◀─────────────│                                     │ │
│   │    │  (继续执行)    │ 数据准备+复制                       │ │
│   │    │ ◀──信号/回调──│ 完成通知                            │ │
│   │                                                           │ │
│   │  特点: 真正的异步，但 Linux 原生支持有限                   │ │
│   │       (io_uring 是现代 Linux 的解决方案)                  │ │
│   └──────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### epoll vs select/poll

```
┌─────────────────────────────────────────────────────────────────┐
│                   I/O 多路复用对比                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│           │  select      │  poll       │  epoll                │
│   ────────┼──────────────┼─────────────┼──────────────────     │
│   fd 数量  │  1024        │  无限制      │  无限制               │
│   fd 传递  │  每次全量传递  │  每次全量传递 │  只传递一次          │
│   就绪检测 │  O(n) 遍历    │  O(n) 遍历   │  O(1) 回调           │
│   触发方式 │  水平触发      │  水平触发    │  水平/边缘触发        │
│   适用场景 │  连接数少      │  连接数中等  │  高并发服务器         │
│                                                                 │
│   epoll 工作模式:                                                │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ 水平触发 (LT - Level Triggered)                          │  │
│   │ - 只要缓冲区有数据，就一直通知                             │  │
│   │ - 编程简单，但效率略低                                    │  │
│   │                                                          │  │
│   │ 边缘触发 (ET - Edge Triggered)                           │  │
│   │ - 只在状态变化时通知一次                                   │  │
│   │ - 效率高，但必须一次性读完所有数据                         │  │
│   │ - 需要配合非阻塞 I/O 使用                                 │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、网络编程优化

### 1. TCP 优化参数

```bash
# /etc/sysctl.conf

# 连接队列
net.core.somaxconn = 65535                  # listen backlog
net.ipv4.tcp_max_syn_backlog = 65535        # SYN 队列长度

# 端口范围
net.ipv4.ip_local_port_range = 1024 65535   # 可用端口范围

# TIME_WAIT 优化
net.ipv4.tcp_tw_reuse = 1                   # 重用 TIME_WAIT 连接
net.ipv4.tcp_max_tw_buckets = 262144        # TIME_WAIT 最大数量
net.ipv4.tcp_fin_timeout = 30               # FIN_WAIT2 超时

# 缓冲区
net.core.rmem_max = 16777216                # 接收缓冲区最大值
net.core.wmem_max = 16777216                # 发送缓冲区最大值
net.ipv4.tcp_rmem = 4096 87380 16777216     # TCP 接收缓冲区
net.ipv4.tcp_wmem = 4096 65536 16777216     # TCP 发送缓冲区

# Keepalive
net.ipv4.tcp_keepalive_time = 600           # 空闲后开始探测时间
net.ipv4.tcp_keepalive_probes = 3           # 探测次数
net.ipv4.tcp_keepalive_intvl = 15           # 探测间隔

# 拥塞控制
net.ipv4.tcp_congestion_control = bbr       # 使用 BBR 算法

# 应用修改
sysctl -p
```

### 2. HTTP 连接池

```go
// 连接池配置
var httpClient = &http.Client{
    Transport: &http.Transport{
        // 连接池
        MaxIdleConns:        100,              // 最大空闲连接
        MaxIdleConnsPerHost: 10,               // 每个 host 最大空闲连接
        MaxConnsPerHost:     100,              // 每个 host 最大连接数
        IdleConnTimeout:     90 * time.Second, // 空闲连接超时

        // 超时配置
        DialContext: (&net.Dialer{
            Timeout:   30 * time.Second,  // 连接超时
            KeepAlive: 30 * time.Second,  // keepalive 间隔
        }).DialContext,
        TLSHandshakeTimeout:   10 * time.Second,
        ResponseHeaderTimeout: 10 * time.Second,
        ExpectContinueTimeout: 1 * time.Second,

        // 禁用压缩（如果服务端响应已压缩）
        DisableCompression: false,
    },
    Timeout: 30 * time.Second,  // 总超时（包含连接、请求、响应）
}

// 使用示例
func fetchURL(ctx context.Context, url string) ([]byte, error) {
    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
    if err != nil {
        return nil, err
    }

    resp, err := httpClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    return io.ReadAll(resp.Body)
}

// ❌ 坑点: 不关闭 response body 导致连接无法复用
func badFetch(url string) (int, error) {
    resp, err := http.Get(url)
    if err != nil {
        return 0, err
    }
    // 忘记 resp.Body.Close()
    // 连接不会被放回连接池
    return resp.StatusCode, nil
}

// ✅ 正确: 即使不需要 body 也要关闭
func goodFetch(url string) (int, error) {
    resp, err := http.Get(url)
    if err != nil {
        return 0, err
    }
    defer resp.Body.Close()
    io.Copy(io.Discard, resp.Body)  // 丢弃内容，但必须读完
    return resp.StatusCode, nil
}
```

### 3. gRPC 连接优化

```go
import (
    "google.golang.org/grpc"
    "google.golang.org/grpc/keepalive"
)

// 客户端配置
func NewGRPCClient(addr string) (*grpc.ClientConn, error) {
    return grpc.Dial(addr,
        grpc.WithInsecure(),

        // 连接池 (grpc 默认复用连接)
        grpc.WithDefaultServiceConfig(`{
            "loadBalancingConfig": [{"round_robin":{}}]
        }`),

        // Keepalive 配置
        grpc.WithKeepaliveParams(keepalive.ClientParameters{
            Time:                10 * time.Second,  // 发送 ping 间隔
            Timeout:             3 * time.Second,   // ping 超时
            PermitWithoutStream: true,              // 无流时也发 ping
        }),

        // 初始窗口大小
        grpc.WithInitialWindowSize(1 << 20),     // 1MB
        grpc.WithInitialConnWindowSize(1 << 20), // 1MB

        // 消息大小限制
        grpc.WithDefaultCallOptions(
            grpc.MaxCallRecvMsgSize(50*1024*1024),  // 50MB
            grpc.MaxCallSendMsgSize(50*1024*1024),
        ),
    )
}

// 服务端配置
func NewGRPCServer() *grpc.Server {
    return grpc.NewServer(
        // Keepalive 配置
        grpc.KeepaliveParams(keepalive.ServerParameters{
            MaxConnectionIdle:     15 * time.Minute,
            MaxConnectionAge:      30 * time.Minute,
            MaxConnectionAgeGrace: 5 * time.Minute,
            Time:                  5 * time.Minute,
            Timeout:               1 * time.Minute,
        }),

        // 强制客户端 keepalive
        grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{
            MinTime:             5 * time.Second,
            PermitWithoutStream: true,
        }),

        // 并发流限制
        grpc.MaxConcurrentStreams(1000),

        // 消息大小
        grpc.MaxRecvMsgSize(50 * 1024 * 1024),
        grpc.MaxSendMsgSize(50 * 1024 * 1024),
    )
}
```

---

## 三、零拷贝技术

### 1. 传统数据传输

```
┌─────────────────────────────────────────────────────────────────┐
│                    传统文件传输 (4 次拷贝)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   read(fd, buf, len)                                            │
│   ┌────────────────────────────────────────────────────────┐   │
│   │                                                         │   │
│   │   磁盘 ─────▶ 内核缓冲区 ─────▶ 用户缓冲区              │   │
│   │              (Page Cache)       (应用内存)              │   │
│   │                  ↑                   ↓                   │   │
│   │            DMA 拷贝            CPU 拷贝                  │   │
│   │                                                         │   │
│   └────────────────────────────────────────────────────────┘   │
│                                                                 │
│   write(socket, buf, len)                                       │
│   ┌────────────────────────────────────────────────────────┐   │
│   │                                                         │   │
│   │   用户缓冲区 ─────▶ Socket 缓冲区 ─────▶ 网卡           │   │
│   │   (应用内存)       (内核缓冲区)                         │   │
│   │        ↑                   ↓                            │   │
│   │   CPU 拷贝            DMA 拷贝                          │   │
│   │                                                         │   │
│   └────────────────────────────────────────────────────────┘   │
│                                                                 │
│   总计: 4 次拷贝 + 4 次上下文切换                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. sendfile 零拷贝

```
┌─────────────────────────────────────────────────────────────────┐
│                sendfile 零拷贝 (2 次拷贝)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   sendfile(out_fd, in_fd, offset, count)                        │
│                                                                 │
│   ┌────────────────────────────────────────────────────────┐   │
│   │                                                         │   │
│   │   磁盘 ─────▶ 内核缓冲区 ─────▶ Socket 缓冲区 ─▶ 网卡   │   │
│   │              (Page Cache)       (只传描述符)            │   │
│   │        ↑                               ↓                │   │
│   │   DMA 拷贝                         DMA 拷贝             │   │
│   │                                                         │   │
│   │   数据完全在内核空间完成传输，不经过用户空间               │   │
│   │                                                         │   │
│   └────────────────────────────────────────────────────────┘   │
│                                                                 │
│   总计: 2 次 DMA 拷贝 + 2 次上下文切换                          │
│   适用: 静态文件服务 (Nginx 使用此技术)                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Go 中的零拷贝

```go
import (
    "io"
    "net"
    "os"
    "syscall"
)

// 方式 1: io.Copy (Go 会自动优化)
func serveFile(conn net.Conn, filepath string) error {
    file, err := os.Open(filepath)
    if err != nil {
        return err
    }
    defer file.Close()

    // Go 会在底层使用 sendfile (如果可能)
    _, err = io.Copy(conn, file)
    return err
}

// 方式 2: 直接使用 syscall
func sendFileDirectly(dst, src *os.File, count int64) (int64, error) {
    srcFd := int(src.Fd())
    dstFd := int(dst.Fd())

    var written int64
    var offset int64 = 0

    for written < count {
        n, err := syscall.Sendfile(dstFd, srcFd, &offset, int(count-written))
        if err != nil {
            return written, err
        }
        written += int64(n)
    }
    return written, nil
}

// 方式 3: mmap (内存映射)
func mmapRead(filepath string) ([]byte, error) {
    file, err := os.Open(filepath)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    info, _ := file.Stat()
    size := int(info.Size())

    data, err := syscall.Mmap(
        int(file.Fd()),
        0,
        size,
        syscall.PROT_READ,
        syscall.MAP_SHARED,
    )
    if err != nil {
        return nil, err
    }

    return data, nil
    // 注意: 使用完后需要 syscall.Munmap(data)
}
```

---

## 四、磁盘 I/O 优化

### 1. 顺序写 vs 随机写

```
┌─────────────────────────────────────────────────────────────────┐
│                   磁盘 I/O 性能对比                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    HDD           SSD                            │
│   ────────────────────────────────────────────────────────     │
│   顺序读           150 MB/s      500+ MB/s                      │
│   顺序写           150 MB/s      400+ MB/s                      │
│   随机读 (4K)      0.5 MB/s      50 MB/s                        │
│   随机写 (4K)      0.5 MB/s      30 MB/s                        │
│   IOPS (随机)      100-200       50,000-100,000                 │
│                                                                 │
│   结论: 顺序 I/O 比随机 I/O 快 100-1000 倍                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 日志优化策略

```go
// ❌ 坑点: 每次写入都 Sync
func badWrite(f *os.File, data []byte) error {
    _, err := f.Write(data)
    if err != nil {
        return err
    }
    return f.Sync()  // 每次都刷盘，性能很差
}

// ✅ 策略 1: 批量写入
type BatchWriter struct {
    file      *os.File
    buffer    *bufio.Writer
    mu        sync.Mutex
    batchSize int
    count     int
}

func (w *BatchWriter) Write(data []byte) error {
    w.mu.Lock()
    defer w.mu.Unlock()

    _, err := w.buffer.Write(data)
    if err != nil {
        return err
    }

    w.count++
    if w.count >= w.batchSize {
        w.count = 0
        return w.buffer.Flush()
    }
    return nil
}

// ✅ 策略 2: 定时刷盘
type TimedWriter struct {
    file    *os.File
    buffer  *bufio.Writer
    mu      sync.Mutex
    ticker  *time.Ticker
}

func NewTimedWriter(f *os.File, interval time.Duration) *TimedWriter {
    w := &TimedWriter{
        file:   f,
        buffer: bufio.NewWriter(f),
        ticker: time.NewTicker(interval),
    }

    go func() {
        for range w.ticker.C {
            w.Flush()
        }
    }()

    return w
}

func (w *TimedWriter) Flush() error {
    w.mu.Lock()
    defer w.mu.Unlock()
    return w.buffer.Flush()
}

// ✅ 策略 3: WAL (Write-Ahead Log)
type WAL struct {
    file   *os.File
    buffer *bufio.Writer
    mu     sync.Mutex
}

func (w *WAL) Append(entry []byte) error {
    w.mu.Lock()
    defer w.mu.Unlock()

    // 写入长度前缀
    length := uint32(len(entry))
    binary.Write(w.buffer, binary.LittleEndian, length)

    // 写入数据
    _, err := w.buffer.Write(entry)
    return err
}

func (w *WAL) Sync() error {
    w.mu.Lock()
    defer w.mu.Unlock()

    if err := w.buffer.Flush(); err != nil {
        return err
    }
    return w.file.Sync()  // 确保数据落盘
}
```

### 3. 直接 I/O

```go
// 绕过 Page Cache，直接读写磁盘
// 适用于: 数据库等自己管理缓存的场景

import "golang.org/x/sys/unix"

func openDirectIO(path string) (*os.File, error) {
    return os.OpenFile(path,
        os.O_RDWR|os.O_CREATE|unix.O_DIRECT,  // O_DIRECT 绕过缓存
        0644,
    )
}

// 注意: Direct I/O 要求:
// 1. 缓冲区必须对齐 (通常 512 字节或 4K)
// 2. 读写偏移量必须对齐
// 3. 读写大小必须对齐

// 对齐内存分配
func alignedAlloc(size, alignment int) []byte {
    buf := make([]byte, size+alignment-1)
    offset := alignment - (int(uintptr(unsafe.Pointer(&buf[0]))) % alignment)
    if offset == alignment {
        offset = 0
    }
    return buf[offset : offset+size]
}
```

---

## 五、数据库连接优化

### 1. 连接池配置

```go
import (
    "database/sql"
    _ "github.com/go-sql-driver/mysql"
)

func NewDBPool(dsn string) (*sql.DB, error) {
    db, err := sql.Open("mysql", dsn)
    if err != nil {
        return nil, err
    }

    // 连接池配置
    db.SetMaxOpenConns(100)                 // 最大连接数
    db.SetMaxIdleConns(10)                  // 最大空闲连接
    db.SetConnMaxLifetime(30 * time.Minute) // 连接最大生命周期
    db.SetConnMaxIdleTime(5 * time.Minute)  // 空闲连接超时

    // 验证连接
    if err := db.Ping(); err != nil {
        return nil, err
    }

    return db, nil
}

// 连接池大小计算
// MaxOpenConns = (CPU 核数 * 2) + 有效磁盘数
// 例: 8 核 CPU + 1 块 SSD → MaxOpenConns ≈ 17-20

// 监控连接池状态
func monitorDB(db *sql.DB) {
    go func() {
        ticker := time.NewTicker(10 * time.Second)
        for range ticker.C {
            stats := db.Stats()
            log.Printf("DB Pool: Open=%d, Idle=%d, InUse=%d, WaitCount=%d",
                stats.OpenConnections,
                stats.Idle,
                stats.InUse,
                stats.WaitCount,
            )
        }
    }()
}
```

### 2. 预编译语句

```go
// ❌ 每次都编译 SQL
func badQuery(db *sql.DB, id int64) (*User, error) {
    row := db.QueryRow("SELECT * FROM users WHERE id = ?", id)
    // 每次调用都会编译 SQL
}

// ✅ 使用预编译语句
type UserDAO struct {
    db          *sql.DB
    stmtGetByID *sql.Stmt
    stmtList    *sql.Stmt
}

func NewUserDAO(db *sql.DB) (*UserDAO, error) {
    dao := &UserDAO{db: db}

    var err error
    dao.stmtGetByID, err = db.Prepare("SELECT * FROM users WHERE id = ?")
    if err != nil {
        return nil, err
    }

    dao.stmtList, err = db.Prepare("SELECT * FROM users LIMIT ? OFFSET ?")
    if err != nil {
        return nil, err
    }

    return dao, nil
}

func (dao *UserDAO) GetByID(id int64) (*User, error) {
    row := dao.stmtGetByID.QueryRow(id)  // 复用预编译语句
    // ...
}

func (dao *UserDAO) Close() {
    dao.stmtGetByID.Close()
    dao.stmtList.Close()
}
```

---

## 六、检查清单

### I/O 优化检查

- [ ] 是否选择了合适的 I/O 模型？
- [ ] 是否配置了合理的缓冲区大小？
- [ ] 是否使用了批量写入减少 syscall？
- [ ] 对于大文件是否考虑了零拷贝？

### 网络优化检查

- [ ] TCP 内核参数是否调优？
- [ ] 连接池是否正确配置？
- [ ] 是否复用了 HTTP 连接？
- [ ] 超时配置是否合理？

### 数据库优化检查

- [ ] 连接池大小是否合适？
- [ ] 是否使用了预编译语句？
- [ ] 是否有连接泄漏监控？
- [ ] 慢查询是否有告警？
