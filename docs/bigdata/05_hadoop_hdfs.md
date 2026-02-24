# Hadoop生态与HDFS分布式存储

## 1. Hadoop生态系统概览

### 1.1 发展历史与版本演进

Hadoop起源于Doug Cutting在2005年基于Google发表的GFS和MapReduce论文实现的开源项目。
经过近20年的发展，Hadoop已从单一的批处理框架演变为完整的大数据生态系统。

**版本演进时间线**：

```
Hadoop版本演进
┌──────────────────────────────────────────────────────────────────┐
│  2006        2012         2017         2021                      │
│   │           │            │            │                        │
│   ↓           ↓            ↓            ↓                        │
│ ┌──────┐  ┌──────┐    ┌──────┐    ┌──────┐                      │
│ │ 0.x  │→ │ 1.x  │ →  │ 2.x  │ →  │ 3.x  │                     │
│ └──────┘  └──────┘    └──────┘    └──────┘                      │
│   实验版    稳定版      生产主流     最新稳定                     │
│                                                                  │
│ 0.x: HDFS + MapReduce 原型                                      │
│ 1.x: 单NameNode + JobTracker/TaskTracker                        │
│ 2.x: YARN资源管理 + HA NameNode + Federation                    │
│ 3.x: Erasure Coding + 多NameNode Standby + GPU支持              │
└──────────────────────────────────────────────────────────────────┘
```

**版本对比表**：

| 维度 | Hadoop 1.x | Hadoop 2.x | Hadoop 3.x |
|------|-----------|-----------|-----------|
| **NameNode** | 单点，无HA | 1 Active + 1 Standby | 1 Active + N Standby |
| **资源管理** | JobTracker（MR专用） | YARN（通用资源调度） | YARN增强（GPU/FPGA） |
| **存储冗余** | 3副本 | 3副本 | 3副本 + Erasure Coding |
| **Java版本** | Java 6 | Java 7+ | Java 8+ |
| **容器支持** | 无 | Linux Container | Docker原生支持 |
| **端口** | 固定端口 | 固定端口 | 端口范围重新规划 |
| **Timeline Service** | 无 | v1 (单点) | v2 (分布式，基于HBase) |
| **典型发行版** | CDH3/HDP1 | CDH5/6, HDP2/3 | CDP7, 社区版3.3+ |

### 1.2 核心组件

Hadoop生态系统由三大核心层和众多外围工具组成：

```
Hadoop生态系统全景图
┌──────────────────────────────────────────────────────────────────────┐
│                          应用与接口层                                │
│  ┌──────┐ ┌──────┐ ┌───────┐ ┌──────┐ ┌──────┐ ┌───────┐          │
│  │ Hive │ │ Pig  │ │ Sqoop │ │Flume │ │Kafka │ │ Oozie │          │
│  │SQL查询│ │脚本  │ │数据导入│ │日志采集│ │消息队列│ │工作流  │          │
│  └──┬───┘ └──┬───┘ └───┬───┘ └──┬───┘ └──┬───┘ └───┬───┘          │
├─────┼────────┼─────────┼────────┼────────┼─────────┼────────────────┤
│     ↓        ↓         ↓        ↓        ↓         ↓                │
│                        计算引擎层                                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │ MapReduce  │  │   Spark    │  │   Flink    │  │    Tez     │    │
│  │  批处理     │  │ 内存计算    │  │  流式计算   │  │  DAG引擎   │    │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘    │
├────────┼───────────────┼───────────────┼───────────────┼────────────┤
│        ↓               ↓               ↓               ↓            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    YARN 资源调度层                             │   │
│  │            ResourceManager + NodeManager                      │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
├─────────────────────────────┼────────────────────────────────────────┤
│                             ↓                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    HDFS 分布式存储层                           │   │
│  │            NameNode + DataNode (Block存储)                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────────────┤
│                        协调与辅助服务                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ZooKeeper │  │  HBase   │  │  Ambari  │  │  Ranger  │            │
│  │ 分布式协调│  │ NoSQL DB │  │ 集群管理  │  │ 权限管理  │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
└──────────────────────────────────────────────────────────────────────┘
```

**三大核心模块**：

- **HDFS（Hadoop Distributed File System）**：分布式文件系统，负责海量数据的可靠存储。将大文件切分为Block（默认128MB），分布在集群中的多个DataNode上，并通过副本机制保障数据可靠性。
- **YARN（Yet Another Resource Negotiator）**：资源调度框架，负责集群资源（CPU、内存）的统一管理和作业调度。支持MapReduce、Spark、Flink等多种计算引擎共享集群资源。
- **MapReduce**：分布式计算框架，提供Map（映射）和Reduce（归约）两个编程原语，适合大规模离线批处理。

### 1.3 Hadoop 3.x新特性

**Erasure Coding（纠删码）**：

传统3副本方式存储200%的冗余开销，Erasure Coding通过编码算法将开销降至约50%：

```
存储开销对比
┌──────────────────────────────────────────────────┐
│  3副本策略 (Replication Factor = 3)               │
│  原始数据: 1TB → 实际占用: 3TB (200%冗余)         │
│  ┌──────┐  ┌──────┐  ┌──────┐                    │
│  │副本1 │  │副本2 │  │副本3 │                    │
│  │ 1TB  │  │ 1TB  │  │ 1TB  │                    │
│  └──────┘  └──────┘  └──────┘                    │
├──────────────────────────────────────────────────┤
│  Erasure Coding (RS-6-3)                         │
│  原始数据: 6个数据块 + 3个校验块 (50%冗余)        │
│  ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐           │
│  │ D1 ││ D2 ││ D3 ││ D4 ││ D5 ││ D6 │  数据块   │
│  └────┘└────┘└────┘└────┘└────┘└────┘           │
│  ┌────┐┌────┐┌────┐                              │
│  │ P1 ││ P2 ││ P3 │  校验块                      │
│  └────┘└────┘└────┘                              │
│  任意丢失3个块均可恢复                            │
└──────────────────────────────────────────────────┘
```

**其他3.x重要特性**：

- **多Standby NameNode**：支持超过2个NameNode，可配置1 Active + N Standby，进一步提升可用性
- **YARN GPU/FPGA支持**：YARN资源模型扩展到GPU和FPGA，支持深度学习训练等异构计算任务
- **Timeline Service v2**：基于HBase的分布式存储后端，解决v1单点性能瓶颈
- **端口重新规划**：NameNode默认端口从50070改为9870，DataNode从50075改为9864，避免与Linux临时端口冲突
- **Java 8+强制要求**：不再支持Java 7，充分利用Lambda表达式等新特性

| 默认端口变化 | Hadoop 2.x | Hadoop 3.x |
|-------------|-----------|-----------|
| NameNode HTTP | 50070 | 9870 |
| NameNode RPC | 8020 | 8020（不变） |
| DataNode HTTP | 50075 | 9864 |
| DataNode Data | 50010 | 9866 |
| Secondary NN HTTP | 50090 | 9868 |

---

## 2. HDFS架构详解

### 2.1 整体架构

HDFS采用Master/Slave架构，一个集群由一个NameNode（Master）和多个DataNode（Slave）组成：

```
HDFS整体架构
┌──────────────────────────────────────────────────────────────────────┐
│                            Client                                    │
│                    ┌───────────────────┐                              │
│                    │ HDFS Client API   │                              │
│                    │ (读写请求入口)     │                              │
│                    └────────┬──────────┘                              │
│                             │                                        │
│              ┌──────────────┼──────────────┐                         │
│              ↓              ↓              ↓                         │
│     ┌────────────┐  ┌────────────┐  ┌──────────────┐                │
│     │  NameNode  │  │ Secondary  │  │  Standby     │                │
│     │  (Active)  │←→│ NameNode   │  │  NameNode    │                │
│     │ 元数据管理  │  │ Checkpoint │  │  (HA热备)    │                │
│     └─────┬──────┘  └────────────┘  └──────────────┘                │
│           │                                                          │
│           │  元数据操作(文件→Block→DataNode映射)                      │
│           │                                                          │
│    ┌──────┼──────────────────────────────────┐                       │
│    │      ↓           Rack 1                 │                       │
│    │  ┌──────────┐  ┌──────────┐             │                       │
│    │  │DataNode 1│  │DataNode 2│             │                       │
│    │  │┌──┐┌──┐  │  │┌──┐┌──┐  │             │                       │
│    │  ││B1││B3│  │  ││B1││B4│  │             │                       │
│    │  │└──┘└──┘  │  │└──┘└──┘  │             │                       │
│    │  └──────────┘  └──────────┘             │                       │
│    └─────────────────────────────────────────┘                       │
│    ┌─────────────────────────────────────────┐                       │
│    │              Rack 2                     │                       │
│    │  ┌──────────┐  ┌──────────┐             │                       │
│    │  │DataNode 3│  │DataNode 4│             │                       │
│    │  │┌──┐┌──┐  │  │┌──┐┌──┐  │             │                       │
│    │  ││B2││B4│  │  ││B2││B3│  │             │                       │
│    │  │└──┘└──┘  │  │└──┘└──┘  │             │                       │
│    │  └──────────┘  └──────────┘             │                       │
│    └─────────────────────────────────────────┘                       │
└──────────────────────────────────────────────────────────────────────┘
```

**Block副本放置策略（Rack-Aware）**：

HDFS默认的副本放置策略考虑了机架感知，以平衡可靠性和网络带宽：

1. 第1个副本：放在写入Client所在的DataNode（若Client不在集群中，则随机选一个负载低的节点）
2. 第2个副本：放在与第1个副本**不同机架**的某个DataNode上
3. 第3个副本：放在与第2个副本**同一机架**的另一个DataNode上

这样即使一个机架整体故障，数据仍有副本在另一个机架上可用。

### 2.2 NameNode与元数据管理

NameNode是HDFS的核心，负责管理文件系统命名空间和控制客户端访问。

**元数据结构**：

NameNode在内存中维护完整的文件系统元数据，包括：

- 文件目录树（类似Linux的inode）
- 文件与Block的映射关系
- Block与DataNode的映射关系（由DataNode汇报，不持久化）

**FsImage + EditLog 持久化机制**：

```
NameNode元数据持久化
┌──────────────────────────────────────────────────────────┐
│                    NameNode 内存                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │         完整的文件系统命名空间 (Namespace)           │  │
│  │  /user/data/file1.txt → [blk_001, blk_002]        │  │
│  │  /user/data/file2.txt → [blk_003]                 │  │
│  │  blk_001 → [DN1, DN3, DN4]                        │  │
│  │  blk_002 → [DN2, DN3, DN5]                        │  │
│  └────────────────────────────────────────────────────┘  │
│         │                              │                  │
│         ↓                              ↓                  │
│  ┌──────────────┐              ┌──────────────┐          │
│  │   FsImage    │              │   EditLog    │          │
│  │  文件系统快照 │              │  操作日志     │          │
│  │ (定期合并)   │              │ (实时追加写入)│          │
│  └──────────────┘              └──────────────┘          │
│                                                          │
│  Checkpoint过程 (由Secondary NameNode或Standby执行):     │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐        │
│  │ FsImage  │ +  │ EditLog  │ →  │ 新FsImage    │        │
│  │ (旧快照) │    │ (增量日志)│    │ (合并后快照) │        │
│  └──────────┘    └──────────┘    └──────────────┘        │
└──────────────────────────────────────────────────────────┘
```

**HA架构（高可用）**：

生产环境中必须部署NameNode HA，避免单点故障：

```
NameNode HA 架构
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   ┌──────────────┐          ┌──────────────┐                │
│   │  Active NN   │          │ Standby NN   │                │
│   │  (接收读写)  │          │  (热备待命)  │                │
│   └──────┬───────┘          └──────┬───────┘                │
│          │                         │                         │
│          │    共享EditLog          │                         │
│          ↓                         ↓                         │
│   ┌──────────────────────────────────────┐                  │
│   │         JournalNode集群 (QJM)        │                  │
│   │  ┌────────┐ ┌────────┐ ┌────────┐   │                  │
│   │  │  JN1   │ │  JN2   │ │  JN3   │   │                  │
│   │  └────────┘ └────────┘ └────────┘   │                  │
│   │  (至少3个节点，写入多数即成功)        │                  │
│   └──────────────────────────────────────┘                  │
│                                                              │
│   ┌──────────────────────────────────────┐                  │
│   │       ZooKeeper集群 (故障检测)       │                  │
│   │  ┌────────┐ ┌────────┐ ┌────────┐   │                  │
│   │  │  ZK1   │ │  ZK2   │ │  ZK3   │   │                  │
│   │  └────────┘ └────────┘ └────────┘   │                  │
│   │                                      │                  │
│   │  ZKFC (ZooKeeper Failover Controller)│                  │
│   │  监控NN健康状态，触发自动切换         │                  │
│   └──────────────────────────────────────┘                  │
│                                                              │
│  切换流程:                                                   │
│  1. ZKFC检测到Active NN故障                                  │
│  2. ZKFC通过ZK选举确定新的Active                             │
│  3. 原Active被Fencing（隔离）                                │
│  4. Standby NN从JournalNode追回最新EditLog                   │
│  5. Standby晋升为新的Active NN                               │
└──────────────────────────────────────────────────────────────┘
```

**Federation（联邦）**：

当单个NameNode内存不足以存放所有元数据时（通常在文件数过亿的场景），可使用Federation。每个NameNode管理独立的命名空间（Namespace），共享底层DataNode存储池：

```
Federation架构
┌──────────────────────────────────────────────────┐
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ NN1      │  │ NN2      │  │ NN3      │       │
│  │ /user/   │  │ /data/   │  │ /tmp/    │       │
│  │Namespace1│  │Namespace2│  │Namespace3│       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │              │              │             │
│       └──────────────┼──────────────┘             │
│                      ↓                            │
│  ┌────────────────────────────────────────────┐  │
│  │     共享DataNode存储池 (Block Pool)        │  │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐      │  │
│  │  │ DN1  │ │ DN2  │ │ DN3  │ │ DN4  │ ...  │  │
│  │  └──────┘ └──────┘ └──────┘ └──────┘      │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

### 2.3 DataNode与数据存储

DataNode是HDFS的工作节点，负责存储实际的数据Block。

**Block本地存储结构**：

每个Block在DataNode本地磁盘上存储为两个文件：

```bash
# DataNode本地磁盘目录结构
${dfs.datanode.data.dir}/
├── current/
│   ├── BP-123456-172.16.1.1-1234567890/  # Block Pool ID
│   │   ├── current/
│   │   │   ├── finalized/
│   │   │   │   └── subdir0/
│   │   │   │       └── subdir0/
│   │   │   │           ├── blk_1073741825      # Block数据文件
│   │   │   │           └── blk_1073741825.meta  # Block校验和元数据
│   │   │   └── rbw/                             # 正在写入的Block
│   │   └── tmp/
│   └── VERSION
└── in_use.lock
```

**Heartbeat心跳机制**：

DataNode通过心跳向NameNode报告自身状态：

| 机制 | 频率 | 内容 | 作用 |
|------|------|------|------|
| **Heartbeat** | 每3秒 | 节点存活、磁盘容量、负载 | NameNode感知节点状态 |
| **Block Report** | 每6小时 | 节点上所有Block列表 | NameNode校验Block完整性 |
| **Incremental Block Report** | 实时 | 新增/删除的Block | 及时更新Block映射 |
| **Cache Report** | 每10秒 | 缓存的Block列表 | 集中式缓存管理 |

当NameNode超过10分钟（默认）未收到某DataNode心跳，则标记该节点为Dead，并触发该节点上Block的副本重建。

### 2.4 读写流程

**HDFS写入流程**：

```
HDFS文件写入流程
┌──────────┐                ┌──────────┐
│  Client  │                │ NameNode │
└────┬─────┘                └────┬─────┘
     │                           │
     │  1. create(file, rep=3)   │
     │ ─────────────────────────→│  检查权限、目标路径是否存在
     │                           │  在EditLog中记录创建操作
     │  2. 返回FSDataOutputStream │
     │ ←─────────────────────────│
     │                           │
     │  3. addBlock()            │
     │ ─────────────────────────→│  分配Block ID
     │                           │  选择3个DataNode (机架感知)
     │  4. 返回[DN1, DN2, DN3]   │
     │ ←─────────────────────────│
     │                           │
     │  5. 建立Pipeline写入      │
     │ ──→ ┌──────┐ ──→ ┌──────┐ ──→ ┌──────┐
     │     │ DN1  │     │ DN2  │     │ DN3  │
     │     └──┬───┘     └──┬───┘     └──┬───┘
     │        │            │            │
     │  数据以Packet(64KB)为单位在Pipeline中传递
     │  ┌─────────────────────────────────────┐
     │  │ Packet流: Client→DN1→DN2→DN3       │
     │  │ ACK流:    DN3→DN2→DN1→Client       │
     │  └─────────────────────────────────────┘
     │        │            │            │
     │  6. ACK确认(反向)    │            │
     │ ←──────┘ ←──────────┘ ←──────────┘
     │                           │
     │  7. complete(file)        │
     │ ─────────────────────────→│  提交文件，更新元数据
     │                           │
```

**写入关键细节**：

- 数据先写入Client端的本地缓冲（Data Queue），以Packet（64KB）为单位发送
- 同时维护Ack Queue，收到所有DN确认后才移除Packet
- 若Pipeline中某个DN故障，Client会重建Pipeline跳过故障节点，NameNode随后安排副本补齐

**HDFS读取流程**：

```
HDFS文件读取流程
┌──────────┐                ┌──────────┐
│  Client  │                │ NameNode │
└────┬─────┘                └────┬─────┘
     │                           │
     │  1. open(file)            │
     │ ─────────────────────────→│  检查权限
     │                           │  查找文件的Block列表
     │  2. 返回Block位置列表      │  每个Block按距离排序
     │    [blk1→(DN1,DN3,DN4)]  │
     │    [blk2→(DN2,DN3,DN5)]  │
     │ ←─────────────────────────│
     │                           │
     │  3. read blk1 from DN1   │
     │ ──────────→┌──────┐      │
     │            │ DN1  │      │  选择距Client最近的DN读取
     │ ←──────────│(blk1)│      │
     │   数据流    └──────┘      │
     │                           │
     │  4. read blk2 from DN2   │
     │ ──────────→┌──────┐      │
     │            │ DN2  │      │  读完blk1后，继续读blk2
     │ ←──────────│(blk2)│      │
     │   数据流    └──────┘      │
     │                           │
     │  5. close()               │
     │  完成读取                  │
```

**读取关键细节**：

- NameNode返回的DataNode列表按照到Client的网络距离排序（同节点 > 同机架 > 同数据中心 > 其他）
- 支持Short-Circuit Local Read：Client与DataNode在同一节点时，直接读取本地磁盘文件，跳过网络传输
- 读取过程中若某DataNode故障，自动切换到持有该Block副本的其他DN

---

## 3. HDFS操作实战

### 3.1 Shell命令

HDFS Shell提供了类Unix的文件操作命令，所有命令以`hdfs dfs`或`hadoop fs`开头：

```bash
# ==================== 目录与文件操作 ====================

# 创建目录（-p 递归创建父目录）
hdfs dfs -mkdir -p /user/data/logs

# 上传本地文件到HDFS
hdfs dfs -put localfile.txt /user/data/
# 等同于 -copyFromLocal
hdfs dfs -copyFromLocal localfile.txt /user/data/

# 追加内容到已有HDFS文件（不覆盖原内容）
hdfs dfs -appendToFile newdata.txt /user/data/localfile.txt

# 从HDFS下载文件到本地
hdfs dfs -get /user/data/file.txt ./local/
# 等同于 -copyToLocal
hdfs dfs -copyToLocal /user/data/file.txt ./local/

# 合并下载（将HDFS目录下所有文件合并为一个本地文件）
hdfs dfs -getmerge /user/data/output/ ./merged_result.txt

# 列出目录内容（-R 递归，-h 人类可读大小）
hdfs dfs -ls -R /user/data/
hdfs dfs -ls -h /user/data/

# 查看文件内容
hdfs dfs -cat /user/data/file.txt
hdfs dfs -head /user/data/file.txt   # 查看前1KB
hdfs dfs -tail /user/data/file.txt   # 查看后1KB

# 删除文件或目录（-r 递归删除，-skipTrash 直接删除不经过回收站）
hdfs dfs -rm /user/data/temp.txt
hdfs dfs -rm -r /user/data/old/
hdfs dfs -rm -r -skipTrash /user/data/expired/

# 文件移动与重命名
hdfs dfs -mv /user/data/file1.txt /user/data/archive/

# 文件复制
hdfs dfs -cp /user/data/file1.txt /user/backup/

# ==================== 权限与属性 ====================

# 修改文件权限
hdfs dfs -chmod 755 /user/data/
hdfs dfs -chmod -R 644 /user/data/logs/

# 修改文件所有者
hdfs dfs -chown hadoop:hadoop /user/data/
hdfs dfs -chown -R hadoop:hadoop /user/data/logs/

# 查看文件副本数和Block信息
hdfs dfs -stat "%r %b %n" /user/data/file.txt

# 设置文件副本数
hdfs dfs -setrep -w 2 /user/data/cold_data/

# ==================== 空间与统计 ====================

# 查看目录占用空间（-s 汇总，-h 人类可读）
hdfs dfs -du -s -h /user/data/
# 输出示例: 5.2 G  15.6 G  /user/data/
#           逻辑大小 物理大小(含副本)

# 查看文件数和空间统计
hdfs dfs -count -q -h /user/data/
# 输出: QUOTA  REMAINING_QUOTA  SPACE_QUOTA  REMAINING_SPACE_QUOTA  DIR_COUNT  FILE_COUNT  CONTENT_SIZE  PATHNAME

# 文件校验和
hdfs dfs -checksum /user/data/file.txt

# ==================== 集群管理命令 ====================

# 查看集群整体状态报告
hdfs dfsadmin -report

# 查看/设置SafeMode（安全模式：NN启动时的只读状态）
hdfs dfsadmin -safemode get
hdfs dfsadmin -safemode enter    # 手动进入安全模式
hdfs dfsadmin -safemode leave    # 手动退出安全模式

# 文件系统检查
hdfs fsck /user/data/ -files -blocks -locations

# 查看集群中所有DataNode
hdfs dfsadmin -printTopology

# 刷新节点列表（添加/退役节点后执行）
hdfs dfsadmin -refreshNodes

# 平衡集群数据分布
hdfs balancer -threshold 10
```

### 3.2 Java API

使用Hadoop Java API进行HDFS文件操作：

```java
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.IOUtils;

import java.io.*;
import java.net.URI;

/**
 * HDFS Java API 操作示例
 * 涵盖常见文件操作：创建目录、上传、下载、读写、遍历、删除
 */
public class HdfsOperations {

    private FileSystem fs;

    /**
     * 初始化HDFS连接
     * 支持HA配置（高可用NameNode）
     */
    public void init() throws Exception {
        Configuration conf = new Configuration();

        // 基本配置
        conf.set("fs.defaultFS", "hdfs://mycluster");

        // HA配置（如果使用NameNode高可用）
        conf.set("dfs.nameservices", "mycluster");
        conf.set("dfs.ha.namenodes.mycluster", "nn1,nn2");
        conf.set("dfs.namenode.rpc-address.mycluster.nn1", "master1:8020");
        conf.set("dfs.namenode.rpc-address.mycluster.nn2", "master2:8020");
        conf.set("dfs.client.failover.proxy.provider.mycluster",
                "org.apache.hadoop.hdfs.server.namenode.ha.ConfiguredFailoverProxyProvider");

        // 获取FileSystem实例（以hadoop用户身份）
        fs = FileSystem.get(new URI("hdfs://mycluster"), conf, "hadoop");
        System.out.println("HDFS连接成功");
    }

    /**
     * 创建HDFS目录
     */
    public void createDirectory(String path) throws Exception {
        Path hdfsPath = new Path(path);
        if (fs.exists(hdfsPath)) {
            System.out.println("目录已存在: " + path);
        } else {
            boolean result = fs.mkdirs(hdfsPath);
            System.out.println("创建目录 " + path + " : " + (result ? "成功" : "失败"));
        }
    }

    /**
     * 上传本地文件到HDFS
     * @param localPath  本地文件路径
     * @param hdfsPath   HDFS目标路径
     */
    public void uploadFile(String localPath, String hdfsPath) throws Exception {
        // delSrc=false: 不删除本地源文件
        // overwrite=true: 覆盖HDFS上已有文件
        fs.copyFromLocalFile(false, true,
                new Path(localPath), new Path(hdfsPath));
        System.out.println("上传成功: " + localPath + " → " + hdfsPath);
    }

    /**
     * 从HDFS下载文件到本地
     */
    public void downloadFile(String hdfsPath, String localPath) throws Exception {
        // useRawLocalFileSystem=true: 禁用CRC校验文件生成
        fs.copyToLocalFile(false, new Path(hdfsPath),
                new Path(localPath), true);
        System.out.println("下载成功: " + hdfsPath + " → " + localPath);
    }

    /**
     * 通过流写入数据到HDFS文件
     */
    public void writeFile(String path, String content) throws Exception {
        Path hdfsPath = new Path(path);
        // 创建输出流（如果文件存在则覆盖）
        try (FSDataOutputStream out = fs.create(hdfsPath, true)) {
            out.write(content.getBytes("UTF-8"));
            out.hflush(); // 刷新到DataNode内存
            // out.hsync(); // 刷新到DataNode磁盘（更强的持久化保证）
        }
        System.out.println("写入完成: " + path);
    }

    /**
     * 从HDFS读取文件内容
     */
    public String readFile(String path) throws Exception {
        Path hdfsPath = new Path(path);
        if (!fs.exists(hdfsPath)) {
            throw new FileNotFoundException("文件不存在: " + path);
        }

        StringBuilder content = new StringBuilder();
        try (FSDataInputStream in = fs.open(hdfsPath);
             BufferedReader reader = new BufferedReader(new InputStreamReader(in, "UTF-8"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                content.append(line).append("\n");
            }
        }
        return content.toString();
    }

    /**
     * 递归遍历HDFS目录
     */
    public void listFiles(String path) throws Exception {
        // recursive=true 递归遍历
        RemoteIterator<LocatedFileStatus> iterator =
                fs.listFiles(new Path(path), true);

        System.out.println("===== 文件列表: " + path + " =====");
        while (iterator.hasNext()) {
            LocatedFileStatus status = iterator.next();
            System.out.printf("%-10s %-6d %-3d %s%n",
                    status.isDirectory() ? "目录" : "文件",
                    status.getLen(),                    // 文件大小（字节）
                    status.getReplication(),             // 副本数
                    status.getPath().toString());        // 完整路径
        }
    }

    /**
     * 删除HDFS文件或目录
     */
    public void delete(String path, boolean recursive) throws Exception {
        Path hdfsPath = new Path(path);
        if (fs.exists(hdfsPath)) {
            boolean result = fs.delete(hdfsPath, recursive);
            System.out.println("删除 " + path + " : " + (result ? "成功" : "失败"));
        } else {
            System.out.println("路径不存在: " + path);
        }
    }

    /**
     * 获取文件Block位置信息（运维排查常用）
     */
    public void getBlockLocations(String path) throws Exception {
        FileStatus status = fs.getFileStatus(new Path(path));
        BlockLocation[] blocks = fs.getFileBlockLocations(status, 0, status.getLen());

        System.out.println("文件: " + path);
        System.out.println("总大小: " + status.getLen() + " 字节");
        System.out.println("Block数量: " + blocks.length);

        for (int i = 0; i < blocks.length; i++) {
            BlockLocation block = blocks[i];
            System.out.printf("Block %d: offset=%d, length=%d, hosts=%s%n",
                    i, block.getOffset(), block.getLength(),
                    String.join(",", block.getHosts()));
        }
    }

    /**
     * 关闭HDFS连接
     */
    public void close() throws Exception {
        if (fs != null) {
            fs.close();
            System.out.println("HDFS连接已关闭");
        }
    }

    public static void main(String[] args) throws Exception {
        HdfsOperations hdfs = new HdfsOperations();
        try {
            hdfs.init();

            // 创建目录
            hdfs.createDirectory("/user/demo/test");

            // 写入文件
            hdfs.writeFile("/user/demo/test/hello.txt", "你好，HDFS！\n这是Java API写入的数据。");

            // 读取文件
            String content = hdfs.readFile("/user/demo/test/hello.txt");
            System.out.println("文件内容: " + content);

            // 遍历目录
            hdfs.listFiles("/user/demo/");

            // 查看Block位置
            hdfs.getBlockLocations("/user/demo/test/hello.txt");

            // 删除目录（递归）
            hdfs.delete("/user/demo/test", true);
        } finally {
            hdfs.close();
        }
    }
}
```

**Maven依赖配置**：

```xml
<dependencies>
    <!-- Hadoop Client（包含HDFS、YARN、MapReduce客户端） -->
    <dependency>
        <groupId>org.apache.hadoop</groupId>
        <artifactId>hadoop-client</artifactId>
        <version>3.3.6</version>
    </dependency>
</dependencies>
```

### 3.3 Python API

Python可以通过`hdfs`库或`pyarrow`库操作HDFS：

```python
# ==================== 方式一：使用hdfs库（WebHDFS协议） ====================
# 安装: pip install hdfs

from hdfs import InsecureClient

# 连接HDFS（通过WebHDFS REST API，端口9870为Hadoop 3.x默认）
client = InsecureClient("http://namenode-host:9870", user="hadoop")

# 创建目录
client.makedirs("/user/python_demo/data")
print("目录创建成功")

# 上传本地文件
client.upload("/user/python_demo/data/", "local_file.csv")
print("文件上传成功")

# 写入数据到HDFS文件
with client.write("/user/python_demo/data/output.txt",
                  encoding="utf-8", overwrite=True) as writer:
    writer.write("这是Python通过WebHDFS写入的数据\n")
    writer.write("第二行内容\n")
print("数据写入成功")

# 读取HDFS文件
with client.read("/user/python_demo/data/output.txt",
                 encoding="utf-8") as reader:
    content = reader.read()
    print(f"文件内容: {content}")

# 列出目录文件
file_list = client.list("/user/python_demo/data/")
print(f"目录文件: {file_list}")

# 获取文件状态信息
status = client.status("/user/python_demo/data/output.txt")
print(f"文件大小: {status['length']} 字节")
print(f"副本数: {status['replication']}")
print(f"修改时间: {status['modificationTime']}")

# 删除文件（recursive=True 递归删除目录）
client.delete("/user/python_demo/data/output.txt")
print("文件删除成功")


# ==================== 方式二：使用pyarrow（原生HDFS协议） ====================
# 安装: pip install pyarrow
# 注意: 需要配置HADOOP_HOME和CLASSPATH环境变量

import pyarrow.fs as pafs

# 连接HDFS（原生RPC协议，性能优于WebHDFS）
hdfs = pafs.HadoopFileSystem(host="namenode-host", port=8020, user="hadoop")

# 创建目录
hdfs.create_dir("/user/arrow_demo/data")

# 写入文件
with hdfs.open_output_stream("/user/arrow_demo/data/sample.txt") as f:
    f.write(b"PyArrow HDFS write example\n")
    f.write("中文内容测试\n".encode("utf-8"))

# 读取文件
with hdfs.open_input_stream("/user/arrow_demo/data/sample.txt") as f:
    data = f.read()
    print(f"内容: {data.decode('utf-8')}")

# 获取文件信息
info = hdfs.get_file_info("/user/arrow_demo/data/sample.txt")
print(f"大小: {info.size}, 类型: {info.type}")

# 列出目录
selector = pafs.FileSelector("/user/arrow_demo/data/", recursive=True)
file_infos = hdfs.get_file_info(selector)
for fi in file_infos:
    print(f"  {fi.path} - {fi.size} bytes")

# 删除
hdfs.delete_dir_contents("/user/arrow_demo/data/")
```

---

## 4. YARN资源调度

### 4.1 YARN架构

YARN（Yet Another Resource Negotiator）是Hadoop 2.x引入的通用资源管理平台，将资源管理与作业调度解耦：

```
YARN架构
┌──────────────────────────────────────────────────────────────────────┐
│                     ResourceManager (RM)                             │
│            ┌────────────────┐  ┌───────────────────┐                │
│            │   Scheduler    │  │ ApplicationsManager│                │
│            │ (资源分配策略) │  │  (应用管理)        │                │
│            │ 不监控、不重启 │  │  接受提交、协调AM  │                │
│            └───────┬────────┘  └────────┬──────────┘                │
│                    │                    │                            │
└────────────────────┼────────────────────┼────────────────────────────┘
                     │                    │
        ┌────────────┼────────────────────┼───────────────┐
        │            ↓                    ↓               │
        │  ┌──────────────────────────────────────────┐   │
        │  │          NodeManager 1                    │   │
        │  │  ┌──────────────────────────────────┐    │   │
        │  │  │  Container (ApplicationMaster)   │    │   │
        │  │  │  管理自身应用的生命周期           │    │   │
        │  │  │  向RM申请资源                     │    │   │
        │  │  │  分配Task到Container              │    │   │
        │  │  └──────────────────────────────────┘    │   │
        │  │  ┌──────────────┐  ┌──────────────┐      │   │
        │  │  │ Container    │  │ Container    │      │   │
        │  │  │ (Map Task)   │  │ (Reduce Task)│      │   │
        │  │  └──────────────┘  └──────────────┘      │   │
        │  └──────────────────────────────────────────┘   │
        │                                                  │
        │  ┌──────────────────────────────────────────┐   │
        │  │          NodeManager 2                    │   │
        │  │  ┌──────────────┐  ┌──────────────┐      │   │
        │  │  │ Container    │  │ Container    │      │   │
        │  │  │ (Spark Exec) │  │ (Flink TM)  │      │   │
        │  │  └──────────────┘  └──────────────┘      │   │
        │  └──────────────────────────────────────────┘   │
        │                                                  │
        │  ┌──────────────────────────────────────────┐   │
        │  │          NodeManager 3                    │   │
        │  │  ┌──────────────┐  ┌──────────────┐      │   │
        │  │  │ Container    │  │ Container    │      │   │
        │  │  │ (Task)       │  │ (Task)       │      │   │
        │  │  └──────────────┘  └──────────────┘      │   │
        │  └──────────────────────────────────────────┘   │
        └──────────────────────────────────────────────────┘
```

**YARN核心组件**：

- **ResourceManager (RM)**：全局资源管理器，负责集群资源的分配与监控。包含Scheduler（纯资源分配）和ApplicationsManager（应用生命周期管理）。
- **NodeManager (NM)**：每个节点上的代理，管理本节点的Container，监控资源使用。
- **ApplicationMaster (AM)**：每个应用一个AM，负责与RM协商资源、与NM通信启动/监控Task。
- **Container**：YARN中的资源抽象单位，封装了CPU、内存等资源，是任务的运行环境。

**作业提交流程**：

```
YARN作业提交与执行流程
┌──────────┐  1.提交应用    ┌──────────────┐
│  Client  │ ──────────────→│ResourceManager│
└──────────┘                └──────┬───────┘
                                   │
                    2.分配Container启动AM
                                   │
                                   ↓
                            ┌──────────────┐
                            │NodeManager(NM)│
                            │┌────────────┐│
                            ││ AppMaster  ││  3.AM注册到RM
                            │└─────┬──────┘│
                            └──────┼───────┘
                                   │
                    4.AM向RM申请资源(Container)
                                   │
                                   ↓
            ┌──────────────┐  ┌──────────────┐
            │  NM (节点A)  │  │  NM (节点B)  │
            │ ┌──────────┐ │  │ ┌──────────┐ │
            │ │Container │ │  │ │Container │ │  5.AM通知NM启动Container
            │ │ (Task 1) │ │  │ │ (Task 2) │ │
            │ └──────────┘ │  │ └──────────┘ │  6.Task在Container中执行
            └──────────────┘  └──────────────┘
                                   │
                    7.任务完成后AM向RM注销
```

### 4.2 调度器对比

YARN提供三种调度器策略：

| 维度 | FIFO Scheduler | Capacity Scheduler | Fair Scheduler |
|------|---------------|-------------------|----------------|
| **调度策略** | 先进先出 | 基于容量的队列调度 | 公平共享资源 |
| **队列支持** | 单队列 | 多层级队列 | 多层级队列 |
| **资源共享** | 无（独占式） | 队列间可借用资源 | 所有队列公平分配 |
| **抢占** | 不支持 | 支持（可配置） | 支持 |
| **多租户** | 不支持 | 强支持（企业级） | 支持 |
| **适用场景** | 测试环境 | 大规模生产（企业） | 中小规模、混合负载 |
| **默认使用** | Hadoop 1.x | Hadoop 2.x/3.x默认 | 部分发行版默认 |
| **资源保证** | 无 | 最小容量保证 | 最小共享量保证 |

**Capacity Scheduler 队列配置示例**：

```
Capacity Scheduler 队列层级
┌─────────────────────────────────────────────────┐
│                   root (100%)                    │
│  ┌─────────────┐  ┌──────────┐  ┌────────────┐  │
│  │ production  │  │   dev    │  │  default   │  │
│  │   (60%)     │  │  (30%)   │  │   (10%)    │  │
│  │  ┌───┐┌───┐│  │  ┌───┐   │  │            │  │
│  │  │ETL││推荐││  │  │测试│   │  │            │  │
│  │  │30%││30%││  │  │30%│   │  │            │  │
│  │  └───┘└───┘│  │  └───┘   │  │            │  │
│  └─────────────┘  └──────────┘  └────────────┘  │
└─────────────────────────────────────────────────┘
```

### 4.3 资源配置

**yarn-site.xml 核心参数**：

```xml
<configuration>
    <!-- ==================== ResourceManager 配置 ==================== -->

    <!-- 调度器类型（默认Capacity Scheduler） -->
    <property>
        <name>yarn.resourcemanager.scheduler.class</name>
        <value>org.apache.hadoop.yarn.server.resourcemanager.scheduler.capacity.CapacityScheduler</value>
    </property>

    <!-- RM地址 -->
    <property>
        <name>yarn.resourcemanager.hostname</name>
        <value>master1</value>
    </property>

    <!-- RM WebUI端口 -->
    <property>
        <name>yarn.resourcemanager.webapp.address</name>
        <value>master1:8088</value>
    </property>

    <!-- ==================== NodeManager 配置 ==================== -->

    <!-- 单个节点可用内存（根据物理内存的80%配置） -->
    <property>
        <name>yarn.nodemanager.resource.memory-mb</name>
        <value>65536</value>
        <description>单个NM可分配给Container的总内存(MB)，64GB机器配置约52GB</description>
    </property>

    <!-- 单个节点可用CPU核数 -->
    <property>
        <name>yarn.nodemanager.resource.cpu-vcores</name>
        <value>16</value>
        <description>单个NM可分配的虚拟CPU核数</description>
    </property>

    <!-- 内存检查开关 -->
    <property>
        <name>yarn.nodemanager.pmem-check-enabled</name>
        <value>true</value>
        <description>启用物理内存检查，超限则Kill Container</description>
    </property>

    <!-- ==================== Container 配置 ==================== -->

    <!-- 单个Container最小内存 -->
    <property>
        <name>yarn.scheduler.minimum-allocation-mb</name>
        <value>1024</value>
    </property>

    <!-- 单个Container最大内存 -->
    <property>
        <name>yarn.scheduler.maximum-allocation-mb</name>
        <value>32768</value>
    </property>

    <!-- 单个Container最小CPU -->
    <property>
        <name>yarn.scheduler.minimum-allocation-vcores</name>
        <value>1</value>
    </property>

    <!-- 单个Container最大CPU -->
    <property>
        <name>yarn.scheduler.maximum-allocation-vcores</name>
        <value>8</value>
    </property>

    <!-- ==================== 日志聚合 ==================== -->

    <!-- 启用日志聚合（Container日志收集到HDFS） -->
    <property>
        <name>yarn.log-aggregation-enable</name>
        <value>true</value>
    </property>

    <!-- 聚合日志保留时间（秒），7天 -->
    <property>
        <name>yarn.log-aggregation.retain-seconds</name>
        <value>604800</value>
    </property>
</configuration>
```

**YARN常用命令**：

```bash
# 查看所有正在运行的应用
yarn application -list

# 查看所有已完成的应用
yarn application -list -appStates FINISHED

# 查看指定应用详细信息
yarn application -status application_1234567890_0001

# 终止应用
yarn application -kill application_1234567890_0001

# 查看应用日志（需启用日志聚合）
yarn logs -applicationId application_1234567890_0001

# 查看集群节点列表
yarn node -list

# 查看节点详细信息
yarn node -status node1:45454

# 查看队列信息
yarn queue -status default

# 查看集群整体资源使用情况
yarn top
```

---

## 5. MapReduce编程模型

### 5.1 Map-Shuffle-Reduce流程

MapReduce将计算过程分解为Map（映射）和Reduce（归约）两个阶段，中间通过Shuffle（洗牌）进行数据重分布：

```
MapReduce完整执行流程
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  输入文件 (HDFS)                                                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                                    │
│  │ Split 0  │ │ Split 1  │ │ Split 2  │  InputFormat按Block拆分            │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘                                    │
│       │             │            │                                          │
│       ↓             ↓            ↓                                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                                     │
│  │ Mapper 0│  │ Mapper 1│  │ Mapper 2│  Map阶段: 逐行处理，输出<K2,V2>     │
│  └────┬────┘  └────┬────┘  └────┬────┘                                     │
│       │             │            │                                          │
│       ↓             ↓            ↓                                          │
│  ┌─────────────────────────────────────┐                                    │
│  │         Map端处理                    │                                    │
│  │  1. Partition: 按Key的Hash分区      │                                    │
│  │     (决定发送给哪个Reducer)          │                                    │
│  │  2. Sort: 按Key排序                 │                                    │
│  │  3. Combine: 本地预聚合(可选)       │                                    │
│  │     (减少Shuffle传输数据量)          │                                    │
│  └───────────────┬─────────────────────┘                                    │
│                  │                                                          │
│                  ↓  Shuffle (网络传输)                                       │
│       ┌──────────┴──────────┐                                               │
│       ↓                     ↓                                               │
│  ┌──────────┐         ┌──────────┐                                          │
│  │Reducer 0 │         │Reducer 1 │                                          │
│  │ 合并排序  │         │ 合并排序  │  Reduce端: 按Key分组                    │
│  │ 归约计算  │         │ 归约计算  │  对每组<Key, [V1,V2,...]>执行Reduce     │
│  └────┬─────┘         └────┬─────┘                                          │
│       │                    │                                                │
│       ↓                    ↓                                                │
│  ┌──────────┐         ┌──────────┐                                          │
│  │ Output 0 │         │ Output 1 │  输出到HDFS: part-r-00000, part-r-00001  │
│  └──────────┘         └──────────┘                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Shuffle详细过程**：

```
Shuffle内部细节
┌──────────────────────────────┐    ┌──────────────────────────────┐
│        Map端 (写出)           │    │       Reduce端 (拉取)         │
│                              │    │                              │
│  Map输出                     │    │                              │
│    ↓                         │    │                              │
│  ┌──────────────┐            │    │                              │
│  │ 环形缓冲区   │ 100MB     │    │                              │
│  │ (kvbuffer)   │ 默认       │    │                              │
│  └──────┬───────┘            │    │                              │
│         │ 达到80%阈值时Spill  │    │                              │
│         ↓                    │    │                              │
│  ┌──────────────┐            │    │  ┌──────────────┐            │
│  │ Partition     │            │    │  │ Copy阶段    │            │
│  │ (分区)        │            │    │  │ 从各Mapper   │            │
│  ├──────────────┤            │    │  │ 拉取属于自己 │            │
│  │ Sort          │            │    │  │ 分区的数据   │            │
│  │ (排序)        │            │    │  └──────┬───────┘            │
│  ├──────────────┤            │    │         ↓                    │
│  │ Combine       │            │    │  ┌──────────────┐            │
│  │ (可选预聚合)  │            │    │  │ Merge Sort   │            │
│  └──────┬───────┘            │    │  │ (归并排序)   │            │
│         ↓                    │    │  └──────┬───────┘            │
│  ┌──────────────┐            │    │         ↓                    │
│  │ 溢写到磁盘   │            │    │  ┌──────────────┐            │
│  │ (Spill File) │ ──────────────→ │  │ Reduce函数   │            │
│  └──────────────┘            │    │  │ 逐组处理     │            │
│         ↓                    │    │  └──────────────┘            │
│  ┌──────────────┐            │    │                              │
│  │ Merge         │            │    │                              │
│  │ (合并溢写文件)│            │    │                              │
│  └──────────────┘            │    │                              │
└──────────────────────────────┘    └──────────────────────────────┘
```

### 5.2 WordCount完整示例

WordCount是MapReduce的经典入门程序，统计文本中每个单词出现的次数：

```java
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;

/**
 * WordCount - MapReduce经典入门示例
 * 功能：统计输入文件中每个单词出现的次数
 *
 * 输入示例:
 *   hello world
 *   hello hadoop
 *
 * 输出示例:
 *   hadoop  1
 *   hello   2
 *   world   1
 */
public class WordCount {

    /**
     * Mapper阶段
     * 输入: <行偏移量LongWritable, 行内容Text>
     * 输出: <单词Text, 计数1 IntWritable>
     *
     * 每读取一行文本，拆分为单词，输出 <word, 1>
     */
    public static class WordMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

        // 复用对象避免频繁GC（MapReduce性能优化技巧）
        private final Text outputKey = new Text();
        private final IntWritable outputValue = new IntWritable(1);

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            // 获取当前行内容并按空白字符拆分
            String line = value.toString();
            String[] words = line.split("\\s+");

            // 遍历每个单词，输出 <word, 1>
            for (String word : words) {
                if (!word.isEmpty()) {
                    outputKey.set(word.toLowerCase().trim());
                    context.write(outputKey, outputValue);
                }
            }
        }
    }

    /**
     * Combiner/Reducer阶段
     * 输入: <单词Text, 计数列表Iterable<IntWritable>>
     * 输出: <单词Text, 总次数IntWritable>
     *
     * 对相同Key的所有Value求和
     * 此类同时用作Combiner（Map端本地预聚合）和Reducer
     */
    public static class WordReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

        private final IntWritable result = new IntWritable();

        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            // 累加所有计数
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    /**
     * Driver程序 - 配置并提交MapReduce作业
     */
    public static void main(String[] args) throws Exception {
        // 参数校验
        if (args.length != 2) {
            System.err.println("用法: WordCount <输入路径> <输出路径>");
            System.exit(1);
        }

        // 创建Configuration和Job
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "WordCount-单词计数");

        // 设置Jar包（通过主类定位）
        job.setJarByClass(WordCount.class);

        // 设置Mapper
        job.setMapperClass(WordMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        // 设置Combiner（Map端本地预聚合，减少网络传输）
        job.setCombinerClass(WordReducer.class);

        // 设置Reducer
        job.setReducerClass(WordReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        // 设置Reduce任务数量（根据数据量和集群规模调整）
        job.setNumReduceTasks(2);

        // 设置输入输出路径
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        // 提交作业并等待完成
        boolean success = job.waitForCompletion(true);
        System.exit(success ? 0 : 1);
    }
}
```

**编译与运行**：

```bash
# 编译打包
mvn clean package -DskipTests

# 准备输入数据
hdfs dfs -mkdir -p /wordcount/input
hdfs dfs -put sample.txt /wordcount/input/

# 提交MapReduce作业
hadoop jar wordcount-1.0.jar com.example.WordCount \
    /wordcount/input /wordcount/output

# 查看结果
hdfs dfs -cat /wordcount/output/part-r-00000
hdfs dfs -cat /wordcount/output/part-r-00001
```

### 5.3 MapReduce vs Spark对比

| 维度 | MapReduce | Apache Spark |
|------|-----------|-------------|
| **执行模型** | 基于磁盘的两阶段模型(Map+Reduce) | 基于内存的DAG执行引擎 |
| **处理速度** | 慢（中间结果写磁盘） | 快10-100倍（内存计算） |
| **迭代计算** | 每轮迭代都读写HDFS | 数据缓存在内存，迭代高效 |
| **编程模型** | 仅Map和Reduce两个算子 | 丰富的算子(map/filter/join/groupBy等) |
| **开发效率** | 低（大量样板代码） | 高（函数式API简洁） |
| **实时处理** | 不支持（纯批处理） | 支持微批和结构化流 |
| **内存需求** | 低（依赖磁盘） | 高（依赖内存） |
| **容错机制** | 重新执行失败的Task | 基于RDD Lineage重算 |
| **适用场景** | 超大规模离线ETL | 交互式查询、ML、流处理 |
| **社区活跃** | 维护模式 | 非常活跃 |

**选型建议**：

- 新项目优先选择Spark，性能和开发效率全面领先
- MapReduce仍适合超大数据量的简单ETL（稳定、资源消耗低）
- 现有MapReduce作业可逐步迁移到Spark

---

## 6. 实战案例：日志存储与分析平台

### 6.1 需求与架构

构建一个完整的日志收集、存储和分析平台，处理Web服务器每天产生的TB级日志数据。

**整体架构**：

```
日志分析平台架构
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   数据源                                                            │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐                           │
│   │Web Server│ │App Server│ │ Nginx    │                           │
│   │ 日志文件  │ │ 日志文件  │ │ 日志文件  │                           │
│   └────┬─────┘ └────┬─────┘ └────┬─────┘                           │
│        │             │            │                                  │
│        └──────────┬──┘────────────┘                                  │
│                   ↓                                                  │
│   采集层   ┌──────────────────┐                                      │
│           │ Flume / Filebeat │    实时采集日志文件                    │
│           └────────┬─────────┘                                      │
│                    ↓                                                 │
│   缓冲层   ┌──────────────────┐                                      │
│           │  Kafka (消息队列) │    削峰填谷，解耦生产消费             │
│           └────────┬─────────┘                                      │
│                    │                                                 │
│         ┌──────────┼──────────────┐                                  │
│         ↓                         ↓                                  │
│   存储层                    实时处理层                                │
│   ┌──────────────┐    ┌──────────────┐                               │
│   │     HDFS     │    │ Flink/Spark  │    实时告警、实时大盘          │
│   │  按天分区存储 │    │  Streaming   │                               │
│   └──────┬───────┘    └──────┬───────┘                               │
│          │                   ↓                                       │
│          │            ┌──────────────┐                                │
│          │            │   Redis/ES   │    实时指标存储                 │
│          │            └──────────────┘                                │
│          ↓                                                           │
│   计算层                                                             │
│   ┌──────────────┐                                                   │
│   │ Spark / Hive │    离线ETL、指标聚合                               │
│   └──────┬───────┘                                                   │
│          ↓                                                           │
│   服务层                                                             │
│   ┌──────────────┐  ┌──────────────┐                                 │
│   │   Hive表     │  │ BI工具/报表  │    数据分析与可视化              │
│   │ (ORC/Parquet)│  │(Grafana/Superset)│                             │
│   └──────────────┘  └──────────────┘                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 HDFS存储设计

**目录分区策略**：

按照`年/月/日`进行目录分区，便于数据管理和查询裁剪：

```bash
# HDFS目录结构设计
/data/
├── logs/
│   ├── raw/                          # 原始日志（Flume写入）
│   │   ├── 2025/
│   │   │   ├── 01/
│   │   │   │   ├── 01/
│   │   │   │   │   ├── access_log_00.gz
│   │   │   │   │   ├── access_log_01.gz
│   │   │   │   │   └── ...
│   │   │   │   ├── 02/
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   ├── cleaned/                      # 清洗后的结构化数据
│   │   ├── dt=2025-01-01/
│   │   │   ├── part-00000.parquet
│   │   │   └── part-00001.parquet
│   │   └── ...
│   └── aggregated/                   # 聚合结果
│       ├── daily/
│       ├── weekly/
│       └── monthly/
├── dim/                              # 维度表
│   ├── geo/                          # 地理信息
│   └── user/                         # 用户信息
└── tmp/                              # 临时计算数据
```

**压缩策略**：

| 数据层 | 压缩格式 | 压缩比 | 速度 | 是否可分割 | 适用场景 |
|--------|---------|--------|------|-----------|---------|
| 原始日志（热数据） | Snappy | 中等 | 极快 | 否（需配合容器格式） | 频繁读写，低延迟 |
| 清洗数据（温数据） | LZ4 | 中等 | 很快 | 否 | 常规查询分析 |
| 归档数据（冷数据） | GZIP | 高 | 慢 | 否 | 长期存储，节省空间 |
| 列式文件 | Parquet+Zstd | 很高 | 快 | 是（按RowGroup） | Hive/Spark分析查询 |

**hdfs-site.xml 存储核心参数**：

```xml
<configuration>
    <!-- Block大小：大文件建议256MB，减少NameNode元数据压力 -->
    <property>
        <name>dfs.blocksize</name>
        <value>268435456</value>
        <description>Block大小256MB（默认128MB），适用于大文件场景</description>
    </property>

    <!-- 默认副本数 -->
    <property>
        <name>dfs.replication</name>
        <value>3</value>
        <description>默认3副本，冷数据可通过命令调整为2或使用EC</description>
    </property>

    <!-- DataNode磁盘目录（多磁盘配置提升吞吐） -->
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>/data/disk1/hdfs/data,/data/disk2/hdfs/data,/data/disk3/hdfs/data</value>
        <description>配置多块磁盘，HDFS自动轮询写入，提升并行IO</description>
    </property>

    <!-- 垃圾回收站配置（防止误删） -->
    <property>
        <name>fs.trash.interval</name>
        <value>1440</value>
        <description>回收站保留时间（分钟），1440分钟=1天</description>
    </property>

    <!-- Short-Circuit Local Read（本地短路读取，提升同节点读性能） -->
    <property>
        <name>dfs.client.read.shortcircuit</name>
        <value>true</value>
    </property>
    <property>
        <name>dfs.domain.socket.path</name>
        <value>/var/lib/hadoop-hdfs/dn_socket</value>
    </property>

    <!-- 权限检查 -->
    <property>
        <name>dfs.permissions.enabled</name>
        <value>true</value>
    </property>

    <!-- 数据传输最大线程数（高负载集群需调大） -->
    <property>
        <name>dfs.datanode.max.transfer.threads</name>
        <value>8192</value>
        <description>DataNode数据传输最大线程数（默认4096）</description>
    </property>
</configuration>
```

### 6.3 数据生命周期管理

HDFS支持异构存储策略，将不同热度的数据存放在不同存储介质上：

```
存储分层策略
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  热数据 (0-7天)         温数据 (7-30天)      冷数据 (30天+) │
│  ┌─────────────┐       ┌─────────────┐     ┌─────────────┐ │
│  │    SSD      │  ──→  │    HDD      │ ──→ │  Archive    │ │
│  │ (ALL_SSD)   │       │  (DEFAULT)  │     │ (COLD)      │ │
│  │ 高IOPS      │       │ 大容量      │     │ 低成本      │ │
│  │ 实时查询    │       │ 常规分析    │     │ 合规存储    │ │
│  └─────────────┘       └─────────────┘     └─────────────┘ │
│                                                             │
│  策略: HOT → WARM → COLD → 删除                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**HDFS存储策略配置与命令**：

```bash
# 查看可用的存储策略
hdfs storagepolicies -listPolicies
# 输出:
# Block Storage Policies:
#   HOT:       全部放在DISK
#   WARM:      1副本DISK + 其余ARCHIVE
#   COLD:      全部放在ARCHIVE
#   ALL_SSD:   全部放在SSD
#   ONE_SSD:   1副本SSD + 其余DISK
#   LAZY_PERSIST: 先写内存(RAM_DISK)再落盘

# 为热数据目录设置SSD策略
hdfs storagepolicies -setStoragePolicy -path /data/logs/raw -policy HOT

# 为清洗后数据设置默认策略（HDD）
hdfs storagepolicies -setStoragePolicy -path /data/logs/cleaned -policy WARM

# 为归档数据设置冷存策略
hdfs storagepolicies -setStoragePolicy -path /data/logs/archive -policy COLD

# 查看目录当前存储策略
hdfs storagepolicies -getStoragePolicy -path /data/logs/raw

# 触发数据迁移（按策略将数据移动到对应存储介质）
hdfs mover -p /data/logs/

# ==================== 数据生命周期自动化脚本 ====================

#!/bin/bash
# 日志数据生命周期管理脚本 (由Cron定时执行)

TODAY=$(date +%Y-%m-%d)
DAYS_7_AGO=$(date -d "7 days ago" +%Y/%m/%d)
DAYS_30_AGO=$(date -d "30 days ago" +%Y/%m/%d)
DAYS_90_AGO=$(date -d "90 days ago" +%Y/%m/%d)

# 7天前的数据：从HOT降级为WARM
echo "将7天前的数据降级为WARM策略..."
hdfs storagepolicies -setStoragePolicy \
    -path /data/logs/raw/${DAYS_7_AGO} -policy WARM

# 30天前的数据：从WARM降级为COLD
echo "将30天前的数据降级为COLD策略..."
if hdfs dfs -test -d /data/logs/raw/${DAYS_30_AGO}; then
    hdfs storagepolicies -setStoragePolicy \
        -path /data/logs/raw/${DAYS_30_AGO} -policy COLD
fi

# 90天前的原始数据：删除（清洗后的数据已存在）
echo "删除90天前的原始日志..."
if hdfs dfs -test -d /data/logs/raw/${DAYS_90_AGO}; then
    hdfs dfs -rm -r /data/logs/raw/${DAYS_90_AGO}
    echo "已删除: /data/logs/raw/${DAYS_90_AGO}"
fi

# 触发数据迁移
echo "触发Mover执行数据迁移..."
hdfs mover -p /data/logs/

echo "数据生命周期管理完成: ${TODAY}"
```

---

## 7. 运维与最佳实践

### 7.1 集群规划

**硬件配置建议**：

| 角色 | CPU | 内存 | 磁盘 | 网络 | 数量 |
|------|-----|------|------|------|------|
| **NameNode** | 16-32核 | 64-256GB | 2x SSD (RAID1) | 万兆 | 2 (HA) |
| **JournalNode** | 4-8核 | 8-16GB | 1x SSD | 千兆/万兆 | 3 (奇数) |
| **DataNode** | 8-16核 | 64-128GB | 12x 4TB HDD (JBOD) | 万兆 | 按容量需求 |
| **ResourceManager** | 16-32核 | 64-128GB | 2x SSD | 万兆 | 2 (HA) |
| **ZooKeeper** | 4-8核 | 8-16GB | 1x SSD | 千兆/万兆 | 3或5 (奇数) |

**NameNode内存规划**：

NameNode将所有元数据保存在内存中，内存大小直接决定了集群可管理的文件规模：

```
NameNode内存计算公式
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  每个文件对象约 150 bytes                                     │
│  每个Block对象约 150 bytes                                    │
│  每个Block副本引用约 16 bytes                                 │
│                                                              │
│  经验公式:                                                    │
│  NameNode Heap ≈ 文件数 x 150B + Block数 x 150B              │
│                + Block数 x 副本数 x 16B + 系统开销            │
│                                                              │
│  简化估算: 每1百万个Block约需 1GB Heap                        │
│                                                              │
│  示例:                                                        │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ 集群总数据: 1PB                                      │    │
│  │ Block大小:  128MB                                    │    │
│  │ Block数量:  1PB / 128MB ≈ 800万                     │    │
│  │ 副本数:     3                                        │    │
│  │ NameNode Heap: 约 8-12GB (含系统开销)               │    │
│  │                                                      │    │
│  │ 如果大量小文件 (平均1MB/文件):                       │    │
│  │ 文件数: 1PB / 1MB ≈ 10亿                            │    │
│  │ NameNode Heap: 约 150GB+ (小文件问题!)              │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  结论: 避免大量小文件是HDFS运维的首要原则                     │
└──────────────────────────────────────────────────────────────┘
```

**小文件问题与解决方案**：

小文件（远小于Block大小的文件）是HDFS最常见的性能杀手：

| 问题 | 影响 | 解决方案 |
|------|------|---------|
| NameNode内存压力 | 每个文件至少占150B元数据 | 合并小文件为大文件 |
| Map任务过多 | 每个小文件产生一个Map Task | 使用CombineFileInputFormat |
| 读取效率低 | 大量随机IO、频繁寻址 | 使用SequenceFile/Avro容器 |
| Block利用率低 | 1KB文件仍占一个Block槽位 | Hadoop Archive (HAR) |

```bash
# 使用Hadoop Archive合并小文件
# 将/data/small_files/目录下的小文件打包为HAR归档
hadoop archive -archiveName logs.har -p /data/small_files/ /data/archive/

# 访问HAR中的文件
hdfs dfs -ls har:///data/archive/logs.har/
hdfs dfs -cat har:///data/archive/logs.har/file1.txt
```

### 7.2 性能调优

**关键参数调优清单**：

```
HDFS性能调优参数
┌──────────────────────────────────────────────────────────────┐
│  参数                                 │ 默认值  │ 建议值    │
├───────────────────────────────────────┼─────────┼───────────┤
│ dfs.blocksize                         │ 128MB   │ 256MB     │
│ (大文件场景增大Block，减少NameNode压力) │         │ (大文件)  │
├───────────────────────────────────────┼─────────┼───────────┤
│ dfs.namenode.handler.count            │ 10      │ 100-200   │
│ (NN处理RPC请求的线程数)               │         │           │
├───────────────────────────────────────┼─────────┼───────────┤
│ dfs.datanode.handler.count            │ 10      │ 20-50     │
│ (DN处理RPC请求的线程数)               │         │           │
├───────────────────────────────────────┼─────────┼───────────┤
│ dfs.datanode.max.transfer.threads     │ 4096    │ 8192      │
│ (DN数据传输最大线程数)                │         │           │
├───────────────────────────────────────┼─────────┼───────────┤
│ dfs.client.read.shortcircuit          │ false   │ true      │
│ (本地短路读取，跳过DN网络传输)        │         │           │
├───────────────────────────────────────┼─────────┼───────────┤
│ dfs.namenode.edits.noeditlogchannelflush │ false│ true      │
│ (禁用EditLog通道刷写，提升写入吞吐)   │         │ (有风险)  │
├───────────────────────────────────────┼─────────┼───────────┤
│ io.file.buffer.size                   │ 4096    │ 131072    │
│ (读写缓冲区大小)                     │         │ (128KB)   │
└───────────────────────────────────────┴─────────┴───────────┘
```

**NameNode调优**：

```xml
<!-- core-site.xml -->
<configuration>
    <!-- 增大IO缓冲区 -->
    <property>
        <name>io.file.buffer.size</name>
        <value>131072</value>
    </property>
</configuration>

<!-- hdfs-site.xml -->
<configuration>
    <!-- NameNode处理RPC线程数 (经验公式: 20 * ln(集群节点数)) -->
    <property>
        <name>dfs.namenode.handler.count</name>
        <value>128</value>
    </property>

    <!-- NameNode服务线程数 -->
    <property>
        <name>dfs.namenode.service.handler.count</name>
        <value>50</value>
    </property>

    <!-- NameNode JVM参数 (在hadoop-env.sh中配置) -->
    <!-- export HDFS_NAMENODE_OPTS="-Xmx100g -XX:+UseG1GC -XX:MaxGCPauseMillis=200" -->
</configuration>
```

**DataNode调优**：

```xml
<configuration>
    <!-- 增大数据传输线程数（高并发读写场景） -->
    <property>
        <name>dfs.datanode.max.transfer.threads</name>
        <value>8192</value>
    </property>

    <!-- 多磁盘数据目录（JBOD配置，不要RAID） -->
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>[SSD]/data/ssd/hdfs,[DISK]/data/disk1/hdfs,[DISK]/data/disk2/hdfs</value>
    </property>

    <!-- 磁盘故障容忍数（允许N块磁盘故障，DN不下线） -->
    <property>
        <name>dfs.datanode.failed.volumes.tolerated</name>
        <value>2</value>
    </property>
</configuration>
```

**常见调优示例（正确与错误做法）**：

Block大小配置：

```xml
<!-- ✅ 正确: 大文件(GB级)使用256MB Block -->
<property>
    <name>dfs.blocksize</name>
    <value>268435456</value>
</property>

<!-- ❌ 错误: 小文件场景设置过大Block，浪费元数据空间 -->
<!-- 大量1MB小文件用256MB Block并不会节省元数据，因为每个文件至少1个Block -->
```

副本数配置：

```bash
# ✅ 正确: 对冷数据降低副本数节省空间
hdfs dfs -setrep -w 2 /data/logs/archive/

# ✅ 正确: 使用Erasure Coding代替3副本 (Hadoop 3.x)
hdfs ec -enablePolicy -policy RS-6-3-1024k
hdfs ec -setPolicy -path /data/logs/archive -policy RS-6-3-1024k

# ❌ 错误: 将生产热数据副本设为1（任何节点故障即丢数据）
# hdfs dfs -setrep 1 /data/production/  # 千万不要这样做!
```

### 7.3 常见故障处理

**故障排查检查清单**：

**SafeMode无法退出**：

```bash
# 1. 查看SafeMode状态和原因
hdfs dfsadmin -safemode get
# 输出示例: Safe mode is ON. The reported blocks 99990 has not reached
#          the threshold 0.999 of total blocks 100000.

# 2. 查看集群Block状态
hdfs dfsadmin -report | head -20

# 3. 检查是否有DataNode未启动
hdfs dfsadmin -printTopology

# 4. 如果Block确认完整，手动退出SafeMode
hdfs dfsadmin -safemode leave

# 5. 如果有Block缺失，先等DataNode全部上线再退出
```

**Block损坏（Corrupt Blocks）**：

```bash
# 1. 使用fsck检查文件系统健康状态
hdfs fsck / -files -blocks -locations

# 2. 仅查看损坏的Block
hdfs fsck / -list-corruptfileblocks
# 输出: The filesystem under path '/' has N CORRUPT files

# 3. 查看具体损坏文件
hdfs fsck /data/logs/ -files -blocks -locations | grep -i corrupt

# 4. 方案A: 如果有足够副本，删除损坏副本触发自动复制
hdfs debug recoverLease -path /data/logs/damaged_file.txt -retries 3

# 5. 方案B: 如果数据可重新生成，删除损坏文件
hdfs dfs -rm /data/logs/corrupted_file.txt

# 6. 方案C: 移除所有损坏文件（慎用）
hdfs fsck / -delete
```

**DataNode退役（Decommission）**：

```bash
# 1. 在NameNode的exclude文件中添加要退役的节点
# 编辑 dfs.hosts.exclude 指定的文件
echo "datanode-to-remove.example.com" >> /etc/hadoop/conf/dfs.exclude

# 2. 刷新节点列表（NameNode会开始迁移该节点上的Block）
hdfs dfsadmin -refreshNodes

# 3. 监控退役进度
hdfs dfsadmin -report | grep -A 5 "datanode-to-remove"
# 状态变化: Normal → Decommission In Progress → Decommissioned

# 4. 等待所有Block迁移完毕后，安全停止DataNode进程
# 当状态变为 Decommissioned 后:
ssh datanode-to-remove "hdfs --daemon stop datanode"

# 注意: 退役过程中不要直接Kill进程，否则会导致Block副本丢失
```

**NameNode Failover（故障切换）**：

```bash
# 1. 查看当前Active/Standby状态
hdfs haadmin -getServiceState nn1
hdfs haadmin -getServiceState nn2
# 输出: active 或 standby

# 2. 手动触发Failover（将Active切换到另一个NN）
hdfs haadmin -failover nn1 nn2
# nn1(当前Active) → Standby, nn2(当前Standby) → Active

# 3. 如果自动Failover不工作，检查ZKFC状态
# 在两个NameNode节点上分别检查:
jps | grep DFSZKFailoverController

# 4. 检查ZooKeeper连接状态
echo ruok | nc zk1 2181  # 应返回 imok

# 5. 如果NN1彻底无法恢复，强制将NN2切换为Active
hdfs haadmin -transitionToActive --forcemanual nn2
# 警告: 仅在确认NN1已完全停止的情况下使用，否则可能导致脑裂(Split-Brain)
```

**运维监控要点**：

```
日常运维监控清单
┌──────────────────────────────────────────────────────────────┐
│  监控维度         │  告警阈值            │  检查命令           │
├───────────────────┼──────────────────────┼─────────────────────┤
│ NameNode Heap使用 │ > 80% 告警           │ jstat -gcutil <pid> │
│ HDFS磁盘使用率    │ > 85% 告警, >90% 严重│ hdfs dfsadmin -report│
│ Dead DataNode数   │ > 0 告警             │ hdfs dfsadmin -report│
│ Under-replicated  │ > 0 持续30分钟       │ hdfs fsck / -blocks │
│ Corrupt Blocks    │ > 0 立即告警         │ hdfs fsck /          │
│ SafeMode状态      │ ON 持续5分钟         │ hdfs dfsadmin -safemode get│
│ RPC队列长度       │ > 100 告警           │ NN WebUI (9870)     │
│ Block Report耗时  │ > 10分钟             │ NN日志              │
│ 磁盘IO Util       │ > 90% 持续5分钟      │ iostat -x 1         │
│ 网络带宽          │ > 80% 万兆           │ iftop / nload       │
└───────────────────┴──────────────────────┴─────────────────────┘
```

**生产环境运维最佳实践总结**：

1. **NameNode HA是必须的** - 不要在生产环境运行单NameNode，任何硬件故障都会导致集群不可用
2. **避免小文件** - 小文件是HDFS最大的敌人，务必在写入前合并，或使用SequenceFile/ORC/Parquet等容器格式
3. **定期Balancer** - 新节点加入后必须运行Balancer，否则新节点闲置、旧节点过载
4. **监控磁盘健康** - DataNode磁盘故障是最常见的硬件问题，配置`dfs.datanode.failed.volumes.tolerated`容忍部分磁盘故障
5. **合理设置回收站** - 启用`fs.trash.interval`防止误删，但不要设置过长导致空间浪费
6. **定期fsck** - 每周至少一次全量`hdfs fsck`检查Block健康状态
7. **EditLog与JournalNode监控** - HA模式下JournalNode写入延迟直接影响NameNode性能
8. **滚动升级** - 大版本升级使用Rolling Upgrade，先升级NameNode，再逐批升级DataNode
