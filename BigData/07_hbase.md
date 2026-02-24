# HBase分布式NoSQL数据库实战

## 1. HBase架构与数据模型

### 1.1 整体架构

```
HBase集群架构
┌─────────────────────────────────────────────────────────────┐
│                       Client                                │
│              (HBase Shell / Java API / Thrift)               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                     ZooKeeper集群                            │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐               │
│  │    ZK-1   │  │    ZK-2   │  │    ZK-3   │               │
│  └───────────┘  └───────────┘  └───────────┘               │
│  功能: HMaster选举、RegionServer注册、META表定位             │
└─────────────────────────┬───────────────────────────────────┘
                          │
              ┌───────────┼───────────┐
              ↓                       ↓
┌──────────────────────┐  ┌──────────────────────────────────┐
│     HMaster          │  │        RegionServer集群           │
│  ┌────────────────┐  │  │  ┌──────────────────────────┐    │
│  │ 表DDL操作      │  │  │  │    RegionServer-1        │    │
│  │ Region分配     │  │  │  │  ┌────────┐ ┌────────┐   │    │
│  │ 负载均衡       │  │  │  │  │Region-A│ │Region-B│   │    │
│  │ 故障恢复       │  │  │  │  └────────┘ └────────┘   │    │
│  └────────────────┘  │  │  └──────────────────────────┘    │
│  (不参与数据读写)     │  │  ┌──────────────────────────┐    │
└──────────────────────┘  │  │    RegionServer-2        │    │
                          │  │  ┌────────┐ ┌────────┐   │    │
                          │  │  │Region-C│ │Region-D│   │    │
                          │  │  └────────┘ └────────┘   │    │
                          │  └──────────────────────────┘    │
                          └──────────────────────┬───────────┘
                                                 │
                                                 ↓
                          ┌──────────────────────────────────┐
                          │              HDFS                 │
                          │  (底层存储：HFile + WAL)          │
                          └──────────────────────────────────┘
```

**核心组件职责**：

| 组件 | 职责 | 数量 |
|------|------|------|
| **HMaster** | DDL操作、Region分配、负载均衡、故障恢复 | 1 Active + 1+ Standby |
| **RegionServer** | Region管理、数据读写、MemStore刷写、Compaction | 多个 |
| **ZooKeeper** | HMaster选举、RS注册、META表位置 | 奇数（3/5/7） |
| **HDFS** | 底层持久化存储（HFile + WAL） | 独立集群 |

### 1.2 数据模型

```
HBase数据模型
┌─────────────────────────────────────────────────────────────┐
│  Table: user_behavior                                       │
├──────────┬──────────────────────┬──────────────────────────┤
│          │   Column Family: info │  Column Family: event    │
│  RowKey  ├──────┬──────┬────────┼──────┬──────┬────────────┤
│          │ name │ age  │ city   │ type │ page │ duration   │
├──────────┼──────┼──────┼────────┼──────┼──────┼────────────┤
│ user_001 │ 张三 │  25  │ 北京   │ click│ /home│    30      │
│          │      │      │        │ view │ /item│    45      │
├──────────┼──────┼──────┼────────┼──────┼──────┼────────────┤
│ user_002 │ 李四 │  30  │ 上海   │ buy  │ /pay │    60      │
└──────────┴──────┴──────┴────────┴──────┴──────┴────────────┘

逻辑视图: {RowKey, Column Family, Column Qualifier, Timestamp} → Value

物理存储（按Column Family分开存储）:
  info:  user_001/info:name/t3 → "张三"
         user_001/info:age/t2  → "25"
  event: user_001/event:type/t5 → "click"
         user_001/event:type/t3 → "view"  (多版本)
```

**HBase vs RDBMS对比**：

| 维度 | HBase | RDBMS (MySQL) |
|------|-------|---------------|
| **数据模型** | 稀疏多维Map | 二维关系表 |
| **Schema** | 动态列，无需预定义 | 固定Schema |
| **数据规模** | PB级，十亿行 | TB级，千万行 |
| **查询方式** | RowKey + Scan | SQL任意查询 |
| **事务** | 行级原子性 | ACID完整事务 |
| **Join** | 不支持 | 原生支持 |
| **扩展性** | 线性水平扩展 | 垂直扩展为主 |
| **适用场景** | 海量数据随机读写 | 结构化事务处理 |

### 1.3 存储原理

```
HBase存储引擎 (LSM-Tree)

写入流程:
┌──────────┐    ┌──────────────┐    ┌──────────────┐
│  Client   │───→│   WAL(HLog)  │───→│  MemStore    │
│  写请求   │    │  (预写日志)   │    │  (内存缓冲)  │
└──────────┘    └──────────────┘    └──────┬───────┘
                                          │ 达到阈值
                                          │ (128MB)
                                          ↓ Flush
                                   ┌──────────────┐
                                   │   HFile       │
                                   │  (HDFS文件)   │
                                   └──────┬───────┘
                                          │ 积累多个
                                          ↓ Compaction
                               ┌────────────────────┐
                               │  Minor Compaction   │
                               │ (合并小HFile)       │
                               └──────────┬─────────┘
                                          │
                                          ↓
                               ┌────────────────────┐
                               │  Major Compaction   │
                               │ (合并所有HFile,     │
                               │  清理删除/过期数据) │
                               └────────────────────┘

HFile内部结构:
┌─────────────────────────────────────┐
│           HFile v3                  │
├─────────────────────────────────────┤
│  Data Block 1  (64KB, 有序KV对)    │
│  Data Block 2                       │
│  ...                                │
│  Data Block N                       │
├─────────────────────────────────────┤
│  Meta Block (Bloom Filter)          │
├─────────────────────────────────────┤
│  File Info                          │
├─────────────────────────────────────┤
│  Data Block Index                   │
├─────────────────────────────────────┤
│  Meta Block Index                   │
├─────────────────────────────────────┤
│  Trailer (固定长度, 指向各索引)     │
└─────────────────────────────────────┘
```

**WAL (Write-Ahead Log)**：
- 所有写操作先写WAL，再写MemStore
- RegionServer宕机时通过WAL回放恢复数据
- 每个RegionServer一个WAL文件（可配置多个）

## 2. HBase Shell操作

### 2.1 命名空间与表操作

```bash
# 命名空间管理
create_namespace 'bigdata'
list_namespace
describe_namespace 'bigdata'
drop_namespace 'bigdata'   # 必须先删除其中所有表

# 建表 - 基础
create 'bigdata:user_behavior', 'info', 'event'

# 建表 - 高级配置
create 'bigdata:user_behavior',
  {NAME => 'info', VERSIONS => 3, TTL => 7776000,           # 90天TTL
   COMPRESSION => 'SNAPPY', BLOOMFILTER => 'ROW',
   BLOCKSIZE => '65536', MIN_VERSIONS => 1},
  {NAME => 'event', VERSIONS => 5, TTL => 2592000,          # 30天TTL
   COMPRESSION => 'LZ4', BLOOMFILTER => 'ROWCOL',
   BLOCKSIZE => '65536', IN_MEMORY => 'true'},
  {NUMREGIONS => 16, SPLITALGO => 'HexStringSplit'}          # 预分区

# 表管理
list                                    # 列出所有表
list_namespace_tables 'bigdata'         # 列出命名空间下的表
describe 'bigdata:user_behavior'        # 查看表结构
alter 'bigdata:user_behavior', {NAME => 'info', VERSIONS => 5}  # 修改列族
disable 'bigdata:user_behavior'         # 禁用表（修改/删除前必须）
enable 'bigdata:user_behavior'          # 启用表
is_enabled 'bigdata:user_behavior'      # 检查状态
drop 'bigdata:user_behavior'            # 删除表（必须先disable）
```

### 2.2 数据操作

```bash
# 插入数据
put 'bigdata:user_behavior', 'user_001', 'info:name', '张三'
put 'bigdata:user_behavior', 'user_001', 'info:age', '25'
put 'bigdata:user_behavior', 'user_001', 'info:city', '北京'
put 'bigdata:user_behavior', 'user_001', 'event:type', 'click'
put 'bigdata:user_behavior', 'user_001', 'event:page', '/home'
put 'bigdata:user_behavior', 'user_002', 'info:name', '李四'
put 'bigdata:user_behavior', 'user_002', 'event:type', 'buy'

# 读取数据
get 'bigdata:user_behavior', 'user_001'                        # 获取整行
get 'bigdata:user_behavior', 'user_001', 'info'                # 获取指定列族
get 'bigdata:user_behavior', 'user_001', 'info:name'           # 获取指定列
get 'bigdata:user_behavior', 'user_001', {COLUMN => 'info:name', VERSIONS => 3}  # 多版本

# 扫描数据
scan 'bigdata:user_behavior'                                    # 全表扫描
scan 'bigdata:user_behavior', {LIMIT => 10}                    # 限制行数
scan 'bigdata:user_behavior', {STARTROW => 'user_001', STOPROW => 'user_003'}  # 范围扫描
scan 'bigdata:user_behavior', {COLUMNS => ['info:name', 'event:type']}          # 指定列

# 使用Filter
scan 'bigdata:user_behavior', {
  FILTER => "SingleColumnValueFilter('info', 'city', =, 'binary:北京')"
}
scan 'bigdata:user_behavior', {
  FILTER => "PrefixFilter('user_00')"
}
scan 'bigdata:user_behavior', {
  FILTER => "RowFilter(>=, 'binary:user_002')"
}
# 组合Filter
scan 'bigdata:user_behavior', {
  FILTER => "SingleColumnValueFilter('info', 'age', >=, 'binary:25') AND PrefixFilter('user_')"
}

# 删除数据
delete 'bigdata:user_behavior', 'user_001', 'info:city'         # 删除某列
deleteall 'bigdata:user_behavior', 'user_001'                   # 删除整行

# 计数
count 'bigdata:user_behavior'
count 'bigdata:user_behavior', {INTERVAL => 1000}               # 每1000行打印进度

# 清空表
truncate 'bigdata:user_behavior'
```

### 2.3 管理命令

```bash
# 集群状态
status                          # 集群概览
status 'simple'                 # 简要状态
status 'detailed'               # 详细状态

# Region管理
balance_switch true              # 开启自动均衡
balancer                         # 手动触发均衡
move 'REGION_ENCODED_NAME', 'SERVER_NAME'  # 手动移动Region

# Compaction
flush 'bigdata:user_behavior'              # 强制刷写MemStore
compact 'bigdata:user_behavior'            # Minor Compaction
major_compact 'bigdata:user_behavior'      # Major Compaction

# 快照
snapshot 'bigdata:user_behavior', 'snap_20260224'    # 创建快照
list_snapshots                                        # 列出快照
clone_snapshot 'snap_20260224', 'user_behavior_bak'  # 克隆快照为新表
restore_snapshot 'snap_20260224'                      # 恢复快照（表必须先disable）
delete_snapshot 'snap_20260224'                       # 删除快照
```

## 3. Java API编程

### 3.1 连接管理

```java
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.TableName;
import org.apache.hadoop.hbase.client.*;

/**
 * HBase连接管理 - 最佳实践
 * Connection是线程安全的重量级对象，应全局复用
 * Table是轻量级对象，用完即关
 */
public class HBaseConnectionManager {

    private static volatile Connection connection;

    // 获取全局Connection（单例模式）
    public static Connection getConnection() throws Exception {
        if (connection == null || connection.isClosed()) {
            synchronized (HBaseConnectionManager.class) {
                if (connection == null || connection.isClosed()) {
                    Configuration conf = HBaseConfiguration.create();
                    conf.set("hbase.zookeeper.quorum", "zk1,zk2,zk3");
                    conf.set("hbase.zookeeper.property.clientPort", "2181");
                    conf.set("hbase.client.retries.number", "3");
                    conf.set("hbase.client.operation.timeout", "30000");
                    connection = ConnectionFactory.createConnection(conf);
                }
            }
        }
        return connection;
    }

    // 获取Table实例（用完必须关闭）
    public static Table getTable(String tableName) throws Exception {
        return getConnection().getTable(TableName.valueOf(tableName));
    }

    // 获取Admin实例
    public static Admin getAdmin() throws Exception {
        return getConnection().getAdmin();
    }

    // 关闭连接（程序退出时调用）
    public static void close() throws Exception {
        if (connection != null && !connection.isClosed()) {
            connection.close();
        }
    }
}
```

### 3.2 CRUD操作

```java
import org.apache.hadoop.hbase.Cell;
import org.apache.hadoop.hbase.CellUtil;
import org.apache.hadoop.hbase.CompareOperator;
import org.apache.hadoop.hbase.client.*;
import org.apache.hadoop.hbase.filter.*;
import org.apache.hadoop.hbase.util.Bytes;

import java.util.ArrayList;
import java.util.List;

public class HBaseCrudExample {

    private static final String TABLE = "bigdata:user_behavior";
    private static final byte[] CF_INFO = Bytes.toBytes("info");
    private static final byte[] CF_EVENT = Bytes.toBytes("event");

    /**
     * 单行写入
     */
    public void putSingle() throws Exception {
        try (Table table = HBaseConnectionManager.getTable(TABLE)) {
            Put put = new Put(Bytes.toBytes("user_001"));
            put.addColumn(CF_INFO, Bytes.toBytes("name"), Bytes.toBytes("张三"));
            put.addColumn(CF_INFO, Bytes.toBytes("age"), Bytes.toBytes("25"));
            put.addColumn(CF_EVENT, Bytes.toBytes("type"), Bytes.toBytes("click"));
            table.put(put);
        }
    }

    /**
     * 批量写入（推荐，减少RPC次数）
     */
    public void putBatch() throws Exception {
        try (Table table = HBaseConnectionManager.getTable(TABLE)) {
            List<Put> puts = new ArrayList<>();
            for (int i = 0; i < 10000; i++) {
                Put put = new Put(Bytes.toBytes(String.format("user_%06d", i)));
                put.addColumn(CF_INFO, Bytes.toBytes("name"), Bytes.toBytes("用户" + i));
                put.addColumn(CF_INFO, Bytes.toBytes("age"), Bytes.toBytes(String.valueOf(20 + i % 40)));
                puts.add(put);

                // 每1000条提交一次，避免内存溢出
                if (puts.size() >= 1000) {
                    table.put(puts);
                    puts.clear();
                }
            }
            if (!puts.isEmpty()) {
                table.put(puts);
            }
        }
    }

    /**
     * 单行读取
     */
    public void getSingle() throws Exception {
        try (Table table = HBaseConnectionManager.getTable(TABLE)) {
            Get get = new Get(Bytes.toBytes("user_001"));
            get.addColumn(CF_INFO, Bytes.toBytes("name"));   // 只读指定列
            get.setMaxVersions(3);                            // 读多版本

            Result result = table.get(get);
            for (Cell cell : result.rawCells()) {
                System.out.printf("RowKey: %s, CF: %s, Qualifier: %s, Value: %s, Timestamp: %d%n",
                    Bytes.toString(CellUtil.cloneRow(cell)),
                    Bytes.toString(CellUtil.cloneFamily(cell)),
                    Bytes.toString(CellUtil.cloneQualifier(cell)),
                    Bytes.toString(CellUtil.cloneValue(cell)),
                    cell.getTimestamp());
            }
        }
    }

    /**
     * 带Filter的Scan
     */
    public void scanWithFilter() throws Exception {
        try (Table table = HBaseConnectionManager.getTable(TABLE)) {
            Scan scan = new Scan();
            scan.withStartRow(Bytes.toBytes("user_000000"));
            scan.withStopRow(Bytes.toBytes("user_001000"));

            // 组合Filter: 年龄>=25 AND 城市=北京
            FilterList filterList = new FilterList(FilterList.Operator.MUST_PASS_ALL);
            filterList.addFilter(new SingleColumnValueFilter(
                CF_INFO, Bytes.toBytes("age"),
                CompareOperator.GREATER_OR_EQUAL, Bytes.toBytes("25")
            ));
            filterList.addFilter(new SingleColumnValueFilter(
                CF_INFO, Bytes.toBytes("city"),
                CompareOperator.EQUAL, Bytes.toBytes("北京")
            ));
            scan.setFilter(filterList);

            // 设置缓存，提升Scan性能
            scan.setCaching(500);     // 每次RPC返回500行
            scan.setBatch(10);        // 每行最多返回10列

            try (ResultScanner scanner = table.getScanner(scan)) {
                for (Result result : scanner) {
                    String rowKey = Bytes.toString(result.getRow());
                    String name = Bytes.toString(result.getValue(CF_INFO, Bytes.toBytes("name")));
                    System.out.printf("RowKey: %s, Name: %s%n", rowKey, name);
                }
            }
        }
    }

    /**
     * 批量删除
     */
    public void deleteBatch() throws Exception {
        try (Table table = HBaseConnectionManager.getTable(TABLE)) {
            List<Delete> deletes = new ArrayList<>();
            deletes.add(new Delete(Bytes.toBytes("user_000001")));
            deletes.add(new Delete(Bytes.toBytes("user_000002")));

            // 删除指定列
            Delete delete = new Delete(Bytes.toBytes("user_000003"));
            delete.addColumn(CF_INFO, Bytes.toBytes("city"));  // 删除最新版本
            delete.addColumns(CF_INFO, Bytes.toBytes("age"));  // 删除所有版本
            deletes.add(delete);

            table.delete(deletes);
        }
    }
}
```

### 3.3 高级API

```java
/**
 * BulkLoad - 海量数据快速导入
 * 直接生成HFile，跳过WAL和MemStore，速度提升10-100倍
 */
import org.apache.hadoop.hbase.mapreduce.HFileOutputFormat2;
import org.apache.hadoop.hbase.tool.BulkLoadHFilesTool;
import org.apache.hadoop.mapreduce.Job;

public class HBaseBulkLoad {

    public static void bulkLoad() throws Exception {
        Configuration conf = HBaseConfiguration.create();
        Job job = Job.getInstance(conf, "BulkLoad");

        // 配置输入（CSV文件）
        FileInputFormat.addInputPath(job, new Path("/data/input"));

        // 配置Mapper（将CSV转为KeyValue）
        job.setMapperClass(CsvToHFileMapper.class);
        job.setMapOutputKeyClass(ImmutableBytesWritable.class);
        job.setMapOutputValueClass(KeyValue.class);

        // 配置HFile输出
        try (Connection conn = ConnectionFactory.createConnection(conf);
             Table table = conn.getTable(TableName.valueOf("user_behavior"))) {

            HFileOutputFormat2.configureIncrementalLoad(job, table, conn.getRegionLocator(
                TableName.valueOf("user_behavior")));
            FileOutputFormat.setOutputPath(job, new Path("/tmp/hfiles"));

            // 执行MapReduce生成HFile
            if (job.waitForCompletion(true)) {
                // 加载HFile到表
                BulkLoadHFilesTool loader = new BulkLoadHFilesTool(conf);
                loader.bulkLoad(TableName.valueOf("user_behavior"), new Path("/tmp/hfiles"));
            }
        }
    }
}
```

```java
/**
 * Coprocessor - 服务端计算（类似数据库触发器/存储过程）
 *
 * Observer: 类似触发器，在数据操作前后执行
 * Endpoint: 类似存储过程，自定义RPC调用
 */
// Observer Coprocessor 示例：自动记录修改时间
public class TimestampObserver implements RegionObserver, RegionCoprocessor {

    @Override
    public Optional<RegionObserver> getRegionObserver() {
        return Optional.of(this);
    }

    @Override
    public void prePut(ObserverContext<RegionCoprocessorEnvironment> c,
                       Put put, WALEdit edit, Durability durability) {
        // 自动添加修改时间列
        put.addColumn(
            Bytes.toBytes("info"),
            Bytes.toBytes("update_time"),
            Bytes.toBytes(System.currentTimeMillis())
        );
    }
}

// 加载Coprocessor（两种方式）
// 方式1：Shell
// alter 'user_behavior', METHOD => 'table_att',
//   'coprocessor' => 'hdfs:///coprocessor/timestamp-observer.jar|
//   com.example.TimestampObserver|1001|'

// 方式2：建表时指定
// create 'user_behavior', 'info', 'event',
//   {COPROCESSOR => 'hdfs:///coprocessor/timestamp-observer.jar|
//   com.example.TimestampObserver|1001|'}
```

## 4. RowKey设计与性能优化

### 4.1 RowKey设计原则

**三大原则**：
1. **长度原则**：尽量短（10-100字节），减少存储和比较开销
2. **唯一性原则**：RowKey必须能唯一标识一行数据
3. **散列原则**：避免热点，数据均匀分布到各Region

**热点问题与解决方案**：

```
❌ 热点问题：时间戳做RowKey
┌─────────────────────────────────────────────┐
│  RegionServer-1  │  RegionServer-2  │  RS-3  │
│  [空闲]          │  [空闲]          │ [满载] │
│                  │                  │ ←所有  │
│                  │                  │  新数据 │
└─────────────────────────────────────────────┘

✅ 解决方案1：加盐(Salting)
RowKey = hash(timestamp) % 10 + "_" + timestamp
结果：0_1708761600, 3_1708761601, 7_1708761602, 1_1708761603...
数据均匀分散到10个Region

✅ 解决方案2：哈希(Hashing)
RowKey = MD5(user_id).substring(0,8) + "_" + user_id + "_" + timestamp
结果：a1b2c3d4_user001_1708761600
优点：同一用户数据相邻，方便Scan

✅ 解决方案3：反转(Reversing)
RowKey = reverse(phone_number) + "_" + timestamp
138xxxx1234 → 4321xxxx831
避免相同前缀（138/139/137）导致热点
```

**不同场景的RowKey设计**：

| 场景 | RowKey设计 | 说明 |
|------|-----------|------|
| **用户事件** | `md5(uid)[0:8]_uid_reverse(ts)` | 同一用户数据相邻，最新数据先返回 |
| **IoT传感器** | `hash(device_id)%100_device_id_ts` | 加盐打散，设备维度查询 |
| **订单系统** | `reverse(order_id)` | 订单ID本身有序，反转打散 |
| **消息系统** | `md5(chat_id)[0:4]_chat_id_seq_id` | 同一会话消息连续存储 |

### 4.2 列族设计

```bash
# ✅ 推荐：1-3个列族
create 'user_profile',
  {NAME => 'basic', BLOOMFILTER => 'ROW', COMPRESSION => 'SNAPPY',
   BLOCKSIZE => '65536', VERSIONS => 1},                         # 基础信息，读多写少
  {NAME => 'stat', BLOOMFILTER => 'ROWCOL', COMPRESSION => 'LZ4',
   TTL => 2592000, VERSIONS => 1, IN_MEMORY => 'true'}           # 统计数据，频繁更新

# ❌ 避免：太多列族（导致频繁Flush和Compaction）
# create 'bad_table', 'cf1', 'cf2', 'cf3', 'cf4', 'cf5', 'cf6'
```

**列族参数选择**：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| VERSIONS | 1（除非需要多版本） | 版本数越多存储越大 |
| TTL | 按业务设置（秒） | 自动清理过期数据 |
| COMPRESSION | SNAPPY（平衡）/ LZ4（快速） | 减少存储和IO |
| BLOOMFILTER | ROW（Get）/ ROWCOL（Get指定列） | 减少不必要的磁盘读 |
| BLOCKSIZE | 65536（随机读）/ 131072（Scan多） | 影响索引粒度 |
| IN_MEMORY | true（热点数据） | 优先放入BlockCache |

### 4.3 读写优化

```
BlockCache架构
┌──────────────────────────────────────────────────┐
│              RegionServer JVM                     │
│  ┌────────────────────────────────────────────┐  │
│  │  LRUBlockCache (堆内)                      │  │
│  │  - 默认方式，占堆内存40%                   │  │
│  │  - Single(25%) + Multi(50%) + InMemory(25%)│  │
│  │  - 受GC影响大                              │  │
│  └────────────────────────────────────────────┘  │
│                    或                              │
│  ┌────────────────────────────────────────────┐  │
│  │  BucketCache (堆外)                        │  │
│  │  - 数据存放在堆外/SSD/文件                 │  │
│  │  - 不受GC影响                              │  │
│  │  - L1(堆内索引) + L2(堆外数据)            │  │
│  │  ✅ 推荐生产环境使用                       │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

**关键调优参数**：

```xml
<!-- hbase-site.xml 读写优化 -->

<!-- MemStore配置 -->
<property>
  <name>hbase.hregion.memstore.flush.size</name>
  <value>134217728</value>  <!-- 128MB，单个MemStore刷写阈值 -->
</property>
<property>
  <name>hbase.regionserver.global.memstore.size</name>
  <value>0.4</value>  <!-- MemStore总内存占比40% -->
</property>

<!-- BlockCache配置（BucketCache） -->
<property>
  <name>hbase.bucketcache.ioengine</name>
  <value>offheap</value>  <!-- 堆外内存 -->
</property>
<property>
  <name>hbase.bucketcache.size</name>
  <value>4096</value>  <!-- 4GB堆外缓存 -->
</property>

<!-- Compaction配置 -->
<property>
  <name>hbase.hstore.compaction.min</name>
  <value>3</value>  <!-- 触发Minor Compaction的最小HFile数 -->
</property>
<property>
  <name>hbase.hstore.compaction.max</name>
  <value>10</value>  <!-- 单次Minor Compaction最大HFile数 -->
</property>
<property>
  <name>hbase.hregion.majorcompaction</name>
  <value>0</value>  <!-- 关闭自动Major Compaction，改为手动低峰触发 -->
</property>
```

### 4.4 预分区

```bash
# 方式1：指定分区点
create 'user_behavior', 'info', 'event',
  SPLITS => ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']

# 方式2：使用HexStringSplit自动均匀分区
create 'user_behavior', 'info', 'event',
  {NUMREGIONS => 16, SPLITALGO => 'HexStringSplit'}

# 方式3：使用UniformSplit（字节均匀分布）
create 'user_behavior', 'info', 'event',
  {NUMREGIONS => 32, SPLITALGO => 'UniformSplit'}
```

```java
// Java API预分区
byte[][] splitKeys = new byte[][] {
    Bytes.toBytes("1"), Bytes.toBytes("2"), Bytes.toBytes("3"),
    Bytes.toBytes("4"), Bytes.toBytes("5"), Bytes.toBytes("6"),
    Bytes.toBytes("7"), Bytes.toBytes("8"), Bytes.toBytes("9"),
    Bytes.toBytes("a"), Bytes.toBytes("b"), Bytes.toBytes("c"),
    Bytes.toBytes("d"), Bytes.toBytes("e"), Bytes.toBytes("f")
};

TableDescriptor tableDesc = TableDescriptorBuilder.newBuilder(TableName.valueOf("user_behavior"))
    .setColumnFamily(ColumnFamilyDescriptorBuilder.of("info"))
    .setColumnFamily(ColumnFamilyDescriptorBuilder.of("event"))
    .build();

admin.createTable(tableDesc, splitKeys);
```

## 5. HBase与生态集成

### 5.1 HBase + Phoenix (SQL层)

```sql
-- Phoenix为HBase提供SQL查询能力

-- 创建表（自动映射到HBase表）
CREATE TABLE IF NOT EXISTS user_profile (
    user_id VARCHAR PRIMARY KEY,
    name VARCHAR,
    age INTEGER,
    city VARCHAR,
    total_orders BIGINT,
    last_login TIMESTAMP
) SALT_BUCKETS=16;             -- 自动加盐，避免热点

-- 插入数据
UPSERT INTO user_profile VALUES ('user_001', '张三', 25, '北京', 100, NOW());
UPSERT INTO user_profile VALUES ('user_002', '李四', 30, '上海', 50, NOW());

-- 查询
SELECT * FROM user_profile WHERE city = '北京' AND age > 20;

-- 聚合
SELECT city, COUNT(*) AS cnt, AVG(age) AS avg_age
FROM user_profile
GROUP BY city
ORDER BY cnt DESC;

-- 创建二级索引（HBase原生不支持，Phoenix实现）
CREATE INDEX idx_city ON user_profile(city) INCLUDE(name, age);

-- 使用索引查询
SELECT name, age FROM user_profile WHERE city = '上海';

-- 全局索引 vs 本地索引
-- 全局索引：读快写慢，适合读多写少
CREATE INDEX idx_global_city ON user_profile(city);
-- 本地索引：写快读慢，适合写多读少
CREATE LOCAL INDEX idx_local_age ON user_profile(age);
```

### 5.2 HBase + Spark

```scala
// Spark读写HBase（使用hbase-spark connector）
import org.apache.hadoop.hbase.spark.HBaseContext
import org.apache.hadoop.hbase.{HBaseConfiguration, TableName}
import org.apache.hadoop.hbase.client.{Put, Result, Scan}
import org.apache.hadoop.hbase.util.Bytes
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder()
  .appName("Spark-HBase")
  .getOrCreate()

val conf = HBaseConfiguration.create()
conf.set("hbase.zookeeper.quorum", "zk1,zk2,zk3")

val hbaseContext = new HBaseContext(spark.sparkContext, conf)

// 方式1：使用newAPIHadoopRDD读取
val scan = new Scan()
scan.addColumn(Bytes.toBytes("info"), Bytes.toBytes("name"))

val hbaseRDD = hbaseContext.hbaseRDD(
  TableName.valueOf("user_behavior"), scan
).map { case (key, result) =>
  val rowKey = Bytes.toString(key.get())
  val name = Bytes.toString(result.getValue(
    Bytes.toBytes("info"), Bytes.toBytes("name")))
  (rowKey, name)
}

hbaseRDD.collect().foreach(println)

// 方式2：使用Spark SQL + HBase Connector
val df = spark.read
  .format("org.apache.hadoop.hbase.spark")
  .option("hbase.columns.mapping",
    "rowKey STRING :key, name STRING info:name, age STRING info:age")
  .option("hbase.table", "user_behavior")
  .load()

df.show()

// 写入HBase
import spark.implicits._
val data = Seq(("user_100", "王五", "28"), ("user_101", "赵六", "35"))
  .toDF("rowKey", "name", "age")

data.write
  .format("org.apache.hadoop.hbase.spark")
  .option("hbase.columns.mapping",
    "rowKey STRING :key, name STRING info:name, age STRING info:age")
  .option("hbase.table", "user_behavior")
  .save()
```

### 5.3 HBase + Flink

```java
// Flink写入HBase
// 使用Flink SQL创建HBase Sink表
String createHBaseSink = """
    CREATE TABLE hbase_user_behavior (
        rowkey STRING,
        info ROW<name STRING, age STRING, city STRING>,
        event ROW<type STRING, page STRING, duration STRING>,
        PRIMARY KEY (rowkey) NOT ENFORCED
    ) WITH (
        'connector' = 'hbase-2.2',
        'table-name' = 'user_behavior',
        'zookeeper.quorum' = 'zk1:2181,zk2:2181,zk3:2181',
        'zookeeper.znode.parent' = '/hbase',
        'sink.buffer-flush.max-size' = '10mb',
        'sink.buffer-flush.max-rows' = '1000',
        'sink.buffer-flush.interval' = '2s'
    )
    """;

tableEnv.executeSql(createHBaseSink);

// 从Kafka读取并写入HBase
String insertSql = """
    INSERT INTO hbase_user_behavior
    SELECT
        CONCAT(MD5(user_id), '_', user_id, '_', CAST(event_time AS STRING)) AS rowkey,
        ROW(user_name, CAST(age AS STRING), city) AS info,
        ROW(event_type, page_url, CAST(duration AS STRING)) AS event
    FROM kafka_source
    """;

tableEnv.executeSql(insertSql);
```

## 6. 实战案例：用户行为实时存储

### 6.1 需求与架构

```
用户行为实时存储系统
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
│  App/Web │───→│  Kafka   │───→│  Flink   │───→│    HBase     │
│  埋点SDK │    │  集群    │    │  实时ETL │    │  (用户行为)  │
└──────────┘    └──────────┘    └──────────┘    └──────┬───────┘
                                                       │
                                                       ↓
                                                ┌──────────────┐
                                                │  REST API    │
                                                │  (Spring Boot)│
                                                └──────┬───────┘
                                                       │
                                                       ↓
                                                ┌──────────────┐
                                                │  前端应用    │
                                                │  用户画像/推荐│
                                                └──────────────┘

需求:
- 数据量: 日增10亿条用户行为事件
- 写入: 峰值50万TPS
- 读取: P99 < 10ms
- 存储: 保留90天
```

### 6.2 表设计

```bash
# RowKey设计: md5(user_id)[0:8] + user_id + (Long.MAX_VALUE - timestamp)
# 优势:
#   1. MD5前缀打散热点
#   2. 同一用户数据相邻（方便查用户最近行为）
#   3. 时间戳反转（最新数据排在前面，Scan时优先返回）

create 'ns_behavior:user_events',
  {NAME => 'e', VERSIONS => 1, TTL => 7776000,           # 90天
   COMPRESSION => 'LZ4', BLOOMFILTER => 'ROW',
   BLOCKSIZE => '65536',
   DATA_BLOCK_ENCODING => 'FAST_DIFF'},                   # 前缀压缩
  {NUMREGIONS => 64, SPLITALGO => 'HexStringSplit'}

# 列族 'e' (event) 包含:
#   e:type    - 事件类型 (click/view/buy/cart)
#   e:page    - 页面URL
#   e:item    - 商品ID
#   e:dur     - 停留时长
#   e:ext     - 扩展JSON
```

### 6.3 读写性能测试

**测试环境**: 5台RegionServer (32核/64GB/4*SSD)，64 Regions

| 指标 | 数值 |
|------|------|
| **写入QPS** | 52万/秒（BatchPut, 每批1000条） |
| **读取延迟P50** | 1.2ms |
| **读取延迟P99** | 6.8ms |
| **Scan 100条** | 8.5ms |
| **单条写入** | 0.5ms |
| **存储压缩比** | 5.2:1 (LZ4) |

**关键调优参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `hbase.client.write.buffer` | 8MB | 客户端写缓冲 |
| `hbase.hregion.memstore.flush.size` | 256MB | MemStore刷写阈值 |
| `hbase.regionserver.handler.count` | 200 | RPC处理线程数 |
| `hbase.hregion.majorcompaction` | 0 | 关闭自动Major Compact |
| `hfile.block.cache.size` | 0.3 | BlockCache占比（写多读少调低） |
| `hbase.regionserver.global.memstore.size` | 0.45 | MemStore占比（写多调高） |

## 7. 运维与最佳实践

### 7.1 集群规划

| 角色 | CPU | 内存 | 磁盘 | 数量 |
|------|-----|------|------|------|
| HMaster | 8核+ | 16GB+ | SAS 500GB | 2 (Active+Standby) |
| RegionServer | 16-32核 | 48-128GB | SSD 4*1TB (JBOD) | 按数据量 |
| ZooKeeper | 4核+ | 8GB+ | SSD 100GB | 3/5/7 (奇数) |

**Region数量规划**：
- 每台RegionServer: 100-200个Region（太多会导致频繁Flush）
- 单个Region大小: 5-20GB
- Region数 = 数据总量 / (Region大小 * 副本数)

### 7.2 监控指标

| 指标 | 告警阈值 | 说明 |
|------|---------|------|
| `requestsPerSecond` | > 10万 | RPC请求速率 |
| `readRequestLatency_p99` | > 50ms | 读延迟P99 |
| `writeRequestLatency_p99` | > 20ms | 写延迟P99 |
| `memStoreSize` | > 80% limit | MemStore使用率 |
| `blockCacheHitRatio` | < 80% | BlockCache命中率 |
| `compactionQueueSize` | > 10 | Compaction积压 |
| `flushQueueSize` | > 5 | Flush积压 |
| `GC pause` | > 500ms | GC暂停时间 |
| `regionCount` | > 200/RS | Region数量 |

### 7.3 常见问题

**Region热点**：
```
诊断: hbase shell > status 'detailed'  查看各RS请求分布
解决:
  1. 检查RowKey设计，是否有明显前缀聚集
  2. 加盐/哈希打散
  3. 预分区调整
  4. 手动move热点Region到空闲RS
```

**GC暂停过长**：
```
诊断: GC日志分析，关注Full GC频率和耗时
解决:
  1. 使用G1GC: -XX:+UseG1GC -XX:MaxGCPauseMillis=100
  2. 启用BucketCache（堆外），减少堆内存压力
  3. 减少堆内存到32GB以内（压缩指针优化）
  4. 调低BlockCache（写多场景）
```

**Split Storm（分裂风暴）**：
```
诊断: 短时间内大量Region分裂，导致性能抖动
解决:
  1. 建表时预分区，避免运行时分裂
  2. 调大分裂阈值: hbase.hregion.max.filesize = 20GB
  3. 禁用自动分裂，手动管理:
     alter 'table', {METHOD => 'table_att', 'SPLIT_POLICY' =>
       'org.apache.hadoop.hbase.regionserver.DisabledRegionSplitPolicy'}
```

**RIT (Regions In Transition)**：
```
诊断: HBase Master UI显示长时间处于PENDING/OPENING/CLOSING状态的Region
解决:
  1. 等待（通常几分钟内自动恢复）
  2. hbase hbck -details 检查一致性
  3. 手动assign: hbase shell > assign 'REGION_ENCODED_NAME'
  4. 重启相关RegionServer
  5. 极端情况: hbase hbck -fixAssignments
```

**最佳实践检查清单**：

```
✅ 表设计
  ├── RowKey有散列前缀（避免热点）
  ├── RowKey长度 < 100字节
  ├── 列族数量 ≤ 3
  ├── 已启用压缩（SNAPPY/LZ4）
  ├── 已设置合理的TTL
  └── 建表时已预分区

✅ 读写优化
  ├── 写入使用BatchPut（批量提交）
  ├── Scan设置合理的Caching和Batch
  ├── 已启用BloomFilter
  ├── BucketCache已配置（生产环境）
  └── 关闭自动Major Compaction（低峰手动执行）

✅ 运维
  ├── HMaster有Standby节点
  ├── 监控告警已配置
  ├── 定期做Snapshot备份
  ├── GC使用G1GC
  └── JVM堆内存 ≤ 32GB
```
