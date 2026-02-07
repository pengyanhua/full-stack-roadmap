# Kafka 命令参考

## 一、Topic 管理

### 创建 Topic

```bash
# 基本创建
kafka-topics.sh --bootstrap-server localhost:9092 \
  --create \
  --topic my-topic \
  --partitions 3 \
  --replication-factor 1

# 创建并指定配置
kafka-topics.sh --bootstrap-server localhost:9092 \
  --create \
  --topic my-topic \
  --partitions 6 \
  --replication-factor 3 \
  --config retention.ms=604800000 \
  --config max.message.bytes=1048576

# 常用配置参数：
# retention.ms          - 消息保留时间（毫秒）
# retention.bytes       - 消息保留大小（字节）
# max.message.bytes     - 单条消息最大大小
# cleanup.policy        - 清理策略（delete/compact）
# min.insync.replicas   - 最小同步副本数
# segment.bytes         - 日志段大小
```

### 查看 Topic

```bash
# 列出所有 Topic
kafka-topics.sh --bootstrap-server localhost:9092 --list

# 查看 Topic 详情
kafka-topics.sh --bootstrap-server localhost:9092 \
  --describe \
  --topic my-topic

# 输出示例：
# Topic: my-topic    TopicId: abc123    PartitionCount: 3    ReplicationFactor: 3
# Topic: my-topic    Partition: 0    Leader: 1    Replicas: 1,2,3    Isr: 1,2,3
# Topic: my-topic    Partition: 1    Leader: 2    Replicas: 2,3,1    Isr: 2,3,1
# Topic: my-topic    Partition: 2    Leader: 3    Replicas: 3,1,2    Isr: 3,1,2

# 查看所有 Topic 详情
kafka-topics.sh --bootstrap-server localhost:9092 --describe

# 查看有问题的分区（Under Replicated）
kafka-topics.sh --bootstrap-server localhost:9092 \
  --describe \
  --under-replicated-partitions

# 查看没有 Leader 的分区
kafka-topics.sh --bootstrap-server localhost:9092 \
  --describe \
  --unavailable-partitions
```

### 修改 Topic

```bash
# 增加分区数（只能增加，不能减少）
kafka-topics.sh --bootstrap-server localhost:9092 \
  --alter \
  --topic my-topic \
  --partitions 6

# 修改 Topic 配置
kafka-configs.sh --bootstrap-server localhost:9092 \
  --alter \
  --entity-type topics \
  --entity-name my-topic \
  --add-config retention.ms=86400000

# 删除 Topic 配置（恢复默认）
kafka-configs.sh --bootstrap-server localhost:9092 \
  --alter \
  --entity-type topics \
  --entity-name my-topic \
  --delete-config retention.ms

# 查看 Topic 配置
kafka-configs.sh --bootstrap-server localhost:9092 \
  --describe \
  --entity-type topics \
  --entity-name my-topic
```

### 删除 Topic

```bash
# 删除 Topic（需要 delete.topic.enable=true）
kafka-topics.sh --bootstrap-server localhost:9092 \
  --delete \
  --topic my-topic

# 批量删除（使用正则）
kafka-topics.sh --bootstrap-server localhost:9092 \
  --delete \
  --topic 'test-.*'
```

## 二、生产者命令

### 控制台生产者

```bash
# 基本发送
kafka-console-producer.sh --bootstrap-server localhost:9092 \
  --topic my-topic
# 然后输入消息，每行一条，Ctrl+C 退出

# 发送带 Key 的消息
kafka-console-producer.sh --bootstrap-server localhost:9092 \
  --topic my-topic \
  --property "parse.key=true" \
  --property "key.separator=:"
# 输入格式：key:value

# 指定分区
kafka-console-producer.sh --bootstrap-server localhost:9092 \
  --topic my-topic \
  --property "parse.key=true" \
  --property "key.separator=:" \
  --property "partition=0"

# 带确认机制
kafka-console-producer.sh --bootstrap-server localhost:9092 \
  --topic my-topic \
  --producer-property acks=all

# 从文件读取发送
kafka-console-producer.sh --bootstrap-server localhost:9092 \
  --topic my-topic < messages.txt

# 带压缩
kafka-console-producer.sh --bootstrap-server localhost:9092 \
  --topic my-topic \
  --compression-codec lz4
```

### 性能测试

```bash
# 生产者性能测试
kafka-producer-perf-test.sh \
  --topic my-topic \
  --num-records 1000000 \
  --record-size 1024 \
  --throughput -1 \
  --producer-props bootstrap.servers=localhost:9092

# 参数说明：
# --num-records   发送的消息总数
# --record-size   每条消息大小（字节）
# --throughput    吞吐量限制（-1 表示不限制）

# 带更多配置的性能测试
kafka-producer-perf-test.sh \
  --topic my-topic \
  --num-records 1000000 \
  --record-size 1024 \
  --throughput 100000 \
  --producer-props \
    bootstrap.servers=localhost:9092 \
    acks=all \
    batch.size=16384 \
    linger.ms=5 \
    compression.type=lz4
```

## 三、消费者命令

### 控制台消费者

```bash
# 从最新位置开始消费
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic my-topic

# 从最早位置开始消费
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic my-topic \
  --from-beginning

# 指定消费者组
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic my-topic \
  --group my-group

# 显示 Key 和时间戳
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic my-topic \
  --property print.key=true \
  --property print.timestamp=true \
  --property key.separator=":"

# 显示分区和 Offset
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic my-topic \
  --property print.partition=true \
  --property print.offset=true

# 指定分区和 Offset
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic my-topic \
  --partition 0 \
  --offset 100

# 只消费指定数量的消息
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic my-topic \
  --max-messages 10

# 消费多个 Topic
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --whitelist 'topic1|topic2|topic3'

# 按正则匹配 Topic
kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --whitelist 'order-.*'
```

### 消费者性能测试

```bash
# 消费者性能测试
kafka-consumer-perf-test.sh \
  --bootstrap-server localhost:9092 \
  --topic my-topic \
  --messages 1000000 \
  --threads 1

# 输出示例：
# start.time, end.time, data.consumed.in.MB, MB.sec, data.consumed.in.nMsg, nMsg.sec
# 2024-01-01 10:00:00, 2024-01-01 10:00:10, 976.5625, 97.656, 1000000, 100000

# 带消费者组的性能测试
kafka-consumer-perf-test.sh \
  --bootstrap-server localhost:9092 \
  --topic my-topic \
  --messages 1000000 \
  --group perf-test-group \
  --from-latest
```

## 四、消费者组管理

### 查看消费者组

```bash
# 列出所有消费者组
kafka-consumer-groups.sh --bootstrap-server localhost:9092 --list

# 查看消费者组详情
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --describe \
  --group my-group

# 输出示例：
# GROUP     TOPIC     PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG     CONSUMER-ID                                     HOST            CLIENT-ID
# my-group  my-topic  0          1000            1050            50      consumer-1-abc-123                              /192.168.1.100  consumer-1
# my-group  my-topic  1          2000            2010            10      consumer-1-abc-123                              /192.168.1.100  consumer-1
# my-group  my-topic  2          1500            1500            0       consumer-2-def-456                              /192.168.1.101  consumer-2

# 查看所有消费者组详情
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --describe \
  --all-groups

# 查看消费者组成员
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --describe \
  --group my-group \
  --members

# 查看消费者组状态
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --describe \
  --group my-group \
  --state

# 输出：GROUP    COORDINATOR (ID)    ASSIGNMENT-STRATEGY    STATE           MEMBERS
#        my-group broker1:9092 (1)   range                  Stable          2
```

### 重置消费位点

```bash
# 重置到最早位置
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group my-group \
  --topic my-topic \
  --reset-offsets \
  --to-earliest \
  --execute

# 重置到最新位置
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group my-group \
  --topic my-topic \
  --reset-offsets \
  --to-latest \
  --execute

# 重置到指定 Offset
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group my-group \
  --topic my-topic:0 \
  --reset-offsets \
  --to-offset 1000 \
  --execute

# 重置到指定时间
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group my-group \
  --topic my-topic \
  --reset-offsets \
  --to-datetime 2024-01-01T00:00:00.000 \
  --execute

# 向前/向后移动指定数量
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group my-group \
  --topic my-topic \
  --reset-offsets \
  --shift-by -100 \
  --execute

# 重置所有 Topic 的 Offset
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group my-group \
  --all-topics \
  --reset-offsets \
  --to-earliest \
  --execute

# 先预览再执行（不加 --execute）
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group my-group \
  --topic my-topic \
  --reset-offsets \
  --to-earliest \
  --dry-run
```

### 删除消费者组

```bash
# 删除消费者组（组内不能有活跃消费者）
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --delete \
  --group my-group
```

## 五、分区管理

### 查看分区

```bash
# 查看 Topic 分区详情
kafka-topics.sh --bootstrap-server localhost:9092 \
  --describe \
  --topic my-topic

# 查看分区的 Log 信息
kafka-log-dirs.sh --bootstrap-server localhost:9092 \
  --describe \
  --topic-list my-topic

# 查看 Broker 上的日志目录使用情况
kafka-log-dirs.sh --bootstrap-server localhost:9092 \
  --describe \
  --broker-list 0,1,2
```

### 分区重分配

```bash
# 1. 生成重分配计划
cat > topics-to-move.json << 'EOF'
{
  "topics": [
    {"topic": "my-topic"}
  ],
  "version": 1
}
EOF

kafka-reassign-partitions.sh --bootstrap-server localhost:9092 \
  --topics-to-move-json-file topics-to-move.json \
  --broker-list "0,1,2" \
  --generate

# 2. 执行重分配（使用上一步生成的计划）
cat > reassignment.json << 'EOF'
{
  "version": 1,
  "partitions": [
    {"topic": "my-topic", "partition": 0, "replicas": [1, 2]},
    {"topic": "my-topic", "partition": 1, "replicas": [2, 0]},
    {"topic": "my-topic", "partition": 2, "replicas": [0, 1]}
  ]
}
EOF

kafka-reassign-partitions.sh --bootstrap-server localhost:9092 \
  --reassignment-json-file reassignment.json \
  --execute

# 3. 验证重分配进度
kafka-reassign-partitions.sh --bootstrap-server localhost:9092 \
  --reassignment-json-file reassignment.json \
  --verify

# 限制副本同步带宽（避免影响正常流量）
kafka-reassign-partitions.sh --bootstrap-server localhost:9092 \
  --reassignment-json-file reassignment.json \
  --execute \
  --throttle 50000000  # 50MB/s
```

### Preferred Leader 选举

```bash
# 对所有分区执行 Preferred Leader 选举
kafka-leader-election.sh --bootstrap-server localhost:9092 \
  --election-type preferred \
  --all-topic-partitions

# 对指定 Topic 执行
kafka-leader-election.sh --bootstrap-server localhost:9092 \
  --election-type preferred \
  --topic my-topic

# 通过 JSON 文件指定分区
cat > election.json << 'EOF'
{
  "partitions": [
    {"topic": "my-topic", "partition": 0},
    {"topic": "my-topic", "partition": 1}
  ]
}
EOF

kafka-leader-election.sh --bootstrap-server localhost:9092 \
  --election-type preferred \
  --path-to-json-file election.json
```

## 六、集群管理

### 查看集群状态

```bash
# 查看集群元数据
kafka-metadata.sh --snapshot /var/kafka-logs/__cluster_metadata-0/00000000000000000000.log \
  --command "cat"

# 查看 Broker 配置
kafka-configs.sh --bootstrap-server localhost:9092 \
  --describe \
  --entity-type brokers \
  --entity-name 0

# 查看所有 Broker 配置
kafka-configs.sh --bootstrap-server localhost:9092 \
  --describe \
  --entity-type brokers \
  --all

# 动态修改 Broker 配置
kafka-configs.sh --bootstrap-server localhost:9092 \
  --alter \
  --entity-type brokers \
  --entity-name 0 \
  --add-config log.cleaner.threads=2

# 修改集群范围的配置
kafka-configs.sh --bootstrap-server localhost:9092 \
  --alter \
  --entity-type brokers \
  --entity-default \
  --add-config log.retention.hours=72
```

### 副本验证

```bash
# 验证副本一致性
kafka-replica-verification.sh \
  --broker-list localhost:9092 \
  --topic-white-list 'my-topic'
```

## 七、ACL 权限管理

```bash
# 添加权限：允许用户 alice 对 my-topic 进行读写
kafka-acls.sh --bootstrap-server localhost:9092 \
  --add \
  --allow-principal User:alice \
  --operation Read \
  --operation Write \
  --topic my-topic

# 添加权限：允许消费者组访问
kafka-acls.sh --bootstrap-server localhost:9092 \
  --add \
  --allow-principal User:alice \
  --operation Read \
  --group my-group

# 查看权限
kafka-acls.sh --bootstrap-server localhost:9092 \
  --list \
  --topic my-topic

# 删除权限
kafka-acls.sh --bootstrap-server localhost:9092 \
  --remove \
  --allow-principal User:alice \
  --operation Read \
  --topic my-topic

# 使用通配符
kafka-acls.sh --bootstrap-server localhost:9092 \
  --add \
  --allow-principal User:alice \
  --operation All \
  --topic '*' \
  --resource-pattern-type prefixed \
  --topic 'order-'  # 匹配所有以 order- 开头的 Topic
```

## 八、事务管理

```bash
# 查看事务状态
kafka-transactions.sh --bootstrap-server localhost:9092 \
  describe \
  --transactional-id my-transactional-id

# 列出所有事务
kafka-transactions.sh --bootstrap-server localhost:9092 \
  list

# 中止事务
kafka-transactions.sh --bootstrap-server localhost:9092 \
  abort \
  --transactional-id my-transactional-id
```

## 九、消息查看与删除

### 查看消息

```bash
# 获取 Topic 的 Offset 范围
kafka-run-class.sh kafka.tools.GetOffsetShell \
  --broker-list localhost:9092 \
  --topic my-topic

# 输出示例：
# my-topic:0:1000    (分区:最新Offset)
# my-topic:1:2000
# my-topic:2:1500

# 按时间查找 Offset
kafka-run-class.sh kafka.tools.GetOffsetShell \
  --broker-list localhost:9092 \
  --topic my-topic \
  --time -1  # -1 最新，-2 最早

# 使用 kcat（原 kafkacat）查看消息
# 安装：apt install kafkacat 或 brew install kcat
kcat -b localhost:9092 -t my-topic -C -f 'Topic %t [%p] at offset %o: key=%k value=%s\n'
```

### 删除消息

```bash
# 删除消息（按 Offset）
cat > delete-records.json << 'EOF'
{
  "partitions": [
    {"topic": "my-topic", "partition": 0, "offset": 100},
    {"topic": "my-topic", "partition": 1, "offset": 200}
  ],
  "version": 1
}
EOF

kafka-delete-records.sh --bootstrap-server localhost:9092 \
  --offset-json-file delete-records.json

# 注意：这会删除指定 Offset 之前的所有消息
```

## 十、常用脚本组合

### 监控脚本

```bash
#!/bin/bash
# monitor-kafka.sh - Kafka 监控脚本

BOOTSTRAP_SERVER="localhost:9092"

echo "=== Broker 状态 ==="
kafka-broker-api-versions.sh --bootstrap-server $BOOTSTRAP_SERVER | head -5

echo -e "\n=== Topic 列表 ==="
kafka-topics.sh --bootstrap-server $BOOTSTRAP_SERVER --list

echo -e "\n=== Under Replicated 分区 ==="
kafka-topics.sh --bootstrap-server $BOOTSTRAP_SERVER \
  --describe --under-replicated-partitions

echo -e "\n=== 消费者组状态 ==="
kafka-consumer-groups.sh --bootstrap-server $BOOTSTRAP_SERVER --list | while read group; do
  echo "--- Group: $group ---"
  kafka-consumer-groups.sh --bootstrap-server $BOOTSTRAP_SERVER \
    --describe --group $group 2>/dev/null | head -5
done
```

### 数据迁移脚本

```bash
#!/bin/bash
# migrate-topic.sh - Topic 数据迁移

SOURCE_BOOTSTRAP="source-kafka:9092"
TARGET_BOOTSTRAP="target-kafka:9092"
TOPIC="my-topic"

# 使用 MirrorMaker 2.0 进行迁移
kafka-mirror-maker.sh \
  --consumer.config source-consumer.properties \
  --producer.config target-producer.properties \
  --whitelist "$TOPIC" \
  --num.streams 4
```

### 清理脚本

```bash
#!/bin/bash
# cleanup-topics.sh - 清理测试 Topic

BOOTSTRAP_SERVER="localhost:9092"
PREFIX="test-"

kafka-topics.sh --bootstrap-server $BOOTSTRAP_SERVER --list | grep "^${PREFIX}" | while read topic; do
  echo "Deleting topic: $topic"
  kafka-topics.sh --bootstrap-server $BOOTSTRAP_SERVER --delete --topic "$topic"
done
```
