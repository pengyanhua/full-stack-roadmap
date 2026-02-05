"""
============================================================
                Kafka 实战示例（Python）
============================================================
使用 kafka-python 和 confluent-kafka 库实现常见的 Kafka 应用场景

安装依赖：
pip install kafka-python confluent-kafka

启动 Kafka（Docker）：
docker-compose up -d

============================================================
"""

import json
import time
import uuid
import threading
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod

# kafka-python 库
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import KafkaError, TopicAlreadyExistsError

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
#                    配置管理
# ============================================================

@dataclass
class KafkaConfig:
    """
    Kafka 配置类

    集中管理 Kafka 连接配置，便于在不同环境间切换
    """
    # 基础配置
    bootstrap_servers: str = 'localhost:9092'

    # 生产者配置
    producer_acks: str = 'all'              # 确认机制：0, 1, all
    producer_retries: int = 3               # 重试次数
    producer_batch_size: int = 16384        # 批次大小（字节）
    producer_linger_ms: int = 5             # 等待时间（毫秒）
    producer_compression: str = 'gzip'      # 压缩算法：gzip, snappy, lz4, zstd

    # 消费者配置
    consumer_group_id: str = 'default-group'
    consumer_auto_offset_reset: str = 'earliest'  # earliest, latest
    consumer_enable_auto_commit: bool = False     # 是否自动提交
    consumer_max_poll_records: int = 500          # 单次拉取最大消息数
    consumer_session_timeout_ms: int = 30000      # 会话超时
    consumer_heartbeat_interval_ms: int = 10000   # 心跳间隔


# 默认配置实例
default_config = KafkaConfig()


# ============================================================
#                    一、生产者封装
# ============================================================

class Producer:
    """
    Kafka 生产者封装

    功能：
    1. 同步/异步发送消息
    2. 支持消息序列化（JSON）
    3. 支持自定义分区策略
    4. 支持发送回调
    5. 批量发送优化

    使用示例：
        producer = Producer()
        producer.send('my-topic', {'name': 'Alice', 'age': 25})
        producer.close()
    """

    def __init__(self, config: KafkaConfig = None):
        """
        初始化生产者

        Args:
            config: Kafka 配置，默认使用 default_config
        """
        self.config = config or default_config

        # 创建 KafkaProducer 实例
        # key_serializer: 将 Key 序列化为字节
        # value_serializer: 将 Value 序列化为 JSON 字节
        self.producer = KafkaProducer(
            bootstrap_servers=self.config.bootstrap_servers,

            # 序列化配置
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'),

            # 可靠性配置
            acks=self.config.producer_acks,           # 等待所有副本确认
            retries=self.config.producer_retries,     # 失败重试

            # 性能配置
            batch_size=self.config.producer_batch_size,   # 批量发送大小
            linger_ms=self.config.producer_linger_ms,     # 等待更多消息以便批量发送
            compression_type=self.config.producer_compression,  # 压缩

            # 幂等性（防止重复消息）
            enable_idempotence=True,
        )

        logger.info(f"Producer initialized, bootstrap_servers={self.config.bootstrap_servers}")

    def send(
        self,
        topic: str,
        value: Any,
        key: str = None,
        partition: int = None,
        headers: List[tuple] = None,
        callback: Callable = None
    ) -> None:
        """
        发送消息（异步）

        Args:
            topic: 目标 Topic
            value: 消息内容（会被 JSON 序列化）
            key: 消息 Key（相同 Key 的消息会被发送到同一分区）
            partition: 指定分区（可选，不指定则根据 Key 哈希或轮询）
            headers: 消息头（元数据）
            callback: 发送完成回调函数

        示例：
            # 简单发送
            producer.send('orders', {'order_id': '123', 'amount': 100})

            # 带 Key 发送（保证相同用户的消息有序）
            producer.send('orders', {'order_id': '123'}, key='user_001')

            # 带回调发送
            def on_success(record_metadata):
                print(f"Sent to {record_metadata.topic}:{record_metadata.partition}")
            producer.send('orders', {'order_id': '123'}, callback=on_success)
        """
        try:
            # 构建发送参数
            kwargs = {
                'topic': topic,
                'value': value,
            }

            if key:
                kwargs['key'] = key
            if partition is not None:
                kwargs['partition'] = partition
            if headers:
                kwargs['headers'] = headers

            # 发送消息
            future = self.producer.send(**kwargs)

            # 添加回调
            if callback:
                future.add_callback(callback)

            # 添加错误回调
            future.add_errback(lambda e: logger.error(f"Send failed: {e}"))

        except KafkaError as e:
            logger.error(f"Failed to send message to {topic}: {e}")
            raise

    def send_sync(
        self,
        topic: str,
        value: Any,
        key: str = None,
        timeout: float = 10.0
    ) -> Dict:
        """
        同步发送消息（等待确认）

        Args:
            topic: 目标 Topic
            value: 消息内容
            key: 消息 Key
            timeout: 超时时间（秒）

        Returns:
            包含发送结果的字典：
            {
                'topic': 'my-topic',
                'partition': 0,
                'offset': 123,
                'timestamp': 1704067200000
            }

        示例：
            result = producer.send_sync('orders', {'order_id': '123'})
            print(f"Message sent to partition {result['partition']}, offset {result['offset']}")
        """
        try:
            kwargs = {'topic': topic, 'value': value}
            if key:
                kwargs['key'] = key

            # 发送并等待结果
            future = self.producer.send(**kwargs)
            record_metadata = future.get(timeout=timeout)

            return {
                'topic': record_metadata.topic,
                'partition': record_metadata.partition,
                'offset': record_metadata.offset,
                'timestamp': record_metadata.timestamp,
            }

        except KafkaError as e:
            logger.error(f"Sync send failed: {e}")
            raise

    def send_batch(self, topic: str, messages: List[Dict], key_field: str = None) -> int:
        """
        批量发送消息

        Args:
            topic: 目标 Topic
            messages: 消息列表
            key_field: 从消息中提取 Key 的字段名

        Returns:
            成功发送的消息数

        示例：
            orders = [
                {'order_id': '001', 'user_id': 'u1', 'amount': 100},
                {'order_id': '002', 'user_id': 'u2', 'amount': 200},
            ]
            count = producer.send_batch('orders', orders, key_field='user_id')
        """
        count = 0
        for msg in messages:
            key = str(msg.get(key_field)) if key_field and key_field in msg else None
            self.send(topic, msg, key=key)
            count += 1

        # 确保所有消息都被发送
        self.flush()
        return count

    def flush(self, timeout: float = None):
        """
        刷新缓冲区，确保所有消息都被发送

        Args:
            timeout: 超时时间（秒）
        """
        self.producer.flush(timeout=timeout)

    def close(self):
        """关闭生产者"""
        self.producer.close()
        logger.info("Producer closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================
#                    二、消费者封装
# ============================================================

class Consumer:
    """
    Kafka 消费者封装

    功能：
    1. 支持手动/自动提交 Offset
    2. 支持消息反序列化（JSON）
    3. 支持优雅停止
    4. 支持消费回调

    使用示例：
        consumer = Consumer(['my-topic'], group_id='my-group')
        for msg in consumer.consume():
            print(msg)
            consumer.commit()
        consumer.close()
    """

    def __init__(
        self,
        topics: List[str],
        group_id: str = None,
        config: KafkaConfig = None
    ):
        """
        初始化消费者

        Args:
            topics: 订阅的 Topic 列表
            group_id: 消费者组 ID
            config: Kafka 配置
        """
        self.config = config or default_config
        self.topics = topics
        self.group_id = group_id or self.config.consumer_group_id
        self._running = False

        # 创建 KafkaConsumer 实例
        self.consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=self.config.bootstrap_servers,
            group_id=self.group_id,

            # 反序列化配置
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),

            # Offset 配置
            auto_offset_reset=self.config.consumer_auto_offset_reset,
            enable_auto_commit=self.config.consumer_enable_auto_commit,

            # 性能配置
            max_poll_records=self.config.consumer_max_poll_records,
            session_timeout_ms=self.config.consumer_session_timeout_ms,
            heartbeat_interval_ms=self.config.consumer_heartbeat_interval_ms,
        )

        logger.info(f"Consumer initialized, topics={topics}, group_id={self.group_id}")

    def consume(self, timeout_ms: int = 1000) -> Any:
        """
        消费消息（生成器）

        Args:
            timeout_ms: poll 超时时间（毫秒）

        Yields:
            消息字典：
            {
                'topic': 'my-topic',
                'partition': 0,
                'offset': 123,
                'key': 'user_001',
                'value': {'order_id': '123'},
                'timestamp': 1704067200000,
                'headers': []
            }

        示例：
            for msg in consumer.consume():
                print(f"Received: {msg['value']}")
                consumer.commit()
        """
        self._running = True

        while self._running:
            # poll 获取消息
            records = self.consumer.poll(timeout_ms=timeout_ms)

            for topic_partition, messages in records.items():
                for msg in messages:
                    yield {
                        'topic': msg.topic,
                        'partition': msg.partition,
                        'offset': msg.offset,
                        'key': msg.key,
                        'value': msg.value,
                        'timestamp': msg.timestamp,
                        'headers': msg.headers,
                    }

    def consume_with_handler(
        self,
        handler: Callable[[Dict], bool],
        error_handler: Callable[[Exception, Dict], None] = None,
        batch_size: int = 1
    ):
        """
        使用处理器消费消息

        Args:
            handler: 消息处理函数，返回 True 表示处理成功
            error_handler: 错误处理函数
            batch_size: 批量提交大小

        示例：
            def handle_message(msg):
                print(f"Processing: {msg['value']}")
                return True

            def handle_error(e, msg):
                print(f"Error: {e}, message: {msg}")

            consumer.consume_with_handler(handle_message, handle_error)
        """
        processed = 0

        for msg in self.consume():
            try:
                # 调用处理器
                success = handler(msg)

                if success:
                    processed += 1

                    # 批量提交
                    if processed >= batch_size:
                        self.commit()
                        processed = 0

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                if error_handler:
                    error_handler(e, msg)

    def commit(self, async_commit: bool = False):
        """
        提交 Offset

        Args:
            async_commit: 是否异步提交

        说明：
            - 同步提交（async_commit=False）：阻塞等待提交完成，更可靠
            - 异步提交（async_commit=True）：不等待，性能更好但可能丢失
        """
        if async_commit:
            self.consumer.commit_async()
        else:
            self.consumer.commit()

    def seek_to_beginning(self, partitions: List = None):
        """
        重置到分区起始位置

        Args:
            partitions: 分区列表，默认所有分区
        """
        if partitions is None:
            partitions = self.consumer.assignment()
        self.consumer.seek_to_beginning(*partitions)

    def seek_to_end(self, partitions: List = None):
        """
        重置到分区末尾位置

        Args:
            partitions: 分区列表，默认所有分区
        """
        if partitions is None:
            partitions = self.consumer.assignment()
        self.consumer.seek_to_end(*partitions)

    def stop(self):
        """停止消费"""
        self._running = False
        logger.info("Consumer stopping...")

    def close(self):
        """关闭消费者"""
        self._running = False
        self.consumer.close()
        logger.info("Consumer closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================
#                    三、Topic 管理
# ============================================================

class TopicManager:
    """
    Topic 管理器

    功能：
    1. 创建/删除 Topic
    2. 查看 Topic 列表和详情
    3. 修改 Topic 配置

    使用示例：
        manager = TopicManager()
        manager.create_topic('my-topic', partitions=3, replication_factor=1)
        print(manager.list_topics())
    """

    def __init__(self, config: KafkaConfig = None):
        """初始化 Topic 管理器"""
        self.config = config or default_config

        self.admin = KafkaAdminClient(
            bootstrap_servers=self.config.bootstrap_servers,
        )

        logger.info(f"TopicManager initialized")

    def create_topic(
        self,
        name: str,
        partitions: int = 3,
        replication_factor: int = 1,
        configs: Dict[str, str] = None
    ) -> bool:
        """
        创建 Topic

        Args:
            name: Topic 名称
            partitions: 分区数
            replication_factor: 副本因子
            configs: 配置参数

        Returns:
            是否创建成功

        示例：
            # 创建基本 Topic
            manager.create_topic('orders', partitions=6)

            # 创建带配置的 Topic
            manager.create_topic(
                'logs',
                partitions=3,
                configs={
                    'retention.ms': '86400000',      # 保留 1 天
                    'cleanup.policy': 'delete',      # 删除策略
                }
            )
        """
        try:
            topic = NewTopic(
                name=name,
                num_partitions=partitions,
                replication_factor=replication_factor,
                topic_configs=configs,
            )

            self.admin.create_topics([topic])
            logger.info(f"Topic '{name}' created successfully")
            return True

        except TopicAlreadyExistsError:
            logger.warning(f"Topic '{name}' already exists")
            return False
        except Exception as e:
            logger.error(f"Failed to create topic '{name}': {e}")
            raise

    def delete_topic(self, name: str) -> bool:
        """
        删除 Topic

        Args:
            name: Topic 名称

        Returns:
            是否删除成功
        """
        try:
            self.admin.delete_topics([name])
            logger.info(f"Topic '{name}' deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to delete topic '{name}': {e}")
            return False

    def list_topics(self) -> List[str]:
        """
        列出所有 Topic

        Returns:
            Topic 名称列表
        """
        return list(self.admin.list_topics())

    def describe_topic(self, name: str) -> Dict:
        """
        获取 Topic 详情

        Args:
            name: Topic 名称

        Returns:
            Topic 详情字典
        """
        topics = self.admin.describe_topics([name])
        if topics:
            topic = topics[0]
            return {
                'name': topic['topic'],
                'partitions': [
                    {
                        'partition': p['partition'],
                        'leader': p['leader'],
                        'replicas': p['replicas'],
                        'isr': p['isr'],
                    }
                    for p in topic['partitions']
                ],
            }
        return None

    def close(self):
        """关闭管理器"""
        self.admin.close()


# ============================================================
#                    四、消息处理模式
# ============================================================

class MessageHandler(ABC):
    """
    消息处理器基类

    定义消息处理的标准接口，便于实现不同的处理策略
    """

    @abstractmethod
    def handle(self, message: Dict) -> bool:
        """
        处理消息

        Args:
            message: 消息字典

        Returns:
            是否处理成功
        """
        pass

    def on_error(self, error: Exception, message: Dict):
        """
        错误处理

        Args:
            error: 异常对象
            message: 导致错误的消息
        """
        logger.error(f"Error handling message: {error}, message: {message}")


class OrderHandler(MessageHandler):
    """
    订单处理器示例

    处理订单相关消息，包括：
    - 订单创建
    - 订单支付
    - 订单取消
    """

    def handle(self, message: Dict) -> bool:
        """处理订单消息"""
        value = message['value']
        event_type = value.get('event_type')

        if event_type == 'order_created':
            return self._handle_order_created(value)
        elif event_type == 'order_paid':
            return self._handle_order_paid(value)
        elif event_type == 'order_cancelled':
            return self._handle_order_cancelled(value)
        else:
            logger.warning(f"Unknown event type: {event_type}")
            return True  # 跳过未知事件

    def _handle_order_created(self, data: Dict) -> bool:
        """处理订单创建事件"""
        order_id = data.get('order_id')
        user_id = data.get('user_id')
        amount = data.get('amount')

        logger.info(f"Processing order created: order_id={order_id}, user_id={user_id}, amount={amount}")

        # 实际业务逻辑：保存订单、发送通知等
        # ...

        return True

    def _handle_order_paid(self, data: Dict) -> bool:
        """处理订单支付事件"""
        order_id = data.get('order_id')
        payment_id = data.get('payment_id')

        logger.info(f"Processing order paid: order_id={order_id}, payment_id={payment_id}")

        # 实际业务逻辑：更新订单状态、触发发货等
        # ...

        return True

    def _handle_order_cancelled(self, data: Dict) -> bool:
        """处理订单取消事件"""
        order_id = data.get('order_id')
        reason = data.get('reason')

        logger.info(f"Processing order cancelled: order_id={order_id}, reason={reason}")

        # 实际业务逻辑：退款、恢复库存等
        # ...

        return True


# ============================================================
#                    五、Worker 消费者模式
# ============================================================

class ConsumerWorker:
    """
    消费者 Worker

    在独立线程中运行消费者，支持优雅停止

    使用示例：
        handler = OrderHandler()
        worker = ConsumerWorker(
            topics=['orders'],
            group_id='order-processor',
            handler=handler
        )
        worker.start()

        # ... 运行一段时间后 ...

        worker.stop()
    """

    def __init__(
        self,
        topics: List[str],
        group_id: str,
        handler: MessageHandler,
        config: KafkaConfig = None
    ):
        """
        初始化 Worker

        Args:
            topics: 订阅的 Topic 列表
            group_id: 消费者组 ID
            handler: 消息处理器
            config: Kafka 配置
        """
        self.topics = topics
        self.group_id = group_id
        self.handler = handler
        self.config = config or default_config

        self._consumer = None
        self._thread = None
        self._running = False

    def start(self):
        """启动 Worker"""
        if self._running:
            logger.warning("Worker is already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

        logger.info(f"ConsumerWorker started, topics={self.topics}, group_id={self.group_id}")

    def _run(self):
        """Worker 主循环"""
        self._consumer = Consumer(
            self.topics,
            group_id=self.group_id,
            config=self.config
        )

        try:
            for message in self._consumer.consume():
                if not self._running:
                    break

                try:
                    success = self.handler.handle(message)
                    if success:
                        self._consumer.commit()
                except Exception as e:
                    self.handler.on_error(e, message)

        finally:
            self._consumer.close()

    def stop(self, timeout: float = 10.0):
        """
        停止 Worker

        Args:
            timeout: 等待停止的超时时间（秒）
        """
        if not self._running:
            return

        logger.info("Stopping ConsumerWorker...")
        self._running = False

        if self._consumer:
            self._consumer.stop()

        if self._thread:
            self._thread.join(timeout=timeout)

        logger.info("ConsumerWorker stopped")

    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._running


# ============================================================
#                    六、生产者-消费者模式
# ============================================================

class ProducerConsumerPipeline:
    """
    生产者-消费者管道

    实现 Consume-Transform-Produce 模式：
    1. 从源 Topic 消费消息
    2. 转换消息
    3. 发送到目标 Topic

    使用示例：
        def transform(msg):
            data = msg['value']
            data['processed_at'] = datetime.now().isoformat()
            return data

        pipeline = ProducerConsumerPipeline(
            source_topic='raw-events',
            target_topic='processed-events',
            group_id='event-processor',
            transform_func=transform
        )
        pipeline.start()
    """

    def __init__(
        self,
        source_topic: str,
        target_topic: str,
        group_id: str,
        transform_func: Callable[[Dict], Any],
        config: KafkaConfig = None
    ):
        """
        初始化管道

        Args:
            source_topic: 源 Topic
            target_topic: 目标 Topic
            group_id: 消费者组 ID
            transform_func: 转换函数
            config: Kafka 配置
        """
        self.source_topic = source_topic
        self.target_topic = target_topic
        self.group_id = group_id
        self.transform_func = transform_func
        self.config = config or default_config

        self._running = False
        self._thread = None

    def start(self):
        """启动管道"""
        self._running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

        logger.info(f"Pipeline started: {self.source_topic} -> {self.target_topic}")

    def _run(self):
        """管道主循环"""
        consumer = Consumer(
            [self.source_topic],
            group_id=self.group_id,
            config=self.config
        )
        producer = Producer(config=self.config)

        try:
            for message in consumer.consume():
                if not self._running:
                    break

                try:
                    # 转换消息
                    transformed = self.transform_func(message)

                    if transformed is not None:
                        # 发送到目标 Topic
                        # 保留原始 Key 以维持分区亲和性
                        producer.send(
                            self.target_topic,
                            transformed,
                            key=message.get('key')
                        )

                    # 提交 Offset
                    consumer.commit()

                except Exception as e:
                    logger.error(f"Pipeline error: {e}")

        finally:
            consumer.close()
            producer.close()

    def stop(self, timeout: float = 10.0):
        """停止管道"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("Pipeline stopped")


# ============================================================
#                    七、死信队列（DLQ）
# ============================================================

class DeadLetterQueue:
    """
    死信队列处理器

    处理失败的消息会被发送到死信队列，便于后续排查和重试

    使用示例：
        dlq = DeadLetterQueue(main_topic='orders', dlq_topic='orders-dlq')

        def process_order(msg):
            # 可能失败的处理逻辑
            pass

        dlq.consume_with_dlq(process_order, max_retries=3)
    """

    def __init__(
        self,
        main_topic: str,
        dlq_topic: str,
        group_id: str = None,
        config: KafkaConfig = None
    ):
        """
        初始化死信队列处理器

        Args:
            main_topic: 主 Topic
            dlq_topic: 死信 Topic
            group_id: 消费者组 ID
            config: Kafka 配置
        """
        self.main_topic = main_topic
        self.dlq_topic = dlq_topic
        self.group_id = group_id or f"{main_topic}-consumer"
        self.config = config or default_config

        self._running = False

    def consume_with_dlq(
        self,
        handler: Callable[[Dict], bool],
        max_retries: int = 3
    ):
        """
        带死信队列的消费

        Args:
            handler: 消息处理函数
            max_retries: 最大重试次数
        """
        consumer = Consumer(
            [self.main_topic],
            group_id=self.group_id,
            config=self.config
        )
        producer = Producer(config=self.config)

        self._running = True

        try:
            for message in consumer.consume():
                if not self._running:
                    break

                # 获取重试次数
                retry_count = self._get_retry_count(message)

                try:
                    success = handler(message)

                    if success:
                        consumer.commit()
                    else:
                        # 处理失败，发送到 DLQ 或重试
                        self._handle_failure(producer, message, retry_count, max_retries)
                        consumer.commit()

                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    self._handle_failure(producer, message, retry_count, max_retries, error=str(e))
                    consumer.commit()

        finally:
            consumer.close()
            producer.close()

    def _get_retry_count(self, message: Dict) -> int:
        """从消息头获取重试次数"""
        headers = message.get('headers', [])
        for key, value in headers:
            if key == 'retry_count':
                return int(value.decode('utf-8'))
        return 0

    def _handle_failure(
        self,
        producer: Producer,
        message: Dict,
        retry_count: int,
        max_retries: int,
        error: str = None
    ):
        """
        处理失败消息

        如果未超过最大重试次数，重新发送到主 Topic；
        否则发送到死信队列
        """
        if retry_count < max_retries:
            # 重试：发送回主 Topic
            logger.warning(f"Retrying message, attempt {retry_count + 1}/{max_retries}")

            headers = [
                ('retry_count', str(retry_count + 1).encode('utf-8')),
                ('original_topic', self.main_topic.encode('utf-8')),
            ]

            producer.send(
                self.main_topic,
                message['value'],
                key=message.get('key'),
                headers=headers
            )
        else:
            # 超过重试次数，发送到 DLQ
            logger.error(f"Max retries exceeded, sending to DLQ: {self.dlq_topic}")

            dlq_message = {
                'original_message': message['value'],
                'original_topic': self.main_topic,
                'original_partition': message['partition'],
                'original_offset': message['offset'],
                'error': error,
                'retry_count': retry_count,
                'failed_at': datetime.now().isoformat(),
            }

            producer.send(self.dlq_topic, dlq_message, key=message.get('key'))

    def stop(self):
        """停止处理"""
        self._running = False


# ============================================================
#                    八、事件发布者
# ============================================================

@dataclass
class Event:
    """
    事件基类

    用于定义业务事件的结构
    """
    event_type: str
    event_id: str = None
    timestamp: str = None

    def __post_init__(self):
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


@dataclass
class OrderCreatedEvent(Event):
    """订单创建事件"""
    event_type: str = 'order_created'
    order_id: str = None
    user_id: str = None
    items: List[Dict] = None
    total_amount: float = 0


@dataclass
class UserRegisteredEvent(Event):
    """用户注册事件"""
    event_type: str = 'user_registered'
    user_id: str = None
    email: str = None
    registered_at: str = None


class EventPublisher:
    """
    事件发布者

    用于发布领域事件到 Kafka

    使用示例：
        publisher = EventPublisher(topic='domain-events')

        event = OrderCreatedEvent(
            order_id='ORD-001',
            user_id='USR-001',
            items=[{'product_id': 'P001', 'quantity': 2}],
            total_amount=199.99
        )

        publisher.publish(event, key=event.order_id)
    """

    def __init__(self, topic: str, config: KafkaConfig = None):
        """
        初始化事件发布者

        Args:
            topic: 事件 Topic
            config: Kafka 配置
        """
        self.topic = topic
        self.producer = Producer(config=config)

    def publish(self, event: Event, key: str = None) -> None:
        """
        发布事件

        Args:
            event: 事件对象
            key: 消息 Key（用于分区）
        """
        self.producer.send(
            self.topic,
            event.to_dict(),
            key=key or event.event_id
        )
        logger.info(f"Published event: {event.event_type}, id={event.event_id}")

    def publish_sync(self, event: Event, key: str = None) -> Dict:
        """
        同步发布事件

        Args:
            event: 事件对象
            key: 消息 Key

        Returns:
            发送结果
        """
        result = self.producer.send_sync(
            self.topic,
            event.to_dict(),
            key=key or event.event_id
        )
        logger.info(f"Published event: {event.event_type}, id={event.event_id}, offset={result['offset']}")
        return result

    def close(self):
        """关闭发布者"""
        self.producer.close()


# ============================================================
#                    九、分区器
# ============================================================

class CustomPartitioner:
    """
    自定义分区器

    根据业务规则决定消息发送到哪个分区

    使用示例：
        partitioner = CustomPartitioner(num_partitions=6)
        partition = partitioner.partition_by_user_id('user_123')
    """

    def __init__(self, num_partitions: int):
        """
        初始化分区器

        Args:
            num_partitions: 分区数
        """
        self.num_partitions = num_partitions

    def partition_by_key(self, key: str) -> int:
        """
        根据 Key 哈希分区

        相同 Key 的消息会被发送到同一分区，保证顺序性

        Args:
            key: 分区 Key

        Returns:
            分区编号
        """
        if key is None:
            return 0
        return hash(key) % self.num_partitions

    def partition_by_user_id(self, user_id: str) -> int:
        """
        根据用户 ID 分区

        同一用户的所有消息都会发送到同一分区

        Args:
            user_id: 用户 ID

        Returns:
            分区编号
        """
        return self.partition_by_key(user_id)

    def partition_by_region(self, region: str) -> int:
        """
        根据地区分区

        不同地区的消息发送到不同分区，便于按地区处理

        Args:
            region: 地区代码

        Returns:
            分区编号
        """
        region_mapping = {
            'cn-north': 0,
            'cn-south': 1,
            'cn-east': 2,
            'ap-southeast': 3,
            'eu-west': 4,
            'us-east': 5,
        }
        return region_mapping.get(region, 0)


# ============================================================
#                    十、消费进度监控
# ============================================================

class ConsumerLagMonitor:
    """
    消费延迟监控

    监控消费者组的消费进度和延迟

    使用示例：
        monitor = ConsumerLagMonitor()
        lag = monitor.get_consumer_lag('my-group', 'my-topic')
        print(f"Total lag: {lag['total_lag']}")
    """

    def __init__(self, config: KafkaConfig = None):
        """初始化监控器"""
        self.config = config or default_config

        # 创建消费者用于获取元数据
        self._consumer = KafkaConsumer(
            bootstrap_servers=self.config.bootstrap_servers,
        )

    def get_consumer_lag(self, group_id: str, topic: str) -> Dict:
        """
        获取消费延迟

        Args:
            group_id: 消费者组 ID
            topic: Topic 名称

        Returns:
            延迟信息：
            {
                'topic': 'my-topic',
                'group_id': 'my-group',
                'partitions': [
                    {'partition': 0, 'current_offset': 100, 'end_offset': 150, 'lag': 50},
                    ...
                ],
                'total_lag': 100
            }
        """
        from kafka import TopicPartition
        from kafka.admin import KafkaAdminClient

        # 获取分区信息
        partitions = self._consumer.partitions_for_topic(topic)
        if not partitions:
            return None

        topic_partitions = [TopicPartition(topic, p) for p in partitions]

        # 获取最新 Offset
        end_offsets = self._consumer.end_offsets(topic_partitions)

        # 获取消费者组当前 Offset（需要通过 Admin API）
        # 这里简化处理，实际应该使用 kafka-python 的 ConsumerGroupCommand
        # 或者 confluent_kafka 的 AdminClient

        result = {
            'topic': topic,
            'group_id': group_id,
            'partitions': [],
            'total_lag': 0,
        }

        for tp, end_offset in end_offsets.items():
            partition_info = {
                'partition': tp.partition,
                'end_offset': end_offset,
                # current_offset 需要从消费者组获取
                # 这里简化为 0
                'current_offset': 0,
                'lag': end_offset,
            }
            result['partitions'].append(partition_info)
            result['total_lag'] += partition_info['lag']

        return result

    def close(self):
        """关闭监控器"""
        self._consumer.close()


# ============================================================
#                    主程序示例
# ============================================================

def producer_example():
    """生产者示例"""
    print("\n" + "=" * 60)
    print("生产者示例")
    print("=" * 60)

    with Producer() as producer:
        # 发送单条消息
        producer.send('test-topic', {'message': 'Hello, Kafka!'})

        # 发送带 Key 的消息
        producer.send('test-topic', {'user_id': 'u001', 'action': 'login'}, key='u001')

        # 同步发送
        result = producer.send_sync('test-topic', {'message': 'Sync message'})
        print(f"Message sent to partition {result['partition']}, offset {result['offset']}")

        # 批量发送
        messages = [
            {'order_id': f'ORD-{i}', 'amount': i * 100}
            for i in range(10)
        ]
        count = producer.send_batch('test-topic', messages, key_field='order_id')
        print(f"Sent {count} messages")


def consumer_example():
    """消费者示例"""
    print("\n" + "=" * 60)
    print("消费者示例")
    print("=" * 60)

    with Consumer(['test-topic'], group_id='example-group') as consumer:
        print("Starting consumer (Ctrl+C to stop)...")

        count = 0
        for message in consumer.consume():
            print(f"Received: {message['value']}")
            consumer.commit()

            count += 1
            if count >= 5:  # 只消费 5 条消息用于演示
                break


def event_publisher_example():
    """事件发布示例"""
    print("\n" + "=" * 60)
    print("事件发布示例")
    print("=" * 60)

    publisher = EventPublisher(topic='domain-events')

    try:
        # 创建订单事件
        order_event = OrderCreatedEvent(
            order_id='ORD-001',
            user_id='USR-001',
            items=[
                {'product_id': 'P001', 'name': 'iPhone', 'quantity': 1, 'price': 999},
                {'product_id': 'P002', 'name': 'AirPods', 'quantity': 2, 'price': 199},
            ],
            total_amount=1397
        )

        result = publisher.publish_sync(order_event, key=order_event.order_id)
        print(f"Order event published: {order_event.event_id}")

        # 用户注册事件
        user_event = UserRegisteredEvent(
            user_id='USR-002',
            email='alice@example.com',
            registered_at=datetime.now().isoformat()
        )

        publisher.publish(user_event, key=user_event.user_id)
        print(f"User event published: {user_event.event_id}")

    finally:
        publisher.close()


def worker_example():
    """Worker 消费者示例"""
    print("\n" + "=" * 60)
    print("Worker 消费者示例")
    print("=" * 60)

    # 创建处理器
    handler = OrderHandler()

    # 创建 Worker
    worker = ConsumerWorker(
        topics=['orders'],
        group_id='order-processor',
        handler=handler
    )

    # 启动 Worker
    worker.start()

    print("Worker started. Press Ctrl+C to stop...")

    try:
        # 保持运行
        while worker.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()


def main():
    """主函数"""
    print("=" * 60)
    print("Kafka 实战示例")
    print("=" * 60)

    # 创建测试 Topic
    print("\n--- 创建测试 Topic ---")
    manager = TopicManager()

    try:
        # 创建 Topic
        manager.create_topic('test-topic', partitions=3)
        manager.create_topic('domain-events', partitions=3)
        manager.create_topic('orders', partitions=6)
        manager.create_topic('orders-dlq', partitions=3)

        # 列出 Topic
        topics = manager.list_topics()
        print(f"Topics: {[t for t in topics if not t.startswith('__')]}")

    except Exception as e:
        print(f"Note: {e}")
    finally:
        manager.close()

    # 生产者示例
    try:
        producer_example()
    except Exception as e:
        print(f"Producer example error: {e}")

    # 事件发布示例
    try:
        event_publisher_example()
    except Exception as e:
        print(f"Event publisher example error: {e}")

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)
    print("""
提示：
1. 运行消费者示例前，请确保 Kafka 已启动
2. 可以使用 Docker 快速启动 Kafka：
   docker-compose up -d

3. 运行完整示例：
   python 04_practical_examples.py
""")


if __name__ == "__main__":
    main()
