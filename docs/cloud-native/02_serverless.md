# Serverless 架构

## 目录
- [Serverless 概述](#serverless-概述)
- [FaaS 函数即服务](#faas-函数即服务)
- [BaaS 后端即服务](#baas-后端即服务)
- [事件驱动架构](#事件驱动架构)
- [冷启动优化](#冷启动优化)
- [实战案例](#实战案例)

---

## Serverless 概述

### 什么是 Serverless

```
┌────────────────────────────────────────────────────────┐
│           Serverless 的演进历程                        │
├────────────────────────────────────────────────────────┤
│                                                        │
│  物理服务器  ─▶  虚拟机  ─▶  容器  ─▶  Serverless    │
│  (Bare Metal)   (VM)      (Container)   (Function)    │
│                                                        │
│  ┌─────────┐   ┌─────────┐   ┌──────┐   ┌─────┐     │
│  │ 整台服务器│   │ 多个VM  │   │多个容器│   │单个函数│  │
│  │ 月/年级别│   │ 小时级别│   │秒级启动│   │毫秒级  │  │
│  │ $1000+  │   │ $100+   │   │ $10+  │   │ $0.01 │  │
│  └─────────┘   └─────────┘   └──────┘   └─────┘     │
│                                                        │
│  管理复杂度: 高 ──────────────────────────▶ 低       │
│  灵活控制度: 高 ◀────────────────────────── 低       │
└────────────────────────────────────────────────────────┘
```

### Serverless 的核心特征

```
┌───────────────────────────────────────────────────┐
│          Serverless 五大核心特征                  │
├───────────────────────────────────────────────────┤
│                                                   │
│  1. 🚀 无服务器管理                              │
│     开发者无需关心服务器运维                      │
│                                                   │
│  2. 📊 自动弹性伸缩                              │
│     根据请求量自动扩缩容（0 ─▶ ∞）              │
│                                                   │
│  3. 💰 按使用付费                                │
│     仅为实际执行时间付费（按毫秒计费）            │
│                                                   │
│  4. ⚡ 事件驱动                                  │
│     由事件触发函数执行                            │
│                                                   │
│  5. 🔄 无状态执行                                │
│     每次调用独立，不保留状态                      │
└───────────────────────────────────────────────────┘
```

### Serverless vs 传统架构

```
┌──────────────────────────────────────────────────────┐
│               架构对比矩阵                           │
├──────────────┬───────────────┬───────────────────────┤
│   特性       │  传统服务器   │     Serverless        │
├──────────────┼───────────────┼───────────────────────┤
│ 运维负担     │  高           │  极低                 │
│ 启动时间     │  分钟-小时    │  毫秒                 │
│ 扩展方式     │  手动/自动    │  自动瞬时扩展         │
│ 成本模型     │  固定费用     │  按实际使用           │
│ 闲时成本     │  $$$          │  $0                   │
│ 冷启动       │  无           │  有（100ms-数秒）     │
│ 执行时长限制 │  无           │  有（15分钟）         │
│ 状态管理     │  本地状态     │  外部存储             │
│ 监控调试     │  熟悉工具     │  需专门工具           │
│ 供应商绑定   │  低           │  较高                 │
└──────────────┴───────────────┴───────────────────────┘
```

---

## FaaS 函数即服务

### 主流 FaaS 平台对比

```
┌────────────────────────────────────────────────────────────┐
│          AWS Lambda vs Azure Functions vs GCP              │
├──────────────┬───────────────┬──────────────┬─────────────┤
│   特性       │  AWS Lambda   │Azure Functions│Cloud Functions│
├──────────────┼───────────────┼──────────────┼─────────────┤
│ 最大执行时长 │  15 分钟      │  无限制*     │  9 分钟     │
│ 内存范围     │  128MB-10GB   │  128MB-1.5GB │  128MB-8GB  │
│ 并发限制     │  1000（默认） │  200（默认） │  1000       │
│ 冷启动       │  50-200ms     │  100-500ms   │  100-300ms  │
│ 免费额度     │  100万请求/月 │  100万/月    │  200万/月   │
│ 定价（每百万）│ $0.20        │  $0.20       │  $0.40      │
│ 支持语言     │  10+         │  8+          │  7+         │
│ VPC 支持     │  是           │  是          │  是         │
│ 容器镜像     │  是（10GB）   │  是          │  是         │
└──────────────┴───────────────┴──────────────┴─────────────┘
* Azure Durable Functions 支持长时间运行
```

### AWS Lambda 完整示例

```python
# lambda_function.py - 图片压缩函数
import json
import boto3
from PIL import Image
import io
import os

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """
    S3 事件触发的图片压缩函数

    事件流程:
    1. 用户上传图片到 S3 原始桶
    2. S3 触发 Lambda 函数
    3. Lambda 下载、压缩图片
    4. 上传到压缩桶
    """

    # 解析 S3 事件
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        print(f"处理图片: {bucket}/{key}")

        try:
            # 从 S3 下载原始图片
            response = s3.get_object(Bucket=bucket, Key=key)
            image_content = response['Body'].read()

            # 使用 Pillow 压缩图片
            image = Image.open(io.BytesIO(image_content))

            # 生成缩略图 (800x800)
            image.thumbnail((800, 800), Image.Resampling.LANCZOS)

            # 保存到内存
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85, optimize=True)
            buffer.seek(0)

            # 上传到压缩桶
            compressed_bucket = os.environ['COMPRESSED_BUCKET']
            compressed_key = f"compressed/{key}"

            s3.put_object(
                Bucket=compressed_bucket,
                Key=compressed_key,
                Body=buffer,
                ContentType='image/jpeg',
                Metadata={
                    'original-size': str(len(image_content)),
                    'compressed-size': str(buffer.getbuffer().nbytes)
                }
            )

            compression_ratio = (1 - buffer.getbuffer().nbytes / len(image_content)) * 100

            print(f"✅ 压缩完成: {key}")
            print(f"   原始大小: {len(image_content) / 1024:.2f} KB")
            print(f"   压缩后: {buffer.getbuffer().nbytes / 1024:.2f} KB")
            print(f"   压缩率: {compression_ratio:.1f}%")

            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Image compressed successfully',
                    'original_key': key,
                    'compressed_key': compressed_key,
                    'compression_ratio': f"{compression_ratio:.1f}%"
                })
            }

        except Exception as e:
            print(f"❌ 处理失败: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps({'error': str(e)})
            }
```

```yaml
# template.yaml - SAM (Serverless Application Model) 配置
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: Image compression serverless application

Globals:
  Function:
    Timeout: 60
    MemorySize: 1024
    Runtime: python3.11
    Architectures:
      - arm64  # 使用 ARM (Graviton2) 节省成本

Resources:
  # Lambda 函数
  ImageCompressorFunction:
    Type: AWS::Serverless::Function
    Properties:
      FunctionName: image-compressor
      CodeUri: src/
      Handler: lambda_function.lambda_handler
      Description: Compress images uploaded to S3

      # 环境变量
      Environment:
        Variables:
          COMPRESSED_BUCKET: !Ref CompressedBucket
          LOG_LEVEL: INFO

      # S3 事件触发器
      Events:
        S3Upload:
          Type: S3
          Properties:
            Bucket: !Ref OriginalBucket
            Events: s3:ObjectCreated:*
            Filter:
              S3Key:
                Rules:
                  - Name: suffix
                    Value: .jpg
                  - Name: suffix
                    Value: .png

      # IAM 权限
      Policies:
        - S3ReadPolicy:
            BucketName: !Ref OriginalBucket
        - S3WritePolicy:
            BucketName: !Ref CompressedBucket
        - Statement:
            - Effect: Allow
              Action:
                - logs:CreateLogGroup
                - logs:CreateLogStream
                - logs:PutLogEvents
              Resource: '*'

      # 层（依赖）
      Layers:
        - !Ref PillowLayer

  # Lambda Layer - Pillow 库
  PillowLayer:
    Type: AWS::Serverless::LayerVersion
    Properties:
      LayerName: pillow-layer
      Description: Pillow image processing library
      ContentUri: layers/pillow/
      CompatibleRuntimes:
        - python3.11
      CompatibleArchitectures:
        - arm64

  # S3 桶 - 原始图片
  OriginalBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub ${AWS::StackName}-original-${AWS::AccountId}
      LifecycleConfiguration:
        Rules:
          - Id: DeleteOldImages
            Status: Enabled
            ExpirationInDays: 30

  # S3 桶 - 压缩图片
  CompressedBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub ${AWS::StackName}-compressed-${AWS::AccountId}
      PublicAccessBlockConfiguration:
        BlockPublicAcls: false
        BlockPublicPolicy: false
        IgnorePublicAcls: false
        RestrictPublicBuckets: false

  # CloudWatch 告警 - 错误率
  FunctionErrorAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: !Sub ${AWS::StackName}-errors
      AlarmDescription: Alert when function error rate exceeds 5%
      MetricName: Errors
      Namespace: AWS/Lambda
      Statistic: Sum
      Period: 300
      EvaluationPeriods: 1
      Threshold: 5
      ComparisonOperator: GreaterThanThreshold
      Dimensions:
        - Name: FunctionName
          Value: !Ref ImageCompressorFunction

Outputs:
  FunctionArn:
    Description: Lambda Function ARN
    Value: !GetAtt ImageCompressorFunction.Arn

  OriginalBucketName:
    Description: Original images bucket
    Value: !Ref OriginalBucket

  CompressedBucketName:
    Description: Compressed images bucket
    Value: !Ref CompressedBucket
```

```bash
# 部署脚本
#!/bin/bash

# 1. 打包 Pillow 层
mkdir -p layers/pillow/python
pip install Pillow -t layers/pillow/python/

# 2. SAM 构建
sam build --use-container

# 3. SAM 部署
sam deploy \
  --stack-name image-compressor \
  --capabilities CAPABILITY_IAM \
  --region us-east-1 \
  --parameter-overrides \
    Environment=production

# 4. 测试上传
aws s3 cp test-image.jpg s3://image-compressor-original-123456789012/

# 5. 查看日志
sam logs -n ImageCompressorFunction --tail
```

### API Gateway + Lambda

```python
# api_handler.py - RESTful API 示例
import json
import boto3
from datetime import datetime
import uuid

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Users')

def lambda_handler(event, context):
    """
    API Gateway 集成的 CRUD 函数
    """

    http_method = event['httpMethod']
    path = event['path']

    # 路由处理
    if path == '/users':
        if http_method == 'GET':
            return list_users()
        elif http_method == 'POST':
            return create_user(json.loads(event['body']))

    elif path.startswith('/users/'):
        user_id = path.split('/')[-1]
        if http_method == 'GET':
            return get_user(user_id)
        elif http_method == 'PUT':
            return update_user(user_id, json.loads(event['body']))
        elif http_method == 'DELETE':
            return delete_user(user_id)

    return response(404, {'error': 'Not found'})

def list_users():
    """获取用户列表"""
    try:
        result = table.scan(Limit=100)
        return response(200, {
            'users': result['Items'],
            'count': len(result['Items'])
        })
    except Exception as e:
        return response(500, {'error': str(e)})

def get_user(user_id):
    """获取单个用户"""
    try:
        result = table.get_item(Key={'user_id': user_id})
        if 'Item' in result:
            return response(200, result['Item'])
        else:
            return response(404, {'error': 'User not found'})
    except Exception as e:
        return response(500, {'error': str(e)})

def create_user(data):
    """创建用户"""
    try:
        user = {
            'user_id': str(uuid.uuid4()),
            'name': data['name'],
            'email': data['email'],
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }
        table.put_item(Item=user)
        return response(201, user)
    except Exception as e:
        return response(400, {'error': str(e)})

def update_user(user_id, data):
    """更新用户"""
    try:
        table.update_item(
            Key={'user_id': user_id},
            UpdateExpression='SET #name = :name, email = :email, updated_at = :updated',
            ExpressionAttributeNames={'#name': 'name'},
            ExpressionAttributeValues={
                ':name': data['name'],
                ':email': data['email'],
                ':updated': datetime.utcnow().isoformat()
            }
        )
        return response(200, {'message': 'User updated'})
    except Exception as e:
        return response(400, {'error': str(e)})

def delete_user(user_id):
    """删除用户"""
    try:
        table.delete_item(Key={'user_id': user_id})
        return response(204, {})
    except Exception as e:
        return response(400, {'error': str(e)})

def response(status_code, body):
    """标准响应格式"""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE',
            'Access-Control-Allow-Headers': 'Content-Type,Authorization'
        },
        'body': json.dumps(body, default=str)
    }
```

---

## BaaS 后端即服务

### BaaS 服务全景图

```
┌────────────────────────────────────────────────────┐
│              BaaS 服务分类                         │
├────────────────────────────────────────────────────┤
│                                                    │
│  📦 数据库即服务                                   │
│     ├─ DynamoDB (AWS)                             │
│     ├─ Firestore (Google)                         │
│     ├─ Cosmos DB (Azure)                          │
│     └─ Supabase (开源)                            │
│                                                    │
│  🔐 身份认证即服务                                 │
│     ├─ Auth0                                      │
│     ├─ Firebase Auth                              │
│     ├─ AWS Cognito                                │
│     └─ Clerk                                      │
│                                                    │
│  📨 通知服务                                       │
│     ├─ SNS (AWS)                                  │
│     ├─ SendGrid (邮件)                            │
│     ├─ Twilio (短信)                              │
│     └─ FCM (推送)                                 │
│                                                    │
│  💳 支付服务                                       │
│     ├─ Stripe                                     │
│     ├─ PayPal                                     │
│     └─ Square                                     │
│                                                    │
│  🔍 搜索服务                                       │
│     ├─ Algolia                                    │
│     ├─ Elasticsearch Service                      │
│     └─ Typesense                                  │
└────────────────────────────────────────────────────┘
```

### Firebase 全栈示例

```javascript
// firebase-app.js - Firebase Serverless 应用
import { initializeApp } from 'firebase/app';
import {
  getFirestore,
  collection,
  addDoc,
  getDocs,
  query,
  where,
  orderBy,
  limit,
  onSnapshot
} from 'firebase/firestore';
import {
  getAuth,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  onAuthStateChanged
} from 'firebase/auth';
import {
  getStorage,
  ref,
  uploadBytes,
  getDownloadURL
} from 'firebase/storage';

// Firebase 配置
const firebaseConfig = {
  apiKey: "AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXX",
  authDomain: "myapp.firebaseapp.com",
  projectId: "myapp",
  storageBucket: "myapp.appspot.com",
  messagingSenderId: "123456789",
  appId: "1:123456789:web:xxxxx"
};

// 初始化
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
const auth = getAuth(app);
const storage = getStorage(app);

// ========== 认证 ==========

async function registerUser(email, password, displayName) {
  try {
    const userCredential = await createUserWithEmailAndPassword(auth, email, password);
    const user = userCredential.user;

    // 创建用户文档
    await addDoc(collection(db, 'users'), {
      uid: user.uid,
      email: email,
      displayName: displayName,
      createdAt: new Date(),
      role: 'user'
    });

    console.log('✅ 用户注册成功:', user.uid);
    return user;
  } catch (error) {
    console.error('❌ 注册失败:', error.message);
    throw error;
  }
}

async function loginUser(email, password) {
  try {
    const userCredential = await signInWithEmailAndPassword(auth, email, password);
    console.log('✅ 登录成功:', userCredential.user.email);
    return userCredential.user;
  } catch (error) {
    console.error('❌ 登录失败:', error.message);
    throw error;
  }
}

// 监听认证状态
onAuthStateChanged(auth, (user) => {
  if (user) {
    console.log('👤 用户已登录:', user.email);
  } else {
    console.log('👤 用户未登录');
  }
});

// ========== Firestore 数据库 ==========

async function createPost(title, content) {
  const user = auth.currentUser;
  if (!user) throw new Error('未登录');

  try {
    const docRef = await addDoc(collection(db, 'posts'), {
      title: title,
      content: content,
      authorId: user.uid,
      authorEmail: user.email,
      createdAt: new Date(),
      likes: 0,
      comments: []
    });

    console.log('✅ 文章创建成功:', docRef.id);
    return docRef.id;
  } catch (error) {
    console.error('❌ 创建失败:', error);
    throw error;
  }
}

async function getRecentPosts(limitCount = 10) {
  try {
    const q = query(
      collection(db, 'posts'),
      orderBy('createdAt', 'desc'),
      limit(limitCount)
    );

    const querySnapshot = await getDocs(q);
    const posts = [];

    querySnapshot.forEach((doc) => {
      posts.push({ id: doc.id, ...doc.data() });
    });

    console.log(`✅ 获取到 ${posts.length} 篇文章`);
    return posts;
  } catch (error) {
    console.error('❌ 查询失败:', error);
    throw error;
  }
}

// 实时监听
function subscribeToUserPosts(userId) {
  const q = query(
    collection(db, 'posts'),
    where('authorId', '==', userId),
    orderBy('createdAt', 'desc')
  );

  // 返回取消订阅函数
  return onSnapshot(q, (snapshot) => {
    snapshot.docChanges().forEach((change) => {
      if (change.type === 'added') {
        console.log('新文章:', change.doc.data());
      }
      if (change.type === 'modified') {
        console.log('文章更新:', change.doc.data());
      }
      if (change.type === 'removed') {
        console.log('文章删除:', change.doc.id);
      }
    });
  });
}

// ========== Storage 文件上传 ==========

async function uploadImage(file) {
  const user = auth.currentUser;
  if (!user) throw new Error('未登录');

  try {
    // 生成唯一文件名
    const filename = `${user.uid}/${Date.now()}_${file.name}`;
    const storageRef = ref(storage, `images/${filename}`);

    // 上传文件
    const snapshot = await uploadBytes(storageRef, file);
    console.log('✅ 上传成功:', snapshot.totalBytes, 'bytes');

    // 获取下载 URL
    const downloadURL = await getDownloadURL(snapshot.ref);
    console.log('📥 下载链接:', downloadURL);

    return downloadURL;
  } catch (error) {
    console.error('❌ 上传失败:', error);
    throw error;
  }
}

// ========== 使用示例 ==========

async function demo() {
  // 注册
  await registerUser('user@example.com', 'password123', 'John Doe');

  // 登录
  await loginUser('user@example.com', 'password123');

  // 创建文章
  const postId = await createPost('我的第一篇文章', '这是内容...');

  // 获取最近文章
  const posts = await getRecentPosts(5);
  console.log('最近文章:', posts);

  // 订阅实时更新
  const unsubscribe = subscribeToUserPosts(auth.currentUser.uid);

  // 5秒后取消订阅
  setTimeout(() => {
    unsubscribe();
    console.log('取消订阅');
  }, 5000);
}
```

```json
// firestore.rules - 安全规则
{
  "rules": {
    "users": {
      "$uid": {
        // 只有用户本人可以读写自己的数据
        ".read": "auth != null && auth.uid == $uid",
        ".write": "auth != null && auth.uid == $uid"
      }
    },
    "posts": {
      ".read": true,  // 所有人可读
      "$postId": {
        // 只有作者可以修改/删除
        ".write": "auth != null && (!data.exists() || data.child('authorId').val() == auth.uid)"
      }
    }
  }
}
```

---

## 事件驱动架构

### 事件驱动模式

```
┌────────────────────────────────────────────────────┐
│           Serverless 事件驱动架构                  │
├────────────────────────────────────────────────────┤
│                                                    │
│  事件源          触发器          函数              │
│    │              │               │               │
│    ▼              ▼               ▼               │
│  ┌────┐        ┌─────┐        ┌──────┐          │
│  │ S3 │───────▶│Event│───────▶│Lambda│          │
│  └────┘        │Rule │        └──────┘          │
│                └─────┘            │               │
│  ┌────┐                           │               │
│  │API │──────────────────────────▶│               │
│  │GW  │                           │               │
│  └────┘                           │               │
│                                   ▼               │
│  ┌────┐                      ┌────────┐          │
│  │SQS │─────────────────────▶│ 数据库 │          │
│  └────┘                      └────────┘          │
│                                                    │
│  ┌────┐                      ┌────────┐          │
│  │Cron│─────────────────────▶│  SNS   │          │
│  └────┘                      └────────┘          │
└────────────────────────────────────────────────────┘
```

### 复杂事件处理示例

```python
# order_processor.py - 订单处理工作流
import json
import boto3
from datetime import datetime

sqs = boto3.client('sqs')
sns = boto3.client('sns')
dynamodb = boto3.resource('dynamodb')

orders_table = dynamodb.Table('Orders')
inventory_table = dynamodb.Table('Inventory')

def lambda_handler(event, context):
    """
    订单处理流程:
    1. SQS 接收订单
    2. 验证库存
    3. 扣减库存
    4. 创建订单记录
    5. 发送通知
    """

    for record in event['Records']:
        # 解析 SQS 消息
        order = json.loads(record['body'])
        order_id = order['order_id']

        try:
            # 步骤 1: 验证库存
            if not check_inventory(order['items']):
                handle_insufficient_inventory(order)
                continue

            # 步骤 2: 扣减库存
            deduct_inventory(order['items'])

            # 步骤 3: 创建订单
            create_order(order)

            # 步骤 4: 触发支付流程
            trigger_payment(order)

            # 步骤 5: 发送确认通知
            send_notification(order, 'ORDER_CREATED')

            print(f"✅ 订单处理成功: {order_id}")

        except Exception as e:
            print(f"❌ 订单处理失败: {order_id}, 错误: {str(e)}")

            # 发送到死信队列
            send_to_dlq(order, str(e))

            # 回滚操作
            rollback_order(order)

def check_inventory(items):
    """检查库存"""
    for item in items:
        response = inventory_table.get_item(
            Key={'product_id': item['product_id']}
        )

        if 'Item' not in response:
            return False

        available = response['Item']['quantity']
        if available < item['quantity']:
            return False

    return True

def deduct_inventory(items):
    """扣减库存"""
    for item in items:
        inventory_table.update_item(
            Key={'product_id': item['product_id']},
            UpdateExpression='SET quantity = quantity - :qty',
            ConditionExpression='quantity >= :qty',
            ExpressionAttributeValues={':qty': item['quantity']}
        )

def create_order(order):
    """创建订单记录"""
    orders_table.put_item(Item={
        'order_id': order['order_id'],
        'user_id': order['user_id'],
        'items': order['items'],
        'total_amount': order['total_amount'],
        'status': 'PENDING_PAYMENT',
        'created_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat()
    })

def trigger_payment(order):
    """触发支付流程（发送到支付队列）"""
    payment_queue_url = 'https://sqs.us-east-1.amazonaws.com/123456/payment-queue'

    sqs.send_message(
        QueueUrl=payment_queue_url,
        MessageBody=json.dumps({
            'order_id': order['order_id'],
            'amount': order['total_amount'],
            'user_id': order['user_id']
        })
    )

def send_notification(order, event_type):
    """发送 SNS 通知"""
    topic_arn = 'arn:aws:sns:us-east-1:123456:order-events'

    sns.publish(
        TopicArn=topic_arn,
        Subject=f'订单通知: {event_type}',
        Message=json.dumps(order),
        MessageAttributes={
            'event_type': {'DataType': 'String', 'StringValue': event_type}
        }
    )

def handle_insufficient_inventory(order):
    """处理库存不足"""
    print(f"⚠️  库存不足: {order['order_id']}")

    orders_table.put_item(Item={
        'order_id': order['order_id'],
        'status': 'CANCELLED_INSUFFICIENT_INVENTORY',
        'created_at': datetime.utcnow().isoformat()
    })

    send_notification(order, 'ORDER_CANCELLED')

def send_to_dlq(order, error):
    """发送到死信队列"""
    dlq_url = 'https://sqs.us-east-1.amazonaws.com/123456/order-dlq'

    sqs.send_message(
        QueueUrl=dlq_url,
        MessageBody=json.dumps({
            'order': order,
            'error': error,
            'timestamp': datetime.utcnow().isoformat()
        })
    )

def rollback_order(order):
    """回滚订单（补偿事务）"""
    # 恢复库存
    for item in order.get('items', []):
        try:
            inventory_table.update_item(
                Key={'product_id': item['product_id']},
                UpdateExpression='SET quantity = quantity + :qty',
                ExpressionAttributeValues={':qty': item['quantity']}
            )
        except:
            pass  # 记录日志，人工介入
```

---

## 冷启动优化

### 冷启动分析

```
┌────────────────────────────────────────────────────┐
│              Lambda 冷启动时间分解                 │
├────────────────────────────────────────────────────┤
│                                                    │
│  完整请求延迟                                      │
│  ├─ 冷启动 (100ms - 10s)                         │
│  │   ├─ 下载代码包 (50-500ms)                    │
│  │   ├─ 启动运行时 (10-100ms)                    │
│  │   ├─ 初始化代码 (50-数秒)                     │
│  │   │   ├─ import 模块                          │
│  │   │   ├─ 连接数据库                           │
│  │   │   └─ 加载配置                             │
│  │   └─ 执行handler前代码                        │
│  │                                                 │
│  └─ 热启动 (1-10ms)                              │
│      └─ 仅执行 handler 函数                       │
│                                                    │
│  影响因素:                                         │
│  • 运行时类型 (Python < Node.js < Java)          │
│  • 代码包大小                                      │
│  • VPC 配置 (+数秒)                               │
│  • Provisioned Concurrency (消除冷启动)          │
└────────────────────────────────────────────────────┘
```

### 冷启动优化技巧

```python
# optimized_lambda.py - 优化的 Lambda 函数

# ====== 优化 1: 全局变量复用连接 ======
import json
import os
import boto3
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

# 在 handler 外部初始化（全局作用域）
# 后续调用可复用连接
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['TABLE_NAME'])

# 修补 AWS SDK 以支持 X-Ray 追踪
patch_all()

# ====== 优化 2: 懒加载重模块 ======
heavy_module = None

def get_heavy_module():
    global heavy_module
    if heavy_module is None:
        import pandas as pd  # 仅在需要时加载
        heavy_module = pd
    return heavy_module

# ====== 优化 3: 配置预加载 ======
CONFIG_CACHE = {}

def get_config(key):
    if key not in CONFIG_CACHE:
        ssm = boto3.client('ssm')
        response = ssm.get_parameter(
            Name=key,
            WithDecryption=True
        )
        CONFIG_CACHE[key] = response['Parameter']['Value']
    return CONFIG_CACHE[key]

# ====== 优化 4: 连接池 ======
import pymysql
from dbutils.pooled_db import PooledDB

# MySQL 连接池（全局）
db_pool = PooledDB(
    creator=pymysql,
    maxconnections=1,  # Lambda 单并发
    host=os.environ['DB_HOST'],
    user=os.environ['DB_USER'],
    password=get_config('/myapp/db_password'),
    database=os.environ['DB_NAME'],
    charset='utf8mb4'
)

@xray_recorder.capture('lambda_handler')
def lambda_handler(event, context):
    """
    优化后的 handler
    """

    # 使用连接池获取连接
    conn = db_pool.connection()
    cursor = conn.cursor()

    try:
        # 业务逻辑
        cursor.execute("SELECT * FROM users WHERE id = %s", (event['user_id'],))
        user = cursor.fetchone()

        # DynamoDB 操作（复用全局连接）
        table.put_item(Item={
            'user_id': event['user_id'],
            'timestamp': context.request_id,
            'data': json.dumps(user)
        })

        return {
            'statusCode': 200,
            'body': json.dumps({'user': user})
        }

    finally:
        cursor.close()
        conn.close()  # 归还到连接池

# ====== 优化 5: Provisioned Concurrency ======
# 通过 SAM 配置预留并发
"""
Resources:
  MyFunction:
    Type: AWS::Serverless::Function
    Properties:
      AutoPublishAlias: live
      ProvisionedConcurrencyConfig:
        ProvisionedConcurrentExecutions: 5  # 保持5个实例热启动
"""
```

### 冷启动性能对比

```bash
# benchmark.sh - 冷启动测试脚本

#!/bin/bash

echo "测试 Lambda 冷启动性能"
echo "======================================"

# 1. 清空所有热实例（等待15分钟或强制更新函数）
aws lambda update-function-code \
  --function-name my-function \
  --zip-file fileb://function.zip \
  --no-publish

sleep 60  # 等待更新完成

# 2. 测试冷启动（首次调用）
echo "冷启动测试..."
for i in {1..5}; do
  start=$(date +%s%3N)
  aws lambda invoke \
    --function-name my-function \
    --payload '{"test": true}' \
    response.json > /dev/null
  end=$(date +%s%3N)
  duration=$((end - start))
  echo "  调用 $i: ${duration}ms"
done

echo ""
echo "热启动测试..."
# 3. 测试热启动（连续调用）
for i in {1..5}; do
  start=$(date +%s%3N)
  aws lambda invoke \
    --function-name my-function \
    --payload '{"test": true}' \
    response.json > /dev/null
  end=$(date +%s%3N)
  duration=$((end - start))
  echo "  调用 $i: ${duration}ms"
done
```

**典型测试结果**：
```
冷启动测试...
  调用 1: 1247ms  ← 最慢（完整冷启动）
  调用 2: 982ms
  调用 3: 1104ms
  调用 4: 1032ms
  调用 5: 1156ms

热启动测试...
  调用 1: 12ms    ← 快100倍！
  调用 2: 9ms
  调用 3: 11ms
  调用 4: 10ms
  调用 5: 13ms
```

---

## 实战案例

### 案例: Serverless 博客系统

**架构图**:
```
                 Serverless 博客架构
┌──────────────────────────────────────────────────┐
│                                                  │
│  用户 ─▶ CloudFront CDN                         │
│              │                                   │
│              ├─▶ S3 (静态网站)                  │
│              │                                   │
│              └─▶ API Gateway                    │
│                     │                            │
│              ┌──────┼──────┐                    │
│              │      │      │                     │
│          ┌───▼─┐ ┌─▼──┐ ┌─▼──┐                │
│          │Post│ │User│ │Auth│                  │
│          │ λ  │ │ λ  │ │ λ  │                  │
│          └─┬──┘ └─┬──┘ └─┬──┘                 │
│            │      │      │                       │
│            └──────┼──────┘                      │
│                   │                              │
│            ┌──────▼──────┐                      │
│            │  DynamoDB   │                      │
│            │  (NoSQL)    │                      │
│            └─────────────┘                      │
│                                                  │
│  月成本: ~$5 (100K 请求)                        │
└──────────────────────────────────────────────────┘
```

完整代码已省略（参考前面的 API Gateway + Lambda 示例）

---

## 总结

### Serverless 适用场景

```
✅ 适合 Serverless:
  • 间歇性工作负载（定时任务）
  • 突发流量（营销活动）
  • 事件驱动处理（文件上传、消息队列）
  • 快速原型开发
  • 小型 API 服务
  • 数据处理管道

❌ 不适合 Serverless:
  • 长时间运行任务（>15分钟）
  • 持续高负载（热启动比例低）
  • 有状态应用（需要本地缓存）
  • 对延迟极度敏感（<10ms）
  • 需要特定硬件（GPU）
```

### 下一步

- [03_cloud_patterns.md](03_cloud_patterns.md) - 云设计模式
- [04_multi_cloud.md](04_multi_cloud.md) - 多云架构
- [05_cost_optimization.md](05_cost_optimization.md) - 成本优化深入
