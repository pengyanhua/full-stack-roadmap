# 密钥管理与Vault实践

## 目录
- [概述](#概述)
- [HashiCorp Vault完整教程](#hashicorp-vault完整教程)
- [动态密钥生成](#动态密钥生成)
- [密钥轮转策略](#密钥轮转策略)
- [KMS集成](#kms集成)
- [实战案例](#实战案例)

## 概述

### 密钥管理挑战

```
传统密钥管理问题
┌─────────────────────────────────────────────────────────────┐
│ ❌ 硬编码在代码中                                            │
│    const DB_PASSWORD = "admin123"                           │
│                                                              │
│ ❌ 环境变量泄露                                              │
│    export DB_PASS=admin123                                  │
│    (在进程列表中可见)                                        │
│                                                              │
│ ❌ 配置文件明文存储                                          │
│    config.yaml:                                             │
│      database:                                              │
│        password: admin123                                   │
│                                                              │
│ ❌ 共享密钥,难以追踪                                         │
│    多个应用共用同一个数据库密码                              │
│                                                              │
│ ❌ 密钥轮转困难                                              │
│    需要手动更新所有使用方                                    │
└─────────────────────────────────────────────────────────────┘

现代密钥管理方案
┌─────────────────────────────────────────────────────────────┐
│ ✓ 集中式密钥管理                                             │
│   ┌──────────────────────────────────────────┐             │
│   │     Vault / AWS Secrets Manager         │             │
│   │     - 加密存储                           │             │
│   │     - 访问控制                           │             │
│   │     - 审计日志                           │             │
│   └──────────────────────────────────────────┘             │
│                                                              │
│ ✓ 动态密钥生成                                               │
│   应用请求 -> Vault生成临时密钥 -> 自动过期                  │
│                                                              │
│ ✓ 加密即服务                                                 │
│   应用发送明文 -> Vault加密 -> 返回密文                      │
│                                                              │
│ ✓ 密钥版本管理                                               │
│   v1, v2, v3... 支持回滚                                    │
│                                                              │
│ ✓ 自动轮转                                                   │
│   定期自动生成新密钥并通知应用                               │
└─────────────────────────────────────────────────────────────┘
```

### Vault架构

```
┌──────────────────────────────────────────────────────────────┐
│                  HashiCorp Vault Architecture                │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  客户端                                                        │
│  ┌──────────┬──────────┬──────────┬──────────┐              │
│  │   CLI    │   API    │   SDK    │  Agent   │              │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┘              │
│       └──────────┴──────────┴──────────┘                     │
│                    │ HTTPS                                   │
│  ┌─────────────────▼────────────────────────────────┐        │
│  │              Vault Server                        │        │
│  │  ┌────────────────────────────────────────────┐ │        │
│  │  │  HTTP/gRPC API Layer                       │ │        │
│  │  └──────────────────┬─────────────────────────┘ │        │
│  │                     │                            │        │
│  │  ┌──────────────────▼─────────────────────────┐ │        │
│  │  │  Core (核心引擎)                           │ │        │
│  │  │  ┌──────────────────────────────────────┐ │ │        │
│  │  │  │  认证方法 (Auth Methods)             │ │ │        │
│  │  │  │  ┌─────┬─────┬─────┬─────┬─────┐    │ │ │        │
│  │  │  │  │Token│LDAP │Kube │AWS  │OIDC │    │ │ │        │
│  │  │  │  └─────┴─────┴─────┴─────┴─────┘    │ │ │        │
│  │  │  └──────────────────────────────────────┘ │ │        │
│  │  │                                            │ │        │
│  │  │  ┌──────────────────────────────────────┐ │ │        │
│  │  │  │  密钥引擎 (Secrets Engines)          │ │ │        │
│  │  │  │  ┌─────┬─────┬─────┬─────┬─────┐    │ │ │        │
│  │  │  │  │ KV  │DB   │AWS  │PKI  │SSH  │    │ │ │        │
│  │  │  │  └─────┴─────┴─────┴─────┴─────┘    │ │ │        │
│  │  │  └──────────────────────────────────────┘ │ │        │
│  │  │                                            │ │        │
│  │  │  ┌──────────────────────────────────────┐ │ │        │
│  │  │  │  策略引擎 (Policy Engine)            │ │ │        │
│  │  │  │  - ACL策略                           │ │ │        │
│  │  │  │  - 路径权限                          │ │ │        │
│  │  │  │  - RBAC                              │ │ │        │
│  │  │  └──────────────────────────────────────┘ │ │        │
│  │  │                                            │ │        │
│  │  │  ┌──────────────────────────────────────┐ │ │        │
│  │  │  │  审计设备 (Audit Devices)            │ │ │        │
│  │  │  │  - File, Syslog, Socket              │ │ │        │
│  │  │  └──────────────────────────────────────┘ │ │        │
│  │  └────────────────────────────────────────────┘ │        │
│  └───────────────────┬──────────────────────────────┘        │
│                      │                                        │
│  ┌───────────────────▼──────────────────────────────┐        │
│  │        存储后端 (Storage Backend)                │        │
│  │  ┌────────┬────────┬────────┬────────┐          │        │
│  │  │Consul  │etcd    │S3      │File    │          │        │
│  │  └────────┴────────┴────────┴────────┘          │        │
│  │  (加密存储)                                      │        │
│  └───────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

## HashiCorp Vault完整教程

### 安装与初始化

```bash
# 1. 安装Vault (Docker方式)
docker run -d \
  --name vault \
  --cap-add=IPC_LOCK \
  -p 8200:8200 \
  -e 'VAULT_DEV_ROOT_TOKEN_ID=myroot' \
  -e 'VAULT_DEV_LISTEN_ADDRESS=0.0.0.0:8200' \
  vault:latest

# 2. 安装Vault (二进制方式 - Linux)
wget https://releases.hashicorp.com/vault/1.15.0/vault_1.15.0_linux_amd64.zip
unzip vault_1.15.0_linux_amd64.zip
sudo mv vault /usr/local/bin/
vault --version

# 3. 生产环境配置文件
cat > /etc/vault/config.hcl <<EOF
ui = true

storage "consul" {
  address = "127.0.0.1:8500"
  path    = "vault/"
}

listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_cert_file = "/etc/vault/tls/vault.crt"
  tls_key_file  = "/etc/vault/tls/vault.key"
}

api_addr = "https://vault.company.com:8200"
cluster_addr = "https://vault.company.com:8201"

# 启用审计
audit {
  type = "file"
  path = "/var/log/vault/audit.log"
}

# 启用遥测
telemetry {
  prometheus_retention_time = "30s"
  disable_hostname = true
}
EOF

# 4. 启动Vault服务器
vault server -config=/etc/vault/config.hcl

# 5. 初始化Vault (仅首次)
export VAULT_ADDR='http://127.0.0.1:8200'

vault operator init \
  -key-shares=5 \
  -key-threshold=3 \
  -format=json > vault-init-output.json

# 保存输出:
# Unseal Key 1: xxx (保存到安全位置)
# Unseal Key 2: yyy
# Unseal Key 3: zzz
# Unseal Key 4: aaa
# Unseal Key 5: bbb
# Initial Root Token: root-token-here

# 6. 解封Vault (每次重启后需要)
vault operator unseal <unseal-key-1>
vault operator unseal <unseal-key-2>
vault operator unseal <unseal-key-3>
# 3次后解封成功

# 7. 登录
export VAULT_TOKEN='root-token-here'
vault login

# 8. 查看状态
vault status
```

### KV密钥引擎配置

```bash
# 1. 启用KV v2引擎
vault secrets enable -path=secret kv-v2

# 2. 写入密钥
vault kv put secret/database \
  username="admin" \
  password="super-secret-password" \
  host="db.company.com" \
  port="5432"

# 3. 读取密钥
vault kv get secret/database
vault kv get -field=password secret/database

# 4. 读取JSON格式
vault kv get -format=json secret/database | jq

# 5. 更新密钥 (创建新版本)
vault kv put secret/database \
  username="admin" \
  password="new-password" \
  host="db.company.com" \
  port="5432"

# 6. 查看历史版本
vault kv metadata get secret/database

# 7. 读取指定版本
vault kv get -version=1 secret/database

# 8. 删除最新版本 (软删除)
vault kv delete secret/database

# 9. 恢复删除的版本
vault kv undelete -versions=2 secret/database

# 10. 永久删除版本
vault kv destroy -versions=1 secret/database

# 11. 删除所有版本和元数据
vault kv metadata delete secret/database
```

### 策略配置

```bash
# 1. 创建策略文件
cat > app-policy.hcl <<EOF
# 读取数据库密钥
path "secret/data/database/*" {
  capabilities = ["read"]
}

# 列出数据库密钥
path "secret/metadata/database/*" {
  capabilities = ["list"]
}

# 读写应用配置
path "secret/data/app/config" {
  capabilities = ["create", "read", "update"]
}

# 管理员完全权限
path "secret/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# 生成数据库凭证
path "database/creds/readonly" {
  capabilities = ["read"]
}

# 加密数据
path "transit/encrypt/orders" {
  capabilities = ["update"]
}

# 解密数据
path "transit/decrypt/orders" {
  capabilities = ["update"]
}
EOF

# 2. 写入策略
vault policy write app-policy app-policy.hcl

# 3. 查看策略
vault policy read app-policy

# 4. 列出所有策略
vault policy list

# 5. 删除策略
vault policy delete app-policy
```

### 认证方法

#### AppRole认证

```bash
# 1. 启用AppRole认证
vault auth enable approle

# 2. 创建角色
vault write auth/approle/role/web-app \
  token_policies="app-policy" \
  token_ttl=1h \
  token_max_ttl=4h \
  secret_id_ttl=24h

# 3. 获取RoleID
vault read auth/approle/role/web-app/role-id
# role_id: xxxxx

# 4. 生成SecretID
vault write -f auth/approle/role/web-app/secret-id
# secret_id: yyyyy

# 5. 使用AppRole登录
vault write auth/approle/login \
  role_id="xxxxx" \
  secret_id="yyyyy"
# 返回token

# 6. Python应用集成示例
cat > vault_client.py <<'EOF'
import hvac
import os

class VaultClient:
    def __init__(self):
        self.client = hvac.Client(url='http://localhost:8200')
        self._authenticate()

    def _authenticate(self):
        """使用AppRole认证"""
        role_id = os.environ.get('VAULT_ROLE_ID')
        secret_id = os.environ.get('VAULT_SECRET_ID')

        response = self.client.auth.approle.login(
            role_id=role_id,
            secret_id=secret_id
        )

        self.client.token = response['auth']['client_token']

    def get_secret(self, path):
        """读取密钥"""
        secret = self.client.secrets.kv.v2.read_secret_version(
            path=path
        )
        return secret['data']['data']

    def put_secret(self, path, data):
        """写入密钥"""
        self.client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret=data
        )

# 使用示例
vault = VaultClient()
db_creds = vault.get_secret('database/prod')
print(f"Username: {db_creds['username']}")
print(f"Password: {db_creds['password']}")
EOF
```

#### Kubernetes认证

```bash
# 1. 启用Kubernetes认证
vault auth enable kubernetes

# 2. 配置Kubernetes认证
vault write auth/kubernetes/config \
  kubernetes_host="https://kubernetes.default.svc:443" \
  kubernetes_ca_cert=@/var/run/secrets/kubernetes.io/serviceaccount/ca.crt \
  token_reviewer_jwt=@/var/run/secrets/kubernetes.io/serviceaccount/token

# 3. 创建角色绑定
vault write auth/kubernetes/role/web-app \
  bound_service_account_names=web-app \
  bound_service_account_namespaces=production \
  policies=app-policy \
  ttl=1h

# 4. Kubernetes Pod中使用
cat > k8s-vault-example.yaml <<'EOF'
apiVersion: v1
kind: ServiceAccount
metadata:
  name: web-app
  namespace: production
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
      annotations:
        vault.hashicorp.com/agent-inject: "true"
        vault.hashicorp.com/role: "web-app"
        vault.hashicorp.com/agent-inject-secret-db: "secret/data/database/prod"
        vault.hashicorp.com/agent-inject-template-db: |
          {{- with secret "secret/data/database/prod" -}}
          export DB_USER="{{ .Data.data.username }}"
          export DB_PASS="{{ .Data.data.password }}"
          export DB_HOST="{{ .Data.data.host }}"
          {{- end }}
    spec:
      serviceAccountName: web-app
      containers:
      - name: app
        image: myapp:latest
        command: ["/bin/sh", "-c"]
        args:
        - source /vault/secrets/db && ./app
EOF
```

## 动态密钥生成

### 数据库动态凭证

```bash
# 1. 启用数据库密钥引擎
vault secrets enable database

# 2. 配置PostgreSQL连接
vault write database/config/postgresql \
  plugin_name=postgresql-database-plugin \
  allowed_roles="readonly,readwrite" \
  connection_url="postgresql://{{username}}:{{password}}@postgres:5432/mydb?sslmode=disable" \
  username="vaultadmin" \
  password="vaultpass"

# 3. 创建只读角色
vault write database/roles/readonly \
  db_name=postgresql \
  creation_statements="
    CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}';
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO \"{{name}}\";
  " \
  default_ttl="1h" \
  max_ttl="24h"

# 4. 创建读写角色
vault write database/roles/readwrite \
  db_name=postgresql \
  creation_statements="
    CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}';
    GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO \"{{name}}\";
  " \
  default_ttl="1h" \
  max_ttl="24h"

# 5. 生成动态凭证
vault read database/creds/readonly
# Key                Value
# ---                -----
# lease_id           database/creds/readonly/abc123
# lease_duration     1h
# username           v-token-readonly-xyz
# password           A1B2C3D4E5

# 6. 续租凭证
vault lease renew database/creds/readonly/abc123

# 7. 撤销凭证
vault lease revoke database/creds/readonly/abc123

# 8. 应用集成示例
cat > dynamic_db_client.py <<'EOF'
import hvac
import psycopg2
from contextlib import contextmanager

class DynamicDBClient:
    def __init__(self, vault_client):
        self.vault = vault_client

    @contextmanager
    def get_connection(self, role='readonly'):
        """获取数据库连接 (自动管理凭证生命周期)"""
        # 生成动态凭证
        creds = self.vault.client.read(f'database/creds/{role}')
        username = creds['data']['username']
        password = creds['data']['password']
        lease_id = creds['lease_id']

        # 创建连接
        conn = psycopg2.connect(
            host='postgres',
            database='mydb',
            user=username,
            password=password
        )

        try:
            yield conn
        finally:
            # 关闭连接
            conn.close()
            # 撤销凭证
            self.vault.client.sys.revoke_lease(lease_id)

# 使用示例
vault = VaultClient()
db_client = DynamicDBClient(vault)

with db_client.get_connection('readonly') as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users LIMIT 10")
    results = cursor.fetchall()
# 凭证自动撤销
EOF
```

### AWS动态凭证

```bash
# 1. 启用AWS密钥引擎
vault secrets enable aws

# 2. 配置AWS根凭证
vault write aws/config/root \
  access_key=AKIAIOSFODNN7EXAMPLE \
  secret_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY \
  region=us-east-1

# 3. 创建角色 (IAM用户类型)
vault write aws/roles/s3-readonly \
  credential_type=iam_user \
  policy_document=-<<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": ["arn:aws:s3:::mybucket/*"]
    }
  ]
}
EOF

# 4. 创建角色 (STS AssumeRole类型)
vault write aws/roles/ec2-admin \
  credential_type=assumed_role \
  role_arns=arn:aws:iam::123456789012:role/EC2AdminRole \
  default_sts_ttl=3600 \
  max_sts_ttl=86400

# 5. 生成AWS凭证
vault read aws/creds/s3-readonly
# Key                Value
# ---                -----
# lease_id           aws/creds/s3-readonly/abc123
# access_key         AKIAIOSFODNN7EXAMPLE2
# secret_key         secretkey123
# security_token     <none>

# 6. 生成STS临时凭证
vault read aws/sts/ec2-admin
# 返回临时访问密钥和session token
```

## 密钥轮转策略

### 自动密钥轮转

```bash
# 1. 配置数据库根凭证轮转
vault write database/config/postgresql \
  plugin_name=postgresql-database-plugin \
  allowed_roles="*" \
  connection_url="postgresql://{{username}}:{{password}}@postgres:5432/mydb" \
  username="vaultadmin" \
  password="initial-password" \
  rotate=true

# 立即轮转根凭证
vault write -f database/rotate-root/postgresql

# 2. 配置自动轮转策略
cat > rotation-policy.hcl <<EOF
path "database/rotate-root/postgresql" {
  schedule = "0 0 * * *"  # 每天午夜轮转
}
EOF

# 3. KV密钥版本管理
vault kv put secret/api-key value="key-v1"
vault kv put secret/api-key value="key-v2"  # 创建新版本
vault kv put secret/api-key value="key-v3"

# 配置最大版本数
vault kv metadata put -max-versions=5 secret/api-key

# 4. 密钥轮转通知
cat > rotate_notify.py <<'EOF'
"""
密钥轮转通知脚本
"""
import hvac
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta

class SecretRotationMonitor:
    def __init__(self, vault_client):
        self.vault = vault_client

    def check_expiring_secrets(self, days_threshold=7):
        """检查即将过期的密钥"""
        expiring = []

        # 检查所有动态密钥租约
        leases = self.vault.client.sys.list_leases('database/creds')

        for lease_id in leases['data']['keys']:
            lease_info = self.vault.client.sys.read_lease(lease_id)
            ttl = lease_info['data']['ttl']

            if ttl < days_threshold * 86400:  # 转换为秒
                expiring.append({
                    'lease_id': lease_id,
                    'ttl_days': ttl / 86400,
                    'expires_at': datetime.now() + timedelta(seconds=ttl)
                })

        return expiring

    def rotate_secret(self, path):
        """轮转密钥"""
        # 读取当前密钥
        current = self.vault.get_secret(path)

        # 生成新密钥
        new_value = self._generate_new_secret()

        # 写入新版本
        self.vault.put_secret(path, {'value': new_value})

        # 记录轮转事件
        self._log_rotation(path, current['value'], new_value)

        return new_value

    def _generate_new_secret(self):
        """生成新密钥"""
        import secrets
        return secrets.token_urlsafe(32)

    def _log_rotation(self, path, old_value, new_value):
        """记录轮转日志"""
        print(f"[{datetime.now()}] Rotated secret at {path}")

    def send_notification(self, expiring_secrets):
        """发送通知"""
        if not expiring_secrets:
            return

        message = "以下密钥即将过期:\n\n"
        for secret in expiring_secrets:
            message += f"- {secret['lease_id']}: {secret['ttl_days']:.1f} 天后过期\n"

        self._send_email(
            to='security@company.com',
            subject='Vault密钥过期提醒',
            body=message
        )

    def _send_email(self, to, subject, body):
        """发送邮件"""
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = 'vault@company.com'
        msg['To'] = to

        # 发送邮件逻辑
        pass

# 使用示例
vault = VaultClient()
monitor = SecretRotationMonitor(vault)

# 检查即将过期的密钥
expiring = monitor.check_expiring_secrets(days_threshold=7)

# 发送通知
if expiring:
    monitor.send_notification(expiring)

# 轮转密钥
monitor.rotate_secret('secret/api-key')
EOF
```

## KMS集成

### AWS KMS集成

```bash
# 1. 配置Vault使用AWS KMS自动解封
cat > vault-kms.hcl <<EOF
seal "awskms" {
  region     = "us-east-1"
  kms_key_id = "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012"
}

storage "dynamodb" {
  ha_enabled = "true"
  region     = "us-east-1"
  table      = "vault-storage"
}

listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = 0
  tls_cert_file = "/etc/vault/tls/vault.crt"
  tls_key_file  = "/etc/vault/tls/vault.key"
}
EOF

# 2. 启用Transit引擎 (加密即服务)
vault secrets enable transit

# 3. 创建加密密钥
vault write -f transit/keys/orders

# 4. 加密数据
vault write transit/encrypt/orders \
  plaintext=$(echo "sensitive data" | base64)
# 返回: ciphertext: vault:v1:abc123...

# 5. 解密数据
vault write transit/decrypt/orders \
  ciphertext="vault:v1:abc123..."
# 返回: plaintext (base64编码)

# 6. 轮转加密密钥
vault write -f transit/keys/orders/rotate

# 7. 重新包装数据 (使用新密钥)
vault write transit/rewrap/orders \
  ciphertext="vault:v1:abc123..."
# 返回: ciphertext: vault:v2:xyz789...

# 8. Python应用集成
cat > transit_client.py <<'EOF'
import hvac
import base64

class TransitEncryption:
    def __init__(self, vault_client, key_name='orders'):
        self.vault = vault_client
        self.key_name = key_name

    def encrypt(self, plaintext):
        """加密数据"""
        # Base64编码
        plaintext_b64 = base64.b64encode(plaintext.encode()).decode()

        # 加密
        response = self.vault.client.secrets.transit.encrypt_data(
            name=self.key_name,
            plaintext=plaintext_b64
        )

        return response['data']['ciphertext']

    def decrypt(self, ciphertext):
        """解密数据"""
        # 解密
        response = self.vault.client.secrets.transit.decrypt_data(
            name=self.key_name,
            ciphertext=ciphertext
        )

        # Base64解码
        plaintext_b64 = response['data']['plaintext']
        plaintext = base64.b64decode(plaintext_b64).decode()

        return plaintext

    def rotate_key(self):
        """轮转密钥"""
        self.vault.client.secrets.transit.rotate_encryption_key(
            name=self.key_name
        )

    def rewrap(self, ciphertext):
        """使用最新密钥重新加密"""
        response = self.vault.client.secrets.transit.rewrap_data(
            name=self.key_name,
            ciphertext=ciphertext
        )

        return response['data']['ciphertext']

# 使用示例
vault = VaultClient()
transit = TransitEncryption(vault)

# 加密
ciphertext = transit.encrypt("credit card: 1234-5678-9012-3456")
print(f"Encrypted: {ciphertext}")

# 解密
plaintext = transit.decrypt(ciphertext)
print(f"Decrypted: {plaintext}")

# 轮转密钥
transit.rotate_key()

# 重新包装
new_ciphertext = transit.rewrap(ciphertext)
EOF
```

### GCP KMS集成

```bash
# 1. 配置Vault使用GCP KMS
cat > vault-gcp-kms.hcl <<EOF
seal "gcpckms" {
  project     = "my-project"
  region      = "global"
  key_ring    = "vault-keyring"
  crypto_key  = "vault-key"
}

storage "gcs" {
  bucket     = "vault-storage-bucket"
  ha_enabled = "true"
}
EOF

# 2. 设置GCP凭证
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"

# 3. 启动Vault
vault server -config=vault-gcp-kms.hcl
```

### Azure Key Vault集成

```bash
# 1. 配置Vault使用Azure Key Vault
cat > vault-azure-kv.hcl <<EOF
seal "azurekeyvault" {
  tenant_id      = "00000000-0000-0000-0000-000000000000"
  client_id      = "00000000-0000-0000-0000-000000000000"
  client_secret  = "client-secret-here"
  vault_name     = "my-vault"
  key_name       = "vault-key"
}

storage "azure" {
  accountName = "vaultstorageaccount"
  accountKey  = "storage-account-key"
  container   = "vault"
}
EOF
```

## 实战案例

### 案例1: 微服务密钥管理

```yaml
# Kubernetes部署with Vault Agent Injector
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
spec:
  replicas: 3
  template:
    metadata:
      annotations:
        # 启用Vault Agent注入
        vault.hashicorp.com/agent-inject: "true"
        vault.hashicorp.com/role: "order-service"

        # 注入数据库凭证
        vault.hashicorp.com/agent-inject-secret-db: "database/creds/order-db"
        vault.hashicorp.com/agent-inject-template-db: |
          {{- with secret "database/creds/order-db" -}}
          DB_USER="{{ .Data.username }}"
          DB_PASS="{{ .Data.password }}"
          DB_HOST="postgres.production.svc.cluster.local"
          {{- end }}

        # 注入API密钥
        vault.hashicorp.com/agent-inject-secret-api: "secret/data/order-service/api-keys"
        vault.hashicorp.com/agent-inject-template-api: |
          {{- with secret "secret/data/order-service/api-keys" -}}
          STRIPE_API_KEY="{{ .Data.data.stripe_key }}"
          SENDGRID_API_KEY="{{ .Data.data.sendgrid_key }}"
          {{- end }}

        # 注入加密密钥
        vault.hashicorp.com/agent-inject-secret-encryption: "transit/keys/orders"

    spec:
      serviceAccountName: order-service
      containers:
      - name: app
        image: order-service:v1.0.0
        command: ["/bin/sh", "-c"]
        args:
        - |
          # 加载Vault密钥
          source /vault/secrets/db
          source /vault/secrets/api
          # 启动应用
          exec ./order-service
```

### 案例2: CI/CD密钥管理

```yaml
# GitLab CI使用Vault
.vault_auth:
  before_script:
    - export VAULT_ADDR=https://vault.company.com
    - export VAULT_TOKEN=$(vault write -field=token auth/jwt/login role=gitlab-ci jwt=$CI_JOB_JWT)

deploy:production:
  extends: .vault_auth
  script:
    # 从Vault获取密钥
    - vault kv get -field=aws_access_key secret/ci/aws > /tmp/aws_key
    - vault kv get -field=aws_secret_key secret/ci/aws > /tmp/aws_secret

    # 使用密钥部署
    - aws configure set aws_access_key_id $(cat /tmp/aws_key)
    - aws configure set aws_secret_access_key $(cat /tmp/aws_secret)
    - aws eks update-kubeconfig --name production-cluster

    # 部署应用
    - kubectl apply -f k8s/

    # 清理临时文件
    - rm /tmp/aws_*
  only:
    - main
```

## 总结

密钥管理最佳实践:

1. **集中管理**
   - 使用Vault/KMS等专业工具
   - 避免硬编码和明文存储
   - 统一密钥访问入口

2. **动态生成**
   - 数据库动态凭证
   - 云平台临时凭证
   - 自动过期和撤销

3. **定期轮转**
   - 自动轮转策略
   - 通知和监控
   - 版本管理

4. **最小权限**
   - 基于角色的访问控制
   - 最小权限原则
   - 定期审查权限

5. **审计日志**
   - 记录所有访问
   - 异常检测
   - 合规报告

6. **高可用**
   - 多节点集群
   - 自动解封 (KMS)
   - 备份恢复

核心工具:
- **HashiCorp Vault**: 最全面的密钥管理解决方案
- **AWS Secrets Manager**: AWS原生集成
- **GCP Secret Manager**: GCP原生集成
- **Azure Key Vault**: Azure原生集成
