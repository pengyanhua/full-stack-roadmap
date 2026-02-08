# API 网关统一认证授权

## 目录
- [认证机制概述](#认证机制概述)
- [JWT认证](#jwt认证)
- [OAuth2.0](#oauth20)
- [API Key](#api-key)
- [RBAC权限控制](#rbac权限控制)
- [SSO单点登录](#sso单点登录)
- [实战案例](#实战案例)

---

## 认证机制概述

### 认证 vs 授权

```
┌──────────────────────────────────────────────────────┐
│           认证(Authentication) vs 授权(Authorization) │
├──────────────────────────────────────────────────────┤
│                                                      │
│  认证 (Who are you?)                                 │
│  ┌────────────┐      ┌────────────┐                 │
│  │   用户     │ ────▶│  验证身份  │                 │
│  │ username   │      │  password  │                 │
│  │ password   │      │   JWT      │                 │
│  └────────────┘      └────────────┘                 │
│                           │                          │
│                           ▼                          │
│                      身份令牌 (Token)                │
│                                                      │
│  授权 (What can you do?)                             │
│  ┌────────────┐      ┌────────────┐                 │
│  │   令牌     │ ────▶│  检查权限  │                 │
│  │   Token    │      │   Scope    │                 │
│  │   Claims   │      │   Role     │                 │
│  └────────────┘      └────────────┘                 │
│                           │                          │
│                           ▼                          │
│                      允许/拒绝访问                   │
└──────────────────────────────────────────────────────┘
```

### 常见认证方案对比

```
┌────────────────────────────────────────────────────┐
│          认证方案对比                              │
├───────────┬──────────┬──────────┬─────────────────┤
│ 方案      │ 安全性   │ 性能     │  适用场景       │
├───────────┼──────────┼──────────┼─────────────────┤
│ API Key   │ ⭐⭐     │ ⭐⭐⭐⭐ │  简单API        │
│ JWT       │ ⭐⭐⭐   │ ⭐⭐⭐⭐ │  微服务         │
│ OAuth2    │ ⭐⭐⭐⭐ │ ⭐⭐⭐   │  第三方授权     │
│ mTLS      │ ⭐⭐⭐⭐⭐│ ⭐⭐     │  高安全需求     │
└───────────┴──────────┴──────────┴─────────────────┘
```

---

## JWT认证

### JWT结构

```
JWT = Header.Payload.Signature

┌──────────────────────────────────────────────┐
│            JWT 三部分结构                    │
├──────────────────────────────────────────────┤
│                                              │
│ Header (头部)                                │
│ {                                            │
│   "alg": "HS256",                            │
│   "typ": "JWT"                               │
│ }                                            │
│                                              │
│ Payload (载荷)                               │
│ {                                            │
│   "sub": "1234567890",                       │
│   "name": "John Doe",                        │
│   "iat": 1706745600,                         │
│   "exp": 1706749200,                         │
│   "roles": ["admin", "user"]                 │
│ }                                            │
│                                              │
│ Signature (签名)                             │
│ HMACSHA256(                                  │
│   base64UrlEncode(header) + "." +            │
│   base64UrlEncode(payload),                  │
│   secret                                     │
│ )                                            │
└──────────────────────────────────────────────┘
```

### Kong JWT配置

```yaml
# JWT插件配置
plugins:
  - name: jwt
    config:
      uri_param_names: [jwt]
      cookie_names: [jwt_token]
      header_names: [Authorization]
      claims_to_verify:
        - exp  # 验证过期时间
      maximum_expiration: 86400  # 最大24小时
      run_on_preflight: false

# 创建JWT消费者
consumers:
  - username: app-client
    custom_id: app-123
    jwt_secrets:
      - key: app-key-123
        secret: super-secret-key
        algorithm: HS256
```

### JWT验证实现

```python
# Python JWT验证
import jwt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

def generate_token(user_id, roles):
    """生成JWT令牌"""
    payload = {
        'user_id': user_id,
        'roles': roles,
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token):
    """验证JWT令牌"""
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={"verify_exp": True}
        )
        return payload
    except jwt.ExpiredSignatureError:
        return None, "Token expired"
    except jwt.InvalidTokenError:
        return None, "Invalid token"

def require_auth(roles=None):
    """认证装饰器"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # 获取token
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({'error': 'Missing token'}), 401

            token = auth_header.split(' ')[1]

            # 验证token
            payload, error = verify_token(token)
            if error:
                return jsonify({'error': error}), 401

            # 检查角色
            if roles and not any(role in payload.get('roles', []) for role in roles):
                return jsonify({'error': 'Insufficient permissions'}), 403

            # 将用户信息注入请求
            request.user = payload
            return f(*args, **kwargs)

        return decorated_function
    return decorator

# 使用示例
@app.route('/api/admin/users')
@require_auth(roles=['admin'])
def admin_users():
    return jsonify({'users': get_all_users()})
```

---

## OAuth2.0

### OAuth2授权流程

```
┌──────────────────────────────────────────────────────┐
│         OAuth2.0 授权码模式 (Authorization Code)      │
├──────────────────────────────────────────────────────┤
│                                                      │
│  1. 用户访问客户端                                   │
│  ┌──────┐                                            │
│  │ User │                                            │
│  └───┬──┘                                            │
│      │ 2. 重定向到授权服务器                         │
│      ▼                                                │
│  ┌─────────────┐                                     │
│  │  Auth Server│ 3. 用户登录并授权                   │
│  └──────┬──────┘                                     │
│         │ 4. 返回授权码 (code)                       │
│         ▼                                             │
│  ┌──────────┐                                        │
│  │  Client  │                                        │
│  └────┬─────┘                                        │
│       │ 5. 用授权码换取token                         │
│       ▼                                               │
│  ┌─────────────┐                                     │
│  │  Auth Server│ 6. 返回 access_token                │
│  └──────┬──────┘      + refresh_token                │
│         │                                             │
│         ▼ 7. 使用token访问API                        │
│  ┌─────────────┐                                     │
│  │ Resource API│                                     │
│  └─────────────┘                                     │
└──────────────────────────────────────────────────────┘
```

### Kong OAuth2配置

```yaml
# OAuth2插件
plugins:
  - name: oauth2
    config:
      # 启用的授权类型
      enable_authorization_code: true
      enable_client_credentials: true
      enable_implicit_grant: false
      enable_password_grant: true

      # Token配置
      token_expiration: 7200  # 2小时
      refresh_token_ttl: 1209600  # 14天

      # 作用域
      scopes:
        - read
        - write
        - admin
      mandatory_scope: true

      # 回调URL
      provision_key: provision-key-123

# 创建OAuth2应用
consumers:
  - username: my-app
    oauth2_credentials:
      - name: My Application
        client_id: app-client-id
        client_secret: app-client-secret
        redirect_uris:
          - https://app.example.com/callback
```

### OAuth2服务端实现

```python
# Python OAuth2 Server (使用 Authlib)
from authlib.integrations.flask_oauth2 import (
    AuthorizationServer,
    ResourceProtector
)
from authlib.oauth2.rfc6749 import grants

# 授权码模式
class AuthorizationCodeGrant(grants.AuthorizationCodeGrant):
    def save_authorization_code(self, code, request):
        client = request.client
        auth_code = AuthorizationCode(
            code=code,
            client_id=client.client_id,
            redirect_uri=request.redirect_uri,
            scope=request.scope,
            user_id=request.user.id,
        )
        db.session.add(auth_code)
        db.session.commit()
        return auth_code

    def query_authorization_code(self, code, client):
        return AuthorizationCode.query.filter_by(
            code=code,
            client_id=client.client_id
        ).first()

    def delete_authorization_code(self, authorization_code):
        db.session.delete(authorization_code)
        db.session.commit()

    def authenticate_user(self, authorization_code):
        return User.query.get(authorization_code.user_id)

# 初始化授权服务器
server = AuthorizationServer()
server.register_grant(AuthorizationCodeGrant)

# 授权端点
@app.route('/oauth/authorize', methods=['GET', 'POST'])
def authorize():
    if request.method == 'GET':
        # 显示授权页面
        grant = server.get_consent_grant(end_user=current_user)
        return render_template('authorize.html', grant=grant)
    else:
        # 用户同意授权
        return server.create_authorization_response(
            grant_user=current_user
        )

# Token端点
@app.route('/oauth/token', methods=['POST'])
def issue_token():
    return server.create_token_response()
```

---

## API Key

### API Key配置

```yaml
# Kong API Key插件
plugins:
  - name: key-auth
    config:
      key_names: [apikey, api-key, X-API-Key]
      key_in_query: true
      key_in_header: true
      key_in_body: false
      hide_credentials: true
      anonymous: false

# 创建API Key
consumers:
  - username: mobile-app
    keyauth_credentials:
      - key: mobile-app-key-123456

  - username: web-app
    keyauth_credentials:
      - key: web-app-key-789012
```

### API Key管理

```python
# Python API Key管理
import secrets
import hashlib
from datetime import datetime, timedelta

class APIKeyManager:
    def __init__(self, db):
        self.db = db

    def generate_key(self, user_id, name, expires_in_days=365):
        """生成API Key"""
        # 生成安全的随机key
        raw_key = secrets.token_urlsafe(32)

        # 存储hash,不存储明文
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        api_key = APIKey(
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days)
        )

        self.db.session.add(api_key)
        self.db.session.commit()

        # 只返回一次明文key
        return raw_key

    def verify_key(self, raw_key):
        """验证API Key"""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        api_key = APIKey.query.filter_by(
            key_hash=key_hash,
            is_active=True
        ).first()

        if not api_key:
            return None, "Invalid API key"

        # 检查过期
        if api_key.expires_at < datetime.utcnow():
            return None, "API key expired"

        # 更新最后使用时间
        api_key.last_used_at = datetime.utcnow()
        self.db.session.commit()

        return api_key.user, None

    def revoke_key(self, key_id):
        """吊销API Key"""
        api_key = APIKey.query.get(key_id)
        if api_key:
            api_key.is_active = False
            self.db.session.commit()
```

---

## RBAC权限控制

### RBAC模型

```
┌──────────────────────────────────────────────────────┐
│              RBAC (Role-Based Access Control)         │
├──────────────────────────────────────────────────────┤
│                                                      │
│  User (用户) ──▶ Role (角色) ──▶ Permission (权限)  │
│                                                      │
│  ┌─────────┐       ┌─────────┐      ┌──────────┐   │
│  │ Alice   │──────▶│  Admin  │─────▶│  Write   │   │
│  └─────────┘       └─────────┘      │  Read    │   │
│                                      │  Delete  │   │
│  ┌─────────┐       ┌─────────┐      └──────────┘   │
│  │  Bob    │──────▶│ Editor  │─────▶│  Write   │   │
│  └─────────┘       └─────────┘      │  Read    │   │
│                                      └──────────┘   │
│  ┌─────────┐       ┌─────────┐      ┌──────────┐   │
│  │Charlie  │──────▶│ Viewer  │─────▶│  Read    │   │
│  └─────────┘       └─────────┘      └──────────┘   │
│                                                      │
│  一个用户可以有多个角色                              │
│  一个角色可以有多个权限                              │
└──────────────────────────────────────────────────────┘
```

### Kong ACL配置

```yaml
# ACL插件配置
plugins:
  - name: acl
    config:
      allow: [admin, premium]  # 白名单
      deny: [banned]  # 黑名单
      hide_groups_header: false

# 为消费者分配组
consumers:
  - username: alice
    acls:
      - group: admin
      - group: premium

  - username: bob
    acls:
      - group: premium

# 路由级别的ACL
routes:
  - name: admin-api
    paths: [/api/admin]
    plugins:
      - name: acl
        config:
          allow: [admin]
```

### 自定义RBAC实现

```python
# Python RBAC实现
from functools import wraps
from flask import request, jsonify

# 权限定义
PERMISSIONS = {
    'admin': ['read', 'write', 'delete', 'admin'],
    'editor': ['read', 'write'],
    'viewer': ['read']
}

def require_permission(permission):
    """权限检查装饰器"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # 从JWT获取用户角色
            user = request.user
            roles = user.get('roles', [])

            # 检查是否有权限
            has_permission = False
            for role in roles:
                if permission in PERMISSIONS.get(role, []):
                    has_permission = True
                    break

            if not has_permission:
                return jsonify({
                    'error': 'Insufficient permissions',
                    'required': permission
                }), 403

            return f(*args, **kwargs)
        return decorated_function
    return decorator

# 使用示例
@app.route('/api/users', methods=['DELETE'])
@require_auth()
@require_permission('delete')
def delete_user():
    return jsonify({'message': 'User deleted'})
```

---

## SSO单点登录

### SAML SSO流程

```
┌──────────────────────────────────────────────────────┐
│              SAML SSO 流程                           │
├──────────────────────────────────────────────────────┤
│                                                      │
│  1. 用户访问应用                                     │
│  ┌──────┐                                            │
│  │ User │ ─────▶ App (未登录)                        │
│  └──────┘                                            │
│                │                                     │
│                ▼ 2. 重定向到IdP                      │
│         ┌─────────────┐                              │
│         │     IdP     │                              │
│         │ (Okta/Auth0)│                              │
│         └──────┬──────┘                              │
│                │ 3. 用户登录                          │
│                │                                     │
│                ▼ 4. 返回SAML断言                     │
│         ┌──────────┐                                 │
│         │   App    │                                 │
│         └────┬─────┘                                 │
│              │ 5. 验证SAML断言                       │
│              │                                       │
│              ▼ 6. 创建会话                           │
│         ┌──────────┐                                 │
│         │  User    │                                 │
│         │(已登录)  │                                 │
│         └──────────┘                                 │
└──────────────────────────────────────────────────────┘
```

### Kong OIDC配置

```yaml
# OpenID Connect插件
plugins:
  - name: openid-connect
    config:
      # IdP配置
      issuer: https://accounts.google.com
      client_id: your-client-id.apps.googleusercontent.com
      client_secret: your-client-secret

      # 重定向URI
      redirect_uri: https://api.example.com/auth/callback

      # 作用域
      scopes:
        - openid
        - email
        - profile

      # Token配置
      token_endpoint_auth_method: client_secret_post
      session_secret: session-secret-key

      # 声明映射
      authenticated_groups_claim: groups
```

---

## 实战案例

### 案例: 多租户认证架构

```yaml
# 多租户API网关认证
services:
  # 租户A - JWT认证
  - name: tenant-a-api
    url: http://tenant-a-service
    routes:
      - name: tenant-a
        hosts: [a.api.example.com]
    plugins:
      - name: jwt
        config:
          key_claim_name: tid
          claims_to_verify: [exp, tid]
          secret_is_base64: false

  # 租户B - OAuth2认证
  - name: tenant-b-api
    url: http://tenant-b-service
    routes:
      - name: tenant-b
        hosts: [b.api.example.com]
    plugins:
      - name: oauth2
        config:
          scopes: [read, write]
          mandatory_scope: true

  # 公共API - API Key认证
  - name: public-api
    url: http://public-service
    routes:
      - name: public
        paths: [/api/public]
    plugins:
      - name: key-auth
      - name: rate-limiting
        config:
          minute: 100
```

---

## 总结

### 认证方案选择

```
┌────────────────────────────────────────────────┐
│          认证方案选择指南                      │
├────────────────────────────────────────────────┤
│                                                │
│  内部API?                                      │
│    └─ 是 → JWT + mTLS                         │
│                                                │
│  第三方集成?                                   │
│    └─ 是 → OAuth2.0                           │
│                                                │
│  简单认证?                                     │
│    └─ 是 → API Key                            │
│                                                │
│  企业SSO?                                      │
│    └─ 是 → OIDC / SAML                        │
└────────────────────────────────────────────────┘
```

### 安全最佳实践

1. **永远使用HTTPS**: 传输加密
2. **Token过期**: 设置合理过期时间
3. **刷新令牌**: 实现令牌刷新机制
4. **权限最小化**: 按需分配权限
5. **审计日志**: 记录所有认证事件

### 下一步学习
- [04_gateway_comparison.md](04_gateway_comparison.md) - 网关技术对比
