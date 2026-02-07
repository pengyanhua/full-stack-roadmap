# API 网关认证授权

## 目录
- [认证授权概述](#认证授权概述)
- [JWT 认证](#jwt-认证)
- [OAuth2 实现](#oauth2-实现)
- [RBAC 权限控制](#rbac-权限控制)
- [多租户认证](#多租户认证)
- [实战案例](#实战案例)

---

## 认证授权概述

### 认证 vs 授权

```
┌─────────────────────────────────────────────┐
│        认证(Authentication) vs              │
│        授权(Authorization)                   │
├─────────────────────────────────────────────┤
│                                             │
│  认证 (Authentication)                      │
│  ┌─────────────────────────────────────┐   │
│  │  "你是谁?"                          │   │
│  │  ├─ 验证用户身份                    │   │
│  │  ├─ 用户名/密码                     │   │
│  │  ├─ Token (JWT)                     │   │
│  │  ├─ OAuth2                          │   │
│  │  └─ 多因素认证 (MFA)                │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  授权 (Authorization)                       │
│  ┌─────────────────────────────────────┐   │
│  │  "你能做什么?"                      │   │
│  │  ├─ 验证访问权限                    │   │
│  │  ├─ RBAC (基于角色)                 │   │
│  │  ├─ ABAC (基于属性)                 │   │
│  │  ├─ ACL (访问控制列表)              │   │
│  │  └─ Scope (OAuth2)                  │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### 完整认证流程

```
      API 网关认证授权流程
┌─────────────────────────────────────┐
│  1. 客户端请求                      │
│     POST /api/v1/users              │
│     Authorization: Bearer <token>   │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  2. API Gateway 接收                 │
│     提取 Token                       │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  3. Token 验证                       │
│     ├─ 签名验证                      │
│     ├─ 过期检查                      │
│     └─ 格式校验                      │
└──────────────┬───────────────────────┘
               │
          ┌────┴────┐
          │         │
        失败       成功
          │         │
          │         ▼
          │    ┌────────────────────────┐
          │    │  4. 提取用户信息       │
          │    │     user_id, roles等   │
          │    └────┬───────────────────┘
          │         │
          │         ▼
          │    ┌────────────────────────┐
          │    │  5. 权限检查 (RBAC)    │
          │    │     检查 roles/scopes  │
          │    └────┬───────────────────┘
          │         │
          │    ┌────┴────┐
          │    │         │
          │   拒绝      允许
          │    │         │
          ▼    ▼         ▼
┌────────────────┐  ┌───────────────────┐
│  6. 返回错误   │  │  7. 转发到后端    │
│     401/403    │  │     添加 Headers  │
└────────────────┘  └───────────────────┘
```

---

## JWT 认证

### JWT 结构

```
┌──────────────────────────────────────────┐
│            JWT Token 结构                │
├──────────────────────────────────────────┤
│                                          │
│  Header.Payload.Signature                │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Header (Base64)                   │ │
│  │  {                                 │ │
│  │    "alg": "HS256",                 │ │
│  │    "typ": "JWT"                    │ │
│  │  }                                 │ │
│  └────────────────────────────────────┘ │
│                   +                      │
│  ┌────────────────────────────────────┐ │
│  │  Payload (Base64)                  │ │
│  │  {                                 │ │
│  │    "sub": "user123",               │ │
│  │    "name": "John Doe",             │ │
│  │    "iat": 1516239022,              │ │
│  │    "exp": 1516242622,              │ │
│  │    "roles": ["admin", "user"]      │ │
│  │  }                                 │ │
│  └────────────────────────────────────┘ │
│                   +                      │
│  ┌────────────────────────────────────┐ │
│  │  Signature                         │ │
│  │  HMACSHA256(                       │ │
│  │    base64(header) + "." +          │ │
│  │    base64(payload),                │ │
│  │    secret                          │ │
│  │  )                                 │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

### JWT 生成与验证 (Python)

```python
# jwt_handler.py - JWT 生成与验证
import jwt
import datetime
from typing import Dict, Optional
from functools import wraps
from flask import request, jsonify

class JWTHandler:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm

    def generate_token(
        self,
        user_id: str,
        username: str,
        roles: list,
        expires_in: int = 3600
    ) -> str:
        """生成 JWT Token"""
        now = datetime.datetime.utcnow()

        payload = {
            # 标准声明
            "iss": "api-gateway",           # 签发者
            "sub": user_id,                 # 主题 (用户ID)
            "aud": "api",                   # 接收方
            "iat": now,                     # 签发时间
            "exp": now + datetime.timedelta(seconds=expires_in),  # 过期时间
            "nbf": now,                     # 生效时间

            # 自定义声明
            "username": username,
            "roles": roles,
            "permissions": self._get_permissions(roles)
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token

    def verify_token(self, token: str) -> Optional[Dict]:
        """验证 JWT Token"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                audience="api",
                issuer="api-gateway"
            )
            return payload

        except jwt.ExpiredSignatureError:
            return {"error": "Token has expired"}
        except jwt.InvalidTokenError as e:
            return {"error": f"Invalid token: {str(e)}"}

    def _get_permissions(self, roles: list) -> list:
        """根据角色获取权限"""
        permission_map = {
            "admin": ["read", "write", "delete", "manage"],
            "user": ["read", "write"],
            "guest": ["read"]
        }

        permissions = set()
        for role in roles:
            permissions.update(permission_map.get(role, []))

        return list(permissions)

    def refresh_token(self, old_token: str) -> Optional[str]:
        """刷新 Token"""
        payload = self.verify_token(old_token)

        if "error" in payload:
            return None

        # 生成新 Token
        return self.generate_token(
            user_id=payload["sub"],
            username=payload["username"],
            roles=payload["roles"]
        )


# Flask 装饰器
def require_auth(roles: list = None):
    """认证装饰器"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_header = request.headers.get("Authorization")

            if not auth_header or not auth_header.startswith("Bearer "):
                return jsonify({"error": "Missing or invalid token"}), 401

            token = auth_header.split(" ")[1]
            jwt_handler = JWTHandler("your-secret-key")

            payload = jwt_handler.verify_token(token)

            if "error" in payload:
                return jsonify(payload), 401

            # 角色检查
            if roles:
                user_roles = payload.get("roles", [])
                if not any(role in user_roles for role in roles):
                    return jsonify({"error": "Insufficient permissions"}), 403

            # 将用户信息添加到请求上下文
            request.user = payload

            return f(*args, **kwargs)

        return decorated_function
    return decorator


# 使用示例
from flask import Flask

app = Flask(__name__)
jwt_handler = JWTHandler("your-secret-key")

@app.route("/api/login", methods=["POST"])
def login():
    """登录接口"""
    data = request.json
    username = data.get("username")
    password = data.get("password")

    # 验证用户名密码 (示例)
    if username == "admin" and password == "admin123":
        token = jwt_handler.generate_token(
            user_id="user_001",
            username=username,
            roles=["admin", "user"]
        )

        return jsonify({
            "access_token": token,
            "token_type": "Bearer",
            "expires_in": 3600
        })

    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/api/v1/users", methods=["GET"])
@require_auth(roles=["admin", "user"])
def get_users():
    """获取用户列表 (需要认证)"""
    user_info = request.user

    return jsonify({
        "current_user": user_info["username"],
        "roles": user_info["roles"],
        "users": [
            {"id": 1, "name": "User 1"},
            {"id": 2, "name": "User 2"}
        ]
    })


@app.route("/api/v1/admin", methods=["POST"])
@require_auth(roles=["admin"])
def admin_only():
    """管理员接口"""
    return jsonify({"message": "Admin access granted"})


if __name__ == "__main__":
    app.run(port=8080)
```

### Kong JWT 插件配置

```yaml
# Kong JWT 配置
plugins:
  - name: jwt
    config:
      uri_param_names:
        - jwt
      cookie_names:
        - jwt
      header_names:
        - Authorization
      claims_to_verify:
        - exp
        - nbf
      key_claim_name: iss
      secret_is_base64: false
      anonymous: null
      run_on_preflight: true
      maximum_expiration: 7200

consumers:
  - username: api-client-1
    custom_id: client_001

    jwt_secrets:
      # HS256 (对称加密)
      - key: api-gateway
        algorithm: HS256
        secret: your-256-bit-secret

  - username: api-client-2

    jwt_secrets:
      # RS256 (非对称加密)
      - key: api-gateway-rsa
        algorithm: RS256
        rsa_public_key: |
          -----BEGIN PUBLIC KEY-----
          MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA1Z...
          -----END PUBLIC KEY-----
```

### APISIX JWT 插件

```bash
#!/bin/bash
# APISIX JWT 配置

ADMIN_API="http://localhost:9180"
API_KEY="your-api-key"

# 1. 创建 Consumer
curl -X PUT "${ADMIN_API}/apisix/admin/consumers" \
  -H "X-API-KEY: ${API_KEY}" \
  -d '{
    "username": "mobile-app",
    "desc": "移动端应用",
    "plugins": {
      "jwt-auth": {
        "key": "mobile-app-key",
        "secret": "my-secret-key",
        "algorithm": "HS256",
        "exp": 86400,
        "base64_secret": false
      }
    }
  }'

# 2. 在路由上启用 JWT
curl -X PUT "${ADMIN_API}/apisix/admin/routes/1" \
  -H "X-API-KEY: ${API_KEY}" \
  -d '{
    "uri": "/api/v1/users/*",
    "plugins": {
      "jwt-auth": {
        "header": "Authorization",
        "query": "jwt",
        "cookie": "jwt"
      }
    },
    "upstream": {
      "type": "roundrobin",
      "nodes": {
        "user-service:8080": 1
      }
    }
  }'

# 3. 获取 Token
curl -X GET "http://localhost:9080/apisix/plugin/jwt/sign?key=mobile-app-key" \
  -H "Content-Type: application/json"
```

---

## OAuth2 实现

### OAuth2 授权流程

```
        OAuth2 授权码模式流程
┌─────────────────────────────────────────┐
│  1. 用户访问客户端应用                  │
│     User -> Client App                  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  2. 重定向到授权服务器                   │
│     GET /oauth/authorize?                │
│       client_id=xxx&                     │
│       redirect_uri=xxx&                  │
│       response_type=code&                │
│       scope=read+write                   │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  3. 用户登录并授权                       │
│     Authorization Server                 │
│     输入用户名密码,选择授权范围          │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  4. 返回授权码                           │
│     302 Redirect                         │
│     Location: callback?code=AUTH_CODE    │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  5. 使用授权码换取 Token                 │
│     POST /oauth/token                    │
│     {                                    │
│       "grant_type": "authorization_code",│
│       "code": "AUTH_CODE",               │
│       "client_id": "xxx",                │
│       "client_secret": "xxx",            │
│       "redirect_uri": "xxx"              │
│     }                                    │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  6. 返回访问令牌                         │
│     {                                    │
│       "access_token": "ACCESS_TOKEN",    │
│       "token_type": "Bearer",            │
│       "expires_in": 3600,                │
│       "refresh_token": "REFRESH_TOKEN",  │
│       "scope": "read write"              │
│     }                                    │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  7. 使用 Access Token 访问 API           │
│     GET /api/v1/users                    │
│     Authorization: Bearer ACCESS_TOKEN   │
└──────────────────────────────────────────┘
```

### OAuth2 服务器实现 (Python)

```python
# oauth2_server.py - OAuth2 授权服务器
from flask import Flask, request, jsonify, redirect
from authlib.integrations.flask_oauth2 import AuthorizationServer, ResourceProtector
from authlib.integrations.sqla_oauth2 import (
    create_query_client_func,
    create_save_token_func,
    create_bearer_token_validator
)
from authlib.oauth2.rfc6749 import grants
import secrets
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# 简化示例,实际应使用数据库
class Client:
    def __init__(self, client_id, client_secret, redirect_uris, scopes):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uris = redirect_uris
        self.scope = scopes
        self.grant_types = ['authorization_code', 'refresh_token']
        self.response_types = ['code']
        self.token_endpoint_auth_method = 'client_secret_post'

    def check_redirect_uri(self, redirect_uri):
        return redirect_uri in self.redirect_uris

    def check_client_secret(self, client_secret):
        return self.client_secret == client_secret

    def check_response_type(self, response_type):
        return response_type in self.response_types

    def check_grant_type(self, grant_type):
        return grant_type in self.grant_types

# 存储客户端和令牌
clients = {
    "mobile-app": Client(
        client_id="mobile-app",
        client_secret="mobile-secret",
        redirect_uris=["https://app.example.com/callback"],
        scopes="read write"
    )
}

authorization_codes = {}
access_tokens = {}
refresh_tokens = {}

# 授权码授权
class AuthorizationCodeGrant(grants.AuthorizationCodeGrant):
    def save_authorization_code(self, code, request):
        """保存授权码"""
        authorization_codes[code] = {
            "client_id": request.client.client_id,
            "redirect_uri": request.redirect_uri,
            "scope": request.scope,
            "user_id": request.user.id if hasattr(request, 'user') else "user_001",
            "code_challenge": request.data.get("code_challenge"),
            "code_challenge_method": request.data.get("code_challenge_method"),
            "expires_at": time.time() + 600  # 10分钟过期
        }

    def query_authorization_code(self, code, client):
        """查询授权码"""
        item = authorization_codes.get(code)
        if item and item["client_id"] == client.client_id:
            return item
        return None

    def delete_authorization_code(self, authorization_code):
        """删除授权码"""
        for code, item in list(authorization_codes.items()):
            if item == authorization_code:
                del authorization_codes[code]
                break

    def authenticate_user(self, authorization_code):
        """认证用户"""
        class User:
            def __init__(self, user_id):
                self.id = user_id

        return User(authorization_code.get("user_id"))

def query_client(client_id):
    return clients.get(client_id)

def save_token(token, request):
    """保存访问令牌"""
    access_token = token["access_token"]
    access_tokens[access_token] = {
        "client_id": request.client.client_id,
        "user_id": request.user.id if hasattr(request, 'user') else "user_001",
        "scope": token.get("scope"),
        "expires_at": time.time() + token["expires_in"]
    }

    if "refresh_token" in token:
        refresh_tokens[token["refresh_token"]] = {
            "client_id": request.client.client_id,
            "user_id": request.user.id if hasattr(request, 'user') else "user_001",
            "scope": token.get("scope")
        }

# 初始化授权服务器
server = AuthorizationServer()
server.register_grant(AuthorizationCodeGrant)

@app.route("/oauth/authorize", methods=["GET", "POST"])
def authorize():
    """授权端点"""
    if request.method == "GET":
        # 显示授权页面
        client_id = request.args.get("client_id")
        scope = request.args.get("scope")

        return f"""
        <h2>授权请求</h2>
        <p>应用 {client_id} 请求以下权限:</p>
        <ul>
            <li>{scope}</li>
        </ul>
        <form method="POST">
            <input type="hidden" name="client_id" value="{client_id}">
            <input type="hidden" name="scope" value="{scope}">
            <button type="submit" name="confirm" value="yes">同意</button>
            <button type="submit" name="confirm" value="no">拒绝</button>
        </form>
        """

    # 用户确认授权
    if request.form.get("confirm") == "yes":
        # 生成授权码
        code = secrets.token_urlsafe(32)
        client_id = request.form.get("client_id")
        redirect_uri = request.args.get("redirect_uri")

        grant = AuthorizationCodeGrant()
        grant.save_authorization_code(code, type('obj', (object,), {
            'client': clients[client_id],
            'redirect_uri': redirect_uri,
            'scope': request.form.get("scope"),
            'data': request.args
        }))

        return redirect(f"{redirect_uri}?code={code}")

    return jsonify({"error": "access_denied"}), 403


@app.route("/oauth/token", methods=["POST"])
def issue_token():
    """令牌端点"""
    grant_type = request.form.get("grant_type")

    if grant_type == "authorization_code":
        code = request.form.get("code")
        client_id = request.form.get("client_id")
        client_secret = request.form.get("client_secret")

        # 验证客户端
        client = clients.get(client_id)
        if not client or not client.check_client_secret(client_secret):
            return jsonify({"error": "invalid_client"}), 401

        # 验证授权码
        auth_code = authorization_codes.get(code)
        if not auth_code or auth_code["client_id"] != client_id:
            return jsonify({"error": "invalid_grant"}), 400

        # 检查过期
        if auth_code["expires_at"] < time.time():
            return jsonify({"error": "expired_token"}), 400

        # 生成访问令牌
        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)

        token_data = {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": refresh_token,
            "scope": auth_code["scope"]
        }

        save_token(token_data, type('obj', (object,), {
            'client': client,
            'user': type('user', (object,), {'id': auth_code["user_id"]})
        }))

        # 删除授权码
        del authorization_codes[code]

        return jsonify(token_data)

    elif grant_type == "refresh_token":
        refresh_token = request.form.get("refresh_token")
        client_id = request.form.get("client_id")

        # 验证刷新令牌
        token_info = refresh_tokens.get(refresh_token)
        if not token_info or token_info["client_id"] != client_id:
            return jsonify({"error": "invalid_grant"}), 400

        # 生成新的访问令牌
        new_access_token = secrets.token_urlsafe(32)

        token_data = {
            "access_token": new_access_token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": token_info["scope"]
        }

        access_tokens[new_access_token] = {
            "client_id": client_id,
            "user_id": token_info["user_id"],
            "scope": token_info["scope"],
            "expires_at": time.time() + 3600
        }

        return jsonify(token_data)

    return jsonify({"error": "unsupported_grant_type"}), 400


@app.route("/api/userinfo", methods=["GET"])
def userinfo():
    """用户信息端点"""
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "invalid_token"}), 401

    access_token = auth_header.split(" ")[1]
    token_info = access_tokens.get(access_token)

    if not token_info:
        return jsonify({"error": "invalid_token"}), 401

    if token_info["expires_at"] < time.time():
        return jsonify({"error": "expired_token"}), 401

    return jsonify({
        "sub": token_info["user_id"],
        "name": "John Doe",
        "email": "john@example.com",
        "scope": token_info["scope"]
    })


if __name__ == "__main__":
    app.run(port=9000)
```

---

## RBAC 权限控制

### RBAC 模型

```
        RBAC (Role-Based Access Control)
┌──────────────────────────────────────────┐
│                                          │
│  User ───▶ Role ───▶ Permission         │
│   │         │          │                 │
│   │         │          │                 │
│  用户     角色       权限                │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  User: john@example.com            │ │
│  │  ├─ Role: admin                    │ │
│  │  └─ Role: user                     │ │
│  └────────────────────────────────────┘ │
│                │                         │
│                ▼                         │
│  ┌────────────────────────────────────┐ │
│  │  Role: admin                       │ │
│  │  ├─ Permission: user:read          │ │
│  │  ├─ Permission: user:write         │ │
│  │  ├─ Permission: user:delete        │ │
│  │  └─ Permission: system:manage      │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Role: user                        │ │
│  │  ├─ Permission: user:read          │ │
│  │  └─ Permission: user:write         │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

### RBAC 实现 (Go)

```go
// rbac.go - RBAC 权限系统
package main

import (
    "fmt"
    "strings"
)

// Permission 权限
type Permission struct {
    Resource string // 资源: user, order, product
    Action   string // 操作: read, write, delete
}

func (p Permission) String() string {
    return fmt.Sprintf("%s:%s", p.Resource, p.Action)
}

// Role 角色
type Role struct {
    Name        string
    Permissions []Permission
}

// User 用户
type User struct {
    ID    string
    Name  string
    Email string
    Roles []Role
}

// RBAC 权限控制器
type RBAC struct {
    roles map[string]*Role
    users map[string]*User
}

func NewRBAC() *RBAC {
    return &RBAC{
        roles: make(map[string]*Role),
        users: make(map[string]*User),
    }
}

// 添加角色
func (r *RBAC) AddRole(name string, permissions []Permission) {
    r.roles[name] = &Role{
        Name:        name,
        Permissions: permissions,
    }
}

// 添加用户
func (r *RBAC) AddUser(user *User) {
    r.users[user.ID] = user
}

// 给用户分配角色
func (r *RBAC) AssignRole(userID, roleName string) error {
    user, ok := r.users[userID]
    if !ok {
        return fmt.Errorf("user not found: %s", userID)
    }

    role, ok := r.roles[roleName]
    if !ok {
        return fmt.Errorf("role not found: %s", roleName)
    }

    user.Roles = append(user.Roles, *role)
    return nil
}

// 检查用户权限
func (r *RBAC) CheckPermission(userID string, required Permission) bool {
    user, ok := r.users[userID]
    if !ok {
        return false
    }

    for _, role := range user.Roles {
        for _, perm := range role.Permissions {
            if perm.Resource == required.Resource &&
               (perm.Action == required.Action || perm.Action == "*") {
                return true
            }
        }
    }

    return false
}

// 获取用户所有权限
func (r *RBAC) GetUserPermissions(userID string) []Permission {
    user, ok := r.users[userID]
    if !ok {
        return nil
    }

    permMap := make(map[string]Permission)

    for _, role := range user.Roles {
        for _, perm := range role.Permissions {
            key := perm.String()
            permMap[key] = perm
        }
    }

    permissions := make([]Permission, 0, len(permMap))
    for _, perm := range permMap {
        permissions = append(permissions, perm)
    }

    return permissions
}

func main() {
    rbac := NewRBAC()

    // 定义角色和权限
    rbac.AddRole("admin", []Permission{
        {Resource: "user", Action: "*"},
        {Resource: "order", Action: "*"},
        {Resource: "product", Action: "*"},
        {Resource: "system", Action: "manage"},
    })

    rbac.AddRole("user", []Permission{
        {Resource: "user", Action: "read"},
        {Resource: "user", Action: "write"},
        {Resource: "order", Action: "read"},
        {Resource: "order", Action: "write"},
    })

    rbac.AddRole("guest", []Permission{
        {Resource: "product", Action: "read"},
    })

    // 创建用户
    user1 := &User{
        ID:    "user_001",
        Name:  "Admin User",
        Email: "admin@example.com",
    }

    rbac.AddUser(user1)
    rbac.AssignRole("user_001", "admin")

    // 权限检查
    canDelete := rbac.CheckPermission("user_001", Permission{
        Resource: "user",
        Action:   "delete",
    })

    fmt.Printf("Admin can delete user: %v\n", canDelete)

    // 获取所有权限
    permissions := rbac.GetUserPermissions("user_001")
    fmt.Println("User permissions:")
    for _, perm := range permissions {
        fmt.Printf("  - %s\n", perm)
    }
}
```

### Kong RBAC 插件

```yaml
# Kong ACL (Access Control List) 配置
plugins:
  - name: acl
    config:
      allow:
        - admin-group
        - premium-users
      deny:
        - blacklist
      hide_groups_header: false

consumers:
  - username: admin-user
    acls:
      - group: admin-group

  - username: premium-user
    acls:
      - group: premium-users

  - username: blocked-user
    acls:
      - group: blacklist
```

---

## 多租户认证

### 多租户架构

```
       多租户认证隔离模型
┌────────────────────────────────────┐
│  Tenant A (租户 A)                 │
│  ┌──────────────────────────────┐ │
│  │  Users:                      │ │
│  │  ├─ user1@tenant-a.com       │ │
│  │  └─ user2@tenant-a.com       │ │
│  │                              │ │
│  │  Resources:                  │ │
│  │  ├─ Database: tenant_a_db    │ │
│  │  └─ Storage: tenant-a-bucket │ │
│  └──────────────────────────────┘ │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│  Tenant B (租户 B)                 │
│  ┌──────────────────────────────┐ │
│  │  Users:                      │ │
│  │  ├─ user1@tenant-b.com       │ │
│  │  └─ user2@tenant-b.com       │ │
│  │                              │ │
│  │  Resources:                  │ │
│  │  ├─ Database: tenant_b_db    │ │
│  │  └─ Storage: tenant-b-bucket │ │
│  └──────────────────────────────┘ │
└────────────────────────────────────┘

数据隔离策略:
1. 数据库隔离 (每个租户独立数据库)
2. Schema 隔离 (共享数据库,独立 Schema)
3. 行级隔离 (共享表,tenant_id 区分)
```

### 多租户认证实现

```python
# multi_tenant_auth.py
from flask import Flask, request, jsonify
import jwt
from functools import wraps

app = Flask(__name__)

# 租户配置
TENANTS = {
    "tenant-a": {
        "name": "Tenant A",
        "secret_key": "tenant-a-secret",
        "database": "tenant_a_db",
        "features": ["feature1", "feature2"]
    },
    "tenant-b": {
        "name": "Tenant B",
        "secret_key": "tenant-b-secret",
        "database": "tenant_b_db",
        "features": ["feature1", "feature3"]
    }
}

def extract_tenant_id(request):
    """从请求中提取租户 ID"""
    # 方式 1: 从子域名提取
    host = request.host
    if "." in host:
        subdomain = host.split(".")[0]
        if subdomain in TENANTS:
            return subdomain

    # 方式 2: 从 Header 提取
    tenant_id = request.headers.get("X-Tenant-ID")
    if tenant_id and tenant_id in TENANTS:
        return tenant_id

    # 方式 3: 从 JWT 提取
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            # 先用通用密钥解码获取租户 ID
            payload = jwt.decode(token, options={"verify_signature": False})
            tenant_id = payload.get("tenant_id")
            if tenant_id in TENANTS:
                return tenant_id
        except:
            pass

    return None

def require_tenant_auth(f):
    """多租户认证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        tenant_id = extract_tenant_id(request)

        if not tenant_id:
            return jsonify({"error": "Tenant not identified"}), 400

        tenant_config = TENANTS[tenant_id]

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing token"}), 401

        token = auth_header.split(" ")[1]

        try:
            # 使用租户专属密钥验证
            payload = jwt.decode(
                token,
                tenant_config["secret_key"],
                algorithms=["HS256"]
            )

            # 验证租户 ID
            if payload.get("tenant_id") != tenant_id:
                return jsonify({"error": "Invalid tenant"}), 403

            # 将租户信息添加到请求上下文
            request.tenant_id = tenant_id
            request.tenant_config = tenant_config
            request.user = payload

            return f(*args, **kwargs)

        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

    return decorated_function

@app.route("/api/v1/data", methods=["GET"])
@require_tenant_auth
def get_data():
    """获取租户数据"""
    tenant_id = request.tenant_id
    database = request.tenant_config["database"]

    # 从租户专属数据库获取数据
    # db = get_database(database)
    # data = db.query(...)

    return jsonify({
        "tenant_id": tenant_id,
        "database": database,
        "user": request.user["username"],
        "data": ["item1", "item2"]
    })

if __name__ == "__main__":
    app.run(port=8080)
```

---

## 实战案例

### 案例: 企业级 API 网关认证系统

```yaml
# enterprise_auth_architecture.yml
# 企业级认证架构

architecture:
  # Layer 1: 外部认证
  external_auth:
    - oauth2_providers:
        - Google OAuth2
        - Microsoft Azure AD
        - GitHub OAuth
    - saml_providers:
        - Okta
        - Auth0

  # Layer 2: API 网关认证
  api_gateway:
    - jwt_validation:
        issuer: "https://auth.company.com"
        audience: "api.company.com"
        algorithms: ["RS256"]

    - rate_limiting:
        per_user: 1000/hour
        per_ip: 100/minute

    - ip_whitelisting:
        internal_network: "10.0.0.0/8"
        partner_ips: ["203.0.113.0/24"]

  # Layer 3: 服务间认证
  service_mesh:
    - mtls:
        enabled: true
        cert_rotation: 24h

    - service_accounts:
        - name: "user-service"
          permissions: ["user:read", "user:write"]
        - name: "order-service"
          permissions: ["order:*", "user:read"]

monitoring:
  - failed_auth_attempts:
      threshold: 5/minute
      action: block_ip

  - token_usage:
      track_by: user_id
      alert_on: anomaly

compliance:
  - audit_logs:
      retention: 90_days
      include: ["auth_attempts", "token_issued", "token_revoked"]

  - gdpr:
      data_encryption: true
      right_to_be_forgotten: true
```

---

## 总结

### 认证方案选择

```
┌──────────────────────────────────────┐
│      认证方案选择指南                │
├──────────────┬───────────────────────┤
│  场景        │  推荐方案             │
├──────────────┼───────────────────────┤
│ 内部 API     │ JWT (HS256)           │
│ 公开 API     │ JWT (RS256) + OAuth2  │
│ 第三方集成   │ OAuth2                │
│ 移动 APP     │ JWT + Refresh Token   │
│ 微服务间     │ mTLS + JWT            │
│ 企业应用     │ SAML/OIDC             │
└──────────────┴───────────────────────┘
```

### 关键要点

1. **JWT**: 无状态、跨域友好、适合分布式系统
2. **OAuth2**: 标准化授权协议、支持第三方登录
3. **RBAC**: 基于角色的权限控制、易于管理
4. **多租户**: 数据隔离、租户专属配置
5. **安全性**: HTTPS、Token 过期、刷新机制

### 下一步学习

- [04_gateway_comparison.md](04_gateway_comparison.md) - 网关产品对比
