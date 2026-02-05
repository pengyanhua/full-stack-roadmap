# API 设计规范

## 一、RESTful API 设计

### 1. 资源命名

```
┌─────────────────────────────────────────────────────────────────┐
│                   RESTful 资源命名规范                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ✅ 正确                             ❌ 错误                    │
│   ─────────────────────────────────────────────────────────    │
│   GET /users                          GET /getUsers             │
│   GET /users/123                      GET /user?id=123          │
│   POST /users                         POST /createUser          │
│   PUT /users/123                      POST /updateUser          │
│   DELETE /users/123                   GET /deleteUser?id=123    │
│                                                                 │
│   GET /users/123/orders               GET /getUserOrders        │
│   POST /users/123/orders              POST /createOrderForUser  │
│                                                                 │
│   规则:                                                         │
│   1. 使用名词复数 (/users 而非 /user)                           │
│   2. 使用小写和连字符 (/order-items 而非 /orderItems)           │
│   3. 不在 URL 中使用动词 (用 HTTP 方法表示动作)                  │
│   4. 资源层级不超过 3 层                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. HTTP 方法语义

```
┌─────────────────────────────────────────────────────────────────┐
│                    HTTP 方法使用规范                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   方法      │ 语义     │ 幂等 │ 安全 │ 示例                     │
│   ─────────┼──────────┼──────┼──────┼─────────────────────     │
│   GET      │ 查询     │  是  │  是  │ GET /users/123           │
│   POST     │ 创建     │  否  │  否  │ POST /users              │
│   PUT      │ 全量更新 │  是  │  否  │ PUT /users/123           │
│   PATCH    │ 部分更新 │  是  │  否  │ PATCH /users/123         │
│   DELETE   │ 删除     │  是  │  否  │ DELETE /users/123        │
│                                                                 │
│   幂等: 多次请求效果与一次相同                                   │
│   安全: 不修改资源状态                                          │
│                                                                 │
│   特殊操作的处理:                                                │
│   ─────────────────────────────────────────────────────────    │
│   操作              │ RESTful 方式                              │
│   ─────────────────┼────────────────────────────────────────   │
│   登录              │ POST /sessions (创建会话资源)             │
│   登出              │ DELETE /sessions/current                 │
│   批量删除          │ DELETE /users?ids=1,2,3                  │
│   复杂搜索          │ POST /users/search                       │
│   状态变更          │ PATCH /orders/123 {"status": "paid"}     │
│   批量操作          │ POST /orders/batch                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3. 状态码使用

```
┌─────────────────────────────────────────────────────────────────┐
│                    HTTP 状态码规范                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   2xx 成功                                                      │
│   ─────────────────────────────────────────────────────────    │
│   200 OK              │ 通用成功                                │
│   201 Created         │ 创建成功，返回新资源                     │
│   202 Accepted        │ 请求已接受，异步处理中                   │
│   204 No Content      │ 成功，无返回内容 (DELETE)                │
│                                                                 │
│   4xx 客户端错误                                                 │
│   ─────────────────────────────────────────────────────────    │
│   400 Bad Request     │ 请求参数错误                            │
│   401 Unauthorized    │ 未认证 (需要登录)                        │
│   403 Forbidden       │ 无权限 (已登录但权限不足)                 │
│   404 Not Found       │ 资源不存在                              │
│   409 Conflict        │ 资源冲突 (如重复创建)                    │
│   422 Unprocessable   │ 参数验证失败                            │
│   429 Too Many Reqs   │ 请求频率限制                            │
│                                                                 │
│   5xx 服务端错误                                                 │
│   ─────────────────────────────────────────────────────────    │
│   500 Internal Error  │ 服务器内部错误                          │
│   502 Bad Gateway     │ 网关错误 (上游服务故障)                  │
│   503 Unavailable     │ 服务不可用 (过载或维护)                  │
│   504 Gateway Timeout │ 网关超时                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4. 请求响应规范

```json
// 成功响应
{
    "code": 0,
    "message": "success",
    "data": {
        "id": "123",
        "name": "张三",
        "email": "zhang@example.com",
        "created_at": "2024-01-15T10:30:00Z"
    }
}

// 列表响应 (带分页)
{
    "code": 0,
    "message": "success",
    "data": {
        "items": [
            {"id": "1", "name": "用户1"},
            {"id": "2", "name": "用户2"}
        ],
        "pagination": {
            "page": 1,
            "page_size": 20,
            "total": 100,
            "total_pages": 5
        }
    }
}

// 错误响应
{
    "code": 40001,
    "message": "参数验证失败",
    "data": null,
    "errors": [
        {"field": "email", "message": "邮箱格式不正确"},
        {"field": "password", "message": "密码长度至少8位"}
    ],
    "request_id": "req_abc123"
}
```

```go
// Go 响应结构体
type Response struct {
    Code      int         `json:"code"`
    Message   string      `json:"message"`
    Data      interface{} `json:"data,omitempty"`
    Errors    []Error     `json:"errors,omitempty"`
    RequestID string      `json:"request_id,omitempty"`
}

type Error struct {
    Field   string `json:"field"`
    Message string `json:"message"`
}

type Pagination struct {
    Page       int `json:"page"`
    PageSize   int `json:"page_size"`
    Total      int `json:"total"`
    TotalPages int `json:"total_pages"`
}

// 响应工具函数
func Success(c *gin.Context, data interface{}) {
    c.JSON(http.StatusOK, Response{
        Code:      0,
        Message:   "success",
        Data:      data,
        RequestID: c.GetString("request_id"),
    })
}

func Error(c *gin.Context, code int, message string) {
    c.JSON(getHTTPStatus(code), Response{
        Code:      code,
        Message:   message,
        RequestID: c.GetString("request_id"),
    })
}
```

---

## 二、API 版本管理

### 版本策略

```
┌─────────────────────────────────────────────────────────────────┐
│                    API 版本管理策略                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   方式 1: URL 路径版本 (推荐)                                    │
│   ─────────────────────────────────────────────────────────    │
│   /api/v1/users                                                 │
│   /api/v2/users                                                 │
│                                                                 │
│   优点: 直观、易于调试、CDN友好                                  │
│   缺点: URL 变化、可能维护多版本                                 │
│                                                                 │
│                                                                 │
│   方式 2: Header 版本                                            │
│   ─────────────────────────────────────────────────────────    │
│   GET /api/users                                                │
│   Accept: application/vnd.api+json; version=2                   │
│                                                                 │
│   优点: URL 稳定                                                │
│   缺点: 不直观、调试困难                                        │
│                                                                 │
│                                                                 │
│   方式 3: 查询参数版本                                          │
│   ─────────────────────────────────────────────────────────    │
│   /api/users?version=2                                          │
│                                                                 │
│   优点: 灵活                                                    │
│   缺点: 与业务参数混淆                                          │
│                                                                 │
│                                                                 │
│   版本演进策略:                                                  │
│   ─────────────────────────────────────────────────────────    │
│   1. 新功能: 添加新字段 (向后兼容)                               │
│   2. 小改动: 使用特性开关控制                                   │
│   3. 大改动: 发布新版本 (v2)                                    │
│   4. 废弃旧版本: 通知 → 警告 → 下线                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 向后兼容

```go
// 向后兼容的字段添加
type UserV1 struct {
    ID    string `json:"id"`
    Name  string `json:"name"`
    Email string `json:"email"`
}

type UserV2 struct {
    ID        string     `json:"id"`
    Name      string     `json:"name"`
    Email     string     `json:"email"`
    Phone     string     `json:"phone,omitempty"`     // 新增可选字段
    CreatedAt *time.Time `json:"created_at,omitempty"` // 新增可选字段
}

// 废弃字段标记
type User struct {
    ID       string `json:"id"`
    Name     string `json:"name"`
    Username string `json:"username,omitempty"` // Deprecated: use name instead
    Email    string `json:"email"`
}

// 版本路由
func SetupRoutes(r *gin.Engine) {
    v1 := r.Group("/api/v1")
    {
        v1.GET("/users", v1GetUsers)
        v1.POST("/users", v1CreateUser)
    }

    v2 := r.Group("/api/v2")
    {
        v2.GET("/users", v2GetUsers)  // 新版本接口
        v2.POST("/users", v2CreateUser)
    }
}
```

---

## 三、gRPC API 设计

### 1. Proto 定义规范

```protobuf
// user.proto
syntax = "proto3";

package user.v1;

option go_package = "github.com/example/api/user/v1";

import "google/protobuf/timestamp.proto";
import "google/protobuf/empty.proto";

// 用户服务
service UserService {
    // 获取用户
    rpc GetUser(GetUserRequest) returns (GetUserResponse);

    // 创建用户
    rpc CreateUser(CreateUserRequest) returns (CreateUserResponse);

    // 更新用户
    rpc UpdateUser(UpdateUserRequest) returns (UpdateUserResponse);

    // 删除用户
    rpc DeleteUser(DeleteUserRequest) returns (google.protobuf.Empty);

    // 用户列表
    rpc ListUsers(ListUsersRequest) returns (ListUsersResponse);

    // 流式获取用户变更
    rpc WatchUsers(WatchUsersRequest) returns (stream UserEvent);
}

// 用户实体
message User {
    string id = 1;
    string name = 2;
    string email = 3;
    UserStatus status = 4;
    google.protobuf.Timestamp created_at = 5;
    google.protobuf.Timestamp updated_at = 6;
}

enum UserStatus {
    USER_STATUS_UNSPECIFIED = 0;
    USER_STATUS_ACTIVE = 1;
    USER_STATUS_INACTIVE = 2;
    USER_STATUS_BANNED = 3;
}

// 请求消息
message GetUserRequest {
    string id = 1;
}

message CreateUserRequest {
    string name = 1;
    string email = 2;
    string password = 3;
}

message UpdateUserRequest {
    string id = 1;
    optional string name = 2;
    optional string email = 3;
}

message DeleteUserRequest {
    string id = 1;
}

message ListUsersRequest {
    int32 page = 1;
    int32 page_size = 2;
    string filter = 3;  // 过滤条件
    string order_by = 4;  // 排序字段
}

// 响应消息
message GetUserResponse {
    User user = 1;
}

message CreateUserResponse {
    User user = 1;
}

message UpdateUserResponse {
    User user = 1;
}

message ListUsersResponse {
    repeated User users = 1;
    int32 total = 2;
    int32 page = 3;
    int32 page_size = 4;
}

// 事件
message UserEvent {
    EventType type = 1;
    User user = 2;
    google.protobuf.Timestamp timestamp = 3;
}

enum EventType {
    EVENT_TYPE_UNSPECIFIED = 0;
    EVENT_TYPE_CREATED = 1;
    EVENT_TYPE_UPDATED = 2;
    EVENT_TYPE_DELETED = 3;
}
```

### 2. 错误处理

```protobuf
// errors.proto
import "google/rpc/status.proto";

// 使用 google.rpc.Status 标准错误格式
// 错误码使用 google.rpc.Code

// 自定义错误详情
message ErrorDetail {
    string code = 1;       // 业务错误码
    string message = 2;    // 错误消息
    string field = 3;      // 错误字段
}
```

```go
// Go 错误处理
import (
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

func (s *UserServer) GetUser(ctx context.Context, req *pb.GetUserRequest) (*pb.GetUserResponse, error) {
    user, err := s.repo.GetByID(req.Id)
    if err != nil {
        if errors.Is(err, ErrNotFound) {
            return nil, status.Error(codes.NotFound, "user not found")
        }
        return nil, status.Error(codes.Internal, "internal error")
    }

    return &pb.GetUserResponse{User: user}, nil
}

// 带详情的错误
func (s *UserServer) CreateUser(ctx context.Context, req *pb.CreateUserRequest) (*pb.CreateUserResponse, error) {
    // 验证
    if req.Name == "" {
        st := status.New(codes.InvalidArgument, "validation failed")
        st, _ = st.WithDetails(&pb.ErrorDetail{
            Code:    "INVALID_NAME",
            Message: "name is required",
            Field:   "name",
        })
        return nil, st.Err()
    }

    // ...
}
```

---

## 四、API 安全

### 1. 认证方式

```
┌─────────────────────────────────────────────────────────────────┐
│                    API 认证方式对比                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. API Key                                                    │
│   ─────────────────────────────────────────────────────────    │
│   Authorization: ApiKey abc123                                  │
│   X-API-Key: abc123                                            │
│                                                                 │
│   适用: 服务间调用、第三方集成                                   │
│   优点: 简单                                                    │
│   缺点: 泄露风险、无法撤销单次                                   │
│                                                                 │
│   2. JWT (JSON Web Token)                                       │
│   ─────────────────────────────────────────────────────────    │
│   Authorization: Bearer eyJhbGciOiJIUzI1NiIs...                │
│                                                                 │
│   适用: 用户认证、前后端分离                                     │
│   优点: 无状态、包含用户信息                                     │
│   缺点: 无法主动失效、payload 可被解码                           │
│                                                                 │
│   3. OAuth 2.0                                                  │
│   ─────────────────────────────────────────────────────────    │
│   Authorization: Bearer access_token                            │
│                                                                 │
│   适用: 第三方授权、开放平台                                     │
│   优点: 标准化、细粒度授权                                       │
│   缺点: 复杂度高                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. JWT 实现

```go
import (
    "github.com/golang-jwt/jwt/v5"
)

type Claims struct {
    UserID   string   `json:"user_id"`
    Username string   `json:"username"`
    Roles    []string `json:"roles"`
    jwt.RegisteredClaims
}

type JWTService struct {
    secretKey     []byte
    accessExpiry  time.Duration
    refreshExpiry time.Duration
}

func (s *JWTService) GenerateToken(user *User) (string, string, error) {
    // Access Token (短期)
    accessClaims := Claims{
        UserID:   user.ID,
        Username: user.Username,
        Roles:    user.Roles,
        RegisteredClaims: jwt.RegisteredClaims{
            ExpiresAt: jwt.NewNumericDate(time.Now().Add(s.accessExpiry)),
            IssuedAt:  jwt.NewNumericDate(time.Now()),
            Issuer:    "my-service",
        },
    }
    accessToken := jwt.NewWithClaims(jwt.SigningMethodHS256, accessClaims)
    accessStr, err := accessToken.SignedString(s.secretKey)
    if err != nil {
        return "", "", err
    }

    // Refresh Token (长期)
    refreshClaims := jwt.RegisteredClaims{
        ExpiresAt: jwt.NewNumericDate(time.Now().Add(s.refreshExpiry)),
        IssuedAt:  jwt.NewNumericDate(time.Now()),
        Subject:   user.ID,
    }
    refreshToken := jwt.NewWithClaims(jwt.SigningMethodHS256, refreshClaims)
    refreshStr, err := refreshToken.SignedString(s.secretKey)
    if err != nil {
        return "", "", err
    }

    return accessStr, refreshStr, nil
}

func (s *JWTService) ValidateToken(tokenStr string) (*Claims, error) {
    token, err := jwt.ParseWithClaims(tokenStr, &Claims{}, func(token *jwt.Token) (interface{}, error) {
        return s.secretKey, nil
    })
    if err != nil {
        return nil, err
    }

    claims, ok := token.Claims.(*Claims)
    if !ok || !token.Valid {
        return nil, errors.New("invalid token")
    }

    return claims, nil
}

// 中间件
func AuthMiddleware(jwtService *JWTService) gin.HandlerFunc {
    return func(c *gin.Context) {
        authHeader := c.GetHeader("Authorization")
        if authHeader == "" {
            c.JSON(401, gin.H{"error": "missing authorization header"})
            c.Abort()
            return
        }

        tokenStr := strings.TrimPrefix(authHeader, "Bearer ")
        claims, err := jwtService.ValidateToken(tokenStr)
        if err != nil {
            c.JSON(401, gin.H{"error": "invalid token"})
            c.Abort()
            return
        }

        c.Set("user_id", claims.UserID)
        c.Set("roles", claims.Roles)
        c.Next()
    }
}
```

### 3. API 限流

```go
// 令牌桶限流
import "golang.org/x/time/rate"

type RateLimiter struct {
    limiters sync.Map
    rate     rate.Limit
    burst    int
}

func NewRateLimiter(r rate.Limit, burst int) *RateLimiter {
    return &RateLimiter{
        rate:  r,
        burst: burst,
    }
}

func (r *RateLimiter) GetLimiter(key string) *rate.Limiter {
    limiter, exists := r.limiters.Load(key)
    if !exists {
        limiter = rate.NewLimiter(r.rate, r.burst)
        r.limiters.Store(key, limiter)
    }
    return limiter.(*rate.Limiter)
}

// 限流中间件
func RateLimitMiddleware(limiter *RateLimiter) gin.HandlerFunc {
    return func(c *gin.Context) {
        // 按用户 ID 或 IP 限流
        key := c.GetString("user_id")
        if key == "" {
            key = c.ClientIP()
        }

        if !limiter.GetLimiter(key).Allow() {
            c.JSON(429, gin.H{
                "error": "rate limit exceeded",
                "retry_after": 1,
            })
            c.Abort()
            return
        }

        c.Next()
    }
}
```

---

## 五、API 文档

### OpenAPI/Swagger

```yaml
# openapi.yaml
openapi: 3.0.3
info:
  title: User Service API
  description: 用户服务 API 文档
  version: 1.0.0
  contact:
    name: API Support
    email: api@example.com

servers:
  - url: https://api.example.com/v1
    description: 生产环境
  - url: https://api-staging.example.com/v1
    description: 测试环境

paths:
  /users:
    get:
      summary: 获取用户列表
      tags:
        - Users
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: page_size
          in: query
          schema:
            type: integer
            default: 20
      responses:
        '200':
          description: 成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserListResponse'

    post:
      summary: 创建用户
      tags:
        - Users
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
      responses:
        '201':
          description: 创建成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserResponse'
        '400':
          $ref: '#/components/responses/BadRequest'

components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: string
          example: "123"
        name:
          type: string
          example: "张三"
        email:
          type: string
          format: email
          example: "zhang@example.com"
        created_at:
          type: string
          format: date-time

    CreateUserRequest:
      type: object
      required:
        - name
        - email
        - password
      properties:
        name:
          type: string
          minLength: 2
          maxLength: 50
        email:
          type: string
          format: email
        password:
          type: string
          minLength: 8

  responses:
    BadRequest:
      description: 请求参数错误
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'

  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
```

---

## 六、检查清单

### API 设计检查

- [ ] URL 是否使用名词复数？
- [ ] HTTP 方法是否正确使用？
- [ ] 状态码是否规范？
- [ ] 响应格式是否统一？

### 安全检查

- [ ] 是否有认证机制？
- [ ] 是否有限流保护？
- [ ] 敏感数据是否加密传输？
- [ ] 是否防止常见攻击？

### 文档检查

- [ ] 是否有完整的 API 文档？
- [ ] 是否包含请求/响应示例？
- [ ] 错误码是否有说明？
- [ ] 是否有版本变更记录？
