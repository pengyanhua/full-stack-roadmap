# 安全架构基础

## 一、安全威胁模型

### OWASP Top 10

```
┌─────────────────────────────────────────────────────────────────┐
│                   OWASP Top 10 (2021)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. Broken Access Control (访问控制失效)                        │
│   ─────────────────────────────────────────────────────────    │
│   • 越权访问他人数据                                            │
│   • 修改 URL 绕过权限检查                                       │
│   防护: RBAC、最小权限原则                                       │
│                                                                 │
│   2. Cryptographic Failures (加密失效)                          │
│   ─────────────────────────────────────────────────────────    │
│   • 敏感数据明文传输/存储                                       │
│   • 使用弱加密算法                                              │
│   防护: HTTPS、强加密算法、密钥管理                              │
│                                                                 │
│   3. Injection (注入)                                           │
│   ─────────────────────────────────────────────────────────    │
│   • SQL 注入、命令注入、XSS                                     │
│   防护: 参数化查询、输入验证、输出编码                           │
│                                                                 │
│   4. Insecure Design (不安全设计)                               │
│   ─────────────────────────────────────────────────────────    │
│   • 业务逻辑漏洞                                                │
│   防护: 威胁建模、安全设计评审                                   │
│                                                                 │
│   5. Security Misconfiguration (安全配置错误)                   │
│   ─────────────────────────────────────────────────────────    │
│   • 默认配置、调试模式开启                                      │
│   防护: 安全基线、自动化扫描                                    │
│                                                                 │
│   6-10: SSRF、组件漏洞、认证失效、完整性失效、日志监控失效       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、认证与授权

### 1. 认证方式对比

| 方式 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| Session | Web应用 | 简单、安全 | 有状态、扩展难 |
| JWT | API/移动端 | 无状态、跨域 | 无法撤销、payload可见 |
| OAuth 2.0 | 第三方授权 | 标准化、细粒度 | 复杂 |
| API Key | 服务间调用 | 简单 | 安全性较低 |

### 2. JWT 安全实践

```go
import (
    "github.com/golang-jwt/jwt/v5"
)

type TokenService struct {
    accessSecret  []byte
    refreshSecret []byte
    accessTTL     time.Duration
    refreshTTL    time.Duration
}

// ✅ 安全的 JWT 生成
func (s *TokenService) GenerateTokenPair(userID string, roles []string) (*TokenPair, error) {
    now := time.Now()

    // Access Token (短有效期)
    accessClaims := jwt.MapClaims{
        "sub":   userID,
        "roles": roles,
        "type":  "access",
        "iat":   now.Unix(),
        "exp":   now.Add(s.accessTTL).Unix(),
        "jti":   uuid.New().String(),  // Token ID，用于撤销
    }
    accessToken := jwt.NewWithClaims(jwt.SigningMethodHS256, accessClaims)
    accessStr, _ := accessToken.SignedString(s.accessSecret)

    // Refresh Token (长有效期)
    refreshClaims := jwt.MapClaims{
        "sub":  userID,
        "type": "refresh",
        "iat":  now.Unix(),
        "exp":  now.Add(s.refreshTTL).Unix(),
        "jti":  uuid.New().String(),
    }
    refreshToken := jwt.NewWithClaims(jwt.SigningMethodHS256, refreshClaims)
    refreshStr, _ := refreshToken.SignedString(s.refreshSecret)

    return &TokenPair{
        AccessToken:  accessStr,
        RefreshToken: refreshStr,
    }, nil
}

// ✅ Token 黑名单 (用于撤销)
type TokenBlacklist struct {
    cache *redis.Client
}

func (b *TokenBlacklist) Revoke(jti string, expiry time.Duration) error {
    return b.cache.Set(context.Background(), "blacklist:"+jti, "1", expiry).Err()
}

func (b *TokenBlacklist) IsRevoked(jti string) bool {
    exists, _ := b.cache.Exists(context.Background(), "blacklist:"+jti).Result()
    return exists > 0
}

// ✅ Token 验证
func (s *TokenService) ValidateAccessToken(tokenStr string) (*Claims, error) {
    token, err := jwt.Parse(tokenStr, func(token *jwt.Token) (interface{}, error) {
        if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
            return nil, errors.New("invalid signing method")
        }
        return s.accessSecret, nil
    })

    if err != nil || !token.Valid {
        return nil, errors.New("invalid token")
    }

    claims := token.Claims.(jwt.MapClaims)

    // 检查类型
    if claims["type"] != "access" {
        return nil, errors.New("invalid token type")
    }

    // 检查黑名单
    if s.blacklist.IsRevoked(claims["jti"].(string)) {
        return nil, errors.New("token revoked")
    }

    return parseClaims(claims), nil
}
```

### 3. RBAC 权限模型

```go
// 角色权限模型
type Permission struct {
    Resource string // 资源: user, order, product
    Action   string // 操作: create, read, update, delete
}

type Role struct {
    Name        string
    Permissions []Permission
}

type RBACService struct {
    userRoles map[string][]string      // userID -> roles
    rolePerm  map[string][]Permission  // role -> permissions
}

func (s *RBACService) HasPermission(userID string, resource, action string) bool {
    roles := s.userRoles[userID]

    for _, role := range roles {
        permissions := s.rolePerm[role]
        for _, perm := range permissions {
            if perm.Resource == resource && perm.Action == action {
                return true
            }
            // 支持通配符
            if perm.Resource == "*" || perm.Action == "*" {
                return true
            }
        }
    }
    return false
}

// 权限中间件
func RequirePermission(resource, action string) gin.HandlerFunc {
    return func(c *gin.Context) {
        userID := c.GetString("user_id")

        if !rbacService.HasPermission(userID, resource, action) {
            c.JSON(403, gin.H{"error": "permission denied"})
            c.Abort()
            return
        }

        c.Next()
    }
}

// 使用
r.GET("/users", RequirePermission("user", "read"), listUsers)
r.POST("/users", RequirePermission("user", "create"), createUser)
```

---

## 三、输入验证与输出编码

### 1. SQL 注入防护

```go
// ❌ 危险: 字符串拼接
func badQuery(name string) {
    query := fmt.Sprintf("SELECT * FROM users WHERE name = '%s'", name)
    // 输入: ' OR '1'='1
    // 结果: SELECT * FROM users WHERE name = '' OR '1'='1'
}

// ✅ 安全: 参数化查询
func goodQuery(name string) {
    db.Query("SELECT * FROM users WHERE name = ?", name)
}

// ✅ 安全: ORM
func ormQuery(name string) {
    db.Where("name = ?", name).Find(&users)
}

// 输入验证
func ValidateInput(input string) error {
    // 长度限制
    if len(input) > 100 {
        return errors.New("input too long")
    }

    // 字符白名单
    if !regexp.MustCompile(`^[a-zA-Z0-9_-]+$`).MatchString(input) {
        return errors.New("invalid characters")
    }

    return nil
}
```

### 2. XSS 防护

```go
import "html"

// ❌ 危险: 直接输出用户输入
func badHandler(w http.ResponseWriter, r *http.Request) {
    name := r.URL.Query().Get("name")
    fmt.Fprintf(w, "<h1>Hello, %s</h1>", name)
    // 输入: <script>alert('xss')</script>
}

// ✅ 安全: HTML 编码
func goodHandler(w http.ResponseWriter, r *http.Request) {
    name := r.URL.Query().Get("name")
    safeName := html.EscapeString(name)
    fmt.Fprintf(w, "<h1>Hello, %s</h1>", safeName)
    // 输出: &lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;
}

// ✅ 使用模板引擎 (自动编码)
func templateHandler(w http.ResponseWriter, r *http.Request) {
    tmpl := template.Must(template.ParseFiles("template.html"))
    data := map[string]string{
        "Name": r.URL.Query().Get("name"),
    }
    tmpl.Execute(w, data)  // 自动 HTML 编码
}

// CSP 头设置
func SetSecurityHeaders(w http.ResponseWriter) {
    // 内容安全策略
    w.Header().Set("Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self'")
    // 防止 MIME 类型嗅探
    w.Header().Set("X-Content-Type-Options", "nosniff")
    // 防止点击劫持
    w.Header().Set("X-Frame-Options", "DENY")
    // XSS 过滤
    w.Header().Set("X-XSS-Protection", "1; mode=block")
}
```

---

## 四、加密与密钥管理

### 1. 密码存储

```go
import "golang.org/x/crypto/bcrypt"

// ✅ 使用 bcrypt 哈希密码
func HashPassword(password string) (string, error) {
    bytes, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
    return string(bytes), err
}

func CheckPassword(password, hash string) bool {
    err := bcrypt.CompareHashAndPassword([]byte(hash), []byte(password))
    return err == nil
}

// ❌ 危险做法
// MD5/SHA1 直接哈希
// 不加盐
// 自己实现加密算法
```

### 2. 敏感数据加密

```go
import (
    "crypto/aes"
    "crypto/cipher"
    "crypto/rand"
    "encoding/base64"
)

type Encryptor struct {
    key []byte // 32 bytes for AES-256
}

func (e *Encryptor) Encrypt(plaintext []byte) (string, error) {
    block, err := aes.NewCipher(e.key)
    if err != nil {
        return "", err
    }

    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return "", err
    }

    nonce := make([]byte, gcm.NonceSize())
    if _, err := rand.Read(nonce); err != nil {
        return "", err
    }

    ciphertext := gcm.Seal(nonce, nonce, plaintext, nil)
    return base64.StdEncoding.EncodeToString(ciphertext), nil
}

func (e *Encryptor) Decrypt(ciphertext string) ([]byte, error) {
    data, err := base64.StdEncoding.DecodeString(ciphertext)
    if err != nil {
        return nil, err
    }

    block, err := aes.NewCipher(e.key)
    if err != nil {
        return nil, err
    }

    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }

    nonceSize := gcm.NonceSize()
    nonce, ciphertextBytes := data[:nonceSize], data[nonceSize:]

    return gcm.Open(nil, nonce, ciphertextBytes, nil)
}
```

### 3. 密钥管理

```go
// 使用环境变量或密钥管理服务
type SecretManager struct {
    // 方案1: 环境变量
    // 方案2: HashiCorp Vault
    // 方案3: AWS KMS / 阿里云 KMS
}

func GetDatabasePassword() string {
    // ❌ 危险: 硬编码
    // return "password123"

    // ✅ 安全: 从环境变量读取
    return os.Getenv("DB_PASSWORD")
}

// Vault 集成示例
import vault "github.com/hashicorp/vault/api"

func GetSecretFromVault(path string) (string, error) {
    client, _ := vault.NewClient(vault.DefaultConfig())

    secret, err := client.Logical().Read(path)
    if err != nil {
        return "", err
    }

    return secret.Data["value"].(string), nil
}
```

---

## 五、安全检查清单

### 认证授权检查

- [ ] 密码是否使用强哈希算法？
- [ ] JWT 是否有过期时间？
- [ ] 是否有 Token 撤销机制？
- [ ] 是否实现了权限控制？

### 输入输出检查

- [ ] 是否使用参数化查询？
- [ ] 用户输入是否验证？
- [ ] 输出是否编码？
- [ ] 是否设置了安全 Header？

### 加密检查

- [ ] 敏感数据是否加密存储？
- [ ] 传输是否使用 HTTPS？
- [ ] 密钥是否安全管理？
- [ ] 是否使用强加密算法？
