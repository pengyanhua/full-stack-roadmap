# user

::: info 文件信息
- 📄 原文件：`user.go`
- 🔤 语言：go
:::

## 完整代码

```go
// Package models 定义数据模型
package models

import (
	"fmt"
	"time"
)

// ----------------------------------------------------------
// User 用户模型
// ----------------------------------------------------------

// User 表示系统用户
type User struct {
	ID        int       `json:"id"`
	Username  string    `json:"username"`
	Email     string    `json:"email"`
	password  string    // 私有字段，包外不可访问
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// NewUser 创建新用户
func NewUser(username, email, password string) *User {
	now := time.Now()
	return &User{
		Username:  username,
		Email:     email,
		password:  hashPassword(password), // 使用私有函数
		CreatedAt: now,
		UpdatedAt: now,
	}
}

// SetPassword 设置密码
func (u *User) SetPassword(password string) {
	u.password = hashPassword(password)
	u.UpdatedAt = time.Now()
}

// CheckPassword 验证密码
func (u *User) CheckPassword(password string) bool {
	return u.password == hashPassword(password)
}

// String 实现 Stringer 接口
func (u *User) String() string {
	return fmt.Sprintf("User{ID: %d, Username: %s, Email: %s}", u.ID, u.Username, u.Email)
}

// ----------------------------------------------------------
// 私有辅助函数
// ----------------------------------------------------------

// hashPassword 简单的密码哈希（示例，实际应使用 bcrypt 等）
func hashPassword(password string) string {
	// 简化示例，实际不要这样做！
	return fmt.Sprintf("hashed_%s", password)
}

// ----------------------------------------------------------
// Product 产品模型
// ----------------------------------------------------------

// Product 表示商品
type Product struct {
	ID          int
	Name        string
	Price       float64
	Description string
	Stock       int
}

// NewProduct 创建产品
func NewProduct(name string, price float64) *Product {
	return &Product{
		Name:  name,
		Price: price,
		Stock: 0,
	}
}

// IsAvailable 是否有库存
func (p *Product) IsAvailable() bool {
	return p.Stock > 0
}

// AddStock 添加库存
func (p *Product) AddStock(quantity int) {
	if quantity > 0 {
		p.Stock += quantity
	}
}
```
