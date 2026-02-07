# embedding.go

::: info 文件信息
- 📄 原文件：`03_embedding.go`
- 🔤 语言：go
:::

## 完整代码

```go
package main

import "fmt"

// ============================================================
//                      类型嵌入与组合
// ============================================================
// Go 推崇组合（composition）而非继承（inheritance）
// 通过嵌入类型实现代码复用

func main03() {
	fmt.Println("\n==================== 03_embedding ====================")
	fmt.Println("=== 结构体嵌入 ===")

	// ----------------------------------------------------------
	// 基本嵌入
	// ----------------------------------------------------------
	dog := Dog{
		Animal: Animal{Name: "旺财", Age: 3},
		Breed:  "金毛",
	}

	// 字段提升：直接访问嵌入类型的字段
	fmt.Println("名字:", dog.Name)      // 等价于 dog.Animal.Name
	fmt.Println("年龄:", dog.Age)       // 等价于 dog.Animal.Age
	fmt.Println("品种:", dog.Breed)

	// 方法提升：直接调用嵌入类型的方法
	dog.Eat()   // 调用 Animal.Eat()
	dog.Sleep() // 调用 Animal.Sleep()
	dog.Bark()  // Dog 自己的方法

	// ----------------------------------------------------------
	// 方法覆盖
	// ----------------------------------------------------------
	fmt.Println("\n=== 方法覆盖 ===")

	cat := Cat{
		Animal: Animal{Name: "咪咪", Age: 2},
	}

	cat.Speak() // 调用 Cat.Speak()（覆盖了 Animal.Speak）
	cat.Animal.Speak() // 显式调用被覆盖的方法

	// ----------------------------------------------------------
	// 多重嵌入
	// ----------------------------------------------------------
	fmt.Println("\n=== 多重嵌入 ===")

	super := SuperHero{
		Person2: Person2{Name: "彼得·帕克", Age: 18},
		Powers:  Powers{Flight: false, Strength: true, Speed: true},
		Alias:   "蜘蛛侠",
	}

	fmt.Printf("英雄: %s (%s)\n", super.Alias, super.Name)
	fmt.Printf("能力: 飞行=%t, 力量=%t, 速度=%t\n",
		super.Flight, super.Strength, super.Speed)
	super.Introduce()

	// ----------------------------------------------------------
	// 嵌入接口
	// ----------------------------------------------------------
	fmt.Println("\n=== 嵌入接口 ===")

	logger := &FileLogger{filename: "app.log"}
	app := &Application{
		Logger: logger,
		Name:   "MyApp",
	}

	// 通过嵌入的接口调用方法
	app.Log("应用启动")
	app.Log("处理请求")

	// ----------------------------------------------------------
	// 嵌入指针
	// ----------------------------------------------------------
	fmt.Println("\n=== 嵌入指针 ===")

	base := &Base{ID: 1}
	derived := Derived{
		Base:  base,
		Extra: "额外数据",
	}

	fmt.Println("ID:", derived.ID)
	derived.PrintID()

	// 【注意】嵌入指针可能为 nil
	var d2 Derived
	// d2.PrintID() // panic! Base 是 nil

	if d2.Base != nil {
		d2.PrintID()
	} else {
		fmt.Println("Base 为 nil，跳过")
	}

	// ----------------------------------------------------------
	// 组合 vs 继承
	// ----------------------------------------------------------
	fmt.Println("\n=== 组合 vs 继承 ===")

	// 组合方式1：嵌入
	e1 := Engine{Power: 200}
	car1 := CarEmbed{Engine: e1, Brand: "Tesla"}
	car1.Start() // 直接调用

	// 组合方式2：字段
	car2 := CarField{engine: Engine{Power: 150}, Brand: "BMW"}
	car2.engine.Start() // 通过字段调用
	car2.Start()        // 包装方法

	// ----------------------------------------------------------
	// 实际应用：装饰器模式
	// ----------------------------------------------------------
	fmt.Println("\n=== 装饰器模式 ===")

	// 基础计数器
	var counter Counter = &BasicCounter{}

	// 添加日志装饰
	counter = &LoggingCounter{Counter: counter}

	// 添加线程安全装饰（示意）
	// counter = &ThreadSafeCounter{Counter: counter}

	counter.Increment()
	counter.Increment()
	fmt.Println("计数:", counter.Value())
}

// ============================================================
//                      类型定义
// ============================================================

// ----------------------------------------------------------
// 基本嵌入示例
// ----------------------------------------------------------

// Animal 动物基类
type Animal struct {
	Name string
	Age  int
}

func (a Animal) Eat() {
	fmt.Printf("%s 正在吃东西\n", a.Name)
}

func (a Animal) Sleep() {
	fmt.Printf("%s 正在睡觉\n", a.Name)
}

func (a Animal) Speak() {
	fmt.Printf("%s 发出声音\n", a.Name)
}

// Dog 狗（嵌入 Animal）
type Dog struct {
	Animal       // 嵌入
	Breed  string
}

func (d Dog) Bark() {
	fmt.Printf("%s 汪汪叫\n", d.Name)
}

// Cat 猫（嵌入 Animal，覆盖 Speak）
type Cat struct {
	Animal
}

// Speak 覆盖 Animal 的 Speak 方法
func (c Cat) Speak() {
	fmt.Printf("%s 喵喵叫\n", c.Name)
}

// ----------------------------------------------------------
// 多重嵌入
// ----------------------------------------------------------

type Person2 struct {
	Name string
	Age  int
}

type Powers struct {
	Flight   bool
	Strength bool
	Speed    bool
}

// SuperHero 多重嵌入
type SuperHero struct {
	Person2      // 嵌入 Person
	Powers       // 嵌入 Powers
	Alias  string
}

func (s SuperHero) Introduce() {
	fmt.Printf("我是 %s，也被称为 %s\n", s.Name, s.Alias)
}

// ----------------------------------------------------------
// 嵌入接口
// ----------------------------------------------------------

// Logger 日志接口
type Logger interface {
	Log(message string)
}

// FileLogger 文件日志实现
type FileLogger struct {
	filename string
}

func (f *FileLogger) Log(message string) {
	fmt.Printf("[%s] %s\n", f.filename, message)
}

// Application 应用（嵌入 Logger 接口）
type Application struct {
	Logger       // 嵌入接口
	Name   string
}

// ----------------------------------------------------------
// 嵌入指针
// ----------------------------------------------------------

type Base struct {
	ID int
}

func (b *Base) PrintID() {
	fmt.Println("Base ID:", b.ID)
}

type Derived struct {
	*Base        // 嵌入指针
	Extra string
}

// ----------------------------------------------------------
// 组合 vs 继承对比
// ----------------------------------------------------------

type Engine struct {
	Power int
}

func (e Engine) Start() {
	fmt.Printf("引擎启动 (功率: %d)\n", e.Power)
}

// CarEmbed 使用嵌入
type CarEmbed struct {
	Engine       // 嵌入
	Brand  string
}

// CarField 使用字段
type CarField struct {
	engine Engine // 字段
	Brand  string
}

func (c CarField) Start() {
	fmt.Printf("%s 汽车: ", c.Brand)
	c.engine.Start()
}

// ----------------------------------------------------------
// 装饰器模式示例
// ----------------------------------------------------------

// Counter 计数器接口
type Counter interface {
	Increment()
	Value() int
}

// BasicCounter 基础实现
type BasicCounter struct {
	count int
}

func (c *BasicCounter) Increment() {
	c.count++
}

func (c *BasicCounter) Value() int {
	return c.count
}

// LoggingCounter 带日志的装饰器
type LoggingCounter struct {
	Counter // 嵌入接口
}

func (c *LoggingCounter) Increment() {
	fmt.Println("[LOG] Increment 被调用")
	c.Counter.Increment() // 调用被装饰的对象
}

// ============================================================
//                      重要注意事项
// ============================================================
//
// 1. 【嵌入 ≠ 继承】
//    - 嵌入是组合，不是继承
//    - 没有 super 关键字
//    - 被嵌入类型不知道外部类型
//
// 2. 【字段/方法提升】
//    嵌入类型的字段和方法会提升到外层
//    可以直接访问，无需通过嵌入字段名
//
// 3. 【命名冲突】
//    - 外层字段/方法优先级更高
//    - 多个嵌入类型有相同名称时，必须显式指定
//
// 4. 【嵌入指针注意】
//    - 嵌入指针可能为 nil
//    - 调用方法前应检查
//
// 5. 【何时使用嵌入】
//    - 代码复用（共享字段和方法）
//    - 实现接口
//    - 装饰器模式
//
// 6. 【何时使用字段】
//    - 不想暴露内部实现
//    - 需要更细粒度的控制
//    - 避免命名冲突
//
// 7. 【设计原则】
//    "组合优于继承"
//    通过小接口和嵌入实现灵活的代码复用
```
