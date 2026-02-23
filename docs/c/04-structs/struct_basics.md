# struct basics.c

::: info 文件信息
- 📄 原文件：`01_struct_basics.c`
- 🔤 语言：c
:::

## 完整代码

```c
// ============================================================
//                      结构体、联合体与枚举
// ============================================================
// struct：将不同类型的数据组合成一个整体（类似其他语言的 class）
// union：多个成员共享同一内存空间
// enum：命名整数常量的集合
// typedef：为类型创建别名

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

// ============================================================
//                      结构体定义
// ============================================================

// ----------------------------------------------------------
// 1. 基本结构体
// ----------------------------------------------------------
// 【命名约定】结构体名通常首字母大写或全大写

struct Point {
    double x;
    double y;
};

// typedef 创建别名，使用更方便
typedef struct {
    int year;
    int month;
    int day;
} Date;

// 嵌套结构体
typedef struct {
    char  name[50];
    int   age;
    Date  birthday;         // 嵌套 Date
    char  email[100];
    double salary;
} Employee;

// ----------------------------------------------------------
// 2. 结构体操作函数
// ----------------------------------------------------------

// 创建 Point（返回结构体值）
struct Point make_point(double x, double y) {
    struct Point p = { x, y };
    return p;
}

// 传指针（高效，避免复制大结构体）
double distance(const struct Point *p1, const struct Point *p2) {
    double dx = p1->x - p2->x;
    double dy = p1->y - p2->y;
    return sqrt(dx*dx + dy*dy);
}

// 打印 Employee
void print_employee(const Employee *e) {
    printf("  姓名: %s\n", e->name);
    printf("  年龄: %d\n", e->age);
    printf("  生日: %04d-%02d-%02d\n", e->birthday.year,
           e->birthday.month, e->birthday.day);
    printf("  邮箱: %s\n", e->email);
    printf("  薪资: %.2f\n", e->salary);
}

// ----------------------------------------------------------
// 3. 链表节点（递归结构体）
// ----------------------------------------------------------
typedef struct Node {
    int data;
    struct Node *next;  // 指向同类型的指针（必须用 struct Node*）
} Node;

Node *create_node(int data) {
    Node *node = (Node*)malloc(sizeof(Node));
    if (node) {
        node->data = data;
        node->next = NULL;
    }
    return node;
}

void list_prepend(Node **head, int data) {
    Node *node = create_node(data);
    node->next = *head;
    *head = node;
}

void list_print(const Node *head) {
    printf("[");
    while (head) {
        printf("%d%s", head->data, head->next ? " -> " : "");
        head = head->next;
    }
    printf("]\n");
}

void list_free(Node *head) {
    while (head) {
        Node *tmp = head;
        head = head->next;
        free(tmp);
    }
}

// ----------------------------------------------------------
// 4. 联合体（Union）
// ----------------------------------------------------------
// 所有成员共享同一块内存，大小由最大成员决定
// 【用途】节省内存、类型双关（Type Punning）

typedef union {
    int   i;
    float f;
    char  bytes[4];
} FloatBits;

// 带标签的联合体（Tagged Union / Discriminated Union）
// 模拟多态
typedef enum { SHAPE_CIRCLE, SHAPE_RECT, SHAPE_TRIANGLE } ShapeType;

typedef struct {
    ShapeType type;
    union {
        struct { double radius; } circle;
        struct { double width, height; } rect;
        struct { double base, height; } triangle;
    };
} Shape;

double shape_area(const Shape *s) {
    switch (s->type) {
        case SHAPE_CIRCLE:
            return 3.14159 * s->circle.radius * s->circle.radius;
        case SHAPE_RECT:
            return s->rect.width * s->rect.height;
        case SHAPE_TRIANGLE:
            return 0.5 * s->triangle.base * s->triangle.height;
        default:
            return 0.0;
    }
}

const char *shape_name(ShapeType t) {
    switch (t) {
        case SHAPE_CIRCLE:   return "圆形";
        case SHAPE_RECT:     return "矩形";
        case SHAPE_TRIANGLE: return "三角形";
        default:             return "未知";
    }
}

// ============================================================
//                      主函数
// ============================================================
int main(void) {
    printf("=== 结构体基础 ===\n");

    // ----------------------------------------------------------
    // 结构体初始化
    // ----------------------------------------------------------
    // 按成员顺序初始化
    struct Point p1 = { 0.0, 0.0 };

    // 指定成员初始化（C99，推荐）
    struct Point p2 = { .x = 3.0, .y = 4.0 };

    printf("p1 = (%.1f, %.1f)\n", p1.x, p1.y);
    printf("p2 = (%.1f, %.1f)\n", p2.x, p2.y);
    printf("距离 = %.2f\n", distance(&p1, &p2));

    // 结构体赋值（值语义，完整复制）
    struct Point p3 = p2;
    p3.x = 10.0;
    printf("p2.x = %.1f（未改变）\n", p2.x);
    printf("p3.x = %.1f（独立副本）\n", p3.x);

    // ----------------------------------------------------------
    // Employee 结构体
    // ----------------------------------------------------------
    printf("\n=== Employee 结构体 ===\n");

    Employee emp = {
        .name     = "张三",
        .age      = 30,
        .birthday = { .year = 1994, .month = 6, .day = 15 },
        .email    = "zhangsan@company.com",
        .salary   = 15000.0
    };

    printf("员工信息:\n");
    print_employee(&emp);

    // 通过 -> 访问指针指向的结构体成员
    Employee *pe = &emp;
    pe->age = 31;
    pe->salary *= 1.1;  // 涨薪 10%
    printf("更新后年龄: %d，薪资: %.2f\n", pe->age, pe->salary);

    // ----------------------------------------------------------
    // 结构体数组
    // ----------------------------------------------------------
    printf("\n=== 结构体数组 ===\n");

    Employee team[] = {
        { "李四", 25, {1999, 3, 20}, "lisi@co.com", 10000.0 },
        { "王五", 28, {1996, 7, 10}, "wangwu@co.com", 12000.0 },
        { "赵六", 35, {1989, 1, 5}, "zhaoliu@co.com", 20000.0 },
    };
    int team_size = sizeof(team) / sizeof(team[0]);

    // 找最高薪资
    Employee *highest = &team[0];
    for (int i = 1; i < team_size; i++) {
        if (team[i].salary > highest->salary)
            highest = &team[i];
    }
    printf("最高薪资: %s (%.2f)\n", highest->name, highest->salary);

    // ----------------------------------------------------------
    // 链表
    // ----------------------------------------------------------
    printf("\n=== 链表 ===\n");

    Node *head = NULL;
    for (int i = 5; i >= 1; i--)
        list_prepend(&head, i);

    printf("链表: ");
    list_print(head);
    list_free(head);
    head = NULL;

    // ----------------------------------------------------------
    // 联合体
    // ----------------------------------------------------------
    printf("\n=== 联合体 ===\n");

    FloatBits fb;
    fb.f = 3.14f;
    printf("float  = %f\n", fb.f);
    printf("int    = %d（相同字节的不同解释）\n", fb.i);
    printf("bytes  = %02X %02X %02X %02X\n",
           (unsigned char)fb.bytes[0], (unsigned char)fb.bytes[1],
           (unsigned char)fb.bytes[2], (unsigned char)fb.bytes[3]);

    printf("sizeof(FloatBits) = %zu（共享内存）\n", sizeof(FloatBits));

    // ----------------------------------------------------------
    // 带标签的联合体（多态形状）
    // ----------------------------------------------------------
    printf("\n=== 多态形状（Tagged Union）===\n");

    Shape shapes[] = {
        { .type = SHAPE_CIRCLE,   .circle   = { .radius = 5.0 } },
        { .type = SHAPE_RECT,     .rect     = { .width = 4.0, .height = 6.0 } },
        { .type = SHAPE_TRIANGLE, .triangle = { .base = 3.0, .height = 8.0 } },
    };
    int n_shapes = sizeof(shapes) / sizeof(shapes[0]);

    for (int i = 0; i < n_shapes; i++) {
        printf("%s 面积 = %.2f\n", shape_name(shapes[i].type), shape_area(&shapes[i]));
    }

    // ----------------------------------------------------------
    // 内存对齐
    // ----------------------------------------------------------
    printf("\n=== 内存对齐 ===\n");

    // 编译器会添加填充字节（padding）以满足对齐要求
    struct Padded {
        char  a;    // 1 字节
        // 3 字节填充
        int   b;    // 4 字节
        char  c;    // 1 字节
        // 3 字节填充
    };
    struct Packed {
        int   b;    // 4 字节
        char  a;    // 1 字节
        char  c;    // 1 字节
        // 2 字节填充
    };

    printf("Padded  大小: %zu\n", sizeof(struct Padded));
    printf("Packed  大小: %zu\n", sizeof(struct Packed));
    printf("【技巧】按大小降序排列成员可减少填充\n");

    printf("\n=== 结构体演示完成 ===\n");
    return 0;
}
```
