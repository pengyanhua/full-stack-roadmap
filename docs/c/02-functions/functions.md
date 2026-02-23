# functions.c

::: info 文件信息
- 📄 原文件：`01_functions.c`
- 🔤 语言：c
:::

## 完整代码

```c
// ============================================================
//                      函数
// ============================================================
// C 函数：返回类型 函数名(参数列表) { 函数体 }
// 函数原型（声明）告知编译器函数签名
// 函数指针：C 中实现回调和策略模式的核心机制

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>   // 需链接 -lm

// ============================================================
//                      函数声明（原型）
// ============================================================
// 函数原型让编译器在调用前知道函数签名
// 通常放在头文件（.h）中

int add(int a, int b);
double circle_area(double radius);
void swap(int *a, int *b);
int factorial(int n);
int fibonacci(int n);

// 函数指针类型别名（增强可读性）
typedef int (*BinaryOp)(int, int);
typedef int (*Comparator)(const void *, const void *);

// ============================================================
//                      函数定义
// ============================================================

// ----------------------------------------------------------
// 1. 基本函数
// ----------------------------------------------------------
int add(int a, int b) {
    return a + b;
}

// void 函数（无返回值）
void greet(const char *name) {
    printf("你好，%s！\n", name);
}

// 返回 double
double circle_area(double radius) {
    return 3.14159265358979 * radius * radius;
}

// ----------------------------------------------------------
// 2. 参数传递：值传递 vs 指针传递
// ----------------------------------------------------------
// 【重要】C 函数参数默认是值传递（副本），修改不影响原变量

void try_swap_by_value(int a, int b) {
    int temp = a;
    a = b;
    b = temp;
    // 只交换了局部副本，原变量不变
}

// 指针传递：传递地址，可以修改原变量
void swap(int *a, int *b) {
    int temp = *a;  // 解引用：获取地址处的值
    *a = *b;
    *b = temp;
}

// 通过指针返回多个值
void min_max(const int *arr, int len, int *min, int *max) {
    *min = *max = arr[0];
    for (int i = 1; i < len; i++) {
        if (arr[i] < *min) *min = arr[i];
        if (arr[i] > *max) *max = arr[i];
    }
}

// ----------------------------------------------------------
// 3. 递归函数
// ----------------------------------------------------------
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// 尾递归（编译器可能优化为循环，避免栈溢出）
int factorial_tail(int n, int acc) {
    if (n <= 1) return acc;
    return factorial_tail(n - 1, n * acc);
}

// ----------------------------------------------------------
// 4. 可变参数函数
// ----------------------------------------------------------
#include <stdarg.h>  // va_list, va_start, va_arg, va_end

// 实现简单的 printf 风格函数
int my_sum(int count, ...) {
    va_list args;
    va_start(args, count);  // 从 count 后开始

    int sum = 0;
    for (int i = 0; i < count; i++) {
        sum += va_arg(args, int);  // 逐个取出参数
    }

    va_end(args);  // 必须调用，清理
    return sum;
}

// ----------------------------------------------------------
// 5. 内联函数（C99，避免函数调用开销）
// ----------------------------------------------------------
static inline int max(int a, int b) {
    return (a > b) ? a : b;
}

static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

// ----------------------------------------------------------
// 6. 静态函数（internal linkage，只在本文件可见）
// ----------------------------------------------------------
static double power(double base, int exp) {
    double result = 1.0;
    for (int i = 0; i < exp; i++)
        result *= base;
    return result;
}

// ----------------------------------------------------------
// 7. 函数指针
// ----------------------------------------------------------
int subtract(int a, int b) { return a - b; }
int multiply(int a, int b) { return a * b; }

// 接受函数指针作为参数（高阶函数）
int apply(int a, int b, BinaryOp op) {
    return op(a, b);
}

// 函数指针数组（分发表/跳转表）
int dispatch(int op, int a, int b) {
    BinaryOp ops[] = { add, subtract, multiply };
    if (op < 0 || op >= 3) return -1;
    return ops[op](a, b);
}

// 回调函数示例：排序比较器
int compare_asc(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

int compare_desc(const void *a, const void *b) {
    return (*(int*)b - *(int*)a);
}

// ============================================================
//                      主函数
// ============================================================
int main(void) {
    printf("=== 基本函数 ===\n");

    printf("add(3, 5) = %d\n", add(3, 5));
    greet("世界");
    printf("circle_area(5.0) = %.2f\n", circle_area(5.0));

    // 值传递 vs 指针传递
    printf("\n=== 参数传递 ===\n");
    int x = 10, y = 20;
    printf("交换前: x=%d, y=%d\n", x, y);

    try_swap_by_value(x, y);
    printf("值传递后: x=%d, y=%d（未改变）\n", x, y);

    swap(&x, &y);  // 传递地址
    printf("指针传递后: x=%d, y=%d（已交换）\n", x, y);

    // 通过指针返回多个值
    int arr[] = {5, 2, 8, 1, 9, 3};
    int lo, hi;
    min_max(arr, 6, &lo, &hi);
    printf("数组最小值=%d, 最大值=%d\n", lo, hi);

    // 递归
    printf("\n=== 递归 ===\n");
    for (int i = 0; i <= 10; i++) {
        printf("%d! = %d\n", i, factorial(i));
    }

    printf("\n斐波那契: ");
    for (int i = 0; i <= 10; i++) {
        printf("%d ", fibonacci(i));
    }
    printf("\n");

    printf("尾递归 10! = %d\n", factorial_tail(10, 1));

    // 可变参数
    printf("\n=== 可变参数 ===\n");
    printf("my_sum(3, 1,2,3) = %d\n", my_sum(3, 1, 2, 3));
    printf("my_sum(5, 1,2,3,4,5) = %d\n", my_sum(5, 1, 2, 3, 4, 5));

    // 内联函数
    printf("\n=== 内联函数 ===\n");
    printf("max(7, 3) = %d\n", max(7, 3));
    printf("min(7, 3) = %d\n", min(7, 3));
    printf("power(2, 10) = %.0f\n", power(2, 10));

    // 函数指针
    printf("\n=== 函数指针 ===\n");

    BinaryOp op = add;
    printf("op(3, 4) = %d\n", op(3, 4));

    op = subtract;
    printf("op(10, 3) = %d\n", op(10, 3));

    printf("apply(6, 7, multiply) = %d\n", apply(6, 7, multiply));

    // 分发表
    printf("dispatch(0, 5, 3) = %d（加法）\n", dispatch(0, 5, 3));
    printf("dispatch(1, 5, 3) = %d（减法）\n", dispatch(1, 5, 3));
    printf("dispatch(2, 5, 3) = %d（乘法）\n", dispatch(2, 5, 3));

    // qsort 使用回调
    printf("\n=== qsort 回调 ===\n");
    int nums[] = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    int len = sizeof(nums) / sizeof(nums[0]);  // 数组长度

    qsort(nums, len, sizeof(int), compare_asc);
    printf("升序: ");
    for (int i = 0; i < len; i++) printf("%d ", nums[i]);
    printf("\n");

    qsort(nums, len, sizeof(int), compare_desc);
    printf("降序: ");
    for (int i = 0; i < len; i++) printf("%d ", nums[i]);
    printf("\n");

    printf("\n=== 函数演示完成 ===\n");
    return 0;
}
```
