// ============================================================
//                      指针（Pointers）
// ============================================================
// 指针是 C 语言的核心特性，存储内存地址
// 理解指针是掌握 C 语言的关键
// & 取地址运算符，* 解引用运算符

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    printf("=== 指针基础 ===\n");

    // ----------------------------------------------------------
    // 1. 指针声明与基本操作
    // ----------------------------------------------------------
    int x = 42;
    int *p = &x;  // p 是指向 int 的指针，存储 x 的地址

    printf("x 的值:    %d\n", x);
    printf("x 的地址:  %p\n", (void*)&x);
    printf("p 的值:    %p（存储 x 的地址）\n", (void*)p);
    printf("*p（解引用）: %d（x 的值）\n", *p);

    // 通过指针修改值
    *p = 100;
    printf("修改后 x = %d\n", x);  // x 变成 100

    // NULL 指针（不指向任何有效地址）
    int *null_p = NULL;
    printf("NULL 指针: %p\n", (void*)null_p);
    // 【危险】不要解引用 NULL 指针：*null_p  -> 段错误！

    // ----------------------------------------------------------
    // 2. 指针与数组
    // ----------------------------------------------------------
    printf("\n=== 指针与数组 ===\n");

    int arr[] = {10, 20, 30, 40, 50};
    int *pa = arr;  // 数组名即首元素地址

    printf("arr[0]   = %d\n", arr[0]);
    printf("*pa      = %d\n", *pa);       // 等价于 arr[0]
    printf("*(pa+1)  = %d\n", *(pa+1));   // 等价于 arr[1]
    printf("*(pa+2)  = %d\n", *(pa+2));   // 等价于 arr[2]

    // 指针算术（pointer arithmetic）
    // 指针加减的单位是元素大小，不是字节
    printf("\n指针算术:\n");
    for (int i = 0; i < 5; i++) {
        printf("  arr+%d = %p, 值 = %d\n", i, (void*)(arr+i), *(arr+i));
    }

    // 数组下标等价于指针运算
    for (int i = 0; i < 5; i++) {
        // arr[i] 等价于 *(arr + i)
        printf("  arr[%d]=%d, *(arr+%d)=%d\n", i, arr[i], i, *(arr+i));
    }

    // 指针减法（得到两指针间的元素数）
    int *first = arr;
    int *last = arr + 4;
    printf("last - first = %td（元素数）\n", last - first);

    // ----------------------------------------------------------
    // 3. 指针的指针（二级指针）
    // ----------------------------------------------------------
    printf("\n=== 二级指针 ===\n");

    int val = 999;
    int *ptr = &val;
    int **pptr = &ptr;  // 指向指针的指针

    printf("val    = %d\n", val);
    printf("*ptr   = %d\n", *ptr);
    printf("**pptr = %d\n", **pptr);  // 解引用两次

    **pptr = 777;
    printf("修改后 val = %d\n", val);

    // 应用：函数修改指针本身
    // void alloc(int **out) { *out = malloc(sizeof(int)); }

    // ----------------------------------------------------------
    // 4. 指针与 const
    // ----------------------------------------------------------
    printf("\n=== const 指针 ===\n");

    int a = 10, b = 20;

    // 指向常量的指针：不能通过指针修改值，但可以改变指针指向
    const int *cp = &a;
    printf("const int *cp = %d\n", *cp);
    // *cp = 99;  // 错误！不能通过 cp 修改 a
    cp = &b;     // 可以：指针本身可以改变指向
    printf("cp 改指向 b: %d\n", *cp);

    // 常量指针：指针本身不能改变，但可以修改它指向的值
    int *const cp2 = &a;
    *cp2 = 99;   // 可以：修改 a 的值
    // cp2 = &b; // 错误！指针本身是常量
    printf("常量指针 *cp2 = %d\n", *cp2);

    // 指向常量的常量指针：两者都不能改
    const int *const cp3 = &a;
    printf("双 const: %d\n", *cp3);

    // 函数参数中常见：const 防止意外修改
    // void print_array(const int *arr, int len) { ... }

    // ----------------------------------------------------------
    // 5. void 指针（通用指针）
    // ----------------------------------------------------------
    printf("\n=== void* 指针 ===\n");

    // void* 可以指向任何类型，但使用前必须转换
    int    i_val = 42;
    double d_val = 3.14;
    char   c_val = 'X';

    void *vp;

    vp = &i_val;
    printf("void* 指向 int: %d\n", *(int*)vp);  // 必须强制转换

    vp = &d_val;
    printf("void* 指向 double: %.2f\n", *(double*)vp);

    vp = &c_val;
    printf("void* 指向 char: %c\n", *(char*)vp);

    // memcpy、memset 使用 void*
    int src[5] = {1, 2, 3, 4, 5};
    int dst[5];
    memcpy(dst, src, sizeof(src));
    printf("memcpy: ");
    for (int i = 0; i < 5; i++) printf("%d ", dst[i]);
    printf("\n");

    // ----------------------------------------------------------
    // 6. 函数指针详解
    // ----------------------------------------------------------
    printf("\n=== 函数指针 ===\n");

    // 声明：返回类型 (*指针名)(参数类型...)
    int (*fp)(int, int);

    // 赋值并调用
    int add_func(int a, int b);  // 前向声明
    // fp = add_func;
    // printf("fp(3, 4) = %d\n", fp(3, 4));

    // 指向各种函数
    void (*print_fn)(const char *) = printf;  // 指向 printf（简化）
    // print_fn("测试函数指针\n");

    printf("sizeof(int*)   = %zu\n", sizeof(int*));
    printf("sizeof(void*)  = %zu\n", sizeof(void*));
    printf("sizeof(int)    = %zu\n", sizeof(int));

    // ----------------------------------------------------------
    // 7. 常见指针错误（警示）
    // ----------------------------------------------------------
    printf("\n=== 常见指针错误（演示，不实际触发）===\n");
    printf("1. 解引用 NULL 指针 -> 段错误\n");
    printf("2. 悬空指针（Dangling Pointer）：指向已释放的内存\n");
    printf("3. 越界访问：arr[-1] 或 arr[len]\n");
    printf("4. 未初始化指针（野指针）\n");
    printf("5. 内存泄漏：malloc 后忘记 free\n");

    // 正确做法：初始化后检查
    int *safe_p = NULL;
    safe_p = (int*)malloc(sizeof(int));
    if (safe_p == NULL) {
        fprintf(stderr, "内存分配失败\n");
        return 1;
    }
    *safe_p = 42;
    printf("安全分配: %d\n", *safe_p);
    free(safe_p);
    safe_p = NULL;  // 释放后置 NULL，避免悬空指针

    printf("\n=== 指针演示完成 ===\n");
    return 0;
}

int add_func(int a, int b) { return a + b; }
