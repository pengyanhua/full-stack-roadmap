# stdlib overview.c

::: info 文件信息
- 📄 原文件：`01_stdlib_overview.c`
- 🔤 语言：c
:::

## 完整代码

```c
// ============================================================
//                      C 标准库概览
// ============================================================
// C 标准库提供了丰富的函数和宏
// 常用头文件：
//   stdio.h   - 输入输出（printf, scanf, fopen...）
//   stdlib.h  - 通用工具（malloc, atoi, rand, qsort...）
//   string.h  - 字符串操作（strcpy, strcmp, memcpy...）
//   math.h    - 数学函数（sin, cos, sqrt, pow...）
//   time.h    - 时间和日期
//   assert.h  - 断言调试
//   errno.h   - 错误码
//   ctype.h   - 字符分类

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <errno.h>
#include <ctype.h>
#include <limits.h>

int main(void) {
    // ============================================================
    //                      stdlib.h — 通用工具
    // ============================================================
    printf("=== stdlib.h ===\n");

    // abs / labs / llabs：绝对值
    printf("abs(-5) = %d\n", abs(-5));
    printf("abs(-2147483647) = %d\n", abs(-2147483647));

    // div：同时得商和余数
    div_t result = div(17, 5);
    printf("17 / 5: 商=%d, 余=%d\n", result.quot, result.rem);

    // 随机数
    srand((unsigned int)time(NULL));  // 以当前时间为随机种子
    printf("随机数: ");
    for (int i = 0; i < 5; i++) {
        int r = rand() % 100;  // 0~99
        printf("%d ", r);
    }
    printf("\n");

    // 字符串转换
    printf("atoi(\"42\")  = %d\n", atoi("42"));
    printf("atof(\"3.14\") = %.2f\n", atof("3.14"));
    printf("atol(\"123456789\") = %ld\n", atol("123456789"));

    // strtol 更安全（可检测错误）
    char *end;
    long val = strtol("0xFF", &end, 16);  // 解析十六进制
    printf("strtol(\"0xFF\", 16) = %ld\n", val);

    // 环境变量
    const char *path = getenv("PATH");
    if (path) printf("PATH 前50字符: %.50s...\n", path);

    // qsort：通用排序
    int nums[] = {5, 2, 8, 1, 9, 3};
    int n = sizeof(nums) / sizeof(nums[0]);

    int cmp(const void *a, const void *b) {
        return *(int*)a - *(int*)b;
    }
    qsort(nums, n, sizeof(int), cmp);
    printf("qsort: ");
    for (int i = 0; i < n; i++) printf("%d ", nums[i]);
    printf("\n");

    // bsearch：二分查找（数组必须有序）
    int key = 5;
    int *found = (int*)bsearch(&key, nums, n, sizeof(int), cmp);
    printf("bsearch(%d): %s\n", key, found ? "找到" : "未找到");

    // ============================================================
    //                      math.h — 数学函数
    // ============================================================
    printf("\n=== math.h ===\n");

    printf("sqrt(16.0)  = %.2f\n", sqrt(16.0));
    printf("pow(2, 10)  = %.0f\n", pow(2.0, 10.0));
    printf("fabs(-3.14) = %.2f\n", fabs(-3.14));
    printf("ceil(3.2)   = %.0f\n", ceil(3.2));
    printf("floor(3.8)  = %.0f\n", floor(3.8));
    printf("round(3.5)  = %.0f\n", round(3.5));
    printf("fmod(10,3)  = %.0f\n", fmod(10, 3));

    // 三角函数（参数为弧度）
    double pi = acos(-1.0);  // π
    printf("π = %.10f\n", pi);
    printf("sin(π/6) = %.4f（0.5）\n", sin(pi/6));
    printf("cos(π/3) = %.4f（0.5）\n", cos(pi/3));
    printf("tan(π/4) = %.4f（1.0）\n", tan(pi/4));

    // 对数
    printf("log(e)   = %.4f（自然对数）\n", log(M_E));
    printf("log10(100) = %.4f\n", log10(100));
    printf("log2(8)    = %.4f\n", log2(8));

    // ============================================================
    //                      time.h — 时间日期
    // ============================================================
    printf("\n=== time.h ===\n");

    time_t now = time(NULL);  // Unix 时间戳（自 1970-01-01 的秒数）
    printf("Unix 时间戳: %ld\n", (long)now);

    // 转换为本地时间
    struct tm *local = localtime(&now);
    printf("本地时间: %04d-%02d-%02d %02d:%02d:%02d\n",
           local->tm_year + 1900, local->tm_mon + 1, local->tm_mday,
           local->tm_hour, local->tm_min, local->tm_sec);

    // 格式化时间字符串
    char time_str[64];
    strftime(time_str, sizeof(time_str), "%Y年%m月%d日 %H:%M:%S", local);
    printf("格式化: %s\n", time_str);

    // 计算程序运行时间
    clock_t start = clock();
    // 模拟一些计算
    double sum = 0;
    for (int i = 0; i < 10000000; i++) sum += i;
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("计算 sum=%.0f，耗时 %.4f 秒\n", sum, elapsed);

    // ============================================================
    //                      errno.h — 错误处理
    // ============================================================
    printf("\n=== errno.h ===\n");

    errno = 0;
    FILE *fp = fopen("不存在的文件.txt", "r");
    if (fp == NULL) {
        printf("fopen 失败，errno=%d，原因: %s\n", errno, strerror(errno));
    }

    // sqrt 对负数返回 NaN 并设置 errno
    errno = 0;
    double bad_sqrt = sqrt(-1.0);
    printf("sqrt(-1) = %f，errno=%d\n", bad_sqrt, errno);

    // ============================================================
    //                      assert.h — 调试断言
    // ============================================================
    printf("\n=== assert.h ===\n");

    // assert 在条件为假时终止程序（DEBUG 模式）
    // 通过 -DNDEBUG 编译可禁用所有 assert
    int x = 42;
    assert(x == 42);          // 通过
    assert(x > 0);            // 通过
    printf("断言通过: x=%d\n", x);
    // assert(x == 0);        // 失败时程序中止，打印错误信息

    // ============================================================
    //                      ctype.h — 字符操作
    // ============================================================
    printf("\n=== ctype.h ===\n");

    unsigned char chars[] = {'A', 'z', '5', ' ', '!', '\n'};
    for (int i = 0; i < 6; i++) {
        unsigned char c = chars[i];
        printf("'%c': alpha=%d digit=%d space=%d upper=%d lower=%d\n",
               isprint(c) ? c : '?',
               isalpha(c), isdigit(c), isspace(c),
               isupper(c), islower(c));
    }

    // 字符串大小写转换
    char s[] = "Hello, World!";
    printf("原文: %s\n", s);
    for (int i = 0; s[i]; i++) s[i] = toupper(s[i]);
    printf("大写: %s\n", s);
    for (int i = 0; s[i]; i++) s[i] = tolower(s[i]);
    printf("小写: %s\n", s);

    // ============================================================
    //                      limits.h — 类型极值
    // ============================================================
    printf("\n=== limits.h ===\n");
    printf("CHAR_MIN  = %d\n", CHAR_MIN);
    printf("CHAR_MAX  = %d\n", CHAR_MAX);
    printf("INT_MIN   = %d\n", INT_MIN);
    printf("INT_MAX   = %d\n", INT_MAX);
    printf("LONG_MAX  = %ld\n", LONG_MAX);
    printf("UINT_MAX  = %u\n", UINT_MAX);

    printf("\n=== 标准库演示完成 ===\n");
    return 0;
}
```
