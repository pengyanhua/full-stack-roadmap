# arrays strings.c

::: info 文件信息
- 📄 原文件：`02_arrays_strings.c`
- 🔤 语言：c
:::

## 完整代码

```c
// ============================================================
//                      数组与字符串
// ============================================================
// C 数组：连续内存块，固定长度，下标从 0 开始
// C 字符串：以 '\0'（空字符）结尾的 char 数组
// string.h 提供丰富的字符串操作函数

#include <stdio.h>
#include <string.h>
#include <ctype.h>   // 字符分类函数
#include <stdlib.h>

// ============================================================
//                      数组操作示例函数
// ============================================================

// 打印数组
void print_array(const int *arr, int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("%d%s", arr[i], (i < n-1) ? ", " : "");
    }
    printf("]\n");
}

// 冒泡排序
void bubble_sort(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j+1]) {
                int tmp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = tmp;
            }
        }
    }
}

// 二分查找（数组必须有序）
int binary_search(const int *arr, int n, int target) {
    int left = 0, right = n - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;  // 未找到
}

// 字符串反转（原地）
void str_reverse(char *s) {
    int left = 0, right = (int)strlen(s) - 1;
    while (left < right) {
        char tmp = s[left];
        s[left++] = s[right];
        s[right--] = tmp;
    }
}

// 判断回文
int is_palindrome(const char *s) {
    int left = 0, right = (int)strlen(s) - 1;
    while (left < right) {
        if (tolower(s[left]) != tolower(s[right])) return 0;
        left++; right--;
    }
    return 1;
}

// 统计字符出现次数
void char_frequency(const char *s, int freq[256]) {
    memset(freq, 0, 256 * sizeof(int));
    while (*s) {
        freq[(unsigned char)*s]++;
        s++;
    }
}

int main(void) {
    // ============================================================
    //                      一维数组
    // ============================================================
    printf("=== 一维数组 ===\n");

    // 声明并初始化
    int arr1[5] = {10, 20, 30, 40, 50};
    int arr2[] = {1, 2, 3};      // 自动推断大小
    int arr3[5] = {0};           // 全部初始化为 0
    int arr4[5];                 // 未初始化（值不确定）

    // sizeof 计算数组大小
    int len1 = sizeof(arr1) / sizeof(arr1[0]);
    printf("arr1 长度: %d\n", len1);
    printf("arr1: "); print_array(arr1, len1);

    // 访问与修改
    arr1[0] = 99;
    printf("修改arr1[0]后: "); print_array(arr1, len1);

    // 【注意】C 不检查数组越界！
    // arr1[10] = 999;  // 未定义行为！可能崩溃或覆盖其他数据

    // 变长数组（VLA，C99）—— 运行时确定大小
    int n = 5;
    int vla[n];
    for (int i = 0; i < n; i++) vla[i] = i * i;
    printf("VLA: "); print_array(vla, n);

    // 排序与搜索
    printf("\n=== 排序与搜索 ===\n");
    int nums[] = {64, 34, 25, 12, 22, 11, 90};
    int num_len = sizeof(nums) / sizeof(nums[0]);

    printf("排序前: "); print_array(nums, num_len);
    bubble_sort(nums, num_len);
    printf("排序后: "); print_array(nums, num_len);

    int idx = binary_search(nums, num_len, 25);
    printf("二分查找 25: 索引 %d\n", idx);
    printf("二分查找 99: 索引 %d（未找到）\n", binary_search(nums, num_len, 99));

    // ============================================================
    //                      二维数组
    // ============================================================
    printf("\n=== 二维数组 ===\n");

    int matrix[3][4] = {
        {1,  2,  3,  4},
        {5,  6,  7,  8},
        {9, 10, 11, 12}
    };

    // 遍历
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%3d ", matrix[i][j]);
        }
        printf("\n");
    }

    // 矩阵转置
    printf("转置:\n");
    int transposed[4][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            transposed[j][i] = matrix[i][j];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++)
            printf("%3d ", transposed[i][j]);
        printf("\n");
    }

    // ============================================================
    //                      字符串
    // ============================================================
    printf("\n=== 字符串 ===\n");

    // 字符串是 char 数组，以 '\0' 结尾
    char s1[] = "Hello";         // 自动添加 '\0'，占 6 字节
    char s2[20] = "World";       // 固定大小缓冲区
    const char *s3 = "Literal";  // 字符串字面量（只读！）

    printf("s1 = %s，长度 = %zu\n", s1, strlen(s1));
    printf("sizeof(s1) = %zu（含 '\\0'）\n", sizeof(s1));

    // 字符串字面量存储在只读内存段
    // s3[0] = 'l';  // 危险！未定义行为

    // ----------------------------------------------------------
    // string.h 常用函数
    // ----------------------------------------------------------
    printf("\n=== string.h 函数 ===\n");

    // strlen：字符串长度（不含 '\0'）
    printf("strlen(\"Hello\") = %zu\n", strlen("Hello"));

    // strcpy / strncpy：复制字符串
    char dest[50];
    strcpy(dest, "Hello");
    printf("strcpy: %s\n", dest);

    // 【安全】strncpy 限制复制长度，防止缓冲区溢出
    char safe_dest[10];
    strncpy(safe_dest, "LongString", sizeof(safe_dest) - 1);
    safe_dest[sizeof(safe_dest) - 1] = '\0';  // 确保以 \0 结尾
    printf("strncpy: %s\n", safe_dest);

    // strcat / strncat：拼接字符串
    char buf[50] = "Hello";
    strcat(buf, ", World!");
    printf("strcat: %s\n", buf);

    // strcmp / strncmp：比较字符串
    printf("strcmp(\"abc\", \"abc\") = %d\n", strcmp("abc", "abc"));
    printf("strcmp(\"abc\", \"abd\") = %d（负数）\n", strcmp("abc", "abd"));
    printf("strcmp(\"abd\", \"abc\") = %d（正数）\n", strcmp("abd", "abc"));

    // strchr / strstr：查找字符/子串
    char text[] = "Hello, World!";
    char *found = strchr(text, 'W');
    printf("strchr('W'): %s\n", found);

    char *sub = strstr(text, "World");
    printf("strstr(\"World\"): %s\n", sub);

    // sprintf：格式化字符串到 char 数组
    char formatted[100];
    int age = 25;
    sprintf(formatted, "姓名: 张三，年龄: %d", age);
    printf("sprintf: %s\n", formatted);

    // 字符串转数字
    printf("\n=== 字符串转换 ===\n");
    int num = atoi("42");
    double fnum = atof("3.14");
    printf("atoi(\"42\") = %d\n", num);
    printf("atof(\"3.14\") = %.2f\n", fnum);

    // strtol：更安全的转换（可检测错误）
    char *endptr;
    long lnum = strtol("123abc", &endptr, 10);
    printf("strtol(\"123abc\"): 值=%ld, 剩余=\"%s\"\n", lnum, endptr);

    // ----------------------------------------------------------
    // 字符串自定义函数
    // ----------------------------------------------------------
    printf("\n=== 字符串操作 ===\n");

    char rev[] = "Hello, World!";
    str_reverse(rev);
    printf("反转: %s\n", rev);

    printf("\"racecar\" 是回文: %s\n", is_palindrome("racecar") ? "是" : "否");
    printf("\"hello\" 是回文: %s\n", is_palindrome("hello") ? "是" : "否");

    // 字符频率统计
    const char *sentence = "Hello World";
    int freq[256];
    char_frequency(sentence, freq);
    printf("字符频率（\"%s\"）:\n", sentence);
    for (int i = 0; i < 256; i++) {
        if (freq[i] > 0 && i != ' ')
            printf("  '%c': %d\n", i, freq[i]);
    }

    // ctype.h 字符函数
    printf("\n=== ctype.h ===\n");
    char test_chars[] = "Hello World 123!";
    printf("原文: %s\n", test_chars);
    printf("转大写: ");
    for (int i = 0; test_chars[i]; i++)
        printf("%c", toupper(test_chars[i]));
    printf("\n");

    // 统计字母、数字、空格
    int alpha = 0, digit = 0, space = 0;
    for (int i = 0; test_chars[i]; i++) {
        if (isalpha(test_chars[i])) alpha++;
        else if (isdigit(test_chars[i])) digit++;
        else if (isspace(test_chars[i])) space++;
    }
    printf("字母=%d, 数字=%d, 空格=%d\n", alpha, digit, space);

    printf("\n=== 数组与字符串演示完成 ===\n");
    return 0;
}
```
