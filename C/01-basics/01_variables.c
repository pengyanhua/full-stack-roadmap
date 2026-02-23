// ============================================================
//                      变量与数据类型
// ============================================================
// C 是静态类型语言，所有变量必须先声明后使用
// C 是系统级语言：直接操作内存，无垃圾回收，性能极高
// 标准：C89/C90、C99（推荐）、C11、C17
// 编译：gcc -std=c99 -Wall -o program 01_variables.c

#include <stdio.h>    // 标准输入输出
#include <stdint.h>   // 固定宽度整数类型（C99）
#include <stdbool.h>  // bool 类型（C99）
#include <limits.h>   // 整数类型极值
#include <float.h>    // 浮点类型极值

int main(void) {
    printf("=== C 变量与数据类型 ===\n");

    // ----------------------------------------------------------
    // 1. 整数类型
    // ----------------------------------------------------------
    // 【重要】C 标准只规定最小位宽，实际大小依赖平台
    // sizeof 运算符返回类型所占字节数

    char   c  = 'A';          // 字符，1字节，-128~127 或 0~255
    short  s  = 32767;        // 短整数，至少16位
    int    i  = 2147483647;   // 标准整数，至少16位（通常32位）
    long   l  = 2147483647L;  // 长整数，至少32位
    long long ll = 9223372036854775807LL; // 至少64位（C99）

    printf("char:      %d 字节，值=%c\n", (int)sizeof(c), c);
    printf("short:     %d 字节，值=%d\n", (int)sizeof(s), s);
    printf("int:       %d 字节，值=%d\n", (int)sizeof(i), i);
    printf("long:      %d 字节，值=%ld\n", (int)sizeof(l), l);
    printf("long long: %d 字节，值=%lld\n", (int)sizeof(ll), ll);

    // 无符号类型（unsigned）：只存正数，范围是有符号的两倍
    unsigned int ui = 4294967295U;  // U 后缀
    unsigned long long ull = 18446744073709551615ULL;
    printf("unsigned int:       %u\n", ui);
    printf("unsigned long long: %llu\n", ull);

    // ----------------------------------------------------------
    // 2. 固定宽度整数（C99，推荐用于跨平台代码）
    // ----------------------------------------------------------
    // 【推荐】当需要精确宽度时，使用 stdint.h 中的类型
    printf("\n=== 固定宽度整数（stdint.h）===\n");

    int8_t   i8  = 127;
    int16_t  i16 = 32767;
    int32_t  i32 = 2147483647;
    int64_t  i64 = 9223372036854775807LL;
    uint8_t  u8  = 255;
    uint32_t u32 = 4294967295U;

    printf("int8_t:   %d（1字节）\n", i8);
    printf("int16_t:  %d（2字节）\n", i16);
    printf("int32_t:  %d（4字节）\n", i32);
    printf("int64_t:  %lld（8字节）\n", i64);
    printf("uint8_t:  %u\n", u8);
    printf("uint32_t: %u\n", u32);

    // ----------------------------------------------------------
    // 3. 浮点类型
    // ----------------------------------------------------------
    printf("\n=== 浮点类型 ===\n");

    float  f   = 3.14f;               // 单精度，32位，约7位有效数字
    double d   = 3.141592653589793;   // 双精度，64位，约15位有效数字（默认）
    long double ld = 3.14159265358979323846L; // 扩展精度，80或128位

    printf("float:       %.7f（%d字节）\n", f, (int)sizeof(f));
    printf("double:      %.15f（%d字节）\n", d, (int)sizeof(d));
    printf("long double: %.18Lf（%d字节）\n", ld, (int)sizeof(ld));

    // 【警告】浮点精度问题
    double a = 0.1, b = 0.2;
    printf("0.1 + 0.2 = %.17f（不精确！）\n", a + b);
    printf("0.1 + 0.2 == 0.3：%s\n", (a + b == 0.3) ? "true" : "false");

    // ----------------------------------------------------------
    // 4. bool 类型（C99，需要 stdbool.h）
    // ----------------------------------------------------------
    printf("\n=== bool 类型 ===\n");

    bool is_valid = true;
    bool is_empty = false;
    printf("is_valid: %d，is_empty: %d\n", is_valid, is_empty);
    // 【注意】C 中 bool 本质是整数，0 为假，非 0 为真

    // ----------------------------------------------------------
    // 5. 变量声明与初始化
    // ----------------------------------------------------------
    printf("\n=== 变量声明 ===\n");

    // C89：变量必须在代码块开头声明
    // C99+：可以在使用前任意位置声明

    int x;          // 未初始化（局部变量值不确定，危险！）
    int y = 10;     // 声明并初始化
    int z = y * 2;  // 用表达式初始化

    // 【最佳实践】始终初始化变量
    x = 42;
    printf("x=%d, y=%d, z=%d\n", x, y, z);

    // 多变量声明（不推荐，降低可读性）
    int m = 1, n = 2, p = 3;
    printf("m=%d, n=%d, p=%d\n", m, n, p);

    // ----------------------------------------------------------
    // 6. 常量
    // ----------------------------------------------------------
    printf("\n=== 常量 ===\n");

    // const 修饰符：运行时常量（值不可修改）
    const double PI = 3.141592653589793;
    const int MAX_SIZE = 100;
    // PI = 3.14;  // 错误！const 变量不能修改

    // #define 宏：编译时文本替换
    #define GRAVITY 9.81
    #define APP_NAME "C 学习系统"

    printf("PI = %f\n", PI);
    printf("MAX_SIZE = %d\n", MAX_SIZE);
    printf("GRAVITY = %.2f\n", GRAVITY);
    printf("APP_NAME = %s\n", APP_NAME);

    // 枚举常量
    enum Day { MON=1, TUE, WED, THU, FRI, SAT, SUN };
    enum Day today = WED;
    printf("今天是第 %d 天\n", today);

    // ----------------------------------------------------------
    // 7. 字面量类型后缀
    // ----------------------------------------------------------
    printf("\n=== 字面量 ===\n");

    int dec  = 255;      // 十进制
    int oct  = 0377;     // 八进制（0 前缀）
    int hex  = 0xFF;     // 十六进制（0x 前缀）
    printf("十进制=%d, 八进制=%d, 十六进制=%d（同一个数）\n", dec, oct, hex);

    // 字符字面量（转义字符）
    char nl = '\n';   // 换行
    char tb = '\t';   // 制表符
    char bs = '\\';   // 反斜杠
    char sq = '\'';   // 单引号
    char nul = '\0';  // 空字符（字符串终止符）
    printf("转义：换行[%c]制表[%c]反斜杠[%c]单引号[%c]\n", nl, tb, bs, sq);

    // ----------------------------------------------------------
    // 8. 类型转换
    // ----------------------------------------------------------
    printf("\n=== 类型转换 ===\n");

    // 隐式转换（整数提升）
    char ch = 65;
    int promoted = ch;  // char 自动提升为 int
    printf("char %d -> int %d，字符 '%c'\n", ch, promoted, ch);

    // 混合类型运算
    int  dividend = 7;
    int  divisor  = 2;
    double result = dividend / divisor;   // 整数除法！结果 3.0
    double result2 = (double)dividend / divisor;  // 强制转换，结果 3.5
    printf("7/2（整数除法）= %.1f\n", result);
    printf("(double)7/2  = %.1f\n", result2);

    // 显式强制转换
    double pi = 3.99;
    int truncated = (int)pi;  // 截断小数，不是四舍五入
    printf("(int)3.99 = %d（截断）\n", truncated);

    printf("\n=== 变量与类型演示完成 ===\n");
    return 0;
}
