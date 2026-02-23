// ============================================================
//                      流程控制
// ============================================================
// C 的流程控制与大多数语言相同：if/else、switch、for、while
// C 特有：goto 语句（可用于跳出多层嵌套，需谨慎使用）
// 条件表达式：0 为假，任意非 0 值为真

#include <stdio.h>
#include <stdbool.h>

int main(void) {
    printf("=== 条件语句 ===\n");

    // ----------------------------------------------------------
    // 1. if / else if / else
    // ----------------------------------------------------------
    int score = 85;

    if (score >= 90) {
        printf("优秀\n");
    } else if (score >= 80) {
        printf("良好\n");       // 命中此分支
    } else if (score >= 60) {
        printf("及格\n");
    } else {
        printf("不及格\n");
    }

    // 三元运算符（条件表达式）
    const char *result = (score >= 60) ? "通过" : "未通过";
    printf("结果: %s\n", result);

    // C 中任何非 0 值都是真
    int x = 5;
    if (x)       printf("x 非零（真）\n");
    if (!0)      printf("!0 为真\n");
    if (x > 3)   printf("x > 3\n");

    // ----------------------------------------------------------
    // 2. switch 语句
    // ----------------------------------------------------------
    printf("\n=== switch ===\n");

    int day = 3;
    switch (day) {
        case 1:
            printf("周一\n");
            break;  // 必须 break，否则会贯穿（fall-through）
        case 2:
            printf("周二\n");
            break;
        case 3:
        case 4:
        case 5:
            // 多个 case 共享同一处理逻辑
            printf("工作日（第 %d 天）\n", day);
            break;
        case 6:
        case 7:
            printf("周末\n");
            break;
        default:
            printf("无效\n");
    }

    // 【特性】有意为之的贯穿（fall-through）
    printf("\n贯穿示例（case 1 -> 2 -> 3）:\n");
    int val = 1;
    switch (val) {
        case 1: printf("  case 1\n"); /* 故意不 break */
        case 2: printf("  case 2\n"); /* 故意不 break */
        case 3: printf("  case 3\n"); break;
        default: printf("  default\n");
    }

    // ----------------------------------------------------------
    // 3. for 循环
    // ----------------------------------------------------------
    printf("\n=== for 循环 ===\n");

    // 标准 for 循环
    for (int i = 0; i < 5; i++) {
        printf("%d ", i);
    }
    printf("\n");

    // 倒序
    for (int i = 5; i > 0; i--) {
        printf("%d ", i);
    }
    printf("\n");

    // 步长为 2
    for (int i = 0; i <= 10; i += 2) {
        printf("%d ", i);
    }
    printf("\n");

    // 嵌套循环（九九乘法表）
    printf("\n九九乘法表:\n");
    for (int i = 1; i <= 9; i++) {
        for (int j = 1; j <= i; j++) {
            printf("%d×%d=%-3d", j, i, i * j);
        }
        printf("\n");
    }

    // ----------------------------------------------------------
    // 4. while 循环
    // ----------------------------------------------------------
    printf("\n=== while ===\n");

    int n = 1;
    while (n <= 5) {
        printf("%d ", n++);  // n++ 后置递增
    }
    printf("\n");

    // 无限循环（常见于嵌入式/服务器）
    // while (1) { ... }  // 或 for (;;) { ... }

    // ----------------------------------------------------------
    // 5. do-while 循环（至少执行一次）
    // ----------------------------------------------------------
    printf("\n=== do-while ===\n");

    int m = 0;
    do {
        printf("%d ", m);
        m++;
    } while (m < 3);
    printf("\n");

    // 【用途】用于需要先执行后判断的场景（如菜单选择）
    // do {
    //     printf("请选择（1-3）: ");
    //     scanf("%d", &choice);
    // } while (choice < 1 || choice > 3);

    // ----------------------------------------------------------
    // 6. break / continue / goto
    // ----------------------------------------------------------
    printf("\n=== break / continue ===\n");

    // break：跳出当前循环
    for (int i = 0; i < 10; i++) {
        if (i == 5) break;
        printf("%d ", i);
    }
    printf("(break at 5)\n");

    // continue：跳过本次迭代
    for (int i = 0; i < 10; i++) {
        if (i % 2 == 0) continue;  // 跳过偶数
        printf("%d ", i);
    }
    printf("(奇数)\n");

    // break 只跳出一层循环，使用标志变量或 goto 跳出多层
    printf("\n跳出多层循环（使用 goto）:\n");
    bool found = false;
    for (int i = 0; i < 5 && !found; i++) {
        for (int j = 0; j < 5; j++) {
            if (i * j == 6) {
                printf("找到：%d × %d = 6\n", i, j);
                found = true;
                break;
            }
        }
    }

    // goto（C 的特殊跳转，错误处理时常用）
    printf("\ngoto 示例（资源清理模式）:\n");
    int *ptr = NULL;
    // 模拟分配失败场景
    if (ptr == NULL) {
        printf("  分配失败，跳到清理代码\n");
        goto cleanup;  // 跳转到 cleanup 标签
    }
    printf("  这行不会执行\n");

cleanup:
    printf("  执行清理操作（goto 目标）\n");
    // 在实际代码中这里释放资源：free(ptr);

    // ----------------------------------------------------------
    // 7. 逻辑运算符
    // ----------------------------------------------------------
    printf("\n=== 逻辑运算符 ===\n");

    int age = 25;
    bool has_id = true;

    // && (AND)、|| (OR)、! (NOT)
    if (age >= 18 && has_id) {
        printf("可以入场\n");
    }

    // 短路求值：&& 左侧为假时右侧不执行，|| 左侧为真时右侧不执行
    int divisor = 0;
    if (divisor != 0 && 10 / divisor > 2) {
        printf("条件成立\n");
    } else {
        printf("安全避免了除零（短路求值）\n");
    }

    // 位运算符（作用于整数的每一位）
    printf("\n=== 位运算 ===\n");
    unsigned int flags = 0;
    unsigned int FLAG_READ    = 0x01;  // 0001
    unsigned int FLAG_WRITE   = 0x02;  // 0010
    unsigned int FLAG_EXECUTE = 0x04;  // 0100

    flags |= FLAG_READ | FLAG_WRITE;  // 设置位：OR
    printf("设置读写权限: 0x%02X\n", flags);

    if (flags & FLAG_READ) printf("可读\n");
    if (flags & FLAG_WRITE) printf("可写\n");
    if (!(flags & FLAG_EXECUTE)) printf("不可执行\n");

    flags &= ~FLAG_WRITE;  // 清除位：AND NOT
    printf("清除写权限后: 0x%02X\n", flags);

    flags ^= FLAG_EXECUTE;  // 翻转位：XOR
    printf("翻转执行位: 0x%02X\n", flags);

    // 移位运算
    printf("\n移位运算:\n");
    int a = 1;
    printf("1 << 3 = %d（乘以8）\n", a << 3);
    printf("16 >> 2 = %d（除以4）\n", 16 >> 2);

    printf("\n=== 流程控制演示完成 ===\n");
    return 0;
}
