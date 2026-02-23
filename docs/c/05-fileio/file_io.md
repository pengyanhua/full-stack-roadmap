# file io.c

::: info 文件信息
- 📄 原文件：`01_file_io.c`
- 🔤 语言：c
:::

## 完整代码

```c
// ============================================================
//                      文件 I/O
// ============================================================
// C 通过 FILE* 句柄和 stdio.h 函数操作文件
// 文件模式："r"只读、"w"写入（覆盖）、"a"追加
//          "rb"/"wb"二进制模式，"r+"读写
// 【原则】打开文件后必须检查是否成功，用完必须关闭

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

// 临时文件路径
#define TEMP_FILE  "temp_test.txt"
#define TEMP_BIN   "temp_test.bin"

// 结构体（用于二进制文件读写）
typedef struct {
    int    id;
    char   name[32];
    double score;
} Student;

// ============================================================
//                      文本文件操作
// ============================================================

void demo_text_write(void) {
    printf("=== 写入文本文件 ===\n");

    // fopen 打开文件，返回 FILE* 指针
    FILE *fp = fopen(TEMP_FILE, "w");
    if (fp == NULL) {
        fprintf(stderr, "打开文件失败: %s\n", strerror(errno));
        return;
    }

    // fprintf：格式化写入（类似 printf）
    fprintf(fp, "# 学生成绩单\n");
    fprintf(fp, "ID,姓名,分数\n");

    Student students[] = {
        {1, "张三", 95.5},
        {2, "李四", 87.0},
        {3, "王五", 92.5},
    };
    int n = sizeof(students) / sizeof(students[0]);

    for (int i = 0; i < n; i++) {
        fprintf(fp, "%d,%s,%.1f\n",
                students[i].id, students[i].name, students[i].score);
    }

    // fputs：写入字符串
    fputs("# End of File\n", fp);

    // fclose：关闭文件（刷新缓冲区）
    fclose(fp);
    printf("文件已写入: %s\n", TEMP_FILE);
}

void demo_text_read(void) {
    printf("\n=== 读取文本文件 ===\n");

    FILE *fp = fopen(TEMP_FILE, "r");
    if (fp == NULL) {
        fprintf(stderr, "打开文件失败: %s\n", strerror(errno));
        return;
    }

    // 方法1：逐行读取（fgets）
    printf("--- fgets 逐行读取 ---\n");
    char line[256];
    while (fgets(line, sizeof(line), fp) != NULL) {
        // fgets 保留换行符，去掉它
        line[strcspn(line, "\n")] = '\0';
        printf("  |%s|\n", line);
    }
    fclose(fp);

    // 方法2：fscanf 格式化读取
    printf("\n--- fscanf 格式化读取 ---\n");
    fp = fopen(TEMP_FILE, "r");

    char buf[256];
    // 跳过前两行（注释行和表头）
    fgets(buf, sizeof(buf), fp);
    fgets(buf, sizeof(buf), fp);

    int id;
    char name[32];
    double score;
    while (fscanf(fp, "%d,%31[^,],%lf\n", &id, name, &score) == 3) {
        printf("  ID=%d, 姓名=%s, 分数=%.1f\n", id, name, score);
    }
    fclose(fp);
}

// ============================================================
//                      二进制文件操作
// ============================================================

void demo_binary_write(void) {
    printf("\n=== 写入二进制文件 ===\n");

    FILE *fp = fopen(TEMP_BIN, "wb");
    if (fp == NULL) {
        fprintf(stderr, "打开二进制文件失败\n");
        return;
    }

    Student students[] = {
        {1, "张三", 95.5},
        {2, "李四", 87.0},
        {3, "王五", 92.5},
        {4, "赵六", 78.5},
    };
    int n = sizeof(students) / sizeof(students[0]);

    // 写入记录数
    fwrite(&n, sizeof(int), 1, fp);

    // fwrite：按块写入二进制数据
    // fwrite(数据指针, 每块大小, 块数, 文件指针)
    fwrite(students, sizeof(Student), n, fp);

    fclose(fp);
    printf("写入 %d 条学生记录（二进制格式）\n", n);
    printf("文件大小: %zu 字节\n", sizeof(int) + n * sizeof(Student));
}

void demo_binary_read(void) {
    printf("\n=== 读取二进制文件 ===\n");

    FILE *fp = fopen(TEMP_BIN, "rb");
    if (fp == NULL) {
        fprintf(stderr, "打开二进制文件失败\n");
        return;
    }

    // 读取记录数
    int n;
    fread(&n, sizeof(int), 1, fp);
    printf("共 %d 条记录:\n", n);

    // fread：按块读取
    Student *students = (Student*)malloc(n * sizeof(Student));
    if (students == NULL) {
        fclose(fp);
        return;
    }

    size_t read = fread(students, sizeof(Student), n, fp);
    printf("实际读取 %zu 条\n", read);

    for (int i = 0; i < (int)read; i++) {
        printf("  [%d] %s: %.1f\n", students[i].id, students[i].name, students[i].score);
    }

    free(students);
    fclose(fp);
}

// ============================================================
//                      文件定位与随机访问
// ============================================================

void demo_seek(void) {
    printf("\n=== 文件定位（随机访问）===\n");

    FILE *fp = fopen(TEMP_BIN, "rb");
    if (fp == NULL) return;

    // ftell：获取当前位置
    printf("初始位置: %ld\n", ftell(fp));

    // fseek：移动文件指针
    // SEEK_SET：从头，SEEK_CUR：从当前，SEEK_END：从尾
    fseek(fp, sizeof(int), SEEK_SET);  // 跳过记录数字段

    // 直接读取第3条记录（索引2）
    int record_index = 2;
    fseek(fp, sizeof(int) + record_index * sizeof(Student), SEEK_SET);
    printf("跳转到第%d条记录，位置: %ld\n", record_index + 1, ftell(fp));

    Student s;
    fread(&s, sizeof(Student), 1, fp);
    printf("第%d条: ID=%d, 姓名=%s, 分数=%.1f\n",
           record_index + 1, s.id, s.name, s.score);

    // 获取文件大小
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    printf("文件大小: %ld 字节\n", file_size);

    fclose(fp);
}

// ============================================================
//                      标准流
// ============================================================

void demo_stdio(void) {
    printf("\n=== 标准流 ===\n");

    // stdin：标准输入
    // stdout：标准输出
    // stderr：标准错误（不缓冲，立即输出）

    // 写到 stderr（错误信息）
    fprintf(stderr, "这是错误信息（写到 stderr）\n");

    // fflush：强制刷新缓冲区
    printf("刷新缓冲区中...");
    fflush(stdout);
    printf("完成\n");

    // getchar / putchar（单字符 I/O）
    // printf("请输入一个字符: ");
    // int ch = getchar();
    // printf("你输入了: %c\n", ch);
}

// ============================================================
//                      主函数
// ============================================================

int main(void) {
    printf("=== C 文件 I/O 演示 ===\n");

    // 文本文件
    demo_text_write();
    demo_text_read();

    // 二进制文件
    demo_binary_write();
    demo_binary_read();

    // 文件定位
    demo_seek();

    // 标准流
    demo_stdio();

    // 清理临时文件
    remove(TEMP_FILE);
    remove(TEMP_BIN);
    printf("\n临时文件已清理\n");

    // ----------------------------------------------------------
    // 文件操作最佳实践
    // ----------------------------------------------------------
    printf("\n=== 最佳实践 ===\n");
    printf("1. 始终检查 fopen 返回值是否为 NULL\n");
    printf("2. 始终在 return 前 fclose 文件\n");
    printf("3. 文本模式用 fgets（避免 gets：缓冲区溢出危险）\n");
    printf("4. 二进制模式用 fread/fwrite\n");
    printf("5. 用 ferror 检查读写错误，feof 检查文件末尾\n");
    printf("6. 大文件操作用 fseek/ftell 实现随机访问\n");

    printf("\n=== 文件 I/O 演示完成 ===\n");
    return 0;
}
```
