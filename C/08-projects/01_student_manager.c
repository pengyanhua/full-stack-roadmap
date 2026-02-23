// ============================================================
//                      项目实战：学生成绩管理系统
// ============================================================
// 综合运用 C 语言核心特性：
//   结构体、动态内存、文件 I/O、排序、指针、函数指针
// 功能：添加/查询/删除/排序/统计/保存/加载学生记录
// 编译：gcc -std=c99 -Wall -o student_manager 01_student_manager.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// ============================================================
//                      数据定义
// ============================================================

#define MAX_NAME_LEN   32
#define MAX_SUBJECT    5
#define DATA_FILE      "students.dat"

typedef struct {
    int    id;
    char   name[MAX_NAME_LEN];
    double scores[MAX_SUBJECT];  // 各科成绩
    double average;              // 平均分
} Student;

typedef struct {
    Student *data;    // 动态数组
    int      size;    // 当前学生数
    int      cap;     // 容量
    int      next_id; // 下一个 ID
} StudentDB;

static const char *SUBJECTS[] = {"语文", "数学", "英语", "物理", "化学"};

// ============================================================
//                      数据库操作
// ============================================================

StudentDB *db_create(int initial_cap) {
    StudentDB *db = (StudentDB*)malloc(sizeof(StudentDB));
    if (!db) return NULL;
    db->data = (Student*)malloc(initial_cap * sizeof(Student));
    if (!db->data) { free(db); return NULL; }
    db->size = 0;
    db->cap = initial_cap;
    db->next_id = 1;
    return db;
}

void db_destroy(StudentDB *db) {
    if (db) { free(db->data); free(db); }
}

// 扩容
static int db_grow(StudentDB *db) {
    int new_cap = db->cap * 2;
    Student *new_data = (Student*)realloc(db->data, new_cap * sizeof(Student));
    if (!new_data) return 0;
    db->data = new_data;
    db->cap = new_cap;
    return 1;
}

// 计算平均分
static double calc_average(const double scores[], int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += scores[i];
    return sum / n;
}

// 添加学生
int db_add(StudentDB *db, const char *name, const double scores[]) {
    if (db->size == db->cap && !db_grow(db)) return -1;

    Student *s = &db->data[db->size++];
    s->id = db->next_id++;
    strncpy(s->name, name, MAX_NAME_LEN - 1);
    s->name[MAX_NAME_LEN - 1] = '\0';
    memcpy(s->scores, scores, MAX_SUBJECT * sizeof(double));
    s->average = calc_average(scores, MAX_SUBJECT);

    return s->id;
}

// 按 ID 查找
Student *db_find_by_id(StudentDB *db, int id) {
    for (int i = 0; i < db->size; i++)
        if (db->data[i].id == id) return &db->data[i];
    return NULL;
}

// 按姓名查找（返回所有匹配）
int db_find_by_name(StudentDB *db, const char *name, Student *results[], int max) {
    int cnt = 0;
    for (int i = 0; i < db->size && cnt < max; i++) {
        if (strstr(db->data[i].name, name)) {
            results[cnt++] = &db->data[i];
        }
    }
    return cnt;
}

// 按 ID 删除
int db_delete(StudentDB *db, int id) {
    for (int i = 0; i < db->size; i++) {
        if (db->data[i].id == id) {
            // 将最后一个元素移到此位置（O(1) 删除，不保持顺序）
            db->data[i] = db->data[--db->size];
            return 1;
        }
    }
    return 0;
}

// ============================================================
//                      排序（函数指针）
// ============================================================

typedef int (*CmpFn)(const void *, const void *);

int cmp_by_id(const void *a, const void *b) {
    return ((Student*)a)->id - ((Student*)b)->id;
}

int cmp_by_name(const void *a, const void *b) {
    return strcmp(((Student*)a)->name, ((Student*)b)->name);
}

int cmp_by_average_desc(const void *a, const void *b) {
    double da = ((Student*)a)->average;
    double db = ((Student*)b)->average;
    if (da > db) return -1;
    if (da < db) return 1;
    return 0;
}

void db_sort(StudentDB *db, CmpFn cmp) {
    qsort(db->data, db->size, sizeof(Student), cmp);
}

// ============================================================
//                      统计
// ============================================================

typedef struct {
    double min, max, avg;
    int    min_id, max_id;
    int    pass_count;    // 平均分 >= 60
    int    excel_count;   // 平均分 >= 90
} Statistics;

Statistics db_stats(const StudentDB *db) {
    Statistics st = {0};
    if (db->size == 0) return st;

    st.min = st.max = db->data[0].average;
    st.min_id = st.max_id = db->data[0].id;

    double sum = 0;
    for (int i = 0; i < db->size; i++) {
        double avg = db->data[i].average;
        sum += avg;
        if (avg < st.min) { st.min = avg; st.min_id = db->data[i].id; }
        if (avg > st.max) { st.max = avg; st.max_id = db->data[i].id; }
        if (avg >= 60) st.pass_count++;
        if (avg >= 90) st.excel_count++;
    }
    st.avg = sum / db->size;
    return st;
}

// ============================================================
//                      文件持久化
// ============================================================

int db_save(const StudentDB *db, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return 0;

    fwrite(&db->size, sizeof(int), 1, fp);
    fwrite(&db->next_id, sizeof(int), 1, fp);
    fwrite(db->data, sizeof(Student), db->size, fp);
    fclose(fp);
    return 1;
}

int db_load(StudentDB *db, const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return 0;

    int size, next_id;
    fread(&size, sizeof(int), 1, fp);
    fread(&next_id, sizeof(int), 1, fp);

    // 确保容量足够
    if (size > db->cap) {
        Student *nd = (Student*)realloc(db->data, size * sizeof(Student));
        if (!nd) { fclose(fp); return 0; }
        db->data = nd;
        db->cap = size;
    }

    fread(db->data, sizeof(Student), size, fp);
    db->size = size;
    db->next_id = next_id;
    fclose(fp);
    return 1;
}

// ============================================================
//                      打印函数
// ============================================================

void print_header(void) {
    printf("%-4s %-10s", "ID", "姓名");
    for (int i = 0; i < MAX_SUBJECT; i++)
        printf(" %-6s", SUBJECTS[i]);
    printf(" %-8s\n", "平均分");
    printf("%s\n", "----+----------+------+------+------+------+------+--------");
}

void print_student(const Student *s) {
    printf("%-4d %-10s", s->id, s->name);
    for (int i = 0; i < MAX_SUBJECT; i++)
        printf(" %-6.1f", s->scores[i]);
    printf(" %-8.2f\n", s->average);
}

void print_all(StudentDB *db) {
    printf("\n共 %d 名学生:\n", db->size);
    print_header();
    db_sort(db, cmp_by_id);
    for (int i = 0; i < db->size; i++)
        print_student(&db->data[i]);
}

// ============================================================
//                      主函数（演示）
// ============================================================

int main(void) {
    printf("=== 学生成绩管理系统 ===\n\n");

    StudentDB *db = db_create(8);
    assert(db != NULL);

    // 添加学生
    double scores[][MAX_SUBJECT] = {
        {92, 88, 95, 78, 85},
        {75, 82, 70, 68, 73},
        {98, 95, 92, 96, 94},
        {60, 55, 65, 58, 62},
        {88, 91, 85, 90, 87},
    };
    const char *names[] = {"张三", "李四", "王五", "赵六", "钱七"};
    int n_students = sizeof(names) / sizeof(names[0]);

    printf("--- 添加学生 ---\n");
    for (int i = 0; i < n_students; i++) {
        int id = db_add(db, names[i], scores[i]);
        printf("添加: ID=%d，%s\n", id, names[i]);
    }

    // 显示所有学生
    print_all(db);

    // 按平均分排序
    printf("\n--- 按平均分降序 ---\n");
    db_sort(db, cmp_by_average_desc);
    print_header();
    for (int i = 0; i < db->size; i++) {
        printf("第%d名: ", i+1);
        print_student(&db->data[i]);
    }

    // 查找
    printf("\n--- 查找 ---\n");
    Student *found = db_find_by_id(db, 3);
    if (found) {
        printf("ID=3: ");
        print_student(found);
    }

    Student *results[10];
    int cnt = db_find_by_name(db, "三", results, 10);
    printf("姓名含'三': 找到 %d 条\n", cnt);
    for (int i = 0; i < cnt; i++) print_student(results[i]);

    // 删除
    printf("\n--- 删除 ID=4 ---\n");
    if (db_delete(db, 4)) printf("删除成功\n");
    print_all(db);

    // 统计
    printf("\n--- 统计信息 ---\n");
    Statistics st = db_stats(db);
    printf("最高平均分: %.2f（ID=%d）\n", st.max, st.max_id);
    printf("最低平均分: %.2f（ID=%d）\n", st.min, st.min_id);
    printf("全体平均分: %.2f\n", st.avg);
    printf("及格人数（>=60）: %d/%d\n", st.pass_count, db->size);
    printf("优秀人数（>=90）: %d/%d\n", st.excel_count, db->size);

    // 保存和加载
    printf("\n--- 文件持久化 ---\n");
    if (db_save(db, DATA_FILE)) printf("保存到 %s 成功\n", DATA_FILE);

    StudentDB *db2 = db_create(4);
    if (db_load(db2, DATA_FILE)) {
        printf("从文件加载 %d 条记录\n", db2->size);
        print_all(db2);
    }

    // 清理
    db_destroy(db);
    db_destroy(db2);
    remove(DATA_FILE);

    printf("\n=== 学生管理系统演示完成 ===\n");
    return 0;
}
