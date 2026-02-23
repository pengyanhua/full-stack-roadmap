// ============================================================
//                      动态内存管理
// ============================================================
// C 手动管理堆内存：malloc/calloc/realloc/free
// 【黄金法则】每个 malloc 必须有对应的 free
// 内存错误：泄漏、悬空指针、双重释放、越界访问
// Valgrind 是检测内存错误的利器

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================
//                      动态数组实现
// ============================================================

typedef struct {
    int   *data;     // 指向堆内存
    size_t size;     // 当前元素数
    size_t capacity; // 当前容量
} DynArray;

DynArray *dynarray_create(size_t initial_cap) {
    DynArray *da = (DynArray*)malloc(sizeof(DynArray));
    if (!da) return NULL;

    da->data = (int*)malloc(initial_cap * sizeof(int));
    if (!da->data) {
        free(da);
        return NULL;
    }
    da->size = 0;
    da->capacity = initial_cap;
    return da;
}

// 动态扩容（类似 C++ vector）
int dynarray_push(DynArray *da, int val) {
    if (da->size == da->capacity) {
        size_t new_cap = da->capacity * 2;
        int *new_data = (int*)realloc(da->data, new_cap * sizeof(int));
        if (!new_data) return -1;  // 扩容失败

        da->data = new_data;
        da->capacity = new_cap;
    }
    da->data[da->size++] = val;
    return 0;
}

void dynarray_print(const DynArray *da) {
    printf("[");
    for (size_t i = 0; i < da->size; i++)
        printf("%d%s", da->data[i], (i < da->size-1) ? ", " : "");
    printf("] size=%zu, cap=%zu\n", da->size, da->capacity);
}

void dynarray_destroy(DynArray *da) {
    if (da) {
        free(da->data);  // 先释放数据
        free(da);        // 再释放结构体
    }
}

// ============================================================
//                      内存池（简单实现）
// ============================================================

#define POOL_SIZE 1024

typedef struct {
    char   buffer[POOL_SIZE];
    size_t used;
} MemoryPool;

void pool_init(MemoryPool *pool) {
    pool->used = 0;
}

void *pool_alloc(MemoryPool *pool, size_t size) {
    // 对齐到 8 字节边界
    size_t aligned = (size + 7) & ~7;
    if (pool->used + aligned > POOL_SIZE) return NULL;

    void *ptr = pool->buffer + pool->used;
    pool->used += aligned;
    return ptr;
}

// 池内存一次性全部释放（重置）
void pool_reset(MemoryPool *pool) {
    pool->used = 0;
}

// ============================================================
//                      内存布局演示
// ============================================================

// 全局变量（BSS 段 / Data 段）
static int global_var = 100;
static int zero_var;  // 未初始化，BSS 段，自动清零

int main(void) {
    printf("=== 内存区域 ===\n");

    // 栈（Stack）：局部变量，自动管理
    int stack_var = 42;

    // 堆（Heap）：动态分配，手动管理
    int *heap_var = (int*)malloc(sizeof(int));
    *heap_var = 99;

    printf("全局变量地址（Data段）:  %p\n", (void*)&global_var);
    printf("零初始化地址（BSS段）:   %p\n", (void*)&zero_var);
    printf("栈变量地址（Stack）:     %p\n", (void*)&stack_var);
    printf("堆变量地址（Heap）:      %p\n", (void*)heap_var);
    printf("main 函数地址（Text段）: %p\n", (void*)main);

    free(heap_var);
    heap_var = NULL;

    // ============================================================
    //                      malloc / calloc / realloc
    // ============================================================
    printf("\n=== malloc / calloc / realloc ===\n");

    // malloc：分配指定字节，内容未初始化
    int *arr = (int*)malloc(5 * sizeof(int));
    if (!arr) { fprintf(stderr, "malloc 失败\n"); return 1; }
    printf("malloc（未初始化）: ");
    // 必须手动初始化，否则值不确定
    for (int i = 0; i < 5; i++) arr[i] = i * 10;
    for (int i = 0; i < 5; i++) printf("%d ", arr[i]);
    printf("\n");

    // calloc：分配并清零（内容全为 0）
    int *zarr = (int*)calloc(5, sizeof(int));
    if (!zarr) { free(arr); return 1; }
    printf("calloc（全为零）: ");
    for (int i = 0; i < 5; i++) printf("%d ", zarr[i]);
    printf("\n");

    // realloc：调整大小（可能移动内存）
    int *big = (int*)realloc(arr, 10 * sizeof(int));
    if (!big) { free(arr); free(zarr); return 1; }
    arr = big;  // 更新指针（原指针可能已失效）
    for (int i = 5; i < 10; i++) arr[i] = i * 10;
    printf("realloc 到10个: ");
    for (int i = 0; i < 10; i++) printf("%d ", arr[i]);
    printf("\n");

    free(arr);   arr = NULL;
    free(zarr);  zarr = NULL;

    // ============================================================
    //                      动态数组演示
    // ============================================================
    printf("\n=== 动态数组 ===\n");

    DynArray *da = dynarray_create(4);
    printf("初始: "); dynarray_print(da);

    for (int i = 1; i <= 10; i++) {
        dynarray_push(da, i * i);
        if (i == 4 || i == 8 || i == 10)
            printf("push %d个后: ", i); dynarray_print(da);
    }

    dynarray_destroy(da);
    da = NULL;

    // ============================================================
    //                      二维动态数组
    // ============================================================
    printf("\n=== 二维动态数组 ===\n");

    int rows = 3, cols = 4;

    // 方法1：指针数组（每行独立分配）
    int **matrix = (int**)malloc(rows * sizeof(int*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(cols * sizeof(int));
        for (int j = 0; j < cols; j++)
            matrix[i][j] = i * cols + j;
    }

    printf("矩阵（指针数组）:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("%3d ", matrix[i][j]);
        printf("\n");
    }

    // 释放：先释放每行，再释放指针数组
    for (int i = 0; i < rows; i++) free(matrix[i]);
    free(matrix);

    // 方法2：一维数组模拟二维（连续内存，更高效）
    int *flat = (int*)malloc(rows * cols * sizeof(int));
    #define MAT(i,j) flat[(i)*cols + (j)]
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            MAT(i,j) = i * cols + j;

    printf("矩阵（连续内存）:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("%3d ", MAT(i,j));
        printf("\n");
    }
    free(flat);
    #undef MAT

    // ============================================================
    //                      内存池
    // ============================================================
    printf("\n=== 内存池 ===\n");

    MemoryPool pool;
    pool_init(&pool);

    // 从池中分配（零碎小分配效率高，无 malloc 系统调用）
    int *a = (int*)pool_alloc(&pool, sizeof(int));
    double *b = (double*)pool_alloc(&pool, sizeof(double));
    char *s = (char*)pool_alloc(&pool, 32);

    if (a && b && s) {
        *a = 42;
        *b = 3.14;
        strcpy(s, "池分配的字符串");
        printf("int=%d, double=%.2f, str=%s\n", *a, *b, s);
        printf("已使用: %zu / %d 字节\n", pool.used, POOL_SIZE);
    }

    pool_reset(&pool);  // 一次性全部释放
    printf("池已重置，已使用: %zu\n", pool.used);

    // ============================================================
    //                      常见内存错误演示（注释形式，不执行）
    // ============================================================
    printf("\n=== 内存错误（演示不执行）===\n");
    printf("1. 内存泄漏: malloc 后不 free\n");
    printf("   int *p = malloc(100); /* 忘记 free(p) */\n");
    printf("2. 悬空指针: free 后继续使用\n");
    printf("   free(p); *p = 1; /* 危险！ */\n");
    printf("3. 双重释放: free 同一指针两次\n");
    printf("   free(p); free(p); /* 未定义行为！ */\n");
    printf("4. 越界访问: int a[5]; a[5] = 1; /* 越界！ */\n");
    printf("5. 栈溢出: 无限递归或超大局部数组\n");
    printf("\n检测工具: Valgrind, AddressSanitizer (-fsanitize=address)\n");

    printf("\n=== 动态内存演示完成 ===\n");
    return 0;
}
