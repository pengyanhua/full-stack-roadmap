import java.util.Arrays;

/**
 * ============================================================
 *                    Java 数组
 * ============================================================
 * 本文件介绍 Java 中的数组创建、访问、遍历和常用操作。
 * ============================================================
 */
public class Arrays_Demo {

    public static void main(String[] args) {
        arrayBasics();
        arrayOperations();
        multiDimensionalArrays();
        arraysUtility();
    }

    /**
     * ============================================================
     *                    1. 数组基础
     * ============================================================
     */
    public static void arrayBasics() {
        System.out.println("=".repeat(60));
        System.out.println("1. 数组基础");
        System.out.println("=".repeat(60));

        // 【声明数组】
        int[] arr1;           // 推荐写法
        int arr2[];           // C 风格，不推荐

        // 【创建数组】
        // 方式1：指定长度
        int[] numbers = new int[5];  // 默认值为 0
        System.out.println("new int[5]: " + Arrays.toString(numbers));

        // 方式2：直接初始化
        int[] scores = {90, 85, 78, 92, 88};
        System.out.println("直接初始化: " + Arrays.toString(scores));

        // 方式3：匿名数组
        printArray(new int[]{1, 2, 3});

        // 【访问元素】
        System.out.println("\n--- 访问元素 ---");
        System.out.println("scores[0] = " + scores[0]);
        System.out.println("scores[4] = " + scores[4]);

        // 修改元素
        scores[0] = 100;
        System.out.println("修改后 scores[0] = " + scores[0]);

        // 【数组长度】
        System.out.println("\n数组长度: scores.length = " + scores.length);

        // 【默认值】
        System.out.println("\n--- 默认值 ---");
        int[] intArr = new int[3];
        double[] doubleArr = new double[3];
        boolean[] boolArr = new boolean[3];
        String[] strArr = new String[3];

        System.out.println("int 默认值: " + intArr[0]);
        System.out.println("double 默认值: " + doubleArr[0]);
        System.out.println("boolean 默认值: " + boolArr[0]);
        System.out.println("String 默认值: " + strArr[0]);

        // 【越界异常】
        System.out.println("\n【警告】数组越界会抛出 ArrayIndexOutOfBoundsException");
        try {
            System.out.println(scores[10]);
        } catch (ArrayIndexOutOfBoundsException e) {
            System.out.println("捕获异常: " + e.getMessage());
        }
    }

    public static void printArray(int[] arr) {
        System.out.println("匿名数组: " + Arrays.toString(arr));
    }

    /**
     * ============================================================
     *                    2. 数组操作
     * ============================================================
     */
    public static void arrayOperations() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. 数组操作");
        System.out.println("=".repeat(60));

        int[] numbers = {5, 2, 8, 1, 9, 3, 7, 4, 6};

        // 【遍历数组】
        System.out.println("--- 遍历数组 ---");

        // for 循环
        System.out.print("for 循环: ");
        for (int i = 0; i < numbers.length; i++) {
            System.out.print(numbers[i] + " ");
        }
        System.out.println();

        // for-each 循环
        System.out.print("for-each: ");
        for (int num : numbers) {
            System.out.print(num + " ");
        }
        System.out.println();

        // 【查找元素】
        System.out.println("\n--- 查找元素 ---");
        int target = 8;
        int index = -1;
        for (int i = 0; i < numbers.length; i++) {
            if (numbers[i] == target) {
                index = i;
                break;
            }
        }
        System.out.println("找到 " + target + " 在索引 " + index);

        // 【最大值和最小值】
        System.out.println("\n--- 最大值和最小值 ---");
        int max = numbers[0];
        int min = numbers[0];
        for (int num : numbers) {
            if (num > max) max = num;
            if (num < min) min = num;
        }
        System.out.println("最大值: " + max);
        System.out.println("最小值: " + min);

        // 【求和和平均值】
        System.out.println("\n--- 求和和平均值 ---");
        int sum = 0;
        for (int num : numbers) {
            sum += num;
        }
        double avg = (double) sum / numbers.length;
        System.out.println("求和: " + sum);
        System.out.println("平均值: " + avg);

        // 【复制数组】
        System.out.println("\n--- 复制数组 ---");
        int[] original = {1, 2, 3, 4, 5};

        // 方式1：System.arraycopy
        int[] copy1 = new int[5];
        System.arraycopy(original, 0, copy1, 0, original.length);
        System.out.println("System.arraycopy: " + Arrays.toString(copy1));

        // 方式2：Arrays.copyOf
        int[] copy2 = Arrays.copyOf(original, original.length);
        System.out.println("Arrays.copyOf: " + Arrays.toString(copy2));

        // 方式3：clone
        int[] copy3 = original.clone();
        System.out.println("clone: " + Arrays.toString(copy3));

        // 【验证是深拷贝还是浅拷贝】
        copy1[0] = 100;
        System.out.println("\n修改 copy1[0] = 100:");
        System.out.println("original: " + Arrays.toString(original));  // 不变
        System.out.println("copy1: " + Arrays.toString(copy1));        // 改变
    }

    /**
     * ============================================================
     *                    3. 多维数组
     * ============================================================
     */
    public static void multiDimensionalArrays() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("3. 多维数组");
        System.out.println("=".repeat(60));

        // 【二维数组】
        System.out.println("--- 二维数组 ---");

        // 创建方式1：指定大小
        int[][] matrix1 = new int[3][4];  // 3行4列
        matrix1[0][0] = 1;

        // 创建方式2：直接初始化
        int[][] matrix2 = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };

        // 遍历二维数组
        System.out.println("二维数组:");
        for (int i = 0; i < matrix2.length; i++) {
            for (int j = 0; j < matrix2[i].length; j++) {
                System.out.print(matrix2[i][j] + " ");
            }
            System.out.println();
        }

        // for-each 遍历
        System.out.println("\nfor-each 遍历:");
        for (int[] row : matrix2) {
            for (int val : row) {
                System.out.print(val + " ");
            }
            System.out.println();
        }

        // 【不规则数组】每行长度可以不同
        System.out.println("\n--- 不规则数组 ---");
        int[][] jagged = new int[3][];
        jagged[0] = new int[2];
        jagged[1] = new int[3];
        jagged[2] = new int[4];

        // 填充数据
        int val = 1;
        for (int i = 0; i < jagged.length; i++) {
            for (int j = 0; j < jagged[i].length; j++) {
                jagged[i][j] = val++;
            }
        }

        // 打印
        for (int[] row : jagged) {
            System.out.println(Arrays.toString(row));
        }

        // 【三维数组】
        System.out.println("\n--- 三维数组 ---");
        int[][][] cube = new int[2][3][4];  // 2层，每层3行4列
        cube[0][0][0] = 1;
        System.out.println("cube[0][0][0] = " + cube[0][0][0]);
    }

    /**
     * ============================================================
     *                4. Arrays 工具类
     * ============================================================
     */
    public static void arraysUtility() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("4. Arrays 工具类");
        System.out.println("=".repeat(60));

        int[] arr = {5, 2, 8, 1, 9, 3, 7, 4, 6};
        System.out.println("原数组: " + Arrays.toString(arr));

        // 【排序】
        System.out.println("\n--- 排序 ---");
        int[] sorted = Arrays.copyOf(arr, arr.length);
        Arrays.sort(sorted);
        System.out.println("排序后: " + Arrays.toString(sorted));

        // 部分排序
        int[] partial = Arrays.copyOf(arr, arr.length);
        Arrays.sort(partial, 2, 6);  // 只排序索引 2-5
        System.out.println("部分排序 [2,6): " + Arrays.toString(partial));

        // 【二分查找】（必须先排序）
        System.out.println("\n--- 二分查找 ---");
        int index = Arrays.binarySearch(sorted, 5);
        System.out.println("在排序数组中查找 5，索引: " + index);

        // 【填充】
        System.out.println("\n--- 填充 ---");
        int[] filled = new int[5];
        Arrays.fill(filled, 42);
        System.out.println("填充 42: " + Arrays.toString(filled));

        // 【比较】
        System.out.println("\n--- 比较 ---");
        int[] a = {1, 2, 3};
        int[] b = {1, 2, 3};
        int[] c = {1, 2, 4};
        System.out.println("a.equals(b): " + (a == b));           // false
        System.out.println("Arrays.equals(a, b): " + Arrays.equals(a, b));  // true
        System.out.println("Arrays.equals(a, c): " + Arrays.equals(a, c));  // false

        // 【转换为 List】
        System.out.println("\n--- 转换为 List ---");
        String[] strArr = {"a", "b", "c"};
        var list = Arrays.asList(strArr);
        System.out.println("Arrays.asList: " + list);
        // 【注意】返回的 List 是固定大小的，不能 add/remove

        // 【并行排序】（大数组时更快）
        System.out.println("\n--- 并行排序 ---");
        int[] bigArr = new int[10];
        for (int i = 0; i < bigArr.length; i++) {
            bigArr[i] = (int) (Math.random() * 100);
        }
        System.out.println("排序前: " + Arrays.toString(bigArr));
        Arrays.parallelSort(bigArr);
        System.out.println("并行排序后: " + Arrays.toString(bigArr));

        // 【mismatch】找出第一个不同元素的索引（Java 9+）
        System.out.println("\n--- mismatch (Java 9+) ---");
        int[] x = {1, 2, 3, 4, 5};
        int[] y = {1, 2, 4, 4, 5};
        int mismatch = Arrays.mismatch(x, y);
        System.out.println("第一个不同的索引: " + mismatch);

        // 【compare】比较数组（Java 9+）
        System.out.println("\n--- compare (Java 9+) ---");
        System.out.println("compare(x, y): " + Arrays.compare(x, y));  // 负数表示 x < y
    }
}
