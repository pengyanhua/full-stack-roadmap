# FileIO

::: info 文件信息
- 📄 原文件：`FileIO.java`
- 🔤 语言：java
:::

============================================================
                   Java 文件 I/O
============================================================
本文件介绍 Java 中的传统 I/O 和 NIO 文件操作。
============================================================

## 完整代码

```java
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * ============================================================
 *                    Java 文件 I/O
 * ============================================================
 * 本文件介绍 Java 中的传统 I/O 和 NIO 文件操作。
 * ============================================================
 */
public class FileIO {

    public static void main(String[] args) throws IOException {
        fileBasics();
        traditionalIO();
        nioFiles();
        pathOperations();
        directoryOperations();
    }

    /**
     * ============================================================
     *                    1. File 基础
     * ============================================================
     */
    public static void fileBasics() throws IOException {
        System.out.println("=".repeat(60));
        System.out.println("1. File 基础");
        System.out.println("=".repeat(60));

        // 【创建 File 对象】
        System.out.println("--- 创建 File 对象 ---");
        File file = new File("test.txt");
        File dir = new File("testdir");
        File nested = new File("parent", "child.txt");

        System.out.println("文件路径: " + file.getPath());
        System.out.println("绝对路径: " + file.getAbsolutePath());
        System.out.println("规范路径: " + file.getCanonicalPath());

        // 【文件属性】
        System.out.println("\n--- 文件属性 ---");
        File currentDir = new File(".");
        System.out.println("exists(): " + currentDir.exists());
        System.out.println("isDirectory(): " + currentDir.isDirectory());
        System.out.println("isFile(): " + currentDir.isFile());
        System.out.println("canRead(): " + currentDir.canRead());
        System.out.println("canWrite(): " + currentDir.canWrite());

        // 【创建和删除】
        System.out.println("\n--- 创建和删除 ---");
        File tempFile = new File("temp_test.txt");
        if (tempFile.createNewFile()) {
            System.out.println("创建文件: " + tempFile.getName());
        }

        File tempDir = new File("temp_dir");
        if (tempDir.mkdir()) {
            System.out.println("创建目录: " + tempDir.getName());
        }

        // 清理
        tempFile.delete();
        tempDir.delete();
        System.out.println("清理完成");

        // 【列出目录内容】
        System.out.println("\n--- 列出目录内容 ---");
        File cwd = new File(".");
        String[] files = cwd.list();
        if (files != null) {
            System.out.println("当前目录文件数: " + files.length);
            for (int i = 0; i < Math.min(5, files.length); i++) {
                System.out.println("  " + files[i]);
            }
            if (files.length > 5) {
                System.out.println("  ...");
            }
        }

        // 【文件过滤】
        System.out.println("\n--- 文件过滤 ---");
        File[] javaFiles = cwd.listFiles((d, name) -> name.endsWith(".java"));
        if (javaFiles != null) {
            System.out.println("Java 文件数: " + javaFiles.length);
        }
    }

    /**
     * ============================================================
     *                    2. 传统 I/O
     * ============================================================
     */
    public static void traditionalIO() throws IOException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. 传统 I/O");
        System.out.println("=".repeat(60));

        String filename = "io_test.txt";

        // 【字节流写入】
        System.out.println("--- 字节流 ---");
        try (FileOutputStream fos = new FileOutputStream(filename)) {
            fos.write("Hello, 字节流!\n".getBytes(StandardCharsets.UTF_8));
            System.out.println("字节流写入完成");
        }

        // 【字节流读取】
        try (FileInputStream fis = new FileInputStream(filename)) {
            byte[] buffer = new byte[1024];
            int bytesRead = fis.read(buffer);
            System.out.println("字节流读取: " + new String(buffer, 0, bytesRead, StandardCharsets.UTF_8));
        }

        // 【字符流写入】推荐用于文本
        System.out.println("\n--- 字符流 ---");
        try (FileWriter fw = new FileWriter(filename, StandardCharsets.UTF_8)) {
            fw.write("Hello, 字符流!\n");
            fw.write("第二行内容\n");
            System.out.println("字符流写入完成");
        }

        // 【字符流读取】
        try (FileReader fr = new FileReader(filename, StandardCharsets.UTF_8)) {
            char[] buffer = new char[1024];
            int charsRead = fr.read(buffer);
            System.out.println("字符流读取:\n" + new String(buffer, 0, charsRead));
        }

        // 【缓冲流】推荐用于大文件
        System.out.println("--- 缓冲流 ---");
        try (BufferedWriter bw = new BufferedWriter(
                new FileWriter(filename, StandardCharsets.UTF_8))) {
            bw.write("缓冲流第一行");
            bw.newLine();
            bw.write("缓冲流第二行");
            bw.newLine();
            System.out.println("缓冲流写入完成");
        }

        try (BufferedReader br = new BufferedReader(
                new FileReader(filename, StandardCharsets.UTF_8))) {
            String line;
            System.out.println("缓冲流逐行读取:");
            while ((line = br.readLine()) != null) {
                System.out.println("  " + line);
            }
        }

        // 【PrintWriter】便捷写入
        System.out.println("\n--- PrintWriter ---");
        try (PrintWriter pw = new PrintWriter(filename, StandardCharsets.UTF_8)) {
            pw.println("PrintWriter 第一行");
            pw.printf("格式化: %d + %d = %d%n", 1, 2, 3);
            System.out.println("PrintWriter 写入完成");
        }

        // 清理测试文件
        new File(filename).delete();
        System.out.println("测试文件已清理");
    }

    /**
     * ============================================================
     *                    3. NIO Files 类
     * ============================================================
     */
    public static void nioFiles() throws IOException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("3. NIO Files 类");
        System.out.println("=".repeat(60));

        System.out.println("""
            java.nio.file.Files 提供便捷的静态方法：
            - 一行代码读写文件
            - 复制、移动、删除文件
            - 遍历目录
            - 监视目录变化
            """);

        Path testFile = Path.of("nio_test.txt");

        // 【写入文件】
        System.out.println("--- 写入文件 ---");

        // 写入字符串
        Files.writeString(testFile, "Hello NIO!\n第二行\n");
        System.out.println("writeString 完成");

        // 写入行
        Files.write(testFile, List.of("行1", "行2", "行3"));
        System.out.println("write(List) 完成");

        // 追加
        Files.writeString(testFile, "追加内容\n", StandardOpenOption.APPEND);

        // 【读取文件】
        System.out.println("\n--- 读取文件 ---");

        // 读取所有内容
        String content = Files.readString(testFile);
        System.out.println("readString:\n" + content);

        // 读取所有行
        List<String> lines = Files.readAllLines(testFile);
        System.out.println("readAllLines: " + lines);

        // 读取所有字节
        byte[] bytes = Files.readAllBytes(testFile);
        System.out.println("readAllBytes: " + bytes.length + " bytes");

        // 【流式读取】适合大文件
        System.out.println("\n--- 流式读取 ---");
        try (var stream = Files.lines(testFile)) {
            stream.filter(line -> line.contains("行"))
                  .forEach(System.out::println);
        }

        // 【文件属性】
        System.out.println("\n--- 文件属性 ---");
        System.out.println("size: " + Files.size(testFile));
        System.out.println("exists: " + Files.exists(testFile));
        System.out.println("isRegularFile: " + Files.isRegularFile(testFile));
        System.out.println("lastModified: " + Files.getLastModifiedTime(testFile));

        // 【复制和移动】
        System.out.println("\n--- 复制和移动 ---");
        Path copy = Path.of("nio_copy.txt");
        Files.copy(testFile, copy, StandardCopyOption.REPLACE_EXISTING);
        System.out.println("copy 完成");

        Path moved = Path.of("nio_moved.txt");
        Files.move(copy, moved, StandardCopyOption.REPLACE_EXISTING);
        System.out.println("move 完成");

        // 清理
        Files.deleteIfExists(testFile);
        Files.deleteIfExists(moved);
        System.out.println("测试文件已清理");
    }

    /**
     * ============================================================
     *                    4. Path 操作
     * ============================================================
     */
    public static void pathOperations() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("4. Path 操作");
        System.out.println("=".repeat(60));

        // 【创建 Path】
        System.out.println("--- 创建 Path ---");
        Path p1 = Path.of("folder", "subfolder", "file.txt");
        Path p2 = Paths.get("folder/subfolder/file.txt");
        Path p3 = Path.of("C:/Users/test/file.txt");

        System.out.println("Path.of: " + p1);
        System.out.println("Paths.get: " + p2);

        // 【Path 组件】
        System.out.println("\n--- Path 组件 ---");
        Path path = Path.of("/home/user/documents/report.pdf");
        System.out.println("toString: " + path);
        System.out.println("getFileName: " + path.getFileName());
        System.out.println("getParent: " + path.getParent());
        System.out.println("getRoot: " + path.getRoot());
        System.out.println("getNameCount: " + path.getNameCount());
        System.out.println("getName(0): " + path.getName(0));

        // 【路径操作】
        System.out.println("\n--- 路径操作 ---");
        Path base = Path.of("/home/user");
        Path resolved = base.resolve("documents/file.txt");
        System.out.println("resolve: " + resolved);

        Path relative = Path.of("/home/user/docs").relativize(Path.of("/home/user/images/photo.jpg"));
        System.out.println("relativize: " + relative);

        Path normalized = Path.of("/home/user/../user/./docs").normalize();
        System.out.println("normalize: " + normalized);

        // 【路径比较】
        System.out.println("\n--- 路径比较 ---");
        Path pa = Path.of("a/b/c");
        Path pb = Path.of("a/b/c");
        System.out.println("equals: " + pa.equals(pb));
        System.out.println("startsWith(a/b): " + pa.startsWith("a/b"));
        System.out.println("endsWith(b/c): " + pa.endsWith("b/c"));
    }

    /**
     * ============================================================
     *                    5. 目录操作
     * ============================================================
     */
    public static void directoryOperations() throws IOException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("5. 目录操作");
        System.out.println("=".repeat(60));

        Path testDir = Path.of("test_directory");

        // 【创建目录】
        System.out.println("--- 创建目录 ---");
        Files.createDirectories(testDir.resolve("sub1/sub2"));
        System.out.println("createDirectories 完成");

        // 创建测试文件
        Files.writeString(testDir.resolve("file1.txt"), "content1");
        Files.writeString(testDir.resolve("file2.txt"), "content2");
        Files.writeString(testDir.resolve("sub1/file3.txt"), "content3");

        // 【遍历目录】
        System.out.println("\n--- 遍历目录 (list) ---");
        try (var stream = Files.list(testDir)) {
            stream.forEach(System.out::println);
        }

        // 【递归遍历】
        System.out.println("\n--- 递归遍历 (walk) ---");
        try (var stream = Files.walk(testDir)) {
            stream.forEach(System.out::println);
        }

        // 【限制深度】
        System.out.println("\n--- 限制深度 (walk maxDepth=1) ---");
        try (var stream = Files.walk(testDir, 1)) {
            stream.forEach(System.out::println);
        }

        // 【查找文件】
        System.out.println("\n--- 查找文件 (find) ---");
        try (var stream = Files.find(testDir, Integer.MAX_VALUE,
                (path, attrs) -> path.toString().endsWith(".txt"))) {
            stream.forEach(System.out::println);
        }

        // 【使用 glob 模式】
        System.out.println("\n--- glob 模式 ---");
        try (var stream = Files.newDirectoryStream(testDir, "*.txt")) {
            for (Path p : stream) {
                System.out.println(p);
            }
        }

        // 【删除目录】递归删除
        System.out.println("\n--- 删除目录 ---");
        try (var stream = Files.walk(testDir)) {
            stream.sorted(java.util.Comparator.reverseOrder())
                  .forEach(p -> {
                      try {
                          Files.delete(p);
                      } catch (IOException e) {
                          e.printStackTrace();
                      }
                  });
        }
        System.out.println("目录已删除");

        // 【临时文件和目录】
        System.out.println("\n--- 临时文件和目录 ---");
        Path tempFile = Files.createTempFile("prefix_", ".tmp");
        System.out.println("临时文件: " + tempFile);

        Path tempDir = Files.createTempDirectory("tempdir_");
        System.out.println("临时目录: " + tempDir);

        // 清理
        Files.delete(tempFile);
        Files.delete(tempDir);
    }
}
```
