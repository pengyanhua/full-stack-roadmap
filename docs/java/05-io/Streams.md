# Streams

::: info 文件信息
- 📄 原文件：`Streams.java`
- 🔤 语言：java
:::

============================================================
                   Java I/O 流详解
============================================================
本文件介绍 Java 中各种 I/O 流的使用。
============================================================

## 完整代码

```java
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.zip.*;

/**
 * ============================================================
 *                    Java I/O 流详解
 * ============================================================
 * 本文件介绍 Java 中各种 I/O 流的使用。
 * ============================================================
 */
public class Streams {

    public static void main(String[] args) throws IOException {
        streamHierarchy();
        byteStreams();
        characterStreams();
        bufferedStreams();
        dataStreams();
        objectStreams();
        compressionStreams();
    }

    /**
     * ============================================================
     *                    1. 流的层次结构
     * ============================================================
     */
    public static void streamHierarchy() {
        System.out.println("=".repeat(60));
        System.out.println("1. 流的层次结构");
        System.out.println("=".repeat(60));

        System.out.println("""
            【字节流】处理二进制数据
            InputStream
            ├── FileInputStream      - 文件输入
            ├── ByteArrayInputStream - 字节数组输入
            ├── BufferedInputStream  - 缓冲输入
            ├── DataInputStream      - 基本类型输入
            ├── ObjectInputStream    - 对象输入
            └── ...

            OutputStream
            ├── FileOutputStream      - 文件输出
            ├── ByteArrayOutputStream - 字节数组输出
            ├── BufferedOutputStream  - 缓冲输出
            ├── DataOutputStream      - 基本类型输出
            ├── ObjectOutputStream    - 对象输出
            └── ...

            【字符流】处理文本数据
            Reader
            ├── FileReader       - 文件读取
            ├── StringReader     - 字符串读取
            ├── BufferedReader   - 缓冲读取
            ├── InputStreamReader - 字节转字符
            └── ...

            Writer
            ├── FileWriter        - 文件写入
            ├── StringWriter      - 字符串写入
            ├── BufferedWriter    - 缓冲写入
            ├── OutputStreamWriter - 字符转字节
            ├── PrintWriter       - 格式化输出
            └── ...

            【装饰器模式】
            可以嵌套使用多个流来增强功能：
            new BufferedReader(new InputStreamReader(
                new FileInputStream("file.txt"), StandardCharsets.UTF_8))
            """);
    }

    /**
     * ============================================================
     *                    2. 字节流
     * ============================================================
     */
    public static void byteStreams() throws IOException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("2. 字节流");
        System.out.println("=".repeat(60));

        String filename = "byte_test.bin";

        // 【FileOutputStream】
        System.out.println("--- FileOutputStream ---");
        try (FileOutputStream fos = new FileOutputStream(filename)) {
            // 写入单个字节
            fos.write(65);  // 'A'
            fos.write(66);  // 'B'
            fos.write(67);  // 'C'

            // 写入字节数组
            byte[] data = {68, 69, 70};  // 'D', 'E', 'F'
            fos.write(data);

            // 写入部分数组
            byte[] more = {71, 72, 73, 74, 75};
            fos.write(more, 1, 3);  // 'H', 'I', 'J'

            System.out.println("写入完成");
        }

        // 【FileInputStream】
        System.out.println("\n--- FileInputStream ---");
        try (FileInputStream fis = new FileInputStream(filename)) {
            // 读取单个字节
            int b;
            System.out.print("逐字节读取: ");
            while ((b = fis.read()) != -1) {
                System.out.print((char) b);
            }
            System.out.println();
        }

        // 读取到数组
        try (FileInputStream fis = new FileInputStream(filename)) {
            byte[] buffer = new byte[4];
            int bytesRead;
            System.out.print("按块读取: ");
            while ((bytesRead = fis.read(buffer)) != -1) {
                System.out.print(new String(buffer, 0, bytesRead));
            }
            System.out.println();
        }

        // 【ByteArrayOutputStream / ByteArrayInputStream】
        System.out.println("\n--- ByteArray 流 ---");
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        baos.write("Hello, ByteArray!".getBytes());
        byte[] result = baos.toByteArray();
        System.out.println("ByteArrayOutputStream: " + new String(result));

        ByteArrayInputStream bais = new ByteArrayInputStream(result);
        System.out.print("ByteArrayInputStream: ");
        int ch;
        while ((ch = bais.read()) != -1) {
            System.out.print((char) ch);
        }
        System.out.println();

        // 清理
        new File(filename).delete();
    }

    /**
     * ============================================================
     *                    3. 字符流
     * ============================================================
     */
    public static void characterStreams() throws IOException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("3. 字符流");
        System.out.println("=".repeat(60));

        String filename = "char_test.txt";

        // 【FileWriter / FileReader】
        System.out.println("--- FileWriter / FileReader ---");
        try (FileWriter fw = new FileWriter(filename, StandardCharsets.UTF_8)) {
            fw.write("你好，世界！\n");
            fw.write("Hello, World!\n");
        }

        try (FileReader fr = new FileReader(filename, StandardCharsets.UTF_8)) {
            char[] buffer = new char[1024];
            int charsRead = fr.read(buffer);
            System.out.println("FileReader:\n" + new String(buffer, 0, charsRead));
        }

        // 【InputStreamReader / OutputStreamWriter】编码转换
        System.out.println("--- 编码转换 ---");
        try (OutputStreamWriter osw = new OutputStreamWriter(
                new FileOutputStream(filename), StandardCharsets.UTF_8)) {
            osw.write("使用 OutputStreamWriter 指定编码\n");
        }

        try (InputStreamReader isr = new InputStreamReader(
                new FileInputStream(filename), StandardCharsets.UTF_8)) {
            char[] buffer = new char[1024];
            int charsRead = isr.read(buffer);
            System.out.println("InputStreamReader: " + new String(buffer, 0, charsRead));
        }

        // 【StringWriter / StringReader】
        System.out.println("--- StringWriter / StringReader ---");
        StringWriter sw = new StringWriter();
        sw.write("写入 StringWriter");
        sw.append(" 追加内容");
        System.out.println("StringWriter: " + sw.toString());

        StringReader sr = new StringReader("从 StringReader 读取");
        char[] chars = new char[50];
        int len = sr.read(chars);
        System.out.println("StringReader: " + new String(chars, 0, len));

        // 清理
        new File(filename).delete();
    }

    /**
     * ============================================================
     *                    4. 缓冲流
     * ============================================================
     */
    public static void bufferedStreams() throws IOException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("4. 缓冲流");
        System.out.println("=".repeat(60));

        System.out.println("""
            缓冲流优势：
            - 减少实际 I/O 操作次数
            - 提供便捷方法（如 readLine）
            - 显著提升性能
            """);

        String filename = "buffered_test.txt";

        // 【BufferedWriter / BufferedReader】
        System.out.println("--- BufferedWriter / BufferedReader ---");
        try (BufferedWriter bw = new BufferedWriter(
                new FileWriter(filename, StandardCharsets.UTF_8))) {
            bw.write("第一行");
            bw.newLine();  // 平台无关的换行
            bw.write("第二行");
            bw.newLine();
            bw.write("第三行");
            System.out.println("BufferedWriter 完成");
        }

        try (BufferedReader br = new BufferedReader(
                new FileReader(filename, StandardCharsets.UTF_8))) {
            String line;
            System.out.println("逐行读取:");
            while ((line = br.readLine()) != null) {
                System.out.println("  " + line);
            }
        }

        // 【BufferedReader.lines() 返回 Stream】
        System.out.println("\n--- BufferedReader.lines() ---");
        try (BufferedReader br = new BufferedReader(
                new FileReader(filename, StandardCharsets.UTF_8))) {
            br.lines()
              .map(String::toUpperCase)
              .forEach(System.out::println);
        }

        // 【BufferedInputStream / BufferedOutputStream】
        System.out.println("\n--- 缓冲字节流 ---");
        String binFile = "buffered_bin.dat";
        try (BufferedOutputStream bos = new BufferedOutputStream(
                new FileOutputStream(binFile), 8192)) {  // 指定缓冲区大小
            for (int i = 0; i < 1000; i++) {
                bos.write(i % 256);
            }
            System.out.println("BufferedOutputStream 完成");
        }

        try (BufferedInputStream bis = new BufferedInputStream(
                new FileInputStream(binFile))) {
            int count = 0;
            while (bis.read() != -1) {
                count++;
            }
            System.out.println("读取字节数: " + count);
        }

        // 清理
        new File(filename).delete();
        new File(binFile).delete();
    }

    /**
     * ============================================================
     *                    5. 数据流
     * ============================================================
     */
    public static void dataStreams() throws IOException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("5. 数据流");
        System.out.println("=".repeat(60));

        System.out.println("""
            DataInputStream/DataOutputStream:
            - 读写 Java 基本数据类型
            - 以二进制格式存储，跨平台
            """);

        String filename = "data_test.dat";

        // 【写入基本类型】
        try (DataOutputStream dos = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(filename)))) {
            dos.writeInt(42);
            dos.writeDouble(3.14159);
            dos.writeBoolean(true);
            dos.writeUTF("Hello, DataStream!");  // 写入 UTF-8 字符串
            System.out.println("DataOutputStream 完成");
        }

        // 【读取基本类型】必须按写入顺序读取
        try (DataInputStream dis = new DataInputStream(
                new BufferedInputStream(new FileInputStream(filename)))) {
            int intVal = dis.readInt();
            double doubleVal = dis.readDouble();
            boolean boolVal = dis.readBoolean();
            String strVal = dis.readUTF();

            System.out.println("DataInputStream 读取:");
            System.out.println("  int: " + intVal);
            System.out.println("  double: " + doubleVal);
            System.out.println("  boolean: " + boolVal);
            System.out.println("  String: " + strVal);
        }

        // 清理
        new File(filename).delete();
    }

    /**
     * ============================================================
     *                    6. 对象流（序列化）
     * ============================================================
     */
    public static void objectStreams() throws IOException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("6. 对象流（序列化）");
        System.out.println("=".repeat(60));

        System.out.println("""
            对象序列化：
            - 将对象转换为字节流
            - 类必须实现 Serializable 接口
            - transient 字段不会被序列化
            - serialVersionUID 用于版本控制
            """);

        String filename = "object_test.ser";

        // 【写入对象】
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream(filename))) {
            Person person = new Person("张三", 25);
            oos.writeObject(person);

            // 可以写入多个对象
            oos.writeObject(new Person("李四", 30));
            System.out.println("对象序列化完成");
        }

        // 【读取对象】
        try (ObjectInputStream ois = new ObjectInputStream(
                new FileInputStream(filename))) {
            Person p1 = (Person) ois.readObject();
            Person p2 = (Person) ois.readObject();

            System.out.println("对象反序列化:");
            System.out.println("  " + p1);
            System.out.println("  " + p2);
        } catch (ClassNotFoundException e) {
            System.out.println("类未找到: " + e.getMessage());
        }

        // 清理
        new File(filename).delete();
    }

    /**
     * ============================================================
     *                    7. 压缩流
     * ============================================================
     */
    public static void compressionStreams() throws IOException {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("7. 压缩流");
        System.out.println("=".repeat(60));

        // 【GZIP 压缩】
        System.out.println("--- GZIP 压缩 ---");
        String original = "这是一段需要压缩的文本。".repeat(100);
        String gzipFile = "test.gz";

        // 压缩
        try (GZIPOutputStream gzos = new GZIPOutputStream(
                new FileOutputStream(gzipFile))) {
            gzos.write(original.getBytes(StandardCharsets.UTF_8));
        }

        System.out.println("原始大小: " + original.getBytes().length + " bytes");
        System.out.println("压缩后: " + new File(gzipFile).length() + " bytes");

        // 解压
        try (GZIPInputStream gzis = new GZIPInputStream(
                new FileInputStream(gzipFile))) {
            byte[] buffer = gzis.readAllBytes();
            String decompressed = new String(buffer, StandardCharsets.UTF_8);
            System.out.println("解压成功: " + (original.equals(decompressed)));
        }

        // 【ZIP 压缩多个文件】
        System.out.println("\n--- ZIP 压缩 ---");
        String zipFile = "test.zip";

        try (ZipOutputStream zos = new ZipOutputStream(
                new FileOutputStream(zipFile))) {
            // 添加第一个文件
            zos.putNextEntry(new ZipEntry("file1.txt"));
            zos.write("文件1的内容".getBytes(StandardCharsets.UTF_8));
            zos.closeEntry();

            // 添加第二个文件
            zos.putNextEntry(new ZipEntry("folder/file2.txt"));
            zos.write("文件2的内容".getBytes(StandardCharsets.UTF_8));
            zos.closeEntry();

            System.out.println("ZIP 压缩完成");
        }

        // 解压 ZIP
        try (ZipInputStream zis = new ZipInputStream(
                new FileInputStream(zipFile))) {
            ZipEntry entry;
            System.out.println("ZIP 内容:");
            while ((entry = zis.getNextEntry()) != null) {
                System.out.println("  " + entry.getName() + " (" + entry.getSize() + " bytes)");
                zis.closeEntry();
            }
        }

        // 清理
        new File(gzipFile).delete();
        new File(zipFile).delete();
    }
}

/**
 * 可序列化的 Person 类
 */
class Person implements Serializable {
    private static final long serialVersionUID = 1L;

    private String name;
    private int age;
    private transient String password;  // 不会被序列化

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
        this.password = "secret";
    }

    @Override
    public String toString() {
        return "Person{name='" + name + "', age=" + age +
               ", password=" + password + "}";
    }
}
```
