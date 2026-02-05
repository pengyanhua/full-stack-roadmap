/**
 * ============================================================
 *                Node.js 文件系统
 * ============================================================
 * 本文件介绍 Node.js 中的文件系统操作。
 * ============================================================
 */

const fs = require("fs");
const fsPromises = require("fs/promises");
const path = require("path");

console.log("=".repeat(60));
console.log("1. 同步文件操作");
console.log("=".repeat(60));

// ============================================================
//                    1. 同步文件操作
// ============================================================

const testDir = path.join(__dirname, "test_files");
const testFile = path.join(testDir, "test.txt");

// --- 创建目录 ---
console.log("\n--- 创建目录 ---");
if (!fs.existsSync(testDir)) {
    fs.mkdirSync(testDir, { recursive: true });
    console.log("目录已创建:", testDir);
} else {
    console.log("目录已存在:", testDir);
}

// --- 写入文件 ---
console.log("\n--- 写入文件 ---");
fs.writeFileSync(testFile, "Hello, Node.js!\n第二行内容\n");
console.log("文件已写入:", testFile);

// --- 读取文件 ---
console.log("\n--- 读取文件 ---");
const content = fs.readFileSync(testFile, "utf8");
console.log("文件内容:");
console.log(content);

// --- 追加内容 ---
console.log("--- 追加内容 ---");
fs.appendFileSync(testFile, "追加的内容\n");
console.log("内容已追加");

// --- 文件信息 ---
console.log("\n--- 文件信息 ---");
const stats = fs.statSync(testFile);
console.log("文件大小:", stats.size, "bytes");
console.log("是否文件:", stats.isFile());
console.log("是否目录:", stats.isDirectory());
console.log("创建时间:", stats.birthtime);
console.log("修改时间:", stats.mtime);

// --- 复制文件 ---
console.log("\n--- 复制文件 ---");
const copyFile = path.join(testDir, "copy.txt");
fs.copyFileSync(testFile, copyFile);
console.log("文件已复制到:", copyFile);

// --- 重命名/移动文件 ---
console.log("\n--- 重命名文件 ---");
const renamedFile = path.join(testDir, "renamed.txt");
fs.renameSync(copyFile, renamedFile);
console.log("文件已重命名为:", renamedFile);

// --- 读取目录 ---
console.log("\n--- 读取目录 ---");
const files = fs.readdirSync(testDir);
console.log("目录内容:", files);

// --- 带详细信息读取目录 ---
console.log("\n--- 目录详情 ---");
const entries = fs.readdirSync(testDir, { withFileTypes: true });
entries.forEach(entry => {
    const type = entry.isDirectory() ? "目录" : "文件";
    console.log(`  ${type}: ${entry.name}`);
});


console.log("\n" + "=".repeat(60));
console.log("2. 异步文件操作 (回调)");
console.log("=".repeat(60));

// ============================================================
//                    2. 异步文件操作
// ============================================================

console.log("\n--- 异步读取文件 ---");

fs.readFile(testFile, "utf8", (err, data) => {
    if (err) {
        console.error("读取错误:", err);
        return;
    }
    console.log("异步读取结果:");
    console.log(data);

    // 继续执行 Promise 版本
    runPromiseVersion();
});


async function runPromiseVersion() {
    console.log("\n" + "=".repeat(60));
    console.log("3. Promise 版文件操作（推荐）");
    console.log("=".repeat(60));

    // ============================================================
    //                    3. Promise 版文件操作
    // ============================================================

    try {
        // --- 读取文件 ---
        console.log("\n--- Promise 读取 ---");
        const data = await fsPromises.readFile(testFile, "utf8");
        console.log("Promise 读取结果:");
        console.log(data);

        // --- 写入文件 ---
        console.log("\n--- Promise 写入 ---");
        const newFile = path.join(testDir, "promise.txt");
        await fsPromises.writeFile(newFile, "使用 Promise 写入\n");
        console.log("Promise 写入完成:", newFile);

        // --- 读取目录 ---
        console.log("\n--- Promise 读取目录 ---");
        const files = await fsPromises.readdir(testDir);
        console.log("目录内容:", files);

        // --- 并行读取多个文件 ---
        console.log("\n--- 并行读取 ---");
        const filePaths = files
            .filter(f => f.endsWith(".txt"))
            .map(f => path.join(testDir, f));

        const contents = await Promise.all(
            filePaths.map(async (filePath) => {
                const content = await fsPromises.readFile(filePath, "utf8");
                return { file: path.basename(filePath), lines: content.split("\n").length };
            })
        );

        console.log("文件行数:");
        contents.forEach(({ file, lines }) => {
            console.log(`  ${file}: ${lines} 行`);
        });

    } catch (error) {
        console.error("操作失败:", error);
    }

    runPart4();
}


async function runPart4() {
    console.log("\n" + "=".repeat(60));
    console.log("4. 流操作");
    console.log("=".repeat(60));

    // ============================================================
    //                    4. 流操作
    // ============================================================

    // --- 创建大文件 ---
    console.log("\n--- 创建测试数据 ---");
    const largeFile = path.join(testDir, "large.txt");
    const writeStream = fs.createWriteStream(largeFile);

    for (let i = 0; i < 1000; i++) {
        writeStream.write(`这是第 ${i + 1} 行数据\n`);
    }
    writeStream.end();

    await new Promise(resolve => writeStream.on("finish", resolve));
    console.log("大文件已创建:", largeFile);

    // --- 读取流 ---
    console.log("\n--- 读取流 ---");
    const readStream = fs.createReadStream(largeFile, { encoding: "utf8" });

    let lineCount = 0;
    let buffer = "";

    await new Promise((resolve, reject) => {
        readStream.on("data", (chunk) => {
            buffer += chunk;
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";
            lineCount += lines.length;
        });

        readStream.on("end", () => {
            if (buffer.length > 0) lineCount++;
            console.log("总行数:", lineCount);
            resolve(null);
        });

        readStream.on("error", reject);
    });

    // --- 管道操作 ---
    console.log("\n--- 管道复制 ---");
    const srcStream = fs.createReadStream(largeFile);
    const destFile = path.join(testDir, "large_copy.txt");
    const destStream = fs.createWriteStream(destFile);

    await new Promise((resolve, reject) => {
        srcStream.pipe(destStream);
        destStream.on("finish", () => {
            console.log("管道复制完成:", destFile);
            resolve(null);
        });
        destStream.on("error", reject);
    });

    runPart5();
}


async function runPart5() {
    console.log("\n" + "=".repeat(60));
    console.log("5. Path 模块");
    console.log("=".repeat(60));

    // ============================================================
    //                    5. Path 模块
    // ============================================================

    console.log("\n--- 路径操作 ---");

    const filePath = "/home/user/documents/file.txt";

    console.log("原路径:", filePath);
    console.log("目录名:", path.dirname(filePath));
    console.log("文件名:", path.basename(filePath));
    console.log("扩展名:", path.extname(filePath));
    console.log("无扩展名:", path.basename(filePath, ".txt"));

    console.log("\n--- 路径拼接 ---");
    const joined = path.join("home", "user", "documents", "file.txt");
    console.log("join:", joined);

    const resolved = path.resolve("src", "components", "Button.js");
    console.log("resolve:", resolved);

    console.log("\n--- 路径解析 ---");
    const parsed = path.parse(filePath);
    console.log("parse:", parsed);

    console.log("\n--- 路径格式化 ---");
    const formatted = path.format({
        dir: "/home/user",
        name: "file",
        ext: ".txt"
    });
    console.log("format:", formatted);

    console.log("\n--- 相对路径 ---");
    const from = "/home/user/documents";
    const to = "/home/user/pictures/photo.jpg";
    console.log("relative:", path.relative(from, to));

    console.log("\n--- 规范化路径 ---");
    const messy = "/home/user/../user/./documents//file.txt";
    console.log("原路径:", messy);
    console.log("normalize:", path.normalize(messy));

    // 清理测试文件
    await cleanup();
}


async function cleanup() {
    console.log("\n" + "=".repeat(60));
    console.log("清理测试文件");
    console.log("=".repeat(60));

    try {
        // 递归删除测试目录
        await fsPromises.rm(testDir, { recursive: true, force: true });
        console.log("测试目录已删除:", testDir);
    } catch (error) {
        console.log("清理失败:", error.message);
    }

    console.log("\n【总结】");
    console.log(`
文件操作：
- fs.readFileSync / fs.readFile / fsPromises.readFile
- fs.writeFileSync / fs.writeFile / fsPromises.writeFile
- fs.appendFileSync / fsPromises.appendFile
- fs.copyFileSync / fsPromises.copyFile
- fs.renameSync / fsPromises.rename
- fs.unlinkSync / fsPromises.unlink (删除)

目录操作：
- fs.mkdirSync / fsPromises.mkdir
- fs.readdirSync / fsPromises.readdir
- fs.rmdirSync / fsPromises.rmdir
- fsPromises.rm({ recursive: true })

流操作：
- fs.createReadStream
- fs.createWriteStream
- stream.pipe(dest)

Path 模块：
- path.join() - 拼接路径
- path.resolve() - 解析绝对路径
- path.dirname() - 目录名
- path.basename() - 文件名
- path.extname() - 扩展名
- path.parse() / path.format()

推荐：
- 使用 fsPromises（fs/promises）
- 大文件使用流
- 路径使用 path 模块处理
`);
}
