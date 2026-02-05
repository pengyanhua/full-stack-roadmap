import java.io.*;
import java.nio.file.*;
import java.time.*;
import java.time.format.*;
import java.util.*;

/**
 * ============================================================
 *                    Todo 命令行应用
 * ============================================================
 * 一个简单的命令行 Todo 应用，演示 Java 实战开发。
 *
 * 功能：
 * - 添加任务
 * - 列出任务
 * - 完成任务
 * - 删除任务
 * - 持久化存储
 * ============================================================
 */
public class TodoApp {

    private static final String DATA_FILE = "todos.txt";
    private static final DateTimeFormatter FORMATTER =
        DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private final List<TodoItem> todos = new ArrayList<>();
    private final Scanner scanner = new Scanner(System.in);

    public static void main(String[] args) {
        TodoApp app = new TodoApp();
        app.run();
    }

    /**
     * 主运行循环
     */
    public void run() {
        loadTodos();
        printWelcome();

        boolean running = true;
        while (running) {
            printMenu();
            String choice = scanner.nextLine().trim();

            switch (choice) {
                case "1" -> addTodo();
                case "2" -> listTodos();
                case "3" -> completeTodo();
                case "4" -> deleteTodo();
                case "5" -> {
                    saveTodos();
                    running = false;
                    System.out.println("再见！");
                }
                default -> System.out.println("无效选项，请重试。");
            }
        }
    }

    private void printWelcome() {
        System.out.println("""

            ╔════════════════════════════════════╗
            ║       Todo 命令行应用 v1.0        ║
            ╚════════════════════════════════════╝
            """);
    }

    private void printMenu() {
        System.out.println("""

            ━━━━━━━━━━ 菜单 ━━━━━━━━━━
            1. 添加任务
            2. 查看任务
            3. 完成任务
            4. 删除任务
            5. 退出
            ━━━━━━━━━━━━━━━━━━━━━━━━━━
            """);
        System.out.print("请选择: ");
    }

    /**
     * 添加新任务
     */
    private void addTodo() {
        System.out.print("\n输入任务内容: ");
        String content = scanner.nextLine().trim();

        if (content.isEmpty()) {
            System.out.println("任务内容不能为空！");
            return;
        }

        TodoItem item = new TodoItem(
            generateId(),
            content,
            false,
            LocalDateTime.now(),
            null
        );

        todos.add(item);
        saveTodos();
        System.out.println("✓ 任务已添加: " + content);
    }

    /**
     * 列出所有任务
     */
    private void listTodos() {
        System.out.println("\n━━━━━━━━━━ 任务列表 ━━━━━━━━━━");

        if (todos.isEmpty()) {
            System.out.println("  (暂无任务)");
            return;
        }

        // 分离已完成和未完成
        List<TodoItem> pending = new ArrayList<>();
        List<TodoItem> completed = new ArrayList<>();

        for (TodoItem item : todos) {
            if (item.completed()) {
                completed.add(item);
            } else {
                pending.add(item);
            }
        }

        // 显示未完成任务
        if (!pending.isEmpty()) {
            System.out.println("\n【待办】");
            for (TodoItem item : pending) {
                System.out.printf("  [%d] ○ %s%n", item.id(), item.content());
                System.out.printf("      创建于: %s%n", item.createdAt().format(FORMATTER));
            }
        }

        // 显示已完成任务
        if (!completed.isEmpty()) {
            System.out.println("\n【已完成】");
            for (TodoItem item : completed) {
                System.out.printf("  [%d] ● %s%n", item.id(), item.content());
                System.out.printf("      完成于: %s%n",
                    item.completedAt() != null ? item.completedAt().format(FORMATTER) : "");
            }
        }

        System.out.printf("%n共 %d 个任务，%d 个待办，%d 个已完成%n",
            todos.size(), pending.size(), completed.size());
    }

    /**
     * 完成任务
     */
    private void completeTodo() {
        List<TodoItem> pending = todos.stream()
            .filter(t -> !t.completed())
            .toList();

        if (pending.isEmpty()) {
            System.out.println("\n没有待办任务！");
            return;
        }

        System.out.println("\n待办任务:");
        for (TodoItem item : pending) {
            System.out.printf("  [%d] %s%n", item.id(), item.content());
        }

        System.out.print("\n输入要完成的任务 ID: ");
        try {
            int id = Integer.parseInt(scanner.nextLine().trim());
            boolean found = false;

            for (int i = 0; i < todos.size(); i++) {
                if (todos.get(i).id() == id && !todos.get(i).completed()) {
                    TodoItem old = todos.get(i);
                    todos.set(i, new TodoItem(
                        old.id(),
                        old.content(),
                        true,
                        old.createdAt(),
                        LocalDateTime.now()
                    ));
                    saveTodos();
                    System.out.println("✓ 已完成: " + old.content());
                    found = true;
                    break;
                }
            }

            if (!found) {
                System.out.println("未找到该任务或已完成！");
            }
        } catch (NumberFormatException e) {
            System.out.println("无效的 ID！");
        }
    }

    /**
     * 删除任务
     */
    private void deleteTodo() {
        if (todos.isEmpty()) {
            System.out.println("\n没有任务可删除！");
            return;
        }

        System.out.println("\n所有任务:");
        for (TodoItem item : todos) {
            String status = item.completed() ? "●" : "○";
            System.out.printf("  [%d] %s %s%n", item.id(), status, item.content());
        }

        System.out.print("\n输入要删除的任务 ID: ");
        try {
            int id = Integer.parseInt(scanner.nextLine().trim());
            boolean removed = todos.removeIf(t -> t.id() == id);

            if (removed) {
                saveTodos();
                System.out.println("✓ 任务已删除");
            } else {
                System.out.println("未找到该任务！");
            }
        } catch (NumberFormatException e) {
            System.out.println("无效的 ID！");
        }
    }

    /**
     * 生成唯一 ID
     */
    private int generateId() {
        return todos.stream()
            .mapToInt(TodoItem::id)
            .max()
            .orElse(0) + 1;
    }

    /**
     * 加载任务数据
     */
    private void loadTodos() {
        Path path = Path.of(DATA_FILE);
        if (!Files.exists(path)) {
            return;
        }

        try {
            List<String> lines = Files.readAllLines(path);
            for (String line : lines) {
                String[] parts = line.split("\\|");
                if (parts.length >= 4) {
                    int id = Integer.parseInt(parts[0]);
                    String content = parts[1];
                    boolean completed = Boolean.parseBoolean(parts[2]);
                    LocalDateTime createdAt = LocalDateTime.parse(parts[3], FORMATTER);
                    LocalDateTime completedAt = parts.length > 4 && !parts[4].isEmpty()
                        ? LocalDateTime.parse(parts[4], FORMATTER)
                        : null;

                    todos.add(new TodoItem(id, content, completed, createdAt, completedAt));
                }
            }
            System.out.println("已加载 " + todos.size() + " 个任务");
        } catch (IOException e) {
            System.out.println("加载数据失败: " + e.getMessage());
        }
    }

    /**
     * 保存任务数据
     */
    private void saveTodos() {
        try {
            List<String> lines = new ArrayList<>();
            for (TodoItem item : todos) {
                String line = String.join("|",
                    String.valueOf(item.id()),
                    item.content(),
                    String.valueOf(item.completed()),
                    item.createdAt().format(FORMATTER),
                    item.completedAt() != null ? item.completedAt().format(FORMATTER) : ""
                );
                lines.add(line);
            }
            Files.write(Path.of(DATA_FILE), lines);
        } catch (IOException e) {
            System.out.println("保存数据失败: " + e.getMessage());
        }
    }
}

/**
 * Todo 项目记录类
 */
record TodoItem(
    int id,
    String content,
    boolean completed,
    LocalDateTime createdAt,
    LocalDateTime completedAt
) {}
