"""
============================================================
                    树结构实现
============================================================
包含二叉树、二叉搜索树、AVL树的实现和常见算法。
============================================================
"""

from typing import TypeVar, Generic, Optional, List, Callable
from collections import deque

T = TypeVar('T')


# ============================================================
#                    二叉树节点
# ============================================================

class TreeNode(Generic[T]):
    """二叉树节点"""

    def __init__(self, val: T, left: 'TreeNode[T]' = None,
                 right: 'TreeNode[T]' = None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return f"TreeNode({self.val})"


# ============================================================
#                    二叉树遍历
# ============================================================

class BinaryTreeTraversal:
    """二叉树遍历方法"""

    # ==================== 递归遍历 ====================

    @staticmethod
    def preorder_recursive(root: TreeNode) -> List:
        """
        前序遍历（递归）: 根 → 左 → 右

        时间复杂度: O(n)
        空间复杂度: O(h)，h为树高
        """
        result = []

        def dfs(node):
            if not node:
                return
            result.append(node.val)  # 访问根
            dfs(node.left)           # 遍历左子树
            dfs(node.right)          # 遍历右子树

        dfs(root)
        return result

    @staticmethod
    def inorder_recursive(root: TreeNode) -> List:
        """
        中序遍历（递归）: 左 → 根 → 右

        对于 BST，中序遍历得到有序序列
        """
        result = []

        def dfs(node):
            if not node:
                return
            dfs(node.left)
            result.append(node.val)
            dfs(node.right)

        dfs(root)
        return result

    @staticmethod
    def postorder_recursive(root: TreeNode) -> List:
        """
        后序遍历（递归）: 左 → 右 → 根
        """
        result = []

        def dfs(node):
            if not node:
                return
            dfs(node.left)
            dfs(node.right)
            result.append(node.val)

        dfs(root)
        return result

    # ==================== 迭代遍历 ====================

    @staticmethod
    def preorder_iterative(root: TreeNode) -> List:
        """
        前序遍历（迭代）

        使用栈模拟递归
        """
        if not root:
            return []

        result = []
        stack = [root]

        while stack:
            node = stack.pop()
            result.append(node.val)

            # 先压右子节点，再压左子节点（栈是后进先出）
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)

        return result

    @staticmethod
    def inorder_iterative(root: TreeNode) -> List:
        """
        中序遍历（迭代）

        不断将左子节点入栈，然后回溯访问
        """
        result = []
        stack = []
        current = root

        while current or stack:
            # 一直向左走
            while current:
                stack.append(current)
                current = current.left

            # 访问节点
            current = stack.pop()
            result.append(current.val)

            # 转向右子树
            current = current.right

        return result

    @staticmethod
    def postorder_iterative(root: TreeNode) -> List:
        """
        后序遍历（迭代）

        方法：前序遍历（根右左）的反转
        """
        if not root:
            return []

        result = []
        stack = [root]

        while stack:
            node = stack.pop()
            result.append(node.val)

            # 先左后右（与前序相反）
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)

        return result[::-1]

    @staticmethod
    def level_order(root: TreeNode) -> List[List]:
        """
        层序遍历（BFS）

        时间复杂度: O(n)
        空间复杂度: O(n)
        """
        if not root:
            return []

        result = []
        queue = deque([root])

        while queue:
            level_size = len(queue)
            level = []

            for _ in range(level_size):
                node = queue.popleft()
                level.append(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            result.append(level)

        return result


# ============================================================
#                    二叉搜索树
# ============================================================

class BinarySearchTree(Generic[T]):
    """
    二叉搜索树实现

    特性：左 < 根 < 右
    """

    def __init__(self):
        self._root: Optional[TreeNode[T]] = None
        self._size = 0

    @property
    def root(self):
        return self._root

    def __len__(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    # ==================== 插入 ====================

    def insert(self, val: T) -> None:
        """
        插入值

        时间复杂度: 平均 O(log n)，最坏 O(n)
        """
        self._root = self._insert(self._root, val)
        self._size += 1

    def _insert(self, node: TreeNode, val: T) -> TreeNode:
        if not node:
            return TreeNode(val)

        if val < node.val:
            node.left = self._insert(node.left, val)
        elif val > node.val:
            node.right = self._insert(node.right, val)
        # 相等时不插入（或可选择更新）

        return node

    # ==================== 查找 ====================

    def search(self, val: T) -> Optional[TreeNode[T]]:
        """
        查找值

        时间复杂度: 平均 O(log n)
        """
        return self._search(self._root, val)

    def _search(self, node: TreeNode, val: T) -> Optional[TreeNode]:
        if not node or node.val == val:
            return node

        if val < node.val:
            return self._search(node.left, val)
        return self._search(node.right, val)

    def contains(self, val: T) -> bool:
        return self.search(val) is not None

    def __contains__(self, val: T) -> bool:
        return self.contains(val)

    # ==================== 删除 ====================

    def remove(self, val: T) -> bool:
        """
        删除值

        时间复杂度: 平均 O(log n)
        """
        if not self.contains(val):
            return False

        self._root = self._remove(self._root, val)
        self._size -= 1
        return True

    def _remove(self, node: TreeNode, val: T) -> Optional[TreeNode]:
        if not node:
            return None

        if val < node.val:
            node.left = self._remove(node.left, val)
        elif val > node.val:
            node.right = self._remove(node.right, val)
        else:
            # 找到要删除的节点
            if not node.left:
                return node.right
            if not node.right:
                return node.left

            # 有两个子节点：用后继（右子树最小值）替换
            successor = self._find_min(node.right)
            node.val = successor.val
            node.right = self._remove(node.right, successor.val)

        return node

    # ==================== 辅助方法 ====================

    def _find_min(self, node: TreeNode) -> TreeNode:
        """找最小值节点"""
        while node.left:
            node = node.left
        return node

    def _find_max(self, node: TreeNode) -> TreeNode:
        """找最大值节点"""
        while node.right:
            node = node.right
        return node

    def minimum(self) -> Optional[T]:
        """返回最小值"""
        if not self._root:
            return None
        return self._find_min(self._root).val

    def maximum(self) -> Optional[T]:
        """返回最大值"""
        if not self._root:
            return None
        return self._find_max(self._root).val

    def inorder(self) -> List[T]:
        """中序遍历（有序）"""
        return BinaryTreeTraversal.inorder_recursive(self._root)


# ============================================================
#                    AVL 树
# ============================================================

class AVLNode(Generic[T]):
    """AVL 树节点"""

    def __init__(self, val: T):
        self.val = val
        self.left: Optional[AVLNode] = None
        self.right: Optional[AVLNode] = None
        self.height = 1  # 节点高度


class AVLTree(Generic[T]):
    """
    AVL 树实现

    特性：任意节点左右子树高度差不超过 1
    时间复杂度：所有操作 O(log n)
    """

    def __init__(self):
        self._root: Optional[AVLNode[T]] = None

    @property
    def root(self):
        return self._root

    def _height(self, node: AVLNode) -> int:
        """获取节点高度"""
        return node.height if node else 0

    def _balance_factor(self, node: AVLNode) -> int:
        """计算平衡因子"""
        return self._height(node.left) - self._height(node.right)

    def _update_height(self, node: AVLNode) -> None:
        """更新节点高度"""
        node.height = 1 + max(self._height(node.left),
                              self._height(node.right))

    # ==================== 旋转操作 ====================

    def _right_rotate(self, y: AVLNode) -> AVLNode:
        """
        右旋（LL 型）

            y                x
           / \             /   \
          x   T3   →      T1    y
         / \                   / \
        T1  T2                T2  T3
        """
        x = y.left
        T2 = x.right

        x.right = y
        y.left = T2

        self._update_height(y)
        self._update_height(x)

        return x

    def _left_rotate(self, x: AVLNode) -> AVLNode:
        """
        左旋（RR 型）

          x                  y
         / \               /   \
        T1  y     →       x    T3
           / \           / \
          T2  T3        T1  T2
        """
        y = x.right
        T2 = y.left

        y.left = x
        x.right = T2

        self._update_height(x)
        self._update_height(y)

        return y

    def _rebalance(self, node: AVLNode) -> AVLNode:
        """重新平衡"""
        self._update_height(node)
        balance = self._balance_factor(node)

        # LL 型：右旋
        if balance > 1 and self._balance_factor(node.left) >= 0:
            return self._right_rotate(node)

        # RR 型：左旋
        if balance < -1 and self._balance_factor(node.right) <= 0:
            return self._left_rotate(node)

        # LR 型：先左旋后右旋
        if balance > 1 and self._balance_factor(node.left) < 0:
            node.left = self._left_rotate(node.left)
            return self._right_rotate(node)

        # RL 型：先右旋后左旋
        if balance < -1 and self._balance_factor(node.right) > 0:
            node.right = self._right_rotate(node.right)
            return self._left_rotate(node)

        return node

    # ==================== 插入 ====================

    def insert(self, val: T) -> None:
        """插入值"""
        self._root = self._insert(self._root, val)

    def _insert(self, node: AVLNode, val: T) -> AVLNode:
        if not node:
            return AVLNode(val)

        if val < node.val:
            node.left = self._insert(node.left, val)
        elif val > node.val:
            node.right = self._insert(node.right, val)
        else:
            return node

        return self._rebalance(node)

    # ==================== 删除 ====================

    def remove(self, val: T) -> None:
        """删除值"""
        self._root = self._remove(self._root, val)

    def _remove(self, node: AVLNode, val: T) -> Optional[AVLNode]:
        if not node:
            return None

        if val < node.val:
            node.left = self._remove(node.left, val)
        elif val > node.val:
            node.right = self._remove(node.right, val)
        else:
            if not node.left:
                return node.right
            if not node.right:
                return node.left

            # 找后继
            successor = node.right
            while successor.left:
                successor = successor.left

            node.val = successor.val
            node.right = self._remove(node.right, successor.val)

        return self._rebalance(node)


# ============================================================
#                    常见二叉树算法
# ============================================================

def max_depth(root: TreeNode) -> int:
    """
    二叉树的最大深度

    时间复杂度: O(n)
    空间复杂度: O(h)
    """
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))


def is_same_tree(p: TreeNode, q: TreeNode) -> bool:
    """判断两棵树是否相同"""
    if not p and not q:
        return True
    if not p or not q:
        return False
    return (p.val == q.val and
            is_same_tree(p.left, q.left) and
            is_same_tree(p.right, q.right))


def is_symmetric(root: TreeNode) -> bool:
    """
    判断是否是对称二叉树

    时间复杂度: O(n)
    """
    def is_mirror(left: TreeNode, right: TreeNode) -> bool:
        if not left and not right:
            return True
        if not left or not right:
            return False
        return (left.val == right.val and
                is_mirror(left.left, right.right) and
                is_mirror(left.right, right.left))

    return is_mirror(root, root) if root else True


def invert_tree(root: TreeNode) -> TreeNode:
    """
    翻转二叉树

    时间复杂度: O(n)
    """
    if not root:
        return None

    root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root


def is_valid_bst(root: TreeNode) -> bool:
    """
    验证二叉搜索树

    方法：中序遍历检查是否递增
    """
    prev = [float('-inf')]

    def inorder(node):
        if not node:
            return True

        if not inorder(node.left):
            return False

        if node.val <= prev[0]:
            return False
        prev[0] = node.val

        return inorder(node.right)

    return inorder(root)


def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    二叉树的最近公共祖先

    时间复杂度: O(n)
    """
    if not root or root == p or root == q:
        return root

    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)

    if left and right:
        return root
    return left if left else right


def diameter_of_binary_tree(root: TreeNode) -> int:
    """
    二叉树的直径（任意两节点间最长路径的边数）

    时间复杂度: O(n)
    """
    diameter = [0]

    def depth(node):
        if not node:
            return 0

        left_depth = depth(node.left)
        right_depth = depth(node.right)

        # 经过当前节点的最长路径
        diameter[0] = max(diameter[0], left_depth + right_depth)

        return 1 + max(left_depth, right_depth)

    depth(root)
    return diameter[0]


def build_tree_from_preorder_inorder(preorder: List, inorder: List) -> TreeNode:
    """
    从前序和中序遍历构造二叉树

    时间复杂度: O(n)
    """
    if not preorder or not inorder:
        return None

    # 前序第一个是根
    root_val = preorder[0]
    root = TreeNode(root_val)

    # 在中序中找根的位置
    mid = inorder.index(root_val)

    # 递归构建左右子树
    root.left = build_tree_from_preorder_inorder(
        preorder[1:mid + 1], inorder[:mid])
    root.right = build_tree_from_preorder_inorder(
        preorder[mid + 1:], inorder[mid + 1:])

    return root


def serialize(root: TreeNode) -> str:
    """
    序列化二叉树

    使用层序遍历
    """
    if not root:
        return "[]"

    result = []
    queue = deque([root])

    while queue:
        node = queue.popleft()
        if node:
            result.append(str(node.val))
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append("null")

    # 去除末尾的 null
    while result and result[-1] == "null":
        result.pop()

    return "[" + ",".join(result) + "]"


def deserialize(data: str) -> TreeNode:
    """反序列化二叉树"""
    if data == "[]":
        return None

    values = data[1:-1].split(",")
    root = TreeNode(int(values[0]))
    queue = deque([root])
    i = 1

    while queue and i < len(values):
        node = queue.popleft()

        if i < len(values) and values[i] != "null":
            node.left = TreeNode(int(values[i]))
            queue.append(node.left)
        i += 1

        if i < len(values) and values[i] != "null":
            node.right = TreeNode(int(values[i]))
            queue.append(node.right)
        i += 1

    return root


# ============================================================
#                    辅助函数
# ============================================================

def build_tree(values: List) -> TreeNode:
    """从层序列表构建二叉树（None 表示空节点）"""
    if not values:
        return None

    root = TreeNode(values[0])
    queue = deque([root])
    i = 1

    while queue and i < len(values):
        node = queue.popleft()

        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1

        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1

    return root


def print_tree(root: TreeNode, level=0, prefix="Root: "):
    """打印二叉树"""
    if root is not None:
        print(" " * (level * 4) + prefix + str(root.val))
        if root.left or root.right:
            if root.left:
                print_tree(root.left, level + 1, "L--- ")
            else:
                print(" " * ((level + 1) * 4) + "L--- None")
            if root.right:
                print_tree(root.right, level + 1, "R--- ")
            else:
                print(" " * ((level + 1) * 4) + "R--- None")


# ============================================================
#                    测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("二叉树遍历测试")
    print("=" * 60)

    # 构建测试树
    #       1
    #      / \
    #     2   3
    #    / \
    #   4   5
    root = build_tree([1, 2, 3, 4, 5])

    print("\n--- 树结构 ---")
    print_tree(root)

    print("\n--- 遍历结果 ---")
    print(f"前序遍历: {BinaryTreeTraversal.preorder_recursive(root)}")
    print(f"中序遍历: {BinaryTreeTraversal.inorder_recursive(root)}")
    print(f"后序遍历: {BinaryTreeTraversal.postorder_recursive(root)}")
    print(f"层序遍历: {BinaryTreeTraversal.level_order(root)}")

    print("\n" + "=" * 60)
    print("二叉搜索树测试")
    print("=" * 60)

    bst = BinarySearchTree()
    values = [8, 3, 10, 1, 6, 14, 4, 7, 13]

    print(f"\n插入序列: {values}")
    for v in values:
        bst.insert(v)

    print(f"中序遍历（有序）: {bst.inorder()}")
    print(f"最小值: {bst.minimum()}, 最大值: {bst.maximum()}")
    print(f"查找 6: {bst.contains(6)}")
    print(f"查找 100: {bst.contains(100)}")

    bst.remove(3)
    print(f"删除 3 后: {bst.inorder()}")

    print("\n" + "=" * 60)
    print("AVL 树测试")
    print("=" * 60)

    avl = AVLTree()
    values = [10, 20, 30, 40, 50, 25]

    print(f"\n插入序列: {values}")
    for v in values:
        avl.insert(v)
        print(f"插入 {v}, 根节点: {avl.root.val}, 高度: {avl.root.height}")

    print("\n" + "=" * 60)
    print("算法测试")
    print("=" * 60)

    root = build_tree([1, 2, 3, 4, 5])

    print(f"\n最大深度: {max_depth(root)}")
    print(f"直径: {diameter_of_binary_tree(root)}")

    symmetric_root = build_tree([1, 2, 2, 3, 4, 4, 3])
    print(f"对称树检查: {is_symmetric(symmetric_root)}")

    print("\n--- 序列化测试 ---")
    serialized = serialize(root)
    print(f"序列化: {serialized}")
    deserialized = deserialize(serialized)
    print(f"反序列化后前序: {BinaryTreeTraversal.preorder_recursive(deserialized)}")
