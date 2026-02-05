"""
============================================================
                   图实现
============================================================
包含图的表示、遍历、最短路径、拓扑排序、最小生成树等算法。
============================================================
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import heapq


# ============================================================
#                    图的表示
# ============================================================

class Graph:
    """
    图的邻接表实现

    支持有向图和无向图
    """

    def __init__(self, directed: bool = False):
        """
        Args:
            directed: True 为有向图，False 为无向图
        """
        self.graph: Dict[int, List[int]] = defaultdict(list)
        self.directed = directed
        self.vertices: Set[int] = set()

    def add_vertex(self, v: int) -> None:
        """添加顶点"""
        self.vertices.add(v)
        if v not in self.graph:
            self.graph[v] = []

    def add_edge(self, u: int, v: int) -> None:
        """添加边"""
        self.vertices.add(u)
        self.vertices.add(v)
        self.graph[u].append(v)
        if not self.directed:
            self.graph[v].append(u)

    def remove_edge(self, u: int, v: int) -> None:
        """删除边"""
        if v in self.graph[u]:
            self.graph[u].remove(v)
        if not self.directed and u in self.graph[v]:
            self.graph[v].remove(u)

    def get_neighbors(self, v: int) -> List[int]:
        """获取邻居"""
        return self.graph[v]

    def has_edge(self, u: int, v: int) -> bool:
        """判断边是否存在"""
        return v in self.graph[u]

    def vertex_count(self) -> int:
        """顶点数量"""
        return len(self.vertices)

    def edge_count(self) -> int:
        """边数量"""
        count = sum(len(neighbors) for neighbors in self.graph.values())
        return count if self.directed else count // 2

    def __repr__(self) -> str:
        return f"Graph({dict(self.graph)})"


class WeightedGraph:
    """
    带权图的邻接表实现
    """

    def __init__(self, directed: bool = False):
        # 存储 {顶点: [(邻居, 权重), ...]}
        self.graph: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.directed = directed
        self.vertices: Set[int] = set()

    def add_edge(self, u: int, v: int, weight: float) -> None:
        """添加带权边"""
        self.vertices.add(u)
        self.vertices.add(v)
        self.graph[u].append((v, weight))
        if not self.directed:
            self.graph[v].append((u, weight))

    def get_neighbors(self, v: int) -> List[Tuple[int, float]]:
        """获取邻居及权重"""
        return self.graph[v]

    def get_weight(self, u: int, v: int) -> Optional[float]:
        """获取边权重"""
        for neighbor, weight in self.graph[u]:
            if neighbor == v:
                return weight
        return None


class AdjacencyMatrix:
    """
    邻接矩阵实现

    适合稠密图
    """

    def __init__(self, n: int, directed: bool = False):
        """
        Args:
            n: 顶点数量
            directed: 是否有向
        """
        self.n = n
        self.directed = directed
        # 初始化为无穷大表示不连通
        self.matrix = [[float('inf')] * n for _ in range(n)]
        # 自己到自己距离为 0
        for i in range(n):
            self.matrix[i][i] = 0

    def add_edge(self, u: int, v: int, weight: float = 1) -> None:
        """添加边"""
        self.matrix[u][v] = weight
        if not self.directed:
            self.matrix[v][u] = weight

    def has_edge(self, u: int, v: int) -> bool:
        """判断边是否存在"""
        return self.matrix[u][v] != float('inf') and u != v

    def get_weight(self, u: int, v: int) -> float:
        """获取边权重"""
        return self.matrix[u][v]


# ============================================================
#                    图的遍历
# ============================================================

def bfs(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    广度优先搜索

    时间复杂度: O(V + E)
    空间复杂度: O(V)

    Returns:
        访问顺序列表
    """
    visited = set()
    result = []
    queue = deque([start])
    visited.add(start)

    while queue:
        vertex = queue.popleft()
        result.append(vertex)

        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return result


def dfs_recursive(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    深度优先搜索（递归）

    时间复杂度: O(V + E)
    空间复杂度: O(V)
    """
    visited = set()
    result = []

    def dfs(vertex: int):
        visited.add(vertex)
        result.append(vertex)
        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)
    return result


def dfs_iterative(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    深度优先搜索（迭代）

    使用栈模拟递归
    """
    visited = set()
    result = []
    stack = [start]

    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            # 逆序加入保证访问顺序一致
            for neighbor in reversed(graph.get(vertex, [])):
                if neighbor not in visited:
                    stack.append(neighbor)

    return result


# ============================================================
#                    最短路径
# ============================================================

def bfs_shortest_path(
    graph: Dict[int, List[int]],
    start: int,
    end: int
) -> Tuple[int, List[int]]:
    """
    BFS 最短路径（无权图）

    Returns:
        (距离, 路径)
    """
    if start == end:
        return 0, [start]

    visited = {start}
    queue = deque([(start, [start])])

    while queue:
        vertex, path = queue.popleft()

        for neighbor in graph.get(vertex, []):
            if neighbor == end:
                return len(path), path + [neighbor]

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return -1, []  # 不可达


def dijkstra(
    graph: Dict[int, List[Tuple[int, float]]],
    start: int
) -> Tuple[Dict[int, float], Dict[int, int]]:
    """
    Dijkstra 最短路径算法

    使用优先队列优化

    时间复杂度: O((V + E) log V)

    Args:
        graph: 带权邻接表 {vertex: [(neighbor, weight), ...]}
        start: 起点

    Returns:
        (distances, predecessors)
        distances: 起点到各顶点的最短距离
        predecessors: 最短路径的前驱节点
    """
    # 初始化距离
    distances = {v: float('inf') for v in graph}
    distances[start] = 0

    # 前驱节点，用于重建路径
    predecessors = {v: None for v in graph}

    # 优先队列: (距离, 顶点)
    pq = [(0, start)]
    visited = set()

    while pq:
        dist, vertex = heapq.heappop(pq)

        if vertex in visited:
            continue
        visited.add(vertex)

        for neighbor, weight in graph.get(vertex, []):
            if neighbor in visited:
                continue

            # 松弛操作
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                predecessors[neighbor] = vertex
                heapq.heappush(pq, (new_dist, neighbor))

    return distances, predecessors


def get_path(predecessors: Dict[int, int], start: int, end: int) -> List[int]:
    """根据前驱节点重建路径"""
    path = []
    current = end

    while current is not None:
        path.append(current)
        current = predecessors[current]

    path.reverse()
    return path if path[0] == start else []


def bellman_ford(
    vertices: List[int],
    edges: List[Tuple[int, int, float]],
    start: int
) -> Tuple[Dict[int, float], bool]:
    """
    Bellman-Ford 最短路径算法

    支持负权边，可检测负环

    时间复杂度: O(VE)

    Args:
        vertices: 顶点列表
        edges: 边列表 [(u, v, weight), ...]
        start: 起点

    Returns:
        (distances, has_negative_cycle)
    """
    # 初始化距离
    distances = {v: float('inf') for v in vertices}
    distances[start] = 0

    # V-1 次松弛
    for _ in range(len(vertices) - 1):
        for u, v, weight in edges:
            if distances[u] != float('inf'):
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight

    # 检测负环
    has_negative_cycle = False
    for u, v, weight in edges:
        if distances[u] != float('inf'):
            if distances[u] + weight < distances[v]:
                has_negative_cycle = True
                break

    return distances, has_negative_cycle


def floyd_warshall(matrix: List[List[float]]) -> List[List[float]]:
    """
    Floyd-Warshall 多源最短路径

    时间复杂度: O(V³)
    空间复杂度: O(V²)

    Args:
        matrix: 邻接矩阵，matrix[i][j] 表示 i 到 j 的距离

    Returns:
        最短距离矩阵
    """
    n = len(matrix)
    # 复制矩阵
    dist = [row[:] for row in matrix]

    # 动态规划
    for k in range(n):      # 中间顶点
        for i in range(n):  # 起点
            for j in range(n):  # 终点
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist


# ============================================================
#                    拓扑排序
# ============================================================

def topological_sort_kahn(graph: Dict[int, List[int]]) -> List[int]:
    """
    拓扑排序（Kahn 算法 / BFS）

    时间复杂度: O(V + E)

    Returns:
        拓扑排序结果，如果有环返回空列表
    """
    # 计算入度
    in_degree = defaultdict(int)
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        for v in neighbors:
            in_degree[v] += 1
            all_vertices.add(v)

    # 入度为 0 的顶点入队
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    result = []

    while queue:
        vertex = queue.popleft()
        result.append(vertex)

        for neighbor in graph.get(vertex, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # 检查是否有环
    if len(result) != len(all_vertices):
        return []  # 存在环

    return result


def topological_sort_dfs(graph: Dict[int, List[int]]) -> List[int]:
    """
    拓扑排序（DFS）

    Returns:
        拓扑排序结果，如果有环返回空列表
    """
    # 0: 未访问, 1: 访问中, 2: 已完成
    state = defaultdict(int)
    result = []
    has_cycle = False

    def dfs(vertex: int):
        nonlocal has_cycle
        if has_cycle:
            return

        state[vertex] = 1  # 访问中

        for neighbor in graph.get(vertex, []):
            if state[neighbor] == 1:  # 发现环
                has_cycle = True
                return
            if state[neighbor] == 0:
                dfs(neighbor)

        state[vertex] = 2  # 已完成
        result.append(vertex)

    # 获取所有顶点
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)

    # DFS 所有顶点
    for vertex in all_vertices:
        if state[vertex] == 0:
            dfs(vertex)

    if has_cycle:
        return []

    result.reverse()
    return result


# ============================================================
#                    并查集
# ============================================================

class UnionFind:
    """
    并查集（Union-Find）

    支持路径压缩和按秩合并
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n  # 连通分量数量

    def find(self, x: int) -> int:
        """
        查找根节点（带路径压缩）

        时间复杂度: 近似 O(1)
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        合并两个集合（按秩合并）

        Returns:
            是否成功合并（已在同一集合返回 False）
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # 按秩合并
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

        self.count -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """判断是否连通"""
        return self.find(x) == self.find(y)

    def get_count(self) -> int:
        """获取连通分量数量"""
        return self.count


# ============================================================
#                    最小生成树
# ============================================================

def kruskal(
    n: int,
    edges: List[Tuple[int, int, float]]
) -> Tuple[List[Tuple[int, int, float]], float]:
    """
    Kruskal 最小生成树

    时间复杂度: O(E log E)

    Args:
        n: 顶点数量
        edges: 边列表 [(u, v, weight), ...]

    Returns:
        (MST 边列表, 总权重)
    """
    # 按权重排序
    sorted_edges = sorted(edges, key=lambda x: x[2])

    uf = UnionFind(n)
    mst = []
    total_weight = 0

    for u, v, weight in sorted_edges:
        if uf.union(u, v):  # 不形成环
            mst.append((u, v, weight))
            total_weight += weight

            if len(mst) == n - 1:  # MST 完成
                break

    return mst, total_weight


def prim(graph: Dict[int, List[Tuple[int, float]]]) -> Tuple[List[Tuple[int, int, float]], float]:
    """
    Prim 最小生成树

    时间复杂度: O(E log V)

    Args:
        graph: 带权邻接表

    Returns:
        (MST 边列表, 总权重)
    """
    if not graph:
        return [], 0

    # 从第一个顶点开始
    start = next(iter(graph))
    visited = {start}

    # 优先队列: (权重, 起点, 终点)
    edges = [(weight, start, neighbor) for neighbor, weight in graph[start]]
    heapq.heapify(edges)

    mst = []
    total_weight = 0

    while edges and len(visited) < len(graph):
        weight, u, v = heapq.heappop(edges)

        if v in visited:
            continue

        visited.add(v)
        mst.append((u, v, weight))
        total_weight += weight

        # 添加新顶点的边
        for neighbor, w in graph[v]:
            if neighbor not in visited:
                heapq.heappush(edges, (w, v, neighbor))

    return mst, total_weight


# ============================================================
#                    常见图算法
# ============================================================

def has_cycle_undirected(graph: Dict[int, List[int]]) -> bool:
    """
    检测无向图是否有环

    使用 DFS，若遇到已访问且非父节点，则有环
    """
    visited = set()

    def dfs(vertex: int, parent: int) -> bool:
        visited.add(vertex)

        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                if dfs(neighbor, vertex):
                    return True
            elif neighbor != parent:  # 遇到已访问的非父节点
                return True

        return False

    for vertex in graph:
        if vertex not in visited:
            if dfs(vertex, -1):
                return True

    return False


def has_cycle_directed(graph: Dict[int, List[int]]) -> bool:
    """
    检测有向图是否有环

    使用三色标记法
    """
    # 0: 白色(未访问), 1: 灰色(访问中), 2: 黑色(已完成)
    color = defaultdict(int)

    def dfs(vertex: int) -> bool:
        color[vertex] = 1  # 灰色

        for neighbor in graph.get(vertex, []):
            if color[neighbor] == 1:  # 遇到灰色节点，有环
                return True
            if color[neighbor] == 0:
                if dfs(neighbor):
                    return True

        color[vertex] = 2  # 黑色
        return False

    for vertex in graph:
        if color[vertex] == 0:
            if dfs(vertex):
                return True

    return False


def is_bipartite(graph: Dict[int, List[int]]) -> bool:
    """
    判断是否为二分图

    使用 BFS 染色，相邻节点颜色不同
    """
    color = {}

    for start in graph:
        if start in color:
            continue

        queue = deque([start])
        color[start] = 0

        while queue:
            vertex = queue.popleft()

            for neighbor in graph.get(vertex, []):
                if neighbor not in color:
                    color[neighbor] = 1 - color[vertex]
                    queue.append(neighbor)
                elif color[neighbor] == color[vertex]:
                    return False

    return True


def count_connected_components(graph: Dict[int, List[int]]) -> int:
    """
    计算连通分量数量
    """
    visited = set()
    count = 0

    def dfs(vertex: int):
        visited.add(vertex)
        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                dfs(neighbor)

    for vertex in graph:
        if vertex not in visited:
            dfs(vertex)
            count += 1

    return count


def find_bridges(
    graph: Dict[int, List[int]]
) -> List[Tuple[int, int]]:
    """
    查找桥（割边）

    桥是删除后会使图不连通的边
    使用 Tarjan 算法
    """
    timer = [0]
    discovery = {}  # 发现时间
    low = {}        # 能到达的最小发现时间
    bridges = []

    def dfs(vertex: int, parent: int):
        discovery[vertex] = low[vertex] = timer[0]
        timer[0] += 1

        for neighbor in graph.get(vertex, []):
            if neighbor not in discovery:
                dfs(neighbor, vertex)
                low[vertex] = min(low[vertex], low[neighbor])

                # 如果子节点不能回到祖先，则是桥
                if low[neighbor] > discovery[vertex]:
                    bridges.append((vertex, neighbor))
            elif neighbor != parent:
                low[vertex] = min(low[vertex], discovery[neighbor])

    for vertex in graph:
        if vertex not in discovery:
            dfs(vertex, -1)

    return bridges


# ============================================================
#                    测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("图的基本操作")
    print("=" * 60)

    # 创建无向图
    g = Graph(directed=False)
    edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
    for u, v in edges:
        g.add_edge(u, v)

    print(f"图: {g}")
    print(f"顶点数: {g.vertex_count()}")
    print(f"边数: {g.edge_count()}")
    print(f"顶点 0 的邻居: {g.get_neighbors(0)}")

    print("\n" + "=" * 60)
    print("图的遍历")
    print("=" * 60)

    graph = {
        0: [1, 2],
        1: [0, 3, 4],
        2: [0, 4],
        3: [1],
        4: [1, 2]
    }

    print(f"图: {graph}")
    print(f"BFS 从 0 开始: {bfs(graph, 0)}")
    print(f"DFS 从 0 开始（递归）: {dfs_recursive(graph, 0)}")
    print(f"DFS 从 0 开始（迭代）: {dfs_iterative(graph, 0)}")

    print("\n" + "=" * 60)
    print("最短路径（BFS - 无权图）")
    print("=" * 60)

    dist, path = bfs_shortest_path(graph, 0, 3)
    print(f"从 0 到 3: 距离={dist}, 路径={path}")

    print("\n" + "=" * 60)
    print("Dijkstra 算法")
    print("=" * 60)

    weighted_graph = {
        0: [(1, 4), (2, 1)],
        1: [(0, 4), (2, 2), (3, 5)],
        2: [(0, 1), (1, 2), (3, 8)],
        3: [(1, 5), (2, 8)]
    }

    distances, predecessors = dijkstra(weighted_graph, 0)
    print(f"带权图: {weighted_graph}")
    print(f"从 0 出发的最短距离: {distances}")
    print(f"从 0 到 3 的路径: {get_path(predecessors, 0, 3)}")

    print("\n" + "=" * 60)
    print("Bellman-Ford 算法")
    print("=" * 60)

    vertices = [0, 1, 2, 3]
    edges_list = [
        (0, 1, 4), (0, 2, 1),
        (1, 2, -2), (1, 3, 5),
        (2, 3, 8)
    ]

    distances, has_neg = bellman_ford(vertices, edges_list, 0)
    print(f"边: {edges_list}")
    print(f"从 0 出发的最短距离: {distances}")
    print(f"存在负环: {has_neg}")

    print("\n" + "=" * 60)
    print("Floyd-Warshall 算法")
    print("=" * 60)

    INF = float('inf')
    matrix = [
        [0, 4, 1, INF],
        [4, 0, 2, 5],
        [1, 2, 0, 8],
        [INF, 5, 8, 0]
    ]

    result = floyd_warshall(matrix)
    print("多源最短路径矩阵:")
    for row in result:
        print([x if x != INF else "∞" for x in row])

    print("\n" + "=" * 60)
    print("拓扑排序")
    print("=" * 60)

    dag = {
        0: [1, 2],
        1: [3],
        2: [3],
        3: []
    }

    print(f"DAG: {dag}")
    print(f"拓扑排序（Kahn）: {topological_sort_kahn(dag)}")
    print(f"拓扑排序（DFS）: {topological_sort_dfs(dag)}")

    print("\n" + "=" * 60)
    print("并查集")
    print("=" * 60)

    uf = UnionFind(5)
    print(f"初始连通分量数: {uf.get_count()}")

    uf.union(0, 1)
    uf.union(2, 3)
    print(f"合并 (0,1) 和 (2,3) 后: {uf.get_count()}")

    uf.union(1, 2)
    print(f"合并 (1,2) 后: {uf.get_count()}")

    print(f"0 和 3 连通: {uf.connected(0, 3)}")
    print(f"0 和 4 连通: {uf.connected(0, 4)}")

    print("\n" + "=" * 60)
    print("最小生成树")
    print("=" * 60)

    n = 4
    mst_edges = [
        (0, 1, 4), (0, 2, 1), (0, 3, 3),
        (1, 2, 2), (2, 3, 5)
    ]

    mst, total = kruskal(n, mst_edges)
    print(f"边: {mst_edges}")
    print(f"Kruskal MST: {mst}")
    print(f"总权重: {total}")

    prim_graph = {
        0: [(1, 4), (2, 1), (3, 3)],
        1: [(0, 4), (2, 2)],
        2: [(0, 1), (1, 2), (3, 5)],
        3: [(0, 3), (2, 5)]
    }

    mst, total = prim(prim_graph)
    print(f"Prim MST: {mst}")
    print(f"总权重: {total}")

    print("\n" + "=" * 60)
    print("其他图算法")
    print("=" * 60)

    # 环检测
    undirected_with_cycle = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    undirected_no_cycle = {0: [1], 1: [0, 2], 2: [1]}

    print(f"无向图 {undirected_with_cycle} 有环: {has_cycle_undirected(undirected_with_cycle)}")
    print(f"无向图 {undirected_no_cycle} 有环: {has_cycle_undirected(undirected_no_cycle)}")

    directed_with_cycle = {0: [1], 1: [2], 2: [0]}
    directed_no_cycle = {0: [1], 1: [2], 2: []}

    print(f"有向图 {directed_with_cycle} 有环: {has_cycle_directed(directed_with_cycle)}")
    print(f"有向图 {directed_no_cycle} 有环: {has_cycle_directed(directed_no_cycle)}")

    # 二分图检测
    bipartite = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}
    not_bipartite = {0: [1, 2], 1: [0, 2], 2: [0, 1]}

    print(f"\n图 {bipartite} 是二分图: {is_bipartite(bipartite)}")
    print(f"图 {not_bipartite} 是二分图: {is_bipartite(not_bipartite)}")

    # 连通分量
    disconnected = {0: [1], 1: [0], 2: [3], 3: [2], 4: []}
    print(f"\n图 {disconnected} 的连通分量数: {count_connected_components(disconnected)}")

    # 桥
    bridge_graph = {0: [1, 2], 1: [0, 2], 2: [0, 1, 3], 3: [2]}
    print(f"\n图 {bridge_graph} 的桥: {find_bridges(bridge_graph)}")
