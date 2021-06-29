---
layout: post
title: DFS/BFS篇
---

主要数据结构：

- 非线性数据结构
  - 树：单入度、多出度
  - 图：多入度、多出度

主要应用：

- 拓扑排序



二叉树的DFS有 __前序/中序/后序__ 三种，无论哪一种，都是优先访问左子节点，用栈（显式栈或调用栈）保存父节点，直到当前路径无左子节点时，才弹出父节点访问其右子节点

DFS和BFS是一种遍历方法，并不是一种优化

### 98. Validate binary search tree

核心方法：`二叉树前序遍历`

解题思路：

- 向左遍历（递减）时，下界一直是-inf，上界是父节点值
- 向右遍历（递增）时，上界一直是 inf，下界时父节点值

```python
"""
给定一个二叉树，判断其是否是一个有效的二叉搜索树。

假设一个二叉搜索树具有如下特征：
节点的左子树只包含小于当前节点的数。
节点的右子树只包含大于当前节点的数。
所有左子树和右子树自身必须也是二叉搜索树。

链接：https://leetcode-cn.com/problems/validate-binary-search-tree
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        return self.helper(root, float('-inf'), float('inf'))
    
    def helper(self, root: TreeNode, lower: int, upper: int) -> bool:
        if root is None:
            return True
        
        # 检查当前节点
        if root.val <= lower or root.val >= upper:
            return False
        
        # 检查子树，当前节点值是左子树的上界、右子树的下界
        return self.helper(root.left, lower, root.val) and self.helper(root.right, root.val, upper)
```



### 99. Recover binary search tree*

核心方法：`二叉树中序遍历`

解题思路：二叉树的DFS有前序/中序/后序三种，本题以中序遍历为框架

```python
"""
给你二叉搜索树的根节点 root ，该树中的两个节点被错误地交换。请在不改变其结构的情况下，恢复这棵树。

进阶：使用 O(n) 空间复杂度的解法很容易实现。你能想出一个只使用常数空间的解决方案吗？

链接：https://leetcode-cn.com/problems/recover-binary-search-tree
"""
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    """
    基于中序遍历分析，可能出现一个或两个逆序对
    """
    def recoverTree(self, root: TreeNode) -> None:
        stack = []
        pre, cur = None, root
        inverse_pairs = []
        while stack or cur is not None:
            # find the left most
            while cur is not None:
                stack.append(cur)
                cur = cur.left
            # 从当前子树的最左节点开始回溯
            cur = stack.pop()
            # 是否出现逆序对
            if pre is not None and pre.val >= cur.val:
                inverse_pairs.append((pre, cur))
            pre = cur
            cur = cur.right
        
        if len(inverse_pairs) == 2:
            inverse_pairs[0][0].val, inverse_pairs[1][1].val = inverse_pairs[1][1].val, inverse_pairs[0][0].val
        if len(inverse_pairs) == 1:
            inverse_pairs[0][0].val, inverse_pairs[0][1].val = inverse_pairs[0][1].val, inverse_pairs[0][0].val
```



### 113. Path sum II

核心方法：`二叉树前序遍历`

解题思路：在前序遍历中若维护状态，需要回撤

```python
"""
给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。

叶子节点 是指没有子节点的节点。

链接：https://leetcode-cn.com/problems/path-sum-ii
"""
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        results = []
        self.helper(root, targetSum, [], results)
        return results
    
    def helper(self, root: TreeNode, curSum: int, path: List[int], results: List[List[int]]) -> None:
        if root is None:
            return
        
        path.append(root.val)
        curSum -= root.val
        if root.left is None and root.right is None and 0 == curSum:
            results.append(list(path))
        else:
            self.helper(root.left, curSum, path, results)
            self.helper(root.right, curSum, path, results)
        # 注意：在前序遍历中若维护状态，需要回撤
        path.pop()
```



### 114. Flatten binary tree to linked list

核心方法：`二叉树后序遍历`

解题思路：

```python
"""
给你二叉树的根结点 root ，请你将它展开为一个单链表：

展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
展开后的单链表应该与二叉树 先序遍历 顺序相同。

链接：https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list
"""
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: TreeNode) -> None:
        self.helper(root)
    
    def helper(self, root: TreeNode) -> List[TreeNode]:
        """
        return: 子树flatten后的头节点和尾节点，都是TreeNode类型
        """
        if root is None:
            return None
        
        # 非叶节点
        if root.left is not None and root.right is not None:
            left_head, left_tail = self.helper(root.left)
            right_head, right_tail = self.helper(root.right)
            root.right = left_head
            root.left = None
            left_tail.right = right_head
            return [root, right_tail]
        elif root.left is not None:
            head, tail = self.helper(root.left)
            root.right = head
            root.left = None
            return [root, tail]
        elif root.right is not None:
            head, tail = self.helper(root.right)
            root.right = head
            root.left = None
            return [root, tail]
        # 叶节点
        else:
            return [root, root]
```



### 129. Sum root to leaf numbers

核心方法：`二叉树前序遍历`

解题思路：

```python
"""
给你一个二叉树的根节点 root ，树中每个节点都存放有一个 0 到 9 之间的数字。
每条从根节点到叶节点的路径都代表一个数字：

例如，从根节点到叶节点的路径 1 -> 2 -> 3 表示数字 123 。
计算从根节点到叶节点生成的 所有数字之和 。

叶节点 是指没有子节点的节点。

链接：https://leetcode-cn.com/problems/sum-root-to-leaf-numbers
"""
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def __init__(self):
        self.sum_ = 0

    def sumNumbers(self, root: TreeNode) -> int:
        self.helper(root, 0)
        return self.sum_

    def helper(self, root: TreeNode, cur_sum: int) -> None:
        if root is None:
            return

        cur_sum = cur_sum * 10 + root.val
        if root.left is None and root.right is None:
            self.sum_ += cur_sum
        
        if root.left is not None:
            self.helper(root.left, cur_sum)

        if root.right is not None:
            self.helper(root.right, cur_sum)
```



### 130. Surrounded regions

核心方法：`图的BFS`

解题思路：将与边界上的O相连的O都置为一个中间态，遍历结束后剩下的O就是要被X填充的

```python
"""
给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' ，找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。

链接：https://leetcode-cn.com/problems/surrounded-regions/
"""
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        rows = len(board)
        cols = len(board[0])
        offset = [0, 1, 0, -1, 0]
		# 从4个边界出发遍历
        queue = collections.deque()
        for i in range(rows):
            if board[i][0] == "O":
                queue.append((i, 0))
            if board[i][cols - 1] == "O":
                queue.append((i, cols - 1))
        for j in range(cols):
            if board[0][j] == "O":
                queue.append((0, j))
            if board[rows - 1][j] == "O":
                queue.append((rows - 1, j))

        while queue:
            row, col = queue.popleft()
            board[row][col] = "Y"
            for i in range(4):
                next_row = row + offset[i]
                next_col = col + offset[i + 1]
                if 0 <= next_row < rows and 0 <= next_col < cols and board[next_row][next_col] == "O":
                    queue.append((next_row, next_col))

        for i in range(rows):
            for j in range(cols):
                if board[i][j] == "O":
                    board[i][j] = "X"
                elif board[i][j] == "Y":
                    board[i][j] = "O"
```



### 133. Clone graph

核心方法：`图的DFS`

解题思路：前序遍历，先处理当前节点，在递归处理其子节点

```python
"""
给你无向连通图中一个节点的引用，请你返回该图的 深拷贝（克隆）。

图中的每个节点都包含它的值val（int）和其邻居的列表（list[Node]）。

链接：https://leetcode-cn.com/problems/clone-graph
"""
"""
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""
class Solution:
    def __init__(self):
        self.existed_nodes = {}

    # clone a node and all its neighbors
    def cloneGraph(self, node: 'Node') -> 'Node':
        if node is None:
            return None
        
        # node has been cloned
        if node in self.existed_nodes:
            return self.existed_nodes[node]
        
        # clone the new node
        copied_node = Node(node.val)
        self.existed_nodes[node] = copied_node
        # clone its neighbors
        for neighbor in node.neighbors:
            copied_neighbor = self.cloneGraph(neighbor)
            copied_node.neighbors.append(copied_neighbor)

        # clone the node itself
        return copied_node
```



### 156. Binary tree upside down

核心方法：`二叉树后序遍历`

解题思路：

```python
"""
给定一个二叉树，其中所有的右节点要么是具有兄弟节点（拥有相同父节点的左节点）的叶节点，要么为空，将此二叉树上下翻转并将它变成一棵树， 原来的右节点将转换成左叶节点。返回新的根。

链接：https://leetcode-cn.com/problems/binary-tree-upside-down
"""
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def upsideDownBinaryTree(self, root: TreeNode) -> TreeNode:
        if root is None:
            return None
        if root.left is None and root.left is None:
            return root

        left = root.left
        right = root.right
        root.left = None
        root.right = None

        # 翻转左孩子
        new_left_root = self.upsideDownBinaryTree(left)
        # 翻转右孩子
        new_right_root = self.upsideDownBinaryTree(right)

        # left成为翻转后新树的叶节点
        left.right = root
        left.left = new_right_root

        return new_left_root
```



### 199. Binary tree right side view

核心方法：`二叉树层序遍历`

解题思路：

```python
"""
给定一棵二叉树，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

链接：https://leetcode-cn.com/problems/binary-tree-right-side-view/
"""
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        queue = collections.deque()
        if root is not None:
            queue.append(root)
        results = []
        while queue:
            results.append(queue[-1].val)
            # 每一层都要清空
            cur_size = len(queue)
            for _ in range(cur_size):
                cur_node = queue.popleft()
                if cur_node.left is not None:
                    queue.append(cur_node.left)
                if cur_node.right is not None:
                    queue.append(cur_node.right)
        return results
```



### 200. Number of islands

核心方法：`图的DFS`，`并查集`

解题思路：

```python
"""
给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

链接：https://leetcode-cn.com/problems/number-of-islands
"""
class UnionFind:
    def __init__(self, n: int):
        self.set_count = n
        self.parent = [i for i in range(n)]
        self.rank = [0 for _ in range(n)]
    
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        a_root = self.find(a)
        b_root = self.find(b)

        # 本来就在一个集合中，无需合并
        if a_root == b_root:
            return

        # 在不同集合中，需要合并
        if self.rank[a_root] > self.rank[b_root]:
            self.parent[b_root] = a_root
        else:
            self.parent[a_root] = b_root
            if self.rank[a_root] == self.rank[b_root]:
                self.rank[b_root] += 1
        
        # 合并之后，集合数减一
        self.set_count -= 1

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        uf = UnionFind(m * n)

        for i in range(m):
            for j in range(n):
                # 以第i * n + j个元素为中心，考察其邻居
                if grid[i][j] == '1':
                    if i < m - 1 and grid[i + 1][j] == '1':
                        uf.union(i * n + j, (i + 1) * n + j)
                    if i > 0 and grid[i - 1][j] == '1':
                        uf.union(i * n + j, (i - 1) * n + j)
                    if j < n - 1 and grid[i][j + 1] == '1':
                        uf.union(i * n + j, i * n + j + 1)
                    if j > 0 and grid[i][j - 1] == '1':
                        uf.union(i * n + j, i * n + j - 1)
        
        island = []
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    island.append(uf.find(i * n + j))
        return len(set(island))
```



### 222. Count complete tree nodes

核心方法：`二叉树前序遍历`

解题思路：

```python
"""
给你一棵 完全二叉树 的根节点 root ，求出该树的节点个数。

完全二叉树 的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 h 层，则该层包含 1~ 2h 个节点。

链接：https://leetcode-cn.com/problems/count-complete-tree-nodes
"""
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        if root is None:
            return 0
        return 1 + self.countNodes(root.left) + self.countNodes(root.right)
```



### 207. Course schedule

核心方法：`图的DFS`，`拓扑排序`

解题思路：

1. 图遍历过程中需维护节点状态（防止循环访问），通常采用 __{0: 未访问, 1: 已访问}__，而本题是 __{0: 未访问, 1: 已访问, 2: 访问中}__
2. 维护节点状态是一种 `memoization` 机制（短路），对DP可优化计算复杂度，对图遍历还确保程序是可终止的

```python
"""
你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。

在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。

例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。

链接：https://leetcode-cn.com/problems/course-schedule
"""
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 由边的集合生成邻接表
        adjacencies = {i: [] for i in range(numCourses)}
        for pre in prerequisites:
            adjacencies[pre[1]].append(pre[0])

        # 分别以每个点为起点进行DFS遍历
        # 维护所有节点的全局状态：
        # {
        #     visited: 已经在其他遍历中被访问过，经过该节点的路径不成环
        #     visiting: 正处于当前遍历中，尚无法确定是否在成环路径上
        #     unvisited: 还未被放问过
        # }
		# 
        # 终止条件：遍历到了visited或visiting节点
        #   - visited: return True
        #   - visiting: return False
        def dfs(cur_course: int, status: Dict[int, str]) -> bool:
            # 后序遍历
            # 所有子节点开始的路径均不成环，当前节点才不成环

            if status[cur_course] == 'visited':
                return True
            if status[cur_course] == 'visiting':
                return False
            
            # 当前节点还未被访问过，开始一次新的遍历，标记为“访问中”
            status[cur_course] = 'visiting'
            for next_course in adjacencies[cur_course]:
                if not dfs(next_course, status):
                    return False
            # 以当前节点为起点的所有路径都探索完毕，且未成环，将当前节点标记为“已访问”
            status[cur_course] = 'visited'

            return True

        #### 主流程
        # 初始将每个节点都标记为“未访问”
        status = {i: 'unknown' for i in range(numCourses)}
        # 分别以每个节点为起点进行遍历，并修改status
        for start_course in range(numCourses):
            if not dfs(start_course, status):
                return False
        
        return True
```



### 210. Course schedule II

核心方法：`图的DFS`，`拓扑排序`

解题思路：对课程做拓扑排序

```python
"""
现在你总共有n门课需要选，记为0到n-1。

在选修某些课程之前需要一些先修课程。例如，想要学习课程0，你需要先完成课程1，我们用一个匹配来表示他们: [0,1]

给定课程总量以及它们的先决条件，返回你为了学完所有课程所安排的学习顺序。

可能会有多个正确的顺序，你只要返回一种就可以了。如果不可能完成所有课程，返回一个空数组。

链接：https://leetcode-cn.com/problems/course-schedule-ii
"""
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # 建立邻接表
        adjacency = {i: [] for i in range(numCourses)}
        for pre in prerequisites:
            adjacency[pre[1]].append(pre[0])
        
        # 后序遍历，过程中做两件事，更新节点状态 + 记录路径
        # return: 是否不成环
        def dfs(cur_course: int, status: Dict[int, str], path: List[int]) -> bool:
            # 已访问过的路径不会被访问第二次，因此节点不会被重复添加到path中
            if status[cur_course] == 'visited':
                return True
            if status[cur_course] == 'visiting':
                return False
            
            status[cur_course] = 'visiting'
            for next_course in adjacency[cur_course]:
                if not dfs(next_course, status, path):
                    return False

            # cur_course的子节点都不成环且都已加入路径中，更新cur_course状态并将其加入路径中
            status[cur_course] = 'visited'
            path.append(cur_course)
            return True
        
        status = {i: 'unvisited' for i in range(numCourses)}
        path = []
        for start_course in range(numCourses):
            if not dfs(start_course, status, path):
                return []
        path.reverse()
        return path
```



#### 模板-图DFS

```python
# 双状态:
def graph_dfs(seed_node, g, node_status):
    # 检查状态（短路或终止）
    if node_status...:
        return ...
    # 设置状态
    node_status...
    # 递归访问子节点
    for neighbor in g[seed_node]:
        graph_dfs(neighbor, g, node_status)
    # 其他辅助操作
    pass

# 多状态:
def graph_dfs(seed_node, g, node_status):
    # 检查状态（短路或终止）
    if node_status...:
        return ...
    # 设置状态
    node_status...
    # 递归访问子节点
    for neighbor in g[seed_node]:
        graph_dfs(neighbor, g, node_status)
    # 更新状态
    node_status...
    # 其他辅助操作
    pass

# 主调部分:
# 考虑好seeds是什么，状态存什么
for seed in seeds:
    graph_dfs(seed, g, node_status)
```

---



### 230. Kth smallest element in a BST

核心方法：`二叉树中序遍历`

解题思路：二叉树的中序遍历输出的就是排序数组，有递归和迭代两种实现

```python
"""
给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 个最小元素（从 1 开始计数）。

链接：https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/
"""
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        stack = []
        while True:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if not k:
                return root.val
            root = root.right
```



### 236. Lowest common ancestor of a binary tree

核心方法：

解题思路：

```python
"""
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

链接：https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree
"""
```

