---
layout: post
title: DFS/BFS篇
---

综述：

DFS/BFS是一种针对 __非线性数据结构__ 的 __有计划__ 的 __遍历__ 方法，并不是一种优化算法（不以降低时间/空间复杂度为目标）。DFS使用栈来控制访问顺序，BFS使用队列来控制访问顺序。

主要数据结构：

- 非线性数据结构
  - 树：单入度、多出度
  - 图：多入度、多出度

主要应用：

- 拓扑排序



目录：

- 二叉树的DFS（前序、中序、后序）
- 二叉树的BFS（层序）
- 图的DFS/BFS



## 二叉树的DFS

有 __前序/中序/后序__ 三种，从栈管理的角度，三种方式都是：

1. 父节点入栈（显式栈或调用栈），`stack.append(root)`
2. 优先访问左子节点直到无左子节点，`root = root.left`
3. 弹出父节点并访问其右子节点，`root = stack.pop(); root = root.right`

```python
## 递归式
def dfs(root):
  if root is None:
      return
  # 1. 父节点被隐式压入“调用栈”
  dfs(root.left)  # 2. 优先访问左子节点
  # 3.1. 父节点随函数退出而被弹出
  dfs(root.right)  # 3.2. 访问其右子节点

## 迭代式
def dfs(root):
  if root is not None:
      stack.append(root)
  while stack:
      while root is not None:
    	    stack.append(root)  # 1. 父节点被压入“显式栈”
          root = root.left  # 2. 优先访问左子节点
      root = stack.pop()  # 3.1. 弹出父节点
      root = root.right  # 3.2. 访问其右子节点
```



```python
# 对树节点的统一定义
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
```

### 98. Validate binary search tree

核心方法：`二叉树前序遍历`

解题思路：

- 在遍历过程中维护值域
- 左分支遍历（递减）时，node.val in [-inf, parent.val)
- 右分支遍历（递增）时，node.val in (pareng.val, inf]

```python
"""
给定一个二叉树，判断其是否是一个有效的二叉搜索树。

假设一个二叉搜索树具有如下特征：
- 节点的左子树只包含小于当前节点的数。
- 节点的右子树只包含大于当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树。

链接：https://leetcode-cn.com/problems/validate-binary-search-tree
"""
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        return self.helper(root, float('-inf'), float('inf'))
    
    def helper(self, root: TreeNode, lower: int, upper: int) -> bool:
        """
        [lowerm, upper]维护了当前节点的值遇
        """
        if root is None:
            return True
        
        # 检查当前节点是否在值域内
        if root.val <= lower or root.val >= upper:
            return False
        
        # 检查子树，当前节点值是左子树的上界、右子树的下界
        return self.helper(root.left, lower, root.val) and self.helper(root.right, root.val, upper)
```



### 99. Recover binary search tree*

核心方法：`二叉树中序遍历`

解题思路：

```python
"""
给你二叉搜索树的根节点root，该树中的两个节点被错误地交换。请在不改变其结构的情况下，恢复这棵树。

进阶：使用O(n)空间复杂度的解法很容易实现。你能想出一个只使用常数空间的解决方案吗？

链接：https://leetcode-cn.com/problems/recover-binary-search-tree
"""
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

解题思路：

```python
"""
给你二叉树的根节点root和一个整数目标和targetSum，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。

叶子节点 是指没有子节点的节点。

链接：https://leetcode-cn.com/problems/path-sum-ii
"""
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
        self.helper(root.left, curSum, path, results)
        self.helper(root.right, curSum, path, results)

        # 注意：在前序遍历中若维护全局状态，需要回撤
        path.pop()
```



### 437. Path sum III

核心方法：`二叉树前序遍历`

解题思路：在每一条路径的遍历中，记录根节点至当前节点的前缀和

```python
"""
给定一个二叉树，它的每个结点都存放着一个整数值。

找出路径和等于给定数值的路径总数。

路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

二叉树不超过1000个节点，且节点数值范围是 [-1000000,1000000] 的整数。

链接：https://leetcode-cn.com/problems/path-sum-iii
"""
class Solution:
    def __init__(self):
        # 前缀和相同的路径可能有多个，即加和为0的子区间可能有多个，因此要用dict记录同一前缀和的出现次数
        self.preSums = {}
        self.total_count = 0

    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        def dfs(root: TreeNode, curSum: int):
            if root is None:
                return

            # 计算截至当前节点的路径前缀和
            curSum += root.val
            if curSum == self.targetSum:
                self.total_count += 1
            if curSum - self.targetSum in self.preSums:
                self.total_count += self.preSums[curSum - self.targetSum]
            self.preSums[curSum] = self.preSums.get(curSum, 0) + 1

            dfs(root.left, curSum)
            dfs(root.right, curSum)

            self.preSums[curSum] -= 1
            if self.preSums[curSum] == 0:
                del self.preSums[curSum]

        self.targetSum = targetSum
        dfs(root, 0)
        return self.total_count
```



### 114. Flatten binary tree to linked list

核心方法：`二叉树后序遍历`

解题思路：

```python
"""
给你二叉树的根结点root，请你将它展开为一个单链表：

展开后的单链表应该同样使用TreeNode，其中right子指针指向链表中下一个结点，而左子指针始终为null。
展开后的单链表应该与二叉树 先序遍历 顺序相同。

链接：https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list
"""
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
        
        self.helper(root.left, cur_sum)
        self.helper(root.right, cur_sum)
```



### 156. Binary tree upside down

核心方法：`二叉树后序遍历`

解题思路：

```python
"""
给定一个二叉树，其中所有的右节点要么是具有兄弟节点（拥有相同父节点的左节点）的叶节点，要么为空，将此二叉树上下翻转并将它变成一棵树， 原来的右节点将转换成左叶节点。返回新的根。

链接：https://leetcode-cn.com/problems/binary-tree-upside-down
"""
class Solution:
    def upsideDownBinaryTree(self, root: TreeNode) -> TreeNode:
        new_root = self.helper(root)
        # 将root置为叶节点
        if root is not None:
            root.left, root.right = None, None
        return new_root

    def helper(self, root: TreeNode) -> TreeNode:
        if root is None:
            return None
        if root.left is None and root.left is None:
            return root

        # 翻转左孩子
        new_left_root = self.upsideDownBinaryTree(root.left)
        # 翻转右孩子
        new_right_root = self.upsideDownBinaryTree(root.right)

        # root.left成为翻转后新树的叶节点
        root.left.left = new_right_root
        root.left.right = root

        return new_left_root
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
class Solution:
    def countNodes(self, root: TreeNode) -> int:
        if root is None:
            return 0
        return 1 + self.countNodes(root.left) + self.countNodes(root.right)
```



### 230. Kth smallest element in a BST

核心方法：`二叉树中序遍历`

解题思路：二叉搜索树的中序遍历可输出排序数组

```python
"""
给定一个二叉搜索树的根节点root，和一个整数k，请你设计一个算法查找其中第k个最小元素（从1开始计数）。

链接：https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/
"""
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

核心方法：`二叉树后序遍历`

解题思路：	

```python
"""
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

链接：https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree
"""
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root is None or root == p or root == q:
            return root

        left_res = self.lowestCommonAncestor(root.left, p, q)
        right_res = self.lowestCommonAncestor(root.right, p, q)

        if left_res is not None and right_res is not None:
            return root
        elif left_res is None:
            return right_res
        else:
            return left_res
```



### 337. House robber III

核心方法：`二叉树后序遍历`

解题思路：在遍历的过程中有 _偷_ 和 _不偷_ 两种选择

```python
"""
在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。

计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。

链接：https://leetcode-cn.com/problems/house-robber-iii
"""
class Solution:
    def rob(self, root: TreeNode) -> int:
        return max(self.helper(root))

    def helper(self, root: TreeNode) -> List[int]:
        """
        return: [选择当前节点的最高金额，不选当前节点的最高金额]
        """
        if root is None:
            return [0, 0]

        select_left_max_amount, skip_left_max_amount = self.helper(root.left)
        select_right_max_amount, skip_right_max_amount = self.helper(root.right)

        # 选择当前节点
        select_root_max_amount = root.val + skip_left_max_amount + skip_right_max_amount
        # 跳过当前节点
        skip_root_max_amount = (max(select_left_max_amount, skip_left_max_amount) + 
                                max(select_right_max_amount, skip_right_max_amount))

        return [select_root_max_amount, skip_root_max_amount]
```



### 366. Find leaves of binary tree

核心方法：`二叉树后序遍历`

解题思路：计算每个节点的bottom-up深度

```python
"""
给你一棵二叉树，请按以下要求的顺序收集它的全部节点：

依次从左到右，每次收集并删除所有的叶子节点
重复如上过程直到整棵树为空

链接：https://leetcode-cn.com/problems/find-leaves-of-binary-tree/
"""
class Solution:
    def __init__(self):
        self.result = []

    # 按bottom-up深度重新编排节点，从小到大将同一深度的节点放在一起
    def findLeaves(self, root: TreeNode) -> List[List[int]]:
        self._traverse(root)
        return self.result

    # 后序遍历，返回当前节点的bottom-up深度，返回值的设置很关键
    def _traverse(self, root: TreeNode) -> int:
        if root is None:
            return -1

        left_depth = self._traverse(root.left)
        right_depth = self._traverse(root.right)

        cur_depth = max(left_depth, right_depth) + 1
        if cur_depth >= len(self.result):
            self.result.append([])
        self.result[cur_depth].append(root.val)

        return cur_depth
```



### 513. Find bottom left tree value

核心方法：`二叉树前序遍历`

解题思路：深度最深的第一个值

```python
"""
给定一个二叉树，在树的最后一行找到最左边的值。

链接：https://leetcode-cn.com/problems/find-bottom-left-tree-value/
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    max_depth = 0
    result = None

    def findBottomLeftValue(self, root: TreeNode) -> int:
        def dfs(root: TreeNode, depth: int):
            if root is None:
                return

            if depth > self.max_depth:
                self.max_depth = depth
                self.result = root.val

            dfs(root.left, depth + 1)
            dfs(root.right, depth + 1)

        dfs(root, 1)
        return self.result
```



### 538. Convert BST to greater tree

核心方法：`二叉树中序遍历`

解题思路：

1. 倒排数组的前缀和（涉及数组排序，需使用中序遍历）
2. 通常的二叉树遍历优先访问左分支，本题是倒排，因此要优先访问右分支

```python
"""
给出二叉 搜索 树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），使每个节点 node 的新值等于原树中大于或等于 node.val 的值之和。

提醒一下，二叉搜索树满足下列约束条件：

节点的左子树仅包含键 小于 节点键的节点。
节点的右子树仅包含键 大于 节点键的节点。
左右子树也必须是二叉搜索树。

链接：https://leetcode-cn.com/problems/convert-bst-to-greater-tree
"""
class Solution:
    def __init__(self):
        # 维护遍历过程中的当前最大值
        self.cum_sum = 0

    def convertBST(self, root: TreeNode) -> TreeNode:
        self.helper(root)
        return root

    def helper(self, root: TreeNode):
        if root is None:
            return
        
        self.helper(root.right)

        self.cum_sum += root.val
        root.val = self.cum_sum

        self.helper(root.left)
```



### 669. Trim a binary search tree

核心方法：`二叉树前序遍历`

解题思路：

```python
"""
给你二叉搜索树的根节点 root ，同时给定最小边界low 和最大边界 high。通过修剪二叉搜索树，使得所有节点的值在[low, high]中。修剪树不应该改变保留在树中的元素的相对结构（即，如果没有被移除，原有的父代子代关系都应当保留）。 可以证明，存在唯一的答案。

所以结果应当返回修剪好的二叉搜索树的新的根节点。注意，根节点可能会根据给定的边界发生改变。

链接：https://leetcode-cn.com/problems/trim-a-binary-search-tree
"""
class Solution:
    def trimBST(self, root: TreeNode, low: int, high: int) -> TreeNode:
        if root is None:
            return None
        if low <= root.val <= high:
            root.left = self.trimBST(root.left, low, high)
            root.right = self.trimBST(root.right, low, high)
            return root
        if root.val < low:
            return self.trimBST(root.right, low, high)
        if root.val > high:
            return self.trimBST(root.left, low, high)
```



### 863. All nodes distance K in binary tree

核心方法：`二叉树前序遍历`，`图的BFS`

解题思路：

1. 以根节点为锚点，计算每个节点到根节点的距离，将相对距离转化为绝对距离
2. 给每个节点增加一个指向父节点的指针，把树（有向图）变成一个无向图，使用BFS求k-hop neighbors

```python
"""
给定一个二叉树（具有根结点 root），一个目标结点 target ，和一个整数值 K 。

返回到目标结点 target 距离为 K 的所有结点的值的列表。 答案可以以任何顺序返回。

链接：https://leetcode-cn.com/problems/all-nodes-distance-k-in-binary-tree
"""
class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        def dfs_add_par(cur_node):
            if cur_node is None:
                return
            if cur_node.left is not None:
                setattr(cur_node.left, "par", cur_node)
            if cur_node.right is not None:
                setattr(cur_node.right, "par", cur_node)
            dfs_add_par(cur_node.left)
            dfs_add_par(cur_node.right)

        # 给每个节点增加指向父节点的指针
        dfs_add_par(root)
        results = []
        # BFS
        queue = collections.deque()
        queue.append(target)
        visited = [target]
        depth = 0
        while queue:
            if depth == k:
                results.extend(list(queue))
                break
            cur_size = len(queue)
            depth += 1
            for _ in range(cur_size):
                cur_node = queue.popleft()
                for next_node in [getattr(cur_node, "par", None), cur_node.left, cur_node.right]:
                    if next_node is not None and next_node not in visited:
                        visited.append(next_node)
                        queue.append(next_node)

        return results
```



### 22. Generate parentheses

核心方法：`多叉树DFS`，`回溯`

解题思路：

```python
"""
数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。

链接：https://leetcode-cn.com/problems/generate-parentheses/
"""
class Solution:
    def __init__(self):
        self.results = []

    def generateParenthesis(self, n: int) -> List[str]:
        self.dfs(n, 0, 0, '')
        return self.results
    
    def dfs(self, n: int, left_count: int, right_count: int, result: str) -> None:
        if n == left_count and n == right_count:
            self.results.append(result)
        
        # 任何时候都可以放置左括号
        if left_count < n:
            self.dfs(n, left_count + 1, right_count, result + '(')
        # 右括号数不能多于左括号
        if right_count < left_count and right_count < n:
            self.dfs(n, left_count, right_count + 1, result + ')')
```



## 二叉树的BFS

只有迭代实现，没有递归实现（不使用栈）

```python
## Type 1
def bfs(root):
    queue = deque()
    if root is not None:
        queue.append(root)
    while queue:
        node = queue.popleft()
        if node.left is not None:
            queue.append(node.left)
        if node.right is not None:
            queue.append(node.right)

## Type 2: exhausted mode
def bfs(root):
    queue = deque()
    if root is not None:
        queue.append(root)
    while queue:
        layer_size = len(queue)
        for _ in range(layer_size):
            node = queue.popleft()
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)
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
            cur_size = len(queue)
            for _ in range(cur_size):
                cur_node = queue.popleft()
                if cur_node.left is not None:
                    queue.append(cur_node.left)
                if cur_node.right is not None:
                    queue.append(cur_node.right)
        return results
```



### 310. Minimum height trees

核心方法：`多叉树层序遍历`

解题思路：不断删除叶节点，最后剩下的节点就是最小高度树的根节点

```python
"""
树是一个无向图，其中任何两个顶点只通过一条路径连接。 换句话说，一个任何没有简单环路的连通图都是一棵树。

给你一棵包含 n 个节点的树，标记为 0 到 n - 1 。给定数字 n 和一个有 n - 1 条无向边的 edges 列表（每一个边都是一对标签），其中 edges[i] = [ai, bi] 表示树中节点 ai 和 bi 之间存在一条无向边。

可选择树中任何一个节点作为根。当选择节点 x 作为根节点时，设结果树的高度为 h 。在所有可能的树中，具有最小高度的树（即，min(h)）被称为 最小高度树 。

请你找到所有的 最小高度树 并按 任意顺序 返回它们的根节点标签列表。

树的 高度 是指根节点和叶子节点之间最长向下路径上边的数量。

链接：https://leetcode-cn.com/problems/minimum-height-trees
"""
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n == 1:
            return [0]

        # 统计每个节点的出度，以及每个节点的近邻
        degree = [0] * n
        neighbors = {i: [] for i in range(n)}
        for edge in edges:
            degree[edge[0]] += 1
            degree[edge[1]] += 1
            neighbors[edge[0]].append(edge[1])
            neighbors[edge[1]].append(edge[0])

        q = collections.deque()
        # 将当前叶子节点加入队列中
        for i in range(n):
            if degree[i] == 1:
                q.append(i)

        while q:
            leaf_count = len(q)
            results = []
            for _ in range(leaf_count):
                cur_leaf = q.popleft()
                results.append(cur_leaf)
                for neighbor in neighbors[cur_leaf]:
                    # 删除叶节点 := 邻居的度 - 1
                    degree[neighbor] -= 1
                    # 判断是否生成了新的叶节点
                    if degree[neighbor] == 1:
                        q.append(neighbor)
        
        # 返回最后的叶节点
        return results
```



### 515. Find largest value in each tree row

核心方法：`二叉树层序遍历`

解题思路：

```python
"""
您需要在二叉树的每一行中找到最大的值。

链接：https://leetcode-cn.com/problems/find-largest-value-in-each-tree-row/
"""
class Solution:
    def largestValues(self, root: TreeNode) -> List[int]:
        queue = collections.deque()
        if root is not None:
            queue.append(root)
        results = []
        while queue:
            cur_size = len(queue)
            results.append(max(queue, key=lambda x: x.val).val)
            for _ in range(cur_size):
                cur_node = queue.popleft()
                if cur_node.left is not None:
                    queue.append(cur_node.left)
                if cur_node.right is not None:
                    queue.append(cur_node.right)
        return results
```



### 662. Maximum width of binary tree

核心方法：`二叉树层序遍历`

解题思路：在队列中压入每个节点及其索引号

```python
"""
给定一个二叉树，编写一个函数来获取这个树的最大宽度。树的宽度是所有层中的最大宽度。这个二叉树与满二叉树（full binary tree）结构相同，但一些节点为空。

每一层的宽度被定义为两个端点（该层最左和最右的非空节点，两端点间的null节点也计入长度）之间的长度。

链接：https://leetcode-cn.com/problems/maximum-width-of-binary-tree
"""
class Solution:
    def widthOfBinaryTree(self, root: TreeNode) -> int:
        queue = collections.deque()
        if root is not None:
            queue.append((0, root))

        max_width = 0
        while queue:
            cur_size = len(queue)
            leftmost = queue[0][0]
            rightmost = queue[-1][0]
            max_width = max(max_width, rightmost - leftmost + 1)

            for _ in range(cur_size):
                i, cur_node = queue.popleft()
                if cur_node.left is not None:
                    queue.append((i * 2, cur_node.left))
                if cur_node.right is not None:
                    queue.append((i * 2 + 1, cur_node.right))

        return max_width
```



## 图的DFS

树的遍历路径不会成环，而图的遍历路径可能成环，因此图遍历中需要额外保存节点状态，一方面防止遍历陷入无限循环，另一方面状态作为memoization实现短路。

当需要维护与路径有关的状态时，需注意在每一轮递归退出前状态也要同步回退。

```python
## 节点双状态:
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

## 节点多状态:
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

## 主调部分:
# 考虑好seeds是什么，状态存什么
for seed in seeds:
    graph_dfs(seed, g, node_status)
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



### 399. Evaluate division

核心方法：`图的DFS`

解题思路：

```python
"""
给你一个变量对数组 equations 和一个实数值数组 values 作为已知条件，其中 equations[i] = [Ai, Bi] 和 values[i] 共同表示等式 Ai / Bi = values[i] 。每个 Ai 或 Bi 是一个表示单个变量的字符串。

另有一些以数组 queries 表示的问题，其中 queries[j] = [Cj, Dj] 表示第 j 个问题，请你根据已知条件找出 Cj / Dj = ? 的结果作为答案。

返回 所有问题的答案 。如果存在某个无法确定的答案，则用 -1.0 替代这个答案。如果问题中出现了给定的已知条件中没有出现的字符串，也需要用 -1.0 替代这个答案。

注意：输入总是有效的。你可以假设除法运算中不会出现除数为 0 的情况，且不存在任何矛盾的结果。

链接：https://leetcode-cn.com/problems/evaluate-division
"""
from typing import Dict, Tuple, Set
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        # 创建临接表
        adjacency = {}
        for i in range(len(equations)):
            equation = equations[i]
            weight = values[i]
            if equation[0] not in adjacency:
                adjacency[equation[0]] = []
            adjacency[equation[0]].append((equation[1], weight))
            if equation[1] not in adjacency:
                adjacency[equation[1]] = []
            adjacency[equation[1]].append((equation[0], 1 / weight))

        def dfs(src: str, dst: str, 
                adjacency: Dict[str, Tuple[str, float]], 
                visited: Set[str], product: float) -> float:
            if src in visited:
                return None
            if src == dst:
                return product
            visited.add(src)
            for neigh in adjacency[src]:
                prod = dfs(neigh[0], dst, adjacency, visited, product * neigh[1])
                if prod is not None:
                    return prod
            visited.remove(src)
            return None

        # 主调
        results = []
        for query in queries:
            if query[0] not in adjacency or query[1] not in adjacency:
                results.append(-1)
            else:
                product = dfs(query[0], query[1], adjacency, set(), 1)
                if product is not None:
                    results.append(product)
                else:
                    results.append(-1)

        return results
```



### 417. Pacific Atlantic water flow

核心方法：`图的DFS`

解题思路：从四个方向出发，求出能到达的所有点

```python
"""
给定一个 m x n 的非负整数矩阵来表示一片大陆上各个单元格的高度。“太平洋”处于大陆的左边界和上边界，而“大西洋”处于大陆的右边界和下边界。

规定水流只能按照上、下、左、右四个方向流动，且只能从高到低或者在同等高度上流动。

请找出那些水流既可以流动到“太平洋”，又能流动到“大西洋”的陆地单元的坐标。

提示：

输出坐标的顺序不重要
m 和 n 都小于150

链接：https://leetcode-cn.com/problems/pacific-atlantic-water-flow
"""
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        rows = len(heights)
        cols = len(heights[0])

        pacific_visited = [[False for _ in range(cols)] for _ in range(rows)]
        oceanic_visited = [[False for _ in range(cols)] for _ in range(rows)]
        for i in range(rows):
            # 从太平洋出发
            self.dfs(i, 0, rows, cols, heights, pacific_visited)
            # 从大西洋出发
            self.dfs(i, cols - 1, rows, cols, heights, oceanic_visited)
        for i in range(cols):
            # 从太平洋出发
            self.dfs(0, i, rows, cols, heights, pacific_visited)
            # 从大西洋出发
            self.dfs(rows - 1, i, rows, cols, heights, oceanic_visited)
        
        results = []
        for i in range(rows):
            for j in range(cols):
                if pacific_visited[i][j] and oceanic_visited[i][j]:
                    results.append([i, j])
        return results
        
    def dfs(self, row, col, n_rows, n_cols, heights, visited):
        if visited[row][col]:
            return
        visited[row][col] = True
        # 四个方向，往更高的地方走
        if 0 <= col + 1 < n_cols and heights[row][col + 1] >= heights[row][col]:
            self.dfs(row, col + 1, n_rows, n_cols, heights, visited)
        if 0 <= col - 1 < n_cols and heights[row][col - 1] >= heights[row][col]:
            self.dfs(row, col - 1, n_rows, n_cols, heights, visited)
        if 0 <= row + 1 < n_rows and heights[row + 1][col] >= heights[row][col]:
            self.dfs(row + 1, col, n_rows, n_cols, heights, visited)
        if 0 <= row - 1 < n_rows and heights[row - 1][col] >= heights[row][col]:
            self.dfs(row - 1, col, n_rows, n_cols, heights, visited)
```



### 529. Minesweeper

核心方法：`图的DFS`

解题思路：

```python
"""
让我们一起来玩扫雷游戏！

给定一个代表游戏板的二维字符矩阵。 'M' 代表一个未挖出的地雷，'E' 代表一个未挖出的空方块，'B' 代表没有相邻（上，下，左，右，和所有4个对角线）地雷的已挖出的空白方块，数字（'1' 到 '8'）表示有多少地雷与这块已挖出的方块相邻，'X' 则表示一个已挖出的地雷。

现在给出在所有未挖出的方块中（'M'或者'E'）的下一个点击位置（行和列索引），根据以下规则，返回相应位置被点击后对应的面板：

如果一个地雷（'M'）被挖出，游戏就结束了- 把它改为 'X'。
如果一个没有相邻地雷的空方块（'E'）被挖出，修改它为（'B'），并且所有和其相邻的未挖出方块都应该被递归地揭露。
如果一个至少与一个地雷相邻的空方块（'E'）被挖出，修改它为数字（'1'到'8'），表示相邻地雷的数量。
如果在此次点击中，若无更多方块可被揭露，则返回面板。

链接：https://leetcode-cn.com/problems/minesweeper
"""
class Solution:
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        # 每个位置有8个儿子
        self.helper(click[0], click[1], board)
        return board

    def helper(self, cur_row: int, cur_col: int, board: List[List[str]]):
        rows = len(board)
        cols = len(board[0])

        if not (0 <= cur_row < rows and 0 <= cur_col < cols):
            return

        # 挖到雷游戏结束
        if board[cur_row][cur_col] == 'M':
            board[cur_row][cur_col] = 'X'
        # 挖到空白，检查邻居
        elif board[cur_row][cur_col] == 'E':
            status_1 = 1 if cur_row < rows - 1 and board[cur_row + 1][cur_col] == 'M' else 0
            status_2 = 1 if cur_row > 0 and board[cur_row - 1][cur_col] == 'M' else 0
            status_3 = 1 if cur_col < cols - 1 and board[cur_row][cur_col + 1] == 'M' else 0
            status_4 = 1 if cur_col > 0 and board[cur_row][cur_col - 1] == 'M' else 0
            status_5 = 1 if cur_row < rows - 1 and cur_col < cols - 1 and board[cur_row + 1][cur_col + 1] == 'M' else 0
            status_6 = 1 if cur_row < rows - 1 and cur_col > 0 and board[cur_row + 1][cur_col - 1] == 'M' else 0
            status_7 = 1 if cur_row > 0 and cur_col < cols - 1 and board[cur_row - 1][cur_col + 1] == 'M' else 0
            status_8 = 1 if cur_row > 0 and cur_col > 0 and board[cur_row - 1][cur_col - 1] == 'M' else 0
            nums = status_1 + status_2 + status_3 + status_4 + status_5 + status_6 + status_7 + status_8

            # 只有邻居都没有雷的时候才需要递归
            if nums == 0:
                board[cur_row][cur_col] = 'B'
                self.helper(cur_row + 1, cur_col, board)
                self.helper(cur_row - 1, cur_col, board)
                self.helper(cur_row, cur_col + 1, board)
                self.helper(cur_row, cur_col - 1, board)
                self.helper(cur_row + 1, cur_col + 1, board)
                self.helper(cur_row + 1, cur_col - 1, board)
                self.helper(cur_row - 1, cur_col + 1, board)
                self.helper(cur_row - 1, cur_col - 1, board)
            else:
                board[cur_row][cur_col] = str(nums)
```



### 547. Number of provinces

核心方法：`图的DFS`

解题思路：计算连通分量个数

```python
"""
有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连。

省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。

给你一个 n x n 的矩阵 isConnected ，其中 isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 isConnected[i][j] = 0 表示二者不直接相连。

返回矩阵中 省份 的数量。

链接：https://leetcode-cn.com/problems/number-of-provinces
"""
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        def dfs(cur_city: int, isConnected: List[List[int]], visited: List[int]):
            if visited[cur_city] == 1:
                return
            visited[cur_city] = 1
            for i, connected in enumerate(isConnected[cur_city]):
                if connected == 1:
                    dfs(i, isConnected, visited)

        ## 主调
        visited = [0] * len(isConnected)
        connected_count = 0
        for city_i in range(len(isConnected)):
            if visited[city_i] > 0:
                continue
            connected_count += 1
            dfs(city_i, isConnected, visited)
        
        return connected_count
```



### 695. Max area of island

核心方法：`图的DFS`

解题思路：

```python
"""
给定一个包含了一些 0 和 1 的非空二维数组 grid 。

一个 岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在水平或者竖直方向上相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。

找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为 0 。)

链接：https://leetcode-cn.com/problems/max-area-of-island
"""
class Solution:
    def __init__(self):
        self.max_area = 0

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid
        # 主调
        for row in range(self.rows):
            for col in range(self.cols):
                self.dfs(row, col)
        return self.max_area
    
    def dfs(self, cur_row, cur_col) -> int:
        if not (0 <= cur_row < self.rows and 0 <= cur_col < self.cols):
            return 0
        if self.grid[cur_row][cur_col] == 0:
            return 0

        # 4个邻居
        up = cur_row - 1
        down = cur_row + 1
        left = cur_col - 1
        right = cur_col + 1

        # 记录当前非0根节点已访问
        self.grid[cur_row][cur_col] = 0
        cur_area = (self.dfs(up, cur_col) + self.dfs(down, cur_col) + 
                    self.dfs(cur_row, left) + self.dfs(cur_row, right) + 1)
        self.max_area = max(self.max_area, cur_area)

        return cur_area
```



### 721. Accounts merge

核心方法：`图的DFS`，`后序遍历`

解题思路：求连通分量的个数，当两个账户的邮箱列表有重叠时可以合并

```python
"""
给定一个列表 accounts，每个元素 accounts[i] 是一个字符串列表，其中第一个元素 accounts[i][0] 是 名称 (name)，其余元素是 emails 表示该账户的邮箱地址。

现在，我们想合并这些账户。如果两个账户都有一些共同的邮箱地址，则两个账户必定属于同一个人。请注意，即使两个账户具有相同的名称，它们也可能属于不同的人，因为人们可能具有相同的名称。一个人最初可以拥有任意数量的账户，但其所有账户都具有相同的名称。

合并账户后，按以下格式返回账户：每个账户的第一个元素是名称，其余元素是按字符 ASCII 顺序排列的邮箱地址。账户本身可以以任意顺序返回。

链接：https://leetcode-cn.com/problems/accounts-merge
"""
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        # 建无向图的临接表，节点是账户index，边是两账户内的邮箱列表是否有重叠
        email_to_accounts = {}
        for i, emails in enumerate(accounts):
            for email in set(emails[1:]):
                if email not in email_to_accounts:
                    email_to_accounts[email] = []
                email_to_accounts[email].append(i)
        adjacency = {i: [] for i in range(len(accounts))}
        for account_index in email_to_accounts.values():
            for i in range(len(account_index) - 1):
                for j in range(i + 1, len(account_index)):
                    if account_index[j] not in adjacency[account_index[i]]:
                        adjacency[account_index[i]].append(account_index[j])
                    if account_index[i] not in adjacency[account_index[j]]:
                        adjacency[account_index[j]].append(account_index[i])

        self.accounts = accounts
        self.adjacency = adjacency
        results = []
        # 主调
        visited = [False] * len(accounts)
        for i in range(len(accounts)):
            if visited[i]:
                continue
            result = self.dfs(i, visited)
            results.append(accounts[i][:1] + result)
        return results
    
    def dfs(self, account_i: int, visited: List[bool]) -> List[str]:
        """
        return: 当前节点所在团的合并邮箱列表
        """
        if visited[account_i]:
            return []
        visited[account_i] = True

        merged_emails = []
        # 合并邻居节点的邮箱列表
        for neighbor_i in self.adjacency[account_i]:
            merged_emails.extend(self.dfs(neighbor_i, visited))
        # 合并当前用户的邮箱列表
        merged_emails.extend(self.accounts[account_i][1:])

        merged_emails = sorted(set(merged_emails))
        return merged_emails
```



### 785. Is graph bipartite

核心方法：`图的DFS`

解题思路：着色法

```python
"""
存在一个 无向图 ，图中有 n 个节点。其中每个节点都有一个介于 0 到 n - 1 之间的唯一编号。给你一个二维数组 graph ，其中 graph[u] 是一个节点数组，由节点 u 的邻接节点组成。形式上，对于 graph[u] 中的每个 v ，都存在一条位于节点 u 和节点 v 之间的无向边。该无向图同时具有以下属性：
不存在自环（graph[u] 不包含 u）。
不存在平行边（graph[u] 不包含重复值）。
如果 v 在 graph[u] 内，那么 u 也应该在 graph[v] 内（该图是无向图）
这个图可能不是连通图，也就是说两个节点 u 和 v 之间可能不存在一条连通彼此的路径。
二分图 定义：如果能将一个图的节点集合分割成两个独立的子集 A 和 B ，并使图中的每一条边的两个节点一个来自 A 集合，一个来自 B 集合，就将这个图称为 二分图 。

如果图是二分图，返回 true ；否则，返回 false 。

链接：https://leetcode-cn.com/problems/is-graph-bipartite
"""
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        node_count = len(graph)
        node_color = [None] * node_count

        for i in range(node_count):
            if node_color[i] is None:
                # 以当前未着色点为起点进行一轮着色
                if not self.dfs(i, graph, node_color, "one"):
                    return False
        return True

    def dfs(self, node_i: int, graph: List[List[int]], node_color: List[str], cur_color: str) -> bool:
        # 给当前节点着色
        node_color[node_i] = cur_color
        # 给邻居节点着色（与当前节点颜色相反）
        for neighbor in graph[node_i]:
            next_color = "one" if cur_color == "two" else "two"
            if node_color[neighbor] is None:
                if not self.dfs(neighbor, graph, node_color, next_color):
                    return False
            elif node_color[neighbor] != next_color:
                return False
        return True
```



### 322. Coin change

核心方法：`图的DFS`，`背包问题`，`回溯`

解题思路：

1. 贪心的思路——每次尽量取最大面值，未必有解，因此本题没有最优子结构，只能老老实实遍历所有可能性（暴力回溯）
2. 回溯过程可以用一个有向图（多叉树）来描述，因此是一个图上的DFS，并使用memoization实现短路

```python
"""
给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。

你可以认为每种硬币的数量是无限的。

链接：https://leetcode-cn.com/problems/coin-change
"""
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        count = self.helper(coins, amount, {})
        return count

    def helper(self, coins: List[int], amount: int, mem: Dict[int, int]) -> int:
        """
        mem: 组成amount所需的最少硬币数
        return: 组成当前amount所需最少硬币数
        """
        if amount == 0:
            mem[amount] = 0
            return mem[amount]

        # 已查询过的amount
        if amount in mem:
            return mem[amount]

        # 未查询过的amount
        mem[amount] = float('inf')
        for coin in coins:
            if coin <= amount:
                rest_count = self.helper(coins, amount - coin, mem)
                if -1 < rest_count < mem[amount]:
                    mem[amount] = rest_count + 1

        return mem[amount] if mem[amount] < float('inf') else -1
```



### 96. Unique binary search trees

核心方法：

解题思路：

```python
"""
给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？返回满足题意的二叉搜索树的种数。

链接：https://leetcode-cn.com/problems/unique-binary-search-trees/
"""
class Solution:
    def __init__(self):
        self.results = {0: 1, 1: 1}

    def numTrees(self, n: int) -> int:
        # n表示参与建树的节点数，返回值表示n个节点组成的二叉搜索树个数
        if n in self.results:
            return self.results[n]

        total_count = 0
        # root节点必须占用一个节点，因此剩下共分配的总结点数是n - 1
        for left_count in range(0, n):
            right_count = n - 1 - left_count
            self.results[left_count] = self.numTrees(left_count)
            self.results[right_count] = self.numTrees(right_count)
            total_count += self.results[left_count] * self.results[right_count]
        self.results[n] = total_count

        return total_count
```



### 95. Unique binary search trees II

核心方法：`回溯`，`分治`，`DFS`

解题思路：

```python
"""
给你一个整数 n ，请你生成并返回所有由 n 个节点组成且节点值从 1 到 n 互不相同的不同 二叉搜索树 。可以按 任意顺序 返回答案。

链接：https://leetcode-cn.com/problems/unique-binary-search-trees-ii/
"""
class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        vals = [i+1 for i in range(n)]
        results = self.helper(vals)
        return results
    
    def helper(self, vals: List[int]) -> List[TreeNode]:
        if not vals:
            return None

        # vals中的任何一点都可以作为子树顶点
        roots = []
        for i in range(len(vals)):
            left_trees = self.helper(vals[:i])
            right_trees = self.helper(vals[(i+1):])
            
            if not left_trees and not right_trees:
                root = TreeNode(vals[i])
                roots.append(root)
            elif not left_trees:
                for right_tree in right_trees:
                    root = TreeNode(vals[i])
                    root.right = right_tree
                    roots.append(root)
            elif not right_trees:
                for left_tree in left_trees:
                    root = TreeNode(vals[i])
                    root.left = left_tree
                    roots.append(root)
            else:
                for left_tree in left_trees:
                    for right_tree in right_trees:
                        root = TreeNode(vals[i])
                        root.left = left_tree
                        root.right = right_tree
                        roots.append(root)
        
        return roots
```





#### 回溯和遍历的区别

回溯：

回溯是遍历，遍历的过程需要记录路径，所有分支组成一颗多叉树

回溯是在遍历的过程中记录路径，背后是一颗树，可以使用DFS或BFS来实现遍历



#### 背包问题



## 图的BFS

### 279. Perfect squares

核心方法：`图的BFS`

解题思路：在行表示dp，列表示square的二维矩阵上做BFS

```python
"""
给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。

给你一个整数 n ，返回和为 n 的完全平方数的 最少数量 。

完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。

链接：https://leetcode-cn.com/problems/perfect-squares
"""
class Solution:
    def numSquares(self, n: int) -> int:
        # 不大于n的全部平方数是很容易枚举出来的
        squares = [i ** 2 for i in range(1, int(math.sqrt(n)) + 1)]
        # dp[i]表示i对应的“最小”完全平方数
        dp = [0] * (n + 1)
        # 依次求出从1到n每个数对应的最少平方数
        for num in range(1, n + 1):
            rest_min_count = float('inf')
            for square in squares:
                if num < square:
                    break
                # num可以分为 square 和 (num - square) 两部分
                rest_min_count = min(rest_min_count, dp[num - square])
            # 当前square还要算1个
            dp[num] = rest_min_count + 1

        return dp[-1]
```



### 286. Walls and gates

核心方法：`图的BFS`

解题思路：以门为种子点开始探索，`rooms`同时作为棋盘和状态存储

```python
"""
你被给定一个 m × n 的二维网格 rooms ，网格中有以下三种可能的初始化值：

-1 表示墙或是障碍物
0 表示一扇门
INF 表示一个空的房间。然后，我们用 231 - 1 = 2147483647 代表 INF。你可以认为通往门的距离总是小于 2147483647 的。
你要给每个空房间位上填上该房间到 最近门的距离 ，如果无法到达门，则填 INF 即可。

链接：https://leetcode-cn.com/problems/walls-and-gates
"""
class Solution:
    def wallsAndGates(self, rooms: List[List[int]]) -> None:
        """
        Do not return anything, modify rooms in-place instead.
        """
        q = queue.Queue()
        # 以门为起点
        for i in range(len(rooms)):
            for j in range(len(rooms[0])):
                if rooms[i][j] != 0:
                    continue
                q.put((i, j))

        while not q.empty():
            pos_i, pos_j = q.get()
            if 0 <= pos_i - 1 < len(rooms) and rooms[pos_i - 1][pos_j] == 2147483647:
                rooms[pos_i - 1][pos_j] = rooms[pos_i][pos_j] + 1
                q.put((pos_i - 1, pos_j))
            if 0 <= pos_i + 1 < len(rooms) and rooms[pos_i + 1][pos_j] == 2147483647:
                rooms[pos_i + 1][pos_j] = rooms[pos_i][pos_j] + 1
                q.put((pos_i + 1, pos_j))
            if 0 <= pos_j - 1 < len(rooms[0]) and rooms[pos_i][pos_j - 1] == 2147483647:
                rooms[pos_i][pos_j - 1] = rooms[pos_i][pos_j] + 1
                q.put((pos_i, pos_j - 1))
            if 0 <= pos_j + 1 < len(rooms[0]) and rooms[pos_i][pos_j + 1] == 2147483647:
                rooms[pos_i][pos_j + 1] = rooms[pos_i][pos_j] + 1
                q.put((pos_i, pos_j + 1))
```



### 365. Water and jug problem

核心方法：`图的BFS`，`剪枝`

解题思路：

1. 解析解 or 遍历(枚举)
2. 两只水壶的状态是有限的（可枚举的），状态间可由操作转移，操作（边）也是有限的，因此所有状态可以用图来表示

```python
"""
有两个容量分别为 x升 和 y升 的水壶以及无限多的水。请判断能否通过使用这两个水壶，从而可以得到恰好 z升 的水？

如果可以，最后请用以上水壶中的一或两个来盛放取得的 z升 水。

你允许：

装满任意一个水壶
清空任意一个水壶
从一个水壶向另外一个水壶倒水，直到装满或者倒空

链接：https://leetcode-cn.com/problems/water-and-jug-problem
"""

```



### 542. 01 matrix

核心方法：`图的BFS`

解题思路：从与0相邻的1开始

```python
"""
给定一个由 0 和 1 组成的矩阵，找出每个元素到最近的 0 的距离。

两个相邻元素间的距离为1。

链接：https://leetcode-cn.com/problems/01-matrix/
"""
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        rows = len(mat)
        cols = len(mat[0])
        offset = [0, 1, 0, -1, 0]
        # 队列中保存距离已确定的1	
        queue = collections.deque()
        # 所有与0相邻的1作为种子点，其值为1，其余1初始化为-1，表示距离待定
        for row in range(rows):
            for col in range(cols):
                if mat[row][col] == 1:
                    boundary = False
                    for i in range(4):
                        neigh_row, neigh_col = row + offset[i], col + offset[i + 1]
                        if 0 <= neigh_row < rows and 0 <= neigh_col < cols and mat[neigh_row][neigh_col] == 0:
                            queue.append((row, col))
                            boundary = True
                            break
                    if not boundary:
                        mat[row][col] = -1
        
        while queue:
            row, col = queue.popleft()
            for i in range(4):
                neigh_row, neigh_col = row + offset[i], col + offset[i + 1]
                # 领域内有距离待决的1
                if 0 <= neigh_row < rows and 0 <= neigh_col < cols and mat[neigh_row][neigh_col] == -1:
                    mat[neigh_row][neigh_col] = mat[row][col] + 1
                    queue.append((neigh_row, neigh_col))
        
        return mat
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



