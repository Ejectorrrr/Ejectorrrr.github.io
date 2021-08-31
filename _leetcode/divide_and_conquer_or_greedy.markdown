---
layout: post
title: 分治与贪心篇
---



## 一、分治

基本思想：原问题可以被递归地分解为多个子问题，直到子问题规模变得足够简单。

具体分为两步：

1. 下钻至叶节点时，如何解决足够简单的子问题
2. 回退过程中，如何融合子问题解得到父问题解

整体流程类似后序遍历



### 105. Construct binary tree from preorder and inorder traversal

核心方法：`分治`，`二分`

解题思路：

```python
"""
根据一棵树的前序遍历与中序遍历构造二叉树。

注意:
你可以假设树中没有重复的元素。

链接：https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal
"""
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        """
        注意：任何一轮迭代中，preorder和inorder的长度都一样
        """
        if not preorder or not inorder:
            return None

        # 根据“前序遍历”确定当前root节点
        root = TreeNode(preorder[0])

        # 根据root节点将“中序遍历”划分为左右两部分
        i = 0
        while i < len(inorder) and inorder[i] != preorder[0]:
            i += 1
        left_inorder = inorder[0:i] if i > 0 else []
        right_inorder = inorder[i+1:] if i < len(inorder) - 1 else []

        # 根据“中序遍历”左半边元素数，决定剩余“前序遍历”的划分方式
        # left_inorder和left_preorder的长度始终保持一致，对right_inorder和right_preorder也同理
        left_preorder = preorder[1:(len(left_inorder)+1)] if left_inorder else []
        right_preorder = preorder[(len(left_inorder)+1):] if right_inorder else []
        root.left = self.buildTree(left_preorder, left_inorder)
        root.right = self.buildTree(right_preorder, right_inorder)

        return root
```



### 106. Construct binary tree from inorder and postorder traversal

核心方法：`分治`，`二分`

解题思路：

```python
"""
根据一棵树的中序遍历与后序遍历构造二叉树。

注意:
你可以假设树中没有重复的元素。

链接：https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        """
        前序遍历和后序遍历用来查找顶点
        中序遍历用来查找左右子树
        区别：前序遍历从头部确定顶点，后序遍历从尾部确定顶点
        """
        if not inorder or not postorder:
            return None
        
        root = TreeNode(postorder[-1])

        i = 0
        while i < len(inorder):
            if inorder[i] == root.val:
                break
            i += 1
        left_inorder = inorder[:i] if i > 0 else []
        right_inorder = inorder[(i+1):] if i < len(inorder) - 1 else []

        # 任何一轮迭代中，inorder和postorder的长度都相同（因为要表示同一颗子树）
        right_postorder = postorder[-(len(right_inorder)+1):-1] if right_inorder else []
        left_postorder = postorder[:len(left_inorder)] if left_inorder else []

        root.left = self.buildTree(left_inorder, left_postorder)
        root.right = self.buildTree(right_inorder, right_postorder)

        return root
```





## 二、贪心

基本思想：在每一步中都选取当前状态下的最优解，从而希望最终结果也是最优的（局部最优解能决定全局最优解）

适合存在最优子结构的问题

分治和贪心都有分解原问题的过程，使得最终要解决的子问题比原问题更简单直观，但分治强调在回退过程中如何融合各子问题的解，而贪心强调如何选取子问题下的最优解



### 55. Jump game

核心方法：`贪心`

解题思路：

```python
"""
给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。

链接：https://leetcode-cn.com/problems/jump-game/
"""
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        farmost_pos = 0
        for i in range(len(nums)):
            if i > farmost_pos:
                return False
            
            farmost_pos = max(farmost_pos, i + nums[i])
        
        return True
```



### 45. Jump game II

核心方法：`贪心`

解题思路：

```python
"""
给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。

假设你总是可以到达数组的最后一个位置。

链接：https://leetcode-cn.com/problems/jump-game-ii
"""
```



### 120. Triangle

核心方法：`贪心`

解题思路：

```python
"""
给定一个三角形 triangle ，找出自顶向下的最小路径和。

每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。

链接：https://leetcode-cn.com/problems/triangle
"""
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        # 用一个下三角阵保存状态
        # 从上到下遍历
        rows = len(triangle)
        for row in range(1, rows):
            cols = len(triangle[row])
            for col in range(cols):
                triangle[row][col] += min(
                    triangle[row - 1][col] if col < cols - 1 else float('inf'),
                    triangle[row - 1][col - 1] if col > 0 else float('inf'))
        
        return min(triangle[-1])
```



背包问题



### 743. Network delay time

核心方法：

解决思路：

- 多源最短路，Floyd，邻接矩阵
- 单源最短路，Dijkstra，邻接矩阵/邻接表

```python
"""
有 n 个网络节点，标记为 1 到 n。

给你一个列表 times，表示信号经过 有向 边的传递时间。 times[i] = (ui, vi, wi)，其中 ui 是源节点，vi 是目标节点， wi 是一个信号从源节点传递到目标节点的时间。

现在，从某个节点 K 发出一个信号。需要多久才能使所有节点都收到信号？如果不能使所有节点收到信号，返回 -1 。

链接：https://leetcode-cn.com/problems/network-delay-time
"""
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # Dij算法

        # 1. 将节点分为“已确定最短路径”和“未确定最短路径”的两部分
        # 2. 选取当前“未确定最短路径”的所有节点中距离起点最近的，并将其设置为“已确定最短路径”的节点
        # 3. 更新该“已确定最短路径”节点的所有邻居节点到达起点的最短距离

        # 注：所有节点（除起点）到起点的初始距离都是inf

        # 状态变量
        min_dist = [float("inf")] * n
        determined = [False] * n

        min_dist[k - 1] = 0

        # 一共有n个节点，因此需要对状态变量更新n次
        for _ in range(n):
            chosen_node = -1
            # 选取当前“未确定”节点中距离起点最近的，设置为“已确定”
            for cur_node, status in enumerate(determined):
                if not status and (chosen_node == -1 or min_dist[cur_node] < min_dist[chosen_node]):
                    chosen_node = cur_node
            determined[chosen_node] = True

            # 更新该“已确定”节点的邻居节点到达起点的最短距离
            for src, dst, wgt in times:
                if src - 1 == chosen_node:
                    min_dist[dst - 1] = min(min_dist[dst - 1], min_dist[chosen_node] + wgt)
        
        max_min_dist = max(min_dist)
        return -1 if max_min_dist == float("inf") else max_min_dist
```

