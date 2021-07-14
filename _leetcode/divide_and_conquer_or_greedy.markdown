---
layout: post
title: 分治与贪心篇
---



## 一、分治

### 105. Construct binary tree from preorder and inorder traversal

核心方法：`二叉树`，`二分`

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

核心方法：`二叉树`，`二分`

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





## 二、greedy



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

